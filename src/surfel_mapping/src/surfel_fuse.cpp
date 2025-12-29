#include "surfel_mapping/surfel_fuse.hpp"

using namespace surface_inspection_planning;

SurfelFusion::SurfelFusion() : params_(), frame_count_(0) {}
SurfelFusion::SurfelFusion(const Params& p, const SurfelMap::Params& map_p) : params_(p), frame_count_(0), map_(map_p) {}

/* PUBLIC */
void SurfelFusion::process_scan(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const Eigen::Isometry3f& pose, uint64_t timestamp) {
    if (cloud->size() != normals->size()) {
        std::cerr << "[SurfelFusion] Point/Normal size mismatch!" << std::endl;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    last_stats_ = FusionStats{};
    last_stats_.points_processed = cloud->size();

    std::vector<bool> surfel_updated(map_.size(), false);

    const Eigen::Matrix3f R = pose.rotation();
    const Eigen::Vector3f t = pose.translation();

    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = cloud->points[i];
        const auto& n = normals->points[i];

        if (!std::isfinite(p.x) || !std::isfinite(n.normal_x)) {
            last_stats_.points_rejected++;
            continue;
        }

        Eigen::Vector3f normal_local(n.normal_x, n.normal_y, n.normal_z);
        if (normal_local.norm() < params_.min_normal_quality) {
            last_stats_.points_rejected++;
            continue;
        }

        // Transform point
        Eigen::Vector3f point_local(p.x, p.y, p.z);
        Eigen::Vector3f point_world = R * point_local + t;
        Eigen::Vector3f normal_world = (R * normal_local).normalized();

        int best_idx = map_.find_best_association(point_world, normal_world);

        if (best_idx >= 0) {
            fuse_point_to_surfel(static_cast<size_t>(best_idx), point_world, normal_world, timestamp);
            surfel_updated[best_idx] = true;
            last_stats_.points_associated++;
        }
        else {
            accumulate_point(point_world, normal_world, timestamp);
            last_stats_.points_accumulated++;
        }
    }

    for (bool updated : surfel_updated) {
        if (updated) last_stats_.surfels_updated++;
    }

    frame_count_++;
    if (frame_count_ % params_.accumulator_process_interval == 0) {
        process_accumulator(timestamp);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

void SurfelFusion::flush_accumulator(uint64_t timestamp) {
    process_accumulator(timestamp);
    point_accumulator_.clear();
}

void SurfelFusion::reset() {
    map_.clear();
    point_accumulator_.clear();
    frame_count_ = 0;
    last_stats_ = FusionStats{};
}

/* PRIVATE */

void SurfelFusion::fuse_point_to_surfel(size_t surfel_idx, const Eigen::Vector3f& point, const Eigen::Vector3f& normal, uint64_t timestamp) {
    Surfel& surfel = map_.get_surfels_mutable()[surfel_idx];

    auto [tangent_coords, normal_dist] = surfel.project_point(point);

    float weight = params_.base_point_weight;

    // Update center (weighe EMA)
    float alpha_c = params_.center_update_rate * weight / (surfel.total_weight + weight);
    Eigen::Vector3f tangent_offset = tangent_coords.x() * surfel.tangent_u + tangent_coords.y() * surfel.tangent_v;
    Eigen::Vector3f point_on_plane = surfel.center + tangent_offset;
    surfel.center = surfel.center + alpha_c * (point_on_plane - surfel.center);

    // Update normal (weighted spherical averaging) - done every 10 observation (to reduce jitter)
    surfel.sum_normals += weight * normal;
    if (surfel.observation_count % 10 == 0) {
        Eigen::Vector3f avg_normal = surfel.sum_normals.normalized();
        float alpha_n = params_.normal_update_rate;
        surfel.normal = (surfel.normal + alpha_n * (avg_normal - surfel.normal)).normalized();
        surfel.compute_tangential_basis(); // recompute tangent frame
    }

    // Update statistics for covariance
    surfel.sum_tangent += weight * tangent_coords;
    surfel.sum_outer += weight * (tangent_coords * tangent_coords.transpose());
    surfel.total_weight += weight;
    surfel.point_count++;

    // Update Covariance - done every 5 observations
    if (surfel.point_count % 5 == 0) {
        surfel.recompute_covariance();
        surfel.update_eigen();

        // clamp radius
        for (int i = 0; i < 2; ++i) {
            float min_var = map_.params().min_surfel_radius * map_.params().min_surfel_radius;
            float max_var = map_.params().max_surfel_radius * map_.params().max_surfel_radius;
            surfel.eigenvalues(i) = std::clamp(surfel.eigenvalues(i), min_var, max_var);
        }

        surfel.covariance = surfel.eigenvectors * surfel.eigenvalues.asDiagonal() * surfel.eigenvectors.transpose();
    }

    // update confidence
    surfel.confidence = std::min(params_.max_confidence, surfel.confidence + params_.confidence_boost);

    // timestamp and counters
    surfel.last_update_stamp = timestamp;
    surfel.observation_count++;

    // OBS!!
    // Should update spatial index if center moved significantly??
}

void SurfelFusion::accumulate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, uint64_t timestamp) {
    if (point_accumulator_.size() >= params_.max_accumulator_size) {
        point_accumulator_.pop_front();
    }

    point_accumulator_.push_back({point, normal, timestamp});
}

void SurfelFusion::process_accumulator(uint64_t timestamp) {
    if (point_accumulator_.size() < params_.min_points_for_new_surfel) return;

    float cluster_size = params_.new_surfel_coherence_thresh * 2.0f;
    float inv_size = 1.0f / cluster_size;

    struct ClusterCell {
        std::vector<size_t> point_indices;
        Eigen::Vector3f center_sum = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal_sum = Eigen::Vector3f::Zero();
    };

    std::unordered_map<int64_t, ClusterCell> clusters;

    // hash function for clustering
    auto hash_point = [inv_size](const Eigen::Vector3f& p) -> int64_t {
        int32_t x = static_cast<int32_t>(std::floor(p.x() * inv_size));
        int32_t y = static_cast<int32_t>(std::floor(p.y() * inv_size));
        int32_t z = static_cast<int32_t>(std::floor(p.z() * inv_size));
        return (static_cast<int64_t>(x) << 42) ^ 
                (static_cast<int64_t>(y) << 21) ^ 
                static_cast<int64_t>(z);
    };

    // assign to clusters
    for (size_t i = 0; i < point_accumulator_.size(); ++i) {
        const auto& ap = point_accumulator_[i];
        int64_t hash = hash_point(ap.position);
        auto& cell = clusters[hash];
        cell.point_indices.push_back(i);
        cell.center_sum += ap.position;
        cell.normal_sum += ap.normal;
    }

    // create surfels from valid clusters
    std::vector<size_t> points_to_remove;

    for (auto& [hash, cell] : clusters) {
        if (cell.point_indices.size() < params_.min_points_for_new_surfel) continue;

        // cluster centroid and avg normal
        Eigen::Vector3f center = cell.center_sum / static_cast<float>(cell.point_indices.size());
        Eigen::Vector3f normal = cell.normal_sum.normalized();

        // check if coherent
        float max_dist_sq = 0.0f;
        for (size_t idx : cell.point_indices) {
            float d_sq = (point_accumulator_[idx].position - center).squaredNorm();
            max_dist_sq = std::max(max_dist_sq, d_sq);
        }
        float coherence_thresh_sq = params_.new_surfel_coherence_thresh * params_.new_surfel_coherence_thresh;
        if (max_dist_sq > coherence_thresh_sq * 4.0f) continue;

        // normal consistency
        float normal_variance = 0.0f;
        for (size_t idx : cell.point_indices) {
            float dot = point_accumulator_[idx].normal.dot(normal);
            normal_variance += (1.0f - dot * dot);
        }
        normal_variance /= static_cast<float>(cell.point_indices.size());
        if (normal_variance > 0.3f) continue;

        // create new surfel
        map_.create_surfel(center, normal, params_.new_surfel_initial_radius, timestamp);
        last_stats_.surfels_created++;

        // mark points in cluster for removal
        for (size_t idx : cell.point_indices) {
            points_to_remove.push_back(idx);
        }
    }

    // remove processed points (sort descending to preserved indices)
    std::sort(points_to_remove.rbegin(), points_to_remove.rend());
    points_to_remove.erase(std::unique(points_to_remove.begin(), points_to_remove.end()), points_to_remove.end());
    for (size_t idx : points_to_remove) {
        if (idx < point_accumulator_.size()) {
            point_accumulator_.erase(point_accumulator_.begin() + static_cast<long>(idx));
        }
    }
}