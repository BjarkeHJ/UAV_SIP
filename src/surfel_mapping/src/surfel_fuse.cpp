#include "surfel_mapping/surfel_fuse.hpp"

using namespace surface_inspection_planning;

SurfelFusion::SurfelFusion() : params_(), map_(), graph_() {
    frame_count_ = 0;
    last_graph_update_frame_ = 0;
}
SurfelFusion::SurfelFusion(const Params& p, const SurfelMap::Params& map_p) : params_(p), map_(map_p), graph_() {
    frame_count_ = 0;
    last_graph_update_frame_ = 0;
}
SurfelFusion::SurfelFusion(const Params& p, const SurfelMap::Params& map_p, const ConnectivityParams& graph_p) : params_(p), map_(map_p), graph_(graph_p) {
    frame_count_ = 0;
    last_graph_update_frame_ = 0;
}

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

        // If good association found -> fuse it
        // Else accumulate and process later
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

    // if (params_.enable_graph) {
    //     maybe_update_graph();
    // }
    // std::cout << "GRAPH SIZE: " << graph_.num_nodes() << std::endl; 

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();


    last_stats_.graph_nodes = graph_.num_nodes();
    last_stats_.graph_edges = graph_.num_edges();
}

void SurfelFusion::flush_accumulator(uint64_t timestamp) {
    process_accumulator(timestamp);
    point_accumulator_.clear();
}

void SurfelFusion::reset() {
    map_.clear();
    graph_.clear();
    point_accumulator_.clear();
    pending_changes_.clear();
    frame_count_ = 0;
    last_graph_update_frame_ = 0;
    last_stats_ = FusionStats{};
}

/* PRIVATE */

void SurfelFusion::fuse_point_to_surfel(size_t surfel_idx, const Eigen::Vector3f& point, const Eigen::Vector3f& normal, uint64_t timestamp) {
    Surfel& surfel = map_.get_surfels_mutable()[surfel_idx];

    auto [tangent_coords, normal_dist] = surfel.project_point(point);

    // Accumulate surface fit (squared)
    surfel.sum_sq_normal_dist += normal_dist * normal_dist;

    const float weight = params_.base_point_weight;

    // Update center (weighe EMA)
    float alpha_c = params_.center_update_rate * weight / (surfel.total_weight + weight);
    Eigen::Vector3f tangent_offset = tangent_coords.x() * surfel.tangent_u + tangent_coords.y() * surfel.tangent_v;
    Eigen::Vector3f point_on_plane = surfel.center + tangent_offset;
    surfel.center = surfel.center + alpha_c * (point_on_plane - surfel.center);

    // Update normal (weighted spherical averaging) - done every 10 observation (to reduce jitter)
    surfel.sum_normals += weight * normal;
    if (surfel.observation_count % 5 == 0) {
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

    // Update Covariance - done every 5 point count
    if (surfel.point_count % 5 == 0) {
        surfel.recompute_covariance();
        surfel.update_eigen();
        surfel.covariance = surfel.eigenvectors * surfel.eigenvalues.asDiagonal() * surfel.eigenvectors.transpose();
    }

    // Temporal tracking
    if (surfel.last_update_stamp != timestamp) {
        surfel.observation_count++;
    }
    surfel.last_update_stamp = timestamp;

    // Update confidence
    surfel.update_confidence(params_.confidence);
    surfel.update_maturity(map_.params().min_surfel_radius);
    map_.update_spatial_index(surfel_idx);

    // track update
    record_surfel_updated(surfel_idx);
}

void SurfelFusion::accumulate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, uint64_t timestamp) {
    if (point_accumulator_.size() >= params_.max_accumulator_size) {
        point_accumulator_.pop_front();
    }

    point_accumulator_.push_back({point, normal, timestamp});
}

void SurfelFusion::process_accumulator(uint64_t timestamp) {
    if (point_accumulator_.size() < params_.min_points_for_new_surfel) return;

    float cluster_size = params_.new_surfel_coherence_thresh * 2.0f; // cluster diameter
    float coherence_thresh_sq = cluster_size * cluster_size;
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

    // assign to clusters and collect data statistics
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

        // check if spatially coherent
        float max_dist_sq = 0.0f;
        for (size_t idx : cell.point_indices) {
            // distance to centroid -> store max distance in cluster
            float d_sq = (point_accumulator_[idx].position - center).squaredNorm();
            max_dist_sq = std::max(max_dist_sq, d_sq);
        }

        if (max_dist_sq > coherence_thresh_sq) continue;

        // normal consistency
        float normal_variance = 0.0f;
        for (size_t idx : cell.point_indices) {
            float dot = point_accumulator_[idx].normal.dot(normal);
            normal_variance += (1.0f - dot * dot);
        }
        normal_variance /= static_cast<float>(cell.point_indices.size());
        if (normal_variance > 0.3f) continue;

        // check for existing surfel to merge with before creation
        size_t merge_target = map_.find_merge_target(center, normal);
        if (merge_target != INVALID_SURFEL_IDX) {
            for (size_t idx : cell.point_indices) {
                fuse_point_to_surfel(merge_target, point_accumulator_[idx].position, point_accumulator_[idx].normal, timestamp);
            }
        }
        else {
            size_t new_idx = map_.create_surfel(center, normal, params_.new_surfel_initial_radius, timestamp);
            if (new_idx != INVALID_SURFEL_IDX) {
                last_stats_.surfels_created++;
                record_surfel_created(new_idx);
            }
        }

        // mark points in cluster for removal (removed from accumulator)
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

    // periodic merge
    static size_t merge_counter = 0;
    if (++merge_counter % 5 == 0) {
        size_t merged = map_.merge_similar_surfels();
        if (merged > 0) {
            last_stats_.surfels_merged = merged;
        }
    }
}

void SurfelFusion::maybe_update_graph() {
    if (frame_count_ - last_graph_update_frame_ < params_.graph_update_interval) {
        return;
    }
    update_graph_now();
}

GraphUpdateStats SurfelFusion::update_graph_now() {
    auto start_time = std::chrono::high_resolution_clock::now();
   
    GraphUpdateStats stats;
    
    if (!pending_changes_.empty()) {
        std::vector<size_t> new_vec(pending_changes_.new_surfels.begin(), pending_changes_.new_surfels.end());
        std::vector<size_t> updated_vec(pending_changes_.updated_surfels.begin(), pending_changes_.updated_surfels.end());
        std::vector<size_t> removed_vec(pending_changes_.removed_surfels.begin(), pending_changes_.removed_surfels.end());
    
        stats = graph_.update_incremental(map_, new_vec, updated_vec, removed_vec);
        pending_changes_.clear();
    }
    else {
        // no tracked changes
        stats = graph_.update_from_map(map_);
    }

    last_graph_update_frame_ = frame_count_;

    auto end_time = std::chrono::high_resolution_clock::now();
    stats.update_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return stats;
}

void SurfelFusion::rebuild_graph() {
    graph_.build_from_map(map_);pending_changes_.clear();
    last_graph_update_frame_ = frame_count_;
}

void SurfelFusion::record_surfel_created(size_t surfel_idx) {
    pending_changes_.new_surfels.insert(surfel_idx);
    pending_changes_.updated_surfels.erase(surfel_idx);
}

void SurfelFusion::record_surfel_updated(size_t surfel_idx) {
    // if not in new
    if (pending_changes_.new_surfels.find(surfel_idx) == pending_changes_.new_surfels.end()) {
        pending_changes_.updated_surfels.insert(surfel_idx);
    }
}

void SurfelFusion::record_surfel_removed(size_t surfel_idx) {
    pending_changes_.removed_surfels.insert(surfel_idx);
    pending_changes_.new_surfels.erase(surfel_idx);
    pending_changes_.updated_surfels.erase(surfel_idx);
}