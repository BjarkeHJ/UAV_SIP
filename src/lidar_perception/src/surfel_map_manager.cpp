#include "lidar_perception/surfel_map_manager.hpp"

namespace surface_inspection_planner {

SurfelMap::SurfelMap(const SurfelMapConfig& config) : config_(config) {
    surfel_centers_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    kdtree_ = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
}

void SurfelMap::integrate(const std::vector<Surfel2D>& local_surfels, const Eigen::Isometry3f& sensor_pose, int frame_id) {
    if (local_surfels.empty()) return;

    // std::vector<Surfel2D> global_local_surfels;
    // global_local_surfels.reserve(local_surfels.size());

    // if data in local_surfels is not transformed to global frame do it here!

    // find associations between local and global
    std::vector<std::pair<int,int>> associations = find_associations(local_surfels, sensor_pose);
    std::vector<bool> associated(local_surfels.size(), false);

    // update matches in global map
    for (const auto& [local_idx, global_idx] : associations) {
        if (local_idx > 0 && global_idx > 0) {
            fuse_surfel(surfel_map_[global_idx], local_surfels[local_idx], frame_id);
            associated[local_idx] = true;
        }
    }

    // add new surfels
    for (size_t i = 0; i < local_surfels.size(); ++i) {
        if (!associated[i]) {
            add_surfel(local_surfels[i], frame_id);
        }
    }

    // maintenance
    if (config_.enable_surfel_merging) {
        merge_surfels();
    }

    // remove_stale();

    update_spatial_index();
}

std::vector<std::pair<int,int>> SurfelMap::find_associations(const std::vector<Surfel2D>& local_surfels, const Eigen::Isometry3f& sensor_pose) {
    std::vector<std::pair<int,int>> associations;
    associations.reserve(local_surfels.size());

    // no global map -> all new
    if (surfel_map_.empty()) {
        for (size_t i = 0; i < local_surfels.size(); ++i) {
            associations.push_back({i, -1});
        }
        return associations;
    }

    // build kdtree of global map if needed
    if (!kdtree_->getInputCloud() || kdtree_->getInputCloud()->empty()) {
        build_kdtree();
    }

    for (size_t i = 0; i < local_surfels.size(); ++i) {
        const auto& local = local_surfels[i];

        std::vector<int> cands;
        std::vector<float> d2s;
        pcl::PointXYZ query(local.center.x(), local.center.y(), local.center.z());
        kdtree_->radiusSearch(query, config_.max_association_distance, cands, d2s);

        // mahalanobis distance best fit
        int best_match = -1;
        float best_score = config_.max_mahanalobis_distance;

        for (int global_idx : cands) {
            const auto& global = surfel_map_[global_idx];

            float normal_angle = std::acos(std::max(-1.0f, std::min(1.0f, local.normal.dot(global.normal)))) * 180.0f / M_PI;

            if (normal_angle > config_.max_normal_angle_deg) continue;

            float maha_dist = compute_mahalanobis_distance(local, global);

            if (maha_dist < best_score) {
                best_score = maha_dist;
                best_match = global_idx;
            }
        }

        associations.push_back({i, best_match});
    }

    return associations;
}

float SurfelMap::compute_mahalanobis_distance(const Surfel2D& local_surfel, const GlobalSurfel2D& global_surfel) const {

    Eigen::Vector3f pos_diff = local_surfel.center - global_surfel.center;
    Eigen::Vector3f pos_variance = global_surfel.position_variance.array() + config_.measurement_noise_position;
    float pos_maha = (pos_diff.array().square() / pos_variance.array()).sum();
    
    float normal_angle = std::acos(std::max(-1.0f, std::min(1.0f, local_surfel.normal.dot(global_surfel.normal))));
    float normal_variance = global_surfel.normal_variance.mean() + config_.measurement_noise_normal;
    float normal_maha = (normal_angle * normal_angle) / normal_variance;

    return std::sqrt(pos_maha + normal_maha);
}




}; // namespace surface_inspection_planner
