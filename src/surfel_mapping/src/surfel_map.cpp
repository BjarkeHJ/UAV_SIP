#include "surfel_mapping/surfel_map.hpp"

using namespace surface_inspection_planning;

SurfelMap::SurfelMap() : params_(), next_surfel_id_(1) {
    cos_normal_thresh_ = std::cos(params_.normal_thresh_deg * M_PI / 180.0f);
}

SurfelMap::SurfelMap(const Params& p) : params_(p), next_surfel_id_(1) {
    cos_normal_thresh_ = std::cos(params_.normal_thresh_deg * M_PI / 180.0f);
}

/* PUBLIC */
size_t SurfelMap::add_surfel(const Surfel& surfel) {
    Surfel s = surfel;
    s.id = next_surfel_id_;
    s.is_valid = true;
    size_t idx = surfels_.size();
    surfels_.push_back(s);

    VoxelKey key = point_to_voxel(s.center);
    voxel_index_[key].push_back(idx);
    return idx;
}

size_t SurfelMap::create_surfel(const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, uint64_t timestamp) {
    Surfel s;
    s.initialize(center, normal, radius);
    s.creation_stamp = timestamp;
    s.last_update_stamp = timestamp;
    return add_surfel(s);
}

void SurfelMap::find_association_candidates(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, std::vector<std::pair<size_t, float>>& candidates) const {
    candidates.clear();

    VoxelKey center_key = point_to_voxel(point);
    
    // check proximity voxels for association surfels
    constexpr int search_radius = 1;
    for (int dx = -search_radius; dx <= search_radius; ++dx) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dz = -search_radius; dz <= search_radius; ++dz) {
                
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};

                auto it = voxel_index_.find(key);
                if (it == voxel_index_.end()) continue;

                for (size_t surfel_idx : it->second) {
                    if (surfel_idx >= surfels_.size()) continue;

                    const Surfel& surfel = surfels_[surfel_idx];
                    if (!surfel.is_valid) continue;

                    // Check normal similarity
                    float normal_dot = std::abs(normal.dot(surfel.normal));
                    if (normal_dot < cos_normal_thresh_) continue;

                    // project point onto surfel
                    auto [tangent_coords, normal_dist] = surfel.project_point(point);
                    
                    if (std::abs(normal_dist) > params_.normal_dist_thresh) continue;

                    float mahal_sq = surfel.mahalanobis_distance_sq(tangent_coords);
                    if (mahal_sq < params_.mahal_thresh) {
                        candidates.emplace_back(surfel_idx, mahal_sq);
                    }
                }
            }
        }
    }

    // sort candidates by mahalanobis distance
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
}

int SurfelMap::find_best_association(const Eigen::Vector3f& point, const Eigen::Vector3f& normal) const {
    std::vector<std::pair<size_t, float>> candidates;
    find_association_candidates(point, normal, candidates);
    if (candidates.empty()) return -1;
    return static_cast<int>(candidates[0].first);
}

void SurfelMap::update_spatial_index(size_t surfel_idx) {
    if (surfel_idx >= surfels_.size()) return;
    const Surfel& surfel = surfels_[surfel_idx];
    VoxelKey new_key = point_to_voxel(surfel.center);

    // remove key from old voxel(s) - could be optimized
    for (auto& [key, indices] : voxel_index_) {
        auto it = std::find(indices.begin(), indices.end(), surfel_idx);
        if (it != indices.end()) {
            indices.erase(it);
        }
    }

    // add to new voxel
    voxel_index_[new_key].push_back(surfel_idx);
}

void SurfelMap::prune_and_rebuild() {
    // mark low confidence
    int marked_to_remove = 0;
    for (auto& surfel : surfels_) {
        bool marked = !surfel.is_valid || (surfel.confidence < params_.min_confidence && surfel.observation_count >= params_.min_observations);
        if (marked) {
            surfel.is_valid = false;
            marked_to_remove++;
        }
    }

    // dont rebuild if non to remove
    if (marked_to_remove == 0) return;

    std::vector<Surfel> new_surfels;
    new_surfels.reserve(surfels_.size());
    for (const auto& surfel : surfels_) {
        if (surfel.is_valid) {
            new_surfels.push_back(surfel);
        }
    }

    surfels_ = std::move(new_surfels);

    // rebuild spatial index
    voxel_index_.clear();
    for (size_t i = 0; i < surfels_.size(); ++i) {
        VoxelKey key = point_to_voxel(surfels_[i].center);
        voxel_index_[key].push_back(i);
    }
}

void SurfelMap::apply_confidence_decay() {
    for (auto& surfel : surfels_) {
        surfel.confidence *= params_.confidence_decay;
    }
}

SurfelMap::MapStats SurfelMap::get_stats() const {
    MapStats stats;
    stats.total_surfels = surfels_.size();
    stats.voxels_occupied = voxel_index_.size();

    float conf_sum = 0.0f;
    float count_sum = 0.0f;
    for (const auto& s : surfels_) {
        if (s.is_valid) {
            stats.valid_surfels++;
            conf_sum += s.confidence;
            count_sum += static_cast<float>(s.point_count);
        }
    }

    if (stats.valid_surfels > 0) {
        stats.avg_confidence = conf_sum / static_cast<float>(stats.valid_surfels);
        stats.avg_point_count = count_sum / static_cast<float>(stats.valid_surfels);
    }

    return stats;
}

/* Private */

VoxelKey SurfelMap::point_to_voxel(const Eigen::Vector3f& p) const {
    float inv_size = 1.0f / params_.voxel_size;
    return VoxelKey{
        static_cast<int32_t>(std::floor(p.x() * inv_size)),
        static_cast<int32_t>(std::floor(p.y() * inv_size)),  
        static_cast<int32_t>(std::floor(p.z() * inv_size))  
    };
}