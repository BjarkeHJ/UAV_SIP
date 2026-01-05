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
    VoxelKey key = point_to_voxel(surfel.center);

    auto it = voxel_index_.find(key);
    if (it != voxel_index_.end() && it->second.size() >= params_.max_surfels_per_voxel) {

        // Voxel full - try merging with most similar existing surfel
        size_t best_merge_idx = INVALID_SURFEL_IDX;
        float best_similarity = -1.0f;

        for (size_t idx : it->second) {
            if (idx >= surfels_.size() || !surfels_[idx].is_valid) continue;

            if (should_merge(surfels_[idx], surfel)) {
                float normal_dot = std::abs(surfel.normal.dot(surfels_[idx].normal));
                float dist = (surfel.center - surfels_[idx].center).norm();
                float similarity = normal_dot / (1.0f + dist * 10.0f);

                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_merge_idx = idx;
                }
            }
        }

        // merge into existing if fit
        if (best_merge_idx != INVALID_SURFEL_IDX) {
            Surfel& target = surfels_[best_merge_idx];
            float total_w = target.total_weight + surfel.total_weight;
            float alpha = surfel.total_weight / total_w;
            target.center = target.center + alpha * (surfel.center - target.center);
            target.sum_normals = target.sum_normals + surfel.sum_normals;
            target.normal = target.sum_normals.normalized();
            target.compute_tangential_basis();
            target.total_weight = total_w;
            target.point_count += surfel.point_count;
            target.observation_count += surfel.observation_count;
            target.confidence = std::max(target.confidence, surfel.confidence);
            target.needs_eigen_update = true;
            return best_merge_idx;
        }

        // no merge candidates - replace weakest if new is better
        size_t weakest_idx = find_weakest_in_voxel(key);
        if (weakest_idx != INVALID_SURFEL_IDX && surfel.confidence > surfels_[weakest_idx].confidence) {
            Surfel s = surfel;
            s.id = next_surfel_id_++;
            s.is_valid = true;
            s.voxel_x = key.x;
            s.voxel_y = key.y;
            s.voxel_z = key.z;
            surfels_[weakest_idx] = s;
            return weakest_idx;
        }

        // cannot add - reject
        return INVALID_SURFEL_IDX;
    }

    // normal case: voxel has space - add new
    Surfel s = surfel;
    s.id = next_surfel_id_;
    s.is_valid = true;

    // VoxelKey key = point_to_voxel(s.center);
    s.voxel_x = key.x;
    s.voxel_y = key.y;
    s.voxel_z = key.z;

    size_t idx = surfels_.size();
    surfels_.push_back(s);
    voxel_index_[key].push_back(idx);

    return idx;
}

size_t SurfelMap::create_surfel(const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, uint64_t timestamp) {
    
    // check if mergin is possible (priority)
    size_t merge_target = find_merge_target(center, normal);
    if (merge_target != INVALID_SURFEL_IDX) {
        // dont create new - accumulator points go to existing surfel
        // returning merge target, so caller knows to direct points there
        return merge_target;
    }

    Surfel s;
    s.initialize(center, normal, radius);
    s.creation_stamp = timestamp;
    s.last_update_stamp = timestamp;
    return add_surfel(s);
}

size_t SurfelMap::find_merge_target(const Eigen::Vector3f& center, const Eigen::Vector3f& normal) const {
    VoxelKey center_key = point_to_voxel(center);
    size_t best_idx = INVALID_SURFEL_IDX;
    float best_score = -1.0f;

    // searching for best merge target in 3x3x3 neighborhood
    int merge_rad = 1;
    for (int dx = -merge_rad; dx <= merge_rad; ++dx) {
        for (int dy = -merge_rad; dy <= merge_rad; ++dy) {
            for (int dz = -merge_rad; dz <= merge_rad; ++dz) {
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};

                auto it = voxel_index_.find(key);
                if (it == voxel_index_.end()) continue;

                for (size_t idx : it->second) {
                    if (idx >= surfels_.size() || !surfels_[idx].is_valid) continue;

                    const Surfel& existing = surfels_[idx];
                    if (!existing.is_valid) continue;

                    float normal_dot = std::abs(normal.dot(existing.normal));
                    if (normal_dot < params_.merge_normal_dot) continue;

                    float dist = (center - existing.center).norm();
                    float merge_radius = existing.get_radius() * 2.0f + params_.merge_center_dist;
                    if (dist > merge_radius) continue;

                    float score = normal_dot * existing.confidence / (1.0f + dist);
                    if (score > best_score) {
                        best_score = score;
                        best_idx = idx;
                    }
                }
            }
        }
    }
    return best_idx;
}

void SurfelMap::find_association_candidates(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, std::vector<std::pair<size_t, float>>& candidates) const {
    candidates.clear();

    // get voxel key from point
    VoxelKey center_key = point_to_voxel(point);
    
    // check proximity voxels for surfel assciation
    constexpr int search_radius = 1;
    for (int dx = -search_radius; dx <= search_radius; ++dx) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dz = -search_radius; dz <= search_radius; ++dz) {
                
                // key from nb voxel
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};

                auto it = voxel_index_.find(key);
                if (it == voxel_index_.end()) continue;

                // it points to vector of surfels (idx) in the voxel
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

                    // check mahalanobis distance (point to surfel-distribution)
                    float mahal_sq = surfel.mahalanobis_distance_sq(tangent_coords);
                    if (mahal_sq < params_.mahal_thresh) {
                        candidates.emplace_back(surfel_idx, mahal_sq);
                    }
                }
            }
        }
    }

    // sort candidates by mahalanobis distance (best fit is smallest mahalanobis distance)
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
}

int SurfelMap::find_best_association(const Eigen::Vector3f& point, const Eigen::Vector3f& normal) const {
    // Extract candidates and return the best fit surfel 
    std::vector<std::pair<size_t, float>> candidates;
    find_association_candidates(point, normal, candidates);
    if (candidates.empty()) return -1;
    return static_cast<int>(candidates[0].first);
}

void SurfelMap::update_spatial_index(size_t surfel_idx) {
    // Takes surfel index and remaps to new voxel if center changed significantly
    if (surfel_idx >= surfels_.size()) return;
    
    Surfel& surfel = surfels_[surfel_idx];
    VoxelKey new_key = point_to_voxel(surfel.center);
    
    // Fast path: voxel unchanged (most common case)
    if (new_key.x == surfel.voxel_x && 
        new_key.y == surfel.voxel_y && 
        new_key.z == surfel.voxel_z) {
        return;
    }
    
    // Slow path: voxel changed, need to update index
    VoxelKey old_key{surfel.voxel_x, surfel.voxel_y, surfel.voxel_z};
    
    // Remove from old voxel 
    auto old_it = voxel_index_.find(old_key);
    if (old_it != voxel_index_.end()) {
        auto& indices = old_it->second;
        auto it = std::find(indices.begin(), indices.end(), surfel_idx);
        if (it != indices.end()) {
            // Swap-and-pop for O(1) removal (order doesn't matter)
            std::swap(*it, indices.back());
            indices.pop_back();
        }
        // Clean up empty voxels
        if (indices.empty()) {
            voxel_index_.erase(old_it);
        }
    }
    
    // Add to new voxel
    voxel_index_[new_key].push_back(surfel_idx);
    
    // Update cache
    surfel.voxel_x = new_key.x;
    surfel.voxel_y = new_key.y;
    surfel.voxel_z = new_key.z;
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

size_t SurfelMap::merge_similar_surfels() {
    size_t merge_count = 0;

    std::vector<std::pair<size_t, size_t>> to_merge;

    // Check voxel surfels for similarity and fuse if deemed fitting
    for (auto& [key, indices] : voxel_index_) {
        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx_i = indices[i];
            if (idx_i >= surfels_.size() || !surfels_[idx_i].is_valid) continue;

            for (size_t j = i + 1; j < indices.size(); ++j) {
                size_t idx_j = indices[i];
                if (idx_j >= surfels_.size() || !surfels_[idx_j].is_valid) continue;

                if (should_merge(surfels_[idx_i], surfels_[idx_j])) {
                    if (surfels_[idx_i].confidence >= surfels_[idx_j].confidence) {
                        to_merge.emplace_back(idx_i, idx_j);
                    }
                    else {
                        to_merge.emplace_back(idx_j, idx_i);
                    }
                }
            }
        }
    }

    for (auto& [into, from] : to_merge) {
        if (surfels_[into].is_valid && surfels_[from].is_valid) {
            merge_surfels(into, from);
            merge_count++;
        }
    }

    return merge_count;
}

SurfelMap::MapStats SurfelMap::get_stats() const {
    MapStats stats;
    stats.total_surfels = surfels_.size();
    stats.voxels_occupied = voxel_index_.size();

    float conf_sum = 0.0f;
    float count_sum = 0.0f;
    float r_sum = 0.0f;

    for (const auto& s : surfels_) {
        // if (s.is_valid) {
        if (s.is_mature) {
            stats.valid_surfels++; // actual valid and mature surfels
            conf_sum += s.confidence;
            count_sum += static_cast<float>(s.point_count);
            r_sum += s.get_radius();
        }
    }

    if (stats.valid_surfels > 0) {
        stats.avg_confidence = conf_sum / static_cast<float>(stats.valid_surfels);
        stats.avg_point_count = count_sum / static_cast<float>(stats.valid_surfels);
        stats.avg_radius = r_sum / static_cast<float>(stats.valid_surfels);
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

bool SurfelMap::should_merge(const Surfel& s1, const Surfel& s2) const {
    float normal_dot = std::abs(s1.normal.dot(s2.normal));
    if (normal_dot < params_.merge_normal_dot) return false;

    float dist = (s1.center - s2.center).norm();
    float combined_radius = s1.get_radius() + s2.get_radius();

    if (dist > combined_radius + params_.merge_center_dist) return false;

    return true;
}

void SurfelMap::merge_surfels(size_t into_idx, size_t from_idx) {
    if (into_idx >= surfels_.size() || from_idx >= surfels_.size()) return;
    if (into_idx == from_idx) return;

    Surfel& target = surfels_[into_idx];
    Surfel& source = surfels_[from_idx];

    if (!target.is_valid || !source.is_valid) return;

    float total_w = target.total_weight + source.total_weight;
    float alpha = source.total_weight / total_w;

    target.center = target.center + alpha * (source.center - target.center);

    target.total_weight = total_w;
    target.point_count += source.point_count;
    target.observation_count += source.observation_count;
    target.sum_sq_normal_dist += source.sum_sq_normal_dist;

    float max_radius = std::max(target.get_radius(), source.get_radius());
    float separation = (target.center - source.center).norm();
    float new_radius = std::max(max_radius, (max_radius + separation) * 0.6f);
    float new_var = new_radius * new_radius;
    target.eigenvalues = Eigen::Vector2f::Constant(new_var);
    target.covariance = target.eigenvalues.asDiagonal();
    target.needs_eigen_update = true;

    target.confidence = std::max(target.confidence, source.confidence);
    target.last_update_stamp = std::max(target.last_update_stamp, source.last_update_stamp);

    source.is_valid = false;

    update_spatial_index(into_idx);
}

size_t SurfelMap::find_weakest_in_voxel(const VoxelKey& key) const {
    auto it = voxel_index_.find(key);
    if (it == voxel_index_.end()) return INVALID_SURFEL_IDX;

    size_t weakest_idx = INVALID_SURFEL_IDX;
    float weakest_conf = std::numeric_limits<float>::max();

    for (size_t idx : it->second) {
        if (idx < surfels_.size() && surfels_[idx].is_valid) {
            if (surfels_[idx].confidence < weakest_conf) {
                weakest_conf = surfels_[idx].confidence;
                weakest_idx = idx;
            }
        }
    }

    return weakest_idx;
}
