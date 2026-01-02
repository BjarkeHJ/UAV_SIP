#include "surfel_mapping/surface_graph.hpp"

using namespace surface_inspection_planning;


SurfaceGraph::SurfaceGraph() 
    : params_()
    , weights_()
    , voxel_size_(0.5f) 
{}

SurfaceGraph::SurfaceGraph(const ConnectivityParams& params) 
    : params_(params)
    , weights_()
    , voxel_size_(params_.max_edge_distance)
{}

void SurfaceGraph::clear() {
    nodes_.clear();
    edges_.clear();
    free_node_slots_.clear();
    free_edge_slots_.clear();
    surfel_to_node_.clear();
    surfel_to_node_.clear();
    node_voxel_index_.clear();
    valid_node_count_ = 0;
    valid_edge_count_ = 0;
    last_update_frame_ = 0;
}

void SurfaceGraph::build_from_map(const SurfelMap& map) {
    // Full graph rebuild from map
    clear();

    const auto& surfels = map.get_surfels();

    for (size_t i = 0; i < surfels.size(); ++i) {
        const Surfel& surfel = surfels[i];
        if (!surfel_qualifies_as_node(surfel)) continue;
        create_node(map, i);
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].is_valid) continue;
        create_edges_for_node(i);
    }
}

GraphUpdateStats SurfaceGraph::update_incremental(const SurfelMap& map, const std::vector<size_t>& new_surfels, const std::vector<size_t>& updated_surfels, const std::vector<size_t>& removed_surfels) {
    auto start_time = std::chrono::high_resolution_clock::now();
    GraphUpdateStats stats;

    const auto& surfels = map.get_surfels();
    
    // handle nodes for removed surfels
    for (size_t surfel_idx : removed_surfels) {
        auto it = surfel_to_node_.find(surfel_idx);
        if (it != surfel_to_node_.end()) {
            invalidate_node(it->second);
            stats.nodes_removed++;
        }
    }

    // handle updating existing nodes
    for (size_t surfel_idx : updated_surfels) {
        if (surfel_idx >= surfels.size()) continue;
        const Surfel& surfel = surfels[surfel_idx];

        auto it = surfel_to_node_.find(surfel_idx);

        if (it != surfel_to_node_.end()) {
            if (!surfel.is_valid || !surfel_qualifies_as_node(surfel)) {
                invalidate_node(it->second);
                stats.nodes_removed++;
            }
            else {
                update_node_properties(map, it->second);
                stats.nodes_updated++;
            }
        }
        else {
            if (surfel.is_valid && surfel_qualifies_as_node(surfel)) {
                create_node(map, surfel_idx);
                stats.nodes_added++;
            }
        }
    }

    // handle node from new surfels
    for (size_t surfel_idx : new_surfels) {
        if (surfel_idx >= surfels.size()) continue;
        const Surfel& surfel = surfels[surfel_idx];

        if (surfel.is_valid && surfel_qualifies_as_node(surfel)) {
            if (surfel_to_node_.find(surfel_idx) == surfel_to_node_.end()) {
                create_node(map, surfel_idx);
                stats.nodes_added++;
            }
        }
    }

    // create edges for new nodes
    for (size_t surfel_idx : new_surfels) {
        auto it = surfel_to_node_.find(surfel_idx);
        if (it != surfel_to_node_.end() && nodes_[it->second].is_valid) {
            size_t edges_before = valid_edge_count_;
            create_edges_for_node(it->second);
            stats.edges_added += (valid_edge_count_ - edges_before);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    stats.update_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return stats;
}

GraphUpdateStats SurfaceGraph::update_from_map(const SurfelMap& map) {
    auto start_time = std::chrono::high_resolution_clock::now();
    GraphUpdateStats stats;

    const auto& surfels = map.get_surfels();
    std::unordered_set<size_t> seen_surfel_indices;

    for (size_t i = 0; i < surfels.size(); ++i) {
        const Surfel& surfel = surfels[i];
        if (!surfel.is_valid) continue;

        seen_surfel_indices.insert(i);

        auto node_it = surfel_to_node_.find(i);
        if (node_it == surfel_to_node_.end()) {
            if (surfel_qualifies_as_node(surfel)) {
                create_node(map, i);
                create_edges_for_node(nodes_.size() - 1);
                stats.nodes_added++;
            }
        }
        else {
            size_t node_idx = node_it->second;
            GraphNode& node = nodes_[node_idx];

            if (!node.is_valid) continue;

            if (surfel.last_update_stamp > node.last_surfel_stamp) {
                if (!surfel_qualifies_as_node(surfel)) {
                    invalidate_node(node_idx);
                    stats.nodes_removed++;
                }
                else {
                    update_node_properties(map, node_idx);
                    stats.nodes_updated;
                }
            }
        }
    }

    std::vector<size_t> nodes_to_remove;
    for (const auto& [surfel_idx, node_idx] : surfel_to_node_) {
        if (seen_surfel_indices.find(surfel_idx) == seen_surfel_indices.end()) {
            if (nodes_[node_idx].is_valid) {
                nodes_to_remove.push_back(node_idx);
            }
        }
    }

    for (size_t node_idx : nodes_to_remove) {
        invalidate_node(node_idx);
        stats.nodes_removed++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    stats.update_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return stats;
}

bool SurfaceGraph::surfel_qualifies_as_node(const Surfel& surfel) const {
    if (!surfel.is_valid) return false;
    if (surfel.confidence < params_.min_node_confidence) return false;
    if (surfel.get_radius() < params_.min_node_radius) return false;
    if (surfel.observation_count < params_.min_node_observations) return false;
    return true;
}

size_t SurfaceGraph::create_node(const SurfelMap& map, size_t surfel_idx) {
    const Surfel& surfel = map.get_surfels()[surfel_idx];

    GraphNode node;
    node.surfel_idx = surfel_idx;
    node.center = surfel.center;
    node.normal = surfel.normal;
    node.major_axis = compute_major_axis_3d(surfel);
    node.major_radius = std::sqrt(surfel.eigenvalues.maxCoeff());
    node.minor_radius = std::sqrt(surfel.eigenvalues.minCoeff());
    node.anisotropy = surfel.get_planarity();
    node.importance = compute_node_importance(surfel);
    node.is_valid = true;
    node.last_surfel_stamp = surfel.last_update_stamp;

    size_t node_idx;

    // reuse free slots (avoid allocation if possible)
    if (!free_node_slots_.empty()) {
        node_idx = free_node_slots_.back();
        free_node_slots_.pop_back();
        nodes_[node_idx] = node;
    }
    else {
        node_idx = nodes_.size();
        nodes_.push_back(node);
    }

    surfel_to_node_[surfel_idx] = node_idx;
    add_node_to_spatial_index(node_idx);
    valid_node_count_++;

    return node_idx;
}

void SurfaceGraph::update_node_properties(const SurfelMap& map, size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;

    GraphNode& node = nodes_[node_idx];
    const Surfel& surfel = map.get_surfels()[node.surfel_idx];

    Eigen::Vector3f old_center = node.center;
    VoxelKey old_voxel = point_to_voxel(old_center);

    node.center = surfel.center;
    node.normal = surfel.normal;
    node.major_axis = compute_major_axis_3d(surfel);
    node.major_radius = std::sqrt(surfel.eigenvalues.maxCoeff());
    node.minor_radius = std::sqrt(surfel.eigenvalues.minCoeff());
    node.anisotropy = surfel.get_planarity();
    node.importance = compute_node_importance(surfel);
    node.last_surfel_stamp = surfel.last_update_stamp;

    VoxelKey new_voxel = point_to_voxel(node.center);
    if (new_voxel.x != old_voxel.x || new_voxel.y != old_voxel.y || new_voxel.z != new_voxel.z) {
        remove_node_from_spatial_index(node_idx);
        add_node_to_spatial_index(node_idx);
    }

    float center_change = (node.center - old_center).norm();
    if (center_change > 0.05f) {
        update_edges_for_node(node_idx);
    }
}

void SurfaceGraph::invalidate_node(size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;

    GraphNode& node = nodes_[node_idx];

    remove_edges_for_node(node_idx);
    remove_node_from_spatial_index(node_idx);
    surfel_to_node_.erase(node.surfel_idx);

    node.is_valid = false;
    node.edge_indices.clear();
    free_edge_slots_.push_back(node_idx);
    valid_node_count_--;
}

float SurfaceGraph::compute_node_importance(const Surfel& surfel) const {
    float area = M_PI * surfel.eigenvalues(0) * surfel.eigenvalues(1);
    float area_score = std::min(1.0f, area / 0.1f);

    float conf_score = surfel.confidence;

    float obs_score = std::min(1.0f, static_cast<float>(surfel.observation_count) / 20.0f);

    return 0.4f * area_score + 0.4f * conf_score + 0.2f * obs_score;
}

Eigen::Vector3f SurfaceGraph::compute_major_axis_3d(const Surfel& surfel) const {
    int max_idx = (surfel.eigenvalues(0) > surfel.eigenvalues(1)) ? 0 : 1;
    Eigen::Vector2f major_2d = surfel.eigenvectors.col(max_idx);
    Eigen::Vector3f major_3d = major_2d.x() * surfel.tangent_u + major_2d.y() * surfel.tangent_v;
    return major_3d.normalized();
}

// NEXT UP: create_edges_for_node... (in claude file)









