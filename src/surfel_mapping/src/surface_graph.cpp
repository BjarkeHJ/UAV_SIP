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
                    stats.nodes_updated++;
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
    // if (!surfel.is_valid) return false;
    if (!surfel.is_mature) return false;
    if (surfel.confidence < params_.min_node_confidence) return false;
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
    if (new_voxel.x != old_voxel.x || new_voxel.y != old_voxel.y || new_voxel.z != old_voxel.z) {
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

void SurfaceGraph::create_edges_for_node(size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;

    const GraphNode& node = nodes_[node_idx];

    // candidates within max edge distance
    std::vector<size_t> candidates = find_nodes_in_radius(node.center, params_.max_edge_distance);
    std::vector<std::pair<size_t, float>> scored;

    for (size_t other_idx : candidates) {
        if (other_idx == node_idx) continue;
        if (!nodes_[other_idx].is_valid) continue;

        bool exists = false;
        for (size_t edge_idx : nodes_[node_idx].edge_indices) {
            if (!edges_[edge_idx].is_valid) continue;
            if (edges_[edge_idx].from_node == other_idx || edges_[edge_idx].to_node == other_idx) {
                exists = true;
                break;
            }
        }

        if (exists) continue;

        auto edge_opt = try_create_edge(node_idx, other_idx);
        if (edge_opt.has_value()) {
            scored.emplace_back(other_idx, edge_opt->total_cost);
        }
    }

    // sort by best scoring edge candidates
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    size_t current_neighbors = nodes_[node_idx].edge_indices.size();
    size_t max_new = (params_.max_neighbors_per_node > current_neighbors) ? (params_.max_neighbors_per_node - current_neighbors) : 0;
    size_t num_to_add = std::min(scored.size(), max_new);

    // add eges 
    for (size_t k = 0; k < num_to_add; ++k) {
        size_t other_idx = scored[k].first;

        // nb limit of other node
        if (nodes_[other_idx].edge_indices.size() >= params_.max_neighbors_per_node) continue;

        // return optional Edge
        auto edge_opt = try_create_edge(node_idx, other_idx);
        if (!edge_opt.has_value()) continue;

        GraphEdge edge = edge_opt.value();
        size_t edge_idx;
        if (!free_edge_slots_.empty()) {
            edge_idx = free_edge_slots_.back();
            free_edge_slots_.pop_back();
            edges_[edge_idx] = edge;
        }
        else {
            edge_idx = edges_.size();
            edges_.push_back(edge);
        }

        nodes_[node_idx].edge_indices.push_back(edge_idx);
        nodes_[other_idx].edge_indices.push_back(edge_idx);
        valid_edge_count_++;
    }
}

void SurfaceGraph::remove_edges_for_node(size_t node_idx) {
    if (node_idx >= nodes_.size()) return;

    GraphNode& node = nodes_[node_idx];

    for (size_t edge_idx : node.edge_indices) {
        if (edge_idx >= edges_.size()) continue;

        GraphEdge& edge = edges_[edge_idx];
        if (!edge.is_valid) continue;

        // remove from other node's edge list
        size_t other_idx = (edge.from_node == node_idx) ? edge.to_node : edge.from_node;
        if (other_idx < nodes_.size()) {
            auto& other_edges = nodes_[other_idx].edge_indices;
            other_edges.erase(
                std::remove(other_edges.begin(), other_edges.end(), edge_idx),
                other_edges.end()
            );
        }

        edge.is_valid = false;
        free_edge_slots_.push_back(edge_idx);
        valid_edge_count_--;
    }

    node.edge_indices.clear();
}

void SurfaceGraph::update_edges_for_node(size_t node_idx) {
    // do better update instead of remove + create...
    remove_edges_for_node(node_idx);
    create_edges_for_node(node_idx);
}

bool SurfaceGraph::check_connectivity(const GraphNode& n1, const GraphNode& n2) const {
    // distance
    float dist = (n1.center - n2.center).norm();
    if (dist < params_.min_edge_distance || dist > params_.max_edge_distance) return false;

    // normal dot
    float normal_dot = std::abs(n1.normal.dot(n2.normal));
    if (normal_dot < params_.min_normal_dot) return false;

    // plane step
    Eigen::Vector3f delta = n2.center - n1.center;
    float step1 = std::abs(delta.dot(n1.normal));
    float step2 = std::abs(delta.dot(n2.normal));
    if (step1 > params_.max_plane_step || step2 > params_.max_plane_step) return false;
    
    return true;
}

std::optional<GraphEdge> SurfaceGraph::try_create_edge(size_t from_node, size_t to_node) {
    if (from_node >= nodes_.size() || to_node >= nodes_.size()) {
        return std::nullopt;
    }

    const GraphNode& n1 = nodes_[from_node];
    const GraphNode& n2 = nodes_[to_node];

    if (!n1.is_valid || !n2.is_valid) {
        return std::nullopt;
    }

    if (!check_connectivity(n1, n2)) {
        return std::nullopt;
    }

    GraphEdge edge;
    edge.from_node = from_node;
    edge.to_node = to_node;
    edge.is_valid = true;

    Eigen::Vector3f edge_vec = n2.center - n1.center;
    edge.distance = edge_vec.norm();
    edge.normal_change = 1.0f - std::abs(n1.normal.dot(n2.normal));

    Eigen::Vector3f edge_dir = edge_vec.normalized();
    edge.alignment_cost = compute_alignment_cost(n1, n2, edge_dir);

    float step1 = std::abs(edge_vec.dot(n1.normal));
    float step2 = std::abs(edge_vec.dot(n2.normal));
    edge.step_height = std::max(step1, step2);

    // check structural alignment based on node surfel major axis (here )
    float align1 = std::abs(edge_dir.dot(n1.major_axis));
    float align2 = std::abs(edge_dir.dot(n2.major_axis));
    edge.is_structural = (n1.anisotropy > 0.5f && align1 > params_.alignment_threshold) || (n2.anisotropy > 0.5f && align2 > params_.alignment_threshold);

    compute_edge_cost(edge);

    return edge;
}

float SurfaceGraph::compute_alignment_cost(const GraphNode& from, const GraphNode& to, const Eigen::Vector3f& edge_dir) const {
    float cost = 0.0f;

    if (from.anisotropy > 0.3f) {
        float align = std::abs(edge_dir.dot(from.major_axis));
        cost += (1.0f - align) * from.anisotropy;
    }

    if (to.anisotropy) {
        float align = std::abs(edge_dir.dot(to.major_axis));
        cost += (1.0f - align) * to.anisotropy;
    }

    return cost * 0.5f;
}

void SurfaceGraph::compute_edge_cost(GraphEdge& edge) const {
    edge.total_cost = weights_.w_distance * edge.distance + 
                      weights_.w_normal_change * edge.normal_change + 
                      weights_.w_alignment * edge.alignment_cost + 
                      weights_.w_step * edge.step_height;

    if (edge.is_structural) {
        edge.total_cost -= weights_.structural_bonus;
        edge.total_cost = std::max(0.01f, edge.total_cost);
    }
}

void SurfaceGraph::set_weights(const EdgeCostWeights& w) {
    weights_ = w;
    for (auto& edge : edges_) {
        if (edge.is_valid) {
            compute_edge_cost(edge);
        }
    }
}

VoxelKey SurfaceGraph::point_to_voxel(const Eigen::Vector3f& p) const {
    float inv_size = 1.0f / voxel_size_;
    return VoxelKey{
        static_cast<int32_t>(std::floor(p.x() * inv_size)),
        static_cast<int32_t>(std::floor(p.y() * inv_size)),
        static_cast<int32_t>(std::floor(p.z() * inv_size))
    };
}

void SurfaceGraph::add_node_to_spatial_index(size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;

    auto key = point_to_voxel(nodes_[node_idx].center);
    node_voxel_index_[key].push_back(node_idx);
}

void SurfaceGraph::remove_node_from_spatial_index(size_t node_idx) {
    if (node_idx >= nodes_.size()) return;

    auto key = point_to_voxel(nodes_[node_idx].center);
    auto it = node_voxel_index_.find(key);

    if (it != node_voxel_index_.end()) {
        auto& indices = it->second;
        indices.erase(std::remove(indices.begin(), indices.end(), node_idx), indices.end());
        if (indices.empty()) {
            node_voxel_index_.erase(it);
        }
    }
}

size_t SurfaceGraph::find_nearest_node(const Eigen::Vector3f& point) const {
    if (nodes_.empty()) return 0;
    auto center_key = point_to_voxel(point);
    float best_dist_sq = std::numeric_limits<float>::max();
    size_t best_idx = 0;

    // searhc in growing voxel "circle" (NN)
    for (int r = 0; r <= 5; ++r) {
        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dz = -r; dz <= r; ++dz) {
                    if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) continue;

                    VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
                    auto it = node_voxel_index_.find(key);
                    if (it == node_voxel_index_.end()) continue;

                    for (size_t idx : it->second) {
                        if (!nodes_[idx].is_valid) continue;
                        float dist_sq = (nodes_[idx].center - point).squaredNorm();
                        if (dist_sq < best_dist_sq) {
                            best_dist_sq = dist_sq;
                            best_idx = idx;
                        }
                    }
                }
            }
        }

        // found
        if (best_dist_sq < std::numeric_limits<float>::max()) {
            break;
        }
    }

    return best_idx;
}

std::vector<size_t> SurfaceGraph::find_nodes_in_radius(const Eigen::Vector3f& point, float radius) const {
    std::vector<size_t> result;

    auto center_key = point_to_voxel(point);
    int search_radius = static_cast<int>(std::ceil(radius / voxel_size_)) + 1;
    float radius_sq = radius * radius;

    for (int dx = -search_radius; dx <= search_radius; ++dx) {
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dz = -search_radius; dz <= search_radius; ++dz) {
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
                auto it = node_voxel_index_.find(key);
                if (it == node_voxel_index_.end()) continue;

                for (size_t idx : it->second) {
                    if (!nodes_[idx].is_valid) continue;
                    if ((nodes_[idx].center - point).squaredNorm() <= radius_sq) {
                        result.push_back(idx);
                    }
                }
            }
        }
    }
    
    return result;
}

std::optional<size_t> SurfaceGraph::find_node_by_surfel(size_t surfel_idx) const {
    auto it = surfel_to_node_.find(surfel_idx);
    if (it != surfel_to_node_.end() && nodes_[it->second].is_valid) {
        return it->second;
    }

    return std::nullopt;
}

std::vector<size_t> SurfaceGraph::get_neighbors(size_t node_idx) const {
    std::vector<size_t> neighbors;
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) {
        return neighbors;
    }

    for (size_t edge_idx : nodes_[node_idx].edge_indices) {
        if (edge_idx >= edges_.size() || !edges_[edge_idx].is_valid) continue;

        const GraphEdge& edge = edges_[edge_idx];
        size_t neighbor = (edge.from_node == node_idx) ? edge.to_node : edge.from_node;
        if (nodes_[neighbor].is_valid) {
            neighbors.push_back(neighbor);
        }
    }

    return neighbors;
}








