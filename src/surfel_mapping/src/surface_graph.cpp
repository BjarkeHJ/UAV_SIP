#include "surfel_mapping/surface_graph.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace surface_inspection_planning {

SurfaceGraph::SurfaceGraph(const SurfelMap& map)
    : map_(map)
    , params_()
    , voxel_size_(params_.max_edge_distance)
{}

SurfaceGraph::SurfaceGraph(const SurfelMap& map, const GraphParams& params)
    : map_(map)
    , params_(params)
    , voxel_size_(params.max_edge_distance)
{}

void SurfaceGraph::clear() {
    nodes_.clear();
    edges_.clear();
    free_node_slots_.clear();
    free_edge_slots_.clear();
    surfel_to_node_.clear();
    spatial_index_.clear();
    valid_node_count_ = 0;
    valid_edge_count_ = 0;
}

// BUILD / UPDATE
void SurfaceGraph::rebuild() {
    clear();
    
    const auto& surfels = map_.get_surfels();
    
    // Phase 1: Create nodes for qualifying surfels
    for (size_t i = 0; i < surfels.size(); ++i) {
        if (surfel_qualifies(i)) {
            create_node(i);
        }
    }
    
    // Phase 2: Create edges
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].is_valid) {
            create_edges_for_node(i);
        }
    }
}

GraphUpdateStats SurfaceGraph::update_incremental(const std::vector<size_t>& new_surfels, const std::vector<size_t>& updated_surfels, const std::vector<size_t>& removed_surfels) {
    auto start = std::chrono::high_resolution_clock::now();
    GraphUpdateStats stats;
    
    const auto& surfels = map_.get_surfels();
    
    // Step 1: Handle removed surfels
    for (size_t surfel_idx : removed_surfels) {
        auto it = surfel_to_node_.find(surfel_idx);
        if (it != surfel_to_node_.end()) {
            invalidate_node(it->second);
            stats.nodes_removed++;
        }
    }
    
    // Step 2: Handle updated surfels
    for (size_t surfel_idx : updated_surfels) {
        if (surfel_idx >= surfels.size()) continue;
        
        auto it = surfel_to_node_.find(surfel_idx);
        bool currently_qualifies = surfel_qualifies(surfel_idx);
        
        if (it != surfel_to_node_.end()) {
            // Node exists
            if (!currently_qualifies) {
                // No longer qualifies - remove
                invalidate_node(it->second);
                stats.nodes_removed++;
            } else {
                // Still qualifies - update spatial index and edges
                size_t node_idx = it->second;
                remove_from_spatial_index(node_idx);
                add_to_spatial_index(node_idx);
                
                // Recreate edges (geometry may have changed)
                remove_edges_for_node(node_idx);
                create_edges_for_node(node_idx);
            }
        } else if (currently_qualifies) {
            // Node doesn't exist but now qualifies - create
            size_t node_idx = create_node(surfel_idx);
            create_edges_for_node(node_idx);
            stats.nodes_added++;
        }
    }
    
    // Step 3: Handle new surfels
    for (size_t surfel_idx : new_surfels) {
        if (surfel_idx >= surfels.size()) continue;
        if (surfel_to_node_.count(surfel_idx) > 0) continue;  // Already processed
        
        if (surfel_qualifies(surfel_idx)) {
            size_t node_idx = create_node(surfel_idx);
            create_edges_for_node(node_idx);
            stats.nodes_added++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    stats.update_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return stats;
}

void SurfaceGraph::compact() {
    // Rebuild nodes without invalid entries
    std::vector<GraphNode> new_nodes;
    std::unordered_map<size_t, size_t> old_to_new;
    
    new_nodes.reserve(valid_node_count_);
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].is_valid) {
            old_to_new[i] = new_nodes.size();
            new_nodes.push_back(nodes_[i]);
            new_nodes.back().edge_indices.clear();
        }
    }
    
    // Rebuild edges
    std::vector<GraphEdge> new_edges;
    new_edges.reserve(valid_edge_count_);
    
    for (const auto& edge : edges_) {
        if (!edge.is_valid) continue;
        
        auto from_it = old_to_new.find(edge.from_node);
        auto to_it = old_to_new.find(edge.to_node);
        if (from_it == old_to_new.end() || to_it == old_to_new.end()) continue;
        
        GraphEdge new_edge = edge;
        new_edge.from_node = from_it->second;
        new_edge.to_node = to_it->second;
        
        size_t edge_idx = new_edges.size();
        new_edges.push_back(new_edge);
        
        new_nodes[new_edge.from_node].edge_indices.push_back(edge_idx);
        new_nodes[new_edge.to_node].edge_indices.push_back(edge_idx);
    }
    
    // Update surfel_to_node mapping
    std::unordered_map<size_t, size_t> new_surfel_to_node;
    for (const auto& [surfel_idx, old_node_idx] : surfel_to_node_) {
        auto it = old_to_new.find(old_node_idx);
        if (it != old_to_new.end()) {
            new_surfel_to_node[surfel_idx] = it->second;
        }
    }
    
    // Apply changes
    nodes_ = std::move(new_nodes);
    edges_ = std::move(new_edges);
    surfel_to_node_ = std::move(new_surfel_to_node);
    
    free_node_slots_.clear();
    free_edge_slots_.clear();
    
    valid_node_count_ = nodes_.size();
    valid_edge_count_ = edges_.size();
    
    rebuild_spatial_index();
}

// NODE MANAGEMENT
bool SurfaceGraph::surfel_qualifies(size_t surfel_idx) const {
    const auto& surfels = map_.get_surfels();
    if (surfel_idx >= surfels.size()) return false;
    
    const Surfel& s = surfels[surfel_idx];
    
    if (!s.is_valid) return false;
    if (s.get_radius() < params_.min_node_radius) return false;
    if (s.confidence < params_.min_node_confidence) return false;
    if (s.observation_count < params_.min_node_observations) return false;
    
    return true;
}

size_t SurfaceGraph::create_node(size_t surfel_idx) {
    GraphNode node;
    node.surfel_idx = surfel_idx;
    node.is_valid = true;
    node.inspected = false;
    node.inspection_quality = 0.0f;
    
    size_t node_idx;
    if (!free_node_slots_.empty()) {
        node_idx = free_node_slots_.back();
        free_node_slots_.pop_back();
        nodes_[node_idx] = node;
    } else {
        node_idx = nodes_.size();
        nodes_.push_back(node);
    }
    
    surfel_to_node_[surfel_idx] = node_idx;
    add_to_spatial_index(node_idx);
    valid_node_count_++;
    
    return node_idx;
}

void SurfaceGraph::invalidate_node(size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;
    
    GraphNode& node = nodes_[node_idx];
    
    // Remove edges
    remove_edges_for_node(node_idx);
    
    // Remove from indices
    remove_from_spatial_index(node_idx);
    surfel_to_node_.erase(node.surfel_idx);
    
    // Mark invalid
    node.is_valid = false;
    node.edge_indices.clear();
    free_node_slots_.push_back(node_idx);
    valid_node_count_--;
}

// EDGE MANAGEMENT
void SurfaceGraph::create_edges_for_node(size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;
    
    // Find nearby nodes
    std::vector<size_t> candidates = find_nodes_in_radius(
        center(node_idx), params_.max_edge_distance
    );
    
    // Score and filter
    std::vector<std::pair<size_t, float>> scored;
    
    for (size_t other_idx : candidates) {
        if (other_idx == node_idx) continue;
        if (!nodes_[other_idx].is_valid) continue;
        
        // Check if edge already exists
        bool exists = false;
        for (size_t eidx : nodes_[node_idx].edge_indices) {
            if (!edges_[eidx].is_valid) continue;
            if (edges_[eidx].from_node == other_idx || edges_[eidx].to_node == other_idx) {
                exists = true;
                break;
            }
        }
        if (exists) continue;
        
        if (!check_connectivity(node_idx, other_idx)) continue;
        
        GraphEdge edge;
        float cost = compute_edge_cost(node_idx, other_idx, edge);
        scored.emplace_back(other_idx, cost);
    }
    
    // Sort by cost, limit neighbors
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t current_count = nodes_[node_idx].edge_indices.size();
    size_t max_new = (params_.max_neighbors_per_node > current_count)
                     ? (params_.max_neighbors_per_node - current_count) : 0;
    size_t to_add = std::min(scored.size(), max_new);
    
    for (size_t k = 0; k < to_add; ++k) {
        size_t other_idx = scored[k].first;
        
        // Check other node's neighbor limit
        if (nodes_[other_idx].edge_indices.size() >= params_.max_neighbors_per_node) {
            continue;
        }
        
        // Create edge
        GraphEdge edge;
        edge.from_node = node_idx;
        edge.to_node = other_idx;
        edge.is_valid = true;
        compute_edge_cost(node_idx, other_idx, edge);
        
        size_t edge_idx;
        if (!free_edge_slots_.empty()) {
            edge_idx = free_edge_slots_.back();
            free_edge_slots_.pop_back();
            edges_[edge_idx] = edge;
        } else {
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
        
        // Remove from other node
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

bool SurfaceGraph::check_connectivity(size_t node_i, size_t node_j) const {
    const Eigen::Vector3f& c1 = center(node_i);
    const Eigen::Vector3f& c2 = center(node_j);
    const Eigen::Vector3f& n1 = normal(node_i);
    const Eigen::Vector3f& n2 = normal(node_j);
    
    // Distance
    float dist = (c2 - c1).norm();
    if (dist < params_.min_edge_distance || dist > params_.max_edge_distance) {
        return false;
    }
    
    // Normal compatibility
    float normal_dot = std::abs(n1.dot(n2));
    if (normal_dot < params_.min_normal_dot) {
        return false;
    }
    
    // Plane step
    Eigen::Vector3f delta = c2 - c1;
    float step1 = std::abs(delta.dot(n1));
    float step2 = std::abs(delta.dot(n2));
    if (step1 > params_.max_plane_step || step2 > params_.max_plane_step) {
        return false;
    }
    
    return true;
}

float SurfaceGraph::compute_edge_cost(size_t node_i, size_t node_j, GraphEdge& edge) const {
    const Eigen::Vector3f& c1 = center(node_i);
    const Eigen::Vector3f& c2 = center(node_j);
    const Eigen::Vector3f& n1 = normal(node_i);
    const Eigen::Vector3f& n2 = normal(node_j);
    
    Eigen::Vector3f delta = c2 - c1;
    Eigen::Vector3f edge_dir = delta.normalized();
    
    // Cost components
    edge.distance = delta.norm();
    edge.normal_change = 1.0f - std::abs(n1.dot(n2));
    edge.alignment_cost = compute_alignment_cost(node_i, node_j, edge_dir);
    edge.step_height = std::max(std::abs(delta.dot(n1)), std::abs(delta.dot(n2)));
    
    // Structural classification
    float anis_i = anisotropy(node_i);
    float anis_j = anisotropy(node_j);
    float align_i = std::abs(edge_dir.dot(major_axis(node_i)));
    float align_j = std::abs(edge_dir.dot(major_axis(node_j)));
    
    edge.is_structural = 
        (anis_i > params_.anisotropy_threshold && align_i > params_.alignment_threshold) ||
        (anis_j > params_.anisotropy_threshold && align_j > params_.alignment_threshold);
    
    // Total cost
    edge.total_cost = params_.w_distance * edge.distance +
                      params_.w_normal_change * edge.normal_change +
                      params_.w_alignment * edge.alignment_cost +
                      params_.w_step * edge.step_height;
    
    if (edge.is_structural) {
        edge.total_cost -= params_.structural_bonus;
        edge.total_cost = std::max(0.01f, edge.total_cost);
    }
    
    return edge.total_cost;
}

float SurfaceGraph::compute_alignment_cost(size_t node_i, size_t node_j, const Eigen::Vector3f& edge_dir) const {
    float cost = 0.0f;
    
    float anis_i = anisotropy(node_i);
    if (anis_i > 0.3f) {
        float align = std::abs(edge_dir.dot(major_axis(node_i)));
        cost += (1.0f - align) * anis_i;
    }
    
    float anis_j = anisotropy(node_j);
    if (anis_j > 0.3f) {
        float align = std::abs(edge_dir.dot(major_axis(node_j)));
        cost += (1.0f - align) * anis_j;
    }
    
    return cost * 0.5f;
}

// SPATIAL INDEX
VoxelKey SurfaceGraph::point_to_voxel(const Eigen::Vector3f& p) const {
    float inv = 1.0f / voxel_size_;
    return VoxelKey{
        static_cast<int32_t>(std::floor(p.x() * inv)),
        static_cast<int32_t>(std::floor(p.y() * inv)),
        static_cast<int32_t>(std::floor(p.z() * inv))
    };
}

void SurfaceGraph::add_to_spatial_index(size_t node_idx) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;
    auto key = point_to_voxel(center(node_idx));
    spatial_index_[key].push_back(node_idx);
}

void SurfaceGraph::remove_from_spatial_index(size_t node_idx) {
    if (node_idx >= nodes_.size()) return;
    auto key = point_to_voxel(center(node_idx));
    auto it = spatial_index_.find(key);
    if (it != spatial_index_.end()) {
        auto& vec = it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), node_idx), vec.end());
        if (vec.empty()) spatial_index_.erase(it);
    }
}

void SurfaceGraph::rebuild_spatial_index() {
    spatial_index_.clear();
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].is_valid) {
            add_to_spatial_index(i);
        }
    }
}

// GRAPH QUERIES
std::optional<size_t> SurfaceGraph::find_node_by_surfel(size_t surfel_idx) const {
    auto it = surfel_to_node_.find(surfel_idx);
    if (it != surfel_to_node_.end() && nodes_[it->second].is_valid) {
        return it->second;
    }
    return std::nullopt;
}

size_t SurfaceGraph::find_nearest_node(const Eigen::Vector3f& point) const {
    if (valid_node_count_ == 0) return 0;
    
    auto center_key = point_to_voxel(point);
    float best_dist_sq = std::numeric_limits<float>::max();
    size_t best_idx = 0;
    
    for (int r = 0; r <= 5; ++r) {
        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dz = -r; dz <= r; ++dz) {
                    if (r > 0 && std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) {
                        continue;  // Only check shell
                    }
                    
                    VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
                    auto it = spatial_index_.find(key);
                    if (it == spatial_index_.end()) continue;
                    
                    for (size_t idx : it->second) {
                        if (!nodes_[idx].is_valid) continue;
                        float dist_sq = (center(idx) - point).squaredNorm();
                        if (dist_sq < best_dist_sq) {
                            best_dist_sq = dist_sq;
                            best_idx = idx;
                        }
                    }
                }
            }
        }
        if (best_dist_sq < std::numeric_limits<float>::max()) break;
    }
    
    return best_idx;
}

std::vector<size_t> SurfaceGraph::find_nodes_in_radius(const Eigen::Vector3f& point, float rad) const {
    std::vector<size_t> result;
    
    auto center_key = point_to_voxel(point);
    int search_r = static_cast<int>(std::ceil(rad / voxel_size_)) + 1;
    float rad_sq = rad * rad;
    
    for (int dx = -search_r; dx <= search_r; ++dx) {
        for (int dy = -search_r; dy <= search_r; ++dy) {
            for (int dz = -search_r; dz <= search_r; ++dz) {
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
                auto it = spatial_index_.find(key);
                if (it == spatial_index_.end()) continue;
                
                for (size_t idx : it->second) {
                    if (!nodes_[idx].is_valid) continue;
                    if ((center(idx) - point).squaredNorm() <= rad_sq) {
                        result.push_back(idx);
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<size_t> SurfaceGraph::get_neighbors(size_t node_idx) const {
    std::vector<size_t> neighbors;
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return neighbors;
    
    for (size_t edge_idx : nodes_[node_idx].edge_indices) {
        if (edge_idx >= edges_.size() || !edges_[edge_idx].is_valid) continue;
        const GraphEdge& e = edges_[edge_idx];
        size_t other = (e.from_node == node_idx) ? e.to_node : e.from_node;
        if (nodes_[other].is_valid) {
            neighbors.push_back(other);
        }
    }
    
    return neighbors;
}

// PATH PLANNING
PathResult SurfaceGraph::find_path(size_t from_node, size_t to_node) const {
    PathResult result;
    result.valid = false;
    
    if (from_node >= nodes_.size() || to_node >= nodes_.size()) return result;
    if (!nodes_[from_node].is_valid || !nodes_[to_node].is_valid) return result;
    
    if (from_node == to_node) {
        result.node_indices = {from_node};
        result.total_cost = 0.0f;
        result.valid = true;
        return result;
    }
    
    // A* search
    struct AStarEntry {
        size_t node;
        float g, f;
        bool operator>(const AStarEntry& o) const { return f > o.f; }
    };
    
    std::priority_queue<AStarEntry, std::vector<AStarEntry>, std::greater<AStarEntry>> open;
    std::unordered_map<size_t, float> g_cost;
    std::unordered_map<size_t, size_t> came_from;
    
    const Eigen::Vector3f& goal_pos = center(to_node);
    auto heuristic = [&](size_t n) { return (center(n) - goal_pos).norm(); };
    
    open.push({from_node, 0.0f, heuristic(from_node)});
    g_cost[from_node] = 0.0f;
    
    while (!open.empty()) {
        auto [current, g, f] = open.top();
        open.pop();
        
        if (current == to_node) {
            // Reconstruct path
            result.valid = true;
            result.total_cost = g;
            size_t n = to_node;
            while (n != from_node) {
                result.node_indices.push_back(n);
                n = came_from[n];
            }
            result.node_indices.push_back(from_node);
            std::reverse(result.node_indices.begin(), result.node_indices.end());
            return result;
        }
        
        if (g > g_cost[current]) continue;
        
        for (size_t edge_idx : nodes_[current].edge_indices) {
            if (edge_idx >= edges_.size() || !edges_[edge_idx].is_valid) continue;
            const GraphEdge& edge = edges_[edge_idx];
            
            size_t neighbor = (edge.from_node == current) ? edge.to_node : edge.from_node;
            if (!nodes_[neighbor].is_valid) continue;
            
            float tentative_g = g + edge.total_cost;
            auto it = g_cost.find(neighbor);
            if (it == g_cost.end() || tentative_g < it->second) {
                g_cost[neighbor] = tentative_g;
                came_from[neighbor] = current;
                open.push({neighbor, tentative_g, tentative_g + heuristic(neighbor)});
            }
        }
    }
    
    return result;  // No path found
}

// INSPECTION STATE
void SurfaceGraph::mark_inspected(size_t node_idx, float quality, uint64_t timestamp) {
    if (node_idx >= nodes_.size() || !nodes_[node_idx].is_valid) return;
    GraphNode& node = nodes_[node_idx];
    node.inspected = true;
    node.inspection_quality = std::max(node.inspection_quality, quality);
    node.last_inspection_stamp = timestamp;
}

void SurfaceGraph::reset_inspection_state() {
    for (auto& node : nodes_) {
        if (node.is_valid) {
            node.inspected = false;
            node.inspection_quality = 0.0f;
            node.last_inspection_stamp = 0;
        }
    }
}

std::vector<size_t> SurfaceGraph::get_uninspected_nodes() const {
    std::vector<size_t> result;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].is_valid && !nodes_[i].inspected) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<size_t> SurfaceGraph::get_frontier_nodes() const {
    std::vector<size_t> result;
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].is_valid || nodes_[i].inspected) continue;
        
        bool has_inspected_neighbor = false;
        for (size_t edge_idx : nodes_[i].edge_indices) {
            if (edge_idx >= edges_.size() || !edges_[edge_idx].is_valid) continue;
            const GraphEdge& e = edges_[edge_idx];
            size_t other = (e.from_node == i) ? e.to_node : e.from_node;
            if (nodes_[other].is_valid && nodes_[other].inspected) {
                has_inspected_neighbor = true;
                break;
            }
        }
        
        if (has_inspected_neighbor) {
            result.push_back(i);
        }
    }
    
    return result;
}

CoverageStats SurfaceGraph::get_coverage_stats() const {
    CoverageStats stats;
    
    float total_importance = 0.0f;
    float inspected_importance = 0.0f;
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (!nodes_[i].is_valid) continue;
        
        stats.total_nodes++;
        float imp = importance(i);
        float a = area(i);
        total_importance += imp;
        stats.total_surface_area += a;
        
        if (nodes_[i].inspected) {
            stats.inspected_nodes++;
            inspected_importance += imp;
            stats.inspected_surface_area += a * nodes_[i].inspection_quality;
        }
    }
    
    if (stats.total_nodes > 0) {
        stats.coverage_ratio = static_cast<float>(stats.inspected_nodes) /
                               static_cast<float>(stats.total_nodes);
    }
    if (total_importance > 0.0f) {
        stats.weighted_coverage = inspected_importance / total_importance;
    }
    
    return stats;
}

} // namespace surface_inspection_planning