#include "sparse_surfel_mapping/planner/a_star.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <chrono>
#include <iostream>

namespace sparse_surfel_map {

bool AStarPlanner::is_valid_surface_adjacent_node(const VoxelKey& coarse_key) const {
    if (!map_) return false;

    const auto& voxels = map_->voxels();

    // Must be FREE (not occupied, not unknown)
    if (voxels.coarse_cell_state(coarse_key) != SpatialHash::CoarseCellState::FREE) {
        return false;
    }

    // Must have at least 1 OCCUPIED neighbor (adjacent to surface)
    for (const auto& offset : neighbor_offsets_6) {
        VoxelKey neighbor{
            coarse_key.x + offset[0],
            coarse_key.y + offset[1],
            coarse_key.z + offset[2]
        };

        if (voxels.coarse_cell_state(neighbor) == SpatialHash::CoarseCellState::OCCUPIED) {
            return true;  // Adjacent to surface!
        }
    }

    return false;  // Not adjacent to any surface
}

std::vector<VoxelKey> AStarPlanner::get_valid_neighbors(const VoxelKey& coarse_key) const {
    std::vector<VoxelKey> valid_neighbors;
    valid_neighbors.reserve(6);

    for (const auto& offset : neighbor_offsets_6) {
        VoxelKey neighbor{
            coarse_key.x + offset[0],
            coarse_key.y + offset[1],
            coarse_key.z + offset[2]
        };

        if (is_valid_surface_adjacent_node(neighbor)) {
            valid_neighbors.push_back(neighbor);
        }
    }

    return valid_neighbors;
}

float AStarPlanner::heuristic(const VoxelKey& from, const VoxelKey& to) const {
    // Euclidean distance in coarse grid space
    int32_t dx = to.x - from.x;
    int32_t dy = to.y - from.y;
    int32_t dz = to.z - from.z;
    return std::sqrt(static_cast<float>(dx*dx + dy*dy + dz*dz));
}

Eigen::Vector3f AStarPlanner::coarse_key_to_position(const VoxelKey& coarse_key) const {
    const auto& voxels = map_->voxels();
    const float coarse_size = voxels.voxel_size() * SpatialHash::COARSE_FACTOR;

    // Return center of coarse cell
    return Eigen::Vector3f(
        (static_cast<float>(coarse_key.x) + 0.5f) * coarse_size,
        (static_cast<float>(coarse_key.y) + 0.5f) * coarse_size,
        (static_cast<float>(coarse_key.z) + 0.5f) * coarse_size
    );
}

std::vector<Eigen::Vector3f> AStarPlanner::plan(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) {
    if (!map_) return {};

    auto t_start = std::chrono::high_resolution_clock::now();

    const auto& voxels = map_->voxels();

    // Convert to coarse keys
    VoxelKey start_fine = voxels.point_to_key(start);
    VoxelKey goal_fine = voxels.point_to_key(goal);
    VoxelKey start_coarse = voxels.fine_to_coarse(start_fine);
    VoxelKey goal_coarse = voxels.fine_to_coarse(goal_fine);

    // Check if start and goal are at least FREE (allow non-surface-adjacent)
    if (voxels.coarse_cell_state(start_coarse) != SpatialHash::CoarseCellState::FREE) {
        std::cout << "[A*] Start not in FREE space" << std::endl;
        return {};
    }
    if (voxels.coarse_cell_state(goal_coarse) != SpatialHash::CoarseCellState::FREE) {
        std::cout << "[A*] Goal not in FREE space" << std::endl;
        return {};
    }

    // A* data structures
    struct Node {
        VoxelKey key;
        float f_score;

        bool operator>(const Node& other) const {
            return f_score > other.f_score;
        }
    };

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
    std::unordered_set<VoxelKey, VoxelKeyHash> in_open_set;
    std::unordered_set<VoxelKey, VoxelKeyHash> closed_set;
    std::unordered_map<VoxelKey, VoxelKey, VoxelKeyHash> came_from;
    std::unordered_map<VoxelKey, float, VoxelKeyHash> g_score;

    // Initialize
    g_score[start_coarse] = 0.0f;
    float h = heuristic(start_coarse, goal_coarse);
    open_set.push({start_coarse, h});
    in_open_set.insert(start_coarse);

    size_t nodes_expanded = 0;
    const size_t max_expansions = 10000;  // Safety limit

    // A* main loop
    while (!open_set.empty() && nodes_expanded < max_expansions) {
        // Get node with lowest f_score
        Node current = open_set.top();
        open_set.pop();
        in_open_set.erase(current.key);

        // Goal reached?
        if (current.key == goal_coarse) {
            // Reconstruct path
            std::vector<VoxelKey> coarse_path;
            VoxelKey curr = goal_coarse;

            while (came_from.find(curr) != came_from.end()) {
                coarse_path.push_back(curr);
                curr = came_from[curr];
            }
            coarse_path.push_back(start_coarse);
            std::reverse(coarse_path.begin(), coarse_path.end());

            // Convert to waypoints
            std::vector<Eigen::Vector3f> waypoints;
            waypoints.reserve(coarse_path.size());
            for (const auto& key : coarse_path) {
                waypoints.push_back(coarse_key_to_position(key));
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            std::cout << "[A*] Path found: " << waypoints.size() << " waypoints | "
                      << "Expanded: " << nodes_expanded << " nodes | "
                      << "Time: " << std::chrono::duration<double, std::milli>(t_end - t_start).count()
                      << " ms" << std::endl;

            return waypoints;
        }

        // Already processed?
        if (closed_set.count(current.key)) continue;
        closed_set.insert(current.key);
        nodes_expanded++;

        // Expand neighbors
        float current_g = g_score[current.key];

        // Get all FREE neighbors
        std::vector<VoxelKey> neighbors;
        for (const auto& offset : neighbor_offsets_6) {
            VoxelKey neighbor{
                current.key.x + offset[0],
                current.key.y + offset[1],
                current.key.z + offset[2]
            };

            // Allow any FREE neighbor
            if (voxels.coarse_cell_state(neighbor) != SpatialHash::CoarseCellState::FREE) {
                continue;
            }

            // For intermediate nodes (not start/goal), require surface-adjacent
            if (neighbor != start_coarse && neighbor != goal_coarse) {
                if (!is_valid_surface_adjacent_node(neighbor)) {
                    continue;
                }
            }

            neighbors.push_back(neighbor);
        }

        for (const auto& neighbor : neighbors) {
            if (closed_set.count(neighbor)) continue;

            // Compute tentative g_score (uniform edge cost = 1.0)
            float tentative_g = current_g + 1.0f;

            // Check if this path is better
            auto it = g_score.find(neighbor);
            if (it == g_score.end() || tentative_g < it->second) {
                // Better path found
                came_from[neighbor] = current.key;
                g_score[neighbor] = tentative_g;
                float f = tentative_g + heuristic(neighbor, goal_coarse);

                if (!in_open_set.count(neighbor)) {
                    open_set.push({neighbor, f});
                    in_open_set.insert(neighbor);
                }
            }
        }
    }

    // No path found
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "[A*] No path found | Expanded: " << nodes_expanded << " nodes | "
              << "Time: " << std::chrono::duration<double, std::milli>(t_end - t_start).count()
              << " ms" << std::endl;

    return {};
}

bool AStarPlanner::path_exists(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) {
    // Simple BFS to check connectivity (faster than full A*)
    if (!map_) return false;

    const auto& voxels = map_->voxels();
    VoxelKey start_coarse = voxels.fine_to_coarse(voxels.point_to_key(start));
    VoxelKey goal_coarse = voxels.fine_to_coarse(voxels.point_to_key(goal));

    if (!is_valid_surface_adjacent_node(start_coarse)) return false;
    if (!is_valid_surface_adjacent_node(goal_coarse)) return false;

    std::queue<VoxelKey> queue;
    std::unordered_set<VoxelKey, VoxelKeyHash> visited;

    queue.push(start_coarse);
    visited.insert(start_coarse);

    const size_t max_iterations = 5000;
    size_t iterations = 0;

    while (!queue.empty() && iterations < max_iterations) {
        VoxelKey current = queue.front();
        queue.pop();
        iterations++;

        if (current == goal_coarse) return true;

        auto neighbors = get_valid_neighbors(current);
        for (const auto& neighbor : neighbors) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }

    return false;
}

} // namespace
