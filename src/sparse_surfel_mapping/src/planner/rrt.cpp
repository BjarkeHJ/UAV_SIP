#include "sparse_surfel_mapping/planner/rrt.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>

namespace sparse_surfel_map {

RRTPlanner::RRTPlanner() : config_(), astar_planner_() {}
RRTPlanner::RRTPlanner(const RRTConfig& config) : config_(config), astar_planner_() {}

std::vector<Eigen::Vector3f> RRTPlanner::plan(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) {
    if (!map_) return {};

    // Use A* on coarse grid for surface-following paths
    astar_planner_.set_map(map_);

    auto path = astar_planner_.plan(start, goal);

    // Optionally simplify path (remove redundant waypoints)
    if (path.size() > 2) {
        path = simplify_path(path);
    }

    return path;
}

std::vector<Eigen::Vector3f> RRTPlanner::extract_path(const std::vector<Node>& tree, int goal_idx) const {
    std::vector<Eigen::Vector3f> path;
    int current = goal_idx; // from goal to root (-1)
    while (current >= 0) {
        path.push_back(tree[current].position);
        current = tree[current].parent;
    }

    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Eigen::Vector3f> RRTPlanner::simplify_path(const std::vector<Eigen::Vector3f>& path) const {
    if (path.size() <= 2) return path;

    std::vector<Eigen::Vector3f> simplified;
    simplified.push_back(path.front());

    size_t current = 0;
    while (current < path.size() - 1) {
        // find furthest visible point from current
        size_t furthest = current + 1;
        for (size_t i = current + 2; i < path.size(); ++i) {
            if (is_edge_free(path[current], path[i])) {
                furthest = i;
            }
        }

        simplified.push_back(path[furthest]);
        current = furthest;
    }

    return simplified;
}

bool RRTPlanner::is_straight_path_free(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const {
    return is_edge_free(start, goal);
}

int RRTPlanner::validate_path(const std::vector<Eigen::Vector3f>& path) const {
    if (path.size() < 2) return -1;

    for (size_t i = 0; i < path.size() - 1; ++i) {
        // check if edge is still valid between waypoints
        if (!is_edge_free(path[i], path[i+1])) {
            return static_cast<int>(i);
        }
    }

    return -1; // valid path
}

bool RRTPlanner::is_collision_free(const Eigen::Vector3f& point) const {
    if (!map_) return false;

    const auto& voxels = map_->voxels();
    const auto& neighbors = voxels.get_neighbors_in_radius(point, collision_radius_);
    return neighbors.empty();
}

bool RRTPlanner::is_edge_free(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const {
    if (!map_) return false;

    Eigen::Vector3f direction = to - from;
    float length = direction.norm();
    if (length < 1e-6f) return is_collision_free(from);

    direction /= length;
    int num_checks = static_cast<int>(std::ceil(length / config_.step_size));

    for (int i = 0; i <= num_checks; ++i) {
        float t = static_cast<float>(i) / num_checks;
        Eigen::Vector3f point = from + direction * (t * length);
        if (!is_collision_free(point)) return false;
    }

    return true;
}

Eigen::Vector3f RRTPlanner::sample(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound) {
    std::uniform_real_distribution<float> dist_x(min_bound.x(), max_bound.x());
    std::uniform_real_distribution<float> dist_y(min_bound.y(), max_bound.y());
    std::uniform_real_distribution<float> dist_z(min_bound.z(), max_bound.z());
    return Eigen::Vector3f(dist_x(rng_), dist_y(rng_), dist_z(rng_));
}

int RRTPlanner::find_nearest(const std::vector<Node>& tree, const Eigen::Vector3f& point) const {
    int nearest = 0;
    float min_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < tree.size(); ++i) {
        float dist = (tree[i].position - point).squaredNorm();
        if (dist < min_dist) {
            min_dist = dist;
            nearest = static_cast<int>(i);
        }
    }

    return nearest;
}

Eigen::Vector3f RRTPlanner::steer(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const {
    // Take step from -> to
    Eigen::Vector3f direction = to - from;
    float dist = direction.norm();
    if (dist < config_.step_size) {
         return to;
    }
    return from + direction.normalized() * config_.step_size;
}



} // namespace