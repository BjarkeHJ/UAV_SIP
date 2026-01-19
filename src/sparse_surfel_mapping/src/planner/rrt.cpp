#include "sparse_surfel_mapping/planner/rrt.hpp"
#include <algorithm>
#include <cmath>

namespace sparse_surfel_map {

RRTPlanner::RRTPlanner() : config_() {}
RRTPlanner::RRTPlanner(const RRTConfig& config) : config_(config) {}

std::vector<Eigen::Vector3f> RRTPlanner::plan(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) {
    if (!map_) return {};

    if (is_straight_path_free(start, goal)) {
        return {start, goal};
    }

    // define the random sample space
    Eigen::Vector3f min_bound = start.cwiseMin(goal).array() - config_.sample_margin;
    Eigen::Vector3f max_bound = start.cwiseMax(goal).array() + config_.sample_margin;
    min_bound.z() = std::max(min_bound.z(), 0.5f); // keep above ground

    // Initialize tree
    std::vector<Node> tree;
    tree.reserve(config_.max_iterations);
    tree.push_back({start, -1});

    std::uniform_real_distribution<float> goal_dist(0.0f, 1.0f);
    int goal_node = -1;

    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // sample random point with goal bias
        Eigen::Vector3f sample_point;
        if (goal_dist(rng_) < config_.goal_bias) {
            sample_point = goal;
        }
        else {
            sample_point = sample(min_bound, max_bound);
        }

        // find nearest node and steer toward sample
        int nearest_idx = find_nearest(tree, sample_point);
        Eigen::Vector3f new_pos = steer(tree[nearest_idx].position, sample_point);

        // check if new pos is valid
        if (!is_collison_free(new_pos)) continue;
        if (!is_edge_free(tree[nearest_idx].position, new_pos)) continue;

        // add to tree
        int new_idx = static_cast<int>(tree.size());
        tree.push_back({new_pos, nearest_idx});

        // check if goal is reached
        float dist_to_goal = (new_pos - goal).norm();
        if (dist_to_goal < config_.goal_threshold) {
            // try direct connect
            if (is_edge_free(new_pos, goal)) {
                tree.push_back({goal, new_idx});
                goal_node = static_cast<int>(tree.size()) - 1;
                break;
            }
        }
    }

    if (goal_node < 0) return {}; // failed to find path
    
    // extract, simplify, and return
    auto path = extract_path(tree, goal_node);
    return simplify_path(path);
}

bool RRTPlanner::is_straight_path_free(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const {
    return is_edge_free(start, goal);
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

bool RRTPlanner::is_collison_free(const Eigen::Vector3f& point) const {
    if (!map_) return false;

    const auto& voxels = map_->voxels();
    const auto& neighbors = voxels.get_neighbors_in_radius(point, collision_radius_);
    return neighbors.empty();
}

bool RRTPlanner::is_edge_free(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const {
    if (!map_) return false;

    Eigen::Vector3f direction = to - from;
    float length = direction.norm();
    if (length < 1e-6f) return is_collison_free(from);

    direction /= length;
    int num_checks = static_cast<int>(std::ceil(length / config_.collision_check_step));

    for (int i = 0; i <= num_checks; ++i) {
        float t = static_cast<float>(i) / num_checks;
        Eigen::Vector3f point = from + direction * (t * length);
        if (is_collison_free(point)) return false;
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
    Eigen::Vector3f direction = to - from;
    float dist = direction.norm();
    if (dist < config_.step_size) {
         return to;
    }
    return from + direction.normalized() * config_.step_size;
}

std::vector<Eigen::Vector3f> RRTPlanner::extract_path(const std::vector<Node>& tree, int goal_idx) const {
    std::vector<Eigen::Vector3f> path;
    int current = goal_idx;
    while (current >= 0) {
        path.push_back(tree[current].position);
        current = tree[current].parent;
    }

    std::reverse(path.begin(), path.end());
    return path;
}



} // namespace