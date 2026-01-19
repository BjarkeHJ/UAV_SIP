#ifndef RRT_HPP_
#define RRT_HPP_

#include <vector>
#include <random>
#include <Eigen/Core>
#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class RRTPlanner {
public:
    RRTPlanner();
    explicit RRTPlanner(const RRTConfig& config);

    void set_map(const SurfelMap* map) { map_ = map; }
    void set_collision_radius(float radius) { collision_radius_ = radius; }

    std::vector<Eigen::Vector3f> plan(const Eigen::Vector3f& start, const Eigen::Vector3f& goal);
    bool is_straight_path_free(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const;

    std::vector<Eigen::Vector3f> simplify_path(const std::vector<Eigen::Vector3f>& path) const;

private:
    struct Node {
        Eigen::Vector3f position;
        int parent = -1;
    };

    bool is_collison_free(const Eigen::Vector3f& point) const;
    bool is_edge_free(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const;

    Eigen::Vector3f sample(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound);
    int find_nearest(const std::vector<Node>& tree, const Eigen::Vector3f& point) const;
    Eigen::Vector3f steer(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const;
    std::vector<Eigen::Vector3f> extract_path(const std::vector<Node>& tree, int goal_idx) const;

    RRTConfig config_;
    const SurfelMap* map_;
    float collision_radius_{0.7f};
    mutable std::mt19937 rng_{std::random_device{}()};
};

// Struct for RRT path
struct RRTPath {
    std::vector<Eigen::Vector3f> positions;
    std::vector<float> yaws;
    std::vector<int> viewpoint_indices; // index to original viewpoint list (-1 for rrt waypoints)

    bool empty() const { return positions.empty(); }
    size_t size() const { return positions.size(); }
};



} // namespace

#endif