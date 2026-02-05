#ifndef VIEWPOINT_GENERATOR_HPP_
#define VIEWPOINT_GENERATOR_HPP_

#include <chrono>

#include <queue>
#include <deque>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

struct SurfelCluster {
    Eigen::Vector3f centroid{Eigen::Vector3f::Zero()};
    Eigen::Vector3f avg_normal{Eigen::Vector3f::Zero()};
    std::vector<VoxelKey> members;
    float total_score{0.0f};
};

class ViewpointGenerator {
public:
    ViewpointGenerator();
    explicit ViewpointGenerator(const InspectionPlannerConfig& config);

    void set_map(const SurfelMap* map) { map_ = map; }
    void set_coverage_tracker(const CoverageTracker* ct) { coverage_tracker_ = ct; }

    // Main interface
    bool update_exploration_goal(const Eigen::Vector3f& current_position, const Eigen::Vector3f& current_direction);
    std::deque<Viewpoint> generate_exploration_viewpoints(const Eigen::Vector3f& current_position, size_t max_viewpoints = 10);

    // Goal management
    bool has_valid_goal() const { return current_goal_.is_valid; }
    const ExplorationGoal& current_goal() const { return current_goal_; }
    bool is_goal_stable(float distance_threshold = 2.0f) const;

    // Path access
    const std::vector<Eigen::Vector3f>& path_to_goal() const { return path_to_goal_; }

private:
    // Frontier clustering
    std::vector<FrontierCluster> cluster_frontiers(const VoxelKeySet& frontiers, const Eigen::Vector3f& drone_pos);
    FrontierCluster select_best_cluster(const std::vector<FrontierCluster>& clusters, const Eigen::Vector3f& current_direction);

    // Sphere sampling
    std::vector<Eigen::Vector3f> sample_spheres_along_path(const std::vector<Eigen::Vector3f>& path, float sphere_radius, float overlap_ratio);

    // Viewpoint generation within sphere
    std::vector<Viewpoint> generate_viewpoints_in_sphere(const Eigen::Vector3f& sphere_center, size_t max_viewpoints);
 
    // Surfel clustering within sphere
    std::vector<SurfelCluster> cluster_surfels_in_sphere(const std::vector<std::pair<VoxelKey, float>>& scored_surfels, size_t target_clusters);
    std::vector<Viewpoint> generate_viewpoints_from_clusters(const std::vector<SurfelCluster>& clusters,
                                                              const Eigen::Vector3f& sphere_center,
                                                              size_t total_surfels_in_sphere);

    // Helpers
    bool adjust_viewpoint_if_obstructed(Viewpoint& vp, const Eigen::Vector3f& target_surfel_pos, const Eigen::Vector3f& target_surfel_normal);
    bool is_near_path(const Eigen::Vector3f& point, const std::vector<Eigen::Vector3f>& path, float tolerance) const;
    bool is_near_frontier(const VoxelKey& key) const;

    InspectionPlannerConfig config_;
    const SurfelMap* map_{nullptr};
    const CoverageTracker* coverage_tracker_{nullptr};

    // State
    ExplorationGoal current_goal_;
    std::vector<Eigen::Vector3f> path_to_goal_;
    float sphere_radius_;
};

} //

#endif