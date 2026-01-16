#ifndef VIEWPOINT_GENERATOR_HPP_
#define VIEWPOINT_GENERATOR_HPP_

#include <random>
#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/frontier_finder.hpp"
#include "sparse_surfel_mapping/planner/collision_checker.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

class ViewpointGenerator {
public:
    ViewpointGenerator();
    explicit ViewpointGenerator(const InspectionPlannerConfig& config);

    void set_map(const SurfelMap* map);
    void set_collision_checker(const CollisionChecker* cc) { collision_checker_ = cc; }
    void set_coverage_tracker(const CoverageTracker* ct) { coverage_tracker_ = ct; }

    std::vector<Viewpoint> generate_next_viewpoints(const Eigen::Vector3f& position, float yaw);
    std::vector<Viewpoint> generate_continuation(const Viewpoint& start_viewpoint);

    size_t last_frontiers_found() const { return last_frontiers_found_; }
    size_t last_clusters_formed() const { return last_clusters_formed_; }
    size_t last_candidates_generated() const { return last_candidates_generated_; }
    size_t last_candidates_in_collision() const { return last_candidates_in_collision_; }
    double last_generation_time_ms() const { return last_generation_time_ms_; }

private:
    std::vector<Viewpoint> build_chain(const VoxelKeySet& initial_coverage, const VoxelKeySet& seed_visible, const Eigen::Vector3f& seed_position);
    Eigen::Vector3f compute_expansion_center(const VoxelKeySet& visible_voxels, const VoxelKeySet& already_covered) const;
    std::vector<Viewpoint> generate_candidates_for_clusters(const std::vector<FrontierCluster>& clusters, const VoxelKeySet& already_covered);
    Viewpoint generate_viewpoint_for_cluster(const FrontierCluster& cluster, const VoxelKeySet& already_covered);
    Viewpoint* select_best_for_chain(std::vector<Viewpoint>& candidates, const VoxelKeySet& cumulative_coverage, const Eigen::Vector3f& previous_position, const std::vector<Viewpoint>& existing_chain);
    
    bool is_position_valid(const Eigen::Vector3f& position) const;
    void score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const FrontierCluster& target_cluster);
    float compute_yaw_to_target(const Eigen::Vector3f& from_pos, const Eigen::Vector3f& target_pos) const;
    Eigen::Vector3f key_to_position(const VoxelKey& key) const;
    uint64_t generate_id() { return next_viewpoint_id_++; }

    InspectionPlannerConfig config_;
    const SurfelMap* map_{nullptr};
    const CollisionChecker* collision_checker_{nullptr};
    const CoverageTracker* coverage_tracker_{nullptr};
    FrontierFinder frontier_finder_;

    uint64_t next_viewpoint_id_{0};
    std::mt19937 rng_;

    // debug statistics
    mutable size_t last_frontiers_found_;
    mutable size_t last_clusters_formed_;
    mutable size_t last_candidates_generated_;
    mutable size_t last_candidates_in_collision_;
    mutable double last_generation_time_ms_{0.0};
};

} //

#endif