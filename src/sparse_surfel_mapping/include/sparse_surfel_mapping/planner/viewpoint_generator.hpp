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
    
    std::vector<Viewpoint> generate_from_seed(const Eigen::Vector3f& seed_position, float seed_yaw);
    std::vector<Viewpoint> generate_from_viewpoint(const Viewpoint& from_viewpoint);

    std::vector<Viewpoint> grow_step(const VoxelKeySet& current_coverage, const Eigen::Vector3f& search_center, const VoxelKeySet& already_covered);

    std::vector<StructuralFeature> analyze_structure(const VoxelKeySet& visible_voxels);
    std::vector<Viewpoint> generate_for_features(const std::vector<StructuralFeature>& features, const VoxelKeySet& current_coverage);

    size_t last_frontiers_found() const { return last_frontiers_found_; }
    size_t last_clusters_formed() const { return last_clusters_formed_; }
    size_t last_candidates_generated() const { return last_candidates_generated_; }
    size_t last_candidates_in_collision() const { return last_candidates_in_collision_; }
    double last_generation_time_ms() const { return last_generation_time_ms_; }
private:
 
    Viewpoint generate_viewpoint_for_cluster(const FrontierCluster& cluster, const VoxelKeySet& already_covered);

    bool find_valid_view_position(const FrontierCluster& cluster, Eigen::Vector3f& out_pos, float& out_yaw);
    bool is_position_valid(const Eigen::Vector3f& position) const;

    void score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const FrontierCluster& target_cluster);
    
    std::vector<Viewpoint> select_best_viewpoints(std::vector<Viewpoint>& candidates, size_t max_count, const VoxelKeySet& already_covered);

    float compute_yaw_to_target(const Eigen::Vector3f& from_pos, const Eigen::Vector3f& target_pos) const;

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