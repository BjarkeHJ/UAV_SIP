#ifndef VIEWPOINT_GENERATOR_HPP_
#define VIEWPOINT_GENERATOR_HPP_

#include <queue>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

class ViewpointGenerator {
public:
    ViewpointGenerator();
    explicit ViewpointGenerator(const InspectionPlannerConfig& config);

    void set_map(const SurfelMap* map) { map_ = map; }
    void set_coverage_tracker(const CoverageTracker* ct) { coverage_tracker_ = ct; }

    // Generates n_new (if possible) viewpoints (continuation) from starting viewpoint with a prior observation assumption (optional)
    std::vector<Viewpoint> generate_viewpoints(const Viewpoint& start_viewpoint, size_t n_new, const VoxelKeySet& current_obs={});

    std::vector<Viewpoint> candidates() const { return cands_; }
    
private:
    std::optional<Viewpoint> generate_initial_viewpoint(const Eigen::Vector3f& drone_position);
    std::vector<Viewpoint> build_chain(
        const VoxelKeySet& initial_coverage, 
        const VoxelKeySet& seed_visible, 
        const Eigen::Vector3f& seed_position, 
        float seed_yaw, 
        size_t n_new
    );
    
    std::vector<VoxelKeySet> cluster_frontiers(const VoxelKeySet& frontiers_keys) const; // region growing frontier clustering
    Eigen::Vector3f compute_cluster_centroid(const VoxelKeySet& cluster) const;
    Eigen::Vector3f compute_cluster_mean_normal(const VoxelKeySet& cluster) const;
    float compute_cluster_priority(const VoxelKeySet& cluster, const Eigen::Vector3f& drone_pos, float drone_yaw) const;

    Viewpoint generate_viewpoint_for_cluster(const VoxelKeySet& cluster);

    Viewpoint* select_best_for_chain(
        std::vector<Viewpoint>& candidates, 
        const VoxelKeySet& cumulative_coverage, 
        const Eigen::Vector3f& previous_position, 
        const std::vector<Viewpoint>& existing_chain
    );   

    // scoring and validation
    bool is_viewpoint_valid(const Viewpoint& vp) const;
    void score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const VoxelKeySet& target_cluster);

    // utils
    float compute_yaw_to_target(const Eigen::Vector3f& from_pos, const Eigen::Vector3f& target_pos) const;
    float angular_distance(float yaw1, float yaw2) const;
    uint64_t generate_id() { return next_viewpoint_id_++; }

    InspectionPlannerConfig config_;
    const SurfelMap* map_{nullptr};
    const CoverageTracker* coverage_tracker_{nullptr};
    uint64_t next_viewpoint_id_{0};


    // test / debug
    std::vector<Viewpoint> cands_;
    bool is_init_{true};

    // debug statistics
    mutable size_t last_frontiers_found_;
    mutable size_t last_clusters_formed_;
    mutable size_t last_candidates_generated_;
    mutable size_t last_candidates_in_collision_;
    mutable double last_generation_time_ms_{0.0};
};

} //

#endif