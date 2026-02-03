#ifndef COVERAGE_TRACKER_HPP_
#define COVERAGE_TRACKER_HPP_

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"

namespace sparse_surfel_map {

class CoverageTracker {
public:
    CoverageTracker();
    explicit CoverageTracker(const InspectionPlannerConfig& config);

    void set_map(const SurfelMap* map) { map_ = map; }

    // Core coverage tracking
    void mark_observed(const VoxelKeySet& voxels, uint64_t viewpoint_id);
    void record_visited_viewpoint(const Viewpoint& viewpoint);
    bool is_observed(const VoxelKey& key) const;
    size_t get_observation_count(const VoxelKey& key) const;

    // Coverage statistics
    const VoxelKeySet& observed_surfels() const { return observed_surfels_; }
    size_t num_observed() const { return observed_surfels_.size(); }
    
    // Coverage frontiers
    const VoxelKeySet& coverage_frontiers() const { return coverage_frontiers_; }
    size_t num_frontiers() const { return coverage_frontiers_.size(); }
    
    // Map frontiers (query)
    VoxelKeySet map_frontiers() const;
    
    // Static helper for frontier updates (also used by viewpoint generator for speculative planning...)
    static void update_frontier_set(VoxelKeySet& frontier_set, const VoxelKeySet& newly_covered, const VoxelKeySet& total_coverage, const SurfelMap& map);

    // Viewpoint history
    bool is_viewpoint_visited(const Viewpoint& viewpoint) const;
    const std::vector<ViewpointState>& visited_viewpoints() const { return visited_viewpoints_; }    
    size_t num_visited_viewpoints() const { return visited_viewpoints_.size(); }
    const std::unordered_map<VoxelKey, size_t, VoxelKeyHash>& observations_counts() const { return observation_counts_; }
        
    // Statistics
    void update_statistics(size_t total_surfels);    
    const PlanningStatistics& stats() const { return stats_; }
    PlanningStatistics& stats() { return stats_; }
    float coverage_ratio() const { return stats_.coverage_ratio; }
    float get_local_coverage_ratio(const VoxelKeySet& local_voxels) const;
    
    void reset();

private:
    void compute_full_frontier_set();
    bool is_map_frontier_surfel(const VoxelKey& key) const;

    InspectionPlannerConfig config_;
    const SurfelMap* map_{nullptr};

    VoxelKeySet coverage_frontiers_; // frontiers in surfel space
    VoxelKeySet observed_surfels_;
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> observation_counts_;
    std::vector<ViewpointState> visited_viewpoints_;
    
    PlanningStatistics stats_;
};

} // namespace

#endif