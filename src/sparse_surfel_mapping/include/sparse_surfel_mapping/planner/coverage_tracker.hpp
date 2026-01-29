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

    void mark_observed(const VoxelKeySet& voxels, uint64_t viewpoint_id);
    void record_visited_viewpoint(const Viewpoint& viewpoint);

    bool is_viewpoint_visited(const Viewpoint& viewpoint) const;
    bool is_observed(const VoxelKey& key) const;
    size_t get_observation_count(const VoxelKey& key) const;
    
    void update_statistics(size_t total_surfels);

    void reset();

    const VoxelKeySet& observed_surfels() const { return observed_surfels_; }
    size_t num_observed() const { return observed_surfels_.size(); }

    const VoxelKeySet& frontier_surfels() const { return frontier_surfels_; }
    size_t num_frontiers() const { return frontier_surfels_.size(); }

    const std::vector<ViewpointState>& visited_viewpoints() const { return visited_viewpoints_; }    
    size_t num_visited_viewpoints() const { return visited_viewpoints_.size(); }

    const std::unordered_map<VoxelKey, size_t, VoxelKeyHash>& observations_counts() const { return observation_counts_; }

    // Static helper for frontier updates (also used by viewpoint generator for speculative planning...)
    static void update_frontier_set(VoxelKeySet& frontier_set, const VoxelKeySet& newly_covered, const VoxelKeySet& total_coverage, const SurfelMap& map);

    // Statistics
    const PlanningStatistics& stats() const { return stats_; }
    PlanningStatistics& stats() { return stats_; }
    float coverage_ratio() const { return stats_.coverage_ratio; }
    
    float get_local_coverage_ratio(const VoxelKeySet& local_voxels) const;

private:
    void compute_full_frontier_set();

    InspectionPlannerConfig config_;
    const SurfelMap* map_{nullptr};

    VoxelKeySet frontier_surfels_;
    VoxelKeySet observed_surfels_;
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> observation_counts_;
    std::vector<ViewpointState> visited_viewpoints_;
    
    PlanningStatistics stats_;
};

} // namespace

#endif