#ifndef COVERAGE_TRACKER_HPP_
#define COVERAGE_TRACKER_HPP_

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"

namespace sparse_surfel_map {

class CoverageTracker {
public:
    struct Params {
        // todo
    };

    CoverageTracker();
    explicit CoverageTracker(const InspectionPlannerConfig& config);

    void mark_observed(const VoxelKeySet& voxels, uint64_t viewpoint_id);
    void record_visited_viewpoint(const Viewpoint& viewpoint);

    bool is_viewpoint_visited(const Viewpoint& viewpoint) const;
    bool is_observed(const VoxelKey& key) const;
    size_t get_observation_count(const VoxelKey& key) const;
    
    void update_statistics(size_t total_surfels);

    void reset();

    const VoxelKeySet& observed_voxels() const { return observed_voxels_; }
    const std::vector<ViewpointState>& visited_viewpoints() const { return visited_viewpoints_; }
    
    size_t num_observed() const { return observed_voxels_.size(); }
    size_t num_visited_viewpoints() const { return visited_viewpoints_.size(); }

    const PlanningStatistics& stats() const { return stats_; }
    PlanningStatistics& stats() { return stats_; }
    float coverage_ratio() const { return stats_.coverage_ratio; }
    
    float get_local_coverage_ratio(const VoxelKeySet& local_voxels) const;

private:
    InspectionPlannerConfig config_;

    VoxelKeySet observed_voxels_;
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> observation_counts_;
    
    std::vector<ViewpointState> visited_viewpoints_;
    
    PlanningStatistics stats_;
};

} // namespace

#endif