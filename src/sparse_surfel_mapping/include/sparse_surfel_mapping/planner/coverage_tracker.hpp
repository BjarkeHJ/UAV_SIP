#ifndef COVERAGE_TRACKER_HPP_
#define COVERAGE_TRACKER_HPP_

#include <unordered_set>
#include <deque>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"

namespace sparse_surfel_map {

class CoverageTracker {
public:
    CoverageTracker();
    explicit CoverageTracker(const CameraConfig& config);

private:
    CameraConfig camera_config_;
    VoxelKeySet observed_voxels_;
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> observation_counts_;

    std::vector<ViewpointState> visited_viewpoints_;
    
};



} // namespace

#endif