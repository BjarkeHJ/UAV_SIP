#include "sparse_surfel_mapping/planner/viewpoint_generator.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace sparse_surfel_map {

ViewpointGenerator::ViewpointGenerator() {}

void ViewpointGenerator::select_exploration_goal(const Eigen::Vector3f& current_position) {
    if (!map_ || !coverage_tracker_) return;

    VoxelKeySet map_frontiers = coverage_tracker_->map_frontiers();
    if (map_frontiers.empty()) return; // should decide on a coverage direction if possible?

    // cluster frontiers
    size_t cluster_id = 0;
    std::vector<bool> visited(map_frontiers.size(), false);
    for (size_t i = 0; i < map_frontiers.size(); ++i) {
        if (visited.at(i)) continue; // already clustered
        
    }

}



} // namespace

