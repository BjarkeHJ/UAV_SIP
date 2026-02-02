#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

CoverageTracker::CoverageTracker() : config_(), stats_() {}
CoverageTracker::CoverageTracker(const InspectionPlannerConfig& config)
    : config_(config)
    , stats_()
{}

void CoverageTracker::mark_observed(const VoxelKeySet& voxels, uint64_t viewpoint_id) {
    (void)viewpoint_id; // could track viewpoint observed

    if (!map_ || voxels.empty()) return;

    bool is_first_observation = observed_surfels_.empty();

    // Update observed set
    for (const auto& key : voxels) {
        observed_surfels_.insert(key);
        observation_counts_[key]++;
    }

    // Update frontier set
    compute_full_frontier_set();
    // if (is_first_observation) compute_full_frontier_set(); // full if first
    // else update_frontier_set(frontier_surfels_, voxels, observed_surfels_, *map_); // incremental ...

    stats_.covered_surfels = observed_surfels_.size();
}

void CoverageTracker::record_visited_viewpoint(const Viewpoint& viewpoint) {
    ViewpointState state = viewpoint.state();
    state.status = ViewpointStatus::VISITED;
    state.timestamp_visited = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Log visited viewpoints
    visited_viewpoints_.push_back(state);
    stats_.viewpoints_visited = visited_viewpoints_.size();

    // Update observed surface
    mark_observed(viewpoint.visible_voxels(), viewpoint.id());
}

void CoverageTracker::compute_full_frontier_set() {
    frontier_surfels_.clear();
    if (!map_) return;

    for (const auto& covered_key : observed_surfels_) {

        // check nb-26 conn
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue; // skip self

                    VoxelKey nb{covered_key.x + dx, covered_key.y + dy, covered_key.z + dz};

                    if (observed_surfels_.count(nb) > 0) continue; // already covered
                    if (frontier_surfels_.count(nb) > 0) continue; // already frontier

                    // frontier found: uncovered with covered neighbor(s)
                    auto voxel_opt = map_->get_voxel(nb);
                    if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
                        frontier_surfels_.insert(nb);
                    }
                }
            }
        }
    }
}

void CoverageTracker::update_frontier_set(VoxelKeySet& frontier_set, const VoxelKeySet& newly_covered, const VoxelKeySet& total_coverage, const SurfelMap& map) {
    // Remove newly covered keys from frontier set (no longer frontiers as they are covered)
    for (const auto& key : newly_covered) {
        frontier_set.erase(key);
    }

    // Add frontiers adjacent to newly covered voxels (move the frontier-horizon)
    for (const auto& covered_key : newly_covered) {
        // check nb-26 conn
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue; // skip self

                    VoxelKey nb{covered_key.x + dx, covered_key.y + dy, covered_key.z + dz};

                    if (total_coverage.count(nb) > 0) continue; // already covered

                    // frontier found: uncovered with covered neighbor(s)
                    auto voxel_opt = map.get_voxel(nb);
                    if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
                        frontier_set.insert(nb);
                    }
                }
            }
        }
    }
}

bool CoverageTracker::is_viewpoint_visited(const Viewpoint& viewpoint) const {
    const float pos_th = config_.revisit_distance_th;
    const float angle_th_rad = config_.revist_angle_th_deg * M_PI / 180.0f;

    for (const auto& visited : visited_viewpoints_) {
        const float pos_dist = (viewpoint.position() - visited.position).norm();
        if (pos_dist > pos_th) continue;

        float yaw_diff = std::abs(viewpoint.yaw() - visited.yaw);
        if (yaw_diff > M_PI) {
            yaw_diff = 2.0f * M_PI - yaw_diff;
        }

        if (yaw_diff <= angle_th_rad) return true;
    }

    return false;
}

bool CoverageTracker::is_observed(const VoxelKey& key) const {
    return observed_surfels_.count(key) > 0;
}

size_t CoverageTracker::get_observation_count(const VoxelKey& key) const {
    auto it = observation_counts_.find(key);
    return (it != observation_counts_.end()) ? it->second : 0;
}

void CoverageTracker::update_statistics(size_t total_surfels) {
    stats_.total_surfles = total_surfels;
    stats_.covered_surfels = observed_surfels_.size();
    stats_.viewpoints_visited = visited_viewpoints_.size();

    if (total_surfels > 0) {
        stats_.coverage_ratio = static_cast<float>(stats_.covered_surfels) / static_cast<float>(total_surfels);
    }
    else {
        stats_.coverage_ratio = 0.0f;
    }
}

void CoverageTracker::reset() {
    observed_surfels_.clear();
    observation_counts_.clear();
    visited_viewpoints_.clear();
    stats_ = PlanningStatistics();
}

float CoverageTracker::get_local_coverage_ratio(const VoxelKeySet& local_voxels) const {
    if (local_voxels.empty()) return 0.0f;

    size_t total_in_region = 0;
    size_t observed_in_region = 0;

    for (const auto& key : local_voxels) {
        total_in_region++;
        if (observed_surfels_.count(key) > 0) {
            observed_in_region++;
        }
    }

    return static_cast<float>(observed_in_region) / total_in_region;
}


} // namespace