#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

CoverageTracker::CoverageTracker() : config_(), stats_() {}
CoverageTracker::CoverageTracker(const InspectionPlannerConfig& config)
    : config_(config)
    , stats_()
{}

void CoverageTracker::mark_observed(const VoxelKeySet& voxels, uint64_t viewpoint_id) {
    (void)viewpoint_id; // could track viewpoint observed

    for (const auto& key : voxels) {
        observed_voxels_.insert(key);
        observation_counts_[key]++;
    }

    stats_.covered_surfels = observed_voxels_.size();
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
    return observed_voxels_.count(key) > 0;
}

size_t CoverageTracker::get_observation_count(const VoxelKey& key) const {
    auto it = observation_counts_.find(key);
    return (it != observation_counts_.end()) ? it->second : 0;
}

void CoverageTracker::update_statistics(size_t total_surfels) {
    stats_.total_surlfes = total_surfels;
    stats_.covered_surfels = observed_voxels_.size();
    stats_.viewpoints_visited = visited_viewpoints_.size();

    if (total_surfels > 0) {
        stats_.coverage_ratio = static_cast<float>(stats_.covered_surfels) / static_cast<float>(total_surfels);
    }
    else {
        stats_.coverage_ratio = 0.0f;
    }

    if (stats_.viewpoints_visited > 0) {
        stats_.average_coverage_per_viewpoint = static_cast<float>(stats_.covered_surfels) / stats_.viewpoints_visited;
    }
}

void CoverageTracker::reset() {
    observed_voxels_.clear();
    observation_counts_.clear();
    visited_viewpoints_.clear();
    stats_ = PlanningStatistics();
}

std::vector<ViewpointState> CoverageTracker::find_nearest_visited(const Eigen::Vector3f& position, size_t n) const {
    if (visited_viewpoints_.empty() || n == 0) return {};

    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(visited_viewpoints_.size());

    for (size_t i = 0; i < visited_viewpoints_.size(); ++i) {
        float dist = (visited_viewpoints_[i].position - position).norm();
        distances.emplace_back(dist, i);
    }

    const size_t k = std::min(n, distances.size());
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    std::vector<ViewpointState> result;
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        result.push_back(visited_viewpoints_[distances[i].second]);
    }

    return result;
}


float CoverageTracker::get_local_coverage_ratio(const Eigen::Vector3f& center, float radius, const VoxelKeySet& local_voxels) const {
    if (local_voxels.empty()) return 0.0f;

    size_t total_in_region = 0;
    size_t observed_in_region = 0;

    const float radius_sq = radius * radius;

    for (const auto& key : local_voxels) {
        total_in_region++;
        if (observed_voxels_.count(key) > 0) {
            observed_in_region++;
        }
    }

    return static_cast<float>(observed_in_region) / total_in_region;
}






} // namespace