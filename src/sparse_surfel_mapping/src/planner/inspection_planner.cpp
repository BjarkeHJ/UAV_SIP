#include "sparse_surfel_mapping/planner/inspection_planner.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace sparse_surfel_map {

InspectionPlanner::InspectionPlanner()
    : config_()
    , coverage_tracker_()
    , viewpoint_generator_()
    , rrt_planner_()
{}

InspectionPlanner::InspectionPlanner(const InspectionPlannerConfig& config)
    : config_(config)
    , coverage_tracker_(config)
    , viewpoint_generator_(config)
    , rrt_planner_(config.rrt)
{}

void InspectionPlanner::initialize(SurfelMap* map) {
    map_ = map;
    coverage_tracker_.set_map(map_);
    viewpoint_generator_.set_map(map_);
    viewpoint_generator_.set_coverage_tracker(&coverage_tracker_);
    rrt_planner_.set_map(map_);
    rrt_planner_.set_collision_radius(config_.collision.inflation_radius());
}

void InspectionPlanner::update_pose(const Eigen::Vector3f& position, float yaw) {
    // Update current direction based on movement
    Eigen::Vector3f delta = position - current_position_;
    if (delta.norm() > 0.01f) {
        current_direction_ = delta.normalized();
    }

    current_position_ = position;
    current_yaw_ = yaw;
}

bool InspectionPlanner::plan() {
    if (!map_) return false;

    // Update statistics
    coverage_tracker_.update_statistics(map_->num_valid_surfels());

    // Check if we need to replan
    if (!needs_replan_ && !planned_viewpoints_.empty()) {
        return false; // plan still valid
    }

    // Update exploration goal
    bool goal_found = viewpoint_generator_.update_exploration_goal(current_position_, current_direction_);

    if (!goal_found) {
        // No frontiers, exploration complete
        needs_replan_ = false;
        return false;
    }

    // Generate viewpoints toward goal
    auto new_viewpoints = viewpoint_generator_.generate_exploration_viewpoints(
        current_position_,
        config_.max_viewpoints_in_plan);

    // Convert to ViewpointState and store
    planned_viewpoints_.clear();
    candidate_viewpoints_.clear();

    for (auto& vp : new_viewpoints) {
        ViewpointState state = vp.state();
        state.status = ViewpointStatus::PLANNED;
        planned_viewpoints_.push_back(state);
        candidate_viewpoints_.push_back(state);
    }

    needs_replan_ = false;
    return !planned_viewpoints_.empty();
}

bool InspectionPlanner::is_complete() const {
    if (!map_) return false;

    const auto& stats = statistics();

    // Check coverage ratio
    // if (stats.coverage_ratio >= config_.target_coverage_ratio) {
    //     return true;
    // }

    // Check max viewpoints limit
    if (stats.viewpoints_visited >= config_.max_total_viewpoints) {
        return true;
    }

    // Check if no more frontiers
    if (coverage_tracker_.map_frontiers().empty()) {
        return true;
    }

    return false;
}

bool InspectionPlanner::validate_viewpoints() {
    if (!map_ || planned_viewpoints_.empty()) return false;

    // Check if first viewpoint is still collision-free
    const auto& first_vp_state = planned_viewpoints_.front();
    Viewpoint first_vp(first_vp_state.position, first_vp_state.yaw, config_.camera);

    if (first_vp.is_in_collision(*map_, config_.collision.inflation_radius())) {
        needs_replan_ = true;
        return false;
    }

    return true;
}

bool InspectionPlanner::validate_path() {
    // Stub: just call validate_viewpoints for now
    return validate_viewpoints();
}

RRTPath InspectionPlanner::generate_path() {
    // Generate path through planned viewpoints
    RRTPath path;

    if (planned_viewpoints_.empty()) return path;

    // Start from current position
    path.positions.push_back(current_position_);
    path.yaws.push_back(current_yaw_);
    path.viewpoint_indices.push_back(-1);  // not a viewpoint

    // Add each viewpoint position
    int vp_idx = 0;
    for (const auto& vp_state : planned_viewpoints_) {
        path.positions.push_back(vp_state.position);
        path.yaws.push_back(vp_state.yaw);
        path.viewpoint_indices.push_back(vp_idx++);
    }

    return path;
}

void InspectionPlanner::mark_target_reached() {
    if (!planned_viewpoints_.empty()) {
        // Mark first viewpoint as visited
        ViewpointState reached_vp = planned_viewpoints_.front();
        planned_viewpoints_.pop_front();

        // Create Viewpoint object for coverage tracking
        Viewpoint vp(reached_vp.position, reached_vp.yaw, config_.camera);
        vp.compute_visibility(*map_, true);

        // Record in coverage tracker
        coverage_tracker_.record_visited_viewpoint(vp);

        // If no more viewpoints, request replan
        if (planned_viewpoints_.empty()) {
            needs_replan_ = true;
        }
    }
}

int InspectionPlanner::get_viewpoint_index_for_path_index(size_t path_idx) const {
    // Simple mapping: path index 0 is current position, 1+ are viewpoints
    if (path_idx == 0) return -1;  // current position, not a viewpoint
    return static_cast<int>(path_idx - 1);
}

} // namespace