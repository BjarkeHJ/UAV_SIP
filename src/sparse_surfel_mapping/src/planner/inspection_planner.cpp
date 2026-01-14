#include "sparse_surfel_mapping/planner/inspection_planner.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace sparse_surfel_map {

InspectionPlanner::InspectionPlanner()
    : config_()
    , coverage_tracker_()
    , collision_checker_()
    , viewpoint_generator_()
    , planner_state_(PlannerState::IDLE)
{}

InspectionPlanner::InspectionPlanner(const InspectionPlannerConfig& config)
    : config_(config)
    , coverage_tracker_(config)
    , collision_checker_(config.collision, nullptr)
    , viewpoint_generator_(config)
    , planner_state_(PlannerState::IDLE)
{}

void InspectionPlanner::initialize(SurfelMap* map) {
    map_ = map;
    initialize_components();
    planner_state_ = PlannerState::IDLE;

    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Initialized with map" << std::endl;
    }
}

void InspectionPlanner::initialize_components() {
    collision_checker_.set_map(map_);

    viewpoint_generator_.set_map(map_);
    viewpoint_generator_.set_collision_checker(&collision_checker_);
    viewpoint_generator_.set_coverage_tracker(&coverage_tracker_);

    // maybe ensure configs here as well, but i dont think i need to
}

void InspectionPlanner::update_state(const Eigen::Vector3f& position, float yaw) {
    current_position_ = position;
    current_yaw_ = yaw;

    // update coverage statistics
    if (map_) {
        coverage_tracker_.update_statistics(map_->num_valid_surfels());
    }
}

void InspectionPlanner::update_state(const ViewpointState& current_state) {
    update_state(current_state.position, current_state.yaw);
}

bool InspectionPlanner::needs_replan() const {
    if (needs_replan_) return true;
    if (planned_viewpoints_.empty()) return true;

    // Replan if close to current target
    const Viewpoint& target = planned_viewpoints_.front();
    float dist_to_target = (current_position_ - target.position()).norm();
    if (dist_to_target < config_.replan_distance_th) return true;

    // Map changed significantly?
    if (has_map_changed_significantly()) return true;

    return false;
}

bool InspectionPlanner::plan() {

    auto t_start = std::chrono::high_resolution_clock::now();

    needs_replan_ = false;
    planner_state_ = PlannerState::PLANNING;

    if (!map_) {
        planner_state_ = PlannerState::FAILED;
        return false;
    }

    coverage_tracker_.update_statistics(map_->num_valid_surfels());
    map_surfels_at_last_plan_ = map_->num_valid_surfels();

    if (is_inspection_complete()) {
        planner_state_ = PlannerState::COMPLETE;
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] Inspection Complete! Coverage: " 
                      << std::fixed << std::setprecision(1) 
                      << coverage_tracker_.coverage_ratio() * 100.0f << "%" << std::endl;
        }
        return true;
    }

    std::vector<Viewpoint> candidates = viewpoint_generator_.generate_from_seed(current_position_, current_yaw_);

    std::cout << "[DEBUG] Candidates size: " << candidates.size() << std::endl;

    if (candidates.empty()) {
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] No viewpoints generated" << std::endl;
            std::cout << "  Frontiers found: " << viewpoint_generator_.last_frontiers_found() << std::endl;
            std::cout << "  Clusters formed: " << viewpoint_generator_.last_clusters_formed() << std::endl;
            std::cout << "  Candidates: " << viewpoint_generator_.last_candidates_generated() << std::endl;
            std::cout << "  In collision: " << viewpoint_generator_.last_candidates_in_collision() << std::endl;
        }

        planner_state_ = PlannerState::IDLE;
        return false;
    }

    // clear old and plan new viewpoints
    planned_viewpoints_.clear();

    // For now: Simple straight-line path validation
    // TODO: Integrate A* path planner for proper planning (in grid)
    Eigen::Vector3f path_start = current_position_;
    for (auto& vp : candidates) {
        bool path_valid = collision_checker_.is_path_collision_free(path_start, vp.position());
        path_valid = true;

        if (path_valid) {
            vp.set_status(ViewpointStatus::PLANNED);
            planned_viewpoints_.push_back(std::move(vp));

            // Chain
            path_start = planned_viewpoints_.back().position();

            // limit 
            if (planned_viewpoints_.size() >= config_.max_viewpoints_per_plan) {
                break;
            }
        }
        else {
            if (config_.debug_output) {
                std::cout << "[InspectionPlanner] Viewpoint " << vp.id() << " unreachable (path blocked)" << std::endl;
            }
            stats_.viewpoints_rejected++;
        }
    }

    if (planned_viewpoints_.empty()) {
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] No reachable viewpoints found" << std::endl;
        }
        planner_state_ = PlannerState::FAILED;
        return false;
    }

    // Create simple path to first viewpoint
    current_path_.clear();
    current_path_.waypoints.push_back(current_position_);
    current_path_.waypoints.push_back(planned_viewpoints_.front().position());
    current_path_.yaw_angles.push_back(current_yaw_);
    current_path_.yaw_angles.push_back(planned_viewpoints_.front().yaw());
    current_path_.compute_length();
    current_path_.is_valid = true;

    stats_.viewpoints_planned = planned_viewpoints_.size();

    auto t_end = std::chrono::high_resolution_clock::now();
    last_plan_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats_.total_planning_time_ms = last_plan_time_ms_;

    planner_state_ = PlannerState::EXECUTING;

    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Planned " << planned_viewpoints_.size()
                  << " viewpoints in " << std::fixed << std::setprecision(2)
                  << last_plan_time_ms_ << " ms" << std::endl;
    }

    return true;
}

void InspectionPlanner::mark_target_reached() {
    if (planned_viewpoints_.empty()) return;

    Viewpoint& reached = planned_viewpoints_.front();
    reached.set_status(ViewpointStatus::VISITED);

    if (map_) {
        reached.compute_visibility(*map_, true);
    }

    coverage_tracker_.record_visited_viewpoint(reached);
    stats_.total_path_length += current_path_.total_length;

    total_viewpoints_visited_++;

    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Viewpoint w. ID: " << reached.id() << " reached." << std::endl;
        std::cout << "  Observed: " << reached.num_visible() << " surfels" << std::endl;
        std::cout << "  Coverage: " << std::fixed << std::setprecision(1) << coverage_tracker_.coverage_ratio() * 100.0f << "%" << std::endl;
    }

    planned_viewpoints_.pop_front();

    if (!planned_viewpoints_.empty()) {
        current_path_.clear();
        current_path_.waypoints.push_back(current_position_);
        current_path_.waypoints.push_back(planned_viewpoints_.front().position());
        current_path_.yaw_angles.push_back(current_yaw_);
        current_path_.yaw_angles.push_back(planned_viewpoints_.front().yaw());
        current_path_.compute_length();
        current_path_.is_valid = true;
    }
    else {
        current_path_.clear();
        needs_replan_ = true;
    }

    update_statistics();
}

const Viewpoint& InspectionPlanner::get_next_target() const {
    if (planned_viewpoints_.empty()) return invalid_viewpoint_;
    return planned_viewpoints_.front();
}

bool InspectionPlanner::is_inspection_complete() const {
    if (coverage_tracker_.coverage_ratio() >= config_.target_coverage_ratio) return true;
    if (total_viewpoints_visited_ >= config_.max_total_viewpoints) return true;
    return false;
}

bool InspectionPlanner::has_map_changed_significantly() const {
    if (!map_) return false;

    size_t current_count = map_->num_valid_surfels();
    if (map_surfels_at_last_plan_ == 0) return true;

    float change_ratio = static_cast<float>(std::abs(
        static_cast<int>(current_count) - static_cast<int>(map_surfels_at_last_plan_))) / static_cast<float>(map_surfels_at_last_plan_);

    return change_ratio > config_.replan_coverage_th;
}

void InspectionPlanner::update_statistics() {
    if (!map_) return;

    stats_.total_surlfes = map_->num_valid_surfels();
    stats_.covered_surfels = coverage_tracker_.num_observed();
    stats_.coverage_ratio = coverage_tracker_.coverage_ratio();
    stats_.viewpoints_visited = total_viewpoints_visited_;
    if (total_viewpoints_visited_ > 0) {
        stats_.average_coverage_per_viewpoint = static_cast<float>(stats_.covered_surfels) / total_viewpoints_visited_;
    }
}


} // namespace