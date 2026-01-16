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
        std::cout << "  Commit horizon: " << config_.commit_horizon << " viewpoints" << std::endl;
        std::cout << "  Min horizon buffer: " << config_.min_horizon_buffer << " viewpoints" << std::endl;
    }
}

void InspectionPlanner::initialize_components() {
    collision_checker_.set_map(map_);
    viewpoint_generator_.set_map(map_);
    viewpoint_generator_.set_coverage_tracker(&coverage_tracker_);
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

size_t InspectionPlanner::commit_index() const {
    return std::min(config_.commit_horizon, planned_viewpoints_.size()); // viewpoints with index >= commit_index can be replanned
}

size_t InspectionPlanner::committed_count() const {
    return commit_index();
}

size_t InspectionPlanner::uncommitted_count() const {
    size_t ci = committed_count();
    return (planned_viewpoints_.size() > ci) ? (planned_viewpoints_.size() - ci) : 0; 
}

bool InspectionPlanner::needs_extension() const {
    return uncommitted_count() < config_.min_horizon_buffer;
}

void InspectionPlanner::update_viewpoint_statuses() {
    size_t ci = commit_index();
    for (size_t i = 0; i < planned_viewpoints_.size(); ++i) {
        if (i < ci) {
            planned_viewpoints_[i].set_status(ViewpointStatus::COMMITED);
        }
        else {
            planned_viewpoints_[i].set_status(ViewpointStatus::PLANNED);
        }
    }
}

PathEvaluationResult InspectionPlanner::evaluate_path() const {
    PathEvaluationResult result;
    stats_.path_evaluations++;

    if (!map_) {
        result.status = PathSafetyStatus::NO_MAP;
        return result;
    }

    if (current_path_.waypoints.size() < 2) {
        result.status = PathSafetyStatus::INVALID_PATH;
        return result;
    }

    // TODO NEED TO EVALUATE PATH BETWEEN ALL VIEWPOINTS
    for (size_t i = 0; i < current_path_.waypoints.size() - 1; ++i) {
        // Check all viewpoint for collision (before RRT local planning)
    }

    result.status = PathSafetyStatus::SAFE;
    return result;
}

InspectionPlanner::PlannerState InspectionPlanner::evaluate_and_react() {
    PathEvaluationResult eval = evaluate_path(); 

    switch (eval.status) {
        case PathSafetyStatus::SAFE:
            planner_state_ = PlannerState::EXECUTING;
            if (needs_extension()) {
                extend_plan();
            }
            break;
        
        case PathSafetyStatus::COLLISION_COMMITED:
            planner_state_ = PlannerState::EMERGENCY_STOP;
            if (config_.debug_output) {
                std::cout << "[InspectionPlanner] !!! EMERGENCY STOP !!!" << std::endl;
                std::cout << "  Collision detected in COMMITTED path segment " 
                          << eval.collision_segment << std::endl;
                std::cout << "  Collision point: (" << eval.collision_point.transpose() << ")" << std::endl;
                std::cout << "  Distance to collision: " << eval.collision_distance << "m" << std::endl;
            }
            break;
        
        case PathSafetyStatus::COLLISION_UNCOMMITED:
            if (config_.debug_output) {
                std::cout << "[InspectionPlanner] Collision in uncommitted segment " 
                          << eval.collision_segment << " - regenerating" << std::endl;
            }
            regenerate_from_commit_horizon();
            break;
        
        case PathSafetyStatus::INVALID_PATH:
        case PathSafetyStatus::NO_MAP:
            needs_full_replan_ = true;
            break;
    }

    return planner_state_;
}

void InspectionPlanner::clear_emergency() {
    if (planner_state_ == PlannerState::EMERGENCY_STOP) {
        planner_state_ = PlannerState::IDLE;
        needs_full_replan_ = true;
        planned_viewpoints_.clear();
        current_path_.clear();

        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] Emergency cleared - will replan" << std::endl;
        }
    }
}

bool InspectionPlanner::plan() {
    auto t_start = std::chrono::high_resolution_clock::now();

    needs_full_replan_ = false;
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

    // Generate viewpoint chain
    std::vector<Viewpoint> chain;

    if (seed_viewpoint_.has_value()) {
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] Generating continuation from VP " 
                      << seed_viewpoint_->id() << std::endl;
        }
        chain = viewpoint_generator_.generate_continuation(*seed_viewpoint_);
    } 
    else {
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] Generating initial chain from ("
                      << current_position_.transpose() << ")" << std::endl;
        }
        chain = viewpoint_generator_.generate_next_viewpoints(current_position_, current_yaw_);
    }

    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] ViewpointGenerator returned " 
                  << chain.size() << " viewpoints" << std::endl;
    }

    if (chain.empty()) {
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] No viewpoints generated" << std::endl;
        }
        planner_state_ = PlannerState::IDLE;
        return false;
    }

    // Validate paths and build final chain
    planned_viewpoints_.clear();
    for (auto& vp : chain) {
        if (!vp.is_in_collision(*map_, config_.viewpoint.min_view_distance)) {
            planned_viewpoints_.push_back(std::move(vp));
        } else {
            if (config_.debug_output) {
                std::cout << "[InspectionPlanner] VP " << vp.id() << " unreachable!" << std::endl;
            }
            stats_.viewpoints_rejected++;
        }
    }

    if (planned_viewpoints_.empty()) {
        if (config_.debug_output) {
            std::cout << "[InspectionPlanner] No reachable viewpoints" << std::endl;
        }
        planner_state_ = PlannerState::FAILED;
        return false;
    }

    update_viewpoint_statuses();
    update_path();

    stats_.viewpoints_planned = planned_viewpoints_.size();

    auto t_end = std::chrono::high_resolution_clock::now();
    last_plan_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    stats_.total_planning_time_ms += last_plan_time_ms_;

    planner_state_ = PlannerState::EXECUTING;

    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Plan: " << planned_viewpoints_.size()
                  << " viewpoints in " << std::fixed << std::setprecision(2)
                  << last_plan_time_ms_ << " ms" << std::endl;
        
        std::cout << "Chain:" << std::endl;
        for (size_t i = 0; i < planned_viewpoints_.size(); ++i) {
            const auto& vp = planned_viewpoints_[i];
            float dist = (i == 0) 
                ? (vp.position() - current_position_).norm()
                : (vp.position() - planned_viewpoints_[i-1].position()).norm();
            
            std::cout << "  [" << i << "] VP " << vp.id() 
                      << " @ (" << vp.position().transpose() << ")"
                      << " dist=" << std::setprecision(2) << dist << "m"
                      << " visible=" << vp.num_visible() << std::endl;
        }
    }

    return true;
}

bool InspectionPlanner::extend_plan() {
    if (planner_state_ == PlannerState::EMERGENCY_STOP) return false;

    auto t_start = std::chrono::high_resolution_clock::now();
    planner_state_ = PlannerState::EXTENDING;
    stats_.path_extensions++;

    if (!map_) {
        planner_state_ = PlannerState::FAILED;
        return false;
    }

    if (!needs_extension() && uncommitted_count() > 0) {
        planner_state_ = PlannerState::EXECUTING;
        return true; // no extension needed
    }

    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Extending plan (committed=" << committed_count()
                  << ", uncommitted=" << uncommitted_count() << ")" << std::endl;
    }

    // Determine the seed position for extension
    // We extend from the LAST COMMITTED viewpoint (or current position if none)
    Eigen::Vector3f extension_seed_pos;
    float extension_seed_yaw;
    const Viewpoint* seed_vp = nullptr;
    
    size_t ci = commit_index();
    if (ci > 0 && !planned_viewpoints_.empty()) {
        // Extend from the last committed viewpoint
        seed_vp = &planned_viewpoints_[ci - 1];
        extension_seed_pos = seed_vp->position();
        extension_seed_yaw = seed_vp->yaw();
        
        if (config_.debug_output) {
            std::cout << "  Extension seed: VP " << seed_vp->id() 
                      << " (last committed)" << std::endl;
        }
    } else if (!planned_viewpoints_.empty()) {
        // No committed viewpoints yet, extend from first planned
        seed_vp = &planned_viewpoints_.front();
        extension_seed_pos = seed_vp->position();
        extension_seed_yaw = seed_vp->yaw();
    } else {
        // No viewpoints at all, need full replan
        needs_full_replan_ = true;
        planner_state_ = PlannerState::IDLE;
        return false;
    }
    
    // Remove uncommitted viewpoints (they will be regenerated)
    while (planned_viewpoints_.size() > ci) {
        planned_viewpoints_.pop_back();
    }
    
    // Generate new viewpoints from the extension seed
    std::vector<Viewpoint> extension;
    if (seed_vp) {
        extension = viewpoint_generator_.generate_continuation(*seed_vp);
    } else {
        extension = viewpoint_generator_.generate_next_viewpoints(extension_seed_pos, extension_seed_yaw);
    }
    
    if (extension.empty()) {
        if (config_.debug_output) {
            std::cout << "  No extension viewpoints generated" << std::endl;
        }
        update_viewpoint_statuses();
        update_path();
        planner_state_ = PlannerState::EXECUTING;
        return planned_viewpoints_.size() > 0;
    }
    
    // Add extension viewpoints (validate collision-free)
    Eigen::Vector3f path_start = planned_viewpoints_.empty() 
        ? current_position_ 
        : planned_viewpoints_.back().position();
    
    size_t added = 0;
    for (auto& vp : extension) {
        if (planned_viewpoints_.size() >= config_.max_viewpoints_per_plan) {
            break;
        }
        
        // REMOVE
        bool path_valid = collision_checker_.is_path_collision_free(path_start, vp.position());
        path_valid = true;

        if (path_valid) {
            planned_viewpoints_.push_back(std::move(vp));
            path_start = planned_viewpoints_.back().position();
            added++;
        } else {
            if (config_.debug_output) {
                std::cout << "  Extension VP " << vp.id() << " blocked - stopping" << std::endl;
            }
            break;
        }
    }
    
    update_viewpoint_statuses();
    update_path();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double extension_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    if (config_.debug_output) {
        std::cout << "  Extended by " << added << " viewpoints in " 
                  << std::fixed << std::setprecision(2) << extension_time << " ms" << std::endl;
        std::cout << "  Total: " << planned_viewpoints_.size() << " (" 
                  << committed_count() << " committed, " 
                  << uncommitted_count() << " uncommitted)" << std::endl;
    }
    
    planner_state_ = PlannerState::EXECUTING;
    return true;
}

bool InspectionPlanner::regenerate_from_commit_horizon() {
    // This is called when there's a collision in the uncommitted segment
    // We remove uncommitted viewpoints and regenerate
    
    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Regenerating from commit horizon" << std::endl;
    }
    
    return extend_plan();  // extend_plan already handles removing uncommitted and regenerating
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
        std::cout << "[InspectionPlanner] === VP " << reached.id() << " REACHED ===" << std::endl;
        std::cout << "  Position: (" << reached.position().transpose() << ")" << std::endl;
        std::cout << "  Observed: " << reached.num_visible() << " surfels" << std::endl;
        std::cout << "  Total visited: " << total_viewpoints_visited_ << std::endl;
        std::cout << "  Coverage: " << std::fixed << std::setprecision(1) 
                  << coverage_tracker_.coverage_ratio() * 100.0f << "%" << std::endl;
    }

    // Store for potential continuation
    seed_viewpoint_ = reached;

    // Remove the reached viewpoint
    planned_viewpoints_.pop_front();
    
    // Update statuses after removal
    update_viewpoint_statuses();

    if (!planned_viewpoints_.empty()) {
        update_path();
        
        if (config_.debug_output) {
            std::cout << "  Next target: VP " << planned_viewpoints_.front().id() 
                      << " (COMMITTED)" << std::endl;
            std::cout << "  Remaining: " << planned_viewpoints_.size() 
                      << " (" << committed_count() << " committed)" << std::endl;
        }
        
        // Check if we need to extend
        if (needs_extension()) {
            if (config_.debug_output) {
                std::cout << "  Horizon buffer low - triggering extension" << std::endl;
            }
        }
    } else {
        current_path_.clear();
        needs_full_replan_ = true;
        
        if (config_.debug_output) {
            std::cout << "  Chain exhausted - need full replan" << std::endl;
        }
    }

    update_statistics();
}

void InspectionPlanner::update_path() {
    if (planned_viewpoints_.empty()) {
        current_path_.clear();
        return;
    }

    current_path_.clear();
    current_path_.waypoints.push_back(current_position_);
    current_path_.yaw_angles.push_back(current_yaw_);
    
    for (const auto& vp : planned_viewpoints_) {
        current_path_.waypoints.push_back(vp.position());
        current_path_.yaw_angles.push_back(vp.yaw());
    }

    current_path_.compute_length();
    current_path_.is_valid = true;
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

void InspectionPlanner::update_statistics() {
    if (!map_) return;

    stats_.total_surfles = map_->num_valid_surfels();
    stats_.covered_surfels = coverage_tracker_.num_observed();
    stats_.coverage_ratio = coverage_tracker_.coverage_ratio();
    stats_.viewpoints_visited = total_viewpoints_visited_;
    
    if (total_viewpoints_visited_ > 0) {
        stats_.average_coverage_per_viewpoint = 
            static_cast<float>(stats_.covered_surfels) / total_viewpoints_visited_;
    }
}

void InspectionPlanner::reset() {
    coverage_tracker_.reset();
    planned_viewpoints_.clear();
    current_path_.clear();
    
    seed_viewpoint_.reset();
    needs_full_replan_ = true;
    
    map_surfels_at_last_plan_ = 0;
    total_viewpoints_visited_ = 0;
    last_plan_time_ms_ = 0.0;
    
    stats_ = PlanningStatistics();
    planner_state_ = PlannerState::IDLE;
    
    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Reset complete" << std::endl;
    }
}


} // namespace