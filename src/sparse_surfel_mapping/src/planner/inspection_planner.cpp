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
    , planner_state_(PlannerState::IDLE)
{}

InspectionPlanner::InspectionPlanner(const InspectionPlannerConfig& config)
    : config_(config)
    , coverage_tracker_(config)
    , viewpoint_generator_(config)
    , rrt_planner_(config.rrt)
    , planner_state_(PlannerState::IDLE)
{}

void InspectionPlanner::initialize(SurfelMap* map) {
    map_ = map;
    viewpoint_generator_.set_map(map_);
    viewpoint_generator_.set_coverage_tracker(&coverage_tracker_);
    rrt_planner_.set_map(map_);
    rrt_planner_.set_collision_radius(config_.collision.inflation_radius());
    planner_state_ = PlannerState::IDLE;
}

void InspectionPlanner::update_pose(const Eigen::Vector3f& position, float yaw) {
    current_position_ = position;
    current_yaw_ = yaw;
}

bool InspectionPlanner::validate_viewpoints() {
    if (!map_) return false;

    const float collision_radius = config_.collision.inflation_radius();
    const float voxel_size = map_->voxel_size();

    // Check all viewpoints in plan for collision risk
    bool modified = false;
    auto it = viewpoints_.begin();
    while (it != viewpoints_.end()) {
        if (!it->is_in_collision(*map_, collision_radius)) {
            ++it;
            continue; // not in collision risk
        }

        // In collision -> Try to repair by moving agains view direction (Intuitively away from structure)
        bool repaired = false;
        for (float d = voxel_size; d <= collision_radius; d += voxel_size) {
            Eigen::Vector3f new_pos = it->position() - it->state().forward_direction() * d;
            Viewpoint test_vp(new_pos, it->yaw(), config_.camera);
            
            if (!test_vp.is_in_collision(*map_, collision_radius)) {
                it->set_position(new_pos);
                repaired = true;
                modified = true;
                break;
            }
        }

        if (repaired) {
            ++it; // continue if repaired
        }
        else {
            it = viewpoints_.erase(it);
        }
    }

    if (modified) path_cache_valid_ = false;

    return !viewpoints_.empty();
}

bool InspectionPlanner::plan() {
    if (!map_ || planner_state_ == PlannerState::EMERGENCY_STOP) return false;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    planner_state_ = PlannerState::PLANNING;
    coverage_tracker_.update_statistics(map_->num_valid_surfels());

    // Compute the current planned viewpoint visibility (only what viewpoint in plan sees)
    VoxelKeySet already_observed;
    size_t desired_new = config_.max_viewpoints_in_plan;
    if (!viewpoints_.empty()) {
        for (auto& vp : viewpoints_) {
            vp.compute_visibility(*map_, true); // Recompute visibility - map could have updated
            for (const auto& key : vp.visible_voxels()) {
                already_observed.insert(key);
            }
        }

        desired_new -= viewpoints_.size() - 1; // desired new viewpoints
    }

    if (desired_new <= 0) return false;

    std::vector<Viewpoint> new_vpts;
    // If the planner needs full replan: Scrap old and replan from current drone pose
    if (needs_replan_) {
        viewpoints_.clear();
        Viewpoint drone_vp(current_position_, current_yaw_, config_.camera);
        drone_vp.compute_visibility(*map_, true);
        new_vpts = viewpoint_generator_.generate_viewpoints(drone_vp, desired_new, already_observed);
        if (!new_vpts.empty()) needs_replan_ = false;
    }

    // Space in current plan? Bound the number of viewpoints to optimize
    else if (viewpoints_.size() < config_.max_viewpoints_in_plan) {
        Viewpoint& seed_vp = viewpoints_.back(); // generate from last in queue 
        new_vpts = viewpoint_generator_.generate_viewpoints(seed_vp, desired_new, already_observed);
    }

    // Planner failed
    if (new_vpts.empty()) {
        return false;
    }

    // Order current viewpoints    
    for (auto& vp : new_vpts) {
        std::cout << vp.position().transpose() << std::endl;
        viewpoints_.push_back(std::move(vp));
    }

    order_viewpoints();
    path_cache_valid_ = false;

    auto t_end = std::chrono::high_resolution_clock::now();
    stats_.total_planning_time_ms += std::chrono::duration<double, std::milli>(t_end - t_start).count();;

    return true;
}

void InspectionPlanner::order_viewpoints() {
    if (viewpoints_.size() < 2) return;

    // Order by travel cost: Yaw-diff + distance
    std::vector<Viewpoint> pool(viewpoints_.begin(), viewpoints_.end());
    std::sort(pool.begin(), pool.end(),
        [this](const Viewpoint& a, const Viewpoint& b) {
            float cost_a = compute_travel_cost(a.state());
            float cost_b = compute_travel_cost(b.state());
            return cost_a < cost_b;
        });
    
    viewpoints_.assign(pool.begin(), pool.end()); 
    two_opt_optimize();
}

void InspectionPlanner::two_opt_optimize() {
    return; // TODO
}

float InspectionPlanner::compute_travel_cost(const ViewpointState& target) const {
    float dist = (target.position - current_position_).norm();
    float yaw_diff = std::abs(target.yaw - current_yaw_);
    if (yaw_diff > M_PI) yaw_diff = 2.0f * M_PI - yaw_diff; // wrap

    return dist + yaw_diff; // maybe scale yaw diff down (*0.5f)
}

RRTPath InspectionPlanner::generate_path() {
    if (path_cache_valid_) {
        return cached_path_;
    }

    RRTPath path;
    if (viewpoints_.empty()) return path;

    Eigen::Vector3f prev_pos = current_position_;
    // float prev_yaw = current_yaw_;

    for (size_t vp_idx = 0; vp_idx < viewpoints_.size(); ++vp_idx) {
        const Viewpoint& vp = viewpoints_[vp_idx];

        auto rrt_path = rrt_planner_.plan(prev_pos, vp.position());

        if (rrt_path.empty()) {
            if (config_.debug_output) {
                std::cout << "[InspectionPlanner] Warning: RRT failed!" << std::endl;
            }
            rrt_path = {prev_pos, vp.position()};
        }

        // add rrt intermediate waypoints (skip first if not actually first - to aviud dupes)
        size_t start_idx = (vp_idx == 0) ? 0 : 1;
        for (size_t i = start_idx; i < rrt_path.size() - 1; ++i) {
            path.positions.push_back(rrt_path[i]);

            Eigen::Vector3f to_next = rrt_path[i + 1] - rrt_path[i];
            float waypoint_yaw = std::atan2(to_next.y(), to_next.x());
            path.yaws.push_back(waypoint_yaw);

            path.viewpoint_indices.push_back(-1); // not actual viewpoint
        }

        path.positions.push_back(vp.position());
        path.yaws.push_back(vp.yaw());
        path.viewpoint_indices.push_back(static_cast<int>(vp_idx));

        prev_pos = vp.position();
        // prev_yaw = vp.yaw();
    }

    cached_path_ = path;
    path_cache_valid_ = true;

    return cached_path_;
}

int InspectionPlanner::get_viewpoint_index_for_path_index(size_t path_index) {
    if (!path_cache_valid_) {
        generate_path();
    }
    
    if (path_index >= cached_path_.viewpoint_indices.size()) {
        return -1;
    }

    return cached_path_.viewpoint_indices[path_index];
}

float InspectionPlanner::interpolate_yaw(float start_yaw, float end_yaw, float t) const {
    float diff = end_yaw - start_yaw;
    while (diff > M_PI) diff -= 2.0f * M_PI;
    while (diff < -M_PI) diff += 2.0f * M_PI;
    return start_yaw + t * diff;
}

void InspectionPlanner::mark_target_reached() {
    if (viewpoints_.empty()) return;

    Viewpoint& reached = viewpoints_.front();
    reached.set_status(ViewpointStatus::VISITED);
    coverage_tracker_.record_visited_viewpoint(reached);
    stats_.viewpoints_visited++;
    
    ViewpointState reached_state = reached.state(); // copy
    visited_viewpoints_.push_back(reached_state); 
    
    // Remove the reached viewpoint
    viewpoints_.pop_front();

    if (viewpoints_.empty()) {
        needs_replan_ = true;
    }

    if (is_complete()) {
        planner_state_ = PlannerState::COMPLETE;
    }
    
    update_statistics();
}

const Viewpoint* InspectionPlanner::next_target() const {
    return viewpoints_.empty() ? nullptr : &viewpoints_.front();
}

bool InspectionPlanner::is_complete() {
    if (!map_) return false;
    coverage_tracker_.update_statistics(map_->num_valid_surfels());
    if (coverage_tracker_.coverage_ratio() >= config_.target_coverage_ratio) return true;
    return false;
}

void InspectionPlanner::update_statistics() {
    if (!map_) return;
    coverage_tracker_.update_statistics(map_->num_valid_surfels());
    stats_.total_surfles = map_->num_valid_surfels();
    stats_.covered_surfels = coverage_tracker_.num_observed();
    stats_.coverage_ratio = coverage_tracker_.coverage_ratio();
    stats_.viewpoints_planned = viewpoints_.size();
}

void InspectionPlanner::reset() {
    viewpoints_.clear();
    coverage_tracker_.reset();
    stats_ = PlanningStatistics();
    planner_state_ = PlannerState::IDLE;
    needs_replan_ = true;
    
    if (config_.debug_output) {
        std::cout << "[InspectionPlanner] Reset complete" << std::endl;
    }
}


} // namespace