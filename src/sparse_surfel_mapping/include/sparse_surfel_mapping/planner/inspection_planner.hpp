#ifndef INSPECTION_PLANNER_HPP_
#define INSPECTION_PLANNER_HPP_

#include <deque>
#include <optional>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/viewpoint_generator.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"
#include "sparse_surfel_mapping/planner/collision_checker.hpp"

namespace sparse_surfel_map {

class InspectionPlanner {
public:
    enum class PlannerState {
        IDLE,
        PLANNING,
        EXECUTING,
        REPLANNING,
        COMPLETE,
        FAILED,
    };

    InspectionPlanner();
    explicit InspectionPlanner(const InspectionPlannerConfig& config);

    // initialize planner with map
    void initialize(SurfelMap* map);

    void update_state(const Eigen::Vector3f& position, float yaw); // update drone state
    void update_state(const ViewpointState& current_state);

    bool needs_replan() const;
    bool plan();

    void mark_target_reached();
    const Viewpoint& get_next_target() const;
    const InspectionPath& get_current_path() const { return current_path_; }

    const std::deque<Viewpoint>& get_planned_viewpoints() const { return planned_viewpoints_; }

    bool is_inspection_complete() const;

    const PlanningStatistics& statistics() const { return stats_; }

    void request_replan() { needs_replan_ = true; }
    void reset();

    std::string get_debug_info() const;

    const CoverageTracker& coverage_tracker() const { return coverage_tracker_; }
    const CollisionChecker& collision_checker() const { return collision_checker_; }
    const ViewpointGenerator& viewpoint_generator() const { return viewpoint_generator_; }

private:

    void initialize_components();
    bool has_map_changed_significantly() const;
    void update_statistics();

    InspectionPlannerConfig config_;

    CoverageTracker coverage_tracker_;
    CollisionChecker collision_checker_;
    ViewpointGenerator viewpoint_generator_;
    
    SurfelMap* map_{nullptr};

    // Current state
    Eigen::Vector3f current_position_{Eigen::Vector3f::Zero()};
    float current_yaw_{0.0f};
    PlannerState planner_state_{PlannerState::IDLE};

    // Planning state
    std::deque<Viewpoint> planned_viewpoints_;
    InspectionPath current_path_;
    bool needs_replan_{true};

    Viewpoint invalid_viewpoint_; // return when no plan

    // detect map changes
    size_t map_surfels_at_last_plan_{0};

    double last_plan_time_ms_{0.0};

    size_t total_viewpoints_visited_{0};
    mutable PlanningStatistics stats_;
};

} // namespace

#endif