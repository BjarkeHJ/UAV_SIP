#ifndef INSPECTION_PLANNER_HPP_
#define INSPECTION_PLANNER_HPP_

#include <deque>
#include <optional>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/viewpoint_generator.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"
#include "sparse_surfel_mapping/planner/rrt.hpp"

namespace sparse_surfel_map {

class InspectionPlanner {
public:
    enum class PlannerState {
        IDLE,
        PLANNING,
        EXECUTING,
        COMPLETE,
        FAILED,
        EMERGENCY_STOP,
    };

    InspectionPlanner();
    explicit InspectionPlanner(const InspectionPlannerConfig& config);

    // initialize planner with map
    void initialize(SurfelMap* map);
    void update_pose(const Eigen::Vector3f& position, float yaw); // update drone state
    
    bool plan();
    bool validate_viewpoints();

    void mark_target_reached(); // mark front viewpoint visited
    bool is_complete();

    const Viewpoint* next_target() const;
    const std::deque<Viewpoint>& viewpoints() const { return viewpoints_; }
    size_t remaining_count() const { return viewpoints_.size(); }
    bool has_plan() const { return !viewpoints_.empty(); }

    RRTPath generate_path(); // return planned path with position/yaw (rrt if necessary)
    int get_viewpoint_index_for_path_index(size_t path_index); // return -1 if index is rrt intermediate point

    PlannerState get_planner_state() const { return planner_state_; }
    const PlanningStatistics& statistics() const { return stats_; }
    const CoverageTracker& coverage() const { return coverage_tracker_; }

    void clear_emergency(); // allow replan again
    void request_replan() { needs_replan_ = true; }
    void reset();
    
private:
    void order_viewpoints();
    float compute_travel_cost(const ViewpointState& target) const;
    void two_opt_optimize();
    void update_statistics();
    float interpolate_yaw(float start_yaw, float end_yaw, float t) const;
    
    InspectionPlannerConfig config_;
    CoverageTracker coverage_tracker_;
    ViewpointGenerator viewpoint_generator_;
    RRTPlanner rrt_planner_;
    SurfelMap* map_{nullptr};

    // Current state
    Eigen::Vector3f current_position_{Eigen::Vector3f::Zero()};
    float current_yaw_{0.0f};
        
    // Planning state
    PlannerState planner_state_{PlannerState::IDLE};
    std::deque<Viewpoint> viewpoints_;
    bool needs_replan_{true};

    // cached executable path
    mutable RRTPath cached_path_;
    mutable bool path_cache_valid_{false};
    
    std::vector<ViewpointState> visited_viewpoints_; // tracking visited viewpoints
    mutable PlanningStatistics stats_;
};

} // namespace

#endif