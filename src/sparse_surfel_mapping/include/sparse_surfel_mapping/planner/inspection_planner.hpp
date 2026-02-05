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
    InspectionPlanner();
    explicit InspectionPlanner(const InspectionPlannerConfig& config);

    void initialize(SurfelMap* map);
    void update_pose(const Eigen::Vector3f& position, float yaw);

    // Main planning interface
    bool plan();  // Returns true if new plan generated
    bool has_plan() const { return !planned_viewpoints_.empty(); }
    const std::deque<ViewpointState>& viewpoints() const { return planned_viewpoints_; }
    bool is_complete() const;

    // Replan management
    void request_replan() { needs_replan_ = true; }
    bool validate_viewpoints();  // Check if current plan is still valid
    bool validate_path();  // Check if current path is still valid

    // Path generation
    RRTPath generate_path();
    void mark_target_reached();
    int get_viewpoint_index_for_path_index(size_t path_idx) const;

    // Access to subcomponents
    const CoverageTracker& coverage() const { return coverage_tracker_; }
    CoverageTracker& coverage() { return coverage_tracker_; }
    const PlanningStatistics& statistics() const { return coverage_tracker_.stats(); }

    // Candidate viewpoints (for visualization)
    const std::vector<ViewpointState>& vpt_cand() const { return candidate_viewpoints_; }

private:
    InspectionPlannerConfig config_;
    CoverageTracker coverage_tracker_;
    ViewpointGenerator viewpoint_generator_;
    RRTPlanner rrt_planner_;
    SurfelMap* map_{nullptr};

    // Current state
    Eigen::Vector3f current_position_{Eigen::Vector3f::Zero()};
    float current_yaw_{0.0f};
    Eigen::Vector3f current_direction_{Eigen::Vector3f::UnitX()};

    // Planning state
    std::deque<ViewpointState> planned_viewpoints_;
    std::vector<ViewpointState> candidate_viewpoints_;
    bool needs_replan_{true};

};

} // namespace

#endif