#ifndef INSPECTION_PLANNER_HPP_
#define INSPECTION_PLANNER_HPP_

#include <deque>
#include <optional>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

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
    void update_state(const ViewpointState& current_state);


    void reset();

private:

    SurfelMap* map_{nullptr};

    // Plan triggers
    bool needs_replan_{true};
    size_t map_surfels_at_last_plan_{0}; // new arrived?

    mutable PlanningStatistics stats_;
    size_t total_viewpoints_visited_{0};
};

} // namespace

#endif