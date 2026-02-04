#ifndef VIEWPOINT_GENERATOR_HPP_
#define VIEWPOINT_GENERATOR_HPP_

#include <queue>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

class ViewpointGenerator {
public:
    ViewpointGenerator();
    
    void set_map(const SurfelMap* map) { map_ = map; }
    void set_coverage_tracker(const CoverageTracker* ct) { coverage_tracker_ = ct; }

    // Choose frontier (if feasible) to inspect towards
    void select_exploration_goal(const Eigen::Vector3f& current_position);
    void generate_coverage_viewpoints();


private:


    const SurfelMap* map_{nullptr};
    const CoverageTracker* coverage_tracker_{nullptr};

};

} //

#endif