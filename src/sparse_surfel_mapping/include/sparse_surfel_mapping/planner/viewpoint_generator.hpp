#ifndef VIEWPOINT_GENERATOR_HPP_
#define VIEWPOINT_GENERATOR_HPP_

#include <random>
#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class ViewpointGenerator {
public:
    ViewpointGenerator();
    explicit ViewpointGenerator(const InspectionPlannerConfig& config);

    void set_map(const SurfelMap* map);
    // void set_collsion_check(const CollisionChecker checker);
    // void set_coverage_tracker(const CoverageTracker tracker);
    

private:
 
    const SurfelMap* map_{nullptr};

    InspectionPlannerConfig config_;

  
    mutable double last_generation_time_ms_{0.0};
    mutable uint64_t next_viewpoint_id_{0};
};

} //

#endif