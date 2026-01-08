#ifndef SURFEL_FUSE_HPP_
#define SURFEL_FUSE_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <vector>
#include <deque>
#include <unordered_set>
#include <memory>
#include <chrono>

#include "surfel_mapping/surfel_map.hpp"

namespace surface_inspection_planning {

class SurfelFusion {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Params {
        // Point filtering
        float min_normal_quality = 0.1f;

        // Fusion weights
        float base_point_weight = 1.0f;

        // Update parameters
        float center_update_rate = 0.3f;
        float normal_update_rate = 0.1f;
        float covariance_update_rate = 0.2f;

        float max_confidence = 1.0f;

        // New surfel creation
        uint32_t min_points_for_new_surfel = 5;
        float new_surfel_initial_radius = 0.01f;
        float new_surfel_coherence_thresh = 0.3f; // clustering distance (accumulator process)

        // Point accumulator management
        size_t max_accumulator_size = 5000;
        uint32_t accumulator_process_interval = 5; // process every x frames
        
        ConfidenceParams confidence; // params for surfel fusion confidence tracking (in surfel.hpp)
    };

    struct FusionStats {
        size_t points_processed = 0;
        size_t points_associated = 0;
        size_t points_accumulated = 0;
        size_t points_rejected = 0;
        size_t surfels_updated = 0;
        size_t surfels_created = 0;
        size_t surfels_merged = 0;
        double processing_time_ms = 0.0;
    };

    SurfelFusion();
    explicit SurfelFusion(const Params& p, const SurfelMap::Params& map_p);

    const Params& params() const { return params_; }
    void set_params(const Params& p) {params_ = p; }

    // Map
    const SurfelMap& map() const { return map_; }
    SurfelMap& map_mutable() { return map_; }

    // Stats
    const FusionStats& last_stats() const { return last_stats_; }

    void process_scan(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const pcl::PointCloud<pcl::Normal>::Ptr& normals, 
        const Eigen::Isometry3f& pose, uint64_t 
        timestamp = 0
    );
    
    void flush_accumulator(uint64_t timestamp = 0);
    void reset();

private:
    struct AccumulatedPoint {
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
        uint64_t timestamp;
    };

    void fuse_point_to_surfel(size_t surfel_idx, const Eigen::Vector3f& point, const Eigen::Vector3f& normal, uint64_t timestamp);
    void accumulate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, uint64_t timestamp);
    void process_accumulator(uint64_t timestamp);

    Params params_;
    SurfelMap map_;

    std::deque<AccumulatedPoint> point_accumulator_;
    uint64_t frame_count_;
    FusionStats last_stats_;
};

}; // end namespace

#endif