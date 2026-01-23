#ifndef VIEWPOINT_CREATOR_HPP_
#define VIEWPOINT_CREATOR_HPP_

#include <vector>
#include <random>

#include "sparse_surfel_mapping/planner/surface_graph.hpp"
#include "sparse_surfel_mapping/planner/geodesic_potential_field.hpp"
#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include "sparse_surfel_mapping/planner/coverage_tracker.hpp"

namespace sparse_surfel_map {

class ViewpointCreator {
public:
    struct Params {
        // Camera configuration 
        CameraConfig cam_config;

        // View geometry
        float optimal_view_distance{2.0f};
        float min_view_distance{1.0f};
        float max_view_distance{3.0f};
        float min_altitude{0.5f};

        // Normal analysis
        float normal_variance_th{0.3f}; // below this -> region is planar
        size_t max_viewpoints_per_pool{3}; // for high-curvature region
        size_t normal_cluster_k_means_iter{10};

        // Selection
        size_t max_viewpoints_total{25}; // max vp per planning cycle

        // Priority weighting
        float priority_weight_coverage{0.50f}; // marginal coverage
        float priority_weight_potential{0.25f}; // pool potential
        float priority_weight_distance{0.15f}; // distance to current position
        float priority_weight_efficiency{0.10f}; // density in FOV

        // Collision
        float collision_check_radius{0.8f};
    };

    ViewpointCreator();
    
    void set_map(const SurfelMap* map) { map_ = map; }
    void set_coverage_tracker(const CoverageTracker* ct) { coverage_tracker_ = ct; }

    std::vector<Viewpoint> generate_viewpoints(const SurfaceGraph& graph, std::vector<FrontierPool>& pools, const Eigen::Vector3f& current_position);

private:
    std::vector<Viewpoint> generate_for_pool(const SurfaceGraph& graph, FrontierPool& pool);
    Viewpoint create_viewpoint(const Eigen::Vector3f& centroid, const Eigen::Vector3f& normal, float normal_variance, int pool_id);
    std::vector<std::vector<VoxelKey>> cluster_by_normal(const SurfaceGraph& graph, const std::vector<VoxelKey>& surfels, size_t max_clusters);
    bool is_valid(const Viewpoint& vp) const;
    
    std::vector<Viewpoint> select_best(std::vector<Viewpoint>& candidates, const std::vector<FrontierPool>& pools, const SurfaceGraph& graph, const Eigen::Vector3f& position);
    float compute_priority(const Viewpoint& vp, const FrontierPool& pool, const SurfaceGraph& graph, const Eigen::Vector3f& position);

    uint64_t generate_id() { return next_id_++; }

    const SurfelMap* map_{nullptr};
    const CoverageTracker* coverage_tracker_{nullptr};

    uint64_t next_id_{0};
    std::mt19937 rng_{std::random_device{}()};

    Params params_;
};


} // namespace


#endif 