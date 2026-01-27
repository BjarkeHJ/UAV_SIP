#ifndef GEODESIC_POTENTIAL_FIELD_HPP_
#define GEODESIC_POTENTIAL_FIELD_HPP_

#include <queue>
#include <functional>
#include "sparse_surfel_mapping/planner/surface_graph.hpp"

/*
This module updates the surface graph node information using the proposed geodesic field potential field 
for increasing coverage attraction. 
*/


namespace sparse_surfel_map {

struct DijkstraElement {
    VoxelKey key;
    float distance;
    size_t hops;
    bool operator>(const DijkstraElement& other) const {
        return distance > other.distance;
    }
};

struct FrontierPool {
    int id{-1};
    std::vector<VoxelKey> frontier_surfels;
    VoxelKey peak_surfel;
    float peak_potential{0.0f};

    Eigen::Vector3f centroid{Eigen::Vector3f::Zero()};
    Eigen::Vector3f mean_normal{Eigen::Vector3f::Zero()};
    float normal_variance{0.0f};
    float total_area_estimate{0.0f};

    size_t size() const { return frontier_surfels.size(); }
    bool empty() const { return frontier_surfels.empty(); }
    void compute_geometry(const SurfaceGraph& graph);
};

class GeodesicPotentialField {
public:
    // TODO: wire params into framework and set good default
    struct Params {
        // Geodesic distances (bounds)
        size_t source_horizon_hops{100};
        float source_horizon_distance{10.0f};
        float frontier_propagation_radius{10.0f};

        // Field Potential weights
        float alpha_source{1.0f};
        float beta_frontier{1.0f};
        float gamma_density{1.0f};
        float lambda{5.0f};
        float radius{25.0f};

        // Frontier pools
        size_t min_pool_size{1};
        size_t max_n_pools{50};
        size_t max_ascend_steps{100};
    };

    GeodesicPotentialField();

    // Geodesic distances
    void compute_distances_from_seed(SurfaceGraph& graph, const VoxelKey& seed); // compute geodesic distances from seed
    void compute_distances_to_frontiers(SurfaceGraph& graph); // compute geodesic distance to nearest frontier

    // Potential field
    void compute_potential_field(SurfaceGraph& graph, float voxel_size);

    // Frontier/Basin clustering
    std::vector<FrontierPool> detect_frontier_pools(SurfaceGraph& graph);

    // statistics
    size_t nodes_visited_source() const { return nodes_visited_source_; } // nubmer of nodes visisted in dijkstra from source
    size_t nodes_visited_frontier() const { return nodes_visited_frontier_; } // number of nodes visited in dijkstra from frontiers

    void set_params(const GeodesicPotentialField::Params& params) { params_ = params; }

    

private:
    using PriorityQueue = std::priority_queue<DijkstraElement, std::vector<DijkstraElement>, std::greater<DijkstraElement>>; // min-heap (keep smallest element on top of queue)

    // basin extraction
    VoxelKey ascend_to_attractor(const SurfaceGraph& graph, const VoxelKey& start, size_t& steps) const;

    // stats from geodesic distance computations (delete?)
    size_t nodes_visited_source_{0};
    size_t nodes_visited_frontier_{0};

    Params params_;
};



} // namespace



#endif