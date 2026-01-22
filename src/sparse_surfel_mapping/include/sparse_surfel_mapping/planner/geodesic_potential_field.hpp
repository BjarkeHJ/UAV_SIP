#ifndef GEODESIC_POTENTIAL_FIELD_HPP_
#define GEODESIC_POTENTIAL_FIELD_HPP_

#include <queue>
#include <functional>
#include "sparse_surfel_mapping/planner/surface_graph.hpp"

namespace sparse_surfel_map {

struct DijkstraElement {
    VoxelKey key;
    float distance;
    size_t hops;
    bool operator>(const DijkstraElement& other) const {
        return distance > other.distance;
    }
};

struct FrontierBasin {
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
    GeodesicPotentialField();

    // Geodesic distances
    void compute_distances_from_seed(SurfaceGraph& graph, const VoxelKey& seed); // compute geodesic distances from seed
    void compute_distances_to_frontiers(SurfaceGraph& graph); // compute geodesic distance to nearest frontier

    // Potential field
    void compute_potential_field(SurfaceGraph& graph, float voxel_size);

    // Frontier/Basin clustering
    std::vector<FrontierBasin> detect_basins(SurfaceGraph& graph);

    // statistics
    size_t nodes_visited_source() const { return nodes_visited_source_; } // nubmer of nodes visisted in dijkstra from source
    size_t nodes_visited_frontier() const { return nodes_visited_frontier_; } // number of nodes visited in dijkstra from frontiers

private:
    using PriorityQueue = std::priority_queue<DijkstraElement, std::vector<DijkstraElement>, std::greater<DijkstraElement>>; // min-heap (keep smallest element on top of queue)

    // geodesic distances
    void compute_approximate_density(SurfaceGraph& graph, float voxel_size);
    void compute_gaussian_density(SurfaceGraph& graph);

    // basin extraction
    std::vector<VoxelKey> find_local_maxima(const SurfaceGraph& graph); // find local maxima of potential on frontier surfels
    std::vector<FrontierBasin> watershed_assign(SurfaceGraph& graph, const std::vector<VoxelKey>& seeds); // propagate basin labels from seeds (maximas)
    void merge_similar_basins(SurfaceGraph& graph, std::vector<FrontierBasin>& basins);
    void filter_small_basins(std::vector<FrontierBasin>& basins);
    void reassign_ids(SurfaceGraph& graph, std::vector<FrontierBasin>& basins);

    // stats from geodesic distance computations (delete?)
    size_t nodes_visited_source_{0};
    size_t nodes_visited_frontier_{0};
};



} // namespace



#endif