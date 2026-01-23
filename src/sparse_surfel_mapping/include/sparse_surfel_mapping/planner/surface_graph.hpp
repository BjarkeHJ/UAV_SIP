#ifndef SURFACE_GRAPH_HPP_
#define SURFACE_GRAPH_HPP_

#include <unordered_map>
#include <Eigen/Core>

#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/common/planning_types.hpp"

namespace sparse_surfel_map {

struct SurfelNode {
    VoxelKey key;
    Eigen::Vector3f position;
    Eigen::Vector3f normal;

    enum class State { UNKNOWN, COVERED, FRONTIER };
    State state{State::UNKNOWN}; 

    // Geodesic distances
    float d_source{std::numeric_limits<float>::infinity()}; // From seed surfel
    float d_frontier{std::numeric_limits<float>::infinity()}; // TO nearest frontier
    size_t hop_count{std::numeric_limits<size_t>::max()}; // BFS hop count from seed

    // Potential field values
    float psi_source{0.0f};
    float psi_frontier{0.0f};
    float psi_density{0.0f};
    float phi{0.0f}; // combined potential

    int basin_id{-1}; // frontier basin (cluster) id
};

class SurfaceGraph {
public:
    // todo: wire into framework
    struct Params {
        // Edge weight
        float normal_dot_th{0.0f};
        float edge_weight_crease_penalty{0.0f};
    };

    using NodeMap = std::unordered_map<VoxelKey, SurfelNode, VoxelKeyHash>;

    SurfaceGraph();
    void build_from_map(const SurfelMap& map);
    void classify_surfels(const VoxelKeySet& covered_set);
    void clear();

    SurfelNode* get_node(const VoxelKey& key);
    const SurfelNode* get_node(const VoxelKey& key) const;
    bool has_node(const VoxelKey& key) const;
    size_t size() const { return nodes_.size(); }
    bool empty() const { return nodes_.empty(); }

    // Make the class iterable 
    NodeMap::iterator begin() { return nodes_.begin(); }
    NodeMap::iterator end() { return nodes_.end(); }
    NodeMap::const_iterator begin() const { return nodes_.begin(); }
    NodeMap::const_iterator end() const { return nodes_.end(); }

    std::vector<VoxelKey> get_surface_neighbors(const VoxelKey& key) const; // valid surfels in 26-nbh

    float compute_edge_weight(const VoxelKey& from, const VoxelKey& to) const;
    float compute_edge_weight(const SurfelNode& fromm, const SurfelNode& to) const;

    const VoxelKeySet& frontier_set() const { return frontier_set_; }
    const VoxelKeySet& covered_set() const { return covered_set_; }
    size_t num_frontiers() const { return frontier_set_.size(); }
    size_t num_covered() const { return covered_set_.size(); }
    bool is_frontier(const VoxelKey& key) const { return frontier_set_.count(key) > 0; }
    bool is_covered(const VoxelKey& key) const { return covered_set_.count(key) > 0; }

    void reset_distances();
    void reset_potentials();
    void reset_basins();

private:
    NodeMap nodes_;
    VoxelKeySet frontier_set_;
    VoxelKeySet covered_set_;
};

} // namespace

#endif