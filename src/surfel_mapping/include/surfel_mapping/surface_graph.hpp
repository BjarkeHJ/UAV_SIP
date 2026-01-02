#ifndef SURFEL_GRAPH_HPP
#define SURFEL_GRAPH_HPP

#include <chrono>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <optional>
#include <Eigen/Dense>
#include "surfel_mapping/surfel_map.hpp"

namespace surface_inspection_planning {

struct GraphNode {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // cached surfel properties
    size_t surfel_idx;
    Eigen::Vector3f center;
    Eigen::Vector3f normal;
    Eigen::Vector3f major_axis;
    float major_radius; // sqrt(lambda_max)
    float minor_radius; // sqrt(lambda_min)
    float anisotropy;
    float importance;

    // node inspection state
    bool inspected = false;
    float inspection_quality = 0.0f;
    uint64_t last_inspection_stamp = 0;

    // graph connectivity
    std::vector<size_t> edge_indices;

    bool is_valid = true;
    uint64_t last_surfel_stamp = 0;
};


struct GraphEdge {
    size_t from_node;
    size_t to_node;

    float distance;
    float normal_change;
    float alignment_cost;
    float step_height;

    float total_cost;

    bool is_structural;
    bool is_valid;
};

struct EdgeCostWeights {
    float w_distance = 1.0f;
    float w_normal_change = 0.5f;
    float w_alignment = 0.3f;
    float w_step = 2.0f;
    float structural_bonus = 0.2f;
};

struct ConnectivityParams {
    float max_edge_distance = 0.8f;
    float min_edge_distance = 0.05f;
    
    float min_normal_dot = 0.5; // 60 deg
    float max_plane_step = 0.1f;

    float alignment_threshold = 0.7f;

    float min_node_confidence = 0.1f;
    float min_node_radius = 0.03f;

    size_t min_node_observations = 3;

    size_t max_neighbors_per_node = 12;
};

struct PathResult {
    std::vector<size_t> node_indices;
    float total_cost = 0.0f;
    bool valid = false;
};

struct CoverageStats {
    size_t total_nodes = 0;
    size_t inspected_nodes = 0;
    float coverage_ratio = 0.0f;
    float weighted_coverage = 0.0f;
    float total_surface_area = 0.0f;
    float inspected_surface_area = 0.0f;
};

struct GraphUpdateStats {
    size_t nodes_added = 0;
    size_t nodes_updated = 0;
    size_t nodes_removed = 0;
    size_t edges_added = 0;
    size_t edges = removed = 0;
    double update_time_ms = 0.0;
};

class SurfaceGraph {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SurfaceGraph();
    explicit SurfaceGraph(const ConnectivityParams& params);

    void build_from_map(const SurfelMap& map);
    GraphUpdateStats update_incremental(const SurfelMap& map, const std::vector<size_t>& new_surfels, const std::vector<size_t>& updated_surfels, const std::vector<size_t>& removed_surfels);
    GraphUpdateStats update_from_map(const SurfelMap& map);
    void compact();

    std::optional<size_t> find_node_by_surfel(size_t surfel_idx) const;
    size_t find_nearest_node(const Eigen::Vector3f& point) const;
    std::vector<size_t> find_nodes_in_radius(const Eigen::Vector3f& point, float radius) const;

    std::vector<size_t> get_neighbors(size_t node_idx) const;

    PathResult find_path(size_t from_node, size_t to_node) const;

    void mark_inspected(size_t node_idx, float quality, uint64_t timestamp);
    std::vector<size_t> get_uninspected_nodes() const;
    std::vector<size_t> get_frontier_nodes() const;
    CoverageStats get_coverage_stats() const;

    const std::vector<GraphNode>& nodes() const { return nodes_; }
    const std::vector<GraphEdge>& edges() const { return edges_; }

    size_t num_nodes() const { return valid_node_count_; }
    size_t num_edges() const { return valid_edge_count_; }

    const ConnectivityParams& params() const { return params_; }
    void set_params(const ConnectivityParams& p) { params_ = p; }

    const EdgeCostWeights& weights() const { return weights_; }
    void set_weights(const EdgeCostWeights& w);

    void clear();

private:
    bool surfel_qualifies_as_node(const Surfel& surfel) const;
    size_t create_node(const SurfelMap& map, size_t surfel_idx);
    void update_node_properties(const SurfelMap& map, size_t node_idx);
    void invalidate_node(size_t node_idx);

    float compute_node_importance(const Surfel& surfel) const;
    Eigen::Vector3f compute_major_axis_3d(const Surfel& surfel) const;

    void create_edges_for_node(size_t node_idx);
    void remove_edges_for_node(size_t node_idx);
    void update_edges_for_node(size_t node_idx);

    bool check_connectivity(const GraphNode& n1, const GraphNode& n2) const;
    std::optional<GraphEdge> try_create_edge(size_t from_node, size_t to_node);
    float compute_alignment_cost(const GraphNode& from, const GraphNode& to, const Eigen::Vector3f& edge_dir) const;
    void compute_edge_cost(GraphEdge& edge) const;

    VoxelKey point_to_voxel(const Eigen::Vector3f& p) const;
    void add_node_to_spatial_index(size_t node_idx);
    void remove_node_from_spatial_index(size_t node_idx);
    void rebuild_spatial_index();

    /* DATA */
    ConnectivityParams params_;
    EdgeCostWeights weights_;
    float voxel_size_;

    std::vector<GraphNode> nodes_;
    std::vector<GraphEdge> edges_;

    std::vector<size_t> free_node_slots_;
    std::vector<size_t> free_edge_slots_;

    size_t valid_node_count_ = 0;
    size_t valid_edge_count_ = 0;

    std::unordered_map<size_t, size_t> surfel_to_node_;
    
    std::unordered_map<VoxelKey, std::vector<size_t>, VoxelKeyHash> node_voxel_index_;

    uint64_t last_update_frame_ = 0;
};



};


#endif