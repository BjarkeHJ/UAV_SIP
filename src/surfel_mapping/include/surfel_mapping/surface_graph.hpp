#ifndef SURFACE_GRAPH_HPP_
#define SURFACE_GRAPH_HPP_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <optional>
#include <Eigen/Dense>
#include "surfel_mapping/surfel_map.hpp"

namespace surface_inspection_planning {

// ============================================================================
// LIGHTWEIGHT NODE - Only graph-specific data, geometry via surfel lookup
// ============================================================================
struct GraphNode {
    size_t surfel_idx;                      // Reference into SurfelMap
    std::vector<size_t> edge_indices;       // Indices into graph's edge vector
    
    // Inspection state (graph-specific, not in surfel)
    bool inspected = false;
    float inspection_quality = 0.0f;
    uint64_t last_inspection_stamp = 0;
    
    // Validity
    bool is_valid = true;
};

// ============================================================================
// GRAPH EDGE
// ============================================================================
struct GraphEdge {
    size_t from_node;
    size_t to_node;
    
    // Cost components (for debugging/tuning)
    float distance = 0.0f;
    float normal_change = 0.0f;
    float alignment_cost = 0.0f;
    float step_height = 0.0f;
    
    // Final cost
    float total_cost = 0.0f;
    
    // Classification
    bool is_structural = false;
    bool is_valid = true;
};

// ============================================================================
// PARAMETERS
// ============================================================================
struct GraphParams {
    // Node qualification (in addition to surfel's is_mature)
    float min_node_confidence = 0.1f;
    float min_node_radius = 0.1f;
    size_t min_node_observations = 10;
    
    // Edge connectivity
    float max_edge_distance = 1.0f;
    float min_edge_distance = 0.1f;
    float min_normal_dot = 0.5f;            // cos(60°)
    float max_plane_step = 0.1f;
    size_t max_neighbors_per_node = 2;
    
    // Edge cost weights
    float w_distance = 1.0f;
    float w_normal_change = 0.5f;
    float w_alignment = 0.3f;
    float w_step = 2.0f;
    float structural_bonus = 0.2f;
    
    // Structural detection
    float anisotropy_threshold = 0.5f;
    float alignment_threshold = 0.7f;
};

// ============================================================================
// RESULTS / STATS
// ============================================================================
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
    size_t nodes_removed = 0;
    size_t edges_created = 0;
    size_t edges_removed = 0;
    double update_time_ms = 0.0;
};

// ============================================================================
// SURFACE GRAPH - Lightweight view over SurfelMap
// ============================================================================
class SurfaceGraph {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Must be constructed with a surfel map reference
    explicit SurfaceGraph(const SurfelMap& map);
    SurfaceGraph(const SurfelMap& map, const GraphParams& params);
    
    // No default constructor - map reference is required
    SurfaceGraph() = delete;
    
    // === Parameters ===
    const GraphParams& params() const { return params_; }
    void set_params(const GraphParams& p) { params_ = p; }
    
    // === Build / Update ===
    void rebuild();
    GraphUpdateStats update_incremental(
        const std::vector<size_t>& new_surfels,
        const std::vector<size_t>& updated_surfels,
        const std::vector<size_t>& removed_surfels
    );
    
    // Compact graph (remove invalid nodes/edges, rebuild indices)
    void compact();
    void clear();
    
    // === Geometry Access (via surfel lookup) ===
    // These are the primary way to get node geometry
    const Eigen::Vector3f& center(size_t node_idx) const;
    const Eigen::Vector3f& normal(size_t node_idx) const;
    float radius(size_t node_idx) const;
    float confidence(size_t node_idx) const;
    float anisotropy(size_t node_idx) const;
    Eigen::Vector3f major_axis(size_t node_idx) const;
    float importance(size_t node_idx) const;
    float area(size_t node_idx) const;
    
    // Access underlying surfel directly
    const Surfel& node_surfel(size_t node_idx) const;
    size_t node_surfel_idx(size_t node_idx) const;
    
    // === Graph Queries ===
    std::optional<size_t> find_node_by_surfel(size_t surfel_idx) const;
    size_t find_nearest_node(const Eigen::Vector3f& point) const;
    std::vector<size_t> find_nodes_in_radius(const Eigen::Vector3f& point, float rad) const;
    std::vector<size_t> get_neighbors(size_t node_idx) const;
    
    // === Path Planning ===
    PathResult find_path(size_t from_node, size_t to_node) const;
    
    // === Inspection State ===
    void mark_inspected(size_t node_idx, float quality, uint64_t timestamp);
    void reset_inspection_state();
    std::vector<size_t> get_uninspected_nodes() const;
    std::vector<size_t> get_frontier_nodes() const;
    CoverageStats get_coverage_stats() const;
    
    // === Accessors ===
    const std::vector<GraphNode>& nodes() const { return nodes_; }
    const std::vector<GraphEdge>& edges() const { return edges_; }
    size_t num_nodes() const { return valid_node_count_; }
    size_t num_edges() const { return valid_edge_count_; }
    bool empty() const { return valid_node_count_ == 0; }
    
    const SurfelMap& surfel_map() const { return map_; }
    
private:
    // === Node Management ===
    bool surfel_qualifies(size_t surfel_idx) const;
    size_t create_node(size_t surfel_idx);
    void invalidate_node(size_t node_idx);
    
    // === Edge Management ===
    void create_edges_for_node(size_t node_idx);
    void remove_edges_for_node(size_t node_idx);
    bool check_connectivity(size_t node_i, size_t node_j) const;
    float compute_edge_cost(size_t node_i, size_t node_j, GraphEdge& edge) const;
    float compute_alignment_cost(size_t node_i, size_t node_j, 
                                  const Eigen::Vector3f& edge_dir) const;
    
    // === Spatial Index ===
    VoxelKey point_to_voxel(const Eigen::Vector3f& p) const;
    void add_to_spatial_index(size_t node_idx);
    void remove_from_spatial_index(size_t node_idx);
    void rebuild_spatial_index();
    
    // === Data ===
    const SurfelMap& map_;              // Reference to surfel map (required)
    GraphParams params_;
    
    std::vector<GraphNode> nodes_;
    std::vector<GraphEdge> edges_;
    
    // Free lists for slot reuse
    std::vector<size_t> free_node_slots_;
    std::vector<size_t> free_edge_slots_;
    
    size_t valid_node_count_ = 0;
    size_t valid_edge_count_ = 0;
    
    // Mappings
    std::unordered_map<size_t, size_t> surfel_to_node_;
    std::unordered_map<VoxelKey, std::vector<size_t>, VoxelKeyHash> spatial_index_;
    
    float voxel_size_;
};


// Inline accessors
inline const Surfel& SurfaceGraph::node_surfel(size_t node_idx) const {
    return map_.get_surfels()[nodes_[node_idx].surfel_idx];
}

inline size_t SurfaceGraph::node_surfel_idx(size_t node_idx) const {
    return nodes_[node_idx].surfel_idx;
}

inline const Eigen::Vector3f& SurfaceGraph::center(size_t node_idx) const {
    return node_surfel(node_idx).center;
}

inline const Eigen::Vector3f& SurfaceGraph::normal(size_t node_idx) const {
    return node_surfel(node_idx).normal;
}

inline float SurfaceGraph::radius(size_t node_idx) const {
    return node_surfel(node_idx).get_radius();
}

inline float SurfaceGraph::confidence(size_t node_idx) const {
    return node_surfel(node_idx).confidence;
}

inline float SurfaceGraph::anisotropy(size_t node_idx) const {
    return node_surfel(node_idx).get_planarity();
}

inline Eigen::Vector3f SurfaceGraph::major_axis(size_t node_idx) const {
    const Surfel& s = node_surfel(node_idx);
    int max_idx = (s.eigenvalues(0) > s.eigenvalues(1)) ? 0 : 1;
    Eigen::Vector2f major_2d = s.eigenvectors.col(max_idx);
    return (major_2d.x() * s.tangent_u + major_2d.y() * s.tangent_v).normalized();
}

inline float SurfaceGraph::area(size_t node_idx) const {
    const Surfel& s = node_surfel(node_idx);
    return M_PI * std::sqrt(s.eigenvalues(0) * s.eigenvalues(1));
}

inline float SurfaceGraph::importance(size_t node_idx) const {
    const Surfel& s = node_surfel(node_idx);
    float a = area(node_idx);
    float area_score = std::min(1.0f, a / 0.1f);
    float obs_score = std::min(1.0f, static_cast<float>(s.observation_count) / 20.0f);
    return 0.4f * area_score + 0.4f * s.confidence + 0.2f * obs_score;
}

} // namespace surface_inspection_planning

#endif // SURFACE_GRAPH_HPP_