#include "sparse_surfel_mapping/planner/surface_graph.hpp"

namespace sparse_surfel_map {

SurfaceGraph::SurfaceGraph() {}

void SurfaceGraph::build_from_map(const SurfelMap& map) {
    clear();

    const auto& voxels = map.voxels();

    for (auto it = voxels.begin(); it != voxels.end(); ++it) {
        const Voxel& voxel = it->second;

        if (!voxel.has_valid_surfel()) continue;

        const VoxelKey& key = voxel.key();
        const Surfel& surfel = voxel.surfel();
        
        SurfelNode node;
        node.key = key;
        node.position = surfel.mean();
        node.normal = surfel.normal();
        node.state = SurfelNode::State::UNKNOWN;

        nodes_[key] = node;
    }
}

void SurfaceGraph::classify_surfels(const VoxelKeySet& covered_set) {
    // clear frontier?
    covered_set_ = covered_set;

    for (auto& [key, node] : nodes_) {
        if (covered_set.count(key) > 0) {
            node.state = SurfelNode::State::COVERED;
        }
        else {
            // check if neighbors are covered
            // frontier if: uncovered with covered neighbors
            bool has_covered_nb = false;
            for (const auto& nb_key : get_surface_neighbors(key)) {
                if (covered_set.count(nb_key) > 0) {
                    has_covered_nb = true;
                    break;
                }
            }

            if (has_covered_nb) {
                node.state = SurfelNode::State::FRONTIER;
                frontier_set_.insert(key);
            }
            else {
                node.state = SurfelNode::State::UNKNOWN;
            }
        }
    }
}

std::vector<VoxelKey> SurfaceGraph::get_surface_neighbors(const VoxelKey& key) const {
    std::vector<VoxelKey> nbs;
    nbs.reserve(26);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue; // skip self

                VoxelKey nb_key{key.x + dx, key.y + dy, key.z + dz};

                if (nodes_.count(nb_key) > 0) {
                    nbs.push_back(nb_key);
                }
            }
        }
    }

    return nbs;
}

float SurfaceGraph::compute_edge_weight(const VoxelKey& from, const VoxelKey& to) const {
    const SurfelNode* from_node = get_node(from);
    const SurfelNode* to_node = get_node(to);

    if (!from_node || !to_node) {
        return std::numeric_limits<float>::infinity();
    }

    return compute_edge_weight(*from_node, *to_node);
}

float SurfaceGraph::compute_edge_weight(const SurfelNode& from, const SurfelNode& to) const {
    
    const float normal_dot_th = 0.5f;
    const float edge_weight_crease_penalty = 2.0f;
    
    float eucl_dist = (to.position - from.position).norm();
    float normal_dot = from.normal.dot(to.normal);
    float normal_factor = 1.0f;


    // penalize if sharp turn (>60 deg)
    if (normal_dot < normal_dot_th) {
        normal_factor = edge_weight_crease_penalty / (1.0f + normal_dot); 
    }

    return eucl_dist * normal_factor;
}


// Simple convinience accessor
SurfelNode* SurfaceGraph::get_node(const VoxelKey& key) {
    auto it = nodes_.find(key);
    return (it != nodes_.end()) ? &it->second : nullptr;
}
const SurfelNode* SurfaceGraph::get_node(const VoxelKey& key) const {
    auto it = nodes_.find(key);
    return (it != nodes_.end()) ? &it->second : nullptr;
}
bool SurfaceGraph::has_node(const VoxelKey& key) const {
    return nodes_.count(key) > 0;
}
void SurfaceGraph::clear() {
    nodes_.clear();
    frontier_set_.clear();
    covered_set_.clear();
}

// resets
void SurfaceGraph::reset_distances() {
    for (auto& [key, node] : nodes_) {
        node.d_source = std::numeric_limits<float>::infinity();
        node.d_frontier = std::numeric_limits<float>::infinity();
        node.hop_count = std::numeric_limits<size_t>::max();
    }
}

void SurfaceGraph::reset_potentials() {
    for (auto& [key, node] : nodes_) {
        node.psi_source = 0.0f;
        node.psi_frontier = 0.0f;
        node.psi_density = 0.0f;
        node.phi = 0.0f;
    }
}

void SurfaceGraph::reset_basins() {
    for (auto& [key, node] : nodes_) {
        node.basin_id = -1;
    }
}


} // namespace