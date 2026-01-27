#include "sparse_surfel_mapping/planner/geodesic_potential_field.hpp"

namespace sparse_surfel_map {

GeodesicPotentialField::GeodesicPotentialField() {}


// Compute geodesic distances on surface graph
void GeodesicPotentialField::compute_distances_from_seed(SurfaceGraph& graph, const VoxelKey& seed) {
    
    const size_t max_horizon_hops = 100;
    const float max_horizon_distance = 5.0f;
    
    nodes_visited_source_ = 0;

    graph.reset_distances();

    SurfelNode* seed_node = graph.get_node(seed);
    if (!seed_node) return;

    // from seed-source: compute geodesic distances and hop count to graph nodes (can be bounded)
    PriorityQueue pq;
    seed_node->d_source = 0.0f;
    seed_node->hop_count = 0;
    pq.push({seed, 0.0f, 0}); // key, distance, hops

    while (!pq.empty()) {
        DijkstraElement current = pq.top();
        pq.pop();

        if (current.hops > max_horizon_hops) continue; // max horizon hops
        if (current.distance > max_horizon_distance) continue; // max horizon distance

        SurfelNode* current_node = graph.get_node(current.key);
        if (!current_node) continue;

        if (current.distance > current_node->d_source) continue; // skip if we already know a better path (seed->current)

        nodes_visited_source_++;

        for (const auto& nb_key : graph.get_surface_neighbors(current.key)) {
            SurfelNode* nb_node = graph.get_node(nb_key);
            if (!nb_node) continue;

            float edge_weight = graph.compute_edge_weight(*current_node, *nb_node);
            float new_dist = current.distance + edge_weight;
            size_t new_hops = current.hops + 1;

            if (new_dist < nb_node->d_source) {
                nb_node->d_source = new_dist;
                nb_node->hop_count = new_hops;
                pq.push({nb_key, new_dist, new_hops});
            }
        }
    }
}

void GeodesicPotentialField::compute_distances_to_frontiers(SurfaceGraph& graph) {
    
    const float frontier_propagation_radius = 5.0f; // param
    
    nodes_visited_frontier_ = 0;
    
    // iterate over graph nodes - reset frontier distances 
    for (auto& [key, node] : graph) {
        node.d_frontier = std::numeric_limits<float>::infinity();
    }

    const VoxelKeySet& frontiers = graph.frontier_set();
    if (frontiers.empty()) return; // nothing to do

    PriorityQueue pq;

    // init frontier distances with 0
    for (const auto& f_key : frontiers) {
        SurfelNode* node = graph.get_node(f_key);
        if (node) {
            node->d_frontier = 0.0f;
            pq.push({f_key, 0.0f, 0});
        }
    }

    while (!pq.empty()) {
        DijkstraElement current = pq.top();
        pq.pop();

        if (current.distance > frontier_propagation_radius) continue; // limit frontier propagation distance (attraction region to frontier)

        SurfelNode* current_node = graph.get_node(current.key);
        if (!current_node) continue;

        if (current.distance > current_node->d_frontier) continue; // node already closer to a different frontier

        nodes_visited_frontier_++;

        for (const auto& nb_key : graph.get_surface_neighbors(current.key)) {
            SurfelNode* nb_node = graph.get_node(nb_key);
            if (!nb_node) continue;

            float edge_weight = graph.compute_edge_weight(*current_node, *nb_node);
            float new_dist = current.distance + edge_weight;

            if (new_dist < nb_node->d_frontier) {
                nb_node->d_frontier = new_dist;
                pq.push({nb_key, new_dist, 0});
            }
        }
    }
}


// Compute potential field (assign potential to each graph node)
void GeodesicPotentialField::compute_potential_field(SurfaceGraph& graph, float voxel_size) {
    
    const float alpha_source = 2.0f;
    const float beta_frontier = 2.0f;
    const float gamma_density = 0.5f;

    const float lambda = 5.0f; // frontier attraction decay
    const float radius = 10.0f; // density truncation radius (geodesic distance)
    
    const VoxelKeySet& frontiers = graph.frontier_set();

    // find maximum finite distance for normalization
    float max_d = 0.0f;
    for (const auto& [key, node] : graph) {
        if (node.d_source < std::numeric_limits<float>::infinity()) {
            max_d = std::max(max_d, node.d_source);
        }
    }

    if (max_d < 1e-6f) max_d = 1.0f;
    for (auto& [key, node] : graph) {
        if (node.d_source < std::numeric_limits<float>::infinity()) {
            node.psi_source = node.d_source / max_d;
            node.psi_frontier = std::exp(-node.d_frontier / lambda); // make the attraction decay exponentially controlled by lambda
        }
        else {
            node.psi_source = 1.0f; // max cost (unreachable)
            node.psi_frontier = 0.0f; // no attraction
        }

        // Density weight (approximate density kernel)
        // skip non-frontiers unless we want a full computation
        bool compute_density_for_all = true;
        if (!compute_density_for_all && node.state != SurfelNode::State::FRONTIER) {
            node.psi_density = 0.0f;
            continue;
        }
 
        // BFS to count frontiers within geodesic radius
        size_t frontier_count = 0;
        VoxelKeySet visisted;
        std::queue<std::pair<VoxelKey, float>> bfs_queue; // FIFO queue

        bfs_queue.push({key, 0.0f});
        visisted.insert(key);

        while (!bfs_queue.empty()) {
            auto [curr_key, curr_dist] = bfs_queue.front();
            bfs_queue.pop();

            if (frontiers.count(curr_key) > 0) {
                frontier_count++;
            }

            if (curr_dist >= radius) continue; // truncate

            for (const auto& nb_key : graph.get_surface_neighbors(curr_key)) {
                if (visisted.count(nb_key) > 0) continue;
                visisted.insert(nb_key);

                float edge_w = graph.compute_edge_weight(curr_key, nb_key);
                bfs_queue.push({nb_key, curr_dist + edge_w});
            }
        }

        float max_count = std::pow(2.0f * radius / voxel_size, 2.0);
        node.psi_density = static_cast<float>(frontier_count) / std::max(1.0f, max_count);

        // combine total potential value
        node.phi = alpha_source * node.psi_source + beta_frontier * node.psi_frontier + gamma_density * node.psi_density;
    }
}


// Detect frontier pools/clusters and extract
std::vector<FrontierPool> GeodesicPotentialField::detect_frontier_pools(SurfaceGraph& graph) {

    const size_t min_pool_size = 1;
    const size_t max_n_pools = 50;

    const VoxelKeySet& frontiers = graph.frontier_set();
    if (frontiers.empty()) return {};

    graph.reset_basins();

    // gradient ascend for each frontier -> attractor
    std::unordered_map<VoxelKey, VoxelKey, VoxelKeyHash> surfel_to_attractor;
    surfel_to_attractor.reserve(frontiers.size());

    size_t total_steps = 0;
    for (const auto& f_key : frontiers) {
        size_t steps = 0;
        VoxelKey attractor = ascend_to_attractor(graph, f_key, steps);
        surfel_to_attractor[f_key] = attractor;
        total_steps += steps;
    }

    // merge small and similar pools??


    // group by attractor
    std::unordered_map<VoxelKey, FrontierPool, VoxelKeyHash> pool_map;
    for (const auto& [surfel, attractor] : surfel_to_attractor) {
        auto& pool = pool_map[attractor];
        if (pool.frontier_surfels.empty()) {
            // first is the attractor
            pool.peak_surfel = attractor;
            if (const SurfelNode* n = graph.get_node(attractor)) {
                pool.peak_potential = n->phi;
            }
        }
        // otherwise corresponding basin surfels
        pool.frontier_surfels.push_back(surfel);
    }

    // convert, filter, sort
    std::vector<FrontierPool> pools;
    pools.reserve(pool_map.size());

    for (auto& [att, pool] : pool_map) {
        if (pool.size() >= min_pool_size) {
            pools.push_back(std::move(pool));
        }
    }

    // sort by highest potential first
    std::sort(pools.begin(), pools.end(),
        [](const FrontierPool& a, const FrontierPool& b) {
            return a.peak_potential > b.peak_potential;
    });

    if (pools.size() > max_n_pools) {
        pools.resize(max_n_pools); // truncate to best maxN
    }

    for (size_t i = 0; i < pools.size(); ++i) {
        pools[i].id = static_cast<int>(i);
        for (const auto& key : pools[i].frontier_surfels) {
            if (SurfelNode* n = graph.get_node(key)) {
                n->basin_id = static_cast<int>(i);
            }
        }
        pools[i].compute_geometry(graph);
    }

    return pools;
}

VoxelKey GeodesicPotentialField::ascend_to_attractor(const SurfaceGraph& graph, const VoxelKey& start, size_t& steps) const {
    VoxelKey current = start;
    steps = 0;

    const size_t max_ascend_steps = 25;

    for (size_t i = 0; i < max_ascend_steps; ++i) {
        const SurfelNode* node = graph.get_node(current);
        if (!node) break;

        VoxelKey best = current;
        float best_phi = node->phi;

        for (const auto& nb_key : graph.get_surface_neighbors(current)) {
            if (!graph.is_frontier(nb_key)) continue;

            if (const SurfelNode* nb = graph.get_node(nb_key)) {
                if (nb->phi > best_phi) {
                    best_phi = nb->phi;
                    best = nb_key;
                }
            }
        }

        if (best == current) break; // reached attractor

        current = best;
        steps++;
    }

    return current;
}

void FrontierPool::compute_geometry(const SurfaceGraph& graph) {
    // update frontier cluster/pool geometry
    if (frontier_surfels.empty()) return;

    centroid = Eigen::Vector3f::Zero();
    mean_normal = Eigen::Vector3f::Zero();

    for (const auto& key : frontier_surfels) {
        const SurfelNode* node = graph.get_node(key);
        if (node) {
            centroid += node->position;
            mean_normal += node->normal;
        }
    }

    centroid /= static_cast<float>(frontier_surfels.size());
    
    if (mean_normal.norm() > 1e-6f) {
        mean_normal.normalize();
    }
    else {
        mean_normal = Eigen::Vector3f::Zero(); // invalid
    }

    float sum_sq_diff = 0.0f;
    for (const auto& key : frontier_surfels) {
        const SurfelNode* node = graph.get_node(key);
        if (node) {
            float dot = node->normal.dot(mean_normal);
            float diff = 1.0f - dot;
            sum_sq_diff += diff * diff;
        }
    }

    normal_variance = sum_sq_diff / static_cast<float>(frontier_surfels.size());
    total_area_estimate = static_cast<float>(frontier_surfels.size());
}

} // namespace