#include "sparse_surfel_mapping/planner/geodesic_potential_field.hpp"

namespace sparse_surfel_map {

GeodesicPotentialField::GeodesicPotentialField() {}


// Compute geodesic distances on surface graph
void GeodesicPotentialField::compute_distances_from_seed(SurfaceGraph& graph, const VoxelKey& seed) {
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

        if (current.hops > 100) continue; // max horizon hops
        if (current.distance > 50) continue; // max horizon distance

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

        if (current.distance > 10.0f) continue; // limit frontier propagation distance (attraction region to frontier)

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
    const float gamme_density = 0.5f;

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

        // Density weight
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
        node.phi = alpha_source * node.psi_source + beta_frontier * node.psi_frontier + gamme_density * node.psi_density;
    }
}


// Detect frontier basins/clusters and extract
std::vector<FrontierBasin> GeodesicPotentialField::detect_basins(SurfaceGraph& graph) {
    graph.reset_basins();

    // find local maxima from frontiers
    std::vector<VoxelKey> maxima = find_local_maxima(graph);
    if (maxima.empty()) return {};

    std::vector<FrontierBasin> basins = watershed_assign(graph, maxima);

    merge_similar_basins(graph, basins);
    filter_small_basins(basins);
    reassign_ids(graph, basins);

    for (auto& basin : basins) {
        basin.compute_geometry(graph);
    }

    return basins;
}

std::vector<VoxelKey> GeodesicPotentialField::find_local_maxima(const SurfaceGraph& graph) {
    std::vector<VoxelKey> maxima;
    const VoxelKeySet& frontiers = graph.frontier_set();

    for (const auto& f_key : frontiers) {
        const SurfelNode* node = graph.get_node(f_key);
        if (!node) continue;

        // find frontier maximum among 26-conn nbs
        bool is_maximum = true;
        for (const auto& nb_key : graph.get_surface_neighbors(f_key)) {
            if (!graph.is_frontier(nb_key)) continue;

            const SurfelNode* nb_node = graph.get_node(nb_key);
            if (!nb_node) continue;

            if (nb_node->phi > node->phi) {
                is_maximum = false;
                break;
            }
        }

        if (is_maximum && node->phi > -std::numeric_limits<float>::max()) {
            maxima.push_back(f_key);
        }
    }

    return maxima;
}

std::vector<FrontierBasin> GeodesicPotentialField::watershed_assign(SurfaceGraph& graph, const std::vector<VoxelKey>& seeds) {
    std::vector<FrontierBasin> basins;

    // create basin for each seed
    for (size_t i = 0; i < seeds.size(); ++i) {
        FrontierBasin basin;
        basin.id = static_cast<int>(i);
        basin.peak_surfel = seeds[i];

        SurfelNode* node = graph.get_node(seeds[i]);
        if (node) {
            basin.peak_potential = node->phi;
            node->basin_id = basin.id;
            basin.frontier_surfels.push_back(seeds[i]);
        }

        basins.push_back(std::move(basin));
    }

    // watershed: propagate labels downhill (decreasing phi)
    using PQElement = std::pair<float, VoxelKey>;
    std::priority_queue<PQElement> pq; // max-heap (keeps max value on top)

    // Initialize max-heap with seeds (local maxima in 26-conn nbh)
    for (const auto& seed : seeds) {
        const SurfelNode* node = graph.get_node(seed);
        if (node) {
            pq.push({node->phi, seed});
        }
    }

    while (!pq.empty()) {
        auto [phi, key] = pq.top();
        pq.pop();

        SurfelNode* node = graph.get_node(key);
        if (!node) continue;

        int current_basin = node->basin_id;
        if (current_basin < 0) continue;

        // propagate to unassigned frontier nbs
        for (const auto& nb_key : graph.get_surface_neighbors(key)) {
            if (!graph.is_frontier(nb_key)) continue;

            SurfelNode* nb_node = graph.get_node(nb_key);
            if (!nb_node) continue;
            if (nb_node->basin_id >= 0) continue; // already assigned

            nb_node->basin_id = current_basin;
            basins[current_basin].frontier_surfels.push_back(nb_key);

            pq.push({nb_node->phi, nb_key});
        }
    }

    return basins;
}

void GeodesicPotentialField::merge_similar_basins(SurfaceGraph& graph, std::vector<FrontierBasin>& basins) {
    if (basins.size() < 2) return;

    const float basin_merge_th = 0.8f;

    std::vector<bool> merged(basins.size(), false);

    for (size_t i = 0; i < basins.size(); ++i) {
        if (merged[i]) continue;

        for (size_t j = i + 1; j < basins.size(); ++j) {
            if (merged[j]) continue;

            float peak_diff = std::abs(basins[i].peak_potential - basins[j].peak_potential);
            float peak_max = std::max(std::abs(basins[i].peak_potential), std::abs(basins[j].peak_potential));

            if (peak_max > 1e-6f && peak_diff / peak_max > (1.0f - basin_merge_th)) continue; // dont merge if peaks are very different (??)

            bool adjacent = false;
            for (const auto& key_i : basins[i].frontier_surfels) {
                if (adjacent) break;
                for (const auto& nb_key : graph.get_surface_neighbors(key_i)) {
                    auto it = std::find(basins[j].frontier_surfels.begin(), basins[j].frontier_surfels.end(), nb_key); 
                    if (it != basins[j].frontier_surfels.end()) {
                        adjacent = true;
                        break;
                    }
                }
            }

            if (adjacent) {
                // merge j into i
                for (const auto& key : basins[j].frontier_surfels) {
                    basins[i].frontier_surfels.push_back(key);
                    SurfelNode* node = graph.get_node(key);
                    if (node) {
                        node->basin_id = static_cast<int>(i);
                    }
                }

                if (basins[j].peak_potential > basins[i].peak_potential) {
                    basins[i].peak_potential = basins[j].peak_potential;
                    basins[i].peak_surfel = basins[j].peak_surfel;
                }

                merged[j] = true;
            }
        }
    }

    // remove merged basins
    std::vector<FrontierBasin> remaining;
    for (size_t i = 0; i < basins.size(); ++i) {
        if (!merged[i]) {
            remaining.push_back(std::move(basins[i]));
        }
    }

    basins = std::move(remaining);
}

void GeodesicPotentialField::filter_small_basins(std::vector<FrontierBasin>& basins) {
    std::vector<FrontierBasin> filtered;

    const size_t min_basin_size = 4;
    const size_t max_basins = 50;

    for (auto& basin : basins) {
        if (basin.size() >= min_basin_size) {
            filtered.push_back(std::move(basin));
        }
    }

    if (filtered.size() > max_basins) {
        std::sort(filtered.begin(), filtered.end(),
            [](const FrontierBasin& a, const FrontierBasin& b) {
                return a.peak_potential > b.peak_potential;
            });
        filtered.resize(max_basins); // truncate 
    }

    basins = std::move(filtered);
}

void GeodesicPotentialField::reassign_ids(SurfaceGraph& graph, std::vector<FrontierBasin>& basins) {
    // reset and reassign with sequential ids (only valid this plan cycle)
    graph.reset_basins();
    for (size_t i = 0; i < basins.size(); ++i) {
        basins[i].id = static_cast<int>(i);

        for (const auto& key : basins[i].frontier_surfels) {
            SurfelNode* node = graph.get_node(key);
            if (node) {
                node->basin_id = static_cast<int>(i);
            }
        }
    }
}


void FrontierBasin::compute_geometry(const SurfaceGraph& graph) {
    // update frontier cluster/basin geometry
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