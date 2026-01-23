#include "sparse_surfel_mapping/planner/viewpoint_creator.hpp"

namespace sparse_surfel_map {

ViewpointCreator::ViewpointCreator() {}

std::vector<Viewpoint> ViewpointCreator::generate_viewpoints(const SurfaceGraph& graph, std::vector<FrontierPool>& pools, const Eigen::Vector3f& current_position) {

    // Generate viewpoints for all frontier pools
    std::vector<Viewpoint> all_candidates;
    for (auto& pool : pools) {
        std::vector<Viewpoint> pool_vps = generate_for_pool(graph, pool);
        all_candidates.insert(all_candidates.end(), pool_vps.begin(), pool_vps.end());
    }

    std::cout << "[Viewpoint] All candidates: " << all_candidates.size() << " viewpoints" << std::endl;

    // Select best viewpoints among candidates
    std::vector<Viewpoint> selected_viewpoints = select_best(all_candidates, pools, graph, current_position);
    return selected_viewpoints;
}

std::vector<Viewpoint> ViewpointCreator::generate_for_pool(const SurfaceGraph& graph, FrontierPool& pool) {
    std::vector<Viewpoint> pool_viewpoints;
    if (pool.empty()) return pool_viewpoints;

    if (pool.centroid.isZero()) {
        pool.compute_geometry(graph);
    }

    // Planar region -> single viewpoint
    // Non-planar region -> multiple viewpoint based on normal clustering
    
    if (pool.normal_variance < params_.normal_variance_th) {
        Viewpoint vp = create_viewpoint(pool.centroid, pool.mean_normal, pool.normal_variance, pool.id);
        
        if (is_valid(vp)) {
            vp.set_id(generate_id());
            vp.compute_visibility(*map_, true);
            pool_viewpoints.push_back(std::move(vp));
        }

        // try adjusting or?
    }
    else {
        auto clusters = cluster_by_normal(graph, pool.frontier_surfels, params_.max_viewpoints_per_pool);

        for (const auto& cluster : clusters) {
            if (cluster.empty()) continue;

            Eigen::Vector3f cluster_centroid = Eigen::Vector3f::Zero();
            Eigen::Vector3f cluster_normal = Eigen::Vector3f::Zero();

            for (const auto& key : cluster) {
                const SurfelNode* node = graph.get_node(key);
                if (node) {
                    cluster_centroid += node->position;
                    cluster_normal += node->normal;
                }
            }

            cluster_centroid /= static_cast<float>(cluster.size());
            
            if (cluster_normal.norm() > 1e-6f) {
                cluster_normal.normalize();
            }
            else {
                cluster_normal = Eigen::Vector3f::UnitZ();
            }

            // cluster variance
            float cluster_variance = 0.0f;
            for (const auto& key : cluster) {
                const SurfelNode* node = graph.get_node(key);
                if (node) {
                    float dot = node->normal.dot(cluster_normal);
                    float diff = 1.0f - dot;
                    cluster_variance += diff * diff;
                }
            }
            cluster_variance /= static_cast<float>(cluster.size());

            Viewpoint vp = create_viewpoint(cluster_centroid, cluster_normal, cluster_variance, pool.id);

            if (is_valid(vp)) {
                vp.set_id(generate_id());
                vp.compute_visibility(*map_, true);
                pool_viewpoints.push_back(std::move(vp));
            }
        }
    }

    return pool_viewpoints;
}

Viewpoint ViewpointCreator::create_viewpoint(const Eigen::Vector3f& centroid, const Eigen::Vector3f& normal, float normal_variance, int pool_id) {
    Eigen::Vector3f surface_dir = normal;
    surface_dir.z() *= 0.3f; // reduce vertical component of surface direction

    if (surface_dir.norm() < 0.1f) {
        // make optional -> dont return viewpoint for unobservable surface!
        std::cout << "CANNOT LOOK VERTICAL!" << std::endl;
        surface_dir = Eigen::Vector3f::UnitX(); // temporary!!!
    }

    surface_dir.normalize();
    float distance = params_.optimal_view_distance;
    distance += normal_variance * 0.5f; // for larger surface normal variance -> take small step back
    distance = std::clamp(distance, params_.min_view_distance, params_.max_view_distance); // bound distance

    Eigen::Vector3f position = centroid + surface_dir * distance;
    position.z() = std::max(params_.min_altitude, position.z()); // ensure minimum altitude for viewpoint

    Eigen::Vector3f look_dir = centroid - position;
    float yaw = std::atan2(look_dir.y(), look_dir.x());
    
    Viewpoint vp{position, yaw, params_.cam_config};
    vp.state().pool_id = pool_id;

    return vp;
}

std::vector<std::vector<VoxelKey>> ViewpointCreator::cluster_by_normal(const SurfaceGraph& graph, const std::vector<VoxelKey>& surfels, size_t max_clusters) {
    if (surfels.empty()) return {};

    if (surfels.size() <= max_clusters) {
        std::vector<std::vector<VoxelKey>> result;
        for (const auto& key : surfels) {
            result.push_back({key});
        }
        return result; // not enough surfels in pool to cluster
    }

    // K-means on normals
    std::vector<Eigen::Vector3f> centers;
    std::vector<std::vector<VoxelKey>> clusters(max_clusters);

    // Random initialization
    std::vector<size_t> indices(surfels.size());
    std::iota(indices.begin(), indices.end(), 0); // fill 0 -> n
    std::shuffle(indices.begin(), indices.end(), rng_); // random shuffle

    for (size_t i = 0; i < max_clusters; ++i) {
        const SurfelNode* node = graph.get_node(surfels[indices[i]]); // get random surfel
        if (node) {
            centers.push_back(node->normal); // initialize with random centers
        }
        else {
            centers.push_back(Eigen::Vector3f::UnitZ()); // invalid normal...
        }
    }

    // K-mean iters
    for (int iter = 0; iter < 10; ++iter) {
        for (auto& c : clusters) c.clear();

        // assign to nearest
        for (const auto& key : surfels) {
            const SurfelNode* node = graph.get_node(key);
            if (!node) continue;

            float best_sim = -2.0f;
            size_t best_c = 0;

            for (size_t c = 0; c < max_clusters; ++c) {
                float sim = node->normal.dot(centers[c]);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_c = c;
                }
            }

            clusters[best_c].push_back(key);
        }

        // update centers
        for (size_t c = 0; c < max_clusters; ++c) {
            if (clusters[c].empty()) continue;

            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            for (const auto& key : clusters[c]) {
                const SurfelNode* node = graph.get_node(key);
                if (node) {
                    sum += node->normal;
                }
            }

            if (sum.norm() > 1e-6f) {
                centers[c] = sum.normalized();
            }
        }
    }

    // remove empty clusters
    std::vector<std::vector<VoxelKey>> result;
    for (auto& c : clusters) {
        if (!c.empty()) {
            result.push_back(std::move(c));
        }
    }

    return result;
}

bool ViewpointCreator::is_valid(const Viewpoint& vp) const {
    if (!map_) return false;

    if (vp.position().z() < params_.min_altitude) return false;

    if (vp.is_in_collision(*map_, params_.collision_check_radius)) return false;

    if (coverage_tracker_ && coverage_tracker_->is_viewpoint_visited(vp)) return false;

    return true;
}

std::vector<Viewpoint> ViewpointCreator::select_best(std::vector<Viewpoint>& candidates, const std::vector<FrontierPool>& pools, const SurfaceGraph& graph, const Eigen::Vector3f& position) {
    const VoxelKeySet& all_frontiers = graph.frontier_set();
    if (candidates.empty() || all_frontiers.empty()) return {};

    size_t total = map_->num_valid_surfels();
    size_t total_frontiers = all_frontiers.size();
    size_t total_unknown = total - total_frontiers;

    // max potential for normalization (could be slightly off iff max_potential is in pool w. invalid viewpoint -> no cands)
    float max_potential = 0.0f;
    for (const auto& p : pools) {
        max_potential = std::max(max_potential, p.peak_potential);
    }
    if (max_potential < 1e-6f) max_potential = 1.0f;

    // Greedy selection with dynamic updates
    std::vector<Viewpoint> selected;
    VoxelKeySet covered_frontiers;
    VoxelKeySet covered_unknown;
    while (selected.size() < params_.max_viewpoints_total) {
        int best_idx = -1;
        float best_priority = -std::numeric_limits<float>::max();

        if (best_idx < 0) break; // no (more) valid candidates 
    }

}

float ViewpointCreator::compute_priority(const Viewpoint& vp, const FrontierPool& pool, const SurfaceGraph& graph, const Eigen::Vector3f& position) {
    
    size_t frontiers_visible = 0;
    size_t unknown_visisble = 0;
    for (const auto& key : vp.visible_voxels()) {
        const SurfelNode* node = graph.get_node(key);
        if (!node) continue;
        if (node->state == SurfelNode::State::FRONTIER) {
            frontiers_visible++;
        }
        else if (node->state == SurfelNode::State::UNKNOWN) {
            unknown_visisble++;
        }
    }

    size_t covered_visible = vp.visible_voxels().size() - frontiers_visible - unknown_visisble; // already marked visible
    

}



} // namespace