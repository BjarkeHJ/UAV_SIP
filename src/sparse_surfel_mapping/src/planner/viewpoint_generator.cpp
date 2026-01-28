#include "sparse_surfel_mapping/planner/viewpoint_generator.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace sparse_surfel_map {

ViewpointGenerator::ViewpointGenerator()
    : config_()
    , map_(nullptr)
    , coverage_tracker_(nullptr)
    , frontier_finder_()
{}

ViewpointGenerator::ViewpointGenerator(const InspectionPlannerConfig& config)
    : config_(config)
    , map_(nullptr)
    , coverage_tracker_(nullptr)
    , frontier_finder_()
{}

void ViewpointGenerator::set_map(const SurfelMap* map) {
    map_ = map;
    frontier_finder_.set_map(map);
}

std::vector<Viewpoint> ViewpointGenerator::generate_viewpoints(const Viewpoint& start_viewpoint, size_t n_new, const VoxelKeySet& current_obs) {
    if (!map_) return {};
    auto t_start = std::chrono::high_resolution_clock::now();

    // Create observation initial (visited + current plan observations)
    VoxelKeySet already_covered = coverage_tracker_->observed_voxels();
    if (!current_obs.empty()) {
        for (const auto& key : current_obs) {
            already_covered.insert(key); // estimate from previous viewpoints if chaining
        }
    }

    // Build chain of viewpoints from observation prior
    std::vector<Viewpoint> new_viewpoints = build_chain(already_covered, start_viewpoint.visible_voxels(), start_viewpoint.position(), n_new);
    auto t_end = std::chrono::high_resolution_clock::now();
    last_generation_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Viewpoint Chain: " << new_viewpoints.size() << " viewpoints in " << last_generation_time_ms_ << " ms" << std::endl;
    }

    return new_viewpoints;
}

std::vector<Viewpoint> ViewpointGenerator::build_chain(const VoxelKeySet& initial_coverage, const VoxelKeySet& seed_visible, const Eigen::Vector3f& seed_position, size_t n_new) {
    std::vector<Viewpoint> chain;
    const auto& vp_config = config_.viewpoint;

    // Initialize cumulative coverage: already covered + seed visible
    VoxelKeySet cumulative_coverage = initial_coverage;
    for (const auto& key : seed_visible) {
        cumulative_coverage.insert(key);
    }

    // Compute initial expansion center from seed
    Eigen::Vector3f expansion_center = compute_expansion_center(seed_visible, initial_coverage);
    Eigen::Vector3f previous_position = seed_position;

    // Iteratively build the chain based on a desired number of new viewpoints
    for (size_t i = 0; i < n_new; ++i) {
        // Find frontiers from coverage boundary (set-based, centered on expansion)
        std::vector<FrontierSurfel> frontiers = frontier_finder_.find_frontiers_from_coverage(cumulative_coverage, expansion_center, vp_config.max_expansion_radius);

        last_frontiers_found_ += frontiers.size();
        if (frontiers.empty()) break;

        // Cluster frontiers
        std::vector<FrontierCluster> clusters = frontier_finder_.cluster_frontiers(frontiers, vp_config.frontier_wavefront_width, vp_config.frontier_cluster_radius, vp_config.min_cluster_size);
        last_clusters_formed_ += clusters.size();

        if (clusters.empty()) break;

        // Generate viewpoint candidates for clusters
        std::vector<Viewpoint> candidates = generate_candidates_for_clusters(clusters, cumulative_coverage);

        std::cout << "Number of viewpoint candidates: " << candidates.size() << std::endl; 

        if (candidates.empty()) break;

        // Select the best candidate for this chain step
        Viewpoint* selected = select_best_for_chain(candidates, cumulative_coverage, previous_position, chain);

        if (!selected) break;

        // Finalize the selected viewpoint
        selected->compute_visibility(*map_, true);
        selected->compute_coverage_score(cumulative_coverage);
        selected->set_status(ViewpointStatus::PLANNED);

        // Compute expansion center with coverage_before and selected's visible (to identify newly found)
        expansion_center = compute_expansion_center(selected->visible_voxels(), cumulative_coverage);

        // Update cumulative coverage for next iteration
        for (const auto& key : selected->visible_voxels()) {
            cumulative_coverage.insert(key);
        }
        
        // Update position
        previous_position = selected->position();

        // Add to chain
        chain.push_back(std::move(*selected));
    }

    return chain;
}

Eigen::Vector3f ViewpointGenerator::compute_expansion_center(const VoxelKeySet& visible_voxels, const VoxelKeySet& already_covered) const {
    if (visible_voxels.empty()) {
        std::cout << "COMPUTE EXPANSION CENTER: NO VISIBLE VOXELS!" << std::endl;
        return Eigen::Vector3f::Zero();
    }

    // First try: mean of NEW voxels (visible but not in already_covered)
    Eigen::Vector3f sum_new = Eigen::Vector3f::Zero();
    size_t count_new = 0;

    for (const auto& key : visible_voxels) {
        if (already_covered.count(key) == 0) {
            sum_new += key_to_position(key);
            count_new++;
        }
    }

    if (count_new > 0) {
        return sum_new / static_cast<float>(count_new);
    }

    // Fallback: mean of ALL visible voxels
    Eigen::Vector3f sum_all = Eigen::Vector3f::Zero();
    for (const auto& key : visible_voxels) {
        sum_all += key_to_position(key);
    }

    return sum_all / static_cast<float>(visible_voxels.size());
}

std::vector<Viewpoint> ViewpointGenerator::generate_candidates_for_clusters(const std::vector<FrontierCluster>& clusters, const VoxelKeySet& already_covered) {
    std::vector<Viewpoint> candidates;
    const auto& vp_config = config_.viewpoint;

    for (auto cluster : clusters) {  // Copy to allow modification
        frontier_finder_.compute_cluster_view_suggestion(cluster, vp_config.optimal_view_distance);

        Viewpoint vp = generate_viewpoint_for_cluster(cluster, already_covered);

        if (vp.num_visible() > 0) {
            candidates.push_back(std::move(vp));
            last_candidates_generated_++;
        }
    }

    return candidates;
}

Viewpoint ViewpointGenerator::generate_viewpoint_for_cluster(const FrontierCluster& cluster, const VoxelKeySet& already_covered) {
    const auto& vp_config = config_.viewpoint;
    
    Viewpoint vp(cluster.suggested_view_position, cluster.suggested_yaw, config_.camera);
    vp.set_id(generate_id());

    // Try suggested position first
    if (is_viewpoint_valid(vp)) {
        vp.compute_visibility(*map_, true);
        if (vp.num_visible() > 0) {
            score_viewpoint(vp, already_covered, cluster);
            return vp;
        }
    } else {
        last_candidates_in_collision_++;
    }

    // Try different distances along view direction
    Eigen::Vector3f view_dir = cluster.mean_normal;
    view_dir.z() *= 0.3f;
    if (view_dir.norm() > 0.01f) {
        view_dir.normalize();
    } else {
        view_dir = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    }

    Viewpoint best_vp;
    size_t best_visible = 0;

    for (float dist = vp_config.optimal_view_distance; dist >= vp_config.min_view_distance; dist -= 0.5f) {
        Eigen::Vector3f test_pos = cluster.centroid + view_dir * dist;

        std::vector<Eigen::Vector3f> offsets = {
            Eigen::Vector3f::Zero(),
            Eigen::Vector3f(0.5f, 0.0f, 0.0f),
            Eigen::Vector3f(-0.5f, 0.0f, 0.0f),
            Eigen::Vector3f(0.0f, 0.5f, 0.0f),
            Eigen::Vector3f(0.0f, -0.5f, 0.0f),
            Eigen::Vector3f(0.0f, 0.0f, 0.3f),
            Eigen::Vector3f(0.0f, 0.0f, -0.3f)
        };

        for (const auto& offset : offsets) {
            Eigen::Vector3f pos = test_pos + offset;
            float yaw = compute_yaw_to_target(pos, cluster.centroid);
            Viewpoint test_vp(pos, yaw, config_.camera);
            if (!is_viewpoint_valid(test_vp)) {
                last_candidates_in_collision_++;
                continue;
            }
            test_vp.set_id(generate_id());
            test_vp.compute_visibility(*map_, true);

            if (test_vp.num_visible() > best_visible) {
                best_visible = test_vp.num_visible();
                best_vp = std::move(test_vp);
            }
        }

        if (best_visible > 0) {
            break;
        }
    }

    if (best_visible > 0) {
        score_viewpoint(best_vp, already_covered, cluster);
        return best_vp;
    }

    return vp;
}

Viewpoint* ViewpointGenerator::select_best_for_chain(std::vector<Viewpoint>& candidates, const VoxelKeySet& cumulative_coverage, const Eigen::Vector3f& previous_position, const std::vector<Viewpoint>& existing_chain) {
    if (candidates.empty()) return nullptr;

    const auto& vp_config = config_.viewpoint;

    // Score each candidate
    for (auto& vp : candidates) {
        // New coverage relative to CUMULATIVE coverage
        size_t new_coverage = 0;
        for (const auto& key : vp.visible_voxels()) {
            if (cumulative_coverage.count(key) == 0) {
                new_coverage++;
            }
        }

        float new_ratio = vp.num_visible() > 0 
            ? static_cast<float>(new_coverage) / vp.num_visible() 
            : 0.0f;

        // Distance score: prefer closer for chain connectivity
        float dist = (vp.position() - previous_position).norm();
        float max_reasonable_dist = vp_config.max_view_distance * 2.0f;
        float dist_score = 1.0f - std::min(dist / max_reasonable_dist, 1.0f);

        // Combined score
        float chain_score = new_ratio * 0.7f + dist_score * 0.3f;

        // Penalty for redundancy with existing chain
        for (const auto& existing : existing_chain) {
            float existing_dist = (vp.position() - existing.position()).norm();
            if (existing_dist < vp_config.frontier_cluster_radius) {
                chain_score *= 0.2f;
                break;
            }
        }

        vp.state().total_score = chain_score;
        vp.state().coverage_score = new_ratio;
    }

    // Sort by score
    std::sort(candidates.begin(), candidates.end(),
        [](const Viewpoint& a, const Viewpoint& b) {
            return a.total_score() > b.total_score();
        });

    // Select best acceptable candidate
    for (auto& vp : candidates) {
        if (vp.coverage_score() < vp_config.min_new_coverage_ratio) {
            continue;
        }
        if (vp.total_score() < 0.1f) {
            continue;
        }
        return &vp;
    }

    return nullptr;
}

bool ViewpointGenerator::is_viewpoint_valid(const Viewpoint& vp) const {
    const auto& vp_config = config_.viewpoint;

    // Minimum altitude
    if (vp.state().position.z() < 0.5f) return false;
    // Collision in min_view_distance radius?
    // if (vp.is_in_collision(*map_, vp_config.min_view_distance)) return false;
    if (vp.is_in_collision(*map_, config_.collision.inflation_radius())) return false;
    // Similar to already visited viewpoint?
    if (coverage_tracker_->is_viewpoint_visited(vp)) return false;
    return true;
}

void ViewpointGenerator::score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const FrontierCluster& target_cluster) {
    const auto& vp_config = config_.viewpoint;

    vp.compute_coverage_score(already_covered);
    float coverage_score = vp.coverage_score();

    // Frontier alignment score
    float frontier_score = 0.0f;
    size_t cluster_surfels_visible = 0;
    for (const auto& fs : target_cluster.surfels) {
        if (vp.visible_voxels().count(fs.key) > 0) {
            cluster_surfels_visible++;
        }
    }
    if (!target_cluster.surfels.empty()) {
        frontier_score = static_cast<float>(cluster_surfels_visible) / target_cluster.surfels.size();
    }

    // Distance score
    float dist = (vp.position() - target_cluster.centroid).norm();
    float dist_score = 0.0f;
    if (dist >= vp_config.min_view_distance && dist <= vp_config.max_view_distance) {
        float dist_from_optimal = std::abs(dist - vp_config.optimal_view_distance);
        dist_score = 1.0f - (dist_from_optimal / vp_config.optimal_view_distance);
        dist_score = std::max(0.0f, dist_score);
    }
    
    vp.state().total_score = 
        vp_config.new_coverage_weight * coverage_score +
        vp_config.frontier_priority_weight * frontier_score +
        vp_config.distance_weight * dist_score;

    // Overlap penalty
    float overlap = 1 - vp.coverage_score();
    if (overlap < vp_config.target_overlap_ratio * 0.5) {
        vp.state().total_score *= 0.5f;
    } else if (overlap > vp_config.target_overlap_ratio * 2.0f) {
        vp.state().total_score *= 0.7f;
    }
}

float ViewpointGenerator::compute_yaw_to_target(const Eigen::Vector3f& from_pos, const Eigen::Vector3f& target_pos) const {
    Eigen::Vector3f direction = target_pos - from_pos;
    return std::atan2(direction.y(), direction.x());
}

Eigen::Vector3f ViewpointGenerator::key_to_position(const VoxelKey& key) const {
    if (!map_) return Eigen::Vector3f::Zero();
    
    const float voxel_size = map_->voxel_size();
    return Eigen::Vector3f(
        (key.x + 0.5f) * voxel_size,
        (key.y + 0.5f) * voxel_size,
        (key.z + 0.5f) * voxel_size
    );
}

} // namespace

