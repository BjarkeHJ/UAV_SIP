#include "sparse_surfel_mapping/planner/viewpoint_generator.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace sparse_surfel_map {

ViewpointGenerator::ViewpointGenerator()
    : config_()
    , map_(nullptr)
    , collision_checker_(nullptr)
    , coverage_tracker_(nullptr)
    , frontier_finder_()
    , rng_(std::random_device{}())
{}

ViewpointGenerator::ViewpointGenerator(const InspectionPlannerConfig& config)
    : config_(config)
    , map_(nullptr)
    , collision_checker_(nullptr)
    , coverage_tracker_(nullptr)
    , frontier_finder_()
    , rng_(std::random_device{}())
{}

void ViewpointGenerator::set_map(const SurfelMap* map) {
    map_ = map;
    frontier_finder_.set_map(map);
}

std::vector<Viewpoint> ViewpointGenerator::generate_next_viewpoints(const Eigen::Vector3f& position, float yaw) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    if (!map_) return {};

    // Reset debug counters
    last_frontiers_found_ = 0;
    last_clusters_formed_ = 0;
    last_candidates_generated_ = 0;
    last_candidates_in_collision_ = 0;

    // Get globally observed voxels
    VoxelKeySet already_covered;
    if (coverage_tracker_) {
        already_covered = coverage_tracker_->observed_voxels();
    }

    // Compute seed observation from current
    Viewpoint seed_observation(position, yaw, config_.camera);
    seed_observation.compute_visibility(*map_, false);

    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Seed observation at (" 
                  << position.transpose() << ") yaw=" 
                  << yaw * 180.0f / M_PI << "°" << std::endl;
        std::cout << "  Sees " << seed_observation.num_visible() << " voxels" << std::endl;
    }

    // Build the chain
    std::vector<Viewpoint> chain = build_chain(already_covered, seed_observation.visible_voxels(), position);

    auto t_end = std::chrono::high_resolution_clock::now();
    last_generation_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Generated chain of " << chain.size() 
                  << " viewpoints in " << last_generation_time_ms_ << " ms" << std::endl;
    }

    return chain;
}

std::vector<Viewpoint> ViewpointGenerator::generate_continuation(const Viewpoint& start_viewpoint) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // NOTE: THIS COULD BE THE ONYL GENERATE FUNCTION (ASSUMING A START VIEWPOINT CAN ALWAYS BE GENERATED FROM DRONE)

    if (!map_) return {};

    last_frontiers_found_ = 0;
    last_clusters_formed_ = 0;
    last_candidates_generated_ = 0;
    last_candidates_in_collision_ = 0;

    // Get globally observed voxels (includes the reached viewpoint)
    VoxelKeySet already_covered;
    if (coverage_tracker_) {
        already_covered = coverage_tracker_->observed_voxels();
    }

    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Continuing from VP " << start_viewpoint.id()
                  << " at (" << start_viewpoint.position().transpose() << ")" << std::endl;
    }

    // Build chain starting from reached viewpoint
    std::vector<Viewpoint> chain = build_chain(already_covered, start_viewpoint.visible_voxels(), start_viewpoint.position());

    auto t_end = std::chrono::high_resolution_clock::now();
    last_generation_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Continuation chain: " << chain.size() 
                  << " viewpoints in " << last_generation_time_ms_ << " ms" << std::endl;
    }

    return chain;
}

std::vector<Viewpoint> ViewpointGenerator::build_chain(const VoxelKeySet& initial_coverage, const VoxelKeySet& seed_visible, const Eigen::Vector3f& seed_position) {
    
    std::vector<Viewpoint> chain;
    const auto& vp_config = config_.viewpoint;
    const size_t max_chain_length = vp_config.max_chain_length;

    // Initialize cumulative coverage: already covered + seed visible
    VoxelKeySet cumulative_coverage = initial_coverage;
    for (const auto& key : seed_visible) {
        cumulative_coverage.insert(key);
    }

    // Compute initial expansion center from seed
    Eigen::Vector3f expansion_center = compute_expansion_center(seed_visible, initial_coverage);
    Eigen::Vector3f previous_position = seed_position;

    if (config_.debug_output) {
        std::cout << "  Initial expansion center: (" << expansion_center.transpose() << ")" << std::endl;
        std::cout << "  Cumulative coverage: " << cumulative_coverage.size() << " voxels" << std::endl;
    }

    // Iteratively build the chain
    for (size_t i = 0; i < max_chain_length; ++i) {
        if (config_.debug_output) {
            std::cout << "\n  === Chain Step " << (i + 1) << " ===" << std::endl;
            std::cout << "  Expansion center: (" << expansion_center.transpose() << ")" << std::endl;
        }

        // Find frontiers from coverage boundary (set-based, centered on expansion)
        std::vector<FrontierSurfel> frontiers = frontier_finder_.find_frontiers_from_coverage(
            cumulative_coverage,
            expansion_center,
            vp_config.max_expansion_radius);

        last_frontiers_found_ += frontiers.size();

        if (config_.debug_output) {
            std::cout << "  Found " << frontiers.size() << " frontiers within radius " 
                      << vp_config.max_expansion_radius << "m" << std::endl;
        }

        if (frontiers.empty()) {
            if (config_.debug_output) {
                std::cout << "  No frontiers found - chain terminated" << std::endl;
            }
            break;
        }

        // Cluster frontiers
        std::vector<FrontierCluster> clusters = frontier_finder_.cluster_frontiers(
            frontiers,
            vp_config.frontier_cluster_radius,
            vp_config.min_cluster_size);

        last_clusters_formed_ += clusters.size();

        if (config_.debug_output) {
            std::cout << "  Formed " << clusters.size() << " clusters" << std::endl;
        }

        if (clusters.empty()) {
            if (config_.debug_output) {
                std::cout << "  No valid clusters - chain terminated" << std::endl;
            }
            break;
        }

        // Generate viewpoint candidates for clusters
        std::vector<Viewpoint> candidates = generate_candidates_for_clusters(
            clusters, 
            cumulative_coverage);

        if (config_.debug_output) {
            std::cout << "  Generated " << candidates.size() << " candidates" << std::endl;
        }

        if (candidates.empty()) {
            if (config_.debug_output) {
                std::cout << "  No valid candidates - chain terminated" << std::endl;
            }
            break;
        }

        // Select the best candidate for this chain step
        Viewpoint* selected = select_best_for_chain(
            candidates,
            cumulative_coverage,
            previous_position,
            chain);

        if (!selected) {
            if (config_.debug_output) {
                std::cout << "  No acceptable candidate - chain terminated" << std::endl;
            }
            break;
        }

        // Finalize the selected viewpoint
        selected->compute_visibility(*map_, true);
        selected->compute_coverage_score(cumulative_coverage, vp_config);
        selected->set_status(ViewpointStatus::PLANNED);

        if (config_.debug_output) {
            std::cout << "  Selected VP " << selected->id() 
                      << " at (" << selected->position().transpose() << ")" << std::endl;
            std::cout << "    visible=" << selected->num_visible()
                      << ", new=" << selected->num_new_coverage()
                      << ", score=" << selected->total_score() << std::endl;
        }

        // Store coverage before adding newly observed
        VoxelKeySet coverage_before_this = cumulative_coverage;
        
        // Update cumulative coverage
        for (const auto& key : selected->visible_voxels()) {
            cumulative_coverage.insert(key);
        }

        // Compute expansion center with coverage_before and selected's visible (to identify newly found)
        expansion_center = compute_expansion_center(
            selected->visible_voxels(),
            coverage_before_this);

        previous_position = selected->position();

        // Add to chain
        chain.push_back(std::move(*selected));

        if (config_.debug_output) {
            std::cout << "  New expansion center: (" << expansion_center.transpose() << ")" << std::endl;
            std::cout << "  Cumulative coverage now: " << cumulative_coverage.size() << " voxels" << std::endl;
        }
    }

    return chain;
}

Eigen::Vector3f ViewpointGenerator::compute_expansion_center(const VoxelKeySet& visible_voxels, const VoxelKeySet& already_covered) const {
    if (visible_voxels.empty()) {
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
        frontier_finder_.compute_cluster_view_suggestion(
            cluster,
            vp_config.optimal_view_distance,
            vp_config.min_view_distance);

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
    if (is_position_valid(cluster.suggested_view_position)) {
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
            
            if (!is_position_valid(pos)) {
                last_candidates_in_collision_++;
                continue;
            }

            float yaw = compute_yaw_to_target(pos, cluster.centroid);
            
            Viewpoint test_vp(pos, yaw, config_.camera);
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

bool ViewpointGenerator::is_position_valid(const Eigen::Vector3f& position) const {
    const auto& vp_config = config_.viewpoint;

    // Minimum altitude
    if (position.z() < 0.5f) return false;

    // Collision check
    if (collision_checker_ && collision_checker_->is_sphere_in_collision(position)) {
        return false;
    }

    // Minimum distance to obstacles
    if (collision_checker_) {
        float dist = collision_checker_->distance_to_nearest_obstacle(
            position, vp_config.min_view_distance);
        if (dist < vp_config.min_view_distance * 0.5f) {
            return false;
        }
    }

    // Check if already visited
    if (coverage_tracker_) {
        Viewpoint temp_vp(position, 0.0f, config_.camera);
        if (coverage_tracker_->is_viewpoint_visited(temp_vp)) {
            return false;
        }
    }

    return true;
}

void ViewpointGenerator::score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const FrontierCluster& target_cluster) {
    const auto& vp_config = config_.viewpoint;

    vp.compute_coverage_score(already_covered, vp_config);
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
    float overlap = vp.overlap_score();
    if (overlap < vp_config.min_overlap_ratio) {
        vp.state().total_score *= 0.5f;
    } else if (overlap > vp_config.max_overlap_ratio) {
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

