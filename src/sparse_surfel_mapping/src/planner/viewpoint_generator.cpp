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
{}

ViewpointGenerator::ViewpointGenerator(const InspectionPlannerConfig& config)
    : config_(config)
    , map_(nullptr)
    , coverage_tracker_(nullptr)
{}

std::vector<Viewpoint> ViewpointGenerator::generate_viewpoints(const Viewpoint& start_viewpoint, size_t n_new, const VoxelKeySet& current_obs) {
    if (!map_ || !coverage_tracker_) return {};
    
    auto t_start = std::chrono::high_resolution_clock::now();

    last_frontiers_found_ = 0;
    last_clusters_formed_ = 0;
    last_candidates_generated_ = 0;
    last_candidates_in_collision_ = 0;

    // check if this is initial planning (no coverage to expand on)
    bool is_initial = coverage_tracker_->observed_surfels().empty() && current_obs.empty();

    std::vector<Viewpoint> result;

    // if (is_init_) {
    if (is_initial) {
        auto initial_vp = generate_initial_viewpoint(start_viewpoint.position());
        if (initial_vp) {
            result.push_back(std::move(*initial_vp));

            std::cout << "[ViewpointGenerator] Generated initial viewpoint!" << std::endl;
        }
        // is_init_ = false;
    }
    else {
        // coverage frontier-based generation (chaining)
        result = build_chain(
            coverage_tracker_->observed_surfels(),
            start_viewpoint.visible_voxels(),
            start_viewpoint.position(),
            start_viewpoint.yaw(),
            n_new
        );
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    last_generation_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Viewpoint Chain: " << result.size() << " viewpoints in " << last_generation_time_ms_ << " ms" << std::endl;
    }

    return result;
}

std::optional<Viewpoint> ViewpointGenerator::generate_initial_viewpoint(const Eigen::Vector3f& drone_position) {
    if (!map_) return std::nullopt;

    const auto& vp_config = config_.viewpoint;

    float search_radius = vp_config.max_view_distance * 3.0f;
    auto nearby_surfels = map_->query_surfels_in_radius(drone_position, search_radius);

    if (nearby_surfels.empty()) {
        search_radius *= 2.0f; // expand
        nearby_surfels = map_->query_surfels_in_radius(drone_position, search_radius);

        if (nearby_surfels.empty()) {
            std::cout << "[ViewpointGenerator] Error: No surfels found for initial viewpoint!" << std::endl;
            return std::nullopt;
        }
    }

    float min_dist = std::numeric_limits<float>::max();
    const Surfel* closest = nullptr;

    for (const auto& surfel_ref : nearby_surfels) {
        float dist = (surfel_ref.get().mean() - drone_position).norm();
        if (dist < min_dist) {
            min_dist = dist;
            closest = &surfel_ref.get();
        }
    }

    if (!closest) return std::nullopt;

    // Generate viewpoint
    Eigen::Vector3f surfel_pos = closest->mean();
    Eigen::Vector3f surfel_normal = closest->normal();

    Eigen::Vector3f surface_dir = surfel_normal;
    surface_dir.z() *= 0.3f;
    if (surface_dir.norm() < 0.1f) {
        surface_dir = drone_position - surfel_pos;
        surface_dir.z() *= 0.3f;
        if (surface_dir.norm() < 0.1f) {
            surface_dir = Eigen::Vector3f::UnitX();
        }
    }

    surface_dir.normalize();

    Eigen::Vector3f vp_pos = surfel_pos + surface_dir * vp_config.optimal_view_distance;
    if (vp_pos.z() < 0.3f) {
        vp_pos.z() = 0.5f;
    }

    float vp_yaw = std::atan2(-surface_dir.y(), -surface_dir.x());

    Viewpoint vp(vp_pos, vp_yaw, config_.camera);
    vp.set_id(generate_id());
    if (is_viewpoint_valid(vp)) {
        vp.set_status(ViewpointStatus::PLANNED);
        return vp;
    }
    else {
        vp_pos += surface_dir * (vp_config.max_view_distance - vp_config.optimal_view_distance); // move away from surface to max_view_dist
        vp.set_position(vp_pos);
        if (is_viewpoint_valid(vp)) {
            return vp;
        }
    }

    std::cout << "[ViewpointGenerator] Error: Could not generate initial viewpoint (obstructed)!" << std::endl;
    return std::nullopt;
}

std::vector<Viewpoint> ViewpointGenerator::build_chain(const VoxelKeySet& initial_coverage, const VoxelKeySet& seed_visible, const Eigen::Vector3f& seed_position, float seed_yaw, size_t n_new) {
    std::vector<Viewpoint> chain;

    // TEMPORARY
    cands_.clear();

    // Initialize cumulative coverage: already (actual) covered + seed_visible
    VoxelKeySet cumulative_coverage = initial_coverage; 
    for (const auto& key : seed_visible) {
        cumulative_coverage.insert(key);
    }

    // Working copy of frontier set for speculative updates
    VoxelKeySet working_frontier = coverage_tracker_->coverage_frontiers();

    if (working_frontier.empty() && !cumulative_coverage.empty()) {
        std::cout << "[ViewpointGenerator] Error: Not able to start chain - no frontiers!" << std::endl;
        return chain;
    }

    last_frontiers_found_ = working_frontier.size();
    
    Eigen::Vector3f previous_position = seed_position;
    float previous_yaw = seed_yaw;
    
    for (size_t i = 0; i < n_new; ++i) {
        if (working_frontier.empty()) {
            std::cout << "[ViewpointGenerator] No more frontiers, stopping chain..." << std::endl;
            break;
        }
    
        std::vector<VoxelKeySet> clusters = cluster_frontiers(working_frontier);
        last_clusters_formed_ = clusters.size();

        if (clusters.empty()) {
            std::cout << "[ViewpointGenerator] No valid clusters formed!" << std::endl;

            break;
        }

        std::vector<std::pair<float, size_t>> cluster_priorities;
        cluster_priorities.reserve(clusters.size());

        for (size_t c = 0; c < clusters.size(); ++c) {
            float priority = compute_cluster_priority(clusters[c], previous_position, previous_yaw);
            cluster_priorities.emplace_back(priority, c);
        }

        std::sort(cluster_priorities.begin(), cluster_priorities.end(), 
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        std::vector<Viewpoint> candidates;

        for (const auto& [priority, cluster_idx] : cluster_priorities) {
            Viewpoint vp = generate_viewpoint_for_cluster(clusters[cluster_idx]);

            if (vp.num_visible() > 0) {
                // score_viewpoint(vp, cumulative_coverage, clusters[cluster_idx]);
                cands_.push_back(vp); // copy TEMPORARY
                candidates.push_back(std::move(vp));
                last_candidates_generated_++;
            }
        }

        if (candidates.empty()) {
            std::cout << "[ViewpointGenerator] No valid candidates generated!" << std::endl;
            break;
        }

        Viewpoint* selected = select_best_for_chain(candidates, cumulative_coverage, previous_position, chain);

        if (!selected) {
            std::cout << "[ViewpointGenerator] No acceptable candidate found!" << std::endl;
            break;
        }

        selected->set_status(ViewpointStatus::PLANNED);

        // extract newly covered and add to cumulative
        VoxelKeySet newly_covered;
        for (const auto& key : selected->visible_voxels()) {
            if (cumulative_coverage.count(key) == 0) {
                newly_covered.insert(key);
                cumulative_coverage.insert(key);
            }
        }

        // update workign frontier set
        CoverageTracker::update_frontier_set(working_frontier, newly_covered, cumulative_coverage, *map_);

        previous_position = selected->position();
        previous_yaw = selected->yaw();

        chain.push_back(std::move(*selected));
    }

    return chain;
}

std::vector<VoxelKeySet> ViewpointGenerator::cluster_frontiers(const VoxelKeySet& frontier_keys) const {
    std::vector<VoxelKeySet> clusters;

    if (frontier_keys.empty() || !map_) return clusters;

    const float normal_th = std::cos(config_.camera.max_incidence_angle_deg * M_PI /180.0f);
    const size_t min_cluster_size = 5;

    VoxelKeySet remaining = frontier_keys;

    // region growing BFS
    while (!remaining.empty()) {
        auto seed_it = remaining.begin();
        VoxelKey seed_key = *seed_it;
        remaining.erase(seed_it);

        auto seed_voxel = map_->get_voxel(seed_key);
        if (!seed_voxel || !seed_voxel->get().has_valid_surfel()) continue;

        VoxelKeySet cluster;
        cluster.insert(seed_key);
        std::queue<VoxelKey> queue;
        queue.push(seed_key);

        while(!queue.empty()) {
            VoxelKey current_key = queue.front();
            queue.pop();

            auto current_voxel = map_->get_voxel(current_key);
            if (!current_voxel || !current_voxel->get().has_valid_surfel()) continue;
            Eigen::Vector3f current_normal = current_voxel->get().surfel().normal();

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        VoxelKey nb_key{current_key.x + dx, current_key.y + dy, current_key.z + dz};

                        if (remaining.count(nb_key) == 0) continue; // already handled

                        auto nb_voxel = map_->get_voxel(nb_key);
                        if (!nb_voxel || !nb_voxel->get().has_valid_surfel()) continue;
                        Eigen::Vector3f nb_normal = nb_voxel->get().surfel().normal();

                        if (current_normal.dot(nb_normal) >= normal_th) {
                            cluster.insert(nb_key);
                            queue.push(nb_key);
                            remaining.erase(nb_key);
                        }
                    }
                }
            }
        }

        // Significant size?
        if (cluster.size() >= min_cluster_size) {
            clusters.push_back(std::move(cluster));
        }
    }

    return clusters;
}

Eigen::Vector3f ViewpointGenerator::compute_cluster_centroid(const VoxelKeySet& cluster) const {
    if (cluster.empty() || !map_) return Eigen::Vector3f::Zero();

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    size_t count = 0;
    for (const auto& key : cluster) {
        auto voxel_opt = map_->get_voxel(key);
        if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
            sum += voxel_opt->get().surfel().mean();
            count++;
        }
    }

    if (count > 0) {
        sum /= static_cast<float>(count);
    }

    return sum;
}

Eigen::Vector3f ViewpointGenerator::compute_cluster_mean_normal(const VoxelKeySet& cluster) const {
    if (cluster.empty() || !map_) return Eigen::Vector3f::Zero();

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();

    for (const auto& key : cluster) {
        auto voxel_opt = map_->get_voxel(key);
        if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
            sum += voxel_opt->get().surfel().normal();
        }
    }

    float norm = sum.norm();
    if (norm > 1e-6f) {
        sum /= norm; 
    }
    else {
        sum = Eigen::Vector3f::Zero();
    }

    return sum;
}

float ViewpointGenerator::compute_cluster_priority(const VoxelKeySet& cluster, const Eigen::Vector3f& drone_pos, float drone_yaw) const {
    if (cluster.empty()) return 0.0f;

    Eigen::Vector3f centroid = compute_cluster_centroid(cluster);
    float distance = (centroid - drone_pos).norm();

    float required_yaw = std::atan2(centroid.y() - drone_pos.y(), centroid.x() - drone_pos.x());
    float yaw_change = angular_distance(drone_yaw, required_yaw);

    const float yaw_w = 0.5f; // weight for yaw cost

    float cost = distance + yaw_change * yaw_w;

    // bonus for large clusters
    float size_bonus = std::log(static_cast<float>(cluster.size()) + 1.0f); // * 0.1f??

    return 1.0f / (cost + 0.1f) + size_bonus;
}

Viewpoint ViewpointGenerator::generate_viewpoint_for_cluster(const VoxelKeySet& cluster) {
    const auto& vp_config = config_.viewpoint;

    Eigen::Vector3f centroid = compute_cluster_centroid(cluster);
    Eigen::Vector3f normal = compute_cluster_mean_normal(cluster);

    Eigen::Vector3f surface_dir = normal;
    surface_dir.z() *= 0.3;
    if (surface_dir.norm() < 0.1f) {
        surface_dir = Eigen::Vector3f::UnitZ(); // lookup (invalid viewpoint)
    }
    surface_dir.normalize();

    Eigen::Vector3f vp_pos = centroid + surface_dir * vp_config.optimal_view_distance;
    float vp_yaw = compute_yaw_to_target(vp_pos, centroid);
    
    Viewpoint vp(vp_pos, vp_yaw, config_.camera);
    vp.set_id(generate_id());

    if (is_viewpoint_valid(vp)) {
        vp.compute_visibility(*map_, true);
        if (vp.num_visible() > 0) return vp;
    }
    else {
        vp.set_position(vp_pos + surface_dir * (vp_config.max_view_distance - vp_config.optimal_view_distance)); // move back
        if (is_viewpoint_valid(vp)) {
            vp.compute_visibility(*map_, true);
            if (vp.num_visible() > 0) return vp;
        }
    }

    std::cout << "[ViewpointGenerator] Not able to generate viewpoint for cluster!" << std::endl;
    
    // TEMPORARY: returns bad viewpoint
    return Viewpoint(Eigen::Vector3f::Zero(), 0.0f, config_.camera);
}

Viewpoint* ViewpointGenerator::select_best_for_chain(std::vector<Viewpoint>& candidates, const VoxelKeySet& cumulative_coverage, const Eigen::Vector3f& previous_position, const std::vector<Viewpoint>& existing_chain) {
    if (candidates.empty()) return nullptr;

    const auto& vp_config = config_.viewpoint;

    // score each candidate
    for (auto& vp : candidates) {
        size_t new_coverage = 0;
        for (const auto& key : vp.visible_voxels()) {
            if (cumulative_coverage.count(key) == 0) {
                new_coverage++;
            }
        }

        float new_ratio = vp.num_visible() > 0 ? static_cast<float>(new_coverage) / vp.num_visible() : 0.0f;

        float dist = (vp.position() - previous_position).norm();
        float max_reasonable_dist = vp_config.max_view_distance * 3.0f;
        float dist_score = 1.0f - std::min(dist / max_reasonable_dist, 1.0f);
        float chain_score = new_ratio * 0.7f + dist_score * 0.3f;

        // penalize redundancy in existing chain
        for (const auto& existing : existing_chain) {
            // min_view_dist and 10 degree similarity
            if (vp.is_similar_to(existing, vp_config.min_view_distance, 10.0f)) {
                chain_score *= 0.2f; 
                break;
            }
        }

        vp.state().total_score = chain_score;
        vp.state().coverage_score = new_ratio;
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const Viewpoint& a, const Viewpoint& b) {
            return a.total_score() > b.total_score();
        }
    );

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
    if (!map_ || !coverage_tracker_) return false;

    if (vp.state().position.z() < 0.3f) return false;

    if (vp.is_in_collision(*map_, config_.collision.inflation_radius())) return false;

    if (coverage_tracker_->is_viewpoint_visited(vp)) return false;

    return true;
}

void ViewpointGenerator::score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const VoxelKeySet& target_cluster) {
    const auto& vp_config = config_.viewpoint;

    vp.compute_coverage_score(already_covered);
    float coverage_score = vp.coverage_score();

    size_t cluster_surfels_visible = 0;
    for (const auto& key : target_cluster) {
        if (vp.visible_voxels().count(key) > 0) {
            cluster_surfels_visible++;
        }
    }

    float cluster_score = 0.0f;
    if (!target_cluster.empty()) {
        cluster_score = static_cast<float>(cluster_surfels_visible) / target_cluster.size();
    }

    Eigen::Vector3f centroid = compute_cluster_centroid(target_cluster);
    float dist = (vp.position() - centroid).norm();
    
    float dist_score = 0.0f;
    if (dist >= vp_config.min_view_distance && vp_config.max_view_distance) {
        float dist_from_optimal = std::abs(dist - vp_config.optimal_view_distance);
        dist_score = 1.0f - (dist_from_optimal) / vp_config.optimal_view_distance;
        dist_score = std::max(0.0f, dist_score);
    }

    vp.state().total_score = vp_config.new_coverage_weight * coverage_score + 
                             vp_config.frontier_priority_weight * cluster_score + 
                             vp_config.distance_weight * dist_score;
}

float ViewpointGenerator::compute_yaw_to_target(const Eigen::Vector3f& from_pos, const Eigen::Vector3f& target_pos) const {
    Eigen::Vector3f direction = target_pos - from_pos;
    return std::atan2(direction.y(), direction.x());
}

float ViewpointGenerator::angular_distance(float yaw1, float yaw2) const {
    float diff = std::abs(yaw1 - yaw2);
    if (diff > M_PI) {
        diff = 2.0f * M_PI - diff;
    }

    return diff;
}

} // namespace

