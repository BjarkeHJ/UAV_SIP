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

std::vector<Viewpoint> ViewpointGenerator::generate_from_seed(const Eigen::Vector3f& seed_position, float seed_yaw) {

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<Viewpoint> result;
    if (!map_) return  result;

    last_frontiers_found_ = 0;
    last_clusters_formed_ = 0;
    last_candidates_generated_ = 0;
    last_candidates_in_collision_ = 0;

    const auto& vp_config = config_.viewpoint;

    // Get covered voxels from coverage tracker
    VoxelKeySet already_covered;
    if (coverage_tracker_) {
        already_covered = coverage_tracker_->observed_voxels();
    }

    // Create seed viewpoint from position and yaw
    Viewpoint seed_vp(seed_position, seed_yaw, config_.camera);
    seed_vp.set_id(generate_id());
    seed_vp.compute_visibility(*map_, false);
    seed_vp.compute_coverage_score(already_covered, vp_config);

    if (seed_vp.num_visible() > 0 && seed_vp.coverage_score() >= vp_config.min_overlap_ratio) {
        seed_vp.set_status(ViewpointStatus::PLANNED);
        result.push_back(seed_vp);
    }

    // Current coverage: already covered + seed's visibility (unique)
    VoxelKeySet current_coverage = already_covered;
    for (const auto& key : seed_vp.visible_voxels()) {
        current_coverage.insert(key);
    }

    // Run coverage region growing iterations
    Eigen::Vector3f search_center = seed_position;
    for (size_t step = 0; step < vp_config.growth_steps; ++step) {
        if (config_.debug_output) {
            std::cout << "[ViewpointGenerator] Growth Step " << (step + 1) << ", current coverage: " << current_coverage.size() << std::endl;
        }

        // grow from current coverage
        std::vector<Viewpoint> step_viewpoints = grow_step(current_coverage, search_center, already_covered);
        if (step_viewpoints.empty()) {
            if (config_.debug_output) {
                std::cout << "[ViewpointGenerator] No more expansions at step " << (step + 1) << std::endl;
            }
            break;
        }

        // add viewpoints and update coverage
        for (auto& vp : step_viewpoints) {
            vp.set_status(ViewpointStatus::PLANNED);
            result.push_back(std::move(vp));

            for (const auto& key : result.back().visible_voxels()) {
                current_coverage.insert(key);
            }

            search_center = result.back().position();

            if (result.size() >= vp_config.max_total_viewpoints) break;
        }

        if (result.size() >= vp_config.max_total_viewpoints) break;
    }

    if (vp_config.enable_structural_analysis && result.size() < vp_config.max_total_viewpoints) {
        auto features = analyze_structure(current_coverage);

        if (!features.empty()) {
            auto feature_viewpoints = generate_for_features(features, current_coverage);

            for (auto& vp : feature_viewpoints) {
                if (result.size() >= vp_config.max_total_viewpoints) {
                    break;
                }
                vp.set_status(ViewpointStatus::PLANNED);
                result.push_back(std::move(vp));
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    last_generation_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (config_.debug_output) {
        std::cout << "[ViewpointGenerator] Generated " << result.size() << " viewpoints in " << last_generation_time_ms_ << " ms" << std::endl;
    }

    return result;
}

std::vector<Viewpoint> ViewpointGenerator::generate_from_viewpoint(const Viewpoint& from_viewpoint) {
    std::vector<Viewpoint> result;
    if (!map_) return result;

    const auto& vp_config = config_.viewpoint;

    VoxelKeySet already_covered;
    if (coverage_tracker_) {
        already_covered = coverage_tracker_->observed_voxels();
    }

    VoxelKeySet current_coverage = already_covered;
    for (const auto& key : from_viewpoint.visible_voxels()) {
        current_coverage.insert(key);
    }

    Eigen::Vector3f search_center = from_viewpoint.position();
    for (size_t step = 0; step < vp_config.growth_steps; ++step) {
        std::vector<Viewpoint> step_viewpoints = grow_step(current_coverage, search_center, already_covered);

        if (step_viewpoints.empty()) break;

        for (auto& vp : step_viewpoints) {
            vp.set_status(ViewpointStatus::PLANNED);
            result.push_back(std::move(vp));

            for (const auto& key : result.back().visible_voxels()) {
                current_coverage.insert(key);
            }

            search_center = result.back().position();

            if (result.size() >= vp_config.max_total_viewpoints) break;
        }

        if (result.size() >= vp_config.max_total_viewpoints) break;
    }

    return result;
}

std::vector<Viewpoint> ViewpointGenerator::grow_step(const VoxelKeySet& current_coverage, const Eigen::Vector3f& search_center, const VoxelKeySet& already_covered) {
    std::vector<Viewpoint> result;
    const auto& vp_config = config_.viewpoint;
    float search_radius = vp_config.max_view_distance * 2.0f;
    // find frontier surfel in current view
    std::vector<FrontierSurfel> frontiers = frontier_finder_.find_frontier(current_coverage, search_center, search_radius);
    last_frontiers_found_ = frontiers.size();
    if (frontiers.empty()) return result;

    // cluster frontiers
    std::vector<FrontierCluster> clusters = frontier_finder_.cluster_frontiers(frontiers, vp_config.frontier_cluster_radius, vp_config.min_cluster_size);
    last_clusters_formed_ = clusters.size();
    if (clusters.empty()) return result;

    // generate viewpoint candidates from clusters
    std::vector<Viewpoint> candidates;
    for (auto& cluster : clusters) {
        frontier_finder_.compute_cluster_view_suggestion(cluster, vp_config.optimal_view_distance, vp_config.min_view_distance);
        Viewpoint vp = generate_viewpoint_for_cluster(cluster, already_covered);

        if (vp.num_visible() > 0) {
            candidates.push_back(std::move(vp));
            last_candidates_generated_++;
        }
    }

    if (candidates.empty()) return result;

    result = select_best_viewpoints(candidates, vp_config.max_viewpoints_per_steps, current_coverage);

    return result;
}

Viewpoint ViewpointGenerator::generate_viewpoint_for_cluster(const FrontierCluster& cluster, const VoxelKeySet& already_covered) {
    const auto& vp_config = config_.viewpoint;
    Viewpoint vp(cluster.suggested_view_position, cluster.suggested_yaw, config_.camera);
    vp.set_id(generate_id());

    // try suggested position
    if (is_position_valid(cluster.suggested_view_position)) {
        vp.compute_visibility(*map_, false);
        if (vp.num_visible() > 0) {
            score_viewpoint(vp, already_covered, cluster);
            return vp; // accepted
        }
    }
    else {
        last_candidates_in_collision_++;
    }

    // try different distances along view direction - Should probably expand out aswell
    for (float dist = vp_config.optimal_view_distance; dist >= vp_config.min_view_distance; dist -= 0.5f) {
        Eigen::Vector3f view_dir = -cluster.mean_normal;
        view_dir.z() *= 0.3f;
        view_dir.normalize();

        Eigen::Vector3f test_pos = cluster.centroid + view_dir * dist;
        float test_yaw = compute_yaw_to_target(test_pos, cluster.centroid);

        if (!is_position_valid(test_pos)) {
            last_candidates_in_collision_++;
            continue;
        }

        std::vector<Eigen::Vector3f> offsets = {
            Eigen::Vector3f::Zero(),
            Eigen::Vector3f(0.5f, 0, 0),
            Eigen::Vector3f(-0.5f, 0, 0),
            Eigen::Vector3f(0, 0.5f, 0),
            Eigen::Vector3f(0, -0.5f, 0),
            Eigen::Vector3f(0, 0, 0.5f)
        };

        for (const auto& offset : offsets) {
            Eigen::Vector3f pos = test_pos + offset;
            if (!is_position_valid(pos)) continue;

            float yaw = compute_yaw_to_target(pos, cluster.centroid);

            Viewpoint test_vp(pos, yaw, config_.camera);
            test_vp.set_id(generate_id());
            test_vp.compute_visibility(*map_, false);

            if (test_vp.num_visible() > vp.num_visible()) {
                vp = std::move(test_vp);
            }
        }

        if (vp.num_visible() > 0) {
            break;
        }
    }

    if (vp.num_visible() > 0) {
        score_viewpoint(vp, already_covered, cluster);
    }

    return vp;
}

bool ViewpointGenerator::find_valid_view_position(const FrontierCluster& cluster, Eigen::Vector3f& out_position, float& out_yaw) {
    const auto& vp_config = config_.viewpoint;

    Eigen::Vector3f view_dir = -cluster.mean_normal;
    view_dir.z() *= 0.3f;
    view_dir.normalize();

    for (float dist = vp_config.optimal_view_distance; dist >= vp_config.min_view_distance; dist -= 0.5f) {
        Eigen::Vector3f pos = cluster.centroid + view_dir * dist;
        if (is_position_valid(pos)) {
            out_position = pos;
            out_yaw = compute_yaw_to_target(pos, cluster.centroid);
            return true;
        }
    }

    return false;
}

bool ViewpointGenerator::is_position_valid(const Eigen::Vector3f& position) const {
    const auto& vp_config = config_.viewpoint;

    if (position.z() < 0.5f) return false;

    // collision check
    if (collision_checker_ && collision_checker_->is_sphere_in_collision(position)) return false;

    if (collision_checker_) {
        float dist = collision_checker_->distance_to_nearest_obstacle(position, vp_config.min_view_distance);
        if (dist < vp_config.min_view_distance * 0.5f) return false;
    }

    if (coverage_tracker_) {
        Viewpoint temp_vp(position, 0.0f, config_.camera);
        if (coverage_tracker_->is_viewpoint_visited(temp_vp)) return false;
    }

    return true;
}

void ViewpointGenerator::score_viewpoint(Viewpoint& vp, const VoxelKeySet& already_covered, const FrontierCluster& target_cluster) {
    const auto& vp_config = config_.viewpoint;

    // Coverage score: How much new surface is covered
    vp.compute_coverage_score(already_covered, vp_config);
    float coverage_score = vp.coverage_score();

    // Frontier score: How well does this cover the target cluster
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

    // distance score
    float dist = (vp.position() - target_cluster.centroid).norm();
    float dist_score = 1.0f - std::min(dist / vp_config.max_view_distance, 1.0f);
    
    // Combined score using configured weights
    vp.state().total_score = vp_config.new_coverage_weight * coverage_score +
                             vp_config.frontier_priority_weight * frontier_score +
                             vp_config.distance_weight * dist_score;
                             
    // Punish score if too little or too much overlap in visibility
    float overlap = vp.overlap_score();
    if (overlap < vp_config.min_overlap_ratio) {
        vp.state().total_score *= 0.5f;
    }
    else if (overlap > vp_config.max_overlap_ratio) {
        vp.state().total_score *= 0.7f;
    }
}

std::vector<Viewpoint> ViewpointGenerator::select_best_viewpoints(std::vector<Viewpoint>& candidates, size_t max_count, const VoxelKeySet& already_covered) {
    std::vector<Viewpoint> selected;

    if (candidates.empty()) return selected;

    const auto& vp_config = config_.viewpoint;

    std::sort(candidates.begin(), candidates.end(),
        [](const Viewpoint& a, const Viewpoint& b) {
            return a.total_score() > b.total_score();
        });

    VoxelKeySet will_be_covered = already_covered;
    for (auto& vp : candidates) {
        if (selected.size() >= max_count) break;

        size_t new_coverage = 0;
        for (const auto& key : vp.visible_voxels()) {
            if (will_be_covered.count(key) == 0) {
                new_coverage++;
            }
        }

        float adjusted_coverage = (vp.num_visible() > 0) ? static_cast<float>(new_coverage) / vp.num_visible() : 0.0f;

        float overlap = 1.0f - adjusted_coverage;

        // For first viewpoint be more loose - for subsequent be stricter and enforce overlap range
        bool acceptable = (adjusted_coverage > 0.05f); 
        if (selected.empty()) {
            acceptable = acceptable && (adjusted_coverage >= vp_config.min_overlap_ratio * 0.5f);
        }
        else {
            acceptable = acceptable && (overlap >= vp_config.min_overlap_ratio) && (overlap <= vp_config.max_overlap_ratio);
        }

        if (acceptable) {
            for (const auto& key : vp.visible_voxels()) {
                will_be_covered.insert(key);
            }

            selected.push_back(std::move(vp));
        }
    }

    return selected;
}

std::vector<StructuralFeature> ViewpointGenerator::analyze_structure(const VoxelKeySet& visible_voxels) {

    std::vector<StructuralFeature> features;

    if (!map_ || visible_voxels.empty()) return features;

    const auto& vp_config = config_.viewpoint;
    const float corner_th = std::cos(vp_config.corner_angle_th_deg * M_PI / 180.0f);

    // Analyze visible voxels for structural features
    for (const auto& key : visible_voxels) {
        auto voxel_opt = map_->get_voxel(key);
        if (!voxel_opt || !voxel_opt->get().has_valid_surfel()) continue;

        const Surfel& surfel = voxel_opt->get().surfel();
        Eigen::Vector3f normal = surfel.normal();
        Eigen::Vector3f position = surfel.mean();

        float max_angle_diff = 0.0f;
        Eigen::Vector3f most_different_normal = normal;

        // check nbs for normal discontinuities
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    VoxelKey nb{key.x + dx, key.y + dy, key.z + dz};
                    auto nb_opt = map_->get_voxel(nb);
                    if (!nb_opt || !nb_opt->get().has_valid_surfel()) continue;

                    Eigen::Vector3f nb_normal = nb_opt->get().surfel().normal();
                    float cos_angle = normal.dot(nb_normal);
                    float angle_diff = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));
                    if (angle_diff > max_angle_diff) {
                        max_angle_diff = angle_diff;
                        most_different_normal = nb_normal;
                    }
                }
            }
        }

        // Large normal discontinuity > 120deg
        if (max_angle_diff > (M_PI - vp_config.corner_angle_th_deg * M_PI / 180.0f)) {
            StructuralFeature feature;
            feature.type = StructuralFeatureType::CORNER;
            feature.position = position;

            feature.direction = (normal + most_different_normal).normalized(); // corner bisector direction

            feature.view_direction = feature.direction;
            feature.importance = max_angle_diff / M_PI;
            feature.associated_surfels.push_back(key);

            features.push_back(feature);
        }
        // Moderate normal discontinuities (Edge) 30-120 deg
        else if (max_angle_diff > vp_config.edge_curvature_th * M_PI) {
            StructuralFeature feature;
            feature.type = StructuralFeatureType::EDGE;
            feature.position = position;

            // edge direction is perpendicular to both normals 
            feature.direction = normal.cross(most_different_normal).normalized();

            feature.view_direction = (normal + most_different_normal).normalized();
            feature.importance = max_angle_diff / M_PI;
            feature.associated_surfels.push_back(key);

            features.push_back(feature);
        }
    }

    std::sort(features.begin(), features.end(), 
        [](const StructuralFeature& a, const StructuralFeature& b) {
            return a.importance > b.importance;
        });

    const size_t max_features = 5;
    if (features.size() > max_features) {
        features.resize(max_features); // truncate
    }

    return features;
}

std::vector<Viewpoint> ViewpointGenerator::generate_for_features(const std::vector<StructuralFeature>& features, const VoxelKeySet& current_coverage) {

    std::vector<Viewpoint> result;
    if (!map_) return result;

    const auto& vp_config = config_.viewpoint;

    for (const auto& feature : features) {
        Eigen::Vector3f view_pos = feature.position + feature.view_direction * vp_config.optimal_view_distance;

        view_pos.z() = std::max(view_pos.z(), feature.position.z());

        if (!is_position_valid(view_pos)) {
            bool found = false;
            for (float angle = 0; angle < 2.0f * M_PI; angle += M_PI / 4.0f) {
                Eigen::Vector3f offset(std::cos(angle)*0.5, std::sin(angle)*0.5f, 0.0f);
                Eigen::Vector3f alt_pos = view_pos + offset;

                if (is_position_valid(alt_pos)) {
                    view_pos = alt_pos;
                    found = true;
                    break;
                }
            }

            if (!found) continue;
        }

        float yaw = compute_yaw_to_target(view_pos, feature.position);

        Viewpoint vp(view_pos, yaw, config_.camera);
        vp.set_id(generate_id());
        vp.compute_visibility(*map_, false);
        vp.compute_coverage_score(current_coverage, vp_config);

        if (vp.num_visible() > 0 && vp.coverage_score() >= vp_config.min_overlap_ratio * 0.5f) {
            vp.state().total_score = feature.importance * 0.5f + vp.coverage_score() * 0.5f;
            result.push_back(std::move(vp));
        }
    }

    return result;
}

float ViewpointGenerator::compute_yaw_to_target(const Eigen::Vector3f& from_pos, const Eigen::Vector3f& target_pos) const {
    Eigen::Vector3f direction = target_pos - from_pos;
    return std::atan2(direction.y(), direction.x());
}


} // namespace