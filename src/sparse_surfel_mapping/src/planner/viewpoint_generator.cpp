#include "sparse_surfel_mapping/planner/viewpoint_generator.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <numeric>

namespace sparse_surfel_map {

ViewpointGenerator::ViewpointGenerator()
    : config_()
    , sphere_radius_(config_.camera.max_range * 1.0f)
{}

ViewpointGenerator::ViewpointGenerator(const InspectionPlannerConfig& config)
    : config_(config)
    , sphere_radius_(config_.camera.max_range * 1.0f)
{}

bool ViewpointGenerator::update_exploration_goal(const Eigen::Vector3f& current_position, const Eigen::Vector3f& current_direction) {
    if (!map_ || !coverage_tracker_) return false;

    // Get map frontiers
    VoxelKeySet map_frontiers = coverage_tracker_->map_frontiers();
    if (map_frontiers.empty()) {
        current_goal_.is_valid = false;
        return false;
    }

    // Cluster frontiers
    auto clusters = cluster_frontiers(map_frontiers, current_position);
    if (clusters.empty()) {
        current_goal_.is_valid = false;
        return false;
    }

    std::cout << "Number of Frontier Clusters: " << clusters.size() << std::endl;

    // Select best cluster
    auto best_cluster = select_best_cluster(clusters, current_direction);

    // Update goal
    current_goal_.position = best_cluster.centroid + config_.viewpoint.optimal_view_distance * best_cluster.avg_normal;
    current_goal_.direction = (best_cluster.centroid - current_position).normalized();
    current_goal_.yaw = std::atan2(-best_cluster.avg_normal.y(), -best_cluster.avg_normal.x());
    current_goal_.last_centroid = best_cluster.centroid;
    current_goal_.distance_to_drone = (best_cluster.centroid - current_position).norm();
    current_goal_.is_valid = true;

    return true;
}

std::vector<FrontierCluster> ViewpointGenerator::cluster_frontiers(const VoxelKeySet& frontiers, const Eigen::Vector3f& drone_pos) {
    if (frontiers.empty()) return {};

    const float voxel_size = map_->voxel_size();
    const float cluster_distance_threshold = 1.0f; // 1 meter clustering threshold
    const size_t min_cluster_size = 3;
    const size_t max_cluster_size = 10;

    // Convert VoxelKeys to positions
    std::vector<std::pair<VoxelKey, Eigen::Vector3f>> frontier_positions;
    frontier_positions.reserve(frontiers.size());

    for (const auto& key : frontiers) {
        auto voxel_opt = map_->get_voxel(key);
        if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
            Eigen::Vector3f pos = voxel_opt->get().surfel().mean();
            frontier_positions.push_back({key, pos});
        }
    }

    // Connectivity-based clustering (BFS)
    std::vector<bool> visited(frontier_positions.size(), false);
    std::vector<FrontierCluster> clusters;

    for (size_t i = 0; i < frontier_positions.size(); ++i) {
        if (visited[i]) continue;

        // Start new cluster
        FrontierCluster cluster;
        std::queue<size_t> to_visit;
        to_visit.push(i);
        visited[i] = true;

        while (!to_visit.empty()) {
            size_t current_idx = to_visit.front();
            to_visit.pop();

            const auto& [key, pos] = frontier_positions[current_idx];
            cluster.members.push_back(key);
            cluster.centroid += pos;
            if (cluster.members.size() > max_cluster_size) break; // limit cluster size

            // Find neighbors within threshold
            for (size_t j = 0; j < frontier_positions.size(); ++j) {
                if (visited[j]) continue;

                const auto& [other_key, other_pos] = frontier_positions[j];
                float dist = (pos - other_pos).norm();

                if (dist < cluster_distance_threshold) {
                    visited[j] = true;
                    to_visit.push(j);
                }
            }
        }

        // Finalize cluster
        if (cluster.size() >= min_cluster_size) {
            cluster.centroid /= static_cast<float>(cluster.size());

            // Compute average normal
            Eigen::Vector3f avg_normal = Eigen::Vector3f::Zero();
            for (const auto& key : cluster.members) {
                auto voxel_opt = map_->get_voxel(key);
                if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
                    avg_normal += voxel_opt->get().surfel().normal();
                }
            }
            cluster.avg_normal = avg_normal.normalized();

            // Direction from drone
            cluster.direction_from_drone = (cluster.centroid - drone_pos).normalized();

            clusters.push_back(cluster);
        }
    }

    return clusters;
}

FrontierCluster ViewpointGenerator::select_best_cluster(const std::vector<FrontierCluster>& clusters, const Eigen::Vector3f& current_direction) {
    if (clusters.empty()) return FrontierCluster();

    float best_score = -1.0f;
    FrontierCluster best_cluster;

    for (const auto& cluster : clusters) {
        // Score based on: cluster size, distance (prefer closer), alignment with current direction
        float size_score = std::min(static_cast<float>(cluster.size()) / 3.0f, 1.0f);

        float distance = cluster.direction_from_drone.norm();
        float distance_score = std::exp(-distance / 5.0f); // prefer within 5m

        float alignment = cluster.direction_from_drone.dot(current_direction);
        float alignment_score = std::max(0.0f, alignment); // [0, 1]

        // Combined score (heavily favor alignment to stick to direction)
        float score = size_score * 0.2f + distance_score * 0.3f + alignment_score * 0.5f;

        if (score > best_score) {
            best_score = score;
            best_cluster = cluster;
            best_cluster.score = score;
        }
    }

    return best_cluster;
}

bool ViewpointGenerator::is_goal_stable(float distance_threshold) const {
    if (!current_goal_.is_valid) return false;

    // Goal is stable if drone is close to it (distance-based stability)
    return current_goal_.distance_to_drone < distance_threshold;
}

std::deque<Viewpoint> ViewpointGenerator::generate_exploration_viewpoints(const Eigen::Vector3f& current_position, size_t max_viewpoints) {
    std::deque<Viewpoint> viewpoints;

    if (!current_goal_.is_valid || !map_ || !coverage_tracker_) {
        return viewpoints;
    }

    // TODO: Generate RRT* path from current_position to current_goal_.position
    // For now, use straight line as placeholder

    path_to_goal_.clear();
    path_to_goal_.push_back(current_position);
    path_to_goal_.push_back(current_goal_.position);

    // Sample spheres along path
    const float overlap_ratio = 0.3f; // 30% overlap

    const auto ts = std::chrono::high_resolution_clock::now();
    auto sphere_centers = sample_spheres_along_path(path_to_goal_, sphere_radius_, overlap_ratio);
    const auto te = std::chrono::high_resolution_clock::now();
    std::cout << "Sphere Centers Size: " << sphere_centers.size() << std::endl;
    std::cout << "Sphere center time: " << std::chrono::duration<double,std::milli>(te-ts).count() << " ms" << std::endl;

    // Generate viewpoints in each sphere
    int s = 0;
    for (const auto& sphere_center : sphere_centers) {
        if (viewpoints.size() >= max_viewpoints) break;

        const auto t1 = std::chrono::high_resolution_clock::now();
        auto sphere_viewpoints = generate_viewpoints_in_sphere(sphere_center, std::min(3ul, max_viewpoints - viewpoints.size()));
        const auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Sphere " << s << " gen-time: " << std::chrono::duration<double,std::milli>(t2-t1).count() << " ms" << std::endl;

        for (auto& vp : sphere_viewpoints) {
            viewpoints.push_back(vp);
        }

        s++;
    }

    return viewpoints;
}

std::vector<Eigen::Vector3f> ViewpointGenerator::sample_spheres_along_path(const std::vector<Eigen::Vector3f>& path, float sphere_radius, float overlap_ratio) {
    std::vector<Eigen::Vector3f> sphere_centers;
    if (path.size() < 2) return sphere_centers;

    const float spacing = sphere_radius * (1.0f - overlap_ratio); // 30% overlap -> 70% spacing

    float distance_traveled = 0.0f;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        const Eigen::Vector3f& start = path[i];
        const Eigen::Vector3f& end = path[i + 1];
        const Eigen::Vector3f segment = end - start;
        const float segment_length = segment.norm();

        float local_dist = 0.0f;
        while (distance_traveled + local_dist < distance_traveled + segment_length) {
            float t = local_dist / segment_length;
            Eigen::Vector3f sphere_center = start + t * segment;
            sphere_centers.push_back(sphere_center);

            local_dist += spacing;
            if (local_dist >= segment_length) break;
        }

        distance_traveled += segment_length;
    }

    // Add final sphere at goal
    if (!path.empty()) {
        sphere_centers.push_back(path.back());
    }

    return sphere_centers;
}

std::vector<Viewpoint> ViewpointGenerator::generate_viewpoints_in_sphere(const Eigen::Vector3f& sphere_center, size_t max_viewpoints) {
    if (!map_ || max_viewpoints == 0) return {};

    const auto t1 = std::chrono::high_resolution_clock::now();

    // Query surfels in sphere using spatial hash
    const float radius = sphere_radius_;
    const float opt_view_dist = config_.viewpoint.optimal_view_distance;

    // Get surfels in sphere
    const auto& surfels_in_sphere = map_->query_surfels_in_radius(sphere_center, radius);
    if (surfels_in_sphere.empty()) return {};

    // Score surfels by observability (quick filter)
    std::vector<std::pair<VoxelKey, float>> scored_surfels;
    scored_surfels.reserve(surfels_in_sphere.size());

    for (const auto& s_ref : surfels_in_sphere) {
        const Surfel& s = s_ref.get();
        float obs_score = s.observability(sphere_center, opt_view_dist);
        if (obs_score < 0.15f) continue;  // Higher threshold for pre-filtering
        scored_surfels.push_back({s.key(), obs_score});
    }

    if (scored_surfels.empty()) return {};

    const auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Scored surfels: " << scored_surfels.size() << std::endl;

    // Cluster surfels in sphere
    auto clusters = cluster_surfels_in_sphere(scored_surfels, std::min(5ul, max_viewpoints));

    const auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Clusters: " << clusters.size() << " | Cluster time: "
              << std::chrono::duration<double,std::milli>(t3-t2).count() << " ms" << std::endl;

    // Generate viewpoints from clusters adaptively
    return generate_viewpoints_from_clusters(clusters, sphere_center, scored_surfels.size());
}

std::vector<SurfelCluster> ViewpointGenerator::cluster_surfels_in_sphere(const std::vector<std::pair<VoxelKey, float>>& scored_surfels, size_t target_clusters) {
    std::vector<SurfelCluster> clusters;
    if (scored_surfels.empty()) return clusters;

    // Simple spatial + normal clustering
    const float spatial_threshold = 0.5f;  // 0.5m
    const float normal_angle_threshold = 0.7f;  // cos(45°) ≈ 0.7

    std::vector<bool> assigned(scored_surfels.size(), false);

    for (size_t i = 0; i < scored_surfels.size() && clusters.size() < target_clusters; ++i) {
        if (assigned[i]) continue;

        // Start new cluster
        SurfelCluster cluster;
        const auto& [seed_key, seed_score] = scored_surfels[i];

        auto seed_voxel = map_->get_voxel(seed_key);
        if (!seed_voxel) continue;

        const Surfel& seed_surfel = seed_voxel->get().surfel();
        const Eigen::Vector3f seed_pos = seed_surfel.mean();
        const Eigen::Vector3f seed_normal = seed_surfel.normal();

        cluster.members.push_back(seed_key);
        cluster.centroid = seed_pos;
        cluster.avg_normal = seed_normal;
        cluster.total_score = seed_score;
        assigned[i] = true;

        // Add nearby surfels with similar normals
        for (size_t j = i + 1; j < scored_surfels.size(); ++j) {
            if (assigned[j]) continue;

            const auto& [key, score] = scored_surfels[j];
            auto voxel = map_->get_voxel(key);
            if (!voxel) continue;

            const Surfel& surfel = voxel->get().surfel();
            const Eigen::Vector3f pos = surfel.mean();
            const Eigen::Vector3f normal = surfel.normal();

            // Check spatial proximity
            float spatial_dist = (pos - seed_pos).norm();
            if (spatial_dist > spatial_threshold) continue;

            // Check normal similarity
            float normal_similarity = seed_normal.dot(normal);
            if (normal_similarity < normal_angle_threshold) continue;

            // Add to cluster
            cluster.members.push_back(key);
            cluster.centroid += pos;
            cluster.avg_normal += normal;
            cluster.total_score += score;
            assigned[j] = true;

            if (cluster.members.size() >= 50) break;  // Cap cluster size
        }

        // Finalize cluster
        if (cluster.members.size() > 0) {
            cluster.centroid /= static_cast<float>(cluster.members.size());
            cluster.avg_normal.normalize();
            clusters.push_back(cluster);
        }
    }

    return clusters;
}

std::vector<Viewpoint> ViewpointGenerator::generate_viewpoints_from_clusters(const std::vector<SurfelCluster>& clusters, const Eigen::Vector3f& sphere_center, size_t total_surfels_in_sphere) {
    std::vector<Viewpoint> viewpoints;
    if (clusters.empty()) return viewpoints;

    const auto t1 = std::chrono::high_resolution_clock::now();

    VoxelKeySet covered_surfels;
    const size_t target_coverage = static_cast<size_t>(0.9f * total_surfels_in_sphere);  // 75% threshold

    // Sort clusters by score descending
    std::vector<size_t> cluster_indices(clusters.size());
    std::iota(cluster_indices.begin(), cluster_indices.end(), 0);
    std::sort(cluster_indices.begin(), cluster_indices.end(),
              [&](size_t a, size_t b) { return clusters[a].total_score > clusters[b].total_score; });

    for (size_t idx : cluster_indices) {
        const auto& cluster = clusters[idx];

        // Generate viewpoint from cluster
        Eigen::Vector3f target_pos = cluster.centroid;
        Eigen::Vector3f target_normal = cluster.avg_normal;

        // Viewpoint position: standoff from cluster centroid
        Eigen::Vector3f vp_pos = target_pos + config_.viewpoint.optimal_view_distance * target_normal;

        // Yaw: point toward cluster
        float vp_yaw = std::atan2(-target_normal.y(), -target_normal.x());

        Viewpoint vp(vp_pos, vp_yaw, config_.camera);

        // Surface-repulsive adjustment if obstructed
        if (!adjust_viewpoint_if_obstructed(vp, target_pos, target_normal)) {
            continue;
        }

        // Check path adherence
        if (!is_near_path(vp.position(), path_to_goal_, 0.5f)) {
            continue;
        }

        // Compute visibility (no occlusion for speed)
        vp.compute_visibility(*map_, true);
        if (vp.num_visible() == 0) continue;

        // Track coverage
        for (const auto& vis_key : vp.visible_voxels()) {
            covered_surfels.insert(vis_key);
        }

        viewpoints.push_back(vp);

        // Adaptive stopping: if we've covered enough, stop generating
        if (covered_surfels.size() >= target_coverage) {
            std::cout << "Early stop: covered " << covered_surfels.size() << "/" << total_surfels_in_sphere
                      << " with " << viewpoints.size() << " viewpoints" << std::endl;
            break;
        }

        if (viewpoints.size() >= 5) break;  // Hard cap
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Generated " << viewpoints.size() << " viewpoints from " << clusters.size()
              << " clusters | Time: " << std::chrono::duration<double,std::milli>(t2-t1).count() << " ms" << std::endl;

    return viewpoints;
}

bool ViewpointGenerator::adjust_viewpoint_if_obstructed(Viewpoint& vp, const Eigen::Vector3f& target_surfel_pos, const Eigen::Vector3f& target_surfel_normal) {
    const float radius = config_.collision.inflation_radius();

    // Check initial collision
    if (!vp.is_in_collision(*map_, radius)) {
        return true; // already good
    }

    // Surface-repulsive adjustment
    const int max_attempts = 8;
    const float angle_step = M_PI / 4.0f; // 45° increments

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        // Move tangentially around the target surfel
        float angle = attempt * angle_step;

        // Tangent vectors (perpendicular to normal)
        Eigen::Vector3f tangent1 = target_surfel_normal.cross(Eigen::Vector3f::UnitZ()).normalized();
        if (tangent1.norm() < 0.1f) { // normal is vertical
            tangent1 = target_surfel_normal.cross(Eigen::Vector3f::UnitX()).normalized();
        }

        Eigen::Vector3f tangent2 = target_surfel_normal.cross(tangent1).normalized();

        // Circular sampling in tangent plane
        Eigen::Vector3f tangent_offset =
            std::cos(angle) * tangent1 + std::sin(angle) * tangent2;

        // Try varying distances (grazing angles)
        for (float dist = 0.8f; dist <= 1.2f; dist += 0.2f) {
            Eigen::Vector3f new_pos = target_surfel_pos +
                dist * (target_surfel_normal + 0.3f * tangent_offset).normalized();

            // Update viewpoint
            vp.set_position(new_pos);

            // Adjust yaw to keep looking at target
            Eigen::Vector3f to_target = target_surfel_pos - new_pos;
            float new_yaw = std::atan2(to_target.y(), to_target.x());
            vp.set_yaw(new_yaw);

            // Check collision
            if (!vp.is_in_collision(*map_, radius)) {
                return true; // found collision-free position
            }
        }
    }

    return false; // couldn't find collision-free position
}

bool ViewpointGenerator::is_near_path(const Eigen::Vector3f& point, const std::vector<Eigen::Vector3f>& path, float tolerance) const {
    if (path.empty()) return true;
    if (path.size() == 1) return (point - path[0]).norm() < tolerance;

    // Check distance to path segments
    for (size_t i = 0; i < path.size() - 1; ++i) {
        const Eigen::Vector3f& start = path[i];
        const Eigen::Vector3f& end = path[i + 1];

        Eigen::Vector3f segment = end - start;
        float segment_length = segment.norm();
        if (segment_length < 1e-6f) continue;

        Eigen::Vector3f segment_dir = segment / segment_length;
        Eigen::Vector3f to_point = point - start;

        float projection = to_point.dot(segment_dir);
        projection = std::clamp(projection, 0.0f, segment_length);

        Eigen::Vector3f closest_point = start + projection * segment_dir;
        float distance = (point - closest_point).norm();

        if (distance < tolerance) return true;
    }

    return false;
}

bool ViewpointGenerator::is_near_frontier(const VoxelKey& key) const {
    if (!coverage_tracker_) return false;

    const auto& map_frontiers = coverage_tracker_->map_frontiers();

    // Check if key is in frontier set
    if (map_frontiers.count(key) > 0) return true;

    // Check 26-connectivity neighbors
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                VoxelKey nb{key.x + dx, key.y + dy, key.z + dz};
                if (map_frontiers.count(nb) > 0) return true;
            }
        }
    }

    return false;
}


} // namespace
