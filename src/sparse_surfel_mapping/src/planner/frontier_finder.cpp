#include "sparse_surfel_mapping/planner/frontier_finder.hpp"

namespace sparse_surfel_map {

FrontierFinder::FrontierFinder() : map_(nullptr) {}
FrontierFinder::FrontierFinder(const SurfelMap* map) : map_(map) {}

std::vector<FrontierSurfel> FrontierFinder::find_frontiers_from_coverage(const VoxelKeySet& cumulative_coverage, const Eigen::Vector3f& expansion_center, float max_expansion_radius) const {
    
    std::vector<FrontierSurfel> frontiers;

    if (!map_ || cumulative_coverage.empty()) return frontiers;

    const float max_radius_sq = max_expansion_radius * max_expansion_radius;
    VoxelKeySet checked; // tracked checked neighbors

    // might be possible to extract ROI here with radius from expansion center to avoid iterating over all covered voxels/surfels

    // Iterate over all covered voxels 
    for (const auto& covered_key : cumulative_coverage) {
        for (const auto& nb_key : get_neighbors_6(covered_key)) {
            if (checked.count(nb_key) > 0) continue;
            checked.insert(nb_key);

            if (!has_surface(nb_key)) continue; // skip if nb has no surfel
            if (cumulative_coverage.count(nb_key) > 0) continue; // skip if nb already covered

            // This neighbor is a frontier (uncovered surface)
            auto voxel_opt = map_->get_voxel(nb_key);
            if (!voxel_opt || !voxel_opt->get().has_valid_surfel()) continue; // in free or without valid surfel
            const Surfel& surfel = voxel_opt->get().surfel();
            Eigen::Vector3f position = surfel.mean();

            // Check distance from expansion center
            float dist_sq = (position - expansion_center).squaredNorm();
            if (dist_sq > max_radius_sq) continue;

            // Count the number of uncovered surface neighbors the frontier has
            size_t uncovered_neighbors = count_uncovered_surface_neighbors(nb_key, cumulative_coverage);

            FrontierSurfel fs;
            fs.key = nb_key;
            fs.position = position;
            fs.normal = surfel.normal();
            fs.uncovered_neighbor_count = uncovered_neighbors;
            fs.is_covered = false;
            fs.distance_to_expansion = std::sqrt(dist_sq);
            fs.frontier_score = static_cast<float>(uncovered_neighbors) + 1.0f;
            
            frontiers.push_back(fs);
        }
    }

    // sort by distance from expansion center (increasing)
    std::sort(frontiers.begin(), frontiers.end(), 
        [](const FrontierSurfel& a, const FrontierSurfel&b) {
            return a.distance_to_expansion < b.distance_to_expansion;
        }
    );

    return frontiers;
}

std::vector<FrontierCluster> FrontierFinder::cluster_frontiers(const std::vector<FrontierSurfel>& frontiers, float cluster_radius, size_t min_cluster_size) const {
    
    std::vector<FrontierCluster> clusters;
    if (frontiers.empty()) return clusters;

    std::vector<bool> assigned(frontiers.size(), false);

    // greedy clustering: Start from closest frontiers (already sorted by distance to expansion center)
    for (size_t i = 0; i < frontiers.size(); ++i) {
        if (assigned[i]) continue;

        FrontierCluster cluster;
        cluster.surfels.push_back(frontiers[i]);
        assigned[i] = true;
        
        // Add nearby to cluster
        for (size_t j = i + 1; j < frontiers.size(); ++j) {
            if (assigned[j]) continue;

            bool close_enough = false;
            for (const auto& cluster_surfel : cluster.surfels) {
                if ((frontiers[j].position - cluster_surfel.position).norm() < cluster_radius) {
                    close_enough = true;
                    break;
                }
            }

            if (close_enough) {
                cluster.surfels.push_back(frontiers[j]);
                assigned[j] = true;
            }
        }

        // Keep reasonably cluster size
        if (cluster.surfels.size() >= min_cluster_size) {
            cluster.compute_centroid();
            clusters.push_back(std::move(cluster));
        }
    
    }

    // sort cluster by priority (accumulation of frontier score)
    std::sort(clusters.begin(), clusters.end(),
        [](const FrontierCluster& a, const FrontierCluster& b) {
            return a.total_priority > b.total_priority;
        });

    return clusters;
}

void FrontierFinder::compute_cluster_view_suggestion(FrontierCluster& cluster, float optimal_distance) const {
    if (cluster.surfels.empty()) return;

    cluster.compute_centroid();

    // Direction along the normal to find where camera will be placed
    Eigen::Vector3f view_direction = cluster.mean_normal;

    view_direction.z() *= 0.3f; // reduce vertical component
    if (view_direction.norm() < 0.1f) {
        view_direction = Eigen::Vector3f(1.0f, 0.0f, 0.0f); // arbitrary view direction if normal is mostly vertical
    }
    view_direction.normalize();

    cluster.suggested_view_position = cluster.centroid + view_direction * optimal_distance;
    Eigen::Vector3f look_dir = cluster.centroid - cluster.suggested_view_position; // view_position -> centroid
    cluster.suggested_yaw = std::atan2(look_dir.y(), look_dir.x());
}

std::vector<VoxelKey> FrontierFinder::get_neighbors_6(const VoxelKey& key) const {
    return {
        {key.x + 1, key.y, key.z},
        {key.x - 1, key.y, key.z},
        {key.x, key.y + 1, key.z},
        {key.x, key.y - 1, key.z},
        {key.x, key.y, key.z + 1},
        {key.x, key.y, key.z - 1}
    };
}

std::vector<VoxelKey> FrontierFinder::get_neighbors_26(const VoxelKey& key) const {
    std::vector<VoxelKey> neighbors;
    neighbors.reserve(26);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue; // skip center
                neighbors.push_back({key.x + dx, key.y + dy, key.z + dz});
            }
        }
    }

    return neighbors;
}

size_t FrontierFinder::count_uncovered_surface_neighbors(const VoxelKey& key, const VoxelKeySet& covered) const {
    size_t count = 0;
    for (const auto& nb : get_neighbors_6(key)) {
        if (has_surface(nb) && covered.count(nb) == 0) {
            count++;
        }
    }
    return count;
}

bool FrontierFinder::has_surface(const VoxelKey& key) const {
    if (!map_) return false;
    auto voxel_opt = map_->get_voxel(key);
    return voxel_opt && voxel_opt->get().has_valid_surfel();
}

Eigen::Vector3f FrontierFinder::key_to_position(const VoxelKey& key) const {
    if (!map_) return Eigen::Vector3f::Zero();

    const float voxel_size = map_->voxel_size();
    return Eigen::Vector3f(
        (key.x + 0.5f) * voxel_size,
        (key.y + 0.5f) * voxel_size,
        (key.z + 0.5f) * voxel_size
    );
}

} // namespace