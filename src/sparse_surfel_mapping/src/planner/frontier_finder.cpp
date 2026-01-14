#include "sparse_surfel_mapping/planner/frontier_finder.hpp"

namespace sparse_surfel_map {

FrontierFinder::FrontierFinder() : map_(nullptr) {}
FrontierFinder::FrontierFinder(const SurfelMap* map) : map_(map) {}

std::vector<FrontierSurfel> FrontierFinder::find_frontier(const VoxelKeySet& covered_voxels, const Eigen::Vector3f& search_center, float search_radius) const {
    
    std::vector<FrontierSurfel> frontiers;

    if (!map_) return frontiers;

    const float voxel_size = map_->voxel_size();
    const int voxel_radius = static_cast<int>(std::ceil(search_radius / voxel_size));

    const VoxelKey center_key{
        static_cast<int32_t>(std::floor(search_center.x() / voxel_size)),
        static_cast<int32_t>(std::floor(search_center.y() / voxel_size)),
        static_cast<int32_t>(std::floor(search_center.z() / voxel_size))
    };

    const float radius_sq = search_radius * search_radius;

    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx) {
        for (int dy = -voxel_radius; dy <= voxel_radius; ++dy) {
            for (int dz = -voxel_radius; dz <= voxel_radius; ++dz) {
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};

                Eigen::Vector3f voxel_center(
                    (key.x + 0.5) * voxel_size,
                    (key.y + 0.5) * voxel_size,
                    (key.z + 0.5) * voxel_size
                );
                if ((voxel_center - search_center).squaredNorm() > radius_sq) continue;

                if (!has_surface(key)) continue;

                bool is_covered = covered_voxels.count(key) > 0;
                size_t uncovered_neighbors = count_uncovered_surface_neighbors(key, covered_voxels);

                bool is_frontier = false;

                if (is_covered && uncovered_neighbors > 0) {
                    is_frontier = true; // at boundary between covered and uncovered
                }
                else if (!is_covered) {
                    // check face-neighbors
                    for (const auto& neighbor : get_neighbors_6(key)) {
                        if (covered_voxels.count(neighbor) > 0 && has_surface(neighbor)) {
                            is_frontier = true;
                            break;
                        }
                    }
                }

                if (!is_frontier) continue;

                auto voxel_opt = map_->get_voxel(key);
                if (!voxel_opt) continue;

                const Surfel& surfel = voxel_opt->get().surfel();
                FrontierSurfel fs;
                fs.key = key;
                fs.position = surfel.mean();
                fs.normal = surfel.normal();
                fs.uncovered_neighbor_count = uncovered_neighbors;
                fs.is_covered = is_covered;

                fs.frontier_score = static_cast<float>(uncovered_neighbors);
                if (!is_covered) {
                    fs.frontier_score += 2.0f;
                }

                frontiers.push_back(fs);
            }
        }
    }

    // Sort by highest frontier score
    std::sort(frontiers.begin(), frontiers.end(),
        [](const FrontierSurfel& a, const FrontierSurfel& b) {
            return a.frontier_score > b.frontier_score;
        });

    return frontiers;
}

std::vector<FrontierSurfel> FrontierFinder::find_frontier_around_coverage(const VoxelKeySet& visible_from_viewpoint, const VoxelKeySet& already_covered) const {
    std::vector<FrontierSurfel> frontiers;

    if (!map_) return frontiers;

    // combine alread covered with what this viewpoint sees
    VoxelKeySet total_covered = already_covered;
    for (const auto& key : visible_from_viewpoint) {
        total_covered.insert(key);
    }

    VoxelKeySet checked;
    for (const auto& visible_key : visible_from_viewpoint) {
        for (const auto& nb : get_neighbors_26(visible_key)) {
            if (checked.count(nb) > 0) continue;
            checked.insert(nb);

            if (!has_surface(nb)) continue;

            bool is_covered = total_covered.count(nb) > 0;
            size_t uncovered_neighbors = count_uncovered_surface_neighbors(nb, total_covered);

            bool is_frontier = false;
            if (!is_covered) {
                is_frontier = true;
            }
            else if (uncovered_neighbors > 0) {
                is_frontier = true;
            }

            if (!is_frontier) continue;

            auto voxel_opt = map_->get_voxel(nb);
            if (!voxel_opt) continue;

            const Surfel& surfel = voxel_opt->get().surfel();

            FrontierSurfel fs;
            fs.key = nb;
            fs.position = surfel.mean();
            fs.normal = surfel.normal();
            fs.uncovered_neighbor_count = uncovered_neighbors;
            fs.is_covered = is_covered;

            fs.frontier_score = static_cast<float>(uncovered_neighbors);
            if (!is_covered) {
                fs.frontier_score = 3.0f;
            }
        }
    }

    std::sort(frontiers.begin(), frontiers.end(),
        [](const FrontierSurfel& a, const FrontierSurfel& b) {
            return a.frontier_score > b.frontier_score;
        });

    return frontiers;
}

std::vector<FrontierCluster> FrontierFinder::cluster_frontiers(const std::vector<FrontierSurfel>& frontiers, float cluster_radius, size_t min_cluster_size) const {
    std::vector<FrontierCluster> clusters;
    if (frontiers.empty()) return clusters;

    std::vector<bool> assigned(frontiers.size(), false);

    for (size_t i = 0; i < frontiers.size(); ++i) {
        if (assigned[i]) continue;

        FrontierCluster cluster;
        cluster.surfels.push_back(frontiers[i]);
        assigned[i] = true;
        
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

void FrontierFinder::compute_cluster_view_suggestion(FrontierCluster& cluster, float optimal_distance, float min_distance) const {
    if (cluster.surfels.empty()) return;

    cluster.compute_centroid();

    Eigen::Vector3f view_direction = cluster.mean_normal;

    view_direction.z() *= 0.3f; // reduce vertical component
    if (view_direction.norm() < 0.1f) {
        view_direction = Eigen::Vector3f(1.0f, 0.0f, 0.0f); // arbitrary view direction if normal is mostly vertical
    }
    view_direction.normalize();

    cluster.suggested_view_position = cluster.centroid + view_direction * optimal_distance;
    Eigen::Vector3f look_dir = cluster.centroid - cluster.suggested_view_position;
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

} // namespace