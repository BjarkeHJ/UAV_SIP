#include "sparse_surfel_mapping/planner/collision_checker.hpp"
#include <cmath>

namespace sparse_surfel_map {

CollisionChecker::CollisionChecker() : config_(), map_(nullptr) {}
CollisionChecker::CollisionChecker(const CollisionConfig& config, const SurfelMap* map) : config_(config), map_(map) {}

bool CollisionChecker::is_in_collision(const Eigen::Vector3f& point) const {
    if (!map_) return false;

    const VoxelKey key = point_to_key(point);
    return is_voxel_occupied(key);
}

bool CollisionChecker::is_sphere_in_collision(const Eigen::Vector3f& point) const {
    return is_sphere_in_collision(point, config_.inflation_radius());
}

bool CollisionChecker::is_sphere_in_collision(const Eigen::Vector3f& point, float radius) const {
    if (!map_) return false;

    const std::vector<VoxelKey> voxels = get_voxels_in_sphere(point, radius);

    const float voxel_size = map_->voxel_size();

    for (const auto& key : voxels) {
        if (!is_voxel_occupied(key)) continue;

        const Eigen::Vector3f voxel_center(
            (key.x + 0.5f) * voxel_size,
            (key.y + 0.5f) * voxel_size,
            (key.z + 0.5f) * voxel_size
        );

        const float dist_sq = (voxel_center - point).squaredNorm();
        
        const float voxel_radius = voxel_size * 0.866f; // sqrt(3) / 2 -> half voxel diagonal 
        const float th_sq = (radius + voxel_radius) * (radius + voxel_radius);

        if (dist_sq < th_sq) return true; // collision!
    }

    return false;
}

bool CollisionChecker::is_path_collision_free(const Eigen::Vector3f& start, const Eigen::Vector3f& end) const {
    if (!map_) return false;
    const Eigen::Vector3f diff = end - start;
    const float path_len = diff.norm();

    if (path_len < 1e-6f) {
        return !is_sphere_in_collision(start);
    }

    const Eigen::Vector3f direction = diff / path_len;
    const float step = config_.path_resolution;
    const float radius = config_.inflation_radius();

    if (is_sphere_in_collision(start, radius)) return false;

    // Walk along path
    for (float t = step; t < path_len; t += step) {
        const Eigen::Vector3f point = start + direction * t;
        if (is_sphere_in_collision(point, radius)) return false;
    }

    if (is_sphere_in_collision(end, radius)) return false;

    return true;
}

bool CollisionChecker::is_path_collision_free(const std::vector<Eigen::Vector3f>& path) const {
    if (path.size() < 2) {
        return path.empty() || !is_sphere_in_collision(path[0]);
    }

    for (size_t i = 0; i < path.size() - 1; ++i) {
        if (!is_path_collision_free(path[i], path[i + 1])) return false;
    }

    return true;
}

float CollisionChecker::find_first_collision(const Eigen::Vector3f& point, const Eigen::Vector3f& direction, float max_dist) const {
    if (!map_) return max_dist;
    const float step = config_.path_resolution;
    const float radius = config_.inflation_radius();

    for (float t = 0.0f; t < max_dist; t += step) {
        const Eigen::Vector3f sample_point = point + direction * t;
        if (is_sphere_in_collision(sample_point, radius)) return t;
    }
    
    return max_dist;
}

float CollisionChecker::distance_to_nearest_obstacle(const Eigen::Vector3f& point, float max_dist) const {
    if (!map_) return false;

    float min_dist = max_dist;
    const float voxel_size = map_->voxel_size();

    const int max_voxel_radius = static_cast<int>(std::ceil(max_dist / voxel_size));
    const VoxelKey center_key = point_to_key(point);

    for (int r = 0; r <= max_voxel_radius; ++r) {
        bool found_in_shell = false;

        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dz = -r; dz <= r; ++dz) {
                    if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) continue; // only check shell surface

                    VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};

                    if (is_voxel_occupied(key)) {
                        const Eigen::Vector3f voxel_center(
                            (key.x + 0.5f) * voxel_size,
                            (key.y + 0.5f) * voxel_size,
                            (key.z + 0.5f) * voxel_size
                        );

                        const float dist = (voxel_center - point).norm() - voxel_size * 0.5f;
                        min_dist = std::min(min_dist, std::max(0.0f, dist));
                        found_in_shell = true;
                    }
                } 
            }
        }

        // return early if obstacle found
        if (found_in_shell && min_dist < (r + 1) * voxel_size) return min_dist;
    }

    return min_dist;
}

bool CollisionChecker::is_voxel_occupied(const VoxelKey& key) const {
    if (!map_) return false;

    auto voxel_opt = map_->get_voxel(key);
    if (!voxel_opt) return false;

    return voxel_opt->get().has_valid_surfel();
}

VoxelKey CollisionChecker::point_to_key(const Eigen::Vector3f& point) const {
    if (!map_) return {0, 0, 0};

    const float voxel_size = map_->voxel_size();
    return VoxelKey{
        static_cast<int32_t>(std::floor(point.x() / voxel_size)),
        static_cast<int32_t>(std::floor(point.y() / voxel_size)),
        static_cast<int32_t>(std::floor(point.z() / voxel_size))
    };
}

std::vector<VoxelKey> CollisionChecker::get_voxels_in_sphere(const Eigen::Vector3f& center, float radius) const {
    std::vector<VoxelKey> result;

    if (!map_) return result;

    const float voxel_size = map_->voxel_size();
    const int voxel_radius = static_cast<int>(std::ceil(radius / voxel_size)) + 1;
    const VoxelKey center_key = point_to_key(center);

    result.reserve(static_cast<size_t>((2 * voxel_size + 1) * (2 * voxel_size + 1) * (2 * voxel_size + 1)));

    for (int dx = -voxel_radius; dx <= voxel_radius; ++dx) {
        for (int dy = -voxel_radius; dy <= voxel_radius; ++dy) {
            for (int dz = -voxel_radius; dz <= voxel_radius; ++dz) {
                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};

                const Eigen::Vector3f voxel_center(
                    (key.x + 0.5) * voxel_size,
                    (key.y + 0.5) * voxel_size,
                    (key.z + 0.5) * voxel_size
                );

                const float voxel_radius_f = voxel_size * 0.866f;
                const float combined_radius = radius + voxel_radius_f;

                if ((voxel_center - center).squaredNorm() <= combined_radius * combined_radius) {
                    result.push_back(key);
                }
            }
        }
    }

    return result;
}


} // namespace