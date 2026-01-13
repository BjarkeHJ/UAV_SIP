#ifndef COLLISION_CHECKER_HPP_
#define COLLISION_CHECKER_HPP_

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class CollisionChecker {
public:
    CollisionChecker();
    CollisionChecker(const CollisionConfig& config, const SurfelMap* map);

    void set_map(const SurfelMap* map) { map_ = map; }

    bool is_in_collision(const Eigen::Vector3f& point) const;
    bool is_sphere_in_collision(const Eigen::Vector3f& point) const; // from drone radius + safe margin
    bool is_sphere_in_collision(const Eigen::Vector3f& point, float radius) const;
    bool is_path_collision_free(const Eigen::Vector3f& start, const Eigen::Vector3f& end) const;
    bool is_path_collision_free(const std::vector<Eigen::Vector3f>& path) const;
    float find_first_collision(const Eigen::Vector3f& point, const Eigen::Vector3f& direction, float max_dist) const;
    float distance_to_nearest_obstacle(const Eigen::Vector3f& point, float max_dist = 5.0f) const;
    bool is_voxel_occupied(const VoxelKey& key) const;
    float inflation_radius() const { return config_.inflation_radius(); }

private:
    VoxelKey point_to_key(const Eigen::Vector3f& point) const;
    std::vector<VoxelKey> get_voxels_in_sphere(const Eigen::Vector3f& center, float radius) const;

    CollisionConfig config_;
    const SurfelMap* map_{nullptr};
};

struct CollisionCheckResult {
    bool collision_free{true};
    Eigen::Vector3f collision_point{Eigen::Vector3f::Zero()};
    float collision_distance{std::numeric_limits<float>::infinity()};
    VoxelKey collision_voxel;
};

} // namespace

#endif