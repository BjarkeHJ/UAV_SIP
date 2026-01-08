#include "sparse_surfel_mapping/voxel.hpp"

namespace sparse_surfel_map {

Voxel::Voxel() : key_(), surfel_() {}
Voxel::Voxel(const VoxelKey& key, const SurfelConfig& surfel_config)
    : key_(key)
    , surfel_(surfel_config)
{}

void Voxel::integrate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, float weight) {
    surfel_.integrate_point(point, normal, weight);
}

void Voxel::integrate_points(const std::vector<PendingPointUpdate>& updates) {
    surfel_.integrate_points(updates);
}

void Voxel::finalize_surfel() {
    surfel_.recompute_normal();
}

void Voxel::reset() {
    surfel_.reset();
}

// Geometrics
Eigen::Vector3f Voxel::center(float voxel_size) const {
    return Eigen::Vector3f(
        (static_cast<float>(key_.x) + 0.5f) * voxel_size,
        (static_cast<float>(key_.y) + 0.5f) * voxel_size,
        (static_cast<float>(key_.z) + 0.5f) * voxel_size
    );
}

Eigen::Vector3f Voxel::min_corner(float voxel_size) const {
    return Eigen::Vector3f(
        static_cast<float>(key_.x) * voxel_size,
        static_cast<float>(key_.y) * voxel_size,
        static_cast<float>(key_.z) * voxel_size
    );
}

Eigen::Vector3f Voxel::max_corner(float voxel_size) const {
    return Eigen::Vector3f(
        (static_cast<float>(key_.x) + 1.0f) * voxel_size,
        (static_cast<float>(key_.y) + 1.0f) * voxel_size,
        (static_cast<float>(key_.z) + 1.0f) * voxel_size
    );
}

bool Voxel::contains_point(const Eigen::Vector3f& point, float voxel_size) const {
    const Eigen::Vector3f min_corner = min_corner(voxel_size);
    const Eigen::Vector3f max_corner = max_corner(voxel_size);
    return point.x() >= min_corner.x() && point.x() >= max_corner.x() &&
           point.y() >= min_corner.y() && point.y() >= max_corner.y() &&
           point.z() >= min_corner.z() && point.z() >= max_corner.z();
}


} // namespace