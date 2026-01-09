#ifndef VOXEL_HPP_
#define VOXEL_HPP_

#include "sparse_surfel_mapping/surfel.hpp"
#include <memory>

namespace sparse_surfel_map {

class Voxel {
public:
    Voxel();
    Voxel(const VoxelKey& key, const SurfelConfig& surfel_config);
    Voxel(const Voxel& other) = default;
    Voxel(Voxel&& other) noexcept = default;
    Voxel& operator=(const Voxel& other) = default;
    Voxel& operator=(Voxel&& other) noexcept = default;
    ~Voxel() = default;

    void integrate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, float weight /*Eigen::Vector3f view_direction*/);
    void integrate_points(const std::vector<PointWithNormal>& updates);
    void finalize_surfel();
    void reset();

    // Accessors
    const VoxelKey& key() const { return key_; }
    const Surfel& surfel() const { return surfel_; }
    Surfel& surfel() { return surfel_; } // mutable
    bool has_valid_surfel() const { return surfel_.is_valid(); }
    bool is_empty() const { return surfel_.point_count() == 0; }
    size_t point_count() const { return surfel_.point_count(); }

    // Voxel Geometries
    Eigen::Vector3f center(float voxel_size) const;
    Eigen::Vector3f min_corner(float voxel_size) const;
    Eigen::Vector3f max_corner(float voxel_size) const;
    bool contains_point(const Eigen::Vector3f& point, float voxel_size) const;

private:
    VoxelKey key_;
    Surfel surfel_;
};


} // namespace sparse_surfel_map

#endif