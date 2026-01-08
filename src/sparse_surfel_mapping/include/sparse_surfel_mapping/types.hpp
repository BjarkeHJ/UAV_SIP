#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <functional>
#include <limits>

namespace sparse_surfel_map {

/* VoxelKey and VoxelKeyHash */
struct VoxelKey {
    int32_t x,y,z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const VoxelKey& other) const {
        return !(*this == other);
    }

    bool operator<(const VoxelKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z; 
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        // large primes for better distribution
        constexpr std::size_t p1 = 73856093;
        constexpr std::size_t p2 = 19349663;
        constexpr std::size_t p3 = 83492791;
        return (static_cast<std::size_t>(k.x) * p1) ^
               (static_cast<std::size_t>(k.y) * p2) ^
               (static_cast<std::size_t>(k.z) * p3);
    }
};

/* Pending Update Type */
struct PendingPointUpdate {
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    float weight;
    Eigen::Vector3f view_direction;
};

/* Configurations */
struct SurfelConfig {
    size_t min_points_for_validity{10};
    float planarity_threshold{0.01f};
    float scale_threshold{0.01f};
    float degeneracy_threshold{0.1f};
    float max_eigenvalue_ratio{100.0f}; 
};

struct SurfelMapConfig {

};

/* Statistics (Verbose) */
struct MapStatistics {
    size_t num_voxels{0};
    size_t num_valid_surfels{0};
    size_t num_invalid_surfels{0};
    size_t total_points_integrated{0};

    double last_update_time_ms{0.0};
    double association_time_ms{0.0};
    double update_time_ms{0.0};

    Eigen::Vector3f min_bound{Eigen::Vector3f::Constant(std::numeric_limits<float>::max())};
    Eigen::Vector3f max_bound{Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest())};
};


/* Constants */
namespace constants {
    constexpr float EPSILON = 1e-8f;


};


} // namsepace sparse_surfel_map

#endif