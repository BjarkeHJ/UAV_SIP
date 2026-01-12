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

/* Measurement Type - In sensor frame*/
struct PointWithNormal {
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Vector3f normal{Eigen::Vector3f::Zero()};
    float weight{0.0f};
};

/* Default Configurations */
struct PreprocessConfig {
    bool enable_ground_filter{true};
    float ground_z_min{0.2f};

    size_t width{240};
    size_t height{180};
    float hfov_deg{106.0f};
    float vfov_deg{86.0f};
    float min_range{0.1f};
    float max_range{10.0f};

    size_t ds_factor{1};
    size_t normal_est_px_radius{3};
    bool orient_towards_sensor{true};

    size_t range_smooth_iters{3};
    float depth_sigma_m{0.05f};
    float spatial_sigma_px{1.0f};
    float max_depth_jump_m{0.10f};
};

struct SurfelConfig {
    size_t min_points_for_validity{25};
    float planarity_threshold{0.01f};
    float scale_threshold{0.01f};
    float degeneracy_threshold{0.1f};
    float max_eigenvalue_ratio{100.0f}; 
};

struct SurfelMapConfig {
    float voxel_size{0.3f};
    float max_range{10.0f};
    float min_range{0.1f};
    size_t initial_bucket_count{10000};
    float max_load_factor{0.75f};
    
    SurfelConfig surfel_config;
    std::string map_frame{"odom"};
    
    bool compute_eigenvalues{true};
    bool debug_output{true};

    PreprocessConfig preprocess_config;
};

/* Statistics (Verbose) */
struct MapStatistics {
    size_t num_voxels{0};
    size_t num_valid_surfels{0};
    size_t num_invalid_surfels{0};
    size_t total_points_integrated{0};

    double last_update_time_ms{0.0};
    double last_association_time_ms{0.0};
    double last_commit_time_ms{0.0};

    Eigen::Vector3f min_bound{Eigen::Vector3f::Constant(std::numeric_limits<float>::max())};
    Eigen::Vector3f max_bound{Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest())};
};


/* Constants */
namespace constants {
    constexpr float EPSILON = 1e-6f;
    constexpr float MAX_EVAL = 5.0f;
};


} // namsepace sparse_surfel_map

#endif