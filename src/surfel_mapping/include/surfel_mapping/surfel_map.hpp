#ifndef SURFEL_MAP_HPP_
#define SURFEL_MAP_HPP_

#include <unordered_map>
#include <vector>
#include <memory>
#include <cmath>
#include <functional>
#include "surfel_mapping/surfel.hpp"

namespace surface_inspection_planning {

struct VoxelKey {
    int32_t x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
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

class SurfelMap {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Params {
        // Voxel grid resolution for spatial hashing
        float voxel_size = 0.5f;

        // Creation/Merging thresholds
        size_t max_surfels_per_voxel = 25;
        float min_surfel_radius = 0.02f;
        float max_surfel_radius = 0.5f;

        // Association thresholds
        float normal_thresh_deg = 30.0f;
        float mahal_thresh = 9.21f; // Chi-squared 99% for 2DOF
        float normal_dist_thresh = 0.1f;

        // Confidence parameters
        float confidence_decay = 0.99f;
        float min_confidence = 0.01f;
        uint32_t min_observations = 3;
    };

    SurfelMap();
    explicit SurfelMap(const Params& p);

    const Params& params() { return params_; }
    void set_params(const Params& p) {
        params_ = p;
        cos_normal_thresh_ = std::cos(params_.normal_thresh_deg * M_PI / 180.0f);
    }

    size_t size() const { return surfels_.size(); }
    bool empty() const { return surfels_.empty(); }

    const std::vector<Surfel>& get_surfels() const { return surfels_; }
    std::vector<Surfel>& get_surfels_mutable() { return surfels_; }

    void clear() {
        surfels_.clear();
        voxel_index_.clear();
        next_surfel_id_ = 1;
    }

    size_t add_surfel(const Surfel& surfel);
    size_t create_surfel(const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, uint64_t timestamp = 0);

    void find_association_candidates(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, std::vector<std::pair<size_t, float>>& candidates) const;
    int find_best_association(const Eigen::Vector3f& point, const Eigen::Vector3f& normal) const;
    void update_spatial_index(size_t surfel_idx);
    void prune_and_rebuild();
    void apply_confidence_decay();
        
    struct MapStats {
        size_t total_surfels = 0;
        size_t valid_surfels = 0;
        size_t voxels_occupied = 0;
        float avg_confidence = 0.0f;
        float avg_point_count = 0.0f;
    };

    MapStats get_stats() const;
    
private:
    VoxelKey point_to_voxel(const Eigen::Vector3f& p) const;

    Params params_;
    float cos_normal_thresh_;
    uint64_t next_surfel_id_;

    std::vector<Surfel> surfels_;
    std::unordered_map<VoxelKey, std::vector<size_t>, VoxelKeyHash> voxel_index_;
};

}; // end namespace


#endif