#ifndef SURFEL_MAP_HPP_
#define SURFEL_MAP_HPP_

#include <iostream>
#include <chrono>
#include <map>
#include <set>
#include <shared_mutex>

#include "sparse_surfel_mapping/mapper/spatial_hash.hpp"

namespace sparse_surfel_map {

class SurfelMap {
public:
    SurfelMap();
    explicit SurfelMap(const SurfelMapConfig& config);
    SurfelMap(const SurfelMap&) = delete; // do not allow copy constructor
    SurfelMap& operator=(const SurfelMap&) = delete; // no copy operator
    SurfelMap(SurfelMap&&) noexcept = default;
    SurfelMap& operator=(SurfelMap&&) noexcept = default;
    ~SurfelMap() = default;

    void begin_update();
    void associate_points(const std::vector<PointWithNormal>& pns, const Eigen::Transform<float, 3, Eigen::Isometry>& transform);
    size_t commit_update();
    // eq to begin_update() + associate_points() + commit_update
    size_t integrate_points(const std::vector<PointWithNormal>& pns, const Eigen::Transform<float, 3, Eigen::Isometry>& transform); 
    
    // Pending updates
    bool has_pending_update() const { return !pending_updates_.empty(); }
    size_t pending_update_count() const;
    void discard_pending_update();
    
    // Map queries
    std::optional<std::reference_wrapper<const Voxel>> get_voxel_at(const Eigen::Vector3f& point) const;
    std::optional<std::reference_wrapper<const Voxel>> get_voxel(const VoxelKey& key) const;
    std::optional<std::reference_wrapper<const Surfel>> get_surfel_at(const Eigen::Vector3f& point) const;
    std::vector<std::reference_wrapper<const Surfel>> get_valid_surfels() const;

    const SpatialHash& voxels() const { return spatial_hash_; }
    
    std::vector<std::reference_wrapper<const Surfel>> query_surfels_in_box(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound) const;
    std::vector<std::reference_wrapper<const Surfel>> query_surfels_in_radius(const Eigen::Vector3f center, float radius) const;

    // Map management
    void clear();
    void reset(const SurfelMapConfig& config);
    size_t prune_invalid_surfels();
    size_t prune_outside_box(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound);
    size_t prune_shadow_surfels(float normal_th = 0.85f); // 0.85 ~32 degree

    // Map Statistics
    MapStatistics get_statistics() const;
    float voxel_size() const { return config_.voxel_size; }
    size_t num_voxels() const { return spatial_hash_.size(); }
    size_t num_valid_surfels() const;
    size_t total_point_count() const { return total_points_integrated_; }
    const std::string& map_frame() const { return config_.map_frame; }
    const SurfelMapConfig config() const { return config_; }

    // Timing info
    double last_association_time_ms() const { return last_association_time_ms_; }
    double last_commit_time_ms() const { return last_commit_time_ms_; }
    double last_update_time_ms() const { return last_association_time_ms_ + last_commit_time_ms_; }

    mutable std::shared_mutex mutex_; // mutex for map usage across multiple nodes (mapper + planner)

private:
    using Clock = std::chrono::high_resolution_clock;
    using PendingUpdateMap = std::map<VoxelKey, std::vector<PointWithNormal>>; // queued updates (phase 1)

    bool is_point_in_range(const Eigen::Vector3f& point) const;
    VoxelKey compute_voxel_key(const Eigen::Vector3f& point) const;

    SurfelMapConfig config_;
    SpatialHash spatial_hash_;
    PendingUpdateMap pending_updates_;

    size_t total_points_integrated_{0};
    double last_association_time_ms_{0.0};
    double last_commit_time_ms_{0.0};
    double last_update_time_ms_{0.0};

    bool update_in_progress_{false};

};

} // namespace


#endif