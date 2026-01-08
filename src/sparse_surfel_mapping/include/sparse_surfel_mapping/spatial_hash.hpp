#ifndef SPATIAL_HASH_HPP_
#define SPATIAL_HASH_HPP_

#include "sparse_surfel_mapping/voxel.hpp"
#include <unordered_map>
#include <vector>
#include <optional>
#include <mutex>

namespace sparse_surfel_map {

class SpatialHash {
public:
    using VoxelMap = std::unordered_map<VoxelKey, Voxel, VoxelKeyHash>;
    using iterator = VoxelMap::iterator;
    using const_iterator = VoxelMap::const_iterator;

    SpatialHash();
    SpatialHash(float voxel_size, const SurfelConfig& surfel_config, size_t initial_bucket_count = 10000, float max_load_factor = 0.75f);
    ~SpatialHash() = default;
    SpatialHash(const SpatialHash&) = delete; // no copy constructor
    SpatialHash& operator=(const SpatialHash&) = delete; // no copy operator
    SpatialHash(SpatialHash&&) noexcept = default;
    SpatialHash& operator=(SpatialHash&&) noexcept = default;

    Voxel& get_or_create(const VoxelKey& key); // get if exist - create if not
    std::optional<std::reference_wrapper<const Voxel>> get(const VoxelKey& key) const; // get const
    std::optional<std::reference_wrapper<Voxel>> get(const VoxelKey& key); // get mutable

    bool contains(const VoxelKey& key) const; // voxel exists at key?
    bool remove(const VoxelKey& key); // remove voxel at key

    VoxelKey point_to_key(const Eigen::Vector3f& point) const;
    Eigen::Vector3f key_to_point(const VoxelKey& key) const;
    float voxel_size() const { return voxel_size_; }
    
    iterator begin() { return voxels_.begin(); } // iterator begin 
    const_iterator begin() const { return voxels_.begin(); }
    iterator end() { return voxels_.end(); } // iterator end
    const_iterator end() const { return voxels_.end(); }

    std::vector<VoxelKey> get_sorted_keys() const;
    std::vector<VoxelKey> get_valid_surfel_keys() const;

    // Bulk ops
    void clear();
    void reserve(size_t count);
    size_t prune_invalid();
    size_t prune_outside_bounds(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound);

    // Stats
    size_t size() const { return voxels_.size(); } // number of voxels
    bool empty() const { return voxels_.empty(); }
    size_t bucket_count() const { return voxels_.bucket_count(); } // number of buckets in hash table
    float load_factor() const { return voxels_.load_factor(); } // current load factor

    MapStatistics compute_statistics() const;

    // Neighbor queries
    std::vector<std::reference_wrapper<const Voxel>> get_neighbors_6(const VoxelKey& key) const;
    std::vector<std::reference_wrapper<const Voxel>> get_neighbors_26(const VoxelKey& key) const;
    std::vector<std::reference_wrapper<const Voxel>> get_neighbors_in_radius(const Eigen::Vector3f& center, float radius) const;

private:
    VoxelMap voxels_; // hash map storage
    size_t voxel_size_; // voxel size (map resolution)
    SurfelConfig surfel_config_; // surfel config
};


} // namespace

#endif