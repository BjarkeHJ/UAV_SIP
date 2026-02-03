#ifndef SPATIAL_HASH_HPP_
#define SPATIAL_HASH_HPP_

#include "sparse_surfel_mapping/mapper/voxel.hpp"
#include <iostream>
#include <unordered_map>
#include <unordered_set>
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

    // Coarse grid (for frontiers)
    static constexpr int COARSE_FACTOR = 10;
    enum class CoarseCellState : uint32_t { UNKNOWN = 0, FREE = 1, OCCUPIED = 2 }; // state of coarse cell 
    void on_surfel_added(const VoxelKey& key);
    void on_surfel_removed(const VoxelKey& key);
    VoxelKey fine_to_coarse(const VoxelKey& fine) const;
    CoarseCellState coarse_cell_state(const VoxelKey& key) const;

    void trace_ray(const Eigen::Vector3f& from, const Eigen::Vector3f& to); // tracing observation ray marking traversed cells FREE
    void observe_frustum(const Eigen::Vector3f& sensor_pos, float yaw, float hfov_deg, float vfov_deg, float max_range); // NOTE: SWITCH TO TAKE SENSOR CONFIG!

    bool is_frontier_coarse(const VoxelKey& coarse_key) const; // is coarse cell a frontier of current map
    bool is_reachable_unknown(const VoxelKey& unknown_key) const; // is unknown coarse cell reachable (connected via FREE) around corner vs behind surface
    std::vector<VoxelKey> get_frontier_cells() const; // get all frontier coarse cells 
    std::vector<VoxelKey> get_frontier_cells_in_radius(const Eigen::Vector3f& center, float radius) const; // ... in radius
    size_t get_coarse_cell_surfel_count(const VoxelKey& coarse_key) const;

    struct CoarseGridStats {
        size_t num_free{0};
        size_t num_occupied{0};
        size_t num_frontiers{0};
    };
    CoarseGridStats get_coarse_grid_stats() const;

private:
    void mark_coarse_free(const VoxelKey& coarse_key);
    void mark_coarse_occupied(const VoxelKey& coarse_key);

    VoxelMap voxels_; // hash map storage
    float voxel_size_; // voxel size (map resolution)
    SurfelConfig surfel_config_; // surfel config

    // coarse grid
    std::unordered_map<VoxelKey, CoarseCellState, VoxelKeyHash> coarse_state_; // cell key -> state 
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> coarse_surfel_counts_; // cell key -> number of valid fine-level surfels
};


} // namespace

#endif