#include "sparse_surfel_mapping/spatial_hash.hpp"

namespace sparse_surfel_map {

SpatialHash::SpatialHash()
    : voxels_()
    , voxel_size_(0.1f)
    , surfel_config_()
{}

SpatialHash::SpatialHash(float voxel_size, const SurfelConfig& surfel_config, size_t initial_bucket_count, float max_load_factor)
    : voxels_()
    , voxel_size_(voxel_size)
    , surfel_config_(surfel_config)
{
    voxels_.max_load_factor(max_load_factor); // ratio between size / bucket_count. (size is #elements, bc)
    voxels_.reserve(initial_bucket_count); // reserve buckets
}

Voxel& SpatialHash::get_or_create(const VoxelKey& key) {
    auto it = voxels_.find(key);
    if (it != voxels_.end()) {
        return it->second;
    }

    // create new
    auto result = voxels_.emplace(key, Voxel(key, surfel_config_));
    return result.first->second;
}

std::optional<std::reference_wrapper<const Voxel>> SpatialHash::get(const VoxelKey& key) const {
    auto it = voxels_.find(key);
    if (it != voxels_.end()) {
        return std::cref(it->second);
    }
    return std::nullopt;
}

std::optional<std::reference_wrapper<Voxel>> SpatialHash::get(const VoxelKey& key) {
    auto it = voxels_.find(key);
    if (it != voxels_.end()) {
        return std::ref(it->second);
    }
    return std::nullopt;
}

bool SpatialHash::contains(const VoxelKey& key) const {
    return voxels_.find(key) != voxels_.end();
}

bool SpatialHash::remove(const VoxelKey& key) {
    return voxels_.erase(key) > 0;
}

// Coordinate convert
VoxelKey SpatialHash::point_to_key(const Eigen::Vector3f& point) const {
    const float inv_size = 1.0f / voxel_size_;
    return VoxelKey{
        static_cast<int32_t>(std::floor(point.x() * inv_size)),
        static_cast<int32_t>(std::floor(point.y() * inv_size)),  
        static_cast<int32_t>(std::floor(point.z() * inv_size)) 
    };
}

Eigen::Vector3f SpatialHash::key_to_point(const VoxelKey& key) const {
    // return center of voxel
    return Eigen::Vector3f(
        (static_cast<float>(key.x) + 0.5f) * voxel_size_,
        (static_cast<float>(key.y) + 0.5f) * voxel_size_,
        (static_cast<float>(key.z) + 0.5f) * voxel_size_
    );
}

std::vector<VoxelKey> SpatialHash::get_sorted_keys() const {
    std::vector<VoxelKey> keys;
    keys.reserve(voxels_.size());
    
    for (const auto& [key, voxel] : voxels_) {
        keys.push_back(key);
    }

    std::sort(keys.begin(), keys.end());
    return keys;
}

std::vector<VoxelKey> SpatialHash::get_valid_surfel_keys() const {
    std::vector<VoxelKey> keys;
    keys.reserve(voxels_.size());

    for (const auto& [key, voxel] : voxels_) {
        if (voxel.has_valid_surfel()) {
            keys.push_back(key);
        }
    }

    std::sort(keys.begin(), keys.end());
    return keys;
}

// bulk ops
void SpatialHash::clear() {
    voxels_.clear();
}

void SpatialHash::reserve(size_t count) {
    voxels_.reserve(count);
}

size_t SpatialHash::prune_invalid() {
    size_t removed = 0;

    auto it = voxels_.begin();
    while (it != voxels_.end()) {
        if (!it->second.has_valid_surfel()) {
            it = voxels_.erase(it);
            removed++;
        }
        else {
            ++it;
        }
    }

    return removed;
}

size_t SpatialHash::prune_outside_bounds(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound) {
    size_t removed = 0;

    auto it = voxels_.begin();
    while (it != voxels_.end()) {
        const Eigen::Vector3f center = key_to_point(it->first);
        bool inside = center.x() >= min_bound.x() && center.x() >= max_bound.x() &&
                      center.y() >= min_bound.y() && center.y() >= max_bound.y() &&
                      center.z() >= min_bound.z() && center.z() >= max_bound.z();
        if (!inside) {
            it = voxels_.erase(it);
            removed++;
        }
        else {
            ++it;
        }
    }

    return removed;
}

MapStatistics SpatialHash::compute_statistics() const {
    MapStatistics stats;
    stats.num_voxels = voxels_.size();

    for (const auto& [key, voxel] : voxels_) {
        if (voxel.has_valid_surfel()) {
            stats.num_valid_surfels++;
        }
        else {
            stats.num_invalid_surfels++;
        }

        stats.total_points_integrated += voxel.point_count();

        // update map bounds
        const Eigen::Vector3f center = key_to_point(key);
        stats.min_bound = stats.min_bound.cwiseMin(center); // min coeffs between current min_bound and center
        stats.max_bound = stats.max_bound.cwiseMax(center); // max
    }
}

// Neighbor queries

std::vector<std::reference_wrapper<const Voxel>> SpatialHash::get_neighbors_6(const VoxelKey& key) const {
    std::vector<std::reference_wrapper<const Voxel>> nbs;
    nbs.reserve(6);

    // 6-connected nbs (faces)
    static const std::array<std::array<int32_t, 3>, 6> offsets = {{
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    }};

    for (const auto& offset : offsets) {
        VoxelKey nb_key{key.x + offset[0], key.y + offset[1], key.z + offset[2]};
        auto voxel = get(nb_key); // std::optional 
        if (voxel) {
            // if voxel has points/surfel
            nbs.push_back(voxel.value());
        }
    }
    return nbs;
}

std::vector<std::reference_wrapper<const Voxel>> SpatialHash::get_neighbors_26(const VoxelKey& key) const {
    std::vector<std::reference_wrapper<const Voxel>> nbs;
    nbs.reserve(26);

    for (int32_t dx = -1; dx <= 1; ++dx) {
        for (int32_t dy = -1; dy <= 1; ++dy) {
            for (int32_t dz = -1; dz <= 1; ++dz) {
                if (dz == 0 && dy == 0 && dz == 0) continue; // skip center (self)

                VoxelKey nb_key{key.x + dx, key.y + dy, key.z + dz};
                auto voxel = get(nb_key); // std::optional
                if (voxel) {
                    // if voxel occupied
                    nbs.push_back(voxel.value());
                }
            }
        }
    }
    return nbs;
}

std::vector<std::reference_wrapper<const Voxel>> SpatialHash::get_neighbors_in_radius(const Eigen::Vector3f& center, float radius) const {
    std::vector<std::reference_wrapper<const Voxel>> result;

    // bounding box in voxel coordinates
    const VoxelKey center_key = point_to_key(center);
    const int32_t voxel_radius = static_cast<int32_t>(std::ceil(radius / voxel_size_));
    const float radius_sq = radius * radius;

    for (int32_t dx = -voxel_radius; dx <= voxel_radius; ++dx) {
        for (int32_t dy = -voxel_radius; dy <= voxel_radius; ++dy) {
            for (int32_t dz = -voxel_radius; dz <= voxel_radius; ++dz) {

                VoxelKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
                auto voxel = get(key); // std::optional
                if (voxel) {
                    const Eigen::Vector3f voxel_center = key_to_point(key);
                    if ((voxel_center - center).squaredNorm() <= radius_sq) {
                        result.push_back(voxel.value());
                    }
                }
            }
        }
    }
    return result;
}

} // namespace