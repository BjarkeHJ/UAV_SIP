#include "sparse_surfel_mapping/mapper/spatial_hash.hpp"

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
        bool inside = center.x() >= min_bound.x() && center.x() <= max_bound.x() &&
                      center.y() >= min_bound.y() && center.y() <= max_bound.y() &&
                      center.z() >= min_bound.z() && center.z() <= max_bound.z();
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

    return stats;
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

                if (dx == 0 && dy == 0 && dz == 0) continue; // skip center (self)

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
                if (voxel && voxel->get().is_occupied()) {
                    // Approximate by voxel center
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


// Coarse grid
VoxelKey SpatialHash::fine_to_coarse(const VoxelKey& fine) const {
    auto floor_div = [](int32_t a, int32_t b) -> int32_t {
        int32_t d = a / b;
        int32_t r = a % b;
        return (r != 0 && ((a ^ b) < 0)) ? d - 1 : d;
    };

    return VoxelKey{
        floor_div(fine.x, COARSE_FACTOR),
        floor_div(fine.y, COARSE_FACTOR),
        floor_div(fine.z, COARSE_FACTOR)
    };
}

SpatialHash::CoarseCellState SpatialHash::coarse_cell_state(const VoxelKey& coarse_key) const {
    auto it = coarse_state_.find(coarse_key);
    return (it != coarse_state_.end()) ? it->second : CoarseCellState::UNKNOWN;
}

void SpatialHash::mark_coarse_free(const VoxelKey& coarse_key) {
    auto it = coarse_state_.find(coarse_key);
    if (it == coarse_state_.end()) {
        // only mark as FREE if currently UNKNOWN (not in hash)
        coarse_state_[coarse_key] = CoarseCellState::FREE;
    }
    // dont downgrade occupied to free
}

void SpatialHash::mark_coarse_occupied(const VoxelKey& coarse_key) {
    coarse_state_[coarse_key] = CoarseCellState::OCCUPIED;
}

void SpatialHash::on_surfel_added(const VoxelKey& key) {
    VoxelKey coarse_key = fine_to_coarse(key);
    coarse_surfel_counts_[coarse_key]++; // increment number of surfels in coarse cell
    mark_coarse_occupied(coarse_key); // mark occupied (ensure)
}

void SpatialHash::on_surfel_removed(const VoxelKey& key) {
    VoxelKey coarse_key = fine_to_coarse(key);

    auto it = coarse_surfel_counts_.find(coarse_key);
    if (it == coarse_surfel_counts_.end()) return; // not existing

    it->second--; // decrement
    if (it->second == 0) {
        // last surfel removed -> erase and downgrade to free (still observed)
        coarse_surfel_counts_.erase(it);
        coarse_state_[coarse_key] = CoarseCellState::FREE;
    }
}

size_t SpatialHash::get_coarse_cell_surfel_count(const VoxelKey& coarse_key) const {
    auto it = coarse_surfel_counts_.find(coarse_key);
    return (it != coarse_surfel_counts_.end()) ? it->second : 0;
}

void SpatialHash::trace_ray(const Eigen::Vector3f& from, const Eigen::Vector3f& to) {
    const float coarse_size = voxel_size_ * COARSE_FACTOR;
    Eigen::Vector3f dir = to - from;
    const float l = dir.norm();
    if (l < 1e-6f) return;
    dir /= l;

    const float step = coarse_size * 0.5f;
    VoxelKey prev_coarse{INT32_MAX, INT32_MAX, INT32_MAX};
    VoxelKey end_coarse = fine_to_coarse(point_to_key(to));

    bool hit = false;

    for (float t = 0; t < l; t += step) {
        Eigen::Vector3f pos = from + dir * t;
        VoxelKey coarse = fine_to_coarse(point_to_key(pos));

        if (coarse != prev_coarse) {
            if (coarse_cell_state(coarse) == CoarseCellState::OCCUPIED) {
                mark_coarse_occupied(coarse);
                hit = true;
                continue;
            }

            if (coarse == end_coarse) break; // dont change the last 

            if (hit) {
                coarse_state_.erase(coarse); // mark UNKNOWN after occulision
                coarse_surfel_counts_.erase(coarse);
                hit = false;
            }
            else {
                mark_coarse_free(coarse);
            }

            prev_coarse = coarse;
        }
    }
}

void SpatialHash::observe_frustum(const Eigen::Vector3f& sensor_pos, float yaw, float hfov_deg, float vfov_deg, float max_range) {
    const float coarse_size = voxel_size_ * COARSE_FACTOR;

    const float cos_yaw = std::cos(yaw);
    const float sin_yaw = std::sin(yaw);
    const Eigen::Vector3f forward(cos_yaw, sin_yaw, 0.0f);
    const Eigen::Vector3f right(sin_yaw, -cos_yaw, 0.0f);
    const Eigen::Vector3f up = Eigen::Vector3f::UnitZ();

    const float tan_half_h = std::tan(hfov_deg * 0.5f * M_PI / 180.0f);
    const float tan_half_v = std::tan(vfov_deg * 0.5 * M_PI / 180.0f);

    const float far_half_width = max_range * tan_half_h;
    const float far_half_height = max_range * tan_half_v;

    const int h_samples = std::max(3, static_cast<int>(std::ceil(2.0f * far_half_width / coarse_size)) + 1);
    const int v_samples = std::max(3, static_cast<int>(std::ceil(2.0f * far_half_height / coarse_size)) + 1);

    for (int vi = 0; vi < v_samples; ++vi) {
        for (int hi = 0; hi < h_samples; ++hi) {
            const float u = 2.0f * hi / (h_samples - 1) - 1.0f;
            const float v = 2.0f * vi / (v_samples - 1) - 1.0f;

            Eigen::Vector3f ray_dir = forward + right * (u * tan_half_h) + up * (v * tan_half_v);
            ray_dir.normalize();

            Eigen::Vector3f far_point = sensor_pos + ray_dir * max_range;
            trace_ray(sensor_pos, far_point);
        }
    }
}

bool SpatialHash::is_frontier_coarse(const VoxelKey& coarse_key) const {
    if (coarse_cell_state(coarse_key) != CoarseCellState::OCCUPIED) {
        return false; // frontier must contain map
    }

    // Check nb-6 connectivity for unknown
    static const std::array<std::array<int32_t, 3>, 6> offsets = {{
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    }};

    for (const auto& offset : offsets) {
        VoxelKey nb{coarse_key.x + offset[0], coarse_key.y + offset[1], coarse_key.z + offset[2]};
        if (coarse_cell_state(nb) == CoarseCellState::UNKNOWN && is_reachable_unknown(nb)) return true;
    }

    return false;
}

bool SpatialHash::is_reachable_unknown(const VoxelKey& unknown_key) const {
    // UNKNOWN cell is reachable if it is adjacent to FREE cell 
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz ==0) continue;

                VoxelKey nb{unknown_key.x + dx, unknown_key.y + dy, unknown_key.z + dz};
                if (coarse_cell_state(nb) == CoarseCellState::FREE) return true; // Adjacent to observed free space = could be observed
            }
        }
    }

    return false; // isolated in occupied space
}

std::vector<VoxelKey> SpatialHash::get_frontier_cells() const {
    std::vector<VoxelKey> result;
    for (const auto& [coarse_key, state] : coarse_state_) {
        if (state == CoarseCellState::OCCUPIED && is_frontier_coarse(coarse_key)) {
            result.push_back(coarse_key);
        }
    }

    return result;
}

SpatialHash::CoarseGridStats SpatialHash::get_coarse_grid_stats() const {
    CoarseGridStats stats;
    for (const auto& [key, state] : coarse_state_) {
        switch (state) {
        case CoarseCellState::FREE:
            stats.num_free++;
            break;
        case CoarseCellState::OCCUPIED:
            stats.num_occupied++;
            if (is_frontier_coarse(key)) {
                stats.num_frontiers++;
            }
        default:
            break;
        }
    }
    return stats;
}

} // namespace