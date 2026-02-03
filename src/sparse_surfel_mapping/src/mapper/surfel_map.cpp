#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include <cmath>

namespace sparse_surfel_map {

SurfelMap::SurfelMap()
    : config_()
    , spatial_hash_(config_.voxel_size,
                    config_.surfel_config,
                    config_.initial_bucket_count,
                    config_.max_load_factor)
    , pending_updates_()
    , total_points_integrated_(0)
    , last_association_time_ms_(0.0)
    , last_commit_time_ms_(0.0)
    , update_in_progress_(false)
{}

SurfelMap::SurfelMap(const SurfelMapConfig& config)
    : config_(config)
    , spatial_hash_(config.voxel_size,
                    config.surfel_config,
                    config.initial_bucket_count,
                    config.max_load_factor)
    , pending_updates_()
    , total_points_integrated_(0)
    , last_association_time_ms_(0.0)
    , last_commit_time_ms_(0.0)
    , update_in_progress_(false)
{}

void SurfelMap::begin_update() {
    pending_updates_.clear();
    update_in_progress_ = true;
}

void SurfelMap::associate_points(const std::vector<PointWithNormal>& pns, const Eigen::Transform<float, 3, Eigen::Isometry>& transform) {
    const auto start_time = Clock::now();

    for (const auto& pn : pns) {
        if (!is_point_in_range(pn.position)) continue;

        // Updated point transformed to global frame 
        PointWithNormal update;
        update.position = transform * pn.position;
        update.normal = transform.rotation() * pn.normal;
        update.weight = pn.weight;

        // Find corresponding voxel
        const VoxelKey key = compute_voxel_key(update.position);

        pending_updates_[key].push_back(update);
    }

    const auto end_time = Clock::now();
    last_association_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

size_t SurfelMap::commit_update() {
    const auto start_time = Clock::now();
    size_t points_integrated = 0;
    
    for (auto& [key, updates] : pending_updates_) {
        if (updates.empty()) continue;
        Voxel& voxel = spatial_hash_.get_or_create(key);
        const bool was_valid = voxel.has_valid_surfel();
        voxel.integrate_points(updates);
        voxel.finalize_surfel();
        points_integrated += updates.size();
        const bool is_valid = voxel.has_valid_surfel();

        if (!was_valid && is_valid) {
            // surfel just became valid
            spatial_hash_.on_surfel_added(key);
        }
        else if (was_valid && !is_valid) {
            // surfel just became invalid
            spatial_hash_.on_surfel_removed(key);
        }
    }

    total_points_integrated_ += points_integrated;
    pending_updates_.clear();
    update_in_progress_ = false;

    prune_shadow_surfels();

    const auto end_time = Clock::now();
    last_commit_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return points_integrated;
}

size_t SurfelMap::integrate_points(const std::vector<PointWithNormal>& pns, const Eigen::Transform<float, 3, Eigen::Isometry>& transform) {
    // full monty
    begin_update();
    associate_points(pns, transform);
    size_t integrated = commit_update();

    // Update spatial hash for frontier detection (move to integrate_points and adjust timer to wrap the calls there)
    Eigen::Vector3f position = transform.translation();
    Eigen::Vector3f forward = transform.rotation() * Eigen::Vector3f::UnitX();
    float yaw = std::atan2(forward.y(), forward.x());
    spatial_hash_.observe_frustum(position, yaw, 106.0f, 86.0f, 10.0f); // OBS: UPDATE TO SENSOR CONFIG STRUCT!

    return integrated;
}

size_t SurfelMap::pending_update_count() const {
    size_t count = 0;
    for (const auto& [key, updates] : pending_updates_) {
        count += updates.size();
    }
    return count;
}

void SurfelMap::discard_pending_update() {
    pending_updates_.clear();
    update_in_progress_ = false;
}

// Map Queries
std::optional<std::reference_wrapper<const Voxel>> SurfelMap::get_voxel_at(const Eigen::Vector3f& point) const {
    const VoxelKey key = compute_voxel_key(point);
    return spatial_hash_.get(key);
}

std::optional<std::reference_wrapper<const Voxel>> SurfelMap::get_voxel(const VoxelKey& key) const {
    return spatial_hash_.get(key);
}

std::optional<std::reference_wrapper<const Surfel>> SurfelMap::get_surfel_at(const Eigen::Vector3f& point) const {
    auto voxel_opt = get_voxel_at(point); // optional 
    if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
        return std::cref(voxel_opt->get().surfel());
    }
    return std::nullopt; // no voxel at point or no valid surfel
}

std::vector<std::reference_wrapper<const Surfel>> SurfelMap::get_valid_surfels() const {
    std::vector<std::reference_wrapper<const Surfel>> surfels;
    surfels.reserve(spatial_hash_.size());

    const auto keys = spatial_hash_.get_valid_surfel_keys();

    for (const auto& key : keys) {
        auto voxel_opt = spatial_hash_.get(key);
        if (voxel_opt) {
            surfels.push_back(std::cref(voxel_opt->get().surfel()));
        }
    }

    return surfels;
}

std::vector<std::reference_wrapper<const Surfel>> SurfelMap::query_surfels_in_box(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound) const {
    std::vector<std::reference_wrapper<const Surfel>> result;

    const VoxelKey min_key = compute_voxel_key(min_bound);
    const VoxelKey max_key = compute_voxel_key(max_bound);

    for (int32_t x = min_key.x; x <= max_key.x; ++x) {
        for (int32_t y = min_key.y; y <= max_key.y; ++y) {
            for (int32_t z = min_key.z; z <= max_key.z; ++z) {
                VoxelKey key{x, y, z};
                auto voxel_opt = spatial_hash_.get(key);
                if (voxel_opt && voxel_opt->get().has_valid_surfel()) {
                    result.push_back(std::cref(voxel_opt->get().surfel()));
                }
            }
        }
    }

    return result;
}

std::vector<std::reference_wrapper<const Surfel>> SurfelMap::query_surfels_in_radius(const Eigen::Vector3f center, float radius) const {
    std::vector<std::reference_wrapper<const Surfel>> result;

    const auto voxels = spatial_hash_.get_neighbors_in_radius(center, radius);

    for (const auto& voxel : voxels) {
        if (voxel.get().has_valid_surfel()) {
            result.push_back(std::cref(voxel.get().surfel()));
        }
    }

    return result;
}

// Map Management
void SurfelMap::clear() {
    spatial_hash_.clear();
    pending_updates_.clear();
    total_points_integrated_ = 0;
    update_in_progress_ = false;
}

void SurfelMap::reset(const SurfelMapConfig& config) {
    config_ = config;
    spatial_hash_ = SpatialHash(config.voxel_size, config.surfel_config, config.initial_bucket_count, config.max_load_factor);
    pending_updates_.clear();
    total_points_integrated_ = 0;
    update_in_progress_ = 0;
}

size_t SurfelMap::prune_invalid_surfels() {
    return spatial_hash_.prune_invalid();
}

size_t SurfelMap::prune_outside_box(const Eigen::Vector3f& min_bound, const Eigen::Vector3f& max_bound) {
    return spatial_hash_.prune_outside_bounds(min_bound, max_bound);
}

size_t SurfelMap::prune_shadow_surfels(float normal_th) {
    std::set<VoxelKey> to_remove;

    const auto keys = spatial_hash_.get_valid_surfel_keys();
    for (const auto& key : keys) {
        if (to_remove.count(key)) continue;

        auto voxel_opt = spatial_hash_.get(key);
        if (!voxel_opt || !voxel_opt->get().has_valid_surfel()) continue;

        const Surfel& surfel = voxel_opt->get().surfel();
        const Eigen::Vector3f& mean = surfel.mean();
        const Eigen::Vector3f& normal = surfel.normal();

        Eigen::Vector3f behind_point = mean - normal * config_.voxel_size;
        VoxelKey behind_key = compute_voxel_key(behind_point);
        auto behind_opt = spatial_hash_.get(behind_key);
        if (!behind_opt || !behind_opt->get().has_valid_surfel()) continue;

        const Surfel& behind_surfel = behind_opt->get().surfel();

        float normal_dot = normal.dot(behind_surfel.normal());
        if (normal_dot < normal_th) continue;
        
        float signed_dist = (behind_surfel.mean() - mean).dot(normal);
        if (signed_dist >= 0.0f) continue; // not behind

        to_remove.insert(behind_key);
    }

    for (const auto& key : to_remove) {
        spatial_hash_.remove(key);
    }

    return to_remove.size();
}

// Statistics
MapStatistics SurfelMap::get_statistics() const {
    MapStatistics stats = spatial_hash_.compute_statistics();
    stats.last_update_time_ms = last_association_time_ms_ + last_commit_time_ms_;
    stats.last_association_time_ms = last_association_time_ms_;
    stats.last_commit_time_ms = last_commit_time_ms_;
    return stats;
}

size_t SurfelMap::num_valid_surfels() const {
    size_t count = 0;
    for (const auto& [key, voxel] : spatial_hash_) {
        if (voxel.has_valid_surfel()) {
            count++;
        }
    }

    return count;
}

// Internal
bool SurfelMap::is_point_in_range(const Eigen::Vector3f& point) const {
    const float range_sq = point.squaredNorm();
    return range_sq >= config_.min_range * config_.min_range && range_sq <= config_.max_range * config_.max_range;
}

VoxelKey SurfelMap::compute_voxel_key(const Eigen::Vector3f& point) const {
    return VoxelKey{
        static_cast<int32_t>(std::floor(point.x() / config_.voxel_size)),  
        static_cast<int32_t>(std::floor(point.y() / config_.voxel_size)),  
        static_cast<int32_t>(std::floor(point.z() / config_.voxel_size))  
    };
}

} // namespace