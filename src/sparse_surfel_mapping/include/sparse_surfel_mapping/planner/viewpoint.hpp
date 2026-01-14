#ifndef VIEWPOINT_HPP_
#define VIEWPOINT_HPP_

#include <iostream>

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class FrustumCalculator {
public:
    FrustumCalculator();
    explicit FrustumCalculator(const CameraConfig& config);

    FrustumPlanes compute_frustum(const Eigen::Vector3f& position, float yaw) const;
    bool is_point_visible(const FrustumPlanes& frustum, const Eigen::Vector3f& point) const;
    bool is_voxel_visible(const FrustumPlanes& frustum, const Eigen::Vector3f& voxel_min, const Eigen::Vector3f& voxel_max) const;
    bool is_surfel_visible(const FrustumPlanes& frustum, const Eigen::Vector3f& camera_position, float yaw, const Eigen::Vector3f& surfel_position, const Eigen::Vector3f& surfel_normal) const;

private:
    void precompute();
    CameraConfig config_;

    float tan_half_hfov_;
    float tan_half_vfov_;
    float cos_max_angle_;
};


class Viewpoint {
public:
    Viewpoint();
    Viewpoint(const Eigen::Vector3f& position, float yaw, const CameraConfig& camera_config);
    Viewpoint(const Viewpoint& other) = default;
    Viewpoint(Viewpoint&& other) noexcept = default;
    Viewpoint& operator=(const Viewpoint& other) = default;
    Viewpoint& operator=(Viewpoint&& other) noexcept = default;
    ~Viewpoint() = default;

    size_t compute_visibility(const SurfelMap& map, bool check_occlusion = false);
    float compute_coverage_score(const VoxelKeySet& observed_voxels, const ViewpointConfig& config);
    void compute_total_score(const Eigen::Vector3f& current_pos, const VoxelKeySet& observed_voxels, const ViewpointConfig& config);

    ViewpointState& state() { return state_; }
    const ViewpointState& state() const { return state_; }
    // convenience accessors
    const Eigen::Vector3f& position() const { return state_.position; }
    float yaw() const { return state_.yaw; }
    uint64_t id() const { return state_.id; }
    ViewpointStatus status() const { return state_.status; }

    void set_position(const Eigen::Vector3f& pos) { state_.position = pos; }
    void set_yaw(float yaw) { state_.yaw = yaw; }
    void set_id(uint64_t id) { state_.id = id; }
    void set_status(ViewpointStatus status) { state_.status = status; }
    
    // Visibility results
    const VoxelKeySet& visible_voxels() const { return state_.visible_voxels; }
    const VoxelKeySet& new_coverage_voxels() const { return state_.new_coverage_voxels; }
    size_t num_visible() const { return state_.visible_voxels.size(); }
    size_t num_new_coverage() const { return state_.new_coverage_voxels.size(); }

    float coverage_score() const { return state_.coverage_score; }
    float overlap_score() const { return state_.overlap_score; }
    float distance_score() const { return state_.distance_score; }
    float total_score() const { return state_.total_score; }

    // frustum acces
    const FrustumPlanes& frustum() const { return frustum_; }

    bool is_similar_to(const Viewpoint& other, float pos_th, float angle_th) const;

private:
    ViewpointState state_;
    FrustumCalculator frustum_calc_;
    FrustumPlanes frustum_;
    bool frustum_computed_{false};
};

// for viewpoint priority ordering
struct ViewpointComparator {
    bool operator()(const Viewpoint& a, const Viewpoint& b) const {
        return a.total_score() < b.total_score();
    }
};

} // namespace

#endif