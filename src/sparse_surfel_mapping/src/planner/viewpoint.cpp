#include "sparse_surfel_mapping/planner/viewpoint.hpp"
#include <cmath>
#include <algorithm>

namespace sparse_surfel_map {
    
// Frustum Calculator
FrustumCalculator::FrustumCalculator() : config_() {
    precompute();
}

FrustumCalculator::FrustumCalculator(const CameraConfig& config) : config_(config) {
    precompute();
}

void FrustumCalculator::precompute() {
    const float half_hfov_rad = config_.hfov_deg * 0.5f * M_PI / 180.0f;
    const float half_vfov_rad = config_.vfov_deg * 0.5f * M_PI / 180.0f;

    tan_half_hfov_ = std::tan(half_hfov_rad);
    tan_half_vfov_ = std::tan(half_vfov_rad);

    cos_max_angle_ = std::cos(config_.max_incidence_angle_deg * M_PI / 180.0f);
    min_r2_ = config_.min_range * config_.min_range;
    max_r2_ = config_.max_range * config_.max_range;
}

FrustumPlanes FrustumCalculator::compute_frustum(const Eigen::Vector3f& position, float yaw) const {
    FrustumPlanes frustum;    

    // camera coordinate frame  
    const float cos_yaw = std::cos(yaw);
    const float sin_yaw = std::sin(yaw);

    // Camera basis vectors in world frame
    const Eigen::Vector3f forward(cos_yaw, sin_yaw, 0.0f);
    const Eigen::Vector3f right(sin_yaw, -cos_yaw, 0.0f);
    const Eigen::Vector3f up = Eigen::Vector3f::UnitZ();

    const float n = config_.min_range; // near
    const float f = config_.max_range; // far

    const float nh = n * tan_half_hfov_; // near half width
    const float nv = n * tan_half_vfov_; // near half height
    const float fh = f * tan_half_hfov_; // far half width
    const float fv = f * tan_half_vfov_; // far half height

    // Near plane corners
    const Eigen::Vector3f near_center = position + forward * n;
    frustum.corners[0] = near_center + up * nv - right * nh; // near top left
    frustum.corners[1] = near_center + up * nv + right * nh; // near top right
    frustum.corners[2] = near_center - up * nv + right * nh; // near bottom right
    frustum.corners[3] = near_center - up * nv - right * nh; // near bottom left

    // Far plane corners
    const Eigen::Vector3f far_center = position + forward * f;
    frustum.corners[4] = far_center + up * fv - right * fh; // far top left
    frustum.corners[5] = far_center + up * fv + right * fh; // far top right
    frustum.corners[6] = far_center - up * fv + right * fh; // far bottom right
    frustum.corners[7] = far_center - up * fv - right * fh; // far bottom left

    // computing the 6 frustum planes normal.dot(p) + d = 0
    // normal pointing inward (towards camera)
    auto make_plane =
    [&](const Eigen::Vector3f& p0,
        const Eigen::Vector3f& p1,
        const Eigen::Vector3f& p2) -> Eigen::Vector4f
    {
        Eigen::Vector3f normal = (p1 - p0).cross(p2 - p0).normalized();
        float d = -normal.dot(p0);

        Eigen::Vector4f plane(normal.x(), normal.y(), normal.z(), d);

        // Ensure inside_point is classified as inside
        Eigen::Vector3f inside_point =
            position + forward * (config_.min_range + 1e-3f);

        if (normal.dot(inside_point) + d < 0.0f)
            plane *= -1.0f;

        return plane;
    };

    // Near plane
    frustum.planes[0] = make_plane(frustum.corners[3], frustum.corners[0],frustum.corners[1]);

    // Far plane
    frustum.planes[1] = make_plane(frustum.corners[5], frustum.corners[4], frustum.corners[7]);

    // Left plane: from near_bl -> near_tl -> far_tl
    frustum.planes[2] = make_plane(frustum.corners[3], frustum.corners[0], frustum.corners[4]);

    // Rigth plane: from near_tr -> near_br -> far_br
    frustum.planes[3] = make_plane(frustum.corners[1], frustum.corners[2], frustum.corners[6]);

    // Top plane: from near_tl -> near_tr -> far_tr
    frustum.planes[4] = make_plane(frustum.corners[0], frustum.corners[1], frustum.corners[5]);

    // Bottom plane: from near_br -> near_bl -> far_bl
    frustum.planes[5] = make_plane(frustum.corners[2], frustum.corners[3], frustum.corners[7]);

    return frustum;
}

bool FrustumCalculator::is_point_visible(const FrustumPlanes& frustum, const Eigen::Vector3f& point) const {
    return frustum.contains_point(point);
}

bool FrustumCalculator::is_voxel_visible(const FrustumPlanes& frustum, const Eigen::Vector3f& voxel_min, const Eigen::Vector3f& voxel_max) const {
    // for each frustum plane check if voxel is completely outside
    for (const auto& plane : frustum.planes) {
        const Eigen::Vector3f normal = plane.head<3>();
        const float d = plane.w();

        // voxel vertex 
        Eigen::Vector3f p_vertex;
        p_vertex.x() = (normal.x() >= 0) ? voxel_max.x() : voxel_min.x();
        p_vertex.y() = (normal.y() >= 0) ? voxel_max.y() : voxel_min.y();
        p_vertex.z() = (normal.z() >= 0) ? voxel_max.z() : voxel_min.z();

        // if p_vertex is outside -> voxel is entirely outside
        if (normal.dot(p_vertex) + d < 0) {
            return false;
        }
    }
    
    return true;
}

bool FrustumCalculator::is_surfel_visible(const FrustumPlanes& frustum, const Eigen::Vector3f& camera_position, const Eigen::Vector3f& surfel_position, const Eigen::Vector3f& surfel_normal) const {   
    Eigen::Vector3f diff = camera_position - surfel_position;
    float dist2 = diff.squaredNorm();
    if (dist2 < min_r2_ || dist2 > max_r2_) return false;
    
    float dot = surfel_normal.dot(diff);
    if (dot <= 0) return false;

    if (dot * dot < dist2 * cos_max_angle_ * cos_max_angle_) return false;
    
    return frustum.contains_point(surfel_position);
}

// --- VIEWPOINT ---
Viewpoint::Viewpoint() : state_(), frustum_calc_(), frustum_(), frustum_computed_(false) {}

Viewpoint::Viewpoint(const Eigen::Vector3f& position, float yaw, const CameraConfig& camera_config) 
    : state_()
    , frustum_calc_(camera_config)
    , frustum_()
    , frustum_computed_(false)
{
    state_.position = position;
    state_.yaw = yaw;
}

size_t Viewpoint::compute_visibility(const SurfelMap& map, bool check_occlusion) {
    if (!frustum_computed_) {
        frustum_ = frustum_calc_.compute_frustum(state_.position, state_.yaw);
        frustum_computed_ = true;
    }

    state_.visible_voxels.clear();

    const float voxel_size = map.voxel_size();
    const auto& voxels = map.voxels();

    Eigen::Vector3f frustum_min = frustum_.corners[0];
    Eigen::Vector3f frustum_max = frustum_.corners[0];
    for (const auto& corner : frustum_.corners) {
        frustum_min = frustum_min.cwiseMin(corner);
        frustum_max = frustum_max.cwiseMax(corner);
    }

    const VoxelKey min_key{
        static_cast<int32_t>(std::floor(frustum_min.x() / voxel_size)),  
        static_cast<int32_t>(std::floor(frustum_min.y() / voxel_size)),  
        static_cast<int32_t>(std::floor(frustum_min.z() / voxel_size))  
    };

    const VoxelKey max_key{
        static_cast<int32_t>(std::ceil(frustum_max.x() / voxel_size)),  
        static_cast<int32_t>(std::ceil(frustum_max.y() / voxel_size)),  
        static_cast<int32_t>(std::ceil(frustum_max.z() / voxel_size))  
    };

    for (int32_t x = min_key.x; x <= max_key.x; ++x) {
        for (int32_t y = min_key.y; y <= max_key.y; ++y) {
            for (int32_t z = min_key.z; z <= max_key.z; ++z) {
                VoxelKey key{x,y,z};

                // Check voxel center visibility
                Eigen::Vector3f voxel_min(
                    key.x * voxel_size,
                    key.y * voxel_size,
                    key.z * voxel_size
                );

                Eigen::Vector3f voxel_max = voxel_min + Eigen::Vector3f::Constant(voxel_size);
                if (!frustum_calc_.is_voxel_visible(frustum_, voxel_min, voxel_max))
                    continue;

                auto voxel_opt = voxels.get(key);
                if (!voxel_opt || !voxel_opt->get().has_valid_surfel()) {
                    continue;
                }

                const Voxel& voxel = voxel_opt->get();
                const Surfel& surfel = voxel.surfel();
                
                // Surfel visible?
                if (frustum_calc_.is_surfel_visible(frustum_, state_.position, surfel.mean(), surfel.normal())) {
                    
                    // Optional ray-cast occlusion check 
                    if (check_occlusion) {
                        bool occluded = false;

                        const Eigen::Vector3f ray_dir = (surfel.mean() - state_.position).normalized();
                        const float dist = (surfel.mean() - state_.position).norm();
                        const float step = voxel_size * 0.5f;

                        // step along ray dir
                        for (float t = step; t < dist; t += step) {
                            Eigen::Vector3f sample_point = state_.position + ray_dir * t;
                            VoxelKey sample_key{
                                static_cast<int32_t>(std::floor(sample_point.x() / voxel_size)),  
                                static_cast<int32_t>(std::floor(sample_point.y() / voxel_size)),  
                                static_cast<int32_t>(std::floor(sample_point.z() / voxel_size))  
                            };

                            if (sample_key == key) continue;

                            // hit valid surfel?
                            auto blocking_voxel = voxels.get(sample_key);
                            if (blocking_voxel && blocking_voxel->get().has_valid_surfel()) {
                                occluded = true;
                                break;
                            }
                        }

                        if (occluded) continue;
                    }

                    // mark center + 26-neighborhood visible (noise smoothing on visibility)
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        for (int32_t dy = -1; dy <= 1; ++dy) {
                            for (int32_t dz = -1; dz <= 1; ++dz) {
                                VoxelKey vis_key{key.x + dx, key.y + dy, key.z + dz};
                                const auto voxel_opt = map.get_voxel(vis_key);
                                if (voxel_opt && voxel_opt->get().is_occupied()) {
                                    state_.visible_voxels.insert(vis_key);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return state_.visible_voxels.size();
}

float Viewpoint::compute_coverage_score(const VoxelKeySet& observed_voxels) {
    state_.new_coverage_voxels.clear();

    for (const auto& key : state_.visible_voxels) {
        if (observed_voxels.find(key) == observed_voxels.end()) {
            state_.new_coverage_voxels.insert(key);
        }
    }

    // coverage score: ratio of new to total visible
    if (state_.visible_voxels.empty()) {
        state_.coverage_score = 0.0f;
        state_.overlap_score = 0.0f;
    }
    else {
        const float new_ratio = static_cast<float>(state_.new_coverage_voxels.size()) / static_cast<float>(state_.visible_voxels.size());
        state_.coverage_score = new_ratio;
        state_.overlap_score = 1.0f - new_ratio; // overlap is inverse
    }

    return state_.coverage_score;
}

void Viewpoint::compute_total_score(const VoxelKeySet& observed_voxels) {
    compute_coverage_score(observed_voxels);
    state_.total_score = state_.coverage_score;
}

bool Viewpoint::is_similar_to(const Viewpoint& other, float pos_th, float angle_th) const {
    const float pos_dist = (state_.position - other.state_.position).norm();
    if (pos_dist > pos_th) return false;

    float yaw_diff = std::abs(state_.yaw - other.state_.yaw);
    if (yaw_diff > M_PI) {
        yaw_diff = 2.0f * M_PI - yaw_diff;
    }

    const float angle_th_rad = angle_th * M_PI / 180.0f;
    return yaw_diff <= angle_th_rad;
}

bool Viewpoint::is_in_collision(const SurfelMap& map, float radius) const {
    const auto& voxels = map.voxels();
    if (voxels.empty()) return true; // cant tell -> in collision

    const auto& nb_radius = voxels.get_neighbors_in_radius(state_.position, radius);
    std::cout << nb_radius.size() << std::endl;
    if (nb_radius.size() > 0) return true; // is in collision as there are occupied voxels in

    return false;
}

float Viewpoint::distance_to_nearest_surface(const SurfelMap& map, float max_radius) const {
    const auto& voxels = map.voxels();
    if (voxels.empty()) return 0.0f;
    
    float min_dist = max_radius;
    const float voxel_size = map.voxel_size();
    const int max_voxel_radius = static_cast<int>(std::ceil(max_radius / voxel_size));
    const VoxelKey& vp_key = voxels.point_to_key(state_.position);

    for (int r = 0; r <= max_voxel_radius; ++r) {
        bool found_in_shell = false;
        
        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dz = -r; dz <= r; ++dz) {
                    
                    if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) continue; // check only in shell

                    VoxelKey key{vp_key.x + dx, vp_key.y + dy, vp_key.z + dz};
                    const auto& voxel = map.get_voxel(key);
                    if (voxel.has_value() && voxel->get().is_occupied()) {
                        const Eigen::Vector3f center = voxel->get().center(voxel_size);
                        const float dist = (center - state_.position).norm() - voxel_size * 0.5f;
                        min_dist = std::min(min_dist, std::max(0.0f, dist));
                        found_in_shell = true;
                    }
                }
            }
        }

        if (found_in_shell && min_dist < (r + 1) * voxel_size) return min_dist;
    }

    return min_dist;
}

} // namespace