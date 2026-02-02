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

    const Eigen::Vector3f forward = state_.forward_direction();
    const Eigen::Vector3f up = Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f right = forward.cross(up);
    
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

    constexpr int BUFFER_W = 64;
    constexpr int BUFFER_H = 48;
    constexpr int BUFFER_SIZE = BUFFER_W * BUFFER_H; // depth image size

    const Eigen::Vector3f near_center = 0.25 * (frustum_.corners[0] + frustum_.corners[1] + frustum_.corners[2] + frustum_.corners[3]);
    const float near_dist = forward.dot(near_center - state_.position);

    const float tan_half_hfov = std::abs(right.dot(frustum_.corners[1] - near_center)) / near_dist;
    const float tan_half_vfov = std::abs(up.dot(frustum_.corners[0] - near_center)) / near_dist;

    const float depth_tol = voxel_size * 2.0f;

    struct Candidate {
        VoxelKey key;
        int px, py;
        float depth;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(2048);

    auto project_to_pixel = [&](const Eigen::Vector3f& world_pos, int& px, int& py, float& depth) -> bool {
        const Eigen::Vector3f local = world_pos - state_.position;
        depth = forward.dot(local);

        if (depth <= 0.0f) return false;

        const float x_cam = right.dot(local);
        const float y_cam = up.dot(local);

        const float u_ndc = x_cam / (depth * tan_half_hfov);
        const float v_ndc = y_cam / (depth * tan_half_vfov);

        if (u_ndc < -1.0f || u_ndc > 1.0f || v_ndc < -1.0f || v_ndc > 1.0f) return false;

        px = std::clamp(static_cast<int>((u_ndc + 1.0f) * 0.5f * BUFFER_W), 0, BUFFER_W - 1);
        py = std::clamp(static_cast<int>((v_ndc + 1.0f) * 0.5f * BUFFER_H), 0, BUFFER_H - 1);
        return true;
    };

    // First pass: Determine all visible surfels in frustum
    for (int32_t x = min_key.x; x <= max_key.x; ++x) {
        for (int32_t y = min_key.y; y <= max_key.y; ++y) {
            for (int32_t z = min_key.z; z <= max_key.z; ++z) {
                const VoxelKey key{x, y, z};
                const auto voxel_opt = voxels.get(key);

                if (!voxel_opt || !voxel_opt->get().has_valid_surfel()) {
                    continue;
                }

                const Surfel& surfel = voxel_opt->get().surfel();

                if (!frustum_calc_.is_surfel_visible(frustum_, state_.position, surfel.mean(), surfel.normal())) {
                    continue;
                }

                int px, py;
                float depth; 
                if (project_to_pixel(surfel.mean(), px, py, depth)) {
                    candidates.push_back({key, px, py, depth});
                }
            }
        }
    }

    if (candidates.empty()) return 0;

    // Second pass: Occlusion checking (depth selection)
    if (!check_occlusion) {
        for (const auto& c : candidates) {
            state_.visible_voxels.insert(c.key);
        }
    }
    else {
        std::array<float, BUFFER_SIZE> depth_buffer;
        depth_buffer.fill(std::numeric_limits<float>::max());

        for (const auto& c : candidates) {
            const int idx = c.py * BUFFER_W + c.px; // determine candidate image pixel
            depth_buffer[idx] = std::min(depth_buffer[idx], c.depth); // keep minimum distance
        }

        // Keep closest plus a small tolerance in depth (slightly occluded but visible)
        for (const auto& c : candidates) {
            const int idx = c.py * BUFFER_W + c.px;
            if (c.depth <= depth_buffer[idx] + depth_tol) {
                state_.visible_voxels.insert(c.key);
            }
        }
    }

    return state_.visible_voxels.size();
}

float Viewpoint::compute_coverage_score(const VoxelKeySet& observed_voxels) {
    size_t new_count = 0;
    for (const auto& key : state_.visible_voxels) {
        if (observed_voxels.find(key) == observed_voxels.end()) {
            new_count++;
        }
    }

    // coverage score: ratio of new to total visible
    if (state_.visible_voxels.empty()) {
        state_.coverage_score = 0.0f;
    }
    else {
        const float new_ratio = static_cast<float>(new_count) / static_cast<float>(state_.visible_voxels.size());
        state_.coverage_score = new_ratio;
    }

    return state_.coverage_score;
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