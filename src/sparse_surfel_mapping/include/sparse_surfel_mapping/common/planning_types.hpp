#ifndef PLANNING_TYPES_HPP_
#define PLANNING_TYPES_HPP_

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "sparse_surfel_mapping/common/mapping_types.hpp"

namespace sparse_surfel_map {

using VoxelKeySet = std::unordered_set<VoxelKey, VoxelKeyHash>;

struct CameraConfig {
    float hfov_deg{90.0f}; // half fov for ensure quality (??)
    float vfov_deg{60.0f}; 

    float min_range{0.1f};
    float max_range{10.0f};

    float max_incidence_angle_deg{75.0f};  
};

struct CollisionConfig {
    float drone_radius{0.5f};
    float safety_margin{0.2f};
    float path_resolution{0.3f}; // voxel size?
    float inflation_radius() const { return drone_radius + safety_margin; }
};

struct ViewpointConfig {
    // Viewpoint Geometry
    float optimal_view_distance{1.5f};
    float min_view_distance{1.0f};
    float max_view_distance{3.0f};

    // Region growing parameters
    size_t max_chain_length{5};
    float max_expansion_radius{3.0f};

    // Frontier clustering
    float frontier_cluster_radius{0.5f};
    size_t min_cluster_size{1};

    // Coverage overlap
    float target_overlap_ratio{0.20f}; // adjacent viewpoints has ~20% surface overlap
    float min_overlap_ratio{0.10f};
    float max_overlap_ratio{0.40f};
    float min_new_coverage_ratio{0.01f};

    // Scoring
    float new_coverage_weight{0.5f};
    float frontier_priority_weight{0.3f};
    float distance_weight{0.2f};
};

struct InspectionPlannerConfig {
    CameraConfig camera;
    CollisionConfig collision;
    ViewpointConfig viewpoint;

    // Planning strategy
    size_t max_viewpoints_per_plan{5};

    // Path commitment
    size_t commit_horizon{3}; // dont replan (unless collision detected)
    size_t min_horizon_buffer{2};  // extend path trigger if: total_planned - commit_horizon < min_horizon_buffer

    // Termination
    float target_coverage_ratio{0.95f};
    size_t max_total_viewpoints{1000};

    // Visited viewpoint tracking
    float revisit_distance_th{0.5f};
    float revist_angle_th_deg{15.0f};

    bool debug_output{true};
};

enum class ViewpointStatus {
    CANDIDATE,
    PLANNED,
    COMMITED,
    ACTIVE,
    VISITED,
    UNREACHABLE,
    SKIPPED
};

struct ViewpointState {
    uint64_t id{0};

    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    float yaw{0.0f}; // CCW

    VoxelKeySet visible_voxels;
    VoxelKeySet new_coverage_voxels;

    float coverage_score{0.0f};
    float overlap_score{0.0f};
    float distance_score{0.0f};
    float total_score{0.0f};

    ViewpointStatus status{ViewpointStatus::CANDIDATE};
    float path_cost{std::numeric_limits<float>::infinity()};
    double timestamp_visited{0.0};

    Eigen::Vector3f forward_direction() const {
        return Eigen::Vector3f(std::cos(yaw), std::sin(yaw), 0.0f);
    }
};

struct FrustumPlanes {
    // a plane is stored as the normal + the plane thickness
    // plane equation then says normal.dot(point) + d >= 0 means inside plane
    std::array<Eigen::Vector4f, 6> planes; // near, far, left, right, top, bottom
    std::array<Eigen::Vector3f, 8> corners;

    bool contains_point(const Eigen::Vector3f& point) const {
        for (const auto& plane : planes) {
            float dist = plane.head<3>().dot(point) + plane.w();
            if (dist < 0) return false;
        }

        return true;
    }
};

struct FrontierSurfel {
    VoxelKey key;
    Eigen::Vector3f position;
    Eigen::Vector3f normal;

    float distance_to_expansion{0.0f};
    size_t uncovered_neighbor_count{0};
    float frontier_score{0.0f};

    bool is_covered{false};
};

struct FrontierCluster {
    std::vector<FrontierSurfel> surfels;

    Eigen::Vector3f centroid{Eigen::Vector3f::Zero()};
    Eigen::Vector3f mean_normal{Eigen::Vector3f::Zero()};
    float total_priority{0.0f};

    Eigen::Vector3f suggested_view_position{Eigen::Vector3f::Zero()};
    float suggested_yaw;

    void compute_centroid() {
        if (surfels.empty()) return;
        centroid = Eigen::Vector3f::Zero();
        mean_normal = Eigen::Vector3f::Zero();
        total_priority = 0.0f;

        for (const auto& s : surfels) {
            centroid += s.position;
            mean_normal += s.normal;
            total_priority += s.frontier_score;
        }

        centroid /= static_cast<float>(surfels.size());
        mean_normal.normalize();
    }
};

struct InspectionPath {
    std::vector<Eigen::Vector3f> waypoints;
    std::vector<float> yaw_angles;
    float total_length{0.0f};
    bool is_valid{false};

    size_t size() const { return waypoints.size(); }
    bool empty() const { return waypoints.empty(); }

    void compute_length() {
        total_length = 0.0f;
        for (size_t i = 1; i < waypoints.size(); ++i) {
            total_length += (waypoints[i] - waypoints[i-1]).norm();
        }
    }

    void clear() {
        waypoints.clear();
        yaw_angles.clear();
        total_length = 0.0f;
        is_valid = false;
    }
};

enum class PathSafetyStatus {
    SAFE, // path collision-free
    COLLISION_COMMITED, // collision in committed segment - EMERGENCY
    COLLISION_UNCOMMITED, // collision in uncommitted segment - need re-extension
    INVALID_PATH, // path is invalid or empty
    NO_MAP // map not available
};

struct PathEvaluationResult {
    PathSafetyStatus status{PathSafetyStatus::INVALID_PATH};

    int first_collision_index{-1};
    Eigen::Vector3f collision_point{Eigen::Vector3f::Zero()};
    float collision_distance{std::numeric_limits<float>::infinity()};
    bool collision_in_commited{false};
    int collision_segment{-1}; // path index of start of collision segment
    float min_clearance{std::numeric_limits<float>::infinity()};
    int min_clearance_index{-1};

    bool is_safe() const { return status == PathSafetyStatus::SAFE; }
    bool need_emergency_stop() const { return status == PathSafetyStatus::COLLISION_COMMITED; }
    bool needs_reextension() const { return status == PathSafetyStatus::COLLISION_UNCOMMITED; }
};



// Statistics
struct PlanningStatistics {
    size_t total_surfles{0};
    size_t covered_surfels{0};
    float coverage_ratio{0.0f};

    size_t viewpoints_visited{0};
    size_t viewpoints_planned{0};
    size_t viewpoints_rejected{0};

    double total_planning_time_ms{0.0};

    float total_path_length{0.0f};
    float average_coverage_per_viewpoint{0.0f};

    size_t path_evaluations{0};
    size_t path_extensions{0};
};




} // namespace

#endif