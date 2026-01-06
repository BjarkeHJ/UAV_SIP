#ifndef SURFEL_HPP_
#define SURFEL_HPP_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cstdint>
#include <limits>

namespace surface_inspection_planning {

struct ConfidenceParams {
    float support_halflife = 100.0f;      // Points for c_support ≈ 0.63
    float fit_sigma = 0.02f;              // RMS error (m) for c_fit ≈ 0.37
    float observation_halflife = 5.0f;    // Frames for c_temporal ≈ 0.63
    float temporal_weight = 0.3f;         // Balance between frame count and diversity
    float area_scale = 0.05f;
};

struct Surfel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Identifier
    uint64_t id = 0;
    int32_t voxel_x = 0;
    int32_t voxel_y = 0;
    int32_t voxel_z = 0;

    // Geometric Properties (World Frame)
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f tangent_u = Eigen::Vector3f::UnitX();
    Eigen::Vector3f tangent_v = Eigen::Vector3f::UnitY();

    // 2D Gaussian in tangent plane
    Eigen::Matrix2f covariance = Eigen::Matrix2f::Identity() * 0.01f;
    Eigen::Vector2f eigenvalues = Eigen::Vector2f::Constant(0.01f);
    Eigen::Matrix2f eigenvectors = Eigen::Matrix2f::Identity();

    // Statistics for incremental updates
    float total_weight = 0.0f;
    Eigen::Vector2f sum_tangent = Eigen::Vector2f::Zero();
    Eigen::Matrix2f sum_outer = Eigen::Matrix2f::Zero();
    Eigen::Vector3f sum_normals = Eigen::Vector3f::Zero();
    float sum_sq_normal_dist = 0.0f;
    uint32_t point_count = 0;
    uint32_t observation_count = 0;

    // Confidence metric
    float confidence = 0.0f;
    float planarity = 0.0f;
    float rms_error = 0.0f;

    // Temporal info
    uint64_t creation_stamp = 0;
    uint64_t last_update_stamp = 0;

    // State flags
    bool is_valid = false;
    bool is_mature = false;
    bool needs_eigen_update = false;
    bool observed = false; // inspection state (is in fov of a viewpoint)

    void initialize(const Eigen::Vector3f& init_center, const Eigen::Vector3f& init_normal, float init_radius = 0.1f) {
        center = init_center;
        normal = init_normal.normalized();
        compute_tangential_basis();

        float var = init_radius * init_radius;
        covariance = Eigen::Matrix2f::Identity() * var;
        eigenvalues = Eigen::Vector2f::Constant(var);
        eigenvectors = Eigen::Matrix2f::Identity();

        total_weight = 1.0f;
        sum_tangent = Eigen::Vector2f::Zero();
        sum_outer = Eigen::Matrix2f::Zero();
        sum_normals = normal;
        sum_sq_normal_dist = 0.0f;
        point_count = 1;
        observation_count = 1;

        confidence = 0.01f;
        planarity = 1.0f;
        rms_error = 0.0f;

        // is_valid = true;
        is_valid = false;
        needs_eigen_update = false;
    }

    void compute_tangential_basis() {
        Eigen::Vector3f ref = (std::abs(normal.z()) < 0.9f) ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitX();
        tangent_u = normal.cross(ref).normalized();
        tangent_v = normal.cross(tangent_u).normalized();
    }

    std::pair<Eigen::Vector2f, float> project_point(const Eigen::Vector3f& point) const {
        Eigen::Vector3f diff = point - center;
        float normal_dist = diff.dot(normal);
        Eigen::Vector3f projected = point - normal_dist * normal;
        Eigen::Vector3f in_plane = projected - center;
        Eigen::Vector2f tangent_coords(in_plane.dot(tangent_u), in_plane.dot(tangent_v));
        return {tangent_coords, normal_dist};
    }

    float mahalanobis_distance_sq(const Eigen::Vector2f& tangent_coords) const {
        Eigen::Vector2f centered = tangent_coords - get_centered_tangent();
        Eigen::Vector2f in_eigen = eigenvectors.transpose() * centered; // dot product between centered and each eigenvector

        float d_sq = 0.0f;
        for (int i = 0; i < 2; ++i) {
            if (eigenvalues(i) > 1e-8f) {
                d_sq += (in_eigen(i) * in_eigen(i)) / eigenvalues(i);
            }
        }

        return d_sq;
    }

    Eigen::Vector2f get_centered_tangent() const {
        if (point_count > 0 && total_weight > 1e-6f) {
            return sum_tangent / total_weight;
        }
        return Eigen::Vector2f::Zero();
    }

    void update_eigen() {
        if (!needs_eigen_update) return;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(covariance);
        if (es.info() == Eigen::Success) {
            eigenvalues = es.eigenvalues();
            eigenvectors = es.eigenvectors();

            eigenvalues = eigenvalues.cwiseMax(1e-8f);

            float lambda_max = eigenvalues.maxCoeff();
            float lambda_min = eigenvalues.minCoeff();
            if (lambda_max > 1e-8f) {
                planarity = (lambda_max - lambda_min) / lambda_max;
            }
        }
        needs_eigen_update = false;
    }

    void recompute_covariance() {
        if (point_count < 3 || total_weight < 1e-6f) return;
        Eigen::Vector2f mean = sum_tangent / total_weight;
        covariance = (sum_outer / total_weight) - (mean * mean.transpose()); // E[xx^T] - E[x]E[x]^T
        covariance += Eigen::Matrix2f::Identity() * 1e-6f; // regularization
        covariance = (covariance + covariance.transpose()) * 0.5f; // ensure symmetry
        needs_eigen_update = true;
    }

    void update_confidence(const ConfidenceParams& params) {
        // Support: more points -> higher confidence
        float c_support = 1.0f - std::exp(-static_cast<float>(point_count) / params.support_halflife);
        // Fit: lower RMS error -> higher confidence
        float rms = get_rms_error();
        float c_fit = std::exp(-(rms * rms) / (params.fit_sigma * params.fit_sigma));
        // Planar: 
        // Normal Consistency: Better alignment -> higher confidence
        float c_normal = get_normal_consistency();
        // Temporal: Seen accros multiple frames -> higher confidence
        float c_temporal = get_temporal_score(params.observation_halflife, params.temporal_weight);
        // Area: Larger surfel -> important/confident patch
        float area = M_PI * std::sqrt(eigenvalues(0)) * std::sqrt(eigenvalues(1));
        float c_area = 1.0f - std::exp(-area / params.area_scale);

        confidence = c_support * c_fit * c_normal * c_temporal * c_area;
    }

    float get_rms_error() const {
        return (point_count > 0) ? std::sqrt(sum_sq_normal_dist) / static_cast<float>(point_count) : 0.0f;
    }

    float get_planarity() const {
        float lambda_max = eigenvalues.maxCoeff();
        float lambda_min = eigenvalues.minCoeff();
        return (lambda_max > 1e-8f) ? (lambda_max - lambda_min) / lambda_max : 0.0f;
    }

    float get_normal_consistency() const {
        if (total_weight < 1e-6f) return 0.0f;
        // if fused normals are identical -> ||sum_normals|| = total_weight
        float consistency_raw = sum_normals.norm() / total_weight;
        return std::clamp(std::pow(consistency_raw, 2.0f), 0.0f, 1.0f);
    }

    float get_temporal_score(float halflife, float diversity_w) const {
        float saturation = 1.0f - std::exp(-static_cast<float>(observation_count) / halflife);
        float diversity = (point_count > 0) ? static_cast<float>(observation_count) / static_cast<float>(point_count) : 0.0f;
        diversity = std::min(1.0f, diversity);
        return saturation * (1.0f - diversity) + diversity * diversity_w;
    }

    float get_radius() const {
        return std::sqrt(eigenvalues.maxCoeff());
    }

    float get_bounding_radius() const {
        return 3.0 * get_radius();
    }

    void update_maturity(float min_radius) {
        is_mature = is_valid && get_radius() >= min_radius;
    }

};

}; // end namespace

#endif