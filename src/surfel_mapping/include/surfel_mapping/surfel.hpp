#ifndef SURFEL_HPP_
#define SURFEL_HPP_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cstdint>
#include <limits>

namespace surface_inspection_planning {

struct Surfel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Identifier
    uint64_t id = 0;

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
    uint32_t point_count = 0;

    // Confidence metric
    float confidence = 0.0f;
    float planarity = 0.0f;
    float rms_error = 0.0f;

    // Temporal info
    uint64_t creation_stamp = 0;
    uint64_t last_update_stamp = 0;
    uint32_t observation_count = 0;

    // State flags
    bool is_valid = false;
    bool needs_eigen_update = false;

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
        point_count = 1;

        confidence = 0.1f;
        planarity = 1.0f;
        rms_error = 0.0f;

        observation_count = 1;
        is_valid = true;
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
        // E[xx^T] - E[x]E[x]^T
        covariance = (sum_outer / total_weight) - (mean * mean.transpose());
        // add small regularization for stability
        covariance += Eigen::Matrix2f::Identity() * 1e-6f;
        // ensure symmetry
        covariance = (covariance + covariance.transpose()) * 0.5f;
        
        needs_eigen_update = true;
    }

    float get_radius() const {
        return std::sqrt(eigenvalues.maxCoeff());
    }

    float get_bounding_radius() const {
        return 3.0 * get_radius();
    }

};



}; // end namespace

#endif