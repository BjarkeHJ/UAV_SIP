#ifndef SURFEL_HPP_
#define SURFEL_HPP_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cstdint>
#include <limits>

namespace surface_inspection_planning {

struct VoxelKey {
    int32_t x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct ConfidenceParams {
    float support_halflife = 100.0f;      // Points for c_support ≈ 0.63
    float fit_sigma = 0.02f;              // RMS error (m) for c_fit ≈ 0.37
    float observation_halflife = 5.0f;    // Frames for c_temporal ≈ 0.63
    float temporal_weight = 0.3f;         // Balance between frame count and diversity
};

struct Surfel {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Identifier
    uint64_t id;
    VoxelKey voxel_key;

    // Geometric Properties (World Frame)
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f tangent_u = Eigen::Vector3f::UnitX();
    Eigen::Vector3f tangent_v = Eigen::Vector3f::UnitY();

    // 2D Gaussian in tangent plane
    Eigen::Matrix2f covariance = Eigen::Matrix2f::Identity() * 0.01f;
    Eigen::Vector2f eigenvalues = Eigen::Vector2f::Constant(0.01f);
    Eigen::Matrix2f eigenvectors = Eigen::Matrix2f::Identity();

    // 1D Gaussian along normal (2.5D component)
    float normal_variance = 0.001f;

    // Statistics for incremental updates
    float total_weight = 0.0f;
    uint32_t point_count = 0;
    uint32_t observation_count = 0;

    Eigen::Matrix2f M2_tangent = Eigen::Matrix2f::Zero(); // tangent plane statistics Cov = M2_tangent / total_weight
    float mean_normal_dist = 0.0f;
    float M2_normal = 0.0f; // normal direction statistics Var = M2_normal / total_weight
    Eigen::Vector3f sum_normals = Eigen::Vector3f::Zero();

    // Confidence metric
    float confidence = 0.0f;

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
        normal_variance = 0.1f * var;

        total_weight = 1.0f;
        point_count = 1;
        observation_count = 1;

        M2_tangent = Eigen::Matrix2f::Zero();
        mean_normal_dist = 0.0f;
        M2_normal = 0.0f;
        sum_normals = normal;
        
        confidence = 0.01f;

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
        Eigen::Vector3f in_plane = diff - normal_dist * normal;
        Eigen::Vector2f tangent_coords(in_plane.dot(tangent_u), in_plane.dot(tangent_v));
        return {tangent_coords, normal_dist};
    }

    float mahalanobis_distance_3d(const Eigen::Vector3f point) const {
        /* Main */
        auto [tangent_coords, normal_dist] = project_point(point);
        
        Eigen::Vector2f in_eigen = eigenvectors.transpose() * tangent_coords;
        float mahal_tangent = 0.0f;
        for (int i = 0; i < 2; ++i) {
            if (eigenvalues(i) > 1e-8f) {
                mahal_tangent += (in_eigen(i) * in_eigen(i)) / eigenvalues(i);
            }
        }

        float mahal_normal = 0.0f;
        if (normal_variance > 1e-8f) {
            float centered = normal_dist - mean_normal_dist;
            mahal_normal = (centered * centered) / normal_variance;
        }

        return mahal_tangent + mahal_normal;
    }

    bool update(const Eigen::Vector3f& new_point, const Eigen::Vector3f& new_normal, float weight = 1.0f) {
        /* Update surfel statistics + center + normal */
        if (weight <= 0.0f) return false;

        // transform point to tangent_basis (3D)
        auto [tangent_coords, normal_dist] = project_point(new_point);
        Eigen::Vector3f tangent_offset = tangent_coords.x() * tangent_u + tangent_coords.y() * tangent_v;
        Eigen::Vector3f point_on_plane = center + tangent_offset;

        float old_w = total_weight;
        float new_w = old_w + weight;

        // Update surfel center
        Eigen::Vector3f delta_center = point_on_plane - center;
        center += (weight / new_w) * delta_center * 0.3; 

        // Update surfel normal
        sum_normals += weight * new_normal;
        Eigen::Vector3f avg_normal = sum_normals.normalized();
        normal = (normal + 0.1 * (avg_normal - normal)).normalized();
        compute_tangential_basis(); // udapte tangential basis

        // Update tangent covariance
        auto [new_tangent, _] = project_point(point_on_plane); // 2D point
        M2_tangent += weight * (tangent_coords * new_tangent.transpose());

        // Update normal variance
        float delta_n = normal_dist - mean_normal_dist;
        mean_normal_dist += (weight / new_w) * delta_n;
        float delta2_n = normal_dist - mean_normal_dist; // delta after mean adjust
        M2_normal += weight * delta_n * delta2_n;

        total_weight = new_w;
        point_count++;

        return true;
    }

    void recompute_covariance() {
        if (point_count < 3 || total_weight < 1e-6f) return;
        covariance = M2_tangent / total_weight;
        covariance += Eigen::Matrix2f::Identity() * 1e-6f;
        covariance = (covariance + covariance.transpose()) * 0.5f;
        needs_eigen_update = true;
    }

    void recompute_normal_variance() {
        if (point_count < 2 || total_weight < 1e-6f) return;
        normal_variance = std::max(M2_normal / total_weight, 1e-8f);
    }

    void update_eigen() {
        if (!needs_eigen_update) return;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(covariance);
        if (es.info() == Eigen::Success) {
            eigenvalues = es.eigenvalues().cwiseMax(1e-8f);
            eigenvectors = es.eigenvectors();
        }
        needs_eigen_update = false;
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

        confidence = c_support * c_fit * c_normal * c_temporal;
    }

    Eigen::Matrix3f get_orientation_matrix() const {
        Eigen::Vector2f ev1_2d = eigenvectors.col(0);
        Eigen::Vector2f ev2_2d = eigenvectors.col(1);
        Eigen::Vector3f principal_1 = (ev1_2d.x() * tangent_u + ev1_2d.y() * tangent_v).normalized();
        Eigen::Vector3f principal_2 = (ev2_2d.x() * tangent_u + ev2_2d.y() * tangent_v).normalized();
    
        Eigen::Matrix3f R;
        R.col(0) = principal_1;
        R.col(1) = principal_2;
        R.col(2) = normal;

        if (R.determinant() < 0) {
            R.col(1) = -R.col(1);
        }
        
        return R;
    }


    /* Accessors - need cleanup */

    float get_rms_error() const {
        if (point_count <= 1) return 0.0f;
        return std::sqrt(M2_normal / static_cast<float>(point_count));
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