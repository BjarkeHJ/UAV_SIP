#include "sparse_surfel_mapping/mapper/surfel.hpp"
#include <cmath>
#include <algorithm>

namespace sparse_surfel_map {
Surfel::Surfel()
    : mean_(Eigen::Vector3f::Zero())
    , covariance_(Eigen::Matrix3f::Zero())
    , count_(0)
    , sum_weights_(0.0f)
    , normal_(Eigen::Vector3f::Zero())
    , eigenvalues_(Eigen::Vector3f::Zero())
    , eigenvectors_(Eigen::Matrix3f::Zero())
    , is_valid_(false)
    , config_()
{
}

Surfel::Surfel(const SurfelConfig& config) : Surfel() {
    config_ = config;
}

void Surfel::integrate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, float weight) {
    if (weight <= 0.05f) {
        return;
    }

    count_++;
    sum_weights_ += weight;
    sum_weights_sq_ += weight * weight;

    const Eigen::Vector3f delta = point - mean_;
    mean_ += (weight / sum_weights_) * delta;

    // Update covariance: w * (x - mean_old) * (x - mean_new)^T
    const Eigen::Vector3f delta2 = point - mean_;
    covariance_ += weight * delta * delta2.transpose();

    // Track measurement normal (for surface orientation)
    const Eigen::Vector3f n_delta = normal - avg_measurement_normal_;
    avg_measurement_normal_ += (weight / sum_weights_) * n_delta;
}

void Surfel::integrate_points(const std::vector<PointWithNormal>& points) {
    for (const auto& update : points) {
        integrate_point(update.position, update.normal, update.weight);
    }
}

void Surfel::recompute_normal() {
    // eigen decomp
    compute_eigen_decomp();

    normal_ = eigenvectors_.col(0);
    if (normal_.dot(avg_measurement_normal_) < 0) {
        normal_ = -normal_;
    }

    // check validity of surfel
    update_validity();
}

void Surfel::compute_eigen_decomp() {
    Eigen::Matrix3f C;
    if (sum_weights_ > 1.0f) {
        C = normalized_covariance();
    }
    else {
        C = covariance_;
    }

    C = 0.5 * (C + C.transpose()); // ensure symmetry
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(C);
    if (solver.info() != Eigen::Success) {
        eigenvalues_.setZero();
        eigenvectors_.setZero();
        is_valid_ = false;
        return;
    }

    // eigenvalues in ascending order from solver
    eigenvalues_ = solver.eigenvalues();
    eigenvectors_ = solver.eigenvectors();

    // ensure non-negative eigenvalues
    for (int i = 0; i < 3; ++i) {
        if (eigenvalues_(i) < 0.0f) {
            eigenvalues_(i) = 0.0f;
        }
    }
}

void Surfel::update_validity() {
    is_valid_ = false;

    // Gate by effective samples (measurement weights)
    if (effective_samples() < 5.0f) return;

    // Reject very large eigenvalues 
    if (eigenvalues_(2) > 2.0f * config_.voxel_size * config_.voxel_size) return;

    // Gate degeneracy
    if (eigenvalues_(1) < config_.degeneracy_threshold * eigenvalues_(2)) return;

    is_valid_ = true;
}

float Surfel::effective_samples() const {
    return (sum_weights_ * sum_weights_) / (sum_weights_sq_ + 1e-6f);
}

float Surfel::confidence() const {
    float ess = effective_samples();
    if (ess < 10.0f) return 0.0f;
    float c = 1.0f - std::exp(-ess / 25.0f);
    // float c = 1.0f;
    float planarity = std::clamp(1.0f - eigenvalues_(0) / (eigenvalues_(1) + 1e-6f), 0.0f, 1.0f);
    c *= planarity; // multiply be planarity;
    return c;
}

float Surfel::observability(const Eigen::Vector3f& camera_pos, float opt_dist) const {
    // float dot = normal_.dot(-view_dir);

    Eigen::Vector3f to_camera = camera_pos - mean_;
    float dist = to_camera.norm();
    Eigen::Vector3f view_dir = to_camera / dist;
    float cos_angle = normal_.dot(view_dir);
    if (cos_angle <= 0.0f) return 0.0f; // back facing

    float dist_ratio = 1.0f;
    if (opt_dist > 0.0f) {
        dist_ratio = dist / opt_dist;
    }
    float w_distance = std::exp(-0.5f * std::pow(dist_ratio - 1.0f, 2) / 0.3f);
    float w_angle = cos_angle;
    float w_surfel = confidence();
        
    return w_distance * w_angle * w_surfel;
}

Eigen::Matrix3f Surfel::normalized_covariance() const {
    if (sum_weights_ > 1.0f) {
        return covariance_ / (sum_weights_ - 1.0f);
    }
    return covariance_;
}

void Surfel::reset() {
    mean_.setZero();
    covariance_.setZero();
    count_ = 0;
    sum_weights_ = 0.0f;
    normal_.setZero();
    eigenvalues_.setZero();
    eigenvectors_.setZero();
    is_valid_ = false;
}

}