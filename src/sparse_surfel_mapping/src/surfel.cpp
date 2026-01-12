#include "sparse_surfel_mapping/surfel.hpp"
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
    , eigen_dirty_(true)
    , config_()
{
}

Surfel::Surfel(const SurfelConfig& config) : Surfel() {
    config_ = config;
}

void Surfel::integrate_point(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, float weight) {
    if (weight <= 0.1f) {
        return;
    }

    count_++;
    const float prev_sum_weight = sum_weights_;
    sum_weights_ += weight;

    const Eigen::Vector3f delta = point - mean_;
    mean_ += (weight / sum_weights_) * delta;

    // Update covariance: w * (x - mean_old) * (x - mean_new)^T
    const Eigen::Vector3f delta2 = point - mean_;
    covariance_ += weight * delta * delta2.transpose();

    // Track measurement normal (for surface orientation)
    const Eigen::Vector3f n_delta = normal - avg_measurement_normal_;
    avg_measurement_normal_ += (weight / sum_weights_) * n_delta;

    eigen_dirty_ = true;
}

void Surfel::integrate_points(const std::vector<PointWithNormal>& points) {
    for (const auto& update : points) {
        integrate_point(update.position, update.normal, update.weight);
    }
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
    eigen_dirty_ = true;
}

void Surfel::recompute_normal() {
    if (count_ < config_.min_points_for_validity || sum_weights_ < constants::EPSILON) {
        is_valid_ = false;
        return;
    }

    // eigen decomp
    compute_eigen_decomp();

    normal_ = eigenvectors_.col(0);
    if (normal_.dot(avg_measurement_normal_) < 0) {
        normal_ = -normal_;
    }

    // check validity of surfel
    update_validity();
    eigen_dirty_ = false;
}

void Surfel::compute_eigen_decomp() {
    Eigen::Matrix3f C;
    if (sum_weights_ > 1.0f) {
        C = covariance_ / (sum_weights_ - 1.0f);
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
    if (count_ < config_.min_points_for_validity) {
        is_valid_ = false;
        return;
    }

    // at least two non-zero eigenvalues
    int non_zero_count = 0;
    for (int i = 0; i < 3; ++i) {
        if (eigenvalues_(i) > constants::EPSILON) non_zero_count++;
        if (eigenvalues_(i) > constants::MAX_EVAL) {
            is_valid_ = false;
            return;
        }
    }
    if (non_zero_count < 2) {
        is_valid_ = false;
        return;
    }

    if (eigenvalues_(1) > constants::EPSILON) {
        const float ratio = eigenvalues_(0) / eigenvalues_(1); // smallest / middle: close to 1 -> not planar!
        if (ratio >= config_.planarity_threshold * config_.max_eigenvalue_ratio) {
            is_valid_ = false;
            return;
        }
    }
    
    is_valid_ = true;
}

bool Surfel::is_planar() const {
    if (eigen_dirty_ || eigenvalues_(1) < constants::EPSILON) {
        return false;
    }
    return eigenvalues_(0) < config_.planarity_threshold * eigenvalues_(1);
}

bool Surfel::is_degenerate() const {
    if (eigen_dirty_) return true;
    return eigenvalues_(1) < config_.degeneracy_threshold * eigenvalues_(2); // lambda_1 very small compared to lambda_2
}

Eigen::Matrix3f Surfel::normalized_covariance() const {
    if (sum_weights_ > 1.0f) {
        return covariance_ / (sum_weights_ - 1.0f);
    }
    return covariance_;
}

}