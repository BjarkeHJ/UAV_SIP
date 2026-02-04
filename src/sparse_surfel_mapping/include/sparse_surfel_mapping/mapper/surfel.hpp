#ifndef SURFEL_HPP_
#define SURFEL_HPP_

#include <Eigen/Eigenvalues>
#include "sparse_surfel_mapping/common/mapping_types.hpp"

namespace sparse_surfel_map {

class Surfel {
public:
    Surfel();
    explicit Surfel(const SurfelConfig& config);
    Surfel(const Surfel& other) = default; // copy constructor
    Surfel(Surfel&& other) noexcept = default; // move constructor 
    Surfel& operator=(const Surfel& other) = default; // copy assignment
    Surfel& operator=(Surfel&& other) noexcept = default; // move assignment 
    ~Surfel() = default; // destructor

    void integrate_point(const Eigen::Vector3f& point, 
                         const Eigen::Vector3f& normal, 
                         float weight = 1.0f);
    
    void integrate_points(const std::vector<PointWithNormal>& points);
    void reset();
    void recompute_normal();
    bool is_valid() const { return is_valid_; }

    // Accessors
    const Eigen::Vector3f& mean() const { return mean_; }
    const Eigen::Vector3f& normal() const { return normal_; }
    const Eigen::Matrix3f& covariance() const { return covariance_; }
    Eigen::Matrix3f normalized_covariance() const;
    const Eigen::Vector3f eigenvalues() const { return eigenvalues_; }
    const Eigen::Matrix3f eigenvectors() const { return eigenvectors_; }

    size_t point_count() const { return count_; }
    float confidence() const;
    float sum_weights() const { return sum_weights_; }
    float effective_samples() const;    
    float observability(const Eigen::Vector3f& view_dir, float opt_dist = 0.0f) const; // view dir is camera sensor forward in global frame

    const VoxelKey& key() const { return key_; }
    void set_key(const VoxelKey& key) { key_ = key; }

    void set_config(const SurfelConfig& config) { config_ = config; }
    const SurfelConfig& get_config() const { return config_; }
    
private:
    void compute_eigen_decomp();
    void update_validity();
    
    // Surfel statistics
    Eigen::Vector3f mean_{Eigen::Vector3f::Zero()};
    Eigen::Matrix3f covariance_{Eigen::Matrix3f::Zero()};
    size_t count_{0};
    float sum_weights_{0.0f};
    float sum_weights_sq_{0.0f};
    
    // Derived quantities
    Eigen::Vector3f normal_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f eigenvalues_{Eigen::Vector3f::Zero()};
    Eigen::Matrix3f eigenvectors_{Eigen::Matrix3f::Zero()};

    // View direction
    Eigen::Vector3f avg_measurement_normal_{Eigen::Vector3f::Zero()};

    // State flags
    bool is_valid_{false};
    VoxelKey key_;

    // Surfel Configuration
    SurfelConfig config_;
};

} // namspace sparse_surfel_map

#endif