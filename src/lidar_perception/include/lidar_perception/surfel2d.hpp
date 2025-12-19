#ifndef SURFEL2D_HPP_
#define SURFEL2D_HPP_

#include <Eigen/Dense>

namespace surface_inspection_planner {

struct Surfel2D {
    // 3D Properties
    Eigen::Vector3f center;
    Eigen::Vector3f normal;

    // Tangent frame basis vectors (Orthonormal)
    Eigen::Vector3f u_axis;
    Eigen::Vector3f v_axis;

    // 2D Gaussian parameters
    Eigen::Matrix2f covariance;
    Eigen::Vector2f eigen_values;
    Eigen::Matrix2f eigen_vectors;

    // Quality metrics
    float confidence;
    float radius;
    int num_points;

    // Identifier
    int id;
    double timestamp;
    
    Surfel2D()
        : center(Eigen::Vector3f::Zero()),
          normal(Eigen::Vector3f::UnitZ()),
          u_axis(Eigen::Vector3f::UnitX()),
          v_axis(Eigen::Vector3f::UnitY()),
          covariance(Eigen::Matrix2f::Identity()),
          eigen_values(Eigen::Vector2f::Ones()),
          eigen_vectors(Eigen::Matrix2f::Identity()),
          confidence(0.0f),
          radius(0.0f),
          num_points(0),
          id(-1),
          timestamp(0.0) {};

    bool isValid() const {
        return num_points > 0 &&
               confidence > 0.0f && 
               !std::isnan(center.norm()) && 
               !std::isnan(normal.norm());
    }

    float getAspectRatio() const {
        if (eigen_values(1) < 1e-6f) return 1.0f;
        return std::sqrt(eigen_values(0) / eigen_values(1));
    }
};

}; // end namespace


#endif