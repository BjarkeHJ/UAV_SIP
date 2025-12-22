#include "lidar_perception/surfel_extractor.hpp"
#include <numeric>
#include <queue>

namespace surface_inspection_planner {

SurfelExtractor::SurfelExtractor(const SurfelExtractionConfig& config) : config_(config) {
    kdtree_ = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
}

std::vector<Surfel2D> SurfelExtractor::extract_surfels(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    if (!cloud || cloud->empty()) return {};
    if (!normals || normals->size() != cloud->size()) { 
        throw std::runtime_error("Normals size must match cloud size");
        return {};
    }

    build_kdtree(cloud);

    // Region growing wrt. surface normal similarity
    std::vector<std::vector<int>> patches = segment_into_patches(cloud, normals);

    std::vector<Surfel2D> surfels;
    surfels.reserve(patches.size());

    for (size_t i = 0; i < patches.size(); ++i) {
        if (patches[i].size() < static_cast<size_t>(config_.min_points_per_surfel)) continue;

        Surfel2D surfel = fit_surfel_to_patch(cloud, normals, patches[i]);
        surfel.id = static_cast<int>(i);

        if (is_valid_surfel(surfel)) {
            surfels.push_back(surfel);
        }
    }

    return surfels;
}

void SurfelExtractor::build_kdtree(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    kdtree_->setInputCloud(cloud);
}

std::vector<std::vector<int>> SurfelExtractor::segment_into_patches(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    /**
     * Region growing patch segmentation
     * 
     */

    const size_t num_points = cloud->size();
    labels_.assign(num_points, -1); // -1 unlabeled

    std::vector<std::vector<int>> patches;
    int current_label = 0;

    std::vector<float> normal_consistency(num_points, 0.0f); // area curvature-ish
    
    for (size_t i = 0; i < num_points; ++i) {
        if (!std::isfinite(normals->points[i].normal_x)) {
            normal_consistency[i] = -1.0f;
            continue;
        }
        
        std::vector<int> nbs;
        std::vector<float> d2s;
        kdtree_->nearestKSearch(cloud->points[i], config_.knn_neighbors, nbs, d2s);
        
        Eigen::Vector3f ref_normal
            (normals->points[i].normal_x,
            normals->points[i].normal_y,
            normals->points[i].normal_z
        );
            
        float consistency = 0.0f;
        int valid_nbs = 0;
    
        for (int nb_idx : nbs) {
            if (!std::isfinite(normals->points[nb_idx].normal_x)) continue;
            
            Eigen::Vector3f nb_normal(
                normals->points[nb_idx].normal_x,
                normals->points[nb_idx].normal_y,
                normals->points[nb_idx].normal_z
            );
            
            consistency += std::abs(ref_normal.dot(nb_normal));
            valid_nbs++;
        }
        
        if (valid_nbs > 0) {
            normal_consistency[i] = consistency / valid_nbs;
        }
    }

    std::vector<size_t> sorted_indices(num_points);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0); // fill with (0 ... num_points-1)
    std::sort(sorted_indices.begin(), sorted_indices.end(), // sort indices based on normal consistency
            [&normal_consistency](size_t a, size_t b) {
                return normal_consistency[a] > normal_consistency[b];
            });
    
    for (size_t idx : sorted_indices) {
        if (labels_[idx] != -1) {
            continue;
        }

        // select seed from consistent area
        if (normal_consistency[idx] < 0.8f) continue;
    
        std::vector<int> region;
        std::queue<int> seeds;

        labels_[idx] = current_label;
        region.push_back(idx);
        seeds.push(idx);

        Eigen::Vector3f seed_normal(
            normals->points[idx].normal_x,
            normals->points[idx].normal_y,
            normals->points[idx].normal_z
        );

        while (!seeds.empty() && region.size() < static_cast<size_t>(config_.min_points_per_surfel)) {
            int current_seed = seeds.front();
            seeds.pop();

            std::vector<int> nbs;
            std::vector<float> d2s;
            kdtree_->nearestKSearch(cloud->points[current_seed], config_.knn_neighbors, nbs, d2s);

            for (int nb_idx : nbs) {
                if (labels_[nb_idx] != -1) continue;

                if (d2s.size() > 0) {
                    float dist = std::sqrt(d2s[std::distance(
                        nbs.begin(),
                        std::find(nbs.begin(), nbs.end(), nb_idx)
                    )]);

                    if (dist > config_.spatial_distance_threshold) continue;
                }

                Eigen::Vector3f nb_normal(
                    normals->points[nb_idx].normal_x,
                    normals->points[nb_idx].normal_y,
                    normals->points[nb_idx].normal_z
                );

                if (!normals_are_similar(seed_normal, nb_normal)) continue;

                labels_[nb_idx] = current_label;
                region.push_back(nb_idx);
                seeds.push(nb_idx);
            }
        }

        // If segmented into small region -> try again (NOTE: may slow down performance)
        if (region.size() >= static_cast<size_t>(config_.min_region_size)) {
            patches.push_back(region);
            current_label++;
        }
        else {
            for (int pidx : region) {
                labels_[pidx] = -1;
            }
        }
    }

    return patches;
}

Surfel2D SurfelExtractor::fit_surfel_to_patch(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::vector<int>& indices) {
    /**
     * 2D Gaussian fitting in tangent plane 
     */

    Surfel2D surfel;
    surfel.num_points = indices.size();

    if (indices.empty()) return surfel;

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal_sum = Eigen::Vector3f::Zero();

    for (int idx : indices) {
        const auto& pt = cloud->points[idx];
        centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);

        const auto& n = normals->points[idx];
        normal_sum += Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z);
    }

    centroid /= static_cast<float>(indices.size());
    normal_sum.normalize();

    surfel.center = centroid;
    surfel.normal = normal_sum;

    // Create orthonormal tangent basis
    Eigen::Vector3f arbitrary = (std::abs(normal_sum.z()) < 0.9f) ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitX();
    surfel.u_axis = normal_sum.cross(arbitrary).normalized();
    surfel.v_axis = normal_sum.cross(surfel.u_axis).normalized();

    // Project points in segment onto tangent basis
    Eigen::Matrix2f cov_2d = Eigen::Matrix2f::Zero();
    std::vector<Eigen::Vector2f> points_2d;
    points_2d.reserve(indices.size());
    for (int idx : indices) {
        const auto& pt = cloud->points[idx];
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f p_centered = p - centroid;
        Eigen::Vector2f p_2d(p_centered.dot(surfel.u_axis), p_centered.dot(surfel.v_axis));

        points_2d.push_back(p_2d);
    }

    // 2D covariance for ellipse parameters
    Eigen::Vector2f mean_2d = Eigen::Vector2f::Zero();
    for (const auto& p : points_2d) {
        mean_2d += p;
    }
    mean_2d /= static_cast<float>(points_2d.size());

    for (const auto& p : points_2d) {
        Eigen::Vector2f p_centered = p - mean_2d;
        cov_2d += p_centered * p_centered.transpose();
    }

    cov_2d /= static_cast<float>(points_2d.size() - 1);

    surfel.covariance = cov_2d;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigen_solver(cov_2d);
    surfel.eigen_values = eigen_solver.eigenvalues();
    surfel.eigen_vectors = eigen_solver.eigenvectors();
    surfel.eigen_values = surfel.eigen_values.cwiseMax(1e-6f); // ensure positive

    surfel.radius = 3.0f * std::sqrt(surfel.eigen_values.maxCoeff()); // 3 * sqrt(largest eval) - 3 sigma bound

    // Compute rms error (point-to-plane)
    float rms_error = compute_rms_error(cloud, indices, centroid, normal_sum);
    float rms_val = std::exp(-rms_error / 0.1f);
    // surfel.confidence = rms_val;

    // Planarity based on 3d covariance 
    Eigen::Matrix3f cov_3d = Eigen::Matrix3f::Zero();
    for (int idx : indices) {
        const auto& pt = cloud->points[idx];
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f p_centered = p - centroid;
        cov_3d += p_centered * p_centered.transpose();
    }

    cov_3d /= static_cast<float>(indices.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver_3d(cov_3d);
    Eigen::Vector3f eigenvalues_3d = eigen_solver_3d.eigenvalues();
    float planarity = compute_planarity(eigenvalues_3d);
    // surfel.confidence = compute_planarity(eigenvalues_3d);

    surfel.confidence = 0.5 * rms_val + 0.5 * planarity;
    return surfel;
}

bool SurfelExtractor::is_valid_surfel(const Surfel2D& surfel) const {
    if (surfel.num_points < config_.min_points_per_surfel) return false;
    if (surfel.confidence < config_.min_confidence) return false;
    if (surfel.getAspectRatio() > config_.max_aspect_ratio) return false;
    if (!surfel.center.allFinite() || !surfel.normal.allFinite()) return false;
    if (std::abs(surfel.normal.norm() - 1.0f) > 0.01f) return false;
    return true;
}

bool SurfelExtractor::normals_are_similar(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2) const {
    float dot_product = n1.dot(n2);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
    float angle_rad = std::acos(dot_product);
    float angle_deg = angle_rad * 180.0f / M_PI;
    return angle_deg <= config_.normal_angle_threshold_deg;
}

float SurfelExtractor::compute_planarity(const Eigen::Vector3f& eigenvalues) const {
    float lambda0 = eigenvalues(0); // smallest
    float lambda1 = eigenvalues(1); // mid
    float lambda2 = eigenvalues(2); // largest

    if (lambda2 < 1e-8f) return 0.0;

    float planarity = (lambda1 - lambda0) / lambda2;

    return std::max(0.0f, std::min(1.0f, planarity));
}

float SurfelExtractor::compute_rms_error(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices, const Eigen::Vector3f& center, const Eigen::Vector3f& normal) const {
    if (indices.empty()) return std::numeric_limits<float>::max();

    float sum_sq_dist = 0.0f;
    for (int idx : indices) {
        const auto& pt = cloud->points[idx];
        Eigen::Vector3f p(pt.x, pt.y, pt.z);

        // signed distance to plane
        float distance = normal.dot(p - center);
        sum_sq_dist += distance * distance;
    }

    float rms = std::sqrt(sum_sq_dist / indices.size());
    return rms;
}


};