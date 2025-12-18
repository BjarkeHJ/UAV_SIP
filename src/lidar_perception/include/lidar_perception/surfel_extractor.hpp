#ifndef SURFEL_EXTRACTOR_HPP
#define SURFEL_EXTRACTOR_HPP

#include "surfel2d.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace surface_inspection_planner {

struct SurfelExtractionConfig {
    float normal_angle_threshold_deg = 15.0f;
    float spatial_distance_threshold = 0.05f;
    int min_points_per_surfel = 10;
    int max_points_per_surfel = 100;

    float min_planarity = 0.7f;
    float max_aspect_ratio = 5.0f;

    int knn_neighbors = 20;
    int min_region_size = 10;
};

class SurfelExtractor {
public:
    explicit SurfelExtractor(const SurfelExtractionConfig& config = SurfelExtractionConfig());

    std::vector<Surfel2D> extract_surfels(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals);

    void set_config(const SurfelExtractionConfig& config) { config_ = config; }

    const SurfelExtractionConfig& get_config() { return config_; }

    const std::vector<int>& get_segmentation_labels() const { return labels_; }

private:
    std::vector<std::vector<int>> segment_into_patches(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals);
    Surfel2D fit_surfel_to_patch(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals, const std::vector<int>& indices);

    bool is_valid_surfel(const Surfel2D& surfel) const;
    bool normals_are_similar(const Eigen::Vector3f& n1, const Eigen::Vector3f& n2) const;
    float compute_planarity(const Eigen::Vector3f& eigenvalues) const;
    void build_kdtree(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    SurfelExtractionConfig config_;
    std::vector<int> labels_;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_;
};


}; // end namespace


#endif