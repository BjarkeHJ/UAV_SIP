#ifndef PREPROCESS_HPP_
#define PREPROCESS_HPP_

#include "sparse_surfel_mapping/common/mapping_types.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/* 
* ScanPreprocess: Efficient pointcloud surface normal estimation via spherical projection and finite depth difference
* 
* Pipeline: 
* 1. Project 3D points to 2D spherical grid (yaw/pitch angles)
* 2. Downsample by selecting closest point in each block
* 3. Build range (depth) image
* 4. Smooth range image (Optional)
* 5. Estimate surfa normals from range image using finite differences (gradient approximation)
* 
*/

namespace sparse_surfel_map {

struct GridCell {
    Eigen::Vector3f point;
    float range_sq;
    bool valid;
};

class ScanPreprocess {
public:
    ScanPreprocess();
    explicit ScanPreprocess(const PreprocessConfig& config);

    void set_transform(const Eigen::Transform<float, 3, Eigen::Isometry>& tf);
    bool set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in);
    void process();
    void get_points_with_normal(std::vector<PointWithNormal>& pns);

private:
    inline size_t idx(size_t u, size_t v) const { return v * W_ + u; }
    void load_config();
    void grid_downsample();
    void build_range_image();
    void smooth_range_image();
    void estimate_normals();

    // Eigen::Transform<float, 3, Eigen::Isometry> tf_;
    Eigen::Vector3f gnd_normal_z_{Eigen::Vector3f::Zero()};
    float gnd_offset_z_{0.0f};
    bool have_tf_{false};

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_;
    std::vector<GridCell> grid_ds_;
    std::vector<float> range_img_;
    std::vector<float> range_img_smooth_;
    std::vector<PointWithNormal> points_with_normal_out_;

    size_t W_{0}, H_{0};
    size_t ds_{0};
    float yaw_min_, yaw_max_, pitch_min_, pitch_max_;
    float yaw_scale_, pitch_scale_;
    float min_range_sq_, max_range_sq_;

    PreprocessConfig config_;

};


} // namespace

#endif