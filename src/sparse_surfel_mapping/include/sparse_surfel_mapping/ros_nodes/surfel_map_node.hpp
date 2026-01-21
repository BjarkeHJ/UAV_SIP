#ifndef SURFEL_MAP_NODE_
#define SURFEL_MAP_NODE_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "sparse_surfel_mapping/mapper/preprocess.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class SurfelMapNode : public rclcpp::Node {
public:
    explicit SurfelMapNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    // accessor for cross-node usage    
    SurfelMap* get_surfel_map() { return surfel_map_.get(); }
    const SurfelMap* get_surfel_map() const { return surfel_map_.get(); }

private:
    void declare_parameters();
    SurfelMapConfig load_configuration();

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publish_visualization();
    void publish_statistics();

    bool get_transform(const rclcpp::Time& stamp, Eigen::Transform<float, 3, Eigen::Isometry>& tf);

    // map
    std::unique_ptr<ScanPreprocess> preproc_;
    std::unique_ptr<SurfelMap> surfel_map_;

    // tf
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // pubs
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr surfel_marker_pub_;

    // subs
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    // timer
    rclcpp::TimerBase::SharedPtr viz_timer_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_;

    // params
    std::string global_frame_;
    std::string sensor_frame_;
    std::string pointcloud_topic_;
    double publish_rate_;
    bool publish_visualization_;
    double surfel_marker_scale_;
};

} // namespace

#endif