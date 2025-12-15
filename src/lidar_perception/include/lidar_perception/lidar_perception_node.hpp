#ifndef LIDAR_PERCEPTION_NODE_
#define LIDAR_PERCEPTION_NODE_

#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2/transform_datatypes.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "lidar_perception/preprocess.hpp"

class LidarPerceptionNode : public rclcpp::Node {
public:
    LidarPerceptionNode();

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void pointcloud_preprocess();
    void publishNormals(pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_nrms, std::string &frame_id, double scale);

    void filtering(const sensor_msgs::msg::PointCloud2::SharedPtr msg);


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>  tf_listener_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nrm_pub_;

    float gnd_th{0.2};

    int cloud_count_; 
    std::string cloud_in_topic_;
    std::string global_frame_;
    std::string drone_frame_;
    std::string lidar_frame_;

    geometry_msgs::msg::TransformStamped latest_tf_;
    Eigen::Vector3f latest_pos_;
    Eigen::Quaternionf latest_q_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr latest_normals_;
    pcl::PointCloud<pcl::PointNormal>::Ptr latest_pts_w_nrms_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_buff_;
};

#endif