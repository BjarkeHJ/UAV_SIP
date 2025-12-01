#ifndef LIDAR_PERCEPTION_NODE_
#define LIDAR_PERCEPTION_NODE_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

class LidarPerceptionNode : public rclcpp::Node {
public:
    LidarPerceptionNode();

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>  tf_listener_;

    int cloud_count_; 
    std::string cloud_in_topic_;
    std::string global_frame_;
    std::string drone_frame_;
    std::string lidar_frame_;
};

#endif