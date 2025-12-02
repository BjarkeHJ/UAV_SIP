#ifndef LIDAR_PERCEPTION_NODE_
#define LIDAR_PERCEPTION_NODE_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2/transform_datatypes.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>

class LidarPerceptionNode : public rclcpp::Node {
public:
    LidarPerceptionNode();

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void pointcloud_preprocess();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>  tf_listener_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

    int cloud_count_; 
    std::string cloud_in_topic_;
    std::string global_frame_;
    std::string drone_frame_;
    std::string lidar_frame_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;
    geometry_msgs::msg::TransformStamped latest_tf_;

};

#endif