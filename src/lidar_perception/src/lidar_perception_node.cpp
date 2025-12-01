#include "lidar_perception/lidar_perception_node.hpp"


LidarPerceptionNode::LidarPerceptionNode() : Node("LidarPerceptionNode") {
    cloud_in_topic_ = this->declare_parameter<std::string>("pointcloud_in", "/lidar_front/points_raw");
    global_frame_ = this->declare_parameter<std::string>("global_frame", "world");
    drone_frame_ = this->declare_parameter<std::string>("drone_frame", "base_link");
    lidar_frame_ = this->declare_parameter<std::string>("lidar_frame", "lidar_sensor_link");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    auto qos_profile = rclcpp::SensorDataQoS();
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_in_topic_,
        qos_profile,
        std::bind(&LidarPerceptionNode::pointcloud_callback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "LidarPerceptionNode Started...");
}

void LidarPerceptionNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud...");
        return;
    }

    

}



int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarPerceptionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}