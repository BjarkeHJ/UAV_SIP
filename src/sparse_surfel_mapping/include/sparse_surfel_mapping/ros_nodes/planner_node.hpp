#ifndef PLANNER_NODE_HPP_
#define PLANNER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/inspection_planner.hpp"

namespace sparse_surfel_map {

class InspectionPlannerNode : public rclcpp::Node {
public:
    InspectionPlannerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    void set_surfel_map(SurfelMap* map);

private:
    void declare_parameters();
    InspectionPlannerConfig load_configuration();

    void planner_timer_callback();

    bool get_current_pose(Eigen::Vector3f& position, float& yaw);
    bool has_reached_target();

    void publish_path();
    void publish_statistics();

    void publish_fov_pointcloud();

    std::unique_ptr<InspectionPlanner> planner_;
    SurfelMap* map_{nullptr}; // raw pointer to map ("just observing")
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fov_cloud_pub_;
    rclcpp::TimerBase::SharedPtr fov_timer_;

    std::string global_frame_;
    std::string drone_frame_;
    double planner_rate_{0.0};

    // config
    InspectionPlannerConfig config_;

    // tracking
    float target_reach_th_{0.5f};
    float target_yaw_th_{0.3f};

    // bool is_active_{false};
    bool is_active_{true};
    bool received_first_pose_{false};
    Eigen::Vector3f current_position_{Eigen::Vector3f::Zero()};
    float current_yaw_{0.0f};

    bool target_published_{false};
    size_t last_target_index_{0};
};

} // namespace

#endif