#ifndef PLANNER_NODE_HPP_
#define PLANNER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class InspectionPlannerNode : public rclcpp::Node {
public:
    InspectionPlannerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    void set_surfel_map(SurfelMap* map) { map_ = map; }

private:
    void declare_parameters();
    
    void planner_timer_callback();
    void publish_path();
    void publish_statistics();

    SurfelMap* map_; // raw pointer to map ("just observing")
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    std::string global_frame_;
    std::string drone_frame_;
    double planner_rate_{0.0};

    // tracking
    float target_reach_th_{0.0f};
    float target_yaw_th_{0.0f};

    bool target_published_{false};
    size_t last_target_index_{0};
    
};

} // namespace

#endif