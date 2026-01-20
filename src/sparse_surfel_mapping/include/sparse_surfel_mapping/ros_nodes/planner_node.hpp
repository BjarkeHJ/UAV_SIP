#ifndef PLANNER_NODE_HPP_
#define PLANNER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "sparse_surfel_mapping/mapper/surfel_map.hpp"
#include "sparse_surfel_mapping/planner/inspection_planner.hpp"

#include "sip_interface/action/execute_path.hpp"

namespace sparse_surfel_map {

using ExecutePath = sip_interface::action::ExecutePath;
using GoalHandleExecutePath = rclcpp_action::ClientGoalHandle<ExecutePath>;

class InspectionPlannerNode : public rclcpp::Node {
public:
    InspectionPlannerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    void set_surfel_map(SurfelMap* map);

private:
    void declare_parameters();
    InspectionPlannerConfig load_configuration();

    void safety_timer_callback();
    void planner_timer_callback();

    // Action Interface
    void send_path_goal(); // Send current path
    void cancel_execution(); // cancel current execution
    void goal_response_callback(const GoalHandleExecutePath::SharedPtr& goal_handle);
    void feedback_callback(GoalHandleExecutePath::SharedPtr, const std::shared_ptr<const ExecutePath::Feedback> feedback);
    void result_callback(const GoalHandleExecutePath::WrappedResult& result);
    
    bool get_current_pose(Eigen::Vector3f& position, float& yaw);
    nav_msgs::msg::Path convert_to_nav_path(const RRTPath& planned_path); // internal path -> nav_msg
    bool should_send_new_path() const;
    
    void publish_emergency_stop();
    void publish_fov_pointcloud();

    void process_path_progress(uint32_t new_index);

    std::unique_ptr<InspectionPlanner> planner_;
    SurfelMap* map_{nullptr}; // raw pointer to map ("just observing")
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp_action::Client<ExecutePath>::SharedPtr trajectory_client_;
    GoalHandleExecutePath::SharedPtr current_goal_handle_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fov_cloud_pub_;
    
    rclcpp::TimerBase::SharedPtr plan_timer_;
    rclcpp::TimerBase::SharedPtr fov_timer_;
    rclcpp::TimerBase::SharedPtr safety_timer_;

    std::string global_frame_;
    std::string drone_frame_;
    std::string action_name_;

    double planner_rate_{0.0};
    double safety_rate_{10.0};

    // Trajectory executing params
    float trajectory_pos_tolerance_{0.1f};
    float trajectory_yaw_tolerance_{0.05f};
    float trajectory_v_max_{1.0f};
    float trajectory_a_max_{0.3f};

    // config
    InspectionPlannerConfig config_;

    // State flags
    bool is_active_{true};
    bool emergency_stop_active_{false};
    bool received_first_pose_{false};
    bool goal_in_progress_{false};
    bool waiting_for_goal_response_{false};

    // Drone State
    Eigen::Vector3f current_position_{Eigen::Vector3f::Zero()};
    float current_yaw_{0.0f};

    RRTPath current_planned_path_;
    int last_completed_viewpoint_idx_{-1};

    uint32_t trajectory_active_index_{0};
    float trajectory_progress_{0.0f};
    std::string trajectory_state_{"IDLE"};

    uint64_t current_plan_id_{0};
    size_t last_sent_path_size_{0};
    uint64_t last_sent_first_vp_id_{0};
};

} // namespace

#endif