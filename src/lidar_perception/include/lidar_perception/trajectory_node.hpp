#ifndef TRAJECTORY_NODE_HPP_
#define TRAJECTORY_NODE_HPP_

#include <chrono>
#include <deque>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/path.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sip_interface/action/execute_path.hpp"
using ExecutePath = sip_interface::action::ExecutePath;

struct Waypoint {
    Eigen::Vector3f pos;
    float yaw;
};

class TrajectoryNode : public rclcpp::Node
{
public:
    TrajectoryNode();
private:
    rclcpp_action::Server<ExecutePath>::SharedPtr planner_action_server_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr vis_path_pub_;

    std::string target_topic_;
    int timer_tick_ms_;
    
    void timer_callback();
    void update_reference(float dt);

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID&, std::shared_ptr<const ExecutePath::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePath>> goal_handle);
    void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePath>> goal_handle);

    float yaw_step_towards(float yaw_goal, float dist_to_wp, float dt);

    geometry_msgs::msg::Quaternion yaw_to_quat(float yaw);
    float quat_to_yaw(geometry_msgs::msg::Quaternion q);
    float wrap_pi(float a);
    float compute_remaining_distance();
    float compute_progress();

    void publish_target();
    void publish_path_vis();

    std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePath>> active_goal_;
    nav_msgs::msg::Path active_path_;
    size_t active_index_{0};
    geometry_msgs::msg::PoseStamped current_ref_;
    uint64_t path_id_{0};

    bool have_path_{false};
    std::deque<Waypoint> path_;
    rclcpp::Time last_path_time_;

    Eigen::Vector3f pos_ref_{0.0f, 0.0f, 0.5f}; // hover 50cm above ground
    Eigen::Vector3f vel_ref_{0.0f, 0.0f, 0.0f};
    float yaw_ref_{0.0f};

    float vel_max_{2.0f}; // max velocity
    float acc_max_{2.0f}; // max acceleration
    float lookahead_{1.0f};
    float pos_tol_{0.3f};
    float yaw_tol_{5.0f}; // degree
    float yaw_rate_max_{0.5f}; // rad/s
    float stale_timeout_{0.5f};

    float yaw_dist_scale_{1.0f};
    float yaw_min_factor_{0.2};
};

#endif