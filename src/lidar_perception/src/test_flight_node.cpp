/* 
Test Flight Node: 
A test setup for emulating the high-level planner action
client. It provides a set of waypoints to the trajectory node whilst listening
to the feedback and updates accordingly.

A set of waypoints is send as an action to the path executor (trajectory node)
*/

#include <chrono>
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

class TestFlight : public rclcpp::Node {
public:
    TestFlight() : Node("test_flight_node") {
      planner_action_client_ = rclcpp_action::create_client<ExecutePath>(this, "execute_path");
      timer_ = this->create_wall_timer(std::chrono::milliseconds(timer_tick_ms_), std::bind(&TestFlight::send_once, this));
      RCLCPP_INFO(this->get_logger(), "Begining Test Flight!");
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp_action::Client<ExecutePath>::SharedPtr planner_action_client_;
    int timer_tick_ms_{50};
    bool sent_{false};
    void send_once(); // test send goal

    geometry_msgs::msg::Quaternion yaw_to_quat(float yaw) {
      geometry_msgs::msg::Quaternion q;
      q.w = std::cos(yaw * 0.5f);
      q.x = 0.0f;
      q.y = 0.0f;
      q.z = std::sin(yaw * 0.5f);
      return q;
    }
    
};

void TestFlight::send_once() {
    if (sent_) return;
    sent_ = true;
    timer_->cancel();

    if (!planner_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "Action server not available.");
      return;
    }

    // Create Action Goal
    auto goal = ExecutePath::Goal();
    goal.plan_id = 1;
    goal.pos_tolerance = 0.1f;
    goal.yaw_tolerance = 0.08f;
    goal.v_max = 1.0f;
    goal.a_max = 0.5f;

    nav_msgs::msg::Path path;
    path.header.frame_id = "odom";

    auto make_pose = [&](double x, double y, double z, float yaw=0.0f) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "odom";
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = z;
      ps.pose.orientation = yaw_to_quat(yaw); // yaw ignored in this demo
      return ps;
    };

    // Hardcoded square in warehouse world
    path.poses.push_back(make_pose(0, 0, 5, 0.0f));
    path.poses.push_back(make_pose(0, 5, 5, -M_PI_4));
    path.poses.push_back(make_pose(5, 5, 5, -M_PI_2));
    path.poses.push_back(make_pose(10, 5, 5, -M_PI_2));
    path.poses.push_back(make_pose(20, 5, 5, -M_PI_2));
    path.poses.push_back(make_pose(30, 5, 5, -M_PI_2 - M_PI_4));
    path.poses.push_back(make_pose(30, -5, 5, M_PI - M_PI_4));
    path.poses.push_back(make_pose(20, -5, 5, M_PI_2));
    path.poses.push_back(make_pose(10, -5, 5, M_PI_2));
    path.poses.push_back(make_pose(5, -5, 5, M_PI_2));
    path.poses.push_back(make_pose(0, -5, 5, M_PI_4));
    path.poses.push_back(make_pose(0, 0, 2, 0.0f));
    goal.path = path;

    // Define feedback callback
    rclcpp_action::Client<ExecutePath>::SendGoalOptions opts;
    opts.feedback_callback =
      [this](auto, const std::shared_ptr<const ExecutePath::Feedback> fb) {
        RCLCPP_INFO(this->get_logger(),
          "FB plan_id=%lu idx=%u rem=%.2f prog=%.2f state=%s",
          (unsigned long)fb->plan_id,
          fb->active_index,
          fb->remaining_distance,
          fb->progress,
          fb->state.c_str());
      };

    // Define result callback 
    opts.result_callback =
      [this](const auto& res) {
        auto code = res.code;
        const auto& r = res.result;
        RCLCPP_INFO(this->get_logger(),
          "RESULT code=%d success=%d plan_id=%lu msg=%s",
          (int)code, r->success, (unsigned long)r->plan_id, r->message.c_str());
      };

    RCLCPP_INFO(get_logger(), "Sending ExecutePath goal plan_id=%lu (%zu poses)",
            (unsigned long)goal.plan_id, goal.path.poses.size());

    planner_action_client_->async_send_goal(goal, opts);
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TestFlight>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}