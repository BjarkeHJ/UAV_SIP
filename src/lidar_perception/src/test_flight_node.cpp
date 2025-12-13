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
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/waypoints_path", 5);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(timer_tick_ms_), std::bind(&TestFlight::send_once, this));
        
        RCLCPP_INFO(this->get_logger(), "Begining Test Flight!");
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp_action::Client<ExecutePath>::SharedPtr client_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    void send_once();
    int timer_tick_ms_{50};
    bool sent_{false};

    int plan_id{0};
};

void TestFlight::send_once() {
    if (sent_) return;
    sent_ = true;
    timer_->cancel();

    if (!client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "Action server not available.");
      return;
    }

    auto goal = ExecutePath::Goal();
    goal.plan_id = 1;
    goal.pos_tolerance = 0.5f;
    goal.yaw_tolerance = 0.5f;
    goal.v_max = 2.0f;
    goal.a_max = 2.0f;

    nav_msgs::msg::Path path;
    path.header.frame_id = "odom";

    auto make_pose = [&](double x, double y, double z) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "map";
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.position.z = z;
      ps.pose.orientation.w = 1.0; // yaw ignored in this demo
      return ps;
    };

    path.poses.push_back(make_pose(0, 0, 5));
    path.poses.push_back(make_pose(5, 0, 5));
    path.poses.push_back(make_pose(5, 5, 5));
    path.poses.push_back(make_pose(0, 5, 5));

    goal.path = path;

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

    client_->async_send_goal(goal, opts);
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TestFlight>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}