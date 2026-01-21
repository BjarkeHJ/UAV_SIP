/*
Trajectory Node 
Key feature: Takes waypoints (position + orientation) from high
level planner and computes a feasible trajectory for the UAV to track. Sends
command to PX4 Offboard bridge node that handles all direct interface with the
flight stack.

This node is a Action Server. 

*/

#include "lidar_perception/trajectory_node.hpp"

TrajectoryNode::TrajectoryNode() : Node("trajectory_node") {
    timer_tick_ms_ = this->declare_parameter<int>("timer_tick_ms", 50);
    target_topic_ = this->declare_parameter<std::string>("target_topic", "/gz_px4_sim/target_pose");

    planner_action_server_ = rclcpp_action::create_server<ExecutePath>(
        this,
        "execute_path",
        std::bind(&TrajectoryNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&TrajectoryNode::handle_cancel, this, std::placeholders::_1),
        std::bind(&TrajectoryNode::handle_accepted, this, std::placeholders::_1) 
    );

    timer_ = this->create_wall_timer(std::chrono::milliseconds(timer_tick_ms_), std::bind(&TrajectoryNode::timer_callback, this));
    target_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(target_topic_, 5);
    vis_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("path_vis_", 10);

    RCLCPP_INFO(this->get_logger(), "Trajectory node (Action Server) started...");
}

void TrajectoryNode::timer_callback() {
    if (!active_goal_) return;

    if (active_goal_->is_canceling()) {
        auto result = std::make_shared<ExecutePath::Result>();
        result->success = false;
        result->message = "Canceled";
        result->plan_id = path_id_;
        active_goal_->canceled(result);
        active_goal_.reset();
        return;
    }

    // Update current reference command to send to the PX4 Offboard controller
    const auto& wp = active_path_.poses[active_index_];
    current_ref_ = wp;

    const float dt_s = static_cast<float>(timer_tick_ms_) / 1000.0f;
    update_reference(dt_s);
    publish_path_vis();
    publish_target();
    
    // Provide feedback (Topic stream) to high level planner
    auto feedback = std::make_shared<ExecutePath::Feedback>();
    feedback->active_index = static_cast<uint32_t>(active_index_);
    feedback->remaining_distance = compute_remaining_distance();
    feedback->progress = compute_progress();
    feedback->state = "EXECUTING";
    feedback->plan_id = path_id_;
    active_goal_->publish_feedback(feedback);
    
    // When finished return result to high level planner 
    const size_t N = active_path_.poses.size();
    const auto &last = active_path_.poses[N - 1].pose.position;
    Eigen::Vector3f last_p(last.x, last.y, last.z);

    bool pos_ok = (last_p - pos_ref_).norm() < pos_tol_;
    bool vel_ok = vel_ref_.norm() < 0.2f;

    if (active_index_ >= (N - 1) && pos_ok && vel_ok) {
        auto result = std::make_shared<ExecutePath::Result>();
        result->success = true;
        result->message = "Reached end of path";
        result->plan_id = path_id_;
        active_goal_->succeed(result);
        active_goal_.reset();
    }
}

void TrajectoryNode::update_reference(float dt) {
    // No active goal -> halt
    if (!active_goal_ || active_path_.poses.empty()) {
        Eigen::Vector3f dv = -vel_ref_;
        float dv_norm = dv.norm();
        float dv_max = acc_max_ * dt;
        if (dv_norm > dv_max && dv_norm > 1e-6f) {
            dv *= (dv_max / dv_norm);
        }
        vel_ref_ += dv;
        pos_ref_ += vel_ref_ * dt;
        return;
    }

    const size_t N = active_path_.poses.size();
    if (N == 1) {
        // Only one waypoint: just go/hold there
        const auto &p0 = active_path_.poses[0].pose.position;
        Eigen::Vector3f goal_p(p0.x, p0.y, p0.z);

        Eigen::Vector3f err = goal_p - pos_ref_; // Vector from reference to next wp
        if (err.norm() < pos_tol_) {
            // snap + stop
            pos_ref_ = goal_p;
            vel_ref_.setZero();
        } else {
            // simple chase
            Eigen::Vector3f dir = err.normalized();
            Eigen::Vector3f vel_des = dir * vel_max_;
            
            Eigen::Vector3f dv = vel_des - vel_ref_;
            float dv_norm = dv.norm();
            float dv_max = acc_max_ * dt;
            if (dv_norm > dv_max && dv_norm > 1e-6f) dv *= (dv_max / dv_norm);
            vel_ref_ += dv;
            pos_ref_ += vel_ref_ * dt;
        }
        return;
    }

    // active_index_ means: "last reached waypoint index" (0..N-1)
    active_index_ = std::min(active_index_, N - 1);

    // Advance index while we're within tolerance of the NEXT waypoint
    while ((active_index_ + 1) < N) {
        const auto &next = active_path_.poses[active_index_ + 1].pose.position;
        Eigen::Vector3f next_p(next.x, next.y, next.z);
        if ((next_p - pos_ref_).norm() < pos_tol_) {
            active_index_++;
        } else {
            break;
        }
    }

    // If we've reached the last waypoint, brake to stop and hold it
    if (active_index_ >= (N - 1)) {
        const auto &last = active_path_.poses[N - 1].pose.position;
        Eigen::Vector3f last_p(last.x, last.y, last.z);

        // Hold position at the last waypoint (optional snap when close)
        Eigen::Vector3f err = last_p - pos_ref_;
        if (err.norm() < pos_tol_) {
            pos_ref_ = last_p;
        }

        // Brake to zero velocity smoothly
        Eigen::Vector3f dv = -vel_ref_;
        float dv_norm = dv.norm();
        float dv_max = acc_max_ * dt;
        if (dv_norm > dv_max && dv_norm > 1e-6f) dv *= (dv_max / dv_norm);
        vel_ref_ += dv;
        pos_ref_ += vel_ref_ * dt;

        // Yaw: converge to final waypoint yaw (rate-limited)
        float yaw_goal = -quat_to_yaw(active_path_.poses[N - 1].pose.orientation);
        float dist_last = (last_p - pos_ref_).norm();
        yaw_ref_ = yaw_step_towards(yaw_goal, dist_last, dt);
        return;
    }
    
    const auto& tgt_pose = active_path_.poses[active_index_ + 1].pose;
    Eigen::Vector3f target(tgt_pose.position.x, tgt_pose.position.y, tgt_pose.position.z);
    float dist_to_target = (target - pos_ref_).norm();
    float current_speed = vel_ref_.norm();

    // Compute the current distance it will take to stop
    float stopping_dist = (current_speed * current_speed) / (2.0f *acc_max_);
    stopping_dist += pos_tol_ * 2.0f;

    float speed_limit = vel_max_;
    
    if (dist_to_target < stopping_dist) {
        speed_limit = std::sqrt(2.0f * acc_max_ * std::max(dist_to_target - pos_tol_, 0.01f));
        speed_limit = std::max(speed_limit, 0.1f);
    }

    Eigen::Vector3f dir = (target - pos_ref_).normalized();
    Eigen::Vector3f vel_des = dir * std::min(speed_limit, vel_max_);

    Eigen::Vector3f dv = vel_des - vel_ref_;
    float dv_norm = dv.norm();
    float dv_max = acc_max_ * dt;
    if (dv_norm > dv_max) dv *= (dv_max / dv_norm);

    vel_ref_ += dv;
    pos_ref_ += vel_ref_ * dt;

    // Yaw: ALWAYS follow waypoint yaw (shortest-angle), distance-weighted turning
    float yaw_goal = -quat_to_yaw(tgt_pose.orientation);
    yaw_ref_ = yaw_step_towards(yaw_goal, dist_to_target, dt);
}

rclcpp_action::GoalResponse TrajectoryNode::handle_goal(const rclcpp_action::GoalUUID&, std::shared_ptr<const ExecutePath::Goal> goal) {
    if (goal->path.poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Rejected goal: empty path.");
        return rclcpp_action::GoalResponse::REJECT;
    }

    RCLCPP_INFO(this->get_logger(), "Accepted goal request plan_id=%lu with %zu poses", (unsigned long)goal->plan_id, goal->path.poses.size());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse TrajectoryNode::handle_cancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePath>> goal_handle) {
    // Blindly accept cancellation 
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Cancel requested.");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void TrajectoryNode::handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePath>> goal_handle) {
    // “replace plan, don’t append”: if new goal arrives, abort current and switch
    if (active_goal_) {
      RCLCPP_WARN(get_logger(), "New goal accepted: preempting previous plan_id=%lu", (unsigned long)path_id_);
      auto result = std::make_shared<ExecutePath::Result>();
      result->success = false;
      result->message = "Preempted by a new goal";
      result->plan_id = path_id_;
      active_goal_->abort(result);
    }

    // Set acctepted goal
    active_goal_ = goal_handle;  
    auto goal = goal_handle->get_goal();

    // Copy goal content
    active_path_ = goal->path;
    path_id_ = goal->plan_id;
    active_index_ = 0;

    pos_tol_ = goal->pos_tolerance;
    yaw_tol_ = goal->yaw_tolerance;
    vel_max_ = goal->v_max;
    acc_max_ = goal->a_max;

    RCLCPP_INFO(get_logger(), "Started executing plan_id=%lu", (unsigned long)path_id_);
}

float TrajectoryNode::yaw_step_towards(float yaw_goal, float dist_to_wp, float dt) {
    float f = dist_to_wp / std::max(1e-3f, yaw_dist_scale_);
    f = std::clamp(f, 0.0f, 1.0f);
    f = yaw_min_factor_ + (1.0f - yaw_min_factor_) * f;

    float dy = wrap_pi(yaw_goal - yaw_ref_);
    float max_dy = (yaw_rate_max_ * f) * dt;
    dy = std::clamp(dy, -max_dy, +max_dy);
    return wrap_pi(yaw_ref_ + dy);
}

float TrajectoryNode::compute_remaining_distance() {
    // You would compute from current_ref_ along remaining waypoints or along spline length.
    // Simple placeholder:
    return static_cast<float>(active_path_.poses.size() - 1 - active_index_);
}

float TrajectoryNode::compute_progress() {
    if (active_path_.poses.size() <= 1) return 1.0f;
    return static_cast<float>(active_index_) / static_cast<float>(active_path_.poses.size() - 1);
}

geometry_msgs::msg::Quaternion TrajectoryNode::yaw_to_quat(float yaw) {
    geometry_msgs::msg::Quaternion q;
    q.w = std::cos(yaw * 0.5f);
    q.x = 0.0f;
    q.y = 0.0f;
    q.z = std::sin(yaw * 0.5f);
    return q;
}

float TrajectoryNode::quat_to_yaw(geometry_msgs::msg::Quaternion q) {
    Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
    Eigen::Matrix3f R = quat.toRotationMatrix();
    float yaw = std::atan2(R(1,0), R(0,0));
    return yaw;
}

float TrajectoryNode::wrap_pi(float a) {
    while (a > M_PI) a -= 2.f * M_PI;
    while (a < -M_PI) a += 2.f * M_PI;
    return a;
}

void TrajectoryNode::publish_target() {
    // Publish command
    geometry_msgs::msg::PoseStamped target_msg;
    target_msg.header.frame_id = "odom";
    target_msg.header.stamp = this->get_clock()->now();
    target_msg.pose.position.x = pos_ref_.x();
    target_msg.pose.position.y = pos_ref_.y();
    target_msg.pose.position.z = pos_ref_.z();
    target_msg.pose.orientation = yaw_to_quat(yaw_ref_);
    target_pub_->publish(target_msg);
}

void TrajectoryNode::publish_path_vis() {
    nav_msgs::msg::Path rem;
    rem.header.frame_id = active_path_.header.frame_id;
    rem.header.stamp = this->get_clock()->now();
    for (size_t i=active_index_+1; i<active_path_.poses.size(); ++i) {
        rem.poses.push_back(active_path_.poses[i]);
    }
    vis_path_pub_->publish(rem);
}


int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajectoryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}