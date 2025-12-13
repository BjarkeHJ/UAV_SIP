#include "lidar_perception/trajectory_node.hpp"

TrajectoryNode::TrajectoryNode() : Node("trajectory_node") {
    timer_tick_ms_ = this->declare_parameter<int>("timer_tick_ms", 50);
    path_topic_ = this->declare_parameter<std::string>("path_topic", "/waypoints_path");
    target_topic_ = this->declare_parameter<std::string>("target_topic", "/gz_px4_sim/target_pose");

    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(path_topic_, 5, std::bind(&TrajectoryNode::path_callback, this, std::placeholders::_1));
    target_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(target_topic_, 5);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(timer_tick_ms_), std::bind(&TrajectoryNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "Trajectory node started...");
}

void TrajectoryNode::path_callback(const nav_msgs::msg::Path::SharedPtr path_msg) {
    last_path_time_ = this->get_clock()->now();

    std::cout << "MESSAGE" << std::endl;

    if (path_msg->poses.size() < 1) {
        RCLCPP_WARN(this->get_logger(), "Received Empty Path - Nothing to do!");
        have_path_ = false;
        path_.clear();
        return;
    }

    path_.clear();
    path_.push_back(Waypoint{pos_ref_, yaw_ref_});

    // Incoming waypoints
    for (const auto& ps : path_msg->poses) {
        Eigen::Vector3f p(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
        float yaw = quat_to_yaw(ps.pose.orientation);
        path_.push_back({p, yaw});
    }

    std::cout << path_.size() << std::endl;
    have_path_ = (path_.size() >= 2 );
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

    const float dt_s = static_cast<float>(timer_tick_ms_) / 1000.0f;
    update_reference(dt_s);

    const auto& wp = active_path_.poses[active_index_];
    current_ref_ = wp;

    auto feedback = std::make_shared<ExecutePath::Feedback>();
    feedback->active_index = static_cast<uint32_t>(active_index_);
    feedback->remaining_distance = compute_remaining_distance();
    feedback->progress = compute_progress();
    feedback->state = "EXECUTING";
    feedback->plan_id = path_id_;
    active_goal_->publish_feedback(feedback);
    
    if (active_index_ + 1 < active_path_.poses.size()) {
        active_index_++;
        return;
    }

    auto result = std::make_shared<ExecutePath::Result>();
    result->success = true;
    result->message = "Reached end of path";
    result->plan_id = path_id_;
    active_goal_->succeed(result);
    active_goal_.reset();

    geometry_msgs::msg::PoseStamped target_msg;
    target_msg.header.frame_id = "odom";
    target_msg.header.stamp = this->get_clock()->now();
    target_msg.pose.position.x = pos_ref_.x();
    target_msg.pose.position.y = pos_ref_.y();
    target_msg.pose.position.z = pos_ref_.z();
    target_msg.pose.orientation = yaw_to_quat(yaw_ref_);
    target_pub_->publish(target_msg);
}

void TrajectoryNode::update_reference(float dt) {
    // if (!have_path_ || (this->get_clock()->now() - last_plan_time_).seconds() > stale_timeout_) {
    if (!have_path_) {
        // brake to stop smoothly
        Eigen::Vector3f dv = -vel_ref_;
        float dv_norm = dv.norm();
        float dv_max = acc_max_ * dt;
        if (dv_norm > dv_max) {
            dv *= (dv_max / dv_norm);
        }
        vel_ref_ += dv;
        pos_ref_ += vel_ref_ * dt;
        return;
    }

    while (path_.size() >= 2) {
        Eigen::Vector3f to_wp = path_[1].pos - pos_ref_;
        if (to_wp.norm() < pos_tol_) {
            path_.pop_front(); // advance waypoint
        }
        else {
            break;
        }
    }

    if (path_.size() < 2) {
        RCLCPP_WARN(this->get_logger(), "[Trajectory Generator] Less than 2 waypoints given... Cannot generate trajectory!");
        have_path_ = false;
        return;
    }

    Eigen::Vector3f target = path_[1].pos;
    Eigen::Vector3f d = target - pos_ref_;
    float dist = d.norm();
    Eigen::Vector3f dir = Eigen::Vector3f::Zero();
    if (dist > 1e-6f) dir = d / dist;
    
    float step = std::min(lookahead_, dist);
    Eigen::Vector3f carrot = pos_ref_ + dir * step;
    
    Eigen::Vector3f vel_des = (carrot - pos_ref_);
    float vel_des_norm = vel_des.norm();
    if (vel_des_norm > 1e-6f) {
        vel_des = vel_des / vel_des_norm * std::min(vel_max_, vel_des_norm / dt); // velocity-ish
    }
    else {
        vel_des.setZero();
    }

    // limit acceleration
    Eigen::Vector3f dv = vel_des - vel_ref_;
    float dv_norm = dv.norm();
    float dv_max = acc_max_ * dt;
    if (dv_norm > dv_max) {
        dv *= (dv_max / dv_norm);
    }
    vel_ref_ += dv;
    pos_ref_ += vel_ref_ * dt; // integrate pos;
    
    // limit yaw rate
    float yaw_des = yaw_ref_;
    if (vel_ref_.head<2>().norm() > 0.2f) {
        yaw_des = std::atan2(vel_ref_.y(), vel_ref_.x());
    }

    float dy = wrap_pi(yaw_des - yaw_ref_);
    float max_dy = yaw_rate_max_ * dt;
    dy = std::clamp(dy, -max_dy, +max_dy);
    yaw_ref_ = wrap_pi(yaw_ref_ + dy);
}

rclcpp_action::GoalResponse TrajectoryNode::handle_goal(const rclcpp_action::GoalUUID&, std::shared_ptr<const ExecutePath::Goal> goal) {
    if (goal->path.poses.empty()) {
    RCLCPP_WARN(this->get_logger(), "Rejected goal: empty path.");
    return rclcpp_action::GoalResponse::REJECT;
    }
    RCLCPP_INFO(this->get_logger(), "Accepted goal request plan_id=%lu with %zu poses",
                (unsigned long)goal->plan_id, goal->path.poses.size());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse TrajectoryNode::handle_cancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePath>> goal_handle) {
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

    active_goal_ = goal_handle;
    auto goal = goal_handle->get_goal();

    active_path_ = goal->path;
    active_index_ = 0;
    pos_tol_ = goal->pos_tolerance;
    yaw_tol_ = goal->yaw_tolerance;
    vel_max_ = goal->v_max;
    acc_max_ = goal->a_max;
    path_id_ = goal->plan_id;

    // Anchor continuity trick: start execution from current_ref_ instead of first waypoint
    // Minimal version: just overwrite the first pose with current_ref_ position.
    // Better: insert current_ref_ at front (or anchor your spline/queue).
    if (!active_path_.poses.empty()) {
      active_path_.poses.front().pose.position = current_ref_.pose.position;
      active_path_.poses.front().pose.orientation = current_ref_.pose.orientation;
    }

    RCLCPP_INFO(get_logger(), "Started executing plan_id=%lu", (unsigned long)path_id_);
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
    while (a < -M_1_PI) a += 2.f * M_PI;
    return a;
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajectoryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}