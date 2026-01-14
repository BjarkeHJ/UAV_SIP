#include "sparse_surfel_mapping/ros_nodes/planner_node.hpp"

namespace sparse_surfel_map {

InspectionPlannerNode::InspectionPlannerNode(const rclcpp::NodeOptions& options) : Node("inspection_planner_node", options) {

    declare_parameters();
    // load config and create planner object
    InspectionPlannerConfig config = load_configuration();
    planner_ = std::make_unique<InspectionPlanner>(config);
    
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("inspection_planner/path", 10);

    if (planner_rate_ > 0.0) {
        plan_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / planner_rate_),
            std::bind(&InspectionPlannerNode::planner_timer_callback, this)
        );
    }

    RCLCPP_INFO(this->get_logger(), "Inspection planner node initialized");
    RCLCPP_INFO(this->get_logger(), "  Planning rate: %.2f Hz", planner_rate_); 

}

void InspectionPlannerNode::set_surfel_map(SurfelMap* map) {
    map_ = map;
    planner_->initialize(map);
    RCLCPP_INFO(this->get_logger(), "SurfelMap connected to planner");
}

void InspectionPlannerNode::declare_parameters() {
    // Frames
    this->declare_parameter("global_frame", "odom");
    this->declare_parameter("drone_frame", "base_link");
    
    // Rates
    this->declare_parameter("planner_rate", 1.0);
    
    // Target threshold
    this->declare_parameter("target_reach_th", 0.5);
    this->declare_parameter("target_yaw_th", 0.3);

    // Camera
    this->declare_parameter("camera.hfov_deg", 90.0);
    this->declare_parameter("camera.vfov_deg", 60.0);
    this->declare_parameter("camera.min_range",0.5);
    this->declare_parameter("camera.max_range", 15.0);

    // Viewpoint

    
    // Collision safety
    this->declare_parameter("collision.robot_radius", 0.5);
    this->declare_parameter("collision.safety_distance", 0.5);
    this->declare_parameter("collision.path_resolution", 0.1);

    // Path Planning
    this->declare_parameter("path.max_iterations", 100);
    this->declare_parameter("path.heuristic_weigt", 1.2);
    this->declare_parameter("path.enable_smoothing", true);

    // Inspection
    this->declare_parameter("inspection.max_viewpoints_per_plan", 5);

    // Load immediate parameters
    global_frame_ = this->get_parameter("global_frame").as_string();
    drone_frame_ = this->get_parameter("drone_frame").as_string();
    planner_rate_ = this->get_parameter("planner_rate").as_double();
    target_reach_th_ = this->get_parameter("target_reach_th").as_double();
    target_yaw_th_ = this->get_parameter("target_yaw_th").as_double();
}

InspectionPlannerConfig InspectionPlannerNode::load_configuration() {
    InspectionPlannerConfig config;

    return config;
}

void InspectionPlannerNode::planner_timer_callback() {
    if (!is_active_ || !map_) return;

    if (!get_current_pose(current_position_, current_yaw_)) {
        if (!received_first_pose_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Waiting for drone pose...");
            return;
        }
    }
    received_first_pose_ = true;

    // Lock map whilst reading map (shared_lock blocks only other writers)
    std::shared_lock lock(map_->mutex_);

    planner_->update_state(current_position_, current_yaw_);

    if (has_reached_target()) {
        RCLCPP_INFO(this->get_logger(), "Target Reached!");
        planner_->mark_target_reached();
    }

    if (planner_->is_inspection_complete()) {
        RCLCPP_INFO(this->get_logger(), "Inspection complete! Coverage: %.1f%%", planner_->statistics().coverage_ratio * 100.0f);
        is_active_ = false;
        lock.unlock();
        return;
    }

    if (planner_->needs_replan()) {
        RCLCPP_INFO(this->get_logger(), "Replanning...");

        if (planner_->plan()) {
            publish_path();    
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Planning failed");
        }
    }
    else {
        RCLCPP_INFO(this->get_logger(), "Executing Path...");
    }
    
    lock.unlock(); // unlock map
}

bool InspectionPlannerNode::get_current_pose(Eigen::Vector3f& position, float& yaw) {
    try {
        auto transform = tf_buffer_->lookupTransform(global_frame_, drone_frame_, tf2::TimePointZero);
        position.x() = transform.transform.translation.x;
        position.y() = transform.transform.translation.y;
        position.z() = transform.transform.translation.z;

        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        );

        double roll, pitch, yaw_d;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw_d);
        yaw = static_cast<float>(yaw_d);
        return true;
    }
    catch (const tf2::TransformException& ex) {
        return false;
    }
}

bool InspectionPlannerNode::has_reached_target() {
    const Viewpoint& target = planner_->get_next_target();

    if (target.num_visible() == 0 && target.id() == 0) {
        return false;
    }

    float pos_error = (current_position_ - target.position()).norm();
    if (pos_error > target_reach_th_) return false;

    float yaw_error = std::abs(current_yaw_ - target.yaw());
    if (yaw_error > M_PI) {
        yaw_error = 2.0f * M_PI - yaw_error;
    }
    if (yaw_error > target_yaw_th_) return false;

    return true;
}

void InspectionPlannerNode::publish_path() {
    const auto& path = planner_->get_current_path();
    std::cout << "[PublishPath] Path Lenght: " << path.size() << std::endl;
    if (!path.is_valid) return;

    nav_msgs::msg::Path msg;
    msg.header.frame_id = global_frame_;
    msg.header.stamp = this->get_clock()->now();

    for (size_t i = 0; i < path.waypoints.size(); ++i) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = msg.header;
        pose.pose.position.x = path.waypoints[i].x();
        pose.pose.position.y = path.waypoints[i].y();
        pose.pose.position.z = path.waypoints[i].z();

        if (i < path.yaw_angles.size()) {
            tf2::Quaternion q;
            q.setRPY(0, 0, path.yaw_angles[i]);
            pose.pose.orientation = tf2::toMsg(q);
        }

        msg.poses.push_back(pose);
    }

    path_pub_->publish(msg);
}


} // namespace