#include "sparse_surfel_mapping/ros_nodes/planner_node.hpp"

namespace sparse_surfel_map {

InspectionPlannerNode::InspectionPlannerNode(const rclcpp::NodeOptions& options) : Node("inspection_planner_node", options) {

    declare_parameters();
    // load config and create planner object here
    
    
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

void InspectionPlannerNode::declare_parameters() {
    // Frames
    this->declare_parameter("global_frame", "odom");
    this->declare_parameter("drone_frame", "base_link");
    
    // Rates
    this->declare_parameter("planner_rate", 2.0);
    
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

void InspectionPlannerNode::planner_timer_callback() {

    // Lock map whilst reading map (shared_lock blocks only other writers)
    std::shared_lock lock(map_->mutex_);
    std::cout << "Num valid surfels: " << map_->num_valid_surfels() << std::endl;
    lock.unlock(); // unlock map

    // do planning?

    // publish path... 

    return;
}




} // namespace