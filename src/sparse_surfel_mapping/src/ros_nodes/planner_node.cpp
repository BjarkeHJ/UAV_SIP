#include "sparse_surfel_mapping/ros_nodes/planner_node.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace sparse_surfel_map {

InspectionPlannerNode::InspectionPlannerNode(const rclcpp::NodeOptions& options) : Node("inspection_planner_node", options) {

    declare_parameters();
    // load config and create planner object
    config_ = load_configuration();
    planner_ = std::make_unique<InspectionPlanner>(config_);
    
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("inspection_planner/path", 10);

    if (safety_rate_ > 0.0) {
        safety_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / safety_rate_),
            std::bind(&InspectionPlannerNode::safety_timer_callback, this)
        );
    }

    if (planner_rate_ > 0.0) {
        plan_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / planner_rate_),
            std::bind(&InspectionPlannerNode::planner_timer_callback, this)
        );
    }

    fov_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inspection_planner/fov_cloud", 10);
    fov_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / 5.0),
        std::bind(&InspectionPlannerNode::publish_fov_pointcloud, this)
    );

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
    if (emergency_stop_active_) return;

    if (!get_current_pose(current_position_, current_yaw_)) {
        if (!received_first_pose_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Waiting for drone pose...");
            return;
        }
    }
    received_first_pose_ = true;

    std::shared_lock lock(map_->mutex_);

    planner_->update_state(current_position_, current_yaw_);

    // Check inspection completion
    if (planner_->is_inspection_complete()) {
        RCLCPP_INFO(this->get_logger(), "Inspection complete! Coverage: %.1f%%", 
            planner_->statistics().coverage_ratio * 100.0f);
        is_active_ = false;
        lock.unlock();
        return;
    }

    // === PLANNING LOGIC ===
    // Priority: 
    // 1. If no plan exists -> full replan
    // 2. If horizon buffer low -> extend
    // 3. Otherwise -> do nothing (already have good plan)
    
    bool plan_changed = false;

    if (!planner_->has_active_plan()) {
        // No plan at all - need full replan
        RCLCPP_INFO(this->get_logger(), "No active plan - initiating full replan");
        
        if (planner_->plan()) {
            plan_changed = true;
            RCLCPP_INFO(this->get_logger(), "Full replan succeeded: %zu viewpoints", 
                planner_->remaining_viewpoints());
        } else {
            RCLCPP_WARN(this->get_logger(), "Full replan failed!");
        }
    }
    else if (planner_->needs_extension()) {
        // Horizon buffer is low - extend the plan
        RCLCPP_INFO(this->get_logger(), 
            "Horizon buffer low (%zu uncommitted) - extending", 
            planner_->uncommitted_count());
        
        if (planner_->extend_plan()) {
            std::cout << "GOES HERE" << std::endl;
            plan_changed = true;
            RCLCPP_INFO(this->get_logger(), 
                "Plan extended: %zu total (%zu committed, %zu uncommitted)",
                planner_->remaining_viewpoints(),
                planner_->committed_count(),
                planner_->uncommitted_count());
        }
    }
    // else: Plan is good, nothing to do
    
    lock.unlock();

    if (plan_changed) {
        std::cout << "HELLO!" << std::endl;
        publish_path();
    }

    std::cout << "RUNNING!" << std::endl;
}

void InspectionPlannerNode::safety_timer_callback() {
    
    if (!is_active_ || !map_) return;
    if (emergency_stop_active_) return;

    if (!get_current_pose(current_position_, current_yaw_)) return;
    
    planner_->update_state(current_position_, current_yaw_);
    
    if (has_reached_target()) {
        RCLCPP_INFO(this->get_logger(), "Target reached!");
        
        std::shared_lock lock(map_->mutex_);
        planner_->mark_target_reached();
        lock.unlock();
        
        publish_path();
        return;
    }

    std::shared_lock lock(map_->mutex_);
    auto state = planner_->evaluate_and_react();
    lock.unlock();
    
    if (state == InspectionPlanner::PlannerState::EMERGENCY_STOP) {
        // !!! EMERGENCY - collision in committed path segment !!!
        emergency_stop_active_ = true;
        publish_emergency_stop();
        
        RCLCPP_ERROR(this->get_logger(), 
            "!!! EMERGENCY STOP !!! Collision detected in committed path");
        
        // Deactivate normal planning
        is_active_ = false;
    }
    else if (state == InspectionPlanner::PlannerState::EXECUTING) {
        // Path was modified due to collision in uncommitted segment
        // Publish updated path
        publish_path();
    }

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
    if (!path.is_valid) return; // maybe publish empty path?

    nav_msgs::msg::Path msg;
    msg.header.frame_id = global_frame_;
    msg.header.stamp = this->get_clock()->now();

    if (path.waypoints.size() < 2) return;

    for (size_t i = 1; i < path.waypoints.size(); ++i) {
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

    RCLCPP_INFO(this->get_logger(), "Publishing path with %zu waypoints", msg.poses.size());
    path_pub_->publish(msg);
}

void InspectionPlannerNode::publish_emergency_stop() {
    nav_msgs::msg::Path empty_path;
    empty_path.header.frame_id = global_frame_;
    empty_path.header.stamp = this->get_clock()->now();
    path_pub_->publish(empty_path);
}

void InspectionPlannerNode::publish_fov_pointcloud() {
    if (!map_) return;

    // Create a viewpoint at current drone position to compute visibility
    Viewpoint current_view(current_position_, current_yaw_, config_.camera);
    
    // Shared lock for map access
    std::shared_lock lock(map_->mutex_);
    
    // Compute which voxels are visible from current pose
    // Using occlusion checking for accurate visibility
    current_view.compute_visibility(*map_, true);
    
    lock.unlock();

    const VoxelKeySet& visible_voxels = current_view.visible_voxels();
    
    if (visible_voxels.empty()) {
        return;
    }

    // Create PointCloud2 message
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = global_frame_;
    cloud_msg.header.stamp = this->get_clock()->now();

    // Set up point cloud fields (x, y, z, intensity)
    cloud_msg.height = 1;
    cloud_msg.width = visible_voxels.size();
    cloud_msg.is_dense = true;
    cloud_msg.is_bigendian = false;

    // Define fields
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(visible_voxels.size());

    // Iterators for filling the cloud
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");

    const float voxel_size = map_->voxel_size();

    // Check which voxels have been observed before (for coloring)
    const VoxelKeySet& observed_voxels = planner_->coverage_tracker().observed_voxels();

    for (const auto& key : visible_voxels) {
        // Compute voxel center
        float x = (key.x + 0.5f) * voxel_size;
        float y = (key.y + 0.5f) * voxel_size;
        float z = (key.z + 0.5f) * voxel_size;

        *iter_x = x;
        *iter_y = y;
        *iter_z = z;

        // Color: Green = new (not yet observed), Yellow = already observed
        bool already_observed = observed_voxels.count(key) > 0;
        
        if (already_observed) {
            // Yellow - already covered
            *iter_r = 255;
            *iter_g = 200;
            *iter_b = 0;
        } else {
            // Green - new coverage
            *iter_r = 0;
            *iter_g = 255;
            *iter_b = 100;
        }

        ++iter_x; ++iter_y; ++iter_z;
        ++iter_r; ++iter_g; ++iter_b;
    }

    fov_cloud_pub_->publish(cloud_msg);
}


} // namespace