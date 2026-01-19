#include "sparse_surfel_mapping/ros_nodes/planner_node.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace sparse_surfel_map {

InspectionPlannerNode::InspectionPlannerNode(const rclcpp::NodeOptions& options) : Node("inspection_planner_node", options) {
    declare_parameters();
    config_ = load_configuration();
    planner_ = std::make_unique<InspectionPlanner>(config_);
    
    // TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // ACTION
    trajectory_client_ = rclcpp_action::create_client<ExecutePath>(this, action_name_);

    // PUBLISHERS
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("inspection_planner/path", 10);
    fov_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inspection_planner/fov_cloud", 10);
    fov_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / 5.0),
        std::bind(&InspectionPlannerNode::publish_fov_pointcloud, this)
    );

    // TIMERS
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
    
    // Action and Topics
    this->declare_parameter("trajectory_action", "execute_path");

    // Rates
    this->declare_parameter("planner_rate", 2.0);
    this->declare_parameter("safety_rate", 10.0);
    
    // Trajectory

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

    // Inspection
    this->declare_parameter("inspection.max_viewpoints_per_plan", 5);

    // Load immediate parameters
    global_frame_ = this->get_parameter("global_frame").as_string();
    drone_frame_ = this->get_parameter("drone_frame").as_string();
    action_name_ = this->get_parameter("trajectory_action").as_string();
    planner_rate_ = this->get_parameter("planner_rate").as_double();
    safety_rate_ = this->get_parameter("safety_rate").as_double();
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
        }
        return;
    }
    received_first_pose_ = true;

    if (!trajectory_client_->action_server_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Waiting for trajectory action server...");
        return;
    }

    std::shared_lock lock(map_->mutex_);
    planner_->update_pose(current_position_, current_yaw_);

    // Check inspection completion
    if (planner_->is_complete()) {
        RCLCPP_INFO(this->get_logger(), "Inspection complete! Coverage: %.1f%%", 
            planner_->statistics().coverage_ratio * 100.0f);
        is_active_ = false;
        lock.unlock();
        return;
    }

    // Plan
    if (planner_->plan()) {
        lock.unlock();
        send_path_goal();
        // if (should_send_new_path()) send_path_goal();
    }
    else lock.unlock();


}

void InspectionPlannerNode::safety_timer_callback() {
    if (!is_active_ || !map_) return;
    if (emergency_stop_active_) return;

    if (!get_current_pose(current_position_, current_yaw_)) return;
    
    planner_->update_pose(current_position_, current_yaw_);

    std::shared_lock lock(map_->mutex_);
    planner_->validate_viewpoints();
    lock.unlock();

}

void InspectionPlannerNode::send_path_goal() {
    if (!planner_->has_plan()) {
        RCLCPP_WARN(this->get_logger(), "Cannot send goal - no active plan");
        return;
    }

    if (waiting_for_goal_response_) {
        RCLCPP_WARN(this->get_logger(), "Already waiting for goal response, skipping...");
        return;
    }

    // Get current path
    nav_msgs::msg::Path nav_path = convert_to_nav_path();

    if (nav_path.poses.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty path, not sending goal...");
        return;
    }

    auto goal = ExecutePath::Goal();
    goal.path = nav_path;
    goal.plan_id = current_plan_id_;
    goal.pos_tolerance = trajectory_pos_tolerance_;
    goal.yaw_tolerance = trajectory_yaw_tolerance_;
    goal.v_max = trajectory_v_max_;
    goal.a_max = trajectory_a_max_;

    last_sent_path_size_ = nav_path.poses.size();
    if (!planner_->viewpoints().empty()) {
        last_sent_first_vp_id_ = planner_->viewpoints().front().id();
    }

    auto send_goal_options = rclcpp_action::Client<ExecutePath>::SendGoalOptions();
    send_goal_options.goal_response_callback = std::bind(&InspectionPlannerNode::goal_response_callback, this, std::placeholders::_1);
    send_goal_options.feedback_callback = std::bind(&InspectionPlannerNode::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback = std::bind(&InspectionPlannerNode::result_callback, this, std::placeholders::_1);

    // RCLCPP_INFO(this->get_logger(), "ExecutePath goal: plan_id=%lu, %zu waypoints, v_max=%.1f", current_plan_id_, nav_path.poses.size(), trajectory_v_max_);

    waiting_for_goal_response_ = true;
    trajectory_client_->async_send_goal(goal, send_goal_options);
}

void InspectionPlannerNode::cancel_execution() {
    if (current_goal_handle_) {
        RCLCPP_INFO(this->get_logger(), "Cancelling current path");
        trajectory_client_->async_cancel_goal(current_goal_handle_);
        current_goal_handle_.reset();
        goal_in_progress_ = false;
    }
}

void InspectionPlannerNode::goal_response_callback(const GoalHandleExecutePath::SharedPtr& goal_handle) {
    waiting_for_goal_response_ = false;

    if (!goal_handle) {
        // RCLCPP_ERROR(this->get_logger(), "Goal was rejected by trajectory server");
        goal_in_progress_ = false;
        return;
    }

    // RCLCPP_INFO(this->get_logger(), "Goal accepted by trajectory server");
    current_goal_handle_ = goal_handle;
    goal_in_progress_ = true;
    trajectory_active_index_ = 0;
}

void InspectionPlannerNode::feedback_callback(GoalHandleExecutePath::SharedPtr, const std::shared_ptr<const ExecutePath::Feedback> feedback) {
    uint32_t new_index = feedback->active_index;
    trajectory_progress_ = feedback->progress;
    trajectory_state_ = feedback->state;

    if (new_index > trajectory_active_index_) {
        size_t viewpoints_reached = new_index - trajectory_active_index_;
        RCLCPP_INFO(this->get_logger(), 
            "Trajectory Feedback: index=%u->%u, progress=%.1f%%m state=%s",
            trajectory_active_index_, new_index, trajectory_progress_ * 100.0f, trajectory_state_.c_str()
        );

        // mark visited
        std::shared_lock lock(map_->mutex_);
        for (size_t i = 0; i < viewpoints_reached; ++i) {
            if (planner_->has_plan()) {
                RCLCPP_INFO(this->get_logger(), "Marking viewpoints as reached (feedback index advance)");
                planner_->mark_target_reached();
            }
        }
        lock.unlock();
    
        trajectory_active_index_ = new_index;
    }
}

void InspectionPlannerNode::result_callback(const GoalHandleExecutePath::WrappedResult& result) {
    goal_in_progress_ = false;
    current_goal_handle_.reset();
    trajectory_active_index_ = 0;

    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "Trajectory completed succesfully: %s", result.result->message.c_str());
            if (planner_->has_plan()) {
                std::shared_lock lock(map_->mutex_);
                planner_->mark_target_reached();
                lock.unlock();
            }
            break;

        case rclcpp_action::ResultCode::ABORTED:
            // RCLCPP_WARN(this->get_logger(), "Trajectory aborted: %s", result.result->message.c_str());
            break;
        
        case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_INFO(this->get_logger(), "Trajectory cancelled: %s", result.result->message.c_str());
            break;
        
        default:
            RCLCPP_ERROR(this->get_logger(), "Unkown trajectory result code");
            break;
    }

    if (planner_->has_plan() && is_active_ && !emergency_stop_active_) {
        // RCLCPP_INFO(this->get_logger(), "Trajectory complete, checking for more waypoints...");
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

nav_msgs::msg::Path InspectionPlannerNode::convert_to_nav_path() {
    nav_msgs::msg::Path nav_path;
    nav_path.header.frame_id = global_frame_;
    nav_path.header.stamp = this->get_clock()->now();
    
    auto internal_path = planner_->viewpoints(); // copy
    if (internal_path.size() < 1) return nav_path;

    auto it = internal_path.begin();
    while (it != internal_path.end()) {
        const ViewpointState vpstate = it->state();
        geometry_msgs::msg::PoseStamped pose;
        pose.header = nav_path.header;
        pose.pose.position.x = vpstate.position.x();
        pose.pose.position.y = vpstate.position.y();
        pose.pose.position.z = vpstate.position.z();

        tf2::Quaternion q;
        q.setRPY(0, 0, vpstate.yaw);
        pose.pose.orientation = tf2::toMsg(q);

        nav_path.poses.push_back(pose);
        it++;
    }

    return nav_path;
}

bool InspectionPlannerNode::should_send_new_path() const {
    if (!planner_->has_plan()) return false;
    if (waiting_for_goal_response_) return false;
    if (!goal_in_progress_) return true;
    
    // Check if path has changed
    if (!planner_->viewpoints().empty()) {
        uint64_t current_firts_id = planner_->viewpoints().front().id();
        if (current_firts_id != last_sent_first_vp_id_) {
            return true;
        }
    }

    // Check if path extended
    size_t current_size = planner_->viewpoints().size();
    if (current_size > last_sent_path_size_ + 1) {
        return true;
    }

    return false;
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
    const VoxelKeySet& observed_voxels = planner_->coverage().observed_voxels();

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