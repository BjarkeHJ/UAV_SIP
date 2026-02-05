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
        // std::bind(&InspectionPlannerNode::publish_fov_pointcloud, this)
        std::bind(&InspectionPlannerNode::publish_cands, this)
    );
    surfel_map_coverage_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/surfel_map/coverage_map", 10);
    marker_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / 5.0),
        std::bind(&InspectionPlannerNode::publish_surfel_coverage, this)
    );

    vpt_cands_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("temporary/cand_vpts", 10);
    
    pcd_map_coverage_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/surfel_map/coverage_cloud", 10);
    pcd_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / 5.0),
        std::bind(&InspectionPlannerNode::publish_pcd_coverage, this)
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
    this->declare_parameter("target_reach_th", 0.1);
    this->declare_parameter("target_yaw_th", 0.05);

    // Camera
    this->declare_parameter("camera.hfov_deg", 90.0);
    this->declare_parameter("camera.vfov_deg", 60.0);
    this->declare_parameter("camera.min_range",0.5);
    this->declare_parameter("camera.max_range", 8.0);

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

        RCLCPP_INFO(this->get_logger(), "PLANNER: Planned path of %lu viewpoints...", planner_->viewpoints().size());

        if (should_send_new_path()) {
            send_path_goal();
        }
    }
    else {
        lock.unlock();
        if (planner_->viewpoints().empty()) {
            RCLCPP_WARN(this->get_logger(), "PLANNER: Failed to generate viewpoints...");
            planner_->request_replan();
        }
    }
}

void InspectionPlannerNode::safety_timer_callback() {
    if (!is_active_ || !map_ || !planner_) return;
    if (!planner_->has_plan()) return; // nothing to safety check

    if (!get_current_pose(current_position_, current_yaw_)) return;
    planner_->update_pose(current_position_, current_yaw_);

    std::shared_lock lock(map_->mutex_);

    // Check viewpoints
    bool viewpoints_left = planner_->validate_viewpoints();
    if (!viewpoints_left) {
        RCLCPP_WARN(this->get_logger(), "[ViewpointValidaton] All viewpoint invalid! Replan");
        planner_->request_replan();
        return;
    }

    // Check path 
    bool path_valid = planner_->validate_path();
    lock.unlock();

    // If attempted repair on broken path failed -> cancel current execution and replan
    if (!path_valid) {
        RCLCPP_WARN(this->get_logger(), "Path replan failed - cancelling execution");
        if (goal_in_progress_) {
            cancel_execution();
            planner_->request_replan();
        }
    }
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

    // From the currently proposed viewpoints - generate/ensure a collision-free path (RRT)
    current_planned_path_ = planner_->generate_path();
    if (current_planned_path_.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty planned path, not sending goal...");
        return;
    }

    nav_msgs::msg::Path nav_path = convert_to_nav_path(current_planned_path_);
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
    last_completed_viewpoint_idx_ = -1; // reset viewpoint tracking

    if (!planner_->viewpoints().empty()) {
        last_sent_first_vp_id_ = planner_->viewpoints().front().id;
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
        process_path_progress(new_index);
        trajectory_active_index_ = new_index;
    }
}

void InspectionPlannerNode::process_path_progress(uint32_t new_index) {
    // check all indices from last processed to new_index
    for (uint32_t idx = trajectory_active_index_; idx < new_index; ++idx) {
        int vp_idx = planner_->get_viewpoint_index_for_path_index(idx);

        // this was an actual viewpoint (not rrt waypoint) -> mark as visited in coverage tracker
        if (vp_idx >= 0 && vp_idx > last_completed_viewpoint_idx_) {

            std::shared_lock lock(map_->mutex_);
            planner_->mark_target_reached();
            lock.unlock();
            RCLCPP_INFO(this->get_logger(), "Viewpoint reached - Current coverage: %.1f%%", planner_->coverage().coverage_ratio() * 100.0f);

            last_completed_viewpoint_idx_ = vp_idx;
        }
    }
}

void InspectionPlannerNode::result_callback(const GoalHandleExecutePath::WrappedResult& result) {
    goal_in_progress_ = false;
    current_goal_handle_.reset();
    trajectory_active_index_ = 0;

    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "Viewpoint reached - Current coverage: %.1f%%", planner_->coverage().coverage_ratio() * 100.0f);
            RCLCPP_INFO(this->get_logger(), "Trajectory completed succesfully: %s", result.result->message.c_str());
            
            std::cout << "DEBUG1: viewpoints_ size after reached end of traj" << planner_->viewpoints().size() << std::endl;

            if (planner_->has_plan()) {
                int final_vp_idx = planner_->get_viewpoint_index_for_path_index(current_planned_path_.size() - 1);
                if (final_vp_idx >= 0 && final_vp_idx > last_completed_viewpoint_idx_) {
                    std::shared_lock lock(map_->mutex_);
                    planner_->mark_target_reached();
                    lock.unlock();
                }
            }
            
            std::cout << "DEBUG2: viewpoints_ size after reached end of traj" << planner_->viewpoints().size() << std::endl;
            planner_->request_replan();

            break;

        case rclcpp_action::ResultCode::ABORTED:
            // RCLCPP_WARN(this->get_logger(), "Trajectory aborted: %s", result.result->message.c_str());
            break;
        
        case rclcpp_action::ResultCode::CANCELED:
            // RCLCPP_INFO(this->get_logger(), "Trajectory cancelled: %s", result.result->message.c_str());
            break;
        
        default:
            // RCLCPP_ERROR(this->get_logger(), "Unkown trajectory result code");
            break;
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

nav_msgs::msg::Path InspectionPlannerNode::convert_to_nav_path(const RRTPath& planned_path) {
    nav_msgs::msg::Path nav_path;
    nav_path.header.frame_id = global_frame_;
    nav_path.header.stamp = this->get_clock()->now();
    
    for (size_t i = 0; i < planned_path.size(); ++i) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = nav_path.header;
        pose.pose.position.x = planned_path.positions[i].x();
        pose.pose.position.y = planned_path.positions[i].y();
        pose.pose.position.z = planned_path.positions[i].z();

        tf2::Quaternion q;
        q.setRPY(0, 0, planned_path.yaws[i]);
        pose.pose.orientation = tf2::toMsg(q);
        
        nav_path.poses.push_back(pose);
    }

    return nav_path;
}

bool InspectionPlannerNode::should_send_new_path() const {
    if (!planner_->has_plan()) return false;
    if (waiting_for_goal_response_) return false;
    if (!goal_in_progress_) return true;
    
    // Check if path has changed
    if (!planner_->viewpoints().empty()) {
        uint64_t current_firts_id = planner_->viewpoints().front().id;
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
    const VoxelKeySet& observed_voxels = planner_->coverage().observed_surfels();

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

void InspectionPlannerNode::publish_surfel_coverage() {
    if (!map_) return;
    visualization_msgs::msg::MarkerArray marker_array;

    const auto& surfels = map_->get_valid_surfels();
    const VoxelKeySet& obs_set = planner_->coverage().observed_surfels();
    const std::unordered_map<VoxelKey, size_t, VoxelKeyHash> obs_counts = planner_->coverage().observations_counts();

    const VoxelKeySet& frontier_set = planner_->coverage().coverage_frontiers();

    auto viz_now = this->get_clock()->now();

    // delete previous
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = global_frame_;
    delete_marker.header.stamp = viz_now;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_marker.ns = "surfel_coverage_state";
    marker_array.markers.push_back(delete_marker);

    // ellipses
    visualization_msgs::msg::Marker surfel;
    surfel.header.frame_id = global_frame_;
    surfel.header.stamp = viz_now;
    surfel.ns = "surfel_coverage_state";
    surfel.type = visualization_msgs::msg::Marker::CYLINDER;
    surfel.action = visualization_msgs::msg::Marker::ADD;

    int marker_id = 0;
    // size_t max_obs = 0;

    for (const auto& sref : surfels) {
        const Surfel& s = sref.get();
        surfel.id = marker_id++;

        // pos
        const Eigen::Vector3f& m = s.mean();
        surfel.pose.position.x = m.x();
        surfel.pose.position.y = m.y();
        surfel.pose.position.z = m.z();

        // ori
        const Eigen::Matrix3f& C = s.eigenvectors();
        const Eigen::Vector3f& ev1 = C.col(1);
        const Eigen::Vector3f& ev2 = C.col(2);
        Eigen::Matrix3f R;
        R.col(0) = ev1;
        R.col(1) = ev2;
        R.col(2) = s.normal();
        if (R.determinant() < 0) {
            R.col(1) = -R.col(1);
        }

        Eigen::Quaternionf q(R);
        q.normalize();

        surfel.pose.orientation.x = q.x();
        surfel.pose.orientation.y = q.y();
        surfel.pose.orientation.z = q.z();
        surfel.pose.orientation.w = q.w();

        // scale
        const Eigen::Vector3f& evals = s.eigenvalues();
        surfel.scale.x = 2.0f * std::sqrt(std::max(evals(1), 1e-6f));
        surfel.scale.y = 2.0f * std::sqrt(std::max(evals(2), 1e-6f));
        surfel.scale.z = 0.005f;

        // color (coverage state)        
        const VoxelKey& skey = s.key();
        auto it = obs_set.find(skey);
        auto fit = frontier_set.find(skey);

        if (it != obs_set.end()) {
            // const size_t count = obs_counts.find(skey)->second;
            // if (count > max_obs) max_obs = count;
            surfel.color.r = 0.0f; 
            // surfel.color.g = static_cast<float>(count) / static_cast<float>(max_obs);
            surfel.color.g = 1.0f;
            surfel.color.b = 0.0f;
            surfel.color.a = 0.8f;
        }
        else if (fit != frontier_set.end()) {
            surfel.color.r = 1.0f;
            surfel.color.g = 0.0f;
            surfel.color.b = 1.0f;
            surfel.color.a = 0.8f;
        }
        else {
            surfel.color.r = 0.0f; 
            surfel.color.g = 0.0f;
            surfel.color.b = 0.0f;
            surfel.color.a = 0.4f;
        }

        marker_array.markers.push_back(surfel);
    }

    surfel_map_coverage_pub_->publish(marker_array);

}

void InspectionPlannerNode::publish_cands() {
    if (!map_) return;

    geometry_msgs::msg::PoseArray cands_msg;
    cands_msg.header.frame_id = global_frame_;
    cands_msg.header.stamp = this->get_clock()->now();

    const auto& cands = planner_->vpt_cand();
    if (cands.empty()) return;

    for (const auto& c : cands) {
        geometry_msgs::msg::Pose p;
        p.position.x = c.position.x();
        p.position.y = c.position.y();
        p.position.z = c.position.z();

        tf2::Quaternion q;
        q.setRPY(0, 0, c.yaw);
        p.orientation = tf2::toMsg(q);

        cands_msg.poses.push_back(p);
    }

    vpt_cands_pub_->publish(cands_msg);

}

void InspectionPlannerNode::publish_pcd_coverage() {
    if (!map_) return;
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = global_frame_;
    cloud_msg.header.stamp = this->get_clock()->now();

    // get surfels
    std::shared_lock lock(map_->mutex_);
    const auto& surfels = map_->get_valid_surfels();
    lock.unlock();

    cloud_msg.height = 1;
    cloud_msg.width = surfels.size();
    cloud_msg.is_dense = true;
    cloud_msg.is_bigendian = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

    // iterators over message fields
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");

    // get keys for observed surfels
    const VoxelKeySet& obs_set = planner_->coverage().observed_surfels();
    // const VoxelKeySet& coverage_frontier_set = planner_->coverage().coverage_frontiers();
    const VoxelKeySet map_frontier_set = planner_->coverage().map_frontiers();

    for (const auto& sref : surfels) {
        const Surfel& surfel = sref.get();
        const Eigen::Vector3f smean = surfel.mean();
        float conf = surfel.confidence();

        *iter_x = smean.x();
        *iter_y = smean.y();
        *iter_z = smean.z();

        // float test = surfel.eigenvalues()(1) / surfel.eigenvalues()(2);
        // *iter_r = static_cast<uint8_t>(255.0f * test);
        // *iter_g = 0;
        // *iter_b = static_cast<uint8_t>(255.0f * (1.0f - test));

        *iter_r = static_cast<uint8_t>(255.0f * conf);
        *iter_g = 0;
        *iter_b = static_cast<uint8_t>(255.0f * (1.0f - conf));

        // color according to map frontier, observed, frontier, not-covered
        if (map_frontier_set.count(surfel.key()) > 0) {
            *iter_r = 255;
            *iter_g = 255;
            *iter_b = 255;
        }
        else if (obs_set.count(surfel.key()) > 0) {
            *iter_r = 0;
            *iter_g = 255;
            *iter_b = 0;
        }
        // else if (coverage_frontier_set.count(surfel.key()) > 0) {
        //     *iter_r = 255;
        //     *iter_g = 0;
        //     *iter_b = 255;
        // }
        // else {
        //     *iter_r = 10;
        //     *iter_g = 10;
        //     *iter_b = 10;
        // }

        // increment iterators
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_r;
        ++iter_g;
        ++iter_b;
    }

    pcd_map_coverage_pub_->publish(cloud_msg);
}

} // namespace