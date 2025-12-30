#include "surfel_mapping/surfel_mapping_node.hpp"

namespace surface_inspection_planning {

SurfelMappingNode::SurfelMappingNode() : Node("surfel_mapping_node") {
    // declare parameters
    declare_parameters();

    // initialize components
    initialize_preprocessor();
    initialize_fuser();

    // TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Pub
    surfel_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("surfel_map/markers", 10);
    processed_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("surfel_map/processed_cloud", 10);

    // Sub
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/x500/lidar_front/points_raw", 
        rclcpp::SensorDataQoS(), 
        std::bind(&SurfelMappingNode::cloud_callback, this, std::placeholders::_1)
    );

    // Viz timer
    double viz_rate = this->get_parameter("viz_rate").as_double();
    if (viz_rate > 0) {
        viz_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / viz_rate),
            std::bind(&SurfelMappingNode::publish_visualization, this)
        );
    }

    RCLCPP_INFO(this->get_logger(), "Surfel Mapping Node initialized");
}

void SurfelMappingNode::declare_parameters() {
    // Frame Parameters
    this->declare_parameter("world_frame", "odom");
    this->declare_parameter("sensor_frame", "camera_link");
    this->declare_parameter("use_tf", true);
    this->declare_parameter("viz_rate", 5.0);

    // Preprocessor Parameters
    this->declare_parameter("preprocess.width", 240);
    this->declare_parameter("preprocess.height", 180);
    this->declare_parameter("preprocess.hfov_deg", 106.0);
    this->declare_parameter("preprocess.vfov_deg", 86.0);
    this->declare_parameter("preprocess.ds_factor", 1);
    this->declare_parameter("preprocess.min_range", 0.1);
    this->declare_parameter("preprocess.max_range", 10.0);
    this->declare_parameter("preprocess.enable_gnd_filter", true);
    this->declare_parameter("preprocess.gnd_z_min", 0.25);
    this->declare_parameter("preprocess.range_smooth_iters", 3);

    // Surfel Map Parameters
    this->declare_parameter("map.voxel_size", 0.5);
    this->declare_parameter("map.normal_thresh_deg", 30.0);
    this->declare_parameter("map.mahal_thresh", 9.21);
    this->declare_parameter("map.normal_dist_thresh", 0.1);
    this->declare_parameter("map.min_surfel_radius", 0.02);
    this->declare_parameter("map.max_surfel_radius", 0.5);

    // Fusion Parameters
    this->declare_parameter("fuser.min_points_for_new_surfel", 5);
    this->declare_parameter("fuser.new_surfel_initial_radius", 0.1);
    this->declare_parameter("fuser.center_update_rate", 0.3);
    this->declare_parameter("fuser.normal_update_rate", 0.1);
    this->declare_parameter("fuser.confidence_boost", 0.05);
}

void SurfelMappingNode::initialize_preprocessor() {
    CloudPreprocess::Params pp;
    pp.width = this->get_parameter("preprocess.width").as_int();
    pp.height = this->get_parameter("preprocess.height").as_int();
    pp.hfov_deg = this->get_parameter("preprocess.hfov_deg").as_double();
    pp.vfov_deg = this->get_parameter("preprocess.vfov_deg").as_double();
    pp.ds_factor = this->get_parameter("preprocess.ds_factor").as_int();
    pp.min_range = this->get_parameter("preprocess.min_range").as_double();
    pp.max_range = this->get_parameter("preprocess.max_range").as_double();
    pp.enable_gnd_filter = this->get_parameter("preprocess.enable_gnd_filter").as_bool();
    pp.gnd_z_min = this->get_parameter("preprocess.gnd_z_min").as_double();
    pp.range_smooth_iters = this->get_parameter("preprocess.range_smooth_iters").as_int();
    
    preprocessor_ = std::make_unique<CloudPreprocess>(pp);
}

void SurfelMappingNode::initialize_fuser() {
    SurfelMap::Params mp;
    mp.voxel_size = this->get_parameter("map.voxel_size").as_double();
    mp.normal_thresh_deg = this->get_parameter("map.normal_thresh_deg").as_double();
    mp.mahal_thresh = this->get_parameter("map.mahal_thresh").as_double();
    mp.normal_dist_thresh = this->get_parameter("map.normal_dist_thresh").as_double();
    mp.min_surfel_radius = this->get_parameter("map.min_surfel_radius").as_double();
    mp.max_surfel_radius = this->get_parameter("map.max_surfel_radius").as_double();
    
    SurfelFusion::Params fp;
    fp.min_points_for_new_surfel = this->get_parameter("fuser.min_points_for_new_surfel").as_int();
    fp.new_surfel_initial_radius = this->get_parameter("fuser.new_surfel_initial_radius").as_double();
    fp.center_update_rate = this->get_parameter("fuser.center_update_rate").as_double();
    fp.normal_update_rate = this->get_parameter("fuser.normal_update_rate").as_double();
    fp.confidence_boost = this->get_parameter("fuser.confidence_boost").as_double();
    
    fuser_ = std::make_unique<SurfelFusion>(fp, mp);
}

void SurfelMappingNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // get pose 
    Eigen::Isometry3f pose;
    if (!get_sensor_pose(msg->header.stamp, pose)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to get sensor pose, skipping cloud");
        return;
    }

    // Conver msg to pcl
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_in);

    if (cloud_in->empty()) return;

    // set transform for gnd filtering
    Eigen::Quaternionf q(pose.rotation());
    preprocessor_->set_world_transform(pose.translation(), q);

    auto pp_start = std::chrono::high_resolution_clock::now();
    // preprocess
    preprocessor_->set_input_cloud(cloud_in);
    preprocessor_->downsample();
    preprocessor_->normal_estimation();
    // preprocessor_->transform_output_to_world();
    auto pp_end = std::chrono::high_resolution_clock::now();
    double pp_time = std::chrono::duration<double, std::milli>(pp_end - pp_start).count();
    RCLCPP_INFO(this->get_logger(),
        "Preprocess Time: %.f ms", pp_time
    );

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    preprocessor_->get_points(cloud);
    preprocessor_->get_normals(normals);


    if (cloud->empty() || normals->empty()) return;

    uint64_t timestamp = rclcpp::Time(msg->header.stamp).nanoseconds();
    fuser_->process_scan(cloud, normals, pose, timestamp);

    const auto& stats = fuser_->last_stats();
    RCLCPP_INFO(this->get_logger(),
        "Fusion: %zu pts, %zu assoc, %zu new surfels, %.f ms",
        stats.points_processed, stats.points_associated,
        stats.surfels_created, stats.processing_time_ms
    );

    if (processed_cloud_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*cloud, out_msg);
        out_msg.header.frame_id = this->get_parameter("world_frame").as_string();
        out_msg.header.stamp = msg->header.stamp;
        processed_cloud_pub_->publish(out_msg);
    }
}

bool SurfelMappingNode::get_sensor_pose(const rclcpp::Time& stamp, Eigen::Isometry3f& pose) {
    std::string world_frame = this->get_parameter("world_frame").as_string();
    std::string sensor_frame = this->get_parameter("sensor_frame").as_string();

    try {
        auto tf = tf_buffer_->lookupTransform(world_frame, sensor_frame, stamp, rclcpp::Duration::from_seconds(0.1));
        pose = tf2::transformToEigen(tf.transform).cast<float>();
        return true;
    }
    catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF Lookup Failed: %s", ex.what());
        return false;
    }

    // TODO: Enable direct odometry (use_tf = false)
}

void SurfelMappingNode::publish_visualization() {
    if (surfel_marker_pub_->get_subscription_count() == 0) return;

    const auto& surfels = fuser_->map().get_surfels();
    if (surfels.empty()) return;

    visualization_msgs::msg::MarkerArray markers;
    std::string world_frame = this->get_parameter("world_frame").as_string();
    auto now = this->get_clock()->now();
    
    // Surfel ellipses
    visualization_msgs::msg::Marker ellipse_marker;
    ellipse_marker.header.frame_id = world_frame;
    ellipse_marker.header.stamp = now;
    ellipse_marker.ns = "surfel_ellipses";
    ellipse_marker.type = visualization_msgs::msg::Marker::CYLINDER;
    ellipse_marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Surfel normals
    visualization_msgs::msg::Marker normal_marker;
    normal_marker.header.frame_id = world_frame;
    normal_marker.header.stamp = now;
    normal_marker.ns = "surfel_normals";
    normal_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    normal_marker.action = visualization_msgs::msg::Marker::ADD;
    normal_marker.id = 0;
    normal_marker.scale.x = 0.005;
    normal_marker.color.r = 0.0f;
    normal_marker.color.g = 0.0f;
    normal_marker.color.b = 1.0f;
    normal_marker.color.a = 0.8f;
    
    int id = 0;
    for (const auto& surfel : surfels) {
        if (!surfel.is_valid) continue;
        
        // Ellipse Visualization
        ellipse_marker.id = id++;
        ellipse_marker.pose.position.x = surfel.center.x();
        ellipse_marker.pose.position.y = surfel.center.y();
        ellipse_marker.pose.position.z = surfel.center.z();
        
        // Get eigenvalues (standard deviations along principal axes)
        float sigma1 = std::sqrt(std::max(surfel.eigenvalues(0), 1e-6f));
        float sigma2 = std::sqrt(std::max(surfel.eigenvalues(1), 1e-6f));
        
        // Eigenvectors in 2D tangent space
        Eigen::Vector2f ev1_2d = surfel.eigenvectors.col(0);
        Eigen::Vector2f ev2_2d = surfel.eigenvectors.col(1);
        
        // Transform eigenvectors to 3D world space using tangent basis
        // principal_axis_1 = ev1_2d.x * tangent_u + ev1_2d.y * tangent_v
        Eigen::Vector3f principal_axis_1 = (ev1_2d.x() * surfel.tangent_u + 
                                            ev1_2d.y() * surfel.tangent_v).normalized();
        Eigen::Vector3f principal_axis_2 = (ev2_2d.x() * surfel.tangent_u + 
                                            ev2_2d.y() * surfel.tangent_v).normalized();
        
        // Build rotation matrix: columns are the local axes
        // For CYLINDER marker: X and Y are the disc plane, Z is the axis (normal)
        // We want: local X → principal_axis_1, local Y → principal_axis_2, local Z → normal
        Eigen::Matrix3f R;
        R.col(0) = principal_axis_1;
        R.col(1) = principal_axis_2;
        R.col(2) = surfel.normal;
        
        // Ensure right-handed (fix if eigenvectors created left-handed system)
        if (R.determinant() < 0) {
            R.col(1) = -R.col(1);
        }
        
        // Convert rotation matrix to quaternion
        Eigen::Quaternionf q(R);
        q.normalize();
        
        ellipse_marker.pose.orientation.x = q.x();
        ellipse_marker.pose.orientation.y = q.y();
        ellipse_marker.pose.orientation.z = q.z();
        ellipse_marker.pose.orientation.w = q.w();
        
        // Scale: diameter along each principal axis (2 * sigma for 1-sigma ellipse)
        // Use 2-sigma for better visibility
        const float sigma_scale = 1.0f;
        ellipse_marker.scale.x = sigma1 * 2.0f * sigma_scale;  // Diameter along principal axis 1
        ellipse_marker.scale.y = sigma2 * 2.0f * sigma_scale;  // Diameter along principal axis 2
        ellipse_marker.scale.z = 0.005f;                       // Thin disc
        
        // Color based on confidence
        float conf = std::clamp(surfel.confidence, 0.0f, 1.0f);
        ellipse_marker.color.r = 1.0f - conf;
        ellipse_marker.color.g = conf;
        ellipse_marker.color.b = 0.2f;
        ellipse_marker.color.a = 0.6f;
        
        markers.markers.push_back(ellipse_marker);
        
        // --- Normal line ---
        geometry_msgs::msg::Point p1, p2;
        p1.x = surfel.center.x();
        p1.y = surfel.center.y();
        p1.z = surfel.center.z();
        
        float normal_length = 0.1f;
        p2.x = surfel.center.x() + surfel.normal.x() * normal_length;
        p2.y = surfel.center.y() + surfel.normal.y() * normal_length;
        p2.z = surfel.center.z() + surfel.normal.z() * normal_length;
        
        normal_marker.points.push_back(p1);
        normal_marker.points.push_back(p2);
    }
    
    markers.markers.push_back(normal_marker);
    
    // Delete old markers (send deleteall first, then new markers)
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = world_frame;
    delete_marker.header.stamp = now;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_marker.ns = "surfel_ellipses";
    // Note: DELETEALL should be sent in a separate publish or handled differently
    // For simplicity, we rely on marker IDs being reused
    
    surfel_marker_pub_->publish(markers);
    
    // Log map stats
    auto stats = fuser_->map().get_stats();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Map: %zu surfels, %zu voxels, avg conf: %.2f, avg pts: %.1f",
        stats.valid_surfels, stats.voxels_occupied, 
        stats.avg_confidence, stats.avg_point_count);    
}

// void SurfelMappingNode::publish_visualization() {
//     // Show discs
//     if (surfel_marker_pub_->get_subscription_count() == 0) return;

//     const auto& surfels = fuser_->map().get_surfels();
//     if (surfels.empty()) return;

//     visualization_msgs::msg::MarkerArray markers;
//     std::string world_frame = this->get_parameter("world_frame").as_string();
//     auto now = this->get_clock()->now();
    
//     // Surfel discs
//     visualization_msgs::msg::Marker disc_marker;
//     disc_marker.header.frame_id = world_frame;
//     disc_marker.header.stamp = now;
//     disc_marker.ns = "surfel_discs";
//     disc_marker.type = visualization_msgs::msg::Marker::CYLINDER;
//     disc_marker.action = visualization_msgs::msg::Marker::ADD;
    
//     // Surfel normals
//     visualization_msgs::msg::Marker normal_marker;
//     normal_marker.header.frame_id = world_frame;
//     normal_marker.header.stamp = now;
//     normal_marker.ns = "surfel_normals";
//     normal_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
//     normal_marker.action = visualization_msgs::msg::Marker::ADD;
//     normal_marker.id = 0;
//     normal_marker.scale.x = 0.005;  // Line width
//     normal_marker.color.r = 0.0f;
//     normal_marker.color.g = 0.0f;
//     normal_marker.color.b = 1.0f;
//     normal_marker.color.a = 0.8f;
    
//     int id = 0;
//     for (const auto& surfel : surfels) {
//         if (!surfel.is_valid) continue;
        
//         // Disc visualization
//         disc_marker.id = id++;
//         disc_marker.pose.position.x = surfel.center.x();
//         disc_marker.pose.position.y = surfel.center.y();
//         disc_marker.pose.position.z = surfel.center.z();
        
//         // Orientation: cylinder axis along Z, we want it along normal
//         Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(
//             Eigen::Vector3f::UnitZ(), surfel.normal);
//         disc_marker.pose.orientation.x = q.x();
//         disc_marker.pose.orientation.y = q.y();
//         disc_marker.pose.orientation.z = q.z();
//         disc_marker.pose.orientation.w = q.w();
        
//         // Size based on eigenvalues
//         float radius = surfel.get_radius();
//         disc_marker.scale.x = radius * 2.0f;  // Diameter X
//         disc_marker.scale.y = radius * 2.0f;  // Diameter Y
//         disc_marker.scale.z = 0.01f;          // Thin disc
        
//         // Color based on confidence
//         float conf = std::clamp(surfel.confidence, 0.0f, 1.0f);
//         disc_marker.color.r = 1.0f - conf;
//         disc_marker.color.g = conf;
//         disc_marker.color.b = 0.2f;
//         disc_marker.color.a = 0.6f;
        
//         markers.markers.push_back(disc_marker);
        
//         // Normal line
//         geometry_msgs::msg::Point p1, p2;
//         p1.x = surfel.center.x();
//         p1.y = surfel.center.y();
//         p1.z = surfel.center.z();
        
//         float normal_length = 0.1f;
//         p2.x = surfel.center.x() + surfel.normal.x() * normal_length;
//         p2.y = surfel.center.y() + surfel.normal.y() * normal_length;
//         p2.z = surfel.center.z() + surfel.normal.z() * normal_length;
        
//         normal_marker.points.push_back(p1);
//         normal_marker.points.push_back(p2);
//     }
    
//     markers.markers.push_back(normal_marker);
    
//     // Delete old markers
//     visualization_msgs::msg::Marker delete_marker;
//     delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
//     delete_marker.ns = "surfel_discs";
    
//     surfel_marker_pub_->publish(markers);
    
//     // Log map stats
//     auto stats = fuser_->map().get_stats();
//     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
//         "Map: %zu surfels, %zu voxels, avg conf: %.2f, avg pts: %.1f",
//         stats.valid_surfels, stats.voxels_occupied, 
//         stats.avg_confidence, stats.avg_point_count);    
// }

};