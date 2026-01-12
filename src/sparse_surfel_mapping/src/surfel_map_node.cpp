#include "sparse_surfel_mapping/surfel_map_node.hpp"

namespace sparse_surfel_map {

SurfelMapNode::SurfelMapNode(const rclcpp::NodeOptions& options) : Node("surfel_map_node", options) {
    // Declare and load parameters
    declare_parameters();

    // Create SurfelMap
    SurfelMapConfig config = load_configuration();
    surfel_map_ = std::make_unique<SurfelMap>(config);
    preproc_ = std::make_unique<ScanPreprocess>(config.preprocess_config);

    // TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Point Cloud sub
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, 
        rclcpp::SensorDataQoS(), 
        std::bind(&SurfelMapNode::pointcloud_callback, this, std::placeholders::_1)
    );
    
    // Surfel viz
    surfel_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("surfel_map/markers", 10);
    if (publish_rate_ > 0.0) {
        viz_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_rate_),
            std::bind(&SurfelMapNode::publish_visualization, this)
        );
    }

    cloud_in_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    RCLCPP_INFO(this->get_logger(), "SurfelMap node initialized");
    RCLCPP_INFO(this->get_logger(), "  Global Frame: %s", global_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Sensor Frame: %s", sensor_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Voxel Size: %.3f m", config.voxel_size);
}

void SurfelMapNode::declare_parameters() {
    // Topic
    this->declare_parameter("pointcloud_topic", "/x500/lidar_front/points_raw");

    // Frame
    this->declare_parameter("global_frame", "odom");
    this->declare_parameter("sensor_frame", "lidar_frame");

    // Sensor
    this->declare_parameter("width", 240);
    this->declare_parameter("height", 180);
    this->declare_parameter("hfov_deg", 106.0);
    this->declare_parameter("vfov_deg", 86.0);
    this->declare_parameter("min_range", 0.1);
    this->declare_parameter("max_range", 10.0);

    // Preprocess
    this->declare_parameter("enable_ground_filter", true);
    this->declare_parameter("ground_z_min", 0.2);
    this->declare_parameter("downsample_factor", 1);
    this->declare_parameter("normal_estimation_px_radius", 3);
    this->declare_parameter("orient_towards_sensor", true);
    this->declare_parameter("range_smooth_iters", 3);
    this->declare_parameter("depth_sigma_m", 0.05);
    this->declare_parameter("spatial_sigma_px", 1.0);
    this->declare_parameter("max_depth_jump_m", 0.1);

    // Map
    this->declare_parameter("voxel_size", 0.3);
    this->declare_parameter("initial_bucket_count", 10000);

    // Surfel
    this->declare_parameter("min_points_per_surfel", 10);
    this->declare_parameter("planarity_threshold", 0.01);
    this->declare_parameter("scale_threshold", 0.01);
    this->declare_parameter("degeneracy_threshold", 0.1);

    // Visualization
    this->declare_parameter("publish_rate", 2.0);
    this->declare_parameter("publish_visualization", true);
    this->declare_parameter("surfel_marker_scale", 0.05);

    // Debug
    this->declare_parameter("debug_out", true);

    // Load node parameters immediately
    pointcloud_topic_ = this->get_parameter("pointcloud_topic").as_string();
    global_frame_ = this->get_parameter("global_frame").as_string();
    sensor_frame_ = this->get_parameter("sensor_frame").as_string();
    publish_rate_ = this->get_parameter("publish_rate").as_double();
}

SurfelMapConfig SurfelMapNode::load_configuration() {
    SurfelMapConfig config;
    
    // sensor
    config.min_range = this->get_parameter("min_range").as_double();
    config.max_range = this->get_parameter("max_range").as_double();
    config.preprocess_config.min_range = config.min_range;
    config.preprocess_config.max_range = config.max_range;
    config.preprocess_config.width = this->get_parameter("width").as_int();
    config.preprocess_config.height = this->get_parameter("height").as_int();
    config.preprocess_config.hfov_deg = this->get_parameter("hfov_deg").as_double();
    config.preprocess_config.vfov_deg = this->get_parameter("vfov_deg").as_double();
    
    // preproc
    config.preprocess_config.enable_ground_filter = this->get_parameter("enable_ground_filter").as_bool();
    config.preprocess_config.ground_z_min = this->get_parameter("ground_z_min").as_double();
    config.preprocess_config.ds_factor = this->get_parameter("downsample_factor").as_int();
    config.preprocess_config.normal_est_px_radius = this->get_parameter("normal_estimation_px_radius").as_int();
    config.preprocess_config.orient_towards_sensor = this->get_parameter("orient_towards_sensor").as_bool();
    config.preprocess_config.range_smooth_iters = this->get_parameter("range_smooth_iters").as_int();
    config.preprocess_config.depth_sigma_m = this->get_parameter("depth_sigma_m").as_double();
    config.preprocess_config.spatial_sigma_px = this->get_parameter("spatial_sigma_px").as_double();
    config.preprocess_config.max_depth_jump_m = this->get_parameter("max_depth_jump_m").as_double();

    // map
    config.voxel_size = this->get_parameter("voxel_size").as_double();
    config.initial_bucket_count = static_cast<size_t>(this->get_parameter("initial_bucket_count").as_int());
    config.map_frame = global_frame_;
    config.debug_output = this->get_parameter("debug_out").as_bool();
    
    // surfel
    config.surfel_config.min_points_for_validity = static_cast<size_t>(this->get_parameter("min_points_per_surfel").as_int());
    config.surfel_config.planarity_threshold = this->get_parameter("planarity_threshold").as_double();
    config.surfel_config.scale_threshold = this->get_parameter("scale_threshold").as_double();
    config.surfel_config.degeneracy_threshold = this->get_parameter("degeneracy_threshold").as_double();


    return config;
}

void SurfelMapNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Capture transform 
    Eigen::Transform<float, 3, Eigen::Isometry> tf;
    if (!get_transform(msg->header.stamp, tf)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Failed to get transform from %s to %s",
            msg->header.frame_id.c_str(), global_frame_.c_str());
    }

    pcl::fromROSMsg(*msg, *cloud_in_);
    
    const auto ts = std::chrono::high_resolution_clock::now();

    std::vector<PointWithNormal> pns;
    preproc_->set_transform(tf);
    if (!preproc_->set_input_cloud(cloud_in_)) return; 
    preproc_->process();
    preproc_->get_points_with_normal(pns);    

    // lock map with map mutex - unique_lock blocks other writers AND readers of the shared object
    std::unique_lock lock(surfel_map_->mutex_);
    size_t integrated = surfel_map_->integrate_points(pns, tf);
    lock.unlock();

    const auto te = std::chrono::high_resolution_clock::now();
    double telaps = std::chrono::duration<double, std::milli>(te - ts).count();

    RCLCPP_INFO(this->get_logger(), "Total Time: %.3f ms", telaps);
    RCLCPP_INFO(this->get_logger(), "Integrated: %zu points", integrated);
}

bool SurfelMapNode::get_transform(const rclcpp::Time& stamp, Eigen::Transform<float, 3, Eigen::Isometry>& tf) {
    try {
        auto transform = tf_buffer_->lookupTransform(global_frame_, sensor_frame_, stamp, rclcpp::Duration::from_seconds(0.1));
        tf = tf2::transformToEigen(transform.transform).cast<float>();
        return true;
    }
    catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF Lookup Failed: %s", ex.what());
        return false;
    }
}

void SurfelMapNode::publish_visualization() {
    visualization_msgs::msg::MarkerArray marker_array;

    const auto surfels = surfel_map_->get_valid_surfels();
    if (surfels.empty()) return;

    auto viz_now = this->get_clock()->now();

    // delete previous
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = global_frame_;
    delete_marker.header.stamp = viz_now;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_marker.ns = "surfel_ellipses";
    marker_array.markers.push_back(delete_marker);

    // Surfel Ellipses
    visualization_msgs::msg::Marker surfel_ellipse;
    surfel_ellipse.header.frame_id = global_frame_;
    surfel_ellipse.header.stamp = viz_now;
    surfel_ellipse.ns = "surfel_ellipses";
    surfel_ellipse.type = visualization_msgs::msg::Marker::CYLINDER;
    surfel_ellipse.action = visualization_msgs::msg::Marker::ADD;

    int marker_id = 0;
    for (const auto& surfel_ref : surfels) {
        const Surfel& surfel = surfel_ref.get();

        surfel_ellipse.id = marker_id++;
        
        // position
        const Eigen::Vector3f& m = surfel.mean();
        surfel_ellipse.pose.position.x = m.x();
        surfel_ellipse.pose.position.y = m.y();
        surfel_ellipse.pose.position.z = m.z();

        const Eigen::Matrix3f& C = surfel.eigenvectors();
        const Eigen::Vector3f& ev1 = C.col(1);
        const Eigen::Vector3f& ev2 = C.col(2);
        Eigen::Matrix3f R;
        R.col(0) = ev1;
        R.col(1) = ev2;
        R.col(2) = surfel.normal();
        if (R.determinant() < 0) {
            R.col(1) = -R.col(1);
        }

        Eigen::Quaternionf q(R);
        q.normalize();

        // orientation
        surfel_ellipse.pose.orientation.x = q.x();
        surfel_ellipse.pose.orientation.y = q.y();
        surfel_ellipse.pose.orientation.z = q.z();
        surfel_ellipse.pose.orientation.w = q.w();

        // scale
        const Eigen::Vector3f& evals = surfel.eigenvalues();
        surfel_ellipse.scale.x = 2.0f * std::sqrt(std::max(evals(1), 1e-6f));
        surfel_ellipse.scale.y = 2.0f * std::sqrt(std::max(evals(2), 1e-6f));
        surfel_ellipse.scale.z = 0.005f;

        // color
        surfel_ellipse.color.r = (surfel.normal().x() + 1.0f) * 0.5f;
        surfel_ellipse.color.g = (surfel.normal().y() + 1.0f) * 0.5f;
        surfel_ellipse.color.b = (surfel.normal().z() + 1.0f) * 0.5f;
        surfel_ellipse.color.a = 0.8f;

        marker_array.markers.push_back(surfel_ellipse);
    }

    surfel_marker_pub_->publish(marker_array);
}

} // namespace