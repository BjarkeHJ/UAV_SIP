#include "sparse_surfel_mapping/surfel_map_node.hpp"

namespace sparse_surfel_map {

SurfelMapNode::SurfelMapNode(const rclcpp::NodeOptions& options) : Node("surfel_map_node") {
    // Declare and load parameters
    declare_parameters();

    // Create SurfelMap
    SurfelMapConfig config = load_configuration();
    surfel_map_ = std::make_unique<SurfelMap>(config);

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

    RCLCPP_INFO(this->get_logger(), "SurfelMap node initialized");
    RCLCPP_INFO(this->get_logger(), "  Map Frame: %s", map_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Sensor Frame: %s", sensor_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Voxel Size: %.3f m", config.voxel_size);
}

void SurfelMapNode::declare_parameters() {
    // Topic
    this->declare_parameter("pointcloud_topic", "/x500/lidar_front/points_raw");

    // Frame
    this->declare_parameter("map_frame", "odom");
    this->declare_parameter("sensor_frame", "lidar_frame");

    // Sensor
    this->declare_parameter("min_range", 0.1);
    this->declare_parameter("max_range", 10.0);

    // Map
    this->declare_parameter("voxel_size", 0.1);
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
    map_frame_ = this->get_parameter("map_frame").as_string();
    sensor_frame_ = this->get_parameter("sensor_frame").as_string();
    publish_rate_ = this->get_parameter("publish_rate").as_double();
    publish_visualization_ = this->get_parameter("publish_visualization").as_bool();
    surfel_marker_scale_ = this->get_parameter("surfel_marker_scale").as_double();
}

SurfelMapConfig SurfelMapNode::load_configuration() {
    SurfelMapConfig config;

    config.voxel_size = this->get_parameter("voxel_size").as_double();
    config.min_range = this->get_parameter("min_range").as_double();
    config.max_range = this->get_parameter("max_range").as_double();
    config.initial_bucket_count = static_cast<size_t>(this->get_parameter("initial_bucket_count").as_int());
    config.map_frame = map_frame_;
    config.debug_output = this->get_parameter("deub_out").as_bool();
    
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
            msg->header.frame_id.c_str(), map_frame_.c_str());
    }

    std::vector<PointWithNormal> pns;
    


}


} // namespace