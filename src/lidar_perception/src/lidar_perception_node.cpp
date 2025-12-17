#include "lidar_perception/lidar_perception_node.hpp"

LidarPerceptionNode::LidarPerceptionNode() : Node("LidarPerceptionNode") {
    cloud_in_topic_ = this->declare_parameter<std::string>("pointcloud_in", "/x500/lidar_front/points_raw");
    global_frame_ = this->declare_parameter<std::string>("global_frame", "odom");
    drone_frame_ = this->declare_parameter<std::string>("drone_frame", "base_link");
    lidar_frame_ = this->declare_parameter<std::string>("lidar_frame", "lidar_frame");

    pp_params_.width = this->declare_parameter<int>("tof_px_W", 240);
    pp_params_.height = this->declare_parameter<int>("tof_px_H", 180);
    pp_params_.hfov_deg = this->declare_parameter<double>("tof_fov_h", 106.0f);
    pp_params_.vfov_deg = this->declare_parameter<double>("tof_fov_v", 86.0f);
    pp_params_.ds_factor = this->declare_parameter<double>("cloud_ds_factor", 2.0f);
    pp_params_.min_range = this->declare_parameter<double>("tof_min_range", 0.1f);
    pp_params_.max_range = this->declare_parameter<double>("tof_max_range", 10.0f);
    pp_params_.keep_closest = this->declare_parameter<bool>("ds_keep_closest", true);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    auto qos_profile = rclcpp::SensorDataQoS();
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_in_topic_,
        qos_profile,
        std::bind(&LidarPerceptionNode::pointcloud_callback, this, std::placeholders::_1)
    );

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/test_pcd", 10);
    nrm_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/test_nrms", 10);

    latest_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    latest_normals_ = std::make_shared<pcl::PointCloud<pcl::Normal>>();
    latest_pts_w_nrms_ = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    cloud_buff_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    tree_ = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    
    preproc_ = std::make_unique<CloudPreprocess>(pp_params_);

    SurfelParams smp_;
    smapper_ = std::make_shared<SurfelMapping>(smp_);

    SurfelVizConfig sfv_cfg;
    sfv_cfg.frame_id = global_frame_;
    sfv_cfg.topic = "/surfels_debug";
    sfv_cfg.k_sigma = 2.0f;
    sfv_cfg.lifetime_sec = 0.3;
    sfv_cfg.stride = 1;
    surfel_viz_ = std::make_unique<SurfelDebugViz>(*this, sfv_cfg);

    RCLCPP_INFO(this->get_logger(), "LidarPerceptionNode Started...");
}

void LidarPerceptionNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud...");
        return;
    }

    // TODO: Have this callback only grab the latest pointcloud and store it
    // Then have a worker thread do the heavy lifting so this executor is not stalled...
    // have separate timer for planning (if planning is heavy -> multithread executor)

    
    // Capture latest TF
    try {
        latest_tf_ = tf_buffer_->lookupTransform(global_frame_, lidar_frame_, msg->header.stamp);
        latest_pos_.x() = latest_tf_.transform.translation.x;
        latest_pos_.y() = latest_tf_.transform.translation.y;
        latest_pos_.z() = latest_tf_.transform.translation.z;
        latest_q_.w() = latest_tf_.transform.rotation.w;
        latest_q_.x() = latest_tf_.transform.rotation.x;
        latest_q_.y() = latest_tf_.transform.rotation.y;
        latest_q_.z() = latest_tf_.transform.rotation.z;
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "Transform Lookup Failed: %s", ex.what());
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    filtering(msg);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> te = t2 - t1;
    // std::cout << "Filtering + NE duration: " <<  te.count() << "s." << std::endl;
    std::cout << latest_cloud_->points.size() << std::endl;

    sensor_msgs::msg::PointCloud2 msg_clean;
    pcl::toROSMsg(*latest_cloud_, msg_clean);
    msg_clean.header.frame_id = global_frame_;
    msg_clean.header.stamp = msg->header.stamp;
    cloud_pub_->publish(msg_clean);

    // normal_estimation();

    smapper_->set_local_frame(latest_pts_w_nrms_);
    smapper_->run();
    std::vector<Surfel2D>& s2ds = smapper_->get_local_surfels();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tee = t3 - t2;
    std::cout << "Surfel Extract Duration: " <<  tee.count() << "s." << std::endl;

    // TODO: Batch accumulator over N scans using transform information? (Flag to run/wait preprocess on batch)
    surfel_viz_->publish(s2ds);
    publishNormals(latest_pts_w_nrms_, global_frame_, 0.1);

}

void LidarPerceptionNode::filtering(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!latest_cloud_ || msg->data.empty()) return;

    pcl::fromROSMsg(*msg, *cloud_buff_);

    // Custom preprocessing
    preproc_->set_world_transform(latest_pos_, latest_q_);
    auto t0 = std::chrono::high_resolution_clock::now();
    preproc_->set_input_cloud(cloud_buff_);
    auto t1 = std::chrono::high_resolution_clock::now();
    preproc_->normal_estimation();
    auto t2 = std::chrono::high_resolution_clock::now();
    preproc_->downsample();
    auto t3 = std::chrono::high_resolution_clock::now();
    preproc_->transform_output_to_world();
    auto t4 = std::chrono::high_resolution_clock::now();
    preproc_->get_points(latest_cloud_);
    preproc_->get_points_with_normals(latest_pts_w_nrms_);
    
    std::chrono::duration<double> t01 = t1 - t0;
    std::chrono::duration<double> t12 = t2 - t1;
    std::chrono::duration<double> t23 = t3 - t2;
    std::chrono::duration<double> t34 = t4 - t3;
    std::cout << "Set PointCloud: " << t01.count() << "s." << std::endl;
    std::cout << "Normal Estimation: " << t12.count() << "s." << std::endl;
    std::cout << "Downsample: " << t23.count() << "s." << std::endl;
    std::cout << "Transform: " << t34.count() << "s." << std::endl;

    // Statistical outlier removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(latest_cloud_);
    sor.setMeanK(10);
    sor.setStddevMulThresh (0.5);
    sor.filter(*cloud_buff_);
    latest_cloud_ = cloud_buff_;
}

void LidarPerceptionNode::normal_estimation() {
    if (!latest_cloud_ || latest_cloud_->points.empty()) return;

    // Surface normal estimation 
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(latest_cloud_);
    ne.setSearchMethod(tree_);
    ne.setKSearch(5);
    ne.setViewPoint(latest_pos_.x(), latest_pos_.y(), latest_pos_.z());
    ne.compute(*latest_normals_);

    pcl::concatenateFields(*latest_cloud_, *latest_normals_, *latest_pts_w_nrms_);

    publishNormals(latest_pts_w_nrms_, global_frame_, 0.1);
}


void LidarPerceptionNode::publishNormals(pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_nrms, std::string &frame_id, double scale) {
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.reserve(cloud_w_nrms->size());

    for (size_t i=0; i<cloud_w_nrms->size(); ++i) {
        const auto& pt_nrm = cloud_w_nrms->points[i];
        visualization_msgs::msg::Marker arrow;
        arrow.header.frame_id = frame_id;
        arrow.header.stamp = this->now();
        arrow.ns = "normals";
        arrow.id = i;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point start, end;
        start.x = pt_nrm.x;
        start.y = pt_nrm.y;
        start.z = pt_nrm.z;
        end.x = pt_nrm.x + pt_nrm.normal_x * scale;
        end.y = pt_nrm.y + pt_nrm.normal_y * scale;
        end.z = pt_nrm.z + pt_nrm.normal_z * scale;

        arrow.points.push_back(start);
        arrow.points.push_back(end);

        arrow.scale.x = 0.005;  // shaft diameter
        arrow.scale.y = 0.01;   // head diameter
        arrow.scale.z = 0.01;   // head length

        arrow.color.r = 0.0f;
        arrow.color.g = 0.0f;
        arrow.color.b = 1.0f;
        arrow.color.a = 1.0f;

        marker_array.markers.push_back(arrow);
    }
    nrm_pub_->publish(marker_array);
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarPerceptionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}