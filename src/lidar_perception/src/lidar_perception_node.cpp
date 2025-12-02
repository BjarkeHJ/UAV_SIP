#include "lidar_perception/lidar_perception_node.hpp"

LidarPerceptionNode::LidarPerceptionNode() : Node("LidarPerceptionNode") {
    cloud_in_topic_ = this->declare_parameter<std::string>("pointcloud_in", "/lidar_front/points_raw");
    global_frame_ = this->declare_parameter<std::string>("global_frame", "odom");
    drone_frame_ = this->declare_parameter<std::string>("drone_frame", "base_link");
    lidar_frame_ = this->declare_parameter<std::string>("lidar_frame", "lidar_sensor_link");

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


    latest_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    latest_normals_.reset(new pcl::PointCloud<pcl::Normal>);

    RCLCPP_INFO(this->get_logger(), "LidarPerceptionNode Started...");
}

void LidarPerceptionNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud...");
        return;
    }

    // Capture transform tree
    try {
        latest_tf_ = tf_buffer_->lookupTransform(global_frame_, lidar_frame_, msg->header.stamp);
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "Transform Lookup Failed: %s", ex.what());
    }

    sensor_msgs::msg::PointCloud2 msg_tf;
    tf2::doTransform(*msg, msg_tf, latest_tf_);
    pcl::PointCloud<pcl::PointXYZ> tmp_cloud;
    pcl::fromROSMsg(msg_tf, tmp_cloud);

    // Remove inf/NaN
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(tmp_cloud, *latest_cloud_, indices);
    
    // Republish cloud... (temp)
    sensor_msgs::msg::PointCloud2 msg_clean;
    pcl::toROSMsg(*latest_cloud_, msg_clean);
    msg_clean.header.frame_id = global_frame_;
    msg_clean.header.stamp = msg->header.stamp;
    cloud_pub_->publish(msg_clean);

    pointcloud_preprocess();

    // TODO: Batch accumulator over N scans using transform information? (Flag to run/wait preprocess on batch)
}

void LidarPerceptionNode::pointcloud_preprocess() {
    if (!latest_cloud_ || latest_cloud_->points.empty()) return;

    // GND filtering 

    // Denoise

    // Surface normal estimation 
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setInputCloud(latest_cloud_);
    ne.setSearchMethod(tree);
    ne.setKSearch(10);
    ne.compute(*latest_normals_);

    // Down sample
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*latest_cloud_, *latest_normals_, *cloud_w_normals);

    pcl::VoxelGrid<pcl::PointNormal> vgf;
    vgf.setInputCloud(cloud_w_normals);
    vgf.setLeafSize(0.1, 0.1, 0.1);
    vgf.filter(*cloud_w_normals);

    publishNormals(cloud_w_normals, global_frame_, 0.1);
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