#ifndef SURFEL_MAPPING_NODE_
#define SURFEL_MAPPING_NODE_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "surfel_mapping/preprocess.hpp"
#include "surfel_mapping/surfel_fuse.hpp"

namespace surface_inspection_planning {

enum class SurfelColorMode {
    CONFIDENCE,
    POINT_COUNT,
    NORMAL_DIRECTION
};

enum class GraphNodeColorMode {
    IMPORTANCE,          // Color by node importance
    INSPECTION_STATE,    // Inspected vs uninspected
    CONFIDENCE,          // Underlying surfel confidence  
    DEGREE               // Number of connections
};

enum class GraphEdgeColorMode {
    COST,                // Edge traversal cost
    STRUCTURAL,          // Structural vs non-structural
    UNIFORM              // Single color
};

class SurfelMappingNode : public rclcpp::Node {
public:
    SurfelMappingNode();

private:
    void declare_parameters();
    void initialize_preprocessor();
    void initialize_fuser();

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    bool get_sensor_pose(const rclcpp::Time& stamp, Eigen::Isometry3f& pose);

    void publish_visualization();
    void publish_surfel_markers(visualization_msgs::msg::MarkerArray& markers);

    // Components
    std::unique_ptr<CloudPreprocess> preprocessor_;
    std::unique_ptr<SurfelFusion> fuser_;

    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr surfel_marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr processed_cloud_pub_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr viz_timer_;
};
    


}; // end namespace

#endif