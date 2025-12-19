#ifndef SURFEL_VISUALIZER_HPP
#define SURFEL_VISUALIZER_HPP

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <Eigen/Dense>
#include <vector>

#include "lidar_perception/surfel2d.hpp"

namespace surface_inspection_planner {

class SurfelVisualizer {
public:
    enum class VisualizationMode {
        ELLIPSE_DISKS,
        NORMAL_ARROWS,
        CONFIDENCE_SPHERES,
        WIREFRAME_ELLIPSES,
        COMBINED
    };

    explicit SurfelVisualizer(rclcpp::Node* node);

    visualization_msgs::msg::MarkerArray create_visualization(const std::vector<Surfel2D>& surfels, const std::string& frame_id, VisualizationMode mode = VisualizationMode::COMBINED, const rclcpp::Time& timestamp = rclcpp::Time(0));

    void set_normal_scale(float scale) { normal_scale_ = scale; }
    void set_ellipse_resolution(int res) { ellipse_resolution_ = res; }
    void set_sigma_multiplier(float sigma) { sigma_multiplier_ = sigma; } 
    void set_alpha(float alpha) { alpha_ = alpha; }

private:

    void create_ellipse_disk_markers(const std::vector<Surfel2D>& surfels, const std::string& frame_id, const rclcpp::Time& timestamp, visualization_msgs::msg::MarkerArray& marker_array);
    void create_normal_arrow_markers(const std::vector<Surfel2D>& surfels, const std::string& frame_id, const rclcpp::Time& timestamp, visualization_msgs::msg::MarkerArray& marker_array);
    void create_confidence_sphere_markers(const std::vector<Surfel2D>& surfels, const std::string& frame_id, const rclcpp::Time& timestamp, visualization_msgs::msg::MarkerArray& marker_array);
    void create_wireframe_ellipse_markers(const std::vector<Surfel2D>& surfels, const std::string& frame_id, const rclcpp::Time& timestamp, visualization_msgs::msg::MarkerArray& marker_array);
    std::vector<Eigen::Vector2f> generate_ellipse_points(float semi_major, float semi_minor, int num_points);
    std::vector<Eigen::Vector3f> transform_2d_to_3d(const std::vector<Eigen::Vector2f>& points_2d, const Surfel2D& surfel);
    std_msgs::msg::ColorRGBA get_confidence_color(float confidence);
    visualization_msgs::msg::Marker create_ellipse_disk_mesh(const Surfel2D& surfel, const std::string& frame_id, const rclcpp::Time& timestamp, int id);

    rclcpp::Node* node_;

    float normal_scale_ = 0.15f;
    int ellipse_resolution_ = 32;
    float sigma_multiplier_ = 2.0f;
    float alpha_ = 0.7f;

    int marker_id_offset_ = 0;
};

};

#endif