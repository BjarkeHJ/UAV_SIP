#include "lidar_perception/surfel_visualizer.hpp"
#include <cmath>

namespace surface_inspection_planner {

SurfelVisualizer::SurfelVisualizer(rclcpp::Node* node) : node_(node) {

}

visualization_msgs::msg::MarkerArray SurfelVisualizer::create_visualization(
    const std::vector<Surfel2D>& surfels,
    const std::string& frame_id,
    VisualizationMode mode,
    const rclcpp::Time& timestamp) {
    
    visualization_msgs::msg::MarkerArray marker_array;

    if (surfels.empty()) {
        return marker_array;
    }
    
    rclcpp::Time stamp = (timestamp.nanoseconds() == 0) ? node_->now() : timestamp;
    
    marker_id_offset_ = 0;
    
    switch (mode) {
        case VisualizationMode::ELLIPSE_DISKS:
            create_ellipse_disk_markers(surfels, frame_id, stamp, marker_array);
            break;
            
        case VisualizationMode::NORMAL_ARROWS:
            create_normal_arrow_markers(surfels, frame_id, stamp, marker_array);
            break;
            
        case VisualizationMode::CONFIDENCE_SPHERES:
            create_confidence_sphere_markers(surfels, frame_id, stamp, marker_array);
            break;
            
        case VisualizationMode::WIREFRAME_ELLIPSES:
            create_wireframe_ellipse_markers(surfels, frame_id, stamp, marker_array);
            break;
            
        case VisualizationMode::COMBINED:
            // Create all visualization types
            create_ellipse_disk_markers(surfels, frame_id, stamp, marker_array);
            marker_id_offset_ += surfels.size();
            create_normal_arrow_markers(surfels, frame_id, stamp, marker_array);
            marker_id_offset_ += surfels.size();
            create_wireframe_ellipse_markers(surfels, frame_id, stamp, marker_array);
            break;
    }
    
    return marker_array;
}

void SurfelVisualizer::create_ellipse_disk_markers(
    const std::vector<Surfel2D>& surfels,
    const std::string& frame_id,
    const rclcpp::Time& timestamp,
    visualization_msgs::msg::MarkerArray& marker_array) {
    
    for (size_t i = 0; i < surfels.size(); ++i) {
        visualization_msgs::msg::Marker disk = 
            create_ellipse_disk_mesh(surfels[i], frame_id, timestamp, marker_id_offset_ + i);
        marker_array.markers.push_back(disk);
    }
}

visualization_msgs::msg::Marker SurfelVisualizer::create_ellipse_disk_mesh(
    const Surfel2D& surfel,
    const std::string& frame_id,
    const rclcpp::Time& timestamp,
    int id) {
    
    visualization_msgs::msg::Marker marker;
    marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    marker.header.frame_id = frame_id;
    marker.header.stamp = timestamp;
    marker.ns = "surfel_disks";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Get ellipse parameters from eigenvalues
    float semi_major = sigma_multiplier_ * std::sqrt(surfel.eigen_values(0));
    float semi_minor = sigma_multiplier_ * std::sqrt(surfel.eigen_values(1));
    
    // Generate 2D ellipse points
    std::vector<Eigen::Vector2f> points_2d = generate_ellipse_points(
        semi_major, semi_minor, ellipse_resolution_
    );
    
    // Transform to 3D
    std::vector<Eigen::Vector3f> points_3d = transform_2d_to_3d(points_2d, surfel);
    
    // Create triangle fan from center
    Eigen::Vector3f center = surfel.center;
    
    for (size_t i = 0; i < points_3d.size(); ++i) {
        size_t next = (i + 1) % points_3d.size();
        
        // Triangle: center -> point[i] -> point[i+1]
        geometry_msgs::msg::Point p0, p1, p2;
        
        p0.x = center.x();
        p0.y = center.y();
        p0.z = center.z();
        
        p1.x = points_3d[i].x();
        p1.y = points_3d[i].y();
        p1.z = points_3d[i].z();
        
        p2.x = points_3d[next].x();
        p2.y = points_3d[next].y();
        p2.z = points_3d[next].z();
        
        marker.points.push_back(p0);
        marker.points.push_back(p1);
        marker.points.push_back(p2);

        marker.points.push_back(p0);
        marker.points.push_back(p2);
        marker.points.push_back(p1);
    }
    
    // Set scale (not used for TRIANGLE_LIST but required)
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    
    // Color based on confidence
    marker.color = get_confidence_color(surfel.confidence);
    marker.color.a = alpha_;
    
    return marker;
}

void SurfelVisualizer::create_normal_arrow_markers(
    const std::vector<Surfel2D>& surfels,
    const std::string& frame_id,
    const rclcpp::Time& timestamp,
    visualization_msgs::msg::MarkerArray& marker_array) {
    
    for (size_t i = 0; i < surfels.size(); ++i) {
        const auto& surfel = surfels[i];
        
        visualization_msgs::msg::Marker arrow;
        arrow.lifetime = rclcpp::Duration::from_seconds(0.2);
        arrow.header.frame_id = frame_id;
        arrow.header.stamp = timestamp;
        arrow.ns = "surfel_normals";
        arrow.id = marker_id_offset_ + i;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;
        
        geometry_msgs::msg::Point start, end;
        start.x = surfel.center.x();
        start.y = surfel.center.y();
        start.z = surfel.center.z();
        
        end.x = surfel.center.x() + surfel.normal.x() * normal_scale_;
        end.y = surfel.center.y() + surfel.normal.y() * normal_scale_;
        end.z = surfel.center.z() + surfel.normal.z() * normal_scale_;
        
        arrow.points.push_back(start);
        arrow.points.push_back(end);
        
        arrow.scale.x = 0.01;  // Shaft diameter
        arrow.scale.y = 0.02;  // Head diameter
        arrow.scale.z = 0.02;  // Head length
        
        // Cyan color for normals
        arrow.color.r = 0.0f;
        arrow.color.g = 0.8f;
        arrow.color.b = 1.0f;
        arrow.color.a = 1.0f;
        
        marker_array.markers.push_back(arrow);
    }
}

void SurfelVisualizer::create_confidence_sphere_markers(
    const std::vector<Surfel2D>& surfels,
    const std::string& frame_id,
    const rclcpp::Time& timestamp,
    visualization_msgs::msg::MarkerArray& marker_array) {
    
    for (size_t i = 0; i < surfels.size(); ++i) {
        const auto& surfel = surfels[i];
        
        visualization_msgs::msg::Marker sphere;
        sphere.lifetime = rclcpp::Duration::from_seconds(0.2);
        sphere.header.frame_id = frame_id;
        sphere.header.stamp = timestamp;
        sphere.ns = "surfel_centers";
        sphere.id = marker_id_offset_ + i;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.action = visualization_msgs::msg::Marker::ADD;
        
        sphere.pose.position.x = surfel.center.x();
        sphere.pose.position.y = surfel.center.y();
        sphere.pose.position.z = surfel.center.z();
        sphere.pose.orientation.w = 1.0;
        
        // Size based on radius
        float sphere_size = std::max(0.02f, surfel.radius * 0.2f);
        sphere.scale.x = sphere_size;
        sphere.scale.y = sphere_size;
        sphere.scale.z = sphere_size;
        
        // Color by confidence
        sphere.color = get_confidence_color(surfel.confidence);
        sphere.color.a = 1.0f;
        
        marker_array.markers.push_back(sphere);
    }
}

void SurfelVisualizer::create_wireframe_ellipse_markers(
    const std::vector<Surfel2D>& surfels,
    const std::string& frame_id,
    const rclcpp::Time& timestamp,
    visualization_msgs::msg::MarkerArray& marker_array) {
    
    for (size_t i = 0; i < surfels.size(); ++i) {
        const auto& surfel = surfels[i];
        
        visualization_msgs::msg::Marker line_strip;
        line_strip.lifetime = rclcpp::Duration::from_seconds(0.2);
        line_strip.header.frame_id = frame_id;
        line_strip.header.stamp = timestamp;
        line_strip.ns = "surfel_wireframes";
        line_strip.id = marker_id_offset_ + i;
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_strip.action = visualization_msgs::msg::Marker::ADD;
        
        // Get ellipse parameters
        float semi_major = sigma_multiplier_ * std::sqrt(surfel.eigen_values(0));
        float semi_minor = sigma_multiplier_ * std::sqrt(surfel.eigen_values(1));
        
        // Generate ellipse points
        std::vector<Eigen::Vector2f> points_2d = generate_ellipse_points(
            semi_major, semi_minor, ellipse_resolution_
        );
        std::vector<Eigen::Vector3f> points_3d = transform_2d_to_3d(points_2d, surfel);
        
        // Add points to line strip (close the loop)
        for (const auto& pt : points_3d) {
            geometry_msgs::msg::Point p;
            p.x = pt.x();
            p.y = pt.y();
            p.z = pt.z();
            line_strip.points.push_back(p);
        }
        
        // Close the ellipse
        if (!points_3d.empty()) {
            geometry_msgs::msg::Point p;
            p.x = points_3d[0].x();
            p.y = points_3d[0].y();
            p.z = points_3d[0].z();
            line_strip.points.push_back(p);
        }
        
        line_strip.scale.x = 0.005;  // Line width
        
        // White wireframe
        line_strip.color.r = 1.0f;
        line_strip.color.g = 1.0f;
        line_strip.color.b = 1.0f;
        line_strip.color.a = 0.8f;
        
        marker_array.markers.push_back(line_strip);
    }
}

std::vector<Eigen::Vector2f> SurfelVisualizer::generate_ellipse_points(
    float semi_major, 
    float semi_minor, 
    int num_points) {
    
    std::vector<Eigen::Vector2f> points;
    points.reserve(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        float theta = 2.0f * M_PI * i / num_points;
        
        // Parametric ellipse equation
        float x = semi_major * std::cos(theta);
        float y = semi_minor * std::sin(theta);
        
        points.push_back(Eigen::Vector2f(x, y));
    }
    
    return points;
}

std::vector<Eigen::Vector3f> SurfelVisualizer::transform_2d_to_3d(
    const std::vector<Eigen::Vector2f>& points_2d,
    const Surfel2D& surfel) {
    
    std::vector<Eigen::Vector3f> points_3d;
    points_3d.reserve(points_2d.size());
    
    // Rotate 2D points by eigenvectors to align with principal axes
    Eigen::Matrix2f rotation = surfel.eigen_vectors;
    
    for (const auto& pt_2d : points_2d) {
        // Rotate in 2D tangent plane
        Eigen::Vector2f rotated = rotation * pt_2d;
        
        // Transform to 3D world coordinates
        Eigen::Vector3f pt_3d = surfel.center + 
                                rotated.x() * surfel.u_axis + 
                                rotated.y() * surfel.v_axis;
        
        points_3d.push_back(pt_3d);
    }
    
    return points_3d;
}

std_msgs::msg::ColorRGBA SurfelVisualizer::get_confidence_color(float confidence) {
    std_msgs::msg::ColorRGBA color;
    
    // Clamp confidence to [0, 1]
    confidence = std::max(0.0f, std::min(1.0f, confidence));
    
    // Color gradient: Red (low) -> Yellow -> Green (high)
    if (confidence < 0.5f) {
        // Red to Yellow
        float t = confidence * 2.0f;
        color.r = 1.0f;
        color.g = t;
        color.b = 0.0f;
    } else {
        // Yellow to Green
        float t = (confidence - 0.5f) * 2.0f;
        color.r = 1.0f - t;
        color.g = 1.0f;
        color.b = 0.0f;
    }
    
    color.a = 1.0f;
    return color;
}

};