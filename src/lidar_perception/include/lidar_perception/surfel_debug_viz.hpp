#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

struct Surfel
{
  Eigen::Vector3f centroid;   // world/frame coordinates
  Eigen::Vector3f normal;     // unit vector preferred
  Eigen::Vector3f t1;         // unit tangent
  Eigen::Vector3f t2;         // unit tangent
  Eigen::Matrix2f cov2d;      // covariance in (t1,t2) coords of projected points
};

struct SurfelVizConfig
{
  std::string frame_id = "map";
  std::string topic = "/surfels_debug";

  // sizes
  float centroid_radius = 0.05f;
  float normal_len = 0.20f;
  float tangent_len = 0.15f;

  // arrow thickness
  float arrow_shaft_d = 0.01f;
  float arrow_head_d  = 0.02f;
  float arrow_head_l  = 0.04f;

  // covariance ellipse
  int   ellipse_segments = 32;
  float ellipse_line_width = 0.005f;

  // How “big” ellipse is:
  // k_sigma = 2 means 2σ along principal axes (good for debugging).
  // If you want a confidence ellipse, you can instead use chi2_2d (see comment in cpp).
  float k_sigma = 2.0f;

  // lifetime: 0 = forever. For live debugging, 0.2–0.5s is nice.
  double lifetime_sec = 0.3;

  // If you have tons of surfels, visualize every Nth
  int stride = 1;

  // namespace toggles
  bool show_centroids = true;
  bool show_normals   = true;
  bool show_tangents  = true;
  bool show_cov       = true;
};

class SurfelDebugViz
{
public:
  SurfelDebugViz(rclcpp::Node& node, SurfelVizConfig cfg = {})
  : node_(node), cfg_(std::move(cfg))
  {
    pub_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(cfg_.topic, 10);
  }

  void publish(const std::vector<Surfel>& surfels)
  {
    visualization_msgs::msg::MarkerArray ma;
    const auto stamp = node_.now();

    // IDs must be stable per namespace within a MarkerArray update.
    int id_cent = 0, id_n = 0, id_t1 = 0, id_t2 = 0, id_cov = 0;

    const int stride = std::max(1, cfg_.stride);

    for (size_t i = 0; i < surfels.size(); i += stride)
    {
      const auto& s = surfels[i];

      if (cfg_.show_centroids) ma.markers.push_back(make_centroid_marker(s, stamp, id_cent++));
      if (cfg_.show_normals)   ma.markers.push_back(make_arrow_marker(s, stamp, id_n++,  "normals",  s.normal, cfg_.normal_len, 0.f, 1.f, 0.f));
      if (cfg_.show_tangents) {
        ma.markers.push_back(make_arrow_marker(s, stamp, id_t1++, "t1",       s.t1,     cfg_.tangent_len, 1.f, 0.f, 0.f));
        ma.markers.push_back(make_arrow_marker(s, stamp, id_t2++, "t2",       s.t2,     cfg_.tangent_len, 0.f, 0.f, 1.f));
      }
      if (cfg_.show_cov)       ma.markers.push_back(make_cov_ellipse_marker(s, stamp, id_cov++));
    }

    pub_->publish(ma);
  }

private:
  rclcpp::Node& node_;
  SurfelVizConfig cfg_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;

  visualization_msgs::msg::Marker make_centroid_marker(const Surfel& s,
                                                       const rclcpp::Time& stamp,
                                                       int id) const
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = cfg_.frame_id;
    m.header.stamp = stamp;
    m.ns = "centroids";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.pose.position.x = s.centroid.x();
    m.pose.position.y = s.centroid.y();
    m.pose.position.z = s.centroid.z();
    m.pose.orientation.w = 1.0;

    m.scale.x = cfg_.centroid_radius * 2.f;
    m.scale.y = cfg_.centroid_radius * 2.f;
    m.scale.z = cfg_.centroid_radius * 2.f;

    m.color.a = 1.0;
    m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; // white

    if (cfg_.lifetime_sec > 0.0) m.lifetime = rclcpp::Duration::from_seconds(cfg_.lifetime_sec);
    return m;
  }

  visualization_msgs::msg::Marker make_arrow_marker(const Surfel& s,
                                                    const rclcpp::Time& stamp,
                                                    int id,
                                                    const std::string& ns,
                                                    const Eigen::Vector3f& dir,
                                                    float len,
                                                    float r, float g, float b) const
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = cfg_.frame_id;
    m.header.stamp = stamp;
    m.ns = ns;
    m.id = id;
    m.type = visualization_msgs::msg::Marker::ARROW;
    m.action = visualization_msgs::msg::Marker::ADD;

    geometry_msgs::msg::Point p0, p1;
    p0.x = s.centroid.x(); p0.y = s.centroid.y(); p0.z = s.centroid.z();

    Eigen::Vector3f d = dir;
    const float n = d.norm();
    if (n > 1e-6f) d /= n; // normalize if needed

    const Eigen::Vector3f tip = s.centroid + d * len;
    p1.x = tip.x(); p1.y = tip.y(); p1.z = tip.z();

    m.points = {p0, p1};

    // Arrow scale semantics: shaft diameter, head diameter, head length
    m.scale.x = cfg_.arrow_shaft_d;
    m.scale.y = cfg_.arrow_head_d;
    m.scale.z = cfg_.arrow_head_l;

    m.color.a = 1.0;
    m.color.r = r; m.color.g = g; m.color.b = b;

    if (cfg_.lifetime_sec > 0.0) m.lifetime = rclcpp::Duration::from_seconds(cfg_.lifetime_sec);
    return m;
  }

  visualization_msgs::msg::Marker make_cov_ellipse_marker(const Surfel& s,
                                                          const rclcpp::Time& stamp,
                                                          int id) const
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = cfg_.frame_id;
    m.header.stamp = stamp;
    m.ns = "cov";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.scale.x = cfg_.ellipse_line_width;
    m.color.a = 1.0;
    m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; // yellow

    if (cfg_.lifetime_sec > 0.0) m.lifetime = rclcpp::Duration::from_seconds(cfg_.lifetime_sec);

    // Ensure t1,t2 are usable basis vectors
    Eigen::Vector3f t1 = s.t1;
    Eigen::Vector3f t2 = s.t2;
    if (t1.norm() > 1e-6f) t1.normalize();
    if (t2.norm() > 1e-6f) t2.normalize();

    // Symmetrize cov just in case of numerical noise
    Eigen::Matrix2f C = 0.5f * (s.cov2d + s.cov2d.transpose());

    // Eigen decomposition (C is PSD ideally)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
    Eigen::Vector2f eval = es.eigenvalues();     // ascending
    Eigen::Matrix2f evec = es.eigenvectors();    // columns are eigenvectors

    // Clamp negative tiny eigenvalues
    eval.x() = std::max(0.0f, eval.x());
    eval.y() = std::max(0.0f, eval.y());

    // Scale axes:
    // Debug-friendly: k_sigma * sqrt(lambda)
    // If you want a 95% confidence ellipse in 2D, multiply by sqrt(chi2),
    // where chi2 (df=2, p=0.95) ≈ 5.991, so factor would be sqrt(5.991) ≈ 2.448.
    const float k = cfg_.k_sigma;
    const float a = k * std::sqrt(eval.y());  // major axis (largest eigenvalue)
    const float b = k * std::sqrt(eval.x());  // minor axis

    const int N = std::max(8, cfg_.ellipse_segments);
    m.points.reserve(N + 1);

    for (int i = 0; i <= N; ++i)
    {
      const float th = (2.0f * float(M_PI)) * (float(i) / float(N));
      Eigen::Vector2f u_local(a * std::cos(th), b * std::sin(th));

      // Rotate by eigenvectors in 2D
      Eigen::Vector2f u = evec * u_local; // [u1; u2] in (t1,t2) coords

      // Map to 3D point in world/frame coordinates
      Eigen::Vector3f p3 = s.centroid + u.x() * t1 + u.y() * t2;

      geometry_msgs::msg::Point p;
      p.x = p3.x(); p.y = p3.y(); p.z = p3.z();
      m.points.push_back(p);
    }

    return m;
  }
};
