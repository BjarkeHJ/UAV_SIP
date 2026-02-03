#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class TrajTracker : public rclcpp::Node {
public:
    TrajTracker() : Node("trajectory_tracker")
    {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>("/trajectory/full_trajectory", 10);
        traj_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / 10.0),
            std::bind(&TrajTracker::timer_callback, this)
        );

        traj_.header.frame_id = "odom";

        RCLCPP_INFO(this->get_logger(), "Trajectory Tracker Node Initialized...");
    } 

private:
    void timer_callback();
    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    // TIMER
    rclcpp::TimerBase::SharedPtr traj_timer_;
    // PUBLISHER
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    // Traj path
    nav_msgs::msg::Path traj_;
    double traj_length_{0.0};
};

void TrajTracker::timer_callback() {
    geometry_msgs::msg::TransformStamped tf;

    try {
        tf = tf_buffer_->lookupTransform("odom", "lidar_frame", tf2::TimePointZero);
    }
    catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "TF Unavailable: %s", ex.what());
        return;
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "odom";
    pose.header.stamp = this->get_clock()->now();

    pose.pose.position.x = tf.transform.translation.x;
    pose.pose.position.y = tf.transform.translation.y;
    pose.pose.position.z = tf.transform.translation.z;
    pose.pose.orientation = tf.transform.rotation;

    // std::cout << "Drone Position: " << pose.pose.position.x << " " << pose.pose.position.y << " " << pose.pose.position.z << std::endl;

    if (!traj_.poses.empty()) {
        const auto& last = traj_.poses.back().pose.position;
        double dx = pose.pose.position.x - last.x;
        double dy = pose.pose.position.y - last.y;
        double dz = pose.pose.position.z - last.z;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < 0.05) return;
        traj_length_ += dist;
    }



    traj_.poses.push_back(pose);
    traj_.header.stamp = pose.header.stamp;
    
    RCLCPP_INFO(this->get_logger(), "Published path - Size: %lu - Length: %.3f", traj_.poses.size(), traj_length_);
    traj_pub_->publish(traj_);
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajTracker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}