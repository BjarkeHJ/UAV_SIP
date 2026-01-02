#include "surfel_mapping/surfel_mapping_node.hpp"

using namespace surface_inspection_planning;

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<SurfelMappingNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}