#include "sparse_surfel_mapping/surfel_map_node.hpp"

using namespace sparse_surfel_map;

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<SurfelMapNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
