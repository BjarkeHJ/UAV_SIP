#include "sparse_surfel_mapping/ros_nodes/surfel_map_node.hpp"
#include "sparse_surfel_mapping/ros_nodes/planner_node.hpp"

using namespace sparse_surfel_map;

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto mapper_node = std::make_shared<SurfelMapNode>();
    auto planner_node = std::make_shared<InspectionPlannerNode>();

    planner_node->set_surfel_map(mapper_node->get_surfel_map());

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(mapper_node);
    executor.add_node(planner_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
