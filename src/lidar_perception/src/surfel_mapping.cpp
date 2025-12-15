#include "lidar_perception/surfel_mapping.hpp"

SurfelMapping::SurfelMapping(SurfelParams p = {}) : p_(p) {
    tree = std::make_shared<pcl::search::KdTree<pcl::PointNormal>>();

}

