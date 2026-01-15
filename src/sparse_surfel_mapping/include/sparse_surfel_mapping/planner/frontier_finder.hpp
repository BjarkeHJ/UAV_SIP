#ifndef FRONTIER_FINDER_HPP_
#define FRONTIER_FINDER_HPP_

#include "sparse_surfel_mapping/common/planning_types.hpp"
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class FrontierFinder {
public:
    FrontierFinder();
    explicit FrontierFinder(const SurfelMap* map);
    void set_map(const SurfelMap* map) { map_ = map; }

    // find_frontier_from_coverage -> main
    std::vector<FrontierSurfel> find_frontiers_from_coverage(const VoxelKeySet& cumulative_coverage, const Eigen::Vector3f& expansion_center, float max_expansion_radius) const;
    std::vector<FrontierCluster> cluster_frontiers(const std::vector<FrontierSurfel>& frontiers, float cluster_radius, size_t min_cluster_size) const;
    void compute_cluster_view_suggestion(FrontierCluster& cluster, float optimal_distance, float min_distance) const;

private:
    std::vector<VoxelKey> get_neighbors_6(const VoxelKey& key) const;
    std::vector<VoxelKey> get_neighbors_26(const VoxelKey& key) const;
    size_t count_uncovered_surface_neighbors(const VoxelKey& key, const VoxelKeySet& covered) const;
    bool has_surface(const VoxelKey& key) const;
    Eigen::Vector3f key_to_position(const VoxelKey& key) const;

    const SurfelMap* map_{nullptr};
};


} // namespace


#endif