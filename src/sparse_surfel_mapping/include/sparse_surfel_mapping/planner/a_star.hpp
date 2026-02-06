#ifndef A_STAR_HPP_
#define A_STAR_HPP_

#include <vector>
#include <Eigen/Core>
#include "sparse_surfel_mapping/mapper/surfel_map.hpp"

namespace sparse_surfel_map {

class AStarPlanner {
public:
    AStarPlanner() = default;

    void set_map(const SurfelMap* map) { map_ = map; }

    // Plan path through FREE coarse cells adjacent to OCCUPIED cells
    // Returns waypoints as positions (coarse cell centers)
    std::vector<Eigen::Vector3f> plan(const Eigen::Vector3f& start, const Eigen::Vector3f& goal);

    // Check if path exists without computing full path (faster for validation)
    bool path_exists(const Eigen::Vector3f& start, const Eigen::Vector3f& goal);

private:
    // Node validity check (FREE with at least 1 OCCUPIED neighbor)
    bool is_valid_surface_adjacent_node(const VoxelKey& coarse_key) const;

    // Get valid neighboring coarse cells (6-connectivity)
    std::vector<VoxelKey> get_valid_neighbors(const VoxelKey& coarse_key) const;

    // Euclidean distance heuristic
    float heuristic(const VoxelKey& from, const VoxelKey& to) const;

    // Convert coarse keys to 3D positions
    Eigen::Vector3f coarse_key_to_position(const VoxelKey& coarse_key) const;

    const SurfelMap* map_{nullptr};

    // 6-connected neighbor offsets (faces only)
    static constexpr std::array<std::array<int32_t, 3>, 6> neighbor_offsets_6 = {{
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    }};
};

} // namespace

#endif
