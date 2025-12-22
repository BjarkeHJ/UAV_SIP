#ifndef SURFEL_MAP_MANAGER_
#define SURFEL_MAP_MANAGER_

#include "lidar_perception/surfel2d.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace surface_inspection_planner {

struct SurfelMapConfig {
    float max_association_distance = 0.10f;
    float max_normal_angle_deg = 20.0f;
    float max_mahanalobis_distance = 3.0f;

    float measurement_noise_position = 0.02f;
    float measurement_noise_normal = 0.1f;
    float measurement_noise_covariance = 0.001f;

    int min_observation = 2;
    int max_observation = 50;
    float min_confidence_threshold = 0.5f;
    int observation_timeout_frames = 100; // not needed

    bool enable_surfel_merging = true;
    float merge_distance_threshold = 0.05f;
    float merge_normal_threshold_deg = 10.0f;

    float voxel_size = 0.1f;
};

class SurfelMap {
public:
    explicit SurfelMap(const SurfelMapConfig& config = SurfelMapConfig());

    void integrate(const std::vector<Surfel2D>& local_surfels, const Eigen::Isometry3f& sensor_pose, int frame_id);

    std::vector<GlobalSurfel2D> get_surfel_map() const;
    std::vector<GlobalSurfel2D> get_surfels_in_region(const Eigen::Vector3f& center, float radius) const;
    std::vector<GlobalSurfel2D> get_surfels_in_box(const Eigen::Vector3f& min_bounds, const Eigen::Vector3f& max_bounds) const;
    GlobalSurfel2D* get_nearest_surfel(const Eigen::Vector3f& point);

    struct MapStats {
        int total_surfels;
        int stable_surfels;
        int unstable_surfels;
        float average_confidence;
        float average_observations;
    };

    MapStats get_map_stats() const;

    void clear();
    void prune();
    
    bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);

private:
    std::vector<std::pair<int,int>> find_associations(const std::vector<Surfel2D>& local_surfels, const Eigen::Isometry3f& sensor_pose);
    float compute_mahalanobis_distance(const Surfel2D& local_surfel, const GlobalSurfel2D& global_surfel) const;
    void fuse_surfel(GlobalSurfel2D& global_surfel, const Surfel2D& local_surfel, int frame_id);
    void add_surfel(const Surfel2D& surfel, int frame_id);
    void merge_surfels();
    void remove_stale();
    void update_spatial_index();
    void build_kdtree();
    bool are_surfels_similar(const Surfel2D& s1, const GlobalSurfel2D& s2) const;

    SurfelMapConfig config_;
    std::vector<GlobalSurfel2D> surfel_map_;
    int next_id = 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr surfel_centers_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_;

    struct VoxelKey {
        int x, y, z;
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        size_t operator()(const VoxelKey& k) const {
            return ((std::hash<int>()(k.x) ^
                    (std::hash<int>()(k.y) << 1)) >> 1) ^
                    (std::hash<int>()(k.z) << 1);
        }
    };

    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> voxel_grid_;
};

};

#endif