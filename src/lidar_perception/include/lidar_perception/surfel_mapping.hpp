#ifndef SURFEL_MAPPING_HPP_
#define SURFEL_MAPPING_HPP_

#include <chrono>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>

struct Surfel2D {
    size_t id;
    Eigen::Vector3f centroid;
    Eigen::Vector3f normal;
    Eigen::Vector3f t1, t2; // tangetial basis
    Eigen::Matrix2f cov2d;
    float support_r = 0.0f;
    int support = 0;
    float planarity = 0.0f;
    float rms_plane = 0.0f;
};

struct SurfelParams {
    float seed_voxel = 0.5f;
    int max_surfels = 5000;
    int knn = 10;
    float r_scale = 2.5f;
    float r_min = 0.08f;
    float r_max = 0.60f;

    int max_support_pts = 100;

    float max_angle_deg = 10.0f;
    float max_plane_rms = 0.1f;
    float min_planarity = 0.70f;

    bool weight_by_normal_agreement = true;
};

class SurfelMapping {
public:
    SurfelMapping(SurfelParams p = {});

    void run();
    void set_local_frame(const pcl::PointCloud<pcl::PointNormal>::Ptr pts_w_nrm) { pts_w_nrm_ = pts_w_nrm; }
    std::vector<Surfel2D>& get_local_surfels() { return sframe_; }

private:
    SurfelParams p_;
    
    struct VoxelKey { int x,y,z; };
    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey& k) const noexcept {
            std::size_t h = 1469598103934665603ull;
            auto mix = [&](int v){
                h ^= (std::size_t)v + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
            };
            mix(k.x);
            mix(k.y);
            mix(k.z);
            return h;
        }
    };

    struct VoxelKeyEq {
        bool operator()(const VoxelKey& a, const VoxelKey& b) const noexcept {
            return a.x==b.x && a.y==b.y && a.z==b.z;
        }
    };

    /* Functions */
    inline Eigen::Vector3f P(const pcl::PointNormal& pn) const { return {pn.x, pn.y, pn.z}; }
    inline Eigen::Vector3f N(const pcl::PointNormal& pn) const { return {pn.normal_x, pn.normal_y, pn.normal_z}; }
    inline bool finitePN(const pcl::PointNormal& pn) const { return std::isfinite(pn.x) && std::isfinite(pn.y) && std::isfinite(pn.z) && std::isfinite(pn.normal_x) && std::isfinite(pn.normal_y) && std::isfinite(pn.normal_z); }    
    inline float clamp(float v, float lo, float hi) const { return std::max(lo, std::min(hi, v)); }
    inline float deg2rad(float deg) const { return deg * M_PI / 180.0f; }

    // Local frame extract
    void make_tangent_basis(const Eigen::Vector3f& n, Eigen::Vector3f& t1, Eigen::Vector3f& t2) const;
    std::vector<int> voxel_seeds(const pcl::PointCloud<pcl::PointNormal>& cloud) const;
    float estimate_adaptive_radius(pcl::search::KdTree<pcl::PointNormal>& kdtree, const pcl::PointCloud<pcl::PointNormal>& cloud, int seed_idx) const;
    bool build_surfel_from_nbh(const pcl::PointCloud<pcl::PointNormal>& cloud, const std::vector<int>& idx, const Eigen::Vector3f& seed_n, Surfel2D& out) const;
    std::vector<Surfel2D> local_surfel_extraction(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud);

    // Global map fuse
    void closest_surfel_registration();

    /* Data */
    int frame_count{0};
    pcl::PointCloud<pcl::PointNormal>::Ptr pts_w_nrm_;
    std::vector<Surfel2D> sframe_; // current frame
    std::vector<Surfel2D> smap_; // global surfel map

    /* Utils */
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree_;
};

#endif

