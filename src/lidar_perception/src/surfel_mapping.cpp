#include "lidar_perception/surfel_mapping.hpp"

SurfelMapping::SurfelMapping(SurfelParams p) : p_(p) {
    tree_ = std::make_shared<pcl::search::KdTree<pcl::PointNormal>>();

}

void SurfelMapping::run() {
    sframe_ = local_surfel_extraction(pts_w_nrm_);
    std::cout << "Completed Local Frame Surfel Extraction. Number of Surfels: " << sframe_.size() << std::endl;
}

std::vector<Surfel2D> SurfelMapping::local_surfel_extraction(const pcl::PointCloud<pcl::PointNormal>::Ptr& cloud) {
    std::vector<Surfel2D> surfels;
    if (!cloud || cloud->empty()) return surfels;

    tree_->setInputCloud(cloud);

    std::vector<int> seeds = voxel_seeds(*cloud);

    surfels.reserve(seeds.size());

    std::vector<int> nn_idx;
    std::vector<float> nn_d2;

    for (int s=0; s<(int)seeds.size(); ++s) {
        int seed_i = seeds[s];
        const auto& seed_pn = cloud->points[seed_i];
        if (!finitePN(seed_pn)) continue;

        Eigen::Vector3f seed_n = N(seed_pn);
        float nn = seed_n.norm();
        if (nn < 1e-6f) continue;

        seed_n /= nn;

        float r = estimate_adaptive_radius(*tree_, *cloud, seed_i);

        nn_idx.clear();
        nn_d2.clear();
        int found = tree_->radiusSearch(seed_pn, r, nn_idx, nn_d2, p_.max_support_pts);
        if (found < 8) continue;

        Surfel2D sf;
        if (!build_surfel_from_nbh(*cloud, nn_idx, seed_n, sf)) continue;
        sf.support_r = r;

        surfels.push_back(sf);
        if ((int)surfels.size() >= p_.max_surfels) break;
    }

    return surfels;
}


void SurfelMapping::make_tangent_basis(const Eigen::Vector3f& n, Eigen::Vector3f& t1, Eigen::Vector3f& t2) const {
    Eigen::Vector3f a = (std::abs(n.z()) < 0.9 ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitY());
    t1 = n.cross(a).normalized();
    t2 = n.cross(t1).normalized();
}

std::vector<int> SurfelMapping::voxel_seeds(const pcl::PointCloud<pcl::PointNormal>& cloud) const {
    std::unordered_set<VoxelKey, VoxelKeyHash, VoxelKeyEq> used; // hashtable set of unique keys
    used.reserve(cloud.size());

    std::vector<int> seeds;
    seeds.reserve(cloud.size() / 8); // very conservative

    const float inv = 1.0f / p_.seed_voxel;

    for (int i=0; i<(int)cloud.size(); ++i) {
        const auto& pn = cloud.points[i];
        if (!finitePN(pn)) continue;

        VoxelKey k {
            (int)std::floor(double(pn.x) * inv),
            (int)std::floor(double(pn.y) * inv),
            (int)std::floor(double(pn.z) * inv),
        };

        if (used.insert(k).second) {
            // second "true" if inserted (unique)
            seeds.push_back(i);
            if ((int)seeds.size() >= p_.max_surfels) break;
        }
    }
    return seeds;
}

float SurfelMapping::estimate_adaptive_radius(pcl::search::KdTree<pcl::PointNormal>& kdtree, const pcl::PointCloud<pcl::PointNormal>& cloud, int seed_idx) const {
    std::vector<int> nn_idx;
    std::vector<float> nn_d2;
    nn_idx.resize(p_.knn);
    nn_d2.resize(p_.knn);

    const auto& seed = cloud.points[seed_idx];
    if (!finitePN(seed)) return p_.r_min;

    int found = kdtree.nearestKSearch(seed, p_.knn, nn_idx, nn_d2);
    if (found < 5) return p_.r_min;

    std::vector<float> d;
    d.reserve(found);
    for (int i=0; i<found; ++i) {
        if (nn_d2[i] > 0.0f && std::isfinite(nn_d2[i])) {
            d.push_back(std::sqrt(nn_d2[i]));
        }
    }
    if (d.size() < 3) return p_.r_min;
    std::nth_element(d.begin(), d.begin() + d.size()/2, d.end());
    float med = d[d.size()/2]; // median distance
    float r = p_.r_scale * med;
    return clamp(r, p_.r_min, p_.r_max);
}

bool SurfelMapping::build_surfel_from_nbh(const pcl::PointCloud<pcl::PointNormal>& cloud, const std::vector<int>& idx, const Eigen::Vector3f& seed_n, Surfel2D& out) const {
    /**
     Build a Surfel2D on the plane defined by seed_n. 
     The surfel is based on the neighboring points contained in idx


    **/
    
    if ((int)idx.size() < 8) return false;
    
    const float cos_max = std::cos(deg2rad(p_.max_angle_deg));
    
    Eigen::Vector3f mu = Eigen::Vector3f::Zero();
    Eigen::Vector3f n_acc = Eigen::Vector3f::Zero();
    float W = 0.0f;

    std::vector<int> kept;
    kept.reserve(std::min((int)idx.size(), p_.max_support_pts));

    // TODO: Get average mean before 

    // fit to neighborhood
    for (int j=0; j<(int)idx.size() && (int)kept.size() < p_.max_support_pts; ++j) {
        const auto& pn = cloud.points[idx[j]];
        if (!finitePN(pn)) continue;

        Eigen::Vector3f n = N(pn);
        Eigen::Vector3f p = P(pn);

        float nn = n.norm();
        if (nn < 1e-6) continue;
        n /= nn;

        if (n.dot(seed_n) < 0.0f) {
            n = -n;
        }

        // if seed point is noisy -> ignores potentially good normals and falls apart!
        if (n.dot(seed_n) < cos_max) continue;

        float w = 1.0f;
        if (p_.weight_by_normal_agreement) {
            float c = std::max(0.0f, n.dot(seed_n));
            w = 0.25f + 0.75f * c; // [0.25, 1.0]
        }

        mu += w * p;
        n_acc += w * n;
        W += w;
        kept.push_back(idx[j]);
    }

    if ((int)kept.size() < 8 || W <= 0.0f) return false;

    mu /= W;
    Eigen::Vector3f n = n_acc.normalized();
    if (!std::isfinite(n.x()) || n.norm() < 1e-6) return false;

    // Tangent basis from patch surface normal
    Eigen::Vector3f t1, t2;
    make_tangent_basis(n, t1, t2);

    // 3D cov matrix
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    float rms = 0.0f;

    for (int id : kept) {
        Eigen::Vector3f p = P(cloud.points[id]);
        Eigen::Vector3f d = p - mu;
        C += d * d.transpose();

        float dist_plane = n.dot(d); // signed distance to plane
        rms += dist_plane * dist_plane;
    }

    C /= float(kept.size());
    rms = std::sqrt(rms / float(kept.size()));

    // Eigen values (ascending order)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    if (es.info() != Eigen::Success) return false;
    Eigen::Vector3f eval = es.eigenvalues();
    float lmin = eval(0);
    float lmid = eval(1);
    float lmax = eval(2);

    float planarity = 0.0f;
    if (lmax > 1e-12) {
        planarity = 1.0f - (lmin / lmax);
    }

    if (rms > p_.max_plane_rms) return false;
    if (planarity < p_.min_planarity) return false;

    Eigen::Matrix2f cov2 = Eigen::Matrix2f::Zero();
    for (int id : kept) {
        Eigen::Vector3f d = P(cloud.points[id]) - mu;
        Eigen::Vector2f u(t1.dot(d), t2.dot(d));
        cov2 += u * u.transpose();
    }
    cov2 /= float(kept.size());

    out.centroid = mu;
    out.normal = n;
    out.t1 = t1;
    out.t2 = t2;
    out.cov2d = cov2;
    out.support = (int)kept.size();
    out.planarity = planarity;
    out.rms_plane = rms;
    return true;
}




