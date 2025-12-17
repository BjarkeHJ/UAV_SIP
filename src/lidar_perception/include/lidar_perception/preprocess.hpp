#ifndef PREPROCESS_HPP_
#define PREPROCESS_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class CloudPreprocess {
public:
    struct Params {
        bool enable_gnd_filter = true;
        float gnd_z_min = 0.25;

        int width = 240;
        int height = 180;
        double hfov_deg = 106.0;
        double vfov_deg = 86.0;
        int ds_factor = 2;
        double min_range = 0.1;
        double max_range = 10.0;
        bool keep_closest = true; // true: keept closes point per block - false: average point per block

        int normal_radius_px = 3;
        float depth_sigma_m = 0.05f; // depth aware similarity for edge aware weights
        float spatial_sigma_px = 1.0f;
        int range_smooth_iters = 2;
        float max_depth_jump_m = 0.10f;
        bool orient_towards_sensor = true;
    };

    CloudPreprocess(const Params& p) : params_(p) {
        cloud_out_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        normals_out_ = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        cloud_w_normals_out_ = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        allocate();
    }

    void set_params(const Params& p) {
        params_ = p;
        allocate();
    }

    void set_world_transform(const Eigen::Vector3f& t, const Eigen::Quaternionf& q) {
        t_ws_ = t;
        R_ws_ = q.toRotationMatrix();
        have_tf_ = true;
    }

    void set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
        if (!in || in->points.empty()) {
            std::cout << "[PREPROCESS] Error: Nullptr or empty cloud!" << std::endl;
            return;
        }
        cloud_ = in;
        clear_grid();
        project_to_grid(cloud_);
    }

    void downsample() {
        /* Requires normal estimation ATM */
        if (!cloud_ || cloud_->points.empty()) {
            std::cout << "[PREPROCESS] Error: No point cloud set!" << std::endl;
            // raise error here instead
            return;
        }

        reduce_grid();
    }

    void normal_estimation() {
        build_range_image();
        // smooth_range_image();
        estimate_normals();
    }

    void get_normals(pcl::PointCloud<pcl::Normal>::Ptr& cloud_n) {
        // Reflect if downsample has been called prior to getting
        cloud_n = normals_out_;
    }

    void get_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_p) {
        // Reflect if downsample has been called prior to getting
        cloud_p = cloud_out_;
    }

    void get_points_with_normals(pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_pn) {
        // Reflect if downsample has been called prior to getting
        cloud_pn = cloud_w_normals_out_;
    }

    void transform_output_to_world() {
        if (!have_tf_) {
            std::cout << "ERROR" << std::endl;
            return;
        }

        if (!cloud_out_ || cloud_out_->empty()) {
            return;
        }

        if (!normals_out_ || normals_out_->size() != cloud_out_->size()) {
            return;
        }

        for (size_t i=0; i<cloud_out_->size(); ++i) {
            auto& p = cloud_out_->points[i];
            auto& n = normals_out_->points[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            Eigen::Vector3f ps(p.x, p.y, p.z);
            Eigen::Vector3f pw = R_ws_ * ps + t_ws_;
            p.x = pw.x();
            p.y = pw.y();
            p.z = pw.z();
            
            if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z)) continue;
            Eigen::Vector3f ns(n.normal_x, n.normal_y, n.normal_z);
            Eigen::Vector3f nw = R_ws_ * ns;
            nw.normalize();
            n.normal_x = nw.x();
            n.normal_y = nw.y();
            n.normal_z = nw.z();
        }

        cloud_w_normals_out_->clear();
        cloud_w_normals_out_->resize(cloud_out_->size());

        for (size_t i=0; i<cloud_out_->size(); ++i) {
            const auto& p = cloud_out_->points[i];
            const auto& n = normals_out_->points[i];
            pcl::PointNormal pn;
            pn.x = p.x;
            pn.y = p.y;
            pn.z = p.z;
            pn.normal_x = n.normal_x;
            pn.normal_y = n.normal_y;
            pn.normal_z = n.normal_z;
            cloud_w_normals_out_->points[i] = pn;
        }

        cloud_w_normals_out_->width = static_cast<uint32_t>(cloud_out_->size());
        cloud_w_normals_out_->height = 1;
        cloud_w_normals_out_->is_dense = false;
    }

private:
    struct Cell {
        pcl::PointXYZ p;
        float r2 = std::numeric_limits<float>::infinity();
        int count = 0;
        bool valid = false;
    };

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_out_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals_out_;

    std::vector<float> range_;
    std::vector<float> range_filt_;
    std::vector<pcl::Normal> normals_grid_;

    Params params_;
    std::vector<Cell> grid_;
    int W_{0}, H_{0}, ds_{0};
    int Wd_{0}, Hd_{0};
    float yaw_min_, yaw_max_, pitch_min_, pitch_max_;
    float yaw_span_, pitch_span_;

    Eigen::Matrix3f R_ws_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_ws_ = Eigen::Vector3f::Zero();
    bool have_tf_ = false;

    void allocate() {
        W_ = params_.width;
        H_ = params_.height;
        ds_ = std::max(1, params_.ds_factor);
        Wd_ = std::max(1, W_ / ds_);
        Hd_ = std::max(1, H_ / ds_);

        grid_.resize(static_cast<size_t>(W_ * H_));
        const float hfov = static_cast<float>(params_.hfov_deg * M_PI / 180.0);
        const float vfov = static_cast<float>(params_.vfov_deg * M_PI / 180.0);

        yaw_min_ = -hfov * 0.5f;
        yaw_max_ = hfov * 0.5f;
        pitch_min_ = -vfov * 0.5f;
        pitch_max_ = vfov * 0.5f;

        yaw_span_ = hfov;
        pitch_span_ = vfov;
    
        range_.assign(static_cast<size_t>(W_ * H_), std::numeric_limits<float>::quiet_NaN());
        range_filt_ = range_;
        normals_grid_.resize(static_cast<size_t>(W_ * H_));
    }

    inline void clear_grid() {
        for (auto& c : grid_) {
            c.valid = false;
            c.count = 0;
            c.r2 = std::numeric_limits<float>::infinity();
        }
    }

    bool inline finite(const pcl::PointXYZ& p) const {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    }

    inline size_t idx(int u, int v) const { return static_cast<size_t>(v * W_ + u); }

    void project_to_grid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
        for (const auto& p : in->points) {
            if (!finite(p)) continue;

            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
            const float r = std::sqrt(r2);

            // range check 
            if (r < params_.min_range || r > params_.max_range) continue;

            // gnd check 
            if (params_.enable_gnd_filter && have_tf_) {
                Eigen::Vector3f ps(p.x, p.y, p.z);
                float z_world = (R_ws_ * ps + t_ws_).z();
                if (z_world < params_.gnd_z_min) continue;
            }

            const float yaw = std::atan2(p.y, p.x);
            const float pitch = std::atan2(p.z, std::sqrt(p.x*p.x + p.y*p.y));

            if (yaw < yaw_min_ || yaw > yaw_max_ || pitch < pitch_min_ || pitch > pitch_max_) continue;

            const int u = static_cast<int>((yaw - yaw_min_) * (W_ - 1) / yaw_span_ + 0.5f);
            const int v = static_cast<int>((pitch - pitch_min_) * (H_ - 1) / pitch_span_ + 0.5f);
            
            if (u < 0 || u >= W_ || v < 0 || v >= H_) continue;

            // Cell& c = grid_[static_cast<size_t>(v * W_ + u)];
            Cell& c = grid_[idx(u,v)];

            if (params_.keep_closest) {
                if (!c.valid || r2 < c.r2) {
                    c.p = p;
                    c.r2 = r2;
                    c.valid = true;
                }
            }
            else {
                if (!c.valid) {
                    c.p = p;
                    c.valid = true;
                    c.count = 1;
                }
                else {
                    c.p.x += p.x;
                    c.p.y += p.y;
                    c.p.z += p.z;
                    c.count++;
                }
            }
        }
    }

    void reduce_grid() {
        cloud_out_->clear();
        cloud_out_->reserve(static_cast<size_t>(Wd_ * Hd_));
        normals_out_->clear();
        normals_out_->reserve(static_cast<size_t>(Wd_ * Hd_));

        // if (Wd_ == W_ && Hd_ == H_) {
        //     // not reduced (ds_factor = 1)
        //     // out = *cloud_;
        //     for (const auto& c : grid_) {
        //         cloud_out_->push_back(c.p);
        //         // same for normals?
        //     }
        //     return;
        // }

        for (int vd = 0; vd < Hd_; ++vd) {
            for (int ud = 0; ud < Wd_; ++ud) {
                // Cell best;
                bool found = false;
                int best_u = -1;
                int best_v = -1;
                float best_r2 = std::numeric_limits<float>::infinity();

                for (int dv = 0; dv < ds_; ++dv) {
                    for (int du = 0; du < ds_; ++du) {
                        const int u = ud * ds_ + du;
                        const int v = vd * ds_ + dv;
                        if (u >= W_ || v >= H_) continue;

                        const Cell& c = grid_[idx(u,v)];
                        if (!c.valid) continue;
                        if (!found || c.r2 < best_r2) {
                            best_r2 = c.r2;
                            best_u = u;
                            best_v = v;
                            found = true;
                        }
                    }
                }

                if (!found) continue;
                cloud_out_->push_back(grid_[idx(best_u, best_v)].p);
                normals_out_->push_back(normals_grid_[idx(best_u, best_v)]);
            }
        }
    }

    void build_range_image() {
        /* Create 2D Depth Map from point cloud distances from sensor */
        for (int v=0; v<H_; ++v) {
            for (int u=0; u<W_; ++u) {
                const Cell& c = grid_[idx(u,v)];
                if (!c.valid) {
                    range_[idx(u,v)] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                range_[idx(u,v)] = std::sqrt(c.r2);
            }
        }
        range_filt_ = range_;
    }

    void smooth_range_image() {
        if (params_.range_smooth_iters <= 0) return;

        const int R = std::max(1, params_.normal_radius_px);
        const float inv2_sp = 1.0f / (2.0f * params_.spatial_sigma_px * params_.spatial_sigma_px);
        const float inv2_dp = 1.0f / (2.0f * params_.depth_sigma_m * params_.depth_sigma_m);

        std::vector<float> tmp = range_filt_;

        for (int it=0; it<params_.range_smooth_iters; ++it) {
            for (int v=0; v<H_; ++v) {
                for (int u=0; u<W_; ++u) {
                    const float rc = tmp[idx(u,v)];
                    if (!std::isfinite(rc)) {
                        range_filt_[idx(u,v)] = rc; // set infinite val
                        continue;
                    }

                    float wsum = 0.0f;
                    float vsum = 0.0f;
                    
                    const int u0 = std::max(0, u-R);
                    const int u1 = std::min(W_-1, u+R);
                    const int v0 = std::max(0, v-R);
                    const int v1 = std::min(H_-1, v+R);

                    for (int vv=v0; vv<=v1; ++vv) {
                        for (int uu=u0; uu<=u1; ++uu) {
                            const float r = tmp[idx(uu,vv)];
                            if (!std::isfinite(r)) continue;

                            const float du = float(uu - u);
                            const float dv = float(vv - v);
                            const float ds2 = du*du + dv*dv;

                            const float dr = r - rc;

                            // gate based on depth jump
                            if (std::fabs(dr) > params_.max_depth_jump_m) continue;

                            const float w = std::exp(-ds2 * inv2_sp) * std::exp(-(dr*dr) * inv2_dp);
                            wsum += w;
                            vsum += w * r;
                        }
                    }
                    range_filt_[idx(u,v)] = (wsum > 1e-6f) ? (vsum / wsum) : rc;
                }
            }
            tmp = range_filt_;
        }
    }

    bool fetch_point(int u, int v, Eigen::Vector3f& P, float& r) const {
        const Cell& c = grid_[idx(u,v)];
        if (!c.valid) return false;
        r = range_filt_[idx(u,v)];
        if (!std::isfinite(r)) return false;
        P = Eigen::Vector3f(c.p.x, c.p.y, c.p.z);
        return true;
    }

    void estimate_normals() {
        const float jump = params_.max_depth_jump_m;

        for (int v=0; v<H_; ++v) {
            for (int u=0; u<W_; ++u) {
                pcl::Normal n;
                n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();

                // center point
                Eigen::Vector3f Pc;
                float rc;
                if (!fetch_point(u, v, Pc, rc)) {
                    normals_grid_[idx(u,v)] = n;
                    continue;
                }

                // left, right, up, down
                Eigen::Vector3f Pl, Pr, Pu, Pd;
                float rl, rr, ru, rd;
                bool okL = (u > 0) && fetch_point(u-1, v, Pl, rl) && (std::fabs(rl - rc) <= jump);
                bool okR = (u < W_-1) && fetch_point(u+1, v, Pr, rr) && (std::fabs(rr - rc) <= jump);
                bool okU = (v > 0) && fetch_point(u, v-1, Pu, ru) && (std::fabs(ru - rc) <= jump);
                bool okD = (v < H_-1) && fetch_point(u, v+1, Pd, rd) && (std::fabs(rd - rc) <= jump);

                Eigen::Vector3f tx, ty;
                if (okL && okR) {
                    tx = (Pr - Pl);
                }
                else if (okR) {
                    tx = (Pr - Pc);
                }
                else if (okL) {
                    tx = (Pc - Pl);
                }
                else {
                    normals_grid_[idx(u,v)] = n;
                    continue;
                }

                if (okU && okD) {
                    ty = (Pd - Pu);
                }
                else if (okD) {
                    ty = (Pd - Pc);
                }
                else if (okU) {
                    ty = (Pc - Pu);
                }
                else {
                    normals_grid_[idx(u,v)] = n;
                    continue;
                }

                Eigen::Vector3f nn = tx.cross(ty);
                const float norm = nn.norm();
                if (norm < 1e-6f) {
                    normals_grid_[idx(u,v)] = n;
                    continue;
                }

                nn /= norm;

                if (params_.orient_towards_sensor) {
                    if (nn.dot(Pc) > 0.0f) nn = -nn;
                }

                n.normal_x = nn.x();
                n.normal_y = nn.y();
                n.normal_z = nn.z();
                normals_grid_[idx(u,v)] = n;
            }
        }
    }
};

#endif