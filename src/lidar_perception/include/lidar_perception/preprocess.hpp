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
        bool keep_closest = true; // true: keep closest point per block - false: average point per block

        int normal_radius_px = 3;
        float depth_sigma_m = 0.05f; // depth aware similarity for edge aware weights
        float spatial_sigma_px = 1.0f;
        int range_smooth_iters = 3;
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
        gnd_R_z_ = R_ws_.row(2);
        gnd_t_z_ = t_ws_.z();
        have_tf_ = true;
    }

    void set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
        if (!in || in->points.empty()) {
            std::cout << "[PREPROCESS] Error: Nullptr or empty cloud!" << std::endl;
            return;
        }
        cloud_ = in;
        invalidate_all();
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
            std::cerr << "[PREPROCESS] Error: world transform not set (call set_world_transform)" << std::endl;
            return;
        }

        if (!cloud_out_ || cloud_out_->empty()) {
            return;
        }

        if (!normals_out_ || normals_out_->size() != cloud_out_->size()) {
            return;
        }

        cloud_w_normals_out_->resize(cloud_out_->size());

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
            const float nw_norm = nw.norm();
            if (nw_norm > 1e-6f) {
                const float norm_inv = 1.0f / nw_norm;
                n.normal_x = nw.x() * norm_inv;
                n.normal_y = nw.y() * norm_inv;
                n.normal_z = nw.z() * norm_inv;
            }
            else {
                n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();
            }

            pcl::PointNormal& pn = cloud_w_normals_out_->points[i];
            pn.x = p.x;
            pn.y = p.y;
            pn.z = p.z;
            pn.normal_x = n.normal_x;
            pn.normal_y = n.normal_y;
            pn.normal_z = n.normal_z;
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
        uint32_t stamp = 0; // 0 is never written
    };

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_out_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals_out_;
    
    std::vector<Cell> grid_;
    std::vector<float> range_;
    std::vector<float> range_filt_;
    std::vector<pcl::Normal> normals_grid_;
    std::vector<Cell> grid_ds_;
    std::vector<float> range_ds_;
    std::vector<float> range_filt_ds_;
    std::vector<pcl::Normal> normals_ds_;

    Params params_;
    int W_{0}, H_{0}, ds_{0};
    int Wd_{0}, Hd_{0};
    float yaw_min_, yaw_max_, pitch_min_, pitch_max_;
    float yaw_span_, pitch_span_;
    float inv_yaw_span_, inv_pitch_span_;
    float min_range_sq_, max_range_sq_;
    
    uint32_t frame_id_{1}; // unique id of the scan
    
    Eigen::Matrix3f R_ws_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_ws_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f gnd_R_z_;
    float gnd_t_z_;
    bool have_tf_ = false;

    void allocate() {
        W_ = params_.width;
        H_ = params_.height;
        ds_ = std::max(1, params_.ds_factor);
        Wd_ = std::max(1, W_ / ds_);
        Hd_ = std::max(1, H_ / ds_);

        const float hfov = static_cast<float>(params_.hfov_deg * M_PI / 180.0);
        const float vfov = static_cast<float>(params_.vfov_deg * M_PI / 180.0);
        
        yaw_min_ = -hfov * 0.5f;
        yaw_max_ = hfov * 0.5f;
        pitch_min_ = -vfov * 0.5f;
        pitch_max_ = vfov * 0.5f;
        
        yaw_span_ = hfov;
        pitch_span_ = vfov;
        
        inv_yaw_span_ = (W_ - 1) / yaw_span_;
        inv_pitch_span_ = (H_ - 1) / pitch_span_;
        min_range_sq_ = params_.min_range * params_.min_range;
        max_range_sq_ = params_.max_range * params_.max_range;
        
        grid_.resize(static_cast<size_t>(W_ * H_));
        range_.assign(static_cast<size_t>(W_ * H_), std::numeric_limits<float>::quiet_NaN());
        range_filt_ = range_;
        normals_grid_.resize(static_cast<size_t>(W_ * H_));

        grid_ds_.resize(static_cast<size_t>(Wd_ * Hd_));
        range_ds_.assign(static_cast<size_t>(Wd_ * Hd_), std::numeric_limits<float>::quiet_NaN());
        range_filt_ds_ = range_ds_;
        normals_ds_.resize(static_cast<size_t>(Wd_ * Hd_));
    }

    inline void invalidate_all() {
        if (++frame_id_ == 0) {
            frame_id_ = 1;
            for (auto& c : grid_) c.stamp = 0;
        }
    }

    inline bool is_valid(const Cell& c) const { return c.stamp == frame_id_; }
    
    bool inline finite(const pcl::PointXYZ& p) const {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    }

    inline size_t idx(int u, int v) const { return static_cast<size_t>(v * W_ + u); }

    inline size_t idx_ds(int u, int v) const { return static_cast<size_t>(v * Wd_ + u); }

    void project_to_grid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
        for (const auto& p : in->points) {
            if (!finite(p)) continue;

            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;

            // range check 
            if (r2 < min_range_sq_ || r2 > max_range_sq_) continue;

            // gnd check 
            if (params_.enable_gnd_filter && have_tf_) {
                const float z_world = gnd_R_z_.x() * p.x + gnd_R_z_.y() * p.y + gnd_R_z_.z() * p.z + gnd_t_z_;
                if (z_world < params_.gnd_z_min) continue;
            }

            const float yaw = std::atan2(p.y, p.x);
            const float xy_dist = std::sqrt(p.x * p.x + p.y * p.y);
            const float pitch = std::atan2(p.z, xy_dist);

            if (yaw < yaw_min_ || yaw > yaw_max_ || pitch < pitch_min_ || pitch > pitch_max_) continue;

            const int u = static_cast<int>((yaw - yaw_min_) * (W_ - 1) / yaw_span_ + 0.5f);
            const int v = static_cast<int>((pitch - pitch_min_) * (H_ - 1) / pitch_span_ + 0.5f);
            
            if (u < 0 || u >= W_ || v < 0 || v >= H_) continue;

            Cell& c = grid_[idx(u,v)];
            if (params_.keep_closest) {
                if (!is_valid(c) || r2 < c.r2) {
                    c.p = p;
                    c.r2 = r2;
                    c.stamp = frame_id_;
                }
            }
            else {
                if (!is_valid(c)) {
                    c.p = p;
                    c.stamp = frame_id_;
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
 
        for (auto& c : grid_ds_) {
            c.stamp = 0;
            c.r2 = std::numeric_limits<float>::quiet_NaN();
        }

        for (int vd = 0; vd < Hd_; ++vd) {
            for (int ud = 0; ud < Wd_; ++ud) {
                // Cell best;
                bool found = false;
                float best_r2 = std::numeric_limits<float>::infinity();
                pcl::PointXYZ best_p;

                const int v_end = std::min((vd + 1) * ds_, H_);
                const int u_end = std::min((ud + 1) * ds_, W_);

                for (int v = vd*ds_; v < v_end; ++v) {
                    for (int u = ud*ds_; u < u_end; ++u) {
                        const Cell& c = grid_[idx(u,v)];
                        if (!is_valid(c)) continue;
                        if (!found || c.r2 < best_r2) {
                            best_r2 = c.r2;
                            best_p = c.p;
                            found = true;
                        }
                    }
                }

                if (!found) continue;
                Cell& cd = grid_ds_[idx_ds(ud, vd)];
                cd.p = best_p;
                cd.r2 = best_r2;
                cd.stamp = frame_id_;
                cloud_out_->push_back(best_p);
            }
        }

        cloud_out_->width = static_cast<uint32_t>(cloud_out_->size());
        cloud_out_->height = 1;
        cloud_out_->is_dense = false;
    }

    void build_range_image() {
        /* Create 2D Depth Map from point cloud distances from sensor */

        for (int vd=0; vd<Hd_; ++vd) {
            for (int ud=0; ud<Wd_; ++ud) {
                const size_t i = idx_ds(ud, vd);
                const Cell& c = grid_ds_[i];

                if (c.stamp != frame_id_) {
                    range_ds_[i] = std::numeric_limits<float>::quiet_NaN();
                }
                else {
                    range_ds_[i] = std::sqrt(c.r2);
                }
            }
        }
        range_filt_ds_ = range_ds_;
    }

    void smooth_range_image() {
        if (params_.range_smooth_iters <= 0) return;
        const int R = std::max(1, params_.normal_radius_px);
        const float sigma = params_.spatial_sigma_px;
        const float depth_sigma = params_.depth_sigma_m;
        const float max_jump = params_.max_depth_jump_m;

        const int kernel_size = 2 * R + 1;
        std::vector<float> kernel(kernel_size);
        const float inv2_sp = 1.0f / (2.0f * sigma * sigma);

        for (int i=0; i<kernel_size; ++i) {
            const float d = float(i - R);
            kernel[i] = std::exp(-d * d * inv2_sp);
        }

        std::vector<float> tmp(W_ * H_, std::numeric_limits<float>::quiet_NaN());
        const float inv2_dp = 1.0f / (2.0f * depth_sigma * depth_sigma);

        for (int it=0; it < params_.range_smooth_iters; ++it) {
            // horizontal
            for (int v=0; v<H_; ++v) {
                const size_t row_base = v * W_;

                for (int u=0; u<W_; ++u) {
                    const size_t center_idx = row_base + u;
                    const float rc = range_filt_[center_idx];

                    if (!std::isfinite(rc)) {
                        tmp[center_idx] = rc;
                        continue;
                    }

                    float wsum = 0.0f;
                    float vsum = 0.0f;

                    const int u0 = std::max(0, u-R);
                    const int u1 = std::min(W_-1, u+R);

                    for (int uu=u0; uu<=u1; ++uu) {
                        const size_t j = row_base + uu;
                        const float r = range_filt_[j];
                        if (!std::isfinite(r)) continue;

                        const float dr = r - rc;
                        if (std::fabs(dr) > max_jump) continue;

                        const float w_spatial = kernel[uu - u + R];
                        const float w_depth = std::exp(-dr * dr * inv2_dp);
                        const float w = w_spatial * w_depth;

                        wsum += w;
                        vsum += w * r;
                    }

                    tmp[center_idx] = (wsum > 1e-6f) ? (vsum / wsum) : rc;
                }
            }

            // vertical 
            for (int v=0; v<H_; ++v) {
                const int v0 = std::max(0, v-R);
                const int v1 = std::min(H_-1, v+R);

                for (int u=0; u<W_; ++u) {
                    const size_t center_idx = idx(u,v);
                    const float rc = tmp[center_idx];

                    if (!std::isfinite(rc)) {
                        range_filt_[center_idx] = rc;
                        continue;
                    }

                    float wsum = 0.0f;
                    float vsum = 0.0f;

                    for (int vv=v0; vv<=v1; ++vv) {
                        const size_t j = idx(u,vv);

                        const float r = tmp[j];
                        if (!std::isfinite(r)) continue;

                        const float dr = r - rc;
                        if (std::fabs(dr) > max_jump) continue;

                        const float w_spatial = kernel[vv - v + R];
                        const float w_depth = std::exp(-dr * dr * inv2_dp);
                        const float w = w_spatial * w_depth;

                        wsum += w;
                        vsum += w * r;
                    }

                    range_filt_[center_idx] = (wsum > 1e-6f) ? (vsum / wsum) : rc;
                }
            }
        }
    }

    bool fetch_point(int u, int v, Eigen::Vector3f& P, float& r) const {
        if (u < 0 || u >= Wd_ || v < 0 || v >= Hd_) return false;
        const size_t i = idx_ds(u,v);
        const Cell& c = grid_ds_[i];
        if (c.stamp != frame_id_) return false;
        r = range_filt_ds_[i];
        if (!std::isfinite(r)) return false;
        P = Eigen::Vector3f(c.p.x, c.p.y, c.p.z);
        return true;
    }

    void estimate_normals() {
        normals_out_->clear();
        normals_out_->resize(cloud_out_->size());
        const float jump = params_.max_depth_jump_m;

        size_t out_i = 0;

        for (int vd=0; vd<Hd_; ++vd) {
            for (int ud=0; ud<Wd_; ++ud) {
                const Cell& c = grid_ds_[idx_ds(ud, vd)];
                if (c.stamp != frame_id_) continue;

                pcl::Normal n;
                n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();

                // center point
                Eigen::Vector3f Pc;
                float rc;
                if (!fetch_point(ud, vd, Pc, rc)) {
                    (*normals_out_)[out_i++] = n;
                    continue;
                }

                // left, right, up, down
                Eigen::Vector3f Pl, Pr, Pu, Pd;
                float rl, rr, ru, rd;
                bool okL = (ud > 0) && fetch_point(ud-1, vd, Pl, rl) && (std::fabs(rl - rc) <= jump);
                bool okR = (ud < Wd_-1) && fetch_point(ud+1, vd, Pr, rr) && (std::fabs(rr - rc) <= jump);
                bool okU = (vd > 0) && fetch_point(ud, vd-1, Pu, ru) && (std::fabs(ru - rc) <= jump);
                bool okD = (vd < Hd_-1) && fetch_point(ud, vd+1, Pd, rd) && (std::fabs(rd - rc) <= jump);

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
                    continue;
                }

                Eigen::Vector3f nn = tx.cross(ty);
                const float norm = nn.norm();
                if (norm < 1e-6f) continue;

                const float inv_norm = 1.0f / norm;
                nn *= inv_norm;

                if (params_.orient_towards_sensor && nn.dot(Pc) > 0.0f) {
                    nn = -nn;
                }

                n.normal_x = nn.x();
                n.normal_y = nn.y();
                n.normal_z = nn.z();

                (*normals_out_)[out_i++] = n;
            }
        }

        normals_out_->width = static_cast<uint32_t>(normals_out_->size());
        normals_out_->height = 1;
        normals_out_->is_dense = false;
    }

};

#endif