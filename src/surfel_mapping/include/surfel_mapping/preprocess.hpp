#ifndef PREPROCESS_HPP_
#define PREPROCESS_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

/* 
* CloudPreprocess: Efficient pointcloud surface normal estimation via spherical projection and finite depth difference
* 
* Pipeline: 
* 1. Project 3D points to 2D spherical grid (yaw/pitch angles)
* 2. Downsample by selecting closest point in each block
* 3. Build range (depth) image
* 4. Smooth range image (Optional)
* 5. Estimate surfa normals from range image using finite differences (gradient approximation)
* 6. Transform to world frame for outputs
* 
*/

class CloudPreprocess {
public:
    struct Params {
        // Ground Filtering
        bool enable_gnd_filter = true;
        float gnd_z_min = 0.25f;

        // Grid resolution and range (Based on sensor specs)
        int width = 240;
        int height = 180;
        float hfov_deg = 106.0f;
        float vfov_deg = 86.0f;
        float min_range = 0.1f;
        float max_range = 10.0f;

        // Downsampling
        int ds_factor = 2;
        
        // Normal estimation 
        int normal_radius_px = 3;
        bool orient_towards_sensor = true;
        
        // Range image smoothing (0 iterations for disabling)
        int range_smooth_iters = 3; // bilateral filter iterations 
        float depth_sigma_m = 0.05f; // depth based weight
        float spatial_sigma_px = 1.0f; // spatial weight
        float max_depth_jump_m = 0.10f;
    };

    CloudPreprocess(const Params& p) : params_(p) {
        cloud_out_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        normals_out_ = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        cloud_w_normals_out_ = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        allocate();
    }

    // Setters
    void set_params(const Params& p) {
        params_ = p;
        allocate();
    }
    void set_world_transform(const Eigen::Vector3f& t, const Eigen::Quaternionf& q) {
        t_ws_ = t;
        R_ws_ = q.toRotationMatrix();
        gnd_normal_z_ = R_ws_.row(2);
        gnd_offset_z_ = t_ws_.z();
        have_tf_ = true;
    }
    void set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
        if (!in || in->points.empty()) {
            std::cerr << "[PREPROCESS] Empty input cloud!" << std::endl;
            return;
        }
        cloud_in_ = in;
    }

    // Main API
    void downsample() {
        project_to_grid_and_downsample(cloud_in_);
    }

    void normal_estimation() {
        build_range_image();
        if (params_.range_smooth_iters > 0) {
            smooth_range_image();
        }
        estimate_normals();
    }
    
    void transform_output_to_world() {
        if (have_tf_) {
            transform_to_world();
        }
    }

    // Getters
    void get_normals(pcl::PointCloud<pcl::Normal>::Ptr& cloud_n) {
        cloud_n = normals_out_;
    }
    void get_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_p) {
        cloud_p = cloud_out_;
    }
    void get_points_with_normals(pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_pn) {
        cloud_pn = cloud_w_normals_out_;
    }


private:
    struct GridCell {
        pcl::PointXYZ point;
        float range_sq;
        bool valid;
    };

    /* Member Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_out_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals_out_;
    
    std::vector<GridCell> grid_ds_;
    std::vector<float> range_img_;
    std::vector<float> range_smooth_;
    
    Params params_;
    
    int W_full_{0}, H_full_{0}; // original grid_size
    int W_{0}, H_{0}; // downsampeld grid size
    int ds_{0};

    float yaw_min_, yaw_max_, pitch_min_, pitch_max_;
    float yaw_scale_, pitch_scale_;
    float min_range_sq_, max_range_sq_;
    
    Eigen::Matrix3f R_ws_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_ws_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f gnd_normal_z_;
    float gnd_offset_z_;
    bool have_tf_ = false;

    void allocate() {
        W_full_ = params_.width;
        H_full_ = params_.height;
        ds_ = std::max(1, params_.ds_factor);
        W_ = (W_full_ + ds_ - 1) / ds_;
        H_ = (H_full_ + ds_ - 1) / ds_;

        const float hfov = params_.hfov_deg * M_PI / 180.0;
        const float vfov = params_.vfov_deg * M_PI / 180.0;
        
        yaw_min_ = -hfov * 0.5f;
        yaw_max_ = hfov * 0.5f;
        pitch_min_ = -vfov * 0.5f;
        pitch_max_ = vfov * 0.5f;
        
        yaw_scale_ = (W_full_ - 1) / hfov;
        pitch_scale_ = (H_full_ - 1) / vfov;

        min_range_sq_ = params_.min_range * params_.min_range;
        max_range_sq_ = params_.max_range * params_.max_range;

        grid_ds_.resize(W_ * H_);
        range_img_.resize(W_ * H_);
        range_smooth_.resize(W_ * H_);
    }
   
    inline size_t idx(int u, int v) const { return static_cast<size_t>(v * W_ + u); }

    void project_to_grid_and_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {

        // Clear downsampled grid
        for (auto& cell : grid_ds_) {
            cell.valid = false;
            cell.range_sq = std::numeric_limits<float>::infinity();
        }

        // full size grid (on stack)
        std::vector<GridCell> temp_grid(W_full_ * H_full_); 
        for (auto& cell : temp_grid) {
            cell.valid = false;
            cell.range_sq = std::numeric_limits<float>::infinity();
        }

        for (const auto& p : in->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

            // range check 
            const float r_sq = p.x*p.x + p.y*p.y + p.z*p.z;
            if (r_sq < min_range_sq_ || r_sq > max_range_sq_) continue;

            // gnd check 
            if (params_.enable_gnd_filter && have_tf_) {
                // const float z_world = gnd_normal_z_.x() * p.x + gnd_normal_z_.y() * p.y + gnd_normal_z_.z() * p.z + gnd_offset_z_;
                const float z_world = gnd_normal_z_.dot(Eigen::Vector3f(p.x, p.y, p.z)) + gnd_offset_z_;
                if (z_world < params_.gnd_z_min) continue;
            }

            // angle check
            const float yaw = std::atan2(p.y, p.x);
            const float xy_dist = std::sqrt(p.x * p.x + p.y * p.y);
            const float pitch = std::atan2(p.z, xy_dist);

            if (yaw < yaw_min_ || yaw > yaw_max_ || pitch < pitch_min_ || pitch > pitch_max_) continue;

            // convert to pixel coordinates 
            const int u = static_cast<int>((yaw - yaw_min_) * yaw_scale_ + 0.5f);
            const int v = static_cast<int>((pitch - pitch_min_) * pitch_scale_ + 0.5f);
            
            if (u < 0 || u >= W_full_ || v < 0 || v >= H_full_) continue;

            GridCell& cell = temp_grid[v * W_full_ + u];
            if (r_sq < cell.range_sq) {
                cell.point = p;
                cell.range_sq = r_sq;
                cell.valid = true;
            }
        }

        // Downsample the grid: take closest in each block
        for (int vd = 0; vd < H_; ++vd) {
            for (int ud = 0; ud < W_; ++ud) {
                float best_r_sq = std::numeric_limits<float>::infinity();
                pcl::PointXYZ best_p;
                bool found = false;

                const int v_start = vd * ds_;
                const int u_start = ud * ds_;
                const int v_end = std::min(v_start + ds_, H_full_);
                const int u_end = std::min(u_start + ds_, W_full_);

                for (int v = v_start; v < v_end; ++v) {
                    for (int u = u_start; u < u_end; ++u) {
                        const GridCell& cell = temp_grid[v * W_full_ + u];
                        if (cell.valid && cell.range_sq < best_r_sq) {
                            best_r_sq = cell.range_sq;
                            best_p = cell.point;
                            found = true;
                        }
                    }
                }

                GridCell& ds_cell = grid_ds_[idx(ud, vd)];
                if (found) {
                    ds_cell.point = best_p;
                    ds_cell.range_sq = best_r_sq;
                    ds_cell.valid = true;
                }
                else {
                    ds_cell.valid = false;
                }
            }
        }

        // Build output cloud from downsampled grid
        cloud_out_->clear();
        cloud_out_->reserve(W_ * H_);

        for (const auto& cell : grid_ds_) {
            if (cell.valid) {
                cloud_out_->points.push_back(cell.point);
            }
        }

        cloud_out_->width = cloud_out_->size();
        cloud_out_->height = 1;
        cloud_out_->is_dense = false;
    }

    void build_range_image() {
        /* Create 2D Depth Map from point cloud distances from sensor */
        for (int i = 0; i < W_ * H_; ++i) {
            if (grid_ds_[i].valid) {
                range_img_[i] = std::sqrt(grid_ds_[i].range_sq);
            }
            else {
                range_img_[i] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    void smooth_range_image() {
        if (params_.range_smooth_iters <= 0) return;
        
        // Initialize smooth buffer to original
        range_smooth_ = range_img_;
        
        const int R = std::max(1, params_.normal_radius_px);
        const float spatial_sigma = params_.spatial_sigma_px;
        const float depth_sigma = params_.depth_sigma_m;
        const float max_jump = params_.max_depth_jump_m;

        const float inv_2sigma_sp_sq = 1.0f / (2.0f * spatial_sigma * spatial_sigma);
        const float inv_2sigmag_dp_sq = 1.0f / (2.0f * depth_sigma * depth_sigma);

        const int kernel_size = 2 * R + 1;
        std::vector<float> spatial_kernel(kernel_size);
        
        // precompute spatial kernel (gaussian-ish)
        for (int i=0; i<kernel_size; ++i) {
            const float d = float(i - R);
            spatial_kernel[i] = std::exp(-d * d * inv_2sigma_sp_sq);
        }
        
        // buffers
        std::vector<float> temp(W_ * H_);
        std::vector<float>* src = &range_smooth_;
        std::vector<float>* dst = &temp;

        for (int iter=0; iter < params_.range_smooth_iters; ++iter) {
            // horizontal pass
            for (int v = 0; v < H_; ++v) {
                for (int u = 0; u < W_; ++u) {
                    const int idx_c = idx(u, v);
                    const float r_center = (*src)[idx_c];

                    if (!std::isfinite(r_center)) {
                        (*dst)[idx_c] = r_center;
                        continue; 
                    }

                    float weight_sum = 0.0f;
                    float value_sum = 0.0f;
                    
                    const int u_min = std::max(0, u - R);
                    const int u_max = std::min(W_ - 1, u + R);

                    for (int uu = u_min; uu <= u_max; ++uu) {
                        const float r = (*src)[idx(uu, v)];
                        if (!std::isfinite(r)) continue;

                        const float dr = r - r_center;
                        if (std::fabs(dr) > max_jump) continue;

                        const float w_spatial = spatial_kernel[uu - u + R];
                        const float w_depth = std::exp(-dr * dr * inv_2sigmag_dp_sq);
                        const float w = w_spatial * w_depth;

                        weight_sum += w;
                        value_sum += w * r;
                    }

                    (*dst)[idx_c] = (weight_sum < 1e-6f) ? (value_sum / weight_sum) : r_center;
                }
            }

            // swap buffers for the second pass (src->temp, dst->range_smooth_)
            std::swap(src, dst);

            // vertical pass
            for (int v = 0; v < H_; ++v) {
                const int v_min = std::max(0, v - R);
                const int v_max = std::min(H_ - 1, v + R);

                for (int u = 0; u < W_; ++u) {
                    const int idx_c = idx(u, v);
                    const float r_center = (*src)[idx_c];
                    
                    if (!std::isfinite(r_center)) {
                        (*dst)[idx_c] = r_center;
                        continue;
                    }

                    float weight_sum = 0.0f;
                    float value_sum = 0.0f;

                    for (int vv = v_min; vv <= v_max; ++vv) {
                        const float r = (*src)[idx(u, vv)];
                        if (!std::isfinite(r)) continue;

                        const float dr = r - r_center;
                        if (std::fabs(dr) > max_jump) continue;

                        const float w_spatial = spatial_kernel[vv - v + R];
                        const float w_depth = std::exp(-dr * dr * inv_2sigmag_dp_sq);
                        const float w = w_spatial * w_depth;

                        weight_sum += w;
                        value_sum += w * r;
                    }
                    
                    (*dst)[idx_c] = (weight_sum < 1e-6f) ? (value_sum / weight_sum) : r_center;
                }
            }

            // swap buffers for next iteration (src->range_smooth_, dst->temp)
            std::swap(src, dst);
        }

        // Ensure final results (src contains the final output)
        if (src != &range_smooth_) {
            range_smooth_ = *src;
        }
    }

    void estimate_normals() {
        normals_out_->clear();
        normals_out_->resize(cloud_out_->size());
        const float jump_thresh = params_.max_depth_jump_m;
        size_t out_idx = 0;

        // use smoothed if enabled
        const std::vector<float>& range_to_use = (params_.range_smooth_iters > 0) ? range_smooth_ : range_img_;

        for (int v = 0; v < H_; ++v) {
            for (int u = 0; u < W_; ++u) {
                const int idx_c = idx(u, v);
                const GridCell& center = grid_ds_[idx_c];

                if (!center.valid) continue;

                pcl::Normal n;
                n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();

                const Eigen::Vector3f Pc(center.point.x, center.point.y, center.point.z);
                const float rc = range_to_use[idx_c];

                // fetch nb function
                auto fetch_nb = [&](int du, int dv) -> std::pair<bool, Eigen::Vector3f> {
                    const int uu = u + du;
                    const int vv = v + dv;
                    if (uu < 0 || uu >= W_ || vv < 0 || vv >= H_) return {false, {}};

                    const int idx_n = idx(uu, vv);
                    if (!grid_ds_[idx_n].valid) return {false, {}};

                    const float rn = range_to_use[idx_n];
                    if (!std::isfinite(rn) || std::fabs(rn - rc) > jump_thresh) return {false, {}};

                    const auto& pt = grid_ds_[idx_n].point;
                    return {true, Eigen::Vector3f(pt.x, pt.y, pt.z)};
                };

                auto [okL, Pl] = fetch_nb(-1, 0);
                auto [okR, Pr] = fetch_nb(+1, 0);
                auto [okU, Pu] = fetch_nb(0, -1);
                auto [okD, Pd] = fetch_nb(0, +1);

                Eigen::Vector3f tangent_u, tangent_v;
                bool has_u = false;
                bool has_v = false;

                if (okL && okR) {
                    tangent_u = Pr - Pl;
                    has_u = true;
                }
                else if (okR) {
                    tangent_u = Pr - Pc;
                    has_u = true;
                }
                else if (okL) {
                    tangent_u = Pc - Pl;
                    has_u = true;
                }

                if (okU && okD) {
                    tangent_v = Pd - Pu;
                    has_v = true;
                }
                else if (okD) {
                    tangent_v = Pd - Pc;
                    has_v = true;
                }
                else if (okU) {
                    tangent_v = Pc - Pu;
                    has_v = true;
                }
                
                if (!has_u || !has_v) {
                    (*normals_out_)[out_idx++] = n;
                    continue;
                }

                Eigen::Vector3f normal = tangent_u.cross(tangent_v);
                const float norm = normal.norm();
                if (norm < 1e-6f) {
                    (*normals_out_)[out_idx++] = n;
                    continue;
                }

                normal /= norm;

                if (params_.orient_towards_sensor && normal.dot(Pc) > 0.0f) {
                    normal = -normal;
                }

                n.normal_x = normal.x();
                n.normal_y = normal.y();
                n.normal_z = normal.z();
                (*normals_out_)[out_idx++] = n;
            }
        }

        normals_out_->width = static_cast<uint32_t>(normals_out_->size());
        normals_out_->height = 1;
        normals_out_->is_dense = false;
    }

    void transform_to_world() {
        if (cloud_out_->empty() || normals_out_->size() != cloud_out_->size()) return;

        cloud_w_normals_out_->resize(cloud_out_->size());

        for (size_t i=0; i<cloud_out_->size(); ++i) {
            auto& p = cloud_out_->points[i];
            auto& n = normals_out_->points[i];

            // transform point
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            Eigen::Vector3f pw = R_ws_ * Eigen::Vector3f(p.x, p.y, p.z) + t_ws_;
            p.x = pw.x();
            p.y = pw.y();
            p.z = pw.z();
            
            // transform normal
            if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z)) continue;
            Eigen::Vector3f nw = R_ws_ * Eigen::Vector3f(n.normal_x, n.normal_y, n.normal_z);
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
};

#endif