#include "sparse_surfel_mapping/preprocess.hpp"

namespace sparse_surfel_map {

ScanPreprocess::ScanPreprocess() : config_() {
    load_config(); // default parameters
}

ScanPreprocess::ScanPreprocess(const PreprocessConfig& config) : config_(config) {
    load_config(); // custom parameters
}

void ScanPreprocess::set_transform(const Eigen::Transform<float, 3, Eigen::Isometry>& tf) {
    gnd_normal_z_ = tf.rotation().row(2);
    gnd_offset_z_ = tf.translation().z();
    have_tf_ = true;
}

bool ScanPreprocess::set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
    if (!in || in->points.empty()) return false;
    cloud_in_ = in; // preprocessor has shared ownership of the cloud while processing
    return true;
}

void ScanPreprocess::process() {
    grid_downsample();
    build_range_image();
    smooth_range_image();
    estimate_normals();
}

void ScanPreprocess::get_points_with_normal(std::vector<PointWithNormal>& pns) {
    pns = std::move(points_with_normal_out_);
    have_tf_ = false; // reset
    points_with_normal_out_.clear();
}

void ScanPreprocess::load_config() {
    ds_ = std::max<size_t>(1, config_.ds_factor);
    W_ = (config_.width + ds_ - 1) / ds_;
    H_ = (config_.height + ds_ - 1) / ds_;

    const float hfov = config_.hfov_deg * M_PI / 180.0f;
    const float vfov = config_.vfov_deg * M_PI / 180.0f;

    yaw_min_ = -hfov * 0.5f;
    yaw_max_ = hfov * 0.5;
    pitch_min_ = -vfov * 0.5f;
    pitch_max_ = vfov * 0.5f;
    yaw_scale_ = (config_.width - 1) / hfov;
    pitch_scale_ = (config_.height - 1) / vfov;

    min_range_sq_ = config_.min_range * config_.min_range;
    max_range_sq_ = config_.max_range * config_.max_range;

    grid_ds_.resize(W_ * H_);
    range_img_.resize(W_ * H_);
    range_img_smooth_.resize(W_ * H_);
    points_with_normal_out_.reserve(W_ * H_);
}

void ScanPreprocess::grid_downsample() {
    // reset grid cells
    for (auto& cell : grid_ds_) {
            cell.valid = false;
            cell.range_sq = std::numeric_limits<float>::infinity();
        }

    // project to downsampled grid
    for (const auto& p : cloud_in_->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

        Eigen::Vector3f pv(p.x, p.y, p.z);

        // Range check
        const float r_sq = p.x * p.x + p.y * p.y + p.z * p.z;
        if (r_sq < min_range_sq_ || r_sq > max_range_sq_) continue;

        // Gnd point check
        if (config_.enable_ground_filter && have_tf_) {
            const float z_world = gnd_normal_z_.dot(pv) + gnd_offset_z_;
            if (z_world < config_.ground_z_min) continue;
        }

        const float yaw = std::atan2(pv.y(), pv.x()); // atan(y / z)
        const float xy_dist = std::sqrt(pv.x() * pv.x() + pv.y() * pv.y());
        const float pitch = std::atan2(pv.z(), xy_dist); // atan(z / sqrt(x² + y²))
        if (yaw < yaw_min_ || yaw > yaw_max_ || pitch < pitch_min_ || pitch > pitch_max_) continue;

        const int u = static_cast<int>((yaw - yaw_min_) * yaw_scale_ / ds_ + 0.5f);
        const int v = static_cast<int>((pitch - pitch_min_) * pitch_scale_ / ds_ + 0.5f);

        if (u < 0 || u >= W_ || v < 0 || v >= H_) continue;

        GridCell& cell = grid_ds_[idx(u,v)];

        // atomic update
        // ensure thread safety here if parallelizing loop (#pragma omp critical)
        {
            if (r_sq < cell.range_sq) {
                cell.point = pv;
                cell.range_sq = r_sq;
                cell.valid = true;
            }
        }
    }
}

void ScanPreprocess::build_range_image() {
    for (size_t i = 0; i < W_ * H_; ++i) {
        if (grid_ds_[i].valid) {
            range_img_[i] = std::sqrt(grid_ds_[i].range_sq);
        }
        else {
            range_img_[i] = std::numeric_limits<float>::infinity();
        }
    }
}

void ScanPreprocess::smooth_range_image() {
    if (config_.range_smooth_iters <= 0) return;

    range_img_smooth_ = range_img_;
    const size_t R = std::max<size_t>(1, config_.normal_est_px_radius);
    const float spatial_sigma = config_.spatial_sigma_px;
    const float depth_sigma = config_.dpeth_sigma_m;
    const float max_jump = config_.max_depth_jump_m;
    
    const float inv_2sigma_sp_sq = 1.0f / (2.0f * spatial_sigma * spatial_sigma);
    const float inv_2sigma_dp_sq = 1.0f / (2.0f * depth_sigma * depth_sigma);

    const int kernel_size = 2 * R + 1;
    std::vector<float> spatial_kernel(kernel_size);
    
    // precompute spatial kernel (gaussian-ish)
    for (int i=0; i<kernel_size; ++i) {
        const float d = float(i - R);
        spatial_kernel[i] = std::exp(-d * d * inv_2sigma_sp_sq);
    }
    
    // buffers
    std::vector<float> temp(W_ * H_);
    std::vector<float>* src = &range_img_smooth_;
    std::vector<float>* dst = &temp;

    for (int iter=0; iter < config_.range_smooth_iters; ++iter) {
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
                
                const int u_min = std::max<size_t>(0, u - R);
                const int u_max = std::min(W_ - 1, u + R);

                for (int uu = u_min; uu <= u_max; ++uu) {
                    const float r = (*src)[idx(uu, v)];
                    if (!std::isfinite(r)) continue;

                    const float dr = r - r_center;
                    if (std::fabs(dr) > max_jump) continue;

                    const float w_spatial = spatial_kernel[uu - u + R];
                    const float w_depth = std::exp(-dr * dr * inv_2sigma_dp_sq);
                    const float w = w_spatial * w_depth;

                    weight_sum += w;
                    value_sum += w * r;
                }

                (*dst)[idx_c] = (weight_sum > 1e-6f) ? (value_sum / weight_sum) : r_center;
            }
        }

        // swap buffers for the second pass (src->temp, dst->range_smooth_)
        std::swap(src, dst);

        // vertical pass
        for (int v = 0; v < H_; ++v) {
            const int v_min = std::max<size_t>(0, v - R);
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
                    const float w_depth = std::exp(-dr * dr * inv_2sigma_dp_sq);
                    const float w = w_spatial * w_depth;

                    weight_sum += w;
                    value_sum += w * r;
                }
                
                (*dst)[idx_c] = (weight_sum > 1e-6f) ? (value_sum / weight_sum) : r_center;
            }
        }

        // swap buffers for next iteration (src->range_smooth_, dst->temp)
        std::swap(src, dst);
    }

    // Ensure final results (src contains the final output)
    if (src != &range_img_smooth_) {
        range_img_smooth_ = *src;
    }
}

void ScanPreprocess::estimate_normals() {
    const float jump_thresh = config_.max_depth_jump_m;

    // use smoothed if enabled
    const std::vector<float>& range_to_use = (config_.range_smooth_iters > 0) ? range_img_smooth_ : range_img_;

    for (int v = 0; v < H_; ++v) {
        for (int u = 0; u < W_; ++u) {
            const int idx_c = idx(u, v);
            const GridCell& center = grid_ds_[idx_c];

            if (!center.valid) continue;

            // pcl::Normal n;
            // n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();

            Eigen::Vector3f n = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
            const Eigen::Vector3f& Pc = center.point;

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
                return {true, pt};
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
                continue;
            }

            Eigen::Vector3f normal = tangent_u.cross(tangent_v);
            const float norm = normal.norm();
            if (norm < 1e-6f) {
                continue;
            }

            normal /= norm;

            if (config_.orient_towards_sensor && normal.dot(Pc) > 0.0f) {
                normal = -normal;
            }

            // create point w normal
            PointWithNormal pn;
            pn.position = Pc;
            pn.normal = normal;
            pn.weight = 1.0f; // TODO
            points_with_normal_out_.push_back(pn);
        }
    }
}

} // namespace