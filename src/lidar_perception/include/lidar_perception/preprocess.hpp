#ifndef PREPROCESS_HPP_
#define PREPROCESS_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class CloudPreprocess {
public:
    struct Params {
        int width = 240;
        int height = 180;
        double hfov_deg = 106.0;
        double vfov_deg = 86.0;
        int ds_factor = 2;
        double min_range = 0.1;
        double max_range = 10.0;
        bool keep_closest = true; // true: keept closes point per block - false: average point per block
    };

    CloudPreprocess(const Params& p) : params_(p) {
        allocate();
    }

    void setParams(const Params& p) {
        params_ = p;
        allocate();
    }

    void set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& in) {
        cloud_ = in;
    }

    void downsample(const pcl::PointCloud<pcl::PointXYZ>& in, pcl::PointCloud<pcl::PointXYZ>& out) {
        clearGrid();
        projectToGrid(in);
        reduceGrid(out);
    }

    void normal_estimation() {

    }

    void ground_filtering(const float th) {

    }

    void outlier_removal() {

    }


private:
    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;


    struct Cell {
        pcl::PointXYZ p;
        float r2 = std::numeric_limits<float>::infinity();
        int count = 0;
        bool valid = false;
    };

    Params params_;
    std::vector<Cell> grid_;
    int W_{0}, H_{0}, ds_{0};
    int Wd_{0}, Hd_{0};
    float yaw_min_, yaw_max_, pitch_min_, pitch_max_;
    float yaw_span_, pitch_span_;

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
    }

    inline void clearGrid() {
        for (auto& c : grid_) {
            c.valid = false;
            c.count = 0;
            c.r2 = std::numeric_limits<float>::infinity();
        }
    }

    bool inline finite(const pcl::PointXYZ& p) const {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    }

    void projectToGrid(const pcl::PointCloud<pcl::PointXYZ>& in) {
        for (const auto& p : in.points) {
            if (!finite(p)) continue;

            const float r2 = p.x*p.x + p.y*p.y + p.z*p.z;
            const float r = std::sqrt(r2);

            if (r < params_.min_range || r > params_.max_range) continue;

            const float yaw = std::atan2(p.y, p.x);
            const float pitch = std::atan2(p.z, std::sqrt(p.x*p.x + p.y*p.y));

            if (yaw < yaw_min_ || yaw > yaw_max_ || pitch < pitch_min_ || pitch > pitch_max_) continue;

            const int u = static_cast<int>((yaw - yaw_min_) * (W_ - 1) / yaw_span_ + 0.5f);
            const int v = static_cast<int>((pitch - pitch_min_) * (H_ - 1) / pitch_span_ + 0.5f);
            
            if (u < 0 || u >= W_ || v < 0 || v >= H_) continue;

            Cell& c = grid_[static_cast<size_t>(v * W_ + u)];

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

    void reduceGrid(pcl::PointCloud<pcl::PointXYZ>& out) {
        out.clear();
        out.reserve(static_cast<size_t>(Wd_ * Hd_));

        for (int vd = 0; vd < Hd_; ++vd) {
            for (int ud = 0; ud < Wd_; ++ud) {
                Cell best;
                for (int dv = 0; dv < ds_; ++dv) {
                    for (int du = 0; du < ds_; ++du) {
                        const int u = ud * ds_ + du;
                        const int v = vd * ds_ + dv;
                        if (u >= W_ || v >= H_) continue;

                        const Cell& c = grid_[static_cast<size_t>(v * W_ + u)];
                        if (!c.valid) continue;

                        if (!best.valid || c.r2 < best.r2) best = c;
                    }
                }

                if (best.valid) {
                    if (!params_.keep_closest && best.count > 1) {
                        const float inv = 1.0f / static_cast<float>(best.count);
                        best.p.x *= inv;
                        best.p.y *= inv;
                        best.p.z *= inv;
                    }
                    out.push_back(best.p);
                }
            }
        }
    }
};

#endif