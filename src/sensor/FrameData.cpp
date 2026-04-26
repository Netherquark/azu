#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include <Eigen/Geometry>
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace sensor {

namespace {

inline float depthJumpThreshold(float depth_m) {
    return std::max(0.03f, depth_m * 0.05f);
}

void invalidatePixel(FrameData& frame, int idx) {
    frame.depth_meters[idx] = 0.0f;
    frame.vertices[idx]     = Eigen::Vector3f::Zero();
    frame.normals[idx]      = Eigen::Vector3f::Zero();
}

void suppressBackgroundShadowPixels(FrameData& frame) {
    const int W = frame.width;
    const int H = frame.height;
    std::vector<uint8_t> invalidate(frame.depth_meters.size(), 0);

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            const int idx = y * W + x;
            const float d = frame.depth_meters[idx];
            if (d <= 0.0f) continue;

            const float jump = depthJumpThreshold(d);
            const int neighbors[4] = {
                idx - 1,
                idx + 1,
                idx - W,
                idx + W
            };

            int closer_neighbors = 0;
            for (int nidx : neighbors) {
                const float dn = frame.depth_meters[nidx];
                if (dn > 0.0f && d - dn > jump) {
                    ++closer_neighbors;
                }
            }

            // Pixels significantly behind nearby foreground on multiple sides are
            // usually Kinect edge-shadow bleed rather than trustworthy background.
            if (closer_neighbors >= 2) {
                invalidate[idx] = 1;
            }
        }
    }

    for (size_t i = 0; i < invalidate.size(); ++i) {
        if (invalidate[i]) invalidatePixel(frame, static_cast<int>(i));
    }
}

void medianFilter3x3(FrameData& frame) {
    const int W = frame.width;
    const int H = frame.height;
    std::vector<float> filtered = frame.depth_meters;

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            if (frame.depth_meters[idx] <= 0.0f) continue;

            float window[9];
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float d = frame.depth_meters[(y + dy) * W + (x + dx)];
                    if (d > 0.0f) window[count++] = d;
                }
            }

            if (count >= 5) { // Need a majority to be valid
                std::sort(window, window + count);
                filtered[idx] = window[count / 2];
            }
        }
    }
    frame.depth_meters = std::move(filtered);
}

void updateVerticesFromDepth(FrameData& frame) {
    const int W = frame.width;
    const int H = frame.height;
    const float fx_inv = 1.0f / static_cast<float>(FX);
    const float fy_inv = 1.0f / static_cast<float>(FY);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < W * H; ++idx) {
        float d = frame.depth_meters[idx];
        if (d > 0.0f) {
            int x = idx % W;
            int y = idx / W;
            float vx = (static_cast<float>(x) - static_cast<float>(CX)) * fx_inv * d;
            float vy = (static_cast<float>(y) - static_cast<float>(CY)) * fy_inv * d;
            frame.vertices[idx] = Eigen::Vector3f(vx, vy, d);
        } else {
            frame.vertices[idx] = Eigen::Vector3f::Zero();
        }
    }
}

} // namespace

void buildFrameData(const uint16_t* raw_depth,
                    const uint8_t*  raw_rgb,
                    FrameData&      out,
                    float           min_depth,
                    float           max_depth)
{
    const int W = out.width;
    const int H = out.height;
    const float fx_inv = 1.0f / static_cast<float>(FX);
    const float fy_inv = 1.0f / static_cast<float>(FY);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float d = rawDepthToMeters(raw_depth[idx]);
            if (d < min_depth || d > max_depth) {
                out.depth_meters[idx] = 0.0f;
                out.vertices[idx]     = Eigen::Vector3f::Zero();
                out.normals[idx]      = Eigen::Vector3f::Zero();
            } else {
                out.depth_meters[idx] = d;
                float vx = (static_cast<float>(x) - static_cast<float>(CX)) * fx_inv * d;
                float vy = (static_cast<float>(y) - static_cast<float>(CY)) * fy_inv * d;
                out.vertices[idx] = Eigen::Vector3f(vx, vy, d);
                out.normals[idx]  = Eigen::Vector3f::Zero(); // computed separately
            }
            // Copy RGB
            out.rgb[idx * 3 + 0] = raw_rgb[idx * 3 + 0];
            out.rgb[idx * 3 + 1] = raw_rgb[idx * 3 + 1];
            out.rgb[idx * 3 + 2] = raw_rgb[idx * 3 + 2];
        }
    }

    suppressBackgroundShadowPixels(out);
    medianFilter3x3(out);
    updateVerticesFromDepth(out);
    computeNormals(out);
}

void computeNormals(FrameData& frame) {
    const int W = frame.width;
    const int H = frame.height;

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int c  = y * W + x;
            int r  = y * W + (x + 1);
            int l  = y * W + (x - 1);
            int u  = (y - 1) * W + x;
            int d  = (y + 1) * W + x;

            if (frame.depth_meters[c] <= 0.0f ||
                frame.depth_meters[r] <= 0.0f ||
                frame.depth_meters[l] <= 0.0f ||
                frame.depth_meters[u] <= 0.0f ||
                frame.depth_meters[d] <= 0.0f) {
                frame.normals[c] = Eigen::Vector3f::Zero();
                continue;
            }

            const float dc = frame.depth_meters[c];
            const float jump = depthJumpThreshold(dc);
            if (std::abs(dc - frame.depth_meters[r]) > jump ||
                std::abs(dc - frame.depth_meters[l]) > jump ||
                std::abs(dc - frame.depth_meters[u]) > jump ||
                std::abs(dc - frame.depth_meters[d]) > jump) {
                frame.normals[c] = Eigen::Vector3f::Zero();
                continue;
            }

            Eigen::Vector3f dx = frame.vertices[r] - frame.vertices[l];
            Eigen::Vector3f dy = frame.vertices[d] - frame.vertices[u];
            Eigen::Vector3f n  = dx.cross(dy);
            float len = n.norm();
            if (len > 1e-6f)
                frame.normals[c] = n / len;
            else
                frame.normals[c] = Eigen::Vector3f::Zero();
        }
    }
}

// Simple 2x box filter for downsampling depth + vertex + normal
static FrameData downsample(const FrameData& src) {
    FrameData dst;
    dst.width  = src.width  / 2;
    dst.height = src.height / 2;
    int n = dst.width * dst.height;
    dst.vertices.assign(n, Eigen::Vector3f::Zero());
    dst.normals.assign(n, Eigen::Vector3f::Zero());
    dst.depth_meters.assign(n, 0.0f);
    dst.rgb.assign(n * 3, 0);

    const int SW = src.width;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dst.height; ++y) {
        for (int x = 0; x < dst.width; ++x) {
            int sx = x * 2;
            int sy = y * 2;

            int idx_dst = y * dst.width + x;

            // Collect valid samples from 2x2 block
            Eigen::Vector3f sum_v = Eigen::Vector3f::Zero();
            Eigen::Vector3f sum_n = Eigen::Vector3f::Zero();
            float sum_d = 0.0f;
            int count = 0;

            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int idx_src = (sy + dy) * SW + (sx + dx);
                    float d = src.depth_meters[idx_src];
                    if (d > 0.0f) {
                        sum_v += src.vertices[idx_src];
                        sum_n += src.normals[idx_src];
                        sum_d += d;
                        ++count;
                    }
                }
            }

            if (count > 0) {
                float inv = 1.0f / static_cast<float>(count);
                dst.vertices[idx_dst]     = sum_v * inv;
                dst.depth_meters[idx_dst] = sum_d * inv;
                float nlen = sum_n.norm();
                dst.normals[idx_dst] = (nlen > 1e-6f) ? Eigen::Vector3f(sum_n / nlen) : Eigen::Vector3f::Zero();
            }

            // Nearest neighbor for RGB
            int src_idx = sy * SW + sx;
            dst.rgb[idx_dst * 3 + 0] = src.rgb[src_idx * 3 + 0];
            dst.rgb[idx_dst * 3 + 1] = src.rgb[src_idx * 3 + 1];
            dst.rgb[idx_dst * 3 + 2] = src.rgb[src_idx * 3 + 2];
        }
    }

    return dst;
}

void buildFramePyramid(const FrameData& full_res, FramePyramid& pyramid) {
    // Level 0: copy full res
    pyramid.levels[0] = full_res;

    // Level 1: half res
    pyramid.levels[1] = downsample(pyramid.levels[0]);

    // Level 2: quarter res
    pyramid.levels[2] = downsample(pyramid.levels[1]);
}

} // namespace sensor
} // namespace kfusion
