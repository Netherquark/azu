#ifdef CUDA_ENABLED

#include "tracking/ICPTracker.h"
#include "sensor/KinectSensor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

namespace kfusion {
namespace tracking {

__device__ uchar3 float3_to_rgb(float3 c) { return make_uchar3((uint8_t)fminf(255.f, fmaxf(0.f, c.x*255.f)), (uint8_t)fminf(255.f, fmaxf(0.f, c.y*255.f)), (uint8_t)fminf(255.f, fmaxf(0.f, c.z*255.f))); }

__global__ void applyCASKernel(uchar3* d_rgb, int width, int height, float sharpness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    auto get = [&](int dx, int dy) {
        uchar3 c = d_rgb[(y+dy)*width + (x+dx)];
        return make_float3(c.x/255.f, c.y/255.f, c.z/255.f);
    };

    float3 b = get(0, -1); 
    float3 d = get(-1,  0); float3 e = get(0,  0); float3 f = get(1,  0);
    float3 h = get(0,  1); 

    float3 minRGB = make_float3(
        fminf(fminf(fminf(b.x, d.x), fminf(e.x, f.x)), h.x),
        fminf(fminf(fminf(b.y, d.y), fminf(e.y, f.y)), h.y),
        fminf(fminf(fminf(b.z, d.z), fminf(e.z, f.z)), h.z)
    );
    float3 maxRGB = make_float3(
        fmaxf(fmaxf(fmaxf(b.x, d.x), fmaxf(e.x, f.x)), h.x),
        fmaxf(fmaxf(fmaxf(b.y, d.y), fmaxf(e.y, f.y)), h.y),
        fmaxf(fmaxf(fmaxf(b.z, d.z), fmaxf(e.z, f.z)), h.z)
    );

    float3 min_p = make_float3(fminf(minRGB.x, 1.0f - maxRGB.x), fminf(minRGB.y, 1.0f - maxRGB.y), fminf(minRGB.z, 1.0f - maxRGB.z));
    float3 w = make_float3(sqrtf(min_p.x), sqrtf(min_p.y), sqrtf(min_p.z));
    w.x = w.x * sharpness * -0.125f; w.y = w.y * sharpness * -0.125f; w.z = w.z * sharpness * -0.125f;

    float3 out;
    out.x = (b.x*w.x + d.x*w.x + f.x*w.x + h.x*w.x + e.x) / (1.0f + 4.0f*w.x);
    out.y = (b.y*w.y + d.y*w.y + f.y*w.y + h.y*w.y + e.y) / (1.0f + 4.0f*w.y);
    out.z = (b.z*w.z + d.z*w.z + f.z*w.z + h.z*w.z + e.z) / (1.0f + 4.0f*w.z);

    d_rgb[y*width + x] = float3_to_rgb(out);
}

// ---------------------------------------------------------------------------
// GPU-Based Pyramid Generation (Downsampling)
// ---------------------------------------------------------------------------
__global__ void downsampleKernel(
    const float3* v_src, const float3* n_src,
    int src_w, int src_h,
    float3* v_dst, float3* n_dst,
    int dst_w, int dst_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    int sx = x * 2;
    int sy = y * 2;

    float3 v_sum = make_float3(0,0,0);
    float3 n_sum = make_float3(0,0,0);
    int count = 0;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int sidx = (sy + dy) * src_w + (sx + dx);
            float3 v = v_src[sidx];
            float3 n = n_src[sidx];
            if (v.z > 0.0f && n.x*n.x + n.y*n.y + n.z*n.z > 0.5f) {
                v_sum.x += v.x; v_sum.y += v.y; v_sum.z += v.z;
                n_sum.x += n.x; n_sum.y += n.y; n_sum.z += n.z;
                count++;
            }
        }
    }

    int didx = y * dst_w + x;
    if (count > 0) {
        v_dst[didx] = make_float3(v_sum.x/count, v_sum.y/count, v_sum.z/count);
        float len = sqrtf(n_sum.x*n_sum.x + n_sum.y*n_sum.y + n_sum.z*n_sum.z);
        if (len > 1e-6f) {
            n_dst[didx] = make_float3(n_sum.x/len, n_sum.y/len, n_sum.z/len);
        } else {
            n_dst[didx] = make_float3(0,0,0);
        }
    } else {
        v_dst[didx] = make_float3(0,0,0);
        n_dst[didx] = make_float3(0,0,0);
    }
}

// ---------------------------------------------------------------------------
// Hessian Reduction Kernel
// Parallelized over live frame pixels
// ---------------------------------------------------------------------------
__global__ void computeHessianKernel(
    const float3* live_vertices,
    const float3* live_normals,
    const float3* model_vertices,
    const float3* model_normals,
    int width, int height,
    int model_w, int model_h,
    float fx, float fy, float cx, float cy,
    float dist_thresh, float angle_thresh_cos,
    // Relative transform: ref_cam <- live_cam
    float r00, float r01, float r02,
    float r10, float r11, float r12,
    float r20, float r21, float r22,
    float tx,  float ty,  float tz,
    // World to Ref transform (for model vertices/normals)
    float rw00, float rw01, float rw02,
    float rw10, float rw11, float rw12,
    float rw20, float rw21, float rw22,
    float twx,  float twy,  float twz,
    // Output: 21 (A) + 6 (b) + 1 (res) + 1 (inliers) + 5 (diagnostics) = 34 floats
    float* global_stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float local_A[21] = {0};
    float local_b[6]  = {0};
    float local_res   = 0;
    int   local_inliers = 0;
    int   local_valid_live = 0;
    int   local_valid_model = 0;
    int   local_projected = 0;
    int   local_dist_filtered = 0;
    int   local_angle_filtered = 0;

    if (x < width && y < height) {
        int idx = y * width + x;
        float3 v_live = live_vertices[idx];
        
        if (v_live.z > 0.001f) {
            // Transform live vertex to reference camera space
            float3 v_ref;
            v_ref.x = r00 * v_live.x + r01 * v_live.y + r02 * v_live.z + tx;
            v_ref.y = r10 * v_live.x + r11 * v_live.y + r12 * v_live.z + ty;
            v_ref.z = r20 * v_live.x + r21 * v_live.y + r22 * v_live.z + tz;

            if (v_ref.z > 0.001f) {
                local_valid_live++;
                // Project into reference model
                int mx = __float2int_rd(fx * v_ref.x / v_ref.z + cx + 0.5f);
                int my = __float2int_rd(fy * v_ref.y / v_ref.z + cy + 0.5f);

                if (mx >= 0 && mx < model_w && my >= 0 && my < model_h) {
                    local_projected++;
                    int midx = my * model_w + mx;
                    float3 v_model_world = model_vertices[midx];
                    float3 n_model_world = model_normals[midx];

                    if (v_model_world.x * v_model_world.x + v_model_world.y * v_model_world.y + v_model_world.z * v_model_world.z > 1e-6f &&
                        n_model_world.x * n_model_world.x + n_model_world.y * n_model_world.y + n_model_world.z * n_model_world.z > 1e-6f) { // validity check
                        // Transform world model point/normal to reference camera space
                        float3 v_model;
                        v_model.x = rw00 * v_model_world.x + rw01 * v_model_world.y + rw02 * v_model_world.z + twx;
                        v_model.y = rw10 * v_model_world.x + rw11 * v_model_world.y + rw12 * v_model_world.z + twy;
                        v_model.z = rw20 * v_model_world.x + rw21 * v_model_world.y + rw22 * v_model_world.z + twz;
                        
                        float3 n_model;
                        n_model.x = rw00 * n_model_world.x + rw01 * n_model_world.y + rw02 * n_model_world.z;
                        n_model.y = rw10 * n_model_world.x + rw11 * n_model_world.y + rw12 * n_model_world.z;
                        n_model.z = rw20 * n_model_world.x + rw21 * n_model_world.y + rw22 * n_model_world.z;

                        local_valid_model++;
                        // Correspondence checks
                        float dx = v_ref.x - v_model.x;
                        float dy = v_ref.y - v_model.y;
                        float dz = v_ref.z - v_model.z;
                        float dist_sq = dx*dx + dy*dy + dz*dz;

                        if (dist_sq < dist_thresh * dist_thresh) {
                            float3 n_live = live_normals[idx];
                            // Rotate live normal to ref space
                            float3 n_live_ref;
                            n_live_ref.x = r00 * n_live.x + r01 * n_live.y + r02 * n_live.z;
                            n_live_ref.y = r10 * n_live.x + r11 * n_live.y + r12 * n_live.z;
                            n_live_ref.z = r20 * n_live.x + r21 * n_live.y + r22 * n_live.z;

                            float dot = n_live_ref.x * n_model.x + n_live_ref.y * n_model.y + n_live_ref.z * n_model.z;
                            if (fabsf(dot) > angle_thresh_cos) {
                                // Point-to-plane error
                                float err = n_model.x * dx + n_model.y * dy + n_model.z * dz;
                                
                                // Jacobian in current (live) camera space
                                // J = [n_model_in_live; cross(v_live, n_model_in_live)]
                                // Rotate n_model back to live space: n_model_live = R_rel^T * n_model
                                float3 n_model_live;
                                n_model_live.x = r00 * n_model.x + r10 * n_model.y + r20 * n_model.z;
                                n_model_live.y = r01 * n_model.x + r11 * n_model.y + r21 * n_model.z;
                                n_model_live.z = r02 * n_model.x + r12 * n_model.y + r22 * n_model.z;

                                float J[6];
                                J[0] = n_model_live.x; J[1] = n_model_live.y; J[2] = n_model_live.z;
                                J[3] = v_live.y * n_model_live.z - v_live.z * n_model_live.y;
                                J[4] = v_live.z * n_model_live.x - v_live.x * n_model_live.z;
                                J[5] = v_live.x * n_model_live.y - v_live.y * n_model_live.x;

                                // Huber weight for robustness (matches CPU reference)
                                float abs_err = fabsf(err);
                                float huber_k = 0.02f; 
                                float w = (abs_err <= huber_k) ? 1.0f : huber_k / abs_err;
                                float weighted_err = err * w;

                                // Accumulate A = J*J^T (upper triangle)
                                int count = 0;
                                for (int i = 0; i < 6; ++i) {
                                    for (int j = i; j < 6; ++j) {
                                        local_A[count++] += J[i] * J[j];
                                    }
                                    local_b[i] -= J[i] * weighted_err;
                                }
                                local_res += weighted_err * weighted_err;
                                local_inliers++;
                            } else {
                                local_angle_filtered++;
                            }
                        } else {
                            local_dist_filtered++;
                        }
                    }
                }
            }
        }
    }

    // Pack into array for warp reduction
    float vals[34];
    for (int i = 0; i < 21; ++i) vals[i] = local_A[i];
    for (int i = 0; i < 6; ++i)  vals[21 + i] = local_b[i];
    vals[27] = local_res;
    vals[28] = (float)local_inliers;
    vals[29] = (float)local_valid_live;
    vals[30] = (float)local_valid_model;
    vals[31] = (float)local_projected;
    vals[32] = (float)local_dist_filtered;
    vals[33] = (float)local_angle_filtered;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = (blockDim.x * blockDim.y) / 32;

    __shared__ float s_reduce[8][34]; // Up to 8 warps per block (256 threads)

    for (int i = 0; i < 34; ++i) {
        float val = vals[i];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) s_reduce[wid][i] = val;
    }
    __syncthreads();

    // The first 34 threads sum the partial warp results and do a single atomicAdd to global memory
    if (tid < 34) {
        float sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            sum += s_reduce[w][tid];
        }
        atomicAdd(&global_stats[tid], sum);
    }
}

// ---------------------------------------------------------------------------
// Host-side implementation
// ---------------------------------------------------------------------------

void ICPTracker::initGPU() {
    d_hessian_ = utils::make_cuda_unique<float>(34); // 34 floats used
}

void ICPTracker::freeGPU() {
    d_hessian_.reset();
}

ICPResult ICPTracker::trackLevelGPU(const float3*            d_v_live,
                                   const float3*            d_n_live,
                                   int                      width,
                                   int                      height,
                                   const ModelFrame&        model,
                                   const Eigen::Matrix4f&   pose_estimate,
                                   const Eigen::Matrix4f&   ref_pose,
                                   int                      level,
                                   int                      max_iter)
{
    ICPResult result;
    result.pose = pose_estimate;

    // Intrinsics
    float fx = kfusion::sensor::FX / (1 << level);
    float fy = kfusion::sensor::FY / (1 << level);
    float cx = kfusion::sensor::CX / (1 << level);
    float cy = kfusion::sensor::CY / (1 << level);

    float angle_thresh_cos = cosf(params_.angle_threshold * 3.14159f / 180.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        CUDA_CHECK(cudaMemset(d_hessian_.get(), 0, 34 * sizeof(float)));

        Eigen::Matrix4f rel = ref_pose.inverse() * result.pose;
        Eigen::Matrix3f R = rel.block<3,3>(0,0);
        Eigen::Vector3f t = rel.block<3,1>(0,3);

        Eigen::Matrix4f w2ref = ref_pose.inverse();
        Eigen::Matrix3f Rw = w2ref.block<3,3>(0,0);
        Eigen::Vector3f tw = w2ref.block<3,1>(0,3);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        computeHessianKernel<<<grid, block>>>(
            d_v_live, d_n_live, 
            model.d_vertices.get(), model.d_normals.get(),
            width, height,
            model.width, model.height,
            fx, fy, cx, cy,
            params_.dist_threshold, angle_thresh_cos,
            R(0,0), R(0,1), R(0,2),
            R(1,0), R(1,1), R(1,2),
            R(2,0), R(2,1), R(2,2),
            t.x(), t.y(), t.z(),
            Rw(0,0), Rw(0,1), Rw(0,2),
            Rw(1,0), Rw(1,1), Rw(1,2),
            Rw(2,0), Rw(2,1), Rw(2,2),
            tw.x(), tw.y(), tw.z(),
            d_hessian_.get()
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_hessian[34];
        CUDA_CHECK(cudaMemcpy(h_hessian, d_hessian_.get(), 34 * sizeof(float), cudaMemcpyDeviceToHost));

        Eigen::Matrix<float,6,6> A = Eigen::Matrix<float,6,6>::Zero();
        Eigen::Matrix<float,6,1> b = Eigen::Matrix<float,6,1>::Zero();
        int count = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                A(i,j) = A(j,i) = h_hessian[count++];
            }
            b(i) = h_hessian[21 + i];
        }
        float residual = h_hessian[27];
        int inliers = (int)h_hessian[28];
        
        // Populate diagnostics
        result.valid_live_points  = (int)h_hessian[29];
        result.valid_model_points = (int)h_hessian[30];
        result.projected_points   = (int)h_hessian[31];
        result.dist_filtered      = (int)h_hessian[32];
        result.angle_filtered     = (int)h_hessian[33];

        if (inliers < 10) break;

        // Solve update with Tikhonov regularization
        A += Eigen::Matrix<float,6,6>::Identity() * 0.1f;
        Eigen::Matrix<float,6,1> update = A.ldlt().solve(b);
        
        // Update pose (Rodrigues approximation)
        Eigen::Matrix3f dR;
        dR = Eigen::AngleAxisf(update.segment<3>(3).norm(), update.segment<3>(3).normalized()).toRotationMatrix();
        if (update.segment<3>(3).norm() < 1e-6) dR = Eigen::Matrix3f::Identity();

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = dR;
        delta.block<3,1>(0,3) = update.head<3>();

        // IMPORTANT: The twist (update) was calculated in the LIVE camera frame.
        // Therefore, we must apply it by right-multiplying the delta transform.
        // Applying it via left-multiplication would treat it as a twist in the global world frame, causing divergence.
        result.pose = result.pose * delta;
        result.inliers = inliers;
        result.error = residual / fmaxf(1.0f, (float)inliers);

        if (update.norm() < 1e-5) {
            result.converged = true;
            break;
        }
    }

    return result;
}

} // namespace tracking
} // namespace kfusion

#endif // CUDA_ENABLED
