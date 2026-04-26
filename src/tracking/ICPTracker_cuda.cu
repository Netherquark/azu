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

    // Block reduction using shared memory
    __shared__ float s_A[21];
    __shared__ float s_b[6];
    __shared__ float s_res;
    __shared__ int   s_inliers;
    __shared__ int   s_valid_live;
    __shared__ int   s_valid_model;
    __shared__ int   s_projected;
    __shared__ int   s_dist_filtered;
    __shared__ int   s_angle_filtered;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 21) s_A[tid] = 0;
    if (tid < 6)  s_b[tid] = 0;
    if (tid == 21) s_res = 0;
    if (tid == 22) s_inliers = 0;
    if (tid == 23) s_valid_live = 0;
    if (tid == 24) s_valid_model = 0;
    if (tid == 25) s_projected = 0;
    if (tid == 26) s_dist_filtered = 0;
    if (tid == 27) s_angle_filtered = 0;
    __syncthreads();

        // Sum within block using atomics to shared memory
    for (int i = 0; i < 21; ++i) atomicAdd(&s_A[i], local_A[i]);
    for (int i = 0; i < 6; ++i)  atomicAdd(&s_b[i], local_b[i]);
    atomicAdd(&s_res, local_res);
    atomicAdd(&s_inliers, local_inliers);
    atomicAdd(&s_valid_live, local_valid_live);
    atomicAdd(&s_valid_model, local_valid_model);
    atomicAdd(&s_projected, local_projected);
    atomicAdd(&s_dist_filtered, local_dist_filtered);
    atomicAdd(&s_angle_filtered, local_angle_filtered);
    __syncthreads();

    // One thread per block writes to global
    if (tid == 0) {
        for (int i = 0; i < 21; ++i) atomicAdd(&global_stats[i], s_A[i]);
        for (int i = 0; i < 6; ++i)  atomicAdd(&global_stats[21 + i], s_b[i]);
        atomicAdd(&global_stats[27], s_res);
        atomicAdd(&global_stats[28], (float)s_inliers);
        atomicAdd(&global_stats[29], (float)s_valid_live);
        atomicAdd(&global_stats[30], (float)s_valid_model);
        atomicAdd(&global_stats[31], (float)s_projected);
        atomicAdd(&global_stats[32], (float)s_dist_filtered);
        atomicAdd(&global_stats[33], (float)s_angle_filtered);
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

        result.pose = delta * result.pose;
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
