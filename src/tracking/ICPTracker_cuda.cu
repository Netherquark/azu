#ifdef CUDA_ENABLED

#include "tracking/ICPTracker.h"
#include "sensor/KinectSensor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

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
    // Output: 21 (A) + 6 (b) + 1 (res) + 1 (inliers) = 29 floats
    float* global_stats)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float local_A[21] = {0};
    float local_b[6]  = {0};
    float local_res   = 0;
    int   local_inliers = 0;

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
                // Project into reference model
                int mx = __float2int_rd(fx * v_ref.x / v_ref.z + cx + 0.5f);
                int my = __float2int_rd(fy * v_ref.y / v_ref.z + cy + 0.5f);

                if (mx >= 0 && mx < model_w && my >= 0 && my < model_h) {
                    int midx = my * model_w + mx;
                    float3 v_model = model_vertices[midx];
                    float3 n_model = model_normals[midx];

                    if (v_model.z > 0.001f && n_model.z != 0) { // simplified validity check
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
                                // Rotate n_model back to live space
                                float3 n_model_live;
                                n_model_live.x = r00 * n_model.x + r10 * n_model.y + r20 * n_model.z;
                                n_model_live.y = r01 * n_model.x + r11 * n_model.y + r21 * n_model.z;
                                n_model_live.z = r02 * n_model.x + r12 * n_model.y + r22 * n_model.z;

                                float J[6];
                                J[0] = n_model_live.x; J[1] = n_model_live.y; J[2] = n_model_live.z;
                                J[3] = v_live.y * n_model_live.z - v_live.z * n_model_live.y;
                                J[4] = v_live.z * n_model_live.x - v_live.x * n_model_live.z;
                                J[5] = v_live.x * n_model_live.y - v_live.y * n_model_live.x;

                                // Accumulate A = J*J^T (upper triangle)
                                int count = 0;
                                for (int i = 0; i < 6; ++i) {
                                    for (int j = i; j < 6; ++j) {
                                        local_A[count++] += J[i] * J[j];
                                    }
                                    local_b[i] -= J[i] * err;
                                }
                                local_res += err * err;
                                local_inliers++;
                            }
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

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < 21; ++i) s_A[i] = 0;
        for (int i = 0; i < 6; ++i) s_b[i] = 0;
        s_res     = 0;
        s_inliers = 0;
    }
    __syncthreads();

    // Sum within block using atomics to shared memory
    for (int i = 0; i < 21; ++i) atomicAdd(&s_A[i], local_A[i]);
    for (int i = 0; i < 6; ++i)  atomicAdd(&s_b[i], local_b[i]);
    atomicAdd(&s_res, local_res);
    atomicAdd(&s_inliers, local_inliers);
    __syncthreads();

    // One thread per block writes to global
    if (tid == 0) {
        for (int i = 0; i < 21; ++i) atomicAdd(&global_stats[i], s_A[i]);
        for (int i = 0; i < 6; ++i)  atomicAdd(&global_stats[21 + i], s_b[i]);
        atomicAdd(&global_stats[27], s_res);
        atomicAdd(&global_stats[28], (float)s_inliers);
    }
}

// ---------------------------------------------------------------------------
// Host-side implementation
// ---------------------------------------------------------------------------

void ICPTracker::initGPU() {
    d_hessian_ = utils::make_cuda_unique<float>(32);
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
    float fx = sensor::FX / (1 << level);
    float fy = sensor::FY / (1 << level);
    float cx = sensor::CX / (1 << level);
    float cy = sensor::CY / (1 << level);

    float angle_thresh_cos = cosf(params_.angle_threshold * 3.14159f / 180.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        cudaMemset(d_hessian_.get(), 0, 32 * sizeof(float));

        Eigen::Matrix4f rel = ref_pose.inverse() * result.pose;
        Eigen::Matrix3f R = rel.block<3,3>(0,0);
        Eigen::Vector3f t = rel.block<3,1>(0,3);

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
            d_hessian_.get()
        );

        float h_hessian[32];
        cudaMemcpy(h_hessian, d_hessian_.get(), 32 * sizeof(float), cudaMemcpyDeviceToHost);

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
