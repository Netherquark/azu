#ifdef CUDA_ENABLED

#include "meshing/MarchingCubes.h"
#include "tsdf/VoxelGPU.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <iostream>
#include "tsdf/VoxelGPU.h"
#include "utils/CudaUniquePtr.h"

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
namespace meshing {

// ---------------------------------------------------------------------------
// Standard Marching Cubes Tables
// ---------------------------------------------------------------------------
static const int edge_table[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x835, 0xb3f, 0xa36,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

static const int tri_table[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
    {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
    {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
    {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
    {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
    {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
    {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
    {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
    {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
    {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
    {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
    {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
    {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
    {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
    {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
    {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
    {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
    {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
    {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
    {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
    {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
    {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
    {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
    {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
    {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
    {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
    {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
    {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
    {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
    {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
    {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
    {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
    {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
    {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
    {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
    {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
    {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
    {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
    {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
    {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
    {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
    {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
    {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
    {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
    {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
    {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
    {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
    {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
    {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
    {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
    {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
    {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
    {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
    {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
    {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
    {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
    {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
    {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
    {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
    {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
    {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
    {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
    {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
    {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
    {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
    {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
    {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
    {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
    {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
    {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
    {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
    {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
    {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
    {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
    {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
    {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
    {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
    {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
    {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
    {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
    {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
    {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
    {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
    {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
    {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
    {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
    {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
    {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
    {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
    {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
    {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
    {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
    {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
    {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
    {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
    {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
    {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
    {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
    {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
    {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
    {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
    {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
    {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
    {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
    {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
    {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
    {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
    {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
    {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
    {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
    {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
    {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
    {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
    {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
    {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
    {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
    {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
    {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
    {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
    {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
    {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
    {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
    {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
    {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
    {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
    {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
    {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
    {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
    {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
    {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
    {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
    {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
    {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
    {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
    {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
    {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
    {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
    {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
    {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
    {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
    {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
    {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
    {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

// ---------------------------------------------------------------------------
// Constant Tables (Copied to Device)
// ---------------------------------------------------------------------------
__constant__ int c_edge_table[256];
__constant__ int c_tri_table[256][16];
__constant__ int c_corner_offsets[8][3];
__constant__ int c_edge_corners[12][2];

// ---------------------------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------------------------

__device__ float3 interpolateEdgeGPU(float3 p1, float v1, float3 p2, float v2) {
    if (fabsf(v1) < 1e-6f) return p1;
    if (fabsf(v2) < 1e-6f) return p2;
    if (fabsf(v1 - v2) < 1e-6f) return p1;
    float t = v1 / (v1 - v2);
    return make_float3(p1.x + t * (p2.x - p1.x),
                        p1.y + t * (p2.y - p1.y),
                        p1.z + t * (p2.z - p1.z));
}

// Compute normal via central difference on GPU
__device__ float3 computeNormalGPU(void* voxels_void, int resolution, int x, int y, int z) {
    tsdf::VoxelGPU* voxels = (tsdf::VoxelGPU*)voxels_void;
    
    auto sample = [&](int xi, int yi, int zi) {
        if (xi < 0 || xi >= resolution || yi < 0 || yi >= resolution || zi < 0 || zi >= resolution)
            return 1.0f;
        tsdf::VoxelGPU& v = voxels[zi * resolution * resolution + yi * resolution + xi];
        return (v.weight <= 0.001f) ? 1.0f : v.tsdf;
    };
    
    float dx = sample(x+1, y, z) - sample(x-1, y, z);
    float dy = sample(x, y+1, z) - sample(x, y-1, z);
    float dz = sample(x, y, z+1) - sample(x, y, z-1);
    float rlen = 1.0f / sqrtf(dx*dx + dy*dy + dz*dz + 1e-9f);
    return make_float3(dx * rlen, dy * rlen, dz * rlen);
}

__global__ void classifyVoxelKernel(
    void* voxels_void, int resolution, uint32_t* tri_counts)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution - 1 || y >= resolution - 1 || z >= resolution - 1) return;

    tsdf::VoxelGPU* voxels = (tsdf::VoxelGPU*)voxels_void;

    bool valid_cube = true;
    int cube_idx = 0;
    for (int i = 0; i < 8; ++i) {
        tsdf::VoxelGPU& v = voxels[(z + c_corner_offsets[i][2]) * resolution * resolution + 
                                   (y + c_corner_offsets[i][1]) * resolution + 
                                   (x + c_corner_offsets[i][0])];
        if (v.weight <= 0.001f) {
            valid_cube = false;
            break;
        }
        if (v.tsdf < 0) cube_idx |= (1 << i);
    }
    if (!valid_cube) cube_idx = 0;

    int tri_count = 0;
    if (c_edge_table[cube_idx] != 0) {
        for (int i = 0; i < 16 && c_tri_table[cube_idx][i] != -1; i += 3) {
            tri_count++;
        }
    }
    tri_counts[z * resolution * resolution + y * resolution + x] = tri_count;
}

__global__ void generateMeshKernel(
    void* voxels_void, int resolution, float voxel_size, float3 origin,
    const uint32_t* offsets, size_t max_tris,
    float3* out_v, float3* out_n, uint8_t* out_c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution - 1 || y >= resolution - 1 || z >= resolution - 1) return;

    int idx = z * resolution * resolution + y * resolution + x;
    uint32_t offset = offsets[idx];
    uint32_t next_offset = offsets[idx + 1];
    if (offset == next_offset) return;
    
    // Drop the strict next_offset > max_tris check to allow partial emission of triangles
    if (offset >= max_tris) return;

    tsdf::VoxelGPU* voxels = (tsdf::VoxelGPU*)voxels_void;

    auto get_voxel = [&](int xi, int yi, int zi) -> tsdf::VoxelGPU& {
        return voxels[zi * resolution * resolution + yi * resolution + xi];
    };

    bool valid_cube = true;
    int cube_idx = 0;
    float corner_vals[8];
    for (int i = 0; i < 8; ++i) {
        tsdf::VoxelGPU& v = get_voxel(x + c_corner_offsets[i][0], y + c_corner_offsets[i][1], z + c_corner_offsets[i][2]);
        if (v.weight <= 0.001f) valid_cube = false;
        corner_vals[i] = v.tsdf;
        if (corner_vals[i] < 0) cube_idx |= (1 << i);
    }
    if (!valid_cube) return;

    float3 edge_v[12];
    float3 edge_n[12];
    uchar3 edge_c[12];
    for (int i = 0; i < 12; ++i) {
        if (c_edge_table[cube_idx] & (1 << i)) {
            int c1 = c_edge_corners[i][0];
            int c2 = c_edge_corners[i][1];
            float3 p1 = make_float3(origin.x + (x + c_corner_offsets[c1][0]) * voxel_size,
                                    origin.y + (y + c_corner_offsets[c1][1]) * voxel_size,
                                    origin.z + (z + c_corner_offsets[c1][2]) * voxel_size);
            float3 p2 = make_float3(origin.x + (x + c_corner_offsets[c2][0]) * voxel_size,
                                    origin.y + (y + c_corner_offsets[c2][1]) * voxel_size,
                                    origin.z + (z + c_corner_offsets[c2][2]) * voxel_size);
            edge_v[i] = interpolateEdgeGPU(p1, corner_vals[c1], p2, corner_vals[c2]);
            
            float3 n1 = computeNormalGPU(voxels_void, resolution, x+c_corner_offsets[c1][0], y+c_corner_offsets[c1][1], z+c_corner_offsets[c1][2]);
            float3 n2 = computeNormalGPU(voxels_void, resolution, x+c_corner_offsets[c2][0], y+c_corner_offsets[c2][1], z+c_corner_offsets[c2][2]);
            float t = corner_vals[c1] / (corner_vals[c1] - corner_vals[c2] + 1e-6f);
            edge_n[i] = make_float3(n1.x + t*(n2.x-n1.x), n1.y + t*(n2.y-n1.y), n1.z + t*(n2.z-n1.z));
            float len = 1.0f / sqrtf(edge_n[i].x*edge_n[i].x + edge_n[i].y*edge_n[i].y + edge_n[i].z*edge_n[i].z + 1e-9f);
            edge_n[i].x *= len; edge_n[i].y *= len; edge_n[i].z *= len;
            
            tsdf::VoxelGPU& v1 = get_voxel(x+c_corner_offsets[c1][0], y+c_corner_offsets[c1][1], z+c_corner_offsets[c1][2]);
            tsdf::VoxelGPU& v2 = get_voxel(x+c_corner_offsets[c2][0], y+c_corner_offsets[c2][1], z+c_corner_offsets[c2][2]);
            edge_c[i] = make_uchar3(
                (uint8_t)fminf(255.0f, fmaxf(0.0f, v1.r + t * (v2.r - v1.r))),
                (uint8_t)fminf(255.0f, fmaxf(0.0f, v1.g + t * (v2.g - v1.g))),
                (uint8_t)fminf(255.0f, fmaxf(0.0f, v1.b + t * (v2.b - v1.b)))
            );
        }
    }

    uint32_t out_idx_start = offset * 3;
    int v_count = 0;
    int current_tri = offset;
    for (int i = 0; i < 16 && c_tri_table[cube_idx][i] != -1; i += 3) {
        if (current_tri >= max_tris) break;
        for (int k = 0; k < 3; ++k) {
            int e = c_tri_table[cube_idx][i + k];
            out_v[out_idx_start + v_count] = edge_v[e];
            out_n[out_idx_start + v_count] = edge_n[e];
            out_c[(out_idx_start + v_count)*3+0] = edge_c[e].x;
            out_c[(out_idx_start + v_count)*3+1] = edge_c[e].y;
            out_c[(out_idx_start + v_count)*3+2] = edge_c[e].z;
            v_count++;
        }
        current_tri++;
    }
}

// ---------------------------------------------------------------------------
// Host Logic
// ---------------------------------------------------------------------------

void MarchingCubes::initGPU(int res) {
    if (res == last_resolution_ && d_voxel_tri_counts_) return;
    freeGPU();

    size_t n = (size_t)res * res * res;
    d_voxel_tri_counts_ = utils::make_cuda_unique<uint32_t>(n);
    d_voxel_offsets_    = utils::make_cuda_unique<uint32_t>(n + 1);
    
    d_mesh_vertices_ = utils::make_cuda_unique<float3>(max_triangles_ * 3);
    d_mesh_normals_  = utils::make_cuda_unique<float3>(max_triangles_ * 3);
    d_mesh_colors_   = utils::make_cuda_unique<uint8_t>(max_triangles_ * 3 * 3);

    // Tables from MarchingCubes.cpp
    const int corner_offsets[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const int edge_corners[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};

    cudaMemcpyToSymbol(c_edge_table, edge_table, 256 * sizeof(int));
    cudaMemcpyToSymbol(c_tri_table, tri_table, 256 * 16 * sizeof(int));
    cudaMemcpyToSymbol(c_corner_offsets, corner_offsets, 8 * 3 * sizeof(int));
    cudaMemcpyToSymbol(c_edge_corners, edge_corners, 12 * 2 * sizeof(int));
    
    last_resolution_ = res;
}

void MarchingCubes::freeGPU() {
    d_voxel_tri_counts_.reset();
    d_voxel_offsets_.reset();
    d_mesh_vertices_.reset();
    d_mesh_normals_.reset();
    d_mesh_colors_.reset();
}

std::shared_ptr<MeshData> MarchingCubes::extractGPU(const tsdf::TSDFVolume& volume) {
    const auto& params = volume.params();
    initGPU(params.resolution);

    size_t n = (size_t)params.resolution * params.resolution * params.resolution;
    dim3 block(8, 8, 8);
    dim3 grid((params.resolution + 7)/8, (params.resolution + 7)/8, (params.resolution + 7)/8);

    // Initialize counts to zero to prevent uninitialized memory corruption from padding voxels
    CUDA_CHECK(cudaMemset(d_voxel_tri_counts_.get(), 0, n * sizeof(uint32_t)));

    classifyVoxelKernel<<<grid, block>>>((void*)volume.getGPUVoxels(), params.resolution, d_voxel_tri_counts_.get());
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    thrust::device_ptr<uint32_t> d_counts(d_voxel_tri_counts_.get());
    thrust::device_ptr<uint32_t> d_offsets(d_voxel_offsets_.get());
    thrust::exclusive_scan(d_counts, d_counts + n, d_offsets);
    
    uint32_t total_tris;
    CUDA_CHECK(cudaMemcpy(&total_tris, d_voxel_offsets_.get() + n - 1, 4, cudaMemcpyDeviceToHost));
    uint32_t last_count;
    CUDA_CHECK(cudaMemcpy(&last_count, d_voxel_tri_counts_.get() + n - 1, 4, cudaMemcpyDeviceToHost));
    total_tris += last_count;
    
    CUDA_CHECK(cudaMemcpy(d_voxel_offsets_.get() + n, &total_tris, 4, cudaMemcpyHostToDevice));

    std::shared_ptr<MeshData> mesh = std::make_shared<MeshData>();
    if (total_tris > 0) {
        uint32_t capped_tris = (total_tris > max_triangles_) ? (uint32_t)max_triangles_ : total_tris;
        
        float3 origin = {params.origin.x(), params.origin.y(), params.origin.z()};
        generateMeshKernel<<<grid, block>>>(
            (void*)volume.getGPUVoxels(), params.resolution, params.voxel_size, origin,
            d_voxel_offsets_.get(), max_triangles_, d_mesh_vertices_.get(), d_mesh_normals_.get(), d_mesh_colors_.get()
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float3> raw_pos(capped_tris * 3);
        std::vector<float3> raw_norm(capped_tris * 3);
        std::vector<uint8_t> raw_col(capped_tris * 3 * 3);
        
        CUDA_CHECK(cudaMemcpy(raw_pos.data(), d_mesh_vertices_.get(), capped_tris * 3 * sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(raw_norm.data(), d_mesh_normals_.get(), capped_tris * 3 * sizeof(float3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(raw_col.data(), d_mesh_colors_.get(), capped_tris * 9, cudaMemcpyDeviceToHost));
        
        // Unify vertices on CPU to save memory (reduces size by ~6x)
        struct VectorHash {
            size_t operator()(const Eigen::Vector3f& v) const {
                size_t h1 = std::hash<float>{}(v.x());
                size_t h2 = std::hash<float>{}(v.y());
                size_t h3 = std::hash<float>{}(v.z());
                return h1 ^ (h2 << 1) ^ (h3 << 2);
            }
        };
        std::unordered_map<Eigen::Vector3f, uint32_t, VectorHash> global_map;
        
        mesh->positions.reserve(capped_tris);
        mesh->normals.reserve(capped_tris);
        mesh->colors.reserve(capped_tris * 3);
        mesh->indices.reserve(capped_tris * 3);

        for (uint32_t i = 0; i < capped_tris * 3; ++i) {
            Eigen::Vector3f pos(raw_pos[i].x, raw_pos[i].y, raw_pos[i].z);
            auto it = global_map.find(pos);
            if (it != global_map.end()) {
                mesh->indices.push_back(it->second);
            } else {
                uint32_t new_idx = static_cast<uint32_t>(mesh->positions.size());
                global_map[pos] = new_idx;
                mesh->positions.push_back(pos);
                mesh->normals.push_back(Eigen::Vector3f(raw_norm[i].x, raw_norm[i].y, raw_norm[i].z));
                mesh->colors.push_back(raw_col[i*3+0]);
                mesh->colors.push_back(raw_col[i*3+1]);
                mesh->colors.push_back(raw_col[i*3+2]);
                mesh->indices.push_back(new_idx);
            }
        }
        
        if (total_tris > max_triangles_) {
            std::cerr << "[MC] Reached max_triangles limit (" << max_triangles_ << "). Mesh is truncated.\n";
        }
    }

    return mesh;
}

} // namespace meshing
} // namespace kfusion

#endif // CUDA_ENABLED
