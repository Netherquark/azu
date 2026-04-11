#pragma once

#include "tsdf/TSDFVolume.h"
#include "tracking/ICPTracker.h"

namespace kfusion {
namespace app {

/**
 * User-tunable fusion settings (depth gating, TSDF grid, ICP).
 * Depth range is applied when building frame geometry and mirrored into ICP params.
 */
struct FusionHyperparams {
    float min_depth = 0.05f;
    float max_depth = 8.0f;

    tsdf::TSDFParams    tsdf{};
    tracking::ICPParams icp{};

    static FusionHyperparams defaults() { return FusionHyperparams{}; }
};

/** Copy top-level depth clip range into ICP params (keeps tracking consistent with geometry). */
inline void syncIcpDepthFromRange(FusionHyperparams& h) {
    h.icp.min_depth = h.min_depth;
    h.icp.max_depth = h.max_depth;
}

} // namespace app
} // namespace kfusion
