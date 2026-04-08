#pragma once

#include "meshing/MeshData.h"
#include <string>

namespace kfusion {
namespace export_io {

class GLBExporter {
public:
    // Write Unity-ready GLB (left-handed, Y-up, 1 unit = 1 meter)
    // Returns true on success.
    static bool write(const meshing::MeshData& mesh,
                      const std::string& filepath);
};

} // namespace export_io
} // namespace kfusion
