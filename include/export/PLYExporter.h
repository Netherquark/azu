#pragma once

#include "meshing/MeshData.h"
#include <string>

namespace kfusion {
namespace export_io {

class PLYExporter {
public:
    // Write binary PLY. Returns true on success.
    static bool writeBinary(const meshing::MeshData& mesh,
                            const std::string& filepath);

    // Write ASCII PLY. Returns true on success.
    static bool writeASCII(const meshing::MeshData& mesh,
                           const std::string& filepath);
};

} // namespace export_io
} // namespace kfusion
