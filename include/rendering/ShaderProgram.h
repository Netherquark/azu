#pragma once

#include <QOpenGLFunctions_3_3_Core>
#include <string>

namespace kfusion {
namespace rendering {

class ShaderProgram : protected QOpenGLFunctions_3_3_Core {
public:
    ShaderProgram();
    ~ShaderProgram();

    bool load(const char* vert_src, const char* frag_src);
    void use();
    void disuse();

    void setUniformMat4(const char* name, const float* data);
    void setUniformVec3(const char* name, float x, float y, float z);
    void setUniformFloat(const char* name, float v);
    void setUniformInt(const char* name, int v);

    unsigned int programId() const { return program_id_; }
    bool isValid() const { return program_id_ != 0; }

private:
    unsigned int program_id_ = 0;

    unsigned int compileShader(unsigned int type, const char* src);
};

} // namespace rendering
} // namespace kfusion
