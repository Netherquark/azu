#include "rendering/ShaderProgram.h"
#include <iostream>
#include <vector>

namespace kfusion {
namespace rendering {

ShaderProgram::ShaderProgram() {
    initializeOpenGLFunctions();
}

ShaderProgram::~ShaderProgram() {
    if (program_id_) {
        glDeleteProgram(program_id_);
        program_id_ = 0;
    }
}

unsigned int ShaderProgram::compileShader(unsigned int type, const char* src) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        int log_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
        std::vector<char> log(log_len);
        glGetShaderInfoLog(shader, log_len, nullptr, log.data());
        std::cerr << "[Shader] Compile error: " << log.data() << "\n";
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool ShaderProgram::load(const char* vert_src, const char* frag_src) {
    initializeOpenGLFunctions();

    unsigned int vert = compileShader(GL_VERTEX_SHADER,   vert_src);
    unsigned int frag = compileShader(GL_FRAGMENT_SHADER, frag_src);
    if (!vert || !frag) {
        if (vert) glDeleteShader(vert);
        if (frag) glDeleteShader(frag);
        return false;
    }

    program_id_ = glCreateProgram();
    glAttachShader(program_id_, vert);
    glAttachShader(program_id_, frag);
    glLinkProgram(program_id_);

    int success = 0;
    glGetProgramiv(program_id_, GL_LINK_STATUS, &success);
    if (!success) {
        int log_len = 0;
        glGetProgramiv(program_id_, GL_INFO_LOG_LENGTH, &log_len);
        std::vector<char> log(log_len);
        glGetProgramInfoLog(program_id_, log_len, nullptr, log.data());
        std::cerr << "[Shader] Link error: " << log.data() << "\n";
        glDeleteProgram(program_id_);
        program_id_ = 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return program_id_ != 0;
}

void ShaderProgram::use()   { glUseProgram(program_id_); }
void ShaderProgram::disuse() { glUseProgram(0); }

void ShaderProgram::setUniformMat4(const char* name, const float* data) {
    int loc = glGetUniformLocation(program_id_, name);
    if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, data);
}

void ShaderProgram::setUniformVec3(const char* name, float x, float y, float z) {
    int loc = glGetUniformLocation(program_id_, name);
    if (loc >= 0) glUniform3f(loc, x, y, z);
}

void ShaderProgram::setUniformFloat(const char* name, float v) {
    int loc = glGetUniformLocation(program_id_, name);
    if (loc >= 0) glUniform1f(loc, v);
}

void ShaderProgram::setUniformInt(const char* name, int v) {
    int loc = glGetUniformLocation(program_id_, name);
    if (loc >= 0) glUniform1i(loc, v);
}

} // namespace rendering
} // namespace kfusion
