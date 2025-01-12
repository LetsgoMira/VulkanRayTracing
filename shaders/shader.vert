#version 440 core

layout (location = 0) in vec3 vPosition;  // cpu传入的顶点坐标

layout (location = 0) out vec3 pix;

void main() {
    gl_Position = vec4(vPosition.x,-vPosition.y,vPosition.z, 1.0);
    pix = vPosition;
}
