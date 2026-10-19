#version 450

layout(push_constant) uniform PushConstants {
    uint sampleCount;
    uint width;
    uint height;
    uint encodeSrgb;
    uint frameSeed;
};
layout(location = 0) out vec4 historyColor;
layout(location = 1) out vec4 fragColor;

const float PI = 3.14159265358979323846;
const float FAR_DISTANCE = 1.0e30;
const float RAY_MIN_T = 1.0e-5;
const int STACK_CAPACITY = 64; // CPU builder caps depth at 48.
const int MAX_BOUNCES = 8;

// std430: pos=0, color=16, emissive=28, roughness=32, stride=48.
struct Vertex {
    vec3 pos;
    vec3 color;
    uint emissive;
    float roughness;
};
// std430: four ints=0..15, AA=16, BB=32, stride=48.
struct BVHNode {
    int left;
    int right;
    int n;
    int index;
    vec3 AA;
    vec3 BB;
};
layout(std430, binding = 0) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer IndexBuffer { uint indices[]; };
layout(std430, binding = 2) readonly buffer TriangleBuffer { uint triangles[]; };
layout(std430, binding = 3) readonly buffer BVHBuffer { BVHNode BVHNodes[]; };
layout(binding = 4) uniform sampler2D historySampler;

struct Ray { vec3 origin; vec3 direction; };
struct Hit {
    bool found;
    float distance;
    vec3 position;
    vec3 normal;
    vec3 color;
    uint emissive;
    float roughness;
};

Hit miss(float maxDistance) {
    return Hit(false, maxDistance, vec3(0), vec3(0), vec3(0), 0u, 1.0);
}

Hit hitTriangle(uint triangleId, Ray ray, float maxDistance) {
    Vertex a = vertices[indices[3u * triangleId]];
    Vertex b = vertices[indices[3u * triangleId + 1u]];
    Vertex c = vertices[indices[3u * triangleId + 2u]];
    vec3 edge1 = b.pos - a.pos;
    vec3 edge2 = c.pos - a.pos;
    vec3 normal = cross(edge1, edge2);
    float normalLength = length(normal);
    Hit result = miss(maxDistance);
    if (normalLength <= 1.0e-20) return result;

    // Two-sided Moller-Trumbore; inclusive barycentric edges.
    vec3 p = cross(ray.direction, edge2);
    float determinant = dot(edge1, p);
    if (abs(determinant) <= 1.0e-8 * normalLength) return result;
    float inverse = 1.0 / determinant;
    vec3 fromA = ray.origin - a.pos;
    float u = dot(fromA, p) * inverse;
    if (u < 0.0 || u > 1.0) return result;
    vec3 q = cross(fromA, edge1);
    float v = dot(ray.direction, q) * inverse;
    if (v < 0.0 || u + v > 1.0) return result;
    float t = dot(edge2, q) * inverse;
    if (t <= RAY_MIN_T || t >= maxDistance) return result;

    normal /= normalLength;
    if (dot(normal, ray.direction) > 0.0) normal = -normal;
    return Hit(true, t, ray.origin + t * ray.direction, normal, a.color, a.emissive, a.roughness);
}

float hitAABB(Ray ray, vec3 lower, vec3 upper, float maxDistance) {
    float nearT = 0.0;
    float farT = maxDistance;
    for (int axis = 0; axis < 3; ++axis) {
        // A parallel ray on a slab boundary must not produce 0 * infinity (NaN).
        if (ray.direction[axis] == 0.0) {
            if (ray.origin[axis] < lower[axis] || ray.origin[axis] > upper[axis]) return -1.0;
        } else {
            float t0 = (lower[axis] - ray.origin[axis]) / ray.direction[axis];
            float t1 = (upper[axis] - ray.origin[axis]) / ray.direction[axis];
            nearT = max(nearT, min(t0, t1));
            farT = min(farT, max(t0, t1));
            if (nearT > farT) return -1.0;
        }
    }
    // Return zero for origins inside/on the box; zero is a valid hit.
    return nearT;
}

Hit hitArray(Ray ray, int first, int count, float maxDistance) {
    Hit closest = miss(maxDistance);
    for (int i = first; i < first + count; ++i) {
        Hit candidate = hitTriangle(triangles[i], ray, closest.distance);
        if (candidate.found) closest = candidate;
    }
    return closest;
}

Hit hitBVH(Ray ray) {
    Hit closest = miss(FAR_DISTANCE);
    int stack[STACK_CAPACITY];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        BVHNode node = BVHNodes[stack[--sp]];
        if (hitAABB(ray, node.AA, node.BB, closest.distance) < 0.0) continue;
        if (node.n > 0) {
            Hit candidate = hitArray(ray, node.index, node.n, closest.distance);
            if (candidate.found) closest = candidate;
            continue;
        }
        float leftT = hitAABB(ray, BVHNodes[node.left].AA, BVHNodes[node.left].BB, closest.distance);
        float rightT = hitAABB(ray, BVHNodes[node.right].AA, BVHNodes[node.right].BB, closest.distance);
        if (leftT >= 0.0 && rightT >= 0.0) {
            // Defensive fallback if a future builder violates the depth contract.
            if (sp + 2 > STACK_CAPACITY) return hitArray(ray, 0, triangles.length(), FAR_DISTANCE);
            stack[sp++] = leftT <= rightT ? node.right : node.left;
            stack[sp++] = leftT <= rightT ? node.left : node.right;
        } else if (leftT >= 0.0 || rightT >= 0.0) {
            if (sp + 1 > STACK_CAPACITY) return hitArray(ray, 0, triangles.length(), FAR_DISTANCE);
            stack[sp++] = leftT >= 0.0 ? node.left : node.right;
        }
    }
    return closest;
}

uint seed;
uint wangHash(inout uint state) {
    state = (state ^ 61u) ^ (state >> 16u);
    state *= 9u;
    state ^= state >> 4u;
    state *= 0x27d4eb2du;
    state ^= state >> 15u;
    return state;
}
float randomFloat() {
    // 24 significant bits: the float conversion can never round up to 1.0.
    return float(wangHash(seed) >> 8u) * (1.0 / 16777216.0);
}
vec3 sampleCosineHemisphere(vec3 normal) {
    float r = sqrt(randomFloat());
    float phi = 2.0 * PI * randomFloat();
    vec3 helper = abs(normal.x) > 0.9 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(helper, normal));
    vec3 bitangent = cross(normal, tangent);
    return normalize(r * cos(phi) * tangent + r * sin(phi) * bitangent +
        sqrt(max(0.0, 1.0 - r * r)) * normal);
}

vec3 pathTrace(Ray ray) {
    vec3 throughput = vec3(1.0);
    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        Hit hit = hitBVH(ray);
        if (!hit.found) return vec3(0.0);
        if (hit.emissive != 0u) return throughput * hit.color;

        // roughness is the diffuse mixture weight, NOT a GGX roughness.
        // Choose a Lambertian lobe (pdf=cos/pi) or an ideal mirror lobe.
        // The selection probability cancels the matching BSDF mixture weight.
        float diffuseWeight = clamp(hit.roughness, 0.0, 1.0);
        if (randomFloat() < diffuseWeight) {
            ray.direction = sampleCosineHemisphere(hit.normal);
        } else {
            ray.direction = normalize(reflect(ray.direction, hit.normal));
        }
        throughput *= clamp(hit.color, vec3(0.0), vec3(1.0));
        float scale = max(1.0, max(abs(hit.position.x), max(abs(hit.position.y), abs(hit.position.z))));
        ray.origin = hit.position + hit.normal * (1.0e-4 * scale);
        if (bounce >= 3) {
            float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
            if (randomFloat() >= survival) return vec3(0.0);
            throughput /= survival;
        }
    }
    return vec3(0.0);
}

vec3 linearToSrgb(vec3 value) {
    return mix(1.055 * pow(value, vec3(1.0 / 2.4)) - 0.055,
        12.92 * value, lessThanEqual(value, vec3(0.0031308)));
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    seed = (uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + frameSeed * 26699u) | 1u;
    vec2 jitter = vec2(randomFloat(), randomFloat()) - 0.5;
    vec2 uv = (gl_FragCoord.xy + jitter) / vec2(width, height);
    vec2 ndc = vec2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    ndc.x *= float(width) / float(height);
    Ray ray = Ray(vec3(0, 0, 2), normalize(vec3(ndc, -1.0)));
    vec3 sampleColor = pathTrace(ray);
    vec3 average = sampleColor;
    if (sampleCount > 1u) {
        vec3 previous = texelFetch(historySampler, pixel, 0).rgb;
        average = previous + (sampleColor - previous) / float(sampleCount);
    }
    // Keep linear HDR data in history; only the display output is clamped/encoded.
    historyColor = vec4(average, 1.0);
    vec3 displayColor = clamp(average, vec3(0.0), vec3(1.0));
    if (encodeSrgb != 0u) displayColor = linearToSrgb(displayColor);
    fragColor = vec4(displayColor, 1.0);
}
