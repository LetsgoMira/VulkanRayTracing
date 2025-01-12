#version 440
layout(push_constant) uniform PushConstants {
    int frameCounter;
};

layout(location = 0) in vec3 pix;

layout(location = 0) out vec4 changeColor;
layout(location = 1) out vec4 fragColor;

#define PI 3.1415926535

struct vertex{
    vec3 pos;
    vec3 color;
    bool emissive;
    float roughness;
};

struct BVHNode {
    int left, right;    
    int n, index;      
    vec3 AA, BB;       
};

layout(binding = 0) buffer vertexBuffer {
    vertex vertices[]; 
};
layout(binding = 1) buffer indexBuffer {
    uint indices[]; 
};
layout(binding = 2) buffer triangleBuffer {
    uint triangles[]; 
};
layout(binding = 3) buffer BVHBuffer {
    BVHNode BVHNodes[]; 
};
layout(binding = 4) uniform sampler2D changeSampler;

struct Ray {
    vec3 startPoint;
    vec3 direction;
};

struct Triangle {
    vec3 p1, p2, p3;   
    vec3 color;    
    bool emissive;
    float roughness;
};


struct HitResult {
    bool isHit;             
    bool isInside;//内部击中
    float distance;
    vec3 hitPoint;
    vec3 normal;
    vec3 viewDir;//击中此点的光线方向          
    vec3 color;
    bool emissive;
    float roughness;
};

// 光线和三角形求交 
HitResult hitTriangle(Triangle triangle, Ray ray) {
    HitResult res;
    res.distance = 100;
    res.isHit = false;
    res.isInside = false;
    res.emissive=false;

    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 p3 = triangle.p3;

    vec3 S = ray.startPoint;    // 射线起点
    vec3 d = ray.direction;     // 射线方向
    vec3 N = normalize(cross(p2-p1, p3-p1));    // 法向量

    // 从三角形背后（模型内部）击中
    if (dot(N, d) > 0.0f) {
        N = -N;   
        res.isInside = true;
    }

    // 如果视线和三角形平行
    if (abs(dot(N, d)) < 0.00001f) return res;

    // 距离
    float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
    if (t < 0.0005f) return res;    // 如果三角形在光线背面

    // 交点计算
    vec3 P = S + d * t;

    // 判断交点是否在三角形中
    vec3 c1 = cross(p2 - p1, P - p1);
    vec3 c2 = cross(p3 - p2, P - p2);
    vec3 c3 = cross(p1 - p3, P - p3);
    bool r1 = (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0);
    bool r2 = (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0);

    // 命中，封装返回结果
    if (r1 || r2) {
        res.roughness=triangle.roughness;
        res.color=triangle.color;
        res.emissive=triangle.emissive;
        res.isHit = true;
        res.hitPoint = P;
        res.distance = t;
        res.normal = N;
        res.viewDir = d;
        // 根据交点位置插值顶点法线
        float alpha = (-(P.x-p2.x)*(p3.y-p2.y) + (P.y-p2.y)*(p3.x-p2.x)) / (-(p1.x-p2.x-0.00005)*(p3.y-p2.y+0.00005) + (p1.y-p2.y+0.00005)*(p3.x-p2.x+0.00005));
        float beta  = (-(P.x-p3.x)*(p1.y-p3.y) + (P.y-p3.y)*(p1.x-p3.x)) / (-(p2.x-p3.x-0.00005)*(p1.y-p3.y+0.00005) + (p2.y-p3.y+0.00005)*(p1.x-p3.x+0.00005));
        float gama  = 1.0 - alpha - beta;
        res.normal = N;
    }

    return res;
}

Triangle getTriangle(int i) {
    Triangle t;

    t.p1 = vertices[indices[3*triangles[i]]].pos;
    t.p2 = vertices[indices[3*triangles[i]+1]].pos;
    t.p3 = vertices[indices[3*triangles[i]+2]].pos;
    
    t.color = vertices[indices[3*triangles[i]]].color;
    t.emissive=vertices[indices[3*triangles[i]]].emissive;
    t.roughness=vertices[indices[3*triangles[i]+2]].roughness;

    return t;
}


HitResult hitArray(Ray ray, int l, int r) {
    HitResult res;
    res.isHit = false;
    res.emissive=false;
    res.distance = 100;
    for(int i=l; i<=r; i++) {
        Triangle triangle = getTriangle(i);
        HitResult r = hitTriangle(triangle, ray);
        if(r.isHit && r.distance<res.distance) {
            res = r;
        }
    }
    return res;
}

float hitAABB(Ray r,vec3 AA,vec3 BB){
    vec3 invdir = 1.0 / r.direction;

    vec3 f = (BB - r.startPoint) * invdir;
    vec3 n = (AA - r.startPoint) * invdir;

    vec3 tmax = max(f, n);
    vec3 tmin = min(f, n);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

HitResult hitBVH(Ray ray){
    HitResult res;
    res.emissive=false;
    res.isHit = false;
    res.distance = 100;

    int stack[10000];
    int sp = 0;
    stack[sp++] = 0;
    while(sp>0){
        int top=stack[--sp];
        BVHNode node=BVHNodes[top];
        if(node.n>0){
            int L=node.index;
            int R=L+node.n-1;
            HitResult r=hitArray(ray,L,R);
            if(r.isHit && r.distance<res.distance) res = r;
            continue;
        }
        float d1 = 0; // 左盒子距离
        float d2 = 0; // 右盒子距离
        if(node.left>0) {
            BVHNode leftNode = BVHNodes[node.left];
            d1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if(node.right>0) {
            BVHNode rightNode = BVHNodes[node.right];
            d2 = hitAABB(ray, rightNode.AA, rightNode.BB);
        }
        // 在最近的盒子中搜索
        if(d1>0 && d2>0) {
            if(d1<d2) { // d1<d2, 左边先
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            } else {    // d2<d1, 右边先
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        } else if(d1>0) {   // 仅命中左边
            stack[sp++] = node.left;
        } else if(d2>0) {   // 仅命中右边
            stack[sp++] = node.right;
        }
    }
    return res;
}

uint seed = uint(
    uint((pix.x * 0.5 + 0.5) * 800)  * uint(1973) + 
    uint((pix.y * 0.5 + 0.5) * 600) * uint(9277) + 
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float rand() {
    return float(wang_hash(seed)) / 4294967296.0;
}
// 半球均匀采样
vec3 SampleHemisphere() {
    float z = rand();
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * rand();
    return vec3(r * cos(phi), r * sin(phi), z);
}

// 将向量 v 投影到 N 的法向半球
vec3 toNormalHemisphere(vec3 v, vec3 N) {
    vec3 helper = vec3(1, 0, 0);
    if(abs(N.x)>0.999) helper = vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

vec3 pathTracing(Ray ray,int maxBounce){
    vec3 history = vec3(1);
    while(maxBounce-->0){
        HitResult res=hitBVH(ray);
        if(!res.isHit) return vec3(0);
        if(res.emissive) return res.color*history;
        vec3 ref=normalize(reflect(ray.direction,res.normal));
        vec3 random = toNormalHemisphere(SampleHemisphere(), res.normal);
        vec3 wi=mix(ref,random,res.roughness);
        float pdf=1.0/(2.0*PI);
        float cosine=max(0,dot(wi,res.normal));
        vec3 f_r=res.color/PI;
        history*=(f_r*cosine/pdf);
        ray.startPoint=res.hitPoint;
        ray.direction=wi;
    }
    return vec3(0);
}

void main()
{
    Ray ray;
    ray.startPoint = vec3(0, 0, 2);
    vec3 dir = vec3(pix.x+(rand()-0.5)/800.0,pix.y+(rand()-0.5)/600,1)-ray.startPoint;
    //vec3 dir = vec3(pix.xy,2) - ray.startPoint;
    ray.direction = normalize(dir);
    vec3 color=vec3(0);
    color=pathTracing(ray,3);
    vec3 lastColor=texture(changeSampler,vec2(pix.x*0.5+0.5,0.5-pix.y*0.5)).rgb;
    //color=mix(lastColor,color,1.0/float(frameCounter));
    color=(lastColor*(frameCounter-1)+color)/float(frameCounter);
    fragColor=vec4(color,1.0);
    changeColor=vec4(color,1.0);
}

