# VulkanRayTracing

使用 Vulkan Graphics Queue、片元着色器和 CPU 构建的 SAH BVH 实现静态路径追踪。
不依赖硬件 Ray Tracing 扩展。保留原项目的全屏绘制和内置房间场景。

特别感谢 [AKGWSB](https://github.com/AKGWSB) 的教程。

本次修订基于 [LetsgoMira/VulkanRayTracing](https://github.com/LetsgoMira/VulkanRayTracing)
的 `master` 快照，提交为 `72dd23d544773eef4fc6801338d229fc75387870`。

**验证状态（2026-10-19）：本次修改仅进行了源码静态复核，没有编译 C++、编译着色器或启动渲染器。
编译兼容性、Validation Layer 输出、实际画面和帧率仍待验证。**
具体静态检查记录见 [verification.json](verification.json)。

## 本次修复

| 部分 | 原问题 | 修改后的行为 |
| --- | --- | --- |
| 历史累积 | 同一图像在一次绘制中同时采样和写入；没有可靠的跨帧依赖 | 两张 RGBA32F 图像交替读写，显式同步前一帧颜色写入和历史读取 |
| 累积精度 | R8G8B8A8_UNORM 反复量化并截断亮度 | 历史保存线性浮点平均值；使用逐像素 `texelFetch`，显示编码与历史分离 |
| 首帧 | 读取未初始化历史 | 两张图像初始化为零，首个样本直接写入，不读取历史 |
| 窗口缩放 | 只重建交换链，累积图像尺寸和采样计数未更新 | 重建历史图像、描述符、帧缓冲、Render Pass、Pipeline，并重置样本计数 |
| 呈现 | 交换链附件使用 DONT_CARE store；呈现信号量按飞行帧复用 | 改为 STORE；呈现信号量按交换链图像索引复用 |
| BVH 排序 | 比较器把 `==` 写成 `=`；对三角形 ID 再次用 `triangles` 映射 | 使用稳定的三角形 ID 取顶点，按质心坐标比较，相同坐标按 ID 排序 |
| BVH 边界／栈 | 包围盒初值写死；着色器每片元使用 10000 个栈元素 | 包围盒使用正负无穷初始化；树深上限 48，遍历栈 64，超限时回退全遍历 |
| 数据布局 | CPU `bool` 与 GPU 字段宽度不一致；SSBO 分段未按设备要求对齐 | 使用 `uint32_t` / `uint`、显式 std430、ABI 静态断言和 `minStorageBufferOffsetAlignment` |
| 求交 | 包围盒平行射线可能出现 NaN；三角形严格排除边；距离写死 100 | 显式处理平行轴和盒内起点，双面 Möller–Trumbore 求交，最近距离剪枝 |
| 采样 | 混合两种方向后仍使用均匀半球 PDF，方向也未归一化 | 随机选择余弦加权 Lambertian 或理想镜面分量，权重与采样匹配 |
| 相机 | 分辨率固定为 800×600，宽高比变化导致拉伸 | 使用实际帧缓冲尺寸生成射线，像素内抖动，保持垂直视场角 |
| 启动／退出 | 默认 OBJ 不在仓库中，启动失败；屏幕缓冲未释放 | 默认模型缺失时显示内置房间；显式指定的坏模型报错；补齐异常路径和资源清理 |
| 工程配置 | SDK 路径固定为个人 D 盘路径，同时链接 GLFW 静态库和 DLL 导入库 | 通过环境变量定位依赖，默认链接 GLFW 静态库，源码使用 UTF-8 |

Debug 模式还会在上传前检查实际生成的 BVH：节点可达性、环／重复节点、深度、叶子范围、
三角形覆盖和父子包围关系。这个运行时检查已写入，但本次没有执行。

## Windows x64 使用方式

沿用 Visual Studio 2022 / v143 / C++17 工程。当前交付以 Windows x64 为目标；
保留的 Win32 配置未适配 SDK 库目录，勿直接选用。没有进行 macOS / MoltenVK 移植。

准备 Vulkan SDK、GLFW、GLM 和 tinyobjloader，然后通过环境变量或自己的 VS 属性表设置：

| 变量 | 目录含义 |
| --- | --- |
| `VULKAN_SDK` | Vulkan SDK 安装目录，包含 `Include`、`Lib` 和 `Bin/glslangValidator.exe` |
| `GLFW_ROOT` | GLFW 根目录，下面有 `include/GLFW/glfw3.h` |
| `GLFW_LIBRARY_DIR` | 与目标架构、编译器和运行库设置匹配的 `glfw3.lib` 所在目录 |
| `GLM_ROOT` | 包含 `glm/glm.hpp` 的目录 |
| `TINYOBJLOADER_ROOT` | 直接包含 `tiny_obj_loader.h` 的目录 |

设置环境变量后重新启动 Visual Studio，以便继承新值。
已移除未使用的 stb_image 依赖；tinyobjloader 的实现仍只在 `main.cpp` 定义一次。

以下步骤由你在准备验证时手动执行，交付过程中没有执行它们：

1. 在项目根目录运行 `shaders\compile.bat`。脚本可以从任意工作目录调用，内部会切换到着色器目录，失败返回非零退出码。
2. 打开 `VulkanLearn.sln`，选择 `Debug | x64`，按本机依赖配置生成程序。
3. 用项目根目录作为运行工作目录；VS 的 `LocalDebuggerWorkingDirectory` 已设置为 `$(ProjectDir)`。
4. 启动程序。Debug 配置需要 SDK 中的 `VK_LAYER_KHRONOS_validation`。

新程序仅加载：

- `shaders/pathtrace.vert.spv`
- `shaders/pathtrace.frag.spv`

这两个文件必须由新版 GLSL 生成，**本次交付没有生成它们**。
上游已有的 `vert.spv`、`frag.spv` 和实验用 `test.frag` 保留为历史文件，不被新版程序或脚本使用。
不要把旧 SPIR-V 改名充当新文件；Push Constants 和场景数据布局已经发生变化。

## 场景和材质

不提供参数时，优先加载 `models/bunny.obj`。该文件不在上游快照中，缺失时会输出提示并渲染内置房间。
也可以传入一个 OBJ 路径：

```text
VulkanLearn.exe path/to/model.obj
VulkanLearn.exe --help
```

显式传入的模型加载失败、没有三角形、顶点索引越界或坐标不合法时会报错。
OBJ 会要求 tinyobjloader 做三角化。模型仍使用原项目的变换 `p * 6 + (0.2, -1, 0)`，
粉色材质和固定相机；没有新增 OBJ 材质、贴图或平滑顶点法线支持。

`roughness` 为兼容已有字段而保留名称，本版含义是**漫反射混合比例**：

- `0`：带颜色的理想镜面反射。
- `1`：Lambertian 漫反射。
- 中间值：按该比例随机选择两个分量，避免随意混合方向却使用错误 PDF。

它不是 GGX 粗糙度，也没有 Fresnel 或微表面模型。
最多追踪 8 次命中，第四次非发光命中后使用俄罗斯轮盘终止。
有限弹射上限仍会截断更长光路；本版不宣称完整渲染方程的无偏解。
三角形求交处理了边界比较，但不是经过证明的 watertight 算法。

历史数据保持线性 HDR；显示输出在 `[0, 1]` 截断。sRGB 附件由硬件编码，
UNORM 附件由着色器编码，避免重复 gamma 转换。没有新增曝光或 tone mapping。
窗口标题显示每像素样本数；缩放后重新累积，达到 `2^24` 样本后重新开始计数，以避免整数转 float 的精度问题。

## 待运行的验收清单

这些是后续验证要求，不是已通过的测试。

| 场景 | 预期检查 |
| --- | --- |
| 无 `bunny.obj` 启动 | 显示内置房间并提示模型缺失；样本数增长；Debug BVH 校验通过 |
| 指定正常 OBJ | 三角化和 BVH 构建成功，模型显示；保留原来的模型变换 |
| 指定不存在、空、越界索引或非有限坐标的 OBJ | 明确报错，正常释放已创建资源 |
| 缺失／损坏新版 SPIR-V | 提示具体文件；不会静默加载旧二进制 |
| 连续缩放、最大化、最小化再恢复、最小化后关闭 | 无卡死，历史尺寸与窗口一致，恢复后重新累积 |
| 同一画面累计 1、32、256、1024 spp | 噪声逐步降低，观察亮度稳定性和是否存在闪烁；不要求逐像素单调收敛 |
| 启用同步验证，连续呈现与缩放 | 检查读写冲突、描述符偏移、附件布局和信号量复用；记录实际 Validation 输出 |
| Debug 正常退出、加载失败退出 | 检查 Validation 是否报告仍存活的设备对象 |
| 长宽比不同的窗口 | 场景没有因窗口拉伸而改变物体比例；视野随宽高比改变 |
| 镜面比例 0／1／中间值、平行射线、共边三角形 | 对照解析场景检查求交和材质行为，确认无 NaN、漏面或异常增亮 |

## 已知边界和后续工作

- 继续使用 Vulkan 1.0。交换链销毁／重建沿用 `vkDeviceWaitIdle` 的常见做法；它不能从规范上证明呈现引擎已结束对所有资源的使用。若需要严格的呈现生命周期保证，应新增 `VK_EXT_swapchain_maintenance1` 的 present fence 支持。
- RGBA32F 双历史图像占用约 `宽 × 高 × 32` 字节：800×600 约 14.6 MiB，4K 约 253 MiB，未计交换链和场景资源。
- 要求 RGBA32F 支持采样和颜色附件。不支持时拒绝设备；不会回退到低精度历史图像。
- SAH 构建仍采用逐轴排序，复杂大模型的启动时间尚未优化；深度上限处可能产生大叶子。
- 还没有直接光采样 / MIS、降噪、图像导出、交互相机或完整 PBR 材质。这些适合在基础正确性通过 GPU 验证之后再迭代。

同步与布局处理参考 Khronos 官方资料：
[同步示例](https://docs.vulkan.org/guide/latest/synchronization_examples.html)、
[呈现信号量复用及生命周期限制](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html)、
[Shader Memory Layout](https://docs.vulkan.org/guide/latest/shader_memory_layout.html)。
