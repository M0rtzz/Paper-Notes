---
title: >-
  [论文解读] Demeter: A Parametric Model of Crop Plant Morphology from the Real World
description: >-
  [3D视觉] Demeter 是一个数据驱动的参数化植物形态模型，将植物形态分解为拓扑、关节、形状和变形四个因素，支持形状生成、3D 重建和生物物理仿真。
tags:
  - 3D视觉
---

# Demeter: A Parametric Model of Crop Plant Morphology from the Real World

## 基本信息

- **会议**: ICCV 2025
- **arXiv**: 2510.16377
- **代码**: [项目主页](项目将开源)
- **领域**: 3D 视觉 (3D Vision)
- **关键词**: 参数化形状模型, 植物形态学, 3D 重建, PCA, 生物物理仿真

## 一句话总结

Demeter 是一个数据驱动的参数化植物形态模型，将植物形态分解为拓扑、关节、形状和变形四个因素，支持形状生成、3D 重建和生物物理仿真。

## 研究背景与动机

参数化形状模型（如人体的 SMPL、动物的 SMAL）在计算机视觉和图形学中取得了巨大成功，但植物领域缺乏同等表达能力的模型。现有植物建模方法主要依赖：

**程序化模型**（L-system 等）：需要大量手工规则，逆向建模困难

**通用 3D 重建**（NeRF/3DGS 等）：不包含形状先验，在薄结构和遮挡场景下效果差

植物建模面临独特挑战：
- **拓扑变化**：同一物种的不同个体具有不同的分枝结构（区别于固定拓扑的人体/动物模型）
- **多源变化**：形状变化来自关节（articulation）、子组件形状、非刚性变形三个独立因素
- **数据稀缺**：真实农田场景的高质量 3D 标注数据极少

## 方法详解

### 整体框架

Demeter 将植物形状表达为参数化原语图的函数：

$$(\mathbf{V}, \mathbf{F}) = \mathcal{M}_\Phi(\Gamma, \theta, \beta, \gamma)$$

其中 $\Gamma$ 为拓扑结构, $\theta$ 为关节参数, $\beta$ 为形状参数, $\gamma$ 为变形参数, $\Phi = \{\Phi_s, \Phi_d\}$ 为学习到的 PCA 基。

顶点计算公式：

$$\mathbf{v}_f = T(\theta; \Gamma) \cdot \mathcal{D}(\mathbf{v}_t + \mathcal{S}(\beta); \gamma)$$

### 四大参数组件

**1. 拓扑 ($\Gamma$)**

用树数据结构表示，$n = n_l + n_s + n_o$ 个节点，每个节点代表叶片或茎。每个节点有类型 $\text{tp}(i) \in \{\text{leaf}, \text{stem}\}$ 和父节点 $\text{pa}(i)$。区别于 SMPL 的固定拓扑，Demeter 允许同一物种的不同个体具有不同拓扑图。

**2. 关节 ($\theta$)**

每个节点的关节参数 $\theta_i = (\tau_i, d_i, s_i)$，分别表示旋转四元数、沿父茎的路径长度和缩放因子。通过正向运动学链计算全局变换：

$$\mathbf{T}_i(\theta; \Gamma) = \prod_{i' \in \text{ans}(i)} T(\tau_{i'}, d_{i'}, s_{i'})$$

**3. 形状 ($\beta$)**

定义规范空间中基于模板的偏移。对于叶片，模板为 $m_{l_1} \times m_{l_2}$ 的 2D 网格，控制点由 Catmull-Rom 样条曲线定义。使用 PCA 压缩高维参数：

$$\mathbf{v}_t + \mathcal{S}(\beta) = \Phi_s^T \beta + \mathbf{v}_t$$

其中 $\Phi_s \in \mathbb{R}^{2 m_{l_1} m_{l_2} \times |\beta|}$ 为从 2D 叶片扫描学习的 PCA 基。

**4. 变形 ($\gamma$)**

将叶片关节转化为 2D 骨架结构驱动 3D 变形。使用正向运动学将关节角度转换为变形后的 3D 位置：

$$\mathcal{D}(\mathbf{v}_j; \gamma_i) = \prod_{j' \in \text{ans}(j)} T(\tau_{j'}, d_j, 1) \cdot \mathbf{v}_j$$

同样使用 PCA 降维：$\Phi_d^T \gamma$。变形保持局部刚性（叶面积和茎长度近似不变）。

### 从真实数据学习 Demeter

数据收集：从伊利诺伊州大豆农场收集约 600 株植物的多视角 RGB 视频，使用 4DGS 进行动态重建，再以 2DGS 提取高质量网格，手动标注实例分割和拓扑。

学习过程采用多阶段策略：
1. **Stage 1**：从原始 3D 点云拟合形状参数 $\hat{\beta}_i$，通过最小化 Chamfer 距离；从对齐的变形模板学习 PCA 基 $\Phi_d$
2. **Stage 2**：给定初始关节和已估计的 $\beta_i$, $\gamma_i$，通过正向运动学拟合完整点云优化关节参数

### 3D 重建

- **多视角重建**：点云实例分割（PointTransformer-V3）→ 最小生成树推断拓扑 → 参数拟合
- **单张图像重建**：SAM 分割 → Mask-RCNN 实例预测 → 深度估计提升到 3D → 拓扑推断 → Demeter 拟合

## 实验关键数据

### 主实验：3D 重建量化对比

| Method | 平滑 | 解耦 | 可学习 | 大豆 CD ↓ | 玉米 CD ↓ | 大豆 Size (KB) ↓ | 玉米 Size (KB) ↓ |
|---|---|---|---|---|---|---|---|
| NKSR | ✓ | | | 0.0030 | 0.0023 | 5785.6 | 3686.4 |
| SimpleProc | ✓ | ✓ | | 0.0376 | 0.0557 | **0.2754** | **0.2236** |
| **Demeter (Ours)** | ✓ | ✓ | ✓ | **0.0016** | 0.0071 | 3.375 | 1.766 |

Demeter 在大豆上 CD 最优（0.0016），存储远小于 NKSR（3.375 KB vs 5785.6 KB），同时具备解耦和可学习特性。

### 消融/对比：单张图像重建

| Method | IoU ↑ |
|---|---|
| One2345++ | 0.296 |
| Meshy | 0.206 |
| **Demeter (Ours)** | **0.328** |

- One2345++ 生成粗糙形状但缺少细节
- Meshy 重建逼真但与输入图像不对齐
- Demeter 完整且忠实于输入

### 关键发现

- **跨物种泛化**：Demeter 在胡椒、玫瑰、烟草、玉米等多个物种上均能有效拟合
- **生物物理仿真**：将 Demeter 生成的网格输入 Helios 模拟器，成功模拟了一天中光合作用速率的变化
- **参数可解释性**：拓扑、关节、形状、变形四个参数可独立控制，实现灵活的形态编辑

## 亮点与洞察

1. **填补领域空白**：首个数据驱动的植物参数化形状模型，将 SMPL 的成功扩展到植物领域
2. **变拓扑处理**：巧妙地使用树结构图表示可变拓扑，解决了现有参数化模型均假设固定拓扑的限制
3. **多源变化解耦**：将形状变化因素分解为关节、形状和变形三个正交来源，比 SMPL 更加精细
4. **实际应用价值**：为作物产量分析、环境监测提供了计算基础（通过光合作用仿真等下游任务证实）

## 局限性

- 假设叶片为 2D 形状、茎为均匀粗度曲线，不适用于仙人掌、藻类、榕树等复杂结构植物
- 目前忽略花朵和果实等器官
- 3D 数据收集和标注过程耗时，扩展到新物种需要重新采集
- 单张图像重建的 IoU 仍然较低（0.328），有较大提升空间

## 相关工作与启发

- **SMPL/FLAME/SMAL** 等人体/面部/动物参数化模型提供了核心设计灵感，但 Demeter 创新性地处理了变拓扑问题
- **L-system** 等程序化建模方法适合前向生成但逆向建模困难，Demeter 的原语化设计更适合逆问题
- **Catmull-Rom 样条**曲线用于叶片形状表示是巧妙选择（控制点恰好在曲线上，易于优化）
- 可启发更多领域（如珊瑚、血管等分枝结构）的参数化建模研究

## 评分

⭐⭐⭐⭐ — 开创性工作，首次将数据驱动参数化建模从人体/动物领域扩展到植物，设计精巧且有实际农业应用价值。数据集贡献突出，但适用物种范围仍有限。

<!-- RELATED:START -->

## 相关论文

- [NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)

<!-- RELATED:END -->
