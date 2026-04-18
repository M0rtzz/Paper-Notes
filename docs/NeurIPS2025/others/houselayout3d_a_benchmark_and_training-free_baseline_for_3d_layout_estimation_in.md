---
title: >-
  [论文解读] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild
description: >-
  [NeurIPS 2025][3D layout estimation] 提出 HouseLayout3D——首个面向大规模多层建筑的真实世界 3D layout 估计基准，以及 MultiFloor3D——一个无需训练的基线方法，通过组合现代 3D 重建和分割模型在多层建筑 layout 估计上超越现有深度学习方法。
tags:
  - NeurIPS 2025
  - 3D layout estimation
  - multi-floor buildings
  - benchmark
  - training-free
  - scene graph
---

# HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild

**会议**: NeurIPS 2025  
**arXiv**: [2512.02450](https://arxiv.org/abs/2512.02450)  
**代码**: [https://houselayout3d.github.io](https://houselayout3d.github.io)  
**领域**: 3D Vision / Scene Understanding  
**关键词**: 3D layout estimation, multi-floor buildings, benchmark, training-free, scene graph

## 一句话总结

提出 HouseLayout3D——首个面向大规模多层建筑的真实世界 3D layout 估计基准，以及 MultiFloor3D——一个无需训练的基线方法，通过组合现代 3D 重建和分割模型在多层建筑 layout 估计上超越现有深度学习方法。

## 研究背景与动机

当前 3D layout 估计模型主要在合成数据集上训练，这些数据集只包含简单的单房间或单层环境。这导致两个关键问题：

**无法处理多层建筑**：现有方法需要将场景预先分割成独立楼层再单独处理，丢失了理解楼梯等跨层结构所需的全局空间上下文

**训练数据缺乏多样性**：合成数据虽然可大规模自动生成，但缺乏真实大型建筑的复杂性——多房间、多层、非曼哈顿几何、部分开放空间等

现有数据集（SceneCAD、ASE、Stru3D 等）在多个维度上存在局限：要么不是真实世界数据，要么不支持多层，要么缺少门窗标注。HouseLayout3D 是首个在所有维度上都满足需求的基准。

## 方法详解

### 整体框架

MultiFloor3D 是一个四阶段、无需训练的 pipeline：

1. **Mesh 重建**：从 RGB 图像重建 3D mesh
2. **Layout Skeleton 提取**：从 mesh 中提取结构性几何元素
3. **Layout Prototype 拟合**：通过优化修补骨架中的缺陷
4. **Scene Graph 生成**：将 prototype 转换为最终的 3D layout

### 关键设计

1. **Mesh 重建（Stage 1）**：使用 DN-Splatter（基于 3D Gaussian Splatting）从无位姿 2D 图像获得三角 mesh 和 depth map。先用 COLMAP 估计相机位姿，结合 Metric3D depth 模型训练 3DGS，再通过 Poisson 表面重建生成 mesh。

2. **Layout Skeleton 提取（Stage 2）**：将 mesh 分为四类语义元素——结构组件（墙壁、天花板、地板、大型家具）、几何不精确表面（窗户、镜子）、物体（小型家具）、楼梯。使用 OneFormer 在输入图像上做语义分割，通过反投影将标签转移到 3D mesh，再用 superpoint 聚类做多数投票精炼。最后仅保留结构组件作为 skeleton。

3. **Layout Prototype 拟合（Stage 3）**：这是方法的核心创新。针对 skeleton 中的伪影（空洞、未观测区域），通过梯度下降优化一组 3D 多边形：

    - **$\mathcal{L}_{\text{geo}}$（几何损失）**：包含 $\mathcal{L}_{\text{prox}}$（最小化 skeleton 顶点到最近多边形的距离）和 $\mathcal{L}_{\text{empty}}$（防止多边形遮挡已知空白空间，基于相机光线与 depth 的交叉检测）
    - **$\mathcal{L}_{\text{connect}}$（连接损失）**：鼓励多边形共享边界，减少小间隙
    - **$\mathcal{L}_{\text{simple}}$（简化损失）**：惩罚非共享边的长度，促使非必要边缩小直至消除
    - **Vertex Merging**：周期性简化多边形——合并近距离顶点、RDP 算法简化边界、合并法向量相似的相近多边形
    - **地板/墙壁空洞填补**：投影物体 mesh 到最近地板平面补全地板空洞；延伸墙壁多边形到天花板/地板填补墙壁空洞

4. **Scene Graph 生成（Stage 4）**：

    - 识别建筑楼层（基于地板多边形高度聚类）
    - 为每层创建 2D 平面图（合并地板和天花板多边形）
    - 用 Hov-SG 的房间分割算法将每层分割为房间，生成以房间为节点、门/开口为边的 scene graph
    - 检测楼梯并在 scene graph 中添加跨层连接边
    - **Room Extrusion**：用 2D Constrained Delaunay Triangulation 三角化平面图，向上投射光线分配天花板，将每个地板三角形拉伸到分配的天花板平面生成封闭 3D 房间

### 损失函数 / 训练策略

MultiFloor3D 不需要训练（training-free），核心优化发生在 Stage 3 的 prototype 拟合阶段，使用梯度下降优化多边形顶点位置：

$$\mathcal{L} = \mathcal{L}_{\text{geom}} + \mathcal{L}_{\text{connect}} + \mathcal{L}_{\text{simple}}$$

其中 $\mathcal{L}_{\text{geom}} = \mathcal{L}_{\text{prox}} + \mathcal{L}_{\text{empty}}$。优化过程中，约束每个多边形的顶点保持共面，并允许多边形共享顶点。

## 实验关键数据

### HouseLayout3D 数据集统计

- 16 栋建筑，33 个独立楼层，317 个房间
- 超过 26,000 帧 RGB-D 数据
- 标注了 292 个门、379 个窗户、34 个楼梯
- 每栋建筑 1-5 层，4-40 个房间
- 每栋建筑标注耗时 4-10 小时

### 主实验（HouseLayout3D）

| 方法 | Structures F1@0.5 | Doors F1@0.5 | Windows F1@0.5 | Stairs F1@0.5 | Depth Δ₅ | Depth Δ₁₀ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| RoomFormer (per floor) | 0.24 | 0.23 | 0.07 | – | 24.9 | 32.9 |
| RoomFormer (per room) | 0.18 | 0.18 | 0.08 | – | 37.3 | 44.8 |
| SceneScript (per floor) | 0.28 | 0.23 | 0.16 | – | 22.5 | 33.8 |
| SceneScript (per room) | 0.23 | 0.31 | 0.11 | – | 23.5 | 32.9 |
| **MultiFloor3D** | **0.40** | **0.55** | **0.43** | **0.42** | **61.1** | **76.3** |

MultiFloor3D 在所有指标上大幅超越基线方法，且不需要使用 ground-truth 楼层/房间分割信息。

### ScanNet++ 实验

| 方法 | #Vertices | Depth Δ₅ | Depth Δ₁₀ |
|------|:-:|:-:|:-:|
| DN-Splatter Mesh | 354k | 84.1 | 92.6 |
| RoomFormer | **32.5** | 36.8 | 48.9 |
| SceneScript | 41.2 | 55.1 | 68.5 |
| **MultiFloor3D** | 83.1 | **67.8** | **84.7** |

### 消融实验

| 配置 | Avg F1 | #Vertices | 说明 |
|------|:-:|:-:|------|
| Input Mesh + QSlim | 0.109 | 2000 | 原始 mesh 直接简化 |
| Layout Skeleton + QSlim | 0.223 | 2000 | 仅骨架提取无拟合 |
| Layout Prototype | 0.373 | 2553 | 有拟合无 scene graph |
| **MultiFloor3D** | **0.381** | **1957** | 完整 pipeline |
| w/o prototype fitting | 0.214 | 2270 | 去掉 Stage 3 |
| w/o room segmentation | 0.359 | 2442 | 去掉房间分割 |

### 关键发现

1. **每个 stage 都有贡献**：从 skeleton（0.223）到 prototype（0.373）再到完整 pipeline（0.381），每步都带来显著改善
2. **Prototype fitting 最关键**：去掉后 F1 从 0.381 降至 0.214，降幅最大
3. **Training-free 超越训练方法**：MultiFloor3D 不使用任何训练数据，却超越了在约 10 万合成样本上训练的 RoomFormer 和 SceneScript
4. **基线方法的根本局限**：RoomFormer 和 SceneScript 只能预测矩形原语，无法表示复杂形状（如斜天花板）
5. **门和窗户检测大幅领先**：Doors F1 从 0.23/0.31 提升到 0.55，Windows F1 从 0.07/0.16 提升到 0.43
6. **唯一能预测楼梯的方法**：其他基线完全不支持楼梯检测

## 亮点与洞察

- **首个多层建筑 3D layout 基准**，填补了该领域的数据集空白
- **Training-free 方法超越训练方法**的反直觉结果，说明当前训练方法的泛化能力严重不足
- **模块化 pipeline 设计**让每个阶段都可以独立替换升级
- **Scene Graph 表示**自然支持导航等下游应用（论文展示了结合 LLM 的室内导航 demo）
- 三个几何损失函数的设计精巧：$\mathcal{L}_{\text{prox}}$ 保真、$\mathcal{L}_{\text{empty}}$ 避障、$\mathcal{L}_{\text{connect}}$ 连通、$\mathcal{L}_{\text{simple}}$ 简化

## 局限与展望

1. **运行时间长**：每个 HouseLayout3D 场景需 1-2 小时（NVIDIA RTX 4090），而 SceneScript/RoomFormer 只需 1-2 分钟
2. **室外元素干扰**：通过大窗户感知到的室外元素可能引入伪影
3. **依赖多个预训练模型**：COLMAP、Metric3D、OneFormer、DN-Splatter 等，各环节的误差会累积
4. **数据集规模有限**：16 栋建筑，与训练数据集（10 万+）相比较小
5. **不支持实时推理**：pipeline 式处理不适合实时应用场景
6. **深度估计质量依赖**：窗户和反射面的深度估计不准确直接影响后续处理

## 相关工作与启发

- 传统 Manhattan 假设方法（Scan2Bim、DuLaNet）限制了可处理场景的多样性
- RoomFormer 用 Transformer 做 2D 平面图预测，SceneScript 引入结构化场景语言，但两者都受限于合成训练数据
- Hov-SG 的房间分割算法被 MultiFloor3D 直接采用
- DN-Splatter 的 3DGS + depth 监督重建策略为后续 pipeline 提供高质量几何输入
- 该工作的核心启示：**在数据受限时，组合现有强预训练模型可能比端到端训练更有效**

## 评分
- 新颖性: ⭐⭐⭐⭐ (首个多层建筑基准 + training-free pipeline 的新范式)
- 实验充分度: ⭐⭐⭐⭐ (HouseLayout3D + ScanNet++ + 充分消融)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图示丰富)
- 价值: ⭐⭐⭐⭐ (基准贡献持久，暴露了现有方法的根本局限)
