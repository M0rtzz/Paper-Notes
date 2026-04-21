---
title: >-
  [论文解读] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation
description: >-
  [NeurIPS 2025][自动驾驶][3D语义场景生成] 提出首个"草图→3D户外语义场景"生成任务与基准数据集 SketchSem3D，并设计 CymbaDiff（Cylinder Mamba Diffusion）去噪网络，通过柱坐标扫描+笛卡尔扫描的双路 Mamba 块实现结构化空间建模，在 FID 上比 3D Latent Diffusion 低 75%、比 3D DiT 低 71%。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 3D语义场景生成
  - 草图引导
  - 扩散模型
  - 状态空间模型
  - 柱坐标Mamba
---

# CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.13245](https://arxiv.org/abs/2510.13245)  
**代码**: [https://github.com/Lillian-research-hub/CymbaDiff](https://github.com/Lillian-research-hub/CymbaDiff)  
**领域**: autonomous_driving / 3D场景生成  
**关键词**: 3D语义场景生成, 草图引导, 扩散模型, 状态空间模型, 柱坐标Mamba  

## 一句话总结
提出首个"草图→3D户外语义场景"生成任务与基准数据集 SketchSem3D，并设计 CymbaDiff（Cylinder Mamba Diffusion）去噪网络，通过柱坐标扫描+笛卡尔扫描的双路 Mamba 块实现结构化空间建模，在 FID 上比 3D Latent Diffusion 低 75%、比 3D DiT 低 71%。

## 研究背景与动机

**领域现状**：3D 户外语义场景生成近年受到关注，UrbanDiff 等方法依赖 BEV 地图作为条件输入，但 BEV 缺少细粒度 3D 结构信息，限制了语义丰富度和几何保真度。同时多尺度方法需反复在多分辨率合成，计算复杂度高。

**现有痛点**：(a) 缺乏公开大规模标准基准——UrbanDiff 使用自建预处理数据集，无法公平比较；(b) 草图引导方法仅限于单物体或简单室内场景；(c) 笛卡尔坐标系中相邻体素序列可能错误表示空间邻近关系，影响序列建模效果。

**核心矛盾**：户外大场景空间结构复杂、语义多样，现有生成方法既缺合适数据集，也缺考虑圆柱连续性和垂直层次的空间编码策略。

**切入角度**：(a) 构建首个草图+伪标注卫星图→3D GT 的大规模 benchmark；(b) 设计结合柱坐标和笛卡尔坐标双重扫描的 SSM 模块来保持空间一致性。

**核心idea**：Cylinder Mamba Block——在柱坐标下按 (θ, r, z) 排序做 Mamba 扫描以保持角度-径向连续性，再融合笛卡尔三方向 Mamba 以保留精确几何距离关系。

## 方法详解

### 整体框架
输入为草图（Canny 边缘图）和伪标注卫星图 PSA，经 Scene Structure Estimation Network (SSEN) 提取粗结构，再通过 Latent Mapping Network 编码到 VAE 潜空间。CymbaDiff 去噪网络在潜空间进行扩散去噪，最终解码为 $256 \times 256 \times 32$ 的体素语义网格。

### 关键设计

1. **SketchSem3D 数据集构建**：

    - 功能：首个面向草图引导 3D 户外场景生成的大规模 benchmark。
    - 核心思路：利用 KITTI/KITTI-360 的 GPS 信息获取卫星图，CLIP 编码类别描述文本 + SAM 提取 mask-level 嵌入，余弦相似度匹配生成 PSA 伪标注。草图由 Canny 边缘检测从 BEV 投影获取。
    - 规模：Sketch-based SemanticKITTI 58987 帧 + Sketch-based KITTI-360 36057 帧，总计 95044 帧，远超 UrbanDiff 的 34149 帧（NuScenes），体素分辨率 $256^2 \times 32$ vs $192^2 \times 16$。

2. **Scene Structure Estimation Network (SSEN)**：

    - 功能：产生目标 3D 场景的粗结构先验，加速扩散收敛。
    - 核心思路：多尺度特征提取模块（级联 $3 \times 3 \times 3$ 卷积代替大核）+ Dimensional Decomposition Residual (DDR) 块——将 $k^3$ 3D 卷积分解为 $1 \times 1 \times k$、$1 \times k \times 1$、$k \times 1 \times 1$ 三步，参数从 $C_{in} \times C_{out} \times k^3$ 降至 $C_{in} \times C_{out} \times 3k$。
    - 设计动机：粗结构引导让扩散模型在早期生成步骤就朝合理几何方向走。

3. **VAE / Latent Mapping Network**：

    - 编码器将输入体素网格空间分辨率缩小 $f=4$ 倍，用交叉熵 + Lovász-Softmax 联合训练，避免 L2 导致的模糊。

4. **CymbaDiff 去噪网络——Cylinder Mamba Block**：

    - 功能：核心去噪模块，融合笛卡尔和柱坐标表示增强空间一致性。
    - 核心思路：
        - **Triple Mamba 层**（笛卡尔空间）：对残差 LayerNorm 后的特征 $z_{TMB}(t)$ 进行前向 $\psi_i^f$、反向 $\psi_i^b$、随机切片间 $\psi_i^u$ 三方向 SSM 扫描，输出 $\psi_i(z_{TMB}) = \psi_i^f + \psi_i^b + \psi_i^u$。
        - **C-Mamba 层**（柱坐标空间）：将体素按 $(θ, r, z)$ 角度-径向-垂直排序后做同样三方向扫描 $\omega_i(z_{CMB})$，输出映射回笛卡尔空间。
        - **融合**：$\psi_i^{all} = \text{MLP}(\text{LN}(\psi_i)) + \psi_i + \text{MLP}(\text{LN}(\omega_i)) + \omega_i$。
    - SSM 离散化公式：$h(t) = \bar{A} h(t-1) + \bar{B} z(t),\; y(t) = \bar{C} h(t)$，其中 $\bar{A} = \exp(\Delta A)$。
    - 设计动机：笛卡尔保留精确几何距离关系，柱坐标提供以车辆为中心的角度-径向语义连贯性，二者互补。

5. **Cross-Scale Contextual Block (CSCB) & Dilated Decomposed Conv Block (DDCB)**：

    - CSCB：级联 $3 \times 3 \times 3$ 卷积多尺度提取 + skip connection + 残差。
    - DDCB：使用膨胀率 1/2/3 的 DDR 块捕获不同尺度上下文。

### 训练与评估
- VAE 阶段用交叉熵 + Lovász-Softmax 训练。
- CymbaDiff 阶段在 VAE 潜空间进行扩散去噪训练。
- 评估指标：3D FID 和 MMD，比 UrbanDiff 所用 2D FID 更能衡量体素空间的几何保真度。

## 实验关键数据

### 主实验——3D 语义场景生成

| 数据集 | 方法 | 条件 | FID ↓ | MMD ↓ |
|--------|------|------|-------|-------|
| SemanticKITTI | SSD | - | 112.82 | - |
| SemanticKITTI | Semcity | - | 56.55 | - |
| SemanticKITTI | 3D Latent Diffusion | SK+PSA | 165.65 | 0.09 |
| SemanticKITTI | 3D DiT | SK+PSA | 138.86 | 0.08 |
| **SemanticKITTI** | **CymbaDiff** | **SK+PSA** | **40.67** | **0.04** |
| KITTI-360 | 3D Latent Diffusion | SK+PSA | 330.86 | 0.12 |
| KITTI-360 | 3D DiT | SK+PSA | 272.83 | 0.11 |
| **KITTI-360** | **CymbaDiff** | **SK+PSA** | **107.53** | **0.08** |

跨数据集泛化：在 SemanticKITTI 训练、直接迁移 KITTI-360（16 个重叠类），仍保持最优。

### 消融实验

| 配置 | FID ↓ | MMD ↓ | 说明 |
|------|-------|-------|------|
| w/o CSCB | 90.53 | 0.06 | 去掉跨尺度上下文块，FID 翻倍 |
| w/o DDCB | 76.57 | 0.06 | 去掉膨胀分解卷积块 |
| w/o C-Mamba | 74.09 | 0.05 | 仅保留笛卡尔 Triple Mamba |
| **CymbaDiff (full)** | **40.67** | **0.04** | 完整模型 |

### 3D 语义场景补全

在 SemanticKITTI 验证集上，CymbaDiff 仅用草图+PSA 条件就达到 IoU 43.2%、mIoU 14.6%，超越 MonoScene/TPVFormer/NDC-Scene/OccFormer 等使用真实单目/双目 RGB 输入的方法。

### 关键发现
- CSCB 对性能影响最大（去掉后 FID 从 40.67→90.53，涨 122%）。
- 柱坐标 C-Mamba 单独贡献也显著（去掉后 FID 涨 82%），说明柱坐标扫描对户外场景建模有独特价值。
- 跨数据集零样本泛化能力强，说明 CymbaDiff 学到了通用空间结构。

## 亮点与洞察
- **柱坐标+笛卡尔双路 Mamba** 是本文最核心创新——巧妙利用户外驾驶场景以车辆为中心的辐射特性，在柱坐标下做序列扫描保持角度连续性，避免笛卡尔体素序列化时的空间关系失真。
- **DDR 分解**将 3D 卷积参数量降至原来的 $3k/k^3$ ≈ 1%~3%，适合大规模 3D 场景。
- **草图+PSA 的条件组合**比 BEV 更容易获取（测试时只需手绘草图+卫星图），提高了实用性。

## 局限与展望
- 对小物体（行人、交通标志等）的重建较弱，因为训练数据中小类样本稀疏。
- PSA 伪标注基于卫星图（~2025）与 GT（~2013）存在时间差，可能引入语义偏移。
- 仅在 KITTI 家族数据上验证，未涉及 NuScenes/Waymo 等不同传感器配置的泛化。

## 相关工作与启发
- **vs UrbanDiff**：UrbanDiff 用 BEV 条件+纯笛卡尔扩散，分辨率仅 $192^2 \times 16$，FID 291.4；CymbaDiff 用草图+PSA+柱坐标 Mamba，$256^2 \times 32$，FID 40.67。
- **vs Semcity/SSD**：传统场景生成不支持条件控制；CymbaDiff 允许用户通过草图灵活指定场景布局。
- **vs SegMamba**：SegMamba 用于 SSC 判别任务；CymbaDiff 首次将 SSM 应用于 3D 生成任务。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 定义新任务 + 新基准 + 柱坐标 Mamba 去噪的独创设计
- 实验充分度: ⭐⭐⭐⭐ 生成+补全两个任务，消融全面，但仅限 KITTI 家族
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 开源数据集+代码对社区有较大价值，但更偏 benchmark 贡献

<!-- RELATED:START -->

## 相关论文

- [SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [Decoupled Diffusion Sparks Adaptive Scene Generation](../../ICCV2025/autonomous_driving/decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [Neurosymbolic Diffusion Models](neurosymbolic_diffusion_models.md)
- [FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)
- [X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)

<!-- RELATED:END -->
