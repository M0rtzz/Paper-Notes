---
title: >-
  [论文解读] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection
description: >-
  [CVPR 2025][自动驾驶][V2X协同感知] 本文构建了首个包含 LiDAR、相机和 4D 雷达三种模态的 V2X 仿真数据集 V2X-R，提出了协同 LiDAR-4D 雷达融合流水线及 Multi-modal Denoising Diffusion (MDD) 模块，利用天气鲁棒的 4D 雷达特征指导扩散模型去噪含噪 LiDAR 特征，在雾天/雪天条件下提升检测性能高达 5.73%/6.70% 且几乎不影响正常天气性能。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "V2X协同感知"
  - "4D毫米波雷达"
  - "LiDAR融合"
  - "扩散去噪"
  - "恶劣天气"
---

# V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection

**会议**: CVPR 2025  
**arXiv**: [2411.08402](https://arxiv.org/abs/2411.08402)  
**代码**: [https://github.com/ylwhxht/V2X-R](https://github.com/ylwhxht/V2X-R)  
**领域**: 自动驾驶  
**关键词**: V2X协同感知, 4D毫米波雷达, LiDAR融合, 扩散去噪, 恶劣天气

## 一句话总结

本文构建了首个包含 LiDAR、相机和 4D 雷达三种模态的 V2X 仿真数据集 V2X-R，提出了协同 LiDAR-4D 雷达融合流水线及 Multi-modal Denoising Diffusion (MDD) 模块，利用天气鲁棒的 4D 雷达特征指导扩散模型去噪含噪 LiDAR 特征，在雾天/雪天条件下提升检测性能高达 5.73%/6.70% 且几乎不影响正常天气性能。

## 研究背景与动机

**领域现状**：V2X（车路协同）感知通过多智能体间的信息共享，能有效扩展感知范围和解决遮挡问题。当前研究主要集中在 LiDAR 单模态和 LiDAR-Camera 双模态融合。LiDAR 提供精确的 3D 几何信息，Camera 提供细粒度语义，二者融合已取得良好效果。

**现有痛点**：LiDAR 和 Camera 都对恶劣天气极其敏感——雾、雪、雨会严重降低 LiDAR 点云质量（散射噪声增多、检测距离缩短）和相机图像质量（能见度下降）。虽然 4D 毫米波雷达具有全天候感知能力和多普勒速度信息，但现有 V2X 数据集（OPV2V、V2X-Sim 等）均不包含 4D 雷达数据，导致协同 LiDAR-4D 雷达融合完全无人探索。

**核心矛盾**：多智能体通信虽然解决了恶劣天气下单智能体检测距离缩短的问题（通过共享其他智能体的数据），但同时也放大了噪声——多个智能体的含噪 LiDAR 特征叠加后噪声更密集。传统的点级去噪耗时且与特征级融合策略不兼容，直接的特征级去噪又因为天气噪声分布复杂多变而难以拟合。

**本文目标**：(1) 构建首个包含 4D 雷达的 V2X 数据集；(2) 提出协同 LiDAR-4D 雷达融合流水线；(3) 设计 MDD 模块利用 4D 雷达的天气鲁棒性来去噪 LiDAR 特征。

**切入角度**：4D 雷达的毫米波信号能轻易穿透雾雪中的颗粒物，天生抗恶劣天气。将 4D 雷达特征作为"干净条件"指导扩散模型对 LiDAR 特征去噪——先用扩散过程将复杂的天气噪声分布重参数化为高斯分布（更容易拟合），再用 4D 雷达特征作为条件进行多步去噪。

**核心 idea**：通过高斯重参数化将天气噪声的复杂分布转化为易拟合的高斯分布 + 用天气鲁棒的 4D 雷达 BEV 特征作为条件引导 U-Net 去噪器多步去噪含噪 LiDAR BEV 特征。

## 方法详解

### 整体框架

V2X-R 系统包含三类智能体（自车 E、协同车 C、路侧单元 I），每个智能体采集 LiDAR 和 4D 雷达点云。融合流水线四阶段：(1) **逐智能体编码**：共享编码器分别提取各智能体各模态的 BEV 特征 $\mathcal{F}_j^i$。(2) **智能体融合**：相同模态的多智能体特征通过 agent-fusion 模块（如自注意力）融合为多智能体特征 $\mathcal{F}_\mathcal{A}^i$。(3) **模态融合**：先用 MDD 模块对含噪 LiDAR 特征去噪得到 $\tilde{\mathcal{F}_\mathcal{A}^L}$，再与 4D 雷达特征融合。(4) **检测头**：预测 3D 边界框。

### 关键设计

1. **V2X-R 数据集**:

    - 功能：提供首个包含 LiDAR/Camera/4D 雷达的 V2X 协同感知数据集
    - 核心思路：基于 CARLA + OpenCDA 仿真平台构建。每个智能体配备 64 线 LiDAR（120m 范围）、4 个 RGB 相机、1 个 4D 雷达（150m 范围、30° 垂直 FOV）。数据集包含 12,079 场景、37,727 帧 LiDAR/4D 雷达点云、150,908 张图像、170,859 个 3D 车辆框标注。通过物理反射和几何光学方法模拟雾天和雪天的 LiDAR 退化。分析显示多智能体协同后 4D 雷达的实例覆盖率和点数显著提升，尤其在中远距离
    - 设计动机：缺乏 4D 雷达的 V2X 数据集是该领域研究的根本瓶颈

2. **Multi-modal Denoising Diffusion (MDD) 模块**:

    - 功能：利用天气鲁棒的 4D 雷达特征指导去噪含噪 LiDAR BEV 特征
    - 核心思路：分两步。**扩散过程**（重参数化）：对智能体融合后的 LiDAR 特征 $\mathcal{F}_{init}$ 应用 $\mathcal{T}$ 步高斯扩散 $\mathcal{F}_\mathcal{T} = \sqrt{\bar{\alpha}_\mathcal{T}} \mathcal{F}_{init} + \sqrt{1-\bar{\alpha}_\mathcal{T}} \epsilon$，将原始复杂天气噪声分布 $\delta_{raw}$ 变换为接近高斯的 $\delta_{gau} \sim \mathcal{N}(\sqrt{\bar{\alpha}_\mathcal{T}} \delta_{raw}, \sqrt{1-\bar{\alpha}_\mathcal{T}})$。**去噪过程**：在每步去噪中，将当前特征与 4D 雷达特征 $\mathcal{F}_\mathcal{A}^R$ 在通道维度拼接，送入 U-Net 去噪器 $\mathcal{F}_{t-1} = U_\theta([\mathcal{F}_t, \mathcal{F}_\mathcal{A}^R], t)$，经 $\mathcal{T}$ 步后输出去噪后的干净 LiDAR 特征 $\tilde{\mathcal{F}_\mathcal{A}^L} = \mathcal{F}_0$
    - 设计动机：恶劣天气下的 LiDAR 噪声分布复杂多变（不同天气、不同距离、不同密度），直接拟合极其困难。扩散过程的重参数化将复杂分布"平滑"为接近高斯的分布，大大降低了去噪模型的学习难度。4D 雷达特征作为条件提供了"干净参考"，帮助去噪器辨别哪些是噪声

3. **自适应损失权重调度**:

    - 功能：平衡去噪学习和检测学习的训练进度
    - 核心思路：MDD 损失 $\mathcal{L}_{MDD} = \mathcal{L}_{MSE}(\tilde{\mathcal{F}_\mathcal{A}^L}, \mathcal{F}_l^L) \cdot \gamma(e, \psi)$，其中 $\mathcal{F}_l^L$ 是从干净 LiDAR 点云（屏蔽天气噪声后）提取的 ground truth 特征。权重 $\gamma(e, \psi) = (1 - \tanh(\frac{e}{\tau} - \varphi)) \cdot \psi$ 随训练 epoch 非线性递减——早期模型专注于去噪学习，后期逐渐转向检测任务
    - 设计动机：如果去噪损失权重固定过高，会影响检测头的训练；固定过低则去噪效果不足。非线性递减策略让两个任务的学习进度自然衔接

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{all} = \beta_{cls} \mathcal{L}_{cls} + \beta_{loc} \mathcal{L}_{loc} + \mathcal{L}_{MDD}$。使用 Adam 优化器，$lr=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$。训练/验证/测试划分为 8,084/829/3,166 帧。检测范围 $x \in [0,140]m$, $y \in [-40,40]m$。广播范围 70m。提供了两种融合实现方式：SA2MA（从单智能体多模态扩展到多智能体）和 SM2MM（从多智能体单模态扩展到多模态）。

## 实验关键数据

### 主实验

**协同 LiDAR 基线** (Testing, IoU=0.3/0.5/0.7)：

| 方法 | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|------|---------|---------|---------|
| AttFuse (LiDAR only) | 91.21 | 89.51 | 80.01 |
| AdaFusion (LiDAR only) | 92.72 | 91.64 | 84.81 |
| AttFuse (LiDAR+4D Radar) | 91.50 | 90.04 | 82.44 |
| AdaFusion (LiDAR+4D Radar) | **93.44** | **92.43** | **86.09** |

**MDD 模块在恶劣天气下的效果** (AttFuse, IoU=0.5)：

| 天气 | 无 MDD | 有 MDD | 提升 |
|------|--------|--------|------|
| 正常 | 89.51 | ~89.4 | 几乎不变 |
| 雾天 | ~83.8 | ~89.5 | **+5.73%** |
| 雪天 | ~82.8 | ~89.5 | **+6.70%** |

### 消融实验

**不同融合实现方式对比** (Testing, IoU=0.7)：

| 策略 | 方法来源 | mAP |
|------|---------|-----|
| SA2MA | InterFusion | 69.63 |
| SA2MA | L4DR | 82.26 |
| SM2MM | AttFuse | 82.44 |
| SM2MM | AdaFusion | 86.09 |

### 关键发现

- LiDAR-4D 雷达融合在正常天气下相比纯 LiDAR 就有稳定的 1-2% 提升，说明 4D 雷达的多普勒和额外几何信息确实有价值
- MDD 模块在恶劣天气下带来 5-7% 的巨大提升，同时在正常天气下几乎零损失——这是通过重参数化实现的：正常天气下 $\delta_{raw} \approx 0$，扩散-去噪过程近似恒等变换
- SM2MM 策略整体优于 SA2MA，说明先在各模态内做多智能体融合再跨模态融合比反过来更有效
- 多智能体协同显著提升了 4D 雷达的实例覆盖率（尤其中远距离），弥补了 4D 雷达单智能体分辨率低的弱点
- MDD 的扩散步数较少（文中 MDD 本身就是轻量级特征空间扩散）即可有效去噪

## 亮点与洞察

- 首次在 V2X 协同感知中引入 4D 雷达模态，填补了数据集空白
- MDD 模块的设计非常巧妙——用重参数化解决"天气噪声分布难拟合"的核心问题，将不确定的复杂噪声变为确定性更强的高斯噪声
- "正常天气几乎零损失"的性质使 MDD 可以作为即插即用模块，无需根据天气条件切换模型
- 4D 雷达在多智能体场景下的价值得到了系统性验证——多智能体协同弥补了 4D 雷达单点分辨率低的天然缺陷

## 局限与展望

- V2X-R 为仿真数据集，真实世界的传感器噪声和通信延迟可能带来额外挑战
- 目前仅检测车辆一类目标，需扩展到行人、自行车等更多类别
- 4D 雷达仅提供前视数据（受限于传感器 FOV），360° 全向感知需要额外雷达布局
- 天气模拟的物理精度可能与真实天气条件有差异
- 未来可探索：(1) 在真实 4D 雷达数据集上验证 MDD；(2) 引入通信带宽约束下的自适应融合；(3) 扩展到更多恶劣条件（如大雨、沙尘）

## 相关工作与启发

- 与 OPV2V/V2X-Sim 等数据集的区别在于新增了 4D 雷达模态和恶劣天气模拟
- MDD 的扩散去噪思路可推广到其他需要处理分布不确定噪声的传感器融合场景（如水下声呐、医学超声）
- 4D 雷达作为"天气锚点"指导其他模态去噪的范式，对多传感器鲁棒感知有广泛启发

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个 V2X 4D 雷达数据集 + 扩散去噪融合的新范式
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多种融合策略、多种天气、多种评估指标的全面 benchmark
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集/方法/实验三部分安排合理
- **价值**: ⭐⭐⭐⭐⭐ — 开创了 V2X 4D 雷达融合研究方向，数据集和 benchmark 将惠及社区

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2025\] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection](sparsealign_a_fully_sparse_framework_for_cooperative_object_detection.md)

</div>

<!-- RELATED:END -->
