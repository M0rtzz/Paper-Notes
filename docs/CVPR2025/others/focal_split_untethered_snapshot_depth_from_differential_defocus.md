---
title: >-
  [论文解读] Focal Split: Untethered Snapshot Depth from Differential Defocus
description: >-
  [CVPR 2025][差分离焦深度] 受跳蛛视觉启发，构建首个无线（电池供电）的快照式差分离焦深度相机 Focal Split，用分光镜将光路分给两个不同焦距的传感器，仅需 500 FLOPs/像素和 4.9W 功率即可在树莓派上实时估计深度。 领域现状 领域现状：深度感知是机器人和AR的基础需求。主流方案包括主动方法（结…
tags:
  - "CVPR 2025"
  - "差分离焦深度"
  - "跳蛛仿生"
  - "分光镜"
  - "被动测距"
  - "低功耗边缘设备"
---

# Focal Split: Untethered Snapshot Depth from Differential Defocus

**会议**: CVPR 2025  
**arXiv**: [2504.11202](https://arxiv.org/abs/2504.11202)  
**代码**: [https://focal-split.qiguo.org](https://focal-split.qiguo.org) (有，含DIY指南)  
**领域**: 其他 / 计算摄影  
**关键词**: 差分离焦深度, 跳蛛仿生, 分光镜, 被动测距, 低功耗边缘设备

## 一句话总结

受跳蛛视觉启发，构建首个无线（电池供电）的快照式差分离焦深度相机 Focal Split，用分光镜将光路分给两个不同焦距的传感器，仅需 500 FLOPs/像素和 4.9W 功率即可在树莓派上实时估计深度。

## 研究背景与动机

### 领域现状

**领域现状**：深度感知是机器人和AR的基础需求。主流方案包括主动方法（结构光/ToF/LiDAR，功耗高）和被动方法（双目/单目DNN，计算量大）。差分离焦（DfDD）是一种优雅的被动方案——通过比较不同焦距下的图像模糊程度推断深度。

**现有痛点**：现有 DfDD 方法要么需要时序两帧（改变焦距拍两次，动态场景失败），要么需要高端工作站（如 Focal Track 需要 GPU 服务器）。没有能在边缘设备上实时运行的无线深度相机。

**核心矛盾**：DfDD 需要两个不同焦距的图像，但时序采集会引入运动模糊，空间采集（两个传感器）通常被认为标定困难且体积大。

**切入角度**：跳蛛眼睛的启发——跳蛛通过多层视网膜（不同焦距）同时获取不同焦平面的图像来感知深度。用分光镜+两个放置在不同距离的传感器模拟这一机制。

**核心 idea**：分光镜 + 双传感器不同焦距 = 500 FLOPs/像素的实时被动深度感知。

### 解决思路

**本文目标**：### 关键设计

1. **光学系统设计**:

    - 功能：同时获取两个不同焦距的图像
    - 核心思路：30mm 镜头后放一个分光镜，将入射光分成两路。


## 方法详解

### 整体框架


### 关键设计

1. **光学系统设计**:

    - 功能：同时获取两个不同焦距的图像
    - 核心思路：30mm 镜头后放一个分光镜，将入射光分成两路。两个 OV5647 传感器分别放在距镜头不同距离 $s_1$、$s_2$ 的位置。更近的传感器看到"前焦"图像，更远的看到"后焦"图像
    - 设计动机：空间分离消除了时序采集的运动伪影，分光镜保证两路图像完全同步

2. **超低计算量的深度估计**:

    - 功能：从焦距差中推断场景深度
    - 核心思路：深度 $Z = a/(b + \tilde{I}_s / \nabla^2 \tilde{I})$，其中 $\tilde{I}_s$ 是焦距方向的梯度（两传感器图像差分得到），$\nabla^2 \tilde{I}$ 是空间拉普拉斯算子。只需简单的图像差分和卷积，每像素仅 500 FLOPs
    - 设计动机：DNN 方法每像素需要百万级 FLOPs，此方法靠解析公式实现 3 个数量级的计算节省

3. **置信度度量与放大倍率校正**:

    - 功能：识别不可靠深度估计并校正因传感器距离不同导致的放大倍率差异
    - 核心思路：置信度 $C = \tilde{I}_s^2$——焦距梯度越大说明信号越强、估计越准。放大倍率校正通过仿射变换对齐两传感器图像
    - 设计动机：在纹理稀疏区域 $\tilde{I}_s \approx 0$，深度估计退化，置信度自动标记这些区域

### 损失函数 / 训练策略

无训练过程——完全是解析方法。参数 $(a, b)$ 通过光学标定确定。L=21 的 box filter 做噪声平滑。硬件成本约 $500，体积 4×5×6 cm³。

## 实验关键数据

### 主实验

| 指标 | Focal Split | Focal Flow | Focal Track |
|------|-------------|------------|-------------|
| 功耗 | **4.9W** | ~100W | ~300W |
| 计算量/像素 | **500 FLOPs** | ~10⁶ | ~10⁷ |
| 动态场景 MAE | **~42mm** | 179.25mm | 107.69mm |
| 工作距离 | 860mm | - | - |

### 关键发现
- **动态场景优势明显**：快照式采集消除了运动伪影，比时序方法在动态场景下误差小 60-75%
- **极低功耗**：4.9W 总功耗（含树莓派），首次实现电池供电的被动深度相机
- **稀疏但可靠**：丢弃低置信度 40% 像素后工作范围达 860mm

## 亮点与洞察
- **仿生设计的优雅实现**——跳蛛用多层视网膜感知深度，Focal Split 用分光镜+双传感器实现同样原理，且成本仅 $500
- **3个数量级的计算节省**——从 DNN 的百万 FLOPs 到解析公式的 500 FLOPs，真正适合边缘部署
- **完全开源的 DIY 方案**——提供 3D 打印文件、代码和组装指南，任何人都可以复现

## 局限与展望
- 仅生成稀疏深度图（纹理缺失区域无法估计）
- 工作范围受限于 SNR 衰减（~1 米）
- 需要精确光学标定
- 分辨率较低（480×360）
- 被动方法的固有局限——纹理稀疏区域失效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 仿生设计+首个无线快照深度相机，跨领域创新
- 实验充分度: ⭐⭐⭐ 真实硬件演示充分但定量评估场景有限
- 写作质量: ⭐⭐⭐⭐ 光学推导清晰，DIY 指南实用
- 价值: ⭐⭐⭐⭐ 为边缘设备深度感知开辟了极低功耗新路线


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Three-View Focal Length Recovery From Homographies](three-view_focal_length_recovery_from_homographies.md)
- [\[ICCV 2025\] Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration](../../ICCV2025/others/omni-dc_highly_robust_depth_completion_with_multiresolution_depth_integration.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](../../NeurIPS2025/others/depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](../../ICML2026/others/private_and_stable_test-time_adaptation_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
