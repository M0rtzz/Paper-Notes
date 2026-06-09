---
title: >-
  [论文解读] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses
description: >-
  [CVPR 2025][3D视觉][立体深度估计] 提出 MultiHeadDepth 和 HomoDepth 两个模型，分别通过硬件友好的多头代价体积（LayerNorm+点积近似余弦相似度 + 分组点卷积）和单应性矩阵估计网络 + 2D 矫正位置编码 (RPE) 来优化立体深度估计中代价体积和预处理的延迟…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "立体深度估计"
  - "AR眼镜"
  - "代价体积优化"
  - "单应性估计"
  - "矫正位置编码"
---

# Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses

**会议**: CVPR 2025  
**arXiv**: [2411.10013](https://arxiv.org/abs/2411.10013)  
**代码**: [https://github.com/UCI-ISA-Lab/MultiHeadDepth-HomoDepth](https://github.com/UCI-ISA-Lab/MultiHeadDepth-HomoDepth)  
**领域**: 3D视觉  
**关键词**: 立体深度估计, AR眼镜, 代价体积优化, 单应性估计, 矫正位置编码

## 一句话总结

提出 MultiHeadDepth 和 HomoDepth 两个模型，分别通过硬件友好的多头代价体积（LayerNorm+点积近似余弦相似度 + 分组点卷积）和单应性矩阵估计网络 + 2D 矫正位置编码 (RPE) 来优化立体深度估计中代价体积和预处理的延迟瓶颈，在 AR 眼镜场景下精度提升 11.8-30.3% 的同时端到端延迟降低 44.5%。

## 研究背景与动机

**领域现状**：深度估计是 AR/VR 中的基础组件，下游应用包括新视角渲染、遮挡推理、AR 物体放置等。立体深度估计因精度优势和 AR 眼镜上双目摄像头的天然适配而被广泛采用。然而 AR 眼镜的可穿戴形态面临严格的计算资源约束，需要在 100ms 内完成设备端推理。

**现有痛点**：当前 SOTA 模型（如 Meta 的 Argos）的延迟中，预处理（校准+矫正）占 30.2%，代价体积计算占 29.3%，非模型部分占据了近 60% 的总延迟。更严重的是，AR 眼镜的软性材质会导致相机发生显著弯曲（>10°），使得外参持续变化，在线矫正失败率高达 15-23%，且外参求解本身可能需要 200-2000ms。

**核心矛盾**：代价体积中的余弦相似度计算涉及逐像素的范数计算和除法，这些操作对高度优化矩阵乘法的 GPU/NPU 硬件极不友好；而矫正预处理依赖精确的相机外参，但 AR 眼镜上外参不稳定。

**本文目标** 如何在保持/提升精度的前提下，大幅降低深度估计中代价体积和预处理的延迟？

**切入角度**：(1) 对于代价体积，用 LayerNorm+点积近似余弦相似度，配合分组点卷积实现硬件加速；(2) 对于预处理，引入单应性估计头+矫正位置编码，允许模型直接接受未矫正图像输入。

**核心 idea**：将深度估计中的非ML瓶颈（余弦相似度计算和图像矫正）分别替换为硬件友好的ML操作（LayerNorm+多头点积 和 单应性估计+位置编码），使全流程可在ML加速器上高效运行。

## 方法详解

### 整体框架

论文提出两个递进式模型：(1) MultiHeadDepth：在 Argos 基础上将代价体积替换为多头代价体积，保留编解码器结构不变，聚焦代价体积延迟优化；(2) HomoDepth：在 MultiHeadDepth 基础上增加单应性估计头和 2D RPE，消除预处理需求。HomoDepth 采用多任务学习，共享编码器同时输出深度图和单应性矩阵，训练时使用同方差不确定性自动平衡两个任务的损失。

### 关键设计

1. **LayerNorm+点积近似 (LND) 代替余弦相似度**:

    - 功能：将代价体积中硬件不友好的逐像素余弦相似度替换为高效的近似操作
    - 核心思路：余弦相似度 $D_{cos}(a,b) = a \cdot b / |a||b|$ 中分母的范数计算是瓶颈。作者用 2D LayerNorm 替代逐像素的范数归一化，将向量归一化为标准分布后直接做点积。LayerNorm 将大量逐像素范数计算压缩为少量通道级统计量计算，大幅减少归一化的计算量。额外地在 LND 后添加权重层来自适应调整近似精度。
    - 设计动机：LayerNorm 在硬件和编译器中已被高度优化，且其归一化后的缓冲区可以方便地融合其他编码信息（如位置编码），为后续 RPE 的引入提供了天然接口。

2. **多头代价体积 (Multi-head Cost Volume)**:

    - 功能：进一步降低代价体积的计算复杂度并提供更丰富的匹配感知
    - 核心思路：借鉴多头注意力和 dot scale 的思想，将输入通道分为多个 head（如 $C/\text{heads}$ 通道/头），每个 head 独立做分组点积（group-wise dot product），然后通过 1×1 逐点卷积（pointwise Conv）聚合各 head 的结果。整个操作等价于 group-pointwise convolution，这是硬件高度优化的标准算子。
    - 设计动机：多头设计不仅降低了单次点积的维度从而减少计算量，还提供了多个视角来感知左右特征图的匹配关系，类似多头注意力能捕获不同子空间的信息。

3. **单应性估计头 + 2D 矫正位置编码 (RPE)**:

    - 功能：使模型直接接受未矫正的立体图像输入，消除矫正预处理需求
    - 核心思路：基于 $q_r = \frac{d_l}{d_r} H_{l \to r} q_l$ 的3D投影关系，当物体距相机较远时 $d_l/d_r \approx 1$，单应性矩阵可近似表示立体图像间的位置关系。设计一个与深度估计共享编码器的 CNN 头来估计单应性矩阵 $H$，然后将 $H$ 转化为 2D RPE：对左图像素 $q_l$ 使用标准 2D 正弦位置编码 $PE(q_l)$，对右图像素使用 $RPE(q_r) = PE(H_{l \to r} q_l)$，使得同一世界点在左右图中获得相近的位置编码值，融入 LayerNorm 归一化后的特征中参与代价体积计算。
    - 设计动机：传统矫正需要实际变换图像（会丢失边缘信息），而 RPE 将位置关系以编码形式嵌入特征中，既无信息损失又避免了图像变换的计算开销。共享编码器的多任务设计则避免了额外的延迟开销。

### 损失函数

- **深度估计损失**：$L_D(y, \hat{y}) = SL_1(y, \hat{y}) + \sum_{l=0}^{4} SL_1(\nabla y^l, \nabla \hat{y}^l)$，SmoothL1 + 多尺度梯度损失
- **单应性估计损失**：$L_H = \|weight_w(y) - weight_w(\hat{y})\|_F$，加权 Frobenius 范数，权重矩阵对角线和反对角线位用 $w=50$ 放大小值元素
- **多任务联合损失**：$L = \frac{L_H}{2\sigma_H^2} + \frac{L_D}{2\sigma_D^2} + \log \sigma_H \sigma_D$，其中 $\sigma_H, \sigma_D$ 为可训练的同方差不确定性参数

## 实验关键数据

### 主实验表（未预处理输入）

| 模型 | SceneFlow AbsRel↓ | ADT AbsRel↓ | DTU AbsRel↓ | 延迟(ms) |
|:--|:--|:--|:--|:--|
| MobileStereoNet-2D | 0.172 | 0.199 | 0.147 | - |
| Argos (CVPR2023) | 0.102 | 0.133 | 0.122 | 748.5 |
| Selective-Stereo (CVPR2024) | 0.053 | 0.082 | 0.128 | - |
| **MultiHeadDepth** | **0.091** | **0.094** | **0.101** | **598.9** |

### HomoDepth 在未矫正输入上的表现

| 方案 | DTU AbsRel↓ | 端到端延迟 |
|:--|:--|:--|
| Argos（无预处理） | 0.122 | 基准 |
| 预处理 + Argos | 0.109 | 100%基准 |
| 预处理 + MultiHeadDepth | 0.101 | ~75% |
| **HomoDepth（无需预处理）** | **0.106** | **~55.5%** |

### 量化模型结果（INT8）

| 模型 | SceneFlow AbsRel↓ | ADT AbsRel↓ | CPU延迟(ms) | GPU延迟(ms) |
|:--|:--|:--|:--|:--|
| Argos | 0.109 | 0.146 | 748.5 | 54.4 |
| **MultiHeadDepth** | **0.098** | **0.097** | **598.9** | **45.3** |

### 关键发现

- **代价体积优化全面有效**：MultiHeadDepth 在所有数据集上精度优于或持平 Argos，同时延迟降低 22.9-25.2%
- **消除预处理带来巨大延迟收益**：HomoDepth 的端到端延迟降低 44.5%，且在未矫正输入上精度仅略低于"预处理+MultiHeadDepth"
- **多任务学习进一步提升鲁棒性**：引入 RPE 后在未对齐立体输入上 AbsRel 降低 10.0-24.3%
- **量化友好**：INT8 量化后精度几乎无退化，GPU 延迟仅 45.3ms

## 亮点与洞察

1. **工程导向的系统优化**：不是简单追求精度最高，而是系统性分析延迟瓶颈（预处理30%+代价体积30%），有针对性地逐一解决
2. **ML操作替代非ML操作的思路**：将传统算法（余弦相似度、图像矫正）替换为硬件友好的ML操作（LayerNorm、CNN），使全流程可在GPU/NPU上高效运行
3. **RPE的零信息损失优势**：传统矫正会裁切图像边缘，RPE 以编码形式传递位置信息完全避免了这一问题
4. **真实AR硬件验证**：在 Meta Aria 眼镜采集的 ADT 数据集上验证，具有强工业落地意义

## 局限性与可改进方向

1. **单应性近似的局限**：$d_l/d_r \approx 1$ 假设在近距离物体上不成立，可能影响近处深度精度
2. **仅测试静态基线数据集**：AdT虽来自AR眼镜，但缺少动态弯曲的极端场景测试
3. **分辨率受限**：实验分辨率较低，高分辨率下多头代价体积的内存开销需验证
4. **未与Transformer类方法对比**：如 RAFT-Stereo 等新兴方法未纳入对比

## 相关工作与启发

- **Argos (CVPR 2023)**：本文的基础模型，Meta出品的AR深度估计SOTA，本文在此基础上做加速优化
- **Multi-head Attention (Transformer)**：多头注意力的分组计算思想被迁移到代价体积中，group-pointwise conv 是其工程化实现
- **MVSNet**：在已知相机参数下使用单应性进行多视角立体匹配，本文将其扩展到外参未知的场景
- **启发**：在边缘设备部署中，"什么操作硬件友好"应该成为方法设计的第一性原则而非事后优化

## 评分

⭐⭐⭐⭐ — 工程价值极高，系统性延迟分析+针对性优化的思路值得借鉴；多任务单应性估计消除预处理的方案新颖实用。但理论创新有限，更多是将已有技术（LayerNorm、多头注意力、单应性）巧妙组合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
