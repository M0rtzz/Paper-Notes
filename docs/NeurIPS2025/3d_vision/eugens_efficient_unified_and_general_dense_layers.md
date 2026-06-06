---
title: >-
  [论文解读] EUGens: Efficient, Unified, and General Dense Layers
description: >-
  [NeurIPS 2025][3D视觉][efficient neural network] EUGens 提出一类新的高效稠密层，利用随机特征（Random Features）将全连接前馈层的推理复杂度从二次降到线性，统一了已有的高效 FFL 扩展…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "efficient neural network"
  - "random features"
  - "feedforward layer"
  - "Transformer"
  - "NeRF"
---

# EUGens: Efficient, Unified, and General Dense Layers

**会议**: NeurIPS 2025  
**arXiv**: [2410.09771](https://arxiv.org/abs/2410.09771)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: efficient neural network, random features, feedforward layer, Transformer, NeRF

## 一句话总结

EUGens 提出一类新的高效稠密层，利用随机特征（Random Features）将全连接前馈层的推理复杂度从二次降到线性，统一了已有的高效 FFL 扩展，在 LLM 预训练、ViT 图像分类、NeRF/iSDF 三维重建等任务中实现高达 27% 加速和 30% 参数压缩，且支持无需反向传播的层级知识蒸馏。

## 背景与动机

- 全连接前馈层（FFL）是 Transformer 和隐式神经表示（NeRF, iSDF 等）的核心组件，占据大量参数和计算
- FFL 推理时间复杂度为 O(d^2 + dl)，随隐藏层维度二次增长
- 现有加速手段（剪枝、量化、知识蒸馏、结构化矩阵）各有局限，缺乏通用的 FFL 替代方案
- Random Features 方法可以解耦权重和输入处理，但已有工作（URF/SNNK）要求激活函数有傅里叶变换，无法处理 ReLU 等无界激活函数

## 核心问题

如何设计一种通用的高效稠密层，能以线性复杂度无偏近似标准 FFL（包括任意多项式激活函数），同时保持表达能力并兼容预训练模型？

## 方法详解

### 1. EUGen 层定义

k 阶 EUGen 层解耦权重 w 和输入 x 的处理：

$$\text{EUGen}^k(\mathbf{w}, \mathbf{x}) = g(\mathbf{w})^\top f(\mathbf{x})$$

其中 x+ = [x; ||x||_2] 引入对输入范数的直接依赖，G_j^i 为随机投影矩阵，通过 Hadamard 积和拼接构造不同阶的特征映射。

### 2. 理论保证

**定理 3.1**: 对任意多项式激活函数 f，EUGens 可构造无偏估计器近似 FFL 输出。这是首个针对任意多项式激活的无偏近似结果。

**定理 3.2-3.3**: 方差公式和指数小概率集中不等式。失败概率在随机特征数 m 上指数递减。

**定理 3.4**: 通过多项式逼近，可推广到一般连续激活函数（如 ReLU, GeLU, Softplus）。

### 3. 推理复杂度

权重侧可预计算。推理时复杂度为 O(mdk^2 + ml)。实际使用 k <= 3，当 m << min(d, l) 时从 O(d^2) 降至 O(d)。

### 4. QMC 改进

用高斯正交矩阵（GOMs）替代标准高斯投影，降低估计方差。

### 5. 层级知识蒸馏

存储目标层输入输出，用 MSE 最优化 EUGen 层参数。当 G_j^i 采样自固定分布时有闭式解，无需反向传播。

## 实验关键数据

### LLM 预训练（GPT-2, 124M, OpenWebText）

EUGen 替换 FFL 后验证 loss 接近原始 GPT-2，同时显著减少推理参数。替换 6 层可减少约 30% 参数。

### ViT 图像分类（ViT-Base）

| 设置 | ImageNet Acc | 推理参数占比 |
|------|-------------|-------------|
| Vanilla ViT | 基准 | 100% |
| EUGen 替换 6 层 | 接近基准 | ~70% |
| Low-Rank 等量参数 | 明显下降 | ~70% |

EUGens 以相同参数量显著优于 Low-Rank 基线。

### NeRF 三维重建

| 方法 | PSNR | 推理加速 | 模型压缩 |
|------|------|---------|---------|
| NeRF | 基准 | 1x | 1x |
| **EUGen-NeRF** | 接近基准 | **24% 加速** | **30% 压缩** |
| Mip-NeRF 360 | 基准 | 1x | 1x |
| **EUGen-Mip-NeRF** | 接近基准 | **27% 加速** | — |

### iSDF 实时 SDF 重建

EUGen-iSDF 推理加速 **22.6%**，训练加速 5%，重建质量相当。

### 知识蒸馏

NeRF 蒸馏可在不重训练的情况下实现高达 **26%** 推理加速。

## 亮点

- 理论功底扎实：首个对任意多项式激活函数的无偏 FFL 近似
- 真正的通用组件：同一层设计无缝集成到 LLM, ViT, NeRF, iSDF 四类架构
- 闭式知识蒸馏避免了重训练，对预训练模型的即插即用价值极高
- 输入范数直接依赖扩展了标准 FFL 的表达空间

## 局限与展望

- 近似误差随替换层数累积，大规模替换可能影响性能
- 当前仅展示 k <= 2 的实验，更高阶的实用性未充分验证
- 与正交化、量化等其他压缩手段的组合效果未探索
- 在超大模型（如 70B LLM）上的验证缺失

## 与相关工作的对比

- vs **SNNK/URF**: EUGens 的特例；SNNK 需要激活函数有傅里叶变换，不适用于 ReLU
- vs **Low-Rank**: 同等参数量下 EUGens 精度显著更高
- vs **剪枝/量化**: 正交互补，可组合使用
- vs **Instant-NGP/3DGS**: 不同层面的 NeRF 加速（hash/splatting vs FFL 替换）

## 启发与关联

- 随机特征方法在深度学习加速领域的潜力被低估
- 解耦权重-输入的思路可能启发注意力层的类似加速
- 闭式蒸馏对大模型部署（边缘设备、实时推理）有直接应用价值

## 评分

- ⭐ 新颖性: 4/5 — 理论贡献（无偏多项式近似）突出，架构设计巧妙
- ⭐ 实验充分度: 4.5/5 — 横跨 NLP/CV/3D 四类任务，消融全面
- ⭐ 写作质量: 4/5 — 理论与实验平衡好，结构清晰
- ⭐ 价值: 4/5 — 通用高效层替换方案，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](../../CVPR2025/3d_vision/a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [\[CVPR 2025\] Generative Omnimatte: Learning to Decompose Video into Layers](../../CVPR2025/3d_vision/generative_omnimatte_learning_to_decompose_video_into_layers.md)
- [\[NeurIPS 2025\] Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction](locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[CVPR 2025\] Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](../../CVPR2025/3d_vision/flow-nerf_joint_learning_of_geometry_poses_and_dense_flow_within_unified_neural_.md)

</div>

<!-- RELATED:END -->
