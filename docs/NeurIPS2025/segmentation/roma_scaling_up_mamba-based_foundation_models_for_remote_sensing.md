---
title: >-
  [论文解读] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing
description: >-
  [NeurIPS 2025][图像分割][遥感基础模型] 提出RoMA——首个面向遥感领域的Mamba架构自监督自回归预训练框架，通过自适应旋转编码策略和多尺度token预测机制，解决遥感图像的方向多样性和尺度极端变化问题，验证了Mamba在遥感领域遵循数据和参数缩放定律。
tags:
  - NeurIPS 2025
  - 图像分割
  - 遥感基础模型
  - Mamba
  - 自回归预训练
  - 旋转不变性
  - 多尺度预测
---

# RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing

**会议**: NeurIPS 2025  
**arXiv**: [2503.10392](https://arxiv.org/abs/2503.10392)  
**代码**: [GitHub](https://github.com/VisionXLab/RoMA)  
**领域**: 图像分割  
**关键词**: 遥感基础模型, Mamba, 自回归预训练, 旋转不变性, 多尺度预测

## 一句话总结

提出RoMA——首个面向遥感领域的Mamba架构自监督自回归预训练框架，通过自适应旋转编码策略和多尺度token预测机制，解决遥感图像的方向多样性和尺度极端变化问题，验证了Mamba在遥感领域遵循数据和参数缩放定律。

## 研究背景与动机

遥感基础模型（RSFMs）近年来取得了巨大进展，主要依赖于ViT架构的自监督预训练（如MAE）。然而ViT的自注意力机制具有**二次复杂度**，在处理高分辨率遥感图像时面临严重的计算瓶颈——例如DOTA数据集中的4000×4000像素图像会产生海量token。

Mamba架构凭借**线性复杂度**成为有前景的替代方案，已在遥感下游任务中展现高效推理能力。但现有遥感Mamba应用仅限于小规模监督训练，未能利用大规模无标注遥感数据的潜力。

将自回归预训练应用于遥感Mamba面临三个独特挑战：

**信息稀疏与分布不均**：遥感图像中前景目标稀疏分布在复杂背景中（如机场跑道中的飞机）

**方向任意性**：俯视角度下目标可呈任意朝向，不像自然图像受重力约束

**尺度极端变化**：遥感图像中目标尺度差异极大，从建筑物到车辆跨越数个数量级

此外，Mamba的自回归预训练能否像ViT+MAE那样随数据量和模型规模增长而稳定提升性能，也是一个未经验证的关键问题。

## 方法详解

### 整体框架

RoMA采用自回归预训练范式，将图像划分为patch序列后进行next-token prediction。Mamba编码器处理完整图像计算所有token的Key和Value，然后通过可学习的Query向量与KV交互，计算预测损失。在此基础上引入自适应旋转编码策略和多尺度预测策略两个核心创新。

**为什么用自回归而非MAE？** MAE的掩码操作会破坏Mamba线性扫描所依赖的token序列连续性。自回归模型构建顺序依赖关系，天然契合Mamba的逐token扫描机制。

### 关键设计

1. **自适应旋转编码策略（Adaptive Rotation Encoding Strategy, ARES）**: 针对遥感图像中目标方向任意且分布稀疏的问题，ARES分五步实现：(1) 将输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 分割为 $N = (H \times W)/p^2$ 个不重叠patch；(2) 使用特征描述子（如LBP）为每个patch计算分数；(3) 选择得分最高的patch $\text{token}_{\text{top}}$；(4) 以该patch为中心扩展为 $L \in \{96, 64, 32\}$ 的候选区域，选择平均特征值高于全图均值的区域；(5) 对选中区域进行随机旋转，裁剪内接正方形替换原始区域。同时引入可学习的角度嵌入，作为隐式方向先验帮助模型学习旋转不变表征。

2. **多尺度预测策略（Multi-scale Prediction Strategy, MSP）**: 由于自回归方法将2D图像展平为1D序列，单向建模会破坏遥感图像中垂直方向和远距离空间关系的信息。MSP在标准token级MSE损失之外，将相邻token聚合为更大尺度的块（如 $6 \times 6$ 像素），在这些更高尺度上也进行next-block prediction。总损失为：$\ell(\theta) = \frac{1}{K-1}\sum_{k=2}^{K}\|\hat{x}_k - x_k\|_2^2 + \frac{\lambda}{N-1}\sum_{n=2}^{N}\|\hat{y}_n - y_n\|_2^2$，其中 $K$ 是token总数，$N$ 是聚合块数。大尺度信息帮助Mamba捕获更完整的目标结构。

3. **缩放定律验证（Scaling Laws）**: 系统性地验证了Mamba在遥感领域的缩放行为：(a) 数据量缩放——在62.5K到4M数据量上预训练Mamba-B，性能随数据量增长持续提升，无明显瓶颈；(b) 模型规模缩放——训练Tiny/Small/Base/Large四种变体，更大模型一致取得更优结果。

### 损失函数 / 训练策略

在OpticalRS-4M数据集上预训练Mamba-B，输入196×196，patch size 16，AdamW优化器，cosine学习率调度，初始lr=1.5e-4，batch size 256，训练400 epochs。多尺度损失中权重 $\lambda=0.1$，使用 $6 \times$ 的聚合尺度效果最佳。

## 实验关键数据

### 主实验

**三个下游任务性能对比**:

| 方法 | 骨干 | 参数量 | AID分类(OA) | UCM分类(OA) | OSCD变化检测(F1) | SpaceNet分割(mF1) |
|------|------|--------|------------|------------|-----------------|------------------|
| MAE | ViT-B | 86M | 84.21 | 52.75 | - | - |
| ARM | Mamba-B | 85M | 81.14 | 50.41 | 47.28 | 77.89 |
| RVSA | ViT-B+RVSA | 86M | 84.06 | 50.86 | 50.28 | 79.56 |
| SatMAE++ | ViT-L | 307M | 85.98 | 55.72 | 53.10 | 79.21 |
| MA3E | ViT-B | 86M | 85.86 | 55.69 | - | - |
| **RoMA** | Mamba-B | 85M | **87.36** | **59.45** | **55.63** | **79.50** |

RoMA以85M参数量超越了307M参数的SatMAE++（ViT-L）。

### 消融实验

| 配置 | AID OA(TR=20%) | AID OA(TR=50%) | 说明 |
|------|---------------|---------------|------|
| Baseline (ARM) | 69.59 | 76.80 | 无ARES和MSP |
| +ARES | 71.70 | 78.00 | 加入旋转编码，+1.2% |
| +ARES+MSP | **72.69** | **79.16** | 完整RoMA，+2.4% |

**高分辨率扩展性**:

| 分辨率 | RoMA-B显存(MB) | ViT-B显存(MB) | RoMA-B速度(s/s) | ViT-B速度(s/s) |
|--------|---------------|--------------|----------------|---------------|
| 1248 | 6526 | 24531 | 11.43 | 4.99 |
| 2048 | 16934 | OOM | 4.37 | OOM |
| 4096 | 66357 | OOM | 1.15 | OOM |

### 关键发现

- Mamba在遥感领域确实遵循数据和参数缩放定律
- RoMA-B在1248分辨率下推理速度是ViT-B的2.29倍，显存仅为26.6%
- ViT-B在2048分辨率以上直接OOM，而RoMA可稳定扩展到4096
- 特征描述子的选择（LBP vs HOG vs Wavelet）对结果影响不大，说明ARES的核心价值在于框架设计
- 过多的多尺度信息（如2×+4×+6×）反而降低性能，单尺度（6×）效果最佳

## 亮点与洞察

- 首次系统验证Mamba的自回归预训练在遥感领域可行，填补了重要空白
- "高价值区域旋转"策略巧妙：不是随机旋转整图，而是自适应找到信息密集区域进行旋转增强
- 多尺度预测策略简洁有效，缓解了1D序列建模对2D空间关系的损失
- 实验中RoMA以小参数量超越大模型的结果具有很强的实用价值

## 局限与展望

- 像素级任务（语义分割）上Mamba的优势不如分类和检测任务明显
- Mamba-Large的训练不够充分（仅300 epochs），未能展示大模型的全部潜力
- 自回归预训练天然偏向patch级目标预测，对精细像素级任务可能需要额外适配
- 仅在光学RGB遥感数据上验证，多光谱/SAR等模态的表现未知

## 相关工作与启发

- 与ARM（Mamba自回归预训练的先驱工作）相比，RoMA通过遥感特定设计取得显著提升
- MA3E提出了角度因子融入MIM训练的思路，RoMA将其拓展到自回归框架并结合自适应裁剪
- 遥感领域的多模态基础模型（如SkySense、AnySat）是未来重要方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个遥感Mamba自回归预训练框架，旋转编码和多尺度设计有创意
- 实验充分度: ⭐⭐⭐⭐ 缩放实验系统性强，消融全面，但Large模型训练不充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述有力
- 价值: ⭐⭐⭐⭐⭐ 为遥感领域的高效基础模型提供了实用方案

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

<!-- RELATED:END -->
