---
title: >-
  [论文解读] Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism
description: >-
  [NeurIPS 2025][图像生成][异构协作感知] 提出GenComm——一种基于生成式通信机制的异构多智能体协作感知方法，通过空间消息提取和条件扩散模型在ego端生成对齐的协作者特征，无需修改原始网络即可以极低代价接纳新异构智能体。
tags:
  - NeurIPS 2025
  - 图像生成
  - 异构协作感知
  - 生成式通信
  - 条件扩散模型
  - BEV特征生成
  - 可扩展性
---

# Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism

**会议**: NeurIPS 2025  
**arXiv**: [2510.19618](https://arxiv.org/abs/2510.19618)  
**代码**: [有](https://github.com/jeffreychou777/GenComm)  
**领域**: 自动驾驶 / 协作感知  
**关键词**: 异构协作感知, 生成式通信, 条件扩散模型, BEV特征生成, 可扩展性

## 一句话总结

提出GenComm——一种基于生成式通信机制的异构多智能体协作感知方法，通过空间消息提取和条件扩散模型在ego端生成对齐的协作者特征，无需修改原始网络即可以极低代价接纳新异构智能体。

## 研究背景与动机

多智能体协作感知通过信息共享提升单个智能体的感知能力，但实际场景中不同智能体采用不同传感器和模型，产生域差距（domain gap）问题。现有方法分为两类：

**适配型方法**（如MPDA、PnPDA、STAMP、HEAL）：通过适配器或反向对齐来转换特征，但存在侵入式重训练问题，会破坏已建立的语义一致性

**重建型方法**（如CodeFilling）：通过共享码本索引重建特征，但扩展新智能体的计算代价高

这两类方法的核心限制在于：(1) 侵入式地修改编码器或核心模块破坏了智能体间的语义一致性；(2) 接纳新智能体时需要较高的计算与参数开销，可扩展性差。本文提出的核心问题是：如何以最小代价将新智能体融入协作，同时保持现有智能体间的语义一致性？

## 方法详解

### 整体框架

GenComm的核心思路是：每个ego智能体利用接收到的空间消息在本地**生成**协作者的特征，确保生成特征与ego语义空间对齐，同时保留协作者的空间信息。框架包含三个核心组件：

- **Deformable Message Extractor**：从BEV特征中提取空间消息
- **Spatial-Aware Feature Generator**：基于条件扩散模型生成对齐特征
- **Channel Enhancer**：在通道维度精炼生成特征

整体流程：各智能体用各自编码器提取BEV特征 → 消息提取器压缩为空间消息并传输 → ego端用扩散模型以消息为条件生成协作者特征 → 通道增强后融合 → 解码输出。

### 关键设计

**Deformable Message Extractor**：采用可变形卷积（deformable convolution），不仅关注单像素还动态参考周围像素，增强前景/背景区分能力。通过偏移预测网络预测采样偏移量，用加权可变形卷积提取空间信息，并使用可学习的resizer适配不同分辨率。提取的消息维度为 $C' \times H_j \times W_j$，远小于原始中间特征，降低了通信量。

**Spatial-Aware Feature Generator**：采用条件扩散模型，对初始特征（从ego特征初始化）进行前向加噪，然后以接收的空间消息作为条件输入，通过条件U-Net逐步去噪生成与ego语义空间对齐的特征。生成过程直接用MSE损失监督：

$$\mathcal{L}_{gen} = \sum_{j \in \mathcal{G}_i} \|\hat{\mathcal{F}}_j - \mathcal{F}_j\|_2^2$$

**Channel Enhancer**：引入PConv操作增强信息性元素，结合门控机制抑制冗余通道信息，并使用通道注意力强调关键特征。将特征沿通道维度分为可修改部分和静态部分，通过深度可分离卷积和注意力机制进行精炼。

### 损失函数 / 训练策略

采用两阶段训练策略：

- **阶段一（同构训练）**：端到端训练，损失函数 $\mathcal{L}_{stage1} = \alpha_1 \mathcal{L}_{cls} + \alpha_2 \mathcal{L}_{reg} + \alpha_3 \mathcal{L}_{gen}$，其中分类用focal loss，回归用smooth L1 loss，生成用MSE loss
- **阶段二（异构扩展）**：仅微调轻量级消息提取器，解决异构智能体间空间信息的数值不一致问题。损失函数 $\mathcal{L}_{stage2} = \alpha_1 \mathcal{L}_{cls} + \alpha_2 \mathcal{L}_{reg}$

接纳新智能体时，只需为其初始化并微调一个轻量提取器，不修改ego核心模块。

## 实验关键数据

### 主实验

| 融合网络 | 方法 | OPV2V-H LP64-LS32 AP50/AP70 | OPV2V-H LP64-CE AP50/AP70 | DAIR-V2X LP64-LS40 AP30/AP50 | 通信量(log2) |
|---------|------|---------------------------|--------------------------|----------------------------|------------|
| AttFuse | MPDA | 0.767/0.570 | 0.737/0.574 | 0.425/0.364 | 22.0 |
| AttFuse | BackAlign | 0.787/0.584 | 0.685/0.524 | 0.456/0.373 | 22.0 |
| AttFuse | CodeFilling | 0.722/0.536 | 0.666/0.510 | 0.385/0.319 | 15.0 |
| AttFuse | STAMP | 0.759/0.569 | 0.726/0.561 | 0.447/0.391 | 22.0 |
| AttFuse | **GenComm** | **0.804/0.633** | **0.753/0.601** | **0.459/0.379** | **16.0** |
| V2X-ViT | MPDA | 0.850/0.660 | 0.687/0.502 | 0.472/0.379 | 22.0 |
| V2X-ViT | BackAlign | 0.855/0.693 | 0.691/0.523 | 0.490/0.392 | 22.0 |
| V2X-ViT | CodeFilling | 0.860/0.689 | 0.560/0.416 | 0.445/0.356 | 15.0 |
| V2X-ViT | STAMP | 0.844/0.628 | 0.751/0.544 | 0.542/0.494 | 22.0 |
| V2X-ViT | **GenComm** | **0.867/0.699** | **0.763/0.576** | **0.565/0.467** | **16.0** |

### 消融实验 — 可扩展性成本

| 方法 | 新增智能体的训练参数量 | 新增智能体的计算量(FLOPs) |
|------|---------------------|------------------------|
| MPDA | 基准 | 基准 |
| BackAlign | 高 | 高 |
| STAMP | 高 | 高 |
| CodeFilling | 中等 | 中等 |
| **GenComm** | **降低81%** | **降低81%** |

### 关键发现

1. GenComm在仿真（OPV2V-H）和真实（DAIR-V2X、V2X-Real）数据集上均超越现有SOTA方法
2. 通信量从22.0降至16.0（log2），通信效率显著提升
3. 接纳新智能体的计算和参数开销比适配型方法降低81%，比重建型方法降低62%
4. 消融实验验证了三个组件（消息提取器、特征生成器、通道增强器）各自的贡献

## 亮点与洞察

- **范式创新**：首次提出基于生成（而非适配或重建）的异构协作通信机制，避免侵入式修改原始网络
- **轻量可扩展**：新智能体加入只需微调一个小提取器，极大降低了系统对新成员的准入成本
- **通信效率**：传输压缩后的空间消息而非完整中间特征，降低带宽需求
- **扩散模型的巧妙应用**：用条件扩散模型在ego端"想象"出协作者的特征，是扩散模型在协作感知中的新颖应用

## 局限与展望

- 条件扩散模型的推理速度可能成为实时系统的瓶颈，需要探索加速策略
- 当前仅在3D目标检测任务上验证，可扩展至语义分割等其他下游任务
- 两阶段训练策略仍需为每对新的异构组合做微调，探索zero-shot异构协作是值得的方向
- 空间消息的压缩率与信息保留的权衡可进一步研究

## 相关工作与启发

- 与HEAL/STAMP等适配型方法不同，GenComm不需要定义共享的协议语义空间
- 与DiffBEV/CoDiff类似地使用扩散模型生成BEV特征，但GenComm将其创新地应用于跨智能体的异构特征翻译
- 启发：生成式方法可能是解决多模态/多架构系统互操作性的通用范式

## 评分

- 新颖性：⭐⭐⭐⭐⭐（首创生成式协作通信范式）
- 技术深度：⭐⭐⭐⭐（扩散模型+可变形卷积设计合理）
- 实验充分度：⭐⭐⭐⭐⭐（三个数据集、多种设定、全面消融）
- 实用性：⭐⭐⭐⭐（可扩展性优异，但扩散推理速度待验证）

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](../../CVPR2025/image_generation/traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[ACL 2025\] Multimodal Pragmatic Jailbreak on Text-to-image Models](../../ACL2025/image_generation/multimodal_pragmatic_jailbreak_on_text-to-image_models.md)
- [\[NeurIPS 2025\] Janus-Pro-R1: Advancing Collaborative Visual Comprehension and Generation via Reinforcement Learning](janus-pro-r1_advancing_collaborative_visual_comprehension_and_generation_via_rei.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)

<!-- RELATED:END -->
