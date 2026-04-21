---
title: >-
  [论文解读] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer
description: >-
  [CVPR 2026][3D视觉][4D重建] 提出 MoRe，一种前馈式运动感知 4D 重建 Transformer，通过注意力强制策略在训练时解耦动态运动与静态结构，结合分组因果注意力实现高效流式推理，在动态场景的相机位姿估计和深度预测上达到 SOTA。
tags:
  - CVPR 2026
  - 3D视觉
  - 4D重建
  - 动态场景
  - 注意力强制
  - 流式推理
  - 运动解耦
---

# MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer

**会议**: CVPR 2026  
**arXiv**: [2603.05078](https://arxiv.org/abs/2603.05078)  
**代码**: [项目页面](https://hellexf.github.io/MoRe/)  
**领域**: 3D视觉  
**关键词**: 4D重建, 动态场景, 注意力强制, 流式推理, 运动解耦

## 一句话总结

提出 MoRe，一种前馈式运动感知 4D 重建 Transformer，通过注意力强制策略在训练时解耦动态运动与静态结构，结合分组因果注意力实现高效流式推理，在动态场景的相机位姿估计和深度预测上达到 SOTA。

## 研究背景与动机

从单目视频重建时序演化的 3D 结构（4D 重建）是 AR/机器人/数字孪生等应用的核心需求。现有方法面临的困境：

- **静态假设的前馈模型**（DUSt3R、VGGT、Fast3R）：直接回归点图和位姿，速度快但在动态物体存在时，用于相机估计的特征被严重污染（注意力分散到运动物体上），导致位姿精度显著退化
- **优化式管线**（MonST3R、CasualSAM）：集成光流/分割/深度估计等模块，处理动态场景更鲁棒，但多阶段结构计算开销大，不适合实时或长序列处理
- **流式重建方法**（CUT3R、StreamVGGT）：采用 LLM 风格的因果注意力实现在线处理，但标准因果注意力破坏帧内 token 之间的空间一致性，且误差沿时间累积

核心挑战：如何设计一个快速、可泛化的框架，在动态场景和流式输入下同时保证位姿和深度的精度？

## 方法详解

### 整体框架

MoRe 基于强静态重建骨干（VGGT 架构），从单目视频帧序列 $\{I_t\}_{t=1}^T$ 联合估计深度 $\{D_t\}$、相机参数 $\{g_t\}$、点图 $\{P_t\}$ 和运动掩码 $\{M_t\}$。训练阶段通过注意力强制策略注入运动感知能力，推理阶段完全不依赖额外运动先验或分割输入。支持全注意力模式（最优质量）和流式因果注意力模式（在线处理）。

### 关键设计

1. **运动对齐注意力（Motion-aligned Attention）**：核心创新，在训练时利用 GT 运动掩码显式引导 camera token 的注意力分布。将运动掩码按 patch 大小 $s \times s$ 划分，计算每个 image token 的静态得分：
   $$a_i = 1 - \frac{1}{s^2} \sum_{(u,v) \in m_i} m_i(u,v)$$
   其中 $a_i \in [0,1]$，越高表示越静态。通过监督 camera token 的注意力权重 $\alpha_i$ 对齐 $a_i$，使模型学会聚焦静态区域、忽略运动物体。关键优势：**完全无测试时开销**——GT 掩码仅用于训练，推理时模型已内化了运动解耦能力。设计动机来自对 VGGT 注意力图的直接观察：动态场景中 camera token 注意力均匀分散到运动和静态区域，导致位姿估计混乱。

2. **分组因果注意力（Grouped Causal Attention）**：将标准因果注意力改造为帧感知因果注意力，允许同帧内的 image token 双向注意（保持空间一致性），同时帧间仅允许前向注意（保持时序因果性）。流式推理时，首帧对初始化 KV 缓存，后续帧增量处理：
   $$F_t = \text{Attn}(\mathbf{Q}_t, [\mathbf{K}_{1:t-1}, \mathbf{K}_t], [\mathbf{V}_{1:t-1}, \mathbf{V}_t])$$
   设计动机：标准 LLM 因果注意力将 image token 视为扁平序列，破坏了帧内空间对应关系。

3. **BA-like Token 聚合优化**：流式因果推理后的轻量级全局优化步骤。缓存所有帧的 camera query $\mathbf{Q}_t^{\text{cam}}$ 和 KV 特征，序列处理完成后让每个 camera token 重新注意所有帧的特征：
   $$\mathbf{C}_t^{\text{opt}} = \text{Attn}(\mathbf{Q}_t^{\text{cam}}, [\mathbf{K}_{1:T}], [\mathbf{V}_{1:T}])$$
   类比 Bundle Adjustment 的全局一致性优化，但仅需一次额外的注意力计算，开销极小。训练时通过复制 camera token 在序列末尾并行监督两条路径（流式 + 全局），确保两者一致。

### 损失函数 / 训练策略

- **深度/点图**：置信度加权回归损失 $\mathcal{L}_{\text{conf}} = \sum_i (\hat{c}_i \|\hat{y}_i - y_i\|_2^2 - \lambda \log(\hat{c}_i))$
- **运动掩码**：标准 BCE 损失 $\mathcal{L}_{\text{motion}}$
- **注意力对齐**：$\mathcal{L}_{\text{attn}} = \frac{1}{M} \sum_i \max(0, a_i - C) \cdot \alpha_i$，仅惩罚动态区域的注意力权重
- **相机位姿**：相对变换监督 $\mathcal{L}_{\text{cam}}$，对流式 token 截断早期梯度，对复制的全局 token 保留完整梯度
- 训练数据：12个数据集混合（Dynamic Replica, PointOdyssey, Spring, KITTI, ScanNet, Co3Dv2 等），覆盖室内外、动态静态

## 实验关键数据

### 主实验

**相机位姿估计（动态场景）**：

| 方法 | 类型 | Sintel ATE↓ | Bonn ATE↓ | TUM-dyn ATE↓ | ScanNet ATE↓ |
|------|------|-------------|-----------|--------------|--------------|
| VGGT | 全注意力 | 0.1715 | 0.0141 | 0.0109 | 0.0347 |
| MoRe (FA) | 全注意力 | **0.0877** | **0.0138** | 0.0115 | 0.0375 |
| CUT3R | 流式 | 0.2163 | 0.0420 | 0.0438 | 0.0929 |
| Stream3R | 流式 | 0.2144 | 0.0235 | 0.0240 | 0.0521 |
| MoRe | 流式 | **0.1474** | **0.0211** | **0.0260** | **0.0605** |

**视频深度估计**：

| 方法 | 类型 | Sintel AbsRel↓ | Bonn AbsRel↓ | KITTI AbsRel↓ |
|------|------|----------------|--------------|---------------|
| VGGT | 全注意力 | 0.387 | 0.055 | 0.073 |
| MoRe (FA) | 全注意力 | **0.335** | **0.055** | **0.066** |
| Stream3R | 流式 | 0.397 | 0.070 | 0.079 |
| MoRe | 流式 | **0.254** | **0.068** | **0.072** |

### 消融实验

| 配置 | Sintel ATE↓ | Sintel RPE_trans↓ | TUM ATE↓ | 说明 |
|------|-------------|-------------------|----------|------|
| w/o 注意力强制 | 0.163 | 0.092 | 0.028 | 去掉运动对齐注意力 |
| w/o BA-like 优化 | 0.155 | 0.085 | 0.027 | 去掉全局 token 聚合 |
| Full MoRe | **0.147** | **0.082** | **0.026** | 完整方法 |

| 配置 | Sintel AbsRel↓ | Bonn AbsRel↓ | KITTI AbsRel↓ | 说明 |
|------|----------------|--------------|---------------|------|
| w/o GCA | 0.277 | 0.070 | 0.079 | 标准因果注意力 |
| w/ GCA | **0.254** | **0.068** | **0.072** | 分组因果注意力 |

### 关键发现

- 注意力强制策略在 Sintel（大量动态物体）上效果最显著：ATE 从 0.163→0.147，验证了运动解耦的有效性
- 分组因果注意力在所有深度估计基准上一致提升，证明帧内空间一致性对几何推理至关重要
- 全注意力模式下，MoRe 在 Sintel ATE 上将 VGGT 从 0.1715 大幅降低到 0.0877（-49%），突破性提升
- 流式模式下全面超越同类方法（CUT3R、StreamVGGT、Wint3R、Stream3R），且支持增量处理
- 零样本泛化：所有动态评测数据集均未在训练中出现

## 亮点与洞察

1. **注意力强制**思路极其优雅：不修改推理架构，仅在训练时通过注意力监督教会模型"看哪里"，实现免费的运动解耦能力
2. 从 VGGT 注意力图的直接观察出发（图3），动机可视化清晰有力，问题定义精准
3. 分组因果注意力设计简洁有效，保留因果性同时恢复帧内空间一致——针对图像 token 对 LLM 因果注意力的最小改造
4. BA-like token 聚合仅需一次额外注意力计算就获得全局一致性，比传统 BA 高效得多

## 局限与展望

1. 训练依赖 GT 运动掩码，限制了可用训练数据的规模和多样性（需要带分割标注的动态数据集）
2. 流式模式下在 ScanNet（静态场景）上 ATE 0.0605 高于全注意力的 0.0375，说明因果限制在长序列静态场景中仍有损失
3. BA-like 优化需等待全序列处理完毕才能执行，不算严格的实时流式处理
4. 未报告运动掩码预测本身的精度，也未做下游任务（如动态物体分割/移除）的评测
5. 可探索自监督的运动掩码生成以摆脱对 GT 标注的依赖

## 相关工作与启发

- **VGGT**：MoRe 的直接改进基线，通过注意力强制修复了其在动态场景中的注意力混淆问题
- **CUT3R**：引入 Transformer 持久隐状态做在线重建，但注意力设计不区分帧内外
- **MonST3R/CasualSAM**：优化式管线处理动态场景更鲁棒，但速度慢，MoRe 通过前馈方式兼具速度和鲁棒性
- 启发：注意力强制作为一种通用的"在训练时教模型关注什么"的策略，可迁移到其他需要运动/静态解耦的视觉任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 注意力强制策略新颖且效果显著，分组因果注意力设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 4个基准、10+对比方法、全注意力+流式两种模式、完整消融
- 写作质量: ⭐⭐⭐⭐ 注意力图可视化动机清晰，公式推导严谨，实验组织有条理
- 价值: ⭐⭐⭐⭐ 为4D重建提供了实用的前馈解决方案，注意力强制策略具有广泛迁移潜力

<!-- RELATED:START -->

## 相关论文

- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4.md)
- [MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_4d_synthesis.md)

<!-- RELATED:END -->
