---
title: >-
  [论文解读] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration
description: >-
  [CVPR 2026][图像恢复][图像修复] 提出IQPIR框架，引入预训练NR-IQA模型的图像质量先验(IQP)作为条件信号，通过质量条件化Transformer、双Codebook结构和离散表示空间质量优化三个机制，引导图像修复过程趋向最高感知质量，在盲人脸修复等任务上全面超越SOTA。
tags:
  - CVPR 2026
  - 图像恢复
  - 图像修复
  - 图像质量先验
  - 双Codebook
  - NR-IQA
  - 质量条件化
---

# Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration

**会议**: CVPR 2026  
**arXiv**: [2603.29773](https://arxiv.org/abs/2603.29773)  
**代码**: https://github.com/fengyang1399-pixel/IQPIR (有)  
**领域**: 图像修复  
**关键词**: 图像修复, 图像质量先验, 双Codebook, NR-IQA, 质量条件化

## 一句话总结
提出IQPIR框架，引入预训练NR-IQA模型的图像质量先验(IQP)作为条件信号，通过质量条件化Transformer、双Codebook结构和离散表示空间质量优化三个机制，引导图像修复过程趋向最高感知质量，在盲人脸修复等任务上全面超越SOTA。

## 研究背景与动机

**领域现状**：真实世界图像修复旨在从复杂退化的低质输入恢复高质图像。基于Codebook的方法将修复转化为离散表示空间的编码预测问题，有效降低重建歧义。

**现有痛点**：所有方法都隐式假设GT是完美的唯一监督源。但如图1所示，GT数据集（如FFHQ）的感知质量不一致——多数GT质量分在5-8之间，极少达到9。模型会收敛到GT的**平均质量水平**而非最高可达质量。

**核心矛盾**：(1) 仅用最高质量GT训练→数据多样性不足→伪影和退化特征；(2) 用全部GT训练→被平均质量拉低。

**切入角度**：不同质量等级的GT提供不同功能——HQ+ GT擅长精细结构控制，平均GT更适合大面积模糊恢复。

**核心idea**：将NR-IQA分数作为条件信号注入修复模型，推理时设为最大值→引导网络产出最高质量；双Codebook分别学习通用结构和HQ+细节。

## 方法详解

### 整体框架
两阶段：(1) Codebook学习阶段——双Codebook架构分别学习通用和HQ+特征；(2) 编码预测阶段——质量条件化Transformer预测双编码序列+质量优化损失。

### 关键设计

1. **双Codebook架构**:

    - 功能：分离通用结构特征和高质量特有细节
    - 核心思路：Common Codebook对所有GT训练；HQ+ Codebook仅当GT质量分 $S > S_{thr}$ 时参与量化。融合特征 $Z_q = Z_q^1 + \alpha Z_q^2$（或仅 $Z_q^1$ 若 $S \leq S_{thr}$）。解码器从融合表示重建图像
    - 设计动机：HQ GT的精细视觉细节（如发丝端部）需要专门的Codebook来编码，而通用Codebook保证广泛的退化恢复能力

2. **质量条件化Transformer**:

    - 功能：以质量分数为条件预测编码序列
    - 核心思路：NR-IQA模型评估GT质量分 $S$，嵌入为向量 $\mathbf{s} \in \mathbb{R}^{h \times w \times c}$，直接加到LQ特征 $\hat{Z}_l = Z_l + \mathbf{s}$。Transformer读入 $\hat{Z}_l$ 预测两个编码序列 $\mathbf{c}_1, \mathbf{c}_2$，分别查询两个Codebook
    - **推理时**：设 $S$ 为最大值→引导网络产出最高感知质量的修复结果
    - 设计动机：类似类条件生成，让模型学会质量-图像对应关系，实现**可控质量修复**

3. **离散表示空间的质量优化**:

    - 功能：在离散表示空间做质量优化避免连续空间的过度优化
    - 核心思路：用NR-IQA模型计算修复结果的质量损失 $\mathcal{L}_{quality}$
    - 设计动机：连续空间中直接优化IQA分数容易过度优化产生伪影，离散Codebook限制了输出空间，天然避免此问题

4. **质量先验集成**:

    - 融合多个NR-IQA模型的评分取平均 $S = \frac{1}{n}\sum s_i$，缓解单一模型偏差

### 损失函数 / 训练策略
Codebook阶段：重建损失+量化承诺损失+感知损失。预测阶段：编码预测交叉熵+质量优化损失。

## 实验关键数据

### 主实验（盲人脸修复，LFW-Test）

| 方法 | TOPIQ-G↑ | Musiq-G↑ | Q-Align↑ | CLIP-IQA↑ |
|------|----------|----------|----------|-----------|
| CodeFormer | 0.809 | 0.832 | 4.31 | 0.697 |
| DAEFR | 0.814 | 0.827 | 4.33 | 0.696 |
| WaveFace | 0.786 | 0.799 | 4.43 | 0.788 |
| Interlcm | 0.831 | 0.834 | 4.55 | 0.721 |
| **IQPIR (Ours)** | **0.861** | **0.878** | **4.67** | **0.790** |

WebPhoto-Test和WIDER-Test上同样全面领先。

### 消融实验

| 配置 | 主要指标 | 说明 |
|------|---------|------|
| 无质量条件 | 下降 | 证明IQP条件化的必要性 |
| 单Codebook | 下降 | HQ+ Codebook对精细细节很重要 |
| 连续空间质量优化 | 过度优化 | 离散空间的优势 |
| 单一IQA模型 | 略下降 | 多模型集成更鲁棒 |

### 关键发现
- **IQP是通用的质量提升策略**：将DifFace加上本文的质量条件方法(DifFace+)也能显著提升修复质量，证明了plug-and-play特性
- 双Codebook中HQ+ Codebook主要改善毛发端部、皮肤纹理等精细细节
- 推理时设质量分为最大值→感知质量显著超过GT平均水平

## 亮点与洞察
- **挑战GT完美假设**：首次系统地揭示了GT数据质量不一致对修复模型的影响，提出了"超越GT"的修复范式
- **质量条件化的plug-and-play特性**：IQP可以作为独立模块插入任何修复架构，无需结构修改
- **离散空间质量优化**：巧妙利用了VQ-VAE的离散性来避免连续空间中IQA奖励过度优化的问题

## 局限与展望
- NR-IQA模型自身也有偏差（某些模型可能偏好特定风格），虽然用集成缓解但不能完全消除
- $S_{thr}$ 阈值和 $\alpha$ 权重需要手动调整
- 当GT质量极低时，HQ+ Codebook能学到的内容有限
- 可探索将质量先验扩展到视频修复和3D修复

## 相关工作与启发
- **vs CodeFormer/DAEFR**: 它们假设GT完美，直接监督。本文引入质量维度突破这一限制
- **vs 基于GAN/扩散的修复**: 生成先验强大但可能产生幻觉。Codebook先验+质量先验更可控
- **vs NR-IQA研究**: 将IQA从评估工具转变为训练信号，拓展了IQA的应用边界

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 质量先验用于修复的思路新颖，双Codebook+条件化+离散优化的系统设计完整
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多指标、消融充分，plug-and-play验证有说服力
- 写作质量: ⭐⭐⭐⭐ 动机图(GT质量分布)直观有力
- 价值: ⭐⭐⭐⭐⭐ 通用质量引导策略，对修复领域有广泛影响

<!-- RELATED:START -->

## 相关论文

- [Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)

<!-- RELATED:END -->
