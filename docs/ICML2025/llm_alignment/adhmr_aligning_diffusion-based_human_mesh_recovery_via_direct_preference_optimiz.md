---
title: >-
  [论文解读] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization
description: >-
  [ICML2025][LLM对齐][人体网格恢复] 将DPO思想引入扩散式人体网格恢复(HMR)：训练HMR-Scorer评估预测质量，构建偏好数据集(winner/loser对)，用DPO微调基座扩散模型，无需3D标注即可提升in-the-wild图像上的HMR性能。
tags:
  - ICML2025
  - LLM对齐
  - 人体网格恢复
  - 扩散模型
  - DPO
  - 偏好优化
  - HMR-Scorer
---

# ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization

**会议**: ICML2025  
**arXiv**: [2505.10250](https://arxiv.org/abs/2505.10250)  
**代码**: [GitHub - ADHMR](https://github.com/shenwenhao01/ADHMR)  
**领域**: llm_alignment  
**关键词**: 人体网格恢复, 扩散模型, DPO, 偏好优化, HMR-Scorer

## 一句话总结
将DPO思想引入扩散式人体网格恢复(HMR)：训练HMR-Scorer评估预测质量，构建偏好数据集(winner/loser对)，用DPO微调基座扩散模型，无需3D标注即可提升in-the-wild图像上的HMR性能。

## 研究背景与动机

### 概率式HMR的缺陷
确定性HMR产生单一预测，概率式方法(扩散模型)生成多个候选但缺乏对齐：
1. 3D mesh预测与2D图像线索不一致
2. in-the-wild图像上性能差

### 根本原因
扩散训练损失关注分布匹配而非精确对齐。端到端扩散模型在早期去噪步骤中避免使用重投影损失。伪3D标注本身含噪。

### DPO的适用性
传统逐关节/逐像素损失容易过拟合噪声标签。DPO关注相对质量而非绝对质量，更鲁棒。

## 方法详解

### 整体框架（三步）
1. **训练HMR-Scorer**：奖励模型，评估mesh预测与输入图像的对齐质量
2. **构建偏好数据集**：用HMR-Scorer对基座模型生成的候选排序
3. **DPO微调**：用偏好数据集微调扩散式HMR基座模型

### HMR-Scorer设计
- 多尺度图像特征提取（全局+局部）
- 通过重投影人体关键点采样像素对齐特征
- 训练目标：预测重建质量分数（PVE/MPJPE/PA-MPJPE综合）

### DPO for Diffusion HMR
- 借鉴Wallace et al. (2024)的扩散DPO
- 偏好对：高分预测(winner) vs 低分预测(loser)
- 相比逐点损失，DPO对噪声标签更鲁棒

### 副产品：数据清洗
HMR-Scorer还可用于筛除伪标注低质量的训练样本，仅保留高分样本。实验证明即使数据量减少，性能反而提升。

## 实验关键数据

### 与SOTA概率式HMR对比

| 方法 | MPJPE | PA-MPJPE | PVE |
|------|-------|---------|-----|
| ProHMR | 59.8 | 41.2 | 72.4 |
| ScoreHypo | 52.3 | 36.1 | 63.8 |
| **ADHMR** | **48.7** | **33.5** | **59.2** |

### HMR-Scorer数据清洗效果

| 训练数据 | 原始性能 | 清洗后性能 | 数据量变化 |
|---------|---------|----------|-----------|
| 100%原始数据 | 基线 | - | - |
| HMR-Scorer Top-80% | **优于基线** | +2.3 MPJPE | -20% |
| HMR-Scorer Top-60% | **优于基线** | +1.8 MPJPE | -40% |

### 关键发现
1. DPO比传统3D损失更有效地提升in-the-wild性能
2. HMR-Scorer的多尺度特征对细微不对齐高度敏感
3. 数据清洗让模型用更少数据达到更好性能
4. 不需要3D标注即可在in-the-wild图像上微调

## 亮点与洞察

1. 将LLM对齐中的DPO思想成功迁移到CV的HMR任务。
2. HMR-Scorer作为副产品可替代人工数据审核。
3. 对伪标注噪声的鲁棒性是DPO在CV中的重要优势。
4. 无需3D标注即可in-the-wild微调，实用性极强。

## 局限与展望

1. HMR-Scorer本身的质量上限受训练数据分布影响。
2. DPO需要生成多个候选来构建偏好对，增加计算。
3. 仅验证了单人场景，多人HMR待扩展。
4. 对极端遮挡或罕见姿态的处理能力未专门评估。

## 相关工作与启发

- 与ScoreHypo区别：ScoreHypo用额外网络在测试时选择，ADHMR直接改善生成质量。
- 与Diffusion-DPO的关系：借鉴其公式但适配到结构化输出（mesh而非图像）。
- 启发：DPO范式可推广到其他结构化预测任务（深度估计、姿态估计等）。

## 评分
- 新颖性: 4.5/5 — DPO迁移到HMR+HMR-Scorer副产品
- 实验充分度: 4.5/5 — 全面对比+数据清洗实验
- 写作质量: 4.0/5
- 价值: 4.5/5 — 对概率式3D视觉有重要推动

## 补充技术分析

### HMR-Scorer多尺度特征设计
全局特征捕捉整体姿态合理性，局部特征通过重投影关键点采样捕捉细粒度对齐。两者拼接后输入评分MLP。

### DPO相比传统损失的优势
传统逐关节损失容易过拟合噪声标签，DPO关注相对质量（winner比loser好即可），对不完美伪标注更鲁棒。

### 数据清洗的意外发现
HMR-Scorer过滤掉底部20%低质量伪标注数据后，模型用80%数据反而性能提升+2.3 MPJPE。这说明数据质量比数量更重要。

<!-- RELATED:START -->

## 相关论文

- [Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)
- [D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)
- [Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [DiffPO: Diffusion Alignment with Direct Preference Optimization](../../ACL2025/llm_alignment/diffpo_diffusion_alignment.md)

<!-- RELATED:END -->
