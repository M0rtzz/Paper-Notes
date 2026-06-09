---
title: >-
  [论文解读] Robustness of Vision Foundation Models to Common Perturbations
description: >-
  [CVPR 2026][自监督学习][foundation model] 首次系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）的鲁棒性，提出三种鲁棒性度量并形式化五个数学性质，发现基础模型普遍不鲁棒，并提出微调方法改善鲁棒性而不牺牲效用。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "foundation model"
  - "robustness"
  - "common perturbation"
  - "embedding"
  - "CLIP"
  - "DINOv2"
---

# Robustness of Vision Foundation Models to Common Perturbations

**会议**: CVPR 2026  
**arXiv**: [2604.14973](https://arxiv.org/abs/2604.14973)  
**代码**: 无  
**领域**: AI 安全/鲁棒性  
**关键词**: foundation model, robustness, common perturbation, embedding, CLIP, DINOv2

## 一句话总结

首次系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）的鲁棒性，提出三种鲁棒性度量并形式化五个数学性质，发现基础模型普遍不鲁棒，并提出微调方法改善鲁棒性而不牺牲效用。

## 研究背景与动机

视觉基础模型输出图像的嵌入向量用于下游任务，但常见编辑操作（JPEG 压缩、亮度/对比度调整等）会改变嵌入向量。与对抗扰动不同，常见扰动在非对抗的真实场景中频繁发生。三个核心问题：(1) 基础模型本身有多鲁棒？(2) 下游应用有多鲁棒？(3) 如何提升鲁棒性？设计合适的度量来量化鲁棒性是关键挑战。

## 方法详解

### 整体框架

(1) 提出三种鲁棒性度量并分析其数学性质；(2) 系统评估六个工业级基础模型在九类常见扰动下的鲁棒性；(3) 提出微调方法平衡鲁棒性和效用。

### 关键设计

1. **DivergenceRadius 度量**: 使用嵌入空间中最小包含球的半径作为鲁棒性度量，满足全部五个期望数学性质（有界域、单调性、最优鲁棒性、最差鲁棒性、旋转不变性），优于余弦相似度和欧氏距离度量（不满足最差鲁棒性性质）。证明余弦相似度度量和欧氏距离度量等价（$\mathcal{R}_{ed} = \sqrt{\mathcal{R}_{cs}}$）。

2. **鲁棒性-性能线性关系**: 发现下游分类准确率和深度估计 MSE 与图像的鲁棒性值之间近似线性关系，可通过简单线性回归模型准确预测扰动图像的下游性能。

3. **鲁棒性感知微调**: 优化目标为鲁棒性损失和效用损失的加权和。鲁棒性目标最小化扰动图像嵌入间的变化，效用目标保持原有下游任务性能。实验证实方法成功提升鲁棒性而不损害效用。

### 损失函数 / 训练策略

微调损失 = 效用损失（保持原有表示质量）+ $\alpha$ × 鲁棒性损失（最小化扰动嵌入变化），$\alpha$ 控制平衡。

## 实验关键数据

### 主实验

评估 CLIP（OpenAI，3 种架构）和 DINOv2（Meta，3 种架构）在 9 类扰动下的表现：

| 发现 | 详情 |
|------|------|
| 普遍不鲁棒 | 所有基础模型对常见扰动产生显著嵌入变化 |
| 架构影响 | ViT 架构比 ResNet 架构更鲁棒 |
| 下游影响 | 玻璃模糊使零样本 ImageNet 分类准确率下降 9.4% |
| 可预测性 | 鲁棒性值可准确预测下游性能（线性关系 R² 高） |

### 消融实验

- 扰动参数域扩大时鲁棒性单调下降（验证单调性性质）
- 不同扰动类型对嵌入的影响程度差异显著
- 微调后的模型在多数扰动类型上鲁棒性提升

### 关键发现

- 基础模型的鲁棒性问题被严重忽视——简单 JPEG 压缩即可显著改变嵌入
- ViT 比 ResNet 更鲁棒可能源于 Transformer 的全局注意力机制
- 鲁棒性度量可作为下游性能预测的代理指标

## 亮点与洞察

- 形式化五个数学性质并证明哪些度量满足/违反，理论分析严谨
- DivergenceRadius 的最小包含球定义直观且数学完备
- 鲁棒性-性能线性关系具有实际应用价值

## 局限与展望

- 仅考虑了九种常见扰动，组合扰动的效果未分析
- 微调方法需要为每种扰动类型分别训练
- 多模态基础模型（如 CLIP 的文本编码器）的鲁棒性未涉及

## 相关工作与启发

- 为基础模型的部署提供了重要的鲁棒性参考基线
- DivergenceRadius 度量可推广到其他需要量化表示稳定性的场景
- 鲁棒性预测性能的线性关系简化了实际部署中的质量评估

## 评分

7/10 — 系统性强、理论分析严谨、实际价值明确，是基础模型鲁棒性研究的重要基线工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](../../CVPR2025/self_supervised/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)

</div>

<!-- RELATED:END -->
