---
title: >-
  [论文解读] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay
description: >-
  [CVPR 2026][医学图像][continual learning] 提出 FORGE，首个专为跨站点 fMRI 脑疾病诊断设计的持续学习框架，通过结构感知 VAE 生成逼真的功能连接矩阵进行隐私保护式生成回放，结合双层知识蒸馏和层次化上下文赌博机采样策略，有效缓解灾难性遗忘。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "continual learning"
  - "fMRI"
  - "functional connectivity"
  - "generative replay"
  - "知识蒸馏"
---

# Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay

**会议**: CVPR 2026  
**arXiv**: [2604.14259](https://arxiv.org/abs/2604.14259)  
**代码**: [github.com/4me808/FORGE](https://github.com/4me808/FORGE)  
**领域**: 医学影像  
**关键词**: continual learning, fMRI, functional connectivity, generative replay, knowledge distillation

## 一句话总结

提出 FORGE，首个专为跨站点 fMRI 脑疾病诊断设计的持续学习框架，通过结构感知 VAE 生成逼真的功能连接矩阵进行隐私保护式生成回放，结合双层知识蒸馏和层次化上下文赌博机采样策略，有效缓解灾难性遗忘。

## 研究背景与动机

fMRI 功能连接 (FC) 矩阵是脑疾病诊断的强大表示，但临床数据通常从不同机构依次到达。现有诊断模型要么在单站点训练要么需要完整多站点数据访问，面临灾难性遗忘问题。传统持续学习方法主要针对图像数据设计，对图结构的医学数据（特别是 fMRI）研究不足。隐私法规进一步限制了跨机构的原始数据共享。

## 方法详解

### 整体框架

FORGE 将生成回放融入双层知识蒸馏方案。在每个新站点，学生分类器同时学习当前站点数据和回放的合成 FC 矩阵，通过 logit 级和图表示级蒸馏与前一站点教师对齐。FCM-VAE 在每个站点更新以生成站点特定的合成样本。

### 关键设计

1. **FCM-VAE 结构感知编码器**: 采用图 Transformer 架构，每个 ROI 节点特征由 FC 连接谱、谱嵌入和节点度拼接而成。注意力层融合局部邻接偏置（保持局部拓扑约束）和谱位置偏置（捕获全局几何信息），使编码器能够感知 FC 网络的内在拓扑和谱属性。

2. **低秩解码器与可达性门控**: 解码器利用 FC 矩阵的已知低秩结构，对 Fisher-z 变换和标准化后的连接强度建模为 Poisson 似然。低秩假设与大规模功能连接的已知特性一致，提升生成样本的保真度。

3. **层次化上下文赌博机自适应回放采样**: 不同于均匀采样回放缓冲区，使用层次化上下文赌博机策略自适应选择最有信息量的合成样本进行回放，提高回放效率。双层蒸馏包括分类 logit 对齐和图读出表示对齐。

### 损失函数 / 训练策略

统一目标包含：(1) 当前站点数据的分类损失；(2) 回放合成数据的分类损失；(3) logit 级蒸馏（L2 对齐学生与教师 logits）；(4) 图读出级蒸馏（L2 对齐图级表示）。教师输出在加入缓冲区时计算一次后固定不变。

## 实验关键数据

### 主实验

在 ABIDE（自闭症）、REST-meta-MDD（抑郁症）和 BSNIP（精神分裂症）大规模神经影像数据集上评估：

| 数据集 | 任务 | 方法 | 遗忘缓解 | 预测准确率 |
|--------|------|------|----------|-----------|
| ABIDE | ASD 诊断 | FORGE | **最优** | **最优** |
| REST-meta-MDD | MDD 诊断 | FORGE | **最优** | **最优** |
| BSNIP | SZ 诊断 | FORGE | **最优** | **最优** |

### 消融实验

- FCM-VAE 生成的 FC 矩阵质量显著优于现有图生成模型
- 图读出蒸馏相比仅 logit 蒸馏进一步提升性能
- 层次化采样策略优于均匀随机采样

### 关键发现

- 结构感知编码有效捕获了 FC 矩阵的拓扑特性
- 生成回放在隐私保护的前提下有效缓解灾难性遗忘
- 低秩解码器生成的 FC 矩阵具有更高的生物学保真度

## 亮点与洞察

- 首个将持续学习与 fMRI 功能连接分析结合的完整框架
- 隐私保护式生成回放解决了医疗数据共享的法规限制
- 结构感知设计充分利用了 FC 矩阵的领域特性

## 局限与展望

- 116 个 ROI 的固定脑图谱（AAL-116）限制了更精细的脑区分析
- 生成模型的保真度在极小样本站点上可能下降
- 未考虑站点间扫描仪差异的域适应

## 相关工作与启发

- 图级持续学习的设计可推广到其他图分类场景
- 谱位置编码在图 Transformer 中的应用值得关注
- 低秩解码器的思路适用于其他具有低秩结构的数据生成

## 评分

7/10 — 问题定义清晰、方法设计全面，是跨学科的有价值工作，但下游任务影响力受 fMRI 领域规模限制。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry](../../NeurIPS2025/medical_imaging/riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[ICML 2026\] Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis](../../ICML2026/medical_imaging/learning_multi-scale_hypergraph_for_high-order_brain_connectivity_analysis.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](../../NeurIPS2025/medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)

</div>

<!-- RELATED:END -->
