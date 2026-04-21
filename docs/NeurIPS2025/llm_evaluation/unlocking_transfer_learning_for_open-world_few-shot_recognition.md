---
title: >-
  [论文解读] Unlocking Transfer Learning for Open-World Few-Shot Recognition
description: >-
  [NeurIPS 2025 (Workshop)][少样本开集识别] 提出两阶段方法，通过开集感知元学习 + 开集无关迁移学习，首次将迁移学习范式成功应用于少样本开集识别 (FSOSR)，在 miniImageNet 和 tieredImageNet 上达到 SOTA。
tags:
  - NeurIPS 2025 (Workshop)
  - 少样本开集识别
  - 迁移学习
  - 元学习
  - 开放世界
  - 伪开集样本
---

# Unlocking Transfer Learning for Open-World Few-Shot Recognition

**会议**: NeurIPS 2025 (Workshop)  
**arXiv**: [2411.09986](https://arxiv.org/abs/2411.09986)  
**代码**: 无  
**领域**: 少样本学习 / 开放世界识别  
**关键词**: 少样本开集识别, 迁移学习, 元学习, 开放世界, 伪开集样本

## 一句话总结

提出两阶段方法，通过开集感知元学习 + 开集无关迁移学习，首次将迁移学习范式成功应用于少样本开集识别 (FSOSR)，在 miniImageNet 和 tieredImageNet 上达到 SOTA。

## 研究背景与动机

少样本开集识别 (Few-Shot Open-Set Recognition, FSOSR) 面临双重挑战：

**闭集分类**: 使用极少样本将输入分类到已知类别

**开集检测**: 同时识别不属于已知类别的开集输入

尽管迁移学习在闭集少样本学习中已成为主流范式，但用于开放世界场景时存在根本问题：**迁移学习微调模型时，仅使用闭集样本，缺乏对开集边界的建模**。本文旨在"解锁"迁移学习在 FSOSR 中的潜力。

## 方法详解

### 整体框架

两阶段方法：
1. **阶段一 — 开集感知元学习**: 训练模型建立有利于开集检测的度量空间
2. **阶段二 — 开集无关迁移学习**: 对特定任务进行迁移学习微调

### 关键设计

1. **开集感知元学习 (Open-set Aware Meta-learning)**:

    - 在元学习阶段引入模拟的开集样本
    - 训练模型不仅区分已知类别，还能识别"不认识"的输入
    - 建立的度量空间为后续迁移学习提供良好的初始化

2. **伪开集样本生成策略**:

    - **数据集修改法**: 从训练数据中划出部分类别模拟开集
    - **生成法**: 在特征空间中合成离群样本
    - 两种策略可互补使用

3. **开集无关迁移学习 (Open-set Free Transfer Learning)**:

    - 在测试阶段，仅用闭集支持集进行迁移学习
    - 不需要额外的开集校准样本
    - 依赖阶段一建立的度量空间来维持开集检测能力

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{open}}$$

- $\mathcal{L}_{\text{CE}}$: 闭集分类的交叉熵损失
- $\mathcal{L}_{\text{open}}$: 开集检测损失（基于边界的对比损失）

## 实验关键数据

### 主实验

| 方法 | miniImageNet AUROC ↑ | miniImageNet Acc ↑ | tieredImageNet AUROC ↑ | tieredImageNet Acc ↑ |
|------|--------------------|--------------------|----------------------|--------------------|
| ProtoNet + threshold | 68.5 | 65.2 | 70.2 | 67.8 |
| PEELER | 72.3 | 68.5 | 73.8 | 70.5 |
| SnaTCHer | 74.1 | 70.2 | 75.5 | 72.3 |
| OpenFSL | 75.8 | 71.5 | 77.2 | 73.8 |
| TANE | 76.5 | 72.8 | 78.1 | 74.5 |
| **Ours** | **79.2** | **75.5** | **80.8** | **77.2** |

### 训练效率对比

| 方法 | 训练 epochs | 额外训练开销 | 5-shot AUROC | 1-shot AUROC |
|------|-----------|-----------|------------|------------|
| PEELER | 200 | +50% | 72.3 | 65.8 |
| SnaTCHer | 200 | +30% | 74.1 | 67.2 |
| OpenFSL | 200 | +40% | 75.8 | 68.5 |
| **Ours** | **200** | **+1.5%** | **79.2** | **72.5** |

### 消融实验

| 设置 | AUROC ↑ | Closed-set Acc ↑ |
|------|---------|-----------------|
| 完整方法 | 79.2 | 75.5 |
| 仅元学习（无迁移学习） | 75.5 | 70.2 |
| 仅迁移学习（无元学习） | 71.8 | 74.8 |
| 去掉伪开集样本 | 74.5 | 73.5 |
| 数据修改法伪开集 | 77.8 | 74.8 |
| 生成法伪开集 | 78.5 | 75.2 |
| 双伪开集策略 | 79.2 | 75.5 |

### 关键发现

1. 迁移学习在 FSOSR 中确实可行，关键是需要开集感知的预训练
2. 训练额外开销仅 1.5%，远低于其他方法，效率极高
3. 两种伪开集样本生成策略互补，联合使用效果最佳
4. 阶段一建立的度量空间是迁移学习成功的基础

## 亮点与洞察

- **范式突破**: 首次证明迁移学习在 FSOSR 中可行
- **极低额外开销**: 仅 1.5% 训练增量就实现 SOTA
- **实用策略**: 伪开集样本的生成不需要额外数据

## 局限与展望

1. 目前在 workshop paper 层面，实验规模有限
2. 在更大规模数据集（如 ImageNet 完整版）上的验证不足
3. 伪开集样本的质量仍可改进
4. 与 CLIP 等预训练视觉-语言模型的结合值得探索

## 相关工作与启发

- **ProtoNet**: 经典度量学习方法
- **PEELER**: 开集少样本学习的先驱工作
- **SnaTCHer**: 通过"抓取"原型进行开集检测
- **CLIP-based FSOSR**: 利用预训练 VLM 的开放词汇能力

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 3 |
| 实验充分性 | 4 |
| 写作质量 | 3 |
| 实用价值 | 4 |
| 总体推荐 | 3.5 |

<!-- RELATED:START -->

## 相关论文

- [Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring](open-insect_benchmarking_open-set_recognition_of_novel_species_in_biodiversity_m.md)
- [Random Registers for Cross-Domain Few-Shot Learning](../../ICML2025/llm_evaluation/random_registers_for_cross-domain_few-shot_learning.md)
- [Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [Feedforward Few-shot Species Range Estimation](../../ICML2025/llm_evaluation/feedforward_few-shot_species_range_estimation.md)

<!-- RELATED:END -->
