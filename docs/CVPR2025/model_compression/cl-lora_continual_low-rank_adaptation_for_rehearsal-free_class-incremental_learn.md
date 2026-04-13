---
title: >-
  [论文解读] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning
description: >-
  [CVPR 2025][模型压缩][Class-Incremental Learning] 提出 CL-LoRA，设计双适配器架构（任务共享 + 任务特定 LoRA），结合知识蒸馏与梯度重分配以及可学习块级权重，在仅 0.3% 可训练参数下实现 SOTA 持续学习性能。
tags:
  - CVPR 2025
  - 模型压缩
  - Class-Incremental Learning
  - LoRA
  - PEFT
  - 知识蒸馏
  - catastrophic forgetting
---

# CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning

**会议**: CVPR 2025  
**arXiv**: [2505.24816](https://arxiv.org/abs/2505.24816)  
**代码**: [JiangpengHe/CL-LoRA](https://github.com/JiangpengHe/CL-LoRA)  
**机构**: MIT / Purdue University
**领域**: 持续学习 / 模型压缩  
**关键词**: Class-Incremental Learning, LoRA, PEFT, knowledge distillation, catastrophic forgetting

## 一句话总结
提出 CL-LoRA，设计双适配器架构（任务共享 + 任务特定 LoRA），结合知识蒸馏与梯度重分配以及可学习块级权重，在仅 0.3% 可训练参数下实现 SOTA 持续学习性能。

## 研究背景与动机

1. **领域现状**：类增量学习（CIL）需要模型按顺序学习新类别同时保留旧知识。近年来，预训练模型（PTM）结合参数高效微调（PEFT）在无回放 CIL 中展现出良好效果，无需存储旧任务样本。
2. **现有痛点**：
   - **Prompt-based 方法**（L2P、DualPrompt、CODA-Prompt）：需要大量 prompt 参数，且任务选择机制复杂
   - **Adapter-based 方法**（EASE、O-LoRA、InfLoRA）：每个任务创建新适配器，导致参数冗余
   - 现有方法未能有效利用跨任务的共享知识，每个任务独立学习导致知识碎片化
3. **核心矛盾**：如何在保持参数效率的同时，既学习跨任务共享知识又捕获任务特定特征，实现稳定性（不遗忘）与可塑性（学新知识）的平衡。
4. **切入角度**：设计双适配器架构，前半 Transformer 层用共享 LoRA 学跨任务知识，后半层用任务特定 LoRA 学特有特征。
5. **核心 idea 一句话**：共享 LoRA（带知识蒸馏+梯度重分配） + 特定 LoRA（带块级权重+正交约束） = 高效持续学习。

## 方法详解

### 整体架构
基于 ViT-B/16 预训练模型，将 12 层 Transformer 分为两半：前 6 层插入任务共享 LoRA，后 6 层插入任务特定 LoRA。推理时使用原型分类器（prototype classifier）。

### 关键设计

1. **任务共享适配器（Task-Shared Adapter）**
   - 做什么：在前 $l=6$ 个 Transformer 块中插入共享 LoRA，学习跨任务通用知识
   - 核心思路：使用随机正交矩阵初始化 $\mathbf{A}_s$，仅更新 $\mathbf{B}_s$
   - 知识蒸馏：在学习新任务时，用上一任务的共享适配器作为教师，对当前适配器进行知识蒸馏
   - 早退出策略：仅在共享适配器的最后一层计算蒸馏损失，减少计算开销

2. **梯度重分配（Gradient Reassignment）**
   - 做什么：识别并保护共享适配器中对旧任务重要的参数
   - 核心思路：计算教师模型和学生模型梯度的差异，对重要参数降低学习率
   - 实现：$\nabla \mathcal{L}'_{kd} = \nabla \mathcal{L}_{kd} \odot |\nabla_{\mathbf{B}_s^{t-1}} \mathcal{L}_{kd} - \nabla_{\mathbf{B}_s^{t}} \mathcal{L}_{kd}|$
   - 效果：更精准地保留重要共享知识

3. **任务特定适配器与块级权重（Block-wise Weights）**
   - 做什么：在后 $N-l$ 个 Transformer 块中，每个任务有独立的 LoRA
   - 块级权重：为每个特定适配器学习可训练的逐块缩放因子 $w_i^j$
   - 正交约束：$\mathcal{L}_{orth} = \sum_{j=l+1}^{N} \sum_{i \neq k} \| \mathbf{B}_i^j {}^\top \mathbf{B}_k^j \|_F^2$
   - 设计动机：不同任务可能在不同 Transformer 层需要不同程度的适配

4. **原型分类器推理**
   - 对每个已学任务，使用对应的特定适配器计算特征
   - 计算特征与各任务原型的余弦相似度
   - 预测相似度最高的类别

### 训练目标
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_1 \mathcal{L}_{kd} + \lambda_2 \mathcal{L}_{orth}$$

超参数：$\lambda_1 = 5$，$\lambda_2 = 0.0001$，LoRA rank $r = 10$。

## 实验关键数据

### 主实验：4 个基准的平均准确率 $\bar{A}$（%）
| 方法 | 参数量(%) | CIFAR-100 T=10 | ImageNet-R T=20 | ImageNet-A T=20 | VTAB T=10 |
|------|----------|---------------|----------------|----------------|----------|
| L2P | 0.2 | 79.51 | 65.82 | 39.81 | 78.96 |
| DualPrompt | 0.5 | 80.44 | 67.41 | 56.43 | 82.51 |
| EASE | 1.4 | 85.71 | 78.04 | 68.92 | 93.01 |
| RanPAC | 3.1 | **87.62** | **78.53** | 66.14 | 89.61 |
| InfLoRA | 0.3 | 80.97 | 73.22 | 56.91 | 88.83 |
| O-LoRA | 0.4 | 81.26 | 72.52 | 55.02 | 87.22 |
| **CL-LoRA** | **0.3** | 85.32 | **81.58** | **70.15** | **94.57** |

### ImageNet-R 长序列（T=40）平均准确率
| 方法 | $\bar{A}$ |
|------|----------|
| InfLoRA | 47.04 |
| O-LoRA | 47.53 |
| **CL-LoRA** | **60.54** |

CL-LoRA 在长序列上优势更大，超过第二名 13 个百分点。

### 消融实验（CIFAR-100 T=10 / ImageNet-R T=20）
| KD | GR | BW | CIFAR-100 | ImageNet-R |
|----|----|----|-----------|------------|
| ✗ | ✗ | ✗ | 88.20 | 82.24 |
| ✓ | ✗ | ✗ | 90.83 | 83.42 |
| ✓ | ✓ | ✗ | 91.69 | 84.08 |
| ✗ | ✗ | ✓ | 89.01 | 82.93 |
| ✓ | ✓ | ✓ | **91.85** | **84.77** |

三个组件各有贡献，KD 提升最大。

## 亮点与洞察
- **双适配器架构**设计优雅：前层共享知识、后层特化适配
- 仅 **0.3% 可训练参数**即达 SOTA，比 RanPAC（3.1%）少 10 倍参数仍有竞争力
- 梯度重分配机制比简单知识蒸馏更精准地保护重要参数
- 在有分布偏移的困难基准（ImageNet-R/A）上优势尤为明显
