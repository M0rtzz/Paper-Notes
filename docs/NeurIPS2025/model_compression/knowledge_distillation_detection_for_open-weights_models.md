---
title: >-
  [论文解读] Knowledge Distillation Detection for Open-weights Models
description: >-
  [NeurIPS 2025][模型压缩][知识蒸馏检测] 提出知识蒸馏检测任务，通过无数据输入合成和统计评分框架，判断一个开放权重的学生模型是否由特定教师模型蒸馏而来。 领域现状：知识蒸馏技术广泛用于模型压缩，从大型教师模型向小型学生模型传递知识，在图像分类、LLM、文生图等领域取得成功。 现有痛点：蒸馏技术可能被滥用于未经…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "知识蒸馏检测"
  - "模型溯源"
  - "数据无关合成"
  - "统计检测"
  - "文本到图像生成"
---

# Knowledge Distillation Detection for Open-weights Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.02302](https://arxiv.org/abs/2510.02302)  
**代码**: [GitHub](https://github.com/shqii1j/distillation_detection)  
**领域**: 图像生成  
**关键词**: 知识蒸馏检测, 模型溯源, 数据无关合成, 统计检测, 文本到图像生成

## 一句话总结

提出知识蒸馏检测任务，通过无数据输入合成和统计评分框架，判断一个开放权重的学生模型是否由特定教师模型蒸馏而来。

## 研究背景与动机

**领域现状**：知识蒸馏技术广泛用于模型压缩，从大型教师模型向小型学生模型传递知识，在图像分类、LLM、文生图等领域取得成功。

**现有痛点**：蒸馏技术可能被滥用于未经授权地克隆专有模型，侵犯知识产权；但目前缺乏有效手段检测模型是否被蒸馏。

**核心矛盾**：现有方法（如成员推理攻击、OOD检测）主要关注训练数据检测，无法直接判断模型间的蒸馏关系。

**本文目标**：在仅有学生模型权重和教师模型API的实际场景下，检测学生模型是否由特定教师蒸馏。

**切入角度**：将问题建模为多选题形式，从候选教师集合中选择最可能的蒸馏来源。

**核心 idea**：通过数据无关的输入合成+统计评分的通用框架，比较学生与候选教师的输出对齐程度来检测蒸馏。

## 方法详解

### 整体框架

三阶段检测流水线：**输入构造** → **评分计算** → **决策预测**。给定开放权重学生模型和K个候选教师模型的API，生成合成输入来探测模型行为，计算学生与各教师的对齐分数，选择分数最高的教师作为蒸馏来源。

### 关键设计

1. **预测决策（Score Maximization）**：

    - 功能：从K个候选教师中选出蒸馏来源
    - 为什么：多选题形式避免了阈值校准问题
    - 怎么做：$k^* = \arg\max_{k \in \{1,...,K\}} S(g_\theta, f^{(k)}, \mathcal{P})$
    - 区别：可通过阈值自然扩展到二分类检测

2. **点级评分（Point-wise Score）**：

    - 功能：逐样本计算学生与教师输出的差异
    - 为什么：简单有效，即使单个输入也能工作
    - 怎么做：对每个输入 $x_n$ 计算 $s_n^{(k)} = \frac{1}{\delta(g_\theta(x_n), f^{(k)}(x_n)) + \epsilon}$，取平均
    - 分类任务使用KL散度，文生图任务使用LPIPS

3. **集合级评分（Set-level Score）**：

    - 功能：度量整体分布对齐
    - 为什么：捕获全局分布级别的对齐模式
    - 怎么做：分类任务使用Aligned Cosine Similarity (ACS)，文生图任务使用CKA with RBF Kernels
    - 区别：需要多个输入样本才能计算

4. **输入构造（Data-free Input Synthesis）**：

    - 功能：在无训练数据条件下生成合成查询
    - 为什么：实际场景中无法获取训练数据
    - 怎么做（分类）：训练生成器 $G_\phi$，使用mixup策略 + BNS损失对齐批归一化统计量
    - 怎么做（文生图）：直接使用空字符串作为prompt（利用CFG训练中的无条件生成）
    - 关键公式：$\min_\phi \sum_{i=1}^C w_i \cdot \mathcal{L}_{hard}(g_\theta(\hat{x}(\phi)), y_i) + \mathcal{L}_{BNS}(g_\theta, \hat{x}(\phi))$

### 损失函数 / 训练策略

- 生成器训练：交叉熵损失 $\mathcal{L}_{hard}$ + BNS对齐损失 $\mathcal{L}_{BNS}$
- BNS损失：$\mathcal{L}_{BNS} = \sum_{l=1}^{L} \|\mu_l^r - \mu_l\|_2^2 + \|\sigma_l^r - \sigma_l\|_2^2$
- 文生图场景用空字符串作为输入，利用CFG训练的无条件生成特性

## 实验关键数据

### 主实验

**CIFAR-10 蒸馏检测** (N=100合成输入, Acc./AUC)：

| 方法 | N=1 | N=50 | N=100 | 平均 |
|------|-----|------|-------|------|
| MIA Filter + KL | 0.43/0.66 | 0.54/0.79 | 0.55/0.80 | 0.51/0.75 |
| OOD Filter + KL | 0.42/0.68 | 0.55/0.79 | 0.54/0.80 | 0.50/0.75 |
| **Ours (KL)** | **0.62/0.75** | **0.87/0.94** | **0.87/0.94** | **0.81/0.89** |
| Oracle | 0.45/0.64 | 0.87/0.96 | 0.95/0.99 | 0.70/0.84 |

**文生图模型蒸馏检测** (Acc./AUC)：

| 方法 | N=1 | N=10 | N=100 | 平均 |
|------|-----|------|-------|------|
| GPT-2 + DINO | 0.81/0.87 | 0.80/0.94 | 0.80/0.95 | 0.80/0.92 |
| Blip-Base + CLIP | 0.71/0.78 | 0.81/0.93 | 0.83/0.96 | 0.79/0.91 |
| **Ours (LPIPS)** | **0.89/1.00** | **0.97/1.00** | **1.00/0.99** | **0.96/1.00** |

### 消融实验

| 设置 | CIFAR-10 | ImageNet | 平均 |
|------|----------|----------|------|
| OOD filter + ACS | 0.56/0.77 | 0.37/0.52 | 0.47/0.65 |
| Synthetic Data + CKA | 0.82/0.77 | — | — |
| Ours (完整) | **0.87/0.94** | **0.75/0.92** | — |

### 关键发现

- 即使只用1个合成输入，方法就能取得远超基线的效果
- 在CIFAR-10上，方法甚至超越使用真实数据的Oracle
- 文生图检测N=10时就达到0.97准确率，利用CFG的空字符串prompt效果极佳
- vanilla KD比RKD和OFAKD更容易被检测到

## 亮点与洞察

- **问题新颖**：首次系统定义了知识蒸馏检测任务，具有重要的知识产权保护意义
- **方法通用**：同一框架适用于分类和生成模型，模型无关
- **数据无关**：完全不需要训练数据，仅依赖模型权重和API
- **空字符串prompt的巧妙**：利用CFG训练中的无条件建模特性，使得空字符串成为天然的分布内探测输入

## 局限与展望

- 当前仅考虑多选题设定，二分类检测需要阈值校准
- 分类任务中依赖BatchNorm统计量的生成器训练，对无BN架构（如ViT）可能受限
- 未考虑对抗性场景：蒸馏者可能故意混淆模型行为以规避检测
- 可扩展到LLM蒸馏检测场景

## 相关工作与启发

- 与模型水印互补：水印是主动防御，本文是被动检测
- 数据无关量化/蒸馏中的合成数据技术被创造性地用于检测目的
- AIforGood视角：开源模型的安全性和可追溯性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创蒸馏检测任务，问题定义独到
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类和生成两大类任务，多种蒸馏方法
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导规范
- 价值: ⭐⭐⭐⭐ 对AI安全和知识产权保护有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[NeurIPS 2025\] Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation](fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)

</div>

<!-- RELATED:END -->
