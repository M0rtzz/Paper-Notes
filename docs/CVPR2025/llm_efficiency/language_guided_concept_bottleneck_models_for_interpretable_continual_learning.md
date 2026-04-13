---
title: >-
  [论文解读] Language Guided Concept Bottleneck Models for Interpretable Continual Learning
description: >-
  [CVPR 2025][LLM效率][concept bottleneck model] 将语言引导的概念瓶颈模型 (CBM) 整合到持续学习中，通过 ChatGPT 生成类别概念、CLIP 编码的概念对齐模块和语义引导的原型增强策略，在 ImageNet-subset 上实现最终准确率 +3.06% 的提升，同时提供透明可解释的决策过程。
tags:
  - CVPR 2025
  - LLM效率
  - concept bottleneck model
  - continual learning
  - interpretability
  - CLIP
  - prototype augmentation
---

# Language Guided Concept Bottleneck Models for Interpretable Continual Learning

**会议**: CVPR 2025  
**arXiv**: [2503.23283](https://arxiv.org/abs/2503.23283)  
**代码**: https://github.com/FisherCats/CLG-CBM  
**领域**: LLM效率  
**关键词**: concept bottleneck model, continual learning, interpretability, CLIP, prototype augmentation

## 一句话总结
将语言引导的概念瓶颈模型 (CBM) 整合到持续学习中，通过 ChatGPT 生成类别概念、CLIP 编码的概念对齐模块和语义引导的原型增强策略，在 ImageNet-subset 上实现最终准确率 +3.06% 的提升，同时提供透明可解释的决策过程。

## 研究背景与动机

1. **领域现状**：持续学习领域近年涌现了大量基于预训练 ViT 的方法（EASE、CPrompt 等），但这些方法的决策过程仍然是黑盒的——用户无法理解模型为什么将某张图片分类为特定类别。

2. **现有痛点**：(1) 现有持续学习方法缺乏可解释性，在医疗、自动驾驶等高风险场景中无法满足需求。(2) 概念瓶颈模型 (CBM) 能提供可解释性，但直接用于持续学习会面临概念空间增长和旧类别概念遗忘的问题。(3) 简单拼接 CBM 和持续学习会导致概念冲突和负迁移。

3. **核心矛盾**：可解释性能力（通过人类可理解的概念做决策）和持续学习能力（在增量任务中保持性能）之间需要同时优化，但两者有固有冲突——新概念的引入可能干扰旧概念。

4. **本文要解决什么？** 如何设计一个既能持续学习新任务、又能通过人类可理解概念提供透明决策的模型？

5. **切入角度**：利用 ChatGPT 生成类别相关概念、CLIP 编码概念并对齐视觉特征、语义引导的原型增强防止旧类别遗忘。

6. **核心idea一句话**：用 ChatGPT + CLIP 构建自动化的概念瓶颈层，结合概念对齐和语义原型增强，让持续学习模型的每一个决策都可以用人类语言解释。

## 方法详解

### 整体框架
每个新任务到来时：(1) ChatGPT 生成该任务类别的概念描述 → (2) CLIP 编码概念并选择最有信息量的 100 个 → (3) 概念对齐模块将视觉特征与概念分数对齐 → (4) 语义引导的原型增强为旧类别生成伪特征 → (5) 通过概念瓶颈层做可解释的分类。

### 关键设计

1. **Concept Bottleneck Layer (CBL)**
   - 做什么：建立人类可理解概念到分类决策的桥梁
   - 核心思路：ChatGPT 为每个类生成概念描述（外观、颜色、大小、形状、环境），T5 提取概念，CLIP text encoder 编码。Learning-to-search 算法选择每个任务最有信息量的 100 个概念
   - 设计动机：概念空间随任务累积增长，选择机制确保概念池保持精简且相关

2. **Concept Alignment Module**
   - 做什么：对齐学习到的概念分数与 CLIP 概念激活分数
   - 核心思路：$\mathcal{L}_{sim} = -\frac{1}{|C_{1:t}|} \sum \cos(\hat{E}^i, \hat{E}_{clip}^i)$，其中 $\hat{E} = E^3$（立方化增强锐度）
   - 弹性网络稀疏惩罚：$\mathcal{L}_{sparse} = \varphi||W_l||_1 + (1-\varphi)||W_l||_F$，$\varphi=0.99$
   - 设计动机：无概念对齐时模型可能学到负面概念（如"不是猫"），对齐确保模型聚焦正面相关概念

3. **Semantic-Guided Prototype Augmentation**
   - 做什么：为旧类别生成伪特征防止遗忘
   - 核心思路：找到语义最相似的当前任务类别 $h = \arg\max_i \cos(f_T(y^j), p^i)$，生成伪特征 $\tilde{V}^j = p^j + (V^h - p^h)$
   - 设计动机：假设语义相似的类别有相似的特征漂移模式，用当前任务的漂移信息来补偿旧类别

### 损失函数 / 训练策略
- $\mathcal{L} = \mathcal{L}_{ce} + \lambda\mathcal{L}_{sim} + \sigma\mathcal{L}_{sparse}$，$\lambda=1$，$\sigma=0.001$

## 实验关键数据

### 主实验

| 数据集 | Split | 平均准确率 $\bar{A}$ | 最终准确率 $A_{last}$ |
|--------|-------|---------------------|---------------------|
| CIFAR-100 | B10-Inc10 | 84.49±0.26 | 76.82±0.50 |
| ImageNet-subset | B10-Inc10 | 86.83±1.32 | 78.97±0.39 |
| CUB-200 | B10-Inc10 | 85.40±0.61 | 82.20±0.65 |
| Stanford-cars | B14-Inc14 | 88.60±0.55 | 85.07±0.82 |

vs EASE: ImageNet-subset +1.79% $\bar{A}$, +3.06% $A_{last}$; CUB-200 +4.43%; Stanford-cars +4.69%

### 消融实验

| 配置 | 结果 |
|------|------|
| Base only | 11.02% |
| Base + Prototype | 69.15% |
| Base + Prototype Augmentation | **75.91%** |
| w/o Concept Alignment | 负面概念多，可解释性差 |
| w/ Concept Alignment | 正面概念主导，可解释 |

### 关键发现
- 语义引导原型增强比简单原型回放提升 6.76%（69.15%→75.91%）
- 概念对齐模块是可解释性的关键：无对齐时模型学到"这个不是X"类型的负面概念
- 细粒度数据集提升更大（CUB-200 +4.43%，Stanford-cars +4.69%），概念描述在细粒度区分中更有价值
- 概念选择策略（learning-to-search 选 100 个）平衡了表达力和效率

## 亮点与洞察
- **可解释性 + 持续学习的首次有效结合**：不是简单拼接，而是让两者互相增强
- **ChatGPT 生成概念**：利用 LLM 的知识自动构建概念空间，消除了人工标注的需求
- **语义引导的伪特征生成**：假设"语义近似的类别有相似漂移"非常巧妙
- 概念对齐的立方化 $E^3$ 简单有效地增强了概念分数的锐度

## 局限性 / 可改进方向
- ChatGPT 生成的概念质量不一致，可能引入噪声
- 每个任务 100 个概念的上限可能不适用于超大规模类别数
- 概念空间的语义重叠（不同类别的相似概念）未充分处理
- 推理时的可解释性展示方式（如何向用户呈现概念）未深入讨论

## 相关工作与启发
- **vs EASE**: 性能更优且提供可解释性，但计算开销更大
- **vs CPrompt**: CPrompt 纯 prompt-based，无可解释性
- **vs 标准 CBM**: 标准 CBM 不考虑增量学习，概念空间固定

## 评分
- 新颖性: ⭐⭐⭐⭐ 可解释持续学习的新范式
- 实验充分度: ⭐⭐⭐⭐ 六个数据集，消融完整
- 写作质量: ⭐⭐⭐⭐ 概念清晰
- 价值: ⭐⭐⭐⭐⭐ 可解释性在持续学习中的实际需求巨大
