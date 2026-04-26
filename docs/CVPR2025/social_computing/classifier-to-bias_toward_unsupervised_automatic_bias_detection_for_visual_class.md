---
title: >-
  [论文解读] Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers
description: >-
  [CVPR 2025][公平性] 提出C2B，首个无需标注数据的视觉分类器偏差自动发现框架，仅依靠任务文本描述，利用LLM生成偏差候选并通过检索增强验证来识别预训练模型中的系统性偏差。
tags:
  - CVPR 2025
  - 公平性
  - 偏差检测
  - LLM
  - 无监督
  - 视觉分类器
---

# Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers

**会议**: CVPR 2025  
**arXiv**: [2504.20902](https://arxiv.org/abs/2504.20902)  
**代码**: https://github.com/mardgui/C2B  
**领域**: AI安全 / 公平性  
**关键词**: 偏差检测, 无监督, LLM, 检索增强, 模型公平性

## 一句话总结

提出 C2B（Classifier-to-Bias），首个仅依靠分类任务的文本描述（无需任何标注数据）即可自动发现预训练视觉分类器偏差的框架，通过 LLM 生成类特定偏差候选、生成检索标题收集图像数据集、最后计算偏差分数，在 CelebA 和 ImageNet-X 上超越需要监督的 SOTA 偏差检测方法。

## 研究背景与动机

**领域现状**：预训练模型在 HuggingFace 等平台广泛共享，用户下载使用时可能不了解模型的偏差和失败模式。现有偏差检测方法（如 B2T、UDIS）依赖标注数据来发现模型偏差。

**现有痛点**：所有现有方法都需要任务特定的标注数据（至少需要分类标签），大大限制了偏差检测的适用范围。普通用户可能没有资源收集标注数据。

**核心矛盾**：偏差检测本应是模型部署前的必要步骤，但标注数据依赖使其成本高昂，形成了"检测偏差需要数据，但获取数据本身就是瓶颈"的矛盾。

**本文目标**：在零标注数据条件下，仅用分类任务的文本描述自动识别模型偏差。

**切入角度**：LLM 具备对不同分类任务可能存在何种偏差的推理能力，而大规模图像数据库+文本检索可以提供无标注的测试数据。

**核心 idea**：用 LLM 提出偏差候选 → 生成检索标题 → 从大规模库中检索图像 → 测试模型在不同偏差类下的表现差异。

## 方法详解

### 整体框架

C2B 三步流程：(1) 用 LLM 为每个类别生成类特定的偏差属性和偏差类列表；(2) 用 LLM 生成描述「目标类+偏差类」组合的标题，通过 CLIP 检索或 Bing 搜索收集对应图像；(3) 在收集的伪标注数据集上测试目标模型，比较不同偏差类间的准确率差异来量化偏差。

### 关键设计

1. **类特定偏差候选生成**:

    - 功能：为每个目标类自动生成可能的偏差属性及其取值
    - 核心思路：将任务描述 $T$ 和类名 $y$ 拼接后输入 LLM（Llama 3.1 8B），让 LLM 输出该类的偏差属性列表 $\mathcal{B}^y$。例如对鸟类分类，可能输出"背景"（森林/天空/水面）、"姿态"（飞行/站立）等。不同类有不同偏差列表，因为不同物体的偏差因素不同。
    - 设计动机：偏差具有任务和类特定性，统一的偏差列表无法覆盖所有场景。LLM 的世界知识恰好适合推理"什么因素可能影响分类"。

2. **检索增强的数据收集**:

    - 功能：为每个目标类-偏差类组合收集测试图像
    - 核心思路：两步标题生成——先让 LLM 生成任务适配的标题模板，再为每个具体的目标类和偏差类细化标题。用生成的标题通过 CLIP 从 CC12M 大规模数据库检索 top-k 图像，或通过 Bing 搜索引擎获取。每个标题检索 20 张图像，标题本身提供了伪标注的目标标签和偏差标签。
    - 设计动机：避免收集和标注数据，利用已有的大规模无标注图像库。通过精心构造的标题确保检索图像同时匹配目标类和偏差类。

3. **偏差分数计算**:

    - 功能：量化模型在各偏差类上的表现差异
    - 核心思路：对每个目标类 $y$ 和偏差类 $b_{i,j}$，计算模型在该子集上的准确率 $A_y(f, b_{i,j})$，偏差分数为该准确率与其他偏差类平均准确率之差：$\phi_{y,i,j} = A_y(f, b_{i,j}) - \frac{1}{n_i-1}\sum_{k \neq j} A_y(f, b_{i,k})$。分数为正表示模型偏向该偏差类，为负表示偏见。
    - 设计动机：通过相对比较消除基线准确率的影响，直接度量分类器在不同偏差类间的表现差异。

### 损失函数 / 训练策略

C2B 完全无训练，是一个推理时框架。只需要一个 LLM 和一个检索引擎即可运行。

## 实验关键数据

### 主实验

| 方法 | CelebA Hit↑ | CelebA FH↓ | CelebA Miss↓ |
|------|-------------|------------|--------------|
| B2T (监督) | 较低 | - | 较高 |
| C2B-cc12m (无监督) | **较高** | 较低 | **较低** |
| C2B-Bing (无监督) | **最高** | 最低 | **最低** |

*C2B 在完全无监督条件下超越了需要标注数据的 B2T*

### 消融实验

| 配置 | Hit Rate | 说明 |
|------|----------|------|
| CC12M 检索 | 基础 | CLIP 嵌入检索 |
| Bing 搜索 | +提升 | 网络搜索覆盖更广 |
| 多标题变体 | +提升 | 增加标题多样性 |

### 关键发现
- C2B 能发现数据集标注之外的新偏差（如 CelebA 中"年龄"偏差未在原始 40 个属性中标注）
- Bing 搜索比 CC12M 检索效果更好，因为网络覆盖的领域更广
- 检索图像的准确性（是否匹配标题描述）是 C2B 性能的关键瓶颈
- 类特定偏差列表比全局统一列表更有效

## 亮点与洞察
- **零标注偏差检测的新范式**：从"先收集数据再检测偏差"转变为"先推理可能的偏差再按需收集数据"，大幅降低了偏差检测的门槛
- **LLM 作为偏差假设生成器**：利用 LLM 的世界知识替代领域专家，实现偏差假设的自动化
- **模块化设计**：LLM、检索引擎和目标模型完全解耦，可随技术进步无缝升级各组件

## 局限与展望
- 依赖检索图像质量，错误检索的图像会引入偏差分数的噪声
- LLM 可能遗漏某些偏差（如 LLM 未考虑到的文化特定因素）
- 目前仅在分类任务上验证，检测/分割等任务的偏差检测需要扩展
- Bing 搜索的商业 API 增加了使用成本

## 相关工作与启发
- **vs B2T**: B2T 需要标注数据来识别失败样本然后提取偏差关键词，C2B 完全无监督
- **vs OpenBias**: OpenBias 检测生成模型偏差，C2B 检测判别模型偏差，方法论互补
- 框架可迁移到 LLM 偏差检测——用 LLM 生成偏差测试用例

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次实现零标注偏差检测，问题定义有价值
- 实验充分度: ⭐⭐⭐⭐ CelebA 和 ImageNet-X 两个数据集，含多种评估协议
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐⭐ 极具实用价值，降低了公平性审计的门槛

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification](classifier-guided_clip_distillation_for_unsupervised_multi-label_classification.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](../../AAAI2026/social_computing/argumentative_debates_for_transparent_bias_detection_technic.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](../../ACL2025/social_computing/gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ACL 2025\] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades](../../ACL2025/social_computing/taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)

<!-- RELATED:END -->
