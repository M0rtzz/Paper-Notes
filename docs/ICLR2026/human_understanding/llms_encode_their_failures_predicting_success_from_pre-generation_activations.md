---
title: >-
  [论文解读] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
description: >-
  [ICLR 2026][人体理解][难度预测] 本文证明 LLM 在生成前的内部激活中编码了模型特有的成功概率信息，训练线性探针可以提取该信号用于高效的模型路由，在 MATH 等基准上实现匹配最强模型精度的同时降低 70% 推理成本。
tags:
  - ICLR 2026
  - 人体理解
  - 难度预测
  - 线性探针
  - 模型路由
  - 推理时计算
  - 成功预测
---

# LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations

**会议**: ICLR 2026  
**arXiv**: [2602.09924](https://arxiv.org/abs/2602.09924)  
**代码**: https://github.com/KabakaWilliam/llms_know_difficulty  
**领域**: LLM 效率 / 模型路由  
**关键词**: 难度预测, 线性探针, 模型路由, 推理时计算, 成功预测

## 一句话总结
本文证明 LLM 在生成前的内部激活中编码了模型特有的成功概率信息，训练线性探针可以提取该信号用于高效的模型路由，在 MATH 等基准上实现匹配最强模型精度的同时降低 70% 推理成本。

## 研究背景与动机

**领域现状**：LLM 在数学和编程任务上取得了显著成果，但运行扩展推理（如 CoT）对每个问题都很昂贵。模型路由系统需要准确估计模型在给定输入上的成功概率，但低方差估计需要多次昂贵的采样。

**现有痛点**：先前工作已展示模型包含正确性相关信号，但不清楚这些信号代表的是人类难度还是模型特有难度，也不清楚它们是否可靠到足以支持实际决策。现有路由方法依赖间接代理如输入长度、困惑度或启发式置信度。

**核心矛盾**：人类对难度的判断与模型对难度的"感知"是两回事——随着扩展推理能力增强，模型越来越能解决人类觉得难的问题，导致两种难度概念的分离加剧。

**本文目标** 1) 区分 LLM 内部激活中编码的人类难度和模型难度信号；2) 评估这些信号在不同推理策略下的可靠性；3) 将探针用于实际的模型路由以降低推理成本。

**切入角度**：使用 E2H-AMC 数据集（同时有人类 IRT 难度标签和模型表现），直接比较从相同预生成激活中提取的人类难度和模型难度信号。

**核心 idea**：LLM 在生成答案之前就在激活中编码了自身成功概率的信息，通过线性探针提取该信号可实现高效的成本-准确率权衡路由。

## 方法详解

### 整体框架
对给定 LLM，在指令 token 后、生成第一个 token 前，提取最后一层的激活向量。在此激活上训练线性探针来预测模型在特定解码策略下的成功/失败。然后利用探针的预测概率进行模型路由决策。

### 关键设计

1. **双目标线性探针体系**:

    - 做什么：分别预测人类 IRT 难度和模型成功概率
    - 核心思路：对相同的预生成激活 $\mathbf{h} \in \mathbb{R}^d$，训练两种探针：(a) 回归探针预测期望成功率 $\hat{s}_{MC}(\pi, q) = \mathbf{w}^\top \mathbf{h} + b$（MSE 损失）；(b) 二分类探针预测特定解码策略下的成功/失败（BCE 损失），目标可以是 Greedy 或 Maj@K 成功
    - 设计动机：区分人类难度和模型难度是两种不同的信号，后者对路由更有价值。使用监督线性探针而非无监督方向提取，在推理任务上获得更强的判别力

2. **级联路由 (Cascade Routing)**:

    - 做什么：在基础模型和强模型间进行成本感知的查询分配
    - 核心思路：给定基础模型 $M_s$ 和强模型 $M_l$，基于阈值规则路由：$M(x) = M_l$ if $\hat{p}_s(x) < \tau$，否则 $M(x) = M_s$。阈值 $\tau$ 控制性能-成本权衡
    - 设计动机：简单的阈值策略即可有效利用探针信号，无需复杂的路由学习

3. **效用最大化路由 (Utility-Based Routing)**:

    - 做什么：在异构模型池中进行最优选择
    - 核心思路：给定模型池 $\{M_1, \ldots, M_K\}$ 和归一化成本 $\{\hat{c}_1, \ldots, \hat{c}_K\}$，选择 $\hat{M}(x) = \arg\max_i (\hat{p}_i(x) - \lambda \hat{c}_i)$，其中 $\lambda$ 控制成功概率与成本之间的权衡
    - 设计动机：当模型池异构时（不同大小、不同推理预算），需要同时考虑每个模型的成功概率和成本

### 损失函数 / 训练策略
线性探针训练使用 80/20 训练-验证分割，层和位置选择基于验证集。回归探针用 MSE，分类探针用 BCE。探针极其轻量——仅需一个线性层，训练成本可忽略。

## 实验关键数据

### 主实验

| 模型 | 推理方式 | 任务准确率 | 线性探针 AUROC | TF-IDF AUROC | 长度 AUROC |
|------|---------|-----------|---------------|-------------|-----------|
| Qwen2.5-Math-1.5B | Greedy | 0.724 | 0.84 | 0.64 | 0.61 |
| Qwen2.5-Math-1.5B | Maj@5 | 0.763 | 0.76 | 0.63 | 0.66 |
| Qwen2.5-Math-7B | Greedy | 0.809 | 0.79 | 0.68 | 0.67 |
| Qwen2.5-Math-7B | Maj@5 | 0.827 | 0.80 | 0.72 | - |
| GPT-OSS-20B (低推理) | Maj@5 | 0.866 | 0.78 | - | - |
| GPT-OSS-20B (高推理) | Maj@5 | 0.920 | 0.64 | - | - |

### 消融实验

| 信号类型 | Spearman ρ 范围 | 说明 |
|---------|----------------|------|
| 人类 IRT 难度 | 0.83-0.87 | 高度线性可提取 |
| 模型成功率（低推理）| 0.58 | 中等可提取 |
| 模型成功率（高推理）| 0.40 | 推理增强后显著退化 |

| 路由策略 | 准确率 | 成本节省 | 基准 |
|---------|--------|---------|------|
| 级联 (τ=0.6) | 91.2% (匹配) | 17% | MATH |
| 效用路由 (5模型) | 92% (匹配) | 70% | MATH |
| 效用路由 | 93.3% (匹配) | 37% | AIME 2025 |

### 关键发现
- 线性探针大幅超越表面特征（TF-IDF、问题长度），AUROC 通常高 10-20 个点
- 扩展推理提升了任务准确率但降低了探针质量（AUROC 从 0.78 降至 0.64），表明难度信息在推理链中变得不那么线性可分
- 人类难度和模型难度是不同的信号，模型推理能力越强，两者越分离
- 推理链长度与人类难度正相关但与模型成功负相关——模型在"人类觉得难"的问题上花更多 token，即使自己能轻松解决

## 亮点与洞察
- **预生成激活包含丰富的决策信号**：模型在开始生成前就"知道"自己能否解对，这个发现对自适应推理系统有深远影响
- **人类难度≠模型难度的实证证明**：随着推理能力增强，两者分离加剧，这意味着用人类难度标签来评估模型可能越来越不可靠

## 局限与展望
- 仅使用线性探针在单一位置探测，可能遗漏非线性编码的难度信息
- 未探索跨域/跨数据集的探针迁移能力
- 路由策略较简单（固定阈值），自适应路由策略可能进一步缩小与 oracle 差距
- 探针性能对 token 位置敏感，限制了实用性

## 相关工作与启发
- **vs Kadavath et al. (P(True))**: 他们通过显式提示获取模型自评，需要额外生成开销；本文从预生成激活零成本提取信号
- **vs Cencerrado et al. (正确性方向)**: 他们用无监督均值差方法在推理任务上 AUROC 仅 0.6-0.7，本文训练监督探针达到 >0.7

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统区分人类/模型难度信号并展示推理与探针质量的反比关系
- 实验充分度: ⭐⭐⭐⭐ 跨模型、跨数据集、跨推理策略的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑推进清晰，发现层层递进
- 价值: ⭐⭐⭐⭐ 对模型路由和自适应推理有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/human_understanding/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [Failures to Surface Harmful Contents in Video Large Language Models](../../AAAI2026/human_understanding/failures_to_surface_harmful_contents_in_video_large_language_models.md)
- [Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)
- [MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation](mollangbench_a_comprehensive_benchmark_for_language-prompted_molecular_structure.md)

<!-- RELATED:END -->
