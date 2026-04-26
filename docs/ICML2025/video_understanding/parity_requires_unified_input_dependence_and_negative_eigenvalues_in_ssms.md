---
title: >-
  [论文解读] Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs
description: >-
  [ICML2025][视频理解][状态空间模型] 从理论上证明了线性SSM（如S4/Mamba）无法计算奇偶校验(parity)函数——即使允许输入依赖参数化——除非状态转移矩阵包含负特征值，为SSM的表达力瓶颈提供了精确的数学刻画。
tags:
  - ICML2025
  - 视频理解
  - 状态空间模型
  - 奇偶校验
  - 表达力
  - 负特征值
  - Mamba
---

# Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs

**会议**: ICML2025  
**arXiv**: [2508.07395](https://arxiv.org/abs/2508.07395)  
**代码**: 无  
**领域**: video_understanding  
**关键词**: 状态空间模型, 奇偶校验, 表达力, 负特征值, Mamba

## 一句话总结
从理论上证明了线性SSM（如S4/Mamba）无法计算奇偶校验(parity)函数——即使允许输入依赖参数化——除非状态转移矩阵包含负特征值，为SSM的表达力瓶颈提供了精确的数学刻画。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：S4、Mamba等线性状态空间模型在序列建模上取得了与Transformer竞争的性能，且推理更高效。但其表达力边界不如Transformer研究充分。

### 奇偶校验(Parity)作为基本测试

Parity是最简单的非平凡序列计算之一：给定0/1序列，输出1的个数是奇数还是偶数。它是序列模型表达力的试金石。

### 现有痛点

**现有痛点**：经验上观察到SSM在parity任务上表现差，但缺乏理论解释。

## 方法详解

### 主要定理
**定理1**：任何线性SSM（包括输入依赖的变体如Mamba），如果状态转移矩阵的特征值全为非负，则无法计算parity函数。

**定理2**：引入负特征值后，SSM可以计算parity——但需要O(log n)维的状态空间。

### 统一输入依赖(Unified Input Dependence)
分析涵盖了所有主流SSM变体：
- S4（固定参数）
- S5（对角化）
- Mamba（输入依赖的B/C矩阵）
- 完全输入依赖的A矩阵
证明了parity的不可能性独立于参数化方式。

### 负特征值的必要性
直观理解：parity需要"反转"操作（奇→偶或偶→奇），而正特征值只能做单调变换。负特征值提供了必要的符号翻转能力。

### 与RNN的对比
标准RNN（带非线性激活）可以轻松计算parity，因为非线性提供了等价于负特征值的能力。

## 实验关键数据

### Parity任务验证


### 主实验

| 模型 | 正特征值 | 负特征值 | Parity准确率 |
|------|---------|---------|-------------|
| S4 (标准) | 全正 | 无 | ~50% (random) |
| Mamba | 可正 | 无 | ~50% |
| S4 + 负特征值 | 混合 | 有 | **100%** |
| RNN | N/A | N/A | 100% |

### 序列长度vs表达力


### 消融实验

| 序列长度 | 需要的状态维度(负特征值) |
|---------|---------------------|
| 16 | 4 |
| 64 | 6 |
| 256 | 8 |
| 1024 | 10 |

遵循O(log n)的理论预测。

### 关键发现
1. 所有主流SSM在正特征值下都无法计算parity
2. 负特征值是充分必要条件
3. 所需状态维度随序列长度对数增长
4. 输入依赖参数化不能弥补符号限制
5. 这解释了SSM在需要"计数"的任务上系统性弱于Transformer

## 亮点与洞察

1. 极其干净的理论贡献——parity的不可能性结果与可能性结果。
2. 涵盖了所有主流SSM变体的统一分析。
3. "负特征值"的发现有直接的工程指导意义（修改初始化即可）。
4. 理论预测(O(log n)状态维度)被实验精确验证。
5. 为SSM vs Transformer的表达力辩论提供了关键理论基准。

## 局限与展望

1. Parity是极端简化的任务，与实际NLP/CV任务的关联待建立。
2. 理论仅覆盖线性SSM，对非线性变体(如RWKV的某些模式)不适用。
3. 负特征值的引入对训练稳定性的影响未讨论。
4. 实际大规模SSM中特征值的分布特征未分析。
5. 与其他表达力度量(如TC/WL测试)的关系未建立。

## 相关工作与启发

- 与Transformer表达力研究(Hahn 2020等)互补。
- 与SSM初始化研究(HiPPO等)的直接联系。
- 启发：SSM的初始化应包含负特征值以确保基本表达力。

## 评分
- 新颖性: 5.0/5 — 干净的不可能性定理
- 实验充分度: 4.0/5 — 理论为主但验证清晰
- 写作质量: 5.0/5 — 定理和证明清晰
- 价值: 5.0/5 — 对SSM设计有根本性指导

## 补充分析

### 负特征值的物理含义
正特征值意味着状态单调衰减/增长，负特征值引入振荡（符号翻转）。Parity本质上是符号翻转操作，因此绝对需要负特征值。

### 对Mamba初始化的建议
Mamba的HiPPO初始化通常产生负特征值，但某些简化版本可能将其移除。本文的结果警示：切不可移除负特征值。

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](../../ICLR2026/video_understanding/the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](../../CVPR2025/video_understanding/gg-ssms_graph-generating_state_space_models.md)
- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](../../CVPR2025/video_understanding/ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)
- [\[ICML 2025\] MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition](moma_modulating_mamba_for_adapting_image_foundation_models_to_video_recognition.md)
- [\[CVPR 2025\] BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](../../CVPR2025/video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)

<!-- RELATED:END -->
