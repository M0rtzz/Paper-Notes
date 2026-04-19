---
title: >-
  [论文解读] Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time
description: >-
  [ICML 2025][LLM对齐][推理时对齐] 提出 SITAlign——基于有界理性的满意决策框架，在推理时最大化主要目标（如有用性）同时确保次要目标（如无害性）满足阈值约束，通过对偶理论求解，在 GPT-4 评估上相比多目标解码 SOTA 提升 22.3% 胜率。
tags:
  - ICML 2025
  - LLM对齐
  - 推理时对齐
  - 有界理性
  - 满意决策
  - 多目标对齐
  - 受控解码
---

# Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time

**会议**: ICML 2025  
**arXiv**: [2505.23729](https://arxiv.org/abs/2505.23729)  
**代码**: 无  
**领域**: LLM对齐  
**关键词**: 推理时对齐, 有界理性, 满意决策, 多目标对齐, 受控解码

## 一句话总结
提出 SITAlign——基于有界理性的满意决策框架，在推理时最大化主要目标（如有用性）同时确保次要目标（如无害性）满足阈值约束，通过对偶理论求解，在 GPT-4 评估上相比多目标解码 SOTA 提升 22.3% 胜率。

## 研究背景与动机

**领域现状**：LLM 对齐需要同时满足多个属性（安全、有用、真实、简洁），现有方法通常将其建模为多目标加权优化。

**现有痛点**：加权组合假设所有目标自维度都应同时最大化，但确定权重困难且忽略了人类实际决策方式——Herbert Simon 的有界理性理论表明，人类采用"满意决策"（satisficing）策略：最优化关键目标，其他目标满足阈值即可。

**核心矛盾**：多目标加权优化难以处理冲突目标；且微调方法计算昂贵，无法适应不同用户/场景的个性化阈值。

**本文目标**：如何在推理时动态地实现"主目标最大化 + 次要目标达标"的满意决策对齐？

**切入角度**：将对齐建模为约束优化——主目标为目标函数，次要目标为不等式约束。

**核心 idea**：用对偶理论将约束问题转化为无约束拉格朗日问题，在受控解码框架下逐 token 求解。

## 方法详解

### 整体框架
SITAlign 在推理时逐 token 生成：
1. 给定主奖励模型 $r_1$ 和次要奖励模型 $r_2, \dots, r_m$，以及阈值 $\beta_2, \dots, \beta_m$
2. 目标：$\max_\pi \mathbb{E}[Q_1^*(s_t, z)] - \beta_1 D_{KL}[\pi || \pi_{sft}]$ s.t. $\mathbb{E}[Q_i^*(s_t, z)] \geq \beta_i$
3. 通过拉格朗日对偶将约束吸收，动态调整 token 分布

### 关键设计

1. **满意对齐建模**:

    - 功能：将多维偏好分解为"最大化主目标 + 次要目标达标"
    - 核心思路：inspiration 来自有界理性——无害性超过某阈值后继续提升边际收益递减，不如把资源集中在有用性上
    - 设计动机：实验证实90% 的无害回复奖励分数 ≥ -12（PKU-SafeRLHF），设定阈值即可

2. **对偶求解受控解码**:

    - 功能：将约束优化转为拉格朗日对偶问题
    - 核心思路：引入拉格朗日乘子 $\lambda_i$，将约束目标合并为 $r_1 + \sum \lambda_i r_i$ 的加权解码，$\lambda_i$ 随解码过程动态更新
    - 设计动机：对偶方法将约束满足问题转为自适应权重调整，避免手动设定权重

### 损失函数 / 训练策略
- 完全推理时方法，无需微调
- 乘子更新使用次梯度上升
- 理论提供次优性界

## 实验关键数据

### 主实验

| 设置 | 方法 | GPT-4 胜率 (主目标) | 约束满足 |
|------|------|-------------------|---------|
| 有用性↑ + 无害性≥阈值 | Multi-obj decoding | 35.2% | ✓ |
| | **SITAlign** | **57.5%** (+22.3%) | ✓ |
| 有用性↑ + 幽默≥阈值 | Multi-obj decoding | 41.1% | ✓ |
| | **SITAlign** | **51.3%** (+10.2%) | ✓ |

### 消融实验
| 配置 | 主目标胜率 | 约束满足率 | 说明 |
|------|-----------|-----------|------|
| 仅主目标（无约束） | 高 | 低 (~60%) | 忽略安全性 |
| 等权重多目标 | 中 | 中 | 权重难调 |
| SITAlign | 高 | 高 (>95%) | 最优平衡 |

### 关键发现
- SITAlign 在主目标上始终优于多目标加权方法，同时约束满足率 >95%
- 满意决策范式比加权最优化更符合实际需求
- 推理时方法可动态调整阈值，无需重新训练

## 亮点与洞察
- **有界理性视角**与 LLM 对齐的结合非常自然——不需要所有维度都最优，达标即可
- 推理时方法的灵活性：不同用户可设不同阈值而无需重新训练
- 对偶方法自动学习约束的"松紧"程度，避免手动权重调优

## 局限与展望
- 需要预定义阈值，虽然文中提供指导但仍需领域知识
- Q 函数估计的质量影响性能（需要合理的值函数近似）
- 推理时逐 token 优化增加延迟
- 仅在 7B 模型上验证

## 相关工作与启发
- **vs Shi et al. (多目标解码)**: 加权组合所有目标，不区分主次
- **vs ARGS/DeAL**: 单目标受控解码，不处理多维约束
- **vs ConfPO**: ConfPO 是训练时方法，SITAlign 是推理时方法，互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 有界理性 × LLM 对齐的创新视角
- 实验充分度: ⭐⭐⭐⭐ 三个评估设置，GPT-4 评估
- 写作质量: ⭐⭐⭐⭐ 动机有说服力，理论+实验兼备
- 价值: ⭐⭐⭐⭐⭐ 提出对齐的新范式

<!-- RELATED:START -->

## 相关论文

- [Inference-time Alignment in Continuous Space](../../NeurIPS2025/llm_alignment/inference-time_alignment_in_continuous_space.md)
- [MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignment](mpo_an_efficient_post-processing_framework_for_mixing_diverse_preference_alignme.md)
- [Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models](layer-wise_alignment_examining_safety_alignment_across_image_encoder_layers_in_v.md)
- [On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [AlphaPO: Reward Shape Matters for LLM Alignment](alphapo_reward_shape_matters_for_llm_alignment.md)

<!-- RELATED:END -->
