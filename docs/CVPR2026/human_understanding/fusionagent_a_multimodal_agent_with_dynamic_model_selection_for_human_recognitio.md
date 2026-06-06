---
title: >-
  [论文解读] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition
description: >-
  [CVPR 2026][人体理解][模型融合] 本文提出 FusionAgent，一个基于多模态大语言模型（MLLM）的智能体框架，用于全身生物特征识别中的动态样本级模型选择——将每个专家模型（人脸识别/步态识别/行人重识别）封装为工具…
tags:
  - "CVPR 2026"
  - "人体理解"
  - "模型融合"
  - "多模态大语言模型"
  - "动态模型选择"
  - "生物特征识别"
  - "强化学习微调"
---

# FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition

**会议**: CVPR 2026  
**arXiv**: [2603.26908](https://arxiv.org/abs/2603.26908)  
**代码**: [https://github.com/FusionAgent](https://github.com/FusionAgent) (项目页面)  
**领域**: 人体理解  
**关键词**: 模型融合, 多模态大语言模型, 动态模型选择, 生物特征识别, 强化学习微调

## 一句话总结
本文提出 FusionAgent，一个基于多模态大语言模型（MLLM）的智能体框架，用于全身生物特征识别中的动态样本级模型选择——将每个专家模型（人脸识别/步态识别/行人重识别）封装为工具，通过强化微调（RFT）让 agent 学会根据每个测试样本的特征自适应选择最优模型组合，配合新提出的 ACT 分数融合策略，显著超越现有 SOTA 融合方法。

## 研究背景与动机

1. **领域现状**：全身人体识别（whole-body recognition）需要融合人脸、步态、体型等多种生物特征模态。不同的专家模型（FR/GR/ReID）各有擅长场景，通常通过分数融合（score fusion）进行集成。现有融合方法分为规则方法（Z-score、Min-max等）和学习方法（QME等），但它们都采用固定的模型组合策略——对所有测试样本使用相同的全模型组合。

2. **现有痛点**：（1）静态融合假设所有模型对每个样本都有贡献，但实际中人脸模型对背向镜头的人无法提供有用信息；（2）低质量输入的分数会污染融合结果，即使考虑质量感知（如 QME），低质量模型的分数仍会影响最终输出；（3）对所有样本调用所有模型，计算效率低且不必要。

3. **核心矛盾**：最优模型组合是样本相关的——不同质量、角度、分辨率的输入需要不同的模型子集。对所有样本使用相同的模型全集既浪费计算资源，又因为低质量分数的引入而降低了融合质量。

4. **本文目标**（1）如何为每个样本自适应选择最优模型子集？（2）如何在动态选择的异构模型分数之间进行有效融合？

5. **切入角度**：将模型选择建模为 MLLM agent 的工具调用决策问题。agent 可以观察输入样本的特征，reasoning 后决定调用哪些模型，通过强化学习从结果反馈中学习最优策略。

6. **核心 idea**：用 MLLM agent 做样本级动态模型选择，把"用什么模型"的决策从人工规则变成可学习的 reasoning 过程，配合锚定式 top-k 分数融合实现鲁棒的选择性集成。

## 方法详解

### 整体框架
FusionAgent 的流程为：每个生物特征模型被封装为工具（tool），提供分数向量和预测标签。MLLM agent（基于 Qwen2.5-VL-3B）接收多模态输入，通过多轮 ReAct 式推理（reasoning→action→observation→...），逐步选择模型、观察结果、决定是否继续选择或输出最终答案。agent 通过 GRPO（Group Relative Policy Optimization）进行强化微调，使用包含格式奖励、工具成功奖励、准确性奖励和度量奖励的复合奖励函数。最终分数通过 ACT（Anchor-based Confidence Top-k）融合策略整合。

### 关键设计

1. **ReAct 式多轮工具调用**

    - 功能：让 agent 逐步选择模型，每一步可以根据前一步的结果调整策略
    - 核心思路：采用 ReAct（reason-then-act）风格的多轮控制器，而非一次性生成完整计划。每一轮 agent 先推理当前样本特征和已有结果，然后选择一个模型调用，获得该模型的分数向量和预测标签。agent 再决定是否需要调用更多模型或直接输出答案。最大轮次限制为 4
    - 设计动机：（1）将指数级的模型组合空间分解为每步单模型选择，大幅降低学习难度；（2）允许 agent 根据中间结果动态调整策略（如第一个模型预测不确定时再调第二个）；（3）支持有效的 credit assignment

2. **度量奖励（Metric-based Reward）**

    - 功能：引导 agent 学习有效的模型选择策略，使选择结果最大化整体性能指标
    - 核心思路：采样 $N=6$ 个 rollout，每个 rollout 得到一个模型组合 $M_{o_i}$。以该组合为基础，保留 $\gamma=0.8$ 比例的样本组合不变，其余 20% 随机采样新组合以促进探索。然后用 ACT 融合这些选择下的分数矩阵，在整个训练集上计算综合度量 $R_{mat} = \text{Rank} + \text{mAP} + \text{TAR} - \text{FNIR}$。这个数据集级别的奖励让 agent 理解不同模型组合对全局性能的影响
    - 设计动机：不同于 per-sample 的准确性奖励，度量奖励考虑了 TAR@FAR 和 FNIR@FPIR 等需要在数据集级别计算的阈值依赖指标，更贴近实际部署需求

3. **ACT（Anchor-based Confidence Top-k）分数融合**

    - 功能：在动态选择的异构模型分数之间进行鲁棒融合
    - 核心思路：agent 选择的第一个模型作为"锚定模型" $m_a$，其分数向量直接全部保留。对其余选定模型，先做 Z-score 标准化解决尺度不一致问题，然后只保留每个模型的 top-$k$ 最高分数条目的贡献（$c_{m,q,g} = z_{m,q,g} \cdot s_{m,q,g}$ if $g \in \mathcal{T}_{m,q}$，否则为 0）。最终融合分数为 $\mathbf{s}_q' = \frac{1}{1 + |\mathbf{M}_q|}(\mathbf{s}_{m_a,q} + \sum_{m \in \mathbf{M}_q} \mathbf{c}_{m,q})$
    - 设计动机：锚定模型提供全局排序基础，top-k 过滤防止低置信度的冒名者分数拉高非匹配分数。这种"锚定+稀疏贡献"的策略有效解决了动态模型选择下分数尺度不对齐和异构性问题

### 损失函数 / 训练策略
- 使用 GRPO 优化，复合奖励 $R = R_f + R_{tool} + R_{acc} + R_{mat}$
- 基础模型为 Qwen2.5-VL-3B，使用 LoRA（rank=64, α=128）
- 学习率 $2 \times 10^{-5}$（线性衰减），KL 系数 $\beta = 0.04$
- 训练 200 步，4 张 H100 GPU，约 4 小时完成
- 所有生物特征模型权重在训练过程中冻结

## 实验关键数据

### 主实验——CCVID 数据集

| 方法 | Rank1↑ | mAP↑ | TAR↑ | FNIR↓ |
|------|--------|------|------|-------|
| AdaFace (单模型) | 94.0 | 87.9 | 75.7 | 13.0±3.5 |
| Z-score | 92.2 | 90.6 | 73.9 | 15.1±1.5 |
| QME (之前SOTA) | **94.1** | 90.8 | 76.2 | 12.3±1.4 |
| **FusionAgent (CoT)** | 93.4 | **92.6** | **85.9** | **10.1±1.5** |

TAR 从 76.2% 提升到 85.9%（+9.7%），FNIR 从 12.3% 降至 10.1%。

### LTCC 数据集

| 方法 | Rank1↑ | mAP↑ | TAR↑ | FNIR↓ |
|------|--------|------|------|-------|
| QME | 73.8 | 39.6 | 35.0 | 64.3±8.0 |
| **FusionAgent (CoT)** | **75.5** | **41.0** | **37.0** | **50.0±8.5** |

FNIR 从 64.3% 降至 50.0%（-14.3%），开放集搜索性能大幅提升。

### 消融实验

| 配置 | Rank1 | mAP | TAR | FNIR |
|------|-------|-----|-----|------|
| QME (baseline) | 73.8 | 39.6 | 35.0 | 64.3 |
| Agent + Z-score | 74.8 | **41.7** | **37.1** | 63.7 |
| Agent + FarSight | 74.8 | **41.7** | **37.2** | 62.5 |
| Agent + ACT (Ours) | **75.5** | 41.4 | 36.5 | **51.0** |

- Agent 选择 + 任何融合方法都优于 QME，证明动态选择是关键
- ACT 在 FNIR 上优势最大（-11.5%），因为 top-k 过滤有效抑制了冒名者分数

### 关键发现
- **动态模型选择是性能提升的主要驱动力**：即使用简单的 Z-score 融合，加上 agent 选择也优于 QME
- **Hard selection（使用全部模型）反而不如 agent 动态选择**：证明"用更多模型≠更好"，选择性融合至关重要
- **FNIR 获益最大**：开放集搜索场景下，冒名者分数的噪声被 top-k 过滤有效控制
- **Cross-domain 泛化**：MEVID 训练→LTCC 测试的零样本设置下仍能获得接近 in-domain 的性能
- **模型选择统计揭示数据集特性**：CCVID（面部清晰）主要选 AdaFace，LTCC/MEVID（监控低质量）主要选 ReID 模型

## 亮点与洞察
- **将模型融合重新定义为 agent 的工具选择问题**：这一框架将多年来的 score fusion 研究提升到了新的层面。不再是设计更好的融合公式，而是让 AI 自己决定用什么模型、怎么融合
- **ReAct 多轮设计**：将 $2^Z$ 的模型组合搜索空间分解为每步单选，使 RL 学习可行。同时支持根据中间结果动态调整策略
- **度量奖励的设计**：巧妙地将数据集级别的评估指标（TAR@FAR, FNIR@FPIR）编码为 RL 的奖励信号，使 agent 能学到超越 per-sample 准确性的全局优化策略
- **CoT 推理提供可解释性**：agent 的推理链条可以解释为什么选择某个模型组合（如"检测到清晰的正面人脸，选择人脸识别模型作为锚"），增强了系统可信度

## 局限与展望
- 基于 Qwen2.5-VL-3B，推理速度相对慢（CoT 模式 2.81s/样本），在实时场景中可能受限
- 3B 模型的 reasoning 能力有限，更大的 MLLM 可能带来更好的选择策略
- 当前的 tool 定义是固定的生物特征模型集合，泛化到新模型需要重新训练 agent
- ACT 的 top-k 超参数需要在训练集上调优，对不同数据集需要不同的 k 值
- 穷举搜索样本级最优模型组合计算上不可行，所以无法得知当前 agent 决策与理论最优的差距

## 相关工作与启发
- **vs QME**：QME 是质量感知的加权融合，但仍使用全部模型。FusionAgent 通过动态选择子集就能超越 QME，说明"选择用什么模型"比"给每个模型分配什么权重"更关键
- **vs 传统 score fusion (Z-score, FarSight)**：加上 agent 动态选择后，简单的融合方法就能超越复杂的学习方法。这启示我们融合策略的复杂度可能不是瓶颈，模型选择才是
- **vs SapiensID**：端到端的多模态识别模型，但缺乏模块化和可解释性。FusionAgent 通过 agent 框架实现了可解释的模块化融合

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 MLLM agent 引入模型融合/选择问题是新颖的，度量奖励设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多个基线、全面消融、跨域评估、统计分析、可视化案例
- 写作质量: ⭐⭐⭐⭐ 问题建模清晰，框架完整，但部分公式可以更简洁
- 价值: ⭐⭐⭐⭐ agent + tool use 的范式对多模型融合领域有启发性，可扩展到其他需要模型选择的场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[ICCV 2025\] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds](../../ICCV2025/human_understanding/egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)
- [\[CVPR 2026\] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambi.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
