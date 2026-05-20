---
title: >-
  [论文解读] Self-Consistency Preference Optimization
description: >-
  [ICML 2025][LLM对齐][自一致性] 将推理时的自一致性(self-consistency)概念引入训练阶段，通过投票机制构建偏好对并使用加权DPO损失进行迭代训练，在无需金标签的情况下大幅提升LLM的数学和逻辑推理能力。
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "自一致性"
  - "偏好优化"
  - "自训练"
  - "无监督推理对齐"
  - "DPO"
---

# Self-Consistency Preference Optimization

**会议**: ICML 2025  
**arXiv**: [2411.04109](https://arxiv.org/abs/2411.04109)  
**代码**: 无  
**领域**: LLM对齐/RLHF  
**关键词**: 自一致性, 偏好优化, 自训练, 无监督推理对齐, DPO

## 一句话总结

将推理时的自一致性(self-consistency)概念引入训练阶段，通过投票机制构建偏好对并使用加权DPO损失进行迭代训练，在无需金标签的情况下大幅提升LLM的数学和逻辑推理能力。

## 研究背景与动机

LLM的自对齐(self-alignment)是当前热门方向，核心目标是让模型在无人工标注的情况下自我提升。然而现有方法在**复杂推理任务**上效果有限，主要瓶颈在于：

**LLM无法可靠地自我评估推理正确性**：Huang et al. (2024) 证明LLM在有唯一正确答案的复杂问题上很难判断自己的回答是否正确，导致Self-Rewarding等方法在推理任务上失效

**外部奖励模型(RM)泛化能力差**：即使RM在推理任务上训练过，面对分布外问题仍会失败

**现有迭代训练方法依赖金标签**：STaR、IRPO等方法需要金标准答案来构建训练数据，限制了可扩展性

与此同时，**self-consistency**作为一种推理时技术已经被验证有效——对同一问题采样多个回答，选择最频繁出现的答案。其背后的直觉是：模型的错误通常是随机的，不太可能多次产生同样的错误答案。

**核心洞察**：能否将这种推理时的一致性信号用于训练阶段？一致的回答更可能正确，不一致的回答更可能错误——这天然形成了偏好对。

## 方法详解

### 整体框架

ScPO是一个**无监督迭代训练框架**，每轮迭代包含三个步骤：

1. **问题生成与过滤**：用当前模型生成新的推理问题，基于一致性过滤掉无法回答的问题
2. **构建自一致性偏好对**：对每个问题采样多个回答，选最一致的作为chosen、最不一致的作为rejected
3. **加权偏好优化训练**：用基于投票差距的加权损失函数训练模型

整个流程从种子模型 $M_0$ 出发，迭代产生 $M_1, M_2$（实验证明2轮迭代即可）。

### 关键设计

1. **投票函数与偏好对构建**：对每个问题 $x$，用当前模型采样 $k$ 个回答 $\{y_1, ..., y_k\}$。投票函数 $\mathcal{V}(y)$ 计算与 $y$ 最终答案相同的回答数量。选票最高的作为 chosen $y^+$，最低的作为 rejected $y^-$，且要求 $\mathcal{V}(y^+) \geq \tau$（阈值过滤）。这一设计的精妙之处在于：不需要知道正确答案是什么，仅通过回答之间的一致性来判断质量。

2. **基于一致性的问题生成与过滤**：用few-shot prompting让模型生成新问题（只需要问题模板，不需要对应答案）。然后对生成的问题采样回答并计算一致性，过滤掉所有回答投票都 $< \tau$ 的问题——这些很可能是无法解答的或格式不良的问题。这种方式允许生成更多样的问题，因为不需要同时生成正确答案。

3. **实例级加权机制**：偏好对的质量不均——投票差距大的对更可靠，差距小的更嘈杂。ScPO引入实例权重 $w(x) = (\mathcal{V}(y^+) - \mathcal{V}(y^-)) / k$，归一化到 $[0, 1]$。高置信度的对获得更大权重，低置信度的对权重趋近于0。当chosen和rejected投票相同时，$w(x)=0$，自动忽略该样本。

4. **迭代训练中的阈值递增**：随着训练迭代，模型变得更一致（实验验证），因此需要提高过滤阈值。例如GSM8K上，$M_1$ 训练用 $\tau=0.5k$，$M_2$ 训练提高到 $\tau=0.7k$。同时，为了在模型更一致后仍能采样到rejected回答，用更高温度(1.2)额外采样8个回答来增加多样性。

5. **半监督扩展(ScPO Semi-Sup.)**：当部分数据有金标签时，对有标签的问题直接用正确/错误构建偏好对且设 $w(x)=1$；对无标签的问题仍用自一致性。当所有数据都有标签时，退化为IRPO损失。这种设计使得ScPO可以无缝整合已有标注数据。

### 损失函数 / 训练策略

ScPO损失函数由两部分组成：

$$\mathcal{L}_{\text{ScPO}}(y^+, y^- | x) = \underbrace{-w(x) \log \sigma\left(\beta \log \frac{M_\theta(y^+|x)}{M_t(y^+|x)} - \beta \log \frac{M_\theta(y^-|x)}{M_t(y^-|x)}\right)}_{\text{Weighted DPO Loss}} \underbrace{- \frac{\alpha \cdot w(x)}{|y^+|} \log M_\theta(y^+|x)}_{\text{Weighted NLL Loss}}$$

- **加权DPO项**：推高chosen概率、压低rejected概率，权重由投票差距决定
- **加权NLL项**：直接在chosen回答上做监督学习，防止模型偏移太大

关键超参数设置：$\beta=0.5$, $\alpha=1$, 学习率 $5 \times 10^{-6}$（cosine调度），batch size 16，训练10个epoch。每轮迭代以上一轮模型 $M_t$ 作为DPO的参考模型。GSM8K/MATH用 $k=8$，ZebraLogic用 $k=16$（因为逻辑谜题很难采样到一致答案）。

## 实验关键数据

### 主实验

**GSM8K（Llama-3 Base 8B）**：

| 方法 | 迭代 | Greedy Acc. | SC Acc. | 说明 |
|------|------|-------------|---------|------|
| Seed (M0) | - | 41.17% | 51.80% | 零样本基线 |
| IRPO_RM | M2 | 50.11% | 61.25% | 外部RM打分 |
| LMSI | M2 | 56.71% | 62.55% | 一致性+NLL微调 |
| **ScPO_Unsup.** | **M2** | **63.91%** | **71.11%** | **+22.74% vs seed** |
| IRPO_Gold | M2 | 64.29% | 72.56% | 有金标签上界 |
| **ScPO_Semi-Sup.** | **M2** | **66.64%** | **74.75%** | **超越金标签IRPO** |

**MATH（Llama-3 Base 8B）**：

| 方法 | 迭代 | Greedy Acc. | SC Acc. | 说明 |
|------|------|-------------|---------|------|
| Seed (M0) | - | 14.46% | 18.20% | 零样本基线 |
| IRPO_RM | M2 | 18.08% | 22.64% | 外部RM打分 |
| LMSI | M2 | 16.96% | 20.20% | 一致性+NLL微调 |
| **ScPO_Unsup.** | **M2** | **19.72%** | **24.58%** | **+5.26% vs seed** |
| IRPO_Gold | M2 | 20.32% | 26.88% | 有金标签上界 |
| **ScPO_Semi-Sup.** | **M1** | **19.88%** | **27.35%** | **SC超越IRPO_Gold** |

**ZebraLogic（Llama-3 Instruct 8B）**：

| 方法 | Puzzle Acc. | Easy | Hard | Cell Acc. |
|------|------------|------|------|-----------|
| Llama-3 70B | 17.2% | 52.1% | 3.6% | 42.9% |
| Gemma-2 27B | 16.3% | 50.7% | 2.9% | 41.2% |
| Claude-3 Haiku | 14.3% | 47.9% | 1.2% | 37.9% |
| M0 (8B) | 11.6% | 40.0% | 0.4% | 39.1% |
| **ScPO M2 (8B)** | **18.1%** | **58.2%** | **2.5%** | **45.2%** |

### 消融实验

**加权损失 vs 无权重消融**：

| 配置 | GSM8K Greedy | MATH Greedy | 说明 |
|------|-------------|-------------|------|
| M1 w/ w(x)=1 | 58.53% | 15.92% | 无权重 |
| M1 w/ ScPO | 61.03% | 17.36% | 加权，+2.5%/+1.44% |
| M2 w/ w(x)=1 | 62.62% | 18.74% | 无权重 |
| M2 w/ ScPO | 63.91% | 19.72% | 加权，+1.29%/+0.98% |

**过滤阈值消融（MATH）**：

| 阈值 τ | 准确率差距 | 训练数据量 | 测试准确率 |
|--------|----------|-----------|-----------|
| 0.1k | 18% | 6.7K | 15.44% |
| 0.3k | 44% | 2.4K | 16.34% |
| **0.5k** | **57%** | **1.8K** | **17.36%** |
| 0.7k | 68% | 0.7K | 14.76% |

### 关键发现

1. **无监督ScPO与有金标签的IRPO差距<1%**：在GSM8K上greedy准确率仅差0.38%，证明自一致性可以有效替代金标签
2. **8B模型通过ScPO训练后超越70B模型**：ZebraLogic上ScPO训练的8B模型超越Llama-3 70B (+0.9%)、Gemma-2 27B (+1.8%)、Claude-3 Haiku (+3.8%)
3. **自一致性比外部RM更可靠**：分析显示ArmoRM在分类偏好对时比自一致性产生了更多错误排序（尤其在OOD场景ZebraLogic上多12.3%错误排序）
4. **模型一致性随迭代递增**：投票份额在每轮训练后显著提升，说明ScPO有效地将self-consistency分布蒸馏进了模型的单样本分布
5. **阈值存在最优权衡点**：τ=0.5k时达到最佳平衡——更低则噪声太多，更高则数据量不足

## 亮点与洞察

- **优雅的无监督信号设计**：将推理时技术(self-consistency)转化为训练信号，避免了LLM自评估和外部RM的缺陷。一致性作为正确性的代理指标，既简单又有效
- **加权损失的理论直觉清晰**：投票差距反映了模型对偏好标注的置信度，高置信度的对应该更被重视，这与不确定性加权的思路一脉相承
- **可扩展性强**：不依赖金标签意味着可以用模型自生成的问题无限扩展训练数据，突破了现有训练集的限制
- **与推理时SC互补**：训练阶段用ScPO、推理阶段用SC可以叠加收益，但SC边际收益随训练迭代递减（因为模型已经更一致了）

## 局限与展望

1. **依赖可解析的最终答案**：当前方法要求能从回答中提取离散的最终答案来计算一致性，对开放式任务（摘要、翻译等）不适用，作者建议未来用universal self-consistency扩展
2. **基础模型能力限制**：如果种子模型在某任务上完全无法产生一致回答，ScPO就无法bootstrap——MATH的数据量远小于GSM8K就是这个原因
3. **只验证了8B规模**：更大模型（70B+）是否仍有类似收益未知
4. **迭代收益递减**：第3轮迭代几乎无增益，SC推理收益也递减
5. **未结合过程奖励模型(PRM)**：与step-level验证的组合可能进一步提升

## 相关工作与启发

- **IRPO (Pang et al., 2024)**：ScPO的损失函数基于IRPO的DPO+NLL结构，但增加了实例级加权
- **Self-Rewarding LM (Yuan et al., 2024)**：类似的迭代训练框架，但用LLM-as-judge替代自一致性，在推理任务上效果差
- **STaR (Zelikman et al., 2022)**：迭代bootstrapping推理能力，但需要金标签过滤
- **LMSI (Huang et al., 2023)**：也用自一致性做无监督训练，但只用NLL损失而非偏好优化

**启发**：自一致性可能是一种被低估的训练信号。在更广泛的场景中，任何形式的"模型内部一致性"（跨采样、跨reformulation、跨语言）都可能作为偏好优化的代理标签。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将推理时技术转化为训练信号的思路新颖，但各组件（DPO、self-consistency、迭代训练）都是已有工作
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个数据集、多种基线、充分的消融（加权/阈值/一致性分析）
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述到位，图表直观
- **价值**: ⭐⭐⭐⭐ — 在无金标签场景下接近监督性能的实用方法，但局限于可解析答案的推理任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICML 2025\] AMPO: Active Multi-Preference Optimization for Self-play Preference Selection](ampo_active_multi-preference_optimization_for_self-play_preference_selection.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](../../ACL2026/llm_alignment/consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](../../NeurIPS2025/llm_alignment/longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)

</div>

<!-- RELATED:END -->
