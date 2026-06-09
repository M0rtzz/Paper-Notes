---
title: >-
  [论文解读] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness
description: >-
  [ICML 2025][LLM推理][Chain-of-Thought] 本文从有效性（effectiveness）和忠实性（faithfulness）两个维度系统分析了 CoT 的性能影响因素，发现问题难度、信息增益和信息流是影响 CoT 有效性的关键因子…
tags:
  - "ICML 2025"
  - "LLM推理"
  - "Chain-of-Thought"
  - "推理有效性"
  - "推理忠实性"
  - "信息增益"
  - "信息流"
  - "归因分析"
---

# Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness

**会议**: ICML 2025  
**arXiv**: [2405.18915](https://arxiv.org/abs/2405.18915)  
**代码**: [BugMakerzzz/better_cot](https://github.com/BugMakerzzz/better_cot)  
**领域**: LLM推理  
**关键词**: Chain-of-Thought, 推理有效性, 推理忠实性, 信息增益, 信息流, 归因分析

## 一句话总结
本文从有效性（effectiveness）和忠实性（faithfulness）两个维度系统分析了 CoT 的性能影响因素，发现问题难度、信息增益和信息流是影响 CoT 有效性的关键因子，而不忠实 CoT 的根因在于模型在预测答案时绕过 CoT 直接从问题中召回了正确信息，并据此提出 QUIRE 方法同时提升 CoT 的有效性和忠实性。

## 研究背景与动机
1. **领域现状**：CoT 技术让 LLM 在复杂推理任务上表现出色，尤其 DeepSeek-R1、o1 等推理模型通过 RL 扩展 CoT 过程甚至超越了人类在竞赛数学上的表现。
2. **现有痛点**：CoT 并非万能药——在某些任务上 CoT 不仅无效甚至有害（如常识推理），且存在不忠实问题（错误 CoT 仍能导致正确答案）。
3. **评估工作的不足**：
    - 有效性方面：已有工作仅停留在"CoT 在含数学符号的任务上表现好"的表面结论，缺乏对底层影响因素的深入分析。
    - 忠实性方面：已有工作主要设计各种方法判断 CoT 是否忠实，但缺乏对不忠实现象成因的解释。
4. **本文切入角度**：从信息论视角出发，通过信息增益（Information Gain）和信息流（Information Flow）等工具量化 CoT 在推理过程中的信息交互模式，不仅诊断问题，还提供了可操作的改进方案。
5. **核心 idea**：不忠实的 CoT 丢失了问题中的关键信息，但模型在预测答案时能从问题中"召回"这些丢失的信息——利用这一机制反向增强 CoT 生成，可同时改善有效性和忠实性。

## 方法详解

### 整体框架
本文分为分析（§3-§4）和应用（§5）两大部分：
- **§3 有效性分析**：识别问题难度、信息增益、信息流三个影响 CoT 有效性的关键因子
- **§4 忠实性分析**：通过问题→CoT→答案的三方信息交互解释不忠实 CoT 的成因
- **§5 QUIRE 方法**：基于分析结论设计的算法，同时提升有效性和忠实性

### 有效性分析：三大关键因子

#### 因子1：问题难度（Problem Difficulty）
- **难度量化方式**：对每个问题采样 10 次无 CoT 回答，按 pass@1 率分为 5 个难度等级（pass@1 < 0.1 为最难的 level 5，> 0.8 为最简单的 level 1）
- **结论 Cl.1**：CoT 在困难问题上更有效。在低难度问题上 CoT 提升很小甚至降低性能，但在高难度问题上显著提升准确率
- **任务差异的解释**：数学推理数据集中高难度问题占比更多，常识推理中低难度问题占比更多，因此 CoT 在数学推理上更有效

#### 因子2：信息增益（Information Gain）
通过信息论中的信息增益量化 CoT 从问题中获取了多少信息：

$$IG(C, Q) = H(C) - H(C|Q) = -\sum_{i=1}^{n} p(c_i|C_{i-1}) \log p(c_i|C_{i-1}) + \sum_{i=1}^{n} p(c_i|C_{i-1};Q) \log p(c_i|C_{i-1};Q)$$

其中 $p(\cdot)$ 为模型输出概率，$C_{i-1}$ 为 CoT 的前 $i-1$ 个 token。IG 越大说明 CoT 对问题的"依赖"越高，即 CoT 自身提供的额外信息越少。

- **结论 Cl.2**：当 CoT 提供了问题本身不包含的额外信息时（即 IG 低），CoT 更有效。数学推理的 IG 最低（CoT 提供大量额外推导信息），常识推理的 IG 最高（CoT 基本复述问题信息）

#### 因子3：信息流（Information Flow）
使用积分梯度归因（Integrated Gradient Attribution, IGA）追踪 CoT 各位置到答案的信息传递：

$$I(x_n, y_m) = E(x_n) \int_{\alpha=0}^{1} \frac{\partial f(\alpha y_m)}{\partial E(x_n)} d\alpha$$

归一化后得到归因效应分数 $AE(x_n, y_m)$，再对答案 token 取平均得到 $AAE(c, A)$。

进一步定义**信息流单调性（MIF）**为 CoT 步骤位置与 AAE 之间的 Spearman 相关系数：

$$MIF(C, A) = 1 - \frac{6\sum_{i=1}^{n}[n+1-i-R(AAE(c_i, A))]^2}{n(n^2-1)}$$

- **结论 Cl.3**：当信息流随 CoT 过程递增时（MIF 高），CoT 更有效。如 GSM8K 的 AAE 曲线明显上升，而 ECQA 基本平稳

### 忠实性分析：三步诊断

通过手动评估 50 组 CoT-答案对发现：**逻辑推理任务的不忠实比例最高**（ProntoQA 高达 17/50）。

#### 步骤1：问题→CoT（Cl.4）
- 不忠实 CoT 的 IG(Q,C) 更低，说明 CoT 从问题上下文获取的信息更少
- **结论**：不忠实 CoT 丢失了上下文中的关键正确信息

#### 步骤2：CoT→答案（Cl.5）
- 不忠实 CoT 到答案的 AAE 显著低于忠实 CoT
- **结论**：不忠实 CoT 与答案之间的信息交互更少

#### 步骤3：问题→答案（Cl.6）
- 对问题中每个陈述按其到答案的 AAE 排序，在不忠实情况下，CoT 中丢失的正确陈述更多地出现在 top-k 高 AAE 位置
- **结论**：当不忠实 CoT 发生时，模型在预测答案时绕过 CoT，直接从问题中"召回"了丢失的正确信息

### QUIRE 方法

基于以上六条结论，提出 **QUestion Information Recall and Enhancement（QUIRE）**：

#### 组件1：AAE Recall（信息召回）
1. 先用 Self-Consistency 生成初始答案 $A$
2. 计算问题中每个陈述 $S$ 到答案的 $AAE(S, A)$
3. 选取 top-k 个 AAE 最高的陈述作为额外提示（hints）
4. 将这些提示加入输入 prompt，引导生成信息更完整的新 CoT

#### 组件2：IG Vote（信息增益加权投票）
1. 生成多条信息增强的 CoT（可结合 SC 技术）
2. 对每条 CoT 计算 $IG(Q, C)$ 作为质量权重——IG 越高说明 CoT 从问题获取了更多信息，幻觉陈述越少
3. 用 IG 值作为 SC 投票权重，选出最终答案

## 实验关键数据

### 主实验结果（Llama3.1-8B）

| 方法 | ProofWriter Acc | ProofWriter BS | ProofWriter FBS | ProntoQA Acc | ProntoQA BS | ProntoQA FBS |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| CoT | 59.2 | 64.9 | 55.7 | 86.8 | 86.1 | 78.0 |
| Self-Consistency | 60.6 | 65.0 | 57.8 | 93.2 | 87.5 | 83.6 |
| Least-to-Most | 54.0 | 60.4 | 56.4 | 90.0 | 77.3 | 72.6 |
| Self-Refine | 51.6 | 65.9 | 53.4 | 88.5 | 91.5 | 84.5 |
| **QUIRE (Ours)** | **63.0** | **66.6** | **58.0** | **95.0** | **92.7** | **89.2** |
| - AAE Recall | 60.2 | 65.1 | 57.0 | 95.0 | 87.5 | 84.6 |
| - IG Vote | 62.8 | 64.1 | 56.6 | 94.3 | 87.0 | 83.4 |

> **关键发现**：QUIRE 在准确率上最高提升 2.4%（ProofWriter），在忠实性（FBS）上最高提升 5.6%（ProntoQA）。消融实验表明 AAE Recall 和 IG Vote 均有独立贡献。

### CoT 不忠实统计（Llama3.1-8B, 50 samples/dataset）

| CoT→答案 | GSM | AQuA | ProofWriter | ProntoQA | WinoGrande | SocialIQA |
|:--------:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓→✓ | 41 | 25 | 14 | 27 | 34 | 40 |
| ✓→✗ | 0 | 0 | 0 | 0 | 1 | 0 |
| ✗→✓（不忠实） | 1 | 1 | **7** | **17** | 1 | 0 |
| ✗→✗ | 8 | 24 | 29 | 6 | 14 | 10 |

> **关键发现**：逻辑推理任务（ProofWriter 14%, ProntoQA 34%）的不忠实比例远高于数学推理和常识推理。

## 亮点与洞察
1. **信息论视角系统化**：首次用 IG、AAE、MIF 三个量化指标统一解释 CoT 的有效性和忠实性，形成完整的分析框架
2. **不忠实成因的清晰解释**：通过三步信息交互分析，揭示了"模型从问题中直接召回丢失信息"这一机制，比单纯判断"忠实/不忠实"更有洞察力
3. **分析到应用的闭环**：QUIRE 方法直接源于分析结论，且实验证明了"忠实性提升→有效性提升"的因果关系
4. **白盒分析方法论价值**：IGA + 信息增益的组合为 CoT 机理研究提供了可复用的分析工具

## 局限性 / 可改进方向
1. **仅限开源模型**：由于需要梯度信息，无法分析 GPT-4 等黑盒模型
2. **缺乏理论证明**：忠实性→有效性的因果关系仅有实证支持，缺少理论保证
3. **数据规模有限**：不忠实性分析基于人工评估 50 个样本，规模较小
4. **QUIRE 计算开销**：需要额外计算 AAE 和 IG，需要梯度回传，推理成本较高
5. **任务覆盖**：分析和方法主要验证在逻辑推理任务上，数学推理和常识推理的改进幅度待验证

## 相关工作与启发
- **CoT 有效性评估**：Sprague et al. (2024) 和 Xu & Ma (2024) 评估了 CoT 在不同任务类型上的效果，但本文进一步识别了底层因素
- **CoT 忠实性评估**：Bao et al. (2024) 用因果中介分析衡量忠实度，Lanham et al. (2023) 测量忠实性但缺乏解释，本文补充了成因分析
- **信息流分析**：Wu et al. (2023) 和 Wang et al. (2024) 使用 IGA 分析模型内部信息传递，本文将其应用到 CoT-答案信息交互上
- **启发**：QUIRE 的"先生成初始答案再用归因信号增强重生成"范式可推广到 RAG、self-reflection 等场景

## 评分与推荐度
⭐⭐⭐⭐ — 分析深入系统，信息论工具使用巧妙，但应用范围局限于需要梯度信息的开源模型和逻辑推理任务，QUIRE 的实用性受限于计算开销。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](pcot_persuasion-augmented_chain_of_thought_for_detecting_fake_news_and_social_me.md)
- [\[NeurIPS 2025\] Reasoning Models Better Express Their Confidence](../../NeurIPS2025/llm_reasoning/reasoning_models_better_express_their_confidence.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](../../NeurIPS2025/llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](../../CVPR2025/llm_reasoning/interleaved-modal_chain-of-thought.md)
- [\[ICML 2025\] MARGE: Improving Math Reasoning for LLMs with Guided Exploration](marge_improving_math_reasoning_for_llms_with_guided_exploration.md)

</div>

<!-- RELATED:END -->
