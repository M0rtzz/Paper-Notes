---
title: >-
  [论文解读] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling
description: >-
  [强化学习] RuleReasoner 通过构建多样化的规则推理数据集 RuleCollection-32K 和提出域感知动态采样（Dads）策略，在 RLVR 框架下训练 8B 模型，在域内推理任务上比 OpenAI-o1 高 4.1%，在域外任务上高 10.4%，同时训练效率提升 ~1.4×。
tags:
  - "强化学习"
---

# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling

- **会议**: ICLR2026
- **arXiv**: [2506.08672](https://arxiv.org/abs/2506.08672)
- **代码**: 待公开
- **领域**: LLM 推理 / 强化学习
- **关键词**: rule-based reasoning, RLVR, dynamic sampling, GRPO, domain reweighting, OOD generalization

## 一句话总结

RuleReasoner 通过构建多样化的规则推理数据集 RuleCollection-32K 和提出域感知动态采样（Dads）策略，在 RLVR 框架下训练 8B 模型，在域内推理任务上比 OpenAI-o1 高 4.1%，在域外任务上高 10.4%，同时训练效率提升 ~1.4×。

## 研究背景与动机

**领域现状**: 基于规则的推理（rule-based reasoning）是 AI 推理的基础能力，涵盖法律、数学、医疗诊断等领域。大型推理模型（LRM）通过强化学习（RL）获得了长链思考能力，但在真实场景中仍面临规则格式多样、类型复杂、组合爆炸等挑战。

**痛点**: (1) 传统方法依赖模型规模扩大或从更强模型蒸馏，成本高且不可持续；(2) 随着上下文窗口扩大，模型出现"lost in the middle"现象，难以关注相关的规则和事实；(3) 现有 RLVR 方法对多域训练数据的采样策略粗糙——静态混合导致域间不平衡，过度优化简单域而欠优化困难域。

**核心矛盾**: RLVR 在数学/代码推理上的成功尚未迁移到规则推理领域——缺乏高质量多样化的训练数据，且多域训练的数据调度问题未被充分研究。

**目标**: 训练一个小型（4B/8B）但在规则推理上超越前沿 LRM（o1, R1）的专用推理模型，同时提高训练效率。

**切入角度**: 从**数据**和**采样策略**两个维度同时改进 RLVR：(1) 构建覆盖 8 类规则推理任务的 RuleCollection-32K；(2) 设计基于历史奖励的动态域采样算法 Dads，自动调度训练批次中各域的比例。

**核心 idea**: 用域感知动态采样（Dads）替代静态数据混合——每个训练步骤根据各域的历史奖励计算"欠优化程度"，动态提高欠优化域的采样权重，实现自适应在线数据调度。

## 方法详解

### 整体框架

RuleReasoner 想解决的是：怎么用一个只有 4B/8B 的小模型，在规则推理上反超 o1、R1 这类前沿大模型，同时不靠堆规模、不靠从更强模型蒸馏。它的答案是把改进同时压在「数据」和「采样调度」两个点上——准备一份覆盖面足够广的规则推理数据，再让训练过程自己决定每一步该多练哪个域。

整条流水线是标准 RLVR 套一个动态调度器：先在 RuleCollection-32K 上跑 RLVR，每个训练步先 rollout、算出各域这一批的平均奖励，用指数加权移动平均把它并进对该域奖励的历史估计，再经 softmax 算出各域权重，按这个权重重采样下一批次。策略优化用一个精简过的 GRPO 变体（去掉 KL 项和熵奖励），并且每个样本的上下文规则顺序都随机打乱，防止模型靠位置记规则。

### 关键设计

**1. 域感知动态采样（Dads）：让训练自己把算力从已收敛域挪到欠优化域。**

静态数据混合最大的问题是它对训练动态一无所知——像 ProntoQA 这种简单域很快就收敛了，继续按固定比例采样纯属浪费；而 AR-LSAT 这种困难域明明还没练透，却拿不到足够的样本。Dads 要解决的就是这个失配。

它的做法是给每个域 $d_i$ 维护一个指数加权移动平均奖励 $\widetilde{r}_{s,d_i}$，逐步更新对该域当前水平的估计：

$$\widetilde{r}_{s,d_i} = \alpha \widetilde{r}_{s-1,d_i} + (1-\alpha) \bar{r}_{s,d_i}$$

然后把「欠优化程度」定义成它离满分的差距 $v_{s,d_i} = 1 - \widetilde{r}_{s,d_i}$，再用带温度的 softmax 把这个差距转成下一步的采样权重：

$$w_{s,d_i} = \frac{\exp(v_{s,d_i}/\tau) + \epsilon}{\sum_{j=1}^n [\exp(v_{s,d_j}/\tau) + \epsilon]}$$

其中 $\alpha=0.5$ 做平滑、避免奖励估计来回抖动，$\tau=0.5$ 控制权重的锐度，$\epsilon=0.1$ 给每个域兜一个最小采样概率、不至于彻底饿死任何一个域。整体效果就是一个**在线调度器**：哪个域奖励越低（越欠练），下一批就分到越多样本，算力自动从收敛域流向困难域。这也是它和 AdaRFT 这类课程学习的关键区别——后者要靠人类先验或对成功率的预估来排课程，Dads 只看历史奖励、完全自适应。

**2. RuleCollection-32K 数据集：用五条原则把规则推理的多样性铺开。**

RLVR 在数学、代码上的成功之所以没能迁移到规则推理，很大程度上是缺一份足够多样的训练数据。RuleCollection-32K 就是按五条原则攒出来补这个缺口的：**变化深度**上覆盖 0–7 跳推理，天然支持从简单到复杂的课程；**格式**上既有显式规则（前提/约束）也有隐式规则（原则/上下文）；**推理类型**上同时包含演绎、归纳、分析推理；**上下文依赖**上要求规则必须结合具体问题自适应套用、不能靠死记；**评测鲁棒性**上优先选布尔/选择题，方便精确判分。落地成 8 个任务：Clutrr（归纳），ProntoQA/ProofWriter（演绎），FOLIO/LogicNLI（一阶逻辑），以及 AR-LSAT/Logical Deduction/LogiQA（其他）。这种刻意拉开格式和难度的设计，正是 Dads 能发挥作用的前提——各域差异够大，动态调度才有意义。

**3. 训练正则化：三个去记忆化措施，逼模型学规则而不是背数据集。**

为了避免模型识别出特定数据集、或直接记住规则模式，作者加了三项措施。一是**去掉熵奖励**，因为这里没有冷启动 bootstrap，留着熵奖励容易导致熵爆炸。二是**去掉 KL 散度项**：规则奖励函数本身是精确匹配、消除了分布偏移的顾虑，去掉 KL 既省计算又能鼓励探索。三是**打乱规则顺序**，每个训练样本里上下文规则的排列都随机重排，断掉模型靠位置去记规则的捷径。三者合起来把"投机取巧"的路都堵死，让奖励真正来自规则推理本身。

### 损失函数

使用基于精确匹配的**规则奖励函数**：

$$\mathcal{R}_{\text{EM}}(\hat{y}, y) = \begin{cases} 1 & \text{is\_equivalent}(\hat{y}, y) \\ -1 & \text{otherwise} \end{cases}$$

策略优化采用 GRPO 目标，去除 KL 项和熵奖励，仅保留裁剪后的策略梯度。严格在线策略——每次 rollout 后仅更新一次梯度。

## 实验关键数据

### 主实验：域内 8 任务 Pass@1 对比

| 方法 | Clutrr | ProntoQA | ProofWriter | FOLIO | LogicNLI | AR-LSAT | Log.Ded. | LogiQA | **Avg** |
|------|--------|----------|-------------|-------|----------|---------|----------|--------|---------|
| OpenAI o1 | 52.2 | 91.0 | 91.0 | 77.0 | 60.0 | 98.0 | 88.0 | 82.1 | 79.9 |
| Claude-3.7 | 65.7 | 92.8 | 90.0 | 74.7 | 58.0 | 76.2 | 97.0 | 81.5 | 79.5 |
| DeepSeek-R1 | 71.6 | 40.0 | 27.0 | 72.7 | 49.0 | 89.7 | 98.3 | 85.0 | 66.7 |
| DAPO | 86.5 | 96.0 | 94.8 | 80.9 | 65.8 | 40.0 | 95.3 | 74.6 | 79.2 |
| AdaRFT | 92.5 | 96.0 | 97.4 | 81.8 | 64.4 | 44.6 | 96.6 | 80.5 | 81.7 |
| **RuleReasoner-8B** | **95.5** | **96.4** | **97.0** | **84.7** | 70.4 | 46.8 | 98.3 | 83.5 | **84.0** |

> RuleReasoner-8B 以 84.0% 平均准确率超越 OpenAI-o1（79.9%）4.1 个百分点，且各域性能方差最低。

### 消融实验：域外 OOD 泛化

| 方法 | BBH | BBEH | ProverQA | **OOD Avg** |
|------|-----|------|----------|------------|
| Qwen3-8B (base) | 22.9 | 13.0 | 15.3 | — |
| SFT w/ Long CoT | 31.3 | 28.0 | 43.8 | 34.4 |
| GRPO | 35.5 | 24.3 | 34.1 | 31.3 |
| DAPO | 39.8 | 27.3 | 42.8 | 36.6 |
| OpenAI o1 | 46.4 | 33.5 | 52.5 | 44.1 |
| **RuleReasoner-8B** | **52.3** | **45.8** | **65.4** | **54.5** |
| **RuleReasoner-4B** | — | — | — | **54.5** (Δ+7.3 vs o1) |

> RuleReasoner-8B 在三个 OOD 基准上平均超越 o1 达 10.4 个百分点。即使 4B 版本也达到 78.3% 三基准平均。

### 关键发现

1. **Dads 优于静态课程学习**: 相比 data-balance RL（79.1%）和 easy-to-hard RL（80.4%），Dads（84.0%）在 ID 任务上高出 3-5 个百分点——在线调度远优于静态排列
2. **训练效率**: RuleReasoner 达到 DAPO 同等 OOD 性能需要少 ~72 步（约 1.4× 加速），且无需额外 rollout 计算
3. **SFT vs RLVR**: SFT 在 ID 上接近 RLVR（81.9 vs 84.0），但 OOD 上差距巨大（34.4 vs 54.5），确认 **RL 泛化、SFT 记忆**
4. **元自省能力涌现**: 训练后模型展现出自我验证和逻辑一致性检查的能力——在未见过的规则上也能自纠错

## 亮点与洞察

- Dads 的设计精妙简约：仅用历史奖励估计域权重，无需代理模型或人类先验，可作为 RLVR 的通用数据调度插件
- 8B 模型超越 o1/R1 等前沿推理模型，说明规则推理场景下**专用数据+智能调度**比模型规模更重要
- RuleCollection-32K 的构建原则（变化深度、多格式、上下文依赖）值得其他推理数据集借鉴
- 实验中 DeepSeek-R1 在部分任务上崩溃（ProntoQA 40.0%、ProofWriter 27.0%），暴露了通用推理模型在结构化规则推理上的脆弱性

## 局限性

- 训练数据受限于当前可获得的规则推理任务，可能未覆盖自然语言中所有规则格式和复杂度
- 规则过滤质量有限——含噪或冗余规则可能影响推理质量
- 未在 >8B 模型上验证，大规模模型可能获益更多但受计算限制
- 评测以精确匹配为主，对需要自由文本输出的规则应用场景评估不足

## 相关工作与启发

- **Logic-RL** (Xie et al., 2025): 逻辑推理上的 RLVR，但属于"无规则"推理，与本文"规则给定"设定不同
- **DAPO** (Yu et al., 2025): 通过过采样+过滤提升 RLVR 效率，但未做细粒度域调度
- **AdaRFT** (Shi et al., 2025): 课程学习式采样但依赖人类先验或模型成功率估计，Dads 完全自适应
- **GRPO** (Shao et al., 2024): 基础策略优化算法，RuleReasoner 在此基础上加入域感知采样
- **启发**: "域感知动态采样" 的思路可推广到数学推理、代码生成等任何多域 RLVR 训练场景

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — Dads 作为 RLVR 数据调度的通用方法有较强新意，但核心是 softmax 重加权并非全新范式
- **实验**: ⭐⭐⭐⭐⭐ — 基线覆盖前沿 LRM + SFT + 多种 RLVR + 课程学习，消融详尽，5 次运行报告均值和标准差
- **实用性**: ⭐⭐⭐⭐ — 方法简洁高效，可直接集成到现有 RLVR 流程
- **写作**: ⭐⭐⭐⭐ — 算法伪代码清晰，但表格较多导致部分内容密集

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)

</div>

<!-- RELATED:END -->
