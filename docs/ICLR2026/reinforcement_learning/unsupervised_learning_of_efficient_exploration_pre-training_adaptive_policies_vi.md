---
title: >-
  [论文解读] Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals
description: >-
   提出 ULEE 方法，通过对抗式目标生成和基于适应后难度的课程学习，在无监督环境中元学习出具备高效探索和快速适应能力的预训练策略。
tags:

---

# Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2601.19810](https://arxiv.org/abs/2601.19810)
- **代码**: 开源（文中提到 all code is open-sourced）
- **领域**: 强化学习 / 无监督预训练 / 元学习
- **关键词**: unsupervised RL, meta-learning, goal generation, curriculum learning, exploration

## 一句话总结

提出 ULEE，一种无监督元学习方法，通过对抗式自生成目标课程训练自适应策略，在 XLand-MiniGrid 基准上实现高效探索与少样本适应。

## 研究背景与动机

大规模预训练在 CV 和 NLP 领域取得巨大成功，但 RL 中仍以单任务从零训练为主流范式。无监督 RL 旨在无外部奖励下获取可迁移策略（foundation policies），核心挑战包括：

**数据收集**：智能体应收集什么数据？收集的数据直接由其行为决定。

**目标生成**：自主生成目标的智能体如何有效地生成、选择和利用目标？

**下游泛化**：当目标任务分布广泛且超出预训练分布时，零样本解决所有任务不可行，需要高效的适应能力。

现有方法存在以下不足：
- GoalGAN 等基于即时表现评估目标难度，没有考虑适应后的表现
- 目标条件策略在目标未知或表示不可解释时失效
- 无监督元学习设置仍然 largely underexplored

## 方法详解

### 整体框架

ULEE（Unsupervised Learning of Efficient Exploration）包含四个核心组件：

1. **Pre-trained Policy $\pi$**：非目标条件的 in-context learner，通过多 episode 交互历史进行适应
2. **Goal-search Policy $\pi_{gs}$**：对抗式训练的目标搜索策略，寻找困难目标
3. **Difficulty Predictor**：预测目标在适应后的难度
4. **Goal Selection**：基于中等难度范围的目标采样

### 关键设计 1：后适应难度度量

不同于先前工作基于即时表现评估难度，ULEE 定义基于**适应后**表现的难度：

$$d(g; \pi, M) = 1 - \mathbb{E}_{\rho_M, P_M, \pi}\left[\frac{1}{K}\sum_{j=H-K+1}^{H} \mathbf{1}\left\{\exists t: f(s_{t+1}^{(j)}) = g\right\}\right]$$

其中 $H$ 为总 episode 数，仅评估最后 $K$ 个 episode 的成功率，前面 $H-K$ 个 episode 用于探索和适应。这更好地匹配了评估场景。

### 关键设计 2：对抗式目标生成

Goal-search Policy 训练目标为最大化所发现目标的难度：

$$r_t^{gs} = r^{gs}(s_t; \pi, M) = d(f(s_t); \pi, M)$$

通过在每个环境中先运行 $\pi_{gs}$ 收集候选目标集 $GC_M$，再根据难度范围 $[LB, UB]$ 进行采样。

### 关键设计 3：In-context 元学习

策略 $\pi$ 训练为最大化多 episode lifetime 的累积折扣回报：

$$\mathcal{J}(\pi) = \mathbb{E}_{M \sim \mu^{\text{unsup}}, g \sim p(g|M)}\left[\mathbb{E}_{\rho_M, P_M, \pi}\left[\sum_{j=1}^{H}\sum_{t=0}^{T-1} \gamma^{(j-1)T+t} r_t^{(j)}\right]\right]$$

使用 Transformer-XL 作为骨架网络，在多 episode 交互中通过历史上下文进行 in-context 学习。

### 损失函数

- **Difficulty Predictor**：监督 L2 回归损失

$$\mathcal{L}_{DP}(\phi) = \frac{1}{|B_g|}\sum_{(g,\xi,\tilde{d}) \in B_g} \left(\hat{d}_\phi(g, \xi) - \tilde{d}(g)\right)^2$$

- **策略优化**：PPO + Transformer-XL 骨架

## 实验

### 主实验

在 XLand-MiniGrid 的三个基准（4Rooms-Trivial, 4Rooms-Small, 6Rooms-Small）上评估：

| 评估维度 | ULEE vs Random | ULEE vs DIAYN |
|---------|---------------|---------------|
| 探索（20 ep） | 2× 以上目标发现率 | 2× 以上 |
| 快速适应（30 ep） | 3× mean return 提升 | 显著优于 |
| 微调（1B steps） | 持续领先 | DIAYN 优势短暂 |
| 元学习初始化 | 一致领先 | - |

### 消融实验

| 变体 | 描述 | 结果 |
|-----|------|------|
| adversarial + bounded | 完整 ULEE | **最佳** |
| random + bounded | 随机搜索 + 中等难度采样 | 初期适应后停滞 |
| adversarial + uniform | 对抗搜索 + 均匀采样 | 次于完整版 |
| ULEE (SED) | 基于即时而非适应后表现的难度 | 随难度增加差距增大 |

### 泛化实验

预训练于 4Rooms-Small 后在多种 MiniGrid 任务上评估：
- ULEE ($f_{\text{counts}}$) 在所有 14 个测试环境上获得非零回报
- 在 Unlock、UnlockPickUp 等任务上大幅领先

### 关键发现

1. 后适应难度度量比即时难度度量更有效，差距随基准难度增大
2. 对抗式目标搜索 + 有界采样的组合效果最佳
3. 目标映射 $f$ 作为归纳偏置对结果有显著影响
4. 预训练策略在更多环境步数后持续提升

## 亮点

- **后适应难度度量**：首次在无监督 RL 中引入基于适应后表现的难度指标
- **统一框架**：将无监督目标生成、自动课程学习和元学习整合在一个系统中
- **多尺度评估**：从零样本到长期微调全面验证
- **无需目标条件**：策略直接部署，无需目标编码

## 局限性

- 实验限于离散动作空间的网格世界，连续控制任务适用性未验证
- 对 finer-grained 的任务层次结构（如深度 > 1 的规则树）性能仍有较大提升空间
- 60% 以上的测试任务在少样本设置下回报为零，表明困难任务仍是挑战
- 计算预算较大（高达 5B 步）

## 相关工作

- **内在奖励方法**: RND (Burda et al., 2018), DIAYN (Eysenbach et al., 2018)
- **自动课程学习**: GoalGAN (Florensa et al., 2018), AMIGo (Campero et al., 2020)
- **无监督元学习**: Gupta et al. (2018), Jabri et al. (2019)
- **In-context RL**: RL² (Duan et al., 2016), Ada (Team et al., 2023)

## 评分

- **新颖性**: 8/10 — 后适应难度度量和对抗式目标生成的结合是新颖的
- **技术深度**: 8/10 — 四个组件的协同设计考虑周全
- **实验**: 7/10 — 评估全面但环境复杂度有限
- **写作**: 8/10 — 条理清晰，数学描述严谨
- **总评**: 7.5/10
# Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2601.19810](https://arxiv.org/abs/2601.19810)
- **代码**: 已开源
- **领域**: 强化学习 / 无监督预训练 / 元学习
- **关键词**: 无监督RL, 自动课程学习, 元学习, 目标生成, 探索策略

## 一句话总结
提出 ULEE 方法，通过对抗式目标生成和基于适应后难度的课程学习，在无监督环境中元学习出具备高效探索和快速适应能力的预训练策略。

## 研究背景与动机

### 核心问题
大规模预训练在视觉和语言领域取得了巨大成功，但强化学习仍然以从头训练为主。如何在没有外部奖励的情况下预训练出通用的 RL 策略（foundation policy），使其具备可迁移的探索和适应能力？

### 现有方法的局限
1. **基于内在奖励的方法**（如 DIAYN）：学到的技能多样性有限，性能随训练推进后容易停滞甚至下降
2. **目标条件策略**：在目标未知或无法编码时表现不佳
3. **固定目标空间假设**：多数课程学习方法假设目标空间在训练和评估时一致
4. **基于即时性能的难度估计**：没有考虑适应预算，不适用于需要多轮适应的评估场景

### 关键动机
人类通过自主设定和追求目标来发展能力。论文关注三个核心问题：**目标如何生成**、**如何选择**、**如何从中学习**。在下游任务分布广泛且未知的场景下，零样本解决所有任务不可能，因此需要优化多轮探索和适应效率。

## 方法详解

### 整体框架 ULEE (Unsupervised Learning of Efficient Exploration)

ULEE 由四个核心组件组成：
1. **预训练策略** $\pi$（in-context learner）
2. **目标搜索策略** $\pi_{gs}$（对抗式目标生成）
3. **难度预测网络**（估计适应后性能）
4. **目标采样策略**（从中等难度范围中选择）

### 预训练策略

采用黑箱元学习方法，策略根据完整交互历史选择动作（包含过去的观察、动作和奖励），实现上下文内适应。训练目标为最大化整个 lifetime 的期望折扣回报：

$$\mathcal{J}(\pi) = \mathbb{E}_{M \sim \mu^{\text{unsup}}, g \sim p(g|M)} \left[ \mathbb{E}_{\rho_M, P_M, \pi} \left[ \sum_{j=1}^{H} \sum_{t=0}^{T-1} \gamma^{(j-1)T+t} r_t^{(j)} \right] \right]$$

其中 $j$ 索引 lifetime 内的各 episode，$H$ 为 episode 总数。

### 基于适应后表现的目标难度度量

**核心创新**：难度定义为策略在适应预算后的性能补数，而非即时成功率：

$$d(g; \pi, M) = 1 - \mathbb{E}_{\rho_M, P_M, \pi} \left[ \frac{1}{K} \sum_{j=H-K+1}^{H} \mathbf{1}\{ \exists t: f(s_{t+1}^{(j)}) = g \} \right]$$

仅统计最后 $K$ 个 episode 的成功率，忽略前 $H-K$ 轮的探索和适应过程。

### 对抗式目标搜索

训练目标搜索策略 $\pi_{gs}$ 以最大化所发现目标的难度：

$$r_t^{gs} = r^{gs}(s_t; \pi, M) = d(f(s_t); \pi, M)$$

$\pi_{gs}$ 在每个环境中先于预训练策略运行若干 episode，收集候选目标集合。

### 目标选择与采样

从候选目标中选择中等难度目标：

$$g_M \sim \text{Unif}(S), \quad S = \{g \in GC_M : LB \leq d(g; \pi, M) \leq UB \}$$

其中 $LB = 0.1$，$UB = 0.9$，避免过易或过难的无信息目标。

### 难度预测网络

引入监督学习的难度预测器，使用 L2 回归损失：

$$\mathcal{L}_{DP}(\phi) = \frac{1}{|B_g|} \sum_{(g, \xi, \tilde{d}) \in B_g} (\hat{d}_\phi(g, \xi) - \tilde{d}(g))^2$$

提供近似的即时难度估计，避免额外的环境交互。

## 实验

### 实验设置
- **环境**: XLand-MiniGrid，基于 JAX 的程序化生成部分可观测格子环境
- **三个基准**: 4Rooms-Trivial、4Rooms-Small、6Rooms-Small
- **基线**: DIAYN、PPO（从头训练）、RND（在线探索）、RL²（元学习）

### 主实验结果

| 指标 | ULEE | DIAYN | Random |
|------|------|-------|--------|
| 20-episode 探索覆盖率 | **最高（2x+）** | 中等 | 低 |
| Few-shot 适应（30 episode） | **3× 提升** | 单跳式改进 | - |
| Fine-tuning（1B steps） | **持续领先** | 短暂优势 | - |
| Meta-RL 初始化 | **全面提升** | - | 基线 |

### 消融实验

| 变体 | 目标搜索 | 采样策略 | 相对性能 |
|------|----------|----------|----------|
| ULEE (adversarial + bounded) | 对抗式 | 中等难度 | **最优** |
| ULEE (random + bounded) | 随机 | 中等难度 | 次优 |
| ULEE (adversarial + uniform) | 对抗式 | 均匀 | 略低 |
| ULEE (SED) | 对抗式 | 即时难度 | 难度增大时退化 |

### 关键发现
1. 对抗式目标搜索 + 中等难度采样效果最佳
2. 基于适应后性能的难度度量在更难环境中优势更大
3. 目标映射 $f_{\text{counts}}$ 比 $f_{\text{grid}}$ 更适合作为归纳偏差
4. 预训练预算增加到 50 亿步仍能持续改善
5. ULEE 能够泛化到不同网格大小和房间结构的 MiniGrid 任务

## 亮点
1. **适应后难度度量**：将课程学习从即时评估扩展到考虑适应预算的元学习场景，是一个重要概念创新
2. **无条件策略预训练**：不依赖目标条件化，直接部署，适用范围更广
3. **多层次评估**：覆盖零样本探索、few-shot 适应、长期微调和元学习初始化四个维度
4. **系统性消融**：清楚展示每个组件的贡献

## 局限性
1. 目前仅在 2D 网格世界中验证，向高维连续控制任务的扩展性未知
2. 目标映射 $f$ 的选择仍需人工设计，如何自动发现合适的目标空间是开放问题
3. 对抗式训练引入了额外的 25% 计算开销
4. 在最困难的分布外任务上仍有 60% 的任务无法获得回报

## 相关工作
- **无监督 RL**: DIAYN、RND 等内在奖励方法
- **自动课程学习**: GoalGAN、AMIGo 等，但都未考虑适应后度量
- **无监督元学习**: Gupta et al. (2018) 首先探索，ULEE 在此基础上引入对抗式课程
- **Ada (DeepMind)**: 在程序生成环境中大规模元学习，但使用外部任务分布而非自生成目标

## 评分
- **创新性**: ⭐⭐⭐⭐ — 适应后难度度量和无条件策略元学习的组合很有新意
- **实验充分性**: ⭐⭐⭐⭐ — 多维度评估，但环境难度有限
- **写作质量**: ⭐⭐⭐⭐ — 方法和实验组织清晰
- **实用性**: ⭐⭐⭐ — 突破性应用还需进一步扩展到更复杂环境

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](../../ICML2025/reinforcement_learning/online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/reinforcement_learning/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)

<!-- RELATED:END -->
