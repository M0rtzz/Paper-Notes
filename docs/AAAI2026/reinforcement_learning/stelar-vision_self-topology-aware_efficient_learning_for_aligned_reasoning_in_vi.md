---
title: >-
  [论文解读] STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision
description: >-
  [AAAI 2026][强化学习][拓扑推理] 提出 STELAR-Vision，一个拓扑感知的视觉语言推理训练框架，通过 TopoAug 数据生成管线引入 Chain/Tree/Graph 多种推理拓扑结构，配合 SFT+RL 后训练，在分布内外数据集上分别提升 9.7% 和最高 28.4% 的准确率，并通过 Frugal Learning 减少 18.1% 的输出长度。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "拓扑推理"
  - "视觉语言模型"
  - "思维链/树/图"
  - "高效推理"
---

# STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision

**会议**: AAAI 2026  
**arXiv**: [2508.08688](https://arxiv.org/abs/2508.08688)  
**代码**: [stellar-neuron.github.io/stelar-vision](https://stellar-neuron.github.io/stelar-vision/)  
**领域**: 强化学习  
**关键词**: 拓扑推理, 视觉语言模型, 思维链/树/图, 强化学习, 高效推理

## 一句话总结

提出 STELAR-Vision，一个拓扑感知的视觉语言推理训练框架，通过 TopoAug 数据生成管线引入 Chain/Tree/Graph 多种推理拓扑结构，配合 SFT+RL 后训练，在分布内外数据集上分别提升 9.7% 和最高 28.4% 的准确率，并通过 Frugal Learning 减少 18.1% 的输出长度。

## 研究背景与动机

### 问题背景

当前的视觉语言模型（VLM）在推理任务上主要依赖链式思维（Chain-of-Thought, CoT）范式。然而，作者的分析揭示了一个关键问题：**不同问题适合不同的推理拓扑结构**。CoT 只是众多推理结构中的一种，树状（Tree）和图状（Graph）推理在某些问题上表现更优。

### 核心发现

通过在 MATH-V 数据集上对 Qwen2-VL-7B 和 GPT-4o-Mini 的系统评估，得出：

- **Chain 结构赢率 49%，Tree 28%，Graph 23%**——Tree 和 Graph 合计超过一半
- **不同学科适合不同拓扑**：图论、统计学等科目中 Tree/Graph 表现明显更好
- **Chain 推理最冗长**：生成 token 长度分布右偏严重，平均最长；Tree/Graph 分布更集中、更简洁

### 动机推导

1. 现有 VLM 的训练数据几乎全是 CoT 风格，导致模型默认使用链式推理，即使这不是最优选择
2. CoT 推理容易"过度思考"（overthinking），产生不必要的冗长输出

**拓扑多样性与输出长度相关**——引入 Tree/Graph 拓扑可以自然地减少输出冗余
4. 假设 1：用拓扑多样的数据训练（不增加数据量），模型能学会自适应选择最优拓扑
5. 假设 2：在此基础上可以设计机制鼓励简洁输出，在少量精度损失下大幅提高效率

## 方法详解

### 整体框架

STELAR-Vision 包含三个核心组件：
1. **TopoAug**：合成数据生成管线，为每个问题生成多种拓扑结构的推理回答
2. **两阶段后训练**：SFT → RL（SimPO）
3. **Frugal Learning**：鼓励简洁输出的训练变体

### 关键设计

#### 1. **TopoAug：拓扑增强数据生成管线**

为每个问题，使用两个模型（Qwen2-VL-7B-Instruct + GPT-4o-Mini）反复生成三种拓扑的推理回答：Chain、Tree、Graph。每种拓扑支持不同的最大深度、子节点数、邻居数等参数。

为每个问题计算两类标签：

- **拓扑标签 $\mathcal{F}_{q,t}$**：连续值 $[0,1]$，表示拓扑 $t$ 在问题 $q$ 上的正确率
  $$\mathcal{F}_{q,t} = \frac{N_{\text{correct}}(q, t)}{N_{\text{total}}(q, t)}$$

- **结果标签 $\mathcal{H}_r$**：二值 $\{0, 1\}$，每个回答是否正确

基于拓扑标签分布将问题分为三个难度级别：
- **Easy**：所有三种拓扑得分 > 85% 分位数
- **Hard**：所有三种拓扑得分 < 15% 分位数
- **Medium**：其余

使用两个不同规模的模型生成确保了正负样本的分布平衡和推理拓扑的多样性。

#### 2. **两阶段后训练**

**阶段 1：SFT（监督微调）**

数据准备采用三步过滤：
1. 从 Easy/Medium/Hard 问题中平衡采样
2. 仅保留结果标签 $\mathcal{H}_r = 1$ 的正确回答
3. 用 7B ORM（Outcome Reward Model，同时用拓扑标签和结果标签训练）进行拒绝采样，选择更高质量的样本

混合 TopoAug 数据与三个通用 VQA 数据集（OKVQA、A-OKVQA、LLaVA-150k），后者不做拓扑增强。使用 LoRA 微调，标准 NTP 损失：

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)$$

**阶段 2：RL（强化学习）**

从 SFT checkpoint 初始化，使用 SimPO 进行偏好优化：

$$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

正确回答作为 preferred response，关键细节：**训练时移除拓扑提示**，迫使模型在测试时自主推断最优推理结构。

#### 3. **Frugal Learning：高效推理的训练变体**

两种变体：

- **STELAR-Vision-Short†**：SFT 阶段过滤"短且正确"的回答（token 长度 < 25% 分位数），RL 阶段将"短且正确"作为 winner
- **STELAR-Vision-Short‡**：在 Short† 基础上，将"正确但冗长"的回答也作为 loser，同时惩罚错误和冗长

### 损失函数 / 训练策略

- 基座模型：Qwen2VL-7B-Instruct
- SFT 数据量：约 50K-60K 样本
- SFT 耗时：约 5-7 小时（8×A100/H100），RL 耗时：约 8-10 小时
- 所有 5 个 OOD 数据集使用训练时的同一权重，无针对性微调

## 实验关键数据

### 主实验

| 模型 | VLM_S2H | MATH-V | 总体 (ID) | Geometry3K | We-Math | PolyMath | SciBench | LogicVista |
|------|---------|--------|-----------|------------|---------|----------|----------|------------|
| GPT-4o | 32.0 | 28.0 | 30.7 | 57.0 | 66.4 | 25.0 | 31.1 | 34.6 |
| Qwen2VL-7B-Instruct | 21.0 | 13.0 | 18.3 | 35.2 | 46.6 | 16.0 | 10.7 | 17.0 |
| Qwen2VL-72B-Instruct | 21.0 | 20.0 | 20.7 | 50.2 | 60.6 | 13.0 | 25.4 | 28.8 |
| Chain-Only | 25.0 | 21.0 | 23.7 | 31.4 | 42.2 | 17.2 | 10.7 | 25.4 |
| **STELAR-Vision** | **31.0** | **22.0** | **28.0** | 36.8 | 51.0 | 23.8 | 12.4 | 29.0 |

STELAR-Vision 在分布内数据上超过基座模型 **+9.7%**，超过 10× 更大的 Qwen2VL-72B-Instruct **+7.3%**。

### 消融实验

| 模型 | SFT | RL | VLM_S2H | MATH-V | 总体 |
|------|-----|-----|---------|--------|------|
| Qwen2VL-7B-Instruct | × | × | 21.0 | 13.0 | 18.3 |
| Chain-Only-SFT | ✓ | × | 18.5 | 19.0 | 18.7 |
| Chain-Only | ✓ | ✓ | 25.0 | 21.0 | 23.7 |
| STELAR-Vision-SFT | ✓ | × | 28.0 | 24.0 | 26.7 |
| **STELAR-Vision** | ✓ | ✓ | **31.0** | 22.0 | **28.0** |

**拓扑增强 vs Chain-Only 的总体提升**：23.7% → 28.0%（+4.3%）。

**Frugal Learning 效率对比**：

| 模型 | 准确率 (%) | ID 生成 token 数 | OOD 生成 token 数 |
|------|-----------|-----------------|-----------------|
| Qwen2VL-7B-Instruct | 26.2 | 613.5 | 543.3 |
| Chain-Only | 28.7 | 878.4 | 742.6 |
| STELAR-Vision | 31.6 | 556.7 | 523.4 |
| **STELAR-Vision-Short†** | **28.7** | **455.7** | **498.6** |

STELAR-Vision-Short† 减少 18.1% 输出长度，仍超基座模型 +2.5%。

### 关键发现

1. **拓扑多样性扩展了 RL 的探索空间**：RL 在 TopoAug 数据上持续提升，而 Chain-Only 数据上的 RL 收益递减
2. **训练后模型自主选择拓扑**（无需显式提示）：在 Geometry3K 上 96.4% 选择 Tree，LogicVista 上 61.7% 选择 Chain——说明模型真正学会了按问题特性选择最优结构
3. **SFT 可能过拟合**导致 -SFT 变体在某些 OOD 数据集上不及完整模型
4. **Chain-Only-Short† 的 Frugal Learning 失效**：RL 微调后 Chain-Only 模型倾向生成冗长回答，即使加入 Frugal Learning 也无法有效缩短

## 亮点与洞察

- **系统验证了推理拓扑多样性的价值**——不同问题确实需要不同推理结构，这是对"CoT 万能"假设的有力反驳
- **以小搏大**：7B 模型超越 72B 模型 7.3%，证明训练范式比模型规模更重要
- **Frugal Learning 只有在拓扑多样性基础上才有效**——仅有 Chain 训练时无法兼顾简洁和准确
- 模型在 OOD 数据集上的拓扑选择分布与问题结构高度一致（简单逻辑→Chain，复杂几何→Tree/Graph），说明泛化是真实的

## 局限与展望

- 当前拓扑类型是预定义的 {Chain, Tree, Graph}，更灵活的端到端拓扑发现有待探索
- 与 Qwen2.5-VL 不兼容（该模型生成多样拓扑不稳定），限制了基座模型选择
- 问题结构与最优拓扑之间的动态关系尚未深入研究
- Frugal Learning 的 Short‡ 变体（同时惩罚冗长）效果反而不好，优化信号冲突
- OOD 泛化强但在某些特定数据集（如 SciBench）上提升有限

## 相关工作与启发

- CoT、ToT、GoT 此前主要通过采样/规则方式使用，本文首次将多拓扑训练纳入 VLM 后训练框架
- SimPO 的使用是一个巧妙选择：无需单独的奖励模型，且与解码行为对齐
- 与 CuRPO 互补：CuRPO 发现 CoT 在视觉定位中有害并用课程学习缓解，本文发现 CoT 不够用并引入更多拓扑替代
- Frugal Learning 方向与 L1（RL 控制推理长度）和 SelfBudgeter 相关但更简单

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 将多拓扑推理结构引入 VLM 后训练的思路新颖且有说服力
- **实验充分度**: ⭐⭐⭐⭐ — 7 个数据集（含 5 个 OOD）、多基线对比、消融完整，但仅在 7B 模型上实验
- **写作质量**: ⭐⭐⭐⭐ — 分析系统、图表丰富，但篇幅较长（扩展版）
- **价值**: ⭐⭐⭐⭐⭐ — 以小搏大的思路极具实用价值，TopoAug 管线可直接复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](../../ICLR2026/reinforcement_learning/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](../../ICML2026/reinforcement_learning/d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[NeurIPS 2025\] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](../../NeurIPS2025/reinforcement_learning/open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](../../ICLR2026/reinforcement_learning/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
