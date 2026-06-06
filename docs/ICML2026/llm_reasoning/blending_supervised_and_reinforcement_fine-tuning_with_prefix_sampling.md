---
title: >-
  [论文解读] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling
description: >-
  [ICML 2026][LLM推理][后训练] 提出 Prefix-RFT，通过从专家示范中采样前缀拼接模型续写来构建混合轨迹，在保持 RFT 目标导向优化的同时注入 SFT 的知识引导，在数学推理任务上显著超越独立 SFT、RFT 及已有混合方法。
tags:
  - "ICML 2026"
  - "LLM推理"
  - "后训练"
  - "监督微调"
  - "强化微调"
  - "前缀采样"
  - "数学推理"
---

# Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling

**会议**: ICML 2026  
**arXiv**: [2507.01679](https://arxiv.org/abs/2507.01679)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 后训练, 监督微调, 强化微调, 前缀采样, 数学推理  

## 一句话总结
提出 Prefix-RFT，通过从专家示范中采样前缀拼接模型续写来构建混合轨迹，在保持 RFT 目标导向优化的同时注入 SFT 的知识引导，在数学推理任务上显著超越独立 SFT、RFT 及已有混合方法。

## 研究背景与动机

**领域现状**：LLM 后训练主要分两种范式——监督微调（SFT）通过模仿专家示范来注入知识，强化微调（RFT）通过试错探索和奖励信号来提升任务表现。实践中通常先做 SFT 再做 RFT 的两阶段流程。

**现有痛点**：SFT 本质是行为克隆，虽然能学到正确的解题模式，但泛化和鲁棒性有问题；RFT 虽能直接优化任务性能，但学习信号稀疏、容易产生语言混用等意外行为，且性能高度依赖初始策略的能力上限——近期研究质疑 RL 是否真能突破模型的内在能力天花板。

**核心矛盾**：SFT 提供密集监督但过度约束解空间，RFT 鼓励探索但受限于当前策略能力。简单的"RL + SFT Loss"联合训练会因示范数据的梯度主导 RFT 梯度而适得其反；两阶段串联（SFT→RFT）则无法在训练过程中动态平衡两种学习信号。

**本文目标**：设计一个统一框架，在 RFT 训练过程中有机融合 SFT 的过程监督和 RFT 的目标导向优化，实现知识注入与能力提升的动态平衡。

**切入角度**：作者首先建立 SFT 与 RFT 的统一视角——两者的梯度更新都是对 token 对数概率施加加权梯度，区别仅在权重设定。基于此统一框架，只需设计合适的权重分配即可自然融合两种范式。

**核心 idea**：从专家示范中截取前缀，让模型从前缀位置续写生成，拼接成混合轨迹后与标准 rollout 一起做 PPO 更新，用轨迹级别的 advantage 自动调控示范数据的学习强度。

## 方法详解

### 整体框架
给定一个 prompt $x$ 和专家示范 $y^*$，Prefix-RFT 先用当前策略 $\pi_{\theta_{\text{old}}}$ 生成 $N-1$ 条标准 rollout。对第 $N$ 条，从 $y^*$ 中截取前缀 $y^*_{<L}$，然后由模型续写得到 $y_{\geq L}$，拼接形成混合轨迹 $y^{(N)}$。所有 $N$ 条轨迹统一估计 advantage 并做 PPO 更新，其中前缀 token 和续写 token 使用相同的 PPO 权重 $\mathcal{W}_{i,t}^{\text{PPO}} = \mathbb{I}_{\text{clip}}(r_t, \hat{A}_t) \hat{A}_t r_t$。这样不增加额外 rollout 开销，仅用一条混合轨迹替换一条标准 rollout。

### 关键设计

1. **前缀采样与混合轨迹构建**:

    - 功能：将离线示范数据有机嵌入在线 RFT 训练流程
    - 核心思路：截取专家示范的前 $L$ 个 token 作为前缀，模型从第 $L$ 个位置开始续写。前缀部分虽来自离线策略，但通过整条混合轨迹的 advantage 来决定其梯度权重：如果带前缀的轨迹获得更高奖励，前缀自然被正向强化；反之则被抑制。相比 SFT 强制模仿整个序列，前缀采样赋予模型"受约束的自主权"——沿着专家指引的方向出发，但仍可探索更优的续写路径
    - 设计动机：解决纯 RFT 探索效率低、无法突破策略能力上限的问题，同时避免 SFT 过度约束解空间

2. **基于熵的裁剪策略（Entropy-based Clipping）**:

    - 功能：防止离线示范数据的梯度主导整个优化过程
    - 核心思路：只保留前缀中熵最高的 top-$k$%（默认 20%）token 参与梯度更新，其余 token 的 advantage 设为零。低熵 token 要么已被当前策略匹配（学习信号小），要么是高置信度的偏离（会导致剧烈覆写更新）；高熵 token 对应模型最不确定的位置，学习价值最大
    - 设计动机：当离线策略 $\pi_{\text{off}}$ 与当前策略差距大时，前缀 token 的概率极低，梯度量级远大于 RFT 梯度，不加约束会退化为简单 SFT

3. **余弦衰减前缀长度调度器（Cosine Decay Scheduler）**:

    - 功能：控制前缀长度的动态变化，实现从 SFT 到 RFT 的课程式过渡
    - 核心思路：前缀长度由 $L = \lfloor l \cdot |y^*| \rfloor$ 决定，其中 $l \sim U(\text{low}, \text{high})$。训练初期 low 接近 high（前缀长，接近 SFT），随训练进行 low 按余弦衰减到接近零（前缀短，接近 RFT）。这既缓解了均匀采样导致的位置偏差（尾部总结/结论段被采到的概率低），又自然实现课程学习
    - 设计动机：均匀采样时模型天然更多接触示范开头而非结尾的技能（如总结、推理收束），且训练后期模型已足够强，应减少对示范的依赖

## 实验关键数据

### 主实验（Qwen2.5-Math-7B）

| 方法 | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | 数学平均 |
|------|--------|--------|-----|----------|---------|----------|----------|
| Base | 11.5 | 4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 |
| SFT | 22.2 | 22.3 | 52.8 | 82.6 | 40.8 | 43.7 | 44.1 |
| RFT | 25.1 | 15.3 | 62.0 | 84.4 | 39.3 | 46.8 | 45.5 |
| SFT+RFT | 25.8 | 23.1 | 62.7 | 87.2 | 39.7 | 50.4 | 48.2 |
| RL w/ SFT Loss | 19.5 | 16.4 | 49.7 | 80.4 | 34.9 | 39.4 | 40.1 |
| LUFFY | 29.4 | 23.1 | 65.6 | 87.6 | 37.5 | 57.2 | 50.1 |
| ReLIFT | 28.2 | 20.1 | 64.9 | 87.4 | 33.8 | 52.5 | 47.8 |
| **Prefix-RFT** | **31.8** | **26.4** | **68.2** | **88.4** | **40.3** | **55.7** | **51.8** |

### 消融实验（Qwen2.5-Math-1.5B）

| 配置 | AIME24 | AIME25 | AMC | MATH-500 | 平均 | 说明 |
|------|--------|--------|-----|----------|------|------|
| SFT | 11.7 | 13.2 | 37.8 | 70.6 | 31.9 | 纯 SFT 基线 |
| RFT | 11.8 | 7.7 | 40.2 | 61.8 | 30.0 | 纯 RFT 基线 |
| Prefix-RFT (full) | 17.7 | 17.1 | 50.5 | 81.4 | 41.1 | 完整方法 |
| 数据量 10% (4.5k) | 17.8 | 15.9 | 49.7 | 79.0 | 40.8 | 仅掉 0.3 |
| 数据量 1% (0.45k) | 15.2 | 11.8 | 46.3 | 76.0 | 37.6 | 减 99% 数据仍超基线 |
| 生成器 1.5B | 15.9 | 12.6 | 47.7 | 79.0 | 39.8 | 弱生成器也有效 |
| 生成器 32B | 18.1 | 15.3 | 50.9 | 81.2 | 40.6 | 质量影响小 |

### 关键发现
- Prefix-RFT 在 6 个数学推理和 3 个通用推理基准上全面超越所有基线，数学平均 51.8 vs LUFFY 50.1、RFT 45.5
- Pass@2048 实验表明 Prefix-RFT 是唯一能真正提升模型推理能力上限的方法，在 AIME24 和 AIME25 上比 base model 提升 6.67 个百分点
- 熵裁剪 top-20% 显著优于 top-50%/80%/random-20%/bottom-20%，验证了高熵 token 筛选的必要性
- 余弦衰减调度器优于均匀采样，训练动态表现为前缀 advantage 逐步缩小——模型自动从依赖示范过渡到自主探索
- 方法对示范数据量和质量都很鲁棒：数据减少 99% 仅掉 3.5 分，用 1.5B 小模型生成的示范也接近 DeepSeek-R1 的效果

## 亮点与洞察
- **统一视角极简但深刻**：SFT 和 RFT 的梯度结构本质一致（加权 log-prob 梯度），差异仅在权重设定，这为混合方法提供了理论基础。从这个视角出发，Prefix-RFT 的设计变得自然而优雅——无需引入额外损失函数或复杂的多阶段调度
- **Advantage 驱动的自适应学习**：前缀的学习强度由轨迹级 advantage 自动调控——对难题，前缀的 advantage 高，模型从示范学到更多；对简单题，advantage 低，模型主要靠自身探索。这种 instance-level 的动态平衡无需人工设定权重
- **高熵 token 筛选**：用信息论指标过滤离线数据的梯度贡献，是一种通用的 off-policy 训练稳定性技巧，可迁移到其他混合在线/离线学习场景

## 局限与展望
- 实验主要集中在可验证推理任务（数学、代码），对开放式生成和噪声奖励场景未验证
- 当每个 prompt 有多条候选示范时，简单随机选取可能非最优，系统化的示范选择策略留待未来
- 熵裁剪比例（20%）和调度器参数的最优值可能因任务/模型而异，统一超参搜索策略尚未探讨
- 代码生成实验仅为初步验证（Qwen3-1.7B），更大规模和更多领域的泛化性需要进一步确认

## 相关工作与启发
- **LUFFY**（Yan et al., 2025）：将完整离线示范混入 rollout 做 RFT，不做前缀截取
- **UFT**（Liu et al., 2025b）：同样采样前缀，但对前缀用 SFT loss、续写用 RFT loss，且使用静态小权重
- **ReLIFT**（Ma et al., 2025）：多阶段交替 SFT 和 RFT，SFT 聚焦 RFT 解不出的难题
- 本文的优势在于：统一权重（PPO advantage）替代多损失函数设计，熵裁剪替代静态权重，余弦衰减替代人工分阶段——整体更简洁、更易集成到现有 RFT 流程

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](../../ACL2025/llm_reasoning/enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](../../ACL2025/llm_reasoning/tract_regression_cot.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](../../ACL2025/llm_reasoning/dcot_diverse_cot_refinement.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)

</div>

<!-- RELATED:END -->
