---
title: >-
  [论文解读] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation
description: >-
  [ICLR 2026][RLVR] 提出 RLVRR 框架，将 RLVR（强化学习+可验证奖励）从数学/代码推理扩展到开放式文本生成：从高质量参考答案中提取关键词序列（内容奖励）和可执行 Python 检查函数（风格奖励），构成"奖励链"替代单点验证信号，在 10+ 个 benchmark 上以 10K 数据超越 100K SFT 和高级奖励模型。
tags:
  - ICLR 2026
  - RLVR
  - 开放式生成
  - 奖励链
  - 可验证奖励
  - GRPO
---

# From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation

**会议**: ICLR 2026  
**arXiv**: [2601.18533](https://arxiv.org/abs/2601.18533)  
**代码**: https://github.com/YJiangcm/RLVRR  
**领域**: 强化学习 / LLM对齐  
**关键词**: RLVR, 开放式生成, 奖励链, 可验证奖励, GRPO

## 一句话总结

提出 RLVRR 框架，将 RLVR（强化学习+可验证奖励）从数学/代码推理扩展到开放式文本生成：从高质量参考答案中提取关键词序列（内容奖励）和可执行 Python 检查函数（风格奖励），构成"奖励链"替代单点验证信号，在 10+ 个 benchmark 上以 10K 数据超越 100K SFT 和高级奖励模型。

## 研究背景与动机

**领域现状**：RLVR（如 DeepSeek-R1、GRPO）在数学和代码生成上取得巨大成功——通过检查最终答案的正确性（一个"可验证点"）提供奖励信号。RLHF 则用偏好奖励模型指导开放式生成任务的对齐。

**现有痛点**：(a) RLVR 无法直接用于开放式生成——开放式回答没有唯一正确答案，单点验证不适用；(b) RLHF 的奖励模型容易 reward hacking（过拟合表面特征），且需要大规模偏好标注数据，训练成本高且不稳定。

**核心矛盾**：开放式生成需要同时评估多维度质量（内容完整性、格式、风格），但缺乏像数学答案那样的确定性验证信号。

**本文目标**：设计一种从参考答案中自动提取多维度可验证信号的方法，使 RLVR 范式能扩展到开放式生成。

**切入角度**：把参考答案视为"规则来源"——就像数学推理从 ground truth 推导规则一样，从高质量参考中提取有序的语言学信号（奖励链），将单点监督升级为链式监督。

**核心 idea**：把参考答案分解为关键词（内容）+ Python 验证函数（风格），用这两个可验证维度的规则化奖励替代奖励模型。

## 方法详解

### 整体框架

RLVRR 两阶段：
- **数据构建**：给定问题 $x$ 和参考答案 $z$，用 GPT-4o-mini 提取：(a) 内容维度的层次化关键词；(b) 风格维度的可执行 Python 检查代码
- **RL 训练**：用 GRPO 优化策略 $\pi_\theta$，奖励 = 内容奖励 $r_c$ 和风格奖励 $r_s$ 的平均
- 总奖励：$r_\phi(x,y) = \mathcal{F}(r_c(x,y,z), r_s(x,y,z))$

### 关键设计

1. **两级层次关键词提取（内容奖励）**:

    - 功能：从参考答案中提取核心内容的可验证关键词
    - 核心思路：LLM 先提取 M 个 key points（如"解释风险"、"拒绝有害请求"），每个 key point 下再提取具体关键词（<3 词）。内容奖励用 LCS（最长公共子序列）计算 rollout 与参考关键词的对齐度：$r_c = \frac{1}{M}\sum_{m=1}^{M}\frac{\text{len}(\text{LCS}(K_z^m, K_y^m))}{\max(\text{len}(K_z^m), \text{len}(K_y^m))}$
    - 设计动机：两级提取比直接提关键词覆盖更广更系统；LCS 保留了关键词顺序和重复，比 bag-of-words 更精细；关键词仅占参考的约 15%，保持灵活的表达空间

2. **Python 验证函数（风格奖励）**:

    - 功能：评估 rollout 是否满足参考答案的风格属性
    - 核心思路：LLM 为每个参考生成 N 个 Python CodeEval 函数（检查长度、markdown 格式等），每个带权重 $w_n$。风格奖励 $r_s = \sum_{n=1}^{N} w_n \cdot \text{CodeEval}_n(y)$
    - 设计动机：Python 代码检查是确定性的、可验证的、零成本的——比奖励模型更可靠

3. **多参考容错**:

    - 支持 I=3 个参考答案，对每个 key point 取最高对齐分
    - 消融实验证明多参考比单参考一致性更好

### 损失函数 / 训练策略

- 优化算法：GRPO（Group Relative Policy Optimization）
- KL 散度约束：$\beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]$
- 训练数据：仅 10K 条开放式指令-回答对（从 100K 中筛选），数据构建用 GPT-4o-mini
- 质量过滤：丢弃内容+风格奖励 < 0.7 的样本

## 实验关键数据

### 主实验

Qwen2.5-3B-Instruct 上 5 个开放式 benchmark 对比：

| 方法 | 数据量 | AlpacaEval2 (LC%) | ArenaHard (WR%) | MTBench | IFEval | FollowBench |
|------|--------|-------------------|-----------------|---------|--------|-------------|
| SFT | 100K | 25.1 | 32.9 | 7.5 | 35.9 | 51.3 |
| RM (Skywork-8B) | 10K | 28.8 | 32.3 | 7.6 | 34.5 | 51.4 |
| GRM (GPT-4o-mini) | 10K | 27.1 | 28.7 | 7.4 | 35.2 | 50.9 |
| DPO | 10K | 24.8 | 28.8 | 7.5 | 35.5 | 49.5 |
| **RLVRR** | **10K** | **31.5** | **36.2** | **7.7** | **36.8** | **53.1** |

RLVRR 用 10K 数据在所有指标上超越 100K SFT 和 8B 奖励模型。

### 消融实验

| 配置 | AlpacaEval2 | ArenaHard | 说明 |
|------|-------------|-----------|------|
| Full RLVRR | 31.5 | 36.2 | 完整框架 |
| w/o 层次提取（直接提关键词） | 30.6 | 35.0 | 层次化贡献 +0.9 |
| w/o 风格奖励 | 29.8 | 33.1 | 风格信号有效 |
| w/o 多参考（I=1） | 30.2 | 34.5 | 多参考提升鲁棒性 |
| BLEU 作奖励 | 24.3 | 27.5 | n-gram 远不如关键词 |
| Random 奖励 | 22.5 | 25.1 | 基线 |

### 关键发现

- RLVRR 的计算开销极低：相比随机奖励仅增加 0.71%，而加载奖励模型需要额外 GPU 内存和计算
- RLVRR 可无缝与 RLVR 结合——统一训练推理任务和开放式生成任务
- 深入分析表明 RLVRR 在保持输出多样性的同时提升了质量（不像 SFT 容易产生单一模式输出）
- BLEU 作为奖励信号非常差——n-gram 精度无法捕捉与人类偏好对齐的关键内容

## 亮点与洞察

- **"奖励链"概念巧妙**：从"验证一个点"到"验证一条链"，是 RLVR 范式的自然延伸。关键词链保留了内容的确定性可验证属性，同时允许表达自由度——兼顾了 SFT 的精确指导和 RL 的探索性
- **去掉奖励模型**：用规则化检查（正则匹配、Python 代码）替代数十亿参数的奖励模型，大幅降低 RL 训练成本和不稳定性。这个思路可推广到任何有参考答案的场景
- **少量数据大效果**：10K 数据超过 100K SFT，说明 RL 的探索机制在对齐任务中的数据效率远高于监督学习

## 局限与展望

- **依赖参考答案质量**：关键词和风格检查都从参考提取，如果参考质量差或存在偏见，RLVRR 也会学到错误模式
- **关键词提取依赖 GPT-4o-mini**：数据构建阶段需要调用强大 LLM，开源替代方案的效果未验证
- **风格检查较浅**：目前只检查长度、格式等表面属性，语气、逻辑连贯性等深层风格无法用简单代码验证
- **仅在 ≤7B 模型上验证**：更大模型（如 70B+）上 RLVRR 是否仍有优势未知

## 相关工作与启发

- **vs RLHF/DPO**: RLHF 需要偏好数据+奖励模型，成本高且易 hack；DPO 离线优化但缺乏在线探索。RLVRR 保留 RL 在线探索优势但去掉了奖励模型
- **vs BLEU-as-reward** (Chang et al. 2025): BLEU 是 n-gram 精度，无法区分关键内容和填充文本。RLVRR 用层次化关键词精准捕捉核心概念
- **vs RLPR** (Yu et al. 2025): RLPR 用模型自身的 token 概率做奖励，但只在短答案上有效。RLVRR 适用于长文本开放式生成

## 评分

- 新颖性: ⭐⭐⭐⭐ "奖励链"概念新颖，但内容奖励本质是关键词匹配，技术上不算突破性创新
- 实验充分度: ⭐⭐⭐⭐⭐ 10+ benchmark、多模型系列、详细消融、多样性分析、效率分析
- 写作质量: ⭐⭐⭐⭐ 叙事清晰，"dot→chain"的类比直观
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高——为没有标准答案的对齐任务提供了低成本、可扩展的 RL 训练方案

<!-- RELATED:START -->

## 相关论文

- [LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [Generalizing Verifiable Instruction Following](../../NeurIPS2025/reinforcement_learning/generalizing_verifiable_instruction_following.md)
- [References Improve LLM Alignment in Non-Verifiable Domains](references_improve_llm_alignment_in_non-verifiable_domains.md)

<!-- RELATED:END -->
