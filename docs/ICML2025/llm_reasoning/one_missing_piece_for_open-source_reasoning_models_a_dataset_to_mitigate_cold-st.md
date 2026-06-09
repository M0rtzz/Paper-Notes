---
title: >-
  [论文解读] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL
description: >-
  [LLM推理] 提出 Long CoT Collection——一个由短CoT LLM（如GPT-4o）标注的100K长链推理数据集，通过从o1提取推理流程（reasoning flow）作为间接引导，使短CoT模型也能生成高质量长推理链，从而有效缓解开源推理模型在强化学习阶段的冷启动问题…
tags:
  - "LLM推理"
---

# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL

## 基本信息

- **会议**: ICML 2025
- **arXiv**: [2506.02338](https://arxiv.org/abs/2506.02338)
- **代码**: 论文中提及公开（codes, datasets, models）
- **领域**: LLM推理 (LLM Reasoning)
- **关键词**: Long CoT, 冷启动, 强化学习, 推理模型, 思考预算控制, 数据集构建

## 一句话总结

提出 Long CoT Collection——一个由短CoT LLM（如GPT-4o）标注的100K长链推理数据集，通过从o1提取推理流程（reasoning flow）作为间接引导，使短CoT模型也能生成高质量长推理链，从而有效缓解开源推理模型在强化学习阶段的冷启动问题，初始化后的模型在RLVR中获得2-3倍的性能提升。

## 背景与动机

- **LRM的崛起与闭源困境**: 以o1/R1为代表的大型推理模型(LRM)通过测试时推理扩展(test-time scaling)展现了突破性推理能力，但闭源特性限制了学术研究和实际应用。
- **冷启动问题**: DeepSeek-R1揭示了RLVR训练前需要先在Long CoT数据上做SFT来稳定RL训练，但这一冷启动数据的构建方法一直不透明。现有方法严重依赖R1等已有LRM的输出蒸馏，本质上仍是"借鸡生蛋"。
- **核心研究问题**: 能否用只会短推理的普通LLM（如GPT-4o）来构建高质量的Long CoT数据集？从而摆脱对现有LRM输出的依赖，实现独立的推理模型开发。
- **overthinking问题**: LRM倾向于对简单问题也生成过长的推理链（如QwQ-32B对"1+1+3?"生成~1500 token），需要可控的思考预算机制。

## 方法

### 整体框架

数据构建分为两大阶段：(1) 收集1K种子数据集（从o1提取推理流程）；(2) 利用种子数据引导短CoT LLM扩展生成100K长CoT数据。

### 阶段一：种子数据收集（1K Teacher Demonstrations）

- **推理流程标注 (Reasoning Flow Annotation)**: 从ChatGPT网站手动收集o1在magpie-reasoning-V1数据集1K问题上的推理流程 $S_{ref} = \{s_1, s_2, ..., s_n\}$，每个 $s_i$ 是一个推理步骤的概要描述。
- **思考预算记录**: 通过OpenAI API计算o1的思考token数 $b_{ref}$（总completion token减去返回response的token数）。
- **种子数据集**: $\mathcal{D}_{ref} = \{q, S_{ref}, b_{ref}\}$，包含问题、推理流程和思考预算三元组。

### 阶段二：100K数据扩展（Short CoT LLM标注）

#### Step 1: 推理流程检索 (Reasoning Flow Retrieval)

对新问题 $q$，从种子数据集中动态检索相关demonstration，考虑两个因素：

- **领域匹配**: 相同/相似领域的问题倾向于共享推理过程（如算术推理中o1倾向于验证计算），利用magpie数据集的主域和子域计算匹配分。
- **思考预算控制**: 检索长度相似的推理流程，相似度公式 $1 - \left|\frac{\min(x,y)}{\max(x,y)} - 1\right|$，使生成的推理链长度与LRM对齐。

#### Step 2: 推理流程生成 (Reasoning Flow Generation)

GPT-4o根据检索到的demonstration，为新问题生成推理流程 $\hat{S}$：先预测所需步骤数 $|S|$，再生成一系列推理outline。关键发现：没有demonstration引导时，LLM只会线性思考，不会使用LRM特有的验证、多路径探索等策略。

#### Step 3: 逐步长CoT生成 (Step-by-step Long CoT Generation)

以生成的推理流程 $\hat{S}$ 为指引，LLM逐步生成长推理链。对每个步骤 $\hat{s}_i$，基于已有推理 $\{r_k\}_0^{i-1}$、当前步骤 $\hat{s}_i$ 和下一步骤 $\hat{s}_{i+1}$ 生成推理内容 $r_i$。所有步骤消耗后生成最终答案，再聚合为完整序列。

#### Step 4: 正确性过滤 (Correctness Filtering)

用GPT-4o验证生成答案与参考答案的一致性，过滤错误推理链。保留率约76%。

### 思考预算控制机制

通过控制推理outline的数量来调节生成推理链的长度，构建了三个版本：100%、50%、25% budget的Long CoT Collection，为解决overthinking问题提供了灵活手段。

## 实验

### 实验设置

- **基座模型**: Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Qwen-2.5-0.5B
- **评估基准**: MATH-500, AIME24, GPQA Diamond, MMLU-Pro
- **RL方法**: GRPO, 训练数据为NuminaMATH筛选的10K整数答案样本，最大序列长度16K

### 主实验一：Best-of-N采样（RL起点质量评估）

| 模型 | MATH-500 Pass@1 | MATH-500 Pass@32 | AIME24 Pass@1 | AIME24 Pass@32 |
|------|:---:|:---:|:---:|:---:|
| Llama-3.1-8B-Instruct | ~48 | ~78 | ~3 | ~20 |
| Llama-3.1-8B-LC (Ours) | ~55 | ~88 | ~7 | ~33 |
| Qwen-2.5-7B-Instruct | ~72 | ~88 | ~10 | ~33 |
| Qwen-2.5-7B-LC (Ours) | ~72 | ~93 | ~10 | ~43 |

- 在Llama-8B上，Long CoT Collection训练后BoN结果在所有N下均显著提升
- Qwen-7B LC在大N时提升明显（Pass@16/32），说明SFT后模型能探索更多样的推理路径

### 主实验二：通用推理能力（Table 1）

| 模型 | Size | GPQA Diamond | MMLU-Pro |
|------|:---:|:---:|:---:|
| o1-mini | N/A | 60.0 | 80.3 |
| o1 | N/A | 77.3 | - |
| R1 | 671B | 71.5 | 84.0 |
| QwQ-32B | 32B | 65.2 | 71.0 |
| Sky-T1 | 32B | 56.8 | 69.2 |
| Bespoke-7B | 7B | 38.9 | - |
| OpenThinker-7B | 7B | 42.4 | - |
| Llama-3.1-8B-Instruct | 8B | 22.7 | 43.7 |
| **Llama-3.1-8B-LC (Ours)** | 8B | **36.4** | **44.5** |
| Qwen-2.5-7B-Instruct | 7B | 37.6 | 49.9 |
| **Qwen-2.5-7B-LC (Ours)** | 7B | **39.9** | **51.4** |

- Llama-8B-LC在GPQA上提升 **+13.7**（22.7→36.4），提升显著
- Qwen-7B-LC GPQA略超Bespoke-7B（39.9 vs 38.9），后者是R1蒸馏模型
- MMLU-Pro上也有温和提升，说明推理策略可迁移到通用领域

### 消融实验：思考预算对性能的影响（Table 2）

| 训练数据预算比例 | MATH-500 |
|:---:|:---:|
| 100% | 66.6 |
| 50% | 60.7 |
| 25% | 57.6 |

- 预算越充足，下游数学推理能力越强
- 25%预算过度压缩导致推理混乱（信息被强行压缩到过少的outline中）

### 关键发现

1. **RL增益2-3倍**: 在Qwen-0.5B上，用Long CoT Collection初始化后做RLVR，在MATH-500和GPQA上的性能增益是直接从base做RL的2-3倍
2. **数据质量接近R1**: 用o3-mini作为评估器的头对头对比中，Long CoT Collection在推理流程质量上优于R1输出，在推理策略和正确性上略弱但仍有竞争力
3. **思考token分配合理**: 与R1相比，Long CoT Collection的token分布更紧凑，与o1-mini更接近，有效避免了overthinking
4. **推理触发词丰富**: 生成的推理链包含"Wait"、"To verify"等推理触发词，有助于探索多样推理路径

## 亮点

- 🔑 首次证明短CoT LLM（GPT-4o）在推理流程引导下可以生成高质量Long CoT数据，无需依赖LRM输出蒸馏
- 🔑 提出推理流程（reasoning flow）作为"间接引导"的新范式：先提取宏观推理框架，再由短CoT模型填充细节
- 🔑 思考预算控制机制为解决overthinking提供了实用方案
- 🔑 SFT+RL两阶段实验设计清晰验证了数据集对冷启动问题的价值（2-3x RL gain）

## 局限性

- 仅在7-8B模型上做SFT、0.5B模型上做RL（GPU资源限制，16×A100 40GB），未验证更大模型的效果
- 仅使用o1作为参考LRM，未探索其他LRM（如Gemini Thinking）
- 未在专家领域（代码、科学）进行验证
- 种子数据需要手动从ChatGPT网站收集o1的推理流程，自动化程度有限
- 76%的正确率意味着24%的数据被丢弃，数据利用效率可进一步提升

## 相关工作

- **DeepSeek-R1** (2025): 开源推理模型，揭示SFT冷启动+RLVR的训练范式
- **Sky-T1, Bespoke-Stratos**: 通过蒸馏LRM输出构建训练数据的代表作
- **GRPO** (Shao et al., 2024): 本文RL阶段使用的策略优化算法
- **Yeo et al. (2025)**: 详细分析SFT后RLVR阶段的作用

## 评分

⭐⭐⭐⭐ (4/5)

**理由**: 提出了一个重要且实用的研究问题（摆脱LRM依赖构建冷启动数据），方法简洁有效（推理流程引导+逐步生成），实验设计合理且验证了核心假设（2-3x RL增益）。不足之处在于受限于计算资源，RL实验仅在0.5B模型上进行，且方法仍需o1的推理流程作为种子，并未完全脱离LRM依赖。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ICML 2025\] PENCIL: Long Thoughts with Short Memory](pencil_long_thoughts_with_short_memory.md)
- [\[ICML 2025\] MARGE: Improving Math Reasoning for LLMs with Guided Exploration](marge_improving_math_reasoning_for_llms_with_guided_exploration.md)
- [\[ICML 2025\] Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs](putnam-axiom_a_functional_and_static_benchmark_for_measuring_higher_level_mathem.md)

</div>

<!-- RELATED:END -->
