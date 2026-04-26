---
title: >-
  [论文解读] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems
description: >-
  [AAAI 2026][运筹学] OR-R1提出了一个数据高效的两阶段训练框架（SFT + TGRPO），仅使用ORLM所需1/10的合成数据即达到67.7%的平均求解准确率，超越现有SOTA方法，并通过测试时强化学习将单次生成（Pass@1）与多次生成（Pass@8）的性能差距从13%缩小到7%。
tags:
  - AAAI 2026
  - 运筹学
  - LLM微调
  - 强化学习
  - 测试时适应
  - 数据高效
---

# OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems

**会议**: AAAI 2026  
**arXiv**: [2511.09092](https://arxiv.org/abs/2511.09092)  
**代码**: [GitHub](https://github.com/SCUTE-ZZ/OR-R1)  
**领域**: 大语言模型 / 运筹优化自动化  
**关键词**: 运筹学, LLM微调, 强化学习, 测试时适应, 数据高效

## 一句话总结

OR-R1提出了一个数据高效的两阶段训练框架（SFT + TGRPO），仅使用ORLM所需1/10的合成数据即达到67.7%的平均求解准确率，超越现有SOTA方法，并通过测试时强化学习将单次生成（Pass@1）与多次生成（Pass@8）的性能差距从13%缩小到7%。

## 研究背景与动机

运筹学（Operations Research, OR）优化问题广泛存在于物流、资源分配、调度等工业场景中。传统上，将自然语言问题描述转化为精确的数学模型和可执行的求解器代码需要高度专业化的人工操作，既耗时又容易出错。

近年来，大语言模型（LLM）在自动化OR建模和求解方面展现了潜力，相关方法分为两类：

**基于提示的方法**（Prompt-based）：利用LLM的上下文学习能力，通过精心设计的提示、少样本示例或链式思维直接生成优化模型或代码。代表工作有Chain-of-Experts、Optimus、MAMO等，无需领域特化微调。

**基于学习的方法**（Learning-based）：通过领域特定数据微调LLM。代表工作有ORLM（使用Llama3-8B，在部分测试中超越GPT-4o）和LLMOPT。这类方法性能更强，但面临两个核心挑战：

**数据需求量大**：ORLM使用了3万条合成数据进行训练，但合成数据的准确率仅约70%，同时人工标注成本极高。如何显著降低标注数据需求？

**输出一致性差**：LLM的单次输出（Pass@1）与多次采样中的最好结果（Pass@8）之间存在显著差距（约13%），表明模型具备能力但缺乏一致性。如何提升单次生成的可靠性？

OR-R1正是为解决这两个问题而设计的。

## 方法详解

### 整体框架

OR-R1采用两阶段训练流程，基座模型为Qwen3-8B：

1. **第一阶段：监督微调（SFT）**——使用少量标注数据教会模型OR建模和代码生成的基本推理模式
2. **第二阶段：测试时组相对策略优化（TGRPO）**——在未标注的测试数据上通过强化学习进一步提升能力和一致性

### 关键设计

#### 1. 监督微调（SFT）阶段

SFT阶段最小化标准的负对数似然损失：

$$\mathcal{L}_\text{SFT}(\theta) = -\mathbb{E}_{(x,o)\sim\mathcal{D}_\text{SFT}}\left[\sum_{t=1}^{|o|}\log P(o_t|x,o_{<t};\theta)\right]$$

关键特点：仅使用ORInstruct数据集中的**3000条**（甚至**100条**）合成样本进行训练，为ORLM使用的3万条数据的1/10。SFT的目的不是追求完美性能，而是让模型习得OR问题建模和求解代码生成的基本范式，为后续RL阶段打下基础。

#### 2. 测试时组相对策略优化（TGRPO）

TGRPO是本文的核心贡献，它将DeepSeek-R1中的GRPO机制与测试时强化学习（TTRL）结合，用于OR领域。

**核心思想**：对于未标注的测试问题，让LLM生成一组 $G$ 个候选输出，通过多数投票确定伪标签，然后将伪标签作为奖励信号进行策略优化。

目标函数继承PPO的裁剪策略：

$$\mathcal{J}_\text{TGRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i\right) - \beta\mathbb{D}_\text{KL}(\pi_\theta||\pi_\text{ref})\right)\right]$$

优势函数通过组内归一化估计：$A_i = \frac{R_i - \text{mean}(\{R_1,...,R_G\})}{\text{std}(\{R_1,...,R_G\})}$

**关键优势**：
- 无需独立的Critic模型，通过组内分数估计基线，大幅降低计算开销
- 利用**未标注**测试数据训练，无需额外的标注成本
- 通过多数投票生成高质量伪标签，实现自监督式学习

#### 3. 复合奖励函数设计

为OR场景定制了三种互补的奖励信号：

**Format Reward**（格式奖励）：检查输出是否包含6个必需字段（数学模型、决策变量、目标函数、约束条件、Python代码、python代码块标记），按比例计分：

$$R_\text{format}(o_i) = \frac{\text{检测到的必需字段数}}{6}$$

**Valid-Code Reward**（代码有效性奖励）：检查生成的代码是否能正确调用coptpy求解器，二值奖励：

$$R_\text{code}(o_i) = \begin{cases}1, & \text{代码能正确调用coptpy} \\ 0, & \text{否则}\end{cases}$$

**Majority Voting Reward**（多数投票奖励）：来自TTRL框架，通过多数投票建立共识输出作为代理标签：

$$R_\text{voting}(y_i, y) = \begin{cases}1, & y_i = y \\ 0, & \text{否则}\end{cases}$$

其中 $y$ 为多数投票结果，仅考虑正常执行的代码结果（排除"No Best Solution"或"None"）。

最终复合奖励：$R_i = R_\text{format}(o_i) + R_\text{code}(o_i) + R_\text{voting}(y_i, y)$

### 损失函数 / 训练策略

- **SFT阶段**：AdamW优化器，warmup-decay学习率调度
- **TGRPO阶段**：AdamW + cosine学习率调度 + PEFT（LoRA），KL散度正则化防止策略过度偏移
- 硬件：4×A100(40G) GPU，BF16精度
- TGRPO仅需约50条未标注数据即可获得显著提升

## 实验关键数据

### 主实验

| 模型 | NL4OPT | MAMO EasyLP | MAMO ComplexLP | IndustryOR | NLP4LP | ComplexOR | OptiBench | ICML Comp. | AVG |
|------|--------|-------------|----------------|------------|--------|-----------|-----------|------------|-----|
| ORLM (Llama3-8B) | 86.9 | 81.6 | 39.3 | 32.0 | 82.0 | 50.0 | 56.5 | 79.3 | 63.5 |
| LLMOPT (Qwen2.5-14B) | 80.3 | 89.5 | 44.1 | 29.0 | 73.4 | 35.3 | 53.8 | 75.3 | 60.1 |
| Qwen3-8B SFT(3K) | 86.0 | 87.0 | 39.9 | 33.0 | 82.9 | 40.7 | 61.4 | 85.8 | 64.6 |
| **OR-R1 SFT(100)-TGRPO** | 88.0 | 87.4 | 45.7 | 30.3 | 84.0 | 46.3 | 61.2 | 84.1 | **65.9** |
| **OR-R1 SFT(3K)-TGRPO** | 88.3 | 86.1 | 49.9 | 35.3 | 84.6 | 46.3 | 62.9 | 88.3 | **67.7** |

OR-R1以仅1/10数据量超越ORLM（63.5%→67.7%，+4.2%）。即使仅用100条SFT样本+TGRPO（65.9%），也超过了使用3万条数据的ORLM。

### 消融实验：奖励组件分析

| 奖励配置 | AVG | 相对SFT基线变化 |
|---------|-----|---------------|
| SFT(3K) 基线 | 66.0 | — |
| +RL($R_\text{format}$) | 66.5 | +0.5 |
| +RL($R_\text{code}$) | 67.2 | +1.2 |
| +RL($R_\text{voting}$) | 68.0 | +2.0 |
| +RL($R_\text{format}$+$R_\text{code}$) | 67.8 | +1.8 |
| +RL($R_\text{format}$+$R_\text{voting}$) | 68.5 | +2.5 |
| +RL($R_\text{code}$+$R_\text{voting}$) | 69.7 | +3.7 |
| +RL(全部三种) | **70.8** | **+4.8** |

三种奖励组件互补，全部组合使用效果最佳。投票奖励单独贡献最大（+2.0），代码有效性次之（+1.2），格式奖励贡献最小（+0.5）。

### 关键发现

1. **极致数据效率**：100条SFT样本+TGRPO即可超越使用3万条数据的ORLM，数据需求降低了300倍
2. **一致性显著提升**：Pass@1与Pass@8的差距从13%缩小到7%，TGRPO有效提升了单次生成的可靠性
3. **TGRPO数据效率**：仅需约50条未标注数据即可获得主要性能提升，进一步增加数据量收益递减
4. **奖励组件互补性**：三种奖励分别把控结构正确性、代码可执行性和数值准确性，组合使用相比任何子集都有明显提升
5. **训练未饱和**：Pass@1在训练末期仍呈上升趋势，受算力限制未能继续训练，暗示更多计算资源可进一步提升性能

## 亮点与洞察

- **数据效率的突破**：合成数据质量低（ORLM报告仅约70%准确率）且人工标注昂贵的问题，通过TGRPO的自监督机制得到巧妙解决——模型从自身对未标注数据的多次采样中学习一致性
- **多数投票作为伪标签的可行性**：在OR场景中，代码执行结果是客观的数值，多数投票可以高可靠度地识别正确答案，这使得TTRL框架在此领域特别适用
- **复合奖励的层次化设计**：从格式→可执行性→数值准确性，形成了从粗到细的奖励层次，确保模型在生成流程的每个阶段都受到有效引导
- **Pass@1 vs Pass@8差距的系统性分析**：将一致性问题明确提出并量化，而非仅关注最佳情形性能

## 局限与展望

1. **测试数据用于训练**：TGRPO使用未标注的测试数据进行训练，虽然不看标签，但存在分布泄露的争议——模型在评估集上微调可能高估了泛化能力
2. **求解器依赖**：奖励函数和评估均绑定coptpy求解器，迁移到其他求解器（如Gurobi、CPLEX）需要重新适配
3. **训练未充分**：作者承认受算力限制训练提前终止，性能曲线仍在上升，最终性能天花板未知
4. **ComplexOR表现不稳定**：在ComplexOR数据集上，完整方法有时反而低于SFT基线（如+RL($R_\text{code}$)下降5.5%），暗示复合奖励在复杂问题上可能引入冲突信号
5. **仅限线性/混合整数规划**：未涉及非线性优化、组合优化等更复杂的OR问题类型

## 相关工作与启发

- OR-R1的TGRPO直接借鉴了DeepSeek-R1的GRPO和TTRL的测试时强化学习思想，首次将这些技术应用于OR领域，证明了RL方法在**多数投票可获得可靠伪标签**的场景中特别有效
- 复合奖励的设计思路（格式+可执行性+数值正确性）可以推广到其他需要LLM生成可执行代码的领域，如科学计算代码生成、数据库查询生成等
- 数据效率的突破（100条→65.9%）展示了SFT+RL的两阶段范式在低资源场景下的潜力，对于标注数据昂贵的专业领域（如医疗、法律）具有启发意义

## 评分

- **理论深度**: ★★★☆☆ — 方法设计清晰但理论贡献有限，主要是已有RL方法的组合应用
- **实验充分性**: ★★★★★ — 8个多样化基准测试、多种消融实验、训练动态分析、数据规模效应分析
- **创新性**: ★★★★☆ — 首次将TTRL应用于OR领域，TGRPO的数据效率令人印象深刻
- **实用性**: ★★★★★ — 直接降低OR自动化的数据和专业门槛，工业应用潜力大
- **综合**: 8/10

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](../../NeurIPS2025/others/hybrid-balance_gflownet_for_solving_vehicle_routing_problems.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](../../ACL2025/others/mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)
- [\[AAAI 2026\] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)

<!-- RELATED:END -->
