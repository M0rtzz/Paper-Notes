---
title: >-
  [论文解读] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning
description: >-
  [ICLR 2026][强化学习] 提出 Supervised Reinforcement Learning (SRL)，将问题求解重新建模为逐步动作生成过程，通过基于序列相似度的密集奖励信号，使小模型能够从专家轨迹中学习原本 SFT 和 RLVR 都无法解决的困难推理问题。
tags:
  - ICLR 2026
  - 强化学习
  - 监督学习
  - 逐步推理
  - 序列相似度奖励
  - 难题学习
---

# Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning

**会议**: ICLR 2026  
**arXiv**: [2510.25992](https://arxiv.org/abs/2510.25992)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 强化学习, 监督学习, 逐步推理, 序列相似度奖励, 难题学习

## 一句话总结

提出 Supervised Reinforcement Learning (SRL)，将问题求解重新建模为逐步动作生成过程，通过基于序列相似度的密集奖励信号，使小模型能够从专家轨迹中学习原本 SFT 和 RLVR 都无法解决的困难推理问题。

## 研究背景与动机

大语言模型在多步推理任务中面临一个根本性困境：

**RLVR 的局限性**：基于最终答案正确性的强化学习（如 GRPO）依赖于模型在有限 rollout 中采样到正确解的能力。对于小模型（如 7B），在困难问题上 pass@k 接近零，导致奖励信号极度稀疏，模型无法从中学到有意义的策略。DAPO 等方法通过过滤掉全错/全对的样本来缓解，但本质上放弃了这些难题。

**SFT 的局限性**：监督微调通过 token 级别的模仿学习，强制模型逐字复制专家轨迹。对于长而复杂的推理链，这种刚性模仿容易导致过拟合和浅层推理行为。实验表明，在 s1K 数据集上直接做 SFT 反而导致性能下降（见 Figure 1）。

**核心矛盾**：难题数据量少且推理链复杂，SFT 学不好；模型又采样不到正确解，RLVR 也学不好。这在训练小型开源模型时尤为突出。

作者将这类问题定义为 $\mathcal{D}_{\text{hard}}$——模型在 $k$ 次采样中成功率趋近于零的问题集合。SRL 的目标就是在这个困难区域提供有效的学习信号。

## 方法详解

### 整体框架

SRL 将问题求解重新表述为顺序决策过程。核心思想是：不再要求模型一次性生成完整解答或逐 token 模仿专家，而是训练模型在每一步生成一个"动作"（即推理步骤），并通过该动作与专家动作的相似度来提供奖励。

整个流程分为三个阶段：
1. 从专家轨迹中提取逐步动作序列
2. 构建逐步训练数据
3. 使用序列相似度奖励进行 RL 训练

### 关键设计

1. **基于动作的问题建模**：给定专家解答轨迹 $\mathbf{y}$，将其分解为动作元组序列 $\mathbf{y} = \{\mathbf{y}_{\text{step}}^n\}_{n=1}^N$。每个步骤代表一个逻辑动作——在数学推理中是代数运算，在软件工程中是终端命令。这种建模方式是领域无关的。

   → 核心思路：将连续的推理过程离散化为可对比的原子操作  
   → 设计动机：细粒度分解降低了每步的学习难度，使模型在局部就能获得有意义的反馈

2. **逐步训练数据构建**：从一个包含 $N$ 步的完整解中构建 $N-1$ 个部分轨迹。对第 $k$ 步，输入为 $\mathbf{x}_{\text{step}}^k = [\mathbf{x}, \mathbf{y}_{\text{step}}^1, \ldots, \mathbf{y}_{\text{step}}^{k-1}]$，任务是预测下一步 $\mathbf{y}_{\text{step}}^k$。

   → 核心思路：将一条专家解答转换为多条训练实例，每条对应不同的中间状态  
   → 设计动机：极大增加训练数据量，同时教会模型如何从各种中间状态继续推理

3. **序列相似度奖励与内部独白**：模型首先生成被 `<think>` 标签包裹的内部推理过程 $\mathbf{y}'_{\text{think}}$，然后输出动作 $\mathbf{y}'^k_{\text{step}}$。奖励仅基于动作与专家动作的序列相似度计算：

   $$R(\mathbf{y}'^k_{\text{step}}, \mathbf{y}^k_{\text{step}}) = \frac{2M}{T}$$

   其中 $T$ 是两个序列的总元素数，$M$ 是所有非重叠匹配块中元素的总数。实际使用 Python 的 `difflib.SequenceMatcher` 实现。如果输出格式不正确则奖励为 $-1$。

   → 核心思路：在动作层面而非 token 层面比较，给模型思考的自由空间  
   → 设计动机：奖励是密集的（$r \in [0,1]$）而非稀疏的二值信号，即使所有 rollout 都不完全正确也能提供梯度信息

4. **动态采样策略**：过滤掉 rollout 奖励方差过低的样本（标准差低于阈值 $\epsilon$），因为这些样本的优势函数接近零，提供的学习信号微弱。持续采样和过滤直到批次填满。

   → 核心思路：推广了 DAPO 中针对二值奖励的过滤策略到连续奖励  
   → 设计动机：避免在"已经学好"或"完全无法区分"的样本上浪费计算

### 损失函数 / 训练策略

使用 GRPO 优化目标函数，结合上述序列相似度奖励。关键超参数：
- 批次大小 512（SRL），128（GRPO，因过滤率高）
- 学习率 5e-7
- Rollout 数量 8
- KL 损失系数 0（无 KL 约束）
- 最多训练 30 个 epoch，根据验证集选最佳 checkpoint

训练流程可以是 SRL 单独使用，也可以是 SRL → RLVR 的两阶段课程学习。

## 实验关键数据

### 主实验

**数学推理**（基座模型：Qwen2.5-7B-Instruct，训练数据：s1K-1.1，1000 道难题）

| 方法 | AMC23 Avg@32 | AIME24 Avg@32 | AIME25 Avg@32 | Minerva Math | 平均 |
|------|-------------|---------------|---------------|-------------|------|
| Base Model | 49.3 | 10.5 | 7.5 | 34.9 | 24.6 |
| SFT (R1 reasoning) | 26.8 | 3.9 | 5.4 | 20.2 | 16.6 |
| RLVR (GRPO) | 52.0 | 11.1 | 7.4 | 33.8 | 24.5 |
| **SRL** | **51.5** | **13.2** | **7.1** | **36.4** | **27.6** |
| **SRL → RLVR** | **52.1** | **13.3** | **8.6** | **36.4** | **28.3** |

关键观察：SFT 在难数据上性能严重下降（-8 个点）；RLVR 基本持平；SRL 带来显著提升（+3.0%）；SRL→RLVR 达到最强（+3.7%）。

**软件工程**（基座模型：Qwen2.5-Coder-7B-Instruct，5000 条专家轨迹）

| 方法 | Oracle File Edit | End-to-End |
|------|-----------------|------------|
| Base Model | 5.8 | 3.2 |
| SWE-Gym-7B (SFT) | 8.4 | 4.2 |
| **SRL** | **14.8** | **8.6** |

SRL 在 Oracle 设置下相对 SWE-Gym-7B 提升 74%，端到端性能翻倍。

### 消融实验

| 配置 | 平均性能 | 说明 |
|------|---------|------|
| SRL w/o 动态采样 | 24.7 | 过滤低方差样本带来 +2.9% |
| SRL w/ 动态采样 | 27.6 | 确认过滤策略的重要性 |
| 最终答案奖励 (RLVR) | 24.5 | 稀疏奖励效果有限 |
| 整体序列相似度（单步）| 25.9 | 有一定提升但不如多步 |
| 多步序列相似度 (SRL) | 27.6 | 细粒度引导效果最优 |

### 关键发现

1. **推理长度未显著增加**：SRL 训练后的模型与基座模型的推理长度分布几乎一致，说明性能提升来自推理质量而非更多 token
2. **涌现交错推理模式**：SRL→RLVR 模型展示了独特的推理行为——(1) 前期规划，(2) 过程中动态调整，(3) 反思性验证。这些模式在传统模型中不存在
3. **跨领域泛化**：SRL 不仅在数学推理中有效，在软件工程 Agent 任务中同样表现出色，证明框架的通用性

## 亮点与洞察

1. **填补了一个重要空白**：在 SFT 过拟合和 RLVR 稀疏奖励之间找到了一个优雅的中间方案。通过逐步分解 + 序列相似度奖励，既保留了专家指导又给予模型推理自由
2. **奖励函数设计精妙**：只对动作计算相似度、不约束内部思考过程，这个设计让模型可以发展出自己的推理风格。使用 `difflib.SequenceMatcher` 使得奖励计算既快速又稳定
3. **课程学习策略**：SRL → RLVR 的组合将 SRL 视为一种更好的初始化手段，先通过细粒度专家指导建立基础推理能力，再通过自由探索进一步优化
4. **实用性强**：无需训练额外的奖励模型，无需复杂的过程奖励标注，仅利用现有的 SFT 数据就能构建训练信号

## 局限与展望

1. **依赖专家轨迹的结构化格式**：SRL 要求解答轨迹具有明确的步骤划分（如 DeepSeek R1 的编号步骤格式），并非所有数据都满足
2. **学生模型需要基本的指令跟随能力**：如果基座模型完全不能生成格式正确的输出，初始 rollout 就无法提供有用的学习信号
3. **序列相似度奖励可能不够精细**：基于字符串匹配的相似度可能无法区分语义等价但表述不同的数学步骤
4. **未探索更大模型**：实验仅在 7B 模型上进行，更大模型上 SRL 的边际收益尚不清楚
5. **可扩展至过程奖励模型**：结合 PRM 可能提供比序列相似度更语义化的步骤级奖励

## 相关工作与启发

- **DeepSeek-R1** 和 **GRPO** 是 RLVR 的代表工作，SRL 建立在 GRPO 的优化框架之上
- **s1K**、**LIMO** 等工作证明少量高质量数据可以有效蒸馏推理能力，SRL 在同样的数据上走得更远
- **SWE-Gym** 和 **SWE-Smith** 提供了软件工程领域的 SFT 数据，SRL 在相同数据上大幅超越 SFT 基线
- 启发：将 RL 和模仿学习的思想结合，在"模仿什么"和"自由探索什么"之间找到平衡点

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — SFT 和 RLVR 之间的巧妙融合，填补了重要空白
- 实验充分度: ⭐⭐⭐⭐ — 数学+SWE 两个领域验证，消融充分，但仅限 7B 模型
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰，方法描述精确，图示直观
- 价值: ⭐⭐⭐⭐⭐ — 为训练小模型处理难题提供了实用的新范式

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ICML 2025\] AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence](../../ICML2025/code_intelligence/adaptivestep_automatically_dividing_reasoning_step_through_model_confidence.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)

<!-- RELATED:END -->
