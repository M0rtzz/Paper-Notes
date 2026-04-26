---
title: >-
  [论文解读] Improving Continual Pre-training Through Seamless Data Packing
description: >-
  [ACL 2025][持续预训练] 提出 Seamless Packing (SP) 数据打包策略，通过两阶段方法——滑动窗口处理长文本 + FFD 算法打包短文本——在持续预训练中保持上下文连续性、最小化截断和填充，在 99% 的实验设置中超越基线方法。
tags:
  - ACL 2025
  - 持续预训练
  - 数据打包
  - 滑动窗口
  - 上下文连续性
  - bin packing
---

# Improving Continual Pre-training Through Seamless Data Packing

**会议**: ACL 2025  
**arXiv**: [2505.22018](https://arxiv.org/abs/2505.22018)  
**代码**: [GitHub](https://github.com/Infernus-WIND/Seamless-Packing)  
**领域**: NLP / 预训练  
**关键词**: 持续预训练, 数据打包, 滑动窗口, 上下文连续性, bin packing

## 一句话总结

提出 Seamless Packing (SP) 数据打包策略，通过两阶段方法——滑动窗口处理长文本 + FFD 算法打包短文本——在持续预训练中保持上下文连续性、最小化截断和填充，在 99% 的实验设置中超越基线方法。

## 研究背景与动机

持续预训练（Continual Pre-training）是将通用 LLM 适配到特定领域的有效策略。在训练前，需要将变长文本打包成固定长度的序列以便并行训练。当前最常用的方法是**拼接截断（Concatenation and Truncation, CT）**：将所有文本首尾拼接，然后按模型输入长度切分为等长序列。

这种简单方法有两个关键问题：

**上下文断裂**: 随意的截断会将语义相关的内容割裂到不同序列中。例如"事件A将于某日在某地举办"被切开后，时间地点信息与事件描述分离，模型无法学习它们的关联

**截断导致幻觉**: 在摘要生成任务中，关键解释性内容被截断可能导致不忠实的输出

另一类方法使用 padding 避免截断，但 padding 占据了宝贵的输入空间却不提供任何信息。作者将数据打包形式化为一个优化问题：**在序列长度约束下，最大化上下文连续性，同时最小化截断和填充**。

## 方法详解

### 整体框架

Seamless Packing 分为两个顺序执行的阶段：
- **Stage 1 (滑动窗口)**: 处理长文本，通过重叠实现完整序列覆盖
- **Stage 2 (FFD打包+丢弃)**: 处理剩余短文本，最小化填充和截断

### 关键设计

#### 1. Stage 1: 滑动窗口 (Sliding Window)

**功能**: 对满足条件的长文本，使用动态滑动窗口生成连续序列，使一篇长文本完整覆盖多个序列，无需与其他文本拼接。

**核心思路**: 传统方法中，长文本除以序列长度后的余数部分无法填满一个完整序列，需要与下一篇文本拼接，导致上下文断裂。SP 通过在连续序列间引入适量重叠（overlap），让余数部分被吸收，使文本完整覆盖 n+1 个序列而非 n 个。

**关键参数**: 最大重复率 $r_{max}$，控制连续序列间的重叠量。给定文本长度 $L_{original}$ 和可分为 $n$ 个完整序列：

$$L_{max\_overlap} = \lceil n \times r_{max} \times L_{seq} \rceil$$

当 $L_{original} + L_{max\_overlap} \geq (n+1) \times L_{seq}$ 时，可以使用滑动窗口。实际重叠量动态计算：

$$L_{final\_overlap} = \lceil \frac{(n+1) \times L_{seq} - L_{original}}{n} \rceil$$

**设计动机**: 
- 固定步幅的传统滑动窗口无法适应变长文本，要么覆盖不完整要么产生过多重叠
- 用 $r_{max}$ 而非步幅作为控制参数，更直观地控制冗余程度
- 减少了不同文档片段混合在同一序列中的情况

**理论分析**: 当 $n \geq \lceil 1/r \rceil$ 时，所有长文本都可用滑动窗口处理。设 $r_{max}=0.3$ 时，理论上约 62% 的文本可在 Stage 1 处理。

#### 2. Stage 2: FFD 打包 + 丢弃 (Packing with Dropping)

**功能**: 将 Stage 1 遗留的短文本片段高效打包为固定长度序列。

**核心思路**: 将问题建模为 bin packing 变体，使用 First-Fit-Decreasing (FFD) 近似算法。关键创新是允许 bin 容量略大于序列长度（$c_{bin} = L_{seq} + c_{extra}$），超出部分直接丢弃（dropping），而非用 padding 填充。

**算法流程**:
1. 按长度降序排列所有短文本
2. 对每个文本，找到第一个能容纳它的 bin 放入
3. 超过 $L_{seq}$ 的 token 被丢弃
4. 少量仍未满的 bin，将其内容拼接后重新切分

**设计动机**:
- Dropping 消除了 padding 的低效——训练不被无意义 token 稀释
- $c_{extra}$ 精确控制丢弃量，设置得当时丢弃的 token 占比极小（约 3.4%）
- 选择 FFD 而非 BFD（Best-Fit-Decreasing）：FFD 的早停机制使计算速度更快（BFD 慢 29%），性能相当

### 损失函数 / 训练策略

SP 是纯数据预处理方法，不改变训练损失函数或优化策略。它与现有的数据采样、灾难性遗忘缓解方法互补，可以无缝集成。

## 实验关键数据

### 主实验：持续预训练困惑度

| 模型 | 领域 | CT | BFD | SP |
|------|------|-----|-----|-----|
| GPT2-812M | News | 10.79 | 10.28 | **10.13** |
| Llama3.2-1B | News | 11.48 | 10.52 | **10.07** |
| Llama3.2-1B | Finance | 6.27 | 5.71 | **5.52** |
| Qwen2.5-1.5B | Med | 6.71 | 6.47 | **6.43** |

SP 在所有 9 个模型×领域组合中均取得最低困惑度。

### 主实验：下游任务性能（全参数微调）

| 任务 | 模型 | OM | CT | BFD | SP |
|------|------|------|------|------|------|
| BBC News | Llama3.2-1B | 97.22 | 97.44 | 97.54 | **97.64** |
| AG News | Qwen2.5-1.5B | 88.53 | 89.39 | 89.71 | **90.37** |
| Fin Sentiment | Llama3.2-1B | 87.25 | 88.18 | 88.18 | **88.68** |
| ChemProt | Qwen2.5-1.5B | 81.39 | 81.89 | 81.83 | **82.64** |
| PubMed Class | Qwen2.5-1.5B | 85.10 | 86.32 | 86.22 | **86.87** |

SP 在 99% 的设置中表现最佳；BFD 在部分任务中反而不如 CT，而 SP 始终稳健。

### 消融实验

| 方法 | PubMed Class (3模型平均) | ChemProt (3模型平均) |
|------|------------------------|---------------------|
| FFD | 85.65 | 80.04 |
| BFD | 85.61 | 80.34 |
| BFD-m (增大bin) | 85.57 | 80.53 |
| SP (完整) | **86.16** | **81.48** |

- BFD-m > BFD：增大 bin 容量并丢弃有效
- SP > BFD-m：滑动窗口阶段贡献显著
- FFD ≈ BFD：BFD 的额外计算成本不值得

### 泛化性验证

| 设置 | 结果 |
|------|------|
| 混合领域 (News+Finance+Med) | SP 在所有3个任务上最优 |
| 通用领域 (RedPajama → GLUE) | SP 在 MNLI/QNLI/RTE 等所有任务上最优 |
| 跨语言 (法语 C4 → XNLI) | SP 69.04 > BFD 68.64 > CT 67.72 |
| LoRA 微调 | SP 一致优于 BFD 和 CT |

### 超参数分析

- **$r_{max}$ = 0.3** 最优：更高导致过多重复，更低则覆盖不足
- **$c_{extra}$ = 50** 最优（$L_{seq}$ = 2048 时）：更大导致过多截断，更小则效果不明显

### 关键发现

1. **SP 提升幅度约为 BFD 的 4 倍**: SP 平均提升 0.96% vs CT，BFD 仅 0.24%
2. **BFD 不稳定**: 每个模型上都有 2-4 个任务 BFD 不如 CT，SP 则始终一致
3. **Case Study**: 注入合成事件描述后，SP 训练的模型能正确回忆日期（3/5）和地点（5/5），BFD 全部幻觉
4. **理论预测与实际吻合**: Stage 1 覆盖约 62% 文本，Stage 2 仅处理约 3.4% 的 token

## 亮点与洞察

- **问题角度新颖**: 在持续预训练中，大家关注数据选择、采样策略、灾难性遗忘，但很少有人认真研究"数据怎么排"这个基础问题
- **方法简洁高效**: 两阶段设计分别对应长文本和短文本，互补而非冗余
- **dropping > padding**: 在 SP 框架下，丢弃极少量 token 比填充大量 padding 更有效，这是一个反直觉但有价值的发现
- **实验极其全面**: 3 个模型架构 × 3 个领域 × 8 个下游任务 + 混合领域/通用领域/跨语言/LoRA 验证 + 消融 + 超参数分析

## 局限与展望

1. **Dropping 的理论解释不完善**: 为什么丢弃少量 token 比 padding 更好？缺乏理论框架
2. **仅验证了持续预训练**: 在标准预训练（from scratch）中的效果未知
3. **未覆盖代码领域**: 代码的结构化特性可能需要不同的打包策略
4. **模型规模有限**: 最大仅 3B 参数，更大模型的效果待验证
5. **可探索内容感知的打包**: 当前打包仅基于长度，可以考虑语义相关性来决定哪些文本放在一起

## 相关工作与启发

- **Krell et al. (2021)**: 提出 packing 算法最大化序列利用率，但未考虑上下文连续性
- **Ding et al. (2024)**: 分析截断影响并提 BFD 算法，SP 的 Stage 2 在此基础上改进
- **In-context Pretraining (Shi et al., 2024)**: 按语义相关性排列文档，与 SP 关注不同维度（排列 vs 打包）
- **启发**: 数据工程（data engineering）在模型训练中的作用被低估——即使数据质量相同，打包方式不同也能带来显著性能差异

## 评分

- **新颖性**: ⭐⭐⭐ — 方法本身是已有技术（滑动窗口 + bin packing）的巧妙组合，核心创新在于问题发现和工程设计
- **实验充分度**: ⭐⭐⭐⭐⭐ — 极其全面，包括多模型、多领域、多任务、跨语言、LoRA、消融、超参数等全维度验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，理论分析与实验验证结合良好，Figure 1 直观
- **价值**: ⭐⭐⭐⭐ — 提供了一个即插即用的数据预处理工具，对持续预训练实践有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](../../ICCV2025/llm_pretraining/ace-g_improving_generalization_of_scene_coordinate_regression_through_query_pre-.md)
- [\[ACL 2025\] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)

<!-- RELATED:END -->
