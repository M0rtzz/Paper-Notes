---
title: >-
  [论文解读] Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study
description: >-
  [AAAI 2026][人体理解][数据分析] 系统研究了开源 LLM 在数据分析任务中的能力瓶颈，将数据分析分解为数据理解、代码生成和战略规划三个维度，发现**战略规划是决定性因素**而非编码或数据理解；并提出了一种策略引导的数据合成方法，使微调后的 7B/14B 模型达到与 GPT-4o 竞争的性能。
tags:
  - AAAI 2026
  - 人体理解
  - 数据分析
  - LLM Agent
  - 战略规划
  - 数据合成
  - 开源模型
---

# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study

**会议**: AAAI 2026  
**arXiv**: [2506.19794](https://arxiv.org/abs/2506.19794)  
**代码**: [github.com/zjunlp/DataMind](https://github.com/zjunlp/DataMind)  
**领域**: 人体理解  
**关键词**: 数据分析, LLM Agent, 战略规划, 数据合成, 开源模型

## 一句话总结

系统研究了开源 LLM 在数据分析任务中的能力瓶颈，将数据分析分解为数据理解、代码生成和战略规划三个维度，发现**战略规划是决定性因素**而非编码或数据理解；并提出了一种策略引导的数据合成方法，使微调后的 7B/14B 模型达到与 GPT-4o 竞争的性能。

## 研究背景与动机

### 问题定义

数据分析是一个复杂的交互过程，核心在科学发现、商业智能和决策中。它要求模型理解自然语言查询、解释结构化数据、制定假设、生成可执行代码，并跨多轮交互迭代精化推理。LLM 在数据分析中的性能仍由 GPT-4、DeepSeek-R1 等大型闭源模型主导，开源小型模型在真实场景中表现挣扎。

### 核心研究问题

**如何有效提升开源 LLM 在复杂、推理密集型数据分析任务上的能力？**

数学和代码生成领域已证明在高质量合成数据上微调可提升推理能力，但数据分析任务涉及多步交互、动态环境和混合目标，哪些训练数据属性（任务难度、场景多样性、交互结构）能真正带来更好的泛化能力，仍不清楚。

### 研究方法论

作者采用**能力感知方法**，将数据分析过程分解为三个核心维度：

**数据理解（Data Comprehension）**：理解和有效利用结构化数据

**代码生成（Code Generation）**：生成正确高效的分析代码

**战略规划（Strategic Planning）**：将复杂问题分解为可管理步骤

形式化定义数据分析函数：$f_\theta: (\mathcal{D}, \mathcal{Q}, \mathcal{T}) \rightarrow (\mathcal{S}, \mathcal{R})$，其中 $\mathcal{D}$ 是结构化数据，$\mathcal{Q}$ 是分析目标，$\mathcal{T}$ 是工具库，$\mathcal{S}$ 是中间分析状态序列，$\mathcal{R}$ 是最终报告。

## 方法详解

### 整体框架

研究分为两个阶段：
1. **能力诊断**（Section 4）：通过系统的消融实验分析三个核心维度中哪些因素真正影响性能
2. **数据合成**（Section 5）：基于诊断洞察设计策略引导的数据合成方法，提升模型性能

### 关键设计

#### 1. **数据理解实验**：探究表格信息和输入复杂度的影响

**实验 1: 表格信息可见性**

| 设置 | QRData (7B) | QRData (14B) | DiscoveryBench (7B) | DiscoveryBench (14B) |
|------|------------|-------------|-------------------|---------------------|
| w/o Info | 6.57 | 15.09 | 0.42 | 0.42 |
| w/ Info | 7.54 | 15.82 | 1.26 | 0.00 |

提供表格信息（列名、数据类型、样本条目）仅带来微小改善。14B 模型在 DiscoveryBench 上甚至下降，可能因为输入变长导致输出不够聚焦。

**实验 2: 数据复杂度**

引入无关表格作为语义噪声：

| 设置 | QRData (7B) | QRData (14B) | DiscoveryBench (7B) | DiscoveryBench (14B) |
|------|------------|-------------|-------------------|---------------------|
| w/o Extra | 37.96 | 52.55 | 5.44 | 10.88 |
| w/ Extra | 34.55 | 52.07 | 4.18 | 12.13 |

14B 模型几乎不受影响，7B 模型轻微下降。说明**数据理解不是主要瓶颈**，模型在预训练中已内化了基本数据理解能力。

#### 2. **代码能力实验**：评估编码在数据分析中的角色

多模型比较（多轮交互设置）：

| 模型 | QRData | DiscoveryBench |
|------|--------|---------------|
| Qwen2.5-7B-Instruct | 39.71% | 14.64% |
| Qwen2.5-14B-Instruct | 53.53% | 24.27% |
| Qwen2.5-32B-Instruct | 57.18% | 27.62% |
| Qwen2.5-7B-Coder | 36.50% | 13.60% |
| R1-Distill-Qwen-7B | 30.41% | 7.95% |
| GPT-4o | 59.85% | 28.03% |
| DeepSeek-v3 | 65.21% | 36.82% |
| DeepSeek-R1 | 63.26% | 37.66% |

**关键发现**：
- **代码专业化不等于数据分析能力**：Qwen2.5-7B-Coder 反而不如通用 Instruct 模型
- **蒸馏可能导致功能幻觉**：R1-Distill-Qwen-7B 表现最差，经常"幻想"文件解释而非生成可执行代码
- **长上下文 ≠ 高效执行**：1M 上下文版本与标准版编码能力相当，但后者规划效率更高（用更少轮完成任务）

**错误分析**（354 个错误样本的手动标注）：仅少数错误源于语法/语义代码缺陷（如无效语法），多数错误源于高层推理失败（如错误假设、过早终止），进一步证明**规划比编码更重要**。

#### 3. **战略规划实验**：系统评估四个关键方面

**（a）交互轮次**

三种策略：Short（2-3 轮）、Medium（4-5 轮）、Long（6+ 轮）、Mixed。

| 轮次类别 | 样本数 | QRData | DiscoveryBench |
|---------|-------|--------|---------------|
| All | 5613 | 48.66% | 15.00% |
| Short | 1034 | 47.68% | 23.85% |
| Medium | 3559 | 49.15% | 18.83% |
| Long | 1020 | 47.94% | 18.41% |
| Medium + Short | 4593 | 47.45% | 21.34% |

核心发现：
- 中等长度交互通常表现最好（推理深度与聚焦的平衡）
- 混合策略一致表现最差（变化的轮次长度干扰了模型学习稳定交互模式）
- **数据质量 > 数据数量**：中等轮次的子集持续优于全数据集训练

**（b）推理长度**

三种设置：Original（原始）、Full（完整 \<think\> 轨迹）、Summarized（思维轨迹的摘要）。

核心发现：
- **更长的推理 ≠ 更好**：Full 设置在大多数配置下反而不如 Original
- **信息相关性 > 推理长度**：Summarized 设置持续匹配或超过基线
- Token 预算存在**收益递减**：QRData 上增加预算有用，DiscoveryBench 上反而降低性能

**（c）任务复杂度**

按模型能力分级：Easy（7B 可解）、Medium（仅 14B 可解）、Hard（需 DeepSeek-R1）。

| 难度 | QRData | DiscoveryBench |
|------|--------|---------------|
| Easy | 42.58% | 20.50% |
| Medium | 51.34% | 18.83% |
| Hard | 48.18% | 19.50% |
| Medium + Hard | 51.34% | 23.01% |

中等+困难数据组合在两个数据集上都表现最好，说明接触更复杂任务有助于增强模型泛化。

**（d）问题多样性**

| 多样性 | QRData | DiscoveryBench |
|--------|--------|---------------|
| 原始分布 | 46.72% | 20.92% |
| 均衡采样 | 45.00% | 21.76% |

领域多样性的影响微乎其微，说明**推理策略的多样性和丰富性比问题领域多样性更重要**。

### 损失函数 / 训练策略

**策略引导的数据合成三阶段**：

1. **Prompt-Based Answer Generation**：利用提示生成技术为每个查询生成多个候选答案
2. **Targeted Instance Selection**：优先选择中等长度对话和中等-高难度样本
3. **Reasoning-Driven Data Enrichment**：为每个选定实例添加简洁的推理摘要

最终数据集：2.8k 实例，用于 SFT 微调。基于 Qwen2.5-7B/14B-Instruct，使用 LoRA 微调（用于 Strategic Planning 评估）。

## 实验关键数据

### 主实验

| 模型 | QRData | DiscoveryBench |
|------|--------|---------------|
| GPT-4o | 59.85% | 28.03% |
| DeepSeek-v3 | 65.21% | 36.82% |
| DeepSeek-R1 | 63.26% | 37.66% |
| Qwen2.5-7B (baseline) | 39.71% | 14.64% |
| **Qwen2.5-7B (ours)** | **53.77%** | **22.59%** |
| Qwen2.5-14B (baseline) | 53.53% | 24.27% |
| **Qwen2.5-14B (ours)** | **58.15%** | **36.82%** |

7B 模型提升巨大：QRData +14.06%，DiscoveryBench +7.95%。14B 模型在 DiscoveryBench 上达到 36.82%，与 DeepSeek-v3 持平，超过 GPT-4o（28.03%）。

### 消融实验

| 消融维度 | 关键发现 | 效果量级 |
|---------|---------|---------|
| 去掉表格信息 | 性能几乎不变 | 数据理解非瓶颈 |
| 使用代码专用模型 | 性能反而下降 | 通用 Instruct 更优 |
| 短轮次 vs 中等轮次 | 中等轮次通常更好 | 任务依赖 |
| 完整推理 vs 摘要推理 | 摘要持续优于完整 | 质量 > 长度 |
| 中等难度 vs 简单 | 中等 + 困难组合最优 | 复杂任务增强泛化 |
| 领域多样性 | 原始分布 ≈ 均衡采样 | 多样性不重要 |

### 关键发现

1. **战略规划是决定性因素**：数据理解和代码生成已在预训练中被充分学习，规划能力是开源模型与闭源模型差距的主要来源
2. **交互设计和任务复杂度显著影响推理能力**：中等长度交互 + 中等-高难度数据 = 最佳学习效率
3. **数据质量远比多样性重要**：2.8k 精心策划的样本就能带来与数万样本训练相当或更好的效果
4. **收益递减现象**：随着模型规模增大，微调带来的改善减小（14B 的改善不如 7B 显著）

## 亮点与洞察

1. **能力分解方法论**：将数据分析分解为三个正交维度并逐一分析，方法论清晰严谨，为社区提供了有价值的研究范式
2. **"规划 > 编码 > 数据理解"的发现**：颠覆了"开源 LLM 缺乏编码能力"的常见假设，揭示真正瓶颈在于战略规划
3. **推理摘要优于完整推理**：简洁的推理摘要比冗长的思维链更有效，说明信息密度和逻辑连贯性比长度更重要
4. **小数据大效果**：仅 2.8k 样本的策略引导合成数据就能让 14B 模型匹配 DeepSeek-v3 在 DiscoveryBench 上的表现

## 局限与展望

1. **数据集规模有限**：2.8k 样本可能不足以覆盖更广泛的分析场景
2. **训练数据分布偏向 7B 模型**：策略是基于 Qwen2.5-7B 构建的，可能与更大模型的归纳偏置不太匹配
3. **仅评估两个基准**：DiscoveryBench 和 QRData 可能不足以全面反映真实场景的复杂性
4. **缺乏 RL 方法对比**：仅使用 SFT，未探索强化学习（如 GRPO）在数据分析任务上的效果
5. **静态交互模式**：混合策略表现差但仍有潜力——需要课程调度或自适应控制等更精细的策略设计
6. **评估依赖 GPT-4o-mini**：使用 LLM 作为评判可能引入评估偏差

## 相关工作与启发

- **S1 & LIMO**：分别在数学推理中展示了"预算控制"和"少量精选样本涌现复杂能力"的可能性，启发本文的数据策略
- **Data Interpreter**：使用层次依赖图表示工作流，实现自动任务分解
- **ReAct 框架**：本文所有模型都使用 ReAct 进行多轮交互
- **DeepSeek-R1**：用于生成高质量推理轨迹和作为"难题标定器"
- **启发**：在 Agent 场景中，规划能力的重要性远超单一技能（编码、数据理解），这对 Agent 训练策略有重要指导意义

## 评分

- 新颖性: ⭐⭐⭐⭐ （能力分解分析方法论新颖，洞察有价值）
- 实验充分度: ⭐⭐⭐⭐⭐ （多维度系统消融，覆盖多个模型和设置）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，发现总结精炼，图表丰富）
- 价值: ⭐⭐⭐⭐ （为开源 LLM 数据分析能力提升提供了实用的指导路线图）

<!-- RELATED:START -->

## 相关论文

- [Improving Model Alignment through Collective Intelligence of Open-Source LLMs](../../ICML2025/human_understanding/improving_model_alignment_through_collective_intelligence_of_open-source_llms.md)
- [Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)
- [Bias Association Discovery Framework for Open-Ended LLM Generations](bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [Self-Correction Distillation for Structured Data Question Answering](self-correction_distillation_for_structured_data_question_answering.md)

<!-- RELATED:END -->
