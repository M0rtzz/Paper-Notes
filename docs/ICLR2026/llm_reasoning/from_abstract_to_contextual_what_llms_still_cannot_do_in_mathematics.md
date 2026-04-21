---
title: >-
  [论文解读] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics
description: >-
  [ICLR 2026][LLM推理][数学推理] 本文提出 ContextMATH 基准，通过将 AIME 和 MATH-500 的抽象数学问题转换为两种情境变体（场景嵌入 SG 和复杂度缩放 CS），系统揭示了LLM在情境化数学推理中的大幅性能下降——开源模型在 SG 上平均下降 13%，CS 上下降 34%——并识别出"问题建模"和"推理执行"是两个互补的性能瓶颈。
tags:
  - ICLR 2026
  - LLM推理
  - 数学推理
  - 情境推理
  - 问题建模
  - 基准评测
  - LLM能力评估
---

# From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

**会议**: ICLR 2026  
**arXiv**: [2601.23048](https://arxiv.org/abs/2601.23048)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 数学推理, 情境推理, 问题建模, 基准评测, LLM能力评估

## 一句话总结

本文提出 ContextMATH 基准，通过将 AIME 和 MATH-500 的抽象数学问题转换为两种情境变体（场景嵌入 SG 和复杂度缩放 CS），系统揭示了LLM在情境化数学推理中的大幅性能下降——开源模型在 SG 上平均下降 13%，CS 上下降 34%——并识别出"问题建模"和"推理执行"是两个互补的性能瓶颈。

## 研究背景与动机

LLM 在数学基准测试上已取得接近完美的表现（AIME 接近满分，甚至达到 IMO 金牌水平），但这种成功仍局限于定义明确的抽象问题。在实际应用中，数学问题往往以叙事情境的形式出现——从金融分析到科学研究，数学核心需要从具体描述中提取和建模。

现有数学基准（如 GSM8K、MATH、AIME）几乎全部针对抽象问题，即使包含简单叙事（如"Jack 有 8 支笔"），这些上下文也浅薄且有限。这导致一个关键能力——**情境化数学推理（contextual mathematical reasoning）**——一直未被充分评估。本文试图回答：当数学核心被嵌入叙事场景时，LLM 的表现如何？

## 方法详解

### 整体框架

ContextMATH 基于 AIME 2024、AIME 2025 和 MATH-500（难度 ≥3）构建，将每道原始问题转换为两种情境变体，并在 61 个模型（46 个开源 + 15 个商用）上进行系统评估。

### 关键设计

**场景嵌入（Scenario Grounding, SG）**：将抽象数学结构嵌入具有真实实体和交互的叙事场景中，数学推理核心不变。例如，将变量 (a, b, c) 映射为系统组件。SG 主要测试模型在存在情境噪声时能否正确理解和应用其数学知识。

**复杂度缩放（Complexity Scaling, CS）**：将显式条件隐藏在子问题中。例如，直接给出"25 个指示灯"变为"指示灯的唯一配对数恰好为 300"。CS 不仅引入情境化，还要求模型先解决子问题恢复原始条件，模拟现实中约束条件间接给出的情况，同时减少对表面模式匹配的依赖。

**问题建模评估**：为深入分析失败原因，专门设计了三个指标评估模型从情境中抽取数学建模的能力：
- **建模准确率（Formulation Accuracy）**：模型正确将情境翻译为数学问题的比率
- **建模必要性（Formulation Necessity）**：P(建模正确 | 答案正确)，衡量正确建模对正确求解的必要程度
- **建模充分性（Formulation Sufficiency）**：P(答案正确 | 建模正确)，衡量正确建模能否可靠地导向正确答案

**基准构建**：使用 o1-mini 通过多步提示生成情境变体，然后由 3 位具有计算机科学高级学位和竞赛数学背景的专家独立审核，确保数学等价性、叙事可信度和无歧义性。SG 平均 133 词，CS 平均 176 词。

### 损失函数 / 训练策略

在训练实验部分，作者探索了两种策略：

**端到端微调**：基于 DeepMath-103K 数据集，使用 Qwen3-Base 系列在三种数据配置下 SFT：
- SFT_Ori：仅原始数学问题（50k）
- SFT_Syn：仅合成情境问题（50k）
- SFT_Mix：两者混合（100k）

**专用建模模型训练**：将情境-原始问题对作为训练数据，训练专门的建模模型，再与求解器串联。测试了未微调和微调两种设置。

## 实验关键数据

### 主实验

商用模型在 ContextMATH 上的性能下降：

| 模型 | AIME24-Ori | AIME24-SG | AIME24-CS | AIME25-Ori | AIME25-SG | AIME25-CS |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| GPT-5 | 90.0% | 83.3% (-7%) | 80.0% (-11%) | 90.0% | 80.0% (-11%) | 66.7% (-26%) |
| DeepSeek-R1 | 93.3% | 70.0% (-25%) | 66.7% (-29%) | 86.7% | 73.3% (-15%) | 53.3% (-38%) |
| Gemini 2.5 Pro | 83.3% | 73.3% (-12%) | 76.7% (-8%) | 83.3% | 56.7% (-32%) | 50.0% (-40%) |
| o3 | 83.3% | 70.0% (-16%) | 66.7% (-20%) | 76.7% | 70.0% (-9%) | 60.0% (-22%) |
| Qwen3-32B | 81.2% | 67.9% (-16%) | 57.1% (-30%) | 70.0% | 54.4% (-22%) | 45.0% (-36%) |

关键发现：即使 GPT-5 在 AIME25-CS 上也下降 26%；DeepSeek-R1 在 AIME25-CS 上下降 38%。

### 消融实验

| 实验维度 | 关键结论 |
|---------|---------|
| 建模分析 (GPT-5) | 平均建模准确率 81.4%，必要性 85.6%，充分性 82.7% |
| 模型规模效应 | Qwen3-0.6B 建模 42.8% → Qwen3-32B 75.0%，规模有帮助但不解决问题 |
| 错误类型分析 | 建模错误占约 80%，远超计算、逻辑和其他错误 |
| SFT 效果 (14B) | SFT_Mix 在 AIME24-SG 从 11.0%→52.5%，平均提升 31.9% |
| SFT_Ori vs SFT_Syn | 情境数据(SFT_Syn)在情境问题上更有效，混合(SFT_Mix)整体最优 |
| 专用建模模型 | 直接求解(57.7%) > 未微调建模+求解(56.2%) > 微调建模+求解(24.6%) |
| 泛化评估 | SFT_Mix 在 AMC23 和 Math-Perturb 上也有提升，不损害抽象推理能力 |

### 关键发现

1. **情境化是普遍瓶颈**：无论开源还是商用模型，从抽象到情境的性能下降是普遍的，CS 比 SG 下降更严重
2. **建模是首要瓶颈**：正确求解的样本中，建模准确率显著高于平均水平（必要性高）；但正确建模并不总能导向正确答案（充分性不足）
3. **两个互补瓶颈**：建模能力和推理能力是两个独立但互补的瓶颈，随模型规模增大两者都改善但不消除
4. **过度专业化风险**：进一步的 SFT/RL 可能过拟合到标准格式，在 SG/CS 上反而下降更多
5. **建模不可独立训练**：专用建模模型训练后串联求解器，性能反而崩溃（从 57.7% 降至 ~22%），说明建模能力很难从配对监督中单独习得

## 亮点与洞察

- **精准问题定义**：将"情境化数学推理"明确为一个独立的、未被充分研究的能力维度，不同于简单的数学应用题
- **实验设计巧妙**：SG 保持推理难度不变只增加情境，CS 同时增加条件恢复的认知负担，两者互补形成完整的评估谱系
- **瓶颈分析深入**：通过必要性和充分性的概率框架，精确刻画了建模与推理之间的依赖关系
- **大规模评估**：61 个模型的评估覆盖从 0.6B 到 GPT-5 的完整规模谱，家族内对比（base→SFT→RL）提供了训练策略的系统证据

## 局限与展望

1. 情境变体由 LLM 生成后人工审核，可能存在分布偏差；未来可探索更多样的情境生成方式
2. 未为 MATH-500 构建 CS 变体（部分问题过于简单），覆盖范围有待扩展
3. 专用建模模型训练失败的原因分析不够深入，未探索其他训练策略（如 RL、多任务等）
4. 评估局限于最终答案正确性，未分析中间推理链的质量

## 相关工作与启发

本文与 GSM8K/MATH 等传统数学基准形成互补，将评估从"模型能否解方程"推进到"模型能否从叙事中建立方程"。与 Cheng et al. (2025) 的结论形成有趣对比：后者在简单基准上发现建模不是主要瓶颈，而本文在复杂情境中证明建模是首要瓶颈。这启示我们：LLM 的数学能力评估需要区分"格式化解题"和"理解问题"两个层面。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题定义清晰且重要，SG/CS 双变体设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ — 61 个模型的大规模评估，多维度分析和训练实验
- 写作质量: ⭐⭐⭐⭐⭐ — 结构严谨，分析层层递进，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ — 揭示了 LLM 数学推理中被忽视的核心短板

<!-- RELATED:START -->

## 相关论文

- [Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](doxing_via_the_lens_revealing_location-related_privacy_leakage_on_multi-modal_la.md)
- [AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)

<!-- RELATED:END -->
