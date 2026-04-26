---
title: >-
  [论文解读] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition
description: >-
  [ACL 2026][人体理解][科学发现] 提出 ResearchBench，首个大规模评估LLM科学发现能力的基准，基于"灵感驱动假设生成"的理论分解，覆盖12个学科1386篇论文，将科学发现分解为灵感检索、假设组合、假设排序三个充分子任务，发现LLM在跨学科灵感检索上表现出色。
tags:
  - ACL 2026
  - 人体理解
  - 科学发现
  - 灵感检索
  - 假设生成
  - LLM基准
  - 跨学科
---

# ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition

**会议**: ACL 2026  
**arXiv**: [2503.21248](https://arxiv.org/abs/2503.21248)  
**代码**: 无  
**领域**: 科学发现  
**关键词**: 科学发现, 灵感检索, 假设生成, LLM基准, 跨学科

## 一句话总结

提出 ResearchBench，首个大规模评估LLM科学发现能力的基准，基于"灵感驱动假设生成"的理论分解，覆盖12个学科1386篇论文，将科学发现分解为灵感检索、假设组合、假设排序三个充分子任务，发现LLM在跨学科灵感检索上表现出色。

## 研究背景与动机

**领域现状**：LLM已展现出辅助科学研究的潜力，但其发现有效新假设的能力尚缺乏系统性评估基准。

**现有痛点**：（1）缺乏专门的科学发现基准——现有基准（Chatbot Arena、MixEval）评估通用能力而非发现能力；（2）IdeaBench仅覆盖生物医学的假设生成，不评估完整的发现子任务集；（3）DiscoveryBench和ScienceAgentBench关注特定子任务（如写代码），不分析科学发现的基本分解。

**核心矛盾**：科学发现过程的不可分解性使评估困难——需要一个理论上"充分"的子任务分解，使得完美解决这些子任务等价于完美解决整体发现任务。

**本文目标**：构建首个跨学科、大规模的科学发现能力基准，基于理论上充分的子任务分解。

**切入角度**：基于认知科学发现——创意通常源于两个看似无关知识的组合联想——将假设生成分解为灵感检索→假设组合→假设排序。

**核心idea**：大多数假设 $h = f(b, i_1, ..., i_k)$ 可视为研究背景 $b$ 与灵感知识 $i$ 的组合，据此分解为三个可独立评估的子任务，完美解决这三个子任务即完美解决发现任务。

## 方法详解

### 整体框架

ResearchBench构建流程：（1）从Nature/Science等顶级期刊下载1386篇2024年后论文；（2）用LLM-based agentic框架自动提取研究问题、背景综述、灵感知识和主假设；（3）构建三级负面灵感样本（引用邻近、同学科、跨学科）；（4）在三个子任务上评估LLM：灵感检索（从候选集中选出正确灵感）、假设组合（结合背景和灵感生成假设）、假设排序（对候选假设排序）。

### 关键设计

1. **理论上充分的子任务分解**：

    - 功能：确保子任务评估可推广到整体发现能力
    - 核心思路：基于 $P(h|b) \approx \prod_{j=1}^{k} P(i_j|b,h_{j-1},I) \cdot P(h_j|b,h_{j-1},i_j)$，将发现分解为灵感检索（找到 $i_j$）、假设组合（生成 $h_j$）和排序（选最佳 $h$）。这三个子任务的充分性意味着完美解决它们可完美解决发现
    - 设计动机：认知科学支持——"想法不过是旧元素的新组合"；跨12个学科与专家验证确认普适性

2. **LLM-based灵感提取框架**：

    - 功能：自动从论文中提取研究组件
    - 核心思路：灵感分解模块迭代提取潜在灵感（表示为被引论文的标题和摘要）→必要性检查器验证每个灵感对假设的必要性→充分性检查器确保提取的灵感足以覆盖假设的信息范围。专家验证91.9%准确率
    - 设计动机：自动化框架可随LLM预训练截止日期推移自动更新为更新论文，防止数据泄露

3. **三级负面灵感设计**：

    - 功能：提供精细的灵感检索难度梯度
    - 核心思路：Level 1——被论文引用或语义相似标题的论文（最难区分的近邻）；Level 2——同学科论文（中等难度）；Level 3——完全不同学科论文（最容易排除）
    - 设计动机：简单负面样本无法区分LLM的真实灵感检索能力，三级设计提供更精细的诊断

## 实验关键数据

### 主实验（灵感检索 - 选择top 4%候选）

| 模型 | 总体准确率 |
|------|----------|
| GPT-4o | 45.7% |
| GPT-4o-mini | 42.3% |
| Qwen2.5-72B | ~40% |
| Llama-3.1-70B | ~35% |

### 关键发现
- LLM在灵感检索上表现出人意料地好——选择top 4%候选时真正灵感被包含的概率达45.7%
- 灵感检索本质上是OOD（分布外）任务——灵感应该是"不被认为与研究问题相关但实际有用"的知识，LLM能找到这种非显而易见的关联
- 假设组合和排序任务上LLM也表现不错
- 跨12个学科结果一致，验证了基于灵感的分解框架的普适性
- 将LLM定位为"研究假设矿山"——性能更好的LLM是更富的矿，更多推理计算等于更多矿工

## 亮点与洞察
- **理论基础扎实**：基于认知科学的充分分解，不是ad hoc的评估设计
- **OOD灵感检索的发现意义深远**：说明LLM具备发现非显而易见的知识关联的能力
- **12学科覆盖**：从物理到法学，验证了方法的广泛适用性
- **自动可更新**：框架可随时间自动提取新论文，避免数据泄露

## 局限与展望
- **假设的评估依赖语义匹配**：难以评估真正全新的假设
- **灵感提取准确率91.9%**：仍有改进空间
- **仅评估假设发现**：不评估假设的实验验证
- 未来方向：与实验Agent结合完成完整科学发现循环、评估假设新颖性和影响力

## 相关工作与启发
- **vs IdeaBench**：仅覆盖生物医学、无灵感检索评估、规则提取（非LLM）、单一领域
- **vs DiscoveryBench/ScienceAgentBench**：关注代码编写等特定子任务，不分析发现的基本分解
- **vs MOOSE-Chem**：提出灵感驱动发现框架但仅限化学和材料科学，ResearchBench扩展到12学科

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个基于理论充分分解的跨学科科学发现基准，灵感检索作为OOD任务的洞察独特
- 实验充分度: ⭐⭐⭐⭐ 12学科覆盖、多模型对比、专家验证，但某些任务的评估细节较少
- 写作质量: ⭐⭐⭐⭐ 理论框架阐述清晰，反向传播的灵感例子直观
- 价值: ⭐⭐⭐⭐⭐ 为AI辅助科学发现提供了首个系统性评估框架，"研究假设矿山"的定位有启发性

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] CAP: Controllable Alignment Prompting for Unlearning in LLMs](cap_controllable_alignment_prompting_for_unlearning_in_llms.md)
- [\[ICLR 2026\] AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations](../../ICLR2026/human_understanding/autofigure_generating_and_refining_publication-ready_scientific_illustrations.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ACL 2026\] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)

<!-- RELATED:END -->
