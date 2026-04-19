---
title: >-
  [论文解读] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming
description: >-
  [ACL 2025] 提出首个全面评估**人类-LLM协作竞赛编程**的基准ELABORATION，通过覆盖编程全流程的人类反馈分类体系和8320题精标注数据集，揭示LLM独立解题能力有限（困难题仅3.4% Pass@1），但人类反馈（尤其编码阶段的专家反馈）可带来平均9.3%的显著提升。
tags:
  - ACL 2025
---

# ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming

**会议**: ACL 2025  
**arXiv**: [2505.16667](https://arxiv.org/abs/2505.16667)  
**代码**: [有](https://github.com/SCUNLP/ELABORATION)  
**领域**: NLP / LLM评估 / 代码生成  
**机构**: 四川大学、天津科技大学、Vanderbilt University
**作者**: Xinwei Yang, Zhaofeng Liu, Chen Huang, Jiashuai Zhang, Tong Zhang, Yifan Zhang, Wenqiang Lei

## 一句话总结

提出首个全面评估**人类-LLM协作竞赛编程**的基准ELABORATION，通过覆盖编程全流程的人类反馈分类体系和8320题精标注数据集，揭示LLM独立解题能力有限（困难题仅3.4% Pass@1），但人类反馈（尤其编码阶段的专家反馈）可带来平均9.3%的显著提升。

## 背景与动机

竞赛编程要求掌握**问题理解→方案规划→代码生成→调试**四个阶段，对精确性和效率要求极高。虽然LLM在代码生成方面取得了进展，但在专家级竞赛题上的实用性仍然有限。近年来研究转向**人类-LLM协作编程**（human-in-the-loop），通过多轮人类反馈增强LLM效能。

然而现有研究存在两个核心问题：

1. **反馈类型碎片化**：不同研究仅关注局部阶段（如Mozannar等只关注策略建议，Zheng等只关注对话式错误识别），缺乏对编程全流程的系统覆盖
2. **缺乏标准化评估**：现有基准大多面向纯自动编程，不支持人类-LLM协作场景的系统评估

因此需要一个覆盖编程全流程、支持多粒度人类反馈的综合性基准。

## 方法详解

### 整体框架

ELABORATION基准包含三个核心组件：

1. **人类反馈分类体系**：首次将编程全流程的人类反馈系统化为四阶段分类
2. **Elaborationset数据集**：8320道竞赛编程题，附带精标注以支持模拟和真人反馈实验
3. **评估协议**：支持LLM模拟器和真人参与者两种模式，在每个阶段注入反馈并评估效果

评估流程中，LLM与人类（真人或模拟器）迭代交互，人类在每个阶段提供文本反馈，直到生成正确解答或达到最大迭代次数（10轮）。

### 关键设计1：四阶段人类反馈分类体系

将竞赛编程全流程划分为四个阶段，每阶段定义具体的人类反馈形式：

| 阶段 | 人类反馈内容 | 示例 |
|------|-------------|------|
| **问题理解** | 提供关键需求和规格说明 | 指出边界情况、明确功能需求、强调时间复杂度约束 |
| **方案规划** | 建议算法、提供伪代码 | 对最短路径问题建议Dijkstra算法并给出伪代码 |
| **代码生成** | 建议解决策略和实现细节 | 建议使用栈结构、二叉堆优先队列等具体实现 |
| **代码调试** | 识别错误直到通过所有测试 | 定位导致无限循环的逻辑缺陷 |

该分类体系的独特价值在于：现有方法大多局限于调试阶段，而本工作证明了**全流程反馈**的必要性。

### 关键设计2：Elaborationset精标注数据集

数据集来源于Codeforces和AtCoder（2011.10~2024.11），包含三类难度共8320题。核心标注包括：

- **题目澄清说明**（平均8.1~12.1条/题）：每道题的需求和规格标注
- **算法知识摘要**（平均2.4~3.8条/题）：所需算法定义和伪代码
- **标准解**（平均4.8个/题）：从平台直接获取
- **真人交互记录**（300题）：含多轮对话和人工标注的LLM代码错误

标注流程采用"LLM初始生成 + 人工审核"两阶段策略，标准解直接来源于竞赛平台。数据集按日期划分以支持无污染评估。

### 关键设计3：双模式人类模拟器

设计两个水平的LLM模拟参与者（均基于O1-Mini）：

- **学生程序员**（中级）：仅凭模型内部知识提供反馈，模拟一般编程者
- **教师程序员**（专家级）：利用完整数据集标注信息，确保专家级反馈质量

以八皇后问题为例：教师反馈会详细指定占位值、迭代放置策略和回溯条件，而学生反馈可能遗漏这些关键细节。

## 实验结果

### 表1：闭源模型在无污染评估上的Pass@1（%）

| 模型 | Easy | Middle | Hard | Overall |
|------|------|--------|------|---------|
| O1-Mini | 80.6 | 66.6 | 30.8 | 59.3 |
| GPT-4o | 74.1 | 31.7 | 10.3 | 38.7 |
| + 学生反馈 | 76.2 | 34.8 | 15.1 | 42.0 |
| + 教师反馈 | 80.1 | 42.9 | 23.3 | 48.8 |
| Claude-3.5 | 74.5 | 34.3 | 5.4 | 38.1 |
| + 教师反馈 | 83.1 | 44.2 | 16.5 | 47.9 |
| **闭源平均** | **71.8** | **31.5** | **7.7** | **37.0** |
| + 教师反馈 | 79.9(+8.1) | 41.8(+10.3) | 19.6(+11.9) | 47.1(+10.1) |

### 表2：真人调试实验对比（GPT-4 Turbo，300题无污染）

| 调试方式 | Precision | Recall | 原始P@1 | +调试P@1 |
|---------|-----------|--------|---------|---------|
| 自动调试（编译器+O1-Mini模拟） | 0.23 | 0.40 | 0.33 | 0.38(+5%) |
| 人类调试（5名CS研究生） | 0.81 | 0.71 | 0.38 | 0.62(+24%) |

## 亮点

- **系统性最强**：首次构建覆盖编程全流程四阶段的人类反馈分类体系，弥补了现有研究只关注局部阶段的碎片化问题
- **发现编码阶段反馈收益最大**：细粒度分析表明理解阶段反馈效果最小（LLM已能较好理解题意），编码阶段反馈贡献最大，打破了"调试最重要"的直觉假设
- **真人实验揭示人机互补性**：自动调试擅长语法错误，人类擅长语义错误（引用、计算、逻辑缺陷），两者结合效果最佳
- **数据集设计精巧**：通过多层标注支持不同水平的反馈模拟，无污染划分确保评估可靠性

## 局限性

- **提示敏感性**：与所有LLM提示研究一样，结果对prompt设计敏感，尽管在8000+题上取平均可缓解
- **泛化性有限**：仅在竞赛编程场景验证，能否推广到实际软件开发尚待验证
- **模拟器与真人的差距**：LLM模拟器虽然比规则模拟器更真实，但与真实人类反馈仍存在质量差距
- **成本效率**：编码阶段反馈虽效果最好，但token开销也最大；规划阶段可能性价比更优

## 相关工作

- **竞赛编程基准**：APPS、CODE-CONTESTS、LiveCodeBench等面向纯自动编程，ELABORATION首次专门支持人类-LLM协作评估
- **人类-LLM编程**：OpenCoderInterpreter（Zheng 2024）最接近但仅覆盖调试阶段，本工作实现全流程覆盖
- **人类反馈模拟**：从规则模拟器发展到LLM模拟器，本工作同时使用模拟器和真人参与者
- **代码LLM**：CodeLlama、DeepSeek-Coder、Qwen2.5-Coder等作为被评估对象

## 评分

⭐⭐⭐⭐

**理由**：问题定义清晰，分类体系系统完整，数据集规模大且标注精细，实验设计兼顾模拟和真人两个维度。发现"编码阶段反馈最有效"和"人机互补bug检测"两个核心洞见具有实践指导意义。局限在于仅针对竞赛编程，方法层面创新性相对有限（主要贡献在benchmark构建而非新方法）。

<!-- RELATED:START -->

## 相关论文

- [MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [Navigating Rifts in Human-LLM Grounding: Study and Benchmark](navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)
- [CFBench: A Comprehensive Constraints-Following Benchmark for LLMs](cfbench_a_comprehensive_constraints-following_benchmark_for_llms.md)
- [HalluLens: LLM Hallucination Benchmark](hallulens_llm_hallucination_benchmark.md)

<!-- RELATED:END -->
