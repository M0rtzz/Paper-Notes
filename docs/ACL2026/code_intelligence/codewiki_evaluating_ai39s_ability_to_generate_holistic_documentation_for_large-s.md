---
title: >-
  [论文解读] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases
description: >-
  [ACL 2026][代码文档生成] 提出 CodeWiki，一个基于层次化分解和递归多智能体处理的开源框架，用于自动生成仓库级代码文档，并构建了 CodeWikiBench 基准，在七种编程语言上以 68.79% 的质量分数超越了闭源系统 DeepWiki（64.06%）。
tags:
  - ACL 2026
  - 代码文档生成
  - 仓库级理解
  - 多智能体系统
  - 层次化分解
  - 代码基准
---

# CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases

**会议**: ACL 2026  
**arXiv**: [2510.24428](https://arxiv.org/abs/2510.24428)  
**代码**: [GitHub](https://github.com/FSoft-AI4Code/CodeWiki)  
**领域**: 代码智能  
**关键词**: 代码文档生成, 仓库级理解, 多智能体系统, 层次化分解, 代码基准

## 一句话总结

提出 CodeWiki，一个基于层次化分解和递归多智能体处理的开源框架，用于自动生成仓库级代码文档，并构建了 CodeWikiBench 基准，在七种编程语言上以 68.79% 的质量分数超越了闭源系统 DeepWiki（64.06%）。

## 研究背景与动机

**领域现状**：随着代码库规模和复杂度不断增长，维护全面且及时的文档已成为软件开发中的核心瓶颈。约 31% 的开发者已大量使用 AI 来辅助代码文档化，这反映出自动化文档生成的迫切需求。

**现有痛点**：现有方法主要集中在函数级和文件级文档生成（如 CodeBERT、DocAgent 等），难以扩展到仓库级别。仓库级文档需要捕获架构模式、跨模块交互、数据流和系统级设计决策，但现有工具缺乏对这些语义依赖和层次结构的建模能力。此外，评估体系也存在不足——传统的 BLEU/ROUGE 指标无法捕捉文档质量的多维度特征，且缺乏针对仓库级文档的系统性基准。

**核心矛盾**：仓库级文档生成需要同时理解局部实现细节和全局架构关系，但 LLM 的上下文窗口有限，无法一次性处理大型代码库；现有多语言支持也严重不足，大多数研究仅关注 Python。

**本文目标**：构建一个可扩展的、支持多语言的仓库级文档自动生成框架，同时提供可靠的评估方法论。

**切入角度**：借鉴动态规划思想，通过层次化分解将大型仓库拆分为可管理的模块，然后递归地自底向上生成并合成文档。

**核心idea**：将仓库级文档生成分为三阶段——静态分析与模块分解、递归智能体文档生成、层次化组装与合成——通过动态委托机制实现对任意规模仓库的自适应处理。

## 方法详解

### 整体框架

CodeWiki 框架分为三个主要阶段：（1）仓库分析阶段——通过 AST/LLM 解析构建依赖图并识别高层组件，然后进行层次化模块分解；（2）递归文档生成阶段——为每个叶子模块分配专用智能体，智能体具备源码访问、模块树浏览、文档工作空间操作和依赖图遍历能力，当模块复杂度超过单次处理能力时可动态委托给子智能体；（3）层次化组装阶段——自底向上合成子模块文档为父模块的架构概述，最终生成包含架构图、数据流可视化等多模态的综合文档。

### 关键设计

1. **层次化模块分解（Hierarchical Module Decomposition）**：

    - 功能：将大型仓库拆分为可管理的模块单元
    - 核心思路：利用 Tree-Sitter 解析器提取 AST，构建有向依赖图 $G=(V,E)$，通过拓扑排序识别零入度的入口组件（如 main 函数、API 端点），然后递归分区为模块树。为保证可扩展性，模块树仅使用组件 ID 作为输入
    - 设计动机：借鉴动态规划的"分治"思想，解决 LLM 上下文窗口有限而仓库规模巨大的矛盾，同时保持架构一致性

2. **动态委托递归智能体（Dynamic Delegation Recursive Agents）**：

    - 功能：自适应地处理不同复杂度的模块
    - 核心思路：每个叶子模块的专用智能体在处理时，根据代码复杂度指标（圈复杂度、嵌套深度）、语义多样性和上下文窗口利用率判断是否需要委托。当模块复杂度超过阈值时，智能体将子模块委托给新的子智能体递归处理
    - 设计动机：确保框架能够处理任意规模的仓库，同时维持每个模块的文档质量，实现有界复杂度和架构一致性

3. **跨模块引用管理与层次化合成（Cross-Module Reference & Hierarchical Synthesis）**：

    - 功能：维护跨模块的文档一致性并生成全局架构文档
    - 核心思路：通过全局注册表追踪已记录的组件和位置，遇到外部组件时创建交叉引用而非重复内容。父模块的合成通过多阶段 LLM 处理：分析子文档主题模式→生成架构概览→创建特性摘要→开发使用指南→生成可视化图表
    - 设计动机：避免内容冗余，生成互联的文档体系，使文档真实反映代码库的实际结构和交互关系

### 评估框架 CodeWikiBench

CodeWikiBench 的核心创新在于层次化评估标准（Hierarchical Rubric）的设计：从开源项目的官方文档中自动生成评估标准，以层次化结构镜像项目架构。评估过程由多个 Judge Agent（使用不同模型家族）独立评判叶子级需求，然后通过加权聚合自底向上计算最终分数和可靠性指标。多模型共识机制有效降低了单模型偏差。

## 实验关键数据

### 主实验

| 仓库 | 语言 | LOC | CodeWiki | DeepWiki | 提升 |
|------|------|-----|----------|----------|------|
| OpenHands | Python | 229K | 82.45% | 73.04% | +9.41% |
| svelte | JavaScript | 125K | 71.96% | 68.51% | +3.45% |
| puppeteer | TypeScript | 136K | 83.00% | 64.46% | +18.54% |
| ml-agents | C# | 86K | 79.78% | 74.80% | +4.98% |
| logstash | Java | 117K | 57.90% | 54.80% | +3.10% |
| wazuh | C | 1.4M | 64.17% | 68.68% | -4.51% |
| electron | C++ | 184K | 42.30% | 44.10% | -1.80% |
| **平均** | | | **68.79%** | **64.06%** | **+4.73%** |

### 跨语言分析

| 语言类别 | CodeWiki | DeepWiki | 提升 |
|---------|----------|----------|------|
| 脚本语言 (Python/JS/TS) | 79.14% | 68.67% | +10.47% |
| 托管语言 (C#/Java) | 68.84% | 64.80% | +4.04% |
| 系统语言 (C/C++) | 53.24% | 56.39% | -3.15% |

### 关键发现
- CodeWiki 在 7 个仓库中的 5 个上超越所有基线，在 TypeScript 仓库上获得最大提升（+18.54%）
- 高级脚本语言上的优势最为显著（+10.47%），但在系统编程语言（C/C++）上略逊于 DeepWiki
- 性能差异主要归因于语言特征而非仓库规模
- 初步人类研究显示 CodeWiki 在 9 次评估中的 7 次被偏好

## 亮点与洞察
- **层次化分解思路精妙**：将动态规划思想应用于文档生成，既解决了规模可扩展性问题，又保持了架构语义的一致性
- **评估方法论创新**：CodeWikiBench 的层次化标准生成和多模型共识评判机制为仓库级文档评估提供了系统性解决方案
- **开源透明性**：在闭源系统占主导的背景下，CodeWiki 的开源释放具有重要社区价值

## 局限与展望
- **系统编程语言表现不佳**：C/C++ 上低于 DeepWiki，指针操作和模板元编程等底层构造的解析能力不足
- **评估标准未经充分人类验证**：语义可靠性 73.65%、结构可靠性 70.84%
- **人类评估规模有限**：仅 3 名参与者 × 3 个仓库
- 未来方向：针对系统语言开发专用解析模块、多版本文档追踪、利用文档支持下游任务

## 相关工作与启发
- **vs DocAgent**：DocAgent 使用多智能体协作进行函数级文档生成，而 CodeWiki 聚焦于仓库级层次化合成
- **vs DeepWiki**：闭源商用系统，整体表现不错但缺乏层次化分解能力
- **vs OpenDeepWiki/deepwiki-open**：开源替代方案采用整仓库直接提示方式，性能明显落后

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化分解+动态委托的设计新颖，CodeWikiBench 评估方法论有创新
- 实验充分度: ⭐⭐⭐⭐ 覆盖 7 种语言和 7 个仓库，有跨语言和可扩展性分析，人类评估规模偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，三个研究问题组织得当
- 价值: ⭐⭐⭐⭐ 填补仓库级文档自动生成和评估的重要空白，开源对社区有积极影响

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)

<!-- RELATED:END -->
