---
title: >-
  [论文解读] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs
description: >-
  [ACL 2025][LaTeX生成] 提出TeXpert——首个系统评估LLM从自然语言指令生成科学文档LaTeX代码能力的多难度级别基准，包含440个高质量样本（Simple/Average/Hard三级），在9个开闭源LLM上的评估揭示了LaTeX生成是LLM的显著短板（Hard任务准确率普遍低于17.5%），逻辑错误和格式错误是主要瓶颈。
tags:
  - ACL 2025
  - LaTeX生成
  - LLM评估
  - benchmark
  - 代码生成
  - 科学文档
---

# TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs

**会议**: ACL 2025  
**arXiv**: [2506.16990](https://arxiv.org/abs/2506.16990)  
**代码**: https://github.com/knowledge-verse-ai/TeXpert  
**领域**: NLP Generation / 代码生成评估  
**关键词**: LaTeX生成, LLM评估, benchmark, 代码生成, 科学文档

## 一句话总结
提出TeXpert——首个系统评估LLM从自然语言指令生成科学文档LaTeX代码能力的多难度级别基准，包含440个高质量样本（Simple/Average/Hard三级），在9个开闭源LLM上的评估揭示了LaTeX生成是LLM的显著短板（Hard任务准确率普遍低于17.5%），逻辑错误和格式错误是主要瓶颈。

## 研究背景与动机
LaTeX是科学文档排版的事实标准，处理数学公式、引用管理等方面具有不可替代的优势。随着LLM在指令跟随能力上的提升，一个自然的问题是：**LLM能否从自然语言描述直接生成出版级别的LaTeX代码？**

**领域空白**：
- 现有工作仅覆盖LaTeX的狭窄子任务：MATH数据集评估数学公式生成、STRUC-BENCH仅测试表格、im2latex-100k关注图片→公式转换
- 没有任何基准系统评估LLM生成科学文档中各类组件（表格、图片、参考文献、文档结构等）的LaTeX代码能力
- LaTeX模板通常太大无法直接生成，且局限于基础命令

**两个实际价值**：
1. 分析LLM生成LaTeX代码的常见错误模式，可为检测AI生成学术材料提供信号
2. 明确LLM能可靠完成的LaTeX任务复杂度边界，帮助研究者减少排版工作

## 方法详解

### 整体框架
TeXpert的构建遵循两步流程：先收集科学文档中常用的原子LaTeX命令，再基于这些命令按三个难度级别生成自然语言→LaTeX的任务样本。评估时使用GPT-4o作为judge，对LLM生成的代码判断正确性并归类错误类型。

### 关键设计

1. **原子命令收集与分类**：

    - 从Overleaf文档、学术期刊LaTeX模板（各25个文档）中提取原子命令
    - 使用正则表达式自动提取 + 人工验证
    - 319个命令分为5类：文本格式(86)、公式符号(83)、文档结构(75)、引用参考(39)、表格图片(36)
    - 设计动机：通过原子命令限定数据集范围，确保聚焦于科学文档写作中的真实需求

2. **三级难度设计**：

    - **Simple（250样本）**：平均115.8字符指令，约10.9个原子命令，0.5个额外包
    - **Average（150样本）**：平均299.1字符指令，约51.2个原子命令，3.6个额外包
    - **Hard（40样本）**：平均558.4字符指令，约85.9个原子命令，6.6个额外包
    - 每个样本经过人工验证，确保LaTeX生成需求清晰且与难度约束一致
    - 设计动机：递增的复杂度可以精确定位LLM的能力边界

3. **评估框架**：

    - 生成：给LLM提供包含任务指令的prompt，设固定参数保证确定性输出
    - 评判：GPT-4o作为judge（对GPT-4o/mini使用DeepSeek v3评判避免偏差）
    - 5类错误分类：CE（能力错误）、SE（语法错误）、LE（逻辑错误）、PE（包错误）、FE（格式错误）
    - Hard集额外提供人工编写验证的参考LaTeX代码辅助评估

### 损失函数 / 训练策略
本文不涉及模型训练，是纯评估基准工作。评估prompt经过迭代优化，通过人工抽检确保评判的可靠性和一致性。

## 实验关键数据

### 主实验（各模型在三个难度级别的准确率）
| 模型 | Simple | Average | Hard | Overall |
|------|--------|---------|------|---------|
| GPT-4o | **78.8** | **58.7** | 15.0 | **66.1** |
| DeepSeek V3 | 71.2 | 58.7 | 10.0 | 61.4 |
| DeepSeek Coder 33b | 69.2 | 58.0 | **17.5** | 60.7 |
| Mistral Large | 64.4 | 59.3 | 10.0 | 57.7 |
| Claude 3.5 Sonnet | 62.8 | 56.7 | 0 | 55.0 |
| Grok 2-1212 | 62.4 | 52.0 | 5.0 | 53.6 |
| GPT-4o-mini | 62.4 | 45.3 | 5.0 | 51.4 |
| Codestral 22B | 60.8 | 41.3 | 0 | 48.6 |
| Gemini 1.5 Flash | 53.6 | 33.3 | 0 | 41.8 |

### 消融实验（错误类型分布）
| 模型 | 能力错误% | 语法错误% | 逻辑错误% | 包错误% | 格式错误% |
|------|----------|----------|----------|---------|----------|
| GPT-4o | 0.0 | 2.1 | **59.1** | 15.2 | 23.6 |
| Claude 3.5 Sonnet | 0.0 | 5.3 | 44.3 | **29.9** | 20.6 |
| DeepSeek V3 | 1.2 | 3.8 | 54.3 | 18.7 | 22.0 |
| Codestral 22B | 0.6 | 2.8 | 52.5 | 18.8 | **25.3** |

### 关键发现
- **Hard任务是普遍痛点**：即使是GPT-4o也只有15%准确率，Claude和Gemini完全失败（0%），存在明确的能力悬崖
- **逻辑错误占主导**（44-59%）：主要表现为遗漏指令要求和错误的结构放置，尤其在多公式/多表格任务中
- **格式错误出乎意料地高**（20-25%）：即使大模型也在环境选择和表格/图注格式上频繁出错，说明规模不是解决方案
- **包错误令人担忧**：不正确/不完整的包引入，尤其是参考文献相关包，Claude最严重（29.9%）
- **开源模型与闭源竞争力强**：DeepSeek V3和Coder 33b在整体准确率上接近甚至在Hard集上超越GPT-4o

## 亮点与洞察
- **填补空白**：LaTeX代码生成评估领域第一个系统性基准，与MATH/STRUC-BENCH等单一维度工作形成互补
- **能力边界的量化**：通过三级难度设计，精确刻画了LLM在LaTeX生成上的能力天花板——平均难度约50-60个原子命令、3-4个包是当前极限
- **训练数据瓶颈假说**：格式错误和包错误的普遍性强烈暗示LLM训练数据中缺乏多样化的LaTeX示例
- **DeepSeek Coder在Hard集上的领先**表明专门的代码训练对结构化输出生成有独特优势
- **评估设计的务实性**：使用LLM-as-Judge + 人工参考代码的混合评估方式，在成本和准确性之间取得了平衡

## 局限与展望
- Hard集仅40个样本，统计可靠性有限
- 仅覆盖科学文档，未涉及简历、书籍等其他文档类型
- 未尝试使用TeXpert数据集对模型进行微调
- 未探索高级prompt策略（如few-shot、self-debug）对性能的影响
- LLM-as-Judge本身的偏差未充分讨论
- 未分析不同LaTeX命令类别（公式 vs 表格 vs 引用）的分项准确率

## 相关工作与启发
- **MATH (Hendrycks et al., 2021)**, **MathBridge**, **STEM-POM** 聚焦数学LaTeX生成，TeXpert将评估扩展到完整的科学文档排版
- **STRUC-BENCH (Tang et al., 2024)** 测试表格生成，TeXpert覆盖全部5类组件更为全面
- **对实际工作流的启示**：当前LLM最多只能可靠处理Simple级别的LaTeX任务（约60-78%准确率），研究者使用LLM辅助排版时应限于简单组件的生成，复杂布局仍需手工处理
- **潜在应用**：TeXpert的错误模式分析可以为AI生成学术材料的检测提供特征

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性LaTeX生成基准，问题重要且此前空白
- 实验充分度: ⭐⭐⭐⭐ 9个模型、3级难度、5类错误分析，相当全面
- 写作质量: ⭐⭐⭐ 结构清晰但部分分析略浅，Hard集统计讨论不足
- 价值: ⭐⭐⭐⭐ 对LLM辅助学术写作有直接指导意义，数据集公开促进社区研究

<!-- RELATED:START -->

## 相关论文

- [FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)
- [DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)
- [GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)

<!-- RELATED:END -->
