---
title: >-
  [论文解读] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs
description: >-
  [ICML2025][符号回归] 提出LLM-SRBench基准（239题/4个科学领域），通过方程变换(LSR-Transform)和合成问题(LSR-Synth)防止LLM的记忆化，当前最好方法仅达31.5%符号准确率。
tags:
  - ICML2025
  - 符号回归
  - 科学方程发现
  - LLM基准
  - 记忆化防止
  - 方程变换
---

# LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs

**会议**: ICML2025  
**arXiv**: [2504.10415](https://arxiv.org/abs/2504.10415)  
**代码**: [GitHub - LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench)  
**领域**: llm_nlp  
**关键词**: 符号回归, 科学方程发现, LLM基准, 记忆化防止, 方程变换

## 一句话总结
提出LLM-SRBench基准（239题/4个科学领域），通过方程变换(LSR-Transform)和合成问题(LSR-Synth)防止LLM的记忆化，当前最好方法仅达31.5%符号准确率。

## 研究背景与动机

### LLM在方程发现中的机会与风险
LLM嵌入了大量科学知识可辅助假设生成。但现有基准（Feynman/SRBench）使用教科书方程，LLM可能直接"背诵"而非"发现"。

### 记忆化的证据
实验显示LLM在Feynman问题上的数值误差曲线先急剧下降（记忆化信号），而非渐进改善（真正发现信号）。

### 基准设计目标
设计防记忆化的测试集，同时利用LLM的科学先验知识，模拟真实发现场景。

## 方法详解

### LSR-Transform（变换类）
将已知物理方程变换为不常见的数学表示：
- 符号改变输入输出映射
- 生成同一物理问题的罕见数学形式
- 挑战LLM超越记忆的推理能力

### LSR-Synth（合成类）
引入合成的发现驱动问题：
- 需要数据驱动推理
- 不依赖已知方程
- 测试真正的发现能力

### 评估体系
- 数据保真度（数值拟合）
- 符号准确率（精确匹配）
- 计算效率

## 实验关键数据

### 当前方法表现

| 方法 | Feynman准确率 | LSR-Transform | LSR-Synth |
|------|-------------|-------------|----------|
| Llama-3.1-8B 直接采样 | 高(记忆化) | 低 | 低 |
| 最佳LLM方法 | ~80% | ~35% | ~28% |
| **整体最佳** | - | - | **31.5%** |

### 记忆化 vs 发现的区分

| 特征 | Feynman(旧) | LLM-SRBench(新) |
|------|-----------|---------------|
| 数值误差曲线 | 急剧下降 | 渐进改善 |
| 符号误差 | 极低 | 显著高 |
| 是否记忆化 | 很可能 | 不太可能 |

### 关键发现
1. 当前最佳方法在LLM-SRBench上仅31.5%——远低于Feynman
2. 开源LLM在LSR-Transform上表现比Synth好（仍有部分先验可用）
3. 闭源LLM(GPT-4)略优但差距有限
4. 方程变换有效防止了简单记忆化

## 亮点与洞察

1. 精准戳破了LLM在方程发现上的虚假繁荣——大部分成功来自记忆化。
2. LSR-Transform的设计巧妙：同一物理问题的不同数学形式。
3. 239题/4领域的规模远超之前的5题自定义集。
4. 数值误差曲线分析是区分记忆vs发现的简洁工具。
5. 为LLM+SR社区提供了标准化评测平台。

## 局限与展望

1. 31.5%的上限说明问题极其困难，也可能是基准过难。
2. 变换方法可能引入不自然的数学形式。
3. 仅在4个科学领域验证。
4. 人类科学家的基线对比缺失。
5. 多步推理和工具使用的系统评测不够。

## 相关工作与启发

- 与SRBench/SRSD的关系：LLM-SRBench是其LLM-aware版本。
- 与PAN+SR的互补：PAN解决高维，LLM-SRBench解决记忆化。
- 启发：其他LLM评估也应考虑记忆化问题。

## 评分
- 新颖性: 5.0/5 — 首个防记忆化的LLM方程发现基准
- 实验充分度: 4.5/5 — 多方法多LLM
- 写作质量: 4.5/5
- 价值: 5.0/5 — 对LLM评估方法论有重要贡献

## 补充分析

### 记忆化检测指标
数值误差曲线的形状是关键区分指标：急剧下降=记忆化，渐进改善=真正发现。这个分析工具本身就是重要贡献。

### 变换类(LSR-Transform)的设计细节
将已知方程的输入输出映射做符号变换，生成同一物理问题的不常见数学形式。例如将F=ma变换为对数形式或积分形式。

### 与SRBench的互补性
SRBench评估传统符号回归方法，LLM-SRBench专门评估LLM方法——两者互补。

<!-- RELATED:START -->

## 相关论文

- [YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](../../ACL2025/llm_evaluation/yescieval_llm_judge_science.md)
- [MisMatched: A Benchmark for Scientific Natural Language Inference](../../ACL2025/llm_evaluation/a_mismatched_benchmark_for_scientific_natural_language_inference.md)
- [RGB-to-Polarization Estimation: A New Task and Benchmark Study](../../NeurIPS2025/llm_evaluation/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [HalluLens: LLM Hallucination Benchmark](../../ACL2025/llm_evaluation/hallulens_llm_hallucination_benchmark.md)
- [On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

<!-- RELATED:END -->
