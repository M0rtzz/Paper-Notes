---
title: >-
  [论文解读] Solving Inequality Proofs with Large Language Models
description: >-
  [NeurIPS 2025][LLM/NLP][数学推理] 提出 IneqMath（首个大规模奥林匹克级不等式 benchmark），将不等式证明定义为两个可自动验证的子任务（界估计与关系预测），并开发五模块 LLM-as-Judge 框架，发现即便 o1 在逐步推理审查下整体准确率也不到 10%。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 数学推理
  - 不等式证明
  - LLM-as-Judge
  - benchmark
  - 推理评估
---

# Solving Inequality Proofs with Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.07927](https://arxiv.org/abs/2506.07927)  
**代码**: [IneqMath](https://ineqmath.github.io/)  
**领域**: llm_nlp  
**关键词**: 数学推理, 不等式证明, LLM-as-Judge, benchmark, 推理评估

## 一句话总结

提出 IneqMath（首个大规模奥林匹克级不等式 benchmark），将不等式证明定义为两个可自动验证的子任务（界估计与关系预测），并开发五模块 LLM-as-Judge 框架，发现即便 o1 在逐步推理审查下整体准确率也不到 10%。

## 研究背景与动机

不等式证明是数学推理中的核心挑战，需要发现紧界、战略性应用经典定理（AM-GM、Cauchy-Schwarz 等）以及精确的符号变换。

现有资源不足：

**通用 ATP 数据集**（MiniF2F、ProofNet）：不等式比例极少

**合成数据集**（INT、AIPS）：基于模板生成，缺乏结构多样性

**人工整理集**（ChenNEQ）：规模太小

**格式限制**：多数采用形式化表示（Lean/Isabelle），而 LLM 在自然语言推理上可能更强

核心问题：**仅看最终答案正确率会严重高估 LLM 的数学推理能力**——模型可能猜对答案但推理过程漏洞百出。

## 方法详解

### 整体框架

三大贡献：(1) 非形式化但可验证的任务定义；(2) IneqMath 数据集；(3) 五模块 LLM-as-Judge 评测框架。

### 关键设计一：非形式化可验证任务

将不等式证明分解为两个可自动检查的子任务：

**界估计（Bound Estimation）**：找到使不等式成立的最优常数
$$C^* = \sup\{C \in \mathbb{R} : f(\mathbf{x}) \geq C \cdot g(\mathbf{x}), \forall \mathbf{x} \in \mathcal{D}\}$$

**关系预测（Relation Prediction）**：确定两个表达式间的关系（$>, \geq, =, \leq, <$）

### 关键设计二：IneqMath 数据集

| 统计项 | 数量 | 界估计 | 关系预测 |
|--------|:----:|:-----:|:--------:|
| 测试集 | 200 | 96 | 104 |
| 开发集 | 100 | 50 | 50 |
| 训练集 | 1,252 | 626 | 626 |
| 定理类别 | 29 | - | - |
| 命名定理 | 83 | - | - |
| 带定理标注 | 962 | 482 | 480 |
| 每题最多解法数 | 4 | 4 | 4 |

测试集由 IMO 奖牌得主原创设计，经独立专家组验证。训练集包含逐步解法和定理标注。

### 关键设计三：五模块 LLM-as-Judge 框架

仅当解答通过全部五个 Judge 时才被视为正确：

1. **Final Answer Judge**：两阶段验证——提取结论句 + 数学等价性检查（处理 $\frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$），F1 = 1.00
2. **Toy Case Judge**：检测仅通过特定数值验证就做出一般性结论的错误，F1 = 0.91
3. **Logical Gap Judge**：标记缺失推理步骤和无支撑断言，F1 = 0.96
4. **Numerical Approximation Judge**：标记不当使用数值近似，F1 = 0.96
5. **Numerical Computation Judge**：验证算术步骤正确性（Python/SymPy），F1 = 0.80

整体 F1 = 0.93。

### 损失函数 / 训练策略

本文是评测框架，不涉及模型训练。训练集的逐步解法和定理标注可用于后续研究。

## 实验关键数据

### 主实验

对 29 个领先 LLM 的零样本评估：

| 模型 | 答案准确率 | 整体准确率 | 下降幅度 |
|------|:---------:|:---------:|:-------:|
| o1 | ~73.5% | **8.0%** | **-65.5%** |
| o3 | ~65% | ~31% | -34% |
| Qwen2.5-72B | 42.0% | 2.5% | -39.5% |
| GPT-4o | ~40% | ~2% | -38% |
| Llama-4-Maverick | 40.5% | 2.5% | -38% |

### 消融实验

**模型规模**：更大模型提升最终答案准确率，但对整体准确率帮助有限

**测试时计算量**：o1 的整体准确率在 5K 到 40K max tokens 中保持 8.0% 不变；o3 饱和在约 31%

**改进策略**：
- 定理引导：o3-mini 整体准确率提升 **+11%**
- 批评者引导自修正：Gemini 2.5 Pro 提升 **+5%**

### 关键发现

1. **答案正确不等于推理正确**：最高 65.5% 的准确率下降
2. **扩展不是万能药**：模型规模和推理时间对步骤正确性改善极有限
3. **LLM 推理缺陷分类**：逻辑跳跃 > 玩具案例泛化 > 数值近似滥用 > 计算错误
4. **定理知识是关键**：提供正确定理显著提升推理质量

## 亮点与洞察

1. **揭示评估盲区**：首次系统性展示步骤审查带来的巨大准确率下降
2. **模块化 Judge 设计**优于通用 Judge
3. **数据集质量极高**：IMO 奖牌得主原创 + 独立专家验证 + 多解法标注
4. **指明改进方向**：定理引导推理和自精炼，而非简单扩展规模
5. **评估范围全面**：29 个模型覆盖开源和闭源

## 局限与展望

1. **LLM-as-Judge 不完美**：数值计算 Judge F1 仅 0.80
2. **测试集规模有限**：200 题统计显著性有限
3. **仅零样本评测**：未探索 few-shot 或微调后表现
4. **不等式类型偏向竞赛**：对分析学/概率论中的实际不等式覆盖不足
5. 可探索形式化与非形式化互验的混合方案

## 相关工作与启发

- **MiniF2F**（Zheng et al., 2022）：形式化数学基准
- **MATH/GSM8K**：传统数学基准，侧重最终答案
- **LLM-as-Judge**（Zheng et al., 2024）：通用评判框架

## 评分

⭐⭐⭐⭐⭐ (4.5/5)

- **创新性** ⭐⭐⭐⭐⭐：任务定义 + 评估框架 + 数据集三位一体
- **实验充分度** ⭐⭐⭐⭐⭐：29 个模型系统评估
- **影响力** ⭐⭐⭐⭐⭐：揭示 LLM 数学推理根本缺陷
- **写作质量** ⭐⭐⭐⭐⭐：逻辑清晰，图表精美

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](../../ACL2025/llm_nlp/perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](scaling_up_active_testing_to_large_language_models.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[NeurIPS 2025\] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)

<!-- RELATED:END -->
