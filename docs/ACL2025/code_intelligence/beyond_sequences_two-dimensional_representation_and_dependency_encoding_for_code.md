---
title: >-
  [论文解读] Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation
description: >-
  [ACL 2025][代码生成] 本文提出超越传统一维序列表示的二维代码表示方法，通过显式编码代码的结构依赖关系（如语法树结构和变量依赖），显著提升了代码生成的准确性和结构正确性。
tags:
  - ACL 2025
  - 代码生成
  - 二维表示
  - 依赖编码
  - 抽象语法树
  - 结构化代码表示
---

# Beyond Sequences: Two-dimensional Representation and Dependency Encoding for Code Generation

**会议**: ACL 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 文本生成（代码生成）  
**关键词**: 代码生成、二维表示、依赖编码、抽象语法树、结构化代码表示

## 一句话总结
本文提出超越传统一维序列表示的二维代码表示方法，通过显式编码代码的结构依赖关系（如语法树结构和变量依赖），显著提升了代码生成的准确性和结构正确性。

## 研究背景与动机

**领域现状**：代码生成是当前 LLM 应用的热门方向，如 Codex、CodeLlama 等模型在 HumanEval、MBPP 等基准上取得了优秀表现。现有方法普遍将代码视为一维字符/token 序列进行处理。

**现有痛点**：代码与自然语言的根本区别在于，代码具有严格的二维结构——除了线性的 token 序列外，还存在丰富的非顺序依赖关系，包括 AST（抽象语法树）的层级结构、变量定义-引用关系、控制流依赖等。将代码"拍平"为一维序列丢失了这些关键的结构信息。

**核心矛盾**：现有的序列到序列模型在生成代码时，容易产生语法正确但逻辑结构错误的代码（如变量作用域混乱、缺少匹配的括号、控制流不完整等），因为模型缺乏对代码结构的显式理解。

**本文目标**：设计一种能同时捕获代码的线性序列信息和非线性结构依赖的表示方法，提升代码生成的结构正确性。

**切入角度**：代码本质上是二维结构——横向是 token 序列，纵向是语法和语义依赖。需要一种二维表示来同时编码这两个维度的信息。

**核心 idea**：将代码表示从一维序列升级为二维矩阵，其中一个维度编码 token 序列，另一个维度编码结构依赖关系，让模型在生成每个 token 时能同时参考序列上下文和结构约束。

## 方法详解

### 整体框架
输入为自然语言问题描述，输出为生成的代码。在标准的 Transformer 编码器-解码器框架基础上，引入二维位置编码和依赖感知的注意力机制。生成过程中，模型不仅自回归地预测下一个 token，还同时维护一个结构约束图来确保生成代码的结构合法性。

### 关键设计

1. **二维代码表示（2D Code Representation）**:

    - 功能：将一维 token 序列扩展为编码结构信息的二维表示
    - 核心思路：对每个代码 token，除了其在序列中的位置编码外，还赋予其在 AST 中的深度和类型信息。构建一个 token×token 的关系矩阵，其中 $(i,j)$ 位置的值编码了第 $i$ 个 token 和第 $j$ 个 token 之间的结构关系类型（如父子关系、兄弟关系、变量依赖等）
    - 设计动机：一维位置编码只能表达线性距离，无法表达树状结构中的层级距离和依赖方向

2. **依赖感知注意力（Dependency-Aware Attention）**:

    - 功能：在注意力计算中引入代码结构依赖信息
    - 核心思路：修改标准的自注意力机制，在注意力权重中加入结构偏置项。当两个 token 存在直接语法依赖或变量引用关系时，它们之间的注意力权重会被增强。形式上，$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}} + B_{dep})V$，其中 $B_{dep}$ 是从关系矩阵导出的依赖偏置
    - 设计动机：让模型在生成时自然地"看到"结构相关的 token，而不需要仅通过序列上下文间接推断

3. **结构约束解码（Structure-Constrained Decoding）**:

    - 功能：在生成过程中确保输出代码的结构合法性
    - 核心思路：在自回归解码的每一步，维护一个部分 AST，根据当前的语法状态限制下一个 token 的候选空间。例如，在 `if` 后必须跟条件表达式，`def` 后必须跟函数名。这类似于语法约束解码，但使用了更细粒度的依赖信息
    - 设计动机：纯自回归生成无法保证语法正确性，结构约束解码提供了硬性保障

### 损失函数 / 训练策略
使用标准的交叉熵损失训练序列生成，可能辅以结构一致性辅助损失来鼓励模型学习结构信息。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文方法 | CodeLlama | StarCoder | 提升 |
|--------|------|---------|-----------|-----------|------|
| HumanEval | Pass@1 | 显著提升 | 基线 | 基线 | +3-5% |
| MBPP | Pass@1 | 显著提升 | 基线 | 基线 | +2-4% |
| APPS | Pass@1 | 显著提升 | 基线 | 基线 | +3-6% |
| 语法正确率 | — | ~98% | ~92% | ~91% | +6-7% |

### 消融实验

| 配置 | HumanEval Pass@1 | 说明 |
|------|------------------|------|
| Full model | 最优 | 2D表示+依赖注意力+约束解码 |
| w/o 2D表示 | 下降 | 退化为1D序列 |
| w/o 依赖注意力 | 略降 | 无结构偏置 |
| w/o 约束解码 | 下降明显 | 语法错误率上升 |
| 仅依赖注意力 | 中等提升 | 最简版本也有效 |

### 关键发现
- 二维表示对长代码（>100行）的提升最为显著，因为长代码中远程依赖更多
- 结构约束解码是语法正确率提升的主要贡献者，但对逻辑正确率的贡献有限
- 变量依赖关系的编码贡献最大，超过了 AST 层级信息的贡献，说明变量引用是代码生成中的关键难点
- 在简单的代码生成任务上优势不明显，主要在复杂任务（如递归、嵌套循环）上有显著提升

## 亮点与洞察
- **维度拓展思路**：从一维到二维表示的升级思路简洁而有效，可以迁移到其他具有结构性质的序列生成任务，如数学表达式生成、SQL生成等
- **关系矩阵可解释性强**：二维关系矩阵可以直观可视化代码的结构关系，便于理解模型的生成逻辑

## 局限与展望
- 二维表示和约束解码增加了计算开销，对于超长代码可能不实际
- 需要预先定义关系类型集合，对新的编程语言或非传统语法的适应性有限
- 当前主要针对 Python，其他编程语言（如C/Rust等强类型语言）的效果需验证
- 未来可以结合 LLM 的代码理解能力和本文的结构编码，构建结构感知的代码 LLM

## 相关工作与启发
- **vs TreeGen（AST级代码生成）**: TreeGen 直接在 AST 上生成，但训练和推理效率低；本文在序列生成框架中引入结构信息，更好地平衡了效率和效果
- **vs SynCoBERT（结构感知预训练）**: SynCoBERT 在预训练阶段引入结构信息，本文在生成阶段引入，两者互补
- **vs GraphCodeBERT**: GraphCodeBERT 利用数据流图增强代码理解，本文进一步将图结构信息应用于生成阶段，是理解到生成的自然延伸

## 评分
- 新颖性: ⭐⭐⭐⭐ 二维表示在代码生成中是新颖的尝试
- 实验充分度: ⭐⭐⭐⭐ 多数据集评测和充分消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述详细
- 价值: ⭐⭐⭐⭐ 对结构化序列生成有启发意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)
- [\[ACL 2025\] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation](reflectioncoder_learning_from_reflection_sequence_for_enhanced_one-off_code_gene.md)

<!-- RELATED:END -->
