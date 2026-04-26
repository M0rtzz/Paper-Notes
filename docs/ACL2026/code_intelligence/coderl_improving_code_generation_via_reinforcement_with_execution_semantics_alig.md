---
title: >-
  [论文解读] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment
description: >-
  [ACL 2026][代码生成] 本文提出 CodeRL+，将执行语义对齐集成到 RLVR 训练管道中，通过让模型推断变量级执行轨迹来弥合代码文本表示与执行语义之间的差距，在代码生成上平均 pass@1 提升 4.6%，在代码推理和测试输出生成基准上分别提升 15.5% 和 4.4%。
tags:
  - ACL 2026
  - 代码生成
  - 执行语义对齐
  - RLVR
  - GRPO
  - 程序执行轨迹
---

# CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment

**会议**: ACL 2026  
**arXiv**: [2510.18471](https://arxiv.org/abs/2510.18471)  
**代码**: [https://github.com/jiangxxxue/CODERLPLUS](https://github.com/jiangxxxue/CODERLPLUS)  
**领域**: 代码生成 / 强化学习  
**关键词**: 代码生成, 执行语义对齐, RLVR, GRPO, 程序执行轨迹

## 一句话总结

本文提出 CodeRL+，将执行语义对齐集成到 RLVR 训练管道中，通过让模型推断变量级执行轨迹来弥合代码文本表示与执行语义之间的差距，在代码生成上平均 pass@1 提升 4.6%，在代码推理和测试输出生成基准上分别提升 15.5% 和 4.4%。

## 研究背景与动机

**领域现状**：LLM 通过自回归预训练学习代码的文本模式，已具备强大的代码生成能力。RLVR（带可验证奖励的强化学习）通过测试用例执行提供确定性反馈，尝试弥合文本模式与功能正确性之间的语义差距。

**现有痛点**：RLVR 仅依赖二元通过/失败信号，不足以建立代码文本表示与执行语义之间的良好对齐。实验表明 RLVR 训练后的模型在执行轨迹推断任务上仅比基线提升 4%，无法追踪循环中的变量变化等基本执行语义。

**核心矛盾**：LLM 的预训练目标（拟合文本分布）与评估标准（执行正确性）之间存在根本性错位。仅依赖最终执行结果的稀疏奖励无法让模型学习理解代码的运行时行为。

**本文目标**：在 RLVR 中引入执行语义对齐，使模型能推断变量级执行轨迹，提供直接的执行语义学习信号。

**切入角度**：将失败的代码探索重新利用为执行语义对齐的训练数据——让模型学习推断失败程序中每个变量的最终值。

**核心 idea**：代码生成（合成状态转换函数 $\Phi_p$）和执行语义对齐（理解 $\Phi_p$）是互补的双向关系，联合优化可以超越表面文本模式的学习。

## 方法详解

### 整体框架

CodeRL+ 在 GRPO 训练管道中引入双目标优化：(1) 代码生成——生成解决编程问题的代码并通过测试用例验证；(2) 执行语义对齐——推断程序中每个变量的最终值。两个目标通过混合 prompt 分布 $\mathcal{B}_{\text{mixed}} = \alpha \cdot \mathcal{B}_{\text{code}} + (1-\alpha) \cdot \mathcal{B}_{\text{align}}$ 联合训练。

### 关键设计

1. **执行语义对齐任务**:

    - 功能：让模型学习推断代码的运行时行为
    - 核心思路：给定程序 $p$ 和输入 $x$，模型需要推断每个变量 $var_i$ 在执行轨迹中最后一次赋值时的值。这比推断完整执行轨迹可行，同时隐含编码了控制流路径和数据依赖关系
    - 设计动机：完整执行轨迹在循环等场景中状态数爆炸，用最终变量值作为可行的近似

2. **基于失败探索的动态数据构建**:

    - 功能：从模型自身的失败代码中动态构建执行语义对齐数据
    - 核心思路：在代码生成的 rollout 阶段，失败的程序被重新用于构建对齐 prompt $q' = \langle p_{\text{fail}}, x, V \rangle$，利用执行失败程序获得的真实执行语义 $\mathcal{F}_{p_{\text{fail}}}(x)$ 作为标签。初始迭代全部为代码生成任务，后续逐步引入对齐样本
    - 设计动机：无需额外数据源，对齐数据与模型能力共同演化，失败程序恰好暴露了模型理解执行语义的不足

3. **细粒度变量级奖励**:

    - 功能：为执行语义对齐提供比二元信号更细粒度的奖励
    - 核心思路：奖励为模型正确推断的变量比例 $R_{\text{sem}}^{(i)} = \frac{1}{|V|}\sum_{v_k \in V} \mathbb{1}[\hat{v}_k^{\text{final}} = v_k^{\text{final},*}]$，允许部分正确的推断获得正奖励
    - 设计动机：相比代码生成的全有或全无奖励，变量级奖励提供更密集的学习信号

### 损失函数 / 训练策略

联合优化目标 $\mathcal{J}_{\text{CodeRL+}}(\theta) = \mathbb{E}[r(\theta) \cdot A_{\text{gen}}] + \mathbb{E}[r'(\theta) \cdot A_{\text{sem}}]$，使用 GRPO 框架。训练数据比例 $\alpha = 0.6$（代码生成 60%，语义对齐 40%）。基于 Qwen2.5-Coder-7B-Instruct，batch size 128，8 rollout 采样，8×A100 GPU。

## 实验关键数据

### 主实验

**Qwen2.5-Coder-7B-Instruct 上的 pass@1（%）**

| 方法 | HumanEval | LeetCode | LiveCodeBench | Avg | 代码推理 | 测试输出 |
|------|-----------|----------|---------------|-----|---------|---------|
| Base | 88.4 | 50.6 | 34.3 | 57.8 | 60.8 | 48.8 |
| GRPO | 87.2 | 60.0 | 35.4 | 60.9 | 66.0 | 48.4 |
| OlympicCoder | 75.6 | 45.3 | 30.9 | 50.6 | 68.5 | 31.1 |
| CodeReasoner | 88.4 | 50.0 | 34.8 | 57.7 | 78.5 | 65.1 |
| **CodeRL+** | **90.9** | **63.3** | **36.9** | **63.7** | **85.0** | 53.2 |

### 消融实验

| 配置 | 代码生成 Avg | 代码推理 | 说明 |
|------|-------------|---------|------|
| GRPO（基线） | 60.9 | 66.0 | 仅代码生成 |
| + 执行语义对齐 | **63.7** | **85.0** | CodeRL+ 完整版 |
| 仅执行语义对齐 | - | 提升 | 单独对齐也有效 |
| 不同 RL 算法（REINFORCE++, DAPO） | 均提升 | 均提升 | 跨算法一致 |

### 关键发现

- CodeRL+ 在代码生成上比 GRPO 平均提升 4.6%（相对提升），在代码推理上提升 15.5%
- CodeRL+ 成功弥合了代码生成和代码推理之间的性能鸿沟——以往专注代码推理的方法往往损害代码生成，反之亦然
- 跨模型（Qwen、DeepSeek、Llama）和跨 RL 算法（GRPO、REINFORCE++、DAPO）均有稳定提升
- 探测实验证明 CodeRL+ 训练后，模型在生成代码时更多地考虑执行语义

## 亮点与洞察

- 失败探索的重新利用是关键设计亮点——不浪费任何计算资源，失败代码直接成为语义对齐的训练数据
- 双目标联合优化实现了"合成 $\Phi_p$"和"理解 $\Phi_p$"的良性循环
- 无需额外数据源或教师模型蒸馏，对齐数据完全来自模型自身的探索

## 局限与展望

- 执行轨迹近似（仅推断最终变量值）可能丢失中间状态的关键信息
- 依赖可执行测试用例提供奖励，对无法自动化验证的编程任务（如 UI 开发）不适用
- 仅评估了 Python 代码生成，对其他编程语言的泛化效果待验证

## 相关工作与启发

- **vs CODEI/O**: CODEI/O 通过教师蒸馏+SFT 学习执行，CodeRL+ 通过 RL 自我探索学习，泛化性更好
- **vs CodeReasoner/CodeBoost**: 这些方法仅优化代码推理，可能损害代码生成；CodeRL+ 联合优化两者
- **vs 标准 GRPO**: 标准 GRPO 对执行语义理解提升有限（4%），CodeRL+ 通过显式对齐大幅提升

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在 RLVR 中集成执行语义对齐，利用失败探索构建训练数据
- 实验充分度: ⭐⭐⭐⭐⭐ 五个基准、多模型、多 RL 算法、探测分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，形式化严谨，但部分符号较重
- 价值: ⭐⭐⭐⭐⭐ 为代码生成 RL 训练提供了重要的执行语义学习信号

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi_agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)

<!-- RELATED:END -->
