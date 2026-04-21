---
title: >-
  [论文解读] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion
description: >-
  [模型压缩] 提出 ThinkingF 流水线，通过大规模知识蒸馏与模板引导的推理轨迹合成分别增强 LLM 的形式语言领域知识和非形式到形式的推理能力，再经两阶段 SFT + RLVR 融合两种能力，7B/32B 模型在 FormalMATH-Lite 和 ProverBench 上达到 SOTA。
tags:
  - 模型压缩
---

# StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion

- **会议**: AAAI 2026
- **arXiv**: [2508.04440](https://arxiv.org/abs/2508.04440)
- **代码**: [stepfun-ai/StepFun-Formalizer](https://github.com/stepfun-ai/StepFun-Formalizer)
- **领域**: 自动形式化 / 数学推理
- **关键词**: Autoformalization, Lean 4, 知识蒸馏, 推理模板, RLVR, SFT, 形式化验证

## 一句话总结

提出 ThinkingF 流水线，通过大规模知识蒸馏与模板引导的推理轨迹合成分别增强 LLM 的形式语言领域知识和非形式到形式的推理能力，再经两阶段 SFT + RLVR 融合两种能力，7B/32B 模型在 FormalMATH-Lite 和 ProverBench 上达到 SOTA。

## 背景与动机

自动形式化（Autoformalization）旨在将自然语言数学表述翻译为 Lean 4 等形式语言中的可验证语句。现有方法存在两类典型缺陷：

1. **通用大模型缺乏形式语言知识**：如 Claude4-thinking 不熟悉 Lean 4 特定定义（如 Euler 函数的形式化表达），导致代码实现错误
2. **专用模型缺乏推理能力**：如 Kimina-Autoformalizer 在面对复杂自然语言问题时，无法正确理解语境并完成非形式-形式对齐

作者通过 GPT-4o 对约 10K 输出进行错误分类，发现 Kimina-Autoformalizer 在"问题理解错误"和"非形式-形式对齐错误"两类上占比极高，证实了推理能力不足是核心瓶颈。

## 方法详解

### 总体框架：ThinkingF 流水线

ThinkingF 包含 4 个阶段：知识蒸馏与筛选 → 推理数据合成 → 两阶段 SFT → RLVR。

### 设计一：知识蒸馏与三层质量筛选

**目标**：构建大规模高质量非形式-形式数据对，补充模型的形式语言知识。

1. **非形式问题准备**：从 NuminaMath-1.5 数据集筛选约 256K 非形式数学问题
2. **形式语句生成**：用 Kimina-Autoformalizer 为每个问题生成 16 个候选形式语句 $\{y_{ij}\}_{j=1}^{16}$
3. **三层筛选**：
    - **语法检查**：用 Lean4 REPL 过滤语法错误，保留正确语句 $\{y_{ij}^*\}_{j=1}^{m_i}$
    - **多数投票**：通过 BEq 验证将语句划分等价类，从最大等价类中选取最优形式化：

$$y_i^{**} = \arg\max_{y_{ij}^*} \sum_{k=1}^{m_i} \mathbb{1}(y_{ik}^* \sim y_{ij}^*)$$

   - **LLM 有效性评估**：用 DeepSeek-V3 过滤过于简单或包含内在矛盾的语句

最终保留约 183K 对高质量知识数据。

### 设计二：模板引导的推理轨迹合成

**目标**：为模型注入非形式到形式的推理能力，弥补直接蒸馏推理过程效果差的问题。

推理模板包含两个核心部分：

1. **非形式问题理解**：重述原始问题 → 分析高层逻辑结构 → 分解数学概念与对象
2. **非形式到形式分析**：预判形式化中可能遇到的陷阱 → 采用分治范式将自然语言数学对象逐一映射到 Lean 4 形式表达

基于人工标注的非形式-形式对 $\{(\hat{x}_i, \hat{y}_i)\}$（来自 MiniF2F、ProofNet、PutnamBench 等，共约 5.8K 条），用 Claude 3.7 Sonnet 按模板生成推理轨迹 $\hat{c}_i$。

**关键发现**：直接从通用推理模型（如 Claude4-thinking）蒸馏推理轨迹效果显著更差，因为通用模型在形式化过程中倾向于"解题"而非"翻译"。

### 设计三：两阶段 SFT

- **第一阶段**：用知识数据 $\{(x_i, y_i)\}$ 微调，输出前插入 `<think></think>` 空标签以保持格式一致性
- **第二阶段**：用推理数据 $\{(\hat{x}_i, \hat{c}_i, \hat{y}_i)\}$ 微调，将推理轨迹包裹在 `<think>...</think>` 中

基座模型选择 DeepSeek-R1-Distill-Qwen（7B / 32B），因其在非形式数学和编码上表现优异。

### 设计四：RLVR 强化学习

使用 BEq 等价验证作为可验证奖励，奖励函数为：

$$R(y_i, \hat{y}_i) = \begin{cases} 1, & \text{if } y_i \sim \hat{y}_i \\ 0, & \text{otherwise} \end{cases}$$

采用 GRPO 算法，结合 DAPO 的动态采样和 token 级损失改进。训练 7B 模型 450 步、32B 模型 350 步后，奖励从 0.232 提升到 0.347，平均 BEq@1 从 0.258 提升到 0.303。

## 实验结果

### 主要结果对比（BEq@1 / BEq@16，%）

| 模型 | FormalMATH-Lite | ProverBench | CombiBench |
|------|:-:|:-:|:-:|
| OpenAI o3-pro | 22.6 / 35.5 | 24.7 / 36.2 | 9.0 / 16.0 |
| Claude4-thinking | 20.8 / 32.2 | 24.4 / 35.6 | 9.7 / 18.0 |
| DeepSeek-R1-671B | 18.4 / 31.3 | 23.5 / 34.5 | 8.1 / 20.0 |
| Kimina-Autoformalizer-7B | 35.1 / 60.2 | 13.3 / 25.3 | 2.6 / 6.0 |
| **StepFun-Formalizer-7B** | **38.3 / 61.2** | **25.1 / 37.9** | **5.2 / 11.0** |
| **StepFun-Formalizer-32B** | **40.5 / 59.3** | **26.7 / 38.5** | **6.9 / 14.0** |

### 消融实验（OOD 基准，BEq@1 / BEq@16，%）

| 配置 | ProverBench | CombiBench |
|------|:-:|:-:|
| ThinkingF（完整） | 25.1 / 37.9 | 5.2 / 11.0 |
| 去除知识数据 | 24.5 / 37.4 | 3.9 / 10.0 |
| 去除推理数据 | 21.8 / 25.3 | 5.3 / 6.0 |

## 关键发现

1. **推理数据是主要贡献者**：去除推理数据后 BEq@16 大幅下降（37.9→25.3），说明推理能力对性能上限至关重要
2. **知识数据起互补作用**：虽单独贡献较小，但与推理数据融合后效果显著
3. **模板引导优于直接蒸馏**：直接从 Claude4-thinking 蒸馏推理轨迹，BEq@1 在 ProverBench 上从 25.1 降至 21.8
4. **RL 持续有效**：即使在 SFT 已使用的 5.8K 数据上继续 RL，仍带来稳定提升
5. **端到端定理证明**：用 Kimina-Prover 对 10K 问题验证，StepFun-Formalizer-7B 产生 4940 个可证明语句，优于 Kimina-Autoformalizer-7B 的 4549 个

## 亮点

- **首个结合领域知识与非形式到形式推理的自动形式化训练流水线**，填补了通用模型和专用模型各自的能力缺口
- **模板引导推理合成**的设计巧妙，避免了通用推理模型"解题而非翻译"的问题
- **7B 小模型超越 671B 通用模型**（DeepSeek-R1）在 ProverBench 上的表现，体现了极高的参数效率
- 用 BEq 等价验证作为 RL 奖励信号，实现了可靠的可验证反馈闭环

## 局限性

1. **数据量有限**：仅 5.8K 推理数据和 183K 知识数据；作者指出 32B 模型需要更多数据才能进一步超越 7B
2. **形式语言局限于 Lean 4**：未验证在 Coq、Isabelle 等其他形式语言上的泛化性
3. **CombiBench 表现仍较低**：涉及复杂现实场景和长上下文的组合数学问题仍具挑战（7B 仅 5.2%）
4. **依赖外部模型**：知识蒸馏依赖 Kimina-Autoformalizer，推理合成依赖 Claude 3.7 Sonnet，流水线可复现性受限
5. **评估方法局限**：BEq 使用 `exact?` 策略进行等价验证，是最严格但可能遗漏语义等价的情况

## 相关工作

- **自动形式化**：LeanFormalizer（SFT/PPO）、Kimina-Autoformalizer（专家迭代）、Goedel-Formalizer（人工标注扩展）、Herald-Translator、Mathesis（首个加入 RL 但无推理过程）
- **推理增强 LLM**：DeepSeek-R1、OpenAI o1 范式下的领域特化——Fin-R1（金融）、Table-R1（表格）、CodeV-R1（代码验证）、R1-Code-Interpreter
- **定理证明**：DeepSeek-Prover-V2、Kimina-Prover、Goedel-Prover

## 评分

⭐⭐⭐⭐ — 方法设计系统且有效，知识-推理双数据流水线思路清晰，实验充分；7B 模型超越 671B 通用模型令人印象深刻；局限在于数据规模偏小且仅支持 Lean 4。

<!-- RELATED:START -->

## 相关论文

- [S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)

<!-- RELATED:END -->
