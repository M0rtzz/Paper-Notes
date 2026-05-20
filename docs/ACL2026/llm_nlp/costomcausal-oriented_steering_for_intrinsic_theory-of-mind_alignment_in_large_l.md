---
title: >-
  [论文解读] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][心智理论] 提出 CoSToM 框架，先用因果追踪定位 LLM 中编码心智理论（ToM）特征的关键层（发现主要在早期层），再通过激活转向在这些层上进行轻量级对齐，使 LLM 在谈判和说服对话中显著提升社会推理质量——从"知道但不会用"变为"知道且会用"。
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "心智理论"
  - "因果追踪"
  - "激活转向"
  - "对话系统"
  - "社会推理"
---

# CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.10031](https://arxiv.org/abs/2604.10031)  
**代码**: [GitHub](https://github.com/CGCL-codes/CoSToM)  
**领域**: LLM/NLP  
**关键词**: 心智理论, 因果追踪, 激活转向, 对话系统, 社会推理

## 一句话总结
提出 CoSToM 框架，先用因果追踪定位 LLM 中编码心智理论（ToM）特征的关键层（发现主要在早期层），再通过激活转向在这些层上进行轻量级对齐，使 LLM 在谈判和说服对话中显著提升社会推理质量——从"知道但不会用"变为"知道且会用"。

## 研究背景与动机

**领域现状**：心智理论（ToM）——理解他人信念、欲望和意图的能力——是人类社会智能的标志。LLM 在标准 ToM 基准上表现不错，但研究发现它们在任务特定场景中难以泛化，依赖精心设计的 prompt 来模拟推理。

**现有痛点**：存在关键的"内部知识-外部行为"错位：LLM 可以正确回答 ToM 问题（推断用户想要柴火），但在实际谈判中却生成不连贯的提议（提供水而非柴火）。一旦去掉"推断并回应"的显式指令，模型就无法将内部编码的心理状态落地为行为。

**核心矛盾**：LLM 展现的 ToM 能力可能不是稳定的内在认知，而是由指令触发的临时模拟——内部有知识但无法自发外化为行为。

**本文目标**：(1) 发现 LLM 是否真正具有 ToM 相关的内部表征；(2) 这些表征在模型的哪些层；(3) 能否通过干预这些表征来提升下游对话质量。

**切入角度**：从机制性可解释性出发，用因果追踪定位 ToM 特征，然后用激活转向主动干预。

**核心 idea**：用冻结的 probe 解码器作为可微分验证器，反向传播 ToM 对齐损失到编码器的 ToM 关键层，通过梯度桥机制只更新浅层的 LoRA 适配器。

## 方法详解

### 整体框架
两阶段：(1) ToM 解释阶段——因果追踪定位 ToM 特征编码的关键层；(2) ToM 转向阶段——在关键层安装 LoRA 适配器，用 probe 解码器的 ToM 问答准确率作为监督信号反向传播更新编码器。推理时编码器生成 ToM 增强的激活，解码器据此生成对话。

### 关键设计

1. **因果追踪定位 ToM 层（Interpreting ToM）**:

    - 功能：识别 LLM 中编码 BDI（信念-欲望-意图）信息的关键层
    - 核心思路：实例化两个模型副本——上下文编码器处理对话历史，探针解码器接收编码器某层 $\ell$ 的冻结激活并尝试回答 ToM 问题。通过逐层扫描解码器的回答准确率，确定哪些层包含足够的 ToM 信息
    - 设计动机：先理解再干预——知道 ToM 特征在哪里，才能精确地进行干预

2. **梯度桥机制（Gradient Bridge）**:

    - 功能：通过冻结的解码器反向传播 ToM 对齐损失到编码器
    - 核心思路：编码器处理对话历史 → 在 ToM 关键层 $\ell$ 截取激活 → 注入冻结的 probe 解码器 → 解码器回答 ToM 问题 → 与 BDI 标签计算交叉熵损失 → 梯度穿过冻结解码器、穿过激活接口，流入编码器的层 0 到层 $\ell$，只更新这些浅层的 LoRA 适配器
    - 设计动机：直接微调 ToM QA 任务与对话生成任务不对齐，效果差。梯度桥绕开了这个问题——不训练"如何回答 ToM 问题"，而是训练"如何生成 ToM 丰富的表征"

3. **推理时的 ToM 增强对话生成**:

    - 功能：将 ToM 对齐的内部表征转化为高质量对话
    - 核心思路：推理时，ToM 增强的编码器处理对话历史，解码器不再回答 ToM 问题而是生成任务特定的对话（如谈判/说服），条件化在 ToM 丰富的激活上
    - 设计动机：训练和推理的解耦设计——训练时解码器是 ToM 验证器，推理时解码器是对话生成器。CoSToM 作为即插即用模块可以泛化到不同社交任务

## 实验关键数据

### 主实验（谈判和说服对话质量）

| 方法 | 对话质量提升 | 说明 |
|------|------------|------|
| 标准 prompt | 基线 | 仅用通用指令 |
| ToM-explicit prompt | +显著 | 需要精心设计的 prompt |
| Full-layer LoRA | +中等 | 参数多但改进不明显 |
| **CoSToM** | **+最大** | 仅更新 ToM 关键层的 LoRA |

### 因果追踪发现

| 发现 | 说明 |
|------|------|
| ToM 特征主要编码在**早期层** | 与直觉相反——通常认为高层编码语义 |
| BDI 三要素在不同层有不同峰值 | 信念/欲望/意图的编码位置不完全重合 |
| 跨模型一致 | Llama-3-8B 和 Qwen2.5-7B 都呈现类似模式 |

### 关键发现
- **ToM 特征主要编码在早期层**这一发现颠覆了"高层=高级语义"的常见假设
- **CoSToM 作为即插即用模块跨任务泛化**：在谈判和说服两个不同社交任务上都有效
- **梯度桥比直接 ToM QA 微调更有效**：因为后者训练目标与对话生成不对齐
- **轻量级**：只需更新 ToM 关键层的 LoRA 适配器，参数量远小于全层微调

## 亮点与洞察
- **"从解释到干预"的方法论**非常有启发性——先用因果追踪回答"在哪里"，再用激活转向回答"怎么用"，这种两步范式可以应用于任何 LLM 内部能力的对齐
- **冻结解码器作为可微分验证器**的设计巧妙地解决了"ToM 推理 ≠ 对话生成"的任务不对齐问题
- **"内部知识-外部行为"错位**的诊断对整个 LLM 对齐领域都有警示意义

## 局限与展望
- 需要 BDI 标注数据，获取成本较高
- 双模型架构的内存占用是 2N，对资源有要求
- ToM 关键层的定位可能因任务和数据分布变化
- 仅在谈判和说服两个任务上验证，更多社交场景（如安慰、教育）待探索
- 因果追踪的计算成本在大模型上可能较高

## 相关工作与启发
- **vs Prompt-based ToM**: prompt 方法是外部支架，CoSToM 是内部对齐——前者需要每次精心设计 prompt，后者一次训练永久生效
- **vs MindDial (Qiu et al., 2024)**: MindDial 显式追踪信念文本并拼接到输入，会传播错误。CoSToM 在激活空间操作，避免了文本层面的误差传播
- **vs Mechanistic Interpretability**: 大多数工作止步于诊断，CoSToM 从诊断走到了治疗

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 因果追踪+梯度桥的 ToM 对齐范式非常新颖
- 实验充分度: ⭐⭐⭐⭐ 跨模型验证+消融，但下游任务只有两个
- 写作质量: ⭐⭐⭐⭐⭐ RQ 驱动的结构非常清晰，图示优秀
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 社会智能和对齐研究都有深远启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](../../ACL2025/llm_nlp/theory_of_mind_llm.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](dual_alignment_between_language_model_layers_and_human_sentence_processing.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
