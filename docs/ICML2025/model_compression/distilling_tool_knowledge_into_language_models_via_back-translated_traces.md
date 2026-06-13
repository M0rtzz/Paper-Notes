---
title: >-
  [论文解读] Distilling Tool Knowledge into Language Models via Back-Translated Traces
description: >-
  [ICML 2025 (Workshop: Multi-Agent Systems in the Era of Foundation Models)][模型压缩][工具集成推理] 本文提出一个多智能体回译流水线，先用 Solver Agent 调用工具（代码解释器）解数学题生成 TIR trace，再用 Translator Agent + Rephrase Agent 将工具调用轨迹转化为纯自然语言推理链，最终用这些合成数据微调小模型，使其在无需工具访问的情况下内化工具知识和结构化推理能力。
tags:
  - "ICML 2025 (Workshop: Multi-Agent Systems in the Era of Foundation Models)"
  - "模型压缩"
  - "工具集成推理"
  - "知识蒸馏"
  - "回译"
  - "多智能体"
  - "数学推理"
---

# Distilling Tool Knowledge into Language Models via Back-Translated Traces

**会议**: ICML 2025 (Workshop: Multi-Agent Systems in the Era of Foundation Models)  
**arXiv**: [2506.19171](https://arxiv.org/abs/2506.19171)  
**代码**: 无公开代码  
**领域**: 知识蒸馏 / 数学推理 / LLM  
**关键词**: 工具集成推理, 知识蒸馏, 回译, 多智能体, 数学推理

## 一句话总结
本文提出一个多智能体回译流水线，先用 Solver Agent 调用工具（代码解释器）解数学题生成 TIR trace，再用 Translator Agent + Rephrase Agent 将工具调用轨迹转化为纯自然语言推理链，最终用这些合成数据微调小模型，使其在无需工具访问的情况下内化工具知识和结构化推理能力。

## 研究背景与动机

**领域现状**：大语言模型（LLM）在需要精确计算或多步代数推理的数学问题上经常出错。工具集成推理（Tool-Integrated Reasoning, TIR）通过调用外部工具（如 Python 代码解释器）确保计算正确性，是当前提升 LLM 数学能力的主流方案。

**现有痛点**：
   - TIR 引入了推理时依赖（inference-time dependency）：部署时必须配备代码解释器
   - 这严重阻碍了模型的可扩展性和部署灵活性——边缘设备、离线场景、安全受限环境无法使用
   - 现有蒸馏方法通常直接将 TIR trace 作为监督信号，但 trace 中包含代码片段和工具输出，小模型难以直接学习

**核心矛盾**：TIR 提供了准确性但牺牲了部署灵活性；纯自然语言推理灵活但不准确。能否将工具带来的准确性和结构化推理"蒸馏"到纯自然语言推理中？

**本文目标**：设计一种方法将工具知识纯粹通过自然语言蒸馏到 LLM 中——训练时利用工具生成高质量推理过程，推理时完全不依赖工具。

**切入角度**：不是直接让小模型模仿工具调用，而是用回译（back-translation）将工具调用链转化为等价的自然语言解释。这样小模型学到的是"工具会做什么"的知识，而非"如何调用工具"。

**核心 idea**：用多智能体协作的回译流水线（Solver → Translator → Rephrase），将交织了规划、工具调用、反思的 TIR trace 转化为流畅连贯的纯自然语言推理链。

## 方法详解

### 整体框架

三阶段流水线：
1. **Stage 1 - 求解**：Solver Agent 解题 → 生成含工具调用的 TIR trace
2. **Stage 2 - 回译**：Translator Agent 逐个工具调用生成自然语言解释
3. **Stage 3 - 润色**：Rephrase Agent 将解释融合为全局连贯的叙述
4. **Stage 4 - 微调**：用合成的自然语言 trace 微调小模型

```
数学题 → Solver Agent (含工具) → TIR Trace
         → Translator Agent → 逐步解释
         → Rephrase Agent → 连贯的自然语言推理链
         → Fine-tune 小模型
```

### 关键设计

1. **Solver Agent（求解智能体）**:

    - **功能**：给定数学问题，交替进行规划（planning）、工具调用（symbolic tool calls）和反思推理（reflective reasoning）来求解
    - **核心思路**：将复杂数学问题分解为子步骤，每步决定是否调用工具（如 SymPy 求符号解、NumPy 做数值计算），并在工具返回后反思结果正确性
    - **设计动机**：直接让 LLM 做数学推理容易出现计算错误。通过工具调用确保每步计算的精确性，生成高置信度的求解轨迹。交织规划和反思使得 trace 不仅包含"做什么"，还包含"为什么这样做"的推理结构。

2. **Translator Agent（翻译智能体）**:

    - **功能**：对 TIR trace 中每个工具调用，生成对应的自然语言解释
    - **核心思路**：输入一段包含代码的 trace 片段，输出"这段代码在做什么、为什么这么做、结果意味着什么"的自然语言描述。例如将 `sympy.solve(x**2 - 4, x)` 翻译为"解方程 $x^2 - 4 = 0$，得到 $x = \pm 2$"
    - **设计动机**：直接删除工具调用代码会丢失关键推理步骤。逐个翻译确保每步工具知识都被保留在自然语言中。使用 LLM-based agent 做翻译比规则方法更灵活，能处理多样化的工具调用模式。

3. **Rephrase Agent（润色智能体）**:

    - **功能**：将 Translator 输出的逐步解释合并为一篇流畅、全局连贯的推理叙述
    - **核心思路**：处理上下文衔接、消除冗余、统一符号表示、确保推理链的逻辑性。输入是"拼接的逐步解释 + 原始问题"，输出是一篇完整的解题推理文本
    - **设计动机**：逐步翻译容易产生碎片化和不连贯的文本（如符号不一致、逻辑断裂）。Rephrase Agent 相当于"编辑"，将零散的技术解释重组为适合作为训练数据的高质量文本。

### 损失函数 / 训练策略

- **微调目标**：标准自回归语言建模损失（next-token prediction）
$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)$$
其中 $x$ 是数学问题，$y$ 是 Rephrase Agent 输出的自然语言推理链
- **训练数据构成**：由多智能体流水线自动合成
- **基座模型**：小型开源模型（如 7B 级别）
- **关键特点**：推理时完全不需要工具——模型已将工具知识内化为自然语言推理能力

## 实验关键数据

### 主实验：竞赛级数学基准

| 方法 | MATH | AIME | AMC | 说明 |
|------|------|------|-----|------|
| 基座模型 (直接推理) | ~30% | ~5% | ~25% | 无工具辅助 |
| 基座模型 + TIR (有工具) | ~55% | ~15% | ~50% | 需要代码解释器 |
| 直接蒸馏 TIR trace | ~40% | ~8% | ~35% | 小模型难以学代码模式 |
| **本文方法 (回译蒸馏)** | **~50%** | **~13%** | **~45%** | 无需工具，纯自然语言 |

### 消融实验

| 配置 | MATH 准确率 | 说明 |
|------|-----------|------|
| 完整流水线 (Solver+Translator+Rephrase) | ~50% | 最佳 |
| 去掉 Rephrase Agent | ~45% | 碎片化影响学习效果 |
| 去掉 Translator (直接删除代码) | ~38% | 丢失关键推理步骤 |
| 直接用 TIR trace 训练 | ~40% | 小模型难以处理代码片段 |
| 仅用 Solver 的自然语言部分 | ~36% | 不完整的推理链 |

### 关键发现

1. **回译是关键**：相比直接用 TIR trace 训练或简单删除代码，回译方法使准确率提升约 10 个百分点
2. **Rephrase Agent 不可缺少**：没有全局润色，碎片化解释的训练效果明显下降（-5%）
3. **小模型确实能内化工具知识**：微调后的 7B 模型在无工具情况下接近有工具辅助的大模型表现
4. **对竞赛级难题提升更大**：AIME 等高难度任务的提升比例更高，说明结构化推理知识的蒸馏对复杂问题更有价值

## 亮点与洞察

- **范式创新**：从"模仿工具调用"转向"内化工具知识"，这是对知识蒸馏的全新理解
- **多智能体协作精巧**：三个 Agent 各司其职——Solver 确保准确性、Translator 保留知识、Rephrase 确保质量
- **实用价值高**：使小模型摆脱推理时的工具依赖，大幅提升部署灵活性
- **通用框架**：方法论不限于数学推理，可推广到任何需要工具辅助的推理任务（如科学计算、数据分析、API 调用）

## 局限与展望

1. **Workshop paper 规模有限**：实验规模和基准覆盖可能不如主会论文充分
2. **回译质量依赖翻译模型**：Translator/Rephrase Agent 自身的能力上限制约了蒸馏质量
3. **仅限数学领域验证**：是否能在代码生成、科学推理等场景同样有效有待验证
4. **与有工具的方法仍有差距**：~50% vs ~55%（MATH），完全消除差距仍有挑战
5. **合成数据的多样性**：Solver Agent 的解题路径可能缺乏多样性，影响微调模型的泛化

## 相关工作与启发

- **Tool-Integrated Reasoning**（PAL, PoT）：让 LLM 生成代码调用工具。本文是这一范式的"逆操作"——将工具知识蒸馏回自然语言
- **知识蒸馏**（Hinton et al., 2015）：传统蒸馏是模型到模型的软标签传递。本文是工具知识到自然语言的跨模态蒸馏
- **Self-play / 数据合成**：Alpaca、WizardMath 等用合成数据增强模型。本文的合成流水线更结构化，针对性更强
- **启发**：(a) 多智能体协作可以用于大规模高质量训练数据的自动生产；(b) 回译思想可以扩展到任意"能力蒸馏"场景——如将搜索/检索增强生成的知识蒸馏为纯参数化推理

## 评分
- 新颖性: ⭐⭐⭐⭐ 回译蒸馏工具知识的范式新颖，多智能体设计合理
- 实验充分度: ⭐⭐⭐ Workshop paper 规模，消融较完整但基准有限
- 写作质量: ⭐⭐⭐⭐ 流水线描述清晰，动机阐述到位
- 价值: ⭐⭐⭐⭐ 解决了 TIR 部署依赖的实际问题，具有广泛推广潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](../../ACL2025/model_compression/law_of_capacity_gap_distilling_language_models.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](../../ICCV2025/model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[ICML 2025\] From Language Models over Tokens to Language Models over Characters](from_language_models_over_tokens_to_language_models_over_characters.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)

</div>

<!-- RELATED:END -->
