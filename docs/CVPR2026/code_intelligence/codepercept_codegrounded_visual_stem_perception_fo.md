---
title: >-
  [论文解读] CodePercept: Code-Grounded Visual STEM Perception for MLLMs
description: >-
  [CVPR 2026][MLLM] 通过系统性缩放分析揭示**感知而非推理**是 MLLM 在 STEM 视觉任务上的真正瓶颈，提出以可执行代码为媒介增强感知能力的范式，构建 100 万级 Image-Caption-Code 三元组数据集 ICC-1M，包含代码锚定的标题生成和 STEM 图到代码翻译两个训练任务。
tags:
  - CVPR 2026
  - MLLM
  - STEM visual perception
  - code generation
  - perception bottleneck
  - ICC-1M
---

# CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.10757](https://arxiv.org/abs/2603.10757)  
**代码**: [GitHub](https://github.com/TongkunGuan/Qwen-CodePercept)  
**领域**: 代码智能 / 多模态大模型  
**关键词**: MLLM, STEM visual perception, code generation, perception bottleneck, ICC-1M

## 一句话总结

通过系统性缩放分析揭示**感知而非推理**是 MLLM 在 STEM 视觉任务上的真正瓶颈，提出以可执行代码为媒介增强感知能力的范式，构建 100 万级 Image-Caption-Code 三元组数据集 ICC-1M，包含代码锚定的标题生成和 STEM 图到代码翻译两个训练任务。

## 研究背景与动机

当 MLLM 在 STEM 视觉推理（数学、物理、化学、电气工程）上失败时，一个根本问题是：**失败源于感知不足还是推理不足？**

本文通过创新的**缩放分析实验**回答了这个问题：将 STEM 视觉推理解耦为感知（图像→描述）和推理（描述→答案）两个阶段，分别独立缩放一个阶段同时保持另一个不变：

- **感知@4B + 推理@4/8/32B 缩放**（蓝线） vs **推理@4B + 感知@4/8/32B 缩放**（红线）
- 结果一致表明：**缩放感知的收益持续优于缩放推理**

这一发现揭示了：感知是解锁当前 STEM 视觉推理的真正杠杆。

然而，直接通过知识蒸馏（如让 GPT/Gemini 生成描述性标题）来增强 STEM 感知面临两大障碍：
1. **幻觉问题**：教师模型会产生空间位置、数量关系、元素交互上的错误描述
2. **描述失语症**：许多 STEM 图像的复杂空间关系和精确数值**无法被自然语言充分捕获**（如多面体几何中的辅助线构造）

## 方法详解

### 整体框架

CodePercept 管线包含三个核心部分：

1. **Image-Code 对构建**（数据引擎）：三条互补管线生成大规模图像-代码对
2. **两个代码锚定训练任务**：Code-Grounded Caption Generation + STEM Image-to-Code Translation
3. **两阶段后训练**：SFT（CodePercept-S1）+ RL（CodePercept-R1）

### 关键设计

1. **三条数据生成管线** 产生 ICC-1M 数据集（100 万+ 三元组）：

    - **Image Reproduce (IR)**：$\mathbf{c} = G_{code}(\mathbf{I}, G_{caption}(\mathbf{I}))$。先让 MLLM 生成图像描述，再基于描述和图像生成再现代码。直觉简单但受限于源数据集的多样性。
    - **Image Diversity (ID)**：$[\mathbf{c}_1, \dots, \mathbf{c}_K] = G_{code}(\mathbf{I}, G_{principle}(\mathbf{I}))$。核心洞察：STEM 图像的底层原理可以被抽象并在不同上下文中重新实例化。例如从多米诺逻辑谜题种子图出发，生成圆形多米诺轮、三角组合、瓢虫斑点矩阵等变体，保持 STEM 严谨性同时引入结构新颖性。
    - **Solid Geometry Synthesis (SG)**：$\mathcal{C}_{geo} = \{\mathbf{c}_i \mid \mathbf{c}_i = \tilde{\mathbf{c}}_i(\boldsymbol{\theta})\}$。使用参数化代码模板生成立体几何图像，覆盖正方体展开、正交三视图、截面分析、多面体构造等 8 种典型场景。解决了当前 MLLM 在生成立体几何代码方面的根本性不足。

2. **Code-Grounded Caption Generation**：以代码为真值生成高质量标题，消除幻觉
    - 第一步：直接让 MLLM 描述图像得到 $\mathbf{t}_{draft}$（自然流畅但有事实错误）
    - 第二步：分析代码 + 执行追踪器 $\xi(\mathbf{c})$ 提取验证过的视觉事实 $\mathbf{t}_{code}$
    - 第三步：合成最终标题 $\mathbf{t}_{new} = G_{refine}(\mathbf{t}_{draft}, \mathbf{t}_{code})$，保留语言流畅性同时修正事实错误
    
    执行追踪器 $\xi(\mathbf{c})$ 是关键创新——它记录代码执行过程中的所有视觉元素：几何精度（坐标、尺寸、空间关系）、数量属性（计数、RGB 规格）、渲染语义（z-order 层级、变换矩阵）和 STEM 参数-视觉映射。

3. **STEM Image-to-Code Translation**：训练模型直接从图像生成可执行再现代码
    - 生成带有教学风格注释的解释性代码 $\mathbf{c}_{new} = G_{refine}(G_{code}(\mathbf{x}), \mathbf{c})$
    - 草稿代码有良好的教学模式（逐步分解、参数说明）但有事实错误
    - 用真值代码修正错误同时保留解释性结构

### 损失函数 / 训练策略

**Stage 1: SFT (CodePercept-S1)**：
- 基于 Qwen3-VL 系列，联合优化图像描述任务和图像到代码翻译任务
- 在 ICC-1M 上训练 1 epoch，32 块 A100 GPU，使用 SWIFT 框架

**Stage 2: RL (CodePercept-R1)**：
- 仅对代码生成应用 GRPO 强化学习
- 两类奖励信号：
    - Format Reward $r_{fmt}$：验证代码格式（```python``` 块）
    - Content Reward $r_{cnt}$：执行奖励（$r_{exec}$，是否可执行）+ 代码级奖励（$r_{code}$，GPT-4o 评估语义等价性）+ 图像级奖励（$r_{image}$，GPT-4o 评估视觉相似度）
- 总奖励 $r = r_{fmt} + r_{cnt}$
- 在 ICC-1M 中选择 10K 样本训练 1 epoch，使用 VeRL 框架

## 实验关键数据

### 主实验：感知评估（Captioner-Solver 设置，LLM Solver: Qwen3-30A3-Thinking）

| Image Captioner | MathVision | MathVista | MathVerse | DynaMath | WeMath | LogicVista | 平均 |
|----------------|-----------|----------|----------|---------|-------|-----------|------|
| Gemini2.5-Pro | 66.80 | 74.80 | 73.47 | 81.42 | 60.29 | 66.44 | 70.53 |
| Claude-Opus 4.1 | 59.61 | 71.10 | 56.19 | 73.25 | 44.86 | 59.28 | 60.72 |
| Qwen3-VL-4B | 54.21 | 67.30 | 64.59 | 69.40 | 46.10 | 54.14 | 59.29 |
| **CodePercept-4B-S1** | **57.63** | **69.60** | **65.59** | **71.38** | **47.81** | **60.40** | **62.07 (+2.8)** |
| Qwen3-VL-8B | 54.37 | 69.60 | 63.75 | 72.19 | 45.43 | 56.82 | 60.36 |
| **CodePercept-8B-S1** | **59.31** | **70.20** | **66.52** | **73.20** | **49.14** | **61.52** | **63.32 (+3.0)** |
| Qwen3-VL-32B | 58.55 | 72.20 | 71.09 | 75.78 | 48.00 | 62.19 | 64.63 |
| **CodePercept-32B-S1** | **62.27** | **72.90** | **71.70** | **77.41** | **54.19** | **65.33** | **67.30 (+2.7)** |

### STEM2Code-Eval 基准（图像再现感知评估）

| 模型 | Image Score | Code Score | 平均 | Exec Rate |
|------|-----------|-----------|------|----------|
| Gemini2.5-Pro-Thinking | 68.89 | 75.41 | 72.15 | 91.70% |
| GPT5-Thinking | 64.97 | 64.98 | 64.98 | 96.60% |
| Qwen3-VL-8B-Instruct | 28.59 | 28.23 | 28.41 | 85.30% |
| **CodePercept-8B-S1** | **44.53** | **46.78** | **45.66** | 87.60% |
| **CodePercept-8B-R1** | **50.25** | **47.04** | **48.65** | **93.40%** |
| Qwen3-VL-32B-Instruct | 36.85 | 39.98 | 38.42 | 81.80% |
| **CodePercept-32B-S1** | **61.14** | **56.99** | **59.07** | 93.00% |
| **CodePercept-32B-R1** | **68.97** | **62.53** | **65.75** | **95.90%** |

### 消融实验

| 数据配置 | 平均感知评分 | 增量 |
|----------|:---:|:---:|
| Baseline (Qwen3-VL-8B) | 60.36 | - |
| + IR-CodeCap | 60.91 | +0.55 |
| + ID-CodeCap | 62.15 | +1.79 |
| + SG-CodeCap | 62.75 | +2.39 |
| NativeCap (直接标题) | 60.78 | +0.42 |
| **CodeCap (代码锚定)** | **62.75** | **+2.39** |
| CodeCap + ImCode | **63.32** | **+2.96** |

### 关键发现

- **感知是 STEM 瓶颈**：缩放感知一致性优于缩放推理，各数据集均成立
- **代码锚定标题优于直接标题**：CodeCap 比 NativeCap 高 2.0 个百分点，证实代码可消除幻觉
- **三管线互补**：ID（多样化）增益最大，SG（立体几何）进一步提升
- **代码与标题互补**：联合训练图到标题 + 图到代码（63.32）优于仅标题（62.75），代码作为"结构化标题"提供精确空间/数量信息
- **RL 显著提升代码质量**：CodePercept-8B-R1 在 STEM2Code-Eval 上相比 S1 提升 +3.0（45.66 → 48.65）
- **超越大型模型**：CodePercept-8B-S1（8B）在感知任务上超越 Qwen2.5-VL-72B（6.2 个百分点）

## 亮点与洞察

- **核心发现极其重要**：感知而非推理是 STEM 瓶颈——这挑战了当前主流对 MLLM "推理能力不足" 的叙事，可能重新定向社区研究方向
- **"代码即感知" 范式**：可执行代码提供了自然语言无法比拟的精确语义——坐标、数量、空间关系都可以通过代码精确表达并通过执行验证
- **执行追踪器的巧妙利用**：将代码执行日志作为 LLM 分析代码的"外部说明书"，解决了 LLM 难以分析复杂递归/嵌套逻辑的问题
- **STEM2Code-Eval 基准**：首个通过代码生成直接评估 STEM 视觉感知的基准——只有完全理解图像才能通过代码忠实重现

## 局限与展望

- RL 阶段依赖 GPT-4o 作为评估器，成本高且可能引入偏差
- ICC-1M 数据仅使用 matplotlib 生成，限制了视觉多样性（如手绘图、照片级 STEM 图像）
- STEM2Code-Eval 仅 1,000 张图像，规模偏小
- 当前仅在 Qwen3-VL 系列上验证，对其他 MLLM 架构的适用性未知
- 代码生成限制在 Python matplotlib，无法覆盖需要 3D 渲染的复杂几何场景
- 感知与推理的解耦分析使用了 caption 作为中间表示，可能引入信息瓶颈

## 相关工作与启发

- **Qwen3-VL [2026]**：CodePercept 的基础模型，Qwen 团队的 MLLM
- **GRPO [DeepSeek 2024]**：Group Relative Policy Optimization，用于 RL 阶段
- **MathVision, MathVista, MathVerse**：STEM 视觉推理基准
- **InternVL / MiniCPM-V**：竞争性 MLLM，在 STEM2Code-Eval 上落后 CodePercept
- 启发：代码作为感知媒介的范式可扩展至其他需要精确描述的领域（如地图、建筑图、电路图、化学结构式）

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 综合评价 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](../../ACL2025/code_intelligence/codedpo_code_alignment.md)
- [Improving Code Localization with Repository Memory](../../ICLR2026/code_intelligence/improving_code_localization_with_repository_memory.md)

<!-- RELATED:END -->
