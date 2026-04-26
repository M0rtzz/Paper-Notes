---
title: >-
  [论文解读] CodePercept: Code-Grounded Visual STEM Perception for MLLMs
description: >-
  [CVPR 2026][STEM视觉感知] 通过系统性缩放分析发现感知（perception）而非推理（reasoning）是 MLLM 在 STEM 领域的真正瓶颈，提出以可执行 Python 代码为锚定媒介的 CodePercept 范式——构建 100 万级 ICC-1M 数据集和 STEM2Code-Eval 基准，在 SFT+RL 两阶段训练后显著提升 MLLM 的 STEM 视觉感知和下游推理能力。
tags:
  - CVPR 2026
  - STEM视觉感知
  - 可执行代码
  - 图像重建
  - 代码锚定字幕
  - 多模态大模型
  - 感知增强
---

# CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.10757](https://arxiv.org/abs/2603.10757)  
**代码**: [TongkunGuan/Qwen-CodePercept](https://github.com/TongkunGuan/Qwen-CodePercept)  
**领域**: 多模态VLM / STEM感知  
**关键词**: STEM视觉感知, 可执行代码, 图像重建, 代码锚定字幕, 多模态大模型, 感知增强

## 一句话总结

通过系统性缩放分析发现感知（perception）而非推理（reasoning）是 MLLM 在 STEM 领域的真正瓶颈，提出以可执行 Python 代码为锚定媒介的 CodePercept 范式——构建 100 万级 ICC-1M 数据集和 STEM2Code-Eval 基准，在 SFT+RL 两阶段训练后显著提升 MLLM 的 STEM 视觉感知和下游推理能力。

## 研究背景与动机

**领域现状**：当前大量工作聚焦于用强化学习增强 MLLM 的推理能力（冷启动数据、RL 奖励设计、单模态推理数据迁移），但始终未回答一个根本问题：模型 STEM 任务失败到底是感知不足还是推理不足？

**现有痛点**：作者将 STEM 视觉推理解耦为感知（image→caption）和推理（caption→answer）两个独立阶段，固定一端独立缩放另一端，在 MathVision 数据集上发现：扩展感知始终优于扩展推理（如 Perception@32B+Reasoning@4B 大幅优于 Perception@4B+Reasoning@32B）。这证明感知才是 STEM 视觉推理的真正杠杆，但几乎无人系统性地解决感知问题。

**核心矛盾**：直觉方案——用 GPT/Gemini 生成描述性字幕进行知识蒸馏——有两大致命缺陷：(1) 教师模型在空间关系、数量细节上易产生幻觉；(2) 复杂 STEM 图像存在"描述失语"（descriptive aphasia），自然语言无法精确刻画辅助线构造、多面体空间关系等结构信息。

**本文目标** (1) 如何系统性增强 MLLM 的 STEM 视觉感知能力？(2) 如何直接评估感知能力而非用问题解答准确度作为代理指标？

**切入角度**：可执行代码天然具有精确语义、可验证性和结构化表达——只有完整理解图像才能生成正确的重建代码。代码是比自然语言更精确的"结构化字幕"。

**核心 idea**：用可执行 Python 代码作为 STEM 图像的精确感知媒介，同时作为训练信号（代码锚定字幕 + 图像转代码）和评估标准（代码重建图像的保真度）。

## 方法详解

### 整体框架

CodePercept 包含三大模块：(1) 图像-代码对构建引擎（三条互补 pipeline 生成高质量 image-code pairs）;(2) 两种代码锚定训练任务（Code-Grounded Caption Generation + STEM Image-to-Code Translation）;(3) SFT+RL 两阶段后训练。基于 Qwen3-VL 系列模型。

### 关键设计

1. **图像-代码对构建引擎（三条互补 Pipeline）**:

    - 功能：从现有 STEM 数据出发，大规模生成高质量 image-code 配对数据
    - 核心思路：三条并行管线——(i) Image Reproduction：对种子图像先生成详细描述再基于图像+描述生成 matplotlib 重建代码；(ii) Image Diversity：抽取种子图像的底层科学原理 $G_{\text{principle}}$，基于同一原理生成 K 个不同视觉实例化代码（如从多米诺谜题生成圆形轮盘、三角组合、网格图等变体）；(iii) Solid Geometry：针对 LLM 在立体几何代码上的根本缺陷，构建参数化模板库（立方体展开/折叠、正投影三视图、截面分析、多面体构造共 8 类），通过参数采样批量生成。统一质量控制过滤图像质量 $Q_I$、代码质量 $Q_C$ 和图像-代码一致性 $Q_{IC}$
    - 设计动机：Image Reproduction 受限于源图像多样性；Image Diversity 通过"抽象原理→重新实例化"突破数据多样性瓶颈；Solid Geometry 补齐 LLM 在空间推理代码生成上的能力短板

2. **代码锚定字幕生成（Code-Grounded Caption Generation）**:

    - 功能：利用 ground-truth 代码消除 MLLM 生成字幕中的幻觉
    - 核心思路：三步流程——(1) Native Caption：MLLM 直接从图像生成描述草稿 $t_{\text{draft}}$（语言自然但可能有事实错误）；(2) Code Analysis：利用代码本身 + 执行追踪器 $\xi(\mathbf{c})$（记录精确坐标、维度、z-order 层级等所有渲染细节）提取经验证的视觉事实 $t_{\text{code}}$；(3) Code-Grounded Refinement：以代码分析结果为依据外科手术式修正草稿中的数量错误、空间关系错误，同时保持原始语言风格和描述流畅度。公式 $t_{\text{new}} = G_{\text{refine}}(G_{\text{caption}}(\mathbf{x}), G_{\text{analyze}}(\mathbf{c}, \xi(\mathbf{c})))$
    - 设计动机：执行追踪器解决了"直接分析复杂代码对 LLM 太难"的问题——即使代码有深层递归和嵌套循环，执行日志也能提供确定性的渲染信息作为事实参考

3. **STEM 图像转代码翻译（STEM Image-to-Code Translation）**:

    - 功能：训练模型从图像直接生成可执行重建代码，作为自然语言的互补感知信号
    - 核心思路：先让 MLLM 从图像生成解释性代码草稿 $c_{\text{draft}}$（有逐步分解和参数解释但可能有事实错误），再用 ground-truth 代码修正错误同时保留解释性结构，得到 $c_{\text{new}} = G_{\text{refine}}(G_{\text{code}}(\mathbf{x}), \mathbf{c})$
    - 设计动机：代码提供了与自然语言互补的结构化视觉描述——用编程构造表达几何关系、数学约束和结构细节，解决自然语言"描述失语"问题

### 损失函数 / 训练策略

- **Stage 1 (SFT)**：基于 Qwen3-VL，在 ICC-1M 上联合训练图像字幕和图像转代码两个任务，1 epoch，32 A100
- **Stage 2 (RL)**：仅对代码生成任务做 GRPO 强化学习，选 1 万样本。奖励函数包含格式奖励 $r_{\text{fmt}}$（代码块格式是否正确）和内容奖励 $r_{\text{cnt}}$（执行成功率 + GPT-4o 评估的代码语义等价性 + 图像视觉相似度）

## 实验关键数据

### 主实验（STEM2Code-Eval 图像重建）

| 模型 | Image Score | Code Score | Avg | Exec Rate |
|------|-----------|-----------|------|-----------|
| Gemini2.5-Pro-Thinking | 68.89 | 75.41 | 72.15 | 91.7% |
| GPT5-Thinking | 64.97 | 64.98 | 64.98 | 96.6% |
| Qwen3-VL-4B-Instruct | 24.55 | 26.42 | 25.49 | 79.4% |
| CodePercept-4B-S1 | 38.13 | 43.43 | 40.78 | 80.7% |
| **CodePercept-4B-R1** | **47.17** | **45.86** | **46.52** | **91.3%** |
| Qwen3-VL-32B-Instruct | 36.85 | 39.98 | 38.42 | 81.8% |
| **CodePercept-32B-R1** | **68.97** | **62.53** | **65.75** | **95.9%** |

### 感知能力评估（Captioner-Solver Setup，LLM Solver: Qwen3-30A3-Thinking）

| 模型（Captioner） | MathVision | MathVista | MathVerse | DynaMath | WeMath | LogicVista | Avg |
|------|------|------|------|------|------|------|------|
| Qwen3-VL-4B-Instruct | 54.21 | 67.30 | 64.59 | 69.40 | 46.10 | 54.14 | 59.29 |
| CodePercept-4B-S1 | **57.63** | **69.60** | **65.59** | **71.38** | **47.81** | **60.40** | **62.07** |
| Qwen3-VL-32B-Instruct | 58.55 | 72.20 | 71.09 | 75.78 | 48.00 | 62.19 | 64.63 |
| CodePercept-32B-S1 | **62.27** | **72.90** | **71.70** | **77.41** | **54.19** | **65.33** | **67.30** |

### 消融实验

| 数据配置 | MathVision 提升 | Avg 提升 |
|---------|---------------|---------|
| 仅 Image Reproduce | 基线 | 基线 |
| + Image Diversity | +显著 | +显著 |
| + Solid Geometry | +进一步提升 | +进一步提升 |
| + CodeCap (代码锚定字幕) | +额外提升 | 字幕与代码互补 |
| + ImCode (图像转代码) | 最高 | 最高 |

### 关键发现

- CodePercept-4B-R1 在 STEM2Code-Eval 上 Image Score 从 24.55→47.17（+92%），Exec Rate 从 79.4%→91.3%，证明 RL 阶段有效提升代码质量
- CodePercept-32B-R1 的 Avg Score 65.75 接近 GPT5-Thinking 的 64.98，仅用开源模型就逼近最强闭源
- 感知提升直接传导到下游推理：CodePercept-32B-S1 作为 captioner 时，下游推理平均提升 2.7 个点
- 代码和字幕是互补的：只用字幕或只用代码训练效果都不如联合训练

## 亮点与洞察

- 缩放分析揭示"感知是瓶颈"——大家都在卷推理（RL、思维链、奖励设计），但真正制约 STEM 性能的是感知。这个发现可能重新引导整个领域的研究方向
- 代码作为感知媒介的范式转换——代码是"可验证、可执行的结构化字幕"，自然语言描述不了的空间关系、精确数值在代码中都有确定性表达。执行追踪器进一步解决了"LLM 看不懂复杂代码"的问题
- Image Diversity Pipeline 的"原理抽象→重新实例化"策略是高效的数据扩增思路——不是简单增强，而是保持科学严谨性的概念级多样化，可迁移到其他科学领域数据构建

## 局限与展望

- STEM2Code-Eval 基准仅 1000 个样本，覆盖的 STEM 子领域和难度分布可能不够全面
- 代码重建依赖 matplotlib，对非 2D 可视化内容（如真实实验照片、显微图像）不适用
- RL 阶段的奖励函数依赖 GPT-4o 评分，引入了外部模型偏差
- 立体几何模板库是手工设计的，覆盖范围受限于模板种类

## 相关工作与启发

- **vs 传统 STEM 推理增强**（KeyeVL、InternS1）：这些方法聚焦推理侧（RL、思维链），本文证明感知侧才是杠杆——同等算力投入感知增强收益更大
- **vs 知识蒸馏方法**：用强模型生成字幕的传统路线受限于教师模型幻觉；CodePercept 用代码执行结果作为 ground truth 消除幻觉，是更可靠的知识传递方式
- **vs 领域特定代码生成**（UI-to-code、Chart-to-code）：这些工作面向下游应用；CodePercept 的 image-code 配对兼具评测和训练双重价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 缩放分析揭示感知瓶颈 + 代码作为感知媒介是范式级创新
- 实验充分度: ⭐⭐⭐⭐ 6 个 STEM 基准 + STEM2Code-Eval + 消融，但 RL 消融不够细致
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，缩放分析图很有说服力
- 价值: ⭐⭐⭐⭐⭐ 可能重新引导 STEM 多模态研究从推理侧转向感知侧

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](../../ACL2026/code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](../../ACL2026/code_intelligence/collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)

<!-- RELATED:END -->
