---
title: >-
  [论文解读] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use
description: >-
   提出 VTool-R1，首个通过强化学习微调训练 VLM 生成交错文本和视觉中间推理步骤的框架，使模型学会"用图像思考"。
tags:

---

# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2505.19255](https://arxiv.org/abs/2505.19255)
- **代码**: [https://github.com/VTOOL-R1/vtool-r1](https://github.com/VTOOL-R1/vtool-r1)
- **领域**: 视觉语言模型 / 强化学习微调 / 工具使用 / 多模态推理
- **关键词**: RFT, VLM, 视觉推理, 工具使用, GRPO, 多模态思维链

## 一句话总结
提出 VTool-R1，首个通过强化学习微调训练 VLM 生成交错文本和视觉中间推理步骤的框架，使模型学会"用图像思考"。

## 研究背景与动机

### 核心问题
RFT（强化学习微调）已大幅提升 LLM 的推理能力，但在 VLM 领域的复制尝试仍局限于**纯文本推理**：模型仅在初始编码阶段处理图像，推理链完全以文本形式生成，缺乏中间视觉推理步骤。

### 为什么纯文本推理不够？
即使最先进的 VLM 也可能依赖语言捷径。例如，展示一只六指手的图片并询问"有几个手指"，模型可能基于"一只手有五个手指"的文本推理路径回答"五"，忽略视觉证据。

### 现有方法局限
- **Visual Sketchpad**：推理时引入视觉步骤，但没有训练机制，仅在 GPT-4o 等强模型上有效
- **Refocus**：生成视觉编辑但依赖商业模型预生成，在开源弱模型上效果差
- **R1-VL 等**：仅训练纯文本 CoT，不包含视觉推理步骤

## 方法详解

### 核心思想

VTool-R1 将 Python 视觉编辑工具集成到 RFT 过程中，使 VLM 通过**结果导向奖励**自主学习何时、如何生成视觉推理步骤。

### 推理与 Rollout 流程

两轮模型执行：
1. **第一轮**：VLM 根据图像和问题生成 Thought 0（分析关注点）+ Action 0（工具调用或直接回答）
2. **执行工具**：在 Python 沙箱环境中运行生成的代码，产生编辑后图像 $I'$
3. **第二轮**：VLM 对原始图像和编辑后图像一起推理，生成最终答案

形式化表示：
$$y \sim \pi_\theta(\cdot | I, x; \texttt{T}) = \pi_\theta(\cdot | I \oplus I', x) = \pi_\theta(\cdot | I \oplus \texttt{T}(y', I), x)$$

其中 $\oplus$ 为双图像拼接输入。

### RFT 训练目标

仅优化最终推理响应 $y$（而非中间工具调用 $y'$）：

$$\max_{\pi_\theta} \mathbb{E}_{[I,x] \sim \mathcal{D}, y \sim \pi_\theta(\cdot|I,x;\texttt{T})} [r_\phi(I,x,y)] - \beta \mathbb{D}_{KL}[\pi_\theta(\cdot|I,x;\texttt{T}) \| \pi_{\text{ref}}(\cdot|I,x;\texttt{T})]$$

基于 GRPO 的优化：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta\mathbb{D}_{KL}[\pi_\theta||\pi_{\text{ref}}]\right]$$

### 奖励设计

采用**纯结果导向奖励**：轻量 LLM 评判器评估预测答案与 ground truth 的匹配度，匹配则奖励 1。

**关键发现**：过程奖励（惩罚失败工具调用或奖励成功调用）导致奖励 hacking——模型要么完全避免工具使用，要么生成虚假的"成功"工具调用。

### 视觉编辑工具集

表格任务：
- Highlight Column/Row：半透明红色覆盖
- Mask Column/Row：白色遮罩无关区域
- Draw Column/Row：红色边界框标注

图表任务：类似操作应用于条形图的各个柱状。

## 实验

### 主实验结果

| 模型 | 配置 | Chart Split | Table Split |
|------|------|-------------|-------------|
| Qwen2.5-VL 3B | Pure Run | 51.8 | 41.3 |
| Qwen2.5-VL 3B | Tool Use (无训练) | 24.6 | 24.3 |
| **Qwen2.5-VL 3B** | **VTool-R1** | **64.0** | **57.9** |
| Qwen2.5-VL 7B | Pure Run | 76.2 | 64.7 |
| **Qwen2.5-VL 7B** | **VTool-R1** | **80.7** | **71.7** |
| GPT-4o | Pure Run | 82.9 | 75.7 |
| GPT-4o | Tool Use | 80.5 | 77.0 |

### 与其他方法对比

| 方法 | Chart Split | Table Split |
|------|-------------|-------------|
| Deepeyes (7B) | 60.0 | - |
| R1-VL (7B) | 63.8 | 45.4 |
| **VTool-R1 (7B)** | **80.7** | **71.7** |

### 关键发现

1. **RFT 使更好的工具使用成为可能**：训练后 3B/7B 模型学会有效使用工具
2. **工具使用非单调递增**：训练过程中工具调用频率和成功率波动，模型学会选择性使用
3. **结果导向奖励最可靠**：过程奖励导致奖励 hacking
4. **VTool-R1 显著优于 Deepeyes**：80.7 vs 60.0（Chart Split）
5. **约 50 步训练内收敛**

### 失败案例分析
- 正确生成视觉步骤但第二轮推理错误
- 视觉增强有轻微瑕疵（数字被边界框遮挡）
- 错误判断不需要工具但直接回答错误
- 工具代码执行失败

## 亮点

1. **首个 RFT 训练 VLM 生成多模态思维链的框架**
2. **优雅的设计**：仅优化最终响应，让模型自主决定是否使用工具
3. **实际有效**：3B 模型经训练后媲美或超越 GPT-4o 的工具使用能力
4. **深入的训练动态分析**：工具使用频率、成功率的演化揭示自适应行为

## 局限性

1. 当前仅支持单轮工具调用，多轮视觉推理留待未来
2. 工具集限于选择性注意力操作，尚未扩展到更复杂的视觉工具
3. 需要 VLM 支持多图像输入
4. 缺乏精确的工具调用正确性 oracle 验证器
5. 训练需要大量 GPU 资源（32B 模型需 8×H200）

## 相关工作

- **视觉 CoT**: ViperGPT (通过 Python 程序)、Visual Sketchpad (推理时画板)
- **LLM/VLM 工具使用**: Search-R1、ReTool — 文本工具的 RFT
- **VLM RFT**: R1-V、Vision-R1 — 仅文本推理链
- **并发工作**: Deepeyes、OpenThink-IMG — 不同的工具和任务设计

## 评分
- **创新性**: ⭐⭐⭐⭐⭐ — 首次成功训练 VLM 生成多模态推理链
- **实验充分性**: ⭐⭐⭐⭐ — 多尺度模型对比，训练动态分析充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，定义明确
- **实用性**: ⭐⭐⭐⭐ — 开源框架，实际可操作

<!-- RELATED:START -->

## 相关论文

- [Empowering Small VLMs to Think with Dynamic Memorization and Exploration](empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)
- [AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)

<!-- RELATED:END -->
