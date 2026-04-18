---
title: >-
  [论文解读] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
description: >-
  [CVPR 2026][图像生成][多智能体协作] 提出 coDrawAgents，一个交互式多智能体对话框架（Interpreter-Planner-Checker-Painter），通过分而治之的增量布局规划、视觉上下文驱动的空间推理和显式错误纠正机制，大幅提升复杂场景下组合式文本到图像生成的忠实度。
tags:
  - CVPR 2026
  - 图像生成
  - 多智能体协作
  - 组合式图像生成
  - 布局规划
  - 文本-图像对齐
  - 扩散模型
---

# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**代码**: 暂无（论文称发表后公开）  
**领域**: 图像生成 / 组合式文本到图像生成  
**关键词**: 多智能体协作, 组合式图像生成, 布局规划, 文本-图像对齐, 扩散模型

## 一句话总结

提出 coDrawAgents，一个交互式多智能体对话框架（Interpreter-Planner-Checker-Painter），通过分而治之的增量布局规划、视觉上下文驱动的空间推理和显式错误纠正机制，大幅提升复杂场景下组合式文本到图像生成的忠实度。

## 研究背景与动机

现有 T2I 模型在处理多物体、多属性的复杂场景时面临三大核心挑战：

**布局复杂度爆炸**：全局布局规划中物体间关系复杂度为二次方，单一规划器难以捕获所有依赖

**缺乏视觉上下文**：大多数方法在纯文本空间预测布局，无法参照实际画面，导致空间不合理

**无显式纠错机制**：扩散模型在去噪早期即确定粗略结构，一旦引入空间错误便难以修正

现有方案（包括单智能体和固定流水线型多智能体系统）都缺乏闭环推理能力——不能在规划、验证和合成之间形成迭代反馈。

## 方法详解

### 整体框架

coDrawAgents 由四个专门化智能体组成闭环对话系统：

- **Interpreter（解释器）**：决定生成模式（layout-free 或 layout-aware），解析文本为结构化物体描述，按语义显著性排序并分组
- **Planner（规划器）**：在 layout-aware 模式下增量式布局推理
- **Checker（检查器）**：空间一致性和语义对齐验证 + 纠错
- **Painter（画家）**：逐步合成图像，提供视觉上下文

工作流程：Interpreter 判断场景复杂度 → 简单场景直接 Painter 生成 → 复杂场景进入 Planner-Checker-Painter 循环，按语义优先级逐层迭代。

### 关键设计

1. **分而治之的增量布局规划（Divide-and-conquer Planning）**：Interpreter 将物体按语义显著性分组（同一语义优先级的物体为一组），Planner 每次只处理一组物体的布局，将全局 $O(n^2)$ 复杂度分解为多轮 $O(k^2)$ 的局部问题。核心动机是降低 LLM 单次推理的空间关系复杂度。

2. **视觉化思维链（Visualization Chain-of-Thought, VCoT）**：Planner 使用 GPT-5 作为 MLLM，输入包括全局文本提示、当前优先级物体描述、历史布局、Painter 渲染的部分图像 $I_{i-1}$ 以及物体 grounding 信息。推理分三步：

    - **Canvas State Analysis**：分析当前画面中已有物体的空间分布
    - **Context-aware Planning**：基于世界知识推理新物体与已有场景的合理交互
    - **Physics Constraint Enforcement**：施加物理约束（避免悬浮、不合理接触等）

3. **两阶段检查-修正机制（Check-then-Refine）**：Checker 在每轮迭代中执行两级验证：

    - **单物体级别**：检查尺寸、比例、边界覆盖
    - **全局级别**：审查所有历史布局 $\{L_1, \ldots, L_i\}$ 中的跨物体冲突（重叠、遮挡顺序、尺度漂移），并逐步修正传播到后续布局

4. **即插即用 Painter**：支持任意 T2I（layout-free 模式，本文使用 Flux）和 L2I（layout-aware 模式，使用 3DIS）模型，无需额外训练。画面逐步演化为后续规划提供真实视觉上下文。

### 损失函数 / 训练策略

本框架为 training-free 的推理时方法，不涉及模型训练。各智能体通过精心设计的 prompt 和 CoT 引导，在推理时动态协作。Painter 使用现成的预训练 T2I/L2I 模型。

## 实验关键数据

### 主实验

| 数据集 | 指标 | coDrawAgents | 之前 SOTA | 提升 |
|--------|------|-------------|-----------|------|
| GenEval | Overall Score ↑ | **0.94** | 0.84 (GPT Image 1) | +0.10 |
| GenEval | Counting ↑ | **0.94** | 0.85 (GPT Image 1) | +0.09 |
| GenEval | Position ↑ | **0.95** | 0.75 (GPT Image 1) | +0.20 |
| GenEval | Color Attri. ↑ | **0.81** | 0.70 (UniWorld-V1) | +0.11 |
| DPG-Bench | Overall ↑ | **85.17** | 84.08 (SD3-Medium) | +1.09 |
| DPG-Bench | Relation ↑ | **92.92** | 90.87 (FLUX.1-dev) | +2.05 |

### 消融实验

| 配置 | DPG-Bench Overall ↑ | 说明 |
|------|---------------------|------|
| Layout-free mode only | 77.60 | 仅直接 T2I 生成 |
| + Layout-aware mode | 82.61 | 加入分而治之布局规划，+5.01 |
| + Visual context | 84.51 | Planner 利用画面上下文，+1.90 |
| + Checker (完整 coDrawAgents) | **85.17** | 显式纠错机制，+0.66 |

### 关键发现

- 框架效率出色：DPG-Bench 上平均每张图只需 Interpreter 1.00 次、Planner 1.52 次、Checker 1.62 次、Painter 1.95 次调用，远少于场景中平均物体数 2.79
- 在 GenEval 的 Position 子指标上达到 0.95，相比 GPT Image 1 的 0.75 提升巨大，证明增量式视觉 grounding 规划对空间精度的关键作用
- Checker 的跨迭代全局审查解决了累积误差传播问题

## 亮点与洞察

- **闭环 vs. 流水线**：区别于固定流水线系统，四个智能体形成真正的迭代对话，规划-检查-合成之间相互反馈
- **语义优先级分组**是关键创新——不仅降低了单步推理复杂度，还使同语义层级物体获得一致的布局处理
- VCoT 的三步推理（状态分析→上下文规划→物理约束）为 MLLM 布局规划提供了结构化且可解释的推理框架
- 即插即用设计使框架能自然受益于未来更强的 T2I/L2I 模型

## 局限与展望

1. 多智能体调用引入额外计算开销，推理时间高于单次生成方法
2. Painter 性能受底层 T2I/L2I 模型限制（如属性渲染偏差会传播）
3. Planner 和 Checker 依赖 MLLM，存在幻觉和过度自信问题
4. 目前仅支持 2D 场景，扩展到 3D 可控生成是重要方向

## 相关工作与启发

- 与 GoT（一次性全局推理所有 bbox）相比，coDrawAgents 的增量式局部规划在 GenEval 上大幅领先（0.94 vs. 0.64）
- 与 T2I-Copilot（多智能体固定流水线）相比，闭环对话机制带来 DPG-Bench 10+ 分提升
- 启发：多智能体系统的关键不在智能体数量，而在于闭环反馈和分治策略的结合

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将分而治之策略与视觉上下文驱动的闭环多智能体框架结合，系统性解决组合生成难题
- **实验充分度**: ⭐⭐⭐⭐ GenEval 和 DPG-Bench 双基准验证，消融完整，效率分析到位
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，框架设计逻辑性强
- **价值**: ⭐⭐⭐⭐ 在 GenEval 上取得 0.94 的突破性结果，training-free 且即插即用，实用价值高
