---
title: >-
  [论文解读] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning
description: >-
  [CVPR 2026][机器人][视觉推理] 提出DeepSketcher套件——包含31k高质量代码渲染的图文交错CoT数据集和一个自包含的Embedding Editor模型，使VLM无需外部工具即可在视觉嵌入空间直接生成"视觉思考"进行多模态推理。
tags:
  - "CVPR 2026"
  - "机器人"
  - "视觉推理"
  - "图文交错推理"
  - "视觉思考"
  - "嵌入编辑器"
  - "代码渲染"
---

# DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning

**会议**: CVPR 2026  
**arXiv**: [2509.25866](https://arxiv.org/abs/2509.25866)  
**代码**: [GitHub](https://github.com/MiliLab/DeepSketcher)  
**领域**: 机器人  
**关键词**: 视觉推理, 图文交错推理, 视觉思考, 嵌入编辑器, 代码渲染

## 一句话总结

提出DeepSketcher套件——包含31k高质量代码渲染的图文交错CoT数据集和一个自包含的Embedding Editor模型，使VLM无需外部工具即可在视觉嵌入空间直接生成"视觉思考"进行多模态推理。

## 研究背景与动机

"thinking with images"是VLM推理的新范式，通过让模型在推理过程中操作视觉输入（裁剪、缩放、画辅助线等），实现更深层的视觉理解。但现有方法面临三个核心矛盾：

1. **动作空间有限**：VILASR等方法只支持预定义的操作集（缩放、裁剪），灵活性差
2. **空间定位困难**：DeepEyes等通过RL学习操作，但依赖精确的坐标回归，训练数据噪声大
3. **训练难度极高**：Bagel等尝试统一生成和推理，但"想象力"空间太大，有效性未充分验证

DeepSketcher从代码渲染VQA数据出发，提出互补视角：所有图像通过代码渲染生成，视觉操作通过修改代码实现——精确、可复现、无空间定位噪声。

## 方法详解

### 整体框架

代码渲染图像 + 问题 → VLM生成推理文本 + 编辑指令 → Embedding Editor在视觉嵌入空间操作 → 更新的视觉嵌入注入上下文 → 继续推理 → 最终答案。

### 关键设计

1. **代码空间的数据构建**:
    - 功能：生成高质量的图文交错CoT训练轨迹
    - 核心思路：双Agent协作系统——Solver LLM进行推理并发出操作请求，Code Editor LLM修改渲染代码并重新生成图像，构成"推理→指令→代码修改→渲染→推理"的闭环
    - 设计动机：代码空间的编辑精确可控，避免了像素级操作的定位噪声和生成模型的不可控性

2. **Embedding Editor**:
    - 功能：在视觉嵌入空间直接执行视觉操作，无需外部工具调用
    - 核心思路：采用Q-Former风格的交叉注意力架构——视觉token作为Query，编辑指令隐藏状态经自适应池化后作为Key/Value，通过交叉注意力+FFN更新视觉嵌入
    - 设计动机：消除对代码执行、外部工具和重复图像编码的依赖，实现更灵活的"thinking with images"

3. **三阶段渐进训练**:
    - 功能：逐步解耦模型对GT视觉输入的依赖
    - 核心思路：Phase 1（推理预热，使用GT图像特征）→ Phase 2（Editor训练，L1损失对齐预测嵌入与GT编辑图像嵌入，冻结其他模块）→ Phase 3（联合适配，解冻LLM骨干适应Editor输出）
    - 设计动机：直接端到端训练会导致Editor产生噪声嵌入干扰推理，渐进训练确保每个组件稳定

### 损失函数 / 训练策略

- Phase 1: 标准自回归语言建模损失（仅监督文本token）
- Phase 2: L1嵌入重建损失 + 条件语言建模损失
- Phase 3: 与Phase 2相同目标，但解冻LLM骨干

## 实验关键数据

### 主实验（多模态推理基准）

| 模型 | MathVerse | MathVision | MathVista | LogicVista | WeMath | 平均 |
|------|-----------|------------|-----------|------------|--------|------|
| Qwen2.5-VL-7B | 41.1 | 27.0 | 68.2 | 39.8 | 34.3 | 42.1 |
| DeepEyes-7B | 42.2 | 26.6 | 70.1 | 47.7 | 38.9 | 45.1 |
| Mirage-7B (Inner Visual) | 27.3 | 28.6 | 63.7 | 40.7 | 16.7 | 35.4 |
| DeepSketcher-7B | 43.2 | 32.3 | 69.1 | 48.1 | 37.1 | 46.0 |

### 消融实验

| 阶段 | 设置 | MathVerse | WeMath | Indicator-500 |
|------|------|-----------|--------|---------------|
| Phase 2 | 纯文本基线 | 37.2 | 28.3 | 38.3 |
| Phase 2 | +Editor | 41.6 | 37.5 | 33.8 |
| Phase 3 | 纯文本基线 | 38.1 | 31.2 | 37.5 |
| Phase 3 | +Editor | 43.2 | 37.1 | 40.5 |

### 关键发现

- 在几何和计数任务上改进最显著（MathVision +5.3），涉及符号操作的任务改进较小
- 双Agent协作（Solver+Code Editor）比单独推理显著提升（GPT-4.1 pass@8: 0.72→0.80）
- Embedding Editor的差异图可视化显示编辑区域与指令高度一致

## 亮点与洞察

- 代码空间的数据构建是优雅的解决方案：精确、可复现、可验证，避免了坐标回归和图像生成的噪声
- Embedding Editor在嵌入空间操作的设计独特——不生成像素图像，而是直接修改视觉表示
- 作为"Inner Visual Thought VLM"中的最强方法，证明了内化视觉操作的可行性
- 31k数据集覆盖多学科（数学、物理、化学等），高质量且可扩展

## 局限与展望

- 代码渲染数据限制了应用范围（主要是结构化图形），自然图像场景未覆盖
- Embedding Editor的编辑质量仍不如GT代码渲染图像（Indicator-500上有差距）
- 比工具调用方法慢（因为需要经过Editor的前向传播）
- Phase 3解冻LLM后Indicator-500性能有时下降，说明适配不完全

## 相关工作与启发

- **vs VILASR/DeepEyes**: 预定义操作集+坐标回归；DeepSketcher动作空间开放且无需坐标
- **vs Mirage/Bagel**: 在压缩潜空间编辑图像；DeepSketcher在视觉token空间操作，保留更多语义信息
- **vs Visual Sketchpad**: 依赖外部工具执行；DeepSketcher内化整个操作链路

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 代码空间数据构建+嵌入空间视觉编辑，双重创新
- 实验充分度: ⭐⭐⭐⭐ 多基准覆盖，消融详细，但缺少自然图像场景评估
- 写作质量: ⭐⭐⭐⭐ 方法流程清晰，三阶段训练设计合理
- 价值: ⭐⭐⭐⭐ 为"thinking with images"范式提供了新的数据和模型路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)
- [\[ACL 2026\] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](../../ACL2026/robotics/govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)

</div>

<!-- RELATED:END -->
