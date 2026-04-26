---
title: >-
  [论文解读] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method
description: >-
  [ICLR 2026][目标检测][视觉定位推理] 提出 TreeBench（首个可追溯视觉推理基准，405道高挑战 VQA，OpenAI-o3 仅 54.87%）和 TreeVGR（通过双 IoU 奖励的强化学习联合监督定位与推理的训练范式），7B 模型在 V\*Bench +16.8、MME-RealWorld +12.6、TreeBench +13.4，证明可追溯性是推进视觉推理的关键。
tags:
  - ICLR 2026
  - 目标检测
  - 视觉定位推理
  - 可追溯证据
  - 二阶推理
  - TreeBench
  - 强化学习
  - Dual IoU
---

# Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method

**会议**: ICLR 2026  
**arXiv**: [2507.07999](https://arxiv.org/abs/2507.07999)  
**代码**: [GitHub](https://github.com/Haochen-Wang409/TreeVGR)  
**领域**: 视觉推理 / 多模态评测  
**关键词**: 视觉定位推理, 可追溯证据, 二阶推理, TreeBench, 强化学习, Dual IoU

## 一句话总结

提出 TreeBench（首个可追溯视觉推理基准，405道高挑战 VQA，OpenAI-o3 仅 54.87%）和 TreeVGR（通过双 IoU 奖励的强化学习联合监督定位与推理的训练范式），7B 模型在 V\*Bench +16.8、MME-RealWorld +12.6、TreeBench +13.4，证明可追溯性是推进视觉推理的关键。

## 研究背景与动机

**领域现状**：OpenAI-o3 开创了"用图像思考"（thinking with images）的范式——在推理过程中动态引用、放大任务相关的视觉区域，已展示出超越纯文本推理的潜力。然而，目前没有任何基准能全面评估这种能力。

**现有痛点**：
1. POPE、MMBench、SEED-Bench 等经典基准忽略精细定位和可验证的推理链
2. V\*Bench 仅支持简单空间查询（"A是否在B左边？"），且基于COCO图像存在数据泄露风险
3. MME-RealWorld、HR-Bench 支持高分辨率输入，但缺乏可追溯证据和复杂推理
4. 现有 RL 训练方法（DeepEyes、Pixel-Reasoner 等）仅监督最终答案，不监督中间定位过程

**核心矛盾**：没有基准同时满足三个关键要求——聚焦视觉感知（密集场景中识别细微目标）、可追溯证据（评估推理链中每步的定位质量）、二阶推理（超越简单定位的物体交互和空间层级推理）。训练方面，现有方法无法量化"定位-回答"框架中定位的实际贡献。

**本文方案**：双管齐下——TreeBench 建立评测标准，TreeVGR 建立训练方法，二者共同推进"用图像思考"能力的评估和提升。

## 方法详解

### 整体框架

**TreeBench 构建流程**：SA-1B 采样 1K 高质量图片（优选密集物体场景）→ 8位 LMM 专家标注 → 3阶段质量控制 → 405 道高挑战 VQA 对（含目标实例的 bounding box 标注）

**TreeVGR 训练流程**：冷启动 SFT 初始化 → 带可追溯证据的强化学习后训练

### 关键设计1：TreeBench 的三大评测原则

**1) 聚焦视觉感知（Focused Visual Perception）**：所有问题聚焦于复杂真实场景中的极小目标——目标实例平均仅占图像面积的 **3.05%**。要求模型通过详细、精确、唯一的文本描述识别细微目标。

**2) 可追溯证据（Traceable Evidence）**：不仅评最终答案准确率，还评推理链中生成的 bounding box 的质量（mIoU）。通过对比预测框和 ground-truth 框，可以精确诊断错误来源——是理解错误还是定位失败。

**3) 二阶推理（Second-Order Reasoning）**：超越简单的"什么/在哪"查询，包含5类感知任务（属性/材质/物理状态/目标检索/OCR）和5类推理任务（视角变换/排序/接触遮挡/空间包含/比较），其中视角变换（"从人A的视角，物体B在哪个方向？"）是最具挑战性的类别。

### 关键设计2：TreeVGR 的双 IoU 奖励机制

TreeVGR 的总奖励由三部分组成：

$$R = R_{\text{acc}} + R_{\text{format}} + R_{\text{IoU}}$$

其中双 IoU 奖励 $R_{\text{IoU}}$ 是核心创新，同时优化召回率和精确率：

**召回项**（每个 GT 框至少被一个预测框匹配）：

$$R_{\text{IoU}}^{\text{R}} = \frac{1}{M} \sum_{k=1}^{M} \max_i \text{IoU}(\hat{b}_i, b_k)$$

**精确项**（每个预测框至少匹配一个 GT 框，防止模型滥发框）：

$$R_{\text{IoU}}^{\text{P}} = \frac{1}{N} \sum_{i=1}^{N} \max_k \text{IoU}(b_k, \hat{b}_i)$$

$$R_{\text{IoU}} = \frac{1}{2}(R_{\text{IoU}}^{\text{R}} + R_{\text{IoU}}^{\text{P}})$$

这种双向约束解决了单向召回奖励导致模型"穷举所有可能框"的问题。

### 关键设计3：冷启动初始化

直接用 RL 训练视觉定位推理效率极低（DeepEyes 需 32 块 H100 训练 48 小时）。本文先用精心构造的 SFT 数据进行冷启动——每个样本包含图像、问题、带 bounding box 的推理轨迹和最终答案——确保模型在 RL 前已具备基本的"定位-推理"能力。这种初始化策略大幅降低了 RL 的计算成本。

## 实验关键数据

### 主实验：TreeBench 各类别性能

| 模型 | Overall | 属性 | 物理状态 | 目标检索 | OCR | 视角变换 | 排序 | 接触遮挡 | 空间包含 | 比较 | mIoU |
|------|---------|------|----------|----------|-----|----------|------|----------|----------|------|------|
| o3-0416 | 54.8 | 69.0 | 69.2 | 65.2 | 68.8 | 79.4 | 22.4 | 38.6 | 61.0 | 86.2 | –† |
| Gemini-2.5-Pro | 54.1 | 51.7 | 61.5 | 56.5 | 75.0 | 83.8 | 20.0 | 36.8 | 65.9 | 86.2 | – |
| Qwen2.5-VL-72B | 42.2 | 65.5 | 69.2 | 56.5 | 56.3 | 48.5 | 11.8 | 33.3 | 51.2 | 72.4 | – |
| Qwen2.5-VL-7B | 37.0 | 55.2 | 53.8 | 56.5 | 62.5 | 27.9 | 20.0 | 35.1 | 39.0 | 44.8 | – |
| DeepEyes-7B | 37.5 | 62.1 | 53.8 | 65.2 | 68.8 | 51.5 | 11.8 | 24.6 | 36.6 | 51.7 | 30.0 |
| Pixel-Reasoner-7B | 39.0 | 58.6 | 61.5 | 65.2 | 50.0 | 48.5 | 14.1 | 31.6 | 39.0 | 44.8 | 35.7 |
| **TreeVGR-7B** | **50.4** | 65.5 | 53.8 | **82.6** | 68.8 | **63.3** | **22.4** | 36.8 | **61.0** | 69.0 | **44.0** |

### 消融实验：各基准提升对比

| 基准 | Qwen2.5-VL-7B（基线） | TreeVGR-7B | 提升幅度 |
|------|----------------------|------------|----------|
| TreeBench Overall | 37.0 | 50.4 | **+13.4** |
| V\*Bench Overall | 74.3 | 91.1 | **+16.8** |
| V\*Bench Attr. | 77.4 | 94.0 | +16.6 |
| V\*Bench Spatial | 69.7 | 87.0 | +17.3 |
| MME-RealWorld-Lite | 42.3 | 54.9 | **+12.6** |
| HR-Bench-4K | 72.1 | 77.1 | +5.0 |
| HR-Bench-8K | 68.8 | 73.1 | +4.3 |

### 核心发现

- **没有模型在 TreeBench 上超过 60%**：最强的 o3 也仅 54.87%，证明基准确实有挑战性
- **TreeVGR-7B 媲美 InternVL3-78B**：7B 模型通过定位-推理联合训练达到 78B 通用模型的水平
- **mIoU 高度相关于最终准确率**：TreeVGR 的 mIoU=44.0 显著优于 DeepEyes（30.0）和 Pixel-Reasoner（35.7），验证了精确定位对推理的促进作用
- **接触遮挡和排序是最难类别**：所有模型在这两类上表现最差（<25%），反映二阶推理的根本困难

## 亮点与洞察

- **"o3 不到 55%"的震撼**：当前最强多模态模型在精细视觉推理上仍然很弱——TreeBench 暴露了真实能力 gap
- **可追溯性 = 可验证性**：不只看最终答案，而是评估推理链每步的定位证据——使评测更可靠、更具诊断价值
- **双 IoU 奖励设计优雅**：同时约束召回和精确，避免了模型"穷举框"的 reward hacking 策略
- **冷启动+RL 范式高效**：相比 DeepEyes 的纯 RL 方案（32×H100, 48h），冷启动大幅降低了计算成本

## 局限与展望

- TreeBench 规模较小（仅 405 题），统计显著性受限
- TreeVGR 不实际裁剪和回看图像（仅文本空间定位），可能错失视觉细节
- 冷启动 SFT 数据的质量直接影响 RL 的上限，数据构造过程存在人工成本
- 二阶推理（视角变换/空间包含）的训练样本较少，RL 训练可能不充分
- 未探索多轮交互式定位推理的可能性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](../../ICCV2025/object_detection/visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[ICLR 2026\] Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)

<!-- RELATED:END -->
