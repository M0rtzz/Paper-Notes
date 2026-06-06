---
title: >-
  [论文解读] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging
description: >-
  [ICML 2025][多模态VLM][模型融合] 通过将数学推理 LLM 的参数与 VLM 的文本部分直接加权平均（模型融合），在无需训练的情况下将推理能力迁移到 VLM，并发现感知能力集中在前层、推理能力集中在中后层的层级分布规律。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "模型融合"
  - "VLM推理"
  - "感知与推理解耦"
  - "跨模态迁移"
  - "层级分析"
---

# Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging

**会议**: ICML 2025  
**arXiv**: [2505.05464](https://arxiv.org/abs/2505.05464)  
**代码**: [https://github.com/shiqichen17/VLM_Merging](https://github.com/shiqichen17/VLM_Merging)  
**领域**: 多模态VLM  
**关键词**: 模型融合, VLM推理, 感知与推理解耦, 跨模态迁移, 层级分析

## 一句话总结
通过将数学推理 LLM 的参数与 VLM 的文本部分直接加权平均（模型融合），在无需训练的情况下将推理能力迁移到 VLM，并发现感知能力集中在前层、推理能力集中在中后层的层级分布规律。

## 研究背景与动机

### 领域现状

**领域现状**：VLM 在视觉感知+语言任务上表现优秀，但在复杂多模态推理（如数学图表解读）上远落后于纯文本 LLM，部分原因是多模态推理数据匮乏。

**现有痛点**：提升 VLM 推理能力通常需要收集大量多模态推理数据并微调，成本高昂。

**核心矛盾**：感知和推理能力能否解耦？推理能力能否从 LLM 直接迁移到 VLM？

**本文目标**：探索模型融合作为跨模态能力迁移的途径。

**切入角度**：VLM 的文本部分与 LLM 共享相同架构和初始化，满足模型融合的连通子空间假设。

**核心 idea**：对 VLM 和数学 LLM 的文本参数做加权平均，实现零训练的推理能力迁移。

## 方法详解

### 整体框架
1. 选择共享 base model 的 VLM（如 LLaVA-NeXT/8B）和数学 LLM（如 Dart-Math）
2. 计算 task vector：$\tau = W_{\text{math}} - W_{\text{base}}$
3. 融合：$W_{\text{merged}} = W_{\text{VLM}} + \alpha \cdot \tau$
4. 直接使用融合后的模型，无需训练

### 关键设计

1. **跨模态模型融合**:

    - 功能：将数学 LLM 的 task vector 加到 VLM 的文本部分参数上
    - 核心思路：VLM 和 LLM 从相同 base model 微调，参数空间相连，加权平均可传递能力
    - 设计动机：推理能力应编码在文本处理层，视觉编码器不变即可保持感知能力

2. **层级能力分析（Knockout Analysis）**:

    - 功能：逐层屏蔽融合参数，观察感知和推理的变化
    - 核心发现：(a) 感知能力集中在前层（early layers）; (b) 推理能力集中在中后层; (c) 融合后推理能力扩展到所有层，但感知分布不变
    - 设计动机：理解融合如何在参数空间中影响不同能力

### 损失函数 / 训练策略
- 完全无训练（training-free）
- 融合系数 $\alpha$ 通常在 0.3-0.7 之间最优

## 实验关键数据

### 主实验

| 模型 | MathVista (Math) | MathVerse (Vision-Only) | 感知任务 |
|------|-----------------|------------------------|---------|
| LLaVA-NeXT | 38.2 | 21.3 | 78.5 |
| + Dart-Math 融合 | **41.8** (+3.6) | **22.7** (+1.4) | 78.1 (-0.4) |
| + MetaMath 融合 | 40.9 (+2.7) | 22.0 (+0.7) | 78.3 (-0.2) |

### 消融实验

| 配置 | Math↑ | Perception | 说明 |
|------|-------|-----------|------|
| 仅前层融合 | +0.8 | 不变 | 前层主要编码感知 |
| 仅中后层融合 | +3.1 | 不变 | 推理主要在中后层 |
| 全层融合 | +3.6 | 微降 | 最佳但感知略受影响 |
| $\alpha=0.3$ | +2.1 | 不变 | 保守融合 |
| $\alpha=0.7$ | +3.6 | -0.8 | 激进融合 |

### 关键发现
- 模型融合可在多个 VLM（LLaVA、Idefics2、InternVL2）和多个数学 LLM 上一致提升推理
- 感知和推理在参数空间中确实大致解耦——前者在前层，后者在中后层
- 融合改变了推理的层分布（扩展到所有层），但感知分布保持不变

## 亮点与洞察
- **零训练能力迁移**极其实用——只需参数加权平均，成本几乎为零
- 层级能力分布的发现对理解 VLM 内部机制有重要价值
- 为"模型融合作为多模态集成工具"开辟了新方向

## 局限与展望
- 要求 VLM 和 LLM 共享 base model，限制了适用范围
- 仅测试数学推理，其他类型推理（逻辑、常识）未验证
- 感知任务的轻微下降在更激进融合时可能加剧
- 未探索更复杂的融合策略（如 TIES、DARE）

## 相关工作与启发
- **vs 传统模型融合**: 之前只融合同类模型，本文首次跨模态融合
- **vs 推理数据微调**: 需要大量多模态推理数据，本文零数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 跨模态模型融合+层级分析的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 多VLM、多LLM、层级消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，分析深入
- 价值: ⭐⭐⭐⭐⭐ 实用且有洞察力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](../../ACL2025/multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/multimodal_vlm/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/multimodal_vlm/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ICML 2025\] Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning](diffusion-vla_generalizable_and_interpretable_robot_foundation_model_via_self-ge.md)
- [\[ICML 2025\] Enhancing Target-unspecific Tasks through a Features Matrix](enhancing_target-unspecific_tasks_through_a_features_matrix.md)

</div>

<!-- RELATED:END -->
