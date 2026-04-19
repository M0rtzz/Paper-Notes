---
title: >-
  [论文解读] Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning
description: >-
  [CVPR 2025][图像生成][扩散模型微调] 提出 DeMe，将扩散模型训练视为多任务学习问题——不同时间步是不同"任务"且存在梯度冲突，通过将时间步解耦为 N 个范围分别微调再合并回单一模型，在零推理开销下显著提升生成质量（CIFAR10 FID 从 4.42 降至 3.51）。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散模型微调
  - 时间步梯度冲突
  - 多任务学习
  - 模型合并
  - 知识保留
---

# Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning

**会议**: CVPR 2025  
**arXiv**: [2410.06664](https://arxiv.org/abs/2410.06664)  
**代码**: 有（论文中提及 GitHub）  
**领域**: 扩散模型  
**关键词**: 扩散模型微调, 时间步梯度冲突, 多任务学习, 模型合并, 知识保留

## 一句话总结
提出 DeMe，将扩散模型训练视为多任务学习问题——不同时间步是不同"任务"且存在梯度冲突，通过将时间步解耦为 N 个范围分别微调再合并回单一模型，在零推理开销下显著提升生成质量（CIFAR10 FID 从 4.42 降至 3.51）。

## 研究背景与动机

**领域现状**：扩散模型在所有时间步共享同一组参数进行训练。不同时间步实际上执行截然不同的任务——高噪声步骤生成全局结构，低噪声步骤精修细节。

**现有痛点**：不同时间步的梯度方向可能相互冲突（多任务学习中的经典问题）。这种冲突限制了单一模型在所有时间步上的最优性。直接为每个时间步维护独立模型虽能消除冲突但参数量爆炸。

**核心矛盾**：需要消除时间步间的梯度冲突以提升质量，但不能增加推理时的参数和计算量。

**本文目标** 在微调阶段解耦时间步冲突、训练完成后合并回单一模型，实现零推理开销的质量提升。

**切入角度**：分而治之——将时间步分成 N 个不重叠范围，每个范围从预训练 checkpoint 独立微调，消除跨范围梯度冲突，然后用 task vector 模型合并技术合并回单一模型。

**核心 idea**：将时间步解耦为 N 个范围分别微调（消除梯度冲突），通过一致性损失+概率采样+通道投影保留跨时间步知识，最终用 task vector 合并回一个模型。

## 方法详解

### 关键设计

1. **时间步范围解耦**：将 [0,T] 分成 N 个不重叠范围，每个范围从预训练模型独立微调。消除了跨范围的梯度冲突。

2. **三种知识保留技术**：(a) **一致性损失**：正则化当前模型与预训练模型的差异，防止灾难性遗忘全局知识；(b) **概率采样**：偶尔从完整时间步范围采样，保持跨步知识；(c) **通道投影**：可学习的通道映射（仅 1.06% 额外参数），增强阶段间信息传递。

3. **Task Vector 模型合并**：训练后计算每个范围模型与预训练模型的差值（task vector），加权合并回单一模型。权重通过网格搜索确定。零推理开销。

## 实验关键数据

| 模型 | 数据集 | 基线FID↓ | DeMe FID↓ | 提升 |
|------|--------|---------|----------|------|
| DDPM | CIFAR10 | 4.42 | **3.51** | -0.91 |
| DDPM | LSUN-Church | 10.69 | **7.27** | -3.42 |
| SD | COCO | 13.42 | **12.41** | -1.01 |
| SD | PartiPrompts CLIP | 29.78 | **30.10** | +0.32 |

所有设置一致提升，LSUN-Church 提升最大（-3.42 FID）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将扩散训练视为多任务学习的视角非常新颖，解耦-合并范式优雅
- 实验充分度: ⭐⭐⭐⭐ DDPM+SD+多数据集+消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 零推理开销的微调增强，直接可用于任何扩散模型

<!-- RELATED:START -->

## 相关论文

- [ToMA: Token Merge with Attention for Diffusion Models](../../ICML2025/image_generation/toma_token_merge_with_attention_for_diffusion_models.md)
- [MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](../../ICCV2025/image_generation/mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](../../AAAI2026/image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)
- [Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)

<!-- RELATED:END -->
