---
title: >-
  [论文解读] Visual Intention Grounding for Egocentric Assistants
description: >-
  [ICCV 2025][多模态][视觉意图定位] 提出首个面向**自我中心视觉意图定位**（egocentric visual intention grounding）的任务和数据集 **EgoIntention**（26K 图像 + 52K 意图描述 + 89K 边界框），揭示现有 MLLM 在隐式意图推理和第一人称视觉定位上的重大不足，并提出 **Reason-to-Ground (RoG)** 指令微调方法，通过解耦意图推理和物体定位显著提升性能。
tags:
  - ICCV 2025
  - 多模态
  - 视觉意图定位
  - 第一人称视角
  - 自我中心视觉
  - 物体功能推理
  - 指令微调
---

# Visual Intention Grounding for Egocentric Assistants

**会议**: ICCV 2025  
**arXiv**: [2504.13621](https://arxiv.org/abs/2504.13621)  
**代码**: 待公开  
**领域**: multimodal_vlm  
**关键词**: 视觉意图定位, 第一人称视角, 自我中心视觉, 物体功能推理, 指令微调

## 一句话总结

提出首个面向**自我中心视觉意图定位**（egocentric visual intention grounding）的任务和数据集 **EgoIntention**（26K 图像 + 52K 意图描述 + 89K 边界框），揭示现有 MLLM 在隐式意图推理和第一人称视觉定位上的重大不足，并提出 **Reason-to-Ground (RoG)** 指令微调方法，通过解耦意图推理和物体定位显著提升性能。

## 研究背景与动机

### 核心场景

想象一个穿戴式 AI 助手：用户在凌乱的工作室中寻找坐处来整理工具，或一个孩子想要够到厨房水槽。助手需要在**无需用户明确指出物体名称**的情况下，根据上下文意图找到合适的物体（如椅子）。

### 传统视觉定位 vs 意图定位

| 维度 | 传统视觉定位 | 视觉意图定位 |
|------|------------|------------|
| 视角 | 第三人称 | 第一人称（自我中心） |
| 查询 | 明确物体描述（"白色椅子"） | 隐式意图（"收拾手机和行李"） |
| 推理 | 词到物体的直接匹配 | 需推理物体功能/用途 |
| 挑战 | 物体识别 | 遮挡、动态视角 + 功能推理 |

### 关键挑战

**误导性显式提及**：意图句子中可能显式提到某些物体但它们并非目标——如"收拾我的手机和行李"中提到"手机"，但目标是**手提包**（用来装这些东西）

**非常规物体用途**：椅子通常用来坐，但在"够到水槽"场景中用来**踩着增高**——需要理解物体的**可供性**（affordance）

**第一人称视角困难**：运动模糊、小物体、透视畸变等

### 现有方法的不足

**两阶段方案**（GPT-4 推理 + GroundingDINO 检测）：两个模型在不同特征空间操作，存在模态对齐不一致，可能幻觉出不存在的物体

**MLLMs**：主要为第三人称视觉定位设计，缺少将自我中心视觉与意图句子连接的训练数据

## 方法详解

### 整体框架：EgoIntention 数据集

#### 数据来源

- 图像来自 **Ego4D**（最大真实世界自我中心视觉数据集）
- 边界框标注继承 **PACO-Ego4D**（物体部件和属性数据集）
- 在此基础上增添精心策划的人类意图描述

#### 数据构建三阶段流程

**Stage 1: 意图句子生成**（GPT-4）
- 两类意图描述：
    - **上下文意图**（context-aware）：符合环境预期——"我注意到桌腿晃动需要修理"→ 锤子。通过率 97.2%
    - **非常规意图**（uncommon）：非典型用途——"下雨时用背包临时遮雨"。通过率 74.1%

**Stage 2: 人工验证**（Amazon MTurk）
- 评估语义有效性和真实世界适用性
- 辅以 GPT-4 验证器（与人类判断 92% 一致）

**Stage 3: 替代物体标注**
- 针对意图的主观性，标注多个可能满足意图的物体（如"桌面装饰"→ 花盆/瓶子/杯子均可）
- 补充边界框，同样需要通过双重验证

#### 数据集统计

| 划分 | 图像数 | 上下文边界框 | 非常规边界框 |
|------|-------|------------|------------|
| 训练 | 15,667 | 25,772 | 25,933 |
| 验证 | 825 | 1,402 | 1,366 |
| 测试 | 9,892 | 17,699 | 17,669 |
| **总计** | **26,384** | **44,873** | **44,968** |

### 核心方法：Reason-to-Ground (RoG) 指令微调

#### 设计动机

观察实验发现：
- **先推理再检测（R-D）远优于先检测再推理（D-R）**：R-D 的 P@0.5 为 46.6% vs D-R 的 21.1%（差距 25%）
- 原因：先用 GPT-4 缩小目标范围到 1-2 个类别，GroundingDINO 就能更精准检测
- 但两阶段方案的特征空间不一致是瓶颈

#### RoG 两阶段解耦

传统方法直接输入意图句子 + `<ref>` token → 输出边界框

RoG 将任务分解为：
1. **意图推理**：`<reason>` token + 隐式意图句子 → 模型输出**目标物体类别**
2. **物体定位**：`<ref>` token + 第一阶段推理出的显式物体描述 → 模型输出**边界框**

核心优势：防止模型将意图句子中**显式提及但非目标的物体**直接映射到边界框。

#### 训练策略

- 混合训练数据：RefCOCO/+/g（传统 REC）+ EgoIntention（意图定位）
- 使用 LoRA 高效微调
- Model-agnostic：适用于 MiniGPTv2、Qwen-VL 等

## 实验关键数据

### 主实验：零样本评估（Table 3）

| 方法 | 上下文 P@0.5 | 非常规 P@0.5 | 总体 P@0.5 |
|------|-------------|-------------|-----------|
| D-R (GroundingDINO→GPT4) | 21.1 | 14.6 | 17.8 |
| R-D (GPT4→GroundingDINO) | 46.6 | 23.6 | 35.1 |
| CogVLM-grounding | 3.4 | 2.4 | 2.9 |
| Groma | 4.8 | 4.3 | 4.5 |
| MiniGPT-v2 | 18.8 | 15.7 | 17.2 |
| Qwen-VL | 26.3 | 12.6 | 19.4 |

**关键发现**：定位专用 MLLM（CogVLM、Groma）几乎完全失败——缺乏意图推理能力。

### RoG 微调结果（Table 4）

| 模型 | 方法 | RefCOCO val | RefCOCO+ val | EgoIntention 上下文 | EgoIntention 非常规 | 总体 |
|------|------|------------|-------------|-------------------|-------------------|------|
| MiniGPTv2 | Zero-shot | 87.37 | 79.00 | 18.73 | 15.72 | 17.22 |
| MiniGPTv2 | Naive SFT | 86.60 | 78.98 | 41.31 | 36.92 | 39.11 |
| MiniGPTv2 | **RoG SFT** | **87.83** | **79.76** | **45.06** | **40.21** | **42.64** |
| Qwen-VL | Zero-shot | 89.32 | 83.18 | 26.30 | 12.60 | 19.45 |
| Qwen-VL | **RoG SFT** | 89.26 | 83.29 | **38.25** | **31.56** | **34.91** |

**核心结果**：RoG 相比 Naive SFT 在 EgoIntention 上提升 3.5 P@0.5，且在 RefCOCO 系列上也略有提升——不会损害传统定位能力。

### 消融实验：训练数据组合（Table 5 & 6）

| 训练数据 | 方法 | RefCOCO val | EgoIntention 上下文 | EgoIntention 非常规 |
|---------|------|------------|-------------------|-------------------|
| 仅 EgoInt. | Naive SFT | 66.53 (灾难性遗忘!) | 38.26 | 35.53 |
| RefCOCO/+/g + EgoInt. | Naive SFT | 87.48 | 41.07 | 36.91 |
| 全部数据 | Naive SFT | 86.60 | 41.31 | 36.92 |
| 全部数据 | **RoG SFT** | **87.83** | **45.06** | **40.21** |

**关键发现**：
1. 仅用 EgoInt. 微调导致 RefCOCO 性能暴跌 20%+——典型的灾难性遗忘
2. 混合 RefCOCO 数据保持传统 REC 能力 + 提升意图定位
3. RoG 解耦策略在所有数据组合中都优于 Naive SFT
4. RoG 还提升了显式物体查询的 EgoIntention 性能（44.26% → 47.50%）

## 亮点与洞察

1. **任务定义的开创性**：首次系统定义并研究自我中心视觉意图定位，明确区分传统物体定位与意图驱动定位
2. **数据集设计精良**：双类型意图（上下文+非常规）+ 多替代物体标注 + 三阶段验证，充分捕捉意图的多义性
3. **RoG 的简洁有效**：仅通过引入一个 `<reason>` token 和两阶段解耦，就显著提升了意图定位，同时保持甚至提升传统定位能力
4. **诊断性洞察**：先推理再检测远优于先检测再推理，说明**缩小搜索空间**是视觉意图理解的关键

## 局限性

1. **数据集规模**：26K 图像在深度学习标准下偏小，可能限制复杂意图推理的学习
2. **GPT-4 依赖**：意图生成和验证依赖 GPT-4，引入偏见和成本
3. **单一模态输入**：仅处理静态图像，真实自我中心场景多为视频流
4. **未探索多轮交互**：真实 AI 助手可能需要与用户对话澄清意图
5. **MiniGPTv2 vs Qwen-VL 差异**：RoG 对 Qwen-VL 效果较弱，可能因其指令遵循能力较差

## 相关工作与启发

- **与 IntentionVG 的区别**：IntentionVG 的第一人称图像是固定视角对准物体的，EgoIntention 使用 Ego4D 真实自我中心视频帧（运动模糊、小物体、畸变）
- **与 COCO-Tasks 的区别**：COCO-Tasks 使用短语级查询，EgoIntention 使用自由形式自然语言 + 多意图标注
- **启发**：RoG 的"先推理再行动"范式可推广到其他需要隐式推理的视觉任务（如导航中理解"找个安全的地方停车"）

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐⭐ — 新任务 + 新数据集 + 新方法，体系完整  
**实用性**: ⭐⭐⭐⭐ — 直接面向穿戴式 AI 助手场景  
**实验深度**: ⭐⭐⭐⭐ — 零样本+微调+消融全覆盖，6个模型8个数据集  
**写作质量**: ⭐⭐⭐⭐ — 场景引入生动，方法动机清晰

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[CVPR 2026\] HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](../../CVPR2026/multimodal_vlm/hammer_harnessing_mllm_via_cross-modal_integration_for_intention-driven_3d_affor.md)
- [\[ACL 2025\] Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users](../../ACL2025/multimodal_vlm/evaluating_multimodal_language_models_as_visual_assistants_for_visually_impaired.md)

<!-- RELATED:END -->
