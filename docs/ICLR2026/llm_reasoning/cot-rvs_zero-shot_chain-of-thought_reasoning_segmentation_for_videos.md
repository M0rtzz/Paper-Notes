---
title: >-
  [论文解读] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos
description: >-
  [ICLR 2026][LLM推理][推理视频分割] 提出CoT-RVS，一种完全无训练的多智能体框架，利用预训练MLLM的零样本CoT推理能力进行时间-语义关联分析与关键帧选取，在推理视频分割任务上大幅超越微调方法（Refer-DAVIS J&F 79.1 vs 71.2，ReasonVOS J&F 65.5 vs 49.9）。
tags:
  - ICLR 2026
  - LLM推理
  - 推理视频分割
  - Chain-of-Thought
  - 零样本
  - 关键帧选择
  - 多模态大模型
---

# CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos

**会议**: ICLR 2026  
**arXiv**: [2505.18561](https://arxiv.org/abs/2505.18561)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 推理视频分割, Chain-of-Thought, 零样本, 关键帧选择, 多模态大模型

## 一句话总结

提出CoT-RVS，一种完全无训练的多智能体框架，利用预训练MLLM的零样本CoT推理能力进行时间-语义关联分析与关键帧选取，在推理视频分割任务上大幅超越微调方法（Refer-DAVIS J&F 79.1 vs 71.2，ReasonVOS J&F 65.5 vs 49.9）。

## 研究背景与动机

- **领域现状**: 推理视频分割(Reasoning VOS)要求模型根据复杂隐式文本查询（如"哪个球员投了三分球"）生成对应目标的视频掩码序列，属于视频理解中最具挑战性的任务之一
- **现有痛点**: 现有方法（VISA/VideoLISA/HyperSeg）微调MLLM生成分割token，但在时间敏感查询下表现不佳。核心原因在于这些方法缺乏帧间时间推理能力——它们关注帧内语义理解，但无法有效推理"哪个时间段发生了什么"
- **核心矛盾**: 图像域的CoT推理分割（Seg-Zero/ThinkFirst）已取得成功，但视频域需要额外的时间维度"思考"能力。直接从图像扩展到视频并不可行，因为视频中的目标对象会随时间发生遮挡、运动、出现或消失
- **本文切入点**: 不做任何微调，而是利用GPT-4o/Gemma3等预训练MLLM的零样本CoT能力，设计任务特定prompt引导模型进行时间-语义推理。这与推理时计算(test-time compute)的趋势高度一致
- **核心idea**: 让MLLM通过CoT自问自答的方式分析关键帧候选：从语义维度（帧内有哪些对象匹配查询）和时间维度（哪个帧的目标最容易观察）两个角度建立关联，最终选出每个实例的最佳关键帧

## 方法详解

### 整体框架

三模块协作的多智能体框架，将视频推理分割分解为关键帧选择→帧级分割→视频追踪三个子任务：
- MLLM关键帧选择器 $\mathcal{F}_{key}$: 负责时间-语义关联推理
- 推理图像分割模型 $\mathcal{F}_{seg}$: 在关键帧上生成key mask
- 视频处理器 $\mathcal{F}_{vid}$ (SAM2): 沿时间轴追踪掩码

### 关键设计

**1. MLLM关键帧选择器（核心创新）**:
- 均匀采样 $T' = \lfloor T/\xi \rfloor$ 个关键帧候选
- 对每个候选帧自动合成CoT问答序列，从粗到细：通用语义（"帧中有什么"）→ 时间推理（"是否为更好的关键帧"）→ 细节确认（"是否有新的目标对象"）
- 最终输出包含：目标实例列表 + 对应关键帧索引 + 帧内目标描述（如"穿黑色球衣正在投篮的球员"）
- 设计为Reasoning VIS框架（k≥1个实例），Reasoning VOS是k=1的特例
- 兼容闭源(GPT-4o)和开源(Gemma3-12B/LLaVA1.5-7B)MLLM

**2. 推理图像分割**: 使用Seg-Zero等在关键帧上根据文本描述生成key mask

**3. 视频处理器**: 使用SAM2将key mask追踪到所有帧，多实例通过贪心后处理保证不重叠：$m_{i,t} = \bigcap_{j=1}^{i-1} \neg m_{j,t} \cap \hat{m}_{i,t}$

**4. 在线推理扩展(Online CoT-RVS)**:
- 每 $\xi$ 帧周期性调用MLLM判断当前帧是否应替换现有关键帧
- 贪心更新策略：新帧更好→更新目标和掩码；否则→沿用历史
- 首次实现流式推理视频分割，适合实时视频流场景

### 训练策略

完全无训练——所有模块均使用预训练权重，无任何微调。

## 实验关键数据

### 主实验表

| 数据集 | 指标 | CoT-RVS(GPT-4o) | GLUS | SAMWISE | VideoLISA(Po) | VISA-13B |
|--------|------|:----------------:|:----:|:-------:|:-------------:|:--------:|
| MeViS | J&F | **52.2** | 51.3 | 49.5 | 44.4 | 44.5 |
| Refer-DAVIS | J&F | **79.1** | — | 70.6 | 68.8 | 70.4 |
| ReasonVOS | J&F | **65.5** | 49.9 | — | 47.5 | — |

### 消融实验表

| 配置 | MeViS J&F | Refer-DAVIS J&F | 说明 |
|------|:---------:|:---------------:|------|
| CoT-RVS-GPT-4o | 52.2 | 79.1 | 最强闭源配置 |
| CoT-RVS-Gemma3-12B | 44.2 | 74.6 | 最强开源配置 |
| CoT-RVS-LLaVA1.5-7B | 45.9 | 73.9 | 最轻量开源配置 |
| w/o CoT(直接prompt) | — | ~65 | CoT推理带来约14个点的提升 |
| Online CoT-RVS(GPT-4o) | — | 77.8 | 在线版本性能接近离线 |

### 关键发现

- ReasonVOS上比GLUS高+15.6个点，时间敏感查询优势极为突出（如投三分球、特定动作时刻），验证了时间推理的核心价值
- 开源Gemma3版本在无API成本下仍超越VISA/VideoLISA等微调方法，说明预训练MLLM的通用推理能力被低估
- 在线版本(Online CoT-RVS)与离线版本差距仅约1.3个点，但支持流式处理，实用性显著

## 亮点与洞察

- **完全无训练的突破**: 首个兼容闭源/开源MLLM的零样本推理VOS框架，挑战了"推理分割必须微调"的范式
- **时间推理的价值**: CoT过程让MLLM真正"思考"帧间的时间语义关系，这是微调分割token方法本质上缺失的能力
- **模块化设计的灵活性**: 分割模型(LISA/Seg-Zero)和视频处理器(SAM2/Cutie)可灵活替换，未来各模块进步可直接带来系统提升
- **在线扩展的实用性**: 在线推理VOS方案极少见，对实时监控、自动驾驶等场景有意义

## 局限与展望

- GPT-4o版本推理成本高（每个视频多次API调用），大规模应用不现实
- 开源版本(Gemma3)与闭源(GPT-4o)在MeViS上差8个点，说明MLLM视觉推理能力仍是瓶颈
- 均匀帧采样可能错过关键运动帧，自适应采样策略（如运动检测引导）可能更优
- 多实例贪心后处理较简单，无法处理严重遮挡场景
- 未探索将CoT推理模块与分割/追踪模块端到端联合训练的可能性

## 相关工作与启发

- 相比VISA/VideoLISA等微调方法，CoT-RVS用零样本推理替代微调，代表了不同的技术路线
- 与Seg-Zero/ThinkFirst(图像域CoT分割)一脉相承，但增加了时间维度推理
- 体现了test-time compute趋势在视觉任务中的应用前景

## 评分

- 新颖性: ⭐⭐⭐⭐ (零样本CoT应用于视频推理分割，概念新颖)
- 实验充分度: ⭐⭐⭐⭐ (4数据集+多MLLM+在线扩展+消融)
- 写作质量: ⭐⭐⭐⭐ (框架描述清晰，公式完整)
- 价值: ⭐⭐⭐⭐ (证明无训练CoT在视频分割中的可行性)
- **vs VISA/VideoLISA**: 这些方法微调MLLM生成分割token，CoT-RVS完全无训练
- **vs Seg-Zero/ThinkFirst**: CoT推理图像分割方法，本文首次扩展到视频时间域
- **vs SAM2**: 作为即插即用的视频追踪模块，展示了与推理系统组合的潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ 零样本CoT用于视频时间推理是首创
- 实验充分度: ⭐⭐⭐⭐ 4个benchmark + 在线扩展 + 模块替换消融
- 写作质量: ⭐⭐⭐⭐ 示例生动、框架描述清晰
- 价值: ⭐⭐⭐⭐ 无训练范式有实用意义，但依赖强MLLM

<!-- RELATED:START -->

## 相关论文

- [GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization](../../CVPR2026/llm_reasoning/graze_grounded_refinement_and_motion-aware_zero-shot_generation.md)
- [Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)
- [SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)

<!-- RELATED:END -->
