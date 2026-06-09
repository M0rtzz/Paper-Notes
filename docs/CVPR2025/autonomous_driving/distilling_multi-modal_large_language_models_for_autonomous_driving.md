---
title: >-
  [论文解读] Distilling Multi-modal Large Language Models for Autonomous Driving
description: >-
  [CVPR 2025][自动驾驶][多模态大语言模型] 本文提出DiMA框架，通过联合训练在多模态大语言模型（MLLM）和视觉端到端规划器之间进行知识蒸馏，设计了遮蔽重建、未来预测和场景编辑三种代理任务来丰富场景表示，推理时可丢弃LLM仅用视觉规划器…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "多模态大语言模型"
  - "知识蒸馏"
  - "端到端自动驾驶"
  - "长尾场景"
  - "视觉规划器"
---

# Distilling Multi-modal Large Language Models for Autonomous Driving

**会议**: CVPR 2025  
**arXiv**: [2501.09757](https://arxiv.org/abs/2501.09757)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 多模态大语言模型, 知识蒸馏, 端到端自动驾驶, 长尾场景, 视觉规划器

## 一句话总结
本文提出DiMA框架，通过联合训练在多模态大语言模型（MLLM）和视觉端到端规划器之间进行知识蒸馏，设计了遮蔽重建、未来预测和场景编辑三种代理任务来丰富场景表示，推理时可丢弃LLM仅用视觉规划器，在nuScenes上实现L2轨迹误差降低37%、碰撞率降低80%。

## 研究背景与动机

端到端自动驾驶系统在一般导航场景中表现良好，但在长尾场景（如三点掉头、超车等罕见操作）中表现不佳，主要因为训练数据集规模有限且缺乏多样性。大语言模型（LLM）在海量互联网数据上预训练，具备丰富的世界知识和链式推理能力，近期被用作驾驶规划器来提升长尾场景的泛化能力。

然而，LLM作为规划器面临两大问题：(1) 推理时计算开销巨大，不切实际；(2) 标准的图像token化策略使用冻结的预训练图像编码器生成密集、无结构的token嵌入，对于自动驾驶的结构化场景理解不够高效。

本文的核心思路是：训练时利用MLLM的世界知识，推理时丢弃LLM，仅保留高效的视觉规划器——实现"训练时有LLM，推理时无LLM"的最佳平衡。关键创新在于将视觉规划器的场景编码器作为MLLM的可训练tokenizer，使其学习到语言grounded的结构化表示。

## 方法详解

### 整体框架
DiMA由两个主要组件组成：(1) 视觉端到端规划器（场景编码器 + 规划Transformer）；(2) 多模态大语言模型（适配层 + LLM + 任务特定解码头）。场景编码器在两个组件间共享，既为规划Transformer提供特征，也作为MLLM的tokenizer。训练分两阶段：先预训练视觉规划器60个epoch，再联合训练视觉规划器和MLLM 30个epoch。

### 关键设计

1. **BEAM场景Token嵌入**:

    - 场景编码器将输入多视角图像序列编码为四种结构化token嵌入：
        - **B**EV token：鸟瞰图特征
        - **E**go token：自车交互特征，由可学习嵌入初始化
        - **A**gent token：周围智能体特征
        - **M**ap token：地图元素特征
    - 这些BEAM token通过各自的Q-former适配层投影到LLM的嵌入空间
    - 相比密集无结构的图像token，BEAM提供了物理意义明确的结构化输入
    - 关键区别：场景编码器与MLLM联合训练（不像TOKEN等方法冻结场景编码器）

2. **代理任务设计（Surrogate Tasks）**:

    - **遮蔽BEV重建**：随机遮蔽BEV token，要求MLLM从其余多模态序列的上下文中重建被遮蔽的BEV特征。使用L2损失监督，促使MLLM学习全局场景理解
    - **未来BEV预测**：给定当前BEV token，预测未来两个时间步的BEV token嵌入。使用L2损失监督，鼓励LLM学习时空线索
    - **场景编辑**：通过添加或删除周围智能体来增强场景，同时构建对应的问答对。添加时考虑地图约束和预测轨迹，创建新智能体token。该任务迫使模型学习周围智能体如何影响自车轨迹

3. **知识蒸馏与VQA**:

    - **特征蒸馏**：最小化LLM倒数第二层特征与规划Transformer特征之间的KL散度，使两个分支学习一致的表示
    - **视觉问答（VQA）**：基于DriveLM数据集训练，覆盖感知、预测、规划和行为四类QA。不在DriveLM中的nuScenes样本，用Llama3-70B生成类似的QA对
    - **LLM规划**：MLLM也预测自车和智能体轨迹

### 损失函数 / 训练策略
- 总损失 = L_planning + L_LLM + L_recon + L_future + L_distill
- 第一阶段（60 epoch）：仅训练视觉规划器（感知 + 预测 + 规划）
- 第二阶段（30 epoch）：联合训练，LLM用LoRA微调
- 使用LLaVA-v1.5-7B作为LLM基座
- 代理任务解码头使用3层Linear + ReLU

## 实验关键数据

### 主实验（标准化评估 - 全验证集）

| 数据集 | 指标 | 本文 (DiMA VAD-Base) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| nuScenes | Avg L2 (m) ↓ | 0.47 | 0.56 (PARA-Drive) | -16.1% |
| nuScenes | Avg Collision (%) ↓ | 0.06 | 0.17 (PARA-Drive) | -64.7% |
| nuScenes (targeted) | Avg L2 (m) ↓ | 0.71 | 0.91 (PARA-Drive) | -22.0% |
| nuScenes (targeted) | Avg Collision (%) ↓ | 0.05 | 0.14 (PARA-Drive) | -64.3% |

### VAD评估方式下的结果

| 方法 | Avg L2 ↓ | Avg Collision ↓ | FPS |
|------|----------|-----------------|-----|
| VAD-Base | 0.72 | 0.22 | 4.5 |
| DiMA (VAD-Tiny) | 0.38 | 0.15 | 16.8 |
| DiMA (VAD-Base) | 0.29 | 0.10 | 4.5 |
| DriveVLM-Dual | 0.31 | 0.10 | - |

### 长尾场景性能（零样本三点掉头）

| 方法 | Avg L2 ↓ | Collision ↓ |
|------|----------|-------------|
| VAD-Base | 1.57 | 0.00 |
| PARA-Drive | 1.29 | 5.33 |
| TOKEN | 1.18 | 4.00 |
| DiMA (VAD-Base) | 1.05 | 0.00 |

### 消融实验

| 配置 | Avg L2 ↓ | Avg Collision ↓ | 说明 |
|------|---------|--------|------|
| VAD-Tiny baseline | 0.60 | 0.29 | 无MLLM |
| + VQA + BEV tokens | 0.62 | 0.26 | 仅BEV token反而略退步 |
| + All BEAM tokens | 0.52 | 0.21 | 结构化token显著提升 |
| + Distillation | 0.48 | 0.19 | 特征蒸馏有效 |
| + Masked recon | 0.42 | 0.18 | 遮蔽重建进一步提升 |
| + Future pred | 0.39 | 0.16 | 未来预测任务有效 |
| + Scene editing (完整) | 0.38 | 0.15 | 场景编辑带来最终提升 |

### 关键发现
- 仅使用BEV token训练MLLM效果不稳定，必须使用全部BEAM结构化token
- 每个代理任务都逐步贡献了性能提升，场景编辑任务虽然最后加入但提升显著
- DiMA (VAD-Tiny)在性能上超越了VAD-Base，同时快4倍
- 三点掉头是零样本场景（仅在验证集出现），DiMA仍能取得最佳性能，验证了世界知识迁移的有效性
- 推理时不需要LLM，与基线视觉规划器保持相同的FPS

## 亮点与洞察
- "训练时有LLM，推理时无LLM"是非常务实的设计哲学，完美平衡了知识利用和效率
- BEAM结构化token作为MLLM输入的设计远优于密集图像token，物理意义明确
- 场景编辑任务的设计很有创意：通过虚拟添加/删除智能体，迫使模型学习因果关系
- 联合训练而非冻结场景编码器是关键，使得视觉表示与语言知识真正融合
- 长尾场景的显著提升（44%的L2误差降低）证明了LLM世界知识的迁移价值

## 局限与展望
- 开环评估（open-loop）的局限性：预测轨迹好不代表闭环驾驶表现好
- 训练流程较复杂：两阶段训练，需要DriveLM数据集和额外生成的QA对
- LLaVA-v1.5-7B作为基座LLM并非最新最强，换用更强的LLM可能进一步提升
- 场景编辑目前仅支持添加/删除车辆，未涉及行人、天气变化等更复杂的编辑
- 缺少计算成本分析：联合训练MLLM的训练开销未详细说明
- VQA结果仅提供了定性展示，缺少与其他VQA方法的定量对比

## 相关工作与启发
- 与TOKEN的对比：TOKEN冻结场景编码器作为tokenizer，而DiMA联合训练，效果更好
- 与DriveVLM的对比：DriveVLM使用密集视觉token，DiMA使用结构化BEAM token，且DiMA推理时不需要LLM
- 与OmniDrive的对比：OmniDrive推理效率低且碰撞率高，DiMA在两方面都优于它
- 代理任务的设计思想来源于自监督学习（MAE的遮蔽重建、时序预测等），但创新地与LLM蒸馏结合
- 该蒸馏框架可推广到其他需要"训练时用大模型，推理时要快"的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ BEAM结构化token和代理任务设计有新意，但蒸馏的宏观思路并非首创
- 实验充分度: ⭐⭐⭐⭐⭐ 多种评估协议、长尾场景分析、详细消融实验非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，但部分段落信息密度较高需要反复阅读
- 价值: ⭐⭐⭐⭐⭐ 解决了LLM在自动驾驶中"好但慢"的核心矛盾，nuScenes SOTA，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [\[ECCV 2024\] Navigation Instruction Generation with BEV Perception and Large Language Models](../../ECCV2024/autonomous_driving/navigation_instruction_generation_with_bev_perception_and_large_language_models.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[CVPR 2025\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](../../AAAI2026/autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)

</div>

<!-- RELATED:END -->
