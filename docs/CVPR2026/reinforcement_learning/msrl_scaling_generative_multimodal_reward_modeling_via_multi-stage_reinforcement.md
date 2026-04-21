---
title: >-
  [论文解读] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning
description: >-
  [CVPR 2026][多模态奖励模型] 提出MSRL(Multi-Stage Reinforcement Learning)，通过多阶段RL扩展生成式多模态奖励建模——先在大规模文本偏好数据(400K)上做RL学习通用奖励推理能力，再经caption-based RL和跨模态知识蒸馏向多模态迁移，最后用少量多模态偏好数据微调适配，无需额外多模态标注即在VL-RewardBench上从66.6%提升到75.9%、GenAI-Bench上从70.2%到75.7%。
tags:
  - CVPR 2026
  - 多模态奖励模型
  - 生成式奖励
  - 多阶段RL
  - 跨模态知识蒸馏
  - 偏好对齐
---

# MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2603.25108](https://arxiv.org/abs/2603.25108)  
**代码**: https://github.com/wangclnlp/MSRL (有)  
**领域**: 强化学习 / 多模态奖励建模  
**关键词**: 多模态奖励模型, 生成式奖励, 多阶段RL, 跨模态知识蒸馏, 偏好对齐

## 一句话总结
提出MSRL(Multi-Stage Reinforcement Learning)，通过多阶段RL扩展生成式多模态奖励建模——先在大规模文本偏好数据(400K)上做RL学习通用奖励推理能力，再经caption-based RL和跨模态知识蒸馏向多模态迁移，最后用少量多模态偏好数据微调适配，无需额外多模态标注即在VL-RewardBench上从66.6%提升到75.9%、GenAI-Bench上从70.2%到75.7%。

## 研究背景与动机

**领域现状**：多模态奖励模型(MRM)对MLLM偏好对齐至关重要。近期从判别式MRM转向生成式MRM(CoT推理+文本输出奖励分数)，并引入RLVR进一步增强推理能力。

**核心瓶颈**：RLVR依赖人工标注的多模态偏好数据做可验证奖励→获取成本高昂→无法扩展。而文本偏好数据丰富(100万+)→能否利用文本数据来扩展多模态奖励建模？

**核心假设**：奖励推理的核心能力可从文本数据学习并跨模态迁移——偏好判断的"推理过程"在很大程度上是模态无关的。

**核心idea**：多阶段课程——Stage1文本RL(学习通用奖励推理)→Stage2 Caption-based RL+CMKD(迁移到多模态)→Stage3多模态RL(精调适配)。

## 方法详解

### 整体框架
SFT(学CoT格式和推理结构) → **Stage 1: 文本RL**(400K文本偏好数据，GRPO优化，冻结视觉编码器) → **Stage 2: Caption-based RL + CMKD**(用caption替换图像的多模态偏好数据做RL + 跨模态知识蒸馏对齐文本/视觉表示) → **Stage 3: 多模态RL**(少量真正的多模态偏好数据微调适配)。

### 关键设计

1. **Stage 1: 文本RL**:

    - 功能：在大规模文本偏好数据上学习通用奖励推理
    - 冻结视觉编码器+投影器（纯文本阶段无需视觉）
    - 可验证奖励：格式奖励($r_{format}$) + 准确度奖励($r_{accuracy}$) → $r_v = r_{format} + r_{accuracy}$
    - 设计动机：文本偏好数据丰富(~1M)，可充分训练推理能力→为后续多模态阶段提供强基础

2. **Stage 2: Caption-based RL + CMKD**:

    - **Caption-based RL**：将多模态偏好数据中的图像/视频替换为caption→保持纯文本训练但保留多模态任务结构→桥接文本和多模态
    - 新增任务识别奖励$r_{task}$：模型需先输出任务类型(Image Understanding/Generation等)→鼓励任务感知推理
    - 经验回放：混入Stage 1的高质量文本样本防遗忘
    - **CMKD(跨模态知识蒸馏)**：给定偏好样本s和其caption c，从caption-trained模型$\pi_{\theta_{text}}$采样n个推理→投票+格式过滤+置信度选择→最优推理$o^*$→在多模态输入上做SFT使模型学会从视觉输入复现文本推理→弥合模态鸿沟

3. **Stage 3: 多模态RL**:

    - 在少量多模态偏好数据上微调→前面阶段已赋予强推理能力→所需多模态数据大幅减少
    - 支持视觉理解(图像/视频QA偏好判断)和视觉生成(图像/视频生成质量评估)

### 训练目标
GRPO优化：$\mathcal{L}_{RLVR} = -\mathbb{E}[r_v(s,o)] - \beta\mathbb{D}_{KL}(\pi_\theta || \pi_{\theta_{old}})$

## 实验关键数据

### 主实验

| 方法 | VL-RewardBench Avg | Multimodal RewardBench Avg | GenAI-Bench |
|------|:---:|:---:|:---:|
| 基线 (SFT only) | 66.6 | 76.2 | 70.2 |
| + 单阶段多模态RL | ~70 | ~78 | ~72 |
| **MSRL (多阶段)** | **75.9** | **80.5** | **75.7** |

VL-RewardBench: +9.3, Multimodal RewardBench: +4.3, GenAI-Bench: +5.5。

### 跨规模验证

| 模型规模 | 文本RL收益 |
|----------|:---:|
| 1B | 有效 |
| 7B | 更好 |
| 14B | **最大收益** |

→ 更大模型从文本RL中获益更多，验证了扩展行为的鲁棒性。

### 消融实验

| 配置 | VL-RewardBench |
|------|:---:|
| 无Stage 1 (文本RL) | 显著下降 |
| 无Stage 2 (Caption RL+CMKD) | 中等下降 |
| 无CMKD (仅Caption RL) | 轻微下降 |
| **完整MSRL** | **75.9** |

### 关键发现
- **文本RL是核心引擎**：Stage 1贡献最大——通用奖励推理能力的重要性超过多模态数据适配
- **CMKD有效弥合模态鸿沟**：让caption-trained推理在视觉输入下也能复现→模态无关的推理迁移
- **任务识别奖励有帮助**：让模型先识别任务类型再推理→不同任务需要不同评估逻辑
- **跨规模一致**：1B→14B都受益，且更大模型收益更大→方法具有良好的scaling特性

## 亮点与洞察
- **"文本偏好扩展多模态奖励"的核心假设被验证**：奖励推理的核心能力是模态无关的→可从丰富的文本数据学习并迁移。这挑战了"多模态能力必须从多模态数据学习"的直觉
- **多阶段课程的设计巧妙**：文本→caption(桥接)→多模态的渐进迁移，避免了直接跨模态的巨大鸿沟
- **CMKD的"投票+过滤+选择"策略**：从采样推理中提取最佳→比直接蒸馏更鲁棒
- **统一的视觉理解+生成奖励模型**：同一模型既可评估VQA答案质量也可评估生成图像质量——高度通用

## 局限与展望
- 文本偏好数据的质量和覆盖度影响Stage 1的基础能力
- Caption作为视觉的代理仍有信息损失——能否用更丰富的文本描述缩小差距？
- 当前CMKD的投票需要n次采样→增加Stage 2的训练成本
- Stage 3使用的多模态数据量仍需手动确定最优比例

## 相关工作与启发
- **vs 单阶段RLVR**: 直接在少量多模态数据上做RL→推理能力不足。MSRL先在文本上充分训练再迁移
- **vs 判别式MRM**: 判别式输出标量分数→缺乏可解释性。生成式输出CoT+答案→可审查推理过程
- **vs GRAM-R2**: 文本RL的先驱，MSRL进一步扩展到多模态并引入CMKD
- **启发**：文本→多模态的渐进迁移策略可用于任何需要扩展多模态RL训练的场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多阶段RL+文本扩展多模态的思路新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ VL-RewardBench/MM-RewardBench/GenAI-Bench+跨规模+充分消融
- 写作质量: ⭐⭐⭐⭐ 多阶段设计清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了多模态奖励建模的数据瓶颈问题

<!-- RELATED:START -->

## 相关论文

- [CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)
- [Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](../../ICLR2026/reinforcement_learning/rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)

<!-- RELATED:END -->
