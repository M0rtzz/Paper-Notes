---
title: >-
  [论文解读] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning
description: >-
  [CVPR 2026][强化学习] 提出多阶段强化学习（MSRL）方法，通过先在大规模文本偏好数据上学习奖励推理能力，再逐步迁移到多模态任务，解决多模态奖励模型训练中标注数据稀缺的瓶颈问题，在 VL-RewardBench 上将准确率从 66.6% 提升至 75.9%。
tags:
  - CVPR 2026
  - 强化学习
  - 强化学习
  - 跨模态迁移
  - 知识蒸馏
  - 偏好对齐
---

# MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2603.25108](https://arxiv.org/abs/2603.25108)  
**代码**: [GitHub](https://github.com/wangclnlp/MSRL)  
**领域**: Reinforcement Learning / Multimodal Reward Modeling  
**关键词**: 多模态奖励模型, 强化学习, 跨模态迁移, 知识蒸馏, 偏好对齐

## 一句话总结

提出多阶段强化学习（MSRL）方法，通过先在大规模文本偏好数据上学习奖励推理能力，再逐步迁移到多模态任务，解决多模态奖励模型训练中标注数据稀缺的瓶颈问题，在 VL-RewardBench 上将准确率从 66.6% 提升至 75.9%。

## 研究背景与动机

多模态奖励模型（MRM）是对齐多模态大语言模型（MLLM）与人类偏好的核心组件。近期研究从判别式转向生成式奖励建模（通过 CoT 推理生成偏好预测），并开始采用 RLVR（Reinforcement Learning from Verifiable Rewards）进一步增强 MRM 的能力。

然而，RLVR 面临一个根本性瓶颈：**高质量多模态偏好标注数据极度稀缺**。标注成本高昂，无法像文本领域那样大规模扩展 RL 训练。已有替代方案（如置信度估计、自验证）容易产生误差累积，性能快速饱和。

本文的核心洞见是：**偏好推理的核心能力可以从丰富的纯文本数据中学习，并有效迁移到多模态场景**。这打破了"必须用更多多模态数据来解决多模态数据不足"的固有假设。

## 方法详解

### 整体框架

MSRL 采用三阶段课程式训练策略：
1. **Stage 1**：在大规模文本偏好数据上进行 RL，建立通用奖励推理能力
2. **Stage 2**：在 caption-based 数据上进行 RL + 跨模态知识蒸馏，实现偏好迁移
3. **Stage 3**：在少量真实多模态数据上进行 RL，完成最终适配

### 关键设计

1. **文本数据上的大规模 RL（Stage 1）**：
    - 先用 40k HelpSteer3 数据做 SFT（学习 CoT 输出格式）
    - 再在 400k GRAM-R2 文本偏好数据上执行 GRPO 优化
    - 冻结视觉编码器和投射层参数，仅训练语言部分
    - 核心动机：文本偏好数据量大、获取成本低，可以充分利用 RL 的 scaling 特性

2. **Caption-based RL + 偏好泛化（Stage 2）**：
    - 将多模态偏好数据中的图像/视频替换为对应的文字描述（caption），构建纯文本但保留多模态语义的训练数据
    - 引入**任务识别奖励** $r_{\text{task}}$：模型需先输出任务类型标签（如 `<type>Image Understanding</type>`），正确识别得 0.2 奖励
    - 采用**经验回放策略**防止灾难性遗忘：训练批次中混入 Stage 1 的高质量文本样本（新旧比 5:1）

3. **跨模态知识蒸馏（CMKD）**：
    - 解决模态差距问题：给定偏好样本和 caption，用 caption 训练的 MRM 生成 n 个候选推理
    - 三步筛选得到最优教师信号 $o^*$：(1) 多数投票确定伪标签 (2) 格式过滤 (3) 选最高置信度
    - 用 $[c, o^*]$ 对做 SFT，让模型即使只看视觉输入也能复现蒸馏的推理过程
    - 后续 RL 阶段要求模型先生成 `<caption>` 再进行奖励推理

4. **多模态 RL 微调（Stage 3）**：
    - 仅用 20k 多模态数据进行最终适配
    - 由于前两个阶段已建立强大的奖励推理能力，此阶段所需数据量很少
    - 同样使用任务识别奖励

### 损失函数 / 训练策略

- 三阶段均基于 GRPO 优化，核心目标：$\mathcal{L}_{\text{RLVR}} = -\mathbb{E}[r_v(s,o)] - \beta \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\theta_{\text{old}}})$
- 可验证奖励 $r_v = r_{\text{format}} + r_{\text{accuracy}}$（+ Stage 2/3 的 $r_{\text{task}}$）
- 采样大小 8，学习率 1e-6，批大小 128

## 实验关键数据

### 主实验

| 基准测试 | 指标 | MSRL (8B) | Generative MRM | 提升 |
|---|---|---|---|---|
| VL-RewardBench | Avg Acc | 75.9% | 66.6% | +9.3% |
| Multimodal RewardBench | Avg Acc | 80.5% | 76.2% | +4.3% |
| GenAI-Bench (Image Gen.) | Acc | 75.7% | 70.2% | +5.5% |
| ShareGPT (Video Under.) | Acc | 85.5% | 80.6% | +4.9% |
| GenAI-Bench (Video Gen.) | Acc | 81.4% | 68.3% | +13.1% |

MSRL 8B + voting@16 在 VL-RewardBench 上达到 77.5%，甚至超过 Claude-3.7-Sonnet (66.5%) 和 GPT-4o (62.4%)。

### 消融实验

| 配置 | VL-RewardBench Avg | 说明 |
|---|---|---|
| Generative Baseline | 66.6% | 仅用多模态数据训练 |
| w/o Stage 1 | 68.8% | 去掉文本 RL → 损失最大 (-7.1%) |
| w/o Stage 2 (Caption) | 74.3% | 去掉 caption RL → -1.6% |
| w/o Stage 2 (CMKD) | 73.4% | 去掉跨模态蒸馏 → -2.5% |
| w/o Stage 3 | 72.6% | 去掉多模态 RL → -3.3% |
| **Full MSRL** | **75.9%** | 完整方法 |

### 关键发现

- **文本 RL 是最关键的阶段**：Stage 1 贡献了最大的性能增益（+6.9%），证明奖励推理能力可以从纯文本中学习
- **Scaling 行为一致**：从 1B 到 14B 模型，MSRL 的提升始终存在且更大模型受益更多
- **数据效率极高**：仅 5k 多模态数据的 MSRL 已大幅超过仅用多模态数据的 baseline，表明文本 RL 建立的能力使多模态信号的边际收益递减
- **视频任务提升最大**：视频生成任务提升 +13.1%，说明时序视觉数据更依赖强推理能力

## 亮点与洞察

1. **突破数据瓶颈的巧妙思路**：不是寻求更多多模态数据，而是利用跨模态迁移——这是一种"降维打击"式的解决方案
2. **Caption 作为模态桥接**：将图像替换为 caption 实现"文本→多模态"的平滑过渡，简洁而有效
3. **任务识别奖励**：让模型先识别任务类型再推理，提升了统一 MRM 在不同任务间的区分能力
4. **工程友好**：强调了 scalable axis——只需增加文本数据量就能持续提升多模态性能，无需昂贵的多模态标注

## 局限性 / 可改进方向

- 仅在 InternVL3.5 系列上验证，是否对其他架构（如 Qwen-VL、LLaVA）同样有效待验证
- CMKD 中的 caption 由 GPT-5 生成，对 caption 质量有依赖
- Stage 2 的经验回放比例（5:1）是否最优缺乏充分讨论
- 未探讨 MSRL 训练的 MRM 在实际 MLLM 对齐中的下游效果（如用于 rejection sampling / PPO）

## 相关工作与启发

- 与 UnifiedReward 的区别：后者直接用 RLVR 在多模态数据上训练，受限于数据量；MSRL 通过多阶段策略绕过了这个限制
- 受 LLaVA 和 VILA 的启发——caption-based 数据可以有效迁移文本知识到视觉任务
- 与文本 LLM 中"RL 可以 scaling 推理能力"的发现一致，将这一洞见推广到多模态

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多阶段 RL 课程设计新颖，但各组件（GRPO、caption bridging、知识蒸馏）本身不算新
- 实验充分度: ⭐⭐⭐⭐⭐ — 多尺度（1B-14B）、多任务（理解+生成）、多基准，消融完整
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，motivation 阐述充分
- 价值: ⭐⭐⭐⭐⭐ — 提供了一条实用、可扩展的多模态奖励模型训练路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)
- [\[CVPR 2026\] Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)

</div>

<!-- RELATED:END -->
