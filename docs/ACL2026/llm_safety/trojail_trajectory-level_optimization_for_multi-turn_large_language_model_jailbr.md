---
title: >-
  [论文解读] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards
description: >-
  [ACL 2026][LLM安全][多轮越狱攻击] 本文将自动化多轮越狱攻击建模为多轮强化学习问题，提出 TROJail，通过两个启发式过程奖励（过度有害惩罚和语义相关性递进）缓解结果奖励的稀疏监督问题，在多个模型和基准上显著提升攻击成功率。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "多轮越狱攻击"
  - "轨迹级优化"
  - "过程奖励"
  - "强化学习"
  - "红队测试"
---

# TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards

**会议**: ACL 2026  
**arXiv**: [2512.07761](https://arxiv.org/abs/2512.07761)  
**代码**: [GitHub](https://github.com/xxiqiao/TROJail)  
**领域**: AI 安全 / LLM 推理  
**关键词**: 多轮越狱攻击, 轨迹级优化, 过程奖励, 强化学习, 红队测试

## 一句话总结

本文将自动化多轮越狱攻击建模为多轮强化学习问题，提出 TROJail，通过两个启发式过程奖励（过度有害惩罚和语义相关性递进）缓解结果奖励的稀疏监督问题，在多个模型和基准上显著提升攻击成功率。

## 研究背景与动机

**领域现状**：LLM 面临越狱攻击的安全威胁。多轮越狱攻击因反映真实交互场景而受到关注。现有训练型方法使用 DPO 或拒绝采样微调在每轮独立优化攻击者 LLM。

**现有痛点**：(1) 逐轮优化是短视的——最大化每轮直接响应的有害程度，无法学习跨轮的长期攻击策略；(2) 早期看似无害但战略关键的 prompt 因未触发即时有害响应而被低估；(3) 无训练方法依赖人工设计策略，需要大量试验且在受害模型偏离预期时容易崩溃。

**核心矛盾**：轨迹级优化是自然的解决方案，但仅依赖最终响应的有害程度作为结果奖励面临严重的稀疏监督问题——攻击者无法推断中间 prompt 如何贡献于最终攻击成功。

**本文目标**：设计更丰富的中间反馈信号来估计中间 prompt 的效用，从而支持长期攻击策略的学习。

**切入角度**：通过控制实验发现两个经验模式——(1) 适度有害的中间 prompt 最有效，过度有害触发拒绝机制反而失败；(2) 成功轨迹中响应的语义相关性逐渐递增，失败轨迹则不显示此模式。

**核心 idea**：在多轮 GRPO 框架中引入两个过程奖励——过度有害惩罚 $r_{h_1}$ 和语义相关性递进 $r_{h_2}$，将它们整合到优势估计中，为中间 prompt 提供细粒度的训练信号。

## 方法详解

### 整体框架

TROJail 基于多轮 GRPO。对每个有害 prompt $x_0$，攻击者 $\pi_\theta$ 与受害模型 $\pi_\phi$ 交互最多 T 轮生成 G 条轨迹。结果奖励 $r_o$ 为最终响应的有害程度。两个过程奖励 $r_{h_1}$、$r_{h_2}$ 评估中间 prompt 的效用。最终优势 $\hat{A}_{i,t} = \hat{A}_{i,t}^o + \lambda \hat{A}_{i,t}^h$ 结合结果和过程优势，通过 PPO 风格的截断目标优化 $\pi_\theta$。

### 关键设计

1. **过度有害惩罚（Over-Harm Penalization, $r_{h_1}$）**:

    - 功能：防止中间 prompt 过于有害而触发受害模型的拒绝机制
    - 核心思路：如果中间响应触发拒绝则 $r_{h_1} = 0$，否则等于直接响应的有害程度 $r(x_0, y_t)$。这鼓励攻击者保持适度的恶意程度——既要推进攻击又不能打草惊蛇
    - 设计动机：控制实验显示中间 prompt 有害程度与最终攻击成功呈倒 U 型关系——适度有害最优，过度有害导致结果奖励急剧下降

2. **语义相关性递进（Semantic Relevance Progression, $r_{h_2}$）**:

    - 功能：鼓励中间响应逐步引导向目标有害内容
    - 核心思路：计算中间响应与原始有害 prompt 的句子嵌入余弦相似度，按轮次比例加权：$r_{h_2}(x_t) = \frac{t}{|\tau|} \cdot \text{cosine}(e(x_0), e(y_t))$。后期轮次权重更大，鼓励稳步递增的语义对齐
    - 设计动机：成功轨迹的语义相关性呈平稳递增，而有害程度奖励只在最后一轮才飙升——语义相关性提供了更可靠且逐渐的中间反馈信号

3. **过程优势估计与整合**:

    - 功能：将过程奖励整合到轨迹级优化的优势估计中
    - 核心思路：对所有轨迹和轮次的启发式奖励集合 $\mathcal{D}_h$ 计算归一化的过程优势 $\hat{A}_{i,t}^h = \sum_{s=t}^{|\tau_i|} \frac{r_h(x_{i,s}) - \text{mean}(\mathcal{D}_h)}{\text{std}(\mathcal{D}_h)}$，使用前缀和累积未来奖励。最终优势为 $\hat{A}_{i,t} = \hat{A}_{i,t}^o + \lambda \hat{A}_{i,t}^h$
    - 设计动机：结果优势提供全局指引，过程优势提供局部指导——两者互补，既优化最终目标又给出中间步骤的梯度信号

### 损失函数 / 训练策略

使用多轮 GRPO 的 PPO 风格截断目标 + KL 正则化。攻击者基于 Qwen2.5-3B-Instruct。受害模型包括 Llama-3.1-8B、Qwen2.5-7B、Gemma-2-9B、Mistral-7B 等。

## 实验关键数据

### 主实验

**跨模型平均攻击成功率（ASR）对比**

| 方法 | 类型 | 平均 ASR |
|------|------|---------|
| ActorAttack | 无训练多轮 | ~60% |
| HARM | 训练型逐轮 | ~58% |
| Siren (DPO) | 训练型逐轮 | ~65% |
| **TROJail** | 训练型轨迹级 | **~72%** |

### 消融实验

**过程奖励消融**

| 配置 | 说明 |
|------|------|
| w/o 两个过程奖励 | 退化为纯 MT-GRPO，ASR 显著下降 |
| w/o 过度有害惩罚 | 攻击者倾向生成过于激进的 prompt，触发更多拒绝 |
| w/o 语义递进 | 中间轮次容易偏离目标有害内容 |

### 关键发现

- TROJail 在所有受害模型和基准上一致超越逐轮优化方法
- 两个过程奖励对性能贡献相当，但语义递进在长轨迹上更关键
- 控制实验验证了过度有害的倒 U 型关系——L3-L4 级的中间 prompt 最有效
- 轨迹可视化显示 TROJail 学到了"先铺垫后触发"的长期策略模式

## 亮点与洞察

- 两个经验模式的发现是全文的基石——通过精心设计的控制实验量化了中间 prompt 的效用
- 将多轮越狱建模为多轮 RL 问题的视角自然且优雅，过程奖励的设计有理论和实证双重支撑
- 研究虽聚焦攻击，但其发现直接服务于防御——理解攻击策略才能更好地设计安全机制

## 局限与展望

- 攻击成功的评判依赖外部有害程度评估器，其本身可能不完美
- 仅在 7-9B 级别的受害模型上评估，未测试更大或更新的模型
- 过程奖励是启发式设计，可能存在更好的中间反馈信号
- 伦理考量——攻击方法的公开可能被滥用，需要负责任的披露

## 相关工作与启发

- **vs Siren/MTSA (DPO 逐轮优化)**: 后者在每轮独立优化，无法学习跨轮策略；TROJail 的轨迹级优化能发现"先铺垫后触发"的长期模式
- **vs ActorAttack (无训练多轮)**: 后者依赖预设策略，受害模型偏离预期时容易崩溃；TROJail 通过 RL 自动学习策略
- **vs MT-GRPO**: 纯结果奖励面临稀疏监督，TROJail 的过程奖励提供了关键的中间指导

## 评分

- 新颖性: ⭐⭐⭐⭐ 将多轮越狱建模为多轮 RL 并设计过程奖励的思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4 受害模型 × 3 基准 + 控制实验 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 从经验模式到方法设计的逻辑清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 安全研究有重要推动，攻防两面都有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](../../ICML2026/llm_safety/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)
- [\[ACL 2026\] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)

</div>

<!-- RELATED:END -->
