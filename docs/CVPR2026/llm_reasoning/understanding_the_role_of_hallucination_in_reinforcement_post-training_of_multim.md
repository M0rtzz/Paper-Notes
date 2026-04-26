---
title: >-
  [论文解读] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models
description: >-
  [CVPR 2026][LLM推理][多模态推理] 本文提出 Hallucination-as-Cue 分析框架，通过三种模态特定腐蚀策略（空白图像、随机图像、文本移除）系统研究 RL 后训练对多模态推理模型的真实作用机制，发现即使在 100% 腐蚀视觉输入下 GRPO 训练仍能显著提升推理性能，挑战了"RL 训练能有效利用视觉信息"的主流假设。
tags:
  - CVPR 2026
  - LLM推理
  - 多模态推理
  - 强化学习后训练
  - 幻觉分析
  - GRPO
  - 模态腐蚀
---

# Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models

**会议**: CVPR 2026  
**arXiv**: [2604.03179](https://arxiv.org/abs/2604.03179)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 多模态推理、强化学习后训练、幻觉分析、GRPO、模态腐蚀

## 一句话总结

本文提出 Hallucination-as-Cue 分析框架，通过三种模态特定腐蚀策略（空白图像、随机图像、文本移除）系统研究 RL 后训练对多模态推理模型的真实作用机制，发现即使在 100% 腐蚀视觉输入下 GRPO 训练仍能显著提升推理性能，挑战了"RL 训练能有效利用视觉信息"的主流假设。

## 研究背景与动机

1. **领域现状**：受 DeepSeek-R1 等文本推理 LLM 的成功启发，大量工作将 GRPO 等 RL 后训练方法应用到多模态 LLM（如 Qwen2.5-VL）上，在视觉数学推理等任务上取得显著提升。
2. **现有痛点**：虽然 RL 后训练能提升 benchmark 分数，但目前没有工作系统研究过"这些提升到底来自真正的视觉理解，还是仅仅强化了文本推理能力"。当前 RL 奖励仅基于最终答案的对错，与模型是否正确使用了视觉信息无关。
3. **核心矛盾**：如果 RL 训练主要强化的是文本推理模式而非视觉感知，那么当前方向的投入可能事倍功半——模型只是在学"猜答案"而不是"看图推理"。
4. **本文目标**：设计系统的诊断框架，定量回答"RL 后训练是否真正利用了视觉信息"。
5. **切入角度**：把幻觉当作"诊断线索"而非需要消除的缺陷，通过故意诱导幻觉来暴露训练的真实机制。
6. **核心 idea**：设计三种模态特定腐蚀（空白图像/随机图像/文本移除），分别在训练和推理阶段施加，通过 8 种设定组合全面分析 RL 训练动态。

## 方法详解

### 整体框架

Hallucination-as-Cue 框架包含三个部分：(1) 模态特定腐蚀策略设计 → (2) 幻觉诱导训练（用腐蚀数据做 GRPO 训练）→ (3) 幻觉诱导推理与分析（8 种设定的交叉评估）。框架的目的不是训练更好的模型，而是诊断当前 RL 训练方法的内在机制。

### 关键设计

1. **空白图像替换（Blank Image, BI）**

    - 功能：完全移除视觉信息，迫使模型纯靠文本推理
    - 核心思路：将所有训练/测试图像替换为空白图像。在 GRPO 训练中，模型必须从纯文本条件出发生成推理链，如果碰巧得到正确答案则获得正向奖励
    - 设计动机：如果 BI 训练后模型仍能提升，说明 RL 训练可以不依赖视觉信息就增强推理能力

2. **随机图像替换（Random Image, RI）**

    - 功能：提供错误视觉信息，测试模型是否会被误导
    - 核心思路：将每张训练/测试图像替换为数据集中随机另一张图像，构造文图不匹配的训练对
    - 设计动机：比 BI 更具挑战性——模型不仅缺乏正确视觉信息，还面临干扰。如果 RI 训练仍有效，说明模型学会了忽略视觉干扰而依赖文本推理

3. **文本信息移除（Textual Removal, TR）**

    - 功能：移除文本条件，迫使模型依赖视觉输入
    - 核心思路：通过规则匹配移除题目中的变量条件和问题描述，仅保留模板化指令和图像输入
    - 设计动机：作为对照——如果 RL 确实能利用视觉信息，TR 训练应该表现最好（因为图像中仍包含问题条件和标注），但实验表明 TR 训练并未显著优于 BI/RI

### 损失函数 / 训练策略

使用标准 GRPO 算法，组归一化优势 $A_i = \frac{R_i - \mu_{group}}{\sigma_{group} + \epsilon}$，PPO-style clipped surrogate + KL 惩罚。训练 15 个 episode，rollout size 5，温度 0.7，KL 权重 0.01，学习率 $1 \times 10^{-6}$。唯一区别在于输入数据是否经过模态腐蚀。

## 实验关键数据

### 主实验

| 模型 | 训练方式 | MathVision | MathVerse | MathVista | WeMath | AVG |
|------|----------|------------|-----------|-----------|--------|-----|
| Qwen2.5-VL-3B | Base | 18.19 | 34.82 | 51.40 | 54.48 | 39.72 |
| Qwen2.5-VL-3B | +GRPO | 22.73 | 37.72 | 58.40 | 60.11 | 44.74 |
| Qwen2.5-VL-3B | +GRPO-BI | 20.95 | 35.10 | 56.40 | 56.55 | 42.25 |
| Qwen2.5-VL-3B | +GRPO-RI | 20.86 | 35.76 | 58.00 | 55.17 | 42.45 |
| Qwen2.5-VL-7B | Base | 27.70 | 45.20 | 67.00 | 63.68 | 50.89 |
| Qwen2.5-VL-7B | +GRPO | 28.13 | 47.56 | 70.00 | 68.39 | 53.52 |
| Qwen2.5-VL-7B | +GRPO-BI | 28.39 | 48.86 | 68.50 | 66.84 | 53.15 |
| Qwen2.5-VL-7B | **+GRPO-RI** | **27.27** | **49.90** | **71.40** | **68.33** | **54.23** |

### 消融实验

| 设定 | 训练数据 | MathVision | MathVerse | MathVista | WeMath | AVG |
|------|----------|------------|-----------|-----------|--------|-----|
| GRPO | Geometry3K | 22.73 | 37.72 | 58.40 | 60.11 | 44.74 |
| GRPO | MMR1-V0 | 26.18 | 39.26 | 65.00 | 62.47 | 48.23 |
| GRPO | CLEVR | 23.06 | 35.96 | 58.20 | 55.75 | 43.24 |
| GRPO-BI | Geometry3K | 20.95 | 35.10 | 56.40 | 56.55 | 42.25 |
| GRPO-BI | MMR1-V0 | 24.28 | 40.03 | 61.20 | 61.61 | 46.78 |
| GRPO-BI | CLEVR | 21.51 | 35.05 | 58.20 | 54.20 | 42.24 |

### 关键发现

- **最震撼的发现**：7B 模型在随机图像（GRPO-RI）训练下 AVG 达到 54.23%，**超过正常 GRPO 训练的 53.52%**。这意味着用完全错误的图片训练反而更好
- **BI 训练在 MathVision 上的反常**：3B 基座模型在 BI 推理下准确率从 18.19% 升到 18.91%（+0.72%），说明视觉信息甚至可能干扰小模型的推理
- **模型规模效应**：大模型从幻觉轨迹中受益更多——7B 的 GRPO-BI/RI 与正常 GRPO 的差距远小于 3B
- **TR 未优于 BI/RI**：即使 TR 保留了图像中的视觉线索，训练效果与完全无视觉的 BI 差距不大，进一步证实当前 RL 训练未有效利用视觉信息
- **视觉密集型问题受损最大**：BI 推理下 Vision Intensive 问题准确率下降 20-26%，但 Text Dominant 问题仅下降 4-7%

## 亮点与洞察

- **反直觉的核心发现极具冲击力**：用错误图片训练比正确图片效果更好，这不仅是一个有趣的实验观察，更是对整个多模态 RL 训练范式的深刻质疑
- **Hallucination-as-Cue 的诊断思路可广泛复用**：把"缺陷"转化为"诊断信号"的思路可以迁移到其他场景，如用噪声音频训练来诊断语音模型的文本依赖程度
- **8 种评估设定的交叉矩阵设计非常系统**：训练×推理×腐蚀的组合覆盖全面，确保结论的可靠性

## 局限与展望

- 仅研究了 GRPO 算法，PPO、DPO 等其他 RL 方法的行为可能不同
- 实验限于 Qwen2.5-VL 的 3B 和 7B 规模，更大规模（72B）模型是否仍然依赖文本先验？
- 训练数据主要是视觉数学推理（Geometry3K、MMR1-V0），结论能否推广到自然图像 VQA、视频推理等场景尚不清楚
- 文章侧重诊断和分析，未提出具体的改进方案来让 RL 训练真正利用视觉信息
- 后续可探索模态感知的奖励函数设计（如基于 visual grounding 质量的额外奖励）以弥补当前最终答案奖励的不足

## 相关工作与启发

- **vs DeepSeek-R1 / OpenAI-o1**: 这些纯文本推理模型的成功本身就暗示了"推理能力可能主要来自语言模块"，本文在多模态场景下验证了这一猜想
- **vs Ma et al. (2603.27201)**: 该并行工作聚焦于 MCoT 模型推理阶段的幻觉缓解，本文则聚焦于训练阶段的幻觉角色，两者互补——一个管"推理时怎么减少幻觉"，一个管"训练时幻觉为何反而有用"
- **vs concurrent work (models retaining accuracy without images)**: 该工作在推理时去除图像评估，本文进一步在训练时也去除图像，发现训练阶段的影响更深远

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将幻觉从"需要消除的问题"重新定义为"诊断工具"的视角极具创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 2个模型规模×3种腐蚀×3个数据集×8种设定×5个benchmark，实验矩阵极为全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，但部分图表信息量过于密集
- 价值: ⭐⭐⭐⭐⭐ 对多模态RL训练的"皇帝的新衣"式揭示具有重要警示意义，可能深刻影响后续研究方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[CVPR 2026\] Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing](harnessing_chain-of-thought_reasoning_in_multimodal_large_language_models_for_fa.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)

<!-- RELATED:END -->
