---
title: >-
  [论文解读] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning
description: >-
   构建基于心智理论（ToM）的层次化多模态情感理解基准 HitEmotion，并提出 TMPO 框架通过中间心理状态作为过程级监督来增强 MLLM 的情感推理能力。
tags:

---

# Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2602.00971](https://arxiv.org/abs/2602.00971)
- **代码**: [https://HitEmotion.github.io/](https://HitEmotion.github.io/)
- **领域**: 多模态情感计算 / 心智理论 / 强化学习 / 大模型
- **关键词**: Theory of Mind, 情感推理, MLLM, 层次基准, GRPO, 推理链优化

## 一句话总结
构建基于心智理论（ToM）的层次化多模态情感理解基准 HitEmotion，并提出 TMPO 框架通过中间心理状态作为过程级监督来增强 MLLM 的情感推理能力。

## 研究背景与动机

### 核心问题
尽管多模态大语言模型（MLLM）在各种任务上表现出色，但在深层情感理解方面仍然存在明显缺陷。核心原因在于：

**缺乏统一认知框架**：现有基准仅提供粗粒度得分，无法定位模型推理能力的断点

**推理链不忠实**：CoT 推理看似连贯但实质是模板匹配，缺乏对心理状态的真正追踪

**情感幻觉**：模型在跨模态冲突线索下产生扭曲的情感归因

### 现有基准局限
- EQ-Bench、EmoBench 等仅覆盖文本模态
- EmoBench-M、EmotionHallucer 等虽然多模态但任务设计分散，没有按认知深度组织
- 无一基准同时提供推理链和理由评估

## 方法详解

### HitEmotion 基准：三层认知层次

**Level 1 - 情感感知与识别 (EPR)**：10 个任务
- 从多模态信号映射到预定义情感类别
- 如面部表情识别、多模态情感识别等

**Level 2 - 情感理解与分析 (EUA)**：8 个任务
- 需要上下文感知和关系推理
- 如幽默理解、讽刺检测、多方对话情感等

**Level 3 - 情感认知与推理 (ECR)**：6 个任务
- 要求因果推理和二阶心智推理
- 如情感诱发推理、情感解释、反讽理解等

总计 **24 个任务**，**20,114 个实例**，覆盖视频和图像模态。

### TMPO 训练框架

#### Stage 1: ToM 对齐的监督微调 (SFT)

使用结构化推理模板，将中间推理步骤用 `<think></think>` 标签包裹，最终输出用 `<answer></answer>` 标签包裹：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{((\mathcal{P},T,A,V), y)} [\log \pi_\theta(y | \mathcal{P}, T, A, V)]$$

推理链的黄金标准通过四步流水线构建：LLM 生成 → 过滤 → 增强 → 校正。

#### Stage 2: 基于 GRPO 的 ToM 偏好优化

对每个输入采样 $N$ 个候选输出，通过多维奖励函数评估：

$$R(y) = \mu_1 R_{\text{structure}} + \mu_2 R_{\text{content}} + \mu_3 R_{\text{process}} + \mu_4 R_{\text{consistency}}$$

四个奖励分量：
- **Structure Reward**：推理步骤的正确顺序
- **Content Reward**：最终答案的正确性
- **Process Reward**：领域特定语言的使用
- **Consistency Reward**：逻辑和事实一致性惩罚

GRPO 优化目标：
$$\max_{\pi_\theta} \mathbb{E}_{y_i \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(y_i)}{\pi_{\text{old}}(y_i)} A_i \right] - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### ToM 风格提示机制

三层认知复杂度的提示设计：
- **Level 1**: 一阶心理状态归因 — 整合可观察信号推断情感
- **Level 2**: 关系与上下文心智建模 — 将情感与特定实体或沟通目标关联
- **Level 3**: 因果归因与二阶推理 — 解释情感产生原因和社交解读

## 实验

### 基线模型评测（EPR Level 1）

| 模型 | FESD | ISA | MESA | MER | MSA | OSA | SIA |
|------|------|-----|------|-----|-----|-----|-----|
| VideoLLaMA3-7B | 61.78 | 46.85 | 21.60 | 52.18 | 64.62 | 67.89 | 35.20 |
| LLaVA-One-Vision-7B | 63.44 | 49.19 | 17.05 | 39.50 | 65.40 | 63.00 | 27.00 |

### 关键发现

1. **SOTA 模型在高层认知任务上表现不一致**：即使最强的闭源模型在 ECR 层仍存在显著缺陷
2. **ToM 推理链单独作为提示策略就能显著提升闭源模型表现**：验证了 ToM 作为推理"脚手架"的有效性
3. **TMPO 优化带来一致性提升**：在所有评估任务上超越基线，生成的推理链在忠实度和逻辑一致性方面显著更优
4. **从"通用涌现"到"领域获取"**：TMPO 将推理能力从通用属性转化为认知特化技能

## 亮点
1. **首个将心理学理论与 MLLM 推理过程和理由生成统一的评估框架**
2. **ToM 提示机制设计精妙**：三层认知层次对应三种不同深度的推理模板
3. **GRPO + 过程级奖励的创新组合**：中间心理状态既作为监督信号也作为奖励来源
4. **规模性**：24 个数据集、20K+ 实例的综合基准

## 局限性
1. 金标准推理链依赖 LLM 生成，可能引入 LLM 固有偏差
2. 基于重构已有数据集，原始标注质量不一
3. GRPO 训练计算成本较高
4. 主要评估在单轮 QA 场景，对多轮交互的情感推理未充分探索

## 相关工作
- **多模态情感计算**: SALV、PAD 等融合策略从早期/晚期发展到中间交互方案
- **情感智能评估**: EQ-Bench → EmoBench-M → EmotionHallucer 的演进
- **ToM 推理**: 从 ToMBench 到 MMToM-QA 揭示 MLLM 的 ToM 缺陷
- **推理优化**: DeepSeek-R1 的 GRPO 方法在文本推理中的成功

## 评分
- **创新性**: ⭐⭐⭐⭐ — ToM 认知框架与 MLLM 评估/训练的深度融合
- **实验充分性**: ⭐⭐⭐⭐⭐ — 24 个数据集的全面评估
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法动机充分
- **实用性**: ⭐⭐⭐⭐ — 提供评估工具包和优化方法

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind](rebuttalagent_strategic_persuasion_in_academic_rebuttal_via_theory_of_mind.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](../../AAAI2026/reinforcement_learning/intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)
- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](../../ACL2026/reinforcement_learning/attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)
- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)

<!-- RELATED:END -->
