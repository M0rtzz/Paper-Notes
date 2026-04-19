---
title: >-
  [论文解读] RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind
description: >-
  [ICLR2026][academic rebuttal] 首次将心智理论（ToM）引入学术 rebuttal，提出 ToM-Strategy-Response 三阶段框架：先建模审稿人心理状态，再制定说服策略，最后生成证据支撑的回复，结合自奖励 RL 训练和专用 Rebuttal-RM 评估器，平均指标超越基座模型 18.3%。
tags:
  - ICLR2026
  - academic rebuttal
  - Theory of Mind
  - strategic persuasion
  - GRPO
  - self-reward
  - reward model
---

# RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind

**会议**: ICLR2026  
**arXiv**: [2601.15715](https://arxiv.org/abs/2601.15715)  
**代码**: [GitHub](https://github.com/Zhitao-He/RebuttalAgent)  
**领域**: reinforcement_learning  
**关键词**: academic rebuttal, Theory of Mind, strategic persuasion, GRPO, self-reward, reward model  

## 一句话总结
首次将心智理论（ToM）引入学术 rebuttal，提出 ToM-Strategy-Response 三阶段框架：先建模审稿人心理状态，再制定说服策略，最后生成证据支撑的回复，结合自奖励 RL 训练和专用 Rebuttal-RM 评估器，平均指标超越基座模型 18.3%。

## 背景与动机
- 学术 rebuttal 不是简单的技术辩论，而是在严重信息不对称下的**战略沟通**（类似不完全信息动态博弈）
- 作者不了解审稿人的知识背景、内在偏见、或回复的连锁效应
- **现有方法的根本缺陷**：主要依赖 SFT 训练在 review 数据集上，只能模仿表层语言模式（表面礼貌但模板化），缺乏战略深度
- 成功的 rebuttal 本质是**换位思考**（perspective-taking），需要分析何时让步、何时坚持、何时重新框架叙事
- 这种能力在认知科学中称为 Theory of Mind（ToM）——理解他人的信念、意图和观点以预测其行为

## 方法详解

### 整体框架：ToM-Strategy-Response (TSR) 三阶段推理
将复杂的 rebuttal 任务分解为三个连贯步骤：

**输入**：原始手稿 $M$、审稿意见 $R_i$、目标评论 $c_{target}$  
**输出**：策略性回复 $r_{target} = \mathcal{G}(M, R_i, c_{target})$

### 关键设计 1：层次化审稿人画像建模
- **宏观分析（Macro-level）**：推断审稿人整体意图
    - 四个维度：Overall Stance、Overall Attitude、Dominant Concern、Reviewer Expertise
    - 生成结构化宏观画像，指导全局策略和语气
- **微观分析（Micro-level）**：分解具体评论
    - 四个维度：Significance、Methodology、Experimental Rigor、Presentation
    - 生成微观画像，指导针对性回应

### 关键设计 2：ToM-Driven 策略生成 + 证据融合
- **策略生成**：基于完整审稿人画像和目标评论，LLM 综合输出简洁的高层战略计划
    - 先决定"如何回复"（how），再决定"回复什么"（what）
    - 确保回复不仅是被动回应表面问题，而是战略对齐审稿人深层关切
- **证据基础回复**：三阶段上下文检索模块
    - 手稿分段 → 嵌入编码 → cosine 相似度排序 → top-k chunks
    - 回复生成同时条件化于：审稿人画像 $\mathcal{P}$、策略 $S$、检索块 $C_E$、原始回复 $r_{orig}$

### 关键设计 3：自奖励强化学习
**数据构建（RebuttalBench，70K 样本）**：
- 来源：Re2-rebuttal 数据集，GPT-4.1 解析 200K+ 评论-回复对
- 多教师模型（GPT-4.1、Claude 3.5）混合生成 TSR 链
- 排除需要新实验的评论，聚焦语言说服和战略论证

**两阶段训练**：
1. **SFT 冷启动**：Qwen3-8B 基座，学习 TSR 结构化推理
2. **GRPO 强化学习 + 自奖励**：
    - 四维奖励函数：$R(o) = w_1 R_{format} + w_2 R_{think} + w_3 R_{resp} + w_4 R_{div}$
    - $R_{format}$：格式正确性（二值）
    - $R_{think}$：推理质量（模型自评 Analysis + Strategy）
    - $R_{resp}$：回复质量（模型自评说服力、清晰度、证据使用）
    - $R_{div}$：回复多样性（与预设负样本的语义差异，抗 reward hacking）

### 评估器：Rebuttal-RM
- 基于 Qwen3-8B 微调，训练数据 102K 样本（三源：原始作者回复、GPT-4.1 精修回复、多模型生成回复）
- 多维打分 + 解释输出
- 与人类一致性（0.812 平均分）显著超越 GPT-4.1（0.745）

## 实验

### Rebuttal-RM 与人类评估一致性

| 评分模型 | Attitude | Clarity | Persuasiveness | Constructiveness | 平均 |
|---------|----------|---------|----------------|-----------------|------|
| GPT-4.1 | 0.752 | 0.720 | 0.761 | 0.747 | 0.745 |
| DeepSeek-R1 | 0.690 | 0.694 | 0.698 | 0.688 | 0.705 |
| **Rebuttal-RM** | **0.859** | **0.740** | **0.814** | **0.828** | **0.812** |

### 主实验：Rebuttal 质量评估

| 模型 | Rigor(C/P/Co) | Soundness(C/P/Co) | 平均 |
|------|-----------|-----------|------|
| o3 | 9.00/8.99/9.55 | 8.84/8.78/9.45 | 9.21 |
| GPT-4.1 | 8.34/7.86/8.80 | 8.27/7.79/8.62 | 8.50 |
| DeepSeek-R1 | 8.47/7.90/8.90 | 8.46/8.03/8.75 | 8.64 |
| Qwen3-8B (Base) | 7.96/7.33/8.18 | 7.84/7.11/7.76 | 7.96 |
| **RebuttalAgent** | **8.72/8.25/9.25** | **8.65/8.28/9.10** | **8.88** |

（RebuttalAgent 为 Self-Refined 版本，超越 base model 平均 18.3% 提升，接近 DeepSeek-R1 水平）

### 关键发现
1. **ToM 分析是核心贡献**：去掉 ToM 阶段后性能显著下降，证明审稿人建模的必要性
2. **自奖励 RL 有效**：相比纯 SFT，RL 阶段在说服力和建设性维度上均有显著提升
3. **8B 模型可媲美大模型**：RebuttalAgent 在多项指标上接近甚至超越 GPT-4.1 和 DeepSeek-R1
4. **Rebuttal-RM 超越通用 judge**：专门训练的评估器比通用 LLM judge 更可靠
5. **多样性奖励抗 reward hacking**：$R_{div}$ 有效防止模型退化为模板化回复

## 亮点
- 首次将 Theory of Mind 引入学术 rebuttal，从博弈论视角重新定义问题
- TSR 三阶段框架优雅地将复杂任务分解为可训练的子任务
- 自奖励机制避免了额外 reward model 的训练开销
- Rebuttal-RM 评估器本身具有独立应用价值
- 开源代码和模型

## 局限性
- 排除了需要新实验/数据的评论，实际 rebuttal 中这类评论占比不小
- 上下文检索基于论文全文，长论文可能导致关键信息遗漏
- 自奖励的质量上限受 SFT 模型能力限制
- 评估主要基于自动化指标，人类评估规模有限
- 实际应用中需谨慎使用，不应替代作者自身的批判性思考

## 相关工作
- **LLM 辅助研究**：从文献总结到假设生成到完整论文写作
- **学术 rebuttal 研究**：Re2 数据集（Zhang et al., 2025）提供原始 review-rebuttal 数据
- **Theory of Mind in AI**：Machine ToM（Rabinowitz et al., 2018），GPT-4 已展示初步 ToM 能力
- **推理型 LLM**：DeepSeek-R1 的 GRPO 算法为 RL 后训练提供基础

## 评分
⭐⭐⭐⭐ (4/5)

问题定义新颖，将 rebuttal 建模为信息不对称的博弈并引入 ToM 是有创意的。TSR 框架设计逻辑清晰。主要concern 是排除了需实验的评论后任务难度降低，且实战效果需要更多验证。

<!-- RELATED:START -->

## 相关论文

- [Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)
- [Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [PreferThinker: Reasoning-based Personalized Image Preference Assessment](preferthinker_reasoning-based_personalized_image_preference_assessment.md)

<!-- RELATED:END -->
