---
title: >-
  [论文解读] Context-Value-Action Architecture for Value-Driven Large Language Model Agents
description: >-
  [ACL 2026 (Findings)][价值驱动智能体] 提出 CVA（Context-Value-Action）架构，基于 S-O-R 心理学模型和 Schwartz 价值理论，通过训练在真实人类数据上的 Value Verifier 解耦行为生成与认知推理，有效缓解 LLM 智能体的行为极化问题，在超过 110 万真实交互轨迹的 CVABench 上显著优于基线。
tags:
  - ACL 2026 (Findings)
  - 价值驱动智能体
  - 行为模拟
  - Schwartz价值理论
  - 行为极化
  - 验证器
---

# Context-Value-Action Architecture for Value-Driven Large Language Model Agents

**会议**: ACL 2026 (Findings)  
**arXiv**: [2604.05939](https://arxiv.org/abs/2604.05939)  
**代码**: 无  
**领域**: LLM Agent / 可解释性  
**关键词**: 价值驱动智能体, 行为模拟, Schwartz价值理论, 行为极化, 验证器

## 一句话总结
提出 CVA（Context-Value-Action）架构，基于 S-O-R 心理学模型和 Schwartz 价值理论，通过训练在真实人类数据上的 Value Verifier 解耦行为生成与认知推理，有效缓解 LLM 智能体的行为极化问题，在超过 110 万真实交互轨迹的 CVABench 上显著优于基线。

## 研究背景与动机

**领域现状**：基于 LLM 的类人智能体（游戏 NPC、社交模拟体、任务助手等）需要忠实捕捉人类行为的复杂性、多样性和随机性。现有方法主要依赖心理提示（如角色扮演、CoT 推理）来模拟人类认知过程。

**现有痛点**：现有 LLM 智能体频繁表现出行为僵化和刻板印象。更关键的是，这个问题被当前评估方式掩盖了——"LLM-as-a-judge"评估存在自参照偏差：评判模型与被评智能体共享预训练偏见，倾向于认可极化行为而非惩罚其缺乏真实感。

**核心矛盾**：增加提示驱动推理的强度不会提升行为忠实度，反而加剧价值极化——LLM 将微妙的价值维度简化为"漫画式"原型（如将"易怒"人格极端化为始终攻击性回应），导致群体多样性坍缩。

**本文目标**：构建能忠实再现人类行为多样性的智能体，以真实人类数据为评估标准而非 LLM 自评。

**切入角度**：借鉴心理学的 S-O-R（刺激-有机体-反应）模型和 Schwartz 基本人类价值理论——人的行为不是人格的静态输出，而是情境激活特定价值维度的动态过程。

**核心 idea**：用外部 Value Verifier（在真实人类数据上训练）替代 LLM 自身的价值判断，解耦行为生成与认知推理，避免自参照偏差导致的极化。

## 方法详解

### 整体框架
CVA 采用"生成-验证"范式：首先通过 SFT+DPO 校准基础 LLM 的价值-行为映射（VMC 阶段），然后用独立训练的 Value Verifier 从多个候选行为中选择最符合当前激活价值的行为（VDR 阶段）。

### 关键设计

1. **价值-行为映射校准（VMC）**:

    - 功能：纠正 LLM 内在的价值扭曲
    - 核心思路：两步流程——SFT 在 CVABench 真实轨迹上微调，将概率空间对齐到真实条件分布 $P(A|C,V)$；DPO 进一步用偏好对（细腻一致 vs 漫画夸张）强化真实的价值-行为关联，抑制扭曲推理路径
    - 设计动机：直接从真实数据学习，避免 LLM 将价值 $V$ 简化为漫画原型 $V'$

2. **价值驱动验证器（Value Verifier）**:

    - 功能：作为独立判别器评估候选行为与激活价值的一致性
    - 核心思路：在真实 $(C, V, A)$ 三元组上训练的验证器，推理时采用"生成-选择"协议——校准后的模型采样 $N$ 个候选行为，验证器对每个计算一致性得分 $s_i = f_{ver}(a_i, C, V)$，选择得分最高的作为最终输出
    - 设计动机：用自身作为验证器会产生自参照循环放大偏差，独立验证器打破了这个循环

3. **CVABench 基准**:

    - 功能：基于真实人类行为数据的训练和评估框架
    - 核心思路：聚合三个领域超过 110 万条真实交互轨迹（Yelp 评论 54K + Reddit 对话 155K + Foursquare 移动 871K），覆盖 15,571 个用户。用 GPV（General Psychometric Verification）将用户行为映射到 Schwartz 10 维价值空间
    - 设计动机：用真实数据替代 LLM 自评，建立客观的行为忠实度基准

### 损失函数 / 训练策略
SFT：标准自回归损失在真实轨迹上。DPO：偏好优化，优选细腻一致的行为而非极化夸张的行为。验证器：在真实 $(C,V,A)$ 上训练判别模型。

## 实验关键数据

### 主实验

| 方法 | 行为忠实度 | 多样性保持 | 价值极化程度 |
|------|----------|----------|------------|
| Raw LLM | 低 | 低 | 高 |
| Role Play Agent | 低 | 低 | 高 |
| Prompt-Reasoning Agent | 更低 | 更低 | **更高** |
| CVA (VMC) | 中 | 中 | 中 |
| CVA (VMC + VDR) | **最高** | **最高** | **最低** |

### 关键发现

| 发现 | 说明 |
|------|------|
| 推理强度 vs 极化 | 增强提示推理反而加剧极化，与直觉相反 |
| 验证器峰值现象 | 行为忠实度不随候选数 N 单调递增，存在最优峰值 |
| 可解释性 | 验证器注意力可透明展示哪些价值维度决定了选择 |

### 关键发现
- 增加推理强度（更多 CoT 步骤）不仅未提升忠实度，反而加剧价值极化并坍缩群体多样性
- 行为忠实度存在最优候选数峰值，模拟了人类认知约束中有限评估范围的现象
- CVA 在所有三个领域（评论/对话/移动）上均显著优于基线

## 亮点与洞察
- **"推理越多极化越严重"的发现**非常重要——直接挑战了"更多思考=更好表现"的直觉，揭示了 LLM 在人类模拟任务中的核心缺陷
- **验证器峰值效应**巧妙地映射了认知科学中"有限理性"的概念
- **评估范式的纠正**：从"LLM-as-a-judge"转向"真实数据为基准"，为智能体评估树立了新标准

## 局限与展望
- CVABench 的三个数据源（Yelp/Reddit/Foursquare）可能不代表所有人类行为模式
- Schwartz 10 维价值模型虽然经典但可能不够精细——某些行为可能受未建模的因素影响
- 验证器训练依赖大量真实数据，在数据稀缺场景下的效果未知

## 相关工作与启发
- **vs Park et al. (Generative Agents)**：依赖角色提示模拟，会产生行为僵化；CVA 用真实数据训练的验证器替代
- **vs VLA 系统**：VLA 关注具身任务执行，CVA 关注社会心理行为忠实度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将心理学价值理论与 LLM 智能体深度融合，解耦验证思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 110 万条真实数据，多范式对比深入
- 写作质量: ⭐⭐⭐⭐ 理论基础扎实，发现有深度
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 人类模拟和智能体评估有根本性贡献

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](../../NeurIPS2025/interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](../../NeurIPS2025/interpretability/steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](../../NeurIPS2025/interpretability/transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)

<!-- RELATED:END -->
