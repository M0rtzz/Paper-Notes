---
title: >-
  [论文解读] Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering
description: >-
  [ICCV 2025][AI安全][continual learning] 提出QUAD，通过仅存储先前任务的问题（不存储图像）进行重放，配合注意力一致性蒸馏保持跨任务的模态内和模态间注意力模式，在隐私保护的前提下实现持续VQA的SOTA性能。 持续VQA要求模型在学习新视觉-语言技能（可塑性）的同时保留先前知识（稳定性）…
tags:
  - "ICCV 2025"
  - "AI安全"
  - "continual learning"
  - "VQA"
  - "question-only replay"
  - "注意力机制"
  - "privacy-preserving"
---

# Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: [GitHub](https://github.com/IemProg/QUAD)  
**领域**: 持续学习 / 视觉问答  
**关键词**: continual learning, VQA, question-only replay, attention distillation, privacy-preserving

## 一句话总结

提出QUAD，通过仅存储先前任务的问题（不存储图像）进行重放，配合注意力一致性蒸馏保持跨任务的模态内和模态间注意力模式，在隐私保护的前提下实现持续VQA的SOTA性能。

## 研究背景与动机

持续VQA要求模型在学习新视觉-语言技能（可塑性）的同时保留先前知识（稳定性）。现有方法主要面向单模态设计，在多模态场景下力不从心。内存重放方法需存储完整的图像-问题对（隐私风险大、存储开销高），尤其图像包含人脸、车牌等敏感信息。而无记忆方法虽保护隐私但遗忘严重。关键问题：**是否真的需要存储视觉数据，还是仅保留过去的问题就足够抑制遗忘？** 此外，持续VQA特有的"答案集外问题"——模型过拟合当前任务的答案空间，对先前任务的问题使用当前任务的答案类型回答。

## 方法详解

### 整体框架

QUAD（Question-only replay with Attention Distillation）在仅存储问题的设置(VQACL-QR)下工作，包含两个核心组件：(1) 仅问题重放机制——选择性重用先前任务的问题来正则化当前模型；(2) 注意力一致性蒸馏——强制跨任务保持模态内和模态间的注意力一致性。

### 关键设计

1. **仅问题重放(Question-only Replay)**: 存储先前任务的问题而非图像-问题对。在当前任务训练时，将存储的旧问题与当前图像配对。即使图像不匹配，问题中的任务类型信息（如计数、颜色等）足以正则化模型的答案空间分布，防止模型遗忘先前任务的答案类型。通过策略性选择过去问题来对抗"答案集外问题"——避免模型将所有问题都用当前任务的答案类型回答。

2. **注意力一致性蒸馏(Attention Consistency Distillation)**: 在任务切换时保持三种注意力模式的一致性：(a) 文本-文本自注意力——保留语言理解能力；(b) 图像-图像自注意力——保留视觉感知能力；(c) 文本-图像交叉注意力——保留视觉-语言关联。使用前一任务的模型作为教师，通过KL散度或余弦相似度约束当前模型的注意力模式与教师一致。

3. **隐私保护设计**: 仅存储文本问题（通用且不可识别身份），完全消除图像存储需求。满足GDPR等数据保护法规要求，特别适用于医疗、金融、监控等敏感领域。

### 损失函数 / 训练策略

总损失 = VQA任务损失 + λ₁·问题重放正则化损失 + λ₂·注意力一致性蒸馏损失。每个任务结束后存储该任务的代表性问题子集（无图像），下个任务训练时随机采样重放。

## 实验关键数据

### 主实验

| 方法 | 数据存储 | VQAv2性能 | NExT-QA性能 | 隐私 |
|------|---------|----------|------------|------|
| Sequential FT | 无 | 严重遗忘 | 严重遗忘 | ✓ |
| VQACL(+图像重放) | 图像+问题 | 较好 | 较好 | ✗ |
| **QUAD(仅问题)** | **仅问题** | **SOTA** | **SOTA** | **✓** |

QUAD甚至超越了需要存储图像的重放方法，证明仅存储问题即足够。

### 消融实验

- 仅问题重放 vs 无重放：显著减少答案集外问题
- 注意力蒸馏：三种注意力模式的一致性都重要，交叉注意力贡献最大
- 存储问题数量：适中数量即可有效正则化
- 问题选择策略：多样性选择优于随机选择

### 关键发现

- 仅存储问题足以抑制遗忘——问题蕴含任务结构信息
- 答案集外问题是持续VQA的核心挑战——模型倾向于用当前任务答案回答所有问题
- 注意力模式的保持对维护视觉-语言关联至关重要
- 隐私保护和性能可以兼得

## 亮点与洞察

- "仅问题重放"的设置定义新颖——在重放和无重放之间找到了隐私友好的中间地带
- 答案集外问题的分析和可视化（混淆矩阵）直观有力
- 注意力蒸馏在多模态持续学习中的应用有效
- 实际上超越了存储更多数据的方法

## 局限与展望

- 问题选择策略可能影响效果，最优选择需要领域知识
- 注意力蒸馏增加了训练时间
- 假设问题是非敏感的，但某些领域的问题本身可能包含隐私信息
- 仅在VQA任务上验证，对其他多模态持续学习任务的扩展性待验证

## 相关工作与启发

- VQACL建立了持续VQA的基准和设置
- 知识蒸馏在传统CL中广泛使用，本文将其扩展到多模态注意力
- 隐私保护持续学习是新兴方向，问题-only设置可启发更多模态分离的方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 仅问题重放设置定义新颖，发现深刻
- 技术深度: ⭐⭐⭐⭐ — 注意力蒸馏设计针对性强
- 实验充分性: ⭐⭐⭐⭐ — 两基准、多基线、混淆矩阵可视化
- 写作质量: ⭐⭐⭐⭐⭐ — 动机图出色，问题定义清晰
- 实用价值: ⭐⭐⭐⭐⭐ — 隐私保护+SOTA性能，实际价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](fret_feature_redundancy_elimination_for_test_time_adaptation.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)
- [\[ICCV 2025\] Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack](mind_the_cost_of_scaffold_benign_clients_may_even_become_accomplices_of_backdoor.md)

</div>

<!-- RELATED:END -->
