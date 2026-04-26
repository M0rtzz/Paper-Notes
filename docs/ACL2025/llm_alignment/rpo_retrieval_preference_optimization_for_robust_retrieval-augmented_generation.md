---
title: >-
  [论文解读] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation
description: >-
  [ACL 2025][LLM对齐] 提出 Retrieval Preference Optimization (RPO)，一种专为 RAG 设计的轻量级偏好对齐方法，通过将检索质量评估隐式地集成到生成过程中，使 LLM 能够自适应地在参数知识和检索知识之间做出选择，无需额外组件即可缓解知识冲突导致的幻觉问题。
tags:
  - ACL 2025
  - LLM对齐
---

# RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation


**会议**: ACL 2025  
**arXiv**: [2501.13726](https://arxiv.org/abs/2501.13726)  
**作者**: Shi-Qi Yan, Quan Liu, Zhen-Hua Ling（中国科大 / 科大讯飞）
**代码**: 未公开  
**领域**: LLM Alignment / RAG

## 一句话总结

提出 Retrieval Preference Optimization (RPO)，一种专为 RAG 设计的轻量级偏好对齐方法，通过将检索质量评估隐式地集成到生成过程中，使 LLM 能够自适应地在参数知识和检索知识之间做出选择，无需额外组件即可缓解知识冲突导致的幻觉问题。

## 研究背景与动机

**核心问题**：RAG 在检索到的非参数知识与 LLM 内部参数知识冲突时，模型倾向于盲目信任检索内容，导致生成质量下降。

**现有方案的不足**：

**Pre-Eval 方法**（如 CRAG）：在生成前用额外分类器/LLM 评估检索质量，需多次 API 调用，计算开销大（~6 次调用）

**Post-Eval 方法**（如 Self-RAG）：基于多个检索文档独立生成多个回答再选优，推理代价高（2-11 次调用）
3. 两类方法都会移除部分信息，使生成器更依赖评估器，影响最终性能

**DPO 直接用于 RAG 的三个数学障碍**：
1. RLHF/DPO 的优化目标与 RAG 冲突缓解目标不一致——KL 散度约束阻碍了从检索过度依赖中恢复
2. 当 preferred/dispreferred 回答的输入不同（有/无检索）时，DPO 的配分函数无法消去，损失函数不可计算
3. 为绕过配分函数而伪造参数知识答案会引入似然差异，训练后模型仍倾向选择非参数答案

## 方法详解

### 整体框架

RPO 是一个两阶段训练流程：

**Phase 1: 监督微调 (SFT)**
- 对每个问题 x，分别生成有检索和无检索的回答
- 筛选发生知识冲突的样本（有检索和无检索的回答仅一方正确）
- 用正确答案对模型做 SFT，初步激活模型的检索质量感知能力

**Phase 2: RPO 偏好优化**
- 用 SFT 后的模型重新生成回答并再次筛选冲突样本
- 构建偏好对，标注哪个答案更优
- 用 RPO 损失进行对齐训练

### RPO 损失函数设计

核心创新在于奖励模型中引入**检索奖励项**。RPO 的损失函数包含三部分：

1. **Preferred 生成奖励**（与 DPO 一致）：奖励优选回答
2. **Dispreferred 生成奖励**（与 DPO 一致）：惩罚劣选回答
3. **检索奖励**（RPO 独有）：当非参数答案更优时为正号（鼓励利用检索），当参数答案更优时为负号（抑制检索依赖）

关键设计：
- 检索奖励项的正负号根据哪种知识源更准确而自适应翻转
- 长度归一化除以检索文档长度，消除文档长度带来的偏差
- 检索相关性通过模型对检索文档的条件概率隐式表征，无需外部评估器

### 数据构建策略

训练数据包含两个互补子集：
- **D1（增强检索利用）**：参数知识回答错误、非参数知识回答正确的样本
- **D2（缓解检索过度依赖）**：参数知识回答正确、被检索知识干扰后回答错误的样本

## 实验关键数据

### 表1: 主实验结果（Accuracy %）

| Method | Adaptive类别 | API/LM调用 | PopQA | NQ | TriviaQA | RGB |
|--------|-------------|-----------|-------|------|----------|------|
| RAG (LLaMA2-7B) | - | 1 | 48.8 | 22.0 | 52.5 | 91.6 |
| RAG+DPO | - | 1 | 53.6 | 43.5 | 51.7 | 96.3 |
| CRAG | Pre-Eval | 6 | 54.9 | 38.4 | 59.6 | 92.0 |
| Self-RAG | Post-Eval | 2-11 | 54.9 | 42.4 | 68.9 | 92.6 |
| **RPO (LLaMA2)** | Integrated | **1** | **55.8** | **45.3** | 57.6 | **97.3** |
| RAG (LLaMA3-8B) | - | 1 | 59.0 | 41.3 | 65.8 | 96.3 |
| InstructRAG | - | 1 | 65.0 | 46.7 | 65.1 | 99.3 |
| **RPO (LLaMA3)** | Integrated | **1** | **65.4** | **51.9** | **74.4** | **100.0** |

**关键发现**：RPO 在 LLaMA3 上比 RAG 基线提升 4-10%（PopQA +6.4%, NQ +10.6%, TriviaQA +8.6%, RGB +3.7%），且仅需 1 次推理调用。

### 表2: 消融实验（Accuracy %）

| 变体 | PopQA | NQ | TriviaQA | RGB |
|------|-------|------|----------|------|
| **RPO (完整)** | **55.8** | **45.3** | **57.6** | **97.3** |
| RPO w/o 检索奖励 (=DPO) | 53.6 | 43.5 | 51.7 | 96.3 |
| RPO w/o 偏好优化 (=SFT) | 51.3 | 36.0 | 54.3 | 94.6 |
| RPO w/o SFT | 52.5 | 34.9 | 50.1 | 90.6 |

检索奖励项贡献 +2.2 (PopQA)，偏好优化贡献 +4.5 (PopQA)，两阶段均不可或缺。

### 表3: 低质量检索鲁棒性（全部检索信息错误时）

| Method | Accuracy | 相对提升 |
|--------|----------|---------|
| RAG | 18.6 | 0.0% |
| SFT | 19.5 | +4.8% |
| DPO | 19.3 | +3.7% |
| **RPO** | **23.5** | **+26.3%** |

RPO 在全错检索环境下仍保持显著优势，证明模型真正学会了评估检索质量而非简单偏向某一方。

## 亮点

1. **理论贡献扎实**：数学证明了 DPO 直接用于 RAG 的三大限制（优化目标不一致、配分函数不可消去、过度依赖非参数知识），为方法设计提供了理论支撑
2. **极高的推理效率**：将检索评估与生成集成为单次推理，避免了 Pre-Eval/Post-Eval 方法的多次 API 调用
3. **检索奖励项设计巧妙**：通过隐式表征检索相关性并根据正负号自适应调节，统一处理了"检索有用"和"检索有害"两种场景

## 局限性

1. **训练数据领域受限**：仅在 NQ 上训练，可能对其他领域泛化不足
2. **奖励函数设计空间未充分探索**：作者承认可能存在更优的奖励函数形式
3. **检索评估仍为隐式**：无法显式解释模型在生成时如何权衡两种知识来源
4. **依赖冲突样本筛选**：数据构建需要多次前向推理来检测冲突，增加训练成本
5. **未与更大规模模型对比**：实验仅在 7B/8B 模型上验证

## 相关工作

- **Adaptive RAG**：CRAG（Pre-Eval，检索纠正）、Self-RAG（Post-Eval，自反思生成与批判）、AstuteRAG（迭代知识筛选）
- **模型对齐**：RLHF/PPO -> DPO（简化为闭式解）-> RPO（专为 RAG 设计的对齐）
- **知识冲突**：Longpre et al. 2021 研究实体级冲突；Zou et al. 2024、Xiang et al. 2024 研究 RAG 中的冲突倾向

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次从数学角度论证 DPO 不适用于 RAG，并提出检索感知的偏好优化
- 实验充分度: ⭐⭐⭐⭐ — 4 个数据集 + 2 个基座模型 + 充分的消融和鲁棒性分析
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，结构完整
- 价值: ⭐⭐⭐⭐ — 解决了 RAG 中知识冲突的关键痛点，方法实用高效

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2025\] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](synthesizeme_persona_prompts.md)
- [\[ACL 2025\] Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](kpo_protein_safety.md)

<!-- RELATED:END -->
