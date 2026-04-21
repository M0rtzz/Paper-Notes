---
title: >-
  [论文解读] Reinforcement Learning Teachers of Test Time Scaling
description: >-
  [NeurIPS 2025][推理语言模型] 提出强化学习教师（RLT）框架，将问题和答案同时提供给教师模型，训练其生成有效的解释性推理链条，而非从零解题，从而用7B参数的小教师模型产出比数量级更大模型更优的蒸馏数据。
tags:
  - NeurIPS 2025
  - 推理语言模型
  - 知识蒸馏
  - 强化学习
  - 测试时推理
  - 教师-学生框架
---

# Reinforcement Learning Teachers of Test Time Scaling

**会议**: NeurIPS 2025  
**arXiv**: [2506.08388](https://arxiv.org/abs/2506.08388)  
**代码**: [GitHub](https://github.com/SakanaAI/RLT)  
**领域**: 强化学习  
**关键词**: 推理语言模型, 知识蒸馏, 强化学习, 测试时推理, 教师-学生框架

## 一句话总结

提出强化学习教师（RLT）框架，将问题和答案同时提供给教师模型，训练其生成有效的解释性推理链条，而非从零解题，从而用7B参数的小教师模型产出比数量级更大模型更优的蒸馏数据。

## 研究背景与动机

当前推理语言模型（reasoning LM）的训练范式面临两个根本性挑战：

**1. RL的探索困境**：RL训练采用一锤定音的正确性奖励（one-hot correctness reward），只有当模型已经能以一定概率解决问题时才能提供学习信号。这意味着RL本质上只是强化模型已有的能力，而非真正学习新技能。小模型由于初始能力不足，很难通过RL有效提升。

**2. 训练目标与实际用途的错位**：通过RL训练的推理模型，其核心用途往往不是直接部署，而是作为教师，生成推理链条（reasoning traces）供学生模型蒸馏或冷启动下一轮RL迭代。然而，以"正确解题"为目标训练出的推理链条，并不一定适合学生学习。现有流程依赖大量启发式后处理（如用GPT清理格式、过滤错误答案等）来提升蒸馏数据质量。

**核心洞察**：现实中优秀教师的能力不在于能否独立发现复杂定理，而在于能否利用现有答案，为学生设计清晰有效的解释。本文据此重新定义教师模型的任务——不是从零解题，而是在已知答案的情况下"连接线索"（connect-the-dots），生成对学生最有效的教学性解释。

## 方法详解

### 整体框架

RLT框架颠覆了传统的RL推理训练范式：
- **传统方式**：只给模型题目，让它思考然后解题（稀疏奖励：对/错）
- **RLT方式**：给模型题目+答案，让它生成逐步解释（密集奖励：基于学生的理解程度）

### 关键设计

1. **任务重新定义**：RLT模型的系统提示中包含题目和标准答案，任务是生成连接二者的教学性解释。在测试时，直接提取教师输出的think tokens，替换标签后即可作为学生的蒸馏数据——无需过滤、无需后处理。

2. **密集奖励函数**：通过学生模型的反馈评估教师解释的质量，包含两个互补项：

    - **学生理解度奖励** $r^{SS}$：衡量学生在看到教师的解释后，对标准答案的理解程度。通过学生对答案token的对数概率来量化：

   $$r^{SS}(o_i, s_i, q_i) = \text{avg}\{\log \pi_s^{s_i}\} + \alpha \min\{\log \pi_s^{s_i}\}$$

   其中$\pi_s^{s_i} = \pi_s(s_i | t_{o_i}.q_i)$是学生在看到解释$t_{o_i}$和题目$q_i$后对答案$s_i$的概率。使用avg+min组合确保不忽略任何单个答案token。

   - **逻辑可解释性奖励** $r^{KL}$：确保教师解释中的每一步在学生看来都是合理的逻辑推进。通过教师分布与学生分布在think tokens上的KL散度来衡量：

   $$r^{KL}(o_i, s_i, q_i) = \text{avg}\{\mathbb{D}_{KL}(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}})\} + \alpha \max\{\mathbb{D}_{KL}(\pi_\theta^{t_{o_i}} \| \pi_s^{t_{o_i}})\}$$

   关键区别在于：教师的分布以题目+答案为条件，学生的分布仅以题目为条件。如果教师的某个解释步骤只有在看到答案时才合理，KL散度就会很大，从而惩罚这种"泄露答案"的解释。

   - **最终奖励**：$r_i^{RLT} = r^{SS}(o_i, s_i, q_i) - \lambda r^{KL}(o_i, s_i, q_i)$

3. **训练目标**：基于GRPO算法，使用上述RLT奖励替代传统正确性奖励：

$$J^{RLT}(\theta) = \mathbb{E}_{q,s \sim D, \{o\}_1^G \sim \pi_\theta(\cdot|s,q)} \left[\frac{1}{G}\sum_{i=1}^G \left(A_i^{RLT} - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right)\right]$$

### 训练策略

- 7B参数的Qwen2.5-7B-Instruct作为基座
- RL前进行短暂SFT，适应新的系统提示格式
- 训练仅125步（不到一个epoch），批大小1024，学习率$1 \times 10^{-6}$
- RL期间使用另一个7B模型作为学生计算奖励

## 实验关键数据

### 主实验：蒸馏效果对比

| 模型 | 数据量 | AIME 2024 | MATH 500 | GPQA Diamond | Overall |
|------|--------|-----------|----------|--------------|---------|
| Qwen2.5-7B-Instruct | N.A. | 10.00 | 74.20 | 33.30 | 39.17 |
| Bespoke-7B（R1蒸馏+后处理） | 17K | 20.00 | 82.00 | 37.80 | 46.60 |
| **RLT-7B（无后处理）** | **17K** | **23.30** | **82.80** | **42.40** | **49.50** |
| s1-32B | 1K | 50.00 | 92.60 | 56.60 | 66.40 |
| Bespoke-32B | 17K | 63.30 | 93.00 | 58.10 | 71.47 |
| **RLT-32B** | **17K** | **66.70** | **93.40** | **59.60** | **73.23** |

### 冷启动RL效果对比

| 方法 | AIME 2024 | MATH 500 | GPQA Diamond | Overall |
|------|-----------|----------|--------------|---------|
| RL无冷启动 | 13.30 | 74.20 | 34.80 | 40.77 |
| 传统RL教师(raw)+RL | 10.00 | 71.00 | 34.80 | 38.60 |
| 传统RL教师(GPT后处理)+RL | 16.70 | 78.20 | 36.90 | 43.93 |
| Bespoke-7B+RL | 16.70 | 82.80 | 45.40 | 48.30 |
| **RLT-7B+RL** | **26.70** | **84.00** | **40.90** | **50.53** |

### 关键发现

1. **以小博大**：7B参数的RLT直接产出的原始解释，蒸馏效果优于使用数量级更大模型（如DeepSeek-R1的671B）经过精心过滤和GPT后处理的推理链条。
2. **跨尺度有效**：7B的RLT训练出的解释用于蒸馏32B学生时依然优于所有基线，证明小教师可以教大学生。
3. **零样本迁移**：RLT在从未训练过的countdown任务上零样本产出蒸馏数据，竟然超过了在该任务上直接做RL的效果（55.7% vs 50.8%）。
4. **奖励与蒸馏效果高度相关**：Pearson系数超过0.89，验证了RLT奖励函数的设计有效性。
5. **R1推理链的定性缺陷**：低RLT奖励的R1链条常试图调用外部工具（如计算器）、使用训练数据中的特质语言模式（如幽默评论），RLT解释则更扎实且能自动添加验证步骤。

## 亮点与洞察

- 通过简化任务（从"解题"变"解释"），优雅地规避了RL的探索难题，使得小模型也能通过RL有效训练。
- 密集奖励函数的设计体现了深刻的教学直觉：好的解释不仅要让学生"答对"，每一步推理还要在学生的认知框架内说得通。
- 彻底消除了蒸馏流程中对验证器过滤和后处理的依赖，大幅简化了推理模型的训练pipeline。

## 局限与展望

- RLT训练需要一个学生模型实时计算奖励，增加了训练的计算开销。
- 目前仅在数学和编程任务上验证，其他推理领域（如多步逻辑推理、常识推理）的效果有待探索。
- 教师和学生之间的最佳配对关系尚未充分研究。
- 未来可探索教师-学生协同训练、同一模型交替扮演教师和学生的自蒸馏方案。

## 相关工作与启发

- 与DeepSeek-R1等传统RL推理方法形成鲜明对比：不追求模型独立解题能力，而专注优化蒸馏效果。
- 对推理模型训练的成本民主化有重要意义：将昂贵的RL负担转移到小型专用教师，大模型只需做廉价的SFT。
- 启发了一种新的思考方式——训练目标应与模型的实际部署用途对齐，而非追求表面上的能力提升。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将教师任务从"解题"重定义为"解释"的思路极具创造性
- 实验充分度: ⭐⭐⭐⭐⭐ 多维度评估（蒸馏、冷启动、跨域迁移、奖励分析、定性分析）
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰深刻，实验设计对照严谨
- 价值: ⭐⭐⭐⭐⭐ 为推理模型训练提供了全新且实用的范式

<!-- RELATED:START -->

## 相关论文

- [Test-Time Adaptation with Binary Feedback](../../ICML2025/reinforcement_learning/test-time_adaptation_with_binary_feedback.md)
- [DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](../../ICML2025/reinforcement_learning/revise_learning_to_refine_at_test-time_via_intrinsic_self-verification.md)
- [Complexity Scaling Laws for Neural Models using Combinatorial Optimization](complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)
- [Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)

<!-- RELATED:END -->
