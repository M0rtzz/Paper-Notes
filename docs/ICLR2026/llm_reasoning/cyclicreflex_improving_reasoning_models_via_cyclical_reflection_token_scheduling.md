---
title: >-
  [论文解读] CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling
description: >-
  [ICLR 2026][LLM推理][大语言推理模型] 将推理过程中的反思token（如"wait"、"but"）视为可调度的"资源"，借鉴优化中周期性学习率的思想，提出CyclicReflex——一种免训练的解码策略，通过三角波形动态调控反思token的logit，在多个数学推理基准上（MATH500, AIME2024/2025, AMC2023）一致性提升1.5B-8B模型准确率。
tags:
  - ICLR 2026
  - LLM推理
  - 大语言推理模型
  - 反思token调度
  - 测试时缩放
  - 周期性学习率
  - 解码策略
---

# CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling

**会议**: ICLR 2026  
**arXiv**: [2506.11077](https://arxiv.org/abs/2506.11077)  
**代码**: [https://github.com/OPTML-Group/CyclicReflex](https://github.com/OPTML-Group/CyclicReflex)  
**领域**: LLM Reasoning  
**关键词**: 大语言推理模型, 反思token调度, 测试时缩放, 周期性学习率, 解码策略

## 一句话总结
将推理过程中的反思token（如"wait"、"but"）视为可调度的"资源"，借鉴优化中周期性学习率的思想，提出CyclicReflex——一种免训练的解码策略，通过三角波形动态调控反思token的logit，在多个数学推理基准上（MATH500, AIME2024/2025, AMC2023）一致性提升1.5B-8B模型准确率。

## 研究背景与动机
大型推理模型（LRM）如OpenAI o1、DeepSeek-R1通过多步推理来解决复杂问题，推理过程由"反思token"（如"wait"、"but"、"alternatively"）引导。这些token在推理轨迹中起到关键的转折和自我评估作用。

然而，现有LRM存在两个对称性的问题：
- **思考不足（under-reflection）**：反思token过少，模型过早终止推理，无法充分探索解题路径，类似于学习率过小导致优化过早收敛
- **过度思考（over-reflection）**：反思token过多，模型反复循环（如不停输出"wait"），浪费计算资源且无法收敛到正确答案，类似于学习率过大导致优化发散

现有方法如TIP（Thought switching penalty）只能单向抑制反思token、且使用固定的logit惩罚，无法同时应对不同难度问题的under/over-reflection问题。作者提出核心问题：**如何通过资源分配策略动态调节反思token的使用频率和位置？** 其核心洞察是：将反思token调度类比为优化中的学习率调度，特别是借鉴周期性学习率（cyclical learning rate）的"步长对冲"思想。

## 方法详解

### 整体框架
CyclicReflex是一种training-free的解码策略，在自回归生成过程中，根据当前token位置动态调整反思token的logit值。输入为问题$\mathbf{x}$，输出为推理轨迹$\mathbf{r}$和最终答案$\mathbf{y}$。不需要修改模型参数，仅在推理阶段工作。

### 关键设计

1. **反思token资源分配问题的形式化**：将推理过程中的反思token（"wait"、"but"、"alternatively"等）视为可调度的"资源"，其频率和位置直接影响推理质量。通过TIP的扩展实验（允许正负$\alpha$），发现：TIP(-3)在Hard问题上提升最大但在Easy问题上严重下降；TIP(+1)在Easy上略有提升。这说明单一固定策略无法适应不同难度。

2. **思维景观验证类比**：利用Landscape of Thoughts可视化工具，将推理步骤投影到2D空间。验证了三种模式：

    - under-reflection：推理轨迹过于保守，无法远离起始点
    - desired-reflection：轨迹结构良好，收敛到正确答案
    - over-reflection：模型曾到达接近正确答案的区域（如"Alternatively, perhaps the correct answer is..."），但反思过度导致快速越过该区域，最终偏离正确答案

3. **CyclicReflex核心公式**：采用位置相关的双向三角波形调制反思token的logit：

$$\hat{z}_{t,v} = \begin{cases} z_{t,v} + \delta(t) & \text{if } v \in \hat{V} \\ z_{t,v} & \text{otherwise} \end{cases}$$

$$\delta(t) = A \left| \frac{4 \cdot (t - C/4) \bmod C}{C} - 2 \right| - A$$

其中$A$是振幅（控制调整强度），$C$是周期（控制振荡频率），$\hat{V}$是反思token集合。$\delta(C/4) = A$为正值促进反思，$\delta(3C/4) = -A$为负值抑制反思。

4. **与TIP的关键区别**：

    - TIP是**单向静态**的（固定$\alpha \leq 0$），仅抑制反思token
    - CyclicReflex是**双向动态**的，交替促进和抑制反思
    - 上升阶段鼓励探索（转换思路），下降阶段促进收敛（稳定推理）
    - 类似于周期性学习率的stepsize hedging策略

### 训练策略
本方法完全不需要训练，是纯粹的inference-time策略。超参数通过网格搜索确定：$A \in [1, 10]$，$C \in [200, 2000]$（因数据集而异）。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 指标 | Original | TIP | S1 | Silver | CyclicReflex |
|--------|------|------|----------|-----|----|----- --|-------------|
| MATH500 | Qwen-7B | Acc | 0.86 | 0.87 | 0.83 | 0.88 | **0.89** |
| AIME2024 | Qwen-7B | Acc | 0.43 | 0.43 | 0.33 | 0.37 | **0.50** |
| AIME2025 | Qwen-7B | Acc | 0.31 | 0.30 | 0.33 | 0.30 | **0.37** |
| AMC2023 | Qwen-7B | Acc | 0.81 | 0.85 | 0.85 | 0.85 | **0.90** |
| AIME2024 | Llama-8B | Acc | 0.42 | 0.47 | 0.43 | 0.47 | **0.53** |
| AMC2023 | Llama-8B | Acc | 0.81 | 0.85 | 0.75 | 0.85 | **0.90** |
| MATH500 | Qwen-1.5B | Acc | 0.74 | 0.75 | 0.73 | 0.75 | **0.77** |
| AIME2024 | Qwen-1.5B | Acc | 0.23 | 0.23 | 0.17 | 0.27 | **0.30** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同难度级别 | Easy/Medium/Hard均提升 | TIP仅在Hard上有效，Easy反而下降 |
| +Best-of-N(N=8) | 持续提升BoN准确率 | 与外部test-time方法兼容互补 |
| +Beam Search | 持续提升BS准确率 | 低预算时增益更明显 |
| 初始相位$\phi=0$最优 | - | 初期鼓励反思、后期抑制效果最佳 |
| 周期$C$影响更大 | 准确率对$C$更敏感 | $C=600$在Qwen-7B/MATH500最优 |
| 振幅$A$控制长度 | 主要影响反思token数量和生成长度 | $A$越大推理越长 |

### 关键发现
- CyclicReflex在所有模型规模（1.5B-8B）和所有数据集上均一致性提升，同时生成长度与原始解码策略相当
- 自我纠错能力显著增强：在给定错误推理轨迹（100%长度）的条件下，CyclicReflex的纠错率远高于TIP和原始解码
- 生成的思维景观更集中，干扰区域更少，推理轨迹更容易直接收敛到正确答案
- S1（强制插入"Wait"）在AMC2023上严重下降，说明简单地增加反思token是不够的

## 亮点与洞察
- **类比眼光非常精到**：将反思token调度类比为学习率调度，under-reflection ↔ 学习率过小 ↔ 过早收敛，over-reflection ↔ 学习率过大 ↔ 振荡发散。这个类比不仅直觉上合理，而且通过思维景观可视化得到了很好的验证
- **极简但有效的设计**：整个方法就是一个三角波形函数，没有任何可学习参数，极易实现且零额外开销
- **双向性是关键创新**：相比TIP的单向抑制，CyclicReflex交替促进和抑制反思的能力使其能够适应不同难度的问题
- **与外部test-time scaling方法完美兼容**：与Best-of-N和Beam Search的组合均能进一步提升

## 局限与展望
- 理论基础仍偏弱：为什么LRM会出现over/under-reflection的根本原因未被阐明
- 超参数（$A$和$C$）需要针对每个数据集做网格搜索，缺乏自适应机制
- 仅在数学推理任务上验证，未测试代码生成、逻辑推理等其他推理场景
- 反思token的定义（"wait"、"but"等）较为启发式，不同模型的反思模式可能不同
- 初始相位$\phi$的最优值（$\phi=0$）暗示了更深层的推理动态规律，值得进一步探索

## 相关工作与启发
- **TIP**（Wang et al., 2025a）：通过固定惩罚抑制反思token，解决overthinking问题，是本文直接的baseline
- **S1**（Muennighoff et al., 2025）：强制在thinking tag后插入"Wait"，但效果不稳定
- **Silver Stepsize Schedule**（Altschuler & Parrilo, 2024）：优化理论中的步长对冲策略，理论上可证明加速收敛
- **Cyclical Learning Rates**（Smith, 2017）：深度学习中的周期性学习率策略，是本文核心灵感来源
- 启发：优化理论中的调度策略可能对LLM推理过程有更广泛的指导意义

## 评分
- 新颖性: ⭐⭐⭐⭐ （类比新颖，但方法本身相对简单）
- 实验充分度: ⭐⭐⭐⭐⭐ （多模型、多数据集、消融实验详尽、可视化分析到位）
- 写作质量: ⭐⭐⭐⭐⭐ （叙事流畅，类比清晰，图表出色）
- 价值: ⭐⭐⭐⭐ （实用性强，但理论基础有待加强）

<!-- RELATED:START -->

## 相关论文

- [Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](../../NeurIPS2025/llm_reasoning/one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](../../ACL2025/llm_reasoning/towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)
- [Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_ch.md)
- [ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](../../ACL2025/llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)

<!-- RELATED:END -->
