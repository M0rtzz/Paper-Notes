---
title: >-
  [论文解读] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction
description: >-
  [ACL 2025 (Findings)][LLM 其他][Theory of Mind] 本文探索了从开源 LLM（LLaMA）内部表征中提取心智理论（ToM）相关信息的可行性，并利用 BDI（信念-愿望-意图）框架操纵这些表征来生成更符合人类社交认知的对话回复，ToM 对齐后的模型在 3B 和 8B 上分别达到 67% 和 63% 的胜率。
tags:
  - "ACL 2025 (Findings)"
  - "LLM 其他"
  - "Theory of Mind"
  - "BDI 框架"
  - "对话对齐"
  - "LLM 可解释性"
  - "LatentQA"
---

# Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction

**会议**: ACL 2025 (Findings)  
**arXiv**: [2502.14171](https://arxiv.org/abs/2502.14171)  
**代码**: 有（论文提及代码仓库）  
**领域**: 其他  
**关键词**: Theory of Mind, BDI 框架, 对话对齐, LLM 可解释性, LatentQA

## 一句话总结

本文探索了从开源 LLM（LLaMA）内部表征中提取心智理论（ToM）相关信息的可行性，并利用 BDI（信念-愿望-意图）框架操纵这些表征来生成更符合人类社交认知的对话回复，ToM 对齐后的模型在 3B 和 8B 上分别达到 67% 和 63% 的胜率。

## 研究背景与动机

随着 LLM 驱动的对话助手在各个领域的深入应用，它们在社交语境理解和非字面语言对齐方面的不足逐渐暴露。人类在交流中本能地依赖心智理论（Theory of Mind, ToM）——即推断对话者的信念、愿望和意图——来调整自己的表达。然而，当前 LLM 在这方面的能力还存在很大争议：

**ToM 缺失问题**：现有 LLM 在谈判、博弈等需要推理对方心智状态的场景中常常表现不佳，生成不符合语境的回复

**对齐层面的挑战**：传统对齐方法（RLHF、DPO 等）在语用层面的对齐效果有限，无法捕捉到对话的语用和社交细微差别

**研究空白**：虽然已有工作评估 LLM 的 ToM 能力或设计 mind module，但尚无研究尝试从 LLM 内部提取 ToM 表征并将其应用于对齐回复生成

本文的核心动机是：如果 LLM 在因果语言建模过程中确实保留了 ToM 相关线索，那么能否利用这些线索来改善对话回复的社交对齐性？

## 方法详解

### 整体框架

本文围绕三个研究问题，设计了三组互补实验：
- **RQ1（读取 ToM）**：LLM 内部表征中编码了多少 ToM 信息？
- **RQ2（ToM 一致性）**：提取的 ToM 信息是否可靠、非幻觉性的？
- **RQ3（ToM 可控生成）**：能否利用 ToM 表征来增强回复对齐？

### 关键设计

1. **ToM 信息读取（RQ1）**：

    - **线性探测（Linear Probing）**：从 LLM 残差流中提取最后一个 token 的隐藏状态 $h(S)$，学习权重矩阵 $W$ 和偏置 $b$ 映射到 ToM 标签空间：$\hat{y} = \text{softmax}(Wh(S) + b)$
    - **LatentQA**：将对话输入冻结的目标模型获取内部表征 $R(S)$，然后由解码器模型接收 ToM 问题和 $R(S)$ 生成答案——这种方法能利用整个激活序列而非单一嵌入
    - **多层深度实验**：分别在浅层、中间层和深层提取表征，研究层深度对 ToM 读取准确率的影响

2. **ToM 一致性验证（RQ2）**：

    - 使用 FanToM 和 NegotiationToM 等专门设计的一致性评估数据集
    - 评估时要求模型对相同 ToM 场景的不同类型问题给出逻辑一致的答案——如果多选题答对但事实问答矛盾，则不算正确
    - 比较 LatentQA、微调和 CoT 推理三种方法的一致性

3. **ToM 可控生成（RQ3）**：

    - 核心思路：将目标模型的内部表征从 $R(S)$ 修改为 $R'(S)$，针对性地操纵特定角色的信念、愿望或意图
    - 梯度流通过比较生成答案与实际 ToM 答案来优化解码器，然后用该梯度 boost 目标表征中的特定 ToM 组件
    - 修改后仅使用目标模型生成对齐回复 $C''$，与未修改模型的回复 $C'$ 对比

### 损失函数 / 训练策略

- 线性探测：标准交叉熵损失
- LatentQA 解码器训练：使用 ground-truth ToM 标注的监督学习
- ToM 可控生成：基于梯度的表征 boosting，无额外损失函数，而是通过反向传播路径增强目标表征

## 实验关键数据

### 主实验：ToM 读取准确率（表格）

| 模型 | 层深度 | CaSiNo（精确匹配，Both-A1-A2） | CRAIGSLISTBARGAIN（R²，Seller-Buyer） |
|------|--------|-------------------------------|--------------------------------------|
| LLaMA3-1B | 中间 | LP: 02-16-13 / LQA: 20-42-39 | LP: 0.26-0.26 / LQA: 0.89-0.92 |
| LLaMA3-3B | 中间 | LP: 05-23-21 / LQA: 29-60-44 | LP: 0.19-0.27 / LQA: 0.96-0.98 |
| LLaMA3-8B | 中间 | LP: 02-10-23 / LQA: 46-62-70 | LP: 0.36-0.40 / LQA: 0.93-0.91 |

> LP = Linear Probing, LQA = LatentQA。LatentQA 在中间层表现最佳，远超线性探测。

### ToM 一致性与可控性结果（表格）

| 模型 | 方法 | FanToM ALL* | NegotiationToM ALL |
|------|------|------------|-------------------|
| LLaMA3-3B | LatentQA | 11.9 | 6.2 |
| LLaMA3-3B | 微调 | 8.2 | 11.2 |
| LLaMA3-8B | LatentQA | 16.4 | 15.2 |
| LLaMA3-8B | 微调 | 12.8 | 17.7 |
| GPT-4o-mini | CoT | 0.5 | 4.8 |

> ToM 一致性整体仍较低，但 LatentQA 比 CoT 推理有明显改善。

### 关键发现

1. **中间层最优**：LatentQA 在 6 组实验中有 5 组在中间层表现最佳，浅层语义不够丰富，深层可能受预训练自我中心偏差影响
2. **模型越大效果越好**：8B 模型在 ToM 读取上显著优于 1B 和 3B
3. **可控生成有效**：ToM 对齐后，3B 模型的加权平均胜率为 67.15%，8B 为 63.25%
4. **3B 改善幅度更大**：可能因为 8B 未对齐时基线已较强
5. **"共情表达"和"需求描述"效果最差**：这些意图在预训练文本中已经很常见，模型天生就能表达

## 亮点与洞察

- **理论贡献突出**：首次将 ToM 的 BDI 框架与 LLM 内部表征操控结合，建立了从"读取→验证→控制"的完整链路
- **LatentQA 优势明显**：相比线性探测只看单个激活向量，LatentQA 利用完整激活序列，能捕获更丰富的 ToM 信息
- **实用潜力**：证明了即使 ToM 一致性不完美，通过增强正确的 ToM 信息仍能在实际中改善对齐效果
- **无需微调整个模型**：通过操控内部表征即可实现对齐，计算效率远优于 RLHF/DPO

## 局限与展望

1. **评估依赖 LLM 裁判**：使用 GPT-4o、o1、Gemini 作为评判，缺乏人类评估
2. **数据集单一**：可控性实验仅在 NegotiationToM 一个数据集上进行
3. **超参敏感**：对齐生成过程高度依赖超参调优，影响稳定性和可复现性
4. **模型家族有限**：仅测试了 LLaMA3 家族，未验证不同架构的泛化性
5. **实际部署差距**：当前 ToM 问答对需人工设计，实际应用需要用 LLM 动态规划执行
6. **伦理风险**：ToM 能力可能被用于操纵用户心理状态，需要透明设计和知情同意

## 相关工作与启发

- **LatentQA (Pan et al., 2024)** 是本文方法的核心基础，将内部表征解读框架化为视觉问答问题
- **NegotiationToM (Chan et al., 2024)** 提供了带有 BDI 标注的谈判对话数据集
- **街头 ToM 对齐展望 (Street, 2024)** 指出了将 ToM 用于 LLM 对齐的未来方向
- 本文启发：内部表征操控是一种有潜力的轻量级对齐手段，适合在对话系统中追求更细粒度的人格和社交行为控制

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次将 ToM+BDI 与内部表征操控结合用于对话对齐 |
| 实验充分度 | 3.5 | 三组 RQ 实验设计完整，但数据集和模型家族偏少 |
| 写作质量 | 4 | 问题形式化清晰，实验结构条理分明 |
| 价值 | 4 | 为 LLM 对齐提供了新范式，兼具理论深度和实用潜力 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[ACL 2025\] Enhancing Hyperbole and Metaphor Detection with Their Bidirectional Dynamic Interaction and Emotion Knowledge](hyperbole_metaphor_detection.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](../../ICCV2025/others/syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)

</div>

<!-- RELATED:END -->
