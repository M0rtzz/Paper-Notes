---
title: >-
  [论文解读] FastMCTS: A Simple Sampling Strategy for Data Synthesis
description: >-
  [ACL 2025][蒙特卡洛树搜索] FastMCTS 提出了一种受 MCTS 启发的轻量级推理数据合成策略，通过自适应停留策略、动态探索和保留模拟三个改进，比拒绝采样多生成 30% 以上的正确推理路径，训练出的模型在多个数学基准上平均提升 3.9%。
tags:
  - ACL 2025
  - 蒙特卡洛树搜索
  - 数据合成
  - 推理数据
  - 拒绝采样
  - 偏好优化
---

# FastMCTS: A Simple Sampling Strategy for Data Synthesis

**会议**: ACL 2025  
**arXiv**: [2502.11476](https://arxiv.org/abs/2502.11476)  
**代码**: 无（论文声明即将发布）  
**领域**: LLM推理  
**关键词**: 蒙特卡洛树搜索、数据合成、推理数据、拒绝采样、偏好优化

## 一句话总结

FastMCTS 提出了一种受 MCTS 启发的轻量级推理数据合成策略，通过自适应停留策略、动态探索和保留模拟三个改进，比拒绝采样多生成 30% 以上的正确推理路径，训练出的模型在多个数学基准上平均提升 3.9%。

## 研究背景与动机

**领域现状**：合成高质量的多步推理数据已被证明是提升 LLM 推理能力的有效途径。主流方法是拒绝采样（Rejection Sampling）——对每个问题独立生成多个候选推理路径，保留答案正确的路径作为训练数据。

**现有痛点**：拒绝采样存在两个根本问题：（1）效率低，尤其对于长推理链和困难问题，每次独立采样都从头开始，大量计算浪费在重复生成相同的正确推理前缀上；（2）采样不均衡，简单问题容易生成大量正确路径，困难问题可能一条都找不到，导致训练数据在不同难度级别上严重偏斜。

**核心矛盾**：MCTS 作为一种能有效探索状态空间的搜索算法，理论上可以解决上述问题。但语言模型的推理与围棋等游戏有本质区别——状态空间定义不清晰、推理成本极高（每一步都需要自回归生成）、结果评估相对确定性更强。直接将 MCTS 应用于 LLM 数据合成会带来巨大的模拟开销。

**本文目标**：设计一种适配 LLM 特性的 MCTS 变体，在保持树搜索优势的同时大幅降低额外开销。

**切入角度**：作者观察到 LLM 推理中模拟结果与推理路径质量高度相关（不同于围棋中单次模拟不能确定状态好坏），因此可以保留模拟过程中生成的所有路径而非丢弃，同时通过自适应策略避免在不值得深入的节点上浪费搜索。

**核心 idea**：对 MCTS 进行三个关键修改——自适应停留策略（根据节点胜率决定是否继续深入）、动态探索参数（基于节点评分调整探索-利用平衡）、保留模拟（将模拟过程中的完整推理轨迹作为树节点缓存而非丢弃）。

## 方法详解

### 整体框架

FastMCTS 以输入问题 $q$ 为根节点，每个推理步骤为子节点，迭代构建搜索树。每次迭代包含选择（Selection）和模拟/扩展（Simulation/Expansion）两个阶段（合二为一），然后通过验证结果回传更新节点评分。搜索完成后，从树中提取正确的推理路径用于 SFT 训练，利用不同分支间的评分差异构造偏好对用于 DPO 训练。

### 关键设计

1. **自适应停留策略（Adaptive Stay Policy）**:

    - 功能：动态决定选择阶段是否继续向叶节点深入
    - 核心思路：在标准 MCTS 的选择阶段，算法总是递归选择到叶节点再扩展。FastMCTS 引入停留条件：如果当前节点的蒙特卡洛估计分数 $score = win\_count / visit\_count$ 要么非常高（接近 1）要么非常低（接近 0），则"停留"在当前节点直接扩展新分支，而不继续向下选择。高分停留保证简单问题的路径多样性（类似退化为拒绝采样），低分停留避免在几乎必定失败的路径上继续浪费。具体地，当分数 $\in (0, l_{low}] \cup [l_{high}, 1)$ 时触发停留
    - 设计动机：对于简单问题，过深的树搜索反而减少了路径多样性（共享前缀过多）；对于极难问题，继续深入搜索不如尝试全新的路径

2. **动态探索参数（Dynamic Exploration）**:

    - 功能：根据节点评分动态调整 UCT 公式中的探索参数 $c$
    - 核心思路：标准 UCT 公式为 $UCT(i) = \frac{w_i}{n_i} + c \cdot \sqrt{\frac{\ln N_i}{n_i}}$，FastMCTS 在节点被访问超过一次后，将 $c$ 乘以该节点的评分。对于高分节点（有前途的），增大探索权重以发现更多正确路径；对于低分节点（不太有前途的），减小探索权重以集中利用已知的较好分支
    - 设计动机：数据合成的目标是尽可能多地生成正确路径，因此在有前途的状态下应该更积极探索，这与游戏中的探索-利用平衡策略不同

3. **保留模拟（Reserve Simulation）**:

    - 功能：将模拟阶段生成的完整推理轨迹保留为树节点，避免浪费
    - 核心思路：标准 MCTS 的模拟阶段会从选中节点开始随机推演到终局，但推演结果只用于回传评分更新，生成的具体路径被丢弃。在 LLM 场景下，每次模拟都需要自回归生成完整推理链，丢弃这些路径是巨大的浪费。FastMCTS 将扩展和模拟合并为一个阶段，从选中节点开始生成新的推理分支直到得出最终答案，整个生成过程中的每一步都作为新节点添加到树中。此外，每次模拟前随机拼接不同的 few-shot 示例以增加推理路径多样性
    - 设计动机：LLM 推理中最终答案的正确性与中间步骤高度相关（不同于围棋的随机模拟），因此模拟结果本身就是有价值的数据

### 损失函数 / 训练策略

FastMCTS 生成的数据支持两种训练方式：（1）SFT：从树中提取正确推理路径，每个问题优先从不同分支选择以最大化多样性；（2）Branch-DPO：利用树节点的评分差异构造偏好对——蒙特卡洛估计分数持续为零的节点被视为"低质量节点"，与通向正确结果的节点配对构成 step-level 或 branch-level 偏好数据。DPO 使用 $\beta = 0.4$，学习率 $1 \times 10^{-6}$，AdamW 优化器。

## 实验关键数据

### 主实验

基于 Qwen2.5-7B 在英文数学推理基准上的训练结果（EN Math Hard 数据源）：

| 方法 | #Data | GSM8K | MATH | AIME24 | AMC23 | OmniMath | 平均 |
|------|-------|-------|------|--------|-------|----------|------|
| Qwen2.5-7B（基线） | - | 88.2 | 66.8 | 0 | 47.5 | 35.5 | 49.7 |
| RS (5 traj/prob) | 111K | 89.1 | 72.0 | 6.7 | 52.5 | 38.3 | 52.4 |
| FastMCTS (5 traj) | 132K | 88.9 | 73.0 | 13.3 | 57.5 | 39.8 | 54.8 |
| RS (16 traj/prob) | 197K | 87.1 | 70.0 | 10.0 | 52.5 | 37.2 | 52.7 |
| FastMCTS (16 traj) | 288K | 88.9 | 74.0 | 20.0 | 60.0 | 38.3 | 55.6 |
| + Branch-DPO | 152K | 89.9 | 75.4 | 20.0 | 57.5 | 39.2 | **56.6** |

### 消融实验

在 300 个 AIME 问题上，每个问题采样 25 条推理轨迹：

| 方法 | 问题解决率(%) | 平均正确路径数 |
|------|-------------|--------------|
| Rejection Sampling | 61.3 | 7.22 |
| FastMCTS（完整） | 61.7 | 7.95 |
| w/o Adaptive Stay | 55.9 | 7.59 |
| w/o Dynamic Exploration | 61.7 | 7.28 |
| w/o Stay & Dynamic | 55.9 | 7.32 |

### 关键发现

- **采样效率优势显著**：随着生成 token 数量增加，FastMCTS 比拒绝采样多生成超过 30% 的正确推理路径，有效 token 占比也更高
- **自适应停留策略主要影响问题解决率**（61.7% vs 55.9%）：它根据当前路径的估计质量决定是深入还是开辟新路径，对困难问题尤为重要
- **动态探索主要影响正确路径数量**（7.95 vs 7.28）：它通过调整探索参数在有前途的分支上发现更多正确路径
- FastMCTS 在困难问题上优势最明显——在 AIME 级别的竞赛题上，模型从 0 分提升到 20 分，而拒绝采样只到 10 分
- Branch-DPO 提供了额外的性能增益（55.6% → 56.6%），证明树结构数据的 step-level 偏好信号确实有效
- 中文数学数据集上趋势一致：FastMCTS 在高考数学 2024 上达到 62.3%，拒绝采样为 60.9%

## 亮点与洞察

- **保留模拟**的设计非常巧妙——LLM 与围棋的关键区别在于，LLM 的模拟结果本身就是有价值的推理路径，丢弃它们纯粹是浪费。这个看似简单的修改却带来了显著的效率提升，体现了对问题本质的深刻理解
- **难度自适应采样**是 FastMCTS 的核心优势之一。拒绝采样对所有问题一视同仁，FastMCTS 则自动在困难问题上投入更多搜索预算，在简单问题上退化为高效的拒绝采样。这种难度感知的平衡在实际应用中非常有价值
- 从树结构自然衍生出的 step-level 偏好数据是一个额外收益——同一棵树上的不同分支天然形成对比对，不需要额外标注

## 局限与展望

- 仅使用 Qwen2.5-72B-Instruct 作为策略模型，未测试更强的推理模型（如 DeepSeek-R1、o1），性能上限可能更高
- 由于计算资源限制，训练实验仅在 Qwen2.5-7B 上进行，更大模型上的效果未知
- 树结构中共享前缀导致的推理路径多样性下降是一个待研究的问题
- 可以探索将 FastMCTS 与 process reward model 结合，用学习到的过程奖励替代蒙特卡洛评估
- 代码尚未开源，可复现性有待确认

## 相关工作与启发

- **vs Rejection Sampling**: FastMCTS 在相同计算预算下系统性优于拒绝采样——更多正确路径、更高有效 token 率、更均衡的难度分布
- **vs 标准 MCTS (AlphaMath, REST-MCTS*)**: 这些方法直接应用 MCTS 到 LLM 推理，模拟开销大。FastMCTS 通过保留模拟和自适应策略大幅降低了开销
- **vs DART-Math**: 同样关注难度均衡的采样，但 DART-Math 通过事后过滤实现，FastMCTS 在采样过程中动态调节，更高效
- 该方法的树搜索思路可以扩展到代码生成、逻辑推理等其他需要多步推理的数据合成场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 三个改进点各自简单但组合效果好，特别是保留模拟的洞察敏锐
- 实验充分度: ⭐⭐⭐⭐⭐ 采样效率对比 + 中英文训练结果 + 详细消融 + 难度分布分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细且配有完整算法伪代码
- 价值: ⭐⭐⭐⭐⭐ 作为拒绝采样的直接替代方案，实用价值极高，对推理数据合成方向有直接影响

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)
- [\[ACL 2025\] AceCoder: Acing Coder RL via Automated Test-Case Synthesis](acecoder_acing_coder_rl_via_automated.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)

<!-- RELATED:END -->
