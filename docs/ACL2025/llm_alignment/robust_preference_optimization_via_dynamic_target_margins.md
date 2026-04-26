---
title: >-
  [论文解读] Robust Preference Optimization via Dynamic Target Margins
description: >-
  [ACL 2025 (Findings)][LLM对齐][偏好优化] 本文提出 γ-PO，一种通过在偏好对级别动态调整目标奖励边际的方式来增强 DPO 鲁棒性的即插即用方法，在 AlpacaEval2 和 Arena-Hard 上平均提升 4.4%。
tags:
  - ACL 2025 (Findings)
  - LLM对齐
  - 偏好优化
  - 动态边际
  - 噪声鲁棒性
  - DPO改进
  - 奖励边际
---

# Robust Preference Optimization via Dynamic Target Margins

**会议**: ACL 2025 (Findings)  
**arXiv**: [2506.03690](https://arxiv.org/abs/2506.03690)  
**代码**: [https://github.com/sunjie279/gammaPO](https://github.com/sunjie279/gammaPO)  
**领域**: 对齐RLHF  
**关键词**: 偏好优化, 动态边际, 噪声鲁棒性, DPO改进, 奖励边际

## 一句话总结

本文提出 γ-PO，一种通过在偏好对级别动态调整目标奖励边际的方式来增强 DPO 鲁棒性的即插即用方法，在 AlpacaEval2 和 Arena-Hard 上平均提升 4.4%。

## 研究背景与动机

**领域现状**：大语言模型 (LLM) 的对齐是保证安全性与实用性的关键步骤。Direct Preference Optimization (DPO) 作为一种无需单独训练奖励模型的高效对齐方法，通过直接利用偏好对来优化模型策略，大幅降低了资源需求，已成为主流对齐方案之一。

**现有痛点**：DPO 的有效性严重依赖训练数据的质量，但现实中的偏好数据往往包含大量噪声——标注者之间的分歧、标注标准不一致、偏好程度的模糊性等问题普遍存在。现有 DPO 方法对所有偏好对施加统一的优化目标，无法区分高置信度样本与模糊样本，导致模型可能被噪声数据误导。

**核心矛盾**：DPO 隐含的假设是所有偏好对都同等可靠，但实际上不同偏好对的置信度差异很大——有些对之间的质量差距非常明显（高奖励边际），有些对之间几乎没有差别（低奖励边际），甚至可能存在标注错误。对这些样本一视同仁会让模型学到错误的偏好信号。

**本文目标**：设计一个能够根据偏好对的质量差异自适应调整优化目标的算法，使模型优先从高置信度样本中学习，同时抑制低置信度（可能含噪声）样本的影响。

**切入角度**：作者观察到偏好对中 chosen 和 rejected 响应之间的奖励边际（reward margin）天然反映了样本的置信度——边际越大说明偏好越明确，越值得信赖。

**核心 idea**：引入实例级别的动态目标边际（dynamic target margin），根据每个偏好对的奖励差值自动校准优化强度，对高置信度对加大学习力度，对模糊对降低学习力度。

## 方法详解

### 整体框架

γ-PO 建立在标准 DPO 框架之上。输入是偏好数据集中的 (prompt, chosen, rejected) 三元组，输出是经过对齐的策略模型。整体流程分为两步：首先利用参考模型计算每个偏好对的隐式奖励边际，然后根据边际值动态设定该偏好对的目标边际 γ，将其融入 DPO 类损失函数中进行训练。

### 关键设计

1. **动态目标边际校准 (Dynamic Target Margin Calibration)**:

    - 功能：为每个偏好对计算一个实例特定的目标边际 γ，替代 DPO 中隐含的固定边际
    - 核心思路：利用参考模型对 chosen 和 rejected 响应分别计算对数概率，其差值 $r_{\text{margin}} = \log \pi_{\text{ref}}(y_w|x) - \log \pi_{\text{ref}}(y_l|x)$ 作为该偏好对的置信度指标。然后通过一个单调递增的映射函数将奖励边际转化为目标边际 γ，使得高置信度对获得更大的目标边际（更强的优化推力），低置信度对获得更小的目标边际（更弱的优化推力）
    - 设计动机：参考模型在训练前就已经存在对样本质量的先验判断，利用这个信号可以无需额外标注就过滤噪声。高奖励边际意味着参考模型已经能明确区分两个响应的优劣，这类样本大概率是正确标注的

2. **噪声抑制机制 (Noise Suppression via Margin Thresholding)**:

    - 功能：通过边际阈值机制自动降低模糊偏好对的训练影响
    - 核心思路：当偏好对的奖励边际低于某个阈值时，其对应的目标边际 γ 接近于0，相当于该样本对训练梯度的贡献被压制。这实现了一种软性的样本过滤效果，而非简单地丢弃样本。具体来说，映射函数被设计为在边际较低时有较平缓的斜率，在边际较高时有较陡峭的斜率
    - 设计动机：硬性过滤（直接丢弃低置信度样本）会浪费数据且需要手动设定阈值，而软性抑制机制允许所有样本参与训练，只是自动调整其影响权重，更加灵活和数据高效

3. **即插即用的通用适配 (Plug-and-Play Compatibility)**:

    - 功能：γ-PO 可以无缝集成到所有基于奖励边际的 DPO 变体中
    - 核心思路：γ-PO 的核心修改仅在损失函数中加入动态目标边际项，不改变模型架构、训练流程或数据格式。对于 SimPO、IPO、KTO 等 DPO 变体，只需在其损失函数的边际相关部分插入 γ 即可
    - 设计动机：偏好优化领域有众多 DPO 变体，设计为即插即用方式可以最大化 γ-PO 的实用价值。实验表明只需几行代码的修改，且对训练效率几乎没有影响

### 损失函数 / 训练策略

γ-PO 的损失函数在标准 DPO 损失基础上加入了动态边际项。以 DPO 为例，标准损失为 $\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta (r_w - r_l))$，γ-PO 将其修改为 $\mathcal{L}_{\gamma\text{-PO}} = -\log \sigma(\beta (r_w - r_l - \gamma))$，其中 $\gamma$ 是根据参考模型计算的动态目标边际。训练策略保持与基线方法一致，不引入额外的超参数调节。

## 实验关键数据

### 主实验

| 基准测试 | 指标 | γ-PO (DPO) | DPO | SimPO | γ-PO (SimPO) |
|----------|------|------------|-----|-------|-------------|
| AlpacaEval2 | LC Win Rate (%) | +4.2 vs DPO | baseline | baseline | +3.8 vs SimPO |
| Arena-Hard | Win Rate (%) | +4.6 vs DPO | baseline | baseline | +4.9 vs SimPO |
| 平均 | 提升幅度 | **+4.4%** | - | - | **+4.4%** |

γ-PO 在多个 DPO 变体上均取得了一致性的提升，且在不同基模型（如 Llama 系列、Mistral 系列）上都有效。

### 消融实验

| 配置 | AlpacaEval2 | Arena-Hard | 说明 |
|------|------------|------------|------|
| Full γ-PO | 最优 | 最优 | 完整动态边际 |
| 固定 γ (常数) | 下降 | 下降 | 退化为带偏移的DPO |
| 无 γ (γ=0) | 基线 | 基线 | 等同标准DPO |
| 反向 γ (低边际→大目标) | 显著下降 | 显著下降 | 验证了方向正确性 |

### 关键发现

- 动态边际的方向至关重要：高置信度样本配大边际是正确的，反向操作会显著伤害性能
- γ-PO 在不同 DPO 变体（DPO、SimPO、IPO）上都能带来一致提升，验证了方法的通用性
- 训练效率方面几乎没有额外开销，代码改动极小（仅几行），工程落地门槛极低
- 在数据含噪严重的场景下，γ-PO 的提升更加明显，验证了其噪声鲁棒性

## 亮点与洞察

- **利用参考模型的先验知识做样本加权**：这一思路非常巧妙——参考模型本身就携带了关于响应质量的信息，直接利用它来衡量偏好对的可靠性，避免了额外的质量评估步骤
- **即插即用的设计哲学**：在研究社区中 DPO 变体层出不穷的背景下，设计一个能无缝适配所有变体的通用改进方案，极大地提升了实际影响力
- **动态边际可迁移到其他对比学习场景**：这种根据样本难度自适应调整边际的思路不仅适用于偏好优化，也可以推广到检索、推荐等使用对比损失的场景

## 局限与展望

- 论文主要在英文对话场景下验证，多语言场景和特定领域（如代码生成、数学推理）的效果有待进一步确认
- 参考模型本身的质量会直接影响边际估计的准确性，如果参考模型本身很弱，则边际信号的可靠性存疑
- 动态边际的映射函数形式（如线性、分段线性）的选择对最终性能的影响未充分探讨
- 可以考虑结合在线学习，随着训练进行根据模型当前状态动态更新边际估计，而非仅依赖固定的参考模型

## 相关工作与启发

- **vs DPO**: DPO 对所有偏好对施加统一优化目标，γ-PO 引入实例级自适应边际，在保持简洁性的同时显著提升鲁棒性
- **vs SimPO**: SimPO 通过改变奖励定义引入长度归一化边际，γ-PO 与之正交互补——可以在 SimPO 基础上再叠加动态边际
- **vs RSO/DPOP**: 部分工作通过改变损失函数形式或加入正则项来抑制噪声，而 γ-PO 从样本加权角度出发，思路更直接且实现更简便

## 评分

- 新颖性: ⭐⭐⭐ 核心思路（动态边际）简洁但不算全新，类似课程学习和样本加权的idea在其他领域已有广泛应用
- 实验充分度: ⭐⭐⭐⭐ 多基准、多基模型、多DPO变体的全面验证，消融设计合理
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机论述逻辑通顺
- 价值: ⭐⭐⭐⭐ 即插即用且有效的RLHF改进方法，工程价值高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/llm_alignment/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](../../ICLR2026/llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)
- [\[ICCV 2025\] MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](../../ICCV2025/llm_alignment/magicid_hybrid_preference_optimization_for_id-consistent_and_dynamic-preserved_v.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)

<!-- RELATED:END -->
