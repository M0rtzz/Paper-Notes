---
title: >-
  [论文解读] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs
description: >-
  [ICCV 2025][强化学习][帧选择] 提出 mDP3，一种免训练、模型无关的视频帧选择方法，通过条件高斯核在 RKHS 中估计帧相似度，结合行列式点过程（DPP）捕获查询相关性和列表级多样性，再通过马尔可夫决策过程（MDP）建模时序性，在多个长视频 benchmark 上以仅 8 帧输入显著超越均匀采样和现有帧选择方法。
tags:
  - "ICCV 2025"
  - "强化学习"
  - "帧选择"
  - "行列式点过程"
  - "马尔可夫决策过程"
  - "视频问答"
  - "免训练"
---

# mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs

**会议**: ICCV 2025  
**arXiv**: [2501.02885](https://arxiv.org/abs/2501.02885)  
**代码**: [github.com/sunh-23/MDP3](https://github.com/sunh-23/MDP3)  
**领域**: 强化学习  
**关键词**: 帧选择, 行列式点过程, 马尔可夫决策过程, 视频问答, 免训练

## 一句话总结

提出 mDP3，一种免训练、模型无关的视频帧选择方法，通过条件高斯核在 RKHS 中估计帧相似度，结合行列式点过程（DPP）捕获查询相关性和列表级多样性，再通过马尔可夫决策过程（MDP）建模时序性，在多个长视频 benchmark 上以仅 8 帧输入显著超越均匀采样和现有帧选择方法。

## 研究背景与动机

### 领域现状

Video-LLM 在视频理解上取得了显著进展，但处理多帧导致的超长视觉 token 序列带来了严重挑战：LLM 上下文长度限制、无关帧引发的 "lost-in-the-middle" 问题、边缘设备资源约束、以及 API 调用的 token 成本。

### 现有痛点

**均匀采样**：无视当前查询，可能遗漏关键帧并包含无关帧

**Query-frame 检索**：仅考虑单帧与查询的逐点相关性，忽略帧间的列表级关系

**缺乏多样性**：视频中大量相似帧，冗余选择增加计算负担却几乎无信息增益

**忽视时序性**：视频帧的时序结构对理解因果关系至关重要

### 核心洞察

有效的帧选择应同时满足三个原则：**查询相关性**（所选帧与问题相关）、**列表级多样性**（所选帧之间信息互补）、**时序性**（保留帧的顺序结构）。此外，帧选择遵循边际效用递减规律——增加帧数的边际收益递减，甚至可能产生负面效果（如 64 帧不如 32 帧）。

## 方法详解

### 整体框架

mDP3 是一个即插即用的推理时帧选择模块，复用预训练 VLM 的视觉和文本编码器提取帧和查询的嵌入。流程：(1) 在 RKHS 中计算条件帧相似度矩阵；(2) 通过 DPP 的 MAP 推理选择兼顾相关性和多样性的帧子集；(3) 将视频分段，通过 MDP 动态规划跨段分配选择配额。

### 关键设计

#### 1. **条件多高斯核（CMGK）——高维条件相似度估计**

- **功能**：在 RKHS 中估计以查询为条件的帧间高维相似度，替代低维余弦相似度
- **核心思路**：定义条件核 $\tilde{k}(f_i, f_j | q) = g(f_i, q) \cdot k(f_i, f_j) \cdot g(f_j, q)$，其中 $k, g \in \mathcal{K}$ 是多核的凸组合。构建条件相似度矩阵：

$$\tilde{\mathbf{L}} = \text{diag}(\mathbf{r}) \cdot \mathbf{L} \cdot \text{diag}(\mathbf{r})$$

$\mathbf{L}_{ij} = k(f_i, f_j)$ 为帧间相似度，$\mathbf{r}_i = g(f_i, q)$ 为帧-查询相关度
- **设计动机**：直接用余弦相似度估计帧关系过于粗糙，RKHS 中的核方法能捕获更丰富的高维关系。条件核进一步将查询信息注入相似度估计，降低无关帧的权重

#### 2. **行列式点过程（DPP）——联合建模相关性与多样性**

- **功能**：在条件相似度矩阵上应用 DPP，同时捕获查询相关性和列表级多样性
- **核心思路**：子集 $S$ 的概率正比于子矩阵行列式 $\det(\tilde{\mathbf{L}}_S)$，即 RKHS 中子集的几何体积——体积越大说明帧越分散。引入权重因子 $\lambda$ 平衡相关性与多样性：

$$\log(\det(\tilde{\mathbf{L}}_S)) = \frac{1}{\lambda}\sum_{i \in S}\log(\mathbf{r}_i^2) + \log(\det(\mathbf{L}_S))$$

通过贪心 MAP 推理求解，保证 $(1-1/e)$ 近似比，Cholesky 分解使复杂度降至 $\mathcal{O}(nk^2)$
- **设计动机**：DPP 天然建模"排斥"效应（源自费米子的泡利不相容原理），非常适合表达列表级多样性。与 NP-hard 的子集选择问题相比，DPP + 贪心算法提供了高效的近似解

#### 3. **马尔可夫决策 DPP（MDP）——时序性建模与容量分配**

- **功能**：将视频分为多个连续段，通过 MDP 建模跨段的帧选择过程，并用动态规划算法最优分配总选择配额 $k$
- **核心思路**：将视频分为 $T = \lceil n/m \rceil$ 段，第 $t$ 段的 DPP 以前一段的选择为条件：

$$\mathcal{P}(S_t | S_{t-1}) = \frac{\det(\tilde{\mathbf{L}}_{S_{t-1} \cup S_t})}{\det(\tilde{\mathbf{L}}_t + \mathbf{I}_t)}$$

定义状态三元组 $(t, k_t, C_t)$，通过动态规划更新价值函数 $Q^*(t, C_t) = \max_{k_t} Q(t, k_t, C_t)$，时间复杂度为伪多项式 $\mathcal{O}(nk^3)$

- **设计动机**：标准 DPP 将帧视为可交换的集合，忽略了时序结构。与查询相关的帧通常集中在特定段落，需要非均匀地分配选择配额。MDP + 动态规划避免了整数规划的约束松弛，能找到最优分配

### 损失函数 / 训练策略

**免训练方法**：mDP3 完全不需要训练，复用预训练 VLM（如 CLIP）的编码器。在推理时从 128 帧候选中选择 8 帧输入 Video-LLM，额外计算开销可忽略。

## 实验关键数据

### 主实验

| 模型 | 帧数 | Video-MME (wo/w subs) | MLVU | LVBval |
|------|------|----------------------|------|--------|
| LLaVA-OneVision | 8 | 53.6 / 53.9 | 59.3 | 54.2 |
| **+mDP3** | **8** | **59.6 / 59.1** (+6.0) | **69.8** (+10.5) | **59.0** (+4.8) |
| MiniCPM-V2.6 | 8 | 52.6 / 53.1 | 55.4 | 51.2 |
| **+mDP3** | **8** | **58.0 / 61.8** (+5.4) | **66.6** (+11.2) | **57.1** (+5.9) |
| Ovis2 | 8 | 58.9 / 62.1 | 60.9 | 56.9 |
| **+mDP3** | **8** | **63.9 / 66.1** (+5.0) | **73.9** (+13.0) | **62.7** (+5.8) |
| VILA-V1.5 | 8 | 47.5 / 50.0 | 46.3 | 47.1 |
| **+mDP3** | **8** | **53.3 / 56.6** (+5.8) | **58.6** (+12.3) | **50.8** (+3.7) |

关键对比：LLaVA-OneVision + mDP3 (8帧) 超过了官方报告的 LLaVA-OneVision* (精调帧数, 58.2 Video-MME)。

### 消融实验

| 方法 | 查询相关 | 多样性 | 时序性 | 高维相似度 | Acc. (Video-MME) |
|------|---------|--------|--------|-----------|-----------------|
| Uniform | ✗ | ✗ | ✗ | ✗ | 47.5 |
| KNN (CLIP) | ✓ | ✗ | ✗ | ✗ | 48.5 |
| DPP | ✗ | ✓ | ✗ | ✗ | 49.6 |
| CDPP | ✓ | ✓ | ✗ | ✗ | 51.5 |
| CDPP + HDS | ✓ | ✓ | ✗ | ✓ | 52.1 |
| **mDP3** | **✓** | **✓** | **✓** | **✓** | **53.3** |

不同视频时长的提升（LLaVA-OneVision, LVBval）：

| 视频时长 | Baseline | +mDP3 | 提升 |
|---------|---------|-------|------|
| 8-15s | 68.3 | 70.4 | +2.1 |
| 15s-1m | 66.9 | 73.3 | +6.4 |
| 3-10m | 52.4 | 57.8 | +5.4 |
| 15-60m | 46.8 | 51.8 | +5.0 |

### 关键发现

- **mDP3 对所有测试的 Video-LLM 都有效**，验证了模型无关性
- **8 帧 mDP3 超过 32 帧均匀采样的最佳结果**：Selection > Quantity
- **仅 2 帧选择即可达到最优性能的 90.8%**：极致的效率
- **长视频受益更大**：短视频（8-15s）提升有限，长视频（3-60m）提升显著
- **查询相关性、多样性、时序性三者缺一不可**：消融实验清晰显示每个组件的贡献
- **超过训练帧数（64帧）时性能反而下降**：当前 Video-LLM 难以泛化到超出训练帧数的设置

## 亮点与洞察

1. **理论优雅**：将帧选择问题形式化为 NP-hard 子模最大化问题，DPP 提供 $(1-1/e)$ 近似保证，MDP 提供伪多项式时间最优分配
2. **免训练即插即用**：不需要额外数据或训练，直接复用预训练 VLM，适用于任意 Video-LLM
3. **边际效用递减的深刻洞察**：以实验和子模性理论双重验证了"帧不在多而在精"
4. **物理学启发**：DPP 源自量子物理的费米子排斥原理，用"排斥力"建模帧多样性在直觉上非常自然

## 局限与展望

1. **依赖预训练 VLM 的嵌入质量**：如果 VLM 对特定域的帧表示能力差，帧选择效果会打折扣
2. **从 128 帧候选中选择**：初始均匀采样 128 帧可能已经错过关键帧
3. **视频分段长度 $m$ 为固定超参**：自适应分段可能更优
4. **仅在 VidQA 任务上验证**：视频描述、时序定位等任务的有效性待验证
5. **$\mathcal{O}(nk^3)$ 复杂度**：虽然实践中快，但理论复杂度仍有优化空间

## 相关工作与启发

- Frame-Voyager 是针对 VILA 定制训练的帧选择器，mDP3 以免训练方式超越了它
- DPP 在视频摘要中的应用（Gong et al., Zheng & Lu）为本文的分段 DPP 提供了基础
- Video-MME 等长视频 benchmark 揭示了帧选择对长视频理解的关键性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — DPP + MDP 的组合在帧选择中首次使用，理论贡献扎实
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个 Video-LLM、3 个 benchmark、详尽的消融和分析
- **写作质量**: ⭐⭐⭐⭐ — 数学推导严谨清晰，但符号较多需要一定背景知识
- **价值**: ⭐⭐⭐⭐⭐ — 免训练、模型无关、显著提升，具有极高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](../../ICML2026/reinforcement_learning/single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)
- [\[ICCV 2025\] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)
- [\[ICML 2025\] VinePPO: Refining Credit Assignment in RL Training of LLMs](../../ICML2025/reinforcement_learning/vineppo_refining_credit_assignment_in_rl_training_of_llms.md)
- [\[CVPR 2025\] ThinkStream: Thinking in Streaming Video](../../CVPR2025/reinforcement_learning/thinking_in_streaming_video.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](../../AAAI2026/reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)

</div>

<!-- RELATED:END -->
