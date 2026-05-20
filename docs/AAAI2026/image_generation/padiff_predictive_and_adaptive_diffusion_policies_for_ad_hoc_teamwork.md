---
title: >-
  [论文解读] PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork
description: >-
  [AAAI 2026][图像生成][Ad Hoc Teamwork] 首次将扩散模型应用于 Ad Hoc Teamwork 问题，提出 PADiff 框架，通过 Adaptive Feature Modulation Net（AFM-Net）实现对动态队友的实时适应…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "Ad Hoc Teamwork"
  - "扩散模型"
  - "多模态策略"
  - "预测引导"
  - "自适应特征调制"
---

# PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork

**会议**: AAAI 2026  
**arXiv**: [2511.07260](https://arxiv.org/abs/2511.07260)  
**代码**: 无  
**领域**: 图像生成 / 多智能体协作  
**关键词**: Ad Hoc Teamwork, 扩散模型, 多模态策略, 预测引导, 自适应特征调制

## 一句话总结

首次将扩散模型应用于 Ad Hoc Teamwork 问题，提出 PADiff 框架，通过 Adaptive Feature Modulation Net（AFM-Net）实现对动态队友的实时适应，通过 Predictive Guidance Block（PGB）将队友意图预测信息注入去噪过程，在多模态合作场景中比现有方法平均提升 35.25%。

## 研究背景与动机

### 领域现状

**Ad Hoc Teamwork (AHT)** 是多智能体系统中的核心挑战：要求智能体在**没有预定义协调协议**的情况下，与**从未见过的队友**进行有效协作。这在灾害救援（与陌生救援人员协作）、自动驾驶（与未知驾驶风格的其他车辆交互）等现实场景中至关重要。

现有 AHT 方法主要基于强化学习（如 LIAM、GPL-SPI、ODITS），通过学习队友表示来适应不同队友行为。

### 现有痛点

**策略坍缩为单一模式**：RL 方法优化固定的期望回报，导致策略坍缩为单一主导行为。在 AHT 中，同一状态下可能存在多种有效的合作模式（如传球给队友 A、传球给队友 B、自己射门），但 RL 只能学到一种。即使使用最大熵 RL（如 SAC），其无方向性的行为分散也无法结构化建模多模态分布。

**扩散模型缺乏预测能力**：扩散模型天然适合建模多模态分布，但标准扩散模型是为分布重建设计的，**缺乏对队友意图的预测能力**，而 AHT 要求能预判队友行为并做出前瞻性决策。

**传统去噪网络适应性不足**：MLP 和 UNet 等传统去噪架构缺乏对动态变化队友行为的适应机制。Transformer（如 DiT）虽然有注意力机制但计算开销太大，不适合快节奏 AHT 场景。

### 核心矛盾

AHT 需要同时具备三个能力：(1) 多模态策略表示——捕捉多种合作模式；(2) 预测能力——预判队友意图做前瞻决策；(3) 实时适应——对队友行为变化的快速响应。现有方法至多满足其中一个。

### 核心 Idea

将扩散模型作为策略基础以天然捕捉多模态合作模式，然后针对 AHT 的预测和适应需求设计两个专用模块：(1) **AFM-Net**——基于 FiLM 机制的自适应特征调制网络，用队友上下文动态调制去噪过程中的特征；(2) **PGB**——预测引导块，在训练时预测队友的协作回报和协作目标，通过梯度传播让去噪过程内化队友意图预测能力，推理时无需额外计算。

## 方法详解

### 整体框架

PADiff 的训练流程：
1. 从多样化的队友策略池中采样队友
2. Ego agent 与队友在环境中交互收集轨迹
3. 将轨迹存入离线数据集
4. 用扩散策略优化 ego agent

框架包含三个核心组件：扩散策略表示、Teammates Adaptation Block（含 AFM-Net）、Predictive Guidance Block（PGB）。

### 关键设计

#### 1. **离散扩散策略表示（Discrete Diffusion Policy Representation）**

**功能**：将 ego agent 的策略建模为条件扩散过程，在离散动作空间上进行去噪。

**核心思路**：对于离散动作空间，采用 D3PM（Discrete Denoising Diffusion Probabilistic Models）框架。前向过程使用分类分布：

$$q(\mathbf{a}_t^k | \mathbf{a}_t^{k-1}) = \text{Cat}(\mathbf{a}_t^k; p = \mathbf{a}_t^{k-1} Q_k)$$

其中 $Q_k$ 是转移矩阵，采用 Uniform 调度。训练通过最小化变分下界：

$$\mathcal{L}_{Diff} = \mathbb{E}[D_{KL}[q(\mathbf{a}^{k-1}|\mathbf{a}^k, \mathbf{a}^0) \| p_\theta(\mathbf{a}^{k-1}|\mathbf{a}^k, s, k)]]$$

**设计动机**：扩散模型可以表示任何可归一化的分布，天然适合建模 AHT 中的多模态合作策略。通过迭代去噪，策略可以同时"准备"多种合作方案。

#### 2. **State Encoder 与 Teammates Adaptation Block**

**State Encoder**：使用历史窗口 $s_{t-m:t}$ 捕捉队友合作上下文，编码为多元高斯分布的潜变量：

$$(μ_{z_t}, σ_{z_t}) = f_ξ(s_{t-m:t}), \quad z_t \sim \mathcal{N}(μ_{z_t}, σ_{z_t})$$

**AFM-Net（Adaptive Feature Modulation Net）**：基于 FiLM（Feature-wise Linear Modulation）机制的去噪网络，具有三个核心特征：

**(a) 条件特征调制**：从队友上下文 $z_t$ 和扩散步 $k$ 生成 scale/shift 参数 $\gamma, \beta$，动态调制中间特征：

$$\gamma_1, \beta_1, \gamma_2, \beta_2 = MLP(\mathbf{z}_t + k)$$
$$\text{AFM}(\mathbf{x}, z_t, k) = \gamma_2 \cdot (MLP(\gamma_1 \cdot LN(\mathbf{x}) + \beta_1)) + \beta_2 + \mathbf{x}$$

**(b) 残差连接**：保证训练稳定性和鲁棒表示。

**(c) Dropout 正则化**：提升对未见过队友的泛化能力。

**设计动机**：UNet 缺乏队友感知能力，DiT 的注意力机制计算开销大。FiLM 机制以极低的计算成本（仅一个 MLP 生成 scale/shift）实现了条件调制，同时 Layer Norm + Residual + Dropout 的组合确保了鲁棒性。框架使用两层级联 AFM-Net。

#### 3. **Predictive Guidance Block (PGB)**

**功能**：在训练过程中让去噪过程学会预测队友意图，推理时无需额外模块即可生成队友感知的动作。

**核心思路**：PGB 包含两个预测任务：

**(a) Collaborative Return (CoReturn)** — 预测期望累积团队回报：
$$L_{\text{CoReturn}} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_{t=1}^{T}\|R_\phi(h_t^k, s_t) - R_t\|^2\right]$$

**(b) Collaborative Goal (CoGoal)** — 预测未来团队状态（子目标）：
$$L_{\text{CoGoal}} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[-\frac{1}{N}\sum_{t}\sum_{i}(G_{t,i}\log\hat{G}_{t,i} + (1-G_{t,i})\log(1-\hat{G}_{t,i}))\right]$$

两个预测任务使用 AFM-Net 的中间特征 $h_t^k$ 和状态 $s_t$ 作为输入，通过梯度传播优化中间表示：

$$\nabla_{h_t^k}L_{\text{total}} = \nabla_{h_t^k}L_{\text{Diffusion}} + \alpha\nabla_{h_t^k}L_{\text{CoReturn}} + \beta\nabla_{h_t^k}L_{\text{CoGoal}}$$

**推理时 PGB 完全被移除**——因为 AFM-Net 已通过训练内化了预测能力，其中间特征自然具备队友感知性。

**设计动机**：(1) CoReturn 提供"总体方向"——哪些动作序列能带来更高团队回报；(2) CoGoal 提供"具体意图"——队友接下来要去哪、要做什么。层级式预测（先 Return 后 Goal）模拟了团队协作中"评估→规划"的自然过程。推理时零开销是核心优势。

### 损失函数 / 训练策略

总训练损失：
$$L_{\text{total}} = L_{\text{Diffusion}} + \alpha L_{\text{CoReturn}} + \beta L_{\text{CoGoal}}$$

队友池构建：使用 CSP 框架的 Soft-Value Diversity (SVD) 方法，训练 4 个独立多智能体种群，3 个用于训练交互，1 个（含 12 个策略检查点）用于测试。训练 20 epochs，每 2 epochs 评估一次。

## 实验关键数据

### 主实验

三个环境上的评估回报对比（从论文 Figure 5 和附录 Table 1 摘取关键数据）：

| 环境 | 方法 | 平均回报 | 说明 |
|------|------|---------|------|
| Predator-Prey | LIAM | ~18 | 基于队友建模的 RL |
| Predator-Prey | Diffusion-BC | ~16 | 扩散行为克隆 |
| Predator-Prey | Diffusion-QL | ~20 | 扩散 Q-learning |
| Predator-Prey | **PADiff** | **~26** | +30% vs Diffusion-QL |
| LBF | ODITS | ~0.45 | 在线自适应 AHT |
| LBF | **PADiff** | **~0.65** | +44% vs ODITS |
| Overcooked | Diffusion-QL | ~80 | 扩散 Q-learning |
| Overcooked | **PADiff** | **~130** | +62% vs Diffusion-QL |

PADiff 在所有环境中显著超越所有基线，**平均性能提升 35.25%**。

### 消融实验

去噪网络架构消融（替换 AFM-Net）：

| 架构 | PP 回报 | LBF 回报 | Overcooked 回报 | 说明 |
|------|---------|---------|----------------|------|
| MLP | ~20 | ~0.50 | ~90 | 缺乏条件调制 |
| UNet | ~22 | ~0.55 | ~95 | 图像导向设计不适用 |
| **AFM-Net** | **~26** | **~0.65** | **~130** | FiLM+残差+Dropout |

PGB 模块消融：

| 配置 | PP 回报 | LBF 回报 | Overcooked 回报 | 说明 |
|------|---------|---------|----------------|------|
| **Full PADiff** | **~26** | **~0.65** | **~130** | 完整版本 |
| w/o CoReturn | ~23 | ~0.58 | ~110 | 移除协作回报预测 |
| w/o CoGoal | ~22 | ~0.55 | ~105 | 移除协作目标预测 |
| w/o PGB | ~20 | ~0.50 | ~95 | 移除整个预测引导 |

**两个预测任务均不可缺少**，CoGoal 的移除造成略大的性能下降，说明意图预测比回报预测更关键。

### 关键发现

1. **多模态策略可视化验证**：将相同状态输入策略多次，ego agent 产生了不同的合作路径（如从不同方向包围猎物），证明扩散策略确实学到了多模态分布。
2. **AFM-Net 优于 UNet**：UNet 为图像设计的架构（多尺度特征聚合）不适合 AHT 的低维决策场景，FiLM 机制更高效。
3. **PGB 推理零开销**：训练时通过梯度传播让 AFM-Net 内化预测能力，推理时移除 PGB 不影响性能。
4. **在困难环境（Overcooked）中优势最大**：Overcooked 的复杂空间布局和高协作需求让 PADiff 的优势更加突出（+62%）。

## 亮点与洞察

1. **首次将扩散模型应用于 AHT**：证明了扩散模型在多智能体协作中的价值——多模态策略表示是 AHT 的天然需求。
2. **PGB 的"训练时预测、推理时免费"设计**：通过辅助预测任务的梯度信号优化中间表示，是一种富有启发性的设计模式。类似 DreamerV3 的世界模型预测，但专门面向协作场景。
3. **FiLM 机制的妙用**：用极简的 scale/shift 调制代替复杂的注意力机制，在保持高效的同时实现条件化去噪，适合实时决策。
4. **35.25% 的平均提升**：在所有环境中一致性地大幅超越基线，说明方法的通用性。

## 局限与展望

1. **仅在格子世界和厨房环境验证**：PP、LBF、Overcooked 都是相对简单的离散环境，未在连续控制或高维视觉输入场景验证。
2. **离散动作空间限制**：使用 D3PM 建模离散动作，未处理连续动作空间（如机器人关节控制）。
3. **队友池构建依赖 SVD**：需要预训练多样化的队友策略池，增加了准备成本。
4. **扩散推理步数**：文中未明确报告推理时的去噪步数，扩散模型的多步采样可能在实时 AHT 场景中仍有延时问题。
5. **缺少与 decision transformer 类方法的对比**：TAGET 是最新的 DT-based AHT 方法，但实验中的对比结果不够突出。

## 相关工作与启发

- **Diffusion Policy (Wang et al. 2022)**：提供了扩散模型作为策略的基础框架，PADiff 在此基础上针对 AHT 引入预测和适应能力。
- **FiLM (Perez et al. 2018)**：AFM-Net 的条件调制机制直接借鉴自 FiLM，在视觉推理中也有成功应用。
- **DreamerV3**：世界模型通过环境预测增强决策的思路类似 PGB，但 PGB 专注于队友协作预测。
- **对多智能体学习的启发**：扩散模型的多模态能力 + 预测辅助任务的范式可推广到更广泛的合作/竞争多智能体场景。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将扩散模型应用于 AHT，三个核心组件设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 三个环境、多基线对比、架构消融和模块消融完整
- **写作质量**: ⭐⭐⭐⭐ — 动机论证清晰，方法描述详细
- **价值**: ⭐⭐⭐⭐ — 为 AHT 提供了新的技术路线，35% 的提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](../../ICLR2026/image_generation/improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](../../ICML2026/image_generation/recovering_hidden_reward_in_diffusion-based_policies.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/image_generation/offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](../../NeurIPS2025/image_generation/failure_prediction_at_runtime_for_generative_robot_policies.md)

</div>

<!-- RELATED:END -->
