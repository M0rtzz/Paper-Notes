---
title: >-
  [论文解读] Human-assisted Robotic Policy Refinement via Action Preference Optimization
description: >-
  [NeurIPS 2025][LLM对齐][VLA模型] 提出 Action Preference Optimization (APO)，通过人机协作框架收集交互轨迹，利用基于前景理论的二元期望信号和自适应重加权方法对 VLA 模型进行偏好对齐优化，使其能从失败中学习并持续迭代改进。
tags:
  - NeurIPS 2025
  - LLM对齐
  - VLA模型
  - 偏好对齐
  - 人机协作
  - 机器人操作
  - 自适应重加权
---

# Human-assisted Robotic Policy Refinement via Action Preference Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2506.07127](https://arxiv.org/abs/2506.07127)  
**代码**: [GitHub](https://github.com/GeWu-Lab/Action-Preference-Optimization)  
**领域**: llm_alignment  
**关键词**: VLA模型, 偏好对齐, 人机协作, 机器人操作, 自适应重加权

## 一句话总结

提出 Action Preference Optimization (APO)，通过人机协作框架收集交互轨迹，利用基于前景理论的二元期望信号和自适应重加权方法对 VLA 模型进行偏好对齐优化，使其能从失败中学习并持续迭代改进。

## 研究背景与动机

VLA（Vision-Language-Action）模型作为机器人部署的基础模型，严重依赖离线专家示范数据，缺乏部署后持续改进的能力。现有方法面临两大困境：

**行为克隆（BC）方法**：无法充分利用失败轨迹中的宝贵信号，且在大规模 VLA 模型上存在专家数据和交互数据的分布偏移问题

**强化学习（RL）方法**：在训练大规模 VLA 模型时面临梯度不稳定和价值函数泛化困难的问题

此外，将 LLM 中成熟的偏好对齐方法迁移到 VLA 模型时存在两个核心挑战：
- **不可逆交互**：机器人操作的不可逆性使得在相同条件下获取配对正负样本极其困难
- **Token 概率不匹配**：自回归 VLA 模型将连续动作离散化为 token，导致 token 概率与真实动作损失之间存在根本性不匹配

## 方法详解

### 整体框架

APO 由两个核心组件构成：

1. **人机协作部署框架**：人类操作员实时监控机器人执行，在遇到困难场景时进行干预纠正，同时收集交互轨迹
2. **动作偏好优化过程**：利用自适应重加权的偏好对齐算法优化 VLA 模型

整体流程为迭代式的"部署-优化"循环：先用专家数据行为克隆得到初始策略 $\pi_\theta^0$，然后反复进行部署收集交互数据和偏好优化更新模型。

### 关键设计

**交互轨迹标注**：收集交互轨迹时，每个动作带有标注 $c_t$：
- $c_t = 1$：策略自动执行的动作
- $c_t = 2$：人类干预纠正的动作  
- $c_t = 0$：人类干预前 $K$ 步内的动作（标记为不期望的）

**基于前景理论的偏好对齐**：采用 Kahneman & Tversky 的前景理论，利用二元期望信号（而非配对偏好数据）进行偏好学习。首先估计奖励函数：

$$r_\theta(o, \hat{a}) = \log \frac{\pi_\theta(\hat{a}|o)}{\pi_{ref}(\hat{a}|o)}$$

然后定义效用函数：

$$v(o, \hat{a}) = \begin{cases} \lambda_D \sigma(r_\theta(o,\hat{a}) - z_0) & \text{if } \hat{a} \sim \hat{a}_{desirable} \\ \lambda_U \sigma(z_0 - r_\theta(o,\hat{a})) & \text{if } \hat{a} \sim \hat{a}_{undesirable} \end{cases}$$

其中 $z_0 = KL(\pi_\theta || \pi_{ref})$ 作为惩罚项防止模型过度偏离参考模型。

**自适应重加权方法**：为解决 token 概率与连续动作回归之间的不匹配，首先计算每个样本的 L1 连续动作损失，然后进行 batch 级归一化：

$$w_i = \frac{l_i}{\sum_{i=1}^{B} l_i}$$

利用归一化权重自适应调整 $\lambda_D$ 和 $\lambda_U$：
- 期望数据：$\lambda_D = 1 - e^{-\beta_D \cdot w}$（动作预测误差大的样本权重更高）
- 不期望数据：$\lambda_U = e^{-\beta_U \cdot w}$（接近失败动作的样本权重更高）

### 损失函数 / 训练策略

最终损失函数：

$$L(\pi_\theta, \pi_{ref}) = \mathbb{E}_{x,y \sim D^h}[-v(x,y)]$$

训练策略上采用**平衡采样**：每个 batch 包含 50% 专家动作、25% 人类干预动作、25% 失败动作，确保不同类型数据的均衡表示。

实现细节：使用 LoRA (rank=32) 微调 OpenVLA 模型，学习率 5e-5，batch size 8，4 张 A100 GPU；$K=10$ 用于标注不期望行为。

## 实验关键数据

### 主实验

**RoboMimic 仿真环境**（4 个长视界操作任务，50 轨迹/任务，50 次评估）：

| 方法 | Coffee | StackThree | ThreePiece | Square | 平均 |
|------|--------|------------|------------|--------|------|
| Base policy | 44% | 46% | 44% | 28% | 40.5% |
| Dagger | 42% | 50% | 36% | 28% | 39.0% |
| DPO | 52% | 46% | 28% | 22% | 37.0% |
| KTO | 48% | 52% | 46% | 32% | 43.5% |
| **APO** | **60%** | **54%** | **46%** | **32%** | **48.0%** |

**扰动场景泛化实验**（仅 20 条交互轨迹 + 20 条原任务专家数据）：

| 方法 | 位置扰动 | 背景扰动 | 纹理扰动 | 平均 |
|------|----------|----------|----------|------|
| Base policy | 12% | 42% | 10% | 21.3% |
| KTO | 20% | 46% | 6% | 24.0% |
| **APO** | **26%** | **46%** | **12%** | **28.0%** |

**真实世界实验**（插入方块任务）：

| 方法 | 原分布 | 位置扰动 | 背景扰动 | 纹理扰动 |
|------|--------|----------|----------|----------|
| Base policy | 65% | 25% | 10% | 25% |
| TPO | 75% | 40% | 20% | 45% |
| **APO** | **85%** | **55%** | **30%** | **55%** |

### 消融实验

**跨 VLA 架构泛化**（π0-FAST 模型）：

| 方法 | Coffee | StackThree | Insert Square |
|------|--------|------------|---------------|
| Base policy | 68% | 64% | 85% |
| Dagger | 64% | 66% | 85% |
| **APO** | **76%** | **74%** | **95%** |

**终身学习实验**：每 20 次交互后更新模型，APO 在专家数据带来的改进趋于饱和后仍能持续提升性能，同时人类干预率逐步降低。

### 关键发现

1. 行为克隆方法在 VLA 模型上无法超越基础模型，主要因为专家轨迹和交互轨迹之间的分布偏移
2. DPO 依赖合成负样本而非真实交互失败，效果最差
3. TPO 使用随机采样负样本导致训练不稳定
4. APO 在扰动场景下不仅适应新场景，还能提升原始任务表现（其他方法出现灾难性遗忘）
5. APO 学会了从失败中自主纠正（如重新抓取、迭代调整位置）

## 亮点与洞察

1. **巧妙的问题解耦**：将 LLM 偏好学习迁移到 VLA 的两个核心障碍（不可逆交互 + token 不匹配）分别用前景理论和自适应重加权来解决
2. **实用的人机协作范式**：人类干预既保证了部署可靠性，又自然地产生了偏好学习所需的数据
3. **无需配对偏好数据**：利用 KTO 的思想，只需二元信号（好/坏）而非配对比较，大大降低了数据标注要求
4. **从失败中学习的能力**：APO 不仅避免失败动作，还能在遭遇失败后主动自我纠正

## 局限与展望

1. 实验仅基于自回归 VLA 模型（OpenVLA、π0-FAST），未验证在回归式或扩散策略模型上的泛化性
2. 人类干预仍需实时监控，如何降低人类参与成本是开放问题
3. $K=10$ 的不期望行为标注窗口是固定的，可能需要根据任务动态调整
4. 扰动场景中的提升幅度有限（如纹理扰动从 10% 到 12%），鲁棒性仍有较大改进空间

## 相关工作与启发

- **KTO**：本文的偏好对齐核心思想来源于 KTO 的前景理论优化，但增加了自适应重加权
- **DPO/RLHF**：经典偏好对齐方法在机器人领域面临配对数据获取困难的问题
- **Grape (TPO)**：轨迹级偏好对齐方法，需要相同条件下的配对轨迹，在真实场景中不可行
- 本文从 NLP 偏好对齐出发，为具身智能中的 VLA 模型提供了可迁移的持续学习范式

## 评分

- 新颖性: ⭐⭐⭐⭐ (将偏好对齐迁移到 VLA 并解决关键技术障碍)
- 实验充分度: ⭐⭐⭐⭐ (仿真+真实世界，多任务多扰动，终身学习，跨模型验证)
- 写作质量: ⭐⭐⭐⭐ (问题定义清晰，动机充分)
- 价值: ⭐⭐⭐⭐ (为 VLA 部署后持续改进提供了实用方案)

<!-- RELATED:START -->

## 相关论文

- [SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [Capturing Individual Human Preferences with Reward Features](capturing_individual_human_preferences_with_reward_features.md)

<!-- RELATED:END -->
