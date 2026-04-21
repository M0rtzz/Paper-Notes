---
title: >-
  [论文解读] Intention-Conditioned Flow Occupancy Models
description: >-
  [图像生成] 提出 InFOM，利用流匹配（flow matching）构建意图条件化的占据模型（occupancy model），通过变分推断推理数据中的潜在意图，实现无标注数据上的 RL 预训练，在 36 个状态任务和 4 个视觉任务上取得 1.8× 中位回报提升和 36% 成功率提升。
tags:
  - 图像生成
---

# Intention-Conditioned Flow Occupancy Models

## 一句话总结

提出 InFOM，利用流匹配（flow matching）构建意图条件化的占据模型（occupancy model），通过变分推断推理数据中的潜在意图，实现无标注数据上的 RL 预训练，在 36 个状态任务和 4 个视觉任务上取得 1.8× 中位回报提升和 36% 成功率提升。

## 研究背景与动机

大规模预训练 - 微调范式在 NLP 和 CV 中取得巨大成功，但在强化学习（RL）中仍然是一个开放问题。RL 的核心难点在于：

**时间推理**：智能体需要推理当前动作的长期影响，而世界模型（world model）受累积误差限制，长距离推理能力有限

**意图推理**：大规模离线数据集通常由多个用户执行不同任务收集而来，这些隐含的"意图"未被显式标注

**现有方法的局限**：行为克隆（BC）只模仿动作、不捕获意图；判别式占据模型训练困难；后继特征（successor features）方法通常忽略用户意图

本文提出 InFOM（Intention-conditioned Flow Occupancy Models），同时学习一个概率模型来捕获时间信息和意图信息，使预训练模型能够感知不同用户的行为目的，从而在下游任务微调时实现更高效的策略学习。

## 方法详解

InFOM 包含预训练和微调两个阶段：

### 预训练阶段

**1. 变分意图推断（Variational Intention Inference）**

- 给定无标注数据集 $D=\{(s,a,s',a')\}$，通过变分推断推理潜在意图 $z$
- 意图编码器 $p_e(z|s',a')$ 从下一步转移 $(s',a')$ 推断意图（基于一致性假设：连续转移共享同一意图）
- 最大化 ELBO：$\mathbb{E}[\log q_d(s_f|s,a,z)] - \lambda D_{KL}(p_e(z|s',a') \| p(z))$
- 先验 $p(z) = \mathcal{N}(0,I)$，$\lambda$ 控制 KL 正则化强度

**2. SARSA 流占据模型（SARSA Flow Occupancy Models）**

- 使用流匹配（flow matching）学习生成式占据模型 $q_d(s_f|s,a,z)$，预测折扣状态占据度量
- 引入时序差分（TD）思想到流匹配损失中，实现动态规划和组合泛化
- 损失函数分为两部分：当前流损失 $(1-\gamma)\mathcal{L}_{\text{current}}$ 和未来流损失 $\gamma \mathcal{L}_{\text{future}}$
- SARSA 变体比 Q-learning 变体更简单稳定，在大数据集上表现更好

### 微调阶段

**3. 生成式价值估计（Generative Value Estimation）**

- 固定预训练的占据模型，采样 $N=16$ 个未来状态 $s_f^{(i)} \sim q_d(s_f|s,a,z)$
- 蒙特卡洛估计意图条件化 Q 函数：$Q_z(s,a) = \frac{1}{(1-\gamma)N}\sum_i r(s_f^{(i)})$
- 意图 $z$ 从先验 $p(z)$ 采样而非后验

**4. 隐式广义策略改进（Implicit GPI）**

- 将朴素 GPI 中对有限意图集合取 max 替换为上分位数期望损失（upper expectile loss）
- 将多个 $Q_z$ 蒸馏为单一标量 Q 函数：$\mathcal{L}(Q) = \mathbb{E}[L_2^\mu(Q_z(s,a) - Q(s,a))]$
- 避免通过 ODE 求解器反向传播梯度，训练更稳定
- 附加行为克隆正则化抑制 OOD 动作

## 实验

### 实验一：ExORL 和 OGBench 基准测试

在 36 个状态任务和 4 个视觉任务上与 8 种基线方法对比：

| 任务域 | InFOM | 最强基线 | 提升 |
|--------|-------|----------|------|
| walker (4 tasks avg) | **380.9** | 327.6 (MBPO+ReBRAC) | ~16% |
| jaco (4 tasks avg) | **727.4** | 67.7 (IQL) | ~20× |
| cube single (5 tasks) | **92.5** | 77.8 (MBPO+ReBRAC) | ~19% |
| visual tasks (4 tasks) | — | — | +31% over best |

- 在 9 个域中的 7 个上匹配或超越所有基线
- jaco 域改进最为显著（约 20×），归因于高维状态空间和稀疏奖励
- image-based 任务比最强基线高 31%
- 整体中位回报提升 1.8×，成功率提升 36%

### 实验二：隐式 GPI 消融实验

| 策略提取方式 | quadruped jump 回报 | scene task 1 成功率 |
|-------------|---------------------|---------------------|
| InFOM (implicit GPI) | **最高** | **最高** |
| InFOM + GPI (朴素 max) | 低 44% | 低，方差 8× |
| FOM + one-step PI | 显著更低 | 显著更低 |

- 隐式 GPI 比朴素 GPI 性能高 44%、方差小 8×
- 去除意图编码器（FOM + one-step PI）导致性能大幅下降，验证意图推理的重要性

## 亮点

- **统一框架**：首次将意图推断和流匹配占据模型结合，在一个框架中同时捕获时间和意图信息
- **隐式 GPI**：用 expectile loss 替代显式 max 操作，避免了 ODE 反向传播不稳定问题和有限意图集合的局限
- **强实验表现**：36+4 个任务上全面优于 8 种基线，jaco 域有 20× 改进
- **意图可视化**：t-SNE 可视化表明 InFOM 能发现与真实意图对齐的聚类结构，而 FB 和 HILP 的表征混杂

## 局限性

1. 从连续状态-动作对推断意图的简化可能无法准确捕获完整轨迹级别的原始意图
2. MC Q 估计带来方差（部分任务跨种子标准差较大）
3. 需要同时预训练编码器和流模型，计算开销高于纯 BC 方法
4. 一致性假设（连续转移共享意图）在实际复杂场景中可能不成立

## 相关工作

- **离线无监督 RL**：FB（Touati & Ollivier, 2021）、HILP（Park et al., 2024）学习技能/表征但通常不同时建模占据度量
- **占据模型/后继表征**：Dayan (1993)、Janner et al. (2020)、TD flows（Farebrother et al., 2025）使用流匹配建模占据度量但不建模意图
- **生成式 RL**：Decision Transformer、Diffuser 等用生成模型建模轨迹/策略，但通常不显式预测长期状态分布
- **表征学习**：对比学习、MAE 等学习通用表征，但不保证有利于策略适应
- **InFOM 的创新点**：相比最接近的 TD flows，引入变分潜变量建模意图 + 隐式 GPI 替代有限集上的显式 GPI

## 评分

⭐⭐⭐⭐ (4/5)

- 理论动机清晰，将变分推断与流匹配占据模型有机结合
- 实验覆盖广、基线充分，36+4 任务 × 8 基线 × 8 种子
- 隐式 GPI 是优雅的工程/理论贡献
- 扣分点：意图一致性假设较强，MC 估计方差问题未完全解决

<!-- RELATED:START -->

## 相关论文

- [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)

<!-- RELATED:END -->
