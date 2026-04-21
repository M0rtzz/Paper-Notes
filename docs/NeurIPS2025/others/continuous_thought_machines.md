---
title: >-
  [论文解读] Continuous Thought Machines
description: >-
  [NeurIPS 2025][神经动力学] 提出 Continuous Thought Machine (CTM)，通过私有参数化的 Neuron-Level Models (NLMs) 产生神经元级时间动力学，并以神经同步矩阵作为核心潜在表征，在迷宫求解、ImageNet 分类、奇偶校验等任务上展现复杂推理、自适应计算和可解释注意力行为。
tags:
  - NeurIPS 2025
  - 神经动力学
  - 神经同步
  - 自适应计算
  - 循环架构
  - 生物启发
---

# Continuous Thought Machines

**会议**: NeurIPS 2025  
**arXiv**: [2505.05522](https://arxiv.org/abs/2505.05522)  
**代码**: [github.com/SakanaAI/ContinuousThoughtMachines](https://github.com/SakanaAI/ContinuousThoughtMachines)  
**领域**: others  
**关键词**: 神经动力学, 神经同步, 自适应计算, 循环架构, 生物启发

## 一句话总结
提出 Continuous Thought Machine (CTM)，通过私有参数化的 Neuron-Level Models (NLMs) 产生神经元级时间动力学，并以神经同步矩阵作为核心潜在表征，在迷宫求解、ImageNet 分类、奇偶校验等任务上展现复杂推理、自适应计算和可解释注意力行为。

## 研究背景与动机

**领域现状**: 现代神经网络为追求大规模训练效率，刻意抽象掉了生物神经元的精确时序和交互复杂性。标准 MLP/Transformer 中，神经元仅执行简单激活函数，不具备独立的时间处理能力。

**现有痛点**: (a) 深度学习缺乏人类认知的灵活性、效率和常识——这些可能与时间处理相关；(b) 自适应计算需要额外的停机模块（PonderNet/ACT）；(c) 循环网络虽有序列处理能力，但其状态表征是静态快照而非动态时间模式。

**核心矛盾**: 神经时序在生物大脑中至关重要（时间编码、同步、振荡），但在 AI 中被完全忽视。重新引入时间维度可能解锁缺失的认知能力。

**本文切入角度**: 在保持梯度可微的框架内，设计具有神经元级时间处理和同步表征的新架构，在生物真实性和计算可行性之间取得平衡。

## 方法详解

### CTM 架构总览
CTM 在一个与数据维度解耦的**内部时间维度** $t \in \{1, \ldots, T\}$ 上运行。每个 "internal tick" 执行以下流程：

### 1. 突触模型 (Synapse Model)
共享的突触网络 $f_{\theta_{\rm syn}}$（U-NET 式 MLP）连接 $D$ 维潜空间：

$$\mathbf{a}^t = f_{\theta_{\rm syn}}([\mathbf{z}^t; \mathbf{o}^t]) \in \mathbb{R}^D$$

输入为上一步后激活 $\mathbf{z}^t$ 和注意力输出 $\mathbf{o}^t$ 的拼接，输出为预激活。保留最近 $M$ 步的预激活历史 $\mathbf{A}^t \in \mathbb{R}^{D \times M}$。

### 2. Neuron-Level Models（核心创新之一）
每个神经元 $d$ 拥有**私有参数化**的 NLM $g_{\theta_d}$（单层 MLP），处理其 $M$ 维预激活历史：

$$z_d^{t+1} = g_{\theta_d}(\mathbf{A}_d^t)$$

关键差异：标准网络中所有神经元共享激活函数（ReLU/GELU），CTM 中每个神经元有独立权重处理自己的时间历史，能产生丰富多样的时序动力学。

### 3. 神经同步作为表征（核心创新之二）
收集后激活历史 $\mathbf{Z}^t = [\mathbf{z}^1, \ldots, \mathbf{z}^t] \in \mathbb{R}^{D \times t}$，计算同步矩阵：

$$\mathbf{S}^t = \mathbf{Z}^t \cdot (\mathbf{Z}^t)^\top \in \mathbb{R}^{D \times D}$$

$\mathbf{S}_{ij}^t$ 度量神经元 $i$ 和 $j$ 的活动模式随时间的相关性。为控制 $O(D^2)$ 规模，随机抽样 $D_{\rm out}$ 和 $D_{\rm action}$ 对神经元对，分别用于输出和注意力查询：

$$\mathbf{y}^t = \mathbf{W}_{\rm out} \cdot \mathbf{S}_{\rm out}^t, \quad \mathbf{q}^t = \mathbf{W}_{\rm in} \cdot \mathbf{S}_{\rm action}^t$$

### 4. 可学习时间衰减
为每对神经元 $(i,j)$ 引入可学习指数衰减率 $r_{ij} \geq 0$：

$$\mathbf{S}_{ij}^t = \frac{(\mathbf{Z}_i^t)^\top \cdot \text{diag}(\mathbf{R}_{ij}^t) \cdot \mathbf{Z}_j^t}{\sqrt{\sum_\tau [\mathbf{R}_{ij}^t]_\tau}}$$

其中 $[\mathbf{R}_{ij}^t]_\tau = \exp(-r_{ij}(t - \tau))$。高 $r_{ij}$ 偏向近期活动，$r_{ij}=0$ 无衰减。

### 5. 损失函数与自适应计算
每个 tick $t$ 产生输出 $\mathbf{y}^t$，计算损失 $\mathcal{L}^t$ 和确定度 $\mathcal{C}^t = 1 - H(\mathbf{y}^t)/\log C$。选择两个 tick 进行优化：

$$L = \frac{\mathcal{L}^{t_1} + \mathcal{L}^{t_2}}{2}, \quad t_1 = \arg\min_t \mathcal{L}^t, \quad t_2 = \arg\max_t \mathcal{C}^t$$

自适应计算**自然涌现**：简单样本在少量 tick 后即达到高确定度，无需额外停机模块。

## 实验关键数据

### 2D 迷宫（$39 \times 39$，100 步路径预测）

| 模型 | 路径准确率↑ | 泛化到 $99 \times 99$↑ |
|------|-----------|---------------------|
| Feed-Forward | 极低（接近不学习） | — |
| LSTM-1layer (75 ticks) | ~40-50% (短路径) | — |
| LSTM-3layer (50 ticks) | ~50-60% (短路径) | — |
| **CTM (75 ticks)** | **>90% (50步+)** | **通过重复应用有效泛化** |

### ImageNet-1K 分类（ResNet-152 骨干，50 internal ticks）

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | 72.47% |
| Top-5 准确率 | 89.89% |
| 80% 确定度提前停止 | 大部分样本 <10 ticks |
| 模型校准 | 优秀（不同 tick 的平均概率高度校准） |

### 奇偶校验（64 位序列，累积奇偶预测）

| 模型 | 最佳准确率↑ |
|------|-----------|
| LSTM (参数匹配) | 不稳定，低于 90% |
| CTM (50 ticks) | ~95% |
| CTM (75 ticks) | ~98-100%（部分种子完美） |
| CTM (100 ticks) | 100%（部分种子） |

### 消融实验要点
- 去除 NLM（标准激活函数）: 性能大幅下降
- 去除同步表征（用快照表征）: 性能大幅下降
- LSTM + 同步: 未能复现 CTM 的优势 → NLM 和同步缺一不可

### 关键发现
- CTM 在迷宫任务中学会**无位置编码下的空间推理**——通过注意力机制 "想象" 前方路径（episodic future thinking）
- ImageNet 上出现 "环顾" 行为：注意力随 tick 在图像上动态游走，类似人类视觉搜索
- 神经活动展现低频行波（traveling waves），类似生物皮层观测
- More ticks = more capable（奇偶校验），但也 = more compute

## 亮点与洞察
- **同步作为表征**而非后处理属性是核心范式突破：不同于 Reichert & Serre 将同步用于门控/分割，CTM 直接在训练中优化同步模式以编码任务相关信息
- **NLM 的设计哲学**：每个神经元有独立的 "性格"（不同权重处理历史），这是对生物神经元复杂性的恰当抽象
- **自适应计算的涌现性**：无需 PonderNet 的显式停机概率或 ACT 的累积门限，仅凭 min-loss + max-certainty 选择即自然实现
- **架构统一性**：同一 CTM 无修改地用于迷宫/分类/算法任务，展示了通用性

## 局限与展望
- 内部序列维度使训练时间线性延长（$T$ 倍）——当前实验规模有限
- NLM 的私有参数增加参数量（虽提供新的缩放维度）
- ImageNet 精度 72.47% 远低于 SOTA——作者明确说未做超参搜索，这是探索而非竞赛
- 理论分析缺失：同步表征的表达能力、收敛性均无理论支撑
- 语言建模、视频理解等更复杂任务的验证留为 future work

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 神经同步作为可学习表征 + NLM 是全新范式
- 实验充分度: ⭐⭐⭐⭐ 迷宫/分类/算法/RL 等多任务验证 + 消融
- 写作质量: ⭐⭐⭐⭐ 图示丰富，视频补充优秀，但论文较长
- 启发价值: ⭐⭐⭐⭐⭐ 重新思考神经元抽象层次，可能开辟新架构方向
- 综合: ⭐⭐⭐⭐ 大胆创新，方向意义 > 当前性能数字

## 相关工作与启发
- **vs PonderNet / ACT**: 需要显式停机模块学习何时停止。CTM 通过 min-loss + max-certainty 自然涌现自适应计算，无额外参数
- **vs LTCNs (Hasani et al. 2021)**: 液态时间常数网络用 ODE 控制动力学，但不使用同步作为表征。CTM 的同步矩阵提供更丰富的高阶表征空间
- **vs SNNs**: 离散脉冲和专用学习规则与标准深度学习训练不兼容。CTM 保持连续值和梯度可微
- **vs Reichert & Serre (2013)**: 同步仅为后处理门控，用于分割。CTM 将同步作为训练中优化的核心潜在表征
- **vs RIMs (Goyal et al. 2019)**: 模块化异步子网络无同步机制。CTM 的神经同步提供可学习的全局协调
- NLM 思路可迁移到 Transformer：为每个注意力头引入私有时间历史处理，可能增强 chain-of-thought 推理质量

<!-- RELATED:START -->

## 相关论文

- [Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](../../ICML2025/others/continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](../../ACL2025/others/neodiff_unified_text_diffusion.md)
- [Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](../../ICML2025/others/optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)
- [Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)

<!-- RELATED:END -->
