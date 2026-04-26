---
title: >-
  [论文解读] Understanding and Improving Hyperbolic Deep Reinforcement Learning
description: >-
  [ICLR 2026][hyperbolic geometry] 通过闭式梯度分析揭示双曲深度 RL 中 Poincaré Ball 保角因子爆炸和大范数嵌入导致 PPO 信赖域失效的根源，提出 Hyper++（RMSNorm + 可学习缩放 + HL-Gauss + Hyperboloid）四组件方案，在 ProcGen 16 环境和 Atari-5 上全面超越先前基线。
tags:
  - ICLR 2026
  - hyperbolic geometry
  - PPO
  - gradient analysis
  - RMSNorm
  - categorical value loss
---

# Understanding and Improving Hyperbolic Deep Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2512.14202](https://arxiv.org/abs/2512.14202)  
**代码**: [GitHub](https://github.com/Probabilistic-and-Interactive-ML/hyper-rl)  
**领域**: 强化学习  
**关键词**: hyperbolic geometry, PPO, gradient analysis, RMSNorm, categorical value loss

## 一句话总结
通过闭式梯度分析揭示双曲深度 RL 中 Poincaré Ball 保角因子爆炸和大范数嵌入导致 PPO 信赖域失效的根源，提出 Hyper++（RMSNorm + 可学习缩放 + HL-Gauss + Hyperboloid）四组件方案，在 ProcGen 16 环境和 Atari-5 上全面超越先前基线。

## 研究背景与动机

**领域现状**：序贯决策过程天然产生层级数据——每个状态分支为指数多个后续状态，形成树状结构。欧几里得空间体积仅多项式增长（$V_d(r) \propto r^d$），与层级结构的指数增长存在根本几何失配。双曲空间体积指数增长，已在分类、度量学习、图像-文本对齐中取得成功。

**现有痛点**：双曲深度 RL 面临严重优化困难——Cetin et al. (2023) 首次将双曲几何引入 RL，但训练不稳定，依赖 SpectralNorm + S-RYM（$\mathbf{x}_E \mapsto \mathbf{x}_E / \sqrt{d}$）缓解，限制整个编码器表达能力。

**核心矛盾**：双曲空间的几何优势（低失真层级嵌入）vs 训练稳定性（保角因子爆炸 + 梯度病态）之间的根本冲突。

**本文目标** 1）形式化分析双曲 PPO 的梯度病态来源；2）设计既保证稳定性又保持表达力的双曲 RL agent。

**切入角度**：从 Poincaré Ball 和 Hyperboloid 两种双曲模型的闭式梯度推导出发，定位不稳定来源，再逐一设计对应组件。

**核心 idea**：大范数嵌入是双曲 PPO 崩溃的根源，通过 RMSNorm 约束范数 + Hyperboloid 避免保角因子 + 分类损失对齐几何即可根治。

## 方法详解

### 整体框架
Hyper++ 采用 hybrid Euclidean-hyperbolic encoder 架构（Impala-ResNet），共享欧几里得编码器 + 双曲 actor/critic 头。核心改进集中在编码器最后层到双曲层之间的接口：RMSNorm → TanH → 可学习缩放 → Hyperboloid 指数映射 → HL-Gauss 分类值损失。

### 关键设计

1. **RMSNorm 正则化（替代 SpectralNorm）**
    - 功能：约束编码器输出的欧几里得嵌入范数，防止双曲指数映射梯度爆炸
    - 核心思路：仅在最后线性层预激活输出应用 RMSNorm + $1/\sqrt{d}$ 缩放。Proposition 4.2 保证对 1-Lipschitz 激活函数（ReLU/TanH）有 $\|\hat{\mathbf{x}}\|_2 < 1$，进而保角因子 $\lambda < 2\cosh^2(\sqrt{c})$
    - 设计动机：Lemma 4.1 证明 SpectralNorm 需应用于所有编码器层才能有效约束范数，但全层应用严重限制 Lipschitz 常数和表达力。RMSNorm 仅需最后一层，保留其余层自由度

2. **可学习特征缩放（Learned Euclidean Feature Scaling）**
    - 功能：扩大 RMSNorm 约束后双曲空间的可用容积
    - 核心思路：学习标量 $\xi_\theta$，缩放嵌入为 $\hat{\mathbf{x}}_E^{\text{rescale}} = \rho_{\max} \cdot \sigma(\xi_\theta) \cdot \hat{\mathbf{x}}_E$，其中 $\rho_{\max} = \operatorname{atanh}(\alpha)/\sqrt{c}$，$\alpha=0.95$
    - 设计动机：RMSNorm 将 Poincaré Ball 可用半径限制为 0.76（$c=1$），可用体积 $\propto r^d$，$d=32$ 时 $(0.95/0.76)^{32} \approx 1200\times$ 体积增益

3. **Hyperboloid 模型 + HL-Gauss 分类值损失**
    - 功能：从几何和损失两个层面消除不稳定源
    - 核心思路：Hyperboloid MLR 无保角因子（$v^{\text{HB}}$ 公式中不含 $(1-c\|\mathbf{x}\|^2)^{-1}$ 项），梯度更稳定。HL-Gauss 将值函数学习转化为 51 个离散 bin 的分类问题，与双曲 MLR 的超平面距离输出几何对齐
    - 设计动机：Poincaré Ball MLR 梯度 $\propto (1-c\|\mathbf{x}_H\|^2)^{-2}$ 在边界附近爆炸；MSE 回归与双曲 MLR 几何不匹配——分类损失更自然

### 损失函数 / 训练策略

PPO clipped surrogate objective 不变。Critic 使用 HL-Gauss 损失（51 bins, $[-10, 10]$），TanH 替代 ReLU 作为最后激活。Corollary 4.3 通过 Poincaré Ball-Hyperboloid 等距，将 RMSNorm + 缩放的范数约束传递到 Hyperboloid 时间分量 $x_0^{\max}$ 的有界性。

## 实验关键数据

### 主实验 — ProcGen（PPO, 16 环境, 25M steps, 6 seeds）

| 指标 | Hyper++ | Hyper+S-RYM | Euclidean | Hyper(无正则) |
|------|---------|-------------|-----------|-------------|
| Test IQM ↑ | **0.41** | 0.27 | 0.26 | 0.19 |
| Train IQM ↑ | **0.55** | 0.46 | 0.45 | 0.37 |
| 前向时间 | 14.7ms | 19.3ms | 14ms | — |
| NameThisGame 全程 | 35h25m | 58h21m | 17h52m | — |

### 消融实验（ProcGen Test IQM, 6 seeds + bootstrap CI）

| 配置 | Test IQM | 说明 |
|------|---------|------|
| Hyper++ (完整) | **0.40** | 基线 |
| −RMSNorm | 0.00 | 完全学习失败，范数爆炸 |
| −Scaling | 0.33 | 可用体积不足 |
| +MSE (替代 HL-Gauss) | 0.33 | 几何不匹配 |
| +C51 | 0.27 | 分布式损失不如 HL-Gauss |
| +Poincaré (替代 Hyperboloid) | 0.34 | 保角因子轻微影响 |
| +SN Full / +SN Penult. | 0.00 / 0.00 | SpectralNorm 均失败 |
| Euclidean + 全套正则化 | 0.35 | 接近但不如双曲 |

### 关键发现
- Hyper++ 在 PPG（更强基线）上同样有效：PPG IQM 0.52 vs Hyper+S-RYM 0.34 vs Euclidean 0.47
- Atari-5（DDQN, 10M steps）：Hyper++ 在所有 5 个游戏的 IQM/median/mean/optimality gap 全面最优
- 欧几里得 + HL-Gauss 反而不如欧几里得 + MSE → 分类损失需要双曲几何配合才有效
- 每个消融都不如完整 Hyper++，证明组件间存在协同效应

## 亮点与洞察
- **梯度分析驱动设计**：不是经验性尝试，而是先推导 $\partial v / \partial \mathbf{x}_H$ 和 $\partial \mathbf{x}_H / \partial \mathbf{x}_E$ 的闭式表达式，定位 $(1-c\|\mathbf{x}\|^2)^{-2}$ 为罪魁祸首，再针对性设计 RMSNorm
- **一个理论结果解决多个问题**：Proposition 4.2 同时保证了嵌入范数有界、保角因子有界、梯度稳定——一石三鸟
- **组件分工清晰**：categorical loss 稳定 $\partial L / \partial v$，Hyperboloid 稳定 $\partial v / \partial \mathbf{x}_H$，RMSNorm+scaling 稳定 $\partial \mathbf{x}_H / \partial \mathbf{x}_E$——方程 (3) 的链式法则每一项都有对应组件
- **性能+效率双赢**：回报提升 52% 的同时前向时间减少 ~30%（去掉了 SpectralNorm 的 power iteration）

## 局限与展望
- 聚焦优化视角，未分析双曲表示实际学到了什么层级结构
- 未研究哪些环境最适合双曲表示（哪些 MDP 的状态空间更"树形"）
- 几何选择（曲率 $c$、维度 $d$）与不同 RL 算法设计的交互未探索
- ProcGen Phoenix 上 Hyper++ 出现可塑性丧失（plasticity loss），与基线表现相似

## 相关工作与启发
- **vs Cetin et al. (2023) Hyper+S-RYM**：消除 SpectralNorm 的稳定性-表达力权衡，test IQM 从 0.27 → 0.41
- **vs Farebrother et al. (2024) HL-Gauss**：他们发现 HL-Gauss 在欧几里得 RL 中不一致地有效，本文揭示其与双曲几何的特殊协同
- **vs Mishne et al. (2023) 双曲数值稳定性**：他们分析了一般双曲网络的数值稳定性，本文扩展到 RL 的 actor-critic 训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度分析驱动的双曲 RL 修复方案，理论与实践紧密结合
- 实验充分度: ⭐⭐⭐⭐⭐ ProcGen 16环境 + Atari-5 + PPO/PPG/DDQN 三算法 + 大量消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，方程-组件-实验三线呼应
- 价值: ⭐⭐⭐⭐ 为双曲深度 RL 提供了首个可靠的实践方案

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](cuda-l1_improving_cuda_optimization_via_contrastive_reinforcement_learning.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](deep_spi_safe_policy_improvement_via_world_models.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)

<!-- RELATED:END -->
