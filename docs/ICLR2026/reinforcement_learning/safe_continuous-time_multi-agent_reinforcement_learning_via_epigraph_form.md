---
title: >-
  [论文解读] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form
description: >-
  [ICLR 2026][连续时间RL] 提出首个显式处理状态约束的连续时间多智能体RL框架，通过Epigraph形式将不连续的约束值函数转化为连续表示，结合改进的PINN actor-critic方法实现安全、稳定的连续时间多智能体控制。
tags:
  - ICLR 2026
  - 连续时间RL
  - 多智能体
  - 安全约束
  - HJB方程
  - Epigraph重构
---

# Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form

**会议**: ICLR 2026  
**arXiv**: [2602.17078](https://arxiv.org/abs/2602.17078)  
**代码**: [GitHub链接](https://github.com/xuefeng-wang/EPI)  
**领域**: 强化学习  
**关键词**: 连续时间RL, 多智能体, 安全约束, HJB方程, Epigraph重构

## 一句话总结

提出首个显式处理状态约束的连续时间多智能体RL框架，通过Epigraph形式将不连续的约束值函数转化为连续表示，结合改进的PINN actor-critic方法实现安全、稳定的连续时间多智能体控制。

## 研究背景与动机

多智能体强化学习（MARL）的大多数算法基于离散时间MDP和Bellman方程，假设固定的决策时间间隔。然而，许多实际场景（自动驾驶、金融交易、机器人协作）本质上是连续时间控制问题，离散时间离散化在高频或不均匀时间间隔下会导致性能退化和训练不稳定。

现有的连续时间MARL方法基于Hamilton-Jacobi-Bellman（HJB）方程，使用物理信息神经网络（PINN）逼近值函数。但它们**几乎不考虑安全约束**（如碰撞惩罚），原因是状态约束引入值函数的不连续性，使得HJB-PINN难以准确逼近。

核心矛盾：安全MARL需要处理约束，但约束导致值函数不连续，而PINN只能逼近光滑函数。本文通过**Epigraph重构**巧妙地将不连续值化为连续表示来解决这一矛盾。

## 方法详解

### 整体框架

EPI框架包含：（1）将安全CT-MARL形式化为连续时间约束MDP（CT-CMDP）；（2）Epigraph重构引入辅助状态 $z$ 统一目标值和约束；（3）改进的inner-outer优化的actor-critic架构，包含PINN-based critic和分散式actor。

### 关键设计

1. **Epigraph重构（核心理论贡献）**:

    - 功能：引入辅助状态 $z(t)$ 将约束优化转化为无约束连续值函数
    - 核心思路：定义辅助值函数 $V(x,z) = \min_{u} \max\{\max_\tau c(x(\tau)), \int_t^\infty \gamma^{\tau-t} l(x(\tau),u(\tau))d\tau - z\}$
    - Lemma 3.1证明 $v(x) = \min\{z \in \mathbb{R} | V(x,z) \leq 0\}$，使得约束值 $v$ 的获取转化为 $V$ 的零水平集搜索
    - 设计动机：$V(x,z)$ 是连续的（Theorem 3.3），而原始约束值函数不连续，PINN可以逼近连续函数

2. **改进的Outer优化（$z^*$ 计算）**:

    - 功能：在训练中直接计算最优 $z^*$ 而非随机采样
    - 核心思路：$z^* = \min\{z | \max\{V_\phi^{\text{cons}}(x), V_\psi^{\text{ret}}(x) - z\} \leq 0\}$
    - 设计动机：先前方法（EPPO等）随机采样 $z$ 引入非平稳噪声，破坏策略更新稳定性；且执行时需要昂贵的根查找。EPI将return和constraint网络设计为仅依赖 $x$（不依赖 $z$），训练时直接用 $z^*$，执行时无需根查找

3. **PINN-based Critic（三重损失）**:

    - 功能：用三种互补损失训练值函数
    - 核心思路：
        - **残差损失**：惩罚HJB PDE的违反 $\mathcal{L}_{\text{Residual}} = (\max\{c(x)-\tilde{V}, \min_u \mathcal{H}\})^2$
        - **目标损失**：基于轨迹的数值目标 $\mathcal{L}_{\text{Target}} = (V_{\text{tgt}} - \tilde{V})^2$，无限时域下无边界条件时作为锚点
        - **值梯度迭代（VGI）**：约束值梯度一致性，确保 $\nabla_x V$ 的准确性
    - 设计动机：残差损失在无界问题中不足以单独工作；值梯度对策略更新至关重要

4. **分散式Actor学习**:

    - 功能：基于epigraph优势函数更新分散策略
    - 核心思路：$A(x_t,z_t^*,u_t) = \max\{c(x_t)-V, \nabla_x V \cdot f(x,u) - \partial_z V \cdot l(x,u) + \ln\gamma \cdot V\}$
    - 通过学习的动力学网络 $f_\xi$ 和代价网络 $l_\phi$ 替代未知真实函数
    - 设计动机：集中训练分散执行（CTDE），每个agent仅需本地观测

### 损失函数 / 训练策略

Critic总损失：$\mathcal{L}_{\text{Critic}} = \lambda_{\text{res}}\mathcal{L}_{\text{Residual}} + \lambda_{\text{tgt}}\mathcal{L}_{\text{Target}} + \lambda_{\text{vgi}}\mathcal{L}_{\text{VGI}}$。Actor损失：$\mathcal{L}_{\text{actor}} = \mathbb{E}[A_\theta(x,z^*,u)]$。权重通过网格搜索确定。

## 实验关键数据

### 主实验（连续时间Safe MPE + MuJoCo）
| 方法 | 方向 | 约束与代价优势 |
|------|------|---------------|
| MACPO | 信赖域约束 | 过于保守 |
| MAPPO-Lag | 拉格朗日松弛 | 平衡不稳定 |
| SAC-Lag | 离策略+拉格朗日 | 约束满足差 |
| EPPO | 随机采样z | 卡在次优 |
| CBF | 控制屏障函数 | 保守但合理 |
| **EPI (ours)** | **$z^*$直接优化** | **代价和约束均接近最优** |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整EPI | 最优 | 三重损失+$z^*$优化 |
| 去除Target损失 | 显著退化 | 无界问题中值函数漂移 |
| 去除VGI损失 | 严重退化 | 值梯度不准确→策略更新有害 |
| 去除Residual损失 | 轻微影响 | PDE结构在有VGI时不太关键 |
| 过度加权任一损失(×20) | 退化 | 平衡权重最优 |

### 关键发现
- EPPO因随机采样 $z$ 导致收敛到次优解
- Target和VGI损失对无限时域问题至关重要，残差损失相对次要
- EPI在Formation、Line、Target等MPE场景中一致性达到最低代价和约束违反
- 在MuJoCo（HalfCheetah、Ant）中也优于基线

## 亮点与洞察

- **首次将安全约束引入CT-MARL**：填补了连续时间安全MARL的空白
- **Epigraph重构的巧妙性**：将不连续值→连续值的转换使得PINN方法可以工作
- **$z^*$ 直接优化**：消除了先前方法的噪声源和执行时开销
- **理论保证**（Theorem 3.3）：证明了epigraph HJB PDE的粘性解的存在唯一性

## 局限与展望

- 需要学习动力学和代价网络（$f_\xi, l_\phi$），增加了模型复杂度
- 值函数损失权重 $(\lambda_{\text{res}}, \lambda_{\text{tgt}}, \lambda_{\text{vgi}})$ 通过网格搜索确定，可考虑自适应方案
- 当前实验环境规模有限（2-6个agent），大规模agent的可扩展性待验证
- PINN方法在高维状态空间下可能面临训练困难

## 相关工作与启发

- Wang et al. (2025)首次系统研究CT-MARL但忽略安全约束，EPI直接补充了这一缺失
- Zhang et al. (2025b)的EPPO引入epigraph但随机采样 $z$，EPI的改进方案更稳定
- So and Fan (2023)的epigraph形式用于单agent安全控制，本文扩展到多agent RL
- 启示：PDE-based RL方法的关键不是残差损失，而是值梯度的准确性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将安全约束+连续时间+多智能体统一处理，方法新颖
- 实验充分度: ⭐⭐⭐⭐ MPE和MuJoCo双基准，详细消融，但agent规模有限
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，框架图清晰
- 价值: ⭐⭐⭐⭐ 开拓了安全CT-MARL新方向，理论和方法贡献并重
