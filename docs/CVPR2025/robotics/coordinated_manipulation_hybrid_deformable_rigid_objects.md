---
title: >-
  [论文解读] Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments
description: >-
  [CVPR 2025][机器人][变形线性物体] 本文提出基于应变参数化 Cosserat 杆模型（GVS）的准静态轨迹优化框架，用于双臂机器人在约束环境中协调操纵混合变形-刚性线性物体（hDLO），利用解析梯度实现比有限差分快 33 倍的求解速度，并在真实双臂平台上验证了 ~3cm 的变形误差。
tags:
  - "CVPR 2025"
  - "机器人"
  - "变形线性物体"
  - "Cosserat杆模型"
  - "轨迹优化"
  - "双臂协调"
  - "约束环境"
---

# Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments

**会议**: CVPR 2025  
**arXiv**: [2603.12940](https://arxiv.org/abs/2603.12940)  
**代码**: 无  
**领域**: 机器人操作 / 变形物体  
**关键词**: 变形线性物体, Cosserat杆模型, 轨迹优化, 双臂协调, 约束环境

## 一句话总结

本文提出基于应变参数化 Cosserat 杆模型（GVS）的准静态轨迹优化框架，用于双臂机器人在约束环境中协调操纵混合变形-刚性线性物体（hDLO），利用解析梯度实现比有限差分快 33 倍的求解速度，并在真实双臂平台上验证了 ~3cm 的变形误差。

## 研究背景与动机

**领域现状**：变形线性物体（DLO，如电缆、绳索）的操纵是机器人领域的活跃研究方向。已有大量工作关注纯 DLO 的操纵规划，使用采样方法（RRT/PRM）或数据驱动方法（RL/IL）。

**现有痛点**：(1) 真实应用中物体往往是变形和刚性部件的混合体（如带接头的线缆、有关节的柔性机构），纯 DLO 模型不适用；(2) 采样方法不保证轨迹最优性，且随机采样可能违反物理约束；(3) 优化方法因高维配置空间和非线性力学而计算昂贵，且强依赖初始猜测。

**核心矛盾**：hDLO 的配置空间理论上是无穷维的（DLO 每点都有连续的位姿变化），需要一个既能降维又能统一处理变形和刚体的建模框架。

**本文目标**：为约束环境中 hDLO 的双臂协调操纵提供基于优化的规划方案，利用 GVS 模型的结构优势和解析梯度实现高效求解。

**切入角度**：GVS 模型用应变参数天然统一了 DLO 的连续变形和刚体关节，提供有限维配置空间和解析 Jacobian，使梯度优化成为可能。

**核心 idea**：用 GVS 应变模型将 hDLO 的无穷维配置空间降为有限维，通过逆运动静力学求解提供暖启动，再进行满足环境约束的轨迹优化。

## 方法详解

### 整体框架

输入：hDLO 模型（DLO + 刚体链接的拓扑和力学参数）+ 双臂操纵器 + 约束环境（圆形通道）+ 目标末端位姿。分两阶段求解：(1) 逆运动静力学（IKS）：找到满足目标的静态平衡配置；(2) 轨迹优化：以 IKS 解为暖启动，求解满足环境约束的时间序列。

### 关键设计

1. **GVS 应变参数化建模**:

    - 功能：将 hDLO 的无穷维配置空间降为有限维，统一表示变形和刚体运动
    - 核心思路：Cosserat 杆模型将 DLO 的位姿描述为沿弧长的连续映射 $g(X) \in SE(3)$。GVS 用有限基函数参数化应变场 $\xi = \Phi(X)q + \xi^*$，其中 $q$ 为广义坐标。刚体关节是特例（$\Phi$ 不依赖 $X$）。闭链约束通过 $e_c(q) = \log(g_A^{-1}g_B) = 0$ 在 $\mathfrak{se}(3)$ 中表示，完整静态平衡条件为 $Kq - F - Bu - A^T\lambda = 0$
    - 设计动机：相比 FEM（最大坐标，高维）、离散弹性杆（不共享关节抽象）、数据驱动（不可解释），GVS 提供最小坐标表示且可直接复用刚体运动学

2. **解析梯度加速的优化求解**:

    - 功能：高效求解 IKS 和轨迹优化问题
    - 核心思路：利用最近提出的 GVS 静力学解析导数（对 $q, u, \lambda$ 的偏导数），避免有限差分。环境约束（圆形通道）通过 SE(3) 插值在任意弧长处评估 DLO 位置，解析导数通过链式法则传递
    - 设计动机：解析梯度在 IKS 问题上比有限差分快 33 倍，且轨迹优化在有限差分下根本不可行（需要暖启动+解析梯度才能收敛）

3. **IKS 暖启动的轨迹优化**:

    - 功能：在约束环境中生成可行的操纵轨迹
    - 核心思路：先忽略环境约束求解 IKS（找到目标配置），得到一条从初始到目标的直线插值轨迹作为初始猜测。然后以此暖启动进行完整的约束轨迹优化（包括圆形通道约束），将轨迹离散为 $N$ 个步骤并联立求解
    - 设计动机：直接求解带约束的轨迹优化非凸且容易发散，IKS 暖启动大幅缩小搜索空间

### 损失函数 / 训练策略

优化目标为最小化执行器运动（$\sum \|u_{k+1} - u_k\|^2$），约束包括：每步静态平衡、闭链约束、环境约束（DLO 不穿过通道壁）、关节限位。使用 IPOPT 求解器。

## 实验关键数据

### 主实验

| 指标 | 本文(优化) | BiRRT | 说明 |
|------|----------|-------|------|
| 3-link hDLO, 目标达成 | ✓ | ✓ | 两种方法都成功 |
| 平均变形误差(实验) | ~3cm (5%链接长) | N/A | 仅优化方法做了真实实验 |
| IKS 求解速度(解析) | 0.8s | N/A | 解析梯度 |
| IKS 求解速度(有限差分) | 26.4s | N/A | 慢33倍 |
| 轨迹优化可行性 | 解析梯度可行 | N/A | 有限差分不收敛 |

### 消融实验

| hDLO 配置 | IKS 解析(s) | IKS 有限差分(s) | 加速比 |
|-----------|-----------|---------------|-------|
| 2-link (1DLO+1rigid) | 0.3 | 8.2 | 27x |
| 3-link (2DLO+1rigid) | 0.8 | 26.4 | 33x |
| 5-link (3DLO+2rigid) | 2.1 | 65+ | 31x |

### 关键发现

- 解析梯度是使轨迹优化可行的关键——有限差分基线在轨迹优化阶段完全无法收敛
- IKS 暖启动对轨迹优化至关重要，无暖启动时优化常陷入局部最优或不满足约束
- 真实实验中平均 ~3cm（5% 链接长度）的变形误差，验证了仿真到实际的转移能力
- 优化方法找到的轨迹比 BiRRT 更平滑、更短，但计算时间更长

## 亮点与洞察

- **GVS 模型的统一性**：一个框架同时处理连续变形和离散关节，这种"最小坐标"表示为优化方法铺平了道路。可推广到更多机器人结构
- **解析梯度的必要性**：不仅提速 33 倍，更重要的是使轨迹优化从"不可行"变为"可行"。在高维非线性系统中，解析导数的价值远超简单的加速
- **实际验证的完整性**：从仿真到双臂机器人实验的完整验证链，3cm 误差的实际精度令人信服

## 局限与展望

- 仅考虑准静态运动，不适用于需要动态操纵的场景
- 圆形通道是简化的约束模型，真实环境（如器官）形状更复杂
- 轨迹优化可能找到局部最优而非全局最优
- DLO 材料参数需要预先标定，对未知材料泛化能力有限
- 可结合力控制实现在线闭环调整

## 相关工作与启发

- **vs 采样方法 (RRT/PRM)**: 不保证最优性且随机采样可能违反物理；本文优化方法找到平滑最优轨迹
- **vs 数据驱动方法 (RL/IL)**: 样本效率低且不保证物理可行性；本文基于精确物理模型
- **vs FEM 方法**: 最大坐标导致高维系统；GVS 的最小坐标表示更适合优化

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 GVS 解析梯度用于 hDLO 约束轨迹优化
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实实验+与 BiRRT 对比
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，实验设计完整
- 价值: ⭐⭐⭐⭐ 对工业装配和微创手术中的 hDLO 操纵有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Automaton Constrained Q-Learning](../../NeurIPS2025/robotics/automaton_constrained_q-learning.md)
- [\[ICML 2025\] Action-Constrained Imitation Learning](../../ICML2025/robotics/action-constrained_imitation_learning.md)
- [\[CVPR 2025\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)
- [\[CVPR 2026\] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](../../CVPR2026/robotics/forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](../../NeurIPS2025/robotics/safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)

</div>

<!-- RELATED:END -->
