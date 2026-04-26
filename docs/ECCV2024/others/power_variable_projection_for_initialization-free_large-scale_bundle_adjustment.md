---
title: >-
  [论文解读] Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment
description: >-
  [ECCV 2024][光束法平差] 提出 Power Variable Projection (PoVar) 算法，将幂级数展开方法扩展到变量投影（VarPro）框架，并进一步推广到黎曼流形优化，首次实现了无初始化大规模光束法平差（BA）的高效求解。
tags:
  - ECCV 2024
  - 光束法平差
  - 无初始化优化
  - 变量投影
  - 幂级数展开
  - 黎曼流形优化
---

# Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment

**会议**: ECCV 2024  
**arXiv**: [2405.05079](https://arxiv.org/abs/2405.05079)  
**代码**: [GitHub](https://github.com/tum-vision/povar)  
**领域**: 其他  
**关键词**: 光束法平差, 无初始化优化, 变量投影, 幂级数展开, 黎曼流形优化

## 一句话总结

提出 Power Variable Projection (PoVar) 算法，将幂级数展开方法扩展到变量投影（VarPro）框架，并进一步推广到黎曼流形优化，首次实现了无初始化大规模光束法平差（BA）的高效求解。

## 研究背景与动机

**领域现状**: 光束法平差（Bundle Adjustment, BA）是 SfM 和三维重建的核心组件。传统 BA 依赖 Levenberg-Marquardt (LM) 算法配合 Schur complement trick，需要良好的初始化。近年来 PoBA 通过幂级数展开逆 Schur complement 显著提升了传统 BA 的速度和精度。

**现有痛点**: 无初始化 BA——仅从图像观测出发恢复相机位姿和路标点——是一个几乎未被探索的领域。现有方法（pOSE 等）使用分层 BA 策略，但都依赖直接分解（Cholesky/QR），只能处理几十个相机的小规模问题。VarPro 算法有宽收敛域的优势，但其可扩展性一直是盲点。

**核心矛盾**: VarPro 适合无初始化 BA（收敛域宽），但缺乏高效求解器；直接分解不可扩展，预条件共轭梯度（PCG）的收敛性对 VarPro 的无阻尼结构不友好。

**本文目标**: 让无初始化 BA 能够扩展到数千个相机的大规模场景。

**切入角度**: 将幂级数求逆方法分别扩展到 VarPro（第一阶段）和黎曼流形优化（第二阶段），为分层 BA 的两个阶段都提供高效求解器。

**核心 idea**: VarPro 的 Schur complement 虽然与传统 BA 结构相似但因无阻尼 landmark 而收敛行为不同，可以证明幂级数展开仍然成立，并进一步推广到齐次坐标下的黎曼流形优化。

## 方法详解

### 整体框架

采用分层 BA（stratified BA）的三阶段策略：
1. **第一阶段**: 用 pOSE 目标函数（介于仿射和投影之间）求解，VarPro 消去路标变量，PoVar 高效求解 reduced camera system
2. **第二阶段**: 用投影标准目标函数在齐次坐标下精化，RiPoBA 在黎曼流形上使用幂级数展开求解
3. **第三阶段**: Metric upgrade，将投影相机矩阵约束为 $SE(3)$

### 关键设计

1. **Power Variable Projection (PoVar)**: VarPro 的幂级数求解器

    - VarPro 核心思想：将最小二乘问题 $\min_{x_p, \tilde{x}_l} \|G(x_p)\tilde{x}_l - z(x_p)\|_2^2$ 中的路标变量 $\tilde{x}_l$ 用解析解 $\tilde{x}_l^*(x_p) = G(x_p)^\dagger z(x_p)$ 替换，仅对相机参数 $x_p$ 优化。
    - VarPro 的 Schur complement 为 $S^V = U_\lambda - WV_0^{-1}W^\top$，关键区别在于路标 Hessian $V_0$ 无阻尼（仅保证半正定）。
    - **核心定理**: 证明 $U_\lambda^{-1}WV_0^\dagger W^\top$ 的特征值满足 $0 \leq \mu < 1$，因此幂级数展开收敛：
    $x(m) = -\sum_{i=0}^{m}(U_\lambda^{-1}WV_0^{-1}W^\top)^i U_\lambda^{-1}(b_p - WV_0^{-1}b_l)$
    - **设计动机**: 虽然 PoVar 与 PoBA 的算法结构相似，但由于 VarPro 只阻尼相机参数，收敛行为完全不同——PoVar 收敛更平滑，尤其在高精度要求下表现更优。

2. **Riemannian Power BA (RiPoBA)**: 将幂级数扩展到黎曼流形优化

    - 第二阶段在齐次坐标下优化（相机 $\text{vec}(\tilde{x}_p^i) \in S^{12}$，路标 $\tilde{x}_l^j \in S^4$），存在局部尺度自由度，需要黎曼流形优化。
    - 将 Jacobian 和阻尼参数投影到切空间：$\tilde{J}_p = J_p \tilde{x}_p^\perp$, $\tilde{J}_l = J_l \tilde{x}_l^\perp$
    - 统一记号后 normal equation 结构与标准 BA 形式一致，证明黎曼 Schur complement 的幂级数展开同样成立：
    $\tilde{S}^{-1} \approx \sum_{i=0}^{m}(\tilde{U}_{\tilde{\lambda}}^{-1}\tilde{W}\tilde{V}_{\tilde{\lambda}}^{-1}\tilde{W}^\top)^i \tilde{U}_{\tilde{\lambda}}^{-1}$
    - **设计动机**: 直接分解在大规模问题上不可行，PCG 在黎曼框架下收敛不稳定。利用矩阵的块对角结构实现内存高效的切空间投影和存储。

3. **高效存储策略**: 利用 BA 问题的稀疏结构，将 landmark 组织为 dense block，对每个 block 中的 pose Jacobian 和 landmark Jacobian 分别应用切空间投影矩阵，保持内存效率（如 pose Jacobian 从 $\mathbb{R}^{2 \times 12}$ 投影为 $\mathbb{R}^{2 \times 11}$，landmark 从 $\mathbb{R}^{2 \times 4}$ 投影为 $\mathbb{R}^{2 \times 3}$）。

### 损失函数 / 训练策略

- 每阶段最大迭代次数 50，相对函数容差 $10^{-6}$ 提前终止
- 阻尼因子 $\lambda$ 初始 $10^{-4}$，按 LM 策略更新
- 幂级数最大阶数 20，阈值 0.01
- 迭代方法最大内迭代 500 次
- pOSE 参数 $\eta = 0.1$
- 实现基于 C++，在 PoBA 代码基础上扩展

## 实验关键数据

### 主实验

在 BAL 数据集的 97 个真实世界 BA 问题上评估（16 到 13682 个相机），使用 performance profile 同时评估速度和精度：

| 实验阶段 | 指标 (τ=0.001) | PoVar | PoBA | 直接分解 | 迭代法 |
|----------|------|------|------|------|------|
| 第一阶段 | 求解百分比 (高精度) | **最佳** | 第二 | 最差 | 第三 |
| 两阶段组合 | PoVar+RiPoBA | **最佳** | — | — | — |

PoVar+RiPoBA 在所有容差水平和所有相对时间下都优于竞争组合。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| PoVar vs PoBA (第一阶段) | PoVar 在 τ=0.001 时显著更优 | VarPro 的无阻尼结构导致不同收敛行为 |
| RiPoBA vs RiPCG (第二阶段) | RiPoBA 在所有容差下更优 | 幂级数优于预条件共轭梯度 |
| PoVar 收敛曲线 | 更平滑，无卡顿 | 对比 PoBA 在早期迭代中可能停滞 |
| Venice-1672 (大规模) | PoVar+RiPoBA 收敛到最低误差 | 首次在千级相机上无初始化求解 |

### 关键发现

- PoVar 在高精度要求下（τ=0.001）的优势尤为突出，这是 VarPro 宽收敛域的直接体现
- PoVar 的收敛曲线比 PoBA 平滑得多——PoBA 在早期迭代常出现"卡顿"
- 相同第一阶段求解器下，RiPoBA 始终优于 RiPCG，验证了黎曼幂级数展开的有效性
- 在 Venice-1672 等大规模问题上，PoVar+RiPoBA 是唯一能收敛到高精度的组合

## 亮点与洞察

- **首次解决大规模无初始化 BA 的可扩展性问题**: 此前工作仅处理几十个相机，本文扩展到数千个
- **理论严密**: 对 VarPro Schur complement 的半正定 $V_0$ 和黎曼框架都给出了幂级数收敛的数学证明
- **PoVar 与 PoBA 的微妙区别**: 虽然算法结构相似，但 VarPro 仅阻尼相机变量导致完全不同的收敛性质，这一洞察具有理论深度
- **黎曼流形优化与幂级数的桥接**: 统一了切空间投影和幂级数框架，为流形上的大规模优化提供了新思路

## 局限与展望

- 假设离群点轨迹已被预过滤，未处理鲁棒 BA
- Metric upgrade 阶段使用已知近似焦距（来自 BAL 数据集），真实场景需要估计内参
- 实验仅在 BAL 数据集上验证，可扩展到 MegaDepth 等更新数据集
- 未与 COLMAP 等完整 SfM 流水线集成测试端到端效果

## 相关工作与启发

- **VarPro 系列**: 从 Golub-Pereyra 原始算法到 Hong et al. 在视觉中的应用，本文解决了其可扩展性瓶颈
- **pOSE**: 提出的介于仿射和投影之间的目标函数提供了宽收敛域，是本文第一阶段的基础
- **PoBA**: 在传统 BA 中用幂级数替代 PCG/直接分解取得突破，本文将其推广到 VarPro 和黎曼框架
- 启发：将幂级数展开推广到新类型的 Schur complement（如不同阻尼结构）是一个有潜力的通用方法论

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 理论贡献扎实，首次解决无初始化大规模 BA 的可扩展性
- 实验充分度: ⭐⭐⭐⭐ BAL 全部 97 个问题 + performance profile，但缺少端到端 SfM 评测
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但数学符号较多，阅读门槛高
- 价值: ⭐⭐⭐⭐ 为初始化-free SfM 开辟可扩展解法，有重要工程价值

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Event-based Mosaicing Bundle Adjustment](event-based_mosaicing_bundle_adjustment.md)
- [\[ECCV 2024\] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)
- [\[ECCV 2024\] Free-Viewpoint Video of Outdoor Sports Using a Flying Camera](free-viewpoint_video_of_outdoor_sports_using_a_flying_camera.md)
- [\[ECCV 2024\] Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)
- [\[ACL 2025\] Code-Switching and Syntax: A Large-Scale Experiment](../../ACL2025/others/code-switching_and_syntax_a_large-scale_experiment.md)

<!-- RELATED:END -->
