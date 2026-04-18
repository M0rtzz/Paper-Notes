---
title: >-
  [论文解读] Contextual Dynamic Pricing with Heterogeneous Buyers
description: >-
  [NeurIPS 2025][动态定价] 首次系统研究买家类型异质（$K_\star$ 种未知类型）的上下文动态定价问题，提出基于乐观后验采样 (OPS) 的算法实现 $\tilde{O}(K_\star\sqrt{dT})$ 遗憾界（对 $d$ 和 $T$ 最优），并在非上下文情形通过方差感知自适应离散化算法 ZoomV 实现 $\tilde{O}(\sqrt{K_\star T})$ 最优遗憾。
tags:
  - NeurIPS 2025
  - 动态定价
  - 异质买家
  - Contextual Bandits
  - 后验采样
  - 不一致系数
  - Lipschitz Bandits
---

# Contextual Dynamic Pricing with Heterogeneous Buyers

**会议**: NeurIPS 2025  
**arXiv**: [2512.09513](https://arxiv.org/abs/2512.09513)  
**领域**: others  
**关键词**: 动态定价, 异质买家, Contextual Bandits, 后验采样, 不一致系数, Lipschitz Bandits

## 一句话总结
首次系统研究买家类型异质（$K_\star$ 种未知类型）的上下文动态定价问题，提出基于乐观后验采样 (OPS) 的算法实现 $\tilde{O}(K_\star\sqrt{dT})$ 遗憾界（对 $d$ 和 $T$ 最优），并在非上下文情形通过方差感知自适应离散化算法 ZoomV 实现 $\tilde{O}(\sqrt{K_\star T})$ 最优遗憾。

## 研究背景与动机

**领域现状**: 在线上下文定价中，卖家根据 $d$ 维特征向量 $u_t$ 为商品设置价格 $p_t$，买家根据估值 $v_t = \langle \theta^*, u_t \rangle$ 决定是否购买。现有算法（多维二分搜索或回归方法）均假设所有买家同质——共享单一类型 $\theta^*$。

**现有痛点**: (a) 二分搜索方法无法处理类型可变的反馈——知识集会被矛盾反馈破坏；(b) 回归方法假设 i.i.d. 噪声，而异质类型引入的随机性是上下文依赖的；(c) 朴素离散化的 EXP4 遗憾因动作空间无穷而爆炸。

**核心矛盾**: 买家类型 $\theta_t$ 不可观测，无法将反馈关联到具体类型的不确定性集合，且定价的收益函数不连续（提价 $\epsilon$ 可能导致买家放弃购买）。

**本文切入角度**: 将问题建模为类型分布 $D_\star \in \Delta(\Theta)$ 的学习问题，利用 OPS 框架的 disagreement coefficient 巧妙控制探索-利用权衡。

## 方法详解

### 问题设置
每轮 $t$，买家类型 $\theta_t \sim D_\star$（$|{\rm supp}(D_\star)| = K_\star$ 未知），估值 $v_t = \langle u_t, \theta_t \rangle$，购买决策 $y_t = \mathbb{1}\{v_t \geq p_t\}$。卖家目标：最小化定价遗憾

$$R(T) = \sum_{t=1}^T \left[\mathsf{rev}_\star(\mathsf{br}_\star(u_t), u_t) - \mathsf{rev}_\star(p_t, u_t)\right]$$

其中 $\mathsf{rev}_Q(p) = p \cdot \mathsf{dem}_Q(p)$ 为期望收益，$\mathsf{br}_Q$ 为最优定价。

### 有限模型类的 OPS (Algorithm 1)
维护模型后验 $\mu_t \in \Delta(\mathcal{D})$。每轮：
1. 采样模型 $D_t \sim \mu_t$，定价 $p_t = \mathsf{br}_{D_t}(u_t)$
2. 观测 $y_t$，通过组合损失更新后验：

$$\ell_\lambda(Q, p, y) = \underbrace{(y - \mathsf{dem}_Q(p))^2}_{\text{模型失配}} - \underbrace{\lambda \cdot \mathsf{rev}_Q(\mathsf{br}_Q)}_{\text{乐观偏差}}$$

模型失配惩罚预测与观测的偏差，乐观偏差鼓励探索高潜力模型。

### 不一致系数的控制（核心技术贡献）
定义 disagreement coefficient：

$$\mathsf{dis}(\mathcal{F}) = \sup_{\varepsilon, \delta > 0} \sup_{\nu \in \Delta(\mathcal{X})} \frac{\delta^2}{\varepsilon^2} \mathbb{P}_{p \sim \nu}\left(\exists f \in \mathcal{F}: \mathbb{E}_{q \sim \nu}[f(q)^2] \leq \varepsilon^2 \wedge |f(p)| > \delta\right)$$

**关键洞察**: 对固定上下文 $u$，聚合需求函数 $\mathsf{dem}_\star(\cdot, u)$ 有至多 $K_\star$ 个跳跃点，因此 $\mathsf{dem}_D(\cdot, u) - \mathsf{dem}_\star(\cdot, u)$ 在 $K_\star + 1$ 段上各自单调递减。通过：
- **Lemma 3.4**: 单调递减函数类的 $\mathsf{dis} \leq 2$
- **Lemma 3.5**: $N$-composite 函数类的 $\mathsf{dis}$ 至多放大 $N$ 倍

得到 $\mathsf{dis}(\mathcal{D}, D_\star) \leq 2(K_\star + 1)$，配合 OPS 遗憾界 $\tilde{O}(\lambda \cdot \mathsf{dis} \cdot T + \log|\mathcal{D}|/\lambda)$ 即得 Theorem 3.2。

### 一般情形的 Perturbed OPS (POPS, Algorithm 2)
为处理无限模型类和未知 $K_\star$：
1. **平滑需求函数**: $\mathsf{dem}_Q^\varepsilon(p) = \mathbb{E}_{\delta \sim U[0,\varepsilon]}[\mathsf{dem}_Q(p - \delta)]$，通过随机扰动将离散跳跃平滑化
2. **价格离散化 + 保守扰动**: 定价 $p_t = \max\{\hat{p}_t - \delta_t, 0\}$（$\delta_t \sim U[0,\varepsilon]$），利用收益的单边 Lipschitz 性（提价至多增加 $p' - p$ 收益）控制扰动代价
3. **轨迹耦合**: 证明 $D_\star$ 在有限覆盖 $\mathcal{D}$ 内存在 Lévy 度量邻居，使得 POPS 的购买序列高概率一致
4. **非均匀先验适配未知 $K_\star$**: 对支撑大小大的模型赋予更小先验权重

### 非上下文情形的 ZoomV (Algorithm 4)
$d=1$ 时，结合方差感知置信区间的自适应离散化：
- 标准 zooming dimension $\text{ZoomDim}(c) \leq 1$ → $\tilde{O}(T^{2/3})$
- **方差感知 zooming dimension**: 利用 $\sigma^2(p) = p^2 \mathsf{dem}_\star(p)(1 - \mathsf{dem}_\star(p))$，对低需求类型的高价区间，方差接近零，置信区间极窄
- 证明 $\text{ZoomDimV}(10K_\star) = 0$ → $\tilde{O}(\sqrt{K_\star T})$

## 实验关键数据

### 理论遗憾界汇总

| 设置 | 算法 | 遗憾上界 | 下界 |
|------|------|---------|------|
| 上下文 ($d>1$, $K_\star$ 已知) | OPS | $\tilde{O}(K_\star\sqrt{dT})$ | — |
| 上下文 ($d>1$, $K_\star$ 未知) | POPS | $\tilde{O}(K_\star\sqrt{dT})$ | $\Omega(\sqrt{K_\star dT})$ |
| 非上下文 ($d=1$) | ZoomV | $\tilde{O}(\sqrt{K_\star T})$ | $\Omega(\sqrt{K_\star T})$ |
| 类型可见 (标识符) | Alg 5 | $\tilde{O}(K_\star\sqrt{dT})$ | — |
| 类型可见 (完整向量) | 插值估计 | $\tilde{O}(\sqrt{\min\{K_\star, d\}T})$ | — |

### 关键发现
- 上下文情形遗憾 $\tilde{O}(K_\star\sqrt{dT})$ 对 $d$ 和 $T$ 最优（差 $\sqrt{K_\star}$ 因子）
- 非上下文情形 $\tilde{O}(\sqrt{K_\star T})$ 完全最优（matching lower bound）
- 类型完全可见时遗憾骤降至 $\tilde{O}(\sqrt{\min\{K_\star, d\}T})$，展示反馈丰富度的实质收益
- POPS 的计算复杂度与模型类大小成正比（指数级），这是理论上的瓶颈

## 亮点与洞察
- **Composite disagreement coefficient 分解**是本文最优雅的技术贡献：将无穷动作空间的探索复杂度降维到有限类型数 $K_\star$
- **单边 Lipschitz 性**的系统利用贯穿全文：提价只会减少需求，所以保守定价的遗憾损失有限
- **ZoomV 的方差感知维度**揭示了定价问题的特殊结构：低需求区间虽覆盖数多，但方差小、遗憾贡献低
- 问题建模清晰，将异质买家定价与 contextual bandits 优雅对接

## 局限与展望
- POPS 的运行时间与离散化模型类大小成正比，在 $K_\star$ 和 $d$ 较大时计算不可行
- 上下文情形中 $K_\star$ 的最优依赖（上界 $K_\star$ vs 下界 $\sqrt{K_\star}$）仍有 gap
- 噪声鲁棒性有限：类型分布 $D_\star$ 到 $K_\star$-支撑分布的 Lévy 距离为 $\delta$ 时，遗憾额外增加 $\sqrt{\delta} T^2$
- 仅考虑线性估值模型，非线性估值结构的推广未讨论
- 无实验验证——纯理论论文

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究异质上下文定价，多项新技术工具
- 理论深度: ⭐⭐⭐⭐⭐ 上下界分析 + 多层次方法（OPS/POPS/ZoomV）
- 实验充分度: ⭐⭐ 纯理论，无实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，层次递进，符号略重但一致
- 综合: ⭐⭐⭐⭐ 重要理论突破，对在线学习定价有深远影响

## 相关工作与启发
- **vs Cohen et al. (2020) / Lobel et al. (2018)**: 经典多维二分搜索方法，限于同质买家 $K_\star=1$。异质类型产生的矛盾反馈会破坏知识集
- **vs Javanmard & Nazerzadeh (2019)**: 回归方法假设 i.i.d. 噪声，无法处理异质类型引入的上下文依赖随机性
- **vs Cesa-Bianchi et al. (2019)**: 非上下文异质定价的先驱工作，但遗憾依赖实例参数 $V$ 可在最坏情形下爆炸。ZoomV 通过方差感知消除了该依赖
- **vs Krishnamurthy et al. (2023)**: 腐败鲁棒上下文搜索（$C=o(T)$ 个腐败响应），不适用于真正大规模异质种群
- Composite disagreement coefficient 分解可迁移到其他结构化 contextual bandits（如分段线性收益的拍卖设计、推荐系统中的异质用户建模）
