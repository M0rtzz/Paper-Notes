---
title: >-
  [论文解读] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics
description: >-
  [ICML2025][传感器调度] 提出一种面向连续-离散卡尔曼滤波 (CD-KF) 的最优传感器调度框架：将多传感器观测建模为独立 Poisson 过程，推导后验协方差矩阵的可微上界，利用梯度优化方法联合优化观测频率与辅助动力学输入，并通过 Wasserstein-2 最优量化确定性地选取观测时刻。
tags:
  - ICML2025
  - 传感器调度
  - 卡尔曼滤波
  - 连续-离散系统
  - Poisson过程
  - 最优控制
  - 高斯过程回归
---

# Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics

**会议**: ICML2025  
**arXiv**: [2507.11240](https://arxiv.org/abs/2507.11240)  
**代码**: [GitHub](https://github.com/MOHAMMADZAHD93/When2measureKF)  
**领域**: 控制/滤波 (Control & Filtering)  
**关键词**: 传感器调度, 卡尔曼滤波, 连续-离散系统, Poisson过程, 最优控制, 高斯过程回归

## 一句话总结

提出一种面向连续-离散卡尔曼滤波 (CD-KF) 的最优传感器调度框架：将多传感器观测建模为独立 Poisson 过程，推导后验协方差矩阵的可微上界，利用梯度优化方法联合优化观测频率与辅助动力学输入，并通过 Wasserstein-2 最优量化确定性地选取观测时刻。

## 研究背景与动机

许多真实系统（如血压、海洋温度、辐射水平）的状态连续演化，但只能通过离散、不规则时刻的传感器进行间歇观测。这就是 **连续-离散卡尔曼滤波 (CD-KF)** 的典型场景。

现实中的难点在于：

- **多传感器异构性**：不同传感器具有不同精度、不同能耗和不同约束
- **辅助状态耦合**：观测过程与辅助状态空间模型耦合（如传感器温度影响精度、能量随观测消耗）
- **资源有限**：不能简单地以最高频率对所有传感器采样

现有工作在以下方面存在空白：

1. 大多数工作仅考虑离散时间设置，未涉及连续-离散场景
2. 尚无方法同时考虑传感器调度与一般辅助状态动力学的耦合

本文填补了这一空白，提出了一个统一的最优控制框架。

## 方法详解

### 1. 系统模型

连续-离散状态空间模型 (SSM) 由以下方程描述：

**状态演化**（连续时间 SDE）：

$$dx = A(\xi, t) x \, dt + \sigma(\xi, t) \, dW, \quad x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

**离散观测**（传感器 $s$ 在时刻 $t_i$）：

$$y^s(t_i) = C_s(\xi(t_i), t_i) x(t_i) + v^s(\xi(t_i), t_i)$$

其中 $\xi$ 为辅助状态（分为受观测扰动部分 $\xi_p$ 和不受扰动部分 $\xi_u$），$v^s \sim \mathcal{N}(0, R_s(\xi, t))$。

### 2. Poisson 过程建模

将每个传感器 $s$ 的观测到达建模为强度为 $\lambda_s(t)$ 的非齐次 Poisson 过程 $N_s(t)$。随机化后的协方差演化为：

$$d\Sigma = \left(A\Sigma + \Sigma A^\top + \sigma\sigma^\top\right)dt - \sum_{s=1}^{S} K_s \cdot C_s \cdot \Sigma \, dN_s$$

### 3. 协方差上界（Proposition 6.1）

核心理论贡献：推导出均值后验协方差矩阵的**连续可微上界** $\hat{\Sigma}(t)$，其满足 ODE：

$$\frac{d\hat{\Sigma}}{dt} = A\hat{\Sigma} + \hat{\Sigma}A^\top + \sigma\sigma^\top - \sum_{s=1}^{S} \lambda_s(t) K_s C_s \hat{\Sigma}$$

关键性质：$\bar{\Sigma}(t; \xi^*) \preceq \hat{\Sigma}(t)$（Loewner 序），且 $\hat{\Sigma}$ 对 $\lambda_s$ 连续可微，可直接使用梯度优化。

### 4. 辅助状态上界（Proposition 6.2）

在凹/凸假设（Assumption 5.1）下，辅助状态均值 $\bar{\xi}$ 可被确定性轨迹 $\hat{\xi}$ 上/下界化：

$$\frac{d\hat{\xi}_p}{dt} = f_p(\hat{\xi}, u, t) + \sum_{s=1}^{S} \lambda_s(t) g_s(\hat{\xi}, u, t)$$

对仿射系统，$\hat{\xi}_p = \bar{\xi}_p$（精确等式）。

### 5. 最优控制问题 (OCP)

联合优化观测率 $\lambda$ 和辅助输入 $u$：

$$\min_{\lambda \geq 0, \, u \in \mathcal{U}} \int_0^T \mathcal{L}(\hat{\xi}, \hat{\Sigma}, u, \lambda) \, dt + \mathcal{L}_T(\cdot)$$

运行代价示例：$\mathcal{L} = w_\Sigma \operatorname{tr}(\hat{\Sigma}) + w_\lambda \|\lambda\|^2 + w_u \|u\|^2 + w_\varepsilon \varepsilon^2$

约束包括：能量下界 $\hat{\eta} \geq c_\eta$、输入上下界、协方差迹约束 $\operatorname{tr}(\hat{\Sigma}) \leq c_\Sigma + \varepsilon$ 等。

### 6. 确定性观测时刻选取（Proposition 8.1）

优化得到连续速率 $\lambda_s(t)$ 后，通过最小化 **Wasserstein-2 距离** 选取确定性观测时刻：

$$\bar{t}_i^s = \mathbb{E}[\tau_s \mid \tau_s \in [a_{i-1}^s, a_i^s]] = \frac{\int_{a_{i-1}^s}^{a_i^s} t \lambda_s(t) dt}{\int_{a_{i-1}^s}^{a_i^s} \lambda_s(t) dt}$$

将时间轴按累积强度 $\Lambda_s(t)$ 等分为 $n_s$ 段，每段取条件质心。该方法保持一阶矩匹配，且量化误差最小。

## 实验关键数据

### 实验设置

- **场景**：机器人搭载两个异构传感器进行时间高斯过程回归（Matérn 核）
- **传感器 1**：精度高（$R_{1_{\max}}$ 小），能耗高（$c_1$ 大）
- **传感器 2**：精度低，能耗低
- **观测噪声**：依赖机器人与目标距离 $R_s(p_r) = R_{s_{\max}} \exp(r_s \|p_r - p_p\|^2)$
- **扩展场景**：放射性环境，观测导致辐射累积退化 $\zeta_s$

### 对比方法

| 方法 | 说明 |
|------|------|
| **Optimized** (本文) | 联合优化 $\lambda$ 和 $u$，确定性量化选取时刻 |
| M-Optimized | 同上优化速率，但通过多次采样 Poisson 取最优 |
| Greedy | 每步贪心选取分数最高的传感器 |
| Random | 按均匀速率随机 Poisson 采样 |

### 核心结果（放射性环境）

| 指标 | 方法 | 均值 | 标准差 | 最大值 |
|------|------|------|--------|--------|
| 协方差迹 | **Optimized** | **1.902** | **0.341** | **2.945** |
| | M-Optimized | 1.925 | 0.364 | 3.001 |
| | Greedy | 2.608 | 0.488 | 3.219 |
| | Random | 2.228 | 0.470 | 2.921 |
| 剩余能量 $\eta$ | **Optimized** | **21.54** | 10.17 | 50.0 |
| | Greedy | -1.67 | 14.06 | 50.0 |
| | Random | -17.44 | 31.35 | 50.0 |
| 辐射退化 | **Optimized** | **0.091** | 0.065 | 0.186 |
| | Greedy | 0.193 | 0.081 | 0.232 |
| | Random | 0.792 | 0.558 | 1.576 |

**关键观察**：Optimized 在协方差迹（估计精度）最低的同时保持正能量（21.54 vs Greedy -1.67、Random -17.44），且辐射退化最低。

## 亮点与洞察

1. **理论优雅**：将随机离散观测通过 Poisson 过程建模转为连续可微目标，巧妙地绕过了组合优化难题
2. **统一框架**：首次将传感器调度、辅助状态动力学、轨迹优化统一在一个最优控制问题中
3. **确定性量化**：基于 Wasserstein-2 最优量化选取观测时刻，具闭式解、计算高效、保持一阶矩
4. **广泛适用**：框架可扩展至非线性 SSM（EKF/UKF 近似）、水质监测（附录 E）、航天器监测（附录 F）
5. **实际意义**：Greedy 和 Random 方法在带约束场景（能量有限、辐射退化）中频繁违约，本方法天然满足

## 局限与展望

1. **线性假设**：核心理论针对线性 SSM，非线性情况仅提供近似保证（类似 EKF 假设）
2. **开环规划**：解的是有限时域开环 OCP，未考虑在线反馈/闭环重规划（MPC 式）
3. **计算成本**：OCP 的数值求解（直接配点法/多重打靶法）可能对高维状态空间较昂贵
4. **Poisson 近似**：实际场景中传感器的观测事件可能不服从 Poisson 过程
5. **单目标过程**：实验仅考虑单个被观测过程，多目标同时监测的扩展尚未讨论

## 相关工作与启发

- 与 (Ny et al., 2009) 的连续时间传感器管理相比，本文处理**离散非规则观测**
- 与 (Qin et al., 2024) 的 Neural ODE 方法相比，本文提供**解析可微上界**而非黑箱学习
- Wasserstein 最优量化的思路可启发其他需要从连续分布选取离散点的问题
- 辅助状态建模（能量、退化）的思路可推广到传感器网络、自动驾驶中的主动感知

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次统一 CD-KF + 辅助动力学 + Poisson 调度
- 实验充分度: ⭐⭐⭐ — 场景有说服力但仅为低维仿真
- 写作质量: ⭐⭐⭐⭐ — 数学严谨、结构清晰
- 价值: ⭐⭐⭐⭐ — 对资源受限下的主动感知有直接指导意义
