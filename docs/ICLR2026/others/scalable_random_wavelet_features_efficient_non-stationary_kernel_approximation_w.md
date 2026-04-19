---
title: >-
  [论文解读] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees
description: >-
   提出 Random Wavelet Features (RWF)，通过从小波族中随机采样构建可扩展的非平稳核近似，保留随机特征的线性时间复杂度，同时具有正定性、无偏性和一致收敛保证。
tags:

---

# Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2602.00987](https://arxiv.org/abs/2602.00987)
- **代码**: 未公开
- **领域**: others
- **关键词**: kernel methods, random features, wavelets, non-stationary, Gaussian processes, multi-resolution

## 一句话总结
提出 Random Wavelet Features (RWF)，通过从小波族中随机采样构建可扩展的非平稳核近似，保留随机特征的线性时间复杂度，同时具有正定性、无偏性和一致收敛保证。

## 研究背景与动机
- **GP 的困境**：表达力 vs 效率。精确 GP 计算 $O(N^3)$ 且通常用平稳核。
- **Random Fourier Features (RFF)**：基于 Bochner 定理将平稳核近似为线性模型，$O(ND^2)$ 训练，但本质上限于平稳核。
- **非平稳建模挑战**：地理空间、语音等领域的统计特性随位置变化。Deep GP、谱混合核等方法虽能建模非平稳性但代价高。
- **核心 Gap**：缺乏一种既像 RFF 一样可扩展、又能原生捕获非平稳性的随机特征框架。

## 方法详解

### 核心思想
用小波代替傅里叶基函数。小波天然具有空间和频率上的局部化，通过随机采样不同尺度和平移的小波原子，构建非平稳核。

### 小波核构建
从母小波 $\psi: \mathbb{R}^d \to \mathbb{R}$ 生成原子族：
$$\psi_{s,\mathbf{t}}(\mathbf{x}) = s^{-d/2} \psi\left(\frac{\mathbf{x} - \mathbf{t}}{s}\right)$$

非平稳核通过积分得到：
$$k(\mathbf{x}, \mathbf{y}) = \int_0^\infty \int_{\mathbb{R}^d} \psi_{s,\mathbf{t}}(\mathbf{x}) \psi_{s,\mathbf{t}}(\mathbf{y}) \, p(s, \mathbf{t}) \, d\mathbf{t} \, ds$$

其中 $p(s, \mathbf{t})$ 是尺度-平移的采样密度。

### Random Wavelet Features
Monte Carlo 近似上述积分：

$$z(\mathbf{x}) = \frac{1}{\sqrt{D}} [\psi_{s_1,\mathbf{t}_1}(\mathbf{x}), \ldots, \psi_{s_D,\mathbf{t}_D}(\mathbf{x})]^\top$$

核近似：$\hat{k}(\mathbf{x}, \mathbf{y}) = z(\mathbf{x})^\top z(\mathbf{y})$

然后转为贝叶斯线性回归：
$$\mathbf{S}_{\mathbf{w}} = (\mathbf{I}_D + \sigma^{-2} \mathbf{Z}^\top \mathbf{Z})^{-1}, \quad \mathbf{m}_{\mathbf{w}} = \sigma^{-2} \mathbf{S}_{\mathbf{w}} \mathbf{Z}^\top \mathbf{y}$$

### 采样策略
- 尺度 $s$：对数均匀分布（覆盖多分辨率）
- 平移 $\mathbf{t}$：均匀覆盖数据凸包
- 母小波选择：Morlet（时频分析）或 Daubechies（尖锐转变）

### 计算复杂度
| 方法 | 训练复杂度 | 预测复杂度 |
|------|-----------|-----------|
| 精确 GP | $O(N^3)$ | $O(N^2)$ |
| SVGP | $O(NM^2)$ / 步 (迭代) | $O(M^2)$ |
| RFF-GP | $O(ND^2)$ (单次) | $O(D^2)$ |
| **RWF-GP** | $O(ND^2)$ (单次) | $O(D^2)$ |

> RWF 与 RFF 同等高效，但能建模非平稳性。

## 理论保证

### 定理 4.1（正定性）
RWF 构建的核 $k(\mathbf{x}, \mathbf{y})$ 对任意非负测度 $p(s, \mathbf{t})$ 都是正定核。

### 引理 4.1（无偏性）
$\mathbb{E}[\hat{k}(\mathbf{x}, \mathbf{y})] = k(\mathbf{x}, \mathbf{y})$，对所有 $\mathbf{x}, \mathbf{y} \in \mathcal{X}$。

### 一致收敛保证
$$\Pr\left[\sup_{\mathbf{x}, \mathbf{y} \in \mathcal{X}} |\hat{k}(\mathbf{x}, \mathbf{y}) - k(\mathbf{x}, \mathbf{y})| > \epsilon\right] \leq \mathcal{O}\left(\exp(-D\epsilon^2 / B^4)\right)$$

即 $D = O(B^4 / \epsilon^2)$ 个特征即可保证 $\epsilon$-精度的一致近似。

## 实验关键数据

### 主实验：回归任务

| 方法 | 合成非平稳 | 语音数据 | 大尺度回归 |
|------|-----------|---------|-----------|
| RFF-GP | 欠拟合局部结构 | RMSE 高 | 快但不准 |
| SVGP | 较好 | 较好 | 中速中准 |
| Deep GP | 最好 | 最好 | 慢但准 |
| **RWF-GP** | **接近 Deep GP** | **接近 Deep GP** | **快且准** |

> RWF-GP 在准确率上接近 Deep GP，速度接近 RFF-GP。

### 消融实验：特征数量影响

| 特征数 $D$ | RFF RMSE | RWF RMSE |
|-----------|----------|----------|
| 50 | 0.85 | 0.62 |
| 100 | 0.78 | 0.45 |
| 500 | 0.72 | 0.31 |
| 1000 | 0.70 | 0.28 |

> RWF 在相同特征数下一致优于 RFF，增加特征数时改善更显著（因小波局部化优势累积）。

### 关键发现
1. RWF 在非平稳信号上显著优于 RFF，且不增加计算复杂度
2. 多分辨率结构使精细小波捕获局部事件、粗糙小波建模长程趋势
3. 不同小波族适配不同数据特性（Morlet 适合振荡信号，Mexican Hat 适合脉冲检测）
4. 与 Deep GP 精度接近但快一个数量级

## 亮点与洞察
- **填补理论空白**：首次为基于小波的随机特征提供完整的正定性→无偏性→一致收敛理论链
- **概念优雅**：RFF 用全局正弦基 → 平稳核；RWF 用局部小波基 → 非平稳核，形成自然推广
- **实用效率**：保留 $O(ND^2)$ 训练的同时获得非平稳建模能力
- **多分辨率灵活性**：通过调整 $p(s, \mathbf{t})$ 适配不同数据特性

## 局限性
- 母小波和采样分布的选择仍有一定的超参调优需求
- 各向同性缩放可能在高维各向异性问题中受限
- 对于某些特定的非平稳模式（如突变点），固定小波族可能不够灵活
- 与端到端可学习的 Deep GP 相比，表达力仍有理论上界

## 相关工作
- **Random Fourier Features**: Rahimi & Recht (2007) 开创工作；后续扩展到自适应和结构化采样
- **小波核**: Zhang et al. (2004) 小波 SVM，Guo et al. (2024) 固定小波基贝叶斯回归
- **非平稳 GP**: Deep GP (Damianou et al., 2013), 谱混合核 (Wilson & Adams, 2013)
- **可扩展 GP**: KISS-GP (Wilson & Nickisch, 2015), Deep Kernel Learning (Wilson et al., 2016)

## 评分
- 新颖性: ⭐⭐⭐⭐ — 小波 + 随机特征的自然结合，但概念上不算革新
- 理论深度: ⭐⭐⭐⭐⭐ — 正定性、无偏性、方差界、一致收敛全覆盖
- 实验充分性: ⭐⭐⭐⭐ — 合成+语音+大尺度回归，但缺少更多实际应用
- 实用价值: ⭐⭐⭐⭐ — 对需要非平稳建模且要求可扩展性的场景直接可用

<!-- RELATED:START -->

## 相关论文

- [Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [Consistent Low-Rank Approximation](consistent_low-rank_approximation.md)
- [Active Learning for Decision Trees with Provable Guarantees](active_learning_for_decision_trees_with_provable_guarantees.md)
- [Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)

<!-- RELATED:END -->
