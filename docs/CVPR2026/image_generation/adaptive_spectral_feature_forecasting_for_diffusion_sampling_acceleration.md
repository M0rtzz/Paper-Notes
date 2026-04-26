---
title: >-
  [论文解读] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration
description: >-
  [CVPR 2026][图像生成][扩散采样加速] 提出 Spectrum，一种基于切比雪夫多项式的全局谱域特征预测方法，将扩散模型去噪器的中间特征视为时间函数并用岭回归拟合系数，实现误差不随步长增长的长程特征预测，在 FLUX.1 上达到 4.79× 加速、在 Wan2.1-14B 上达到 4.67× 加速而质量几乎无损。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散采样加速
  - 特征缓存
  - 切比雪夫多项式
  - 谱方法
  - training-free
---

# Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration

**会议**: CVPR 2026  
**arXiv**: [2603.01623](https://arxiv.org/abs/2603.01623)  
**代码**: [GitHub](https://hanjq17.github.io/Spectrum)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 扩散采样加速, 特征缓存, 切比雪夫多项式, 谱方法, training-free

## 一句话总结

提出 Spectrum，一种基于切比雪夫多项式的全局谱域特征预测方法，将扩散模型去噪器的中间特征视为时间函数并用岭回归拟合系数，实现误差不随步长增长的长程特征预测，在 FLUX.1 上达到 4.79× 加速、在 Wan2.1-14B 上达到 4.67× 加速而质量几乎无损。

## 研究背景与动机

扩散模型（特别是 Diffusion Transformer）生成高质量图像/视频，但推理需要数十到上百次的去噪器前向传播，计算成本极高。现有加速方案中，**特征缓存复用**思路无需额外训练，通过在选定时步缓存特征并在后续时步复用来跳过昂贵的网络计算。

然而现有缓存方法依赖**局部近似**：
- **朴素复制**（Naive reusing）：直接复制最近缓存的特征，过于简化时序变化
- **TaylorSeer**：基于离散 Taylor 展开的局部预测，但其误差以 $((j-k)\delta_t)^{P+1}$ 增长——**步长越大误差越大**，在高加速比时质量严重退化

核心矛盾在于：高加速比要求大跨度跳步，而局部预测器的误差恰恰在大跨度时急剧恶化。作者从理论分析中发现 Taylor 预测器的最坏情况误差，并指出其根本局限：无法捕捉采样轨迹的全局长程动态。

切入角度：**从时域局部近似转向频域全局建模**。将去噪器输出的每个特征通道视为关于时间的函数，用切比雪夫多项式——一组具有良好数值性质的正交基——在全局范围上逼近，从而打破局部预测的误差瓶颈。

## 方法详解

### 整体框架

Spectrum 的核心流程：在 $N$ 步扩散采样中，选定一部分时步集合 $\mathbb{U}$ 执行实际网络前向传播，在剩余时步 $\mathbb{V} = \mathbb{T} \setminus \mathbb{U}$ 使用谱域预测器代替。这是一个 **在线拟合-预测（fitting-then-forecasting）** 的过程。

### 关键设计

1. **切比雪夫多项式谱分解**：

    - **功能**：将去噪器输出特征 $\mathbf{h}_t = [h_1(t), \cdots, h_F(t)]$ 的每个通道视为时间函数，用 $M$ 阶切比雪夫多项式逼近：
    $h_i(t) = \sum_{m=0}^{M} c_{m,i} T_m(\tau), \quad \tau = 2t - 1$
    - **设计动机**：切比雪夫多项式形成正交基，其逼近误差由多项式阶数 $M$ 控制而不依赖步长——即使预测很远的未来步也有可控的精度。根据 Theorem 3.2，对于解析扩展到 Bernstein 椭圆的函数，截断切比雪夫级数的误差以 $\rho^{-M}$ 指数衰减

2. **在线岭回归系数拟合**：

    - **功能**：利用已缓存的特征点在线拟合切比雪夫系数
    - **核心公式**：构建设计矩阵 $\mathbf{\Phi}_{t_j}$ 和特征矩阵 $\mathbf{H}_{t_j}$，求解岭回归问题：
    $\mathbf{C}_{t_j} = (\mathbf{\Phi}_{t_j}^\top \mathbf{\Phi}_{t_j} + \lambda \mathbf{I})^{-1} \mathbf{\Phi}_{t_j}^\top \mathbf{H}_{t_j}$
    - 矩阵逆的维度仅为 $(M+1) \times (M+1)$，当 $M$ 很小时计算开销可忽略（通过 Cholesky 分解求解）
    - **正则化项 $\lambda$**：防止过拟合、增强数值稳定性，实验证实其关键作用

3. **自适应时步调度**：

    - **功能**：在采样早期更密集地执行实际前向传播，后期逐渐增大预测器使用比例
    - **核心思路**：选择 $\mathbb{U} = \{\tau_j : j = \lfloor\alpha \frac{r(r+1)}{2}\rfloor\}$，间隔随 $r$ 增大而增大
    - **设计动机**：早期步骤的误差会通过 ODE 积分传播到后续步骤并放大，因此早期需要更多实际网络计算以保证基础精度

4. **仅缓存最终层**：

    - **功能**：只对最终注意力块的输出实例化 Spectrum，而非逐层缓存
    - **设计动机**：原始 TaylorSeer 对每层都缓存，引入 $L$ 倍额外开销；实验发现仅缓存最终层质量相当甚至更优

### 理论分析

**核心定理 (Theorem 3.3)**：Spectrum 的误差上界不依赖步长 $\tau_j - \tau_k$，而是由多项式阶数 $M$、设计矩阵最小奇异值 $\sigma_{\min}(\mathbf{\Phi})$ 和正则化强度 $\lambda$ 控制。这与 Taylor 方法的误差 $\propto ((j-k)\delta_t)^{P+1}$ 形成鲜明对比。

## 实验关键数据

### 主实验一：文本到图像生成（DrawBench, Table 1）

| 方法 | FLUX Speedup | FLUX PSNR↑ | FLUX SSIM↑ | FLUX LPIPS↓ | FLUX ImageReward↑ |
|------|-------------|-----------|-----------|------------|------------------|
| 50 steps (ref) | 1.00× | - | - | - | 1.00 |
| TaylorSeer (N=4,O=1) | 3.13× | 22.31 | 0.841 | 0.215 | 0.99 |
| TaylorSeer (N=4,O=2) | 3.03× | 20.76 | 0.812 | 0.247 | 1.02 |
| **Spectrum (α=0.75)** | **3.47×** | **24.32** | **0.854** | **0.217** | 0.99 |
| TaylorSeer (N=6,O=1) | 4.14× | 20.24 | 0.785 | 0.294 | 1.00 |
| **Spectrum (α=3.0)** | **4.79×** | **22.21** | **0.788** | **0.261** | 1.00 |

### 主实验二：文本到视频生成（VBench, Table 2）

| 方法 | Wan2.1-14B Speedup | PSNR↑ | SSIM↑ | VBench Quality↑ |
|------|-------------------|-------|-------|----------------|
| 50 steps (ref) | 1.00× | - | - | 83.15 |
| TaylorSeer (N=4,O=1) | 3.01× | 19.46 | 0.660 | 82.74 |
| **Spectrum (α=0.75)** | **3.40×** | **22.78** | **0.749** | **82.80** |
| TaylorSeer (N=6,O=1) | 3.94× | 17.24 | 0.585 | 81.38 |
| **Spectrum (α=3.0)** | **4.67×** | **21.24** | **0.694** | **82.21** |

在高加速比场景（4–5×）下，Spectrum 相对 TaylorSeer 的 PSNR 优势达 2–4 dB。

### 消融实验

- **正则化强度 $\lambda$**：$\lambda = 0$ 时效果不佳，$\lambda = 0.1$ 最优——正则化对防止过拟合至关重要
- **多项式阶数 $M$**：$M = 4$ 已足够，更高阶无明显增益
- **自适应调度 vs 固定间隔**：自适应调度在高加速比下比固定间隔好 1–2 dB PSNR
- **仅缓存最终层 vs 逐层缓存**：仅最终层不仅节省内存，效果甚至更优

### 关键发现

- Taylor 预测器在高加速比时夸大局部细节但丢失全局语义；Spectrum 保持了色彩一致性和语义正确性
- Spectrum 的计算开销相对于网络前向传播可忽略不计（时间复杂度主导项为 $O(K(M+1)F)$，$K$ 和 $M$ 都很小）
- 方法对图像和视频扩散模型都有效，且与不同 ODE solver 兼容

## 亮点与洞察

1. **从局部到全局的范式转变**：将特征缓存从时域局部近似推进到谱域全局建模，是方法论上的跳跃
2. **理论保证**：误差不随步长积累的定理是该方法的核心理论贡献，为高加速比场景提供了信心
3. **工程简洁性**：仅需岭回归拟合系数、Cholesky 分解求逆，额外开销极小
4. **广泛适用性**：在 FLUX.1、SD3.5-Large、Wan2.1-14B、HunyuanVideo 四个 SOTA 模型上都有效

## 局限与展望

- 需要至少 $M+1$ 个缓存点才能开始预测，初始阶段仍需执行完整网络
- 假设特征关于时间的函数是解析的（可扩展到 Bernstein 椭圆），对实际特征的平滑性假设是否总成立待验证
- 自适应调度的超参数 $\alpha$ 需要针对不同模型调优
- 与蒸馏方法、token 剪枝等正交技术的联合使用未探索

## 相关工作与启发

- **TaylorSeer**：最直接的对比方法，用离散 Taylor 展开预测缓存特征
- **TeaCache**：动态决定何时缓存的方案，与 Spectrum 的调度策略互补
- **FORA/ToCa**：基于直接缓存复用的方法，效果不如预测式方案
- **启发**：切比雪夫多项式在数值分析中的经典地位被巧妙引入深度学习推理加速，提示更多数学工具（如 Fourier 基、小波基）可能也适用于类似场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将谱域方法引入扩散特征缓存加速，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖4个SOTA模型（图像+视频），两个加速档位，完整消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，从Taylor误差分析自然引出动机，逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 4-5×加速且质量近无损，training-free，实际价值很高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)
- [\[CVPR 2026\] TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration](tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)
- [\[ICML 2025\] SADA: Stability-guided Adaptive Diffusion Acceleration](../../ICML2025/image_generation/sada_stability-guided_adaptive_diffusion_acceleration.md)

<!-- RELATED:END -->
