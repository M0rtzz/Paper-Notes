---
title: >-
  [论文解读] Hessian Geometry of Latent Space in Generative Models
description: >-
  [ICML2025][图像生成][Fisher信息度量] 提出通过重建 Fisher 信息度量来分析生成模型潜空间几何的方法，发现扩散模型潜空间中存在分形结构的相变边界，在相边界处 Lipschitz 常数发散。
tags:
  - ICML2025
  - 图像生成
  - Fisher信息度量
  - Hessian几何
  - 潜空间分析
  - 相变
  - 扩散模型
  - 测地线
---

# Hessian Geometry of Latent Space in Generative Models

**会议**: ICML2025  
**arXiv**: [2506.10632](https://arxiv.org/abs/2506.10632)  
**代码**: [GitHub](https://github.com/alobashev/hessian-geometry-of-diffusion-models)  
**领域**: 生成模型理论 / 信息几何  
**关键词**: Fisher信息度量, Hessian几何, 潜空间分析, 相变, 扩散模型, 测地线

## 一句话总结

提出通过重建 Fisher 信息度量来分析生成模型潜空间几何的方法，发现扩散模型潜空间中存在分形结构的相变边界，在相边界处 Lipschitz 常数发散。

## 研究背景与动机

- **核心问题**：图像生成模型在潜空间插值时经常出现突变（如内容骤变），说明潜空间并非平滑的，但现有方法缺乏系统的几何分析工具
- **两条研究线索**：
    1. **潜空间黎曼几何**：Park et al. (2023) 通过 Jacobian 奇异向量构建潜空间基，Shao et al. (2018) 用像素空间的 pullback 度量计算测地线，但仅适用于确定性生成器
    2. **统计物理中的相变学习**：ML 方法用于识别 Ising 模型等的相变边界，Walker et al. (2020) 发现 VAE 隐含提取了充分统计量
- **动机**：将生成模型统一视为统计物理系统，用信息几何方法分析其潜空间结构，适用于随机生成过程（如扩散模型的随机采样）

## 方法详解

### 核心框架：两步法重建 Fisher 度量

**Step 1：后验分布近似**

给定生成模型 $p(x|t)$，从采样 $x_1, \dots, x_N \sim p(x|t')$ 近似后验分布 $p(t|x_1, \dots, x_N)$。

**关键定理 (Theorem 3.1)**：对指数族分布，当 $N \to \infty$ 时：

$$\lim_{N \to \infty} \left(p(t|x_1, \dots, x_N)\right)^{1/N} = e^{-D_{\log Z(t)}(t, t')}$$

其中 $D_{\log Z(t)}(t, t')$ 是 Bregman 散度，对指数族等价于 KL 散度 $D_{\text{KL}}(p(x|t') \| p(x|t))$。

**两种后验近似策略**：

- **直接训练映射**（U²-Net）：适用于统计物理模型（Ising, TASEP），样本具有随机性，像素级不相关
- **特征提取器**（CLIP）：适用于图像领域，利用 $D_{\text{KL}} \approx \frac{1}{2}\|\mathcal{E}(x_1) - \mathcal{E}(x_2)\|^2$ 近似

**Step 2：Fisher 度量重建**

**关键定理 (Theorem 3.2)**：通过最小化 Bregman 散度的 MSE 损失可恢复 $\log Z(t)$ 的 Hessian（即 Fisher 度量）：

$$g_F(t) = \nabla^2 \log Z(t)$$

实际训练使用 Jensen-Shannon 散度替代 MSE（避免梯度消失）：

$$\mathcal{L}_1(\theta) = \int_{\mathcal{S}} D_{\text{JS}}\left(p(t|x_1, \dots, x_N),\; p_{\log Z_\theta}(t|t')\right) dt'$$

- 用 MLP（5层隐藏层，512维，ReLU）参数化 $\log Z_\theta(t)$
- 不要求 MLP 满足凸性约束，训练过程中自然收敛至凸函数

### 测地线近似

获得 Fisher 度量后，离散化曲线 $\gamma(t)$ 为 $\{\gamma_0, \gamma_1, \dots, \gamma_N\}$，用 Adam 优化中间点最小化路径长度：

$$L[\gamma(t)] = \int_0^1 \sqrt{\dot{\gamma}(t)^T g_F(\gamma(t)) \dot{\gamma}(t)} \, dt$$

### 理论基础

- **Bryant–Amari–Armstrong 定理**：任何 2D 解析黎曼度量局部可表示为 Hessian 结构，保证方法对任意 2D 潜空间切片理论合理
- **指数族性质**：Fisher 度量等于 $\log$ 配分函数的 Hessian，$\log Z(t)$ 仅需恢复至仿射变换即可

## 实验关键数据

### Ising 模型 & TASEP：精确可解验证

| 模型 | 方法 | F RMSE | dF/d(param1) RMSE | dF/d(param2) RMSE |
|------|------|--------|-------------------|-------------------|
| **Ising** | Convex (Ours) | **0.0883±0.0006** | **0.1106±0.0002** | **0.1237±0.0016** |
| Ising | Mean-as-Stat | 0.0981±0.0010 | 0.4766±0.0023 | 1.0936±0.0033 |
| Ising | PCA-VAE | 0.1669±0.0018 | 0.7428±0.0025 | 0.7988±0.0022 |
| **TASEP** | Convex (Ours) | **0.0112±0.00008** | **0.1165±0.0025** | **0.1135±0.0017** |
| TASEP | Mean-as-Stat | 0.0529±0.0005 | 0.3832±0.0038 | 0.3833±0.0031 |
| TASEP | PCA-VAE | 0.0524±0.0006 | 0.3837±0.0038 | 0.3872±0.0022 |

偏导数重建精度提升 3–8 倍，对识别相变边界至关重要。

### 扩散模型：2D 潜空间切片分析

实验基于 StableDiffusion 1.5 (Dreamshaper8)，DDIM 调度器（50步，CFG=5），每组生成 60000 张图像。

| 指标 | Geodesic (Ours) | Linear | Geodesic (Wang/Shao) |
|------|----------------|--------|---------------------|
| CLIP Length | **72.3±4.00** | 73.6±3.54 | 73.6±4.37 |
| Pixel Length | 2.77×10⁶ | 2.76×10⁶ | **2.74×10⁶** |
| PPL | **3.12±0.16** | 3.17±0.23 | 3.19±0.21 |

**关键发现**：

- 重建的 $\log Z(\alpha, \beta)$ 非光滑，导数存在突变 → 反映图像空间的相变
- **分形相边界**：放大相边界可观察到自相似结构，尺度从 $10^{-5}$ 延伸到 float16 精度 $10^{-8}$
- **单相内测地线近似线性**，但在相边界处线性性破缺
- 扩散模型在相边界处 Lipschitz 常数关于潜空间发散（据作者所知为新发现）

## 亮点与洞察

1. **统一框架**：将统计物理与生成模型用信息几何统一，方法同时适用于指数族（Ising, TASEP）和非指数族（扩散模型的 2D 切片，通过 Bryant–Amari–Armstrong 定理保证）
2. **分形相变发现**：扩散模型潜空间的相边界具有分形结构，这是与经典连续相边界的本质区别
3. **Lipschitz 发散**：首次报告扩散模型关于潜空间的 Lipschitz 常数发散现象
4. **实用价值**：Fisher 度量引导的测地线插值在感知上比线性插值更平滑
5. **JSD 训练技巧**：用 Jensen-Shannon 散度替代 MSE 损失解决梯度消失问题

## 局限与展望

1. **维度受限**：当前方法仅处理 2D 潜空间切片（依赖 Bryant–Amari–Armstrong 定理），高维潜空间的完整分析仍不可行
2. **计算开销大**：需为每个 2D 切片生成 60000 张图像，扩展性差
3. **CLIP 近似的局限**：CLIP 距离作为 KL 散度的近似依赖正态性假设，实际中可能不成立
4. **确定性 vs 随机采样**：DDIM η>0 时 CLIP 方案的相边界被模糊，方法对噪声敏感
5. **测地线改进有限**：表2中各方法差异不大，测地线的实际优势主要体现在跨相边界场景
6. **仅验证 SD1.5**：未在更先进的扩散模型（如 SDXL, SD3）上验证

## 相关工作与启发

- Park et al. (2023)：Jacobian 奇异向量的潜空间基构建
- Shao et al. (2018)：pullback 度量测地线（仅限确定性模型）
- Walker et al. (2020)：VAE 隐含提取 Ising 模型充分统计量
- Wang et al. (2021)：GAN 潜空间的 LPIPS pullback 度量
- Yang et al. (2023)：扩散模型关于时间变量的 Lipschitz 常数讨论

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（信息几何 + 统计物理 + 生成模型的三重交叉，分形相变发现具有原创性）
- 实验充分度: ⭐⭐⭐⭐（精确可解模型验证充分，但扩散模型实验仅限 SD1.5 和 2D 切片）
- 写作质量: ⭐⭐⭐⭐（理论推导清晰，但 LaTeX 排版中符号较密集）
- 价值: ⭐⭐⭐⭐（为理解生成模型潜空间提供新视角，但实用性受维度限制）

<!-- RELATED:START -->

## 相关论文

- [DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [Latent Space Imaging](../../CVPR2025/image_generation/latent_space_imaging.md)
- [Reimagining Parameter Space Exploration with Diffusion Models](reimagining_parameter_space_exploration_with_diffusion_models.md)
- [ETTA: Elucidating the Design Space of Text-to-Audio Models](etta_elucidating_the_design_space_of_text-to-audio_models.md)
- [Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering](../../AAAI2026/image_generation/stabilizing_self-consuming_diffusion_models_with_latent_space_filtering.md)

<!-- RELATED:END -->
