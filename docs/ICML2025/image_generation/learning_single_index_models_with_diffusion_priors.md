---
title: >-
  [论文解读] Learning Single Index Models with Diffusion Priors
description: >-
  [ICML2025][图像生成][扩散模型] 提出利用扩散模型先验从半参数单指标模型（SIM）的非线性观测中恢复信号的高效方法，只需一轮无条件采样和部分反演，无需已知链接函数，在1-bit和三次测量上以极少的NFE显著优于现有方法。 传统压缩感知假设线性测量模型 $\boldsymbol{y} = \mathbf{A}\bo…
tags:
  - "ICML2025"
  - "图像生成"
  - "扩散模型"
  - "信号恢复"
  - "单指标模型"
  - "非线性测量"
  - "逆问题"
  - "压缩感知"
---

# Learning Single Index Models with Diffusion Priors

**会议**: ICML2025  
**arXiv**: [2505.21135](https://arxiv.org/abs/2505.21135)  
**代码**: 待确认  
**领域**: 扩散模型理论  
**关键词**: 扩散模型, 信号恢复, 单指标模型, 非线性测量, 逆问题, 压缩感知

## 一句话总结

提出利用扩散模型先验从半参数单指标模型（SIM）的非线性观测中恢复信号的高效方法，只需一轮无条件采样和部分反演，无需已知链接函数，在1-bit和三次测量上以极少的NFE显著优于现有方法。

## 研究背景与动机

传统压缩感知假设线性测量模型 $\boldsymbol{y} = \mathbf{A}\boldsymbol{x}^* + \boldsymbol{e}$，但许多实际问题中测量过程是非线性的。**单指标模型（SIM）** 是最流行的非线性测量模型之一：

$$\boldsymbol{y} = f(\mathbf{A}\boldsymbol{x}^*)$$

其中 $f$ 是未知的、可能不连续的逐元素非线性链接函数。目标是仅利用测量矩阵 $\mathbf{A}$ 和观测 $\boldsymbol{y}$ 恢复信号 $\boldsymbol{x}^*$，而无需知道 $f$。

现有基于扩散模型（DM）的信号恢复工作存在以下局限：

- **DPS、DAPS 等方法**：假设链接函数 $f$ 已知且可微，无法处理不连续函数（如 $\text{sign}(\cdot)$）
- **QCS-SGM**：仅限于量化压缩感知，且重建速度极慢（需上万次 NFE）
- **DDRM、MCG 等**：主要针对线性设置

本文的核心动机是：能否设计一种**不依赖链接函数知识**的高效扩散模型方法来解决 SIM 下的信号恢复问题？

## 方法详解

### 核心思想：将 $\mathbf{A}^T\boldsymbol{y}/m$ 视为带噪信号

论文的关键观察源于以下引理：在 SIM 的温和条件下，

$$\left\|\frac{1}{m}\mathbf{A}^T\boldsymbol{y} - \mu\boldsymbol{x}^*\right\|_\infty \leq \frac{C'\sqrt{\log(2n)}}{\sqrt{m}}$$

其中 $\mu = \mathbb{E}[f(\boldsymbol{a}^T\boldsymbol{x}^*)\boldsymbol{a}^T\boldsymbol{x}^*]$。这表明 $\mathbf{A}^T\boldsymbol{y}/m$ 本质上是 $\mu\boldsymbol{x}^*$ 的带噪版本，噪声水平与 $1/\sqrt{m}$ 成正比。

### 三种方法对比

论文提出三种策略，核心区别在于如何利用扩散模型的采样 $G$ 和反演 $G^\dagger$：

| 方法 | 公式 | 操作 |
|------|------|------|
| **SIM-DMFIS** | $\hat{\boldsymbol{x}} = G \circ G^\dagger(\mathbf{A}^T\boldsymbol{y}/m)$ | 从 $\epsilon$ 做完整反演再完整采样 |
| **SIM-DMS** | $\hat{\boldsymbol{x}} = G_{t^*}(\alpha_{t^*}C_s'\mathbf{A}^T\boldsymbol{y}/m)$ | 仅从 $t^*$ 做部分采样（去噪） |
| **SIM-DMIS** ⭐ | $\hat{\boldsymbol{x}} = G \circ G^\dagger_{t^*}(\alpha_{t^*}C_s'\mathbf{A}^T\boldsymbol{y}/m)$ | 从 $t^*$ 做部分反演到 $T$，再完整采样 |

### 中间时刻 $t^*$ 的确定

通过将 $\mathbf{A}^T\boldsymbol{y}/m$ 的噪声水平与扩散前向过程的噪声调度匹配，选取中间时刻 $t^*$ 满足：

$$\frac{\sigma_{t^*}}{\alpha_{t^*}} = \frac{C_s}{\sqrt{m}}$$

其中 $C_s$ 为可调参数。这是理论驱动的设计：噪声越大（$m$ 越小），反演起点越靠近 $T$。

### 算法流程（SIM-DMIS）

1. **输入**：测量矩阵 $\mathbf{A}$、观测 $\boldsymbol{y}$、预训练 DM 的数据预测网络 $\boldsymbol{x}_\theta$
2. 根据 $C_s/\sqrt{m}$ 计算中间时刻 $t^*$
3. 构造初始向量 $\alpha_{t^*}C_s'\mathbf{A}^T\boldsymbol{y}/m$
4. 从 $t^*$ 到 $T$ 执行部分反演 $G^\dagger_{t^*}$（使用 DM2M 二阶反演方法）
5. 从 $T$ 到 $\epsilon$ 执行完整采样 $G$（使用 DDIM 采样）
6. **输出**：重建信号 $\hat{\boldsymbol{x}}$

## 理论分析

| 定理/引理 | 内容 | 意义 |
|-----------|------|------|
| Lemma 2 | $\|\mathbf{A}^T\boldsymbol{y}/m - \mu\boldsymbol{x}^*\|_\infty = O(\sqrt{\log n/m})$ | 建立噪声水平估计，指导 $t^*$ 选取 |
| Lemma 3 | 生成器 $G$ 在 Lipschitz 条件下 $L$-Lipschitz 连续 | 保证误差不被采样过程放大 |
| Theorem 3 | $\|\bar{\boldsymbol{x}}_\epsilon - G \circ G^\dagger_t(\bar{\boldsymbol{x}}_t)\|_2 = O(\sqrt{n}(h_{\max}^{k_2} + Lh_{\max}^{k_1}))$ | SIM-DMIS 的误差上界，与步长 $h_{\max}$ 和数值阶数 $k_1, k_2$ 相关 |
| Assumption 1 | 数据预测网络 $\boldsymbol{x}_\theta(\cdot, t)$ 关于第一个参数 $L_t$-Lipschitz | 标准假设，被大量 DM 理论工作采用 |

理论表明：使用高阶数值方法（$k_1, k_2 \geq 2$）可显著降低重建误差。

## 实验关键数据

### FFHQ 256×256, 1-bit 测量 ($m = n/8$)

| 方法 | NFE | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----|--------|--------|---------|
| QCS-SGM | 11555 | 12.91 | 0.51 | 0.50 |
| DPS-N | 1000 | 11.14 | 0.37 | 0.69 |
| **SIM-DMS** | **50** | — | — | — |
| **SIM-DMIS** | **150** | **最优** | **最优** | **最优** |

### 关键实验发现

- SIM-DMIS 只需 **150 NFE** 即超越需 **11555 NFE** 的 QCS-SGM，效率提升 **77倍**
- 在1-bit测量中，即使 DPS-N 和 DAPS-N 利用了链接函数 $f$ 的知识，SIM-DMIS 仍然更优
- 部分反演（SIM-DMIS）显著优于完整反演（SIM-DMFIS），验证了从中间时刻 $t^*$ 启动反演的理论直觉
- 在 FFHQ 和 ImageNet（CIFAR-10 见附录）上均表现一致

## 亮点与洞察

1. **不需要链接函数知识**：这是核心优势。现实中非线性测量模型的链接函数往往未知或不可微，本方法完全绕开了这一限制
2. **理论驱动的中间时刻选取**：通过 Lemma 2 将 $\mathbf{A}^T\boldsymbol{y}/m$ 的噪声水平与扩散噪声调度 $\sigma_t/\alpha_t$ 对齐，优雅地确定反演起始点 $t^*$
3. **极高的计算效率**：仅需一轮采样+部分反演（150 NFE），无需迭代优化或梯度计算
4. **部分反演优于完整反演**的反直觉发现：从 $\epsilon$ 开始做完整反演会错误地假设输入符合数据分布 $q_0$，而从匹配噪声水平的 $t^*$ 开始更合理
5. **统一框架**处理不同非线性测量（1-bit、三次、量化等），无需针对每种测量单独设计

## 局限与展望

1. **不适用于相位恢复**：条件 $\mu \neq 0$ 排除了 $f(x) = x^2$ 或 $f(x) = |x|$ 的情况
2. **调参依赖**：$C_s$ 和 $C_s'$ 需要针对不同测量模型和数据集调参
3. **理论与实践的 gap**：Theorem 3 的误差界依赖于 Lipschitz 常数 $L$，而实际 DM 的 $L$ 值可能很大
4. **矩阵存储开销**：需要显式存储 $m \times n$ 的测量矩阵 $\mathbf{A}$，对高分辨率图像不友好
5. **未探索结构化测量矩阵**：仅考虑 i.i.d. 高斯测量，实际应用中测量矩阵通常有结构

## 相关工作与启发

- **DPS** (Chung et al., 2023)：基于后验采样的信号恢复，需已知前向模型
- **DAPS** (Zhang et al., 2024)：扩展 DPS 到非线性设置，但仍需 $f$ 可微
- **QCS-SGM** (Meng & Kabashima, 2022)：用 SGM 做量化压缩感知，但需上万 NFE
- **CSGM** (Bora et al., 2017)：开创性地用生成模型先验做信号恢复
- 本文方法可启发其他逆问题：只要能将观测表示为信号的带噪版本，就可利用扩散模型的部分反演+采样框架

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从噪声水平匹配角度确定反演起点的思路新颖且理论扎实
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多测量模型、多基线对比，附录含丰富消融
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，符号规范，三种方法的对比图示直观
- 价值: ⭐⭐⭐⭐ — 为非线性逆问题提供了高效且通用的扩散模型解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Visual Generative Priors without Text](../../CVPR2025/image_generation/learning_visual_generative_priors_without_text.md)
- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](../../CVPR2025/image_generation/ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](../../CVPR2025/image_generation/difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[CVPR 2025\] Nested Diffusion Models Using Hierarchical Latent Priors](../../CVPR2025/image_generation/nested_diffusion_models_using_hierarchical_latent_priors.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)

</div>

<!-- RELATED:END -->
