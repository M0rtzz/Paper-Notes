---
title: >-
  [论文解读] Image Super-Resolution with Guarantees via Conformalized Generative Models
description: >-
  [NEURIPS2025][图像生成][超分辨率] 基于共形预测（Conformal Prediction）技术，为生成式图像超分辨率模型构建二值"置信度掩码"，能可靠地标识生成图像中可信赖的区域，并提供严格的统计保证。
tags:
  - NEURIPS2025
  - 图像生成
  - 超分辨率
  - 不确定性量化
  - 共形预测
  - 扩散模型
  - 置信度掩码
---

# Image Super-Resolution with Guarantees via Conformalized Generative Models

**会议**: NEURIPS2025  
**arXiv**: [2502.09664](https://arxiv.org/abs/2502.09664)  
**代码**: [adamesalles/experiments-conformal-superres](https://github.com/adamesalles/experiments-conformal-superres)  
**领域**: image_generation  
**关键词**: 超分辨率, 不确定性量化, 共形预测, 扩散模型, 置信度掩码

## 一句话总结

基于共形预测（Conformal Prediction）技术，为生成式图像超分辨率模型构建二值"置信度掩码"，能可靠地标识生成图像中可信赖的区域，并提供严格的统计保证。

## 背景与动机

- 生成式基础模型（尤其是扩散模型）在图像超分辨率任务上取得了巨大进步，但其预测的可信度缺乏量化手段——模型可能在某些区域产生幻觉（hallucination）
- 现有不确定性量化方法的两大痛点：
    1. **可解释性差**：如 Angelopoulos et al. (2022b) 生成区间值图像（每个像素用一个置信区间表示），用户难以直觉理解
    2. **缺乏概率保证**：如 Kutiel et al. (2023) 虽产生连续置信度分数，但没有严格的统计有效性证明
- 实际部署场景（消费设备、医学影像等）迫切需要一种**既直观可解释又有理论保证**的不确定性量化框架

## 核心问题

如何在不依赖生成模型内部结构的前提下（黑盒/API模式亦可），为超分辨率生成图像的每个像素判定"可信/不可信"，同时提供可控的保真度误差保证？

## 方法详解

### 2.1 共形掩码校准（Conformal Mask Calibration）

**输入**：任意黑盒超分辨率模型 $\mu$，模型不确定性估计函数 $\sigma$，以及 $n$ 张未标注的高分辨率校准图像 $(X_i, Y_i)_{i=1}^n$。

**核心思路**：通过阈值化 $\sigma$ 的输出来构建二值掩码 $M_\alpha(X)$，掩码中标记为"可信"的像素满足不确定性低于阈值 $t_\alpha$。

**阈值校准公式**：

$$t_\alpha = \sup\left\{t \in \mathbb{R} \cup \{+\infty\}: \frac{1}{n+1}\sum_{i=1}^n \sup_{p;[\sigma(X_i)]_p \le t} D_p(Y_i, \mu(X_i)) + \frac{3}{n+1} \le \alpha \right\}$$

其中 $D_p$ 为局部图像相似度度量，$\alpha$ 为期望的保真度水平。

**生成掩码**：$M_\alpha(X) = \{p : [\sigma(X)]_p \le t_\alpha\}$

**核心定理（Theorem 2.1）**：若校准数据与测试样本 i.i.d.，则：

$$\mathbb{E}\left[\sup_{p \in M_\alpha(X_{n+1})} D_p(\mu(X_{n+1}), Y_{n+1})\right] \le \alpha$$

即掩码内可信区域的最大保真度误差在期望下不超过 $\alpha$。

**计算效率**：利用动态规划可在 $O(nd\log d)$ 时间内求解 $t_\alpha$（$n$ 为校准图像数量，$d$ 为像素数），远优于暴力搜索的 $\Omega(n^2 d^2)$。

### 2.2 不确定性分数掩码（Score Mask）的构造

好的 $\sigma$ 应在不确定性大的区域取较高值。本文提出三种策略：

1. **逐像素方差** $\sigma^{\text{var}}$：对同一低分辨率图生成 $M$ 次超分辨率结果，取每个像素的经验方差。缺点是过于局部，边缘微小偏移会导致方差虚高
2. **基于低通滤波的Patch方差**：将卷积核 $K$ 应用于方差分解的二阶矩计算中。$K$ 为 1-box 核时退化为逐像素方差。关键公式：$[\sigma^K(X)]_p = \hat{\mathbb{E}}_M[[\mu(X)^2 * K]_p] - (\hat{\mathbb{E}}_M[[\mu(X) * K]_p])^2$
3. **额外高斯模糊**：在 Patch 方差之上再施加高斯模糊，抑制边缘伪影被过度强调的问题

### 2.3 局部相似度度量 $D_p$ 的选择

论文探索了三种度量，均需满足 $0 \le D_p \le 3$：

| 度量 | 定义 | 特点 |
|------|------|------|
| 逐点度量 | $D_p = \|[Y]_p - [\hat{Y}]_p\|_1$ | 最简单但对单像素误差敏感 |
| 邻域平均度量 | $D_p = \|[Y*K]_p - [\hat{Y}*K]_p\|_1$ | 引入空间上下文，掩码更大更稳定 |
| 语义度量 | $D_p = [S(Y, \hat{Y})]_p$（人工标注差异） | 最能捕捉语义差异，但需要人工标注 |

所有比较均在 Lab 色彩空间中进行，以保证感知均匀性。

## 额外理论保证

### PSNR 控制（Proposition 3.1）

方法不仅控制自定义的保真度误差，还可推导出对 PSNR 的下界保证：

$$\mathbb{E}[\text{PSNR}(\mu(X_{n+1}), Y_{n+1} | M_\alpha(X_{n+1}))] \ge -20\log_{10}\alpha$$

例如 $\alpha = 0.1$ 时，PSNR 下界为 20 dB。

### 数据泄漏鲁棒性（Proposition 3.2）

当 $n$ 个校准样本中有 $n_{\text{leaked}}$ 个实际来自训练数据时，保真度误差上界放大为：

$$\text{Fidelity Error} \le \alpha \cdot \frac{n_{\text{new}} + n_{\text{leaked}} + 1}{n_{\text{new}} + 1}$$

当泄漏比例不大时，保证仅轻微退化——这对使用大规模预训练基础模型的场景尤为重要。

## 实验关键数据

**数据集**：Liu4K（1600 训练 + 400 验证），4K 高分辨率真实图像。
**基础模型**：SinSR（基于扩散的单步超分辨率方法）。
**硬件**：Intel Xeon E5-2696 v2 + NVIDIA RTX 6000 Ada 48GB。

| 保真度 $\alpha$ | 语义 $D_p$ PSNR | 语义掩码大小 | 非语义 $D_p$ PSNR | 非语义掩码大小 |
|:---:|:---:|:---:|:---:|:---:|
| 0.075 | 32.75 ± 1.55 | 0.77 ± 0.07 | 30.23 ± 1.12 | 0.43 ± 0.09 |
| 0.100 | 32.65 ± 1.48 | 0.73 ± 0.07 | 28.64 ± 0.93 | 0.23 ± 0.06 |
| 0.200 | 31.63 ± 1.32 | 0.60 ± 0.08 | 26.82 ± 1.03 | 0.00 ± 0.00 |
| 无方法 | 26.83 ± 1.06 | N/A | 26.82 ± 1.08 | N/A |

**关键发现**：
- 保真度误差被严格控制在 $\alpha$ 以下，实验值与理论上界几乎吻合
- 语义度量 $D_p$ 下的掩码明显大于非语义度量——说明加入语义信息可提升可信区域覆盖
- 即使在基础模型失败（如误判模糊区域为高频缺失而产生幻觉）的情况下，置信度掩码仍能准确识别不可靠区域

## 亮点

1. **完全黑盒兼容**：仅需调用超分辨率模型的输入输出接口，不要求访问模型权重或中间特征，甚至支持 API-only 模型
2. **严格的统计保证体系**：从保真度误差控制（Theorem 2.1）到 PSNR 下界（Proposition 3.1）再到数据泄漏鲁棒性（Proposition 3.2），理论链完整
3. **高效动态规划校准**：$O(nd\log d)$ 复杂度使得在大规模数据上校准仍然可行
4. **灵活的度量设计**：$D_p$ 可从逐像素到语义级别自由选择，适配不同应用场景
5. **超越超分辨率**：附录展示了直接迁移到图像着色任务的成功案例，暗示框架的广泛适用性
6. **对 Kutiel et al. 的反例**：附录 B 严格证明了 Kutiel et al. 的方法实际上不满足其声称的统计保证

## 局限与展望

1. **可交换性假设**：核心定理依赖校准数据与测试数据的 i.i.d.（或可交换）假设，显著的分布偏移可能使保证失效
2. **计算开销**：需要多次调用生成模型来估计 $\sigma$（像素方差），增加推理时间
3. **$\sigma$ 与 $\mu$ 解耦**：目前不确定性估计与超分辨率图像的生成是分离的，联合估计可能进一步提升效果
4. **语义度量的标注成本**：使用语义 $D_p$ 需要人工标注差异区域，限制了规模化应用
5. **PSNR 下界较松**：Proposition 3.1 的理论下界在较大 $\alpha$ 时与实际值差距明显

## 与相关工作的对比

| 方法 | 输出形式 | 统计保证 | 可解释性 | 模型要求 |
|------|---------|---------|---------|---------|
| **本文** | 二值置信度掩码 | ✅ 保真度误差 + PSNR + 数据泄漏鲁棒性 | ⭐⭐⭐ 直观 | 黑盒 |
| Angelopoulos et al. (2022b) | 区间值图像 | ✅ 像素区间覆盖 | ⭐ 难以解读 | 黑盒 |
| Kutiel et al. (2023) | 连续置信度分数 | ❌（本文附录反例证伪） | ⭐⭐ 尚可 | 需内部访问 |
| BNN / MC Dropout | 方差图 | ❌ 无形式保证 | ⭐⭐ 连续值 | 需模型结构 |

## 启发与关联

- **共形预测在视觉中的新范式**：将共形预测从分类/语义分割扩展到像素级生成质量评估，提供了一种通用的后验不确定性量化模板
- **可信AI部署**：在医学影像超分辨率等高风险场景中，这种有统计保证的置信度掩码比传统热力图更具实用价值
- **与超分辨率以外任务的联系**：框架天然适用于任何"低质量→高质量"的图像修复任务（去噪、着色、修复），附录实验已验证着色任务的可行性
- **低通滤波的双重作用**：既用于改善 $\sigma$ 的估计（减少边缘方差虚高），又用于放松 $D_p$ 的局部性（获得更大掩码），设计思路值得在其他像素级评估场景中借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ (将共形风险控制引入图像超分辨率得置信度掩码，方法新颖且理论扎实)
- 实验充分度: ⭐⭐⭐⭐ (Liu4K 数据集 + 分布偏移 + 数据泄漏 + 着色迁移，覆盖全面)
- 写作质量: ⭐⭐⭐⭐⭐ (定理陈述清晰，实验可视化直观，附录详尽)
- 价值: ⭐⭐⭐⭐ (为生成模型的可信部署提供了实用工具，但计算开销和 i.i.d. 假设限制了直接应用)

<!-- RELATED:START -->

## 相关论文

- [FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](../../CVPR2025/image_generation/faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](../../CVPR2025/image_generation/uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](../../CVPR2026/image_generation/oars_processaware_online_alignment_for_generative.md)
- [PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](../../CVPR2025/image_generation/arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)

<!-- RELATED:END -->
