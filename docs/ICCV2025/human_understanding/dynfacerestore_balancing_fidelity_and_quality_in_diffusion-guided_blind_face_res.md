---
title: >-
  [论文解读] DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration
description: >-
  [ICCV 2025][人体理解][盲人脸修复] 提出 DynFaceRestore，通过动态模糊等级映射（DBLM）将盲退化转化为高斯去模糊问题，结合动态起始步查找表（DSST）和区域自适应引导缩放器（DGSA），在扩散模型采样中实现保真度与感知质量的最优平衡。
tags:
  - ICCV 2025
  - 人体理解
  - 盲人脸修复
  - 扩散模型引导
  - 动态模糊映射
  - 保真度-质量平衡
  - 区域自适应引导
---

# DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration

**会议**: ICCV 2025  
**arXiv**: [2507.13797](https://arxiv.org/abs/2507.13797)  
**代码**: [项目页面](https://github.com/)  
**领域**: 扩散模型/人脸修复  
**关键词**: 盲人脸修复, 扩散模型引导, 动态模糊映射, 保真度-质量平衡, 区域自适应引导

## 一句话总结

提出 DynFaceRestore，通过动态模糊等级映射（DBLM）将盲退化转化为高斯去模糊问题，结合动态起始步查找表（DSST）和区域自适应引导缩放器（DGSA），在扩散模型采样中实现保真度与感知质量的最优平衡。

## 研究背景与动机

盲人脸修复（BFR）旨在从未知退化源的低质量面部图像中恢复高保真、细节丰富的面部图像，其核心难点在于同时增强面部细节和保持身份一致性。预训练扩散模型已被广泛用作图像先验来生成精细细节，但现有方法存在三个关键问题：

**固定扩散起始步问题**：现有方法（如 DifFace）假设所有低质量输入的退化程度相同，使用固定的扩散采样起始时间步。这导致对严重退化的图像产生"欠扩散"（细节不足），对轻度退化的图像产生"过扩散"（引入伪影），如论文 Fig. 2 中 t-SNE 可视化所清晰展示的那样。

**核不匹配问题**：在盲设定下，退化核的估计往往不精确，且实际退化核形式复杂多样（混合了模糊、下采样、噪声、JPEG 压缩等），将其简单估计为单一核会导致扩散采样过程中的引导偏差，降低修复保真度。

**全局引导缩放问题**：现有结合引导的方法（如 DPS、PGDiff）对所有像素使用统一的引导缩放因子。然而高频区域（头发、皱纹）需要更强的扩散模型影响来增强感知质量，而低频区域（面部轮廓）需要更强的观测引导来保持结构保真度。全局缩放无法平衡这一矛盾。

## 方法详解

### 整体框架

DynFaceRestore 框架包含三个核心组件，将 BFR 问题重新表述为高斯去模糊问题：
1. **DBLM**（动态模糊等级映射）：将未知退化输入转化为高斯模糊图像
2. **DSST**（动态起始步查找表）：根据模糊等级确定最优扩散起始步
3. **DGSA**（动态引导缩放调整器）：区域自适应地调整引导强度

引导扩散采样过程为：
$$x_{t-1} = x'_{t-1} - A_t \times \nabla_{x_t} \| \acute{y} - k_t \otimes x^0_t \|^2$$

### 关键设计

1. **动态模糊等级映射（DBLM）**：将盲退化输入 $y$ 转化为高斯模糊形式 $\acute{y} = k^{\hat{std^*}}_y \otimes RM(y)$，其中 $RM$ 可以是任何预训练的修复模型（本文使用 SwinIR）。核心思想是：不试图完美估计复杂的退化核，而是将其转化为已知形式的高斯核，从而在扩散采样中提供可靠引导。最优标准差 $std^*$ 通过 SE 网络估计，该网络包含 Transfer Model（TM）和 Standard Deviation Estimator（SDE）两个子模块。DBLM 的关键优势是：即使 $RM$ 的修复不完美，通过再施加高斯模糊，可以将误差限制在可控范围内，有效缓解核不匹配问题。

2. **动态起始步查找表（DSST）**：基于关键观察——高质量图像 $x_0$ 和其模糊版本 $\tilde{y}^{std}_0$ 经前向扩散后会在某个时间步 $t$ 统计收敛。该收敛点即为模糊观测 $\acute{y}$ 的最佳引导插入点。公式为：

   $$t_{std} = \underset{t}{argmin} \left( log(\mathbf{X}_t) - log(\tilde{\mathbf{Y}}^{std}_t) \leq tol \right)$$

   预计算不同 $std$ 对应的最优起始步，存储为查找表。推理时根据 SE 估计的 $\hat{std^*}$ 直接查表获取 $t_{start}$，避免欠扩散和过扩散。实验表明采样步骤范围从固定的 1000 缩减到 [690, 925]。

3. **动态引导缩放调整器（DGSA）**：一个轻量 CNN（3 层卷积），输出范围 $[0,1]$ 的区域引导缩放图 $A_t$。输入为当前测量 $\acute{y}$、高质量预测 $x^0_t$ 和时间步 $t$。在高频纹理区域（头发、皱纹）输出较小的 $A_t$（减弱引导，让扩散模型自由生成细节），在低频结构区域（轮廓、皮肤）输出较大的 $A_t$（增强引导，保持保真度）。训练使用平稳小波变换分频带监督和 DISTS 感知损失：

   $$L_{DGSA} = \sum_i \gamma_i \mathbb{D}(SWT(x^0_{t-1})_i, SWT(x_0)_i) + DISTS(x^0_{t-1}, x_0)$$

### 损失函数 / 训练策略

- **SE 网络**：先预训练 SDE 估计高斯模糊等级，再端到端训练 SE = TM + SDE
- **DGSA**：随机采样时间步 $t$，使用 SWT 分频带 L1 损失 + DISTS 感知损失
- **核自适应更新**：采样过程中同步更新估计核 $std_{t-1} = std_t - s \nabla_{std_t} \| \acute{y} - k_t \otimes x^0_t \|^2$
- **多引导扩展**：可选地生成 3 个不同模糊等级的引导（$\hat{std^*}$, $\hat{std^*}-1$, $\hat{std^*}-2$），加权组合以平衡 DM 质量和 RM 保真度
- 预训练扩散模型与 DifFace/PGDiff 使用同一模型，训练在 FFHQ 数据集上

## 实验关键数据

### 主实验

**CelebA-Test 定量对比**：

| 类型 | 方法 | PSNR↑ | SSIM↑ | FID↓ | IDA↓ | LMD↓ |
|------|------|-------|-------|------|------|------|
| GAN | GPEN | 23.77 | 0.659 | 30.25 | 0.837 | 6.377 |
| GAN | GFP-GAN | 22.84 | 0.620 | 23.86 | 0.822 | 4.793 |
| Codebook | CodeFormer | 23.83 | 0.637 | 18.08 | 0.775 | 3.509 |
| DM | DifFace | 23.95 | 0.659 | 15.03 | 0.867 | 3.781 |
| DM | DiffBIR | 24.13 | 0.647 | 19.19 | 0.767 | 3.535 |
| DM | 3Diffusion | 23.39 | 0.651 | 15.45 | 0.943 | 3.781 |
| **DM** | **DynFaceRestore** | **24.35** | **0.664** | **14.78** | **0.748** | **3.419** |

**真实世界数据集 FID 对比**：

| 方法 | LFW↓ | WebPhoto↓ | Wider↓ |
|------|------|----------|--------|
| CodeFormer | 52.35 | 83.19 | 38.80 |
| DAEFR | 47.53 | 75.45 | 36.72 |
| DiffBIR | 43.45 | 91.20 | 36.72 |
| **DynFaceRestore** | **42.52** | 95.32 | **36.05** |

### 消融实验

**各组件效果消融（CelebA-Test）**：

| 设置 | DBLM | 多引导 | DSST | DGSA | PSNR↑ | FID↓ | IDA↓ | 采样步范围 |
|------|------|--------|------|------|-------|------|------|----------|
| Baseline | | | | | 11.13 | 55.78 | 1.461 | 1000 |
| A | ✓ | | | | 24.99 | 18.30 | 0.725 | 1000 |
| C | ✓ | ✓ | ✓ | | 25.11 | 19.79 | 0.724 | [690,925] |
| E | ✓ | | ✓ | ✓ | 24.33 | **14.69** | 0.755 | [690,925] |
| **F** | **✓** | **✓** | **✓** | **✓** | **24.35** | 14.78 | **0.748** | [690,925] |

### 关键发现

- DBLM 是性能提升的最关键组件：从 baseline 的 PSNR 11.13 跃升至 24.99
- DGSA 显著提升感知质量（FID 从 19.79 降到 14.78），代价是 PSNR 略降
- DSST 在提升指标的同时缩短了采样范围（1000 → [690,925]），既提质又提速
- 方法在多个保真度指标（PSNR/SSIM/IDA/LMD）和感知质量（FID）上实现了同时领先，成功平衡了二者的权衡

## 亮点与洞察

- **问题重新表述的巧妙性**：将复杂的盲修复问题转化为可控的高斯去模糊问题，是整个方法的核心洞察。通过引入中间的高斯模糊表示，既保留了可靠的低频信息，又为扩散模型提供了明确的引导形式。
- **动态 vs 静态的全面体现**：三个组件（DBLM、DSST、DGSA）分别从退化映射、时间步选择、引导强度三个维度实现了"动态"调整，系统性地解决了现有方法的"一刀切"问题。
- **区域自适应引导**：DGSA 的设计抓住了保真度和质量在空间上分布不均匀这一关键特点，是对 DPS 引导的有效改进。

## 局限与展望

- 推理时间较长（91.82s），远高于 CodeFormer（0.06s），限制了实时应用
- WebPhoto 数据集上 FID 表现不如部分方法
- 多引导的超参数（引导数量、$std$ 间距）需手动设定，缺乏自适应机制
- DGSA 作为单独训练的网络，其泛化性能有待验证
- 可探索将核更新过程（Eq. 7）与 DGSA 统一优化

## 相关工作与启发

- DPS（Diffusion Posterior Sampling）提供了理论基础，本文在其上实现了三项关键扩展
- DifFace 的固定步策略的局限性直接激发了 DSST 的设计
- SwinIR 作为修复模型的选择体现了"不重复造轮子"的工程思想

## 评分

- **新颖性**: ⭐⭐⭐⭐ 问题重新表述巧妙，三个动态组件设计合理且相互协作
- **实验充分度**: ⭐⭐⭐⭐⭐ 合成和真实数据集均有评估，消融研究详尽
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，公式推导严谨
- **价值**: ⭐⭐⭐⭐ 对扩散引导修复领域有重要贡献，但推理速度限制了实际部署

<!-- RELATED:START -->

## 相关论文

- [A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](a_quality-guided_mixture_of_score-fusion_experts_framework_for_human_recognition.md)
- [Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](../../CVPR2025/human_understanding/optimal_transport-guided_source-free_adaptation_for_face_anti-spoofing.md)
- [Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography](degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)
- [Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity](hineus_high-fidelity_neural_surface_mitigating_low-texture_and_reflective_ambigu.md)

<!-- RELATED:END -->
