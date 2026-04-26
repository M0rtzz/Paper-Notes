---
title: >-
  [论文解读] Detecting Generated Images by Fitting Natural Image Distributions
description: >-
  [图像生成] 提出一致性验证框架 ConV，利用自然图像流形与生成图像之间的几何差异，通过两个梯度正交的函数实现无需训练的生成图像检测，并引入 Normalizing Flow 增强版 F-ConV 进一步放大流形偏差。
tags:
  - 图像生成
---

# Detecting Generated Images by Fitting Natural Image Distributions

## 基本信息
- **arXiv**: 2511.01293
- **会议**: NeurIPS 2025
- **作者**: Yonggang Zhang, Jun Nie, Xinmei Tian, Mingming Gong, Kun Zhang, Bo Han
- **机构**: HKUST, HKBU, USTC, University of Melbourne, CMU, MBZUAI
- **代码**: https://github.com/tmlr-group/ConV

## 一句话总结
提出一致性验证框架 ConV，利用自然图像流形与生成图像之间的几何差异，通过两个梯度正交的函数实现无需训练的生成图像检测，并引入 Normalizing Flow 增强版 F-ConV 进一步放大流形偏差。

## 背景与动机
生成模型（如 Stable Diffusion、Sora）产出的图像越来越逼真，亟需鲁棒的检测方法。现有方法的核心问题：

1. **依赖二分类器**：需要大量自然图像和生成图像训练，泛化性受限于训练集覆盖的生成模型
2. **跨模型泛化难**：在 Diffusion 模型训练的检测器未必能识别 GAN 或 Sora 生成的图像
3. **持续收集成本高**：每出现新生成模型就需要更新训练数据

**关键洞察**：能否构建一个仅依赖自然图像分布的检测器，而不需要任何生成图像？

## 核心问题
如何利用仅在自然图像上拟合的模型来区分自然图像和生成图像，而无需训练生成图像的检测分类器？

## 方法详解

### 1. 流形视角的动机
- 自然图像位于数据流形 $\mathcal{M}$ 上，生成图像 $\mathbf{x}_g$ 偏离 $\mathcal{M}$
- 生成图像到流形的投影：$\mathbf{x}_{\mathcal{M}}(\mathbf{x}_g) = \arg\min_{\mathbf{x}' \in \mathcal{M}} d(\mathbf{x}', \mathbf{x}_g)$
- 偏差向量 $\mathbf{p} = \mathbf{x}_g - \mathbf{x}_{\mathcal{M}}$ 正交于流形切空间 $\mathcal{T}(\mathbf{x}_{\mathcal{M}})$：
$$\mathbf{v}^\top (\mathbf{x}_{\mathcal{M}}(\mathbf{x}_g) - \mathbf{x}_g) = 0, \quad \mathbf{v} \in \mathcal{T}(\mathbf{x}_{\mathcal{M}})$$

### 2. 一致性验证目标
设计两个函数 $f_1, f_2$，满足：
- **自然图像一致**：$\delta(\mathbf{x}_{\mathcal{M}}) = |f_1(\mathbf{x}_{\mathcal{M}}) - f_2(\mathbf{x}_{\mathcal{M}})| = 0$
- **生成图像不一致**：$\delta(\mathbf{x}_g) > 0$

**正交性原则**（核心设计指导）：
$$\nabla f_1(\mathbf{x}_{\mathcal{M}}) \in \mathcal{O}(\mathbf{x}_{\mathcal{M}}), \quad \nabla f_2(\mathbf{x}_{\mathcal{M}}) \in \mathcal{T}(\mathbf{x}_{\mathcal{M}}), \quad f_1(\mathbf{x}_{\mathcal{M}}) = f_2(\mathbf{x}_{\mathcal{M}})$$

即两个函数的梯度分别落在切空间和法空间中，确保：
$$\delta(\mathbf{x}_g) \geq |\nabla f_1(\mathbf{x}_{\mathcal{M}})^\top \mathbf{p}| > 0 = \delta(\mathbf{x}_{\mathcal{M}})$$

### 3. 无训练实现
- **$f_1$**：预训练自监督模型（如 DINOv2）的损失函数 $\ell(\cdot)$
    - 训练良好的模型对流形上的变换不敏感，故 $\frac{\partial \ell}{\partial \mathbf{x}_{\mathcal{M}}} \perp \mathcal{T}(\mathbf{x}_{\mathcal{M}})$
- **$f_2 = f_1 \circ h$**：将数据变换 $h$（如仿射变换）嵌入 $f_1$ 中
    - $h$ 建模沿流形切空间的变换，其 Jacobian $\mathbf{J}_h$ 张成切空间

最终检测准则：
$$\delta(\mathbf{x}) = |f_1(\mathbf{x}) - f_1(h(\mathbf{x}))| \begin{cases} = 0, & \mathbf{x} \in \mathcal{M} \\ > 0, & \mathbf{x} \notin \mathcal{M} \end{cases}$$

实际实现用特征相似度 $\mathbf{r}^\top \mathbf{r}_h$ 替代损失值，避免负样本计算。

### 4. F-ConV：基于 Normalizing Flow 的流形挤出
当生成模型足够先进，流形偏差变小时，引入 NF 主动放大偏差：
- 用可逆变换 $f$ 将自然图像分布映射到高斯分布：$z = f(v), \; z \sim \mathcal{N}(0, I)$
- 损失函数包含两部分：
$$\mathcal{L} = \underbrace{-\mathbb{E}_{v \sim \mathcal{D}_n} \log p(v) + \mathbb{E}_{v \sim \mathcal{D}_g} \log p(v)}_{\text{Shaping Loss}} \underbrace{- \mathbb{E}_{v \sim \mathcal{D}_n} \cos(f(v), f(T(v))) + \mathbb{E}_{v \sim \mathcal{D}_g} \cos(f(v), f(T(v)))}_{\text{Consistency Loss}}$$
- Shaping Loss 将生成图像推离自然流形；Consistency Loss 放大一致性差异

## 实验关键数据

### ImageNet 检测（AUROC↑ / AP↑，9 种生成模型平均）

| 方法 | 类别 | AUROC | AP |
|------|------|-------|-----|
| CNNspot | 训练型 | 67.04 | 66.78 |
| Ojha | 训练型 | 85.35 | 84.25 |
| NPR | 训练型 | 86.00 | 80.84 |
| FatFormer | 训练型 | 93.68 | 93.11 |
| **F-ConV** | **训练型** | **93.77** | **93.38** |
| AEROBLADA | 免训练 | 57.87 | 57.85 |
| **ConV** | **免训练** | **87.13** | **85.15** |

### Sora / OpenSora 检测

| 方法 | Sora AUROC | OpenSora AUROC |
|------|-----------|---------------|
| CNNspot | 52.85 | 50.14 |
| DRCT | 82.53 | 81.79 |
| FatFormer | 89.95 | 88.76 |
| **F-ConV** | **91.74** | **90.16** |
| **ConV** | **87.74** | **82.84** |

- ConV 作为免训练方法在 Sora 检测上远超大多数训练型方法
- F-ConV 在所有基准上接近或达到 SOTA

## 亮点
1. **理论优雅**：正交性原则提供了清晰的设计指导，将生成图像检测与流形几何联系
2. **无需训练**：ConV 只需一个预训练自监督模型，零训练成本即可部署
3. **对未知模型鲁棒**：不依赖生成图像分布，对 Sora 等新模型仍有效
4. **F-ConV 的流形挤出**：用 NF 主动放大偏差是对抗生成模型进步的前瞻性设计
5. **实用性强**：多次随机变换投票可进一步提升准确率

## 局限性
1. **假设前提**：理论依赖生成图像偏离自然流形——超高质量生成模型可能挑战此假设
2. **F-ConV 仍需少量生成图像**：Shaping Loss 需要生成图像样本，并非完全免训练
3. **变换 $h$ 的选择**：依赖自监督训练中使用的数据增强，鲁棒性可能因增强策略而异
4. **计算开销**：多次变换 + 前向传播的检测速度可能影响实时应用

## 与相关工作的对比
- **vs. Ojha+2023 (CLIP-based)**：同样利用预训练模型特征，但 Ojha 需训练分类器头，ConV 完全免训练
- **vs. DIRE (Diffusion-based)**：DIRE 利用扩散模型重建误差检测，但泛化性差（AUROC ~52%）
- **vs. FatFormer**：训练型 SOTA，ConV 在免训练条件下接近其性能
- **vs. AEROBLADA**：同为免训练，但 AEROBLADA 只利用重建误差，ConV 利用流形几何理论

## 启发与关联
- **自监督模型的"副产品"价值**：DINOv2 等自监督模型对流形变换的不变性本身就是强大的检测信号
- **流形几何 × AI 安全**：正交性原则为生成图像检测提供了优雅的几何解释
- **与对抗样本检测的联系**：对抗样本也偏离自然流形，ConV 的框架可能适用于对抗样本检测

## 评分
- 新颖性：⭐⭐⭐⭐⭐ — 从流形几何推导出免训练检测准则，理论贡献突出
- 技术深度：⭐⭐⭐⭐⭐ — 正交性原则 + NF 挤出 + 免训练实现三位一体
- 实验完整度：⭐⭐⭐⭐⭐ — 9 种生成模型 + Sora + 多个基准全面覆盖
- 写作质量：⭐⭐⭐⭐☆ — 理论推导清晰，但符号较多需仔细阅读

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)

<!-- RELATED:END -->
