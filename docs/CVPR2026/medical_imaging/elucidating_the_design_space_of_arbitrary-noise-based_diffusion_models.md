---
title: >-
  [论文解读] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models (EDA)
description: >-
  [CVPR2026][医学图像][扩散模型] 提出 EDA 框架，将 EDM 的设计空间从高斯噪声扩展到任意噪声模式，通过多元高斯分布参数化协方差矩阵实现灵活的噪声扩散，在 MRI 偏置场校正、CT 金属伪影去除和自然图像阴影去除三个任务上仅用 5 步采样即达到或超越 100 步 EDM 方法和专用方法。
tags:
  - CVPR2026
  - 医学图像
  - 扩散模型
  - 任意噪声
  - 设计空间
  - 图像复原
  - MRI偏置场校正
  - CT金属伪影去除
  - 阴影去除
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models (EDA)

**会议**: CVPR2026  
**arXiv**: [2507.18534](https://arxiv.org/abs/2507.18534)  
**代码**: [PerceptionComputingLab/EDA](https://github.com/PerceptionComputingLab/EDA)  
**领域**: 医学图像 / 图像复原  
**关键词**: 扩散模型, 任意噪声, 设计空间, 图像复原, MRI偏置场校正, CT金属伪影去除, 阴影去除

## 一句话总结

提出 EDA 框架，将 EDM 的设计空间从高斯噪声扩展到任意噪声模式，通过多元高斯分布参数化协方差矩阵实现灵活的噪声扩散，在 MRI 偏置场校正、CT 金属伪影去除和自然图像阴影去除三个任务上仅用 5 步采样即达到或超越 100 步 EDM 方法和专用方法。

## 研究背景与动机

**EDM 局限于高斯噪声**：EDM 统一了大多数扩散模型的设计空间，但其前向过程仅支持逐像素独立的高斯噪声（协方差为 $\sigma^2(t)\mathbf{I}$），无法涵盖 Flow Matching 等支持任意噪声扩散的新方法。

**强制注入高斯噪声损害图像复原**：在复原任务中，EDM 方法需要在退化图像上额外叠加高斯噪声才能启动逆过程，这破坏了退化图像中的任务特定信息。

**复原距离被人为拉长**：高斯噪声的注入使复原起点偏离退化图像分布，增加了复原距离和任务复杂度，导致需要更多采样步数。

**SDE 框架优于 ODE**：虽然 Flow Matching 提供了灵活的 ODE 扩散框架打破了高斯噪声限制，但 SDE 方法在结果多样性和质量上表现更优。

**缺乏统一 SDE 设计空间**：目前没有一个既支持灵活噪声模式又保留 SDE 优势的统一设计空间，阻碍了扩散模型的理论发展。

**直接从退化图像启动的需求**：理想情况下，可以定制噪声模式使逆过程直接从已知退化图像启动，缩短复原距离、降低复杂度。

## 方法详解

### 整体框架

EDA（Elucidating the Design space of Arbitrary-noise diffusion models）通过多元高斯分布刻画扩散过程，将 EDM 中的对角协方差 $\sigma^2(t)\mathbf{I}$ 推广为由任意基函数集合定义的协方差矩阵 $\boldsymbol{\Sigma}_{x_0} = H_{x_0}H_{x_0}^\top$，从而支持任意噪声模式的扩散和去除。

### 关键设计

**广义前向过程**：EDA 的扩散噪声定义为

$$N = \sum_{m=1}^{M} \frac{\eta + \epsilon_m}{\eta + 1} h_{m, x_0}$$

其中 $H_{x_0} = [h_{1,x_0}, \ldots, h_{M,x_0}]$ 是调节噪声模式的基函数集合，$\epsilon_m \sim \mathcal{N}(0,1)$ 为独立高斯变量，$\eta \geq 0$ 控制噪声随机性（$\eta=0$ 最大随机性，$\eta \to \infty$ 趋向确定性）。

**多 Wiener 过程 SDE**：前向过程由多个独立 Wiener 过程驱动：

$$\mathrm{d}\boldsymbol{x} = [f(t)\boldsymbol{x} + \phi_{x_0}(t)]\mathrm{d}t + g(t)\sum_{m=1}^{M} h_{m,x_0} \mathrm{d}\omega_t^{(m)}$$

**关键理论结果**：
- **Proposition 1**：EDA 支持任意噪声的扩散和去除，通过三种配置覆盖所有场景——统一基集（最优情况）、样本依赖基函数（通用情况）、非高斯噪声离散采样
- **Proposition 2**：从简单高斯噪声推广到复杂任意模式不引入额外计算开销——PFODE 求解后额外项可解析化简消除，最终确定性采样公式与 EDM 完全一致
- **Proposition 3**：EDM 是 EDA 的特殊情况（$\eta=0$，基集取像素级单位矩阵）

### 损失函数

采用与 EDM 相同的去噪器训练目标：

$$\mathcal{L} = \mathbb{E}_{x_0 \sim P_{\text{data}}} \mathbb{E}_{x \sim P(x_t | y)} \| D_\theta(x; \sigma) - x_0 \|^2$$

去噪器结构保持 EDM 的 skip-connection 形式 $D_\theta(x; \sigma) = c_{\text{skip}}(\sigma)x + c_{\text{out}}(\sigma)F_\theta(c_{\text{in}}(\sigma)x; c_{\text{noise}}(\sigma))$，网络 $F_\theta$ 预测扩散噪声。采样使用 Euler 一阶方法的确定性采样。

## 实验

### 实验设置

- **框架**：PyTorch，单卡 NVIDIA RTX 3090
- **参数**：$s(t)=1$，$\sigma = \sqrt{1-\bar{\alpha_t}}$，训练总步数 $T=100$
- **三个任务**：MRI 偏置场校正（HCP 数据集，2206/1000 训练/测试切片）、CT 金属伪影去除（DeepLesion，1000/200 训练/测试图像）、自然图像阴影去除（ISTD，1330/540 训练/测试图像）

### MRI 偏置场校正

| 方法 | SSIM ↑ | PSNR ↑ | COCO ↑ | CV(WM) ↓ |
|------|--------|--------|--------|----------|
| N4 | 0.95 | 25.62 | 0.95 | 7.95 |
| ABCNet | 0.98 | 29.58 | 0.97 | 7.69 |
| Refusion (100步) | 0.98 | 34.67 | 0.98 | 7.72 |
| **EDA (5步)** | **0.99** | **38.02** | **0.99** | **7.40** |

### 阴影去除 (ISTD)

| 方法 | ALL PSNR ↑ | ALL SSIM ↑ | NS PSNR ↑ | NS RMSE ↓ |
|------|-----------|-----------|----------|----------|
| ShadowFormer | 31.81 | 0.967 | 33.89 | 3.90 |
| Refusion | 27.23 | 0.882 | 28.64 | 6.99 |
| **EDA** | **32.01** | **0.968** | **34.31** | **3.77** |

### 消融与关键发现

- **采样步数效率**：EDA 仅用 5 步采样即达到或超越 Refusion 100 步的效果，速度加速约 **53 倍**（BFC 任务 0.182 vs 9.665 sec/slice）
- **ODE vs SDE**：MeanFlow（ODE）在所有三个复原任务上显著落后，因为 ODE 产生平均化解而非高保真复原——BFC 中 CV(GM) 最高（15.49），MAR 中伪影区域模糊，SR 中 ALL RMSE 高达 9.77
- **仅图像域 vs 双域**：在 CT MAR 任务中，EDA 仅使用图像域就超越了部分双域方法（LI、CNNMAR、DSCMAR 等），但与 SOTA 双域方法（InDuDoNet+、DICDNet）尚存差距
- **非阴影区域保真度**：在阴影去除中 EDA 的非阴影区域 PSNR 达到 34.31 dB，RMSE 仅 3.77，优于所有竞争方法，说明框架能精确界定阴影边界

## 亮点

- 理论贡献扎实：严格证明了任意噪声扩散不增加采样计算量，且 EDM 是 EDA 的特殊情况
- 从退化图像直接启动逆过程，避免高斯噪声注入带来的信息损失和距离增加
- 5 步采样即可达到 SOTA，比 100 步 Refusion 快 53 倍，具有临床应用潜力
- 统一了 SDE 框架下的噪声灵活性和结构参数灵活性

## 局限性

- SDE 框架在随机性与适用范围之间存在固有权衡：Case 1（最大随机性）仅适用于噪声可分解为固定基集的情况，Cases 2-3（通用但随机性较低）更接近确定性方法
- CT MAR 任务中仅用图像域信息，与 SOTA 双域方法仍有差距
- 仅在特定医学和自然图像复原任务上验证，缺乏对其他退化类型（如超分辨、去模糊）的实验
- 基集 $H_{x_0}$ 的选择依赖任务先验知识，缺少自动化基集学习机制

## 相关工作

- **EDM** [Karras et al.]：统一设计空间但限于高斯噪声，EDA 的直接推广对象
- **Flow Matching** [Lipman et al.]：ODE 框架支持任意分布变换，但缺乏 SDE 的随机性优势
- **MeanFlow** [Geng et al., 2025]：单步生成 SOTA，但平均化轨迹导致复原任务表现不佳
- **Cold Diffusion** [Bansal et al.]：用确定性退化算子替代高斯噪声，但缺乏严格理论基础
- **Refusion** [CVPR NTIRE]：高斯扩散复原代表，100 步采样性能被 EDA 5 步超越

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从多元高斯角度统一任意噪声扩散的 SDE 设计空间，理论贡献有价值
- 实验充分度: ⭐⭐⭐⭐ — 三个不同噪声类型的复原任务，医学+自然图像覆盖，消融充分
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，图示直观，但部分符号稍密集
- 价值: ⭐⭐⭐⭐ — 为扩散模型复原任务提供了更高效的统一框架，53 倍加速具有实际意义

<!-- RELATED:START -->

## 相关论文

- [Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](eda_arbitrary_noise_diffusion_design_space.md)
- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)
- [Elucidating the Design Space of Multimodal Protein Language Models](../../ICML2025/medical_imaging/elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)
- [MRI Contrast Enhancement Kinetics World Model](mri_contrast_enhancement_kinetics_world_model.md)

<!-- RELATED:END -->
