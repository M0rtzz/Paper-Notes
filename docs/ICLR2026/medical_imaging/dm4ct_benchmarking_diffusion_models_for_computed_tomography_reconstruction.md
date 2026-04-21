---
title: >-
  [论文解读] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction
description: >-
  [ICLR 2026][医学图像][CT重建] 提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。
tags:
  - ICLR 2026
  - 医学图像
  - CT重建
  - 扩散模型
  - benchmark
  - 逆问题
  - 稀疏视图重建
---

# DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction

**会议**: ICLR 2026  
**arXiv**: [2602.18589](https://arxiv.org/abs/2602.18589)  
**代码**: [有](https://github.com/DM4CT/DM4CT)  
**领域**: 医学图像  
**关键词**: CT重建, 扩散模型, benchmark, 逆问题, 稀疏视图重建

## 一句话总结

提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。

## 研究背景与动机

CT重建是典型的逆问题，从投影测量中恢复未知物体。当测量稀疏或含噪时，问题是病态的，需要先验知识。先验方法从经典正则化（TV）发展到深度学习（监督学习、DIP），再到最近的扩散模型。

扩散模型在图像生成领域成功后被引入逆问题求解，但CT成像面临特殊挑战：**相关噪声、伪影结构、系统几何依赖、值域不匹配**等，使直接应用扩散模型比自然图像生成困难得多。然而缺乏统一基准来系统评估各种扩散方法。

**核心贡献**：不是提出新算法，而是构建首个系统性基准，回答"扩散模型在CT中到底表现如何"。

## 方法详解

### 整体框架

DM4CT从贝叶斯视角组织扩散方法：后验分布 $p(\boldsymbol{x}|\boldsymbol{y}) \propto p(\boldsymbol{x})p(\boldsymbol{y}|\boldsymbol{x})$，将逆向SDE修改为条件逆向SDE，关键在于如何近似测量条件项 $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t)$。

### 关键设计：统一分类体系

将10种扩散方法按数据一致性和先验知识策略分为五大类：

**1. 数据一致性梯度引导（DC-grad）**：在每步去噪后计算数据保真梯度 $\boldsymbol{g}_t = \nabla_{\boldsymbol{x}_t}\mathcal{L}(\boldsymbol{A}\hat{\boldsymbol{x}}_0 - \boldsymbol{y})$，用步长 $\eta$ 控制引导强度。代表：DPS、PSLD、Reddiff等。

**2. 数据一致性优化步（DC-step）**：在去噪迭代间插入完整的数据一致性优化 $\boldsymbol{x}_t^* = \arg\min \mathcal{L}(\boldsymbol{A}\boldsymbol{x}_t - \boldsymbol{y})$。代表：ReSample、DMPlug。

**3. 即插即用（Plug-and-Play）**：解耦数据保真和先验，交替执行数据一致性子问题和无条件去噪步。代表：DMPlug。

**4. 伪逆引导（Pseudo Inverse）**：使用伪逆重建（FBP/SIRT近似）在测量空间和图像空间间传递信息。代表：MCG、PGDM。

**5. 变分贝叶斯（Variational Bayes）**：用参数化分布近似后验，无需沿扩散轨迹采样。代表：Reddiff、HybridReg。

### 数据集与配置

三类数据集覆盖不同应用场景：

- **医疗CT**：2016 Low Dose CT Challenge（9卷训练+1卷测试，512×512）
- **工业CT**：LoDoInd（管状样品含15种材料，3000+500切片）
- **同步辐射CT**（新采集）：两块岩石样品在高能同步辐射设施扫描，768×768高分辨率

五种仿真配置系统测试鲁棒性：40角度无噪/20角度轻噪/80角度强噪/80角度噪声+环形伪影/40角度限角。

### 基线方法

七种强基线覆盖各类方法范式：

- 经典方法：FBP、SIRT
- 神经网络先验：DIP、INR
- 高斯散布：R2Gaussian
- 迭代重建：FISTA-SBTV、ADMM-PDTV
- 监督学习：SwinIR

## 实验关键数据

### 主实验

**医疗数据集重建性能（PSNR/SSIM，部分配置）**

| 方法 | Config i (40角无噪) | Config ii (20角轻噪) | Config iv (80角噪声+环形) |
|------|:---:|:---:|:---:|
| FBP | 26.98/0.69 | 9.89/0.03 | 14.50/0.13 |
| SIRT | 30.40/0.80 | 26.23/0.47 | 25.86/0.40 |
| SwinIR (监督) | 32.45/0.88 | **29.92/0.83** | **30.79/0.85** |
| DDS (扩散最佳) | **31.43/0.84** | - | - |
| ReSample | 32.03/0.85 | 27.92/0.73 | 29.70/0.76 |
| INR | 33.21/0.86 | 26.15/0.76 | 29.50/0.74 |

**同步辐射真实数据（PSNR/SSIM）**

| 方法 | 200投影 | 100投影 | 60投影 |
|------|:---:|:---:|:---:|
| SwinIR | **33.75/0.76** | **33.05/0.73** | **32.41/0.70** |
| Reddiff | 28.43/0.56 | 28.24/0.54 | 28.06/0.51 |
| DDS | 28.36/0.55 | 28.10/0.51 | 27.90/0.49 |
| SIRT | 28.16/0.56 | 28.06/0.54 | 27.92/0.52 |

### 消融实验

**先验与数据一致性权衡**：以DPS为例，步长 $\eta$ 过小则先验主导（模糊），过大则测量噪声主导（崩溃）。最优 $\eta$ 需精细调节。

**像素空间 vs 潜空间扩散**：
- 潜空间（PSLD）：梯度需经VQ-VAE解码器传播，无噪声条件下也产生不连续伪影
- 优化步（ReSample）可修复不连续，但有噪声时过拟合测量

**零空间分析**：DC-grad（DPS）允许更多零空间内容，DC-step（ReSample）更严格约束，伪逆（PGDM）居中。

### 关键发现

1. **无单一扩散方法全面最优**，性能因数据集和配置而异
2. 扩散模型整体略优于经典/MBIR方法，但通常不如全监督SwinIR
3. 扩散模型恢复的细节虽然视觉逼真但可能偏离真值，导致指标不如INR/SwinIR的平滑重建
4. 真实数据上性能普遍低于仿真数据，暴露训练数据质量和分布偏移问题
5. 像素扩散通常比潜空间扩散更节省内存和时间

## 亮点与洞察

- **首个系统性CT扩散基准**：统一代码框架（diffusers），公平对比，开源代码和数据
- **统一分类体系**清晰梳理了方法间的设计选择和权衡
- **真实同步辐射数据集**弥补了现有评估缺乏真实数据的空白
- **深刻的实践洞察**：揭示了值域不匹配、有限训练数据、几何复杂性等真实部署挑战
- **零空间分析**提供了理解不同数据一致性策略的新视角

## 局限与展望

- 仅评估2D切片重建，未涉及3D重建（螺旋/锥束几何更具挑战性）
- 未包含flow-based方法（如FlowDPS），这是新兴方向
- 临床相关性评估不足（未做分割/放射科医师评分等下游任务评价）
- 扩散模型训练成本高，潜空间模型总训练时间更长
- 自然图像预训练的自编码器未必适合CT数据
- 跨设备/跨协议泛化性未测试

## 相关工作与启发

- 将DIP中DC loss的思想与扩散模型结合是有趣方向（参照DC loss论文）
- INR+扩散先验的混合方法可能兼具结构保真和细节恢复
- 扩散模型的不确定性量化能力（多次采样取均值/方差）有临床价值
- 对稀疏视图和高噪声场景，学习型先验优势最大；密集/低噪声时经典方法已足够

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★★☆ |
| 理论深度 | ★★★☆☆ |
| 实验充分性 | ★★★★★ |
| 实用价值 | ★★★★★ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography](../../ICCV2025/medical_imaging/coordinate-based_speed_of_sound_recovery_for_aberration-corrected_photoacoustic_.md)
- [Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)
- [Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [Protein Counterfactuals via Diffusion-Guided Latent Optimization](protein_counterfactuals_via_diffusion-guided_latent_optimization.md)

<!-- RELATED:END -->
