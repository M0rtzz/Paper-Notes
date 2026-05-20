---
title: >-
  [论文解读] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching
description: >-
  [ICCV 2025][图像生成][无监督图像复原] DDM4IP 提出一种无监督框架，利用条件流匹配（Conditional Flow Matching）建模退化分布，同时通过分布匹配损失自动学习未知的前向退化模型，仅需少量非配对数据即可在去模糊、非均匀 PSF 标定和盲超分辨率任务上达到或超过现有方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "无监督图像复原"
  - "逆问题"
  - "条件流匹配"
  - "分布匹配"
  - "前向模型学习"
---

# Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching

**会议**: ICCV 2025  
**arXiv**: [2506.14605](https://arxiv.org/abs/2506.14605)  
**代码**: [https://github.com/inria-thoth/ddm4ip](https://github.com/inria-thoth/ddm4ip)  
**领域**: 扩散模型  
**关键词**: 无监督图像复原、逆问题、条件流匹配、分布匹配、前向模型学习

## 一句话总结

DDM4IP 提出一种无监督框架，利用条件流匹配（Conditional Flow Matching）建模退化分布，同时通过分布匹配损失自动学习未知的前向退化模型，仅需少量非配对数据即可在去模糊、非均匀 PSF 标定和盲超分辨率任务上达到或超过现有方法。

## 研究背景与动机

**领域现状**：图像复原（Image Restoration）通常被建模为逆问题：已知退化观测 $y$，恢复原始图像 $x$。传统方法假设前向模型 $y = A(x) + \epsilon$ 完全已知（如已知模糊核、下采样算子），然后利用先验和优化方法求解。近年来扩散模型作为强大的图像先验被广泛用于逆问题求解（如 DPS、DiffPIR、GSPnP）。

**现有痛点**：现有方法面临两个关键限制。第一，大多数方法假设前向退化模型已知（non-blind setting），但实际场景中模糊核、噪声分布等往往未知或被错误指定。第二，即便是盲方法（blind methods）也通常假设可以获得配对的退化-干净图像对进行训练，但在真实场景中收集这样的配对数据代价极高——例如显微镜镜头标定需要复杂的实验设备。

**核心矛盾**：强大的扩散先验可以提供高质量的图像复原，但现有框架依赖已知的前向模型或配对训练数据，这在真实应用中往往不成立。如何在仅有少量非配对数据的情况下，同时学习前向模型和利用扩散先验进行复原？

**本文目标**：设计一个无监督框架，(1) 仅使用少量非配对的退化图像和干净图像；(2) 自动学习前向退化模型；(3) 利用学到的前向模型进行高质量图像复原。

**切入角度**：作者观察到条件流匹配（Conditional Flow Matching）可以高效地建模数据分布，且其训练目标自然地引出一个分布匹配损失。如果用一个可学习的前向算子将干净图像退化，然后要求退化后的分布与真实退化图像的分布匹配，就可以在不需要配对数据的情况下学习前向模型。

**核心 idea**：先用流匹配模型拟合退化图像的分布作为"参考"，再训练一个辅助流匹配模型建模"可学习前向算子生成的退化图像"的分布，通过匹配两个流模型的速度场来学习前向算子——本质上是一个分布级别的匹配而非样本级别的配对。

## 方法详解

### 整体框架

DDM4IP 采用三阶段流程：

**第一阶段**（学习退化分布）：在退化图像数据集上训练一个条件流匹配模型 $v_\theta$，使其学会建模退化图像的分布 $p(y)$。这个模型将高斯噪声映射到退化图像的分布，训练目标是标准的流匹配损失。

**第二阶段**（分布匹配学习前向模型）：同时训练两个网络——(1) 一个可学习的前向算子 $A_\phi$（kernel network，用于将干净图像退化），(2) 一个辅助流匹配模型 $v_\psi$（auxiliary flow network，建模 $A_\phi$ 生成的退化图像分布）。核心损失是让辅助模型的速度场匹配第一阶段学到的参考模型的速度场，从而间接约束 $A_\phi$ 生成的退化图像分布与真实退化分布一致。

**第三阶段**（模型反演）：利用第二阶段学到的前向模型 $A_\phi$，使用标准逆问题求解器（GSPnP、DPS、DiffPIR 等）对退化图像进行复原。

### 关键设计

1. **条件流匹配建模退化分布**:

    - 功能：建模退化图像的分布 $p(y)$，作为后续分布匹配的参考目标
    - 核心思路：流匹配定义从噪声 $x_0 \sim \mathcal{N}(0,I)$ 到数据 $x_1 \sim p(y)$ 的线性插值路径 $x_t = (1-t)x_0 + tx_1$，训练速度场网络 $v_\theta$ 预测方向 $x_1 - x_0$。损失为 $L_{FM} = \mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$。网络架构使用无预条件的 UNet（RFNoPrecond），支持条件输入以便建模条件依赖
    - 设计动机：相比传统扩散模型，流匹配的线性插值路径更简单、训练更稳定，且速度场的匹配自然地提供了分布匹配的接口——两个速度场一致等价于两个分布一致

2. **分布匹配损失学习前向模型（DiffInstruct-on-Y）**:

    - 功能：在不需要配对数据的情况下学习前向退化算子 $A_\phi$
    - 核心思路：给定干净图像 $x$，通过可学习的 kernel network $A_\phi$ 生成退化图像 $\hat{y} = A_\phi(x)$。在 $\hat{y}$ 上训练辅助流模型 $v_\psi$，同时计算分布匹配损失 $L_{DI} = \mathbb{E}[(v_\psi(y_t, t) - v_\theta^{ref}(y_t, t)) \cdot x_1]$，其中 $v_\theta^{ref}$ 是第一阶段固定的参考模型。这个损失反向传播到 $A_\phi$，驱动其生成的退化分布逼近真实退化分布。$A_\phi$ 还受到多种正则化约束（稀疏性、高斯性、中心性、归一化等），确保学到的退化核物理合理
    - 设计动机：传统的配对学习需要 $(x, y)$ 对，分布匹配将约束从样本级别提升到分布级别——只要生成的退化图像"看起来像"真实退化图像的分布即可，不需要知道每张退化图像对应的干净原图。这极大降低了数据需求

3. **标准逆问题求解器进行模型反演**:

    - 功能：利用学到的前向模型 $A_\phi$ 将 non-blind 逆问题求解器应用于图像复原
    - 核心思路：第二阶段学到的 $A_\phi$ 可以被嵌入任何标准逆问题框架。论文使用 DeepInv 库中的 GSPnP（Gradient-Step Plug-and-Play with RED prior）、DPS（Diffusion Posterior Sampling）和 DiffPIR 作为求解器，以 DRUNet 或 DiffUNet 作为去噪先验。已知 $A_\phi$ 后，这些求解器可以迭代式地恢复干净图像
    - 设计动机：解耦前向模型学习和逆问题求解，使得框架高度灵活——任何新的逆问题求解器都可以即插即用地替换第三阶段

### 损失函数 / 训练策略

- **第一阶段**：标准流匹配损失 $L_{FM}$，在 1000 张退化图像上训练
- **第二阶段**：分布匹配损失 $L_{DI}$ + kernel 正则化（稀疏正则、高斯正则、中心正则、归一化正则），辅助流模型与 kernel network 交替优化
- **第三阶段**：使用学到的前向模型，GSPnP/DPS/DiffPIR 求解器进行推理，不需要额外训练

## 实验关键数据

### 主实验

**FFHQ 运动去模糊**（256×256，训练集 1000 退化 + 100 干净非配对图像）：

| 方法 | 类型 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| Wiener (known kernel) | Non-blind | 27.5 | 0.82 | 0.22 |
| GSPnP (known kernel) | Non-blind | 30.2 | 0.88 | 0.12 |
| DPS (known kernel) | Non-blind | 29.8 | 0.87 | 0.13 |
| BlindDPS | Blind (single) | 25.3 | 0.74 | 0.31 |
| GibbsDDRM | Blind (single) | 26.1 | 0.76 | 0.28 |
| **DDM4IP + GSPnP** | **Unsupervised** | **29.5** | **0.87** | **0.14** |

**DIV2K 盲超分辨率**（DIV2KRK benchmark，×2 和 ×4）：

| 方法 | 类型 | PSNR↑ (×2) | PSNR↑ (×4) |
|------|------|-----------|-----------|
| KernelGAN + ZSSR | Blind (single) | 31.2 | 27.8 |
| DCLS | Supervised | 32.1 | 28.5 |
| Real-ESRGAN | Supervised | 30.8 | 27.4 |
| **DDM4IP + ESRGAN** | **Unsupervised** | **31.9** | **28.3** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | 说明 |
|------|-------|-------|------|
| Full DDM4IP (Step1+2+3) | 29.5 | 0.87 | 完整三阶段框架 |
| w/o kernel regularization | 27.8 | 0.81 | 无正则约束时核退化为任意卷积 |
| w/o auxiliary flow model | 26.4 | 0.77 | 没有辅助流模型无法做分布匹配 |
| Step1 only (直接用 DPS) | 25.1 | 0.73 | 仅建模退化分布不学前向模型 |
| w/ paired data (oracle) | 30.1 | 0.88 | 使用配对数据作为上界参考 |

### 关键发现

- DDM4IP 在去模糊任务上接近已知核的 non-blind 方法表现（PSNR 差距仅 0.7 dB），大幅超过单图盲方法（超过 3-4 dB），证明了分布匹配学习前向模型的有效性
- 在盲超分辨率上与 SOTA 监督方法表现接近（差距 0.2 dB），说明框架的泛化性
- Kernel 正则化对性能影响显著（去掉后 PSNR 下降 1.7 dB），说明物理先验约束对于从分布匹配中恢复正确退化核至关重要
- 真实场景 parking lot 镜头标定实验是一大亮点：传统方法需要专业设备，DDM4IP 仅需少量非配对照片即可完成非均匀 PSF 估计，展示了强大的实际应用潜力
- 辅助流模型是分布匹配的关键桥梁——去掉后性能下降 3.1 dB，说明直接在前向模型上施加分布约束行不通

## 亮点与洞察

- **分布匹配替代配对学习**是最核心的创新点：将逆问题中的数据需求从"配对样本"降低为"非配对分布"，极大拓展了扩散模型在真实场景图像复原中的适用范围。这个思路可以迁移到任何需要学习未知退化/变换的任务（如域适应、风格迁移等）
- **三阶段解耦设计**很优雅：预训练流模型 → 分布匹配学前向模型 → 标准求解器复原。每个阶段独立可替换，例如第三阶段可以无缝接入未来更强的逆问题求解器
- **流匹配的速度场匹配 = 分布匹配**这一洞察非常深刻：两个流模型的速度场一致意味着它们建模的分布一致，这为分布匹配提供了一个可微分的、易于优化的代理损失
- 仅需约 100 张干净图像和 1000 张退化图像即可工作，数据效率极高

## 局限与展望

- 三阶段流程较为复杂，总训练时间较长——第一阶段在 FFHQ 上训练约 420 万步，第二阶段额外训练 100 万步
- 前向模型目前限于卷积核形式的退化（模糊、下采样），对于更复杂的退化（如 JPEG 压缩、天气退化）需要设计不同的参数化
- 第三阶段的求解器（GSPnP、DPS 等）本身推理速度较慢，限制了实时应用
- 非均匀 PSF 实验中网格大小（8×8）固定，对于更精细的空间变化退化可能需要调整
- 实验主要在人脸（FFHQ）和自然图像（DIV2K）上进行，在医学图像等专业领域的表现未验证

## 相关工作与启发

- **vs DPS/DiffPIR**: 这些方法需要已知前向模型，DDM4IP 的核心贡献就是将已知模型放松为未知模型，先学再用。学到的前向模型可以直接输入DPS/DiffPIR
- **vs BlindDPS/GibbsDDRM**: 这些盲方法在单图上联合估计前向模型和复原图像，受限于单图信息不足，质量明显低于 DDM4IP 利用分布级信息的方案
- **vs CycleDiffusion/Unpaired IR**: 传统非配对方法通常使用 CycleGAN 式的循环一致性，DDM4IP 的分布匹配机制更严格和稳定

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 分布匹配学习前向模型的思路原创性强，自然且优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖去模糊、超分辨率和真实镜头标定三个场景，但缺少更多退化类型的验证
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，框架描述清晰，代码开源
- 价值: ⭐⭐⭐⭐⭐ 无监督逆问题求解是高实际价值方向，真实场景镜头标定的应用展示令人信服

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](../../NeurIPS2025/image_generation/npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](../../ICML2026/image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2025\] Unsupervised Learning for Class Distribution Mismatch (UCDM)](../../ICML2025/image_generation/unsupervised_learning_for_class_distribution_mismatch.md)

</div>

<!-- RELATED:END -->
