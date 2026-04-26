---
title: >-
  [论文解读] Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal
description: >-
  [CVPR 2026][图像生成][veiling glare removal] 提出 VeilGen + DeVeiler 框架，通过物理引导的 Stable Diffusion 生成模型学习潜在透射率和眩光图以合成逼真的复合退化训练数据，并用可逆约束训练修复网络，实现简化光学系统中像差与雾化眩光的联合去除。
tags:
  - CVPR 2026
  - 图像生成
  - veiling glare removal
  - 简化光学系统
  - 扩散模型
  - 物理引导生成
  - 可逆修复
---

# Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal

**会议**: CVPR 2026  
**arXiv**: [2511.17353](https://arxiv.org/abs/2511.17353)  
**代码**: [GitHub](https://github.com/XiaolongQian/DeVeiler)  
**领域**: 图像修复  
**关键词**: veiling glare removal, 简化光学系统, Stable Diffusion, 物理引导生成, 可逆修复

## 一句话总结

提出 VeilGen + DeVeiler 框架，通过物理引导的 Stable Diffusion 生成模型学习潜在透射率和眩光图以合成逼真的复合退化训练数据，并用可逆约束训练修复网络，实现简化光学系统中像差与雾化眩光的联合去除。

## 背景与动机

1. **简化光学系统的复合退化**：单透镜和超表面透镜等紧凑光学系统不仅存在像差模糊，还因非理想光学表面和镀膜产生雾化眩光（veiling glare），导致全局对比度下降。现有计算像差校正（CAC）方法无法处理这种复合退化。
2. **缺乏真实配对数据**：物理精确的眩光模拟需要完整光机模型和非序列光线追踪，计算代价极高，无法大规模生成训练数据。
3. **现有方法不适用**：去雾方法基于大气散射模型（深度相关），与镜头内部散射（深度无关）不兼容；镜头光斑去除方法针对结构化伪影（亮斑/条纹），无法处理弥散性的雾化眩光。
4. **生成模型的黑箱问题**：现有基于 SD 的退化生成方法缺乏物理约束，生成质量不稳定，且难以为修复网络提供有意义的引导信号。

## 方法详解

### 问题建模

退化模型将观测图像分解为像差与眩光两步：

$$I_{de}^{p} = \underbrace{(I_c^{p} \otimes K^{p})}_{I_{ab}^{p}} \cdot T^{p} + I_g^{p}$$

其中 $K^p$ 为局部点扩散函数（PSF），$T^p$ 为透射率图（描述对比度衰减），$I_g^p$ 为眩光图。修复目标是从 $I_{de}$ 恢复 $I_c$，这是一个高度病态的盲反问题。

### 整体框架：三阶段物理引导管线

#### Stage I: VeilGen — 物理引导的退化生成

基于 Stable Diffusion v2-1 构建，核心创新是在扩散生成过程中嵌入物理先验：

- **LOTGMP（潜在光学透射率与眩光图预测器）**：从噪声潜变量 $z_t$、目标退化潜变量 $z_{de}^{\mathcal{T}}$ 和时间步 $t$ 预测潜在图 $c_{vg} = (z_{trans}, z_{glare})$
- **VGIM（雾化眩光施加模块）**：利用预测的潜在图调制特征，模拟前向退化过程

混合训练策略：源域（有配对、无眩光）用固定映射 $(\mathbf{1}, \mathbf{0})$，目标域（无配对、有复合退化）用 LOTGMP 预测潜在图。总损失为：

$$\mathcal{L}_{gen} = p \, \mathcal{L}_{\mathcal{S}} + (1-p) \, \mathcal{L}_{\mathcal{T}}$$

其中 $p = 0.3$ 平衡两个域的贡献。训练完成后，VeilGen 离线生成配对数据集用于修复网络训练。

#### Stage II: DDN — 前向模型蒸馏

由于 VeilGen 的多步扩散采样在训练时计算代价过高，将其蒸馏为轻量级蒸馏退化网络（DDN）：

$$\mathcal{L}_{distill} = \|DDN(I_c, c_{vg}) - VeilGen(I_c, c_{vg})\|_1$$

冻结的 DDN 在 Stage III 中作为前向退化模型提供可逆性约束监督。

#### Stage III: DeVeiler — 可逆修复网络

核心思想是让修复网络学习退化的逆映射，而非黑箱统计映射：

- **VG-Enc（眩光编码器）**：从退化图像预测潜在图 $\hat{c}_{vg}$
- **VGCM（眩光补偿模块）**：利用预测的潜在图进行逆向特征调制，与 VGIM 形成结构对称

**可逆性约束**：冻结的 DDN 用预测图 $\hat{c}_{vg}$ 正向施加退化，要求能重建出观测的退化图像：

$$\mathcal{L}_{rev} = \|DDN(I_c, \hat{c}_{vg}) - I_{de}\|_1$$

总训练目标：

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_{rev} \mathcal{L}_{rev}$$

**两阶段训练**：Phase I 在源域预训练建立像差校正基线；Phase II 在源域 + VeilGen 合成对的混合数据集上微调，防止过拟合。

### 关键设计洞察

- LOTGMP 预测的潜在图需同时参与正向（VGIM）和逆向（VGCM）路径才有效。单向使用因域适配问题反而降低性能
- 可逆约束迫使潜在图获得物理可解释性——低透射率区域和高眩光区域可在特征图中可视化验证

## 实验结果

### Screen-Compound 域全参考评估

| 方法 | 类型 | Screen-SL PSNR↑ | SSIM↑ | LPIPS↓ | Screen-MRL PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|---------|-------|--------|----------|-------|--------|
| SwinIR | 单退化 | 18.18 | 0.686 | 0.298 | 19.34 | 0.722 | 0.354 |
| NAFNet | 单退化 | 18.75 | 0.684 | 0.363 | 18.91 | 0.723 | 0.377 |
| DiffBIR | 单退化 | 17.95 | 0.621 | 0.398 | 18.70 | 0.625 | 0.412 |
| SwinIR+Flare7K++ | 级联 | 21.67 | 0.723 | 0.297 | 20.74 | 0.745 | 0.336 |
| QDMR | 域适应 | 18.45 | 0.681 | 0.291 | 20.67 | 0.725 | 0.315 |
| **DeVeiler (Ours)** | **域适应** | **22.38** | **0.729** | **0.261** | **21.57** | **0.746** | **0.301** |

DeVeiler 相比最强竞争方法（SwinIR+Flare7K++）在 Screen-SL 上 PSNR 提升 +0.71 dB，LPIPS 改善 12.1%。

### Realworld-Compound 域无参考评估

| 方法 | Real-SL (CLIPIQA↑/Q-Align↑/NIQE↓) | Real-MRL (CLIPIQA↑/Q-Align↑/NIQE↓) |
|------|-------------------------------------|--------------------------------------|
| SwinIR | 0.424 / 3.518 / 5.710 | 0.374 / 3.191 / 6.696 |
| SwinIR+DiffDehaze | 0.573 / 3.679 / 6.141 | 0.437 / 3.542 / 5.230 |
| QDMR | 0.405 / 3.864 / 4.773 | 0.376 / 3.337 / 5.509 |
| **DeVeiler (Ours)** | **0.607** / **3.987** / **4.448** | **0.440** / **3.586** / **5.296** |

在无 GT 的真实场景中，DeVeiler 在大部分指标上保持领先，验证了泛化能力。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| 无 LOTGMP | 20.82 | 0.708 | 0.273 | 无物理先验引导 |
| LOTGMP 无 SD 先验 | 21.39 | 0.708 | 0.268 | 缺少扩散模型上下文 |
| 完整 VeilGen | 21.56 | 0.712 | 0.264 | SD 先验对生成质量至关重要 |
| 单向 VGCM | 20.83 | 0.712 | 0.264 | 域不匹配导致无增益 |
| **双向 VGIM/VGCM** | **22.38** | **0.729** | **0.261** | 可逆约束带来 +0.82 dB |

关键发现：单向使用潜在图（Concat 或仅 VGCM）不仅无益甚至有害，只有通过 VGIM/VGCM 的双向一致性约束才能有效利用物理先验。

## 亮点

- **物理引导而非黑箱生成**：将散射模型的透射率和眩光图显式嵌入 SD 生成过程，使合成数据具有物理意义
- **可逆性约束的精巧设计**：通过 VGIM/VGCM 的结构对称和冻结 DDN 的循环一致性，将修复与退化建模统一为互逆操作
- **可解释性强**：VGCM 内部特征图可可视化验证低透射率/高眩光区域的定位与抑制
- **实用性高**：在真实单透镜和超表面混合透镜两种系统上验证，覆盖 AR/VR 和移动摄影应用场景
- **高效推理**：相比 DiffBIR (66s) 或 DiffDehaze (92s)，DeVeiler 推理延迟低一个数量级

## 局限性

- **依赖源域配对数据**：框架仍需屏幕采集的像差-清晰配对数据作为源域，完全无监督场景不适用
- **退化耦合假设**：将像差和眩光建模为顺序退化，实际物理过程可能更复杂（如光学串扰）
- **数据规模有限**：Screen-Compound 测试集仅 42/25 张图，统计显著性有待更大规模验证
- **蒸馏损失**：DDN 蒸馏 VeilGen 不可避免地引入近似误差，对极端眩光场景的覆盖度存疑

## 评分

- ⭐⭐⭐⭐ 新颖性：将物理散射模型嵌入扩散生成 + 可逆修复的双向设计思路新颖
- ⭐⭐⭐⭐ 实用性：解决 AR/VR 和移动摄影中的真实痛点，推理速度可接受
- ⭐⭐⭐ 实验充分度：两种光学系统、多类基线对比充分，但测试集规模偏小
- ⭐⭐⭐⭐ 写作质量：问题定义清晰，三阶段框架逻辑连贯，消融实验有说服力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[CVPR 2026\] Learning Latent Proxies for Controllable Single-Image Relighting](learning_latent_proxies_for_controllable_single-image_relighting.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)

<!-- RELATED:END -->
