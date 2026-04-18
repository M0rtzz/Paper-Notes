---
title: >-
  [论文解读] Mobile-VTON: High-Fidelity On-Device Virtual Try-On
description: >-
  [CVPR 2026][人体理解][虚拟试穿] 提出 Mobile-VTON，首个可完全在移动设备上离线运行的扩散模型虚拟试穿系统，通过 TeacherNet-GarmentNet-TryonNet（TGT）架构和特征引导对抗蒸馏策略，以 415M 参数和 2.84GB 显存实现媲美服务器端基线的高质量试穿效果。
tags:
  - CVPR 2026
  - 人体理解
  - 虚拟试穿
  - 移动端部署
  - 知识蒸馏
  - 扩散模型
  - 隐私保护
---

# Mobile-VTON: High-Fidelity On-Device Virtual Try-On

**会议**: CVPR 2026  
**arXiv**: [2603.00947](https://arxiv.org/abs/2603.00947)  
**代码**: [项目主页](https://zhenchenwan.github.io/Mobile-VTON/)  
**领域**: 人体理解  
**关键词**: 虚拟试穿, 移动端部署, 知识蒸馏, 扩散模型, 隐私保护

## 一句话总结

提出 Mobile-VTON，首个可完全在移动设备上离线运行的扩散模型虚拟试穿系统，通过 TeacherNet-GarmentNet-TryonNet（TGT）架构和特征引导对抗蒸馏策略，以 415M 参数和 2.84GB 显存实现媲美服务器端基线的高质量试穿效果。

## 研究背景与动机

虚拟试穿（VTON）近年视觉质量大幅提升，但主流系统依赖云端 GPU 推理，需要上传个人照片，带来三个核心问题：(1) 扩散模型参数量大、内存和延迟超出移动端能力；(2) 服装表征在扩散步间漂移导致语义不一致；(3) 现有方法依赖大规模预训练（如 text-to-image），轻量架构难以独立获得足够的服装合成能力。此外，上传个人照片存在隐私泄露风险。

## 方法详解

### 整体框架

Mobile-VTON 采用模块化 TGT 架构：**TeacherNet**（基于 SD 3.5 Large，冻结）提供监督信号；**GarmentNet**（轻量 Light-UNet）提取服装特征并保持时间步一致性；**TryonNet**（轻量 Light-UNet）融合人-衣表征合成最终试穿图像。核心是 Feature-Guided Adversarial (FGA) Distillation 将教师模型知识蒸馏到轻量学生网络中。

### 关键设计

1. **特征引导对抗蒸馏（FGA Distillation）**: 结合两个互补目标：

    - **特征级蒸馏**: 对齐教师和学生在每个扩散步的 score function 估计，而非回归像素值。给定噪声隐变量 $\tilde{\mathbf{z}}^{(t)}$，分别通过冻结教师 $D_t$ 和学生 $D_s$ 得到 $s_{\text{true}}$ 和 $s_{\text{fake}}$，最小化 $\ell_2$ 距离：
    $\mathcal{L}_{\text{feature}} = \mathbb{E}_t \| s_{\text{true}}(\tilde{\mathbf{z}}^{(t)}, t) - s_{\text{fake}}(\tilde{\mathbf{z}}^{(t)}, t) \|_2^2$
    - **对抗增强**: 轻量判别器 $D$ 区分真实/生成图像，TryonNet 通过欺骗判别器提升真实感：
    $\mathcal{L}_{\text{GAN}} = \mathbb{E}_{X \sim \mathcal{R}}[\log D(X)] + \mathbb{E}_{\hat{X} \sim \mathcal{G}}[\log(1 - D(\hat{X}))]$

2. **轨迹一致 GarmentNet（TCG）**: 解决服装特征在扩散步间的语义漂移问题。对每个时间步 $t$ 确定性地应用扩散过程，要求模型在所有步上一致重建原始服装图像：
    $\mathcal{L}_{\text{cons}} = \mathbb{E}_{t \sim [1,T]} [\| \hat{X}_g^{(t)} - X_g \|_2^2]$
   这种时间正则化使服装语义在扩散轨迹上保持稳定，避免纹理失真和形状扭曲。

3. **服装感知 TryonNet**: 无需大规模预训练，直接在试穿任务上从零训练：

    - **隐变量拼接（Latent Concatenation）**: 将人像和服装图像沿高度维度拼接编码为 $\mathbf{z}_{\text{concat}}$，同时构建参考输入 $X_{\text{condi}} = \text{Concat}_{\text{height}}(X_t, X_g)$ 引导保持身份和服装外观。
    - **多层级特征融合**: 每个自注意力层将 GarmentNet 的多尺度服装特征 $\mathbf{F}_g^{(i)}$ 与 TryonNet 的隐藏状态拼接，双分支交叉注意力同时融合文本条件和 Light-Adapter 的视觉服装语义。
    - **Light-Adapter**: 用 DINOv2-base（替代大型 CLIP 视觉编码器）提取服装视觉特征，通过 IP-Adapter 式解耦交叉注意力注入。

### 损失函数 / 训练策略

GarmentNet: $\mathcal{L}_{\text{GarmentNet}} = \lambda_1 \mathcal{L}_{\text{feature}}^G + \lambda_2 \mathcal{L}_{\text{cons}}$

TryonNet: $\mathcal{L}_{\text{TryonNet}} = \mathcal{L}_{\text{Diff}} + \lambda_1 \mathcal{L}_{\text{feature}}^T + \lambda_3 \mathcal{L}_{\text{GAN}}$

总损失: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GarmentNet}} + \mathcal{L}_{\text{TryonNet}}$，$\lambda_1=1\text{e-}2, \lambda_2=0.5, \lambda_3=5\text{e-}3$。

两阶段训练：第一阶段 140 epochs 在 DressCode+VITON-HD 合并集上（lr=1e-4）；第二阶段 100 epochs 在 DressCode 子集上微调（lr=5e-5）。8×A100 训练，batch size 256。

## 实验关键数据

### 主实验

| 方法 | VITON-HD LPIPS↓ | VITON-HD SSIM↑ | DressCode LPIPS↓ | 内存(GB) | 移动端 |
|------|----------------|----------------|-----------------|----------|--------|
| IDM-VTON | 0.102 | 0.868 | 0.065 | 18.47 | ✗ |
| BooW-VTON | 0.107 | 0.864 | 0.051 | 18.47 | ✗ |
| CatVTON | 0.161 | 0.872 | 0.092 | 5.80 | ✗ |
| StableVITON | 0.142 | 0.875 | 0.113 | 13.84 | ✗ |
| **Mobile-VTON** | **0.088** | **0.893** | **0.053** | **2.84** | **✓** |

### 消融实验

| 配置 | LPIPS↓ | SSIM↑ | CLIP-I↑ | FID↓ | 说明 |
|------|--------|-------|---------|------|------|
| 无 TCG, 无 LC | 0.119 | 0.874 | 0.798 | 11.231 | 基线 |
| +TCG | 0.111 | 0.879 | 0.805 | 10.814 | 服装语义稳定性提升 |
| +TCG +LC | 0.088 | 0.893 | 0.833 | 10.211 | 全部组件效果叠加 |

### 关键发现

- Mobile-VTON 以 2.84GB 内存和 415M 参数在 VITON-HD 上取得最优 LPIPS (0.088) 和 SSIM (0.893)，优于所有服务器端基线
- 在 DressCode 上 SSIM 最优 (0.935)，LPIPS 第二 (0.053)
- 完全 mask-free（不需分割掩码），需合成整张图像（含身体、背景），FID/KID 任务更难但仍有竞争力
- TCG 和 LC 各自贡献显著，组合使用效果叠加（LPIPS 从 0.119 降至 0.088）
- 在 In-the-Wild 场景下同样表现优异（LPIPS 0.133 对比 IDM-VTON 0.137）

## 亮点与洞察

- **首个移动端扩散 VTON**: 证明高质量虚拟试穿可完全在手机上运行，隐私保护具有实际商业价值
- **Score-based 蒸馏 + 对抗训练**: FGA 策略避免像素级回归的模糊问题，结合对抗损失提升真实感
- **轨迹一致性约束**: 简洁有效地解决扩散步间服装语义漂移问题
- **无需预训练**: 通过隐变量拼接和教师监督，轻量模型直接从任务数据学习，降低训练门槛
- **DINOv2 替代 CLIP**: 轻量且语义丰富的视觉特征提取，为移动端优化提供参考

## 局限与展望

- FID/KID 相比部分服务器端方法略高（mask-free 任务更难），分布对齐还有提升空间
- 仅支持上半身试穿，全身/下装/配饰场景待扩展
- 实际移动端推理速度和功耗未报告具体数值
- 两阶段训练流程较复杂，端到端一体化训练可能更高效
- 对极端姿态和遮挡场景的鲁棒性未充分评估

## 相关工作与启发

- **DMD2**: FGA 蒸馏策略的灵感来源（score-based distillation）
- **CatVTON**: 隐变量拼接策略的参考，Mobile-VTON 进一步与蒸馏结合
- **IDM-VTON**: 服装自注意力融合的参考，Mobile-VTON 以轻量方式实现类似效果
- **SnapGen**: Light-UNet 的参考架构，适配移动端高效推理
- 启发：知识蒸馏 + 对抗训练 + 领域特定约束的组合对模型压缩部署有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个移动端扩散 VTON，TGT 架构和 FGA 蒸馏有独创性
- **实验充分度**: ⭐⭐⭐⭐ 3 数据集、多基线对比、消融完整，但缺少实际移动端延迟数据
- **写作质量**: ⭐⭐⭐⭐ 架构图清晰，损失推导完备
- **价值**: ⭐⭐⭐⭐⭐ 直接对接产业需求（电商、隐私），首次证明高质量 VTON 的移动端可行性
