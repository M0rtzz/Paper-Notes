---
title: >-
  [论文解读] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration
description: >-
  [ICCV 2025][图像生成][人脸修图还原] 本文首次提出人脸修图还原(FRR)任务，并设计 MoFRR 框架——借鉴 DeepSeek MoE 思想，通过路由器激活特定修图类型的专家（小波 DDIM）和共享专家（通用 DDIM），在新构建的百万级 RetouchingFFHQ++ 数据集上实现了修图人脸的近真实还原。
tags:
  - ICCV 2025
  - 图像生成
  - 人脸修图还原
  - 混合专家
  - 小波变换
  - 扩散模型
  - 频域分解
---

# MoFRR: Mixture of Diffusion Models for Face Retouching Restoration

**会议**: ICCV 2025  
**arXiv**: [2507.19770](https://arxiv.org/abs/2507.19770)  
**代码**: 暂无  
**领域**: 图像生成 / 人脸修复  
**关键词**: 人脸修图还原, 混合专家, 小波变换, 扩散模型, 频域分解

## 一句话总结

本文首次提出人脸修图还原(FRR)任务，并设计 MoFRR 框架——借鉴 DeepSeek MoE 思想，通过路由器激活特定修图类型的专家（小波 DDIM）和共享专家（通用 DDIM），在新构建的百万级 RetouchingFFHQ++ 数据集上实现了修图人脸的近真实还原。

## 研究背景与动机

**社会问题**：人脸修图（瘦脸、大眼、美白、磨皮）在社交平台泛滥，引发审美退化、商业欺诈和身份造假等问题。挪威、美国、以色列已立法要求标注修图内容。

**研究空白**：现有工作聚焦于修图检测，但**如何从修图后的图像还原真实面孔**尚未被回答。这对追溯严重修图面孔的真实身份至关重要。

**FRR 不同于常规任务**：
   - 不同于**图像修复**（IR）：IR 关注高频纹理恢复，FRR 需恢复低频结构信息（脸型、眼睛大小）
   - 不同于**卸妆**：修图涉及面部结构变化（瘦脸、大眼），卸妆仅涉及纹理/颜色变化
   - 不同类型的修图操作逻辑独立、目标不同，不宜单模型处理

## 方法详解

### 整体框架 (MoE)

受 DeepSeek MoE 启发的分而治之策略：

1. **路由器**：ResNet-MAM 多标签分类器，输出 4 维二值向量 $[b_w, b_s, b_f, b_e]$，分别对应美白、磨皮、瘦脸、大眼
2. **专家网络**：4 个专用专家(WaveFRR) + 1 个共享专家（标准 DDIM）
3. **合并模块**：轻量级 UNet，融合所有激活专家的中间结果和原始图像

### 关键设计：WaveFRR 专家模型

每个 WaveFRR 包含双分支结构：

**低频分支（DDIM + IDEM）**：
- 对输入图像做离散小波变换 (DWT)，得到低频子带 $x_{LL}$ 和高频子带 $x_H$
- **度估计器**：ResNet50 预测特定修图操作的程度 $z$
- **IDEM 模块**：通过多尺度通道注意力网络生成像素级条件：

$$F = \text{MCA}(z+x_{LL}) \otimes x_{LL} + (1-\text{MCA}(z+x_{LL})) \otimes z$$
$$\hat{R} = \text{MCA}(F+\hat{y}_t) \otimes \hat{y}_t + (1-\text{MCA}(F+\hat{y}_t)) \otimes F$$
$$\tilde{x} = \text{Concat}(x_{LL}, \hat{R})$$

- 条件 DDIM 在此指导下采样无修图低频子带

**高频分支（HFCAM）**：
$$\hat{y}_H = x_H + \text{Conv}(\text{CA}(\hat{y}_0, x_H))$$

将恢复的低频子带 $\hat{y}_0$ 与原始高频子带 $x_H$ 通过交叉注意力对齐，修正高频细节。最终通过逆小波变换重建：$\hat{Y}_M = \text{IDWT}(\hat{y}_0, \hat{y}_H)$。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{freq} + \mathcal{L}_{space} + \mathcal{L}_{class}$$

- $\mathcal{L}_{freq} = \mathcal{L}_{IDEM} + \mathcal{L}_{simple} + \mathcal{L}_{high}$：频域损失（IDEM 残差 L2 + 扩散损失 + 高频 L2+TV）
- $\mathcal{L}_{space} = \mathcal{L}_{hyb}(Y,\hat{Y}) + \sum_M\mathcal{L}_{hyb}(Y,\hat{Y}_M)$：空间损失（L1 + 1-SSIM）
- $\mathcal{L}_{class}$：路由器和度估计器的交叉熵损失

### 共享专家设计

- 采用标准 DDIM 架构（无小波分解），与 WaveFRR 架构差异促进功能互补
- 在混合修图数据子集上训练，捕获跨修图类型的通用模式
- 持续激活（不经过路由器门控）

## 实验

### 内 API 测试（混合修图）

| 方法 | PSNR↑ (美白) | PSNR↑ (磨皮) | PSNR↑ (瘦脸) | PSNR↑ (大眼) | PSNR↑ (混合) |
|------|-------------|-------------|-------------|-------------|-------------|
| Input | 29.14 | 35.59 | 29.55 | 35.82 | 28.03 |
| Pix2pix | 27.72 | 28.55 | 27.34 | 28.41 | 28.73 |
| Restormer | 29.89 | - | - | - | - |
| **MoFRR** | **最优** | **最优** | **最优** | **最优** | **最优** |

### 专家可视化分析

| 专家 | 功能 |
|------|------|
| 共享专家 | 全局修图痕迹处理 |
| 瘦脸专家 | 恢复原始脸型 |
| 大眼专家 | 恢复原始眼睛大小 |
| 磨皮专家 | 恢复皮肤纹理细节 |
| 美白专家 | 恢复原始肤色 |

### 关键发现

- RetouchingFFHQ++ 包含 107 万修图图像（4 个商业 API），规模远超前作
- 独立专家的可视化显示各专家确实学到了针对性的修图逆操作
- 面部特征相似度 cosine 分布表明还原结果显著接近原始身份
- 跨 API 测试验证了模型的泛化能力

## 亮点与洞察

1. **首创 FRR 任务**：从修图检测延伸到修图还原，应用场景广泛（司法鉴定、身份验证）
2. **MoE + 扩散模型结合**：受 DeepSeek 启发的共享+专用专家架构
3. **小波域分治**：低频恢复结构、高频修正细节，针对 FRR 特点设计
4. **百万级数据集**扩展了 RetouchingFFHQ，重新定义了修图程度标准

## 局限性

- 仅支持 4 种修图类型，实际商业美颜操作更丰富
- 路由器分类错误会级联影响后续修复效果
- DDIM 推理需要多步采样，实时性有限

## 相关工作

- **卸妆**: PairedCycleGAN, PSGAN++, SSAT
- **图像修复**: DR2, Restormer, ResDiff
- **修图检测**: ResNet-MAM, RetouchingFFHQ

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个 FRR 任务定义 + MoE 扩散专家
- 技术深度：⭐⭐⭐⭐ — 小波 + DDIM + MoE 多层设计
- 实验充分度：⭐⭐⭐⭐ — 内 API + 跨 API 测试
- 实用价值：⭐⭐⭐⭐ — 反修图诈骗、司法取证

<!-- RELATED:START -->

## 相关论文

- [Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [SVFR: A Unified Framework for Generalized Video Face Restoration](../../CVPR2025/image_generation/svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [OSDFace: One-Step Diffusion Model for Face Restoration](../../CVPR2025/image_generation/osdface_one-step_diffusion_model_for_face_restoration.md)
- [VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)

<!-- RELATED:END -->
