---
title: >-
  [论文解读] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces
description: >-
  [CVPR 2026][人体理解][人脸篡改检测] 提出 RecoverMark，一个将人脸内容本身作为水印嵌入背景的鲁棒水印框架，同时实现篡改区域定位、原始内容恢复和版权验证，在水印移除攻击下仍保持有效。
tags:
  - CVPR 2026
  - 人体理解
  - 人脸篡改检测
  - 鲁棒水印
  - 篡改定位
  - 内容恢复
  - 版权验证
---

# RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces

**会议**: CVPR 2026  
**arXiv**: [2602.20618](https://arxiv.org/abs/2602.20618)  
**代码**: 待发布（论文声明接收后公开）  
**领域**: 人体理解  
**关键词**: 人脸篡改检测, 鲁棒水印, 篡改定位, 内容恢复, 版权验证

## 一句话总结

提出 RecoverMark，一个将人脸内容本身作为水印嵌入背景的鲁棒水印框架，同时实现篡改区域定位、原始内容恢复和版权验证，在水印移除攻击下仍保持有效。

## 研究背景与动机

**领域现状**: AI 生成内容（AIGC）技术使人脸篡改变得极为便捷，如 Stable Diffusion 和各种 GAN 变体，对视觉内容的真实性和知识产权构成严重威胁。

**现有痛点**: 现有主动防御方法（如 EditGuard、OmniGuard）采用脆弱水印 + 鲁棒水印的双水印策略。脆弱水印用于篡改检测与定位，鲁棒水印用于版权认证。但这类方法假设攻击者不知道水印的存在，面对水印移除攻击（如低通滤波、再生攻击）时完全失效。

**核心矛盾**: 双水印框架中两种水印相互干扰，且嵌入容量有限，进一步削弱了脆弱水印的有效性；同时现有方法忽视了篡改区域内容恢复这一关键需求。

**本文目标**: 设计一个在对抗水印移除攻击时仍能同时完成篡改定位、内容恢复和版权验证的统一鲁棒水印框架。

**切入角度**: 利用一个关键的现实约束——攻击者必须保留背景的语义一致性以避免被视觉检测发现。因此将人脸内容本身作为水印嵌入到周围背景中，背景保真则水印可提取。

**核心 idea**: 将受保护的人脸内容作为水印鲁棒地嵌入背景区域，通过两阶段渐进式训练实现同时篡改定位、恢复和版权验证。

## 方法详解

### 整体框架

RecoverMark 的流程：首先使用分割工具（MTCNN/YOLOSeg/SAM2）将原始图像分离为显著区域（人脸）$I_{sal}$ 和背景 $I_{bg}$；水印编码器 Enc 将人脸压缩为潜在表示；隐藏网络 HNet 将该表示嵌入背景生成容器图像 $I_{cntr}$；提取网络 ENet + 解码器 Dec 从容器图像中恢复人脸信息 $I'_{sal}$。

### 关键设计

1. **水印编码/解码器（Enc/Dec）**: 基于 CEILNet 架构，将人脸内容压缩为适合嵌入的潜在表示。核心思路是利用图像自身内容作为水印增强提取鲁棒性。设计动机是避免使用独立于图像的脆弱水印，从而天然支持内容恢复。

2. **两阶段渐进式训练**:

    - **Stage 1（初始训练）**: 训练 Enc、HNet、ENet、Dec 四个网络，优化三个损失：保真度损失（容器图像接近原始背景）、水印损失（提取的人脸接近原始人脸）、清洁损失（未嵌入水印的背景提取结果为全白图像，防止误检）。
    - **Stage 2（鲁棒性增强训练）**: 冻结 Enc 和 Dec，在 HNet 和 ENet 之间引入扰动层，模拟三类攻击：显著区域处理（给人脸区域加噪声，防止提取依赖人脸信息）、全局处理（JPEG 压缩、高斯噪声、低通滤波）、高级攻击（再生攻击）。采用渐进策略，先训练最难的再生攻击，再逐步引入其他扰动。

3. **恢复、定位与版权验证**: 从可疑图像提取隐藏的人脸信息 $I'_{sal}$ 后：（1）与可疑图像的人脸进行差异比较生成篡改定位掩码；（2）与原始人脸计算归一化相关系数（NCC > 0.95）完成版权验证。

### 损失函数 / 训练策略

总损失为：$\mathcal{L}_{sum} = \alpha_1 \mathcal{L}_{fidelity} + \alpha_2 \mathcal{L}_{wm} + \alpha_3 \mathcal{L}_{clean}$，其中三个权重均设为 1。

渐进训练策略：再生攻击占总 epoch 的前半部分，其余扰动平分后半部分。实验表明先引入再生攻击至关重要，延迟引入会导致对最强攻击的防御失效。

## 实验关键数据

### 主实验

在 ID 数据集（CelebA）上 Structpix2pix 篡改下的定位性能（F1/AUC）：

| 方法 | 再生攻击 F1 | 再生攻击 AUC | 噪声 F1 | 噪声 AUC | JPEG F1 | JPEG AUC | 低通 F1 | 低通 AUC | Lattice F1 | Lattice AUC |
|------|-----------|-------------|--------|---------|--------|---------|--------|---------|-----------|------------|
| MVSS-Net | 0.041 | 0.723 | 0.062 | 0.711 | 0.157 | 0.776 | 0.184 | 0.755 | 0.034 | 0.719 |
| EditGuard | 0.090 | 0.610 | 0.528 | 0.932 | 0.552 | 0.954 | 0.090 | 0.658 | 0.438 | 0.930 |
| OmniGuard | 0.105 | 0.655 | 0.127 | 0.659 | 0.315 | 0.890 | 0.146 | 0.743 | 0.113 | 0.689 |
| **RecoverMark** | **0.855** | **0.993** | **0.876** | **0.992** | **0.867** | **0.993** | **0.830** | **0.989** | **0.842** | **0.991** |

### 消融实验 / 恢复性能

ID 数据集上人脸恢复质量对比（PSNR/MS-SSIM）：

| 方法 | 再生攻击 PSNR | 再生攻击 MS-SSIM | 噪声 PSNR | 噪声 MS-SSIM | JPEG PSNR | JPEG MS-SSIM | Lattice PSNR | Lattice MS-SSIM |
|------|-------------|-----------------|---------|-------------|---------|-------------|-------------|----------------|
| Imuge+ | 7.252 | 0.339 | 10.432 | 0.563 | 10.778 | 0.629 | 9.089 | 0.424 |
| **RecoverMark** | **22.154** | **0.607** | **23.276** | **0.657** | **23.314** | **0.680** | **23.230** | **0.655** |

### 关键发现

- RecoverMark 在所有攻击类型下均大幅超越所有基线方法，F1 超过 0.7，AUC 超过 0.98。
- 在未见过的 Lattice 攻击下仍保持高性能（F1=0.842），展示了强泛化能力。
- 恢复 PSNR 比 Imuge+ 高出约 12-15 dB，内容恢复质量显著提升。
- 版权验证成功率达 99.9%。
- 当人脸占比 ≤ 60% 时嵌入保真度维持在高水平，超过后显著下降。

## 亮点与洞察

- **统一框架**: 首次将篡改定位、内容恢复、版权验证三个任务统一到单一鲁棒水印框架中，避免了双水印的容量竞争问题。
- **利用现实约束**: 巧妙利用"攻击者必须保留背景一致性"这一现实约束，将人脸嵌入背景实现鲁棒提取。
- **渐进训练策略**: 类比人类学习从难到易的方法学，先训练最难的再生攻击再逐步加入其他扰动。
- **泛化到未见攻击**: 对训练中未包含的 Lattice 攻击仍有效，证明了鲁棒性的泛化性。

## 局限与展望

- 当人脸占图像比例超过 60% 时嵌入质量下降，容量有限。
- 当前仅在 256×256 分辨率上验证，高分辨率场景效果未知。
- 分割工具的精度直接影响嵌入与提取质量。
- 仅在人脸篡改场景下验证，尚未推广到其他类型的图像篡改。

## 相关工作与启发

- 被动检测方法（MVSS-Net、HiFi-Net）依赖篡改痕迹，但这些痕迹易被后处理消除。
- 主动防御方法（EditGuard、OmniGuard）依赖脆弱水印，对移除攻击脆弱。
- 水印自恢复方法（Imuge/Imuge+）首次尝试 DNN 联合定位与恢复，但同样基于脆弱嵌入。
- RecoverMark 的核心突破是从脆弱水印范式转向鲁棒水印范式，同时保持篡改检测灵敏度。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将人脸本身作为水印嵌入背景的思路新颖，统一三个任务的框架设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ — ID/OOD 数据集、多种已见/未见攻击、多种篡改方式、容量分析、定性定量全面
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清楚，从实际场景出发（法庭证据场景），逻辑自洽
- **价值**: ⭐⭐⭐⭐ — 对人脸内容保护有直接应用价值，统一框架减少了实际部署的复杂度

<!-- RELATED:START -->

## 相关论文

- [AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark](all_in_one_unifying_deepfake_detection_tampering_localization_and_source_tracing.md)
- [PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](pad-hand_physics-aware_diffusion_for_hand_motion_recovery.md)
- [A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)
- [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)

<!-- RELATED:END -->
