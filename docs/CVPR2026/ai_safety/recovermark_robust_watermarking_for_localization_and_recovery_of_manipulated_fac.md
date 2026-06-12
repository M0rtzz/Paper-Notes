---
title: >-
  [论文解读] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces
description: >-
  [CVPR 2026][AI安全][人脸篡改检测] 提出 RecoverMark，一个将人脸内容本身作为水印嵌入背景的鲁棒水印框架，同时实现篡改区域定位、原始内容恢复和版权验证，在水印移除攻击下仍保持有效。
tags:
  - "CVPR 2026"
  - "AI安全"
  - "人脸篡改检测"
  - "鲁棒水印"
  - "篡改定位"
  - "内容恢复"
  - "版权验证"
---

# RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces

**会议**: CVPR 2026  
**arXiv**: [2602.20618](https://arxiv.org/abs/2602.20618)  
**代码**: 待发布（论文声明接收后公开）  
**领域**: AI安全  
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

RecoverMark 抓住一个现实约束：篡改者要骗过肉眼，就必须保留背景的语义一致性。于是它干脆把受保护的人脸内容当成水印藏进背景里——背景只要还在，水印就能提出来，从而一并完成篡改定位、内容恢复和版权验证。整条流水线是：先用分割工具（MTCNN/YOLOSeg/SAM2）把原图拆成显著区域（人脸）$I_{sal}$ 和背景 $I_{bg}$，编码器 Enc 把人脸压成潜在表示，隐藏网络 HNet 把这个表示嵌进背景得到容器图像 $I_{cntr}$；验证时提取网络 ENet 加解码器 Dec 再从容器图像里把人脸信息 $I'_{sal}$ 还原出来。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["原图 I_ori"] --> B["分割工具<br/>MTCNN / YOLOSeg / SAM2"]
    B -->|"人脸 I_sal"| C["编码器 Enc<br/>人脸压成潜在表示"]
    B -->|"背景 I_bg"| D["隐藏网络 HNet<br/>潜在表示嵌进背景"]
    C --> D
    D --> E["容器图像 I_cntr<br/>（发布、流传）"]
    E --> F["扰动层（两阶段渐进式训练）<br/>显著 / 全局 / 高级攻击，先难后易"]
    F --> G["提取网络 ENet + 解码器 Dec<br/>还原人脸 I'_sal"]
    G --> H["篡改定位<br/>与现存人脸做差"]
    G --> I["内容恢复<br/>还原原始人脸"]
    G --> J["版权验证<br/>NCC > 0.95"]
```

### 关键设计

**1. 以人脸内容自身为水印的编码/解码器：天然支持内容恢复**

双水印方法用一段独立于图像的脆弱水印做篡改检测，水印一被移除就彻底失效，而且还要和鲁棒水印争嵌入容量。RecoverMark 的 Enc/Dec 基于 CEILNet 架构，嵌入的不是外来比特而是人脸内容本身压缩成的潜在表示。这样做有两个直接好处：提取出来的就是原始人脸，恢复能力成了嵌入的副产品而非额外模块；而且水印内容与图像绑定，不再依赖"攻击者不知道水印存在"这一脆弱假设。

**2. 两阶段渐进式训练：先学会嵌入，再专门扛攻击**

直接在各种攻击下端到端训练很难收敛，于是拆成两段。Stage 1（初始训练）先训 Enc、HNet、ENet、Dec 四个网络，优化三个损失：保真度损失让容器图像贴近原始背景、水印损失让提取的人脸贴近原始人脸、清洁损失要求未嵌水印的背景提取出全白图（防止误检）。Stage 2（鲁棒性增强）冻结 Enc 和 Dec，在 HNet 与 ENet 之间插入扰动层模拟三类攻击——显著区域处理（给人脸区加噪，逼模型别依赖人脸本身去提取）、全局处理（JPEG 压缩、高斯噪声、低通滤波）、高级攻击（再生攻击）。关键在攻击的引入顺序：先单独训练最难的再生攻击，再逐步加入其余扰动；实验表明若把再生攻击延后引入，模型对最强攻击的防御会直接失效。

**3. 恢复、定位与版权验证三合一：一次提取，多个用途**

从可疑图像提出隐藏的人脸 $I'_{sal}$ 后，同一份信息可以同时支撑三件事：拿它和可疑图像里现存的人脸做差异比较，就生成篡改定位掩码；拿它和登记的原始人脸算归一化相关系数（NCC > 0.95），就完成版权验证。三个任务共用一个鲁棒水印，避免了双水印框架里两种水印相互干扰、争抢容量的老问题。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[ACL 2025\] Robust and Minimally Invasive Watermarking for EaaS](../../ACL2025/ai_safety/robust_and_minimally_invasive_watermarking_for_eaas.md)
- [\[CVPR 2026\] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)

</div>

<!-- RELATED:END -->
