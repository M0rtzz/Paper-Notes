---
title: >-
  [论文解读] Enhancing Facial Privacy Protection via Weakening Diffusion Purification
description: >-
  [CVPR 2025][图像生成][人脸隐私保护] 本文通过学习逐时间步的无条件嵌入（unconditional embeddings）来削弱 LDM 反向扩散过程中的净化效应，并利用自注意力图引导保持结构一致性，在 CelebA-HQ 和 LADN 上平均 PSR 达 79.17%，同时 FID 优于所有竞争方法。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "人脸隐私保护"
  - "扩散模型"
  - "对抗样本"
  - "扩散净化"
  - "自注意力引导"
---

# Enhancing Facial Privacy Protection via Weakening Diffusion Purification

**会议**: CVPR 2025  
**arXiv**: [2503.10350](https://arxiv.org/abs/2503.10350)  
**代码**: [https://github.com/parham1998/Facial-Privacy-Protection](https://github.com/parham1998/Facial-Privacy-Protection)  
**领域**: 扩散模型 / 人脸隐私保护  
**关键词**: 人脸隐私保护, 扩散模型, 对抗样本, 扩散净化, 自注意力引导

## 一句话总结
本文通过学习逐时间步的无条件嵌入（unconditional embeddings）来削弱 LDM 反向扩散过程中的净化效应，并利用自注意力图引导保持结构一致性，在 CelebA-HQ 和 LADN 上平均 PSR 达 79.17%，同时 FID 优于所有竞争方法。

## 研究背景与动机

**领域现状**：随着人脸识别（FR）技术的广泛应用，面部隐私保护变得日益重要。目标化去标识（targeted de-identification）通过生成隐私保护图像来冒充目标身份，从而在 FR 系统面前隐藏原始身份。主流方法包括基于噪声的（PGD、MI-FGSM）、基于妆容的（AMT-GAN、DiffAM）和基于扩散的方法。

**现有痛点**：噪声方法生成可感知的扰动影响视觉质量；妆容方法需要参考图像且需为每个目标身份重训模型；DiffProtect 作为首个基于扩散的方法修改 Diff-AE 的语义码，但会改变人脸结构使其偏向目标身份，且受限于**扩散净化效应**——反向去噪过程会将对抗修改当作高频噪声逐步去除，导致保护成功率低。

**核心矛盾**：扩散模型的反向过程天然具有净化能力——训练数据全是干净图像的去噪模型会将任何偏离干净数据流形的扰动"纠正"回来。这恰恰与对抗性保护的目标矛盾：保护需要保留对抗扰动，而去噪过程会移除它。

**本文目标**：(1) 削弱扩散净化效应以保留对抗修改；(2) 在保护能力和视觉质量之间取得更好的平衡；(3) 避免修改语义码导致的结构失真。

**切入角度**：作者受 Null-Text Inversion 启发，发现学习逐时间步的无条件嵌入（null-text embeddings）不仅能提升图像重建质量，更关键的是能**削弱模型对输入的过度净化**——学习后的嵌入引导模型保留输入的细节（包括对抗扰动），而非一味去噪。

**核心 idea**：两阶段学习框架——第一阶段学习无条件嵌入以削弱净化效应并保持图像质量，第二阶段在冻结嵌入的条件下直接优化 LDM 的潜在码来生成保护图像，用自注意力引导保持结构一致性。

## 方法详解

### 整体框架
输入原始人脸图像 $x$，首先通过 DDIM Inversion 获得噪声潜在码 $z_t$。第一阶段：在反向采样过程中学习每个时间步的无条件嵌入 $\{\emptyset_i\}_{i=1}^t$，最小化重建损失以确保高质量重建。第二阶段：冻结已学习的嵌入，从 $z_{adv} = z_t$ 出发优化对抗潜在码，最小化对抗损失（使保护图像在 FR 模型特征空间中接近目标身份）和结构保持损失（对齐自注意力图）。

### 关键设计

1. **无条件嵌入学习（Null-Text Guidance）**:

    - 功能：削弱扩散净化效应，提升重建质量并保留对抗扰动
    - 核心思路：在 DDIM 采样步骤 $z_{t-1} = f(z_t, t, \emptyset_t)$ 中，为每个时间步 $t$ 学习一个独立的无条件嵌入 $\emptyset_t$，最小化 $\mathcal{L}_{rec} = \|z_{t-1} - \bar{z}_{t-1}\|_2^2$。学习后的嵌入为 U-Net 提供额外的学习容量，使其能记住输入的细粒度纹理和结构细节，而非将所有偏差当作噪声去除。AdamW 优化器，学习率 0.1，20 次迭代。
    - 设计动机：标准的 DDIM Inversion 使用固定的 null-text 嵌入会导致重建退化（皮肤和头发细节丢失变模糊）。学习后的嵌入隐式地为反向过程提供了"保留输入"的引导信号，对抗修改因此不被完全净化。

2. **对抗潜在码优化**:

    - 功能：生成能欺骗 FR 模型的保护图像
    - 核心思路：在黑盒设置下，使用 $K$ 个白盒代理 FR 模型优化对抗损失 $\mathcal{L}_{adv} = \frac{1}{K}\sum_{k=1}^K [1 - \cos(\mathcal{F}_k(x_p), \mathcal{F}_k(x_t))]$，其中 $x_p$ 是解码后的保护图像，$x_t$ 是目标身份图像。优化直接在 LDM 潜在空间中进行，无约束地修改 $z_{adv}$（不施加 $L_\infty$ 范数约束），因为结构一致性由自注意力引导保证。AdamW 优化器，学习率 0.01，35 次迭代。
    - 设计动机：DiffProtect 修改语义码并约束修改幅度以维持结构，但这限制了保护能力。本文方法将结构保持从潜在码上解耦到自注意力引导上，允许潜在码更自由地优化以最大化保护效果。

3. **自注意力引导的结构保持**:

    - 功能：在无约束优化潜在码的同时维持原始图像的结构一致性
    - 核心思路：对比修改前后 DDIM 采样过程中 U-Net 的自注意力图：$\mathcal{L}_{str} = \|S(z_{adv}) - S(\bar{z}_t)\|_2^2$。自注意力图编码了图像的几何和形状信息（如面部轮廓、五官位置），约束其一致性可确保保护图像与原图结构相同，而身份相关的修改集中在纹理层面。总损失为 $\mathcal{L} = \lambda_{adv}\mathcal{L}_{adv} + \mathcal{L}_{str}$，$\lambda_{adv}=0.003$。
    - 设计动机：研究表明自注意力图控制图像的空间布局和结构，而交叉注意力图控制文本-图像对齐。由于本方法不使用文本条件，自注意力是保持结构的最佳选择。相比 $L_\infty$ 约束，自注意力引导允许更大的潜在码修改空间，提升保护性能。

### 损失函数 / 训练策略
- 阶段一：$\mathcal{L}_{rec} = \|z_{t-1} - \bar{z}_{t-1}\|_2^2$，20 步 DDIM Inversion，从第 3 个时间步开始反向，学习 20 次迭代
- 阶段二：$\mathcal{L} = \lambda_{adv}\mathcal{L}_{adv} + \mathcal{L}_{str}$，35 次迭代优化 $z_{adv}$
- 总生成时间约 15 秒/图（单张 RTX 4090）

## 实验关键数据

### 主实验：黑盒保护成功率（PSR%）

| 方法 | 类别 | IRSE50 | IR152 | FaceNet | MobileFace | 平均 |
|------|------|--------|-------|---------|------------|------|
| TIP-IM | 噪声 | 54.40 | 37.23 | 40.74 | 48.72 | 50.06 |
| AMT-GAN | 妆容 | 76.96 | 35.13 | 16.62 | 50.71 | 52.84 |
| CLIP2Protect | 妆容 | 81.10 | 48.42 | 41.72 | 75.26 | 64.90 |
| DiffAM | 妆容 | 92.00 | 63.13 | 64.67 | 83.35 | 77.88 |
| DiffProtect | 扩散 | 67.75 | 60.14 | 35.19 | 64.33 | 51.05 |
| **Ours** | **扩散** | **88.87** | **67.25** | **59.53** | **91.57** | **79.17** |

### 图像质量对比

| 方法 | PSR | FID↓ | PSNR↑ | SSIM↑ |
|------|-----|------|-------|-------|
| DiffAM | 77.88 | 26.10 | 20.53 | 0.886 |
| DiffProtect | 51.05 | 28.29 | 24.21 | 0.879 |
| **Ours** | **79.17** | **15.32** | **27.72** | 0.839 |

### 关键发现
- 本方法比 DiffProtect 平均 PSR 提高约 28%，比 DiffAM 提高约 1.3%，同时 FID 从 26.10 降至 15.32（图像更自然）
- 无条件嵌入的效果在深时间步（$t=5,7$）最为显著——没有嵌入时 PSR 和 FID 双双恶化
- 去掉自注意力引导后 PSR 更高但 FID 显著上升，说明自注意力引导在保护能力和视觉质量之间起平衡作用
- 对高斯和均值滤波的自适应攻击测试中，PSR 从 88.87 降至 86.66（5×5 均值滤波），鲁棒性较好
- 在 Face++ 和 Tencent 商业 API 上的置信度得分均高于所有竞争方法

## 亮点与洞察
- **以子之矛攻子之盾**：利用扩散模型本身的净化属性作为切入点——通过学习无条件嵌入来"反净化"，思路精巧。这表明扩散模型的净化能力并非不可控制，通过适当引导可以选择性地保留或移除信息。
- **结构保持与保护解耦**：将结构约束从潜在码上移到自注意力空间，允许潜在码更大自由度的修改，同时自注意力图自然编码了面部结构信息。这种解耦策略可以迁移到其他需要在保持结构的同时进行语义修改的任务。
- **无条件嵌入的多重作用**：不仅提升重建质量，还能削弱净化效应保留对抗扰动。一个组件同时解决两个问题，设计优雅。

## 局限与展望
- **生成速度**：15 秒/图仍偏慢，不适合实时应用。作者提到可通过 UNet 语义空间直接攻击来加速
- **SSIM 不是最优**：SSIM 指标低于 DiffAM（0.839 vs 0.886），说明像素级一致性有所牺牲
- **依赖代理模型**：黑盒攻击依赖 3 个白盒代理模型的迁移性，若目标模型架构差异大则效果可能下降
- **伦理考量**：冒充真实身份存在伦理风险。作者在讨论中提出冒充合成身份（synthesized target）作为替代方案，在四个 FR 模型上 PSR 为 85-90%

## 相关工作与启发
- **vs DiffProtect**：DiffProtect 修改 Diff-AE 的语义码并约束修改幅度，导致保护能力受限且结构变形；本文在 LDM 潜在空间无约束优化，用自注意力引导替代幅度约束，PSR 从 51.05% 提升到 79.17%
- **vs DiffAM**：DiffAM 需要两个扩散模型（去妆 + 化妆迁移）且需为每个目标身份重训；本文方法更streamlined，且 FID 优于 DiffAM（15.32 vs 26.10）
- **vs Null-Text Inversion (Mokady et al.)**：原方法用于真实图像编辑的精确重建；本文创新性地将其双重利用为削弱净化效应的工具

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 null-text 嵌入学习用于反净化的思路新颖，自注意力引导的结构保持替代 $L_\infty$ 约束也有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、四个 FR 模型、九个竞争方法对比，消融分析充分，还测试了商业 API 和自适应攻击
- 写作质量: ⭐⭐⭐⭐ 论文组织清晰，动机推导有说服力，图表丰富
- 价值: ⭐⭐⭐⭐ 对人脸隐私保护领域有明确的实用价值，方法设计思路可推广到其他对抗扰动保持任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](../../NeurIPS2025/image_generation/perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)

</div>

<!-- RELATED:END -->
