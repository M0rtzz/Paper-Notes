---
title: >-
  [论文解读] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation
description: >-
  [CVPR 2026][医学图像][MRI合成] 提出 MSG-LDM，在潜在扩散模型中引入多尺度结构-风格解耦机制，通过高频注入、多模态结构特征融合和结构感知损失，实现缺失模态场景下保留解剖结构和精细细节的多模态 MRI 合成。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "MRI合成"
  - "潜在扩散模型"
  - "结构引导"
  - "风格-结构解耦"
  - "缺失模态"
---

# Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation

**会议**: CVPR 2026  
**arXiv**: [2603.12581](https://arxiv.org/abs/2603.12581)  
**代码**: [有](https://github.com/ziyi-start/MSG-LDM)  
**领域**: 医学图像  
**关键词**: MRI合成, 潜在扩散模型, 结构引导, 风格-结构解耦, 缺失模态

## 一句话总结

提出 MSG-LDM，在潜在扩散模型中引入多尺度结构-风格解耦机制，通过高频注入、多模态结构特征融合和结构感知损失，实现缺失模态场景下保留解剖结构和精细细节的多模态 MRI 合成。

## 研究背景与动机

### 1. 领域现状
多模态 MRI（T1、T2、T1CE、FLAIR）提供互补的解剖和病理信息，广泛用于脑肿瘤分割和病变分析。但临床中由于采集时间长、患者耐受性差、设备限制等原因，完整的多模态数据常常不可得。

### 2. 痛点
扩散模型已在 MRI 合成中超越 GAN，但现有方法仍存在三个问题：(1) 解剖结构可能失真；(2) 高频细节（边缘、纹理）退化；(3) 结构信息与模态特定风格纠缠，限制合成保真度和一致性。

### 3. 核心矛盾
传统扩散模型缺乏结构感知能力——如论文 Fig.1 所示，标准扩散去噪过程中结构重建不稳定且低效。需要显式结构先验来加速生成并保持解剖保真度。

### 4. 切入角度
在潜在空间中显式分离模态不变的结构特征和模态特定的风格特征，将结构先验注入扩散过程。

## 方法详解

### 整体框架

MSG-LDM 要解决的是缺失模态下的多模态 MRI 合成：手里只有部分模态（如 T1、T2、FLAIR），要补出缺的那个（如 T1CE），且必须保住解剖结构和高频细节不失真。它的核心思路是在 VAE 潜在空间里把"模态不变的结构"和"模态特定的风格"显式拆开，只用纯净的结构特征去引导扩散，避免风格纠缠污染结构。

整个网络由四块拼起来。每个模态各有一套**结构编码器** $E_j^{\mathrm{str}}$（内含 HFIB）、**风格编码器** $E_j^{\mathrm{sty}}$ 和**重建解码器** $D_j^{\mathrm{rec}}$，负责把该模态的图像分别编码成结构特征和风格特征；所有模态**共享一个分割解码器** $D_{\mathrm{seg}}$，用分割这个辅助任务逼着各模态的结构特征落到同一个模态不变的空间里。各模态的结构特征先经 **MMSF** 跨模态融合、再经 **MSSE** 多尺度增强，汇成一份统一结构表示 $F_s$，最后由**潜在扩散模型**以 $F_s$ 为条件去噪，生成缺失模态。

### 关键设计

**1. 高频注入模块（HFIB）：把编码器丢掉的边缘纹理捡回来。**

结构编码器用 ViT/CNN 提特征时有个老毛病——倾向于压缩高频信息，而医学图像里恰恰是边缘和纹理这些高频成分对诊断最关键，丢了它们合成图就会糊掉。HFIB 的做法是在结构编码器的每个尺度上单独把高频成分拎出来再补回去：对第 $l$ 层内容特征 $C^l$，先用一个**可学习的动态高斯滤波器** $\mathcal{G}_{\theta_l}$ 滤出低频分量，原特征减去低频得到的残差就是高频分量，再把高频叠加回原特征：

$$C_{\mathrm{high}}^l = C^l - \mathcal{G}_{\theta_l}(C^l), \quad S_j^l = C^l + C_{\mathrm{high}}^l$$

关键在于滤波器参数 $\theta_l$ 是可学习的——比起固定核的高斯模糊，它能按层、按数据自适应地决定哪些频段算"低频"该被压、哪些边缘细节该被强化，从而把容易被编码器抹掉的高频结构稳稳保留在 $S_j^l$ 里。

**2. 多模态结构特征融合（MMSF）：让可用模态各取所长、不让某一个说了算。**

缺失模态场景下手里只有 $M$ 个可用模态，而 T1、FLAIR 这些模态对不同组织的敏感度并不一样（T1 看解剖、FLAIR 看病灶），它们的结构信息是互补的——简单相加或拼接容易让某个强模态主导、淹没其他模态的有用结构。MMSF 在每个尺度 $l$ 上用一个 Sigmoid 门控网络给每个模态算一个注意力权重 $w_j \in [0,1]$，按权重加权求和后再过一层可学习卷积得到该尺度的融合结构特征：

$$F_l = \mathrm{Fusion}\left(\sum_{j=1}^{M} w_j S_j^{(l)}\right)$$

这样每个模态贡献多少由门控自适应决定，哪个模态在当前区域结构更可靠就给更大权重，避免了固定权重下的模态偏置。

**3. 多尺度结构特征增强（MSSE）：把全局布局和精细结构拧成一份统一表示。**

融合后每个尺度都有一份结构特征 $F_l$，但它们各管一摊：低尺度（$F_1 \dots F_{L-1}$）捕获的是全局解剖布局，最高尺度 $F_L$ 保留的是精细结构，直接丢给扩散模型当条件并不好用。MSSE 把低尺度特征统一对齐再补进最高层——低尺度特征先经 $1 \times 1$ 卷积投影、上采样到最高尺度的分辨率，然后以 $F_L$ 为 query 做交叉注意力，把对齐后的多尺度信息选择性地注入最高层表示：

$$F_s = F_L + \alpha \, \mathrm{Attn}\left(F_L, \sum_{l=1}^{L-1} \mathrm{Up}(\mathrm{Proj}(F_l))\right)$$

交叉注意力的好处是高层不会被低层信息生硬覆盖，而是按需从低尺度里"取用"全局布局来补全自己缺的上下文，最终得到的统一结构表示 $F_s$ 同时携带宏观解剖和微观细节，作为条件去引导后续的潜在扩散去噪。

### 损失函数 / 训练策略

总损失：$L_{\text{total}} = L_{\text{seg}} + \lambda_1 L_{\text{sc}} + \lambda_2 L_{\text{sa}} + \lambda_3 L_{\text{ldm}}$

- **$L_{\text{seg}}$**：辅助分割损失，确保结构特征模态不变
- **$L_{\text{sc}}$（风格一致性损失）**：类似对比学习——同模态风格拉近，异模态风格推开，抑制模态特定风格对结构的污染

$$L_{\text{sc}} = -\frac{1}{(M \times B)^2} \sum_{p,q} [T_{pq} \log \sigma(z_{pq}) + (1-T_{pq}) \log \sigma(-z_{pq})]$$

- **$L_{\text{sa}}$（结构感知损失）**：$L_1$ 重建损失 + 频域 SSIM 损失（DCT 变换后比较幅度谱一致性）

$$L_{\text{sa}} = L_{\text{rec}} + L_{\text{freq}}, \quad L_{\text{freq}} = 1 - \text{SSIM}(|\mathcal{D}(\hat{X}_j)|, |\mathcal{D}(X_j)|)$$

- **$L_{\text{ldm}}$**：标准去噪扩散损失

训练配置：PyTorch 2.1.0，Adam（lr=$1 \times 10^{-4}$），batch size 9，3×NVIDIA 4090，100 epochs。

## 实验关键数据

### 主实验

**表1：BraTS2020 数据集（$\bar{M}=3$，三个可用模态生成第四个）**

| 方法 | T1 PSNR/SSIM | T2 PSNR/SSIM | T1CE PSNR/SSIM | FLAIR PSNR/SSIM |
|------|-------------|-------------|---------------|----------------|
| MM-GAN | 27.35/92.32 | 27.85/93.18 | 28.65/94.19 | 27.95/92.95 |
| SynDiff | 28.95/93.34 | 29.36/93.95 | 30.65/94.86 | 29.62/93.23 |
| MISA-LDM | 29.01/93.86 | 29.66/94.12 | 30.68/95.62 | 29.66/93.28 |
| **MSG-LDM** | **30.26/94.37** | **30.33/94.38** | **31.35/96.29** | **29.68/93.62** |

**表2：WMH 数据集**

| 方法 | FLAIR→T1 PSNR/SSIM | T1→FLAIR PSNR/SSIM |
|------|---------------------|---------------------|
| MISA-LDM | 28.86/95.23 | 28.10/94.65 |
| **MSG-LDM** | **29.16/96.80** | **28.38/95.55** |

MSG-LDM 在所有设置下全面 SOTA。BraTS2020 上 PSNR 平均提升 ~1 dB，SSIM 提升 ~0.5%。

### 消融实验

| 配置 | PSNR | SSIM% | Dice% |
|------|------|-------|-------|
| w/o 解耦+MMSF | 27.92 | 92.41 | 85.03 |
| w/o HFIB | 28.17 | 92.68 | 85.41 |
| w/o MSSE | 29.04 | 93.28 | 86.55 |
| w/o $L_{\text{sa}}$ | 27.36 | 91.82 | 84.27 |
| w/o $L_{\text{sc}}$ | 27.11 | 91.54 | 83.89 |
| **完整模型** | **29.68** | **93.62** | **87.60** |

### 关键发现

1. **风格一致性损失贡献最大**：移除 $L_{\text{sc}}$ 后 PSNR 下降 2.57 dB（29.68→27.11），说明风格干扰是 MRI 合成的核心挑战
2. **结构感知损失同样关键**：移除 $L_{\text{sa}}$ 后 Dice 下降 3.33%，频域约束对保持结构一致性不可或缺
3. **HFIB 提升细节保真**：移除后 SSIM 下降 0.94%，高频注入对纹理/边缘质量重要
4. **可用模态越多越好**：随可用模态数从 1 增加到 3，合成质量持续改善
5. **结构引导加速去噪**：如 Fig.1 所示，有结构先验的去噪在中间步骤就已重建出清晰结构，无先验的仍模糊

## 亮点与洞察

1. **风格-结构解耦的系统性设计**：不仅在编码端解耦（独立编码器），还在损失端双重约束（$L_{\text{sc}}$ 推开风格、$L_{\text{sa}}$ 保持结构），形成闭环
2. **HFIB 的简洁有效**：可学习动态高斯滤波→残差提取高频→重新注入，无额外参数开销，即插即用
3. **共享分割解码器巧妙利用**：作为辅助任务强制结构特征模态不变，这是一种间接但有效的正则化
4. **频域 SSIM 损失新颖**：DCT 变换后比较幅度谱的 SSIM，同时约束全局布局和频率分布

## 局限与展望

1. **仅验证脑 MRI**：BraTS2020（肿瘤）和 WMH（白质高信号），其他解剖区域/疾病未验证
2. **2D 处理**：将 3D MRI 切片为 2D（192×192）处理，丢失了体积上下文
3. **固定模态数**：假设模态集合固定（T1/T2/T1CE/FLAIR），无法动态适应新模态
4. **训练成本**：每个模态单独的编码器/解码器，$M$ 增大时参数线性增长
5. **下游评估有限**：Dice 仅评估分割，缺乏放射科医生主观评估和临床诊断任务验证

## 相关工作与启发

- **从 GAN 到 Diffusion 的演进**：MM-GAN → SynDiff → MISA-LDM → MSG-LDM，扩散模型在结构保真度上持续进步
- **结构引导思想**：将结构先验注入生成过程是医学图像合成的关键——不能让模型"自由发挥"，需要约束解剖一致性
- **频域损失的启发**：DCT+SSIM 的组合可推广到其他医学图像生成/超分任务

## 评分

- 新颖性: ⭐⭐⭐ 各组件（HFIB、MMSF、MSSE）单独看设计中规中矩，但系统集成和风格-结构解耦的完整性较好
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多种缺失模态场景、完整消融和可视化
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融分析系统
- 价值: ⭐⭐⭐ MRI 合成的增量改进，核心 insight（结构引导扩散）有一定通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](../../AAAI2026/medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)

</div>

<!-- RELATED:END -->
