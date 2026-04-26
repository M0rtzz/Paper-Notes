---
title: >-
  [论文解读] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis
description: >-
  [医学图像] > 提出 ViCTr 两阶段框架，结合 Rectified Flow 与 Tweedie 校正的扩散过程实现高保真的病理感知医学图像合成，将推理步数从50步降至3-4步，并首次实现分级严重程度的腹部MRI病理合成。
tags:
  - 医学图像
---

# ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2505.04963](https://arxiv.org/abs/2505.04963)
- **作者**: Onkar Susladkar, Gayatri Deshmukh, Yalcin Tur, Gorkem Durak, Ulas Bagci (Northwestern University, Stanford University, UIUC)
- **代码**: [GitHub](https://github.com/Onkarsus13/ViCTr-2D) / [权重](https://huggingface.co/onkarsus13/ViCTr-2D)
- **领域**: 医学影像 / 医学图像合成
- **关键词**: 医学图像合成, Rectified Flow, Tweedie's Formula, 病理感知, 肝硬化, 数据增强, LoRA

## 一句话总结

> 提出 ViCTr 两阶段框架，结合 Rectified Flow 与 Tweedie 校正的扩散过程实现高保真的病理感知医学图像合成，将推理步数从50步降至3-4步，并首次实现分级严重程度的腹部MRI病理合成。

## 研究背景与动机

### 问题定义
医学图像合成旨在生成具有解剖学真实性和病理学多样性的合成医学影像，用于数据增强以缓解医学影像数据稀缺问题。

### 现有挑战

**数据稀缺**: 隐私法规、机构间数据碎片化和互操作性限制导致医学影像数据严重不足

**解剖保真不足**: 现有方法难以在保持解剖结构准确性的同时建模病理特征

**弥散性病理困难**: 肝硬化等弥散性病变涉及多器官系统的细微组织变化，远比肿瘤等局灶性病变复杂

**采样效率低**: 传统扩散模型需要50步以上的多步采样，计算开销大

**缺少病理控制**: 现有方法无法精细控制合成病理的严重程度

### 核心动机
- Rectified Flow 提供近线性的采样轨迹，减少步数
- Tweedie's Formula 可校正采样偏差，提升初始化精度
- 两者结合 + 两阶段训练 = 高效、高保真、病理可控的合成

## 方法详解

### 整体框架
ViCTr 包含两个训练阶段：
1. **Stage 1 — 预训练**: 在 ATLAS-8k 数据集上建立解剖学先验
2. **Stage 2 — 微调**: 使用 LoRA 适配下游任务（CT/MRI生成 + 病理合成）

### 关键设计

#### 1. Rectified Flow 轨迹
定义插值: $x_t = (1-t)x_0 + tx_1$，其中 $x_0 \sim p_0$（噪声），$x_1 \sim p_{target}$（目标数据）

速度模型训练:
$$\hat{\theta} = \arg\min_\theta \mathbb{E}_{t \sim \text{Uniform}(0,1)} \left[ \|(x_1 - x_0) - v_\theta(x_t, t)\|^2 \right]$$

一步蒸馏: $\hat{\mathcal{T}}(x_0) = x_0 + v(x_0, 0)$

#### 2. Tweedie's Formula 校正
将 Tweedie 校正融入 Rectified Flow ODE:
$$dx_t = v_{\hat{\theta}}(x_t, t)dt + (1 - \bar{\alpha}_t)\nabla_{x_t} \log p(x_t) dt$$

额外的 score 项 $(1 - \bar{\alpha}_t)\nabla_{x_t} \log p(x_t)$ 纠正采样偏差，使 $x_t$ 更准确地趋向目标分布。

一步采样扩展:
$$\hat{\mathcal{T}}(x_0) = x_0 + v(x_0, 0) + (1 - \bar{\alpha}_0)\nabla_{x_0} \log p(x_0)$$

#### 3. Stage 1 — ATLAS-8k 预训练

- **输入**: CT图像 $X_I$、分割掩码 $X_S$、文本提示 $X_p$
- **编码**: 冻结 VAE 编码器提取潜在表示 $Z_o$, $Z_s$；预训练文本编码器提取 $Z_p$
- **正向扩散**: $P(Z_t|Z_o) = (1-t) \cdot Z_o + t \cdot \epsilon_{true}$
- **反向扩散**: $P(Z_{t-1}|Z_t, Z_s, Z_p, t) = Z_t + \delta T \times \phi_\theta(Z_t, Z_s, Z_p, t)$
- **EWC (弹性权重巩固)**: 选择性解冻关键层，保持模型稳定性

损失函数:
$$L_{diff} = -|\phi_\theta(Z_t, Z_s, Z_p, t) - (\epsilon_{true} - Z_o)|^2$$

复合损失: $L_{diff} + L_2 + L_{SSIM}$

#### 4. Stage 2 — LoRA 微调
- **双网络**: $\phi_{base}$（冻结，保留解剖知识）+ $\phi_{adapt}$（LoRA 可训练）
- **一致性损失**: $L_{consistency}$ 对齐两网络的中间输出
- **时间一致性**: $L_{temporal}$ 保证反向扩散时间步间平滑过渡
- **空间一致性**: $L_{spatial}$ 确保输出重建对齐

#### 5. 病理生成
- 数据集: CirrMRI600+（T1/T2 MRI）
- 肝脏分割掩码 + 文本提示指定严重程度: "low", "mild", "severe"
- 渐进式病理控制

### 支持多种扩散模型骨干

| 扩散方法 | Denoiser | 文本编码器 |
|----------|----------|-----------|
| Stable Diffusion | UNet | CLIP-B/16 |
| Pixart-alpha | DiT | T5-XXXL |
| SDXL | Dual UNet | CLIP-L/14 |
| Flux | MultiModal Transformer | T5-XXXL + CLIP |
| SD-3 | MultiModal Transformer | T5-XXXL + CLIP |

## 实验关键数据

### 主实验 — 合成数据生成质量 (FID/MFID)

| 骨干 | BTCV(CT) Vanilla/ViCTr | AMOS(MRI) Vanilla/ViCTr | CirrMRI600+ Vanilla/ViCTr | 步数 Vanilla→ViCTr |
|------|------|------|------|------|
| Stable Diffusion | 25.44/19.67 → **21.98/19.02** | 25.43/21.76 → **20.37/19.11** | 28.34/23.43 → **25.57/21.46** | 40→4 |
| SDXL | 23.47/18.21 → **20.33/17.44** | 24.11/20.23 → **19.44/18.45** | 27.34/22.11 → **24.02/20.76** | 30→4 |
| SD-3 | 19.07/16.22 → **17.37/16.02** | 22.32/19.76 → **18.02/19.08** | 24.49/21.78 → **21.28/19.34** | 50→3 |
| Pixart-alpha | 21.32/17.09 → **19.22/16.96** | 23.78/20.04 → **18.76/18.56** | 26.06/20.07 → **23.04/18.92** | 25→3 |
| **Flux** | 15.52/15.01 → **13.28/14.08** | 19.02/18.28 → **15.55/16.58** | 22.46/18.88 → **19.96/17.01** | 30→3 |

ViCTr + Flux 在 CirrMRI600+ 上达到 **MFID 17.01**，比现有最优低 28%。

### 分割性能提升 (mDSC%↑ / mHD95↓)

| 分割模型 | 仅原始数据 | +增强 | +30% Vanilla合成 | +30% ViCTr合成 |
|----------|-----------|------|-------------|------------|
| **BTCV** |
| UNet | 76.72 | 78.45 | 79.32 | **81.22** |
| TransUNet | 85.52 | 87.01 | 87.54 | **89.78** |
| nnUNet | 80.48 | 82.54 | 83.37 | **85.19** |
| MedSegDiff | 87.91 | 88.65 | 89.78 | **91.92** |
| **CirrMRI600+** |
| UNet | 68.74 | 69.38 | 70.12 | **73.39** |
| nnUNet | 71.02 | 72.49 | 73.56 | **78.89** |
| MedSegDiff | 76.92 | 77.11 | 78.03 | **81.37** |

nnUNet在CirrMRI600+上使用ViCTr合成数据后提升 **+7.87% mDSC** (+3.8% 相比vanilla合成)。

### 消融实验

| 消融项 | FID |
|--------|-----|
| w/o Stage-1 预训练 | 17.33 |
| w/o Tweedie 校正 | 18.78 |
| 使用 Reflow | 20.19 |
| 使用 Flow Straight and Fast | 21.37 |
| 使用 Distribution Matching Distillation | 22.33 |
| 仅 $L_{diff}$ | 18.77 |
| $L_{diff} + L_{spatial}$ | 18.21 |
| $L_{diff} + L_{consistency}$ | 17.02 |
| $L_{diff} + L_{spatial} + L_{consistency}$ | **15.55** |
| LoRA r=8 | 18.46 |
| LoRA r=16 | 17.52 |
| LoRA r=32 | 16.66 |
| **LoRA r=64 (最终)** | **15.55** |

### 关键发现
1. **Tweedie 校正关键**: 移除后 FID 从 15.55 升到 18.78，分布对齐明显下降
2. **预训练必不可少**: 省略 Stage 1 使 FID 从 15.55 升到 17.33
3. **一致性损失最重要**: 在三个损失组件中，$L_{consistency}$ 贡献最大（17.02 vs 18.77）
4. **LoRA rank越高越好**: r=64 表现最佳，高维适配空间提供更精细的参数更新
5. **推理效率**: 采样步数从50步降至3-4步，推理时间从18.98s降至2.78s (SD-3)
6. **放射科医师验证**: 3位放射科医师在 Visual Turing Test 中无法区分合成与真实肝硬化 MRI

## 亮点与洞察

1. **首次腹部MRI病理合成**: 首个实现分级严重程度控制的腹部MRI病理合成方法，填补重要空白
2. **理论创新**: 将 Tweedie's Formula 嵌入 Rectified Flow 框架是理论上的原创贡献，不是简单组合
3. **极致效率**: 从50步到3步的减少（10x加速），使临床部署成为可能
4. **广泛兼容**: 支持 5 种主流扩散模型骨干，证明方法的通用性
5. **双重验证**: 既有定量指标（FID/MFID/mDSC），又有放射科医师的定性验证
6. **两阶段范式优雅**: Stage 1 解剖先验 + Stage 2 LoRA 病理适配，巧妙平衡通用性和特异性

## 局限性

1. **2D切片级合成**: 当前仅处理2D切片，3D体积一致性未保证
2. **病理类型有限**: 仅验证了肝硬化，其他弥散性病变（如脂肪肝、纤维化）未测试
3. **分辨率限制**: 256×256 分辨率可能不足以捕捉某些细微病理特征
4. **训练开销**: 在 8×8 A100 上预训练约52小时，资源需求较高
5. **VAE瓶颈**: 依赖冻结的VAE编码/解码器，其信息损失会影响最终合成质量

## 相关工作与启发

### 相关研究
- **医学扩散**: MedSegDiff, EMIT-Diff, DiNO-Diffusion
- **Rectified Flow**: ReFlow, Flow Straight and Fast, Distribution Matching Distillation
- **一致性模型**: Consistency Models
- **医学数据增强**: DiffuseMix, DreamDA, ControlPolypNet

### 启发
- Rectified Flow + Tweedie 校正的组合可推广到其他需要高保真少步采样的场景
- 两阶段"通用预训练 + LoRA特异性适配"的范式在数据稀缺领域特别有效
- 医学影像合成的评估需同时关注生成质量（FID）和下游任务效用（分割mDSC）

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/medical_imaging/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](../../CVPR2025/medical_imaging/noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](../../ICML2025/medical_imaging/scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](../../ICML2025/medical_imaging/deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)

<!-- RELATED:END -->
