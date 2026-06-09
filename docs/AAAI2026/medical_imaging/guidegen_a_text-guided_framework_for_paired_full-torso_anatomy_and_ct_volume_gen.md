---
title: >-
  [论文解读] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation
description: >-
  [医学图像] GuideGen 提出了一个仅需文本输入的可控框架，通过分类扩散模型合成全躯干解剖掩码，结合解剖感知高动态范围自编码器和潜在特征生成器，生成配对的全躯干 CT 体积，为下游分割任务提供高质量合成训练数据。
tags:
  - "医学图像"
---

# GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation

## 论文信息

- **会议**: AAAI 2026
- **arXiv**: [2403.07247](https://arxiv.org/abs/2403.07247)
- **代码**: [https://github.com/OvO1111/GuideGen](https://github.com/OvO1111/GuideGen)
- **领域**: 医学图像
- **关键词**: CT生成, 文本引导, 解剖掩码合成, 全躯干, 分类扩散模型, 高动态范围, 分割数据增强

## 一句话总结

GuideGen 提出了一个仅需文本输入的可控框架，通过分类扩散模型合成全躯干解剖掩码，结合解剖感知高动态范围自编码器和潜在特征生成器，生成配对的全躯干 CT 体积，为下游分割任务提供高质量合成训练数据。

## 研究背景与动机

大规模医学图像数据集的获取受限于隐私问题和标注成本，条件生成模型是有前景的解决方案。现有方法存在以下不足：

**语义条件方法**（如 MAISI）：样本多样性有限，依赖成本高昂的精细解剖掩码

**文本条件方法**（如 GenerateCT）：虽灵活但难以完整捕捉解剖结构间的精确空间关系

**混合方法的局限**：Kim et al. 需推理时同时输入掩码和文本（限制适用性），MedSyn 仅覆盖胸部（无法扩展到全躯干的复杂解剖）

**下游可用性不足**：现有文本引导生成主要服务于分类任务，对分割任务的支持不足

GuideGen 的核心目标：仅文本输入 → 自动生成全躯干解剖掩码 + 对应 CT → 构建分割训练数据集。

## 方法详解

### 整体框架（三阶段管线）

1. **文本条件语义合成器 (TCSS)**：文本 → 离散解剖掩码
2. **解剖感知 HDR 自编码器**：CT → 高保真潜在特征
3. **潜在引导特征生成器**：语义 + 文本潜在特征 → CT 潜在特征 → CT 图像

### 关键设计

#### 1. 文本条件语义合成器 (TCSS)

**消歧分类建模**：

现有方法（如 MedGen3D）基于连续扩散模型生成掩码，在语义边界附近无法捕捉锐利过渡，导致歧义。TCSS 采用**分类扩散模型**直接建模离散标签索引：

- 扩散变量 $\mathbf{x}_0 = \mathbf{m} \in \{1,...,N\}^{H \times W \times D}$（$N$ 为语义类别数）
- 前向过程将类别标签逐步转化为均匀分类噪声：
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{C}_N(\mathbf{x}_t; (1-\beta_t)\mathbf{e}(\mathbf{x}_{t-1}) + \beta_t \cdot \frac{\mathbf{1}}{N})$$
- 训练目标为 KL 散度最小化，使用重参数化版本直接预测 $\mathbf{x}_0$

**知识注入模块**：

- 使用 ERNIE 系列医学文本编码器将结构化提示映射到潜在空间
- 通过可学习的任务特定查询 $Q$ 与 Transformer 解码器块交互，提取任务相关响应 $R_{\text{task}}$
- 进一步派生逐层响应 $R_{\text{layer}}$，分别关注全局解剖或局部结构
- 通过交叉注意力将逐层引导注入扩散骨干

#### 2. 解剖感知高动态范围 (HDR) 自编码器

**解剖保留**：
- 采用金字塔式方法将语义掩码重采样到自编码器各层潜在特征的分辨率并拼接
- 帮助模型在编码时关注语义细节，在解码时重建语义准确的图像
- 解决小肿瘤在低分辨率潜在空间中被忽视的问题

**HDR 适配**：
- 不截断 CT 强度到固定范围（如仅肺窗或腹窗），而是保留全动态范围
- 设计强度变换模块 $h(\mathbf{x})$，随机采样窗中心 $w_c$ 和窗宽 $w_r$：
$$h(\mathbf{x}) = k \max\{\min\{\frac{\mathbf{x} - w_c + w_r}{2w_r}, 1\}, 0\} + b$$
- 可学习系数 $k$, $b$ 将截断结果映射回输入空间

**训练损失**：
$$\mathcal{L}_2 = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{perc}} + \mathcal{L}_{\text{disc}}(\mathbf{D}_f) + \mathcal{L}_{\text{disc}}(\mathbf{D}_v)$$

包含帧判别器 $\mathbf{D}_f$（随机切片）和体积判别器 $\mathbf{D}_v$（3D），以及 VGG-16 感知损失。

#### 3. 潜在引导特征生成器

- 标准高斯扩散模型在自编码器潜在空间操作
- 将重采样的语义掩码拼接到扩散变量 $\mathbf{z}_t$
- 通过知识注入和交叉注意力注入文本信息
- 训练目标：标准 $\epsilon$-预测损失
$$\mathcal{L}_3 = \mathbb{E}_{t,\boldsymbol{\epsilon}}[\|\boldsymbol{\epsilon} - f_\varphi(\mathbf{z}_t; \text{Resample}(\hat{\mathbf{m}}), \mathbf{p})\|_2^2]$$

### 损失函数

三阶段分别优化 $\mathcal{L}_1$（TCSS 的 KL 散度）、$\mathcal{L}_2$（HDR 自编码器的混合损失）、$\mathcal{L}_3$（潜在生成器的去噪损失）。

## 实验

### 数据构建

- **训练集**：12 个公开 TCIA 数据集 + 1 个私有结直肠癌数据集 (RJ)，共 4534 训练 / 1179 验证
- **推理评估**：BTCV（多器官分割）、AMOS22（腹部多器官）、MSD-LU/CO（肿瘤）、KiTS21（肾肿瘤）
- 文本提示由医学 LLM 从结构化记录生成，格式："The patient is {demographics}. In this imaging, ..."
- 伪标签由 TotalSegmentator 和 nnU-Net 生成

### 主实验表格

**掩码生成质量**：

| 方法 | 参数 | 全解剖 LPIPS↓/FID↓ | 肿瘤 LPIPS↓/FID↓ |
|------|------|------------------|-----------------|
| MedGen3D | 48.8M | 0.70/201 | 0.29/33.5 |
| LDM | 115.2M | 0.67/98.6 | 0.30/69.1 |
| **GuideGen** | 51.5M | **0.33/7.1** | **0.29/27.9** |

全解剖 FID 从次优的 98.6 降至 7.1，提升巨大。

**CT 图像生成质量**：

| 方法 | 推理条件 | LPIPS↓ | FID↓ | FVD↓ |
|------|---------|--------|------|------|
| MedSyn (text-only) | 文本 | 0.396 | 50.0 | 2012 |
| MedSyn (mask+text) | 掩码+文本 | 0.282 | 26.7 | 1288 |
| MAISI | 掩码 | 0.393 | 54.6 | 1791 |
| **GuideGen** | **仅文本** | **0.248/0.256** | **20.2/19.4** | **791/745** |

GuideGen 仅文本输入即超越所有需要掩码的方法。

**下游多器官分割（DSC）**：

| 方法 | 训练样本 | 脾 | 肾 | 肝 | 胃 | 胰腺 | 平均 |
|------|---------|-----|-----|-----|-----|------|------|
| Real | 24/240 | 0.92/0.95 | 0.79/0.94 | 0.94/0.96 | 0.86/0.89 | 0.70/0.81 | 0.74/0.84 |
| MAISI | 200 | 0.91/0.83 | 0.89/0.84 | 0.94/0.91 | 0.80/0.74 | 0.61/0.60 | 0.69/0.65 |
| **GuideGen** | 200 | **0.96/0.95** | **0.91/0.92** | **0.98/0.95** | **0.90/0.90** | **0.76/0.70** | **0.79/0.78** |

在 BTCV 上合成数据训练的模型甚至超过真实数据训练（0.79 vs 0.74），令人瞩目。

### 消融实验

| 配置 | LPIPS↓ | FID↓ | DSC↑ | Acc.↑ |
|------|--------|------|------|-------|
| 无掩码输入 | 0.42 | 54.3 | - | 0.32 |
| 无知识注入 | 0.26 | 21.7 | 0.25 | 0.57 |
| 无解剖保留 | 0.27 | 32.4 | 0.40 | 0.61 |
| 无 HDR 适配 | 0.33 | 40.9 | 0.36 | 0.64 |
| **完整 GuideGen** | **0.25** | **20.2** | **0.52** | **0.69** |

三个核心组件（TCSS、知识注入、HDR）均不可或缺。

### 关键发现

1. **分类扩散 vs 连续扩散**：分类建模在语义类别数 $N$ 大时优势明显（全解剖 FID 7.1 vs 98.6），类别少时差距缩小
2. **掩码质量决定 CT 质量**：CT 生成质量与输入掩码质量正相关
3. **合成数据的惊人效果**：200 例合成数据在 BTCV 上超越 24 例真实数据，在肺肿瘤分割上合成数据甚至优于真实数据（DSC 0.71 vs 0.69）
4. **增强效果持续增长**：生成样本量从 100→200→500→1000 持续提升下游性能

## 亮点与洞察

1. **全躯干覆盖**：首次实现从胸部到盆腔的全躯干 CT 生成，大幅扩展了适用场景
2. **分类扩散的消歧思路**：优雅地解决了连续扩散模型在语义边界的歧义问题，对离散标签建模更自然
3. **HDR 适配设计巧妙**：模拟医生使用不同窗宽窗位查看不同解剖区域，训练时随机采样强度窗口，保留全动态范围信息
4. **从仅文本到分割数据**：完整闭环——用户输入文本描述 → 自动生成掩码+CT → 直接用于训练分割模型
5. **知识注入优于简单交叉注意力**：通过可学习查询提取任务相关信息，比直接交叉注意力更精确地聚焦关键描述（如肿瘤位置 > 人口统计）

## 局限性

- 无法直接从自由文本输入生成（需要结构化提示格式，由 LLM 转换）
- 生成分辨率受限于 $128^3$（掩码）和 $256^3$（CT），低于临床分辨率
- 金字塔式解剖标签注入增加了自编码器的计算开销
- 结直肠癌和肾癌的肿瘤分割 DSC 仍显著低于真实数据（CO: 0.21 vs 0.47, KI: 0.64 vs 0.72）
- 训练数据依赖伪标签（TotalSegmentator），标签噪声可能影响上界

## 相关工作

- **语义引导 CT 生成**：MAISI, MedGen3D, Label-efficient GAN
- **文本引导 CT 生成**：GenerateCT, MedSyn, RoentGen
- **混合条件生成**：Kim et al. (需配对输入), MedSyn (null mask)
- **分类扩散模型**：Argmax Flows, Stochastic Segmentation

## 评分

⭐⭐⭐⭐⭐ (5/5)

- 问题定义有重大实际价值（文本驱动的医学数据集自动构建）
- 三阶段设计逻辑严密，每个组件都有清晰的设计动机
- 实验评估维度全面：生成质量 + 条件对齐 + 下游分割可用性
- 合成数据超越真实数据的结果极具说服力
- 唯一的局限在于分辨率约束和结构化提示依赖，但不影响学术贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)

</div>

<!-- RELATED:END -->
