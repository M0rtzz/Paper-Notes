---
title: >-
  [论文解读] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification
description: >-
  [CVPR 2026][医学图像][多示例学习] 提出 MIL-PF，利用冻结的基础视觉编码器（DINOv2/MedSigLIP）预计算特征，再用仅约 40K 参数的轻量 MIL 头进行乳腺 X 线分类，在大规模 EMBED 数据集上达到 SOTA 性能，同时大幅降低训练成本。
tags:
  - CVPR 2026
  - 医学图像
  - 多示例学习
  - 乳腺X线摄影
  - 预计算特征
  - 基础模型
  - 注意力聚合
---

# MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**会议**: CVPR 2026  
**arXiv**: [2603.09374](https://arxiv.org/abs/2603.09374)  
**代码**: 有（论文承诺公开）  
**领域**: 医学图像  
**关键词**: 多示例学习, 乳腺X线摄影, 预计算特征, 基础模型, 注意力聚合

## 一句话总结

提出 MIL-PF，利用冻结的基础视觉编码器（DINOv2/MedSigLIP）预计算特征，再用仅约 40K 参数的轻量 MIL 头进行乳腺 X 线分类，在大规模 EMBED 数据集上达到 SOTA 性能，同时大幅降低训练成本。

## 研究背景与动机

乳腺 X 线摄影（Mammography）分类面临三重挑战：(1) **图像分辨率极高**（最大可达 4708×5844 像素），端到端微调大模型计算代价高昂；(2) **标注稀缺**——通常只有乳房级别的 BI-RADS 报告标签（弱监督），缺少像素级或区域级标注；(3) **多视图特性**——每个乳房的 CC/MLO 视图需要联合推理。现有方法（如 GMIC、FPN-AbMIL）要么需要端到端训练大骨干网络（参数量 1.7-22.9M），要么在大规模多样数据上表现不稳。

本文核心洞察：最新通用视觉基础模型（DINOv2-g、MedSigLIP）即使完全冻结，也能对分布外的乳腺 X 线域产生出色的表征——因此无需微调编码器，只需训练极轻量的任务特定聚合模块。

## 方法详解

### 整体框架

MIL-PF 分为两个阶段：

**阶段 1：特征预计算**——将每个乳房 bag $\mathcal{S}_i$ 中的图像通过冻结编码器 $\mathcal{F}$ 提取两种嵌入：(a) 全局嵌入 $\mathcal{G}_i = \{\mathcal{F}(I_i^{(n)})\}_n$——对每张完整图像编码一次；(b) 局部 tile 嵌入 $\mathcal{T}_i = \bigcup_n \bigcup_k \{\mathcal{F}(C_i^{(n)(k)})\}$——将每张图像切成编码器输入大小的 tile 网格（448×448/518×518），丢弃纯背景 tile。最终形成嵌入数据集 $\mathcal{E} = \{(\mathcal{G}_i, \mathcal{T}_i, y_i)\}_i$。

**阶段 2：MIL Head 训练**——在预计算的 $\mathcal{E}$ 上训练轻量聚合头。全局流和局部流分别经各自的聚合器处理后拼接，通过最终分类层输出预测。

### 关键设计

1. **双流架构（Global + Local Stream）**: 全局流 $\mathcal{A}_\psi^G$ 聚合全图级别特征，捕捉乳腺整体组织结构等低频信号；局部流 $\mathcal{A}_\omega^T$ 聚合 tile 级别特征，捕捉稀疏分布的病灶等高频信号。推理时 $\hat{y}_i = h_\theta(\text{concat}(\mathcal{A}_\psi^G(\mathcal{G}_i), \mathcal{A}_\omega^T(\mathcal{T}_i)))$。两流的 MLP 将嵌入从编码器维度投影到 16 维再到 8 维。这种后期融合策略虽然简单，但已足够有效。

2. **Perceiver 交叉注意力局部聚合器**: 对于局部流，关键挑战是病灶 tile 极其稀疏——绝大部分 tile 是正常组织。Mean pooling 会被背景稀释信号，max pooling 只能捕获单个最显著 tile，self-attention 侧重建模 tile 间相互依赖关系而非选择相关 tile。MIL-PF 采用 **Perceiver-style 交叉注意力**：一个可训练的潜在向量 $z$ 作为 query，对所有局部 tile 嵌入（Key/Value）做交叉注意力 $\text{softmax}(zK^T)V$，将最相关信息"拉"入一个汇总向量。仅用单个 latent query 就足够，无需标准 Perceiver 中的多 latent。

3. **冻结编码器 + 预计算范式**: 编码器 $\mathcal{F}$ 完全冻结，所有 bag 的嵌入预先计算并存储。MIL 头仅有约 40K 可训练参数，整个 EMBED 数据集（~50万张图）的训练集可装入单个 batch（A100 40GB），单次训练仅需 5-7 分钟。这使得 (a) 36 次独立运行选最优成为可能；(b) 编码器可灵活替换；(c) 计算资源需求极低。

### 损失函数 / 训练策略

- Binary Cross-Entropy 损失
- 数据划分：70%/10%/20%（训/验/测），按 BI-RADS 值分层，严格防止患者泄漏
- 每个实验跑 36 次独立运行，以最佳验证 AUC 选模型
- 分类不使用 overlapping tiles；注意力图计算使用 75% overlap
- BI-RADS 1 为阴性，BI-RADS 4/5/6 为阳性

## 实验关键数据

### 主实验

| 数据集 | 指标 | MIL-PF (最优) | 之前SOTA | 提升 |
|--------|------|--------------|---------|------|
| EMBED (BI-RADS) | AUC | **0.918** (MedSigLIP, max) | 0.875 (SILIL GMIC) | +4.3% |
| EMBED (BI-RADS) | Spec@Sens=0.9 | **0.762** (DINOv2, attn) | 0.566 (SILIL GMIC) | +19.6% |
| VinDr (BI-RADS) | AUC | 0.911 (MedSigLIP, attn) | **0.920** (FPN-AbMIL) | -0.9% |
| VinDr (BI-RADS) | Spec@Sens=0.9 | **0.792** (DINOv2, attn) | 0.720 (FPN-AbMIL) | +7.2% |
| RSNA (Cancer) | AUC | **0.925** (MedSigLIP, max) | 0.914 (FPN-AbMIL mean) | +1.1% |
| VinDr (Calcification) | AUC | **0.967** (两个编码器均达) | 0.962 (FPN-AbMIL mean) | +0.5% |

### 消融实验

| 配置 | EMBED AUC | 说明 |
|------|-----------|------|
| DINOv2-g (attn) | 0.916 | 通用编码器表现极强 |
| MedSigLIP (attn) | 0.914 | 医学编码器略低但接近 |
| MammoCLIP (专门乳腺X线) | 较低 | 领域专用编码器反而泛化差 |
| $\mathcal{A}^T$: max pooling | 0.905/0.918 | 简单但已有效 |
| $\mathcal{A}^T$: attention | 0.916/0.914 | 注意力聚合略优且更稳定 |
| 可训练参数 | ~40K/50K | 仅为对比方法的 1/35~1/450 |

### 关键发现

- **通用基础模型惊人地好**: DINOv2 和 MedSigLIP 在完全冻结的情况下，对分布外的乳腺 X 线域的泛化能力超越了专门在乳腺 X 线上预训练的 MammoCLIP
- **在最大规模数据集上优势最明显**: MIL-PF 在 EMBED（~50万张，最大最多样的公开乳腺X线数据集）上优势显著，说明其鲁棒性来自表征和聚合设计而非数据量
- **Spec@Sens=0.9 指标大幅领先**: 临床上最关注的"高灵敏度下的特异性"指标提升尤为突出，有直接临床价值
- **检测性能的局限**: 注意力图在中/大病灶上有竞争力，但因 tile 粒度较粗（448/518px），小病灶的 IoU@0.25 不佳

## 亮点与洞察

- **极简主义哲学**: 40K 参数就能达到 SOTA——当编码器足够强时，不需要复杂的端到端微调
- **预计算范式的复兴**: 利用现代基础模型的强表征能力，将"特征预计算+轻量头"这一经典范式带回前沿，对计算资源有限的研究组尤其有意义
- **MIL 问题形式化**: 对乳腺 X 线 MIL 问题的嵌套层次结构（bag → image → tiles → ROIs）和双流信号（全局组织 vs 局部病灶）做了清晰的形式化定义
- **工程友好**: 5-7 分钟训练 + 36 次运行选优 的范式，让超参数搜索和模型选择变得极其高效

## 局限与展望

- 检测精度受 tile 大小限制（448/518px），小病灶定位不准，可考虑多尺度 tile 或更细粒度的分割
- 全局流和局部流采用简单拼接的后期融合，未建模两者间的交互——早期或中间融合可能带来提升
- 依赖于预训练编码器的表征质量——如果编码器对特定异常类型不敏感，MIL 头无法弥补
- 单 latent query 的 Perceiver 可能限制了对多个独立病灶的同时关注能力
- 未探索与放射学报告文本的联合建模

## 相关工作与启发

- GMIC [Shen et al.] 用 14.1M 参数端到端训练，FPN-AbMIL [Mourão et al.] 用 1.76M——MIL-PF 用 40K 参数达到或超越它们，说明表征质量 > 下游头的复杂度
- DINOv2 的自监督预训练+大规模策展数据使其成为通用视觉特征提取器的新标杆
- 启发：对于高分辨率医学影像，"冻结基础模型+任务特定轻量头"可能是比端到端微调更实用的范式

## 评分

- 新颖性: ⭐⭐⭐ 技术组件均非全新（MIL+冻结编码器+Perceiver），但组合方式和"少即是多"的理念有价值
- 实验充分度: ⭐⭐⭐⭐ 三个数据集（EMBED/VinDr/RSNA），多编码器对比，多基线，36次运行
- 写作质量: ⭐⭐⭐⭐ MIL问题形式化清晰，设计选择有据可依
- 价值: ⭐⭐⭐⭐ 对资源受限的医学AI研究组有直接实用价值，工程意义大

<!-- RELATED:START -->

## 相关论文

- [LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)
- [Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segment.md)
- [Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiationinduced_cont.md)

<!-- RELATED:END -->
