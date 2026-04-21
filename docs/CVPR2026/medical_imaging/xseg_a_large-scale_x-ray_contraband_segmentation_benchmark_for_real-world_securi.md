---
title: >-
  [论文解读] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening
description: >-
  [CVPR 2026][医学图像][X光违禁品分割] 本文构建了目前最大的 X 光违禁品分割数据集 XSeg（98,644 张图像、295,932 个实例 mask、30 个细粒度类别），并提出域特化模型 APSAM，通过 Energy-Aware Encoder 利用 X 光双能量物理特性 + Adaptive Point Generator 智能扩展用户点击提示，mIoU 达 72.83%，比 SAM 微调高 4.96%。
tags:
  - CVPR 2026
  - 医学图像
  - X光违禁品分割
  - 安检数据集
  - SAM适配
  - 双能量编码器
  - 自适应点提示
---

# XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening

**会议**: CVPR 2026  
**arXiv**: [2604.03706](https://arxiv.org/abs/2604.03706)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: X光违禁品分割、安检数据集、SAM适配、双能量编码器、自适应点提示

## 一句话总结

本文构建了目前最大的 X 光违禁品分割数据集 XSeg（98,644 张图像、295,932 个实例 mask、30 个细粒度类别），并提出域特化模型 APSAM，通过 Energy-Aware Encoder 利用 X 光双能量物理特性 + Adaptive Point Generator 智能扩展用户点击提示，mIoU 达 72.83%，比 SAM 微调高 4.96%。

## 研究背景与动机

1. **领域现状**：X 光安检图像的违禁品检测是机场、地铁、物流中心的核心安全需求。现有 X 光数据集（SIXray、PIXray、PIDray）规模小（<5 万张）、类别少（<15 类）、且主要提供检测框而非分割 mask。
2. **现有痛点**：(1) 数据匮乏——最大的 PIDray 也只有 47K 图像且只覆盖 12 类；(2) SAM 等通用分割模型在 X 光域迁移效果差——X 光的色域、纹理与自然图像差异巨大；(3) 单点提示在复杂重叠场景下不够信息化——安检图像中物品高度重叠。
3. **核心矛盾**：X 光安检的精确分割需要大规模高质量标注数据和域适配的分割方法，但两者都严重不足。
4. **本文目标**：同时解决数据和方法两个瓶颈——构建大规模分割 benchmark + 设计域特化的 SAM 适配方案。
5. **切入角度**：X 光成像的物理特性提供了独特信号——双能量通道（高能/低能）可以区分不同材质，这是 RGB 图像不具备的领域先验。
6. **核心 idea**：EAE 利用 X 光 max/min 通道提取双能量特征初始化解码器；APG 将用户单点扩展为两个信息化提示点。

## 方法详解

### 整体框架

X 光图像 → SAM ViT-L 编码器提取图像特征 + EAE 提取双能量特征初始化解码器 token → 用户点击一个点 $p_0$ → APG 生成初始 mask → K-means 找两个代表性点 $(p_1, p_2)$ → SAM 解码器用增强的提示预测最终 mask。

### 关键设计

1. **Energy-Aware Encoder (EAE)**

    - 功能：从 X 光图像中提取双能量域先验知识
    - 核心思路：提取高能通道 $I_H = \max_c I(\cdot,\cdot,c)$ 和低能通道 $I_L = \min_c I(\cdot,\cdot,c)$，拼接后经三层 Conv+LayerNorm+GELU+MaxPool 编码。通过 channel-wise attention + top-k 特征选择生成初始化 query，替代 SAM 的随机 token 初始化
    - 设计动机：X 光的 RGB 通道实际编码了不同能量级别的透射信息——金属在高能通道和低能通道的响应差异很大，利用这个物理先验能帮助模型区分不同材质

2. **Adaptive Point Generator (APG)**

    - 功能：将用户单点击扩展为两个信息量更大的提示点
    - 核心思路：用单点 $p_0$ 先生成初始 soft mask $M_0$，提取 bounding box 后随机缩放（$s \sim \mathcal{U}(0.9, 1.1)$），在 box 内做 K-means (K=2) 聚类，取两个簇的最远点对 $(p_1^*, p_2^*)$ 作为新提示（若簇心距离不够则退化为随机采样）
    - 设计动机：安检物品常高度重叠，单点无法充分描述目标范围。APG 自适应找到两个代表性位置，提供更丰富的空间线索。消融显示 APG 比 2 个随机点高 1.59 mIoU

3. **XSeg 数据集构建**

    - 功能：提供大规模高质量 X 光分割基准
    - 核心思路：整合 114Xray + PIXray + PIDray + 真实安检图像，经分辨率/宽高比/清晰度过滤后从 ~150K 精选到 98,644 张。标注采用 MobileSAM 辅助 + 安检专家人工校验的闭环策略，5 轮迭代。30 个细粒度类别（如剪刀分为金属柄/塑料柄）
    - 设计动机：现有数据集的规模和标注粒度都不足以训练和评估可部署的安检分割模型

### 损失函数 / 训练策略

标准 SAM 训练损失（Dice + Cross-Entropy）。ViT-L/14 骨干，512×512 输入，AdamW 优化器，lr=1e-5，batch 16，12 epochs。大部分参数冻结，仅训练 EAE + APG + adapter（11.91M 可训练参数）。

## 实验关键数据

### 主实验

| 方法 | Backbone | mIoU↑ | Dice↑ | 可训练参数 |
|------|----------|-------|-------|-----------|
| DeepLabV3+ | ResNet101 | 57.29 | 72.84 | 60.21M |
| Mask2former | Swin-L | 69.59 | 81.44 | 144.85M |
| SAM (frozen) | ViT-L | 53.82 | 64.99 | 0M |
| SAM (finetune) | ViT-L | 67.87 | 77.45 | 10.06M |
| SAMUS | ViT-L | 68.56 | 78.46 | 43.21M |
| **APSAM** | **ViT-L** | **72.83** | **82.31** | **11.91M** |

### 消融实验

| 配置 | mIoU↑ | Dice↑ | 说明 |
|------|-------|-------|------|
| w/o EAE & APG (SAM FT) | 67.87 | 77.45 | 基线 |
| w/o APG | 70.89 | 79.50 | EAE 贡献 +3.02 |
| w/o EAE | 71.90 | 81.62 | APG 贡献 +4.03 |
| **Full (EAE + APG)** | **72.83** | **82.31** | 两者互补 |
| 1个随机点 | 67.87 | 77.45 | 基线 |
| 2个随机点 | 70.31 | 80.18 | 多点有帮助 |
| **APG 2点** | **71.90** | **81.62** | 智能选点更优 |

### 关键发现

- APG 贡献（+4.03 mIoU）大于 EAE（+3.02），说明提示质量对 SAM 的影响比编码器初始化更大
- 跨域泛化强：在 PIDray 和 PIXray 上分别达 71.23% 和 83.61% mIoU，比 SAMUS 高 4.22% 和 3.70%
- SAM 零样本在 X 光上仅 53.82% mIoU——域差距非常大
- APSAM 用 11.91M 参数超越了使用 144.85M 参数的 Mask2former（72.83 vs 69.59）

## 亮点与洞察

- **物理先验的巧妙利用**：X 光的双能量通道不是简单的 RGB 分解，而是携带了材质信息的物理信号——EAE 用 max/min 操作提取高低能特征，简单但有效
- **APG 的实用性**：安检场景中操作员只有时间点击一下——APG 自动将单击扩展为更信息化的两点提示，降低了部署时的人工负担
- **数据集的长期价值**：98K 图像 + 30 类细粒度标注，填补了安检领域分割数据的空白

## 局限与展望

- 数据源虽多但主要来自中国安检系统，不同国家/厂商的 X 光设备色域差异可能影响泛化
- 30 类分类仍可能不够——实际安检中违禁品种类更多（如液体、粉末状物品）
- 闭环标注策略虽有 5 轮迭代，但某些高度重叠场景的标注质量仍难保证
- APG 的 K-means 聚类在极端瘦长物体上可能失效（两个点在同一方向上）
- 从安全角度看，模型的漏检率（false negative）比 IoU 更关键，但论文未重点分析

## 相关工作与启发

- **vs SAMUS**: 同样是 SAM 的域适配方法，但 SAMUS 使用 43.21M 可训练参数且不利用 X 光物理特性。APSAM 用更少参数（11.91M）取得更好效果
- **vs Mask2former**: 完全监督方法需要 145M 参数但 mIoU 仅 69.59%，说明基于 SAM 的半监督适配方案是更高效的路径
- **vs PIDray/PIXray**: 规模上 XSeg 是两者总和的 2 倍，类别粒度是 2.5 倍——量变可能带来质变

## 评分

- 新颖性: ⭐⭐⭐⭐ EAE和APG设计均有新意但不算突破性
- 实验充分度: ⭐⭐⭐⭐⭐ 完整消融+跨域+多框架对比+点提示策略对比
- 写作质量: ⭐⭐⭐⭐ 数据集和方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 数据集贡献对安检领域有长期影响，方法可直接部署

<!-- RELATED:START -->

## 相关论文

- [Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/medical_imaging/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](../../AAAI2026/medical_imaging/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)

<!-- RELATED:END -->
