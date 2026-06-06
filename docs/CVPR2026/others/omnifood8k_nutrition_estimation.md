---
title: >-
  [论文解读] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion
description: >-
  [CVPR 2026][食物营养估计] 构建了涵盖 8036 个样本的中式食物多模态营养数据集 OmniFood8K 和 115K 合成数据集 NutritionSynth-115K，并提出端到端框架通过 Scale-Shift 深度适配器、频域对齐融合和掩码预测头从单张 RGB 图像预测营养信息。
tags:
  - "CVPR 2026"
  - "食物营养估计"
  - "多模态数据集"
  - "深度估计"
  - "频域融合"
  - "中国菜"
---

# OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion

**会议**: CVPR 2026  
**arXiv**: [2604.12356](https://arxiv.org/abs/2604.12356)  
**代码**: [https://yudongjian.github.io/OmniFood8K-food/](https://yudongjian.github.io/OmniFood8K-food/)  
**领域**: 食物计算 / 多模态融合  
**关键词**: 食物营养估计, 多模态数据集, 深度估计, 频域融合, 中国菜

## 一句话总结

构建了涵盖 8036 个样本的中式食物多模态营养数据集 OmniFood8K 和 115K 合成数据集 NutritionSynth-115K，并提出端到端框架通过 Scale-Shift 深度适配器、频域对齐融合和掩码预测头从单张 RGB 图像预测营养信息。

## 研究背景与动机

**领域现状**：食物营养估计在公共健康中至关重要，深度学习方法在自动识别和估计食物质量、体积和营养方面展现潜力。

**现有痛点**：(1) 数据限制：现有数据集严重偏向西方菜系，对中式食物覆盖不足；(2) 算法限制：先进方法依赖深度相机获取深度信息，日常场景中食物照片通常用 RGB 相机拍摄。

**核心矛盾**：深度信息对准确估计食物体积和营养至关重要，但实际部署场景通常只有 RGB 图像。

**本文目标**：(1) 构建覆盖中式菜系的综合多模态食物数据集；(2) 提出仅需单张 RGB 图像的端到端营养预测框架。

**切入角度**：利用预训练深度估计模型从 RGB 图像预测深度，通过适配器校正和频域融合替代实际深度传感器。

**核心 idea**：预测深度图 → 适配器校正 → 频域对齐融合 RGB 和深度特征 → 掩码感知预测。

## 方法详解

### 整体框架

给定单张 RGB 图像，首先用预训练深度估计模型预测深度图，通过 SSRA 校正深度图的尺度偏差和局部结构误差。然后 FAFM 在频域层级融合 RGB 和校正深度特征。最后 MPH 通过动态通道选择和区域感知注意力预测营养值。

### 关键设计

1. **Scale-Shift Residual Adapter (SSRA)**:

    - 功能：校正预训练深度估计的全局尺度偏差和局部结构误差
    - 核心思路：学习全局尺度因子和偏移量进行仿射变换实现全局校准，同时用残差网络预测局部修正以保留精细结构
    - 设计动机：预训练深度模型在食物图像上的预测存在尺度不一致和局部失真

2. **Frequency-Aligned Fusion Module (FAFM)**:

    - 功能：在频域层级融合 RGB 和深度特征
    - 核心思路：将特征转换到频域，对齐 RGB 和深度的不同频率成分（低频捕获全局形状，高频捕获纹理细节），实现跨模态的层级融合
    - 设计动机：RGB 和深度特征在空间域直接融合可能因模态差异导致信息冲突，频域对齐提供更自然的融合方式

3. **Mask-based Prediction Head (MPH)**:

    - 功能：聚焦关键食材区域提升预测准确性
    - 核心思路：通过动态通道选择筛选信息量最大的特征通道，结合区域感知注意力强调关键食材区域
    - 设计动机：食物图像中不同区域的营养信息密度不同，背景和容器对预测是噪声

### 损失函数 / 训练策略

标准回归损失预测热量和宏量营养素。使用 NutritionSynth-115K 合成数据进行预训练增强泛化能力。

## 实验关键数据

### 主实验

| 方法 | 热量 MAE↓ | 蛋白质 MAE↓ | 脂肪 MAE↓ | 碳水 MAE↓ |
|------|----------|-----------|----------|----------|
| Im2Calories | 224.5 | 15.8 | 13.2 | 22.1 |
| Nutrition5K | 198.3 | 13.5 | 11.4 | 19.7 |
| RoDE | 185.7 | 12.8 | 10.6 | 18.3 |
| FBFPN (RGB+D) | 172.4 | 11.2 | 9.8 | 16.5 |
| **本文 (仅RGB)** | **165.8** | **10.5** | **9.2** | **15.8** |

### 消融实验

| 配置 | 热量 MAE↓ | 说明 |
|------|----------|------|
| 完整模型 | 165.8 | SSRA + FAFM + MPH |
| 无 SSRA | 178.2 | 不校正深度 |
| 无 FAFM (直接拼接) | 175.6 | 空间域拼接 |
| 无 MPH | 171.3 | 标准MLP头 |
| 无深度分支 | 182.5 | 仅RGB |

### 关键发现

- SSRA 贡献最大：去掉深度校正后 MAE 增加约 12 个点，说明预训练深度的原始预测确实存在显著偏差
- 频域融合优于空间域拼接，验证了 FAFM 的设计动机
- 仅用 RGB 输入的性能超过了使用真实深度传感器的 FBFPN 方法

## 亮点与洞察

- 用预训练深度估计替代深度传感器的思路具有实用价值：使得营养估计在日常场景中可部署
- OmniFood8K 数据集覆盖完整烹饪流程（原料→食谱→烹饪视频→多视图成品），是该领域最全面的数据集之一
- 合成数据集 NutritionSynth-115K 的构建方法对数据稀缺场景有借鉴价值

## 局限与展望

- 仅覆盖中式菜系，跨文化泛化性未验证
- 数据集规模（8036 样本）在深度学习标准下仍较小
- 预训练深度模型在食物图像上的适用性需要更多分析
- 可结合食材识别和份量估计进一步提升

## 相关工作与启发

- **vs Nutrition5K**: Nutrition5K 以西方食物为主且需多视图，本文覆盖中式食物且仅需单视图
- **vs FBFPN**: FBFPN 需要真实 RGB-D 输入，本文从单张 RGB 预测深度反而效果更好

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据集和频域融合框架都有新意
- 实验充分度: ⭐⭐⭐⭐ 多数据集对比和详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 数据集贡献突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos](v_nutri_nutrition_estimation_cooking_videos.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](../../AAAI2026/others/private_frequency_estimation_via_residue_number_systems.md)
- [\[ECCV 2024\] Intrinsic Single-Image HDR Reconstruction](../../ECCV2024/others/intrinsic_single-image_hdr_reconstruction.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](../../AAAI2026/others/hierarchical_semantic_alignment_for_image_clustering.md)
- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](../../ECCV2024/others/hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)

</div>

<!-- RELATED:END -->
