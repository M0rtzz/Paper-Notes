---
title: >-
  [论文解读] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions
description: >-
  [ECCV 2024][3D视觉][单目深度估计] 利用text-to-image扩散模型（ControlNet/T2I-Adapter）将简单场景图像转化为保持同一3D结构的恶劣条件图像，通过自蒸馏微调现有单目深度估计网络，统一解决恶劣天气和非朗伯表面等分布外挑战。
tags:
  - ECCV 2024
  - 3D视觉
  - 单目深度估计
  - 扩散模型
  - 数据增强
  - 恶劣天气
  - 非朗伯表面
---

# Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions

**会议**: ECCV 2024  
**arXiv**: [2407.16698](https://arxiv.org/abs/2407.16698)  
**代码**: https://diffusion4robustdepth.github.io/  
**领域**: 3D视觉  
**关键词**: 单目深度估计, 扩散模型, 数据增强, 恶劣天气, 非朗伯表面

## 一句话总结

利用text-to-image扩散模型（ControlNet/T2I-Adapter）将简单场景图像转化为保持同一3D结构的恶劣条件图像，通过自蒸馏微调现有单目深度估计网络，统一解决恶劣天气和非朗伯表面等分布外挑战。

## 研究背景与动机

**领域现状**：单目深度估计已因深度学习取得巨大进步。DPT、Depth Anything等SOTA模型通过在大量混合数据集上训练，可在多数正常场景下表现出色。训练方式包括LiDAR监督、自监督（立体对/视频序列）等。

**现有痛点**：即使是最强的泛化模型，在分布外数据（长尾场景）上仍面临严重挑战——恶劣天气（雨天、夜间、雾天）和非朗伯表面（透明/镜面物体）。原因是：(1) 高质量标注数据严重不足（LiDAR在雨天/雪天也不可靠、对透明物体失效）；(2) 主动传感器本身在这些条件下也会失败。

**核心矛盾**：需要大量恶劣条件下的深度标注数据来训练鲁棒模型，但这些标注几乎无法获取。现有方法通常分别针对不同挑战设计独立方案（如专门针对夜间的GAN、专门针对透明物体的方法），缺乏统一框架。

**本文目标** 提出统一方案同时应对恶劣天气和非朗伯表面两大类挑战，不需要真实恶劣条件数据，仅从简单图像出发。

**切入角度**：depth-conditioned的text-to-image扩散模型可以在保持3D结构（深度图）不变的前提下，通过文本提示将简单场景变换为任意复杂场景。这意味着简单图像上的深度标签可以直接迁移到生成的恶劣条件图像上。

**核心 idea**：用扩散模型做"条件保持的风格迁移"——从简单图像生成保持相同深度结构的恶劣条件图像，用原模型在简单图像上的深度预测作为伪标签，自蒸馏微调模型以适应恶劣场景。

## 方法详解

### 整体框架

pipeline包含两个阶段：
1. **数据生成阶段**：
    - 选取简单图像 $e_i$（正常天气/朗伯表面）
    - 用预训练深度网络（教师）预测深度图 $d_i$
    - 将 $(e_i, d_i, p_c)$（图像+深度+文本提示）输入depth-conditioned扩散模型（T2I-Adapter），生成恶劣条件图像 $h_i^c$
    - 深度图 $d_i$ 同时作为扩散模型的空间控制条件和生成图像的伪深度标签
2. **自蒸馏微调阶段**：
    - 以相同预训练网络为学生模型
    - 用 $(h_i^c, d_i)$ 对和原 $(e_i, d_i)$ 对进行微调
    - 使用scale-and-shift-invariant loss

### 关键设计

1. **Depth-aware条件扩散数据生成**:

    - 功能：从简单图像生成保持相同3D结构的恶劣条件图像
    - 核心思路：利用ControlNet/T2I-Adapter，输入深度图 $d_i$ 作为空间条件约束3D结构一致性，文本提示 $p_c$ 控制生成场景的恶劣条件类型。两个条件输入各司其职：深度保持结构，文本控制风格。
    - 设计动机：不同于GAN需要为每种条件训练单独模型，扩散模型通过文本提示可生成**无限种**复杂场景（雨/雪/雾/夜/透明/镜面...），极大扩展了覆盖范围。且无需任何目标条件的实际样本。

2. **自蒸馏训练协议（Self-Distillation）**:

    - 功能：用教师-学生范式微调深度估计网络
    - 核心思路：预训练的深度网络同时扮演教师（在简单图像上生成伪标签 $d_i$）和学生（在恶劣图像上被微调）。使用scale-and-shift-invariant loss：
    $L_{ssi}(\hat{d}, \hat{d}^*) = \frac{1}{2M} \sum_{i=1}^{M} \rho(\hat{d}_i - \hat{d}_i^*)$
      其中 $\hat{d}$ 是归一化预测，$\hat{d}^*$ 是教师给出的归一化伪标签。
    - 设计动机：简单图像上的深度预测已经足够准确，可作为可靠的伪标签；scale-and-shift-invariant loss可处理不同网络/数据集间的尺度差异；该方案对底层模型完全无关（model-agnostic），可应用于任何预训练深度网络。

3. **非朗伯表面的纯生成式数据构造**:

    - 功能：无需真实透明/镜面物体数据集，纯从文本生成训练数据
    - 核心思路：先用Stable Diffusion从文本提示生成~20K张包含朗伯表面物体（如陶瓷瓶、木质容器）的图像，然后用T2I-Adapter将这些物体转换为透明/反射材质的版本，同时保持深度结构不变。
    - 设计动机：真实场景中收集透明/镜面物体的配对数据非常困难（需要手动布置纯朗伯物体场景）。而现有方法（如Depth4ToM）需要真实的透明物体数据集和GT分割图。扩散模型可完全绕过这一限制。

### 损失函数 / 训练策略

- **损失函数**：Scale-and-shift-invariant loss（SSI loss），比较归一化后的预测深度和伪标签深度
- 驾驶场景：按[27]的训练协议，使用md4all框架
- 非朗伯场景：按[18]的训练协议，使用Depth4ToM框架
- 其他网络微调：30K迭代，学习率 $10^{-6}$ → $10^{-7}$（25K时衰减），Depth Anything微调5K迭代
- 单卡3090 GPU，batch size 8，AdamW优化器
- 数据增强：color jitter、RGB shift、水平翻转等

## 实验关键数据

### 主实验（nuScenes单目深度估计）

| 方法 | day-clear AbsRel | night AbsRel | day-rain AbsRel |
|------|-----------------|--------------|-----------------|
| Depth Anything（原始） | 0.137 | 0.291 | 0.167 |
| **Depth Anything + Ours** | **0.134** | **0.219** | **0.157** |
| DPT（原始） | 0.189 | 0.354 | 0.237 |
| **DPT + Ours** | **0.184** | **0.224** | **0.199** |
| MiDaS（原始） | 0.171 | 0.261 | 0.218 |
| **MiDaS + Ours** | **0.168** | **0.254** | **0.195** |

夜间场景改善最显著：Depth Anything夜间AbsRel从0.291降至0.219（24.7%↓），DPT从0.354降至0.224（36.7%↓）。

### 非朗伯表面（Booster + ClearGrasp数据集）

| 方法 | 真实ToM数据 | GT分割 | Booster ToM MAE(mm) | ClearGrasp ToM MAE(mm) |
|------|-----------|--------|---------------------|----------------------|
| DPT (baseline) | ✗ | ✗ | 113.14 | 41.04 |
| **DPT + Ours** | **✗** | **✗** | **79.64** | **31.32** |
| DPT + Costanzino | ✓ | ✓ | 70.68 | 31.55 |
| Depth Anything (baseline) | ✗ | ✗ | 137.96 | 82.22 |
| **Depth Anything + Ours** | **✗** | **✗** | **54.31** | **33.88** |

不依赖任何真实非朗伯数据和GT分割，性能接近甚至超越需要这些额外数据的专用方法。

### 消融/跨数据集泛化实验

| 方法 | DrivingStereo rain AbsRel | nuScenes night AbsRel | RobotCar night AbsRel |
|------|--------------------------|----------------------|---------------------|
| DPT (baseline) | 0.188 | 0.354 | 0.154 |
| **DPT + Ours** | **0.124** | **0.263** | **0.130** |
| Depth Anything (baseline) | 0.112 | 0.291 | 0.125 |
| **Depth Anything + Ours** | **0.110** | **0.250** | **0.117** |

仅用Mapillary/Cityscapes/KITTI/Apolloscapes等纯白天数据集也能在从未见过的恶劣条件数据集上取得显著提升。

### 关键发现

- 在所有指标和所有天气条件下（包括简单的白天场景）都有一致提升
- 方法完全model-agnostic，对4个不同SOTA深度网络都有效
- 无需知道目标条件的具体特征（不需要真实的雨天/夜间图像）
- 用不同网络（DPT/Depth Anything）生成的深度图作为扩散模型条件，结果差异不大
- 非朗伯场景中，Depth Anything的ToM类别MAE从137.96mm降至54.31mm（61%↓），提升巨大

## 亮点与洞察

- **统一框架解决多种挑战**：不像以往需要为夜间、雨天、透明物体分别设计方案，本文用一个text-to-image扩散模型+文本提示覆盖所有场景。这是真正通用和可扩展的思路
- **零真实恶劣条件数据**：完全不需要任何目标条件的真实数据。只需简单场景图像+文本描述即可生成训练数据。这对标注困难的领域有巨大启示
- **Model-agnostic设计**：底层深度估计网络可以是任意架构和训练方式，方法作为微调策略具有很强的通用性
- **自蒸馏的优雅性**：同一网络既是教师又是学生——在简单场景上是好教师，在恶劣场景上需要学习。利用网络自身的知识完成能力扩展

## 局限与展望

- 扩散生成的图像质量和真实程度受限于基础模型能力，某些极端条件可能生成不够逼真
- 深度一致性假设只在语义层面成立——扩散模型可能引入微小的3D结构变化（如改变了物体表面细节）
- 伪标签的质量受限于预训练模型在简单场景上的精度
- 文本提示的设计需要一定的人工经验，不同措辞可能影响生成质量
- 非朗伯场景的训练数据完全来自生成（无真实图像），domain gap可能存在
- 未探索视频一致性——在时序深度估计中可能出现帧间不一致

## 相关工作与启发

- **vs md4all (Gasperini et al.)**: 基于ForkGAN的方法，需要为每种条件训练单独的GAN，且需要知道目标条件特征（真实雨天/夜间图像）。本文用统一扩散模型+文本提示替代，更灵活
- **vs Depth4ToM (Costanzino et al.)**: 针对透明/镜面物体的专用方法，需要真实透明物体数据集和GT语义分割图。本文纯用文本提示生成数据，无需任何额外标注，效果接近
- **vs R4Dyn**: 利用雷达辅助数据的方法，需要额外传感器。本文纯基于单目图像，无需硬件扩展

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将depth-conditioned扩散模型用于解决深度估计的分布外鲁棒性问题，思路新颖且自然
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖4个深度网络、5+个数据集、3类挑战条件（夜间/雨天/非朗伯），跨数据集泛化实验设计周到
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验对比公平详尽
- 价值: ⭐⭐⭐⭐⭐ 提供了一种通用、可扩展的方案来增强已有模型的分布外鲁棒性，对自动驾驶和机器人领域有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [Camera Height Doesn't Change: Unsupervised Training for Metric Monocular Road-Scene Depth Estimation](camera_height_doesnapost_change_unsupervised_training_for_metric_monocular_road-.md)

<!-- RELATED:END -->
