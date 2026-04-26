---
title: >-
  [论文解读] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation
description: >-
  [ICCV 2025][图像分割][Affordance Learning] 提出一套完整的 affordance 学习系统：(1) 从第一人称视频自动提取精确的可抓取/功能性 affordance 分割标注，(2) 基于 DINOv2 + 深度几何引导的 GAT 模型实现跨域 affordance 分割（mIoU 提升 13.8%），(3) Aff-Grasp 框架在 179 次真实机器人试验中达到 77.1% 抓取成功率。
tags:
  - ICCV 2025
  - 图像分割
  - Affordance Learning
  - Egocentric Video
  - Robotic Manipulation
  - Tool Grasping
---

# Learning Precise Affordances from Egocentric Videos for Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2408.10123](https://arxiv.org/abs/2408.10123)  
**代码**: [https://reagan1311.github.io/affgrasp](https://reagan1311.github.io/affgrasp)  
**领域**: segmentation  
**关键词**: Affordance Learning, Egocentric Video, Robotic Manipulation, Affordance Segmentation, Tool Grasping

## 一句话总结

提出一套完整的 affordance 学习系统：(1) 从第一人称视频自动提取精确的可抓取/功能性 affordance 分割标注，(2) 基于 DINOv2 + 深度几何引导的 GAT 模型实现跨域 affordance 分割（mIoU 提升 13.8%），(3) Aff-Grasp 框架在 179 次真实机器人试验中达到 77.1% 抓取成功率。

## 研究背景与动机

Affordance（物体提供的潜在动作可能性）是具身智能的核心概念。例如，切东西时抓刀柄、递刀时抓刀刃。但当前研究面临三大挑战：

**数据稀缺**：大规模精确 affordance 标注数据集缺乏，标注细小物体部件（如勺柄）非常困难

**泛化差**：现有模型难以跨域或泛化到未见物体/affordance 类别

**真实部署少**：很少有工作在真实机器人上验证

这三个问题相互关联：缺乏大规模多样数据 → 模型泛化差 → 无法可靠部署。

现有从视频学习 affordance 的方法（如 VRB、Robo-ABC）有两个关键局限：
- 仅关注"人如何抓取物体"（可抓取 affordance），忽略"工具哪部分在使用"（功能性 affordance）
- affordance 表示为粗糙的高斯热力图而非精确分割掩码

## 方法详解

### 整体框架

三部分组成的完整系统：
1. **自动数据收集管线**：从第一人称视频 → 精确 affordance 分割标注
2. **GAT 模型**：几何引导的 affordance 分割
3. **Aff-Grasp 框架**：affordance 驱动的机器人操作

### 关键设计

1. **自动 Affordance 数据收集管线**：

    - **可抓取点定位**：从手-物交互视频中提取接触帧，用手-物检测器定位交互区域，在手掩码与物体框的交集采样接触点。找到接触前帧（物体完全可见），通过单应性变换将接触点投影到该帧
    - **功能性点定位**：从工具-物体交互视频中定位功能区域。找到工具与目标物IoU最小的接触前帧，计算工具掩码内到目标物掩码最短距离的点。无动作视频时用与抓取点最远采样替代
    - **数据生成**：通过点对应将功能性点映射到手-物交互前帧，然后用 SAM 以这些点为 prompt 生成精确分割掩码

   设计动机：同时获取可抓取和功能性affordance的精确分割标注，无需人工标注。

2. **Geometry-guided Affordance Transformer (GAT)**：

    - **DINOv2 编码器 + LoRA**：使用自监督视觉基础模型 DINOv2 作为特征提取器，通过 LoRA 微调避免过拟合且适应多域数据。LoRA 将可训练矩阵分解为 $W_0 + \Delta W = W_0 + BA$，其中 $r \ll \min(d,k)$
    - **Depth Feature Injector (DFI)**：用 Depth-Anything 生成伪深度图，通过交叉注意力将几何特征注入图像特征。$\hat{F}_i = \beta \cdot \text{softmax}(QK^T/\sqrt{d_k}) \cdot V + F_i$，其中 $\beta$ 初始化为 0 以防止训练初期深度特征主导。DFI 即使仅在训练时使用（推理时丢弃），也能带来提升——起到正则化作用
    - **余弦相似度分类器**：用可学习 embedding $M \in \mathbb{R}^{L \times C}$ 与上采样特征的余弦相似度做分割，无显式背景分类器（低于阈值 τ 即为背景），比线性层更鲁棒

   设计动机：低分辨率训练数据 + 域差异大，用 DINOv2 增强跨域能力，DFI 利用形状信息（柱形=可抓取，锋利边缘=切割）补偿纹理不足。

3. **Aff-Grasp 机器人操作框架**：

    - 开放词汇检测器定位场景内物体 → GAT 预测各物体affordance → 选择具有所需 affordance 的物体
    - Contact-GraspNet 在可抓取区域生成 6-DoF 抓取位姿 → 选最高分方案
    - 执行 affordance 特定的顺序运动原语（工具使用/递交）
    - 支持 CLIP 文本 embedding 替代可学习 embedding 实现开放词汇

### 损失函数 / 训练策略

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{focal} + \mathcal{L}_{dice}$$

使用 focal loss + dice loss 的组合处理严重的类别不平衡。4 个 DFI 模块分布在模型的 4 个 block 开头。推理时可选择去掉 DFI 以加速。

## 实验关键数据

### 主实验 (表格)

**视觉评估 - Affordance Evaluation Dataset (AED)：**

| 预训练 | 方法 | mIoU | F1 | Acc |
|--------|------|------|-----|-----|
| ImageNet | DeepLabV3+ | 13.46 | 22.27 | 23.05 |
| ImageNet | PSPNet | 16.90 | 27.32 | 26.46 |
| ImageNet | SegFormer | 23.72 | 36.86 | 37.19 |
| Foundation | ZegCLIP | 18.33 | 26.41 | 25.55 |
| Foundation | DINOv2 | 46.16 | 62.49 | 63.61 |
| Foundation | ViT-Adapter | 50.86 | 66.88 | 65.21 |
| Foundation | OOAL | 54.82 | 70.58 | 68.00 |
| Foundation | **GAT (Ours)** | **68.62** | **81.09** | **83.51** |

**机器人准确性评估：**

| 方法 | 正确Affordance | 成功抓取 | 成功交互 |
|------|----------------|----------|----------|
| LOCATE | 42/72 (58.3%) | 33/72 (45.8%) | n/a |
| Robo-ABC | 62/72 (86.1%) | 44/72 (61.1%) | n/a |
| **Aff-Grasp** | **70/72 (97.2%)** | **57/72 (80.6%)** | **47/72 (65.3%)** |

### 消融实验 (表格)

**GAT 各组件消融 (AED)：**

| 配置 | mIoU | F1 | Acc |
|------|------|-----|-----|
| Baseline (DeiT III + linear + BCE) | 31.02 | 44.55 | 35.85 |
| + DINOv2 | 45.45 | 61.78 | 70.86 |
| + embedder | 48.83 | 65.10 | 71.07 |
| + embedder & 4× upsample | 51.41 | 64.26 | 67.27 |
| + focal loss | 50.70 | 66.97 | 70.12 |
| + focal & dice loss | 53.12 | 69.13 | 74.55 |
| cosine sim w/o bg | 56.70 | 72.00 | 71.22 |
| + DFI (training only) | 60.15 | 74.92 | 79.87 |
| + DFI (full) | 64.66 | 78.35 | 79.74 |
| + LoRA (完整 GAT) | **68.62** | **81.09** | **83.51** |

**DFI 计算开销：**

| 推理设置 | #Params (M) | GFLOPs | 推理时间 (ms) |
|----------|-------------|--------|--------------|
| w/ DFI | 96.9 (↓5.4%) | 204.9 (↓9.5%) | 10.1 (↓37.6%) |
| w/o DFI | - | - | - |

**泛化评估（未见物体）：**

| 方法 | 正确 Affordance | 成功抓取 | 推理时间 (s) |
|------|----------------|----------|-------------|
| LOCATE | 20/35 (57.1%) | 15/35 (42.9%) | 0.0047 |
| Robo-ABC | 24/35 (68.6%) | 21/35 (60.0%) | 12.92 |
| **Aff-Grasp** | **32/35 (91.4%)** | **28/35 (80.0%)** | **0.0063** |

### 关键发现

1. **Foundation model 远超 ImageNet 预训练**：DINOv2 直接比 SegFormer (ImageNet) 高 22.44 mIoU，验证跨域能力的重要性
2. **DFI 贡献巨大**：仅在训练时使用 DFI 就提升 3.58 mIoU，完整 DFI 提升 7.96 mIoU，说明深度几何信息是有效的正则化
3. **LoRA 提升 3.96 mIoU**：LoRA 微调在不修改原始 DINOv2 参数的情况下实现高效适应
4. **余弦相似度比线性层更鲁棒**：去除背景分类器 + 余弦相似度的组合比 linear w/o bg 高 1.74 mIoU
5. **机器人实验中 Aff-Grasp 全面领先**：affordance 预测准确率 97.2%（比 Robo-ABC 高 11.1%），抓取成功率 80.6%（高 19.5%），且首次支持工具-物体交互
6. **泛化到未见物体非常强**：91.4% 的 affordance 预测正确率，推理仅 6.3ms

## 亮点与洞察

- **首个同时标注可抓取+功能性 affordance 的精确分割数据管线**：从粗糙热力图到精确掩码的质变
- **DFI 作为训练正则化的发现**：推理时丢弃 DFI 仍有提升，且加速 37.6%，实用性极强
- **完整的从感知到操作的闭环验证**：179 次真实机器人试验覆盖 7 个任务、34 个物体，包括递交等复杂场景
- **开放词汇能力**：可用 CLIP 文本 embedding 替代可学习 embedding，支持未见 affordance

## 局限与展望

1. 收集数据依赖视频中的手-物交互和工具-物体交互的时间顺序，视频缺少某些交互类型时需要启发式替代
2. 分割精度受 SAM prompt 质量影响，遮挡严重时 prompt 点可能不准确
3. 功能性 affordance 的"最远点"替代策略基于"功能部件在抓取部件对面"的假设，不适用于所有工具
4. 真实机器人实验中的运动原语仍需预录制，限制了任务复杂度
5. 训练数据分辨率低（裁剪区域通常 <100px），限制了精细部件的识别

## 相关工作与启发

- **从人类视频学习 affordance**：VRB、Robo-ABC 是前身，本文改进了数据质量（精确掩码 vs 热力图）和 affordance 类型（+功能性）
- **视觉基础模型的应用**：DINOv2 的跨域特征表示能力是 affordance 泛化的关键使能技术
- **SAM 做 prompt 分割**：利用点 prompt 生成精确掩码是连接稀疏定位和密集分割的桥梁
- 对任务导向抓取（VLM/LLM 方案）的高效替代：无需逐次推理语言模型

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个同时学习精确可抓取+功能性 affordance 分割的完整系统
- **实验充分度**: ⭐⭐⭐⭐⭐ 视觉+机器人双重评估，179 次真实试验，跨域/零样本/杂乱场景全面覆盖
- **写作质量**: ⭐⭐⭐⭐ 系统完整，图示清晰，从数据到模型到部署逻辑流畅
- **价值**: ⭐⭐⭐⭐⭐ 解决了 affordance 研究的三大痛点（数据/泛化/部署），实用性极强

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)
- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [\[ICCV 2025\] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos](refereverything_towards_segmenting_everything_we_can_speak_of_in_videos.md)
- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)

<!-- RELATED:END -->
