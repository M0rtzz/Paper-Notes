---
title: >-
  [论文解读] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer
description: >-
  [CVPR 2025][图像生成][动物姿态估计] 本文提出 AniMer，首次将高容量 ViT 骨干引入四足动物 SMAL 参数估计，通过动物科级监督对比学习区分不同物种的形状分布，配合基于 ControlNet 的合成数据集 CtrlAni3D（10k图像），在 Animal3D/CtrlAni3D/跨域 Animal Kingdom 上全面超越现有方法。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "动物姿态估计"
  - "SMAL模型"
  - "Transformer"
  - "对比学习"
  - "合成数据集"
---

# AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer

**会议**: CVPR 2025  
**arXiv**: [2412.00837](https://arxiv.org/abs/2412.00837)  
**代码**: [https://luoxue-star.github.io/AniMer_project_page/](https://luoxue-star.github.io/AniMer_project_page/) (项目页)  
**领域**: 图像生成  
**关键词**: 动物姿态估计, SMAL模型, Transformer, 对比学习, 合成数据集

## 一句话总结
本文提出 AniMer，首次将高容量 ViT 骨干引入四足动物 SMAL 参数估计，通过动物科级监督对比学习区分不同物种的形状分布，配合基于 ControlNet 的合成数据集 CtrlAni3D（10k图像），在 Animal3D/CtrlAni3D/跨域 Animal Kingdom 上全面超越现有方法。

## 研究背景与动机

**领域现状**：动物姿态和形状估计对动物福利、生态学和生物力学研究至关重要。SMAL 模型是四足动物的标准参数化表示，类似于人体的 SMPL。但现有方法主要针对单一物种（马或狗），使用 CNN 骨干且训练数据有限。

**现有痛点**：(1) CNN 骨干容量不足以在单一网络中统一建模多物种的形状差异（猫 vs 牛 vs 河马）；(2) 多物种3D标注数据极度稀缺，Animal3D 虽提供大规模基准但仍不够多样；(3) 人体重建中 ViT+大规模数据的"简单有效"范式（如 HMR2.0）在动物领域从未验证。

**核心矛盾**：动物比人体具有更大的物种间形状差异（种间差异>种内差异），且可用的3D标注数据远少于人体领域。

**本文目标**：验证"大容量骨干+大规模数据"范式在动物姿态/形状估计中的有效性，并解决数据稀缺问题。

**切入角度**：(1) 引入 ViT 骨干替代 CNN；(2) 设计科级对比学习捕获物种间形状差异；(3) 用 ControlNet 合成大规模训练数据。

**核心 idea**：ViT 编码器 + Transformer 解码器直接回归 SMAL 参数，通过 class token 的科级对比学习增强不同动物家族的形状判别能力。

## 方法详解

### 整体框架
输入单张 RGB 图像和可学习 class token，ViT 编码器提取图像特征（192×1280），Transformer 解码器输出特征向量（1×1024），独立的 MLP 头分别预测形状参数 $\hat{\beta} \in \mathbb{R}^{41}$、姿态参数 $\hat{\theta} \in \mathbb{R}^{35 \times 3}$ 和相机参数。同时 class token 经预测头输出用于对比学习的动物科特征。

### 关键设计

1. **动物科级监督对比学习（Family Supervised Contrastive Learning）**:

    - 功能：增强网络对不同动物科（猫科、犬科、马科、牛科等）的形状判别能力
    - 核心思路：利用 ViT 的可学习 class token 与图像特征交互，提取动物科信息。在 mini-batch 中应用监督对比损失 $\mathcal{L}_{\text{con}}$：同科样本的 class token 特征拉近，异科样本拉远。温度参数 $\tau$ 控制对比学习的分辨度
    - 设计动机：人体 SMPL 的形状参数来自同一多元正态分布，但动物至少有物种间和物种内两个层次的差异。对比学习在特征空间显式编码这种层次结构

2. **CtrlAni3D 合成数据集**:

    - 功能：缓解3D标注动物数据的稀缺性
    - 核心思路：使用 ControlNet 从 SMAL 动画的 mask 和深度图渲染条件生成逼真的动物图像。文本提示描述动物行为，背景来自 COCO 或 AI 合成。用 SAM2.0 + 人工验证过滤不合格图像。最终包含 9711 张像素对齐的 SMAL 标注图像，覆盖10个物种
    - 设计动机：传统 CG 渲染受限于纹理质量和光照控制，生成式 AI 可以以最少人力产生高质量多样化图像

3. **直接参数解码 + 两阶段训练**:

    - 功能：适应动物领域数据特点的训练策略
    - 核心思路：不同于 HMR2.0 的残差参数解码（依赖大规模运动数据库的均值），AniMer 采用直接参数解码（因为缺乏 SMAL 姿态先验）。两阶段训练：第一阶段仅用 3D 数据（500 epochs）确保网络预测合理形状和姿态，第二阶段引入全部 2D+3D 数据（700 epochs）
    - 设计动机：3D 和 2D 数据量严重不平衡，直接混合训练会导致 3D 回归质量下降

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{\text{total}} = \lambda_{\text{3D}}\mathcal{L}_{\text{3D}} + \lambda_{\text{2D}}\mathcal{L}_{\text{2D}} + \lambda_{\text{prior}}\mathcal{L}_{\text{prior}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}}$。3D 损失包括顶点回归、关节回归、参数回归；2D 损失为关键点重投影误差；先验损失约束 SMAL 参数合理性。

## 实验关键数据

### 主实验

| 方法 | Animal3D PA-MPJPE↓ | Animal3D PCK↑ | CtrlAni3D PA-MPJPE↓ |
|------|-------------------|---------------|---------------------|
| AniMer (ViT-H) | **最优** | **最优** | **最优** |
| HMR2.0 (动物版) | 次优 | 次优 | 次优 |
| WLDO (CNN) | 较差 | 较差 | 较差 |
| HMR (CNN) | 较差 | 较差 | 较差 |

### 消融实验

| 配置 | 关键效果 |
|------|---------|
| 去掉科级对比学习 | 形状估计精度下降 |
| 去掉 CtrlAni3D | OOD 泛化能力显著下降 |
| CNN骨干替代ViT | 性能大幅下降，证明高容量骨干的重要性 |
| 残差解码替代直接解码 | 性能下降，因缺乏 SMAL 姿态先验 |

### 关键发现
- ViT 骨干对动物重建的提升远大于人体领域（因物种间差异更大，需要更强的表示能力）
- 科级对比学习在所有基准上均提升了姿态和形状估计精度
- CtrlAni3D 显著提升了在未见过的 Animal Kingdom 数据集上的泛化能力
- 两阶段训练对处理 3D/2D 数据不平衡至关重要

## 亮点与洞察
- 将人体重建的成功范式（HMR2.0的ViT+大数据）迁移到动物领域，验证了其普适性
- 用 ControlNet 从参数化模型(SMAL)动画生成训练数据的思路可推广到其他缺乏3D标注的领域
- 科级对比学习捕获了动物形状的层次结构，这种"层次化形状建模"的思路可启发其他多类别3D重建任务

## 局限与展望
- SMAL 模型本身基于41个玩具扫描构建，对真实动物的几何精度有限
- 仅覆盖四足动物，鸟类、爬行类等其他类别未涉及
- 合成数据与真实数据仍存在域差距

## 相关工作与启发
- **vs HMR2.0**: 人体重建SOTA，本文验证了类似范式对动物的有效性
- **vs Animal3D**: 提供了首个大规模基准但使用CNN骨干。AniMer通过ViT+对比学习大幅提升
- **vs SPAC-Net**: 类似使用ControlNet生成合成数据，但依赖纹理CAD资产。AniMer直接从无纹理SMAL生成

## 评分
- 新颖性: ⭐⭐⭐⭐ 科级对比学习和ControlNet合成数据管线新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准全面评测+详细消融
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰
- 价值: ⭐⭐⭐⭐ 推进了动物3D重建领域的技术前沿

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)
- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](lavin-dit_large_vision_diffusion_transformer.md)
- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)
- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)

</div>

<!-- RELATED:END -->
