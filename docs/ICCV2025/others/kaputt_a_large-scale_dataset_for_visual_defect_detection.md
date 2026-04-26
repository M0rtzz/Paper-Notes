---
title: >-
  [论文解读] Kaputt: A Large-Scale Dataset for Visual Defect Detection
description: >-
  [ICCV 2025][缺陷检测] Kaputt 发布了一个包含 23 万+ 图像、4.8 万+ 独立商品的大规模零售物流缺陷检测数据集，规模是 MVTec-AD 的 40 倍，首次引入显著的姿态和外观变化，使得 SOTA 异常检测方法的 AUROC 不超过 56.96%，揭示了现有方法在真实零售场景中的严重不足。
tags:
  - ICCV 2025
  - 缺陷检测
  - 异常检测
  - 大规模数据集
  - 零售物流
  - benchmark
---

# Kaputt: A Large-Scale Dataset for Visual Defect Detection

**会议**: ICCV 2025  
**arXiv**: [2510.05903](https://arxiv.org/abs/2510.05903)  
**代码**: [数据集](https://www.kaputt-dataset.com)  
**领域**: 其他  
**关键词**: 缺陷检测, 异常检测, 大规模数据集, 零售物流, benchmark

## 一句话总结

Kaputt 发布了一个包含 23 万+ 图像、4.8 万+ 独立商品的大规模零售物流缺陷检测数据集，规模是 MVTec-AD 的 40 倍，首次引入显著的姿态和外观变化，使得 SOTA 异常检测方法的 AUROC 不超过 56.96%，揭示了现有方法在真实零售场景中的严重不足。

## 研究背景与动机

自动化视觉缺陷检测是质量保障的关键环节。现有异常检测基准（MVTec-AD、VisA）主要面向**制造业场景**，特点是：物体姿态高度受控、类别有限（15 和 12 类）。在这些数据集上，SOTA 方法已达到 99.9% AUROC，趋近饱和。

然而，**零售物流场景**面临本质不同的挑战：

**物品多样性极高**：从食品到电子产品，物理属性各异

**缺陷类型多变**：从细微褶皱到严重破损，许多缺陷连人工检查都困难

**样本极度稀缺**：大多数商品只被观察几次，正常/缺陷样本都很少

**姿态变化显著**：商品随意放置在物流容器中，姿态不可控

已有数据集无法反映这些挑战。MVTec-AD 仅 5354 张图（1258 缺陷），VisA 仅 10821 张图。领先的异常检测方法在迁移到物流场景时表现骤降。

核心问题：**如何在每个商品样本极少、正常和缺陷样本都有限、且类内变化巨大的条件下，构建可泛化的缺陷检测方法？**

## 方法详解

### 整体框架

本文的核心贡献是数据集及其配套的综合评测基准，而非提出新方法。数据集设计体现了精心的工程考量：

### 关键设计

1. **数据集结构**：

    - **查询集（Query）**：100,267 张标注图像，包含 29,316 个缺陷实例
    - **参考集（Reference）**：每个商品 1-3 张未标注的"正常"参考图像（138,154 张）
    - **商品数量**：48,376 种独立商品，按 item ID 严格划分 train/val/test，确保无泄露
    - **分辨率**：12MP RGB 相机，裁剪后 2048×2048 像素
    - 训练集 85%、测试集 10%、验证集 5%

2. **多层次标注体系**：

    - **缺陷严重度**：无缺陷 / 轻微 / 严重，三位标注者独立标注后多数投票
    - **缺陷类型**（7 类，可多标签）：穿透（holes/tears）、变形（dents/crushes）、启动（open box/bag）、解构、溢出、表面（dirt/scratches）、缺失单元
    - **物品材质**：纸板、塑料袋、硬塑料、气泡膜、纸质、书籍等
    - 变形是最常见缺陷类型但多为轻微，溢出和解构多为严重

3. **数据采集方法论**：

    - **硬件**：12MP RGB 相机 + f/12mm 镜头，俯视拍摄，LED 面板均匀照明，减少塑料反光
    - **缺陷样本收集**：两阶段策略——（1）人工标记的缺陷品；（2）迭代挖掘，用已训练分类器筛选候选样本再人工标注
    - **质量控制**：过滤低质图像、限制每商品最多 15 张、平衡缺陷率至 28.6%、排除无正常样本的商品

### 损失函数 / 训练策略

本文不提出新方法，而是系统评测了四类基线：
- **无训练无参考**（零样本）：CLIP、Claude 3.5、Pixtral-12B
- **无训练有参考**（少样本异常检测）：PatchCore、WinCLIP
- **有训练无参考**（监督学习）：ResNet50、ViT-S/DINOv2、AutoGluon
- **有训练有参考**（混合）：PatchCore+微调骨干、AutoGluon+参考

## 实验关键数据

### 主实验

| 方法 | 类型 | APany (%) ↑ | APmajor (%) ↑ | AUROC ↑ |
|------|------|-----------|-------------|---------|
| Random | - | 31.84 | 14.00 | 50.00 |
| CLIP | 零样本 | 36.20 | 17.15 | 56.05 |
| Claude-icl | 零样本+上下文 | 36.57 | 24.76 | 56.96 |
| PatchCore50 | 少样本AD | 35.86 | 17.80 | 54.69 |
| WinCLIP-few | 少样本AD | 34.05 | 19.29 | 52.41 |
| ResNet50 | 监督 | 81.06 | 74.93 | 88.36 |
| **ViT-S** | **监督** | **90.67** | **91.45** | **94.27** |
| PatchCore50-ft | 混合 | 40.18 | 20.98 | 60.14 |

### 消融实验

**减少缺陷训练样本时的性能退化**：

| 配置 | APany (%) | APmajor (%) | AUROC |
|------|----------|------------|-------|
| ViT-S 完整训练集 | 90.67 | 91.45 | 94.27 |
| ViT-S 1%缺陷率 (Query only) | 57.7 | 40.5 | 74.4 |
| ViT-S 1%缺陷率 (Query + ref) | 40.4 | 14.9 | 63.2 |

**关键对比：异常检测方法在不同数据集上的表现**：

| 数据集 | AUROC |
|--------|-------|
| MVTec-AD (SOTA) | 99.9% |
| VisA (SOTA) | 99.5% |
| **Kaputt (最佳无监督)** | **56.96%** |

### 关键发现

1. **异常检测方法全面失效**：所有无监督/少样本方法均不超过 56.96% AUROC，几乎等同于随机
2. **VLM 不好用**：Claude/Pixtral 能描述物体但无法检测细微缺陷，与 Jiang et al. 的发现一致
3. **参考图像反而有害**：天真地融合参考图像（如特征平均）反而降低监督方法性能（96%→87% 训练集 APany）
4. **监督方法的天花板**：ViT-S 达 90.67% APany，但在可变形物品、"对抗性"设计（印有破洞图案的包装）上仍犯错
5. **姿态变化是核心挑战**：异常检测方法将正常的姿态/外观差异误判为异常

## 亮点与洞察

- **真正暴露了异常检测的瓶颈**：不是方法不够好，而是问题性质发生了本质变化——从"受控制造"到"开放零售"
- **数据集设计极其严谨**：按 item ID 划分避免数据泄露、三人标注多数投票、缺陷率对齐已有 benchmark
- **四场景评测框架**：有/无训练 × 有/无参考的 2×2 矩阵提供全面视角
- **规模优势**：48K 独立商品 + 29K 缺陷实例是同类最大

## 局限与展望

- 仅有俯视角度单一视角，未利用多视角信息
- 参考图像质量不保证（<1% 参考图本身有缺陷），可能引入噪声
- 标注仍有错误（如遮挡导致不可观察、设计图案与缺陷易混淆）
- 未提供像素级分割标注，无法评估缺陷定位精度
- 所有实验基于 RGB 图像，未探索深度/红外等模态

## 相关工作与启发

- MVTec-AD/VisA 已饱和，Kaputt 代表了异常检测的下一阶段挑战
- ARMBench 面向类似场景但缺陷样本仅为 Kaputt 的 1/4，缺陷类型仅 2 种
- 如何让异常检测方法适应**大类内变化**是关键开放问题
- 参考图像的有效利用是被低估的研究方向——简单特征平均显然不够

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据集驱动的贡献, 问题定义准确, 但无方法创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 四种场景 × 多种方法 + 训练集缩减实验 + 详尽误差分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰, 数据集文档完善
- **价值**: ⭐⭐⭐⭐⭐ 填补了零售物流缺陷检测的基准空白, 将推动社区进步

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2025/others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[ACL 2025\] Code-Switching and Syntax: A Large-Scale Experiment](../../ACL2025/others/code-switching_and_syntax_a_large-scale_experiment.md)
- [\[ICML 2025\] Cover Learning for Large-Scale Topology Representation](../../ICML2025/others/cover_learning_for_large-scale_topology_representation.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](../../ACL2025/others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[CVPR 2026\] UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](../../CVPR2026/others/unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)

<!-- RELATED:END -->
