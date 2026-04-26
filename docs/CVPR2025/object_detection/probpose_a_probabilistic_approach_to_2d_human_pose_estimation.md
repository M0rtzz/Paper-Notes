---
title: >-
  [论文解读] ProbPose: A Probabilistic Approach to 2D Human Pose Estimation
description: >-
  [CVPR 2025][目标检测][人体姿态估计] ProbPose 提出用标定的概率图（probability map）替代传统热力图进行2D人体关键点定位，引入存在概率（presence probability）显式建模关键点是否在激活窗口内，并通过裁剪数据增强和 OKS 损失的期望风险最小化，显著改善了图像外关键点的定位能力和模型的概率标定质量。
tags:
  - CVPR 2025
  - 目标检测
  - 人体姿态估计
  - 概率图
  - 图像外关键点
  - OKS损失
  - 标定概率
---

# ProbPose: A Probabilistic Approach to 2D Human Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2412.02254](https://arxiv.org/abs/2412.02254)  
**代码**: https://MiraPurkrabek.github.io/ProbPose/  
**领域**: 目标检测 / 人体姿态估计  
**关键词**: 人体姿态估计, 概率图, 图像外关键点, OKS损失, 标定概率

## 一句话总结

ProbPose 提出用标定的概率图（probability map）替代传统热力图进行2D人体关键点定位，引入存在概率（presence probability）显式建模关键点是否在激活窗口内，并通过裁剪数据增强和 OKS 损失的期望风险最小化，显著改善了图像外关键点的定位能力和模型的概率标定质量。

## 研究背景与动机

**领域现状**：自顶向下的热力图方法（如 ViTPose）是当前2D人体姿态估计的主流范式，通过预测密集热力图并取 argmax 定位关键点。热力图用固定 sigma 的高斯作为训练目标，MSE 作为损失函数。

**现有痛点**：（1）所有现有方法在训练和评估中都忽略图像外的关键点——当关键点因裁切或遮挡落在激活窗口之外时，模型不会受到惩罚，反而会将其错误地定位到其他关节上（如左腿被错误对齐到右腿）；（2）热力图未经概率标定——固定高斯 sigma + MSE 训练迫使输出也是固定形状的高斯，无法反映真实的定位不确定性，更不能表达"我不知道"；（3）评估指标（OKS/PCK）仅评估图像内的关键点，对错误猜测不做惩罚。

**核心矛盾**：传统热力图将"定位"、"质量评估"和"是否存在"三个语义混在一个输出（热力图峰值）中，用一个标量编码三种不同的信息，既不精确也不灵活。

**本文目标**：设计一个能完整描述每个关键点状态的系统——(1) 是否在激活窗口内，(2) 在哪里，(3) 定位多可靠，(4) 是否可见——并提供标定的概率。

**切入角度**：作者注意到图像边界本质上是一种遮挡，可以通过裁剪增强产生大量"关键点在窗口外"的训练样本，从而训练存在概率。

**核心 idea**：用归一化到和为1、满足概率公理的概率图替代热力图，用 OKS 损失的期望风险最小化替代 MSE，并加入独立的存在概率预测头。

## 方法详解

### 整体框架

ProbPose 的输出包含四个部分：（1）概率图——关键点在激活窗口内每个位置的标定概率；（2）存在概率——关键点是否在激活窗口内的二值概率；（3）质量估计——预测的 OKS 得分；（4）可见性预测。推理时，先判断存在概率是否超过阈值，如果关键点存在，则从概率图中通过期望 OKS 最大化获取定位。

### 关键设计

1. **概率图（Probability Map）与 OKS 期望风险损失**:

    - 功能：提供标定的关键点位置概率分布
    - 核心思路：概率图通过 Sparsemax 激活函数保证所有值在[0,1]且和为1。每个像素 $p_L(x_i) = p(x_i | k_j \in AW, img)$ 表示关键点在该位置的后验概率。损失函数为期望风险最小化 $R_{exp}(x_i) = (1 - OKS(x_i)) \cdot p_L(x_i)$，加上 Sobel 梯度正则化 $\mathcal{L}_{OKS}(x_i) = (1-\alpha) R_{exp}(x_i) + \alpha g(x_i)$。推理时不用 argmax 而是计算每个像素的期望 OKS 并取其最大值，对双峰分布更鲁棒。
    - 设计动机：传统 MSE + 固定高斯目标隐含了不合理的形状假设——人体关键点的真实后验分布应该反映身体形状而非标注噪声。概率图不假设任何特定形状，且标定后的概率支持更灵活的查询（如"包含95%概率的最小区域"）。梯度正则化防止概率图过早形成尖峰而过拟合。

2. **存在概率（Presence Probability）**:

    - 功能：显式预测关键点是否在激活窗口内
    - 核心思路：为每个关键点 $k_j$ 预测 $p_p(k_j) = p(k_j \in AW | img)$，用二元交叉熵损失训练。当存在概率低于阈值时，模型不输出定位结果；高于阈值时再参考概率图。训练所需的"关键点在窗口外"样本通过裁剪数据增强产生。
    - 设计动机：现有方法把热力图峰值同时用作定位置信度和存在性判断，但这两个语义在数学上并不等价。将存在性独立建模后，CropCOCO 上的存在性分类误差降低了30-45%。

3. **裁剪数据增强与双热力图方法**:

    - 功能：生成训练样本并扩展定位范围
    - 核心思路：（1）随机裁剪训练图像使部分标注关键点落在窗口外，这些样本用于训练存在概率和概率图的空白输出；（2）双热力图方法在标准激活窗口之外增加一个更大的激活窗口（同分辨率但更大视野），专家热力图处理窗口内的精确定位，大窗口热力图处理更远的图像外点。当大窗口判断关键点在小窗口内时，由专家图精细化。
    - 设计动机：裁剪增强类似 Hide-and-Seek 的信息丢弃策略，不仅提供存在概率的训练数据，还改善了图像边界附近的定位精度（约+1% mAP）。双热力图是视野和精度之间的折中。

### 损失函数 / 训练策略

概率图使用修改的 OKS 损失（期望风险最小化 + Sobel 梯度正则化），存在概率使用二元交叉熵损失。概率图和存在概率通过温度缩放在 CropCOCO 上进行事后标定。所有训练在 COCO 上进行，裁剪增强在指定实验中启用。

## 实验关键数据

### 主实验

| 模型 | COCO mAP | CropCOCO mAP | CropCOCO Ex-mAP | OCHuman mAP |
|---|---|---|---|---|
| ViTPose-s | 75.9 | 72.7 | 66.5 | 60.3 |
| HRFormer-s | 75.2 | 70.9 | 64.3 | 60.3 |
| **ProbPose-s** | **76.6** | **81.7** | **73.9** | 60.4 |
| ProbPose-s-DH | 76.2 | 80.9 | 71.4 | **61.4** |

ProbPose 在 CropCOCO 上实现了巨大提升（mAP 从72.7提升到81.7，+9%），同时在标准 COCO 上也有小幅提升（75.9→76.6）。

### 消融实验

| 配置 | COCO mAP | CropCOCO mAP |
|---|---|---|
| ViTPose-s 基线 | 75.9 | 72.7 |
| + 裁剪增强 | ~76.5 | ~79 |
| + 概率图 | ~76 | ~80 |
| + 存在概率(ProbPose) | 76.6 | 81.7 |

每个组件都带来增益，裁剪增强效果最显著，存在概率在 Ex-mAP 上带来额外提升。

### 关键发现

- 存在概率在 CropCOCO 上比用热力图峰值做存在性判断的误差降低了30%（平衡数据集上45%）
- 常用的置信度阈值0.3接近最优，但不同数据集的最优阈值差异较大（0.15-0.4）
- 双热力图方法在 OCHuman（多人遮挡）上带来提升，扩大的视野有助于区分被遮挡的个体
- 概率图的标定曲线近似对角线，表明标定后的概率真正反映了真实的定位不确定性
- COCO 标注本身在边界附近有偏差——标注者倾向于避免在图像边缘放置关键点

## 亮点与洞察

1. **概念清晰**：将热力图混淆的三个语义（位置、质量、存在性）拆分为独立输出，每个都有明确的概率解释
2. **图像外关键点的重新定义**：将图像边界视为遮挡的一种特殊形式，这个视角很有启发性
3. **期望 OKS 最大化**：比简单 argmax 更鲁棒，尤其在双峰分布时不会被尖锐的小峰误导
4. **实用性强**：标定的概率对安全关键应用（如人机交互）至关重要——模型能表达"我不确定"

## 局限与展望

- 实验仅在 ViT-s 规模进行，未验证大模型上的增益是否保持
- CropCOCO 通过人工裁剪创建，与真实场景的图像外关键点分布可能有差异
- 双热力图的精度-视野权衡导致在 COCO 上略有下降
- 未来可将概率建模扩展到3D姿态估计和多人场景

## 相关工作与启发

- 与 ViTPose 的关系：ProbPose 基于 ViTPose 架构，通过改变输出表示和损失函数实现提升
- 与 RLE 的关系：RLE 使用回归方法定位图像外点，ProbPose 保持热力图范式但加入存在概率
- OKSLoss 最初由先前工作提出用于预测关键点，本文将其扩展到概率图的每个像素
- 启发：评估指标的局限性（忽略图像外点）实际上在引导模型往错误方向优化

## 评分

- **新颖性**: 8/10 — 概率图、存在概率、期望OKS最大化都是有理论支撑的新设计
- **实验充分度**: 8/10 — 多数据集评测、消融详细、新基准构建
- **写作质量**: 8/10 — 问题分析清晰，概率框架的数学表述严谨
- **价值**: 7/10 — 对姿态估计的实用性改进明确，但领域影响面相对较窄

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Any6D: Model-free 6D Pose Estimation of Novel Objects](any6d_model-free_6d_pose_estimation_of_novel_objects.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[CVPR 2025\] UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image](unopose_unseen_object_pose_estimation_with_an_unposed_rgb-d_reference_image.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/object_detection/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)

<!-- RELATED:END -->
