---
title: >-
  [论文解读] SAS: Segment Any 3D Scene with Integrated 2D Priors
description: >-
  [ICCV 2025][3D视觉][开放词汇3D分割] 提出 SAS 框架，首次整合多个 2D 开放词汇模型的互补能力来学习更好的 3D 表示：通过 Model Alignment via Text 对齐不同模型的特征空间…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "开放词汇3D分割"
  - "多模型融合"
  - "知识蒸馏"
  - "扩散模型"
  - "模型能力构建"
---

# SAS: Segment Any 3D Scene with Integrated 2D Priors

**会议**: ICCV 2025  
**arXiv**: [2503.08512](https://arxiv.org/abs/2503.08512)  
**代码**: [项目页面](https://peoplelu.github.io/SAS.github.io)  
**领域**: 3D视觉  
**关键词**: 开放词汇3D分割, 多模型融合, 知识蒸馏, 扩散模型, 模型能力构建

## 一句话总结

提出 SAS 框架，首次整合多个 2D 开放词汇模型的互补能力来学习更好的 3D 表示：通过 Model Alignment via Text 对齐不同模型的特征空间，通过 Annotation-Free Model Capability Construction 利用扩散模型合成图像来量化各模型识别不同类别的能力，以此指导多模型特征融合和 3D 蒸馏，在 ScanNet v2/Matterport3D/nuScenes 上大幅超越前作。

## 研究背景与动机

3D 场景理解是自动驾驶、虚拟现实和机器人操控的基础任务。传统闭集方法限于固定类别集训练，无法识别未见类别，促使研究者关注开放词汇 3D 理解。

当前主流方案是将 2D 开放词汇模型（如 LSeg、SEEM）的能力通过特征蒸馏迁移到 3D 模型。但存在一个根本性问题：

**单一教师模型的错误传播**：2D 模型在某些类别上会犯错（如 SEEM 将"画"误识别为"墙"），蒸馏后 3D 模型会继承相同的错误。

**直觉解法——多模型融合——面临两大困难**：
   - **特征空间不对齐**：不同 2D 模型（如 LSeg 用 CLIP、SEEM 用自己的编码器）有不同的图像-文本特征空间，无法直接融合
   - **模型能力量化困难**：需要知道各模型在哪些类别上更强，但开放词汇任务本身就是零样本的，获取测试图像和标注不切实际

**核心洞察**：不同 2D 模型在不同类别上有互补的识别能力。如果能量化每个模型对每个类别的识别能力，就可以对每个点选择更可靠的模型特征，纠正单模型的错误识别。

## 方法详解

### 整体框架

SAS 包含四个阶段：(1) 通过文本桥接对齐多个 2D 模型的特征空间；(2) 利用扩散模型合成图像评估各模型在各类别上的能力；(3) 以构建的模型能力为指导融合多模型的点特征；(4) 通过超点蒸馏和时间集成自蒸馏将融合知识迁移到 3D 网络。

### 关键设计

1. **Model Alignment via Text（文本桥接对齐）**：不同 2D 模型有不同的特征空间，核心思路是用文本作为统一的桥接。对于 SEEM 输出的每个 mask，先用预训练 captioner (TAP) 生成描述性 caption（包含颜色、形状等），再将 caption 中的名词替换为 SEEM 预测的标签（增加语义准确性），最后用 CLIP 文本编码器统一编码。这样 LSeg（本身对齐 CLIP）和 SEEM（通过文本桥接到 CLIP）的特征就对齐到了同一个 CLIP 特征空间。通过像素-点对应关系将像素特征映射到 3D 点上，得到对齐的点特征 $F^{2D}_L$ 和 $F^{2D}_S$。

2. **Annotation-Free Model Capability Construction（无标注模型能力构建）**：这是最具创新性的设计。面临的困境是需要评估模型能力但没有测试数据和标注。解决方案：

    - 使用 Stable Diffusion 为预定义词汇表中的每个类别生成 $m$ 张合成图像
    - 利用扩散模型内部的交叉注意力图 $M_x$ 定位目标物体，结合 SAM 生成精确掩码作为伪标签 $\mathbf{M}^{Pseudo}_{i,j}$
    - 让 LSeg 和 SEEM 分别对合成图像推理，得到各自的掩码预测
    - 用 mIoU 衡量各模型在各类别上的识别能力：
    $S^{LSeg}_j = \frac{1}{m}\sum_{i=1,...,m} \mathbf{mIoU}(\mathbf{M}^{Pseudo}_{i,j}, \mathbf{M}^{LSeg}_{i,j})$
    - 最终构建出各模型的能力向量 $S_L = [S^{LSeg}_1, ..., S^{LSeg}_K]$

3. **Capability-Guided Feature Fusion（能力引导特征融合）**：

    - 用 CLIP 编码词汇表得文本特征 $F_{text}$
    - 计算每个点在 LSeg 和 SEEM 下的预测类别 $\mathcal{P}_{LSeg}$ 和 $\mathcal{P}_{SEEM}$
    - 以两个可能正确类别的能力之和作为各模型做出正确预测的概率
    - 用 softmax 温控加权融合：
    $F^{2D}_{fusion} = \frac{\exp(\mathcal{P}_{LSeg}/\tau)}{\exp(\mathcal{P}_{LSeg}/\tau) + \exp(\mathcal{P}_{SEEM}/\tau)} F^{2D}_L + \frac{\exp(\mathcal{P}_{SEEM}/\tau)}{\exp(\mathcal{P}_{LSeg}/\tau) + \exp(\mathcal{P}_{SEEM}/\tau)} F^{2D}_S$

4. **Superpoint Distillation + Temporal Ensembling Self-Distillation**：

    - 超点蒸馏：提取语义一致的超点，对超点内特征取均值后蒸馏，平滑 2D 模型不一致预测
    - 时间集成自蒸馏：用 EMA 累积历史输出 $\hat{F}^{3D} = \alpha \hat{F}^{3D} + (1-\alpha) F^{3D}$，以历史预测的伪标签自监督当前模型，比 GGSD 的 Mean-teacher 更稳定（避免用可变学生模型监督教师模型导致的训练崩溃）

### 损失函数 / 训练策略

蒸馏阶段：
- **超点蒸馏损失**：$\mathcal{L} = \mathcal{L}_p + \mathcal{L}_{sp}$，分别在点级和超点级计算余弦相似度损失
- **自蒸馏损失**：$\mathcal{L} = \mathcal{L}^{ST}_p + \mathcal{L}^{ST}_{sp}$，使用交叉熵损失

训练 100 epoch：前 70 epoch 仅用超点蒸馏，后 30 epoch 叠加自蒸馏。室内/室外使用不同的预构建词汇表。nuScenes 由于点云以道路为主，不使用超点（每个点即超点）。

## 实验关键数据

### 主实验

| 数据集 | 指标 (mIoU) | 本文 | 之前SOTA (OV3D) | 提升 |
|--------|------------|------|-----------------|------|
| ScanNet v2 | mIoU | 61.9 | 57.3 | +4.6 |
| Matterport3D | mIoU | 48.6 | 45.8 | +2.8 |
| nuScenes | mIoU | 47.5 | 44.6 (Seal=45.0) | +2.5 |

SAS 在所有三个数据集上均为零样本方法的 SOTA。在 ScanNet v2 上 61.9 的 mIoU 已接近部分旧的全监督方法（如 PointConv 61.0）。

### 消融实验

| 配置 | ScanNet v2 | Matterport | nuScenes | 说明 |
|------|-----------|-----------|---------|------|
| 2D LSeg 特征 | 51.2 | 38.6 | - | 仅 LSeg |
| 2D SEEM 特征 | 47.3 | 40.2 | 37.8 | 仅 SEEM |
| 2D 直接相加 | 48.6 | 39.1 | 34.1 | 简单融合无效 |
| 2D 线性融合 | 49.9 | 39.4 | 34.8 | 线性融合有限提升 |
| 2D 本文融合 | 55.5 | 43.6 | 40.0 | 能力引导显著提升 |
| 3D 像素-点蒸馏 | 56.7 | 45.1 | 45.4 | 蒸馏超越 2D 特征 |
| 3D 超点蒸馏 | 59.2 | 46.3 | 45.4 | 结构信息加持 |
| 3D +自蒸馏(完整) | 61.9 | 48.6 | 47.5 | 进一步挖掘潜力 |

扩展到3个 2D 模型（+ODISE）：ScanNet 62.5(+0.6)，Matterport 49.8(+1.2)，证明方法可扩展性。

长尾场景：在 Matterport K=40/80/160 类别设置下均显著优于 OpenScene（如 K=160: 8.1 vs 5.8）。

### 关键发现

- SEEM 在 Matterport3D 上优于 LSeg，但在 ScanNet v2 上反之——两个模型确实有互补能力
- 简单相加或线性融合反而可能比最好的单模型差，证明了能力引导融合的必要性
- 蒸馏后的 3D 模型总是优于 2D 融合特征（3D 空间一致性带来的增益）
- 超点蒸馏的结构先验和自蒸馏各自贡献约 2-3 个 mIoU 点
- 方法在长尾类别（K=160）下优势更明显，体现了开放词汇能力

## 亮点与洞察

- **核心创新点**：用扩散模型合成测试图像来量化教师模型能力，解决了零样本场景下无法评估模型的难题，这个思路非常巧妙且可推广
- 文本桥接对齐是解决多模型特征空间不兼容问题的优雅方案
- 时间集成自蒸馏比 GGSD 的 Mean-teacher 更稳定，避免了训练崩溃
- 推理时直接用纯 3D 模型输出，不需要 2D-3D ensemble，效率更高

## 局限与展望

- 合成图像质量受 Stable Diffusion 限制，某些细粒度类别生成效果可能不佳
- 跨注意力图定位并不总是精确，SAM 修正也可能引入误差
- 当前仅融合2-3个 2D 模型，更多模型的融合策略和效率需要探索
- 超点提取方法（如颜色/法线聚类）的质量会影响蒸馏效果
- 未探索利用 3D 几何信息反过来指导 2D 特征选择的可能性

## 相关工作与启发

- OpenScene 开创了 2D→3D 特征蒸馏范式，SAS 在"用更好的 2D 特征"方向上做了重要推进
- GGSD 的 Mean-teacher 自蒸馏启发了 SAS 的时间集成设计
- Stable Diffusion 的交叉注意力图被巧妙用于定位物体，拓展了扩散模型作为工具的应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出无标注构建模型能力的方法，创新性很强
- 实验充分度: ⭐⭐⭐⭐⭐ 三大数据集、长尾评估、多任务扩展、详尽消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 多教师模型融合的范式对开放词汇3D理解具有广泛影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[ICCV 2025\] Scene Coordinate Reconstruction Priors](scene_coordinate_reconstruction_priors.md)
- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](../../NeurIPS2025/3d_vision/online_segment_any_3d_thing_as_instance_tracking.md)
- [\[ICCV 2025\] AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)
- [\[ICCV 2025\] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)

</div>

<!-- RELATED:END -->
