---
title: >-
  [论文解读] Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models
description: >-
  [ECCV 2024][3D视觉][开放词汇3D语义分割] 提出 Diff2Scene，首次将预训练的文本-图像扩散模型（Stable Diffusion）用于开放词汇3D语义分割，通过创新的掩码蒸馏方法将2D基础模型的语义丰富mask嵌入迁移到3D几何感知mask模型，在 ScanNet200 上超越 SOTA 12%。
tags:
  - ECCV 2024
  - 3D视觉
  - 开放词汇3D语义分割
  - 扩散模型
  - 掩码蒸馏
  - Mask2Former
---

# Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2407.13642](https://arxiv.org/abs/2407.13642)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 开放词汇3D语义分割, 扩散模型, 掩码蒸馏, Mask2Former, Stable Diffusion  

## 一句话总结

提出 Diff2Scene，首次将预训练的文本-图像扩散模型（Stable Diffusion）用于开放词汇3D语义分割，通过创新的掩码蒸馏方法将2D基础模型的语义丰富mask嵌入迁移到3D几何感知mask模型，在 ScanNet200 上超越 SOTA 12%。

## 研究背景与动机

- **领域现状**: 开放词汇3D语义分割致力于为每个3D点分配任意文本描述的语义标签，现有方法主要基于 CLIP 的逐点特征蒸馏（如 OpenScene）
- **现有痛点**: CLIP 基础模型在细粒度类别和组合式文本查询上表现不佳，且其全局表征优化目标不适合需要局部精细表征的稠密预测任务
- **核心矛盾**: 扩散模型在生成任务中展出的强大局部表征和文本对齐能力尚未被用于3D语义理解；但其生成性特征无法直接用于感知任务的逐点蒸馏
- **本文解决什么**: 如何有效利用扩散模型的语义丰富表征进行开放词汇3D分割，特别是克服扩散特征不能直接逐点蒸馏的问题
- **切入角度**: 采用 mask-based 分割范式（Mask2Former风格），利用mask嵌入解耦语义和空间信息，实现从2D到3D的跨模态掩码蒸馏
- **核心idea**: 用2D分支的语义丰富mask嵌入作为固定分类器，让3D分支学习生成几何准确的3D掩码，实现"语义来自2D，几何来自3D"的协同

## 方法详解

### 整体框架

Diff2Scene 包含两个分支：**2D语义理解分支**（基于扩散模型的开放词汇2D分割模型 ODISE）和**3D几何感知mask模型**（基于 MinkowskiNet）。2D分支从 RGB 图像预测 salient mask 及其语义嵌入；3D分支以点云和2D mask嵌入为输入，预测 geometric mask。推理时集成两种mask的预测结果，融合显著性模式和几何信息。

### 关键设计

**模块一：2D语义理解模型（扩散backbone + Mask2Former）**

采用 ODISE 作为2D分支，以 Stable Diffusion（Laion-5B 预训练）为特征骨干，配合 Mask2Former 分割头。模型接受2D图像输入，预测 $N$ 个2D概率掩码 $\{\mathcal{B}_i^{2d}\}_{i=1}^N$ 及其语义嵌入 $\{f_i^{2d}\}_{i=1}^N$。扩散特征维度256，CLIP特征维度768，mask查询数量 $N=100$。

**关键优势**: 扩散模型的生成式预训练赋予了强大的局部表征能力，比 CLIP 的全局对比学习更适合稠密预测；mask-based 范式天然解耦了语义和空间信息。

**模块二：几何感知3D掩码模型**

3D分支用 MinkowskiNet18A 从点云提取特征 $\mathbf{F}^{3d} \in \mathbb{R}^{M \times D}$。2D mask嵌入作为线性分类器计算每个3D点属于对应类别的 logit：

$$\mathcal{S}_i = \langle \mathbf{F}^{3d}, f_i^{2d} \rangle$$

通过 sigmoid 函数得到3D概率掩码 $\mathcal{B}'^{3d}_i$。

**模块三：跨模态掩码蒸馏**

核心创新在于掩码蒸馏损失。将2D掩码通过像素-点对应关系提升到3D空间得到 $\mathcal{B}_i^{3d}$，然后约束3D分支预测的掩码与之一致：

$$\mathcal{L} = \sum_{i=1}^{N} 1 - \cos(\mathcal{B}'^{3d}_i, \mathcal{B}_i^{3d})$$

这一损失隐式地迫使3D模型学习高分辨率、语义丰富的特征表征，而无需直接在像素层面蒸馏冻结的扩散特征（后者会导致训练不收敛）。

### 损失函数 / 训练策略

- **蒸馏损失**: 掩码级余弦相似性损失（Eq. 2），而非传统的逐点特征距离
- **推理集成**: 融合 salient mask 和 geometric mask 的预测：$\mathbf{p}^c = \lambda \sum_i p_i^c \cdot \mathcal{B}_i^{3d} + (1-\lambda) \sum_i p_i^c \cdot \mathcal{B}'^{3d}_i$，$\lambda = 0.5$
- **训练配置**: 200 epochs，batch size 8，Adam 优化器 lr=0.0001，polynomial 调度策略 power=0.9
- **零标注训练**: 无需任何3D ground truth 标注，仅用训练集的 RGB 图像和重建点云
- **文本推理**: 使用 ViT-L/14 CLIP 提取文本嵌入，同时融合判别性（CLIP）和生成性（Stable Diffusion）语义特征

## 实验关键数据

### 主实验

| 方法 | ScanNet | Matterport3D | ScanNet200 Head | ScanNet200 Common | ScanNet200 Tail | ScanNet200 All | Replica All |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenScene (2D/3D) | 47.5 | 42.6 | 20.0 | 9.7 | 5.1 | 11.6 | 14.9 |
| OpenMask3D | 34.0 | - | 19.6 | 7.5 | 4.5 | 10.5 | 4.8 |
| ConceptFusion | 33.3 | - | 17.5 | 6.3 | 2.8 | 8.8 | 4.6 |
| **Diff2Scene** | **48.6** | **45.5** | **25.6** | **11.5** | **6.9** | **14.2** | **17.5** |

ScanNet200 上整体 mIoU 14.2% vs OpenScene 11.6%（+22% 相对提升），尾部类别 6.9% vs 5.1%（+35%相对提升）。

### 消融实验

| 消融项 | Replica mIoU |
|--------|:---:|
| 完整模型 | 17.5 |
| 去除2D salient mask | 12.8 |
| 去除3D geometric mask | 16.5 |
| 去除 CLIP 判别性特征 | 15.5 |
| 去除 Stable Diffusion 生成性特征 | 15.3 |

| 蒸馏方式 | 蒸馏类型 | Replica Head | Replica Tail | Replica All |
|---------|---------|:---:|:---:|:---:|
| Fine-tuned CLIP (OpenScene) | Point-based | 32.6 | 7.7 | 11.1 |
| Frozen diffusion feature | Point-based | 发散 | 发散 | 发散 |
| **Multimodal mask distillation** | **Mask-based** | **43.3** | **8.0** | **12.8** |

### 关键发现

- **扩散特征不能直接逐点蒸馏**: 冻结的 Stable Diffusion 特征用于逐点蒸馏会导致训练不收敛，这是mask蒸馏设计的直接动机
- **判别性与生成性特征互补**: 去除任一类型特征都会导致约 2% mIoU 下降，两者联合使用效果最佳
- **salient mask vs geometric mask**: 2D salient mask 贡献更大（去除后 -4.7%），但3D geometric mask 也不可或缺（去除后 -1.0%）
- 在未见过的 Replica 数据集上显著领先（17.5 vs 14.9），证明泛化能力
- 尾部类别性能接近全监督方法（6.9 vs CSC-Pretrain 7.9），展示长尾分布处理潜力

## 亮点与洞察

- **首次将扩散模型引入开放词汇3D分割**: 开创性地验证了生成式预训练表征在3D理解中的价值
- **掩码蒸馏范式巧妙**: 通过"语义嵌入做分类器+3D特征做被分类对象"绕过了扩散特征不能直接蒸馏的难题
- **组合式查询能力强**: 可处理颜色+形状+位置+用途的复合文本查询（如"find the white sneakers closer to the desk chair"）
- **设计哲学清晰**: "语义来自2D基础模型，几何准确性来自3D模型"的分工合理

## 局限与展望

- 小尺寸稀有类别（如 rail）仍容易误分类
- 语义相近的细粒度类别容易混淆（如 windowsill vs window）
- 推理需要同时运行2D扩散模型和3D模型，计算成本较高
- 2D分支使用的 ODISE 模型本身依赖大量预训练，端到端训练的探索不足
- 未在室外数据集（如 nuScenes）上验证

## 相关工作与启发

- **OpenScene**: 主要基线，首创无标注逐点蒸馏范式，但受限于 CLIP 特征的局部表征能力
- **ODISE**: 2D分支的直接来源，将 Stable Diffusion 用于2D开放词汇分割
- **Mask2Former**: mask-based 分割范式的奠基工作，解耦mask预测与语义分类
- **启发**: 扩散模型作为3D特征提取的"语义桥梁"可延伸到3D目标检测、实例分割等任务；mask蒸馏范式可推广到其他跨模态知识迁移场景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将扩散模型用于3D开放词汇分割，掩码蒸馏设计新颖且动机充分
- **实验充分度**: ⭐⭐⭐⭐ — 四个数据集验证+消融实验完整，但缺少室外场景和效率分析
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法和基线的对比图（Fig. 2）直观易懂
- **实用价值**: ⭐⭐⭐⭐ — ScanNet200 长尾类别上表现出色，对实际应用场景具有参考价值

<!-- RELATED:START -->

## 相关论文

- [DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators](dreamdrone_text-to-image_diffusion_models_are_zero-shot_perpetual_view_generator.md)
- [3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)
- [Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)

<!-- RELATED:END -->
