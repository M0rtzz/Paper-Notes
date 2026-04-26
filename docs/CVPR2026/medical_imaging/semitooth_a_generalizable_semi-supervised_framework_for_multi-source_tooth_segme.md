---
title: >-
  [论文解读] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation
description: >-
  [CVPR 2026][医学图像][牙齿分割] 提出 SemiTooth 框架，通过多教师多学生架构和严格加权置信度约束（SWC），解决多源 CBCT 牙齿分割中的标注稀缺和跨源域间差异问题，同时构建了首个多源半监督牙齿数据集 MS3Toothset。
tags:
  - CVPR 2026
  - 医学图像
  - 牙齿分割
  - CBCT
  - 半监督学习
  - 多源数据
  - 多教师多学生
---

# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 牙齿分割, CBCT, 半监督学习, 多源数据, 多教师多学生

## 一句话总结

提出 SemiTooth 框架，通过多教师多学生架构和严格加权置信度约束（SWC），解决多源 CBCT 牙齿分割中的标注稀缺和跨源域间差异问题，同时构建了首个多源半监督牙齿数据集 MS3Toothset。

## 研究背景与动机

CBCT 牙齿结构分割是智能口腔诊疗的核心任务，但面临两大挑战：

**标注数据稀缺**：体素级标注耗时昂贵，大量去标识化 CBCT 数据未被利用

**多源域间差异**：不同机构/设备的 CBCT 数据在密度分布、强度分布和特征空间上存在显著差距（由 Kernel Density Estimation 和 t-SNE 可视化验证），导致模型跨源泛化困难

现有半监督医学分割方法（Mean Teacher、UA-MT、MCF 等）主要针对单源数据设计，缺乏跨源知识迁移能力。而多源方法（ASDA、Dual-Teacher）要么需要复杂网络或强监督，要么缺乏在多源 CBCT 牙齿数据上的验证。

## 方法详解

### 整体框架

SemiTooth 采用**三学生+两教师**的多分支架构。数据组织为三个子集：
- **Main**（标注数据，来自主源）
- **Other**（未标注数据，来自其他源）
- **Mixed**（未标注数据，与主源分布相似，通过 Wasserstein 距离度量筛选）

每个子集由对应的学生网络处理，两个教师分别监督 Mixed 和 Other 学生，通过 EMA 更新：

$$\theta_t^{(k)} \leftarrow \gamma \theta_t^{(k-1)} + (1-\gamma) \theta_s^{(k)}, \quad \gamma = 0.99$$

### 关键设计

1. **多教师多学生架构（SemiTooth）**：

    - 与 Mean Teacher（单教师-单学生）相比：SemiTooth 为不同源的数据分配专属学生，教师提供跨源知识引导
    - 与 Co-training（多学生共享权重无教师）相比：SemiTooth 通过 EMA 教师提供稳定伪标签
    - **Mixed 子集**作为源间桥梁：包含与主源分布相似的未标注样本，缓解直接跨源训练的不稳定性
    - 学生网络间共享相似架构促进知识迁移，同时保持足够多样性

2. **严格加权置信度约束（SWC）**：

   解决 CBCT 异质性引入噪声导致一致性正则化退化的问题。SWC 融合区域级门控和体素级加权：

    - **区域级门控**：将样本均匀分为非重叠立方体区域 $\{r\}$，计算区域置信度 $c(r) = \mathbb{E}_{i \in r}[\max_c P^T_{i,c}]$，低于阈值 $\tau=0.9$ 的区域视为不可靠并丢弃
    - **体素级加权**：在保留区域内，用体素级置信度 $c_i = \max_c P^T_{i,c}$ 加权教师-学生对齐：
    $\mathcal{SWC}(P^S, P^T) = \mathbb{E}_{r \in \mathcal{R}_\tau}\left[\mathbb{E}_{i \in r}\left[c_i \cdot \mathcal{A}(P^S_i, P^T_i)\right]\right]$
    - 该设计特别适合 3D CBCT 数据：区域级过滤利用了体积数据的空间连续性

3. **多源数据集 MS3Toothset**：

    - 收集三个来源：ShanghaiTech（公开、有标注）、PKU-SS 和 AFMC（私有、无标注）
    - 经过筛选处理后包含 98 个标注样本（20 个测试）和 438 个未标注样本
    - 首个面向多源半监督牙齿分割的综合数据集

### 损失函数 / 训练策略

总损失结合监督损失和两个 SWC 一致性损失：

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{cons}^u + \beta \mathcal{L}_{cons}^h, \quad \alpha = \beta = 0.5$$

- $\mathcal{L}_{sup} = \text{CE}(P^S(x^l), y)$：主源标注数据的监督损失
- $\mathcal{L}_{cons}^u$, $\mathcal{L}_{cons}^h$：分别对应 Other 和 Mixed 源的 SWC 损失
- 骨干网络 V-Net，Adam 优化器，lr=0.0001，训练 300 epochs

## 实验关键数据

### 主实验

| 方法 | mIoU | Dice | Recall | Acc |
|------|------|------|--------|-----|
| V-Net (全监督基线) | 61.36 | 73.65 | 70.77 | 66.75 |
| Mean Teacher | 67.69 | 78.72 | 78.06 | 73.68 |
| UA-MT | 68.37 | 79.18 | 80.42 | 76.17 |
| ASDA | 73.75 | 83.63 | 80.93 | 78.79 |
| CMT | 76.14 | 85.07 | 87.14 | 84.32 |
| Uni-HSSL | 75.76 | 85.42 | 84.26 | 81.88 |
| **SemiTooth** | **76.67** | **85.69** | **88.66** | **86.44** |

### 消融实验

| Exp | 模块组合 | mIoU | Dice | Recall | Acc |
|-----|---------|------|------|--------|-----|
| 1 | V-Net | 61.36 | 73.65 | 70.77 | 66.75 |
| 2 | + Mean Teacher | 67.69 | 78.72 | 78.06 | 73.68 |
| 3 | + SWC (无 SemiTooth) | 69.94 | 80.29 | 79.67 | 75.34 |
| 4 | + SemiTooth (无 SWC) | 75.37 | 84.56 | 83.07 | 80.48 |
| 5 | + SemiTooth + SWC | **76.67** | **85.69** | **88.66** | **86.44** |

### 关键发现

- SemiTooth 多分支架构贡献最大（Exp 2→4: mIoU +7.68），说明多源数据的源感知学习至关重要
- SWC 在单教师（Exp 2→3: +2.25 mIoU）和多教师（Exp 4→5: +1.30 mIoU）上均有稳定提升
- Recall 指标上 SemiTooth 相对优势最明显（88.66 vs 次优 87.14），对临床漏诊率的降低有意义
- t-SNE 可视化显示 SemiTooth 有效缩小了多源特征分布差距

## 亮点与洞察

- **数据集贡献**：MS3Toothset 填补了多源半监督牙齿分割数据集的空白
- SWC 的区域级+体素级两阶段过滤直觉清晰，适合 3D 医学数据的空间连续性
- Mixed 子集通过 Wasserstein 距离选择分布相似样本作为源间桥梁，设计简洁有效
- 消融实验层次分明，清楚展示了各组件的独立和联合贡献

## 局限与展望

- MS3Toothset 仅 98 个标注样本 + 438 未标注，规模较小
- 仅在自有数据集上验证，泛化到公开标准基准的能力未知
- 教师数量和子集划分策略依赖 Wasserstein 距离阈值的选择，缺乏敏感性分析
- 仅使用 V-Net 骨干，未验证在更强骨干（nnUNet、Swin UNETR）上的效果
- SWC 的区域大小和阈值 $\tau$ 的消融不够充分

## 相关工作与启发

- 与 CMT（ACM MM 2024）和 Uni-HSSL（CVPR 2025）等最新多源方法对比，SemiTooth 取得优势
- 多教师多学生范式可推广到其他多域半监督场景（如多中心 CT/MRI 分析）
- SWC 约束可适配到任何基于 Mean Teacher 的框架中

## 评分

- 新颖性: ⭐⭐⭐ 多教师多学生和置信度约束思路不算新，但在牙齿分割场景的系统集成有价值
- 实验充分度: ⭐⭐⭐ 消融完整但仅在单个自有数据集上验证，缺乏外部验证
- 写作质量: ⭐⭐⭐ 框架描述清晰，但论文较短（ICASSP 篇幅），细节不足
- 价值: ⭐⭐⭐ 数据集贡献有临床价值，方法泛化性待验证

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_th.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semisupervised_framework_for_breast_ultrasound_s.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
