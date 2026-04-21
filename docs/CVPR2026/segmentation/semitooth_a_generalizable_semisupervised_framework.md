---
title: >-
  [论文解读] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation
description: >-
  [CVPR 2026][图像分割][半监督学习] 本文提出SemiTooth框架，通过多教师-多学生架构和更严格的加权置信度约束（SWC），解决多源CBCT数据在半监督牙齿分割中的分布差异问题，在构建的MS3Toothset数据集上取得SOTA。
tags:
  - CVPR 2026
  - 图像分割
  - 半监督学习
  - 多源数据
  - 牙齿分割
  - CBCT
  - 伪标签
---

# SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.11616](https://arxiv.org/abs/2603.11616)  
**代码**: 无  
**领域**: 分割 / 医学图像  
**关键词**: 半监督学习, 多源数据, 牙齿分割, CBCT, 伪标签

## 一句话总结
本文提出SemiTooth框架，通过多教师-多学生架构和更严格的加权置信度约束（SWC），解决多源CBCT数据在半监督牙齿分割中的分布差异问题，在构建的MS3Toothset数据集上取得SOTA。

## 研究背景与动机

1. **领域现状**：CBCT（锥束CT）牙齿结构分割是临床口腔诊疗的基础任务，全监督方法取得了不错进展，但体素级标注耗时昂贵，大量去标识化的未标注CBCT数据未能有效利用。
2. **现有痛点**：半监督医学图像分割（SSMIS）方法如Mean Teacher可以利用无标签数据，但主要针对单源数据设计。实际中不同医院/设备采集的CBCT存在显著分布差异（密度、强度、特征分布都不同），直接混用会导致模型泛化能力差。且公开的多源CBCT牙齿分割数据集稀缺。
3. **核心矛盾**：多源数据之间的域差距（domain gap）导致统一训练困难——单教师-单学生的简单框架无法同时兼顾多个数据源的特征差异，伪标签质量在跨源场景下严重退化。
4. **本文目标** ①构建一个多源半监督CBCT牙齿分割数据集；②设计一个能处理多源分布差异的半监督框架。
5. **切入角度**：针对不同数据源分配不同的学生网络进行专门学习，用各自对应的教师网络提供监督，同时设计区域级置信度约束过滤噪声伪标签。
6. **核心 idea**：多教师-多学生分支式框架 + 区域级严格置信度约束，让每个数据源都能在半监督学习中被有效利用。

## 方法详解

### 整体框架
SemiTooth将所有数据源重构为三个子集：main（有标签的主源数据）、other（其他源的无标签数据）、mixed（分布与主源相似的无标签样本，通过Wasserstein距离度量源间相似度归入）。设置3个学生网络分别处理三个子集，2个教师网络分别监督mixed和other子集的学生网络。教师通过EMA更新：$\theta_t^{(k)} \leftarrow \gamma \theta_t^{(k-1)} + (1-\gamma) \theta_s^{(k)}$。

### 关键设计

1. **多教师-多学生架构（SemiTooth）**

    - 功能：让不同源的数据在各自的学生网络中进行专门化学习，避免多源混合训练干扰。
    - 核心思路：与Mean Teacher（单教师-单学生）不同，SemiTooth为每个数据子集配备独立的学生网络，同时两个教师以EMA方式从相应学生更新参数，提供稳定的伪标签监督。学生网络共享相似架构促进知识迁移，但保持足够的多样性。mixed子集作为分布桥梁连接主源和其他源。
    - 设计动机：单一教师-学生框架在多源场景下伪标签质量不稳定，且无法区分不同源的特征风格。通过"专人专事"的分支设计，每个学生可以学到源感知的知识表示。

2. **更严格的加权置信度约束（SWC）**

    - 功能：在多源异构数据上提供可靠的一致性正则化信号，过滤噪声伪标签。
    - 核心思路：将预测概率图均匀划分为不重叠的立方体区域 $\{r\}$，对每个区域计算教师的区域置信度 $c(r) = \mathbb{E}_{i \in r}[\max_c P_{i,c}^T]$，低于阈值 $\tau$ 的区域直接丢弃。在保留的可靠区域内，用体素级置信度加权教师-学生对齐损失：$\mathcal{SWC}(P^S, P^T) = \mathbb{E}_{r \in \mathcal{R}_\tau}[\mathbb{E}_{i \in r}[c_i \cdot \mathcal{A}(P_i^S, P_i^T)]]$。
    - 设计动机：CBCT数据异构性导致普通一致性正则化引入大量噪声，尤其在牙齿边界等难区域。SWC先做区域级门控排除不可靠区域，再在体素级加权精炼对齐信号，兼顾结构可靠性和体素精度，特别适合3D CBCT分割。

3. **MS3Toothset数据集**

    - 功能：提供标准化的多源半监督CBCT牙齿分割评测基准。
    - 核心思路：融合ShanghaiTech的半标注数据与PKU-SS和AFMC的无标注私有数据，筛选处理后包含98个有标签样本（20个测试集）和438个无标签样本。
    - 设计动机：现有公开CBCT牙齿数据集稀缺且通常单源，无法支撑多源半监督方法的开发和验证。

### 损失函数 / 训练策略
总损失为三部分之和：有标签数据的监督损失 $\mathcal{L}_{sup} = CE(P^S(x^l), y)$，以及两个SWC一致性损失对应other源和mixed源：

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{cons}^u + \beta \mathcal{L}_{cons}^h$$

其中 $\alpha = \beta = 0.5$，SWC阈值 $\tau = 0.9$，EMA衰减率 $\gamma = 0.99$。backbone使用V-Net，Adam优化器，学习率0.0001，训练300 epochs。

## 实验关键数据

### 主实验

| 方法 | 发表 | 年份 | mIoU | Dice | Recall | Acc |
|------|------|------|------|------|--------|-----|
| V-Net | IEEE 3DV | 2016 | 61.36 | 73.65 | 70.77 | 66.75 |
| MT | NeurIPS | 2017 | 67.69 | 78.72 | 78.06 | 73.68 |
| UA-MT | MICCAI | 2019 | 68.37 | 79.18 | 80.42 | 76.17 |
| ASDA | IEEE TIP | 2022 | 73.75 | 83.63 | 80.93 | 78.79 |
| CMT | ACM MM | 2024 | 76.14 | 85.07 | 87.14 | 84.32 |
| Uni-HSSL | CVPR | 2025 | 75.76 | 85.42 | 84.26 | 81.88 |
| **SemiTooth** | - | 2025 | **76.67** | **85.69** | **88.66** | **86.44** |

### 消融实验

| Exp | V-Net | MT | ST | SWC | mIoU | Dice | Recall | Acc |
|-----|-------|----|----|-----|------|------|--------|-----|
| 1 | ✓ | | | | 61.36 | 73.65 | 70.77 | 66.75 |
| 2 | ✓ | ✓ | | | 67.69 | 78.72 | 78.06 | 73.68 |
| 3 | ✓ | ✓ | | ✓ | 69.94 | 80.29 | 79.67 | 75.34 |
| 4 | ✓ | ✓ | ✓ | | 75.37 | 84.56 | 83.07 | 80.48 |
| 5 | ✓ | ✓ | ✓ | ✓ | **76.67** | **85.69** | **88.66** | **86.44** |

### 关键发现
- 从Exp2到Exp4看，SemiTooth的多教师-多学生架构贡献最大（mIoU +7.68%），说明多分支设计是核心。
- SWC约束在MT基础上提升2.25% mIoU（Exp2 vs Exp3），在SemiTooth基础上提升1.3%（Exp4 vs Exp5），说明SWC在已有强框架上仍有稳定增益。
- t-SNE可视化证实SemiTooth能有效压缩多源特征分布的差距，实现跨源域泛化。
- 定性结果显示SemiTooth在牙根区域和相邻牙齿边界粘连问题上效果最好。

## 亮点与洞察
- **区域级 + 体素级的分层置信度过滤**是最精巧的设计：先用区域级粗粒度排除噪声密集区，再在可靠区域内用体素级细粒度加权。这种层次化过滤思路可以迁移到其他3D半监督分割任务中。
- **mixed子集作为分布桥梁**的设计值得注意：通过Wasserstein距离从无标签数据中找出与有标签源分布相似的样本，作为连接不同源的纽带。
- 整体框架相对简洁，没有引入过多复杂组件，但通过合理的分支设计和约束机制取得了不错效果。

## 局限与展望
- 数据集规模偏小（98有标签 + 438无标签），实验验证的说服力有限。
- 仅验证了V-Net作为backbone，未测试更强的3D分割backbone（如nnU-Net、Swin UNETR）。
- 多教师-多学生架构的计算和内存开销相对单教师更大（3个学生 + 2个教师），实际部署需考虑资源限制。
- mixed子集的划分依赖Wasserstein距离阈值的选取，文中未详细讨论这一超参的敏感性。
- 可以考虑引入对比学习进一步拉近跨源特征表示。

## 相关工作与启发
- **vs Mean Teacher [15]**: MT只有单教师-单学生，无法感知多源分布差异，SemiTooth通过多分支设计和专用教师显著提升跨源性能（mIoU +9%）。
- **vs CMT [20]**: CMT使用多学生共享权重的Co-training但无教师监督，缺乏稳定的伪标签指导，SemiTooth在mIoU上超过0.53%。
- **vs ASDA [12]**: ASDA是为多源半监督设计的域自适应方法，SemiTooth在Recall上大幅超过（+7.73%），说明对临床敏感性指标更友好。

## 评分
- 新颖性: ⭐⭐⭐ 多教师-多学生思路属于合理推进，SWC约束有新意但改进幅度有限
- 实验充分度: ⭐⭐⭐ 消融和对比较完整，但数据集偏小、backbone单一
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富直观
- 价值: ⭐⭐⭐ 对多源半监督医学分割有参考意义，但泛化性待更多验证

<!-- RELATED:START -->

## 相关论文

- [LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](../../ECCV2024/segmentation/lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction](../../ICCV2025/segmentation/conformalsam_unlocking_the_potential_of_foundational_segmentation_models_in_semi.md)
- [CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](ctfs_collaborative_teacher_framework_for_forward-looking_sonar_image_semantic_se.md)
- [Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](generalizable_knowledge_distillation_from_vision_foundation_models_for_semantic_.md)
- [GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)

<!-- RELATED:END -->
