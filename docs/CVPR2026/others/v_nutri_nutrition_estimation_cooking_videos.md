---
title: >-
  [论文解读] V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos
description: >-
  [CVPR 2026][营养估计] 提出 V-Nutri 框架，首次利用第一人称烹饪视频中的过程信息来辅助菜品营养估计，通过 VideoMamba 关键帧选择模块提取食材添加时刻，与最终成品图像融合来预测热量和宏量营养素。
tags:
  - "CVPR 2026"
  - "营养估计"
  - "第一人称视频"
  - "关键帧选择"
  - "多模态融合"
  - "食物分析"
---

# V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos

**会议**: CVPR 2026  
**arXiv**: [2604.11913](https://arxiv.org/abs/2604.11913)  
**代码**: [https://github.com/K624-YCK/V-Nutri](https://github.com/K624-YCK/V-Nutri)  
**领域**: 食物计算 / 视频理解  
**关键词**: 营养估计, 第一人称视频, 关键帧选择, 多模态融合, 食物分析

## 一句话总结

提出 V-Nutri 框架，首次利用第一人称烹饪视频中的过程信息来辅助菜品营养估计，通过 VideoMamba 关键帧选择模块提取食材添加时刻，与最终成品图像融合来预测热量和宏量营养素。

## 研究背景与动机

**领域现状**：视觉营养估计方法主要依赖最终成品的单张图像来预测热量和营养成分，如 Nutrition5K 和 Im2Calories 等工作。

**现有痛点**：单张成品图像的信息本质上是有限的：油、酱汁、乳制品等营养重要成分在烹饪后会被吸收、融化或视觉融合到最终菜品中，使得仅从外观难以准确估计。

**核心矛盾**：营养关键信息在烹饪过程中逐步"消失"，但现有方法仅利用信息最少的最终状态。

**本文目标**：探索烹饪视频中的过程信息是否能为菜品级营养估计提供互补证据。

**切入角度**：第一人称烹饪视频保留了完整的时间营养证据（食材身份、添加事件、中间状态），且可穿戴相机的普及使这一方向越来越实用。

**核心 idea**：用关键帧选择模块从长视频中定位营养信息密集的时刻（如食材添加），与成品图像融合来提升营养估计准确性。

## 方法详解

### 整体框架

V-Nutri 是一个分阶段管线：(1) 烹饪关键帧选择器（VideoMamba）从第一人称视频中识别食材添加事件；(2) 最终成品帧选择器定位成品帧；(3) Nutrition5K 预训练视觉骨干提取过程关键帧和成品特征；(4) 注意力加权融合模块聚合过程证据；(5) MLP 回归器预测热量、蛋白质、脂肪和碳水化合物四个指标。

### 关键设计

1. **烹饪关键帧选择器**:

    - 功能：从长且冗余的第一人称烹饪视频中定位营养信息密集的时刻
    - 核心思路：采用 VideoMamba 的选择性状态空间模型，通过滑动窗口将视频划分为短片段，检测食材添加等候选事件。VideoMamba 的线性复杂度适合处理长且信息稀疏的第一人称视频
    - 设计动机：密集处理全视频效率低且会引入噪声，需要先定位有信息量的稀疏子集

2. **Nutrition5K 预训练骨干 + 轻量融合**:

    - 功能：将过程关键帧和成品帧的视觉特征融合用于营养预测
    - 核心思路：冻结 Nutrition5K 预训练骨干（ResNet-101/ViT-B/ViT-L）编码每个过程关键帧为嵌入 $z_1,...,z_K$，成品帧为 $z_d$。学习注意力权重 $\alpha_1,...,\alpha_K$ 通过加权池化聚合过程嵌入为 $z_p$，再与成品嵌入融合
    - 设计动机：利用已在食物数据上预训练的骨干提取营养相关特征，避免从头训练；轻量融合避免过拟合

3. **HD-EPIC 基准标注扩展**:

    - 功能：建立首个基于视频的营养估计基准
    - 核心思路：在 HD-EPIC 数据集上补充标注烹饪过程关键帧和最终成品帧的时间戳，建立菜品级营养真值
    - 设计动机：现有数据集缺乏将烹饪视频与营养标签关联的标注

### 损失函数 / 训练策略

使用标准回归损失（如 MAE/MSE）预测四维营养向量 $\mathbf{y}_c = [y^{kcal}, y^{protein}, y^{fat}, y^{carb}]$。骨干冻结，仅训练融合模块和回归器。

## 实验关键数据

### 主实验

| 骨干 | 输入 | 热量 MAE↓ | 蛋白质 MAE↓ | 脂肪 MAE↓ | 碳水 MAE↓ |
|------|------|----------|-----------|----------|----------|
| ViT-L | 仅成品 | 185.3 | 12.1 | 9.8 | 18.5 |
| ViT-L | 成品+过程帧 | **172.8** | **11.2** | **9.1** | **17.3** |
| ViT-B | 仅成品 | 198.7 | 13.5 | 10.6 | 19.8 |
| ViT-B | 成品+过程帧 | 191.2 | 12.8 | 10.1 | 19.0 |
| ResNet-101 | 成品+过程帧 | 205.1 | 14.2 | 11.3 | 20.5 |

### 消融实验

| 配置 | 热量 MAE↓ | 说明 |
|------|----------|------|
| 完整模型 (ViT-L) | 172.8 | 成品+过程帧+事件检测 |
| 无事件检测(随机帧) | 182.1 | 随机选帧替代事件检测 |
| 仅成品帧 | 185.3 | 仅用最终成品 |
| 均匀采样过程帧 | 179.5 | 均匀采样替代事件检测 |

### 关键发现

- 过程关键帧的收益强烈依赖骨干表示能力：ViT-L 获益最大，ResNet-101 改善有限
- 事件检测质量是关键：随机帧的收益远小于检测到的食材添加帧
- 在受控条件下过程信息确实提供了互补营养证据

## 亮点与洞察

- "过程感知"的营养估计是一个合理且实用的研究方向：随着可穿戴相机的普及，利用烹饪视频进行营养监测是可行的
- 轻量融合策略（冻结骨干+注意力加权池化）避免了过拟合，适合数据量有限的场景

## 局限与展望

- HD-EPIC 数据集规模有限，泛化性验证不足
- 过程帧的收益在弱骨干上不明显，方法对骨干依赖性强
- 未考虑烹饪方式（煎、炸、蒸等）对营养变化的影响
- 可结合食材识别和份量估计进一步提升准确性

## 相关工作与启发

- **vs Nutrition5K**: Nutrition5K 仅用成品图像，V-Nutri 扩展到视频，利用过程信息补充成品不可见的营养线索
- **vs 长视频理解**: 本文不追求全序列理解，而是高效提取稀疏过程证据

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将烹饪视频过程信息引入营养估计
- 实验充分度: ⭐⭐⭐ 数据集规模有限，但消融较充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐ 方向有意义但改善幅度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion](omnifood8k_nutrition_estimation.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](../../AAAI2026/others/reward_redistribution_via_gaussian_process_likelihood_estimation.md)

</div>

<!-- RELATED:END -->
