---
title: >-
  [论文解读] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation
description: >-
  [ICCV 2025][人体理解][类别级位姿估计] 首次将因果推理引入类别级物体位姿估计（COPE），通过基于前门调整的因果推理模块消除数据偏差导致的虚假关联，并利用3D基础模型ULIP-2的残差知识蒸馏提供无偏的类别语义监督，在REAL275的严格指标5°2cm上达到61.7%，超越SOTA 4.7%。
tags:
  - ICCV 2025
  - 人体理解
  - 类别级位姿估计
  - 因果推理
  - 知识蒸馏
  - 前门调整
  - 数据偏差
---

# CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation

**会议**: ICCV 2025  
**arXiv**: [2502.01312](https://arxiv.org/abs/2502.01312)  
**代码**: https://github.com/chrislin0621/CleanPose  
**领域**: 3D视觉 / 位姿估计  
**关键词**: 类别级位姿估计, 因果推理, 知识蒸馏, 前门调整, 数据偏差

## 一句话总结

首次将因果推理引入类别级物体位姿估计（COPE），通过基于前门调整的因果推理模块消除数据偏差导致的虚假关联，并利用3D基础模型ULIP-2的残差知识蒸馏提供无偏的类别语义监督，在REAL275的严格指标5°2cm上达到61.7%，超越SOTA 4.7%。

## 研究背景与动机

类别级物体位姿估计（COPE）旨在预测预定义类别内任意物体的9DoF位姿（旋转、平移、尺寸），与实例级方法不同，不需要高质量CAD模型。近年来的方法主要聚焦于模型架构设计以捕获鲁棒的类别特征，但性能提升趋于瓶颈。

作者发现了一个被忽视的根本问题：**数据集中的固有偏差**。具体表现为：
- 重复的训练样本和有限的位姿多样性
- 相似的场景环境导致模型过拟合于特定外观和位姿模式
- 标注偏差和采集设备噪声

这些偏差使模型学到**虚假关联**（spurious correlations）而非真正的因果关系。例如，某类物体总是以相似角度出现在训练集中，模型可能将背景相似性而非物体结构作为位姿判断的依据。创建完全无偏的数据集不切实际（3D标注成本高昂），因此需要从模型端解决偏差问题。

作者从人类观察机制获得启发：人类能有效处理类内变化，是因为能利用**类比联想**推断结构特征和因果关系，对新物体在不同环境和视角下保持稳定感知。

## 方法详解

### 整体框架

CleanPose在AG-Pose基线之上增加两个核心模块：
1. **因果推理模块**（基于前门调整）：消除隐藏混淆变量的影响
2. **残差知识蒸馏模块**：从ULIP-2基础模型转移无偏的3D语义知识

输入RGB-D图像经分割后得到裁剪图像和点云，分别通过PointNet++和DINOv2提取特征，拼接后输入因果推理模块。

### 关键设计

#### 1. 结构因果模型
- **功能**：建立COPE任务中关键变量之间的因果关系图
- **核心思路**：定义四个变量：视觉输入 $\mathcal{X}$、输出位姿 $\mathcal{Y}$、中介变量 $\mathcal{M}$（结构信息/关键点）和隐藏混淆变量 $\mathcal{U}$（数据偏差）
    - **前门路径** $\mathcal{X} \rightarrow \mathcal{M} \rightarrow \mathcal{Y}$：人类先识别结构信息（关键点），再通过类比推断位姿
    - **混淆路径** $\mathcal{X} \leftarrow \mathcal{U} \rightarrow \mathcal{Y}$：数据偏差同时影响输入采样和位姿分布
- **设计动机**：传统方法建模 $P(\mathcal{Y}|\mathcal{X})$，混入了混淆变量的影响；因果推理通过do算子建模 $P(\mathcal{Y}|do(\mathcal{X}))$，切断混淆路径

#### 2. 基于前门调整的因果推理
- **功能**：通过对前门路径的两个阶段（$\mathcal{X} \rightarrow \mathcal{M}$ 和 $\mathcal{M} \rightarrow \mathcal{Y}$）施加干预，消除混淆变量的影响
- **核心思路**：前门调整公式：

$$P(\mathcal{Y}|do(\mathcal{X})) = \sum_{x'} P(x') \sum_m P(\mathcal{Y}|m, x') P(m|\mathcal{X})$$

通过查询机制近似期望值计算：

$$\mathbb{E}_{x'}[\boldsymbol{x}'] \approx \sum_i \frac{\exp(\boldsymbol{g}_1 \boldsymbol{x}_i'^T)}{\sum_j \exp(\boldsymbol{g}_1 \boldsymbol{x}_j'^T)} \boldsymbol{x}_i'$$

其中 $\boldsymbol{g}_1 = q_1(x)$ 为嵌入函数。最终通过多头自注意力（$\mathcal{F}_s$）和交叉注意力（$\mathcal{F}_c$）实现：

$$\mathcal{F}_s = SA(\mathcal{F}_{kpt}), \quad \mathcal{F}_c = CA(\mathcal{F}_{kpt}, \mathcal{F}_{samp})$$
$$\mathcal{F}_f = LN(\mathcal{F}_s + \mathcal{F}_c)$$

- **中介变量实现**：使用关键点特征 $\mathcal{F}_{kpt}$（均匀分布在物体表面的96个关键点）
- **跨样本特征采样**：借鉴MoCo的动态队列（$N_c \times N_q$）。用ULIP-2的3D编码器提取特征初始化，FIFO策略更新。每次从队列随机采样 $N_s=12$ 个特征

- **自适应权重融合**：

$$w_a = \sigma(\mathcal{F}_f W_f + \mathcal{F}_{kpt} W_k)$$
$$\mathcal{F}_f \leftarrow w_a \odot \mathcal{F}_f + (1-w_a) \odot \mathcal{F}_{kpt}$$

- **设计动机**：仅对输入做调整不足以完全阻断混淆效应（因为存在中介变量），需要在前门路径的两个阶段都施加干预

#### 3. 残差知识蒸馏
- **功能**：从3D基础模型ULIP-2中转移无偏的类别语义知识
- **核心思路**：ULIP-2在大规模多样数据上预训练，隐式学到了鲁棒的去偏特征表示。使用冻结的ULIP-2 3D编码器（PointBERT）提取教师特征 $\mathcal{F}_P^{ULIP}$，学生模型的点云特征经残差网络变换后与之对齐：

$$\hat{\mathcal{F}}_P^{avg} = \mathcal{F}_P^{avg} + \mu \times K_2(\delta(K_1(\mathcal{F}_P^{avg})))$$

$$\mathcal{L}_{KD} = \frac{1}{B} \sum_i^B \| \mathcal{F}_P^{ULIP} - \psi(\hat{\mathcal{F}}_P^{avg}) \|_2$$

- **设计动机**：$K_2$ 初始化为零，使初始阶段只包含原始特征，避免引入额外混淆变量。通过渐进更新平衡特征学习和知识转移

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{all} = \alpha_1 \mathcal{L}_{pose} + \alpha_2 \mathcal{L}_{KD}$

- $\mathcal{L}_{pose} = \|\mathcal{R}_{gt} - \mathcal{R}\|_2 + \|t_{gt} - t\|_2 + \|s_{gt} - s\|_2$
- $\alpha_1 = 1, \alpha_2 = 0.01, \mu = 0.1$
- Adam优化器，学习率2e-5到5e-4（triangular2 cyclical schedule），120K迭代，batch size 24

## 实验关键数据

### 主实验

REAL275数据集上与SOTA比较：

| 方法 | 是否需要先验 | IoU75* | 5°2cm | 5°5cm | 10°2cm | 10°5cm |
|------|------------|--------|-------|-------|--------|--------|
| AG-Pose (CVPR24) | ✗ | 61.3 | 57.0 | 64.6 | 75.1 | 84.7 |
| GCE-Pose (CVPR25) | ✓ | - | 57.0 | 65.1 | 75.6 | 86.3 |
| SecondPose (CVPR24) | ✗ | 49.7 | 56.2 | 63.6 | 74.7 | 86.0 |
| **CleanPose** | **✗** | **62.7** | **61.7** | **67.6** | **78.3** | **86.3** |

HouseCat6D数据集：CleanPose在所有指标上超越AG-Pose（IoU50: +2.9%, 5°2cm: +1.1%, 10°5cm: +2.1%）

### 消融实验

因果学习和知识蒸馏的效果（REAL275）：

| 因果推理 | 知识蒸馏 | 5°2cm | 5°5cm | 10°2cm | 10°5cm |
|---------|---------|-------|-------|--------|--------|
| ✗ | ✗ | 57.0 | 64.6 | 75.1 | 84.7 |
| ✓ | ✗ | 59.7 | 66.5 | 77.5 | 86.0 |
| ✗ | ✓ | 57.9 | 65.5 | 76.1 | 85.2 |
| **✓** | **✓** | **61.7** | **67.6** | **78.3** | **86.3** |

特征存储和更新策略对比：

| 存储方式 | 更新策略 | 5°2cm | 5°5cm | 说明 |
|---------|---------|-------|-------|------|
| 动态队列 | FIFO | **61.7** | **67.6** | 最佳：追踪特征动态变化 |
| 动态队列 | 无更新 | 57.1 | 66.1 | 特征过时 |
| 动态队列 | 相似度更新 | 61.0 | 67.1 | 近似但不如FIFO |
| 记忆库 | 无更新 | 57.6 | 66.4 | 无法捕捉训练动态 |

### 关键发现

1. 因果推理模块单独贡献+2.7%（5°2cm），知识蒸馏单独贡献+0.9%，两者结合+4.7%，表明互补性强
2. 动态队列+FIFO策略相比静态记忆库提升4.1%（5°2cm），证实了动态特征更新对因果推理的重要性
3. 残差蒸馏优于直接拼接（Concat）和对比学习（Contrastive），因为残差连接平衡了原始特征和新知识
4. 即使不使用先验模型（shape prior），CleanPose也超越了使用先验的GCE-Pose

## 亮点与洞察

- **问题诊断精准**：首次从数据偏差角度分析COPE性能瓶颈，而非继续在模型架构上内卷
- **因果建模合理**：将数据偏差建模为隐藏混淆变量，前门调整的选择有充分理论依据
- **MoCo式动态队列**：巧妙借用对比学习的技术解决因果推理中跨样本特征采样的技术难题
- **理论与实践结合**：从人类观察机制出发，经因果图建模，最终落地为具体的注意力机制实现

## 局限与展望

- 因果图中的中介变量 $\mathcal{M}$ 固定为关键点特征，未探索其他可能的结构表示
- 动态队列大小 $N_q=80$ 和采样数 $N_s=12$ 为经验值，缺乏自适应调整机制
- 仅在6个类别（REAL275）的有限场景上验证，类别数更多时的可扩展性待验证
- ULIP-2作为教师模型是固定的，未探索多个教师模型集成或自蒸馏

## 相关工作与启发

- **AG-Pose**（CVPR24）是直接基线，通过显式提取局部和全局关键点信息
- **MoCo**的动态队列思想被创新性地引入因果推理场景
- 因果推理在目标检测、图像描述中已有应用，但3D点云位姿估计是全新场景
- 该方法论可推广到其他受数据偏差影响的3D视觉任务（如3D重建、点云分类）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将前门调整因果推理应用于COPE，开辟了全新研究方向
- 实验充分度: ⭐⭐⭐⭐ 三个数据集全面验证，消融设计科学，但类别数有限
- 写作质量: ⭐⭐⭐⭐ 公式推导严谨，但LaTeX渲染格式对阅读有一定影响
- 价值: ⭐⭐⭐⭐⭐ 在REAL275上5°2cm指标实现61.7%的突破性成绩，超越SOTA 4.7%

<!-- RELATED:START -->

## 相关论文

- [GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](../../CVPR2025/human_understanding/gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](../../ECCV2024/human_understanding/u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)
- [GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](../../ECCV2024/human_understanding/gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](../../ECCV2024/human_understanding/lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)

<!-- RELATED:END -->
