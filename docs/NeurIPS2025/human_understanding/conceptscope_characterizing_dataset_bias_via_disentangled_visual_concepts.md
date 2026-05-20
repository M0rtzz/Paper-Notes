---
title: >-
  [论文解读] ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts
description: >-
  [NeurIPS 2025][人体理解][数据集偏差] 提出 ConceptScope 框架，利用在视觉基础模型表征上训练的稀疏自编码器（SAE）自动发现和量化数据集中的视觉概念偏差，无需人工标注即可将概念分类为 target / context / bias 三类。
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "数据集偏差"
  - "稀疏自编码器"
  - "视觉概念"
  - "偏差检测"
  - "可解释性"
---

# ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts

**会议**: NeurIPS 2025  
**arXiv**: [2510.26186](https://arxiv.org/abs/2510.26186)  
**代码**: [GitHub](https://github.com/jjho-choi/ConceptScope)  
**领域**: 人体理解  
**关键词**: 数据集偏差, 稀疏自编码器, 视觉概念, 偏差检测, 可解释性

## 一句话总结

提出 ConceptScope 框架，利用在视觉基础模型表征上训练的稀疏自编码器（SAE）自动发现和量化数据集中的视觉概念偏差，无需人工标注即可将概念分类为 target / context / bias 三类。

## 研究背景与动机

机器学习数据集中的偏差（如特定类别与特定背景的高度相关性）普遍存在，会导致模型泛化能力下降。例如 ImageNet 中约 75%% 的"棱皮海龟"图像拍摄于沙滩，仅 15%% 在水下。现有方法要么依赖昂贵的人工标注，要么依赖 VLM 生成的描述文本，但自然语言描述存在粒度不一致、同义替换等问题，难以结构化提取视觉概念。本文旨在构建一个**全自动、可扩展**的数据集偏差分析框架。

## 方法详解

### 整体框架

ConceptScope 分为两个阶段：
1. **概念字典构建**：在预训练视觉编码器（CLIP-ViT-L/14）的中间层 token embedding 上训练 SAE，将稠密表征解耦为稀疏的可解释概念
2. **概念分类**：基于语义相关性和统计频率，将每个概念归类为 target（目标）、context（上下文）或 bias（偏差）

### 关键设计

**稀疏自编码器（SAE）训练**：给定图像 $x$，提取 patch-level token embedding $\mathbf{z} = \{z_1, \ldots, z_l\}$，SAE 编码-解码过程为：

$$f(z) = \phi(W_{\text{enc}}^T z), \quad \text{SAE}(z) = W_{\text{dec}}^T f(z)$$

其中 $\phi$ 为 ReLU 激活函数，$W_{\text{enc}} \in \mathbb{R}^{d \times d'}$，$d'$ 远大于 $d$（扩展因子 16 或 32）。

**概念分类——Alignment Score**：定义 necessity $N(c,y)$ 和 sufficiency $S(c,y)$ 两个指标，分别衡量移除概念 $c$ 后预测置信度的下降和仅保留 $c$ 时的预测能力：

$$N(c,y) = \frac{1}{|X_y|}\sum_{x \in X_y} \frac{P(y|x)}{P(y|x \odot (1-m_c(x)))}$$

$$S(c,y) = \frac{1}{|X_y|}\sum_{x \in X_y} \frac{P(y|x \odot m_c(x))}{P(y|x)}$$

两者取平均得到 alignment score $A(c,y) = \frac{N(c,y) + S(c,y)}{2}$。当 $A(c,y) \geq \mu_y^{\text{align}} + \alpha \times \sigma_y^{\text{align}}$ 时判定为 target 概念，否则为 context 概念。

**偏差概念识别**：在排除 target 概念后，计算 context 概念的 concept strength $\tilde{f}_{c,y} = \text{avg}_{\mathbf{z} \in Z_y}(f(\mathbf{z})_c)$。当 $\tilde{f}_{c,y} \geq \mu^{c.s.} + \sigma^{c.s.}$ 时判定为 bias 概念。

### 损失函数

SAE 训练损失为重建损失加 L1 稀疏惩罚：

$$\mathcal{L} = \|z - \text{SAE}(z)\|_2^2 + \lambda \|z\|_1$$

## 实验关键数据

### 主实验

**概念预测性能**（6 个标注数据集上的二分类精度，F1 / AUPRC）：

| 方法 | Caltech101 | DTD | Waterbird | CelebA | RAF-DB | Stanford40 | 平均 |
|------|-----------|-----|-----------|--------|--------|-----------|------|
| BLIP-2 | 0.64 | 0.38 | 0.37 | 0.27 | 0.24 | 0.66 | 0.43 |
| LLaVA-NeXT | 0.61 | 0.40 | 0.57 | 0.62 | 0.45 | 0.80 | 0.58 |
| **ConceptScope** | **0.83** | **0.57** | **0.78** | **0.81** | **0.55** | **0.78** | **0.72** |

**偏差发现任务**（Precision@10）：

| 方法 | Waterbirds | CelebA | NICO++(75) | NICO++(90) | NICO++(95) |
|------|-----------|--------|-----------|-----------|-----------|
| DOMINO | 90.0%% | 87.0%% | 24.0%% | 24.0%% | 24.0%% |
| FACTS | 100.0%% | 100.0%% | 55.0%% | 60.8%% | 61.0%% |
| **ConceptScope** | **100.0%%** | **100.0%%** | **72.9%%** | **73.1%%** | **74.0%%** |

### 消融实验

- SAE 空间归因的分割精度：ADE20K 上 AUPRC 达 0.399，显著优于 BLIP-2（0.098）和 LLaVA-NeXT（0.302）
- SAE 激活值与 CLIP 相似度之间 Pearson 相关系数 $r = 0.71$，Spearman $\rho = 0.65$
- 四个随机种子训练的 SAE 标准差低于 0.01，表明框架稳健

### 关键发现

- 在 ImageNet-1K 中发现了此前未标注的偏差：如"项链"频繁出现在"人体模型"类别中，"新郎"类别与东亚文化场景高度相关
- 每个类别平均检测到 2.45 个偏差概念
- 模型鲁棒性诊断实验表明：high-target + high-bias 组准确率最高，low-target + low-bias 组最低，34 个模型中该趋势一致

## 亮点与洞察

1. **全自动无监督**：无需人工标注即可发现数据集偏差，一旦 SAE 训练完成可迁移到其他数据集
2. **概念三分类设计**（target/context/bias）既有理论基础又有实用价值
3. 在 NICO++ 上偏差发现的 Precision@10 比此前 SOTA（ViG-Bias）提升约 10 个百分点
4. 可扩展到多标签场景（MS-COCO）

## 局限性

- 概念受限于 CLIP 表征的知识范围，领域特异数据集（医学影像等）需重新训练 SAE
- 分割掩码为 patch 级（16x16），定位精度有限
- 在领域特异型属性（如情绪、纹理）上表现弱于通用属性

## 相关工作与启发

- 与 SpLiCE 等方法不同，ConceptScope 不需要预定义偏差类别即可自动判别
- SAE 在 LLM 可解释性中的成功经验被迁移到视觉领域
- 启发思考：能否将 ConceptScope 用于自动数据集清洗或主动学习样本选择

## 评分

- ⭐ 新颖性: 4/5 — SAE 用于视觉数据集偏差分析是首次系统性探索
- ⭐ 实验充分度: 5/5 — 6 个属性数据集 + 3 个偏差基准 + 多个真实数据集 + 34 个模型鲁棒性分析
- ⭐ 写作质量: 4/5 — 结构清晰，概念定义严谨
- ⭐ 价值: 4/5 — 为数据集审计和模型诊断提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](../../CVPR2025/human_understanding/d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2025\] Design2GarmentCode: Turning Design Concepts to Tangible Garments Through Program Synthesis](../../CVPR2025/human_understanding/design2garmentcode_turning_design_concepts_to_tangible_garments_through_program_.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](../../ICCV2025/human_understanding/humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[ICCV 2025\] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning](../../ICCV2025/human_understanding/ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)

</div>

<!-- RELATED:END -->
