---
title: >-
  [论文解读] PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation
description: >-
  [CVPR 2026][3D视觉][物理属性估计] 提出 PhysGS，将贝叶斯推断嵌入3D高斯溅射管线，利用视觉-语言模型先验和多视角置信度加权更新，实现逐点物理属性（摩擦力、硬度、密度、刚度）的概率估计与不确定性量化，在质量估计上比 NeRF2Physics 提升 22.8%（APE），岸氏硬度误差降低 61.2%。
tags:
  - CVPR 2026
  - 3D视觉
  - 物理属性估计
  - 贝叶斯推断
  - 3D高斯溅射
  - 不确定性量化
  - 视觉语言模型
---

# PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation

**会议**: CVPR 2026  
**arXiv**: [2511.18570](https://arxiv.org/abs/2511.18570)  
**代码**: [项目主页](https://samchopra2003.github.io/physgs)  
**领域**: 3D视觉  
**关键词**: 物理属性估计, 贝叶斯推断, 3D高斯溅射, 不确定性量化, 视觉语言模型

## 一句话总结
提出 PhysGS，将贝叶斯推断嵌入3D高斯溅射管线，利用视觉-语言模型先验和多视角置信度加权更新，实现逐点物理属性（摩擦力、硬度、密度、刚度）的概率估计与不确定性量化，在质量估计上比 NeRF2Physics 提升 22.8%（APE），岸氏硬度误差降低 61.2%。

## 研究背景与动机
**领域现状**：理解环境的物理属性（摩擦力、硬度、弹性、密度等）对机器人安全交互至关重要。现有3D重建方法（NeRF、3DGS）主要关注几何和外观，无法推断底层物理属性。

**现有痛点**：
   - NeRF2Physics 等方法用语言嵌入做零样本回归，但没有建模不确定性，在模糊区域（泥土 vs 沥青）容易产生脆弱预测
   - 已有方法通常只估计一两种物理属性（如摩擦力或弹性），不够通用
   - 室外场景几乎未被探索
   - 传感器噪声和模型知识不足导致的两类不确定性（偶然+认知）未被显式建模

**核心矛盾**：如何从视觉传感器出发，在统一框架下估计多种物理属性，同时量化估计的可靠程度

**切入角度**：将每个高斯原语视为概率实体，其物理属性信念通过贝叶斯后验更新不断精化

**核心 idea**：用 Dirichlet-Categorical 模型融合离散材质分类 + Normal-Inverse-Gamma 先验建模连续属性的偶然和认知不确定性

## 方法详解

### 整体框架
输入多视角图像 → SAM 部件级分割 → VLM（GPT-5）预测材质标签、密度等物理属性及置信度 → 贝叶斯推断融合多视角证据 → 3DGS 重建 → 逐点物理属性场

### 关键设计

1. **Dirichlet-Categorical 材质分类**

    - 功能：将 VLM 预测的离散材质标签通过贝叶斯更新融合为后验概率
    - 核心思路：Dirichlet 分布作为 Categorical 似然的共轭先验，后验参数递归更新 $\tilde{\alpha}_i \leftarrow \alpha_i(0) + \sum_{m: c_m=i} \lambda p_m$，其中 $p_m$ 是 VLM 对第 $m$ 次预测的置信度。后验预测概率为 $f(z=i | Z, \boldsymbol{\alpha}) = \tilde{\alpha}_i / \sum_j \tilde{\alpha}_j$
    - 设计动机：VLM 的单视角预测可能不一致，需要跨视角融合。Dirichlet-Categorical 的共轭性使得融合可以闭式完成，适合在线更新

2. **连续属性的贝叶斯估计**

    - 功能：对每种材质的连续物理属性（摩擦系数、密度等）估计均值和方差
    - 核心思路：维护置信度加权的三个累积器 $W_i = \sum p_m$、$S_i = \sum p_m \psi_m$、$Q_i = \sum p_m \psi_m^2$，计算后验均值 $\mu_i = S_i / W_i$ 和方差 $\sigma_i^2 = Q_i/W_i - \mu_i^2$。最终属性分布为高斯混合 $f(\psi | Z, \boldsymbol{\alpha}) = \sum_i \frac{\tilde{\alpha}_i}{\sum_j \tilde{\alpha}_j} \mathcal{N}(\mu_i, \sigma_i^2)$
    - 设计动机：用增量在线更新避免存储历史观测，特别适合流式重建场景

3. **Normal-Inverse-Gamma 不确定性建模**

    - 功能：将总预测不确定性分解为偶然不确定性（传感器/感知噪声）和认知不确定性（模型知识不足）
    - 核心思路：对均值 $\mu_i$ 和方差 $\sigma_i^2$ 的联合先验使用 NIG 分布 $p(\mu_i, \sigma_i^2 | \tau_i, \kappa_i, \alpha_i, \beta_i)$。不确定性分解为 $\text{Var}[\psi_i] = \underbrace{\mathbb{E}[\sigma_i^2]}_{\text{偶然}} + \underbrace{\text{Var}[\mu_i]}_{\text{认知}}$，其中 $\mathbb{E}[\sigma_i^2] = \tilde{\beta}_i / (\tilde{\alpha}_i - 1)$，$\text{Var}[\mu_i] = \mathbb{E}[\sigma_i^2] / \tilde{\kappa}_i$
    - 设计动机：机器人决策需要知道"我有多不确定"——高偶然不确定性意味着感知困难，高认知不确定性意味着需要更多观测

4. **3DGS 语义属性映射**

    - 功能：将贝叶斯推断的材质-属性对应关系映射回3D高斯场
    - 核心思路：用贝叶斯推断确定的材质对应颜色重着色场景图像，作为语义输入建3DGS。每个体素关联一个预测属性值，支持逐点查询（如摩擦力）和全局聚合（如总质量）

### 损失函数 / 训练策略
- 使用 Nerfstudio 的 splatfacto-big 变体，20k 迭代，RTX A5000
- VLM 使用 GPT-5，结构化视觉-文本 prompt

## 实验关键数据

### 主实验——质量估计（ABO-500 测试集，100个物体）

| 方法 | ADE↓(kg) | ALDE↓ | APE↓ | MnRE↑ |
|------|---------|--------|------|-------|
| Image2mass | 12.496 | 1.792 | 0.976 | 0.341 |
| 2D CNN | 15.431 | 1.609 | 14.459 | 0.362 |
| LLaVA | 17.328 | 1.893 | 1.837 | 0.306 |
| NeRF2Physics | 8.730 | **0.771** | 1.061 | **0.552** |
| **PhysGS** | **8.254** | 0.999 | **0.819** | 0.474 |

### 消融实验（ABO-500 验证集）

| 配置 | ADE↓ | APE↓ | 说明 |
|------|------|------|------|
| NeRF2Physics | 9.786 | 0.931 | 基线 |
| PhysGS (w/o 贝叶斯) | 9.728 | 0.717 | 去掉贝叶斯更新 |
| **PhysGS (w/ 贝叶斯)** | **9.187** | **0.715** | 完整模型 |

### 关键发现
- PhysGS 在 APE 上相比 NeRF2Physics 降低 22.8%（1.061→0.819），ADE 降低 5.5%
- 贝叶斯推断相比不做贝叶斯更新的变体，ADE 降低 5.6%
- 岸氏硬度误差降低 61.2%，动摩擦系数误差降低 18.1%
- 偶然不确定性高的区域对应传感器噪声大或材质识别困难区域，认知不确定性高的区域对应证据不足区域——与直觉一致

## 亮点与洞察
- **贝叶斯推断 + 3DGS 的结合有理论美感**：将每个高斯原语视为概率实体并通过后验更新，是一个自然且优雅的建模方式。置信度加权的在线更新不需要存储历史数据，适合增量重建
- **不确定性分解的实用性**：偶然 vs 认知的区分对下游机器人决策非常重要。比如在高认知不确定性区域，机器人应该更谨慎或寻求更多观测
- **VLM 作为物理先验源**：利用 GPT-5 作为材质识别和属性估计的零样本先验源是一个可行方案，虽然精度有限但通过贝叶斯多视角融合可以显著改善

## 局限与展望
- VLM 的物理属性估计是零样本的，缺乏领域校准，可能在非常见材质上偏差很大
- 质量估计依赖于密度估计 × 体积积分，体积估计的误差会级联放大
- NeRF2Physics 在 ALDE 和 MnRE 上仍然更好，说明在某些度量下贝叶斯方法反而引入了偏差
- 需要 SAM 分割和 VLM 推理，计算成本较高（相比纯几何方法）
- 室外场景的验证主要是定性的，缺少定量基准

## 相关工作与启发
- **vs NeRF2Physics**：NeRF2Physics 用 CLIP 嵌入 + 核回归做属性估计，无不确定性建模。PhysGS 通过贝叶斯更新在多数指标上更优，但 ALDE 和 MnRE 落后
- **vs GaussianProperty**：同样基于3DGS+VLM，但没有贝叶斯更新和不确定性分解
- **vs EVORA/MEM 等不确定性方法**：它们在2D感知中建模不确定性，PhysGS 是首个在3DGS中同时做物理属性+不确定性的工作

## 评分
- 新颖性: ⭐⭐⭐⭐ 贝叶斯推断+3DGS+物理属性的三者结合有创新性，NIG分解也有理论深度
- 实验充分度: ⭐⭐⭐ 数据集和基线较有限，室外场景缺少定量评估
- 写作质量: ⭐⭐⭐⭐ 数学推导完整清晰，问题动机表述好
- 价值: ⭐⭐⭐⭐ 不确定性感知的物理属性估计对机器人领域有重要意义

<!-- RELATED:START -->

## 相关论文

- [Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_model_for_feed-forward_4d_synthesis.md)
- [GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)

<!-- RELATED:END -->
