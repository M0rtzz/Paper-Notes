---
title: >-
  [论文解读] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
description: >-
  [CVPR 2026][自动驾驶][位姿估计] 提出基于 Shapley 值+Pearson 相关系数的模态贡献评估算法和 Fisher 信息矩阵引导的自适应权重约束（AWC）正则化方法，解决 RGB/LiDAR/mmWave/WiFi 四模态端到端融合中的模态不平衡问题，在 MM-Fi 数据集上 MPJPE 降低 2.71mm 且不引入额外可学参数。
tags:
  - CVPR 2026
  - 自动驾驶
  - 位姿估计
  - modality imbalance
  - Shapley value
  - Fisher Information Matrix
  - 多模态
---

# Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation

**会议**: CVPR 2026  
**arXiv**: [2501.05264](https://arxiv.org/abs/2501.05264)  
**代码**: [GitHub](https://github.com/MICLAB-BUPT/AWC)  
**领域**: 人体理解 / 多模态学习  
**关键词**: 3D pose estimation, modality imbalance, Shapley value, Fisher Information Matrix, multi-modal fusion

## 一句话总结

提出基于 Shapley 值+Pearson 相关系数的模态贡献评估算法和 Fisher 信息矩阵引导的自适应权重约束（AWC）正则化方法，解决 RGB/LiDAR/mmWave/WiFi 四模态端到端融合中的模态不平衡问题，在 MM-Fi 数据集上 MPJPE 降低 2.71mm 且不引入额外可学参数。

## 研究背景与动机

**领域现状**：3D 人体位姿估计（3D HPE）以 RGB 为主，但 RGB 受遮挡和隐私限制。非侵入式传感器（LiDAR/mmWave/WiFi）可提供补充信息，但多模态端到端训练面临模态不平衡问题。

**现有痛点**：强模态（RGB/LiDAR）在训练早期主导梯度更新，抑制弱模态（mmWave/WiFi）的优化，导致弱模态预测退化为近常值输出（标准差趋近于零）。更糟糕的是，简单增加模态反而可能降低性能——四模态融合（53.87mm MPJPE）比 RGB+LiDAR（52.93mm）还差，直接证实了模态竞争的存在。现有平衡方法（G-Blending/OGM-GE/AGM）有两大局限：(1) 专为分类设计，依赖 cross-entropy，不适用于回归任务；(2) 常引入辅助单模态头增加模型复杂度。

**核心矛盾**：如何在不引入额外参数的前提下，实现多模态回归任务中的均衡优化？需要解决两个子问题：在回归任务中准确评估各模态贡献（分类的 cross-entropy 方案不适用），以及自适应地调节各模态学习速率以实现平衡。

**切入角度**：弱模态在回归任务中的特殊表现——预测近乎常值（标准差极低），用 MSE/MAE 评估会错误地认为其"可靠"。Pearson 相关系数衡量预测与 GT 的线性相关性而非距离，不受预测幅度影响，是更合适的贡献度指标。

**核心 idea**：用 Pearson 相关系数替代 MSE 作为回归任务的 Shapley 利润函数来检测模态不平衡，再用 FIM 加权的参数偏移正则化来差异化约束各模态的学习速率。

## 方法详解

### 整体框架

4 个模态（RGB/LiDAR/mmWave/WiFi）分别通过专用编码器（VideoPose3D / Point Transformer / MetaFi++）提取特征，融合模块（concatenation/MLP/attention）合并后送入位姿回归头输出 3D 关节坐标。两个核心组件：Shapley 模块评估模态贡献 → AWC 正则化在 learning window 内约束参数更新。

### 关键设计

1. **Shapley 值 + Pearson 相关的模态贡献评估**:

    - 功能：在回归任务中准确量化每个模态对融合模型的贡献度
    - 核心思路：Shapley 值通过枚举所有模态子集组合计算各模态的边际贡献 $\phi^m(\mathcal{M}) = \sum_{S \subseteq \mathcal{M} \setminus \{m\}} \frac{|S|!(|\mathcal{M}|-|S|-1)!}{|\mathcal{M}|!} V(S,m)$。关键创新在于利润函数 $s(\cdot,\cdot)$——传统方法用 cross-entropy（分类），本文用 Pearson 相关系数 $s(y, \hat{y}) = \sum_{i=1}^{j \times 3} \rho(y_i, \hat{y}_i)$，沿 batch 维度计算每个关节坐标值的线性相关性
    - 设计动机：弱模态（mmWave/WiFi）在回归中产生近常值预测（标准差趋近于零），如果用 MSE 评估，距离小反而被误判为"可靠"。Pearson 相关不受预测幅度影响，能准确识别出无信息量的常值预测。实验证实 RGB 和 LiDAR 一致获得高贡献分，mmWave/WiFi 分数低且随训练下降

2. **AWC（自适应权重约束）正则化**:

    - 功能：根据模态贡献差异化地约束各模态编码器的参数更新速率
    - 核心思路：先用 K-Means 将 4 个模态按 Shapley 分聚类为优势组 $\mathcal{M}_\mathcal{S}$ 和劣势组 $\mathcal{M}_\mathcal{I}$。对每个模态编码器施加参数偏移正则：$\mathcal{L}_{\text{AWC}} = \sum_m [\alpha_\mathcal{S} \cdot \mathbf{1}_{\{m \in \mathcal{M}_\mathcal{S}\}} + \alpha_\mathcal{I} \cdot \mathbf{1}_{\{m \in \mathcal{M}_\mathcal{I}\}}] \cdot \sum_i \frac{[\mathcal{I}_\mathcal{D}]_{ii} (\theta_{t,i}^m - \theta_{0,i}^{m,*})^2}{2}$。FIM 对角近似 $[\mathcal{I}]_{ii}$ 衡量参数重要性——强模态早期梯度大→FIM 高→正则化更强（抑制过快学习）；弱模态 FIM 低→正则化弱（允许继续学）。$\alpha_\mathcal{S} > \alpha_\mathcal{I}$ 确保对强模态约束更大
    - 设计动机：既约束方向又约束幅度。FIM 提供了数据驱动的参数灵敏度估计，自然地将"重要参数"（对损失影响大的）与"不重要参数"区分开。无需额外可学参数

3. **Learning Window 机制**:

    - 功能：AWC 仅在前 K 个 epoch 施加，之后关闭
    - 核心思路：基于先验研究发现，与模态相关的关键信息在训练早期获取。实验验证 K=20 最优（总共 50 epochs），过短或过长都会降低性能
    - 设计动机：后期正则化反而干扰收敛。前期约束给弱模态留出学习空间后，后期自由优化才能充分利用所有模态

### 损失函数 / 训练策略

$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MPJPE}} + \mathcal{L}_{\text{AWC}}$（前 K 个 epoch）/ $\mathcal{L}_{\text{MPJPE}}$（后续 epoch）。Adam 优化器，lr=1e-3，每 30 epoch×0.1，batch=192，共 50 epochs，2×RTX 3090。

## 实验关键数据

### 主实验

| 方法 | Protocol 1 MPJPE↓ | PA-MPJPE↓ | Protocol 3 MPJPE↓ | PA-MPJPE↓ |
|------|-------------------|-----------|-------------------|-----------|
| Concatenation (baseline) | 53.87 | 35.09 | 48.17 | 32.18 |
| + G-Blending | 58.40 | 37.20 | 53.13 | 33.28 |
| + OGM-GE | 55.51 | 35.92 | 51.68 | 32.84 |
| + AGM | 55.80 | 38.10 | 53.88 | 36.30 |
| + Modality-level | 53.24 | 34.81 | 53.98 | 31.85 |
| **+ Ours (AWC)** | **51.16** | **34.46** | **47.55** | **31.79** |

### 消融实验

| α_S | α_I | MPJPE | PA-MPJPE | 说明 |
|-----|-----|-------|----------|------|
| 0 | 0 | 53.87 | 35.09 | 无正则化（baseline） |
| 0 | 10k | 52.92 | 34.94 | 仅约束弱模态 |
| 10k | 0 | 52.09 | 34.81 | 仅约束强模态 |
| **20k** | **10k** | **51.16** | **34.46** | 最优配置 |
| 20k | 20k | 51.69 | 34.84 | 等强度约束效果差 |

### 关键发现

- **仅约束强模态比仅约束弱模态更有效**（52.09 vs 52.92），证实抑制强模态过快学习是平衡的主要机制
- **同时约束两组效果最好**也必须差异化（α_S > α_I），等强度约束（20k/20k）不如差异化（20k/10k）
- **Learning window K=20 最优**：太短（K=10）不足以建立平衡，太长（K=25）干扰收敛
- **模态竞争的直接证据**：R+L+M+W（53.87mm）比 R+L（52.93mm）差，增加弱模态反而有害
- **计算开销极低**：Shapley 计算在 Concat/MLP 融合下仅占训练时间的 0.41%~0.93%

## 亮点与洞察

- **Pearson 相关替代 MSE 作为回归任务的 Shapley 利润函数**：这个观察非常精准——弱模态产生常值预测时 MSE 反而小，会被误判为有用。Pearson 相关完全免疫于这种陷阱，可直接迁移到任何多模态回归任务
- **FIM 自然实现差异化正则**：不需要手动设计哪些参数该约束、哪些不该——FIM 值高的参数（强模态中被频繁更新的参数）正是最需要减速的，实现了自适应。这种思路可迁移到 continual learning 或 domain adaptation
- **Learning Window 概念**：承认"平衡只在早期重要"——后期所有模态都应该自由优化。这个 insight 对其他平衡方法也有参考价值

## 局限与展望

- **仅在 MM-Fi 一个数据集上评估**：4 模态 HPE 本身是个很 niche 的设置，泛化性存疑
- **Shapley 值计算复杂度指数增长**：4 模态时只需 $2^4=16$ 次前向，但模态数增加到 6+ 时变得不可行，需要采样近似
- **简单的 K-Means 二分法**：将模态分为优势/劣势两组过于简化，模态数多时应考虑更精细的分组或连续权重
- **α_S 和 α_I 需手动调参**：虽然方法本身是自适应的，但两个正则化系数仍需搜索

## 相关工作与启发

- **vs G-Blending/OGM-GE/AGM**: 这些方法在本文实验中反而比 baseline 更差（MPJPE 上升 2~5mm），原因是它们依赖 cross-entropy 或仅调节梯度方向/幅度之一，不适用于回归任务
- **vs PMR**: PMR 基于原型的类级别表示做平衡，限于分类任务
- **vs MMPareto**: 通过 Pareto 前沿优化多模态梯度，但依赖单模态辅助头增加参数

## 评分

- 新颖性: ⭐⭐⭐⭐ Pearson-Shapley + FIM-AWC 的组合在多模态回归任务中是首次
- 实验充分度: ⭐⭐⭐ 仅一个数据集，但消融和分析详尽
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，动机推导清晰
- 价值: ⭐⭐⭐⭐ 对多模态回归任务的平衡优化提供了通用框架

<!-- RELATED:START -->

## 相关论文

- [EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)
- [PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)

<!-- RELATED:END -->
