---
title: >-
  [论文解读] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
description: >-
  [CVPR 2026][自动驾驶][多模态融合] 针对多模态3D人体姿态估计中的模态不平衡问题，提出基于Shapley值的模态贡献评估算法和基于Fisher信息矩阵的自适应权重约束（AWC）正则化方法，在不引入额外参数的情况下实现模态间的均衡优化，在MM-Fi数据集上全面超越现有平衡方法。
tags:
  - CVPR 2026
  - 自动驾驶
  - 多模态融合
  - 模态不平衡
  - Shapley值
  - Fisher信息矩阵
  - 3D HPE
---

# Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation

**会议**: CVPR 2026  
**arXiv**: [2501.05264](https://arxiv.org/abs/2501.05264)  
**代码**: [GitHub](https://github.com/MICLAB-BUPT/AWC)  
**领域**: 3D人体姿态估计 / 多模态学习  
**关键词**: 多模态融合, 模态不平衡, Shapley值, Fisher信息矩阵, 3D HPE

## 一句话总结

针对多模态3D人体姿态估计中的模态不平衡问题，提出基于Shapley值的模态贡献评估算法和基于Fisher信息矩阵的自适应权重约束（AWC）正则化方法，在不引入额外参数的情况下实现模态间的均衡优化，在MM-Fi数据集上全面超越现有平衡方法。

## 研究背景与动机

**领域现状**：3D人体姿态估计（3D HPE）是计算机视觉的重要课题，传统方法主要依赖RGB图像，但在遮挡和隐私场景下受限。近年来LiDAR、毫米波雷达（mmWave）、WiFi等非侵入式传感器开始受到关注，多模态融合成为提升3D HPE鲁棒性的有效途径。

**现有痛点**：当多种模态联合训练时，会出现"模态不平衡"现象——信息丰富的优势模态（如RGB、LiDAR）主导优化方向，抑制了劣势模态（如mmWave、WiFi）的学习。实验显示，融合全部四种模态（MPJPE 53.87mm）反而不如只融合RGB+LiDAR（52.93mm），证明了模态竞争的存在。

**核心矛盾**：现有的模态平衡方法（如OGM-GE、AGM）主要针对分类任务设计，依赖交叉熵损失或辅助单模态分支，不适用于回归任务。它们要么忽视了不同模态信息容量的内在差异，要么引入了额外的可学习参数增加模型复杂度。

**本文目标**：如何在不增加模型复杂度的前提下，量化每个模态的贡献程度，并动态调节各模态的学习速度，使优势模态不会过度压制劣势模态的优化。

**切入角度**：将博弈论中的Shapley值引入回归任务的模态贡献评估，并利用Fisher信息矩阵提供的参数重要性信息来自适应约束各模态编码器的参数更新。

**核心 idea**：用Shapley值检测模态不平衡，用FIM加权的权重约束损失在训练早期减缓优势模态的学习速度，从而在不引入额外参数的条件下实现平衡的多模态学习。

## 方法详解

### 整体框架

系统由四个模态专用编码器（RGB用VideoPose3D、LiDAR和mmWave用Point Transformer、WiFi用MetaFi++）提取特征，经过多模态融合模块（支持拼接、MLP、注意力三种策略）后由回归头预测3D关节坐标。训练过程中，Shapley模块评估各模态贡献，AWC损失在学习窗口内约束参数更新。

### 关键设计

1. **Shapley值贡献评估算法**:

    - 功能：量化每个模态在多模态协作中的边际贡献
    - 核心思路：对每个模态 $m$，遍历所有不含 $m$ 的子集 $S$，计算加入 $m$ 后的收益变化。关键创新在于利用Pearson相关系数 $\rho(y_i, \hat{y}_i)$ 替代MSE作为利润函数 $s(\cdot, \cdot)$，因为弱模态在回归任务中的预测趋近于常数（标准差接近零），用MSE会错误地高估其贡献
    - 设计动机：分类任务中弱模态输出接近均匀分布对Softmax影响小，但回归任务中弱模态的恒定输出会被MSE误判为"稳定可靠"。Pearson相关系数只关注预测与真值的相关性趋势，不受输出幅度影响

2. **自适应权重约束（AWC）正则化**:

    - 功能：基于FIM对各模态编码器施加差异化的参数约束，减缓优势模态的学习速度
    - 核心思路：将模态通过K-Means聚类分为优势组 $\mathcal{M}_\mathcal{S}$ 和劣势组 $\mathcal{M}_\mathcal{I}$，分别施加不同强度的正则化系数 $\alpha_\mathcal{S}$ 和 $\alpha_\mathcal{I}$。AWC损失为 $\mathcal{L}_{AWC} = \sum_m [\alpha_\mathcal{S} \cdot \mathbf{1}_{m \in \mathcal{M}_\mathcal{S}} + \alpha_\mathcal{I} \cdot \mathbf{1}_{m \in \mathcal{M}_\mathcal{I}}] \cdot \sum_i \frac{[\mathcal{I}]_{ii}(\theta_{t,i}^m - \theta_{0,i}^{m,*})^2}{2}$，其中FIM对角线元素 $[\mathcal{I}]_{ii}$ 衡量参数的经验重要性
    - 设计动机：优势模态在训练初期产生更大梯度导致更高FIM值，通过FIM加权的惩罚项可以自动对强模态施加更强约束、对弱模态施加较弱约束，无需人工指定

3. **学习窗口机制**:

    - 功能：仅在训练前 $K$ 个epoch内施加AWC约束
    - 核心思路：基于"关键学习期"理论，大部分任务相关信息在训练早期被习得。在前 $K$ 个epoch约束优势模态的快速学习，给劣势模态留出学习有用表征的机会
    - 设计动机：过长的正则化会限制模型的最终表达能力，适度的窗口期（实验中 $K=20$ 最优）兼顾平衡与性能

### 损失函数 / 训练策略

总损失在学习窗口内为 $\mathcal{L}_{total} = \mathcal{L}_{MPJPE} + \mathcal{L}_{AWC}$，窗口外仅用 $\mathcal{L}_{MPJPE}$。训练使用Adam优化器，初始学习率1e-3，每30个epoch衰减10倍，总共50个epoch，batch size 192。每个epoch起始时重新计算FIM。

## 实验关键数据

### 主实验

| 方法 | Protocol 1 MPJPE↓ | Protocol 1 PA-MPJPE↓ | Protocol 3 MPJPE↓ | Protocol 3 PA-MPJPE↓ |
|------|-------|----------|-------|----------|
| MM-Fi | 72.90 | 47.70 | 89.80 | 63.20 |
| Concatenation | 53.87 | 35.09 | 48.17 | 32.18 |
| +G-Blending | 58.40 | 37.20 | 53.13 | 33.28 |
| +OGM-GE | 55.51 | 35.92 | 51.68 | 32.84 |
| +AGM | 55.80 | 38.10 | 53.88 | 36.30 |
| **+Ours** | **51.16** | **34.46** | **47.55** | **31.79** |

在三种协议、三种融合策略下均取得最优，MPJPE平均降低约2-3mm。

### 消融实验

| 配置 ($\alpha_\mathcal{S}$, $\alpha_\mathcal{I}$) | MPJPE | PA-MPJPE | 说明 |
|------|---------|------|------|
| Baseline (0, 0) | 53.87 | 35.09 | 无正则化 |
| (0, 10k) | 52.92 | 34.94 | 仅约束弱模态 |
| (10k, 0) | 52.09 | 34.81 | 仅约束强模态 |
| **(20k, 10k)** | **51.16** | **34.46** | **最优配置** |
| (30k, 20k) | 51.34 | 34.56 | 过强正则化 |

单模态性能验证模态差异：RGB (MPJPE 63.61) > LiDAR (66.95) >> mmWave (102.89) >> WiFi (166.92)。

### 关键发现

- 四模态融合（53.87mm）反而不如RGB+LiDAR（52.93mm），直接证明模态竞争的存在
- Shapley贡献分数显示RGB和LiDAR的贡献远高于mmWave和WiFi，且训练进程中弱模态贡献持续下降
- 同时约束两组模态比只约束一组效果更好（弱模态也需约束以抑制噪声过拟合）
- 学习窗口 $K=20$ 最优，过短或过长都会降低性能
- 计算开销极低：Shapley评估在Concat/MLP融合下仅增加0.4%-0.9%的训练时间

## 亮点与洞察

- Shapley值首次成功应用于回归任务的模态贡献评估，用Pearson相关系数替代MSE解决了弱模态贡献度量偏差的关键问题
- AWC方法完全无额外参数，仅通过正则化实现模态平衡，设计简洁优雅
- FIM加权机制天然适应性强：强模态大梯度→高FIM→强约束，弱模态小梯度→低FIM→弱约束
- 学习窗口设计巧妙，只在关键学习期干预，避免了长期正则化对模型容量的限制

## 局限与展望

- 仅在MM-Fi一个数据集上验证，泛化能力有待更多场景检验
- 四种模态中mmWave和WiFi的固有信息容量很低（WiFi单模态MPJPE高达167mm），引入它们的实际价值有限
- K-Means聚类在只有4个模态时可能不够鲁棒，模态数量更多时的表现未探讨
- 学习窗口长度 $K$ 需要手动调优，缺乏自适应机制

## 相关工作与启发

- **vs OGM-GE**: OGM-GE通过梯度调制减缓优势模态梯度，但在回归任务中效果有限（MPJPE 55.51 vs 本文51.16），因为它只调节梯度大小而不约束参数更新方向
- **vs MMPareto**: MMPareto基于Pareto前沿优化多模态梯度，需要额外的单模态头引入更多参数，本文无额外参数更简洁
- **vs G-Blending**: G-Blending在本数据集上反而劣于baseline（58.40 vs 53.87），说明针对分类设计的方法不能直接迁移到回归任务

## 评分

- 新颖性: ⭐⭐⭐⭐ Shapley值+Pearson相关系数在回归任务中的组合有新意，但整体框架相对直接
- 实验充分度: ⭐⭐⭐ 只有一个数据集，虽然三种协议+三种融合+消融比较全面，但外部验证不足
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，弱模态预测坍缩的分析（Figure 3）和模态竞争的实验证据有说服力
- 价值: ⭐⭐⭐ 方法有一定通用性但应用场景局限，多模态HPE本身还不是主流方向

<!-- RELATED:START -->

## 相关论文

- [CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)

<!-- RELATED:END -->
