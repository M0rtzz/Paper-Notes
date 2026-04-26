---
title: >-
  [论文解读] 3D Test-time Adaptation via Graph Spectral Driven Point Shift
description: >-
  [ICCV 2025][3D视觉][测试时自适应] 提出 GSDTTA，将3D点云测试时自适应从空间域转移到图谱域，仅优化最低10%频率分量即可适配点云的全局结构，配合特征图引导的自训练策略，在 ModelNet40-C 和 ScanObjectNN-C 上达到 SOTA。
tags:
  - ICCV 2025
  - 3D视觉
  - 测试时自适应
  - 点云分类
  - 图谱域
  - 图傅里叶变换
  - 域偏移
---

# 3D Test-time Adaptation via Graph Spectral Driven Point Shift

**会议**: ICCV 2025  
**arXiv**: [2507.18225](https://arxiv.org/abs/2507.18225)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 测试时自适应, 点云分类, 图谱域, 图傅里叶变换, 域偏移

## 一句话总结
提出 GSDTTA，将3D点云测试时自适应从空间域转移到图谱域，仅优化最低10%频率分量即可适配点云的全局结构，配合特征图引导的自训练策略，在 ModelNet40-C 和 ScanObjectNN-C 上达到 SOTA。

## 研究背景与动机
点云分类模型（如 DGCNN）在干净数据上训练后，遇到真实世界噪声（背景干扰、遮挡、LiDAR 噪声等）时精度可下降 35% 以上。测试时自适应（TTA）方法能够在推理阶段动态适配模型，但现有的3D TTA方法存在以下痛点：

**空间域优化维度高**：CloudFixer 和 3DD-TTA 需要学习每个点的偏移 $\Delta P \in \mathbb{R}^{N \times 3}$（N 通常超过1024），优化空间大、收敛慢

**需要额外训练数据**：MATE 需要源域辅助任务，BFTT3D 需要提取源域原型，违反了严格的 TTA 设定

**依赖扩散模型**：CloudFixer 和 3DD-TTA 利用预训练扩散模型修复点云，增加了计算开销

**核心洞察**：点云图谱域具有两个关键性质——(1) 能量高度集中在低频分量（约95%能量集中在低频），只需优化最低10%频率即可控制全局形状，参数量减少约90%；(2) 拉普拉斯特征图是域无关的描述子，不受源域偏差影响，适合在适配早期引导伪标签生成。

**核心 idea**：在图谱域进行可学习的低频调整，通过逆图傅里叶变换生成点偏移（Graph Spectral Driven Point Shift），同时利用特征图引导的自训练策略交替优化输入和模型参数。

## 方法详解

### 整体框架
GSDTTA 由两个核心模块组成：**图谱驱动点偏移（GSDPS）** 和 **图谱引导模型适配（GSGMA）**。两者交替迭代运行：每批测试数据先做4步输入适配，再做1步模型适配，循环10次。

### 关键设计

1. **异常感知图构建**：对输入点云构建 kNN 图，用径向基函数 $w_{ij} = \exp(-d^2(x_i,x_j) / 2\delta^2)$ 作为边权重。通过度阈值 $\tau = \frac{\gamma}{Nk}\sum A_{ij}$ 过滤离群点（度值低于阈值的点被移除），提高后续谱分析的鲁棒性。

2. **图谱域低频调整**：对异常感知图计算拉普拉斯矩阵 $L_o = D_o - A_o$ 并特征分解得到特征向量矩阵 $U_o$。通过 GFT 将点云变换到谱域 $\hat{X} = U_o^T X$，然后仅在前 M=100 个低频分量上加可学习调整量 $\Delta\hat{X} \in \mathbb{R}^{M \times 3}$，高频分量不动。最后通过 IGFT 变回空间域 $X_s = U_o \hat{X}_a$。

    - **设计动机**：约95%的能量集中在低频分量（图2验证），仅优化 M=100 个参数（而非 N=1024 个点 × 3维 = 3072 个参数），大幅降低优化难度。

3. **特征图引导的自训练策略**：综合深度特征描述子 $f_d$（模型提取的全局特征）和谱描述子 $f_s$（特征图 max-pooling），按凸组合生成伪标签：$\hat{y}_i = \arg\min_c (\alpha \frac{f_d^T q_d^c}{\|f_d\|\|q_d^c\|} + (1-\alpha) \frac{f_s^T q_s^c}{\|f_s\|\|q_s^c\|})$，其中 $q_d^c, q_s^c$ 为类中心。

    - **设计动机**：谱描述子是域无关的，在适配初期模型尚未调整时特别有价值，可补偿深度特征的源域偏差。

### 损失函数 / 训练策略
- **输入适配目标**：$\mathcal{L}_{IA} = \mathcal{L}_{pl} + \beta_1(\mathcal{L}_{ent} + \mathcal{L}_{div}) + \beta_2\mathcal{L}_{cd}$
    - $\mathcal{L}_{pl}$：伪标签交叉熵损失
    - $\mathcal{L}_{ent} + \mathcal{L}_{div}$：信息最大化损失（个体确信 + 整体多样）
    - $\mathcal{L}_{cd}$：Chamfer Distance，确保适配后点云不偏离原始位置太远
- **模型适配目标**：$\mathcal{L}_{MA} = \mathcal{L}_{pl} + \beta_3(\mathcal{L}_{ent} + \mathcal{L}_{div})$
- 优化器：AdamW，学习率 0.0001，batch size 32

## 实验关键数据

### 主实验

| Backbone | Method | Background | Occlusion | LiDAR | Mean |
|----------|--------|-----------|-----------|-------|------|
| DGCNN | Source-only | 49.71 | 33.26 | 14.91 | 66.51 |
| DGCNN | TENT | 60.65 | 42.94 | 33.38 | 75.91 |
| DGCNN | CloudFixer | 74.55 | 35.94 | 37.48 | 76.54 |
| DGCNN | **GSDTTA** | **88.57** | **45.38** | 31.52 | **79.07** |
| CurveNet | SHOT | 66.49 | 58.63 | 56.04 | 81.24 |
| CurveNet | CloudFixer | 66.07 | 37.13 | 38.76 | 77.91 |
| CurveNet | **GSDTTA** | **87.84** | **50.73** | **44.45** | **82.63** |
| PointNeXt | TENT | 80.43 | 51.90 | 46.92 | 81.08 |
| PointNeXt | CloudFixer | 79.28 | 38.32 | 35.73 | 76.04 |
| PointNeXt | **GSDTTA** | **91.29** | **55.06** | **46.84** | **82.51** |

在 ModelNet40-C 上，GSDTTA 在 DGCNN backbone 下平均准确率 79.07%，超越 CloudFixer 2.53%，超越 3DD-TTA 7.38%。在 Background 腐蚀上提升尤为显著（88.57% vs CloudFixer 74.55%）。

### 消融实验

| 组件搭配 | Mean Acc (DGCNN) |
|---------|-----------------|
| 仅空间域适配 (Baseline) | ~75 |
| 图谱域适配 (w/o eigenmap 引导) | ~77 |
| 图谱域适配 + 深度特征伪标签 | ~78 |
| 图谱域适配 + 特征图引导伪标签 (Full) | **79.07** |

在 ScanObjectNN-C 上，GSDTTA 同样一致优于所有对比方法，DGCNN backbone 下达到 61.83% 均值（对比 CloudFixer 60.73%）。

### 关键发现
- 图谱域低频调整仅需约100个参数，是空间域方法所需参数（3072）的 3%，在有限测试数据下更容易优化
- 特征图描述子在 Background 腐蚀上帮助最大（+14%），因为该腐蚀改变全局结构但保留局部几何
- 迭代交替优化（输入+模型）比单独优化任一方都好

## 亮点与洞察
- **从图信号处理角度解决 TTA 问题**，是将图谱分析引入3D TTA 的首次尝试，思路新颖
- 低频调整的参数效率极高（减少90%参数量），非常适合在线/流式 TTA 场景
- 特征图作为域无关描述子的使用，为 TTA 领域提供了新的伪标签生成思路
- 不需要额外训练数据或预训练扩散模型，严格遵循 TTA 设定

## 局限与展望
- 图拉普拉斯特征分解计算复杂度为 $O(N^3)$，在大规模点云上开销较大
- 异常点过滤的阈值 $\gamma$ 和 kNN 参数 k 需要手动设定
- 目前仅在分类任务上验证，尚未扩展到分割、检测等密集预测任务
- LiDAR 和 Occlusion 腐蚀上提升相对其他腐蚀类型要小，说明极端稀疏/缺失场景下谱域调整的效果有限

## 相关工作与启发
- **CloudFixer / 3DD-TTA**：空间域点云修复方法，依赖扩散模型，本文在不需要扩散模型的情况下取得更好结果
- **图谱分析**：借鉴了 Global Point Signature 等图信号处理方法的思想，将谱分析从描述子提升到适配工具
- **信息最大化**：ℒ_ent + ℒ_div 的组合来自 LAME 等 TTA 方法
- **启发**：该方法的谱域操作思路可推广到其他3D任务（如点云去噪、补全），也可探索用于2D图像 TTA

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将图谱域分析引入3D TTA，视角独特
- 实验充分度: ⭐⭐⭐⭐ 三个 backbone × 两个数据集，覆盖全面；缺少效率对比
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，公式推导严谨
- 价值: ⭐⭐⭐⭐ 参数高效的 TTA 方案，有实际部署价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](constraint-aware_feature_learning_for_parametric_point_cloud.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene](event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)

<!-- RELATED:END -->
