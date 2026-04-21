---
title: >-
  [论文解读] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior
description: >-
  [ECCV 2024][3D视觉][自监督深度估计] 提出RPrDepth，在训练阶段利用多帧/高分辨率等"富资源"模型的特征和预测作为先验，通过先验深度融合模块和富资源引导损失，使仅用**低分辨率单张图像**推理的模型达到甚至超过多帧高分辨率模型的深度估计精度。
tags:
  - ECCV 2024
  - 3D视觉
  - 自监督深度估计
  - 单目深度估计
  - 知识蒸馏
  - 先验融合
  - 注意力特征选择
---

# High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior

**会议**: ECCV 2024  
**arXiv**: [2408.00361](https://arxiv.org/abs/2408.00361)  
**代码**: https://github.com/wencheng256/RPrDepth (有)  
**领域**: 3D视觉  
**关键词**: 自监督深度估计, 单目深度估计, 知识蒸馏, 先验融合, 注意力特征选择

## 一句话总结

提出RPrDepth，在训练阶段利用多帧/高分辨率等"富资源"模型的特征和预测作为先验，通过先验深度融合模块和富资源引导损失，使仅用**低分辨率单张图像**推理的模型达到甚至超过多帧高分辨率模型的深度估计精度。

## 研究背景与动机

**领域现状**: 自监督单目深度估计通过视点重建损失避免了对激光雷达标注的依赖。当前性能最优的方法大多依赖"富资源"输入——多帧图像、高分辨率图像、甚至未来帧。

**现有痛点**: 富资源输入在实际应用中往往不可用。例如车辆静止时无法获取多帧数据，未来帧在实时系统中根本不存在。这严重限制了高性能方法的实用性。

**核心矛盾**: 单张低分辨率图像缺少富资源输入中编码的关键信息（如帧间视差），信息量的本质差距导致性能天花板。

**本文目标**: 在推理阶段仅用单张低分辨率图像，就能达到富资源模型的深度估计精度。

**切入角度**: 富资源数据虽然推理时不可用，但**训练时是可用的**。将富资源模型的特征和预测作为离线先验信息，通过特征检索和融合弥补低资源输入的信息缺口。

**核心 idea**: 离线构建富资源参考特征库，推理时为每个像素检索相似先验特征，融合后恢复出与富资源模型可比的深度精度。

## 方法详解

### 整体框架

训练包含两个分支：
- **上分支（富资源分支）**: 使用预训练的ManyDepth-HR（多帧+高分辨率），固定参数不更新。提供参考特征 $f_r$ 和参考深度 $D_r$。
- **下分支（单图分支）**: 目标模型基于DIFFNet，接受低分辨率单张图像，通过先验融合提升性能。

推理阶段仅保留单图分支 + 压缩后的先验数据（从260万像素压缩到2.5万像素）。

### 关键设计

1. **先验深度融合模块 (Prior Depth Fusion Module)**: 包含两种融合方式：

   **像素级融合 (Pixel-wise Fusion)**: 通过亲和矩阵从参考特征中检索最相似像素。目标特征 $F_s$ 和参考特征 $F_r$ 通过维度对齐后计算亲和度：

   $$\mathcal{A} = \text{Softmax}(F_s \otimes F_r)$$

   利用亲和矩阵构建像素级参考特征和参考深度：

   $$F_c = \mathcal{A} \times f_r, \quad D_c = \mathcal{A} \times D_r$$

   $F_c$ 提供空间几何先验，$D_c$ 提供直接的深度先验。

   **深度提示融合 (Depth-hint Fusion)**: 使用Transformer的多头注意力机制，以目标特征 $F_s$ 作为Query，参考特征 $f_r$ 作为Key和Value：

   $$F_d = \text{MHA}(Q, K, V)$$

   这种全局注意力融合捕获更宏观的先验信息，与像素级融合互补。

   两种融合结果与单图特征拼接后经卷积压缩到原始通道数，得到信息丰富的特征 $F_o$。

2. **富资源引导损失 (Rich-resource Guided Loss)**: 两部分组成：

   **视点重建损失**: 利用富资源输入（高分辨率多帧）作为重建目标，提供比低分辨率源图像更精确的监督信号：

   $$\mathcal{L}_{\text{vp}} = l_{vp}(\text{Resize}(D_o), I_r)$$

   **一致性损失**: 利用富资源模型的深度预测作为伪标签。由于自监督模型预测的是相对视差而非绝对深度，不同模型存在尺度差异，因此最小化**梯度差异**而非直接值差异：

   $$\mathcal{L}_c = \|\tilde{G}_{x,y}(D_o) - \tilde{G}_{x,y}(D_p)\|_1$$

   其中 $\tilde{G}_{x,y}(\cdot)$ 为归一化后的x/y方向梯度之和。这迫使模型在边缘处生成与富资源模型一致的深度不连续性。

   **辅助损失**: 引导亲和矩阵学习：$\mathcal{L}_{\text{aux}} = \|D_p - \text{Resize}(D_c)\|_1$

   总损失：$\mathcal{L} = \alpha \mathcal{L}_{\text{vp}} + \beta \mathcal{L}_c + \mathcal{L}_{\text{aux}}$

3. **注意力引导特征选择 (Attention Guided Feature Selection)**: 推理时需检索参考特征，完整参考集计算开销大。解决方案：

    - 在验证集上统计所有参考像素的平均注意力权重：$W_{\text{avg}} = \frac{1}{N}\sum_{i=1}^{N}(\mathcal{A}_i + \mathcal{A}_{\text{MHA},i})$
    - 选择权重最高的像素子集作为压缩先验数据
    - 从260万像素压缩到2.5万像素（**1%**），替换后微调几个epoch
    - 惊人发现：压缩后性能不降反升，因为去掉了无关像素的干扰

### 损失函数 / 训练策略

- 使用DIFFNet（基于HR-Net）作为单图基线模型
- ManyDepth-HR作为富资源引导模型（多帧+高分辨率+未来帧）
- 参考数据集：从训练集随机选2000个三元组，离线提取特征（1%像素采样×2000张）
- 端到端训练，富资源分支固定参数
- 特征选择完成后替换参考集并微调

## 实验关键数据

### 主实验 (KITTI Eigen Split, 640×192分辨率, 单目训练)

| 方法 | 输入帧数 | Abs Rel↓ | Sq Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|----------|----------|---------|-------|-----------------|
| Monodepth2 | 1 | 0.115 | 0.903 | 4.863 | 0.877 |
| DIFFNet (基线) | 1 | 0.102 | 0.764 | 4.483 | 0.896 |
| ManyDepth (多帧) | 2 | 0.098 | 0.770 | 4.459 | 0.900 |
| **RPrDepth (单帧)** | **1** | **0.097** | **0.658** | **4.279** | **0.900** |

高分辨率(1024×320)结果：

| 方法 | 输入帧数 | Abs Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|----------|----------|-------|-----------------|
| ManyDepth-HR (引导模型) | 2 | 0.093 | 4.245 | 0.909 |
| **RPrDepth (单帧)** | **1** | **0.091** | **4.098** | **0.910** |

**单帧RPrDepth超越了其自身的多帧高分辨率教师模型！**

### 消融实验

| 组件 | Abs Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|----------|-------|-----------------|
| Baseline (DIFFNet) | 0.102 | 4.483 | 0.896 |
| + PDF (先验融合) | 0.098 | 4.284 | 0.898 |
| + AGFS (特征选择) | 0.098 | 4.240 | 0.898 |
| + RGL (引导损失) | 0.100 | 4.321 | 0.897 |
| + Full (全部) | 0.097 | 4.279 | 0.900 |

### 关键发现

- 先验深度融合模块是最大的性能提升来源（Abs Rel: 0.102 → 0.098）
- 特征选择将参考集压缩99%后RMSE反而继续下降（4.284 → 4.240），精简特征消除了噪声干扰
- RPrDepth在Make3D跨域测试中同样表现最优（AbsRel 0.288 vs BRNet 0.302），泛化能力强
- 梯度一致性损失有效解决了不同模型间的尺度不一致问题

## 亮点与洞察

- **训练-推理解耦**: 训练时用富资源数据构建先验，推理时仅需单张低分辨率图像，实用性极强
- **学生超越教师**: 单帧模型通过先验融合超越了多帧高分辨率教师模型（自动驾驶中运动物体区域，多帧方法反而有害，单帧+先验可以修正）
- **特征选择的哲学**: 少即是多——精选1%最具代表性的参考像素比用全部数据更有效
- **梯度而非值的一致性**: 巧妙绕开自监督深度估计的尺度模糊问题

## 局限与展望

- 参考特征集需要离线预处理和存储，虽然仅为2.5万像素但仍引入额外部署成本
- 先验质量受富资源模型能力约束，若教师模型在某些场景失效则先验信息也不可靠
- 当前仅验证了ManyDepth作为富资源模型，可探索更强的教师（如基于Transformer的DPT）
- 参考数据集的多样性需要覆盖目标场景分布，跨域部署需更新参考集
- 可扩展至其他稠密预测任务（如光流、语义分割）的知识迁移

## 相关工作与启发

- **ManyDepth**: 多帧自监督深度估计的代表，RPrDepth的主要教师模型，提供cost volume特征
- **DIFFNet**: 基于HR-Net的高效单帧基线，作为RPrDepth的学生backbone
- **Monodepth2**: 奠定了自监督单目深度估计的min-reprojection loss等基础设计
- 知识蒸馏视角：可看作更精细的师生学习——不只蒸馏输出，还蒸馏特征级的空间先验

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "训练时有富资源、推理时无富资源"的问题设定有实际价值，先验检索+融合的方案比直接蒸馏更灵活
- **实验充分度**: ⭐⭐⭐⭐ — KITTI/Make3D/Cityscapes三个数据集，多种训练模式，完整消融
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清晰，pipeline图表设计好
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接降低自动驾驶深度估计的传感器需求，且推理速度无额外开销

<!-- RELATED:START -->

## 相关论文

- [ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)
- [Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)

<!-- RELATED:END -->
