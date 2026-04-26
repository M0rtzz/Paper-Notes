---
title: >-
  [论文解读] CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution
description: >-
  [AAAI 2026][医学图像][多对比度MRI超分辨率] 提出 CD-DPE 网络，通过迭代卷积字典特征解耦模块（CD-FDM）将多对比度 MRI 特征分离为跨对比度共有和模态特有成分，再利用双提示特征融合专家模块（DP-FFEM）进行自适应融合重建，在多个公开数据集上超越现有 SOTA 方法。
tags:
  - AAAI 2026
  - 医学图像
  - 多对比度MRI超分辨率
  - 卷积字典
  - 特征解耦
  - 双提示
  - 专家网络
---

# CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2511.14014](https://arxiv.org/abs/2511.14014)  
**代码**: [有](https://github.com/xianming-gu/CD-DPE)  
**领域**: 医学图像  
**关键词**: 多对比度MRI超分辨率, 卷积字典, 特征解耦, 双提示, 专家网络

## 一句话总结

提出 CD-DPE 网络，通过迭代卷积字典特征解耦模块（CD-FDM）将多对比度 MRI 特征分离为跨对比度共有和模态特有成分，再利用双提示特征融合专家模块（DP-FFEM）进行自适应融合重建，在多个公开数据集上超越现有 SOTA 方法。

## 研究背景与动机

MRI 超分辨率（SR）旨在从低分辨率（LR）扫描重建高分辨率（HR）图像以提升诊断精度。临床上通常采集多种对比度序列（T1W、T2W、PD 等），快速获取的 HR 参考图（如 T1W）可辅助增强需要更长扫描时间的 LR 目标图（如 T2W）。

**现有方法的三大局限**：

**简单融合策略**：早期 CNN 方法直接拼接参考和目标图像，无法捕获复杂的跨对比度依赖关系，导致重建细节模糊

**Transformer 方法的局限**：虽然注意力机制能建模长程依赖，但在极低分辨率输入下重建高频细节能力有限，且计算开销大、显存消耗高

**分解方法缺乏严格约束**：现有将参考图分解为共有/特有成分的方法（如 Lei et al.）缺乏对分解和融合机制的严格约束，共有特征可能过度平滑

核心挑战：**如何有效提取多对比度 MRI 中的共有和特有特征，并在保留结构细节的同时消除冗余信息干扰？**

## 方法详解

### 整体框架

CD-DPE 包含两个核心模块：

1. **CD-FDM（卷积字典特征解耦模块）**：提取共有特征和特有特征
2. **DP-FFEM（双提示特征融合专家模块）**：自适应融合特征并重建 HR 图像

### 关键设计

#### 1. 卷积字典特征解耦模块（CD-FDM）

CD-FDM 基于卷积字典学习思想，将多对比度 MRI 分解为三组稀疏表示：

**数学建模**：多对比度图像可分解为：

$$I_x^s = \sum_j^J u_j^x \otimes \theta_d^x + c_j \otimes \theta_d^c, \quad I_y = \sum_j^J u_j^y \otimes \theta_d^y + c_j \otimes \theta_d^c$$

其中 $u_j^x, u_j^y$ 为特有稀疏表示，$c_j$ 为共有稀疏表示，$\theta_d$ 为字典滤波器。

**迭代更新过程**（展开学习思想）：

- **特有特征提取**：$U_x^l = \text{Prox}(U_x^{l-1} - \eta_x \Delta U_x)$，通过 CDME（编码器）和 CDMD（解码器）的残差迭代优化
- **参考图像对齐**：引入 OffNet（偏移网络），使用轻量级 U-Net 学习位移场 $\phi$ 和特征表示 $\mathcal{A}$，通过空间变换对齐参考特征与目标图像
- **共有特征更新**：从公共特征中减去参考和目标特有特征的残差，确保共有特征仅保留两者真正共享的结构信息
- 迭代 L=3 次，逐步优化解耦质量

**关键子结构**：

- CDM 编码器/解码器：3 级多尺度结构，通道数 64/96/128
- MFFN（多尺度前馈网络）：实现近端算子 Prox
- OffNet：处理参考与目标图像的空间错位

#### 2. 双提示特征融合专家模块（DP-FFEM）

DP-FFEM 通过两种提示机制引导特征融合：

**频率提示（Frequency Prompt）**：

- 构建参考表示 $F_r = [F_y^L, F_c^L]$ 和目标表示 $F_t = [F_x^L, F_c^L]$
- 学习可训练的频率原型 $\mathcal{P}_F$，对参考特征的傅里叶变换进行注意力调制
- 生成注意力图 $\mathcal{V}^y = f_{\phi_1}(\mathscr{F}(F_r), \mathcal{P}_F)$
- 将参考图的注意力迁移到目标表示：$\tilde{F}_t = F_t \otimes \mathcal{V}^y + F_t$

**自适应路由提示（Adaptive Routing Prompt）**：

- 可学习路由提示 $\mathcal{P}_R \in \mathbb{R}^{(C \times H \times W) \times E}$
- 与目标特征相乘生成路由 logits，Top-K 选择最相关的 K 个专家分支
- Softmax 归一化得到路由权重 $\mathcal{V}^x$
- 最终重建：$\hat{I}_x = \sum_{i=1}^E \mathcal{V}^x \cdot \mathcal{E}_i(\tilde{F}_t \cdot \mathcal{V}^x)$
- 设置 E=4 个专家，K=2

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{fc} + \lambda_2 \mathcal{L}_{mi}$

- **重建损失** $\mathcal{L}_{rec} = \|\hat{I}_x - I_x^{hr}\|_1$：L1 距离确保内容一致性
- **一致性损失** $\mathcal{L}_{fc}$：约束特有+共有特征的组合能重建原始图像，$\lambda_y = 0.01$ 平衡两项
- **解耦损失** $\mathcal{L}_{mi}$：最小化共有特征与特有特征之间的互信息，强制特征独立性

训练设置：Adam 优化器（lr=1e-4），batch size=4，50 epochs，NVIDIA RTX A6000 48GB，$\lambda_1=1, \lambda_2=0.1$。

## 实验关键数据

### 主实验

在 BraTS2018 和 IXI 两个公开数据集上对比 5 种 SOTA 方法：

**BraTS2018 数据集（T1W→T2W 重建）：**

| 方法 | 2× PSNR↑ | 2× SSIM↑ | 4× PSNR↑ | 4× SSIM↑ | Params(M) | FLOPs(G) |
|------|----------|----------|----------|----------|-----------|----------|
| WavTrans | 39.79 | 0.9874 | 34.83 | 0.9677 | 10.0 | 216.2 |
| A2-CDic | 40.47 | 0.9883 | 35.70 | 0.9704 | 10.1 | 831.1 |
| **CD-DPE** | **40.70** | **0.9885** | **36.00** | **0.9716** | 11.7 | 426.1 |

**IXI 数据集（PD→T2W 重建）：**

| 方法 | 2× PSNR↑ | 4× PSNR↑ | 4× SSIM↑ |
|------|----------|----------|----------|
| WavTrans | 42.88 | 38.51 | 0.9711 |
| A2-CDic | 41.59 | 37.91 | 0.9726 |
| **CD-DPE** | **43.22** | **38.59** | **0.9735** |

### 消融实验

**模块消融（BraTS2018 4× SR）：**

| 设置 | PSNR 变化 | SSIM 变化 |
|------|-----------|-----------|
| w/o CD-FDM（CNN 替代） | -13.48% | -1.92% |
| w/o DP-FFEM（CNN 替代） | -5.93% | -0.55% |
| w/o 双提示（保留 DP-FFEM） | 低于完整模型 | 低于完整模型 |
| w/o $\mathcal{L}_{mi}$ | -3.05% | -0.22% |
| w/o $\mathcal{L}_{fc}$ | -2.90% | -0.31% |

**泛化性实验（IXI 训练 → FastMRI Knee 测试）：**

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| WavTrans | 28.07 | 0.7428 |
| A2-CDic | 25.21 | 0.7517 |
| **CD-DPE** | **29.41** | **0.8387** |

CD-DPE 在未见数据集上 PSNR 提升 4.8%，SSIM 提升 11.6%，泛化能力远超其他方法。

### 关键发现

- CD-FDM 是性能的核心支柱，移除后 PSNR 下降 13.48%——证明卷积字典解耦远优于简单 CNN 分解
- 互信息损失 $\mathcal{L}_{mi}$ 对防止特征纠缠至关重要，移除后特有与共有特征无法正确分离
- DP-FFEM 的双提示机制互补：频率提示引导特征选择，路由提示决定最优融合策略
- CD-DPE 参数量（11.7M）和推理时间（0.061s）适中，FLOPs（426G）低于 A2-CDic（831G）

## 亮点与洞察

1. **卷积字典解耦的严谨性**：用展开学习（unfolding learning）将优化问题转化为可学习的网络层，比启发式分解更有理论支撑
2. **双提示的互补设计**：频率提示在频域捕获结构模式（what to fuse），路由提示决定融合路径（how to fuse），分工明确
3. **MoE 思想的引入**：专家网络+路由提示的组合是 MoE 在图像重建中的自然应用，增加了融合策略的灵活性
4. **泛化能力突出**：在完全未见的 FastMRI Knee 数据集上大幅领先，说明解耦-融合范式具有内在的泛化优势

## 局限与展望

- **对比度差异敏感**：当参考与目标的对比度机制差异极大时性能受限，可引入 MRI 物理先验（如弛豫时间映射）
- **迭代计算开销**：CD-FDM 的迭代展开引入额外计算，需探索更高效的解耦机制
- **仅限 2D 切片**：实验在 2D 切片上进行，未扩展到 3D 体数据
- **单一参考图像**：仅使用一种对比度作为参考，可探索多参考多对比度联合重建

## 相关工作与启发

- **A2-CDic**：同样基于卷积字典但缺乏双提示融合，CD-DPE 在其基础上引入 MoE 思想
- **DiffMSR**：使用扩散模型进行多对比度 SR，推理效率较低
- **DANCE**：邻域引导聚合策略，手工设计泛化性差
- 卷积字典解耦 + MoE 融合的框架可推广到其他多模态医学图像重建任务（CT-MRI fusion 等）

## 评分

- **创新性**: ★★★★☆ — 卷积字典解耦与双提示 MoE 的结合有新意
- **实验充分度**: ★★★★★ — 两个数据集、详细消融、泛化性验证、特征可视化
- **写作质量**: ★★★★☆ — 数学推导严谨，但公式较多读起来偏重
- **实用性**: ★★★★☆ — 推理速度快（0.061s），有代码开源，泛化性强

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[AAAI 2026\] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)
- [\[CVPR 2026\] MRI Contrast Enhancement Kinetics World Model](../../CVPR2026/medical_imaging/mri_contrast_enhancement_kinetics_world_model.md)
- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)

<!-- RELATED:END -->
