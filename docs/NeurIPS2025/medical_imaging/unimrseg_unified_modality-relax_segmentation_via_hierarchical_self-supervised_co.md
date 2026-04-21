---
title: >-
  [论文解读] UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation
description: >-
  [NeurIPS 2025][医学图像][缺失模态分割] 提出UniMRSeg，一种统一的模态缺失分割框架，通过层次化自监督补偿机制（HSSC）——从输入级模态重建、特征级对比学习到输出级一致性约束——用100%共享参数在所有可能的模态组合下实现最优平均性能和最小性能波动。
tags:
  - NeurIPS 2025
  - 医学图像
  - 缺失模态分割
  - 自监督补偿
  - 对比学习
  - 反向注意力适配器
  - 统一参数
---

# UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation

**会议**: NeurIPS 2025  
**arXiv**: [2509.16170](https://arxiv.org/abs/2509.16170)  
**代码**: [GitHub](https://github.com/Xiaoqi-Zhao-DLUT/UniMRSeg)  
**领域**: 医学图像 / 多模态分割  
**关键词**: 缺失模态分割, 自监督补偿, 对比学习, 反向注意力适配器, 统一参数

## 一句话总结

提出UniMRSeg，一种统一的模态缺失分割框架，通过层次化自监督补偿机制（HSSC）——从输入级模态重建、特征级对比学习到输出级一致性约束——用100%共享参数在所有可能的模态组合下实现最优平均性能和最小性能波动。

## 研究背景与动机

多模态图像分割在自动驾驶、医学诊断和机器人等关键应用中至关重要，但现实场景中经常面临模态不完整问题（传感器故障、数据质量低、临床约束等）。例如脑肿瘤诊断理想上需要Flair、T1ce、T1、T2四种MRI模态，但实际扫描中可能无法获取全部模态。

现有方法的核心问题：

**部署成本高**：多数方法为不同模态组合训练单独模型或独立编码器参数，需要穷举式的模型子集和额外的模态分类前置步骤。4种MRI模态产生15种有效缺失组合，需要大量独立模型。

**重建方法的局限**：基于模态重建的方法（M3AE、SSLSOD等）试图预测缺失模态来对齐特征，但：(a) 预训练重建模型优先压缩全局特征，产生的特征表示对需要精确空间信息的分割来说不够充分；(b) 将低质量重建结果级联到分割网络会加剧错误传播。

**自监督技术的孤立使用**：现有工作通常独立发展掩码重建、对比学习或知识蒸馏，未有效整合这三种技术的协同效应。

UniMRSeg的目标：用一套共享参数在所有模态组合下接近完整模态的表示质量。

## 方法详解

### 整体框架

三阶段渐进式学习框架，基于统一的3D U-Net风格编码器-解码器结构（嵌入3D ASPP，膨胀率[1, 6, 12, 18]）：
- Stage 1：多粒度模态重建（输入级补偿）
- Stage 2：模态不变对比学习（特征级补偿）
- Stage 3：不完整模态自适应微调（输出级补偿）

### 关键设计

1. **多粒度模态重建（Stage 1）**：结合三种数据扰动策略——(a) **模态随机丢弃**：以50%概率随机丢弃部分模态，但至少保留1个；(b) **模态顺序打乱**：随机打乱剩余模态顺序，打破模型对固定模态顺序的依赖，解耦模态无关表示；(c) **空间掩码**：随机掩盖部分输入数据区域。将扰动后的样本输入3D U-Net重建网络，用原始完整模态的归一化切片作为重建目标，损失为L1+SSIM。设计动机：通过多粒度扰动迫使模型同时学习细粒度局部模式和整体语义表示。

2. **模态不变对比学习（Stage 2）**：构建正负样本关系——同一样本的完整模态 $I_k$ 和随机缺失版本 $\hat{I}_k$ 为正对，不同样本为负对。在编码器的5个层级上使用NT-Xent损失：

$$l^i(u,v) = -\log \frac{\exp(\text{sim}(\mathbf{f}_u^i, \mathbf{f}_v^i) / \tau)}{\sum_{k=1}^{2B} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(\mathbf{f}_u^i, \mathbf{f}_k^i) / \tau)}$$

关键是同时优化分割约束（Dice loss）引导特征聚类方向，确保学到的对比空间与分割目标对齐，而非通用表示。

3. **反向注意力适配器（Stage 3）**：冻结编码器，仅训练解码器和轻量级适配器。适配器的工作原理：

    - 不完整模态特征 $\hat{F}^i_{cp}$ 通过冻结编码器获得
    - 通过3D卷积生成初始自适应特征 $\hat{F}^i_{ada-in}$
    - 融合后的特征 $\hat{F}^i_h$ 经3D Swin Transformer捕获全局跨模态关联
    - 生成互注意力图后取**反向操作**：高亮编码器无法感知的困难语义区域
    - 反向注意力乘以 $\hat{F}^i_h$ 得到补偿特征 $\hat{F}^i_{ada}$

数学原理：$f_{\text{inc}} + \mathcal{A}(f_{\text{inc}}) \approx f_{\text{com}}$，适配器 $\mathcal{A}$ 作为残差修正。冻结编码器是有意的设计——保护Stage 2中建立的任务引导对比表示。

### 损失函数 / 训练策略

- **Stage 1**：L1 + SSIM重建损失
- **Stage 2**：$L_{\text{NT-Xent}}$（5层级对比损失） + $L_{\text{Dice}}$（分割约束），联合优化
- **Stage 3**：$L_{\text{fc}} = \frac{1}{B} \sum_{k=1}^B \sum_m^M \frac{1}{5} \sum_{i=1}^5 \|F_k^i - \hat{F}_{k,m}^i\|_1$（5层特征一致性） + $L_{\text{pc}} = \frac{1}{B} \sum_{k=1}^B \sum_{m=1}^M l_{\text{Dice}}(P_k, \hat{P}_{k,m})$（预测一致性） + $L_{\text{Dice}}$（完整模态监督），$M=14$种不完整组合
- AdamW优化器，学习率0.0001，权重衰减0.00001，300 epochs，warmup调度

## 实验关键数据

### 主实验（BraTS2020脑肿瘤分割，15种模态组合的平均Dice%）

| 方法 | Whole ↑ | Core ↑ | Enhancing ↑ | Std Dev ↓ |
|------|--------|--------|------------|---------|
| NestedFormer | 52.01 | 39.59 | 40.78 | 23.09/19.53/24.20 |
| SFusion | 73.23 | 60.90 | 48.14 | 10.47/17.07/21.80 |
| ShaSpec | 74.81 | 65.16 | 55.90 | 10.08/15.45/21.62 |
| PASSION | 76.39 | 66.06 | 58.53 | 10.07/15.34/21.84 |
| **UniMRSeg** | **80.64** | **73.33** | **63.10** | **8.43/13.04/19.86** |

### 消融实验（各阶段+各组件贡献）

| 配置 | Whole | Core | Enhancing | 说明 |
|------|-------|------|-----------|------|
| Baseline (3D-UNet) | 63.31 | 51.60 | 38.40 | — |
| + 模态随机丢弃 | 66.98 | 55.47 | 42.25 | +3.7/+3.9/+3.9 |
| + 模态顺序打乱 | 67.78 | 56.85 | 44.17 | 进一步去耦合 |
| + 空间掩码 (Stage 1完成) | 69.35 | 59.89 | 47.12 | 比baseline+14.7% |
| + 对比学习 (编码器) | 72.45 | 64.02 | 51.45 | 特征级补偿显著 |
| + 分割约束 (Stage 2完成) | 74.53 | 65.25 | 53.97 | 任务引导至关重要 |
| + 特征一致性 (适配器) | 78.12 | 69.38 | 59.25 | 反向注意力适配器有效 |
| **+ 预测一致性 (Stage 3完成)** | **80.64** | **73.33** | **63.10** | **比baseline +44.6%** |
| 单阶段联合训练 | 20.32 | 13.67 | 10.03 | 完全无法收敛 |

### 关键发现

- **统一参数的优越性**：UniMRSeg用100%共享参数在所有模态组合上取得最优平均性能和最小标准差，无需模态分类前置步骤
- **层次补偿的协同效应**：三级补偿并非简单叠加——单独输入级补偿+6.0%，单独特征级+9.2%，单独输出级+6.2%，但三者组合达到+17.3%，超过简单相加
- **三阶段设计的必要性**：单阶段联合训练完全无法收敛（Dice降至10-20%），6个损失项相互竞争导致优化不稳定
- **冻结编码器的必要性**：Stage 3中微调编码器导致6.4%性能下降，破坏了适配器-编码器的协同关系
- **跨任务泛化**：在脑肿瘤分割(MRI 15组合)、RGB-D显著目标分割(3组合)、RGB-T显著目标分割(3组合)、RGB-D语义分割(3组合)四个任务上均取得最优

## 亮点与洞察

- **统一参数 + 全组合覆盖**：单一模型处理21种模态组合（MRI-15 + RGB-D-3 + RGB-T-3），大幅简化了临床部署
- **自监督技术的有机整合**：首次将掩码重建、对比学习和知识蒸馏三种SSL范式在同一任务中有效组合，证明了它们的互补性
- **反向注意力的巧妙设计**：不是直接补偿，而是找到编码器"看不到"的区域再针对性补偿——这一逆向思维非常直觉且有效
- **深入的消融分析**：从单阶段失败到三阶段成功的对比极具说服力

## 局限与展望

- 自监督预训练仅在各任务自身训练集上进行，未利用外部大规模数据
- 3D U-Net基础架构相对简单，换用更强的骨干（如nnU-Net V2、SwinUNETR）可能获得进一步提升
- Stage 3需要为所有模态组合（MRI有14种不完整组合）并行前向传播，训练成本较高
- 未探索动态决定最少必需模态数量的策略

## 相关工作与启发

- 继承了MAE的掩码重建思想、SimCLR的对比学习框架和知识蒸馏的师生范式，创新在于三者的有机整合
- 反向注意力思想与显著目标检测中的reverse attention有异曲同工之妙
- 对临床场景中模态缺失的鲁棒AI系统设计具有直接参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 层次化自监督补偿的整合思路新颖，反向注意力适配器设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 四个不同任务、21种模态组合、极其详细的消融实验，说服力极强
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰完整，三阶段流程可视化直观
- 价值: ⭐⭐⭐⭐⭐ 解决了多模态分割中的实际痛点，统一参数的设计对临床部署意义重大

<!-- RELATED:START -->

## 相关论文

- [Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)

<!-- RELATED:END -->
