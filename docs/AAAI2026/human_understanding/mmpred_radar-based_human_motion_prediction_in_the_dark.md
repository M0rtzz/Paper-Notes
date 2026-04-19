---
title: >-
  [论文解读] mmPred: Radar-based Human Motion Prediction in the Dark
description: >-
  [AAAI2026][人体理解][毫米波雷达] 首次将毫米波雷达引入人体运动预测(HMP)任务，提出mmPred——基于扩散模型的框架，通过双域历史运动表示（时域姿态细化TPR + 频域主导运动FDM）和全局骨骼关系Transformer(GST)，有效抑制雷达特有的噪声和时序不一致性，在mmBody和mm-Fi数据集上分别超越SOTA方法8.6%和22%。
tags:
  - AAAI2026
  - 人体理解
  - 毫米波雷达
  - 人体运动预测
  - 扩散模型
  - 频域表示
  - 双域融合
---

# mmPred: Radar-based Human Motion Prediction in the Dark

**会议**: AAAI2026  
**arXiv**: [2512.00345](https://arxiv.org/abs/2512.00345)  
**作者**: Junqiao Fan, Haocong Rao, Jiarui Zhang, Jianfei Yang, Lihua Xie (南洋理工大学)  
**代码**: 未公开  
**领域**: human_understanding  
**关键词**: 毫米波雷达, 人体运动预测, 扩散模型, 频域表示, 双域融合

## 一句话总结

首次将毫米波雷达引入人体运动预测(HMP)任务，提出mmPred——基于扩散模型的框架，通过双域历史运动表示（时域姿态细化TPR + 频域主导运动FDM）和全局骨骼关系Transformer(GST)，有效抑制雷达特有的噪声和时序不一致性，在mmBody和mm-Fi数据集上分别超越SOTA方法8.6%和22%。

## 背景与动机

人体运动预测(HMP)旨在从观测的历史姿态序列预测未来姿态，在人机交互、医疗健康和危险预防等领域有重要应用。现有HMP方法严重依赖高精度的多视角RGB-D动捕系统获取历史姿态，成本高昂且在实际部署中不切实际。单视角RGB方案在黑暗、遮挡等恶劣环境下性能急剧下降，且存在隐私泄露风险。

毫米波(mmWave)雷达工作在30-300 GHz频段，能穿透烟雾等视觉障碍物，在任意光照条件下可靠工作，且有限空间分辨率天然保护隐私，是室内HMP的理想传感器。然而，雷达点云存在两个关键问题：(1) **多径效应**产生"鬼点"噪声；(2) **镜面反射**导致间歇性部件漏检——某些身体部位反射信号偏离接收器。这些问题导致雷达点云时序不一致、噪声大。

现有雷达姿态估计器（如P4Transformer）逐帧独立处理，忽略时序不一致性，产生严重抖动和扭曲的姿态序列，丢失关键运动线索（如关节速度）。传统HMP方法假设输入为干净的MoCap历史姿态，对雷达估计的噪声历史极为敏感，直接套用会产生不真实的未来序列。因此，需要一个专门针对雷达模态设计的HMP框架。

## 核心问题

如何从噪声大、时序不一致的毫米波雷达点云中提取可靠的历史运动信息，并生成真实、稳定的未来人体姿态序列？

## 方法详解

### 整体框架

mmPred采用两阶段训练：(1) 训练双域历史运动估计模块；(2) 训练基于GST的扩散模型进行未来运动生成。输入为历史 $H$ 帧雷达点云 $R^{1:H} = \{R^i \in \mathbb{R}^{N \times 6}\}_{i=1}^H$，输出为未来 $F$ 帧姿态 $\hat{x}^{H+1:H+F}$。

### 关键设计1：双域历史运动估计

**时域姿态细化(TPR)**：对每帧雷达点云先通过预训练姿态估计器 $f_{\text{pose}}$ 得到粗姿态，再通过扩散式细化网络 $f_{\text{refine}}$ 利用肢体长度一致性和相邻帧连续性先验修复：

$$\tilde{x}_{\text{time}}^i = f_{\text{refine}}(f_{\text{pose}}(R^i))$$

**频域主导运动(FDM)**：将所有历史帧整体处理，通过Anchor-based点云编码器 $f_{\text{pc}}$ 和Transformer $\Phi$ 直接预测频域运动表示 $\tilde{X}_{\text{freq}}^{1:N_2} \in \mathbb{R}^{N_2 \times J \times 3}$（$N_2=3$或4个DCT系数）：

$$F_r = f_{\text{pc}}(R^{1:H}), \quad F_r', F_j = \Phi(F_r, \bar{F}_j)$$
$$\tilde{X}_{\text{freq}}^{1:N_2} = f_m(F_j)$$

低频DCT系数捕获主导运动趋势（均值姿态和速度），天然分离高频噪声。

### 关键设计2：跨域融合

将时域和频域表示统一变换到 $N_1$ 个DCT系数空间，通过两个MLP独立投影后逐元素相加：

$$C = f_1(\tilde{X}_{\text{time}}^{1:N_1}) + f_2(\tilde{X}_{\text{freq}}^{1:N_1})$$

得到关节级条件嵌入 $C \in \mathbb{R}^{J \times 384}$ 作为扩散模型的引导信号。

### 关键设计3：全局骨骼关系Transformer (GST)

GST隔离关节特征以防止漏检关节污染全局表示：
- **Skeleton Transformer (S-Transformer)**：将特征reshape为 $\mathbb{R}^{J \times (N_1 \times 384)}$，通过 $J$ 个关节token之间的self-attention建模全局关节协作，使漏检关节从可靠关节聚合信息
- **Frequency Transformer (F-Transformer)**：将特征reshape为 $\mathbb{R}^{N_1 \times (J \times C_a)}$，建模频率维度的时序运动模式，保证生成运动的时间平滑性

训练目标采用标准DDPM的 $\varepsilon$-prediction：

$$\mathcal{L}_2 = \|\hat{\varepsilon}_\theta(X_k, k, C) - \varepsilon_k\|^2$$

## 实验关键数据

### mmBody数据集：不同恶劣环境下的表现

| 方法 | 输入 | Lab1 ADE↓ | Rain ADE↓ | Smoke ADE↓ | Dark ADE↓ | Occlusion ADE↓ | 平均 ADE↓ |
|------|------|----------|----------|-----------|----------|---------------|----------|
| HumanMAC | GT | 0.235 | 0.297 | 0.338 | 0.287 | 0.265 | 0.291 |
| HumanMAC | RGB | 0.390 | 0.479 | 0.560 | 0.693 | 0.739 | 0.547 |
| PSGSN | mmWave | 0.503 | 0.536 | 0.598 | 0.513 | 0.485 | 0.537 |
| HumanMAC | mmWave | 0.411 | 0.455 | 0.496 | 0.406 | 0.391 | 0.460 |
| **mmPred** | **mmWave** | **0.369** | **0.436** | **0.472** | **0.392** | **0.378** | **0.420** |

mmPred比HumanMAC平均ADE降低8.6%、FDE降低6.4%。在Dark/Occlusion场景下，RGB方法ADE暴涨至0.693/0.739，而mmPred仅为0.392/0.378。

### mm-Fi数据集：不同动作类型

| 方法 | Raise Hand ADE↓ | Pickup ADE↓ | Throwing ADE↓ | Kicking ADE↓ | 平均 ADE↓ | 平均 FDE↓ |
|------|----------------|------------|--------------|-------------|----------|----------|
| PSGSN | 0.397 | 0.595 | 0.449 | 0.426 | 0.430 | 0.470 |
| HumanMAC | 0.363 | 0.547 | 0.439 | 0.425 | 0.408 | 0.396 |
| **mmPred** | **0.237** | **0.452** | **0.371** | **0.374** | **0.319** | **0.305** |

mmPred在mm-Fi上ADE降低22%、FDE降低23%。

### 消融实验

| 配置 | mmBody ADE↓ | mmBody FDE↓ | mm-Fi ADE↓ | mm-Fi FDE↓ |
|------|-----------|-----------|----------|----------|
| Baseline (M1) | 0.460 | 0.487 | 0.408 | 0.396 |
| +TPR (M2) | 0.455 | 0.485 | 0.379 | 0.359 |
| +FDM (M3) | 0.456 | 0.486 | 0.337 | 0.326 |
| +GST (M4) | 0.460 | 0.485 | 0.373 | 0.354 |
| TPR+FDM (M5) | 0.448 | 0.476 | 0.355 | 0.327 |
| FDM+GST (M6) | 0.423 | 0.458 | 0.325 | 0.310 |
| **Full (M8)** | **0.420** | **0.456** | **0.319** | **0.305** |

FDM在mm-Fi（点云更稀疏）上提升最为显著，S-Transformer使肢体长度误差从10.67降至9.92、jitter从7.01降至6.09。

## 亮点

- **首次将雷达引入HMP任务**：建立了完整的雷达-HMP框架，开辟了恶劣环境下运动预测的新方向
- **双域互补设计精巧**：时域TPR提供精确关节定位，频域FDM提供稳定运动趋势和速度信息，DCT天然分离高频噪声
- **GST的关节级隔离与协作**：通过隔离关节特征→全局self-attention的设计，使漏检关节能从可靠关节"借用"信息，有效缓解雷达漏检问题
- **恶劣环境下的压倒性优势**：在黑暗和遮挡场景下，mmPred比RGB方案ADE优势超过40%

## 局限与展望

- **依赖预训练姿态估计器**：TPR需要预训练的radar pose estimator和refinement网络，引入了额外的误差源和计算开销
- **仅在室内小规模数据集验证**：mmBody和mm-Fi规模有限，未验证在大规模场景或户外环境的泛化能力
- **雷达硬件依赖**：mmBody使用Phenix雷达、mm-Fi使用低带宽雷达，不同雷达设备间的迁移性未探讨
- **未利用原始雷达信号**：仅基于处理后的点云，原始雷达信号（Range-Doppler图等）可能包含更丰富的运动信息

## 与相关工作的对比

- **PSGSN (Li et al. 2022)**：基于GCN的确定性预测，对噪声历史姿态敏感，mmPred以扩散模型建模运动分布实现更鲁棒的预测
- **HumanMAC (Chen et al. 2023)**：扩散式HMP，假设干净MoCap历史，在雷达噪声下性能退化；mmPred通过FDM提供去噪的历史引导
- **BelFusion (Barquero et al. 2023)**：潜空间多阶段训练，同样假设理想输入，未考虑传感器特有噪声
- **P4Transformer/PointTransformer等**：雷达姿态估计器，逐帧独立处理忽略时序一致性；mmPred的FDM整体处理历史序列
- **milliFlow (Ding et al. 2024)**：雷达关节运动流辅助人体感知，但未推进到运动预测任务

## 启发与关联

- **频域分析作为通用去噪工具**：DCT将运动趋势集中到低频系数的思路可迁移到其他噪声传感器（如WiFi CSI、UWB）的运动理解任务
- **跨模态运动理解**：雷达→姿态→预测的pipeline为多模态运动预测（雷达+IMU、雷达+WiFi）奠定基础
- **关节隔离的特征设计**：GST中将关节特征隔离后再进行全局交互的设计，对其他存在局部失效问题的任务（如遮挡下的姿态估计）有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次定义雷达HMP任务，双域+GST设计针对性强且合理
- 实验充分度: ⭐⭐⭐⭐ — 两个数据集+多种恶劣环境+详细消融，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ — 问题动机阐述清晰，双域互补的直觉通过可视化充分支撑
- 价值: ⭐⭐⭐⭐ — 开辟隐私保护和恶劣环境下运动预测新方向，有明确应用场景

<!-- RELATED:START -->

## 相关论文

- [Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast](radar-aplanc_unsupervised_radar-based_heartbeat_sensing_via_augmented_pseudo-lab.md)
- [SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [FineXtrol: Controllable Motion Generation via Fine-Grained Text](finextrol_controllable_motion_generation_via_fine-grained_text.md)
- [MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](../../CVPR2026/human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)

<!-- RELATED:END -->
