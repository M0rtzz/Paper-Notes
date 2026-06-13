---
title: >-
  [论文解读] One-for-More: Continual Diffusion Model for Anomaly Detection
description: >-
  [CVPR 2025][目标检测][持续学习] 提出CDAD框架，通过梯度投影实现扩散模型的稳定持续学习，配合迭代SVD（iSVD）将内存消耗从157GB降至17GB，并设计异常掩码网络增强条件机制，在MVTec和VisA上17/18个设置中取得第一。 异常检测从"一对一"（为每类训练专门模型）发展到"一对多"（单模型检测所…
tags:
  - "CVPR 2025"
  - "目标检测"
  - "持续学习"
  - "扩散模型"
  - "异常检测"
  - "梯度投影"
  - "奇异值分解"
---

# One-for-More: Continual Diffusion Model for Anomaly Detection

**会议**: CVPR 2025  
**arXiv**: [2502.19848](https://arxiv.org/abs/2502.19848)  
**代码**: [https://github.com/FuNz-0/One-for-More](https://github.com/FuNz-0/One-for-More)  
**领域**: 图像生成/异常检测  
**关键词**: 持续学习, 扩散模型, 异常检测, 梯度投影, 奇异值分解

## 一句话总结

提出CDAD框架，通过梯度投影实现扩散模型的稳定持续学习，配合迭代SVD（iSVD）将内存消耗从157GB降至17GB，并设计异常掩码网络增强条件机制，在MVTec和VisA上17/18个设置中取得第一。

## 研究背景与动机

异常检测从"一对一"（为每类训练专门模型）发展到"一对多"（单模型检测所有类），但实际场景中类别增量不可预测，需要模型具备持续学习能力（"one-for-more"范式）。

基于扩散模型的异常检测方法通过image-to-image生成正常图像，用输入输出差异作为异常得分。但在持续学习中面临两大问题：

1. **灾难性遗忘**：在新类上训练后，模型对旧类的重建能力严重退化。如ControlNet在10个基础类上表现优异，但随着新任务增加性能急剧下降
2. **忠实性幻觉**：扩散模型倾向于"过拟合"生成正常图像而非重建异常区域，导致在持续学习后对旧类样本的输出与实际不一致

同时，现有的持续异常检测方法（如DNE、UCAD）依赖不断扩大的记忆库存储旧知识，不可扩展；IUF的语义压缩损失在单步大量类增时仍会大幅遗忘。

## 方法详解

### 整体框架

CDAD基于预训练的VAE和U-Net构建持续扩散模型。测试时，输入图像编码到潜空间经扩散后，由异常掩码网络（AMN）作为条件引导去噪重建，重建结果与原图的特征距离作为异常得分。

### 关键设计

**设计一：基于梯度投影的持续扩散模型（CDM）**

- **功能**：在学习新任务时保护旧任务的特征空间不被破坏
- **核心思路**：对U-Net中每一层，计算新任务的梯度后，将其投影到旧任务输入特征空间的正交补空间中。设旧任务的k-rank列基为 $\hat{U}_i$，则投影梯度为：$\nabla_{W_i}^{orth} = \nabla_{W_i} - \hat{U}_i \hat{U}_i^T \nabla_{W_i}$
- **设计动机**：由于 $X_{pre} \cdot \nabla_W^{orth} \approx 0$，更新后的模型对旧任务的输出近似不变，从根本上消除遗忘。不需要存储旧数据，也无额外推理开销

$$\hat{O}_{pre} = X_{pre} W_i - \eta X_{pre} \nabla_W^{orth} \approx O_{pre}$$

**设计二：迭代奇异值分解（iSVD）**

- **功能**：将计算显著性表示的内存消耗从~157GB降至~17GB
- **核心思路**：利用线性表示的传递性（Lemma 1），将大矩阵 $M$ 分割为 $\{M_1, M_2, ..., M_n\}$，先对 $M_1$ 做SVD得到k-rank近似 $\hat{U}_1$，然后迭代地将 $\hat{U}_i$ 与 $M_{i+1}$ 拼接后再做SVD，最终得到全局显著性表示 $\hat{U}_n$
- **设计动机**：扩散模型的马尔可夫去噪过程产生大量中间特征，传统SVD需要将所有特征一次性加载到内存。iSVD基于线性传递性理论，证明迭代计算与全局计算等价，内存开销与单batch相当

**设计三：异常掩码网络（Anomaly-Masked Network, AMN）**

- **功能**：增强扩散模型的条件机制，使其聚焦于异常区域的重建
- **核心思路**：使用CNN编码局部特征，Transformer编码全局特征。引入邻域掩码自注意力（来自UniAD）和异常掩码损失，屏蔽异常区域的特征，只将正常区域特征作为U-Net的条件输入
- **设计动机**：传统image-to-image扩散模型倾向于复制整个输入而非重建异常区域（"过拟合"问题）。通过掩码机制，强制模型关注异常区域的正常化重建

### 损失函数

主要损失为标准的隐空间扩散损失：

$$\mathcal{L}_{CDM} = \mathbb{E}_{\mathcal{E}(x), \tilde{x}, \epsilon \sim \mathcal{N}(0,I), t} \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(\tilde{x}))\|_2^2$$

其中 $\tilde{x}$ 是经过随机patch扰动的原始图像作为条件输入，异常掩码损失额外监督AMN学会区分正常/异常特征。

## 实验关键数据

### MVTec持续异常检测（10+1+1+1+1设置）

| 方法 | Image AUROC | Pixel AUROC |
|------|------------|------------|
| IUF | 83.2 | 90.1 |
| UniAD+EWC | 80.5 | 88.7 |
| DNE | 85.1 | 91.3 |
| **CDAD (Ours)** | **91.4** | **95.2** |

### iSVD内存消耗对比

| 方法 | 10张图像 | 100张图像 | 1000张图像 |
|------|---------|----------|-----------|
| 传统SVD | 157 GB | 1570 GB | OOM |
| **iSVD (Ours)** | **17 GB** | **17 GB** | **17 GB** |

### 消融实验（MVTec, 5+10设置）

| 组件 | Image AUROC | Pixel AUROC |
|------|------------|------------|
| Baseline (ControlNet) | 75.3 | 85.6 |
| + Gradient Projection | 87.2 | 92.1 |
| + iSVD (替代传统SVD) | 87.0 | 91.9 |
| + AMN | **91.4** | **95.2** |

### 关键发现

1. CDAD在MVTec和VisA的17/18个持续学习设置中排名第一
2. iSVD将内存消耗降低约9倍，性能损失仅~0.2 AUROC，验证了线性传递性理论的有效性
3. 梯度投影单独带来+11.9 Image AUROC的提升，是解决遗忘问题的核心
4. AMN额外贡献+4.2 Image AUROC，证明了增强条件机制对异常检测的重要性

## 亮点与洞察

1. **One-for-more范式的提出**：清晰地将持续异常检测从"一对一""一对多"中区分出来，聚焦于更实际的增量场景
2. **iSVD的理论贡献**：基于线性传递性定理提出的迭代SVD，不仅解决了扩散模型的内存瓶颈，对其他需要大规模SVD的场景也有启发
3. **问题诊断精准**："忠实性幻觉"的发现和分析非常准确，AMN的设计直接针对这一问题

## 局限与展望

1. 每个任务仍需训练多个epoch，持续学习效率可以进一步提升
2. 梯度投影可能在任务数量非常大时导致可用梯度空间过小，限制新任务学习能力
3. 仅验证了工业缺陷检测场景，未扩展到医学影像等其他异常检测领域
4. AMN依赖随机patch扰动模拟异常，真实异常模式更加多样

## 相关工作与启发

- **UniAD**：多类异常检测的先驱，邻域掩码自注意力被AMN借鉴
- **DiAD**：首次将潜在扩散模型引入多类异常检测，但不支持持续学习
- **GPM**：梯度投影方法在分类任务中的持续学习应用，本文将其扩展到生成模型
- 启发：iSVD方法可以推广到其他需要在大模型上做持续学习的场景，如持续扩散生成、持续文本到图像等

## 评分

⭐⭐⭐⭐ — 问题定义清晰（one-for-more），方法设计系统（CDM+iSVD+AMN三位一体），理论与实验兼备。iSVD的理论贡献尤为突出。不足在于应用场景偏窄，且持续学习的任务数仍然有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Odd-One-Out: Anomaly Detection by Comparing with Neighbors](odd-one-out_anomaly_detection_by_comparing_with_neighbors.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [\[CVPR 2025\] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](../../NeurIPS2025/object_detection/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)

</div>

<!-- RELATED:END -->
