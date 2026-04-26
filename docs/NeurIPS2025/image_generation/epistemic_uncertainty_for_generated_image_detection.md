---
title: >-
  [论文解读] Epistemic Uncertainty for Generated Image Detection
description: >-
  [NeurIPS 2025][图像生成][AI生成图像检测] 提出 WePe（Weight Perturbation），通过对预训练视觉大模型（DINOv2）施加权重扰动来估计认识不确定性（epistemic uncertainty），利用自然图像与 AI 生成图像在不确定性空间的差异实现检测，无需训练即可使用。
tags:
  - NeurIPS 2025
  - 图像生成
  - AI生成图像检测
  - 认识不确定性
  - 权重扰动
  - DINOv2
  - 分布外检测
---

# Epistemic Uncertainty for Generated Image Detection

**会议**: NeurIPS 2025  
**arXiv**: [2412.05897](https://arxiv.org/abs/2412.05897)  
**代码**: [tmlr-group/WePe](https://github.com/tmlr-group/WePe)  
**领域**: 图像生成  
**关键词**: AI生成图像检测, 认识不确定性, 权重扰动, DINOv2, 分布外检测

## 一句话总结

提出 WePe（Weight Perturbation），通过对预训练视觉大模型（DINOv2）施加权重扰动来估计认识不确定性（epistemic uncertainty），利用自然图像与 AI 生成图像在不确定性空间的差异实现检测，无需训练即可使用。

## 研究背景与动机

随着 Stable Diffusion、DALL-E、Midjourney 等生成模型的发展，高度逼真的 AI 生成图像带来了 deepfake 等安全威胁。现有检测方法面临几个核心挑战：

- **泛化性差**：基于二分类训练的方法（CNNspot 等）对未见过的生成器泛化不足
- **数据依赖**：需要持续收集最新生成器的图像来更新训练集
- **不可获取性**：当最新生成模型未开源时，难以获取大量生成图像用于训练

关键观察：即使在 Inception v3 这样的简单网络上，先进的生成模型如 ADM 的 FID 仍达到 11.84（远非 0），说明自然图像与生成图像在特征空间中存在显著的分布差异。在更强大的 DINOv2 上，这种差异更加明显。

本文的核心 insight：预训练视觉模型在自然图像上训练，其认识不确定性对自然图像低、对生成图像高——这种不确定性差异可以直接用于检测，无需见过任何生成图像。

## 方法详解

### 整体框架

WePe 将 AI 生成图像检测问题转化为不确定性估计问题。框架分三步：

1. 使用预训练 DINOv2 模型提取图像特征
2. 对模型权重施加随机扰动，获取多次特征预测
3. 通过特征相似度的方差估计认识不确定性，高不确定性即判为 AI 生成

整个流程不需要任何生成图像作为训练数据。

### 关键设计

**贝叶斯框架下的不确定性分析**：认识不确定性反映模型对数据分布的"知识缺失"，可通过参数后验分布来量化。根据 Bernstein-von Mises 定理，样本量 N 趋于无穷时，后验近似为以 MLE 为均值、Fisher 信息矩阵逆为协方差的高斯分布。对于分布内（自然图像）数据，后验方差随训练数据增多而降低。对于分布外（生成图像），Fisher 信息矩阵与测试分布不匹配，导致认识不确定性持续偏高。

**通过权重扰动估计不确定性**：传统方法如 MC Dropout、Deep Ensemble 在大模型上不适用（DINOv2 训练时不使用 dropout；多模型集成计算不可行）。WePe 提出用权重扰动替代：

对 DINOv2 的学生模型参数 theta 施加 n 次扰动得到多组扰动参数，利用学生-教师特征相似度的方差作为不确定性。但教师模型不一定可获取，因此推导出不确定性的上界：u(x) <= 2 - (2/n) * sum_k cos_sim(f(x; theta_k), f(x; theta))。

此上界仅需原始模型参数和扰动后参数，无需教师模型。核心直觉：如果扰动前后特征相似度高（cosine similarity 接近 1），则不确定性低，图像更可能是自然图像。

**扰动敏感性的理论保证**（Theorem 3.2）：定义扰动敏感度为特征映射对参数的 Jacobian 的 Frobenius 范数平方，证明了自然图像的期望敏感度低于生成图像。即自然图像的特征表示对参数扰动更鲁棒，生成图像更敏感。

**扰动策略**：使用 DINOv2 ViT-L/14（24 个 transformer block），仅扰动前 19 个 block（高层扰动会过度破坏自然图像特征）。高斯扰动的方差与每个 block 参数均值成正比，比例为 0.1。

**WePe*（训练增强版）**：当有训练数据时，通过微调放大不确定性差距。损失函数鼓励自然图像的扰动前后特征相似度高、生成图像的相似度低。

### 损失函数 / 训练策略

- **无训练版 WePe**：直接使用预训练 DINOv2，不需要任何额外训练
- **训练版 WePe***：使用对比损失微调 DINOv2 学生模型，放大自然 vs 生成图像的不确定性差距

## 实验关键数据

### 主实验

**ImageNet 数据集上的检测性能**（9 种生成器，AUROC/AP 百分比）：

| 方法 | 类型 | ADM | BigGAN | GigaGAN | StyleGAN-XL | 平均AUROC | 平均AP |
|------|------|-----|--------|---------|-------------|-----------|--------|
| CNNspot | 训练 | 62.25 | 85.71 | 74.85 | 68.41 | 67.04 | 66.78 |
| FatFormer | 训练 | 91.77 | 98.76 | 97.65 | 97.64 | 93.68 | 93.11 |
| DRCT | 训练 | 90.26 | 95.87 | 86.89 | 89.11 | 90.36 | 89.92 |
| **WePe*** | **训练** | **93.89** | **99.85** | **99.03** | **99.52** | **95.57** | **94.33** |
| RIGID | 无训练 | 87.16 | 90.08 | 86.39 | 86.32 | 83.58 | 81.58 |
| **WePe** | **无训练** | **89.79** | **94.24** | **92.15** | **93.86** | **87.99** | **85.04** |

**不同模型的检测效果**：

| 模型 | AUROC | AP |
|------|-------|-----|
| DINOv2: ViT-S/14 | 72.83 | 71.63 |
| DINOv2: ViT-B/14 | 81.82 | 80.64 |
| DINOv2: ViT-L/14 | **87.99** | **85.04** |
| DINOv2: ViT-g/14 | 84.92 | 81.83 |
| CLIP: ViT-L/14 | 84.82 | 84.20 |

### 消融实验

**扰动类型对比**：

| 扰动类型 | AUROC | AP |
|----------|-------|-----|
| Gaussian noise | 87.99 | 85.04 |
| Uniform noise | **89.06** | **86.32** |
| Laplace noise | 87.13 | 84.22 |
| MC Dropout | 81.63 | 79.71 |

- 三种权重扰动方法均优于 MC Dropout
- Uniform noise 略优于 Gaussian noise

**扰动层数**：前 9~20 个 block 均能获得良好性能，方法对层数选择鲁棒

**扰动强度**：方法对扰动噪声水平相当鲁棒，仅在噪声极大或极小时性能下降

### 关键发现

1. 无训练版 WePe 平均 AUROC 87.99%，超越所有无训练方法（RIGID 83.58%）
2. 训练版 WePe* 在 9 种生成器上平均 AUROC 95.57%，全面超越 SOTA
3. DINOv2 优于 CLIP，因为 DINOv2 纯图像自监督训练更聚焦视觉细节
4. WePe 对图像扰动攻击（JPEG 压缩、高斯噪声、高斯模糊）鲁棒，噪声反而增大分布差距
5. ViT-g/14 反而不如 ViT-L/14，可能因过大模型的特征空间过于冗余

## 亮点与洞察

- **范式转换**：将检测问题转化为不确定性估计问题，从"学习区分"变为"感知未知"
- **理论扎实**：从贝叶斯视角出发，给出了扰动敏感性的理论证明（Theorem 3.2）
- **FID 关联**：WePe 的检测性能与生成器的 FID 分数强相关，验证了分布差异假设
- **无训练即可用**：利用预训练模型的固有属性检测，无需收集生成图像
- **实用性强**：代码已开源，方法简单高效

## 局限与展望

- 对扩散模型生成图像（LDM、DiT）的无训练检测性能相对较弱（78.47、77.13 AUROC）
- 依赖特定预训练模型（DINOv2），当生成器学会"模仿"DINOv2 特征空间时可能失效
- ViT-g/14 性能反而下降的原因未深入分析
- 仅使用 cosine similarity 作为特征距离度量，可能遗漏更细粒度的分布差异信息
- 未讨论计算效率：多次权重扰动推理的时间开销

## 相关工作与启发

- **RIGID (He et al., 2024)**：发现自然图像对输入噪声扰动更鲁棒，启发了 WePe 从权重扰动角度切入
- **AEROBLADE**：基于自编码器重建误差的无训练方法，但假设过强
- **NPR (Tan et al., 2024)**：利用相邻像素关系差异检测，但对攻击不鲁棒
- 启发：预训练模型的不确定性特性可能是通用 OOD 检测的有效信号

## 评分

- **创新性**：4/5 - 不确定性视角新颖，但基本思想与 OOD 检测相似
- **实用性**：5/5 - 无训练方法，代码开源，即插即用
- **实验充分度**：5/5 - 4 个 benchmark、9 种生成器、多种消融和攻击测试
- **写作质量**：4/5 - 动机和理论推导清晰，实验详尽

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Is Artificial Intelligence Generated Image Detection a Solved Problem?](is_artificial_intelligence_generated_image_detection_a_solved_problem.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](../../CVPR2025/image_generation/any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](../../CVPR2026/image_generation/diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)

<!-- RELATED:END -->
