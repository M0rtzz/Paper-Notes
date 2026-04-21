---
title: >-
  [论文解读] Evaluating Deepfake Detectors in the Wild
description: >-
  [ICML 2025][图像恢复][Deepfake检测] 构建包含50万+高质量deepfake图像的新数据集，通过引入JPEG压缩、降分辨率、图像增强等真实场景增强，系统评估6种开源deepfake检测器，揭示不到一半检测器AUC>60%，最低仅约50%（随机水平）。
tags:
  - ICML 2025
  - 图像恢复
  - Deepfake检测
  - 鲁棒性
  - Face Swapping
  - 真实场景评估
---

# Evaluating Deepfake Detectors in the Wild

**会议**: ICML 2025  
**arXiv**: [2507.21905](https://arxiv.org/abs/2507.21905)  
**代码**: [github.com/SumSubstance/Deepfake-Detectors-in-the-Wild](https://github.com/SumSubstance/Deepfake-Detectors-in-the-Wild)  
**领域**: Deepfake检测 / 模型鲁棒性评估  
**关键词**: Deepfake检测, 鲁棒性, Face Swapping, 真实场景评估

## 一句话总结
构建包含50万+高质量deepfake图像的新数据集，通过引入JPEG压缩、降分辨率、图像增强等真实场景增强，系统评估6种开源deepfake检测器，揭示不到一半检测器AUC>60%，最低仅约50%（随机水平）。

## 研究背景与动机
**领域现状**：随着生成模型（GAN/扩散模型）的发展，高质量人脸伪造已变得极易获取。已有大量deepfake检测器被提出，但它们主要在DFDC、FaceForensics等老旧数据集上评估，这些数据集使用2019年前的生成方法，图像质量低、伪影明显。

**现有痛点**：(1) 现有benchmark不反映真实世界的deepfake质量——DFDC/FaceForensics数据集中的deepfake质量远低于当前生成方法；(2) 检测器在现有benchmark上表现近乎完美，但实际部署效果未知；(3) 欺诈者常使用JPEG压缩、降分辨率、GPEN/CodeFormer增强等后处理来绕过检测。

**本文目标** 在真实世界条件下系统评估开源deepfake检测器的实际能力，揭示当前检测技术与实际需求之间的巨大差距。

**切入角度**：使用当前最流行的零样本face-swapping模型（SimSwap、Inswapper/roop）生成新数据集，结合真实欺诈者常用的后处理手段进行评估。

## 方法详解

### 整体框架
设计一个模拟真实场景的测试流水线：(1) 使用SimSwap和Inswapper两个主流face-swapping模型在三个人脸数据集（CelebA-HQ、LFW、FairFace）上生成50万+高质量deepfake图像；(2) 对生成图像施加4类真实场景增强：JPEG压缩（系数75/50/30/10）、降分辨率（128像素）、GPEN图像增强；(3) 评估6种开源检测器：FaceForensics++（XceptionNet）、MAT、M2TR、RECCE、CADDM、SBI。

### 关键设计
1. **真实场景数据集构建**:
    - 功能：用当前主流的face-swapping模型生成高质量deepfake数据集
    - 核心思路：使用SimSwap（$224\times224$和$512\times512$分辨率）和Inswapper（$128$分辨率）对CelebA-HQ/LFW/FairFace三个数据集进行人脸交换，确保性别、年龄、种族匹配。总计生成超过50万张deepfake图像
    - 设计动机：现有DFDC/FaceForensics数据集使用过时生成方法，伪影明显，不能反映当前deepfake技术水平

2. **真实攻击模拟增强**:
    - 功能：模拟欺诈者绕过检测的常用手段
    - 核心思路：分为人工退化（JPEG压缩、降分辨率至128像素）和人工增强（GPEN面部修复）两类。JPEG压缩改变像素值导致压缩伪影；降分辨率模拟低质量录制；GPEN增强可将低质量生成结果提升至视觉上与真实图像难以区分
    - 设计动机：仅在原始图像上测试不够——实际欺诈场景中，攻击者会对生成图像进行后处理以绕过安全措施

3. **多维度系统评估**:
    - 功能：从多个指标和维度全面评估检测器性能
    - 核心思路：使用ROC-AUC、F1（阈值0.5）、PR-AUC、LogLoss、Accuracy五个指标，分别在三个数据集、两个生成器、四种增强条件下交叉评估6种检测器
    - 设计动机：单一指标和单一条件下的评估容易给出误导性结论

### 损失函数 / 训练策略
本文为评估性工作，不涉及新的训练策略。所有检测器使用其原始公开权重进行推理。

## 实验关键数据

### 主实验（原始数据，无增强）

| 检测器 | SimSwap AUC | SimSwap Acc | Inswapper AUC | Inswapper Acc |
|--------|-------------|-------------|---------------|---------------|
| FF (XceptionNet) | 51.7 | 50.1 | 56.7 | 54.8 |
| MAT | 79.7 | 53.7 | 80.3 | 53.7 |
| M2TR | 55.3 | 53.4 | 53.9 | 52.6 |
| RECCE | 56.7 | 53.5 | 56.1 | 53.6 |
| CADDM | 78.2 | 68.5 | 59.8 | 56.9 |
| SBI | **95.5** | **69.7** | 75.9 | 64.2 |

### 增强后性能变化（Overall AUC）

| 检测器 | 原始(SimSwap) | JPEG(75) | 降分辨率 | GPEN增强 |
|--------|--------------|----------|---------|---------|
| FF | 51.7 | 54.9 | 62.9 | 20.2 |
| MAT | 79.7 | 88.1 | 83.0 | 76.3 |
| SBI | 95.5 | 95.0 | 72.3 | 67.1 |
| CADDM | 78.2 | 78.3 | 72.6 | 58.2 |

### 关键发现
- SBI在SimSwap上AUC最高（95.5），但GPEN增强后骤降至67.1——增强显著削弱检测能力
- FF（XceptionNet）AUC仅51.7，接近随机水平，说明FaceForensics++训练的模型泛化极差
- GPEN增强对所有检测器都造成致命影响，将低质量生成结果提升至检测器难以区分的水平
- 不到一半的检测器在综合评价中AUC>60%

## 亮点与洞察
- 系统性地揭示了当前deepfake检测器在真实世界条件下的脆弱性——在benchmark上近乎完美的性能不代表实际可用
- GPEN/CodeFormer等图像增强工具对deepfake检测构成严重威胁——能将低质量生成结果修复到"以假乱真"
- 公开发布50万+高质量deepfake数据集和完整评估代码，为后续研究提供基础设施
- 分人种/性别/年龄段的FairFace评估揭示了检测器的公平性问题

## 局限与展望
- 仅评估face-swapping类deepfake，未覆盖全脸合成（如StableDiffusion生成人脸）
- 仅使用SimSwap和Inswapper两个生成器，未覆盖DeepFaceLab等faceset-based方法
- 未探索针对增强攻击的防御策略或对抗训练方案
- 只测试了开源检测器，商业检测器（如微软Video Authenticator）未纳入评估

## 相关工作与启发
- **vs FaceForensics++ (ICCV19)**: 提供数据集和XceptionNet基线。本文表明在其上训练的模型泛化极差（AUC≈50%）
- **vs SBI (CVPR22)**: 通过自混合图像训练，在本文新数据集上SimSwap AUC达95.5，是最佳检测器，但对GPEN增强不鲁棒
- **vs CADDM (CVPR23)**: 解决隐式身份泄露问题，SimSwap AUC 78.2但对Inswapper仅59.8，跨生成器泛化差
- 启示：未来检测器应将图像增强和压缩纳入训练数据管线，以提升真实场景鲁棒性

## 评分
- 新颖性: ⭐⭐⭐ 主要是评估性工作，方法创新有限，但评估设计和数据集贡献有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 6个检测器×2个生成器×3个数据集×4种增强，交叉评估非常全面
- 写作质量: ⭐⭐⭐⭐ 问题层次清晰，评估设计合理
- 价值: ⭐⭐⭐⭐ 对deepfake检测社区有警示价值，数据集和代码公开

<!-- RELATED:START -->

## 相关论文

- [HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration](harmonica_harmonizing_training_and_inference_for_better_feature_caching_in_diffu.md)
- [Adaptive Estimation and Learning under Temporal Distribution Shift](adaptive_estimation_and_learning_under_temporal_distribution_shift.md)
- [ε-VAE: Denoising as Visual Decoding](epsilon-vae_denoising_as_visual_decoding.md)
- [TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](../../NeurIPS2025/image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)

<!-- RELATED:END -->
