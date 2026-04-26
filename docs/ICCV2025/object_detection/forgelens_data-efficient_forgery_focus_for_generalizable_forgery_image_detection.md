---
title: >-
  [论文解读] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection
description: >-
  [ICCV 2025][目标检测][伪造图像检测] 提出 ForgeLens，一个基于冻结 CLIP-ViT 的特征引导框架，通过轻量级的权重共享引导模块（WSGM）和伪造感知特征集成器（FAFormer），引导冻结预训练网络聚焦伪造特征，仅用 1% 训练数据即达到 SOTA 泛化性能。
tags:
  - ICCV 2025
  - 目标检测
  - 伪造图像检测
  - CLIP-ViT
  - 数据高效
  - 特征引导
  - 泛化能力
---

# ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection

**会议**: ICCV 2025  
**arXiv**: [2408.13697](https://arxiv.org/abs/2408.13697)  
**代码**: [GitHub](https://github.com/Yingjian-Chen/ForgeLens)  
**领域**: 目标检测  
**关键词**: 伪造图像检测, CLIP-ViT, 数据高效, 特征引导, 泛化能力

## 一句话总结

提出 ForgeLens，一个基于冻结 CLIP-ViT 的特征引导框架，通过轻量级的权重共享引导模块（WSGM）和伪造感知特征集成器（FAFormer），引导冻结预训练网络聚焦伪造特征，仅用 1% 训练数据即达到 SOTA 泛化性能。

## 研究背景与动机

随着 GAN 和扩散模型的快速发展，AI 生成的逼真伪造图像对社会安全构成严重威胁。理想的伪造图像检测器应满足两个目标：（1）**高泛化性**——能检测未见过的生成技术产生的图像；（2）**数据高效性**——用最少的训练数据达到最优性能。

现有方法存在明显的局限：

**专用检测方法**（如 FreqNet、F3Net）：在训练集上表现出色，但在未见的生成模型上泛化能力差——因为不同生成模型产生独特的伪影特征，全量训练会过拟合

**冻结网络方法**（如 UniFD）：利用 CLIP 等预训练模型提取通用图像特征 + 线性分类器，泛化性好但精度有限——因为通用特征包含大量与伪造无关的信息

作者通过 t-SNE 可视化直观展示了这一困境：
- ResNet50（全量训练）：在已见数据（ProGAN）上真假图像聚类清晰，但在未见数据上特征混淆
- 冻结 CLIP-ViT：特征按图像类别（汽车、猫等）聚类而非按真假聚类，说明通用特征对伪造检测的区分力不足

**核心动机**：能否在保持冻结预训练权重（确保泛化性）的同时，**引导**网络聚焦于伪造特有的特征（提升精度），并且引入尽量少的可训练参数（确保数据高效性）？

## 方法详解

### 整体框架

ForgeLens 采用纯图像编码器架构（image encoder-only），包含三个步骤：（1）WSGM 引导冻结 CLIP-ViT 聚焦伪造特征；（2）FAFormer 跨阶段整合伪造信息；（3）线性分类器进行最终预测。两阶段训练：先训练 WSGM，再训练 FAFormer。

### 关键设计

1. **权重共享引导模块（WSGM, Weight-Shared Guidance Module）**:

    - 功能：在冻结 ViT 的 MHSA 和 MLP 之间插入轻量级可训练模块，引导通用特征向伪造特有特征转变
    - 核心思路：对于 $n$ 个 ViT block，$m$ 个 WSGM 跨 block 共享，每个 WSGM 负责 $n/m$ 个 block。WSGM 采用瓶颈结构：
    $\text{WSGM}(z) = W_{com} \cdot \text{ReLU}(W_{mid} \cdot \text{ReLU}(W_{exp} \cdot z))$
      其中 $W_{exp} \in \mathbb{R}^{\hat{d} \times d}$（扩展）, $W_{mid} \in \mathbb{R}^{\hat{d} \times \hat{d}}$（中间变换）, $W_{com} \in \mathbb{R}^{d \times \hat{d}}$（压缩）, 瓶颈维度 $\hat{d} \ll d$。
      残差连接：$z_l' = \text{WSGM}_k(z_l) + z_l$
    - 设计动机：(1) 插入在 MHSA 后、MLP 前，对全局特征进行优化后再送入 MLP 处理，引导冻结编码器逐步增强伪造特征聚焦；(2) 权重共享大幅减少参数量；(3) 与 Adapter/LoRA 等微调方法不同，WSGM 是一种**引导机制**而非参数调整

2. **伪造感知特征集成器（FAFormer, Forgery-Aware Feature Integrator）**:

    - 功能：整合 ViT 各阶段的 CLS token，融合浅层细粒度纹理特征和深层语义特征
    - 核心思路：将各阶段输出的 CLS token 与一个新引入的 Focus CLS token 拼接，送入标准 ViT block：
    $c_0 = [CLS_{focus}; CLS_1; CLS_2; \ldots; CLS_N]$
      经过多层自注意力后，Focus CLS token 聚合了所有阶段的伪造相关信息，作为最终分类表示
    - 设计动机：浅层特征捕获细微纹理和伪影（如五官轮廓、头发），深层特征编码语义和结构信息。利用 Transformer 的自注意力机制自然地整合多阶段特征，无需额外嵌入变换（CLS token 天然满足 Transformer 输入要求）

3. **两阶段训练策略**:

    - 功能：确保稳定训练和有效的伪造特征引导
    - 核心思路：第一阶段冻结 CLIP-ViT 仅训练 WSGM；第二阶段冻结所有前述网络，仅训练 FAFormer
    - 设计动机：分阶段训练避免了同时优化多个轻量模块可能导致的不稳定

### 损失函数 / 训练策略

- 使用二元交叉熵损失（Binary Cross-Entropy），区分真假图像
- Adam 优化器，$\beta=(0.9, 0.999)$
- 输入分辨率 224×224，训练时仅使用水平翻转增强
- 训练集仅使用 ProGAN 生成的 4 类图像（car, cat, chair, horse）

## 实验关键数据

### 主实验

| 方法类别 | 方法 | Avg.Acc (%) | Avg.AP (%) | 说明 |
|---------|------|------------|------------|------|
| 专用方法 | FreqNet | 85.09 | 91.95 | 在已见 GAN 上好，扩散模型上差 |
| 预处理方法 | NPR | 87.56 | 92.76 | 利用上采样痕迹 |
| 冻结网络 | UniFD | 81.38 | 90.14 | 基础冻结 CLIP-ViT |
| 冻结网络 | FatFormer | 90.86 | 98.16 | 使用文本+图像编码器 |
| 冻结网络 | C2P-CLIP | 93.79 | 98.66 | SOTA（需两个编码器） |
| **Ours** | **ForgeLens** | **94.99** | **98.83** | 仅图像编码器，更简单 |

ForgeLens 在 19 个生成模型的测试集上平均 Acc 达 94.99%，比基础模型 UniFD 提升 13.61%，比之前 SOTA C2P-CLIP 提升 1.2%，且架构更简单（仅使用图像编码器）。

### 消融实验

| 配置 | Avg.Acc (%) | Avg.AP (%) | 说明 |
|------|------------|------------|------|
| 基础模型（无 WSGM、无 FAFormer） | 81.38 | 90.14 | UniFD 基线 |
| + FAFormer only | 87.89 | 92.26 | 多阶段特征融合 +6.51 |
| + WSGM only | 94.52 | 98.12 | 伪造特征引导 +13.14 |
| **+ WSGM + FAFormer** | **94.99** | **98.83** | 完整 ForgeLens |

WSGM 与微调方法对比：

| 方法 | Avg.Acc (%) | Avg.AP (%) |
|------|------------|------------|
| 无适应 | 81.38 | 90.14 |
| Adapter | 85.49 (+4.11) | 94.29 |
| LoRA | 86.96 (+5.58) | 96.41 |
| **WSGM** | **94.52 (+12.86)** | **98.12** |

WSGM 大幅超越 Adapter 和 LoRA，证明引导机制优于参数微调机制。

### 关键发现

1. **极强的数据高效性**：仅使用 1% 训练数据（约 72 张图像），ForgeLens 即可达到与 100% 数据相当的性能，这对快速应对新兴伪造技术至关重要
2. **WSGM 是核心贡献**：贡献了 13.14% 的 Acc 提升，远超 FAFormer 的 6.51%
3. **CAM 可视化**证实 ForgeLens 成功聚焦于伪造相关区域（如人脸轮廓、头发），而原始 CLIP-ViT 的注意力分散在全图
4. 在高斯噪声、高斯模糊、JPEG 压缩、随机裁剪等扰动下保持鲁棒性

## 亮点与洞察

- **"引导"vs"微调"的哲学差异**：WSGM 不是在调整网络参数（如 LoRA 的低秩更新），而是在引导冻结网络的信息流向——这种设计理念使得泛化保持和精度提升可以兼得
- **数据效率令人印象深刻**：1% 数据即可工作，远超需要大量训练数据的专用方法
- **架构简洁性**：仅使用图像编码器（无需文本编码器），比 FatFormer、C2P-CLIP 等双编码器方法更简单却更有效

## 局限与展望

1. 训练集仅使用 ProGAN（GAN 方法），对未来新型生成方法（如视频生成、3D 生成）的适应能力待验证
2. WSGM 的瓶颈维度 $\hat{d}$ 和共享组数 $m$ 的选择需要消融实验确定
3. 仅验证了二分类（真/假），未探索更细粒度的来源归因任务
4. 在某些扩散模型子集（如 Guided 200 步+CFG）上仍存在误判

## 相关工作与启发

- UniFD 开创了"冻结 CLIP + 线性分类器"的范式，ForgeLens 在此基础上通过引导机制大幅提升精度
- 与 LoRA/Adapter 等 PEFT 方法形成对比，揭示了"引导"和"微调"的本质差异
- FAFormer 利用多阶段 CLS token 的思想可推广到其他需要多层次特征融合的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ WSGM 的引导机制设计新颖，数据高效性表现突出
- 实验充分度: ⭐⭐⭐⭐⭐ 19 个生成模型测试、完整消融、与多种微调方法对比、鲁棒性评估
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，可视化直观有力
- 价值: ⭐⭐⭐⭐⭐ 对 AI 生成图像检测领域有重要实际意义，1% 数据即可部署的特性极具应用价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Semantic Discrepancy-aware Detector for Image Forgery Identification](semantic_discrepancy-aware_detector_for_image_forgery_identification.md)
- [\[CVPR 2025\] Detecting Adversarial Data Using Perturbation Forgery](../../CVPR2025/object_detection/detecting_adversarial_data_using_perturbation_forgery.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](../../NeurIPS2025/object_detection/dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)

<!-- RELATED:END -->
