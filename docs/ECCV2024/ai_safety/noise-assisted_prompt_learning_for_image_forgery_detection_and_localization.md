---
title: >-
  [论文解读] Noise-Assisted Prompt Learning for Image Forgery Detection and Localization
description: >-
  [ECCV 2024][AI安全][图像篡改检测] 本文提出 CLIP-IFDL，一种基于 CLIP 的图像篡改检测与定位模型，通过实例感知的双流提示学习和伪造增强噪声适配器来弥补 CLIP 在篡改检测领域的提示缺失和伪造感知不足问题，将 CLIP 的开放世界泛化能力迁移到篡改检测任务中。
tags:
  - ECCV 2024
  - AI安全
  - 图像篡改检测
  - CLIP
  - 提示学习
  - 噪声适配器
  - 多域融合
---

# Noise-Assisted Prompt Learning for Image Forgery Detection and Localization

**会议**: ECCV 2024  
**作者**: Dong Li, Jiaying Zhu, Xueyang Fu, Xun Guo, Yidi Liu, Gang Yang, Jiawei Liu, Zheng-Jun Zha
**代码**: 无  
**领域**: AI 安全 / 图像篡改检测  
**关键词**: 图像篡改检测, CLIP, 提示学习, 噪声适配器, 多域融合

## 一句话总结

本文提出 CLIP-IFDL，一种基于 CLIP 的图像篡改检测与定位模型，通过实例感知的双流提示学习和伪造增强噪声适配器来弥补 CLIP 在篡改检测领域的提示缺失和伪造感知不足问题，将 CLIP 的开放世界泛化能力迁移到篡改检测任务中。

## 研究背景与动机

**领域现状**：图像篡改检测与定位（Image Forgery Detection and Localization, IFDL）旨在判断图像是否被篡改，并定位篡改区域。随着图像编辑工具（Photoshop）和生成模型（GAN、Diffusion）的普及，IFDL 的重要性日益增加。当前方法主要依赖从头训练的检测网络，泛化性有限。

**现有痛点**：(1) 传统篡改检测方法依赖特定篡改类型（拼接、复制-移动、修复等）的训练数据，对未见过的篡改手法泛化能力差；(2) 近期大规模视觉-语言预训练模型（如 CLIP）展现了强大的开放世界推理能力，但直接用于篡改检测面临两个问题——CLIP 缺乏篡改相关的专用提示（prompt），且其视觉编码器对微妙的篡改痕迹不够敏感（篡改线索通常在噪声域而非语义域）。

**核心矛盾**：CLIP 具有强大的泛化能力（这正是篡改检测需要的），但它学习的是语义级别的视觉-文本对齐，而篡改检测需要捕捉的是像素级别的微妙伪影和噪声不一致性。如何既利用 CLIP 的泛化能力，又赋予它篡改感知能力，是核心挑战。

**本文目标** (1) 如何为 CLIP 设计适合篡改检测的提示？(2) 如何增强 CLIP 视觉编码器对篡改痕迹（噪声异常、边缘不一致等）的感知能力？(3) 如何在保持 CLIP 泛化性的同时使其适应篡改检测任务？

**切入角度**：作者提出从两个层面改造 CLIP：(1) 在语言端，设计可学习的双流提示（正-负样本提示对）并根据每张图像的特征和类别自适应调整；(2) 在视觉端，设计噪声适配器来增强图像编码器的伪造感知能力，通过多域（空间域、频率域、噪声域）特征融合捕捉篡改线索。

**核心 idea**：通过实例感知的双流提示学习让 CLIP 理解"篡改 vs 真实"的语义，并用噪声适配器增强视觉编码器的伪造感知能力，实现兼具泛化性和篡改检测能力的模型。

## 方法详解

### 整体框架

CLIP-IFDL 基于预训练 CLIP 模型，包含两个主要组件：(1) 实例感知双流提示学习（Instance-aware Dual-stream Prompt Learning），处理 CLIP 文本端的提示设计；(2) 伪造增强噪声适配器（Forgery-Enhanced Noise Adapter），增强 CLIP 视觉端的篡改感知能力。输入一张图像后，视觉端提取含噪声信息的特征，文本端生成自适应的真/假提示，然后通过文本-图像相似度计算得到检测结果（图像级别），并通过特征上采样得到定位结果（像素级别）。

### 关键设计

1. **实例感知双流提示学习（Instance-aware Dual-stream Prompt Learning）**:

    - 功能：为 CLIP 生成适合篡改检测的自适应提示，替代手工离散提示
    - 核心思路：不使用固定的文本提示（如"a real photo" / "a forged photo"），而是创建一对可学习的连续提示向量——正提示（对应真实图像）和负提示（对应篡改图像）。这些提示初始化为随机向量，通过训练来学习。关键创新在于"实例感知"：每张输入图像的视觉特征和类别信息会通过一个轻量级网络映射为提示的调制向量，动态调整提示以适应当前图像的特点。同时通过约束正提示与真实图像的相似度高于负提示来更新提示参数
    - 设计动机：固定的离散提示无法覆盖篡改检测的复杂性（不同篡改类型、不同图像内容需要不同的判断依据）。实例感知的可学习提示能自适应地为每张图像生成最优的真/假判别基准

2. **伪造增强噪声适配器（Forgery-Enhanced Noise Adapter）**:

    - 功能：增强 CLIP 视觉编码器对篡改伪影的感知能力
    - 核心思路：在 CLIP 的视觉编码器旁路引入一个轻量级的噪声适配器。该适配器首先从输入图像提取多域特征——空间域（RGB 像素特征）、频率域（经 DCT 或 FFT 变换后的频谱特征）和噪声域（通过高通滤波提取的噪声残差特征）。然后通过多域融合网络将这些特征整合。融合后的特征通过零初始化的线性层（zero linear layers）注入到 CLIP 视觉编码器的中间层。零初始化确保训练初期适配器不会破坏 CLIP 的预训练表示
    - 设计动机：CLIP 的视觉编码器是为语义理解训练的，对像素级别的篡改痕迹（如 JPEG 压缩伪影差异、噪声分布不一致、边缘异常）不敏感。噪声适配器通过引入频率域和噪声域信息，补充了 CLIP 缺乏的低级篡改线索

3. **文本-图像相似度约束（Text-Image Similarity Constraint）**:

    - 功能：通过约束正/负提示与对应图像的相似度来优化提示学习过程
    - 核心思路：对于真实图像，正提示的文本特征应与图像特征有高相似度，负提示应低；对于篡改图像则相反。使用对比损失来最大化正确配对的相似度并最小化错误配对的相似度。这种约束确保了可学习提示能编码出"真实"和"篡改"的语义差异
    - 设计动机：不同于传统的二分类训练，通过相似度约束来优化提示可以保留 CLIP 文本-图像对齐的能力，使模型在判别篡改的同时保持对开放世界的理解能力

### 损失函数 / 训练策略

训练使用多任务损失：(1) **对比损失**：约束文本-图像对的正负配对相似度；(2) **检测分类损失**：二元交叉熵用于图像级别的真/假判别；(3) **分割损失**：像素级别的交叉熵 + Dice loss 用于篡改区域定位。训练策略上，CLIP 的主体参数冻结，只训练可学习提示、噪声适配器和零初始化线性层，参数量增加极少。这种策略保护了 CLIP 的预训练知识，防止微调时的灾难性遗忘。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CLIP-IFDL | 之前SOTA | 说明 |
|--------|------|-----------|---------|------|
| CASIA v2 | AUC ↑ | 领先 | MVSS-Net, SPAN | 拼接+复制移动检测 |
| Columbia | AUC ↑ | 领先 | - | 拼接检测 |
| Coverage | AUC ↑ | 领先 | - | 复制移动检测 |
| NIST16 | F1 ↑ | 领先 | - | 混合篡改类型 |
| 跨数据集泛化 | AUC ↑ | 显著领先 | - | CLIP 泛化能力的体现 |

### 消融实验

| 配置 | AUC (检测) | F1 (定位) | 说明 |
|------|-----------|----------|------|
| Full CLIP-IFDL | 最优 | 最优 | 完整模型 |
| w/o 噪声适配器 | 下降显著 | 下降显著 | 缺乏伪造感知能力 |
| w/o 实例感知 | 中等下降 | 中等下降 | 固定提示不如自适应提示 |
| w/o 双流提示 (单提示) | 有下降 | 有下降 | 正负对比更好 |
| w/o 零初始化 | 下降 | 下降 | 破坏了 CLIP 预训练表示 |
| w/o 频率域特征 | 中等下降 | 中等下降 | 频率域信息对检测重要 |
| w/o 噪声域特征 | 中等下降 | 中等下降 | 噪声残差是关键篡改线索 |

### 关键发现
- 噪声适配器是最关键的组件，去掉后检测和定位性能均大幅下降，说明让 CLIP 感知低级篡改伪影至关重要
- 跨数据集泛化实验中，CLIP-IFDL 远超传统方法，验证了利用 CLIP 泛化能力的有效性
- 零初始化策略对于保护 CLIP 预训练知识非常重要，随机初始化注入点会严重损害性能
- 多域融合（空间+频率+噪声）比任何单域效果都好，说明篡改线索分布在不同域中
- 在 AI 生成图像的检测上也表现出一定泛化能力，虽然不如专门方法但优于传统篡改检测方法

## 亮点与洞察
- **零初始化线性层注入**的设计非常精妙：在训练初期适配器输出为零，不影响 CLIP 原始表示，随着训练逐渐引入篡改感知信息。这种"温柔"的注入方式可以推广到任何需要向预训练模型注入新能力的场景
- **多域特征融合**（RGB + 频率 + 噪声）的思路在篡改检测中非常有效，也可以迁移到其他需要捕捉像素级别异常的任务，如 deepfake 检测、图像质量评估等
- 将 CLIP 的开放世界泛化能力引入篡改检测是一个有前瞻性的方向——随着 AI 生成内容的爆发，需要能泛化到未见篡改类型的检测器

## 局限与展望
- 对于最新的扩散模型生成的高质量伪造图像，噪声域和频率域的线索可能非常微弱，检测难度更大
- 当前方法假设有像素级别的篡改标注用于训练，标注成本高。可以探索弱监督或无监督的篡改定位方式
- 噪声适配器的多域融合增加了推理延迟，在实时检测场景（如社交媒体内容审查）中可能需要优化
- CLIP 冻结的策略虽然保护了泛化性但限制了任务特定的表示学习。可以探索选择性解冻部分层的策略
- 定位精度在篡改边缘区域还不够精细，可以引入更强的边缘感知模块

## 相关工作与启发
- **vs MVSS-Net**: MVSS-Net 使用多尺度多视角特征做篡改检测，但从头训练，泛化性不如基于 CLIP 的方法
- **vs ObjectFormer**: ObjectFormer 利用 Transformer 建模篡改区域与背景的关系，但缺乏预训练模型的泛化优势
- **vs SPAN**: SPAN 通过空间金字塔注意力做篡改定位，但只在空间域工作。CLIP-IFDL 的多域融合覆盖了更多篡改线索
- **vs CoOp/CoCoOp**: CoOp 系列提出了 CLIP 的提示学习方法，CLIP-IFDL 将其扩展为双流形式并加入实例感知，更适合篡改检测这种需要正/负对比的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 CLIP 引入篡改检测的思路新颖，噪声适配器+双流提示的设计有创意
- 实验充分度: ⭐⭐⭐⭐ 多个标准数据集测试，跨数据集泛化实验充分，消融全面
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，方法描述清晰
- 价值: ⭐⭐⭐⭐ 在 AI 生成内容泛滥的时代，具有泛化能力的篡改检测器有很高的实用价值

<!-- RELATED:START -->

## 相关论文

- [Rethinking VLMs for Image Forgery Detection and Localization](../../CVPR2026/ai_safety/rethinking_vlms_for_image_forgery_detection_and_lo.md)
- [Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization](../../NeurIPS2025/ai_safety/forensichub_a_unified_benchmark_codebase_for_all-domain_fake_image_detection_and.md)
- [CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)
- [Towards Multi-modal Transformers in Federated Learning](towards_multimodal_transformers_in_federated_learning.md)

<!-- RELATED:END -->
