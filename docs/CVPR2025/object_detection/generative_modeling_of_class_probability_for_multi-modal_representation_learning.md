---
title: >-
  [论文解读] Generative Modeling of Class Probability for Multi-Modal Representation Learning
description: >-
  [CVPR 2025][目标检测][视频文本检索] CALM 通过类锚点（class anchors）将视频和文本特征映射到统一的概率分布空间，再用跨模态 VAE 建模模态间不确定性，在域内检索（MSR-VTT R@1 50.8%）和跨域检索（MSR-VTT→DiDeMo R@1 41.2%）上均超越 SOTA，仅增加 0.5M 参数。
tags:
  - CVPR 2025
  - 目标检测
  - 视频文本检索
  - 跨模态对齐
  - 类锚点
  - 概率VAE
  - 域外泛化
---

# Generative Modeling of Class Probability for Multi-Modal Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2503.17417](https://arxiv.org/abs/2503.17417)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 视频文本检索、跨模态对齐、类锚点、概率VAE、域外泛化

## 一句话总结

CALM 通过类锚点（class anchors）将视频和文本特征映射到统一的概率分布空间，再用跨模态 VAE 建模模态间不确定性，在域内检索（MSR-VTT R@1 50.8%）和跨域检索（MSR-VTT→DiDeMo R@1 41.2%）上均超越 SOTA，仅增加 0.5M 参数。

## 研究背景与动机

1. **领域现状**：视频-文本检索依赖 CLIP 等预训练模型做跨模态对齐。现有方法（如 T-MASS、DiffusionRet）通过概率嵌入或扩散模型建模对齐关系。
2. **现有痛点**：(1) 视频和文本的信息密度不对称——视频包含大量冗余帧，文本高度抽象，直接对齐容易丢失细粒度语义；(2) 域内训练的模型跨域泛化差——MSR-VTT→DiDeMo 的 R@1 普遍下降 10%+。
3. **核心矛盾**：直接匹配视频/文本嵌入空间中的点（判别式方法）无法捕捉模态间的不确定性和语义多义性——同一个视频可能对应多种文本描述。
4. **本文目标**：用生成式建模（VAE）捕捉跨模态映射中的不确定性，通过共享的类锚点空间对齐概率分布。
5. **切入角度**：用预定义的类标签（如 Charades 157 类动作）作为跨模态共享锚点，视频和文本都映射到这些锚点上的概率分布——分布比点更能表达不确定性。
6. **核心 idea**：类锚点概率化 + 跨模态 VAE（视频分布→潜空间→文本分布）。

## 方法详解

### 整体框架

视频/文本 → CLIP 提取特征 → 与 K=157 个类锚点计算余弦相似度 → Softmax 得到概率分布 $V_p$ 和 $S_p$ → VAE 编码器将 $V_p$ 映射到潜空间 $z \sim \mathcal{N}(\mu, \sigma^2)$ → VAE 解码器从 $z$ 重建 $\hat{S}_p$ → 重建损失 + KL 正则 + 检索/字幕任务损失。

### 关键设计

1. **类锚点概率分布**

    - 功能：将不同模态映射到统一的语义概率空间
    - 核心思路：157 个类标签格式化为 "The content of [label_k]"，通过 CLIP 文本编码器+可学习位置嵌入得到锚点向量。视频/文本特征与锚点计算余弦相似度后 Softmax：$V_p = \text{softmax}(\tau \cdot c^V)$
    - 设计动机：概率分布比点嵌入更能表达"视频同时包含多种语义"的事实；类锚点提供跨模态的语义桥梁

2. **跨模态概率 VAE**

    - 功能：生成式建模视频→文本的概率映射
    - 核心思路：编码器 $q_\phi(z|V_p)$ 将视频概率分布映射到潜空间，解码器 $p_\theta(\hat{S}_p|z)$ 从潜码生成文本概率分布。训练目标为 ELBO：$\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{KL}$，$\alpha=0.1$
    - 设计动机：判别式对齐无法捕捉一对多的映射关系（如同一视频的多种描述），VAE 的潜空间自然建模了这种不确定性

3. **锚点数量与来源的鲁棒性**

    - 功能：证明方法不依赖特定锚点集合
    - 核心思路：从 Charades 157 类替换为 COCO 91 类仍保持 50.3% R@1（仅降 0.5%），说明锚点提供的是语义结构而非特定类别信息
    - 设计动机：避免对特定数据集标签的过度依赖

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{rec} + 0.1 \cdot \mathcal{L}_{KL} + \mathcal{L}_{task}$。潜空间维度 d=256。仅增加 0.5M 参数（152.6M→153.1M），每 batch 额外 0.08s。

## 实验关键数据

### 主实验

| 设定 | 方法 | R@1↑ | R@5↑ | MnR↓ |
|------|------|------|------|------|
| 域内 (MSR-VTT) | T-MASS | 48.9 | 76.3 | 11.7 |
| 域内 (MSR-VTT) | **CALM** | **50.8** | **77.5** | **11.7** |
| 跨域 (→DiDeMo) | T-MASS | 37.3 | 64.8 | 26.3 |
| 跨域 (→DiDeMo) | **CALM** | **41.2** | **66.3** | **16.1** |
| 跨域字幕 (MSVD→MSR-VTT) | CLIP4Caption | 30.5 CIDEr | - | - |
| 跨域字幕 (MSVD→MSR-VTT) | **CALM** | **35.6 CIDEr** | - | - |

### 消融实验

| 生成方式 | MSR-VTT R@1 | DiDeMo R@1 | 说明 |
|---------|-------------|------------|------|
| Baseline | 48.9 | 37.3 | 无概率建模 |
| KL Divergence | 49.5 | 38.8 | 判别式 |
| Cross-Entropy | 50.1 | 38.3 | 判别式 |
| MSE | 48.7 | 37.3 | 无效 |
| **VAE (CALM)** | **50.8** | **41.2** | 生成式最优 |

### 关键发现

- 跨域泛化提升最大（R@1 +3.9, MnR -10.2），证明 VAE 的潜空间学到了更可迁移的跨模态映射
- 仅 0.5M 额外参数和 0.08s/batch 的额外开销——几乎零成本的性能提升
- 锚点数量 157→91 仅掉 0.5%，说明方法对锚点选择鲁棒

## 亮点与洞察

- **概率分布 > 点嵌入**：通过类锚点将特征转化为概率分布后再对齐，规避了点对齐的信息瓶颈
- **生成式 > 判别式在跨域上的优势**：VAE 比 KL divergence / CE 对齐在跨域上高 2.4-2.9 R@1，证明不确定性建模对泛化至关重要
- **极小的额外开销**：0.5M 参数 + 0.08s/batch，是一种近乎免费的即插即用增强

## 局限与展望

- 类锚点需要预定义标签集（如 Charades 157 类），完全无监督的锚点发现更理想
- 跨域场景下仍有明显掉点（50.8→41.2），说明泛化问题未完全解决
- 仅验证了检索和字幕两个任务，视频问答等更复杂任务未测试

## 相关工作与启发

- **vs T-MASS**: 同样用概率嵌入但采用判别式对齐，跨域 R@1 37.3 vs CALM 41.2——生成式建模更适合跨域
- **vs DiffusionRet**: 用扩散模型对齐，MSR-VTT 域内 49.0——CALM 的 VAE 更轻量（0.5M vs 数M参数）

## 评分

- 新颖性: ⭐⭐⭐⭐ 类锚点概率化+VAE的组合有新意
- 实验充分度: ⭐⭐⭐⭐ 域内+跨域+字幕+详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 即插即用的跨域泛化增强方案

<!-- RELATED:START -->

## 相关论文

- [DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)
- [Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [Pippo: High-Resolution Multi-View Humans from a Single Image](pippo_high-resolution_multi-view_humans_from_a_single_image.md)

<!-- RELATED:END -->
