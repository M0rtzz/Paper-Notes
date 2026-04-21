---
title: >-
  [论文解读] Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing
description: >-
  [CVPR 2025][人体理解][人脸活体检测] 提出 OTA 框架：训练阶段学习原型表示编码源域分布，测试阶段通过最优传输(OT)在不访问源模型参数和训练数据的前提下，以 training-free 或轻量训练方式将原型迁移到目标域，同时提出 geodesic mixup 数据增强改善低数据场景的分类器学习。
tags:
  - CVPR 2025
  - 人体理解
  - 人脸活体检测
  - 最优传输
  - 无源域适应
  - 少样本适应
  - 原型学习
---

# Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing

**会议**: CVPR 2025  
**arXiv**: [2503.22984](https://arxiv.org/abs/2503.22984)  
**代码**: 无  
**领域**: 人脸理解  
**关键词**: 人脸活体检测, 最优传输, 无源域适应, 少样本适应, 原型学习

## 一句话总结

提出 OTA 框架：训练阶段学习原型表示编码源域分布，测试阶段通过最优传输(OT)在不访问源模型参数和训练数据的前提下，以 training-free 或轻量训练方式将原型迁移到目标域，同时提出 geodesic mixup 数据增强改善低数据场景的分类器学习。

## 研究背景与动机

**领域现状**：人脸活体检测(FAS)是人脸识别系统的关键安全组件。传统方法通过收集大规模数据集覆盖各种真实用户交互和欺骗尝试，但用户条件的多样性几乎无限。域自适应(DA)需要同时访问源域和目标域数据，域泛化(DG)仅依赖静态参数泛化到下游场景。

**现有痛点**：(1) 法律和隐私限制使得参考数据量有限，且禁止主机方与客户端共享模型参数和训练数据；(2) 为每个客户端微调独立模型不可行，维护成本高；(3) 客户端需要快速定制服务以应对不断变化的使用场景。现有的 SFDA 方法虽然不需要源数据，但不限制对源模型的访问。

**核心矛盾**：服务提供商无法共享模型参数和训练数据（隐私/安全），客户端只有少量目标域标注数据，如何在这种严格约束下实现有效的模型适应？

**本文目标**：构建一个特权系统，允许客户端在测试时使用少量标注样本进行轻量级用户特定定制，无需主机方共享模型参数或训练样本。

**切入角度**：原型可以编码源域分布信息且体积小巧、尊重隐私。最优传输(OT)特别适合少样本场景——利用 Wasserstein 距离有效利用底层特征空间的几何结构，即使数据稀疏也能忠实地对齐源域和目标域分布。

**核心 idea**：训练时学习多中心原型作为源域分布的紧凑表示，测试时通过最优传输将原型变换到目标域。提供两种适应方式：training-free（OT 直接变换原型）和轻量训练（在 geodesic mixup 合成数据上训练轻量分类器）。

## 方法详解

### 整体框架

分为训练阶段和测试适应阶段。训练阶段：在多源域数据上训练特征提取器 $f$ 和多中心原型 $P = \{p^{\text{bona fide}}, p^{\text{spoof}}\} \in \mathbb{R}^{D \times K \times 2}$（K 个子中心）。测试阶段：特征提取器 $f$ 作为黑盒，仅使用原型 $P$ 和少量目标域样本 $\mathbb{D}_t$ 进行适应。提供 training-free 和 lightweight training 两种路径。

### 关键设计

1. **基于原型的训练框架 (Prototype-based Framework)**:

    - 功能：学习紧凑的源域分布表示，同时作为分类器使用
    - 核心思路：为每个类别（真实/欺骗）学习 K 个子中心原型。分类时计算嵌入 $z = f(x)$ 与每组原型的平均余弦相似度。训练损失包括三部分：(1) ArcFace 变体的原型损失 $\mathcal{L}_{proto}$（在角度空间添加 margin $m$）；(2) 粗粒度+细粒度的监督对比损失 $\mathcal{L}_{con}$；(3) 正交正则化 $\mathcal{L}_{orth}$ 防止子中心退化。总损失 $\mathcal{L} = \mathcal{L}_{proto} + \alpha \mathcal{L}_{con}^{coarse} + \beta \mathcal{L}_{con}^{fine} + \eta \mathcal{L}_{orth}$
    - 设计动机：原型既是分类器又是分布代理，可在不暴露模型参数的前提下传递给客户端。多子中心提高表达能力

2. **Training-free 最优传输适应**:

    - 功能：无需任何训练即可将源域原型迁移到目标域
    - 核心思路：构建源域原型(2K 个)与目标域少样本特征($M_t$ 个)之间的最优传输问题。代价矩阵 $M$ 基于余弦距离定义，添加 Laplacian 正则化 $\Omega_\alpha$ 保留数据结构。求解 OT 计划 $\gamma^*$ 后，通过重心投影将每个原型变换到目标域：$p^* = \sum_{j=1}^{M_t} \pi_{i,j} z_{t,j}$。按类别分别进行变换以保持判别能力。变换后的原型 $P^*$ 直接作为分类器
    - 设计动机：OT 理论天然适合离散分布和少样本场景，通过利用特征空间几何结构实现有效对齐

3. **Geodesic Mixup 轻量训练适应**:

    - 功能：生成介于源域和目标域之间的合成训练数据，训练轻量分类器
    - 核心思路：传统 mixup 做逐点线性插值，本文沿 OT 定义的测地线路径生成合成分布。具体地，在 OT 映射的不同插值比例 $t \in [0,1]$ 下采样合成数据 $\mu_t = ((1-t)id + tT)\#\mu_s$，其中 $T$ 是 OT 映射。这些合成数据引导分类器学习域间特征过渡，在适应目标域特性的同时保留源域知识
    - 设计动机：低数据场景下直接训练分类器容易过拟合。沿测地线路径的合成数据比线性插值更好地捕捉了底层特征流形的结构

### 损失函数 / 训练策略

- **训练阶段**：多源域联合训练，ArcFace 变体损失 + 双粒度对比损失 + 正交正则化
- **测试适应**：
    - Training-free：求解正则化 OT 问题 + 重心投影变换原型，无需梯度更新
    - Lightweight training：在 geodesic mixup 合成数据上训练一个轻量 MLP 分类器，特征提取器冻结
- margin $m$ 设置为可学习参数

## 实验关键数据

### 主实验 (Cross-Domain: OCIM 标准协议)

| 方法 | OCI→M | OMI→C | OCM→I | ICM→O | Avg HTER ↓ |
|------|-------|-------|-------|-------|-----------|
| SSDG-R | 7.38 | 10.44 | 11.71 | 15.61 | 11.28 |
| SSAN-R | 6.67 | 10.00 | 8.88 | 13.72 | 9.81 |
| SA-FAS | 5.95 | 8.78 | 6.58 | 10.00 | 7.82 |
| CFPL | 3.09 | 2.56 | 5.43 | 3.33 | 3.60 |
| OTA (zero-shot) | 2.62 | 2.22 | 5.32 | 3.56 | 3.43 |
| OTA (training-free, 5-shot) | 2.38 | 2.67 | — | — | — |
| ViTAF (5-shot) | 3.42 | 1.40 | 3.74 | 7.17 | 3.93 |

### 消融实验

- 子中心数量 K=8 效果最优
- 粗粒度+细粒度对比损失的组合优于单独使用
- Geodesic mixup 优于传统线性 mixup（HTER 改善显著）
- OT 的 Laplacian 正则化对少样本场景至关重要

### 关键发现

- 即使在 zero-shot（不使用目标域数据）下，原型方法已达到接近 SOTA 的性能（平均 HTER 3.43 vs CFPL 的 3.60）
- 5-shot training-free 适应进一步改善性能，无需任何训练
- 相比 HTER 的 19.17% 相对改善和 AUC 的 8.58% 改善，验证了方法有效性
- 该方法在跨攻击设置下同样有效
- 即使只有一个类别的目标域数据（one-class 场景），geodesic mixup 仍能提供有效适应

## 亮点与洞察

- **问题设定极具实际意义**：源模型参数不共享、源数据不可访问、客户端只有少量数据的场景完美对应现实部署需求
- **原型作为分布代理的优雅设计**：紧凑、隐私友好、可作为分类器和分布表示的双重角色
- **OT 理论的自然适配**：OT 在少样本和离散分布场景的理论优势得到充分利用
- **Geodesic Mixup 创新**：沿 OT 测地线路径生成合成数据，比逐点线性插值更好地保留了流形结构
- 提供 training-free 和 lightweight training 两种选择，灵活适配不同需求

## 局限与展望

- 依赖预训练特征提取器的质量，若特征提取器不够泛化，原型迁移效果有限
- 原型数量(K)需要预先设定，不同数据集的最优 K 可能不同
- IPM 假设（逆透视变换中的平坦假设）可能限制某些场景
- 未来可探索无标注目标域数据的适应方案
- 可将该框架扩展到其他安全相关的分类任务

## 相关工作与启发

- **ArcFace / Sub-center ArcFace**：原型训练借鉴了 ArcFace 的 angular margin 思想，并扩展为多子中心
- **SFDA 系列**：区别在于本文严格限制源模型参数不可访问，而多数 SFDA 工作允许完全访问模型
- **OT 在域适应中的应用**：本文创新点在于将 OT 用于原型迁移而非特征对齐
- 启发：对于隐私敏感场景，"轻量代理(原型) + OT 迁移"是比传统域适应更实际的范式

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 技术深度 | 8 |
| 实验充分度 | 8 |
| 写作质量 | 8 |
| 实用价值 | 9 |
| 总评 | 8.2 |

<!-- RELATED:START -->

## 相关论文

- [DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](../../ICCV2025/human_understanding/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](../../CVPR2026/human_understanding/subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)
- [Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](../../ECCV2024/human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)
- [TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](../../ECCV2024/human_understanding/tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](../../AAAI2026/human_understanding/pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)

<!-- RELATED:END -->
