---
title: >-
  [论文解读] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration
description: >-
  [CVPR 2026][模型压缩][自监督学习] 提出 Mine-JEPA，首个面向侧扫声纳（SSS）水雷分类的域内自监督学习流水线——基于 SIGReg 正则化损失、声纳适配增强策略和 ImageNet 初始化，仅用 1,170 张未标注声纳图像预训练即超越了在 17 亿图像上预训练的 DINOv3 基础模型。
tags:
  - CVPR 2026
  - 模型压缩
  - 自监督学习
  - 侧扫声纳
  - 水雷分类
  - 域内预训练
  - SIGReg
---

# MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration

**会议**: CVPR 2026  
**arXiv**: [2604.00383](https://arxiv.org/abs/2604.00383)  
**代码**: 无  
**领域**: 自监督学习 / 水下声纳图像  
**关键词**: 自监督学习, 侧扫声纳, 水雷分类, 域内预训练, SIGReg

## 一句话总结

提出 Mine-JEPA，首个面向侧扫声纳（SSS）水雷分类的域内自监督学习流水线——基于 SIGReg 正则化损失、声纳适配增强策略和 ImageNet 初始化，仅用 1,170 张未标注声纳图像预训练即超越了在 17 亿图像上预训练的 DINOv3 基础模型。

## 研究背景与动机

侧扫声纳（SSS）用于海底勘测和水雷检测，是水雷对抗（MCM）的核心技术。该领域面临三大挑战：

**极端数据稀缺**：公开数据集仅 1,170 张声纳图像（含 668 个标注目标），标注获取极其困难

**巨大的域差距**：声纳图像基于声学回波成像，与 RGB 自然图像在成像机理、纹理统计上截然不同——颜色信息几乎无意义，关键线索来自声学亮/暗区域和海床纹理

**大模型并非万能**：直觉上应该用 DINOv3 等大规模预训练模型进行迁移，但声纳域的特殊性使得这一策略未必有效

核心研究问题：在极端小数据声纳场景下，**精心设计的域内 SSL 能否替代大规模通用基础模型？**

## 方法详解

### 整体框架

Mine-JEPA 遵循三阶段流水线：

**Stage 1（数据准备）**→ 从 1,170 张 SSS 图像中以步长 64 滑窗提取 96×96 patch，得到约 153K 无标注 patch  
**Stage 2（域内SSL预训练）**→ 使用 SIGReg 损失 + SSS 适配增强 + ImageNet-1K 初始化进行 ViT 预训练  
**Stage 3（探针评估）**→ 冻结/微调骨干网络，附加线性/MLP 头进行分类评估

### 关键设计

1. **SIGReg 自监督损失（核心 SSL 目标）**

    - 来自 LeJEPA 框架，无需 teacher-student 架构、动量编码器或大 batch
    - 结合不变性损失和分布正则化损失：$\mathcal{L} = (1-\lambda)\mathcal{L}_{inv} + \lambda\mathcal{L}_{sig}$
    - **不变性损失**：鼓励同一 patch 的不同增强视图收敛到相似表示
     - $\mathcal{L}_{inv} = \frac{1}{NV}\sum_{i}\sum_{v}\|z_{i,v} - \bar{z}_i\|_2^2$
    - **SIGReg 损失**：将嵌入分布正则化为标准正态分布，通过随机投影计算 Epps-Pulley 拟合统计量
     - 防止表示坍缩，无需负样本对或 EMA teacher
     - 复杂度 $O(N)$，适合小数据小 batch 设定
    - 设计动机：在仅 1,170 张源图的极端小数据场景下，需要最简单稳定的 SSL 目标

2. **极低投影维度设计（d=16）**

    - 这不是实现细节而是**有意的结构瓶颈**
    - 在数据有限时，低维投影空间可正则化表示学习、减少过拟合
    - 对比：SimCLR 用 128，VICReg 用 2048
    - 设计动机：作为小数据场景下的隐式正则化手段

3. **SSS 适配增强策略（关键域适配）**

    - **移除的增强**：色调/饱和度 jitter、solarization、灰度转换（对声纳无意义甚至有害）
    - **保留的增强**：水平翻转、随机裁剪缩放（0.5-1.0）、高斯模糊
    - **新增的增强**：垂直翻转、随机旋转（±15°）——反映声纳扫描几何中的方向不变性
    - 使用数据集统计量归一化
    - **实验证明**：使用自然图像增强策略的 F1 低至 0.312（比不预训练的 0.557 还差！），声纳适配增强提升到 0.725

4. **初始化策略**

    - 使用 **ImageNet-1K 预训练权重**初始化 ViT 骨干，投影头随机初始化
    - **不使用 DINOv3 初始化**——实验证明从 DINOv3 出发做域内 SSL 反而**性能劣化 10-13 个百分点**
    - 设计动机：ImageNet 提供有用的低层视觉先验（边缘、纹理），但不像 DINOv3 那样过度特化，留有域适配空间

5. **混合数据组成（Real+Syn）**

    - 除约 153K 真实 patch 外，加入约 256K 合成声纳 patch（灰度），总计约 409K patch
    - 真实和合成 patch 使用各自统计量归一化后拼接训练
    - 正则化式 SSL（SIGReg、VICReg）能从异构数据中获益，而对比式/蒸馏式方法反而退化

### 损失函数 / 训练策略

- SSL 损失：SIGReg = $(1-\lambda)\mathcal{L}_{inv} + \lambda\mathcal{L}_{sig}$，$\lambda=0.1$
- 4 个视图，batch size 1024，AdamW（weight decay 0.05）
- 学习率 $1.4 \times 10^{-3}$，1 epoch warmup + cosine decay，100 epochs
- 评估时使用 4 种探针模式：linear / mlp（冻结骨干）、finetune / ft_mlp（微调骨干）

## 实验关键数据

### 主实验

**3 类分类（BG / MILCO / NOMBO）**

| 方法 | 初始化 | SSL 数据 | macro-F1 | NOMBO F1 | Acc |
|------|--------|---------|----------|----------|-----|
| Random Init | Random | — | 0.557 | 0.439 | 58.3% |
| IN1K Only | IN1K | — | 0.739 | 0.647 | 76.3% |
| DINOv3 | DINOv3 | — | 0.810 | 0.700 | 83.4% |
| SimCLR | IN1K | Real+Syn | 0.801 | 0.693 | 82.6% |
| VICReg | IN1K | Real+Syn | 0.800 | 0.676 | 82.8% |
| BYOL | IN1K | Real+Syn | 0.693 | 0.539 | 72.8% |
| **Mine-JEPA** | **IN1K** | **Real+Syn** | **0.820** | **0.734** | **83.8%** |

**二分类（Mine vs Non-mine）+ 模型规模对比**

| 方法 | 初始化 | 参数量 | 3-class F1 | 2-class F1 | MILCO Recall |
|------|--------|--------|-----------|-----------|-------------|
| DINOv3 | DINOv3 | 21.5M | 0.810 | 0.922 | 88.1% |
| Mine-JEPA (ViT-S) | IN1K | 21.6M | **0.820** | **0.935** | 90.9% |
| Mine-JEPA (ViT-Tiny) | IN1K | **5.5M** | 0.814 | **0.935** | **91.4%** |

### 消融实验

**初始化、增强、数据组成的累积效果**

| 配置 | Init | SSL Data | macro-F1 | Δ |
|------|------|---------|----------|---|
| Random Init（无预训练） | Random | — | 0.557 | — |
| 自然图像增强 SSL | Random | Real* | 0.312 | −24.5%p |
| DINOv3 + SIGReg | DINOv3 | Real | 0.706 | −10.4%p vs DINOv3 |
| DINOv3 + SIGReg | DINOv3 | Real+Syn | 0.677 | −13.3%p vs DINOv3 |
| SSS 增强 SIGReg | Random | Real | 0.725 | 基线 |
| + IN1K init | IN1K | Real | 0.756 | +3.1%p |
| + λ 调优 | IN1K | Real | 0.799 | +4.3%p |
| + Real+Syn 数据 | IN1K | Real+Syn | **0.820** | +2.1%p |

**SSL 方法对比与数据扩展性**

| SSL 方法 | 损失类型 | proj dim | Real only | Real+Syn | Δ |
|---------|---------|---------|----------|---------|---|
| SIGReg | 正则化 | 16 | 0.799 | **0.820** | +2.1%p |
| VICReg | 正则化 | 2048 | 0.774 | 0.800 | +2.6%p |
| SimCLR | 对比 | 128 | 0.806 | 0.801 | −0.5%p |
| BYOL | 蒸馏 | 256 | 0.774 | 0.693 | −8.1%p |

### 关键发现

1. **域内 SSL 超越大规模基础模型**：仅用 1,170 张声纳图预训练的 Mine-JEPA（ViT-S）超过了在 17 亿图像上预训练的 DINOv3
2. **强模型域适配反而退化**：对 DINOv3 继续做域内 SSL 导致性能下降 10-13 个百分点——这挑战了"更强底座一定更好"的直觉
3. **增强策略是域内 SSL 的前提条件**：自然图像增强用于声纳会让 F1 从 0.557 降到 0.312（比不预训练还差）
4. **正则化式 SSL 更稳健**：面对异构合成数据时，SIGReg/VICReg 能获益，而 SimCLR 持平、BYOL 严重退化
5. **ViT-Tiny 极具竞争力**：仅 5.5M 参数（DINOv3 的 1/4）就达到了可比性能，适合 AUV 等资源受限平台

## 亮点与洞察

1. **"小而精"vs"大而全"的深刻洞察**：本文最重要的发现不是 Mine-JEPA 的具体设计，而是证明了在极端域偏移场景下，精心适配的小模型优于未适配的大模型
2. **初始化策略的反直觉结论**：中等强度的 ImageNet 初始化比更强的 DINOv3 更适合作为域内 SSL 的起点——因为过度特化的表示缺乏适配弹性
3. **关于异构数据可用性的积极信号**：RGB 真实 patch + 灰度合成 patch 这种模态异构数据在正则化式 SSL 下仍能有效利用
4. **极简设计哲学**：投影维度 16、单超参 λ、无动量编码器——在小数据场景下，最有效的 SSL 不是最复杂的

## 局限与展望

1. **单一数据集**：仅在一个公开声纳数据集（1,170 张图）上实验，结论的普适性需要更多数据验证
2. **测试集较小**：3 类分类仅 110 个测试样本，方差较大
3. **仅 patch 级分类**：未扩展到滑窗检测或语义分割，实际应用中需要端到端检测
4. **合成数据依赖**：Real+Syn 设定中合成数据来源未详细说明
5. **DINOv3 退化原因未深入分析**：仅提供了一些推测（特征变形、分布偏移），缺乏定量分析

## 相关工作与启发

- **与 LeJEPA 的关系**：Mine-JEPA 直接构建在 LeJEPA/SIGReg 之上，主要贡献在于域适配层面（增强策略、初始化策略、数据组成）
- **医学影像类比**：与医学影像领域"域内 SSL > 通用预训练"的发现一致（如 Models Genesis）
- **启发**：在其他极端域偏移场景（如卫星SAR图像、地震数据波形）中，域内 SSL 可能也优于现成的基础模型
- **对基础模型的警示**：不加思考地对基础模型做继续预训练可能适得其反，需要评估初始化的"可塑性"

## 评分

- **新颖性**: ⭐⭐⭐ — 方法层面是已有技术的适配组合，核心贡献在实验洞察
- **实验充分度**: ⭐⭐⭐⭐ — 消融全面系统，涵盖初始化/增强/数据/方法四个维度
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，结论明确，讨论深入
- **实用价值**: ⭐⭐⭐⭐ — 为数据稀缺海洋视觉任务提供了实用方案

<!-- RELATED:START -->

## 相关论文

- [GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)
- [DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation](../../AAAI2026/model_compression/dos_distilling_observable_softmaps_of_zipfian_prototypes_for_self-supervised_poi.md)
- [AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](../../CVPR2025/model_compression/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [In-Context Learning for Pure Exploration](../../ICLR2026/model_compression/in-context_learning_for_pure_exploration.md)
- [CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)

<!-- RELATED:END -->
