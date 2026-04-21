---
title: >-
  [论文解读] SignRep: Enhancing Self-Supervised Sign Representations
description: >-
  [ICCV 2025][人体理解][手语表征学习] 提出 SignRep，一个可扩展的自监督手语表征学习框架，通过在 Masked Autoencoder 预训练中利用手语骨架先验、特征正则化和对抗式风格无关损失，仅用单一 RGB 模态即超越了复杂的多模态/多分支方法，在手语识别、字典检索和手语翻译三大任务上均取得 SOTA。
tags:
  - ICCV 2025
  - 人体理解
  - 手语表征学习
  - 自监督学习
  - 掩码自编码器
  - 骨架先验
  - 对抗风格损失
  - 字典检索
---

# SignRep: Enhancing Self-Supervised Sign Representations

**会议**: ICCV 2025  
**arXiv**: [2503.08529](https://arxiv.org/abs/2503.08529)  
**领域**: Sign Language Understanding / Self-Supervised Learning  
**关键词**: 手语表征学习, 自监督学习, 掩码自编码器, 骨架先验, 对抗风格损失, 字典检索

## 一句话总结

提出 SignRep，一个可扩展的自监督手语表征学习框架，通过在 Masked Autoencoder 预训练中利用手语骨架先验、特征正则化和对抗式风格无关损失，仅用单一 RGB 模态即超越了复杂的多模态/多分支方法，在手语识别、字典检索和手语翻译三大任务上均取得 SOTA。

## 研究背景与动机

手语是全球数百万人的重要沟通方式，准确理解手语需要模型捕捉手形、动作、身体姿态和面部表情等复杂视觉特征。当前手语理解面临几个关键挑战：

**标注数据稀缺**：各国手语不同，收集标注数据成本高昂，现有数据集通常不超过 2000 个独立手语词

**通用预训练模型域差距大**：常用做法是用 Kinetics 等动作识别数据集预训练再微调，但手语视频的时间动态和精细手势与通用动作差异显著

**多模态/多分支架构过于复杂**：SOTA 方法通常需要 RGB+骨架多模态输入或多分支集成，计算复杂度高

**骨架模型的局限**：基于关键点的方法虽然内存高效，但通常不如 RGB 方法，且关键点容易缺失或误检

**动机**：能否设计一个简单的单模态自监督框架，在预训练阶段利用手语特有的先验知识（如骨架信息），但推理时不依赖关键点，从而兼得骨架先验的信息量和 RGB 方法的性能优势？

## 方法详解

### 整体框架

SignRep 基于 Hiera（分层视觉 Transformer）的 MAE 框架，核心改进在于：
1. 用**手语先验重建**替代像素重建作为预训练目标
2. 引入**特征正则化**（方差+协方差损失）增强表征质量
3. 添加**对抗式风格无关损失**过滤背景和外观干扰
4. 下游任务提出**类概率分布损失**利用检索信息提升识别

### 关键设计 1：手语先验（Sign Priors）

利用手语姿态估计模型提取六类先验作为预训练重建目标：

**关键点先验**：
- **手部关键点** $\mathcal{P}^{\{h,k\}} \in \mathbb{R}^{21 \times 3}$：以手腕为原点归一化的 21 个 3D 关键点，捕捉手形和朝向
- **全身关键点** $\mathcal{P}^{\{b,k\}} \in \mathbb{R}^{61 \times 3}$：61 个 3D 关键点（双手 42 + 身体 19），捕捉手在身体空间中的位置

**关节角度先验**：
- **手部关节角** $\mathcal{P}^{\{h,a\}} \in \mathbb{R}^{41 \times 2}$：41 个手部关节角度（sin/cos 编码）
- **身体关节角** $\mathcal{P}^{\{b,a\}} \in \mathbb{R}^{22 \times 2}$：22 个上身关节角度

**距离先验**：
- **指尖距离** $\mathcal{P}^{\{h,d\}} \in \mathbb{R}^{5 \times 11 \times 3}$：指尖到各关节的距离矩阵
- **手间交互距离** $\mathcal{P}^{\{b,d\}} \in \mathbb{R}^{12 \times 22 \times 3}$：双手与身体各部位的距离

**活动先验** $\mathcal{P}^{\{h,\text{act}\}} \in [0,1]$：判断手是否处于活跃状态（通过手的位置和运动启发式规则）

### 关键设计 2：Sign Decoder（轻量手语解码器）

不同于标准 MAE 的像素重建解码器，SignRep 使用轻量化的手语解码器：
1. 对编码器输出 token 做平均池化得到 $z^{\text{avg}} \in \mathbb{R}^{1 \times D}$
2. 通过 1D 卷积 + GELU + 转置卷积上采样到 $T$ 帧
3. 对每类先验各接一个全连接预测头

**关键**：解码器在下游任务时完全丢弃，推理只需编码器，**不需要关键点提取**。

### 关键设计 3：表征正则化

**方差损失**：鼓励表征在特征空间中分散分布，避免模式坍塌

$$\mathcal{L}_{\text{var}} = \sum_{j=1}^{D} \max(0, 1 - \sigma_j)$$

**协方差损失**：减少特征维度间的冗余相关性

$$\mathcal{L}_{\text{cov}} = \sum_{j \neq k} \mathcal{C}_{j,k}^2$$

### 关键设计 4：对抗式风格无关学习

目标是让编码器学习手语语义特征而过滤背景和外观信息。

**方法**：从同一视频裁剪两个片段 $A_1, A_2$（共享风格），从不同视频取片段 $B$（不同风格）。提取 gram 矩阵风格表征和内容表征，训练判别器区分匹配/不匹配对。然后用对抗损失迫使编码器产生风格无关的表征。

### 损失函数

预训练总损失：

$$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{recon}} + w_{\text{var}} \mathcal{L}_{\text{var}} + w_{\text{cov}} \mathcal{L}_{\text{cov}} + w_{\text{adv}} \mathcal{L}_{\text{adv}}$$

### 字典检索与类概率分布

**检索**：用滑动窗口提取片段表征，以手部活动度加权平均作为最终表征，用余弦相似度匹配。

**类概率分布损失**：从检索相似度矩阵构建类间概率分布 $\phi \in \mathbb{R}^{C \times C}$，在下游识别任务中作为 KL 散度正则项辅助训练。

## 实验关键数据

### 主实验：手语识别（WLASL2000）

| 方法 | 类型 | Top-1 (Instance) | Top-5 (Instance) |
|------|------|:-:|:-:|
| ST-GCN | 骨架 | 34.40 | 66.57 |
| BEST | 骨架 | 46.25 | 79.33 |
| NLA-SLR (3-crop) | 多模态 | 61.26 | 91.77 |
| StepNet (R+F) | 多模态 | 61.17 | 91.94 |
| StepNet | RGB | 56.89 | 88.64 |
| **SignRep** | **RGB** | **61.05** | **90.27** |

**单模态 RGB 方法首次追平多模态集成方法**，比最强单模态 StepNet 高 4.16%。

### NMFs-CSL 中文手语识别

| 方法 | Top-1 | Top-5 |
|------|:-:|:-:|
| StepNet (RGB) | 77.2 | 92.5 |
| NLA-SLR (多模态, 3-crop) | 83.7 | 98.5 |
| **SignRep** | **84.1** | **98.8** |

**超越所有方法（含多模态）**，比最强单模态高 ~7%。

### ASL-Citizen 识别

| 方法 | DCG | MRR | Rec@1 | Rec@5 |
|------|:-:|:-:|:-:|:-:|
| I3D | 79.13 | 73.32 | 63.10 | 86.09 |
| **SignRep** | **90.84** | **88.05** | **81.37** | **96.11** |

**Top-1 超 I3D 基线 18%**。

### 字典检索（无下游训练）

| 特征 | WLASL DCG | WLASL Rec@1 | NMFs-CSL Rec@1 |
|------|:-:|:-:|:-:|
| HieraMAE-Kinetics | 13.21 | 2.08 | 3.96 |
| HieraMAE-YTSL | 14.06 | 2.57 | 7.57 |
| 手部关节角 | 30.61 | 9.42 | 18.13 |
| **SignRep (weighted)** | **57.93** | **29.92** | **63.04** |

检索性能是手部关节角特征的 3 倍以上，证明预训练表征的强大泛化能力。

### 消融实验

| 配置 | WLASL 检索 DCG |
|------|:-:|
| 仅角度先验 + 掩码 | 45.1 |
| 角度+关键点+距离 + 掩码 | 48.5 |
| 全先验 + 掩码 + var/cov | 49.9 |
| **全先验 + 掩码 + var/cov + 对抗** | **50.7** |

去除掩码后检索性能显著下降（48.5 → 46.3），说明掩码学习对鲁棒表征至关重要。

### 手语翻译（作为冻结特征提取器）

| 骨干 | Phoenix14T BLEU-4 | CSL-Daily BLEU-4 |
|------|:-:|:-:|
| DinoV2 (LoRA) | 19.42 | 12.96 |
| **SignRep (frozen)** | **20.38** | **16.33** |

在 CSL-Daily 上 BLEU-4 提升 3.37，且 SignRep 完全冻结，省去了 LoRA 微调的计算开销。

## 亮点与洞察

1. **"预训练时用骨架，推理时不用"的策略极为巧妙**：将骨架信息作为自监督学习的目标而非输入，既获得了手语领域知识又避免了推理时对关键点提取器的依赖
2. **六类先验设计全面**：关键点（空间结构）+ 关节角（手指弯曲）+ 距离矩阵（手间交互）+ 活动检测，覆盖了手语表达的各个维度
3. **对抗式风格无关学习有意义**：手语理解中背景和签名者外观是主要干扰因素，通过对抗训练显式过滤可大幅提升泛化
4. **类概率分布是优雅的检索-识别迁移**：从无监督的检索相似度中提取类间关系，作为软标签辅助有监督识别
5. **实验打击面广**：识别（3 数据集）+ 检索（3 数据集）+ 翻译（3 数据集），且用一个冻结模型搞定，说服力极强

## 局限性

1. 预训练依赖手语专用的姿态估计模型提取先验，低质量关键点可能引入噪声
2. 在 YouTube-SL-25 上预训练，主要覆盖西方手语，对其他手语文化的泛化能力未知
3. 仅关注孤立手语词，对连续手语（句子级别）的建模有限
4. 活动先验的启发式规则较简单，可能无法处理所有边界情况
5. 与 NLA-SLR 在 WLASL2000 上差距不大，需要更多数据集验证优势

## 相关工作

- **有监督手语识别**：I3D、ST-GCN、StepNet、NLA-SLR 等
- **自监督骨架学习**：SignBERT、SignBERT+、BEST、Skeletor
- **手语翻译预训练**：Sign2GPT、SignHiera、GFSLT-VLP
- **视频 MAE**：VideoMAE、Hiera MAE
- **自监督表征正则化**：VICReg、Barlow Twins

## 评分

- **新颖性**: ★★★★☆ — 将骨架先验融入 MAE 预训练的思路新颖，六类先验的系统设计和风格对抗学习有巧思
- **技术深度**: ★★★★☆ — 先验设计细致，检索→识别的类概率分布迁移有理论优雅性
- **实验质量**: ★★★★★ — 三大任务、九个数据集配置、充分消融，单模态超多模态有说服力
- **实用性**: ★★★★★ — 单一 RGB 模型、推理时不需骨架提取、可作冻结特征，高度实用
- **表达清晰度**: ★★★★☆ — 整体清晰，但先验定义部分符号较多，初读有一定门槛

<!-- RELATED:START -->

## 相关论文

- [Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](../../ECCV2024/human_understanding/videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/human_understanding/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](../../ECCV2024/human_understanding/selfsupervised_feature_adaptation_for_3d_industrial_ano.md)
- [Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](../../ECCV2024/human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)

<!-- RELATED:END -->
