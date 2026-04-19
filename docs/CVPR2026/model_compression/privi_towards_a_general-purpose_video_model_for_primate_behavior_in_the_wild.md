---
title: >-
  [论文解读] PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild
description: >-
  [CVPR 2026][模型压缩][灵长类行为识别] PriVi 构建了 424 小时的大规模灵长类视频预训练数据集，并通过在 V-JEPA 上进行**领域级预训练**（非目标数据集级别），首次证明了视频模型的领域级预训练可以跨数据集泛化，在四个灵长类行为识别基准上用仅 220K 参数的冻结分类器超越了全量微调的专用模型。
tags:
  - CVPR 2026
  - 模型压缩
  - 灵长类行为识别
  - 自监督预训练
  - V-JEPA
  - 领域级预训练
  - 数据策划管线
---

# PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild

**会议**: CVPR 2026  
**arXiv**: [2511.09675](https://arxiv.org/abs/2511.09675)  
**代码**: https://privi.eckerlab.org (数据+模型+代码)  
**领域**: 视频理解  
**关键词**: 灵长类行为识别, 自监督预训练, V-JEPA, 领域级预训练, 数据策划管线

## 一句话总结
PriVi 构建了 424 小时的大规模灵长类视频预训练数据集，并通过在 V-JEPA 上进行**领域级预训练**（非目标数据集级别），首次证明了视频模型的领域级预训练可以跨数据集泛化，在四个灵长类行为识别基准上用仅 220K 参数的冻结分类器超越了全量微调的专用模型。

## 研究背景与动机

1. **领域现状**：灵长类行为分析对认知科学、进化生物学和保护生态学至关重要。计算机视觉方法有潜力辅助行为分析，但现有方法主要依赖以人为中心的预训练模型（如 Kinetics），且针对单一数据集训练专用模型，泛化能力受限。

2. **现有痛点**：（a）预训练数据以人为中心，对灵长类等非人类动物是"域外"数据；（b）现有方法通常为每个目标数据集单独训练模型（数据集级预训练），无法跨数据集共享；（c）标注稀缺（专家标注昂贵且受限），需要能在少量标签下工作的方法。

3. **核心矛盾**：语言模型已证明领域级预训练（在相似但非目标数据上预训练）可提升下游性能，但视频模型尚无类似成果。核心挑战是缺乏大规模、多样化的灵长类视频数据集和可扩展的数据策划方法。

4. **本文目标**（a）如何在无种子数据集和文本标注的情况下策划大规模灵长类视频数据集；（b）领域级预训练是否对视频模型有效；（c）如何设计轻量分类器在冻结特征上高效识别行为。

5. **切入角度**：从"模型中心"转向"数据中心"，核心假设是——在足够多样化的同领域数据上预训练可以提供跨数据集的通用表示，比在每个小数据集上单独预训练更有效。

6. **核心 idea**：用可扩展的数据策划管线构建灵长类视频大数据集 PriVi，通过 V-JEPA 领域级预训练获得通用灵长类表示，以冻结评估方式在四个行为基准上达到 SOTA。

## 方法详解

### 整体框架
方法分三个阶段：（A）**领域级预训练**——在 PriVi 数据集上继续训练 V-JEPA；（B）可选的**数据集级预训练**——在目标数据集上进一步无监督预训练；（C）**冻结评估**——冻结编码器，仅训练轻量注意力分类器进行行为识别。

### 关键设计

1. **PriVi 数据集策划管线**:

    - 功能：从 YouTube 和研究录像中自动策划 424 小时高质量灵长类视频
    - 核心思路：数据来自两部分——**R&O（研究数据）**共 174 小时，来自 11 个不同研究场景的行为生态学项目；**YouTube 数据**通过搜索灵长类相关播放列表获取 458 小时原始视频。策划管线包括：(1) **切镜头检测**去除频繁跳切的片段；(2) **相关性过滤**——在 CLIP 嵌入上训练 2 层 MLP 分类器（recall 82.8%, precision 90.3%）筛选相关片段；(3) **检测过滤**——用 GroundingDINO 做零样本灵长类检测，丢弃空帧片段；(4) **子采样**——按源数据集比例调整 R&O 中各数据集的贡献。最终得到 720K 个 3 秒视频片段。
    - 设计动机：与依赖高质量种子数据集或文本标注的现有管线不同，该管线既不需要种子数据集也不需要文本标注，仅需少量人工标注（2500 张图片）训练相关性分类器，可扩展性强。

2. **V-JEPA 领域级预训练**:

    - 功能：学习灵长类特异性视频表示
    - 核心思路：基于预训练的 V-JEPA ViT-L 模型（在 VideoMix2M 上训练过），在 PriVi 上继续训练 75K steps（约 8 个 epoch），batch size 80，学习率 $1.5 \times 10^{-5}$ 恒定（因原始权重已余弦退火）。训练时对视频做随机裁剪，裁剪中心对齐到零样本灵长类检测器预测的包围框，使得模型聚焦于灵长类个体而非背景。
    - 设计动机：以灵长类为中心裁剪避免模型花大量计算学习背景表示；4 个 A100 训练 11 小时，计算开销极小。

3. **注意力分类器 (Attentive Classifier)**:

    - 功能：在冻结编码器特征上进行轻量行为识别
    - 核心思路：先将编码器输出的 $N$ 个 patch token 从 $D=1024$ 降维到 $D'=64$，然后拼接 $C$ 个可学习的 CLS token（每个类别一个），通过 3 层 self-attention 处理，最后每个 CLS token 经线性投影+softmax/sigmoid 得到类别概率。总参数量仅 220K。
    - 设计动机：现有的冻结评估分类器（V-JEPA 用 12M 参数、V-JEPA2 用 49M 参数）对动物行为的小数据集过参数化。通过降维到 64 维 + 每个类别独立 CLS token 避免信息瓶颈，在极少参数下防止过拟合。

### 损失函数 / 训练策略

预训练使用 V-JEPA 的 masked prediction loss $L_{JEPA} = \|P(E(\text{Mask}(X))) - \text{Mask}^C(\bar{E}(X))\|_1$。分类器使用交叉熵损失（BaboonLand 使用 EQL loss）。分类器在不同数据集上训练 1-40 个 epoch 不等。

## 实验关键数据

### 主实验

| 方法 | ChimpACT mAP | PanAf500 B-Acc | BaboonLand B-Acc | ChimpBehave B-Acc |
|------|-------------|---------------|-----------------|------------------|
| X3D | 27.05 | 50.35 | 31.41 | 62.8 |
| VideoMAEv2 | - | - | - | 74.8 |
| VideoPrism-g | 31.5 | - | - | - |
| V-JEPA (人类数据) | 36.33 | 56.69 | 26.99 | 68.41 |
| **PriVi** | **39.25** | **62.75** | **33.99** | **71.30** |
| **PriVi + DaLP** | **40.00** | **62.96** | **38.57** | **75.14** |

PriVi 在所有四个数据集上超越现有方法，包括全量微调的专用模型（ChimpVLM 167M 参数）。

### 消融实验

| 配置 | ChimpACT mAP | PanAf500 B-Acc |
|------|-------------|---------------|
| V-JEPA 基线 | 32.00 | 71.95 |
| + DaLP: ChimpACT | 35.86 | - |
| YT-Random (未过滤) | 33.87 | 71.61 |
| YT-Filtered (过滤后) | 37.88 | 76.33 |
| R&O 仅 | 33.01 | 73.85 |
| **PriVi (YT-F + R&O)** | **38.75** | **79.95** |
| 无灵长类裁剪 | 32.62 | 72.93 |
| 无降维 (37.84M 参数) | 30.15 | 62.52 |

### 关键发现
- **YouTube 相关性过滤至关重要**：YT-Filtered 比 YT-Random 性能大幅提升，验证数据策划的关键性
- **域级预训练 > 数据集级预训练**：PriVi 在两个数据集上都优于仅在目标数据集上预训练，且无跨数据集负迁移
- **灵长类中心裁剪很重要**：去掉后 ChimpACT mAP 下降 6.13 点
- **降维反而帮助**：37.84M 参数的无降维模型反而性能最差（30.15 mAP），说明小数据集上参数少更好
- **极少标签下表现突出**：仅 10% 训练数据时，PriVi 方法仅损失 4.4% 准确率（PanAf500），仍超越 X3D 全量训练
- 两部分数据（YT-Filtered 和 R&O）各有贡献，合并后进一步提升

## 亮点与洞察
- **首次证明视频模型的领域级预训练有效**：此前只在语言模型上验证过。这意味着可以为各细分领域（海洋生物、农业等）构建类似的预训练数据集
- **极轻量的冻结分类器**：仅 220K 参数，超越 167M 参数的全量微调模型，证明好的表示比大的分类头更重要
- **数据策划管线的实用性**：不需要种子数据集和文本标注，仅需 2500 张标注图片，可轻松迁移到其他动物或领域
- **降维到 64 维的设计**：反直觉但有效——对于小数据集，减少分类头参数量是防过拟合的关键

## 局限与展望
- PriVi 仅覆盖 11 个研究场景，灵长类多样性有限（主要是猕猴和黑猩猩）
- 评估仅在黑猩猩和狒狒数据集上进行，其他灵长类未验证
- 3 秒片段对于长时序行为（如社交互动序列）可能不够
- 未探索视觉-语言模型（如 CLIP 微调）作为替代方案
- R&O 中有 14% 数据因隐私原因未公开

## 相关工作与启发
- **vs VideoPrism**: VideoPrism-g 在 ChimpACT 上仅 31.5 mAP，PriVi 达到 40.0。说明领域专属数据比通用大规模数据更有效
- **vs ChimpVLM**: 全量微调 167M 参数的 VLM，在 PanAf500 上 B-Acc 61.94；PriVi 用 220K 冻结分类器达到 62.96。说明好的预训练数据 > 大模型微调
- **vs AlphaChimp**: 在无 GT 检测的 ChimpACT 上，SAM3 + PriVi（30.76 mAP）超越 AlphaChimp（25.35），说明零样本检测+冻结特征的组合很有实用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次验证视频领域级预训练的有效性，数据策划管线设计实用
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、分类器消融、数据消融、低标签实验、跨镜头泛化，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验充分，但部分符号和公式可更简洁
- 价值: ⭐⭐⭐⭐ 对动物行为分析领域影响大，方法论可推广到其他专业视觉领域

<!-- RELATED:START -->

## 相关论文

- [Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [Einstein Fields: A Neural Perspective To Computational General Relativity](../../ICLR2026/model_compression/einstein_fields_a_neural_perspective_to_computational_general_relativity.md)
- [Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](../../ACL2025/model_compression/direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)
- [F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](textf2texthdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_m.md)

<!-- RELATED:END -->
