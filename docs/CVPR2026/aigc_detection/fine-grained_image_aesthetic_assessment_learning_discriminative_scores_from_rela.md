---
title: >-
  [论文解读] Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks
description: >-
  [CVPR 2026][细粒度美学] 定义"细粒度图像美学评估"新任务，构建含32,217张图像/10,028个系列的FGAesthetics基准，提出FGAesQ模型：通过差异保留Tokenization（DiffToken）+ 对比文本辅助对齐（CTAlign）+ 排序感知回归（RankReg）从相对排序中学习判别性审美评分，在细粒度场景准确率0.779的同时保持粗粒度SRCC 0.770。
tags:
  - CVPR 2026
  - 细粒度美学
  - 相对排序
  - 差异保留tokenization
  - 排序回归
  - FGAesthetics
---

# Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks

**会议**: CVPR 2026  
**arXiv**: [2603.03907](https://arxiv.org/abs/2603.03907)  
**代码**: [项目页面](https://yzc-ippl.github.io/FG-IAA/)  
**领域**: 图像美学评估  
**关键词**: 细粒度美学, 相对排序, 差异保留tokenization, 排序回归, FGAesthetics

## 一句话总结

定义"细粒度图像美学评估"新任务，构建含32,217张图像/10,028个系列的FGAesthetics基准，提出FGAesQ模型：通过差异保留Tokenization（DiffToken）+ 对比文本辅助对齐（CTAlign）+ 排序感知回归（RankReg）从相对排序中学习判别性审美评分，在细粒度场景准确率0.779的同时保持粗粒度SRCC 0.770。

## 研究背景与动机

**领域现状**：图像美学评估（IAA）广泛应用于内容推荐、AI生成引导和智能摄影。现有数据集（AVA、TAD66K等）评估图像间差异显著的粗粒度美学，深度模型在此场景下已取得不错效果。

**核心痛点**：实际应用中常需从一系列语义相似但美学差异微妙的图像中挑选最佳——例如连拍照片选优、AIGC生成多样本选优、不同裁剪方案比较。现有IAA模型基于绝对评分独立评估，无法有效区分细微差异。具体挑战有二：(1) **语义干扰**——系列内图像语义高度相似，阻碍提取细粒度美学差异，尤其多数深度模型是为语义任务预训练的；(2) **差异微妙**——色彩、构图的细微变化要求模型具备鲁棒的判别性美学表征。

**现有局限**：已有细粒度相关数据集（SPS、Best Frame Selection）因未完全开源而限制了研究发展。现有模型在FGAesthetics上系列级准确率仅30%-50%，远低于粗粒度场景表现。

**切入角度**：不再独立评分，而是利用系列内图像的**相对排序关系**来学习判别性评分——粗粒度数据建立基础审美感知，细粒度数据校准回归空间以区分微妙差异。

## 方法详解

### 整体框架

FGAesQ采用ViT-B/16（CLIP视觉编码器）作为骨干网络，使用两阶段训练策略：先在粗粒度AVA数据集上预训练建立基础审美感知，再在粗粒度+细粒度FGAesthetics数据上交替联合训练。细粒度批次包含系列内配对图像和排序标签，粗粒度批次包含独立图像和绝对评分。三个关键模块——DiffToken、CTAlign、RankReg——分别解决输入表征、特征对齐和评分校准问题。

### 关键设计

1. **差异保留Tokenization (DiffToken)**
    - 功能：在细粒度输入中，系列内图像大部分区域相似，仅少量区域的差异决定美学排序。DiffToken精确定位这些"美学决定性区域"并保留其高分辨率细节
    - 核心思路：对目标图像 $x$ 和参考图像 $y_1$ 分别切分为大尺寸patch，计算对应位置patch间的SSIM相似度 $s_{i,j} = \text{SSIM}(P_{i,j}^x, P_{i,j}^{y_1})$，按相似度排序后选取低于阈值 $\tau = \text{percentile}(s, p)$ 的patch为差异区域集合 $D$。$D$ 内的patch按原始ViT分辨率tokenize以保留细节，其余patch缩放后随机丢弃以满足token数约束
    - 设计动机：LPIPS分析显示系列内图像在64×64 patch级别感知相似度较低→美学差异集中在局部区域→混合分辨率tokenization让模型聚焦关键区域，同时保持全局构图信息

2. **对比文本辅助对齐 (CTAlign)**
    - 功能：利用对比性文本描述引导视觉模型聚焦细粒度美学差异，增强判别能力
    - 核心思路：利用GPT-4o对带排序标签的图像对生成对比性推理描述 $T_1: x \leftarrow y_1$（使用显式对比词汇）。训练时最小化视觉嵌入差与文本嵌入的余弦距离：$\mathcal{L}_{F\_align} = \cos(E_v(x) - E_v(y_1), E_t(T_1))$。推理时仅使用图像编码器
    - 设计动机：文本描述提供了人类理解美学差异的语义锚点——对比性描述帮助视觉表征学习到审美相关方向上的区分

3. **排序感知回归 (RankReg)**
    - 功能：利用排序标签校准绝对评分预测，确保评分排序与人工标注的美学排序一致
    - 核心思路：通过回归头获得绝对评分后，使用Bradley-Terry模型计算配对优越概率 $P_{(x \succ y_1)} = \frac{e^{Score_x}}{e^{Score_x} + e^{Score_{y_1}}}$，收集系列内所有配对概率分布 $\mathbf{P'}$，以ListMLE损失对齐预测与真实排序
    - 设计动机：直接回归绝对分数在细粒度场景下判别力不足——排序约束迫使模型学习到能正确反映细微美学差距的评分间距

### 数据集构建：FGAesthetics

数据来自三类来源确保多样性：**Natural**（连拍照片/视频帧序列）、**AIGC**（同prompt生成的多张图像）、**Cropping**（同源图像的不同裁剪方案）。经过Metrics-MLLMs-Human三级过滤确保系列内图像视觉相似但可区分。排序标签通过10位标注者的成对比较获得，过滤不可区分的模糊样本。最终数据集：32,217张图像，10,028个系列，系列长度2-10。

### 损失函数 / 训练策略

总损失采用交替训练：$\mathcal{L} = \delta \cdot (\lambda \mathcal{L}_{F\_align} + \mathcal{L}_{F\_RR}) + (1 - \delta) \cdot \mathcal{L}_{C\_EMD}$，其中 $\delta$ 为二值交替指示器，$\lambda = 10$。粗粒度使用EMD损失，细粒度使用CTAlign+RankReg损失。动量更新系数分别为0.615和0.8。粗粒度预训练3 epoch，联合训练7 epoch，A800 GPU。

## 实验关键数据

### 主实验：FGAesthetics上IAA方法对比

| 方法 | 参数量 | Natural Pair Acc | Natural s-SRCC | AIGC Pair Acc | Cropping Pair Acc | Cropping s-SRCC |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| NIMA | 54.3M | 0.589 | 0.225 | 0.566 | 0.655 | 0.312 |
| MUSIQ | 78.6M | 0.607 | 0.233 | 0.535 | 0.731 | 0.495 |
| Charm | 85.7M | 0.672 | 0.404 | 0.616 | 0.707 | 0.432 |
| Q-Align | 8.20B | 0.711 | 0.496 | 0.646 | 0.738 | 0.487 |
| MUSIQ (FT) | 78.6M | 0.654 | 0.356 | 0.572 | 0.770 | 0.556 |
| Charm (FT) | 85.7M | 0.723 | 0.474 | 0.620 | 0.755 | 0.517 |
| **FGAesQ (w/o DiffToken)** | 86.3M | 0.773 | 0.664 | 0.688 | 0.764 | 0.537 |
| **FGAesQ (w DiffToken)** | **86.3M** | **0.779** | **0.729** | **0.709** | **0.774** | **0.590** |

### 消融实验：训练策略与模块贡献

| 配置 | 粗粒度SRCC | 粗粒度PLCC | 细粒度Pair | 细粒度Series |
|------|:---:|:---:|:---:|:---:|
| w/o Fine（仅粗粒度） | 0.713 | 0.726 | 0.578 | 0.364 |
| w/o Coarse（仅细粒度） | 0.031 | 0.050 | 0.565 | 0.299 |
| Coarse→Fine顺序训练 | 0.200 | 0.214 | 0.637 | 0.380 |
| w/o DiffToken | 0.751 | 0.760 | 0.666 | 0.423 |
| w/o CTAlign | 0.770 | 0.780 | 0.747 | 0.581 |
| w/o RankReg | 0.769 | 0.781 | 0.742 | 0.571 |
| **FGAesQ（完整）** | **0.770** | **0.781** | **0.753** | **0.600** |

### 关键发现

- FGAesQ仅86.3M参数即全面超越8.2B的Q-Align，Natural系列级SRCC 0.729 vs 0.496→提升47%
- 仅粗粒度训练→细粒度Series仅0.364；仅细粒度训练→粗粒度SRCC崩溃至0.031→两种粒度性质截然不同
- DiffToken贡献最大（Series从0.423→0.600），其次是RankReg和CTAlign
- Fine-tune现有模型在细粒度提升的同时导致粗粒度严重退化（Charm SRCC 0.777→0.470），FGAesQ通过交替训练同时保持两端性能
- 跨数据集泛化（ICAA17K、AADB、TAD66K）上FGAesQ也展现优势，尤其AADB上SRCC 0.562超越VILA的0.548

## 亮点与洞察

- **新任务定义**：正式定义"细粒度IAA"，填补了IAA领域在微妙差异判别上的空白
- **数据集设计精良**：三类图像来源确保多样性，Metrics-MLLMs-Human三级过滤+成对比较标注确保质量
- **DiffToken巧妙设计**：混合分辨率tokenization在不增加token总数的前提下提升关键区域感知，且推理时无参考图仍超越所有现有方法
- **粗细粒度平衡**：交替训练策略避免了fine-tuning导致的粗粒度灾难性遗忘

## 局限与展望

- DiffToken依赖参考图像计算差异区域→推理时需要系列上下文，独立图像评估时退化为常规tokenization
- 对比文本由GPT-4o生成→引入额外成本和可能的偏差
- 仅在ViT-B/16上验证，能否扩展到更大backbone待验证
- FGAesthetics系列长度最长10，更长序列的排序一致性评估未涉及

## 相关工作与启发

- **vs AVA/TAD66K等粗粒度基准**：独立评分+绝对MOS标签→FGAesthetics系列化+排序标签→本质不同的评估范式
- **vs MLLM方法（Q-Align/UNIAA）**：虽参数量大100倍但粗粒度感知能力不足以迁移到细粒度→说明细粒度判别需要专门设计
- **启发**：DiffToken的混合分辨率策略可迁移到其他需要关注细微差异的视觉任务（如医学影像对比、缺陷检测）；排序学习+回归联合训练范式可推广到quality assessment相关领域

## 评分

⭐⭐⭐⭐ (4/5)

综合评价：定义了有实际需求的新任务，数据集构建严谨，方法设计模块化且各有明确动机。DiffToken是亮点设计。但方法整体偏工程组合（DiffToken+CLIP+CTAlign+RankReg），关键创新集中在任务定义和数据集贡献上。

<!-- RELATED:START -->

## 相关论文

- [HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring](../../ACL2025/aigc_detection/haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)
- [BAID: A Benchmark for Bias Assessment of AI Detectors](../../AAAI2026/aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)
- [Learning to Rewrite: Generalized LLM-Generated Text Detection](../../ACL2025/aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)
- [Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration](../../CVPR2025/aigc_detection/enhancing_few-shot_class-incremental_learning_via_training-free_bi-level_modalit.md)
- [PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](../../ICLR2026/aigc_detection/policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)

<!-- RELATED:END -->
