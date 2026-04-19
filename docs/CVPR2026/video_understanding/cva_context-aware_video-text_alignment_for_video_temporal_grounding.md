---
title: >-
  [论文解读] CVA: Context-aware Video-text Alignment for Video Temporal Grounding
description: >-
  [CVPR 2026][视频理解][视频时序定位] 提出 CVA（Context-aware Video-text Alignment）框架，通过 Query-aware Context Diversification（QCD）、Context-invariant Boundary Discrimination（CBD）损失和 Context-enhanced Transformer Encoder（CTE）三个协同组件，解决视频时序定位中的假阴性和背景关联问题，在 QVHighlights 上 R1@0.7 提升约 5 个点。
tags:
  - CVPR 2026
  - 视频理解
  - 视频时序定位
  - 数据增强
  - 对比学习
  - 上下文不变性
  - 视频-文本对齐
---

# CVA: Context-aware Video-text Alignment for Video Temporal Grounding

**会议**: CVPR 2026  
**arXiv**: [2603.24934](https://arxiv.org/abs/2603.24934)  
**代码**: [https://byeol3325.github.io/projects/CVA/](https://byeol3325.github.io/projects/CVA/) (有)  
**领域**: 视频理解 / 时序定位  
**关键词**: 视频时序定位, 数据增强, 对比学习, 上下文不变性, 视频-文本对齐

## 一句话总结
提出 CVA（Context-aware Video-text Alignment）框架，通过 Query-aware Context Diversification（QCD）、Context-invariant Boundary Discrimination（CBD）损失和 Context-enhanced Transformer Encoder（CTE）三个协同组件，解决视频时序定位中的假阴性和背景关联问题，在 QVHighlights 上 R1@0.7 提升约 5 个点。

## 研究背景与动机

**领域现状**：视频时序定位（VTG）旨在根据文本查询定位未剪辑视频中的目标时段，包含视频时刻检索（VMR）和高光检测（HD）两个子任务。近年来基于 DETR 的端到端方法成为主流。

**现有痛点**：(1) 模型倾向学习虚假关联——将文本查询与静态背景过度关联，而非聚焦目标动作/事件；(2) TD-DETR 提出内容混合增强来打断此关联，但替换片段的选择与文本查询无关，可能引入假阴性（替换了与查询语义相关的片段却标为负样本）。

**核心矛盾**：内容混合增强的有效性取决于替换片段的语义——query-agnostic 的混合无法保证替换片段确实与查询无关。

**本文要解决**：如何在多样化上下文的同时避免假阴性？如何使模型在边界处学到对上下文变化鲁棒的表征？

**切入角度**：(1) 基于 CLIP 预计算文本-视频相似度统计，从数据集级别构建 query-aware 的有效替换池；(2) 用对比学习强化时序边界处的上下文不变表征；(3) 用分层 Transformer 捕获多尺度时序上下文。

**核心 idea**：Query-aware 数据增强 + 边界聚焦对比学习 + 分层时序建模 = SOTA 时序定位。

## 方法详解

### 整体框架
CVA 在 DETR-based VTG 框架上增加三个组件：(1) QCD 在训练时生成语义一致的增强样本；(2) CTE 替换标准 Transformer 编码器，捕获多尺度时序上下文；(3) CBD 在两个增强视图之间施加边界对比约束。

### 关键设计

1. **Query-aware Context Diversification (QCD)**：

    - **功能**：用 CLIP 预计算所有视频片段与所有查询之间的余弦相似度矩阵。根据 GT 对和非 GT 对的分布统计，确定有效采样区间 $[\theta_{\min}, \theta_{\max}]$：
    $\theta_{\min} = \text{Percentile}_\alpha(\mathcal{S}_{\text{non}}), \quad \theta_{\max} = \text{Percentile}_\beta(\mathcal{S}_{\text{gt}})$
    - 只从另一视频中相似度在此区间内的片段中采样替换
    - 同时保留 GT 时段及其前后 $p$ 个相邻片段（上下文保持策略）
    - **设计动机**：下界 $\theta_{\min}$ 过滤太不相关的 trivial 负样本（提供不了有意义的学习信号），上界 $\theta_{\max}$ 过滤可能是假阴性的高相似片段。百分位阈值比固定阈值更鲁棒。

2. **Context-enhanced Transformer Encoder (CTE)**：

    - **功能**：$N_b$ 个堆叠块，每块包含：(a) 窗口自注意力处理视频特征（建模局部时序模式）；(b) 全局自注意力处理可学习 query；(c) 双向 cross-attention 在视频和 query 之间交换信息。
    - **核心公式**：最终输出通过分层聚合 + 可学习权重融合：
    $\mathbf{F}_{\text{CTE}} = \omega \cdot \mathbf{F}_v + (1-\omega) \cdot \text{Norm}(\text{MLP}(\text{Concat}_{l=1}^{N_b}(\mathbf{F}^{(l)})))$
    - **设计动机**：标准 Transformer 直接做全局注意力，缺乏对局部时序模式的显式建模。窗口注意力捕获局部依赖，可学习 query 提供全局语义锚点，双向 cross-attention 实现局部-全局信息交换。

3. **Context-invariant Boundary Discrimination (CBD)**：

    - **功能**：给定 QCD 生成的两个增强视图 $\mathbf{V}'_{\text{mix}}$ 和 $\mathbf{V}''_{\text{mix}}$，提取它们在 GT 时段边界处（起/止帧）的特征作为 anchor 和 positive。负样本来自两个来源：(a) 空间临近的背景帧（hard boundary negatives）；(b) 语义最相似的远处背景帧（hard semantic negatives）。
    - **核心公式**：
    $\mathcal{L}_{CBD} = -\frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \log \frac{\exp(s_{p,b})}{\exp(s_{p,b}) + \sum_{\mathbf{z}_n \in \mathcal{Z}^-} \exp(s_{n,b})}$
    - **设计动机**：边界是定位最关键也最容易出错的区域。在不同上下文增强下强制边界表征一致，使模型学到上下文不变的判别性表征。同时使用邻近和远处硬负样本确保时序和语义两个维度的判别性。

### 损失函数 / 训练策略
- $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MR}} + \mathcal{L}_{\text{HD}} + \lambda_{\text{CBD}} \mathcal{L}_{\text{CBD}}$
- MR 损失：$\lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{gIoU}} \mathcal{L}_{\text{gIoU}}$
- HD 损失：$\lambda_{\text{HD}}(\mathcal{L}_{\text{margin}} + \mathcal{L}_{\text{rank}})$
- $\lambda_{\text{L1}}=10$, $\lambda_{\text{gIoU}}=1$, $\lambda_{\text{HD}}=1$, $\lambda_{\text{CBD}}=0.005$
- QCD 参数：$\alpha=10$, $\beta=60$, 替换比例 0.3, 上下文保持窗口 $p=1$
- AdamW 优化器，cosine annealing，batch size 32

## 实验关键数据

### 主实验——QVHighlights test split

| 方法 | R1@0.5↑ | R1@0.7↑ | mAP Avg↑ | HD mAP↑ |
|------|---------|---------|----------|---------|
| Moment-DETR | 52.89 | 33.02 | 30.73 | 35.69 |
| QD-DETR | 62.40 | 44.98 | 39.86 | 38.94 |
| CG-DETR | 65.43 | 48.38 | 42.86 | 40.33 |
| TD-DETR | 64.53 | 50.37 | 46.69 | - |
| CDTR | 65.79 | 49.60 | 44.37 | - |
| **CVA (Ours)** | **70.05** | **55.32** | **47.49** | **44.43** |

提升幅度：R1@0.5 +4.26 (vs CDTR), R1@0.7 +4.95 (vs TD-DETR), HD mAP +4.1 (vs CG-DETR)

### Charades-STA 和 TACoS

| 数据集 | 方法 | R1@0.5↑ | R1@0.7↑ | mIoU↑ |
|--------|------|---------|---------|-------|
| Charades | BAM-DETR (prev best) | 59.95 | 39.38 | 52.33 |
| Charades | **CVA** | **62.61** | **40.78** | **53.35** |
| TACoS | BAM-DETR (prev best) | 41.45 | 26.77 | 39.31 |
| TACoS | **CVA** | **43.21** | **27.73** | **41.07** |

### 消融实验

| 配置 | R1@0.5↑ | R1@0.7↑ | HD mAP↑ | 说明 |
|------|---------|---------|---------|------|
| Baseline (QCD basic) | ~63 | ~48 | ~39 | 基础增强 |
| + QCD (query-aware) | ~68 | ~52 | ~41 | 大幅提升 R1 |
| + QCD + CTE | ~68.5 | ~53.5 | ~43 | 架构增强 |
| + QCD + CTE + CBD | **70.05** | **55.32** | **44.43** | 完整 CVA |

### 关键发现
- R1 指标的大幅提升（~5 points）直接证明了 QCD 减少假阴性的有效性
- CBD 对精确定位贡献最大，特别是 R1@0.7（严格 IoU 阈值下边界判别更重要）
- 三个组件协同效应明显：QCD 提供多样化高质量训练样本，CTE 提供更好的时序建模，CBD 确保边界判别性
- 方法在三个不同特征的 benchmark（QVHighlights/Charades-STA/TACoS）上一致有效

## 亮点与洞察
- **数据中心视角**：不仅改进模型架构，更重视训练数据的质量——QCD 从数据增强角度解决假阴性是关键创新
- **边界聚焦**：CBD 将对比学习精确定向到最关键的边界区域，比全帧对比更高效
- **统计驱动的阈值**：用数据集级别的相似度分布统计（百分位阈值）替代手动设定，具有自适应性
- **双源硬负样本**：同时考虑时序邻近和语义相似两种硬负样本，更全面地增强判别性

## 局限与展望
- QCD 需要预计算 CLIP 相似度矩阵（一次性开销），数据集非常大时可能较慢
- 窗口大小和 query 数量等超参选择未充分讨论
- 仅使用 SlowFast + CLIP 视频特征，更强的视频编码器可能带来更多提升
- 未与最新的 VLM-based 方法对比

## 相关工作与启发
- TD-DETR 识别了背景关联问题并用内容混合解决，CVA 的 QCD 修复了其假阴性缺陷——这是典型的"问题→partial solution→改进"研究链
- 窗口注意力（Swin Transformer 启发）和可学习 query（DETR 启发）的结合在时序任务中很有效
- 边界对比学习的思路可推广到动作检测、视频分割等需要精确时序边界的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ QCD 的 query-aware 增强和 CBD 的边界聚焦对比有创新，CTE 较常规
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 benchmark+完整消融+组件级分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ R1 提升约 5 点是非常显著的，对 VTG 领域有实质推动

<!-- RELATED:START -->

## 相关论文

- [SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)
- [Hear What Matters! Text-conditioned Selective Video-to-Audio Generation](hear_what_matters_text-conditioned_selective_video-to-audio_generation.md)

<!-- RELATED:END -->
