---
title: >-
  [论文解读] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos
description: >-
  [ECCV 2024][视频理解][long video temporal grounding] 提出 RGNet 将长视频时序定位的片段检索和时序定位两个阶段深度统一到单一网络中，通过 RG-Encoder 的稀疏注意力和对比片段采样实现端到端优化，在 MAD 和 Ego4D 上取得 SOTA。
tags:
  - ECCV 2024
  - 视频理解
  - long video temporal grounding
  - clip retrieval
  - unified network
  - 注意力机制
  - moment localization
---

# RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos

**会议**: ECCV 2024  
**arXiv**: [2312.06729](https://arxiv.org/abs/2312.06729)  
**代码**: https://github.com/Tanveer81/RGNet  
**领域**: 视频理解 / 时序定位  
**关键词**: long video temporal grounding, clip retrieval, unified network, sparse attention, moment localization

## 一句话总结
提出 RGNet 将长视频时序定位的片段检索和时序定位两个阶段深度统一到单一网络中，通过 RG-Encoder 的稀疏注意力和对比片段采样实现端到端优化，在 MAD 和 Ego4D 上取得 SOTA。

## 研究背景与动机
**领域现状**：长视频时序定位（LVTG）需要从 20-120 分钟视频中根据文本查询定位特定时刻（通常仅几秒），如同大海捞针。现有方法分为两阶段：先检索相关片段，再在片段内做精确定位。

**现有痛点**：两阶段是割裂的——检索模块通常采用文本-视频检索技术（如 CLIP），仅需理解视频的高层主题，缺乏细粒度事件理解能力。检索失败后定位网络无法挽回。

**核心矛盾**：LVTG 的检索需要细粒度事件理解（"找到妈妈在农舍外晾衣服的时刻"），但通用视频检索模型是为粗粒度主题匹配设计的（"一部关于农场家庭的电影"）。

**本文目标**：如何统一片段检索和时序定位，让检索模块具备细粒度事件理解能力。

**切入角度**：设计统一的 Transformer 编码器同时在 clip 级和帧级建模，通过稀疏注意力和端到端优化让检索直接受益于定位目标。

**核心 idea**：用 RG-Encoder 统一检索和定位，共享特征互相优化，检索模块通过定位标注直接学习细粒度事件理解。

## 方法详解

### 整体框架
输入长视频被滑动窗口切为多个等长片段。RG-Encoder 处理所有片段和文本查询，输出检索到的最相关片段特征。定位解码器在检索到的片段上预测精确的时间边界 $(\tau_c, \tau_w)$。整个网络端到端训练。

### 关键设计

1. **RG-Encoder（统一检索-定位编码器）**

    - 功能：将检索和定位统一到单一编码器，同时在 clip 级（上下文）和帧级（内容）建模
    - 核心组件包含四部分：
        - **Cross Attention**：帧特征作为 query，文本特征作为 key/value，生成文本条件帧特征：$F^i = \text{softmax}(Q^i K^T) V + Q^i$
        - **Sparsifier**：用 Gumbel-softmax 计算每帧的相关性 $G^j \in [0,1]$，分类帧为相关/不相关，生成注意力掩码 $M(j,k) = \begin{cases} 0 & \text{if } G^j > 0.5 \text{ or } j=k \\ -\infty & \text{otherwise} \end{cases}$
        - **Retrieval Attention**：引入可学习检索 token $R^i$ 与帧特征拼接做自注意力，基于稀疏掩码聚合：$\tilde{Q}^i = \text{softmax}(Q^i K^T + M) V + Q^i$。输出 clip 级上下文 $R^i$ 和帧级内容 $F_c^{i,j}$
        - **特征融合**：检索 clip 特征 = 内容 + 上下文 × 相关性：$P^{i,j} = F_c^{i,j} + R^i \times G^j$
    - 设计动机：稀疏注意力让检索关注事件相关帧，而非整个片段，与定位任务更协同

2. **对比负样本片段挖掘（Contrastive Clip Sampling）**

    - 功能：在训练时模拟长视频的片段检索场景
    - 核心思路：同一视频的多个片段构成一个 batch，使用 InfoNCE 损失对比正确片段和负样本片段：
    $\mathcal{L}_{\text{cont}} = -\sum_i \log \frac{\exp(l_{\text{cont}}(R^{i,i}))}{\sum_j \exp(l_{\text{cont}}(R^{i,j}))}$
    - 大 batch 的负样本（来自同一场景的不同片段）模拟真实推理时从长视频中检索的挑战
    - 设计动机：训练和测试阶段的差距是 LVTG 的关键问题。之前方法训练时仅见少量片段，测试时需在数百个片段中检索

3. **Intra-Clip Attention Loss（帧内注意力损失）**

    - 功能：指导稀疏化器区分时间边界内外的帧
    - 核心思路：margin-based ranking loss，要求属于 ground truth 时刻的帧具有更高相关性分数：
    $\mathcal{L}_{\text{attn}} = \max(0, \Delta + S_c(i, j_{\text{out}}) - S_c(i, j_{\text{in}}))$
    - 其中 $S_c(i,j) = R^i_{\text{proj}} \cdot P^{i,j}_{\text{proj}}$，$\Delta=0.2$
    - 设计动机：让检索注意力聚焦于与查询事件对齐的帧

### 损失函数 / 训练策略
- 总损失：$\mathcal{L}_{\text{total}} = \lambda_{\text{attn}} \mathcal{L}_{\text{attn}} + \lambda_{\text{cont}} \mathcal{L}_{\text{cont}} + \mathcal{L}_g$
- 定位损失 $\mathcal{L}_g$：L1 + gIoU + CE（匈牙利算法匹配预测和 GT）
- 超参：$\lambda_{L1}=10, \lambda_{\text{gIoU}}=1, \lambda_{\text{CE}}=4, \lambda_{\text{cont}}=10$
- MAD 训练 35 epoch，Ego4D 训练 200 epoch
- 使用冻结的 CLIP 和 EgoVLP 提取特征
- 片段长度：MAD 180s，Ego4D 48s

## 实验关键数据

### 主实验

| 数据集 | 指标 | RGNet | CONE | SOONet | 提升 |
|--------|------|-------|------|--------|------|
| Ego4D-NLQ | R1@0.3 | **20.63** | 14.15 | 8.00 | +6.48 |
| Ego4D-NLQ | R5@0.3 | **41.67** | 30.33 | 22.40 | +11.34 |
| Ego4D-NLQ | Avg | **24.96** | 17.67 | 11.31 | +7.29 |
| MAD | R1@0.1 | **12.43** | 8.90 | 11.26 | +1.17 |
| MAD | R5@0.1 | **25.12** | 20.51 | 23.21 | +1.91 |
| MAD | Avg | **13.70** | 11.01 | 13.59 | +0.11 |

### 割裂 vs 统一架构比较

| 数据集 | 阶段 | 割裂基线 | RGNet | 提升 |
|--------|------|----------|-------|------|
| Ego4D | 检索 R@1 | 31.71 | **42.08** | +10.37 |
| Ego4D | 检索 R@5 | 64.63 | **76.28** | +11.65 |
| Ego4D | 定位 R1@0.3 | 29.84 | **36.53** | +6.69 |
| MAD | 检索 R@1 | 12.41 | **25.01** | +12.60 |
| MAD | 检索 R@5 | 24.50 | **50.02** | +25.52 |
| MAD | 定位 R1@0.3 | 29.49 | **33.42** | +3.93 |

### 消融实验

| 配置 | R1@0.3 | R5@0.3 | 说明 |
|------|--------|--------|------|
| RGNet (完整) | 18.28 | 34.02 | 默认（无 NaQ） |
| w/o 检索 Token | 17.80 | 33.99 | -0.48，clip 上下文建模重要 |
| w/o 稀疏化器 | 16.12 | 31.57 | -2.16，帧级筛选至关重要 |
| w/o RG-Encoder（割裂基线）| 14.15 | 30.33 | -4.13，统一架构优势明显 |
| w/o 对比损失 | 17.41 | 32.12 | -0.87，负样本模拟有效 |
| w/o 注意力损失 | 16.21 | 31.59 | -2.07，帧级事件区分关键 |

### 关键发现
- 检索阶段是 LVTG 性能瓶颈：oracle 定位 R1@0.3=36.53 远高于 LVTG 的 20.63，差距来自检索错误
- 统一架构在检索 R@1 上提升 10.4%（Ego4D）和 12.6%（MAD），远超预期
- 即使仅检索 1 个片段，RGNet 也超过基线的最优结果
- 稀疏注意力和注意力损失贡献最大，分别下降 2.16 和 2.07

## 亮点与洞察
- **问题分析透彻**：通过系统性地拆解两阶段，实验证明检索是性能瓶颈（而非定位），为统一架构提供了明确动机。oracle 实验的设计清晰有力。
- **统一架构设计优雅**：clip 级上下文特征和帧级内容特征的拆分并融合（$P = F_c + R \times G$），让同一编码器同时服务于检索（需 clip 表示）和定位（需帧表示），设计自然。
- **检索 R@1 翻倍**（MAD：12.41→25.01）说明端到端训练让检索模块学到了之前割裂方法无法学到的细粒度事件理解能力。

## 局限与展望
- 仍依赖预训练图像编码器提取视觉特征（CLIP/EgoVLP），未端到端训练视觉编码器
- 在 MAD 上的提升（Avg +0.11）远小于 Ego4D（Avg +7.29），可能因为电影视频的视觉特征更难区分
- 固定片段长度（180s/48s）可能不适应不同长度的目标时刻
- 解码器仍需多个查询和匈牙利匹配，对超短时刻可能不够精确

## 相关工作与启发
- **vs CONE**：CONE 也是 proposal-based 方法但检索和定位割裂，RGNet 在 Ego4D 上 Avg 提升 7.29%，证明统一优于割裂
- **vs Moment-DETR**：Moment-DETR 为短视频设计，直接用于长视频时时刻过于微小导致性能差，RGNet 通过多粒度建模解决
- **vs X-Pool/TS2-Net**：视频检索方法针对高层主题，RGNet 说明片段检索需要更细粒度的事件理解

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一两阶段的思路清晰，RG-Encoder 设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 两阶段分析、oracle 实验、消融、定性分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导严密，图清晰，实验分析深入
- 价值: ⭐⭐⭐⭐ 长视频时序定位的重要进展，Ego4D R1@0.3 提升 46%

<!-- RELATED:START -->

## 相关论文

- [AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](goldfish_vision-language_understanding_of_arbitrarily_long_videos.md)
- [DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](../../CVPR2025/video_understanding/decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)
- [SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)
- [Object-Shot Enhanced Grounding Network for Egocentric Video](../../CVPR2025/video_understanding/object-shot_enhanced_grounding_network_for_egocentric_video.md)

<!-- RELATED:END -->
