---
title: >-
  [论文解读] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning
description: >-
  [CVPR 2026][视频理解][Dense Video Captioning] 提出 ROS-DVC，通过将 DETR-based DVC 框架中的共享 query 分离为独立的 localization query 和 caption query，并设计 Overlap Suppression Loss 惩罚 query 间的时序重叠、Cross-Task Contrastive Alignment 保证跨任务语义一致性，在 YouCook2 和 ActivityNet Captions 上实现了 SOTA 的 captioning 和 localization 性能。
tags:
  - CVPR 2026
  - 视频理解
  - Dense Video Captioning
  - 角色特定查询
  - 重叠抑制损失
  - 对比对齐
  - 概念引导
---

# Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning

**会议**: CVPR 2026  
**arXiv**: [2603.11439](https://arxiv.org/abs/2603.11439)  
**代码**: https://github.com/edwardback/ROS-DVC (有)  
**领域**: 视频理解  
**关键词**: Dense Video Captioning, 角色特定查询, 重叠抑制损失, 对比对齐, 概念引导

## 一句话总结
提出 ROS-DVC，通过将 DETR-based DVC 框架中的共享 query 分离为独立的 localization query 和 caption query，并设计 Overlap Suppression Loss 惩罚 query 间的时序重叠、Cross-Task Contrastive Alignment 保证跨任务语义一致性，在 YouCook2 和 ActivityNet Captions 上实现了 SOTA 的 captioning 和 localization 性能。

## 研究背景与动机
Dense Video Captioning (DVC) 的目标是在长视频中**同时**完成事件时序定位和自然语言描述两个子任务。早期方法采用两阶段 "先定位后描述" 策略，两个模块独立训练，缺乏交互。PDVC 首次将 DETR 架构引入 DVC，用一组可学习 query 并行预测事件段和生成描述，实现端到端联合优化。

**现有 query-based DVC 的两大痛点**：

**多任务干扰**：localization 和 captioning 共享同一组 query，单个 query 同时承担"找边界"和"写描述"两个截然不同的任务。在注意力层面，query 的 attention 既不能精确聚焦于事件边界（localization 需要 broad attend），也不能密集关注关键帧的细粒度语义（captioning 需要 dense attend）——两个优化目标冲突，导致注意力模糊。DDVC 虽然尝试了 query 分解，但只是通过 MLP 从 localization query 派生 caption query，两者注意力分布高度相似，并未真正实现任务分离。

**时序冗余**：多个 query 倾向于捕获重复的时序区间，生成冗余的描述。如图 1(a) 所示，baseline 模型对同一段时间重复检测，产生相同的 caption，严重影响 localization 精度和描述多样性。

**核心矛盾**：query 需要同时服务两个异质任务，但 shared representation space 导致优化方向冲突；同时缺乏对 query 间时序关系的显式约束，使得 overlap 问题无法自动消除。

**本文切入角度**：与其让一个 query 身兼两职，不如让两组独立 query "各司其职"——localization query 专注于 broad temporal context 定位边界，caption query 专注于 key frame 的语义细节。同时通过显式的 loss 设计约束 query 间行为：对比损失保证跨任务一致性，overlap 损失惩罚时序冗余。

**核心 idea**：用角色特定的独立 query 消除 DVC 中的多任务干扰，用 Overlap Suppression Loss 消除时序冗余。

## 方法详解

### 整体框架
ROS-DVC 基于 DETR-style 的并行编码-解码架构：
- **输入**：视频帧序列，经预训练 CLIP ViT-L/14 提取帧级特征
- **Transformer Encoder**：对帧特征进行时序上下文建模
- **Decoder**：两组独立的可学习 query（localization + caption）分别通过 cross-attention 与帧特征交互
- **输出**：四个任务头分别预测事件数量、时序边界、描述文本、事件概念

### 关键设计

1. **Role Specific Query Initialization（角色特定 query 初始化）**:

    - 功能：将传统 DVC 中的单一 query 集合分离为两组独立 query——$\{q_{\text{loc}}^j\}_{j=1}^K$ 和 $\{q_{\text{cap}}^j\}_{j=1}^K$
    - 核心思路：两组 query 从**完全独立的可学习嵌入空间**初始化，各自通过 cross-attention 与编码后的帧特征交互。Localization query 在注意力层面 broadly attend to temporal context 以估计事件边界；caption query 则 densely attend on key frames 以捕捉语义细节。两组 query 共享同一个 decoder，在 cross-attention 中引用相同的 visual location（由 localization query 的 reference point 定义），确保视觉定位的一致性
    - 设计动机：区别于 DDVC 通过 MLP 从 localization query 派生 caption query（会产生依赖、限制注意力多样化），本文的完全独立初始化使两组 query 可以各自优化到最适合其任务的注意力模式，从根本上消除多任务干扰

2. **Cross-Task Contrastive Alignment (CTCA) Loss（跨任务对比对齐）**:

    - 功能：确保分离后的 localization 和 caption query 在语义上保持一致
    - 核心思路：通过 Hungarian 匹配确定与 ground truth 对应的 query 索引集合 $\mathcal{M}$。对每个 $j \in \mathcal{M}$，第 $j$ 个 caption query $\tilde{q}_{\text{cap}}^j$ 和对应的 localization query $\tilde{q}_{\text{loc}}^j$ 构成正样本对，其余 localization query 为负样本：
    $\mathcal{L}_{\text{CTCA}} = -\sum_{j \in \mathcal{M}} \log \frac{\exp(\text{sim}(\tilde{q}_{\text{cap}}^j, \tilde{q}_{\text{loc}}^j)/\tau)}{\sum_{j'} \exp(\text{sim}(\tilde{q}_{\text{cap}}^j, \tilde{q}_{\text{loc}}^{j'})/\tau)}$
      其中 $\text{sim}(\cdot)$ 为余弦相似度，$\tau$ 为温度参数
    - 设计动机：query 分离后，如果不加约束，两组 query 可能在语义上漂移导致不一致（同一事件的 localization 和 caption 不对应）。CTCA 通过非对称对比学习拉近对应 query、推远不对应 query，使 localization query 也获得语义感知能力

3. **Overlap Suppression Loss (OSL)（重叠抑制损失）**:

    - 功能：显式惩罚 query 间的时序重叠，鼓励模型学习不重叠的 distinct 事件区域
    - 核心思路：定义预测区间 $B_i, B_j$ 之间的重叠度 $P_o(i,j) = \text{IoU}(B_i, B_j)$。同时定义 ground-truth 对齐度 $P_g(i,j) = \text{IoU}(B_i, G_j)$，构建自适应权重：
    $\alpha = \gamma \cdot P_g + (1-\gamma) \cdot (1-P_g), \quad \gamma \leq 0.5$
      当预测与 GT 对齐良好（$P_g$ 高）时，$\alpha$ 小，抑制力度弱；当预测与 GT 不对齐时，$\alpha$ 大，强力惩罚重叠。最终损失：
    $\mathcal{L}_{\text{OSL}} = -\alpha \cdot \log(\beta - P_o)$
      $\beta$ 为最大重叠阈值超参
    - 设计动机：不能简单地惩罚所有 overlap（会抑制正确检测紧邻事件的能力），需要根据 GT 对齐程度自适应调节——与 GT 匹配好的预测少惩罚，偏离 GT 的冗余预测重惩罚

4. **Concept Guider（概念引导器）**:

    - 功能：轻量级 MLP 辅助头，输出事件级的 multi-hot 概念向量，增强 caption query 的语义丰富度
    - 核心思路：从训练集 caption 中提取 top-$N_c$ 个名词和动词作为概念词汇，为每个事件构建 multi-hot 标签 $Y^c \in \{0,1\}^{N_c}$。概念头取 caption query 输出 $\tilde{q}_{\text{cap}}$ 输入 MLP + sigmoid 预测概念分布，用交叉熵训练：
    $\hat{y}_c = \text{sigmoid}(\text{MLP}(\tilde{q}_{\text{cap}}))$
      推理时不使用此头，仅在训练时引导 caption query 嵌入高层语义
    - 设计动机：不依赖外部记忆库（如 CM2），而是通过辅助任务隐式地让 caption query 学会编码事件的核心概念，使生成的描述更具体、更有上下文感知能力

### 损失函数 / 训练策略
总损失由标准 DVC 损失和新增三个损失组合：
$$\mathcal{L}_{\text{total}} = \lambda_{\text{giou}}\mathcal{L}_{\text{giou}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{cap}}\mathcal{L}_{\text{cap}} + \lambda_{\text{ec}}\mathcal{L}_{\text{ec}} + \lambda_{\text{CTCA}}\mathcal{L}_{\text{CTCA}} + \lambda_{\text{OSL}}\mathcal{L}_{\text{OSL}} + \lambda_{\text{CG}}\mathcal{L}_{\text{CG}}$$

- 视觉编码：CLIP ViT-L/14，帧采样 1 FPS
- 2 层 deformable transformer decoder，4 级多尺度特征
- YouCook2：K=50 queries/组，帧数 F=200；ActivityNet：K=10，F=100
- OSL 超参：$\gamma=0.25$, $\beta=1.0$；概念词数 $N_c=30$

## 实验关键数据

### 主实验 — Captioning 性能

| 方法 | 预训练 | YouCook2 CIDEr↑ | YouCook2 SODA_c↑ | ActivityNet CIDEr↑ | ActivityNet SODA_c↑ |
|------|--------|-----------------|-------------------|--------------------|--------------------|
| PDVC | ✗ | 29.69 | 4.92 | 29.97 | 5.92 |
| CM2 | ✗ | 31.66 | 5.34 | 33.01 | 6.18 |
| MCCL | ✗ | 36.09 | 5.21 | 34.92 | 6.16 |
| E2DVC | ✗ | 34.26 | 5.39 | 33.63 | 6.13 |
| **ROS-DVC (Ours)** | **✗** | **39.18** | **7.06** | **35.04** | **6.45** |

YouCook2 上 CIDEr 比 MCCL（用外部记忆库）高 3.09，SODA_c 高 1.85；ActivityNet 上 CIDEr 最优（35.04），超越所有 non-pretrained 方法。

### 主实验 — Localization 性能

| 方法 | YouCook2 Rec.↑ | YouCook2 Pre.↑ | YouCook2 F1↑ | ActivityNet Rec.↑ | ActivityNet F1↑ |
|------|---------------|---------------|-------------|-------------------|----------------|
| PDVC | 22.89 | 32.37 | 26.81 | 53.27 | 54.78 |
| E2DVC | 24.36 | 34.75 | 28.64 | 54.67 | 56.14 |
| **ROS-DVC** | **29.34** | **35.26** | **32.03** | **55.35** | **55.50** |

YouCook2 上 F1 比 E2DVC 高 3.39；ActivityNet 上 Recall 和 Precision 接近平衡，说明事件计数器预测的事件数更接近 GT。

### 消融实验

| RSQI | CTCA | OSL | CG | CIDEr↑ | SODA_c↑ | F1↑ | 说明 |
|------|------|-----|-----|--------|---------|-----|------|
| ✗ | ✗ | ✗ | ✗ | 29.69 | 5.39 | 26.81 | Baseline (PDVC) |
| ✓ | ✗ | ✗ | ✗ | 32.33 | 5.43 | 27.00 | 仅分离 query→CIDEr +2.64 |
| ✗ | ✗ | ✓ | ✗ | 33.60 | 6.79 | 31.22 | 仅 OSL→F1 大幅提升 +4.41 |
| ✗ | ✗ | ✗ | ✓ | 31.40 | 5.62 | 27.69 | 仅概念引导→CIDEr +1.71 |
| ✓ | ✓ | ✗ | ✗ | 34.48 | 5.58 | 27.59 | query分离+对比→CIDEr +4.79 |
| ✓ | ✓ | ✓ | ✓ | **39.18** | **7.06** | **32.03** | **完整模型，全指标最优** |

### 关键发现
- **OSL 对 localization 贡献最大**：单独加 OSL 使 F1 从 26.81 跳到 31.22（+4.41），是所有单组件中提升最显著的，直接验证了"query 时序重叠是定位瓶颈"的假设
- **RSQI + CTCA 对 captioning 贡献最大**：query 分离 + 对比对齐使 CIDEr 从 29.69 提到 34.48（+4.79），说明任务解耦确实释放了 captioning 能力
- **四组件缺一不可**：完整模型 CIDEr 39.18 显著超越任何三组件组合，各模块互补而非冗余
- **$\gamma=0.25$ 是 OSL 最优平衡点**：更小的 $\gamma$ 导致 captioning 质量下降，更大的 $\gamma$ 削弱 overlap 抑制效果
- **去掉 $\alpha$（即均匀惩罚所有 overlap）会降低 Precision**：验证了 GT-aware 自适应权重的必要性
- **query 数量 50 为最优**：太少会漏事件，太多会产生冗余提议；50 在 captioning 和 localization 间取得最佳平衡

## 亮点与洞察
- **独立 query 初始化的简单有效性**：不需要额外的 encoder 或复杂的 query 交互机制，仅从不同嵌入空间初始化就能让 localization query 和 caption query 自然学会不同的注意力模式（broad vs dense），设计极为简洁
- **OSL 的 GT-aware 自适应设计**巧妙地回避了"惩罚所有 overlap"的过度约束问题，用 $\alpha$ 权重让模型在"减少冗余"和"保留正确检测"之间找到平衡
- **Concept Guider 不增加推理开销**：仅在训练时作为辅助任务引导 query 学习，推理时移除，是典型的"训练增强 + 零开销"范式，可迁移到其他生成任务

## 局限与展望
- Captioning head 使用 LSTM，在长描述生成能力上弱于 GPT-2/LLM-based 方法（DDVC 使用 GPT-2），如果替换为更强的语言模型可能进一步提升
- 概念词汇 $N_c=30$ 较小且从训练集统计得到，面对 open-vocabulary 场景可能不足
- 仅在 YouCook2（烹饪）和 ActivityNet（人类活动）两个 benchmark 上验证，泛化性到其他视频类型（如 Ego4D、电影理解）未验证
- OSL 基于 temporal IoU，假设事件是连续时间段，不适用于 multi-label 或 hierarchical event 场景
- 论文未报告推理速度，两组 query 的 decoder 计算量理论上是 baseline 的 2 倍

## 相关工作与启发
- **vs PDVC**：PDVC 首次将 DETR 用于 DVC，共享 query；本文在此基础上分离 query + 显式 loss 约束，CIDEr +9.49，F1 +5.22，提升非常显著
- **vs DDVC**：DDVC 用 MLP 从 localization query 派生 caption query，本质仍有依赖；本文的完全独立初始化更彻底，且不依赖 GPT-2
- **vs CM2/MCCL**：这两种方法靠外部记忆库增强 captioning，增加了系统复杂度；本文通过内部的 Concept Guider 和 role-specific query 达到甚至超越其效果，更简洁
- **思路可迁移**：OSL 的 GT-aware overlap 惩罚机制可以直接迁移到 temporal action detection、moment retrieval 等时序定位任务中解决 proposal 冗余问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心idea（query分离+overlap loss）直觉清晰、设计简洁，但每个组件的技术复杂度不高
- 实验充分度: ⭐⭐⭐⭐⭐ 双benchmark + 完整消融（单组件、组合、超参、query数量）+ 定性分析，非常详尽
- 写作质量: ⭐⭐⭐⭐ 动机阐述清楚，图表直观（Fig.1的注意力对比很有说服力），但Related Work稍显冗长
- 价值: ⭐⭐⭐⭐ 为query-based DVC提供了清晰的改进范式，OSL可迁移到其他时序任务
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

<!-- RELATED:END -->
