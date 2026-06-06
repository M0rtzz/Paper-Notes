---
title: >-
  [论文解读] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning
description: >-
  [CVPR 2026][视频理解][弱监督密集视频描述] 提出 SAIL，通过跨模态相似度引导的语义感知掩码生成和 LLM 合成字幕的辅助监督，在仅有字幕标注（无时间边界）的弱监督设置下，在 ActivityNet 和 YouCook2 上实现密集视频描述和事件定位的双 SOTA。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "弱监督密集视频描述"
  - "跨模态对齐"
  - "LLM数据增强"
  - "高斯掩码"
  - "事件定位"
---

# SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning

**会议**: CVPR 2026  
**arXiv**: [2603.05437](https://arxiv.org/abs/2603.05437)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 弱监督密集视频描述, 跨模态对齐, LLM数据增强, 高斯掩码, 事件定位

## 一句话总结
提出 SAIL，通过跨模态相似度引导的语义感知掩码生成和 LLM 合成字幕的辅助监督，在仅有字幕标注（无时间边界）的弱监督设置下，在 ActivityNet 和 YouCook2 上实现密集视频描述和事件定位的双 SOTA。

## 研究背景与动机
Dense Video Captioning（DVC）要求在未裁剪视频中同时定位事件并生成描述。全监督方法依赖昂贵的时间边界标注，弱监督 DVC（WSDVC）仅使用字幕标注训练。

**现有方法的核心问题**：当前 SOTA 方法 ILCACM 使用高斯掩码策略，通过互补字幕生成实现隐式事件定位。但其掩码学习存在两个根本缺陷：

**掩码缺乏语义对齐**：仅学习不重叠的掩码分布，不考虑掩码与对应事件的语义关系。实验发现，即使是固定的、不可训练的均匀分布掩码，性能也与 ILCACM 相当——说明现有方法仅学到了覆盖不同时间区域，而非捕捉语义相关区域

**标注稀疏性**：现有数据集事件标注极度稀疏。例如 ActivityNet 中一个 235 秒的视频可能仅有 3 个事件标注，大量潜在事件未被标注。尽管标注可能覆盖整个视频时长，但事件密度始终很低

## 方法详解

### 整体框架
SAIL 在 ILCACM 的高斯掩码互补字幕生成基础上，增加两个关键组件：(1) 基于跨模态相似度的掩码引导训练目标，使掩码聚焦于与对应字幕语义一致的视频区域；(2) LLM 生成合成字幕 + inter-mask 机制，提供更密集的监督信号。

### 关键设计

1. **Similarity-Aware Mask Guide**：通过跨模态对齐引导掩码优化

    - 功能：使高斯掩码强调与对应事件字幕语义最相似的视频区域
    - 核心思路：生成掩码 $M_i$ 后，与视频特征逐元素相乘得到正掩码特征 $\boldsymbol{v}'_i = \boldsymbol{v} \cdot M_i$。利用 CLIP 的跨模态对齐能力，最大化平均池化后的掩码特征 $\bar{\boldsymbol{v}}'_i$ 与对应字幕特征 $\boldsymbol{c}_i$ 的余弦相似度，同时最小化与同视频中其他事件字幕的相似度。使用 margin ranking loss：
    $\mathcal{L}_{\text{sim}} = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{N_s}\sum_{i=1}^{N_s}\max(0, \Delta - s^+_{b,i} + s^-_{b,i})$
   其中 $s^+ = \text{sim}(\bar{\boldsymbol{v}}'_i, \boldsymbol{c}_i)$ 为正对相似度，$s^- = \max_{j \neq i}\text{sim}(\bar{\boldsymbol{v}}'_i, \boldsymbol{c}_j)$ 为强负例相似度
    - 设计动机：将"覆盖不同区域"的弱约束升级为"对齐语义内容"的强约束

2. **LLM-Based Caption Augmentation**：利用 LLM 世界知识生成过渡事件描述

    - 功能：为每对相邻 GT 字幕 $(C_i, C_{i+1})$ 之间的时间间隔生成合成字幕 $C^{syn}_i$，每个视频产生 $N_s - 1$ 个合成字幕
    - 核心思路：设计结构化 prompt，将 LLM 定义为"视频上下文推理专家"，分析前后字幕的叙事流并推断最可能的过渡动作或状态变化。使用 Qwen3-8B 生成
    - 设计动机：解决标注稀疏导致的对齐信号不足，特别是仅有 1-2 个事件标注的视频

3. **Inter-Mask Auxiliary Guidance**：合成字幕的间接利用策略

    - 功能：为合成字幕创建"inter-mask"，定位于相邻事件掩码之间的时间区域
    - 核心思路：对每对相邻事件中心 $(c_i, c_{i+1})$，定义 inter-mask 中心为其平均值 $c^{inter}_i = \frac{c_i + c_{i+1}}{2}$，宽度固定为超参数 $w^{inter}$。将 inter-mask 应用于视频特征后，通过余弦相似度损失对齐增强特征与合成字幕：
    $\mathcal{L}_{\text{aug}} = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{N_s-1}\sum_{i=1}^{N_s-1}(1 - \text{sim}(\bar{\boldsymbol{v}}'^{inter}_{b,i}, \boldsymbol{c}^{syn}_{b,i}))$
    - 设计动机：直接将合成字幕作为强负例会引入噪声并降低性能（Table 6 验证），因此作为独立的辅助信号更稳健

### 损失函数 / 训练策略
- 最终目标：$\mathcal{L} = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}} + \mathcal{L}_{\text{sim}} + \alpha_{\text{aug}}\mathcal{L}_{\text{aug}}$
- $\mathcal{L}_{\text{pos}}$/$\mathcal{L}_{\text{neg}}$：正/负互补字幕生成损失（继承自 ILCACM）
- 超参数：$\Delta=0.1$, $w^{inter}=0.6$, $\alpha_{\text{aug}}=0.25$
- 字幕解码器：Distilled-GPT2，AdamW 优化器
- ActivityNet: lr=1e-4, 10 epochs; YouCook2: lr=5e-5, 5+15 epochs

## 实验关键数据

### 主实验

| 数据集 | 指标 | SAIL | ILCACM (前SOTA) | 提升 |
|--------|------|------|-----------------|------|
| ActivityNet | CIDEr | 35.38 | 33.42 | +1.96 |
| ActivityNet | SODA_c | 6.29 | 6.08 | +0.21 |
| ActivityNet | F1 (定位) | 57.00 | 56.20 | +0.80 |
| YouCook2 | CIDEr | 14.61 | 13.49 | +1.12 |
| YouCook2 | F1 (定位) | 20.94 | 17.88 | +3.06 |

SAIL 弱监督在多数指标上超越全监督方法 CM2 和 E2DVC。

### 消融实验

| 配置 | SODA_c | CIDEr | F1 | 说明 |
|------|--------|-------|-----|------|
| Baseline (ILCACM) | 6.08 | 33.42 | 56.20 | 无语义引导 |
| +Similarity-aware | 6.27 | 35.18 | 56.89 | 语义对齐掩码 |
| +Synthetic captions | 6.29 | 34.92 | 56.79 | LLM 增强监督 |
| +Both (SAIL) | 6.29 | 35.38 | 57.00 | 最佳组合 |

### 关键发现
- 语义感知掩码单独使用即可提升 CIDEr +1.76，证明对齐损失的有效性
- 合成字幕作为辅助信号（inter-mask）优于作为强负例（+HN）策略
- 即使只使用 25% 合成字幕也能改善性能，且随比例增加单调提升
- SAIL 在高斯掩码、Hard Binary 掩码、Cauchy 掩码三种设计下均一致提升，证明方法的通用性
- 训练开销几乎不变：1h41m vs ILCACM 的 1h38m，推理甚至略快（7m01s vs 7m11s）

## 亮点与洞察
1. **固定掩码实验的启示**：均匀分布的不可训练掩码与 ILCACM 性能相当，深刻揭示了现有方法"学到的"掩码其实缺乏语义信息
2. **LLM 增强的精妙用法**：不直接将合成字幕混入主损失（会引入噪声），而是通过 inter-mask 作为独立辅助信号——"软叙事引导"而非"硬约束"
3. **弱监督超越全监督**：在 ActivityNet 定位 F1 上与全监督方法持平，字幕质量部分指标超越，说明语义对齐是比时间边界标注更本质的监督信号

## 局限与展望
- SODA_c 提升幅度较小（+0.21），叙事连贯性改善有限
- LLM 生成的合成字幕质量取决于 LLM 的世界知识，可能在专业领域（如烹饪、体育）不够精确
- inter-mask 宽度 $w^{inter}$ 为固定超参，未自适应调整
- 仅在两个数据集上验证，未在更大规模或不同类型数据集上测试

## 相关工作与启发
- 建立在 ILCACM（当前 WSDVC SOTA）的互补字幕生成基础上，以最小修改获得显著提升
- 借鑒 CLIP 跨模态对齐能力引导时序掩码学习，思路可推广到其他弱监督视频理解任务
- LLM 生成过渡事件描述的思路很有启发——利用 LLM 的叙事推理能力补全稀疏标注
- 对弱监督视频 grounding、时序动作检测等任务有参考价值

## 评分
- 新颖性: ⭐⭐⭐ 核心思路直觉清晰，但技术贡献偏增量（在 ILCACM 上加损失+增强）
- 实验充分度: ⭐⭐⭐⭐ 消融全面，包括掩码类型、数据比例、利用策略等多维度
- 写作质量: ⭐⭐⭐⭐ 动机分析透彻，固定掩码实验的洞察非常有说服力
- 价值: ⭐⭐⭐ 弱监督超全监督有实际意义，但改进幅度不大
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[CVPR 2026\] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](stay_in_your_lane_role_specific_queries_with_overlap_suppression_loss_for_dense_.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](../../CVPR2025/video_understanding/progress-aware_video_frame_captioning.md)
- [\[CVPR 2026\] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding](mamba-vmr_multimodal_query_augmentation_via_generated_videos_for_precise_tempora.md)

</div>

<!-- RELATED:END -->
