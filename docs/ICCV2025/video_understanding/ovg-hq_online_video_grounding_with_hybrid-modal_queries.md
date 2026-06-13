---
title: >-
  [论文解读] OVG-HQ: Online Video Grounding with Hybrid-modal Queries
description: >-
  [ICCV 2025][视频理解][在线视频定位] 提出在线视频定位新任务 OVG-HQ，支持文本/图像/视频片段等混合模态查询，通过参数化记忆块（PMB）保留历史信息和混合蒸馏策略缓解模态不平衡，在流式视频中实时定位目标片段。 传统视频定位（Video Grounding）任务存在两个关键局限性： 离线设定不适用于流式场景…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "在线视频定位"
  - "混合模态查询"
  - "参数化记忆"
  - "跨模态蒸馏"
  - "流式视频"
---

# OVG-HQ: Online Video Grounding with Hybrid-modal Queries

**会议**: ICCV 2025  
**arXiv**: [2508.11903](https://arxiv.org/abs/2508.11903)  
**代码**: [GitHub](https://github.com/maojiaqi2324/OVG-HQ)  
**领域**: 视频理解  
**关键词**: 在线视频定位, 混合模态查询, 参数化记忆, 跨模态蒸馏, 流式视频

## 一句话总结

提出在线视频定位新任务 OVG-HQ，支持文本/图像/视频片段等混合模态查询，通过参数化记忆块（PMB）保留历史信息和混合蒸馏策略缓解模态不平衡，在流式视频中实时定位目标片段。

## 研究背景与动机

传统视频定位（Video Grounding）任务存在两个关键局限性：

**离线设定不适用于流式场景**：传统方法需要完整视频才能预测，无法满足监控等场景中对流式视频的即时检测需求。例如在安防监控中，需要实时分析直播画面并立即定位"一群人聚集在前门附近"这类查询，而非等待处理完整的离线录像。

**单一文本查询限制了多模态应用**：现有方法仅支持自然语言查询，但实际场景中用户可能需要提供图像或视频片段作为查询。例如文本描述可能需要详尽的文字说明，而多模态查询允许安保人员直接上传一段过去的监控片段来检索类似行为。

这两个限制催生了一个新任务：**在线视频定位与混合模态查询（OVG-HQ）**——在流式视频中，使用文本、图像、视频片段及其组合进行在线片段定位。

该任务面临两大新挑战：
- **在线设定的有限上下文**：模型只能访问滑动窗口内的帧，需要高效建模和利用历史信息。
- **训练中的模态不平衡**：不同模态对任务的贡献不均匀，强势模态会压制弱势模态的优化，导致单一统一模型难以有效处理所有模态。

## 方法详解

### 整体框架

OVG-HQ-Unify 是一个统一且灵活的模型，支持混合模态查询输入，实现在线时刻定位。其核心由三部分组成：

1. **记忆引导的多模态融合模块**：提取查询感知视频特征并通过 PMB 增强长期依赖。
2. **记忆引导的时刻预测模块**：基于预定义锚点生成预测，并通过 PMB 精炼结果。
3. **混合蒸馏策略**：用教师模型引导非主导模态的学习。

### 关键设计

1. **参数化记忆块（Parametric Memory Block, PMB）**

   传统方法使用记忆库（memory bank）通过自注意力整合历史信息，但会引入额外存储开销且计算成本随历史数据增长而增加。LSTM 虽然使用固定大小隐藏状态，但表达能力有限。

   PMB 基于 Test-Time Training (TTT) 层，将历史信息压缩到**网络参数**中，利用神经网络更大的容量提供更强的表达能力。其分两步操作：

   **步骤1：记忆当前输入**。核心组件 $f_{\text{PML}}(\cdot; W^m)$ 通过自监督重建损失将当前输入 $r_t$ 压缩进参数 $W^m$：

    $\mathcal{L}_{\text{PML}}(r_t; W^m) = \|f_{\text{PML}}(W_K r_t; W^m) - W_V r_t\|^2$

   然后通过梯度下降更新 $W^m$，其中学习率 $\eta_{\text{PML}} = \sigma(W_{lr} \cdot r_t)$ 是自适应的。

   **步骤2：生成记忆增强输出**。将当前输入投影后通过更新的 $f_{\text{PML}}$，经层归一化和投影得到记忆增强特征：

    $\hat{r}_t = W_O \cdot \text{LN}(f_{\text{PML}}(W_Q r_t; W^m))$

   **关键优势**：推理时参数动态更新，使模型能"记忆"历史信息并适应新场景，这与传统固定参数的推理方式形成鲜明对比。

2. **记忆引导的多模态融合**

    - **查询特征提取**：使用 CLIP 的文本/图像编码器提取特征，视频片段查询按固定间隔提取特征。
    - **视频特征提取**：通过滑动窗口机制处理流式视频，窗口步长为 $M$ 秒，重叠部分特征只计算一次并缓存复用。
    - **跨模态融合**：使用 Transformer decoder 做交叉注意力，视频特征作 query，各模态查询特征作 key/value。每种模态前添加特定标记 $\mathbf{m}_*$ 区分不同模态。
    - **PMB 增强**：融合后的查询感知特征 $\mathbf{F}_{qv}$ 经过 PMB 模块，用 $f_{\text{PML}}$ 替换自注意力层，整合历史上下文得到记忆引导特征 $\hat{\mathbf{F}}_{qv}$。

3. **记忆引导的时刻预测与精炼**

   基于预定义锚点 $A_n = (t - L_n, t)$，其中 $L_n = L_q / 2^{n-1}$，使用 Transformer decoder 处理锚点查询和融合特征，预测前景/背景分类分数和边界回归偏移。

   **预测精炼模块（PRM）**：由于在线设定中历史预测不可修改，PRM 利用 PMB 将当前锚点特征和预测结果压缩进参数，使精炼后的预测融入历史预测信息。仅前景分数超过阈值 $\theta$ 的锚点被保留。

### 损失函数 / 训练策略

**混合蒸馏策略**：直接用混合模态数据训练的统一模型在非文本查询上表现差（模态不平衡）。为此：

- 三种查询类型（文本、视觉、视觉+文本）交替批次训练。
- 训练文本+片段查询的专家教师模型，通过蒸馏引导统一学生模型：

$$\mathcal{L}_d = \frac{1}{N}\sum_{i=1}^{N}\left(\mathcal{L}_{\text{KL}}(\mathbf{F}_{a,i}^s, \mathbf{F}_{a,i}^t) + \mathcal{L}_2(\mathbf{r}_i^s, \mathbf{r}_i^t) + \mathcal{L}_2(\mathbf{c}_i^s, \mathbf{c}_i^t)\right)$$

总损失：$\mathcal{L} = \mathcal{L}_d + \lambda \mathcal{L}_{cls} + \mathcal{L}_{reg}$，其中 $\lambda = 10$。分类头使用 Focal Loss，回归头使用 L1 Loss。

## 实验关键数据

### 主实验

作者构建了 QVHighlights-Unify 数据集，扩展 QVHighlights 添加图像和视频片段查询。

| 方法 | 设定 | oR¹₀.₅ | omAP₀.₅ |
|------|------|---------|----------|
| TaskWeave | 离线→在线 | 7.02 | 5.96 |
| TR-DETR | 离线→在线 | 7.37 | 6.06 |
| R2-Tuning | 离线→在线 | 9.30 | 8.17 |
| TwinNet | 在线 VG | 20.78 | 19.73 |
| **OVG-HQ-Unify** | **在线 VG** | **23.26** | **23.09** |

在 ANet-Captions、TACoS、MAD 数据集上（文本查询）：

| 数据集 | 指标 | TwinNet | 本文 | 提升 |
|--------|------|---------|------|------|
| ANet-Captions | R¹₀.₇ | 12.56 | 14.36 | +1.80 |
| TACoS | R¹₀.₇ | 19.07 | 21.17 | +2.10 |
| MAD | R⁵₀.₅ | 2.00 | 3.27 | +1.27 |

### 消融实验

| 配置 | oR¹₀.₅ (Text) | omAP₀.₅ (Text) | 说明 |
|------|---------------|-----------------|------|
| Ours-ATT (替换为自注意力) | 13.93 | 16.41 | PMB 显著优于自注意力 |
| Ours-LSTM | 22.37 | 21.66 | PMB 也优于 LSTM |
| **Ours (PMB)** | **23.37** | **22.51** | - |
| w/o Refine | 17.64 | 17.43 | 预测精炼模块提升显著 |
| Pred only | 18.99 | 21.07 | 仅用预测信息不够 |
| Pred+AF（完整） | 23.37 | 22.51 | 锚点特征+预测共同最优 |

运行时分析：整体 FPS=45.95，PMB 延迟仅 2.20ms，动态更新仅 0.30ms，满足实时需求。

### 关键发现

- 混合蒸馏使纯图像查询的 oR¹₀.₅ 从 11.43% 提升到 20.41%（+8.98%），有效缓解模态不平衡。
- 片段查询优于图像查询（20.33% vs 16.14%），因视频片段更适合描述动态内容。
- 多模态查询一致优于单模态查询。

## 亮点与洞察

- **任务定义的实际意义**：OVG-HQ 结合了在线推理和混合模态查询，比传统离线+文本设置更贴近实际应用。
- **参数即记忆的新思路**：用 TTT 层的参数作为动态记忆，推理时持续更新参数，比传统记忆库更高效且更具表达力。
- **在线评估指标设计**：引入时效性衰减因子 $\beta$ 的 oR、omAP 指标，惩罚延迟预测，更好评估在线场景。

## 局限与展望

- 当前仅在 QVHighlights 上构建多模态数据集，场景相对简单，未来需在更复杂场景（如监控）验证。
- 滑动窗口大小固定，未来可探索自适应窗口策略。
- 混合蒸馏需要先训练教师模型，增加了训练成本。

## 相关工作与启发

- TTT (Test-Time Training) 的参数化记忆思想可推广到其他在线视频理解任务。
- 跨模态蒸馏策略对多模态统一模型训练有普遍参考价值。
- 在线评估指标的设计思路可用于其他流式任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务定义新颖，PMB 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 4个数据集，多种查询类型，充分的消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ 推动在线多模态视频理解发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[CVPR 2025\] Cross-modal Causal Relation Alignment for Video Question Grounding](../../CVPR2025/video_understanding/cross-modal_causal_relation_alignment_for_video_question_grounding.md)
- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](../../NeurIPS2025/video_understanding/pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)
- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
