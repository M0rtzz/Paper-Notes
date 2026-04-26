---
title: >-
  [论文解读] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing
description: >-
  [ECCV2024][语音][audio-visual video parsing] 提出 CoLeaF 双分支学习框架，通过事件感知对比学习显式优化跨模态上下文的整合，在弱监督音视频解析任务上平均提升 1.9% F-score。
tags:
  - ECCV2024
  - 语音
  - audio-visual video parsing
  - weakly supervised learning
  - 对比学习
  - 知识蒸馏
  - 跨模态
---

# CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing

**会议**: ECCV2024  
**arXiv**: [2405.10690](https://arxiv.org/abs/2405.10690)  
**代码**: [GitHub](https://github.com/faeghehsardari/coleaf)  
**领域**: audio_speech  
**关键词**: audio-visual video parsing, weakly supervised learning, contrastive learning, knowledge distillation, cross-modal learning

## 一句话总结

提出 CoLeaF 双分支学习框架，通过事件感知对比学习显式优化跨模态上下文的整合，在弱监督音视频解析任务上平均提升 1.9% F-score。

## 背景与动机

音视频视频解析（AVVP）任务要求在视频中同时检测三类事件：仅听觉事件（audible-only）、仅视觉事件（visible-only）和听视觉共现事件（audible-visible）。由于逐帧标注各模态事件的标注成本极高，该任务在弱监督设置下进行——训练时只有视频级标签可用。

现有方法（如 HAN、CMPAE）普遍通过 self-attention 和 cross-attention 同时利用单模态和跨模态上下文。然而作者发现一个关键矛盾：**跨模态学习虽然有利于检测音视共现事件，但会损害未对齐事件（仅听觉/仅视觉）的检测效果**，因为跨模态注意力会引入不相关的模态信息。实验验证表明，CMPAE 仅使用单模态信息时，在未对齐事件上的 true positive 率反而更高。

此外，视频中常存在复杂的类别共现关系（如乐器演奏伴随人声），显式建模这些关系能提升性能，但现有方法（如 CVCMS）的类间关系建模带来 $T \times C \times C$ 的计算开销。

## 核心问题

1. **跨模态信息整合的两难**：如何让网络在弱监督场景下，对音视共现事件利用跨模态上下文、对未对齐事件过滤跨模态干扰？
2. **类间关系建模的效率**：如何在不增加推理计算开销的前提下利用类别共现关系？

## 方法详解

### 整体架构：双分支设计

CoLeaF 包含两个并行分支，共享相同的音视频输入特征：

- **Reference 分支**（仅训练时使用）：仅利用单模态信息 + 显式建模类间关系
- **Anchor 分支**（训练+推理）：利用单模态 + 跨模态上下文（可嵌入任意 AVVP 方法如 HAN、CMPAE）

### Reference 分支

将音频/视觉输入特征 $F^a, F^v$ 与可学习的类别 token $C^a, C^v$（每个类别一个 token）拼接后，分别送入各自模态的 self-attention 层。这样设计的好处是：

- self-attention 在时间维度上学习单模态上下文
- 类别 token 与输入 token 的交互显式建模了类间共现关系
- 不使用 cross-attention，避免跨模态干扰

输出的时间 token 经 FC 层预测片段级事件概率，类别 token 经 AvgPool + Sigmoid 也参与 BCE 损失监督。

### Anchor 分支

可复用任意 AVVP 方法（如 HAN 的 self-attention + cross-attention 结构），生成融合跨模态上下文的特征表示和事件概率预测。

### 事件感知对比学习（Event-Aware Contrastive Loss）

核心创新：根据视频中未对齐事件的比例，自适应调节对比学习的强度。

$$\mathcal{L}_{Evt}^{Anch} = -\frac{1}{T} \sum_{\phi \in \{a,v\}} \vartheta^\phi \sum_{t=1}^{T} \log \frac{\exp(\hat{f}_t^{\phi\top} \cdot \ddot{x}_t^\phi / \tau)}{\sum_{n \neq t} \exp(\hat{f}_t^{\phi\top} \cdot \ddot{x}_n^\phi / \tau)}$$

其中权重 $\vartheta^\phi$ 反映未对齐程度：

- 从 Reference 分支的预测中提取伪标签，统计仅听觉事件数 $N^a$、仅视觉事件数 $N^v$ 和共现事件数 $N^{av}$
- $\vartheta^a = N^a / (N^a + N^{av})$：未对齐事件越多，对比学习鼓励越强，迫使 Anchor 向 Reference 的单模态表示靠拢
- 当所有事件都是音视共现时 $\vartheta = 0$，不施加对比约束

### 自模态感知知识蒸馏（Self-Modality-Aware KD）

由于 Reference 只用模态无关的视频级标签训练，其表示能力有限。为此，Anchor 反向向 Reference 蒸馏模态感知的伪标签：

$$\mathcal{L}_{SelfMo}^{Ref} = \sum_{\phi \in \{a,v\}} BCE(G^\phi, \ddot{\mathcal{P}}^\phi)$$

其中 $G^\phi$ 是从 Anchor 的音频/视觉预测概率中阈值化得到的伪标签。这形成了一个**协作闭环**：Anchor 为 Reference 提供模态感知监督，Reference 再通过对比学习优化 Anchor 的跨模态整合。

### 共现类别知识蒸馏（Co-occurrence Class KD）

通过类别相关矩阵传递类间关系：

$$\mathcal{L}_{CoCls}^{Anch} = \sum_{\phi \in \{a,v\}} MSE(\ddot{M}^\phi, M^\phi)$$

其中 $\ddot{M}^\phi_{i,j} = \ddot{\mathcal{P}}_i^\phi \cdot \ddot{\mathcal{P}}_j^\phi$ 是 Reference 的类间相关矩阵。推理时类别 token 和 Reference 分支完全丢弃，不带来额外计算。

### 新评估指标

作者指出传统 A/V 指标的缺陷：只比较单模态预测与单模态标签，会将音视共现预测误算为仅听觉/仅视觉的 true positive。提出 Ao/Vo 指标同时考虑双模态：

$$\hat{y}_t^{ao} = \hat{y}_t^a \odot (1 - \hat{y}_t^v)$$

仅当预测为"有音频且无视频"时才算 audible-only 事件。

## 实验关键数据

**LLP 数据集**（标准设置，Segment-level / Event-level）：

| 方法 | Ao | Vo | AV | Ao | Vo | AV |
|------|-----|-----|-----|-----|-----|-----|
| HAN | 33.1 | 50.7 | 48.9 | 31.0 | 50.1 | 43.0 |
| JoMoLD | 46.2 | 58.8 | 57.2 | 40.9 | 59.0 | 49.6 |
| CMPAE | 48.2 | 57.9 | 57.5 | 43.6 | 57.5 | 49.6 |
| **CoLeaF** | **49.3** | **62.4** | **58.6** | **44.1** | **62.2** | **52.1** |

关键消融结果：

- 事件感知对比损失 → Ao 提升 2.1%（segment）/ 2.5%（event）
- 自模态蒸馏 → 所有事件类型均提升 0.7%~1.6%
- 类别 token + 共现蒸馏 → 全面提升 1.1%~2.3%
- 框架通用性：嵌入 HAN 和 CMPAE 平均提升 2.4% F-score

## 亮点

1. **问题洞察精准**：首次量化分析了跨模态学习对未对齐事件的负面影响，并提出了有针对性的解决方案
2. **事件感知对比学习**：根据视频内容自适应调节跨模态约束强度，设计巧妙
3. **双分支协作机制**：Anchor↔Reference 的双向知识传递形成良性循环
4. **推理零开销**：类间关系建模和 Reference 分支仅在训练时使用，推理时完全不增加计算
5. **框架通用性强**：任何 AVVP 方法均可作为 Anchor 嵌入，即插即用
6. **新指标贡献**：Ao/Vo 指标更准确反映未对齐事件检测能力，揭示了传统指标的误导性

## 局限与展望

1. 仅在 LLP 和 UnAV-100 两个数据集上验证，缺乏更大规模数据集的评估
2. 伪标签的质量依赖预设阈值 $\theta$，对不同数据集可能需要调参
3. Reference 分支和 Anchor 分支的架构选择相对简单，未探索更强的 backbone
4. 未考虑时序上下文的跨模态对齐，仅在全局层面调节对比强度
5. 类别 token 增加了 Reference 41.7% 的 FLOPs，虽不影响推理，但增加了训练成本

## 与相关工作的对比

| 方法 | 跨模态优化方式 | 类间关系 | 推理开销 |
|------|--------------|---------|---------|
| HAN | 直接融合 self+cross attention | 无 | 基准 |
| CVCMS | 学习类间依赖 | 显式建模 | $T \times C^2$ 额外开销 |
| CMPAE | 主观逻辑理论调节 | 无 | 基准 |
| JoMoLD | CLIP/CLAP 离线标签 | 无 | 需两阶段训练 |
| **CoLeaF** | **嵌入空间显式对比优化** | **训练时建模，推理时蒸馏** | **推理零开销** |

## 启发与关联

1. "训练时复杂、推理时简洁"的范式值得借鉴——在训练阶段引入辅助分支学习特定知识，再通过蒸馏传递给主网络
2. 对比学习的强度自适应机制可迁移到其他多模态场景：根据模态对齐程度动态调节跨模态融合
3. 传统指标的缺陷分析提供了一个重要提醒：评估指标的设计应与任务定义严格一致

## 评分
- 新颖性: ⭐⭐⭐⭐ (事件感知对比学习和协作蒸馏机制设计新颖)
- 实验充分度: ⭐⭐⭐⭐ (消融全面但数据集偏少)
- 写作质量: ⭐⭐⭐⭐ (动机清晰，公式完整)
- 价值: ⭐⭐⭐⭐ (通用框架+新指标，对领域有推动)

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](../../CVPR2025/audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[ECCV 2024\] Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)
- [\[ECCV 2024\] Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)
- [\[ECCV 2024\] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)

<!-- RELATED:END -->
