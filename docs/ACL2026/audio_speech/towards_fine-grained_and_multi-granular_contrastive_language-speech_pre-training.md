---
title: >-
  [论文解读] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
description: >-
  [ACL 2026][语音][语音风格建模] 本文提出FCaps大规模数据集（47k小时语音、19M细粒度标注）和CLSP对比学习模型，通过端到端标注管线和细粒度多粒度对比监督，实现了首个能统一表征全局和细粒度语音风格的语音-文本对齐模型。
tags:
  - ACL 2026
  - 语音
  - 语音风格建模
  - 对比学习预训练
  - 细粒度标注
  - 语音-文本对齐
  - 副语言学
---

# Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training

**会议**: ACL 2026  
**arXiv**: [2601.03065](https://arxiv.org/abs/2601.03065)  
**代码**: [GitHub](https://github.com/yfyeung/CLSP)  
**领域**: 音频语音  
**关键词**: 语音风格建模, 对比学习预训练, 细粒度标注, 语音-文本对齐, 副语言学

## 一句话总结

本文提出FCaps大规模数据集（47k小时语音、19M细粒度标注）和CLSP对比学习模型，通过端到端标注管线和细粒度多粒度对比监督，实现了首个能统一表征全局和细粒度语音风格的语音-文本对齐模型。

## 研究背景与动机

**领域现状**：语音风格（speaking style）传递了丰富的副语言信息，包括说话人固有特征（性别、年龄、口音）和情境性特征（语速、情感、表达力）。现有语音-文本表示学习方法通常依赖粗粒度标签或任务特定监督，无法捕捉语音风格的细粒度时间结构。

**现有痛点**：现有语音风格标注数据集主要采用级联标注管线——先用离散标签标注语音，再用大语言模型将标签改写为自然语言描述。这种方法存在根本性的信息瓶颈：中间的离散标签将丰富的、连续的、时变的副语言信息压缩到有限的预定义类别中，导致严重的信息损失和语义偏差。

**核心矛盾**：细粒度语音风格建模需要高质量、大规模的自由文本描述，但现有方法要么依赖人工标注（成本高、一致性差），要么使用级联管线（引入误差传播和信息损失）。

**本文目标**：(1) 构建大规模端到端的细粒度语音风格标注数据集，避免级联管线的信息瓶颈；(2) 训练能统一表征多粒度语音风格的对比学习模型。

**切入角度**：利用最新的多模态标注模型（Qwen3-Omni）直接从音频生成细粒度描述，绕过离散标签中间步骤，并通过智能体验证流程保证标注质量。

**核心 idea**：端到端标注管线 + 细粒度多粒度对比学习，消除信息瓶颈并实现从全局到细粒度的统一语音-文本表示。

## 方法详解

### 整体框架

整体分为数据构建和模型训练两部分。数据端，通过端到端管线构建FCaps数据集，包含FCaps-Emilia（46,787小时、18M细粒度标注）和FCaps-PSCBase（267小时、14万全局标注+93万细粒度标注）。模型端，CLSP采用双编码器架构（SPEAR-XLarge语音编码器 + RoBERTa文本编码器），通过两阶段课程学习进行对比训练：第一阶段在大规模细粒度数据上做标准对比学习，第二阶段引入多正例对比学习实现跨粒度泛化。

### 关键设计

1. **端到端标注管线**:

    - 功能：直接从音频生成高质量细粒度语音风格描述，避免级联管线的信息损失
    - 核心思路：使用Qwen3-Omni-30B作为详细标注器，直接以语音片段为输入生成细粒度描述，通过用户提示约束输出聚焦于说话人风格（抑制转录内容和环境声描述）。为同一语音片段使用不同随机种子多次生成，获取多个正例视图。然后使用Qwen3-30B推理模型作为验证智能体，按预定义检查清单（是否包含背景声/环境噪声描述、是否有缺失声明、是否包含无风格描述的转录内容等）过滤低质量标注
    - 设计动机：级联管线中的离散标签是信息瓶颈，端到端生成直接以音频为条件，保留了完整的副语言信息；多正例生成比纯文本改写更可靠，因为每个标注都基于原始音频信号

2. **细粒度多粒度对比学习**:

    - 功能：学习能在不同粒度上统一表征语音风格的嵌入空间
    - 核心思路：第一阶段使用标准对称InfoNCE损失在大规模细粒度数据上训练 $\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}(\log\frac{\exp(\mathbf{s}_i \cdot \mathbf{t}_{Fi}/\tau)}{\sum_j \exp(\mathbf{s}_i \cdot \mathbf{t}_{Fj}/\tau)} + \text{反向})$。第二阶段使用多正例InfoNCE，每个语音配对两个文本（一全局一细粒度，或两个不同细粒度），通过软目标分布 $D_{i,j}$ 分配概率质量（$\lambda = 0.5$），损失为交叉熵 $\mathcal{L} = \frac{1}{2}(\mathrm{CE}(\mathbf{L}/\tau, \mathbf{D}) + \mathrm{CE}(\mathbf{L}^\top/\tau, \mathbf{D}'))$
    - 设计动机：两阶段课程从纯细粒度对齐逐步过渡到跨粒度泛化，第一阶段建立精确的细粒度对应，第二阶段通过全局+细粒度混合训练实现跨粒度一致性

3. **动态任务调度器**:

    - 功能：在第二阶段平衡跨粒度泛化和细粒度判别
    - 核心思路：每个训练步随机采样两个任务之一——任务1（全局+细粒度配对）或任务2（两个不同细粒度配对）。采样概率 $p_t$ 从 $p_0 = 0.95$ 线性下降到 $p_{min} = 0.50$，在 $T = 10000$ 步内完成过渡：$p_t = \max(p_{min}, p_0 - \frac{t}{T}(p_0 - p_{min}))$
    - 设计动机：训练初期侧重跨粒度对齐（任务1占主导），后期增加细粒度判别（任务2占比上升），实现渐进式学习

### 损失函数 / 训练策略

CLSP共724M参数（SPEAR-XLarge 599M + RoBERTa 125M），在8块A100 80GB上训练。第一阶段1.2M步，第二阶段4k步微调。使用ScaledAdam优化器和Eden学习率调度器，峰值学习率分别为0.045和0.001。温度参数 $\tau$ 可学习。

## 实验关键数据

### 主实验

| 任务 | 指标 | CLSP | 之前SOTA(ParaCLAP) | 提升 |
|------|------|------|-------------------|------|
| 全局检索 S→T | R@1 | 45.6 | 2.1 | +43.5 |
| 全局检索 T→S | R@1 | 40.3 | 0.4 | +39.9 |
| 细粒度检索 S→T | R@1 | 68.1 | 1.2 | +66.9 |
| 细粒度检索 T→S | R@1 | 67.2 | 1.2 | +66.0 |
| 零样本情感(IEMOCAP) | WA/UA | 57.2/56.1 | 46.1/46.5 | +11.1/+9.6 |
| 零样本性别(RAVDESS) | WA/UA | 100.0/100.0 | 99.2/99.2 | +0.8 |
| 风格相似度(固有特征) | Pearson r | 0.893 | 0.663 | +0.230 |
| 风格相似度(情境特征) | Pearson r | 0.903 | 0.323 | +0.580 |

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| 端到端标注 vs 级联标注 | 正确性/覆盖度/自然度 | 4.42/4.55/4.92 vs 3.30/3.10/4.15 |
| 静态调度 vs 动态调度 | 任务采样策略 | 动态优于静态 |
| $\lambda=0.5$ vs 其他 | 多正例权重分配 | 0.5最优 |

### 关键发现

- CLSP在所有任务上均大幅领先现有方法，尤其在检索任务上提升巨大（R@1从个位数到40-68%）
- 端到端标注质量全面优于级联标注：正确性+1.12、覆盖度+1.45、自然度+0.77
- 在语音风格相似度评分上与人类判断高度一致（Pearson r > 0.88），尤其在情境特征上（0.903 vs ParaCLAP的0.323）远超现有方法
- 零样本分类性能强劲，情感识别WA达57.2%，性别识别100%，说明学到的表示具有良好的副语言信息编码能力

## 亮点与洞察

- FCaps是目前最大的细粒度语音风格标注数据集（47k小时、19M标注），填补了关键的数据空白
- 端到端标注管线的设计思路值得推广——直接从原始信号生成描述避免了信息瓶颈，智能体验证保证了质量
- 两阶段课程学习从细粒度到跨粒度的渐进训练策略有效，仅4k步微调就显著提升了跨粒度能力
- CLSP可以作为语音风格评估器使用，与人类判断的高相关性使其有望替代昂贵的主观评估

## 局限与展望

- 当前仅支持英文语音，跨语言的语音风格建模有待扩展
- 端到端标注管线依赖Qwen3-Omni的质量，该模型本身的偏差可能传递到标注中
- 文本编码器使用RoBERTa-base（125M），更大的文本编码器可能进一步提升性能
- 未探索细粒度标注中的时间对齐——当前标注描述话语内的风格变化但不提供精确时间戳

## 相关工作与启发

- **vs ParaCLAP（Jing et al.）**: ParaCLAP聚焦于情感中心的监督，CLSP通过细粒度多粒度监督覆盖更广的副语言维度
- **vs GLAP（Dinkel et al.）**: GLAP使用转录文本配对提供词汇级监督，CLSP使用风格描述提供副语言级监督
- **vs CapSpeech（Wang et al.）**: CapSpeech使用级联标注管线，CLSP的端到端管线避免了信息瓶颈，标注质量更高

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 端到端标注管线和多粒度对比学习都是重要创新，FCaps数据集贡献巨大
- 实验充分度: ⭐⭐⭐⭐⭐ 标注质量评估、4类下游任务、与人类判断的相关性分析全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数据集构建过程详实
- 价值: ⭐⭐⭐⭐⭐ 数据集和模型均开源，对语音风格建模和评估有广泛影响

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[CVPR 2026\] Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](../../CVPR2026/audio_speech/unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)
- [\[ACL 2025\] T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](../../ACL2025/audio_speech/t2a_feedback_audio_gen.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)

<!-- RELATED:END -->
