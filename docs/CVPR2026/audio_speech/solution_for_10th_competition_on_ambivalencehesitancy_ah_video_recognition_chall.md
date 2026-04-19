---
title: >-
  [论文解读] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion
description: >-
  [CVPR2026][语音][多模态融合] 针对第10届 ABAW 竞赛的矛盾/犹豫 (A/H) 视频识别任务，提出基于散度的多模态融合策略，通过计算视觉（AU）、音频（Wav2Vec 2.0）和文本（BERT）三个模态嵌入的逐对绝对差来显式建模跨模态冲突，在 BAH 数据集上以 Macro F1 0.6808 大幅超越基线 0.2827。
tags:
  - CVPR2026
  - 语音
  - 多模态融合
  - 矛盾/犹豫识别
  - Action Units
  - 跨模态冲突
  - 情感计算
---

# Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion

**会议**: CVPR2026  
**arXiv**: [2603.16939](https://arxiv.org/abs/2603.16939)  
**作者**: Aislan Gabriel O. Souza, Agostinho Freire, Leandro Honorato Silva 等 (Universidade de Pernambuco)
**领域**: audio_speech  
**关键词**: 多模态融合, 矛盾/犹豫识别, Action Units, 跨模态冲突, 情感计算

## 一句话总结

针对第10届 ABAW 竞赛的矛盾/犹豫 (A/H) 视频识别任务，提出基于散度的多模态融合策略，通过计算视觉（AU）、音频（Wav2Vec 2.0）和文本（BERT）三个模态嵌入的逐对绝对差来显式建模跨模态冲突，在 BAH 数据集上以 Macro F1 0.6808 大幅超越基线 0.2827。

## 研究背景与动机

矛盾（Ambivalence）和犹豫（Hesitancy）是一类复杂的情感状态，其核心特征在于个体对行为改变持有相互冲突的情感或意图。与离散情感（如快乐、愤怒）不同，A/H 不表现为某种固定的面部表情或语调，而是体现为跨通信渠道之间的**不一致性**——一个人说的话、说话的语调、面部表情三者之间的矛盾才是 A/H 的本质信号。

现有多模态情感计算方法在融合策略上存在根本性的假设偏差：

- **拼接融合（Concatenation）**：直接将各模态特征首尾拼接，假设模态之间是互补关系，让分类器自行学习模态间的交互，但这种隐式学习在数据有限时难以捕捉到微妙的跨模态冲突信号
- **后期混合（Late Blending）**：在决策层面融合各模态的独立预测，同样无法显式建模模态之间的分歧
- **注意力融合（Co-attention）**：虽然提升了模态间交互的建模能力，但计算代价较高，且注意力权重的物理含义不如直接的差异度量清晰

这些方法的共同问题是将模态视为"互补信息源"，而非"可能产生冲突的信号源"。对于 A/H 检测而言，真正有诊断价值的信号恰恰在于模态之间的**不一致**，例如面部微笑但语调犹豫，或言语肯定但面部表情紧张。

此外，在上一届 ABAW-8 竞赛中，Savchenko 使用 EmotiEffLib 面部描述子配合 Wav2Vec 2.0 和 RoBERTa 进行后期混合融合取得了较好成绩；Hallmen 等人发现文本模态提供最强信号。但这些工作都未显式建模跨模态冲突这一 A/H 的核心定义特征。

基于这一洞察，作者提出：既然 A/H 在理论定义上就是"共存的冲突信号"，那么融合模块应当直接计算模态间的**散度/差异**，而非简单堆叠。这一思路简洁且理论动机充分。

## 方法详解

### 整体架构

系统采用三模态流水线：视觉（面部 Action Units）→ 音频（Wav2Vec 2.0）→ 文本（BERT），各模态独立编码后通过 BiLSTM + Attention Pooling 生成固定维度表示，再经投影层映射到共享嵌入空间，最后通过散度融合模块进行分类。

### 特征提取

**视觉模态**：使用 Py-Feat 从预裁剪的人脸图像中逐帧提取 20 个 Action Units (AUs)，采样率为每 3 帧取 1 帧（约 10 fps）。为建模时间动态，设计滑动窗口统计特征——在窗口大小 $W=16$、步长 $S=8$ 的滑动窗口内，对每个 AU 计算四种统计量（均值、标准差、斜率、范围），得到 80 维的窗口描述子。这一设计的关键洞察是：**A/H 体现为面部动作的时间波动，而非某个固定表情**。

**音频模态**：以 16 kHz 单声道提取音频，使用预训练的 Wav2Vec 2.0 (wav2vec2-base-960h) 编码，生成 768 维嵌入，时间分辨率约 50 Hz。

**文本模态**：使用 BERT-base 编码转录文本，取 [CLS] token 的 768 维输出，微调最后两层（学习率降低）以适配下游任务。

### 时序建模与注意力池化

视觉和音频模态的时序特征分别经过 2 层 BiLSTM（隐藏维度 64）处理，随后通过注意力池化将变长序列压缩为固定长度向量。每个模态的输出再经线性投影层映射到 $D=128$ 维的共享嵌入空间，得到 $\mathbf{h}'_v$、$\mathbf{h}'_a$、$\mathbf{h}'_t$。

### 三种融合策略对比

这是本文的核心设计——作者系统比较了三种融合策略：

**Fusion A（隐式融合）**：传统拼接方式，将三个模态的嵌入直接拼接：
$$\mathbf{f}_A = [\mathbf{h}'_v;\, \mathbf{h}'_a;\, \mathbf{h}'_t]$$
分类器需要自行从 $3 \times 128 = 384$ 维向量中学习模态间关系。

**Fusion B（散度融合）**：计算所有模态对之间的逐元素绝对差：
$$\mathbf{f}_B = [|\mathbf{h}'_v - \mathbf{h}'_a|;\, |\mathbf{h}'_v - \mathbf{h}'_t|;\, |\mathbf{h}'_a - \mathbf{h}'_t|]$$
这种设计直接捕捉跨模态冲突——如果视觉表达与音频情感一致，差值趋近于零；如果存在矛盾，差值会在相关维度上产生显著响应。维度同样为 $3 \times 128 = 384$。

**Fusion C（组合融合）**：同时保留原始嵌入和散度信息：
$$\mathbf{f}_C = [\mathbf{f}_A;\, \mathbf{f}_B]$$
维度为 768，信息最丰富但参数量更大。

### 分类与训练

融合向量通过 3 层 MLP（含 Dropout $p=0.3$）进行二分类。训练细节：
- 损失函数：BCEWithLogitsLoss + 类别权重（应对正负样本不均衡）
- 优化器：AdamW，BERT 部分学习率 $5 \times 10^{-5}$，其他参数 $5 \times 10^{-4}$
- 学习率调度：余弦退火，共 30 个 epoch
- 梯度裁剪阈值 1.0，早停 patience 为 8

### AU 时间波动的统计验证

作者在 1132 个视频上进行了 Mann-Whitney U 检验，验证了 AU 特征的判别能力。结论是：AU 的**时间变异性（标准差）**是 A/H 的主要视觉判别因子，而非 AU 的均值强度。这说明 A/H 表现为面部的"不稳定性"——面部肌肉的频繁波动而非持续的某种表情。

## 实验关键数据

### Table 1: AU 特征判别力分析（Mann-Whitney U, N=1132）

| 特征 | 统计量 | A/H vs 非A/H | 效应量 \|r\| |
|---|---|---|---|
| AU06 (颧大肌/脸颊提升) | std | 0.076 vs 0.059 | 0.186 |
| AU09 (鼻翼提升) | std | 0.095 vs 0.084 | 0.186 |
| AU12 (微笑/嘴角提升) | std | 0.089 vs 0.068 | 0.172 |
| AU26 (下颚张开) | zcr | 0.421 vs 0.384 | 0.168 |
| AU02 (外侧眉毛提升) | std | 0.110 vs 0.102 | 0.149 |

所有特征均通过 Bonferroni 校正后仍显著。效应量 \|r\| < 0.2 属于小效应，这解释了为什么纯视觉模型性能有上限（约 0.56 F1），需要多模态融合来弥补。

### Table 2: BAH 数据集结果（Macro F1）

| 模型 | Val F1 | Test F1 |
|---|---|---|
| **单模态** | | |
| Visual AUs (XGBoost) | 0.6194 | 0.5642 |
| Audio Wav2Vec (LSTM) | 0.5218 | **0.6141** |
| Text BERT | 0.5758 | 0.5904 |
| **多模态（原始 AU）** | | |
| Fusion A (隐式拼接) | 0.6788 | 0.6604 |
| Fusion B (散度融合) | 0.6524 | **0.6808** |
| Fusion C (组合融合) | 0.6700 | 0.6766 |
| **多模态（窗口 AU）** | | |
| Fusion B (散度融合) | **0.6912** | 0.6602 |
| **竞赛基线** | — | 0.2827 |

关键发现：
- 音频是最强的单模态（Test F1=0.6141），与先前文献一致
- Fusion B（散度融合）在测试集上取得最优 0.6808，较基线提升 **140.8%**
- 窗口 AU 的 Fusion B 验证集最高（0.6912）但测试集下降（0.6602），反映训练数据有限（仅 598 视频）导致的过拟合
- Fusion C 虽然信息量最大，但并未优于纯散度 Fusion B，说明散度信号已充分捕捉 A/H 特征

## 亮点与洞察

- **理论驱动的融合设计**：散度融合策略直接源于 A/H 的理论定义——共存的冲突信号，而非经验性地堆叠模态。这种"从任务定义推导架构设计"的思路值得借鉴
- **可解释的视觉特征**：选择 Action Units 而非黑盒深度特征，使得统计分析可以揭示"哪些面部行为与 A/H 相关"（如脸颊提升、鼻翼提升的波动），增强了方法的可解释性
- **面部不稳定性假说**：统计分析证实 A/H 表现为 AU 的时间波动而非某种固定表情，这一发现对情感计算领域有启发性——复杂情感状态可能更多体现在动态特征而非静态特征中
- **简洁有效**：整体架构不依赖复杂的注意力机制或 Transformer，BiLSTM + 逐元素差值的方案计算高效，在小数据场景下尤其合适

## 局限性

- **数据规模有限**：训练集仅 598 视频，窗口 AU 特征在验证集上表现最优但测试集反而下降，明确反映了过拟合问题；更丰富的特征表示需要更大的数据来支撑
- **散度度量过于简单**：仅使用逐元素绝对差作为跨模态冲突的度量，未考虑更高阶的分布级散度（如 KL 散度、MMD），可能遗漏非线性冲突模式
- **时间对齐缺失**：三个模态的时间分辨率不同（视觉 10 fps、音频 50 Hz、文本全局），当前方案通过注意力池化压缩为单向量后再计算散度，丢失了时间维度的细粒度对齐信息
- **视觉表示单一**：仅使用 AU 作为视觉特征，AU 的效应量较小（\|r\| < 0.2），未探索深度面部嵌入（如 EmotiEffLib）或结合 MediaPipe blendshapes 等替代方案
- **泛化性待验证**：仅在 BAH 数据集上评测，未在其他情感冲突检测数据集上验证方法的通用性

## 相关工作

- **ABAW 竞赛系列**：从第1届到第10届，逐步从 Aff-Wild2 离散情感/效价-唤醒扩展到 A/H 识别等更复杂的情感任务
- **ABAW-8 方案**：Savchenko 使用 EmotiEffLib + Wav2Vec 2.0 + RoBERTa 后期混合取得最优成绩；Hallmen 等人用 ViT-Huge + Wav2Vec 2.0 + GTE 发现文本信号最强，二者均未显式建模跨模态冲突
- **多模态融合**：从早期拼接、后期混合到注意力瓶颈（Attention Bottleneck），融合策略逐步发展，但针对"模态冲突"的专门设计仍较少
- **Action Unit 检测**：EAC-Net、JAA-Net 等推动了 AU 的深度学习检测，但 AU 在 A/H 判别中的时间动态特性是本文新探索的方向
- **本文定位**：从 A/H 的理论定义出发，提出"散度即特征"的融合思路，填补了现有融合方法对模态冲突显式建模的空白

## 评分

- 新颖性: ⭐⭐⭐ — 散度融合思路简洁且理论动机清晰，但技术上仅是逐元素差值，创新幅度有限
- 实验充分度: ⭐⭐⭐ — 三种融合策略+单模态消融+统计分析较为完整，但仅单一数据集，无跨数据集验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，统计分析与方法设计逻辑自洽，竞赛论文写作规范
- 价值: ⭐⭐⭐ — 散度融合的思路对需要检测"模态冲突"的任务（如讽刺检测、欺骗检测）有参考价值

<!-- RELATED:START -->

## 相关论文

- [Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)
- [BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)
- [TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization](../../ICLR2026/audio_speech/triplesumm_adaptive_triple-modality_fusion_for_video_summarization.md)
- [Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)

<!-- RELATED:END -->
