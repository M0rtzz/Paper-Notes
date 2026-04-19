---
title: >-
  [论文解读] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach
description: >-
  [CVPR 2026 (ABAW Workshop)][语音][犹豫/矛盾识别] 提出四模态（场景 VideoMAE + 人脸 EfficientNetB0 + 音频 Wav2Vec2.0/Mamba + 文本 EmotionDistilRoBERTa）融合管线，通过原型增强 Transformer 融合模块将各模态嵌入投影到共享 128 维空间并以原型分类辅助损失正则化，在 BAH 语料的最终测试集上以 5 模型集成达到 **71.43% Macro F1**，显著超越所有单模态基线。
tags:
  - CVPR 2026 (ABAW Workshop)
  - 语音
  - 犹豫/矛盾识别
  - 多模态融合
  - 原型增强分类
  - Mamba
  - VideoMAE
  - ABAW竞赛
---

# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**会议**: CVPR 2026 (ABAW Workshop)  
**arXiv**: [2603.12848](https://arxiv.org/abs/2603.12848)  
**代码**: [LEYA-HSE/ABAW10-BAH](https://github.com/LEYA-HSE/ABAW10-BAH)  
**领域**: 情感计算 / 多模态理解  
**关键词**: 犹豫/矛盾识别, 多模态融合, 原型增强分类, Mamba, VideoMAE, ABAW竞赛

## 一句话总结

提出四模态（场景 VideoMAE + 人脸 EfficientNetB0 + 音频 Wav2Vec2.0/Mamba + 文本 EmotionDistilRoBERTa）融合管线，通过原型增强 Transformer 融合模块将各模态嵌入投影到共享 128 维空间并以原型分类辅助损失正则化，在 BAH 语料的最终测试集上以 5 模型集成达到 **71.43% Macro F1**，显著超越所有单模态基线。

## 研究背景与动机

- **任务定义**: 第 10 届 ABAW 竞赛的犹豫/矛盾（Ambivalence/Hesitancy, A/H）视频识别挑战——给定一段视频，判断其中是否包含 A/H 行为（视频级二分类）。
- **现实价值**: A/H 与决策不确定性、动机波动密切相关，在数字行为健康干预中可用于判断用户是否准备改变行为、是否出现抵触或脱离风险。
- **核心挑战**: 不同于基本情绪（如开心、惊讶），A/H 表现极其微妙，且常通过**跨模态不一致性**显现——例如话语积极但表情犹豫、语调缺乏信心但内容明确。这使得简单的单模态或浅层融合难以捕捉关键信号。
- **先前工作局限**: 前人（González-González et al., Hallmen et al., Savchenko & Savchenko）主要使用人脸+音频+文本三模态，融合策略相对简单（拼接/MLP/logistic blending），未引入场景模态，也未采用强正则化的融合。
- **本文方案**: 增加场景模态提供全局上下文，使用 Transformer 融合模块建模模态交互，并引入原型增强分类头作为训练时的辅助正则。

## 方法详解

### 整体框架

采用两阶段策略：

1. **阶段一**——独立训练四个专用单模态模型（场景/人脸/音频/文本），每个模型将视频映射到一个固定维度的嵌入向量。
2. **阶段二**——将四个嵌入投影到共享 128 维空间，通过 6 层 Transformer 编码器建模模态间交互，最终做视频级 A/H 二分类。

### 关键设计

#### 1. 场景视觉编码器（VideoMAE）

- **功能**: 捕捉视频中的全局场景动态和行为上下文信息。
- **核心思路**: 使用 VideoMAE（基于 ViT，Kinetics-400 预训练）作为视频级场景编码器。从每段视频均匀采样 $T_v=16$ 帧，resize 到 $224 \times 224$，通过 tubelet embedding 将视频划分为 $2 \times 16 \times 16$ 的时空 patch，投影到 $D=768$ 维空间并加位置编码。Transformer 编码器对所有 token 做时空自注意力，最终通过全局平均池化得到场景嵌入 $h_s$。
- **设计动机**: 场景模态提供被试所处的环境和整体行为模式（如身体姿态、手势运动），是先前 A/H 工作中未利用的互补信号。
- **训练**: 15 epochs, AdamW, lr=2e-5, weight decay=1e-2, batch size=4, cosine annealing, label smoothing=0.1。

#### 2. 人脸情感编码器（EmotionEfficientNetB0）

- **功能**: 提取面部情绪微表情信息，作为 A/H 状态的视觉线索。
- **核心思路**: 用 YOLO 人脸检测器逐帧检测人脸（多人取最大框，无检测则用全帧 fallback），crop 后 resize 到 $224 \times 224$，输入 AffectNet+ 微调的 EfficientNetB0 提取逐帧情感嵌入。对 $F$ 帧嵌入 $\{e_f\}_{f=1}^F$ 做**统计池化**——计算均值 $\mu$ 和标准差 $\sigma$ 并拼接 $[\mu; \sigma]$，形成紧凑且保留分布信息的视频级人脸表征。
- **设计动机**: A/H 可能表现为面部表情的时序波动（如从微笑到皱眉的交替），统计池化中的标准差可以捕捉这种变异性。
- **超参**: 30 帧均匀采样, 16 hidden states, 256 output features, lr=1e-3, AdamW。

#### 3. 音频时序编码器（EmotionWav2Vec2.0 + Mamba）

- **功能**: 从语音韵律中提取犹豫/矛盾的声学线索（语调变化、停顿、语速波动等）。
- **核心思路**: 从视频中提取音频并重采样到 16 kHz，输入 MSP-Podcast 语料微调的 Wav2Vec2.0（取**第 10 层**输出，维度 $T_a \times 1024$），然后用 **Mamba 编码器**建模时序依赖，最后时间平均池化得音频嵌入 $a$。Mamba 的参数：state size=8, conv kernel=4, expansion factor=2, hidden=256, FFN=512, dropout=0.1。
- **设计动机**: Mamba 提供线性复杂度的序列建模，优于 Transformer（实验验证），特别适合长度可变的音频序列。选择第 10 层而非最后层是因为中间层更好地保留了情感相关的韵律特征。
- **损失函数**: 标准交叉熵，Mamba 输出接线性层做分类。

#### 4. 文本语义编码器（EmotionDistilRoBERTa）

- **功能**: 从语言内容中提取语义级 A/H 线索（犹豫用词、自我矛盾的陈述等）。
- **核心思路**: 使用 BAH 语料提供的自动语音转录文本。主配置为 EmotionDistilRoBERTa（情感预训练的 DistilRoBERTa），直接在 A/H 任务上微调，输出经 MLP 分类头做最终预测。备选方案包括 TF-IDF + Logistic Regression/CatBoost（MF1 达 68-69%）和 EmotionTextClassifier 微调（70.00%）。
- **设计动机**: 先前研究一致表明文本是 A/H 识别最强的单模态线索。精调情感预训练模型可同时利用情感先验和任务特定知识。
- **训练**: 部分冻结 backbone, MLP head 1-3 层 hidden=64-128, dropout 0-0.3, AdamW/SGD, lr 1e-5~0.1, batch=16, 3-20 epochs + early stopping。

#### 5. Transformer 多模态融合模块

- **功能**: 将四个模态嵌入在共享空间中建模交互关系，生成融合表征用于最终分类。
- **核心思路**: 每个模态嵌入 $x_m \in \mathbb{R}^{d_m}$ 经**模态专用投影器**（Linear + LayerNorm + GELU + Dropout）映射到共享 $d=128$ 维空间得 $u_m$，加可学习模态嵌入 $E_{\text{mod}}$ 后送入 6 层 Transformer 编码器（4 头注意力, FFN 扩展因子 6, dropout=0.45）。输出通过 masked mean pooling 得融合表示 $z_{\text{fused}}$，送入线性分类器输出 logits。
- **缺失模态处理**: 若某模态不可用，提供二值 mask $\mu_m \in \{0,1\}$ 在自注意力中屏蔽对应 token，增强鲁棒性。
- **设计动机**: Transformer 编码器能自适应地学习模态间的注意力权重，比手动设计的融合策略更灵活。投影到共享低维空间（128维）减少参数量并促进跨模态对齐。

#### 6. 原型增强分类头

- **功能**: 通过原型匹配提供训练时的辅助正则信号，使融合表征在类内更紧凑。
- **核心思路**: 维护每类 $K=16$ 个可学习原型 $p_{c,k}$，融合表示 $z_{\text{fused}}$ 与原型做 $\ell_2$ 归一化后计算余弦相似度（温度 $\tau=0.3$），通过 log-sum-exp 聚合为类别原型得分 $\hat{y}^{\text{proto}}_c$。
- **损失函数**: 总损失 $\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.2 \cdot \mathcal{L}_{\text{proto}} + 0 \cdot \mathcal{L}_{\text{div}}$。其中 $\mathcal{L}_{\text{cls}}$ 为主分类交叉熵，$\mathcal{L}_{\text{proto}}$ 为原型辅助分类损失。多样性正则 $\mathcal{L}_{\text{div}}$ 实验中被禁用（$\lambda_{\text{div}}=0$）。推理时仅使用主线性分类器头。
- **设计动机**: 原型匹配提供隐式聚类约束，使类内表示更紧凑、类间更分散，在小数据集上起到正则化作用。

### 训练与集成策略

- **融合模型训练**: RMSprop, lr=9.44e-5, weight decay=5.55e-4, label smoothing=0.02, gradient clipping=0.5, cosine LR scheduler。
- **稳定性工程**: Optuna 超参搜索 + 5 个固定随机种子（42/2025/7777/12345/31415）训练，每个配置训练 5 次取平均 MF1 做选择。
- **最终集成**: 5 个种子模型的类概率平均作为最终预测。

## 实验关键数据

### 数据集

BAH 语料：1,427 段视频，300 名参与者，共 10.60 小时。通过线上 avatar 引导交互收集，包含视频级和帧级 A/H 标注、人脸裁剪、语音转录等。按参与者划分为 train/valid/public test/private test，评价指标为 Macro F1 (MF1)。

### 主实验

| ID | 配置 | 模态 | Dev MF1(%) | Valid MF1(%) | Avg MF1(%) | Final Test(%) |
|----|------|------|-----------|-------------|------------|--------------|
| 1 | EmotionEfficientNetB0 + 统计特征 + MLP | 人脸 | 65.29 | 60.05 | 62.67 | — |
| 2 | VideoMAE + Linear | 场景 | 61.71 | 62.21 | 61.96 | — |
| 3 | EmotionWav2Vec2.0 + Mamba + Linear | 音频 | 67.20 | 70.87 | 69.03 | — |
| 4 | TF-IDF + Logistic Regression | 文本 | 68.30 | 67.75 | 68.03 | — |
| 5 | TF-IDF + CatBoost | 文本 | 65.56 | 72.02 | 68.79 | — |
| 6 | EmotionTextClassifier 微调 + MLP | 文本 | 69.28 | 70.72 | 70.00 | — |
| 7 | EmotionDistilRoBERTa 微调 + MLP | 文本 | 68.54 | 71.49 | **70.02** | — |
| 11 | 四模态融合（无原型） | 全部 | 85.38 | 79.94 | 82.66 | 68.32 |
| 12 | 四模态融合（原型增强） | 全部 | 83.79 | 82.72 | **83.25** | 65.21 |
| 13 | 5 模型集成（无原型） | 全部 | 81.94 | 80.64 | 81.29 | 70.17 |
| **14** | **5 模型集成（原型增强）** | **全部** | **83.00** | **80.77** | **81.89** | **71.43** |

### 消融实验：模态组合

| ID | 模态组合 | Dev MF1(%) | Valid MF1(%) | Avg MF1(%) |
|----|---------|-----------|-------------|------------|
| 15 | 人脸 + 音频 | 63.36 | 71.44 | 67.40 |
| 16 | 人脸 + 文本 | 65.29 | 61.19 | 63.24 |
| 17 | 人脸 + 场景 | 78.07 | 77.09 | 77.58 |
| 18 | 音频 + 文本 | 67.05 | 70.99 | 69.02 |
| 19 | 场景 + 音频 | 77.37 | 77.66 | 77.51 |
| **20** | **场景 + 文本** | **81.77** | **79.00** | **80.39** |
| 21 | 场景 + 音频 + 文本 | 79.89 | 77.63 | 78.76 |
| 22 | 人脸 + 场景 + 文本 | 79.89 | 77.65 | 78.77 |
| 23 | 人脸 + 场景 + 音频 | 76.10 | 79.15 | 77.62 |
| 24 | 人脸 + 音频 + 文本 | 68.08 | 70.41 | 69.25 |
| 11 | 全部四模态 | 85.38 | 79.94 | **82.66** |

### 关键发现

1. **文本是最强单模态**：EmotionDistilRoBERTa 微调达 70.02%，其次是音频 Mamba（69.03%），人脸和场景较弱（~62%）。
2. **场景+文本是最强双模态组合**（80.39%）：场景提供行为上下文，文本提供语义内容，互补性最强。
3. **四模态融合大幅超越所有子集**：82.66% vs 最强双模态 80.39%（+2.27%），vs 最强三模态 78.77%（+3.89%）。
4. **原型增强提升 dev/valid 表现**（83.25% vs 82.66%），但单模型 final test 反而下降（65.21% vs 68.32%）——提示原型头增加了对验证集的过拟合风险。
5. **集成是泛化的关键**：单模型 68.32% → 5 模型集成 71.43%（+3.11%），集成有效缓解初始化敏感性。
6. **Mamba 优于 Transformer** 作为音频时序编码器（论文明确报告 layer 10 + Mamba 是最优音频配置）。

## 亮点与洞察

- **四模态全覆盖**：相比前人仅用人脸/音频/文本，本文新增场景模态提供了关键的全局上下文信号，场景+文本组合甚至超过人脸+音频+文本三模态（80.39% vs 69.25%）。
- **原型增强的正则效应**：原型头在训练时引导融合表征形成更清晰的类内聚类，推理时不使用，是一种零成本的训练增强技巧。
- **工程上的稳健性设计**：5 种子训练 + Optuna 超参搜索 + 概率平均集成，是竞赛中保证稳定性的完整工程范式。
- **模态缺失处理**：融合模块内置 binary mask 机制，优雅处理部分模态不可用的情况。

## 局限性

- **Dev/Valid 与 Final Test 差距显著**（83.25% → 71.43%），暴露出在仅 1,427 段视频的小数据集上严重的过拟合/泛化问题。
- **未显式建模跨模态不一致性**：A/H 的核心特征是"说的和表现的不一致"，但 Transformer 融合仅做通用注意力聚合，缺乏对模态矛盾信号的显式检测机制（如对比学习或跨模态差异度量）。
- **人脸和场景模态较弱**（~62%），可能是因为预训练任务（AffectNet 情绪分类、Kinetics 动作识别）与 A/H 判别目标不够匹配。
- **原型增强单模型反而降低 final test**（65.21% vs 68.32%），依赖集成才稳定，说明原型头可能在小数据上引入额外的过拟合。

## 相关工作与启发

- **Hallmen et al. (CVPRW 2025)**: 三模态 ViT+LSTM+BERT+MLP 融合。本文增加场景模态并替换 Mamba + 原型增强，系统性更强。
- **Savchenko & Savchenko (CVPRW 2025)**: 轻量文本+人脸融合，best val 靠合并两模态。本文覆盖四模态且 Transformer 融合更灵活。
- **González-González et al. (ICLR 2026)**: BAH 数据集创建者，建立了多种基线。本文在此基础上引入 VideoMAE 场景建模和原型增强。
- **启发方向**: ① A/H 的本质是跨模态"矛盾检测"，未来可用对比学习或差异度量显式建模模态一致/矛盾关系；② 原型增强分类可迁移到其他数据有限的细粒度情感识别任务；③ Mamba 在音频时序建模中的成功为 speech-based 任务提供了轻量替代。

## 评分

- **新颖性**: ⭐⭐⭐ — 竞赛方案，各组件借鉴现有工作，但场景模态引入和原型增强融合有一定新意。
- **实验充分度**: ⭐⭐⭐⭐ — 7 个单模态配置 + 7 个融合/集成配置 + 10 个双/三模态消融，分析全面。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，实验表格信息丰富，各模块描述完整。
- **价值**: ⭐⭐⭐ — ABAW 竞赛技术报告，对情感计算多模态融合具有参考价值，但泛化性问题待解决。

<!-- RELATED:START -->

## 相关论文

- [Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)
- [BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)
- [Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)

<!-- RELATED:END -->
