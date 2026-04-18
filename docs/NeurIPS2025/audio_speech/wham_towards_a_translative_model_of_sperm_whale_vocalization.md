---
title: >-
  [论文解读] WhAM: Towards A Translative Model of Sperm Whale Vocalization
description: >-
  [NeurIPS 2025][语音][抹香鲸声学] 提出 WhAM（Whale Acoustics Model），首个基于 Transformer 的抹香鲸 coda 生成模型，通过微调 VampNet 实现声学翻译、合成生成与下游分类的三合一能力。
tags:
  - NeurIPS 2025
  - 语音
  - 抹香鲸声学
  - 生成式音频模型
  - 声学翻译
  - Masked Acoustic Token Model
  - 跨域风格迁移
---

# WhAM: Towards A Translative Model of Sperm Whale Vocalization

**会议**: NeurIPS 2025  
**arXiv**: [2512.02206](https://arxiv.org/abs/2512.02206)  
**代码**: [GitHub](https://github.com/Project-CETI/wham)  
**领域**: 音频/语音 (生物声学)  
**关键词**: 抹香鲸声学, 生成式音频模型, 声学翻译, Masked Acoustic Token Model, 跨域风格迁移

## 一句话总结

提出 WhAM（Whale Acoustics Model），首个基于 Transformer 的抹香鲸 coda 生成模型，通过微调 VampNet 实现声学翻译、合成生成与下游分类的三合一能力。

## 研究背景与动机

**领域现状**: 抹香鲸通过短促的点击序列（codas）进行复杂的社会交流，codas 的节奏、时间和频谱特征是鲸群方言和社群身份的重要标志。近年来机器学习方法已在 coda 检测分类（Bermant et al., 2019）、GAN 生成（Beguš et al., 2023）和时序分析（Sharma et al., 2024b）方面取得进展。

**现有痛点**: (a) 现有 GAN 生成模型无法以音频提示（audio prompt）为条件进行生成；(b) 基于时序的方法只关注点击间隔（ICI），忽略了原始音频中的频谱特征（如元音模式）；(c) 分类与生成任务分别训练独立模型，缺乏统一框架；(d) 无人尝试跨声学域的翻译任务。

**核心矛盾**: 抹香鲸 coda 数据集极其稀缺（仅约 1 万条录音，总时长 ~6 小时），而现代音频生成模型通常需要海量数据训练。

**本文目标**: 构建一个统一的生成模型，在数据极度稀缺的条件下，同时实现 coda 合成、声学翻译和特征分类。

**切入角度**: 利用大规模音乐数据预训练的 VampNet（Masked Acoustic Token Model），通过两阶段微调（域适应 → 物种特定微调）将其适配到抹香鲸领域。

**核心 idea**: 将预训练的音乐生成 Transformer 通过 LoRA 微调迁移到生物声学领域，实现首个从任意音频到抹香鲸 coda 风格的声学翻译模型。

## 方法详解

### 整体框架

WhAM 基于 VampNet 架构，由三部分组成：
- **Acoustic Tokenizer $T$**：将 $N_{\text{sec}}$ 秒音频（采样率 $N_{\text{sam}}$ Hz）编码为 $\ell$ 个离散 token 序列 $T: \mathbb{R}^{N_{\text{sec}} \times N_{\text{sam}}} \to \Sigma^{\ell}$
- **Masked Acoustic Token Model (MATM) $M$**：双向 Transformer，执行 cloze 任务 $M: (\Sigma \cup \{[\texttt{MASK}]\})^{\ell} \to \Sigma^{\ell}$
- **Detokenizer $T^{-1}$**：将 token 序列还原为音频 $T^{-1}: \Sigma^{\ell} \to \mathbb{R}^{N_{\text{sec}} \times N_{\text{sam}}}$

生成过程采用迭代并行解码（iterative parallel decoding），逐步"揭开"被遮蔽的 token。

### 关键设计

#### 两阶段微调策略
- **阶段 1 — 域适应（Domain Adaptation）**: 在 FSD（7h45m 动物音频）+ AudioSet（~5h 动物音频）+ WMMS（4h8m 海洋哺乳动物）+ BirdSet（110h 子集）上微调 500k 迭代
- **阶段 2 — 物种特定微调**: 在 DSWP（2507 条 coda，1h26m）+ CETI（7653 条 coda，4h33m）上再微调 500k 迭代
- 两个阶段均使用 LoRA（Low Rank Adaptation）进行高效微调

#### 声学翻译机制
输入音频 → Tokenizer → 部分遮蔽（masking）→ MATM 迭代解码生成新 token → Detokenizer → 输出 coda 风格音频。遮蔽策略可灵活设计，例如保留节拍位以保持节奏。

### 损失函数/训练策略

- 训练目标：标准的 masked token prediction loss（交叉熵）
- 微调方式：LoRA，仅更新低秩适配矩阵
- 单 GPU 训练 5 天即可完成

## 实验关键数据

### 主实验 — 声学翻译质量（FAD）

使用 BirdNET 嵌入计算 Fréchet Audio Distance（FAD），归一化到 $[0,1]$：

| 音频来源 | 翻译前 FAD | 翻译后 FAD | 是否 FAD 不可区分 |
|---------|-----------|-----------|-----------------|
| 自然 coda（基线） | — | 0.21 | 基线阈值 |
| 4 种海洋哺乳动物 | 高 | < 0.21 | ✅ |
| 数字 beeps | ~同 coda | ~同 coda | ✅ |
| 12 种海洋哺乳动物 | 高 | 显著降低 | 部分 ✅ |

关键发现：5 种完全不同的音频来源经 WhAM 翻译后与自然 coda 的 FAD 不可区分。

### 专家感知实验

| 任务 | 准确率 | Fleiss' $\kappa$ |
|-----|--------|-----------------|
| 纯听觉 2AFC（Task 1） | 81% | 0.41 |
| 混合分类（Task 2） | — | 0.44 |
| 频谱辅助 2AFC（Task 3） | 83% | 0.41 |

- 自然 coda 被误判为合成的概率为 36%
- 海象→coda 翻译仅被专家以 75% 准确率识别

### 下游分类任务

| 任务 | WhAM | AVES | BirdNET | CLAP | Random |
|-----|------|------|---------|------|--------|
| Coda 检测 | 91.3±0.2 | 92.8±0.1 | 93.0±1.0 | 96.8±1.4 | 60.9 |
| 节奏分类 | 87.4±1.6 | 90.4±1.6 | 88.6±0.2 | 92.4±2.4 | 66.3 |
| 社群分类 | 70.5±5.6 | 92.0±0.7 | 93.2±0.1 | 85.5±1.4 | 42.5 |
| 元音分类 | 85.2±2.5 | 91.8±2.9 | 85.9±4.6 | 84.3±0.9 | 66.3 |

WhAM 尽管以生成为主要训练目标，在下游分类上仍显著高于随机基线。

### 关键发现

1. 纯生成训练的 WhAM 学到了有意义的生物声学特征，支撑了 coda 检测、节奏和元音分类
2. 仅 5 天单 GPU 训练、数据量比典型音频模型小几个数量级，依然取得强结果
3. 专家难以可靠地区分合成 coda 和真实 coda（准确率仅 81%）

## 亮点与洞察

- **首创性**: 首个统一声学翻译+生成+分类的抹香鲸 coda 模型，也是首次对合成 coda 进行专家感知评估
- **迁移学习的巧妙运用**: 音乐预训练 → 动物声学域适应 → 特定物种微调，在数据极度稀缺条件下仍然有效
- **跨域翻译**: 能将任意音频（海象、beeps 等）翻译为 coda 风格，展示了模型对 coda 声学本质的深层理解
- **生成即学习**: 纯生成训练即可获得有用的判别特征，与自监督学习的哲学一脉相承

## 局限与展望

1. **Codec 瓶颈**: 当前仅微调 MATM，保持 codec 不变——可能无法精确表征 3.7–5.7 kHz 频段的元音模式
2. **点击逼真度**: 专家指出合成点击的起始/衰减过于突兀，背景噪声不自然
3. **数据质量**: 训练数据中混入了回声定位序列（非交流用 coda）
4. **语义缺失**: 当前实现的是声学层面的翻译，语义翻译仍是遥远目标

## 相关工作与启发

- **VampNet** (García et al., 2023): 本工作的基础架构
- **AVES** (Hagiwara, 2023): 自监督生物声学编码器，WhAM 在分类上对标的上界
- **Sharma et al. (2024b)**: coda 交流的时序 Transformer 分析
- **启发**: 该方法框架可推广到其他动物交流系统的研究

## 评分

⭐⭐⭐⭐ (4/5)
创新性极强且跨学科价值突出，但在声学逼真度和语义理解方面仍有明显差距。
