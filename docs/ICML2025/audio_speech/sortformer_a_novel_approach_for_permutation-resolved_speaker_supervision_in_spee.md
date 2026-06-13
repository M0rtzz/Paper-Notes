---
title: >-
  [论文解读] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems
description: >-
  [ICML 2025][音频/语音][说话人日志] 提出 Sortformer——一个基于编码器的说话人日志模型，通过 Sort Loss 按说话人到达时间排序来解决排列问题，替代或辅助传统的排列不变损失（PIL），并设计正弦核函数将说话人标签注入 ASR 编码器，使多说话人 ASR 训练可直接使用标准交叉熵损失，在 LibriSpeechMix 上实现 2-mix/3-mix 相对误差降低 30%/25%。
tags:
  - "ICML 2025"
  - "音频/语音"
  - "说话人日志"
  - "多说话人ASR"
  - "排列不变损失"
  - "到达时间排序"
  - "Sort Loss"
  - "端到端日志"
---

# Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems

**会议**: ICML 2025  
**arXiv**: [2409.06656](https://arxiv.org/abs/2409.06656)  
**代码**: [NVIDIA NeMo Framework](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)  
**领域**: 语音/说话人日志  
**关键词**: 说话人日志, 多说话人ASR, 排列不变损失, 到达时间排序, Sort Loss, 端到端日志

## 一句话总结

提出 Sortformer——一个基于编码器的说话人日志模型，通过 Sort Loss 按说话人到达时间排序来解决排列问题，替代或辅助传统的排列不变损失（PIL），并设计正弦核函数将说话人标签注入 ASR 编码器，使多说话人 ASR 训练可直接使用标准交叉熵损失，在 LibriSpeechMix 上实现 2-mix/3-mix 相对误差降低 30%/25%。

## 研究背景与动机

**多说话人 ASR 的核心挑战**：随着 ASR 在工业场景（会议转录、对话分析等）中的广泛部署，越来越多的应用要求识别结果带有说话人标注。然而，说话人日志（diarization）与 ASR 模型的整合长期面临数据稀缺、领域适配困难和排列匹配等关键挑战。

**排列不变损失（PIL）的局限**：现有端到端日志系统（如 EEND-SA、EEND-EDA）依赖 PIL 来解决说话人标签排列问题。PIL 需要遍历所有排列组合寻找最小损失，这带来两个问题：(1) 计算复杂度随说话人数量呈阶乘增长；(2) 需要在模型输出层使用专化的损失函数，限制了与多任务 ASR 系统的集成。

**端到端 vs. 级联系统的差距**：端到端多说话人 ASR 系统（如 SOT）虽然结构简洁，但由于缺乏足够的带说话人标注数据，性能仍然落后于级联系统。级联系统虽强但各模块需要针对具体领域分别调优，部署成本高。

**本文核心动机**：能否设计一种排列解析方法，使多说话人 ASR 训练与单说话人 ASR 训练在损失函数层面完全一致（即标准交叉熵），从而实现"即插即用"的说话人标注能力？

## 方法详解

### 整体框架

Sortformer 框架包含两个核心组件：

1. **Sortformer 日志模型**：基于 Transformer 编码器的端到端说话人日志模型，通过 Sort Loss 学习按到达时间顺序生成说话人标签，输出帧级说话人存在概率矩阵 $\mathbf{P} \in \mathbb{R}^{K \times T}$
2. **多说话人 ASR 系统（MS-ASR）**：将 Sortformer 的输出通过正弦核函数注入到预训练 ASR 模型的编码器状态中，实现说话人信息的可微传递

整体数据流为：音频 → NEST 编码器提取帧级特征 → Sortformer 生成排列解析的说话人概率 → 正弦核将说话人信息编码到 ASR 编码器状态 → ASR 解码器生成带说话人标记的转录文本。

### 关键设计

#### 1. 多标签二值分类建模

Sortformer 将说话人日志建模为帧级多标签二值分类问题。给定 $T$ 帧 $D$ 维嵌入序列 $\{\mathbf{x}_t\}_{t=1}^T$，模型输出 $K$ 个说话人在每帧的存在概率。关键假设是各说话人在给定特征下条件独立，因此使用 Sigmoid 而非 Softmax 作为输出激活函数：

$$P(\xi_1, \ldots, \xi_T \mid \mathbf{x}_1, \ldots, \mathbf{x}_T) = \prod_{k=1}^{K} \prod_{t=1}^{T} P(y_{k,t} \mid \mathbf{x}_1, \ldots, \mathbf{x}_T)$$

这与 EEND 系列使用 Softmax 的设计形成对比，允许处理重叠语音。

#### 2. Sort Loss —— 按到达时间排序的损失函数

Sort Loss 的核心创新在于引入到达时间排序函数 $\eta$：对于每个说话人 $k$，定义其到达时间为首个语音段的起始帧 $\Psi(\mathbf{y}_k) = \min\{t' \mid y_{k,t'} \neq 0\}$。然后将真实标签矩阵按到达时间排序，使最先说话的人对应第 1 行输出，第二个说话的人对应第 2 行输出，以此类推：

$$\mathcal{L}_{\text{Sort}}(\mathbf{Y}, \mathbf{P}) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\text{BCE}}(\mathbf{y}_{\eta(k)}, \mathbf{q}_k)$$

与 PIL 的关键区别：PIL 需要搜索所有 $K!$ 种排列找最优匹配（$O(K!)$ 复杂度），而 Sort Loss 通过确定性排序直接建立标签-输出对应关系（$O(K \log K)$ 复杂度）。

#### 3. 混合损失（Hybrid Loss）

单独使用 Sort Loss 时，到达时间估计可能不准确（尤其说话人较多时）。因此提出混合损失：

$$\mathcal{L}_{\text{hybrid}} = \alpha \cdot \mathcal{L}_{\text{Sort}} + (1 - \alpha) \cdot \mathcal{L}_{\text{PIL}}$$

其中 $\alpha = 0.5$ 为经验权重。PIL 提供排列鲁棒性，Sort Loss 提供排序约束，二者互补。

#### 4. 位置编码的必要性

与 EEND-SA 和 EEND-EDA 不同，Sortformer 必须使用位置编码。原因是 Transformer 的多头自注意力机制在无位置编码时具有排列等变性——输入顺序的排列不会改变输出。而 Sortformer 的核心目标是学习到达时间排序，这要求模型感知序列中的位置信息。

#### 5. 正弦核说话人编码

为将日志结果桥接到 ASR，设计了正弦核函数：

$$\tilde{\mathbf{A}} = \frac{\mathbf{A}}{\|\mathbf{A}\|_2} + \mathbf{\Gamma}^T \cdot \mathbf{P}$$

其中 $\mathbf{A}$ 为 ASR 编码器状态，$\mathbf{\Gamma}$ 为 $K$ 个说话人的正弦核矩阵（$\kappa_{k,z} = \sin(2\pi kz / M)$），$\mathbf{P}$ 为 Sortformer 输出的说话人概率。不同说话人对应不同频率的正弦函数，通过加性注入方式将说话人身份编码到 ASR 特征中。

### 训练策略

**两阶段训练**：
- **预训练阶段**：使用 2,030 小时真实数据 + 5,150 小时模拟混合数据训练，模拟数据重叠率 0.12、平均静音率 0.1
- **微调阶段**：仅使用真实数据微调

**架构参数**：基于 L-size NEST 编码器（115M 参数）+ 18 层 Transformer 编码器（hidden size 192）+ 2 层前馈 + 4 个 Sigmoid 输出，总计 123M 参数。

**MS-ASR 训练**：基于 Canary ASR 模型，170M 和 1B 两种规模。170M 版本全参数微调 50K 步（batch size 64）；1B 版本使用 Adapter（仅学习 Adapter 参数，冻结其余），训练 75K 步。优化器为 AdamW，学习率 $3 \times 10^{-4}$，权重衰减 $10^{-3}$，逆平方根退火，warmup 2,500 步。

**排列解析的转录格式**：提出 Sorted Serialized Transcript（SST），将说话人标记（如 `<spk0>`, `<spk1>`）按到达时间排序插入转录文本中。支持词级和段级两种粒度（词级更优）。训练时直接使用标准交叉熵损失，无需额外的排列处理。

## 实验关键数据

### 主结果：说话人日志性能（DER%）

| 系统 | 参数量 | DH3 (≤4spk, 0s collar) | CALLHOME-2spk (0.25s) | CALLHOME-3spk (0.25s) | CALLHOME-4spk (0.25s) | CH109 (2spk, 0.25s) |
|:---|:---|:---|:---|:---|:---|:---|
| EEND-EDA | 6.4M | 15.55 | 7.83 | 12.29 | 17.59 | - |
| AED-EEND | 11.6M | - | 6.18 | 11.51 | 18.44 | - |
| WavLM-L+EEND-VC† | 317M | - | 6.46 | 10.69 | 11.84 | - |
| Sortformer-PIL | 123M | 17.04 | 6.94 | 10.30 | 17.52 | 6.89 |
| Sortformer-Sort-Loss | 123M | 17.10 | 6.52 | 10.36 | 17.40 | 10.85 |
| **Sortformer-Hybrid** | **123M** | **14.76** | **5.87** | **8.46** | **12.59** | **6.86** |

> Hybrid Loss 在几乎所有数据集上取得最优，DH3 上 DER 从 PIL 的 17.04 降至 14.76（相对降低 13.4%），CALLHOME-4spk 从 17.52 降至 12.59（相对降低 28.1%）。

### 消融实验：多说话人 ASR 组件贡献（AMI-test & CH109）

| 系统 | 说话人监督 | 训练/推理说话人 | Adapter | AMI-test WER | AMI-test cpWER | CH109 WER | CH109 cpWER |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Baseline (Canary-170M) | 无 | - | - | 26.93% | - | 21.81% | - |
| Sys1 (无监督) | 无 | - | - | 19.67% | 32.94% | 18.57% | 24.80% |
| Sys2 (Sortformer冻结) | Sortformer | 冻结 | - | 20.08% | 28.17% | 18.65% | 22.22% |
| Sys4 (GT训练) | GT→Sortformer推理 | - | - | 19.48% | 26.83% | 18.74% | 24.39% |
| **Sys6 (1B+Adapter+词级)** | **Sortformer** | **冻结** | **256** | **18.04%** | **26.71%** | **16.46%** | **21.45%** |

> Sortformer 监督 + Adapter + 词级目标的 Sys6 在所有指标上表现最优。cpWER 从无监督的 32.94% 降至 26.71%。

### LibriSpeechMix 对比结果

| 系统 | 参数量 | 说话人监督 | 1-mix WER | 2-mix WER | 3-mix WER |
|:---|:---|:---|:---|:---|:---|
| SOT-ASR | 135.6M | ✗ | 4.6 | 11.2 | 24.0 |
| DOM-SOT | 33M | ✗ | 5.17 | 5.56 | 9.96 |
| MT-LLM | 8.4B | ✓ | 2.3 | 5.2 | 10.2 |
| MS-Canary (无Sortformer) | 170M | ✗ | 2.74 | 6.55 | 12.14 |
| **Sortformer-MS-Canary** | **293M** | **✓** | **2.26** | **4.61** | **9.05** |

> Sortformer-MS-Canary 仅用 293M 参数，在 2-mix 和 3-mix 上超越了 8.4B 参数的 MT-LLM，且 Sortformer 集成仅增加 0.78% 的推理时间开销。

## 亮点与洞察

1. **排列问题的优雅解法**：Sort Loss 将 $O(K!)$ 的排列搜索简化为 $O(K\log K)$ 的确定性排序，同时保持了与 PIL 可比甚至更优的性能。这是对端到端日志领域十年来依赖 PIL 范式的一次根本性突破。

2. **训练范式统一**：通过 Sorted Serialized Transcript + 正弦核注入，多说话人 ASR 训练可完全使用标准交叉熵损失，与单说话人 ASR 训练流程一致——这极大降低了多说话人能力的集成门槛。

3. **混合损失的互补效应**：Sort Loss 提供全局排序约束但对到达时间估计敏感，PIL 提供排列鲁棒性但缺乏排序感知，二者结合显著优于单独使用任一损失。

4. **参数效率极高**：293M 的 Sortformer-MS-Canary 在 LibriSpeechMix 上超越了 8.4B 的 MT-LLM，参数量仅为后者的 3.5%，且推理时间几乎无增加（+0.78%）。

5. **位置编码的理论洞察**：作者指出 Transformer 无位置编码时具有排列等变性，这解释了为什么 EEND 系列不需要位置编码（排列无关）而 Sortformer 需要（排列有关）。

## 局限性

1. **说话人数量上限**：当前模型固定支持最多 4 个说话人，尚未处理开放说话人数量的场景。虽然可以扩展输出头数量，但对更多说话人的 Sort Loss 稳定性未经验证。

2. **到达时间排序的脆弱性**：当多个说话人几乎同时开始说话时，到达时间排序的确定性可能受影响，单独使用 Sort Loss 时性能下降，需要 PIL 作为补充。

3. **流式推理未验证**：当前模型需要完整的 90 秒音频段才能工作，尚未适配流式/在线日志场景。

4. **对长音频的可扩展性**：训练样本限制在 90 秒，对于多小时会议录音需要分段处理后合并，分段边界处的说话人一致性问题未讨论。

5. **正弦核设计缺乏理论支撑**：选择正弦函数作为说话人核更多是工程直觉（类比位置编码），缺乏理论分析说明为何优于其他编码方案（如可学习嵌入）。

## 相关工作

- **端到端日志**：EEND-SA（Fujita et al., 2019）首次将 PIL 引入帧级日志；EEND-EDA（Horiguchi et al., 2020）使用 encoder-decoder 生成 attractor 处理可变说话人数；AED-EEND（Chen et al., 2024）引入注意力增强。本文的 Sortformer 在同一端到端框架下提出 Sort Loss 作为 PIL 的替代/补充。
- **多说话人 ASR**：SOT（Kanda et al., 2020b）通过序列化输出解决排列问题，t-SOT（Kanda et al., 2022a）扩展到流式场景，DOM-SOT（Shi et al., 2024）使用主导性排序。本文的 SST（Sorted Serialized Transcript）与 SOT 概念相似但使用排序后的说话人索引而非说话人切换标记。
- **级联系统**：TS-VAD（Medennikov et al., 2020a）和 CHiME 挑战获胜系统（Cornell et al., 2023）展示了级联系统的强大性能但部署困难。本文旨在让端到端系统接近级联系统的性能同时保持部署便利性。
- **LLM 集成**：MT-LLM（Meng et al., 2025）使用 8.4B 参数多任务 LLM 处理多说话人场景。本文则展示了更轻量的方案也能达到可比或更优性能。

## 评分

| 维度 | 评分 (1-5) | 说明 |
|:---|:---|:---|
| 创新性 | ⭐⭐⭐⭐ | Sort Loss 提出了排列问题的全新解法，从 PIL 的搜索范式转向确定性排序范式 |
| 理论性 | ⭐⭐⭐⭐ | 排列等变性分析、混合损失设计有扎实的数学支撑 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 3个日志数据集 + 2个ASR数据集，消融全面，对比系统覆盖广 |
| 工程价值 | ⭐⭐⭐⭐⭐ | 代码开源（NeMo），预训练模型公开，训练开销极小（+0.22%~2.26%） |
| 写作质量 | ⭐⭐⭐⭐ | 数学推导清晰，图表丰富，但部分符号较繁重 |
| 综合 | ⭐⭐⭐⭐ | 对说话人日志领域的重要贡献，Sort Loss 思想简洁有效，有望成为该方向新基线 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)
- [\[ICML 2025\] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](../../ACL2025/audio_speech/different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](../../ACL2025/audio_speech/zero-shot_text-to-speech_for_vietnamese.md)
- [\[CVPR 2025\] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](../../CVPR2025/audio_speech/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)

</div>

<!-- RELATED:END -->
