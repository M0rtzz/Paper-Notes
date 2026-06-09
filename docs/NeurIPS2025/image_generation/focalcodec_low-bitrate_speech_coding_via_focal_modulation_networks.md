---
title: >-
  [论文解读] FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks
description: >-
  [NeurIPS 2025][图像生成][语音编解码] 提出 FocalCodec——基于 Focal Modulation 的低比特率语音编解码器，使用**单个二值码本**将语音压缩至 0.16–0.65 kbps，在语音重合成、语音转换和多项下游任务中达到与多码本 SOTA 方法可比甚至更优的性能。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "语音编解码"
  - "低比特率"
  - "Focal Modulation"
  - "二值量化"
  - "单码本"
  - "语音Token化"
  - "VQ-VAE"
---

# FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks

**会议**: NeurIPS 2025  
**arXiv**: [2502.04465](https://arxiv.org/abs/2502.04465)  
**代码**: [lucadellalib/focalcodec-web](https://lucadellalib.github.io/focalcodec-web/)  
**领域**: image_generation (语音编解码/speech tokenization)  
**关键词**: 语音编解码, 低比特率, Focal Modulation, 二值量化, 单码本, 语音Token化, VQ-VAE  

## 一句话总结

提出 FocalCodec——基于 Focal Modulation 的低比特率语音编解码器，使用**单个二值码本**将语音压缩至 0.16–0.65 kbps，在语音重合成、语音转换和多项下游任务中达到与多码本 SOTA 方法可比甚至更优的性能。

## 研究背景与动机

大语言模型的成功推动了将语音离散化为 token 的研究范式。神经语音编解码器（Neural Audio Codec）是这一流程的核心组件，其输出 token 需同时保留**语义信息**（用于 ASR 等理解任务）和**声学信息**（用于高保真重建与说话人保持）。

现有方法面临的关键痛点：

**声学编解码器**（EnCodec, DAC, WavTokenizer 等）：重建质量好，但多依赖多码本 RVQ，增加下游模型复杂度，且语义信息不足

**语义编解码器**（基于 HuBERT/WavLM + k-means）：语义好但声学细节丢失严重，说话人保真度低

**混合编解码器**（SpeechTokenizer, Mimi, Stable Codec 等）：尝试兼顾但依赖复杂的多码本设计、蒸馏损失或有监督微调

**比特率偏高**：大多数方法需 ≥0.7 kbps 才能获得可接受性能

**核心动机**：能否设计一个**纯自监督、单码本、超低比特率**的编解码器，同时保留足够的语义和声学信息？

## 方法详解

### 整体框架

FocalCodec 基于 VQ-VAE 框架，但在编码器和解码器之间引入了**压缩器-量化器-解压缩器**（Compressor-Quantizer-Decompressor）架构：

```
输入波形 → [WavLM编码器(冻结)] → [压缩器] → [二值球面量化器] → [解压缩器] → [Vocos解码器] → 重建波形
```

各模块设计如下：

**编码器（Encoder）**：采用 WavLM-Large 的前 6 层（冻结）。选择低层特征是因为 WavLM 的低层保留了大量声学信息，同时也含一定语义信息。编码器参数约占总模型的 5 倍于解码器——作者认为强大的编码器比强大的解码器更重要。

**解码器（Decoder）**：采用 Vocos 架构（而非常见的 HiFi-GAN），通过 ConvNeXt 块处理特征并投影为傅里叶系数，用逆 STFT 合成波形。更高效且减少混叠。

### 关键设计：Focal Modulation 压缩块

压缩器中的核心创新是用 **Focal Modulation** 替代传统 Transformer 中的 Self-Attention：

$$\mathbf{y}_i = q(\mathbf{x}_i) \odot h\left(\sum_{\ell=1}^{L+1} \mathbf{z}_i^\ell \odot \mathbf{g}_i^\ell \right)$$

与 Self-Attention 不同，Focal Modulation **先聚合全局上下文再调制局部交互**：
- 通过递增卷积核的深度卷积栈从短程到长程捕捉多粒度依赖
- 最后一层用全局平均池化提供全局信息
- 每层用逐点卷积计算门控向量
- 线性复杂度（vs. Self-Attention 的二次复杂度）
- 具备平移等变性、显式输入依赖等归纳偏置

压缩器通过线性投影或 1D 卷积（可额外做时间降采样）减少维度，并使用 Snake 激活函数捕捉周期性模式。三个变体的时间降采样因子分别为 (1,1,1)、(2,1,1)、(2,2,1)，对应 token 率 50Hz、25Hz、12.5Hz。

### 关键设计：二值球面量化（BSQ）

首次在语音领域成功应用**二值球面量化**（Binary Spherical Quantization），属于无查表量化（Lookup-Free）方法：

1. 对输入向量 $\mathbf{v}$ 做 L2 归一化映射到单位超球面：$\mathbf{u} = \mathbf{v}/\|\mathbf{v}\|_2$
2. 对每维独立做二值量化：$\hat{\mathbf{u}} = \mathrm{sign}(\mathbf{u}) / \sqrt{L}$
3. 隐式码本为 $\mathcal{C} = \{-1/\sqrt{L}, 1/\sqrt{L}\}^L$，码本大小 $|\mathcal{C}| = 2^L$

当 $L=13$ 时码本大小为 8192。优势：
- **无参数**：码本由隐式定义，轻量高效
- **高利用率**：二值瓶颈天然鼓励码本均匀使用（归一化熵 ≈ 99%）
- **量化误差有界**：收敛更快
- **适合生成模型**：码本大小与隐维度绑定，避免大码本导致的生成性能退化

### 损失函数

采用**两阶段解耦训练**：

**第一阶段**（训压缩器+量化器+解压缩器，编码器冻结）：
- 重建损失：$\mathcal{L}_\text{recon} = \|\hat{\mathbf{z}} - \mathbf{z}\|_2^2$（重建编码器连续特征）
- 熵损失：鼓励预测自信 + 码本均匀利用
- 无需 commitment loss（BSQ 量化误差有界）

**第二阶段**（训解码器，可与第一阶段并行）：
- 对抗损失（Hinge formulation）
- L1 Log-Mel 谱重建损失
- 特征匹配损失
- 判别器：多周期判别器 + 多尺度判别器（HiFi-GAN 风格）

解耦训练的关键好处：若端到端训练不加约束，重建损失会偏向声学特征而丢失语义。

## 实验关键数据

### 主实验：语音重合成（Table 2）

在 LibriSpeech test-clean 上的核心结果：

| 模型 | 比特率(kbps) | UTMOS↑ | dWER↓ | Sim↑ | 码本利用率↑ | RTF↑ |
|------|-------------|--------|-------|------|-----------|------|
| BigCodec | 1.04 | 4.11 | 2.55 | 98.5 | 100% | 22 |
| Stable Codec | 0.70 | 4.32 | 4.97 | 94.7 | 98.5% | 103 |
| **FocalCodec@50** | **0.65** | 4.05 | **2.18** | 97.4 | **100%** | **185** |
| **FocalCodec@25** | **0.33** | 4.14 | 3.30 | 96.3 | 99.8% | 195 |
| **FocalCodec@12.5** | **0.16** | 4.22 | 7.94 | 93.9 | 98.2% | 208 |

**关键发现**：FocalCodec@50 在 0.65 kbps 下取得最低 dWER（2.18），超越 1.04 kbps 的 BigCodec，且推理速度快 8 倍以上。

多语言（MLS 700 条）：FocalCodec@50 在 dWER（12.57）远优于其他模型（仅用英文数据训练），第二名 BigCodec 15.24（但比特率高 60%）。

噪声环境（VoiceBank）：FocalCodec@50 的 dWER 仅 8.08，远低于第二名 Stable Codec（20.32）。

噪声环境（Libri1Mix，更具挑战性）：FocalCodec@50 dWER 27.89、Sim 91.6，仍大幅领先于 BigCodec（dWER 53.26）和 WavTokenizer（dWER 70.10）。

**值得注意**：FocalCodec 的 UTMOS 在低 token 率下反而升高（@12.5 为 4.22 > @50 的 4.05），这是因为降采样带来的平滑效应。但 UTMOS 存在饱和问题，因此 dWER 和 Sim 是更可靠的评估指标。

### 语音转换（Table 3）

one-shot 语音转换（VCTK 数据集，2521 样本）。单码本模型用 k-NN 替换（k=4）实现转换，多码本模型用第一码本（源内容）+ 后续码本（目标说话人）拼接：

| 模型 | 比特率 | UTMOS↑ | dWER↓ | Sim↑ | RTF↑ |
|------|--------|--------|-------|------|------|
| WavLM6-KM | 0.45 | 2.90 | 26.68 | 92.4 | 57 |
| SpeechTokenizer | 1.00 | 1.49 | 20.32 | 81.2 | 33 |
| Mimi | 0.69 | 2.40 | 110.0 | 89.7 | 71 |
| Stable Codec | 0.70 | 3.76 | 27.63 | 71.1 | 65 |
| **FocalCodec@50** | **0.65** | 3.38 | **21.27** | **92.2** | 116 |
| **FocalCodec@25** | **0.33** | 3.40 | 23.59 | **92.6** | 118 |
| **FocalCodec@12.5** | **0.16** | 3.43 | 29.93 | **92.6** | 117 |

FocalCodec 取得最高说话人相似度（92.6），超越了专门设计多码本解纠缠的 SpeechTokenizer（81.2）和 Mimi（89.7）。声学编解码器（EnCodec Sim 72.2、DAC 67.2、BigCodec 68.9）因无法分离说话人与内容信息而表现极差。WavLM6-KM 排名第二（Sim 92.4），因为与 FocalCodec 共享相同编码器，但 dWER 更高。

### 下游任务（Table 4）

**判别任务**（浅层 BiLSTM 下游模型，接近 linear probing）：

| 模型 | 比特率 | ASR WER↓ | SI ER↓ | SER ER↓ |
|------|--------|----------|--------|---------|
| SpeechTokenizer | 1.00 | 14.97 | 2.73 | 41.50 |
| BigCodec | 1.04 | 26.41 | 2.34 | 47.50 |
| WavTokenizer | 0.48 | 35.62 | 2.44 | 49.80 |
| Stable Codec | 0.70 | 16.85 | 16.50 | 46.54 |
| Mimi | 0.69 | 22.98 | 5.43 | 44.70 |
| **FocalCodec@50** | **0.65** | 17.63 | 4.48 | 45.60 |
| **FocalCodec@25** | **0.33** | 21.12 | 6.07 | 46.80 |

- ASR：FocalCodec@50 WER（17.63）仅次于 SpeechTokenizer（14.97，1.5 倍比特率+双码本）和 Stable Codec（16.85，有监督微调）
- SI：FocalCodec@50 ER（4.48）略高于纯声学编解码器（BigCodec 2.34），但远优于同为混合编解码器的 Stable Codec（16.50，语义微调导致声学信息丢失）
- SER：各模型差异不大，FocalCodec@50（45.60）与最佳模型持平
- **WavLM6-KM 的对照**：使用相同编码器但不同设计，ASR WER 19.04、SI ER 高达 22.30——证明 FocalCodec 的压缩器-量化器设计有效保留了声学信息

**生成任务**（Transformer 下游模型）：

| 模型 | SE dWER↓ | SS dWER↓ | TTS dWER↓ | TTS UTMOS↑ |
|------|----------|----------|-----------|-----------|
| SpeechTokenizer | 29.82 | 83.99 | 35.46 | 2.69 |
| BigCodec | 26.68 | 89.24 | 54.43 | 3.43 |
| Stable Codec | 35.57 | 103.00 | 49.28 | 3.19 |
| **FocalCodec@50** | **10.93** | **73.87** | 28.10 | **4.11** |
| **FocalCodec@25** | 14.74 | 99.96 | **16.75** | 4.16 |

- 语音增强（SE）：FocalCodec@50 的 dWER（10.93）大幅领先所有基线（第二名 BigCodec 26.68）
- 语音分离（SS）：FocalCodec@50 dWER（73.87）最优，但绝对性能仍远不够实用——量化不可避免地丢失了分离所需的精细时频信息
- TTS：FocalCodec@25 取得最优 dWER（16.75）和 UTMOS（4.16），因为**更短序列降低了自回归建模难度**——token 率接近文本率使得 next-token prediction 更高效，这一发现对语音 LLM 设计有重要启示

### 消融实验（Table 5）

| 压缩块 | 激活函数 | 量化器 | dWER↓ | Sim↑ |
|--------|---------|--------|-------|------|
| **Focal Modulation** | **Snake** | **BSQ** | **2.54** | **95.7** |
| Focal Modulation | Snake | FSQ | 2.61 | 94.8 |
| Focal Modulation | Snake | LFQ | 2.75 | 95.4 |
| Conformer | Snake | LFQ | 3.58 | 94.3 |
| AMP | Snake | LFQ | 4.52 | 94.3 |
| Linear | Snake | LFQ | 9.37 | 82.5 |

Focal Modulation 和 BSQ 分别是性能最优的压缩块和量化器，替换为 Conformer 或 FSQ 均导致性能显著下降。

## 亮点与洞察

1. **极致简洁的设计哲学**：单码本 + 二值量化 + 纯自监督，无需蒸馏损失或有监督微调，大幅降低下游模型设计复杂度
2. **Focal Modulation 的语音适配**：首次将视觉领域的 Focal Modulation 成功引入语音编解码，线性复杂度 + 多尺度归纳偏置完美匹配语音信号特性
3. **BSQ 在语音领域的首次成功应用**：二值球面量化天然实现高码本利用率（≈100%），解决了传统 VQ 的码本坍塌问题
4. **解耦训练的妙用**：两阶段可并行训练，且避免了端到端训练中语义信息被重建损失"挤掉"的问题
5. **"编码器 > 解码器"的不对称设计**：编码器参数量约为解码器 5 倍，与常见做法相反，强调表征质量优先

## 局限性

1. **低 token 率变体在多语言场景退化明显**：FocalCodec@12.5 在多语言 dWER（54.15）远高于 @50（12.57），极端压缩在多语言泛化上仍有代价
2. **编码器依赖预训练 WavLM**：WavLM-Large 前 6 层作为冻结编码器带来大量参数（约 127M），限制了端侧部署可能性
3. **语音分离任务表现不佳**：SS 任务的绝对性能距实际应用仍远，量化丢失了分离所需的关键信息
4. **仅用 LibriTTS（约 585h 英文）训练**：数据规模和语言多样性有限，可能制约多语言和多风格场景的上限
5. **采样率限制为 16kHz**：不支持 24kHz 等更高采样率，限制了音乐等宽带音频场景的应用

## 相关工作与启发

- **与 BigCodec 的对比**：同为单码本设计，但 FocalCodec 比特率仅为其 62%（0.65 vs 1.04 kbps），dWER 更低（2.18 vs 2.55），推理速度快 8 倍（RTF 185 vs 22）。核心差异在于 Focal Modulation + BSQ vs 传统 Conformer + VQ。BigCodec 在 SI 上略优（ER 2.34 vs 4.48），但 ASR 差距很大（WER 26.41 vs 17.63），说明纯重建目标难以保留语义
- **与 Stable Codec 的对比**：Stable Codec 用有监督音素微调增强语义，代价是说话人识别能力骤降（SI ER 16.5%）；FocalCodec 纯自监督即达到更好的语义-声学平衡。两者都用双码本/单码本+低比特率路线，但 Stable Codec 需要 force-aligned phoneme 数据
- **与 SpeechTokenizer / Mimi 的对比**：这两个混合编解码器显式地将语义信息蒸馏到第一个码本、声学信息放后续码本，但在语音转换中反而不如 FocalCodec（Sim 81.2/89.7 vs 92.6），说明 FocalCodec 的 WavLM 特征 + k-NN 替换方案天然具备良好的说话人-内容解纠缠
- **与 SemantiCodec 的对比**：SemantiCodec 同为 0.65 kbps 混合编解码器，但使用双编码器+双码本、1033M 参数、RTF 仅 0.62（非实时），FocalCodec 参数量仅为其 14%（142M）且实时性好 300 倍
- **启发**：(1) 二值量化 + 强编码器的范式可推广到其他模态（视觉 token 化、多模态 LLM）；(2) TTS 任务中低 token 率反而更优的发现，暗示语音 LLM 应追求更紧凑的表征而非更高保真的重建；(3) 解耦训练策略（先训 tokenizer 再训 decoder）可作为通用范式应用于需要同时保留多种信息的编解码任务

## 评分

- **新颖性**: ★★★★☆ — Focal Modulation 和 BSQ 在语音编解码中的首次结合，设计简洁但有效
- **技术质量**: ★★★★★ — 实验极为全面（重合成×4数据集 + 语音转换 + 3判别任务 + 3生成任务 + 消融），定量+主观评估
- **实用价值**: ★★★★☆ — 单码本低比特率天然适配语音LLM流水线，但WavLM依赖限制端侧部署
- **写作质量**: ★★★★☆ — 结构清晰、baseline充分，图表丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](../../AAAI2026/image_generation/pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)
- [\[ICML 2025\] Label-Efficient Hyperspectral Image Classification via Spectral FiLM Modulation of Low-Level Pretrained Diffusion Features](../../ICML2025/image_generation/label-efficient_hyperspectral_image_classification_via_spectral_film_modulation_.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](../../ICCV2025/image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)

</div>

<!-- RELATED:END -->
