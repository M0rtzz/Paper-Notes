---
title: >-
  [论文解读] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation
description: >-
  [AAAI 2026][音频/语音][视频到音乐生成] 提出 Diff-V2M，一个基于层次条件扩散 Transformer 的视频到音乐生成框架，通过显式节奏建模（低分辨率 ODF）和层次交叉注意力机制整合情感/语义/节奏特征，在域内和域外数据集上均达到 SOTA。
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "视频到音乐生成"
  - "扩散模型"
  - "节奏建模"
  - "层次交叉注意力"
  - "特征融合"
---

# Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation

**会议**: AAAI 2026  
**arXiv**: [2511.09090](https://arxiv.org/abs/2511.09090)  
**代码**: [Demo](https://Tayjsl97.github.io/Diff-V2M-Demo/)  
**领域**: 图像生成  
**关键词**: 视频到音乐生成, 扩散模型, 节奏建模, 层次交叉注意力, 特征融合

## 一句话总结

提出 Diff-V2M，一个基于层次条件扩散 Transformer 的视频到音乐生成框架，通过显式节奏建模（低分辨率 ODF）和层次交叉注意力机制整合情感/语义/节奏特征，在域内和域外数据集上均达到 SOTA。

## 研究背景与动机

视频到音乐生成（V2M）旨在为视频创作与视觉内容匹配的配乐。随着 YouTube、TikTok 等视频平台和 Sora、Veo 等视频生成模型的兴起，个性化视听内容的需求暴增。现有方法面临两大核心挑战：

**缺乏显式节奏建模**：现有方法通过光流、帧差异或帧级视觉特征隐式学习视觉动态到音乐节奏的映射，无法实现精确的视听时间对齐。基于 LLM 的方法将视频翻译为文本提示，也难以保留细粒度时间动态

**多视角特征融合困难**：视频包含情感、语义、节奏等多种视觉特征，现有渐进式融合增加计算开销，简单拼接又无法捕获特征间依赖关系

## 方法详解

### 整体框架

Diff-V2M 由两个核心模块组成：

- **视觉特征提取模块**：提取语义特征（CLIP）、情感特征（颜色直方图）和节奏特征（低分辨率 ODF + 节奏预测器）
- **条件音乐生成模块**：基于 DiT 的潜在扩散模型，配备层次交叉注意力和时间步感知融合策略

音频采样率 44.1 kHz，视频帧率 1 FPS。使用 Stable Audio Open 的冻结 VAE 编码波形为潜在表示，DiT 从头训练。

### 关键设计

#### 1. 可泛化的节奏表示

系统比较了三种节奏表示方案：

- **低分辨率梅尔频谱图**（$\text{Mel}_{LR}$）：对原始梅尔频谱图归一化+降采样，目标分辨率 $[M, C]$（$C=16$）
- **低分辨率节拍图**（$\text{Tem}_{LR}$）：时间-节拍表示，捕获本地节拍随时间的演化（$B=16$）
- **低分辨率 ODF**（$\text{ODF}_{LR}$）：onset detection function，一维时间序列表征音符起始的强度。对原始 ODF 进行峰值检测后构建秒级向量（$d=1$）

实验证明 **ODF 是最有效的节奏表示**——比梅尔频谱图和节拍图更简洁，直接强调关键节奏事件。

#### 2. 节奏预测器

推理时无法获取音频，因此训练一个 decoder-only Transformer 从视频预测节奏表示。输入包含三部分：

- **CLIP 语义特征** $C_\mathbf{s}$
- **场景转换嵌入** $\mathbf{e} \in \{0,1\}^M$：通过 PySceneDetect 检测场景边界
- **视觉节拍向量** $\mathbf{v} \in \mathbb{R}^M$：帧间差异聚合后的峰值检测结果

三者相加后送入预测器：$\mathbf{X} = C_\mathbf{s} + \text{Embed}(\mathbf{e}) + \text{Linear}(\mathbf{v})$

#### 3. 层次交叉注意力条件模块

采用两层层次结构整合多视角特征：

- **第一层**：情感特征通过交叉注意力塑造整体情感基调
- **第二层**：语义和节奏特征通过**并行交叉注意力**独立处理，防止信息纠缠

#### 4. 时间步感知融合策略

提出两种融合方法自适应平衡语义和节奏分支：

**加权融合**：门控网络根据扩散时间步 $t$ 输出标量权重 $\alpha \in [0,1]$：

$$\alpha = \sigma(f_{\text{gate}}(t)), \quad \mathbf{h}_{\text{fuse}} = \alpha \cdot \mathbf{h}_{\text{sem}} + (1-\alpha) \cdot \mathbf{h}_{\text{rhy}}$$

**FiLM 融合**：基于 Feature-wise Linear Modulation，生成时间步感知的缩放和偏移参数，进行维度级调制：

$$\text{FiLM}_{\text{sem}}(\mathbf{h}_{\text{sem}}) = \gamma_{\text{sem}}^t \cdot \mathbf{h}_{\text{sem}} + \beta_{\text{sem}}^t$$

最佳配置为 **Post-Attention FiLM + Feature Selection** 组合。

### 损失函数 / 训练策略

**扩散训练目标**（v-objective）：

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{t, \mathbf{z}_a^0, \epsilon}\left[\|\epsilon - G(\mathbf{z}_a^t, \mathbf{C}, t)\|_2^2\right]$$

**Scheduled Conditioning 训练策略**：解决训练-推理不一致问题（训练用 GT 节奏，推理用预测节奏）。定义概率调度逐步用预测节奏替代 GT：

$$p_{\text{pred}}(e) = \begin{cases} 0, & e < 10 \\ (e-10)/20, & 10 \leq e < 30 \\ 1, & e \geq 30 \end{cases}$$

节奏预测器与生成器**联合训练**，确保协同适应。优化器为 AdamW（lr=$1\times10^{-4}$），训练 50 epochs，2 块 A100。推理使用 250 步 DDIM，classifier-free guidance scale=3.0。

## 实验关键数据

### 主实验

**表4：与现有方法的定量对比（Mixed Test Set）**

| 方法 | FAD↓ | FD↓ | KL↓ | Den.↑ | Cov.↑ | IB↑ |
|------|------|-----|-----|-------|-------|-----|
| CMT | 8.93 | 47.76 | 1.10 | 0.042 | 0.008 | 0.082 |
| MuMu-LLaMA | 2.84 | 27.12 | 1.25 | 0.107 | 0.090 | 0.145 |
| VidMuse | 3.44 | 21.04 | 0.94 | 0.150 | 0.130 | 0.180 |
| **Diff-V2M** | **1.52** | **10.96** | **0.86** | **0.376** | **0.399** | **0.181** |

**V2M-Bench（域外）**

| 方法 | FAD↓ | FD↓ | IB↑ |
|------|------|-----|-----|
| GVMGen | 2.15 | 21.55 | 0.203 |
| VidMuse | 2.59 | 22.03 | 0.196 |
| **Diff-V2M** | **1.76** | **22.02** | **0.197** |

Diff-V2M 在域内测试集上全面领先，在域外也取得最佳整体表现。主观 A/B 测试中超过所有 baseline。

### 消融实验

**表5：训练策略消融**

| 配置 | FAD↓ | FD↓ | IB↑ |
|------|------|-----|-----|
| **Diff-V2M (完整)** | **1.52** | **10.96** | 0.181 |
| w/o 节奏特征 $C_r$ | 1.83 | 11.95 | 0.189 |
| w/o 情感特征 $C_e$ | 1.68 | 12.89 | 0.181 |
| w/o 视觉节奏 | 2.22 | 13.61 | 0.180 |
| w/o 联合训练 | 1.88 | 13.39 | 0.181 |
| w/o 调度策略 | 1.62 | 10.67 | 0.186 |

### 关键发现

1. **ODF 是最佳节奏表示**：相比梅尔频谱图和节拍图，ODF 更简洁且更有效
2. **Post-Attention FiLM + Feature Selection 是最优融合策略**：FAD 从 2.02 降至 1.52
3. **移除视觉节奏（场景切换+视觉节拍）影响最大**：说明细粒度视觉动态对节奏预测至关重要
4. **Scheduled Conditioning 有效缓解训练-推理差距**
5. 去除节奏特征后 IB 分数反而提升，因为 ImageBind 更偏好语义对齐

## 亮点与洞察

1. **显式节奏建模**：首次系统比较多种节奏表示并证明 ODF 的优越性，为 V2M 领域提供了标准化的节奏建模方案
2. **层次条件设计**：情感→语义+节奏的两层架构既避免了信息纠缠，又允许灵活调节各特征的贡献
3. **时间步感知融合**：FiLM 机制使模型在不同扩散阶段自适应切换语义/节奏的权重——早期关注语义，后期关注节奏
4. **联合训练+调度**：同时训练预测器和生成器+ Scheduled Conditioning 是解决 teacher forcing 问题的优雅方案

## 局限与展望

1. **人体运动场景受限**：依赖场景切割和帧间差异可能遗漏微妙运动线索，在以人为中心的视频上节奏对齐不够精准
2. **缺乏显式风格控制**：无法控制音乐流派、情绪等属性
3. **训练数据规模有限**：BGM909 仅 909 首钢琴曲，SymMV 约 79 小时——更大规模数据可能进一步提升效果
4. 可扩展到更长视频（> 30s）和多乐器编曲场景

## 相关工作与启发

- **VidMuse**：同样生成音频级音乐，但使用离散 token 预测，本文使用连续扩散更自然
- **TiVA**：低分辨率梅尔频谱图作为音频布局的先驱，启发本文探索更多节奏表示
- **Stable Audio Open**：本文基于其 VAE+DiT 架构，证明了将 text-to-music 适配到 V2M 的可行性
- 启发：层次条件注意力+时间步感知融合是一种通用的多条件扩散生成范式

## 评分

- 新颖性: ⭐⭐⭐⭐（显式节奏建模+层次融合）
- 技术深度: ⭐⭐⭐⭐⭐（融合策略和训练调度设计精细）
- 实验完整性: ⭐⭐⭐⭐⭐（主客观评估+域外+消融面面俱到）
- 实用价值: ⭐⭐⭐⭐（通用视频配乐需求大）
- 总体评分: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](../../CVPR2025/audio_speech/enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ICML 2025\] IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](../../ICML2025/audio_speech/impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)

</div>

<!-- RELATED:END -->
