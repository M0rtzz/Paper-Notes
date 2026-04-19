---
title: >-
  [论文解读] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation
description: >-
  [CVPR 2026][视频理解][选择性音频生成] SelVA 提出了文本条件的选择性视频到音频（V2A）生成任务，通过可学习的补充 token [SUP] 和自监督视频混合策略，使模型能够根据文本提示从多声源视频中仅生成用户指定的目标声音，在音频质量、语义对齐和时间同步上均超越现有方法。
tags:
  - CVPR 2026
  - 视频理解
  - 选择性音频生成
  - 视频到音频
  - 文本条件
  - 跨模态注意力
  - 自监督视频混合
---

# Hear What Matters! Text-conditioned Selective Video-to-Audio Generation

**会议**: CVPR 2026  
**arXiv**: [2512.02650](https://arxiv.org/abs/2512.02650)  
**代码**: https://jnwnlee.github.io/selva-demo/  
**领域**: 视频理解 / 音频生成  
**关键词**: 选择性音频生成, 视频到音频, 文本条件, 跨模态注意力, 自监督视频混合

## 一句话总结

SelVA 提出了文本条件的选择性视频到音频（V2A）生成任务，通过可学习的补充 token [SUP] 和自监督视频混合策略，使模型能够根据文本提示从多声源视频中仅生成用户指定的目标声音，在音频质量、语义对齐和时间同步上均超越现有方法。

## 研究背景与动机

1. **领域现状**：视频到音频（V2A）生成已取得显著进展，MMAudio、ReWaS 等模型可以从视频内容生成时间同步的音频。现有方法通常一次性生成包含所有声源的整体音轨。

2. **现有痛点**：在专业音频制作（Foley）中，声音设计师需要逐轨制作每个声源的音频，然后分别混音。但现有 V2A 模型只能生成单一的混合音轨，即使只需微调一个声音也必须重新合成整个音频，严重影响实际可用性。

3. **核心矛盾**：现有方法使用冻结的视觉编码器提取特征，这些特征包含了视频中所有物体的视觉信息（包括与目标声音无关的信息），导致生成器无法选择性地只产生目标声音。文本在现有方法中仅作为辅助语义补充，未被用作声源的显式选择器。

4. **本文目标** (1) 如何用文本作为显式选择器，从多声源视频中提取仅与目标声音相关的视觉特征？(2) 在没有单声源标注数据的情况下，如何训练选择性生成能力？(3) 如何设计高效的视频编码器微调策略，避免注意力机制中的伪相关性？

5. **切入角度**：受人类听觉选择性注意力的启发——人可以在嘈杂环境中聚焦于特定声源，模型也应该能通过文本指引关注视频中特定的声音来源。此外，作者观察到 ViT 中的 high-norm artifact 问题（注意力集中在无关 token 上），提出用额外 token 来吸收这些伪注意力。

6. **核心 idea**：将文本提示重新定位为视频特征的显式调制器，通过可学习 [SUP] token 抑制无关视觉激活，结合自监督视频混合策略实现无需单声源标注的选择性音频生成。

## 方法详解

### 整体框架

SelVA 由两个主要模块组成：(1) 文本条件视频编码器 $\mathcal{F}$，和 (2) 多模态条件选择性音频生成器 $\mathcal{G}$。给定视频 $V$、文本描述 $T_i$，模型目标是 $A_i = \mathcal{G}(\mathcal{F}(V, \mathbf{t}_i), \mathbf{t}_i)$，即仅生成文本描述对应的声音。训练采用两阶段策略：第一阶段用 teacher-student 蒸馏训练视频编码器学会根据文本选择性提取特征；第二阶段冻结视频编码器，训练音频生成器。

### 关键设计

1. **文本引导的跨注意力块（Cross-Attention Block）**:

    - 功能：在冻结的视频编码器（Synchformer）中插入轻量跨注意力层，使视觉特征能被文本语义调制
    - 核心思路：在 Synchformer 的空间-时间注意力块之后插入跨注意力层，以视频隐藏向量 $\mathbf{h_v}$ 为 Query，文本嵌入 $\bar{\mathbf{t}}$ 为 Key/Value 进行交叉注意力 $\mathbf{h_{vt}} = \text{Cross-Attn}(Q=\mathbf{h_v}, K=\bar{\mathbf{t}}, V=\bar{\mathbf{t}})$。最终通过可学习的空间注意力池化层获得视频特征 $\mathbf{v}$
    - 设计动机：不同于以往冻结视觉编码器的做法，通过参数高效微调让编码器学会根据文本意图过滤视觉信息，只保留与目标声音相关的语义

2. **可学习补充 token [SUP]**:

    - 功能：抑制跨注意力中的伪相关激活，使注意力集中在文本描述的目标声源上
    - 核心思路：在文本嵌入前拼接 $N=5$ 个可学习 token，即 $\mathbf{t}_{\texttt{[SUP]}} = [\texttt{[SUP]} \oplus \mathbf{t}]$，然后用拼接后的序列作为跨注意力的 Key/Value。这些额外 token 会"吸收"那些原本会错误地落在非目标运动 patch 上的高范数注意力值
    - 设计动机：直接交叉注意力会导致模型仍然关注视频中非目标物体的运动动态（如旁边的猫在动就生成猫叫声）。从可视化中可以看到，没有 [SUP] 时注意力散布在整个画面，加入后注意力精确聚焦于文本描述的目标区域。相比在视觉 token 序列中加额外 token（会增加所有编码器层的计算成本），在文本端加入更高效

3. **自监督视频混合（Video-Mixing）策略**:

    - 功能：在没有单声源标注的情况下构造训练样本，使模型学会从混合视频中选择性提取特定声源的视觉特征
    - 核心思路：随机选择两个视频 $V_{\text{tar}}$ 和 $V_{\text{pair}}$，按比例 $\lambda \sim \text{Beta}(\alpha, \alpha)$ 水平拼接为混合视频 $V = [V_{\text{tar}} \oplus V_{\text{pair}}]$。随机选其中一个视频的音频-文本对作为训练目标。混合概率 0.75，最小目标视频比例 $\lambda > 0.2$
    - 设计动机：实际视频通常包含多个声源但缺乏分离标注。这种混合策略借鉴了音视频分离领域的思路，用自监督方式构造了"伪多声源"样本，让模型学会通过文本线索区分和选择目标视觉区域

### 损失函数 / 训练策略

- **第一阶段（视频编码器训练）**：采用 teacher-student 蒸馏。Teacher 模型（原始 Synchformer）接收单源视频 $V_{\text{tar}}$ 生成伪标签特征 $\mathbf{v}_{\text{tar}}$；Student 模型接收混合视频和目标文本，最小化 L2 损失 $\|\mathcal{F}_S([V_{\text{tar}} \oplus V_{\text{pair}}], \mathbf{t}_{\text{tar}}) - \mathcal{F}_T(V_{\text{tar}})\|^2$
- **第二阶段（生成器训练）**：冻结视频编码器，仅微调 MM-DiT 生成器的视频特征投影层 $W_{\mathbf{v}}$ 和 adaLN 模块 $W_\gamma, W_\beta$，使用条件流匹配（CFM）目标，推理时用 Euler solver 25 步采样，CFG 强度 $\gamma = 4.5$
- 两阶段仅训练各模型 14% 的参数（编码器 19M，生成器 22M）

## 实验关键数据

### 主实验

在自建 VGG-MonoAudio benchmark 上评估，包含 67 个单声源视频、1071 个混合测试对（560 跨类 + 511 类内）：

| 方法 | FAD↓ | KAD↓ | IS↑ | CLAP↑ | IB↑ | DeSync↓ |
|------|------|------|-----|-------|-----|---------|
| ReWaS | 70.4 | 4.937 | 6.23 | 0.200 | 0.2454 | 1.364 |
| VinTAGe | 50.5 | 1.309 | 11.51 | 0.283 | 0.2850 | 1.292 |
| MMAudio-S-16k | 56.7 | 0.874 | 11.54 | 0.270 | 0.3135 | 0.802 |
| VOS+MMAudio | 60.0 | 0.878 | 12.11 | 0.291 | 0.3010 | 0.991 |
| **SelVA** | **51.7** | **0.676** | **13.07** | **0.292** | **0.3251** | **0.721** |

SelVA 在所有关键指标上取得最优或接近最优，特别是在时间同步（DeSync 0.721）和音频质量（KAD 0.676）上显著领先。

### 消融实验

| 配置 | DeSync↓ (Inter) | DeSync↓ (Intra) | 说明 |
|------|-----------------|-----------------|------|
| SelVA (完整) | 0.721 | 0.639 | 完整模型 |
| w/o Video Enc. FT | 0.868 | 0.734 | 去掉编码器微调，时间同步严重下降 |
| w/o V2A Gen. FT | 0.736 | 0.651 | 去掉生成器微调，音频质量下降 |
| w/o [SUP] tokens | 0.756 | 0.676 | 去掉 SUP，时间对齐变差 |
| w/o two-stage | 0.823 | 0.777 | 联合训练，语义和时间对齐均下降 |

### 关键发现

- 视频编码器微调对时间同步贡献最大（去掉后 DeSync 从 0.721 升到 0.868），说明让编码器学会文本引导的特征选择是核心
- [SUP] token 主要提升时间对齐（抑制了对非目标运动的错误跟踪），对音频质量和语义影响较小
- 联合训练（不分两阶段）导致模型"走捷径"——用文本语义替代视觉对应的声音事件，破坏时间同步
- 人类评估中 VOS 基线的 CLAP 分数与 SelVA 接近，但人类感知中 text-audio alignment 显著更低（3.78 vs 4.53），暴露了自动指标的局限

## 亮点与洞察

- **[SUP] token 的设计非常巧妙**：它不是加在视觉序列中增加编码器计算成本，而是加在文本序列中作为 Key/Value 的一部分，让跨注意力中的"注意力垃圾"被这些 token 吸收，计算开销极小但效果显著
- **自监督视频混合策略**使得训练不需要任何单声源标注数据，直接在 VGGSound 这样的 in-the-wild 数据集上训练，可扩展性强
- **两阶段训练的必要性**：将特征提取和声音生成解耦训练，避免了两个模块之间的循环依赖导致的训练不稳定

## 局限与展望

- VGGSound 训练数据噪声较多（含背景声和离屏声），更干净的数据或更好的数据过滤可能显著提升性能
- 文本标签通常是简单的"名词+动词"结构，模型对复杂文本描述的理解能力有限（如区分"男声歌唱"vs"男声打嗝"）
- 视频编码器偶尔无法持续跟踪目标运动变化，导致残留的声音替换问题
- 评估 benchmark VGG-MonoAudio 较小（仅 67 个视频），泛化性有待进一步验证

## 相关工作与启发

- **vs MMAudio**: MMAudio 不使用文本模态控制，SelVA 在其基础上加入文本条件视频编码器，保持了 MMAudio 强大的生成能力同时获得选择性控制
- **vs VOS+MMAudio**: VOS 方法依赖分割模型提供空间掩码，难以处理弥散声音（雨声、风声），且计算开销大；SelVA 仅用文本即可实现更灵活的控制
- **vs ReWaS / VinTAGe**: 这些方法用文本辅助语义但不用文本调制视觉特征，时间对齐能力明显不足

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个用纯文本做显式声源选择的 V2A 方法，[SUP] token 设计有新意
- 实验充分度: ⭐⭐⭐⭐ 定量/定性/人类评估/消融齐全，但 benchmark 规模偏小
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详尽，图表丰富
- 价值: ⭐⭐⭐⭐ 解决了实际音频制作中的真实需求，有较好的应用前景

<!-- RELATED:START -->

## 相关论文

- [GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals](goal_force_teaching_video_models_to_accomplish_physics-conditioned_goals.md)
- [CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [StreamReady: Learning What to Answer and When in Long Streaming Videos](streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)
- [Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](../../CVPR2025/video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)

<!-- RELATED:END -->
