---
title: >-
  [论文解读] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
description: >-
  [CVPR 2026][图像生成][视频配音] 提出DiFlowDubber，基于**离散流匹配(DFM)**的自动视频配音框架，通过两阶段训练（零样本TTS预训练→视频配音适配）将大规模TTS知识迁移到视频驱动配音，设计FaPro模块捕获面部表情-韵律映射、Synchronizer模块实现精准唇音同步。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "视频配音"
  - "离散流匹配"
  - "跨模态对齐"
  - "语音合成"
  - "唇音同步"
---

# DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization

**会议**: CVPR 2026  
**arXiv**: [2603.14267](https://arxiv.org/abs/2603.14267)  
**代码**: [Demo](https://nngocson2002.github.io/projects/diflowdubber)  
**领域**: Image Generation / 多模态语音生成  
**关键词**: 视频配音, 离散流匹配, 跨模态对齐, 语音合成, 唇音同步  

## 一句话总结

提出DiFlowDubber，基于**离散流匹配(DFM)**的自动视频配音框架，通过两阶段训练（零样本TTS预训练→视频配音适配）将大规模TTS知识迁移到视频驱动配音，设计FaPro模块捕获面部表情-韵律映射、Synchronizer模块实现精准唇音同步。

## 研究背景与动机

**领域现状**：视频配音(V2C)在影视制作、多媒体创作和辅助语音技术中有广泛应用。需要在保持说话人音色的同时，生成与唇型同步、承载视觉情感的自然语音。

**现有痛点**：
   - 直接在有限配音数据上训练的方法难以产生富有表现力的韵律和音质
   - 两阶段方法（先TTS预训练再适配）虽能利用大规模语料，但难以同时保证韵律表达和唇音同步

**核心矛盾**：Speaker2Dubber仅适配了音素编码器，未充分利用TTS的韵律和声学建模能力；ProDubber虽引入韵律增强，但依赖时长预测器估计唇型运动，**不受视频实际长度约束**导致同步能力差（LSE分数低）。

**本文目标**：设计一个框架充分利用大规模TTS预训练的韵律、内容和声学建模能力，并适配到视频配音中同时保证发音准确、自然度和精准唇音同步。

**切入角度**：使用FACodec将语音分解为韵律、内容、声学三类离散token，分别建模不同属性——韵律和声学用生成式DFM建模，内容用确定性架构建模。

**核心idea**：两阶段训练管线 + 离散流匹配骨干 + 面部表情→韵律映射 + 双重对齐同步器。

## 方法详解

### 整体框架

DiFlowDubber 要解决的是：给一段无声的说话人视频和一段目标文本，生成既保住说话人音色、又跟唇型严格同步、还能承载脸上情绪的配音。难点在于配音数据太少，直接训不出有表现力的语音；而想借大规模 TTS 的本钱，又得让生成的语音长度被视频"卡死"住，否则唇音对不齐。

它的做法是先用 FACodec 把语音拆成三套离散 token——韵律、内容、声学——再分而治之。整个流程走两个阶段。阶段一是零样本 TTS 预训练，在 LibriTTS 这种大语料上学会"怎么说话"：内容 token 用一个确定性头直接预测，韵律和声学 token 则交给离散流匹配模块（DFPA）来生成，专门负责把语气、语速这类有表现力的东西建出来。阶段二把这套预训练好的 TTS 适配到配音：FaPro 从人脸抽出全局韵律先验，Synchronizer 把文本、视频、语音三个模态的时间轴对齐，CCTA 保住从 TTS 迁过来的内容知识不跑偏，而 DFPA 这时被改造成"看着视频条件去生成"。

### 关键设计

**1. DFPA：用离散流匹配联合生成韵律与声学 token**

配音最难的不是把字念对，而是把语气、轻重、停顿这些有表现力的韵律念活。自回归逐 token 预测在这件事上容易塌成平淡的"念稿腔"。DFPA 换了个思路：把韵律 token 和声学 token 拼在一起当作要去噪的目标 $\mathbf{x}_t$，用一个 DiT 架构的去噪器在离散流匹配框架下迭代恢复，条件里带上内容潜在表示 $\tilde{\mathbf{h}}_c$、说话人嵌入等信息。训练目标是让每一步预测都逼近干净的目标分布：

$$\mathcal{L}_{\text{DFM}} = -\sum_{i \in \mathcal{T}} \mathbb{E}_{t \sim \mathcal{U}[0,1]} [\log p_{1|t}(\mathbf{x}_1^i | \mathbf{x}_t, \mathbf{c}; \theta)]$$

之所以能这么干，前提是 FACodec 已经把语音解耦成可独立控制的属性 token——韵律归韵律、声学归声学，互不打架。在解耦后的空间里用生成式的 DFM 去采样，比自回归更能覆盖韵律的多样变化，念出来不会千篇一律。

**2. FaPro：从人脸把全局韵律先验抠出来**

人说话的情绪、语速、语调，脸上其实都写着——皱眉、张嘴幅度、表情快慢，和语音韵律强相关。FaPro 就是把这条相关性显式接进来。它拿到面部特征 $\mathbf{v}_{\text{face}}$ 后，先上采样把帧率对齐到语音长度，再过 ConvNeXt V2 强化时序建模，最后用 $m$ 层 FFT 迭代，逐层抽出对应各个 RVQ 码本的韵律表示，得到全局韵律先验 $\tilde{\mathbf{z}}_p \in \mathbb{R}^{m \times L \times D}$。这个先验喂给 DFPA，相当于告诉生成器"这段该用什么情绪和语速说"，把视觉里的情感落到了语音的韵律上。

**3. Synchronizer：双重对齐，把唇型和语音的时间轴焊死**

这是冲着 ProDubber 的痛处来的——它靠时长预测器估唇型运动，根本不受视频实际长度约束，结果同步分（LSE）很差，听起来更像 TTS 而不是配音。Synchronizer 改成显式对齐，而且对齐两次。第一次是视频-文本对齐：以唇部特征作 query、音素特征作 key/value 做 cross-attention，再用 MFA 给出的对齐矩阵 $\mathcal{M}_{VT}$ 施加对比损失 $\mathcal{L}_{VT}$，把"哪个音素对应哪段唇动"压实。接着做时长正则化，通过 MAS 把注意力权重转成每个音素该占几帧，按时长复制音素嵌入并上采样到语音长度。但上采样会引入细小错位，于是再做第二次——语音-文本对齐，对上采样后的表示补一层对比对齐 $\mathcal{L}_{ST}$ 修正偏差。最后用 ConvNeXt V2 细化，得到对齐表示 $\mathbf{h}_{\text{sync}} \in \mathbb{R}^{L \times D}$。两道对齐一前一后，把唇型与语音的时序一致性强行钉住。

**4. CCTA：内容一致性时序适配，迁知识又不丢同步**

阶段二要换掉 TTS 里那个不看视频的时长预测，但又不能把预训练学到的语义内容能力一起丢了。CCTA 的办法是用预训练 TTS 权重初始化，冻结投影层和内容头保住内容知识，只把时长预测部分替换成 Synchronizer。为了进一步防止适配过程把内容表示带偏，它加了一条蒸馏损失，约束适配后的特征 $\mathbf{z}_t$ 与原内容特征 $\mathbf{z}_c$ 在方向上保持一致：

$$\mathcal{L}_{\text{distill}} = \frac{1}{B} \sum_{i=1}^{B} [1 - \cos(\phi(\mathbf{z}_t^{(i)}), \phi(\mathbf{z}_c^{(i)}))]$$

这样一来，发音的准确性靠 TTS 老本撑着，唇音同步靠 Synchronizer 新接的视频约束，两边各管一摊、互不削弱。

### 损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_{\text{CTC}} + \lambda_3 \mathcal{L}_{\text{distill}} + \lambda_4 \mathcal{L}_{\text{DFM}}$$

其中对齐项 $\mathcal{L}_{\text{align}} = \lambda_5 \mathcal{L}_{VT} + \lambda_6 \mathcal{L}_{ST}$，由视频-文本与语音-文本两道对比对齐组成。

## 实验关键数据

### 主实验：Chem数据集 (Setting 1.0)

| 方法 | LSE-C↑ | LSE-D↓ | WER↓ | SECS↑ | UTMOS↑ |
|------|--------|--------|------|-------|--------|
| HPMDubbing (CVPR'23) | 7.85 | 7.19 | 16.05 | 85.09 | 2.16 |
| EmoDubber (CVPR'25) | 8.11 | 6.92 | 11.72 | 90.62 | 3.82 |
| ProDubber (CVPR'25) | 2.58 | 12.54 | **9.45** | 72.13 | 3.85 |
| **DiFlowDubber** | **8.31** | **6.73** | 9.65 | 84.59 | **4.02** |

### GRID数据集 (Setting 1.0)

| 方法 | LSE-C↑ | LSE-D↓ | WER↓ | UTMOS↑ |
|------|--------|--------|------|--------|
| EmoDubber (CVPR'25) | 7.12 | 6.82 | 18.53 | 3.83 |
| **DiFlowDubber** | **7.32** | **6.73** | **16.79** | **3.95** |

### 消融实验

| 配置 | LSE-C↑ | WER↓ | UTMOS↑ |
|------|--------|------|--------|
| 完整模型 | **8.31** | **9.65** | **4.02** |
| 无TTS预训练 | 8.17 | 12.04 | 3.53 |
| 无 $\mathcal{L}_{VT}+\mathcal{L}_{ST}$ | 8.26 | 17.15 | 3.90 |
| 无 $\mathcal{L}_{VT}$ | 8.36 | 16.57 | 3.87 |
| 无 $\mathcal{L}_{ST}$ | 8.31 | 12.60 | 3.93 |
| 无 $\mathcal{L}_{\text{distill}}$ | 8.33 | 12.62 | 3.97 |

### 关键发现

1. **同步性**：DiFlowDubber在LSE-C/LSE-D上显著领先，证明Synchronizer有效。ProDubber虽WER低，但同步性极差（LSE-C仅2.58），更像TTS而非配音
2. **TTS预训练关键**：去掉预训练后WER从9.65→12.04，UTMOS从4.02→3.53，证明大规模TTS知识迁移的必要性
3. **双重对齐互补**：$\mathcal{L}_{VT}$ 主要帮助同步，$\mathcal{L}_{ST}$ 主要帮助发音准确性，二者缺一不可
4. MOS-S主观评测中超越所有最新基线，证明真实感知质量最优

## 亮点与洞察

1. **FACodec分解策略精妙**：将语音解耦为韵律/内容/声学三类token，使得不同属性可以用不同建模策略（确定性 vs 生成式），且便于视觉条件注入
2. **Synchronizer的双重保险**：先做视频-文本对齐，上采样后再做语音-文本对齐修正，比单层对齐更鲁棒
3. **首次将离散流匹配(DFM)应用于视频配音**，证明了DFM在跨模态语音生成中的有效性

## 局限与展望

1. **SECS分数不突出**：说话人相似度指标不如一些基线，虽然作者归因于评估偏差（基线模型与SECS计算共享说话人编码器），但仍是改进空间
2. **依赖MFA外部工具**：需要Montreal Forced Aligner提供音素-帧对齐信息作为训练监督
3. **仅在Chem和GRID两个数据集评测**：均为受控环境数据，在野外(in-the-wild)场景的泛化能力未知
4. 生成速度未详细对比——128 NFE的离散流匹配推理可能较慢

## 相关工作与启发

- **FACodec语音分解**：来自NaturalSpeech 3，将语音属性因子化的思路值得在其他跨模态任务中借鉴
- **对比对齐损失**：视频-文本和语音-文本的双重对比对齐，可推广到任何需要多模态时序对齐的场景
- **两阶段预训练-适配范式**：先在大规模数据上学好单模态能力，再通过适配引入新模态条件——这是跨模态生成的有效通用策略

## 评分

⭐⭐⭐⭐ — 系统性强、模块设计合理、同步性显著提升，但评测规模有限且文字较冗长

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)
- [\[CVPR 2026\] Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring](score2instruct_scaling_up_video_quality-centric_instructions_via_automated_dimen.md)
- [\[CVPR 2026\] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)
- [\[ICLR 2026\] Discrete Adjoint Matching](../../ICLR2026/image_generation/discrete_adjoint_matching.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)

</div>

<!-- RELATED:END -->
