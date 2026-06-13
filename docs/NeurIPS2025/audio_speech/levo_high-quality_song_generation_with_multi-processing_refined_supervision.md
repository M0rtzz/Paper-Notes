---
title: >-
  [论文解读] LeVo: High-Quality Song Generation with Multi-Preference Alignment
description: >-
  [音频/语音] LeVo 提出一种基于语言模型的歌曲生成框架，通过并行预测混合 token 和双轨 token 来同时优化人声-伴奏和谐度与音质，并引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力，在学术方法中全面领先且接近工业系统水平。 歌曲生成是 AIGC 领域中极具挑战性的任务，需要同时生成高质量的人声和伴奏…
tags:
  - "音频/语音"
---

# LeVo: High-Quality Song Generation with Multi-Preference Alignment

## 基本信息

| 项目 | 内容 |
|------|------|
| 标题 | LeVo: High-Quality Song Generation with Multi-Preference Alignment |
| 作者 | Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu |
| 机构 | 清华大学深圳国际研究生院、腾讯 AI Lab、武汉大学、上海交通大学、南京大学等 |
| 会议 | NeurIPS 2025 |
| arXiv | [2506.07520](https://arxiv.org/abs/2506.07520) |
| 代码 | [GitHub](https://github.com/tencent-ailab/songgeneration) |

## 一句话总结

LeVo 提出一种基于语言模型的歌曲生成框架，通过并行预测混合 token 和双轨 token 来同时优化人声-伴奏和谐度与音质，并引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力，在学术方法中全面领先且接近工业系统水平。

## 研究背景与动机

歌曲生成是 AIGC 领域中极具挑战性的任务，需要同时生成高质量的人声和伴奏音轨并将两者无缝融合，同时还要保持音乐性和指令跟随能力。现有方法面临以下核心困难：

1. **混合 token 方法的局限**：Jukebox、SongCreator 等方法将人声和伴奏的混合音频作为单一预测目标，有限的词汇表无法充分捕捉人声和伴奏的复杂组合，导致音质较低。
2. **双轨 token 方法的困境**：YuE、SongGen 等方法分别生成人声和伴奏 token，虽然提升了音质，但独立预测难以维持人声-伴奏的和谐性；交错预测模式则大幅增加序列长度，限制了可扩展性。
3. **数据质量问题**：可用的歌曲数据集质量参差不齐、音乐标注不可靠，模型缺乏音乐性的先验知识，也难以准确跟随歌词和提示词等指令。

## 方法详解

### 整体架构

LeVo 由两大组件构成：**LeLM**（语言模型）和 **Music Codec**（音乐编解码器）。

### LeLM：并行建模混合 token 与双轨 token

LeLM 的核心创新在于同时建模两类 token：
- **混合 token（Mixed Tokens）**：编码人声和伴奏的混合音频，捕捉旋律、节奏、速度等高层结构信息，确保人声-伴奏和谐。
- **双轨 token（Dual-Track Tokens）**：分别编码人声和伴奏，捕捉更精细的声学细节以提升音质。

架构上，LeLM 包含：
1. **语言模型**：采用 decoder-only Transformer，执行混合 token 的下一 token 预测任务。
2. **AR 解码器**：参数量远小于主语言模型的 decoder-only Transformer，基于语言模型的隐藏状态并行预测双轨 token。引入延迟模式（delay pattern），使双轨 token 预测时能利用未来 k 步的混合 token 信息，提供更丰富的上下文。

### Music Codec

基于 MuCodec 构建的 48kHz 音乐编解码器：
- **编码器**：MuEncoder + RVQ，将音频离散化为 token。
- **解码器**：扩散 Transformer + VAE 解码器，从 token 嵌入重建高保真音频，速度显著快于基于 Mel 频谱图的方法。

### 基于 DPO 的多偏好对齐

针对音乐生成的多维度需求，提出三种偏好数据构建策略：

1. **歌词对齐偏好（Strategy 1）**：通过 ASR 计算音素错误率，筛选错误数差异大于 40 的样本对。
2. **提示一致性偏好（Strategy 2）**：使用 MuQ-MuLan 模型计算相似度分数，设定阈值筛选胜负对。
3. **音乐性偏好（Strategy 3）**：三阶段流程——众包人工排序 → 训练奖励模型 → 大规模筛选，最终收集约 60,000 对偏好数据。

最终采用基于插值的多偏好对齐方法（受 DNI 启发）：分别在三种偏好数据上微调得到三组参数，再进行线性插值得到最终模型，实现多维度性能的平衡。

### 三阶段训练范式

1. **预训练**：语言模型在大规模音乐数据上学习混合 token 预测，AR 解码器冻结。
2. **模块化扩展训练**：冻结第一阶段模块，训练 AR 解码器学习双轨 token，避免干扰已学知识。
3. **多偏好对齐**：使用 DPO 损失微调整个 LeLM。

## 实验

### 实验设置

- 训练数据：200 万首歌曲（约 11 万小时）
- LeLM 参数量：约 2B；扩散模型：约 700M；VAE：150M
- 对比系统：工业系统（Suno V4.5、Mureka-O1、Haimian）+ 学术系统（YuE、DiffRhythm、ACE-Step、SongGen）

### 客观评估结果

| 模型 | FAD ↓ | MuQ-T ↑ | MuQ-A ↑ | PER ↓ | CE ↑ | CU ↑ | PC ↑ | PQ ↑ |
|------|-------|---------|---------|-------|------|------|------|------|
| Suno-V4.5 | 2.59 | **0.34** | 0.84 | 21.6 | 7.65 | 7.86 | 5.94 | 8.35 |
| Mureka-O1 | **2.50** | 0.33 | **0.87** | **7.2** | 7.71 | 7.83 | **6.39** | 8.44 |
| YuE | 2.65 | 0.27 | 0.74 | 36.4 | 7.13 | 7.39 | 5.90 | 7.77 |
| DiffRhythm | 4.86 | 0.26 | 0.51 | 12.3 | 6.65 | 7.32 | 5.71 | 7.77 |
| ACE-Step | 2.69 | 0.28 | - | 37.1 | 7.37 | 7.52 | 6.26 | 7.85 |
| SongGen* | 2.68 | 0.25 | 0.80 | 27.5 | 7.63 | 7.79 | 5.94 | 8.37 |
| **LeVo** | 2.68 | **0.34** | 0.83 | **7.2** | **7.78** | **7.90** | 6.03 | **8.46** |

LeVo 在 MuQ-T、PER、CE、CU、PQ 五项指标上均为最优或并列最优，指令跟随能力和音乐性感知显著优于所有学术方法。

### 主观评估结果（MOS）

| 模型 | OVL ↑ | MEL ↑ | HAM ↑ | SSC ↑ | AQ ↑ | LYC ↑ |
|------|-------|-------|-------|-------|------|-------|
| Suno-V4.5 | **3.59** | **4.10** | **3.93** | **4.19** | **4.00** | 3.17 |
| Mureka-O1 | 3.42 | 3.88 | 3.89 | 4.14 | 3.87 | 3.32 |
| YuE | 2.45 | 3.04 | 2.94 | 3.53 | 3.08 | 2.41 |
| SongGen* | 2.91 | 3.43 | 3.44 | 3.66 | 3.69 | 2.84 |
| **LeVo** | 3.42 | 3.93 | 3.90 | 4.09 | 3.96 | **3.38** |

LeVo 全面超越所有学术方法，整体质量接近 Suno-V4.5，且在歌词对齐（LYC）维度上比 Suno 高出 0.21 分。

## 亮点

1. **并行双类型 token 建模**：巧妙地同时利用混合 token 保持和谐性、双轨 token 提升音质，通过 AR 解码器和延迟模式实现两者的无干扰并行预测，避免了交错模式的序列长度爆炸问题。
2. **模块化扩展训练**：分阶段冻结与训练策略有效防止了不同阶段知识的相互干扰，设计简洁有效。
3. **多偏好 DPO 对齐**：首次将多偏好 DPO 应用于歌曲生成，三种针对性偏好策略覆盖歌词对齐、提示一致性和音乐性三个维度，插值融合方法比简单混合训练效果更优。
4. **学术方法新标杆**：在所有客观和主观指标上全面超越现有开源方法，多项指标接近甚至超过工业闭源系统。

## 局限性

1. **音质仍受限于离散 token 和训练数据质量**：与最先进工业模型（如 Suno）仍存在差距。
2. **标注依赖伪标签**：文本描述由 Qwen2-Audio 生成，多样性和丰富度受限；歌词提取、结构识别等流程中的误差积累影响指令跟随的精度。
3. **歌曲结构建模不足**：在 SSC（歌曲结构清晰度）维度上仍落后于 Suno 和 Mureka-O1。
4. **伦理风险**：具备风格迁移和端到端生成能力，可能被滥用于深伪音频或虚假信息创作。

## 相关工作

- **音乐生成**：MusicGen、MusicLM、AudioLDM 2 等通过语言模型或扩散模型实现端到端生成；MeLoDy 结合语言模型与扩散模型取得 SOTA 表现。
- **歌曲生成**：Jukebox 开创语言模型预测离散音乐码的范式；YuE 和 SongGen 探索双轨 token 策略；DiffRhythm 采用扩散方法；工业系统（Suno、Mureka、Udio）展示了强大能力但未公开技术细节。
- **音乐领域的 RL/偏好对齐**：BATON 将奖励模型集成到扩散损失中；MusicRL 通过 RLHF 微调 MusicLM；Tango2 半自动构建偏好数据集用于 DPO 训练。LeVo 首次在歌曲生成中实现多维度偏好对齐。

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 创新性 | 8 | 并行双类 token 建模 + 多偏好 DPO 对齐的组合具有显著新颖性 |
| 技术深度 | 8 | 三阶段训练、AR 解码器延迟模式、插值融合等设计扎实 |
| 实验充分性 | 9 | 客观+主观双维度，7 个对比系统，详尽消融实验 |
| 写作质量 | 8 | 结构清晰，动机阐述充分，实验分析到位 |
| 实用价值 | 7 | 代码开源，但 2B 参数模型和 11 万小时数据门槛较高 |
| 总评 | 8 | 歌曲生成领域的强力工作，系统性地解决了多个核心难题，实验令人信服 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](../../CVPR2025/audio_speech/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)

</div>

<!-- RELATED:END -->
