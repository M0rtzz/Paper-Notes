---
title: >-
  [论文解读] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text
description: >-
  [人体理解] 构建大规模说唱数据集 RapVerse 并提出统一自回归变换器框架，首次实现从歌词文本同时生成连贯的歌声和全身3D运动。
tags:
  - 人体理解
---

# RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2405.20336](https://arxiv.org/abs/2405.20336)
- **代码**: [项目页面](https://jiabenchen.github.io/RapVerse/)
- **领域**: 人体理解
- **关键词**: 文本到动作, 歌声合成, 全身运动生成, 多模态生成, 自回归模型

## 一句话总结

构建大规模说唱数据集 RapVerse 并提出统一自回归变换器框架，首次实现从歌词文本同时生成连贯的歌声和全身3D运动。

## 研究背景与动机

多模态内容生成领域在各单一模态上（文本到音乐、文本到语音、文本到动作、音频到动作）取得了显著进展，但这些工作各自为政，独立处理每种模态。然而：

**声音和动作内在耦合**：心理学证据表明人类的声音和动作生成高度相关耦合。统一生成系统允许更丰富的情感表达，一种模态的生成可指导和辅助另一种。

**级联系统的累积误差**：文本→语音 + 语音→动作的级联方案在每个阶段累积错误。例如语音合成的误差会导致不准确的面部表情和唇同步。同时需要多个模型的独立训练和推理，计算开销大。

**数据集缺失**：不存在同时包含歌词、歌声和3D全身运动标注的数据集。现有歌声数据集多为中文或规模小；动作数据集仅含文本描述无音频；语音-动作数据集关注语音而非唱歌。

## 方法详解

### RapVerse 数据集

分为两个子集：

**Rap-Vocal 子集**（108.44 小时）：32 位歌手的高质量英文说唱歌声，无背景音乐。管线：Spotdl/Spotipy 爬取 → Spleeter 人声分离 → 响度归一化 → 歌词清洗和时间戳对齐 → 切分为 10-20 秒片段。

**Rap-Motion 子集**（26.8 小时）：说唱表演视频 + SMPL-X 3D 全身网格标注 + 歌声 + 歌词。管线：YouTube 爬取 → YOLO 人体检测 + RAFT 运动幅度评估过滤 → Motion-X 管线提取 SMPL-X 参数。运动表示 $\mathcal{M} = \{\mathcal{M}_f, \mathcal{M}_b, \mathcal{M}_h, \zeta, \epsilon\}$，分别为颚部、身体、手部姿态、面部表情和全局平移。

### 运动 VQ-VAE 分词器

采用组合式 VQ-VAE，分别为面部、身体和手部构建三个独立分词器，将连续运动序列编码为离散 token。量化码本查找：

$$Q(z^{\mathcal{M}}; \mathcal{C}^{\mathcal{M}}) = \arg\min_k \|z^{\mathcal{M}} - c_k^{\mathcal{M}}\|_2$$

三个独立分词器的设计避免了单一 VQ-VAE 对全身动态的表达力不足，特别是面部表情——歌手表演中面部动作占主导，单一 VQ-VAE 会损害面部细节。

### Vocal2unit 音频分词器

基于自监督语音重合成框架，包含三个编码器和一个声码器：

- **语义编码器**：预训练 HuBERT + K-means 聚类生成离散语义单元 $z^{\mathcal{S}} = \{z_i^{\mathcal{S}}\}_{i=1}^{L_s}$
- **F0 编码器**：YAAPT 算法提取基频 → VQ-VAE 编码为离散音高 token $z^{\mathcal{P}}$
- **歌手编码器**：从梅尔频谱提取 256 维歌手嵌入 $z^{\mathcal{I}}$，仅依赖歌手身份
- **声码器**：修改版 HiFi-GAN，以语义、音高 token 和歌手嵌入为输入重建波形

### 自回归统一生成

使用 T5-Tokenizer 将歌词转为文本 token，三种模态统一到 token 空间。大型 Text-Motion-Audio 基础模型（decoder-only 时序变换器）执行下一 token 预测：

$$-\sum_{i=1}^{\mathcal{N}^{\mathcal{X}}} \log p_\delta(t_i^{\mathcal{X}}; h^{\mathcal{L}})$$

**多模态 Token 组织**：交错排列。运动 token：$\mathcal{T}^{\mathcal{M}} = \{t_1^{\mathcal{M}_f}, t_1^{\mathcal{M}_b}, t_1^{\mathcal{M}_h}, t_2^{\mathcal{M}_f}, \ldots\}$；音频 token：$\mathcal{T}^{\mathcal{V}} = \{t_1^{\mathcal{V}_h}, t_1^{\mathcal{V}_p}, t_2^{\mathcal{V}_h}, \ldots\}$。

歌声 token 放在运动 token 之前，因为：(1) 歌声与歌词直接相关；(2) 运动 token（如唇动）与前面生成的歌声相关。

## 实验

### 动作生成主实验

| 方法 | FID↓ | DIV↑ | BC↑ | MSE↓ | LVD↓ |
|------|------|------|-----|------|------|
| T2M-GPT (文本→动作) | 23.45 | 11.75 | — | — | — |
| MLD (文本→动作) | 26.34 | 12.15 | — | — | — |
| Talkshow (音频→动作) | 18.23 | 13.14 | 0.482 | 2.05 | 9.20 |
| EMAGE (音频→动作) | 21.18 | 12.65 | 0.488 | 1.96 | 8.45 |
| MotionCraft (音频→动作) | 17.75 | 13.76 | 0.482 | 2.06 | 9.23 |
| 级联系统 | 23.42 | 12.87 | 0.479 | 2.09 | 9.38 |
| **Ours (文本→音频+动作)** | **17.58** | **14.08** | **0.485** | **2.03** | **7.23** |

### 歌声生成实验

| 方法 | MOS↑ |
|------|------|
| Ground Truth | 4.45 ± 0.06 |
| DiffSinger | 3.72 ± 0.12 |
| **Ours** | **3.64 ± 0.15** |
| FastSpeech2 | 3.41 ± 0.18 |

### 消融实验

| 设置 | FID↓ | DIV↑ | LVD↓ |
|------|------|------|------|
| 预训练 LLM | 48.25 | 12.65 | 12.15 |
| 单一运动 token | 19.15 | 12.75 | 10.12 |
| **完整方法** | **17.58** | **14.08** | **7.23** |

### 关键发现

1. **统一生成胜过级联**：联合生成在 FID、DIV、BC 上均优于 DiffSinger + Talkshow 级联系统，同时计算开销更低
2. **歌声质量可竞争专用系统**：虽然同时生成两种模态任务更复杂，歌声质量 MOS 仍接近专用系统 DiffSinger
3. **预训练 LLM 不适合多模态 token**：在语言 token 上预训练的模型迁移到歌声和运动 token 效果极差（FID 48.25 vs 17.58）
4. **组合式 VQ-VAE 关键**：单一 VQ-VAE 导致 LVD 从 7.23 退化到 10.12，面部表达质量严重下降

## 亮点与洞察

1. **开创性任务定义**：首次提出从歌词文本同时生成歌声和全身3D运动，具有高度实际应用价值
2. **精心构建的数据集**：RapVerse 是首个同时含歌词、歌声、全身运动的英文数据集（108小时歌声+27小时运动）
3. **优雅的模态统一**：交错 token 排列 + 统一自回归对所有模态进行建模，简单有效
4. **组合式分词策略**：面部、身体、手部独立 VQ-VAE 平衡了不同部位的表达需求

## 局限性

1. 说唱风格特异性强，对其他音乐类型（如民谣、古典）的泛化性未验证
2. 3D 运动来自单目视频估计（Motion-X 管线），精度低于动捕数据
3. 仅支持英文歌词输入

## 相关工作

- **文本到歌声**：DiffSinger、FastSpeech2、StyleSinger
- **文本到动作**：T2M-GPT、MLD、MotionGPT、HumanTomato
- **音频到动作**：TalkSHOW、EMAGE、MotionCraft
- **VQ-VAE 离散化**：运动量化、HuBERT 语义编码

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 全新任务定义+首个数据集+统一框架
- **技术深度**: ⭐⭐⭐⭐ — 多模态分词和统一建模设计合理
- **实验充分性**: ⭐⭐⭐⭐ — 多维度指标+用户研究+消融完整
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集描述详尽

<!-- RELATED:START -->

## 相关论文

- [SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/human_understanding/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)

<!-- RELATED:END -->
