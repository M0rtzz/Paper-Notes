---
title: >-
  [论文解读] LeVo: High-Quality Song Generation with Multi-Preference Alignment
description: >-
  [NeurIPS 2025][语音][歌曲生成] 提出 LeVo 歌曲生成框架，通过语言模型并行建模混合 token 和双轨 token 以兼顾人声-伴奏和谐性和音质，并创新性地引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力。
tags:
  - NeurIPS 2025
  - 语音
  - 歌曲生成
  - 语言模型
  - 多偏好对齐
  - DPO
  - 音乐编解码器
---

# LeVo: High-Quality Song Generation with Multi-Preference Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2506.07520](https://arxiv.org/abs/2506.07520)  
**代码**: [GitHub](https://github.com/tencent-ailab/songgeneration) (有)  
**领域**: 音频与语音生成  
**关键词**: 歌曲生成, 语言模型, 多偏好对齐, DPO, 音乐编解码器

## 一句话总结

提出 LeVo 歌曲生成框架，通过语言模型并行建模混合 token 和双轨 token 以兼顾人声-伴奏和谐性和音质，并创新性地引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力。

## 研究背景与动机

**领域现状**: LLM 和音频语言模型的进步推动了歌词到歌曲生成的发展。Jukebox 开创了用语言模型预测混合 token 的范式，YuE 引入双轨（人声+伴奏）token 预测，SongGen 探索交错预测模式。工业系统（Suno、Mureka、Udio）已展现出优秀效果但技术不公开。

**现有痛点**:
   - 混合 token 方法词汇有限，无法完整捕捉人声与伴奏的复杂组合，音质受限
   - 双轨 token 方法虽音质高，但独立预测导致人声-伴奏不和谐
   - 交错预测模式序列长度剧增，限制可扩展性和长文本歌曲生成
   - 数据质量参差不齐、音乐注释不可靠，训练效果受制约

**核心矛盾**: 混合 token 保证和谐但音质有限 vs 双轨 token 音质好但和谐性差；高质量数据稀缺限制了音乐性和指令跟随能力。

**本文目标**: 同时优化歌曲生成的音质、音乐性、指令跟随能力和人声-伴奏和谐性。

**切入角度**: 混合 token 和双轨 token 并行建模 + 模块化扩展训练 + 多偏好 DPO 对齐。

**核心 idea**: 用混合 token 掌控全局和谐，用双轨 token 精细化音质，二者并行而不相互干扰。

## 方法详解

### 整体框架

LeVo = **LeLM**（语言模型）+ **Music Codec**（音乐编解码器）

LeLM 由一个大型语言模型（预测混合 token）和一个 AR 解码器（预测双轨 token）组成。Music Codec 基于 MuCodec 扩展，编码器提取 token，解码器（扩散 Transformer + VAE 解码器）将 token 重建为高保真音频。

### 关键设计

1. **LeLM 双层架构**:

    - **语言模型**（decoder-only Transformer）：专注于 next-token prediction 预测混合 token，捕获旋律、节奏、节拍等高层结构信息，确保人声-伴奏和谐：
    $p(\mathbf{S}_m | \mathbf{C}; \boldsymbol{\theta}) = \prod_{t=0}^T p(\mathbf{S}_{m,t} | \mathbf{S}_{m,<t}, \mathbf{C}; \boldsymbol{\theta})$
    - **AR 解码器**（较浅的 decoder-only Transformer）：基于语言模型隐状态并行预测人声和伴奏 token，引入延迟模式（delay pattern）让双轨 token 在预测第 $t$ 步时可以看到语言模型未来 $k$ 步的输出：
    $p(\mathbf{S}_v, \mathbf{S}_a | \mathbf{C}; \boldsymbol{\theta}) = \prod_{t=0}^{T-k} p(\mathbf{S}_{v,t}, \mathbf{S}_{a,t} | \mathbf{S}_{v,<t}, \mathbf{S}_{a,<t}, \mathbf{S}_{m,<t+k}, \mathbf{C}; \boldsymbol{\theta})$

2. **Music Codec 设计**:

    - 编码器 = MuEncoder（提取音乐相关表征）+ RVQ（量化为 token）
    - 解码器 = 扩散 Transformer（从 token 嵌入重建 VAE 特征）+ VAE 解码器（直接生成音频）
    - 两种 token 策略：混合 token（整首歌处理）和双轨 token（先分离人声和伴奏再分别编码）

3. **基于 DPO 的多偏好对齐**:

    - **数据构建**: 用 LLM 生成 20,000 条歌词，每条搭配随机音频提示和文本描述，生成多个样本
    - **策略1 — 歌词对齐偏好**: 用 ASR 计算音素错误（PER），音素错误差距 >40 的构成偏好对
    - **策略2 — 提示一致性偏好**: 用 MuQ-MuLan 模型计算相似度分数，设置阈值筛选偏好对
    - **策略3 — 音乐性偏好**: 三阶段流程——众包排序→训练奖励模型→大规模筛选，最终收集约 60,000 个偏好对
    - **融合方式 — 深度网络插值（DNI）**: 分别在三类偏好数据上 DPO 微调得到三组参数，线性插值合并为最终模型，支持可控系数调节

### 损失函数 / 训练策略

**三阶段训练范式**：

- **阶段1 — 预训练**: 训练语言模型对齐条件输入与混合 token，AR 解码器冻结，音频提示和文本描述各 50% 随机丢弃。带来生成多样性和人声-伴奏和谐性。
- **阶段2 — 模块化扩展训练**: 训练 AR 解码器建模双轨 token，阶段1 所有模块冻结。提升音质和音乐性，不干扰预训练知识。
- **阶段3 — 多偏好对齐**: 用 DPO 损失在多维偏好数据上微调整个 LeLM。显著增强音乐性和指令跟随能力。

模型规模：LeLM ~2B 参数，MuEncoder 300M 参数，扩散模型 ~700M 参数，VAE 150M 参数。数据：200 万首歌曲（约 110,000 小时）。

## 实验关键数据

### 主实验

客观指标对比（开源与闭源系统）：

| 模型 | FAD ↓ | MuQ-T ↑ | MuQ-A ↑ | PER ↓ | CE ↑ | CU ↑ | PQ ↑ |
|------|-------|---------|---------|-------|------|------|------|
| Suno-V4.5 | 2.59 | **0.34** | 0.84 | 21.6 | 7.65 | 7.86 | 8.35 |
| Mureka-O1 | **2.50** | 0.33 | **0.87** | **7.2** | 7.71 | 7.83 | 8.44 |
| YuE | 2.65 | 0.27 | 0.74 | 36.4 | 7.13 | 7.39 | 7.77 |
| SongGen* | 2.68 | 0.25 | 0.80 | 27.5 | 7.63 | 7.79 | 8.37 |
| **LeVo** | 2.68 | **0.34** | 0.83 | **7.2** | **7.78** | **7.90** | **8.46** |

主观 MOS（1-5分）：

| 模型 | OVL | MEL | HAM | SSC | AQ | LYC |
|------|-----|-----|-----|-----|-----|-----|
| Suno-V4.5 | **3.59** | **4.10** | **3.93** | **4.19** | **4.00** | 3.17 |
| Mureka-O1 | 3.42 | 3.88 | 3.89 | 4.14 | 3.87 | 3.32 |
| **LeVo** | 3.42 | 3.93 | 3.90 | 4.09 | 3.96 | **3.38** |
| SongGen* | 2.91 | 3.43 | 3.44 | 3.66 | 3.69 | 2.84 |

### 消融实验

DPO 多偏好对齐策略对比：

| 方法 | FAD ↓ | MuQ-T ↑ | PER ↓ | CE ↑ | PQ ↑ |
|------|-------|---------|-------|------|------|
| w/o DPO | **2.60** | 0.31 | 10.6 | 7.70 | 8.39 |
| 仅策略1 (歌词对齐) | 2.85 | 0.30 | **6.5** | 7.72 | 8.42 |
| 仅策略2 (提示一致) | 2.89 | **0.34** | 10.3 | 7.75 | 8.43 |
| 仅策略3 (音乐性) | 2.63 | 0.32 | 11.2 | **7.78** | 8.45 |
| 混合训练 | 2.75 | 0.33 | 7.5 | 7.76 | 8.43 |
| **LeVo (插值)** | 2.68 | **0.34** | 7.2 | **7.78** | **8.46** |

### 关键发现

- 移除阶段2（模块化扩展训练）或 AR 解码器均导致性能下降，验证了防止混合/双轨 token 干扰的必要性
- 插值融合方法优于简单混合训练，各策略各司其职，且支持平滑过渡
- LeVo 歌词对齐能力 (LYC) 超过 Suno-V4.5 (+0.21 MOS)
- 在开源模型中全面领先，与闭源工业系统竞争力相当

## 亮点与洞察

- **混合+双轨并行建模范式**: 巧妙解决了和谐性与音质的矛盾，不增加序列长度
- **模块化扩展训练策略**: 冻结预训练模块再训练新模块，有效防止知识遗忘和 token 干扰
- **多偏好 DPO 的首次歌曲生成应用**: 三种偏好数据构建策略各有独特设计
- **DNI 参数插值**: 不仅组合多偏好，还提供了可控的偏好权重调节能力
- **工业级对标**: 在学术界开源方法中首次达到与 Suno 等工业系统可比的水平

## 局限与展望

- 音频质量仍受限于训练数据质量参差和离散 token 的信息瓶颈
- 歌曲结构建模（verse/chorus 等）仍不如 Suno 和 Mureka
- 注释伪标签（文本描述由 Qwen2-Audio 生成）的精度有限，影响指令跟随上界
- 未开源训练数据（版权原因），复现受限
- 风格迁移和端到端生成能力可能被滥用于深度伪造

## 相关工作与启发

- **Jukebox** (Dhariwal et al., 2020): 开创混合 token LM 歌曲生成范式
- **YuE** (Yuan et al., 2025): 引入双轨 token 策略
- **SongGen** (Liu et al., 2025): 探索交错预测模式
- **MusicRL** (Cideron et al., 2024): RLHF 在音乐生成中的应用
- **Tango2** (Majumder et al., 2024): DPO 在音频生成中的先驱工作
- DNI (Deep Network Interpolation) 的音乐领域适配是值得关注的技术路线

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 混合+双轨并行建模和多偏好 DPO 的设计具有显著原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 12个对比系统、完整主观+客观评估、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，技术描述详尽
- 价值: ⭐⭐⭐⭐⭐ 开源歌曲生成的里程碑式工作，缩小了学术界与工业界的差距

<!-- RELATED:START -->

## 相关论文

- [A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)
- [Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](../../ACL2025/audio_speech/advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis](deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)
- [MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

<!-- RELATED:END -->
