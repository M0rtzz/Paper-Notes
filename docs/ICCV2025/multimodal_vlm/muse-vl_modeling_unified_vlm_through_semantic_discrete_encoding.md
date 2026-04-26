---
title: >-
  [论文解读] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding
description: >-
  [ICCV2025][多模态][统一视觉语言模型] 提出语义离散编码 (SDE)，通过在视觉 tokenizer 的量化过程中融入预训练 CLIP 语义特征，使离散视觉 token 与语言 token 天然对齐，在仅用 24M 图文对的情况下实现了统一理解与生成的 SOTA 性能。
tags:
  - ICCV2025
  - 多模态
  - 统一视觉语言模型
  - 视觉离散化
  - 语义编码
  - 视觉理解与生成
  - 自回归
---

# MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding

**会议**: ICCV2025  
**arXiv**: [2411.17762](https://arxiv.org/abs/2411.17762)  
**作者**: Rongchang Xie, Chen Du, Ping Song, Chang Liu (ByteDance)
**代码**: 未开源  
**领域**: multimodal_vlm  
**关键词**: 统一视觉语言模型, 视觉离散化, 语义编码, 视觉理解与生成, 自回归

## 一句话总结

提出语义离散编码 (SDE)，通过在视觉 tokenizer 的量化过程中融入预训练 CLIP 语义特征，使离散视觉 token 与语言 token 天然对齐，在仅用 24M 图文对的情况下实现了统一理解与生成的 SOTA 性能。

## 研究背景与动机

统一多模态大模型（同时支持视觉理解和视觉生成）是当前的热门方向。实现这一目标的关键挑战在于：如何将视觉输入转化为像文本一样的离散 token，使 LLM 能够用统一的 next-token prediction 范式同时处理视觉和语言。

现有方法存在以下问题：

**传统 VQ tokenizer（如 VQGAN）仅关注底层像素重建**，训练目标只有图像重建损失，提取的视觉 token 不包含高层语义信息，难以与语言 token 对齐。这导致使用离散 token 的统一模型（如 Chameleon、Show-o）在理解任务上远逊于专用理解模型

**训练代价高**：由于视觉 token 缺乏语义，Emu3 需要从头训练 8B 模型，Chameleon 需要海量数据，VILA-U 需要 720M 图文对来做对齐

**现有解决方案各有缺陷**：Emu3 通过分别微调两个模型来回避统一建模的困难；Janus 使用分离的编码器增加了模型复杂度；VILA-U 尝试结合对比损失和重建损失，但面临损失冲突导致的收敛困难

核心动机：**能否在视觉离散化阶段就注入语义信息，让视觉 token 天生就与语言对齐？** 这样就能大幅降低后续 VLM 训练的数据需求和对齐难度。

## 方法详解

### 3.1 语义离散编码 (Semantic Discrete Encoding, SDE)

SDE 是本文的核心贡献，基于 VQGAN 架构进行改造，在量化过程中同时考虑语义信息和像素信息。

**架构设计**：在标准的 image encoder + codebook + image decoder 基础上，额外引入两个组件：

- **语义编码器 (Semantic Encoder)**：使用冻结的预训练 SigLIP 模型提取图像的语义特征 $T$。SigLIP 已在大规模图文对上训练过，其特征天然与语言对齐
- **语义解码器 (Semantic Decoder)**：一个 Vision Transformer，从量化后的离散特征中重建语义特征，确保离散编码保留了语义信息

**编码流程**：

1. 图像 $x \in \mathbb{R}^{H \times W \times 3}$ 经图像编码器得到特征 $z = Enc(x)$
2. 冻结的 SigLIP 提取语义特征 $T$
3. **关键创新**：将语义特征与图像特征融合后再量化 $z_q = \text{Quant}(T + z)$
4. 量化后的特征 $z_q$ 分别送入图像解码器重建原图 $\hat{x}$ 和语义解码器重建语义特征 $z_s$

**损失函数**：

$$L_{\text{total}} = L_{\text{sem}} + L_{\text{img}} + L_{\text{vq}}$$

- $L_{\text{sem}} = 1 - \cos(Dec_s(z_q), T)$：语义重建损失，最大化解码语义特征与 SigLIP 特征的余弦相似度
- $L_{\text{img}} = \ell_2(x, \hat{x}) + L_P(x, \hat{x}) + \lambda_G L_G(\hat{x})$：像素重建 + 感知损失 + 对抗损失
- $L_{\text{vq}}$：标准的 VQ commitment loss

**与 VILA-U 的关键区别**：VILA-U 用文本编码器做对比学习，存在损失冲突。SDE 改用 CLIP 的**图像编码器**提取语义特征（已包含与文本对齐的信息），并通过特征加法融合 + 语义重建的方式避免了对比损失和重建损失的直接冲突。

### 3.2 统一视觉语言建模 (MUSE-VL)

基于 SDE tokenizer 构建统一 VLM，结构极其简洁：

- 图像经 SDE 转为长度 $h \times w$ 的离散 token 序列，与文本 token 拼接
- 仅需扩展 LLM 的 embedding 层（增加 32768 个视觉 token ID）
- 用 `<soi>` 和 `<eoi>` 标记视觉 token 的起止
- **无需修改 LLM 结构**，训练目标就是标准的 next-token prediction

**训练分两阶段**：

1. **预训练**：使用图文对数据，对所有 token（视觉 + 文本）计算损失，学习视觉 token 的 embedding 和跨模态对齐
2. **指令微调**：
    - 理解任务：视觉 token 在 prompt 中，仅对 response 文本计算损失
    - 生成任务：文本描述在 prompt 中，对生成的视觉 token 计算损失

**基座 LLM**：支持 Yi-1.5 (9B/34B) 和 Qwen-2.5 (7B/32B)，均可直接适配。

## 实验关键数据

### Tokenizer 对比（相同 LLM + 相同数据）

| Tokenizer | MMBench | SEED | MMStar | AVG |
|-----------|---------|------|--------|-----|
| VQGAN | 32.0 | 42.7 | 29.1 | 34.6 |
| SEED | 63.1 | 57.8 | 39.1 | 53.3 |
| LaVIT | 63.3 | 59.5 | 40.3 | 54.4 |
| **SDE (ours)** | **70.6** | **68.1** | **43.8** | **60.8** |

SDE 比 VQGAN 提升 +26.2% (AVG)，比 SEED/LaVIT 提升 +6.4~7.5%。

### 统一模型理解性能

| 模型 | LLM | Token类型 | MMBench | SEED | MMMU | SQA-I | AVG |
|------|-----|----------|---------|------|------|-------|-----|
| Chameleon | 7B scratch | Discrete | 31.1 | 30.6 | 25.4 | 46.8 | 33.3 |
| Emu3 | 8B scratch | Discrete | 58.5 | 68.2 | 31.6 | 89.2 | 58.8 |
| LLaVA-NeXT | Yi-34B | Continuous | 79.3 | 75.9 | 51.1 | 81.8 | 66.4 |
| **MUSE-VL** | Qwen2.5-7B | Discrete | 72.1 | 69.1 | 39.7 | 93.5 | 63.6 |
| **MUSE-VL** | Qwen2.5-32B | Discrete | **81.8** | 71.0 | 50.1 | **95.0** | **70.1** |

- 7B 模型在 MMBench 上比 Emu3 高 +13.6%，平均高 +4.8%
- 32B 模型超越了包括 LLaVA-NeXT 34B 在内的专用理解模型

### 视觉生成性能

| 模型 | MJHQ-30K FID↓ | GenEval |
|------|---------------|---------|
| SD-XL | 9.55 | 0.55 |
| Show-o | 9.24 | 0.53 |
| Emu3 | - | 0.54 |
| **MUSE-VL** | **7.73** | **0.56** |

### 数据效率

| 模型 | 图文对数量 |
|------|----------|
| Show-o | 35M |
| VILA-U | 720M |
| **MUSE-VL** | **24M** |

仅用 24M 数据即超越使用 720M 数据的 VILA-U。

### 消融实验

| Image Branch | Semantic Branch | rFID | MMBench | SEED | MMStar | AVG |
|:---:|:---:|------|---------|------|--------|-----|
| ✓ | - | 2.63 | 42.8 | 48.5 | 38.1 | 43.1 |
| - | ✓ | - | 72.5 | 67.5 | 48.1 | 62.7 |
| ✓ | ✓ | 2.26 | 72.1 | 69.1 | 49.6 | 63.6 |

语义分支对理解性能的提升是决定性的（+20.5% AVG），同时加入图像分支可保持生成能力且还能略微提升理解表现。

## 亮点与洞察

1. **巧妙避免损失冲突**：不直接用对比损失（VILA-U 的痛点），而是用预训练 CLIP 图像编码器的特征做加法融合 + 语义重建。CLIP 图像特征本身已与文本对齐，无需再做跨模态对比
2. **极致的数据效率**：24M 数据量仅为 VILA-U 的 1/30，核心原因是 SDE 使视觉 token 天然语义对齐，大幅降低了 VLM 阶段的对齐难度
3. **即插即用**：不修改 LLM 架构，任何预训练 LLM 只需扩展 embedding 层即可使用，展现了极好的可扩展性
4. **语义编码的可视化**：论文展示了相同 codebook ID 对应相同语义概念（如猫耳朵、草莓），直观证明了离散编码确实捕获了高层语义
5. **Scale-up 特性良好**：从 7B 到 32B，从 256 分辨率到 384，性能持续提升，符合 scaling law

## 局限与展望

1. **生成分辨率有限**：目前最高 384×384，离散 token 为 27×27 = 729 个，对比扩散模型的 1024 分辨率差距明显
2. **图像重建质量的权衡**：SDE 的 rFID (2.26) 与纯重建方法 LLamaGEN (2.19) 基本持平，但在更高分辨率或更复杂场景下语义约束是否会影响重建细节尚不清楚
3. **依赖冻结的 SigLIP**：语义编码器的质量上限受限于 SigLIP 的能力，若 SigLIP 对某些视觉概念理解不足，SDE 的语义注入也会受限
4. **缺少视频理解/生成**：当前仅支持单图，未扩展到视频模态
5. **Codebook 利用率**：32768 大小的 codebook 实际利用率未报告，过大的 codebook 可能导致部分 code 未被充分训练

## 相关工作与启发

- **Emu3**：同为离散统一模型但从头训练 8B 模型，MUSE-VL 证明了利用预训练 LLM + 语义 tokenizer 是更高效的路径
- **Janus**：使用分离编码器（理解用 CLIP，生成用 VQGAN），MUSE-VL 用单一 SDE tokenizer 统一了两者
- **VILA-U**：最直接的对比，同样尝试在 tokenizer 中注入语义，但用对比损失导致收敛困难。SDE 的特征加法 + 语义重建方案更优雅
- **TokenFlow**：并发工作，使用双 codebook 解耦语义和像素特征。MUSE-VL 用单一 codebook 实现了类似效果
- **BEITv2**：SDE 的语义解码器设计灵感来源，用 VIT 解码器重建预训练特征的思路被成功迁移

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Unified Multimodal Understanding via Byte-Pair Visual Encoding](unified_multimodal_understanding_via_byte-pair_visual_encoding.md)
- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests](visual_interestingness_decoded_how_gpt-4o_mirrors_human_interests.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)

<!-- RELATED:END -->
