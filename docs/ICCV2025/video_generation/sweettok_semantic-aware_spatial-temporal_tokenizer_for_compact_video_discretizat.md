---
title: >-
  [论文解读] SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization
description: >-
  [ICCV 2025][视频离散化] 提出 SweetTok 视频 tokenizer，通过解耦查询自编码器（DQAE）分离空间和时间信息压缩、运动增强语言码本（MLC）按词性分配码字，在仅使用 25% token 数量的情况下，rFVD 改善 42.8%，gFVD 改善 15.1%，实现压缩率与重建保真度的最佳平衡。
tags:
  - ICCV 2025
  - 视频离散化
  - 空间-时间解耦
  - 向量量化
  - 语言码本
  - 视频生成
---

# SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization

**会议**: ICCV 2025  
**arXiv**: [2412.10443](https://arxiv.org/abs/2412.10443)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频离散化, 空间-时间解耦, 向量量化, 语言码本, 视频生成

## 一句话总结

提出 SweetTok 视频 tokenizer，通过解耦查询自编码器（DQAE）分离空间和时间信息压缩、运动增强语言码本（MLC）按词性分配码字，在仅使用 25% token 数量的情况下，rFVD 改善 42.8%，gFVD 改善 15.1%，实现压缩率与重建保真度的最佳平衡。

## 研究背景与动机

视觉 tokenizer 是现代视觉生成和理解模型的关键组件。当前视频 tokenizer 面临两大核心问题：

**压缩率低**：传统方法基于 2D patch 或 3D tube 产生 token，每个 token 对应特定空间区域，导致空间和时间维度的冗余。例如 OmniTokenizer 需要 5120 个 token 表示一段 17 帧视频。最新的 LARP 通过自适应查询压缩展平的视频 patch 实现高压缩率，但直接展平视频 token 会将空间和时间信息交织在一起，导致学习困难和重建性能下降。

**高压缩伴随细节损失**：为弥补压缩损失，引入预训练语言嵌入作为码本是常见策略，但现有工作主要关注图像模态，忽略了视频中**文本与运动信息**的关系。

**核心洞察**：视频中的静态外观和动态运动具有本质差异——空间冗余和时间冗余的性质不同。应当**解耦**两者分别压缩，而非混为一体。

## 方法详解

### 整体框架

SweetTok 包含两个核心组件：

1. **解耦查询自编码器（DQAE）**：通过独立的空间和时间查询分别压缩空间和时间信息。
2. **运动增强语言码本（MLC）**：基于词性将语言码本分为空间（名词+形容词）和时间（动词+副词）两部分。

### 关键设计

1. **解耦查询自编码器（Decoupled Query AutoEncoder, DQAE）**

   **Patchify 阶段**：给定视频 $x \in \mathbb{R}^{T \times H \times W \times 3}$，选取第一帧作为空间信息参考，剩余 $T-1$ 帧用于时间信息。使用两个不同的 patch 核：
   - 空间：$\mathcal{P}_s$ 形状 $p_h \times p_w$，将第一帧转为 $v_s \in \mathbb{R}^{1 \times 32 \times 32 \times D}$
   - 时间：$\mathcal{P}_t$ 形状 $p_t \times p_h \times p_w$，将后续帧转为 $v_t \in \mathbb{R}^{4 \times 32 \times 32 \times D}$

   **空间 Tokenization**：将第一帧的 patch $v_s$ 通过空间编码器 $\mathcal{E}_{DQAE_s}$ 压缩为 $L_{spatial}=256$ 个可学习空间查询 $\mathbf{Q_s}$：
   $$\mathbf{Z_{Q_s}} = \mathcal{E}_{DQAE_s}(\mathbf{Q_s}, v_s), \quad \tilde{\mathbf{Z}}_{Q_s} = \mathcal{Q}_{MLC}(\mathbf{Z_{Q_s}})$$
   然 后通过空间解码器重建第一帧 patch：$\tilde{v}_s = \mathcal{D}_{DQAE_s}(\mathbf{Q}_{v_s}, \tilde{\mathbf{Z}}_{Q_s})$

   **时间 Tokenization**：利用视频时间维度的大量冗余，采用**帧间残差** $\Delta v_t^i = v_s^i - v_t^i$ 作为时间信息输入。这是因为空间阶段已经重建了第一帧。将残差压缩为 $L_{temporal}=1024$ 个时间查询：
   $$\mathbf{Z_{Q_t}} = \mathcal{E}_{DQAE_t}(\mathbf{Q_t}, \Delta v), \quad \tilde{\mathbf{Z}}_{Q_t} = \mathcal{Q}_{MLC}(\mathbf{Z_{Q_t}})$$

   **解码策略：先空间后时间**。将重建的第一帧 $\tilde{v}_s$ 复制 $t$ 次，与时间量化残差一起送入时间解码器：
   $$\tilde{v} = \mathcal{D}_{DQAE_t}([\tilde{v}_s \| \cdots \| \tilde{v}_s], \tilde{\mathbf{Z}}_{Q_t})$$
   最终通过像素解码器 $\mathcal{D}_{pixel}$ 重建视频。

   **设计动机**：耦合空间时间压缩会增加解码器学习同一像素跨帧运动信息的难度。解耦后，空间查询捕获静态外观，时间查询捕获动态变化，互不干扰。

2. **运动增强语言码本（Motion-enhanced Language Codebook, MLC）**

   **核心思想**：将词典按词性分为四类——名词、形容词（对应空间静态信息）和动词、副词（对应时间运动信息）。

   具体实现：
   - 从数据集视频字幕中提取候选词汇
   - 使用 CLIP 文本编码器提取嵌入构建码本 $C \in \mathbb{R}^{L \times D}$
   - 使用图卷积网络 $\mathcal{F}$ 将 CLIP 嵌入投影到视觉潜空间，图的边由同一字幕中共现的词对构建

   量化时，空间查询从名词+形容词码本中查找最近邻，时间查询从动词+副词码本中查找：
   $$\hat{z}_s = \mathcal{F}(c_i), \quad i = \arg\min_{c_i \in C_{noun} \cup C_{adj}} \|z_s - \mathcal{F}(c_i)\|$$
   $$\hat{z}_t = \mathcal{F}(c_i), \quad i = \arg\min_{c_i \in C_{verb} \cup C_{adv}} \|z_t - \mathcal{F}(c_i)\|$$

   码本规模：空间 10,481（5,078 名词 + 5,403 形容词），时间 11,139（9,267 动词 + 1,872 副词）。

3. **解耦设计的灵活性**

   空间分支可单独在 ImageNet 上微调，获得强图像 tokenizer。同时编码后的 token 自带语义信息，可直接用于 LLM 的 few-shot 推理。

### 损失函数 / 训练策略

重建损失包含四项：$\mathcal{L}_{rec} = \mathcal{L}_{L2} + \mathcal{L}_{Lpips} + \mathcal{L}_{vq} + \mathcal{L}_g$

- $\mathcal{L}_{L2}$：像素级 L2 重建损失
- $\mathcal{L}_{Lpips}$：LPIPS 感知损失
- $\mathcal{L}_{vq}$：向量量化 commitment loss（空间+时间）
- $\mathcal{L}_g$：GAN 对抗损失

训练配置：8 × A100 GPU，batch size 8，1000K iterations，Adam 优化器，余弦学习率调度（max 1e-4, min 1e-5）。

## 实验关键数据

### 主实验

视频重建（UCF-101, 256×256）：

| Tokenizer | #Tokens | rFVD↓ |
|-----------|---------|-------|
| OmniTok | 5120 | 42 |
| LARP-B | 1024 | 64 |
| LARP-L | 1024 | 35 |
| **SweetTok** | **1280** | **20** |
| SweetTok* (无压缩) | 5120 | 11 |

视频生成（UCF-101, class-conditional）：

| Tokenizer | Generator | #Tokens | gFVD↓ |
|-----------|-----------|---------|-------|
| OmniTok | AR, 650M | 5120 | 191 |
| LARP-L | AR, 632M | 1024 | 99 |
| **SweetTok** | **AR, 650M** | **1280** | **84** |
| SweetTok | AR, 1.9B | 1280 | 65 |

图像重建（ImageNet, 256×256, 256 tokens）：

| Tokenizer | rFID↓ |
|-----------|-------|
| TiTok | 1.71 |
| TokenFlow | 1.03 |
| **SweetTok** | **0.73** |

### 消融实验

压缩方法对比：

| 方法 | #Tokens | rFVD↓ | 说明 |
|------|---------|-------|------|
| Vanilla 下采样 | 1280 | 227.65 | 线性插值压缩效果极差 |
| Vanilla 查询 (LARP) | 1024 | 35.15 | 展平后统一压缩 |
| **DQAE（解耦查询）** | **1280** | **20.46** | 解耦空间时间显著更优 |

码本消融：

| 方法 | rFVD↓ | 说明 |
|------|-------|------|
| Baseline (无 LC) | 29.45 | 无语言码本 |
| + LC（普通语言码本） | 24.80 | 语义信息有帮助 |
| **+ MLC（运动增强）** | **20.46** | 运动词汇关键 |
| + Qwen-based MLC | 20.12 | 更复杂LLM边际收益小 |

### 关键发现

- DQAE 相比展平查询方法提升 42%（35.15→20.46），证明解耦压缩的有效性。
- MLC 比普通语言码本额外降低 17.5% rFVD，运动增强时间码本是关键。
- 视觉语义理解：few-shot 图像分类达 90.8%（miniImageNet），视频动作识别达 90.1%（UCF-101），说明编码 token 确实捕获了丰富语义。
- 展现 scaling law：生成器从 650M 扩展到 1.9B 时 gFVD 从 84 降到 65。

## 亮点与洞察

- **解耦压缩的 "分而治之" 思想**：空间和时间信息的冗余特性不同，分别压缩比混合压缩更高效。
- **语言码本按词性分配**：名词形容词→外观，动词副词→运动，这种语言-视觉的结构化对齐非常直觉且有效。
- **一个 tokenizer，多任务通用**：重建、生成、图像处理、few-shot 理解全部覆盖。
- **帧间残差的巧妙利用**：时间 tokenization 使用残差而非原始帧，天然去除时间冗余。

## 局限与展望

- 空间 token 数 (256) 和时间 token 数 (1024) 的比例固定，不同视频内容应有不同最优分配。
- 仅支持固定长度视频（17帧），自适应长度支持值得探索。
- 图卷积网络投影文本嵌入的设计较重，轻量化替代方案有优化空间。
- 未探索连续（非离散）潜空间版本。

## 相关工作与启发

- TiTok 的 1D token 压缩启发了查询式压缩，但缺乏对视频时间维度的特殊处理。
- LARP 的自适应查询思路被本文采纳并改进为解耦版本。
- 语言码本的词性分配思路可推广到音频（频率词 vs 节奏词）等其他模态。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 解耦设计和运动增强码本均为原创贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 重建/生成/图像/理解全面验证+详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，对比充分
- 价值: ⭐⭐⭐⭐⭐ 对视频 tokenizer 领域贡献大，压缩效率和质量均达 SOTA

<!-- RELATED:START -->

## 相关论文

- [BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](../../CVPR2025/video_generation/bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)
- [TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](../../CVPR2026/video_generation/tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](../../CVPR2025/video_generation/mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)
- [Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](../../CVPR2025/video_generation/semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)

<!-- RELATED:END -->
