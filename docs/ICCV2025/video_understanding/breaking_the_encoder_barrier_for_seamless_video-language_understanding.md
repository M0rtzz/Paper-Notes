---
title: >-
  [论文解读] Breaking the Encoder Barrier for Seamless Video-Language Understanding
description: >-
  [ICCV 2025][视频理解][encoder-free] 提出 ELVA，首个无编码器（encoder-free）的视频大语言模型，通过层级 token 合并、视频引导监督和混合分辨率推理机制，仅用 7M 公开视频-文本对数据即可达到与有编码器架构相当的性能，同时将 FLOPs 降低 95%、推理延迟降低 92%。
tags:
  - ICCV 2025
  - 视频理解
  - encoder-free
  - Video-LLM
  - token merging
  - video guidance
  - hybrid resolution
---

# Breaking the Encoder Barrier for Seamless Video-Language Understanding

**会议**: ICCV 2025  
**arXiv**: [2503.18422](https://arxiv.org/abs/2503.18422)  
**代码**: 无  
**领域**: 视频理解 / 视频大语言模型  
**关键词**: encoder-free, Video-LLM, token merging, video guidance, hybrid resolution

## 一句话总结

提出 ELVA，首个无编码器（encoder-free）的视频大语言模型，通过层级 token 合并、视频引导监督和混合分辨率推理机制，仅用 7M 公开视频-文本对数据即可达到与有编码器架构相当的性能，同时将 FLOPs 降低 95%、推理延迟降低 92%。

## 研究背景与动机

现有 Video-LLM 几乎都采用"编码器+解码器"框架（如 CLIP 编码器 + LLM），面临三大根本性限制：

**计算开销累积**：视频需要逐帧通过视觉编码器提取特征，帧数越多开销越大；大型编码器（如 InternViT-6B）进一步加剧了这一问题

**时空分辨率约束**：编码器对固定尺寸的视觉表示施加分辨率偏差，无法根据内容动态调整分辨率

**多模态交互瓶颈**：依赖预提取特征，限制了视频像素与文本 token 之间的底层交互，以及帧间依赖的建模

encoder-free 方法在图像领域已有探索（Fuyu、EVE），但视频数据由于高维性和时序依赖性带来了额外挑战。ELVA 旨在证明无编码器的 Video-LLM 可以实现具有竞争力的性能。

## 方法详解

### 整体框架

ELVA 基于 Qwen2 LLM 骨干，直接将原始视频像素送入 LLM 处理。关键技术包括：Native Video Tokenizer 保持原始分辨率和宽高比、轻量级视频 patch 嵌入层进行时空预建模、层级 token 合并渐进压缩冗余信息、视频引导监督学习时空表示。

### 关键设计

1. **原生视频 token 化（Native Video Tokenization）**:

    - 直接将视频帧按原始分辨率分割为 patch，不做预处理
    - 引入特殊 token：`<FRAME>` 标记每帧起始，`<LINE>` 标记 patch 行结束（光栅扫描顺序）
    - 优势：支持任意分辨率和帧长度的视频输入

2. **视频 Patch 嵌入层（Video Patch Embedding Layer）**:

    - 仅 9M 参数的轻量级时空预建模模块
    - 为每行 patch 添加 `<LINE>` 可学习 token，为每帧添加 `<FRAME>` 可学习 token
    - 通过交叉注意力层建立长程时空关系：用 `<FRAME>` token 查询帧内嵌入，用 `<LINE>` token 查询行内嵌入
    - 相比朴素 patch 嵌入，在长视频任务上平均提升 2.53%

3. **层级 Token 合并（Hierarchical Token Merging）**:

    - 在 LLM 不同层之间渐进合并时间维度上的冗余 token
    - 维护索引矩阵 $\bm{M} \in \{0,1\}^{T \times (H \cdot W / P^2)}$，计算相邻帧对应位置 token 的余弦相似度：$s_{ij} = \langle f^l_{ij}, f^l_{(i+1)j} \rangle$
    - 相似度超过阈值 $\tau=0.6$ 的 token 通过均值合并
    - 浅层：超阈值即合并；深层：持续合并直到达到目标压缩比（50%）
    - 与直接池化相比，保留了关键时空信息，长视频性能退化远小于池化方法

4. **视频引导监督（Video Guidance Supervisor）**:

    - 使用预训练 SigLIP 视频模型作为教师
    - **Tube-wise 对齐损失**：LLM 最后层视觉特征 $\mathbf{f}_{\text{vis}}$ 与教师模型特征 $\mathbf{f}_{\text{target}}$ 对帧均值池化后做 MSE 对齐：$\mathcal{L}_{\text{MSE}} = \text{MSE}(\frac{\mathbf{f}_{\text{vis}}}{\|\mathbf{f}_{\text{vis}}\|_2}, \frac{\mathbf{f}_{\text{target}}}{\|\mathbf{f}_{\text{target}}\|_2})$
    - **Frame-wise 对比损失**：保留 `<FRAME>` token，帧级均值池化后跨 GPU 计算 InfoNCE 对比损失 $\mathcal{L}_{\text{Con}}$
    - 总训练损失：$L = L_{\text{Gen}} + L_{\text{MSE}} + L_{\text{Con}}$

### 损失函数 / 训练策略

三阶段渐进训练：

- **Stage 1 空间预训练**：图像作为单帧视频训练，使用 ELVA-Image（4M 样本），学习基础视觉信息
- **Stage 2 时空预训练**：加入 ELVA-Video（3M 样本），三个损失函数同时作用，学习时空表示
- **Stage 3 监督微调（SFT）**：仅用文本生成损失，使用 665K 图像 + 178K 视频 SFT 数据

训练数据中大量使用了 Qwen2-VL 重新标注的高质量 dense caption，显著优于原始标注。

## 实验关键数据

### 主实验

| 模型 | 类型 | LLM | MSVD | ActivityNet | VideoMME | MLVU | CinePile |
|------|------|-----|------|-------------|----------|------|----------|
| Video-LLaVA | encoder | 7B | 70.7 | 45.3 | 39.9 | 47.3 | 22.5 |
| VideoLLaMA2 | encoder | 7B | 70.9 | 50.2 | 46.6 | 48.5 | 44.6 |
| Fuyu | encoder-free | 8B | 56.8 | 28.8 | 28.7 | 31.1 | 26.0 |
| EVE | encoder-free | 7B | 61.4 | 41.8 | 29.3 | 36.8 | 26.4 |
| **ELVA** | **encoder-free** | **7B** | **65.2** | **48.7** | **47.1** | **51.8** | **46.1** |

### 推理效率对比（32帧）

| 模型 | MEM (G) | FLOPs (T) | TTFT (s) |
|------|---------|-----------|----------|
| Encoder-based | 20.7 | 260 | 2.59 |
| ELVA (无合并) | 20.0 (-3%) | 75 (-71%) | 0.51 (-80%) |
| ELVA + Merge | 16.4 (-21%) | 25 (-90%) | 0.26 (-90%) |
| ELVA + Merge + HR | **15.5 (-25%)** | **14 (-95%)** | **0.22 (-92%)** |

128帧时更明显：FLOPs 降低96%，TTFT仅0.56s（encoder-based需15.18s）。

### 消融实验

| 预训练目标 | GQA | SEED_I | MSVD | VideoMME |
|-----------|-----|--------|------|----------|
| $\mathcal{L}_{\text{Gen}}$ only | 42.2 | 40.0 | 45.8 | 37.9 |
| + $\mathcal{L}_{\text{MSE}}$ | 43.6 | 42.6 | 47.1 | 38.1 |
| + $\mathcal{L}_{\text{Con}}$ | 42.4 | 41.0 | 47.4 | 38.5 |
| + 两者 | **44.4** | **44.8** | **48.0** | **38.5** |

| 数据质量 | GQA | MSVD | VideoMME |
|---------|-----|------|----------|
| 原始 caption | 42.1 | 46.0 | 34.2 |
| Recap Image+Video | **46.1** | **49.4** | **38.5** |

### 关键发现

- encoder-free Video-LLM 首次证明可以达到 encoder-based 模型的可比性能
- 视频预训练的教师模型（video-pretrained SigLIP）比仅图像预训练的编码器作为引导效果更好（约1点提升每任务）
- 重标注的高质量 caption 至关重要：比原始 caption 在各任务上提升 3-4%
- 短视频 QA 主要依赖空间建模（Stage 1 提升快），长视频需要时空建模（Stage 2 提升明显）
- 混合分辨率推理：保持高分辨率帧数不变，增加低分辨率帧可在几乎不增加 token 开销的情况下大幅提升长视频性能（VideoMME +5.9%）
- 层级合并（50%压缩率）几乎不损失精度，但显著降低推理成本

## 亮点与洞察

- **首次验证 encoder-free Video-LLM 的可行性**：打破了必须使用视觉编码器的惯性思维，证明 LLM 自身可以直接从像素学习视频表示
- **效率优势巨大**：95% FLOPs 减少和 92% 延迟降低使实时视频理解成为可能
- **混合分辨率策略精巧**：充分利用了 encoder-free 架构的灵活性，在同一视频中混用高低分辨率帧
- **数据质量 > 数据规模**：高质量重标注的 caption 带来的提升超过增加数据量

## 局限与展望

- 仅用 7M 数据训练，与使用数十亿数据的 encoder-based 模型相比数据规模差距大
- 在短视频基准上仍略落后于最强 encoder-based 模型
- 层级合并的阈值和压缩率需要手动设定，可探索自适应策略
- 视频引导教师模型仍然是一个外部视觉编码器，理论上不是完全独立的

## 相关工作与启发

- 与 Fuyu（仅线性投影）和 EVE（仅图像预训练）相比，ELVA 的时空预建模和视频引导损失是关键差异化因素
- 层级 token 合并与 ToME 等方法有相似思路，但在 LLM 内部跨层执行，更适合自回归生成
- 混合分辨率推理策略可推广到其他需要处理长序列的多模态模型

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个有效的 encoder-free Video-LLM，多项关键技术创新
- 实验充分度: ⭐⭐⭐⭐ 8个视频基准 + 详细消融 + 效率对比
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，三大局限性提得精准
- 价值: ⭐⭐⭐⭐⭐ 95% FLOPs降低开启视频理解效率新范式

<!-- RELATED:START -->

## 相关论文

- [DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [Unleashing Hour-Scale Video Training for Long Video-Language Understanding](../../NeurIPS2025/video_understanding/unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding](4d-bench_benchmarking_multi-modal_large_language_models_for_4d_object_understand.md)
- [Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)

<!-- RELATED:END -->
