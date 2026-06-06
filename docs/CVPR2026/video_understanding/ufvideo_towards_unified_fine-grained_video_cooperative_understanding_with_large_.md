---
title: >-
  [论文解读] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models
description: >-
  [CVPR 2026][视频理解][统一视频理解] UFVideo 是首个统一全局、像素级和时序级三种粒度视频理解能力的 Video LLM，通过视觉-语言引导对齐策略和 SAM2 mask decoder，在单一模型内同时支持视频问答、目标引用、视频分割和时序定位…
tags:
  - "CVPR 2026"
  - "视频理解"
  - "统一视频理解"
  - "多粒度协同"
  - "像素级分割"
  - "时序定位"
  - "Video LLM"
---

# UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models

**会议**: CVPR 2026  
**arXiv**: [2512.11336](https://arxiv.org/abs/2512.11336)  
**代码**: [https://github.com/Heven-Pan/UFVideo](https://github.com/Heven-Pan/UFVideo)  
**领域**: 视频理解 / 多模态VLM  
**关键词**: 统一视频理解, 多粒度协同, 像素级分割, 时序定位, Video LLM

## 一句话总结
UFVideo 是首个统一全局、像素级和时序级三种粒度视频理解能力的 Video LLM，通过视觉-语言引导对齐策略和 SAM2 mask decoder，在单一模型内同时支持视频问答、目标引用、视频分割和时序定位，并构建了多粒度协同理解基准 UFVideo-Bench。

## 研究背景与动机

1. **领域现状**：当前 Video LLM 已从通用视频问答扩展到多种细粒度理解任务，包括视频目标引用（video referring）、视频分割（video segmentation）、时序定位（temporal grounding）等。这些任务分别对应像素级和时序级的视频理解。

2. **现有痛点**：现有方法各自专注于单一粒度的理解任务，彼此独立训练和推理，无法有效整合不同粒度的感知和推理能力来实现互相增强。例如，擅长目标引用的模型无法处理事件时间定位，专注时序定位的模型无法进行像素级分割。

3. **核心矛盾**：不同粒度的视频知识实际上可以互补——细粒度时序知识能增强对引用目标的理解，全局视频知识能为细粒度任务提供语义支持。但现有模型在生成时各粒度是隔离的，没有显式关联。

4. **本文目标** 如何在单一模型中统一全局（global）、像素级（pixel-level）、时序级（temporal-level）三种粒度的视频理解，并且让它们协同工作。

5. **切入角度**：设计统一的视觉-语言引导对齐策略，通过特殊 token 区分不同任务的输入输出，共享 LLM backbone 实现多任务联合训练。

6. **核心 idea**：用统一的 token 设计（`<Ref>` / `<Seg>` / `<Temp>`）将全局问答、像素级分割、时序定位三类任务统一到同一个 Video LLM 中，实现多粒度协同视频理解。

## 方法详解

### 整体框架
UFVideo 以 LLM 为骨架，视觉编码器将视频编码为离散 token 与文本 token 对齐。输入包括视频 $V$、文本问题 $Q$ 和可选的目标视觉提示 $M$（mask）；输出包括文本回答 $A$、时序定位 $T$ 和分割 mask $S$。通过统一的视觉-语言引导对齐训练，模型可以根据不同任务灵活地生成对应类型的输出。

### 关键设计

1. **多粒度任务对齐（Multi-Grained Video Tasks Alignment）**:

    - 功能：通过特殊 token 区分和统一不同粒度的视频理解任务
    - 核心思路：设计三类特殊 token——`<Temp-τ>` 表示相对时间戳（将视频时间归一化到固定长度 $N_t$ 后编码为 $\tau = \frac{t}{T_n} \times N_t$），`<Ref>` 作为视觉提示注入的占位符用于目标引用任务，`<Seg>` 用于从 LLM 输出中提取分割相关的 language embedding。文本指令被 tokenize 为 $\mathcal{T}_i$，时序 token 为 $\mathcal{T}_t$，形成统一的输入表示。
    - 设计动机：通过特殊 token 而非独立模块来区分任务，可以共用同一个 LLM backbone，避免为每个任务设计专门架构，同时让不同任务的知识在共享参数中互相增强。

2. **多模态编码（Encode for Multi-Modal Input）**:

    - 功能：将视频和目标视觉提示统一编码为 token
    - 核心思路：使用预训练视觉编码器 $\Phi_v$（SigLIP-so400m）分别对视频 $V$ 和目标视觉提示 $M$ 进行编码，获取视频特征 $F_V$ 和目标视觉特征 $F_M$。然后用 VideoRefer 的方法从 $F_M$ 中提取目标空间特征 $S_M$，投影为目标视觉 token $\mathcal{T}_r$ 注入到 `<Ref>` 位置。对于分割任务，随机选取 $K$ 帧用 SAM2 的 Hiera-L 编码器编码，作为 mask decoder 的视觉输入。
    - 设计动机：将不同模态信息统一到 token 空间，使 LLM 能同时处理视频内容和目标级别信息。

3. **多任务解码（Decode for LLM Generative）**:

    - 功能：从 LLM 的统一输出中解码出文本、时间和分割三种结果
    - 核心思路：LLM 输出 hidden state $H$ 后，文本回答和时序定位都通过 text-form token 生成（时序通过 $\mathcal{Y}_m = p_\theta(H) \times \frac{T_n}{N_t}$ 转换回实际时间）。像素级分割则利用 `<Seg>` token 的位置 mask $\rho_s$ 提取分割相关 embedding，通过投影层 $\theta$ 与位置 mask 做逐元素乘法后送入 SAM2 的 mask decoder 生成分割结果。由于不同样本的分割目标数量不同，需要动态 embedding 训练。
    - 设计动机：文本和时间可以直接用 LLM 的 next-token prediction 生成，而像素级分割难以直接用 token 表示，因此借用 SAM2 的 mask decoder 桥接 language embedding 到 pixel-level mask。

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \gamma \cdot \mathcal{L}_{text} + \mathcal{L}_{mask}$。其中 $\mathcal{L}_{text}$ 是标准 next-token prediction 的负对数似然损失；$\mathcal{L}_{mask} = \alpha \cdot \text{BCE}(S_p, S_t) + \beta \cdot \text{DICE}(S_p, S_t)$ 包含二值交叉熵和 DICE 损失。超参数设置为 $\alpha=2.0, \beta=0.5, \gamma=1.0$。训练分两阶段：Stage 1 全局 batch size 512 训练 2 epoch，Stage 2 batch size 256 训练 1 epoch。使用 32 张 A800 GPU，视觉编码器为 SigLIP-so400m-patch14-384，预训练模型为 VideoRefer 7B。

## 实验关键数据

### 主实验

**通用视频理解（MVBench）**:

| 模型 | 参数量 | 平均得分 |
|------|--------|----------|
| GPT-4V | - | 43.5 |
| Qwen2-VL | 7B | 67.0 |
| LLaVA-ST | 7B | 64.2 |
| UniPixel | 3B | 62.5 |
| **UFVideo** | **7B** | **67.3** |

**视频引用描述（VideoRefer-Bench-D）**:

| 模型 | Single-Frame Avg | Multi-Frame Avg |
|------|-----------------|-----------------|
| GPT-4o | 2.95 | 3.25 |
| VideoRefer | 3.42 | 3.46 |
| UniPixel | 3.47 | 3.48 |
| **UFVideo** | **3.59** | **3.61** |

**视频引用问答（VideoRefer-Bench-Q）**:

| 模型 | 平均得分 |
|------|----------|
| GPT-4o | 71.3 |
| RGA3 | 74.0 |
| UniPixel | 73.8 |
| **UFVideo** | **77.9**（Multi-Frame） |

### 消融实验

| 配置 | MVBench Avg | VideoRefer-D (MF) | VideoRefer-Q (MF) |
|------|------------|-------------------|-------------------|
| Full model (UFVideo) | 67.3 | 3.61 | 77.9 |
| w/o 时序级任务 | 性能下降 | - | - |
| w/o 像素级任务 | - | 性能下降 | 性能下降 |

### 关键发现
- UFVideo 在 9 个公共基准上均达到 SOTA，在 MVBench 上以 67.3% 超越 Qwen2-VL（67.0%）
- 多粒度联合训练带来的互增强效果显著——在视频引用任务上大幅超越仅做引用的 VideoRefer
- 在视频分割任务（MeViS、Ref-YouTube-VOS 等）上也超越专门的分割模型
- UFVideo-Bench 三种协同任务（PixRQA/PixHQA/PixTRQA）展示了模型在需要同时输出文本+时序+分割时的综合能力

## 亮点与洞察
- **统一特殊 token 设计是关键 trick**：用 `<Ref>`、`<Seg>`、`<Temp>` 三种 token 实现任务区分而非独立模块，优雅且高效，使得一个 7B 模型能覆盖 4+ 种视频理解任务
- **SAM2 decoder 作为分割桥梁**：直接让 LLM 输出 mask 不现实，通过提取 `<Seg>` 位置的 embedding 送入 SAM2 decoder，巧妙地在 language space 和 pixel space 之间建立映射
- **相对时序 token 设计**：将视频时长归一化到固定长度后编码，使模型能处理不同长度视频的时序定位，且可与文本 token 统一生成

## 局限与展望
- 当前模型在 UFVideo-Bench 上的表现表明多粒度协同理解仍有很大提升空间，特别是 PixTRQA 同时需要时序检索+分割+问答的任务难度很高
- 视频帧数和分辨率受限于 GPU 内存，对超长视频的处理能力有限
- 分割质量受限于 SAM2 decoder 的能力上界
- 仅在 7B 规模实验，未验证 scaling law

## 相关工作与启发
- **vs RGA3/UniPixel**: 这两个工作统一了像素级的引用和分割，但缺乏时序理解。UFVideo 在此基础上增加了时序粒度，实现了真正的三粒度统一
- **vs LLaVA-ST**: LLaVA-ST 做了空间-时间理解但用 bounding box 而非 mask，粒度较粗。UFVideo 使用像素级 mask 实现更精细的理解
- **vs VideoRefer**: UFVideo 的预训练模型即基于 VideoRefer 7B，在其基础上扩展了分割和时序能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个统一三种粒度的 Video LLM，但技术组件多为已有方法的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 9 个公共基准 + 自建 benchmark，对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但符号体系较复杂
- 价值: ⭐⭐⭐⭐ 为多粒度视频统一理解指明了方向，UFVideo-Bench 有一定社区价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](text-guided_fine-grained_video_anomaly_understanding.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)

</div>

<!-- RELATED:END -->
