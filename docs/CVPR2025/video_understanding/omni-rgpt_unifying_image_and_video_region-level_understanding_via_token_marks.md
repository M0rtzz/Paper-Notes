---
title: >-
  [论文解读] Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks
description: >-
  [CVPR 2025][视频理解][区域级理解] Omni-RGPT 提出 Token Mark 机制在视觉特征空间中直接标记目标区域，统一图像和视频的区域级理解，配合 30 万条区域级视频指令数据集 RegVID-300k，在常识推理等任务上达到 SOTA。 多模态大语言模型在全局视觉理解上取得了显著进展…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "区域级理解"
  - "多模态大模型"
  - "Token Mark"
  - "区域级指令数据"
---

# Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks

**会议**: CVPR 2025  
**arXiv**: [2501.08326](https://arxiv.org/abs/2501.08326)  
**代码**: [项目页面](https://miranheo.github.io/omni-rgpt)  
**领域**: Video Understanding / Multimodal LLM  
**关键词**: 区域级理解, 多模态大模型, Token Mark, 视频理解, 区域级指令数据

## 一句话总结

Omni-RGPT 提出 Token Mark 机制在视觉特征空间中直接标记目标区域，统一图像和视频的区域级理解，配合 30 万条区域级视频指令数据集 RegVID-300k，在常识推理等任务上达到 SOTA。

## 研究背景与动机

多模态大语言模型在全局视觉理解上取得了显著进展，但区域级理解仍面临挑战：
- **文本坐标方法**（KOSMOS-2、Shikra）将边界框坐标编码为文本，但视频中标记数量随帧数线性增长，可扩展性差
- **RoI 特征方法**（GPT4RoI、RegionGPT）从每帧提取区域视觉特征，但存在时间漂移问题——同一对象在不同帧的 RoI 特征不一致
- **视觉标记方法**（SoM、ViP-LLaVA）在图像上叠加标记，可能改变原始图像外观
- 仅依赖初始帧的方法（Elysium、Merlin）缺乏对后续帧目标的稳健参考
- 缺乏大规模区域级视频指令数据集——现有数据要么描述简短（Elysium 仅有名词/短语），要么来源单一
- 将多帧的目标表示统一为单一一致向量是一个开放挑战

## 方法详解

### 整体框架

Omni-RGPT 基于 LLaVA 架构，引入 Token Mark 作为区域标识符。输入图像/视频经视觉编码器得到视觉 token $V \in \mathbb{R}^{T \times D \times H \times W}$，Token Mark 被嵌入到区域对应的空间位置并同时注入文本提示中。辅助的 Temporal Region Guide Head 在训练时引导视频中的区域一致性，推理时不引入额外开销。

### 关键设计1：Token Mark 区域表示

**功能**：为每个目标区域分配唯一的可学习 token 标识，建立视觉区域与文本指代之间的直接连接。

**核心思路**：定义一组可学习 token $F \in \mathbb{R}^{N_F \times C}$（类比调色板上的不同颜色）。对于 $N$ 个输入区域 $\{m_i\}_{i=1}^N$，从 $F$ 中均匀采样 $N$ 个 token $R = \{r_i\}_{i=1}^N$。构造 Spatial Token Mark $S_{:,h,w} = \frac{\sum_{i=1}^N m_{i,h,w} \cdot r_i}{\epsilon + \sum_{i=1}^N m_{i,h,w}}$，将其下采样到视觉 token 分辨率后，通过共享投影层映射到 LLM 嵌入空间，作为残差加到视觉 token 上：$\hat{V} = V + \hat{S}$。同时，采样的 token 也替换文本中的 `<region>` 占位符。

**设计动机**：(1) 可扩展性：每个目标的 token 跨帧共享，文本 token 数与帧数无关；(2) 时间一致性：固定 token 确保跨帧一致引用；(3) 保留全局对齐：以残差形式加入不破坏基础模型的视觉-语言对齐。

### 关键设计2：Temporal Region Guide Head

**功能**：在训练阶段引导模型学习视频中目标区域的跨帧一致性，无需依赖显式轨迹标注。

**核心思路**：辅助分类头 $\mathcal{F}_{\text{aux}}$ 作用于 LLM 输出的视觉 token，将每个视觉 token 分类为 $N_F + 1$ 类（$N_F$ 个 Token Mark 类别 + 背景），使用软标签处理多区域重叠的情况。仅在首帧提供区域提示 $\hat{V}_1$，其余帧 $V_2, ..., V_T$ 无区域标注，辅助头通过 Token Mark 的一致性隐式引导模型追踪目标。损失函数 $\mathcal{L} = \mathcal{L}_{\text{LLM}} + \alpha\mathcal{L}_{\text{aux}}$。

**设计动机**：实际应用中用户通常只在一帧标注区域，辅助头使模型学会从首帧标注自动扩展到整个视频，推理时不引入额外开销。

### 关键设计3：RegVID-300k 区域级视频指令数据集

**功能**：提供首个大规模、多样化的区域级视频指令数据集，包含 98k 视频、214k 区域和 294k 指令样本。

**核心思路**：三阶段构建流程：(1) GPT4o 辅助区域级详细描述——使用 SoM 技术在视频帧上叠加掩码索引，输入 GPT4o 生成约 60 词的细粒度描述；(2) 视觉幻觉缓解——过滤 GPT4o 生成描述中的幻觉内容；(3) 描述引导的指令样本生成——从详细描述生成多种对话格式的 QA 对。数据来源于 10 个公开视频数据集。

**设计动机**：现有视频指令数据缺乏区域级标注，限制了区域级视频理解能力的发展。

### 损失函数

$\mathcal{L} = \mathcal{L}_{\text{LLM}} + \alpha\mathcal{L}_{\text{aux}}$，其中 $\mathcal{L}_{\text{LLM}}$ 为标准交叉熵语言建模损失，$\mathcal{L}_{\text{aux}}$ 为辅助分类头的软标签交叉熵损失。

## 实验关键数据

### 主实验：Causal-VidQA 视频常识推理

| 方法 | LLM | Acc@D | Acc@E | Acc@P | Acc@C | Acc@All |
|------|-----|-------|-------|-------|-------|---------|
| **Omni-RGPT** | 7B | **84.0** | **84.6** | **84.2** | **85.4** | - |
| MotionEpic | 7B | 81.2 | 83.0 | 74.3 | 73.7 | 69.4 |
| Video-LLaVA | 7B | 73.7 | 74.4 | 67.6 | 65.4 | 61.8 |
| Video-ChatGPT | 7B | 73.1 | 75.1 | 66.0 | 63.9 | 61.1 |

### 消融实验：Token Mark 关键组件

| 设置 | VCR Q→A | VCR QA→R | 说明 |
|------|---------|----------|------|
| 完整模型 | **最优** | **最优** | Token Mark + Guide Head |
| 仅文本坐标 | 较差 | 较差 | 传统坐标编码 |
| 仅 RoI 特征 | 较差 | 较差 | 存在时间漂移 |
| w/o Guide Head | 略差 | 略差 | 视频一致性下降 |

### 关键发现

- Token Mark 在图像（VCR）和视频（Causal-VidQA）常识推理基准上均达到 SOTA
- 辅助引导头使模型在仅有首帧区域标注的情况下也能稳定追踪视频中的目标
- RegVID-300k 数据集使模型在视频描述任务上获得显著提升
- Token Mark 方法的文本 token 数量与视频帧数无关，具有优异的可扩展性

## 亮点与洞察

- **逆向思维**：传统方法从视觉特征生成区域嵌入，Token Mark 反过来用预定义 token 标记区域，思路新颖
- **极简统一**：用同一套 Token Mark 机制同时处理图像和视频的区域级理解
- **数据贡献**：RegVID-300k 填补了区域级视频指令数据的空白

## 局限与展望

- 依赖首帧的区域标注，对首帧目标遮挡或不可见的情况不够鲁棒
- Token Mark 数量 $N_F$ 固定，可能限制密集区域场景的处理能力
- 辅助引导头的区域跟踪能力依赖于 LLM 的视觉注意力，在快速运动场景中可能退化
- 未来可探索与显式跟踪模型的结合

## 相关工作与启发

- Token Mark 类似于 DETR 中可学习查询的思想，但应用于标记特定空间区域
- 与 SoM（Set-of-Mark）的视觉标记方法相比，Token Mark 在潜空间而非像素空间操作，不改变原始图像
- RegVID-300k 的 GPT4o 辅助标注 + 幻觉缓解流程可复用到其他视频理解数据集构建

## 评分

⭐⭐⭐⭐ — Token Mark 的设计简洁而有效，统一了图像和视频区域理解的方法框架。RegVID-300k 数据集的贡献也很有价值。在多个基准上的 SOTA 表现验证了方法的有效性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](mmvu_measuring_expert-level_multi-discipline_video_understanding.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[ICML 2025\] Unifying Specialized Visual Encoders for Video Language Models](../../ICML2025/video_understanding/unifying_specialized_visual_encoders_for_video_language_models.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](../../ICML2026/video_understanding/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)

</div>

<!-- RELATED:END -->
