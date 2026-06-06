---
title: >-
  [论文解读] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting
description: >-
  [CVPR 2026][多模态VLM][视觉提示] ViKey 通过在视频帧上叠加帧序号的视觉提示（Visual Prompting），配合轻量的关键词-帧映射（KFM）模块，在免训练条件下显著提升 VideoLLM 的时序推理能力，即使只用 20% 的帧也能接近密集帧的性能。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "视觉提示"
  - "视频大语言模型"
  - "时序理解"
  - "帧索引"
  - "免训练"
---

# ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting

**会议**: CVPR 2026  
**arXiv**: [2603.23186](https://arxiv.org/abs/2603.23186)  
**代码**: [https://github.com/MICV-yonsei/ViKey](https://github.com/MICV-yonsei/ViKey)  
**领域**: 多模态VLM  
**关键词**: 视觉提示, 视频大语言模型, 时序理解, 帧索引, 免训练

## 一句话总结

ViKey 通过在视频帧上叠加帧序号的视觉提示（Visual Prompting），配合轻量的关键词-帧映射（KFM）模块，在免训练条件下显著提升 VideoLLM 的时序推理能力，即使只用 20% 的帧也能接近密集帧的性能。

## 研究背景与动机

VideoLLM 在多模态视频任务上表现优异，但处理密集视频帧的计算开销极高，因此帧选择（frame selection）成为标配手段。然而帧选择在提升效率的同时带来一个严重副作用：**打断时序连续性**。

**现有痛点**：当中间帧被移除后，VideoLLM 丧失了推断事件先后关系的能力。例如，一个球员越线后裁判亮红牌的视频，人类从稀疏帧就能推断因果，但 VideoLLM 却可能错误判断裁判踩线。

**核心矛盾**：帧选择使模型只看到时间轴上离散的"快照"，重建时序连贯的事件序列本身就很困难。现有解决方案如增强时序编码、扩展上下文模块等方法复杂且需要大量训练。

**切入角度**：视觉提示（VP）已被证明能有效引导模型关注空间区域，但其在跨帧时序推理中的潜力几乎未被探索。作者发现，简单地在每帧上标注序号就能帮助模型感知时序连续性。

## 方法详解

### 整体框架

ViKey 是一个免训练的即插即用框架：输入视频帧 → 叠加帧序号视觉提示 → 提取查询中的关键文本概念 → 通过 KFM 将关键词映射到最相关的帧 → 改写查询加入帧索引 → 送入 VideoLLM 推理。

### 关键设计

1. **序列化视觉提示（Sequential Visual Prompting）**:

    - 功能：在每帧像素空间中嵌入帧序号信息（如 "frame #01"）
    - 核心思路：在帧的左下角（bottom-left）叠加文本形式的帧序号。字体大小自适应帧分辨率：$fontsize = \min(width, height) / s$。作者通过三组精心设计的实验验证了 VP 的有效性：(1) 位置编码退化实验证明 VP 可独立恢复帧序信息；(2) 帧级引用实验证明 VP 使模型能像字典一样通过序号查找帧内容；(3) 注意力分析表明 VP 在中高层增强了图像 token 的注意力权重
    - 设计动机：放在左下角是因为实验发现存在位置偏差——底部位置准确率远高于顶部（reverse lookup 底部 100% vs 顶部 60-79%），可能因训练数据中字幕/水印常出现在底部

2. **关键词-帧映射（Keyword-Frame Mapping, KFM）**:

    - 功能：将文本查询中的关键概念锚定到最相关的视频帧
    - 核心思路：从用户查询中提取显著关键词，在共享嵌入空间中计算关键词与每帧的相似度，找到最匹配的帧。然后将查询改写为包含帧索引的增强版本，如 "在 frame #03 中，球员做了什么？"。这为推理提供了显式的时序锚点
    - 设计动机：VP 提供了帧级索引能力，KFM 则将文本查询与视觉帧建立显式映射，两者结合实现精确的时序定位

3. **位置偏差分析与优化**:

    - 功能：理解并利用 VideoLLM 对视觉提示位置的偏好
    - 核心思路：系统测试了四个角落位置（TL/TR/BL/BR）的 VP 效果。BL 和 BR 在 reverse lookup 中达到 100% 准确率，而 TL 仅约 60%。TL 的主要错误模式是"差一"（off-by-one）——模型把当前帧的序号与下一帧的内容关联
    - 设计动机：帧 token 被拼接为单一序列，无显式边界。顶部序号容易与后续帧的 token 混淆，底部序号则与当前帧结尾 token 更自然对齐

### 损失函数 / 训练策略

ViKey 是完全免训练的（training-free），不需要修改模型参数或额外训练。

## 实验关键数据

### 主实验

| 模型+设置 | TempCompass | MVBench | VideoMME | LongVideoBench |
|----------|-------------|---------|----------|----------------|
| LLaVA-Video-7B (64帧) | 74.68 | 82.50 | — | 56.42 |
| + ViKey (64帧) | 77.83 | 87.00 | 提升 | 58.66 |
| + ViKey (13帧=20%) | ~75 | ~83 | 接近64帧 | ~56 |

在 TempCompass、MVBench、VideoMME、LongVideoBench 的时序推理子集上一致提升。

### 消融实验

| 配置 | Lookup精度 | Reverse Lookup精度 | 说明 |
|------|-----------|-------------------|------|
| 无 VP | 12.43% | 18.57% | 基线极低 |
| VP (bottom-left) | 64.62% | 100.00% | 帧级引用能力显著提升 |
| VP (top-left) | 55.56% | 60.19% | 位置偏差明显 |
| VP + KFM | 最优 | 最优 | 两者互补 |

### 关键发现

- VP 在位置编码被破坏的极端条件下仍能恢复 2.9-9.9 个百分点的时序理解能力
- VP 使注意力中分配给图像 token 的权重平均增加 11.65%，集中在中高层（第4-6层、11-14层、21层之后）
- 仅 20% 帧 + ViKey 在部分数据集上接近 100% 帧的密集基线，效率极高

## 亮点与洞察

- **极简但有效**：在帧上写个序号就能大幅提升时序推理，这种"不改模型只改输入"的思路既优雅又实用。可以零成本集成到任何 VideoLLM
- **位置偏差的发现**：底部 VP 远优于顶部的发现揭示了 VideoLLM 的训练偏差——模型对底部区域的注意力更强，这一洞察对所有使用 VP 的方法都有指导意义
- **帧即字典的概念**：将帧序号作为键、帧内容作为值的字典隐喻，为 VideoLLM 的细粒度时序控制提供了新范式

## 局限与展望

- KFM 模块的关键词提取依赖额外的嵌入模型，在极长视频中可能成为瓶颈
- VP 本质上占用了帧的像素空间，对于已有字幕/水印的视频可能产生干扰
- 位置偏差暗示模型可能只是在"记住"特定位置的文字，而非真正理解时序关系
- 未来可探索：自适应 VP 大小/位置、与帧选择策略联合优化

## 相关工作与启发

- **vs 传统帧选择方法**: 帧选择只关注"保留哪些帧"，ViKey 关注"如何让保留的帧更有效"，两者互补
- **vs 时序编码增强方法**: 如扩展上下文模块等需要训练的方法，ViKey 完全免训练且效果相当
- **vs 空间 VP 方法**: 之前的 VP 只在空间上引导注意力（如画圈标注），ViKey 首次系统探索了 VP 在跨帧时序推理中的作用

## 评分

- 新颖性: ⭐⭐⭐⭐ 简单但有洞察力的观察，VP 用于时序推理的首次系统探索
- 实验充分度: ⭐⭐⭐⭐⭐ 三组分析实验+四个基准+多个模型，非常扎实
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析清晰，实验设计精巧，分析深入
- 价值: ⭐⭐⭐⭐ 免训练即插即用，实用性很强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](../../CVPR2025/multimodal_vlm/revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)

</div>

<!-- RELATED:END -->
