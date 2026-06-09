---
title: >-
  [论文解读] Number it: Temporal Grounding Videos like Flipping Manga
description: >-
  [CVPR 2025][视频理解][视频时间定位] 本文提出 NumPro，把每一帧的序号直接画在视频帧的右下角，使 Vid-LLM 把"看到事件"和"说出对应帧号"绑成同一个 OCR 任务，从而在零训练或轻量 LoRA 微调下显著提升视频时间定位 (VTG) 的 mIoU 与 mAP。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频时间定位"
  - "帧号叠加"
  - "Vid-LLM"
  - "moment retrieval"
  - "highlight detection"
---

# Number it: Temporal Grounding Videos like Flipping Manga

**会议**: CVPR 2025  
**arXiv**: [2411.10332](https://arxiv.org/abs/2411.10332)  
**代码**: [https://github.com/yongliang-wu/NumPro](https://github.com/yongliang-wu/NumPro)  
**领域**: 视频理解 / Vid-LLM / 视觉提示  
**关键词**: 视频时间定位、帧号叠加、Vid-LLM、moment retrieval、highlight detection

## 一句话总结
本文提出 NumPro，把每一帧的序号直接画在视频帧的右下角，使 Vid-LLM 把"看到事件"和"说出对应帧号"绑成同一个 OCR 任务，从而在零训练或轻量 LoRA 微调下显著提升视频时间定位 (VTG) 的 mIoU 与 mAP。

## 研究背景与动机
1. **领域现状**：Vid-LLM (如 Qwen2-VL、LongVA、GPT-4o) 在视频问答上很强，但在 Video Temporal Grounding (VTG) — 即给一个事件描述要预测起止帧/时间 — 上表现普遍很差。
2. **现有痛点**：现有 VTG 方法要么需要重新训练专用 token / temporal embedding (TimeChat、VTG-LLM)，要么把时间戳硬拼到文本输入中 (VTimeLLM)，扩展性差，且常出现"frame 200 到 599"这种与 10 帧视频严重不符的幻觉。
3. **核心矛盾**：作者通过 attention 分析发现，Vid-LLM 其实能注意到正确的帧，问题不在"看不见"，而在"看见了却说不出对应的时间"——视觉识别和时间表达之间缺乏可对齐的桥梁。
4. **本文目标**：在不改架构、不加额外 token 的前提下，把"时间表达"问题变成 Vid-LLM 已经擅长的"视觉识别"问题。
5. **切入角度**：类比看漫画——每一格都标了页码 / 格号，读者天然能定位事件发生在第几格。
6. **核心 idea**：把帧号当作"视觉提示"画到帧上，让模型通过内置 OCR 能力直接"读"时间，从而把 VTG 转换为视觉对齐任务。

## 方法详解

### 整体框架
NumPro 有两种使用方式，二者共享同一个核心操作 (在帧上叠加数字)：
- **训练-free (NumPro)**：对原视频每一帧画上对应的帧号 (红色、字号 40、贴在右下角)，并在 prompt 前加一句 "The red numbers on each frame represent the frame number."，直接喂给已有的 Vid-LLM。
- **微调 (NumPro-FT)**：把 NumPro 应用到 VTG 训练集上构造增强数据，对 Vid-LLM 做 LoRA 微调 (冻结视觉编码器，只微调投影层 + LoRA 适配的 LLM)。

输入是视频 + 自然语言事件 query；输出是事件的起止帧号 / 时间区间 (moment retrieval) 或每帧 highlight 分数 (highlight detection)。

### 关键设计

1. **数字视觉提示 (Number-Prompt 叠加)**

    - 功能：在每一帧固定位置画上其全局帧号，使帧画面同时承载"内容信息"和"时间信息"。
    - 核心思路：取视频帧序列 $\{f_1, ..., f_N\}$，对每一帧 $f_t$ 用 PIL/OpenCV 画一个文本 $t$，颜色红、字号 40、位置右下角。模型读图时通过 OCR 把 $t$ 当作 token 取回，并和事件 query 关联。
    - 设计动机：Vid-LLM 已有强大的字符识别能力但不会"数帧"；把帧号显式可视化，相当于给模型加了一把可被注意力直接 attend 的"时间标尺"，把"何时发生"压缩为"图里写了多少"。

2. **NumPro 设计参数搜索 (字号/颜色/位置)**

    - 功能：在"数字易识别"和"不遮挡画面"之间找平衡点。
    - 核心思路：用 CLIP ViT-B/32 在 1000 张 MSCOCO 图上做代理实验，每张图用不同字号 (20/40/60/80)、颜色 (红/黄/黑/白等)、位置 (TL/TR/BL/BR/Center) 叠加 0-99 的数字，分别计算 Number Accuracy (CLIP 能否选出正确数字) 和 Caption Accuracy (CLIP 能否仍选出原始 caption)。最终选取红色 + 字号 40 + 右下角，达成两个准确率的 Pareto 前沿。
    - 设计动机：以往视觉提示工作 (red circle、SoM) 表明颜色和位置会强烈影响 VLM 的注意力；用 CLIP 做代理可以低成本扫参，避免在 VTG 数据集上做大规模超参搜索。

3. **NumPro-FT 微调策略**

    - 功能：把"读帧号"这个新行为内化到模型权重，让模型即使在边界情况也能稳定输出帧号。
    - 核心思路：在 Charades-STA / ActivityNet / QVHighlights 训练集上把视频按 NumPro 渲染，把答案改写成"from frame X to frame Y"形式，按自回归语言建模目标 $P(\mathbf{A}\mid V, T_{\text{instruct}}) = \prod_{j} P_\theta(A_j\mid V, X_{\text{instruct}}, \mathbf{A}_{<j})$ 做 LoRA 微调。视觉编码器冻结，只更新视觉投影器与 LoRA 注入的 LLM 部分。
    - 设计动机：训练-free 已经能稳定提分，但 LoRA 微调能让模型学会更精细的边界 (例如 R@0.7 这种严格 IoU 指标) 而不破坏通用能力。

### 损失函数 / 训练策略
仅使用标准的下一 token 预测损失，对答案 token 求平均交叉熵。微调时只更新 LoRA + Visual Projector，参数量极小。训练-free 模式没有任何参数更新。

## 实验关键数据

### 主实验
在三大 VTG benchmark 上对比，本文与 VTG 专用 Vid-LLM、通用 Vid-LLM 都做了对比。

| 数据集 | 指标 | 通用 Vid-LLM 基线 (Qwen2-VL-7B) | +NumPro (training-free) | NumPro-FT (微调) | 之前 SOTA |
|--------|------|----------|----------|----------|----------|
| Charades-STA | R@0.5 | 5.4 | 36.8 | 更高 | 33.8 (VTG-LLM) |
| Charades-STA | mIoU | 7.9 | 38.5 | — | 33.7 (HawkEye) |
| ActivityNet | R@0.5 | 9.4 | 26.4 | — | 27.8 (VTimeLLM) |
| QVHighlights | mAP | 21.5 | 23.6 | — | 16.5 (VTG-LLM) |
| GPT-4o | R@0.5 (Charades) | 32.0 | **35.5** | — | — |

NumPro-FT 在 moment retrieval 上 mIoU 比之前最佳方法高出 **6.9%**，highlight detection 的 mAP 高出 **8.5%**，确立新 SOTA。

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| 字号 20 | 数字过小，OCR Accuracy 显著下降 |
| 字号 40 (默认) | 数字 / caption 准确率综合最佳 |
| 字号 80 | 数字识别率高但严重遮挡画面，caption 准确率掉得最多 |
| 颜色：红 | 数字识别最高，与"red circle"经验一致 |
| 颜色：黑 | 在自然图像背景下识别率最差 |
| 位置：右下 (默认) | 数字-caption 平衡最好 |
| 位置：中央 | caption accuracy 暴跌 (遮挡主体) |
| 不加 prompt 提示 | 模型有时会把数字当噪声忽略 |

### 关键发现
- 通用 Vid-LLM (如 Qwen2-VL-7B) 在 VTG 上几乎不可用 (R@0.5 仅 5.4)，但加了 NumPro 立刻能跑到 36.8，说明问题主要不在"看不见"，而在"说不出时间"。
- GPT-4o 这种闭源模型也能受益于 NumPro，证明该方法是真正的"plug-in"。
- 字号、颜色、位置三个超参中，**位置**对画面干扰最大，**颜色**对识别率影响最大。

## 亮点与洞察
- **把"语言时间表达"问题转化为"视觉 OCR"问题**：这是把 VLM 已有能力迁移到新任务的非常优雅的范式，几乎零成本。
- **CLIP 代理超参搜索**：在不需要昂贵 VTG 标注的情况下，用 MSCOCO + CLIP 就能预先确定最佳视觉提示参数，可被借鉴到任何"在图上加 marker"的任务 (例如 referring expression、动作计数)。
- **视觉提示 vs 文本提示**：以前都是把时间戳塞进 prompt，本文证明把信息画到图上更易被 VLM 利用 — 因为视觉编码器和 LLM 之间的对齐在视觉侧更稳。
- 这种"在像素层面写元信息"的思路可以迁移到：3D 场景里画方向标签、医学图像上叠序号给切片排序、机器人视频里标 timestep 等。

## 局限与展望
- 仅适用于"帧已采样、可枚举"的离散帧表示；对真正的 long-form 流式视频或高 FPS 场景需要先做帧采样，可能丢失时间精度。
- 数字最多到 ~99 (受字号 / 视野限制)，对长视频需要分段或分级编号。
- 训练-free 模式下，模型仍受底座 VLM 在该字号下 OCR 能力的影响 (作者已观察到 LongVA 表现弱于 Qwen2-VL)。
- 尚未探索"非数字"提示 (如时间戳秒数、彩色色条) 的对比，这是一个可扩展的 ideas 方向。
- 改进思路：把帧号 + 时间戳 + 章节号一起渲染，让模型学到层次化时间结构。

## 相关工作与启发
- **vs TimeChat / VTimeLLM**：他们重新设计模型 (加 temporal token 或专门 head)，本文不动模型只动输入；本文优势是即插即用，劣势是受底座 OCR 能力限制。
- **vs SoM (Set-of-Mark)**：SoM 用数字标记图像区域辅助 referring，本文是把同一思路从空间维度扩到时间维度。
- **vs ViP-LLaVA**：ViP-LLaVA 用图形提示提示空间区域；NumPro 第一次把数字提示用于时间维度。
- 启发：任何"模型能看见但说不出"的任务都可以尝试"在像素上画 marker"——例如多视角 3D 任务里画相机编号、Audio-visual 任务里画声源指示。

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路非常简单但确实是 first，跨任务可迁移
- 实验充分度: ⭐⭐⭐⭐ 三大 benchmark + 多个底座 VLM + 设计参数搜索完整
- 写作质量: ⭐⭐⭐⭐ 漫画类比讲得很清楚，attention 分析有说服力
- 价值: ⭐⭐⭐⭐⭐ 训练-free 即可显著提分，工程价值极高，可直接接入任何 Vid-LLM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](videogem_training-free_action_grounding_in_videos.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)
- [\[NeurIPS 2025\] DualGround: Structured Phrase and Sentence-Level Temporal Grounding](../../NeurIPS2025/video_understanding/dualground_phrase_temporal.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)

</div>

<!-- RELATED:END -->
