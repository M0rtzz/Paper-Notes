---
title: >-
  [论文解读] LLAVIDAL: A Large Language Vision Model for Daily Activities of Living
description: >-
  [CVPR 2025][视频理解][日常活动理解] 针对日常生活活动（ADL）理解，构建了多视角多模态指令微调数据集 ADL-X，提出 LLAVIDAL 模型融合视频、3D 骨架和 HOI 线索，采用 MMPro 渐进式训练策略实现 SOTA 性能。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "日常活动理解"
  - "大语言视觉模型"
  - "多模态融合"
  - "骨架特征"
  - "人物交互"
---

# LLAVIDAL: A Large Language Vision Model for Daily Activities of Living

**会议**: CVPR 2025  
**arXiv**: [2406.09390](https://arxiv.org/abs/2406.09390)  
**代码**: [https://adl-x.github.io/](https://adl-x.github.io/)  
**领域**: 视频理解  
**关键词**: 日常活动理解, 大语言视觉模型, 多模态融合, 骨架特征, 人物交互

## 一句话总结

针对日常生活活动（ADL）理解，构建了多视角多模态指令微调数据集 ADL-X，提出 LLAVIDAL 模型融合视频、3D 骨架和 HOI 线索，采用 MMPro 渐进式训练策略实现 SOTA 性能。

## 研究背景与动机

现有大语言视觉模型（LLVM）主要在网络视频上训练，擅长运动、电影等场景，但在**日常活动（ADL）**理解上存在明显短板：

1. **细粒度动作区分困难**：ADL 涉及微妙的动作差异（如"拿起杯子"vs"放下杯子"），Web 视频训练的模型难以捕捉
2. **缺乏时间非结构化处理能力**：ADL 视频不像烹饪教程有严格时序，可能包含无关中断动作（如做饭途中接电话）
3. **缺少关键模态**：3D 骨架（视角不变表示）和人物-物体交互（HOI）是理解 ADL 的重要线索，但现有 LLVM 未利用
4. **没有 ADL 专用指令微调数据集**

## 方法详解

### 整体框架

LLAVIDAL 由三大部分组成：

- **ADL-X 数据集**：基于 NTU RGB+D 120 的 100K 视频指令对，包含 RGBS（RGB + Skeleton）多视角数据
- **多模态特征提取**：视频 (CLIP-L/14)、骨架 (SkeletonCLIP)、HOI (OWLv2) 三路特征
- **MMPro 渐进式训练**：三阶段逐步集成各模态到 LLM 嵌入空间

### 关键设计

1. **ADL-X 数据集构建**：采用半自动化框架，包含三个核心策略：
    - **Person Augmented Generation (PAG)**：利用骨架数据裁剪人体区域，减少背景干扰，使 AI 标注器聚焦于人体动作
    - **Temporal Stitching (TS)**：将短视频片段拼接成长视频，模拟 ADL 中无固定时序的行为模式，用 GPT 生成 160 个组合动作序列
    - **Weakly Supervised Descriptions (WS)**：先用 CogVLM 按 0.5fps 生成帧级描述，再用 GPT-3.5 结合弱监督动作标签生成连贯视频描述（限 300 字），减少幻觉

2. **多模态特征整合**：探索了三种方式将骨架/HOI 信息注入 LLVM：
    - **ℳ as features**：骨架用 SkeletonCLIP 提取语言对齐特征 $\mathcal{X}_s \in \mathbb{R}^{F_s \times D_s}$；HOI 通过 BLIP2 检测动作相关物体 + OWLv2 定位追踪，提取特征 $\mathcal{X}_o^j \in \mathbb{R}^{8 \times D_o}$
    - **ℳ as QA**：将骨架/HOI 坐标转换为自然语言 QA 对加入训练集
    - **ℳ as context**：将骨架/HOI 的运动描述拼接到文本查询中

3. **MMPro 渐进式训练**：三阶段课程学习策略，解决多模态同步对齐困难：
    - **Stage 1**：各模态独立对齐到 LLM 嵌入空间，通过线性投影层 $Q_m = \mathcal{T}_m(\mathcal{X}_m; \theta_m)$
    - **Stage 2**：视频 + 骨架联合对齐（较简单的模态先整合）
    - **Stage 3**：加入 HOI 模态，三者全部整合。按 curriculum loss 确定顺序：骨架比 HOI 更容易对齐
    - 推理时仅需视频输入，无需裁剪或额外模态

### 损失函数 / 训练策略

- 标准因果语言建模损失：$\min_\theta L_{CE}(\text{LLM}(\mathcal{T}_v(\mathcal{X}_i^v)), y_i)$
- 8 × NVIDIA RTX A6000, batch size 32, learning rate $2e^{-5}$, 3 epochs
- 视频输入 $T=100$ 帧, 分辨率 $224 \times 224$
- 视觉特征维度 $D_v=1024$, 骨架特征维度 $D_s=216$, 物体特征维度 $D_o=512$, LLM 嵌入维度 $K=4096$

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | LLAVIDAL | 之前SOTA | 提升 |
|------------|------|----------|----------|------|
| Charades (MCQ-AR) | Accuracy | **55.2** | 53.1 (ChatUniVi) | +2.1 |
| Smarthome (MCQ-AR) | Accuracy | **48.1** | 48.1 (ChatUniVi) | 持平 |
| LEMMA (MCQ-TC) | Accuracy | **34.3** | 32.6 (VideoLlama) | +1.7 |
| TSU (MCQ-TC) | Accuracy | **38.2** | 36.4 (ChatUniVi) | +1.8 |
| Charades (Description) | Avg | **48.6** | 42.4 (ADL-X ChatGPT) | +6.2 |
| TSU (Description) | Avg | **70.8** | 64.8 (ADL-X ChatGPT) | +6.0 |

### 消融实验

| 配置 | Char AR | SH AR | TSU TC | TSU Desc | 说明 |
|------|---------|-------|--------|----------|------|
| ADL-X ChatGPT baseline | 51.0 | 44.5 | 29.5 | 64.8 | 基线 |
| + Skeleton Features (SF) | 52.7 | 42.6 | 30.3 | 66.5 | 骨架有效 |
| + HOI Features (OF) | 53.8 | 48.0 | 37.1 | 68.0 | HOI 最关键 |
| Joint (SF+OF) | 53.8 | 40.7 | 33.1 | 65.8 | 直接联合次优 |
| MMPro Prog.A (Token) | **55.2** | **48.1** | **38.2** | **70.8** | 渐进式最优 |
| MMPro Prog.B (Token) | 52.8 | 49.4 | 33.0 | 69.2 | 不同顺序较差 |
| X-InstructBLIP | 49.0 | 45.6 | 29.9 | 65.5 | 别人的方法不行 |

### 关键发现

- **ADL-X 三策略验证**：人工评估 100 个 QA 对平均评分 4.1/5.0，证明数据质量高
- **HOI features 比 HOI QA/context 有效得多**：QA 形式(50.4) 和 context 形式(50.3) 远不如直接特征(53.8)
- **骨架特征 + 骨架上下文联合 (SC+SF)** 在 Stage 2 前最优，但最终 MMPro 整合后骨架上下文不再需要
- **MMPro 顺序关键**：Prog.A (视频→骨架→HOI) 比 Prog.B (视频→HOI→骨架) 效果好，因 HOI 特征稀疏对齐更难
- LLAVIDAL 推理时仅需视频，无需额外模态，实用性强
- 超越所有在 10 倍数据上训练的 LLVM（如 CogVLM 训练 15 亿图像）

## 亮点与洞察

- **第一个面向 ADL 的 LLVM**，填补了日常活动理解在大模型时代的空白
- **半自动化数据构建流程**具有通用性，可推广到其他领域特定数据集
- **MMPro 渐进式训练**是解决多同步模态对齐矛盾梯度问题的优雅方案，用 curriculum learning 原则确定模态整合顺序
- 推理阶段无需骨架/HOI 是一个很实用的设计——训练时多模态提升学习能力，推理时单模态保持简洁

## 局限与展望

- 数据来源单一（仅 NTU RGB+D 120），背景和主体多样性有限
- 骨架数据来自 GT 标注，实际场景中骨架估计可能不准确
- HOI 提取依赖 BLIP2 + GPT-3.5 筛选，有引入噪声的风险
- 仅验证了 Vicuna 13B 作为 LLM backbone，未探索更强的基座模型
- QA 对生成依赖 GPT-3.5，存在不可控的幻觉问题

## 相关工作与启发

- 与 VideoLLaVA、VideoChat 等相比，LLAVIDAL 的创新在于领域聚焦 + 额外模态引入
- SkeletonCLIP 将骨架编码到语言空间的方法值得借鉴：通过双编码器对比学习实现跨模态对齐
- Temporal Stitching 模拟 ADL 无结构时序的策略简单有效
- MMPro 可用于任何多模态对齐场景，如视频+音频+文本

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个 ADL LLVM + MMPro 策略新颖
- 实验充分度: ⭐⭐⭐⭐ 多基准多消融，但数据源单一
- 写作质量: ⭐⭐⭐⭐ 方法阐述详尽但略冗长
- 价值: ⭐⭐⭐⭐ 面向实际 ADL 场景，数据+模型+基准三位一体

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)
- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)

</div>

<!-- RELATED:END -->
