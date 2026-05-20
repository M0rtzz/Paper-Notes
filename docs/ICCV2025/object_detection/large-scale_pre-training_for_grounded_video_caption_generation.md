---
title: >-
  [论文解读] Large-scale Pre-training for Grounded Video Caption Generation
description: >-
  [ICCV 2025][目标检测][视频定位字幕生成] 提出 GROVE 模型和大规模自动标注方法，构建包含 1M 视频的 HowToGround1M 预训练数据集和 3513 个视频的手动标注 iGround 数据集，实现联合视频字幕生成与多目标时空边界框定位…
tags:
  - "ICCV 2025"
  - "目标检测"
  - "视频定位字幕生成"
  - "大规模预训练"
  - "时空定位"
  - "自动标注"
  - "边界框预测"
---

# Large-scale Pre-training for Grounded Video Caption Generation

**会议**: ICCV 2025  
**arXiv**: [2503.10781](https://arxiv.org/abs/2503.10781)  
**代码**: [项目主页](https://ekazakos.github.io/grounded_video_caption_generation/)  
**领域**: 目标检测  
**关键词**: 视频定位字幕生成, 大规模预训练, 时空定位, 自动标注, 边界框预测

## 一句话总结

提出 GROVE 模型和大规模自动标注方法，构建包含 1M 视频的 HowToGround1M 预训练数据集和 3513 个视频的手动标注 iGround 数据集，实现联合视频字幕生成与多目标时空边界框定位，在 iGround、VidSTG、ActivityNet-Entities 等数据集上取得 SOTA。

## 研究背景与动机

视频中的定位字幕生成（Grounded Video Caption Generation）需要同时完成两项挑战性任务：（1）生成视频级的自然语言描述；（2）为字幕中的名词短语预测时间上稠密且一致的边界框。相比图像中的定位字幕，视频场景增加了物体可能因遮挡消失的困难，且需要跨帧产生时间一致的边界框。

该问题的关键瓶颈在于**缺乏大规模数据集**：

**现有数据集规模小**：VidSTG 有 36.2K 实例，HC-STVG 有 10.1K，人工标注成本高昂

**标注不完整**：许多数据集只为每个短文本描述定位单个时空管（tube），无法处理多目标

**时间一致性差**：ActivityNet-Entities 等数据集每个 segment 仅标注一帧的边界框，时间稀疏

**领域限制**：部分数据集仅限于特定场景（如自我中心视频）

此外，从图像定位字幕模型到视频场景的直接应用面临帧间不一致性问题——逐帧模型产生的名词短语标注在时间上不连贯。

## 方法详解

### 整体框架

GROVE 系统由两部分组成：

1. **大规模自动标注流程**（3 阶段）：从 HowTo100M 数据集中构建 HowToGround1M 预训练数据
2. **GROVE 模型**：基于 GLaMM 扩展到视频域的定位字幕生成模型，包含时空适配器、边界框解码器和时间存在性头

### 关键设计

1. **三阶段自动标注方法**:

    - 功能：将帧级定位字幕聚合为视频级的、时间一致的稠密标注
    - 核心思路：
        - **Stage 1（帧级定位字幕）**：对视频帧逐帧应用 GLaMM 图像定位字幕模型，获取帧级文本描述和边界框（将分割 mask 转为边界框）
        - **Stage 2（视频级字幕聚合）**：从帧级字幕中提取 SVO（主谓宾）三元组，使用 Llama-2 通过 in-context learning 聚合为视频级字幕，并标记关键名词短语
        - **Stage 3（时间一致标注）**：将帧级名词短语与视频级名词短语进行文本分类匹配（使用 LLM），确保同一物体在不同帧中获得一致标签，形成视频目标轨迹
    - 设计动机：纯图像模型逐帧处理会导致时间不一致（同一杯子在不同帧被标注为 "cup"、"mug"、"glass"），三阶段方法通过 LLM 的语义理解能力统一标注

2. **Spatio-temporal Adapters（时空适配器）**:

    - 功能：在冻结的图像编码器层之间插入可训练的时空适配器，使图像基础模型获得视频时序建模能力
    - 核心思路：$a(o) = o + \tanh(\alpha) \times f(o)$，其中 $o$ 是前一编码器层输出，$\alpha$ 是可训练参数（初始化为 0），$f(\cdot)$ 是适配器层。训练开始时适配器输出被 $\tanh(0)=0$ 有效屏蔽，训练网络自动调整适配器贡献
    - 设计动机：端到端微调整个视频编码器计算成本过高，且预训练图像编码器的知识应被保留。通过残差连接和零初始化的可训练参数实现稳定训练

3. **Bounding Box Decoder（边界框解码器）**:

    - 功能：将预训练的 mask 解码器改造为边界框解码器，预测每个检测 token 在每帧的边界框
    - 核心思路：使用嵌入的检测 token 作为 query，Grounding Video Encoder 的视觉特征作为 key/value。虽然 $\mathcal{V}_g(\cdot)$ 执行视频处理，但交叉注意力逐帧应用以预测每帧的目标：$p_{bb} = h_{bb}(o_d) \in \mathbb{R}^{T \times N_d \times 4}$
    - 设计动机：复用大规模预训练的解码器权重（如 SAM），将 mask 预测简化为边界框预测——边界框标注更便宜且对紧凑物体定位精度足够

4. **Temporal Objectness Head（时间存在性头）**:

    - 功能：显式预测每个物体在每帧中是否可见（是否被遮挡或离开画面）
    - 核心思路：$p_{tobj} = h_{tobj}(o_d) \in \mathbb{R}^{T \times N_d \times 1}$，推理时设定阈值过滤不可见帧的边界框预测
    - 设计动机：视频中物体频繁消失/重现是核心挑战。不同于图像检测中的 objectness（判断是否为目标），这里的时间 objectness 判断"目标在该帧是否可见"，解决了传统方法强行为遮挡帧预测边界框导致的误检

### 损失函数 / 训练策略

- 标准的字幕生成损失（语言模型交叉熵）
- 边界框回归损失（L1 + GIoU）
- 时间存在性损失（二元交叉熵）
- 预训练阶段冻结视觉骨干和 LLM，训练 LLM 的嵌入和输出层、适配器、解码器等
- 先在 HowToGround1M 上大规模预训练，再在小规模高质量数据（iGround）上微调

## 实验关键数据

### 主实验

| 数据集 | 指标 | GROVE PT+FT | 之前 SOTA | 提升 |
|--------|------|------------|----------|------|
| iGround (Center) | METEOR | 21.4 | 11.9 (GLaMM) | +9.5 |
| iGround (Center) | CIDEr | 83.5 | 29.9 (GLaMM) | +53.6 |
| iGround (Center) | AP50 | 31.7 | 20.8 (GLaMM) | +10.9 |
| iGround (All) | AP50 | 40.0 | 27.1 (自动标注) | +12.9 |
| iGround (All) | Recall | 28.7 | 20.4 (自动标注) | +8.3 |
| VidSTG (declarative) | msIoU | **63.7** | 61.9 (DenseVOC) | +1.8 |
| VidSTG (interrogative) | msIoU | **55.5** (FT) | 39.7 (VideoGLaMM) | +15.8 |
| ActivityNet-Entities | F1_loc_per_sent | **77.29** | 59.20 (GVD) | +18.09 |

### 消融实验

| 配置 | METEOR | CIDEr | AP50 | Recall | 说明 |
|------|--------|-------|------|--------|------|
| 无适配器、不解冻 | 19.2 | 82.2 | 36.8 | 25.9 | 基线最低 |
| 有适配器、不解冻 | 19.7 | 88.9 | 39.2 | 26.4 | 适配器贡献明显 |
| 有适配器 + 解冻解码器 | **19.7** | **92.6** | **42.0** | **26.9** | 两者互补 |
| 时间 objectness 阈值 0.0 | - | - | ~34 | ~28 | 无 objectness |
| 时间 objectness 阈值 0.3 | - | - | ~42 | ~27 | AP50 大幅提升 |

预训练数据量缩放实验：

| 预训练数据量 | CIDEr (PT) | AP50 (PT+FT) | Recall (PT+FT) |
|-------------|-----------|-------------|----------------|
| 1K | ~20 | ~33 | ~24 |
| 10K | ~30 | ~36 | ~25 |
| 100K | ~40 | ~39 | ~26 |
| 1M | ~50 | **~42** | **~27** |

### 关键发现

- **预训练至关重要**：仅微调（FT）的 GROVE 在 AP50 上只有 15.8，而预训练+微调（PT+FT）达到 40.0——差距巨大
- **数据量持续有效**：从 1K 到 1M 预训练视频，所有指标持续上升，无饱和迹象
- **自动标注 > 基线但 < 训练模型**：直接用自动标注方法获取预测（AP50=27.1）不如训练 GROVE 模型（AP50=33.6），说明模型能在训练中平滑掉标注噪声
- **SVO 三元组优于完整字幕**作为 LLM 的输入——完整字幕导致 LLM 过度精简输出
- **视觉追踪器在 Stage 3 反而有害**（CoTracker3 导致 AP50 下降 3.7%），因为视角变化导致追踪漂移

## 亮点与洞察

- **大规模数据+小规模精标注的组合范式**：用自动标注的 1M 视频预训练 + 3.5K 手动标注微调，这一"先量后质"的策略在多任务视觉-语言模型训练中具有广泛借鉴意义
- **LLM 驱动的标注一致性**：将跨帧名词短语统一问题转化为文本分类任务，巧妙利用 LLM 的语义理解能力，比视觉追踪器更鲁棒
- **时间存在性头的设计**简洁有效——用一个额外 MLP 头就解决了视频定位中物体消失/重现的核心问题
- 从 mask 解码器到 bbox 解码器的转换，通过复用大规模预训练权重实现了高效的知识迁移

## 局限与展望

- HowToGround1M 基于教学视频（HowTo100M），领域偏向性可能限制在其他视频类型上的泛化
- 自动标注流程中 GLaMM 作为帧级模型处理视频，更强的视频级字幕模型可能进一步提升标注质量
- 边界框定位精度不如像素级分割，对于细长或不规则形状物体可能不足
- iGround 标注规模（3.5K 视频）仍然有限，扩大规模可能带来进一步提升
- 当前仅处理教学视频中人与物体的交互，对开放域视频场景的泛化性有待验证

## 相关工作与启发

- GLaMM（图像定位字幕生成）是本文的直接技术基础，GROVE 将其扩展到视频域
- Moment-DETR 等视频时序定位方法通常假设文本查询已给出，而 GROVE 同时生成字幕和定位
- HowTo100M / HowToCaption 作为大规模视频数据源的持续价值——即使标注噪声大，足够的数据量仍能通过训练得到有效利用
- 自动标注方法中使用 POS tagging 提取 SVO 三元组是一个值得借鉴的文本预处理技巧

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题定义和数据构建方法有较强新意，模型设计偏增量
- 实验充分度: ⭐⭐⭐⭐⭐ 五个数据集、详细的数据缩放分析、自动标注各阶段的消融
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，贡献明确，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 数据集(HowToGround1M + iGround)和标注方法的贡献对社区有长久价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[CVPR 2025\] DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](../../CVPR2025/object_detection/diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)
- [\[ECCV 2024\] MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](../../ECCV2024/object_detection/mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)

</div>

<!-- RELATED:END -->
