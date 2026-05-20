---
title: >-
  [论文解读] A Two-Stage Dual-Modality Model for Facial Expression Recognition
description: >-
  [CVPR 2026][人体理解][面部表情识别] 提出两阶段双模态面部表情识别框架：Stage I 通过填充感知增强和训练期 MoE 头在外部数据集上适配 DINOv2 编码器；Stage II 通过多尺度面部裁剪、Wav2Vec 2.0 音频特征提取和门控融合实现帧级音视觉表情分类…
tags:
  - "CVPR 2026"
  - "人体理解"
  - "面部表情识别"
  - "DINOv2"
  - "音视觉融合"
  - "混合专家"
  - "数据增强"
---

# A Two-Stage Dual-Modality Model for Facial Expression Recognition

**会议**: CVPR 2026  
**arXiv**: [2603.12221](https://arxiv.org/abs/2603.12221)  
**代码**: 无  
**领域**: 人体理解 / 表情识别  
**关键词**: 面部表情识别, DINOv2, 音视觉融合, 混合专家, 数据增强

## 一句话总结

提出两阶段双模态面部表情识别框架：Stage I 通过填充感知增强和训练期 MoE 头在外部数据集上适配 DINOv2 编码器；Stage II 通过多尺度面部裁剪、Wav2Vec 2.0 音频特征提取和门控融合实现帧级音视觉表情分类，在 ABAW 2026 竞赛中取得 0.5368 Macro-F1。

## 研究背景与动机

帧级面部表情识别（EXPR）在野外环境中面临巨大挑战：面部定位不稳定、尺度变化大、模糊/遮挡/极端姿态/光照变化普遍。Aff-Wild2 数据集中的原始视频包含大量这些干扰因素，使得单帧视觉特征嘈杂且不一致。

此外，情感信号本质上是多模态的——视觉信息在歧义场景下可能不足以准确判断表情，而音频（如语气词、爆破音）可提供关键补充线索。然而，有效融合多模态数据并保持时间一致性仍是挑战。

本文策略：分两阶段解决——先在外部图像数据集上增强视觉编码器的表情感知能力，再在视频上进行多模态融合和时序平滑。

## 方法详解

### 整体框架

两阶段流水线：Stage I 在 AffectNet + RAF-DB 图像数据集上微调 DINOv2 ViT-L/14 编码器，引入 PadAug 增强和 MoE 训练头；Stage II 从视频中多尺度裁剪面部并提取视觉特征，同时用 Wav2Vec 2.0 提取帧对齐的音频特征，通过门控融合模块整合双模态信息，推理时施加时序平滑。

### 关键设计

1. **填充感知增强（PadAug）**:

    - 功能：提升模型对大尺度面部裁剪导致的边界伪影的鲁棒性
    - 核心思路：在训练图像边界插入黑色填充条并施加小空间扰动，模拟左/右/上/下/角落等各种边界模式。当裁剪区域超出图像边界时（野外视频常见），填充区域会引入分布偏移
    - 设计动机：多尺度面部裁剪（Stage II）中大裁剪框常超出画面，产生填充区域。PadAug 在训练时主动暴露这些模式，防止推理时的分布不匹配

2. **训练期混合专家（Training-only MoE）头**:

    - 功能：在视觉适配阶段提供更强的任务导向监督
    - 核心思路：在 DINOv2 编码器之后添加 MoE 分类头，通过样本依赖的专家路由增强适配效果。关键在于 MoE 头仅在 Stage I 训练时使用，训练完成后丢弃，仅保留微调后的 DINOv2 骨干
    - 设计动机：MoE 提供更丰富的梯度信号帮助 DINOv2 学习表情判别特征，但不增加推理复杂度

3. **门控音视觉融合 + 时序平滑**:

    - 功能：整合双模态信息并保持时序一致性
    - 核心思路：视觉特征来自三种尺度的面部裁剪（经 Stage I 适配的 DINOv2 编码），音频特征来自目标帧附近短窗口的 Wav2Vec 2.0 编码。两种特征通过轻量门控融合模块整合——学习每个模态的权重门控。推理时施加滑动窗口时序平滑以减少帧间预测抖动
    - 设计动机：音频在歧义视觉场景下提供补充线索（如愤怒的语气辅助区分愤怒和厌恶），但不应过度依赖（某些场景无有意义音频）。门控机制自适应调节模态权重

### 损失函数 / 训练策略

Stage I 使用交叉熵损失在 AffectNet + RAF-DB 上微调 DINOv2（8 类表情分类）。Stage II 在 Aff-Wild2 视频上训练门控融合模块。推理时施加时序平滑后处理。

## 实验关键数据

### 主实验

| 评估设置 | Macro-F1 | 说明 |
|---------|----------|------|
| 官方验证集 | **0.5368** | 最终提交结果 |
| 5 折交叉验证 | 0.5122 ± 0.0277 | 稳定性验证 |
| 官方测试集 | 0.391 | 挑战赛服务器 |

### 消融实验

| 配置 | Macro-F1 变化 | 说明 |
|------|-------------|------|
| 无 PadAug | 下降 | 边界伪影影响多尺度裁剪 |
| 无 MoE 头 | 下降 | 视觉适配效果减弱 |
| 仅视觉（无音频） | 下降 | 音频提供补充信息 |
| 无时序平滑 | 下降 | 帧间预测不稳定 |
| 完整模型 | **最优** | 各组件互补 |

### 关键发现

- 视觉编码器的域适配（Stage I）是性能的主要来源——图像通常包含视频中的主要情感信息
- 音频作为补充模态提供了最后一块拼图，但增益有限
- PadAug 对多尺度裁剪场景特别重要
- 验证集和测试集性能差距较大（0.54 vs 0.39），可能存在过拟合或分布偏移

## 亮点与洞察

- **PadAug 针对实际问题设计**：填充伪影是野外视频中普遍但被忽视的问题，这种针对性增强比通用增强更有效
- **训练期 MoE 的"用完即弃"策略**：利用 MoE 的丰富梯度帮助编码器适配，但不引入推理开销——巧妙的设计模式
- **两阶段解耦**：先在干净图像上学好视觉表示，再在嘈杂视频上做融合，避免了端到端训练中视频噪声污染视觉学习

## 局限与展望

- 验证集与测试集性能差距大（0.54→0.39），泛化性存疑
- 门控融合模块过于简单，未建模音视频的时序交互
- 仅使用 DINOv2 一种视觉骨干，未探索 CLIP 等其他预训练模型
- 时序平滑为后处理而非端到端学习的时序模型

## 相关工作与启发

- **vs MAE-based EXPR**: MAE 的重建目标对表情判别特征可能不如 DINOv2 的自蒸馏目标
- **vs CLIP-based EXPR**: CLIP 的文本对齐可能在表情分类的文本提示设计上提供优势
- **vs 多任务方法**: 某些 ABAW 方法联合训练 EXPR + AU + VA，可能通过共享表示提升性能

## 评分

- 新颖性: ⭐⭐⭐ PadAug 和训练期 MoE 有设计巧思，但整体框架较常规
- 实验充分度: ⭐⭐⭐⭐ 消融研究验证了各组件贡献，但测试集性能下降大
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐ 竞赛解决方案，实用技巧有参考价值但学术创新有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Generalizable Facial Expression Recognition](../../ECCV2024/human_understanding/generalizable_facial_expression_recognition.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[CVPR 2026\] A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)

</div>

<!-- RELATED:END -->
