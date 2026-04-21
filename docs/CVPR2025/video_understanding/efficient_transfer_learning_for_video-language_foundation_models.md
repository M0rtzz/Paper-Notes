---
title: >-
  [论文解读] Efficient Transfer Learning for Video-language Foundation Models
description: >-
  [CVPR 2025][视频理解][参数高效微调] 提出多模态时空适配器MSTA，通过视觉-语言共享投影层和时空描述引导的一致性约束，以仅2-7%的可训练参数实现视频-语言基础模型向下游任务的高效迁移。
tags:
  - CVPR 2025
  - 视频理解
  - 参数高效微调
  - 视频动作识别
  - 多模态适配器
  - 迁移学习
  - 泛化能力
---

# Efficient Transfer Learning for Video-language Foundation Models

**会议**: CVPR 2025  
**arXiv**: [2411.11223](https://arxiv.org/abs/2411.11223)  
**代码**: https://github.com/chenhaoxing/ETL4Video (有)  
**领域**: 视频理解  
**关键词**: 参数高效微调, 视频动作识别, 多模态适配器, 迁移学习, 泛化能力

## 一句话总结

提出多模态时空适配器MSTA，通过视觉-语言共享投影层和时空描述引导的一致性约束，以仅2-7%的可训练参数实现视频-语言基础模型向下游任务的高效迁移。

## 研究背景与动机

预训练的视频-语言基础模型（如CLIP、ViCLIP）在下游视频任务上需要适配。现有方法面临两个核心矛盾：

1. **参数量与泛化的矛盾**：ActionCLIP、XCLIP等方法引入大量额外参数来建模时序信息，虽然提升了下游任务性能，但导致灾难性遗忘，严重损害了对未见类别的泛化能力
2. **单模态PEFT的局限**：LoRA、AdaptFormer等参数高效方法虽然参数少，但它们设计用于单模态模型，独立应用于视觉和文本分支时忽略了模态间的交互，无法有效对齐视频与文本表示

此外，ViCLIP作为专门用于视频的预训练模型，现有基于CLIP的方法无法直接迁移，缺乏专门为其设计的高效微调方案。

## 方法详解

### 整体框架

以ViCLIP（用时空注意力替换CLIP原始注意力的视频版CLIP模型）为骨干，在视频编码器和文本编码器的高层Transformer块中注入轻量级的多模态时空适配器MSTA。训练时冻结预训练参数，仅优化适配器。同时使用时空描述引导的一致性约束（$\mathcal{L}_{CC}$）来减少过拟合。

### 关键设计

1. **MSTA适配器架构**:
    - 功能：在视频和文本分支之间建立参数高效的跨模态对齐
    - 核心思路：每个适配器由三部分组成——模态专属的下投影层 $\mathbf{W}_v^{kd}$/$\mathbf{W}_t^{kd}$、**跨模态共享**的中间投影层 $\mathbf{W}^{ks}$、以及模态专属的上投影层。视频分支的上投影分为空间上投影 $\mathbf{W}_v^{ku-s}$（线性层）和时间上投影 $\mathbf{W}_v^{ku-t}$（3D卷积层），两者输出相加。通过缩放系数 $\lambda$ 控制适配器输出的强度：$[c_j, x_j] = \mathcal{E}^j_v([c_{j-1}, x_{j-1}]) + \lambda \mathcal{A}^j_v([c_{j-1}, x_{j-1}])$
    - 设计动机：共享中间层可以在微调过程中同时接收来自视觉和文本两个模态的梯度更新，从而优化模态对齐；分离的上下投影层保留了各模态的特异性；空间+时间上投影分别增强了对空间特征和时序特征的适应能力

2. **选择性层注入策略**:
    - 功能：仅在高层Transformer块中添加适配器，保护低层通用特征
    - 核心思路：从第 $k$ 层开始添加MSTA到最后一层 $L$，低层 $1$ 到 $k-1$ 保持冻结。不同任务设置选择不同的 $k$ 值（base-to-novel用1-12层，few-shot用8-12层）
    - 设计动机：Transformer低层学习通用特征，高层学习任务特定特征。在few-shot等需要泛化的场景中，只微调高层可以更好地保留预训练知识

3. **时空描述引导的一致性约束**:
    - 功能：通过知识蒸馏防止过拟合、增强泛化
    - 核心思路：利用LLM（如DeepSeek）为每个动作类别生成空间描述 $\text{DES}_s$ 和时间描述 $\text{DES}_t$。将标准模板（"a video of {cls}"）输入可训练分支，将LLM生成的描述输入冻结的预训练分支，通过余弦距离一致性约束对齐两个分支的输出：$\mathcal{L}_{CC} = 2 - \cos(w^c, D_s^c) - \cos(w^c, D_t^c)$
    - 设计动机：知识蒸馏迫使可训练编码器不会偏离预训练模型太远；时空描述提供了比简单模板更丰富的语义信息，引导模型在时空语义空间中学习更具区分度的表示

### 损失函数 / 训练策略

最终损失为交叉熵损失与一致性约束的加权和：

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{CC}$$

其中 $\mathcal{L}_{CE}$ 是标准的视频-文本对比损失，$\alpha=1.0$ 为最优权重。使用AdamW优化器，权重衰减0.001，N=2个描述为最优。通过MSTA所有模块采用kaiming初始化。

## 实验关键数据

### 主实验（Base-to-Novel泛化，4个数据集平均HM）

| 方法 | 可训练参数 | K-400 HM | HMDB-51 HM | UCF-101 HM | SSv2 HM |
|------|-----------|----------|------------|------------|---------|
| ViFi-CLIP (全参微调) | 全部 | 68.2 | 62.5 | 78.7 | 14.2 |
| ViCLIP (全参微调) | 124.3M | 71.3 | 62.7 | 81.6 | 17.0 |
| +AdaptFormer | 7.9M | 71.1 | 64.3 | 82.3 | 17.0 |
| +LoRA | 9.4M | 70.9 | 64.0 | 82.1 | 16.0 |
| **+MSTA+$\mathcal{L}_{CC}$** | **8.7M** | **72.0** | **66.3** | **82.9** | **18.9** |

### 消融实验

| 配置 | Base | Novel | HM | 说明 |
|------|------|-------|----|------|
| 仅语言适配器 | 66.1 | 51.5 | 57.9 | 单模态不足 |
| 仅视觉适配器 | 65.7 | 51.7 | 57.9 | 单模态不足 |
| 无共享层 | 68.0 | 52.9 | 59.5 | 缺少跨模态对齐 |
| 完整MSTA | 68.6 | 53.5 | 60.1 | 共享层提升HM 0.6 |

### 关键发现

- MSTA在所有四个评估设置（零样本、少样本、base-to-novel、全监督）上均达到SOTA，且仅用2-7%的原模型参数
- 在SSv2（强时序依赖数据集）上，Novel类别精度从OST的11.5提升至16.5（+43%），验证了时空建模的重要性
- 共享投影层带来稳定的HM提升（59.5→60.1），证明跨模态梯度共享有效
- 一致性约束在few-shot学习中效果尤为明显，有效缓解了小样本过拟合
- 描述数N=2最优，N增大后LLM幻觉导致噪声增加
- 高层注入（8→12）在few-shot中优于全层注入

## 亮点与洞察

- **共享投影层**是整个方法的核心创新：一个简单的理念（跨模态共享中间层）带来显著效果提升，且几乎不增加参数
- **空间+时间双上投影**设计虽然简单但有效，利用线性层捕捉空间信息、3D卷积捕捉时序信息
- **LLM生成描述+知识蒸馏**将大语言模型的知识有效注入到下游任务中，是"LLM as teacher"的一个好范例
- 方法通用性强，可同时适配CLIP和ViCLIP

## 局限与展望

- 一致性约束依赖LLM生成质量，N增大后幻觉问题限制了其可扩展性
- 仅在ViT-B/16上验证，更大规模模型上的效果尚不确定
- 时空描述的生成是离线的、与训练解耦的，在线自适应生成可能更好
- 空间和时间上投影简单相加，更复杂的融合策略（如门控）可能带来进一步提升

## 相关工作与启发

- LoRA和AdaptFormer等PEFT方法虽然高效但忽略了多模态对齐，本文强调跨模态交互的重要性
- OST方法同样利用LLM生成描述，但进行全参数微调易过拟合；本文用描述做蒸馏约束更优雅
- MoTE方法引入temporal expert混合，参数量大（88M vs本文8.7M），效果反而略差

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨模态共享投影+描述引导一致性约束的组合有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 四个评估设置、六个数据集、详尽的消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰但部分公式符号较多需仔细阅读
- 价值: ⭐⭐⭐⭐ 提供了视频-语言模型高效迁移的实用方案，适用范围广

<!-- RELATED:START -->

## 相关论文

- [Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)
- [R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](../../ECCV2024/video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)

<!-- RELATED:END -->
