---
title: >-
  [论文解读] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models
description: >-
  [AAAI 2026][视频理解][音频语言模型] 提出 TimeAudio，通过时间标记（Temporal Markers）、绝对时间编码（Absolute Time-aware Encoding）和段级 Token 合并（Segment-level Token Merging）三个关键模块，赋予大型音频语言模型（LALM）精确的时间定位能力和端到端长音频理解能力，并构建了 FTAR 数据集用于细粒度时间推理的指令微调。
tags:
  - AAAI 2026
  - 视频理解
  - 音频语言模型
  - 时间定位
  - 细粒度音频理解
  - 长音频
  - Token合并
---

# Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models

**会议**: AAAI 2026  
**arXiv**: [2511.11039](https://arxiv.org/abs/2511.11039)  
**代码**: [github](https://github.com/lysanderism/TimeAudio)  
**领域**: 视频理解 / 音频理解  
**关键词**: 音频语言模型, 时间定位, 细粒度音频理解, 长音频, Token合并

## 一句话总结

提出 TimeAudio，通过时间标记（Temporal Markers）、绝对时间编码（Absolute Time-aware Encoding）和段级 Token 合并（Segment-level Token Merging）三个关键模块，赋予大型音频语言模型（LALM）精确的时间定位能力和端到端长音频理解能力，并构建了 FTAR 数据集用于细粒度时间推理的指令微调。

## 研究背景与动机

### 领域现状

大型音频语言模型（LALMs）如 Qwen2-Audio、SALMONN 等在音频内容理解和对话式问答任务上展现了强大能力，能够统一处理语音和环境音。这些模型通过将音频编码器与预训练的解码器式 LLM 集成，实现了自由格式的音频问答。

### 现有痛点

尽管取得了显著进展，当前 LALM 在**细粒度音频理解**方面存在严重不足，具体体现在三个层面：

**时间戳预测不准确**：现有模型难以准确预测事件的起止时间。例如 Qwen2-Audio 在密集音频描述任务上零样本 Eb-F1 仅为 9.8，说明事件定位能力极弱

**长音频处理能力受限**：大多数模型受限于短音频输入，无法端到端理解长音频，面对长音频时会产生严重幻觉

**缺乏合适的数据集和评估指标**：现有指令微调数据和评测基准主要关注通用音频理解，缺少针对细粒度时间推理的支撑

### 核心矛盾

音频理解本质上需要同时捕捉**语义内容**和**时间位置**，但现有 LALM 直接将音频特征映射到共享潜空间时，**没有显式建模细粒度的时间定位信息**，使得 LLM 解码器难以预测精确时间戳。这是一个架构-数据双重缺陷问题。

### 本文切入角度

从**时间戳表示**、**模型架构**和**数据**三个维度同时入手——引入离散时间标记替代连续数值回归，加入绝对时间编码增强时间感知，设计 Token 合并机制支持长音频，并构建 FTAR 数据集强化时间推理能力。

## 方法详解

### 整体框架

TimeAudio 基于 SALMONN 架构，由四个核心组件构成：
1. **滑窗音频编码器**（Sliding Audio Encoder）：将长音频分割为短段，使用 BEATs + Whisper 双编码器分别提取环境音和语音特征
2. **窗口 Q-Former**（Window Q-Former）：将编码后的音频 token 投影到语言空间
3. **段级 Token 合并模块**（Segment-level Token Merging）：基于注意力分数筛选重要 token，压缩冗余信息
4. **LLM**：处理融合后的音视频 token 和文本 token，生成响应

### 关键设计

#### 1. 时间标记（Temporal Markers）

**问题**：直接用数字 token 预测时间戳给 LLM 带来收敛困难；相对时间 token（如 \<0.2\> \<0.4\> \<0.6\>...）会给词表带来沉重负担且引入量化误差。

**核心思路**：设计锚点 token + 偏移 token 的两级离散化方案，将连续时间戳转换为离散时间 token 序列。

- **锚点 token** $\langle a_k \rangle$：表示整数秒级别的粗粒度位置
- **偏移 token** $\langle f_k \rangle$：表示亚秒级别的精细调整

关键创新在于**初始化策略**——不随机初始化这些新 token，而是从已有数字 token 的嵌入中迁移知识：

$$[\mathbf{W}_{\text{token}}]_{\text{ID}(\langle a_0 \rangle)} = [\mathbf{W}_{\text{token}}]_{\text{ID}(0)}$$

$$[\mathbf{W}_{\text{token}}]_{\text{ID}(\langle f_0 \rangle)} = \frac{[\mathbf{W}_{\text{token}}]_{\text{ID}(0)} + [\mathbf{W}_{\text{token}}]_{\text{ID}(\text{.})}}{2}$$

**设计动机**：利用 LLM 已有的数字理解能力，避免随机初始化破坏预训练嵌入空间。仅增加 M=20 个特殊 token，保持精度不受音频长度影响。

#### 2. 绝对时间编码（Absolute Time-aware Encoding, ATE）

**问题**：音频事件的时间顺序理解仍然困难，事件和语音韵律的多样性使得模型难以准确识别真实时间位置。

**核心思路**：构建可学习的绝对时间嵌入 $\mathbf{W}_t$，对每个音频段的特征显式注入绝对时间信息：

$$\hat{\mathbf{W}}_i = \mathbf{W}_i + \mathbf{e}_t(t_i)$$

其中 $\mathbf{e}_t(t_i) = [\mathbf{W}_t]_{j_i}$ 通过 one-hot 查表获得，$j_i$ 为离散化时间索引。

**设计动机**：
- 保留原有序列嵌入的相对位置信息，同时精确反映每个时间点在音频中的绝对位置
- 时间嵌入**零初始化**，保护预训练音频编码器在训练初期的完整性
- 作者认为绝对时间编码与位置编码是**正交**的，实验也验证了这一点

#### 3. 段级 Token 合并（Segment-level Token Merging, SEM）

**问题**：处理长音频时计算开销大，Q-Former 使用固定数量的查询进行对齐，token 数量随音频长度线性增长。

**核心思路**：复用 Q-Former 的注意力信息，在不引入额外计算的情况下自适应筛选和合并 token。分两步：

**步骤一：注意力 Token 选择**
- 计算 Q-Former 中的多头注意力矩阵 $\mathbf{A} = \text{Softmax}(\mathbf{QK}^T / \sqrt{D})$
- 对所有头取平均得到统一分数矩阵
- 保留得分最高的 token 作为"注意力 token"（实验中保留 22 个）

**步骤二：基于聚类的 Token 合并**
- 将剩余 token 均分为目标 token 和合并候选
- 用 key 向量的点积计算相似度：$\mathbf{Sim}(\mathbf{h}_i, \mathbf{h}_j) = \mathbf{k}_i \mathbf{k}_j^T$
- 将候选 token 分配到语义最相似的质心，融合为"上下文 token"（4 个）

**设计动机**：长语音中信息分散，通过自适应选择重要 token 并聚合冗余 token，在保留核心语义的同时压缩 **75%** 的冗余信息。

### 损失函数 / 训练策略

采用**两阶段训练**：

**Stage 1：时间 Token 对齐**
- 从已训练好的 LALM checkpoint 开始继续预训练
- 训练目标：对齐细粒度音频特征与时间信息
- 数据：时间音频定位、密集音频描述、时间线语音摘要
- 可训练部分：LoRA adapter、Window Q-Former、绝对时间嵌入和特殊文本嵌入
- 冻结部分：音频编码器和 LLM
- 10 epochs，学习率 1e-5

**Stage 2：长音频指令微调**
- 弥补 Stage 1 模型面对长音频时的语义不对齐
- 微调 Window Q-Former 和 LoRA
- 少量指令数据，5 epochs，学习率 2e-6

### FTAR 数据集

构建了包含 **260K** 音频-文本对的数据集，三大任务类型：
- **密集音频描述**（110K）：生成带时间戳的事件描述
- **时间音频定位**（100K）：根据文本查询定位时间段
- **时间线语音摘要**（42K）：带时间线的语音内容摘要
- **音频时间问答**（15K）：关于计数、时长、时间顺序的问答

## 实验关键数据

### 主实验

| 任务 | 指标 | TimeAudio | Qwen2-Audio (FTAR微调) | SALMONN-13B (FTAR微调) | 零样本 Qwen2-Audio |
|------|------|-----------|----------------------|---------------------|-------------------|
| 密集音频描述 | METEOR | 20.4 | **22.4** | 20.2 | 6.7 |
| 密集音频描述 | Eb-F1 | **37.4** | 36.5 | 32.0 | 9.8 |
| 密集音频描述 | At-F1 | **70.5** | 67.8 | 67.6 | 50.3 |
| 时间音频定位 | R@0.5 | **75.7** | 72.8 | 69.2 | 32.1 |
| 时间音频定位 | R@0.7 | **61.2** | 55.4 | 53.5 | 18.7 |
| 时间音频定位 | mIoU | **57.8** | 51.7 | 50.3 | 20.5 |
| 语音摘要 | ROUGE-1 | **42.4** | 40.0 | 40.2 | 17.4 |
| 语音摘要 | ROUGE-L | **30.8** | 28.5 | 29.8 | 12.3 |
| 语音摘要 | mIoU | **94.2** | 85.2 | 88.2 | 13.3 |

- 在时间音频定位任务上提升最大：mIoU 57.8 vs SALMONN-7B 的 51.9（+11.4%）
- 语音摘要 mIoU 高达 94.2，超越所有基线和专用模型 Hubert-MiniChat

### 消融实验

| 配置 | Eb-F1 | At-F1 | R@0.5 | mIoU(定位) | ROUGE-1 | mIoU(摘要) | 说明 |
|------|-------|-------|-------|-----------|---------|-----------|------|
| SALMONN(FTAR微调) | 32.4 | 68.0 | 71.4 | 51.9 | 39.5 | 84.3 | 基线 |
| +TM(未初始化) | 33.5 | 69.2 | 71.7 | 52.3 | 40.0 | 87.5 | 时间标记有效 |
| +TM(知识迁移初始化) | 36.0 | 70.8 | 72.7 | 54.9 | 41.0 | 88.7 | 初始化策略关键 |
| +TM+ATE | 37.8 | 71.4 | 73.7 | 56.0 | 41.4 | 90.4 | 绝对时间编码叠加 |
| +TM+ATE+SEM | 37.4 | 70.5 | 75.7 | 57.8 | 42.4 | **94.2** | 完整模型 |

**Token 保留比例消融**：

| 保留比例 | ROUGE-1 | ROUGE-L | mIoU |
|---------|---------|---------|------|
| 0.10 | 24.3 | 18.4 | 72.9 |
| 0.15 | 31.6 | 20.5 | 73.2 |
| 0.20 | 40.2 | 29.7 | 84.8 |
| **0.25** | **42.4** | **30.8** | **94.2** |
| 0.30 | 42.5 | 31.1 | 94.0 |

最终选择 0.25 作为性能和效率的平衡点。

### 关键发现

1. **时间标记的初始化策略至关重要**：从数字 token 迁移嵌入比随机初始化在 Eb-F1 上高 2.5 个点
2. **SEM 在长音频摘要上效果最突出**：mIoU 从 90.4 提升到 94.2，但在密集描述上略有下降（Eb-F1 从 37.8 降到 37.4），存在权衡
3. **绝对时间编码和位置编码正交**：两者可以叠加使用而不冲突
4. **Token 保留比例在 0.25 达到饱和**：再多保留收益甚微

## 亮点与洞察

1. **时间标记设计精巧**：锚点+偏移的两级方案既减少了词表膨胀，又保持了精度，且初始化策略利用了 LLM 已有的数字理解能力
2. **零额外计算的 Token 合并**：复用已有 Q-Former 注意力分数来筛选 token，无需额外模块
3. **FTAR 数据集填补空白**：260K 规模的时间敏感音频指令数据集，涵盖三大核心时间推理任务
4. **mIoU 指标在语音摘要上的创新应用**：首次用于衡量摘要的时间定位准确性

## 局限与展望

1. **METEOR 分数不如微调 Qwen2-Audio**：可能是基模型预训练时音频描述数据不够多样
2. **SEM 对密集描述有轻微负面影响**：信息压缩可能丢失某些局部事件细节
3. **仅 7B 规模**：更大模型（如 13B）可能进一步提升
4. **缺少对流式/超长音频的探索**：当前最大处理 5 个 30 秒段（2.5 分钟），真实场景可能需要更长
5. **FTAR 数据集部分依赖合成语音**：CNN/DailyMail 的 TTS 合成数据与真实语音存在分布差距

## 相关工作与启发

- **时间标记思想可迁移到视频领域**：类似设计可用于视频时间定位（Video Temporal Grounding）
- **Token 合并与 VisionZip 的联系**：该方法受视觉 Token 压缩启发，印证了跨模态方法迁移的可行性
- **双编码器架构的优势**：BEATs（环境音）+ Whisper（语音）的双编码器互补设计值得借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 时间标记和初始化策略有创新，但整体是改进式工作
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个任务、多个基线、完整消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- **价值**: ⭐⭐⭐⭐ — 填补了 LALM 时间推理的空白，FTAR 数据集有长期价值

<!-- RELATED:START -->

## 相关论文

- [Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [Causality Matters: How Temporal Information Emerges in Video Language Models](causality_matters_how_temporal_information_emerges_in_video_language_models.md)
- [On the Consistency of Video Large Language Models in Temporal Comprehension](../../CVPR2025/video_understanding/on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [Video Finetuning Improves Reasoning Between Frames](../../NeurIPS2025/video_understanding/video_finetuning_improves_reasoning_between_frames.md)
- [R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)

<!-- RELATED:END -->
