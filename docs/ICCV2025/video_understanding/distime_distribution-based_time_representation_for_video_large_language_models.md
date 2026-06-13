---
title: >-
  [论文解读] DisTime: Distribution-based Time Representation for Video Large Language Models
description: >-
  [ICCV 2025][视频理解][Video-LLM] 提出DisTime框架，通过一个可学习的时间token和基于分布的时间解码器，在Video-LLM中实现连续时间表示，配合大规模自动标注数据集InternVid-TG（125万事件），在时刻检索、密集视频描述、Grounded-VQA三类时间敏感任务上达到SOTA。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "Video-LLM"
  - "时间表示"
  - "分布式解码"
  - "时序定位"
  - "时间敏感数据集"
---

# DisTime: Distribution-based Time Representation for Video Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2505.24329](https://arxiv.org/abs/2505.24329)  
**代码**: [GitHub](https://github.com/josephzpng/DisTime)  
**领域**: 视频理解 / 时序定位  
**关键词**: Video-LLM, 时间表示, 分布式解码, 时序定位, 时间敏感数据集

## 一句话总结

提出DisTime框架，通过一个可学习的时间token和基于分布的时间解码器，在Video-LLM中实现连续时间表示，配合大规模自动标注数据集InternVid-TG（125万事件），在时刻检索、密集视频描述、Grounded-VQA三类时间敏感任务上达到SOTA。

## 研究背景与动机

当前Video-LLM在通用视频理解上表现出色，但在**精确时间定位**任务上存在根本性缺陷。现有的时间表达方案各有局限：

**文本模态离散化** (如VTimeLLM、TimeMarker)：用文本数字表示时间，但时间和数值共享决策边界，增加分类混淆

**多token离散化** (如Momentor、VTG-LLM)：专门引入大量时间token，但受训练数据长尾分布影响，部分token训练不充分，且缺乏时间连续性建模

**专用时间头** (如InternVideo2.5)：添加大量参数的时间感知模块（如CG-DETR），计算开销大且需要二次输入视觉信息

此外，现有时间敏感数据集存在**时间粒度约束**——VTimeLLM依赖镜头边界、InternVid-MR用固定2秒窗口、Momentor依赖镜头一致性——这些粗粒度方式无法准确捕获事件时间边界。

## 方法详解

### 整体框架

DisTime由五个核心组件构成：视觉编码器+投影器、文本编码器、LLM、时间解码器 $\Phi_{\text{time-dec}}$ 和时间编码器 $\Phi_{\text{time-enc}}$。采样的视频帧经视觉编码后，与对应时间戳的时间token交错拼接，连同用户指令一起输入LLM。当LLM生成 `<TIME_STAMP>` token时，其隐藏状态被送入时间解码器产生连续时间戳。

### 关键设计

1. **基于分布的时间Token (Distribution-based Time Token)**:
   使用单个可学习token `<TIME_STAMP>` 表示连续时间，将其与文本数字token区分开。核心创新在于**不直接回归绝对时间值**，而是先将token转换为时间分布，再通过加权求和得到时间戳。
    - 将归一化时间轴 $[0,1]$ 划分为 $reg_{max}+1$ 个离散锚点
    - 用MLP + softmax将 `<TIME_STAMP>` 的隐藏状态映射为分布向量 $\mathbf{e} \in \mathbb{R}^{2 \times (reg_{max}+1)}$
    - 通过锚点加权求和得到连续时间戳：$st = \sum_{i=0}^{reg_{max}} \mathbf{e}_{st}^{(i)} \cdot a_i$，其中 $a_i = i/reg_{max}$

   分布式解码的优势在于**建模事件边界的模糊性**——例如"一个人喝水"的起始时间是否包含拿杯子的动作？这种标注模糊性使得直接回归容易产生精度误差。

2. **时间编码器 (Time Encoder)**:
   解码器的逆操作，将连续时间戳编码回LLM可处理的时间token。先将时间戳投射为高斯正则化的分布 $p_{st} \sim \mathcal{N}(st, \delta^2)$，离散化后用MLP映射到LLM token空间：
    $\tau = \text{MLP}([\hat{\mathbf{e}}_{st}, \hat{\mathbf{e}}_{et}])$
   编码器极其轻量，仅占InternVL2.5-1B参数量的0.36%。

3. **迭代时间精炼 (Iterative Time Refinement)**:
   在LLM自回归生成过程中，当遇到 `<TIME_STAMP>` 时：将其隐藏状态解码为时间戳 → 重新编码为时间token → 替换原始token用于后续步骤。这种重编码操作将模糊的时间分布转化为标准化高斯表示，确保时间token间的分布对齐，增强LLM的时序理解一致性。

4. **InternVid-TG 数据集构建**:
   提出四步标注范式：
    - **事件提取**: 用GPT-4o从1fps图像序列中识别视频事件（~7事件/视频）
    - **事件定位**: 用三个专用模型（UniMD、Mr.Blip、TFVTG）独立定位事件边界
    - **评分集成**: 用InternVideo2计算视频-文本余弦相似度，为每个事件选择最高分模型的定位结果
    - **指令生成**: 设计5种对话模板转换为单轮训练对话
   最终生成179K视频上125万事件标注，规模超ActivityNet-Caption 55倍。

### 损失函数 / 训练策略

联合三个损失函数，权重均为1：
- $\mathcal{L}_{ntp}$：标准next token prediction损失
- $\mathcal{L}_{reg}$：1D-IoU回归损失，直接优化时间区间重叠度
- $\mathcal{L}_{dist}$：分布Focal Loss，学习时间分布

训练策略：冻结视觉骨干和中间层，仅用LoRA微调LLM，全量训练token embedding、LLM head、时间编/解码器。

## 实验关键数据

### 主实验

| 模型 | 规模 | Charades-STA R@1(IoU=0.3) | R@1(IoU=0.5) | ANet R@1(IoU=0.3) | R@1(IoU=0.5) |
|------|------|-------------------------|-------------|-------------------|-------------|
| VTimeLLM | 13B | 55.3 | 34.3 | 44.8 | 29.5 |
| TimeMarker | 8B | 73.5 | 51.9 | 67.4 | 50.7 |
| InternVL2.5 (基线) | 1B | 3.1 | 1.5 | 5.3 | 2.9 |
| **DisTime-InternVL** | **1B** | **78.1** | **56.3** | **67.1** | **45.4** |
| **DisTime-InternVL** | **8B** | **81.0** | **60.3** | **72.9** | **53.2** |
| Mr.BLIP (专用模型) | 3B | — | 69.3 | — | 53.9 |

### 消融实验

| Direct | Dist. | Re-Enc. | Charades R@1(0.5) | R@1(0.7) | YouCook2 F1 |
|--------|-------|---------|-------------------|---------|-------------|
| ✓ | | | 51.9 | 24.9 | 2.2 |
| | ✓ | | 53.5 | 26.7 | 16.3 |
| | ✓ | ✓ | **56.3** | **29.7** | **20.5** |

| 训练数据 | Charades R@1(0.3) | QVHighlights R@1(0.3) |
|---------|-------------------|----------------------|
| Baseline | 77.4 | 38.7 |
| + VTimeLLM数据 | 76.2 | 51.0 |
| + Momentor数据 | 76.6 | 39.7 |
| + InternVid-TG | **78.1** | **54.1** |

### 关键发现

- **从3.1%到78.1%的跃升**: DisTime将InternVL2.5-1B在Charades-STA上的R@1(IoU=0.3)提升25倍，证明了时间表示方法对LLM时间感知的决定性影响
- **分布式解码明显优于直接回归**: YouCook2上F1从2.2%提升至16.3%，时间重编码进一步提升至20.5%
- **InternVid-TG数据质量优于规模更大的Momentor**: Momentor有146万事件但训练后Charades性能反降，说明标注噪声比规模更重要
- **在Charades-STA上的零样本结果超越所有专用模型和Video-LLM**（R@1(0.3)=81.0%）
- 方法可即插即用到InternVL2.5和LLaVA-OneVision两种不同架构的Video-LLM

## 亮点与洞察

- **极简设计的有效性**：仅一个额外token + 极轻量MLP解码器（占总参数0.36%），就能赋予LLM精确时间感知能力
- **分布 vs 点估计**：事件边界天然模糊，用分布建模比点回归更符合物理意义，这一洞察值得跨领域借鉴
- **数据标注范式创新**：LLM提取事件 + 专用模型定位 + 相似度评分集成，每一步用最擅长的工具，比端到端方案更可靠

## 局限与展望

- InternVL2.5仅采样16帧，对ANet-Caption等需要细粒度时间理解的任务帧数可能不足
- 时间token的自回归生成增加了推理延迟
- InternVid-TG的标注质量仍受三个对齐模型能力上限制约
- 目前仅支持保持时间对齐的输入token序列，不兼容全局时间聚合的模型（如LinVT）

## 相关工作与启发

- DFL（Distribution Focal Loss）最初用于目标检测中的边框回归，本文将其迁移到时间定位，是一个成功的跨领域借鉴
- 与TimeMarker的区别：后者直接将帧时间戳编入多模态输入但仍用文本表示时间，本文彻底分离了时间和数字的表示空间
- 启发：**时间表示的设计**可能是Video-LLM时间理解能力的瓶颈所在，而非模型规模或训练数据

## 评分

- **新颖性**: ⭐⭐⭐⭐ 分布式时间表示的想法巧妙且有理论支撑
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖MR/DVC/Grounded-VQA三类任务+通用QA，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 结构完整，方法描述清楚
- **价值**: ⭐⭐⭐⭐⭐ 解决了Video-LLM的关键短板，数据集贡献巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4d_bench_benchmarking_multimodal_llms_for_4d_object_understanding.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

</div>

<!-- RELATED:END -->
