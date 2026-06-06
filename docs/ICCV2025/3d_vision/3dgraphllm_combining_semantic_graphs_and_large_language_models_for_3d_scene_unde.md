---
title: >-
  [论文解读] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding
description: >-
  [ICCV 2025][3D视觉][3D场景理解] 本文提出3DGraphLLM，将3D场景中物体间的语义关系编码为可学习的图表示并输入LLM，在object grounding、场景描述和视觉问答等多个3D视觉-语言任务上显著超越不使用语义关系的基线方法，同时推理速度比LVLM方法快5倍。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D场景理解"
  - "场景图"
  - "大语言模型"
  - "视觉-语言任务"
  - "语义关系"
---

# 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding

**会议**: ICCV 2025  
**arXiv**: [2412.18450](https://arxiv.org/abs/2412.18450)  
**代码**: [https://github.com/CognitiveAISystems/3DGraphLLM](https://github.com/CognitiveAISystems/3DGraphLLM)  
**领域**: 3D视觉  
**关键词**: 3D场景理解, 场景图, 大语言模型, 视觉-语言任务, 语义关系

## 一句话总结
本文提出3DGraphLLM，将3D场景中物体间的语义关系编码为可学习的图表示并输入LLM，在object grounding、场景描述和视觉问答等多个3D视觉-语言任务上显著超越不使用语义关系的基线方法，同时推理速度比LVLM方法快5倍。

## 研究背景与动机
3D场景理解是具身智能体与用户交互的基础能力，涉及目标定位、场景描述和视觉问答等任务。LLM因其自然语言理解和推理能力，成为解决这类任务的有力工具。

现有将3D场景输入LLM的方法主要分为文本描述和可学习表示两类。文本描述虽然直观，但描述一个物体可能需要数百个token，推理速度慢。可学习表示更紧凑高效，但现有方法（如Chat-Scene）仅使用物体的几何坐标信息，忽略了物体之间丰富的语义关系。

核心矛盾在于：3D场景中物体的定位和描述往往依赖于物体间的空间和语义关系（如"桌子上的杯子"），但现有可学习表示方法无法编码这些关系信息，限制了LLM的推理能力。

本文的切入点是将3D场景图中的语义关系显式地编码为可学习embedding，以三元组形式（物体1, 关系, 物体2）输入LLM，弥补现有方法对语义关系建模的缺失。核心idea：构建基于k近邻子图的场景图平坦化表示，将语义边特征映射到LLM的token embedding空间。

## 方法详解

### 整体框架
3DGraphLLM将场景物体的点云作为输入，通过预训练编码器提取2D/3D物体特征和语义关系特征，经投影层映射到LLM的embedding空间。每个物体表示为一个局部子图（包含自身和k个最近邻），以三元组序列形式输入LLM。训练分为两阶段：先在GT分割上预训练，再在预测分割上微调。

### 关键设计
1. **物体特征编码**:

    - 功能：提取每个物体的2D和3D视觉特征
    - 核心思路：使用DINOv2提取2D特征 $Z_i^{2d} \in \mathbb{R}^{1\times1024}$（从多视角掩码图像聚合），使用Uni3D提取3D点云特征 $Z_i^{v_p} \in \mathbb{R}^{1\times1024}$（与文本描述对齐的特征）
    - 设计动机：DINOv2提供丰富的视觉语义，Uni3D在大规模数据上预训练具有良好的跨域泛化能力

2. **语义关系编码**:

    - 功能：为每对物体生成语义关系的潜在特征
    - 核心思路：使用VL-SAT方法生成关系特征 $Z_{ij}^e \in \mathbb{R}^{1\times512}$，取分类头之前的潜在特征而非离散类别，以捕获关系的组合语义
    - 设计动机：VL-SAT仅需3D坐标输入且利用CLIP知识迁移，跨域性能好；使用潜在特征而非分类结果可以捕获非互斥的关系组合（如"更大"且"靠近"）

3. **场景图平坦化表示**:

    - 功能：将完整场景图转化为适合LLM处理的token序列
    - 核心思路：对每个物体选取k个最近邻构建子图，以三元组 $(F_i^v, F_{ij}^e, F_j^v)$ 形式展开。完整图需要 $n(n-1)$ 条边，k近邻子图仅需 $2n + 3nk$ 个token。当 $k=2$, $n=100$ 时仅需800个token
    - 设计动机：物体与最近邻的关系对回答用户查询最相关；大幅减少token数量提升推理速度

### 损失函数 / 训练策略
- 使用标准的自回归语言模型损失：$L(\theta) = -\sum_{i=1}^{\ell} \log P(s_i^{res} | s_{[1,...,i-1]}^{res}, s^{prefix})$
- 两阶段训练：(1) 在GT实例分割上预训练投影层和LLM；(2) 在神经网络预测的分割上微调
- 联合训练多个任务：视觉定位、场景描述、视觉问答
- LLM使用LoRA（rank=16）微调，4×A100训练约24小时

## 实验关键数据

### 主实验

| 数据集 | 指标 | 3DGraphLLM (LLAMA3) | Chat-Scene (基线) | 提升 |
|--------|------|---------------------|-------------------|------|
| ScanRefer | Acc@0.5 | 56.6 | 50.2 | +6.4 |
| Multi3DRefer | F1@0.5 | 59.9 | 52.4 | +7.5 |
| Scan2Cap | CIDEr@0.5 | 81.0 | 77.1 | +3.9 |
| ScanQA | CIDEr | 88.8 | 87.7 | +1.1 |
| SQA3D | EM | 55.9 | 54.6 | +1.3 |

### 消融实验

| 配置 | ScanRefer Acc@0.5 | Multi3DRef F1@0.5 | 说明 |
|------|-------------------|-------------------|------|
| 3DGraphLLM-2 (LLAMA3, 预训练+3RScan) | 56.6 | 59.9 | 完整模型 |
| 3DGraphLLM-2 (LLAMA3, 无预训练) | 54.3 | 57.3 | 去掉预训练 |
| 3DGraphLLM-0 (LLAMA3, 无语义边) | 52.0 | 55.1 | 去掉语义关系 |
| 3DGraphLLM-2 (Vicuna, 预训练) | 53.1 | 57.3 | 较弱LLM |
| 3DGraphLLM-0 (Vicuna, 无语义边) | 50.2 | 52.4 | 基线Chat-Scene |

### 关键发现
- 加入语义关系后，LLAMA3在ScanRefer上提升+4.6%，Multi3DRefer上提升+4.8%，证明语义关系建模的重要性
- 两阶段预训练策略在GT分割上先学好投影再微调是有效的
- 3DGraphLLM仅需800个token描述场景，而GPT4Scene需要10400个，推理速度快5倍
- NMS过滤和最小距离过滤进一步提升了Mask3D分割下的性能

## 亮点与洞察
- 首个将3D场景图的语义关系编码为可学习表示并输入LLM的方法
- k近邻子图的平坦化策略巧妙地平衡了信息完整性和计算效率
- 使用VL-SAT的潜在特征而非分类结果是一个聪明的设计，避免了离散化导致的信息损失
- 两阶段训练（GT预训练+预测分割微调）有效缓解了分割噪声对语义关系编码的影响
- 推理效率高：每场景仅800个token，ScanRefer推理仅0.4秒/query

## 局限与展望
- n-gram指标（CIDEr/BLEU）不能准确评估LLM生成的丰富描述，可能低估模型性能
- 语义关系编码器VL-SAT在3RScan上训练，跨域到ScanNet可能存在性能差距
- 固定k=2的近邻数量可能对不同复杂度的场景不够灵活
- 对实例分割质量有较强依赖，分割噪声会直接影响图构建和关系提取
- 仅在室内场景（ScanNet/3RScan）验证，室外大规模场景的适用性未知

## 相关工作与启发
- **vs Chat-Scene**: 3DGraphLLM在其基础上引入语义关系编码，object grounding提升显著（ScanRefer +6.4%）
- **vs GPT4Scene**: 质量相当但token数量少13倍（800 vs 10400），推理速度快5倍，体现了紧凑表示的优势
- **vs Robin3D**: 使用370K数据达到Robin3D（1M数据）的水平，证明语义图表示的数据效率
- **vs 3D-VisTA等专家模型**: LLM-based方法的通用性优势，一个模型处理多种任务
- **启发**: 将结构化知识（图）注入LLM的范式可以推广到其他领域（如知识图谱增强的对话系统）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将场景图语义关系以可学习方式融入LLM，idea清晰且有效
- 实验充分度: ⭐⭐⭐⭐ 5个benchmark全面评测，消融实验详尽，包含GT和预测分割两种设定
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，prompt模板示例直观
- 价值: ⭐⭐⭐⭐ 为3D场景理解提供了高效的图-LLM融合范式，对具身智能有实际意义
- 总评: 扎实的工作，在紧凑表示和语义建模之间找到了好的平衡点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[CVPR 2025\] Empowering Large Language Models with 3D Situation Awareness](../../CVPR2025/3d_vision/empowering_large_language_models_with_3d_situation_awareness.md)
- [\[ICCV 2025\] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding](his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)

</div>

<!-- RELATED:END -->
