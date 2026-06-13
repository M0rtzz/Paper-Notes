---
title: >-
  [论文解读] Tracking and Segmenting Anything in Any Modality
description: >-
  [AAAI 2026][语义分割][统一跟踪分割] SATA提出了一个统一的跟踪与分割框架，通过解耦混合专家（DeMoE）机制建模跨模态共享知识和特有信息，并引入任务感知多目标跟踪（TaMOT）管线统一所有任务输出，在18个跟踪和分割benchmark上展现了优越性能。 领域现状：跟踪和分割是视频理解的基础任务…
tags:
  - "AAAI 2026"
  - "语义分割"
  - "统一跟踪分割"
  - "任意模态"
  - "混合专家"
  - "多任务学习"
  - "通用模型"
---

# Tracking and Segmenting Anything in Any Modality

**会议**: AAAI 2026  
**arXiv**: [2511.19475](https://arxiv.org/abs/2511.19475)  
**代码**: 有  
**领域**: 分割 / 视频理解  
**关键词**: 统一跟踪分割, 任意模态, 混合专家, 多任务学习, 通用模型

## 一句话总结
SATA提出了一个统一的跟踪与分割框架，通过解耦混合专家（DeMoE）机制建模跨模态共享知识和特有信息，并引入任务感知多目标跟踪（TaMOT）管线统一所有任务输出，在18个跟踪和分割benchmark上展现了优越性能。

## 研究背景与动机

**领域现状**：跟踪和分割是视频理解的基础任务。现有方法通常使用专用架构或模态特定参数来处理不同的子任务（如VOT、VOS、MOT、VIS等），限制了泛化性和可扩展性。

**现有痛点**：（1）不同模态（RGB、红外、深度等）之间存在分布差距，直接共享参数效果不好；（2）不同任务（跟踪 vs 分割、单目标 vs 多目标）之间存在特征表示差距，导致跨任务知识共享困难；（3）现有尝试统一这些任务的方法忽略了上述两个gap。

**核心矛盾**：要构建真正的通用模型（generalist model），必须同时处理跨模态分布差异和跨任务表示差异。

**本文目标**：构建一个统一框架处理广泛的跟踪和分割子任务以及任意模态输入。

**切入角度**：（1）用解耦MoE分离跨模态共享和特有的知识；（2）用统一的实例集合输出格式消除任务间的输出差异。

**核心 idea**：DeMoE将统一表示学习解耦为跨模态共享知识和模态特有信息的建模，TaMOT将所有任务输出统一为带校准ID的实例集合。

## 方法详解

### 整体框架
输入任意模态的视频序列，通过骨干网络提取特征，DeMoE自适应地分配共享和模态特有的专家进行特征增强，然后通过统一的解码器生成跟踪/分割结果。TaMOT管线统一所有子任务的输出格式为实例集合+ID信息。

### 关键设计

1. **解耦混合专家（DeMoE）**:

    - 功能：在统一框架中处理不同模态的分布差异
    - 核心思路：将标准MoE解耦为两组专家：共享专家学习跨模态不变的知识（如运动模式、物体形状），模态特有专家学习各模态独有的特征。路由器根据输入模态动态分配专家权重，使模型在保持灵活性的同时增强泛化
    - 设计动机：直接共享所有参数会因模态差异导致冲突；完全独立参数又无法利用跨模态共性；DeMoE在两者之间取得平衡

2. **任务感知多目标跟踪（TaMOT）管线**:

    - 功能：统一所有跟踪/分割子任务的输出格式
    - 核心思路：将所有任务的输出定义为统一的实例集合，每个实例包含空间位置（bbox/mask）和时序ID信息。通过任务代码（task token）区分不同子任务的推理模式。训练时用统一的实例匹配和ID关联损失
    - 设计动机：不同子任务的输出格式差异（单目标只有mask、多目标有bbox+ID等）阻碍了多任务统一训练，TaMOT通过格式统一解决

3. **多模态多任务联合训练**:

    - 功能：在一个模型中同时学习多种模态和任务
    - 核心思路：使用混合数据集训练策略，每个batch包含不同模态和任务的数据。DeMoE的路由器自动根据输入分配专家，TaMOT的任务代码指导解码。损失函数是各任务损失的加权和
    - 设计动机：联合训练允许跨任务知识迁移，DeMoE确保不同模态/任务不会相互干扰

### 损失函数 / 训练策略
联合训练采用分割损失（Dice + BCE）、检测损失（L1 + GIoU）和ID关联损失。使用任务代码区分不同子任务。

## 实验关键数据

### 主实验

| 任务 | Benchmark数量 | SATA排名 | 说明 |
|------|-------------|----------|------|
| 单目标跟踪 | 多个 | 顶尖 | RGB+红外+深度 |
| 多目标跟踪 | 多个 | 顶尖 | 统一ID管理 |
| 视频目标分割 | 多个 | 顶尖 | 半监督/无监督 |
| 视频实例分割 | 多个 | 顶尖 | 检测+分割+跟踪 |
| 总计 | 18个 | 全面领先 | 通用模型优势 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full SATA | 最佳 | DeMoE + TaMOT协同 |
| 标准MoE替代DeMoE | 下降 | 共享/特有未解耦 |
| 独立模态训练 | 下降 | 无跨模态知识迁移 |
| 无TaMOT | 各任务分别下降 | 输出格式不统一阻碍知识共享 |

### 关键发现
- 在18个benchmark上都取得顶尖或极具竞争力的结果，验证了通用模型的可行性
- DeMoE的解耦设计比标准MoE有明显优势，证明了跨模态共享/特有知识分离的重要性
- TaMOT的统一输出格式有效缓解了多任务训练中的任务特有知识退化问题

## 亮点与洞察
- **真正的通用跟踪分割模型**：一个模型覆盖18个benchmark、多种模态和多种任务，展示了构建视频理解基础模型的可能性
- **DeMoE的解耦思路**：将共享和特有知识显式分离的设计可以迁移到其他多模态/多任务学习场景
- **TaMOT的格式统一**：将异构任务输出统一为实例集合是一个优雅的工程贡献

## 局限与展望
- 18个benchmark的训练数据量很大，训练成本高
- DeMoE的专家数量和路由策略仍需手动设计
- 对于极端长视频或实时应用的效率未讨论
- 模型规模可能限制在嵌入式设备上的部署

## 相关工作与启发
- **vs SAM 2**：SAM 2在视频分割上很强但不做跟踪ID管理，SATA统一了跟踪和分割
- **vs UniTrack**：UniTrack尝试统一跟踪但不含分割，SATA范围更广
- **vs OneTracker**：OneTracker支持多模态跟踪但不含分割，SATA更全面

## 评分
- 新颖性: ⭐⭐⭐⭐ DeMoE解耦设计新颖，通用框架完整
- 实验充分度: ⭐⭐⭐⭐⭐ 18个benchmark的全面评估非常充分
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对统一视频理解模型有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](segment_and_matte_anything_in_a_unified_model.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](../../CVPR2026/segmentation/robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[ICML 2026\] Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](../../ICML2026/segmentation/beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)

</div>

<!-- RELATED:END -->
