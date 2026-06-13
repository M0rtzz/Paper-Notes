---
title: >-
  [论文解读] URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model
description: >-
  [NeurIPS 2025][3D视觉][关节物体重建] 提出URDF-Anything，首个基于3D多模态大语言模型（MLLM）的端到端关节物体重建框架，通过[SEG] token机制实现几何分割与运动学参数的联合预测，在分割精度（mIoU提升17%）、参数误差（降低29%）和物理可执行性（超越基线50%）上均达到SOTA。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "关节物体重建"
  - "URDF"
  - "3D多模态大语言模型"
  - "数字孪生"
  - "机器人仿真"
---

# URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model

**会议**: NeurIPS 2025  
**arXiv**: [2511.00940](https://arxiv.org/abs/2511.00940)  
**代码**: [项目主页](https://lzvsdy.github.io/URDF-Anything/)  
**领域**: 3D视觉  
**关键词**: 关节物体重建, URDF, 3D多模态大语言模型, 数字孪生, 机器人仿真

## 一句话总结
提出URDF-Anything，首个基于3D多模态大语言模型（MLLM）的端到端关节物体重建框架，通过[SEG] token机制实现几何分割与运动学参数的联合预测，在分割精度（mIoU提升17%）、参数误差（降低29%）和物理可执行性（超越基线50%）上均达到SOTA。

## 研究背景与动机
构建关节物体（如门、抽屉、剪刀等）的高保真数字孪生对机器人仿真训练和具身AI世界模型构建至关重要。这些关节物体的核心特征是由多个刚性部件（links）通过关节（joints）连接，涉及几何形状和运动学参数（类型、原点、轴、限位）的复杂推理。

现有方法存在明显局限：（1）Real2Code将部件抽象为有向包围盒（OBB），丢失了关键几何细节，且用LLM预测运动学参数精度不高；（2）Articulate-Anything依赖网格资产库和迭代VLM精调，流程脆弱且受资产库限制；（3）URDFormer依赖硬编码的离散分类来分配运动学参数和检索mesh，保真度受限。这些方法的共同问题是**多阶段pipeline设计**——分割、参数预测、mesh生成各自独立，误差逐级传播。

本文从根本上转变了思路：利用3D MLLM对点云进行端到端处理，**联合**预测几何分割和运动学参数。3D MLLM天然适合此任务——它能处理多模态输入、编码大规模3D形状先验、直接理解空间关系输出精确坐标。创新性地引入[SEG] token机制，在自回归生成过程中同时输出运动学结构的符号表示和几何分割信号，实现从根本上的一致性。

## 方法详解

### 整体框架
URDF-Anything由三个阶段构成：（1）输入表示——从单视角或多视角RGB图像生成稠密3D点云；（2）多模态关节解析——3D MLLM联合预测部件分割和运动学参数；（3）Mesh转换——将分割后的点云转为mesh并组装URDF文件。

### 关键设计
1. **输入表示（Input Representation）**:

    - 多视角输入：使用DUSt3R从多视角RGB图像生成稠密3D点云$P_{obj} \in \mathbb{R}^{N \times 6}$
    - 单视角输入：先用扩散模型生成一致的多视角图像，再通过LGM重建3D几何
    - 设计动机：适应不同的输入条件（单目/多目），输出统一的整体点云表示

2. **基于3D MLLM的关节解析与[SEG] Token机制**:

    - 以ShapeLLM为骨干网络，结合点云编码器和LLM
    - 点云$P_{obj}$经3D编码器提取特征$F_{pc} \in \mathbb{R}^{M \times d_{pc}}$，文本指令经LLM词嵌入层得到$F_{txt}$
    - MLLM自回归输出：$Y_{output} = \text{MLLM}(F_{pc}, F_{txt})$
    - 关键创新：扩展词汇表引入[SEG] token。每个link描述关联一个[SEG] token（如`"link_0": "base_cabinet[SEG]"`），使得符号输出与几何分割紧密耦合
    - 设计动机：标准MLLM无法做逐点预测。受LISA启发引入[SEG] token，使MLLM在生成运动学结构的同时标记需要分割的几何区域

3. **从[SEG] Token到几何分割**:

    - 对每个生成的[SEG] token，利用其最终隐状态$h_{seg}$和前置类别token状态$h_{category}$融合：$h_{combined} = [h_{category}; h_{seg}]$
    - 融合表示经MLP映射为查询$H_{query}$，通过交叉注意力对点云特征$F'_{pc}$计算逐点分数：$y_{mask} = \text{CrossAttn}(Q=H_{query}, K=F'_{pc}, V=F'_{pc})$
    - sigmoid+threshold得到每个部件的二值分割掩码
    - 设计动机：通过交叉注意力实现[SEG] token隐状态与点云特征的高效交互，既利用了MLLM的语义理解又保持了几何分割的精细性

### 损失函数 / 训练策略
- 总损失为语言建模损失和分割损失的加权和：$L = \lambda_{text}L_{text} + \lambda_{seg}\sum_{i=1}^{N}L_{i,seg}$
- 分割损失结合BCE和Dice：$L_{seg} = \lambda_{bce}\text{BCE}(\hat{M}, M_{gt}) + \lambda_{dice}\text{DICE}(\hat{M}, M_{gt})$
- 使用LoRA（rank=8）高效微调ShapeLLM-7B，AdamW优化器，学习率0.0003
- 单卡A800（80GB）训练2.5小时完成

## 实验关键数据

### 主实验

| 任务/指标 | URDF-Anything | Articulate-Anything | Real2Code Oracle | URDFormer Oracle |
|----------|--------------|--------------------|-----------------|-----------------| 
| 分割 mIoU (ALL) | **0.63** | - | - | - |
| 分割 Count Acc (ALL) | **0.97** | - | - | - |
| Joint Type Error ↓ | **0.008** | 0.025 | 0.537 | 0.556 |
| Joint Axis Error ↓ | **0.132** | 0.145 | 1.006 | 0.374 |
| Joint Origin Error ↓ | **0.164** | 0.207 | 0.294 | 0.581 |
| 物理可执行率 (ALL) | **78%** | 52% | 41% | 24% |
| 物理可执行率 (OOD) | **71%** | 44% | 23% | 15% |

### 消融实验

| 配置 | Type Error ↓ | Axis Error ↓ | Origin Error ↓ | mIoU | Count Acc |
|------|-------------|-------------|---------------|------|-----------|
| OBB输入 | 0.42 | 0.70 | 0.47 | - | - |
| 纯点云(无文本) | 0.34 | 0.29 | 0.26 | - | - |
| Qwen2.5-VL-7B+ft(图像输入) | 0.38 | 0.81 | 0.18 | - | - |
| 仅运动学预测 | 0.009 | 0.138 | 0.175 | - | - |
| 仅分割 | - | - | - | 0.61 | 0.89 |
| **完整模型(点云+文本)** | **0.008** | **0.132** | **0.164** | **0.63** | **0.97** |

### 关键发现
- 2D图像MLLM（即使微调）也无法推理精确的3D运动学参数，证明3D点云输入的必要性
- 联合预测优于解耦预测：分割任务为运动学提供几何正则化，运动学推理为分割提供结构先验——两者互为"增强器"
- OOD物体上物理可执行率从基线的44%提升到71%，展现了强大的泛化能力
- 仅分割模型的mIoU（0.61）和Count Acc（0.89）均低于联合模型（0.63/0.97），说明运动学任务迫使模型学到更连贯的结构表示

## 亮点与洞察
- 首次将3D MLLM引入关节物体URDF重建，开辟了端到端新范式。[SEG] token机制的设计非常优雅——在自回归生成的文本流中自然嵌入几何分割信号，实现了符号和几何的无缝统一
- 联合预测vs解耦预测的消融实验提供了深刻洞察：这不仅是工程选择，而是两个任务之间存在本质的互惠关系。运动学结构约束帮助分割更准确，几何分割反过来正则化运动学参数
- 对比2D图像MLLM和3D点云MLLM的实验设计非常有说服力——即使微调过的Qwen2.5-VL-7B在轴误差上也远不如3D方法，证明了显式3D几何的不可替代性
- 训练效率出色：单卡A800仅需2.5小时，LoRA rank=8的轻量化微调策略使得方法易于复现

## 局限与展望
- 无法生成某些URDF属性（如质量、惯性矩），受训练数据和基座模型限制
- pipeline不是完全端到端：仍依赖外部点云到mesh的转换模块生成link几何
- 数值参数精度受限于token化生成方式，连续值被离散化为有限精度的token序列
- 仅在PartNet-Mobility数据集上评估，真实世界复杂场景（如遮挡、噪声点云）的验证不足
- 可探索结合隐式3D表示（如NeRF/3DGS）替代explicit mesh，提升几何质量
- 未处理多物体场景中的关节物体检测与分割问题

## 相关工作与启发
- **vs Real2Code**: Real2Code用OBB粗糙表示+LLM，丢失几何细节；URDF-Anything直接从点云端到端推理
- **vs Articulate-Anything**: 依赖mesh资产库做迭代精调，流程脆弱且受限于资产库；URDF-Anything不需要外部资产库
- **vs URDFormer**: 用硬编码离散分类分配参数和检索mesh，保真度低；URDF-Anything通过MLLM直接回归连续参数

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将3D MLLM用于端到端URDF重建的工作，[SEG] token机制巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融设计深思熟虑，但数据集单一（仅PartNet-Mobility）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机论述充分，但部分细节需看附录
- 价值: ⭐⭐⭐⭐⭐ 对机器人仿真和具身AI有直接应用价值，端到端范式有望推广到更多3D结构推理任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](../../CVPR2025/3d_vision/riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](../../CVPR2025/3d_vision/iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](../../CVPR2025/3d_vision/perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)

</div>

<!-- RELATED:END -->
