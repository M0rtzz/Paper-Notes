---
title: >-
  [论文解读] TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment
description: >-
  [CVPR2026][表格识别] 提出TDATR框架，通过"先感知后融合"策略和结构引导的单元格定位模块，在有限标注数据下实现端到端表格识别，在7个基准上无需数据集特定微调即达到SOTA。
tags:
  - CVPR2026
  - 表格识别
  - 端到端
  - 细节感知学习
  - 单元格定位
  - 视觉-语言对齐
---

# TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment

**会议**: CVPR2026  
**arXiv**: [2603.22819](https://arxiv.org/abs/2603.22819)  
**代码**: [github.com/Chunchunwumu/TDATR.git](https://github.com/Chunchunwumu/TDATR.git)  
**领域**: 多模态VLM  
**关键词**: 表格识别, 端到端, 细节感知学习, 单元格定位, 视觉-语言对齐

## 一句话总结
提出TDATR框架，通过"先感知后融合"策略和结构引导的单元格定位模块，在有限标注数据下实现端到端表格识别，在7个基准上无需数据集特定微调即达到SOTA。

## 研究背景与动机
表格识别(TR)是文档分析的核心任务，需将表格图像转换为HTML等机器可读格式。现有方法主要分两类：
- **模块化TR**：分别建模表格结构(TSR)和内容(TCR)，独立训练后通过后处理融合，但忽略了结构与内容的内在依赖，导致次优整合和误差累积
- **端到端TR**：统一生成结构化输出，但严重依赖大规模TR标注数据，在数据受限场景下泛化能力差；且大多不提供单元格空间对应关系，缺乏可解释性

核心矛盾：端到端方法虽然简化了流程，但TR数据标注成本极高（需同时标注结构和内容），导致现有方法在真实场景多样性表格上表现不佳。

本文切入角度：将TR能力学习解耦为"感知"和"融合"两阶段——先通过多任务预训练获取细粒度表格细节感知能力，再用少量TR数据学习融合，并引入结构引导的单元格定位增强可解释性。

## 方法详解

### 整体框架
TDATR采用视觉编码器(Swin Transformer) + 多模态语言解码器 + 结构引导单元格定位(SGCL)模块的架构，按"perceive-then-fuse"策略分两阶段训练。

### 关键设计

1. **表格细节感知学习(Table Detail-Aware Learning)**：

    - 在统一的语言建模范式下设计两类预训练任务：
    - **内容识别任务**：空间有序文本检测、带框查询的文本检测、Markdown解析——利用大规模多源文档数据(网页、论文、README等)增强OCR和布局理解
    - **结构理解任务**：单元格检测、跨行跨列检测、行列检测、结构解析——从单元格级和行列级两个层次感知表格结构
    - 设计动机：通过在多样化文档数据上预训练，模型获得强鲁棒的结构和内容感知能力，无需依赖大量TR专用数据

2. **结构引导单元格定位(SGCL)**：

    - 从语言解码器不同层的隐状态中提取单元格表示，用可学习权重聚合
    - 对每个单元格，在`<td`和`</td>`标记之间做平均池化得到初始表示C
    - 将C投影到行/列特征空间，通过内积计算邻接矩阵生成结构掩码：$M_{xy}^k = \mathbb{1}[\text{Sigmoid}(\langle C_x^k, C_y^k \rangle / \text{dim}(C^k)) > 0]$
    - 用结构掩码引导双向上下文注意力增强C → C'
    - 基于C'用MLP回归初始框，再通过多分辨率视觉特征P'₃和P'₄经DAB-DETR解码层精细化
    - 设计动机：锚点由TR隐状态初始化，确保与TR输出一一对应，无需后处理和不稳定的二分匹配

3. **融合微调阶段**：

    - 用HTML表格解析任务训练，同时优化SGCL模块预测精确单元格坐标
    - 模型通过隐式聚合前一阶段学到的表格细节来完成端到端TR

### 损失函数 / 训练策略
感知阶段：所有任务用交叉熵损失 $L_{ce}$

融合阶段：$L_f = \lambda_{ce} L_{ce} + \lambda_b L_b + \lambda_{iou} L_{iou} + \lambda_m L_m + \lambda_s L_s$
- $L_b$: 单元格回归损失; $L_{iou}$: IoU损失
- $L_m$: 掩码对齐损失(Mask-DINO风格)，增强C'与图像特征的对齐
- $L_s$: 结构引导损失(BCE)，优化行列关系矩阵
- 权重: $\lambda_b=0.05, \lambda_{iou}=0.03, \lambda_m=0.03, \lambda_s=0.05, \lambda_{ce}=1.0$

两阶段各训练3个epoch，使用16×64GB 910B NPU。

## 实验关键数据

### 主实验

| 数据集 | 指标 | TDATR | 之前SOTA | 提升 |
|--------|------|-------|----------|------|
| iFLYTAB-full | TEDS | 93.22 | 84.36 (DeepSeek-OCR) | +8.86 |
| TabRecSet | TEDS | 92.70 | 70.70 (EDD) | +22.00 |
| PubTables-1M | TEDS | 97.97 | 95.48 (Dolphin) | +2.49 |
| PubTabNet | TEDS | 提供了-ft版本进一步提升 | - | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无细节感知学习 | 性能显著下降 | 验证了perceive-then-fuse策略 |
| 无SGCL | TEDS-D增大 | 结构-内容对齐变差 |
| TDATR-ft(PubTabNet额外微调) | TSR/TR均提升 | 确认数据量的收益 |

### 关键发现
- 端到端TR的TSR性能超越了专家TSR模型，证明内容识别对结构识别有正向促进
- 使用远少于基线的微调数据仍达SOTA，验证了解耦学习策略的有效性
- TEDS-Delta(TEDS-TEDS_S)显著优于模块化方法，说明端到端避免了后处理误差累积

## 亮点与洞察
- "先感知后融合"是一个优雅的解耦思路：将难以获取的端到端TR标注需求转化为更容易获取的文档数据预训练 + 少量TR微调
- SGCL模块设计精巧，利用TR解码过程的隐状态作为DAB-DETR的锚点初始化，避免了匈牙利匹配的训练不稳定问题
- 新数据集iFLYTAB-full填补了中文真实场景表格识别评估的空白

## 局限与展望
- 在PubTabNet等数字表格上TSR略逊于最佳专家模型，因为完整TR序列长度约为TSR的两倍，加大了生成难度
- 模型参数量600M，相比大型OCR VLM(Qwen2.5-VL-72B)虽小但仍较大
- 未探索与LLM backbone的集成，可能限制了对复杂文档上下文的理解

## 相关工作与启发
- Dolphin是同类端到端TR方法，TDATR在相同建模方式下超越其2.5%+ TEDS
- 与OmniParser等多解码器方法不同，TDATR用单解码器统一结构和内容生成
- 结构引导定位的思路可推广到其他需要布局-内容对齐的文档理解任务

## 评分
- 新颖性: ⭐⭐⭐⭐ perceive-then-fuse策略和SGCL模块设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 7个基准、无数据集特定微调、完整消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详实
- 价值: ⭐⭐⭐⭐ 在数据受限下表格识别的实用性很强

## 补充说明
- 视觉编码器采用Swin Transformer (300M)，语言解码器Transformer (300M)，共600M参数
- SGCL中DAB-DETR解码层数Ld=3，双向增强分支包含2个自注意力块和1个交叉注意力块
- 最大解码长度4096 token，输入图像长边不超过2048像素

<!-- RELATED:START -->

## 相关论文

- [SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)
- [EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)
- [Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)
- [Pixel2Phys: Distilling Governing Laws from Visual Dynamics](pixel2phys_distilling_governing_laws_from_visual_dynamics.md)

<!-- RELATED:END -->
