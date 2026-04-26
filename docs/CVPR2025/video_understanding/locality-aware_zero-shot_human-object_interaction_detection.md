---
title: >-
  [论文解读] Locality-Aware Zero-Shot Human-Object Interaction Detection
description: >-
  [CVPR 2025][视频理解][零样本HOI检测] 提出 LAIN 框架，通过局部适配器（LA）和交互适配器（IA）增强 CLIP 表示的局部细粒度感知和交互推理能力，在多种零样本 HOI 检测设定下达到 SOTA。
tags:
  - CVPR 2025
  - 视频理解
  - 零样本HOI检测
  - CLIP适配
  - 局部感知
  - 交互推理
  - 视觉语言模型
---

# Locality-Aware Zero-Shot Human-Object Interaction Detection

**会议**: CVPR 2025  
**arXiv**: [2505.19503](https://arxiv.org/abs/2505.19503)  
**代码**: http://cvlab.postech.ac.kr/research/LAIN (有)  
**领域**: 视频理解  
**关键词**: 零样本HOI检测, CLIP适配, 局部感知, 交互推理, 视觉语言模型

## 一句话总结

提出 LAIN 框架，通过局部适配器（LA）和交互适配器（IA）增强 CLIP 表示的局部细粒度感知和交互推理能力，在多种零样本 HOI 检测设定下达到 SOTA。

## 研究背景与动机

零样本 HOI 检测的目标是识别训练中未见过的人-物交互类别。现有方法普遍利用 CLIP 的泛化能力，但面临关键适配难题：

1. **CLIP 偏向全局信息**：其图像级预训练导致 CLIP 擅长编码全局语义，但在区域级任务中无法捕捉细粒度的局部细节。例如，CLIP 对"骑自行车"的判断不依赖于人是否真正在自行车区域上，而是基于整体场景
2. **适配反而削弱泛化**：现有方法在适配 CLIP 用于 HOI 检测时，unseen 类别性能甚至低于 CLIP 原始零样本性能（如 UC-RF 和 UV 设定下）
3. **缺乏交互感知**：仅知道物体的局部细节不够，还需理解人体与物体之间的**交互模式**（如"骑"vs"修"自行车取决于手与把手的关系）

## 方法详解

### 整体框架

LAIN 是一个两阶段 HOI 检测框架：

1. 用预训练 DETR 检测图像中的目标
2. 构建所有有效的人-物对，生成 HO tokens
3. HO tokens 与图像 patch tokens 一起通过 CLIP 视觉编码器的 $L$ 层
4. 在每层 CLIP 前端插入 LA 和 IA，增强局部和交互感知
5. 最终 HO token 与文本嵌入计算余弦相似度得到 HOI 分数

### 关键设计

1. **Locality Adapter (LA)** — 增强 CLIP 的局部细粒度感知：
    - 将 patch tokens $F$ 投影到低维 $\tilde{F} \in \mathbb{R}^{H \times W \times D_a}$（$D_a \ll D_{clip}$）
    - 构造空间布局嵌入 $L_{i,j} = \text{FFN}([b_t; c_t; e_t])$，整合检测框坐标、置信度和物体文本嵌入
    - 使用**多尺度卷积**（不同核大小 $k_n$）聚合邻域信息：$L^{k_n} = \text{Conv}^{k_n}(\hat{F})$, $P = \text{FFN}(L^{k_1} + ... + L^{k_{N_c}})$
    - 通过可学习参数 $\gamma_{LA}$ 融合回原始特征：$F' = F + \gamma_{LA} \cdot \text{FFN}(P)$

2. **Interaction Adapter (IA)** — 捕捉人-物交互模式：
    - 用 ROIAlign 从更新后的 patch tokens $F'$ 提取人体和物体区域特征 $R_i^\tau$
    - **Interaction Pattern Reasoning Module (IPRM)**：通过可学习查询 $Q$ 用交叉注意力提取交互相关上下文 $\tilde{R}_i^\tau$，再让人体/物体上下文互相注意 $\hat{R}_i^h = \text{CrossAttn}(\tilde{R}_i^h, \tilde{R}_i^o, \tilde{R}_i^o)$
    - 将 HO token 投影为查询，从交互感知特征中提取信息并更新：$T_i' = T_i + \gamma_{IA} \cdot \text{FFN}([\bar{R}_i^h; \bar{R}_i^o])$

3. **HOI 评分与文本匹配**：
    - 文本模板："A photo of a person [verb-ing] a [object]"，前端插入可学习 tokens
    - HOI 分数：$S = \text{Sigmoid}(T_{(L)} E^\top / \tau)$，用 Sigmoid 而非 Softmax 因为一个人可同时与物体有多种交互
    - 推理时融合检测器置信度：$S_{infer} = S \cdot S_H^\lambda \cdot S_O^\lambda$

### 损失函数 / 训练策略

- 采用 binary focal loss：$\mathcal{L} = \text{FocalBCE}(S, Y)$
- IoU 阈值进行正样本分配
- CLIP 视觉编码器冻结，仅训练 LA 和 IA 中的适配器参数（参数高效）

## 实验关键数据

### 主实验

| 零样本设定 | 指标 (Full mAP) | LAIN | LAIN† (ViT-L) | 之前SOTA | 提升 |
|-----------|-----------------|------|---------------|----------|------|
| RF-UC | Full | **34.41** | 38.13 | 33.17 (LogicHOI) | +1.24 |
| NF-UC | Full | **33.23** | 36.22 | 31.39 (ADA-CM) | +1.84 |
| UO (Unseen Obj) | Full | **34.27** | 37.60 | 28.53 (HOICLIP) | +5.74 |
| UV (Unseen Verb) | Full | **33.12** | 37.20 | 31.09 (HOICLIP) | +2.03 |
| UC (Unseen Comp) | Full | **34.36** | 36.81 | 32.11 (CLIP4HOI) | +2.25 |
| HICO-DET 全监督 | Full | **36.02** | - | 35.33 (CLIP4HOI) | +0.69 |

### 消融实验

| 配置 | Unseen | Seen | Full | 说明 |
|------|--------|------|------|------|
| Baseline (无适配器) | 24.88 | 31.06 | 30.19 | 无 LA/IA |
| + LA only | 27.71 | 32.55 | 31.95 | 局部感知有效 |
| + IA only | 27.37 | 33.57 | 32.70 | 交互推理有效 |
| + LA + IA | **30.50** | **34.80** | **33.95** | 协同效果最佳 |
| LA: w/o 视觉信息 | 26.77 | 32.18 | 31.40 | 视觉上下文重要 |
| LA: w/o 空间布局 | 26.52 | 32.07 | 31.31 | 空间先验重要 |
| LA: Local Attention | 26.46 | 32.39 | 31.56 | 不如卷积 |
| IA: w/o IPRM | 24.32 | 32.76 | 31.57 | IPRM 关键 |
| IA: w/o 上下文提取 | 25.64 | 32.41 | 31.40 | 过滤噪声有效 |

### 关键发现

- **LA 和 IA 互补**：LA 提供细粒度物体细节，IA 利用这些细节进行交互推理。联合使用比各自单独提升更大（Unseen: +5.62 vs +2.83/+2.49）
- **现有方法适配 CLIP 反而降低泛化性**：在 RF-UC 和 UV 设定下，多个方法在 unseen 类上不如 CLIP 原始性能
- LAIN 用 ViT-B 已超越使用 ViT-L 的 BCOM†，说明方法层面的改进比纯增大模型更重要
- 在全监督设定下，LAIN 在稀有 HOI 类别上提升尤为显著（35.70），体现了强泛化能力
- 多尺度卷积比 Local/Window Attention 更适合捕捉邻域局部信息

## 亮点与洞察

- **问题定义精准**：清晰指出 CLIP 适配 HOI 检测时全局编码与区域级任务的 gap，以及适配反而削弱泛化的反直觉现象
- **LA + IA 的互补设计**优雅：LA 不改变 CLIP patch token 维度（通过残差连接），IA 不改变 HO token 维度，保证与冻结 CLIP 层的兼容
- **参数高效**：仅在 CLIP 每层前插入轻量适配器，冻结 CLIP 主体，训练代价小
- 空间布局嵌入引入了检测器输出（框坐标+类别+置信度），无需额外标注

## 局限与展望

- 依赖预训练 DETR 的检测质量，漏检或误检会直接影响后续 HOI 检测
- 仅在图像级 HOI 检测上验证，未拓展到视频 HOI 或时序场景
- 卷积核大小选择（$\mathbb{K}$）需手动设定，可能依赖数据集特性
- 文本模板固定，可能限制在更开放词表场景的泛化
- 部分 zero-shot 设定的 unseen 类别数量有限，需更大规模验证

## 相关工作与启发

- CLIP 的全局表示局限在 HOI/分割等区域级任务中是公共挑战，LA 的多尺度卷积 + 空间布局方案可迁移到其他任务
- IPRM 的交互推理思路与关系推理（Relation Networks）有相似之处，但更轻量
- 可学习的门控参数 $\gamma_{LA}, \gamma_{IA}$ 确保适配器在早期不破坏预训练表示

## 评分

- 新颖性: ⭐⭐⭐⭐ LA + IA 适配器设计新颖，问题分析深入
- 实验充分度: ⭐⭐⭐⭐⭐ 5种零样本设定 + 全监督 + 详尽消融
- 写作质量: ⭐⭐⭐⭐⭐ 动机图清晰, 公式规范, 逻辑严密
- 价值: ⭐⭐⭐⭐ CLIP 适配范式具有通用参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](../../ICCV2025/video_understanding/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](../../NeurIPS2025/video_understanding/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](progress-aware_video_frame_captioning.md)
- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](../../NeurIPS2025/video_understanding/part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)

<!-- RELATED:END -->
