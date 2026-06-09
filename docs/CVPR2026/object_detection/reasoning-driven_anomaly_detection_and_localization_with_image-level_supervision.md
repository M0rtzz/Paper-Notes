---
title: >-
  [论文解读] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision
description: >-
  [CVPR 2026][目标检测][异常检测与定位] 提出 ReAL 和 CGRO 两个模块，通过提取 MLLM 自回归推理过程中的异常相关 token 并聚合其视觉注意力来生成像素级异常图，再通过一致性引导的强化学习对齐推理与视觉证据，实现仅凭图像级监督的端到端异常检测、定位与可解释推理。
tags:
  - "CVPR 2026"
  - "目标检测"
  - "异常检测与定位"
  - "推理驱动"
  - "图像级监督"
  - "MLLM注意力"
  - "强化学习"
---

# Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision

**会议**: CVPR 2026  
**arXiv**: [2603.27179](https://arxiv.org/abs/2603.27179)  
**代码**: [GitHub](https://github.com/YizhouJin313/ReADL)  
**领域**:目标检测
**关键词**: 异常检测与定位, 推理驱动, 图像级监督, MLLM注意力, 强化学习

## 一句话总结

提出 ReAL 和 CGRO 两个模块，通过提取 MLLM 自回归推理过程中的异常相关 token 并聚合其视觉注意力来生成像素级异常图，再通过一致性引导的强化学习对齐推理与视觉证据，实现仅凭图像级监督的端到端异常检测、定位与可解释推理。

## 研究背景与动机

工业异常检测面临多重挑战：
- **传统方法局限**：需要大量正常样本训练产品特定模型，部署成本高，跨产品线泛化能力差
- **MLLM 现有方案**：多数方法仅能做图像级检测和文本推理，像素级定位仍依赖外部视觉模块（如 AnomalyGPT 用预训练视觉专家，EIAD 用 SAM），导致误差传播、推理-定位不对齐、部署复杂度增加
- **端到端方案（如 OmniAD）**的问题：依赖稠密像素级标注和高质量推理标注，获取代价高且引入领域偏差

核心观察（Fig. 1）：在 MLLM 生成推理文本时，**仅有少量 token 的注意力聚焦于真实异常区域**，且这些 token 往往与异常相关语义（如"scratch"、"mark"）关联。大多数推理 token 的注意力分散或聚焦无关区域，会稀释定位精度。

## 方法详解

### 整体框架

给定图像 $\mathbf{X}_v$ 和文本提示（固定问题："Are there any defects or anomalies in the image?"），MLLM 生成包含推理过程和最终答案的输出序列。框架包含两个核心模块：
1. **ReAL（Reasoning-Driven Anomaly Localization）**：从推理 token 中筛选异常相关 token，聚合其视觉注意力生成像素级异常图
2. **CGRO（Consistency-Guided Reasoning Optimization）**：通过强化学习驱动推理-定位一致性，对齐推理 token 与视觉注意力

### 关键设计

1. **异常相关推理 token 识别（ReAL 核心）**：从两个互补维度评估每个推理 token：
    - **跨模态语义相关度 $S_T^r$**：计算推理 token 对输入文本中异常相关词（"defect"/"anomaly"/"abnormal"）的注意力权重之和，衡量语义上与异常概念的关联程度
    - **模态内注意力集中度 $S_I^r$**：对视觉注意力图二值化后提取连通分量，计算空间熵——低熵意味着注意力聚焦于特定区域（可能是异常），高熵意味着注意力分散
   
   通过双阈值筛选（$\hat{S}_T^r > \tau_t$ 且 $\hat{S}_I^r > \tau_i$）后，以复合权重 $w_r = \alpha\hat{S}_T^r + \beta\hat{S}_I^r$ 加权聚合保留 token 的视觉注意力图 $\mathbf{A}_{r,I}$，得到推理驱动的异常图 $\mathbf{A}_{\text{RDAM}}$。

2. **一致性引导的推理优化（CGRO）**：解决 MLLM 在有限监督下推理不一致的问题（如模型回答"存在异常"但推理过程描述图像为正常）。设计类别条件一致性奖励 $R_{\text{cons}}$：
    - 对异常图像（$y=1$）：鼓励 top-$t$ 推理 token 的注意力区域高空间一致性（Jaccard Index $\mathcal{J} > \delta_1$）
    - 对正常图像（$y=0$）：鼓励低空间一致性（$\mathcal{J} < \delta_2$），抑制在良性区域的虚假聚焦
   
   总奖励 $\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{fmt}} + \mathcal{R}_{\text{acc}} + \mathcal{R}_{\text{cons}}$，通过 GRPO 框架优化。

3. **端到端无需外部模块**：整个系统仅需一个 MLLM，不依赖任何外部分割（SAM）或检测模块，真正实现端到端异常检测 + 定位 + 可解释推理。训练仅需图像级标签（正常/异常）。

### 损失函数 / 训练策略

- 基于 Qwen2.5-VL-7B，LoRA 适配语言和跨模态层，视觉编码器冻结
- 训练数据：4K 工业图像（来自 VisA、GoodsAD、Vision、PR-REAL 等），仅图像级标注
- 每批 16 样本，每输入采样 8 个候选生成（GRPO）
- 图像统一缩放至 420×420
- 零样本评估（训练集与测试集无领域重叠）

## 实验关键数据

### 主实验

四个基准平均（MVTec-AD、WFDD、SDD、DTD），图像级 AUROC/ACC：

| 方法 | 参数量 | 监督类型 | 图像级 AVG(AUROC,ACC) | 像素级 AVG(AUROC,ACC) | 推理(ROUGE-L,SBERT) |
|------|--------|----------|----------------------|---------------------|---------------------|
| GPT-4.1 | — | — | 87.2, 88.4 | N/A | 20.8, 69.9 |
| Qwen2.5-VL+CGRO* | 7B | I | **83.9, 86.9** | **80.7, 97.1** | **27.1, 74.7** |
| Qwen2.5-VL+R1* | 7B | I | 80.0, 82.0 | 78.5, 96.7 | 26.3, 73.8 |
| AnomalyGPT | 7B | T+I+P | 71.1, 53.9 | 77.8, 98.4 | 11.9, 36.7 |
| Triad | 7B | T+I | 85.5, 83.8 | N/A | 8.6, 35.9 |

亮点：仅用图像级监督即达到与使用像素级密集标注的 AnomalyGPT 可比的定位性能。

### 消融实验

ReAL + CGRO 消融（Qwen2.5-VL-7B，四数据集平均）：

| 配置 | 图像级 AUROC | 像素级 AUROC | 像素级 ACC |
|------|-------------|-------------|-----------|
| Vanilla | 63.4 | 64.7 | 73.0 |
| Vanilla + ReAL | 63.4 | 61.7 | 85.6 |
| Vanilla + CGRO | 83.9 | 72.7 | 92.6 |
| **Full (ReAL+CGRO)** | **83.9** | **80.7** | **97.1** |

token 选择策略消融（像素级）：
- 仅 $S_I$: AUROC 74.1
- 仅 $S_T$: AUROC 76.7
- $S_T + S_I$（完整）: AUROC **80.7**

### 关键发现

- ReAL 和 CGRO 具有互补作用：CGRO 提升图像级检测（+20.5 AUROC），ReAL 提升像素级定位精度（+8.0 AUROC）
- 一致性奖励消除了推理-回答矛盾：不加 CGRO 时模型常"判异常但推理说正常"，注意力分散
- 从 3B 到 7B 参量模型，CGRO 增益一致（图像级 +15-20 AUROC）
- 推理质量和定位精度的提升相辅相成

## 亮点与洞察

- **核心洞察深刻**：发现 MLLM 推理过程中天然存在异常感知的注意力模式，只需正确地筛选和利用（而非引入外部模块）
- **监督效率极高**：仅用图像级标签（最廉价的标注）即达到像素级密集标注方法的可比性能
- **三维度统一**：一个模型同时完成检测、定位、可解释推理，无需外部模块
- **一致性奖励设计精巧**：通过 Jaccard Index 的类别条件约束，将推理质量和空间聚焦对齐

## 局限与展望

- 定位精度仍有提升空间（像素级 AUPR 13.3%，远低于专用分割方法）
- 推理 token 筛选依赖阈值超参数 $\tau_t, \tau_i$，不同产品可能需要调整
- 训练数据为其他公开 AD 数据集图像，可能引入域偏差
- GRPO 训练成本较高（每输入 8 个候选生成）
- 注意力机制的解释性虽强，但对复杂多缺陷场景的表现未知

## 相关工作与启发

- **与 LISA 对比**：LISA 用 [SEG] token + SAM 做推理分割，本文完全去掉外部分割模块
- **GRPO/R1 范式**：延续 DeepSeek-R1 的强化学习推理优化路线，但创新引入一致性奖励
- **与 OmniAD 对比**：OmniAD 需要密集标注做端到端，本文仅需图像级标注
- **启发**：注意力聚合策略可推广到其他需要 MLLM 空间定位的任务（如 referring segmentation、visual grounding）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 激活 MLLM 内在推理潜能实现像素级定位的思路极具创新性，一致性奖励设计自然优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ — 四个基准、多种 MLLM 对比（含 GPT-4 系列）、详细消融，说服力强
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清晰，但公式符号较多需要仔细跟踪
- **价值**: ⭐⭐⭐⭐⭐ — 显著降低工业异常检测的标注成本，为 MLLM 在工业质检中的应用打开新路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](../../ICCV2025/object_detection/visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](herod_heuristic_inspired_reasoning_data_efficient_rod.md)
- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)

</div>

<!-- RELATED:END -->
