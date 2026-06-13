---
title: >-
  [论文解读] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
description: >-
  [ICCV 2025][视频理解][多模态大语言模型] 提出一种无需训练的自适应推理方法，通过 LLM 前基于嵌入相似度的迭代式 token 合并 + LLM 层内基于 PageRank 多模态重要性的渐进式 token 剪枝，实现多模态 LLM 在 40 倍 FLOPs 减少范围内的灵活精度-效率权衡，在视频和图像理解任务上均取得优异表现。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "多模态大语言模型"
  - "自适应推理"
  - "Token Merging"
  - "剪枝"
  - "视觉Token冗余"
---

# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: [https://github.com/LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM)  
**领域**: 视频理解  
**关键词**: 多模态大语言模型, 自适应推理, Token Merging, Token Pruning, 视觉Token冗余

## 一句话总结

提出一种无需训练的自适应推理方法，通过 LLM 前基于嵌入相似度的迭代式 token 合并 + LLM 层内基于 PageRank 多模态重要性的渐进式 token 剪枝，实现多模态 LLM 在 40 倍 FLOPs 减少范围内的灵活精度-效率权衡，在视频和图像理解任务上均取得优异表现。

## 研究背景与动机

### 核心矛盾
多模态 LLM 依赖大量视觉 token（图像数百、视频数千）来实现视觉理解，但这带来了巨大的计算开销：
- **资源受限场景**：移动端设备、AR 眼镜等无法承受高计算量
- **长视频理解**：随着帧数增加 token 总量暴增，限制了模型处理密集视频帧的能力，导致关键时序信息丢失

### 为什么视觉 token 可以减少？
作者的核心观察是：**视觉数据存在大量内在冗余**。实验验证了仅保留 25% 的视觉 token 即可维持接近完整模型的性能。这种冗余为自适应推理提供了优化空间。

### 现有方法的不足
- **FastV / VTW**：在 LLM 的某一特定层剪枝/抛弃所有视觉 token，无法灵活适应不同计算约束
- **PDrop**：将 LLM 分为 4 个阶段，仅在阶段末剪枝
- **LLaVA-Prumerge**：仅在 LLM 前用视觉编码器的 K-V 对剪枝
- 这些方法要么仅在 LLM 前、要么仅在 LLM 内做 token 减少，且不支持**自适应推理**（即根据不同计算需求动态调整）

### 本文的关键洞察
1. 视觉 token 中约 75% 是冗余的
2. 更少的 token/帧 → 可以采样更多帧 → 对长视频理解更有利
3. LLM 早期层关注跨模态融合，晚期层关注文本推理 → 可以在晚期大量剪枝视觉 token

## 方法详解

### 整体框架

AIM 包含两个核心操作，形成"前后两段式" token 缩减策略：

1. **Token Merging（LLM 前）**：基于嵌入余弦相似度迭代合并高度相似的视觉 token
2. **Token Pruning（LLM 层内）**：基于 PageRank 算法在每个 LLM 层渐进剪枝不重要的视觉 token

关键设计理念：**training-free**（无需额外训练），直接应用于预训练模型的推理过程。

### 关键设计一：Token Merging before LLM

给定 LLM 输入前的视觉 token $v_0 \in \mathbb{R}^{N_0 \times D}$，执行如下迭代合并：

1. 将相邻 token 分为集合 A 和集合 B
2. 计算 A 中每个 token 与 B 中 token 的余弦相似度
3. 找到 A 中每个 token 在 B 中最相似的匹配
4. 将相似度最高的 token 对通过取平均嵌入进行合并
5. 每次迭代最多将 token 数减半，重复迭代（如 2 次）可达到目标保留率

**视频场景的特殊处理**：仅在单帧内合并 token，不跨帧合并。

**为什么不跨帧合并？** 消融实验表明，跨帧合并会破坏 token 的时序顺序，导致关键时序信息丢失，对视频理解任务有害。帧内合并则对最终推理性能影响极小。

**设计优势**：与在视觉编码器每层做合并不同（如 ToMe），本方法在视觉编码器之后做合并，对编码器架构无关，即插即用。

### 关键设计二：Token Pruning within LLM

合并后的视觉 token $v_1$ 与文本 token $t_1$ 拼接为 $x_1 = [v_1; t_1]$ 输入 LLM。在每个 LLM 层中：

**重要性评分 — PageRank 算法**：

利用注意力权重作为邻接矩阵，通过 PageRank 算法计算每个 token 的重要性分数：

$$s_i^l = \frac{1}{N_l + M_l} \sum_{j=1}^{N_l+M_l} A_{i,j}^l \cdot s_j^l$$

其中 $s_j^l$ 初始化为均匀分布，$A^l$ 是 softmax 归一化的注意力权重。

**仅剪枝视觉 token，保留所有文本 token**。

**为什么不剪文本 token？** 实验表明剪枝文本 token 会导致性能大幅下降（VideoMME 从 58.2 降至 45.7），因为 LLM 依赖文本 token 进行以文本为中心的推理。

### 关键设计三：分段式保留率调度器

设计分段函数控制每层的视觉 token 保留率 $r_l$：

$$r_l = \begin{cases} 1, & \text{if } l < l_1 \\ 1 - k(l - l_1), & \text{if } l_1 \leq l \leq l_2 \\ 0, & \text{if } l > l_2 \end{cases}$$

其中 $k = \frac{1}{l_2 - l_1}$ 为剪枝斜率。

| 参数 | 含义 | 视频LLM默认值 | 图像LLM默认值 |
|------|------|-------------|-------------|
| 合并保留率 | LLM 前保留的 token 比例 | 25% | 12.5% |
| $l_1$ | 开始剪枝的层 | 14 | 13 |
| $l_2$ | 完全移除视觉 token 的层 | 22 | 21 |

**为什么这样设计？** 基于关键发现：
- 早期层（<$l_1$）负责跨模态融合，此时剪枝视觉 token 会严重影响性能
- 中间层（$l_1 \leq l \leq l_2$）渐进剪枝，平衡信息保留与效率
- 晚期层（>$l_2$）主要做文本推理，不需要视觉 token

### 自适应推理

通过调整合并保留率和调度器参数 $(l_1, l_2)$，实现从无性能损失到极致效率的连续控制：
- 保守配置：50% 合并保留率 → 46.48 TFLOPs，性能不降反升
- 默认配置：25% 合并 + (14,22) 剪枝 → 14.76 TFLOPs，性能持平
- 极致配置：1.6% 合并 + (14,22) 剪枝 → 2.51 TFLOPs，性能下降约 13%

### 损失函数 / 训练策略

**无需训练**。方法直接作用于预训练模型推理过程。方法引入的额外计算开销极小：
- 视频 LLM（Qwen2-7B）：token merging 88.25 GFLOPs + pruning 4.18 GFLOPs → 仅占 LLM 推理 FLOPs 的 0.6%
- 图像 LLM（Vicuna-v1.5-7B）：总共 0.26 GFLOPs → 仅占 0.03%

## 实验关键数据

### 主实验

**视频基准测试**（基模型：LLaVA-OV-7B，32帧）：

| 方法 | FLOPs(TB) | Prefill(ms) | VideoMME | MVBench | MLVU | EgoSchema |
|------|-----------|-------------|----------|---------|------|-----------|
| LLaVA-OV-7B | 99.63 | 439.58 | 58.2 | 56.7 | 64.7 | 60.1 |
| FastV | 21.24 | 79.56 | 55.9 | 55.9 | 61.1 | 57.5 |
| LLaVA-Prumerge | 23.65 | 86.89 | 57.0 | 56.5 | 60.6 | 61.0 |
| **AIM** | **14.76** | **55.03** | **58.2** | **57.1** | **63.7** | **59.6** |

AIM 以最少的计算量（14.76 TB FLOPs，为基模型的 1/6.8）达到了接近零损失的性能（VideoMME 58.2 持平，MVBench 还略有提升）。

**长视频增强**：相同计算预算下采样 192 帧 vs 基模型 32 帧：

| 配置 | 帧数 | FLOPs(TB) | VideoMME | MLVU |
|------|------|-----------|----------|------|
| LLaVA-OV-7B | 32 | 99.63 | 58.2 | 64.7 |
| AIM | 32 | 14.76 | 58.2 | 63.7 |
| **AIM** | **192** | **99.27** | **59.2** | **69.3** |

在 MLVU（长视频理解）上提升 **+4.6**，验证了"更少 token/帧 → 更多帧 → 更好的长视频理解"假设。

**图像基准测试**（基模型：LLaVA-1.5-7B）：

| 方法 | FLOPs(TB) | VQA-v2 | GQA | MME | POPE |
|------|-----------|--------|-----|-----|------|
| LLaVA-1.5-7B | 8.18 | 78.5 | 62.0 | 1510.7 | 85.9 |
| FastV | 2.58 | 74.1 | 56.6 | 1438.5 | 73.6 |
| LLaVA-Prumerge+ | 2.41 | 74.6 | 57.4 | 1391.9 | 82.2 |
| **AIM** | **2.22** | **75.4** | **58.6** | **1443.5** | **85.7** |

### 消融实验

**Token Merging 保留率消融**（禁用 pruning）：

| 保留率 | FLOPs(TB) | Prefill(ms) | VideoMME |
|--------|-----------|-------------|----------|
| 100% | 99.63 | 439.58 | 58.2 |
| 50% | 46.48 | 182.65 | **58.5** |
| 25% | 22.90 | 83.94 | 58.0 |
| 12.5% | 11.64 | 41.22 | 56.6 |
| 3.1% | 3.85 | 13.68 | 52.3 |

保留 25% 以上 token 时性能几乎不变 → 约 75% 视觉 token 是冗余的。

**Token Pruning 调度器消融**（25% 合并保留率）：

| $l_1$ | $l_2$ | FLOPs(TB) | VideoMME |
|-------|-------|-----------|----------|
| 28 | 29 | 22.90 | 58.0 |
| 14 | 22 | 14.76 | **58.2** |
| 14 | 15 | 12.10 | 54.3 |
| 7 | 8 | 6.71 | 41.9 |

从 layer 8 就开始移除视觉 token 导致性能剧降（58.0→41.9），但从 layer 22 开始移除完全无损。

**文本 Token 剪枝消融**：

| 设置 | VideoMME |
|------|----------|
| 仅剪视觉 token | 58.2 |
| 视觉 + 文本均剪 | 45.7 |

剪枝文本 token 导致 **-12.5** 的严重下降。

### 关键发现

1. **75% 视觉 token 冗余**：仅 25% token 即可维持性能
2. **LLM 层级行为差异**：早期层做跨模态融合（不能剪），晚期层做文本推理（可以大量剪）
3. **文本 token 不可动**：文本 token 是 LLM 推理的核心，任何剪枝都会严重降低性能
4. **长视频理解的加速优势**：压缩 token 使得相同计算预算下可采样更多帧，MLVU +4.6
5. **方法是通用的**：同时适用于视频和图像 LLM，且对不同 LLM 架构（Qwen2、Vicuna）均有效

## 亮点与洞察

1. **自适应推理的实用价值**：一个方法覆盖 40 倍 FLOPs 范围，适配从 AR 眼镜到工作站的多种设备，真正解决了部署问题
2. **Training-free 设计**：无需重新训练或微调，即插即用地应用于已有预训练模型，迁移成本极低
3. **两阶段 token 缩减互补**：Merging（全局冗余去除）+ Pruning（层级自适应精修）的组合设计比单一策略更优
4. **PageRank 的创造性应用**：将网页排名算法引入注意力权重分析，比简单的 attention score 更能全面评估 token 重要性
5. **关于 LLM 层级行为的深刻洞察**：早期跨模态融合 + 晚期文本推理的发现，对未来多模态 LLM 设计有指导意义

## 局限与展望

1. **TextVQA 性能不佳**：对文本密集型图像表现较差，因为合并可能丢失细粒度文本信息
2. **调度器参数需手动设定**：$l_1, l_2$ 的最优值依赖于具体模型和任务，目前需要启发式选择
3. **仅处理视觉 token**：未考虑文本 token 的冗余，虽然实验表明文本 token 不能剪，但优化其表示也值得探索
4. **PageRank 计算开销**：虽然开销很小，但在极端低延迟场景下可能有改进空间
5. **缺乏对生成任务的验证**：当前仅在理解任务（VQA、选择题）上验证，视觉生成场景未探索

## 相关工作与启发

- **ToMe**：在视觉 Transformer 每层做 token 合并的开创性工作，本文将其思想迁移到 LLM 输入端
- **FastV**：首个在 LLM 层内剪枝视觉 token 的工作，但仅在单一层做，非渐进式
- **自适应计算**：本文将经典的自适应推理（adaptive inference）概念自然地引入多模态 LLM，填补了一个重要空白
- 对领域的启发：未来多模态 LLM 设计应考虑层级角色分工（跨模态融合 vs 文本推理），或许可以在架构设计阶段就进行优化

## 评分

- 新颖性: ⭐⭐⭐⭐ （组合已有技术但设计巧妙，自适应推理视角新颖）
- 实验充分度: ⭐⭐⭐⭐⭐ （视频+图像、多基模型、全面消融、计算开销分析）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，结论明确，insight 丰富）
- 价值: ⭐⭐⭐⭐⭐ （training-free + 自适应推理，实用性极强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] Breaking the Encoder Barrier for Seamless Video-Language Understanding](breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)

</div>

<!-- RELATED:END -->
