---
title: >-
  [论文解读] Leveraging Temporal Contextualization for Video Action Recognition
description: >-
  [ECCV 2024][视频理解][视频动作识别] 提出 **TC-CLIP** 框架，通过**时序上下文化(TC)** 机制将全局视频动作线索压缩为少量 context tokens 注入 CLIP 编码过程，并设计**视频条件提示(VP)** 模块将视觉信息注入文本端，在零样本、小样本、base-to-novel 和全监督四种设定下全面超越现有 CLIP-based 视频识别方法。
tags:
  - ECCV 2024
  - 视频理解
  - 视频动作识别
  - CLIP
  - 时序建模
  - Token聚合
  - 视频条件提示
---

# Leveraging Temporal Contextualization for Video Action Recognition

**会议**: ECCV 2024  
**arXiv**: [2404.09490](https://arxiv.org/abs/2404.09490)  
**代码**: [GitHub](https://github.com/naver-ai/tc-clip)  
**领域**: 视频理解 (Video Action Recognition)  
**关键词**: 视频动作识别, CLIP, 时序建模, Token聚合, 视频条件提示

## 一句话总结

提出 **TC-CLIP** 框架，通过**时序上下文化(TC)** 机制将全局视频动作线索压缩为少量 context tokens 注入 CLIP 编码过程，并设计**视频条件提示(VP)** 模块将视觉信息注入文本端，在零样本、小样本、base-to-novel 和全监督四种设定下全面超越现有 CLIP-based 视频识别方法。

## 研究背景与动机

### 问题背景

预训练视觉-语言模型（VLM）如 CLIP 在视频理解中展示了强大的泛化能力，但其图像级预训练的本质使其天然缺乏时序建模能力。现有将 CLIP 扩展到视频的方法在时序信息建模上存在根本局限。

### 现有方法的问题

作者系统梳理了现有时序建模策略的不足：

**Cross-frame attention**（X-CLIP, Vita-CLIP）：仅通过各帧 [CLS] token 交互获取时序信息，**缺乏 patch 级细节**

**Temporal window expansion**（Open-VCLIP）：仅引入相邻帧的 patch tokens 作为 key-value，**时序范围太窄**

**Joint space-time attention**（理论最优方案）：将所有帧的所有 patch 作为参考——但 CLIP 是用短图像-文本对预训练的，直接延长序列会导致**外推问题**，注意力质量严重退化

**帧级平均**（ViFi-CLIP）：简单平均帧表示，完全没有帧间信息交换

### 核心发现与动机

作者通过关键实验（Table 1）发现：
- 使用所有帧的 [CLS] tokens 作参考：几乎无提升（+0.1/−0.1/+0.3）
- 使用相邻帧 patch tokens：提升有限（+0.7/−0.1/+0.8）
- 使用**所有帧的所有 patch tokens**：性能**反而下降**（−3.8/−1.4/+0.0），证实了外推问题
- 但使用通过聚合得到的 **context tokens**：**一致提升**（+0.9/+0.5/+2.3）

**关键洞察**：reference tokens 的最佳形式不是原始 token，而是经过选择和聚合后的紧凑语义摘要。这既保持了 CLIP 的有效序列长度，又传递了全局时序信息。

## 方法详解

### 整体框架

TC-CLIP 在 CLIP 基础上添加两个模块：
1. **Temporal Contextualization (TC)**：在视觉编码器中逐层注入全局时序信息
2. **Video-conditional Prompting (VP)**：利用视觉领域的时序信息增强文本编码

### 关键设计

#### 1. Temporal Contextualization (TC) — 时序上下文化

**功能**：将全视频的关键时序信息压缩为少量 context tokens，作为自注意力的额外 key-value 对注入每层编码过程。

**核心思路**：三步流程——

**Step 1: 信息性 Token 选择**

由于视频中存在大量冗余 token（如背景），先基于注意力分数筛选各帧的重要 token。利用自注意力中 [CLS] token 对各 patch 的注意力分数：

$$\mathbf{a}(\mathbf{z}_t) = \text{Softmax}\left(\frac{\mathbf{q}_{\text{cls}} \mathbf{K}_{\mathbf{z}_t}^{\mathsf{T}}}{\sqrt{d}}\right)$$

对多头注意力分数取平均 $\bar{\mathbf{a}}_{t,i} = \sum_{h=1}^{H} \mathbf{a}_{t,i}^h / H$，选取 Top-$n_s$ 个 token 作为 seed tokens，其中 $\alpha = n_s / N$ 控制选择比例（默认 $\alpha = 0.3$）。

**Step 2: 时序上下文摘要**

收集所有帧的 seed tokens，通过聚合函数 $\phi$ 将其压缩为 $k$ 个 context tokens：

$$\hat{\mathbf{s}} = \phi\left(\{\hat{\mathbf{z}}_{t,i}\}_{(t,i) \in \mathcal{S}}\right)$$

默认采用 **bipartite soft matching** 进行 token 合并，将相似 token 聚为一组并取均值。

**Step 3: 时序上下文注入**

将 context tokens 扩展到自注意力的 key-value 中：

$$\text{Attention}_{\text{TC}}(\mathbf{z}_t, \mathbf{s}) = \text{Softmax}\left(\frac{\mathbf{Q}_{\mathbf{z}_t} [\mathbf{K}_{\mathbf{z}_t} | \mathbf{K}_{\mathbf{s}}]^{\mathsf{T}}}{\sqrt{d}} + \mathbf{B}\right) [\mathbf{V}_{\mathbf{z}_t} | \mathbf{V}_{\mathbf{s}}]$$

其中偏置矩阵 $\mathbf{B}$ 通过可学习参数 $b_{\text{local}}$ 和 $b_{\text{global}}$ 区分帧内局部信息和视频级全局信息，逐层逐头独立学习。

**设计动机**：
- 避免 CLIP 的序列长度外推问题
- 保留全局时序范围的 patch 级细节
- context tokens 如同"时序桥梁"传递视频级上下文

#### 2. Video-conditional Prompting (VP) — 视频条件提示

**功能**：将视觉编码器产生的 context tokens 信息注入文本端的 prompt 向量，生成实例级别的文本提示。

**核心思路**：通过交叉注意力将视频信息融入可学习的 prompt 向量：

$$\mathbf{s}_{\text{proj}}^l = \text{SG}(\mathbf{s}^l \mathbf{W}_{\text{vis}})$$
$$\hat{\mathbf{p}}^{l-1} = \text{MHCA}(\text{LN}_p(\mathbf{p}^{l-1}), \text{LN}_s(\mathbf{s}_{\text{proj}}^l)) + \mathbf{p}^{l-1}$$
$$\tilde{\mathbf{p}}^{l-1} = \text{FFN}(\text{LN}(\hat{\mathbf{p}}^{l-1})) + \hat{\mathbf{p}}^{l-1}$$

其中 $\text{SG}(\cdot)$ 为 stop-gradient，prompt 向量作为 query，context tokens 作为 key/value。VP 在文本编码器最后一层之前执行。

**设计动机**：动作识别数据集的文本描述仅限于类别名（如 "skateboarding"），缺乏详细叙述。VP 通过注入视频实例的视觉信息弥补文本语义的不足，使每个样本获得定制化的 prompt。

#### 3. 逐层构建

TC 以逐层方式运行：第 1 层使用标准 MHSA（因尚无 context tokens），后续每层先执行 token 选择 + 聚合得到新的 context tokens，再用其扩展自注意力的 key-value。context tokens 也通过独立的 FFN 更新。

### 训练策略

- 训练目标：标准交叉熵对比损失
$$\mathcal{L} = -\sum_i \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{c}_i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{v}_i, \mathbf{c}_j)/\tau)}$$
- 端到端全参数微调
- backbone：CLIP ViT-B/16
- 硬件：4 × NVIDIA Tesla V100

## 实验关键数据

### 主实验

**零样本动作识别**（K-400 训练，直接评估其他数据集 + Weight Ensemble）：

| 方法 | HMDB-51 | UCF-101 | K-600 | All Avg |
|------|:-------:|:-------:|:-----:|:-------:|
| Vanilla CLIP | 40.8 | 63.2 | 59.8 | 54.6 |
| X-CLIP | 44.6 | 72.0 | 65.2 | 60.6 |
| ViFi-CLIP (WE) | 52.2 | 81.0 | 73.9 | 69.0 |
| Open-VCLIP (WE) | 53.9 | 83.4 | 73.0 | 70.1 |
| **TC-CLIP (WE)** | **54.2** | 82.9 | **75.8** | **71.0** |
| TC-CLIP + LLM (WE) | **56.0** | **85.4** | **78.1** | **73.2** |

**全监督识别** (K-400)：

| 方法 | Top-1 | Top-5 | 帧数 |
|------|:-----:|:-----:|:----:|
| ActionCLIP | 83.8 | 96.2 | 32 |
| X-CLIP | 84.7 | 96.8 | 16 |
| ViFi-CLIP | 83.9 | 96.3 | 16 |
| **TC-CLIP** | **85.2** | **96.9** | 16 |

### 消融实验

**组件消融**（零样本 + Weight Ensemble）：

| 配置 | HMDB-51 | UCF-101 | K-600 | All (Δ) | 说明 |
|------|:-------:|:-------:|:-----:|:-------:|------|
| Baseline (ViFi-CLIP) | 52.2 | 81.0 | 73.9 | 69.0 | — |
| + TC | 54.3 | 81.9 | 75.5 | 70.6 (+1.6) | 时序上下文化独立有效 |
| + VP | 53.4 | 82.0 | 74.7 | 70.0 (+1.0) | 视频条件提示独立有效 |
| + TC + VP | 54.2 | 82.9 | 75.8 | 71.0 (+2.0) | 二者互补 |

**Token 聚合策略对比**（few-shot 平均 Top-1）：

| 策略 | HMDB | UCF | SSv2 | All (Δ) |
|------|:----:|:---:|:----:|:------:|
| Baseline (无参考 token) | 62.6 | 89.2 | 8.7 | 53.5 |
| 无合并（直接用 seed tokens） | 57.2 | 85.6 | 7.7 | 50.2 (−3.3) |
| 随机合并 | 58.8 | 87.1 | 7.5 | 51.2 (−2.3) |
| K-means | 62.1 | 89.7 | 9.0 | 53.6 (+0.1) |
| DPC-KNN | 63.3 | 90.2 | 9.8 | 54.4 (+0.9) |
| **Bipartite soft matching** | **63.4** | **90.2** | **9.9** | **54.5 (+1.0)** |

**计算成本对比**：

| 方法 | 参数量 | GFLOPs | 吞吐量 | Zero Avg | Full Top-1 |
|------|:-----:|:------:|:-----:|:--------:|:----------:|
| ViFi-CLIP | 124.3M | 285 | 38 | 69.0 | 83.9 |
| Open-VCLIP | 124.3M | 308 | 29 | 70.1 | - |
| TC-CLIP | 127.5M | 304 | 24 | 71.0 | 85.2 |
| TC-CLIP (轻量) | 127.5M | 291 | 34 | 70.7 | 84.9 |

### 关键发现

1. **Context tokens 是唯一一致有效的参考 token 形式**：Table 1 的初步实验清楚展示了其他形式（CLS/相邻 patch/全部 patch）的不足
2. **不合并会导致外推问题**：直接用 seed tokens 性能暴跌 3.3 点，验证了 CLIP 序列长度外推假说
3. **TC 在 WE 场景下增益更大**：从 +0.7 提升到 +1.6，说明 TC 学到的表示与 CLIP 原始表示更互补
4. **VP 弥补了文本端信息不足**：仅用可学习 prompt 向量反而降低零样本性能（−0.2），但用 VP 注入视觉信息后提升 +1.1
5. **SSv2 时序敏感数据集上提升最大**：base-to-novel 设定下 SSv2 从 5.1 HM 提升到 15.2 HM

## 亮点与洞察

1. **问题分析深入透彻**：从四种时序建模方式的系统对比（Figure 2-3），到 Table 1 的初步实验，再到最终方案，推导逻辑非常严密
2. **Context token 的设计理念精妙**：不是简单地扩展 token 序列，而是"先选择、再压缩、再注入"，在信息量和计算量之间找到了最佳平衡点
3. **VP 模块用 stop-gradient 防止视觉信号反传到文本编码器**：细节设计到位
4. **偏置矩阵 B 区分局部/全局信息**：简单但有效的设计，让模型逐层逐头学习如何平衡帧内和视频级信息
5. **可视化分析有说服力**：context token 跨帧追踪飞盘、注意力图对比清楚展示了时序理解能力差异

## 局限与展望

1. **推理吞吐量下降**：相比 ViFi-CLIP (38) 降到 24，轻量版 (34) 有所缓解但仍有开销
2. **仅在 ViT-B/16 上验证**：更大模型（ViT-L）上的表现未知
3. **Token 选择依赖 CLS 注意力**：对 CLS token 质量较为敏感
4. **全参数微调成本高**：相比 adapter 类方法参数效率较低
5. **对时序非常敏感的任务（如 SSv2）仍有提升空间**：few-shot 14.0% 的绝对值仍然偏低

## 相关工作与启发

- **ToMe (Token Merging)**：TC 中的 bipartite soft matching 直接来源于此，但创造性地将 token 压缩用于跨帧时序信息聚合
- **CoOp/CoCoOp (Prompt Learning)**：VP 模块的 prompt 学习思路源自此，但创新在于用视觉 context tokens 而非图像特征来条件化 prompt
- **CLIP → Video 系列**（ActionCLIP → X-CLIP → ViFi-CLIP → Open-VCLIP → TC-CLIP）：体现了该方向从简单到复杂的不断演进

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Context token 的"选择-聚合-注入"流程是一个有效的新范式，但各组件并非首创
- **实验充分度**: ⭐⭐⭐⭐⭐ — 四种评估协议 + 5 个基准 + 极其详尽的消融，实验设计堪称典范
- **写作质量**: ⭐⭐⭐⭐⭐ — 从动机分析到方法推导到实验验证，叙述逻辑严密清晰
- **价值**: ⭐⭐⭐⭐ — 提供了一种高效的 CLIP 视频扩展范式，实用性强，被后续工作广泛引用

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)
- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)

<!-- RELATED:END -->
