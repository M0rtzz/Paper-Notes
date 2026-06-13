---
title: >-
  [论文解读] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition
description: >-
  [AAAI 2026][视频理解][骨骼动作识别] 提出 SUGAR 范式，利用 GPT 生成的**运动描述**和**视觉描述**作为先验知识，通过对比学习监督骨骼编码器学习更离散的表示，再用 LLM（LLaMA2-7B）的未触及预训练权重作为识别器，配合新设计的 Temporal Query Projection（TQP）模块实现高效的骨骼动作分类和零样本推理。
tags:
  - "AAAI 2026"
  - "视频理解"
  - "骨骼动作识别"
  - "大语言模型"
  - "视觉-运动知识"
  - "对比学习"
  - "零样本识别"
---

# SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition

**会议**: AAAI 2026  
**arXiv**: [2511.10091](https://arxiv.org/abs/2511.10091)  
**代码**: 无  
**领域**: 视频理解 / 动作识别  
**关键词**: 骨骼动作识别, 大语言模型, 视觉-运动知识, 对比学习, 零样本识别

## 一句话总结

提出 SUGAR 范式，利用 GPT 生成的**运动描述**和**视觉描述**作为先验知识，通过对比学习监督骨骼编码器学习更离散的表示，再用 LLM（LLaMA2-7B）的未触及预训练权重作为识别器，配合新设计的 Temporal Query Projection（TQP）模块实现高效的骨骼动作分类和零样本推理。

## 研究背景与动机

**领域现状**：骨骼动作识别旨在建模时空图结构以分类人体动作，因数据轻量而适合人机交互和智能监控等应用。ST-GCN、2s-AGCN、CTR-GCN 等方法在图卷积网络上不断改进，但在区分细粒度相似动作（如喝水 vs 吃东西）时仍然困难。近期 LLM（如 Vicuna、LLaMA）展现了跨模态的强大能力，已有工作开始将 LLM 用作动作识别器。

**现有痛点**：

**LLM 如何理解骨骼数据？** LLM 主要在人类语言数据上预训练，骨骼数据（坐标序列）与文本之间存在巨大鸿沟。之前 ActionLLM 使用 VQ-VAE 学习离散 token，但难以保证骨骼 token 与文本 token 的一致性。

**LLM 如何区分相似动作？** 日常活动中充满相似动作（如喝瓶装水 vs 喝罐装水），当缺少外观信息时，仅靠骨骼运动轨迹难以区分。现有方法缺乏高层语义信息来增强骨骼表示的区分度。

**传统线性分类器的局限**：线性层的 logistic 分布无法直接适配来自其他数据集的动作类别，限制了零样本泛化能力。

**核心矛盾**：骨骼数据具有运动信息但缺少视觉上下文，LLM 具有丰富的人类活动知识但无法直接理解骨骼输入。如何**让 LLM 学到更离散（可区分）的骨骼表示**是关键。

**切入角度**：不直接让 LLM 从骨骼学习，而是先用丰富的语言知识（运动描述+视觉描述）**监督骨骼编码器**学习与文本对齐的离散表示，再将这些表示输入 LLM 进行分类。

## 方法详解

### 整体框架

SUGAR 的训练流程分为三步：
- **Step 1: 文本构建** — 用 GPT 和 VLM 生成运动描述和视觉描述
- **Step 2: 骨骼表示学习** — 用对比学习将骨骼表示与文本描述对齐
- **Step 3: 动作识别** — 用 TQP 投射骨骼表示到 LLM 空间，用 LoRA 微调 LLM 进行分类

推理时只需输入骨骼序列即可完成动作识别。

### 关键设计

#### 1. **文本构建（Step 1）**

**运动知识生成**：定义预定义动作列表，将每个动作分解为 6 个身体部位运动（头、手、臂、臀、腿、脚），使用 GPT-3.5-turbo 自动生成细粒度运动描述 $\mathcal{T}_m$。例如"喝水"会被描述为"头部微微后仰，手握杯子，臂抬起至嘴边..."

**视觉知识生成**：使用 GPT-4V 从视频帧中提取与动作相关的视觉信息 $\mathcal{T}_v$。为避免 VLM 生成无关内容（外貌、环境等），设定三条规则约束生成内容。通过 CLIP 视觉编码器计算帧间相似度，选取最不相似的帧集合以过滤重复帧。

**设计动机**：运动描述可以区分不同动作的身体部位运动轨迹差异（如手臂角度不同），但当运动轨迹完全相似时（如吃东西和喝东西手臂运动类似），还需要视觉信息（如手中物体是杯子还是食物）来补充区分。

#### 2. **骨骼-文本对比学习（Step 2）**

**骨骼编码器**：使用 CTR-GCN 作为 backbone，堆叠多个 GCN blocks 进行空间聚合：
$$\text{H}^{l+1} = \sigma(\text{D}^{-1/2}\text{A}\text{D}^{-1/2}\text{H}^l\text{W}^l)$$

同时使用 CTR-GCN 的多尺度时间建模模块，但**丢弃时间维度的池化操作**以保留完整的时间信息。

**文本编码器**：使用 CLIP 文本编码器分别编码运动描述 $m = E_t(\mathcal{T}_m)$ 和视觉描述 $v_i = E_t(\mathcal{T}_{v_i})$，然后随机组合形成 $\mathbf{t} = \{m, v_i | i \in I_v\}$。

**多实例对比学习损失**：不同于一对一对比学习，骨骼表示可以与**多个文本描述正匹配**，使用 MIL-NCE 损失：
$$\mathcal{L}_{MIL} = -\frac{1}{|B|} \sum_i \log \frac{\sum_j \sum_n \exp(\mathbf{s}_i^\top \mathbf{t}_{j,n}/\tau)}{\sum_k \sum_n \exp(\mathbf{s}_i^\top \mathbf{t}_{k,n}/\tau)}$$

#### 3. **Temporal Query Projection (TQP) 模块（Step 3）**

**问题**：骨骼编码器输出的序列很长（如 1000 帧），直接输入 LLM 计算成本过高且 LLM 不擅长建模非语言长序列。但简单池化/压缩会破坏连续时间拓扑。

**设计**：将骨骼表示 $\mathbf{s} \in \mathbb{R}^{L_s \times d}$ 按超参数 $k$ 分段，定义可学习查询向量 $\mathbf{q} \in \mathbb{R}^{k \times d}$，使用多个共享权重的 Q-Former **串联查询**骨骼表示：

$$\hat{\mathbf{s}}_t = f_Q^t(\mathbf{s}_{t-1}, \mathbf{s}_t)$$

**关键创新**：不同于单个 Q-Former 的独立查询，TQP 将前一段骨骼表示的查询结果作为下一段的查询输入，实现**连续的时间信息建模**。最终将序列长度压缩到 $L=128$ 个 token。

**为什么 128？** 消融实验表明 LLM 不擅长建模非语言的超长序列（1000 token 效果差），但压缩到 1 个 token 又导致信号过于同质化无法区分不同动作，128 是最优平衡点。

#### 4. **LLM 微调（Step 3）**

冻结骨骼编码器参数，使用 **LLaMA2-7B** 配合 **LoRA**（r=64, alpha=16）进行 1 epoch 微调。定义固定指令模板，输入动作 token，预测动作类别。损失：
$$\mathcal{L}_{LoRA} = \text{CrossEntropy}(f_{LLM}(\hat{\mathbf{s}}), y)$$

其中 $y$ 包含 ground truth 类别和动作描述。LLM 不仅输出类别还能输出动作描述。

### 损失函数 / 训练策略

两阶段训练：
- **阶段一**：用 $\mathcal{L}_{MIL}$ 训练骨骼编码器，SGD 优化器，lr=0.01，200 epochs，batch=200
- **阶段二**：冻结编码器，用 $\mathcal{L}_{LoRA}$ 微调 LLM，lr=2e-5，1 epoch，batch=128
- 硬件：2 块 NVIDIA A6000

## 实验关键数据

### 主实验

| 方法 | 识别器 | 输入 | Toyota SH X-sub | SH X-view1 | SH X-view2 | NTU60 X-sub | NTU60 X-view | NTU120 X-sub | NTU120 X-view |
|------|--------|------|-----------------|------------|------------|-------------|-------------|-------------|-------------|
| 2s-AGCN | FC | Joint+Bone | 55.7 | 21.6 | 53.3 | 84.2 | 93.0 | 78.2 | 82.9 |
| ST-GCN | FC | Joint+Bone | 62.9 | 40.6 | 51.4 | 81.5 | 88.3 | 82.1 | 84.5 |
| UNIK | FC | Joint+Bone | 62.1 | 33.4 | 63.6 | 86.8 | 94.4 | 80.8 | 86.5 |
| LLM-AR | LLM | Joint | 67.0 | 36.1 | 66.6 | 95.0 | 98.4 | 88.7 | 91.5 |
| **SUGAR** | LLM | Joint | **70.2** | **50.9** | **67.1** | **95.2** | 97.8 | **90.1** | 89.7 |

**零样本实验**：

| 方法 | Protocol 1 (NTU60→NTU120) | | Protocol 2 (NTU60→PKU) | |
|------|---------------------------|---|-------------------------|---|
| | Top-1 | Top-5 | Top-1 | Top-5 |
| ST-GCN (FC) | 30.1 | 45.2 | 36.9 | 55.2 |
| LLM-AR (LLM) | 59.7 | 84.1 | 49.4 | 74.2 |
| **SUGAR (LLM)** | **65.3** | **89.8** | **53.4** | **77.6** |

### 消融实验

**视觉-运动知识的影响**（Toyota Smarthome）：

| 配置 | 准确率 |
|------|--------|
| 无视觉、无运动知识 | 69.2% |
| 仅视觉知识 | 69.4% |
| 仅运动知识 | 72.1% |
| **视觉+运动知识** | **73.4%** |

**桥接模块对比**（Toyota Smarthome）：

| 方法 | 准确率 |
|------|--------|
| Cross-Attention | 52.1% |
| One Q-Former | 70.7% |
| One linear layer | 70.4% |
| **Temporal Query Projection** | **73.4%** |

**Action Token 长度影响**（NTU60）：
- 完整长度（1000）：效果中等
- 128：最优
- 1：最差（过度压缩导致信号同质化）
- 结论：LLM 不擅长超长非语言序列，但过度压缩丢失区分性

### 关键发现

1. **视觉-运动知识带来 4.2% 提升**：从 69.2% 到 73.4%，其中运动知识贡献更大（+2.9%），视觉知识单独贡献较小（+0.2%），但两者结合效果最佳。
2. **Toyota Smarthome 上优势最大**：X-view1 从 36.1% 提升到 50.9%（+14.8%），因为该数据集包含大量需要视觉上下文辅助的复合日常活动。
3. **零样本推理大幅超越线性方法**：线性分类器的 logistic 分布无法适配新数据集类别，而 LLM 基于自然语言的动作列表推理天然支持开放集识别。Protocol 1 Top-1 从 30.1% 提升到 65.3%。
4. **TQP 比单个 Q-Former 提升 2.7%**：串联查询设计保留了时间连续性。
5. **t-SNE 可视化**：训练后相似动作（如"瓶装饮水"vs"罐装饮水"）的骨骼表示变得明显分离，证实视觉-运动知识引导产生了更离散的表示。

## 亮点与洞察

1. **"不改模型改数据"的范式**：SUGAR 不设计更强的骨骼编码器，而是用丰富的语言知识**改善骨骼表示的质量**，这是一种轻量且有效的方法论。
2. **LLM 作为通用识别器的潜力**：仅用 Joint 输入（无 Bone、无多流融合）就超越了使用 Joint+Bone 的传统方法，说明 LLM 的预训练知识可以弥补输入信号的不足。
3. **运动描述的身体部位分解**：将动作分解为 6 个身体部位的运动（参考 HAKE 框架），提供了结构化的先验知识。
4. **TQP 的串联查询设计**：既压缩了序列长度又保持了时间连续性，比独立 Q-Former 和线性投射更有效。
5. **零样本能力**：展示了 LLM-based 方法相比线性分类器在开放集场景下的天然优势。

## 局限与展望

1. **仅使用 Joint 输入**：未利用 Bone 和 Motion 流，多流融合可能带来更大提升。
2. **GPT 生成的描述质量未验证**：依赖 GPT-3.5/4V 的生成质量，可能包含噪声或错误描述。
3. **LLM 规模固定在 7B**：未探索更大/更小模型的影响。
4. **需要预定义动作列表**：虽然支持零样本，但仍需预定义可能的类别名称，无法真正的开放词汇识别。
5. **NTU 数据集上提升幅度有限**：NTU60/120 是单场景录制，骨骼质量较高且动作差异较大，SUGAR 的优势在复杂场景（如 Toyota SH）中更突出。

## 相关工作与启发

- CLIP-based 对比学习从图文扩展到骨骼文本是自然的演进，但本文的关键贡献在于**文本内容的选择**——不用简单的类别名称（如 "A action of {}"），而用细粒度的运动描述和视觉描述。
- ActionLLM 是 LLM 用于骨骼识别的先驱，SUGAR 在其基础上通过引入外部知识监督表示学习实现了 3-15% 的提升。
- TQP 模块的串联 Q-Former 设计可推广到其他长序列 token 压缩场景（如视频 token、音频 token）。

## 评分

- 新颖性: ⭐⭐⭐⭐ （视觉-运动知识监督骨骼学习+TQP 设计，范式创新）
- 实验充分度: ⭐⭐⭐⭐ （4个数据集、零样本实验、多消融、可视化分析）
- 写作质量: ⭐⭐⭐⭐ （流程清晰，逻辑完整，图文配合好）
- 价值: ⭐⭐⭐⭐ （为 LLM-based 骨骼识别提供了新范式，零样本能力实用性强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](../../CVPR2025/video_understanding/heterogeneous_skeleton-based_action_representation_learning.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](../../CVPR2025/video_understanding/h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[ICML 2026\] SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition](../../ICML2026/video_understanding/skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o.md)

</div>

<!-- RELATED:END -->
