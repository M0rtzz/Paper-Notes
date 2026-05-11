---
title: >-
  [论文解读] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
description: >-
  [CVPR 2026][图像分割][panoptic scene graph generation] DSFlash 通过合并分割与关系预测 backbone、双向关系预测头、动态 patch 剪枝等策略，将全景场景图生成速度提升至 RTX 3090 上 56 FPS，同时在 PSG 数据集上达到 mR@50=30.9 的 SOTA 性能。
tags:
  - CVPR 2026
  - 图像分割
  - panoptic scene graph generation
  - real-time inference
  - bidirectional relation prediction
  - 剪枝
  - low-latency
---

# DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**会议**: CVPR 2026  
**arXiv**: [2603.10538](https://arxiv.org/abs/2603.10538)  
**代码**: 无（作者声明 acceptance 后公开）  
**领域**: 场景图生成 / 视觉场景理解  
**关键词**: panoptic scene graph generation, real-time inference, bidirectional relation prediction, token pruning, EoMT

## 一句话总结

DSFlash 通过合并分割与关系预测 backbone、门控双向关系预测头和 mask-based 动态 patch 剪枝，在 PSG 数据集上以 18ms 延迟（56 FPS）实现 mR@50=30.9 的 SOTA 全景场景图生成。

## 研究背景与动机

**领域现状**：场景图生成（SGG）将图像结构化为节点（实例）和边（关系），形成 (subject, predicate, object) 三元组表示，已在视觉问答、图像描述、具身推理等下游任务中展现出广泛价值。全景场景图生成（PSGG）进一步使用分割 mask 替代边界框来定位实例，提供更精确的空间定位信息。

**现有痛点**：当前 PSGG 方法几乎完全忽视了推理效率。DSFormer 虽然达到了 SOTA 性能（mR@50=30.7），但单帧推理耗时 458ms，且使用两个独立 backbone（MaskDINO 用于分割、ResNet 用于关系预测），资源浪费严重。HiLo 等一阶段方法虽然声称更高效，但延迟仍达 427ms，且性能不佳。唯一关注速度的 REACT 虽将延迟降至 19ms，但通过 YOLOv8 做 bbox 检测而非全景分割，mR@50 仅 19.0，性能差距巨大。

**核心矛盾**：高质量的全景场景图生成与实时推理速度之间存在严重冲突——现有方法要么质量高但速度极慢，要么速度快但只做简化任务（bbox 检测而非分割，或只预测显著关系而非全面场景图）。

**本文目标** 在不牺牲场景图质量的前提下，让全景场景图生成达到实时级别推理速度，且计算全面场景图（所有实例间的所有关系），而非仅预测部分显著关系。

**切入角度**：从两阶段方法出发，利用现代高效分割 backbone（EoMT）同时提供分割 mask 和特征表示，消除冗余 backbone 前向传播；通过门控机制实现双向关系预测，将关系分类前向次数减半；利用 mask 覆盖先验剪枝无关 patch token。

**核心 idea**：复用冻结的高效分割 backbone 特征 + 门控双向关系头 + 任务先验驱动的 token 剪枝 = 实时全面全景场景图生成。

## 方法详解

### 整体框架

DSFlash 采用两阶段设计。第一阶段使用冻结的 EoMT（Encoder-only Mask Transformer）作为分割 backbone，提取全景分割 mask 和中间特征。具体来说，从 EoMT 的 block 2/5/8/11（S/B 变体）或 block 5/11/17/23（L 变体）提取 patch token，拼接后得到 768×40×40 的特征张量。第二阶段对每对分割 mask (S₀, S₁)，通过 mask embedding 将主体/客体位置编码到特征 patch 中，经 ViT patch embedding 得到 13×13 的 patch token（384 维），再经轻量 Transformer neck 处理，最后由门控双向关系头同时输出两个方向的关系预测。训练时使用 ground truth mask，推理时使用 EoMT 预测的 mask。

### 关键设计

1. **Merged Backbone（统一 backbone）**:

    - 功能：消除分割和关系预测使用独立 backbone 的冗余，将推理延迟降低一个数量级
    - 核心思路：直接从 EoMT 分割模型的中间层抽取多尺度 patch token 作为关系预测的输入特征，不再需要额外的 ResNet backbone。EoMT 全程冻结，训练时仅更新 neck 和 head 参数
    - 设计动机：DSFormer 的两个独立 backbone（MaskDINO + ResNet）导致两次完整前向传播，是延迟的主要瓶颈（445ms 中分割耗时占大头）。EoMT 作为 encoder-only 架构，去掉了 feature adapter、pixel decoder 和 transformer decoder，推理速度比 Mask2Former 快 4 倍，且通过 DINO/EVA-02 大规模自监督预训练保证了特征质量

2. **Gated Bidirectional Relation Prediction（门控双向关系预测）**:

    - 功能：一次前向传播同时预测 S₀→S₁ 和 S₁→S₀ 两个方向的关系，将前向次数减半
    - 核心思路：对编码后的特征 x，通过 sigmoid 门控 MLP 生成门控向量 g = σ(gate_mlp(x))，将 x 分裂为 t→ = g⊙x 和 t← = (1-g)⊙x 两个分支，共享同一个 MLP 关系头分别输出两个方向的预测。训练时对每对 mask 做两次前向（交换 S₀/S₁ 顺序），通过 MSE consistency loss 约束翻转后的中间特征应交换（t→ ≈ t'←, t← ≈ t'→），确保方向等变性
    - 设计动机：作者发现 PSG 数据集中正标注出现在正向的概率是反向的 3 倍，模型会利用这一统计偏差作弊。共享 MLP head + consistency loss 迫使模型平等处理两个方向，消除了数据偏差，同时额外的双向监督信号还带来了 mR@50 的提升（25.0→28.8）

3. **Mask-based Dynamic Patch Pruning（基于 mask 的动态 patch 剪枝）**:

    - 功能：丢弃与主体和客体 mask 均无重叠的 patch token，减少 model neck 的计算量
    - 核心思路：在 mask embedding 阶段需要计算每个 patch 与 subject/object mask 的重叠比例，对于重叠比例均为零的 patch，其 mask embedding 为纯背景 token，不含有用的定位信息，可以直接丢弃。由于最终预测只依赖 classification token，模型天然支持可变长度输入
    - 设计动机：重叠比例本就需要计算，因此剪枝的判断几乎零开销。在低端 GPU（GTX 1080）上效果尤为显著，延迟从 230ms 降至 205ms

### 损失函数 / 训练策略

- **关系分类损失**：对两个方向的预测分别计算 Binary Cross Entropy loss：BCE(z→, y→) 和 BCE(z←, y←)
- **Feature Consistency Loss**：MSE 损失约束翻转输入后的中间特征应交换，公式为 Consistency = (1/D)Σ[(t→ᵢ - t'←ᵢ)² + (t←ᵢ - t'→ᵢ)²]
- **负采样策略**：每 5 个正样本采 1 个负样本
- **数据增强**：DeiT III 风格——随机水平翻转 + 颜色抖动 + 三选一（灰度/solarization/高斯模糊）
- **优化器**：AdamW，lr=1e-5，cosine schedule + warmup，梯度裁剪 norm≤1，训练 20 epoch
- **训练效率**：backbone 全程冻结，仅训练 neck 和 head，单张 GTX 1080 不到 24 小时即可完成训练

## 实验关键数据

### 主实验

在 PSG 数据集上使用 SGDet 协议评估，batch size=1，RTX 3090 GPU：

| 方法 | mR@50 ↑ | 延迟 (ms) ↓ | 参数量 |
|------|---------|------------|--------|
| MotifNet-R50 | 9.56 | 100 | 109M |
| VCTree-R50 | 10.14 | 116 | 105M |
| MotifNet-MD | 16.32 | 504 | 332M |
| VCTree-MD | 17.58 | 520 | 327M |
| HiLo-R50 | 16.34 | 277 | 59M |
| HiLo-L | 19.08 | 427 | 230M |
| REACT | 19.00 | 19 | 43M |
| DSFormer | 30.70 | 458 | 330M |
| **DSFlash-S*** | 25.05 | **18** | **40M** |
| **DSFlash-B*** | 28.50 | 23 | 116M |
| **DSFlash-L** | **30.90** | 50 | 340M |

### 消融实验

逐步叠加优化的效果（RTX 3090，batch size=1）：

| 优化步骤 | mR@50 ↑ | 延迟 (ms) ↓ | RPS ↑ |
|----------|---------|------------|-------|
| Baseline (DSFormer) | 30.7 | 445 | 435 |
| + 统一 Backbone | 25.0 | 41 (-91%) | 5,745 |
| + 高效 Mask Embedding | 25.0 | 37 (-10%) | 7,132 |
| + 门控双向预测 | 28.8 | 29 (-22%) | 11,491 |
| + 跳过 Mask 上采样 | 28.5 | 23 (-21%) | 12,928 |
| + 切换 EoMT-S | 25.1 | 18 (-22%) | 17,897 |
| + 切换 EoMT-L（替代上行） | 30.9 | 50 (+72%) | 5,996 |

Pruning 与 Token Merging 在不同 GPU 上的影响：

| Prune | ToMe | H100 | RTX 3090 | GTX 1080 | mR@50 |
|-------|------|------|----------|----------|-------|
| ✗ | 0% | 19ms | 29ms | 230ms | 28.80 |
| ✓ | 0% | 20ms | 29ms | 205ms | 26.67 |
| ✓ | 30% | 20ms | 30ms | 173ms | 26.51 |
| ✗ | 50% | 20ms | 29ms | 167ms | 24.87 |
| ✗ | 60% | 21ms | 29ms | 155ms | 21.93 |

### 关键发现

- 统一 backbone 是最大的加速来源，延迟从 445ms 降至 41ms（-91%），但 mR@50 损失 5.7 个点，主要因为 EoMT 分割质量略低于 MaskDINO
- 门控双向预测不仅减少前向次数（RPS 从 7,132 提升至 11,491），还因额外双向监督信号将 mR@50 从 25.0 提升至 28.8
- mR@50 与分割模型的 Panoptic Quality 相关系数高达 0.99，说明分割质量是场景图性能的决定性因素
- Pruning 和 Token Merging 在高端 GPU 上几乎无延迟收益（已饱和），但在 GTX 1080 上效果显著，两者叠加可将延迟从 230ms 降至 173ms
- EoMT-B + 低分辨率 mask 的组合优于 EoMT-S + 高分辨率 mask（更快且性能更好），说明 backbone 能力比 mask 分辨率更重要

## 亮点与洞察

- **首个实时全面全景场景图生成系统**：DSFlash 不仅速度快（56 FPS），还计算所有实例间的所有关系（comprehensive scene graph），而非仅预测部分显著关系。这使得它在边缘设备部署和实时应用中具有独特优势，填补了 PSGG 领域在实时推理上的空白。
- **门控双向预测的巧妙设计**：通过 sigmoid 门控将特征分裂为两个方向分支，共享 MLP head 预测，一次前向得到双向关系。更巧妙的是 consistency loss 不仅解决了数据集中正反向标注不平衡的偏差问题，还作为额外监督信号提升了性能（25.0→28.8 mR@50），实现了"减计算量同时提性能"的双赢。
- **零开销剪枝的任务先验利用**：mask-based patch pruning 利用了 mask embedding 本身就需要计算重叠比例这一事实，判断是否剪枝几乎不增加任何计算开销，这种将任务特有先验转化为加速手段的思路具有很好的通用性。

## 局限与展望

- **Backbone 冻结限制上限**：EoMT 全程冻结意味着关系预测无法反向影响特征提取，可能限制了性能天花板。端到端微调或部分解冻后层可能进一步提升效果
- **数据集规模偏小**：PSG 数据集仅 49k 图像、56 个谓词类别，在更大规模、更多样化的场景下表现未知
- **主客体混淆问题**：作者提到主客体混淆是常见失败模式，门控机制虽缓解但未彻底解决，可考虑引入实例级对比学习来增强主客体区分能力

## 相关工作与启发

- **vs DSFormer**：继承其 mask embedding 和严格解耦的两阶段设计思想，但通过 backbone 合并和双向预测将延迟从 458ms 降至 50ms（9 倍加速），mR@50 略有提升（30.7→30.9）
- **vs REACT**：此前唯一关注速度的 SGG 方法，使用 YOLOv8 + bbox 检测，速度相当（19ms vs 18ms），但 DSFlash 在 PSGG 设定下性能高出 6-12 个 mR@50 点
- **vs HiLo**：一阶段方法，号称更高效但实际延迟 427ms，性能仅 19.08 mR@50，印证了近期研究对"一阶段更优"说法的质疑
- EoMT 的 encoder-only 设计与冻结复用模式可推广到其他两阶段视觉任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 门控双向预测和 mask-based 零开销剪枝设计巧妙，首次实现实时全面 PSGG
- 实验充分度: ⭐⭐⭐⭐⭐ 三种 backbone 变体、三种 GPU、逐步消融、pruning/merging 交叉实验，分析非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，对评估协议的严谨讨论（SingleMPO）值得肯定，公式推导完整
- 价值: ⭐⭐⭐⭐ 填补了实时 PSGG 的空白，40M 参数/18ms 延迟的配置对边缘部署价值很大
---
title: >-
  [论文解读] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
description: >-
  [CVPR 2026][图像分割][panoptic scene graph generation] DSFlash 通过合并分割与关系预测 backbone、双向关系预测头、动态 patch 剪枝等策略，将全景场景图生成速度提升至 RTX 3090 上 56 FPS，同时在 PSG 数据集上达到 mR@50=30.9 的 SOTA 性能。
tags:
  - CVPR 2026
  - 图像分割
  - panoptic scene graph generation
  - real-time inference
  - bidirectional relation prediction
  - 剪枝
  - low-latency
---

# DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**会议**: CVPR 2026  
**arXiv**: [2603.10538](https://arxiv.org/abs/2603.10538)  
**代码**: 待确认（作者声明 acceptance 后公开）  
**领域**: 场景图生成 / 视觉场景理解  
**关键词**: panoptic scene graph generation, real-time inference, bidirectional relation prediction, token pruning, low-latency  

## 一句话总结
DSFlash 通过合并分割与关系预测 backbone、双向关系预测头、动态 patch 剪枝等策略，将全景场景图生成速度提升至 RTX 3090 上 56 FPS，同时在 PSG 数据集上达到 mR@50=30.9 的 SOTA 性能。

## 背景与动机
场景图（Scene Graph）将图像结构化为节点（实例）和边（关系），在 VQA、推理、图像描述等任务中有广泛应用。现有 PSGG 方法几乎不关注延迟，一次推理往往数百毫秒，难以部署到边缘设备或实时系统。DSFormer 虽达到 SOTA 性能但推理耗时 458 ms，且使用两个独立 backbone（MaskDINO + ResNet），资源浪费严重。本文的核心洞察是：两阶段方法可以通过共享 backbone 特征、减少前向次数、剪枝无关 token 等手段实现极低延迟，同时不损失甚至提升场景图质量。

## 核心问题
如何在不牺牲场景图质量的前提下，让全景场景图生成达到实时级别的推理速度？

## 方法详解

### 整体框架
DSFlash 采用两阶段设计：第一阶段用冻结的 EoMT（Encoder-only Mask Transformer）分割模型提取分割 mask 与特征；第二阶段复用 EoMT 的中间特征（从 block 2/5/8/11 抽取 patch token 并拼接为 768×40×40 特征张量），通过 mask embedding 编码主体/客体位置，经轻量 Transformer neck 后由关系预测头输出关系类别。

### 关键设计

1. **Merged Backbone**：不再使用独立的分割与关系预测 backbone，而是直接抽取 EoMT 内部特征，省去了一次完整 backbone 前向推理。EoMT 全程冻结，训练时仅训 neck 和 head，单张 GTX 1080 不到 24 小时即可完成训练。

2. **双向关系预测（Gated Bidirectional Prediction）**：对于一对 mask (S₀, S₁)，原 DSFormer 需要两次前向分别预测 S₀→S₁ 和 S₁→S₀ 方向的关系。DSFlash 设计了一个门控分裂机制——将编码后的特征 x 通过 sigmoid 门控分成 t→ 和 t← 两个分支，共享同一个 MLP 关系头分别预测两个方向。训练时通过翻转 mask 顺序计算 consistency loss（MSE），确保模型对输入顺序等变。推理时只需一次前向即可得到双向预测。

3. **Mask-based Dynamic Patch Pruning**：在 mask embedding 阶段，与主体和客体 mask 均无重叠的 patch 不含有用的定位信息，直接丢弃后送入 neck。因为重叠率本就需要计算，剪枝几乎零开销。

4. **Raw-resolution Segmentation Masks**：不再将 EoMT 输出的 160×160 mask logits 上采样到原图分辨率再下采样，而是直接在低分辨率上计算 patch 重叠比例，省去了昂贵的双线性插值。

5. **Token Merging (ToMe-SD)**：在 backbone attention 层前合并相似 token，attention 后再 unmerge，降低注意力计算量，在老旧 GPU 上效果尤其明显（GTX 1080 延迟从 230ms 降至 173ms）。

### 损失函数 / 训练策略
- 关系分类：Binary Cross Entropy
- 双向一致性：MSE consistency loss（Eq. 7），约束翻转输入后中间特征应交换
- DeiT III 风格数据增强（随机翻转、颜色抖动、灰度/模糊/solarization 三选一）
- AdamW，lr=1e-5，cosine schedule + warmup，梯度裁剪 norm≤1，训练 20 epoch
- 每 5 个正样本采 1 个负样本

## 实验关键数据

| 方法 | mR@50 | 延迟 (ms) | 参数量 |
|------|-------|-----------|--------|
| DSFormer | 30.70 | 458 | 330M |
| REACT | 19.00 | 19 | 43M |
| HiLo-L | 19.08 | 427 | 230M |
| **DSFlash-L** | **30.90** | 50 | 340M |
| DSFlash-B* | 28.50 | 23 | 116M |
| DSFlash-S* | 25.05 | **18** | **40M** |

- DSFlash-L 在 mR@50 上超越 DSFormer（30.9 vs 30.7），延迟仅为其 1/9
- DSFlash-S* 仅 40M 参数、18ms 延迟（56 FPS），性能仍优于 REACT 和 HiLo

### 消融实验要点
- 统一 backbone 将延迟从 458ms 降至 41ms（-91%），但 mR@50 从 30.7 降至 25.0
- 高效 mask embedding：延迟 37ms（-10%），mR@50 不变
- 门控双向预测：延迟 29ms（-22%），mR@50 从 25.0 提至 28.8（额外监督信号带来性能提升）
- 跳过 mask 上采样：延迟 23ms（-21%），mR@50=28.5（轻微下降）
- mR@50 与分割模型的 Panoptic Quality 相关系数高达 0.99

## 亮点
- 实现了首个真正实时的全景场景图生成系统，GTX 1080 上也能以 ~6 FPS 运行
- 双向关系预测设计精巧，通过一次前向同时输出两个方向，还借助 consistency loss 提升质量
- 整体设计简洁实用：冻结 backbone + 轻量 neck + 共享 head，训练成本极低
- 对评估协议的严谨态度值得肯定：严格遵循 SingleMPO 避免多 mask 膨胀 R@k

## 局限与展望
- Backbone 冻结意味着关系预测无法反向影响特征提取，可能限制上限
- PSG 数据集偏小（49k 图像），在更大数据集上的表现未知
- 低分辨率 mask 对小目标的分割精度可能不足
- 双向预测共享 MLP head，可能在谓词方向性强的关系上有信息混淆
- 作者提到主客体混淆是常见失败模式，可考虑对比学习解决

## 与相关工作的对比
- vs DSFormer：继承其 mask embedding 和 strictly decoupled 思想，但通过 backbone 合并和双向预测将延迟降低 9×
- vs REACT：REACT 用 YOLOv8 做 bbox 检测而非全景分割，DSFlash 在 PSGG 设定下性能高出 12 个 mR@50 点
- vs HiLo：一阶段方法，性能（19.08 mR@50）远逊于 DSFlash，延迟也更高

## 启发与关联
- 冻结 backbone + 复用中间特征的思路可以推广到其他两阶段视觉任务
- 双向预测 + consistency loss 的设计思路可借鉴到检测中的方向关系建模
- 动态 patch 剪枝利用任务先验（mask 覆盖）实现零开销加速，适用于类似的 mask-conditioned 架构

## 评分
- 新颖性: ⭐⭐⭐⭐ 双向预测和 mask-based 剪枝在 PSGG 中是新的，系统级优化很到位
- 实验充分度: ⭐⭐⭐⭐ 多 GPU 延迟评估、详尽消融、公平评估协议
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，对评估问题的讨论很有价值
- 价值: ⭐⭐⭐⭐ 将 PSGG 带入实时领域，实用性强，对资源受限场景特别有意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)
- [\[CVPR 2026\] GenMask: Adapting DiT for Segmentation via Direct Mask Generation](genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](unrealpose_leveraging_game_engine_kinematics_for_large-scale_synthetic_human_pos.md)

</div>

<!-- RELATED:END -->
