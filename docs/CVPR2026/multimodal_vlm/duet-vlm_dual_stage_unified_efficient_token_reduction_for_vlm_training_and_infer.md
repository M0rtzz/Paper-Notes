---
title: >-
  [论文解读] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference
description: >-
  [CVPR2026][多模态][VLM token压缩] 提出 DUET-VLM 双阶段视觉 token 压缩框架：第一阶段在视觉编码器内通过 V2V self-attention 选取 dominant tokens 并将剩余 tokens 通过注意力引导局部聚类合并为 contextual tokens；第二阶段在 LLM 内通过 T2V cross-attention 层级裁剪视觉 tokens。在 LLaVA-1.5-7B 上实现 67% token 压缩保持 99%+ 精度、89% 压缩保持 97%+ 精度，训练时间减少 31%。
tags:
  - CVPR2026
  - 多模态
  - VLM token压缩
  - 视觉token冗余
  - 双阶段token裁剪
  - 注意力引导聚合
  - 剪枝
---

# DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference

**会议**: CVPR2026  
**arXiv**: [2602.18846](https://arxiv.org/abs/2602.18846)  
**代码**: https://github.com/AMD-AGI/DUET-VLM  
**领域**: multimodal_vlm  
**关键词**: VLM token压缩, 视觉token冗余, 双阶段token裁剪, 注意力引导聚合, 层级pruning

## 一句话总结
提出 DUET-VLM 双阶段视觉 token 压缩框架：第一阶段在视觉编码器内通过 V2V self-attention 选取 dominant tokens 并将剩余 tokens 通过注意力引导局部聚类合并为 contextual tokens；第二阶段在 LLM 内通过 T2V cross-attention 层级裁剪视觉 tokens。在 LLaVA-1.5-7B 上实现 67% token 压缩保持 99%+ 精度、89% 压缩保持 97%+ 精度，训练时间减少 31%。

## 背景与动机

1. **领域现状**：VLM（如 LLaVA、InternVL）依赖大量视觉 tokens 将图像信息传递给 LLM，但视觉 tokens 存在严重冗余——大量 tokens 对应背景或重复纹理区域，并非语义核心。
2. **现有痛点**：现有 token 压缩方法是**单侧**的——要么只在视觉编码器侧压缩（VisionZip、HiRED），要么只在 LLM 侧压缩（FastV、PyramidDrop），无法同时利用两侧信息进行最优压缩。
3. **核心矛盾**：Vision-only 方法缺乏文本引导信号，不知道哪些视觉 tokens 对当前问题真正重要；Language-only 方法只能在 LLM 内部做后处理，已经浪费了前几层的计算资源。
4. **本文目标**：如何设计一个统一的双阶段框架，在视觉编码器内和 LLM 内分别进行互补的 token 压缩，同时适用于训练和推理？
5. **切入角度**：第一阶段利用视觉 tokens 之间的 self-attention（V2V）做粗粒度压缩；第二阶段利用文本对视觉的 cross-attention（T2V）做细粒度裁剪。
6. **核心 idea**：V2V 阶段通过 attention-guided local cluster aggregation（固定宽度 $w$ 的局部聚类而非全局平均）保留空间上下文信息；T2V 阶段通过层级 pruning 逐步 drop 低相关性视觉 tokens。

## 方法详解

### 整体框架
DUET-VLM 包含两个阶段：(1) V2V（Vision-to-Vision）阶段在视觉编码器（如 CLIP ViT）最后一层内执行；(2) T2V（Text-to-Vision）阶段在 LLM decoder 的多个中间层执行。两阶段串联，可同时用于训练和推理。

### 关键设计

1. **V2V 阶段——视觉编码器内 token 压缩**:

    - 功能：在视觉编码器最后一层，利用 V2V self-attention 将 $N$ 个视觉 tokens 压缩为 $k_1 + k_2$ 个
    - 核心思路：(a) 计算所有视觉 tokens 的 self-attention score（对列求和得到每个 token 的"被关注度"），选 top-$k_1$ 个作为 **dominant tokens**——这些是全局语义最重要的 tokens；(b) 剩余 $N - k_1$ 个 tokens 通过 **attention-guided local cluster aggregation** 合并为 $k_2$ 个 **contextual tokens**——以每个 contextual token 为中心，在固定窗口宽度 $w$ 内选择 attention score 最高的邻居做加权平均
    - 设计动机：全局平均池化（如 VisionZip 的 contextual tokens）会稀释信息，因为语义差异大的 tokens 被混在一起。局部聚类（固定宽度 $w$）保证合并的 tokens 在空间上相近、语义相似，避免信息丢失

2. **T2V 阶段——LLM 内层级视觉 token 裁剪**:

    - 功能：在 LLM 的多个中间层逐步裁剪不重要的视觉 tokens
    - 核心思路：(a) 首先选择 **salient text tokens** 集合 $S$——包含 last token（作为 attention sink）和 attention score 最高的若干文本 tokens；(b) 在每个 pruning stage，计算 $S$ 中文本 tokens 对视觉 tokens 的 T2V cross-attention scores，按分数排序丢弃最低 $\lambda$ 比例的视觉 tokens；(c) 多 stage 执行（如在 LLM 的第 $l_1, l_2, \ldots$ 层），逐步压缩
    - 设计动机：文本 tokens 知道"当前问题需要什么信息"，T2V attention 天然反映了视觉 tokens 对回答的相关性。层级裁剪比一次性裁剪更安全，因为浅层 attention 不够成熟，需要渐进式决策

3. **训练时双阶段压缩**:

    - 功能：将双阶段压缩同时应用于训练过程，减少训练资源消耗
    - 核心思路：训练时使用与推理相同的压缩策略，通过减少送入 LLM 的 token 数量降低 FLOPs 和显存。Dominant/contextual token 的选择和 T2V pruning 都用 straight-through estimator 保持梯度流
    - 设计动机：大多数现有方法（FastV、PyramidDrop）只压缩推理不压缩训练，训练成本依然很高。DUET-VLM 统一训练和推理的压缩策略，实现训练加速

### 损失函数 / 训练策略
- 标准自回归 language modeling loss，与 LLaVA 一致
- 训练时直接应用双阶段 token 压缩，无需额外蒸馏或辅助 loss
- V2V 阶段参数 $k_1, k_2, w$ 和 T2V 阶段参数 $\lambda$、pruning layers 为超参数

## 实验关键数据

### 主实验——LLaVA-1.5-7B 推理

| 方法 | Token 压缩率 | 保留精度 | 备注 |
|------|-------------|---------|------|
| FastV | 50%↓ | ~98% | LLM-only pruning |
| PyramidDrop | 50%↓ | ~98% | LLM-only 层级 |
| VisionZip | 67%↓ | ~97% | Vision-only |
| HiRED | 67%↓ | ~96% | Vision-only 层级 |
| FitPrune | 67%↓ | ~98% | 训练感知 pruning |
| **DUET-VLM** | **67%↓** | **99%+** | 双阶段 |
| **DUET-VLM** | **89%↓** | **97%+** | 双阶段极端压缩 |

### 训练时双阶段压缩

| 压缩率 | 保留精度 | 训练时间节省 |
|--------|---------|-------------|
| 67%↓ | 99.7% | ~31% |
| 89%↓ | 97.6% | ~31% |

### Video-LLaVA-7B

| 压缩率 | 保留精度 | 备注 |
|--------|---------|------|
| 53.1%↓ | 100%+（超 baseline） | 压缩后反而提升 |
| 93.4%↓ | 97.6% | 极端压缩 |

### 关键发现
- **双阶段 > 单侧**：V2V-only 或 T2V-only 均不如双阶段组合，证实两阶段信息互补
- **局部聚类 > 全局平均**：固定宽度 $w$ 的 local cluster aggregation 显著优于 VisionZip 的全局 contextual token 策略
- **视频场景更受益**：Video-LLaVA 在 53.1% 压缩率下精度反超 baseline，说明 token 冗余在视频中更严重，适度压缩反而去噪
- **训练压缩可行**：训练时应用 67% 压缩，精度仅损失 0.3%，但训练时间减少 31%
- **超越所有现有方法**：在相同压缩率下，DUET-VLM 在所有 benchmark 上均优于 VisionZip、FastV、PyramidDrop、HiRED、FitPrune

## 亮点与洞察
- **"双阶段互补"的設計哲学**：V2V 利用视觉内部信息做粗筛（不依赖文本），T2V 利用文本引导做精筛。两阶段分别解决不同层面的冗余，避免了单侧方法的信息盲区
- **局部聚类的简洁与有效**：用固定宽度 $w$ 做局部聚类，避免了复杂的全局聚类算法（如 k-means），计算开销小但效果显著优于全局平均
- **统一训练与推理**：大多数 token 压缩方法只看推理效率，DUET-VLM 在训练阶段即可生效，这在大规模 VLM 训练中有实际价值
- **视频压缩后反超 baseline**：53.1% 压缩率下精度反升，说明冗余 tokens 不仅浪费资源还引入噪声

## 局限与展望
- 仅在 LLaVA-1.5-7B 和 Video-LLaVA-7B 上验证，缺少更大规模模型（如 13B/34B）的实验
- V2V 阶段的 $k_1, k_2, w$ 为固定超参数，不同图像/任务可能需要自适应调整
- 未与最新的 InternVL2、Qwen-VL 等架构验证兼容性
- T2V 阶段的 salient text token 选择依赖 attention sink 假设，对非标准 prompt 格式的鲁棒性未知
- 缺少对压缩后 attention pattern 变化的深入分析

## 相关工作与启发
- **vs VisionZip**: VisionZip 在视觉编码器侧做 dominant + contextual token 选择，但 contextual tokens 用全局平均导致信息稀释。DUET-VLM 改用 local cluster aggregation，并加入 T2V 阶段
- **vs FastV/PyramidDrop**: 两者均在 LLM 侧做 attention-based pruning，但缺少视觉编码器侧的初步筛选。DUET-VLM 的 V2V 阶段先做粗筛，T2V 阶段压力更小
- **vs FitPrune**: FitPrune 通过训练感知优化 pruning 策略，但仍是单侧方法。DUET-VLM 在训练和推理均可应用双阶段压缩
- **vs HiRED**: HiRED 在视觉编码器内做层级 attention-based 压缩，但不涉及 LLM 侧。DUET-VLM 在两侧均做层级压缩

## 评分
- 新颖性: ⭐⭐⭐⭐ 双阶段 V2V+T2V 框架是首次提出，local cluster aggregation 设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 覆盖图像和视频场景，训练和推理均验证，消融实验完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，图示直观
- 价值: ⭐⭐⭐⭐⭐ VLM token 压缩的实用方案，训练加速 31% 有显著工程价值

<!-- RELATED:START -->

## 相关论文

- [VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)
- [Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)
- [GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)

<!-- RELATED:END -->
