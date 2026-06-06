---
title: >-
  [论文解读] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models
description: >-
  [CVPR 2026][多模态VLM][模型剪枝] CoMP 提出协同多模式剪枝框架，通过协同重要性度量（CIM）消除参数和 token 剪枝指标间的不一致性，通过多模式剪枝策略（MPS）自适应选择每阶段的最优剪枝模式，在高剪枝比例下显著优于单模式和简单联合剪枝方案。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "模型剪枝"
  - "视觉语言模型"
  - "参数剪枝"
  - "Token剪枝"
  - "协同压缩"
---

# CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2604.02956](https://arxiv.org/abs/2604.02956)  
**代码**: [https://github.com/Wuzimeng/CoMP.git](https://github.com/Wuzimeng/CoMP.git)  
**领域**: 多模态VLM  
**关键词**: 模型剪枝, 视觉语言模型, 参数剪枝, Token剪枝, 协同压缩

## 一句话总结

CoMP 提出协同多模式剪枝框架，通过协同重要性度量（CIM）消除参数和 token 剪枝指标间的不一致性，通过多模式剪枝策略（MPS）自适应选择每阶段的最优剪枝模式，在高剪枝比例下显著优于单模式和简单联合剪枝方案。

## 研究背景与动机

VLM 基于 Transformer 架构，计算复杂度为 $O(N^2D + ND^2)$，其中 $N$ 是序列长度、$D$ 是特征维度。参数剪枝减小 $D$，token 剪枝减小 $N$，两者互补。

**两个核心挑战**：(1) **重要性度量不一致**——参数重要性的计算依赖所有 token，但 token 剪枝会移除部分 token，导致参数重要性被不重要的 token 主导。反之，token 重要性依赖所有参数，但参数剪枝会移除部分参数，导致 token 重要性失真。(2) **剪枝模式的固定应用**——渐进剪枝中每阶段固定按相同顺序剪参数和 token，但不同阶段的最优剪枝模式不同。

## 方法详解

### 整体框架

嵌套循环结构：外层循环由 MPS 周期性选择最优剪枝模式，内层循环中 CIM 计算协同的参数和 token 重要性分数，执行所选模式的剪枝。

### 关键设计

1. **协同重要性度量 (CIM)**:

    - 功能：消除参数和 token 重要性计算的相互干扰
    - 核心思路：计算参数重要性时，引入 token 加权的输入范数——按 token 重要性对参数重要性计算加权，降低不重要 token 的干扰。计算 token 重要性时，将参数剪枝掩码传递到注意力权重矩阵，抑制已被标记为不重要的参数对 token 重要性的影响
    - 设计动机：实验显示参数重要性计算中最关键的 token 与 token 重要性排名仅有 <30% 的重叠，说明两种度量严重不一致

2. **多模式剪枝策略 (MPS)**:

    - 功能：在渐进剪枝的每个阶段自适应选择最优剪枝模式
    - 核心思路：将剪枝过程分为多个阶段，每阶段估算不同剪枝模式（视觉参数/语言参数/视觉token/语言token）的"剪枝代价"，选择代价最低的模式执行。同时融合历史代价（稳定性）和随机探索（避免局部最优）
    - 设计动机：不同模式在不同阶段的最优性不同——早期可能参数剪枝更好，后期可能 token 剪枝更好。固定顺序无法适应这种变化

3. **跨模态协同剪枝**:

    - 功能：同时对视觉和语言模态进行自适应剪枝
    - 核心思路：CIM 和 MPS 分别应用于视觉编码器和语言模型，不同模态的剪枝比例由 MPS 自适应分配。这允许视觉和语言部分以不同速率被压缩
    - 设计动机：视觉和语言部分的冗余程度不同，均匀剪枝不是最优的

### 损失函数 / 训练策略

基于重要性得分的结构化剪枝，不需要重训练。剪枝代价基于模型在验证集上的性能变化估算。

## 实验关键数据

### 主实验

| 方法 | NLVR2 (50%剪枝) | NLVR2 (70%剪枝) | VQA | 图文检索 |
|------|----------------|----------------|-----|---------|
| 参数剪枝 only | 中 | 差 | 中 | 中 |
| Token剪枝 only | 中 | 差 | 中 | 中 |
| 简单联合 | 中 | 差 | 中 | 中 |
| **CoMP** | **最优** | **显著优于** | **最优** | **最优** |

在高剪枝比例（70%+）下优势尤为显著。

### 消融实验

| 配置 | 高剪枝比例性能 | 说明 |
|------|--------------|------|
| 无 CIM（独立度量） | 明显下降 | 度量不一致导致错误剪枝 |
| 无 MPS（固定模式） | 下降 | 非最优模式顺序 |
| 无随机探索 | 略下降 | 陷入局部最优 |
| 完整 CoMP | 最优 | 所有组件必要 |

### 关键发现

- CIM 的贡献在高剪枝比例下更加明显——低剪枝比例时度量不一致的影响较小
- MPS 的自适应模式选择避免了人工调参——不同任务/模型的最优策略不同
- 视觉和语言部分的最优剪枝比例确实不同，均匀剪枝是次优的

## 亮点与洞察

- **度量不一致的发现**：参数和 token 重要性度量间的干扰之前被忽视，CIM 的协同设计优雅地解决了这个问题
- **自适应模式选择**：借鉴多臂老虎机的思路（代价估计+探索），在剪枝中实现了自动化的策略选择
- **高剪枝比例优势**：在实际部署最需要的高压缩率场景下优势最大

## 局限与展望

- MPS 的模式选择增加了剪枝过程的计算开销
- 当前仅验证在 BLIP 系列模型上，对 LLaVA 等架构的适用性需进一步测试
- Token 剪枝在推理时的动态性需要专用的推理优化
- 未来可探索与量化的联合压缩

## 相关工作与启发

- **vs UPop/EViT**: 单模式剪枝方法，在高压缩率下性能急剧下降
- **vs 简单联合剪枝**: 不处理度量不一致，效果不如分别单模式剪枝
- **vs DepGraph/PLATON**: 参数剪枝专用方法，缺乏 token 维度的压缩

## 评分

- 新颖性: ⭐⭐⭐⭐ 度量不一致问题的发现和CIM设计有新意
- 实验充分度: ⭐⭐⭐⭐ 多任务多剪枝比例全面测试
- 写作质量: ⭐⭐⭐⭐ 问题分析清楚，图示直观
- 价值: ⭐⭐⭐⭐ 对VLM部署有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[CVPR 2026\] Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models](mostly_text_smart_visuals_asymmetric_text-visual_pruning_for_large_vision-langua.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)
- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)

</div>

<!-- RELATED:END -->
