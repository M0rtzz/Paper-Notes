---
title: >-
  [论文解读] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker
description: >-
  [CVPR 2026][多模态][多模态跟踪] 提出 SEATrack 多模态跟踪器，通过 AMG-LoRA 实现跨模态注意力图的动态对齐，以及 HMoE 实现高效全局关系建模的跨模态融合，在 RGB-T/D/E 跟踪中以极少参数实现 SOTA 的性能-效率平衡。
tags:
  - CVPR 2026
  - 多模态
  - 多模态跟踪
  - 参数高效微调
  - 多模态VLM
  - 混合专家
  - LoRA
---

# SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker

**会议**: CVPR 2026  
**arXiv**: [2604.12502](https://arxiv.org/abs/2604.12502)  
**代码**: 有  
**领域**: 目标跟踪 / 多模态  
**关键词**: 多模态跟踪, 参数高效微调, 注意力对齐, 混合专家, LoRA

## 一句话总结

提出 SEATrack 多模态跟踪器，通过 AMG-LoRA 实现跨模态注意力图的动态对齐，以及 HMoE 实现高效全局关系建模的跨模态融合，在 RGB-T/D/E 跟踪中以极少参数实现 SOTA 的性能-效率平衡。

## 研究背景与动机

**领域现状**：多模态跟踪通过融合 RGB 与热红外/深度/事件等互补数据实现全天候鲁棒跟踪，PEFT 范式逐渐取代全量微调以避免灾难性遗忘。

**现有痛点**：PEFT 方法的可调参数量从早期方法到最新 SOTA 膨胀了 16 倍，从根本上违背了 PEFT 的效率初衷。同时，双流架构中的域差距导致不同模态产生冲突的注意力图，阻碍联合表示学习。

**核心矛盾**：性能-效率困境——更多参数换来更好性能，但侵蚀了 PEFT 的核心价值。

**本文目标**：(1) 通过跨模态注意力对齐打破性能-效率权衡；(2) 设计高效的全局关系建模替代注意力融合。

**切入角度**：多模态输入在时空上对齐，模态内目标匹配的注意力图原则上应一致——利用这种一致性进行跨模态互导。

**核心 idea**：AMG-LoRA 用一个模态的匹配信息引导另一个模态的匹配过程，实现双向动态对齐。

## 方法详解

### 整体框架

双流 ViT 架构，冻结预训练 RGB 跟踪器的主干。每 2 层嵌入 AMG-LoRA（注意力对齐）和 HMoE（跨模态融合）。两个模态的候选特征通过元素加法聚合后送入预测头进行目标定位。

### 关键设计

1. **AMG-LoRA (自适应互导低秩自适应)**:

    - 功能：同时实现域适应和跨模态注意力图动态对齐
    - 核心思路：(i) LoRA 适配注意力层的 K/V 投影矩阵实现域适应；(ii) 受 Classifier-Free Guidance 启发，将跨模态对齐重构为多分支权衡问题。对齐公式：$\textbf{attn}_{rgb} = \tilde{\textbf{attn}}_{rgb} + w_X(\tilde{\textbf{attn}}_X - \tilde{\textbf{attn}}_{rgb})$，其中 $w_X$ 是可学习缩放因子
    - 设计动机：目标在不同模态的显著性随场景变化，需要动态而非静态的对齐来避免不可靠模态的负迁移。仅 0.14M 参数即可带来 18.3%/7.2%/6.1% 的 PR 提升

2. **HMoE (层级混合专家)**:

    - 功能：高效的全局关系建模，替代注意力的二次复杂度
    - 核心思路：与现有 MoE 仅在专家级别做集成不同，HMoE 实现从子token到token级别的细粒度交互。使用低秩线性层作为专家函数，通过可学习门控矩阵实现层级软路由
    - 设计动机：注意力融合表达力强但二次复杂度高，局部融合效率高但缺乏全局感受野。HMoE 比注意力对应方案快约 35% 同时保持可比性能

3. **共享 LoRA 的双流设计**:

    - 功能：在双流间建立联合表示学习
    - 核心思路：RGB 和 X 模态共享同一 LoRA 旁路，促进跨模态特征对齐。推理时 LoRA 矩阵可合并到原始权重中，不增加延迟
    - 设计动机：共享参数减少参数量同时促进域适应的跨模态一致性

### 损失函数 / 训练策略

标准跟踪损失（分类+回归）。AMG 的缩放因子初始化为 1，训练中自动适应场景。

## 实验关键数据

### 主实验

| 方法 | 可调参数 | LasHeR PR↑ | DepthTrack PR↑ | VisEvent PR↑ |
|------|---------|-----------|---------------|-------------|
| ProTrack | 0.3M | 52.1 | 58.3 | 65.2 |
| Un-Track | 4.8M | 65.4 | 63.8 | 69.1 |
| SDSTrack | 2.1M | 68.2 | 65.5 | 71.3 |
| **SEATrack** | **0.8M** | **70.4** | **65.5** | **71.3** |

### 消融实验

| 配置 | LasHeR PR | 参数量 | 说明 |
|------|----------|--------|------|
| 基线 (冻结ViT) | 52.1 | 0M | 无适配 |
| + LoRA | 60.8 | 0.12M | 仅域适应 |
| + AMG-LoRA | 70.4 | 0.14M | 域适应+对齐 |
| + HMoE | 70.4 | 0.8M | 完整模型 |
| 用注意力替代 HMoE | 70.2 | 1.6M | 速度慢35% |

### 关键发现

- AMG-LoRA 仅增加 0.02M 参数（从 LoRA 的 0.12M 到 0.14M）就带来近 10% 的 PR 提升
- HMoE 与注意力融合性能相当但速度快 35%
- CFG 启发的动态对齐比静态对齐（固定 $w=1$）效果好 3-5 个百分点

## 亮点与洞察

- 从 Classifier-Free Guidance 借鉴到跟踪的跨模态对齐是一个巧妙的类比：将模态可靠性视为"条件"vs"无条件"分支
- "跨模态注意力对齐是打破性能-效率困境的关键"这个洞察可推广到其他多模态任务

## 局限与展望

- 仅在跟踪任务上验证，未测试在检测或分割上的效果
- HMoE 的专家数和头数需要手动调整
- 未考虑多于两个模态的场景
- 可将 AMG 扩展到更多类型的注意力对齐

## 相关工作与启发

- **vs SDSTrack**: SDSTrack 复用冻结注意力层做全局交互但复杂度高，SEATrack 用 HMoE 替代
- **vs ProTrack**: ProTrack 开创了提示调优范式但表达力有限，SEATrack 的 AMG-LoRA 更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ AMG-LoRA 和 HMoE 的设计都有新意
- 实验充分度: ⭐⭐⭐⭐ RGB-T/D/E 三个任务的全面评估
- 写作质量: ⭐⭐⭐⭐ 动机和设计逻辑清晰
- 价值: ⭐⭐⭐⭐ 对多模态 PEFT 有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](../../ICCV2025/multimodal_vlm/llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)

</div>

<!-- RELATED:END -->
