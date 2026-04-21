---
title: >-
  [论文解读] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models
description: >-
  [ICLR 2026][图像生成][统一多模态模型] Uni-X提出一种两端分离、中间共享的X型架构来缓解统一多模态模型（UMM）中视觉与文本模态的梯度冲突，通过将浅层和深层设为模态专属、中间层共享参数，3B参数即可匹配或超越7B AR-UMM在图像生成和多模态理解上的性能。
tags:
  - ICLR 2026
  - 图像生成
  - 统一多模态模型
  - 梯度冲突
  - 模态分离
  - 自回归生成
  - 图像生成与理解
---

# Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models

**会议**: ICLR 2026  
**arXiv**: [2509.24365](https://arxiv.org/abs/2509.24365)  
**代码**: https://github.com/CURRENTF/Uni-X  
**领域**: 多模态统一模型  
**关键词**: 统一多模态模型, 梯度冲突, 模态分离, 自回归生成, 图像生成与理解

## 一句话总结
Uni-X提出一种两端分离、中间共享的X型架构来缓解统一多模态模型（UMM）中视觉与文本模态的梯度冲突，通过将浅层和深层设为模态专属、中间层共享参数，3B参数即可匹配或超越7B AR-UMM在图像生成和多模态理解上的性能。

## 研究背景与动机

**领域现状**：统一多模态模型（UMM）旨在单一框架中同时支持图像理解和生成。自回归（AR）方式通过VQ将视觉token化为"外语"，架构简洁但性能受限。复杂设计（如MoT、AR+Diffusion混合、任务分支）虽有效但牺牲了参数共享和扩展性。

**现有痛点**：完全模态共享的Transformer在联合训练时存在严重的梯度冲突。作者首次将多任务学习中的梯度冲突概念迁移到UMM，发现视觉和文本梯度在浅层和深层方向严重冲突。

**核心矛盾**：图像token序列的条件熵远高于自然语言（英语、德语、中文），意味着视觉序列内在更难预测，需要建模更长距离的空间纠缠依赖。当共享Transformer同时处理低熵文本和高熵视觉时，浅层和深层被迫调和冲突的低级分布。

**本文目标**：如何在保持纯AR架构简洁性的同时，有效缓解模态间梯度冲突？

**切入角度**：通过梯度冲突的经验分析发现冲突在中间层减弱（抽象语义对齐），据此设计层级结构。

**核心 idea**：浅层和深层做模态专属处理（处理不同低级统计分布），中间层共享参数（利用高级语义对齐），形成X型分离-共享架构。

## 方法详解

### 整体框架
给定预训练LLM的 $L$ 层 $\{\text{Layer}_t^i\}_{i=0}^{L-1}$，将其分为三段：前 $N$ 层和后 $M$ 层为"分离层"，中间层为"共享层"。分离层中引入新的视觉专用层 $\{\text{Layer}_v^i\}$，与原始文本层并行。视觉token通过二值掩码 $M_v$ 在前向传播中路由到对应分支。

### 关键设计

1. **梯度冲突量化**:

    - 功能：定义并测量跨模态梯度冲突
    - 核心思路：计算文本梯度 $g_{\text{text}}$ 和图文梯度 $g_{\text{img}}$ 的余弦相似度 $S_{\text{inter}}$，再通过随机分割数据计算基线相似度 $S_{\text{base}}$，冲突定义为 $c_g = -(S_{\text{inter}} - S_{\text{base}})$
    - 发现：冲突在浅层和深层最严重，中间层最弱

2. **两端分离架构**:

    - 前向传播：$H_x^{l+1} = \text{Layer}_x^l(H_x^l)$ 当 $l < N$ 或 $l \geq L-M$（分离层）；$H_x^{l+1} = [\text{Layer}_t^l(H^l)]_x$ 否则（共享层）
    - 关键约束：分离块内视觉和文本模态严格隔离，无跨模态交互，强制模型先学习鲁棒的单模态表示
    - 设计动机：对齐模型结构与模态特征——低级特征处理模态专属，高级语义融合共享

3. **信息论解释**:

    - 通过n-gram条件熵分析，图像token的熵远高于自然语言
    - 高条件熵意味着预测更难，需要更长距离依赖
    - 这解释了为什么共享浅层/深层会产生冲突：不同复杂度的序列需要不同的低级处理

### 训练目标
标准自回归交叉熵损失：$\mathcal{L} = -\sum_{i=1}^T \log P(s_i | s_{<i})$。使用VQGAN tokenizer（来自Chameleon）将512×512图像编码为32×32离散token（8192码本）。CFG统一设为4.0。

## 实验关键数据

### 文本性能

| 模型 | 参数 | ARC-E | ARC-C | WinoG | BoolQ | MMLU | Avg |
|------|------|-------|-------|-------|-------|------|-----|
| Chameleon | 7B | 76.1 | 46.5 | 70.4 | 81.4 | 52.1 | 65.3 |
| Liquid | 7B | 75.6 | 49.0 | 72.7 | 81.0 | 56.0 | 66.9 |
| **Uni-X** | **3B/4.5B** | **79.0** | 47.9 | 68.9 | **82.2** | **57.6** | **67.1** |

### 图像生成与多模态理解

| 模型 | 参数 | GenEval | DPG | MME | POPE | MMB | SEED |
|------|------|---------|-----|-----|------|-----|------|
| EMU3 | 8B | 66 | 80.6 | 1243.8 | 85.2 | 58.5 | 68.2 |
| Liquid | 7B | 68 | 79.8 | 1107.2 | 81.1 | — | — |
| Janus-Pro | 7B | 80 | 84.1 | — | 87.4 | 79.2 | 72.1 |
| **Uni-X** | **3B/4.5B** | **82** | 79.8 | 1158.3 | 83.6 | 59.3 | 60.2 |

### 关键发现
- 3B参数的Uni-X在GenEval上达到82分，超越大多7B AR-UMM，包括Chameleon（39）、EMU3（66）、Liquid（68）
- Uni-X不使用语义编码器（CLIP/SigLIP），纯AR架构，表明梯度冲突才是限制性能的关键瓶颈
- 梯度冲突分析显示，Uni-X不仅避免了两端冲突，还进一步缓解了中间共享层的残余冲突
- 在相同训练条件下，Uni-X的训练效率优于全共享基线

## 亮点与洞察
- **问题发现比解决方案更重要**：首次在UMM中量化梯度冲突并追溯到信息论根源（熵差异），这一分析框架可广泛应用于其他多模态/多任务系统
- **简洁性的胜利**：相比MoT、AR+Diffusion混合等复杂设计，Uni-X仅通过层级划分就达到竞争性能，保持了纯AR的扩展性优势
- **参数效率**：3B匹配7B的性能意味着架构设计可替代蛮力缩放
- **信息论视角新颖**：用条件熵解释"视觉是更难的外语"，直观且有说服力

## 局限与展望
- 当前仅处理非交错的多模态输入，交错序列（图文混排）场景未验证
- N和M（分离层数量）的选择似乎需要经验调节，缺乏原则性指导
- 分离层内严格隔离可能限制早期跨模态信息交流
- 可以探索动态/自适应的层分配策略

## 相关工作与启发
- **vs Chameleon**：完全共享架构导致严重冲突，7B参数GenEval仅39分；Uni-X 3B达82分
- **vs MoT/UniFork**：这些方法通过增加模块复杂性缓解冲突，但牺牲参数共享；Uni-X保持简洁
- **vs Janus-Pro**：使用额外语义编码器达到80分GenEval；Uni-X无需额外编码器达到82分

## 补充细节
- 预训练数据：72B文本tokens + 65B视觉tokens，来自CCI3-H、DCLM、Fineweb-Edu等
- SFT阶段使用3B tokens，包括MiniGemini、FineVision、OpenOrca等
- 消融在Qwen2.5-1.5B上进行，缩放到Qwen2.5-3B
- 分离层数量 $N$ 和 $M$ 通过消融确定最佳配置
- 训练使用Flash Attention 2和DeepSpeed ZeRO2加速

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度冲突分析+信息论解释+X型架构的组合有新意
- 实验充分度: ⭐⭐⭐⭐ 文本、生成、理解全面评估，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，从分析到设计逻辑严密
- 价值: ⭐⭐⭐⭐ 为UMM设计提供了实用的架构指导原则

<!-- RELATED:START -->

## 相关论文

- [When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](../../NeurIPS2025/image_generation/mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)
- [Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing](draw-in-mind_rebalancing_designer-painter_roles_in_unified_multimodal_models_ben.md)
- [Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)
- [Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)

<!-- RELATED:END -->
