---
title: >-
  [论文解读] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
description: >-
  [NeurIPS 2025][多模态][视觉token剪枝] 从信息流（Information Flow）视角重新理解VLM中视觉token的冗余性：发现CLS token是信息中继站、冗余渐进式涌现、单层单标准评分不够可靠，提出FlowCut——基于信息流感知的多标准累积重要性剪枝框架，在LLaVA-1.5-7B上以88.9%的token减少率超越SOTA 1.6%，在LLaVA-NeXT-7B上超越4.3%。
tags:
  - NeurIPS 2025
  - 多模态
  - 视觉token剪枝
  - 信息流
  - VLM效率
  - 注意力分析
  - training-free
---

# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.19536](https://arxiv.org/abs/2505.19536)  
**代码**: [https://github.com/TungChintao/FlowCut](https://github.com/TungChintao/FlowCut)  
**领域**: 模型压缩 / 多模态VLM  
**关键词**: 视觉token剪枝, 信息流, 多标准评分, 注意力熵, training-free

## 一句话总结

从信息流（Information Flow）视角重新理解VLM中视觉token冗余性的涌现机制，提出FlowCut框架通过层自适应剪枝比例、多标准融合评分和累积重要性跟踪实现与模型内在信息传播行为对齐的token剪枝，在LLaVA-1.5-7B上以88.9% token减少率超越SOTA 1.6%，LLaVA-NeXT-7B上以94.4%减少率超越4.3%。

## 研究背景与动机

大型视觉语言模型（LVLM）在多模态理解上表现出色，但大量视觉token导致计算开销巨大。已有的token剪枝方法（FastV、SparseVLM、VisionZip等）普遍采用单层注意力分数来排序和剪枝冗余视觉token。然而，token间和层间的交互非常复杂，这引出了一个根本性问题：**仅靠单层单标准的注意力分数，是否足以准确识别冗余token？**

作者从信息流（Information Flow）这一基本视角出发，对视觉token在ViT各层中的信息传播模式进行了系统分析。信息流定义为每个token的信息流入（最主要的信息来源token）和信息流出（最主要的信息目的地token），通过注意力图来建模。通过这套分析，作者发现了三个关键洞察，揭示了冗余token产生的本质机制，并据此设计了与模型内在行为对齐的剪枝策略。

## 方法详解

### 整体框架

FlowCut是一个信息流感知的剪枝框架，包含三个核心组件：(1) 基于注意力熵的层自适应剪枝比例，(2) 多标准融合评分策略，(3) 累积流重要性跟踪。剪枝在视觉编码器的中间层执行，完全training-free，推理时即插即用。

### 关键设计

1. **信息流分析与三大发现（分析基础）**:
    - 功能：揭示视觉token冗余涌现的本质机制
    - 核心发现①——CLS token是信息中继站：patch token在浅层主要关注邻近token和CLS token，到深层则统一关注远距离的"hub token"。CLS token先从所有patch token收集全局信息，再传递给各个patch token，是全局信息流的有效代理
    - 核心发现②——冗余渐进涌现：CLS token的注意力分布随层加深逐渐集中（注意力熵递减），第11-15层熵急剧下降。冗余不是静态属性，而是在编码过程中通过层层注意力集中逐步涌现
    - 核心发现③——单标准不可靠：CLS高度关注的某些token可能信息密度很低（Value L1范数小）或语义相关性低（与CLS的余弦相似度低），不同标准会给出矛盾的重要性排序。这源于信息流中噪声的逐层放大

2. **层自适应剪枝比例（Attention Distribution-Aware Prune Ratio）**:
    - 功能：根据每层注意力集中度动态调整剪枝强度
    - 核心思路：使用CLS token的注意力熵作为指标：熵高→注意力分散→冗余少→保守剪枝；熵低→注意力集中→冗余多→激进剪枝。剪枝数量公式为 $P = \frac{N-T}{\sqrt{L}} \cdot (1 - r_H^2)$，其中 $r_H = H(\mathbf{A}^g) / H_{max}$ 是归一化熵
    - 设计动机：匹配发现②中冗余渐进涌现的内在规律，不再使用固定per-layer ratio

3. **多标准融合评分（Multi-Criteria Evaluator）**:
    - 功能：综合多个维度评估token重要性
    - 核心思路：融合三个互补指标——注意力强度 $\mathbf{I}^a$（CLS token对该token的注意力分数）、语义相关性 $\mathbf{I}^s$（与CLS token值向量的余弦相似度）、信息密度 $\mathbf{I}^d$（Value向量的L1范数）。最终分数 $\mathbf{S} = (\overline{\mathbf{I}^a} + \overline{\mathbf{I}^s}) \times \mathbf{I}^d$
    - 设计动机：针对发现③中单标准矛盾问题，多维度交叉验证避免误判

4. **累积流重要性跟踪（Cumulative Flow Importance Tracking）**:
    - 功能：跨层聚合历史和当前的重要性信息
    - 核心思路：每层计算当前多标准分数后，与历史累积分数做加权平均：$\mathbf{S}_{cum}^{(l)} = 0.5 \times \mathbf{I}_{cur}^{(l)} + 0.5 \times \mathbf{S}_{cum}^{(l-1)}$，每2层剪枝一次
    - 设计动机：单层视角不足以捕获token完整贡献，累积机制隐式过滤信息流中的噪声

### 损失函数 / 训练策略

完全无需训练（training-free），推理时即插即用。对于LLaVA-1.5，在视觉编码器倒数第二层完成剪枝。对于LLaVA-NeXT和Qwen2-VL（token更多），剪枝分两阶段：先在视觉编码器中减少到目标数量的2倍，再在LLM第2层进一步减少到最终目标数量。

## 实验关键数据

### 主实验

**LLaVA-1.5-7B（保留64个token，↓88.9%）**:

| 方法 | GQA | MMB | MME | POPE | SQA | VQA-V2 | 平均相对% |
|------|-----|-----|-----|------|-----|--------|----------|
| Vanilla (576 tokens) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 100% |
| FastV (ECCV24) | 46.1 | 48.0 | 1256 | 48.0 | 51.1 | 55.0 | 77.5% |
| SparseVLM (ICML25) | 52.7 | 56.2 | 1505 | 75.1 | 62.2 | 68.2 | 84.6% |
| VisionZip (CVPR25) | 55.1 | 60.1 | 1687 | 77.0 | 69.0 | 72.4 | 94.4% |
| **FlowCut** | **55.6** | **60.8** | **1744** | **80.2** | **69.1** | **72.8** | **96.0%** |

**LLaVA-NeXT-7B（保留160 token，↓94.4%）**: FlowCut 91.9% vs VisionZip 87.6%（**+4.3%**）

**Qwen2-VL-7B（↓88.9%）**: FlowCut 91.3% vs FastV 83.6%（**+7.7%**）

**Video-LLaVA（保留256 token，↓87.5%）**: FlowCut 98.6% vs VisionZip 94.4%（**+4.2%**）

保留192个token（↓66.7%）时FlowCut达到**100.2%**——剪枝后性能甚至微超原模型。

### 消融实验

| 配置 | 平均相对性能 | 说明 |
|------|-------------|------|
| 仅注意力分数（单标准） | ~93% | baseline |
| + 多标准融合 | ~95% | +2% 提升 |
| + 累积重要性 | ~95.5% | 额外+0.5% |
| + 自适应剪枝比例 | **96.0%** | 完整FlowCut |
| 固定比例替换自适应 | ~95% | 损失 ~1% |

### 关键发现

- 三个标准各有贡献：注意力强度是基础，信息密度和语义相关性分别贡献约0.5-1%
- 保留192 token时性能超越原模型，说明冗余token可能反而有害
- prefilling阶段实现3.2×加速

## 亮点与洞察

- **理论驱动**：从信息流视角出发的三个insight层层递进——CLS中继→冗余涌现→单标准矛盾，分析与设计水到渠成
- **CLS作为信息中继**的发现具有独立学术价值，为理解ViT内部机制提供了新视角
- 多标准融合思路简单有效：注意力强度+信息密度+语义相关性的设计直觉清晰
- 在极端压缩率（88.9%~94.4%）下仍保持优异性能，实用价值极高

## 局限与展望

- 仅在ViT架构的视觉编码器上验证，不支持CLS-free的编码器（如SigLIP）需要用全局平均token替代
- 多标准权重（加法+乘法组合）和累积系数0.5/0.5是手动设置的，可以学习自适应权重
- 未与KV cache压缩、层级剪枝等维度级方法结合
- 评估以理解任务为主，生成质量（如详细图像描述）影响未充分评估

## 相关工作与启发

- **vs FastV**: FastV用第2层attention做一次性剪枝，是最简单的baseline；FlowCut通过跨层累积+多标准实现大幅领先
- **vs VisionZip**: VisionZip也关注信息密度但仍只用单标准+固定保留比例；FlowCut全面优于它
- **vs PyramidDrop**: PyramidDrop用预定义的分层schedule，FlowCut的自适应机制更灵活
- 信息流分析框架可推广到LLM文本token分析——文本token可能也存在类似的注意力集中和冗余涌现现象

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 信息流视角和三个insight都是该领域的重要理论贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 4个VLM架构（LLaVA-1.5/NeXT、Qwen2-VL、Video-LLaVA）、12个benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 信息流可视化极其精美，分析-设计逻辑链清晰流畅
- 价值: ⭐⭐⭐⭐⭐ 当前VLM token剪枝领域的SOTA方法，理论和实践价值兼具
---
title: >-
  [论文解读] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
description: >-
  [NeurIPS 2025][多模态][视觉token剪枝] 从信息流（Information Flow）视角重新理解VLM中视觉token的冗余性：发现CLS token是信息中继站、冗余渐进式涌现、单层单标准评分不够可靠，提出FlowCut——基于信息流感知的多标准累积重要性剪枝框架，在LLaVA-1.5-7B上以88.9%的token减少率超越SOTA 1.6%，在LLaVA-NeXT-7B上超越4.3%。
tags:
  - NeurIPS 2025
  - 多模态
  - 视觉token剪枝
  - 信息流
  - VLM效率
  - 注意力分析
  - training-free
---

# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.19536](https://arxiv.org/abs/2505.19536)  
**代码**: [https://github.com/TungChintao/FlowCut](https://github.com/TungChintao/FlowCut)  
**领域**: 模型压缩 / 多模态VLM  
**关键词**: 视觉token剪枝, 信息流, VLM效率, 注意力分析, training-free

## 一句话总结
从信息流（Information Flow）视角重新理解VLM中视觉token的冗余性：发现CLS token是信息中继站、冗余渐进式涌现、单层单标准评分不够可靠，提出FlowCut——基于信息流感知的多标准累积重要性剪枝框架，在LLaVA-1.5-7B上以88.9%的token减少率超越SOTA 1.6%，在LLaVA-NeXT-7B上超越4.3%。

## 背景与动机
VLM因大量视觉token导致计算开销高。已有方法用单层注意力分数做token重要性排序和剪枝（如FastV、SparseVLM），但作者提出一个根本性质疑：token间和层间的交互如此复杂，仅看单层注意力分数是否足以识别冗余？

## 核心问题
从信息流角度理解视觉token冗余的本质是什么？如何设计与模型固有信息传播行为对齐的剪枝策略？

## 方法详解

### 整体框架
三大创新模块：(1) 基于注意力熵的层自适应剪枝比例；(2) 多标准评分策略（注意力强度+信息密度+语义相关性）；(3) 累积流重要性跟踪。

### 关键发现（信息流分析）

1. **Insight 1 — CLS token是信息中继站**: 通过分析ViT各层的信息流入/流出，发现patch token在浅层主要关注邻近token和CLS token，在深层则关注远距离的"hub token"。CLS token充当全局信息广播器——它先从所有patch token收集信息，再传递给每个patch token。因此**CLS token的注意力可以作为全局信息流的代理**。

2. **Insight 2 — 冗余渐进式涌现**: CLS token的注意力分布随层加深而逐步集中（熵递减），冗余不是静态属性而是在编码过程中层层涌现。第11-15层熵急剧下降。

3. **Insight 3 — 单标准不可靠**: CLS高度关注的token可能信息密度很低（Value L1范数小）或语义相关性低（与CLS余弦相似度低）——不同标准会给出矛盾的重要性排序！

### 关键设计

1. **层自适应剪枝比例**: 使用注意力熵指导每层的剪枝强度——熵低（注意力集中→冗余多）的层剪枝更激进，熵高的层保守剪枝。不再用固定的per-layer ratio。

2. **多标准融合评分**: 综合三个维度评估token重要性：

    - **注意力强度**: CLS token对该token的注意力分数
    - **信息密度**: Value向量的L1范数（信号强度）
    - **语义相关性**: 与CLS token的余弦相似度（全局语义相关性）

3. **累积流重要性跟踪**: 不只看当前层，而是跨层累积重要性：`S_cum^(l) = 0.5 × I_cur^(l) + 0.5 × S_cum^(l-1)`。每2层剪枝一次，确保历史和当前信息都被考虑。

### 损失函数 / 训练策略
- 完全无需训练（training-free），推理时即插即用
- 剪枝在ViT的中间层执行

## 实验关键数据

**LLaVA-1.5-7B（保留64个token，原始576个，↓88.9%）**:

| 方法 | 平均准确率 (相对100%) |
|------|---------------------|
| Vanilla (576 tokens) | 100% |
| FastV | 77.5% |
| SparseVLM | 84.6% |
| PDrop | 86.1% |
| VisionZip | 94.4% |
| **FlowCut** | **96.0%** (+1.6%) |

**LLaVA-NeXT-7B（保留32个token，原始2880个，↓94.4%）**: FlowCut 91.9% vs VisionZip 87.6% (+4.3%)

- 保留192个token（↓66.7%）时：FlowCut达到100.2%——**甚至超过原始模型**！
- prefilling阶段3.2×加速
- 在InternVL2和Qwen2-VL上也展示了泛化性

### 消融实验要点
- **多标准 vs 单标准**: 多标准评分比仅用attention显著更好
- **累积 vs 单层**: 累积重要性跟踪提升2-3%
- **自适应比例 vs 固定比例**: 自适应剪枝比固定ratio高1-2%
- **三个标准的贡献**: 注意力强度是基础，信息密度和语义相关性各贡献约0.5-1%提升

## 亮点
- **理论驱动**: 从信息流的基本视角出发，三个insight层层递进，分析→设计水到渠成
- **保留64 token达96%**: 88.9%的token减少率下仍保持96%相对性能——在同等压缩率下远超所有方法
- **超越原模型的可能**: 192 token时100.2%——剪枝后性能甚至微超原模型，说明冗余token可能有害
- **CLS作为信息中继的发现**: 对ViT内部机制的理解有独立学术价值

## 局限与展望
- 仅在ViT架构的视觉编码器上分析信息流，其他架构（如SigLIP without CLS）尚未验证
- 多标准的权重（0.5/0.5）是手动设置的，可以学习自适应权重
- 未与KV-Latent等维度级压缩方法结合
- prefilling加速3.2×，但decoding阶段的加速未报告（KV cache大小也会减少）
- 评估以理解任务为主，生成任务（如图像描述）的影响程度未充分测试

## 与相关工作的对比
- **vs FastV**: FastV用第2层attention做一次性剪枝，FlowCut跨层累积+多标准，大幅领先
- **vs VisionZip**: VisionZip也是strong baseline但只用attention+少量固定保留，FlowCut自适应+多标准更优
- **vs mPLUG-DocOwl2 (cross-attention压缩)**: DocOwl2需要额外的压缩模块和训练，FlowCut完全无需训练

## 启发与关联
- 信息流分析框架可以推广到LLM的token级分析——不仅是视觉token，文本token也可能存在类似的信息集中现象
- 层自适应剪枝比例的思路与TrimLLM（层级压缩）互补——FlowCut在层内做token剪枝，TrimLLM在层间做层删除
- "CLS作为信息中继"的发现可以用于KV-Latent——维度降采之后，CLS token的KV可以保持更高维度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息流视角和三个insight都是该领域的重要理论贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 4个VLM架构、12个benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ Figure 2-4的信息流可视化极其精美和直观
- 价值: ⭐⭐⭐⭐⭐ 当前VLM token剪枝领域的SOTA方法，理论和实践价值兼具

<!-- RELATED:START -->

## 相关论文

- [Cross-modal Information Flow in Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/cross-modal_information_flow_in_multimodal_large_language_models.md)
- [NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)

<!-- RELATED:END -->
