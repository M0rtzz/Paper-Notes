---
title: >-
  [论文解读] Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters
description: >-
  [CVPR 2025][图像生成][概念擦除] 本文提出 AdaVD（Adaptive Value Decomposer），一种免训练的 T2I 扩散模型概念擦除方法，通过在 cross-attention 的 value 空间中将原始 prompt 投影到目标概念的正交补空间，并引入自适应 shift 因子，实现了精确擦除目标概念且极少影响非目标内容。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "概念擦除"
  - "扩散模型安全"
  - "正交补空间"
  - "免训练方法"
  - "先验保持"
---

# Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters

**会议**: CVPR 2025  
**arXiv**: [2412.06143](https://arxiv.org/abs/2412.06143)  
**代码**: [GitHub](https://github.com/WYuan1001/AdaVD)  
**领域**: 图像生成  
**关键词**: 概念擦除, 扩散模型安全, 正交补空间, 免训练方法, 先验保持

## 一句话总结

本文提出 AdaVD（Adaptive Value Decomposer），一种免训练的 T2I 扩散模型概念擦除方法，通过在 cross-attention 的 value 空间中将原始 prompt 投影到目标概念的正交补空间，并引入自适应 shift 因子，实现了精确擦除目标概念且极少影响非目标内容。

## 研究背景与动机

- **T2I 模型的安全隐患**：文生图扩散模型可能生成版权、攻击性或不安全内容，原因是训练数据来自互联网的噪声标注。从零重新训练成本巨大，因此需要低成本的概念擦除技术。
- **擦除效率与先验保持的矛盾**：概念擦除需要在两个目标间取得平衡——(1) 精确删除目标概念（erasure efficacy），(2) 最小化对非目标内容的影响（prior preservation）。
- **训练方法的局限**：ESD、UCE 等需要为每个新概念单独微调，不适合实时场景（如在线平台需要即时封禁新出现的侵权概念）。且正则化难以同时保证擦除效率和先验保持。
- **现有免训练方法的不足**：Negative Prompt 擦除效果弱；SLD 严重影响非目标先验；SuppressEOT 需要人工指定目标位置，不够自动化。
- **核心洞察**：cross-attention 中 Key 控制"Where"（布局），Value 控制"What"（内容）。概念擦除本质是修改视觉内容，因此应在 Value 空间操作。将原始 prompt 的 value 向量投影到目标概念 value 向量的正交补空间，即可精确移除目标语义。

## 方法详解

### 整体框架

AdaVD 在 UNet 的每个 cross-attention 层中执行三步操作：
1. **Token-wise 目标嵌入预处理**：复制目标概念的关键 token 嵌入，强化目标语义信号
2. **正交值分解**：将原始 prompt 的 value 投影到目标 value 的正交补空间
3. **自适应 shift 调整**：根据 token 与目标概念的对齐强度动态调节擦除力度

### 关键设计

**1. Token-wise 目标嵌入复制**
- **功能**：增强目标概念在 value 空间中的语义表达，使正交分解更精确
- **核心思路**：将目标概念的最后一个主题 token（如 "snoopy" 或 "gogh"）复制到除 [SOT] 外的所有位置。利用 CLIP 因果注意力机制，最后一个主题 token 已聚合了完整的 prompt 信息
- **设计动机**：原始目标嵌入中包含大量 [EOT] 填充，语义信号被稀释；复制关键 token 可强化语义并提高正交分解的精度

**2. 正交补投影（核心操作）**
- **功能**：从原始 prompt 的视觉内容中精确移除目标概念的语义成分
- **核心思路**：对每个 token 位置 $j$，将原始 value 向量 $\boldsymbol{v}^j$ 投影到目标 value 向量 $\boldsymbol{v}_t^j$ 的正交补空间：$\boldsymbol{v}_r^j = \boldsymbol{v}^j - \frac{\boldsymbol{v}_t^{j\top}\boldsymbol{v}^j}{\boldsymbol{v}_t^{j\top}\boldsymbol{v}_t^j}\boldsymbol{v}_t^j$。多概念擦除时，先对多个目标 value 做 Gram-Schmidt 正交化，再投影到联合正交补空间
- **设计动机**：正交补投影在数学上保证了移除目标方向的分量后保留最大信息量，是最优的语义分离操作

**3. 自适应 Shift 因子**
- **功能**：区分 token 与目标概念的强/弱对齐，避免过度擦除非目标内容
- **核心思路**：计算每个 token 的投影系数 $\alpha_j = \frac{\boldsymbol{v}_t^{j\top}\boldsymbol{v}^j}{\boldsymbol{v}_t^{j\top}\boldsymbol{v}_t^j}$，衡量该 token 与目标概念的对齐程度。强对齐的 token（如 prompt 中直接包含目标概念词）完全擦除，弱对齐的 token（通用语义如 "[EOT]"）保留更多原始信息
- **设计动机**：所有 token 都会与目标概念有一定程度的对齐（特别是 [EOT] 等通用 token），如果一律完全投影会损害先验知识。自适应 shift 实现了精细的"分级擦除"

### 损失函数

AdaVD 是免训练方法，不涉及损失函数优化。核心操作是推理时直接修改 cross-attention 的 value 矩阵。

## 实验关键数据

### 主实验：单概念擦除性能比较

| 方法 | 类型 | 擦除效率 | 先验保持 (CLIP-T↑) | 运行时间 |
|------|------|:---:|:---:|:---:|
| ESD | 训练 | 高 | 低 | 慢 |
| UCE | 训练 | 高 | 中 | 慢 |
| SLD | 免训练 | 中 | 低 | 快 |
| **AdaVD** | **免训练** | **高/近最优** | **最优 (2-10x↑)** | **快** |

### 多概念擦除

| 方法 | 5概念同时擦除效率 | 先验保持 |
|------|:---:|:---:|
| ESD | 中 | 差 |
| SLD | 差 | 差 |
| **AdaVD** | **高** | **最优** |

### 关键发现

- AdaVD 在先验保持上比第二名提升 **2-10 倍**，同时保持最优或近最优的擦除效率
- 免训练方法首次达到甚至超越训练方法的擦除效果
- Shift 因子对先验保持至关重要——移除后非目标概念生成质量显著下降
- 方法可直接迁移到 SDXL、DreamShaper、Chilloutmix 等不同 T2I 模型
- 支持版权概念（Snoopy）、风格（Van Gogh）、不安全概念（NSFW）等多种擦除场景

## 亮点与洞察

1. **数学优雅性**：正交补投影是线性代数中最基本的操作之一，但在概念擦除中的应用异常有效——最优地分离目标/非目标语义
2. **免训练 + 高精度**：首次在免训练框架下实现超越训练方法的先验保持能力
3. **即插即用**：无需微调模型参数，可实时擦除任意新概念，适合在线部署
4. **可扩展的多概念擦除**：通过 Gram-Schmidt 正交化自然扩展到同时擦除多个概念

## 局限与展望

- 擦除效果依赖 CLIP 文本编码器对目标概念的语义表达质量
- 正交补投影在 value 空间中是线性操作，可能无法完全处理概念间的非线性纠缠
- 极端情况下（目标概念与非目标概念语义高度重叠）可能仍有泄露
- 未来可结合 attention map 的空间信息实现更精细的区域级擦除

## 相关工作与启发

- **ESD (Erased Stable Diffusion)**：训练式擦除先驱，但不考虑先验保持
- **SLD (Safe Latent Diffusion)**：免训练先驱，但先验保持差
- **SAFREE**：同期工作也用正交分解但在文本嵌入空间操作
- **Cross-attention 分析**：Key=Where, Value=What 的认知是方法设计的理论基础
- 启发：经典数学工具（正交投影）与深度模型的内部机制（attention value）的结合可以产生简洁而有效的解决方案

## 评分

⭐⭐⭐⭐ — 方法设计优雅简洁，"正交补投影"的核心思想直觉清晰且数学严谨。先验保持性能大幅领先。免训练特性使其极具实用价值。Shift 因子的自适应设计解决了关键的过度擦除问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](../../ICML2026/image_generation/orthogonal_concept_erasure_for_diffusion_models.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2025\] Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization](concept_replacer_replacing_sensitive_concepts_in_diffusion_models_via_precision_.md)

</div>

<!-- RELATED:END -->
