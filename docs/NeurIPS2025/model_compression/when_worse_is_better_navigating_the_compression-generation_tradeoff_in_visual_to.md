---
title: >-
  [论文解读] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization
description: >-
  [NeurIPS 2025][模型压缩][视觉Tokenizer] 本文通过scaling law系统研究了视觉tokenizer压缩率与生成质量的权衡关系，发现对小模型而言更激进的压缩（虽然重建更差）反而有利于生成…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "视觉Tokenizer"
  - "压缩-生成权衡"
  - "因果正则化"
  - "Scaling Law"
  - "自回归生成"
---

# When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2412.16326](https://arxiv.org/abs/2412.16326)  
**代码**: 无  
**领域**: 模型压缩 / 图像生成  
**关键词**: 视觉Tokenizer, 压缩-生成权衡, 因果正则化, Scaling Law, 自回归生成

## 一句话总结

本文通过scaling law系统研究了视觉tokenizer压缩率与生成质量的权衡关系，发现对小模型而言更激进的压缩（虽然重建更差）反而有利于生成，并提出因果正则化Tokenization(CRT)方法在stage 1训练中嵌入自回归归纳偏置，实现2-3倍计算效率提升，以775M参数和256 token/image匹配LlamaGen-3B的2.18 FID。

## 研究背景与动机

**领域现状**：现代图像生成方法普遍采用两阶段训练：Stage 1训练auto-encoder（如VQGAN）将图像压缩到潜空间；Stage 2训练生成模型（如自回归Transformer）学习潜空间分布。这意味着Stage 1的tokenizer设计深刻影响Stage 2的生成性能。

**现有痛点**：Stage 1和Stage 2的优化目标存在根本冲突——Stage 1追求重建质量（低rFID），但更好的重建不一定带来更好的生成（低gFID）。已有工作零散地观察到这一现象（如增大codebook反而降低生成质量），但缺乏系统的定量分析框架来理解这一权衡。

**核心矛盾**：rate-distortion-generation的三方权衡。一端是不压缩（完美重建但潜空间分布难学），另一端是极端压缩（分布简单但重建太差）。最优点取决于Stage 2模型的容量——这个交互关系此前未被严格研究。

**本文目标** (1) 压缩程度如何影响不同规模的Stage 2模型的生成性能？ (2) 在给定计算预算下，什么是最优的压缩-生成配置？ (3) 能否设计Stage 1 tokenizer来直接优化Stage 2的生成性能？

**切入角度**：用scaling law作为分析工具，在5个数量级的计算资源和2个数量级的模型规模上系统研究这一权衡，然后设计CRT将Stage 2的自回归归纳偏置注入Stage 1。

**核心 idea**：通过在tokenizer训练中加一个轻量因果Transformer的预测损失，使token天然更适合自回归建模，用"更差的重建"换取"更好的生成"。

## 方法详解

### 整体框架

标准的两阶段pipeline：Stage 1是VQGAN（ResNet编码器+解码器+VQ量化），Stage 2是Llama架构的自回归Transformer。本文在Stage 1训练中加入CRT正则项——一个2层因果Transformer对预量化latent做L2 next-token prediction，梯度回传到编码器。Stage 2训练完全不改变。

### 关键设计

1. **Scaling Law分析框架**:

    - 功能：定量揭示压缩率与生成性能在不同计算预算下的关系
    - 核心思路：固定Stage 1 tokenizer，变化Stage 2模型大小和训练计算量，拟合L(C) = L_min + C^α·e^λ的scaling law。在4个数量级的计算FLOPs上观察到log-log-linear的一致趋势。关键发现：(a) 更少的token/image在饱和前通常更具计算效率；(b) 16k codebook优于1k和131k，存在最优压缩率；(c) 1k codebook在低计算量时优于131k（gFID相差5-10），但在高计算量时趋势反转
    - 设计动机：之前的工作只在单一规模上比较不同tokenizer，忽略了计算预算这一关键维度

2. **因果正则化Tokenization (CRT)**:

    - 功能：在Stage 1训练中嵌入自回归归纳偏置，使token更易于Stage 2建模
    - 核心思路：在VQGAN标准训练损失的基础上，引入一个2层因果Transformer对预量化latent做L2 next-token prediction。关键设计选择：(a) 使用L2损失而非交叉熵——因为VQ token在训练中离散变化，L2天然具有相似性感知能力；(b) 作用在预量化latent而非量化后token——避免离散化引起的重建损失过大；(c) 损失权重λ=4，通过消融确定的最优点
    - 设计动机：既然Stage 2是自回归Transformer，那么让token i尽可能可由token 0~i-1预测，就能降低Stage 2的建模难度。这本质上是在信息论意义上降低了条件熵H(X_i|X_{<i})

3. **重建-生成权衡的量化分析**:

    - 功能：解释为什么CRT虽然损害重建但改善生成
    - 核心思路：CRT使rFID从2.21涨到2.36（重建变差），但Stage 2的验证损失在所有位置上均匀降低（尤其是序列尾部），说明CRT减小了条件熵。进一步分析发现：改变codebook size几乎不影响每位置熵（codes按位置特化），所以codebook扩大对scaling影响有限；而改变token数量会线性增加推理成本，对scaling影响更大
    - 设计动机：不能只看重建指标来设计tokenizer——生成性能才是最终目标

### 训练策略

CRT正则器仅增加5%训练FLOPs（2层Transformer极轻量）；为公平对比，CRT模型减少5%训练iteration（从400k降到380k）。CRT损失权重从0平滑退火到4.0，退火时长1k步。使用独立的AdamW优化器。

## 实验关键数据

### 主实验：ImageNet 256×256系统对比

| 模型 | 参数量 | Token/Image | gFID↓ |
|------|--------|-------------|-------|
| LlamaGen-XL | 775M | 576 | 2.62 |
| LlamaGen-3B | 3.1B | 576 | 2.18 |
| CRT-AR-775M | 775M | 256 | 2.35 |
| CRTopt-AR-775M | 775M | 256 | **2.18** |

### 消融实验：CRT vs Baseline（同计算量）

| Tokenizer | rFID | 111M gFID | 340M gFID | 775M gFID |
|-----------|------|-----------|-----------|-----------|
| Baseline (2.21 rFID) | 2.21 | 4.90 | 2.89 | 2.55 |
| CRT (2.36 rFID) | 2.36 | 4.34 | 2.75 | 2.35 |

### 关键发现

- CRT虽然重建更差（2.36 vs 2.21 rFID），但在所有模型规模和计算量下均改善生成性能——验证了"worse is better"的核心论点
- CRT的scaling law斜率从α=-0.65改善到α=-0.73，带来1.5-3倍的计算效率提升
- CRTopt（更长训练+更大decoder+131k codebook）以775M参数、256 token/image匹配了LlamaGen-3B的2.18 FID——推理计算量减少8倍
- L2损失优于交叉熵：CE损失对重建的损害过大但生成收益不足以弥补
- LSUN数据集上CRT同样普遍优于baseline，确认了方法的通用性

## 亮点与洞察

- "Worse is better"的洞察极具启发性：在两阶段系统中，独立优化每个阶段并非最优——应该考虑阶段间的交互。CRT用微小的重建代价换取显著的生成改善，本质上是将Stage 2的学习负担前移到了Stage 1。这一原则可推广到任何多阶段ML pipeline
- Scaling law作为分析工具非常有效：跨5个数量级的计算预算提供了稳健的趋势观察，避免了在单一设定下的偶然结论。log-log-linear关系使得结果可预测和可外推
- CRT的设计极其简洁：仅2层因果Transformer + L2损失 + 5%额外训练FLOPs，却带来了2-3倍计算效率提升。这种"用简单正则项嵌入归纳偏置"的思路值得借鉴

## 局限与展望

- 最大模型仅到775M参数（受限于ImageNet数据量），更大模型下的scaling趋势未验证
- 仅评估了类别条件生成，未扩展到文本到图像生成。CLIP FID在LSUN上的不稳定性说明评估指标本身也存在局限
- CRT的VQGAN架构是固定的——如果应用到1D tokenizer或VAR等架构，效果可能不同
- 未研究CRT与扩散模型Stage 2的兼容性——CRT的因果偏置是否对非自回归生成也有效？

## 相关工作与启发

- **vs LARP (Wang et al.)**：同样使用自回归归纳偏置但用CE+随机VQ+scheduled sampling，本文用更简单的L2预量化损失取得更好效果——因为VQ架构中L2比CE更适合
- **vs VAR (Tian et al.)**：通过多尺度token改变了autoregressive顺序，需要修改tokenizer架构；CRT不修改架构，更通用
- **vs FlexTok / SEED等1D tokenizer**：构建全新的tokenizer架构来适配生成；CRT在标准VQGAN上加正则项即可，工程成本更低

## 评分

- 新颖性: ⭐⭐⭐⭐ Scaling law视角分析压缩-生成权衡有新意，CRT方法简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 跨5个数量级计算量、7种模型规模、3个数据集，消融极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 实验驱动的叙事方式流畅，图表设计精美，结论有说服力
- 价值: ⭐⭐⭐⭐⭐ 对视觉tokenizer设计和多阶段系统优化有深远指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency](mixed_monotonicity_reachability_analysis_of_neural_ode_a_trade-off_between_tight.md)
- [\[NeurIPS 2025\] zip2zip: Inference-Time Adaptive Tokenization via Online Compression](zip2zip_inference-time_adaptive_tokenization_via_online_compression.md)
- [\[ICML 2025\] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective](../../ICML2025/model_compression/rethinking_the_stability-plasticity_trade-off_in_continual_learning_from_an_arch.md)
- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)

</div>

<!-- RELATED:END -->
