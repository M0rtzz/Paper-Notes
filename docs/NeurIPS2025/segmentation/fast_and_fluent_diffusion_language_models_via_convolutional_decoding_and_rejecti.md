---
title: >-
  [论文解读] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning
description: >-
  [NeurIPS 2025][图像分割][扩散模型] 通过卷积解码归一化（替代硬半自回归分块）和基于规则的拒绝微调 R2FT，在 128 步推理下实现与 512+ 步相当的扩散语言模型生成质量，达到 DLM 领域 SOTA。
tags:
  - NeurIPS 2025
  - 图像分割
  - 扩散模型
  - convolutional decoding normalization
  - rejective fine-tuning
  - semi-autoregressive
  - time-gap expansion
---

# Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning

**会议**: NeurIPS 2025  
**arXiv**: [2509.15188](https://arxiv.org/abs/2509.15188)  
**代码**: 未开源  
**领域**: 文本生成 / 扩散语言模型  
**关键词**: diffusion language model, convolutional decoding normalization, rejective fine-tuning, semi-autoregressive, time-gap expansion

## 一句话总结

通过卷积解码归一化（替代硬半自回归分块）和基于规则的拒绝微调 R2FT，在 128 步推理下实现与 512+ 步相当的扩散语言模型生成质量，达到 DLM 领域 SOTA。

---

## 研究背景与动机

**扩散语言模型兴起**：MDLM、SEDD、LLADA 等扩散语言模型（DLM）通过迭代去噪生成文本，天然支持双向依赖建模。与自回归模型（ARM）不同，DLM 可同时考虑上下文信息，理论上表达能力更强。

**推理速度瓶颈**：现有 DLM 需要 512–1024 步才能获得合理的生成质量，这使其推理速度远低于 ARM。减少步数是 DLM 实用化的关键挑战。

**半自回归方法的缺陷**：Block-diffusion 等方法通过将序列切分为固定大小的块进行半自回归（SAR）解码来加速推理，但引入了"时间间隔扩展"（time-gap expansion）问题——训练时假设的连续时间条件在少步推理时被违反，导致块边界处质量急剧下降。

**内容退化问题**：DLM 常生成高先验词（如 "the"、"and" 占据不合理位置）和重复 token，这是扩散过程中缺乏自回归约束的固有退化模式，在减少步数后愈加严重。

**缺乏系统分析**：现有工作缺乏对 DLM 步数-质量 trade-off 的数学分析，也没有区分"窗口不匹配"和"内容退化"这两类独立问题。

**核心思路**：作者提出两个正交解决方案——卷积解码归一化通过连续收缩（而非硬切块）解决窗口不匹配问题；R2FT 通过 DPO 风格后训练消除退化 token，二者组合在 128 步达到 512 步水平。

---

## 方法详解

### 整体框架

本文针对 DLM 的两个独立问题提出两个正交解决方案，可分别或组合使用：
- **卷积解码归一化**（Convolutional Decoding Normalization）：解决半自回归推理中的时间间隔扩展问题
- **拒绝微调 R2FT**（Rejective Fine-Tuning）：消除高先验词和重复 token 的退化模式

### 关键设计 1：时间间隔扩展的理论分析

- **问题定义**：在半自回归解码中，序列被分为 $B$ 个块，每块内并行去噪。训练时 DLM 假设时间步 $t$ 连续变化，但 SAR 推理时每块的有效时间步是 $t_\beta = 1/S_\beta$（$S_\beta$ 为块内步数）。当总步数减少时，$d \cdot t_\beta$ 增大，违反连续时间假设
- **后果**：块边界处出现明显的质量跳跃，表现为不连贯、语法错误和信息缺失
- **意义**：这一理论分析解释了为何简单减少步数会导致 SAR-DLM 质量骤降，为解决方案提供了数学指导

### 关键设计 2：卷积解码归一化

- **核心思路**：用可学习的位置依赖缩放替代硬分块切换。定义缩放因子 $s_i = \tanh(u_i)$，其中 $u_i$ 基于卷积核内已解码邻居的数量计算，实现从"全掩码"到"全解码"的**连续过渡**
- **工作机制**：在每一步去噪中，已解码位置的邻居越多，该位置的解码置信度越高。卷积窗口逐渐从左向右扫过序列，避免了硬边界
- **训练集成**：卷积归一化直接嵌入 DLM 训练流程，使模型学习在连续窗口下工作，推理和训练一致
- **优势**：连续收缩消除了 $d \cdot t_\beta$ 跳变，使 128 步推理不再违反训练假设

### 关键设计 3：拒绝微调 R2FT

- **目标**：消除 DLM 特有的两类退化——远距离位置的高先验词占位和 token 重复
- **负例构造**：通过规则自动识别退化样本——拼接 prompt 片段、插入高频无意义词、复制相邻 token 来合成负例，无需人工标注
- **训练目标**：采用 DPO 风格的偏好学习目标，对退化 token 施加下采权（down-weighting），使模型在微调后倾向于生成多样且语义合理的内容
- **设计优势**：R2FT 不改变解码窗口机制，与卷积归一化正交，可独立使用也可叠加

### 损失函数 / 训练策略

- **基础训练**：标准掩码扩散语言模型目标（masked diffusion loss），加入卷积解码归一化层
- **后训练**：R2FT 使用 DPO 损失对预训练模型微调，正例为正常生成，负例为规则合成的退化样本
- **推理**：128 步卷积解码 + 标准 top-k/nucleus 采样

---

## 实验

### 主实验：开放式文本生成

基于 LLADA-8B 架构在 AlpacaEval 上评测，G-Eval 自动评分：

| 方法 | 步数 | G-Eval ↑ | 相对 512 步基线 |
|------|------|----------|----------------|
| LLADA（原始） | 512 | 基线 | — |
| LLADA（原始） | 128 | 明显下降 | 质量退化严重 |
| + SAR 硬分块 | 128 | 部分恢复 | 仍有块边界伪影 |
| + 卷积归一化 | 128 | 接近 512 步 | 消除窗口不匹配 |
| + R2FT | 128 | 进一步提升 | 消除退化 token |
| **+ 卷积归一化 + R2FT** | **128** | **DLM SOTA** | **匹配或超越 512 步** |

### 消融实验与分析

| 消融条件 | G-Eval 变化 | 说明 |
|----------|------------|------|
| 去掉卷积归一化 | 下降显著 | 块边界伪影回归 |
| 去掉 R2FT | 下降中等 | 高先验词和重复增加 |
| 仅用 SAR 硬分块 | 不如卷积归一化 | 时间间隔扩展未解决 |
| 步数 64 步 | 质量仍可接受 | 卷积归一化鲁棒性强 |
| 步数 256 步 | 接近 512 步满配 | 边际收益递减 |

### 关键发现

- **128 步达到 512 步质量**：约 4 倍加速，是目前 DLM 领域最优步数-质量 trade-off
- **卷积归一化是主要贡献**：单独使用即可大幅缩小步数差距，验证了时间间隔扩展理论
- **R2FT 提供互补增益**：在卷积归一化基础上进一步消除定性退化，两者叠加效果最优
- **退化模式分析**：高先验词在远离 prompt 的位置出现频率更高，R2FT 针对性地下采权这些位置

---

## 亮点

- **理论驱动的工程方案**：先严谨分析时间间隔扩展的数学根因，再针对性设计卷积归一化，避免了盲目试错
- **正交改进可叠加**：卷积归一化解决结构问题，R2FT 解决内容问题，两者独立有效且组合最优
- **无需额外标注**：R2FT 的负例完全通过规则合成，无需人工偏好数据
- **实用性强**：128 步即可获得高质量生成，显著推进 DLM 的实际部署可行性

---

## 局限性

- **评估覆盖有限**：主要在开放式文本生成任务上验证，未涵盖代码生成、数学推理、摘要等其他重要任务
- **双向性优势未充分展示**：DLM 的理论优势在于双向建模，但实验未设计专门验证此特性的场景
- **规模可扩展性**：仅在 8B 参数级别验证，更大或更小模型上的效果未知
- **与 ARM 的差距**：即便在 DLM 内部达到 SOTA，与同规模 ARM（如 LLaMA-3-8B）的绝对质量和速度差距仍然存在

---

## 相关工作

- **扩散语言模型**：MDLM、SEDD 奠定了离散扩散语言建模基础；LLADA 将 DLM 扩展到大规模预训练；Block-diffusion 提出半自回归加速但未解决时间间隔扩展
- **DLM 加速**：已有工作探索步数缩减和缓存策略，但多关注连续扩散模型（图像领域），离散 DLM 的加速研究较少
- **偏好对齐**：DPO、RLHF 等后训练方法在 ARM 上广泛应用，本文将类似思路迁移到 DLM，提出针对扩散退化模式的 R2FT
- **文本生成质量**：自回归模型通过 RLHF/DPO 解决重复和无意义输出问题，DLM 的退化模式不同（高先验占位），需要专门的解决方案

---

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 时间间隔扩展的理论分析和卷积归一化方案均为新颖贡献 |
| 实验充分度 | ⭐⭐⭐ | 核心结论验证充分，但任务类型和模型规模覆盖有限 |
| 写作质量 | ⭐⭐⭐⭐ | 理论分析严谨，动机-方法-实验逻辑链清晰 |
| 实用价值 | ⭐⭐⭐⭐ | 4 倍加速对 DLM 实用化有重要推动意义 |

**综合评分**: ⭐⭐⭐⭐ — 理论分析深入，提出的卷积归一化和 R2FT 方案实用有效，是 DLM 加速方向的重要工作。任务覆盖面可进一步拓展。

<!-- RELATED:START -->

## 相关论文

- [Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](../../ECCV2024/segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model](stead_robust_provably_secure_linguistic_steganography_with_diffusion_language_mo.md)
- [Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](../../ICML2025/segmentation/infosam_fine-tuning_the_segment_anything_model_from_an_information-theoretic_per.md)

<!-- RELATED:END -->
