---
title: >-
  [论文解读] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling
description: >-
  [CVPR 2025][图像生成][多模态] 首次将连续图像表示与离散文本表示整合到统一自回归概率建模框架中，通过轻量扩散头替代 VQ 离散化避免信息损失，并推导出 v-prediction 为最优参数化以解决低精度训练下的数值误差问题。
tags:
  - CVPR 2025
  - 图像生成
  - 多模态
  - continuous image tokens
  - 扩散模型
  - v-prediction
  - joint probability modeling
  - lossless compression
---

# MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling

**会议**: CVPR 2025  
**arXiv**: [2410.10798](https://arxiv.org/abs/2410.10798)  
**代码**: [GitHub](https://github.com/ydcUstc/MMAR)  
**领域**: 图像生成  
**关键词**: multi-modal auto-regressive, continuous image tokens, diffusion head, v-prediction, joint probability modeling, lossless compression

## 一句话总结

首次将连续图像表示与离散文本表示整合到统一自回归概率建模框架中，通过轻量扩散头替代 VQ 离散化避免信息损失，并推导出 v-prediction 为最优参数化以解决低精度训练下的数值误差问题。

## 研究背景与动机

**领域现状**: 多模态大模型同时支持图像理解和生成是热门方向。现有联合概率建模方法分三类：
1. **离散 AR**（Chameleon, EMU3）：VQ-VAE 离散化图像 → codebook 大小限制每 token 的信息容量，导致需要大量 token（如 EMU3 需 4096 tokens/image）
2. **离散扩散**（Show-o）：离散 token + 扩散建模
3. **连续扩散**（Transfusion, MonoFormer）：连续表示 + DiT 扩散建模

**核心矛盾**:
- **离散方法**: codebook 大小瓶颈导致信息损失，理解性能落后于使用连续 CLIP 表示的 LLaVA
- **连续扩散方法（Transfusion）**: 扩散模型将图像信息编码在不同噪声级别中，输入干净图像只能调用与低噪声相关的"图像增强"级表示。作者实验发现，不同理解任务在不同非零噪声级别达到最优（如物体定位任务在高噪声时更好），说明干净输入时无法充分利用学到的图像建模能力

**本文切入点**: 提出连续 AR 方法——直接对连续图像 token 进行自回归建模，通过轻量扩散头为每个图像 patch 采样连续表示。其自回归本质保证所有图像隐藏表示同时为生成和理解优化。

## 方法详解

### 整体框架

MMAR 在预训练 decoder-only LLM 的基础上扩展：

- **文本部分**: 标准自回归，softmax + cross-entropy loss
- **图像部分**: Masked AR（双向编码器-解码器），每个 patch 位置使用轻量 Diffusion MLP 头从条件表示 $z_i = f_\theta(x_{<i})$ 采样连续图像 token
- **统一建模**: [text][img] → P(I|T)；[img][text] → P(T|I)；[img] → P(I)

### 关键设计

#### 1. 连续 Token + 扩散头

对图像 token $x_i$，在 LLM backbone 输出 $z_i$ 条件下，使用轻量 MLP 扩散模型预测：

$$L_i = \mathbb{E}_{x_i, \epsilon, t}[w_t \cdot \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_i + \sqrt{1-\bar\alpha_t}\epsilon, t, z_i)\|^2]$$

关键优势：
- **无信息损失**: 连续 tokenizer 可将 128×128 patch 压缩为单个连续 token（仅需 16 tokens/512×512 image，对比 EMU3 的 4096）
- **扩散与 backbone 解耦**: 扩散过程仅在每个 patch 的 MLP 头中进行，backbone 不受噪声级别影响，所有 hidden representations 对理解任务完全可用

#### 2. v-prediction 最优参数化

从最小化数值误差的第一性原理推导出 v-prediction 是低精度训练下的最优参数化：

**问题分析**: bfloat16 训练中，浮点表示的数值误差与数值大小成正比。在 $\epsilon$-prediction 下，DDIM 采样的误差项为：

$$|\sqrt{1-\bar\alpha_{t-1}} - \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}\sqrt{1-\bar\alpha_t}| \cdot \delta \sigma_t$$

当 SNR 极低（$\bar\alpha_t \to 0$）时误差趋近无穷大。

**解决方案**: 使用 v-prediction 参数化 $v_i^{(t)} = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_i$，理论证明可最小化各噪声级别的数值误差。

#### 3. 两阶段训练策略

- **Stage 1（Image Expert Pretraining）**: 大规模中等质量数据（Capfusion-120M），mask ratio [0.7, 1.0]，学习多样化视觉理解
- **Stage 2（Image Expert Fine-tuning）**: 小规模高质量数据（CC12M + LAION-aesthetics），mask ratio 范围扩展到 [0, 1.0]

**发现的问题**: Stage 1 末尾生成的图像有"空洞"——因为 mask ratio 下界为 0.7，模型未充分学习高补全率（最后 30%）场景。Stage 2 将 mask ratio 下界调到 0 解决此问题。

### 损失函数

$$L = \sum_{i \in I_{img}} L_i - \sum_{i \in I_{txt}} \log p_\theta(x_i | x_{<i})$$

图像部分用 v-prediction diffusion loss，文本部分用标准 cross-entropy。

## 实验关键数据

### 主实验表（视觉理解 18 benchmarks 平均）

| 方法 | LLM | V-Token | AVE@18Und. |
|------|-----|---------|------------|
| Chameleon-7B | 7B scratch | vq-vae | 18.34 |
| Transfusion* | Qwen2-0.5B | vae | 28.26 |
| Show-o | Phi-1.5B | CLIP | 33.06 |
| VILA-U | LLaMA-2-7B | vq-vae | — |
| **MMAR-0.5B** | **Qwen2-0.5B** | **vae** | **34.56** |
| **MMAR-7B** | **Qwen2-7B** | **vae** | **48.25** |
| LLaVA-1.5 (理解only) | Vicuna-7B | CLIP | 47.08 |

MMAR-7B（无 CLIP，仅 256×256）几乎追平使用预训练 CLIP 的 LLaVA-1.5。

### 消融表

| 设置 | MMB | MMEP | AVE@18Und. | FID-30K ↓ |
|------|-----|------|------------|-----------|
| MMAR-0.5B (v-pred) | 48.45 | 882.1 | 34.56 | 36.6 |
| w/ ε-prediction | 45.53 | 880.7 | 32.21 | 61.53 |
| Show-o like (w/VQ) | 37.54 | 618.2 | 29.70 | 66.26 |
| Transfusion-like | 29.47 | 594.3 | 28.26 | 95.38 |

### 关键发现

1. **v-prediction vs ε-prediction**: v-prediction 在理解（+2.35 AVE）和生成（FID 36.6 vs 61.53）上全面优于 ε-prediction
2. **连续 vs 离散**: MMAR（连续）大幅优于 Show-o-like（VQ 离散），验证了离散化的信息损失
3. **MMAR vs Transfusion**: 同等规模下 MMAR 在理解和生成上全面超越 Transfusion-like 方法
4. **仅 16 tokens/image**: 在极少 token 数下实现了与 4096 tokens 的 EMU3 可比的性能
5. **Scale 友好**: 从 0.5B 到 7B 模型性能显著提升，表明方法可随模型/数据规模扩展

## 亮点与洞察

1. **理论贡献突出**: 从数值误差最小化的第一性原理推导出 v-prediction 的最优性，这是对扩散模型理论的新理解
2. **统一框架极简**: 扩散头仅是 MLP，将扩散过程从 backbone 解耦，设计简洁高效
3. **Transfusion 局限性的深入分析**: 通过实验揭示了连续扩散方法在理解任务上的根本缺陷（不同噪声级别编码不同信息）
4. **实用意义大**: 仅 16 个连续 token 即可表示 512×512 图像，对推理效率有重要意义

## 局限性

1. **生成质量仍有差距**: FID 指标未达到专用生成模型水平，需额外高质量数据训练
2. **仅验证 256×256 分辨率**: 未探索更高分辨率下的表现
3. **Masked AR 的局限**: 图像部分使用 masked AR 而非 causal AR，与文本 AR 不完全一致
4. **EmbeddingViT 模块**: 引入额外的编码器模块增加了整体参数量
5. **两阶段训练增加复杂性**: 不同 mask ratio 和数据配比需要仔细调整

## 相关工作与启发

- **MAR**: MMAR 的扩散头直接受 MAR 启发，但将其从 ImageNet 类别生成推广到 LLM + 大规模图文数据
- **Transfusion**: MMAR 的直接竞争者，通过实验证明了 Transfusion 的信息利用不完整问题
- **Chameleon / EMU3**: 离散 AR 的代表，MMAR 用连续 token 证明了更高效的替代方案
- **启发**: 「扩散过程与 backbone 解耦」的思路可推广到任何需要在 AR 框架中建模连续分布的场景

## 评分 ⭐

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)

</div>

<!-- RELATED:END -->
