---
title: >-
  [论文解读] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation
description: >-
  [CVPR 2026][图像分割][图像分割] 提出 SERA 框架，在冻结的视觉-语言骨干网络中引入两阶段轻量级 MoE 专家精炼（骨干级 SERA-Adapter + 融合级 SERA-Fusion），通过表达式引导的自适应路由实现参考图像分割中的空间一致性和边界精度提升，仅更新不到 1% 的骨干参数。
tags:
  - CVPR 2026
  - 图像分割
  - Mixture-of-Experts
  - Parameter-Efficient Tuning
  - 视觉语言
  - Expert Routing
---

# Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.12538](https://arxiv.org/abs/2603.12538)  
**代码**: 无  
**领域**: 分割  
**关键词**: Referring Image Segmentation, Mixture-of-Experts, Parameter-Efficient Tuning, Vision-Language Models, Expert Routing  

## 一句话总结

提出 SERA 框架，在冻结的视觉-语言骨干网络中引入两阶段轻量级 MoE 专家精炼（骨干级 SERA-Adapter + 融合级 SERA-Fusion），通过表达式引导的自适应路由实现参考图像分割中的空间一致性和边界精度提升，仅更新不到 1% 的骨干参数。

## 研究背景与动机

参考图像分割（RIS）需要根据自然语言表达生成像素级掩码，核心难点在于将语言与视觉内容精确对齐，同时处理空间关系、细粒度属性和目标边界。现有方法存在三个关键问题：

**统一精炼策略的局限**：大多数方法对所有参考表达式使用相同的处理路径，无法匹配不同表达式的多样推理需求（有的依赖空间布局，有的依赖外观，有的依赖上下文关系）

**冻结骨干的适应困难**：为节省计算成本而冻结预训练编码器时，视觉表示的适应能力受限，导致掩码碎片化、边界泄漏或目标选错

**MoE 引入的挑战**：直接将 MoE 路由引入 RIS 面临训练不稳定和干扰预训练表示的风险

SERA 的核心动机是：不同的参考表达式需要不同类型的推理专家，因此引入条件化的专家路由机制，在保持预训练表示优势的同时实现表达式感知的特征精炼。

## 方法详解

### 整体框架

SERA 建立在 DINOv2 视觉编码器 + CLIP 文本编码器的预训练视觉-语言框架之上。给定输入图像 $I$ 和参考表达式 $Q$，视觉编码器提取图像 token 序列，文本编码器提取全局表达式嵌入。SERA 在两个互补阶段引入修改：

- **SERA-Adapter**：插入到骨干 Transformer 选定层中，在骨干内部精炼中间视觉 token
- **SERA-Fusion**：在视觉-语言融合阶段，将空间 token 重塑为 2D 特征图后通过 MoE 精炼

### 关键设计

#### SERA-Adapter：骨干级专家精炼

**功能**：在 DINOv2 的选定 Transformer 块中插入表达式条件化的适配器，通过专家引导的精炼和跨模态注意力提升空间一致性和边界精度。

**核心思路**：将视觉 token 投影到 2D 空间网格，经多尺度卷积分支（1x1, 3x3, 5x5）丰富局部上下文后，由两个互补专家精炼：

- **边界专家**：使用可学习深度 $3 \times 3$ 卷积增强轮廓敏感响应，$\mathbf{B} = \text{ReLU}(\text{BN}(\mathbf{G} + \beta \cdot \text{DWConv}_{3\times3}(\mathbf{G})))$，其中 $\beta = 0.1$
- **空间专家**：使用深度 $3 \times 3$ 卷积加带尺度残差增强局部特征一致性，$\mathbf{S} = \phi(\text{DWConv}_{3\times3}(\mathbf{G})) + \alpha \mathbf{G}$，其中 $\alpha = 0.3$

**自适应软路由**：对空间 token 做全局平均池化得到摘要 $\mathbf{z}$，经线性投影 + softmax 得到路由权重 $[w_s, w_b] = \boldsymbol{\sigma}(\mathcal{R}(\mathbf{z}))$，专家输出加权融合：

$$\mathbf{G}_{\text{corr}} = \mathbf{G}_{\text{rich}} + \alpha w_s \mathbf{E}_s + \beta w_b \mathbf{E}_b$$

其中 $\alpha = 0.25, \beta = 0.15$ 是固定缩放系数。最后展平回 token 序列并与文本嵌入做跨模态注意力。

**设计动机**：软路由确保骨干内部的稳定残差精炼，避免稀疏路由在冻结编码器中导致的不稳定。

#### SERA-Fusion：融合级专家引导聚合

**功能**：在视觉-语言融合阶段，对中间空间特征图应用互补的专家精炼，增强掩码预测前的表示质量。

**核心思路**：设计四个互补专家捕获不同视觉线索：

1. **空间专家**：注入显式位置信息，$E_{\text{spa}}(\mathbf{X}) = \mathbf{X} + \alpha \cdot \text{Conv}_{1\times1}(\mathbf{G})$，其中 $\mathbf{G}$ 是归一化坐标网格
2. **上下文专家**：通过多头自注意力捕获长程依赖，将空间维度展平后做 self-attention + FFN + 残差连接
3. **边界专家**：使用固定 Sobel 算子提取水平/垂直梯度及幅值，$E_{\text{bnd}}(\mathbf{X}) = \mathbf{X} + \phi(\text{Conv}_{1\times1}([\mathbf{X}, \mathbf{G}_{\text{mag}}, \mathbf{G}_x + \mathbf{G}_y]))$
4. **形状专家**：结合深度模糊（低频平滑）和拉普拉斯算子（高频结构线索），促进全局结构一致性

**条件路由（Top-K 稀疏门控）**：

$$\mathbf{z} = \text{GAP}(\mathbf{X}), \quad \mathbf{r} = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{z})$$

训练时加高斯噪声鼓励路由多样性，Top-K 选择后 softmax 归一化得到稀疏权重。

**设计动机**：稀疏 Top-K 路由在融合阶段鼓励专家特化，不同于骨干级的软路由策略。两阶段使用不同路由策略是有意为之——骨干内需稳定，融合级需特化。

#### 防止专家坍塌的正则化

引入三个辅助损失（仅训练时使用）：

- **Z-loss**：惩罚路由 logit 的均方幅度，$\mathcal{L}_z = \lambda_z \frac{1}{BE} \|\mathbf{r}\|_2^2$
- **负载均衡损失**：惩罚专家使用量的变异系数，$\mathcal{L}_{\text{balance}} = \lambda_{\text{bal}} \text{CV}(\mathbf{u})^2$
- **Token 分配正则化**：稳定训练中的 token 到专家的分配

### 损失函数 / 训练策略

- 总 MoE 正则化：$\mathcal{L}_{\text{MoE}} = \mathcal{L}_{\text{logit}} + \mathcal{L}_{\text{balance}} + \mathcal{L}_{\text{token}}$
- **参数高效策略**：骨干完全冻结，仅更新 LayerNorm 和 bias 参数（不到 1% 骨干参数），加上提出的模块和任务特定分割层
- 优化器：Adam，初始学习率 $1 \times 10^{-4}$，后期衰减 0.1 倍
- 硬件：单张 NVIDIA A6000，batch size 16

## 实验关键数据

### 主实验

在 RefCOCO/RefCOCO+/G-Ref 三个标准基准上评测（mIoU）：

| 方法 | 类型 | RefCOCO val | RefCOCO+ val | G-Ref val(g) | 平均 |
|------|------|-------------|--------------|--------------|------|
| ETRIS | PET | 70.5 | 60.1 | 57.9 | 62.8 |
| DETRIS-B | PET | 76.0 | 68.9 | 65.9 | 70.4 |
| VATEX | Full FT | 78.2 | 70.0 | 69.7 | 72.8 |
| RISCLIP-B | Full FT | 75.7 | 69.2 | — | 70.6 |
| **SERA (Ours)** | **PET** | **76.5** | **70.4** | **66.6** | **71.1** |

SERA 在冻结骨干的 PET 设置下超越所有 PET 方法，并与多个全量微调方法持平或接近。在 RefCOCO+（无绝对空间术语）上提升尤为显著，说明外观驱动和上下文驱动推理受益更大。

### 消融实验

**组件消融**（RefCOCO / RefCOCO+ / G-Ref(g) mIoU）：

| 配置 | RefCOCO | RefCOCO+ | G-Ref(g) |
|------|---------|----------|----------|
| Baseline | 74.90 | 68.70 | 65.10 |
| + SERA-Adapter | 75.35 (+0.45) | 69.42 (+0.72) | 65.74 (+0.64) |
| + SERA-Adapter + SERA-Fusion | **76.50 (+1.60)** | **70.40 (+1.70)** | **66.62 (+1.52)** |

**Top-K 路由消融**（RefCOCO val mIoU / oIoU）：

| Top-K | val mIoU | val oIoU |
|-------|----------|----------|
| K=1 | 75.46 | 73.32 |
| K=2 | 76.47 (+1.01) | 74.65 (+1.33) |
| K=3 | 76.20 (+0.74) | 74.10 (+0.78) |
| K=4 | 76.50 (+1.04) | 74.74 (+1.42) |

### 关键发现

1. 两个模块提供互补增益：SERA-Adapter 主要改善骨干级特征，SERA-Fusion 在融合阶段进一步增强空间表示
2. K=1 时性能最低，增加到 K>=2 后显著提升，K=4 总体最稳定
3. 在 RefCOCO+ 上增益最大（+1.70 mIoU），表明当空间术语被移除时，外观/上下文驱动的专家精炼更关键
4. 支持零样本跨数据集泛化，表明学到的视觉-语言表示可迁移

## 亮点与洞察

- **两阶段差异化路由策略**是精妙设计：骨干内用软路由保稳定，融合级用稀疏路由促特化
- 仅更新 bias + LayerNorm 的极端参数高效策略（<1% 参数）与 MoE 专家精炼结合，是新颖的设计空间
- 专家设计有明确物理语义（空间/边界/上下文/形状），比纯黑盒 MoE 更可解释
- 正则化策略（Z-loss + 负载均衡 + token 分配）确保了稀疏路由的健康训练

## 局限与展望

- 在 G-Ref 上的提升相对小于 RefCOCO+，长描述性表达的处理仍有改进空间
- 目前仅在 DINOv2 + CLIP 框架上验证，是否能迁移到其他 VLM 骨干（如 SAM、Grounding DINO）未知
- 专家数量和类型是手工设计的（4 个），是否可以自动发现最优专家组合值得探索
- 未在更大规模或更多样的分割任务上验证泛化性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 在 RIS 中首次系统引入 MoE 专家路由，且两阶段差异化路由策略设计精妙
- **实验**: ⭐⭐⭐⭐ — 三个标准基准 + 完整消融 + 零样本泛化 + 丰富定性分析，但缺少效率分析
- **写作**: ⭐⭐⭐⭐ — 结构清晰，公式完整，图表专业，方法阐述系统
- **价值**: ⭐⭐⭐⭐ — 为 VLM 的参数高效适应提供了新的 MoE 视角，对 RIS 和密集预测任务有启发

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [\[CVPR 2026\] Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)
- [\[CVPR 2026\] Weakly-Supervised Referring Video Object Segmentation through Text Supervision](wsrvos_weakly_supervised_rvos.md)

<!-- RELATED:END -->
