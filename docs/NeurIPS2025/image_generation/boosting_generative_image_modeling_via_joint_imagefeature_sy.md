---
title: >-
  [论文解读] Boosting Generative Image Modeling via Joint Image-Feature Synthesis
description: >-
  [NeurIPS 2025 **Spotlight**][图像生成][联合图像-特征生成] 提出ReDi (Representation Diffusion)框架，在扩散模型中联合建模VAE图像latent和DINOv2语义特征——两者在同一扩散过程中从纯噪声同步去噪，仅需最小修改DiT架构即实现23倍训练收敛加速和SOTA FID，并解锁Representation Guidance推理策略。
tags:
  - "NeurIPS 2025 **Spotlight**"
  - 图像生成
  - 联合图像-特征生成
  - 扩散模型
  - DINOv2
  - Representation Guidance
  - DiT
---

# Boosting Generative Image Modeling via Joint Image-Feature Synthesis

**会议**: NeurIPS 2025 **Spotlight**  
**arXiv**: [2504.16064](https://arxiv.org/abs/2504.16064)  
**代码**: [GitHub](https://representationdiffusion.github.io/)  
**领域**: 图像生成  
**关键词**: 联合图像-特征生成, diffusion model, DINOv2, Representation Guidance, DiT

## 一句话总结

提出ReDi (Representation Diffusion)框架，在扩散模型中联合建模VAE图像latent和DINOv2语义特征——两者在同一扩散过程中从纯噪声同步去噪，仅需最小修改DiT架构即实现23倍训练收敛加速和SOTA FID，并解锁Representation Guidance推理策略。

## 研究背景与动机

**领域现状**：潜在扩散模型(LDM)是高质量图像生成的主流方法，而自监督表征学习(如DINOv2)在语义理解上表现卓越。两者各有所长但长期分离——LDM的内部特征缺乏语义，DINOv2没有生成能力。

**现有痛点**：REPA (Yu et al., 2025)首次证明将扩散模型内部表征与DINOv2对齐可同时提升生成质量和训练效率。但REPA需要额外的蒸馏损失(对比/MSE损失)来对齐中间特征，训练目标复杂。

**核心矛盾**：如何优雅地在一个模型中同时做到高质量图像生成和语义表征学习，且不引入复杂的蒸馏机制？

**本文目标** 提出比REPA更直接的方案——不是对齐表征，而是让扩散模型直接联合生成图像和语义特征。

**切入角度**：将DINOv2语义特征视为与VAE latent并列的"第二模态"，在同一扩散过程中联合去噪。

**核心 idea**：与其间接对齐扩散模型的内部表征，不如直接让它学习生成语义特征——联合建模迫使模型在生成过程中自然融合低级视觉和高级语义信息。

## 方法详解

### 整体框架

给定输入图像I，同时提取VAE latent $\mathbf{x}_0 = \mathcal{E}_x(I) \in \mathbb{R}^{L \times C_x}$和DINOv2特征$\mathbf{z}_0 = \mathcal{E}_z(I) \in \mathbb{R}^{L \times C_z}$。对两者施加相同的前向扩散噪声，然后用同一个Transformer联合去噪。训练目标是简单的联合去噪损失，推理时从纯噪声同时生成图像和语义特征。

### 关键设计

1. **联合前向-反向扩散过程**:
    - 功能：对图像latent和语义特征使用相同噪声调度分别加噪，然后联合去噪
    - 核心思路：前向过程 $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}_x$，$\mathbf{z}_t = \sqrt{\bar\alpha_t}\mathbf{z}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}_z$。模型同时预测两组噪声：$\boldsymbol{\epsilon}_\theta^x$和$\boldsymbol{\epsilon}_\theta^z$。联合损失：$\mathcal{L} = \|\boldsymbol{\epsilon}_\theta^x - \boldsymbol{\epsilon}_x\|^2 + \lambda_z\|\boldsymbol{\epsilon}_\theta^z - \boldsymbol{\epsilon}_z\|^2$，默认$\lambda_z=1$
    - 设计动机：让模型被迫学习图像细节和语义结构的联合分布，两者相互提供互补信息，自然产生更好的生成

2. **Token融合策略（Merged vs Separate）**:
    - 功能：将VAE token和语义token输入Transformer的两种方式
    - 核心思路：**Merged**方式将两组token通过各自的线性投影后逐通道相加 $\mathbf{h}_t = \mathbf{x}_t\mathbf{W}_{emb}^x + \mathbf{z}_t\mathbf{W}_{emb}^z$，保持token数不变（$L$个）；**Separate**方式沿序列维度拼接得到$2L$个token。默认使用Merged以保持计算效率
    - 设计动机：Merged通过早期融合让两种信息密切交互，且不增加计算量；Separate提供更强表达力但计算量翻倍

3. **PCA降维语义表征 + Representation Guidance**:
    - 功能：对DINOv2的768维特征用PCA降到8维以平衡计算；推理时利用语义分支引导图像生成
    - 核心思路：PCA降维解决$C_z \gg C_x$导致的容量分配失衡。Representation Guidance类比CFG的思路：$\hat{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + w_r(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{z}_t, t) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))$，训练时以概率$p_{drop}$随机丢弃$\mathbf{z}_t$来同时学习有/无语义条件的去噪
    - 设计动机：PCA避免高维语义特征占据过多模型容量；Representation Guidance利用模型自身学到的语义来引导生成，无需额外模型

### 损失函数 / 训练策略

联合去噪损失 $\mathcal{L}_{joint} = \|\boldsymbol{\epsilon}_\theta^x - \boldsymbol{\epsilon}_x\|^2 + \lambda_z\|\boldsymbol{\epsilon}_\theta^z - \boldsymbol{\epsilon}_z\|^2$（$\lambda_z=1$）。训练时以概率$p_{drop}=0.2$将$\mathbf{z}_t$置零并禁用语义损失。使用SD-VAE-FT-EMA编码图像（$32\times32\times4$），DINOv2-B+Registers提取语义。PCA投影矩阵在76,800张ImageNet随机样本上预计算。

## 实验关键数据

### 主实验

| 模型 | 方法 | 迭代数 | FID↓ | 说明 |
|------|------|--------|------|------|
| DiT-XL/2 | Baseline | 7M | 9.6 | 原始DiT收敛值 |
| DiT-XL/2 | REPA | 400K | 12.3 | 蒸馏对齐方法 |
| DiT-XL/2 | **ReDi** | **400K** | **8.7** | 联合建模，超越7M步baseline |
| SiT-XL/2 | Baseline | 7M | 8.3 | 原始SiT收敛值 |
| SiT-XL/2 | REPA | 4M | 5.9 | 需要10倍迭代 |
| SiT-XL/2 | **ReDi** | **700K** | **5.6** | 6倍更快收敛 |
| SiT-XL/2+CFG | **ReDi** | 350 epochs | **1.72** | SOTA无条件扩散 |
| SiT-XL/2+CFG | **ReDi** | 800 epochs | **1.61** | 当前最佳 |

### 消融实验

| 配置 | FID↓ | 说明 |
|------|------|------|
| Merged tokens (默认) | 8.7 | 高效且效果好 |
| Separate tokens | 8.2 | 更强但计算翻倍 |
| 无PCA (768维) | 性能下降 | 容量分配失衡 |
| PCA到8维 (默认) | 8.7 | 最佳平衡 |
| $\lambda_z=0$ (无语义损失) | ~DiT基线 | 语义分支必要 |
| ReDi + REPA | 3.3 (4M iters) | 两者互补 |

### 关键发现
- ReDi加速DiT-XL/2和SiT-XL/2收敛约**23倍**
- 相比REPA，ReDi收敛**6倍**更快且FID更优
- ReDi和REPA互补——结合后在1M步达FID 3.6（REPA需4M步达5.9）
- Representation Guidance在不使用外部classifier的情况下提升生成质量

## 亮点与洞察
- Spotlight级别的优雅设计——最小化架构修改即获巨大收益
- 联合建模vs蒸馏对齐的范式之争：直接建模联合分布比间接对齐更有效
- Representation Guidance是全新的自包含推理策略，与CFG互补
- 两种方法互补的发现暗示联合建模和对齐捕获了不同的信息

## 局限与展望
- PCA降维是线性的，可能丢失非线性语义结构
- 语义编码器(DINOv2)是冻结的，联合端到端微调可能更强
- 仅在ImageNet 256×256上验证，更高分辨率和text-to-image场景未探索
- Separate tokens方式计算量翻倍，需要更高效的注意力机制

## 相关工作与启发
- **vs REPA**: REPA通过蒸馏对齐中间特征，需额外损失且效果较弱；ReDi直接联合建模，更简洁有效
- **vs MT-Diffusion**: MT-Diffusion也引入CLIP表征但未量化对生成的影响；ReDi系统评估了联合建模的好处
- **vs VideoJam**: 启发了Representation Guidance——类似于VideoJam中motion引导的思路

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 联合图像-特征扩散是全新范式，Representation Guidance创新
- 实验充分度: ⭐⭐⭐⭐ 多尺度模型、多框架、完整消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学形式化规范
- 价值: ⭐⭐⭐⭐⭐ Spotlight当之无愧，开启表征感知生成的新方向

<!-- RELATED:START -->

## 相关论文

- [DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](../../ICML2025/image_generation/dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [IRGen: Generative Modeling for Image Retrieval](../../ECCV2024/image_generation/irgen_generative_modeling_for_image_retrieval.md)
- [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](../../CVPR2025/image_generation/pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)
- [Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](../../ICCV2025/image_generation/mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [Image Super-Resolution with Guarantees via Conformalized Generative Models](image_super-resolution_with_guarantees_via_conformalized_generative_models.md)

<!-- RELATED:END -->
