---
title: >-
  [论文解读] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation
description: >-
  [CVPR 2025][图像生成][视频Tokenizer] 本文提出Divot，一种利用扩散过程进行自监督视频表征学习的连续视频Tokenizer，通过让扩散模型以Tokenizer特征为条件进行去噪来训练表征，并用高斯混合模型（GMM）建模LLM输出的连续视频特征分布，实现了视频理解与生成的统一框架。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频Tokenizer"
  - "扩散模型"
  - "视频理解与生成统一"
  - "高斯混合模型"
  - "大语言模型"
---

# Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation

**会议**: CVPR 2025  
**arXiv**: [2412.04432](https://arxiv.org/abs/2412.04432)  
**代码**: [GitHub](https://github.com/TencentARC/Divot)  
**领域**: 图像生成  
**关键词**: 视频Tokenizer, 扩散模型, 视频理解与生成统一, 高斯混合模型, 大语言模型

## 一句话总结
本文提出Divot，一种利用扩散过程进行自监督视频表征学习的连续视频Tokenizer，通过让扩散模型以Tokenizer特征为条件进行去噪来训练表征，并用高斯混合模型（GMM）建模LLM输出的连续视频特征分布，实现了视频理解与生成的统一框架。

## 研究背景与动机

**领域现状**：多模态大语言模型在图像理解与生成的统一上取得了显著进展，但视频领域的统一相对滞后。近期先驱工作（如LWM、VILA-U）采用离散视频Tokenizer映射为token序列，便于LLM的自回归生成。

**现有痛点**：离散视频token虽便于生成（next-token prediction），但会显著降低多模态理解性能——连续表征更适合理解任务。然而连续表征难以用LLM建模生成，简单的MSE回归会导致LLM学到过度平均的特征，生成的视频呈现重复模式。

**核心矛盾**：离散表征利于生成但损害理解，连续表征利于理解但难以生成。需要一种Tokenizer同时满足两个方向的需求。

**本文目标**：设计连续视频Tokenizer，同时支持LLM的视频理解（作为输入）和视频生成（作为输出条件解码）。

**切入角度**：如果扩散模型能以Tokenizer特征为条件成功去噪，说明该Tokenizer已捕获足够的时空信息；同时该扩散模型天然可作为de-tokenizer解码视频。

**核心 idea**：用扩散去噪作为代理任务训练视频Tokenizer（自监督），用GMM概率建模替代确定性回归来让LLM生成连续视频特征。

## 方法详解

### 整体框架
稀疏采样帧（2fps）输入Tokenizer获得时空表征，密集采样帧（8fps）经VAE编码后加噪，训练U-Net以Tokenizer特征为条件去噪。训练完成后U-Net即为de-tokenizer。LLM端，理解通过next-word prediction输入视频token，生成通过预测GMM参数采样视频特征再由de-tokenizer解码。

### 关键设计

1. **扩散驱动的视频Tokenizer**:

    - 功能：通过自监督学习获取捕获时空信息的连续视频表征
    - 核心思路：Tokenizer由预训练ViT + Spatial-Temporal Transformer + Perceiver Resampler组成。训练时以Tokenizer输出的64个token作为DynamiCrafter U-Net的cross-attention条件，去噪VAE latent。去噪目标迫使Tokenizer编码足够丰富的时空信息
    - 设计动机：扩散去噪要求条件信号包含细粒度的空间和时间信息才能重建视频，因此是一个天然的表征学习代理任务。Perceiver Resampler将patch级别特征压缩为固定数量的高层token，减少LLM需要预测的token数

2. **GMM概率建模视频特征**:

    - 功能：让LLM有效地建模和生成连续视频特征分布
    - 核心思路：LLM输出被训练为预测GMM参数（$2kd+k$个参数：均值、方差和混合概率），使用负对数似然（NLL）损失优化。推理时从预测的GMM分布中采样作为de-tokenizer的条件。对比了三种方案：MSE回归（过度平均）、Diffusion建模（高层特征对噪声敏感）、GMM建模（效果最好）
    - 设计动机：确定性MSE回归会使LLM学到所有可能视频的平均特征，导致重复模式。概率建模允许多样性采样，GMM比扩散建模更稳定因为高层语义特征对噪声更敏感

3. **稀疏-密集帧采样策略**:

    - 功能：在Tokenizer效率与视频重建质量间取得平衡
    - 核心思路：Tokenizer输入稀疏帧（5帧，2fps）减少token序列长度；去噪目标使用密集帧（16帧，8fps）确保时间动态的完整学习
    - 设计动机：相邻帧语义高度冗余，稀疏采样对理解足够；但生成需要密集帧的时间细节

### 损失函数 / 训练策略
Tokenizer训练：标准扩散去噪损失。LLM训练：理解用next-token prediction交叉熵，生成用GMM的NLL损失。分三阶段：Tokenizer预训练（10M视频）→ LLM预训练（视频-文本对）→ SFT（多任务）。

## 实验关键数据

### 主实验

| 模型 | LLM大小 | 视频生成 | EgoSchema | MVBench | ActivityNet |
|------|---------|---------|-----------|---------|-------------|
| Video-LLaVA | 7B | × | 38.4 | 41.0 | 45.3 |
| VideoChat2 | 7B | × | 42.2 | 51.1 | 49.1 |
| Video-LaVIT | 7B | ✓ | - | - | - |
| **Divot-LLM** | 7B | ✓ | **43.6** | **52.8** | **50.2** |

### 消融实验（视频生成，MSR-VTT）

| 特征建模方式 | FVD↓ | 相似度↑ |
|-------------|------|--------|
| MSE回归 | 较差 | 较低 |
| Diffusion建模 | 中等 | 中等 |
| **GMM建模** | **最优** | **最高** |

### 关键发现
- Divot-LLM在视频理解上与专用理解模型竞争力相当，同时新增了视频生成能力
- GMM建模显著优于MSE回归和Diffusion建模，验证了概率建模对连续特征生成的重要性
- Perceiver Resampler产生的无位置依赖的高层token比保留空间结构的patch token更容易被LLM拟合
- 模型支持视频故事讲述——交替生成叙述文本和对应视频片段

## 亮点与洞察
- "扩散即表征学习"的思路新颖——去噪目标天然要求条件包含丰富信息，且训练后的去噪网络直接可用作解码器
- GMM vs MSE vs Diffusion的对比实验很有价值：揭示了高层语义特征与底层像素/latent特征在建模策略上的根本差异
- 连续Tokenizer统一理解与生成的路线可能比离散路线更有前景

## 局限与展望
- 视频生成质量受限于代理扩散模型（DynamiCrafter）的能力
- 当前只用Mistral-7B，在更强LLM上的扩展效果未知
- 视频长度受限（2秒clip），长视频生成需要进一步研究
- GMM的混合分量数k是超参，最优选择可能因任务而异

## 相关工作与启发
- **vs VILA-U/LWM（离散token）**: 离散化损害理解精度；Divot用连续token保持理解同时实现生成
- **vs Emu3（离散VQ token生成）**: Emu3直接用LLM next-token prediction生成离散视频token；Divot通过GMM概率采样连续token+diffusion解码
- **vs MAR（图像域diffusion建模）**: MAR在VAE latent上用diffusion建模效果好；但本文发现在高层语义特征上diffusion建模不如GMM

## 评分
- 新颖性: ⭐⭐⭐⭐ 扩散作为表征学习代理+GMM建模连续token是新颖组合
- 实验充分度: ⭐⭐⭐⭐ 多个理解和生成benchmark，完整建模方式对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰，动机明确
- 价值: ⭐⭐⭐⭐ 为视频LLM的统一理解生成提供了有力方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] StyleMaster: Stylize Your Video with Artistic Generation and Translation](stylemaster_stylize_your_video_with_artistic_generation_and_translation.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [\[CVPR 2025\] SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer](softvq-vae_efficient_1-dimensional_continuous_tokenizer.md)

</div>

<!-- RELATED:END -->
