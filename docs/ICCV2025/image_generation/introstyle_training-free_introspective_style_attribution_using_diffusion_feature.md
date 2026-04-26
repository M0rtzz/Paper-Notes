---
title: >-
  [论文解读] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features
description: >-
  [ICCV 2025][图像生成][风格归因] 提出 IntroStyle，一种无需训练的风格归因方法，仅利用扩散模型自身中间层特征的通道级均值和方差统计量，通过 2-Wasserstein 距离度量图像间的风格相似性，在 WikiArt 和 DomainNet 上大幅超越需要专门训练的 SOTA 方法。
tags:
  - ICCV 2025
  - 图像生成
  - 风格归因
  - 扩散模型特征
  - 免训练
  - Wasserstein距离
  - 版权保护
  - 风格检索
---

# IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features

**会议**: ICCV 2025  
**arXiv**: [2412.14432](https://arxiv.org/abs/2412.14432)  
**代码**: [GitHub](https://anandk27.github.io/IntroStyle)  
**领域**: 图像生成 / 风格归因  
**关键词**: 风格归因, 扩散模型特征, 免训练, Wasserstein距离, 版权保护, 风格检索

## 一句话总结

提出 IntroStyle，一种无需训练的风格归因方法，仅利用扩散模型自身中间层特征的通道级均值和方差统计量，通过 2-Wasserstein 距离度量图像间的风格相似性，在 WikiArt 和 DomainNet 上大幅超越需要专门训练的 SOTA 方法。

## 研究背景与动机

- **核心问题**：T2I 扩散模型（如 Stable Diffusion、DALL-E 3）在大规模网络数据上训练，可能复制受版权保护的艺术家风格。如何在不重训模型、不引入外部模块的情况下，判断生成图像是否模仿了特定艺术风格？
- **现有方法的不足**：
    - 风格"遗忘"（unlearning）需要昂贵的模型重训，且无法完全阻止通过替代 prompt 间接复制
    - 风格"伪装"（cloaking）会影响真实观赏体验，负担落在创作者身上
    - 现有归因方法（如 CSD、GDA）需要训练外部模型（CLIP/DINO 微调），计算量大、部署复杂
    - 语义混淆问题：现有方法倾向于检索语义相似而非风格相似的图像
- **关键洞察**：扩散模型 UNet 的不同层会自然地解耦图像的结构、颜色、纹理等属性，其内部特征已足够用于风格归因——无需任何外部模块或额外训练
- **数据集缺失**：现有评估数据集（WikiArt）缺乏风格与语义的细粒度分离，难以评估方法是否真正区分了"风格"与"内容"

## 方法详解

### 整体框架

IntroStyle 将扩散模型的去噪网络视为自编码器，从其中间特征层提取风格描述符，再用概率距离度量风格相似性。整个过程为前向推理，无需任何训练或微调。

**三步流程**：
1. 图像 $I$ 经 VAE 编码为潜向量 $z_0 = \mathcal{E}(I)$，按公式加噪至时间步 $t$：$z_t = \sqrt{\bar{\alpha_t}} z_0 + \sqrt{1-\bar{\alpha_t}} \epsilon_t$
2. 将 $z_t$ 送入去噪网络 $\epsilon_\theta$（使用空文本嵌入 $c=\varnothing$），从指定上采样块 $idx$ 提取特征张量 $F^{t,idx}$
3. 计算特征张量的通道级统计量作为风格描述符

### 关键设计：IntroStyle 特征

受 StyleGAN 中 AdaIN 思想启发——风格信息主要编码在特征的一阶（均值）和二阶（方差）统计量中。对特征张量 $F^{t,idx}$ 的第 $c$ 个通道：

$$\mu_c = \frac{1}{WH} \sum_{i,j} F^{t,idx}_{c,i,j}, \quad \sigma_c^2 = \frac{1}{WH} \sum_{i,j} (F^{t,idx}_{c,i,j} - \mu_c)^2$$

风格特征向量：$f^{t,idx}(I) = (\mu_1, \ldots, \mu_C, \sigma_1^2, \ldots, \sigma_C^2)^T$

其中 $t$（加噪时间步）和 $idx$（UNet 层索引）为超参数，默认取上采样块 $idx=1$。

### 相似性度量：2-Wasserstein 距离

将每幅图像的 IntroStyle 表示建模为 $C$ 维多元高斯分布（对角协方差），两图的风格距离用 $W_2$ 距离衡量：

$$W_2^2 = \|\mu_1 - \mu_2\|_2^2 + \text{tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})$$

对角协方差假设下计算高效。与 L2 距离、Gram 矩阵、JSD 等替代度量对比后，$W_2$ 效果最佳。

### ArtSplit 合成数据集

为精确评估风格-语义解耦能力，构造了 Artistic Style Split（ArtSplit）数据集：
- 对每幅真实绘画，设计两类 prompt："语义 prompt"（去除风格元素）和"风格 prompt"（去除语义内容）
- 用扩散模型生成覆盖所有"风格 x 语义"组合的合成图像
- 可精确量化方法对风格 vs 语义的敏感度

## 实验关键数据

### 主实验：风格检索性能

| 方法 | WikiArt mAP@1 | WikiArt mAP@10 | WikiArt Recall@100 | DomainNet mAP@1 |
|------|:---:|:---:|:---:|:---:|
| VGG-Net Gram | 0.259 | 0.194 | 0.804 | - |
| CSD (SOTA) | 需外部训练 | 需外部训练 | 需外部训练 | 需外部训练 |
| **IntroStyle** | **大幅领先** | **大幅领先** | **大幅领先** | **大幅领先** |

- IntroStyle 在 WikiArt 和 DomainNet 两个数据集上均大幅超越所有基线（CSD、GDA 等），且无需任何训练
- 定性结果显示：CSD/GDA 检索结果受语义偏差影响严重，IntroStyle 则准确聚焦风格

### 消融实验

| 消融项 | 结论 |
|--------|------|
| UNet 层选择 | 上采样块 1（$idx=1$）效果最佳，编码器层偏语义 |
| 时间步 $t$ | 中等加噪水平效果最优，过高/过低均降低性能 |
| 距离度量 | $W_2$ > L2 > Gram > JSD |
| 文本条件 | 空文本（$c=\varnothing$）优于使用实际 prompt |

### 关键发现

- IntroStyle 在 ArtSplit 数据集上的"风格 vs 语义"消歧实验中展现出极强的风格聚焦能力
- 扩散模型的上采样层天然分离风格与内容——验证了"内省式归因"的可行性
- 可直接用于风格拒绝采样（style-based rejection sampling），阻止模型生成特定风格

## 亮点与洞察

1. **零训练开销**：完全免训练方法，不需要任何额外数据集、微调或外部模型，仅利用扩散模型自身特征
2. **强大的风格-语义解耦**：通过选择合适的 UNet 层和通道统计量，天然避免了语义混淆——这是现有训练方法一直难以解决的问题
3. **理论优雅**：将 AdaIN 的风格表示思想（均值+方差即风格）与概率分布距离（$W_2$）自然结合，形成闭合的理论框架
4. **ArtSplit 数据集**：提供了第一个能精确评估风格归因方法的"风格x语义"完全交叉数据集
5. **实用价值**：直接可用于版权保护——检测生成图像是否抄袭特定艺术家风格

## 局限性

- 依赖特定扩散模型（Stable Diffusion UNet 架构），对 DiT 等新架构的泛化性未验证
- ArtSplit 数据集基于扩散模型生成的合成图像，可能引入分布偏差
- "风格"的定义本身主观模糊——社会建构式定义（按艺术家/流派）不一定覆盖所有视觉风格维度
- 加噪时间步 $t$ 和层索引 $idx$ 为需手动选择的超参数
- 对非西方艺术传统或数字艺术风格的适用性有待验证

## 相关工作

- **风格迁移**：Gram 矩阵 (Gatys et al.)、AdaIN (Huang and Belongie)、StyleGAN 的 AdaIN 层控制
- **风格感知 T2I 模型**：Textual Inversion、DreamBooth 等个性化生成
- **数据归因**：CSD (对比学习风格描述符)、GDA (生成式数据归因)、基于遗忘的归因
- **扩散特征利用**：DDAE (潜表示分类)、REPA (表示对齐)、零样本对应/分割

## 评分

| 维度 | 分数 (1-5) |
|------|:---:|
| 创新性 | 4 |
| 理论深度 | 3.5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4.5 |
| **总评** | **4.0** |

核心贡献在于发现扩散模型中间特征的通道统计量就是优秀的风格描述符，方法极简但效果显著。ArtSplit 数据集填补了风格归因评估的空白。美中不足是方法绑定 UNet 架构，且"内省"能力的理论解释不够深入。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [\[AAAI 2026\] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](../../AAAI2026/image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](../../CVPR2025/image_generation/k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] MosaicDiff: Training-free Structural Pruning for Diffusion Model Acceleration Reflecting Pretraining Dynamics](mosaicdiff_training-free_structural_pruning_for_diffusion_model_acceleration_ref.md)

<!-- RELATED:END -->
