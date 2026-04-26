---
title: >-
  [论文解读] OpenSDI: Spotting Diffusion-Generated Images in the Open World
description: >-
  [CVPR 2025][图像生成][AI生成图像检测] OpenSDI 定义了开放世界扩散图像检测挑战，构建了包含多 VLM 生成指令和多扩散模型的大规模数据集 OpenSDID，并提出 MaskCLIP——通过 Synergizing Pretrained Models（SPM）框架协同 CLIP 和 MAE，在检测和定位任务上大幅超越现有方法。
tags:
  - CVPR 2025
  - 图像生成
  - AI生成图像检测
  - 扩散图像定位
  - 开放世界
  - 基础模型协同
  - CLIP+MAE
---

# OpenSDI: Spotting Diffusion-Generated Images in the Open World

**会议**: CVPR 2025  
**arXiv**: [2503.19653](https://arxiv.org/abs/2503.19653)  
**代码**: https://github.com/iamwangyabin/OpenSDI  
**领域**: 扩散模型 / AI安全  
**关键词**: AI生成图像检测, 扩散图像定位, 开放世界, 基础模型协同, CLIP+MAE

## 一句话总结

OpenSDI 定义了开放世界扩散图像检测挑战，构建了包含多 VLM 生成指令和多扩散模型的大规模数据集 OpenSDID，并提出 MaskCLIP——通过 Synergizing Pretrained Models（SPM）框架协同 CLIP 和 MAE，在检测和定位任务上大幅超越现有方法。

## 研究背景与动机

**领域现状**：随着 Stable Diffusion、FLUX 等先进扩散模型的普及，AI 生成内容的真实性越来越高，区分真实与生成图像成为关键挑战。现有检测方法主要面向传统篡改（如拼接、复制移动）或特定生成器。

**现有痛点**：现有方法和数据集无法应对开放世界场景的三个核心维度：（1）用户多样性——不同用户的风格和意图千差万别；（2）模型创新——扩散模型快速迭代，新模型不断出现；（3）篡改范围——从全局合成到局部编辑的完整谱系。现有数据集通常只覆盖 1-2 个维度。

**核心矛盾**：检测和定位是异质任务——检测需要图像级语义判断，定位需要像素级精确分割；现有方法通常只擅长其一。此外，对特定生成器的过拟合严重限制了泛化能力。

**本文目标**：定义 OpenSDI 挑战，构建全面的基准数据集，提出统一检测和定位的泛化方案。

**切入角度**：利用大规模 VLM 模拟真实用户行为生成多样化篡改指令，通过多个预训练基础模型的协同而非单独训练来保持泛化能力。

**核心 idea**：用"协同预训练模型"（SPM）的策略——通过 prompting 和 attending 机制让 CLIP（擅长语义判断）和 MAE（擅长空间重建）协同工作，同时完成检测和定位，保留各自预训练的泛化能力。

## 方法详解

### 整体框架

MaskCLIP 由三个核心组件构成：（1）CLIP 视觉编码器提取语义特征，（2）CLIP 文本编码器提供"real/fake"类别嵌入，（3）MAE 编码器提取空间重建特征。通过 VCA、TVCA、VSA 三种注意力模块实现协同。基于 FPN 解码器生成像素级预测。检测输出来自 CLIP 全局特征与文本嵌入的余弦相似度，定位输出来自 MAE+TVCA 生成的分割图。

### 关键设计

1. **Prompt-Tuning 保留 CLIP 泛化能力**:

    - 功能：学习 real/fake 概念的连续 prompt 向量，避免修改 CLIP 预训练权重
    - 核心思路：学习一对可学习 prompt $\mathbf{V}_c \in \mathbb{R}^{M \times D}$（$c \in \{\text{real}, \text{fake}\}$），经文本编码器生成类别嵌入 $\mathbf{t}_{\text{real}}$, $\mathbf{t}_{\text{fake}}$。类比手工 prompt "a photo of a [real/fake]"，但用可学习连续向量更好地捕获语义。CLIP 参数完全冻结
    - 设计动机：全参数微调会破坏 CLIP 的广泛视觉-语言知识，prompt-tuning 以极少参数保留泛化性

2. **Visual Cross-Attention (VCA) 对齐 CLIP 和 MAE**:

    - 功能：将 CLIP 的语义理解注入 MAE 的空间特征中
    - 核心思路：在多个层上部署——将 CLIP patch tokens 通过双线性插值 + 1×1 卷积调整为 MAE 特征维度，作为 query。MAE tokens 作为 key/value，执行交叉注意力 $\mathbf{G}^l = \text{Softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}})\mathbf{V}$，结果通过残差连接更新 MAE 特征 $V_m^{l+1} = V_m^l + \mathbf{G}^l$
    - 设计动机：CLIP 善于捕获"假不假"的语义判断，MAE 善于空间重建发现局部异常，VCA 让两者优势互补

3. **TVCA + VSA 完成检测与定位双任务**:

    - 功能：TVCA 将文本语义注入解码特征用于定位；VSA 融合多层 CLS token 用于检测
    - 核心思路：TVCA 用类别文本嵌入作为 query、FPN 解码特征作为 key/value 做交叉注意力，生成分割 logits $M_{\text{fake}}$。VSA 收集 CLIP 多层 CLS token 嵌入 $V_{cls} = \{v_{cls}^l\}_{l \in L}$，通过自注意力 + 全局池化 + 线性投影得到全局表示 $\mathbf{g}$，与文本嵌入做余弦相似度得检测结果
    - 设计动机：不同层的 CLS token 编码不同粒度信息——低层关注低级伪影，高层关注语义一致性；TVCA 将文本语义约束引入像素级预测

### 损失函数 / 训练策略

总损失 = $\mathcal{L}_{CE}$（检测交叉熵）+ $\mathcal{L}_{BCE}$（定位二元交叉熵）+ $\mathcal{L}_{EDG}$（边缘加权损失），权重均为 1。CLIP 完全冻结，只训练 VCA/TVCA/VSA/FPN/prompt 向量和 MAE 编码器。

## 实验关键数据

### 主实验（OpenSDID 跨域测试）

| 方法 | SD1.5 IoU | SDXL IoU | SD3 IoU | Flux.1 IoU | 平均 IoU |
|------|----------|---------|--------|-----------|---------|
| CAT-Net | 0.664 | 0.255 | 0.356 | 0.050 | 0.374 |
| TruFor | 0.634 | 0.266 | 0.323 | 0.076 | 0.369 |
| IML-ViT | 0.665 | 0.215 | 0.236 | 0.061 | 0.325 |
| **MaskCLIP** | **0.671** | **0.310** | **0.438** | **0.162** | **0.427** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| 无 VCA (CLIP+MAE 简单拼接) | 定位 IoU 下降约 7%，跨域泛化退化 |
| 无 TVCA | 失去文本语义引导，定位精度下降 |
| 无 VSA（仅用最后层 CLS） | 检测准确率下降，低层伪影信息丢失 |
| Full fine-tune CLIP (vs prompt-tune) | 泛化性显著变差，过拟合训练集 |

### 关键发现

- **MaskCLIP 在跨域泛化上优势巨大**：在最新的 Flux.1 模型上，IoU 几乎是第二名的两倍（0.162 vs 0.082），说明 SPM 框架有效保留了泛化能力
- 越新的扩散模型越难检测：Flux.1 的 IoU 仅 0.16，远低于 SD1.5 的 0.67
- 定位比检测更具挑战性：全局检测准确率可达 70%+，但像素级定位 IoU 通常低于 50%
- 定位任务相对提升 14.23%（IoU）和 14.11%（F1），检测任务提升 2.05%（Acc）和 2.38%（F1）

## 亮点与洞察

- **OpenSDI 挑战的定义**系统性地梳理了开放世界扩散图像检测的三个维度（用户多样性、模型创新、篡改范围），为后续研究提供了清晰的问题框架
- **用 VLM 模拟用户行为**生成篡改指令是非常聪明的数据构建策略——比人工标注更可扩展，比模板生成更自然多样
- **SPM 框架的"协同而非替代"哲学**——保留每个基础模型的预训练知识不被破坏，通过轻量模块实现协同。这种设计模式可以推广到其他需要多模型协作的任务

## 局限与展望

- 对最新扩散模型（Flux.1）的定位能力仍然很弱（IoU 仅 0.16），说明开放世界检测远未解决
- 数据集中训练集仅用 SD1.5 生成，限制了模型对新模型的适应能力
- MaskCLIP 需要同时运行 CLIP 和 MAE 两个编码器，推理成本较高
- 未来可以探索在线学习或少样本适应策略，对新出现的扩散模型快速适配
- 对 GAN 生成图像的泛化性未充分验证（主要关注扩散模型）

## 相关工作与启发

- **vs TruFor**: TruFor 融合空间、频率和噪声域特征做定位，但对扩散生成图像泛化差；MaskCLIP 通过预训练模型协同实现更好泛化
- **vs CLIP-based 检测器（如 DeCLIP）**: 仅用 CLIP 做二分类，缺乏像素级定位能力；MaskCLIP 通过 MAE 补充空间精度
- **vs IML-ViT**: 基于 ImageNet 预训练的 ViT 做定位，对扩散特有伪影捕获不足；MaskCLIP 的 VCA 机制引入了 CLIP 的语义知识补充

## 评分

- 新颖性: ⭐⭐⭐⭐ OpenSDI 挑战定义有价值，SPM 框架设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模数据集构建完整，检测+定位双任务评估，跨域泛化分析详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但方法部分稍显繁复
- 价值: ⭐⭐⭐⭐⭐ AI 安全方向的重要工作，数据集和方法都有很高的社区需求

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[CVPR 2025\] AvatarArtist: Open-Domain 4D Avatarization](avatarartist_open-domain_4d_avatarization.md)
- [\[CVPR 2025\] Lifting Motion to the 3D World via 2D Diffusion](lifting_motion_to_the_3d_world_via_2d_diffusion.md)
- [\[CVPR 2025\] MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](mirrorverse_pushing_diffusion_models_to_realistically_reflect_the_world.md)
- [\[AAAI 2026\] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images](../../AAAI2026/image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)

<!-- RELATED:END -->
