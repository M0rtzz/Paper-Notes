---
title: >-
  [论文解读] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts
description: >-
  [ECCV 2024][图像生成][多模态提示] 本文提出 MultiGen，通过为每个物体构建"增广token"（融合文本、坐标和图像信息），并训练坐标模型和特征模型来处理推理时的模态缺失，首次实现了从多物体多模态提示进行零样本图像生成，支持纯文本或任意模态组合的灵活输入。
tags:
  - ECCV 2024
  - 图像生成
  - 多模态提示
  - 零样本图像生成
  - 增广token
  - 扩散模型
  - 多物体可控生成
---

# MultiGen: Zero-Shot Image Generation from Multi-modal Prompts

**会议**: ECCV 2024  
**作者**: Zhi-Fan Wu, Lianghua Huang, Wei Wang, Yanheng Wei, Yu Liu
**代码**: 无  
**领域**: 图像生成 / 扩散模型  
**关键词**: 多模态提示, 零样本图像生成, 增广token, 扩散模型, 多物体可控生成

## 一句话总结

本文提出 MultiGen，通过为每个物体构建"增广token"（融合文本、坐标和图像信息），并训练坐标模型和特征模型来处理推理时的模态缺失，首次实现了从多物体多模态提示进行零样本图像生成，支持纯文本或任意模态组合的灵活输入。

## 研究背景与动机

**领域现状**：文本到图像生成（Text-to-Image Generation）近年取得了巨大进展，Stable Diffusion、DALL-E 等模型已能根据文本描述生成高质量图像。但纯文本描述在精确控制物体外观、位置、细节等方面存在天然局限。

**现有痛点**：仅靠文本很难精确描述物体的具体外观（如"一只和我家猫一样的猫"）、精确位置（如"左上角放一个杯子"）等信息。现有方法要么需要微调（如 DreamBooth、Textual Inversion）来学习特定概念，要么只支持单个物体作为额外条件（如 IP-Adapter），无法直接支持多物体、多模态提示的零样本生成。多物体 + 多模态（文本+坐标+参考图像的任意组合）的零样本生成仍是未解决的挑战。

**核心矛盾**：用户想用多种模态（文本描述+空间位置+参考图像）来精确控制多个物体的生成，但(1)训练时需要多模态对齐的数据，(2)推理时用户可能只提供部分模态，需要模型能处理任意模态缺失的情况。

**本文目标** (1) 如何在一个统一框架中同时支持文本、坐标、参考图像三种模态？(2) 如何支持多个物体的同时可控生成？(3) 如何在零样本设置下工作，不需要针对特定物体/概念做微调？(4) 如何处理推理时部分模态缺失的情况？

**切入角度**：作者将每个物体的多模态信息封装成一个"增广token"（augmented token），使其可以与标准的文本 prompt 一起输入扩散模型，从而复用文本到图像模型的生成能力。对于推理时缺失的模态，训练专门的模型来从已知模态生成缺失模态。

**核心 idea**：将每个物体的文本、坐标、图像特征融合成"增广token"与文本 prompt 联合训练扩散模型，并用辅助模型预测缺失模态以实现零样本多模态可控生成。

## 方法详解

### 整体框架

MultiGen 基于预训练的文本到图像扩散模型（如 Stable Diffusion）。给定一张图像-文本对，系统先提取物体级别的三种信息：文本描述（object-level text）、空间坐标（bounding box）、参考图像特征（object image features）。然后将这三种信息融合为每个物体的"增广token"。这些增广token 与全局文本 prompt 一起作为条件输入扩散模型进行去噪训练。推理时，为了解决模态缺失问题，使用训练好的坐标模型（从文本预测布局）和特征模型（从文本预测视觉特征）来补全缺失信息。

### 关键设计

1. **增广Token（Augmented Token）**:

    - 功能：将每个物体的多模态信息（文本、坐标、图像）编码为一个统一的 token 表示
    - 核心思路：对于每个物体，分别获取其文本描述的 CLIP 文本特征、bounding box 的位置编码、以及参考图像的 CLIP 视觉特征。然后通过一个轻量级的融合网络将三种特征整合为一个"增广token"。这个 token 的维度与文本 embedding 兼容，可以直接拼接到全局 prompt 的 token 序列中。训练时使用图像-文本数据集自动提取物体信息来构造训练样本
    - 设计动机：将多模态信息压缩为与文本 token 同构的表示，使得扩散模型的 cross-attention 机制能自然地处理多模态条件，无需修改模型架构

2. **坐标模型（Coordinate Model）**:

    - 功能：在推理时从纯文本描述自动预测物体的空间布局（bounding box 坐标）
    - 核心思路：训练一个轻量级模型，输入为全局文本描述和各物体的文本标签，输出为每个物体的预测 bounding box。这个模型在大量图像-文本数据上训练，学习文本描述与空间布局之间的对应关系。当用户只提供文本而不指定位置时，坐标模型自动补全位置信息
    - 设计动机：用户通常不想手动指定每个物体的精确坐标。坐标模型让 MultiGen 在纯文本输入时也能工作，同时用户也可以覆盖自动预测的位置来手动控制布局

3. **特征模型（Feature Model）**:

    - 功能：在推理时从文本描述生成物体的视觉特征，替代缺失的参考图像
    - 核心思路：训练一个从 CLIP 文本特征到 CLIP 视觉特征的映射模型。当用户没有为某个物体提供参考图像时，特征模型根据文本描述生成近似的视觉特征，作为增广token 的图像特征分量。这种"文本到视觉特征"的转换比直接的"文本到图像"更轻量，只需要在特征空间中做对齐
    - 设计动机：零样本设置要求模型在没有任何参考图像的情况下也能工作。特征模型弥合了文本和视觉模态之间的 gap，使得增广token 在各种输入组合下都能保证质量

### 损失函数 / 训练策略

扩散模型部分使用标准的去噪损失 $L_{denoise} = \mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t, c)\|^2$，其中条件 $c$ 包含全局文本 prompt 和所有物体的增广token。坐标模型使用 L1 坐标回归损失。特征模型使用 cosine similarity 损失来对齐文本特征和视觉特征。训练分阶段进行：先分别训练坐标模型和特征模型，再联合微调扩散模型与增广token 的融合网络。

## 实验关键数据

### 主实验

| 任务/设置 | 指标 | MultiGen | 之前SOTA | 说明 |
|-----------|------|----------|---------|------|
| 多物体布局控制 | FID ↓ | 显著优于 | GLIGEN | 支持多模态而非仅文本+坐标 |
| 单物体外观参考 | CLIP-I ↑ | 有竞争力 | IP-Adapter | 零样本无需微调 |
| 纯文本生成 | FID ↓ | 与 SD 相当 | Stable Diffusion | 坐标和特征模型自动补全 |
| 多模态组合 | 用户研究 | 优选率高 | 无同类方法 | 首个支持此设定的方法 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full MultiGen | 最优 | 三种模态完整输入 |
| w/o 坐标信息 | 布局混乱 | 物体位置无法控制，重叠严重 |
| w/o 参考图像特征 | 外观一致性下降 | 只靠文本描述无法刻画细节外观 |
| w/o 特征模型 | 纯文本模式失效 | 无法处理缺少参考图像的情况 |
| 单物体 vs 多物体 | 多物体稍有下降 | 物体间干扰导致质量有一定损失 |

### 关键发现
- 增广token 的设计使得模型可以自然地支持任意数量的物体，扩展性好
- 坐标模型和特征模型的质量对最终生成效果影响显著——辅助模型的准确性是系统的瓶颈
- 在多物体场景中，物体间的属性混淆（attribute leaking）仍是一个问题，尤其是物体类别相近时
- 零样本效果虽不如微调方法（如 DreamBooth），但远优于其他零样本方法

## 亮点与洞察
- **增广token 的统一表示**是一个优雅的设计：将多模态信息统一为与文本 token 同构的表示，避免了对扩散模型架构的修改。这种设计思路可以推广到视频生成、3D 生成等需要多模态条件的任务中
- **模态缺失的处理策略**——用辅助模型预测缺失模态——是一种通用的方案。相比训练时随机 dropout 模态，这种方式在推理时更稳定
- 首次定义并解决了"多物体多模态零样本生成"问题，为后续研究奠定了基础和 baseline

## 局限与展望
- 多物体生成时的属性泄露（attribute leaking）问题尚未完全解决，物体间的特征容易交叉污染
- 坐标模型的预测精度有限，对于复杂布局（如 >5 个物体）的空间推理能力不足
- 当前只支持 bounding box 级别的空间控制，不支持更精细的分割遮罩或关键点控制
- 特征模型从文本生成的视觉特征与真实图像特征仍有 gap，导致零样本模式的外观保真度不够高
- 训练需要额外的坐标模型和特征模型，增加了系统复杂度

## 相关工作与启发
- **vs GLIGEN**: GLIGEN 支持文本+坐标的可控生成，但不支持参考图像输入。MultiGen 通过增广token 统一了三种模态
- **vs IP-Adapter**: IP-Adapter 支持参考图像作为条件，但只能处理全局参考或单个物体。MultiGen 扩展到了多物体场景
- **vs DreamBooth/Textual Inversion**: 这些方法通过微调来学习特定概念，效果好但不灵活。MultiGen 的零样本方式更适合即时使用场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次提出多物体多模态零样本生成的问题定义和解决方案
- 实验充分度: ⭐⭐⭐ 定性结果丰富，但定量对比受限于缺少同类方法的直接对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程描述清楚
- 价值: ⭐⭐⭐⭐ 开拓了新的问题设定，增广token 设计有启发性

<!-- RELATED:START -->

## 相关论文

- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior](freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)
- [T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting](../../CVPR2025/image_generation/t2icount_enhancing_cross-modal_understanding_for_zero-shot_counting.md)
- [MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)

<!-- RELATED:END -->
