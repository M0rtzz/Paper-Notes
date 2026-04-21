---
title: >-
  [论文解读] Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers
description: >-
  [ECCV 2024][图像生成][无监督目标跟踪] 提出 Diff-Tracker，首次利用预训练文本到图像扩散模型（Stable Diffusion）中蕴含的丰富视觉语义知识进行无监督目标跟踪，通过学习一个表示目标的 prompt 并在线更新来实现持续跟踪。
tags:
  - ECCV 2024
  - 图像生成
  - 无监督目标跟踪
  - 文本到图像扩散模型
  - 提示学习
  - 注意力机制
  - 在线更新
---

# Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers

**会议**: ECCV 2024  
**arXiv**: [2407.08394](https://arxiv.org/abs/2407.08394)  
**代码**: 无  
**领域**: 视觉跟踪 / 扩散模型  
**关键词**: 无监督目标跟踪, 文本到图像扩散模型, Prompt学习, 注意力机制, 在线更新

## 一句话总结

提出 Diff-Tracker，首次利用预训练文本到图像扩散模型（Stable Diffusion）中蕴含的丰富视觉语义知识进行无监督目标跟踪，通过学习一个表示目标的 prompt 并在线更新来实现持续跟踪。

## 研究背景与动机

### 领域现状
视觉目标跟踪是计算机视觉核心任务，深度学习跟踪器（如SiamFC、MixFormer）依赖大量标注数据进行有监督训练。由于标注成本高昂，**无监督视觉跟踪**近年获得大量关注。现有无监督方法（UDT、USOT、ULAST）多采用孪生网络架构，通过前后向一致性或对抗遮挡等自监督策略训练。

### 核心痛点
无监督跟踪面临两大挑战：

**语义和结构信息利用不足**：难以有效利用视频帧中丰富的语义和空间结构信息

**上下文关系挖掘不够**：视频中的时间上下文关系未被充分利用

### 核心矛盾与切入角度
预训练文本到图像扩散模型（如Stable Diffusion）已展现出对视觉表征的全面理解——从像素级语义细节到空间布局（物体纹理、形状、空间排布）。研究者还验证了扩散模型即使未在视频上训练，也对视频上下文关系有良好理解能力。

**关键洞察**：扩散模型中的 **cross-attention map** 将文本prompt的语义与图像内容关联——prompt能在注意力图上激活与其语义相关的区域。如果能学习一个代表跟踪目标的prompt，就能利用扩散模型的丰富知识在任意帧上激活目标区域，从而实现跟踪。

### 两个技术挑战
1. 学到的prompt需要编码目标与背景的关系信息（对处理遮挡和形变至关重要）
2. 目标运动导致外观变化，prompt需要在线动态更新以适应这些变化

## 方法详解

### 整体框架

Diff-Tracker 分为两个阶段：
1. **初始prompt学习器（Initial Prompt Learner）**：在第一帧上学习一个能激活目标区域的初始prompt
2. **在线prompt更新器（Online Prompt Updater）**：根据目标运动信息动态更新prompt，实现后续帧的持续跟踪

### 关键设计

#### 1. 注意力协调方法（Attention Harmonization）

**功能**：将cross-attention map与self-attention map融合，使learned prompt编码目标与背景的关系信息。

**核心思路**：扩散模型的self-attention map $M_s$ 捕获像素间的语义关联。对cross-attention map $M_c$ 上的每个像素，用其对应的self-attention map进行增强：

$$M_c'(:,:) = \sum_{i=1}\sum_{j=1} M_c(i,j) \cdot M_s(i,j,:,:)$$

增强后的cross-attention map $M_c'$ 编码了像素间关系。最终通过加权融合得到输出注意力图：

$$\mathcal{M} = (1-\alpha) \cdot M_c' + \alpha \cdot M_c$$

其中 $\alpha=0.5$ 是平衡超参数。

**设计动机**：单纯的cross-attention只反映prompt与各像素的相关性，缺乏目标与背景的关系建模。引入self-attention能让prompt同时感知"目标是什么"和"目标与周围环境的关系"，在遮挡和相似干扰物场景下尤为重要。

#### 2. 初始Prompt学习（Initial Prompt Learning）

**功能**：从第一帧学习一个能准确激活目标bounding box区域的prompt embedding。

**核心思路**：
1. 将第一帧 $f_1$ 编码到潜空间并加噪
2. 将噪声潜表示 $z$ 和待学习prompt $p_1$ 送入去噪UNet $\epsilon_\theta$
3. 提取cross-attention map $M_c$，通过注意力协调得到 $\mathcal{M}$
4. 计算 $\mathcal{M}$ 与GT注意力图（仅bounding box区域激活）的MSE损失

总损失：

$$L = \|\mathcal{M} - \mathcal{F}_1\|_2^2 + L_{DM}$$

其中 $L_{DM}$ 是扩散模型去噪损失，约束prompt停留在扩散模型可理解的文本嵌入空间内。训练中扩散模型参数冻结，仅更新prompt embedding（维度1024）。

**设计动机**：因为目标任意（无法用文本描述），使用可学习的prompt embedding代替文字。GT注意力图定义为仅在bounding box内激活，简洁有效。

#### 3. 在线Prompt更新器（Online Prompt Updater）

**功能**：根据目标在后续帧中的运动信息，动态更新prompt以适应目标外观变化。

**核心思路**：

**运动信息提取器**：同时提取长期和短期运动信息：
- **短期运动** $m_k^S$：当前帧与前一帧输入运动编码器（ResNet18 + Conv3D）
- **长期运动** $m_k^L$：当前帧与前T帧堆叠输入另一运动编码器

通过cross-attention将全局运动信息与目标外观特征关联，得到目标条件化的运动信息：

$$l_k^L = \text{Cross-Attn}(Q_k, m_k^L), \quad l_k^S = \text{Cross-Attn}(Q_k, m_k^S)$$

其中 $Q_k$ 来自第一帧目标模板的外观特征。融合头（MLP）将长短期运动信息融合为 $l_k$。

**Prompt更新**：通过混合头 $\mathcal{H}_b$ 和残差连接更新prompt：

$$p_k = (1-\beta) \cdot \mathcal{H}_b(p_{k-1} + l_k) + \beta \cdot p_{k-1}$$

$\beta=0.7$ 确保更新稳定性，以前一帧prompt为主，运动信息为辅。

**设计动机**：仅靠短期运动可能因遮挡/光照变化而不可靠，引入长期运动增强时空连续性。残差方式避免prompt剧烈变化导致跟踪漂移。

### 损失函数 / 训练策略

- 初始prompt学习：MSE注意力损失 + 扩散模型去噪损失，Adam优化器，lr=$5 \times 10^{-3}$，3 epochs
- 在线更新器训练：MSE损失（更新后prompt的cross-attention map vs GT），Adam优化器，lr=$5 \times 10^{-4}$，35 epochs
- 伪标签来源：用光流模型在GOT-10k、ImageNet VID、LaSOT、YouTube-VOS上生成
- 测试时：第一帧学习初始prompt，从第6帧开始每帧在线更新prompt

## 实验关键数据

### 主实验

**TrackingNet / VOT2016 / VOT2018 结果（无监督方法对比）**：

| 方法 | 监督 | TrackingNet Suc↑ | TrackingNet Pre↑ | VOT2016 EAO↑ | VOT2018 EAO↑ |
|------|------|-----------------|-----------------|-------------|-------------|
| KCF | 无 | 0.447 | 0.419 | 0.192 | 0.135 |
| USOT* | 无 | 0.616 | 0.566 | 0.402 | 0.344 |
| ULAST*-on | 无 | 0.654 | 0.592 | 0.417 | 0.355 |
| **Diff-Tracker (Ours)** | **无** | **0.675** | **0.614** | **0.430** | **0.365** |
| SiamFC | 有 | 0.571 | 0.533 | 0.235 | 0.188 |
| DiMP | 有 | 0.740 | 0.687 | - | 0.440 |

**OTB2015 / LaSOT 结果**：

| 方法 | 监督 | OTB2015 Suc↑ | OTB2015 Pre↑ | LaSOT Suc↑ | LaSOT Pre↑ |
|------|------|-------------|-------------|-----------|-----------|
| USOT* | 无 | 0.574 | 0.775 | 0.358 | 0.340 |
| ULAST*-on | 无 | 0.648 | 0.879 | 0.471 | 0.451 |
| **Diff-Tracker (Ours)** | **无** | **0.661** | **0.898** | **0.486** | **0.472** |

在全部5个数据集上均刷新无监督跟踪SOTA，且在部分指标上接近有监督方法（如超过SiamFC等经典方法）。

### 消融实验

**VOT2018 上的组件消融**：

| 配置 | EAO↑ | Acc↑ | Rob↓ | 说明 |
|------|------|------|------|------|
| w/o attention harmonization | 0.359 | 0.577 | 0.280 | 注意力协调编码目标-背景关系 |
| w/o online prompt updater | 0.349 | 0.571 | 0.292 | 在线更新适应目标运动变化 |
| w/o long-term motion | 0.355 | 0.572 | 0.286 | 长期运动增强时空连续性 |
| w/o short-term motion | 0.360 | 0.577 | 0.281 | 短期运动捕捉即时变化 |
| **Full (Ours)** | **0.365** | **0.580** | **0.273** | 所有组件协同最优 |

### 关键发现

1. 在线prompt更新器的贡献最大（去除后EAO从0.365降至0.349），证实动态适应目标变化的必要性
2. 注意力协调有效编码了目标与背景的关系（EAO提升0.006）
3. 长期和短期运动信息互补，单独去除任一都会降低性能
4. 预训练扩散模型无需微调即可有效迁移到跟踪任务，展示了强大的视觉理解泛化能力

## 亮点与洞察

1. **新颖的视角转换**：将文本到图像扩散模型重新解读为"语义-图像区域的桥接器"，通过prompt学习实现跟踪
2. **无需文本描述**：跟踪目标可能无法用文字描述，使用可学习embedding巧妙绕过了这一限制
3. **自注意力的巧妙利用**：扩散模型的self-attention天然编码像素间关系，通过attention harmonization优雅地引入目标-背景关系

## 局限与展望

1. 测试时每帧都需学习/更新prompt，推理速度可能较慢
2. GT attention map定义为简单的bounding box激活，未利用更精细的分割信息
3. 仅用单个prompt token表示目标，对复杂目标的描述能力有限
4. 依赖Stable Diffusion的预训练知识，对扩散模型训练数据中罕见的目标类别可能表现不佳
5. 可探索使用更高效的扩散模型（如SDXL Turbo）加速推理

## 相关工作与启发

- **ULAST** (CVPR 2024)：当前无监督跟踪SOTA，使用孪生网络+对抗学习，本文超越之
- **Prompt-to-Prompt** (ICLR 2023)：揭示了扩散模型cross-attention map对图像区域的精确控制
- **DiffPose** (CVPR 2023)：将扩散模型用于姿态估计，与本文同属"扩散模型迁移"思路
- 启发：大规模预训练生成模型蕴含的表征知识，有潜力被迁移到各种判别性视觉任务中

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将文本到图像扩散模型用于无监督跟踪，视角新颖且方案优雅
- **实验充分度**: ⭐⭐⭐⭐ — 5个基准全面SOTA，消融完整
- **写作质量**: ⭐⭐⭐⭐ — 动机推导清晰，从观察到方案逻辑自洽
- **价值**: ⭐⭐⭐⭐ — 开辟了扩散模型在跟踪领域的新范式，启发性强

<!-- RELATED:START -->

## 相关论文

- [Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation](harnessing_text-to-image_diffusion_models_for_category-agnostic_pose_estimation.md)
- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)

<!-- RELATED:END -->
