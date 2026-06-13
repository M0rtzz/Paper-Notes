---
title: >-
  [论文解读] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models
description: >-
  [ICCV 2025][图像生成][文本到图像生成] CoMPaSS通过SCOP数据引擎筛选空间关系无歧义的训练数据，并提出无参数的TENOR模块将token顺序信息注入注意力机制，大幅提升T2I扩散模型的空间关系生成准确率（VISOR +98%、GenEval Position +131%）。 文本到图像扩散模型（如SD、…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "文本到图像生成"
  - "空间理解"
  - "扩散模型"
  - "数据引擎"
  - "token位置编码"
---

# CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2412.13195](https://arxiv.org/abs/2412.13195)  
**代码**: [https://github.com/blurgyy/CoMPaSS](https://github.com/blurgyy/CoMPaSS)  
**领域**: 图像生成  
**关键词**: 文本到图像生成, 空间理解, 扩散模型, 数据引擎, token位置编码

## 一句话总结
CoMPaSS通过SCOP数据引擎筛选空间关系无歧义的训练数据，并提出无参数的TENOR模块将token顺序信息注入注意力机制，大幅提升T2I扩散模型的空间关系生成准确率（VISOR +98%、GenEval Position +131%）。

## 研究背景与动机
文本到图像扩散模型（如SD、FLUX.1）在生成逼真图像方面表现出色，但在渲染准确的空间关系（如"左边"、"上面"）时经常失败。作者深入分析发现两个核心问题：

**数据歧义**：现有数据集（LAION、CC12M、COCO）中空间描述严重含糊——存在视角不一致（viewer vs. object-intrinsic）、空间词的非空间用法（如"the right choice"）、以及参考对象缺失/错误等问题。

**文本编码器的空间理解缺陷**：通过代理任务测试发现，CLIP和T5-XXL等编码器几乎无法区分语义等价的空间描述。例如"A left of B"和"B right of A"应该产生相似的embedding，但实际上T5-XXL只有4.84%的正确率，CLIP系列接近0%。

这两个问题相互叠加：即使文本编码器能正确编码空间关系，数据歧义仍然限制学习；反之，即使数据无歧义，编码器也无法传递正确信号。CoMPaSS同时解决这两个问题。

## 方法详解

### 整体框架
CoMPaSS包含两个互补组件：SCOP数据引擎（提供高质量空间训练数据）和TENOR模块（确保模型能区分不同结构的prompt），两者协同工作以增强T2I模型的空间理解。

### 关键设计

1. **SCOP（Spatial Constraints-Oriented Pairing）数据引擎**:

    - 功能：从图像中提取具有清晰空间关系的对象对，配以准确的文本描述
    - 核心思路：三阶段流程
        - **关系推理**：枚举图像中所有对象对 $\binom{n}{2}$ 个候选对
        - **空间约束执行**：通过5个约束过滤歧义对——视觉显著性（$\frac{\text{Area}(B_i \cup B_j)}{\text{Area}(I)} > \tau_v$）、语义区分（类别不同）、空间清晰度（质心距离/最小对角线 $< \tau_u$）、最小重叠（重叠率 $< \tau_o$）、尺寸均衡（面积比 $> \tau_s$）
        - **关系解码**：将结构化描述符解码为图像裁剪+文本prompt对
    - 设计动机：从COCO训练集中筛选得到28,000+对象对（仅LAION-400M的0.004%），人类标注一致率达85.2%

2. **TENOR（Token ENcoding ORdering）模块**:

    - 功能：将token顺序信息注入diffusion模型的text-image attention层
    - 核心思路：
        - 对UNet架构（SD系列）：在每个text-image attention层的key向量K上添加绝对位置编码
        - 对MMDiT架构（FLUX.1）：在text的query $Q_{\text{text}}$ 和key $K_{\text{text}}$ 上添加位置编码
    - 设计动机：标准Transformer的位置编码只在初始embedding添加一次，经过多层处理后顺序信息丢失。TENOR在每个attention操作中都注入顺序信息，确保"A left of B"和"B left of A"产生不同的conditioning信号。无额外参数，推理开销可忽略（~2.47%时间，~0.6% VRAM）

3. **SCOP与TENOR的协同**:

    - TENOR本身不理解"左/右"的语义含义，它只提供结构性信号确保不同prompt产生不同条件
    - SCOP提供语义真值（无歧义的空间数据），让模型学会将结构差异映射到正确的视觉结果
    - 单独使用SCOP：SD1.5 GenEval Position从0.04提升到0.39
    - 加入TENOR后：进一步提升到0.54

### 损失函数 / 训练策略
使用标准的diffusion训练损失进行微调。训练开销极低（+3.9%时间/+0.7% VRAM），无需训练新的text encoder。仅用500张图像即可获得显著提升（数据效率极高）。

## 实验关键数据

### 主实验

| 模型 | 基准 | 指标 | 原始 | +CoMPaSS | 相对提升 |
|------|------|------|------|----------|---------|
| FLUX.1 | VISOR uncond | 准确率 | 37.96% | 75.17% | +98% |
| FLUX.1 | T2I-CompBench Spatial | 分数 | 0.18 | 0.30 | +67% |
| FLUX.1 | GenEval Position | 分数 | 0.26 | 0.60 | +131% |
| FLUX.1 | DPG-Bench Relation | 分数 | 92.30 | 94.12 | +2% |
| SD1.5 | GenEval Position | 分数 | 0.04 | 0.54 | +1250% |
| SD2.1 | GenEval Position | 分数 | 0.07 | 0.51 | +629% |
| FLUX.1 | FID↓ | 图像质量 | 27.96 | 26.40 | 改善 |
| FLUX.1 | CMMD↓ | 图像质量 | 0.8737 | 0.6859 | 改善 |

### 消融实验

| 配置 | T2I-CompBench Spatial | GenEval Position | 说明 |
|------|----------------------|-----------------|------|
| SD1.5 原始 | 0.08 | 0.04 | 基线 |
| SD1.5 + SCOP | 0.32 | 0.39 | 数据引擎贡献巨大 |
| SD1.5 + SCOP + TENOR | 0.35 | 0.54 | TENOR进一步提升泛化 |
| FLUX.1 原始 | 0.18 | 0.26 | 基线 |
| FLUX.1 + SCOP | 0.29 | 0.56 | 数据引擎贡献巨大 |
| FLUX.1 + SCOP + TENOR | 0.30 | 0.60 | 完整方法最优 |

数据效率消融：仅用500张图像，FLUX.1的GenEval Position即从0.26提升到0.56，接近完整28k数据集的0.60。

### 关键发现
- 现有文本编码器对空间语义几乎完全失效：CLIP正确率0%~0.03%，T5-XXL仅4.84%
- 数据歧义是空间理解失败的首要原因，SCOP单独即可带来巨大提升
- TENOR对泛化至未见prompt至关重要，确保不同结构的prompt产生不同条件信号
- CoMPaSS不仅提升空间指标，还改善了整体生成质量和图像保真度
- 计算开销极低：推理仅增加2.47%时间和0.6% VRAM

## 亮点与洞察
- 深入的问题分析：构造代理任务量化text encoder的空间理解缺陷，清晰定位问题根源
- SCOP数据引擎设计精巧，5个原则性约束完美过滤歧义数据，仅需28k样本即可
- TENOR设计极其简洁（无参数、零开销），但效果显著，体现了"对症下药"的思路
- 方法具有很强的通用性：适用于UNet和MMDiT两种架构，4个不同模型上均有效

## 局限与展望
- 目前仅支持两个对象间的空间关系，多对象复杂场景的扩展有待探索
- SCOP依赖COCO的bounding box标注，数据规模受限于标注数据集
- 空间关系类型仅覆盖left/right/above/below，更复杂的关系（如"between"、"surrounding"）未涉及
- TENOR的位置编码策略较简单（绝对位置编码），是否存在更优的编码方式

## 相关工作与启发
- 与SPRIGHT同为training-based方法，但效率和效果远超SPRIGHT
- inference-only方法（R&B、Attention Refocusing等）依赖bounding box输入且开销大
- 启发：T2I模型的许多失败可能源于训练数据质量，而非模型能力不足

## 评分
- 新颖性: ⭐⭐⭐⭐ 对问题根源的分析非常深入，SCOP+TENOR的组合设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型、4个基准、详细消融、数据效率、计算效率全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，从问题分析到解决方案一气呵成
- 价值: ⭐⭐⭐⭐ 对T2I空间理解问题提出了系统性解决方案，具有较高的实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](../../CVPR2026/image_generation/enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning](../../CVPR2026/image_generation/spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](../../CVPR2025/image_generation/compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](understanding_flatness_in_generative_models_its_role_and_benefits.md)

</div>

<!-- RELATED:END -->
