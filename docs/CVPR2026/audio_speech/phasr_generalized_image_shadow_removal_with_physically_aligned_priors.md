---
title: >-
  [论文解读] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors
description: >-
  [CVPR 2026][语音][阴影去除] 提出PhaSR框架，通过双层物理先验对齐——全局级的PAN执行无参数Retinex分解抑制色彩偏差、局部级的GSRA利用差分注意力对齐DepthAnything深度先验和DINO-v2语义嵌入——实现从单光源直射阴影到多光源环境光场景的泛化阴影去除，在WSRD+和Ambient6K上达到SOTA且FLOPs最低。
tags:
  - CVPR 2026
  - 语音
  - 阴影去除
  - Retinex分解
  - 差分注意力
  - 几何-语义先验对齐
  - 环境光归一化
---

# PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors

**会议**: CVPR 2026  
**arXiv**: [2601.17470](https://arxiv.org/abs/2601.17470)  
**代码**: https://github.com/ming053l/PhaSR (有)  
**领域**: 图像复原  
**关键词**: 阴影去除, Retinex分解, 差分注意力, 几何-语义先验对齐, 环境光归一化

## 一句话总结
提出PhaSR框架，通过双层物理先验对齐——全局级的PAN执行无参数Retinex分解抑制色彩偏差、局部级的GSRA利用差分注意力对齐DepthAnything深度先验和DINO-v2语义嵌入——实现从单光源直射阴影到多光源环境光场景的泛化阴影去除，在WSRD+和Ambient6K上达到SOTA且FLOPs最低。

## 研究背景与动机

**领域现状**：阴影去除是计算机视觉基础任务，核心挑战在于准确区分阴影与物体固有暗色区域，并进行物理合理的颜色校正。学习方法从CNN到Transformer到扩散模型不断进步，但大多在单光源直射阴影基准上评估。

**现有痛点**：(1) 仅依赖RGB线索时，阴影容易与材料固有属性混淆，在纹理边界处产生颜色失真；(2) 现有方法在单光源直射阴影基准上表现好，但面对多光源室内环境光场景（色偏、漫反射间接照明）泛化能力差；(3) 传统编码器-解码器框架无法有效传播物理先验，均匀融合忽略了空间变化的退化特征，导致边缘模糊。

**核心矛盾**：物理先验未对齐（prior misalignment）。几何特征（局部着色变化、法线方向）对光照几何敏感但有噪声，语义特征（物体类别、材料）跨光照稳定但空间粗糙。如果不正确对齐，几何噪声会破坏语义一致性，或语义过度平滑会擦除光照边界——在间接光照下尤其严重。

**本文目标** (1) 全局色彩偏差的抑制；(2) 几何先验和语义先验的跨模态冲突解决；(3) 从单光源到多光源场景的泛化能力。

**切入角度**：从"对齐"的角度统一思考——全局级对齐（PAN做光照-反射率分解）和局部级对齐（GSRA用差分注意力协调几何和语义）。

**核心 idea**：通过双层物理先验对齐（全局无参数Retinex归一化 + 局部差分注意力跨模态校正），使阴影去除系统能从单光源泛化到复杂多光源场景。

## 方法详解

### 整体框架
PhaSR分为两阶段：Stage 1是PAN——无模型参数的预处理模块，执行Gray-world颜色归一化→对数域Retinex分解→动态范围重组合，输出光照一致的图像。Stage 2是多尺度Transformer编码器-解码器，在编码器阶段注入冻结的DINO-v2语义嵌入，在瓶颈层注入DepthAnything-v2几何先验（深度+法线），通过GSRA的跨模态差分注意力对齐两种先验。整个流程不需要阴影mask。

### 关键设计

1. **物理对齐归一化 (PAN)**:

    - 功能：无参数预处理，抑制全局色彩偏差，提供光照一致的输入
    - 核心思路：三步流程——(a) **Gray-world颜色归一化**：$\mathbf{I}_{\text{norm}} = \mathbf{I} \cdot \frac{\mathbb{E}[\mathbf{I}]}{\mathbb{E}_c[\mathbf{I}]+\varepsilon}$，平衡通道光照去除色偏；(b) **对数域Retinex分解**：在对数域将图像分解为反射率和光照分量——$\log\hat{\mathbf{S}} = \mathbb{E}_{H,W}[\log(\mathbf{I}_{\text{norm}}+\varepsilon)]$，$\log\hat{\mathbf{R}} = \log(\mathbf{I}_{\text{norm}}+\varepsilon) - \log\hat{\mathbf{S}}$——利用对数域的加法可分性实现闭式求解；(c) **重组合归一化**：$\hat{\mathbf{I}} = \frac{\hat{\mathbf{R}} \otimes \hat{\mathbf{S}} - \min}{\max - \min + \varepsilon}$
    - 设计动机：与学习型Retinex分解不同，PAN是闭式运算无需训练参数，可作为即插即用模块嵌入任何框架。实验证明作为插件可为OmniSR/DenseSR等方法提升0.15-0.34dB

2. **几何-语义校正注意力 (GSRA)**:

    - 功能：对齐深度几何先验和DINO-v2语义嵌入，解决跨模态冲突
    - 核心思路：(a) **多模态先验注入**：将共享查询特征分别与几何和语义先验相加（带可学习缩放因子 $\alpha$），生成模态特定的键值对；(b) **差分校正**：用共享查询计算两个注意力图 $\mathbf{A}_{\text{geo}}$ 和 $\mathbf{A}_{\text{sem}}$，然后执行校正 $\mathbf{A}_{\text{rect}} = \mathbf{A}_{\text{sem}} - \lambda \cdot \mathbf{A}_{\text{geo}}$，其中可学习的 $\lambda$ 平衡光照变化敏感度和几何正则化强度；(c) 最终输出 $\mathbf{F}_{\text{output}} = \text{Concat}(\mathbf{A}_{\text{rect}}\mathbf{V}_{\text{geo}}, \mathbf{A}_{\text{rect}}\mathbf{V}_{\text{sem}})$
    - 设计动机：几何特征在阴影边缘精确但在均匀光照区域有噪声，语义特征稳定但空间粗糙。差分注意力的减法结构天然实现了物理可解释的门控——在真实光照边界保留几何精度，在均匀区域抑制几何噪声。与原始DiffTransformer（同一自注意力头内减法）不同，GSRA是跨模态减法

3. **多尺度Transformer骨干**:

    - 功能：无mask阴影去除的主干编码器-解码器
    - 核心思路：层次化架构，基础通道维度 $C=32$，每个Transformer块2层。编码器阶段通过冻结DINO-v2注入语义先验，瓶颈层通过DepthAnything-v2注入几何先验。使用GSRA在瓶颈层对齐两种先验
    - 设计动机：将物理先验注入到网络的不同阶段（语义在编码、几何在瓶颈），匹配其各自最适合的抽象层次

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{\text{total}} = 0.95\mathcal{L}_{\text{Charb}} + 0.05\mathcal{L}_{\text{SSIM}}$，即Charbonnier损失保真度+SSIM损失结构一致性。使用AdamW优化器，batch size 9，训练1400 epochs，学习率 $2\times10^{-4}$ 余弦退火。

## 实验关键数据

### 主实验

| 数据集 | 指标 | PhaSR | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| ISTD | PSNR/SSIM | 30.73/0.960 | 30.64(DenseSR) | +0.09 |
| ISTD+ | PSNR/SSIM | 34.48/0.960 | 35.19(StableSD) | -0.71 |
| INS | PSNR/SSIM | 30.38/0.961 | 30.64(DenseSR) | -0.26 |
| WSRD+ | PSNR/SSIM | **28.44/0.842** | 26.28(DenseSR) | **+2.16** |
| Ambient6K | PSNR/SSIM | **23.32/0.834** | 22.54(DenseSR) | **+0.78** |

注：在最具挑战性的WSRD+和Ambient6K（多光源环境光）上提升最大，验证了泛化能力。

### 消融实验（PAN作为插件）

| 框架 + PAN | ISTD(PSNR) | WSRD+(PSNR) | Ambient6K(PSNR) |
|------|---------|------|------|
| OmniSR | 30.45→30.60 | 26.07→26.22 | 23.01→23.15 |
| DenseSR | 30.64→30.98 | 26.28→26.47 | 22.54→22.73 |
| PhaSR(Ours) | 30.73 | 28.44 | 23.32 |

### 复杂度对比

| 方法 | FLOPs(G) | Params(M) |
|------|----------|-----------|
| OmniSR | 118.67 | 21.02 |
| DenseSR | 109.32 | 24.70 |
| PhaSR | **55.63** | **18.95** |

### 关键发现
- **在环境光场景(Ambient6K)上PhaSR大幅领先**——比专门的环境光归一化方法IFBlend(21.44dB)高出1.88dB，说明物理对齐先验在多光源场景中的关键作用
- **PAN作为即插即用模块**可稳定提升多种框架性能，ISTD数据集上误差减少高达26.4%
- **PhaSR的计算效率最高**——FLOPs仅55.63G，约为OmniSR的47%、DenseSR的51%，同时参数量最小(18.95M)
- 在ISTD+上低于StableShadowDiffusion，但后者是基于扩散的方法，计算代价远高于PhaSR
- PAN与传统颜色校正方法（ACE、White-balance等）对比，在所有指标上都优于后者

## 亮点与洞察
- **PAN的即插即用性**是最大亮点——一个无参数的闭式预处理模块就能稳定提升各种阴影去除方法0.15-0.34dB，说明很多方法的输入端就存在color bias问题。这个模块可以迁移到任何图像复原任务
- **GSRA的跨模态差分注意力**：$\mathbf{A}_{\text{sem}} - \lambda \cdot \mathbf{A}_{\text{geo}}$ 有很好的物理可解释性——语义注意力是"全局稳定的基底"，几何注意力是"局部光照敏感的扰动"，减法操作校正语义过平滑同时抑制几何噪声。这种跨模态差分范式可迁移到其他需要融合异质先验的任务
- **从单光源到多光源的泛化思路**：通过物理对齐而非数据驱动来实现泛化，比直接扩大训练集更优雅

## 局限与展望
- PAN基于Gray-world假设，对于颜色分布极不均匀的图像（如大面积单色背景）可能引入偏差
- 对数域Retinex用全局均值估计光照，无法处理空间变化剧烈的复杂光照（如多方向聚光灯）
- 在ISTD+上低于扩散模型方法，精细纹理恢复仍有提升空间
- GSRA中的 $\lambda$ 是全局可学习标量，空间自适应的 $\lambda(x,y)$ 可能在处理局部复杂阴影时更优

## 相关工作与启发
- **vs OmniSR**: OmniSR也用几何-语义先验但融合策略无法正确对齐互补模态强度；PhaSR通过差分注意力显式校正
- **vs DenseSR**: DenseSR将阴影去除重构为密集预测利用自适应融合，但在Ambient6K上依然显著低于PhaSR，说明没有物理对齐的融合在多光源场景下不够
- **vs ReHiT**: ReHiT用Retinex引导的双分支分解做无mask阴影去除，但在环境光场景下性能下降（Ambient6K仅19.98dB），PhaSR通过PAN+GSRA实现了更好的泛化

## 评分
- 新颖性: ⭐⭐⭐⭐ 双层物理对齐（PAN+GSRA）的设计思路系统且有物理直观
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖五个基准包括环境光场景，PAN插件实验、传统方法对比、复杂度分析齐全
- 写作质量: ⭐⭐⭐⭐ 物理动机阐述清晰，图表辅助理解
- 价值: ⭐⭐⭐⭐ PAN作为即插即用模块有广泛应用价值，GSRA的跨模态对齐思路可泛化

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](../../ICLR2026/audio_speech/incentive-aligned_multi-source_llm_summaries.md)
- [\[NeurIPS 2025\] Generating Physically Sound Designs from Text and a Set of Physical Constraints](../../NeurIPS2025/audio_speech/generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)
- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](../../CVPR2025/audio_speech/improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [\[ACL 2026\] Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations](../../ACL2026/audio_speech/affectron_emotional_speech_synthesis_with_affective_and_contextually_aligned_non.md)
- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)

<!-- RELATED:END -->
