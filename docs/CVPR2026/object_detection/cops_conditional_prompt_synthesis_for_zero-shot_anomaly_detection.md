---
title: >-
  [论文解读] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection
description: >-
  [CVPR 2026][目标检测][零样本异常检测] 本文提出 CoPS 框架，通过显式状态token合成（ESTS）和隐式类别token采样（ICTS）两种视觉条件化机制动态生成提示，配合空间感知对齐（SAGA），在13个工业和医学数据集上实现零样本异常检测SOTA。
tags:
  - "CVPR 2026"
  - "目标检测"
  - "零样本异常检测"
  - "条件提示合成"
  - "CLIP"
  - "视觉语言模型"
  - "工业缺陷"
---

# CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection

**会议**: CVPR 2026  
**arXiv**: [2508.03447](https://arxiv.org/abs/2508.03447)  
**代码**: [https://github.com/cqylunlun/CoPS](https://github.com/cqylunlun/CoPS)  
**领域**: 目标检测  
**关键词**: 零样本异常检测, 条件提示合成, CLIP, 视觉语言模型, 工业缺陷

## 一句话总结
本文提出 CoPS 框架，通过显式状态token合成（ESTS）和隐式类别token采样（ICTS）两种视觉条件化机制动态生成提示，配合空间感知对齐（SAGA），在13个工业和医学数据集上实现零样本异常检测SOTA。

## 研究背景与动机
1. **领域现状**：大规模预训练视觉-语言模型在零样本异常检测（ZSAD）中展现出良好的跨类别泛化能力。现有方法通过在单个辅助数据集上微调来实现跨类别异常检测。
2. **现有痛点**：（i）静态可学习token难以捕捉正常和异常状态的连续多样模式，限制了对未见类别的泛化；（ii）固定文本标签提供的类别信息过于稀疏，模型容易过拟合到特定语义子空间。
3. **核心矛盾**：提示学习消除了人工设计提示的需求，但其静态性和稀疏性成为泛化的瓶颈——正常/异常状态是连续多变的，而类别标签空间本身就是高度稀疏的。
4. **本文目标**：设计一种基于视觉特征条件化的动态提示合成框架，使提示能够自适应地建模输入图像的状态和类别信息。
5. **切入角度**：将提示分解为上下文词、状态词、类别词三部分，前者可共享，后两者需根据视觉特征动态生成。
6. **核心idea**：通过从局部特征提取正常/异常原型注入状态词（显式），通过VAE从全局特征采样注入类别词（隐式），实现视觉条件化的动态提示合成。

## 方法详解

### 整体框架
基于预训练CLIP，输入图像经冻结视觉编码器提取全局特征 $\mathbf{g}$ 和局部特征 $\mathbf{F}$。ESTS从局部特征中提取正常/异常原型注入状态词，ICTS通过VAE从全局特征采样多样化类别token，最终通过可学习文本编码器和SAGA模块实现图像级和像素级异常检测。

### 关键设计

1. **显式状态token合成（ESTS）**:
    - 功能：从细粒度局部特征中提取代表性正常和异常原型，显式注入提示的状态词
    - 核心思路：使用一致性自注意力（V-V attention）从冻结视觉编码器提取细粒度局部特征 $\mathbf{F}$，然后通过原型提取器 $\mathcal{P}_\theta$ 在中心约束下生成 $M$ 个正常原型 $\mathbf{P}_n$ 和异常原型 $\mathbf{P}_a$，将其组装为动态状态token替换静态可学习token。
    - 设计动机：固定的状态词（如"good"/"damaged"）无法捕捉连续多样的正常/异常模式。通过从实际图像的局部特征中提取原型，可以自适应地建模当前图像的状态，增强泛化能力。

2. **隐式类别token采样（ICTS）**:
    - 功能：利用VAE对语义全局特征建模，通过采样生成多样化类别token
    - 核心思路：使用变分自编码器 $\mathcal{E}_\psi$ 对全局特征 $\mathbf{g}$ 的潜在分布进行参数化，从中抽取 $R$ 个解码样本 $\mathbf{S} \in \mathbb{R}^{R \times C}$，作为密集的类别token。这样每个输入图像生成 $R$ 组完整的正常/异常提示。
    - 设计动机：固定的文本标签过于稀疏，无法提供丰富的类别语义信息。通过VAE采样，可以隐式地扩增类别表示的多样性，防止模型过拟合到单一语义子空间。

3. **空间感知全局-局部对齐（SAGA）**:
    - 功能：结合距离感知空间注意力实现精细的图像-文本对齐
    - 核心思路：利用查询特征与最近原型之间的距离近似异常状态，引入距离感知空间注意力机制来细化像素级文本-图像对齐。同时采用全局-局部（glocal）相似性交互来增强图像级对齐。最终输出图像级异常分数 $s_{\text{cls}}$ 和像素级异常图 $\mathcal{S}_{\text{seg}}$。
    - 设计动机：标准的全局对齐忽略了局部空间信息，而异常检测本质上需要精确的空间定位能力。

### 损失函数 / 训练策略
采用二元焦点损失用于图像级分类，Dice损失和二元交叉熵损失用于像素级分割。仅在单个辅助训练集（MVTec AD等）上微调，测试时直接应用到未见类别。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CoPS | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 13个数据集平均 | Cls AUROC | SOTA | - | +1.4% |
| 13个数据集平均 | Seg AUROC | SOTA | - | +1.9% |
| MVTec AD | Cls AUROC | 最优 | AnomalyCLIP等 | 显著提升 |
| VisA | Seg AUROC | 最优 | - | 明显优势 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full CoPS | 最优 | 完整模型 |
| w/o ESTS | 下降 | 去掉显式状态合成影响最大 |
| w/o ICTS | 下降 | 去掉隐式类别采样也有明显影响 |
| w/o SAGA | 下降 | 空间感知对齐对分割尤为重要 |
| 静态提示 baseline | 显著低于CoPS | 验证动态提示的必要性 |

### 关键发现
- ESTS贡献最大，说明自适应的状态建模是零样本异常检测的核心挑战。
- ICTS的VAE采样能有效缓解类别标签稀疏问题，尤其在跨域场景（工业→医学）中作用显著。
- 距离感知空间注意力对像素级分割质量提升明显，但对图像级分类影响较小。

## 亮点与洞察
- **提示分解的设计哲学**巧妙：上下文词共享+状态词显式注入+类别词隐式采样，各司其职。
- **VAE隐式扩增**是一个优雅的trick：用采样替代固定标签，自然地增加了类别表示的多样性。
- 一致性自注意力（V-V）的使用避免了额外适配模块的引入，保持了CLIP特征的原始语义。

## 局限与展望
- 依赖CLIP的预训练特征空间，对CLIP未覆盖的视觉域（如特殊工业场景）可能效果有限。
- 原型数量M和采样数量R需要手动调参。
- 未来可探索自适应确定原型数量，或引入更强的视觉基础模型替换CLIP。

## 相关工作与启发
- **vs AnomalyCLIP**: AnomalyCLIP使用静态可学习token，缺乏视觉条件化，本文通过显式/隐式注入克服了这一限制。
- **vs AdaCLIP**: AdaCLIP依赖手工设计的模板集，本文通过端到端学习消除了人工设计的需求。
- **vs VCP-CLIP**: VCP-CLIP直接将图像特征嵌入类别词，本文通过VAE采样提供了更丰富的语义多样性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 显式+隐式双路径动态提示合成是新颖的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 13个数据集全面验证，消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法讲解到位
- 价值: ⭐⭐⭐⭐ 零样本异常检测领域的实用进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)

</div>

<!-- RELATED:END -->
