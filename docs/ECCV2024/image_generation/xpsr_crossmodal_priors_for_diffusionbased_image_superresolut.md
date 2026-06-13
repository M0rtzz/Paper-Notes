---
title: >-
  [论文解读] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution
description: >-
  [ECCV 2024][图像生成][图像超分辨率] XPSR提出将多模态大语言模型（LLaVA）生成的高层与低层语义描述作为跨模态先验，通过Semantic-Fusion Attention融合到扩散模型中，并设计Degradation-Free Constraint提取语义保留特征，实现高保真高真实感的图像超分辨率。
tags:
  - "ECCV 2024"
  - "图像生成"
  - "图像超分辨率"
  - "扩散模型"
  - "多模态大语言模型"
  - "跨模态语义先验"
  - "ControlNet"
---

# XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution

**会议**: ECCV 2024  
**arXiv**: [2403.05049](https://arxiv.org/abs/2403.05049)  
**代码**: [https://github.com/qyp2000/XPSR](https://github.com/qyp2000/XPSR)  
**领域**: 图像复原  
**关键词**: 图像超分辨率, 扩散模型, 多模态大语言模型, 跨模态语义先验, ControlNet

## 一句话总结

XPSR提出将多模态大语言模型（LLaVA）生成的高层与低层语义描述作为跨模态先验，通过Semantic-Fusion Attention融合到扩散模型中，并设计Degradation-Free Constraint提取语义保留特征，实现高保真高真实感的图像超分辨率。

## 研究背景与动机

1. **领域现状**：基于扩散模型的图像超分辨率（ISR）利用预训练T2I模型（如Stable Diffusion）的生成先验，通过ControlNet等方式注入低分辨率图像信息来恢复高分辨率图像。StableSR、DiffBIR、PASD、SeeSR等是代表方法。

2. **现有痛点**：(1) StableSR/DiffBIR直接将prompt设为空，依赖从LR图像提取语义，但LR图经历复杂退化后语义信息丢失严重；(2) PASD/SeeSR用标签模型提取物体类别作为prompt，但缺乏空间位置、场景理解等复杂信息；(3) 现有prompt均忽略了图像质量、噪声、模糊等低层信息，而这些对ISR至关重要。

3. **核心矛盾**：T2I扩散模型的生成过程本质上依靠文本prompt引导，但ISR场景中LR图像退化严重，简单的标签级prompt无法提供足够丰富的语义引导，导致恢复图像内容错误或产生不真实伪影。

4. **本文要解决什么？** (1) 如何获取准确全面的语义条件？(2) 如何有效融合不同层级的跨模态先验？(3) 如何从LR图像中提取语义保留但退化无关的特征？

5. **切入角度**：作者发现高层语义先验（物体描述、空间位置）能帮助恢复语义正确的内容，低层语义先验（质量、清晰度、噪声）能帮助建模退化过程实现更清晰恢复。MLLM（如LLaVA）恰好能同时感知这两类信息。

6. **核心idea一句话**：用MLLM生成高-低双层语义prompt，通过并行交叉注意力融合到扩散模型中，并用像素-潜在双空间约束提取去退化特征。

## 方法详解

### 整体框架

XPSR分两个阶段：(1) 语义先验生成——用LLaVA对LR图像生成高层描述（内容、场景）和低层描述（质量、噪声），通过CLIP文本编码器得到两种embedding；(2) 图像恢复——基于SD+ControlNet架构，用提出的SFA融合双层语义先验，用DFC约束ControlNet提取语义保留特征。推理时只需ControlNet+UNet+MLLM。

### 关键设计

1. **MLLM语义Prompt生成**:
    - 做什么：用LLaVA从LR图像提取高层和低层语义描述
    - 核心思路：设计两个指令——高层："Please provide a descriptive summary of the content of this image"，生成包含物体描述、空间位置、场景等内容；低层："Please describe the quality of this image and evaluate it based on factors such as clarity, color, noise, and lighting"，生成质量、清晰度、噪声等描述
    - 设计动机：高层先验提供丰富语义使恢复内容正确，低层先验帮助建模退化过程使恢复更清晰。实验可视化表明两者缺一不可

2. **Semantic-Fusion Attention (SFA)**:
    - 做什么：将高层和低层语义先验有效融合到扩散模型中
    - 核心思路：采用并行双分支交叉注意力代替串行结构。高层和低层分别通过独立的交叉注意力与特征交互，然后通过融合注意力（高层结果做Q，低层结果做K/V）合并：$\mathbf{x}_{k+1} = \mathcal{CA}_f(\mathcal{CA}_h(\mathbf{x}_k, c_h), \mathcal{CA}_l(\mathbf{x}_k, c_l))$
    - 设计动机：串行结构会导致后处理的信息覆盖前面的信息，并行结构实现两种先验的自适应平衡选择。UNet只用高层注意力（因为输入是噪声不需要低层退化理解），ControlNet用SFA做完整融合

3. **Degradation-Free Constraint (DFC)**:
    - 做什么：约束ControlNet提取语义保留但退化无关的特征
    - 核心思路：在像素空间和潜在空间双层施加L1约束。像素空间：在ControlNet图像编码器的每一层用卷积映射到RGB图像，与HR图像下采样结果对齐；潜在空间：在UNet编码器各层映射到潜在空间，与HR latent下采样对齐。$\mathcal{L}_{DFC} = \sum_{i=1}^{3} \|x_{hr,i} - \hat{x}_i\|_1 + \sum_{j=1}^{3} \|z_{hr,j} - \hat{z}_j\|_1$
    - 设计动机：LR图像包含退化信息和语义信息的混合，DFC通过与HR对齐迫使特征只保留语义，丢弃退化相关成分

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_D + \lambda \mathcal{L}_{DFC}$，其中 $\mathcal{L}_D$ 是标准扩散去噪损失。冻结SD所有参数，只训练ControlNet和Conditional Attention。推理时使用classifier-free guidance，负面prompt为"blurry, dotted, noise, unclear, low-res, over-smoothed"。

## 实验关键数据

### 主实验

| 数据集 | 方法 | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | LIQE↑ |
|--------|------|----------|--------|---------|-------|
| DIV2K-Val | StableSR | 0.621 | 64.22 | 0.395 | 4.13 |
| DIV2K-Val | SeeSR | 0.655 | 66.43 | 0.420 | 4.25 |
| DIV2K-Val | **XPSR** | **0.689** | **68.71** | **0.441** | **4.38** |
| RealSR | StableSR | 0.588 | 63.80 | 0.381 | 3.98 |
| RealSR | **XPSR** | **0.651** | **67.13** | **0.422** | **4.21** |

### 消融实验

| 配置 | CLIPIQA | MUSIQ | 说明 |
|------|---------|-------|------|
| w/o 高层prompt | 0.645 | 65.8 | 内容语义恢复不准确 |
| w/o 低层prompt | 0.652 | 66.1 | 退化建模不充分，细节模糊 |
| 串行注意力代替SFA | 0.661 | 67.0 | 信息覆盖导致次优融合 |
| w/o DFC | 0.658 | 66.5 | 退化信息混入特征 |
| Full XPSR | **0.689** | **68.71** | 完整模型 |

### 关键发现

- **高层和低层语义先验互补不可替代**：去掉任一类prompt均导致显著性能下降，证实了双层语义条件的必要性
- **并行SFA显著优于串行融合**：串行结构因信息覆盖问题导致次优结果
- **DFC的像素+潜在双空间约束缺一不可**：单独去掉任一空间约束均降低性能
- **低层prompt的准确性至关重要**：可视化表明错误的低层描述（如把模糊说成清晰）会导致恢复质量严重下降

## 亮点与洞察

- **MLLM作为ISR的语义条件生成器**：这是一个巧妙的跨界应用，LLaVA同时感知高层内容和低层质量的能力恰好弥补了ISR中语义条件不足的问题
- **SFA并行融合设计**：用并行交叉注意力+融合注意力的三分支结构，优雅解决了多条件融合中的信息覆盖问题，可迁移到任何需要融合多种条件的生成任务
- **低层语义先验的发现**：明确指出图像质量/退化描述对ISR有重要价值，这在之前的工作中被忽视

## 局限性 / 可改进方向

- MLLM推理增加了计算开销，每张图需要额外调用LLaVA生成描述
- LLaVA对LR图像的感知可能不总是准确，尤其在极端退化下
- 训练数据使用合成退化pipeline，与真实世界退化仍有分布差距
- 可以探索端到端训练MLLM与SD的联合优化方案

## 相关工作与启发

- **vs SeeSR**: SeeSR用标签模型提取物体tag作为prompt，XPSR用MLLM获取更丰富的描述和低层质量信息，语义条件更全面
- **vs PASD**: PASD也引入语义信息，但只用物体标签，XPSR的MLLM方案提供了空间位置和场景理解等更高级信息
- **vs StableSR/DiffBIR**: 它们不用文本条件，完全依赖LR图像特征，在复杂退化下语义信息丢失严重

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将MLLM的高低层语义理解引入ISR
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集，多指标全面评估，消融充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化说服力强
- 价值: ⭐⭐⭐⭐ 为ISR领域引入MLLM语义条件的范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](../../CVPR2025/image_generation/faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)

</div>

<!-- RELATED:END -->
