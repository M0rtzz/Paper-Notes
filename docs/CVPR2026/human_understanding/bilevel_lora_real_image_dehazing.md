---
title: >-
  [论文解读] Bilevel Layer-Positioning LoRA for Real Image Dehazing
description: >-
  [CVPR 2026][人体理解][图像去雾] 利用CLIP跨模态能力将去雾重构为语义对齐问题（H2C损失），并通过双层优化自动搜索最佳LoRA注入层（BiLaLoRA），实现即插即用的高效合成到真实域去雾适配。
tags:
  - CVPR 2026
  - 人体理解
  - 图像去雾
  - LoRA
  - CLIP
  - 双层优化
  - 无监督域适配
---

# Bilevel Layer-Positioning LoRA for Real Image Dehazing

**会议**: CVPR 2026  
**arXiv**: [2603.10872](https://arxiv.org/abs/2603.10872)  
**代码**: [GitHub](https://github.com/YanZhang-zy/BiLaLoRA)  
**领域**: 图像恢复 / 低层视觉  
**关键词**: 图像去雾, LoRA, CLIP, 双层优化, 无监督域适配  

## 一句话总结

利用CLIP跨模态能力将去雾重构为语义对齐问题（H2C损失），并通过双层优化自动搜索最佳LoRA注入层（BiLaLoRA），实现即插即用的高效合成到真实域去雾适配。

## 研究背景与动机

**深度学习去雾模型在合成数据上表现优异，但合成-真实域差距导致真实场景性能大幅下降**。两个核心痛点亟需解决：(1) 真实场景没有配对清晰图作为ground truth，缺少有效无监督优化信号；(2) 全模型微调代价高昂且灵活性差。

**作者通过分析发现了一个关键现象**：不同模型架构中，域差距导致的"性能瓶颈层"位置不同且动态变化——在MSBDN中是编码器最后两个卷积块，在DEA中是编码器第三个块。**固定选择LoRA注入层是次优的**，需要一种模型无关的自动定位方法。

**这两个问题相互耦合**：有效的无监督损失是域适配的前提，而高效的参数微调是实用部署的需要。本文提出的H2C损失和BiLaLoRA策略分别解决这两个痛点。

## 方法详解

### 整体框架

以预训练去雾模型（如DEA）为基础，通过H2C损失+BiLaLoRA在真实雾天图像上做无监督域适配。分两阶段：(1) 双层定位——联合优化LoRA权重ω和门控参数α以排序候选层重要性；(2) LoRA微调——选取top-k层固定后仅优化ω。

### 关键设计

1. **H2C文本引导损失**:

    - 功能：在无配对清晰图的条件下为去雾模型提供无监督优化信号
    - 核心思路：利用CLIP图像/文本编码器，定义正向提示"a clear photo"和负向提示"a photo with haze"。计算去雾前后的图像特征差 $\Delta V_{img} = V_{out} - V_{in}$ 与文本方向差 $\Delta T_{text} = T_{pos} - T_{neg}$ 的余弦相似度作为损失：$L_{H2C} = 1 - \cos(\Delta V_{img}, \Delta T_{text})$
    - 设计动机：正负提示的协同约束保证去雾方向性——单用正向导致色彩失真，单用负向导致过度去雾。夜间场景通过修改提示为"nighttime haze"即可适配

2. **BiLaLoRA双层优化**:

    - 功能：自动搜索最佳LoRA注入层，无需人工选择
    - 核心思路：每个候选层配一个可学习门控参数 $\alpha$（sigmoid约束到(0,1)），通过双层优化联合学习层选择（上层：验证集最大化性能）和LoRA权重（下层：训练集最小化损失）。用rank-one外积近似Hessian简化超梯度计算为仅需一阶导数
    - 设计动机：域差距瓶颈层因模型架构和场景特性而异，单层优化无法捕获层选择和权重间的层级依赖关系

3. **即插即用多域适配**:

    - 功能：支持快速切换不同目标域（白天/夜间）
    - 核心思路：利用LoRA的天然特性，为不同场景各学一个轻量adapter，无需重新全量微调。白天500张+夜间100张真实雾图即可完成适配
    - 设计动机：实际去雾场景多样，需要灵活切换而非一个通用模型

### 损失函数 / 训练策略

预训练阶段用L1 loss（合成数据有GT）。域适配阶段仅用H2C loss（无GT）。LoRA rank=8, scaling=2, top-3层。lr=1e-6, Adam, 256×256裁剪+旋转翻转增强。

## 实验关键数据

### 主实验

| 数据集 | 指标 | BiLaLoRA | 之前SOTA | 提升 |
|--------|------|----------|---------|------|
| RTTS | FADE↓ | 0.752 | 0.845(PHATNet) | -11% |
| RTTS | MUSIQ↑ | 61.77 | 59.61(IPC) | +2.16 |
| URHI | MUSIQ↑ | 63.52 | 62.22(IPC) | +1.30 |
| Fattal | MUSIQ↑ | 67.92 | 67.58(IPC) | +0.34 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 全量微调 vs BiLaLoRA | 64.43 vs 64.40 | 性能几乎持平，但训练时间4.2h→0.94h（-77.7%） |
| 双层优化 vs 朴素联合 | 64.40 vs 64.07 | 双层验证集解耦层选择与权重，泛化更好 |
| 自动定位 vs 手动选层 | 64.40 vs 63.31 | 自动始终优于经验选择 |
| H2C去掉正向提示 | 色彩失真 | 显著影响输出颜色质量 |
| H2C去掉负向提示 | 过度去雾 | 模型过度"清理"导致伪影 |

### 关键发现

- 跨模型验证：在MSBDN/DeHamer/ConvIR/DEA四种架构上均有效，证明模型无关性
- 跨域验证：从ITS/OTS/Haze4K/RIDCP四种合成预训练出发均能显著提升
- 最佳层数k=3，超过3层边际收益递减，额外参数仅+3%

## 亮点与洞察

- CLIP作为无监督去雾损失的"裁判"是巧妙的跨模态应用——通过正负文本方向差定义语义轨迹，可推广到其他恢复任务。双层优化自动搜索LoRA层将PEFT从人工选层决策中解放出来。

## 局限与展望

- 无参考评估指标（FADE/MUSIQ）可靠性有限，缺少配对全参考评估
- 仅在去雾任务验证，未探索去雨/去噪等扩展
- top-k固定为3，不同任务/架构可能需不同k
- H2C文本提示设计较简单，更复杂的prompt engineering可能进一步提升

## 相关工作与启发

- **vs RIDCP (CVPR'23)**: 基于VQGAN先验，BiLaLoRA在MUSIQ上显著领先
- **vs IPC (CVPR'25)**: 迭代预测-critic码本解码，BiLaLoRA在整体指标上优于IPC
- **vs CoA (CVPR'25)**: 所有指标上胜出（FADE 0.638 vs 0.700，MUSIQ 64.40 vs 57.58）

## 评分

- 新颖性: ⭐⭐⭐⭐ H2C损失设计巧妙，双层优化定位LoRA层是有价值的新视角
- 实验充分度: ⭐⭐⭐⭐ 跨模型、跨域、消融实验全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，推导完整
- 价值: ⭐⭐⭐ 对真实图像去雾和PEFT有实用价值，但领域相对小众

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](referencefree_image_quality_assessment_for_virtual.md)

<!-- RELATED:END -->
