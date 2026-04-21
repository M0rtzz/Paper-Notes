---
title: >-
  [论文解读] ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement
description: >-
  [ECCV 2024][图像生成][颜色提示学习] 提出ColorPeel方法，通过在目标颜色的基本几何形状上学习颜色提示token（解耦颜色与形状），并引入交叉注意力对齐损失，使T2I扩散模型能精确生成用户指定RGB颜色的物体。
tags:
  - ECCV 2024
  - 图像生成
  - 颜色提示学习
  - 扩散模型
  - 颜色-形状解耦
  - T2I个性化
  - 交叉注意力对齐
---

# ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement

**会议**: ECCV 2024  
**arXiv**: [2407.07197](https://arxiv.org/abs/2407.07197)  
**代码**: https://moatifbutt.github.io/colorpeel/ (项目页面)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 颜色提示学习, 扩散模型, 颜色-形状解耦, T2I个性化, 交叉注意力对齐

## 一句话总结

提出ColorPeel方法，通过在目标颜色的基本几何形状上学习颜色提示token（解耦颜色与形状），并引入交叉注意力对齐损失，使T2I扩散模型能精确生成用户指定RGB颜色的物体。

## 研究背景与动机

**领域现状**：文本到图像（T2I）扩散模型通过文本提示生成图像的能力已非常强大，但在颜色控制方面存在根本性的精度缺陷。当前T2I模型只能通过语言颜色名称（如"红色"、"米色"）来指定颜色。

**现有痛点**：语言对颜色的表达是**离散的**，而颜色本身是**连续的**——一个"红色"可以对应数百种不同的RGB值。即使使用更精细的颜色名称如"beige"或"light green"，生成结果与用户脑中的确切颜色往往大相径庭。这在设计、时尚和艺术领域是严重问题，用户需要生成精确匹配色板的颜色。

**核心矛盾**：现有T2I个性化方法（如Textual Inversion、DreamBooth、Custom Diffusion）虽然能学习新概念，但在学习颜色时会产生**颜色-形状纠缠**——从一个颜色均匀的色块学习颜色token时，模型会同时记忆色块的形状，导致颜色无法迁移到其他物体上。直接在文本提示中输入RGB数值同样效果不佳。

**本文切入角度**：颜色是一种抽象属性，不能从单一形状中学习。如果在**多种不同几何形状**上呈现同一目标颜色，模型自然能将颜色这一"共有属性"从形状中"剥离"出来。

**核心idea**：生成一组目标颜色的基本几何物体（2D/3D形状），联合学习颜色token和形状token，通过交叉注意力对齐损失进一步强制解耦。

## 方法详解

### 整体框架

ColorPeel基于Stable Diffusion v1.4，流程如下：(1) 给定用户选择的RGB颜色，自动生成一组该颜色的基本几何形状图像（2D圆形/方形/六边形/三角形，或3D球体/圆柱/立方体/锥体等）；(2) 为颜色和形状分别引入可学习token $c^*$ 和 $s^*$，使用含解耦标注的文本模板（如"A photo of $s_i^*$ filled with $c^*$"）进行训练；(3) 学习完成后，颜色token $c^*$ 可用于生成任意物体的精确颜色。

### 关键设计

1. **颜色-形状解耦的训练数据构造**：为每个目标颜色自动生成至少两种不同形状的图像。解耦的核心逻辑是：当多张训练图像中颜色相同but形状不同时，颜色token $c^*$ 被迫只编码颜色信息（因为形状在变化），而形状token $s_i^*$ 则编码各自的形状。3D形状由于包含光照、阴影等物理效果，生成的颜色提示更贴近真实场景。

   设计动机：从单一色块学习颜色必然导致纠缠（如Custom Diffusion的失败案例），多形状训练是解耦的关键。

2. **交叉注意力对齐损失 (Cross-Attention Alignment, CAA)**：通过可视化SD-UNet的交叉注意力图发现，颜色token和形状token的注意力区域经常不对齐——颜色注意力泄漏到背景区域，导致颜色不准确。CAA损失通过最大化颜色和形状注意力图的余弦相似度来强制对齐：

   $$\mathcal{L}_{caa} = 1 - \cos(\mathcal{A}_t^{c^*}, \mathcal{A}_t^{s^*})$$

   最终训练目标：
   $$\mathcal{V}^* = \underset{\mathcal{V}}{\arg\min}\ \mathbb{E}[\mathcal{L}_{rec} + \lambda \cdot \mathcal{L}_{caa}]$$

   其中 $\mathcal{L}_{rec}$ 为标准LDM噪声重建损失，$\lambda$ 为权衡超参数（最优值0.2）。

   设计动机：注意力泄漏是颜色不精确的根本原因之一，CAA直接在注意力层面解决颜色-形状的空间对齐问题。

3. **灵活的学习框架**：ColorPeel可兼容多种T2I适配方法——作为Custom Diffusion的增强（优化key/value投影矩阵+token），也可与DreamBooth等方法结合。训练时同时优化颜色token嵌入 $\mathcal{V}^{c^*}$ 和形状token嵌入 $\mathcal{V}^{s^*}$。

### 损失函数 / 训练策略

- **粗粒度颜色学习**：1500步训练
- **细粒度颜色学习**：6000步训练
- 批量大小为2，学习率 $10^{-5}$
- 在A40 GPU上训练时间约19分钟（Custom Diffusion为24分钟，对比也更优）
- 评估指标：CIE Lab色差 ($\Delta E$)、sRGB中的平均角度误差 (MAE)、色调 (Hue) 中的MAE

## 实验关键数据

### 主实验

| 方法 | $\Delta E$ ↓ | $\Delta E_{ch}$ ↓ | MAE(rgb) 10% ↓ | MAE(rgb) 50% ↓ | MAE(Hue) 10% ↓ | MAE(Hue) 50% ↓ | 时间(min) |
|------|-------------|-------------------|----------------|----------------|----------------|----------------|-----------|
| Stable Diffusion | 47.45 | 41.55 | 12.89 | 20.04 | 54.14 | 86.38 | - |
| Rich-Text | 36.62 | 32.48 | 9.91 | 13.29 | 50.55 | 72.77 | - |
| Textual Inversion | 48.98 | 44.29 | 15.22 | 19.51 | 52.66 | 69.35 | 118 |
| DreamBooth | 50.71 | 46.29 | 14.75 | 19.30 | 47.12 | 67.13 | 56 |
| Custom Diffusion | 48.47 | 42.23 | 13.43 | 17.93 | 31.63 | 55.07 | 24 |
| **ColorPeel (3D)** | **21.39** | **16.51** | **4.36** | **7.76** | **2.63** | **6.47** | **19** |
| ColorPeel (2D) | 20.45 | 15.29 | 4.83 | 7.88 | 3.18 | 7.43 | - |

ColorPeel在所有指标上均远超现有方法，$\Delta E$ 误差降低约55%（21.39 vs. 48.47），Hue MAE从31.63大幅降低至2.63。

### 消融实验

| $\lambda$ (CAA权重) | $\Delta E$ ↓ | $\Delta E_{Ch}$ ↓ | MAE(rgb) 10% ↓ | MAE(Hue) 10% ↓ | 说明 |
|---|---|---|---|---|---|
| 0.0 (退化为CD) | 48.47 | 42.23 | 13.43 | 31.63 | 无CAA，颜色形状纠缠 |
| 0.1 | 22.23 | 16.86 | 5.13 | 3.48 | 已显著改善 |
| **0.2** | **21.39** | **16.51** | **4.36** | **2.63** | **最优** |
| 0.4 | 23.37 | 17.10 | 4.91 | 3.87 | 略有退化 |
| 0.8 | 23.79 | 17.01 | 4.98 | 4.06 | 过强约束 |
| 1.0 | 24.43 | 18.64 | 5.03 | 4.27 | 过度约束 |

移除CAA损失（$\lambda=0$）时模型退化为Custom Diffusion，无法解耦颜色和形状。$\lambda=0.2$ 为最优平衡点。

### 关键发现

- **用户实验**：15名参与者的2AFC实验（Thurstone Case V模型分析），ColorPeel在统计上显著优于CD、DB、Rich-Text和TI（z-score最高，95%置信区间不重叠）
- **2D vs 3D形状**：3D形状因包含光照阴影效果，在Hue精度上略优，但2D形状在ΔE上更好。两者都远超基线
- **细粒度颜色**：能够有效区分navy/indigo/cyan等相近颜色，生成高质量差异化结果
- **泛化能力**：方法可直接扩展到纹理学习和材质学习（将纹理/材质图贴到3D形状表面）
- **颜色插值**：学习到的颜色token支持线性插值，可连续生成中间颜色，无需额外训练

## 亮点与洞察

- 问题定义精准：将"颜色提示学习"明确为一个新任务，切中设计/创作领域的真实需求
- 解决方案极为直觉——"在不同形状上展示同一颜色"的解耦思路简洁优雅，且有认知科学类比
- CAA损失通过注意力层面的对齐解决了颜色泄漏问题，具有很好的可解释性
- 完全不需要真实世界图像采集，训练数据由Blender的几何形状自动合成，零标注成本
- 泛化到纹理/材质学习展示了方法的普适性，颜色只是"抽象属性学习"的一个实例

## 局限与展望

- 基于SD v1.4，生成分辨率受限（512×512），可探索更新骨干如SDXL
- 目前仅学习单一颜色token，多颜色组合（如渐变色）的精确控制尚未充分探索
- 3D形状的光照效果可能引入颜色偏移（luminance影响），可探索更好的颜色空间
- 颜色token插值目前是简单线性插值，是否存在更好的颜色空间映射
- 可扩展到视频生成中的颜色一致性控制

## 相关工作与启发

- Break-a-Scene从单张图中分解多概念，但无法确保抽象概念（颜色）与具体概念（形状）的洁净分离
- Textual Inversion只学习token不微调模型，但缺乏解耦机制导致颜色不准确
- CAA损失灵感来自DPL（DPL最小化不同物体注意力的重叠），本文反向使用——最大化颜色与形状注意力的重叠
- 启发：抽象属性学习（风格、材质、光照）都可以用类似的"多实例解耦"范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次提出颜色提示学习任务，解耦方案和CAA损失均为创新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 定量指标+用户实验+消融+细粒度+泛化+插值，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，Fig.2的失败案例分析直观，定量指标设计合理
- 价值: ⭐⭐⭐⭐ 实用性强，训练成本低（19分钟），对设计和创意领域有直接价值

<!-- RELATED:START -->

## 相关论文

- [Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)

<!-- RELATED:END -->
