---
title: >-
  [论文解读] Text to Sketch Generation with Multi-Styles
description: >-
  [NeurIPS 2025][图像生成][草图生成] 提出M3S（Multi-Style Sketch Synthesis），一个无训练框架，通过线性平滑的K/V特征注入、联合AdaIN风格倾向控制和风格-内容分离引导，实现基于文本提示和参考风格草图的单/多风格草图生成。 - 草图作为跨语言视觉媒介，从工业原型到艺术表达应用…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "草图生成"
  - "多风格合成"
  - "扩散模型"
  - "K/V注入"
  - "AdaIN"
---

# Text to Sketch Generation with Multi-Styles

**会议**: NeurIPS 2025  
**arXiv**: [2511.04123](https://arxiv.org/abs/2511.04123)  
**代码**: [GitHub](https://github.com/CMACH508/M3S)  
**领域**: 图像生成, 风格迁移, 草图合成  
**关键词**: 草图生成, 多风格合成, 扩散模型, K/V注入, AdaIN

## 一句话总结
提出M3S（Multi-Style Sketch Synthesis），一个无训练框架，通过线性平滑的K/V特征注入、联合AdaIN风格倾向控制和风格-内容分离引导，实现基于文本提示和参考风格草图的单/多风格草图生成。

## 研究背景与动机
- 草图作为跨语言视觉媒介，从工业原型到艺术表达应用广泛
- 高质量草图数据集稀缺（需专业技能+大量时间），限制了模型训练
- 现有方法局限：
    - CLIPasso/DiffSketcher：缺乏对风格属性的精确控制
    - K/V替换方法(MasaCtrl等)：在跨域场景下Q与替换的K/V不对齐，导致内容泄露和结构不连贯
    - StyleAligned：通过AdaIN对齐统计分布，但在草图等结构差异大的情境下效果差
- 文本条件的风格控制缺乏表达力——无法精确匹配特定风格

## 方法详解

### 整体框架
基于Stable Diffusion v1.5/SDXL的无训练框架，支持单风格和多风格草图生成

### 关键设计

#### 1. 风格特征注入
**拒绝直接替换**：$Attention(Q_{tar}, K_{ref}, V_{ref})$在跨域场景下引入结构不连贯

**拒绝AdaIN对齐**：$Q_{tar} = AdaIN(Q_{tar}, Q_{ref})$ 对草图生成有害

**M3S方案：线性平滑的特征拼接**
$$Attention\left(Q_{tar}, \begin{bmatrix}K_{tar}\\\hat{K}_{ref}\end{bmatrix}, \begin{bmatrix}V_{tar}\\\hat{V}_{ref}\end{bmatrix}\right)$$
$$\hat{K}_{ref} = \lambda K_{tar} + (1-\lambda)K_{ref}, \quad \hat{V}_{ref} = \lambda V_{tar} + (1-\lambda)V_{ref}$$
- $\lambda \in [0,1]$控制内容保真与风格一致性的平衡
- 增大$\lambda$增强美观和文本对齐，但过大可能导致风格退化

#### 2. 多风格倾向控制（Joint AdaIN）
$$z_t^{tar} = \eta \cdot AdaIN(z_t^{tar}, z_t^{ref_1}) + (1-\eta) \cdot AdaIN(z_t^{tar}, z_t^{ref_2})$$
- $\eta \in [0,1]$：风格倾向参数
- 直觉：稠密笔画草图→低均值→AdaIN偏向详细输出；稀疏草图→高均值→极简结果
- 即使在$\eta=0$或$\eta=1$时，由于自注意力包含两个风格特征，仍保持多风格特征

#### 3. 风格-内容分离引导
$$\tilde{\epsilon}_t = \epsilon_\theta(z_t^{tar}, t, \emptyset) + \omega_1 \cdot \underbrace{[\epsilon_\theta^\times(\cdot, text, K_{ref}, V_{ref}) - \epsilon_\theta(\cdot, \emptyset)]}_{\text{内容引导}} + \omega_2 \cdot \underbrace{[\epsilon_\theta^\times(\cdot, \emptyset, K_{ref}, V_{ref}) - \epsilon_\theta(\cdot, \emptyset)]}_{\text{风格引导}}$$
- $\omega_1, \omega_2$分别控制内容和风格引导强度
- $\omega_2$从$\omega_2/3$线性增加到$\omega_2$（去噪过程中逐渐加强风格）

#### 4. 基于轮廓的正则引导（SD v1.5）
- 对去噪潜在表征用Tweedie公式估计$z_{0|t}^{tar}$，解码为图像
- 应用Sobel算子提取方向梯度，最大化边缘响应
- $\mathcal{L}_{edge} = -|grad_x| - |grad_y|$
- 抑制抽象草图中的伪影

## 实验关键数据

### 主实验：6种风格的定量对比

| 方法 | 实现 | Style1 CLIP-T↑ | Style1 DINO↑ | Style1 VGG↓ | Style5 CLIP-T↑ | Style5 DINO↑ |
|-----|------|---------------|-------------|------------|---------------|-------------|
| StyleAligned | - | 0.3130 | 0.6691 | 0.0308 | 0.3004 | 0.5428 |
| AttentionDistill | - | 0.3305 | **0.7738** | 0.0930 | 0.3377 | **0.6221** |
| InstantStyle | - | **0.3512** | 0.4934 | 0.0417 | **0.3480** | 0.4408 |
| CSGO | - | 0.3336 | 0.5276 | 0.0571 | 0.3298 | 0.4288 |
| **M3S (SD v1.5)** | - | 0.3507 | 0.6383 | **0.0200** | 0.3494 | 0.5777 |
| **M3S (SDXL)** | - | **0.3607** | 0.6545 | 0.0165 | 0.3467 | 0.5332 |

### 人类偏好评分（1-8分）

| 方法 | 平均评分 |
|-----|---------|
| StyleAligned | 2.77 |
| CSGO | 3.83 |
| StyleStudio | 4.22 |
| AttentionDistill | 4.28 |
| InstantStyle | 5.08 |
| **M3S (SD v1.5)** | **5.44** |
| **M3S (SDXL)** | **6.19** |

### 多风格生成（$\eta$控制验证）

| $\eta$ | DINO-ref1↑ | DINO-ref2↑ | CLIP-T↑ |
|--------|-----------|-----------|---------|
| 0 | 0.3936 | **0.4944** | 0.3442 |
| 0.25 | 0.4180 | 0.4821 | 0.3514 |
| 0.5 | 0.4408 | 0.4556 | 0.3495 |
| 0.75 | 0.4578 | 0.4221 | 0.3499 |
| 1.0 | **0.4693** | 0.3975 | 0.3470 |

### 关键发现
- M3S(SDXL)在人类评分中以6.19分遥遥领先所有基线
- 线性平滑($\lambda$)有效减少内容泄露——对比直接替换和AdaIN对齐
- AttentionDistillation虽DINO分数高但存在严重内容泄露（参考图像的内容混入目标）
- 多风格$\eta$参数可靠地控制风格倾向：DINO-ref1随$\eta$增加单调递增，DINO-ref2单调递减
- 即使$\eta=0$或$\eta=1$，结果仍保持两种风格特征（因自注意力中包含两组K/V）

## 亮点与洞察
1. **无训练框架**：直接在预训练扩散模型上操作，无需微调
2. **简洁优雅的设计**：线性平滑替代复杂的特征对齐/蒸馏
3. **可控多风格**：首次实现草图的多风格融合和连续风格插值
4. **跨域鲁棒性**：在参考与目标结构差异大时仍保持质量（这是K/V替换方法的痛点）
5. **双平台支持**：同时在SD v1.5和SDXL上验证

## 局限与展望
- $\omega_1, \omega_2, \lambda$需要按风格类型手动调整（如Style 6的抽象草图需不同参数）
- SD v1.5上的抽象草图可能产生伪影（需轮廓正则引导）
- 基础模型在自然图像上训练→过高$\lambda$可能导致自然风格输出
- 100步DDIM采样速度较慢
- 未探索视频/动画草图生成

## 相关工作与启发
- DiffSketcher：文本驱动草图合成先驱，但缺乏风格控制
- Cross-image/MasaCtrl：K/V替换范式——本文分析其跨域失败原因
- B-LoRA/IP-Adapter/CSGO：风格适配器方法——主要针对自然图像
- AdaIN(Huang 2017)：实时任意风格迁移的经典方法——本文将其扩展到多风格草图控制

## 评分
- 新颖性：⭐⭐⭐⭐ （多风格草图生成的首次探索+简洁方法设计）
- 技术深度：⭐⭐⭐⭐ （风格注入机制分析深入，消融全面）
- 实验充分性：⭐⭐⭐⭐⭐ （6种风格×8种基线+多风格+人类评估）
- 写作质量：⭐⭐⭐⭐ （可视化丰富，对比清晰）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](../../CVPR2025/image_generation/flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](../../CVPR2025/image_generation/compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](../../CVPR2025/image_generation/image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[ECCV 2024\] Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](../../ECCV2024/image_generation/be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)
- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)

</div>

<!-- RELATED:END -->
