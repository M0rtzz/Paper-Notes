---
title: >-
  [论文解读] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
description: >-
  [ECCV 2024][图像生成] TextDiffuser-2 利用两个语言模型分别进行布局规划和布局编码，实现了更灵活、更自动化、更多样化的视觉文本渲染，在保持文本准确性的同时大幅提升了字体风格多样性。 扩散模型在图像合成领域取得了巨大成功，但在视觉文本渲染方面仍面临挑战。现有方法存在三大缺陷： 1. 灵活性和自动化不足…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering

**会议**: ECCV 2024  
**arXiv**: [2311.16465](https://arxiv.org/abs/2311.16465)  
**领域**: 图像生成

## 一句话总结

TextDiffuser-2 利用两个语言模型分别进行布局规划和布局编码，实现了更灵活、更自动化、更多样化的视觉文本渲染，在保持文本准确性的同时大幅提升了字体风格多样性。

## 研究背景与动机

扩散模型在图像合成领域取得了巨大成功，但在视觉文本渲染方面仍面临挑战。现有方法存在三大缺陷：

1. **灵活性和自动化不足**：GlyphControl 需要用户手动设计字形图像，GlyphDraw 和 TextDiffuser 需要手动指定关键词，无法直接从自然语言提示生成对应图像。
2. **布局预测能力受限**：GlyphDraw 只能渲染单行文本，TextDiffuser 的 Layout Transformer 生成的布局不够美观。
3. **字体风格多样性受限**：TextDiffuser 使用字符级分割掩码作为控制信号，隐式约束了字符位置，限制了手写体或艺术字体的生成。

这些问题的根源在于现有方法使用了过于严格的字符级控制信号，缺乏灵活的布局规划能力。TextDiffuser-2 旨在释放语言模型在文本渲染中的潜力，解决上述问题。

## 方法详解

### 整体框架

TextDiffuser-2 采用两阶段训练架构，核心是两个语言模型：
- **语言模型 M1**（布局规划器）：基于 Vicuna-7B 微调，将用户提示转换为语言格式的布局
- **语言模型 M2**（布局编码器）：扩散模型内的 CLIP 文本编码器，编码行级文本位置和内容信息

### 关键设计

**1. 语言模型进行布局规划（M1）**

利用 MARIO-10M 数据集中的 caption-OCR 对微调 Vicuna-7B，使其成为布局规划器：

- 支持两种模式：(a) 用户不提供关键词时，自动推断文本内容和布局；(b) 用户提供关键词时，只需确定对应的布局位置
- 输出格式为 "textline x0, y0, x1, y1"，坐标归一化到 0~128 范围
- 通过多轮对话支持布局修改（重新生成、添加、移动关键词）
- 仅需 5k 数据即可达到最优微调效果

**2. 语言模型进行布局编码（M2）**

在扩散模型内利用 CLIP 文本编码器编码行级布局信息：

- **混合粒度分词策略**：对 prompt 保持原始 BPE 分词，对关键词引入字符级分词（如 "WILD" → "[W]", "[I]", "[L]", "[D]"）
- 引入 256 个坐标 token 和 95 个字符 token 编码位置和内容
- 行级边界框提供更灵活的生成控制，不限制字体风格多样性
- 最大序列长度 L 设为 128，覆盖 94% 训练样本

**3. 模型容量**

整体模型基于 SD 1.5，包含 922M 参数，输入图像尺寸为 512×512。

### 损失函数

- **阶段一**（布局规划）：交叉熵损失训练 M1，同时训练有/无关键词场景
- **阶段二**（图像生成）：L2 去噪损失训练 M2 和 U-Net

$$\mathcal{L}_{denoise} = \mathbb{E}_{z_0, \epsilon, t} \| \epsilon - \epsilon_\theta(z_t, c, t) \|^2$$

## 实验关键数据

### 主实验

在 MARIO-Eval 基准上的定量结果和用户研究：

| 指标 | SD-XL | PixArt-α | GlyphControl | TextDiffuser | TextDiffuser-2 |
|------|-------|----------|--------------|--------------|----------------|
| FID↓ | 62.54 | 87.09 | 50.82 | 38.76 | **33.66** |
| CLIPScore↑ | 31.31 | 27.88 | **34.56** | 34.36 | 34.50 |
| OCR Accuracy↑ | 0.31 | 0.02 | 32.56 | **56.09** | 57.58 |
| OCR F-measure↑ | 3.66 | 0.03 | 64.07 | **78.24** | 75.06 |
| 文本质量(人类)↑ | 14.58 | 3.65 | 21.35 | 23.44 | **36.98** |
| 文本-图像匹配(人类)↑ | 7.14 | 3.30 | 29.67 | 19.23 | **40.66** |

TextDiffuser-2 在 FID、OCR 准确率、用户研究等大多数指标上取得最佳结果。

### 消融实验

**微调数据量消融（M1 布局规划器）**：

| 数据量 | Accuracy↑ | Precision↑ | Recall↑ | F-measure↑ | IoU↓ |
|--------|-----------|------------|---------|------------|------|
| 0k-2shot | 49.65 | 84.18 | 69.69 | 76.25 | 19.69 |
| 2.5k | 61.10 | 82.20 | 85.18 | 83.67 | 3.21 |
| **5k** | **64.85** | **84.98** | **86.38** | **85.67** | **3.25** |
| 10k | 64.85 | 84.38 | 86.23 | 85.29 | 4.27 |
| 100k | 62.87 | 85.26 | 85.98 | 85.62 | 4.31 |

5k 数据即可达到最优性能，更多数据反而无明显提升。

**坐标表示和分词粒度消融**：

| 表示方式 | Accuracy↑ | Precision↑ | Recall↑ | F-measure↑ |
|----------|-----------|------------|---------|------------|
| Center (Char) | 35.19 | 61.75 | 62.71 | 62.23 |
| LT (Char) | 28.32 | 54.94 | 55.64 | 55.29 |
| LT+RB (Subword) | 15.48 | 41.74 | 42.53 | 42.13 |
| **LT+RB (Char)** | **57.58** | **74.02** | **76.14** | **75.06** |

使用左上+右下角点 + 字符级分词效果最佳，子词级分词准确率下降 42.1%。

### 关键发现

1. 语言模型具有自主推断关键词的灵活性，如自动纠正拼写错误（"RRAINBOW" → "RAINBOW"）
2. 行级引导比字符级引导产生更多样化的字体风格，但需要接受一定的准确率折中
3. 通过多轮对话可以灵活操控布局，支持重新生成、添加或移动关键词
4. TextDiffuser-2 对重叠框具有更强的鲁棒性

## 亮点与洞察

- **极少数据微调**：仅需 5k caption-OCR 对即可将 7B 语言模型训练为高质量布局规划器，展现了 LLM 跨领域迁移的强大能力
- **控制粒度的权衡**：从字符级→行级控制信号，用少量准确率换取了显著的风格多样性提升，是一个精妙的设计权衡
- **可交互布局编辑**：基于聊天模型微调的布局规划器天然支持多轮对话修改布局，提升了实用性
- **混合粒度分词**：prompt 用 BPE、关键词用字符级分词的混合策略兼顾了效率和拼写能力

## 局限性

1. 无法渲染复杂语言（如中文），因字符集庞大导致 few-shot 甚至 zero-shot 场景
2. 生成分辨率受限于 512×512
3. 行级控制虽提升了多样性，但在需要精确字符对齐的场景下准确率略低于字符级方法

## 评分

- **创新性**: ★★★★☆ — 双语言模型架构设计新颖，混合粒度分词策略巧妙
- **实用性**: ★★★★★ — 自动化程度高，支持多轮交互编辑
- **实验充分度**: ★★★★★ — 消融实验全面，包含人类和 GPT-4V 用户研究
- **写作质量**: ★★★★☆ — 结构清晰，动机阐述充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)

</div>

<!-- RELATED:END -->
