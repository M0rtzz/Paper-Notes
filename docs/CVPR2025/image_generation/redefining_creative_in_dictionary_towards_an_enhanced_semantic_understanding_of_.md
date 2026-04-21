---
title: >-
  [论文解读] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation
description: >-
  [CVPR 2025][图像生成][创意生成] CreTok 将"creative"重定义为一个可学习的通用 token `<CreTok>`，通过在文本嵌入空间持续迭代优化该 token 的语义，赋予扩散模型组合式创意生成的"元创造力"，无需额外训练即可零样本生成多样的概念混合图像，速度比 SOTA 快 10-30 倍。
tags:
  - CVPR 2025
  - 图像生成
  - 创意生成
  - 文本概念组合
  - token重定义
  - 元创造力
  - 扩散模型
---

# Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation

**会议**: CVPR 2025  
**arXiv**: [2410.24160](https://arxiv.org/abs/2410.24160)  
**代码**: https://github.com/fu-feng/CreTok  
**领域**: 扩散模型 / 图像生成  
**关键词**: 创意生成, 文本概念组合, token重定义, 元创造力, 扩散模型

## 一句话总结

CreTok 将"creative"重定义为一个可学习的通用 token `<CreTok>`，通过在文本嵌入空间持续迭代优化该 token 的语义，赋予扩散模型组合式创意生成的"元创造力"，无需额外训练即可零样本生成多样的概念混合图像，速度比 SOTA 快 10-30 倍。

## 研究背景与动机

**领域现状**：当前 T2I 扩散模型（如 SD3、DALL-E 3、Midjourney）擅长生成分布外图像（如"蓝色香蕉"），因为"蓝色"语义明确具体。但对于组合式创意生成（如"一个像生菜又像螳螂的创意混合物"），模型难以理解"creative"这个抽象概念。

**现有痛点**：现有创意生成方法依赖合成参考提示或参考图像。ConceptLab 为每个新概念训练独立 token；BASS 通过规则搜索大量候选图像；MagicMix 等用扩散过程中的语义插值。这些方法都需要为每次生成重新训练或大量计算（ConceptLab 120s/张、BASS 40s/张），缺乏实用性。

**核心矛盾**：扩散模型能理解具体形容词（"蓝色"），却不能理解抽象形容词（"creative"）。根本原因是"creative"在文本编码器中的嵌入缺乏指导组合生成的具体语义。

**本文目标**：让"creative"像"blue"一样成为一个语义清晰的形容词，直接修饰任何概念对来实现零样本组合创意。

**切入角度**：既然问题出在"creative"这个词的语义太抽象，那就用数据驱动的方式重新定义它。

**核心 idea**：将"creative"重定义为一个可学习 token `<CreTok>`，在大量文本对上迭代优化其嵌入，使其编码"如何组合两个概念"的元能力。

## 方法详解

### 整体框架

基于 Stable Diffusion 3，冻结所有扩散模型参数，仅在 CLIP 文本编码器的嵌入空间中优化一个新 token `<CreTok>` 的嵌入向量。训练时从 CangJie 数据集采样文本对，构建限制性提示和自适应提示，最小化二者嵌入的余弦距离。推理时在提示中用 `<CreTok>` 替换"creative"即可直接生成。

### 关键设计

1. **单文本对的概念融合**:

    - 功能：实现两个概念的 token 级语义融合
    - 核心思路：给定文本对 $(t_1, t_2)$（如 Lettuce, Mantis），构建限制性提示 $\mathcal{P}_r$ = "a lettuce mantis" 和自适应提示 $\mathcal{P}_a$ = "a photo of a `<CreTok>` mixture"。优化目标是让两个提示的 CLIP 文本嵌入余弦相似度最大化。为防过拟合引入阈值 $\theta=0.5$ 截断损失。同时对 $(t_1, t_2)$ 和 $(t_2, t_1)$ 两种顺序都计算损失，避免顺序偏差
    - 设计动机：直接在文本嵌入空间操作而非在扩散过程中操作，计算量极小且不改变生成模型参数。阈值 $\theta$ 平衡了概念融合度和过拟合风险

2. **持续迭代精炼 `<CreTok>`**:

    - 功能：让 `<CreTok>` 学到通用的"如何做创意组合"的元能力，而非某个特定概念
    - 核心思路：在 CangJie 数据集（200 个训练文本对）上迭代训练。每步随机采样 n=16 个文本对，计算累积损失 $\mathcal{L}_{iter} = \frac{1}{n}\sum_{i=1}^{n}\tilde{\mathcal{L}}_{mix}^i$，更新 `<CreTok>` 嵌入。每步采样不同文本对确保泛化性
    - 设计动机：如果只在一个文本对上训练，token 会编码特定概念语义（如特定的"生菜-螳螂"混合体）；通过在大量不同对上迭代，token 逐渐从"学某个具体混合"转变为"学如何做混合"的元能力

3. **CangJie 数据集**:

    - 功能：提供多样的文本对训练素材
    - 核心思路：从动物、植物等类别中组合概念对，200 个训练对 + 27 个源自 BASS 的测试对
    - 设计动机：数据集多样性保证 `<CreTok>` 的泛化能力，训练完后能处理从未见过的概念对

### 损失函数 / 训练策略

- 损失函数：带阈值的余弦相似度损失 $\tilde{\mathcal{L}}_{mix} = 1 - \min[\cos(E(\mathcal{P}_r), E(\mathcal{P}_a)), \theta]$
- 双向顺序训练避免文本对的位置偏差
- 训练配置：10K 步，单卡 4090，LR=0.01 + cosine scheduler，batch=1 + gradient accumulation 16 步，约 30 分钟完成
- 仅优化 `<CreTok>` 的嵌入向量，不动扩散模型、文本编码器的任何参数
- 无图像参与训练过程（image-free），纯文本嵌入空间优化

## 实验关键数据

### 主实验

| 方法 | VQAScore↑ | PickScore↑ | ImageReward↑ | 生成速度 |
|------|-----------|------------|--------------|---------|
| CreTok | **0.835** | **21.775** | **1.065** | 4s/张 |
| SD 3.5 | 0.805 | 21.766 | 0.881 | - |
| Kandinsky 3 | 0.771 | 21.637 | 0.634 | - |
| BASS | 0.710 | 20.799 | 0.481 | 40s/张 |
| ConceptLab | - | - | - | 120s/张 |

| GPT-4o 评分 | 集成度 | 对齐度 | 原创性 | 美学 | 综合 |
|------------|--------|--------|--------|------|------|
| CreTok | **9.5** | **9.9** | **9.3** | **9.6** | **9.6** |
| SD 3.5 | 9.1 | 9.9 | 9.1 | 9.4 | 9.4 |
| BASS | 8.9 | 9.3 | 8.7 | 8.3 | 8.8 |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| θ=0.3 (低阈值) | 两个概念各自独立生成，未融合 |
| θ=0.5 (本文选择) | 最佳平衡，概念融合且不过拟合 |
| θ=0.7 (高阈值) | 过拟合到某一概念 |
| 训练 2K 步 | `<CreTok>` 主要吸收单一概念语义 |
| 训练 10K 步 | `<CreTok>` 学到泛化的创意元能力 |

### 关键发现

- `<CreTok>` 在未见过的概念对上表现良好（如训练时未出现的 Lettuce-Mantis 组合），证明了元创造力的泛化性
- 在人类偏好指标（PickScore、ImageReward）上超越 SD 3.5、DALL-E 3 等更大更强的模型
- 可无缝扩展到 3+概念融合和无参考概念的创意生成（CT2I 任务）
- 可自由搭配风格提示（如"油画"、"水彩"），ConceptLab 和 BASS 做不到
- 用户研究中平均排名 1.9，显著优于其他方法（3.1-3.4）

## 亮点与洞察

- **将抽象形容词重定义为可学习 token 的思路极具通用性**：不仅可以应用于"creative"，理论上任何语义模糊的形容词（如"beautiful"、"scary"）都可以用类似方法增强模型理解。是一种全新的模型能力增强范式
- **纯文本空间优化，不碰扩散模型参数**：30 分钟训练、4 秒推理即可获得当前最好的创意生成能力，工程实用性极强
- **元创造力 vs 静态创造力的区分**很有启发：之前方法为每个创意输出单独训练 token，本文学的是"如何做创意"的通用能力
- 与个性化方法的本质区别：Textual Inversion 等表示"什么"，`<CreTok>` 表示"如何"

## 局限与展望

- 依赖 CLIP 的文本编码能力，如果两个概念在 CLIP 空间距离太远，融合效果可能受限
- 仅在组合式创意（TP2O）任务上验证，对更开放的创意场景（如风格创新、构图创新）未探索
- CangJie 数据集主要由动植物类别组成，概念多样性有限
- 仅在 SD 3 上验证，未在其他基座模型上测试泛化性
- 阈值 $\theta=0.5$ 的选择依赖经验，不同基座模型可能需要调整
- 未来可探索多个 `<CreTok>` 变体、可控创意程度的连续化表示

## 相关工作与启发

- **vs ConceptLab**: 为每个新概念训练独立 token（每次 120s），本文训练一次后零样本应用所有概念对（4s）
- **vs BASS**: 通过规则搜索+大量候选过滤来获得创意图像（40s），本文直接生成，不依赖搜索
- **vs MagicMix/DiffMorpher**: 在扩散过程中做插值，严重依赖参考图像且不够灵活，本文无需参考图像
- **vs Textual Inversion**: TI 将特定视觉概念编码为 token，本文将抽象语义能力编码为 token，目标不同
- 启发：**抽象概念的 token 化**可能是增强基础模型能力的通用策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将"创意"这一抽象概念精确编码为可训练 token 的思路非常新颖，元创造力概念有启发性
- 实验充分度: ⭐⭐⭐⭐ 多维评估（自动指标+GPT-4o+用户研究），消融完备
- 写作质量: ⭐⭐⭐⭐ 故事讲得好（"blue banana" 类比），概念清晰
- 价值: ⭐⭐⭐⭐ 方法思路通用性强，工程落地成本极低

<!-- RELATED:START -->

## 相关论文

- [DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [CAP: Evaluation of Persuasive and Creative Image Generation](../../ICCV2025/image_generation/cap_evaluation_of_persuasive_and_creative_image_generation.md)
- [Decomate: Leveraging Generative Models for Co-Creative SVG Animation](../../NeurIPS2025/image_generation/decomate_leveraging_generative_models_for_co-creative_svg_animation.md)
- [ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement](../../CVPR2026/image_generation/showtable_unlocking_creative_table_visualization_with_collaborative_reflection_a.md)
- [EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)

<!-- RELATED:END -->
