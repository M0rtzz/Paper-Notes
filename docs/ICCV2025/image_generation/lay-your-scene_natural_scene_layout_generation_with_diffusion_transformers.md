---
title: >-
  [论文解读] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers
description: >-
  [图像生成] 提出 LayouSyn，基于轻量开源语言模型提取场景元素、结合宽高比感知扩散 Transformer 的开放词汇文本到布局生成流水线，在空间推理和数量推理基准上达到 SOTA。
tags:
  - 图像生成
---

# Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers

> **会议**: ICCV 2025
> **arXiv**: [2505.04718](https://arxiv.org/abs/2505.04718)
> **领域**: 布局生成·扩散模型·可控图像生成
> **关键词**: scene layout generation, diffusion transformer, open-vocabulary, spatial reasoning, text-to-layout

## 一句话总结

提出 LayouSyn，基于轻量开源语言模型提取场景元素、结合宽高比感知扩散 Transformer 的开放词汇文本到布局生成流水线，在空间推理和数量推理基准上达到 SOTA。

## 研究背景与动机

可控图像生成（如 GLIGEN、Instance Diffusion）依赖用户提供精细的场景布局（bounding boxes），但手动标注布局耗时且成本高昂。自动文本到布局（T2L）生成面临以下挑战：

**封闭词汇的局限**：传统布局生成方法（LayoutGAN、LayoutVAE、BLT 等）假定固定的目标类别集合，难以处理自然语言描述的开放词汇场景。

**依赖闭源 LLM**：LayoutGPT、LLM Blueprint 等方法使用 GPT 等商业模型进行布局生成，不透明、高延迟、高成本，且常生成不真实的 bounding box（如不合理的宽高比、不自然的位置）。

**文档布局 vs 场景布局**：大多数布局扩散模型专为文档布局设计和评估，与自然场景布局有本质差异。

**LayouSyn 的解决方案**：将 T2L 任务分解为两步——(1) 轻量开源 LLM（Llama-3.1-8B）提取场景元素描述集；(2) 新型宽高比感知扩散 Transformer 以开放词汇方式生成条件布局。

## 方法详解

### 整体流水线（Fig. 2）

**阶段 1：描述集生成**
- 使用 Llama-3.1-8B 从文本 prompt 提取名词短语、分配数量、过滤不可视化名词
- 输出描述集 $\mathcal{D} = \{d_i\}_{i=1}^N$（如 "teapot: 1, food: 1, plate: 1"）

**阶段 2：条件布局生成**
- 扩散模型在 bounding box 坐标连续空间中运行
- 条件：全局条件（文本 prompt $p$）+ 局部条件（描述集 $\mathcal{D}$）+ 标量条件（宽高比 $r$、时间步 $t$）

### 关键设计 1：Layout Diffusion Transformer (LDiT)

修改标准 DiT block，增加描述 token 与全局 prompt 之间的交叉注意力：

- Bounding box $b_i \in \mathbb{R}^4$ 通过 MLP 编码为 $d_{model}$ 维 token
- 描述 $d_i$ 通过 T5-sentence encoder + MLP 编码为 $d_{model}$ 维 token
- 两者拼接后送入 Transformer blocks
- **新增交叉注意力层**：全局文本嵌入（Google-T5 编码）与描述/bbox token 做交叉注意力，改善全局-局部条件对齐
- 宽高比 $r$ 和时间步 $t$ 通过 adaLN 编码为标量条件

坐标归一化：bbox 坐标按布局尺寸归一化并缩放至 $[-1, 1]$，实现宽高比无关的表示。

### 关键设计 2：噪声调度缩放

低维 bbox 坐标在标准噪声调度下信息破坏过快。引入缩放因子 $s$：

$$\bar{\alpha}'_t = \frac{\bar{\alpha}_t \cdot s^2}{1 + (\bar{\alpha}_t \cdot (s^2 - 1))}$$

$s > 1$ 使信息破坏更缓慢，$s = 2.0$ 配合 CFG scale 2.0 达到最低 L-FID。

### 训练损失

标准 DDPM 噪声预测损失：

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathcal{B}_0 \sim p(\mathcal{B}|C), t \sim \mathcal{U}(1,T)}\left[\|\mathcal{E}_\theta(\mathcal{B}_t, C, t) - \mathcal{E}_t\|^2\right]$$

模型仅 ~18M 参数，2 块 A5000 GPU 训练。

## 实验

### 布局质量评估（Tab. 2, COCO-GR 数据集）

| 方法 | L-FID ↓ |
|------|---------|
| LayoutGPT (GPT-3.5) | 3.51 |
| LayoutGPT (GPT-4o-mini) | 6.72 |
| Llama-3.1-8B (微调) | 13.95 |
| **LayouSyn** | **3.07 (+12.5%)** |
| LayouSyn (GRIT预训练) | 3.31 (+5.6%) |

LayouSyn 在布局 FID 上超越 LayoutGPT(GPT-3.5) **12.5%**，且无需闭源 LLM。

### 空间与数量推理（Tab. 3, NSR-1K 基准）

| 方法 | 数量 Acc ↑ | 数量 Recall ↑ | 空间 Acc ↑ | 空间 GLIP ↑ |
|------|----------|-------------|----------|-----------|
| LayoutGPT (GPT-4o-mini) | 77.51 | 86.84 | 92.01 | 60.49 |
| LLMGroundedDiffusion | 89.94 | 95.94 | 72.46 | 27.09 |
| LLM Blueprint | 38.36 | 67.29 | 73.52 | 50.21 |
| Llama-3.1-8B (微调) | 70.84 | 93.36 | 86.64 | 52.93 |
| **LayouSyn** | **95.14** | **99.23** | 87.49 | 54.91 |
| **LayouSyn (GRIT)** | **95.14** | **99.23** | **92.58** | **58.94** |

数量推理：Recall 99.23%、Accuracy 95.14%——预测对象集与 GT 高度重叠。
空间推理：LayouSyn(GRIT) 达到 92.58% 准确率和 58.94% GLIP 检测准确率（超过 GT 布局的 57.20%）。

### 消融实验

**描述集来源（Tab. 5）**：

| LLM | L-FID ↓ |
|-----|---------|
| GPT-3.5 | 3.49 |
| GPT-4o-mini | 3.22 |
| Llama-3.1-8B | **2.74** |

Llama-3.1-8B 生成的描述集质量甚至优于 GPT 系列——提取名词短语是相对简单的语言任务。

**LDiT 架构（Tab. 7）**：

| 配置 | L-FID ↓ |
|------|---------|
| 无交叉注意力 + 无调制 | 2.82 |
| 有交叉注意力 + 无调制 | 2.81 |
| **有交叉注意力 + 有调制** | **2.74** |

**采样步数（Tab. 6）**：15 步 DDIM 即可获得高保真布局，~5ms/样本。

## 亮点与洞察

1. **去 GPT 化**：证明轻量开源 LLM 完全胜任场景元素提取，无需昂贵的闭源 API
2. **扩散 Transformer 用于布局生成**的架构设计值得关注——LDiT 中全局-局部交叉注意力改善了条件对齐
3. **噪声调度缩放**是针对低维坐标扩散的重要技术洞察
4. **LLM 初始化**应用展示了有趣的组合范式：用 LLM 粗预测初始化 + 扩散模型精化，15 步内完成
5. 自动对象添加流水线（布局补全 + GLIGEN 修补）展示了实际编辑应用潜力

## 局限性

- 未处理遮挡关系（深度排序），可能导致物体重叠不合理
- 描述集生成依赖 LLM 的名词提取能力，对隐含对象可能遗漏
- 模型在训练数据分布外的极端宽高比下性能未验证

## 相关工作

- 布局生成：LayoutGAN、BLT、LayoutDM、Dolfin
- LLM 布局：LayoutGPT、LLM Blueprint、Ranni
- 可控图像生成：GLIGEN、Instance Diffusion、BoxDiff

## 评分

- **新颖性**: ★★★★☆ — 开放词汇布局的扩散 Transformer + 开源 LLM 组合新颖
- **技术深度**: ★★★★☆ — LDiT 架构和噪声调度缩放设计合理
- **实验质量**: ★★★★★ — 三个数据集 + 多基准 + 全面消融 + 应用展示
- **写作质量**: ★★★★☆ — 问题清晰，两阶段流水线直观

<!-- RELATED:START -->

## 相关论文

- [EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)
- [Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](../../CVPR2025/image_generation/dual_prompting_image_restoration_with_diffusion_transformers.md)
- [LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)

<!-- RELATED:END -->
