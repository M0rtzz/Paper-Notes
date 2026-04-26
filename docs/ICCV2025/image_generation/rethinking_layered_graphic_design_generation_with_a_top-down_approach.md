---
title: >-
  [论文解读] Rethinking Layered Graphic Design Generation with a Top-Down Approach
description: >-
  [图像生成] 提出Accordion框架，采用自顶向下策略将AI生成的栅格化设计图转换为可编辑的分层设计（含背景、前景对象、矢量化文本层），由VLM在参考创建、设计规划和层生成三个阶段扮演不同角色。
tags:
  - 图像生成
---

# Rethinking Layered Graphic Design Generation with a Top-Down Approach

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2507.05601](https://arxiv.org/abs/2507.05601)
- **代码**: 未开源
- **领域**: 扩散模型 · 图形设计生成
- **关键词**: 分层设计, VLM, 自顶向下, 文本渲染, 设计自动化

## 一句话总结
提出Accordion框架，采用自顶向下策略将AI生成的栅格化设计图转换为可编辑的分层设计（含背景、前景对象、矢量化文本层），由VLM在参考创建、设计规划和层生成三个阶段扮演不同角色。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：图形设计（海报、广告等）本质上是分层格式——背景层、前景对象层、矢量化文本层。GenAI模型可以生成视觉上精美的栅格化设计图，但缺乏可编辑性（如文本不可修改、元素不可分离）。

现有方法采用**自底向上**策略（如COLE先生成背景、再添加对象、最后放置文本），导致：
1. 视觉元素之间缺乏全局协调——后期添加的元素可能与先前决定冲突
2. 文本可能占据过多空间，或与前景对象重叠
3. 没有全局视觉参考作为设计总蓝图

**核心洞察**：人类设计师通常先参考已有设计获取灵感（颜色、布局、排版风格），再分层创建——这就是**自顶向下**策略。本文首次尝试将AI生成的栅格化设计图转化为可编辑的分层设计。

## 方法详解

### 整体框架：三阶段VLM管线

Accordion围绕VLM（LLaVA-1.5-7B）构建，在三个阶段扮演不同角色：

### Stage 1: 参考创建

VLM角色：**提示词增强器**
- 输入用户短意图$I$或草图稿$S$
- VLM通过In-context Learning将简短描述扩展为详细提示词$P_{des}$
- 送入T2I模型（Flux）生成参考图像$R$

### Stage 2: 设计规划

VLM角色：**设计规划器**
- 输入参考图像$R$和组合提示$P = P_{task} + P_{des} + P_{ocr}$
- VLM输出有序字典序列$\{D_*\}$：每个元素包含边界框和属性
- 文本属性包括内容、颜色、字体、对齐方式、行数、角度等
- 关键能力：修正AI生成的无意义文本为有意义内容

### Stage 3: 层生成

VLM角色：**质量选择器**
- 按规划顺序逐层提取：
  1. 文本移除：根据文本边界框条件移除文本区域
  2. 对象提取：SAM提取前景对象 → 修复模型填充背景
  3. 结果选择：VLM评估多个候选结果，选择最优
- 最终堆叠：背景$B$ + 对象层$\{O_n\}$ + 矢量文本层$T$

### 训练数据构建

利用3类参考训练VLM：
1. **原始设计**：直接解析训练文本去渲染能力
2. **含无意义文本的设计**：用SD1.5修复模型污损文本区域（强度0.5-0.7），训练修复能力
3. **无文本背景**：移除所有文本，训练从背景添加文本的能力

共156,932个训练样本（39K × 3类参考 + 问卷数据集）。

## 实验

### 主实验：DesignIntention基准定量对比

| 方法 | 设计与布局 | 内容相关性 | 排版与颜色 | 图形与图像 | 创新性 | 平均 |
|------|----------|----------|----------|----------|--------|------|
| COLE | 6.0 | 6.9 | 5.7 | 6.2 | 5.1 | 6.0 |
| Open-COLE | 6.3 | 7.0 | 5.6 | 7.1 | 5.3 | 6.3 |
| **Accordion** | **6.7** | **7.4** | **6.1** | **7.3** | 5.1 | **6.5** |

Accordion以平均6.5分超越COLE（6.0）和Open-COLE（6.3），且训练数据仅39K远少于COLE的100K。

### 设计师用户研究（29名设计师，30个案例）

| 评估维度 | Accordion优于COLE(%) |
|---------|---------------------|
| 文本到模板可编辑性 | 73.5% |
| 草图到描述合理性 | 87.2% |

### 关键发现

1. Accordion的文本长度平均61.7字符 vs COLE的42.3字符（1.5倍）——更充分利用空间
2. 美学评分4.98 vs COLE的4.72——全局参考确保元素间的视觉和谐
3. 层数检测MAE：文本层0.494、对象层0.274——分层准确度较高
4. VLM问卷选择提升文本移除PSNR从18.01到21.18

## 亮点与洞察

1. **范式转变**：自顶向下 vs 自底向上——先有全局参考再分层提取，避免了逐步累积的视觉冲突
2. **三重角色VLM**：同一个VLM在不同阶段扮演提示增强器、设计规划器和质量选择器
3. **模型无关性**：可组合任意T2I模型（Flux/SD3/未来模型）作为参考源，不需重新训练
4. **设计变体**：支持上游模型变体、推理时变体和下游模型变体三种创意探索路径

## 局限与展望

- SAM的IOU仅68.4%，对透明/中空对象的提取存在困难
- 假设文本层始终在对象层之上，不支持更复杂的层次关系
- 文本层仅支持2,000种预定义样式，不支持自由形式文本或特效文字
- 推理时间36.7秒/样本，交互体验有待优化

## 相关工作

- **分层设计生成**: COLE, Open-COLE, De-Render
- **文本渲染**: TextDiffuser, TextDiffuser-2
- **VLM应用**: LLaVA, GPT-4V在设计评估中的应用

## 评分
- 新颖性：★★★★☆ — 自顶向下策略和AI设计图到可编辑设计的首次尝试
- 技术深度：★★★★☆ — 三阶段管线设计精巧，训练数据构建有巧思
- 实用性：★★★★★ — 直接面向设计实际需求

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)

<!-- RELATED:END -->
