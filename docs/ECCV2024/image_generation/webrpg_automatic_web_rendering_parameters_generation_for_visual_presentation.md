---
title: >-
  [论文解读] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation
description: >-
  [ECCV 2024][图像生成] 提出Web渲染参数生成（WebRPG）新任务，旨在根据HTML代码自动生成网页元素的视觉呈现参数（布局、文本样式、颜色），通过VAE压缩渲染参数和定制HTML嵌入捕获语义层次信息，建立自回归和扩散两种基线模型，其中自回归模型显著优于扩散模型和GPT-4。
tags:
  - ECCV 2024
  - 图像生成
---

# WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation

**会议**: ECCV 2024  
**arXiv**: [2407.15502](https://arxiv.org/abs/2407.15502)  
**领域**: 图像生成

## 一句话总结

提出Web渲染参数生成（WebRPG）新任务，旨在根据HTML代码自动生成网页元素的视觉呈现参数（布局、文本样式、颜色），通过VAE压缩渲染参数和定制HTML嵌入捕获语义层次信息，建立自回归和扩散两种基线模型，其中自回归模型显著优于扩散模型和GPT-4。

## 研究背景与动机

- 生成式模型已革新图像、文本、音频创作，但**网页设计自动化**这一关键领域还未被充分探索
- 网页设计复杂耗时，缺乏设计经验的开发者往往产出视觉效果差的网页
- 现有工作聚焦于CSS的特定子任务（布局生成、字体推荐、配色），缺乏从零开始的**全面网页视觉设计**方案
- CSS编码实践复杂（大量选择器选项），直接自动生成CSS代码具有挑战性=
- **核心思路**：将CSS标准化为渲染参数（RPs），将问题转化为给定HTML代码生成每个元素的渲染参数

**与已有工作的差距**：
- 图形设计方法限制元素数≤25，用5个token描述一个元素；而网页有数百个元素，每个需要13个渲染参数
- 一维序列表示忽略了网页的层次结构信息

## 方法详解

### 整体框架

潜空间生成方法：
1. **VAE** 将每个元素的所有渲染参数压缩到潜向量
2. **HTML嵌入** 编码语义、层次结构和字符数信息
3. **生成模型**（自回归或扩散）基于HTML嵌入生成潜向量
4. **VAE解码器** 将潜向量解码回渲染参数

### 关键设计

**渲染参数定义**：13种常见CSS属性，分三类：
- **布局属性**：left, top, width, height
- **文本属性**：font-style, font-weight, font-size, line-height, text-align, text-decoration, text-transform
- **颜色属性**：color, background-color

**VAE渲染参数压缩**：
- 将每个元素的 $\mathcal{W}$ 个渲染参数压缩为 $d=128$ 维潜向量
- 编解码器均为5层MLP
- 用合成数据预训练，确保潜空间覆盖尽可能多的元素外观组合
- 使输入长度仅与元素数S相关，而非 $S \times \mathcal{W}$

**HTML嵌入**：
- **语义嵌入**：用冻结的MarkupLM_large提取HTML token特征后平均池化
- **层次嵌入**：用XPath嵌入层编码元素在DOM树中的位置
- **字符数嵌入**：将元素文本字符数映射为密集向量（元素尺寸与字符数正相关）

**两种生成模型**：
- **WebRPG-AR**（自回归）：引入带掩码的真实潜向量 $\mathcal{Z}_{mask}$ 稳定训练，推理时全部掩码
- **WebRPG-DM**（扩散）：在VAE潜空间上进行标准扩散过程

### 损失函数

自回归模型：$L = \log p_\psi(\mathcal{P}|\mathcal{H}, \mathcal{Z}_{mask}) + L_{VAE}$

扩散模型：$L = \mathbb{E}_{\mathcal{Z},\epsilon,t}[\|\epsilon - \epsilon_\psi(\mathcal{Z}_t, t, \mathcal{H})\|_2^2] + L_{VAE}$

VAE损失：重建项 + KL散度正则项

## 实验关键数据

### 主实验

WebRPG基线与LLM的定量对比：

| 模型 | FID ↓ | FID_layout ↓ | Ele. IoU ↑ | FID_style ↓ | SC Score ↑ |
|------|-------|-------------|------------|-------------|------------|
| **WebRPG-AR** | **0.1281** | **0.1520** | **0.7082** | **0.2124** | **0.9474** |
| WebRPG-DM | 62.021 | 60.942 | 0.0357 | 106.95 | 0.3671 |
| GPT-4 | 4.2141 | 47.732 | 0.0347 | 8.8898 | 0.5515 |
| StarCoder2-7b | 11.899 | 51.432 | 0.0309 | 18.186 | 0.3639 |
| DeepSeek-Coder-6.7b | 5.8219 | 55.744 | 0.0330 | 7.4542 | 0.3949 |
| CodeLlama-13b | 9.2826 | 55.427 | 0.0278 | 11.625 | 0.3864 |
| Real Web Page | 0.0027 | 0.0015 | 1.0000 | 0.0074 | 1.0000 |
| Plain HTML | 8.5342 | 52.438 | 0.0354 | 8.4951 | 0.3668 |

WebRPG-AR的FID仅0.1281，远超扩散模型（62.021）和GPT-4（4.2141），尤其在布局方面Ele. IoU达0.7082而其他方法均低于0.04。

### 消融实验

基于WebRPG-AR的组件消融：

| # | VAE | $\mathcal{Z}_{mask}$ | 语义 | 层次 | 字符数 | FID ↓ | Ele. IoU ↑ | SC Score ↑ |
|---|-----|-----|------|------|--------|-------|------------|------------|
| 1 | ✗ | ✓ | ✓ | ✓ | ✓ | 0.9702 | 0.5954 | 0.8053 |
| 2 | ✓ | ✗ | ✓ | ✓ | ✓ | 0.1487 | 0.6462 | 0.9332 |
| 3 | ✓ | ✓ | ✗ | ✓ | ✓ | 0.1797 | 0.6620 | 0.9323 |
| 4 | ✓ | ✓ | ✓ | 1D位置 | ✓ | 0.3003 | 0.6345 | 0.8982 |
| 5 | ✓ | ✓ | ✓ | ✓ | ✗ | 0.1575 | 0.6769 | 0.9434 |
| **6** | **✓** | **✓** | **✓** | **✓** | **✓** | **0.1281** | **0.7082** | **0.9474** |

### 关键发现

1. **VAE至关重要**（#1 vs #6）：不用VAE压缩时FID从0.13恶化到0.97，一维展开序列过长且信息冗余
2. **扩散模型不适合此任务**：网页元素为非欧空间的层次结构，且任务需精细控制，与扩散模型的欧几里得空间假设和模糊生成特性冲突
3. **层次嵌入对布局至关重要**（#4）：用1D位置嵌入替代XPath层次嵌入后布局变得混乱无序
4. **语义嵌入帮助理解元素关系**（#3）：无语义信息时模型难以识别键值对等语义关系
5. **字符数嵌入影响尺寸预测**（#5）：缺少时元素宽度不匹配内容，出现文本截断
6. GPT-4在简单HTML上展现基本设计能力，但面对复杂HTML结构时布局表现有限

## 亮点与洞察

- **开创性任务定义**：WebRPG将复杂的CSS自动生成问题转化为结构化的渲染参数生成，降低了学习难度
- **VAE压缩的精妙设计**：将每个元素的多维渲染参数压缩为单一潜向量，使序列长度仅依赖元素数
- **HTML嵌入的三维信息整合**：语义（MarkupLM）、层次（XPath）、字符数三者缺一不可
- **全自动工作流展望**：LLM生成HTML + WebRPG生成视觉呈现，可实现端到端网页开发
- **与LLM的比较视角**：揭示了GPT-4在此任务上的能力边界——样式处理合理但布局能力不足

## 局限性

- 数据集限于Klarna电商网页，领域多样性有限
- 模型不处理图像内容（仅保留`<img>`标签），无法考虑图像与设计的协调
- 元素数和DOM树深度增加时性能下降（布局和样式指标均降）
- 自回归模型对序列末尾元素更容易出错（位于HTML代码尾部的元素）
- 不支持动态组件和交互设计
- 仅考虑13种CSS属性，真实网页的CSS属性范围远超此范围

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 表达清晰度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_visual_instruction_tuning.md)
- [Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

<!-- RELATED:END -->
