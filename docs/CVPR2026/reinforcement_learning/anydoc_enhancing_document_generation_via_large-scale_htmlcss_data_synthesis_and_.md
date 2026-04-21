---
title: >-
  [论文解读] AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization
description: >-
  [CVPR 2026][文档生成] AnyDoc 提出了一个基于统一 HTML/CSS 表示的通用文档生成框架，通过自动化数据合成管线构建 265K 文档数据集 DocHTML，结合 SFT 和高度感知强化学习（HARL）微调多模态大模型，在意图到文档、文档反渲染和元素到文档三个任务上超越 GPT-4o 等基线。
tags:
  - CVPR 2026
  - 文档生成
  - HTML/CSS
  - 数据合成
  - 强化学习
  - 多模态大模型
---

# AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization

**会议**: CVPR 2026  
**arXiv**: [2603.25118](https://arxiv.org/abs/2603.25118)  
**代码**: 无  
**领域**: Reinforcement Learning / Document Generation  
**关键词**: 文档生成, HTML/CSS, 数据合成, 强化学习, 多模态大模型

## 一句话总结
AnyDoc 提出了一个基于统一 HTML/CSS 表示的通用文档生成框架，通过自动化数据合成管线构建 265K 文档数据集 DocHTML，结合 SFT 和高度感知强化学习（HARL）微调多模态大模型，在意图到文档、文档反渲染和元素到文档三个任务上超越 GPT-4o 等基线。

## 研究背景与动机
**领域现状**：日常工作中广泛使用各类文档（简历、演示文稿、报告等），人工设计高质量文档需要平衡结构、布局、视觉和风格多个原则。近年自动文档生成受到关注。

**现有痛点**：
   - **应用范围有限**：大多数方法只针对单一类别（广告、PPT、信息图），难以处理未见类别；
   - **文档表示次优**：
     - 栅格图像：不可编辑
     - 平面坐标序列（JSON）：对复杂文档需要大量坐标计算，难以表达层次结构
   - **数据稀缺**：人工制作文档成本高，现有数据集规模小（如 Crello 仅 20K）、类别少。

**核心矛盾**：如何同时实现类别通用性、结构可编辑性和数据充足性？

**本文切入角度**：引入 HTML/CSS 作为统一文档表示——天然层次结构、强大布局机制（flexbox/grid）、可大规模合成。

**核心 idea**：HTML/CSS 统一表示 + 自动化数据合成管线 + HARL 解决溢出问题 = 通用高质文档生成。

## 方法详解

### 整体框架
1. 数据合成管线（5 阶段）→ DocHTML 数据集（265K 文档，111 类，32 风格）
2. SFT 微调多模态大模型 → 三个生成任务
3. HARL 后训练 → 解决内容溢出问题

### 关键设计
1. **DocHTML 数据合成管线**：

    - **元数据收集**：从专业文档库出发，用 InternVL3 生成设计意图和内容描述
    - **HTML/CSS 代码生成**：用 Qwen3-Coder-480B 根据元数据生成代码，鼓励使用 flexbox/grid 布局；`<img>` 标签统一格式（占位符 URL + alt 描述）
    - **图像资产合成**：用 FLUX.1-dev 根据 alt 描述生成配图
    - **渲染**：用 Playwright 将 HTML/CSS + 图像渲染为文档截图
    - **数据清洗**：排除尺寸不匹配、img 标签缺失、零高度元素、内容溢出的样本
    - **设计动机**：HTML/CSS 的层次结构天然表达包含关系，flexbox 和 grid 避免了精确坐标计算。代码生成模型可大规模产出多类别文档。

2. **三个文档生成任务**：

    - **I2D（意图→文档）**：输入设计意图文本 + 目标尺寸 → 输出 HTML/CSS
    - **DD（文档反渲染）**：输入文档截图 + 目标尺寸 → 输出 HTML/CSS
    - **E2D（元素→文档）**：输入文本/图像元素集合 + 目标尺寸 → 输出 HTML/CSS
    - 所有任务统一为"条件 → HTML/CSS"的序列生成问题

3. **高度感知强化学习（HARL）**：

    - **功能**：解决 SFT 后生成文档的内容溢出问题。
    - **核心思路**：基于 GRPO，对每个输入采样一组候选输出，用 Playwright 渲染获得实际高度 $\hat{h}$，计算奖励：
    $r = \max\left(0, \begin{cases} 1, & 1-\gamma \leq \rho \leq 1 \\ \gamma + \rho, & \rho < 1-\gamma \\ 1 - \alpha(\rho - 1), & \rho > 1 \end{cases}\right)$
      其中 $\rho = \hat{h}/h$ 是高度偏比
    - 溢出（$\rho > 1$）和严重不足（$\rho < 1-\gamma$）均被惩罚
    - **设计动机**：SFT 模型容易生成超出指定高度的文档，截断渲染后效果差。GRPO 的组内相对优势机制自然区分"好"输出（高度合规）和"坏"输出（溢出），无需人工标注偏好。

### 损失函数 / 训练策略
- **SFT 阶段**：基于 Qwen2.5-VL-7B-Instruct，LoRA rank=32，batch=128，lr=1e-4
- **HARL 阶段**：全参数微调，lr=1e-6，batch=64，GRPO rollout=5
- 选择 SFT 推理中溢出最严重的 20K 样本作为 HARL 训练集

## 实验关键数据

### 主实验（I2D 意图→文档）

| 方法 | Layout | Image | Typography | Content | Height↓ | Intention |
|------|--------|-------|------------|---------|---------|-----------|
| OpenCOLE | 7.91 | 8.03 | 7.79 | 7.52 | - | 8.13 |
| FLUX.1-dev | 7.58 | 7.78 | 6.54 | 5.16 | - | 6.91 |
| GPT-4o | 8.59 | 8.75 | 8.32 | 8.41 | 0.047 | 8.96 |
| **AnyDoc** | **8.64** | **8.92** | **8.36** | **8.44** | **0.005** | 8.95 |

### 消融实验（文档反渲染，DD任务）

| 配置 | Block | Text | Position | Color | Height↓ |
|------|-------|------|----------|-------|---------|
| 坐标序列（JSON） | 0.871 | 0.925 | 0.780 | 0.812 | 0.188 |
| HTML/CSS（10K 数据） | 0.942 | 0.974 | 0.862 | 0.915 | 0.374 |
| HTML/CSS（完整数据） | 0.958 | 0.984 | 0.900 | 0.948 | 0.309 |
| + HARL | **0.965** | **0.986** | **0.910** | **0.958** | 0.309→更优 |

### 关键发现
- HTML/CSS 表示在所有指标上显著优于坐标序列（JSON），尤其是 Position 和 Color
- HARL 将 I2D 任务的 Height 指标从 0.131 降至 0.005（溢出几乎消除）
- DocHTML 数据量从 10K 增至完整集持续提升性能
- AnyDoc（7B）在多个指标上超越 GPT-4o，特别是 Height 控制

## 亮点与洞察
- **HTML/CSS 作为文档表示是关键创新**：将文档生成与 web 开发的成熟技术栈连接
- 数据合成管线完全自动化，可持续扩展类别和风格
- HARL 是精巧的解决方案：将渲染反馈引入 RL 奖励，让模型学会遵守尺寸约束
- 三个任务共享同一框架和数据集

## 局限与展望
- 7B 模型在 DD 任务的 Height 控制上仍不如 GPT-4o（模型规模的限制）
- HARL 需要 Playwright 渲染计算奖励，训练效率较低
- 生成的文档美观度仍依赖底层代码生成模型的能力
- 仅支持静态文档，交互式文档（带动态效果）未涉及

## 相关工作与启发
- 与 Design2Code 相比，不仅做反渲染还支持意图和元素到文档
- HARL 的思路（渲染→度量→奖励）可推广到其他需要视觉约束的代码生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐ HTML/CSS统一表示+数据合成+HARL三位一体
- 实验充分度: ⭐⭐⭐⭐ 三个任务×多个基线×消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析透彻，方法动机清晰
- 价值: ⭐⭐⭐⭐ 对自动化文档生成有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](../../ICLR2026/reinforcement_learning/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](../../ICLR2026/reinforcement_learning/towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)
- [Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification](../../NeurIPS2025/reinforcement_learning/kimina_lean_server_a_high-performance_lean_server_for_large-scale_verification.md)
- [LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](../../ICLR2026/reinforcement_learning/longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)

<!-- RELATED:END -->
