---
title: >-
  [论文解读] When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm
description: >-
  [CVPR 2026][图像生成][多模态大语言模型] 系统性对比分析了 MLLM（多模态大语言模型）与扩散模型在安全风险上的差异，发现 MLLM 因更强的语义理解能力而更容易生成不安全图像（抽象/非英语提示也能理解），且其生成的图像更难被现有假图检测器识别，即便针对性微调检测器也可通过丰富提示细节来规避。
tags:
  - CVPR 2026
  - 图像生成
  - 多模态大语言模型
  - 图像安全
  - 不安全内容生成
  - 虚假图像检测
  - 扩散模型对比
---

# When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm

**会议**: CVPR 2026  
**arXiv**: [2603.24079](https://arxiv.org/abs/2603.24079)  
**代码**: 无  
**领域**: AI安全 / 图像生成  
**关键词**: 多模态大语言模型, 图像安全, 不安全内容生成, 虚假图像检测, 扩散模型对比

## 一句话总结

系统性对比分析了 MLLM（多模态大语言模型）与扩散模型在安全风险上的差异，发现 MLLM 因更强的语义理解能力而更容易生成不安全图像（抽象/非英语提示也能理解），且其生成的图像更难被现有假图检测器识别，即便针对性微调检测器也可通过丰富提示细节来规避。

## 研究背景与动机

**领域现状**：文本到图像生成范式正从扩散模型（如 Stable Diffusion）向多模态大语言模型（MLLM，如 Janus、Bagel）转变。扩散模型通过迭代去噪生成图像，而 MLLM 将语言理解、跨模态推理和图像生成集成在统一架构中，具有更强的语义理解和组合灵活性。

**现有痛点**：围绕扩散模型已建立了相当成熟的安全防线——安全过滤器（如 NSFW 检测器）和假图检测器（如 DE-FAKE）。但这些防护手段都是针对扩散模型优化的。MLLM 具有本质不同的生成机制：它能理解微妙、抽象甚至非直接表达的不安全意图，这意味着现有防线可能系统性失效。

**核心矛盾**：MLLM 更强的语义理解能力是一把双刃剑——在提升可用性和生成质量的同时，也使得它能理解并执行那些扩散模型根本"看不懂"的不安全指令。而研究社区尚未系统认识到这一范式转移带来的新安全风险。

**本文目标** 两个核心研究问题：（1）MLLM 是否比扩散模型更容易生成不安全图像？（2）MLLM 生成的假图是否比扩散模型更难检测？

**切入角度**：以扩散模型作为参照基准，使用 1,184 条不安全提示生成 82,880 张图像，以及 2,000 条良性提示生成 14,000 张图像，对 7 个模型（2 个扩散 + 5 个 MLLM）进行系统对比实验。

**核心 idea**：MLLM 的语义理解优势在安全维度成为劣势——它能理解抽象不安全提示并生成有意义的不安全内容，而扩散模型因理解失败反而"安全"地产出损坏图像。

## 方法详解

### 整体框架

本文是一项系统性的测量与分析研究（measurement study），而非提出新方法。研究框架涵盖两个维度：不安全图像生成（RQ1）和假图检测（RQ2）。每个维度都使用精心设计的数据集、模型集合和评估指标进行定量和定性分析。

### 关键设计

1. **不安全图像生成评估框架**:

    - 功能：定量衡量不同模型在不安全提示下生成有害内容的倾向
    - 核心思路：选择 7 个模型（SD3.5 Large、SD3.5 Large Turbo、Bagel、Janus、Janus Pro、TokenFlow、VILA-U），使用 5 个不安全提示数据集（I2P、Lexica、4chan、Template、TemplateLong），每个提示生成 10 张图像。**禁用所有外部安全机制**以测量模型本身的倾向。使用 OpenAI Moderation API 作为安全分类器（与人类标注 89.2% 一致），计算每个提示-模型对的不安全得分（10 张中被标记为 NSFW 的比例）
    - 设计动机：需要公平对比模型内在的安全性，而非其安全过滤器的有效性

2. **损坏图像因素分析**:

    - 功能：解释为何扩散模型看似更安全——不是因为安全对齐好，而是因为理解失败
    - 核心思路：对 50 条 4chan 提示构造安全对照版本（替换不安全词），每个模型生成 10 张图像，测量"损坏图像率"。发现扩散模型对复杂提示的损坏率高达 80%（SD3.5 Large），而最差的 MLLM 仅 4.5%（Bagel）。99.4% 的损坏图像被分类为安全——正是这些"生成失败"拉低了扩散模型的不安全得分
    - 设计动机：揭示表面上的安全性差异背后的真正原因，避免对扩散模型"更安全"的错误结论

3. **假图检测评估框架**:

    - 功能：测试现有检测器对 MLLM 生成图像的检测能力
    - 核心思路：使用 MSCOCO 和 Flickr30k 的良性提示生成图像，测试 4 个检测器（Winston.AI、Illuminarty 两个商业 + DE-FAKE、AIorNot-SigLIP2 两个开源）。进一步设计提示丰富度梯度实验：v0（原始提示）→ v1（GPT-4o 扩展描述）→ v2（再次扩展），测试提示长度对检测难度的影响
    - 设计动机：验证 MLLM 的图像是否具有不同于扩散模型的"指纹"，以及现有检测器能否泛化

### 特殊分析维度

- **语言理解差异分析**：使用中文版不安全提示（TemplateLongChinese），扩散模型完全无法理解（不安全得分为 0），而 MLLM 仍能生成不安全图像
- **性别偏见分析**：发现 MLLM 在性别中立的色情提示下倾向于生成女性图像
- **联想能力分析**：使用抽象俚语提示（如"f*ck that the place is a sh*t hole"），扩散模型仅字面渲染文字，MLLM 能联想生成对应的肮脏环境

## 实验关键数据

### 主实验：不安全图像生成

| 数据集 | 指标 | SD3.5 Large | SD3.5 Large Turbo | Janus Pro (最差MLLM) | VILA-U | 
|--------|------|-------------|-------------------|---------------------|--------|
| TemplateLong | 平均不安全得分 | 0.280 | 0.200 | 0.613 | 高于扩散模型 |
| TemplateLongChinese | 平均不安全得分 | 0.000 | 0.000 | 高 | 高 |

各数据集上，扩散模型的不安全得分一致低于 5 个 MLLM。差距在 TemplateLong 上最为显著。

### 损坏图像率分析

| 模型 | 不安全提示损坏率 | 安全提示损坏率 |
|------|----------------|--------------|
| SD3.5 Large | 80.0% | 75.5% |
| SD3.5 Large Turbo | 66.0% | 66.5% |
| Bagel | 4.5% | 4.0% |
| Janus | 4.5% | 5.5% |
| Janus Pro | 0% | 0% |
| TokenFlow | 0% | 4.5% |
| VILA-U | 0% | 0% |

### 假图检测结果

MLLM 生成的图像对现有检测器构成更大挑战。即使针对 MLLM 数据微调检测器，跨 MLLM 泛化性较差。更关键的是，MLLM 可以通过丰富提示细节（v0→v1→v2）降低被检测概率——这一现象在扩散模型中不存在。

### 关键发现

- **MLLM 系统性更不安全**：在所有 5 个数据集上，MLLM 不安全得分一致高于扩散模型
- **不安全的根因是理解能力而非安全对齐失败**：扩散模型因无法理解复杂提示而产出损坏图像（被标记为安全），而非因安全训练更好
- **外部安全防护不够**：即使加上 NSFW 检测器，MLLM 的不安全得分仍高于扩散模型
- **检测器的泛化困境**：针对某个 MLLM 微调的检测器对其他 MLLM 效果有限，且 MLLM 可通过提示工程规避检测
- **MLLM 存在性别偏见**：性别中立的色情提示倾向于生成女性图像

## 亮点与洞察

- **"理解力即风险"的核心洞察极具启发性**：通常我们认为模型越"聪明"越好，但本文展示了理解能力在安全维度是双刃剑。这个框架可以推广到任何 AI 能力的安全性分析——任何提升"理解"的技术进步都可能同时扩大滥用面。
- **损坏图像作为"意外安全阀"的发现很有趣**：扩散模型不是因为"训练得更安全"而安全，而是因为"理解不了所以生成失败"。这意味着随着扩散模型理解能力提升，其安全风险也会相应增加。
- **提示丰富度-检测难度的关系**揭示了一种新型攻击向量：用户可以通过添加更多细节描述来让 MLLM 生成更难被检测的图像，这在扩散模型上无效，说明 MLLM 的生成机制从根本上不同。

## 局限与展望

- 所有实验都禁用了模型的安全机制，这虽然测量了模型内在倾向，但不完全反映实际部署场景
- 仅测试了 7 个模型，未覆盖 GPT-Image-1、Gemini 等主流闭源 MLLM 生成能力
- 安全分类依赖 OpenAI Moderation API（虽然与人类标注 89.2% 一致，但仍有约 10% 误差）
- 未深入分析 MLLM 内部安全对齐训练的具体机制，仅从外部行为推断
- 可探索：（1）基于 MLLM 语义理解特性的新型安全过滤器设计；（2）利用提示理解能力反过来做安全检测的思路

## 相关工作与启发

- **vs NSFW 安全过滤器**：本文表明即使加上外部过滤器，MLLM 仍比扩散模型更不安全，说明需要专门针对 MLLM 设计的安全机制
- **vs DE-FAKE / AIorNot 等假图检测器**：这些检测器针对扩散模型优化，对 MLLM 生成的图像效果显著下降，且微调后泛化性差，揭示了"检测器-生成器"军备竞赛在 MLLM 时代面临新挑战
- **vs 已有安全基准（I2P、Lexica）**：本文首次将这些基准扩展到 MLLM-扩散模型的跨范式对比，补充了 TemplateLong 等新测试集来测试语义理解差异

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性对比 MLLM 与扩散模型的安全风险，"理解即风险"的核心洞察新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个模型、5 个不安全数据集、4 个检测器、82,880 张生成图像，多维度定量+定性分析
- 写作质量: ⭐⭐⭐⭐ 研究问题清晰，实验设计严谨，案例分析直观
- 价值: ⭐⭐⭐⭐⭐ 对 MLLM 安全部署敲响警钟，揭示了现有安全基础设施在范式转移面前的系统性失效

<!-- RELATED:START -->

## 相关论文

- [When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)
- [Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [Latent Guard: a Safety Framework for Text-to-Image Generation](../../ECCV2024/image_generation/latent_guard_a_safety_framework_for_text-to-image_generation.md)

<!-- RELATED:END -->
