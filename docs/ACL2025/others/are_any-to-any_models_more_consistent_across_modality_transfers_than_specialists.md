---
title: >-
  [论文解读] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?
description: >-
  [ACL 2025][any-to-any模型] 本文提出 ACON 数据集和三种一致性评估标准（循环一致性、前向等变性、共轭等变性），发现当前 any-to-any 模型在逐点评估中并不比专用模型组合更具跨模态一致性，但通过多编辑操作的分布式分析可以观察到弱一致性。
tags:
  - ACL 2025
  - any-to-any模型
  - 跨模态一致性
  - 循环一致性
  - 等变性
  - 多模态评估
---

# Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?

**会议**: ACL 2025  
**arXiv**: [2505.24211](https://arxiv.org/abs/2505.24211)  
**代码**: [github](https://github.com/JiwanChung/ACON)  
**领域**: 其他（多模态一致性评估）  
**关键词**: any-to-any模型, 跨模态一致性, 循环一致性, 等变性, 多模态评估

## 一句话总结

本文提出 ACON 数据集和三种一致性评估标准（循环一致性、前向等变性、共轭等变性），发现当前 any-to-any 模型在逐点评估中并不比专用模型组合更具跨模态一致性，但通过多编辑操作的分布式分析可以观察到弱一致性。

## 研究背景与动机

Any-to-any 生成模型旨在用统一框架同时理解和生成多种模态（文本、图像、音频等），与模态专用方法相比，它们在大部分参数上跨不同模态共享。然而，这类模型的实际价值仍不确定：在当前发展阶段，它们通常无法在输出质量上一致地超越专用模型（如 Stable Diffusion、Flux），并且由于优化单个大规模系统的计算开销，训练效率也较低。

一个自然的假设是：如果一个 any-to-any 模型成功学习了统一的潜在空间，它应该比两个各自拥有独立潜在表示的专用模型产生更连贯的跨模态转换。但这个假设是否成立？本文通过严格的实验来验证这一猜想。

作者认为，现有的评估方法缺乏对跨模态一致性的系统性度量。检索式指标（如 CLIPScore）不足以评估事实正确性，特别是在组合性和计数任务上。因此需要一种更精细的评估框架来衡量 any-to-any 模型相对于专用模型的真正优势。

## 方法详解

### 整体框架

本文的贡献包括：(1) 形式化定义三种跨模态一致性标准；(2) 构建精心标注的 ACON 数据集用于评估；(3) 系统性地比较 any-to-any 模型与专用模型组合的一致性。

评估框架涉及两类操作：
- **跨模态转换 (f)**：文本到图像 (f^{t→i}) 或图像到文本 (f^{i→t})
- **模态内修改 (g)**：图像编辑 (g^i) 或文本编辑 (g^t)，由现成工具实现

### 关键设计

**三种一致性标准：**

1. **循环一致性 (Cyclic Consistency)**：将输入从一种模态转换到另一种再转换回来后，应能还原原始输入。例如 text → image → text 应该恢复原始文本。形式化为 $f^{i→t}(f^{t→i}(x^t)) = x^t$。

2. **前向等变性 (Forward Equivariance)**：在模态转换前或后应用修改操作应该产生相同结果。形式化为 $f^{t→i}(g^t(x^t, p)) = g^i(f^{t→i}(x^t), p)$。这个评估只比较同方向的模态转换。

3. **共轭等变性 (Conjugated Equivariance)**：在循环转换中间插入一个模态内编辑操作。形式化为 $f^{i→t}(g^i(f^{t→i}(x^t), p)) = g^t(x^t, p)$。这将评估从单点扩展到对中间潜在空间的结构性多点一致性分析。

**ACON 数据集构建：**

- **图像**：1000 张图像，其中 500 张为全新私有照片（未被任何 MLLM 训练过），500 张来自 COCO Captions
- **标注**：采用通信游戏框架 (communication game)，设置三个角色——
    - Teller：为图像编写详细描述（不能看到重建图像）
    - Drawer：根据描述使用 AI 工具重建图像
    - Judge：评估重建质量并提供反馈
- **QA 对**：每张图像配 10 个二元问答对（5 个关注相似性，5 个关注差异），由人工精心设计以捕获微妙的事实性差异
- **编辑操作**：每张图像 3 个编辑提示，每个提示附带 2 个条件化 QA 对
- **质量控制**：约 43% 的初始标注被过滤并替换，两轮独立人工审核

**评估工具：**

- 图像编辑：CosXL (Cos Stable Diffusion XL 1.0 Edit)
- 文本编辑：Qwen2.5
- VQA 评估：PaliGemma2（图像）、Qwen2.5（文本）

### 损失函数 / 训练策略

本文不涉及模型训练，是一个纯评估工作。使用现成模型进行比较：
- **专用模型**：Flux、SDXL（文本→图像）；LLaVA-Next、Qwen2VL（图像→文本）
- **Any-to-any 模型**：Chameleon、Emu-3、VILA-U、Seed-X

## 实验关键数据

### 主实验

**循环一致性（Image→Text→Image）：**

| 模型配对 | Accuracy (%) | F1 (%) |
|---------|-------------|--------|
| Qwen2VL + Flux | 62.86 | 73.20 |
| VILA-U (自身) | 62.15 | 72.88 |
| Seed-X (自身) | 61.78 | 72.53 |
| Chameleon (自身) | 53.93 | 64.33 |

**核心发现**：单个 any-to-any 模型并不一致地优于专用模型的任意组合。例如 Qwen2VL + Flux 的组合在多数指标上超过了 Chameleon 和 Emu3 的自循环。

**Text→Image→Text 方向：**

| 模型配对 | Accuracy (%) | F1 (%) |
|---------|-------------|--------|
| Flux + Qwen2VL | 66.93 | 73.61 |
| Seed-X (自身) | 63.52 | 70.22 |
| VILA-U (自身) | 62.36 | 68.59 |
| Chameleon (自身) | 55.76 | 60.59 |

### 消融实验

**前向等变性**：通过相关性分析，结果再次确认 any-to-any 模型相对于独立专用模型对并不一致地更优。但 Seed-X 和 VILA-U 在文本一致性方面表现出改善的趋势。

**共轭等变性**：这是最有趣的发现——所有 any-to-any 模型（除了 Chameleon 在图像生成方面）在与自身配对时都达到了稳定的自一致性。但它们在与自身配对时并不总是优于与其他模型配对。

### 关键发现

1. **视觉分词策略的影响**：Seed-X 和 VILA-U 表现出显著一致性，它们都采用了语义对齐的视觉分词器（利用预训练 ViT 的特征或优化与文本表示的对齐）。而 Chameleon 和 Emu3 仅依赖图像重建目标的分词器，一致性较差。

2. **循环一致性的局限**：这种评估将每个模态的转换性能与循环一致性混为一谈。例如 VILA-U 作为 text-to-image 算子时，无论与谁配对都表现良好。因此，需要多种互补的评估标准。

3. **共轭等变性揭示分布式一致性**：与循环一致性中几乎不可见的一致性不同，共轭等变性通过分析跨编辑操作分布的变换来捕获更广泛的潜在空间对齐模式。这与共享潜在学习假说一致。

4. **生成风格差异**：自然（ground truth）图像与模型生成图像之间存在风格差异，这削弱了循环一致性作为可靠性度量的价值，支持使用等变性进行更公平的比较。

## 亮点与洞察

1. **形式化一致性框架**：从数学角度严格定义了三种一致性标准，为评估 any-to-any 模型的潜在优势提供了可操作的理论框架。这是首次系统性地将循环一致性和等变性引入多模态模型的评估。

2. **反直觉的核心发现**：any-to-any 模型的宣称优势——统一潜在空间带来的跨模态连贯性——在当前阶段并未充分实现。这对研究社区的模型设计方向有重要指导意义。

3. **数据集的精心构建**：500 张真正未见过的私有图像、通信游戏框架的标注流程、43% 的样本淘汰率，体现了对评测质量的极高标准。

4. **视觉分词器的设计启示**：语义对齐的视觉分词器（如 Seed-X、VILA-U 所用）比纯图像重建目标的分词器在跨模态一致性上更有优势。这为未来 any-to-any 模型的架构设计提供了具体指导。

## 局限与展望

1. 以模型整体进行评估，无法隔离数据、架构和训练过程中特定因素对一致性的贡献。
2. ACON 数据集聚焦于自然照片，不包含艺术图像、2D 画作和 3D 渲染。
3. 标注者为 5 名具有相似文化背景的 NLP 研究人员，存在文化偏见风险。
4. 使用确定性采样，未明确考虑生成多样性问题。
5. 仅评估了 (image, text) 模态对，未扩展到 (speech, text) 等其他模态组合。
6. 仅评估单次循环转换，迭代组合可能揭示更多一致性模式（例如反复应用是否导致输出多样性崩塌）。

## 相关工作与启发

本文连接了 CycleGAN 的循环一致性概念与多模态模型评估。与 MM-R3、MMCBench 等鲁棒性评估工作不同，本文关注的是模态转换的结构性一致性，而非对语义扰动或损坏输入的鲁棒性。

等变性概念借自几何深度学习（Cohen & Welling, 2016），将其应用于多模态评估是一个有创意的跨领域迁移。共轭等变性作为对循环一致性的推广，让评估从单点扩展到了分布式分析，这种思路对其他多模态评估任务也有借鉴价值。

研究结果表明，当前的 any-to-any 模型更像是将多个专用模型"塞"在一起训练，还未真正实现共享潜在空间的承诺。这为社区指明了未来改进方向：视觉分词器的语义对齐可能是关键。

## 评分

- **创新性**: ★★★★☆ — 三种一致性标准的形式化定义新颖
- **实用性**: ★★★★☆ — ACON 数据集对 any-to-any 模型开发有直接指导价值
- **实验充分度**: ★★★★★ — 8 个模型、3 种评估标准、多方向组合，非常全面
- **写作质量**: ★★★★☆ — 数学形式化清晰，动机论述充分

<!-- RELATED:START -->

## 相关论文

- [CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](coral_speculative_drafting.md)
- [Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)
- [One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)
- [A Single Architecture for Representing Invariance Under Any Space Group](../../ICLR2026/others/a_single_architecture_for_representing_invariance_under_any_space_group.md)
- [I0T: Embedding Standardization Method Towards Zero Modality Gap](i0t_embedding_standardization_method_towards_zero_modality_gap.md)

<!-- RELATED:END -->
