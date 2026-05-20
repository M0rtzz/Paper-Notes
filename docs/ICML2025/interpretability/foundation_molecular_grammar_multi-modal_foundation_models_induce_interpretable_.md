---
title: >-
  [论文解读] Foundation Molecular Grammar: Multi-Modal Foundation Models Induce Interpretable Molecular Grammar
description: >-
  [ICML 2025][可解释性][分子语法] FMG 利用多模态基础模型（MMFM）的化学知识，通过将分子渲染为图像并用文本描述，结合 prompt learning 跨模态对齐来归纳出可解释的分子图语法，替代传统依赖专家标注或启发式的语法学习方法。
tags:
  - "ICML 2025"
  - "可解释性"
  - "分子语法"
  - "多模态基础模型"
  - "图语法"
  - "分子生成"
---

# Foundation Molecular Grammar: Multi-Modal Foundation Models Induce Interpretable Molecular Grammar

**会议**: ICML 2025  
**arXiv**: [2505.22948](https://arxiv.org/abs/2505.22948)  
**代码**: [https://github.com/shiningsunnyday/induction](https://github.com/shiningsunnyday/induction)  
**领域**: 可解释性  
**关键词**: 分子语法, 多模态基础模型, 图语法, 可解释性, 分子生成

## 一句话总结
FMG 利用多模态基础模型（MMFM）的化学知识，通过将分子渲染为图像并用文本描述，结合 prompt learning 跨模态对齐来归纳出可解释的分子图语法，替代传统依赖专家标注或启发式的语法学习方法。

## 研究背景与动机

**领域现状**：数据高效的分子生成方法利用图语法引入可解释性，使生成过程可理解且可控。图语法将分子分解为有意义的子结构（产生式规则），并通过这些规则重组生成新分子。

**现有痛点**：现有的语法学习依赖专家标注或不可靠的启发式算法推断。专家标注成本高且不可扩展；启发式方法（如频繁子图挖掘）缺乏化学语义，导致语法规则质量不稳定。

**核心矛盾**：如何既保持图语法的可解释性优势，又避免对专家知识的依赖？需要一种自动化、可扩展且化学上有意义的语法归纳方法。

**本文目标**：利用多模态基础模型内蕴的化学知识来自动归纳高质量的分子图语法。

**切入角度**：将分子渲染为 2D 图像，用 MMFM 描述分子文本，通过 prompt learning 跨模态对齐来识别有意义的分子子结构。

**核心 idea**：借助 MMFM 的化学常识，用图像+文本的多模态描述来自动发现分子的可解释语法规则。

## 方法详解

### 整体框架

- **输入**：分子图（SMILES 或分子图结构）
- **中间过程**：(1) 将分子渲染为 2D 图像 → MMFM 提取视觉特征；(2) 用 MMFM 生成文本描述 → 提取语义特征；(3) Prompt learning 对齐两种模态 → 识别功能性子结构
- **输出**：可解释的分子图语法（一组产生式规则）

### 关键设计

1. **分子图像渲染与视觉编码**:

    - 将分子结构渲染为标准 2D 化学结构图像
    - 使用 MMFM（如 CLIP 或 LLaVA 类模型）的视觉编码器提取分子的视觉表示
    - **设计动机**：MMFM 在预训练过程中已学到了丰富的化学结构知识（如官能团识别、骨架模式）

2. **文本描述与语义编码**:

    - 利用 MMFM 为分子生成文本描述（化学性质、功能团等）
    - 提取文本嵌入作为语义条件
    - **设计动机**：文本提供了与视觉互补的高层语义信息，帮助识别化学上有意义的子结构

3. **Prompt Learning 跨模态对齐**:

    - 设计可学习的 prompt 来对齐视觉和文本模态
    - 通过对齐学习，自动发现在两种模态下一致的分子子结构模式
    - 这些模式即为归纳出的语法产生式规则
    - **设计动机**：跨模态一致性是子结构化学意义的强信号——真正有意义的功能团在图像和文本中都有对应

### 损失函数 / 训练策略

- Prompt learning 使用对比损失对齐视觉和文本嵌入
- 语法规则通过归纳学习从对齐后的子结构模式中提取
- FMG 作为即插即用模块，可替换现有语法学习方法

## 实验关键数据

### 主实验

| 任务 | 指标 | FMG | 之前方法 | 提升 |
|------|------|-----|---------|------|
| 分子生成（有效性） | Validity↑ | 更高 | 基线 | 显著 |
| 分子生成（多样性） | Diversity↑ | 更高 | 基线 | 提升 |
| 可合成性 | SA Score↑ | 更优 | 基线 | 提升 |
| 属性预测 | 准确度↑ | 更优 | 基线 | 提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅视觉模态 | 下降 | 文本提供关键语义信息 |
| 仅文本模态 | 下降 | 视觉提供结构细节 |
| 无 prompt learning | 明显下降 | 跨模态对齐是核心 |
| 替换为启发式语法 | 下降 | FMG 语法化学意义更强 |

### 关键发现

- FMG 在可合成性、多样性和数据效率方面均优于现有语法学习方法
- 归纳出的语法规则具有内置的化学可解释性
- FMG 可作为即插即用替换应用到现有基于语法的分子生成和属性预测框架中
- 多模态信息互补性对语法质量至关重要

## 亮点与洞察

1. **巧妙利用 MMFM**：将大规模预训练模型的知识迁移到分子语法归纳这一专业任务
2. **可解释性与自动化兼顾**：语法规则既有化学意义又不依赖人工标注
3. **即插即用设计**：不需要改变下游生成/预测框架，只替换语法学习部分
4. **数据效率**：在小样本场景下特别有优势

## 局限与展望

1. MMFM 的化学知识有限，可能在高度专业的化学领域（如金属有机化学）表现不足
2. 分子渲染为 2D 图像会丢失 3D 构象信息
3. Prompt learning 的效果可能对 MMFM 选择敏感
4. 扩展到更大分子或聚合物的可扩展性需验证

## 相关工作与启发

- JT-VAE 等基于树分解的分子生成方法提供了语法框架基础
- MMFM 在化学领域的应用是新兴方向，本文是其中的创新应用
- 启发：MMFM 的知识可能在其他科学领域（材料、蛋白质）的结构归纳中同样有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 MMFM 用于分子语法归纳是全新思路
- 实验充分度: ⭐⭐⭐⭐ 多任务验证，但规模相对有限
- 写作质量: ⭐⭐⭐⭐ 思路清晰
- 价值: ⭐⭐⭐⭐ 为可解释分子生成提供了新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](../../CVPR2026/interpretability/dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[ICML 2025\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[NeurIPS 2025\] Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models](../../NeurIPS2025/interpretability/fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](../../ICML2026/interpretability/is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)
- [\[ICML 2025\] What Makes an Ensemble (Un)interpretable?](what_makes_an_ensemble_un_interpretable.md)

</div>

<!-- RELATED:END -->
