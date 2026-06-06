---
title: >-
  [论文解读] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models
description: >-
  [CVPR 2026][图像生成][概念擦除] GrOCE 提出基于动态语义图的免训练概念擦除框架，通过构建语义图→自适应聚类识别→选择性切除三个协同组件，实现对文本到图像扩散模型中目标概念的精确、上下文感知的在线移除。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "概念擦除"
  - "扩散模型"
  - "语义图"
  - "免训练"
  - "在线推理"
---

# GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2511.12968](https://arxiv.org/abs/2511.12968)  
**代码**: 有  
**领域**: 图像生成  
**关键词**: 概念擦除, 扩散模型, 语义图, 免训练, 在线推理

## 一句话总结
GrOCE 提出基于动态语义图的免训练概念擦除框架，通过构建语义图→自适应聚类识别→选择性切除三个协同组件，实现对文本到图像扩散模型中目标概念的精确、上下文感知的在线移除。

## 研究背景与动机
1. **领域现状**：文本到图像扩散模型频繁产生有害、偏见或侵权内容。概念擦除旨在移除目标内容的同时保留非目标语义。
2. **现有痛点**：（i）基于微调的方法计算成本高、存在灾难性遗忘、难以适应新兴风险；（ii）推理时干预方法依赖启发式映射，无法捕捉深层语义纠缠。两类方法都将概念视为孤立实体，忽略了潜空间中丰富的关系结构。
3. **核心矛盾**：扩散模型中概念以纠缠的流形形式编码，概念之间存在模糊边界和高阶依赖。移除一个概念（如"暴力"）可能损害语义相邻概念（如"冲突""动作"）。
4. **本文目标**：设计免训练的在线概念擦除方法，能理解和利用概念间的语义关系，实现精确的目标擦除而不损害邻近概念。
5. **切入角度**：将概念擦除重新表述为图上的切割问题——识别并移除连接到目标概念的最小顶点子集。
6. **核心idea**：构建概念间的动态语义图，通过多跳遍历和扩散评分识别目标概念聚类，然后从提示嵌入中选择性切除该聚类的语义成分。

## 方法详解

### 整体框架
给定文本提示和目标概念，三步协同：（1）Construct：从上下文化嵌入构建动态语义图；（2）Identify：通过多跳遍历+扩散评分识别目标概念聚类；（3）Sever：从提示嵌入中移除聚类的语义成分，保留非目标语义和全局句子结构。

### 关键设计

1. **动态语义图构建（Construct）**:
    - 功能：实时构建词汇概念间的语义关联网络
    - 核心思路：每个节点 $v_i$ 对应一个概念嵌入 $x_i$，如果两个节点的cosine相似度超过局部阈值 $\tau_i$ 则连边，边权 $w_{ij} = \exp(-(τ_i - \langle x_i, x_j \rangle)/\sigma)$。阈值通过局部相似度方差自适应调整 $\tau_i = \tau_0 + \lambda \cdot \text{std}$，适应不同密度的嵌入空间区域。
    - 设计动机：静态阈值无法适应嵌入空间中密度变化的区域。自适应阈值使得在稠密区域连接更保守，稀疏区域更积极。

2. **自适应聚类识别（Identify）**:
    - 功能：识别与目标概念语义纠缠的概念集合
    - 核心思路：从目标概念出发进行多跳遍历（带相似度衰减），识别语义影响范围。同时使用扩散评分量化每个邻居的语义影响力。最终提取紧凑的目标概念聚类。
    - 设计动机：简单的关键词过滤无法处理隐式语义关联（如"bear"→"grizzly"→"polar bear"），多跳遍历能捕捉高阶依赖。

3. **选择性切除（Sever）**:
    - 功能：从提示嵌入中移除目标聚类的语义成分，保留非目标语义
    - 核心思路：基于图引导的软投影，将提示嵌入中与目标聚类关联的语义方向投影去除，同时近似保持正交语义方向。编辑后的提示在扩散推理前注入模型。
    - 设计动机：硬删除可能破坏全局句子结构，软投影在移除目标语义的同时保持了嵌入空间的连贯性。

### 损失函数 / 训练策略
完全免训练，仅在推理时操作。无需梯度访问或模型重训练。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | GrOCE | ConAbl | AdaVD | 说明 |
|------------|------|-------|--------|------|------|
| 概念擦除 | CS↓ | SOTA | 次优 | - | 擦除更彻底 |
| 非目标保真 | FID↓ | SOTA | - | 次优 | 非目标损害更小 |
| 运行时间 | 秒 | ~0.1 | ~数十秒 | ~数秒 | 数量级加速 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full GrOCE | 最优 | 三组件完整 |
| w/o 图引导 | 下降 | 退化为简单关键词过滤 |
| w/o 多跳遍历 | 下降 | 无法捕捉高阶关联 |
| w/o 自适应阈值 | 下降 | 全局阈值不够精确 |

### 关键发现
- GrOCE在擦除准确性和非目标保真度上同时达到SOTA，证明图引导方法比孤立处理更优。
- 运行时间比训练方法快数量级，支持真正的在线概念移除。
- 语义图揭示了概念间的层次关系和共现模式，提供了可解释性。

## 亮点与洞察
- **图视角的引入**将概念擦除从"逐个处理"升级为"结构化推理"，是方法论层面的提升。
- **免训练+在线**的特性使其能快速适应新出现的有害概念，实际部署价值高。
- 语义图本身具有可解释性——不仅知道擦除了什么，还知道为什么。

## 局限与展望
- 仅处理文本可触达的概念，对纯视觉概念（如特定姿态-光照组合）无法处理。
- 假设概念在嵌入空间中线性可分，对非凸概念区域可能失效。
- 聚类识别的阈值和衰减参数需要调优。

## 相关工作与启发
- **vs ESD/CA**: 需要微调模型权重，计算昂贵且存在遗忘。GrOCE完全免训练。
- **vs AdaVD**: 假设线性可分性，对非凸区域失效。GrOCE通过图结构捕捉更复杂的关系。
- **vs UCE**: 推理时干预但假设稳定激活模式，重述提示时可能失效。GrOCE的图结构更鲁棒。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 图引导的概念擦除是全新范式
- 实验充分度: ⭐⭐⭐⭐ 多任务验证（卡通概念/艺术风格），效率对比充分
- 写作质量: ⭐⭐⭐⭐⭐ 数学形式化清晰，问题定义严谨
- 价值: ⭐⭐⭐⭐⭐ AI安全领域的重要贡献，实际部署价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](../../ICML2026/image_generation/orthogonal_concept_erasure_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
