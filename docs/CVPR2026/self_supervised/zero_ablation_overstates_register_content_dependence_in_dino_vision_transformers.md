---
title: >-
  [论文解读] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers
description: >-
  [CVPR 2026 (HOW Workshop)][自监督学习][register tokens] 通过三种替换控制实验（均值替换、噪声替换、跨图像洗牌）证明 DINO 系列 ViT 中零消融方法夸大了对 register token 精确内容的依赖性——模型实际只需"合理的 register-like 激活"而非图像特定值。
tags:
  - "CVPR 2026 (HOW Workshop)"
  - "自监督学习"
  - "register tokens"
  - "Transformer"
  - "zero-ablation"
  - "DINO"
  - "interpretability"
---

# Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers

**会议**: CVPR 2026 (HOW Workshop)  
**arXiv**: [2604.14433](https://arxiv.org/abs/2604.14433)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: register tokens, vision transformers, zero-ablation, DINO, interpretability

## 一句话总结

通过三种替换控制实验（均值替换、噪声替换、跨图像洗牌）证明 DINO 系列 ViT 中零消融方法夸大了对 register token 精确内容的依赖性——模型实际只需"合理的 register-like 激活"而非图像特定值。

## 研究背景与动机

零消融（将 token 激活替换为零向量）是探测 ViT 中 token 功能的常用方法。在 DINOv2+registers 和 DINOv3 中，清零 register token 导致分类下降高达 36.6pp、分割下降 30.9pp，表面上表明 register 不可或缺。然而零向量相对于原生 register 激活是不合理的分布外输入，可能夸大了真实的内容依赖性。这类似于神经科学中的损毁研究混淆——损伤通过互联回路级联传播产生过度定位的假象。

## 方法详解

### 整体框架

这是一篇方法论纠偏的分析工作，核心动作不是提新模型，而是给"零消融"找一组更公平的对照。作者对 DINOv2、DINOv2+registers、DINOv3 三个系列（ViT-S 和 ViT-B）做 hook-based 消融，在每个 block 输出后替换 [CLS] 或 register 的隐藏状态，然后在分类、检索、对应、分割四个下游任务上，把"清零"和三种"换成合理值"的控制实验摆在一起比——看性能崩的到底是因为 register 内容被破坏，还是因为零向量本身就是个分布外输入。立住"内容不重要"后，再用分布内验证排除"替换没生效"的解释，最后用有效秩与注意力流刻画 register 的真实角色。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["冻结 ViT 主干<br/>DINOv2 / DINOv2+R / DINOv3（ViT-S/B）"] --> B["逐 block 输出 hook 替换<br/>[CLS] 或 register 隐藏状态"]
    B --> Z["零消融基线<br/>置零向量（分布外输入）"]
    subgraph CTRL["三种替换控制（分布内对照）"]
        direction TB
        M["均值替换<br/>逐层数据集均值激活"]
        N["噪声替换<br/>均值/方差匹配的高斯"]
        S["跨图 register 洗牌<br/>保留统计、打破图像特定内容"]
    end
    B --> CTRL
    Z --> T["四下游任务评测<br/>分类 / 检索 / 对应 / 分割"]
    CTRL --> T
    T --> V["分布内验证<br/>逐 patch 余弦相似度 + JS 散度"]
    V --> R["有效秩与注意力流分析<br/>刻画 register 真实角色"]
    R --> O["结论：精确内容非必需<br/>register 是结构性上下文通道"]
```

### 关键设计

**1. 三种替换控制：把"移除功能"和"注入异常输入"分开**

零消融把 token 置零，可零向量相对原生 register 激活是严重的分布外输入，性能下降可能是被这个异常输入吓崩、而非真的依赖 register 内容。作者设计三种保持"合理"的替换来隔离这两件事：均值替换，用 5000 张 ImageNet 图像校准出的逐层数据集均值激活顶上；噪声替换，用均值和方差都匹配的逐层高斯噪声；跨图像 register 洗牌，在 batch 内随机排列 register 激活——保留真实激活的统计结构，只打破"图像特定内容"。

三者层层递进：如果模型只是需要一个"register-like 的激活"而非这张图自己的 register，那么前两种替换不该掉点，第三种洗牌更是直接证明内容可以张冠李戴。

**2. 分布内验证：先证明替换确实动了表示，排除"啥也没改"**

一个显然的质疑是：替换后不掉点，会不会是替换根本没改变特征？作者用逐 patch 余弦相似度量化扰动幅度（0.95–0.999，确实变了但没乱），证明三种替换都真切地改写了内部表示；同时用 JS 散度对比，零消融造成的分布偏移是这些合理替换的数十到数百倍。这一步把"替换无效"的解释堵死，剩下唯一能解释结果的就是：零消融夸大了内容依赖。

**3. 有效秩与注意力流分析：解释 register 到底在干什么**

把"内容不重要"立住之后，作者进一步刻画 register 的真实角色。有效秩分析显示 register 压缩了 patch 几何（有效秩从 13.5 降到 4.0，DINOv3 压得最狠）；注意力流分析则发现 register 注意力从中间层开始逐步累积，但分类对它的依赖直到第 10–11 层才突然出现。两者合起来印证 register 是个"结构性上下文通道"——它提供的是一种容器/缓冲角色，而非某张图独有的精确信息。

### 损失函数 / 训练策略

本文为分析性工作，不涉及训练。所有评估在冻结特征上进行。

## 实验关键数据

### 主实验

| 条件 | DINOv2+R 分类 | DINOv3 分类 | DINOv2+R 分割 | DINOv3 分割 |
|------|-------------|------------|-------------|------------|
| Full | 67.3% | 62.0% | 基线 | 基线 |
| Zero registers | -18.9pp | **-36.6pp** | -9.6pp | **-30.9pp** |
| Mean-sub | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 |
| Noise-sub | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 |
| Shuffle | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 | ≤1pp变化 |

### 关键发现

- 仅零消融产生性能下降，三种合理替换均保持所有任务的性能
- Register 缓冲了密集特征对 [CLS] 的依赖（分割下降 37pp vs <1pp）
- 结果在 ViT-B 规模上完全复现

## 亮点与洞察

- 优雅地揭示了零消融的方法论缺陷——注入分布外输入而非移除功能
- 与神经科学中的损毁研究类比恰当且有教育意义
- 结论清晰：register 功能如预期的"上下文通道"，精确内容非必需

## 局限与展望

- 仅在冻结特征评估上测试，微调后的模型可能表现不同
- 仅测试了 DINO 系列模型，其他自监督 ViT 的行为可能不同
- Workshop 论文篇幅有限，部分分析深度受限

## 相关工作与启发

- 为所有使用零消融进行功能探测的工作提供了重要方法论警示
- 激活替换的"分布内控制"思想可推广到 NLP 中的机制可解释性
- Register token 的"结构性通道"角色为 ViT 设计提供指导

## 评分

7/10 — 方法论贡献清晰且重要，但作为 Workshop 论文规模有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](../../CVPR2025/self_supervised/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)

</div>

<!-- RELATED:END -->
