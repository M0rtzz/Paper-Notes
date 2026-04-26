---
title: >-
  [论文解读] A Conditional Probability Framework for Compositional Zero-shot Learning
description: >-
  [ICCV 2025][组合零样本学习] 本文提出条件概率框架CPF，将组合零样本学习中的组合似然分解为物体似然和条件属性似然，通过文本增强的物体学习和物体引导的属性学习模块显式建模属性-物体的语义约束和上下文依赖，在UT-Zappos50K上AUC提升17.9%，在MIT-States上Unseen Accuracy提升5.5%。
tags:
  - ICCV 2025
  - 组合零样本学习
  - 条件概率
  - 属性-物体解耦
  - 跨注意力
  - 视觉-语言模型
---

# A Conditional Probability Framework for Compositional Zero-shot Learning

**会议**: ICCV 2025  
**arXiv**: [2507.17377](https://arxiv.org/abs/2507.17377)  
**代码**: 有（论文中提到"Code is available at here"）  
**领域**: 自然语言处理与多模态  
**关键词**: 组合零样本学习, 条件概率, 属性-物体解耦, 跨注意力, 视觉-语言模型

## 一句话总结
本文提出条件概率框架CPF，将组合零样本学习中的组合似然分解为物体似然和条件属性似然，通过文本增强的物体学习和物体引导的属性学习模块显式建模属性-物体的语义约束和上下文依赖，在UT-Zappos50K上AUC提升17.9%，在MIT-States上Unseen Accuracy提升5.5%。

## 研究背景与动机
组合零样本学习（CZSL）旨在通过已见组合的知识来识别由已知属性和物体构成的新组合（如训练时见过"条纹衬衫"和"蓝色天空"，测试时需识别"蓝色衬衫"）。这是零样本学习的一个重要扩展方向。

现有CZSL方法大多假设属性和物体是独立的，通过独立的网络模块（如FC层、自注意力/跨注意力解耦器）分别学习属性和物体表示。然而，这种独立解耦忽略了两个关键依赖：(1) 语义约束——某些属性只与特定物体搭配（如"条纹"适用于"斑马"但不适用于"天空"）；(2) 上下文依赖——同一属性在不同物体上视觉表现不同（如"年幼的树"vs"年幼的狗"）。

从概率角度看，组合的似然可以分解为 $p(o,a|\mathbf{x}) = p(o|\mathbf{x}) \cdot p(a|o,\mathbf{x})$。已有方法尝试建模上下文依赖（如CoT用物体特征生成属性注意力图，CANet用物体先验动态调整属性学习器参数），但它们仍无法有效建模语义约束。

本文基于条件概率分解这一洞察，提出CPF框架：先用文本增强的模块学习物体表示，再用物体表示引导属性学习，同时通过跨注意力机制实现语义约束和上下文依赖的联合建模。

## 方法详解

### 整体框架
CPF由视觉骨干网络和两个核心模块组成：(1) 文本增强的物体学习模块（TEO），利用深层视觉特征和文本embedding关注语义相关区域来增强物体表示；(2) 物体引导的属性学习模块（OGA），利用增强后的物体特征通过跨注意力从浅层视觉特征中提取上下文相关的属性表示。最终通过组合匹配将视觉和文本的组合特征对齐。

### 关键设计
1. **文本增强的物体学习（TEO）**:

    - 功能：整合文本语义信息增强物体特征的判别性
    - 核心思路：首先构建文本描述子embedding $\mathbf{q}^t$，通过深层class token $\mathbf{v}_h^c$ 与物体文本embedding $\mathbf{W}^o$ 的注意力加权得到：$\mathbf{q}^t = \text{softmax}(\frac{f_{v\to t}^o(\mathbf{v}_h^c)(\mathbf{W}^o)^\top}{\sqrt{d}})\mathbf{W}^o$。然后用 $\mathbf{q}^t$ 对深层patch tokens计算注意力权重，提取语义相关区域特征：$\mathbf{v}^o = \mathbf{v}_h^c + \text{softmax}(\frac{\mathbf{q}^t f_{v\to t}^o(\mathbf{V}_h^p)^\top}{\sqrt{d}})\mathbf{V}_h^p$
    - 设计动机：文本embedding提供物体的语义先验，帮助聚焦于图像中物体相关的区域，而非被无关属性干扰

2. **物体引导的属性学习（OGA）**:

    - 功能：基于已识别的物体上下文来学习属性表示
    - 核心思路：使用增强后的物体特征 $\mathbf{v}^o$ 作为query，对浅层patch tokens $\mathbf{V}_l^p$ 做跨注意力：$\mathbf{v}^a = \text{softmax}(\frac{\mathbf{v}^o(\mathbf{V}_l^p)^\top}{\sqrt{D}})\mathbf{V}_l^p$。物体特征引导注意力聚焦于与该物体语境下最相关的属性区域
    - 设计动机：属性的视觉表现依赖于物体上下文（如"年幼"在树和狗上表现完全不同）；用物体特征作为query天然建模了这种条件依赖

3. **组合匹配与多任务损失**:

    - 功能：将属性和物体特征组合并与文本组合特征对齐
    - 核心思路：组合视觉特征 $\mathbf{v}^c = f_c^v([\mathbf{v}^a, \mathbf{v}^o])$ 与文本特征 $\mathbf{w}^c = f_c^t([\mathbf{w}^a, \mathbf{w}^o])$ 通过交叉熵对齐。推理时用加法融合避免概率消失：$\hat{c} = \arg\max_{c_{i,j}} p(c_{i,j}|\mathbf{x}) + p(a_i|\mathbf{x}, \mathbf{v}^o) + p(o_j|\mathbf{x})$
    - 设计动机：加法而非乘法融合避免条件概率的数值下溢问题

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{com} + \alpha_1 \mathcal{L}_{att} + \alpha_2 \mathcal{L}_{obj}$（$\alpha_1=0.6$, $\alpha_2=0.4$）
- 三个交叉熵损失分别监督组合预测、属性预测和物体预测
- ViT-B骨干学习率1e-4，CLIP骨干学习率3.15e-6，Adam优化器训练10 epochs
- 温度参数 $\tau=0.05$

## 实验关键数据

### 主实验（Closed-World设定）

| 数据集 | 指标 | CPF (ViT-B) | 之前SOTA | 提升 |
|--------|------|-------------|----------|------|
| UT-Zappos50K | AUC | 41.4 | 35.1 (ADE) | +17.9% |
| UT-Zappos50K | HM | 55.7 | 51.1 (ADE) | +9.0% |
| MIT-States | Unseen Acc | 34.8 | 33.0 (CoT) | +5.5% |
| MIT-States | AUC | 11.2 | 10.5 (CoT) | +6.7% |
| C-GQA | HM | 23.9 | 22.1 (CoT) | +8.1% |

### 消融实验（C-GQA数据集）

| 配置 | AUC (CW) | HM (CW) | AUC (OW) | 说明 |
|------|----------|---------|----------|------|
| Full (TEO + OGA) | 8.2 | 23.9 | 2.10 | 完整模型 |
| -TEO | 7.6 | 22.7 | 1.79 | 去掉文本增强物体学习 |
| -TEO-OGA | 6.9 | 21.4 | 1.69 | 去掉两个核心模块 |
| average替代Eq.2注意力 | 7.8 | 22.9 | 1.91 | 简单平均替代跨注意力 |
| average替代Eq.4注意力 | 7.1 | 22.0 | 1.79 | 属性学习中去掉注意力 |

### 关键发现
- TEO和OGA两个模块均有显著贡献：去掉TEO后AUC下降7.3%，进一步去掉OGA再降9.2%
- 跨注意力机制优于简单平均，证明选择性地关注相关图像区域是关键
- 在Open-World设定下改善尤为显著：MIT-States AUC提升175%，C-GQA AUC提升47.9%
- 深层和浅层视觉特征的结合优于仅用深层特征，浅层特征提供了更细粒度的属性信息
- 推理速度1457 fps，与ADE(1445)和CoT(1460)相当，额外计算开销极小

## 亮点与洞察
- 从条件概率分解的角度重新思考CZSL问题，理论动机清晰
- "先识物体再看属性"的设计符合人类视觉认知的直觉——我们通常先识别物体再关注其属性
- 深浅层特征分别用于物体和属性学习的设计很有道理：深层特征捕获语义，浅层特征保留纹理/颜色等属性细节
- 推理时加法融合（而非概率乘法）避免了数值下溢问题，是一个实用的工程trick
- 额外计算开销极小（推理1457 fps），说明CPF的改进主要在于特征组织方式而非计算量

## 局限与展望
- 在CLIP骨干上的提升相对ViT-B较小，说明CLIP已有的组合泛化能力部分覆盖了CPF的优势
- 属性-物体的条件概率分解假设属性依赖于物体，但某些情况下物体也可能依赖于属性（如"方形"决定了是"窗户"而非"门"）
- 目前仅考虑单属性单物体的组合，实际场景中可能存在多属性或多物体组合
- 文本embedding使用冻结的GloVe，可能限制了语义描述的表达能力
- 浅层和深层特征的选择（第3/6/9层 vs 最后一层）是固定的，可能不是所有数据集的最优选择

## 相关工作与启发
- **vs CoT (ICCV2023)**: CoT仅用物体特征生成属性注意力图，CPF更进一步引入文本语义约束并使用跨注意力
- **vs ADE (CVPR2023)**: ADE独立解耦属性和物体，CPF通过条件概率框架建模依赖关系，UT-Zappos50K上AUC提升17.9%
- **vs CANet (CVPR2023)**: CANet用物体先验调整属性学习器参数，但仍设计为独立模块；CPF的跨注意力直接在特征层面建模依赖
- **vs CLIP-based方法(CSP/Troika等)**: CPF在C-GQA上超越所有CLIP方法，证明条件概率框架的通用性
- **启发**: 条件概率分解的思路可推广到其他组合学习问题（如动作-物体交互识别）

## 评分
- 新颖性: ⭐⭐⭐⭐ 条件概率分解视角新颖，TEO+OGA的递进式设计清晰有效
- 实验充分度: ⭐⭐⭐⭐⭐ CW/OW两种设定，3个数据集，ViT-B和CLIP两种骨干，详尽的消融实验
- 写作质量: ⭐⭐⭐⭐ 动机阐述到位，公式推导清晰，图示直观
- 价值: ⭐⭐⭐⭐ 提供了CZSL的新范式，大幅度的性能提升证明了建模属性-物体依赖的重要性

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ICCV 2025\] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ICCV 2025\] Supercharging Floorplan Localization with Semantic Rays](supercharging_floorplan_localization_with_semantic_rays.md)

<!-- RELATED:END -->
