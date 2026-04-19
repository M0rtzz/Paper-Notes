---
title: >-
  [论文解读] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP
description: >-
  [CVPR 2026][稀疏自编码器] 提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS），揭示两类特征在分类、分割、深度估计中的差异化功能角色。
tags:
  - CVPR 2026
  - 稀疏自编码器
  - CLIP
  - 信息范围
  - 上下文依赖性
  - outlier token
---

# Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP

**会议**: CVPR 2026  
**arXiv**: [2604.05724](https://arxiv.org/abs/2604.05724)  
**代码**: 无  
**领域**: 模型压缩/可解释性  
**关键词**: 稀疏自编码器, CLIP, 信息范围, 上下文依赖性, outlier token

## 一句话总结
提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS），揭示两类特征在分类、分割、深度估计中的差异化功能角色。

## 研究背景与动机
**领域现状**：稀疏自编码器（SAE）已成为解释CLIP等视觉模型内部表示的核心工具，能将稠密多义（polysemantic）表示分解为稀疏单义（monosemantic）特征。

**现有痛点**：当前SAE可解释性研究几乎只关注特征的**语义身份**（"这个特征代表什么概念"），但一个标记为"dog"的特征可能是对整个物体的全局编码，也可能仅对局部纹理（如毛发）响应——后语义分析无法区分这两者。

**关键观察**：Vision Transformer中的**outlier token**（异常高范数的patch token）在微小上下文偏移（Shifted Context Cropping，SCC）下表现出极强的**空间不稳定性**——位置随上下文变化剧烈。这暗示全局信号对上下文高度敏感，而局部信号则稳定锚定于视觉内容。

**核心idea**：利用上下文偏移下的空间稳定性差异，量化每个SAE特征的"信息范围"——是聚合局部证据还是全局证据。

## 方法详解

### 整体框架
输入图像 → Shifted Context Cropping生成两个重叠裁剪 → ViT编码 → SAE解码得到每个特征的空间激活图 → 计算两个裁剪间激活图的EMD距离 → 平均得到CDS

### 关键设计
1. **Shifted Context Cropping (SCC)**：

    - 将图像调整为 $(p+s)n \times (p+s)n$ 后裁剪两个 $pn \times pn$ 的图，偏移 $sn$ 像素
    - 两个裁剪共享 $(p-s) \times (p-s)$ 个像素完全相同的patch
    - **设计动机**：隔离纯上下文因素（位置编码差异+注意力上下文变化），排除内容差异

2. **Contextual Dependency Score (CDS)**：
   对每个SAE特征 $f_j$：
    - 选取该特征激活最强的 $k_{CDS}$ 张图像
    - 对每张图像做SCC，获取重叠区域的特征激活图 $M_{j,1}^{(m)}$ 和 $M_{j,2}^{(m)}$
    - 归一化为概率分布，计算Earth Mover's Distance (EMD)
    - 归一化网格对角线距离后取平均：
    $CDS_j = \frac{1}{k_{CDS} \cdot D_{grid}} \sum_{m=1}^{k_{CDS}} \text{EMD}(\mathcal{N}(M_{j,1}^{(m)}), \mathcal{N}(M_{j,2}^{(m)}))$
    - **低CDS** → 空间稳定 → 局部范围特征；**高CDS** → 空间变化大 → 全局范围特征

3. **特征分区与验证**：
   CDS直方图呈多峰分布，用阈值 $\gamma$ 自然分为低CDS组和高CDS组。通过检查两组在outlier/non-outlier token上的激活模式验证：高CDS特征主要在outlier token上激活，低CDS特征主要在normal token上激活。

### 下游分析
- 特征组移除实验：冻结CLIP骨干，线性探针评估
- 三个任务：ImageNet分类、ADE20K语义分割、NYUd深度估计

## 实验关键数据

### 主实验（特征组移除 → 线性探针性能）
| 模型 | 嵌入类型 | ImageNet Top-1↑ | ADE20K mIoU↑ | NYUd RMSE↓ |
|------|---------|----------------|-------------|-----------|
| CLIP-B/16 | 原始 | 74.82 | 25.87 | 0.8841 |
| | 移除高CDS | **75.54** | **26.02** | **0.8616** |
| | 移除低CDS | 64.86 | 11.65 | 0.9481 |
| CLIP-L/14 | 原始 | 80.82 | 26.66 | 0.8029 |
| | 移除高CDS | **81.28** | 26.44 | **0.7994** |
| | 移除低CDS | 78.30 | 13.89 | 0.8878 |

### 消融实验
| 分析 | 关键发现 | 说明 |
|------|---------|------|
| Outlier vs Non-outlier EMD | outlier EMD >> non-outlier EMD | outlier token空间极不稳定 |
| 高CDS在outlier上激活 | 83.45 vs 1.66 (B/16) | 高CDS特征选择性响应outlier |
| 低CDS在normal token上激活 | 31.51 vs 10.39 (B/16) | 低CDS特征编码局部信息 |
| 移除高CDS提升分类 | +0.72 ~ +1.42% | 全局噪声移除反而有益 |

### 关键发现
- **移除低CDS特征**对分割和深度估计的破坏性极大（mIoU从26→12），证明空间细粒度信息主要由低CDS特征携带
- **移除高CDS特征**反而略微提升分类性能，说明全局特征可能包含冗余信息
- CDS分区与outlier token现象高度吻合，提供了outlier token的**特征级机理解释**

## 亮点与洞察
- 提出信息范围作为语义之外的正交可解释性维度，弥补了现有SAE分析的盲区
- CDS指标巧妙地利用SCC隔离上下文因素，设计精妙且物理意义明确
- 建立了SAE特征→outlier token→全局/局部信息的清晰链条

## 局限与展望
- CDS计算需要对每个特征选取top-k图像并做双向推理，计算成本随字典规模增长
- 仅分析了CLIP模型，DINOv2等自监督ViT的信息范围特性待探索
- 二分法（局部vs全局）可能过于粗糙，连续谱上的更细粒度分析有价值
- 未探索CDS引导的特征选择在实际下游任务中的应用

## 相关工作与启发
- 与ViT outlier token研究（Darcet et al.）形成互补：从现象描述深入到特征级机理
- CDS思想可推广到NLP中分析LLM的SAE特征（局部token vs 全局上下文）
- 为SAE特征的"质量控制"（哪些特征值得信任）提供了量化工具

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息范围维度和CDS指标均为全新贡献，视角独特
- 实验充分度: ⭐⭐⭐⭐ 三个CLIP模型、三个下游任务，但缺少非CLIP模型验证
- 写作质量: ⭐⭐⭐⭐⭐ 从现象→假说→指标→验证，逻辑链极为清晰
- 价值: ⭐⭐⭐⭐ 为模型可解释性研究开辟新方向，但实际应用场景还需探索

<!-- RELATED:START -->

## 相关论文

- [Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](../../NeurIPS2025/interpretability/transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)
- [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [Do Sparse Autoencoders Generalize? A Case Study of Answerability](../../ICML2025/interpretability/do_sparse_autoencoders_generalize_a_case_study_of_answerability.md)

<!-- RELATED:END -->
