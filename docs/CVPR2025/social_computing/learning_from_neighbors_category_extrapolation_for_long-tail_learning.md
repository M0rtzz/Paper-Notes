---
title: >-
  [论文解读] Learning from Neighbors: Category Extrapolation for Long-Tail Learning
description: >-
  [CVPR 2025][社会计算][长尾学习] 发现更细粒度的类别划分天然减轻长尾不平衡的影响，提出用 LLM 发现与现有类别相关的细粒度辅助类 + 网络爬虫收集图像 + 邻近静默损失防止辅助类喧宾夺主，在 ImageNet-LT 上 Few 类提升 16 个百分点（41.4→57.4）。 领域现状：长尾学习中少数类（头部）…
tags:
  - "CVPR 2025"
  - "社会计算"
  - "长尾学习"
  - "类别外推"
  - "LLM数据增强"
  - "邻近类发现"
  - "分类器掩蔽"
---

# Learning from Neighbors: Category Extrapolation for Long-Tail Learning

**会议**: CVPR 2025  
**arXiv**: [2410.15980](https://arxiv.org/abs/2410.15980)  
**代码**: 待公开  
**领域**: 社会计算  
**关键词**: 长尾学习、类别外推、LLM数据增强、邻近类发现、分类器掩蔽

## 一句话总结
发现更细粒度的类别划分天然减轻长尾不平衡的影响，提出用 LLM 发现与现有类别相关的细粒度辅助类 + 网络爬虫收集图像 + 邻近静默损失防止辅助类喧宾夺主，在 ImageNet-LT 上 Few 类提升 16 个百分点（41.4→57.4）。

## 研究背景与动机

**领域现状**：长尾学习中少数类（头部）有大量样本，多数类（尾部）样本极少。现有方法包括重采样、重加权、解耦训练等。

**现有痛点**：(1) 重采样/重加权只在现有类别空间内操作，无法增加尾部类的本质特征多样性。(2) 从网络随机爬取的增强数据可能引入噪声甚至伤害性能（如 60.9→56.8）。(3) 细粒度分类数据集（如 iNat18 8142 类）比粗粒度（ImageNet-LT 1000 类）的不平衡影响更小，但没有方法利用这一现象。

**核心矛盾**：尾部类需要更多数据，但随机增加数据不work——增加的数据必须是语义相关的细粒度变体才能真正扩展特征空间。

**本文目标** 利用"更细粒度=更不敏感长尾"的现象，自动发现细粒度辅助类来"外推"现有类别，填充特征空间。

**切入角度**：用 LLM（GPT-4）根据现有类名推荐语义相关的细粒度子类→网络爬虫收集图像→DINOv2 过滤噪声→邻近静默损失防止辅助类在训练中喧宾夺主→推理时直接掩蔽辅助类权重。

**核心 idea**：用 LLM 发现语义邻近的辅助细粒度类 + 网络爬虫收集数据 + 邻近静默损失训练 + 推理时掩蔽辅助类，通过类别外推填充特征空间缓解长尾不平衡。

## 方法详解

### 整体框架
现有长尾数据集类名 → GPT-4 推荐每个类的细粒度邻近类（结构化 prompt + in-context learning）→ 网络爬虫收集候选图像 → DINOv2 过滤（余弦相似度阈值 γ₁=0.7, γ₂=0.98）→ 邻近静默损失训练→ 推理时掩蔽辅助类权重。

### 关键设计

1. **LLM 驱动的细粒度类发现**:

    - 功能：发现与现有类语义相关的细粒度子类
    - 核心思路：结构化 prompt 向 GPT-4 提问"与 [类名] 相关的更细粒度类别有哪些？"。用 in-context learning 提供示例确保格式一致。尾部类分配更多邻近类（与头尾比例成正比）
    - 设计动机：尾部类需要更多样本→更多邻近类→更多爬取数据。头部类已充足→少量邻近类即可

2. **DINOv2 数据过滤**:

    - 功能：从爬取的噪声图像中筛选有效样本
    - 核心思路：双阈值过滤——与类别 prototype 的余弦相似度 > γ₁=0.7（保证语义相关）且 < γ₂=0.98（排除近重复）。DINOv2 的特征空间比 CLIP 更适合做细粒度过滤
    - 设计动机：网络爬取的原始数据噪声大。消融显示不过滤（随机数据 RD）实际上伤害性能（60.9→56.8）

3. **邻近静默损失（Neighbor-Silencing CE）**:

    - 功能：防止辅助类在训练中压制原始类
    - 核心思路：对原始类 i 和其辅助邻近类 j 之间的 logit 交互降权：$\lambda_s=0.1$。这使得模型不过分区分原始类与其邻近类，保持原始类的决策边界
    - 设计动机：不加 NS-CE 时辅助类会与原始类竞争，降低原始类准确率

4. **推理时分类器掩蔽**:

    - 功能：去除辅助类使模型仅在原始类上预测
    - 核心思路：直接将辅助类权重置为零/掩蔽，不需要重新训练分类器。消融显示简单掩蔽比重训练分类器效果更好
    - 设计动机：辅助类只是训练时的"脚手架"，推理时不需要

### 损失函数 / 训练策略
标准 CE 替换为 NS-CE（辅助类对交互降权）。支持 scratch/CLIP/DINOv2 三种初始化。

## 实验关键数据

### 主实验

| 设置 | Overall | Many | Med | Few |
|------|---------|------|------|------|
| ImageNet-LT Baseline | 60.9 | 72.9 | 56.8 | 41.4 |
| **+Ours (scratch)** | **68.2** (+7.3) | 74.5 | 66.2 | **57.4** (+16.0) |
| CLIP 初始化 | 74.0 | - | - | - |
| **+Ours (CLIP)** | **77.3** (+3.5) | - | - | - |
| DINOv2 初始化 | 79.6 | - | - | - |
| **+Ours (DINOv2)** | **82.0** (+2.4) | - | - | - |

### 消融实验

| 配置 | Overall |
|------|---------|
| Baseline | 60.9 |
| +随机数据 (RD) | 56.8 (-4.1!) |
| +邻近类数据 (SD) | 64.9 |
| **+SD + NS-CE** | **68.2** |
| 推理掩蔽 vs 重训分类器 | 掩蔽更好 |

### 关键发现
- **Few 类提升 16 个百分点**（41.4→57.4）：尾部类受益最大，类别外推直接填充了尾部特征空间
- **随机数据有害（-4.1），语义邻近数据有益（+4.0）**：数据必须语义相关才有效
- **三种初始化都有效**：scratch/CLIP/DINOv2 都一致受益，说明方法通用
- **更细粒度天然更平衡**的实证（20 超类 gap 7.3% vs 100 超类 gap 20.8%）解释了方法的理论基础

## 亮点与洞察
- **"细粒度减轻长尾"的现象**是一个有价值的理论发现——解释了为什么 iNat18（8142 类）比 ImageNet-LT（1000 类）对不平衡更鲁棒
- **LLM 数据管线（发现类→爬取→过滤）**完全自动化，可推广到任何长尾场景
- **推理时简单掩蔽 > 重训**是一个反直觉但实际的发现

## 局限与展望
- 依赖 GPT-4 来发现邻近类——对 GPT-4 不擅长的专业领域可能生成质量低
- 网络爬虫受搜索引擎结果质量影响
- NS-CE 的 λ_s=0.1 是固定值，自适应调节可能更好

## 相关工作与启发
- **vs Balanced Softmax / RIDE**：在现有类别空间内重平衡。本方法通过外推新类从根本上扩展特征空间
- **vs CLIP 长尾**：用 CLIP 初始化可进一步叠加（74.0→77.3），两者正交互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "类别外推缓解长尾"的概念新颖深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 3 数据集、3 初始化、详细消融、粒度分析
- 写作质量: ⭐⭐⭐⭐⭐ 从现象到方法的逻辑链严密
- 价值: ⭐⭐⭐⭐⭐ 对长尾学习开辟了全新的数据增强视角

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ECCV 2024\] Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels](../../ECCV2024/social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)
- [\[ICML 2025\] Learning Survival Distributions with the Asymmetric Laplace Distribution](../../ICML2025/social_computing/learning_survival_distributions_with_the_asymmetric_laplace_distribution.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->
