---
title: >-
  [论文解读] MAVias: Mitigate Any Visual Bias
description: >-
  [ICCV2025][多模态][Visual Bias Mitigation] 提出 MAVias，一个开放集视觉偏差缓解框架：利用图像标注基础模型提取视觉属性标签，用 LLM 筛选与目标类别无关的标签作为潜在偏差，再通过 vision-language embedding 编码偏差并融入训练过程以学习偏差不变表示，在 CelebA、Waterbirds、UrbanCars 和 ImageNet9 上大幅超越现有方法。
tags:
  - ICCV2025
  - 多模态
  - Visual Bias Mitigation
  - Open-Set Bias
  - Foundation Models
  - 视觉语言
  - Fairness
---

# MAVias: Mitigate Any Visual Bias

**会议**: ICCV2025  
**arXiv**: [2412.06632](https://arxiv.org/abs/2412.06632)  
**代码**: https://github.com/gsarridis/VB-Mitigator (VB-Mitigator library)  
**领域**: 多模态VLM / 偏差缓解  
**关键词**: Visual Bias Mitigation, Open-Set Bias, Foundation Models, Vision-Language Models, Fairness

## 一句话总结
提出 MAVias，一个开放集视觉偏差缓解框架：利用图像标注基础模型提取视觉属性标签，用 LLM 筛选与目标类别无关的标签作为潜在偏差，再通过 vision-language embedding 编码偏差并融入训练过程以学习偏差不变表示，在 CelebA、Waterbirds、UrbanCars 和 ImageNet9 上大幅超越现有方法。

## 研究背景与动机

**领域现状**：深度学习模型容易学到训练数据中的虚假关联（spurious correlations），例如"水鸟"总是出现在水面背景、"金发"总是女性等。现有偏差缓解方法分为两类：Bias-Aware (BA) 方法需要偏差属性标注，Bias-Unaware (BU) 方法通过训练偏差代理模型推导伪标签。

**现有痛点**：
   - BA 方法依赖已知的预定义偏差标签，但在大规模通用数据集（如 ImageNet）中，偏差多种多样且未知
   - BU 方法仅在偏差极端显著时有效（能训练出偏差代理模型），无法处理多属性、未知偏差
   - 两类方法都无法扩展到开放集场景（open-set），即偏差类型事先未知、数量不确定

**核心矛盾**：实际场景中偏差是实例级别的（每张图像可能有不同的无关属性组合），而现有方法设计用于数据集级别的单一或少数已知偏差。

**本文目标**：如何在不预定义偏差类型的情况下，自动发现并缓解图像中任意数量、任意类型的视觉偏差。

**切入角度**：利用基础模型（图像标注模型 + LLM + vision-language 模型）的组合能力，自动提取视觉属性并判断其与目标类别的相关性，将无关属性编码为偏差信号融入训练。

**核心 idea**：用基础模型自动发现实例级开放集视觉偏差，将偏差编码为 vision-language embedding 并通过 logit 融合实现偏差不变学习。

## 方法详解

### 整体框架
MAVias 分为两个阶段：(1) **偏差建模**：对每张训练图像，提取描述性标签 → LLM 筛选无关标签 → vision-language 模型编码偏差；(2) **偏差缓解训练**：主模型提取图像特征并计算主 logits，投影层将偏差 embedding 映射到同一空间计算偏差 logits，二者相加作为最终预测，通过梯度调制使模型忽略偏差特征。

### 关键设计

1. **语言驱动的偏差建模（Language-driven Bias Modeling）**:

    - 功能：自动发现每张图像中与目标类别无关的视觉属性
    - 核心思路：三步流程——(a) 用 RAM（Recognize Anything Model，4000+ 标签词汇量）对图像提取描述性标签集 $\mathcal{T}^{(i)}$；(b) 用 GPT-4o 判断每个标签是否与目标类别 $y^{(i)}$ 相关，筛选出无关标签子集 $\mathcal{B}^{(i)} \subseteq \mathcal{T}^{(i)}$；(c) 用 OpenCLIP 将无关标签编码为 prompt "a photo of $t_1, t_2, ..., t_k$" 的统一 embedding $\mathbf{e}^{(i)} \in \mathbb{R}^d$
    - 设计动机：(1) RAM 覆盖 4000+ 视觉概念，满足开放集需求；(2) LLM 具备常识推理能力，能判断标签与类别的语义关联；(3) 将所有无关标签聚合为单一 embedding 而非逐标签处理，降低计算开销

2. **偏差缓解训练策略（Bias Mitigation Training）**:

    - 功能：训练主模型学习偏差不变的特征表示
    - 核心思路：定义主模型 $f_\theta$ 输出特征 $\mathbf{h}^{(i)}$ 和 logits $\mathbf{z}_{\text{main}}^{(i)}$。引入投影层 $g_\phi$ 将偏差 embedding $\mathbf{e}^{(i)}$ 投影到主模型特征空间，再通过分类头得到偏差 logits $\mathbf{z}_{\text{tag}}^{(i)}$。最终 logits 为 $\mathbf{z}^{(i)} = \mathbf{z}_{\text{main}}^{(i)} + \mathbf{z}_{\text{tag}}^{(i)}$
    - 设计动机：当样本高度偏差对齐时，$\mathbf{z}_{\text{tag}}$ 会很大，导致 $\mathbf{z}_{\text{main}}$ 对总 logits 的贡献减小，从而减小这些样本的梯度更新。这隐式地让模型减少对偏差特征的依赖

3. **Logit 对齐损失（Logit Alignment Loss）**:

    - 功能：平衡主模型和投影层的训练，防止一方主导
    - 核心思路：损失函数 $\mathcal{L} = \mathcal{L}_{cls}(\mathbf{z}^{(i)}, y^{(i)}) + \alpha \cdot \mathcal{L}_{align}$，其中对齐项 $\mathcal{L}_{align} = \frac{1}{2} \| \|\mathbf{z}_{\text{main}}^{(i)}\| - \lambda \cdot \|\mathbf{z}_{\text{tag}}^{(i)}\| \|^2$
    - 设计动机：$\lambda \in (0,1)$ 控制偏差 logits 与主 logits 的相对大小，偏差越强应设更小的 $\lambda$，使偏差对齐样本的梯度更小。$\alpha$ 平衡分类损失和对齐损失

### 损失函数 / 训练策略
使用 SGD（CelebA 用 Adam），学习率 0.001，每 1/3 epoch 衰减 10 倍。超参数 $(\alpha, \lambda)$ 对不同数据集分别调优。推理时只用主模型 $f_\theta$，投影层 $g_\phi$ 不参与，因此推理无额外开销。

## 实验关键数据

### 主实验（开放集评估）

| 数据集 | 指标 | MAVias | JTT (次优) | LfF | 提升(vs 次优) |
|--------|------|--------|-----------|-----|-------------|
| CelebA | WG Acc | **66.7%** | 31.5% | 14.7% | +35.2% |
| CelebA | Avg Acc | **81.4%** | 61.6% | 67.1% | +14.0% |
| Waterbirds | WG Acc | **75.4%** | 64.7% | 30.0% | +10.7% |
| Waterbirds | Avg Acc | **87.5%** | 85.2% | 72.7% | +2.3% |
| UrbanCars | WG Acc | **84.4%** | 69.0% | 34.6% | +15.4% |
| UrbanCars | Avg Acc | **89.3%** | 77.8% | 61.0% | +11.5% |
| ImageNet9 MIXED-NEXT | Acc | **88.26%** | 87.56% | 78.70% | +0.70% |
| ImageNet9 NO-FG | Acc | **53.02%** | 59.84% | 61.07% | -6.82%(↓更好) |
| ImageNet9 ONLY-BG-B | Acc | **21.83%** | 29.71% | 34.82% | -7.88%(↓更好) |

### 消融实验（偏差检测效果）

| 数据集 | 检测到的 Top 偏差标签 | 是否与已知偏差一致 |
|--------|----------------------|-------------------|
| CelebA | man, woman, suit, tie, dress | ✓ 发现了性别偏差 + 额外偏差 |
| Waterbirds | background (water, bamboo, branch) | ✓ 完美捕捉背景偏差 |
| UrbanCars | path, forest, hydrant, park | ✓ 捕捉城市/乡村背景偏差 |
| ImageNet9 | 每类 10 个无关标签 | 新发现（颜色、纹理、背景） |

### 关键发现
- **开放集场景下碾压式优势**：现有 BU 方法（LfF、JTT、Debian、FLAC-B）在多偏差场景下性能很差，MAVias 在所有数据集上均大幅领先
- **ImageNet9 的背景依赖大幅降低**：在 ONLY-BG-B（只有背景）测试集上，MAVias 的准确率从 vanilla 的 35.18% 降至 21.83%，说明模型不再依赖背景信息做预测
- **偏差发现超越预定义**：在 CelebA 上不仅发现了已知的性别偏差，还发现了服装（suit, tie）等新偏差源

## 亮点与洞察
- **基础模型组合的巧妙应用**：RAM + GPT-4o + OpenCLIP 三个基础模型各司其职，形成了从视觉特征提取 → 语义过滤 → 多模态编码的完整流水线。这种"基础模型工具链"的思路可以迁移到很多需要开放集理解的任务
- **实例级偏差建模**：不同于传统方法在数据集层面定义偏差（如 CelebA 的性别），MAVias 为每张图像独立建模偏差集合，能处理复杂的多属性偏差场景
- **推理零开销**：偏差投影层仅在训练时使用，推理时只需主模型，不增加任何计算或参数

## 局限与展望
- **依赖 GPT-4o 的标签筛选**：标签相关性判断依赖 LLM 的常识，可能存在误判。更换 LLM 或使用开源替代品的效果未探索
- **RAM 标签词汇量有限**：4000+ 标签虽然丰富，但仍可能遗漏某些细粒度偏差
- **超参数敏感性**：$(\alpha, \lambda)$ 需要针对每个数据集调优，增加了使用门槛
- **未在大规模生成任务中验证**：只聚焦于分类任务，在检测、分割、生成等任务中的效果未知

## 相关工作与启发
- **vs LfF/JTT**：这些 BU 方法通过训练偏差模型获取伪标签，只能处理单一显著偏差。MAVias 用基础模型直接发现多属性偏差，无需训练偏差模型
- **vs FLAC**：FLAC 需要间接访问偏差标签，仍局限于预定义偏差。MAVias 完全开放集，无需任何偏差先验知识
- **vs OpenBias**：OpenBias 在文生图领域做开放集偏差检测，但仅用文本描述缺乏视觉 grounding。MAVias 从图像出发做标签提取，再用 LLM 筛选，视觉 grounding 更强

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出开放集视觉偏差缓解框架，创造性地组合多个基础模型
- 实验充分度: ⭐⭐⭐⭐ 4 个数据集、开放集 + 封闭集双协议评估，但缺少更多任务类型验证
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，方法直觉解释到位
- 价值: ⭐⭐⭐⭐⭐ 开放集偏差缓解是重要且被忽视的方向，实用价值高
