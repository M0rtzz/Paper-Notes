---
title: >-
  [论文解读] PLeaS: Merging Models with Permutations and Least Squares
description: >-
  [CVPR 2025][模型合并] 提出 PLeaS，一种两步模型合并算法：第一步利用置换对称性部分匹配两个模型的特征（相似特征合并、不相似特征保留），第二步通过逐层最小二乘优化使合并模型的特征逼近原模型置换后的集成特征，在相同模型大小下比现有方法提升高达 15 个百分点。 1. 领域现状：随着开源模型生态的繁荣…
tags:
  - "CVPR 2025"
  - "模型合并"
  - "置换对称性"
  - "最小二乘"
  - "部分合并"
  - "特征蒸馏"
---

# PLeaS: Merging Models with Permutations and Least Squares

**会议**: CVPR 2025  
**arXiv**: [2407.02447](https://arxiv.org/abs/2407.02447)  
**代码**: [https://github.com/SewoongLab/PLeaS-Merging](https://github.com/SewoongLab/PLeaS-Merging)  
**领域**: 其他 / 模型合并  
**关键词**: 模型合并, 置换对称性, 最小二乘, 部分合并, 特征蒸馏

## 一句话总结
提出 PLeaS，一种两步模型合并算法：第一步利用置换对称性部分匹配两个模型的特征（相似特征合并、不相似特征保留），第二步通过逐层最小二乘优化使合并模型的特征逼近原模型置换后的集成特征，在相同模型大小下比现有方法提升高达 15 个百分点。

## 研究背景与动机

1. **领域现状**：随着开源模型生态的繁荣，大量在特定任务/数据上微调的专用模型涌现（如 Code Llama、Vicuna 等）。如何将这些专用模型的能力合并到一个通用模型中，避免推理时存储和运行多个模型的开销，是一个重要的实际问题。

2. **现有痛点**：（a）传统集成方法需要存储所有模型，内存开销大；（b）现有模型合并方法（如 Task Vectors、TIES）通常限于**从同一base模型微调**的情况，无法合并不同初始化的模型；（c）ZipIt! 等方法支持不同初始化但性能有限；（d）大部分方法要求合并后模型与原模型大小相同，无法灵活权衡模型大小与性能。

3. **核心矛盾**：两个在不同数据上训练的模型，其特征空间差异很大——直接权重平均会导致"破坏性干扰"。但特征中也存在**重叠部分**可以合并。关键在于如何识别并处理"可合并"vs"不可合并"的特征。

4. **本文目标**：设计一种可以合并**不同初始化**模型的方法，支持灵活控制合并后模型大小（1x 到 2x之间），并且可在没有训练数据的情况下工作。

5. **切入角度**：利用神经网络的置换不变性（permutation symmetry）——以任意顺序排列隐层神经元不改变模型功能。通过找到最佳置换后，相似特征平均合并，不相似特征分别保留。

6. **核心 idea**：两步合并：（1）Permutation 步——扩展 Git Re-Basin 支持部分合并，控制每层合并比例；（2）Least Squares 步——逐层求解最小二乘使合并模型特征逼近原模型置换集成。

## 方法详解

### 整体框架
输入为两个同架构的模型 A 和 B（可以是不同初始化、不同训练数据），以及目标计算预算 B。第一步：用 activation matching（或 weight matching）找到模型 B 每层的置换矩阵 $P_i$，按特征相似度选择 $k_i$ 个最匹配的特征合并，其余保留。第二步：逐层求解 least squares 问题使合并模型层的输出逼近两个原模型置换集成的输出。最终输出一个宽度为 $2d_i - k_i$ 的合并模型。

### 关键设计

1. **部分置换合并（Partial Permutation Merging）**:
    - 功能：在每一层灵活选择合并多少特征，控制合并后模型大小
    - 核心思路：扩展 Git Re-Basin 的全量置换为部分置换。给定置换矩阵 $P_i$，选择使 $\|Z_{J,i}^A - (P_i Z_{:,i}^B)_J\|_F^2$ 最小的 $k_i$ 个索引作为合并特征，其余 $d_i - k_i$ 个特征在两个模型中分别保留。合并后第 $i$ 层宽度为 $2d_i - k_i$。不同层的合并比例 $k_i/d_i$ 可独立选择，通过优化一个代理目标在目标计算预算约束下自动确定
    - 设计动机：当两个模型训练数据差异大时，强制合并所有特征会导致破坏性干扰。保留不兼容特征是关键，而 ZipIt! 只能对前缀层统一1x、后缀层统一2x，灵活性不足

2. **置换最小二乘（Permuted Least Squares）**:
    - 功能：优化合并模型的权重使其特征逼近原模型置换集成的效果
    - 核心思路：对每层独立求解 $W_i^M = \arg\min_W \|(Z_i^A + P_i Z_i^B)W - (Z_{i+1}^A + P_{i+1} Z_{i+1}^B)\|^2$。即让合并模型的第 $i$ 层将两模型置换后的集成输入 $\tilde{Z}_i$ 映射到集成输出 $\tilde{Z}_{i+1}$。这等价于逐层模仿集成模型的行为。虽然可以 OLS 闭式求解，实际用梯度下降实现以兼容卷积层，由于目标函数凸性，不到 100 步即可收敛
    - 设计动机：仅靠置换+权重平均（Git Re-Basin）在模型差异大时性能损失严重——平均权重会退化特征质量。通过 least squares 匹配集成特征可以大幅恢复性能。相比 RegMean（不做置换直接最小二乘），置换后让特征对齐再拟合效果好得多

3. **无数据变体 PLeaS-free**:
    - 功能：在无法获取训练领域数据时仍可合并模型
    - 核心思路：用公开通用数据集（如 ImageNet）替代训练领域数据来计算 activation matching 和 least squares。实验表明在 1.2x 模型大小时性能损失小于 2%
    - 设计动机：实际场景中训练数据可能因隐私或商业原因不可用，PLeaS-free 大大扩展了方法的适用范围

### 损失函数 / 训练策略
- PLeaS 不需要梯度训练：置换步通过匈牙利算法求解线性分配问题；least squares 步求解凸优化
- 合并后重新计算 batch-norm 统计量（同 REPAIR 方法的建议）
- 除了目标计算预算外，PLeaS 无任何超参数

## 实验关键数据

### 主实验

DomainNet 共享标签空间（ResNet-50, 1x 大小）:

| 方法 | Clipart | Infograph | Painting | Real | 平均 |
|------|---------|-----------|----------|------|------|
| Simple Avg | 1.2 | 0.8 | 1.9 | 2.1 | 1.5 |
| Git Re-Basin | 18.2 | 7.8 | 18.8 | 26.5 | 17.8 |
| ZipIt! | 26.9 | 12.2 | 27.1 | 37.4 | 25.9 |
| MuDSC | 34.0 | 14.3 | 29.5 | 45.1 | 30.7 |
| **PLeaS** | **41.7** | **16.9** | **40.8** | **55.1** | **38.6** |

不同标签空间（CUB/Pets/Dogs/NABirds, 线性探测）:

| 方法 | CUB | Pets | Dogs | NABird | 平均 |
|------|-----|------|------|--------|------|
| ZipIt! | 67.5 | 83.6 | 60.0 | 56.3 | 66.9 |
| MuDSC | 70.1 | 82.5 | 63.2 | 58.2 | 68.5 |
| **PLeaS** | **75.2** | **85.0** | **69.6** | **69.7** | **74.9** |

### 消融实验

| 配置 | DomainNet 平均 | 说明 |
|------|---------------|------|
| 仅 Permutation（无 LS）| ~17.8 | Git Re-Basin baseline |
| + Least Squares (PLeaS) | 38.6 | LS 步带来约 +20% 提升 |
| RegMean（LS 无置换）| 12.1 | 不做置换直接 LS 效果很差 |
| PLeaS-free (用ImageNet) | ~37.5 | 仅损失约 1% |
| PLeaS 1.2x 大小 | 接近 Ensemble 2x | 20% 额外参数大幅缩小差距 |

### 关键发现
- Least Squares 步是关键贡献，在 DomainNet 上将 Git Re-Basin 的准确率几乎翻倍（17.8→38.6）
- 置换是 Least Squares 的必要前提——RegMean（无置换LS）仅 12.1%，远低于 PLeaS
- 部分合并的灵活性很重要：仅增加 20% 参数（1.2x）就能接近 2x 集成的性能
- PLeaS-free 使用 ImageNet 数据几乎不损失性能（<1%），实用性极强
- 随着模型规模增大（ResNet-18→50→101），PLeaS 的优势更加突出
- PLeaS 在 ViT 模型上同样有效

## 亮点与洞察
- **两步法的互补性设计极精妙**：置换解决特征对齐问题，least squares 解决权重退化问题，二者缺一不可。RegMean（仅LS）和 Git Re-Basin（仅置换）的对比清楚展示了这一点
- **部分合并的自由度**：每层可独立选择合并比例是关键创新，不同层的特征兼容程度不同，uniform 策略（如 ZipIt!）会导致次优。通过代理目标自动搜索最佳配置，消除超参数
- **PLeaS-free 概念**：使用通用数据集替代领域数据进行合并，几乎不损失性能。这打破了"需要训练数据"的限制，大幅扩展了实际应用场景
- **理论统一视角**：将模型合并 formulate 为逐层近似集成模型，提供了清晰的优化目标

## 局限与展望
- 目前仅验证了同架构模型的合并，异构架构（如 CNN 和 ViT）的合并未涉及
- Least Squares 步需要前向传播计算 activation，对超大模型（LLM）的计算开销需评估
- 仅在图像分类任务上验证，扩展到 NLP/多模态任务的效果待验证
- 文中提到 Code Llama 和 Vicuna 的合并场景但未实际实验
- 部分合并的最优层宽度配置搜索算法的效率和最优性值得进一步研究

## 相关工作与启发
- **vs Git Re-Basin**: Git Re-Basin 只做全量置换后平均权重，在不同训练数据的模型上性能很差。PLeaS 的部分置换+LS两步走将其从 17.8% 提升到 38.6%
- **vs ZipIt!**: ZipIt! 也支持部分合并但只能对前缀层用 1x、后缀层用 2x，粒度太粗。PLeaS 每层独立控制，在相同预算下性能高出 10%+
- **vs RegMean**: RegMean 做逐层 LS 但不做置换，导致两个模型的特征未对齐就直接拟合，效果远不如 PLeaS
- **vs Task Vectors**: Task Vectors 需要共享 base model，无法处理不同初始化的情况。PLeaS 放宽了这一限制

## 评分
- 新颖性: ⭐⭐⭐⭐ 部分置换+逐层最小二乘的组合是对模型合并的重要推进
- 实验充分度: ⭐⭐⭐⭐⭐ DomainNet 4域、4个细粒度数据集、3种ResNet+ViT、多种基线、PLeaS-free消融、模型大小-性能权衡曲线，非常全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，图示直观，公式推导简洁
- 价值: ⭐⭐⭐⭐ 解决了不同初始化模型合并的实际痛点，PLeaS-free 进一步提升了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](../../ICLR2026/others/a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)
- [\[CVPR 2025\] On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](../../ACL2025/others/bone_soups_multi_objective_gen.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)

</div>

<!-- RELATED:END -->
