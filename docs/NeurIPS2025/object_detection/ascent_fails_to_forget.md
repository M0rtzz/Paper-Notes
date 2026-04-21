---
title: >-
  [论文解读] Ascent Fails to Forget
description: >-
  [NeurIPS 2025][目标检测][machine unlearning] 本文从遗忘集与保留集之间的统计依赖出发，理论结合实验证明广泛使用的梯度上升/Descent-Ascent（DA）类机器遗忘方法在存在数据相关性时会系统性失败——在 logistic 回归中 DA 解甚至会比原始模型更远离 oracle，且在非凸设置下会将模型困在劣质局部最小值中。
tags:
  - NeurIPS 2025
  - 目标检测
  - machine unlearning
  - 梯度上升
  - 统计依赖
  - descent-ascent
  - logistic 回归
---

# Ascent Fails to Forget

**会议**: NeurIPS 2025  
**arXiv**: [2509.26427](https://arxiv.org/abs/2509.26427)  
**代码**: 无  
**领域**: 机器遗忘 / 可信机器学习  
**关键词**: machine unlearning, 梯度上升, 统计依赖, descent-ascent, logistic 回归

## 一句话总结

本文从遗忘集与保留集之间的统计依赖出发，理论结合实验证明广泛使用的梯度上升/Descent-Ascent（DA）类机器遗忘方法在存在数据相关性时会系统性失败——在 logistic 回归中 DA 解甚至会比原始模型更远离 oracle，且在非凸设置下会将模型困在劣质局部最小值中。

## 研究背景与动机

**领域现状**：机器遗忘（machine unlearning）是一个快速发展的研究方向，旨在从训练好的模型中移除特定训练样本的影响，应用场景涵盖数据隐私合规（GDPR "被遗忘权"）、有毒/过时数据去除、版权保护以及 LLM 对齐等。理想的遗忘结果是：遗忘后的模型 $h_\theta^{\mathrm{UL}}$ 与仅在保留集 $\mathcal{R} = \mathcal{D} \setminus \mathcal{F}$ 上从头训练的 oracle 模型行为一致。从头重训是金标准但在大规模模型上计算成本过高。

**现有痛点**：在凸模型上已有基于噪声梯度下降（不含上升步）的可证明遗忘算法。但对深度神经网络，由于模型的非凸、非光滑、高维特性，缺乏可证明保证。实践中最广泛使用的是 Descent-Ascent（DA）方法：在遗忘集 $\mathcal{F}$ 上做梯度上升（"忘记"这些样本）+ 在保留集 $\mathcal{R}$ 上做梯度下降（"保持"模型性能）。然而，近期评测基准反复表明 DA 方法高度不可靠：(1) 没有理论性能保证；(2) 对学习率和微调时长极度敏感；(3) 缺乏明确的停止准则。

**核心矛盾**：DA 方法隐含一个根本假设——遗忘集和保留集可以被独立操纵。但在现实中，两者来自同一数据分布，必然存在统计依赖关系。当我们在遗忘集上做梯度上升降低其指标时，由于统计相关性，保留集和测试集的指标也不可避免地受损。

**本文目标** 识别并理论化 DA 方法失败的根本原因——数据集之间被忽视的统计依赖。具体分解为：(1) 随机遗忘集下 DA 是否必然损害模型？(2) 在可解析的 logistic 回归中，DA 解与 oracle 解的关系是什么？(3) 在非凸问题中，DA 造成的损害能否通过后续微调恢复？

**切入角度**：作者没有提出新的遗忘方法，而是从分析角度系统揭示 DA 失败的本质原因。从最简单的随机集开始（天然高相关），到 logistic 回归（可解析的凸问题），再到低维非凸例子（局部最小值陷阱），逐层递进地论证。

**核心 idea**：遗忘集和保留集的统计依赖（即使只是简单相关性）足以让基于梯度上升的遗忘方法系统性失败，"什么都不做"反而比执行 DA 遗忘更好。

## 方法详解

### 整体框架

这是一篇理论分析型论文。作者构建了三个递进的分析层次：(1) 随机遗忘集的概率分析，证明 oracle 在遗忘集上的表现必须与测试集一致；(2) 高维 logistic 回归的闭式解分析，证明 DA 解与 oracle 解相对于原始模型在相反方向；(3) 低维非凸的 toy example，展示 DA 将模型困在错误的局部最小值。在理论之外，通过 ResNet-9/ResNet-18 在 CIFAR-10/ImageNetLiving-17 上的实验验证理论预测。评估指标采用 KLoM（KL Divergence of Margins），量化遗忘模型与 100 个 oracle 模型之间的预测分布距离。

### 关键设计

1. **随机遗忘集的不可能性定理 (Lemma 1)**:

    - 功能：证明对随机遗忘集，任何成功的遗忘算法都不应降低遗忘集上的指标
    - 核心思路：当 $\mathcal{F}$ 从 $\mathcal{D}$ 中均匀随机选取时，oracle 模型在遗忘集和测试集上的准确率差异有概率上界 $P(|\mathrm{Acc}_{\mathcal{T}} - \mathrm{Acc}_{\mathcal{F}}| \geq \epsilon) \leq 2\exp(-2|\mathcal{F}|\epsilon^2)$（基于 Hoeffding 不等式）。这意味着 oracle 在遗忘集上的表现应与测试集几乎相同。因此，任何通过降低遗忘集指标来"遗忘"的方法都会使模型偏离 oracle
    - 设计动机：直接否定了 DA 方法的核心操作逻辑——"降低遗忘集上的表现 = 成功遗忘"

2. **Logistic 回归的闭式分析 (Lemma 2-5)**:

    - 功能：在凸设置下精确刻画 DA 解、oracle 解和原始模型解的相对位置
    - 核心思路：考虑带岭正则的二元 logistic 回归，数据满足半正交假设（不同坐标轴上的样本正交，同一轴上可相关）。定义遗忘集比例 $|\mathcal{F}_j| = \alpha \cdot |\mathcal{R}_j|$。通过 Lambert-W 函数得到三个问题的闭式解：$w_j^{\mathcal{D}} = W\left(\frac{(1+\alpha)|\mathcal{R}_j|}{\lambda|\mathcal{D}|}\right)$，$w_j^{\mathcal{R}} = W\left(\frac{|\mathcal{R}_j|}{\lambda|\mathcal{R}|}\right)$，$w_j^{\text{DA}} = W\left(\frac{(1-\alpha|\mathcal{R}|/|\mathcal{F}|)|R_j|}{\lambda|R|}\right)$。关键结论（Lemma 3）：$(w_j^{\text{DA}} - w_j^{\mathcal{D}}) \cdot (w_j^{\mathcal{D}} - w_j^{\mathcal{R}}) \geq 0$，即 DA 解和 oracle 解位于原始模型解的两侧——DA 把模型推向了与 oracle 相反的方向
    - 设计动机：在最简单的凸设置下就能证明 DA 的方向性错误，说明问题不在于非凸性或超参选择，而在于数据依赖本身

3. **跨维度相关性分析 (Lemma 6-10)**:

    - 功能：将单维度结论推广到有跨维度相关性 $\epsilon$ 的二维情况
    - 核心思路：考虑两组样本，$x_i = (1, \epsilon)$ 在保留集中，$x_j = (\epsilon, 1)$ 在遗忘集中，$\epsilon$ 控制两组之间的相关程度。通过坐标变换得到 oracle、原始模型和 DA 解的闭式表达，证明存在一个 $\alpha$ 的区间使得 DA 是有害的。数值结果表明这个有害区间通常很大，且随遗忘比例增加而扩大，在弱相关时窗口更宽
    - 设计动机：现实数据中的遗忘/保留集之间的相关性不总是沿同一维度，需要证明跨维度的弱相关性同样会导致 DA 失败

### 损失函数 / 训练策略

三种优化目标的对比：预训练 $\mathcal{L}_{\mathcal{D}}$ 对全集做梯度下降；oracle $\mathcal{L}_{\mathcal{R}}$ 对保留集做梯度下降；DA $\mathcal{L}_{\text{DA}}$ 对保留集下降 + 遗忘集上升。DA 的损失为 $\mathcal{L}_{\text{DA}} = \frac{1}{|\mathcal{R}|}\sum_{\mathcal{R}} e^{-y_i\langle\mathbf{w},\mathbf{x}_i\rangle} - \frac{1}{|\mathcal{F}|}\sum_{\mathcal{F}} e^{-y_i\langle\mathbf{w},\mathbf{x}_i\rangle} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$，关键问题在于减号项——它不是消除影响而是主动推向相反方向。

低维非凸 toy example 使用 MSE 损失 + 岭正则的 sigmoid 网络 $h_\theta(\mathbf{x}_i) = \sigma(ax_i + bx_i^2)$，4 个带权重的样本。GDA 将遗忘样本的梯度归零（正反梯度抵消），导致模型从全局最优滑向捕获所有样本但实际准确率更低的局部最优决策边界。

## 实验关键数据

### 主实验（ResNet-9 / CIFAR-10）

使用 KLoM 指标评估，KLoM 趋近 0 表示完美遗忘。实验测试了 GA（仅梯度上升）和 GDA（上升 + 下降）在不同遗忘集上的表现。

| 遗忘集类型 | GA/GDA 行为 | 遗忘质量 (KLoM) | 模型性能 |
|-----------|-------------|----------------|---------|
| 高影响力样本 (PC1) | 几乎无遗忘 | 远离 oracle | 严重退化或无变化 |
| 随机 10 个样本 | 部分 run 看似成功 | 需精确选择超参+停止点 | 无法先验确定超参 |
| PC2 样本 | GDA 部分改善 | 遗忘成本 ~25% 重训 | 仅 0.2% 数据的遗忘 |

### 理论结果汇总（消融分析）

| 分析层次 | 设置 | 关键结论 | 含义 |
|---------|------|---------|------|
| Lemma 1 | 随机遗忘集 | $\mathrm{Acc}_{\mathcal{F}} \approx \mathrm{Acc}_{\mathcal{T}}$ (oracle) | DA 降低遗忘集指标 = 降低测试性能 |
| Lemma 3 | 1D logistic 回归 | DA 解与 oracle 在原始模型两侧 | 每一步 DA 都离 oracle 更远 |
| Corollary 1 | $\lambda \to 0$ | $\Delta_{\mathcal{R},\mathcal{D}} \to 0$, $\Delta_{\mathcal{R},\text{DA}} \to \infty$ | DA 方法极度不稳定 |
| Lemma 9-10 | 2D 跨维度相关 | 存在结构性有害的 $\alpha$ 区间 | 弱相关也导致 DA 失败 |
| Toy example | 非凸 sigmoid | DA 落入错误局部最小值 | 后续微调无法恢复 |

### 关键发现

- **DA 在绝大多数超参配置下要么无效（模型不动）要么有害（模型崩溃）**：Fig. 1 显示 GA 和 GDA 的运行结果要么紧贴预训练初始点，要么测试性能严重退化，几乎没有中间状态
- **"遗忘幻觉"（Ascent Forgets Illusion）**：当遗忘集很小（如 10 个随机样本）时，大量超参搜索中偶尔出现看似成功的 run，但这需要同时完美选择学习率和停止时间——没有先验方式确定这些超参，本质上是 cherry-picking
- **遗忘难度高度依赖遗忘集的选择**：高影响力样本（影响矩阵的第一主成分）最难遗忘，从未出现任何成功 run；随机样本偶尔成功但不可靠
- **Corollary 1 的含义深远**：当正则化趋近零（深度学习中常见），oracle 和原始模型的差异趋于零，但 DA 解与 oracle 的距离趋于无穷——DA 的不稳定性随正则化减弱而激增

## 亮点与洞察

- **统计依赖视角的根本性**：之前文献将 DA 失败归因于超参敏感性、非凸性或缺乏停止准则，本文指出即使在凸 + 最优超参的设置下 DA 也必然失败。根本原因不在算法细节而在数据结构——这是一个无法通过调参解决的根本性障碍
- **"什么都不做比 DA 更好"的反直觉结论**：Lemma 3 表明 DA 将模型推到原始模型的另一侧（相对于 oracle），所以保持原始模型不动反而更接近 oracle。这对实践者来说是重要警示
- **分析方法可迁移**：Lambert-W 函数的闭式分析技术以及坐标变换处理跨维度相关性的方法，可以应用于其他优化问题中数据依赖性的分析

## 局限与展望

- **纯负面结论，未提出替代方案**：论文系统性地拆解了 DA 方法，但没有提出"那应该怎么做"。作者简要提及了 rewind 方法和基于噪声的随机方法，但未深入
- **理论假设较强**：半正交数据假设（Assumption 1-2）在真实数据中很少严格满足；logistic 回归使用指数损失作为逻辑损失的代理，可能引入近似误差
- **非随机遗忘集分析不够深入**：实际应用中遗忘集通常不是随机的（如特定用户、特定类别），论文对这类结构化遗忘集的理论分析有限（主要在 toy example 中涉及）
- **深度网络的理论扩展**：理论分析集中于 logistic 回归和 2 参数 sigmoid 网络，向深度网络的理论推广是重要的未来方向

## 相关工作与启发

- **vs Certified Unlearning (Neel et al. 2021, Guo et al. 2023)**：这些方法基于噪声下降（无上升步），在凸模型上有理论保证。本文的分析从反面说明了为什么"上升步"是有害的——不是因为它"不够好"而是因为它"方向错误"
- **vs Rewind 方法 (Mu & Klabjan 2024)**：提供非凸模型的理论保证，但需要存储完整模型状态或大量近端点迭代，计算成本高。本文的结论支持 rewind 类方法的设计理念（避免上升步）
- **vs SCRUB (Kurmanji et al. 2024)**：使用 KL 散度目标的微调方法，本文认为它面临与其他 DA 方法相同的根本挑战
- **vs Georgiev et al. (2024) 的预测数据归因方法**：也观察到 DA 经验性失败，但未给出统计依赖的理论解释。本文提供了可控实验和理论来归因这些失败

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从统计依赖角度解释 DA 失败是全新且深刻的视角，凸设置下的反直觉结论非常有力
- 实验充分度: ⭐⭐⭐⭐ 理论 + ResNet/ViT 实证覆盖全面，但缺少替代方案的对比和更多真实场景验证
- 写作质量: ⭐⭐⭐⭐⭐ 论证从简单到复杂层层递进，数学严谨且直觉明确，图表设计清晰
- 价值: ⭐⭐⭐⭐⭐ 对机器遗忘领域具有深远警示意义，有望改变后续方法的设计思路

<!-- RELATED:START -->

## 相关论文

- [All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)
- [Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)
- [XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)

<!-- RELATED:END -->
