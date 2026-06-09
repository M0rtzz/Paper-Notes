---
title: >-
  [论文解读] What Does It Take to Build a Performant Selective Classifier?
description: >-
  [NeurIPS 2025][选择性分类] 首次对选择性分类的性能差距（selective classification gap）进行有限样本分解，将差距归因于五个源头——贝叶斯噪声、逼近误差、排序误差、统计噪声和实现偏差，并证明单调校准方法对缩小差距效果有限。
tags:
  - "NeurIPS 2025"
  - "选择性分类"
  - "置信度校准"
  - "oracle bound"
  - "误差分解"
  - "不确定性估计"
---

# What Does It Take to Build a Performant Selective Classifier?

**会议**: NeurIPS 2025  
**arXiv**: [2510.20242](https://arxiv.org/abs/2510.20242)  
**代码**: 暂无  
**领域**: 可靠机器学习 / 选择性分类  
**关键词**: 选择性分类, 置信度校准, oracle bound, 误差分解, 不确定性估计

## 一句话总结

首次对选择性分类的性能差距（selective classification gap）进行有限样本分解，将差距归因于五个源头——贝叶斯噪声、逼近误差、排序误差、统计噪声和实现偏差，并证明单调校准方法对缩小差距效果有限。

## 研究背景与动机

在医疗、金融、自动驾驶等高风险场景中，模型需要能够在不确定时"拒绝作答"（abstain），即选择性分类。其核心评估指标是**准确率-覆盖率权衡**：随着模型接受更多输入，准确率如何变化。理论上，存在一个"完美排序 oracle"，它按照真实正确概率排序所有样本，给出一个上界。

**现有理论的局限**：

- 经典的可实现设置（realizable setting）假设数据无噪声且真实预测器在假设类内，过于理想化
- 不可知设置（agnostic setting）中基准本身可能远低于 oracle，且不区分差距的来源
- 实践中模型容量有限、数据有限、存在分布偏移，渐进保证缺乏操作指导

**核心问题**：对于一个有限模型在有限数据上，学习设置的哪些方面真正决定了准确率-覆盖率曲线与 oracle 上界的距离？

**切入角度**：将定性问题转化为定量诊断——定义 coverage-uniform 的选择性分类差距 $\Delta(c)$，并将其分解为五个可测量、可改进的误差项。

## 方法详解

### 整体框架

选择性分类器是一对 $(h, g)$，其中 $h$ 是分类器，$g$ 是选择函数（输出置信度分数）。给定阈值 $\tau$，当 $g(x,h) \geq \tau$ 时输出预测，否则拒绝。核心指标是选择性分类差距：

$$\Delta(c) = \overline{\mathrm{acc}}(a_{\text{full}}, c) - \mathrm{acc}_c(h, g)$$

其中 $\overline{\mathrm{acc}}$ 是完美排序 oracle 的准确率上界（Definition 3）。

### 关键设计

1. **有限样本差距分解（Theorem 1）**：以概率 $1-\delta$：

$$\hat{\Delta}(c) \leq \underbrace{\varepsilon_{\text{Bayes}}(c)}_{\text{不可约}} + \underbrace{\varepsilon_{\text{approx}}(c)}_{\text{容量}} + \underbrace{\varepsilon_{\text{rank}}(c)}_{\text{排序}} + \underbrace{\varepsilon_{\text{stat}}(c)}_{\text{统计}} + \underbrace{\varepsilon_{\text{misc}}(c)}_{\text{优化与偏移}}$$

   各项定义：
    - $\varepsilon_{\text{Bayes}}(c) = \mathbb{E}[1-\max\{\eta(X), 1-\eta(X)\} \mid X \in A_c]$：接受区域内数据固有的标签不确定性
    - $\varepsilon_{\text{approx}}(c) = \mathbb{E}[|\eta_h(X) - \eta(X)| \mid X \in A_c]$：模型假设类无法逼近贝叶斯最优的程度
    - $\varepsilon_{\text{rank}}(c) = \mathbb{E}[\eta_h \mid A_c^*] - \mathbb{E}[\eta_h \mid A_c]$：置信度分数排序与真实正确性排序的偏差
    - $\varepsilon_{\text{stat}}(c) = C\sqrt{\log(1/\delta)/n}$：有限验证集的采样波动
    - $\varepsilon_{\text{misc}}(c)$：优化误差 + 分布偏移

2. **单调校准的有限效果（Section 3.4）**：关键洞察——单调后处理校准（如等序回归、温度缩放中的单调部分）保持分数排序不变，因此 $A_c$ 集合不变，$\Delta(c)$ 不变。虽然温度缩放通过 softmax 的非线性可能产生**微弱**的非单调重排效应，但本质上受限。真正减小差距需要能改变排序的方法：

    - Deep Ensembles：通过多模型聚合改变排序
    - SAT：通过重标记改变排序
    - 特征感知校准头：利用隐层特征直接预测正确性

3. **排序距离的刻画（Remark）**：定义 mis-ordered mass：

$$D_{\text{rank}}(c) = \Pr(X \in A_c^* \setminus A_c) + \Pr(X \in A_c \setminus A_c^*)$$

   即需要在 $A_c$ 和 $A_c^*$ 之间交换的样本总概率。当 $D_{\text{rank}} = 0$ 时 $\varepsilon_{\text{rank}} = 0$。

### 可操作的设计指南

- 减少 $\varepsilon_{\text{Bayes}}$：额外标注、噪声鲁棒损失函数
- 减少 $\varepsilon_{\text{approx}}$：增加模型容量、从更强模型蒸馏
- 减少 $\varepsilon_{\text{rank}}$：Deep Ensembles、学习型正确性预测头
- 减少 $\varepsilon_{\text{stat}}$：增大验证集
- 减少 $\varepsilon_{\text{misc}}$：领域自适应、重要性加权

## 实验关键数据

### 主实验：CIFAR-100 上的校准与选择性分类

| 架构 | 方法 | E-AURC↓ | ECE↓ | 说明 |
|------|------|---------|------|------|
| CNN | MSP | 0.086 | 0.142 | 基线 |
| CNN | TEMP | 0.085 | **0.008** | ECE 大幅改善但 E-AURC 几乎不变 |
| CNN | SAT | 0.081 | 0.116 | 通过重标记同时改善两者 |
| CNN | **DE** | **0.065** | 0.019 | 集成方法最显著缩小差距 |
| ResNet-18 | MSP | 0.033 | 0.052 | 更强容量降低逼近误差 |
| ResNet-18 | **DE** | **0.026** | 0.034 | 最佳 |
| WRN-50 | MSP | 0.031 | 0.066 | |
| WRN-50 | **DE** | **0.026** | **0.030** | |

**核心发现**：温度缩放使 ECE 从 0.142 降到 0.008（17倍改善），但 E-AURC 仅从 0.086 降到 0.085，**几乎无效**。

### 消融实验：误差源头分离

| 实验设置 | 关键观察 | 对应误差项 |
|------|---------|------|
| Two moons 噪声 σ=0.1→1.5 | 准确率-覆盖率曲线系统性下移 | $\varepsilon_{\text{Bayes}}$ |
| Two moons: 逻辑回归→MLP | MLP 显著缩小差距 | $\varepsilon_{\text{approx}}$ |
| CIFAR-10N/100N 噪声标签 | 最嘈杂的 50% 样本差距最大 | $\varepsilon_{\text{Bayes}}$ |
| CNN→ResNet→WRN | 容量越大差距越小 | $\varepsilon_{\text{approx}}$ |
| CIFAR-10C 腐蚀 severity 1→5 | 差距随偏移强度增大 | $\varepsilon_{\text{misc}}$ |
| Camelyon17-WILDS 真实偏移 | 差距显著增大 | $\varepsilon_{\text{misc}}$ |

### 关键发现

- 贝叶斯噪声和逼近误差是差距的主要驱动因素（从 two moons 到 CIFAR 均得到验证）
- 温度缩放改善校准但**不改善排序**，对选择性分类几乎无帮助
- 只有能**改变排序**的方法（SAT、DE）才能实质性缩小差距
- 分布偏移引入独立的松弛项，需要专门的 robust training 来应对

## 亮点与洞察

- **理论与实践的完美桥接**：分解不仅是理论工具，每个误差项都直接对应可测量的实验和可执行的改进方向
- **"校准不等于好的选择性分类"** 是一个重要的实践洞察，打破了常见误解
- **误差预算（error budget）** 的视角使得实践者可以量化诊断瓶颈并据此分配改进资源
- 与 multicalibration / loss prediction 的联系（Section 3.4 末尾）提供了自检机制

## 局限与展望

- 五个误差项之间存在交互（如增加容量同时影响逼近和排序），无法完全独立归因
- SAT、mixup、focal loss 等训练时校准方法同时影响排序和全覆盖准确率，混淆了 budget 分离
- 主要在合成和视觉基准上验证，大语言模型上仅有初步探索（附录 F.2）
- oracle bound 和 gap 定义基于 0-1 损失，推广到非对称或类别相关代价函数需要额外工作
- 未讨论分布外拒绝（OOD rejection）与选择性分类的统一框架

## 相关工作与启发

- 与 Geifman et al. (2019) 的 oracle bound 的关系：本文的 E-AURC 等价于其定义，但提供了更细粒度的分解
- 与 AUGRC (Traub et al., 2024) 的关系：后者通过 coverage 加权避免偏向低覆盖区域，本文的分解可互补
- Deep Ensembles 在选择性分类中的优势可解释为：多模型平均提供了更好的正确性后验估计，直接改善排序
- "loss prediction = multicalibration" 的等价性（Gollakota et al., 2025）提供了一个新的模型自评估视角

## 评分

- 新颖性: ⭐⭐⭐⭐ 差距分解思路并非全新（类似偏差-方差分解），但五项有限样本分解和校准分析是原创
- 实验充分度: ⭐⭐⭐⭐⭐ 从 two moons 到 CIFAR-10C/100N 到 Camelyon17，逐一验证每个误差项
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，图 1 直观，每节末尾的 Takeaway 非常有用
- 价值: ⭐⭐⭐⭐⭐ 提供了可操作的设计指南，对任何需要可靠预测的实际部署场景都有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Processing and Acquisition Traces in Visual Encoders: What Does CLIP Know About Your Camera?](../../ICCV2025/others/processing_and_acquisition_traces_in_visual_encoders_what_does_clip_know_about_y.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [\[ACL 2025\] Interlocking-free Selective Rationalization Through Genetic-based Learning](../../ACL2025/others/interlocking-free_selective_rationalization_through_genetic-based_learning.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[ICML 2025\] Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Settings](../../ICML2025/others/suitability_filter_a_statistical_framework_for_classifier_evaluation_in_real-wor.md)

</div>

<!-- RELATED:END -->
