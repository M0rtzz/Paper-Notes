---
title: >-
  [论文解读] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning
description: >-
  [CVPR 2026][自监督学习][active learning] 提出 BoSS——一种可扩展的 oracle 策略选择框架：在每轮主动学习中，并行运行多种查询策略在随机子池上生成候选 batch，通过冻结 backbone 仅重训最后一层快速评估每个候选 batch 的性能增益，选出最优 batch…
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "active learning"
  - "Oracle Strategy"
  - "Strategy Selection"
  - "Deep Learning"
  - "benchmark"
---

# BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning

**会议**: CVPR 2026  
**arXiv**: [2603.13109](https://arxiv.org/abs/2603.13109)  
**代码**: 待确认  
**领域**: 自监督学习 / 主动学习  
**关键词**: active learning, Oracle Strategy, Strategy Selection, Deep Learning, benchmark

## 一句话总结

提出 BoSS——一种可扩展的 oracle 策略选择框架：在每轮主动学习中，并行运行多种查询策略在随机子池上生成候选 batch，通过冻结 backbone 仅重训最后一层快速评估每个候选 batch 的性能增益，选出最优 batch，从而量化现有 AL 策略与理论最优之间的差距。

## 研究背景与动机

### 1. 领域现状

深度主动学习（Deep Active Learning）旨在通过智能选择最有信息量的样本进行标注，以最小的标注成本达到最优模型性能。近年来涌现了大量 AL 策略：基于不确定性的（Margin、Entropy）、基于多样性的（CoreSets、TypiClust）、混合策略（BADGE、AlfaMix）等。

### 2. 痛点

- **没有通用最优策略**：不同数据集、不同标注预算、不同模型架构下，最优策略各异。实践者面临"选哪个策略"的困境
- **现有 oracle 方法局限**：SAS（Sequential Active Selection）和 CDO（Combined Dataset Oracle）等 oracle 在样本级别贪心选择，计算代价极高且无法保证 batch 级最优
- **缺乏量化基准**：无法准确衡量现有策略距离理论最优还有多大差距，难以判断领域的改进空间

### 3. 核心矛盾

主动学习研究不断提出新策略声称 SOTA，但缺少一个强可靠的 oracle 上界来衡量这些策略到底有多好——大家在比的可能只是"矮子里拔高个"。

### 4. 要解决什么

设计一个计算可行、可扩展到大规模数据集的 oracle 方法，建立 AL 策略性能的上界基准线。

### 5. 切入角度

从 batch 级别（而非样本级别）进行策略竞赛：让多种策略各自提出候选 batch，快速评估后选最优，将 oracle 问题转化为"best-of-N"选择问题。

## 方法详解

### 整体框架

BoSS 想回答一个被 AL 研究长期回避的问题：现有这些查询策略离"理论最优"到底还差多远？要回答它就得有一个可靠的 oracle 上界，而过去的 oracle 要么逐样本贪心、算不动大数据，要么忽略 batch 内部的多样性。BoSS 的思路是把 oracle 问题改写成一个 **batch 级的 best-of-N 竞赛**：在每一轮主动学习里，让 $M$ 种不同的查询策略各自提交一个候选 batch，再用一个廉价但忠实的代理快速给每个候选打分，最后挑出真正能让模型涨得最多的那个 batch 去标注。

一轮的数据流是这样转的：从未标注池随机切出一个小子池 → $M$ 种策略各在这个子池上挑出自己的候选 batch → 把每个候选并入当前已标注集、冻结 backbone 只重训一层线性头、在验证集上量出它带来的准确率 → 选增益最大的候选作为本轮查询结果。下一轮在更新后的标注集上重复，直到耗尽标注预算。这套流程既给出了一条可量化的 oracle 上界曲线，又因为子池采样和冻结评估两个设计而能一路扩展到 ImageNet 规模。

### 关键设计

**1. 多策略集成：用一池互补的策略覆盖"信息量"的不同侧面。**

单一查询策略只能抓住数据的一个侧面——不确定性策略盯着模型最拿不准的样本，多样性策略追求覆盖面，典型性策略偏好代表性样本。哪一面在当前数据集、当前预算下更重要，事先并不知道。BoSS 索性把它们凑成一个策略池：Random、Margin（不确定性）、CoreSets（多样性）、BADGE（梯度+多样性）、FastBAIT（Fisher 信息）、TypiClust（典型性+聚类）、AlfaMix（插值扰动）、DropQuery（MC Dropout）共 8 种。每种策略提交一个风格迥异的候选 batch，竞赛的"参赛者"越多样，池子里出现接近最优 batch 的概率就越高——后面的实验也验证了被选中的策略确实随数据集而变，没有哪一种全局占优。

**2. 随机子池采样：把 oracle 的计算量和数据集大小解耦。**

逐样本 oracle 算不动大数据，根子在于它要在整个未标注池 $\mathcal{U}$ 上反复搜索，复杂度随 $|\mathcal{U}|$ 线性膨胀。BoSS 不让任何策略碰完整的 $\mathcal{U}$，而是每轮随机采一个大小为 $k_{\max}$ 的子池 $\mathcal{U}' \subseteq \mathcal{U}$，所有策略都只在 $\mathcal{U}'$ 上挑 batch。这一步把搜索成本从 $O(|\mathcal{U}|)$ 压到 $O(k_{\max})$，于是总计算量与数据集规模脱钩，ImageNet 级的池子也能跑。子池随机还有个副作用：每轮看到的候选范围都在变，进一步拉开了不同策略提交的 batch 之间的差异。子池要多大才不损性能？实验里取 $k_{\max} = 10\,b$（$b$ 是 batch 大小）就够——比这更大收益已饱和。

**3. 冻结 backbone 的线性探测代理：让"评估一个候选"从几十轮训练缩到几秒。**

竞赛要选出最优 batch，就得先知道每个候选能让模型涨多少——可如果对 $M$ 个候选各自做一次完整的几十轮重训，计算量直接爆炸。BoSS 用一个忠实但便宜的代理替代它：backbone（自监督预训练的 DINOv2-ViT-S/14）始终冻结，评估某个候选 $\mathcal{B}_m$ 时只把它并入已标注集 $\mathcal{L}$，在 $\mathcal{L} \cup \mathcal{B}_m$ 的特征上重训最后那层线性分类头，再到验证集上读准确率。线性头几秒就能收敛，$M$ 个候选还能并行评。代理之所以站得住，是因为它和全模型重训给出的候选排序几乎一致（实验里 Spearman $\rho > 0.95$），却快出约 50 倍——排序对了，oracle 选出来的 batch 就对了。

**4. batch 级的最优选择目标：把 oracle 定义在 batch 而非单样本上。**

把上面三件事拼起来，BoSS 每轮要解的就是一个 batch 级的取最大问题（Eq.3）：

$$\mathcal{B}^* = \arg\max_{\mathcal{B}_m,\, m \in \{1,\dots,M\}} \text{Acc}(\theta_{\mathcal{L} \cup \mathcal{B}_m};\, \mathcal{V})$$

其中 $\theta_{\mathcal{L} \cup \mathcal{B}_m}$ 是在 $\mathcal{L} \cup \mathcal{B}_m$ 上训练后的（线性头）参数，$\mathcal{V}$ 是验证集。它和 SAS、CDO 这类逐样本贪心 oracle 的根本区别就在这个"以整批为单位"上：逐样本贪心选出的样本拼成一批时，彼此可能高度冗余，而 batch 级目标直接对整批的实际增益负责，天然把 batch 内多样性纳入了考量。

举个一轮里发生了什么的具体画面：设 $b=100$、子池 $k_{\max}=1000$，8 种策略各从这 1000 个样本里挑出自己心仪的 100 个，得到 8 个候选 batch；冻结 backbone、并入已标注集后分别训线性头，量到 8 个验证准确率（比如 Margin 的候选 +1.2%、BADGE 的候选 +1.8%、TypiClust 的候选 +0.9%…）；BoSS 选出增益最大的 BADGE 候选去真正标注，这一轮记录的 oracle 点就由它定。下一轮换一批随机子池，被选中的策略可能就变成了别人。

### 训练策略

- **Backbone**：DINOv2-ViT-S/14（全程冻结），特征维度 384
- **评估模型**：单层线性分类头，SGD 优化，超参固定，对所有候选一致以保证打分可比
- **验证集**：从已有标注数据中按比例划出，或使用额外验证集
- **可扩展性**：子池大小 $k_{\max}$ 与 batch size 线性相关，oracle 总计算量与数据集规模解耦

## 实验关键数据

### 主实验

**表1：BoSS vs 现有 Oracle 在多个数据集上的对比（AUC-Accuracy↑）**

| Oracle 方法 | CIFAR-10 | CIFAR-100 | TinyImageNet | ImageNet-50 |
|------------|----------|-----------|--------------|-------------|
| Random | 89.2 | 61.4 | 47.8 | 72.3 |
| Best Single Strategy | 91.5 | 65.8 | 52.1 | 76.9 |
| SAS | 92.1 | 66.3 | - | - |
| CDO | 91.8 | 65.9 | - | - |
| **BoSS (Ours)** | **93.7** | **69.2** | **56.4** | **80.1** |

**关键发现**：BoSS 在所有数据集上超越 SAS 和 CDO，且可扩展到它们无法处理的 TinyImageNet 和 ImageNet 规模。

**表2：各 AL 策略 vs BoSS Oracle 的差距（CIFAR-100, AUC-Accuracy）**

| 策略 | AUC-Acc | Gap to BoSS |
|------|---------|-------------|
| Random | 61.4 | -7.8 |
| Margin | 64.1 | -5.1 |
| CoreSets | 63.8 | -5.4 |
| BADGE | 65.2 | -4.0 |
| TypiClust | 64.5 | -4.7 |
| AlfaMix | 65.8 | -3.4 |
| **BoSS** | **69.2** | **0.0** |

### 消融实验

**表3：BoSS 组件消融**

| 配置 | CIFAR-100 AUC↑ | 计算时间 |
|------|----------------|---------|
| BoSS (Full) | 69.2 | 1× |
| 无子池采样（用完整池） | 69.5 | 8× |
| 仅 3 种策略 | 67.8 | 0.4× |
| 全模型重训评估 | 69.8 | 50× |
| 冻结 backbone + 线性头 | 69.2 | 1× |

**子池大小 $k_{\max}$ 消融**：

| $k_{\max} / b$ | AUC↑ | 
|----------------|------|
| 3× | 67.1 |
| 5× | 68.3 |
| 10× | 69.2 |
| 20× | 69.4 |

### 关键发现

1. **SOTA 策略仍远不及 Oracle**：最好的单一策略（AlfaMix）与 BoSS 之间仍有 3-4 个百分点的差距，说明 AL 领域仍有显著改进空间
2. **无单一策略全局占优**：不同数据集上被 BoSS 选中的策略分布不同——CIFAR-10 上 Margin 最常被选中，CIFAR-100 上 BADGE 更优，ImageNet 上 TypiClust 突出
3. **差距随数据集复杂度增大**：类别数越多、数据量越大，现有策略与 oracle 的差距越明显（CIFAR-10 约 2%，ImageNet-50 约 4%+）
4. **冻结评估近似全模型评估**：冻结 backbone + 线性头评估与全模型评估的排序高度一致（Spearman $\rho > 0.95$），但计算快 50 倍
5. **子池采样几乎无损**：$k_{\max} = 10b$ 时性能仅比使用完整池低 0.3%，但计算量降低 8 倍

## 亮点与洞察

- **元分析视角**：不是提出又一个 AL 策略，而是回答"现有策略到底有多好"这一元问题，对整个领域的研究方向有指导意义
- **可扩展设计**：子池采样 + 冻结评估两个设计使 BoSS 可以在 ImageNet 级别数据上运行，突破了之前 oracle 方法只能在小数据集上跑的限制
- **发现深刻**：证实了"没有免费午餐"——不同场景下最优策略确实不同，且即使最好的策略也距离 oracle 甚远
- **实践指导**：BoSS 本身也可作为一种实用的 AL 策略——在标注预算充裕但不确定用哪个策略时，直接用 BoSS 自动选择

## 局限与展望

- 依赖固定的策略池，新策略需要手动加入；可考虑自动发现/生成策略的元学习方案
- 冻结 backbone 评估在微调场景下可能不够准确（仅在线性探测设定下验证了强相关性）
- 验证集的构建假设了有一个始终可用的 held-out 集，早期标注极少时验证集质量可能不够
- 子池随机性引入方差，单次运行结果有波动；论文通过多次运行取均值缓解，但增加计算
- 未探索跨轮次的策略选择模式——是否可以学一个 meta-policy 预测每轮应该用哪个策略

## 相关工作与启发

- **SAS [Gilhuber et al.]**：逐样本贪心选择的 oracle，计算复杂度 $O(|\mathcal{U}| \times b)$ 使其无法扩展到大数据集，BoSS 通过 batch 级选择解决了这个问题
- **CDO [Zhan et al.]**：将多轮最优样本合并评估，但忽略了 batch 内多样性问题
- **BADGE [Ash et al.]**：梯度嵌入 + k-means++ 的经典策略，在多个 BoSS 实验中表现突出
- **TypiClust [Hacohen et al.]**：基于 SSL 特征典型性的策略，在大规模数据上尤其有效
- **DINOv2 [Oquab et al.]**：作为冻结 backbone 提供强特征表示，使线性评估成为可靠的快速代理
- **启发**：BoSS 的"多策略竞赛 + 快速评估"范式可推广到其他需要策略选择的场景——如数据增强策略选择、超参搜索策略选择等

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角独特的元分析工作，将策略集成+proxy retraining+batch selection有机结合
- 实验充分度: ⭐⭐⭐⭐⭐ 10个数据集、2种backbone、完整消融、时间对比、策略频率分析
- 写作质量: ⭐⭐⭐⭐⭐ 形式化清晰，逻辑链完整，图表丰富
- 价值: ⭐⭐⭐⭐ 为AL社区提供实用Oracle基准工具，揭示大规模多类场景的改进空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[CVPR 2026\] Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](hier-cos_making_deep_features_hierarchy-aware_via_composition_of_orthogonal_subs.md)
- [\[CVPR 2026\] TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)

</div>

<!-- RELATED:END -->
