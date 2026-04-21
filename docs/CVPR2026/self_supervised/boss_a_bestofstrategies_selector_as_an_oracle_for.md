---
title: >-
  [论文解读] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning
description: >-
  [CVPR 2026][自监督学习][主动学习] 提出 BoSS（Best-of-Strategies Selector），通过集成10种互补的AL选择策略生成100个候选批次，冻结预训练backbone仅重训最后线性层来高效评估每个批次的性能增益，选取最优批次作为Oracle上界参考——首个可扩展到ImageNet的深度主动学习Oracle策略，揭示当前SOTA策略在大规模多类数据集上仍有约2倍的准确率提升空间。
tags:
  - CVPR 2026
  - 自监督学习
  - 主动学习
  - Oracle策略
  - 策略集成
  - 批次选择
  - 深度神经网络
---

# BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning

**会议**: CVPR 2026  
**arXiv**: [2603.13109](https://arxiv.org/abs/2603.13109)  
**代码**: [GitHub](https://github.com/dhuseljic/dal-toolbox) (有)  
**领域**: 主动学习 / 数据选择  
**关键词**: 主动学习, Oracle策略, 策略集成, 批次选择, 深度神经网络

## 一句话总结

提出 BoSS（Best-of-Strategies Selector），通过集成10种互补的AL选择策略生成100个候选批次，冻结预训练backbone仅重训最后线性层来高效评估每个批次的性能增益，选取最优批次作为Oracle上界参考——首个可扩展到ImageNet的深度主动学习Oracle策略，揭示当前SOTA策略在大规模多类数据集上仍有约2倍的准确率提升空间。

## 研究背景与动机

**领域现状**：主动学习（AL）旨在通过迭代选择最有价值的样本进行标注，用最少标注获得最佳模型性能。尽管基础模型时代让特征提取更强大，但选择策略（哪些样本该标注）仍缺乏跨模型、跨数据集的鲁棒性。近期多个研究（Munjal 2022; Lüth 2024; Werner 2024）揭示没有任何单一策略一致最优——BADGE在某些评估中领先，Margin在另一些中领先。

**现有痛点**：
- AL策略依赖不确定性/代表性等启发式而非直接优化模型性能→在不同场景下表现不一致
- 一旦选定策略，整个AL过程中固定不变→无法适应迭代标注带来的数据分布偏移
- 现有Oracle方法（SAS：模拟退火25000步；CDO：贪婪逐样本选择，复杂度随批量二次增长）无法扩展到大规模场景→ImageNet级别完全不可行

**核心矛盾**：需要一个可量化的"最优选择"参考点来评估SOTA策略的真实差距，但构建这样的Oracle在计算上极其困难——$\binom{1000}{10} = 2.63 \times 10^{23}$ 种可能的批次组合使得穷举不可行。

**切入角度**：利用策略集成生成高质量候选批次 + 冻结backbone仅训线性层的proxy评估→将组合搜索压缩到仅100个候选批次的线性评估。

## 方法详解

### 整体框架

BoSS 将最优批次选择的优化问题形式化为：

$$\mathcal{B}^{\star}=\arg\min_{\mathcal{B}\in\{\mathcal{B}_1,\dots,\mathcal{B}_T\}} \sum_{(\mathbf{x},y)\in\mathcal{E}} \mathbb{1}[y \neq \arg\max_{c} p(c|\mathbf{x},\mathcal{L}^+)]$$

包含三个核心组件：**批次选择**（从候选中挑最优）、**性能估计**（在测试集上评估准确率）、**重训练**（高效更新模型预测）。具体流程：用10种策略各生成10个候选批次→冻结backbone+重训线性层50 epochs→选取测试集准确率最高的批次。

### 关键设计

1. **策略集成的候选批次生成**
    - 功能：将组合爆炸的搜索空间压缩为少量高质量候选方案
    - 核心思路：选取10种覆盖不确定性(Unc)、代表性(Repr)、多样性(Div)三大启发式的互补策略——Random、Margin、CoreSets、BADGE、FastBAIT、TypiClust、AlfaMix、DropQuery及监督版本 TypiClust\*、DropQuery\*。每种策略在随机采样的候选池 $\mathcal{C} \sim \text{Unif}([\mathcal{U}]^k)$（$k \leq k_{\max}$）上运行
    - 设计动机：不同策略在AL不同阶段各有优势——早期需代表性探索，后期需不确定性利用。通过变化候选池大小增加候选多样性。监督版本（用真实标签做聚类）提供额外的Oracle特权，实验证实在大规模多类场景尤其重要

2. **Selection-via-Proxy 快速重训**
    - 功能：在可接受的计算开销内评估每个候选批次的性能增益
    - 核心思路：冻结特征提取器 $h^{\phi}$，仅重训最后线性层 $g^{\theta}$，epochs从正式训练的200降到50
    - 设计动机：全模型重训 $T=100$ 次在大数据集上不可行。冻结backbone让参数更新限制在简单模型上→稳定性高且计算量小。消融实验显示50 epochs已足够区分批次质量（甚至10 epochs效果相近，5 epochs才出现明显下降）

3. **基于测试集的性能直接评估**
    - 功能：精确衡量候选批次对最终模型性能的真实贡献
    - 核心思路：作为Oracle策略，允许使用测试集 $\mathcal{E}$ + zero-one loss直接对应评估指标（准确率）
    - 设计动机：zero-one loss直接映射到准确率学习曲线。Brier score作为proper scoring rule也有效（AULC差距微小），但cross-entropy效果较差→直接对应最终指标的损失函数更优

### 损失函数 / 训练策略

- **BoSS评估阶段**：冻结backbone + 线性层SGD 50 epochs，lr=0.01, batch=64, weight decay=1e-4, cosine annealing
- **正式训练**（选定批次后）：冻结backbone + 线性层SGD 200 epochs，同上超参
- **Backbone**：DINOv2-ViT-S/14（22M, $D=384$, 自监督）和 SwinV2-B（88M, $D=1024$, 有监督ImageNet预训练）
- **AL设置**：每数据集20个cycle，batch size从10（CIFAR-10）到1000（ImageNet），所有结果取10次平均

## 实验关键数据

### 主实验

**Oracle策略对比（DINOv2-ViT-S/14，对齐运行时间）— 相对Random准确率提升(%)**

| 数据集 (batch) | BoSS | CDO (对齐) | SAS (对齐) |
|---|:---:|:---:|:---:|
| CIFAR-10 (b=10) | **~20%** | ~18% | ~5% |
| Snacks (b=20) | **~20%** | ~18% | ~8% |
| Dopanim (b=50) | **~10%** | ~8% | ~5% |
| DTD (b=50) | **~10%** | ~8% | ~3% |

*SAS默认运行时间达BoSS的100倍以上（DTD：62h vs 22min），对齐后性能大幅下降*

**BoSS vs SOTA AL策略 — ImageNet/DINOv2上BoSS约为最佳Strategy的2倍准确率提升**

| 数据集 | 类别数K | BoSS提升 | 最佳AL策略提升 | 差距倍数 |
|---|:---:|:---:|:---:|:---:|
| CIFAR-10 | 10 | ~5% | ~4% | 1.25x |
| Snacks | 20 | ~10% | ~8% | 1.25x |
| CIFAR-100 | 100 | ~7% | ~4% | 1.75x |
| ImageNet | 1000 | ~8% | ~4% | **2.0x** |

### 消融实验

| 消融维度 | 配置 | CIFAR-10 AULC | DTD AULC |
|---|---|:---:|:---:|
| 候选来源 | 策略集成 (Alg.1) | **90.70** | **71.79** |
| | 随机候选 (Eq.5) | 89.2 | 69.5 |
| 每策略批次数 | T=20/策略 | 90.83 | 71.91 |
| | T=10/策略 | **90.70** | **71.79** |
| | T=1/策略 | 89.90 | 70.45 |
| Batch size | 0.5b | **85.71** | **68.41** |
| | b (默认) | 85.62 | 67.95 |
| | 4b | 84.95 | 66.82 |
| 损失函数 | Zero-one | **90.70** | **71.79** |
| | Brier score | 90.65 | 71.75 |
| | Cross-entropy | 90.10 | 70.80 |

**策略累积增长消融（Dopanim数据集）**

| 累积策略 | AULC |
|---|:---:|
| Random | 75.24 |
| +DropQuery | 75.82 |
| +AlfaMix | 76.01 |
| +TypiClust | 76.08 |
| +全部10种 | **76.52** |

### 关键发现

- **差距随数据集复杂度增长**：简单数据集上AL策略接近Oracle，但K>100的复杂多类数据集差距显著→**大规模多类场景是AL研究最有价值的方向**
- **无单一策略全程最优**：Pick frequency分析显示早期DropQuery\*/TypiClust\*主导（代表性），后期无一致最优甚至Random被频繁选中→集成策略优于固定策略
- **策略集成远优于随机候选**：说明启发式策略虽不完美但有效约束搜索空间
- **加入更多策略只有正面效果**：即使加入弱策略也不损害BoSS性能（表现差的批次自然不会被选中）

## 亮点与洞察

- **首个可扩展到ImageNet的深度AL Oracle**：计算复杂度固定 $O(T \cdot \text{train-eval}(\theta, \mathcal{L}^+, \mathcal{E}))$，与batch size和cycle数无关
- **量化SOTA真实差距**：ImageNet上最佳AL策略仅达Oracle约50%的提升量，为社区设立明确改进目标
- **Pick frequency分析揭示AL动态特性**：冷启动阶段需代表性策略→当前无监督策略在此阶段不足；利用阶段无一致最优→需自适应机制
- **设计理念简洁**：三个简单组件的组合（集成+proxy+选择），易于复现、扩展

## 局限与展望

- Oracle策略依赖全量标签和测试集，不可直接用于实际AL→仅作为评估上界
- 仅在图像分类任务上验证，目标检测/语义分割/NLP等任务范围有限
- 多臂赌博机自适应分配各策略的候选批次数量是一个有前景的扩展方向
- 能否将Oracle的pick frequency pattern训练成meta-learner，在无标签场景下自适应切换策略？

## 相关工作与启发

- **vs CDO (Werner et al., NeurIPS 2024)**：贪婪逐样本评估，复杂度 $O(m \cdot b^2)$ 随批量二次增长，$b=50$ 时必须降 $m$ 至3-4→大batch不可行
- **vs SAS (Zhou et al., AISTATS 2021)**：模拟退火搜索全标注池，默认需30000步，对齐运行时间后性能大幅退化
- **vs 实践中的AL策略**：BADGE和BAIT整体最接近Oracle，CoreSets始终表现不佳
- **启发**：集成多策略思路 + pick frequency动态分析 → 可指导无Oracle的自适应AL策略设计

## 评分

⭐⭐⭐⭐ (4/5)

方法理念简洁有效（策略集成+proxy重训+性能选择），首次将Oracle扩展到ImageNet规模，10个图像+4个文本数据集+2种backbone的实验非常充分，但方法本身新颖性有限（策略集成+暴力搜索），且作为Oracle仅用于评估基准而非实际应用。

<!-- RELATED:START -->

## 相关论文

- [SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)
- [DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)

<!-- RELATED:END -->
