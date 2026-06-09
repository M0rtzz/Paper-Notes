---
title: >-
  [论文解读] MOS: Modeling Object-Scene Associations in Generalized Category Discovery
description: >-
  [CVPR 2025][自监督学习][广义类别发现] 挑战了GCD中"场景信息是噪声"的传统观点，发现场景被误解为噪声是因为"歧义挑战"（目标与场景的base/novel关系冲突），提出MOS框架通过双分支网络+MLP场景感知模块有效利用场景信息，在细粒度GCD上平均提升4%。
tags:
  - "CVPR 2025"
  - "自监督学习"
  - "广义类别发现"
  - "场景信息"
  - "目标-场景关联"
  - "双分支网络"
  - "细粒度分类"
---

# MOS: Modeling Object-Scene Associations in Generalized Category Discovery

**会议**: CVPR 2025  
**arXiv**: [2503.12035](https://arxiv.org/abs/2503.12035)  
**代码**: [GitHub](https://github.com/JethroPeng/MOS)  
**领域**: Others (广义类别发现)  
**关键词**: 广义类别发现, 场景信息, 目标-场景关联, 双分支网络, 细粒度分类

## 一句话总结

挑战了GCD中"场景信息是噪声"的传统观点，发现场景被误解为噪声是因为"歧义挑战"（目标与场景的base/novel关系冲突），提出MOS框架通过双分支网络+MLP场景感知模块有效利用场景信息，在细粒度GCD上平均提升4%。

## 研究背景与动机

广义类别发现（Generalized Category Discovery, GCD）是一个半监督分类任务：给定一个带标签的数据集（包含base类）和一个未标签数据集（包含base类和novel类），目标是对未标签数据进行分类。GCD面临的核心挑战在于如何利用已知类的知识来发现未知的新类。

**传统认知**：现有GCD方法普遍认为场景信息（如森林、海洋、天空等背景）是噪声，会干扰模型泛化，因此通过各种技术最小化其影响。支持这一观点的经验证据是：去除场景后GCD性能通常会提升。

**本文的挑战性发现**：作者在CUB鸟类数据集上对场景信息进行了精细分析，发现性能提升只发生在**目标与场景的base/novel关系冲突**的子集中：
- 新类鸟出现在已知场景中→容易被误分类为已知类（场景特征错误引导）
- 已知鸟出现在新场景中→容易被误认为新类（场景特征干扰判断）

而在关系一致的子集中（如新类鸟+新场景），去除场景反而会**轻微降低**性能。这说明场景信息本质上是有价值的先验，问题出在"歧义挑战"而非场景本身。

## 方法详解

### 整体框架

MOS采用双分支设计：(1) **原始图像分支**：处理包含场景+目标的完整图像；(2) **目标图像分支**：处理通过显著性分割提取的前景目标（背景填充均值像素）。两个分支共享backbone（ViT-B/16）和场景感知模块，但分别计算损失。推理时仅使用目标分支的输出。

### 关键设计

**1. 目标-场景解耦**

使用零样本显著性分割模型IS-Net将前景目标从场景中分离。生成显著性掩码M，前景像素保留、背景像素替换为图像均值μ。均值填充而非零填充是为了最小化因缺失场景带来的分布域漂移。

**2. 场景感知模块（SA Module）**

核心是一个简洁的MLP交互设计。输入目标/图像特征 v_i（i∈{o,x}）和场景特征 v_s（实际使用原始图像特征替代），先拼接再L2归一化，然后通过MLP处理：IM_i = MLP(normalize(v_i ⊕ v_s))。

关键设计细节：
- **场景特征用原始图像特征替代**：因为场景通常位于图像边缘，难以被DINO单独提取；在给定目标特征时，原始图像特征的条件熵等于场景特征的条件熵
- **教师网络输出场景特征**（detach，不传播梯度）：解决 v_s = v_x 时特征空间变化过快和 v_x 过度优化的问题
- 为保证对比公平性，从DINO header中移除一个MLP层来补偿新增的模块

**3. 双分支联合训练**

每个分支独立计算四种损失：有标签数据的监督对比学习损失 + 分类交叉熵损失；无标签数据的无监督对比学习损失 + 教师-学生交叉熵损失。总损失为两分支损失的加权和 L = λ1·L_origin + λ2·L_object，实践中 λ1=λ2=1。

### 损失函数

继承SimGCD的损失设计，每个分支使用：L_branch = (1-λ)(L_un^nce + L_un^cls) + λ(L_sup^nce + L_sup^cls)，其中λ=0.35平衡有/无标注数据。L_un^cls使用EMA教师网络生成伪标签。温度参数τ_u=0.07, τ_c=1.0, τ_t=0.07, τ_s=0.1。

## 实验关键数据

### 主实验：SSB细粒度数据集（Table 2）

| 方法 | CUB All↑ | Stanford Cars All↑ | FGVC All↑ | 平均 All↑ |
|------|:---:|:---:|:---:|:---:|
| SimGCD | 60.3 | 53.8 | 54.2 | 56.1 |
| SPTNet | 65.8 | 59.0 | 59.3 | 61.4 |
| InfoSieve | 69.4 | 55.7 | 56.3 | 60.5 |
| LeGCD | 63.8 | 57.3 | 55.0 | 58.7 |
| **MOS** | **69.6** | **64.6** | **61.1** | **65.1** |

MOS在SSB三个数据集上的平均准确率达到65.1%，超出SimGCD 9%，超出最强基线SPTNet 3.7%。在Stanford Cars上提升尤为显著（+5.6%），表明场景信息在汽车分类中同样有效。

### 消融实验（Table 4，CUB数据集）

| 配置 | All↑ | Base↑ | Novel↑ |
|------|:---:|:---:|:---:|
| 原始SimGCD | 61.50 | 65.70 | 59.40 |
| + Object分支 | 63.06 | 64.71 | 62.23 |
| + MOS双分支 | 67.86 | 70.78 | 66.40 |
| + SA Module | **69.57** | **72.31** | **68.20** |

- 引入目标图像分支已有1.5%提升，说明去场景训练有基础价值
- 双分支联合训练大幅提升4.8%（Novel类从62.2→66.4），说明模型通过对比差异学到了场景信息
- SA Module进一步贡献1.7%（Novel类从66.4→68.2）
- **共享权重对Novel类影响最大**（Fig. 5右图），表明有效提取场景特征对推断新类至关重要

### 关键发现

- 在Oxford-IIIT Pet上准确率达93.2%，超SimGCD 4.4%
- L1偏差在训练过程中持续增长（Fig. 6），说明网络确实在学习区分全局特征和目标特征
- IS-Net分割即使有失败案例也不影响整体效果（Fig. 7），框架对分割质量具有鲁棒性
- 参数λ1, λ2在1附近变化时性能稳定（±1%内），方法对超参不敏感

## 亮点与洞察

1. **挑战传统假设的勇气与证据**：通过精心设计的"四子集"实验，清晰揭示了场景信息被误解为噪声的根本原因是歧义挑战，而非场景本身无价值
2. **场景信息的"先验"价值**：在信息有限的GCD场景中，场景可以提供强有力的类别推断线索（如"水面上的鸟更可能是水禽"）
3. **极简MLP即有效**：场景感知模块只是一个MLP，相比baseline几乎无额外参数开销，却带来显著提升，证明关键在于设计而非复杂度

## 局限性

- CUB数据集的场景标注需要额外人工工作，论文专门为此标注了24个场景类别，在其他数据集上可能不易复现分析
- 显著性分割IS-Net的质量直接影响目标-场景解耦效果，虽然实验显示对分割失败有鲁棒性，但系统性分割错误可能降低效果
- 场景特征用原始图像特征替代是一个近似，在场景占图像比例很小时可能效果有限
- 仅在细粒度数据集上验证了场景信息的价值；在CIFAR-100等通用分类数据集上场景可能不那么有帮助

## 相关工作与启发

- **SimGCD**: MOS的基线，贡献了参数化分类器和对比学习框架；MOS在其上增加的模块极为轻量
- **DCCL/PromptCAL**: 分别从概念发现和提示学习角度改进GCD，但都没有利用场景信息
- **启发**: "被认为是噪声的信号可能是被误用的信号"这一思路可推广到其他任务；场景先验可与开放词汇检测结合探索

## 评分

⭐⭐⭐⭐ — 核心发现（歧义挑战导致场景被误解为噪声）非常有洞察力，方法简洁有效，在细粒度GCD上取得SOTA。论证链条清晰，从观察→分析→设计→验证形成完整闭环。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Hyperbolic Category Discovery](hyperbolic_category_discovery.md)
- [\[NeurIPS 2025\] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../../NeurIPS2025/self_supervised/consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2026\] Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](../../CVPR2026/self_supervised/learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)
- [\[CVPR 2026\] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](../../CVPR2026/self_supervised/omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)

</div>

<!-- RELATED:END -->
