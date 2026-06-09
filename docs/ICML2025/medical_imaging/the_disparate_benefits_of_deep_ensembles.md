---
title: >-
  [论文解读] The Disparate Benefits of Deep Ensembles
description: >-
  [ICML 2025][医学图像][Deep Ensembles] 系统揭示 Deep Ensembles 的"差异化收益效应"：集成带来的性能提升在不同受保护群体间分配不均，往往有利于已优势群体，导致公平性指标恶化；并发现集成成员间的 per-group 预测多样性差异是根本原因…
tags:
  - "ICML 2025"
  - "医学图像"
  - "Deep Ensembles"
  - "算法公平性"
  - "差异化收益"
  - "预测多样性"
  - "后处理"
---

# The Disparate Benefits of Deep Ensembles

**会议**: ICML 2025  
**arXiv**: [2410.13831](https://arxiv.org/abs/2410.13831)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 深度集成, 算法公平性, 群体公平, 预测多样性, 后处理缓解

## 一句话总结
本文系统研究了深度集成（Deep Ensembles）对算法公平性的影响，发现集成会不均匀地提升不同群体的性能（disparate benefits effect），导致公平性下降，并证明Hardt后处理方法能有效缓解该问题同时保留集成的性能增益。

## 研究背景与动机

**领域现状**：深度集成是提升DNN性能最简单有效的方法之一，通过训练多个独立模型并平均输出分布来近似贝叶斯后验预测分布。在医疗、金融等高风险场景中被广泛使用。

**现有痛点**：虽然单个DNN的群体公平性问题已被深入研究，但集成多个网络对公平性的影响却几乎没有被探索。现有工作（如Ko et al., 2023）只关注子群体准确率变化，未使用标准公平性指标，且得出了"集成仅有正面影响"的片面结论。

**核心矛盾**：集成在提升整体性能的同时，其收益可能在不同保护属性定义的群体间分配不均——优势群体可能获得更多提升，导致公平性差距反而加大。

**本文目标**：(1) 系统量化深度集成对群体公平性的影响；(2) 分析不公平效应的根本原因；(3) 探索有效的缓解策略。

**切入角度**：作者从人脸分析和医学影像两个现实高风险场景出发，使用三种标准公平性指标（统计平价SPD、机会均等EOD、平均赔率差AOD）衡量深度集成的公平性影响。

**核心 idea**：深度集成的性能提升在不同群体间存在"差异化受益"效应——这种不均匀的改善可以用各群体间集成成员的"预测多样性"差异来解释，而利用集成更好的校准特性进行Hardt后处理能有效缓解该问题。

## 方法详解

### 整体框架
本文是一项系统性的实证研究，涵盖15个任务（跨3个数据集×多个目标/保护属性组合），使用5种DNN架构（ResNet18/34/50, RegNet-Y, EfficientNetV2-S）各训练10个成员模型，共训练1000个模型。主要分三部分：现象分析→原因追溯→缓解策略。

### 关键设计

1. **Disparate Benefits效应的系统验证**:

    - 功能：量化深度集成对群体公平性的影响
    - 核心思路：对每个任务（目标变量×保护属性组合），比较10成员集成与单个成员平均值在性能和公平性指标上的差异，使用t检验（5次运行）判断显著性。结果显示集成在显著提升准确率（如FairFace上Accuracy +2.2%）的同时，显著恶化多个公平性指标（如SPD增加2.2%）
    - 设计动机：需要严谨统计证据证明"集成不总是有益的"这一反直觉结论

2. **预测多样性差异作为原因解释**:

    - 功能：追溯disparate benefits效应的根本原因
    - 核心思路：计算各群体内集成成员的预测多样性（disagreement rate），发现劣势群体上成员间分歧更大，因此集成带来的性能提升更多来自于对优势群体的"一致性投票"增强。具体地，当集成成员在某群体上的预测多样性更高时，平均预测的改善更大
    - 设计动机：仅观察到现象不够，需要机制层面的解释才能指导缓解策略

3. **Hardt后处理缓解策略**:

    - 功能：在保留集成性能增益的同时恢复公平性
    - 核心思路：深度集成由于更好的概率校准，使得其预测对阈值更加敏感。Hardt后处理通过为不同群体选择不同的决策阈值来满足公平性约束。实验表明该方法能将公平性违规降至接近0，同时几乎不损失集成带来的性能增益
    - 设计动机：集成改善了预测分布的校准性，使得基于阈值调整的后处理方法天然适用

### 损失函数 / 训练策略
模型使用标准交叉熵损失训练。FairFace上训练100 epochs，CheXpert上训练30 epochs，使用SGD+momentum（0.9），学习率1e-2配合线性+余弦退火调度。

## 实验关键数据

### 主实验

| 数据集/任务 | Δ Accuracy/AUROC ↑ | Δ SPD ↓ | Δ EOD ↓ | Δ AOD ↓ |
|---|---|---|---|---|
| FairFace: age/gender | +0.022 | -0.022 (恶化) | -0.017 (恶化) | -0.017 (恶化) |
| FairFace: age/race | +0.022 | -0.009 (恶化) | -0.012 (恶化) | -0.007 (恶化) |
| UTKFace: age/gender | +0.015 | -0.017 (恶化) | -0.015 (恶化) | -0.012 (恶化) |
| UTKFace: race/age | +0.021 | -0.013 (恶化) | -0.007 (恶化) | ~0 |
| CheXpert: age | +0.005 | -0.001 (恶化) | -0.008 (恶化) | -0.003 (恶化) |

### 消融实验

| 缓解策略 | 公平性恢复效果 | 性能保留 |
|---|---|---|
| Hardt后处理 | 公平性违规降至~0 | 几乎完全保留集成性能增益 |
| 简单阈值调整 | 部分恢复 | 保留大部分性能 |
| 无缓解（纯集成） | 公平性恶化 | 性能最高 |

### 关键发现
- 在15个任务中的绝大多数，深度集成显著提升性能但同时显著恶化至少一个公平性指标
- 劣势群体上的预测多样性更高，这解释了为什么集成对优势群体"更有利"
- 该效应在FairFace、UTKFace和CheXpert三个数据集上均稳定存在，跨5种架构也一致
- 集成成员数越多，性能提升越大但公平性恶化也越严重（几乎线性关系）

## 亮点与洞察
- **Disparate Benefits概念的提出**：将集成的"不均匀收益"形式化为一个公平性问题，这是之前完全被忽略的视角。巧妙之处在于"提升性能"这个看似无害的操作也可能加剧不公平
- **预测多样性作为解释机制**：将集成理论（多样性-准确率权衡）与公平性分析桥接，指出不同群体数据分布不同导致模型行为多样性不同，是可迁移的分析框架
- **后处理缓解的自然适配**：发现集成改善校准这一已知特性恰好使后处理方法更有效，形成了"问题与解决方案同源"的优雅闭环

## 局限与展望
- 仅验证了二分类场景，多类别分类的disparate benefits效应有待研究
- 公平性指标限于群体公平性，个体公平性（individual fairness）未涉及
- 缓解策略依赖于测试时可获取保护属性标签，实际部署中这可能不可用
- 未探索训练阶段的公平性约束方法是否能从根本上消除disparate benefits

## 相关工作与启发
- **vs Ko et al. (2023)**: 他们分析集成对子群体性能的影响但用准确率差代替标准公平性指标，且得出"集成仅有正面影响"的结论，本文用更严格的评估证明其结论有误
- **vs Hardt post-processing**: 经典方法在集成场景下的有效性被重新验证，暗示后处理方法在ensemble场景可能比in-processing方法更适合
- 这项工作对"集成总是好的"的共识提出挑战，适用于任何需要在高风险场景部署集成模型的研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究深度集成对公平性的影响，提出disparate benefits概念
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集×5架构×15任务×1000模型，统计检验严谨
- 写作质量: ⭐⭐⭐⭐ 结构清晰，从现象到原因到缓解的逻辑链完整
- 价值: ⭐⭐⭐⭐ 对ML公平性社区有重要警示意义，实操指导明确
# The Disparate Benefits of Deep Ensembles

**会议**: ICML 2025  
**arXiv**: [2410.13831](https://arxiv.org/abs/2410.13831)  
**代码**: 无  
**领域**: AI Safety / 公平性  
**关键词**: 深度集成, 算法公平性, 差异化收益, 群体公平, 后处理缓解

## 一句话总结
本文揭示了 Deep Ensembles 在提升预测性能的同时会不均匀地惠及不同保护群体（disparate benefits effect），并通过实验分析了其成因（成员预测多样性的群体间差异），最后证明 Hardt 后处理方法能有效缓解该不公平性。

## 研究背景与动机

**领域现状**：Deep Ensembles 是提升 DNN 性能和不确定性估计的主流方法，已广泛应用于医疗、金融、法律等高风险场景。算法公平性领域则关注模型在不同保护属性群体（如性别、年龄、种族）间的表现差异。

**现有痛点**：尽管单个 DNN 的群体公平性已被充分研究，但 Deep Ensembles 对群体公平性的影响尚未被探索。集成多个模型是否可能加剧已有的公平性差距？

**核心矛盾**：Deep Ensembles 的性能提升在不同群体间分布不均——优势群体往往获益更多，导致集成反而放大了不公平性。这是因为集成成员在不同群体上的预测多样性存在差异。

**本文目标**：(1) 系统性地分析 Deep Ensembles 对群体公平性的影响；(2) 找出不公平性加剧的原因；(3) 探索有效的缓解策略。

**切入角度**：从集成成员"预测多样性"（predictive diversity）出发，分析为什么不同群体从集成中获益不同。

**核心 idea**：Deep Ensembles 的 disparate benefits effect 源于群体间预测多样性差异，而 Hardt 后处理可以利用集成模型更好的校准性来高效缓解此问题。

## 方法详解

### 整体框架
本文是一项实证研究而非提出新方法。整体流程为：(1) 在面部分析和医学影像数据集上训练大量 Deep Ensembles；(2) 使用三种群体公平指标评估公平性变化；(3) 分析成因；(4) 评估缓解策略。

### 关键设计

1. **实验框架设计**:

    - 功能：系统评估 Deep Ensembles 的公平性影响
    - 核心思路：在 FairFace、UTKFace（面部分析）和 CheXpert（医学影像）三个数据集上，使用 5 种 DNN 架构（ResNet18/34/50、RegNet-Y、EfficientNetV2-S），训练共 1000 个独立模型。评估三种公平指标：Statistical Parity Difference (SPD)、Equal Opportunity Difference (EOD)、Average Odds Difference (AOD)
    - 设计动机：覆盖多种数据集、架构和公平指标组合，确保结论的普适性

2. **Disparate Benefits 成因分析**:

    - 功能：解释为什么集成对不同群体的收益不同
    - 核心思路：使用 predictive diversity 指标量化集成成员间的预测差异。发现在优势群体上成员的 diversity 更高，使得集成的平均化操作对该群体的"纠错"效果更强，从而获得更大的性能提升
    - 设计动机：从集成学习理论出发，diversity 是集成性能提升的关键驱动力，群体间 diversity 差异自然导致差异化收益

3. **Hardt 后处理缓解策略**:

    - 功能：在保持集成性能优势的同时恢复公平性
    - 核心思路：Deep Ensembles 具有更好的概率校准性，因此对预测阈值更敏感。Hardt 后处理通过为不同保护群体设置不同的决策阈值来实现 equalized odds。由于集成的校准性更好，后处理的阈值调整效果更精确
    - 设计动机：后处理方法不需要重新训练模型，可以在已有集成模型上直接应用

### 损失函数 / 训练策略
使用标准交叉熵损失训练各 DNN 成员。SGD 优化器（momentum=0.9），FairFace 训练 100 epochs，CheXpert 训练 30 epochs。线性+余弦退火学习率调度。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | Δ Performance | Δ SPD | Δ EOD | Δ AOD |
|------------|------|---------------|-------|-------|-------|
| FairFace age/gender | Acc/公平 | +0.022 | -0.022（更不公平） | -0.017 | -0.017 |
| FairFace age/race | Acc/公平 | +0.022 | -0.009 | -0.012 | -0.007 |
| UTKFace age/gender | Acc/公平 | +0.015 | -0.017 | -0.015 | -0.012 |
| CheXpert/age | AUROC/公平 | +0.005 | -0.001 | -0.008 | -0.003 |

### 消融实验

| 配置 | 公平性变化 | 说明 |
|------|-----------|------|
| Deep Ensemble (10成员) | SPD/EOD/AOD 显著恶化 | 在 15 个任务中多数表现出 disparate benefits |
| + Hardt 后处理 | SPD/EOD/AOD 恢复到接近公平 | 利用更好的校准性调整阈值 |
| 单模型 + 后处理 | 效果较差 | 校准性不足限制了后处理效果 |

### 关键发现
- 在 15 个评估任务设置中，Deep Ensembles 几乎都提升了性能但同时恶化了至少一个公平指标
- 预测多样性（predictive diversity）的群体间差异是 disparate benefits 的关键解释因素
- Hardt 后处理特别适合 Deep Ensembles，因为集成提供了更好校准的预测分布，使得阈值调整更加精准

## 亮点与洞察
- **核心发现具有普适性**：disparate benefits effect 不仅出现在面部分析，也出现在医学影像中，说明这是 Deep Ensembles 的固有特性而非数据偶然。这提醒研究者在使用集成方法时必须同时关注公平性
- **因果链条清晰**：从 diversity 差异 → 性能提升不均 → 公平性恶化的逻辑链路完整，为后续研究提供了有力的分析工具
- **后处理与集成的天然配合**：集成改善校准性 → 后处理更有效，这个协同效应可迁移到其他需要公平性约束的集成方法

## 局限与展望
- 仅考虑了二元分类和二元保护属性，未推广到多类别、多属性、连续属性场景
- 仅探索了 post-processing 缓解策略，in-processing（训练时加公平约束）和 pre-processing 方法未被评估
- 实验局限于视觉任务（面部分析和胸片），未验证在 NLP 或表格数据等领域的适用性
- 可以进一步研究如何设计 fairness-aware 的集成策略，例如对成员加权以平衡群体间 diversity

## 相关工作与启发
- **vs Ko et al. (2023)**: 同样研究 Deep Ensembles 的子群体性能，但该工作以 worst/best target 子集定义群体，且结论是集成仅有正面影响。本文使用保护属性定义群体，揭示了负面公平性影响
- **vs Hardt et al. (2016)**: 经典的阈值后处理方法，本文发现其与 Deep Ensembles 的校准性特别配合
- 本文的发现对所有在高风险场景中使用 Deep Ensembles 的工作都有警示意义

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 Deep Ensembles 的公平性影响
- 实验充分度: ⭐⭐⭐⭐⭐ 1000个模型、3数据集、5架构、3公平指标，非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验设计严谨
- 价值: ⭐⭐⭐⭐ 对 ML 公平性和集成学习社区都有重要的警示意义
---
title: >-
  [论文解读] The Disparate Benefits of Deep Ensembles
description: >-
  [ICML 2025][医学图像][Deep Ensembles] 系统揭示 Deep Ensembles 的"差异化收益效应"：集成带来的性能提升在不同受保护群体间分配不均，往往有利于已优势群体，导致公平性指标恶化；并发现集成成员间的 per-group 预测多样性差异是根本原因，Hardt 后处理可有效缓解。
tags:
  - ICML 2025
  - 医学图像
  - Deep Ensembles
  - 算法公平性
  - 差异化收益
  - 预测多样性
  - 后处理
---

# The Disparate Benefits of Deep Ensembles

**会议**: ICML 2025  
**arXiv**: [2410.13831](https://arxiv.org/abs/2410.13831)  
**代码**: 无  
**领域**: 算法公平性 / 深度学习  
**关键词**: Deep Ensembles、算法公平性、差异化收益、预测多样性、后处理

## 一句话总结

系统揭示 Deep Ensembles 的"差异化收益效应"：集成带来的性能提升在不同受保护群体间分配不均，往往有利于已优势群体，导致公平性指标恶化；并发现集成成员间的 per-group 预测多样性差异是根本原因，Hardt 后处理可有效缓解。

## 研究背景与动机

- **Deep Ensembles 广泛应用**：通过简单集成多个DNN提升预测性能和不确定性估计，被大量部署在医疗、金融等高风险场景
- **算法公平性关注**：当模型在受保护群体（如性别、种族、年龄）间表现不均时，可能加剧社会不公
- **研究空白**：
    - 单个DNN的公平性问题已被广泛研究
    - 浅层模型集成的公平性有部分探索
    - **但 Deep Ensembles 对群体公平性的影响几乎未被研究过**
- **最相关工作 Ko et al. (2023)**：研究了子群性能变化但未使用标准公平性指标，且结论偏积极（认为集成仅有正面影响）

**核心发现预告**：集成的性能提升并非"水涨船高"式普惠所有群体，而是不成比例地偏向某些群体。

## 方法详解

### 整体框架

**实验驱动的研究**，非提出新模型，而是系统分析现象、原因和缓解策略：

1. **现象发现**：量化 Deep Ensemble 在多个公平性指标上的影响
2. **原因分析**：通过预测多样性（predictive diversity）解释差异化收益
3. **缓解策略**：验证 Hardt 后处理的有效性

### 关键设计

**1. 公平性指标体系**

三种标准群体公平性指标：
- **统计平权差异（SPD）**：$\text{PR}_{A=1} - \text{PR}_{A=0}$（正预测率差异）
- **机会均等差异（EOD）**：$\text{TPR}_{A=1} - \text{TPR}_{A=0}$（真正例率差异）
- **平均赔率差异（AOD）**：$\frac{1}{2}|\text{TPR}_{A=1} - \text{TPR}_{A=0}| + \frac{1}{2}|\text{FPR}_{A=1} - \text{FPR}_{A=0}|$

其中 $A=1$ 表示优势群体，所有指标取值 $\in [0, 1]$，0为最公平。

**2. 预测多样性分析**

利用 Jeffares et al. (2023) 的预测多样性定义：

$$\overline{\text{DIV}} = \frac{1}{K}\sum_{k=1}^K \left[\log\left(\frac{1}{N}\sum_{n=1}^N p(y_k|\mathbf{x}_k, \mathbf{w}_n)\right) - \frac{1}{N}\sum_{n=1}^N \log p(y_k|\mathbf{x}_k, \mathbf{w}_n)\right]$$

- 等式右侧 = 集成对数似然 - 平均成员对数似然
- 直觉：若集成成员对某群体的预测更"分散"（多样性更高），该群体从集成中获益更多
- **核心发现**：群体间预测多样性差异大 → 差异化收益效应显著；差异小 → 效应不显著

**3. Hardt 后处理**

- Deep Ensembles 因更好的校准性（calibration），其预测分布对**决策阈值更敏感**
- Hardt后处理通过为不同群体设置不同阈值来满足公平性约束
- 集成的良好校准使得阈值调整更有效

### 损失函数 / 训练策略

- 5种架构：ResNet18/34/50、RegNet-Y 800MF、EfficientNetV2-S
- 交叉熵损失，SGD+momentum=0.9
- FairFace训练100 epochs，CheXpert训练30 epochs
- 每种配置独立训练10个模型，5个随机种子，共1000个模型

## 实验关键数据

### 主实验

**数据集**：FairFace（人脸分析）、UTKFace（分布偏移测试）、CheXpert（医学影像）

**FairFace + ResNet50**（核心结果表，Δ = 集成 - 平均成员）：

| 目标/群体 | Δ Accuracy↑ | Δ SPD↓ | Δ EOD↓ | Δ AOD↓ |
|-----------|------------|--------|--------|--------|
| age/gender | +0.022 | -0.022 | -0.017 | -0.017 |
| age/race | +0.022 | -0.009 | -0.012 | -0.007 |
| gender/age | +0.014 | -0.001 | -0.007 | -0.004 |
| race/age | +0.015 | -0.004 | +0.005 | -0.001 |

注：Δ SPD/EOD/AOD为负表示公平性**恶化**

- 15个任务中的大多数出现统计显著的差异化收益效应
- 效应在age作为target时最强（SPD恶化达0.022）
- 效应在分布偏移下（UTKFace）同样存在且幅度相似

**CheXpert医学影像**：
- Protected group = age时，SPD/EOD/AOD三个指标均显著恶化
- 效应幅度较小（AUROC提升仅0.005），但公平性指标仍受影响

### 消融实验

- **模型大小影响**：较大模型（ResNet50 > ResNet34 > ResNet18）的差异化收益效应更强
- **架构一致性**：5种架构的结果一致
- **异构集成**：混合不同架构的集成同样出现差异化收益
- **集成成员数量**：添加第一个成员（1→2）时公平性恶化最剧烈，之后趋于平缓

### 关键发现

**预测多样性解释**（核心因果链）：
- 当存在显著差异化收益时，优势群体的 $\overline{\text{DIV}}$ 系统性高于劣势群体
- 当不存在差异化收益时，两群体的 $\overline{\text{DIV}}$ 差异很小
- 合成实验进一步验证：人工制造的 per-group 多样性差异导致可控的公平性变化

**Hardt后处理效果**：
- 单模型上Hardt效果有限
- Deep Ensemble上Hardt效果显著——因为集成的预测分布更好校准，阈值更可靠
- 后处理后可以同时保留集成的性能增益和满足公平性约束

## 亮点与洞察

1. **首次定义"差异化收益效应"**：揭示了一个被忽视的现象——广泛使用的Deep Ensembles可能系统性地加剧群体不公平
2. **因果解释清晰**：通过预测多样性的 per-group 差异，从理论层面解释了效应产生的机制
3. **实践影响重大**：在医疗影像、人脸识别等高风险领域，性能提升伴随公平性恶化的隐患必须被意识到
4. **缓解策略简单有效**：不需要修改训练过程，仅通过Hardt后处理即可兼顾性能和公平性
5. **实验规模大**：1000个模型、15个任务、5种架构的全面评估

## 局限与展望

- 仅考虑二元保护属性和二元分类任务，多类别场景（如多种族分组）未涉及
- 公平性指标有其固有局限——满足数学定义不等于真实公平
- 未分析差异化收益效应与训练数据不平衡的具体关系（虽然隐含相关）
- 后处理方法需要保护属性在推断时可用，某些场景下不现实
- 未探索in-processing方法（如对抗去偏、公平约束训练）能否在集成中更有效
- 缺乏对回归任务和连续公平性指标的分析

## 相关工作与启发

- **Lakshminarayanan et al. (2017)**：Deep Ensembles原始论文
- **Hardt et al. (2016)**：Equalized Odds后处理方法
- **Jeffares et al. (2023)**：预测多样性的理论分析（Theorem 4.3）
- **Ko et al. (2023)**：最相关前期工作，但结论不同（本文发现负面影响）
- **Abe et al. (2022, 2024)**：Deep Ensembles性能提升的机制分析

**启发**：性能提升≠所有群体均匀受益。在部署集成模型到高风险场景前，必须进行per-group的公平性审计。预测多样性可作为预警指标：若 per-group 多样性差异大，集成后公平性可能恶化。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究Deep Ensembles对群体公平性的影响，发现重要新现象
- 实验充分度: ⭐⭐⭐⭐⭐ — 1000个模型、3个数据集、5种架构、3个公平性指标，极其全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，实验详实，但理论分析深度可进一步加强
- 价值: ⭐⭐⭐⭐⭐ — 直接影响Deep Ensembles在高风险场景的部署决策

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Efficient Noise Calculation in Deep Learning-based MRI Reconstructions](efficient_noise_calculation_in_deep_learning-based_mri_reconstructions.md)
- [\[ICCV 2025\] Semi-supervised Deep Transfer for Regression without Domain Alignment](../../ICCV2025/medical_imaging/semi-supervised_deep_transfer_for_regression_without_domain_alignment.md)
- [\[NeurIPS 2025\] DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases](../../NeurIPS2025/medical_imaging/dca_graph-guided_deep_embedding_clustering_for_brain_atlases.md)
- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](../../CVPR2025/medical_imaging/reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2025\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](../../CVPR2025/medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)

</div>

<!-- RELATED:END -->
