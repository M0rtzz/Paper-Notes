---
title: >-
  [论文解读] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates
description: >-
  [AAAI 2026][AI安全][图神经网络] 首次揭示公平感知 GNN 中的"FPR 捷径"问题——现有方法通过大量误判负样本为正来达到公平指标，提出 FairGSE 框架通过最大化二维结构熵重新加权图边来同时改善公平性并降低假阳性率，FPR 降低 39%。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "图神经网络"
  - "公平性"
  - "假阳性率"
  - "结构熵"
  - "对比学习"
---

# FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates

**会议**: AAAI 2026  
**arXiv**: [2511.12132](https://arxiv.org/abs/2511.12132)  
**代码**: 无  
**领域**: AI安全/公平性  
**关键词**: 图神经网络, 公平性, 假阳性率, 结构熵, 对比学习

## 一句话总结
首次揭示公平感知 GNN 中的"FPR 捷径"问题——现有方法通过大量误判负样本为正来达到公平指标，提出 FairGSE 框架通过最大化二维结构熵重新加权图边来同时改善公平性并降低假阳性率，FPR 降低 39%。

## 研究背景与动机

**领域现状**：公平感知 GNN 通过对抗学习（FairGNN）、特征/结构去偏（EDITS）、不变学习（FairINV）等方法，在 $\Delta_{SP}$（统计平等）和 $\Delta_{EO}$（机会均等）等公平指标上取得满意效果。

**现有痛点**：现有方法只关注公平指标而忽略了 GNN 预测负标签的能力。作者发现 FairSIN 在 Credit 数据集上 FPR 接近 100%——几乎把所有客户分类为信用卡违约者。这种"通过把所有人都判正来实现公平"的捷径在高风险场景（信用评估、保释决定）中极其危险。

**核心矛盾**：$\Delta_{SP}$ 和 $\Delta_{EO}$ 只度量各敏感群体中正预测的概率差异——一个将所有节点预测为正的模型会得到完美的 $\Delta_{SP} = \Delta_{EO} = 0$，但 FPR = 100%。

**本文目标** 如何在提升公平性的同时避免 FPR 捷径？

**切入角度**：利用二维结构熵（2D-SE）——一个基于图结构划分的信息论度量，理论证明最大化 2D-SE 可以同时为 $\Delta_{SP}$、$\Delta_{EO}$ 和 FPR 提供上界约束。

**核心 idea**：通过最大化基于敏感属性划分的二维结构熵来重新加权图边，实现跨敏感群体的均衡消息聚合，同时避免 FPR 捷径。

## 方法详解

### 整体框架
三个组件：(1) 图结构学习器——学习可训练邻接矩阵 $\mathbf{A}^l$ 最大化 2D-SE；(2) 对比学习组件——原始图为锚点视图、学习图为学习器视图，通过对比损失保持结构一致性；(3) 结构自举机制——渐进地将学习到的公平结构融入锚点视图。

### 关键设计

1. **图结构学习器**:

    - 功能：通过优化边权重最大化二维结构熵
    - 核心思路：每条边关联可训练参数 $a_{(i,j)}$，通过 sigmoid 转换为权重 $\mathbf{A}^l_{(i,j)} = \sigma(a_{(i,j)})$，然后最小化 $\mathcal{L}_{SE} = -H^{\mathcal{P}_S}(G_l)$
    - 理论保证：Theorem 1 证明 $\Delta_{SP} \leq \sqrt{2(H^{max} - H^{\mathcal{P}_S}(G))}$；Theorem 2 证明 FPR 也有类似上界
    - 设计动机：高 2D-SE 鼓励跨敏感群体的消息聚合，减少节点表示对敏感属性的依赖

2. **对比学习组件**:

    - 功能：防止学习图偏离原始结构过多，保持分类性能
    - 核心思路：共享 GNN 编码器分别编码锚点视图和学习器视图，用 NT-Xent 损失对齐同一节点在两个视图的表示
    - 设计动机：单独优化 2D-SE 会导致训练不稳定和结构失真，对比学习约束最大化 $I(G_a, G_l)$ 保持结构信息

3. **结构自举机制**:

    - 功能：渐进更新锚点视图以消除原始图中的偏见
    - 核心思路：$\mathbf{A}^a = \tau \mathbf{A}^a + (1-\tau) \mathbf{A}^l$，$\tau = 0.9999$
    - 设计动机：固定锚点视图会继承原始图偏见并导致过拟合；慢速融入高 2D-SE 的学习结构逐步去偏

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{cont} - \lambda_2 \mathcal{L}_{SE}$。$\mathcal{L}_{task}$ 为节点分类交叉熵损失。

## 实验关键数据

### 主实验

| 方法 | Credit FPR↓ | Credit $\Delta_{SP}$↓ | Pokec_n FPR↓ | Pokec_z FPR↓ |
|------|------------|---------------------|-------------|-------------|
| Vanilla GCN | 45.96 | 14.63 | 27.06 | 27.56 |
| FairGNN | 46.48 | 8.61 | 19.10 | 27.22 |
| FairSIN | **99.76** | 0.52 | 23.92 | - |
| FairINV | 70.86 | 4.85 | 23.78 | - |
| **FairGSE** | **41.45** | 5.09 | **18.46** | - |

FairGSE 在 Credit 上 FPR 仅 41.45%（比 FairSIN 的 99.76% 低 58 个百分点），同时公平指标仍有竞争力。

### 消融实验
- 去掉 2D-SE 目标：FPR 显著上升，验证结构熵对避免 FPR 捷径的关键作用
- 去掉对比学习：分类性能下降，结构失真
- 去掉结构自举：公平性改善受限，原始偏见通过锚点视图持续影响

### 关键发现
- FPR 捷径是公平感知 GNN 的普遍现象——FairSIN、FairVGNN、FairINV、DAB-GNN 在 Credit 上 FPR 均 >70%
- 最大化 2D-SE 的理论保证成立：$\Delta_{SP}$ 和 FPR 上界随 2D-SE 增大而降低
- FairGSE 在保持 ACC/AUC 竞争力的同时将 FPR 降低了约 39%

## 亮点与洞察
- **FPR 捷径问题的发现本身就很有价值**：揭示公平指标 $\Delta_{SP}/\Delta_{EO}$ 的盲区，提醒研究者和实践者不能只看公平指标
- **信息论与公平性的桥接**：用二维结构熵统一处理公平性和 FPR，理论优雅
- **可迁移思路**：关注模型整体预测质量（而非仅公平指标）的理念可推广到其他公平性研究

## 局限与展望
- 仅考虑二元敏感属性和二元标签
- 2D-SE 的计算和梯度需要按敏感群体划分，多群体扩展的计算开销待分析
- 对比学习增加了额外的训练开销
- 未讨论在动态图或异质图上的适用性

## 相关工作与启发
- **vs FairGNN**: FairGNN 用对抗学习去偏但 FPR 仍高（46.48%），FairGSE 通过结构重加权从根源解决
- **vs FairSIN**: FairSIN 的 FPR 接近 100%，是 FPR 捷径的极端案例
- **vs DAB-GNN**: DAB-GNN 用 Wasserstein 距离约束表示差异，但同样未控制 FPR

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统揭示 FPR 捷径问题，2D-SE 理论框架优雅
- 实验充分度: ⭐⭐⭐⭐ 多个数据集+8个基线+消融，但数据集多为小规模图
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，理论证明严谨，论述逻辑流畅
- 价值: ⭐⭐⭐⭐⭐ 对公平性研究有深远影响的 insight，实际应用意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)
- [\[CVPR 2025\] Lyapunov Stable Graph Neural Flow](../../CVPR2025/ai_safety/lyapunov_stable_graph_neural_flow.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[AAAI 2026\] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity](breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)

</div>

<!-- RELATED:END -->
