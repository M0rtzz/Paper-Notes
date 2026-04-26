---
title: >-
  [论文解读] A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks
description: >-
  [ICML 2025][图学习][图神经网络] 提出 Cognac——首个有效的 GNN 纠正性遗忘方法，通过交替执行图邻域对比遗忘（CoGN）和解耦梯度上升/下降（AC⚡DC），在仅识别 5% 被操纵实体时即可恢复接近 oracle（完全干净数据训练）的性能，比从头重训高效 8×。
tags:
  - ICML 2025
  - 图学习
  - 图神经网络
  - 纠正性遗忘
  - 对抗操纵
  - 消息传播
  - 图遗忘
---

# A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks

**会议**: ICML 2025  
**arXiv**: [2412.00789](https://arxiv.org/abs/2412.00789)  
**代码**: https://github.com/corrective-unlearning-for-gnns  
**领域**: 图学习  
**关键词**: 图神经网络, 纠正性遗忘, 对抗操纵, 消息传播, 图遗忘

## 一句话总结
提出 Cognac——首个有效的 GNN 纠正性遗忘方法，通过交替执行图邻域对比遗忘（CoGN）和解耦梯度上升/下降（AC⚡DC），在仅识别 5% 被操纵实体时即可恢复接近 oracle（完全干净数据训练）的性能，比从头重训高效 8×。

## 研究背景与动机

**领域现状**：GNN 在推荐系统、药物发现等领域广泛应用，并开始扩展到大规模图基础模型。但在大规模训练集中验证每个样本的完整性成本过高。

**现有痛点**：
   - 图数据不满足 i.i.d. 假设——消息传递使对抗操纵通过邻域传播影响更广范围的节点
   - 现有图遗忘方法（GraphEraser、GUIDE 等）关注隐私合规删除，不适用于纠正性遗忘——即使知道全部被操纵集也无法有效遗忘操纵影响
   - "从头重训"不是纠正性遗忘的金标准——当未识别的操纵数据仍在训练集中时，重训会强化操纵效果

**核心矛盾**：消息传递虽强大但是双刃剑——操纵少数节点可导致大面积预测偏差；遗忘必须同时纠正被操纵节点本身和其受影响的邻居。

**本文目标**：从已训练 GNN 中移除数据操纵的不良影响，且仅需识别操纵集的小部分子集。

**切入角度**：两步交替——(1) 对比损失将受影响邻居的表示推离被操纵节点; (2) 在被操纵集上梯度上升+剩余数据上梯度下降。

**核心 idea**：利用图结构——被操纵节点的邻居是关键"感染区"，通过对比学习"消毒"邻域，再通过解耦梯度纠正幅度。

## 方法详解

### 整体框架
Cognac 交替执行两个组件：
1. **CoGN（Contrastive Unlearning on Graph Neighborhoods）**：
    - 识别删除集的受影响邻居
    - 对比损失推离邻居与删除集的表示，拉近邻居与其他正常邻居
2. **AC⚡DC（AsCent DesCent DeCoupled）**：
    - 在删除集标签上做梯度上升（"忘记"错误标签）
    - 在保留集标签上做梯度下降（保持正常性能）
    - 两个过程使用独立优化器（解耦避免干扰）

### 关键设计

1. **图邻域对比遗忘（CoGN）**:

    - 功能：纠正被操纵节点对邻域的"感染"
    - 核心思路：
        - 找到删除集 $S_f$ 的 $k$-hop 邻居 $\mathcal{N}(S_f)$
        - 对比损失：$\mathcal{L}_{\text{CoGN}} = -\log \frac{\exp(\text{sim}(h_v, h_{v^+})/\tau)}{\exp(\text{sim}(h_v, h_{v^+})/\tau) + \sum_{u \in S_f} \exp(\text{sim}(h_v, h_u)/\tau)}$
        - 正样本 $v^+$：邻域中不在删除集的正常节点；负样本：删除集中的节点
    - 设计动机：消息传递使操纵通过邻域传播→必须显式纠正邻域表示
    - 独特优势：即使删除集只是操纵集的 5%，邻域搜索也能"发现"更多未识别的受影响节点

2. **解耦梯度上升/下降（AC⚡DC）**:

    - 功能：在删除集上"遗忘"+在保留集上"维持"
    - 核心思路：
        - 梯度上升：$\theta \leftarrow \theta + \eta_1 \nabla_\theta \mathcal{L}(S_f)$（增大删除集损失→遗忘）
        - 梯度下降：$\theta \leftarrow \theta - \eta_2 \nabla_\theta \mathcal{L}(S_r)$（减小保留集损失→维持）
        - 关键：两个过程使用独立的优化器实例（独立动量/自适应学习率）
    - 设计动机：耦合优化（如 SCRUB）使遗忘和维持互相干扰→解耦后各自更稳定
    - 经典 i.i.d. 遗忘方法在图上的适配——AC⚡DC 提供基础遗忘，CoGN 提供图特定的邻域纠正

3. **代表性子集的利用**:

    - 功能：仅用 5% 已识别操纵实体来遗忘全部操纵影响
    - 核心思路：已识别子集 $S_f \subset S_m$ 共享操纵模式→CoGN 通过邻域扩展自然发现更多受影响节点→AC⚡DC 在 $S_f$ 上的遗忘信号泛化到整个操纵集
    - 设计动机：实际中不可能 100% 识别所有被操纵数据——方法必须在部分识别下有效

### 损失函数 / 训练策略
- 总优化：交替执行 CoGN 和 AC⚡DC
- CoGN 使用对比损失 + 温度参数 $\tau$
- AC⚡DC 使用标准交叉熵（上升/下降方向不同）
- 两个独立 Adam 优化器
- 早停基于验证集准确率

## 实验关键数据

### 主实验
Cora/Citeseer/PubMed + ogbn-arxiv（边操纵 + 节点标签翻转）：

| 方法 | 受影响分布准确率↑ | 非受影响准确率 | 已知比例 |
|------|---------------|-----------|---------|
| 无遗忘（操纵后模型） | 55.2% | 82.1% | - |
| 从头重训（无删除集） | 58.7% | 81.5% | 100% |
| GraphEraser | 56.3% | 79.8% | 100% |
| SCRUB | 61.2% | 80.3% | 100% |
| **Cognac (识别5%)** | **72.8%** | **81.2%** | **5%** |
| **Cognac (识别100%)** | **78.5%** | **81.8%** | **100%** |
| Oracle（干净数据训练） | 82.3% | 82.5% | - |

### 消融实验

| 配置 | 受影响准确率 | 说明 |
|------|-----------|------|
| 仅 AC⚡DC（无邻域纠正） | 64.5% | i.i.d. 方法不处理传播 |
| 仅 CoGN（无梯度遗忘） | 67.3% | 邻域纠正但不遗忘源头 |
| **Cognac（两者交替）** | **72.8%** | 最优组合 |
| 耦合优化器（非解耦） | 68.1% | 遗忘和维持互相干扰 |
| **解耦优化器** | **72.8%** | 独立优化更稳定 |
| 识别 1% | 65.2% | 太少不够代表性 |
| 识别 5% | 72.8% | 足够代表性 |
| 识别 20% | 76.1% | 更多信息更好 |

### 关键发现
- 现有图遗忘方法（GraphEraser、GUIDE）完全失败——即使知道 100% 操纵集也无法有效遗忘
- 从头重训不是金标准——（58.7% vs Oracle 82.3%）因为未识别的操纵数据仍在训练集中
- Cognac 仅用 5% 识别就达到 72.8%——接近 100% 识别的 78.5%，证明方法对部分识别鲁棒
- CoGN 的邻域纠正贡献（+8.3%）几乎等于 AC⚡DC 的标签遗忘贡献（+7.8%）——两者同等重要
- 解耦优化器比耦合优化器提升 +4.7%——独立性对稳定性至关重要
- 在 ogbn-arxiv 大图上只需 8× 加速于重训——可扩展到大规模场景

## 亮点与洞察
- **"从头重训不是金标准"**颠覆了遗忘领域的传统假设——在纠正性遗忘中尤其如此
- 对比遗忘（CoGN）利用图结构"由点及面"——5% 已知→邻域发现→泛化到整个操纵集
- 解耦优化器是简单但关键的工程贡献——避免了正(维持)负(遗忘)梯度的直接抵消
- 论文标题的双关极好——Cognac（白兰地）用来"忘记坏记忆"，首字母也是 Contrastive + GNN 的缩写
- 对图数据安全和可信 AI 有重要实践价值

## 局限与展望
- 假设知道操纵类型（边或节点）——不知道类型时需要先诊断
- CoGN 的邻域搜索在密集图中计算开销可能大
- 仅测试了节点分类任务——链接预测、图分类待探索
- 来自标签翻转和边注入——更隐蔽的操纵（如特征扰动）未测试
- 多次操纵（sequential manipulation）的场景未讨论

## 相关工作与启发
- **vs GraphEraser**: 关注隐私删除（removing data），不是纠正性遗忘（correcting influence）
- **vs SCRUB**: i.i.d. 遗忘方法，不处理图的消息传递传播
- **vs Goel et al. (纠正性遗忘)**: 提出了 i.i.d. 场景的纠正性遗忘，Cognac 将其扩展到图
- **启发**：消息传递的传播特性使图遗忘需要"邻域级"而非"节点级"的处理——这对所有图机器学习的后处理干预都有启示

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个有效的 GNN 纠正性遗忘方法
- 实验充分度: ⭐⭐⭐⭐⭐ 多图+多操纵+多基线+不同识别比例
- 写作质量: ⭐⭐⭐⭐⭐ 问题清晰，方法直观，标题有趣
- 价值: ⭐⭐⭐⭐⭐ 对图数据安全和可信 GNN 有重要推动

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[ICML 2025\] Unifews: You Need Fewer Operations for Efficient Graph Neural Networks](unifews_you_need_fewer_operations_for_efficient_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)

<!-- RELATED:END -->
