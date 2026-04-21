---
title: >-
  [论文解读] GraphNarrator: Generating Textual Explanations for Graph Neural Networks
description: >-
  [ACL 2025][图学习][图神经网络] GraphNarrator是首个为图神经网络生成自然语言解释的方法，通过显著性图语言化生成伪标签、信息论指标驱动的专家迭代自改进、以及知识蒸馏训练端到端解释器，实现了忠实、简洁且人类友好的GNN决策解释。
tags:
  - ACL 2025
  - 图学习
  - 图神经网络
  - 自然语言解释
  - 文本属性图
  - 专家迭代
  - 知识蒸馏
---

# GraphNarrator: Generating Textual Explanations for Graph Neural Networks

**会议**: ACL 2025  
**arXiv**: [2410.15268](https://arxiv.org/abs/2410.15268)  
**代码**: 无  
**领域**: graph_learning  
**关键词**: GNN可解释性, 自然语言解释, 文本属性图, 专家迭代, 知识蒸馏  

## 一句话总结

GraphNarrator是首个为图神经网络生成自然语言解释的方法，通过显著性图语言化生成伪标签、信息论指标驱动的专家迭代自改进、以及知识蒸馏训练端到端解释器，实现了忠实、简洁且人类友好的GNN决策解释。

---

## 研究背景与动机

**研究背景：** 图表示学习在推荐系统、社交网络等领域广泛应用，但GNN的内部决策过程仍然不透明。现有GNN解释方法主要提供节点/边级别的重要性分数，在文本属性图（TAG）场景下，这些基于token重要性的解释冗余且缺乏整合性，难以被人类理解。

**现有方法的局限性：** (1) 显著性图/特征重要性方法（如GNNExplainer）仅给出重要性分数，无法解释语义信息；(2) 在多节点子图中，逐token的重要性标注过于碎片化和冗余；(3) 外部语言模型对模型内部决策过程缺乏认知，零样本生成的解释不可靠。

**核心动机：** 自然语言解释具有总结性、简洁性和可读性，是更适合人类理解的解释模态。但缺乏真实标注的解释数据，需要解决"无标签训练高质量解释生成器"的核心挑战。

---

## 方法详解

### 整体框架

GraphNarrator采用三阶段流水线：(1) 显著性解释生成与语言化 → (2) 专家迭代驱动的伪标签生成器训练 → (3) 知识蒸馏训练端到端解释器。

### 关键设计

**1. 显著性文本图语言化（Saliency Paragraph）：** 先用显著性方法（如LRP、Input×Grad）获取节点和token的重要性分数，然后通过BFS将ego图分解为树结构，再用先序遍历将树组织为层次化的文档段落。每个节点的文本作为一个section，子节点作为subsection，跨边用引用句表示。重要性分数附加在token后面，形如 `token(score)`。

**2. 信息论解释质量度量：** 提出三个训练目标：
- **输入忠实度 $f_S$：** 使用PMI度量解释与重要输入区域的互信息，通过掩码token预测近似计算
- **输出忠实度 $f_F$：** 使用PMI度量解释与模型预测的互信息
- **简洁度 $f_B$：** 解释长度与输入长度的比值，鼓励简洁表达

**3. 专家迭代自改进：** 基于Expert Iteration的闭环优化：测量解释质量 → 筛选高质量解释 → 微调伪标签生成器 → 生成新一轮候选解释。

### 损失函数/训练目标

输入忠实度通过采样不同阈值 $\tau$ 的掩码token预测来估计：

$$f_S \approx \int_0^1 P(\tau) \cdot \log \frac{P_{MLM}(\mathcal{R}_\tau | \mathcal{G}_{M_\tau}, E)}{P_{MLM}(\mathcal{R}_\tau | \mathcal{G}_{M_\tau})} d\tau$$

最终通过多目标优化平衡 $f_S$（最大化）、$f_F$（最大化）和 $f_B$（最小化）。

---

## 实验

### 主实验结果

| 数据集 | 方法 | Simul.↑ | PMI-10%↑ | PMI-20%↑ | PMI-30%↑ | Brevity↓ |
|--------|------|---------|----------|----------|----------|----------|
| DBLP | LLaMA3.1 8B | 0.63 | 0.139 | 0.109 | 0.077 | 0.394 |
| DBLP | GPT-4o | 0.82 | 0.142 | 0.101 | 0.085 | 0.385 |
| DBLP | **GraphNarrator** | **0.95** | **0.155** | **0.108** | **0.085** | **0.354** |
| Cora | GPT-4o | 0.95 | 0.414 | 0.284 | 0.225 | 0.357 |
| Cora | **GraphNarrator** | **0.97** | **0.418** | **0.290** | **0.227** | **0.315** |
| Book-History | GPT-4o | 0.89 | 0.456 | 0.313 | 0.240 | 0.768 |
| Book-History | **GraphNarrator** | **0.96** | **0.533** | **0.374** | **0.291** | **0.506** |

### 消融实验

专家迭代的效果通过迭代轮次对比验证（详见原文），每轮迭代伪标签质量稳步提升。三个训练目标（$f_S$, $f_F$, $f_B$）的平衡配置（top-50%筛选）效果最佳。

| 消融维度 | 观察 |
|----------|------|
| 无Expert Iteration | 解释忠实度和简洁度均下降 |
| 仅保留$f_S$ | 解释冗长，简洁度差 |
| 仅保留$f_B$ | 解释过短，忠实度差 |

### 关键发现

- GraphNarrator在所有数据集上全面超越GPT-4o零样本解释，尤其在仿真度（Simulatability）上优势明显（DBLP: 0.95 vs 0.82）
- 在Book-History数据集上，PMI-10%提升16.9%，简洁度改善34.1%
- 基于LLaMA3.1 8B的小模型经微调后可超越GPT-4o，验证了Expert Iteration的有效性

---

## 亮点

- 首次提出为GNN生成自然语言解释的方法，开辟了图模型可解释性的新范式
- 创新性地将显著性图解释转化为结构化文档（Saliency Paragraph），使LLM能理解图结构信息
- 提出信息论驱动的无监督解释质量度量，无需人工标注即可评估和优化解释质量
- 通过Expert Iteration和知识蒸馏，实现了高效的端到端解释生成

## 局限性

- 依赖显著性方法作为初始信号，其本身的局限性会传递到最终解释中
- 仅在节点分类任务上验证，未扩展到链接预测、图分类等更多任务类型
- 伪标签生成依赖GPT-4o-mini，在资源受限场景下的适用性有限
- 人工评估规模较小（50个样本×3标注员），统计显著性有待验证

## 相关工作

- **GNN可解释性：** GNNExplainer (Ying et al., 2019)、PGExplainer (Luo et al., 2020) 等提供结构化解释
- **自然语言解释：** e-SNLI (Camburu et al., 2018)、Chain-of-Thought (Wei et al., 2022) 等面向文本模型
- **SMV方法：** Feldhus et al. (2022) 将显著性图语言化用于文本分类模型解释
- **Expert Iteration：** Dong et al. (2023) 的迭代自改进框架

---

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总评 | 8/10 |

<!-- RELATED:START -->

## 相关论文

- [Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](can_graph_neural_networks_learn_language.md)
- [Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)
- [Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)

<!-- RELATED:END -->
