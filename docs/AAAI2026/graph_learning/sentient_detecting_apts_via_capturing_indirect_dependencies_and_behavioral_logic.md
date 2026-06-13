---
title: >-
  [论文解读] Sentient: Detecting APTs Via Capturing Indirect Dependencies and Behavioral Logic
description: >-
  [AAAI 2026][图学习][APT检测] 提出 Sentient，一种结合图 Transformer 预训练：和双向 Mamba2 意图分析：的 APT 检测方法，仅用良性数据训练即可捕获间接依赖关系、去除场景噪声、关联行为逻辑，在三个标准数据集上平均降低 44% 误报率。 1. 领域现状：APT（高级持续性威胁）因其…
tags:
  - "AAAI 2026"
  - "图学习"
  - "APT检测"
  - "溯源图"
  - "Transformer"
  - "Mamba"
  - "行为意图分析"
---

# Sentient: Detecting APTs Via Capturing Indirect Dependencies and Behavioral Logic

**会议**: AAAI 2026  
**arXiv**: [2502.06521](https://arxiv.org/abs/2502.06521)  
**代码**: 无  
**领域**: 图学习 / 网络安全  
**关键词**: APT检测, 溯源图, 图Transformer, Mamba, 行为意图分析

## 一句话总结

提出 Sentient，一种结合**图 Transformer 预训练**和**双向 Mamba2 意图分析**的 APT 检测方法，仅用良性数据训练即可捕获间接依赖关系、去除场景噪声、关联行为逻辑，在三个标准数据集上平均降低 44% 误报率。

## 研究背景与动机

1. **领域现状**：APT（高级持续性威胁）因其隐蔽性和复杂性难以检测。基于溯源图（provenance graph）的方法是当前最有效的检测手段，利用系统审计日志中的实体关系挖掘攻击痕迹。
2. **现有痛点**：(a) **间接依赖缺失**——GNN 方法受限于邻居聚合的感受野，无法捕获非直接连接节点间的关联；(b) **噪声复杂场景**——感染实体继续执行大量正常任务，邻居聚合错误地混入弱相关活动；(c) **行为逻辑关联缺失**——孤立的系统行为具有上下文多样性（如 sshd 写日志单独看是正常的），但组合起来才显现恶意意图。
3. **核心矛盾**：GNN 的局部聚合既无法触及间接依赖，又因无选择地聚合邻居引入噪声，同时无法建立远距离行为间的逻辑关联。
4. **本文目标**：设计一个全局感知且能理解行为逻辑的 APT 检测方法。
5. **切入角度**：用图 Transformer 的全局注意力捕获间接依赖，用随机游走构建去噪行为序列，用双向 Mamba2 挖掘行为间的逻辑关联。
6. **核心 idea**：图 Transformer 提供全局节点嵌入 + 双向 Mamba2 在行为序列上挖掘意图逻辑 = 解决间接依赖/噪声/逻辑关联三大挑战。

## 方法详解

### 整体框架

五个组件：(1) 图构建——从系统日志构建溯源图，用 Word2Vec 语义编码 + Laplacian 位置编码初始化节点；(2) 预训练——图 Transformer 重建节点关键信息，学习全局结构语义嵌入；(3) 意图分析模块（IAM）——随机游走构建行为序列，双向 Mamba2 挖掘逻辑关联；(4) 威胁检测——MLP 重建行为动作，重建误差超阈值的标记为恶意；(5) 攻击调查——聚类相似意图的行为。

### 关键设计

1. **图 Transformer 预训练**

    - 功能：学习捕获间接依赖的全局节点嵌入，避免 GNN 的感受野限制。
    - 核心思路：初始嵌入 $h_i^0 = \sigma((A^0\alpha + a^0) + (B^0\beta + b^0))$ 结合语义编码 $\alpha$（Word2Vec）和位置编码 $\beta$（Laplacian 特征向量）。多头注意力使每个节点能关注图中所有其他节点（$w_{ij} = \text{softmax}(Q h_i \cdot K h_j / \sqrt{d_k})$），通过残差连接和 FFN 生成最终嵌入。预训练任务是节点类型重建（加权交叉熵处理类别不平衡）。
    - 设计动机：溯源图中攻击行为涉及多跳关系（如读文件→执行→网络发送），GNN 需多层才能触及，但深层 GNN 有过平滑问题。图 Transformer 的全局注意力一步到位。

2. **意图分析模块（IAM）**

    - 功能：在去噪环境中挖掘行为间的逻辑关联，理解行为意图。
    - 核心思路：基于预训练嵌入 $h$，用随机游走在溯源图上构建行为序列 $\lambda_i = \{e_1, ..., e_W\}$（每个行为 $e_t$ 表示为源节点和目标节点嵌入的拼接 $[h_{\phi(e_t)}; h_{\psi(e_t)}]$）。随机游走天然构建了以目标节点为中心的局部上下文，过滤了不相关邻居（去噪）。然后用**双向 Mamba2**处理序列：$\lambda^{\ell+1} = \mathbf{F}(\mathbf{E}(\lambda^\ell) + \mathcal{R}(\mathbf{E}(\mathcal{R}(\lambda^\ell))), \lambda^\ell)$，其中 $\mathcal{R}$ 为序列反转，$\mathbf{E}$ 为 Mamba2 的状态空间模型运算。双向处理确保前后文逻辑均被捕获。
    - 设计动机：孤立行为看似正常但组合起来才暴露恶意意图。Mamba2 的长序列建模能力优于 RNN，且线性复杂度适合大规模日志。双向捕获因为攻击行为可能依赖前后文。

3. **威胁检测与攻击调查**

    - 功能：基于良性模式偏离检测异常，并聚类攻击行为生成攻击故事。
    - 核心思路：训练阶段掩码关键行为信息（读/写/执行），学习重建良性行为模式。检测阶段，重建误差 $RE = \text{CrossEntropy}(\mathbf{P}(a_t), L(a_t))$ 超过阈值（均值+1.5 倍标准差）的行为被标记为恶意。攻击调查阶段，拼接行为意图嵌入 $h_e$ 和源/目标节点嵌入进行聚类 $C_k = \{e_i | \arg\min_k \|h_{behavior}^{(i)} - \mu_k\|^2\}$，合并相似意图的警报减轻分析负担。
    - 设计动机：仅用良性数据训练避免了攻击样本稀缺的问题。重建误差天然度量行为的"异常程度"。

### 损失函数 / 训练策略

预训练损失为加权交叉熵（节点类型重建），检测损失为交叉熵（行为类型重建）。异常阈值设为训练期均值 + 1.5 标准差。

## 实验关键数据

### 主实验

在 Streamspot、Unicorn Wget、DARPA E3 三个数据集上的结果：

| 数据集 | 方法 | Precision | Recall | F-score | FPR |
|--------|------|-----------|--------|---------|-----|
| Streamspot | Threatrace | 98% | 99% | 98% | 0.4% |
| Streamspot | **Sentient** | **99%** | **99%** | **99%** | **0.2%** |
| Unicorn Wget | Threatrace | 93% | 98% | 95% | 7.4% |
| Unicorn Wget | **Sentient** | **96%** | **99%** | **97%** | **4.1%** |
| DARPA Cadets | Flash | 92% | 99% | 95% | 0.3% |
| DARPA Cadets | Slot | 94% | 96% | 95% | 0.2% |
| DARPA Cadets | **Sentient** | **96%** | **99%** | **97%** | **0.2%** |
| DARPA Theia | Flash | 91% | 99% | 95% | 0.8% |
| DARPA Theia | **Sentient** | **95%** | **99%** | **97%** | **0.4%** |
| DARPA Trace | Flash | 93% | 99% | 96% | 0.4% |
| DARPA Trace | **Sentient** | **97%** | **99%** | **98%** | **0.2%** |

### 消融实验

| 配置 | Precision 变化 | 说明 |
|------|---------------|------|
| w/o 预训练 (PT) | -20.75% | 间接依赖信息缺失 |
| w/o 意图分析 (IAM) | -31.59% | 行为逻辑关联丢失，影响最大 |
| w/o Laplacian PE | -8.2% | 拓扑位置信息缺失 |
| w/o 语义编码 | -12.3% | 节点属性语义丢失 |

### 关键发现

- IAM 贡献最大——移除后精度下降 31.59%，说明行为逻辑关联对 APT 检测至关重要。
- 在复杂场景（Unicorn Wget、DARPA Theia）中优势最明显，因为这些场景的噪声和间接依赖更多。
- 仅用良性数据训练即可实现 SOTA 检测能力，是实际部署的重要优势。
- 性能开销可接受：处理一天日志仅需 63.6 秒，峰值内存 2.01GB。

## 亮点与洞察

-  **图 Transformer + 序列 SSM 的组合**：用图 Transformer 做全局表征，用 Mamba2 做序列逻辑关联，分别解决图上和序列上的长距离依赖，这种组合策略可迁移到其他图+序列的任务。
- **随机游走作为去噪手段**：随机游走天然构建以目标节点为中心的上下文，过滤不相关邻居，是一种巧妙的去噪设计。
- **攻击调查的聚类减负**：不仅检测异常，还将相似意图行为聚类成"攻击故事"，减轻安全分析师的工作负担。

## 局限与展望

- 异常阈值（均值+1.5σ）是启发式设定，自适应阈值可能更好。
- 随机游走的序列长度 $W$ 是固定的，自适应长度可能更灵活。
- 未验证在概念漂移（系统行为模式随时间变化）场景下的鲁棒性。
- 攻击调查的聚类方法较简单（K-means），更复杂的聚类可能生成更好的攻击故事。

## 相关工作与启发

- **vs Flash/Threatrace**：使用 GNN（GraphSAGE）邻居聚合，无法捕获间接依赖且引入噪声。Sentient 用图 Transformer 全局注意力解决。
- **vs Slot**：用图强化学习自适应选择邻居，但仍受 GNN 感受野限制。Sentient 完全绕过了邻居聚合范式。
- **vs Atlas**：需要攻击数据训练；Sentient 仅需良性数据。

## 评分

- 新颖性: ⭐⭐⭐⭐ 图Transformer + 双向Mamba2 + 随机游走去噪的组合策略新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集覆盖真实/模拟攻击，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，挑战用图示说明直观
- 价值: ⭐⭐⭐⭐ 对实际网络安全有部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](../../ICLR2026/graph_learning/structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)

</div>

<!-- RELATED:END -->
