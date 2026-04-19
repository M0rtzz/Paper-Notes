---
title: >-
  [论文解读] Transferable Hypergraph Attack via Injecting Nodes into Pivotal Hyperedges
description: >-
  [AAAI 2026][AI安全][超图神经网络] 提出 TH-Attack，一种面向超图神经网络（HGNNs）的可迁移节点注入攻击框架，通过识别信息聚合路径中的关键超边并注入语义反转的恶意节点，在黑盒场景下实现对多种 HGNN 架构的有效攻击，Accuracy 可从 80%+ 降至 30% 以下。
tags:
  - AAAI 2026
  - AI安全
  - 超图神经网络
  - 对抗攻击
  - 节点注入
  - 超边中枢性
  - 可迁移攻击
---

# Transferable Hypergraph Attack via Injecting Nodes into Pivotal Hyperedges

**会议**: AAAI 2026  
**arXiv**: [2511.10698](https://arxiv.org/abs/2511.10698)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 超图神经网络, 对抗攻击, 节点注入, 超边中枢性, 可迁移攻击

## 一句话总结

提出 TH-Attack，一种面向超图神经网络（HGNNs）的可迁移节点注入攻击框架，通过识别信息聚合路径中的关键超边并注入语义反转的恶意节点，在黑盒场景下实现对多种 HGNN 架构的有效攻击，Accuracy 可从 80%+ 降至 30% 以下。

## 研究背景与动机

超图（Hypergraph）通过允许超边连接两个或多个节点，可以建模比普通图更复杂的高阶关系，在推荐系统、生物网络、3D视觉等领域展现出优越性能。超图神经网络（HGNNs）通过"节点→超边→节点"的两阶段信息聚合机制捕获高阶特征。

随着 HGNNs 被部署在医疗诊断、金融风控等关键领域，其对抗鲁棒性问题变得紧迫。然而现有超图攻击方法存在明显局限：

**现有方法的不足**：
- **超图修改攻击**（如 HyperAttack、MGHGA）：依赖目标模型的梯度信息，白盒/灰盒假设
- **超图注入攻击**（如 IE-Attack、H3NI）：IE-Attack 选择"精英超边"注入 KDE 生成的同质节点，H3NI 用遗传算法选超边，但都依赖特定 HGNN 的信息机制

**核心洞察**：所有现有方法都忽略了 HGNNs 的一个**通用脆弱性**——超边在信息聚合路径中的中枢性（pivotality）存在显著差异。

如图1所示：节点 $v_1$ 仅通过超边 $e_1$ 获取高阶特征，而节点 $v_3$ 有两条聚合路径（$e_2, e_3$）。攻击 $e_1$ 会直接破坏 $v_1$ 的信息传播，使 HGNN 无法正确预测；而攻击 $e_3$ 对 $v_3$ 影响有限，因为 $v_3$ 还有 $e_2$ 作为备用路径。

因此，$e_1$ 比 $e_2, e_3$ 具有更高的中枢性。这一通用脆弱性存在于所有基于"节点-超边-节点"聚合机制的 HGNNs 中，攻击中枢超边可以实现跨架构迁移。

## 方法详解

### 整体框架

TH-Attack 包含三个关键模块：

1. **超边识别器（Hyperedge Recognizer）**：通过中枢性评估识别关键超边
2. **特征反转器（Feature Inverter）**：基于关键超边特征生成语义反转的恶意节点
3. **注入攻击**：将恶意节点注入关键超边，破坏信息传播

### 关键设计

#### 1. **超边中枢性评估与识别器**

从 HGNN 的聚合过程出发：

节点→超边聚合：$\mathbf{z}_j^{(l)} = \frac{1}{|e_j|} \sum_{v_i \in e_j} \frac{1}{\sqrt{d_{v_i}}} \mathbf{x}_i^{(l)} \Theta^{(l)}$

超边→节点聚合：$\mathbf{x}_i^{(l+1)} = \frac{1}{\sqrt{d_{v_i}}} \sum_{e_j \ni v_i} w_{e_j} \mathbf{z}_j^{(l)}$

对于节点 $v_i$，定义其**隔离度**为其超度数（所属超边数）：

$$d_h(v_i) = |\{e_j \in \mathcal{E} \mid v_i \in e_j\}|$$

若 $d_h(v_i) \leq \tau$（中枢性等级阈值），则 $v_i$ 所在超边为关键超边。

**理论支撑**：

**定理1（高中枢性超边的扰动放大）**：当节点 $v_i$ 通过高中枢性超边聚合信息时，其特征扰动的下界为：

$$\|\Delta \mathbf{x}_i^{(l+1)}\|_2 \geq \frac{1}{\sqrt{d_{v_i}}} \min_{e_j \ni v_i} w_{e_j} \cdot \|\Delta \mathbf{z}_j^{(l)}\|_2$$

**定理2（低中枢性超边的扰动衰减）**：当节点 $v_k$ 通过低中枢性超边聚合时，扰动被多条路径分散：

$$\|\Delta \mathbf{x}_k^{(l+1)}\|_2 \leq \frac{1}{\sqrt{d_{v_k}}} \sum_{e_j \ni v_k} w_{e_j} \|\Delta \mathbf{z}_j^{(l)}\|_2$$

**设计动机**：高中枢性超边是信息传播的唯一/稀少通道，攻击这些超边会导致扰动放大效应；低中枢性超边有冗余路径分散扰动能量。这是一种**结构性脆弱性**，独立于具体 HGNN 架构，因此攻击可迁移。

最终选择的关键超边集合：

$$\mathcal{E}_{all\_piv} = \{e_j \in \mathcal{E} \mid \exists v_i \in e_j \text{ s.t. } d_h(v_i) \leq \tau\}$$

#### 2. **基于关键超边的特征反转器**

目标：生成恶意节点特征，使其与所注入超边的特征产生最大语义偏差，从而在聚合时注入"毒药"。

**初始混淆特征生成**：

$$\mathbf{x}_{ini}^{(j)} = \mathbf{x}_{pro}^{(j)} \oplus \mathcal{N}(0, \mu^2)$$

其中 $\mathbf{x}_{pro}^{(j)} = \prod_{v_i \in e_j} \mathbf{x}_i$ 是超边内节点特征的逐元素乘积，保持统计相关性的同时引入高斯噪声增强多样性。

**MLP 增强**：通过多层 MLP + LeakyReLU 增强混淆效果，最终经 Softmax 输出恶意特征。

**损失函数**——最大化语义偏差 + 约束偏差幅度：

$$\mathcal{L}_{cos\_dis} = \cos(\mathbf{x}_{mal}^{(j)}, \mathbf{z}_{e_j}) + \lambda \cdot \mathcal{L}_{reg}$$

其中 $\mathcal{L}_{reg} = \max(\cos(\mathbf{x}_{mal}^{(j)}, \mathbf{z}_{e_j}) - t, 0)$，$t$ 是相似度阈值。

**设计动机**：
- 最小化余弦相似度使恶意节点特征与超边特征方向相反，注入后在聚合中产生最大干扰
- 正则化项约束偏差不能过大，保持攻击隐蔽性
- 超边特征 $\mathbf{z}_{e_j} = H^\top \cdot \mathcal{X}$ 仅依赖超图结构，不需要模型参数——实现黑盒攻击

#### 3. **注入攻击与跨模型迁移**

将生成的恶意节点注入关键超边：$e_j = \{v_1, v_2\} \to \{v_1, v_2, v_{mal}^{(j)}\}$

更新后的关联矩阵 $\hat{H}$ 节点维度增加 $m$，超边维度不变。攻击后的超图 $\hat{\mathcal{G}} = (\hat{\mathcal{V}}, \hat{\mathcal{E}})$ 可直接输入任意 HGNN，无需了解目标模型参数或架构细节。

**迁移性来源**：
- 关键超边的识别仅依赖超图结构，不依赖具体 HGNN
- 特征反转仅依赖超边特征聚合（$H^\top \mathcal{X}$），不依赖模型参数
- 所有基于"节点-超边-节点"聚合的 HGNN 都共享这一结构性脆弱性

### 损失函数 / 训练策略

- 特征反转器通过反向传播优化 $\mathcal{L}_{cos\_dis}$
- 攻击预算 $\Phi$ 由扰动率 $\eta$ 和节点数 $N$ 决定，通常不超过节点总数的 5%
- 最优超参组合：$\lambda=0.1, t=0.9$（低正则化 + 高相似度阈值 = 最大攻击强度）

## 实验关键数据

### 主实验

**6个数据集 × 5个HGNNs × 6种攻击方法，Accuracy 对比（%）**

| 数据集/模型 | Clean | Random | DICE | FGA | IGA | IE-Attack | **TH-Attack** |
|-----------|-------|--------|------|-----|-----|-----------|---------------|
| Cora/HGNN | 76.41 | 74.71 | 74.45 | 74.41 | 73.84 | 73.20 | **36.08** |
| Cora/HyperGCN | 75.95 | 73.14 | 73.93 | 72.62 | 71.09 | 68.37 | **31.55** |
| Cora/UniGCNII | 80.08 | 75.82 | 77.23 | 76.65 | 76.01 | 76.57 | **39.42** |
| Cora-CA/UniGCNII | 84.68 | 79.53 | 80.15 | 80.57 | 79.07 | 83.95 | **32.72** |
| Pubmed/HGNN | 84.28 | 80.99 | 81.29 | 81.99 | 81.60 | 84.53 | **40.96** |
| DBLP/HyperGCN | 89.54 | 83.84 | 81.72 | 85.85 | 81.83 | 82.64 | **46.23** |
| ModelNet40/UniGCNII | 97.86 | 93.45 | 93.51 | 93.36 | 93.48 | 96.65 | **53.50** |

TH-Attack 的攻击效果远超所有基线，Accuracy 下降幅度通常为 **30-50 个百分点**，而基线通常仅下降 2-10 个百分点。

### 消融实验

| 变体 | Cora | Cora-CA | Citeseer | Pubmed |
|------|------|---------|----------|--------|
| w/o 超边识别器(HR) | 41.19 / 34.87 / 43.48 | 38.30 / 24.51 / 40.00 | 28.69 / 21.43 / 34.05 | 45.16 / 36.47 / 47.14 |
| w/o 特征反转器(FI) | 61.80 / 38.65 / 73.85 | 58.40 / 26.57 / 77.72 | 54.26 / 43.93 / 66.94 | 44.25 / 38.97 / 47.95 |
| w/o 余弦距离损失(CDL) | 61.05 / 59.42 / 59.80 | 59.31 / 54.66 / 60.34 | 54.45 / 52.64 / 66.06 | 55.85 / 52.76 / 46.40 |
| **完整 TH-Attack** | **36.08 / 31.55 / 39.42** | **32.03 / 17.02 / 32.72** | **24.17 / 20.59 / 27.00** | **40.96 / 35.14 / 44.13** |

（每组三个数值分别对应 HGNN / HyperGCN / UniGCNII）

- **CDL 缺失影响最大**：移除余弦距离损失导致攻击效果大幅下降，说明最大化语义偏差是攻击的核心驱动力
- **HR 和 FI 各自贡献显著**：验证了关键超边选择和恶意特征生成的互补作用

### 关键发现

1. **极端攻击效果**：Cora 上 HGNN 从 76.41% 降至 36.08%，下降 40+ 个百分点；Cora-CA 上 HyperGCN 从 76.32% 降至 17.72%
2. **极低预算高效率**：$\eta=1\%$ 时仅注入 23 个节点（Cora），HGNN Accuracy 就下降 17.17%，而基线最多下降 5%
3. **跨架构强迁移性**：同一攻击数据对 5 种不同架构的 HGNN 都有效，基线（尤其 IE-Attack）通常只对特定架构有效
4. **中枢性等级 $\tau$ 的影响**：$\tau=1,2$ 时攻击最强，$\tau \geq 3$ 后性能递减，验证了中枢性假设
5. **复杂数据集上优势更大**：在 ModelNet40 等复杂数据上，基线几乎无效（Accuracy 下降 2-4%），TH-Attack 仍能大幅降低性能

## 亮点与洞察

- **问题发现有深度**：首次将攻击思路从"攻击哪些节点/边"转向"攻击信息聚合路径中的结构性瓶颈"，是对 HGNN 架构脆弱性的深层理解
- **理论与实践统一**：定理 1&2 从聚合公式出发形式化了中枢性概念，实验验证了理论预测
- **攻击简洁高效**：不需要梯度信息、不需要代理模型、不需要知道目标架构——纯粹基于超图结构和节点特征的黑盒攻击
- **攻击效果惊人**：在 5% 注入预算下将 SOTA HGNNs 的分类准确率打到接近随机猜测水平

## 局限与展望

1. **仅针对节点分类任务**：未验证在超图link prediction、社区检测等其他任务上的效果
2. **防御视角缺失**：未讨论可能的防御策略，如检测异常度高的注入节点、加权聚合中降低中枢超边权重等
3. **特征反转器的隐蔽性**：虽然有正则化约束，但论文未定量评估注入节点在特征空间中的异常度——是否容易被异常检测器识别？
4. **动态超图未考虑**：作者提到未来工作包括动态 HGNN 攻击
5. **实验中使用 poisoning 设置**：IE-Attack 原本设计为 evasion attack，改为 poisoning 后对比可能不完全公平

## 相关工作与启发

- 中枢性（pivotality）的概念可以推广到普通 GNN 中的"桥边"或"关节点"攻击
- 特征反转（Feature Inverter）思想可用于生成高质量的对抗样本
- 该工作揭示了消息传播机制中结构性瓶颈的通用脆弱性，对 HGNN 鲁棒性研究有重要警示意义
- 启发防御方向：冗余聚合路径设计、中枢超边的自适应保护

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 中枢性概念新颖且有理论支撑，攻击思路从结构性瓶颈出发独辟蹊径
- 实验充分度: ⭐⭐⭐⭐⭐ — 6数据集 × 5模型 × 6基线 × 多扰动率，消融和参数分析全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，motivation 图示直观，但部分公式推导可更紧凑
- 价值: ⭐⭐⭐⭐⭐ — 对 HGNN 安全性研究有重要贡献，揭示了一个通用的结构性脆弱性

<!-- RELATED:START -->

## 相关论文

- [Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](../../CVPR2026/ai_safety/towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)
- [GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [Understanding Model Ensemble in Transferable Adversarial Attack](../../ICML2025/ai_safety/understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)
- [Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

<!-- RELATED:END -->
