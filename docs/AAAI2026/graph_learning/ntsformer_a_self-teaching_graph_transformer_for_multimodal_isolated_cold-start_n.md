---
title: >-
  [论文解读] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification
description: >-
  [AAAI 2026][图学习][冷启动节点分类] 提出 NTSFormer（Neighbor-to-Self Graph Transformer），一个统一的图Transformer框架，通过冷启动注意力掩码实现**自教学范式**——同一模型同时产生基于自身特征的"学生"预测和基于邻居信息的"教师"预测，无需退化为MLP即可处理多模态图上的孤立冷启动节点分类，结合MoE输入投影和多模态图预计算有效处理模态缺失问题。
tags:
  - "AAAI 2026"
  - "图学习"
  - "冷启动节点分类"
  - "Transformer"
  - "自教学"
  - "模态缺失"
  - "混合专家"
---

# NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification

**会议**: AAAI 2026  
**arXiv**: [2507.04870](https://arxiv.org/abs/2507.04870)  
**代码**: [https://github.com/CrawlScript/NTSFormer](https://github.com/CrawlScript/NTSFormer)  
**领域**: 图学习 / 图Transformer  
**关键词**: 冷启动节点分类, 图Transformer, 自教学, 模态缺失, 混合专家

## 一句话总结

提出 NTSFormer（Neighbor-to-Self Graph Transformer），一个统一的图Transformer框架，通过冷启动注意力掩码实现**自教学范式**——同一模型同时产生基于自身特征的"学生"预测和基于邻居信息的"教师"预测，无需退化为MLP即可处理多模态图上的孤立冷启动节点分类，结合MoE输入投影和多模态图预计算有效处理模态缺失问题。

## 研究背景与动机

### 问题定义

**多模态孤立冷启动节点分类**：在多模态图中，新加入的节点（如社交网络新用户）面临双重挑战：

**结构孤立（Isolation）**：没有任何连接边，GNN无法利用图结构

**模态缺失（Missing Modalities）**：某些数据（如文本或图像）缺失（例如新用户只有头像没有个人描述）

### 现有方法的困境

**GNN在冷启动下失效**：实验表明，包括GraphSAGE、MMGCN、MGAT在内的GNN模型在孤立冷启动场景下表现极差，**甚至不如简单的MLP**。原因是GNN训练时使用图结构，但测试时节点完全孤立，产生严重的训练-测试分布偏移。

**MLP-学生方法的容量瓶颈**：
- GLNN、SGKD、SimMLP等方法采用**教师-学生范式**：用GNN教师将结构知识蒸馏到MLP学生中
- MLP学生虽然训练和测试条件一致（都不用图结构），避免了分布偏移
- 但**MLP模型容量有限**，在多模态场景下难以同时处理模态缺失的复杂情况

### 核心洞察

**不必退化为MLP**。通过在统一的图Transformer中设计冷启动注意力掩码，可以让同一个Transformer既产生仅基于自身特征的预测（模拟冷启动），又产生利用邻居信息的预测（提供监督信号），实现端到端的自教学，充分利用Transformer的大容量处理模态缺失。

## 方法详解

### 整体框架

NTSFormer 包含三个关键模块：
1. **多模态图预计算（Multimodal Graph Pre-computation）**：一次性将多跳邻居信息转化为固定长度token序列
2. **混合专家输入投影（MoE Input Projection）**：动态路由不同类型的token到专家网络
3. **通过冷启动掩码的邻居到自身教学（Neighbor-to-Self Teaching）**：冷启动注意力掩码分离学生和教师上下文

### 关键设计

#### 1. **多模态图预计算**：将图结构转化为Transformer输入序列

**目标**：将多跳邻居信息按模态和层次组织为固定长度的token序列，适合Transformer处理。

**具体过程**：
- 对文本特征 $X^{(t)}$ 和视觉特征 $X^{(v)}$，先零填充对齐维度至 $d_{in} = \max(d_t, d_v)$
- 计算 $K$ 跳邻居信息：$\{\hat{A}^k X^{(t)} | k=1,...,K\}$ 和 $\{\hat{A}^k X^{(v)} | k=1,...,K\}$

**Token序列构建**：
- **自信息token（Self Tokens）**：

$$\mathcal{X}_{\text{self}} = [X^{(t)} \text{ or } \langle\text{MISS}\rangle, \ X^{(v)} \text{ or } \langle\text{MISS}\rangle, \ \langle\text{CLS}_S\rangle]$$

如果模态缺失，用可学习的 $\langle\text{MISS}\rangle$ 占位符替代。训练时以概率 $p_{\text{miss}}$ 随机替换模拟缺失。

- **邻居token（Neighbor Tokens）**：

$$\mathcal{X}_{\text{nbr}} = [\hat{A}X^{(t)}, \ \hat{A}X^{(v)}, \ ..., \ \hat{A}^K X^{(t)}, \ \hat{A}^K X^{(v)}, \ \langle\text{CLS}_T\rangle]$$

- 完整序列 $S = \mathcal{X}_{\text{self}} \oplus \mathcal{X}_{\text{nbr}}$，长度 $L = 2K + 4$

**关键优点**：预计算在CPU上一次性完成，不需要梯度，后续可使用标准mini-batch训练。

#### 2. **MoE输入投影**：差异化处理不同来源的token

**设计动机**：输入token来自不同来源（自身/邻居、不同模态、特殊token），统一使用共享MLP投影会丢失信息。

**位置感知门控网络**：
- 拼接token特征和one-hot位置向量：$\tilde{S}[i] = [S[i] \| \mathbf{1}_N e_i^\top]$
- 计算门控分数：$\gamma = \text{softmax}(\tilde{S}[i] \cdot W_{\text{gate}}) \in \mathbb{R}^{N \times M}$
- 每个token选择 top-$\hat{k}$ 个专家，输出加权组合：

$$S'_{\text{RE}}[i]_j = \sum_{m=1}^{M} \mathcal{T}(S[i])_{j,m} \cdot \gamma_{j,m} \cdot \text{LN}(\text{MLP}_{\text{RE}_m}(S[i]_j))$$

- 加上共享专家的输出：$S'[i]_j = S'_{\text{RE}}[i]_j + \text{LN}(\text{MLP}_{\text{SE}}(S[i]_j))$
- MoE负载均衡损失：$\mathcal{L}_{\text{MoE}} = \sum_{m=1}^{M} P_m \cdot f_m$

#### 3. **冷启动注意力掩码的自教学机制**：核心创新

**关键问题**：标准自注意力允许所有token相互注意，会导致邻居信息泄漏到 $\langle\text{CLS}_S\rangle$，破坏冷启动假设。

**冷启动注意力掩码设计**：

$$\mathcal{M} = \begin{pmatrix} \mathbf{1}^{3 \times 3} & \mathbf{0}^{3 \times (L-3)} \\ \mathbf{1}^{(L-3) \times 3} & \mathbf{1}^{(L-3) \times (L-3)} \end{pmatrix}$$

- 自信息token（前3个位置）**只能互相注意**，完全无法访问邻居token
- 邻居token可以注意所有token（包括自信息token）
- 结果：$\langle\text{CLS}_S\rangle$ 的表示是纯自身特征的，适合孤立冷启动推理；$\langle\text{CLS}_T\rangle$ 整合了全部上下文，作为教师监督信号

**两个预测分支**：
- **学生预测**（基于 $\langle\text{CLS}_S\rangle$）：$Z_S = \text{softmax}(\text{MLP}_S(H[:,3]))$
- **教师预测**（基于 $\langle\text{CLS}_T\rangle$ 及倒数第二个token的平均）：$Z_T = \frac{1}{2}(Z_{T_1} + Z_{T_2})$

**自教学损失**（KL散度，教师端停止梯度）：

$$\mathcal{L}_{\text{ST}} = \text{KL}(\text{stopgrad}(Z_T) \| Z_S)$$

### 损失函数 / 训练策略

总训练目标：

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{ST}} + \gamma \mathcal{L}_{\text{MoE}}$$

- $\mathcal{L}_{\text{CE}}$：教师预测的交叉熵损失
- $\mathcal{L}_{\text{ST}}$：自教学KL散度损失（$\lambda=1.0$）
- $\mathcal{L}_{\text{MoE}}$：MoE负载均衡损失（$\gamma=0.1$）
- 优化器：AdamW，学习率 $2 \times 10^{-3}$
- 冷启动推理时只使用 $\mathcal{X}_{\text{self}}$，从 $\langle\text{CLS}_S\rangle$ 输出预测

## 实验关键数据

### 实验设置
- **数据集**：Movies（16K节点）、Ele-fashion（97K节点）、Goodreads-NC（685K节点）
- **划分**：20%有标签训练、60%无标签训练（参与消息传递）、10%验证/10%测试（完全孤立）
- **模态缺失设置**：测试集均分为Text-Miss、Visual-Miss、No-Miss三等份
- **隐层维度**：512，$K=2$，6个路由专家+1个共享专家，2层Transformer

### 主实验

| 方法 | Movies All | Ele-fashion All | Goodreads-NC All | 特点 |
|------|-----------|----------------|-----------------|------|
| MLP | 41.85 | 75.15 | 55.56 | 基线 |
| GraphSAGE | 39.02 | 66.56 | 41.65 | GNN，**不如MLP** |
| MMGCN | 40.77 | 68.52 | 43.69 | 多模态GNN，冷启动下差 |
| GLNN | 43.04 | 74.41 | 54.01 | MLP-学生 |
| MUSE | 43.44 | **80.66** | 48.49 | 处理模态缺失 |
| **NTSFormer** | **46.12** | **83.37** | **61.58** | **全面最优** |

NTSFormer 在所有数据集、所有模态缺失设置下**一致性地大幅领先**所有基线。

### 消融实验

| 配置 | Movies | Ele-fashion | Goodreads-NC | 说明 |
|------|--------|-------------|-------------|------|
| NTSFormer (full) | **46.12** | **83.37** | **61.58** | 完整模型 |
| w/o MMPre | 下降 | 下降 | 下降 | 多模态预计算有效 |
| w/o MoE | 下降 | 下降 | 下降 | MoE投影比共享线性更好 |
| w/o SelfTeach | 明显下降 | 明显下降 | 明显下降 | **自教学是核心**，退化为MLP学生性能大降 |

Transformer层数影响（All设置准确率%）：

| 层数 $L^{(tf)}$ | Movies | Ele-fashion | Goodreads-NC |
|---|--------|-------------|-------------|
| 1 | 45.22 | 83.54 | 61.36 |
| 2 | **46.12** | 83.37 | 61.58 |
| 3 | 45.27 | 83.51 | **61.68** |
| 4 | 45.32 | **83.67** | 61.51 |

### 关键发现

1. **GNN在孤立冷启动下全面劣于MLP**：GraphSAGE在三个数据集上均不如MLP，验证了冷启动对GNN的严重影响
2. **自教学 > MLP-学生**：移除自教学（退化为MLP学生的两分支设置）导致显著性能下降，证明Transformer容量优势
3. **模态缺失处理的优越性**：NTSFormer在Text-Miss和Visual-Miss子集上均领先，特别是在Goodreads-NC Text-Miss上达50.99%（MLP仅40.25%）
4. **训练效率**：在大规模数据集Goodreads-NC上，NTSFormer仅需260秒，显著优于MUSE(1437s)和GLNN(778s)，得益于一次性预计算
5. **MoE专家数量**：$M<3$ 时性能下降，说明多样化token需要多个专家网络差异化处理

## 亮点与洞察

1. **优雅的自教学设计**：通过一个注意力掩码矩阵，在同一个Transformer中实现教师和学生的角色分离，比传统两阶段教师-学生方法更简洁高效
2. **冷启动掩码的简洁性**：核心思想极其简单——阻止自信息token看到邻居token，但效果显著。这种binary mask设计几乎不增加计算开销
3. **一次性预计算的可扩展性**：将图结构转化为规则张量，避免训练时的重复消息传递，在68万节点的Goodreads-NC上训练仅260秒
4. **MoE的位置感知设计**：通过拼接one-hot位置向量让门控网络感知token的语义角色（自身/邻居、文本/视觉），比朴素共享投影更合理
5. **$\langle\text{MISS}\rangle$ 占位符 + 训练时随机drop的设计**使模型天然具备处理模态缺失的能力

## 局限与展望

1. **仅支持两种模态（文本+视觉）**，扩展到更多模态需要调整预计算和掩码设计
2. **预计算假设图结构在训练期间不变**，不适用于动态图场景（论文已提及）
3. **教师信号质量依赖于邻居的标签同质性**，在异质图上效果可能受限
4. **数据集规模较小**（最大68万节点），在更大规模图上的效率和效果有待验证
5. **可探索方向**：动态图适配、自适应掩码设计、与LLM特征结合、半监督场景扩展

## 相关工作与启发

- **NAGphormer** 预计算邻居特征作为Transformer输入但仅处理单模态图，NTSFormer扩展到多模态并增加MoE
- **GLNN/SGKD** 的教师-学生范式启发了自教学设计，但NTSFormer通过注意力掩码实现了端到端训练
- 冷启动注意力掩码的思路可迁移到**推荐系统**（新用户/物品）、**知识图谱补全**（新实体）等场景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 冷启动掩码的自教学范式简洁而有效，是真正的创新
- **实验充分度**: ⭐⭐⭐⭐ — 三数据集+全面消融+效率分析，但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机层层递进，从GNN失效→MLP-学生容量瓶颈→自教学GT
- **实用价值**: ⭐⭐⭐⭐⭐ — 开源、训练效率高、方法通用、一次性预计算可扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation](motorec_sparse-regularized_multimodal_tokenization_for_cold-start_recommendation.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)

</div>

<!-- RELATED:END -->
