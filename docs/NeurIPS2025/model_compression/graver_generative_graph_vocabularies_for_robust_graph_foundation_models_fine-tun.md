---
title: >-
  [论文解读] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning
description: >-
  [NeurIPS 2025][模型压缩][graph foundation model] 提出 Graver 框架，通过 ego-graph 解耦提取可迁移子图词汇、graphon 专家建模词汇分布、MoE-CoE 路由选择性增强 support 样本，解决 GFM 少样本微调中因结构不匹配导致的不稳定性问题。
tags:
  - NeurIPS 2025
  - 模型压缩
  - graph foundation model
  - few-shot fine-tuning
  - graphon
  - vocabulary generation
  - MoE
---

# Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning

**会议**: NeurIPS 2025  
**arXiv**: [2511.05592](https://arxiv.org/abs/2511.05592)  
**代码**: [GitHub](https://github.com/RingBDStack/GRAVER)  
**领域**: 模型压缩  
**关键词**: graph foundation model, few-shot fine-tuning, graphon, vocabulary generation, MoE

## 一句话总结

提出 Graver 框架，通过 ego-graph 解耦提取可迁移子图词汇、graphon 专家建模词汇分布、MoE-CoE 路由选择性增强 support 样本，解决 GFM 少样本微调中因结构不匹配导致的不稳定性问题。

## 研究背景与动机

**领域现状**：图基础模型（GFM）旨在通过"预训练-微调"范式实现跨领域、跨任务的通用图学习。已有方法如 GCOPE、MDGPT、SAMGPT 等在多域预训练和跨域迁移方面取得进展。

**现有痛点**：GFM 的少样本 prompt 微调存在严重不稳定性——性能和适应效率高度依赖 support 样本的随机选择。当选到的 support 节点的结构模式与预训练图匹配时效果好（如三角形），不匹配时效果差（如梯子结构），导致高方差。

**核心矛盾**：少样本设置下标注样本极少（如 one-shot），模型无法从少量样本中充分捕捉目标域结构模式，而预训练知识的迁移又受到领域差异限制。

**本文目标**：如何在随机 support 集下实现鲁棒且高效的 GFM 微调？

**切入角度**：通过生成式增强——利用预训练阶段学到的"图词汇"（可迁移子图模式）来增强 support 样本，减少对特定 support 选择的依赖。

**核心 idea**：从预训练图中提取可迁移子图词汇，通过 graphon 生成器建模其分布，微调时用 MoE-CoE 路由将相关词汇嵌入 support 样本实现上下文增强。

## 方法详解

### 整体框架

三阶段框架：
1. **预训练阶段**：多域对齐 → ego-graph 解耦提取词汇 → 自监督对比预训练
2. **词汇建模阶段**：graphon 专家分别建模结构 token 和特征 token
3. **微调阶段**：MoE-CoE 路由生成词汇 → 增强 support 样本 → prompt 微调

### 关键设计

**1. Factor-aware Ego-Graph 解耦**

将节点 $u$ 的 ego-graph 分解为 $K$ 个因子感知子图（词汇）：
$$\alpha_{v \to k}^{(t)} \propto \text{Softmax}_k(\langle \mathbf{h}_{u,k}^{\mathcal{S}(t)}, \mathbf{h}_{v,k}^{\mathcal{S}(t)} \rangle / \tau)$$

- 通过软路由将邻居分配到 $K$ 个通道
- 互信息正则化确保通道间语义独立：$\mathcal{R}_{\text{MI}}^u = \sum_{i \neq j} I(\mathbf{h}_{u,i}^{\mathcal{S}}; \mathbf{h}_{u,j}^{\mathcal{S}})$
- **Proposition 1**（可迁移性理论保证）：语义差异上界由词汇组合的最优匹配距离决定

**2. Graphon 生成专家**

- **结构 token**：每类收集邻接矩阵 $\{\mathcal{A}_i^{(c)}\}$，估计非参数 graphon $W_c^{\mathcal{A}}: [0,1]^2 \mapsto [0,1]$
- **特征 token**：估计条件于潜在位置的特征分布 $W_c^{\mathcal{X}}: [0,1] \mapsto \mathbb{R}^d$
- **条件生成**：采样潜在位置 $\mathbf{u}_i \sim \mathcal{U}[0,1]$，然后 $\tilde{\mathcal{A}}[i,j] \sim \text{Bern}(W_c^{\mathcal{A}}(\mathbf{u}_i, \mathbf{u}_j))$
- **Proposition 2**（分布收敛）：$\|\mathbf{g}_c^{\text{gen}} - \mathbf{g}_c^{\text{emp}}\|_{\text{TV}} \to 0$ as $N_c \to \infty$

**3. MoE-CoE 路由网络**

两层层次化路由：
- **MoE 层（哪个域）**：$\mathbf{S}_{\text{M}} = \text{Softmax}(\mathbf{W}_{\text{M}}^\top \cdot \phi(\hat{\mathbf{X}}_i^{\mathcal{T}}))$，选择相关源域
- **CoE 层（哪个类）**：$\mathbf{S}_{\text{C}} = \text{Softmax}(\mathbf{W}_{\text{C}}^\top \cdot \phi(\hat{\mathbf{X}}_i^{\mathcal{T}} \| \tilde{\mathcal{X}}))$，组合类级 token
- 组合词汇增强 support：$\tilde{G}_i^{\mathcal{T}} = G_i^{\mathcal{T}} \oplus \tilde{\mathbf{g}}_i^{\text{gen}}$

### 损失函数/训练策略

- **预训练**：对比链接预测 + MI 正则化：$\mathcal{L}_{\text{pre}} = \mathcal{L}_{\text{contrastive}} + \lambda \sum_u \mathcal{R}_{\text{MI}}^u$
- **微调**：类原型匹配 + MoE-CoE 稀疏激活正则：$\mathcal{L}_{\text{ftn}} = \mathcal{L}_{\text{cls}} + \mu \cdot \mathcal{L}_{\text{MoE-CoE}}$
- 预训练参数 $\Theta^*$ 冻结，仅更新 graph prompt $\mathcal{P}_\Omega$ 和 MoE-CoE 权重

## 实验关键数据

### 主实验：One-shot 节点分类（Cross-Dataset）

| 方法 | Cora | CiteSeer | PubMed | arXiv | Tech | Home | Wiki-CS |
|------|------|----------|--------|-------|------|------|---------|
| GCN | 28.40±4.62 | 29.25±3.39 | 40.33±6.90 | 61.59±5.13 | 53.89±3.35 | 36.74±2.53 | 28.58±5.39 |
| SAMGPT | 46.79±6.54 | 38.65±6.35 | 51.92±9.50 | 73.60±— | — | — | — |
| **Graver** | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** | **最优** |

- 平均增益：节点分类 +2.8%，图分类 +3.2%（相对 runner-up）
- 标准差平均相对降低 **54.0%**（节点）和 **54.6%**（图）

### 消融实验

| 移除组件 | 效果 |
|---------|------|
| 无 ego-graph 解耦 | 性能下降，词汇分辨力不足 |
| 无 graphon 生成 | 无法增强 support，退化为标准 prompt tuning |
| 无 MoE-CoE | 无法选择性路由，可能引入负迁移 |
| 无 MI 正则化 | 通道语义重叠，可迁移性下降 |

### 关键发现

- Graver 在 one-shot 和 five-shot 设置中一致超越 15 个 SOTA 基线
- 跨域设置（更困难）中 Graver 优势更大，得益于词汇增强的域适应能力
- 鲁棒性提升最为显著——std 下降超过 54%，说明对 support 选择不敏感
- LLM 增强的语义对齐使得零样本迁移也成为可能

## 亮点与洞察

- **首个将生成式词汇用于 GFM 微调增强的工作**：不同于增强训练数据或增强图结构，而是增强 support 集本身
- **graphon 的优雅应用**：将连续 graphon 函数作为图词汇的生成模型，兼具理论优美性（分布收敛保证）和实用性
- **MoE-CoE 设计精巧**：两层路由分别处理"哪个域"和"哪个类"，避免了负迁移
- **"root + affix" 类比生动**：结构 token = 词根（决定语义类型），特征 token = 词缀（决定域特性）

## 局限与展望

- 词汇数量 $K$ 和 graphon 分辨率是关键超参数，敏感性分析有限
- Graphon 估计在大规模稀疏图上可能不稳定
- 多域预训练的计算成本较高（需要同时处理 7 个数据集 + LLM 增强）
- ego-graph 解耦需要多轮迭代，增加预训练开销
- 仅在引文、购物、网页三个域上验证，缺少分子/蛋白质等科学领域实验

## 相关工作与启发

- GFT 的计算树词汇限于树结构，Graver 的 ego-graph 词汇更通用
- $\mathcal{G}$-Mixup 也用 graphon 但用于数据增强，Graver 用于词汇建模
- 与 NLP 中的 in-context learning 思想相通：通过嵌入上下文词汇来引导适应
- 启发：图领域的 prompt tuning 可能需要比文本/图像更多的结构先验注入

## 评分

⭐⭐⭐⭐ (4/5)

方法设计精巧，理论支撑充分（两个 Proposition），实验全面且鲁棒性提升显著。但系统复杂度较高，可能限制实际应用；缺少更多领域的验证。
