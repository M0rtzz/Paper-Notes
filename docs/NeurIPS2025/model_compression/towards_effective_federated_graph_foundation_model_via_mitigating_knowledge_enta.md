---
title: >-
  [论文解读] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement
description: >-
  [NeurIPS 2025][模型压缩][联邦图学习] 首次提出联邦图基础模型(FedGFM)范式，融合联邦图学习的分布式协作能力与图基础模型的跨域泛化能力，通过 AncDAI（锚点域感知初始化）和 AdaDPP（自适应域敏感提示池）两个模块缓解知识纠缠问题，在8个跨任务跨领域数据集上超越20个基线。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "联邦图学习"
  - "图基础模型"
  - "知识纠缠"
  - "向量量化"
  - "提示学习"
---

# Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement

**会议**: NeurIPS 2025  
**arXiv**: [2505.12684](https://arxiv.org/abs/2505.12684)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 联邦图学习, 图基础模型, 知识纠缠, 向量量化, 提示学习

## 一句话总结

首次提出联邦图基础模型(FedGFM)范式，融合联邦图学习的分布式协作能力与图基础模型的跨域泛化能力，通过 AncDAI（锚点域感知初始化）和 AdaDPP（自适应域敏感提示池）两个模块缓解知识纠缠问题，在8个跨任务跨领域数据集上超越20个基线。

## 研究背景与动机

图机器学习面临两大范式各自的局限：

**联邦图学习(FGL)的限制**：

**数据异构性**：不同客户端的图在特征维度、标签空间和拓扑模式上差异巨大，多数FGL方法仅限于同一数据集的子集协作

**任务异构性**：现有FGL假设统一的图粒度和下游任务（节点级/子图级/图级），难以支持多任务协作

**图基础模型(GFM)的限制**：

**多领域数据孤岛**：训练GFM需要多领域图数据，但现实中数据分布在各机构，受隐私法规限制难以共享

**忽视跨机构资源**：集中式训练无法利用分布在多个机构的存储和计算资源

**互补关系**：FGL为GFM提供分布式训练范式，GFM为FGL提供统一的特征编码和预训练-微调框架。因此将二者结合是自然选择。

**知识纠缠挑战**：朴素地将gVQ-VAE的预训练分布到联邦设置中会导致多领域知识编码成不可区分的表示。实验验证（图2(b)）显示：GFT（集中式训练）产生的域间余弦相似度有明显差异，而GFT*（联邦训练）的域间相似度接近1，表示域特异性崩溃。

## 方法详解

### 整体框架

FedGFM+ 采用联邦预训练 + 微调的范式：
1. **联邦预训练**：各客户端在私有图上自监督学习，服务器聚合局部模型构建全局图基础模型
2. **微调**：全局模型作为GFM，通过监督学习适配特定下游任务
3. **双视角去纠缠**：AncDAI（全局）+ AdaDPP（局部）协同缓解知识纠缠

骨干网络选择 gVQ-VAE，因其：(1) 能联合编码图结构和文本属性为离散语义表示；(2) 参数量极小（如GFT仅7M），天然适合通信受限的联邦场景。

### 关键设计

#### 1. **AncDAI: 锚点域感知初始化（全局视角）**

核心思路：在预训练前，用各客户端的域原型（domain prototype）作为语义锚点初始化全局codebook，注入域区分的归纳偏置。

步骤：
1. 各客户端用全局初始化模型编码本地图：$\mathbf{Z}^k = f_{\theta^{glb}}(\mathbf{X}^k, \mathbf{A}^k)$
2. 均值池化得到域原型：$\mathbf{p}^k = \frac{1}{|\mathcal{V}^k|} \sum_{i \in \mathcal{V}^k} \mathbf{z}_i^k$
3. 理论保证：即使随机初始化共享参数，域原型在不同客户端间仍可区分（Theorem B.1）
4. 围绕每个锚点生成扰动嵌入：$\tilde{\mathbf{p}}_i^k = \mathbf{p}^k + \sigma \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(\mathbf{0}, \mathbf{1})$
5. 聚合所有域的合成嵌入初始化全局codebook

设计动机：让codebook从一开始就具有域感知的结构，为后续联邦训练中保持域间分离奠定基础。

#### 2. **AdaDPP: 自适应域敏感提示池（局部视角）**

**预训练阶段**：每个客户端独立学习一组域特定提示 $\Phi^k = \{\phi_i^k\}_{i=1}^\lambda$，不参与联邦聚合。节点特征增强为：

$$\tilde{x}_i^k = x_i^k + \sum_{j=1}^\lambda \alpha_j^k \phi_j^k, \quad \alpha_j^k = \frac{e^{(\mathbf{w}_j^k)^T x_i^k}}{\sum_{t=1}^\lambda e^{(\mathbf{w}_t^k)^T x_i^k}}$$

**微调阶段**：收集所有客户端的提示构建全局提示池 $\rho$。对目标图，通过注意力机制选择最相关的提示：

$$\tilde{x}_i^{tgt} = x_i^{tgt} + \sum_{p=1}^K \sum_{j=1}^\lambda \alpha_j^p \phi_j^p$$

设计动机：提示不参与联邦聚合避免信息混淆，微调时再组合利用各域知识，实现"先保留域特异性，后自适应迁移"。

### 损失函数 / 训练策略

预训练损失（自监督重建，Eq. 2）：
$$\mathcal{L}_{pretrain} = \mathcal{L}_{feat} + \mathcal{L}_{topo} + \text{codebook alignment} + \text{commitment loss}$$

- $\mathcal{L}_{feat}$: 节点特征重建（余弦相似度）
- $\mathcal{L}_{topo}$: 拓扑重建（邻接矩阵）
- 使用 straight-through estimator 保证端到端梯度

联邦聚合使用 FedAvg 策略：$\Theta^g \leftarrow \frac{N_k}{N} \sum_{k=1}^K \Theta^k$

## 实验关键数据

### 主实验

8个跨领域跨任务数据集上的性能对比：

| 方法 | Cora | PubMed | OGB-arxiv | WikiCS | FB15K | WN18RR | HIV | PCBA |
|---|---|---|---|---|---|---|---|---|
| GCN | 80.17 | 84.70 | 72.50 | 77.24 | 71.24 | 82.27 | 65.37 | 63.41 |
| FedAvg | 81.45 | 85.22 | 71.53 | 77.67 | 73.14 | 83.55 | 66.05 | 68.52 |
| GFT*(联邦变体) | 81.07 | 84.24 | 73.19 | 78.81 | 73.52 | 86.30 | 66.32 | 72.81 |
| GQT*(联邦变体) | 81.92 | 85.59 | 74.07 | 77.52 | 73.40 | 85.66 | 67.93 | 73.22 |
| **FedGFM+** | **83.79** | **88.52** | **76.31** | **80.70** | **75.25** | **89.25** | **69.39** | **77.68** |

相比最佳基线的提升：节点分类 ≥2.70%，边分类 ≥2.18%，图分类 ≥3.09%。

### 消融实验

| 配置 | Cora | PubMed | OGB-arxiv | HIV | PCBA | 说明 |
|---|---|---|---|---|---|---|
| w/o AncDAI | 81.55 | 85.56 | 75.19 | 67.52 | 74.81 | 去掉初始化，性能下降最多 |
| w/o AdaDPP | 83.17 | 87.42 | 75.83 | 67.84 | 76.72 | 去掉提示池 |
| **FedGFM+** | **83.79** | **88.52** | **76.31** | **69.39** | **77.68** | 完整方法 |

### 关键发现

1. **知识纠缠是核心瓶颈**：朴素联邦GFM变体（如GFT*）甚至会出现负迁移，性能低于孤立的监督模型
2. **AncDAI贡献更大**：去除AncDAI的性能下降大于去除AdaDPP，表明全局初始化对抗纠缠至关重要
3. **通信高效**：GFM参数量仅百万级（对比LLM的十亿级），联邦通信开销可接受
4. **超参数鲁棒**：codebook大小和提示数量在较大范围内性能稳定

## 亮点与洞察

1. **范式级贡献**：首次系统化地提出FedGFM范式，整合了FGL和GFM的互补优势
2. **问题定义精准**：通过实证分析（图2）清晰地识别了知识纠缠这一非平凡挑战
3. **双视角解决方案**：全局（初始化）+ 局部（提示）的双管齐下策略设计合理
4. **理论支撑**：证明了域原型在随机初始化下仍可区分（Theorem B.1），以及初始化策略提供结构化归纳偏置（Theorem B.2）

## 局限与展望

1. **隐私风险**：原型和提示的交换可能暴露部分语义信息，需要正式的隐私分析
2. 当前实验中每个数据集分给3个客户端，真实场景的客户端数量和异构性可能更大
3. 可考虑引入差分隐私（DP）或安全计算保护原型和提示的传输
4. 仅验证了gVQ-VAE骨干，能否扩展到更大规模的GFM架构值得探索

## 相关工作与启发

- 将NLP/CV领域的foundation model联邦训练思路迁移到图领域，但图的异构性更强
- 提示学习不参与联邦聚合的设计类似于个性化联邦学习中的local fine-tuning思路
- AncDAI的域原型概念可推广到其他需要处理域异构性的联邦学习场景
- codebook初始化策略对VQ-VAE的联邦训练有通用价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （FedGFM范式是首次提出，问题定义和方法设计都有新意）
- 实验充分度: ⭐⭐⭐⭐⭐ （8数据集、20基线、3大任务类型、消融+超参数分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但方法涉及联邦+GFM+VQ-VAE+提示学习多个概念，读起来信息量大）
- 价值: ⭐⭐⭐⭐ （开辟了有前景的研究方向，但离真实部署还有隐私和规模化的挑战）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation](ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](graph_your_own_prompt.md)
- [\[ICML 2026\] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](../../ICML2026/model_compression/fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)
- [\[NeurIPS 2025\] Explaining and Mitigating Crosslingual Tokenizer Inequities](explaining_and_mitigating_crosslingual_tokenizer_inequities.md)

</div>

<!-- RELATED:END -->
