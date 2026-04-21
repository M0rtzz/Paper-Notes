---
title: >-
  [论文解读] Catastrophic Forgetting in Kolmogorov-Arnold Networks
description: >-
  [AAAI 2026][KAN] 首个系统性研究KAN（Kolmogorov-Arnold Networks）中灾难性遗忘行为的工作：建立了遗忘与激活支持重叠和数据内禀维度之间的理论框架，并提出KAN-LoRA用于语言模型的持续微调知识编辑。
tags:
  - AAAI 2026
  - KAN
  - 灾难性遗忘
  - 持续学习
  - KAN-LoRA
  - 激活支持重叠
---

# Catastrophic Forgetting in Kolmogorov-Arnold Networks

**会议**: AAAI 2026  
**arXiv**: [2511.12828](https://arxiv.org/abs/2511.12828)  
**代码**: [有](https://github.com/marufur-cs/AAAI26)  
**领域**: 模型压缩  
**关键词**: KAN, 灾难性遗忘, 持续学习, KAN-LoRA, 激活支持重叠

## 一句话总结

首个系统性研究KAN（Kolmogorov-Arnold Networks）中灾难性遗忘行为的工作：建立了遗忘与激活支持重叠和数据内禀维度之间的理论框架，并提出KAN-LoRA用于语言模型的持续微调知识编辑。

## 研究背景与动机

灾难性遗忘是持续学习中的核心挑战。MLP由于全局更新的参数化特点天然易受遗忘影响，已有大量缓解策略（正则化、架构修改、经验回放等）。KAN作为新兴替代架构，使用沿网络边的可学习一维激活函数（B-spline），其**局部性**使得数据样本仅影响少量相关的spline系数。

然而，KAN在持续学习中的遗忘行为尚不清楚：
- 先前工作（Liu et al.）仅在简单合成回归任务上展示了KAN的鲁棒性
- 缺乏**理论框架**将KAN的架构局部性与遗忘行为联系起来
- 在高维实际任务（图像分类、语言建模）上的表现未知

本文的目标：从理论和实验两个维度全面揭示KAN的遗忘特性和局限。

## 方法详解

### 整体框架

```
理论分析层：
  ├─ 遗忘度量定义 F_i = L(f^(T), D_i) - L(f^(i), D_i)
  ├─ 激活支持重叠 Δ_{i,j} = max_{l,p,q} μ(S^(i) ∩ S^(j))
  └─ 内禀维度 d_t（任务数据流形维度）

定理体系：
  ├─ Lemma 1: 零重叠→零遗忘
  ├─ Theorem 1: 遗忘上界 ∝ 重叠 × Lipschitz常数
  ├─ Theorem 2: 逐分支累积遗忘分解
  ├─ Theorem 3: 内禀维度遗忘率 O(r^{d_i+d_j})
  └─ Corollary 1-4: 随机支持、饱和、低维保持、碎片化

实验验证层：
  ├─ 二进制/十进制加法（低维合成）
  ├─ 图像分类（MNIST, CIFAR-10, Tiny-ImageNet）
  └─ KAN-LoRA 知识编辑（LLM持续微调）
```

### 关键设计

**1. 激活支持重叠理论（Activation Support Overlap）**

定义分支 $\phi_{\ell,p,q}$ 在任务 $i$ 上的激活支持：

$$S^{(i)}_{\ell,p,q} = \{z \in \mathbb{R} : \phi_{\ell,p,q}(z) \neq 0\}$$

两任务间最大一维重叠：

$$\Delta_{i,j} = \max_{\ell,p,q} \mu(S^{(i)}_{\ell,p,q} \cap S^{(j)}_{\ell,p,q})$$

**Lemma 1（零重叠保持）**：若 $\Delta_{i,j}=0$ 对所有 $j>i$，则 $F_i=0$（完美保持）。

**Theorem 1（重叠遗忘界）**：在Lipschitz和有界损失假设下：

$$F_i \leq C \sum_{\ell=1}^{L} N_\ell L_\ell \Delta_{i,j}$$

核心洞察：**遗忘与激活支持重叠线性相关**。

**2. 累积遗忘分析**

**Theorem 2（逐分支累积遗忘）**：

$$F_i \leq C \sum_{\ell} \sum_{p} \sum_{q} L_\ell \left[\sum_{j=i+1}^{T} \mu(S^{(i)}_{\ell,p,q} \cap S^{(j)}_{\ell,p,q})\right]$$

**Corollary 1（随机支持期望遗忘）**：若各分支支持为 $[0,1]$ 中的长度 $s_j$ 随机区间，则 $\mathbb{E}[F_i] \leq C \sum_\ell N_\ell L_\ell \sum_{j} s_i s_j$。

**Corollary 2（饱和效应）**：通过并集上界，重叠的累积效果有上限（不会无限增长），与实验中观察到的饱和现象一致。

**3. 复杂度驱动遗忘**

**Theorem 3（内禀维度遗忘率）**：当任务数据集中在内禀维度 $d_t$ 的紧子流形上时：

$$F_i = O\left(\sum_{j=i+1}^{T} N_{tot} \bar{L} \cdot r^{d_i + d_j}\right)$$

关键结论：**遗忘随内禀维度指数增长**。这解释了为什么KAN在低维任务上表现优秀但在高维任务上仍然脆弱。

**Corollary 4（碎片化缓解）**：将支持拆分为 $k_t$ 个不相交区间（有效半径 $r/k_t$），可改善遗忘率。

**4. KAN-LoRA 适配器**

为将理论洞察应用于实际LLM持续学习场景，设计了基于KAN的LoRA适配器：
- 将标准MLP-LoRA中的低秩矩阵替换为KAN结构
- 利用KAN的局部激活特性，期望在顺序知识编辑中比MLP-LoRA更好保持先前知识
- 在Llama 2 (7B, 13B) 上进行评估

### 损失函数 / 训练策略

- 合成任务：MSE损失，SGD优化
- 图像分类：交叉熵损失，KAN-Transformer架构（所有MLP层替换为KAN层）
- 知识编辑：在CounterFact和ZsRE基准上顺序编辑，评估编辑accuracy在后续任务后的保持率

## 实验关键数据

### 主实验

**表1：十进制加法任务间遗忘与重叠比例（验证Theorem 1）**

| Task(i) | Task(j) | Grid 10: $F_i/\Delta_{i,j}$ | Grid 15 | Grid 20 |
|---------|---------|------|------|------|
| 1 | 2 | 0.74 | 0.74 | 0.61 |
| 2 | 3 | 0.73 | 0.67 | 0.64 |
| 3 | 4 | 0.77 | 0.74 | 0.63 |
| 4 | 5 | 0.72 | 0.68 | 0.64 |

$F_i/\Delta_{i,j}$ 近似恒定，验证遗忘与重叠的**线性关系**。

**表2：累积遗忘与累积重叠比例（验证Theorem 2）**

| Task(i) | Tasks(j) | Grid 10: $F_i/\sum\mu^{ij}$ | Grid 15 | Grid 20 |
|---------|----------|------|------|------|
| 1 | 2,3,4,5 | 0.15 | 0.15 | 0.16 |
| 2 | 3,4,5 | 0.16 | 0.15 | 0.16 |
| 3 | 4,5 | 0.16 | 0.16 | 0.16 |
| 4 | 5 | 0.18 | 0.17 | 0.17 |

比值更稳定，且Grid越大方差越低，验证spline精度对遗忘一致性的影响。

**表3：内禀维度与遗忘率（验证Theorem 3）**

| 数据集 | $\log(F_i)/d_i$ 范围 |
|-------|---------------------|
| MNIST | 0.071-0.075 |
| CIFAR-10 | 0.046-0.053 |
| Tiny-ImageNet | 0.047-0.054 |

$\log(F_i)/d_i$ 近似恒定，验证遗忘与内禀维度的**指数关系**。

**表4：KAN-LoRA vs MLP-LoRA 知识编辑（Llama 2-7B, rank 8）**

| 数据集 | #Tasks | KAN-LoRA Acc | MLP-LoRA Acc |
|-------|--------|-------------|-------------|
| CounterFact | 2 | 100 | 100 |
| CounterFact | 3 | 65 | 90 |
| CounterFact | 4 | 50 | 65 |
| CounterFact | 5 | 45 | 57 |
| ZsRE | 2 | 100 | 100 |
| ZsRE | 3 | 80 | 95 |
| ZsRE | 5 | 60 | 87 |

高维LLM场景下KAN-LoRA反而不如MLP-LoRA，验证了理论预测。

### 消融实验

- **Grid Size 影响**：Grid从5→20，十进制加法遗忘持续降低（更细的spline减小支持重叠）
- **模型深度影响**：encoder blocks从1→多，遗忘急剧加重（更深网络更脆弱）
- **样本数影响**：训练样本增加，遗忘加重（更多样本增大激活支持范围）
- **KAN vs MLP-EWC**：CIFAR-10上KAN在少任务时优于MLP-EWC，但Tiny-ImageNet上MLP-EWC更优

### 关键发现

1. **KAN在低维结构化任务上天然抗遗忘**：二进制加法中遗忘<$10^{-6}$，甚至优于专门设计的MLP
2. **高维任务是KAN的软肋**：图像分类和语言建模中KAN仍严重遗忘，与理论预测一致
3. **Grid Size是关键超参数**：更大grid通过减小spline支持长度来降低重叠，直接缓解遗忘
4. **遗忘饱和效应**存在：高度重叠的任务序列中，累积遗忘趋于饱和
5. **KAN-LoRA在LLM场景效果有限**：高维度使KAN的局部性优势失效

## 亮点与洞察

- **理论框架的完备性**：从零重叠到累积遗忘到复杂度驱动，层层递进的4个定理+4个推论构成完整理论体系
- **理论-实验高度吻合**：每个定理都有对应的实验表格验证，比值恒定性令人信服
- **诚实的负面结果**：KAN-LoRA在高维场景的失败并非回避而是正面呈现，增强了工作可信度
- **实用指导**：Grid Size ↑ → 遗忘 ↓ 的量化关系为KAN在持续学习中的使用提供了直接调参指导

## 局限与展望

1. KAN-LoRA设计相对直接，未探索更复杂的适配策略（如task-specific spline routing）
2. 理论假设中的Lipschitz和有界损失条件在实际网络中可能不严格成立
3. 碎片化缓解策略（Corollary 4）的实际实现方案未给出
4. 缺少与主流持续学习方法（如EWC on KAN、Memory replay on KAN）的对比
5. 可考虑将grid size自适应嵌入持续学习流程中（动态grid扩展策略）

## 相关工作与启发

- **KAN (Liu et al. 2025)**：原始KAN论文暗示其抗遗忘特性，本文给出了严格条件
- **EWC (Kirkpatrick et al. 2017)**：经典正则化持续学习方法，作为MLP的增强基线
- **LoRA (Hu et al. 2022)**：参数高效微调，本文将其扩展为KAN-based版本
- **对模型压缩的启发**：KAN的局部性在低维场景下的优势提示——压缩后的小模型若内禀维度更低，可能反而更抗遗忘

## 评分

- 新颖性: ⭐⭐⭐⭐ (首个KAN遗忘理论框架 + KAN-LoRA)
- 实验充分度: ⭐⭐⭐⭐ (合成+视觉+语言三层验证)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，实验对应明确)
- 价值: ⭐⭐⭐⭐ (对KAN在持续学习中的能力边界给出了清晰回答)

<!-- RELATED:START -->

## 相关论文

- [When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)
- [Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)
- [Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning](is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform.md)
- [Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](multiplicative_orthogonal_sequential_editing_for_language_models.md)
- [Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)

<!-- RELATED:END -->
