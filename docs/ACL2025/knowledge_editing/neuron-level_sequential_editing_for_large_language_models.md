---
title: >-
  [论文解读] Neuron-Level Sequential Editing for Large Language Models
description: >-
  [ACL2025][知识编辑][NSE] 提出NSE方法用于LLM的序列化模型编辑，通过权重回退（weights rewinding）防止模型崩溃、基于激活值的神经元级选择性权重更新缓解模型遗忘、以及迭代多层编辑提高大规模知识更新的成功率。
tags:
  - "ACL2025"
  - "知识编辑"
  - "NSE"
  - "Sequential Model Editing"
  - "Neuron Selection"
  - "Weights Rewinding"
  - "Iterative Multi-Layer Editing"
---

# Neuron-Level Sequential Editing for Large Language Models

**会议**: ACL2025  
**arXiv**: [2410.04045](https://arxiv.org/abs/2410.04045)  
**代码**: [GitHub](https://github.com/jianghoucheng/NSE)  
**领域**: 模型编辑 / 知识更新 / LLM  
**关键词**: NSE, Sequential Model Editing, Neuron Selection, Weights Rewinding, Iterative Multi-Layer Editing  

## 一句话总结

提出NSE方法用于LLM的序列化模型编辑，通过权重回退（weights rewinding）防止模型崩溃、基于激活值的神经元级选择性权重更新缓解模型遗忘、以及迭代多层编辑提高大规模知识更新的成功率。

## 研究背景与动机

### 问题背景
LLM在预训练中存储了大量事实知识，但真实世界知识会不断演变，导致模型内部信息过时或错误。完全重训代价极高，因此模型编辑（修改特定知识而不重训）成为研究热点。

### 序列化编辑的挑战
现有模型编辑方法（如ROME、MEMIT）在单轮编辑中效果良好，但实际应用需要连续多轮编辑：

**模型遗忘（Model Forgetting）**：累积的参数修改导致模型忘记先前编辑的知识

**模型崩溃（Model Failure）**：过多编辑损害模型生成连贯文本的能力，甚至产生无意义输出

### 为什么现有方法失败
- **ROME/MEMIT**：直接修改FFN层权重，参数变化随编辑轮次累积，最终导致灾难性退化
- **记忆方法（GRACE）**：保留参数不变但存储增量，随编辑数增长存储需求剧增
- 核心矛盾：需要修改足够的参数来编辑知识，但修改过多会破坏模型功能

## 方法详解

### 预备知识

#### 自回归语言模型
L层Transformer中，第l层隐藏状态：
$$\mathbf{h}_t^l = \mathbf{h}_t^{l-1} + \mathbf{a}_t^l + \mathbf{v}_t^l$$

FFN层输出 $\mathbf{v}_t^l = \mathbf{W}_{\text{out}}^l \sigma(\mathbf{W}_{\text{in}}^l \gamma(\mathbf{h}_t^{l-1} + \mathbf{a}_t^l))$

#### 知识存储视角
FFN层的线性操作可视为键值存储：
- **Key** $\mathbf{k}_i^l$：输入subject最后一个token在第l层FFN的激活输出
- **Value** $\mathbf{v}_i^l$：经 $\mathbf{W}_{\text{out}}^l$ 处理后的输出

编辑目标：引入新键值对 $(\mathbf{K}_1, \mathbf{V}_1)$ 同时保留旧键值对 $(\mathbf{K}_0, \mathbf{V}_0)$：
$$\Delta^* = \arg\min_\Delta \left(\|(\mathbf{W}+\Delta)\mathbf{K}_1 - \mathbf{V}_1\|^2 + \|(\mathbf{W}+\Delta)\mathbf{K}_0 - \mathbf{V}_0\|^2\right)$$

### NSE方法

#### 1. 权重回退的值计算（Weights Rewinding for Value Computation）

**问题**：在序列编辑中，使用已更新的模型参数 $f_{\theta_t}$ 计算目标值 $\mathbf{z}_i$ 会导致严重退化。累积的参数变化使值计算偏移。

**解决方案**：始终使用原始模型参数 $f_{\theta_0}$ 计算目标值：
$$\mathbf{z}_i = \mathbf{h}_i^l + \arg\min_{\delta_i}\left(-\log \mathbb{P}_{f_{\theta_0}(\mathbf{h}_i^l += \delta_i)}[o_i|(s_i, r_i)]\right)$$

**实现细节**：
- 只需保存需要更新的 $\mathbf{W}_{\text{out}}$ 权重矩阵，无需存储整个模型
- 仅在计算 $\mathbf{z}_i$ 时使用原始权重
- 通过梯度下降优化 $\delta_i$，最大化模型输出目标 $o_i$ 的概率

#### 2. 神经元级权重更新（Neuron-Level Weights Updating）

**核心思想**：不修改整个权重矩阵，而是选择性地修改一小部分"有影响力的神经元"。

**神经元选择**：基于激活值 $\mathbf{k}_i$ 计算分数 $\mathbf{Q}_i = |\mathbf{k}_i|$，选择累积分数达到总分数预设百分比 $p$ 的最小神经元子集 $\mathcal{I}$：
$$\mathcal{I} = \arg\min_{\mathcal{I} \subseteq \{1,...,N\}} |\mathcal{I}| \quad \text{s.t.} \quad \sum_{j \in \mathcal{I}} \mathbf{Q}_{ij} \geq p \times \sum_{j=1}^N \mathbf{Q}_{ij}$$

**批量编辑**：对于batch中多个知识事实，将各样本的神经元分数求和后统一选择。

**子矩阵更新**：只更新选中神经元对应的子矩阵：
$$\hat{\Delta}^* = \hat{\mathbf{R}} \hat{\mathbf{K}}_1^T \hat{\mathbf{C}}^{-1}$$

其中 $\hat{\mathbf{R}} = \mathbf{V}_1 - \hat{\mathbf{W}} \hat{\mathbf{K}}_1$, $\hat{\mathbf{C}} = \hat{\mathbf{K}}_0\hat{\mathbf{K}}_0^T + \hat{\mathbf{K}}_1\hat{\mathbf{K}}_1^T$。

**知识累积**：每轮编辑后，新编辑的知识加入 $\mathbf{K}_0\mathbf{K}_0^T$，成为下一轮的"旧知识"。

#### 3. 迭代多层编辑（Iterative Multi-Layer Editing）

**MEMIT的局限**：将残差 $\delta_i$ 均分到多层（$\mathbf{v}_i^l += \frac{\delta_i}{l_0 - l + 1}$），但部分知识难以编辑，拟合误差导致失败。

**迭代策略**：
1. 执行一轮多层编辑
2. 检查每个样本的残差 $\|\mathbf{z}_i - \mathbf{h}_i^{l_0}\|^2$
3. 若 $< \alpha$：编辑成功；若 $> \alpha$：未成功
4. 未成功的样本组成新batch，重复多层编辑
5. 直到所有样本成功或达到迭代上限

## 实验

### 实验设置
- **模型**：GPT2-XL (1.5B), GPT-J (6B), Llama3 (8B)
- **数据集**：Counterfact, ZsRE（各2000条编辑）
- **Batch size**：100（每轮同时编辑100条知识）
- **评估指标**：Efficacy（编辑成功率）, Generalization（泛化性）, Specificity（特异性）, Fluency（流畅度）, Consistency（一致性）
- **基线**：FT-L, FT-W, MEND, ROME, MEMIT, GRACE

### 主要结果（Llama3, Counterfact）

| 方法 | Efficacy↑ | Generalization↑ | Specificity↑ | Fluency↑ | Consistency↑ |
|------|-----------|----------------|-------------|---------|-------------|
| MEMIT | 65.65 | 64.65 | 51.56 | 437.43 | 6.58 |
| ROME | 64.40 | 61.42 | 49.44 | 449.06 | 3.31 |
| GRACE | 90.72 | 0.09 | 87.23 | 632.43 | 23.79 |
| **NSE** | **96.14** | **78.42** | **87.66** | **632.72** | **30.20** |

关键发现：

1. **全面领先**：NSE在几乎所有指标上超越所有基线，Llama3上平均提升约30.33%
2. **生成能力保持**：Fluency (632.72) 和 Consistency (30.20) 显著优于参数修改方法，提升超40.75%
3. **GRACE的致命弱点**：Generalization极低（0.09），因为它不修改参数，换个问法就失效
4. **MEMIT/ROME的退化**：随编辑轮次增加，性能急剧下降（尤其Specificity、Fluency）

### 不同模型结果

| 模型 | NSE Efficacy | NSE Specificity | MEMIT Efficacy | MEMIT Specificity |
|------|-------------|-----------------|----------------|-------------------|
| GPT2-XL | 96.80 | 72.10 | 94.70 | 60.50 |
| GPT-J | 99.55 | 78.96 | 98.55 | 63.64 |
| Llama3 | 96.14 | 87.66 | 65.65 | 51.56 |

NSE在所有模型上均保持优异表现，特别是在Llama3上优势最明显。

### Batch Size影响
- MEMIT在batch size=10时性能急剧下降（更多编辑轮次）
- NSE在各种batch size下保持稳定，batch size=10时平均提升45.60%

### 消融实验（Llama3, Counterfact）

| 变体 | Eff. | Gen. | Spe. | Flu. | Consis. |
|------|------|------|------|------|---------|
| NSE（完整） | 96.14 | 78.42 | 87.66 | 632.72 | 30.20 |
| w/o weights rewinding | 98.90↑ | 91.18↑ | **76.60↓↓** | **625.65↓** | 32.30↑ |
| w/o neuron update | 96.00 | 77.13↓ | 87.68 | - | - |

关键发现：
- **权重回退至关重要**：去除后Specificity下降11.06，说明使用更新后权重计算值会偏移
- Efficacy和Generalization虽然上升，但以牺牲Specificity和Fluency为代价（过拟合到新知识）
- **神经元选择贡献稳定**：去除后各指标小幅下降

### 通用能力测试
在MMLU、CMMLU等通用benchmark上测试编辑后的模型，NSE编辑后的模型保持原有通用能力，而ROME/MEMIT编辑后通用能力显著下降。

## 亮点与洞察

1. **权重回退的核心洞察**：序列编辑中累积参数变化会污染值计算，使用原始权重作为"锚点"是解决模型崩溃的关键
2. **神经元选择的优雅性**：基于激活值的稀疏选择（通常只修改少量神经元）实现了精确编辑与保护模型功能的平衡
3. **迭代编辑的自适应性**：通过残差阈值自动适应难易程度不同的知识编辑
4. **三个组件的协同**：权重回退防崩溃 + 神经元选择防遗忘 + 迭代编辑提成功率，环环相扣

## 局限性

1. 需要保存原始权重矩阵 $\mathbf{W}_{\text{out}}$，增加了存储开销
2. 迭代多层编辑增加了计算时间
3. 超参数（选择百分比 $p$、残差阈值 $\alpha$）需要针对不同模型调整
4. 仅在事实知识编辑（subject-relation-object）上验证，未扩展到更复杂的编辑场景（如行为修改）
5. 随着编辑数量持续增长，长期效果仍需更大规模验证

## 相关工作

- **知识定位**：因果追踪识别关键层（Meng et al., 2022），FFN作为知识存储
- **参数修改方法**：ROME（单层编辑）、MEMIT（多层编辑）、Fine-tuning
- **参数保留方法**：GRACE（记忆方法）、MEND（超网络）
- **持续学习**：灾难性遗忘、弹性权重巩固（EWC）
- **神经元分析**：知识神经元、FFN中的信息存储

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 综合评价 | ⭐⭐⭐⭐ |

> NSE是一篇扎实的方法论文，三个技术组件各自解决序列编辑中的一个核心问题（崩溃、遗忘、失败），且设计简洁、易于理解和复现。实验在3个模型×2个数据集上全面验证，消融分析清晰。权重回退的idea虽然看似简单，但其背后的insight（累积更新对值计算的污染效应）深刻而关键。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](../../ICML2026/knowledge_editing/the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ACL 2025\] DocMEdit: Towards Document-Level Model Editing](docmedit_towards_document-level_model_editing.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)

</div>

<!-- RELATED:END -->
