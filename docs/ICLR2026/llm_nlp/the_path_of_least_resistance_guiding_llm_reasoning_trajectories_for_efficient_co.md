---
title: >-
  [论文解读] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency
description: >-
  [ICLR 2026][LLM/NLP][self-consistency] 提出 PoLR（Path of Least Resistance），首个利用推理前缀一致性的推理时方法，通过聚类短前缀并仅扩展主导聚类来实现 Self-Consistency 的高效替代…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "self-consistency"
  - "inference efficiency"
  - "prefix clustering"
  - "reasoning"
  - "token reduction"
---

# The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency

**会议**: ICLR 2026  
**arXiv**: [2601.21494](https://arxiv.org/abs/2601.21494)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: self-consistency, inference efficiency, prefix clustering, reasoning, token reduction

## 一句话总结

提出 PoLR（Path of Least Resistance），首个利用推理前缀一致性的推理时方法，通过聚类短前缀并仅扩展主导聚类来实现 Self-Consistency 的高效替代，可减少高达 60% token 使用和 50% 延迟。

## 研究背景与动机

Self-Consistency (SC) 解码通过采样多条推理轨迹并多数投票选择最终答案，大幅提升 LLM 推理准确率，但计算开销巨大——每条推理轨迹必须完整展开。现有改进方法如 Adaptive Consistency (AC) 和 Early-Stopping SC (ESC) 通过在最终答案达成一致时提前停止，但它们共享一个根本限制：**答案级别的一致性只有在完整推理轨迹生成后才可观察**，无法利用推理过程早期阶段的丰富结构信息。

PoLR 的核心观察是：**推理轨迹的前缀（前几个步骤）已经蕴含了关于最终解答的强信号**，即"前缀一致性"现象。共享相同前缀的推理轨迹几乎达到与完整 SC 相同的准确率，这意味着花在额外轨迹上的大量 token 开销很少对最终答案产生贡献。

## 方法详解

### 整体框架

PoLR 修改标准 SC 管线，引入基于前缀的选择步骤：

1. **前缀采样**：给定输入问题 $x$ 和模型 $\mathcal{M}$，生成 $N$ 个短推理前缀 $p_i = \text{Prefix}(\mathcal{M}(x, t_i), L_p)$（通过设置 `max_new_tokens = L_p` 实现）
2. **嵌入与聚类**：对每个前缀使用 TF-IDF 词袋编码生成稀疏向量表示，然后使用层次聚类（余弦相似度）将前缀聚类为 $\mathcal{C} = \{C_1, \dots, C_m\}$，选择主导聚类 $C^* = \arg\max_{C_j}|C_j|$
3. **扩展**：仅扩展主导聚类中的 $K$ 个前缀为完整推理轨迹 $r_k = \mathcal{M}(x | p_k)$
4. **投票**：$\hat{a} = \arg\max_y \sum_{k=1}^K \mathbf{1}[a_k = y]$

### 关键设计

**Token 效率公式**：

$$\eta = 1 - \frac{T_{\text{PoLR}}}{T_{\text{SC}}} = 1 - \frac{N \cdot \ell_p + K \cdot (\ell_f - \ell_p)}{N \cdot \ell_f}$$

其中 $\ell_p$ 为平均前缀长度，$\ell_f$ 为完整推理长度。

**设计选择的理由**：
- **TF-IDF 而非神经编码器**：轻量级、模型无关、CPU 友好，神经编码器增加的聚类开销远大于 TF-IDF，但准确率收益甚微
- **层次聚类**：适合小 $N$（11–51），无需预设聚类数，产生可解释的分组
- **$L_p = 256$**：经验上在准确率和 token 效率间取得良好平衡

### 理论分析

PoLR 的正确性和效率由两个互补性质保证：

**正确性对齐**：令 $Y \in \{0,1\}$ 为最终推理轨迹的正确性，$Z$ 为前缀的聚类分配。关键条件是 $I(Z;Y) > 0$，即聚类至少弱预测正确性。如果 $H(Y|Z)$ 小，聚类身份可靠预测正确性。

**结构偏斜与效率**：效率不由正确性对齐驱动，而由前缀聚类分布的结构偏斜决定。定义偏斜率 $\kappa = |C^*|/N$。

**命题**：PoLR 相对于 SC 的 token 效率增益满足 $\eta \geq 1 - \frac{K}{M} \cdot \kappa^{-1}$，效率与 $\kappa$ 单调递增。

核心洞察：**互信息保证安全性（不丢失准确率），而偏斜决定节省量**。NMI 保持低值（$\leq 0.18$），但效率饱和在 50–58%，因为前缀聚类表现出强结构偏斜。

### 损失函数

PoLR 为推理时方法，不涉及训练，无损失函数。其核心优化目标为最小化 token 消耗同时保持 SC 准确率。

## 实验关键数据

### 主实验

在 GSM8K、Math500、AIME24/25、GPQA-Diamond 上跨多个 LLM 家族评估：

| 模型 | 数据集 | N | SC Acc | PoLR Δ | η (%) | 开销 kt (ms) |
|------|--------|---|--------|--------|-------|-------------|
| QWQ32B | GSM8K | 51 | 90.8% | -0.3 | 47.6 | 11.2 |
| DSQ7B | Math500 | 31 | 89.6% | +0.1 | 48.5 | 5.1 |
| QWQ32B | GPQA-D | 51 | 68.7% | +1.5 | 53.8 | 11.2 |
| DSQ7B | AIME25 | 31 | 33.7% | +2.7 | - | - |
| Phi-4-15B | AIME25 | 31 | 32.0% | +4.0 | - | - |
| QWQ32B | Math500 | 51 | 91.8% | +0.2 | 51.8 | 11.2 |

**核心发现**：
- token 效率 η 通常在 40–60%，有效将 token 消耗减半
- 聚类开销 kt 仅几毫秒，节省直接转化为更快推理
- 准确率保持甚至偶尔提升，因为 PoLR 强调主导一致推理聚类，过滤噪声轨迹
- AIME25 上 QWQ32B 下降 10 点为特例（仅 30 个样本中的 3 个）

### 消融实验

初步分析（Math500、GSM8K，DSQ7B，40 样本）验证前缀一致性：

| 数据集 | $L_p$ | 扩展率 | 准确率 | 精确前缀匹配 |
|--------|-------|--------|--------|-------------|
| Math500 | SC | 1.00 | 89.8 | - |
| Math500 | 32 | 0.64 | 89.8 | 125 |
| Math500 | 128 | 0.48 | 89.2 | 5 |
| GSM8K | SC | 1.00 | 79.7 | - |
| GSM8K | 32 | 0.52 | 79.7 | 135 |
| GSM8K | 128 | 0.47 | 79.3 | 30 |

### 关键发现

1. PoLR 对不同聚类方法、前缀长度、聚类选择策略均表现鲁棒
2. PoLR 与自适应推理方法（AC、ESC）完全互补，可作为前置过滤器
3. 跨模型家族和规模（1.5B–32B）一致有效
4. 在非数学任务（StrategyQA）上同样表现出一致增益

## 亮点与洞察

1. **"少即是多"的推理效率范式**：通过前缀聚类发现，LLM 在推理早期就编码了结构一致性，后续大部分计算是冗余的
2. **理论与实践的优雅统一**：正确性对齐（互信息）保证安全、结构偏斜（κ）驱动效率的分离分析非常清晰
3. **零训练开销**：TF-IDF + 层次聚类的轻量组合，使方法成为真正的即插即用替代品
4. **互补性设计**：明确定位为 SC 的前置优化，可与 AC、ESC 等方法叠加使用

## 局限性

1. **AIME25 上 QWQ32B 下降 10 点**：在具有挑战性且样本极少的基准上存在波动风险
2. **前缀长度 $L_p$ 需要手动设定**：虽然 256 在多数情况下有效，但自适应确定最优前缀长度仍是开放问题
3. **依赖前缀结构偏斜**：如果问题的推理路径高度多样化（$\kappa \approx 1/m$），PoLR 的效率增益会减小
4. **仅测试开源模型**：未在 GPT-4 等闭源模型上验证

## 相关工作与启发

- **Self-Consistency** (Wang et al., 2023)：PoLR 的直接基线
- **Adaptive Consistency** (Aggarwal et al., 2023)：按需停止生成，但仍依赖完整轨迹
- **Early-Stopping SC** (Li et al., 2024)：类似限制
- **前缀一致性** (Ji et al., 2025)：训练时利用前缀，需要微调

PoLR 的核心启发是：**推理效率优化的关键时机不在结束（何时停止），而在开始（何时分线）**。这一洞察可能启发其他利用推理前缀信号的方法。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个推理时利用前缀一致性替代 SC 的方法，概念新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 跨 5 个基准、6 个模型、多种配置的全面评估，10 次重复
- 写作质量: ⭐⭐⭐⭐ — 理论与实验结合紧密，结构清晰
- 价值: ⭐⭐⭐⭐ — 对高效推理有实际意义，即插即用的推理加速方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](../../ICML2026/llm_nlp/reasoning_on_the_manifold_bidirectional_consistency_for_self-verification_in_dif.md)
- [\[ICLR 2026\] KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](../../ACL2025/llm_nlp/dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)

</div>

<!-- RELATED:END -->
