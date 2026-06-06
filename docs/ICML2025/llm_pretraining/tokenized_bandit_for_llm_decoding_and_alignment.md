---
title: >-
  [论文解读] Tokenized Bandit for LLM Decoding and Alignment
description: >-
  [ICML 2025][预训练][多臂老虎机] 将 LLM 解码与对齐问题形式化为 **tokenized bandit**（token化老虎机）问题，提出 DDMC（Diminishing Distance with More Commons）假设，证明在该假设下贪心解码近似最优…
tags:
  - "ICML 2025"
  - "预训练"
  - "多臂老虎机"
  - "LLM解码"
  - "解码时对齐"
  - "贪心解码"
  - "序列函数优化"
---

# Tokenized Bandit for LLM Decoding and Alignment

**会议**: ICML 2025  
**arXiv**: [2506.07276](https://arxiv.org/abs/2506.07276)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 多臂老虎机, LLM解码, 解码时对齐, 贪心解码, 序列函数优化

## 一句话总结

将 LLM 解码与对齐问题形式化为 **tokenized bandit**（token化老虎机）问题，提出 DDMC（Diminishing Distance with More Commons）假设，证明在该假设下贪心解码近似最优，并设计了具有次线性遗憾的在线学习算法 EOFUL 和 GreedyETC。

## 研究背景与动机

LLM 对齐（alignment）是当前大模型研究的核心挑战之一。主流方法 RLHF 通过微调实现对齐，但存在计算代价高、需要大量人类标注、模型需频繁更新等问题，难以扩展到个性化场景。

近年来 **解码时对齐**（decoding-time alignment）方法逐渐兴起，旨在推理阶段在线调整 LLM 输出以匹配用户偏好，无需微调模型。然而这类方法的理论基础仍十分薄弱，缺乏对样本效率的严格分析。

更基础的问题是：即使是最简单的解码算法（如贪心解码、beam search），也缺少从理论角度的深入理解——为什么贪心解码在很多任务上表现出色？

本文从全新视角出发，将 LLM 解码和对齐问题建模为 **线性/多臂老虎机的 token 化变体**，决策者需要逐 token 不可撤回地构建序列，最终从用户获得效用反馈。

## 方法详解

### 整体框架

本文提出两个核心问题变体：

1. **Tokenized Linear Bandit (TLB)**：效用函数具有线性参数化结构 $u_t(x_t, \mathbf{y}) = \langle \theta, e(x_t, \mathbf{y}) \rangle$，其中 $e(\cdot)$ 是嵌入函数，$\theta$ 是待学习的隐含参数。每轮用户提交不同查询（上下文），决策者需要在线学习 $\theta$。

2. **Tokenized Multi-Armed Bandit (TMAB)**：效用函数可以是任意的（无线性结构），但上下文固定不变 $x_t = x$。问题退化为在固定上下文下学习最优 token 序列。

**统一问题设定**：每轮 $t \in [T]$，用户提交查询 $x_t$，决策者逐 token 从词表 $\mathcal{V}$（$|\mathcal{V}|=n$）中不可撤回地选择 token 构造序列 $\mathbf{y}_t$，序列完成后观测带噪声的奖励 $r_t(\mathbf{y}_t) = u_t(\mathbf{y}_t) + \eta_t$。目标是最小化累积伪遗憾：

$$\text{Reg} = \sum_{t=1}^{T} \left[ \max_{\mathbf{z} \in \mathcal{V}^*} u_t(\mathbf{z}) - u_t(\mathbf{y}_t) \right]$$

### 关键设计

#### 1. 基础不可能性结果

作者首先证明了两个下界：

- **TLB**：若序列函数无结构假设，任何算法的最坏情况遗憾为 $\Omega(T(1-1/2^{L-2}))$，随序列长度 $L$ 指数级增长
- **TMAB**：无结构假设下，最坏情况遗憾下界为 $\Omega(\min(\sqrt{n^L T}, T))$，同样是指数级的

这说明裸问题在计算上不可行，必须引入合理的结构假设。

#### 2. DDMC 假设（Diminishing Distance with More Commons）

这是本文最核心的概念创新。对于任意两个等长序列 $\mathbf{y}, \mathbf{z}$ 和任意 token $\tau$：

$$|u(x_t, \mathbf{y}:\tau) - u(x_t, \mathbf{z}:\tau)| \leq |u(x_t, \mathbf{y}) - u(x_t, \mathbf{z})|$$

直觉解释：如果两个输出在后缀共享越多公共 token，用户感知的效用差异就越小。这是对"共同后缀减小差异"现象的形式化。

**与子模性的关系**：DDMC 与经典的子模性（submodularity）类似但本质不同——子模性描述"边际递减"，DDMC 描述"距离递减"；子模性关注集合的包含关系，DDMC 关注等长序列的公共后缀。两者互不蕴含，是互补关系。

#### 3. EOFUL 算法（Excessive Optimism Under the Face of Uncertainty）

针对 TLB 问题，结合贪心解码与 LinUCB 框架：

- **核心思路**：在每轮解码时，维护参数 $\theta$ 的置信球 $C_t$，对每个候选 token 乐观估计其效用，选择由置信球中最优参数和最优 token 共同决定的 token
- **置信球构造**：采用岭回归估计 $\hat{\theta}_t = \Sigma_t^{-1} \sum_{i=1}^{t-1} r_i \cdot \mathbf{y}_i$，置信球为椭球 $C_t = \{\vartheta: (\vartheta - \hat{\theta}_t)^\top \Sigma_t (\vartheta - \hat{\theta}_t) \leq \beta_t\}$
- **解码过程**：对每个位置 $k$，计算 $\tau^* = \arg\max_{\tau \in \mathcal{V}, \theta \in C_t} \langle \theta, e(x_t, \mathbf{y}:\tau) \rangle$，如果选到 EOS 则停止

**遗憾分析的五步证明**：

| 步骤 | 内容 | 作用 |
|------|------|------|
| Step 1 | 长度均等化 | 将算法序列与最优序列补齐到相同长度 |
| Step 2 | Level-k 遗憾定义 | 定义第 $k$ 层 token 选择产生的遗憾，为递归分析做准备 |
| Step 3 | 虚拟扩展 | 构造 $\mathbf{f}_t^{(1:k)} = \mathbf{y}_t^{(1:k-1)}:\mathbf{o}_t^{(k)}$，将 level-k 遗憾传导到 level-(k-1) |
| Step 4 | 遗憾分解 | 利用 DDMC 将整体遗憾表示为 level-1 遗憾加各前缀的估计误差 |
| Step 5 | 平方和遗憾 | 用标准线性 bandit 的平方和分析得出最终界 |

#### 4. GreedyETC 算法

针对 TMAB 问题，使用"探索-然后-提交"策略 + 贪心解码：

- **探索阶段**：对每个位置 $k$，尝试所有 $n$ 个 token 各 $N$ 次，计算平均奖励
- **提交阶段**：选择平均奖励最高的 token，固定后进入下一层
- **关键技巧**：只沿算法选择的路径取联合界（union bound），避免对所有 $n^L$ 个序列取联合界导致指数爆炸

### 损失函数 / 训练策略

本文是纯理论工作，不涉及梯度训练。核心优化目标是最小化累积伪遗憾。两个算法分别达到：

- **EOFUL (TLB)**：$\text{Reg} = O(cL\sqrt{dT\log T})$，对 $T$ 次线性，对 $L$ 线性
- **GreedyETC (TMAB)**：$\text{Reg} = O(nLT^{2/3}(\log T)^{1/3})$，对 $T$ 次线性，对 $L$ 线性

**LLM 对齐应用**：若用户效用为 $u(x_t, \mathbf{y}) = \gamma v(p(x_t, \mathbf{y})) + (1-\gamma) f(x_t, \mathbf{y})$，其中 $p$ 是冻结 LLM 的概率，$f$ 是线性可实现的未对齐函数，可以构造增广参数 $\theta' = [(1-\gamma)\theta : \gamma]$ 将问题归约为 TLB。

## 实验关键数据

### 主实验

#### DDMC 假设验证

在 TruthfulQA 和 HH-RLHF 数据集上，使用 Llama3-8B-Instruct 提取嵌入，验证两种距离函数下 DDMC 是否成立：

| 数据集 | 距离函数 | DDMC 趋势 | 说明 |
|--------|----------|-----------|------|
| TruthfulQA | $d_1$（$\ell_1$ 距离） | ✅ 递减 | 共同后缀 token 数增加时距离差递减 |
| TruthfulQA | $d_2$（$\ell_2$ 距离） | ✅ 递减 | 同上，$\ell_2$ 距离也满足 |
| HH-RLHF | $d_1$（$\ell_1$ 距离） | ✅ 强递减 | 递减趋势比 TruthfulQA 更显著 |
| HH-RLHF | $d_2$（$\ell_2$ 距离） | ✅ 强递减 | 不同任务有不同的递减结构 |

- 共同后缀 token 较少时递减曲率更明显
- HH-RLHF 的递减比 TruthfulQA 更剧烈，暗示不同任务具有不同的序列函数结构

#### EOFUL 性能验证

合成数据实验，设置 $L=30$，top-15 token 截断，$\gamma=0.8$：

| 算法 | 遗憾趋势 | 说明 |
|------|----------|------|
| EOFUL | 次线性增长 | 有效学习隐含参数 $\theta$ |
| 理论上界（缩放0.1×） | 次线性 | 实际表现远优于理论上界 |
| WrongTheta | 线性增长 | 使用错误 $\theta'=(-0.5,...,-0.5)$，对齐失败 |
| Misaligned Greedy | 线性增长 | 仅用 LLM 概率贪心解码，无法适应偏好 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| Assumption 3.6 验证 | $\rho_t^l = \|\mathbf{y}^{(1:l)}\|_{\Sigma_t} / \|\mathbf{y}\|_{\Sigma_t} \leq 1.25$ | 子序列 Mahalanobis 范数比值全程上界 1.25 |
| Max Ratio | $\leq 1.25$ | 取所有 token 位置的最大比值，仍然有界 |
| Mean Ratio | 远 $< 1.25$ | 平均比值更小，说明假设在实践中宽松 |

### 关键发现

1. **DDMC 在真实数据上成立**：TruthfulQA 和 HH-RLHF 上的嵌入距离实验验证了 DDMC 假设的合理性
2. **EOFUL 实际遗憾远优于理论上界**：暗示分析可能不够紧
3. **Assumption 3.6 实践中非常宽松**：比值上界仅 1.25，对遗憾的影响极小
4. **贪心解码在 DDMC 下近似最优**（Theorem 5.1）：只需 $nLT$ 次查询即可找到最优序列

## 亮点与洞察

1. **概念创新**：DDMC 假设简洁自然，抓住了"公共后缀缩小差异"的直觉，为序列函数的理论分析提供了新工具，与子模性互补
2. **贪心解码的理论辩护**：首次形式化证明在 DDMC 下贪心解码近乎最优，为 Gemini 等模型广泛使用贪心解码提供了理论支撑
3. **解码-对齐统一框架**：将解码和对齐问题统一为 bandit 框架，设定 $\gamma=0$ 退化为纯解码，$\gamma=1$ 退化为纯学习
4. **Level-k 遗憾 + 虚拟扩展**：巧妙的递归分析技巧避开了序列空间的指数爆炸
5. **实用价值**：EOFUL 可在推理时在线学习用户偏好，无需微调 LLM，适合个性化对齐

## 局限与展望

1. **EOFUL 依赖线性可实现假设和 Assumption 3.6**：实际 LLM 的效用函数可能是非线性的，线性近似能力有限
2. **GreedyETC 遗憾阶为 $T^{2/3}$**：不如 TLB 的 $\sqrt{T}$，能否在 TMAB 下也达到 $\sqrt{T}$ 是开放问题
3. **实验规模有限**：仅在合成数据上验证 EOFUL 性能，缺少大规模真实 LLM 对齐的端到端实验
4. **冷启动问题**：EOFUL 初始阶段可能产生严重误导的输出，影响用户体验
5. **单序列反馈**：目前只观测完整序列的奖励，若能利用 token 级反馈可能进一步提升效率
6. **DDMC 验证是均值意义上的**：没有证明对每一对序列都严格成立，理论与实证之间存在 gap
7. **未考虑 KL 正则化**：标准 RLHF 中的 KL 散度约束未被纳入，作者自己也指出这是未来方向

## 相关工作与启发

- **Bandit Tree Search (BTS)**：TMAB 与 BTS 密切相关，BTS 的决策版本可归约为 TMAB
- **序列子模性最大化**：DDMC 与序列子模性互补，分别刻画"距离递减"和"回报递减"
- **LinUCB / OFUL**：EOFUL 直接扩展了经典线性 bandit 的置信球方法到 token 化设定
- **DPO / RLHF**：本文方法不需要微调，是对 DPO/RLHF 的推理时替代
- **Controlled Decoding**：与 Mudgal et al. (2023) 等推理时对齐方法目标一致，但提供了正式的理论保证

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 全新问题形式化 + DDMC 假设 + 贪心解码最优性证明
- **理论深度**: ⭐⭐⭐⭐⭐ — 上下界完整，五步证明分析精巧
- **实验充分性**: ⭐⭐⭐ — 假设验证有说服力，但算法性能仅在合成数据上测试
- **实用价值**: ⭐⭐⭐⭐ — 理论洞察对解码策略选择有指导意义，直接工程应用还需更多工作
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，但符号较重，需要一定 bandit 理论背景

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Evaluating Morphological Alignment of Tokenizers in 70 Languages](evaluating_morphological_alignment_of_tokenizers_in_70_languages.md)
- [\[ACL 2025\] TokAlign: Efficient Vocabulary Adaptation via Token Alignment](../../ACL2025/llm_pretraining/tokalign_vocab_adaptation.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)
- [\[ACL 2025\] Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](../../ACL2025/llm_pretraining/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](../../NeurIPS2025/llm_pretraining/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)

</div>

<!-- RELATED:END -->
