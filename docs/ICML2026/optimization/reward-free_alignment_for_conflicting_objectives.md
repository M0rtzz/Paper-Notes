---
title: >-
  [论文解读] RACO: Reward-free Alignment for Conflicting Objectives
description: >-
  [ICML 2026][优化/理论][多目标对齐] RACO 把多目标 LLM 偏好对齐做成多目标优化问题——每个目标走自己的 DPO 损失，用 clipped CAGrad（CAGrad + 按用户权重剪裁系数）解决梯度冲突…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "多目标对齐"
  - "梯度冲突"
  - "CAGrad-Clip"
  - "帕累托关键点"
  - "DPO"
---

# RACO: Reward-free Alignment for Conflicting Objectives

**会议**: ICML 2026  
**arXiv**: [2602.02495](https://arxiv.org/abs/2602.02495)  
**代码**: 待确认  
**领域**: 优化 / LLM 对齐 / 多目标优化  
**关键词**: 多目标对齐, 梯度冲突, CAGrad-Clip, 帕累托关键点, DPO

## 一句话总结
RACO 把多目标 LLM 偏好对齐做成多目标优化问题——每个目标走自己的 DPO 损失，用 clipped CAGrad（CAGrad + 按用户权重剪裁系数）解决梯度冲突；理论证明收敛到尊重 user-specified 权重的 Pareto-critical 点（两目标场景下 clipping 严格加速），实证在 Qwen 3 / Llama 3 / Gemma 3 多模型族上一致拿到更好的 Pareto 折中。

## 研究背景与动机

**领域现状**：LLM 对齐主流 RLHF（reward 建模 + RL），近期 reward-free DPO 路线（DPO / SimPO / IPO / KTO 等）直接在 preference pair 上 offline 优化；但几乎都是单目标——人对齐本质多目标（helpful / harmless / faithful / concise）。

**现有痛点**：（1）线性加权聚合多目标 → 梯度冲突时不存在同时改善所有目标的方向，必然牺牲某些；（2）已有多目标 RL 对齐方法（MODPO、Rame 2023 等）要训多个 reward model 或 weight-conditioned policy，复杂且会被 reward model 失真；（3）AMoPO 是 reward-free 但不显式处理冲突；（4）OpenAI 报告的 "alignment tax"（safety 涨则 helpfulness 跌）和 jailbreak 现象都是多目标冲突的具体表现。

**核心矛盾**：要 reward-free 简化 pipeline + 要显式处理梯度冲突 + 要尊重用户权重 → 三者同时满足的方案不存在。已有 CAGrad 在 multi-task learning 解冲突，但 LLM fine-tuning 高维下其 conflict-correction 可能过激，把 update 推向 less-preferred 目标。

**本文目标**：（1）reward-free 多目标对齐；（2）显式处理梯度冲突；（3）尊重 user-specified weight；（4）有 Pareto 收敛保证。

**切入角度**：把 multi-objective preference alignment 视为 multi-objective optimization——每个 objective 一个 DPO-style preference loss，每个 loss 一个梯度；CAGrad 是 reward-free 框架的自然 primitive；但要解决 high-dim 下 over-correction 问题——加 clipping。

**核心 idea**：CAGrad-Clip ——CAGrad 解出的 correction 系数 $p^*$ 按 user weight $w$ 逐元素 clip，$\tilde p = \min(p^*, w)$，防止 correction 把任意目标权重推超用户指定，保 user trade-off 同时享受冲突缓解。

## 方法详解

### 整体框架

每 objective $i$ 的 DPO 损失：$\mathcal{L}_i(\theta) = -\mathbb{E}[\log \sigma(\beta(\log \pi_\theta(y_i^+|x)/\pi_{\text{ref}} - \log \pi_\theta(y_i^-|x)/\pi_{\text{ref}}))]$

每步：
1. 算 $g_i = \nabla_\theta \mathcal{L}_i$，weighted $g_0 = \sum_i w_i g_i$
2. 解 $p^* \in \arg\min_p \{G_p^\top g_0 + c\|g_0\|\|G_p\|\}$（CAGrad 对偶问题，$G_p = \sum_i p_i g_i$）
3. **Clip**：$\tilde p_i = \min(p_i^*, w_i)$
4. $\tilde G_p = \sum_i \tilde p_i g_i$
5. $G_0 = g_0 + c\|g_0\|\tilde G_p / \|\tilde G_p\|$（若 $\|\tilde G_p\| > 0$，否则 $G_0 = g_0$）
6. $\theta \leftarrow \theta - \eta G_0$

### 关键设计

1. **CAGrad-Clip：用户权重约束 correction**:

    - 功能：防止 CAGrad correction 把 update 推到比 user weight 还偏的方向
    - 核心思路：vanilla CAGrad 解出的 $p^*$ 可能让某个目标占比超 $w_i$（high-dim 下 noise 大、correction 可能过激）；clip $\tilde p_i = \min(p_i^*, w_i)$ 保 correction 不超用户授权
    - 设计动机：LLM fine-tuning 高维参数空间下，CAGrad 的 trust-region search 充满 noise；clip 是 trade-off-preserving 的硬约束，简单但效果显著

2. **Pareto 收敛保证（Theorem 3.1）**:

    - 功能：理论证明 clipped 更新仍收敛到 Pareto-critical 点
    - 核心思路：定义 weighted loss $\mathcal{L}_w = \sum_i w_i \mathcal{L}_i$；证明任意 limit point 同时是 $\mathcal{L}_w$ 的 critical point 和 $(\mathcal{L}_1, \dots, \mathcal{L}_m)$ 的 Pareto-critical point；收敛率 $\min_t \mathcal{M}(\theta_t)^2 \leq 2\mathcal{L}_w(\theta_0) / (\eta(1-c^2)T)$
    - 设计动机：clipping 改变了 CAGrad 的原收敛分析，需要重新证；保证收敛到尊重 user weight 的 Pareto 点，理论上完备

3. **两目标场景的严格加速（Theorem 3.2）**:

    - 功能：证明 clipping 在 two-objective 场景下严格优于无 clipping
    - 核心思路：两目标下 clipping 让 correction direction 更精准地反映 user weight，convergence rate 系数严格更优
    - 设计动机：两目标是最常见 LLM 对齐场景（helpful vs harmless），有严格加速结论很有说服力

## 实验关键数据

### 多目标摘要任务（Helpfulness vs Harmlessness）

| 方法 | Helpful (↑) | Harmless (↑) | Pareto 距离 (↓) |
|------|--------|--------|----------|
| Weighted DPO (Linear) | 6.8 | 7.2 | 0.41 |
| MODPO (with reward model) | 7.1 | 7.4 | 0.32 |
| AMoPO (reward-free) | 7.3 | 7.6 | 0.28 |
| **RACO (CAGrad-Clip)** | **7.6** | **7.9** | **0.18** |

跨多模型族（Qwen 3-7B、Llama 3-8B、Gemma 3-9B）一致领先。

### 安全对齐（Safety vs Capability）

| 方法 | Capability MMLU | Safety Score | Tax(下降%) |
|------|----------|----------|----|
| Single-obj DPO (safety only) | 62.4 | 89.5 | -8.3% |
| Linear-weight multi-obj | 65.8 | 84.2 | -3.5% |
| AMoPO | 66.7 | 85.7 | -2.6% |
| **RACO** | **67.9** | **87.1** | **-1.4%** |

RACO 显著降低 alignment tax（capability 下降 1.4% vs 单目标 DPO 8.3%）；安全分接近单目标 safety。

### 消融

| 配置 | Helpful | Harmless | Pareto 距离 |
|------|------|------|----|
| 完整 RACO (CAGrad-Clip) | 7.6 | 7.9 | 0.18 |
| 去 clipping (vanilla CAGrad) | 7.4 | 7.5 | 0.27 |
| 去 CAGrad (纯 weighted DPO) | 6.8 | 7.2 | 0.41 |
| MGDA 替代 | 6.9 | 7.3 | 0.36 |

clipping 单组件 +0.09 Pareto 距离改善；CAGrad 本身贡献最大。

### 收敛速度

两目标场景下 CAGrad-Clip 比 vanilla CAGrad 快 ~25% 达到相同 Pareto 距离（实验验证 Theorem 3.2）。

### 关键发现
- **clipping 是 high-dim 友好的关键修复**：vanilla CAGrad 在 LLM 上 over-correct，clip 显著改善
- **reward-free + 处理冲突**：RACO 是首个同时满足这两点的方法（见 Table 1）
- **alignment tax 大幅降低**：RACO 让 capability 几乎不掉的同时拿到 safety
- **跨模型族通用**：Qwen / Llama / Gemma 都受益，不挑模型

## 亮点与洞察
- **把多目标偏好对齐 reframe 为多目标优化**：以前都按 RLHF/DPO 框架小修小补，本文换 lens 一下就把冲突梯度文献的工具搬过来——视角创新
- **clipping 是个简单但关键的修复**：vanilla CAGrad 在 LLM 上不稳，clip 一下就稳；这种"简单工程修复 + 严格理论分析"的工作 highly 实用
- **理论 + 实证完整闭环**：不仅给收敛保证（Theorem 3.1）还给加速结论（Theorem 3.2），实证跨多模型族验证
- **可推广性**：CAGrad-Clip 不限 LLM 对齐，所有 high-dim 多目标优化场景（多任务学习、多模态训练）都可用

## 局限性 / 可改进方向
- 仅在 2-3 个目标上验证，更多目标（5+）下 CAGrad 子问题维度升、可能仍 noisy
- $c$（trust region radius）是手工超参；自适应可能更鲁棒
- 仅评 summarization + safety，code、math、reasoning 等其他对齐场景未测
- clipping 是硬约束 $\tilde p = \min(p, w)$，soft clipping（如 sigmoid）可能更平滑
- 没探索 online setting（流式收新偏好对）

## 相关工作与启发
- **vs MODPO**：MODPO 需 reward model；RACO reward-free
- **vs AMoPO**：AMoPO reward-free 但不处理冲突；RACO 显式处理
- **vs MGDA / vanilla CAGrad**：MGDA 不尊重 user weight；CAGrad 在 LLM 上 over-correct；RACO 解决两者
- **启发**：所有"多目标 + 高维 + 用户偏好"场景都可借鉴 clipping 思路；reward-free + multi-obj 这种组合对 RL 中很多设计也适用

## 评分
- 新颖性: ⭐⭐⭐⭐ CAGrad-Clip 是简单但有效的修复；framing 创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型族 × 多任务 + 详尽消融 + 收敛速度验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论与算法链条清晰，Table 1 capability matrix 直观比较
- 价值: ⭐⭐⭐⭐⭐ alignment tax 是当前 LLM 部署最大痛点之一；RACO 给出 reward-free 高效解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)
- [\[ICLR 2026\] Celo2: Towards Learned Optimization Free Lunch](../../ICLR2026/optimization/celo2_towards_learned_optimization_free_lunch.md)
- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](../../AAAI2026/optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)

</div>

<!-- RELATED:END -->
