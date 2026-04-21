---
title: >-
  [论文解读] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding
description: >-
  [ICLR 2026][LLM/NLP][扩散模型] 提出 SureLock，当 Masked Diffusion LM 中已 unmask 的 token 后验分布稳定后永久锁定该位置（跳过 Q 投影和 FFN，缓存 KV），将每步注意力计算从 $O(N^2d)$ 降为 $O(MNd)$，在 LLaDA-8B 上减少 30-50% FLOPs 且不损生成质量。
tags:
  - ICLR 2026
  - LLM/NLP
  - 扩散模型
  - 推理加速
  - Token Locking
  - KL散度
  - KV Cache
---

# Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding

**会议**: ICLR 2026  
**arXiv**: [2602.06412](https://arxiv.org/abs/2602.06412)  
**代码**: https://daioba.github.io/surelock  
**领域**: 文本生成  
**关键词**: Masked Diffusion LM, 推理加速, Token Locking, KL散度, KV Cache

## 一句话总结
提出 SureLock，当 Masked Diffusion LM 中已 unmask 的 token 后验分布稳定后永久锁定该位置（跳过 Q 投影和 FFN，缓存 KV），将每步注意力计算从 $O(N^2d)$ 降为 $O(MNd)$，在 LLaDA-8B 上减少 30-50% FLOPs 且不损生成质量。

## 研究背景与动机

**领域现状**：Masked Diffusion LM（MDLM，如 LLaDA、Dream）通过迭代去噪生成文本，每步需对所有 $N$ 个 token 位置重算注意力和 FFN，计算复杂度 $O(N^2d)$。

**现有痛点**：很多 token 在被 unmask 后后验分布迅速稳定，但标准采样器仍为它们重复计算——大量无效计算。现有加速方法要么减少步数（temporal），要么跨步复用 KV（reuse），但每步内仍发射 $N$ 个 query 行，空间复杂度不变。

**核心矛盾**：MDLM 的双向注意力要求每步全量计算，但大部分 unmask token 实际上已"收敛"，不需要重算。

**本文目标** 如何识别并永久跳过已收敛 token 的计算，实现单调递减的每步计算量？

**切入角度**：监测每个 token 位置相邻步之间的 KL 散度，低于阈值则永久锁定。

**核心 idea**：后验稳定的 token 永久退出计算（锁定 KV、跳过 Q/FFN），活跃集随采样推进单调收缩。

## 方法详解

### 整体框架
在 MDLM 的每步迭代中，SureLock 维护两个集合：活跃集 $\mathcal{A}_t$（需计算）和锁定集 $\mathcal{L}_t$（跳过计算但 KV 被缓存）。每步仅对活跃位置计算 Q 投影和 FFN，注意力时通过缓存的 KV 让活跃 token 仍能 attend 到锁定 token。

### 关键设计

1. **永久锁定机制 (Permanent Locking)**:

    - 功能：一旦 token 位置 $i$ 被锁定，后续所有步跳过其 Q 投影和 FFN，使用缓存的 $K, V$ 值
    - 核心思路：锁定后 $\hat{p}_t^{(i)} = p_{t^*}^{(i)}$（后验冻结），但 $K^{\text{all}}[\mathcal{L}_t] \leftarrow \mathcal{C}.k[\mathcal{L}_t]$ 使其他 token 仍可 attend 到锁定 token
    - 设计动机：与选择性更新方法不同（如 dLLM-Cache 选"现在计算什么"），SureLock 选"永久移除什么"——活跃集只减不增，保证计算量单调下降

2. **锁定判据：步间 KL 散度**:

    - 功能：$D_t^{(i)} = \text{KL}(p_t^{(i)} \| p_{t-1}^{(i)}) \leq \varepsilon$ 时锁定
    - 核心思路：KL 散度测量相邻步后验分布的变化程度。阈值 $\varepsilon$ 直接转化为终端误差上界 $\delta = C_{\text{tail}}\sqrt{\varepsilon}$
    - 设计动机：局部 KL 廉价可算，且理论上可控——Theorem 1 证明了锁定步的局部 KL 足以约束最终 token 概率的偏差

3. **可选置信度门控 (Confidence Gate)**:

    - 功能：$u_t^{(i)} = 1 - \max_v p_t^{(i)}(v) \leq q_m(u_t)$ 作为辅助条件
    - 设计动机：安全网——只锁定后验峰值足够高的 token，避免在"还在犹豫"的 token 上过早锁定

4. **理论误差上界 (Theorem 1)**:

    - 核心结论：$\|\log p_T^{(i)} - \log \hat{p}_T^{(i)}\|_\infty \leq C_{\text{tail}}\sqrt{D_{t^*}^{(i)}}$，其中 $C_{\text{tail}} = L_{\text{sm}} L / (1 - \sqrt{\rho})$
    - 意义：将局部的、廉价可算的 KL 阈值直接映射到全局的终端误差上界，为超参选择提供理论依据

### 损失函数 / 训练策略
SureLock 是**无训练**的推理时方法，不修改模型参数。与已有的 temporal 和 reuse 加速方法正交可组合。

## 实验关键数据

### 主实验

**LLaDA-8B-Instruct (MT-Bench + WikiText-103)**:

| 配置 | FLOPs 减少 | 生成质量 |
|------|-----------|---------|
| Baseline (无锁定) | 0% | 基线 |
| SureLock (ε=0.01) | ~30% | ≈基线 |
| SureLock (ε=0.05) | ~50% | ≈基线 |
| SureLock (ε=0.1) | ~50%+ | 略降 |

### 消融实验

| 配置 | FLOPs 节省 | 质量 | 说明 |
|------|-----------|------|------|
| 仅 KL 判据 | 与完整版相当 | ≈ | KL 足够作为唯一判据 |
| 仅 confidence gate | 较少 | 保持 | 单独置信度不够激进 |
| 无 KV 缓存 | 同等 FLOPs | 显著下降 | 锁定 token 不可被 attend 导致信息丢失 |

### 关键发现
- 活跃位置数 $M_t$ 随步数**单调递减**，验证了"收敛 token 越来越多"的假设
- 30-50% FLOPs 减少在 WikiText-103 困惑度和 MT-Bench 评分上**零损失或极微下降**
- 与 temporal 方法（减少步数）和 reuse 方法（跨步复用 KV）正交，可叠加使用
- 理论界虽然保守（设计导向而非精确预测），但提供了调参($\varepsilon$)的明确指导

## 亮点与洞察
- **从"选择哪些计算"到"永久移除哪些计算"**：这种视角转换是关键创新。活跃集只收缩不扩张，使计算曲线单调下降，比选择性更新方法预测性更强。
- **理论-实践闭环**：KL 阈值 → 终端误差上界的闭式关系为超参选择提供了不靠调参的理论基础，这在系统加速工作中难得。
- **与 d²Cache 形成互补**：d²Cache 是细粒度选择"每步更新哪些 token"，SureLock 是永久移除已收敛 token——两者可以组合。

## 局限与展望
- Theorem 1 的几何尾收缩假设(A2)较强，实际中后验变化不一定严格满足
- 永久锁定有不可撤销风险——如果 token 在锁定后因其他位置变化而"需要"改变，无法恢复
- 仅在 LLaDA-8B 上验证，其他 MDLM（如 Dream、MDLM-original）未测试
- 锁定阈值 $\varepsilon$ 仍需手动设定，理论界中的 $C_{\text{tail}}$ 常数不易精确估计

## 相关工作与启发
- **vs dLLM-Cache**: dLLM-Cache 在每步选择性更新部分 token（非永久），SureLock 永久锁定，活跃集单调收缩，长期节省更多
- **vs Fast-dLLM**: Fast-dLLM 按块半自回归解码，SureLock 在 token 级操作，更细粒度且正交可组合
- **vs AR KV Cache**: AR 的 KV cache 天然增长，SureLock 的 KV cache 在双向注意力中模拟类似"不变量缓存"的效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 永久锁定收敛 token 的思路简洁有效，理论分析加分
- 实验充分度: ⭐⭐⭐ 仅一个模型、任务覆盖有限
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、算法表述精确、理论推导完整
- 价值: ⭐⭐⭐⭐ 为 MDLM 推理加速提供了新的正交维度

<!-- RELATED:START -->

## 相关论文

- [d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance_estimation_.md)
- [Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](../../ACL2025/llm_nlp/segment_level_diffusion.md)

<!-- RELATED:END -->
