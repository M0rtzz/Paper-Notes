---
title: >-
  [论文解读] On Differential Privacy for Adaptively Solving Search Problems via Sketching
description: >-
  [ICML 2025][AI安全][差分隐私] 首次将差分隐私技术拓展到**搜索问题**（近似最近邻查询和回归解向量输出），在稀疏邻域假设和良好条件数假设下，实现仅需 $\widetilde{O}(\sqrt{T})$ 份数据结构副本即可应对 $T$ 次自适应查询的搜索型数据结构。
tags:
  - ICML 2025
  - AI安全
  - 差分隐私
  - 自适应数据结构
  - 近似最近邻
  - 回归问题
  - sketching
---

# On Differential Privacy for Adaptively Solving Search Problems via Sketching

**会议**: ICML 2025  
**arXiv**: [2506.05503](https://arxiv.org/abs/2506.05503)  
**代码**: 无  
**领域**: AI安全 / 差分隐私  
**关键词**: 差分隐私, 自适应数据结构, 近似最近邻, 回归问题, sketching  

## 一句话总结

首次将差分隐私技术拓展到**搜索问题**（近似最近邻查询和回归解向量输出），在稀疏邻域假设和良好条件数假设下，实现仅需 $\widetilde{O}(\sqrt{T})$ 份数据结构副本即可应对 $T$ 次自适应查询的搜索型数据结构。

## 研究背景与动机

在现代算法设计中，数据结构常被嵌入迭代优化流程，需要对**自适应对手**（adversary 可根据先前输出设计后续输入）保持鲁棒性。传统随机数据结构仅对遗忘型（oblivious）对手提供概率保证。

### 核心挑战
- 最朴素方法：准备 $T$ 份独立数据结构副本 → 空间和预处理时间开销线性于 $T$
- 先前工作 [Hassidim et al., Ben-Eliezer et al.] 通过差分隐私将**数值估计问题**的副本数降为 $\widetilde{O}(\sqrt{T})$
- **关键瓶颈**：这些技术仅适用于输出单个数值的估计问题，而**搜索问题**需要返回完整解向量（如最近邻点或回归解），暴露的信息远多于数值查询

### 研究问题
> 能否设计仅需少于 $\min\{d, T\}$ 份独立副本的鲁棒搜索数据结构？

## 方法详解

### 整体框架

本文针对两类搜索问题提出差分隐私解决方案：
1. **(c,r)-近似最近邻 (ANN)**：给定数据集 $U \subseteq \mathbb{R}^d$ 和自适应查询序列
2. **自适应回归**：设计矩阵 $U$ 和响应向量 $b$ 可被自适应更新，需输出回归解向量

### 关键设计

#### 1. **自适应ANN：差分隐私选择机制**
- **核心思路**：将ANN问题归约为差分隐私选择问题
- **做法**：
    - 为每个数据点 $u \in U$ 创建一个类别
    - 每个数据结构输出所有找到的近邻 → 形成二进制指示向量
    - 对指示向量执行 ReportOneSidedNoisyArgMax（加指数噪声后取最大计数的索引）
- **设计动机**：固定数据集和查询点后，指示向量完全由数据结构的内部随机性决定 → 满足 $(\varepsilon, 0)$-DP → 可用高级组合定理
- **稀疏邻域假设**（Assumption 1.2-1.3）：每个查询的近似近邻数 $\leq s$ → 需要 $O(s \log n)$ 个数据结构保证正确性

#### 2. **稀疏argmax机制**（算法效率优化）
- **问题**：朴素的DP选择需 $\Omega(n)$ 时间（为每个类别生成噪声）
- **解决**：利用计数向量的 $s$-稀疏性，提出新的稀疏argmax机制：
    - 仅对支撑集的 $s$ 个非零项加指数噪声
    - 从 $n$ 阶统计量分布中采样最大噪声
    - 通过投硬币决定最大值来自支撑集还是非支撑集
- **时间复杂度**：$O(s \log n)$，等价于稀疏度 × 对数因子

#### 3. **自适应回归：坐标逐维私有中位数 + $\ell_\infty$ 保证**
- **核心思路**：将 [Ben-Eliezer et al.] 的私有中位数框架从数值估计扩展到向量输出
- **做法**：
    - 准备 $\widetilde{O}(\sqrt{Td})$ 份sketch矩阵
    - 每次查询采样 $O(\log T)$ 份sketch，求解sketched回归问题
    - 对解向量的每个坐标分别计算私有中位数
- **关键创新**：利用SRHT sketch的 $\ell_\infty$ 保证 $\|x_{(i)} - x^*\|_\infty \leq \frac{\alpha}{\sqrt{d}} \cdot \frac{\|Ux^* - b\|_2}{\sigma_{\min}(U)}$
- **效用分析**：$\|U(x^* - g)\|_2 \leq \alpha\kappa(U) \cdot \|Ux^* - b\|_2$ → 通过缩放 $\alpha$ 抵消条件数

### 损失函数/训练策略

本文为纯理论工作，无训练过程。核心复杂度指标：

| 组件 | ANN问题 | 回归问题 |
|------|---------|---------|
| 数据结构副本数 | $\widetilde{O}(\sqrt{T} \cdot s)$ | $\widetilde{O}(\sqrt{Td})$ |
| 空间 | $\widetilde{O}(\sqrt{T} \cdot s \cdot n^{1+\rho} + nd)$ | $\widetilde{O}(\sqrt{T} \cdot d^{2.5}\kappa^2/\alpha^2)$ |
| 查询时间 | $\widetilde{O}(s \cdot n^\rho d)$ | $\widetilde{O}(d^{\omega+1}\kappa^2/\alpha^2)$ |

## 实验关键数据

### 主实验：ANN数据结构复杂度对比（Table 1）

| 方法 | 空间 | 摊销预处理时间 | 查询时间 |
|------|------|---------------|---------|
| $T$ 份副本 | $Tn^{1+\rho} + nd$ | $n^{1+\rho}d$ | $n^\rho d$ |
| $d$ 份副本 | $n^{1+\rho}d$ | $\frac{d}{T}n^{1+\rho}d$ | $n^\rho d$ |
| **本文** | $\sqrt{T}sn^{1+\rho} + nd$ | $\frac{s}{\sqrt{T}}n^{1+\rho}d$ | $sn^\rho d$ |

### 回归数据结构复杂度对比（Table 2）

| 方法 | 空间 | 查询时间 |
|------|------|---------|
| $T$ 份副本 | $Td^2/\alpha^2$ | $d^{\omega+1}/\alpha^2$ |
| **本文** | $\sqrt{T}d^{2.5}\kappa^2/\alpha^2$ | $d^{\omega+1}\kappa^2/\alpha^2$ |

### 关键发现

1. 当 $s = o(\sqrt{T})$ 时，ANN的副本数严格少于 $T$
2. 回归问题在 $d \ll T$ 时副本数从 $T$ 降为 $\sqrt{Td}$
3. 稀疏标签偏移场景下（Theorem 1.9），可通过预条件器完全消除对条件数 $\kappa$ 的依赖

## 亮点与洞察

1. **搜索→选择归约**：将搜索问题巧妙转化为DP选择问题，避免了直接在高维空间做私有聚合的困难
2. **稀疏argmax机制**：在保持DP的同时实现 $O(s\log n)$ 时间复杂度，是独立的技术贡献
3. **$\ell_\infty$ 保证的新应用**：SRHT sketch的 $\ell_\infty$ 保证此前鲜有应用场景，本文发现它是坐标逐维私有中位数的天然搭配
4. **应用驱动**：直接得到在线加权匹配和终端嵌入的改进结果

## 局限与展望

1. **稀疏邻域假设的必要性**：ANN结果依赖于参数 $s$（近似近邻的上界），当 $s$ 接近 $\sqrt{T}$ 时优势消失
2. **条件数依赖**：回归结果（Theorem 1.8）对条件数有二次依赖，尽管Theorem 1.10给出了对数依赖的替代方案
3. **纯理论贡献**：没有实验验证，理论改进在实际场景中的效果未知
4. **局限于特定搜索问题**：尚未扩展到更一般的搜索/优化问题

## 相关工作与启发

- **差分隐私与自适应数据结构**：延续 [Hassidim et al. 2020], [Ben-Eliezer et al. 2022] 等工作的DP-数据结构范式
- **流式算法中的自适应鲁棒性**：[Ben-Eliezer & Yogev 2020], [Woodruff & Zhou 2021] 等工作的自然扩展
- **LSH与ANN**：对 [Indyk & Motwani 1998] 经典LSH结果的自适应增强
- **启发**：DP不仅保护隐私，更可作为算法正确性的通用工具——通过隐藏内部随机性来抵御自适应攻击

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次将DP搜索扩展到高维解向量输出，理论深度突出
- 实验充分度: ⭐⭐ — 纯理论工作，无实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，技术概述部分优秀
- 价值: ⭐⭐⭐⭐ — 在DP+数据结构的交叉领域做出重要推进

<!-- RELATED:START -->

## 相关论文

- [Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy](cape_context-aware_prompt_perturbation_mechanism_with_differential_privacy.md)
- [Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)

<!-- RELATED:END -->
