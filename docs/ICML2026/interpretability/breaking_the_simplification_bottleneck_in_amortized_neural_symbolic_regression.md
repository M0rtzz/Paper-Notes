---
title: >-
  [论文解读] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression
description: >-
  [ICML 2026][可解释性][符号回归] 提出 SimpliPy（基于规则的化简引擎，比 SymPy 快 100 倍）和 Flash-ANSR（基于 Transformer 的摊销符号回归框架），在 FastSRB 基准上以 ~58% 的恢复率匹敌甚至超越遗传编程方法 PySR…
tags:
  - "ICML 2026"
  - "可解释性"
  - "符号回归"
  - "表达式化简"
  - "Transformer"
  - "摊销推理"
  - "科学发现"
---

# Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression

**会议**: ICML 2026  
**arXiv**: [2602.08885](https://arxiv.org/abs/2602.08885)  
**代码**: https://github.com/psaegert/flash-ansr  
**领域**: 可解释性  
**关键词**: 符号回归, 表达式化简, Transformer, 摊销推理, 科学发现  

## 一句话总结

提出 SimpliPy（基于规则的化简引擎，比 SymPy 快 100 倍）和 Flash-ANSR（基于 Transformer 的摊销符号回归框架），在 FastSRB 基准上以 ~58% 的恢复率匹敌甚至超越遗传编程方法 PySR，同时随推理预算增加生成更简洁的表达式。

## 研究背景与动机

**领域现状**：符号回归（Symbolic Regression, SR）旨在从观测数据中发现可解释的解析表达式。传统方法以遗传编程（GP）为主（如 PySR），但每个数据集都从头搜索，无法在任务间迁移结构知识。摊销 SR 通过在海量合成数据上预训练 Transformer 来学习后验 $p(\bm{\tau}|\mathcal{D})$，将计算负担转移到一次性预训练阶段。

**现有痛点**：摊销 SR 面临三重困境。第一，静态语料方案（如 NeSymReS）使用 SymPy 离线化简生成固定数据集（~100M 表达式），但高昂的化简成本限制了覆盖度和维度（$D \leq 3$）。第二，部分方法（如 E2E）放弃化简直接在未规范化表达式上训练，导致模型浪费容量学习语法冗余（$x+0$, $1 \cdot x$ 等）。第三，将 SymPy 嵌入训练循环的方法（如 NSRwH）引入严重的计算瓶颈，SymPy 的中位化简时间约 100ms/表达式。

**核心矛盾**：表达式化简的质量与速度之间存在根本矛盾——通用 CAS 系统的面向对象解析和树遍历机制对 SR 训练场景来说过于重量级，但不化简又导致训练目标冗余和推理效率低下。

**本文目标**：设计一个快速且高质量的化简引擎，打破 CAS 瓶颈，使摊销 SR 能扩展到更大规模、更高维度的训练。

**切入角度**：作者观察到 SR 训练中遇到的表达式具有有限的结构复杂度，因此可以将化简本身也"摊销化"——离线穷举发现所有短模式的等价规则，运行时仅做快速查表匹配。

**核心 idea**：用预计算的哈希索引规则集替代通用 CAS，将符号化简从 $O(100\text{ms})$ 降到 $O(1\text{ms})$，从而实现在训练循环中同步化简在线生成的表达式。

## 方法详解

### 整体框架

Flash-ANSR 的训练 pipeline 分为四阶段：(1) 骨架采样——按长度指数先验采样算子数量，用 Lample & Charton 算法构建前缀骨架；(2) SimpliPy 化简——将冗余表达式归约为标准形式；(3) 去污染——通过符号和数值双重检测剔除与测试集等价的表达式；(4) 数据集渲染——采样常数和数据点，生成 $(X, y)$ 对。推理阶段采用 softmax 采样生成 $K$ 个候选骨架，经 SimpliPy 去重后用 Levenberg-Marquardt 优化常数，最终按拟合质量和简洁性正则化排序选出最优表达式。

### 关键设计

1. **SimpliPy 化简引擎**:

    - 功能：将代数表达式快速归约为最短标准形式，实现 100 倍于 SymPy 的加速
    - 核心思路：分两阶段工作。**离线阶段**——按长度分层穷举所有至多 $L_{\max}=7$ 符号的表达式模式，通过数值等价测试发现化简规则 $\bm{\tau} \to \bm{\tau}'$，每条规则必须满足严格长度缩减 $|\bm{\tau}'| < |\bm{\tau}|$ 和变量不增条件。**在线阶段**——将无变量的 ground 规则用哈希表实现 $O(1)$ 查找，含变量的 pattern 规则按算子和长度分桶存储为树结构做子树匹配。运行时交替执行模式匹配（ApplyRules）和项消去（CancelTerms）至多 $K=5$ 轮，最终排序交换律操作数并替换合并后的常量
    - 设计动机：通用 CAS 从第一性原理求解化简，对 SR 训练场景来说大材小用。通过将化简本身摊销化（离线投入 ~100h 计算，换取运行时毫秒级化简），彻底消除训练循环中的 CAS 瓶颈

2. **可扩展编码器-解码器架构**:

    - 功能：将数据集编码为条件信息，自回归生成前缀表示的表达式骨架
    - 核心思路：编码器采用 Set Transformer 处理变长数据集，引入 masked RMSSetNorm 替代 LayerNorm/SetNorm（统计轴数与 SetNorm 相同但参数量减半，且正确处理 padding）；输入用 32-bit IEEE-754 多热编码（覆盖 $10^{-38}$ 到 $10^{38}$，远超 16-bit 的 $10^{-4}$ 到 $10^{4}$）。解码器使用 Pre-RMSNorm + FlashAttention + RoPE 位置编码，推理时用 softmax 采样替代 beam search 以提高候选多样性
    - 设计动机：Pre-Norm 比 Post-Norm 训练更稳定（消融实验中 Post-Norm 训练直接发散）；32-bit 编码覆盖物理域数据的真实量级；softmax 采样在 $c=4096$ 时产生的语法重写仅为 beam search 的 $1/70$，恢复率高 9.4pp

3. **严格去污染与评估协议**:

    - 功能：防止训练数据泄漏，建立可靠的评估标准
    - 核心思路：去污染时先剪枝所有常数节点得到骨架，然后同时做符号比较（token 相等）和数值比较（在固定网格 $X_{\text{check}} \in \mathbb{R}^{512 \times D}$ 上求值后四位小数取整再哈希，碰撞即拒绝）。评估采用机器精度恢复标准 $\text{FVU} \leq 1.19 \times 10^{-7}$，分析推理时间-恢复率的 Pareto 前沿
    - 设计动机：先前几乎所有工作都未做严格去污染，可能导致性能高估；宽松的成功阈值（如 $R^2 > 0.9$）掩盖了真实失败案例

### 训练策略

训练目标为交叉熵损失：$\hat{\theta} = \arg\min_{\theta} \mathbb{E}[-\sum_{t=1}^{L} \log p_{\theta}(\bar{\tau}_t^* | \bar{\tau}_{<t}^*, \mathcal{D})]$，编码器和解码器端到端联合训练。共训练四个规模的模型（3M / 20M / 120M / 1B 参数），最大模型在 512M 在线生成的数据-表达式对上训练。推理时按简洁性正则化排序：$\hat{\bm{\tau}}^{\star} = \arg\min \log_{10}\text{FVU}(\hat{\bm{\tau}}) + \gamma \cdot |\hat{\bm{\tau}}|$，默认 $\gamma = 0.05$。

## 实验关键数据

### 主实验（FastSRB 基准，115 个表达式）

| 方法 | 类型 | vNRR↑ (~10s) | vNRR↑ (峰值) | 表达式长度比↓ | 说明 |
|------|------|-------------|-------------|-------------|------|
| NeSymReS | 摊销 SR | ~10% | ~10% | — | 饱和，无法泛化 |
| E2E | 摊销 SR | <2.5% | <2.5% | — | 几乎完全失败 |
| PySR | 遗传编程 | ~45% | 50.0% | 0.94→1.85 | 复杂度随时间增长 |
| Flash-ANSR 3M | 摊销 SR | ~25% | ~35% | — | 落后于 PySR |
| Flash-ANSR 120M | 摊销 SR | ~45% | **~58%** | 1.40→1.27 | 超越 PySR，简洁性反转 |

### SimpliPy 化简效率对比

| 化简引擎 | 中位时间 | 化简比 | 超时率(>1s) | 长度增加比例 |
|----------|---------|--------|------------|-------------|
| SymPy | ~100ms | 较好 | 9% | 38%-52% |
| SimpliPy ($L_{\max}=4$) | ~1ms | 接近 SymPy | 0% | 0%（严格不增长） |
| SimpliPy ($L_{\max} \geq 5$) | 数ms | **超越 SymPy** | 0% | 0% |

### 消融实验

| 配置 | vNRR↑ | 长度比 | 说明 |
|------|-------|--------|------|
| Full (SimpliPy, 100M) | 最高 | 最低 | 完整模型 |
| A-U (无化简) | 接近 | +40-50% | 表达式冗余严重 |
| B1 (Post-Norm) | 训练失败 | — | 梯度不稳定 |
| B2 (16-bit 编码) | 显著下降 | 显著上升 | 数值精度不足 |
| Beam Search vs Softmax | -9.4pp | 重写多 70× | beam search 模式坍缩 |

### 关键发现

- **简洁性反转（Parsimony Inversion）**：PySR 随推理时间增长表达式越来越复杂（长度比 0.94→1.85），而 Flash-ANSR 反向收敛到更简洁的形式（1.40→1.27），这是因为更多采样能找到稀有但简洁的"大海捞针"式正确表达式
- **数据稀疏性的三阶段相变**：在 $M \approx 8$ 个数据点处出现"复杂度峰值"，类似于 Deep Double Descent——太少的点导致简洁的高偏差近似，临界点处模型用过多常数插值，充足数据后才收敛到真实表达式
- **噪声鲁棒性不足**：在噪声水平 $\eta \geq 10^{-2}$ 时 PySR 明显优于 Flash-ANSR，因为模型仅在无噪声数据上训练，将噪声误解为高频信号

## 亮点与洞察

- **化简的摊销化思想**：将化简本身视为可预计算的查表问题而非在线求解问题，这种将"一次性重计算"换取"运行时零成本"的思路可迁移到任何需要在训练循环中执行昂贵符号操作的场景
- **softmax 采样优于 beam search**：在多模态后验下，beam search 的模式寻求行为导致 70 倍的冗余重写，softmax 采样以更低成本探索更多功能不同的假设——这一发现对所有序列生成任务都有启示
- **自我发现 scaling law**：作者用 Flash-ANSR 本身对自己的 scaling curve 做符号回归，发现其性能渐近遵循 $\text{vNRR} \propto \log\log T$，而 PySR 有约 53% 的上界——用自己的工具分析自己的行为，方法论上很优雅

## 局限与展望

- **噪声鲁棒性差**：仅在无噪声数据上训练，噪声构成分布外偏移，未来需在训练中引入噪声增强
- **化简规则的离线发现成本高**：$L_{\max}=7$ 需要 ~100h（32 线程），扩展到更长模式的成本呈指数增长
- **评估仍限于 FastSRB**：115 个表达式的规模有限，在更复杂的真实科学场景中的表现有待验证
- **改进方向**：训练时加入噪声数据、探索更宽的生成分布、尝试替代的编码/解码范式（如扩散模型）

## 相关工作与启发

- **NeSymReS / E2E**：先前摊销 SR 代表工作，分别受限于静态数据集和未化简训练，本文统一解决了两者的瓶颈
- **PySR**：当前遗传编程 SOTA，在中等计算预算下被 Flash-ANSR 追平甚至超越
- **启发**：将"化简"视为独立的可摊销组件，而非必须在线求解的子问题，这种解耦思路值得在其他涉及符号操作的 ML 系统中借鉴

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2025\] Ab Initio Nonparametric Variable Selection for Scalable Symbolic Regression with Large p](../../ICML2025/interpretability/ab_initio_nonparametric_variable_selection_for_scalable_symbolic_regression_with.md)
- [\[ICML 2026\] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere](neural_collapse_by_design_learning_class_prototypes_on_the_hypersphere.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)

</div>

<!-- RELATED:END -->
