---
title: >-
  [论文解读] Theoretical Limitations of Ensembles in the Age of Overparameterization
description: >-
  [ICML2025][LLM/NLP][过参数化集成] 在过参数化条件下，无限集成模型与单个无穷宽模型逐点等价，集成方差不再反映传统贝叶斯不确定性而是衡量增加模型容量的预期效果，从理论上解释了深度集成相比大模型无本质泛化优势的经验观察。
tags:
  - ICML2025
  - LLM/NLP
  - 过参数化集成
  - 随机特征模型
  - 不确定性量化
  - 核回归
  - 泛化理论
---

# Theoretical Limitations of Ensembles in the Age of Overparameterization

**会议**: ICML2025  
**arXiv**: [2410.16201](https://arxiv.org/abs/2410.16201)  
**代码**: 未开源  
**领域**: LLM/NLP理论 · 集成学习 · 过参数化理论  
**关键词**: 过参数化集成, 随机特征模型, 不确定性量化, 核回归, 泛化理论

## 一句话总结

在过参数化条件下，无限集成模型与单个无穷宽模型逐点等价，集成方差不再反映传统贝叶斯不确定性而是衡量增加模型容量的预期效果，从理论上解释了深度集成相比大模型无本质泛化优势的经验观察。

## 研究背景与动机

经典集成学习（bagging、boosting、随机森林等）在**欠参数化**模型上表现出色：通过聚合多个弱学习器获得更好的泛化和鲁棒性。然而近年来 empirical 研究发现，在**过参数化**神经网络上，这些直觉不再成立：

**泛化无优势**：Abe et al. (2022b) 发现深度集成（如 4 个 ResNet-18）的预测与单个更大模型（如 4× 宽度的 WideResNet-18）高度一致，无论分布内还是分布外数据

**经典多样性策略失效**：bagging 等增加成员多样性的方法对过参数化集成甚至有害（Nixon et al., 2020; Jeffares et al., 2024）

**不确定性量化可靠性存疑**：多项研究质疑深度集成的不确定性估计的可靠性（Abe et al., 2022b; Theisen et al., 2024）

本文的核心问题是：

- **Q1**：过参数化集成是否比单个大模型有泛化或鲁棒性优势？
- **Q2**：过参数化集成的预测方差是否对应传统的频率或贝叶斯不确定性？

## 方法详解

### 理论框架：随机特征（RF）回归模型

以随机特征线性回归器作为神经网络的可分析代理。RF 模型形式为：

$$h_{\mathcal{W}}(x) = \frac{1}{\sqrt{D}} \sum_{i=1}^{D} \phi(\omega_i, x) \theta_i$$

其中 $\omega_i \sim \pi(\cdot)$ 为随机采样的特征参数，$\theta_i$ 为可学习权重，$D$ 为特征数量（宽度）。过参数化意味着 $D > N$（特征数大于训练样本数）。

### 关键设计 1：最弱假设（Assumption 1）

与先前工作假设高斯随机特征不同，本文仅要求**亚指数（sub-exponential）**条件：

- 白化特征 $w_i w_{\perp i}$ 为亚指数分布（轻尾）
- 特征矩阵 $\Phi$ 几乎必然满秩

这一假设远弱于高斯性，能覆盖 ReLU、softplus 等实际激活函数（通过光滑逼近），使结论不依赖于具体的特征分布。

### 关键设计 2：无穷集成 = 无穷宽单模型（Theorem 1）

**核心定理**：在 Assumption 1 下，无穷集成的过参数化 RF 回归器 $\bar{h}_\infty^{(LN)}$ 与单个无穷宽 RF 回归器 $h_\infty^{(LN)}$ 逐点几乎必然等价：

$$\bar{h}_\infty^{(LN)}(x^*) = h_\infty^{(LN)}(x^*) = k_N(x^*)^\top K^{-1} y$$

即集成均值恰好等于核（ridgeless kernel）回归器。证明的关键引理是：

$$\mathbb{E}_{W, w_\perp}[w_\perp^\top W^\top (WW^\top)^{-1}] = 0$$

这一结果即使在 $w_\perp$ 和 $W$ **有依赖**时也成立，显著推广了此前需要独立性/高斯性的分析。

### 关键设计 3：有限参数预算下的非渐近分析（Theorem 2）

给定总计 $MD$ 个随机特征，比较 $M$ 个各用 $D$ 特征的集成 vs. 使用全部 $MD$ 特征的单模型：

$$\|h_{\mathcal{W}^*}^{(LN)}(\cdot) - \bar{h}_{\mathcal{W}_{1:M}}^{(LN)}(\cdot)\|_2^2 \leq O(\sqrt{\log(1/\delta)}) + O(1/D)$$

即在相同参数预算下，集成和单大模型的差异随单成员宽度 $D$ 增大而消失。

### 关键设计 4：集成方差的本质（Sec 3.3）

集成预测方差等于有限宽度与无穷宽度模型预测之差的期望平方：

$$\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(LN)}(x^*)] = \mathbb{E}_{\mathcal{W}}[(h_{\mathcal{W}}^{(LN)}(x^*) - h_\infty^{(LN)}(x^*))^2]$$

- **高斯特征**下：方差 $= r_\perp^2 \cdot \frac{\|h_\infty^{(LN)}\|_k^2}{D-N-1}$，其中 $r_\perp^2$ 恰好是 GP 后验方差，有贝叶斯解读
- **一般特征**下：方差依赖于 $W$ 和 $w_\perp$ 的复杂联合期望，**不等于** GP 后验方差的标量倍——即集成方差**没有**传统不确定性的含义

### 关键设计 5：小 Ridge 正则化的平滑过渡（Theorem 3）

在加入小 ridge 参数 $\lambda > 0$ 时，无穷集成与无穷宽单模型的差异关于 $\lambda$ 是 Lipschitz 连续的：

$$|\bar{h}_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*)| \leq C \cdot \lambda$$

常数 $C$ 与测试点 $x^*$ 无关（紧空间上），确保了实际使用小正则化时结论仍近似成立。

## 实验关键数据

### 主要验证实验

| 实验设置 | 数据 | 结果 |
|---------|------|------|
| RF 集成 vs 无穷宽 RF (Fig 1) | 合成数据, N=6 | M=10000 的集成与无穷宽模型预测无可感知差异 |
| 过参数化 vs 欠参数化 (Fig 2 左) | California Housing, N=12, softplus | D>N 时集成与大模型差异骤降（hockey stick 形状） |
| 神经网络集成 (Fig 2 右) | California Housing, N=12000, ReLU | 神经网络也呈现类似 hockey stick 模式 |
| 等参数预算比较 (Fig 3 左) | RF, N=12, ReLU, D=200/member | 集成与单模型泛化误差随总参数量几乎相同 |
| 等参数预算 NN (Fig 3 右) | 3层MLP, N=12000, width=256 | 深度集成与更大单模型泛化性能几乎一致 |

### 不确定性量化实验（Fig 4）

| 方差类型 | N=6, D=200, ReLU | 结论 |
|---------|-------------------|------|
| RF 集成方差 | 空间分布不均匀 | 与模型容量敏感度相关 |
| GP 后验方差 | 经典的远离数据点处大 | 反映数据覆盖的不确定性 |
| 两者差异 | **显著不同** | 集成方差≠贝叶斯不确定性 |

### 小 Ridge 平滑性实验（Fig 5）

California Housing, N=12, D=200, ReLU：集成-单模型差异随 $\lambda$ 线性增长，500 个测试点均表现出良好的 Lipschitz 连续性，验证了 Theorem 3。

## 亮点与洞察

1. **最弱假设下的强等价**：仅需亚指数条件即证明集成=大模型等价，不依赖高斯假设或渐近分析，大大扩展了理论适用范围
2. **精确揭示集成方差本质**：方差度量的是"增加容量的预期效果"而非不确定性——这是对 Abe et al. (2022b) 经验发现的首个理论解释
3. **高斯性的双面作用**：在高斯特征下集成方差恰好等于 GP 后验方差，产生了不确定性量化"有效"的假象；放弃高斯假设后这一联系断裂
4. **非渐近 + 渐近双重保证**：Theorem 2 给出有限参数时的高概率界，Theorem 1 覆盖渐近极限
5. **欠参数化 vs 过参数化的清晰对比**：宽度在欠参数化时控制隐式 ridge 参数，在过参数化时对集成预测无影响——hockey stick 图形直观展示了这一转变

## 局限与展望

1. **RF 模型 vs 真实神经网络**：RF 仅训练最后一层、无特征学习，无法完全解释 NN 行为（尽管实验展示了类似趋势）
2. **仅回归任务**：理论分析局限于回归设定，分类任务的推广尚不明确
3. **ReLU 的技术限制**：ReLU 不满足满秩假设（需 softplus 等光滑逼近），严格结论需通过极限论证间接获得
4. **集成实际价值未完全否定**：论文不否认集成在实践中有用（如并行训练、超参数搜索），仅指出其优势可被更大单模型复现
5. **Ridge 正则化的精确界**：Theorem 3 给出 Lipschitz 连续性但未给出显式常数，实际中多大的 $\lambda$ 使等价性崩溃仍需 case-by-case 分析

## 相关工作与启发

- **Deep Ensembles Work, But Are They Necessary?** (Abe et al., NeurIPS 2022)：本文的主要经验动机，发现集成方差与容量敏感度强相关
- **Jacot et al. (2020)**：高斯 RF 下的集成方差分析，本文显著放松了其假设
- **Adlam & Pennington (2020)**：过参数化 RF 的高维渐近分析，依赖高斯普适性
- **Ruben et al. (2024)**：并行工作，也发现 RF 集成无优势，但使用最优 ridge 调优和高斯假设
- **Neal (1996), Williams (1996)**：无穷宽 NN 与 GP 的经典联系，本文在此基础上区分了集成与 GP

## 评分

- 新颖性: ⭐⭐⭐⭐ — 在最弱假设下统一了集成-单模型等价理论，揭示了集成方差的非不确定性本质
- 实验充分度: ⭐⭐⭐⭐ — RF 和 NN 实验互补验证，多种激活函数和数据集，但缺少大规模NN实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论陈述精确，直觉解释清晰，图示精美，hockey stick 图尤为直观
- 价值: ⭐⭐⭐⭐ — 为"大模型时代是否还需要集成"提供了坚实的理论基础，对不确定性量化实践有重要警示

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](on_expressive_power_of_looped_transformers_theoretical_analysis_and_enhancement_.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](../../ACL2025/llm_nlp/argument_mining_in_the_age_of_large_language_models.md)
- [\[CVPR 2026\] Sign Language Recognition in the Age of LLMs](../../CVPR2026/llm_nlp/sign_language_recognition_llms.md)
- [\[ACL 2025\] Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers](../../ACL2025/llm_nlp/limitgen-llms-identify-research-limitations.md)
- [\[ICML 2025\] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)

<!-- RELATED:END -->
