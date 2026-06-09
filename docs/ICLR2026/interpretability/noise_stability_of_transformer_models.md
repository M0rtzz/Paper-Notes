---
title: >-
  [论文解读] Noise Stability of Transformer Models
description: >-
  [ICLR 2026][可解释性][noise stability] 提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%…
tags:
  - "ICLR 2026"
  - "可解释性"
  - "noise stability"
  - "simplicity bias"
  - "Transformer"
  - "grokking"
  - "Fourier analysis"
  - "regularization"
  - "Boolean function analysis"
---

# Noise Stability of Transformer Models

**会议**: ICLR 2026  
**arXiv**: [2602.08287](https://arxiv.org/abs/2602.08287)  
**代码**: 未公开  
**领域**: 可解释性  
**关键词**: noise stability, simplicity bias, Transformer, grokking, Fourier analysis, regularization, Boolean function analysis

## 一句话总结

提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%。

## 研究背景与动机

深度学习中的简单性偏差（simplicity bias）是理解模型泛化、可解释性和鲁棒性的核心概念。神经网络倾向于收敛到能解释训练数据的最简函数。量化这种"简单性"的传统度量源自布尔函数分析中的**平均敏感度（average sensitivity）**，即模型输出对单个 token 扰动的期望变化。

先前工作表明 Transformer 学到的函数比 LSTM 的敏感度更低（Bhattamishra et al., 2022），且 Transformer 难以学习高敏感度函数如 Parity（Hahn 2020）。Vasudeva et al.（2024）将平均敏感度与 grokking 现象联系起来。

然而，作者指出平均敏感度存在两个关键缺陷：

**理论缺陷**：布尔域上的定义难以自然推广到实值域，基于超网格的扩展方法笨拙且采样不切实际

**实证缺陷**：未能解释 GPT-2、Gemma、RoBERTa 等现代 LLM 中观察到的"junta-like"输入依赖现象——输出仅依赖于极小子集的输入 token（实验中 256 个 token 仅 5-10 个有显著影响），而 Friedgut 定理的上界预测高达 1024 个，差距极大

## 方法详解

### 整体框架

本文提出用**噪声稳定性（noise stability）**替代平均敏感度。与平均敏感度逐个扰动不同，噪声稳定性衡量函数对**同时施加于所有输入坐标的关联噪声**的鲁棒性。这一概念可通过实值域上的 Ornstein-Uhlenbeck 半群自然推广。

### 关键设计

**1. 噪声稳定性的形式化定义：用关联噪声一次性替代逐 token 扰动。**

平均敏感度的麻烦在于它逐个翻转 token、又只在布尔域上自然成立。噪声稳定性换了个思路：不再单独扰动每个坐标，而是给所有输入坐标同时加上**一份关联的高斯噪声**，看函数输出还剩多少相关性。形式上，对高斯测度 $\gamma$ 下的函数 $f \in L^2(\gamma)$，先构造关联对 $(X,Y)$，再取两者输出的内积期望：

$$\text{Stab}_\rho(f) := \mathbb{E}_{(X,Y)}[f(X) f(Y)]$$

其中 $Y = \rho X + Z\sqrt{1-\rho^2}$，$Z \sim \gamma$ 独立于 $X$，相关系数 $\rho \in (0,1)$ 控制噪声强度。这个定义天然活在实值域上（靠 Ornstein-Uhlenbeck 半群），不需要平均敏感度那套笨拙的超网格采样。更关键的是，它通过 Hermite-Fourier 系数和频谱直接挂钩——高阶项被 $\rho^{|\alpha|}$ 指数压低：

$$\text{Stab}_\rho(f) = \sum_{\alpha \in \mathbb{N}^d} \rho^{|\alpha|} \tilde{f}(\alpha)^2$$

**2. 谱集中引理（Lemma 1）：把"稳定"翻译成"低频主导"。**

光有定义还不够，得证明高稳定性确实对应简单函数。这条引理给出桥梁：只要噪声稳定性接近函数总能量，Fourier 质量就必然堆在低阶系数上。具体地，若 $\text{Stab}_\rho(f) \geq (1-\delta)\|f\|_2^2$，则 $f$ 是 $(\varepsilon, T)$-谱集中的，截断阶数满足

$$T \geq \log_{1/\rho}\left(1 - \frac{\delta}{\varepsilon}\right)$$

这正是后面在 GPT-2、RoBERTa 等模型上算"度数 ≥15 的 Fourier 尾部质量上界"的理论依据——稳定性越高，尾部被压得越狠，上界越紧。

**3. 单层 ReLU MLP 的噪声稳定性（Theorem 5.1）：算清楚一层非线性吃掉多少稳定性。**

要分析整个 Transformer，先得知道单个组件怎么传播稳定性。对 $\rho$-关联的高斯输入 $(X,Y)$，ReLU 的输出内积有闭式解：

$$\mathbb{E}[\text{ReLU}(X) \cdot \text{ReLU}(Y)] = \frac{1}{2\pi}\left(\sqrt{1-\rho^2} + \rho(\pi - \arccos\rho)\right)$$

做二阶 Taylor 展开是 $\approx \frac{1}{2\pi} + \frac{1}{4}\rho + \frac{1}{4\pi}\rho^2$，主项随 $\rho$ 近似线性。这说明一层 ReLU 不会把关联噪声彻底抹平，而是按一个可控的比例往下传。

**4. 单层注意力层的噪声稳定性：按 $W=W_Q W_K^T$ 的结构分三种情况。**

注意力比 MLP 复杂，因为稳定性取决于 query-key 矩阵 $W = W_Q W_K^T$ 长什么样，论文按结构拆成三档讨论。**恒等矩阵 $W=I_d$**（Theorem 5.2）下，高维极限里注意力矩阵收敛到 $I_n$，稳定性与 $\rho$ 保持线性关系，付出的代价仅 $o(1)$；**低秩矩阵 $W=UU^T$** 则通过 Johnson-Lindenstrauss 变换归约回恒等情况，结论一致；而**非结构化 $W \sim \mathcal{N}(0,I)$**（Theorem 5.3）是最坏情形——注意力矩阵趋向随机排列矩阵，稳定性退化为 $\rho \cdot s(\rho) \cdot \|(W_V)_{:,j}\|_2^2$，多出来的 $s(\rho)$ 是注意力模式被保持的概率。换句话说，结构化的注意力几乎无损地传递稳定性，随机注意力才会额外衰减。

**5. 多层传播分析：把单层结论递推成整网的固定点。**

有了单层的传播率，多层就是反复迭代。ReLU FFN 堆叠时，第 $L$ 层的稳定性由前一层递推决定：

$$\rho_L = \frac{1}{2\pi}\left(\sqrt{1-\rho_{L-1}^2} + \rho_{L-1}(\pi - \arccos\rho_{L-1})\right)$$

做线性近似求解这个不动点方程，得到 $\frac{2}{3\pi} \approx 0.212$。这个非零固定点是个好消息：深层网络对稳定性只造成**弱衰减**，信号收敛到一个有限下界而不会被层数彻底吃光——这也解释了为什么深 Transformer 仍能维持 junta-like 的低敏感行为。

### 损失函数

噪声稳定性正则化器（$S=1$ 鼓励稳定性）：

$$R_{M,S,\rho}(X) = (-1)^S \cdot \sum_{i=1}^C M(X)_i \cdot M(Y)_i$$

其中 $Y_i$ 以概率 $\frac{1+\rho}{2}$ 保持为 $X_i$，否则从 $\text{uniform}([U])$ 采样。

正则化损失：$\ell_{\text{reg}}(M,X) = \ell(M,X) + \gamma \cdot R_{M,S,\rho}(X)$

仅需每次迭代额外一次前向传播，计算开销极低。

## 实验关键数据

### 主实验

**谱集中上界对比（n=256, 度数 ≥15 的 Fourier 尾部质量）**：

| 模型 | 平均敏感度上界 | 噪声稳定性上界 |
|------|---------------|---------------|
| GPT-2 | 0.003 | **0.0005** |
| BERT | 0.04 | **0.02** |
| RoBERTa | 0.19 | **0.02** |
| Gemma | 0.043 | **0.0157** |

噪声稳定性在所有模型上都给出更紧的 Fourier 尾部质量估计（6× 到 9.5× 的改进）。

**Grokking 加速效果**：

| 任务 | 超参数 (γ, ρ) | 无正则化收敛步数 | 有正则化收敛步数 | 加速比 |
|------|--------------|-----------------|-----------------|--------|
| 模加法 (K=113) | (0.75, 0.25) | ~4500 | ~3300 | **36%** |
| 噪声 k-sparse parity | (0.05, 0.05) | 基线 | 加速 | **~35%** |
| WikiText-2 NTP | - | 基线 | 加速 | **~75%** |

### 消融实验

- **LLM 的 junta-like 特性**：在 256 token 输入上，GPT-2/RoBERTa/Gemma 仅 5-10 个 token 具有显著几何影响力，远少于 Friedgut 定理预测的上界 1024 个
- **位置偏差**：首尾 token 一致地具有最高影响力，与 KV Cache 压缩文献中"attention sinks"的观察一致
- **训练动态监控**：在 noisy sparse parity 任务中，Transformer 的噪声稳定性在训练过程中自然下降以匹配目标函数，稳定性变化是泛化的先行指标
- **WikiText-2 语言建模**：正则化模型的噪声稳定性保持高位，而未正则化模型变得越来越不稳定

### 关键发现

1. 噪声稳定性比平均敏感度能更精确地刻画 Transformer 的谱集中（所有模型均给出更紧上界）
2. ReLU MLP 层对稳定性产生弱衰减（收敛到固定点 $2/(3\pi)$），而非完全消除信号
3. 注意力层在恒等/低秩 $W$ 下保持稳定性（线性关系），在非结构化 $W$ 下引入额外衰减因子 $s(\rho)$
4. 噪声稳定性正则化是 grokking 的催化剂，在多种任务上一致地加速训练

## 亮点与洞察

1. **统一理论框架**：通过 Ornstein-Uhlenbeck 半群将布尔域分析自然推广到实值域，保留了与函数频谱的严格联系，比几何影响力更具分析力
2. **跨领域桥接**：建立了信号传播（C-maps/Q-maps）与简单性偏差/可解释性之间的新连接——噪声稳定性可视为相关性映射的更简洁类比
3. **实用正则化**：仅需一次额外前向传播的低成本正则化方法，75% 的 NTP 训练加速极具实用价值
4. **LLM 内部结构洞察**：量化了 GPT-2 等模型的 junta-like 依赖（仅 5-10 个 token 具有显著影响），为 KV cache 压缩、token 剪枝提供理论支撑
5. **训练监控新指标**：噪声稳定性的变化可作为 grokking 的先行信号

## 局限性

1. 理论分析中省略了残差连接、层归一化、注意力掩码等实际 Transformer 组件
2. 语言建模实验仅在小规模 WikiText-2 上进行，缺乏亿级参数 LLM 上的验证
3. 多层 Transformer 的稳定性区间传播的实际紧致度尚未充分验证
4. 正则化超参数 $(\gamma, \rho)$ 需要针对不同任务调优（模加法用 (0.75,0.25)，parity 用 (0.05,0.05)）
5. 未探讨噪声稳定性与对抗鲁棒性之间的定量关系

## 相关工作与启发

与 Vasudeva et al.（2024）使用平均敏感度追踪 grokking 不同，本文的噪声稳定性提供了更强的谱集中保证。与 Hua et al.（2023）的 Transformer 微调噪声稳定性方法在动机（简单性偏差 vs 微调稳定性）、应用范围和关联噪声定义上都有根本区别。最直接的启发来自 Li & Mossel（2025）的层级函数噪声敏感度分析。

**启发**：噪声稳定性可以作为训练监控指标——稳定性下降往往预示 grokking 即将发生，为自适应训练策略提供新思路。此外，junta-like 依赖的量化分析为提示词工程中"哪些 token 真正重要"提供了理论依据。

## 评分

- 新颖性: ⭐⭐⭐⭐ (将信号传播与简单性偏差统一的视角很新颖，理论分析完善)
- 实验充分度: ⭐⭐⭐ (理论扎实但实验规模偏小，缺乏大模型验证)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，行文流畅，图表直观)
- 价值: ⭐⭐⭐⭐ (为理解 Transformer 内部机制提供了新工具，正则化方法有实用潜力)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](../../ICML2026/interpretability/certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](../../NeurIPS2025/interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](../../AAAI2026/interpretability/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)

</div>

<!-- RELATED:END -->
