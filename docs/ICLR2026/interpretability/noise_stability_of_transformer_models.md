---
title: >-
  [论文解读] Noise Stability of Transformer Models
description: >-
  [ICLR 2026][noise stability] 提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%。
tags:
  - ICLR 2026
  - noise stability
  - simplicity bias
  - Transformer
  - grokking
  - Fourier analysis
  - regularization
  - Boolean function analysis
---

# Noise Stability of Transformer Models

**会议**: ICLR 2026  
**arXiv**: [2602.08287](https://arxiv.org/abs/2602.08287)  
**代码**: 未公开  
**领域**: llm_nlp  
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

#### 1. 噪声稳定性的形式化定义

对于高斯测度 $\gamma$ 下的函数 $f \in L^2(\gamma)$，关联对 $(X,Y)$ 通过向 $X$ 添加缩放的高斯噪声生成：

$$\text{Stab}_\rho(f) := \mathbb{E}_{(X,Y)}[f(X) f(Y)]$$

其中 $Y = \rho X + Z\sqrt{1-\rho^2}$，$Z \sim \gamma$ 独立于 $X$，$\rho \in (0,1)$ 为相关系数。

通过 Hermite-Fourier 系数直接关联频谱：

$$\text{Stab}_\rho(f) = \sum_{\alpha \in \mathbb{N}^d} \rho^{|\alpha|} \tilde{f}(\alpha)^2$$

#### 2. 谱集中引理（Lemma 1）

高噪声稳定性意味着 Fourier 质量集中在低阶系数：若 $\text{Stab}_\rho(f) \geq (1-\delta)\|f\|_2^2$，则 $f$ 是 $(\varepsilon, T)$-谱集中的，其中：

$$T \geq \log_{1/\rho}\left(1 - \frac{\delta}{\varepsilon}\right)$$

#### 3. 单层 ReLU MLP 的噪声稳定性（Theorem 5.1）

对于 $\rho$-关联高斯输入 $(X,Y)$：

$$\mathbb{E}[\text{ReLU}(X) \cdot \text{ReLU}(Y)] = \frac{1}{2\pi}\left(\sqrt{1-\rho^2} + \rho(\pi - \arccos\rho)\right)$$

二阶 Taylor 近似：$\approx \frac{1}{2\pi} + \frac{1}{4}\rho + \frac{1}{4\pi}\rho^2$

#### 4. 单层注意力层的噪声稳定性

分三种情况分析 $W = W_Q W_K^T$：

- **恒等矩阵 $W=I_d$**（Theorem 5.2）：高维极限下注意力矩阵收敛到 $I_n$，稳定性与 $\rho$ 呈线性关系，代价为 $o(1)$
- **低秩矩阵 $W=UU^T$**：通过 Johnson-Lindenstrauss 变换归约到恒等情况
- **非结构化 $W \sim \mathcal{N}(0,I)$**（Theorem 5.3）：注意力矩阵趋向随机排列矩阵，稳定性为 $\rho \cdot s(\rho) \cdot \|(W_V)_{:,j}\|_2^2$，其中 $s(\rho)$ 是注意力模式保持概率

#### 5. 多层传播分析

ReLU FFN 中稳定性按递推关系传播：

$$\rho_L = \frac{1}{2\pi}\left(\sqrt{1-\rho_{L-1}^2} + \rho_{L-1}(\pi - \arccos\rho_{L-1})\right)$$

线性近似求解得固定点 $\frac{2}{3\pi} \approx 0.212$，表现为**弱衰减**——稳定性不会完全消失。

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

## 相关论文

- [Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)
- [Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](../../NeurIPS2025/interpretability/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)
- [AdaptGrad: Adaptive Sampling to Reduce Noise](../../NeurIPS2025/interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)
- [Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](../../NeurIPS2025/interpretability/bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)

<!-- RELATED:END -->
