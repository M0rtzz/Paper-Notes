---
title: >-
  [论文解读] On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding
description: >-
  [ICML2025][LLM/NLP][Transformer] 本文首次建立了 Looped Transformer 关于循环次数和目标函数连续性模的逼近速率理论，揭示了循环架构特有的逼近误差来源（上下文连续性与 token 连续性），并提出 Timestep-Modulated Looped Transformer (TMLT) 通过时间步编码消除该限制，在推理、上下文学习和语言建模任务上取得一致提升。
tags:
  - ICML2025
  - LLM/NLP
  - Transformer
  - 表达能力
  - 逼近速率
  - 权重共享
  - 时间步编码
  - 通用逼近
  - 连续性模
---

# On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding

**会议**: ICML2025  
**arXiv**: [2410.01405](https://arxiv.org/abs/2410.01405)  
**代码**: [kevin671/tmlt](https://github.com/kevin671/tmlt)  
**领域**: LLM/NLP  
**关键词**: Looped Transformer, 表达能力, 逼近速率, 权重共享, 时间步编码, 通用逼近, 连续性模

## 一句话总结

本文首次建立了 Looped Transformer 关于循环次数和目标函数连续性模的逼近速率理论，揭示了循环架构特有的逼近误差来源（上下文连续性与 token 连续性），并提出 Timestep-Modulated Looped Transformer (TMLT) 通过时间步编码消除该限制，在推理、上下文学习和语言建模任务上取得一致提升。

## 研究背景与动机

- **Looped Transformer** 将固定大小的 Transformer 层输出递归反馈回输入，通过权重共享实现参数高效，同时递归结构赋予其模拟迭代算法的能力
- 标准 Transformer 的通用逼近性质已被广泛研究（Yun et al., 2020; Kajitsuka & Sato, 2024），但由于权重绑定约束，这些结论**不能直接推广**到 Looped Transformer
- 权重绑定 ReLU 网络的逼近速率仅近期被建立（Zhang et al., 2023），而 Looped Transformer 的逼近速率仍然未知
- **核心问题**：Looped Transformer 能否计算上下文映射（contextual mapping）？是否是通用逼近器？其逼近速率由什么决定？

## 方法详解

### 理论框架：三种连续性模

本文的核心贡献是定义了三种连续性模（modulus of continuity），共同决定 Looped Transformer 的逼近速率：

**1. 序列连续性模（Sequence Continuity）**：度量输入序列整体扰动对输出的影响

$$\omega_f(\delta) \coloneqq \sup\{\|f(\mathbf{X}) - f(\mathbf{X}')\|_p : \|\mathbf{X} - \mathbf{X}'\|_2 \leq \delta\}$$

**2. 上下文连续性模（Contextual Continuity）**：固定某个 token 不变，度量其他 token 扰动对该 token 输出的影响（类比同一个词"write"在不同句子"I write papers" vs "You write books"中的变化）

$$\omega_f^{\text{cont}}(\delta) \coloneqq \sup_{n,\mathbf{X},\mathbf{X}'}\{\|f(\mathbf{X})_{:,n} - f(\mathbf{X}')_{:,n}\|_p : \|\mathbf{X}-\mathbf{X}'\|_2 \leq \delta,\ \mathbf{X}_{:,n}=\mathbf{X}'_{:,n}\}$$

**3. Token 连续性模（Token Continuity）**：固定上下文不变，度量单个 token 扰动对其输出的影响（类比"I write papers" vs "I draft papers"）

$$\omega_f^{\text{tok}}(\delta) \coloneqq \sup_{n,\mathbf{X},\mathbf{X}'}\{\|f(\mathbf{X})_{:,n} - f(\mathbf{X}')_{:,n}\|_p : \|\mathbf{X}_{:,n}-\mathbf{X}'_{:,n}\|_2 \leq \delta,\ \mathbf{X}_{:,q}=\mathbf{X}'_{:,q}\ (\forall q \neq n)\}$$

### 主定理（Theorem 3.6）

给定置换等变连续函数 $f$，当循环次数 $r > N$ 时，存在一个仅含 2 个注意力头、head size 1 的单层 Looped Transformer，使得：

$$\|\mathcal{L}_2 \circ \text{TF}^{\circ r} \circ \mathcal{L}_1 - f\|_{L^p} \leq (Nd)^{1/p}(\omega_f^{\text{tok}}(\delta\sqrt{d}) + \omega_f^{\text{cont}}(\delta\sqrt{Nd})) + \omega_f(\delta\sqrt{Nd}) + \text{低阶项}$$

其中 $\delta = ((r-N)/2)^{-1/((N+1)d+1)}$，参数量为 $O(d)$，**与逼近精度和序列长度无关**。

### 证明三步骤

1. **Token-wise 量化**：FFN 将每个 token $\mathbf{X}_{:,n} \in [0,1]^d$ 映射到 token ID $z \in \{0,\ldots,\delta^{-d}-1\}$
2. **上下文映射**：利用 $N$ 次循环，通过内积 $\mathbf{u}^\top \mathbf{z}$ 计算序列 ID，保证不同序列有不同 ID
3. **函数值映射**：$K-1$ 次循环逐步将上下文 token ID 映射到目标嵌入

### TMLT：引入时间步编码

理论分析揭示 Looped Transformer 的局限：权重绑定 FFN 的逼近误差由相邻上下文 token 嵌入的最大差异决定（Lemma 4.1），无法精确记忆目标值。

**解决方案**：为每次循环引入依赖于时间步 $t$ 的缩放参数：

$$\text{FF}(\mathbf{X}) \to \boldsymbol{\eta}(t) \odot \text{FF}(\mathbf{X})$$

时间步编码通过频率嵌入 + 两层 MLP（SiLU 激活）生成，条件化 RMSNorm 的增益参数和残差缩放系数：

$$\boldsymbol{\alpha}_1(t), \boldsymbol{\alpha}_2(t), \boldsymbol{\gamma}_1(t), \boldsymbol{\gamma}_2(t) = \mathbf{W}_5 \cdot \text{SiLU}(\text{TE}(t)) + \mathbf{b}_5$$

Theorem 4.2 证明：加入时间步编码后，模型可以**精确记忆**目标值（误差为 0），消除了额外的连续性模依赖。

## 实验关键数据

### 推理任务（Table 2）

| 任务 | TF (L=6) | Looped r=32 | TMLT r=32 |
|------|----------|-------------|-----------|
| Sudoku | 0.0 | 87.9 | **90.2** |
| Countdown | 53.8 | 88.1 | **90.5** |

| 任务 | TF (L=12) | Looped r=100 | TMLT r=100 |
|------|-----------|--------------|------------|
| LCS (100) | 39.8 | 98.2 | **98.6** |
| ED (60) | 41.4 | 47.7 | **88.3** |

- ED(60) 任务上 TMLT 的提升最为显著：47.7 → 88.3（+40.6），印证了理论预测

### 上下文学习（Table 3）

| 模型 | MSE ↓ |
|------|-------|
| TF L=12 | 8.6e-03 |
| Looped r=12 | 1.4e-02 |
| TMLT r=12 | **1.7e-03** |

### 语言建模 WikiText-103（Table 4）

| 模型 | 训练困惑度 | 测试困惑度 |
|------|-----------|-----------|
| TF L=12 | 15.9 | 20.5 |
| Looped r=24 | 17.1 | 20.6 |
| TMLT r=24 | **15.9** | **19.6** |

### 关键发现

- 增加循环次数 $r$ 一致提升性能，验证了理论逼近速率
- Timestep encoding 在所有任务上带来额外增益，尤其在函数值变化剧烈（高连续性模）的任务上提升显著
- Looped TF 可在参数量远少于标准 TF 的情况下达到甚至超越其性能

## 亮点与洞察

1. **首次建立 Looped Transformer 的逼近速率**，填补了理论空白，且结论对参数量仅依赖于输入维度 $O(d)$，不依赖逼近精度和序列长度
2. **三种连续性模的定义极具创新性**——序列连续性、上下文连续性、token 连续性精准刻画了序列到序列函数的不同变化模式
3. **理论驱动实践**：从逼近速率分析中自然得出 Looped TF 的局限（额外连续性模依赖），并针对性地提出 TMLT 解决方案
4. 证明了即使使用 hardmax 的权重绑定自注意力机制，Looped Transformer 也能计算上下文映射
5. TMLT 的设计借鉴了扩散模型中的 adaptive instance normalization（DiT），跨领域迁移优雅

## 局限与展望

- 理论分析仅针对**单层** Looped Transformer，多层情况的逼近速率未探讨
- 所有分析限于**固定长度**输入，未涉及变长序列的长度泛化问题
- 时间步编码引入了额外参数（频率嵌入 MLP + 缩放参数生成器），虽然相对于整体参数量开销不大，但增加了实现复杂度
- 缺少对最优记忆容量的刻画——模型能精确记忆多少样本？
- 推理任务的训练样本量非常大（百万级），实际应用中数据效率有待验证

## 相关工作与启发

- **Yun et al. (2020)**：标准 Transformer 的通用逼近定理，本文将其推广到 Looped 设定
- **Zhang et al. (2023)**：权重绑定 ReLU 网络的逼近速率，本文在此基础上处理了上下文映射的额外挑战
- **Dehghani et al. (2019)**：Universal Transformer，最早引入递归结构
- **Saunshi et al. (2025)**：证明 Looped TF 对推理任务有归纳偏置
- **Peebles & Xie (2023)**：DiT 中的 adaptive instance normalization，启发了 TMLT 的时间步条件化设计
- **Bae et al. (2025)**：Relaxed weight-tying (层级 LoRA)，为另一种缓解权重绑定限制的方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次建立 Looped Transformer 逼近速率，三种连续性模定义非常新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖推理/ICL/LM 三类任务，但规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 理论与实践衔接自然，证明结构清晰
- 价值: ⭐⭐⭐⭐ — 对理解 Looped/权重共享架构有重要理论意义，TMLT 实用性有潜力

<!-- RELATED:START -->

## 相关论文

- [Theoretical Limitations of Ensembles in the Age of Overparameterization](theoretical_limitations_of_ensembles_in_the_age_of_overparameterization.md)
- [SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](../../CVPR2025/llm_nlp/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)
- [Interactive and Expressive Code-Augmented Planning with Large Language Models](../../ACL2025/llm_nlp/interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [Theory of Mind in Large Language Models: Assessment and Enhancement](../../ACL2025/llm_nlp/theory_of_mind_llm.md)

<!-- RELATED:END -->
