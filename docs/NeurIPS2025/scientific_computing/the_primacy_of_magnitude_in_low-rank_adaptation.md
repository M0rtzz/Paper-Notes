---
title: >-
  [论文解读] The Primacy of Magnitude in Low-Rank Adaptation
description: >-
  [NeurIPS 2025][科学计算][LoRA] 揭示 LoRA 中权重更新幅度（magnitude）是性能的根本驱动因素，统一了学习率、缩放因子和初始化策略对 LoRA 的影响机制，并提出 LoRAM——一种基于确定性正交基和幅度缩放的高效初始化方法，无需 SVD 即可匹敌甚至超越谱初始化方法。
tags:
  - NeurIPS 2025
  - 科学计算
  - LoRA
  - 参数高效微调
  - 初始化策略
  - 权重更新幅度
  - 低秩适配
---

# The Primacy of Magnitude in Low-Rank Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2507.06558](https://arxiv.org/abs/2507.06558)  
**代码**: [GitHub](https://github.com/zhangzicheng-jd/LoRAM)  
**领域**: Model Compression / Parameter-Efficient Fine-Tuning  
**关键词**: LoRA, 参数高效微调, 初始化策略, 权重更新幅度, 低秩适配

## 一句话总结

揭示 LoRA 中权重更新幅度（magnitude）是性能的根本驱动因素，统一了学习率、缩放因子和初始化策略对 LoRA 的影响机制，并提出 LoRAM——一种基于确定性正交基和幅度缩放的高效初始化方法，无需 SVD 即可匹敌甚至超越谱初始化方法。

## 研究背景与动机

**领域现状**：LoRA 是最流行的参数高效微调方法，通过注入可训练的低秩矩阵 $B \in \mathbb{R}^{n \times r}$、$A \in \mathbb{R}^{r \times m}$，仅需更新 <1% 参数就能微调大模型。近年来，PiSSA、MiLoRA、OLoRA 等基于谱分解（SVD）的初始化方法显著提升了收敛速度和性能。

**现有痛点**：
   - **效率损失**：谱初始化需要对预训练权重做 SVD 分解，带来额外计算和存储开销，在资源受限场景下（如量化 LoRA、联邦学习）不实用
   - **理解不足**：谱初始化的成功通常被归因于"保留主成分中的知识"，但这个直觉缺乏理论根据——LoRA 的非凸优化使得训练动态难以预测

**核心矛盾**：谱初始化方法效果好但代价高，且成功机理不清。能否在不做 SVD 的情况下获得同等效果？

**本文目标** (a) 揭示谱初始化真正起作用的机制；(b) 设计一种无需 SVD 的高效替代方案。

**切入角度**：从**权重更新幅度**（weight update magnitude）的视角出发，$\nu[W_{\text{LoRA}}] = \frac{1}{mn}\|W_{\text{LoRA}}\|_F^2$，分析 LoRA 训练动态中各超参数如何通过幅度影响性能。

**核心 idea**：谱初始化的本质不是"知识保留"而是"幅度放大"，用确定性正交基 + 从预训练权重统计量推导的缩放因子即可复现其效果。

## 方法详解

### 整体框架

分析框架（Magnitude Principle）→ 揭示机制（Demystifying Spectral Gains）→ 高效方案（LoRAM）

### 关键设计

#### 1. 幅度原理（Magnitude Principle）

- **功能**：建立权重更新幅度作为分析 LoRA 训练动态的统一框架
- **核心思路**：LoRA 的权重更新幅度为 $\nu[\Delta W_{\text{LoRA}}^{(t)}] \approx r\alpha^2\eta^2(\nu[B^{(t)}]\nu[\nabla_A L^{(t)}] + \nu[\nabla_B L^{(t)}]\nu[A^{(t)}])$，受到学习率 $\eta$、缩放因子 $\alpha$、初始化幅度的共同控制
- **关键定理**（Proposition 1 - Parameter Scaling Equivalence）：证明了 $\alpha$、初始化幅度和学习率之间存在精确的等价关系——增大 $\alpha$ 等价于增大初始化幅度或调整学习率，三者本质上都是在调控更新幅度

#### 2. 低秩结构的幅度限制（Proposition 2）

- **功能**：证明 LoRA 的低秩结构天然地限制了更新幅度
- **核心发现**：$\nu[W_{\text{LoRA}}^{(t)}] \approx k_1 \gamma t$，其中 $k_1 = r(m\sigma_A^4 + n\sigma_B^4)$。标准 "Noise & Zeros" 初始化给出 $k_1 = r/m$，远小于全参数微调
- **设计动机**：这解释了为什么 LoRA 收敛慢——低秩结构导致更新幅度比全参数方法小数量级。任何放大 $k_1$ 的方法都能改善 LoRA

#### 3. 谱初始化的幅度增益分析

- **功能**：揭示 PiSSA 等谱初始化方法的真正机制
- **核心思路**：PiSSA 用 $A^{(0)} = \sqrt{S_r} V_{:,:r}^\top$，$B^{(0)} = U_{:,:r} \sqrt{S_r}$ 初始化。定义谱集中因子 $\rho[r] = \mathbb{E}_r[s]^2 / \mathbb{E}_{\mathcal{R}[W]}[s^2]$，则 PiSSA 的 $k_1 = Q[r](m+n)\nu[W]$，其中 $Q[r] = \rho[r] \cdot r / \mathcal{R}[W]$ 为"谱增益因子"
- **关键论证**：谱初始化的核心不是让 LoRA 的基方向对齐主成分（"知识保留"），而是通过奇异值缩放放大了更新幅度。用 tracking mode 实验验证——只要匹配了谱初始化的幅度，用任意正交基都能获得相近性能

#### 4. LoRAM 初始化

- **功能**：设计无需 SVD 的高效初始化方案
- **核心思路**：
    - 用离散正弦变换（DST）基 $\Phi_m$ 作为确定性正交基，$\Phi_m[i,j] = \sqrt{2/(m+1)} \sin((i+1)(j+1)\pi/(m+1))$
    - 用对数近似谱增益因子 $Q[r] \approx \log_{\min(n,m)}(r)$
    - 缩放因子 $\beta = (Q[r] \cdot \nu[W] / \nu[\Phi_n \Phi_m^\top])^{1/4}$
    - 初始化：$A^{(0)} = \beta \cdot \Phi_m^\top$，$B^{(0)} = \beta \cdot \Phi_n$，$W \leftarrow W - \beta^2 \cdot \Phi_n \Phi_m^\top$
- **设计动机**：DST 基是解析定义的，无需存储（不同设备可复现）；对数近似有效捕获了 $Q[r]$ 的单调递增和凹性特征

### 损失函数 / 训练策略

- LoRAM 仅修改初始化，不改变训练过程，完全兼容标准 LoRA 训练 pipeline
- 可与 RsLoRA（$\alpha = \sqrt{r}$）组合使用进一步提升
- 初始化后吸收 $B^{(0)}A^{(0)}$ 到冻结权重 $W$ 中

## 实验关键数据

### 主实验

#### NLG 任务（LLaMA-2-7B，Table 1）

| 方法 | GSM8K (r=16) | MATH (r=16) | HumanEval (r=16) | GSM8K (r=128) | MATH (r=128) |
|------|-------------|-------------|-----------------|-------------|-------------|
| LoRA | 31.51 | 4.16 | 15.98 | 40.27 | 4.72 |
| RsLoRA | 39.04 | 4.94 | 18.85 | 50.38 | 7.32 |
| PiSSA | 37.68 | 5.16 | 18.37 | 51.48 | 7.04 |
| **LoRAM** | **40.32** | **5.30** | **18.92** | 51.12 | **7.25** |

#### NLU 任务（DeBERTa-v3-base，Table 2）

| 方法 | MRPC | CoLA | RTE | STS-B |
|------|------|------|-----|-------|
| LoRA | 84.06 | 63.56 | 50.18 | 87.20 |
| PiSSA | 89.21 | 65.06 | 74.36 | 88.90 |
| **LoRAM** | **89.95** | 65.53 | **74.72** | **89.93** |

#### 多模态任务（LLaVA，Table 3）

| 方法 | MME_Cog | MMMU | AI2D | ScienceQA |
|------|---------|------|------|-----------|
| LoRA | 278 | 0.331 | 0.557 | 0.684 |
| PiSSA | 311 | 0.344 | 0.564 | 0.686 |
| **LoRAM** | 308 | **0.350** | **0.571** | **0.700** |

### 消融实验（Table 4，LLaMA-2-7B NLG）

| 消融项 | r=16 GSM8K | r=128 GSM8K | 发现 |
|--------|-----------|-------------|------|
| $Q[r] = \log(r/2)$ | 40.1 | 50.7 | 稍低 |
| $Q[r] = \log(r)$（默认）| **40.3** | **51.1** | 最佳对数近似 |
| DST基（默认）| 40.3 | 51.1 | — |
| 随机正交基 | 36.3 | 50.2 | 基的选择影响有限 |
| Gaussian 基 | 35.8 | 49.8 | 正交性有一定帮助 |
| PiSSA tracking | 36.7 | 49.5 | 匹配幅度即可获得相近效果 |
| LoRAM + RsLoRA | **52.1** | **59.4** | 组合使用进一步提升 |

### 关键发现

1. **幅度是关键**：tracking mode 实验证实，只要匹配谱初始化的幅度，用 DST 基就能获得与 PiSSA 相近的性能
2. **基的选择影响有限**：DST vs 随机正交 vs Gaussian 差异很小，证明谱方向不是关键
3. **低秩时增益更显著**：$Q[r]$ 的凹性意味着 rank 越小，幅度放大的边际效益越大
4. **与 RsLoRA 组合**：LoRAM + RsLoRA 在多数任务上进一步提升，但高 rank 时过度放大可能有害
5. **收敛速度**：LoRAM 的训练损失曲线与 PiSSA 几乎一致，早期收敛更快

## 亮点与洞察

1. **深刻的统一视角**：将学习率、缩放因子、初始化三个看似独立的调参维度，统一到"幅度调控"这一个原理下
2. **破除迷信**：谱初始化的成功不是因为"保留知识方向"，而是简单的幅度放大——这是一个反直觉但有力的发现
3. **极简设计**：LoRAM 的实现只需几行代码（生成 DST 基 + 计算缩放因子），无需 SVD、无需额外存储、无需修改训练流程
4. **理论扎实**：Proposition 1（参数缩放等价）和 Proposition 2（幅度动态演化）给出了严格的数学刻画
5. **实用性强**：保持了 LoRA 的全部效率优势（即插即用、无额外开销），同时匹配谱初始化的性能

## 局限与展望

1. **幅度并非最优**：LoRAM 模仿谱初始化的幅度而非寻找最优幅度，可能存在更好的缩放策略
2. **层间差异**：不同层可能需要不同的缩放因子，联合优化幅度 + 学习率 + rank 是开放问题
3. **优化动态理论不完整**：论文主要分析了固定点附近的线性近似，对非线性训练动态缺乏深入分析
4. **LoRA-GA 的特殊性**：tracking mode 对 LoRA-GA 失效，说明某些情况下方向也有影响，幅度原理不完全
5. **未来方向**：自适应层级幅度调度；与其他 PEFT 方法（如 LoRA+、DoRA）正交组合；大规模模型验证

## 相关工作与启发

- **PiSSA (NeurIPS 2024)**：谱初始化开山之作，本文证明其优势来自幅度而非方向
- **RsLoRA**：$\alpha = \sqrt{r}$ 缩放，本文证明其与 LoRA+ 的学习率调整本质等价
- **LoRA+ (ICML 2024)**：不同学习率用于 A/B 矩阵，本文从幅度角度统一解释
- **LoRA-GA (NeurIPS 2024)**：数据驱动初始化，本文证明它最大化了 LoRA 梯度幅度
- **启发**：参数高效微调的核心可能不是"方向对齐"而是"幅度匹配"，这个 insight 可能推广到其他低秩方法（如 adapter、prefix tuning）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 幅度原理是一个令人信服的新视角，但结论在回头看时有一定直觉性
- 实验充分度: ⭐⭐⭐⭐⭐ — NLU/NLG/VLM/图像生成全覆盖，消融设计精巧（tracking mode 尤其有启发性）
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，但符号略多，阅读成本较高
- 价值: ⭐⭐⭐⭐⭐ — 为 LoRA 社区提供了极简且高效的 baseline，有望成为新的默认初始化方案

<!-- RELATED:START -->

## 相关论文

- [F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)
- [Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)

<!-- RELATED:END -->
