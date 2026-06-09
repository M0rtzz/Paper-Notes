---
title: >-
  [论文解读] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking
description: >-
  [NeurIPS 2025][预训练][neural collapse] 利用 grokking（延迟泛化）作为因果探针，证明 **relative flatness 是泛化的（潜在）必要条件**，而 neural collapse 虽常伴随泛化出现，但并非必要——它只是通往 flatness 的一条路径。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "neural collapse"
  - "relative flatness"
  - "generalization"
  - "grokking"
  - "loss landscape"
---

# Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking

**会议**: NeurIPS 2025  
**arXiv**: [2509.17738](https://arxiv.org/abs/2509.17738)  
**代码**: [GitHub](https://github.com/TrustworthyMachineLearning-Lab/grokking_flatness)  
**作者**: Ting Han, Linara Adilova, Henning Petzka, Jens Kleesiek, Michael Kamp
**单位**: TU Dortmund / Lamarr Institute / Ruhr University Bochum / UK Essen
**领域**: LLM预训练  
**关键词**: neural collapse, relative flatness, generalization, grokking, loss landscape

## 一句话总结

利用 grokking（延迟泛化）作为因果探针，证明 **relative flatness 是泛化的（潜在）必要条件**，而 neural collapse 虽常伴随泛化出现，但并非必要——它只是通往 flatness 的一条路径。

---

## 研究背景与动机

### 核心困惑

深度过参数化网络能记忆任意标签（Zhang et al., 2017），却在自然数据上良好泛化。两个被广泛讨论的泛化几何特征是：

**Neural Collapse (NC)**：训练后期，倒数第二层特征按类坍缩为均值（NC1），均值形成 simplex ETF（NC2），分类器权重与均值对齐（NC3），分类器退化为最近邻（NC4）。已在多种架构中被观察到并被视为泛化指标。

**Loss landscape flatness**：loss 对参数扰动不敏感。经典 Hessian 度量有 reparameterization sensitivity 问题（Dinh et al., 2017），Petzka et al. (2021) 提出的 **relative flatness** 解决了该问题，具有重参数化不变性。

两者在训练后期**同时**出现并与泛化相关，但核心问题是：**它们是泛化的原因还是副产品？谁才是更根本的驱动因素？**

### 为什么现有工作不够

- NC 在不平衡数据上存在争议：Fang et al. (2021) 发现 minority collapse 导致分类失败；Hui et al. (2022) 指出 test-set NC 反而损害 transfer learning 性能；Hong & Ling (2024) 表明需要足够高的 SNR 才能使 NC 与泛化正相关。
- 经典 flatness 度量（如 Hessian trace）对 reparameterization 敏感，且计算代价高。SAM 相关方法受到批评（Andriushchenko et al., 2023）。
- 两者的因果关系从未被严格解耦——之前的研究只观察到共现，无法判断必要性。

### 关键洞察：用 Grokking 做因果探针

**Grokking** 指网络在完全记忆训练集后很久才突然泛化（Power et al., 2022）。这种现象在**时间上分离了记忆与泛化**，可以通过追踪 NC 和 flatness 各自出现的时间节点来判断因果关系——这是本文最核心的方法论创新。

---

## 方法详解

### 1. 度量定义

#### NCC (Neural Collapse Clustering)

采用 Galanti et al. (2022) 的简化度量，捕捉类内聚集和类间分离：

$$\mathrm{NCC} = \sum_{c \neq c'} \frac{V_c + V_{c'}}{2\|\mu_c - \mu_{c'}\|^2}$$

其中 $\mu_c$ 是第 $c$ 类特征均值，$V_c$ 是类内方差。NCC 低 → 类内紧聚、类间分离 → 符合 NC 特性。

#### Relative Flatness (κ)

Petzka et al. (2021) 提出的重参数化不变度量：

$$\kappa^{\phi}_{\mathrm{Tr}}(\mathbf{w}) = \sum_{s,s'=1}^{d} \langle \mathbf{w}_s, \mathbf{w}_{s'} \rangle \cdot \mathrm{Tr}(H_{s,s'}(\mathbf{w}, \phi(S)))$$

其中 $H_{s,s'}$ 是 empirical loss 关于权重行的 Hessian。该度量对 neuron-wise rescaling 和正交变换不变。κ 小 → 解平坦 → 泛化好。实际计算采用 Walter et al. (2024) 在 cross-entropy loss 下的闭式上界，仅需倒数第二层，与模型规模无关。

### 2. Grokking 实验：时间线解耦

在模运算任务（$x + y \mod p$）上训练 2 层 Transformer，AdamW (lr=1e-4, wd=1.0)，训练 $10^6$ 步，50/50 train/val split。

**关键观察**（Figure 1）：
- **NCC** 在记忆阶段就开始下降——即网络尚未泛化时就出现了 collapse 迹象
- **Relative flatness** 在记忆阶段保持高值，仅在泛化开始时才骤降
- 结论：两者都与泛化相关，但 NC **先于泛化出现**，而 flatness 与泛化**同步出现**

### 3. NC 非必要：正则化抑制实验

在 ResNet-18 / CIFAR-10 上使用 NCC 正则：

$$\mathcal{L}_{\text{NC\_REG}} = \mathcal{L}_{\text{CE}} - \lambda \cdot \text{NCC}$$

通过最大化 NCC 来**主动抑制 collapse**（缩小类均值间距、增大类内方差）。

**结果**（Figure 2）：
- **训练/测试精度不受影响**：即使 NCC 持续增大（collapse 被抑制），泛化性能不变
- **Relative flatness 不受影响**：说明 flatness 不依赖于 NC
- 此外，Figure 3 确认当 NC 被抑制时，cluster 角度偏离最优 10-simplex 配置，验证 NCC 度量的有效性

### 4. Flatness 是必要的：反向诱导延迟泛化

使用 relative flatness 正则来鼓励 sharp minima：

$$\mathcal{L}_{\text{RF\_REG}} = \mathcal{L}_{\text{CE}} - \lambda \cdot \kappa^{\phi}_{Tr}(\mathbf{w})$$

在以下架构/数据上验证：
- **ResNet-18 / CIFAR-10**：正则移除前验证精度低，epoch 200 移除后迅速恢复
- **ViT / ImageNet-100**：epoch 150 移除后同样恢复泛化
- **TinyBERT, DistilGPT2 / SST-5**：预训练语言模型也出现延迟泛化

**核心发现**：抑制 flatness → **人为诱导 grokking**，在原本不出现 grokking 的架构和数据上也能复现！这直接证明了 flatness 对泛化的必要性。

### 5. 理论贡献：NC 蕴含 Flatness

**Proposition 5.3**：在经典 NC 假设下（NC1-NC4），relative flatness 有上界：

$$\kappa_{\phi}(w) \leq \lambda^2 k^3 M^4 \cdot \frac{e^{-\lambda M^2 \cdot \frac{k}{k-1}}}{(1 + (k-1)e^{-\lambda M^2 \cdot \frac{k}{k-1}})^2}$$

当 $\lambda$ 足够大时，$\kappa_{\phi}(w) \lesssim \lambda^2 k^3 M^4 e^{-\lambda M^2 \cdot k/(k-1)}$——**指数衰减**。

证明思路：
1. 先证 NC 条件下 bias + 类均值在全局均值上的投影为常数（Lemma A.1）
2. 证明 $\kappa_{\phi}(w) \leq \|w\|^2 \mathrm{Tr}(H(w))$（Lemma A.2，利用 Hessian 半正定性和 Cauchy-Schwarz）
3. 在 NC 结构下展开 Hessian trace 的闭式表达式，利用 softmax margin $\delta = M^2 \cdot k/(k-1)$ 得到指数衰减界

**意义**：NC 是通往 flatness 的一条充分路径，解释了两者经验上的共现，但 flatness 也可通过其他机制实现。

### 6. Representativeness 的角色

flatness 的理论保证还依赖于 **representativeness**（训练特征对真实分布的代表性）。Appendix D 证实 representativeness 仅在泛化开始时改善——这意味着 flatness 和 NC 都不能在缺乏特征对齐的情况下保证泛化。

---

## 实验关键数据

### 表1：Flatness 正则化诱导延迟泛化的定量结果

| 设定 | 正则移除前 κ | 最终 κ | 正则下验证精度 | 最优验证精度 |
|------|-------------|--------|--------------|------------|
| ResNet-18 / CIFAR-10 (epoch 199) | 2209.94 | 1.91 | ~60% | ~78% |
| ViT / ImageNet-100 (epoch 149) | 22212.75 | 313.22 | ~38% | ~43% |

正则移除后 κ 显著下降，验证精度恢复至基线水平，直接展示了 flatness 与泛化的因果关联。

### 表2：NC 抑制实验中不同 λ 的效果

| NCC 正则 λ | NCC 行为 | 验证精度影响 | Relative Flatness 影响 | 训练稳定性 |
|-----------|---------|------------|---------------------|-----------|
| 1×10⁻² | NCC 增大 | 严重下降 (<60%) | 不稳定 | 训练崩溃 (~epoch 70) |
| 1×10⁻³ | NCC 持续增大 | **不受影响** (~78%) | **不受影响** | 稳定 |
| 1×10⁻⁴ | 几乎无变化 | 不受影响 | 不受影响 | 稳定 |
| 1×10⁻⁵ | 无变化 | 不受影响 | 不受影响 | 稳定 |

λ=10⁻³ 是最关键的设定：成功抑制 NC（NCC 增大）但泛化/flatness 完全不受影响。

---

## 亮点与洞察

1. **方法论创新**：将 grokking 从"有趣的异常现象"提升为"因果分析工具"，通过时间维度的分离来解耦相关变量——这一范式可推广到其他深度学习现象研究中。
2. **人为诱导 grokking**：首次证明可以通过正则化 flatness 在 ResNet/CIFAR-10、ViT/ImageNet、预训练 LM/SST-5 等"正常"设定下诱导延迟泛化，跨越视觉和 NLP 模态。
3. **理论统一**：NC → flatness 的指数衰减界将两个独立研究方向纳入同一几何框架，且证明过程利用了 softmax-CE 的具体结构，不依赖 unconstrained features model 或 layer peeling 近似。
4. **实用启示**：Relative flatness 可作为单层诊断指标（计算与模型规模无关），为训练监控和正则化设计提供新工具。

---

## 局限性

1. **仅限分类任务**：所有实验均为分类网络和生成式语言模型（CE loss），是否适用于对比学习预训练、回归任务或大规模 LLM 尚不清楚。
2. **Representativeness 假设**：flatness 的理论保证依赖"标签局部常数"和"特征具有代表性"两个假设，在结构化预测、高噪声回归或复杂场景下可能不成立。
3. **正则器的脆弱性**：flatness 正则器与 CE loss 存在内在冲突（CE 降低预测不确定性，正则器仅在高不确定性下有效），λ 调参敏感，过大导致训练不稳定。
4. **诱导的 grokking 与自然 grokking 机制可能不同**：人为延迟泛化是否真正复现了自然 grokking 的内在机制尚待验证。
5. **flatness 的必要性是经验性结论**：论文用"potentially necessary"措辞，尚无数学证明其严格必要性。

---

## 相关工作与启发

- **NC 与 Transfer Learning 的矛盾**：Hui et al. (2022) 发现 test-set NC 损害下游任务，与本文"NC 非必要"一致。这提示 NC 过度压缩表示空间，可能丢失细粒度信息。
- **SAM 的局限**：虽然 SAM 在视觉任务上提升性能，但 Andriushchenko & Flammarion (2022) 发现它并非只鼓励 flat minima。本文的 relative flatness 提供了更可靠的度量。
- **Grokking 的双阶段动态**：Kumar et al. (2024) 将 grokking 解释为从 lazy 到 rich training 的转变，本文从几何视角提供了互补解释——rich regime 对应 flat 解的出现。
- **对 flatness-aware 训练的启示**：鼓励 flat minima 并不能显著提升泛化（SGD 本身已偏向 flat），但抑制 flatness 会可靠地延迟泛化——这种不对称性暗示 flatness 更适合作为诊断工具而非优化目标。

---

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 8 — Grokking 作为因果探针的方法论在泛化理论中是新颖的 |
| 理论深度 | 8 — NC→flatness 的严格证明和 representativeness 分析 |
| 实验充分性 | 8 — 跨视觉/NLP、多架构、消融实验完整 |
| 论述清晰度 | 9 — 论证逻辑严密，从观察到干预到理论一气呵成 |
| 实用价值 | 6 — Relative flatness 作为诊断指标有价值，但对训练优化的指导有限 |
| **总分** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](../../ICLR2026/llm_pretraining/explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)

</div>

<!-- RELATED:END -->
