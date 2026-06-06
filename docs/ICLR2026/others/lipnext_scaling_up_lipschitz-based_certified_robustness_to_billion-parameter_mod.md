---
title: >-
  [论文解读] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models
description: >-
  [ICLR 2026][Lipschitz约束] 提出LipNeXt——首个无约束、无卷积的1-Lipschitz架构，通过正交流形优化学习正交矩阵 + 由Theorem 1理论驱动的Spatial Shift Module实现空间混合，成功扩展到十亿参数规模…
tags:
  - "ICLR 2026"
  - "Lipschitz约束"
  - "认证鲁棒性"
  - "正交矩阵"
  - "流形优化"
  - "空间位移模块"
---

# LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models

**会议**: ICLR 2026  
**arXiv**: [2601.18513](https://arxiv.org/abs/2601.18513)  
**代码**: 无  
**领域**: 其他 / 对抗鲁棒性  
**关键词**: Lipschitz约束, 认证鲁棒性, 正交矩阵, 流形优化, 空间位移模块

## 一句话总结

提出LipNeXt——首个无约束、无卷积的1-Lipschitz架构，通过正交流形优化学习正交矩阵 + 由Theorem 1理论驱动的Spatial Shift Module实现空间混合，成功扩展到十亿参数规模，在CIFAR-10/100、Tiny-ImageNet和ImageNet上全面刷新认证鲁棒精度(CRA) SOTA，ImageNet上 $\varepsilon=1$ 时CRA提升达+8%。

## 研究背景与动机

**对抗鲁棒性的挑战**：对抗样本是安全关键应用（自动驾驶、医学影像、恶意软件检测）的核心威胁。经验防御无法提供形式化保证，模型可能在更强攻击下依然脆弱。

**认证鲁棒性的两条路线**：(a) 随机平滑(RS)通过噪声平均给出概率保证；(b) Lipschitz方法利用网络的Lipschitz常数给出确定性(worst-case)保证。本文聚焦后者。

**Lipschitz方法的scaling瓶颈**：现有方法大多使用 $\leq 32$M参数的VGG风格架构，在CIFAR-100上就开始欠拟合，ImageNet上性能大幅下降。增大模型带来的增益迅速饱和。

**正交矩阵是性能关键也是开销瓶颈**：紧的Lipschitz界要求所有权重正交。现有显式方法（矩阵指数SOC、Cayley变换、LOT-Orth、Cholesky-Orth）和隐式方法（AOL、CPL、SLL层）都引入大量额外计算（FFT、矩阵逆、power iteration等），限制了扩展性和低精度训练。

**注意力机制不适合Lipschitz控制**：Transformer的attention缺乏直接的Lipschitz约束手段。但ConvNeXt和MetaFormer表明，Transformer时代的宏观设计可以与Lipschitz架构结合。

**核心动机**：能否设计一个无需约束重参数化、无需卷积的1-Lipschitz架构，使认证鲁棒性像标准训练一样享受scaling law的红利？

## 方法详解

### 整体框架

LipNeXt Block由四个1-Lipschitz组件堆叠而成：正交投影 $R \in \mathcal{M}_C$（通道混合）→ Spatial Shift $\mathcal{S}$（空间混合）→ 正交投影 $R^\top$（回投）→ 正交线性 $M$ + $\beta$-Abs激活。完整block：

$$Z = \sigma(M R^\top \mathcal{S}(R(X + p)) + b)$$

其中 $p \in \mathbb{R}^{H \times W \times 1}$ 为可学习位置编码，$\sigma$ 为 $\beta$-Abs激活。最终通过L2 Spatial Pool汇聚空间维度：$[\text{L2Pool}(X)]_c = \sqrt{\sum_{h,w} X_{h,w,c}^2}$，整个网络严格保持1-Lipschitz。

### 关键设计1：FastExp流形优化

**核心观察**：大模型训练时学习率 $\eta \sim 10^{-3}$ 很小，因此Eq.3中指数映射的参数矩阵 $A$ 的Frobenius范数也很小。据此提出自适应截断Taylor展开：

$$\text{FastExp}(A) = \begin{cases} I + A + \frac{1}{2}A^2, & \|A\|_F < 0.05 \\ I + A + \frac{1}{2}A^2 + \frac{1}{6}A^3, & 0.05 \leq \|A\|_F < 0.25 \\ I + A + \frac{1}{2}A^2 + \frac{1}{6}A^3 + \frac{1}{24}A^4, & 0.25 \leq \|A\|_F < 1 \\ \exp(A), & \|A\|_F \geq 1 \end{cases}$$

**两个稳定化技术**：

- **(a) 周期性Polar Retraction**：每epoch结束时做SVD $X = U\Sigma V^\top$，重置 $X \leftarrow UV^\top$，修正截断误差的累积。
- **(b) 流形Lookahead**：标准Lookahead的权重插值 $0.5X_t + 0.5X_{t-K}$ 会破坏正交性。本文改为在正切空间插值skew-symmetric更新：$X_{\text{slow}} \leftarrow X_{\text{slow}} \cdot \text{FastExp}(\frac{1}{2}\sum_{j=t-K+1}^{t} \Delta_j)$，保持流形上的正交性。

额外per-step开销最多只有**5次矩阵乘法**，远低于FFT卷积或power iteration。

### 关键设计2：Spatial Shift Module (Theorem 1)

**Theorem 1**：设 $f_K$ 为kernel $K \in \mathbb{R}^{k \times k}$、单位步长、circular padding的spatial convolution。$f_K$ 是保范的(tight 1-Lipschitz isometric) $\|f_K(X) - f_K(Y)\|_F = \|X - Y\|_F, \forall X,Y$ 当且仅当 $K$ 中恰好有一个非零元素且值为 $\pm 1$。

**含义**：保范的depthwise卷积**必然退化为空间位移**——理论直接导出了架构设计。

**2D实现**：将每个token的特征分为5个partition（上移/下移/左移/右移/不动），对应circular shift。通过正交投影 $R$ 在shift前后混合通道，确保shift不是固定作用在相同通道子集上。经验最优shift比例 $\alpha \in \{1/8, 1/16\}$。

**Circular padding vs Zero padding**：zero-padding隐式引入位置信息，circular padding不引入但保证保范。本文采用circular padding + 显式位置编码 $p$，实验证实这优于zero-padding方案。

### 关键设计3：β-Abs激活

$$[\beta\text{-Abs}(\boldsymbol{x})]_i = \begin{cases} |x_i|, & i \leq \beta d \\ x_i, & \text{otherwise} \end{cases}$$

$\beta \in [0,1]$ 控制非线性程度。当 $\beta = 0.5$ 时可表达常用的MinMax激活：$\exists R \in \mathcal{M}_{2d}, \text{MinMax}(x) = R^\top \beta\text{-Abs}(Rx)$。1-Lipschitz且GPU友好（无需排序或配对操作）。

### 训练策略

使用EMMA loss进行认证鲁棒训练，训练收据沿用LiResNet++。支持bfloat16精度训练（LiResNet因数值溢出只能float32，BRONet因FFT复数运算等效float64）。多类分类采用one-vs-rest分解。

## 实验关键数据

### 表1：CIFAR-10/100 + Tiny-ImageNet 主实验

| 数据集 | 模型 | 参数量 | Clean Acc | CRA@36/255 | CRA@72/255 | CRA@108/255 |
|--------|------|--------|-----------|------------|------------|-------------|
| CIFAR-10 | LiResNet | 83M | 81.0 | 69.8 | 56.3 | 42.9 |
| CIFAR-10 | BRONet | 68M | 81.6 | 70.6 | 57.2 | 42.5 |
| CIFAR-10 | **LipNeXt L32W1024** | **64M** | **81.5** | **71.2** | **59.2** | **45.9** |
| CIFAR-10 | **LipNeXt L32W2048** | **256M** | **85.0** | **73.2** | **58.8** | 43.3 |
| CIFAR-100 | LiResNet | 83M | 53.0 | 40.2 | 28.3 | 19.2 |
| CIFAR-100 | BRONet | 68M | 54.3 | 40.2 | 29.1 | 20.3 |
| CIFAR-100 | **LipNeXt L32W2048** | **256M** | **57.4** | **44.1** | **31.9** | **22.2** |
| Tiny-IN | BRONet | 75M | 41.2 | 29.0 | 19.0 | 12.1 |
| Tiny-IN | **LipNeXt L32W2048** | **256M** | **45.5** | **35.0** | **25.9** | **18.0** |

### 表3：ImageNet 实验

| 模型 | 参数量 | 训练速度(min/epoch) | CRA@ε=1 | Clean@ε=36/255 | CRA@ε=36/255 |
|------|--------|-------------------|---------|----------------|---------------|
| LiResNet | 51M | 5.3 | 14.2 | 45.6 | 35.0 |
| BRONet | 86M | 10.5 | - | 49.3 | 37.6 |
| **LipNeXt 1B** | **1B** | **8.9** | **21.1** | **55.9** | **40.3** |
| **LipNeXt 2B** | **2B** | **17.8** | **22.4** | **57.0** | **41.2** |

ImageNet $\varepsilon=1$ 时CRA较BRONet提升+8%，$\varepsilon=36/255$ 时CRA提升+3%。

### 表4：Scaling实验 (ImageNet 400类, ε=1)

| 配置 | 深度 | 宽度 | Clean Acc | CRA |
|------|------|------|-----------|-----|
| 固定深度=32 | 32 | 1024→4096 | 40.5→51.7 | 22.9→30.0 |
| 固定宽度=2048 | 8→128 | 2048 | 30.7→47.5 | 22.4→26.9 |
| 固定参数=1B | 32 | 4096 | **51.7** | **30.0** |
| 固定参数=1B | 64 | 2896 | 51.2 | 29.6 |

深度32层在固定参数预算下最优。宽度和深度都带来非饱和收益。

## 关键发现

1. **Lipschitz认证可以从scaling中获益**：1B→2B参数模型的CRA仍在持续提升，打破了"认证鲁棒=小模型"的传统认知。
2. **低精度训练的稳定性**：LipNeXt可用bfloat16训练，而LiResNet因power iteration在bf16下数值溢出只能用float32，BRONet的FFT复数运算等效float64。这使得LipNeXt能持续受益于硬件加速。
3. **FastExp近似足够准确**：自适应Taylor截断+周期性SVD retraction+流形Lookahead，三者组合保证了数值稳定性，性能与精确矩阵指数相当。
4. **保范卷积的理论极限**：Theorem 1证明circular-padding下保范depthwise卷积只能是空间位移——这是一个tight的必要充分条件。
5. **位置编码的必要性**：circular padding不引入位置信息，需要显式位置编码才能达到与zero-padding竞争的性能。实验验证了circular padding + 显式PE优于zero-padding。

## 亮点与洞察

- **理论驱动的架构设计**：Theorem 1从保范性条件自然导出Spatial Shift Module，不是经验性的"试出来的"设计。
- **约束→流形**的范式转变：将正交约束从"重参数化后投影"变为"直接在流形上优化"，概念简洁且计算高效（每步仅5次矩阵乘）。
- **首个billion-scale认证鲁棒模型**：证明了确定性认证不必局限于小模型，为后续工作开辟了新空间。
- **训练效率**：尽管模型大10-20倍，训练吞吐量与先前工作相当（1B模型8.9min/epoch vs BRONet 86M模型10.5min/epoch）。

## 局限性

- 仅考虑 $\ell_2$ 范数认证，$\ell_\infty$ 范数的Lipschitz认证更具实际需求但更具挑战。
- 2B参数模型的训练需要16×H100 GPU，部署成本高，实用化需要蒸馏或其他压缩技术。
- 最大CRA@ε=108/255在CIFAR-10上不如AOL（49.0 vs 45.9），AOL牺牲clean acc换取大ε下的鲁棒性。
- 未在大规模图文数据集上训练，与随机平滑方法（可利用预训练CLIP等）的对比可能不完全公平。

## 相关工作对比

| 方法 | 正交矩阵实现 | 空间混合 | 是否可scale | 低精度训练 |
|------|-------------|---------|------------|-----------|
| **LipNeXt (本文)** | 流形优化+FastExp | Spatial Shift (无参数) | ✅ 1-2B | ✅ bf16 |
| LiResNet (Hu et al., 2024) | Cholesky-Orth | 卷积+power iteration | ❌ 83M饱和 | ❌ 需float32 |
| BRONet (Lai et al., 2025) | Block Reflector | FFT-based频域卷积 | ❌ 86M | ❌ 等效float64 |

**vs LiResNet**：LipNeXt延续其宏观结构但替换了所有核心组件——用流形优化替代Cholesky-Orth，用Spatial Shift替代卷积+power iteration，消除了scaling瓶颈。

**vs BRONet**：BRONet的FFT卷积需要complex32运算，LipNeXt的Spatial Shift是无参数的整数索引操作。LipNeXt在相同参数量下已超越BRONet，scaling到更大模型后优势更大。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首创无卷积+流形优化的billion-scale认证鲁棒架构，Theorem 1理论驱动设计
- **实验充分度**: ⭐⭐⭐⭐ 4个数据集 + scaling实验 + 多组消融，但缺少L∞实验
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，Algorithm 1完整，动机层层递进
- **实用价值**: ⭐⭐⭐⭐⭐ 认证鲁棒性的重要里程碑，证明确定性保证可以追踪现代scaling趋势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](../../AAAI2026/others/certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ICML 2025\] Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss](../../ICML2025/others/enhancing_certified_robustness_via_block_reflector_orthogonal_layers_and_logit_a.md)

</div>

<!-- RELATED:END -->
