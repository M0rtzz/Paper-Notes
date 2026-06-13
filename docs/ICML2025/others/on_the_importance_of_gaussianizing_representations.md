---
title: >-
  [论文解读] On the Importance of Gaussianizing Representations
description: >-
  [ICML2025][Gaussianization] 基于信息论动机（正态分布同时是最优信号与最差噪声分布），提出 Normality Normalization 层：在常规归一化之后用 Power Transform 高斯化激活值，并注入缩放高斯噪声进行正则化，在 ViT/ResNet 上普遍提升泛化与鲁棒性，且不引入额外可学习参数。
tags:
  - "ICML2025"
  - "Gaussianization"
  - "Power Transform"
  - "Normalization Layer"
  - "Mutual Information"
  - "Noise Robustness"
  - "information theory"
---

# On the Importance of Gaussianizing Representations

**会议**: ICML2025  
**arXiv**: [2505.00685](https://arxiv.org/abs/2505.00685)  
**代码**: [GitHub](https://github.com/DanielEftekhari/normality-normalization)  
**领域**: 表示学习 / 归一化  
**关键词**: Gaussianization, Power Transform, Normalization Layer, Mutual Information, Noise Robustness, information theory

## 一句话总结

基于信息论动机（正态分布同时是最优信号与最差噪声分布），提出 Normality Normalization 层：在常规归一化之后用 Power Transform 高斯化激活值，并注入缩放高斯噪声进行正则化，在 ViT/ResNet 上普遍提升泛化与鲁棒性，且不引入额外可学习参数。

## 研究背景与动机

### 核心问题

传统归一化层（BN/LN/IN/GN）仅约束激活值的均值和方差，但从未明确规定激活值应服从何种分布。本文从信息论角度论证：**正态分布是深度网络特征表示的最优编码分布**。

### 信息论动机：互信息博弈

在加性噪声信道 $Y = X + Z$ 中，信号 $X$ 希望最大化 $I(X;Y)$，噪声 $Z$ 希望最小化 $I(X;Y)$。信息论表明，在一阶和二阶矩约束下：

$$\min_Z \max_X I(X; X+Z) = \max_X \min_Z I(X; X+Z)$$

**两者的纳什均衡策略均为正态分布**（Theorem 2.1, Cover & Thomas 2006）。这意味着：

**最大信息容量**：正态分布是给定均值和方差下的最大熵分布，单元以正态编码可最大化表示能力

**最优噪声鲁棒性**：正态信号对随机扰动的鲁棒性最强；高斯噪声是最差噪声，对其鲁棒即对任意随机扰动鲁棒

**最大独立性**：联合正态时，不相关 ⟹ 独立；给定任意相关度，联合正态分布下各变量最大程度独立

### 与学习的关联

- 向激活值添加噪声是有效的正则化（Dropout、噪声注入），而高斯化激活使模型能容忍更多正则化噪声
- 加性高斯噪声下的互信息与最小均方误差（MMSE）有闭式对应关系（Guo et al., 2005），为衡量层间信息传递提供可测量代理

## 方法详解

### Normality Normalization 整体流程

在标准归一化（BN/LN/IN/GN）之后、仿射变换之前，插入两个步骤：

```
输入 u → [归一化: μ̂, σ̂²] → h → [Power Transform: ψ(h; λ̂)] → x → [加缩放高斯噪声] → y → [仿射: γ·y+β] → 输出 v
```

### Step 1: Power Transform 高斯化

采用 Yeo-Johnson Power Transform 将归一化后的激活值 $h$ 映射为更接近高斯的 $x$：

$$\psi(h; \lambda) = \begin{cases} \frac{1}{\lambda}\left((1+h)^{\lambda}-1\right), & h \geq 0, \lambda \neq 0 \\ \log(1+h), & h \geq 0, \lambda = 0 \\ \frac{-1}{2-\lambda}\left((1-h)^{2-\lambda}-1\right), & h < 0, \lambda \neq 2 \\ -\log(1-h), & h < 0, \lambda = 2 \end{cases}$$

参数 $\lambda$ 通过最大似然估计获得。利用 NLL 对 $\lambda$ 的凸性，在 $\lambda_0=1$（恒等变换）处做二阶 Taylor 展开，用一步 Newton-Raphson 法直接求解：

$$\hat{\lambda} = 1 - \frac{\mathcal{L}'(\mathbf{h}; \lambda=1)}{\mathcal{L}''(\mathbf{h}; \lambda=1)}$$

关键设计：先归一化再做 Power Transform，使 $h$ 零均值单位方差，简化 $\hat\lambda$ 计算并提升数值稳定性；不引入额外可学习参数。

### Step 2: 缩放加性高斯噪声

训练时对 Power Transform 输出注入：

$$y_i = x_i + z_i \cdot \xi \cdot s, \quad z_i \sim \mathcal{N}(0,1)$$

其中 $\xi \geq 0$ 为噪声因子超参数，$s = \frac{1}{N}\sum_{i=1}^{N}|x_i - \bar{x}|$ 为通道级缩放因子（$\ell_1$ 范数更鲁棒）。$s$ 的梯度被截断（detach），仅用于噪声缩放而非直接参与学习。

测试时不注入噪声，类似 Dropout 的训练/推理差异。

## 实验关键数据

### Table 1: ViT + LayerNormalNorm（含数据增强）

| 数据集 | LayerNorm | LayerNormalNorm | 提升 |
|---|---|---|---|
| SVHN | 94.61±0.31 | **95.78±0.21** | +1.17 |
| CIFAR-10 | 89.97±0.16 | **91.18±0.13** | +1.21 |
| CIFAR-100 | 66.40±0.42 | **70.12±0.22** | +3.72 |
| Food101 | 73.25±0.19 | **79.11±0.09** | +5.86 |
| ImageNet Top-1 | 71.54±0.16 | **75.25±0.07** | +3.71 |
| ImageNet Top-5 | 89.40±0.11 | **92.23±0.04** | +2.83 |

### Table 2: ResNet + BatchNormalNorm（无数据增强）

| 数据集 | 模型 | BatchNorm | BatchNormalNorm | 提升 |
|---|---|---|---|---|
| CIFAR-10 | RN18 | 88.89±0.07 | **90.41±0.09** | +1.52 |
| CIFAR-100 | RN18 | 62.02±0.17 | **65.82±0.11** | +3.80 |
| STL-10 | RN34 | 58.82±0.52 | **63.86±0.45** | +5.04 |
| TinyImageNet Top-1 | RN34 | 58.22±0.12 | **60.57±0.14** | +2.35 |
| Caltech101 | RN50 | 72.60±0.35 | **74.71±0.51** | +2.11 |
| Food101 | RN50 | 61.15±0.44 | **63.51±0.33** | +2.36 |

### 其他关键结论

- **跨归一化layers有效**：Instance/Group/Decorrelated BN 均可被 augment（STL10 上 GNN > GN、INN > IN、DBNN > DBN）
- **宽度鲁棒**：WideResNet 不同 width factor 下均优于 BN，小宽度网络提升尤为显著
- **深度鲁棒**：不同深度 WideResNet 均有效，且深度越大提升越明显（纠正深层非正态偏差）
- **Batch Size 鲁棒**：不同训练 batch size 下性能稳定
- **噪声鲁棒性消融**：缩放加性噪声 > Gaussian Dropout > 无缩放加性噪声
- **高斯化程度消融**：$\alpha$ 从 0→1 单调提升性能，$\alpha=1$（Newton-Raphson 解）最优

## 亮点与洞察

1. **信息论动机扎实**：从互信息博弈出发推导"正态分布是最优特征编码"，理论链条完整
2. **零额外参数**：相对现有 Norm 层不增加任何可学习参数，纯粹通过改变分布形态获益
3. **即插即用**：可 augment 任意 BN/LN/IN/GN，无需改变模型架构
4. **闭式 $\hat\lambda$ 估计**：一步 Newton-Raphson，无需迭代优化，无需额外超参数（步长等）
5. **Q-Q 图可视化**：清晰展示 NormalNorm 训练后各层激活值的高斯性显著优于 BN
6. **Power Transform + Noise 解耦消融**：两个模块各自独立贡献性能提升

## 局限与展望

1. **运行速度**：Power Transform 增加计算开销，训练时偏差较大（附录 A.2），可能不适合极端算力受限场景
2. **仅验证视觉任务**：实验集中在图像分类（ViT/ResNet），缺少 NLP、语音、推荐系统等其他模态验证
3. **未验证大规模预训练**：ImageNet 实验使用从头训练的 ViT，未测试在 ViT-Large/Huge 或 LLM 等大模型上的效果
4. **对抗鲁棒性仅讨论**：论文指出高斯噪声鲁棒性可能迁移到对抗鲁棒性，但仅以分布层面讨论，未做实际对抗攻击实验
5. **噪声因子 $\xi$ 仍为超参数**：虽然 $\hat\lambda$ 不需要超参数，但噪声缩放因子仍需手动设定
6. **NLL 的二阶近似精度**：依赖于 NLL 在 $\lambda_0=1$ 附近的二次近似质量，对强非正态分布可能不够准确

## 相关工作与启发

- **Power Transform 传统**：Box-Cox (1964) 仅处理正值，Yeo-Johnson (2000) 推广到全实数线
- **归一化层演进**：BN → LN → IN → GN → Decorrelated BN，本文提供正交增强
- **噪声正则化**：Dropout (Srivastava et al., 2014)、Gaussian Dropout，本文的缩放加性噪声是更优方案
- **无限宽极限**：Neal (1996), Lee (2018) 等证明无限宽网络→高斯过程；NormalNorm 或可让有限宽网络也近似高斯过程
- **特征解耦**：Decorrelated BN (Huang et al., 2018) 去相关，NormalNorm 进一步推动联合正态→独立

## 评分

- 新颖性: ⭐⭐⭐⭐ — 信息论驱动的分布处方视角新颖，Power Transform 引入深度网络的方式原创
- 实验充分度: ⭐⭐⭐⭐ — 多模型/数据集/配置消融全面，但缺少 NLP/大模型验证
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨，行文逻辑清晰，细节充分
- 价值: ⭐⭐⭐⭐ — 即插即用的通用归一化增强，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](../../ACL2025/others/mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](../../ICML2026/others/continual_learning_of_domain-invariant_representations.md)
- [\[NeurIPS 2025\] RNNs Perform Task Computations by Dynamically Warping Neural Representations](../../NeurIPS2025/others/rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)
- [\[AAAI 2026\] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](../../AAAI2026/others/guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)
- [\[ICLR 2026\] Exchangeability of GNN Representations with Applications to Graph Retrieval](../../ICLR2026/others/exchangeability_gnn_representations.md)

</div>

<!-- RELATED:END -->
