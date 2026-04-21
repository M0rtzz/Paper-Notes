---
title: >-
  [论文解读] NRGPT: An Energy-based Alternative for GPT
description: >-
  [ICLR 2026][优化][能量基模型] 提出NRGPT（eNeRgy-GPT），对标准GPT进行最小修改使其成为能量基模型：设计注意力能量和前馈能量函数，使每层前向传播等价于token在能量landscape上的梯度下降步，证明了渐近能量下降和稳定收敛性质，在ListOps/Shakespeare/OpenWebText上验证了与标准GPT可比的性能。
tags:
  - ICLR 2026
  - 优化
  - 能量基模型
  - GPT
  - 自回归
  - 梯度下降推理
  - 渐近稳定性
---

# NRGPT: An Energy-based Alternative for GPT

**会议**: ICLR 2026  
**arXiv**: [2512.16762](https://arxiv.org/abs/2512.16762)  
**代码**: 无  
**领域**: 语言模型架构/能量模型  
**关键词**: 能量基模型, GPT, 自回归, 梯度下降推理, 渐近稳定性

## 一句话总结

提出NRGPT（eNeRgy-GPT），对标准GPT进行最小修改使其成为能量基模型：设计注意力能量和前馈能量函数，使每层前向传播等价于token在能量landscape上的梯度下降步，证明了渐近能量下降和稳定收敛性质，在ListOps/Shakespeare/OpenWebText上验证了与标准GPT可比的性能。

## 研究背景与动机

**领域现状**：GPT架构是自回归语言建模的主流范式，通过next-token prediction实现文本生成。能量基模型（EBM）则是另一重要范式，将推理视为在能量景观上的动力学过程——低能量对应合理样本、高能量对应异常样本。两者看似完全不同，但近年来越来越多的研究暗示它们之间存在深层联系。

**现有痛点**：
1. **GPT与EBM的联系不明确**：Von Oswald等人证明了ICL可能是梯度下降，但仅考虑线性Transformer（无softmax），过度简化
2. **Energy Transformer不适用于GPT设定**：ET为BERT-like掩码补全设计——掩码token快速演化以匹配缺失部分，而GPT中没有掩码，每个token需要演化为序列中的下一个token
3. **EBM for LLM的现有工作**：如EBT将能量计算放在标准Transformer前向传播的输出端，而非将前向传播本身视为能量优化过程
4. **缺乏将GPT前向传播直接转化为能量landscape探索的理论框架**

**核心矛盾**：如何在不改变训练范式（自监督next-token prediction）的前提下，让GPT的推理过程具有EBM的理论优势（可解释性、系统化解空间探索、自然的对齐机制）？

**本文方案**：对平行Transformer（GPT-J风格）进行最小修改——让注意力和前馈网络分别成为两个能量函数的梯度，从而使每一层的前向传播变成能量梯度下降的一步。

## 方法详解

### 整体框架

NRGPT采用**权重共享的循环架构**：单个模块反复应用 $T$ 次，替代传统 $T$ 层不同权重的Transformer。每次应用对应能量梯度下降的一步：

$$x^{(t+1)} = x^{(t)} - \eta^{(t)} \frac{\partial E}{\partial g^{(t)}}$$

其中 $g^{(t)} = \text{LN}(x^{(t)})$ 是经过LayerNorm/RMSNorm的token表示，$\eta$ 是推理速率矩阵（inference rate matrix）。

### 关键设计一：双能量函数

**注意力能量**（从Dense Associative Memory推导）：

$$E_A^{\text{AT}} = -\frac{1}{\beta} \sum_h \alpha_h \log \left[ \sum_{B<A} \exp(\beta \cdot g_B^\top J_h g_A) \right]$$

其中 $J_h = [W^K_h]^\top W^Q_h$ 合并了Key和Query投影，$\alpha_h$ 是可学习的头权重。对 $g_A$ 求梯度得到的更新与标准多头注意力结构高度对应：

$$\text{Original: } [W^P_h]^\top W^V_h \equiv \text{Energy: } \alpha_h \eta J_h^\top$$

**前馈能量**（两个变体）：

- FF1: $E^{\text{FF}} = -\|\sigma(Wg_A)\|^2$，梯度给出单权重矩阵的前馈更新
- FF2W: $E^{\text{FF}} = -g_A^\top W^2 \sigma(W^1 g_A)$，梯度给出双权重矩阵的前馈更新（更接近标准MLP）

### 关键设计二：能量下降保证与渐近稳定性

**能量下降条件**（Proposition 2.1）：当推理速率 $\eta = c \cdot \text{diag}(\gamma)$（$c > 0$，$\gamma$ 来自LayerNorm）时，更新规则保证渐近能量下降 $\dot{E}_A < 0$。

**渐近稳定性**（利用因果注意力掩码的关键性质）：
- token $A$ 的能量 $E_A$ 仅依赖 $B \leq A$ 的token状态
- 第一个token的能量单调下降且有下界→收敛到不动点
- 第一个token稳定后，第二个token的能量也单调下降→递归论证
- **所有token最终渐近收敛到稳定状态**——这是NRGPT独特的"渐近稳定性"现象

**无LayerNorm情况**（Proposition 2.2）：当 $g = x$ 时，只需 $\eta$ 的对称部分 $\eta_+ = (\eta + \eta^\top)/2$ 半正定即可保证 $\dot{E} < 0$，反对称部分 $\eta_-$ 无约束。

### 关键设计三：与标准Transformer的结构对应

| 模块 | 标准Transformer | NRGPT能量梯度 |
|:---|:---|:---|
| 注意力输出矩阵 | $[W^P]^\top W^V$ | $\alpha \eta J^\top$ |
| 前馈第二层权重 | $W^2$ | $W^1 \eta^\top$ |
| 层间连接 | 不同层不同权重 | 权重共享 + 循环应用 |
| 推进机制 | 逐层传播 | 能量landscape上的梯度下降 |

## 实验结果

### 主实验：ListOps嵌套数学运算

测试最大值、中位数、模20求和三种嵌套运算：

| 模型 | 学习转变点（参数量） | 最终准确率 |
|:---|:---:|:---:|
| GPT_Rec_parallel | 2.3×10⁴ | ~100% |
| NRGPT_H_FF1 | 2.4×10⁴ | ~100% |
| NRGPT_H_FF2W | 2.98×10⁴ | ~100% |

NRGPT各变体在ListOps上与基线性能匹配，学习转变点非常接近。

### OpenWebText语言建模

| 模型 | 参数量 | Val Loss (mean±std) | Val Loss (min) |
|:---|:---:|:---:|:---:|
| GPT (12层) | 124M | 2.921±0.005 | 2.915 |
| GPT_Rec_parallel | 85M | 3.454±0.037 | 3.411 |
| NRGPT_H_FF2W | 90M | 3.467±0.073 | 3.404 |

**关键发现**：NRGPT在参数量少34M的情况下达到了与循环GPT基线可比的验证损失（3.404 vs 3.411）。

### 消融：抗过拟合特性

| 模型 | Shakespeare训练集Loss | 验证集Loss | 过拟合程度 |
|:---|:---:|:---:|:---:|
| GPT | 极低 | 较高 | 严重（大模型） |
| GPT_Rec_parallel | 较低 | 较高 | 中等 |
| **NRGPT** | 中等 | **与验证相当** | **轻微** |

在Shakespeare数据集上，NRGPT在大参数量时表现出显著的抗过拟合特性——最佳验证损失与GPT基线相当，但训练集过拟合程度明显更低。这可能是因为能量landscape的梯度下降过程天然具有正则化效果。

## 论文评价

### 优点

1. **理论优雅**：通过最小修改建立GPT与EBM的严格联系，能量下降和渐近稳定性的证明利用了因果掩码的结构，非常漂亮
2. **开辟新方向**：将推理视为能量优化为LLM提供了全新视角，可能带来对齐（通过能量正则化）、可解释性（通过能量landscape分析）等应用
3. **抗过拟合现象**有趣且有实际价值

### 不足

1. 当前规模仅验证到124M参数，与现代LLM的规模差距巨大，可扩展性尚不明确
2. 权重共享约束使NRGPT比标准GPT参数效率更低（需要更多循环步数补偿）
3. 推理速率 $\eta$ 的约束较强（$\eta = c \cdot \text{diag}(\gamma)$），限制了模型的表达力
4. 验证损失与标准GPT仍有差距（3.404 vs 2.915），尽管参数量不同

### 评分

⭐⭐⭐⭐

**推荐理由**：在GPT与EBM之间建立了迄今最紧密的理论联系，渐近稳定性的证明特别精彩。虽然当前实验规模有限，但理论贡献足以开辟新的研究方向——如何利用能量landscape的显式优化来改进LLM的对齐、鲁棒性和可解释性。

<!-- RELATED:START -->

## 相关论文

- [The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)
- [Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)
- [Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)

<!-- RELATED:END -->
