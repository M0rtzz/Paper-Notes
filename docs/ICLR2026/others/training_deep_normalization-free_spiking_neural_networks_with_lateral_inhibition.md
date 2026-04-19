---
title: >-
  [论文解读] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition
description: >-
   提出基于皮层兴奋-抑制（E-I）回路的无归一化学习框架 DeepEISNN，通过 E-I Init 和 E-I Prop 两项技术实现深度 SNN 的稳定端到端训练，兼顾性能与生物合理性。
tags:

---

# Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2509.23253](https://arxiv.org/abs/2509.23253)
- **代码**: [https://github.com/vwOvOwv/DeepEISNN](https://github.com/vwOvOwv/DeepEISNN)
- **领域**: 脉冲神经网络 / 神经形态计算 / 生物启发计算
- **关键词**: SNN, 侧向抑制, 兴奋-抑制回路, 无归一化训练, 生物合理性

## 一句话总结
提出基于皮层兴奋-抑制（E-I）回路的无归一化学习框架 DeepEISNN，通过 E-I Init 和 E-I Prop 两项技术实现深度 SNN 的稳定端到端训练，兼顾性能与生物合理性。

## 研究背景与动机

### 核心矛盾
SNN 训练面临**性能与生物合理性**的权衡：
- **高性能方法**（反向传播 + 批归一化）：将 SNN 当作普通深度学习构件，牺牲基本生物属性
- **高生物合理性方法**（STDP 等）：训练不稳定，仅适用于浅层网络

### 为什么需要去归一化？
BatchNorm 等归一化方案从整批输入收集统计量，**在生物系统中没有已知类比**。这使得使用归一化的 SNN 作为大规模皮层计算的计算平台变得不合理。

### E-I 回路的重要性
皮层中约 80% 为兴奋性神经元，20% 为抑制性神经元。E-I 交互在增益控制、神经振荡、选择性注意等方面起关键作用，但现有深度 SNN 通常忽略这一基本原理。

## 方法详解

### E-I 回路设计

每层包含 $n_E^{[l]}$ 个兴奋性和 $n_I^{[l]}$ 个抑制性神经元（比例 4:1）。

**兴奋性神经元（LIF 模型）**：
$$\mathbf{u}_E^{[l]}[t+1] = \left(1-\frac{1}{\tau_E}\right)\left(\mathbf{u}_E^{[l]}[t] - \theta_E \cdot \mathbf{s}_E^{[l]}[t]\right) + \mathbf{I}_E^{[l]}[t]$$

**抑制性神经元（快速脉冲近似）**：
$$\mathbf{s}_I^{[l]}[t] \approx \max(0, \mathbf{I}_I^{[l]}[t])$$

由于 $\tau_I \ll \tau_E$，抑制性神经元近似为瞬态稳态，输出类似 ReLU。

**侧向抑制分解**：
- **减法抑制**（E-I 平衡）：$\mathbf{I}_{EI,\text{sub}}^{[l]}[t] = \boldsymbol{W}_{EI}^{[l]} \mathbf{s}_I^{[l]}[t]$
- **除法抑制**（增益控制）：$\mathbf{I}_{EI,\text{div}}^{[l]}[t] = \boldsymbol{W}_{EI}^{[l]}(\mathbf{g}_I^{[l]} \odot \mathbf{s}_I^{[l]}[t])$

**输入整合**：
$$\mathbf{I}_{\text{int}}^{[l]}[t] = \mathbf{g}_E^{[l]} \odot \frac{\mathbf{I}_{EE}^{[l]}[t] - \mathbf{I}_{EI,\text{sub}}^{[l]}[t]}{\mathbf{I}_{EI,\text{div}}^{[l]}[t]} + \mathbf{b}_E^{[l]}$$

### E-I Init：动态参数初始化

标准 Xavier/Kaiming 初始化不适用于 E-I 约束（权重须同号）。

**目标 1：E-I 平衡（减法抑制）**
$$\mathbb{E}[\mathbf{I}_{EE,i}^{[l]}] \approx \mathbb{E}[\mathbf{I}_{EI,\text{sub},i}^{[l]}]$$

使用指数分布初始化兴奋权重，抑制权重设为 $1/n_I^{[l]}$。

**目标 2：增益控制（除法抑制）**
$$\mathbb{E}[\mathbf{I}_{EI,\text{div},i}^{[l]}] = \text{std}(\mathbf{I}_{EE,i}^{[l]})$$

设置 $\mathbf{g}_I^{[l]}$ 的每个元素为 $\sqrt{\frac{2-p}{dp}}$，实现类似归一化的效果。

**动态估计平均发放概率 $p$**：使用训练集第一批数据计算。

### E-I Prop：稳定端到端训练

**自适应稳定化**：替代固定 $\epsilon$，使用样本内最小正值动态替换零值除数。

**直通估计器（STE）**：前向执行自适应稳定化，反向将替换操作视为恒等函数。

**梯度缩放**：将 $\boldsymbol{W}_{EI}^{[l]}$ 的梯度缩放 $1/d$，平衡前向和侧向路径的更新幅度。

## 实验

### 分类任务性能

| 数据集 | 方法 | 架构 | E-I | BN-free | 准确率(%) |
|--------|------|------|-----|---------|----------|
| CIFAR-10 | Vanilla BN | ResNet-18 | ✗ | ✗ | 95.37 |
| CIFAR-10 | TEBN | ResNet-19 | ✗ | ✗ | 94.70 |
| CIFAR-10 | **DeepEISNN** | **ResNet-18** | **✓** | **✓** | **92.05** |
| CIFAR-10 | DANN (ANN) | VGG-16 | ✓ | ✓ | 88.54 |
| CIFAR-10 | BackEISNN | 5-layer CNN | ✓ | ✓ | 90.93 |
| DVS-Gesture | DeepEISNN | VGG-8 | ✓ | ✓ | **94.86** |
| CIFAR10-DVS | DeepEISNN | VGG-8 | ✓ | ✓ | **77.66** |

### 关键发现

1. DeepEISNN (ResNet-18) 在 CIFAR-10 上达 92.05%，**超越所有无归一化基线**
2. 在神经形态数据集上（DVS-Gesture, CIFAR10-DVS）**超越多个使用 BN 的方法**
3. 在 TinyImageNet 上达 50.29%，证明可扩展到更大数据集
4. E-I Init 和 E-I Prop 的每个组件都是必需的——缺少任何一个都导致训练崩溃

### 消融实验

- 无 E-I Init → 训练失败（发放率崩溃）
- 无自适应稳定化 → 数值爆炸
- 无 STE → 梯度方向错误
- 无梯度缩放 → 抑制路径梯度过大

## 亮点

1. **首次在深度 SNN 中实现无归一化训练**的同时保持竞争力性能
2. **生物合理性与工程性能的平衡**：E-I 回路不仅是正则化技巧，也是生物建模
3. **理论分析完善**：从指数分布推导到增益控制条件
4. **为大规模皮层计算模拟提供平台**

## 局限性

1. 与使用 BN 的 SNN 仍有 ~3% 精度差距
2. 固定 4:1 的 E-I 比例是否最优未探索
3. 快速脉冲近似将抑制性神经元简化为 ReLU，可能过度简化
4. 仅在分类任务上验证，未测试生成或序列建模任务

## 相关工作
- **SNN 归一化**: BNTT, tdBN, TEBN, TAB — BN 的 SNN 变体
- **E-I 网络**: Cornford et al. (2021) — ANN 中的 E-I 网络
- **SNN 训练**: STBP, TEBN — 代理梯度和归一化技术

## 评分
- **创新性**: ⭐⭐⭐⭐ — E-I 回路替代归一化的思路新颖且有生物依据
- **实验充分性**: ⭐⭐⭐⭐ — 多数据集多架构验证
- **写作质量**: ⭐⭐⭐⭐ — 从生物原理到工程实现的推导清晰
- **实用性**: ⭐⭐⭐ — 性能差距仍存在，但为 NeuroAI 提供重要基础

<!-- RELATED:START -->

## 相关论文

- [TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](../../AAAI2026/others/tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)
- [Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)

<!-- RELATED:END -->
