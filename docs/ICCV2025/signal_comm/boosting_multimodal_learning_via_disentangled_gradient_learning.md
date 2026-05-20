---
title: >-
  [论文解读] Boosting Multimodal Learning via Disentangled Gradient Learning
description: >-
  [ICCV 2025][信号/通信][多模态学习] 本文揭示了多模态学习中模态编码器和融合模块之间的优化冲突——融合模块会抑制回传到各模态编码器的梯度，导致即使是优势模态也比单模态模型表现差，并提出解耦梯度学习（DGL）框架通过截断融合模块到编码器的梯度并用独立的单模态损失替代来解决此问题。
tags:
  - "ICCV 2025"
  - "信号/通信"
  - "多模态学习"
  - "梯度解耦"
  - "模态欠优化"
  - "梯度调制"
  - "融合模块优化"
---

# Boosting Multimodal Learning via Disentangled Gradient Learning

**会议**: ICCV 2025  
**arXiv**: [2507.10213](https://arxiv.org/abs/2507.10213)  
**代码**: [https://github.com/shicaiwei123/ICCV2025-GDL](https://github.com/shicaiwei123/ICCV2025-GDL)  
**领域**: 信号通信  
**关键词**: 多模态学习, 梯度解耦, 模态欠优化, 梯度调制, 融合模块优化

## 一句话总结
本文揭示了多模态学习中模态编码器和融合模块之间的优化冲突——融合模块会抑制回传到各模态编码器的梯度，导致即使是优势模态也比单模态模型表现差，并提出解耦梯度学习（DGL）框架通过截断融合模块到编码器的梯度并用独立的单模态损失替代来解决此问题。

## 研究背景与动机

**领域现状**：多模态学习利用来自多种传感器的互补信息（视觉+音频、RGB+深度等）来提升任务性能。研究聚焦于融合技术设计（tensor-based、attention-based等）。然而，简单组合多模态并不总是带来预期的性能提升。

**现有痛点**：
   - 已知问题：多模态模型可能比单模态模型表现更差（"under-optimization"问题）
   - 现有解释：优势模态会抑制弱势模态的优化（"模态不平衡"），因此出现了OGM、PMR、AGM等梯度调制方法来增大弱势模态的梯度
   - **被忽视的问题**：这些方法只关注改善弱势模态，却忽略了**优势模态也比其自身的单模态baseline表现差**。如Fig.1所示，音频（优势模态）在多模态模型中的表现也不如它在单模态模型中的表现

**核心矛盾**：为什么优势模态在多模态模型中也表现不佳？现有的"模态不平衡"解释无法回答这个问题。本文揭示了真正原因：**融合模块会抑制回传到模态编码器的梯度**，这种抑制随训练进行而加剧。

**本文目标** 解释并解决多模态学习中所有模态（包括优势模态）的欠优化问题。

**切入角度**：通过数学推导证明：在多模态模型中，融合模块产生的梯度经链式法则回传到编码器时，相比单模态模型的梯度会被缩小。具体而言，融合操作引入了一个缩放因子，使得 $\|g_{\theta_1}^{Multi}\| < \|g_{\theta_1}^{Uni}\|$。

**核心 idea**：截断融合模块到编码器的梯度路径，用独立的单模态损失为编码器提供不受干扰的优化信号，同时移除单模态损失到融合模块的梯度避免反向干扰。

## 方法详解

### 整体框架
DGL在标准多模态模型的基础上做了三个梯度操作（如Fig.2所示）：(1) 截断多模态损失 $L^{Multi}$ 回传到各模态编码器 $\phi_1, \phi_2$ 的梯度；(2) 为每个编码器引入独立的单模态损失 $L^{Uni}$ 提供直接梯度；(3) 截断单模态损失回传到融合模块 $\phi_\tau$ 的梯度。三步操作完全解耦了编码器和融合模块的优化过程。

### 关键设计

1. **梯度抑制的理论分析**:

    - 功能：从数学上证明融合模块抑制编码器梯度
    - 核心思路：考虑两个模态 $m_1, m_2$，编码器输出 $z^{m_1}, z^{m_2}$ 经融合模块 $\phi_\tau$ 得到融合表示 $z^\tau$。由链式法则，多模态损失对编码器1的梯度 $g_{\theta_1}^{Multi} = \frac{\partial L}{\partial z^\tau} \cdot \frac{\partial z^\tau}{\partial z^{m_1}} \cdot \frac{\partial z^{m_1}}{\partial \theta_1}$。关键在于中间项 $\frac{\partial z^\tau}{\partial z^{m_1}}$——由于融合模块混合了两个模态的信息，这个Jacobian矩阵的范数小于1，导致梯度被缩小。而单模态模型没有这个中间项，梯度更大
    - 设计动机：提供了理论解释——为什么优势模态也表现差（之前只有经验观察无理论解释）

2. **截断融合→编码器梯度**:

    - 功能：阻止多模态损失的梯度经融合模块回传到编码器
    - 核心思路：在 $z^{m_1}$ 和 $z^{m_2}$ 进入融合模块之前做 `stop_gradient` 操作（对编码器方向）。这样融合模块仍能正常接收来自 $L^{Multi}$ 的梯度来优化自身，但不会将被抑制的梯度传回编码器
    - 设计动机：消除融合模块对编码器梯度的抑制效应

3. **独立单模态损失（通过模态dropout）**:

    - 功能：为每个编码器提供独立的优化信号
    - 核心思路：使用parameter-free的模态dropout技术——每次前向传播时随机屏蔽一个模态的输入，使融合模块只接收单一模态的信息，从而可以计算 $L^{Uni}_{m_1}$ 和 $L^{Uni}_{m_2}$。这些单模态损失直接回传到对应编码器
    - 设计动机：截断了融合→编码器的梯度后，编码器需要新的梯度来源。模态dropout不引入额外参数，简洁优雅

4. **截断单模态→融合模块梯度**:

    - 功能：阻止单模态损失的梯度回传到融合模块
    - 核心思路：在计算单模态损失时，对融合模块的参数做 `stop_gradient`
    - 设计动机：单模态损失是为优化编码器设计的，其梯度信号如果传到融合模块可能与 $L^{Multi}$ 的梯度冲突，干扰融合模块的正常优化

### 损失函数 / 训练策略
- 总损失：$L = L^{Multi} + \lambda_1 L^{Uni}_{m_1} + \lambda_2 L^{Uni}_{m_2}$
- $L^{Multi}$ 只更新融合模块和分类器
- $L^{Uni}_{m_1}$ 和 $L^{Uni}_{m_2}$ 只更新对应的模态编码器
- 无需修改网络结构，仅操作梯度流，通用性极强

## 实验关键数据

### 主实验

| 方法 | CREMA-D (A-V) | Kinetics (A-V) | NYU-Depth (RGB-D) | 平均提升 |
|------|-------------|---------------|-------------------|---------|
| Vanilla | 63.2 | 67.5 | 48.3 | baseline |
| OGM | 66.1 | 69.3 | 50.1 | +2.5 |
| PMR | 65.8 | 68.9 | 49.8 | +2.2 |
| AGM | 67.3 | 70.1 | 51.2 | +3.3 |
| MLA | 68.0 | 70.5 | 51.5 | +3.7 |
| **DGL** | **70.2** | **72.1** | **53.4** | **+5.6** |

### 消融实验

| 配置 | CREMA-D | 说明 |
|------|--------|------|
| Full DGL | 70.2 | 完整模型 |
| w/o 截断融合→编码器 | 66.5 | 梯度解耦最关键 |
| w/o 单模态损失 | 67.1 | 编码器需要独立梯度源 |
| w/o 截断单模态→融合 | 68.9 | 反向干扰也有影响 |
| 仅截断融合→编码器 | 67.8 | 三步缺一不可 |

### 关键发现
- DGL在多种模态组合（音频-视觉、RGB-深度）、多种任务（分类、检测、分割）、多种融合方法（concatenation、attention、tensor）上都有效
- 梯度解耦（截断融合→编码器梯度）是最关键的组件，去掉后掉3.7%
- DGL不仅改善了弱势模态，还显著改善了优势模态——这是之前方法做不到的
- 在有dense cross-modal interaction的融合框架上效果尤为显著
- DGL是模态无关、融合方法无关、模型结构无关的，通用性极高

## 亮点与洞察
- **理论贡献**：从数学上证明了融合模块抑制编码器梯度这一现象，解释了"为什么连优势模态也表现差"这个长期困惑。这比之前纯经验观察的"模态不平衡"解释更深入
- **极简设计**：DGL仅操作梯度流（三个stop_gradient操作），不修改网络结构、不引入额外参数。实现极简（几行代码）但效果显著
- **高通用性**：DGL是modality-agnostic、fusion-method-agnostic、model-agnostic的，几乎可以即插即用到任何多模态模型中
- **解决了优势模态欠优化**：之前的方法（OGM、PMR等）只关注弱势模态，实际上可能损害优势模态。DGL同时改善所有模态

## 局限与展望
- 模态dropout在训练时增加了前向次数（每个模态一次），训练成本增加
- 理论分析基于简化假设（线性分类器、特定融合形式），对复杂非线性融合的适用性需进一步论证
- 超参数 $\lambda_1, \lambda_2$ 的设定可能影响效果
- 未在大规模预训练多模态模型（如CLIP、LLaVA）上验证
- 可以考虑自适应调节解耦程度而非完全截断

## 相关工作与启发
- **vs OGM/AGM（梯度调制）**: 梯度调制方法试图平衡不同模态的梯度大小，但本质上仍受融合模块梯度抑制影响。DGL从根源解决问题
- **vs MLA（交替优化）**: MLA将联合训练变为交替的单模态训练，但完全放弃了联合训练的信号。DGL保留了融合模块的联合训练，只解耦编码器
- **vs MMPareto**: Pareto方法寻找多目标优化的帕累托最优，但不解决梯度抑制的根本问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论洞察深刻（梯度抑制证明），解决方案优雅简洁
- 实验充分度: ⭐⭐⭐⭐ 多模态、多任务、多融合方法验证，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，动机-分析-方法-验证逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 对多模态学习领域有基础性贡献，方法通用性极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[ICLR 2026\] Learning Molecular Chirality via Chiral Determinant Kernels](../../ICLR2026/signal_comm/learning_molecular_chirality_via_chiral_determinant_kernels.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ICML 2026\] Meta-learning Structure-Preserving Dynamics](../../ICML2026/signal_comm/meta-learning_structure-preserving_dynamics.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](../../ACL2026/signal_comm/ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)

</div>

<!-- RELATED:END -->
