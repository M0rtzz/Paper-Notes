---
title: >-
  [论文解读] Robust Spiking Neural Networks Against Adversarial Attacks
description: >-
  [ICLR2026][AI安全][脉冲神经网络] 从理论上证明阈值邻近脉冲神经元是直接训练SNN对抗鲁棒性的关键瓶颈（它们既设定了对抗攻击强度的理论上界，又最容易发生状态翻转），并提出Threshold Guarding Optimization (TGO) 方法——通过膜电位约束+噪声LIF神经元双管齐下，在多种对抗攻击场景下取得SOTA鲁棒性，且推理阶段零额外开销。
tags:
  - ICLR2026
  - AI安全
  - 脉冲神经网络
  - 对抗鲁棒性
  - 膜电位优化
  - 阈值邻近神经元
  - 噪声LIF模型
---

# Robust Spiking Neural Networks Against Adversarial Attacks

**会议**: ICLR2026  
**arXiv**: [2602.20548](https://arxiv.org/abs/2602.20548)  
**代码**: 待确认  
**领域**: ai_safety  
**关键词**: 脉冲神经网络, 对抗鲁棒性, 膜电位优化, 阈值邻近神经元, 噪声LIF模型

## 一句话总结
从理论上证明阈值邻近脉冲神经元是直接训练SNN对抗鲁棒性的关键瓶颈（它们既设定了对抗攻击强度的理论上界，又最容易发生状态翻转），并提出Threshold Guarding Optimization (TGO) 方法——通过膜电位约束+噪声LIF神经元双管齐下，在多种对抗攻击场景下取得SOTA鲁棒性，且推理阶段零额外开销。

## 研究背景与动机

**领域现状**：脉冲神经网络 (SNN) 凭借事件驱动机制和生物可信的脉冲传递，成为节能型神经形态计算的重要范式。基于代理梯度的直接训练方法（如STBP/BPTT）已使SNN在分类任务上逼近ANN性能。

**现有痛点**：直接训练的SNN继承了ANN的对抗脆弱性——精心设计的微小扰动即可导致分类错误。现有防御方法如对抗训练 (AT)、正则化对抗训练 (RAT) 带来额外训练开销且可移植性有限。

**核心矛盾**：已有针对SNN的鲁棒性优化（如梯度稀疏正则化SR、进化泄漏因子FEEL-SNN等）仅在与AT/RAT结合时才有显著效果，且缺乏对SNN鲁棒性瓶颈的统一理论分析。

**本文目标** 找到直接训练SNN对抗脆弱性的根本原因，并设计无需额外推理开销的防御方法。

**切入角度**：从脉冲神经元的膜电位动态出发，发现阈值邻近神经元同时放大了梯度攻击路径上界和状态翻转概率。

**核心 idea**：将膜电位推离阈值 + 引入噪声脉冲机制 → 降低对抗攻击理论上界 + 减小状态翻转概率。

## 方法详解

### 理论分析：阈值邻近神经元的双重脆弱性

**脆弱性1 — 最大潜在攻击路径上界：** 对抗攻击的最大潜在强度 $\mathcal{R}_{\text{adv}}(f,x,\epsilon)$ 与模型Jacobian矩阵的 $\ell_2$ 范数正相关。由于代理梯度在阈值附近取峰值，阈值邻近神经元越多，$\|J_f(x)\|_2^2$ 越大，从而抬高了对抗扰动强度的理论上界。

**脆弱性2 — 状态翻转概率：** 定理1证明，当高斯噪声 $\eta[t] \sim \mathcal{N}(0,\sigma^2)$ 作用于膜电位时，神经元状态翻转概率 $P_{\text{flip}}$ 随膜电位接近阈值而单调递增。定理2进一步证明，阈值邻近神经元越多，扰动球 $B_\epsilon(x)$ 内可触达的激活区域数 $K$ 越大，对抗鲁棒性上界越松。

### TGO方法：两个核心组件

**组件1 — 膜电位约束 (MC)：** 在每层脉冲神经元的损失函数中加入约束项，惩罚膜电位落入阈值 $V_{\text{th}}$ 附近 $\delta$ 邻域的神经元：

$$\mathcal{C}(V(t)_l) = \frac{1}{TN}\sum_{i=1}^{n}\max(0, \delta - |V(t)_i - V_{\text{th}}|)$$

总损失采用拉格朗日约束形式 $\mathcal{L}(\mathbf{x},\lambda) = \mathcal{L}_{\text{oss}}(\mathbf{x}) + \lambda \sum_l \mathcal{C}(V(t)_l)$，其中 $\lambda$ 采用余弦退火动态调整（初期小值允许探索，后期大值强化约束），避免固定 $\lambda$ 导致的收敛困难。

**组件2 — 噪声LIF神经元 (NLIF)：** 在膜电位中注入高斯白噪声 $\xi[t]$，将确定性发放机制转为概率性发放。理论推导表明，当膜电位接近阈值时（$z^2 < 1$），翻转概率关于噪声标准差 $\sigma$ 单调递减，即适当增大噪声可降低阈值邻近神经元的状态翻转敏感度。

**协同机制：** MC将大部分神经元的膜电位推离阈值；对于训练过程中仍需停留在阈值附近的关键神经元，NLIF进一步降低其翻转概率。两者互补而非独立。

### 关键超参数
- $\lambda_{\max}$：膜电位约束强度的上限，WRN-16设为0.4，VGG-11设为0.6。更大的 $\lambda_{\max}$ 增强鲁棒性但降低Clean准确率
- $\delta$：阈值邻域宽度，控制惩罚的触发范围
- 噪声 $\sigma$：NLIF的噪声标准差，需在鲁棒性提升和训练稳定性之间平衡
- 训练时间步 $T=4$：所有SNN模型统一使用4个时间步的仿真

## 实验关键数据

### 主实验：CIFAR-10 WRN-16 多攻击对比

| 训练策略 | 方法 | Clean | FGSM | RFGSM | PGD10 | PGD20 | PGD40 |
|---------|------|-------|------|-------|-------|-------|-------|
| BPTT | Vanilla | 93.32 | 14.05 | 31.21 | 0.00 | 0.00 | 0.00 |
| BPTT | **TGO** | 88.79 | **51.40** | **71.38** | **6.14** | 1.52 | 0.45 |
| AT | AT | 91.32 | 39.14 | 74.31 | 17.45 | 14.41 | 12.93 |
| AT | **TGO** | 88.16 | **63.03** | **79.69** | **35.01** | **24.76** | **20.11** |
| RAT | RAT | 91.44 | 42.02 | 75.89 | 19.81 | 16.24 | 14.18 |
| RAT | **TGO** | 87.33 | **69.16** | **79.28** | **47.69** | **38.07** | **33.13** |

### 消融实验：CIFAR-100 VGG-11 各组件贡献

| MC | NLIF | Clean (BPTT) | FGSM (BPTT) | Clean (RAT) | FGSM (RAT) | PGD40 (RAT) |
|----|------|-------------|-------------|-------------|-------------|-------------|
| ✗ | ✗ | 71.4 | 5.9 | 67.8 | 20.9 | 6.9 |
| ✓ | ✗ | 64.3 | 17.1 (+11.2) | 61.4 | 26.2 (+5.3) | 6.2 |
| ✗ | ✓ | 70.6 | 8.1 (+2.1) | 68.1 | 25.2 (+4.3) | 9.1 (+2.2) |
| ✓ | ✓ | 66.9 | **21.5 (+15.5)** | 63.3 | **33.8 (+13.0)** | **9.3 (+2.4)** |

### 高级攻击：MTPGD & APGD (CIFAR-100 WRN-16)

| 方法 | MTPGD-7 | MTPGD-40 | APGD-7 | APGD-40 |
|------|---------|----------|--------|---------|
| AT | 10.01 | 3.92 | 9.34 | 3.62 |
| SR+AT | 16.88 | 7.33 | 14.48 | 7.20 |
| **TGO+AT(EoT)** | **21.23** | **7.40** | **18.93** | **7.53** |

- TGO将阈值邻近神经元数量减少约40%，验证了理论假设
- 损失景观分析显示TGO优化后的SNN梯度轨迹更平滑，有效规避局部最优陷阱

## 亮点与洞察
- **理论驱动设计**：不是盲目套用ANN防御方法，而是从SNN的脉冲机制出发识别鲁棒性瓶颈，再有针对性地设计防御组件
- **推理零开销**：MC仅影响训练损失，NLIF噪声可在推理时移除（训练时的概率化已使权重分布更鲁棒），推理阶段与标准SNN相同
- **高兼容性**：TGO可与BPTT/AT/RAT任意组合，在所有组合下均带来显著提升
- **阈值邻近神经元减少40%**：可视化直观验证了理论分析的正确性

## 局限与展望
- **Clean准确率下降3-5%**：推离阈值的约束不可避免地牺牲了部分正常分类性能，存在鲁棒性-准确率权衡
- **仅验证图像分类**：未在目标检测、语义分割等下游任务上验证通用性
- **噪声标准差 $\sigma$ 的选取**：文中未充分讨论如何为不同架构和数据集自动确定最优 $\sigma$
- **自适应攻击评估有限**：虽然测试了APGD和EoT，但未采用AutoAttack等更完整的自适应攻击套件
- **改进方向**：可探索逐层自适应 $\delta$ 和 $\sigma$，或结合知识蒸馏缓解Clean准确率损失

## 相关工作与启发
- **vs SR (梯度稀疏正则化)**：SR直接约束梯度稀疏性，TGO从根源（膜电位分布）出发，间接实现更强的梯度稀疏效果。实验中TGO在所有攻击场景下均优于SR
- **vs FEEL-SNN (进化泄漏因子)**：FEEL-SNN通过随机膜电位衰减增强鲁棒性，但仅在AT协同下有效；TGO即使在BPTT策略下也能大幅提升FGSM鲁棒性（+37%）
- **vs ANN对抗训练**：AT/RAT从ANN迁移而来，未考虑SNN的脉冲特性；TGO利用脉冲机制的特殊性设计防御，与AT/RAT形成补充关系

## 评分
- 新颖性: ⭐⭐⭐⭐ 从阈值邻近神经元角度建立SNN鲁棒性瓶颈理论，视角独到
- 实验充分度: ⭐⭐⭐⭐ 多架构×多攻击×多训练策略，消融完整，但缺少AutoAttack评估
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，方法动机清晰，图示直观
- 价值: ⭐⭐⭐⭐ 为SNN安全部署提供了理论基础和实用工具，推理零开销是重要优势

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](../../AAAI2026/ai_safety/mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](../../AAAI2026/ai_safety/yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)
- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](../../AAAI2026/ai_safety/hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)

<!-- RELATED:END -->
