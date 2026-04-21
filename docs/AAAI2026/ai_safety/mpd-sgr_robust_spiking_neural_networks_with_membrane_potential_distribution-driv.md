---
title: >-
  [论文解读] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization
description: >-
  [AAAI 2026][AI安全][脉冲神经网络] 从理论上建立了 SNN 鲁棒性误差与代理梯度（SG）幅值之间的联系，揭示减少膜电位分布（MPD）与 SG 梯度可用区间的重叠比例可有效降低对抗扰动敏感度，据此提出 MPD-SGR 正则化方法，在 vanilla training 和 adversarial training 设置下均大幅超越现有 SNN 防御方法。
tags:
  - AAAI 2026
  - AI安全
  - 脉冲神经网络
  - 对抗鲁棒性
  - 代理梯度
  - 膜电位分布
  - 正则化
---

# MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization

**会议**: AAAI 2026  
**arXiv**: [2511.12199](https://arxiv.org/abs/2511.12199)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 脉冲神经网络, 对抗鲁棒性, 代理梯度, 膜电位分布, 正则化

## 一句话总结

从理论上建立了 SNN 鲁棒性误差与代理梯度（SG）幅值之间的联系，揭示减少膜电位分布（MPD）与 SG 梯度可用区间的重叠比例可有效降低对抗扰动敏感度，据此提出 MPD-SGR 正则化方法，在 vanilla training 和 adversarial training 设置下均大幅超越现有 SNN 防御方法。

## 研究背景与动机

**脉冲神经网络（SNN）** 模拟大脑以二进制脉冲编码信息的方式，相比 ANN 具有天然的鲁棒性优势——噪声过滤特性和脉冲编码的随机性被认为是关键因素。然而随着代理梯度（SG）方法使深层 SNN 训练成为可能，SNN 也开始暴露于基于梯度的对抗攻击威胁之下。

**现有 SNN 鲁棒性研究的三条路线**：

**结构参数**：泄漏因子 τ、阈值 $v_{th}$ 等（如 FEEL 演化泄漏因子）→ 利用膜电位泄漏特性的噪声过滤效果

**神经编码**：Poisson 编码的随机性比直接编码更鲁棒（如 NDL、StoG）→ 利用信息传递过程中的噪声减弱

**从 ANN 借鉴**：对抗训练（AT）、Lipschitz 正则化（RAT）→ 但未充分考虑 SNN 的独特性

**被忽视的关键因素**：梯度幅值反映了模型对输入扰动的敏感度，而 SNN 中梯度幅值主要由**膜电位分布（MPD）与 SG 函数的交互**决定。现有工作（InfLoR-SNN、RecDis-SNN、LSG 等）研究 MPD 与 SG 的对齐是为了改善训练性能，但忽略了其对鲁棒性的影响。

**核心动机**：减少 MPD 与 SG 梯度可用区间的重叠比例 → 降低 SG 幅值 → 减小对扰动的敏感度 → 提升鲁棒性。但需谨慎平衡——过度减少重叠会阻碍梯度传播，影响训练。

## 方法详解

### 整体框架

MPD-SGR 在训练过程中对 SNN 每一层、每个通道、每个时间步的膜电位分布施加正则化，约束其与 SG 函数梯度可用区间的重叠面积 Ω，在保证训练有效性的同时增强鲁棒性。

### 关键设计

1. **鲁棒性误差的理论分析**

   对抗扰动引起的误差上界：
   $$|\mathcal{L}(x+\delta) - \mathcal{L}(x)| \leq |\delta \odot \nabla_x \mathcal{L}(x)|_1 + g(\delta, x)$$

   利用 LIF 动力学和 BPTT 将输入梯度优化重写为网络内部梯度优化：
   $$\min \sum_t \left|\frac{1}{L} \sum_{l=1}^{L} (P_1 \cdot P_2 \cdot P_3) \frac{\partial \mathcal{L}}{O_l^T}\right|_1$$

   其中三个关键项：
   - $P_1$：扰动项（与泄漏因子相关 → FEEL 方法的依据）
   - $P_2$：权重项（→ Lipschitz 正则化的依据）
   - $P_3 = \prod_v \frac{\partial O_v^t}{\partial U_v^t}$：**SG 项（本文关注且此前被忽视的因素）**

   **设计动机**：降低 SG 项 $P_3$ 的幅值可以直接减小鲁棒性误差上界。而 SG 幅值由膜电位在 SG 梯度可用区间内的比例决定。

2. **膜电位分布的理论建模**

   **定理 1**：在带 tdBN 的迭代 LIF 模型中，膜电位服从高斯分布：
   $$\overline{U}_c^l(t) \sim \mathcal{N}(\beta_c D(\tau, t) - S(t), (\lambda_c \alpha V_{th})^2 D(\tau^2, t))$$

   其中 $D(\tau, t) = \sum_{i=1}^{t} \tau^{t-i}$ 是累积衰减函数。

   即 MPD 的均值 μ 和标准差 σ 由 tdBN 参数（$\beta_c$, $\lambda_c$）、LIF 参数（$\tau$, $V_{th}$）和时间步 t 共同决定 → MPD 可通过学习网络参数来优化。

3. **MPD-SG 重叠面积的推导与正则化**

   假设 SG 函数（三角形）的梯度可用区间为 $[-\gamma, \gamma]$，MPD 为 $\mathcal{N}(\mu, \sigma^2)$，重叠面积：
   $$\Omega = \Phi\left(\frac{\mu + \gamma}{\sigma}\right) - \Phi\left(\frac{\mu - \gamma}{\sigma}\right)$$

   其中 $\Phi$ 是标准正态 CDF。最终 MPD-SGR 正则化损失：
   $$\mathcal{L}_{MPD-SGR}^b = \frac{1}{LCT} \sum_{l,c,t} \left[\Phi\left(\frac{\mu_c^l(t) + \gamma}{\sigma_c^l(t)}\right) - \Phi\left(\frac{\mu_c^l(t) - \gamma}{\sigma_c^l(t)}\right)\right]$$

   对每一层 l、每个通道 c、每个时间步 t（除最后线性输出层）的重叠面积求和。

   **设计动机**：Ω 越小 → SG 幅值越小 → 模型对扰动越不敏感。但 Ω 过小会阻断梯度传播，系数 η 平衡鲁棒性与训练效果。

### 损失函数 / 训练策略

$$\mathcal{L}^b = \mathcal{L}_{task}^b + \eta \mathcal{L}_{MPD-SGR}^b$$

- $\mathcal{L}_{task}$：标准分类交叉熵损失
- η 控制正则化强度
- 对抗训练（AT）时使用 PGD 对抗样本（k=2, ε=2/255）
- 攻击设置：ε=8/255, PGD/BIM 迭代步数 k=7, 步长 α=0.01

## 实验关键数据

### 主实验：与 SOTA 方法对比（VGG11, T=8）

**Vanilla Training**：

| 方法 | Clean | FGSM | PGD | BIM |
|------|:---:|:---:|:---:|:---:|
| REG | 92.49 | 25.18 | 0.88 | 0.60 |
| StoG | 91.64 | 16.22 | 0.28 | 0.12 |
| DLIF | 92.01 | 11.52 | 0.08 | 0.06 |
| FEEL | 90.08 | 29.17 | 6.67 | 5.99 |
| SR | 91.04 | 31.72 | 8.55 | 7.28 |
| **MPD-SGR** | **91.63** | **47.59** | **20.55** | **16.85** |
| **提升** | -0.86 | +15.87 | +12.00 | +9.57 |

**Adversarial Training**：

| 方法 | Clean | FGSM | PGD | BIM |
|------|:---:|:---:|:---:|:---:|
| RAT | 91.41 | 45.00 | 22.95 | 20.80 |
| FEEL | 89.00 | 45.62 | 29.52 | 28.39 |
| SR | 88.26 | 44.28 | 28.63 | 27.03 |
| **MPD-SGR** | **90.69** | **59.27** | **33.38** | **32.61** |
| **提升** | -0.72 | +13.52 | +3.86 | +4.22 |

CIFAR-100 上同样大幅领先（Vanilla: FGSM +18.35%；AT: FGSM +16.35%）。

### 消融实验：不同 SG 函数（CIFAR-10, VGG11）

| 模型+SG函数 | 方法 | Clean | FGSM | PGD | BIM |
|------------|------|:---:|:---:|:---:|:---:|
| VGG11+Rectangular | REG | 91.85 | 24.00 | 3.13 | 2.33 |
| VGG11+Rectangular | **Ours** | 91.23 | **43.28** | **15.82** | **14.20** |
| VGG11+Sigmoid | REG | 92.15 | 19.42 | 0.24 | 0.15 |
| VGG11+Sigmoid | **Ours** | 89.38 | **37.25** | **9.26** | **7.23** |
| VGG11+Superspike | REG | 86.82 | 21.39 | 0.82 | 0.50 |
| VGG11+Superspike | **Ours** | 84.45 | **43.42** | **6.32** | **4.50** |

→ MPD-SGR 在三种 SG 函数上均一致提升鲁棒性，验证了方法的泛化性。

### 不同编码方式（Tiny-ImageNet, VGG16）

| 编码方式 | 方法 | Clean | FGSM | PGD |
|---------|------|:---:|:---:|:---:|
| Direct (DIR) | 基线 | 57.90 | 2.04 | 0.01 |
| Direct (DIR) | +Ours | 54.78 | **14.33** | **5.72** |
| Poisson (POS) | 基线 | 48.14 | 6.79 | 2.68 |
| Poisson (POS) | +Ours | 47.83 | **20.42** | **8.21** |
| RSC | 基线 | 47.47 | 22.63 | 13.75 |
| RSC | +Ours | 46.98 | **35.06** | **17.60** |

→ MPD-SGR 与不同脉冲编码方案兼容，可叠加使用。

### 关键发现

1. **Vanilla Training 下效果最显著**：无 AT 时基线 SNN 在 PGD 下几乎为 0%，MPD-SGR 提升到 ~20%
2. **Clean 精度损失极小**：-0.86%（CIFAR-10）→ 鲁棒性-精度 trade-off 优异
3. **SR 方法虽然也提升鲁棒性，但 clean 精度暴跌**（CIFAR-100 上 66.76%），MPD-SGR 则保持 70.42% → 实用性更强
4. **黑盒攻击下同样有效** → 鲁棒性源于方法的内在特性而非梯度混淆
5. **非梯度攻击（随机噪声）下也适用**：Gaussian Noise 下 CIFAR-100 精度 53.01% vs FEEL 的 32.63%
6. **跨架构有效**：VGG11 和 WRN16 上一致提升

## 亮点与洞察

1. **理论贡献扎实**：
    - 建立了 SG 幅值 → 鲁棒性误差的形式化关系
    - 证明了 MPD 在 LIF+tdBN 下的高斯分布形式（定理 1）
    - 推导了 MPD-SG 重叠面积 Ω 的解析表达式
2. **正则化设计优雅**：基于 CDF 的重叠面积公式可以直接反向传播，无需额外近似
3. **极强的泛化性**：跨 SG 函数、跨编码方式、跨架构、跨攻击类型均有效
4. **与现有方法正交**：可以和对抗训练、编码方法叠加使用
5. **连接了 SG 优化和鲁棒性两个研究方向**：此前 SG-MPD 对齐只用于改善训练，本文首次将其用于增强鲁棒性

## 局限与展望

- η 参数需要调节（虽然附录有分析但没有自适应机制）
- 理论分析基于三角形 SG 函数，虽然实验验证了其他 SG 函数也有效，但理论扩展到任意 SG 函数有待完善
- 仅在图像分类任务上验证，事件驱动任务（如 neuromorphic vision）的适用性未探索
- 时间步 T 固定（T=4 或 T=8），更长时间序列上的效果和效率待研究
- 定理 1 的推导假设了 tdBN 的使用 → 不使用 tdBN 的 SNN 需要重新推导 MPD

## 相关工作与启发

- **与 InfLoR-SNN / RecDis-SNN 的区别**：它们约束 MPD 是为了改善训练（确保适当比例的膜电位有梯度），本文约束 MPD 是为了增强鲁棒性（减少有梯度的比例）→ 目标相反但互补
- **与 FEEL 的区别**：FEEL 通过频率编码和注意力机制抑制不同频率范围的噪声 → 作用于输入层；MPD-SGR 作用于网络内部所有层的梯度传播
- **对 SNN 社区的意义**：揭示了 SG 不仅影响训练性能也影响鲁棒性 → 为 SG 设计提供了新的考量维度

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（SG-MPD 交互用于鲁棒性的全新视角，理论推导严谨）
- 实验充分度: ⭐⭐⭐⭐⭐（3 数据集 × 2 架构 × 4 攻击 × 3 SG 函数 × 3 编码 × AT/非AT）
- 写作质量: ⭐⭐⭐⭐（理论→方法→实验的逻辑链清晰）
- 价值: ⭐⭐⭐⭐⭐（为 SNN 鲁棒性提供了理论根基扎实的通用防御策略）

<!-- RELATED:START -->

## 相关论文

- [Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks](angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)
- [Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference](diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)
- [Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)

<!-- RELATED:END -->
