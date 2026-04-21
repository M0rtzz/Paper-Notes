---
title: >-
  [论文解读] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence
description: >-
  [ICLR2026][Grokking] 通过 Neural Collapse 的视角统一解释 Grokking（延迟泛化）和 Information Bottleneck（压缩阶段）两大训练后期现象，证明群体类内方差的收缩是两者的共同关键因素，并揭示训练损失收敛与 Neural Collapse 发生存在由 weight decay 控制的不同时间尺度。
tags:
  - ICLR2026
  - Grokking
  - 信息瓶颈
  - Neural Collapse
  - 训练动态
  - 类内方差
  - 泛化理论
  - Lyapunov 时间尺度
---

# Explaining Grokking and Information Bottleneck through Neural Collapse Emergence

**会议**: ICLR2026  
**arXiv**: [2509.20829](https://arxiv.org/abs/2509.20829)  
**代码**: [keitaroskmt/collapse-dynamics](https://github.com/keitaroskmt/collapse-dynamics)  
**领域**: others  
**关键词**: Grokking, 信息瓶颈, Neural Collapse, 训练动态, 类内方差, 泛化理论, Lyapunov 时间尺度

## 一句话总结
通过 Neural Collapse 的视角统一解释 Grokking（延迟泛化）和 Information Bottleneck（压缩阶段）两大训练后期现象，证明群体类内方差的收缩是两者的共同关键因素，并揭示训练损失收敛与 Neural Collapse 发生存在由 weight decay 控制的不同时间尺度。

## 背景与动机
1. **Grokking 现象之谜**：训练损失早已收敛但测试精度在长时间后突然跃升，过拟合解突然转为泛化解，机制不明。
2. **Information Bottleneck 的两阶段**：DNN 训练中先出现拟合阶段（I(Z;X) 和 I(Z;Y) 同时增大），后出现压缩阶段（I(Z;X) 减小而 I(Z;Y) 保持），但压缩阶段的触发机制缺乏严格理论解释。
3. **两现象的共性**：Grokking 和 IB 压缩都发生在训练后期，暗示网络内部在此阶段发生了某种共同的结构性变化。
4. **Neural Collapse 的潜力**：Neural Collapse 描述训练后期表示空间的几何结构（类内坍缩、类均值形成 ETF），但其与上述后期现象的联系从未被建立。
5. **现有理论的局限**：Grokking 解释多停留在参数压缩、复杂度度量等经验层面；IB 分析在连续确定性网络中互信息可能无穷大，且缺乏与网络参数的直接联系。
6. **时间尺度分析缺失**：即便知道类内方差收缩可改善泛化，也需要理解其收敛速度相对于训练损失收敛的关系，才能解释为何泛化/压缩"延迟"发生。

## 方法详解
本文是理论驱动的工作，核心贡献是三组定理及其相互关联。

### 第一步：类内方差是统一关键量
- **Grokking 的泛化界 (Theorem 3.2)**：对固定 feature extractor g 和分类器 W，推导基于 Chebyshev 不等式的分类错误上界。上界的关键分母项是群体类内方差 E[||g̃(X) - E[g̃(X)|Y]||²]——该值越小，上界越紧，泛化越好。
- **IB 冗余信息界 (Theorem 3.4)**：在表示 Z = g(X) + B_g·E（加微小高斯噪声）的设定下，证明冗余信息 I(Z;X) - I(Z;Y) = I(Z;X|Y) 被群体类内方差上界控制。即类内方差收缩直接减少冗余信息，对应 IB 压缩阶段。

### 第二步：群体方差可由经验方差逼近
- **方差集中不等式 (Theorem 4.1)**：基于谱范数的均匀收敛分析，证明群体类内方差与训练集经验类内方差之差为 O(1/√n_c)。这为后续用 RNC1（基于训练集的 rescaled NC1 度量）代替群体方差提供了理论保障。
- **RNC1 定义**：RNC1 = (1/B_g²)·Tr(Σ_W)，直接度量归一化后训练集类内方差，比传统 NC1（还除以类间方差）更直接对应泛化分析中需要的量。

### 第三步：Neural Collapse 的时间尺度分析
- **双时间尺度定理 (Theorem 4.3)**：在带 weight decay λ 的梯度下降训练中，训练损失在 τ₁ = Ω((1/η)·log(1/ε₁)) 步收敛，而 RNC1 在 τ₂ = Ω((1/(λη))·log(1/ε₂)) 步收敛。关键差异：τ₂ 包含 1/λ 因子。
- **物理含义**：当 weight decay λ 很小时，τ₂ >> τ₁，即 Neural Collapse（类内方差收缩）远滞后于训练损失收敛——这恰好是 Grokking 和 IB 压缩延迟出现的原因。
- **Weight decay 的角色**：更强的 weight decay 缩小 τ₂/τ₁ 比值，加速 Neural Collapse 发生，从而加速 Grokking 泛化跳跃和 IB 压缩阶段。

### 理论框架整体逻辑链
Grokking/IB压缩 ← 群体类内方差收缩 ← 经验类内方差收缩(RNC1) ← Neural Collapse 动态 ← Weight decay 控制的时间尺度

## 实验关键数据

### 表1：Grokking 实验——不同 weight decay 下的训练动态（MLP on MNIST）

| Weight Decay λ | 训练精度达100%的步数 τ₁ | 测试精度开始提升的步数 τ₂ | RNC1 下降与泛化同步 |
|---|---|---|---|
| 0.3 | ~5,000 | ~10,000 | ✓（两者几乎同步下降） |
| 0.1 | ~5,000 | ~20,000 | ✓ |
| 0.01 | ~5,000 | ~60,000 | ✓（延迟越大 grokking 越明显） |
| 0.003 | ~5,000 | ~100,000+ | ✓（极端 grokking） |

关键观察：RNC1 下降始终与测试精度提升同步，而非与训练损失收敛同步。λ 越小，两个时间尺度分离越严重。

### 表2：IB 实验——冗余信息与 RNC1 的同步性

| Weight Decay λ | RNC1 开始下降时间 | MI估计的冗余信息下降时间 | nHSIC 冗余信息下降时间 |
|---|---|---|---|
| 0.3 | 早（~10K步） | 同步 | 同步 |
| 0.1 | 中（~20K步） | 基本同步 | 同步 |
| 0.01 | 晚（~60K步） | 略滞后但趋势一致 | 同步 |

两种不同的冗余信息估计方式（MI 和 nHSIC）均与 RNC1 行为定性一致，支持 Theorem 3.4。

- 在 CNN 和 Transformer 架构上重复实验，结论一致（附录 D.1/D.2）。
- 表示空间可视化（Figure 2）：过拟合阶段训练样本可分但测试类内方差大；Neural Collapse 后训练样本坍缩为点、测试方差同步收缩。
- NC2（类均值条件数）与 RNC1 高度同步，趋近于 1，表明完整 Neural Collapse 结构在 Grokking 过程中浮现。

## 亮点
- **统一理论框架**：首次用 Neural Collapse 统一解释 Grokking 和 Information Bottleneck 两个看似不同的后期训练现象。
- **强理论支撑**：完整的定理链（3.2→3.4→4.1→4.3），从现象到机制到时间尺度层层递进，逻辑严密。
- **RNC1 优于 NC1**：提出 rescaled NC1 度量直接对应泛化分析，避免了传统 NC1 被类间方差归一化的混淆。
- **时间尺度刻画精确**：明确给出 weight decay 如何定量影响 Grokking 延迟程度的理论表达。
- **实验-理论高度对应**：每个定理都有对应实验验证，Figure 3/4 中不同 λ 下的曲线行为与定理预测完美吻合。
- **Figure 2 的表示可视化极具说服力**：直观展示过拟合阶段 vs Neural Collapse 阶段表示空间的结构差异。
- **实用指导意义**：结论直接指向实践——tracking RNC1 可判断是否值得继续训练，增大 weight decay 可加速泛化跳跃。
- **Proposition 3.3 补全了 IB 第一阶段**：证明当网络初始状态丢失信息时拟合阶段是必要的，与第二阶段的分析形成完整闭环。
- **多验证手段**：同时使用 MI 估计和 nHSIC 两种独立方法验证 IB 压缩行为，增强结论可靠性。

## 局限与展望
- 理论分析依赖特定假设：金字塔网络架构、光滑激活函数、特殊初始化条件，对 ReLU 等非光滑激活的适用性需进一步验证。
- 实验主要在 MNIST 等相对简单的数据集和 MLP/小型CNN 上验证，大规模视觉模型（ResNet、ViT）上的 Grokking/IB/NC 三者关系有待探索。
- 仅考虑带 weight decay 的梯度下降，对 Adam/AdamW 的理论分析尚未完成（实验用 AdamW 但定理基于 GD）。
- 未讨论 Neural Collapse 在不使用 weight decay 时的隐式出现条件，这可能限制理论的通用性。
- Theorem 3.2 的泛化界基于 Chebyshev 不等式，可能较松；更紧的界或数据依赖的界是改进方向。
- IB 分析要求添加微小高斯噪声使互信息有限，该代理设定与真实确定性网络之间仍有理论间隙。
- 多类分类 (K 较大) 时 Theorem 3.2 的 union bound 较松，可能低估实际泛化能力。
- 仅聚焦分类任务，回归、生成等其他任务中 Neural Collapse 能否同样解释后期行为尚不清楚。
- 未定量比较不同优化器（SGD vs Adam）对 τ₂/τ₁ 比值的实际影响。

## 与相关工作的对比
- **vs. 参数压缩解释 Grokking (Liu et al. 2023a, Varma et al. 2023)**：前人从参数空间压缩角度解释 Grokking，本文从表示空间（类内方差收缩/Neural Collapse）给出新视角，两者可能互补。
- **vs. Kernel→Rich regime 转变 (Lyu et al. 2024)**：该工作从优化景观角度解释 Grokking，本文从几何结构角度切入，更直接。
- **vs. IB 的扩散解释 (Shwartz-Ziv et al. 2019)**：将 IB 压缩阶段归因于 SGD 的扩散成分，本文给出更明确的几何机制（类内方差收缩）。
- **vs. UFM 下的 Neural Collapse 分析**：UFM 将特征当优化变量，脱离实际训练动态；本文直接分析参数梯度下降下的 Neural Collapse 动态。
- **vs. Koch and Ghosh (2025)**：讨论了 Grokking 与几何压缩的关系，但缺乏本文这样严格的理论分析和时间尺度刻画。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次建立 Grokking、IB、Neural Collapse 三者的统一理论联系，原创性极高
- 总体: 理论深度和优雅程度属 ICLR 顶级水平，对 DNN 训练理论研究有重要推动
- 实验充分度: ⭐⭐⭐⭐ — 多架构多 λ 验证充分，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ — 定理-命题-实验层层递进，Figure 1 的逻辑图极清晰
- 价值: ⭐⭐⭐⭐ — 对理解 DNN 训练后期行为有深刻洞察，对实践中 weight decay 调参有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](../../NeurIPS2025/llm_pretraining/flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [Deconstructing Positional Information: From Attention Logits to Training Biases](deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_deep_learning.md)
- [Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)

<!-- RELATED:END -->
