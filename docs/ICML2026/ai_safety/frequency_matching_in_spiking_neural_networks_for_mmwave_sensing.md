---
title: >-
  [论文解读] Frequency Matching in Spiking Neural Networks for mmWave Sensing
description: >-
  [ICML 2026][AI安全][LIF 神经元] 本文从「机制-数据对齐」角度证明 LIF 脉冲神经元等价于一个一阶 IIR 低通滤波器，并提出根据毫米波信号的判别频谱来设定膜衰减系数 $\beta$…
tags:
  - "ICML 2026"
  - "AI安全"
  - "LIF 神经元"
  - "IIR 低通滤波"
  - "毫米波感知"
  - "判别频谱"
  - "神经动力学-数据对齐"
---

# Frequency Matching in Spiking Neural Networks for mmWave Sensing

**会议**: ICML 2026  
**arXiv**: [2605.09983](https://arxiv.org/abs/2605.09983)  
**代码**: [GitHub](https://github.com/yudi-mars/Soul)  
**领域**: 边缘感知 / 脉冲神经网络（SNN） / 无线感知  
**关键词**: LIF 神经元、IIR 低通滤波、毫米波感知、判别频谱、神经动力学-数据对齐

## 一句话总结
本文从「机制-数据对齐」角度证明 LIF 脉冲神经元等价于一个一阶 IIR 低通滤波器，并提出根据毫米波信号的判别频谱来设定膜衰减系数 $\beta$，使 SNN 在四个常用 mmWave 数据集上平均比 ANN 提高 6.22% 精度并降低 3.64× 理论能耗。

## 研究背景与动机
**领域现状**：毫米波雷达因隐私友好、抗光照、穿透性好，是边缘端做姿态、手势、活动识别的重要传感器。主流方案是 CNN / Transformer 等 ANN，靠堆深度和手工预处理拿到鲁棒性，能耗与延迟代价不低。

**现有痛点**：mmWave 信号天生稀疏、不规则、被多径与相位抖动产生的高频噪声严重污染；ANN 没有内置时间滤波偏置，要么先做手工低通预处理（连有用的高频判别信息也被砍掉），要么靠更深的网络硬拟合，能耗与延迟难以承受。

**核心矛盾**：判别信息常分布在「低-中频带」，而噪声集中在高频；现有 ANN/低通预处理都不能区分「有用的高频判别成分」与「真正的高频噪声」。已有 SNN 工作虽然展示了能效优势，但都是经验性 hyperparameter tuning，没人讲清楚「SNN 到底什么时候、为什么比 ANN 强」。

**本文目标**：从信号处理的角度回答两个问题——（1）SNN 在 mmWave 上的优势机理是什么；（2）膜衰减系数 $\beta$ 这一关键超参应当依据数据频谱怎样选。

**切入角度**：把 LIF 神经元的离散动力学线性化为一阶 IIR 低通滤波器，把它的截止频率与数据集判别频谱重合度直接量化，从而把「设 $\beta$」变成一个频域上的对齐问题。

**核心 idea**：让 LIF 的有效带宽 $B_{\text{eff}}(\beta)$ 去匹配 mmWave 数据的判别频谱 $\Omega^\star$——「频率匹配」就是 SNN 在这类任务上比 ANN 强的根本机理，也是 $\beta$ 的物理选择准则。

## 方法详解

### 整体框架
论文不引入新网络结构，而是给「LIF 神经元 + LeNet 风格 SNN」配上一套基于频域的机制分析与超参选择方法。整体分三步：（1）用 DFT 沿时间维分析每个 mmWave 数据集，定义 Fisher 风格的判别指数 $\mathrm{DI}(\omega_k)$ 并归一化为概率分布 $\mathrm{DI}_{\text{norm}}$；（2）把 LIF 写成 $u_{t+1}=\beta u_t+(1-\beta)I_t-v_{\text{th}}O_t$ 并忽略 reset 项，得到等价一阶 IIR 滤波器 $H(\omega_k;\beta)=(1-\beta e^{-j\omega_k})^{-1}$，定义 DC-归一化功率模板 $\tilde H(\omega_k;\beta)$ 与半功率截止 $B_{\text{eff}}(\beta)$；（3）以二者的点积 $\mathrm{FMS}_{\text{avg}}(\beta)=\sum_{\omega_k}\mathrm{DI}_{\text{norm}}(\omega_k)\tilde H(\omega_k;\beta)$ 衡量机制-数据对齐度，再用「最大偏离参考对角线」规则识别 over-low-pass 的临界 $\beta^\dagger$，把 $\beta$ 划成「under-filter / stability window / over-low-pass」三个明确区。

### 关键设计

1. **数据侧：判别频谱 $\mathrm{DI}_{\text{norm}}(\omega_k)$**:

    - 功能：把「类别判别信息在每个频率位上的密度」客观量化，作为后续机制匹配的「数据真相」。
    - 核心思路：对每个样本 $\mathbf{X}_i\in\mathbb{R}^{L\times C\times H\times W}$，先把非时间维平均得到一维时序 $\mathbf{s}_i\in\mathbb{R}^L$，再做 sample-wise 去均值与一边 DFT，得幅度谱 $A_i[k]$；按类别估计每频段的类间散度 $S_B[k]=\sum_c\pi_c(\mu_c[k]-\bar\mu[k])^2$ 与类内散度 $S_W[k]=\sum_c\pi_c\,\mathrm{Var}_c[k]$，定义 $\mathrm{DI}(\omega_k)=S_B[k]/(S_W[k]+\varepsilon)$ 并在频域归一化。
    - 设计动机：直接做线性可分性的 Fisher 风格统计，可以同时反映「信号能量分布」与「类别可分性」，是连接「数据」与「机制」的中介。

2. **机制侧：LIF 低通模板与单调带宽控制（Lemma 3.2）**:

    - 功能：把脉冲神经元的时间整合行为转译成「一个由 $\beta$ 单调控制带宽的低通滤波器」，从而能在频域里直接与数据频谱对齐。
    - 核心思路：忽略 reset 项的 LIF 是一阶 IIR，频响为 $H(\omega_k;\beta)=(1-\beta e^{-j\omega_k})^{-1}$；为消除整体幅度差异，定义 DC-归一化功率模板 $\tilde H(\omega_k;\beta)=(1-\beta)^2/[(1-\beta)^2+2\beta(1-\cos\omega_k)]$。Lemma 3.2 证明：$\tilde H\in(0,1]$、$\tilde H(0;\beta)=1$、对 $\omega_k$ 不增、对 $\beta$ 不增。半功率点 $\tilde H(\omega_c;\beta)=1/2$ 定义有效带宽 $B_{\text{eff}}(\beta)=\omega_c$，于是 $\beta$ 成了一个干净的「逆带宽」旋钮。
    - 设计动机：让超参 $\beta$ 具备明确物理意义（带宽控制），后续的「调参」就变成「带宽与数据频谱对齐」，不再是经验玄学。

3. **对齐侧：FMS 评分与 $\beta^\dagger$ 最大偏离规则**:

    - 功能：给出一个不依赖标签精度的、可纯由数据频谱与神经动力学决定的「过度低通起点」$\beta^\dagger$，从而把 $\beta$ 的可调区段定量划分。
    - 核心思路：把模板与数据频谱内积得到 $\mathrm{FMS}_{\text{avg}}(\beta)=\sum_{\omega_k}\mathrm{DI}_{\text{norm}}(\omega_k)\tilde H(\omega_k;\beta)\in[0,1]$，解释为「LIF 在当前 $\beta$ 下保留下来的判别频谱质量」。再做 $\tau=(1-\beta)^{-1}$，对 $\log\tau$ 与 $\mathrm{FMS}_{\text{avg}}$ 都做 min-max 归一化得到 $(\phi_r,\psi_r)$，连接首末端形成参考对角 $\hat L$，取偏离最大的点 $\beta^\dagger=\arg\max_r|\hat L(\phi_r)-\psi_r|$。由此 Proposition 3.5 给出三段：under-filter（$\beta\to 0$，噪声没压下去）、stability window（$0<\beta<\beta^\dagger$，准确率通常在这里出现峰值）、over-low-pass（$\beta\geq\beta^\dagger$，判别信息也被砍掉）。
    - 设计动机：传统 $\beta$ 调参要做 dataset-specific accuracy sweep，既贵又没机制解释；以频域几何特征定义 $\beta^\dagger$ 把「调参」变成「按频谱画线」，对部署边缘 SNN 极有意义。

### 损失函数 / 训练策略
沿用 surrogate gradient 的标准 SNN 训练（具体细节在附录），简单 LeNet 风格 SpikingLeNet（≈4.19M 参数）即可；唯一额外步骤是按上述方法事先选好每个数据集的 $\beta$。

## 实验关键数据

### 主实验：4 个 mmWave 数据集精度（%, 三种子均值）

| 模型 | AOPHand | mmFiT | Pantomime | MMActivity | #Params (M) |
|------|---------|-------|-----------|------------|-------------|
| LeNet | 60.86 | 62.36 | 61.83 | 59.17 | 4.19 |
| VGG9 | 74.39 | 69.36 | 72.63 | 70.00 | 31.6 |
| ResNet50 | 72.54 | 71.84 | 73.90 | 61.67 | 23.5 |
| GRU | 67.52 | 14.11 | 75.45 | 47.50 | 0.075 |
| CNN-GRU | 61.98 | 67.80 | 72.77 | 65.00 | 0.46 |
| ViT | 21.39 | 36.40 | 42.16 | 65.83 | 2.18 |
| **SpikingLeNet** | **83.70** | **73.67** | **78.31** | **75.00** | 4.19 |

### 主实验：每样本理论能耗（μJ）

| 模型 | AOPHand | mmFiT | Pantomime | MMActivity |
|------|---------|-------|-----------|------------|
| LeNet | 251.08 | 251.08 | 251.10 | 251.08 |
| VGG16 | 6017.25 | 6017.26 | 6017.34 | 6017.24 |
| RNN | 7.35 | 7.35 | 7.36 | 7.35 |
| **SpikingLeNet** | **2.53** | **2.04** | **2.44** | **1.45** |

### 消融与诊断

| 设置 | 关键观察 | 说明 |
|------|---------|------|
| 显式低通预处理 + LeNet vs SpikingLeNet | 加 filter 后 LeNet 改善但仍落后 SpikingLeNet | 硬截断频域只能抑制噪声，会同时砍掉高频判别信息；LIF 提供「软低通」更优 |
| $\beta$ sweep（图 4） | 精度随 $\beta$ 先升后降，峰值 $\beta^\ast<\beta^\dagger$ | 直接验证 Proposition 3.5 的 stability window 预测 |
| $T$ sweep | $T$ 小幅增加 → 精度提升然后饱和 | 温和的时间步主要起稳定预测作用，主要驱动来自 $\beta$ |
| t-SNE（图 3） | SNN 特征类间分离明显优于 ANN | 频率匹配带来的高频噪声抑制让特征空间更具判别性 |
| 多平台延迟 | Jetson GPU 上 ~4× 慢于 LeNet，Darwin3 上几乎追平 | 当前 GPU 把脉冲当稠密 kernel 跑；神经形态硬件才能兑现稀疏优势 |

### 关键发现
- 同样 LeNet 骨架的 SpikingLeNet 在 4 个数据集上平均比最强 ANN 高 6.22%，但参数量与 LeNet 相同——说明性能差异不是来自容量，而是来自 LIF 提供的时间频率偏置。
- 能耗上 SpikingLeNet 比次佳能效模型 RNN 还低 ~3.64×，比 VGG / ResNet 低两到三个数量级；只要硬件支持，这套方法对长开机边缘传感设备非常友好。
- 最佳 $\beta^\ast$ 始终在理论给出的 $\beta^\dagger$ 之前出现，且 $\mathrm{FMS}_{\text{avg}}$ 与精度高度相关，证明「频率匹配」假设在所有四个数据集上都成立。
- 现在 SNN 在 GPU 上看起来「不够快」的延迟瓶颈主要是系统层面的，把工作流放到 Darwin3 这种神经形态芯片上就能兑现「事件驱动 + 稀疏」的硬件优势。

## 亮点与洞察
- 把「为什么 SNN 在 mmWave 上比 ANN 好」从经验观察直接抬到了频域机制层面，并配上可证明的引理与命题，几乎不用谈算法新意就把现有 SNN-mmWave 工作的解释维度补齐。
- 把 $\beta$ 翻译成「逆带宽」并配上 $\beta^\dagger$ 的图解式选取规则，让从业者不再做昂贵的 sweep 就能拿到接近最优的 $\beta$，这种 mechanism-based 调参法可以推广到其他「LIF + 频率结构明显」的任务（脑电、惯性传感、雷达跟踪）。
- 把判别频谱 $\mathrm{DI}_{\text{norm}}$ 引入做「数据频谱画像」是个轻量但通用的工具，可以反过来检视各种网络的「频率偏置」是否与目标数据匹配——这对设计/挑选模型也是一个新视角。

## 局限与展望
- 框架完全建立在「LIF + 忽略 reset」的 IIR 线性化上；对带 hard reset、自适应 threshold 或多状态 spiking 神经元，频域分析需要重做。
- 实验全部在小型 LeNet 上做，未触碰深层 / 多分支 SNN，因此「频率匹配」是否仍是关键瓶颈、还是会被其他层级互动稀释，尚需在更大模型上验证。
- $\beta^\dagger$ 是离散候选集上的几何选择，依赖 sweep 的密度；最优 $\beta^\ast$ 仍需训练后才能确定，论文未给「无需任何训练样本上的最佳 $\beta$」的解析解。
- 论文把延迟问题归因为「系统级 artifact」，但实际部署中常需要给出可量化的硬件-算法 co-design 路径，仅给出 Darwin3 上一个 case 还不够。

## 相关工作与启发
- **vs Fang et al. (2025)**：作者把 Fang 等人提出的「LIF ≈ IIR 低通」结论从公式层面推进到「频谱-数据对齐」框架，第一次给出一个可直接用于调参的判据。
- **vs Arsalan et al. 2022/2023、Hu et al. 2025 等 SNN-mmWave 工作**：以前都强调能效或工程改进；本文以频域机制解释「为什么 SNN 适合 mmWave」，并给出可复用的设计原则。
- **vs 经典低通预处理**：传统硬截断频域会同时砍掉高频判别信息；LIF 的软低通在保留判别成分的同时压制噪声，这是「频率匹配 > 硬截断」的实验级证据。

## 评分
- 新颖性: ⭐⭐⭐⭐ 用频域机制解释 SNN 在 mmWave 上的优势，并给出可计算的 $\beta$ 选择规则，角度新颖。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个常用 mmWave 数据集 + 多平台延迟测试，但限于 LeNet 骨架。
- 写作质量: ⭐⭐⭐⭐ 引理与命题清晰，机制叙事完整。
- 价值: ⭐⭐⭐⭐ 对边缘端 SNN 部署具有直接调参指导意义，也为「机制-数据对齐」研究范式提供模板。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](../../ICLR2026/ai_safety/time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](../../AAAI2026/ai_safety/mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
