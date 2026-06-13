---
title: >-
  [论文解读] Guided Diffusion Sampling on Function Spaces with Applications to PDEs
description: >-
  [NeurIPS 2025][物理/科学计算][函数空间扩散模型] 提出 **FunDPS（Function-space Diffusion Posterior Sampling）**，在函数空间中训练无条件扩散模型，推理时通过梯度引导实现 plug-and-play 的 PDE 逆问题后验采样；理论上将 Tweedie 公式推广到无穷维 Banach 空间，实验上在 5 个 PDE 任务中仅用 3% 观测即可获得比 DiffusionPDE 平均高 32% 的精度并减少 4 倍采样步数。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "函数空间扩散模型"
  - "偏微分方程"
  - "后验采样"
  - "Tweedie公式"
  - "神经算子"
  - "分辨率无关"
---

# Guided Diffusion Sampling on Function Spaces with Applications to PDEs

**会议**: NeurIPS 2025  
**arXiv**: [2505.17004](https://arxiv.org/abs/2505.17004)  
**代码**: [neuraloperator/FunDPS](https://github.com/neuraloperator/FunDPS)  
**领域**: 图像生成  
**关键词**: 函数空间扩散模型, PDE逆问题, 后验采样, Tweedie公式, 神经算子, 分辨率无关

## 一句话总结

提出 **FunDPS（Function-space Diffusion Posterior Sampling）**，在函数空间中训练无条件扩散模型，推理时通过梯度引导实现 plug-and-play 的 PDE 逆问题后验采样；理论上将 Tweedie 公式推广到无穷维 Banach 空间，实验上在 5 个 PDE 任务中仅用 3% 观测即可获得比 DiffusionPDE 平均高 32% 的精度并减少 4 倍采样步数。

## 研究背景与动机

### 问题场景

科学计算中大量任务可归结为 **条件采样 / 逆问题**：从稀疏或带噪的测量中恢复完整的物理场（温度场、流场、渗透率场等）。典型例子包括：

- 地下流体：从少量传感器恢复达西流的渗透率场
- 气候预测：从有限站点观测推断全球大气状态
- 弹性力学：由形变场反推材料属性

### 现有方法的不足

**MCMC 方法**：理论完备但高维问题收敛极慢，构造 proposal distribution 困难

**确定性神经 PDE 求解器**（FNO、DeepONet、PINN）：只输出单点估计，无法给出后验分布，且在极稀疏观测下误差极大

**有限维扩散模型**（DiffusionPDE）：在固定分辨率的像素空间上建模，换分辨率需重新训练；采样步数多（2000 步），推理慢

**函数空间扩散模型**（DDO 等）：已有的函数空间扩散模型仅支持无条件生成，或需要针对每种观测配置训练专用的条件 score 模型，灵活性差

### 核心洞察

物理系统本质上由连续函数描述，因此应在 **函数空间** 中建模先验分布。训练一个无条件函数空间扩散模型，推理时以 plug-and-play 方式注入观测约束，即可用同一模型处理各种下游逆问题——无需针对不同传感器配置重新训练。

## 核心问题

给定 PDE 系统中的极稀疏测量 $\boldsymbol{u} = \boldsymbol{A}(\boldsymbol{a}) + \varepsilon$（仅 3% 的观测点），如何在函数空间中从后验分布 $\nu(\boldsymbol{a}|\boldsymbol{u})$ 中采样，实现分辨率无关的、高精度的前向与逆问题求解？

## 方法详解

### 整体框架

FunDPS 分为两个阶段：

1. **训练阶段**：在函数空间中训练无条件扩散模型，学习 PDE 参数与解的联合先验分布
2. **推理阶段**：通过贝叶斯分解将后验 score 拆为先验 score + 似然梯度，用训练好的去噪器近似先验，用 Tweedie 公式近似似然，实现 plug-and-play 引导采样

### 贝叶斯框架

在函数空间设定下，采用贝叶斯视角：

- **先验** $\nu(\boldsymbol{a})$：由扩散模型从数据中学习
- **似然** $p(\boldsymbol{u}|\boldsymbol{a})$：由前向算子 $\boldsymbol{A}$ 和高斯测量噪声 $\eta = \mathcal{N}(0, \mathbf{C}_\eta)$ 确定
- **后验** $\nu^{\boldsymbol{u}}(\boldsymbol{a}) \propto \nu(\boldsymbol{a}) \exp(\Phi(\boldsymbol{a}, \boldsymbol{u}))$：通过 Cameron-Martin 定理得到似然的 Radon-Nikodym 导数

关键的条件 score 分解（类比有限维的 DPS）：

$$\nabla_{\boldsymbol{a}_t} \log \frac{d\nu_t^{\boldsymbol{u}}}{d\gamma_t}(\boldsymbol{a}_t) = \underbrace{\nabla_{\boldsymbol{a}_t} \log \frac{d\nu_t}{d\gamma_t}(\boldsymbol{a}_t)}_{\text{先验 score（已训练）}} + \underbrace{\nabla_{\boldsymbol{a}_t} \tilde{\Phi}_t(\boldsymbol{a}_t, \boldsymbol{u})}_{\text{似然梯度（需近似）}}$$

### 无穷维 Tweedie 公式（核心理论贡献）

Tweedie 公式在有限维中给出条件期望 $\mathbb{E}[\boldsymbol{a}_0|\boldsymbol{a}_t]$ 与 score 函数的闭合关系，是 DPS、MCG 等引导采样方法的理论基石。但此前仅在有限维空间中被证明。

**定理 3.1**：设 $B$ 是可分 Banach 空间，在合适的正则性条件下，对 $\nu$-几乎处处的 $y$：

$$\mathbb{E}[X|Y=y] = R\left(D_{H(\gamma)} \log \frac{d\nu}{d\gamma}(y)\right)$$

其中 $R$ 是 Riesz 表示映射，$D_{H(\gamma)}$ 是沿 Cameron-Martin 空间的 Fréchet 导数。

这一推广使得可以用训练好的去噪器 $\boldsymbol{D}_\theta(\boldsymbol{a}_t, t) \approx \mathbb{E}[\boldsymbol{a}_0|\boldsymbol{a}_t]$ 来近似似然：

$$\nabla_{\boldsymbol{a}_t} \tilde{\Phi}_t(\boldsymbol{a}_t, \boldsymbol{u}) \approx -\frac{c}{2} \nabla_{\boldsymbol{a}_t} \|\boldsymbol{u} - \boldsymbol{A}(\hat{\boldsymbol{a}}_0(\boldsymbol{a}_t))\|_{\mathcal{U}}^2$$

### 引导更新规则

在逆扩散过程的每一步，按以下规则更新样本：

$$\boldsymbol{a}_{i+1} \leftarrow \boldsymbol{a}_i - \boldsymbol{\zeta} \cdot \nabla_{\boldsymbol{a}_i} \|\boldsymbol{u} - \boldsymbol{A}(\boldsymbol{D}_\theta(\boldsymbol{a}_i, t_i))\|_{\mathcal{U}}^2$$

其中 $\boldsymbol{\zeta}$ 是预定义的引导权重向量，综合吸收了噪声协方差的缩放因子与各观测通道的置信度。

### 联合嵌入

将 PDE 的参数函数（系数 $c$、边界条件 $g$）和解函数 $f$ **联合表示**为一个多通道函数 $\boldsymbol{a}$。通过对不同通道施加掩码：

- **前向问题**：完全掩码解通道，保留（稀疏）参数通道
- **逆问题**：完全掩码参数通道，保留（稀疏）解通道
- **混合问题**：同时部分掩码

### 多分辨率训练

利用神经算子（U-shaped Neural Operator）的离散化不变性，采用课程学习策略：

1. 大部分 epoch 在低分辨率数据上训练，学习粗粒度结构
2. 最后少量 epoch 在高分辨率数据上微调，补充高频细节
3. 总训练 GPU 时间减少 **25%**，精度与全程高分辨率训练持平

### 多分辨率推理（ReNoise）

提出 **ReNoise** 双层多分辨率推理策略：

1. 前 80% 的去噪步在低分辨率执行
2. 上采样到目标分辨率
3. 注入额外噪声以消除上采样伪影和噪声级别不匹配
4. 最后 20% 的步在目标分辨率微调高频细节

这一策略额外带来 **2 倍** 速度提升。

### 网络架构

- 去噪器 $\boldsymbol{D}_\theta$：U-shaped Neural Operator（基于 EDM-FS），约 54M 参数
- 噪声采样器：高斯随机场 (GRF) 而非多元高斯，保证函数空间一致性
- 采样器：二阶确定性采样器

## 实验关键数据

### 实验设置

- **5 个 PDE 任务**：Darcy Flow、Poisson、Helmholtz、Navier-Stokes（周期边界）、Navier-Stokes（Dirichlet 边界）
- **观测密度**：仅 3% 的空间点可观测——极稀疏设定
- **基线**：FNO、PINO、DeepONet、PINN、DiffusionPDE
- **分辨率**：128×128

### 主实验结果（$L^2$ 相对误差，%，5 个 PDE × 前向/逆向 = 10 个子任务）

| 方法 | 步数 | Darcy 前 | Darcy 逆 | Poisson 前 | Poisson 逆 | Helmholtz 前 | Helmholtz 逆 | NS 前 | NS 逆 | NS-BC 前 | NS-BC 逆 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **FunDPS** | 200 | 2.88 | 6.78 | 2.04 | 24.04 | 2.20 | 20.07 | 3.99 | 9.87 | 5.91 | 4.31 |
| **FunDPS** | 500 | **2.49** | **5.18** | **1.99** | **20.47** | **2.13** | **17.16** | **3.32** | **8.48** | **4.90** | **4.08** |
| DiffusionPDE | 2000 | 6.07 | 7.87 | 4.88 | 21.10 | 12.64 | 19.07 | 3.78 | 9.63 | 9.69 | 4.18 |
| FNO | - | 28.2 | 49.3 | 100.9 | 232.7 | 98.2 | 218.2 | 101.4 | 96.0 | 82.8 | 69.6 |
| PINN | - | 48.8 | 59.7 | 128.1 | 130.0 | 142.3 | 160.0 | 142.7 | 146.8 | 100.1 | 105.5 |

关键发现：

- FunDPS (500步) 在所有任务上均取得最优结果，平均误差较 DiffusionPDE **降低 32%**
- FunDPS 仅需 200 步即可超越 DiffusionPDE 的 2000 步表现
- 确定性基线（FNO/PINN 等）在 3% 极稀疏设定下完全失败（误差 30%–200%+）

### 推理速度对比

| 方法 | 步数 | 单样本耗时 | 硬件 |
|---|---|---|---|
| FunDPS | 500 步 | **15 s** | RTX 4090 |
| FunDPS + ReNoise | 500 步 | **7.5 s** | RTX 4090 |
| DiffusionPDE | 2000 步 | 190 s | RTX 4090 |

FunDPS 推理速度是 DiffusionPDE 的 **25 倍**（结合 ReNoise），同时精度更高。

### 多分辨率训练效果

多分辨率课程训练仅需全分辨率训练 25% 的 GPU 时间，精度几乎无损。

### 多分辨率推理效果

ReNoise 在 80% 的步数使用低分辨率时，精度与全分辨率推理基本持平，实现 2 倍额外加速。

## 亮点

1. **理论贡献突出**：首次将 Tweedie 公式严格推广到无穷维 Banach 空间，为函数空间后验采样提供数学基础
2. **Plug-and-play 灵活性**：训练一次无条件模型，推理时按需组合任意观测算子，无需重新训练
3. **分辨率无关**：基于神经算子，模型天然支持跨分辨率推理，同一模型可应用于不同网格密度
4. **速度飞跃**：较 DiffusionPDE 在精度提升 32% 的同时实现 25 倍推理加速
5. **多分辨率训练/推理**：课程学习训练节省 75% GPU 时间，ReNoise 推理再提速 2 倍
6. **通用框架**：联合嵌入设计使同一模型可统一处理前向、逆向和混合问题

## 局限与展望

1. **PDE 损失引导效果有限**：直接在引导中加入有限差分近似的 PDE 残差仅带来微弱提升，可能因离散化误差累积
2. **引导权重需手动调参**：不同 PDE 任务需要独立调节 $\boldsymbol{\zeta}$，缺乏自适应权重方案
3. **高噪声水平近似不够准确**：似然近似 $\tilde{\Phi}_t \approx \Phi(\hat{\boldsymbol{a}}_0, \boldsymbol{u})$ 在扩散前期（噪声大）精度下降
4. **尚未验证专用逆问题**：MRI 重建、全波形反演等专业领域表现未知
5. **时间维度有待扩展**：目前聚焦空间稀疏观测，时间序列 PDE 的时空联合推理是自然延伸方向
6. **基础模型潜力**：可探索在多种 PDE 和多物理域上预训练统一的函数空间扩散基座模型

## 与相关工作的对比

| 方法 | 空间 | 条件方式 | 分辨率无关 | 后验采样 | 训练成本 |
|---|---|---|---|---|---|
| **FunDPS (本文)** | 函数空间 | Plug-and-play 引导 | ✅ | ✅ | 中（多分辨率课程减 75%） |
| DiffusionPDE | 像素空间 (固定分辨率) | Plug-and-play 引导 | ❌ | ✅ | 高 |
| DDO (Lim et al.) | 函数空间 | 无（仅无条件生成） | ✅ | ❌ | 中 |
| Baldassari et al. | 函数空间 | 条件 score 模型 | ✅ | ✅ | 高（每种观测需重训） |
| FNO / DeepONet | 函数空间 | 端到端映射 | ✅ | ❌（确定性） | 低 |
| PINN | 离散 | PDE 残差约束 | ❌ | ❌（确定性） | 中 |
| Kerrigan et al. (FFM) | 函数空间 | 条件 flow matching | ✅ | ✅ | 高（需任务专用训练） |

**与 DiffusionPDE 的关键差异**：DiffusionPDE 在 $128 \times 128$ 像素空间建模，换分辨率需完全重训；FunDPS 基于神经算子在函数空间建模，天然支持跨分辨率推理。此外 FunDPS 使用 GRF 噪声（而非多元高斯）保证函数空间一致性，使引导更平滑，步数从 2000 降至 200–500。

**与 DDO 的关键差异**：DDO 仅支持无条件生成，FunDPS 通过将无穷维 Tweedie 公式与 Bayesian 似然分解结合，实现了推理时的 plug-and-play 条件引导，同一模型可处理前向/逆向/混合问题。

**与确定性求解器的差异**：FNO/PINN 等在 3% 极稀疏观测下完全失效（误差 30%–200%+），因为它们输出确定性点估计，无法利用先验正则化；FunDPS 通过后验采样自然处理不适定性。

## 启发与关联

1. **函数空间 ≠ 像素空间的本质区别**：本文核心 insight 是物理系统由连续函数描述，用离散像素建模是 level mismatch。GRF 噪声 + 神经算子组合使得扩散过程在离散化变化时行为一致，这一思路可推广到任何连续场的生成任务（天气场、电磁场、应力场等）。

2. **Plug-and-play 范式在科学计算中的价值**：与 CV 领域的 DPS/MCG 类似，训练一次先验模型即可应对不同观测配置。对于传感器布局频繁变化的工业场景（如油气勘探、环境监测），这意味着巨大的部署灵活性。

3. **多分辨率课程学习的通用性**：先粗后细的训练策略减少 75% GPU 时间，这一技巧可推广到其他基于算子学习的生成模型（如 functional flow matching）。

4. **Tweedie 公式的无穷维推广**：理论贡献独立于 FunDPS 框架，可用于其他需要在函数空间中估计条件期望的场景，如函数空间中的 classifier-free guidance 或 RLHF。

5. **与 Foundation Model 的交汇**：作者在 Outlook 中提出跨 PDE 类型的函数空间扩散基座模型，这与 Foundation Model for Science 的愿景高度契合。若配合大规模多物理场数据预训练，有望成为科学计算的通用后验采样引擎。

6. **局限性的研究机会**：引导权重 $\boldsymbol{\zeta}$ 的自主调节、高噪声时 Tweedie 近似的改进、以及时空联合推理都是有价值的后续方向。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 将 Tweedie 公式推广到无穷维 Banach 空间是扎实的理论贡献；函数空间 plug-and-play 后验采样框架在 PDE 逆问题领域属首创
- 实验充分度: ⭐⭐⭐⭐ — 5 个 PDE × 前向/逆向 = 10 个子任务，覆盖线性/非线性、不同边界条件；消融研究完整；但仅在 128×128 尺度验证，未测试真实物理场景
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严谨，符号一致，图表清晰；理论-方法-实验逻辑链完整
- 价值: ⭐⭐⭐⭐½ — 为科学计算中的逆问题提供了理论完备且实用高效的后验采样框架，有望成为 PDE 逆问题的标准工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)

</div>

<!-- RELATED:END -->
