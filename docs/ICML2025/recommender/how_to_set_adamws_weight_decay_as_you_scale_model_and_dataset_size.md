---
title: >-
  [论文解读] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size
description: >-
  [ICML 2025][AdamW] 将 AdamW 的权重更新解释为指数移动平均（EMA），揭示了 EMA 时间尺度 $\tau = 1/(\eta\lambda)$ 是核心超参数，其以 epoch 为单位的最优值在模型和数据集规模变化时保持稳定，从而给出了 weight decay 随规模缩放的明确规则。
tags:
  - ICML 2025
  - AdamW
  - weight decay
  - 超参数迁移
  - µP
  - 指数移动平均
---

# How to Set AdamW's Weight Decay as You Scale Model and Dataset Size

**会议**: ICML 2025  
**arXiv**: [2405.13698](https://arxiv.org/abs/2405.13698)  
**代码**: 无  
**领域**: 推荐系统  
**关键词**: AdamW, weight decay, 超参数迁移, µP, 指数移动平均

## 一句话总结

将 AdamW 的权重更新解释为指数移动平均（EMA），揭示了 EMA 时间尺度 $\tau = 1/(\eta\lambda)$ 是核心超参数，其以 epoch 为单位的最优值在模型和数据集规模变化时保持稳定，从而给出了 weight decay 随规模缩放的明确规则。

## 研究背景与动机

在大规模模型训练的常见工作流中，研究者通常先在小规模上原型验证，再将最优超参数迁移到大规模训练。对于学习率，µP（Yang et al., 2022）等工作已经给出了清晰的宽度缩放规则（$\eta \propto 1/\text{fan\_in}$）。然而，当使用 AdamW 优化器时，**weight decay 超参数 $\lambda$ 如何随模型大小和数据集大小缩放**，这个问题一直缺乏系统性的理论理解和实验验证。

现有实践中 weight decay 通常被设为常数（如 0.1），但实际上：

- 当数据集变大时，固定 $\lambda$ 会导致性能下降
- 当模型变宽时，µP 的学习率缩放会因固定 $\lambda$ 而失效
- 缺乏一个统一的理论框架来指导 $\lambda$ 的缩放

本文的核心动机就是填补这一空白，给出理论上有依据、实验上经过验证的 weight decay 缩放规则。

## 方法详解

### 整体框架

本文的核心思想是：**AdamW 中的权重本身可以被理解为对近期更新的指数移动平均（EMA）**。注意，这里说的不是 Adam 中用于计算一阶矩 $\hat{m}_t$ 和二阶矩 $\hat{v}_t$ 的 EMA，而是整个权重更新过程本身构成一个 EMA。

AdamW 的更新公式为：

$$w_t = (1 - \eta_t \lambda) w_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

将其与标准 EMA 公式 $\text{ema}_t = (1 - 1/\tau) \text{ema}_{t-1} + (1/\tau) q_t$ 对比，可以建立映射：

| EMA 变量 | AdamW 对应量 | 含义 |
|:---|:---|:---|
| $1/\tau_{\text{iter}}$ | $\eta_t \lambda$ | EMA 遗忘率 |
| $\text{ema}_t$ | $w_t$ | 当前权重 |
| $q_t$ | $-\frac{1}{\lambda}\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | 缩放后的更新量 |

因此，EMA 时间尺度为 $\tau_{\text{iter}} = 1/(\eta\lambda)$，直观地代表了 EMA 对最近多少次迭代的更新进行平均。

### 关键设计

#### 1. 从迭代时间尺度到 epoch 时间尺度

为了衡量 EMA 平均了数据集的多少比例，作者引入 epoch 时间尺度：

$$\tau_{\text{epoch}} = \tau_{\text{iter}} / M = \frac{1}{\eta \lambda N / B}$$

其中 $N$ 是数据集大小，$B$ 是 batch size，$M = N/B$ 是每个 epoch 的迭代次数。$\tau_{\text{epoch}}$ 衡量了 AdamW 的 EMA 对过去多少个 epoch 的更新做平均。

**核心发现**：最优 $\tau_{\text{epoch}}$ 在模型和数据集规模变化时近似不变。

#### 2. Weight decay 随数据集大小的缩放

在固定学习率下，由 $\lambda = 1/(\eta M \tau_{\text{epoch}})$ 可知：

- 当数据集增大时，$M$ 增大
- 若最优 $\tau_{\text{epoch}}$ 不变，则最优 $\lambda$ 应减小

**缩放规则**：$\lambda \propto 1/N$（数据集越大，weight decay 越小）

#### 3. Weight decay 随模型宽度的缩放

µP 要求学习率随模型宽度缩放：$\eta = \eta_{\text{base}} / s$，其中 $s = \text{fan\_in} / \text{fan\_in}_{\text{base}}$。

如果保持 $\lambda$ 不变（标准 µP 做法），则：

$$\tau_{\text{iter}} = s / (\eta_{\text{base}} \lambda_{\text{base}}) = s \cdot \tau_{\text{iter;base}}$$

时间尺度会随模型变大而增大，**导致 µP 的学习率迁移失效**。

解决方案是让 weight decay 也随宽度缩放：

$$\eta = \eta_{\text{base}} / s, \quad \lambda = s \cdot \lambda_{\text{base}}$$

这样 $\tau_{\text{iter}} = 1/(\eta\lambda) = 1/(\eta_{\text{base}} \lambda_{\text{base}}) = \tau_{\text{iter;base}}$，时间尺度保持不变。

**缩放规则**：$\lambda \propto \text{fan\_in}$（模型越宽，weight decay 越大）

#### 4. 尺度不变网络上的理论保证（Theorem 1）

对于尺度不变网络（即 $\text{net}(x; w) = \text{net}(x; w/c)$），在相同的 EMA 时间尺度和相同的初始学习率/初始化比值下，两个不同 $(\eta, \lambda, \sigma, \epsilon)$ 配置的 AdamW 会产生完全相同的网络输出轨迹。这理论上证明了 $\tau_{\text{iter}}$ 是 AdamW 在尺度不变网络上的关键控制参数。

### 损失函数 / 训练策略

本文不提出新的损失函数，而是提出超参数缩放策略。总结如下：

| 缩放维度 | 学习率 $\eta$ | Weight Decay $\lambda$ | EMA 时间尺度 $\tau$ |
|:---|:---|:---|:---|
| 数据集增大 ($N \uparrow$) | 固定 | $\lambda \propto 1/N$ | $\tau_{\text{epoch}}$ 不变 |
| 模型变宽 (µP) | $\eta \propto 1/s$ | $\lambda \propto s$ | $\tau_{\text{iter}}$ 不变 |
| Batch size 增大 | $\eta \propto B$ 或固定 | $\lambda \propto B$（若 $\eta$ 固定） | $\tau_{\text{epoch}}$ 不变 |

训练时采用 cosine decay 学习率调度（衰减到初始值的 0.1 或 0），不对 normalization 层使用 weight decay。

## 实验关键数据

### 主实验

实验覆盖三个架构和三个数据集：

| 模型 | 数据集 | 验证内容 | 结论 |
|:---|:---|:---|:---|
| ResNet-18 | ImageNet 32×32 子集 (80K-1.28M) | $\tau_{\text{epoch}}$ 随数据集大小的稳定性 | 最优 $\tau_{\text{epoch}}$ 稳定，最优 $\lambda$ 剧烈变化 |
| ViT | ImageNet 32×32 子集 | 同上 | 同样验证了 $\tau_{\text{epoch}}$ 的稳定性 |
| NanoGPT 124M | OpenWebText (1/4, 1/2, 全集) | 语言模型上的数据集缩放 | 最优 $\lambda$ 随数据集增大而减小，$\tau_{\text{epoch}}$ 稳定 |
| ResNet-18 (变宽) | ImageNet 32×32 (320K) | $\lambda$ 随模型宽度缩放 | $\lambda \propto s$ 使得最优 $\lambda_{\text{base}}$ 一致 |
| 8层 GPT (256-1024宽) | OpenWebText | LLM 宽度缩放 | 同上，确认 $\lambda \propto s$ 的有效性 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|:---|:---|:---|
| 固定 $\lambda$ + µP (标准做法) | 最优 $\eta_{\text{base}}$ 随宽度剧变 | µP 学习率迁移失效 |
| $\lambda \propto s$ + µP (本文方法) | 最优 $\eta_{\text{base}}$ 跨宽度一致 | 修复了 µP 的迁移能力 |
| Constant LR schedule | $\tau_{\text{epoch}}$ 稳定 | 不同调度策略下结论一致 |
| Cosine decay to 0 | $\tau_{\text{epoch}}$ 稳定 | 同上 |
| $\lambda \propto 1/N$ + 固定 $\eta$ | 最优 $\eta$ 跨数据集大小一致 | 正确缩放 $\lambda$ 可避免调 $\eta$ |

### 关键发现

1. **最优 $\tau_{\text{epoch}}$ 跨数据集和模型大小稳定**：这是最核心的实验发现，在 ResNet、ViT、GPT 上均成立
2. **µP + 固定 $\lambda$ 会失效**：在 CIFAR-10 和 ImageNet 上，标准 µP 做法（固定 $\lambda$）导致最优学习率随宽度大幅变化
3. **$\lambda \propto s$ 修复 µP**：增大 weight decay 与模型宽度成正比后，最优学习率恢复跨宽度一致性
4. **后续大规模验证**：Blake et al. (2024) 和 Dey et al. (2025) 在十亿参数级 LLM 预训练中确认了 $\tau_{\text{epoch}}$ 的可迁移性

## 亮点与洞察

1. **优雅的理论视角**：将 AdamW 理解为 EMA 的视角非常自然且有力，一个简单的重参数化就揭示了隐藏的结构
2. **实用价值极高**：给出了明确的、可操作的 weight decay 缩放规则，直接适用于大规模训练
3. **解释了已有现象**：统一解释了 Loshchilov & Hutter 观察到的 $\eta$ 与 $\gamma=\eta\lambda$ 解耦现象，以及 Lingle (2024) 发现的 µP 在 AdamW 上失效的问题
4. **Theorem 1 的理论深度**：在尺度不变网络上严格证明了 $\tau_{\text{iter}}$ 是唯一的控制参数

## 局限与展望

1. **实验规模有限**：主要在小规模数据集（ImageNet 32×32、CIFAR-10）和中等模型（124M NanoGPT）上验证，虽然后续工作已在更大规模上确认
2. **仅考虑多 epoch 训练**：LLM 预训练通常只训 1 epoch，此时 $\tau_{\text{epoch}}$ 的最优值不再恒定，而是遵循幂律（Bergsma et al., 2025a）
3. **EMA 近似的局限**：标准 EMA 假设 $q_t$ 独立，但 AdamW 中 $q_t$ 依赖于 $w_t$，因此只是近似
4. **深度缩放未探讨**：只研究了宽度缩放，未涉及深度（层数）变化时 $\lambda$ 的缩放
5. **$\lambda$ 与 $s$ 的精确关系可能是超线性的**（$\lambda \propto s^\alpha, \alpha > 1$），需要更大规模实验确认

## 相关工作与启发

- **µP (Yang et al., 2022)**：本文是 µP 的重要补充，解决了 µP 未考虑 weight decay 的失效问题
- **Loshchilov & Hutter (2018)**：原始 AdamW 论文观察到 $(\eta, \gamma)$ 比 $(\eta, \lambda)$ 更解耦，本文给出了理论解释
- **Wortsman et al. (2024)**：发现固定 $\gamma=\eta\lambda$ 时验证损失对 $\eta$ 不敏感，支持了 $\tau_{\text{iter}}$ 作为核心超参数的观点
- **D'Angelo et al. (2024)**：认为 weight decay 的作用不是正则化而是控制 minibatch 噪声，与 EMA 视角互补
- **Bergsma et al. (2025a)**：将本文的框架扩展到 batch size 缩放和单 epoch 场景，发现 $\tau_{\text{epoch}}^{\text{optimal}} \approx (\text{TPP})^{-0.527}$

**启发**：这篇论文提示我们在设计大规模训练的超参数搜索空间时，应该搜索 $\tau_{\text{epoch}}$（或等价地 $\eta\lambda$）而非单独搜索 $\lambda$，这样可以大幅缩小搜索空间并提高跨规模迁移的成功率。

## 评分

| 维度 | 评分 | 说明 |
|:---|:---|:---|
| 新颖性 | ⭐⭐⭐⭐ | EMA 视角虽简单但极具洞察力 |
| 理论深度 | ⭐⭐⭐⭐ | Theorem 1 严格优雅，但主要依赖实验 |
| 实验充分性 | ⭐⭐⭐⭐ | 多模型多数据集验证，但规模偏小 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 直接指导大规模训练，已被后续工作广泛验证 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 逻辑清晰，直觉解释到位 |
| **综合** | **⭐⭐⭐⭐☆** | 简洁有力的工作，实用性极强 |
