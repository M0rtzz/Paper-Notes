---
title: >-
  [论文解读] Improved Noise Schedule for Diffusion Training
description: >-
  [ICCV 2025][图像生成][噪声调度] 提出从概率分布视角统一分析和设计扩散模型噪声调度的框架，发现将采样概率集中在 $\log\text{SNR}=0$ 附近（信号与噪声临界点）的 Laplace 噪声调度，在相同训练预算下比标准 cosine 调度 FID 提升 26.6%…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "噪声调度"
  - "扩散模型训练"
  - "重要性采样"
  - "Laplace分布"
  - "logSNR"
---

# Improved Noise Schedule for Diffusion Training

**会议**: ICCV 2025  
**arXiv**: [2407.03297](https://arxiv.org/abs/2407.03297)  
**代码**: 无  
**领域**: 扩散模型/图像生成  
**关键词**: 噪声调度, 扩散模型训练, 重要性采样, Laplace分布, logSNR

## 一句话总结

提出从概率分布视角统一分析和设计扩散模型噪声调度的框架，发现将采样概率集中在 $\log\text{SNR}=0$ 附近（信号与噪声临界点）的 Laplace 噪声调度，在相同训练预算下比标准 cosine 调度 FID 提升 26.6%，且优于所有损失权重调整方法。

## 研究背景与动机

### 问题定义

扩散模型在训练过程中需要在不同噪声水平上学习去噪，但"在哪些噪声水平上投入更多计算资源"这一根本问题长期被忽视。训练时通常均匀采样时间步 $t \sim \mathcal{U}[0,1]$，但这隐含了一个非均匀的噪声强度分布。

### 已有方法的不足

**架构改进方向**：DiT 的 AdaLN、MM-DiT 的多模态分离权重、U-shaped skip connections 等虽然有效，但没有触及噪声调度本身

**损失权重调整方向**：Min-SNR、Soft-Min-SNR、P2 等通过调整不同噪声水平的损失权重来加速收敛，但本质上等价于调整噪声强度的重要性采样——在实际训练中，直接修改采样分布 $p(\lambda)$ 比放大损失权重 $w(\lambda)$ 更有效

**噪声调度设计零散**：linear、cosine、EDM 等调度的设计缺乏统一理论框架，选择往往是临时性的

### 核心动机

**关键洞察**：在扩散模型的统一训练公式中，调整损失权重 $w(\lambda)$ 和修改噪声采样分布 $p(\lambda)$ 理论上等价，但在有限计算预算下，直接将计算资源（FLOPs）集中在中等噪声水平（$\log\text{SNR} \approx 0$）比增大该区域的损失权重更高效。这意味着噪声调度设计本质上是一个**概率分布设计问题**。

## 方法详解

### 整体框架

将噪声调度重新表述为 $\log\text{SNR}$ 上的概率分布 $p(\lambda)$，通过选择不同的概率分布族来生成新的噪声调度，使模型聚焦于信号与噪声的临界转换点。

### 关键设计

#### 1. **概率视角的噪声调度统一框架**

- **功能**：建立噪声调度 $\lambda(t)$ 与噪声强度采样分布 $p(\lambda)$ 之间的双向转换关系
- **核心思路**：当 $t$ 服从均匀分布时，噪声强度 $\lambda = \log\text{SNR}$ 的采样概率为：
  $$p(\lambda) = -\frac{dt}{d\lambda}$$
  反过来，从任意概率分布 $p(\lambda)$ 可以导出噪声调度：
  $$t = 1 - \int_{-\infty}^{\lambda} p(\lambda) d\lambda = \mathcal{P}(\lambda), \quad \lambda = \mathcal{P}^{-1}(t)$$
  其中 $\mathcal{P}(\lambda)$ 是 $\lambda$ 的累积分布函数。
- **设计动机**：这一框架揭示了噪声调度设计可以被重新定义为概率分布设计问题——不再直接指定噪声如何随时间变化，而是优化如何在不同噪声强度上分配采样资源。

#### 2. **Laplace 噪声调度**

- **功能**：提出以 Laplace 分布为 $p(\lambda)$ 的新噪声调度
- **核心思路**：Laplace 分布的概率密度函数为：
  $$p(\lambda) = \frac{1}{2b} \exp\left(-\frac{|\lambda - \mu|}{b}\right)$$
  对应的噪声调度为：
  $$\lambda = \mu - b \cdot \text{sgn}(0.5 - t) \cdot \ln(1 - 2|t - 0.5|)$$
  默认 $\mu=0$（集中在 $\log\text{SNR}=0$），$b=0.5$（较尖锐的峰值，确保更多采样集中在中间噪声水平）。
- **设计动机**：Laplace 分布具有简洁的指数衰减特性和对称性，其在 $\lambda=0$ 处的尖峰恰好对应信号与噪声的平衡点。实验表明这个临界点是扩散模型训练中最关键的区域。

#### 3. **统一训练公式与实际设置**

- **功能**：基于 VDM++ 的统一训练框架，分析为何修改 $p(\lambda)$ 优于修改 $w(\lambda)$
- **核心思路**：统一损失函数为：
  $$\mathcal{L}_w(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x},\boldsymbol{\epsilon},\lambda\sim p(\lambda)} \left[\frac{w(\lambda)}{p(\lambda)} \|\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_\lambda;\lambda) - \boldsymbol{\epsilon}\|_2^2\right]$$
  虽然调整 $w(\lambda)$ 和 $p(\lambda)$ 理论等价，但修改 $p(\lambda)$ 意味着将更多的前向和反向传播计算直接投入到关键噪声水平，而调整 $w(\lambda)$ 仅改变梯度的大小而不增加该区域的计算量。
- **设计动机**：在有限计算预算下，"把计算花在哪里"比"给哪里的梯度加权"更重要。这解释了为什么 Min-SNR 等损失调整方法虽有效但不如直接修改噪声调度。

### 损失函数 / 训练策略

- 标准 MSE 损失，支持 $\epsilon$、$\mathbf{x}_0$、$\mathbf{v}$ 三种预测目标
- 推理时使用 DDIM 50步，采样 SNR 对齐到 cosine schedule 以确保公平比较
- 对于 Laplace 调度，推理时从 $t_{\max}=0.99$ 开始采样（因为极端噪声水平训练不足）
- 500K 迭代，batch size 256，8×V100 GPU

## 实验关键数据

### 主实验

**ImageNet-256 不同噪声调度与损失权重比较（FID-10K）**：

| 方法 | 类型 | CFG=1.5 | CFG=2.0 | CFG=3.0 |
|------|------|---------|---------|---------|
| Cosine | 基线调度 | 17.79 | 10.85 | 11.06 |
| EDM | 调度 | 26.11 | 15.09 | 11.56 |
| FM-OT | 调度 | 24.49 | 14.66 | 11.98 |
| Min-SNR | 损失权重 | 16.06 | 9.70 | 10.43 |
| Soft-Min-SNR | 损失权重 | 14.89 | 9.07 | 10.66 |
| Cosine Scaled | 本文调度 | 12.74 | **8.04** | 11.02 |
| Cauchy | 本文调度 | 12.91 | 8.14 | 11.02 |
| **Laplace** | **本文调度** | 16.69 | 9.04 | **7.96 (-2.89)** |

### 消融实验

**不同预测目标上的鲁棒性（FID-10K，ImageNet-256）**：

| 预测目标 | 噪声调度 | 100K | 200K | 300K | 400K | 500K |
|---------|---------|------|------|------|------|------|
| $\mathbf{x}_0$ | Cosine | 35.20 | 17.60 | 13.37 | 11.84 | 11.16 |
| $\mathbf{x}_0$ | Laplace | 21.78 | 10.86 | 9.44 | 8.73 | **8.48** |
| $\mathbf{v}$ | Cosine | 25.70 | 14.01 | 11.78 | 11.26 | 11.06 |
| $\mathbf{v}$ | Laplace | 18.03 | 9.37 | 8.31 | 8.07 | **7.96** |
| $\boldsymbol{\epsilon}$ | Cosine | 28.63 | 15.80 | 12.49 | 11.14 | 10.46 |
| $\boldsymbol{\epsilon}$ | Laplace | 27.98 | 13.92 | 11.01 | 10.00 | **9.53** |

**高分辨率 ImageNet-512**：Laplace ($b=0.75$) 的 FID 从 Cosine 的 11.91 降至 **9.09**（↓23.7%）。

### 关键发现

1. **集中在 $\log\text{SNR}=0$ 附近一致最优**：无论 Laplace、Cauchy、Cosine Scaled 哪种分布族，当概率密度集中在 $\lambda=0$ 时都取得最佳性能
2. **修改调度优于修改损失权重**：Laplace 调度（FID 7.96）显著优于同等计算预算下的 Soft-Min-SNR（FID 9.07）
3. **收敛速度更快**：在早期训练阶段（100K-200K），Laplace 调度已大幅领先 Cosine
4. **跨预测目标普适**：在 $\epsilon$、$\mathbf{x}_0$、$\mathbf{v}$ 三种预测目标上均一致优于基线
5. **Laplace 分布的 $b$ 不宜过小或过大**：$b=0.5$ 时最优（ImageNet-256），过小导致极端噪声水平训练不足，过大则回退到接近均匀分布

## 亮点与洞察

1. **概率视角的统一框架**：将噪声调度设计优雅地转化为概率分布选择问题，使得不同调度方案可以在同一坐标系下比较和设计
2. **"在哪里算"比"给多大权重"重要**：这一发现对所有迭代训练过程（不限于扩散模型）都有启发意义
3. **实用价值高**：Laplace 调度的实现只需几行代码（论文附录提供伪代码），可即插即用替换现有调度
4. **与 SD3 的 logit-normal 采样互补**：论文在附录中分析了 SD3 的采样方案也符合"集中在中间时间步"的原理

## 局限与展望

1. **只在 DiT-B 规模验证**：为控制变量未在 XL 规模模型上实验，实际大模型训练中的效果有待进一步验证
2. **超参数需根据分辨率调整**：512 分辨率需要不同的 $b$ 值（0.75 vs 0.5），无自适应方案
3. **推理侧未优化**：论文指出推理过程中的噪声分配同样值得优化，但留作未来工作
4. **FID-10K 评估**：样本量较小的 FID 可能存在波动，更大规模评估会更可靠
5. **未与最新的流匹配方法（如 SD3 的 logit-normal）在统一框架下定量比较**

## 相关工作与启发

- 与 Min-SNR 的关系：Min-SNR 通过截断高 SNR 区域的损失权重（等效于降低 $w(\lambda)$），而本文直接减少高/低 SNR 区域的采样频率
- 与 SD3 logit-normal 的关系：SD3 的 logit-normal 采样在流匹配框架下也实现了"集中在中间时间步"，验证了同一原理
- 启发：这种从采样分布角度优化训练的思路可推广到视频扩散模型等更大规模场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 概率视角的统一框架有洞察力，但核心想法"集中在中间噪声"已有先验工作暗示
- **实验充分度**: ⭐⭐⭐⭐ — 多种调度、多种预测目标、多种分辨率的系统比较，但模型规模偏小
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导清晰，公式、表格、图示配合良好
- **价值**: ⭐⭐⭐⭐ — 实用性极强，几行代码即可改善训练效率，适合被广泛采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Improved Training Technique for Shortcut Models (iSM)](../../NeurIPS2025/image_generation/improved_training_technique_for_shortcut_models.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/image_generation/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[CVPR 2026\] TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](../../CVPR2026/image_generation/taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)

</div>

<!-- RELATED:END -->
