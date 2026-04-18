---
title: >-
  [论文解读] Blind2Sound: Self-Supervised Image Denoising without Residual Noise
description: >-
  [ICCV2025][图像恢复][自监督去噪] 提出 Blind2Sound 框架，通过自适应重可见损失（adaptive re-visible loss）感知噪声水平并实现个性化去噪，配合 Cramer Gaussian 损失提升噪声参数估计精度，在自监督盲去噪中消除残余噪声，性能超越同期所有自监督方法甚至部分有监督基线。
tags:
  - ICCV2025
  - 图像恢复
  - 自监督去噪
  - Poisson-Gaussian噪声
  - 盲去噪
  - 残余噪声消除
  - 噪声感知
---

# Blind2Sound: Self-Supervised Image Denoising without Residual Noise

**会议**: ICCV2025  
**arXiv**: [2303.05183](https://arxiv.org/abs/2303.05183)  
**作者**: Zejin Wang, Jiazheng Liu, Hao Zhai, Hua Han（中国科学院自动化研究所）
**代码**: 论文补充材料中提供  
**领域**: image_restoration  
**关键词**: 自监督去噪, Poisson-Gaussian噪声, 盲去噪, 残余噪声消除, 噪声感知

## 一句话总结

提出 Blind2Sound 框架，通过自适应重可见损失（adaptive re-visible loss）感知噪声水平并实现个性化去噪，配合 Cramer Gaussian 损失提升噪声参数估计精度，在自监督盲去噪中消除残余噪声，性能超越同期所有自监督方法甚至部分有监督基线。

## 研究背景与动机

### 问题定义

实际成像传感器中的噪声通常服从 Poisson-Gaussian 混合模型：$\mathbf{y} = \alpha P + N$，其中 $P \sim \text{Poisson}(\mathbf{x}/\alpha)$ 为信号相关的泊松噪声，$N \sim \mathcal{N}(0, \sigma^2)$ 为信号无关的高斯噪声。自监督盲去噪需要仅从单张噪声图像学习去噪，不依赖配对干净数据。

### 现有方法的局限

**伪监督对方法**（如 Noise2Noise 变体）：从单幅噪声图像构造训练对会二次腐蚀信号，降低性能

**盲点方法**（如 N2V, BSN 系列）：输入遮蔽导致信息损失，产生严重伪影

**Blind2Unblind**：通过重可见（re-visible）过渡实现无损去噪，但 MSE 目标函数无法感知噪声水平，导致像素级贪心拟合产生**明显残余噪声**

**FBI-Denoiser**：高斯损失缺乏精细约束，噪声估计精度不足，且后处理步骤会放大之前的去噪误差

### 核心动机

MSE 作为目标函数无法针对动态噪声水平调整去噪强度。需要设计一种自适应损失，在保持无损框架兼容性的同时，根据感知到的噪声水平进行个性化去噪，彻底消除残余噪声。

## 方法详解

### 整体架构

Blind2Sound 包含两个模块：

1. **去噪网络** $f_\omega(\cdot)$：基于改进的 U-Net，输出遮蔽分支和可见分支的去噪结果（均值 $\mu_m, \mu_v$ 和协方差 $\Sigma_m, \Sigma_v$）
2. **噪声估计器** $g_\theta(\cdot)$：预测 Poisson-Gaussian 噪声参数 $(\alpha, \sigma_1, \sigma_2)$

训练时两模块联合优化；**推理时去除噪声估计器和遮蔽分支**，去噪器直接从原始噪声图像生成结果。

### 自适应重可见损失（Adaptive Re-Visible Loss）

核心思路是将遮蔽分支和可见分支建模为两个独立的高斯生成过程：

- **遮蔽分支**：$\mathbf{z}_1 \sim \mathcal{N}(\mathbf{z}_1 | \mu_m, \Sigma_m)$，从遮蔽噪声体 $\Omega\mathbf{y}$ 生成潜在干净图像
- **可见分支**：$\mathbf{z}_2 \sim \mathcal{N}(\mathbf{z}_2 | \mu_v, \Sigma_v)$，从原始噪声图像 $\mathbf{y}$ 生成，梯度禁用（不参与反向传播）

通过边际似然将噪声模型显式纳入：

$$p(\mathbf{y}) = \int p(\mathbf{y}|\mathbf{x}) p(\mathbf{x}|\mathbf{y}, \Omega\mathbf{y}) d\mathbf{x}$$

最终最小化负对数似然得到自适应重可见损失：

$$\mathcal{L}_{arv} = \frac{1}{2}[(\mathbf{y} - \mu_y)^T \Sigma_y^{-1} (\mathbf{y} - \mu_y)] + \frac{1}{2}\log|\Sigma_y| + \text{const}$$

其中 $\mu_y = \frac{\mu_m + \lambda \mu_v}{1+\lambda}$，$\lambda$ 为从 3 逐步增长到 11 的可见因子。

**关键设计**：

- 两分支建模为 i.i.d.，解耦遮蔽和可见分支的相关性，避免遮蔽结果抑制可见去噪
- 对中间介质 $\mu_m$ 的梯度分析表明需禁用 $\text{diag}(\alpha \mu_y)$ 的梯度以稳定训练
- 收敛时最优估计 $\tilde{\mathbf{x}} = \frac{\mu_m + \lambda \mu_v}{1+\lambda}$，满足 $\mu_m \leq \tilde{\mathbf{x}} \leq \mu_v$
- 无需 Laine19 的后处理 MAP，因为损失本身已包含来自 $\mathbf{y}$ 的信息

### Cramer 高斯损失

针对原始高斯损失仅在全局图像上估计噪声、忽略局部噪声知识的缺陷，引入细粒度约束：

**单通道图像**（如灰度图）：利用子块约束，从四个角裁剪四个重叠子块（原图 3/4 大小），子块和全局的 GAT 变换后噪声方差均应近似单位方差：

$$\mathcal{L}_{est} = \sum_{s=1}^{4} \|\eta(G_{g_\theta}(\mathbf{y}_s)) - 1\|_2^2 + \|\eta(G_{g_\theta}(\mathbf{y})) - 1\|_2^2$$

**多通道图像**（如 sRGB）：引入跨通道噪声水平一致性约束，解决通道间估计误差互相抵消的问题：

$$\mathcal{L}_{est} = \sum_{j \neq k}^{c} \|\eta(G_{g_\theta}(\mathbf{y}_j)) - 1\|_2^2 + \|\eta(G_{g_\theta}(\mathbf{y}_j)) - \eta(G_{g_\theta}(\mathbf{y}_k))\|_2^2$$

Cramer 高斯损失仅作为正则项（权重 0.01），因为去噪后图像的实际噪声水平可能与原始输入不同。

### 总损失

$$\mathcal{L} = \mathcal{L}_{arv} + 0.01 \cdot \mathcal{L}_{est}$$

## 实验关键数据

### 噪声估计精度

在 BSD68（灰度）和 CBSD68（sRGB）上，Cramer 高斯损失相比 FBI-D：
- 灰度图上消除了 FBI-D 的严重高斯参数估计误差
- sRGB 图上跨通道约束使得预测更接近真实值

### 合成灰度去噪（PSNR/SSIM）

| 噪声 | 方法 | BSD68 | Set12 | Urban100 |
|------|------|-------|-------|----------|
| PG1 | Blind2Unblind | 30.61/0.869 | 31.45/0.880 | 30.70/0.900 |
| PG1 | **Blind2Sound** | **30.83/0.875** | **31.68/0.886** | **31.14/0.908** |
| PG3 | Blind2Unblind | 27.02/0.757 | 27.65/0.796 | 26.54/0.805 |
| PG3 | **Blind2Sound** | **27.17/0.766** | **27.96/0.805** | **26.96/0.819** |

- 相比 Blind2Unblind 最大增益 **0.44 dB**，最小增益 **0.15 dB**
- 在 Set12 和 Urban100 上**超越有监督基线** N2C 和 N2N，最大增益 0.4 dB

### 合成 sRGB 去噪

| 噪声 | 方法 | KODAK | SET14 | BSD300 |
|------|------|-------|-------|--------|
| PG1 | Blind2Unblind | 33.88/0.915 | 32.47/0.886 | 32.53/0.913 |
| PG1 | **Blind2Sound** | **34.23/0.920** | **32.75/0.896** | **33.00/0.921** |

### 真实世界去噪

| 方法 | SIDD Benchmark (RAW) | SIDD Validation (RAW) | FMD Confocal |
|------|---------------------|----------------------|--------------|
| N2C (有监督) | 50.61/0.991 | 51.19/0.991 | 38.40/0.966 |
| Blind2Unblind | 50.79/0.991 | 51.36/0.992 | 38.44/0.964 |
| **Blind2Sound** | **50.92/0.991** | **51.50/0.992** | **38.46/0.965** |

- SIDD RAW 上超越所有自监督方法和有监督基线
- 比 FBI-D 在 RAW 空间增益近 **0.3 dB**

### 消融实验要点

| 实验 | 结论 |
|------|------|
| 粒度大小 | 细粒度子块约束提升粗粒度估计精度，但子块太小导致噪声上下文不足 |
| Cramer 损失权重 | 0.01 最优，0（无正则）和 100（过强正则）均劣 |
| 训练方案 | 联合训练 > 固定预训练估计器 ≈ 固定真实噪声（低噪时差异大） |
| 噪声模型 | 增强模型 $\mathcal{M}_E$ 在高噪声下优势明显 |
| 分支独立性 | IID（独立）远优于 non-IID（非独立），验证解耦设计的重要性 |
| 可见因子 | $\lambda_f = 11$ 最优，非单调递增关系 |

## 亮点与洞察

1. **噪声感知 + 无损去噪的统一**：将噪声水平估计嵌入重可见损失，使去噪强度自适应调整——这是消除残余噪声的关键
2. **推理零开销**：噪声估计器和遮蔽分支仅在训练中使用，推理时与 Blind2Unblind 完全相同，不增加计算量
3. **贝叶斯视角的重新建模**：将 Blind2Unblind 的 MSE 损失提升为混合高斯边际似然的负对数似然，理论上更优
4. **跨通道/子块细粒度约束**：Cramer 高斯损失通过多尺度噪声一致性缩小解空间，解决了 FBI-D 估计不准的问题
5. **灰度图上超越有监督**：在 Set12、Urban100 上超越 N2C/N2N，说明自监督方法在噪声感知下的潜力

## 局限与展望

1. **sRGB 空间增益有限**：高噪声下跨通道去噪更困难，增益较灰度图小；跨通道建模可进一步加强
2. **噪声模型假设**：依赖 Poisson-Gaussian 模型，对更复杂的真实噪声（如空间相关噪声）可能需要扩展
3. **可见因子调参**：$\lambda$ 的初始值和最终值需手动设定，自适应调度可能更优
4. **单一网络架构**：仅使用改进 U-Net，引入更强的 backbone（如 Transformer）可能进一步提升
5. **缺少视频去噪**：框架针对静态图像，时序信息未利用

## 相关工作与启发

- **Blind2Unblind (CVPR 2022)**：直接前身，Blind2Sound 在其重可见框架上加入噪声感知
- **FBI-Denoiser (CVPR 2021)**：首次在自监督框架中引入高斯损失做噪声估计，但精度不足
- **Noise2Void / Noise2Self**：盲点去噪的奠基性工作，但信息损失严重
- **NBR2NBR (CVPR 2021)**：子采样构造训练对，但邻近像素近似导致过度平滑
- **GAT + BM3D**：传统方法，数据驱动方法已全面超越
- **启发**：将噪声模型的显式知识纳入自监督损失设计是提升性能的有效路径，后续可探索将此思路应用于其他退化类型（如模糊、压缩伪影）

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评
