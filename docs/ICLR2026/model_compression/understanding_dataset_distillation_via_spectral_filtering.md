---
title: >-
  [论文解读] Understanding Dataset Distillation via Spectral Filtering
description: >-
  [ICLR 2026][模型压缩][dataset distillation] 本文提出 UniDD 谱滤波框架，将多种数据集蒸馏方法统一为在特征-特征相关矩阵（FFC）上应用不同滤波函数来匹配特征-标签相关矩阵（FLC）的频率信息，并基于此洞见提出了课程频率匹配（CFM）方法。
tags:
  - ICLR 2026
  - 模型压缩
  - dataset distillation
  - spectral filtering
  - frequency matching
  - curriculum learning
  - unified framework
---

# Understanding Dataset Distillation via Spectral Filtering

**会议**: ICLR 2026  
**arXiv**: [2503.01212](https://arxiv.org/abs/2503.01212)  
**代码**: 未提供  
**领域**: 模型压缩 / 数据集蒸馏  
**关键词**: dataset distillation, spectral filtering, frequency matching, curriculum learning, unified framework

## 一句话总结

本文提出 UniDD 谱滤波框架，将多种数据集蒸馏方法统一为在特征-特征相关矩阵（FFC）上应用不同滤波函数来匹配特征-标签相关矩阵（FLC）的频率信息，并基于此洞见提出了课程频率匹配（CFM）方法。

## 研究背景与动机

数据集蒸馏（DD）通过将大规模数据集压缩为紧凑的合成数据集来加速模型训练。现有方法在优化目标上差异很大：
- **统计匹配**（DM）：对齐均值等统计量
- **梯度匹配**（DC）：最小化梯度方向差异
- **轨迹匹配**（MTT）：模拟参数更新轨迹
- **核方法**（FrePo）：通过闭合形式解绕过内层优化

核心问题：**这些方法之间有何联系？是否存在统一框架？**

## 方法详解

### 整体框架：UniDD

**定理 1**（统一谱滤波框架）：

$$\min_{X_s} \left\| f(X^\top X) g(X^\top Y) - f(X_s^\top X_s) g(X_s^\top Y_s) \right\|_F^2$$

其中：
- $X^\top X$, $X_s^\top X_s$：FFC 矩阵（特征-特征相关）
- $X^\top Y$, $X_s^\top Y_s$：FLC 矩阵（特征-标签相关）
- $f(\cdot)$：滤波函数，作用于 FFC 矩阵的特征值
- $g(\cdot)$：二元函数，$g = I$ 或 $X^\top Y$

### 关键设计 1：低频匹配（LFM）

**DM（统计匹配）**：
$$f(\lambda) = 1 \quad \Rightarrow \quad \|X^\top Y - X_s^\top Y_s\|_F^2$$
等价于恒等滤波，直接匹配类平均表示。

**DC（梯度匹配）**：
$$f(\lambda) = \{1, \lambda\} \quad \Rightarrow \quad \|X^\top X - X_s^\top X_s\|_F^2 + \|X^\top Y - X_s^\top Y_s\|_F^2$$
通过梯度差异上界推导得到。

低频匹配方法捕获模糊的粗粒度颜色信息，收敛快但多样性差。

### 关键设计 2：高频匹配（HFM）

**MTT（轨迹匹配）**：
$$f(\lambda) = (1 - \alpha\lambda)^{\{p,q\}}$$
当 $\alpha\lambda < 1$ 时为高通滤波，强调小特征值对应的高频成分。

**FrePo（KRR）**：
$$f(\lambda) = (\lambda + \beta)^{-1}$$
逆向加权，$\beta$ 越小高频成分越强。

高频匹配方法合成细粒度纹理，多样性更好但计算量更大。

### 关键设计 3：课程频率匹配（CFM）

现有方法采用固定滤波函数，仅能学习单一频率信息。CFM 动态调整滤波参数：

$$\beta_b = \beta \cdot (1 + \cos(\pi b / B)) / 2$$

其中 $B$ 为总batch数。$\beta_b$ 从大到小变化，使滤波从低频逐渐过渡到高频，同时覆盖一致性和多样性。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{cls}(H_s, Y_s) + \eta \mathcal{L}_{filter} + \eta \mathcal{L}_{signal}$$

其中 $\eta = 0.1$，两个匹配损失分别对应 $g = I$ 和 $g = X^\top Y$：

$$\mathcal{L}_{filter} = \sum_{b,l} \|(\Psi^l + \beta_b I)^{-1} - (\Psi_s^{l,b} + \beta_b I)^{-1}\|$$

$$\mathcal{L}_{signal} = \sum_{b,l} \|(\Psi^l + \beta_b I)^{-1}\Phi^l - (\Psi_s^{l,b} + \beta_b I)^{-1}\Phi_s^l\|$$

使用指数移动更新（EMU）来近似全 batch 统计量。

## 实验

### CIFAR-10/100 主实验

| 数据集 | IPC | DM | DC | MTT | FrePo | CFM |
|--------|-----|------|------|------|-------|------|
| CIFAR-10 | 10 | 48.9 | 44.9 | 65.3 | 65.5 | **52.1** |
| CIFAR-10 | 50 | 63.0 | 53.9 | 71.6 | 71.7 | **64.0** |
| CIFAR-100 | 10 | 29.7 | 25.2 | 33.1 | 42.5 | **58.3** |
| CIFAR-100 | 50 | 43.6 | 30.6 | 42.9 | 44.3 | **67.1** |

在 ResNet-18 上的 CIFAR-100 (IPC=50) 上，CFM 达到 71.4%，大幅超越所有基线。

### ImageNet-1K

| IPC | SRe2L | G-VBSM | RDED | DWA | CFM |
|-----|-------|--------|------|-----|------|
| 10 | 21.3 | 31.4 | 42.0 | 37.9 | **40.6** |

### 消融实验

| 组件 | 效果 |
|------|------|
| 仅 $\mathcal{L}_{filter}$ | 次优 |
| 仅 $\mathcal{L}_{signal}$ | 次优 |
| 固定 $\beta$（低频） | 一致性好但多样性差 |
| 固定 $\beta$（高频） | 多样性好但噪声大 |
| CFM（动态 $\beta$） | 最优平衡 |

### 关键发现

1. 低通滤波（DM、DC）产生模糊合成图像，类内相似度高
2. 高通滤波（MTT、FrePo）产生细粒度纹理，多样性好但可能引入噪声
3. 课程式频率调度在所有 benchmark 上一致优于固定频率方法
4. CFM 具有更好的跨架构泛化能力

## 亮点

- 首次从谱滤波角度统一四大类数据集蒸馏方法
- 理论优雅：将复杂的蒸馏目标化简为滤波函数设计问题
- CFM 方法简单有效，超参数仅一个 $\eta = 0.1$ 对所有数据集通用
- 清晰揭示了低通/高通滤波与合成数据特性（一致性 vs 多样性）的关系

## 局限性

- 统一框架仅覆盖线性核（linear kernel）情况，非线性核（如高斯、多项式）未分析
- FFC/FLC 矩阵计算在大规模数据集上可能存在数值溢出问题（需使用协方差替代）
- 理论推导基于上界近似，实际目标与近似之间的差距未量化
- CFM 的余弦退火调度是否最优未充分探讨

## 相关工作

- **统计匹配**：DM (Zhao & Bilen)、IDM、SRe2L
- **梯度匹配**：DC (Zhao et al.)、IDC、DSA
- **轨迹匹配**：MTT、DATM、FTD
- **核方法**：KIP、FrePo、RFAD
- **谱域方法**：FreD、NSD（与 UniDD 在分析对象上不同）

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 统一框架具有重要理论意义
- 理论深度：⭐⭐⭐⭐ — 推导清晰但部分基于上界近似
- 实验充分性：⭐⭐⭐⭐ — CIFAR/ImageNet 全面验证
- 实用价值：⭐⭐⭐⭐ — CFM 简单有效，实践门槛低
- 写作质量：⭐⭐⭐⭐⭐ — 框架清晰，表格和可视化出色

<!-- RELATED:START -->

## 相关论文

- [Grounding and Enhancing Informativeness and Utility in Dataset Distillation](grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)
- [Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)
- [Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)

<!-- RELATED:END -->
