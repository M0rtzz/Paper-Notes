---
title: >-
  [论文解读] Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems
description: >-
  [ICLR 2026][医学图像][逆问题] 提出分布一致性（DC）损失，用分布级别的校准替代传统逐点数据保真项（如MSE/NLL），避免对噪声的过拟合，在DIP去噪和PET图像重建中显著提升性能且无需早停。
tags:
  - ICLR 2026
  - 医学图像
  - 逆问题
  - 数据保真项
  - 分布一致性
  - PET重建
  - Deep Image Prior
---

# Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems

**会议**: ICLR 2026  
**arXiv**: [2510.13972](https://arxiv.org/abs/2510.13972)  
**代码**: [有](https://github.com/GeorgeWebber/Distributional-Consistency-Loss)  
**领域**: 医学图像  
**关键词**: 逆问题, 数据保真项, 分布一致性, PET重建, Deep Image Prior

## 一句话总结

提出分布一致性（DC）损失，用分布级别的校准替代传统逐点数据保真项（如MSE/NLL），避免对噪声的过拟合，在DIP去噪和PET图像重建中显著提升性能且无需早停。

## 研究背景与动机

逆问题（医学成像、地球物理、信号处理等）的核心挑战是从噪声测量中恢复真实信号。传统方法将目标函数分为**数据保真项**和**正则化项**两部分。数据保真项（如MSE、负对数似然NLL）逐点地衡量预测与含噪测量的差异，但这导致一个根本性问题：**优化目标鼓励模型逐个匹配噪声实现，而非确保测量在统计意义上与模型一致**。

在含噪实现下，真实信号并非逐点数据项的最小化解。正则化必须同时承担"抑制噪声拟合"和"施加结构先验"两个任务，二者相互对抗。虽然早停或偏差准则可以缓解，但它们需要显式调参，并不改变目标函数本身。

**核心动机**：能否设计一种数据保真项，从根本上消除对噪声的拟合激励，使正则化专注于结构约束？

## 方法详解

### 整体框架

DC损失基于**概率积分变换（PIT）**：如果模型正确校准，每个测量值在其预测噪声分布中的分位数值应服从均匀分布。DC损失通过衡量这些分位数值的经验分布与均匀分布的偏离来作为数据保真项。

**三种拟合状态的直觉**：

- **欠拟合**：大多数测量值落在预测分布极端区域，分位数直方图在0或1附近有峰值
- **良好校准**：分位数直方图均匀分布
- **过拟合**：测量值集中在预测分布中心，直方图在0.5附近有尖锐峰值

### 关键设计

**Step 1 - CDF值计算**：对每个测量 $m_i$ 和预测噪声分布 $\mathcal{D}_i(\hat{y}_i)$，计算累积概率：

$$s_i = F_i(m_i | \hat{y}_i) = \mathbb{P}_{c \sim \mathcal{D}_i(\hat{y}_i)}(c \leq m_i)$$

**Step 2 - Logit变换**：直接匹配 $s_i$ 到均匀分布会导致梯度消失（远离解时 $s_i$ 饱和在0或1），因此应用logit变换：

$$r_i = \text{logit}(s_i) = \ln\frac{s_i}{1-s_i}$$

将均匀目标映射为 Logistic(0,1) 分布，同时保持梯度敏感性。

**Step 3 - Wasserstein-1距离**：对排序后的 $r_i$ 和 Logistic(0,1) 参考样本 $u_i$ 计算：

$$\mathcal{L}_{\text{DC}}(\hat{\boldsymbol{\theta}}) = \frac{1}{N}\sum_{i=1}^{N}|r_i - u_i|$$

**远离解时的行为**：通过高斯尾部近似，DC损失的梯度与MSE梯度方向一致（$\partial r_i / \partial \hat{y}_i \approx -(m_i - \hat{y}_i)/\sigma^2$），确保收敛。

**接近解时的行为**：DC损失定义了一个**等价类**——所有使CDF值近似均匀的预测都获得低损失，形成MLE附近的流形。正则化在此流形中选择最优解，而非在保真度和正则之间折中。

### 损失函数 / 训练策略

DC损失作为传统数据保真项的**即插即用替代**：

- 兼容无配对数据的无监督正则化方法
- 使用与传统损失相同的优化方式
- 无需早停即可避免噪声过拟合
- 适用条件：已知噪声分布、大量独立测量值

与正则化的协同关系是核心优势：MSE下正则化与数据保真相互对抗，DC使正则化专注于结构选择。

## 实验关键数据

### 主实验

**实验1：DIP去噪（高斯噪声）**

| 方法 | 是否需要早停 | Peak PSNR (σ=75/255) | 长期稳定性 |
|------|:---:|:---:|:---:|
| DIP-MSE | 需要 | 较低 | 1000次后退化 |
| DIP-DC | **不需要** | **更高** | 10000次仍稳定 |

DIP-DC在所有噪声水平下均超越**最优早停的**DIP-MSE，且高噪声时优势更大。

**实验2：PET图像重建（Poisson噪声）**

| 方法 | 10000次迭代表现 | 噪声伪影 | 是否需早停 |
|------|:---:|:---:|:---:|
| NLL-Adam | 严重退化 | 大量噪声尖峰 | 需要 |
| MLEM | 逐渐退化 | 逐渐累积 | 需要 |
| DC-Adam | **收敛后稳定** | **极少** | **不需要** |

**实验3：DC+TV正则化 vs NLL+TV**

| 指标 | NLL+TV | DC+TV |
|------|:---:|:---:|
| 最优NRMSE | 较高 | **更低** |
| 最优β量级 | 大 | **小数量级** |
| 图像细节 | 过度平滑 | **保留细节** |

### 消融实验

- 噪声模型误指定实验：DC损失在噪声方差估计偏差时仍保持鲁棒性
- 过参数化影响：过参数化程度越高，DC损失优势越明显
- 真实3D PET脑数据验证：DC-Adam在Siemens临床扫描仪数据上同样展现稳定行为

### 关键发现

1. DC损失在远离解时提供与MSE/NLL相同的收敛方向，接近解时自动停止追逐噪声
2. DC+TV的最优正则化强度比NLL+TV小数量级，因为DC已内置噪声抑制
3. 在真实临床PET数据上验证了实际可行性

## 亮点与洞察

- **数据保真项范式转变**：从"逐点匹配测量值"转为"分布级校准一致性"，是逆问题领域的基础性创新
- **正则化角色重定义**：DC使正则化专注结构而非同时抵抗噪声
- **理论优雅**：PIT + logit变换 + Wasserstein距离，每一步有清晰动机
- **实用性强**：无需修改网络结构或优化流程，真正的drop-in replacement

## 局限与展望

- 假设独立测量和已知噪声分布，小数据或噪声未知场景不适用
- 对离散噪声（如Poisson）需要随机化PIT
- 不保证结构属性（稀疏性等），仍需配合先验
- 前向算子病态性不在DC loss解决范围内
- 计算开销略高于逐点方法
- 未与score-based生成模型深入结合，是重要的未来方向

## 相关工作与启发

- 与鲁棒损失（Huber/Student-t）的区别：后者减少异常值影响但不阻止噪声拟合
- 与Noise2Noise的区别：N2N需多次噪声观测，DC仅需单次但要求大量独立测量
- 与经典拟合优度检验（K-S/CvM）的联系：DC可视为其可微优化版本
- 潜在扩展：与plug-and-play先验和score-based生成模型的结合

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★★★ |
| 理论深度 | ★★★★☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★★ |
| 写作质量 | ★★★★★ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes](../../AAAI2026/medical_imaging/distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[ICLR 2026\] Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling](overthinking_reduction_with_decoupled_rewards_and_curriculum_data_scheduling.md)
- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](../../CVPR2026/medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)

</div>

<!-- RELATED:END -->
