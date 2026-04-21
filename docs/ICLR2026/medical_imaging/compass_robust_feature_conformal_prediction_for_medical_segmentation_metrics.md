---
title: >-
  [论文解读] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics
description: >-
  [医学图像] COMPASS 通过在分割网络的中间特征空间沿**对目标度量最敏感的低维子空间**进行线性扰动来构建 conformal prediction 区间，在四个医学分割任务上实现了比传统 CP 方法显著更窄的预测区间，同时保持有效覆盖率。
tags:
  - 医学图像
---

# COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics

- **会议**: ICLR2026
- **arXiv**: [2509.22240](https://arxiv.org/abs/2509.22240)
- **代码**: [GitHub](https://github.com/matthewyccheung/compass)
- **领域**: 医学图像分割 / 不确定性量化
- **关键词**: conformal prediction, medical segmentation, uncertainty quantification, feature perturbation, covariate shift

## 一句话总结

COMPASS 通过在分割网络的中间特征空间沿**对目标度量最敏感的低维子空间**进行线性扰动来构建 conformal prediction 区间，在四个医学分割任务上实现了比传统 CP 方法显著更窄的预测区间，同时保持有效覆盖率。

## 研究背景与动机

**领域现状**: 医学图像分割中，临床价值通常不在于像素级分割精度，而在于由分割推导出的**下游度量**（如器官面积/体积等放射组学指标）。Conformal Prediction (CP) 是一种无分布假设的不确定性量化框架，可为预测提供统计保证。

**痛点**: (1) 像素级 CP 方法（如生成像素级可信集）提供的保证与实际临床关心的标量度量不对齐；(2) 将分割-度量管道当作黑盒直接对输出标量做 CP（Split CP），区间虽对齐但**效率低**（区间太宽），因为未利用神经网络的归纳偏置。

**核心矛盾**: Feature CP (FCP) 已证明在语义特征空间中工作可以生成更紧的区间，但 FCP 需要在高维特征空间中求解复杂的对抗优化问题，对典型 CNN/Transformer 的特征维度**计算不可行**。

**目标**: 设计一种计算可行的 Feature CP 方法，利用分割网络的中间表征为下游临床度量生成**高效**（窄）且**有效**（覆盖率有保证）的预测区间。

**切入角度**: 不在全维特征空间搜索，而是利用**目标度量关于特征的梯度 Jacobian** 找到低维敏感子空间，只沿该方向扰动。

**核心 idea**: 对分割网络中间层特征计算目标度量的 Jacobian，通过 PCA 提取主方向作为扰动方向，沿此方向线性扰动特征即可单调改变度量——因此仅需两次前向传播（正/负端点）即可高效构建嵌套预测区间。

## 方法详解

### 整体框架

将分割网络分解为三部分：编码器 $f: \mathcal{X} \to \mathcal{Z}$、解码器 $g: \mathcal{Z} \to \mathcal{S}$、度量函数 $h: \mathcal{S} \to \mathbb{R}$。COMPASS 在 $\mathcal{Z}$ 空间沿数据特定方向 $\Delta_i$ 扰动特征 $\hat{z}_i$，通过 $g$ 和 $h$ 传播得到度量变化，构建预测区间 $S_\beta(x) = [\min_{b \in [-\beta, \beta]} m_x(b), \max_{b \in [-\beta, \beta]} m_x(b)]$。

### 关键设计

#### 1. 基于 Jacobian PCA 的敏感方向计算（COMPASS-J）

**功能**: 为每个样本找到特征空间中对目标度量**最敏感**的扰动方向。

**核心思路**: 对训练集中每个样本 $i$，计算目标度量 $\hat{y}$ 对特征 $\hat{z}_i$ 的 Jacobian $J_i = \frac{d\, h(g(\hat{z}_i))}{d\hat{z}_i}$，将 Jacobian 沿空间维度求和后得到通道级向量 $\mathcal{J}_i$。对训练集所有 $\mathcal{J}_i$ 做 PCA，取前 $L$ 个主成分 $V_L$。任意新样本的扰动方向为：

$$\mathbf{d}_i = V_L V_L^T \mathcal{J}_i, \quad \Delta_i = \mathbf{d}_i / \|\mathbf{d}_i\|_2$$

**设计动机**: 全维搜索不可行，但 PCA 的第一主成分通常解释了 >90% 的度量方差（实验验证）。沿主方向扰动在实验中一致表现出**单调度量变化**——这使得预测区间只需评估两个端点（正/负 $\beta$），而非全区间扫描。

#### 2. 基于线性扰动的嵌套性保证

**功能**: 证明线性扰动构建的预测集满足嵌套性，从而保证 marginal coverage。

**核心思路**: 预测集定义为扰动区间上度量的范围 $S_\beta(x) = [\min m_x(b), \max m_x(b)]_{b \in [-\beta, \beta]}$。由于 $\beta_1 \leq \beta_2 \Rightarrow [-\beta_1, \beta_1] \subseteq [-\beta_2, \beta_2]$，最大/最小值只会扩大不会缩小，因此 $S_{\beta_1} \subseteq S_{\beta_2}$（嵌套性）。标准 CP 的交换性条件+嵌套性直接保证：

$$\mathbb{P}(Y_{n+1} \in S_{\hat{\beta}}(X_{n+1}) | D_{\text{tr}}) \geq 1 - \alpha$$

**设计动机**: 嵌套性是 CP 有效性的必要条件。对于深度特征的非线性空间，这并不平凡——本文通过"取范围"的 conservative envelope 构造从定义上保证嵌套性。

#### 3. 加权 COMPASS 应对分布偏移

**功能**: 通过密度比重新加权校准样本，在协变量偏移下恢复目标覆盖率。

**核心思路**: 训练辅助分类器区分校准集和测试集，估计密度比 $w(X_i) = p_{\text{test}}(X_i) / p_{\text{cal}}(X_i)$。使用模型深层特征或 Jacobian 作为分类器的输入特征（比类别标签或 logits 更丰富的信号）。加权符合性分位数替代等权分位数。

### 损失函数

COMPASS 不修改分割模型的训练。面积度量通过对 logits 应用 soft sigmoid 后求和得到（可微），使 Jacobian 计算可行。

## 实验关键数据

### 主实验：不同 CP 方法的区间大小（像素², Mean±Std, α=0.10）

| 数据集 | COMPASS-J | COMPASS-L | E2E-CQR | Local CP | Output-CQR | SCP |
|--------|----------|----------|---------|----------|-----------|-----|
| H&E | **3160±336** | 3139±375 | 3433±293 | 4223±558 | 3879±369 | 3509±333 |
| Skin Lesion | **1179±53** | 1208±58 | 1351±75 | 2433±101 | 4581±36 | 1813±127 |
| Nodule | **2444±174** | 2510±180 | 2788±154 | 3311±133 | 5603±57 | 3076±200 |
| PolyP | **4056±293** | 4397±469 | 6184±616 | 5965±1011 | 4981±675 | 6237±564 |

> COMPASS-J 在所有数据集的所有 α 水平上均产生最窄区间。相比 SCP，Skin Lesion 上区间缩窄 **35%**，PolyP 上缩窄 **35%**。

### 消融实验：加权 CP 在分布偏移下的表现（α=0.10）

| 方法 | H&E (hard shift) 覆盖率 | Skin Lesion (easy shift) 覆盖率 |
|------|----------------------|---------------------------|
| 无加权 SCP | ❌ 欠覆盖 | ✅ 过覆盖 |
| 类别加权 SCP | ✅ | ❌ 欠覆盖 |
| COMPASS-L + 特征加权 | ❌ 欠覆盖 | ✅ |
| COMPASS-J + 特征加权 | ✅ **最窄** | ✅ **最窄** |
| COMPASS-J + Jacobian 加权 | ✅ **最窄** | ✅ **最窄** |

> 只有 COMPASS-J（深层特征或Jacobian加权）在**两种偏移方向**上同时维持目标覆盖率，且区间最窄。

### 关键发现

1. **单调性普遍成立**: 沿 COMPASS-J 方向扰动在所有四个数据集上均导致度量单调变化，使得高效端点算法成立
2. **压缩幂律关系**: 特征空间分数 $R_{\text{COMPASS}}$ 与输出空间误差 $R_{\text{SCP}}$ 之间存在亚线性缩放（log-log 斜率 <1），系统性压缩尾部分布，这是区间更紧的根本机制
3. **深层表征 > 浅层**: COMPASS-J（深层特征）始终优于 COMPASS-L（logits），因为深层特征提供更丰富的度量敏感信号

## 亮点与洞察

- "沿敏感子空间扰动"的核心思路优雅简约：Jacobian→PCA→一条直线→两个端点，将 FCP 的不可行优化问题简化到两次前向传播
- 单调性的经验验证非常关键——它是高效算法的前提，且可通过 Jacobian 第一主成分的解释方差预判
- 压缩幂律的发现为 COMPASS 的效率优势提供了深层解释，不仅是经验观察
- 加权 COMPASS 在分布偏移下的鲁棒性有实际临床意义

## 局限性

- COMPASS 性能依赖于预训练模型表征的质量——若特征与度量关系非单调，需退化为全扫描算法
- 加权 CP 在大分布偏移（校准与测试集特征空间重叠不足）时密度比估计不准确
- 仅验证了面积度量，对纹理、形状等更复杂度量的适用性未探讨
- 基于 U-Net 架构验证，对 Transformer 类分割架构的最优层选择可能不同

## 相关工作与启发

- **Feature CP** (Teng et al., 2022): 首次证明特征空间CP可产生更紧区间，但对抗搜索不可行
- **Lambert et al. (2024)**: 端到端 CQR 用 Tversky 损失训练像素级上下界，优化的是代理目标而非目标度量
- **Split CP / CQR**: 标准输出空间方法，简单但区间宽
- **启发**: COMPASS 的"Jacobian→PCA→主方向扰动"范式可推广到任何可微度量的不确定性量化（3D体积、形状指标等）

## 评分

⭐⭐⭐⭐⭐ (5/5)

- **创新性**: ⭐⭐⭐⭐⭐ — 将 FCP 的计算瓶颈通过 Jacobian PCA 降维优雅解决，理论证明和经验验证都很扎实
- **实验**: ⭐⭐⭐⭐⭐ — 4 数据集 × 3 α 水平 × 6 基线 × 100 随机划分，标准+分布偏移+消融，极为充分
- **实用性**: ⭐⭐⭐⭐⭐ — 代码开源、即插即用、对临床度量不确定性量化有直接价值
- **写作**: ⭐⭐⭐⭐ — 理论与实验结构清晰，图示直观

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](../../ICML2025/medical_imaging/deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)
- [Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)

<!-- RELATED:END -->
