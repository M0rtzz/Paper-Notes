---
title: >-
  [论文解读] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning
description: >-
  [ICML2025][人体理解][类增量学习] 提出 DPCR 方法，通过双投影（任务级 TSSP + 类别级 CIP）估计语义漂移，并用岭回归无BP地重建分类器，同时解决无样例类增量学习中的语义漂移和决策偏差问题，在多个基准上超越 SOTA。
tags:
  - ICML2025
  - 人体理解
  - 类增量学习
  - 无样例存储
  - 语义漂移估计
  - 双投影
  - 岭回归分类器重建
---

# Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning

**会议**: ICML2025  
**arXiv**: [2503.05423](https://arxiv.org/abs/2503.05423)  
**代码**: [RHe502/ICML25-DPCR](https://github.com/RHe502/ICML25-DPCR)  
**领域**: 域适应 / 持续学习  
**关键词**: 类增量学习, 无样例存储, 语义漂移估计, 双投影, 岭回归分类器重建

## 一句话总结

提出 DPCR 方法，通过双投影（任务级 TSSP + 类别级 CIP）估计语义漂移，并用岭回归无BP地重建分类器，同时解决无样例类增量学习中的语义漂移和决策偏差问题，在多个基准上超越 SOTA。

## 研究背景与动机

无样例类增量学习（EFCIL）要求模型在不存储旧数据的前提下顺序学习新类别，但面临两大核心挑战：

**语义漂移（Semantic Shift）**：学习新任务后 backbone 更新导致旧类别的嵌入在特征空间中发生偏移，已学特征表示与旧类不再兼容

**决策偏差（Decision Bias）**：分类器仅用新任务数据通过 BP 训练，导致对新类别产生偏好（task-recency bias），破坏新旧知识平衡

现有方法的局限：
- **冻结 backbone**（ACIL、FeCAM）：消除语义漂移但严重限制可塑性（plasticity）
- **NCM 分类器**（SDC、ADC）：依赖表示质量，缺乏可训练参数，适应性差
- **LDC**：仅捕获任务级漂移，忽略类别间差异；且需 BP 训练投影器，计算开销大

## 方法详解

DPCR 包含三个阶段：增量表示学习、双投影漂移估计、岭回归分类器重建。

### 1. 增量表示学习

沿用 LwF 的知识蒸馏框架，训练损失为：

$$\mathcal{L}_{\text{rep}} = \mathcal{L}_{\text{ce}}(h_{\tau_t}^{\text{au}}(f_{\theta_t}(\mathcal{X}_t)), y_t) + \alpha \mathcal{L}_{\text{kd}}(\mathcal{X}_t)$$

其中 $\mathcal{L}_{\text{kd}}$ 约束新旧 backbone 输出的 logit 一致性，$\alpha$ 为蒸馏权重。

### 2. 双投影漂移估计（DP）

**任务级语义漂移投影（TSSP）**：学一个线性投影 $\boldsymbol{P}^{t-1 \to t} \in \mathbb{R}^{d \times d}$ 将旧 backbone 的嵌入映射到新 backbone 空间。通过最小化 MSE 的闭式解获得：

$$\boldsymbol{P}^{t-1 \to t} = (\boldsymbol{X}_t^{\theta_{t-1}\top} \boldsymbol{X}_t^{\theta_{t-1}} + \epsilon \boldsymbol{I})^{-1} \boldsymbol{X}_t^{\theta_{t-1}\top} \boldsymbol{X}_t^{\theta_t}$$

其中 $\epsilon = 10^{-9}$ 防止矩阵病态。关键优势：**无需 BP 训练**，直接闭式求解。

**类别信息投影（CIP）**：TSSP 对同一任务的所有类别共享同一投影，忽略类别差异。CIP 通过行空间投影注入类别信息：

- 对每类的无中心协方差 $\Phi_{t-1,c}^{\theta_{t-1}}$ 做 SVD 分解
- 取非零奇异值对应的奇异向量 $\boldsymbol{U}_{t-1,c}^r$ 构造行空间投影算子
- 最终类别感知投影：$\boldsymbol{P}_{t-1,c}^{t-1 \to t} = \boldsymbol{P}^{t-1 \to t} \boldsymbol{U}_{t-1,c}^r \boldsymbol{U}_{t-1,c}^{r\top}$

CIP 是 **training-free** 的，不增加训练成本。

### 3. 岭回归分类器重建（RRCR）

将分类器训练表示为岭回归问题，避免 BP 带来的决策偏差：

$$\hat{\boldsymbol{W}}_t = \left(\sum_{i=1}^{t} \boldsymbol{\Phi}_i^{\theta_t} + \gamma \boldsymbol{I}\right)^{-1} \sum_{i=1}^{t} \boldsymbol{H}_i^{\theta_t}$$

由于 EFCIL 约束下无法获取旧数据的新嵌入，利用 DP 估计的漂移校准旧信息：

- 协方差校准：$\hat{\boldsymbol{\Phi}}_{i,c}^{\theta_t} = \boldsymbol{P}_{i,c}^{t-1 \to t \top} \boldsymbol{\Phi}_{i,c}^{\theta_{t-1}} \boldsymbol{P}_{i,c}^{t-1 \to t}$
- 原型校准：$\hat{\boldsymbol{\mu}}_{i,c}^{\theta_t} = \boldsymbol{\mu}_{i,c}^{\theta_{t-1}} \boldsymbol{P}_{i,c}^{t-1 \to t}$

**类别归一化（CN）**：双投影非酉变换会引入数值不平衡，对分类器权重按列做 L2 归一化：

$$\hat{\boldsymbol{W}}_t' = \left[\frac{\boldsymbol{w}_1}{\|\boldsymbol{w}_1\|_2}, \frac{\boldsymbol{w}_2}{\|\boldsymbol{w}_2\|_2}, \ldots, \frac{\boldsymbol{w}_{tC}}{\|\boldsymbol{w}_{tC}\|_2}\right]$$

每类只需存储 $d^2 + d$ 大小的协方差和原型，内存代价低。

## 实验关键数据

Backbone: ResNet-18 | 正则化因子 $\gamma$: CIFAR-100=200, Tiny-ImageNet=2000, ImageNet-100=2000

### 主实验（Cold-start 设定，3次运行均值）

| 方法 | CIFAR-100 T=10 $\mathcal{A}_f$/$\mathcal{A}_{avg}$ | CIFAR-100 T=20 | Tiny-IN T=10 | ImageNet-100 T=10 |
|------|------|------|------|------|
| LwF | 42.60/58.51 | 36.34/51.52 | 26.99/42.92 | 42.25/61.23 |
| ACIL | 35.53/50.53 | 27.22/39.58 | 26.10/41.86 | 44.61/59.77 |
| LDC | 46.60/61.67 | 36.76/53.06 | 33.74/47.37 | 49.98/67.47 |
| **DPCR** | **50.24/63.21** | **38.98/54.42** | **35.20/47.55** | **52.16/67.51** |

DPCR 在 CIFAR-100 T=10 上 $\mathcal{A}_f$ 超越次优 LDC **+3.64%**，ImageNet-100 T=20 超越 **+3.48%**。

### 消融实验（CIFAR-100 T=10）

| 组件 | $\mathcal{A}_f$ (%) | $\mathcal{A}_{avg}$ (%) |
|------|------|------|
| RRCR only | 32.17 | 44.89 |
| +TSSP | 40.86 | 55.76 |
| +TSSP+CIP | 45.56 | 62.15 |
| +TSSP+CIP+CN | **51.04** | **64.44** |

TSSP 贡献最大（+8.69%），CIP 进一步提升 +4.70%，CN 修正数值不平衡再涨 +5.48%。

### 大规模数据集（ImageNet-1k T=10）

| 方法 | $\mathcal{A}_f$ (%) | $\mathcal{A}_{avg}$ (%) |
|------|------|------|
| LDC | 35.15 | 53.88 |
| **DPCR** | **35.49** | **54.22** |

## 亮点与洞察

1. **双投影 = 任务级 + 类别级**：TSSP 捕获全局漂移、CIP 注入类别局部信息，比 LDC 仅做任务级更全面
2. **全闭式求解**：TSSP 和 RRCR 均无需 BP 训练，计算效率高且无迭代优化的不稳定性
3. **稳定性-可塑性均衡**：RRCR 不像 NCM 完全依赖表示质量，也不像 BP 训练会覆盖旧决策边界
4. **CIP 同时增强稳定性和可塑性**：消融可视化显示 CIP 对旧类和新类的准确率均有提升
5. **DP-NCM 实验设计巧妙**：固定 backbone 只变漂移估计方法，公平证明 DP 优于 ADC/LDC 的估计

## 局限与展望

1. **线性投影假设**：假设新旧特征间存在线性映射关系，对非线性漂移的捕获能力有限
2. **协方差存储开销随特征维度平方增长**：$d^2 + d$ 每类，当 $d$ 很大时可能成为瓶颈
3. **任务间漂移的累积误差**：多任务链式校准可能累积估计误差
4. **仅验证分类任务**：未扩展到检测、分割等更复杂的增量学习场景
5. **Cold-start 设定限制**：所有任务等分类别数，未考虑类别不均衡的更现实场景
6. **Backbone 固定为 ResNet-18**：未验证在 ViT 等现代架构上的表现

## 相关工作与启发

- **LDC**（Gomez-Villa et al., 2024）：DPCR 的直接改进对象，LDC 仅做任务级线性投影且需 BP 训练
- **ACIL**（Zhuang et al., 2022）：首个将解析学习引入 CIL 的工作，DPCR 继承了岭回归思路但不冻结 backbone
- **SDC/ADC**：仅估计原型平移，DPCR 的双投影捕获了更丰富的变换信息

## 评分

- 新颖性: ⭐⭐⭐⭐ （双投影 + RRCR 组合新颖，CIP 行空间投影思路简洁优雅）
- 实验充分度: ⭐⭐⭐⭐ （5个数据集 + 充分消融 + 公平NCM对比 + 可视化，但缺 ViT 验证）
- 写作质量: ⭐⭐⭐⭐ （公式推导清晰，从问题到方案逻辑自洽）
- 价值: ⭐⭐⭐⭐ （为 EFCIL 中同时解决漂移和偏差提供了统一框架，实用性强）

<!-- RELATED:START -->

## 相关论文

- [External Knowledge Injection for CLIP-Based Class-Incremental Learning](../../ICCV2025/human_understanding/external_knowledge_injection_for_clip-based_class-incremental_learning.md)
- [Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](../../ECCV2024/human_understanding/exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)
- [Policy Compatible Skill Incremental Learning via Lazy Learning Interface](../../NeurIPS2025/human_understanding/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)
- [KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment](kelps_a_framework_for_verified_multi-language_autoformalization_via_semantic-syn.md)

<!-- RELATED:END -->
