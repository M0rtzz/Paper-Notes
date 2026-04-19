---
title: >-
  [论文解读] TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models
description: >-
  [NeurIPS 2025][自动驾驶][retransformation bias] 针对推荐系统中变换 MSE 回归模型的逆变换偏差（retransformation bias）问题，提出先发制人（preemptive）的 TranSUN 方法，通过联合学习辅助分支显式建模偏差，在训练阶段即从模型内部消除偏差，具有理论无偏保证和良好收敛性，并已部署在淘宝首页猜你喜欢的商品和短视频推荐场景。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - retransformation bias
  - regression model
  - recommender system
  - target transformation
  - debiasing
---

# TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models in Recommender Systems

**会议**: NeurIPS 2025  
**arXiv**: [2505.13881](https://arxiv.org/abs/2505.13881)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: retransformation bias, regression model, recommender system, target transformation, debiasing  
# TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models

## 一句话总结

针对推荐系统中变换 MSE 回归模型的逆变换偏差（retransformation bias）问题，提出先发制人（preemptive）的 TranSUN 方法，通过联合学习辅助分支显式建模偏差，在训练阶段即从模型内部消除偏差，具有理论无偏保证和良好收敛性，并已部署在淘宝首页猜你喜欢的商品和短视频推荐场景。

## 背景与动机

1. **回归模型在推荐系统中至关重要**：用户停留时长、交易金额、生命周期价值等预测任务广泛使用回归模型，其预测精度直接影响推荐效果和商业收益。
2. **目标变换的必要性**：推荐场景中目标数据往往高度偏斜（right-skewed），违反 MSE 的高斯假设，导致模型收敛困难；常用 $\log(y+1)$、$\sqrt{y}$ 等变换使分布接近高斯以改善收敛。
3. **逆变换偏差被长期忽视**：变换后模型学到的是 $\mathbb{E}[T(y)|x]$，通过 $T^{-1}$ 逆变换回原始空间时，由 Jensen 不等式产生系统性偏差（如凸 $T^{-1}$ 导致系统低估），这在推荐系统社区长期被忽视。
4. **事后修正方法有局限**：NTE、Smearing 等经典偏差修正方法均为训练后外部施加的 post-hoc 修正，在工业推荐系统中面临实际挑战（需额外残差统计、与在线 serving 架构不兼容等）。
5. **偏差影响商业指标**：以淘宝 GMV 预测为例，$\log$ 变换导致模型系统性低估高价值订单，直接影响排序和商业收入。
6. **缺乏内生去偏范式**：此前所有方法都是在模型外部事后修正，没有从模型训练过程内部根本消除偏差的方案。

## 方法详解

### 整体框架：联合偏差学习的先发制人去偏范式

- **功能**：在变换 MSE 模型基础上，引入辅助分支 $z(x;\theta_z)$ 显式学习偏差比率，训练阶段通过联合优化同时完成主回归任务和去偏任务。
- **为什么**：与 post-hoc 方法不同，将去偏嵌入训练过程可避免事后修正的工程复杂性，且能利用训练数据充分建模偏差。
- **怎么做**：
  1. 主分支：标准变换 MSE 损失 $\mathcal{L}_{\text{MSE}}^T = \mathbb{E}[(f(x;\theta) - T(y))^2]$，学习 $\mathbb{E}[T(y)|x]$。
  2. 辅助分支：偏差学习损失 $\mathcal{L}_{\text{sun}}^T = \mathbb{E}[(z(x;\theta_z) - \text{stop\_grad}[y/(|T^{-1}(f(x;\theta))| + \epsilon)])^2]$，学习真实值与有偏预测的比率。
  3. 总损失：$\mathcal{L}_{\text{TranSUN}}^T = \mathcal{L}_{\text{MSE}}^T + \mathcal{L}_{\text{sun}}^T$。
  4. 推理时：$\hat{y}|x = z(x;\theta_z) \cdot (|T^{-1}(f(x;\theta))| + \epsilon)$，通过比率修正恢复无偏估计。
  5. 关键：`stop_grad` 阻断偏差损失对主分支的梯度，保证两个优化任务独立。

### 关键设计1：乘法偏差建模方案的选择

- **功能**：选择让辅助分支学习 $y / T^{-1}(f)$（真实值/有偏预测）而非 $T^{-1}(f) / y$（有偏预测/真实值）。
- **为什么**：并非所有偏差学习方案都能保证理论无偏。若学习 $T^{-1}(f)/y$，最终预测是 $1/\mathbb{E}[1/y]$ 的估计，由 Jensen 不等式仍有偏；而学习 $y/T^{-1}(f)$ 可严格推导出 $\hat{y}|x = \mathbb{E}[y|x]$。
- **怎么做**：通过 MSE 回归的条件期望性质，$\min_{\theta_z} \mathcal{L}_{\text{sun}}^T$ 使 $z(x;\theta_z) = \mathbb{E}[y/(|T^{-1}(f)| + \epsilon) | x]$，代入推理公式即得 $\hat{y}|x = \mathbb{E}[y|x]$。此外，比率的方差直观上较小，使损失更平滑、训练更稳定。

### 关键设计2：Generalized TranSUN (GTS) 统一框架

- **功能**：将 TranSUN 推广为通用回归模型族 GTS，支持任意条件点估计作为主分支、任意函数 $\kappa$ 作为线性变换斜率。
- **为什么**：揭示 TranSUN 无偏性的本质机制是"条件线性"（conditional linearity）而非偏差学习本身；GTS 提供灵活框架，可直接为任意回归模型添加去偏能力（plug-and-play）。
- **怎么做**：
    - GTS 损失 = 条件点损失 $\mathcal{L}_{\mathcal{H}_q}$（主分支学习 $T(y)|x$ 的某种点估计）+ 线性变换损失（辅助分支通过动态斜率 $\kappa(f)$ 学习）。
    - 模型假设：$-z(x;\theta_z) + y \cdot \kappa(\mathbb{Q}[T(y)|x]) \sim \mathcal{N}(0,\sigma^2)$，本质是条件线性变换 MSE，天然保持无偏。
    - 应用方式：(a) 定制 $\mathcal{H}_q$ 和 $\kappa$ 直接构建新型无偏回归模型；(b) 对已有模型设 $T$ 为恒等变换，即可 plug-and-play 去偏。

## 实验

### 实验设置
- **合成数据**：8 种分布（右偏、左偏、对称），3 种变换（线性、对数、平方）
- **真实数据**：CIKM16（零售预测）、DTMart（市场混合建模）、淘宝工业数据集（GMV 预测）
- **指标**：SRE（有符号相对误差）、TRE（总比率误差）、MRE（均比率误差）、NRMSE、NMAE、XAUC
- **基线**：MSE、LogMSE、MAE、WLR、ZILN、MDME、TPM、CREAD、OptDist、NTE、Smearing、SIR

### 结果

| 方法 | T(y) | CIKM16 TRE↓ | CIKM16 MRE↓ | DTMart TRE↓ | DTMart MRE↓ |
|------|------|-------------|-------------|-------------|-------------|
| T-MSE | ln(y+1) | 0.3468 | 0.3352 | 0.2894 | 0.1432 |
| **TranSUN** | ln(y+1) | **0.0133** (-96.2%) | **0.0171** (-94.9%) | **0.0803** (-72.3%) | **0.0725** (-49.4%) |
| T-MSE | √y | 0.1386 | 0.1243 | 0.4388 | 0.3421 |
| **TranSUN** | √y | **0.0388** (-72.0%) | **0.0283** (-77.2%) | **0.0907** (-79.3%) | **0.0660** (-80.7%) |

| 对比 | TRE↓ | MRE↓ | NRMSE↓ | NMAE↓ |
|------|------|------|--------|-------|
| LogMSE（无修正） | 0.3451 | 0.3667 | 0.6189 | 0.4528 |
| LogMSE + Smearing | 0.0175 | 0.0477 | 0.5617 | 0.4335 |
| **LogSUN**（内生去偏） | **0.0123** | 0.0439 | 0.5625 | 0.4333 |

### 关键发现
1. TranSUN 在所有变换和分布类型上保持 |SRE| < 0.7%，而变换 MSE 在非线性变换下偏差可达 50%+（对数变换系统低估，平方变换系统高估）。
2. 与 post-hoc 修正方法（Smearing、SIR）性能可比，但无需额外残差统计步骤，可直接集成到在线 serving。
3. GTS 框架可为 WLR、ZILN、MDME、OptDist 等已有模型 plug-and-play 去偏，TRE 降幅均超 80%。
4. 已部署在淘宝猜你喜欢的商品推荐和短视频推荐两个核心场景，服务 DAU > 3 亿的主流量。

## 亮点

- 首次提出先发制人范式（preemptive paradigm），从模型内部消除逆变换偏差，颠覆了 post-hoc 修正的传统思路
- 理论无偏保证严格且推导清晰，揭示了"并非所有显式偏差建模方案都能保证无偏"这一非直觉结论
- GTS 统一框架具有很强的通用性，可为任意回归模型 plug-and-play 添加去偏
- 真实大规模工业部署验证（淘宝 DAU 3 亿+），具有极高实用价值

## 局限性

- 辅助分支增加了模型参数和计算，对极端低延迟场景可能有影响（尽管文中称开销很小）
- 理论无偏假设依赖于 MSE 损失下辅助分支能完美学到条件期望，实际模型容量不足时可能有残余偏差
- 实验主要聚焦电商/零售场景的右偏数据，对其他领域（如金融风险、医疗预测）的适用性验证有限
- `stop_grad` 的使用使两个分支完全独立优化，未探索联合优化是否能带来额外收益

## 相关工作对比

| 方面 | 本文 TranSUN | Post-hoc 修正方法（NTE/Smearing） |
|------|------------|-------------------------------|
| 修正时机 | 训练阶段（preemptive） | 训练后外部施加（post-hoc） |
| 架构侵入性 | 仅增加轻量辅助分支 | 不改模型，但需额外统计步骤 |
| 分布假设 | 无需假设变换后分布形式 | NTE 需假设高斯分布 |
| 工业可用性 | 可直接集成到在线 serving | 额外步骤增加工程复杂度 |

| 方面 | 本文 GTS | 条件变换模型（CLTM） |
|------|---------|------------------|
| 目标 | 无偏点估计（value regression） | 区间估计（transformation model） |
| 斜率依赖 | $\kappa$ 依赖 $x$ 和 $y$（通过点估计） | $\beta$ 仅依赖 $x$ |
| 偏差保证 | 理论无偏 + 实验验证 | 不关注逆变换偏差 |

## 评分

- ⭐⭐⭐⭐ 新颖性：先发制人范式是全新视角，揭示乘法方案与无偏性的深层联系
- ⭐⭐⭐⭐ 理论深度：无偏性证明严格，GTS 统一框架提供深入理论洞见
- ⭐⭐⭐⭐⭐ 实验充分度：合成 + 公开 + 工业数据三层验证，含大规模线上部署
- ⭐⭐⭐⭐⭐ 实用价值：已在淘宝核心场景上线，DAU 3 亿+，工业影响力突出

<!-- RELATED:START -->

## 相关论文

- [Neurosymbolic Diffusion Models](neurosymbolic_diffusion_models.md)
- [Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems](causality_meets_locality_provably_generalizable_and_scalable_policy_learning_for.md)
- [Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis](../../ICCV2025/autonomous_driving/leveraging_bev_paradigm_for_ground-to-aerial_image_synthesis.md)
- [SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)

<!-- RELATED:END -->
