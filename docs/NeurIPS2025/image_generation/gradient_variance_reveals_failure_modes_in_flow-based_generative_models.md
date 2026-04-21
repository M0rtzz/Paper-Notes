---
title: >-
  [论文解读] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models
description: >-
  [NeurIPS 2025][图像生成][Rectified Flow] 本文通过分析 CFM 损失的梯度方差（gradient variance），揭示了 Rectified Flow 在确定性插值下会不可避免地记忆训练配对而非学习最优传输映射，并证明引入随机性（stochastic interpolant）可打破该记忆化通道、恢复泛化能力。
tags:
  - NeurIPS 2025
  - 图像生成
  - Rectified Flow
  - 梯度方差
  - 记忆化
  - 最优传输
  - 随机插值
  - Conditional Flow Matching
---

# Gradient Variance Reveals Failure Modes in Flow-Based Generative Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.18118](https://arxiv.org/abs/2510.18118)  
**代码**: 无  
**领域**: 生成模型 / 流匹配理论  
**关键词**: Rectified Flow, 梯度方差, 记忆化, 最优传输, 随机插值, Conditional Flow Matching

## 一句话总结
本文通过分析 CFM 损失的梯度方差（gradient variance），揭示了 Rectified Flow 在确定性插值下会不可避免地记忆训练配对而非学习最优传输映射，并证明引入随机性（stochastic interpolant）可打破该记忆化通道、恢复泛化能力。

## 研究背景与动机
**领域现状**：基于 ODE 的流匹配模型（Flow Matching / Rectified Flow）是当前生成建模的主流范式，通过学习源分布与目标分布间的向量场实现采样。Rectified Flow 迭代 "校直" 传输路径以实现近似一步推理。

**现有痛点**：Rectified Flow 的 "校直" 目标看似合理，但隐藏了根本性的失败模式——在确定性训练下，模型倾向于记忆训练集中的配对关系，而非学会具有泛化能力的传输映射。先前工作声称 1 次或 2 次 rectification 足以得到直线路径，但缺乏严格证明，且存在反例。

**核心矛盾**：低梯度方差被直觉上视为优化进展良好的标志，但本文指出——在确定性插值下，低方差恰恰对应记忆化解（向量场完美拟合训练配对），而非最优传输解。

**本文目标**：（1）阐明梯度方差在何种条件下是解质量的可靠指标；（2）证明确定性 ReFlow 的全局最优解就是记忆化解；（3）表明引入噪声插值可恢复泛化。

**切入角度**：从高斯到高斯的解析可控场景出发，推导闭式梯度方差表达式，然后推广到有限数据集的一般情况。

**核心 idea**：梯度方差揭示向量场质量——确定性插值让 ReFlow 收敛到记忆化，加噪声可恢复泛化。

## 方法详解

### 整体思路
本文是一篇理论分析论文，不提出新方法，而是深入剖析现有 Rectified Flow 框架的失败机制。核心工具是 CFM 训练损失的梯度方差 $\text{Var}[\nabla_\theta L_{\text{MC}}]$，用它来诊断不同训练配置（配对方式 $T$、插值方式 $I$、向量场类 $v$）下优化偏好哪种解。

### 关键理论结果

1. **Lemma 1 — 高斯情形的最优向量场闭式解**:
    - 设 $X_0 \sim \mathcal{N}(0, \mathbf{I}_d)$，$X_1 \sim \mathcal{N}(\mu, \mathbf{M}_d)$
    - OT 向量场：$\hat{v}_{OT}(X_t, t) = \hat{\theta} + \hat{\Theta}[\mathbf{I}_d + t\hat{\Theta}]^{-1}(X_t - t\hat{\theta})$
    - 关键发现：由于矩阵逆 $[\mathbf{I} + t\hat{\Theta}]^{-1}$ 项的存在，MLP/CNN/Transformer 架构无法精确表达该向量场
    - 意义：即使是最简单的高斯-高斯传输，标准神经网络参数化也存在不可消除的逼近误差

2. **Proposition 1 — 梯度方差不由轨迹交叉决定**（非正式表述）:
    - 对 OT 配对 $T_{OT}$ 和旋转配对 $T_{rOT}^R$，配对最优向量场同时取得零损失和零方差
    - 即使旋转角度不是 0°（非 OT），只要向量场与配对一致（pair-optimal），方差仍为零
    - **反直觉结论**：方差不因插值线交叉/密集区域而升高。方差由向量场与配对结构的失配决定，而非几何接近性
    - 纠正了社区中长期存在的误解—— "梯度方差主要来自插值线交叉点"

3. **Proposition 2 — 最小化器记忆化**（核心贡献）:
    - 设有限训练集 $\{(Z_0^{(i)}, Z_1^{(i)} = T(Z_0^{(i)}))\}_{i=1}^N$，确定性插值 $Z_t = (1-t)Z_0 + tZ_1$
    - **定理**：存在（确定性）向量场 $v$ 使经验损失 $L_{\text{MC}}^{\text{det}}(v) = 0$
    - ODE 积分从训练源点出发恢复其配对目标点 $\hat{X}_1 = X_1$——即完美记忆化
    - 直觉：因为从连续分布采样 $t$ 时恰好取到插值线交叉点的概率为零，向量场在交叉点处不受约束；推理时数值积分离散化有效"绕过"交叉点

4. **Counter Example 1 — 1-ReFlow 不保证直线路径**:
    - 反例构造：$T(x_0) = R_{180°} x_0 + 5$，所有插值线在 $t = 1/2$ 交叉
    - 即使传输映射 $T$ 可逆，其诱导的插值仍不可逆，新一轮 ReFlow 无法重建 $(x_0, x_1)$

5. **Remark 2 — 噪声打破记忆化**:
    - 随机插值 $x_t = (1-t)x_0 + tx_1 + f(t,\sigma)Z$ 破坏了 $(x_t, t)$ 与 $(x_0, T(x_0))$ 之间的双射关系
    - 这直接使 Lemma 2（幂等性）和 Proposition 2（记忆化）的证明假设失效
    - 即使小噪声 $\sigma = 0.05$ 也足以恢复泛化

### 关键实验观察（高斯场景）
- **Figure 6 核心实验**：180° 旋转传输使所有插值线在 $t=1/2$ 交叉，但确定性训练下模型仍学会该旋转映射并在推理时完美复现训练配对——证实记忆化
- **Figure 4**：加噪后 OT 向量场的梯度方差显著低于 pair-optimal 向量场（$p < 0.01$），说明噪声使优化偏好 OT 解
- **Figure 5**：随机配对在 OT 场下方差反而低于 120° 结构化配对——方差不由轨迹密度主导

## 实验关键数据

### 高斯混合模型实验（Table 1）

| 维度 | 方法 | 泛化 MMD↓ | 记忆化 MMD↓ |
|------|------|-----------|-------------|
| d=3  | CFM($\sigma=0$) | 0.0034 | **1.758e-6** |
| d=3  | CFM($\sigma=0.05$) | **0.0018** | 3.105e-5 |
| d=50 | CFM($\sigma=0$) | 0.0021 | **9.089e-6** |
| d=50 | CFM($\sigma=0.05$) | **0.0020** | 6.09e-5 |

- 确定性 CFM 的记忆化距离极低（几乎完美配对），但泛化差
- 加噪 $\sigma=0.05$ 的 CFM 泛化更好，且不依赖记忆

### CelebA — 对抗配对实验（Table 2）

| 指标 | CFM($\sigma=0.05$) | CFM($\sigma=0$) |
|------|-------------------|-----------------|
| 5K 泛化（L2 to OT）↓ | **34.25** ± 7.54 | 50.40 ± 16.73 |
| 5K 记忆化（L2 to Shuffled）↓ | 55.02 ± 16.51 | **28.57** ± 5.49 |
| 50K 泛化（L2 to OT）↓ | **30.05** ± 6.77 | 46.78 ± 14.87 |
| 50K 记忆化（L2 to Shuffled）↓ | 56.48 ± 18.55 | **45.98** ± 11.85 |

- 打乱 OT 配对后，确定性 CFM 仍能"记住"打乱后的错误配对
- 加噪 CFM 拒绝记忆错误配对，泛化至真实 OT 映射

### CelebA — 模拟 1-ReFlow（Table 3）

| 指标 | CFM($\sigma=0.05$) | CFM($\sigma=0$) |
|------|-------------------|-----------------|
| 5K 泛化（L2 to OT）↓ | **31.35** ± 7.38 | 43.35 ± 14.21 |
| 5K 记忆化（L2 to Generated）↓ | 25.08 ± 8.59 | **8.63** ± 1.76 |

- 1-ReFlow 进一步加剧记忆化倾向
- 数据集增大（50K vs 5K）可缓解但不消除记忆化

## 亮点
1. **纠正社区误解**：证明梯度方差不是由插值线几何交叉引起的，而是由向量场与配对结构的失配引起——这挑战了 flow matching 社区的常见直觉
2. **记忆化的严格证明**：Proposition 2 首次形式化证明确定性 ReFlow 的全局最小化器就是记忆化解，且 ODE 积分可绕过交叉点精确恢复训练配对
3. **简单而有效的修复**：仅需 $\sigma = 0.05$ 级别的插值噪声即可打破记忆化通道，无需修改网络架构或训练流程
4. **解析与实证结合**：高斯-高斯场景的闭式分析 + GMM + CelebA 真实数据的多层次验证
5. **揭示 Lemma 1 的架构限制**：MLP/Transformer 无法精确表达高斯 OT 向量场中的矩阵逆项

## 局限与展望
1. **实验规模有限**：仅在 CelebA（64×64）上验证，缺少高分辨率图像和其他模态（视频/音频/分子）的实验
2. **未提出新方法**：除了"加噪声"外未给出针对记忆化的专门解决方案；加噪在多次迭代后会收敛到熵正则化 OT，而非标准 OT
3. **大数据集行为**：承认记忆化效应在 50K 样本时已减弱（模型容量不足以完美记忆），未探讨百万级数据下的行为
4. **噪声强度选择**：$\sigma$ 的最优选取缺乏理论指导，不同任务可能需要不同噪声水平
5. **与 Consistency Model 等快速采样方法的关系**：未分析这些方法是否也存在类似的记忆化风险

## 与相关工作的对比

| 方面 | 本文 | Rectified Flow (Liu 2022) | SBM (Shi et al. 2024) |
|------|------|---------------------------|----------------------|
| 训练方式 | 分析确定性 vs 随机插值 | 确定性插值迭代校直 | 双向随机插值 |
| 记忆化 | 理论证明确定性训练必然记忆化 | 未讨论 | 噪声自然避免 |
| 配对方式 | 指出确定性配对是记忆化根源 | 使用确定性配对 | 每次独立采样 |
| 方差分析 | 深入分析梯度方差的诊断作用 | 未涉及 | 未涉及 |
| 最优传输 | 证明低方差 ≠ OT 最优 | 以 OT 为目标 | 近似熵正则化 OT |

本文的记忆化研究与 Bamberger et al. (2025)、Buchanan et al. (2025) 等扩散模型记忆化研究有交叉，但关键区别在于本文聚焦 Rectified Flow 的确定性配对设定而非标准 CFM 的独立采样设定。

## 启发与关联
1. **对 Rectified Flow 实践者的警示**：使用确定性 ReFlow 迭代时应意识到记忆化风险，至少在第一次 rectification 后应引入噪声
2. **梯度方差作为诊断工具**：可将梯度方差监控纳入 flow matching 的训练流程，用于检测是否陷入记忆化
3. **latent diffusion / video diffusion 的启示**：大规模模型（如 Stable Diffusion、Sora）是否存在类似的隐式记忆化？本文框架提供了分析工具
4. **噪声注入的本质**：加噪不仅是正则化手段，更从根本上改变了优化 landscape 的结构——从单一记忆解变为偏好 OT 解
5. **与 SDE vs ODE 之争的关联**：本文为"SDE（有噪声）比 ODE（无噪声）更不容易过拟合"这一经验观察提供了严格的理论基础

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次严格证明 Rectified Flow 的记忆化机制
- 实验充分度: ⭐⭐⭐ — 理论严谨但实验限于 CelebA
- 写作质量: ⭐⭐⭐⭐ — 理论证明清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 对 flow matching 社区有重要的警示和指导意义

<!-- RELATED:START -->

## 相关论文

- [Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [Failure Prediction at Runtime for Generative Robot Policies](failure_prediction_at_runtime_for_generative_robot_policies.md)
- [A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](../../CVPR2025/image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)

<!-- RELATED:END -->
