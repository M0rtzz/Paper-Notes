---
title: >-
  [论文解读] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective
description: >-
  [ICML 2026][优化/理论][零阶优化] 本文用 empirical NTK 作为统一视角，证明 zeroth-order SGD 引出的 eNTK 等价于把 first-order eNTK 投影到由微扰张成的随机子空间…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "零阶优化"
  - "eNTK"
  - "Johnson-Lindenstrauss"
  - "微扰数"
  - "维度无关性"
---

# Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective

**会议**: ICML 2026  
**arXiv**: [2605.03373](https://arxiv.org/abs/2605.03373)  
**代码**: 未提及  
**领域**: 优化理论 / LLM 微调 / 学习动力学  
**关键词**: 零阶优化, eNTK, Johnson-Lindenstrauss, 微扰数, 维度无关性

## 一句话总结
本文用 empirical NTK 作为统一视角，证明 zeroth-order SGD 引出的 eNTK 等价于把 first-order eNTK 投影到由微扰张成的随机子空间，从而通过 Johnson-Lindenstrauss 引理解释为何 ZO 方法在十亿参数 LLM 上仍然 work：误差只取决于输出维度 $V$ 和微扰数 $P$，与模型维度 $d$ 无关。

## 研究背景与动机
**领域现状**：零阶 (ZO) 优化只用函数值差分估计梯度，因为省内存、能黑盒攻击，近期被广泛用于 LLM 微调（MeZO 系列、ZO-LoRA 等）。

**现有痛点**：经典优化理论（Ghadimi-Lan 2013、Nesterov-Spokoiny 2017、Shamir 2017）一致预测 ZO 收敛率会随模型维度 $d$ 线性变慢，单微扰估计器的方差也正比于 $d$，按此推算 ZO 在十亿参数 LLM 上应该慢得不可用；但 MeZO 等实验显示 ZO 在 OPT-13B 上仍能逼近 SGD 效果。**理论与实验完全对不上。**

**核心矛盾**：把学习压缩成"损失值"这种 scalar 视角看不到 ZO 真正影响的是什么——损失下降速率确实与 $d$ 有关，但模型在具体样本上的预测变化（学习动力学）可能与 $d$ 无关。Malladi et al. 2023 的"低有效秩"假设虽给了一个解释，但在 LLM 上无法计算验证。

**本文目标**：(1) 找一个能同时刻画 ZO 与 FO 的"中间量"；(2) 证明它的差异只依赖于 $P$ 和 $V$，与 $d$ 无关。

**切入角度**：把视角从损失函数搬到 function space，借助 Jacot et al. 2018 的 eNTK；ZO 的更新可看成把 FO eNTK 经过一个低秩随机投影 $U_{t,P} U_{t,P}^\top$。这等价于 Johnson-Lindenstrauss 引理的内积保持版本，JL 引理告诉我们投影维度只需 $\mathcal{O}(\ln n / \epsilon^2)$ 与原维度无关。

**核心 idea**：**ZO-eNTK 是 FO-eNTK 的随机投影；JL 引理保证只要微扰数 $P$ 与输出维 $V$ 适配，ZO 与 FO 学习动力学的差异就和模型维数 $d$ 无关。**

## 方法详解

### 整体框架
论文是纯理论分析，没有新算法。核心 pipeline：(1) 推导 ZO-SGD 一步更新后 log-probability 的变化，把 FO 与 ZO 的差异显式写成"FO eNTK − 投影 eNTK"乘以两个 model-dependent 矩阵；(2) 把投影核差异套进 JL 引理；(3) 分别从最优化视角（方差 + 收敛率）与 eNTK 视角（投影误差）比较 Gaussian / Rademacher 微扰；(4) 讨论 $P$ 的合理量级；(5) 用 LeNet/MNIST、OPT-125M / 1.3B、Mistral-7B 等实验验证。

### 关键设计

1. **One-step learning dynamics 与 eNTK 等价形式**:

    - 功能：把 ZO-SGD 的一步参数更新对模型在另一个数据点 $\mathbf{x}_o$ 上的 log-prob 影响显式写出。
    - 核心思路：对 $\Delta \log \pi_t(y \mid \mathbf{x}_o)$ 做一阶 Taylor 展开后代入 ZO-SGD 更新规则，得到 $\Delta \log \pi \approx -\eta \mathcal{A}_t(\mathbf{x}_o) \mathcal{K}_t(\mathbf{x}_o, \mathbf{x}_u; U_{t,P}) \mathcal{G}_t(\mathbf{x}_u, \mathbf{y}_u)$，其中投影核 $\mathcal{K}_t(\mathbf{x}_o, \mathbf{x}_u; U_{t,P}) = \nabla_\theta z(\mathbf{x}_o)^\top U_{t,P} U_{t,P}^\top \nabla_\theta z(\mathbf{x}_u)$，对应 FO 版本则把 $U_{t,P} U_{t,P}^\top$ 换成单位矩阵。区别一眼可见：ZO 比 FO 多了一个由微扰拼成的随机投影矩阵 $U_{t,P} \in \mathbb{R}^{d \times P}$。
    - 设计动机：这种"差只差一个投影矩阵"的等价形式直接打通了 JL 引理，让"维度无关性"的证明变成几行 JL 推论。

2. **Johnson-Lindenstrauss 投影界**:

    - 功能：把 ZO 与 FO 核差异 $\|\Delta \mathcal{K}\|_F$ 控制成 $\epsilon$ 的函数，并显式给出维度无关界。
    - 核心思路：把 $\Delta\mathcal{K}[i,j]$ 写成原内积 $\langle \nabla_\theta z_i(\mathbf{x}_o), \nabla_\theta z_j(\mathbf{x}_u)\rangle$ 与投影内积 $\langle U_{t,P}^\top \nabla_\theta z_i, U_{t,P}^\top \nabla_\theta z_j\rangle$ 之差；JL 引理保证只要 $P \geq (2\ln n + \ln(1/\delta))/(c(\mathcal{Q})\epsilon^2)$，投影后所有内积同时被保持到 $1 \pm \epsilon$，其中 $c(\mathcal{Q})$ 是分布的集中常数。带回核差异的 Frobenius 范数得到 $\|\Delta\mathcal{K}\|_F^2 \leq \frac{\epsilon^2 V}{2}(\|\nabla_\theta z(\mathbf{x}_o)\|_F^2 + \|\nabla_\theta z(\mathbf{x}_u)\|_F^2)^2$，**右端没有 $d$**。
    - 设计动机：这正是论文要的"dimension-free"结论——只要 $V$（vocabulary / 类别数）不爆炸，模型从 LeNet 缩放到 LLaMA 都不会让 ZO 与 FO 的学习轨迹拉开太大。

3. **Gaussian vs Rademacher 微扰对比**:

    - 功能：解释"为何工程实践里二值 Rademacher 经常和 Gaussian 表现一样好"。
    - 核心思路：从最优化视角，单微扰估计器的二阶矩 Gaussian 给出 $(d+2)\|\nabla\ell\|^2$，Rademacher 给出 $d\|\nabla\ell\|^2$，两者**都正比于 $d$**，与传统"高维下都低效"一致；但从 eNTK 视角，两者的 JL 集中常数都 $\approx 1/4$，**且界不依赖于 $d$**。也就是说，二者在"投影质量"上几乎无差，论文称之为 "distribution robustness"——决定 ZO fidelity 的是 $P$ 不是分布。
    - 设计动机：弥合工程经验（Rademacher 与 Gaussian 都好用）与理论直觉（差距应该正比于 $d$）；同时为以后用更激进的二值/三值微扰提供理论支撑。

### 损失函数 / 训练策略
无新训练策略；理论部分给出当学习率 $\eta = \mathcal{O}(\sqrt{P/(dLT)})$ 时 ZO-SGD 的优化视角收敛率 $\mathcal{O}(\sqrt{dL/(PT)})$（仍含 $d$），与 eNTK 视角的维度无关界形成对比，提醒读者"收敛率 vs 学习轨迹相似度"是两件事。

## 实验关键数据

### 主实验
作者用三个实验设置验证理论：

| 设置 | 模型 | 数据 | 观察 |
|---|---|---|---|
| ZO vs FO eNTK Frobenius 误差 | LeNet ($d{=}29{,}624$) | MNIST | 高语义相似对 (4,9) 在 $P{=}125$ 时误差 $\approx 0.338$；低相似对 (0,1) 即使 $P{=}125$ 仍然显著残差 |
| Gaussian vs Rademacher | LeNet | MNIST | Frobenius / CKA / Wasserstein 三种指标曲线几乎完全重合 |
| 大模型 ZO 轨迹 | OPT-125M → OPT-1.3B | SST-2 | 增大 $P$ 时不同模型尺寸的 ZO 轨迹与 FO 接近的速度相当，验证"维度无关" |

### 消融实验

| 因素 | 影响 |
|---|---|
| 微扰数 $P$ | 误差按 $\mathcal{O}(\sqrt{\ln V / P})$ 衰减，与 JL 理论吻合 |
| 微扰分布（高斯 vs Rademacher） | 几乎无影响，验证 distribution robustness |
| 输入对的相似度 | 高相似对 ZO 收敛更快，低相似对需要更多 $P$ |
| 模型维度 $d$（OPT-125M → 1.3B） | 同一 $P$ 下两个模型 ZO 轨迹与 FO 的偏离程度类似 |

### 关键发现
- 验证了 "perturbation count $P$ 是主导，而不是 $d$"——这是对工程界长期经验的首个 kernel-level 解释。
- "样本对相似度决定收敛速度"是一个新的洞察：ZO 估计器更适合对语义相似输入做 fine-grained 区分，对结构差异巨大的输入近似变差。
- 经典优化界 $\mathcal{O}(\sqrt{dL/(PT)})$ 与 kernel 界 $\mathcal{O}(\sqrt{\ln V / P})$ 共存：损失下降速度仍然依赖 $d$，但模型预测变化轨迹的相似度不依赖 $d$。

## 亮点与洞察
- 提出了"分析 ZO 优化的 function-space 视角"这一新框架；以前的分析都困在 parameter space 里被 $d$ 卡住。
- 借 JL 引理把投影维度的依赖性从模型参数维 $d$ 转到输出空间维 $V$，是从工程现象（MeZO 在 LLM 上 work）出发反推理论的一个范例。
- 顺手解释了 Rademacher 与 Gaussian 微扰为何近乎等价——这之前只是经验观察。

## 局限与展望
- 分析是 one-step + small step-size 的局部近似，没有覆盖完整训练轨迹的累积误差。
- "维度无关"的代价是引入了 $V$（输出维），对于词表 $V \sim 10^5$ 的现代 LLM，bound 中的 $V$ 因子并不小，能否更紧仍是 open。
- 没有给出"建议 $P$ 是多少"的实用准则；仅说"$P$ 足够大就行"。
- 没有讨论 LoRA / 部分参数微扰场景下 $d_{\text{eff}}$ 与 $P$ 的关系。

## 相关工作与启发
- **vs Malladi et al. 2023b（MeZO 的低有效秩假设）**：MeZO 用 Hessian 低秩解释维度无关性，但在 LLM 上无法计算验证；本文不依赖低秩假设，直接用 JL 给出严格界。
- **vs Spall / Nesterov 经典 ZO 分析**：他们框架的指标是优化收敛率（含 $d$），本文用 eNTK 视角给出与 $d$ 解耦的指标。
- **vs Achlioptas 2003（稀疏 JL 投影）**：本文用 JL 的内积保持版本而非距离保持版本，更适配 eNTK。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "ZO eNTK = FO eNTK 的随机投影" 这个等价是非常漂亮的观察
- 实验充分度: ⭐⭐⭐ 理论文章的实验偏验证性，没有跑 LLM 微调主实验
- 写作质量: ⭐⭐⭐⭐ 推导链条清晰，公式 (6)(8)(17) 串得很紧
- 价值: ⭐⭐⭐⭐⭐ 为 ZO 用于 LLM 微调提供了首个无 trick 的 dimension-free 解释，理论框架易扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)

</div>

<!-- RELATED:END -->
