---
title: >-
  [论文解读] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization
description: >-
  [ICML 2026][优化/理论][离线 BBO] SPADE 用一个条件扩散模型替代传统回归代理来建模 $p(y\mid\boldsymbol{x})$，并通过"均值/排序校准"+"kNN 支撑度正则（均值收缩 + 方差膨胀）"把数据先验隐式注入到代理里…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "离线 BBO"
  - "条件扩散代理"
  - "kNN 支撑度正则"
  - "LCB 采集函数"
---

# Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization

**会议**: ICML 2026  
**arXiv**: [2605.11246](https://arxiv.org/abs/2605.11246)  
**代码**: https://github.com/HarryYoung2018/spade (有)  
**领域**: 扩散模型 / 离线黑盒优化  
**关键词**: 离线 BBO、条件扩散代理、kNN 支撑度正则、LCB 采集函数

## 一句话总结
SPADE 用一个条件扩散模型替代传统回归代理来建模 $p(y\mid\boldsymbol{x})$，并通过"均值/排序校准"+"kNN 支撑度正则（均值收缩 + 方差膨胀）"把数据先验隐式注入到代理里，使离线黑盒优化在 Design-Bench 和 LLM 数据混合任务上稳定达到 SOTA。

## 研究背景与动机

**领域现状**：离线黑盒优化（offline BBO）只能用一个静态数据集 $\mathcal{D}=\{(\boldsymbol{x}_i,y_i)\}$ 去找最优设计，不能再查询真实 oracle。主流做法分两派：inverse 方法直接学 $p(\boldsymbol{x}\mid y)$ 来按高分条件采样设计；forward 方法学一个回归代理 $f_\theta(\boldsymbol{x})$，然后梯度上升或采集函数搜索。

**现有痛点**：inverse 方法本质上是 ill-posed 的一对多映射，训练难、容易 mode collapse；forward 方法里的确定性 MLP 给不出 epistemic 不确定性，搜索过程会"打洞"——优化器一旦找到代理高估的区域就疯狂往那儿钻，结果在真实环境里完全不靠谱。

**核心矛盾**：好的 forward 代理需要同时具备三件事——分布表达力（能给均值 + 方差）、全局精度（均值要准、排序要对）、以及对 OOD 区域的天然保守性（远离数据流形要自动调低估值）。现有方法每一项都只满足一个。

**本文目标**：1) 让扩散模型也能当 forward 代理用，捕获 $p(y\mid\boldsymbol{x})$ 的全分布；2) 让训练目标额外校准全局均值与配对排序；3) 在不另训生成模型 $p(\boldsymbol{x})$ 的情况下，把先验信息塞进代理。

**切入角度**：把 Bayes 公式 $p(\boldsymbol{x}\mid y)\propto p(y\mid\boldsymbol{x})\,p(\boldsymbol{x})$ 拆开看——forward 部分用条件扩散建模，prior 部分用 kNN 距离作非参密度估计，并在理论上证明这种几何正则化"一阶等价于"在采集函数里加上 $\log p(\boldsymbol{x})$。

**核心 idea**：用条件扩散当 forward 代理 + 校准损失锚定全局统计 + kNN 距离驱动的均值收缩/方差膨胀来注入支撑度先验，最后用 LCB + 进化算法做风险感知搜索。

## 方法详解

### 整体框架
SPADE 分两个阶段。**Surrogate Training 阶段**：在 $\mathcal{D}$ 上同时优化三个损失——基础扩散去噪损失 $\mathcal{L}_{\text{diff}}$、校准损失 $\mathcal{L}_{\text{calib}}$（均值匹配 + pairwise 排序）、支撑度近邻损失 $\mathcal{L}_{\text{prox}}$（kNN 距离驱动的 mean-shrink + variance-floor）。**Optimization 阶段**：用进化算法从高分种子出发演化候选群体，每个候选 $\boldsymbol{x}$ 用 $M$ 次 MC 采样估计 $\hat\mu_\theta(\boldsymbol{x})$ 和 $\hat\sigma_\theta(\boldsymbol{x})$，按 LCB $=\hat\mu-\beta\hat\sigma$ 选最优。

### 关键设计

1. **条件扩散 forward 代理**:

    - 功能：把传统的确定性回归代理换成 DDPM，让 $p_\theta(y\mid\boldsymbol{x})$ 是一个完整的预测分布而不是点估计。
    - 核心思路：用方差调度 $\{\beta_t\}$ 在 $y_0$ 上加噪得到 $q(y_t\mid y_0)=\mathcal{N}(\sqrt{\bar\alpha_t}y_0,(1-\bar\alpha_t)\mathbf{I})$，训一个以 $\boldsymbol{x}$ 为条件的噪声预测网络 $\epsilon_\theta(y_t,t,\boldsymbol{x})$，损失为 $\mathcal{L}_{\text{diff}}=\mathbb{E}\|\epsilon-\epsilon_\theta(y_t,t,\boldsymbol{x})\|_2^2$。推理时短跑 $M$ 步 MC 采样得到 $\{y^{(m)}\}$，从中估计预测均值和方差。
    - 设计动机：MLP 回归只给点估计，没有 $\sigma$ 就没法做 LCB / EI 等风险感知采集函数；用扩散模型天然能捕捉多模态和异方差，比 ensemble / BNN 更易扩展。

2. **Calibrated Diffusion Estimation（校准损失）**:

    - 功能：让 surrogate 在"全局均值"和"配对排序"两个指标上和真实景观保持一致，弥补普通去噪损失只关心局部分布的缺陷。
    - 核心思路：从 mini-batch 里用 $M$ 次短跑 MC 估出 $\hat\mu_\theta(\boldsymbol{x})\approx\frac{1}{M}\sum_m y^{(m)}$，然后两项相加：一阶矩匹配 $(\hat\mu_\theta(\boldsymbol{x})-y)^2$ + pairwise rank consistency $\log(1+\exp\{-s[\hat\mu_\theta(\boldsymbol{x}_i)-\hat\mu_\theta(\boldsymbol{x}_j)]\})$（仅在 $y_i>y_j$ 的有序对上算，温度 $s=1$）。
    - 设计动机：BBO 真正用的是均值的排序而不是分布的形状，单跑 $\mathcal{L}_{\text{diff}}$ 不保证排序对；rank loss 直接把"谁比谁好"显式写进训练目标，相当于把 BBO 的 utility 信号反向传到扩散网络里。

3. **Support-Proximity Regularization（支撑度近邻正则）**:

    - 功能：在不另训生成模型 $p(\boldsymbol{x})$ 的前提下，让代理在远离数据流形的 OOD 区域自动调低均值、抬高方差，从而让 LCB 在 OOD 区天然不友好。
    - 核心思路：用 kNN 第 $k$ 近邻距离 $R_k(\boldsymbol{x})$ 作密度代理，定义 $d(\boldsymbol{x})=\log R_k(\boldsymbol{x})$，则 $-\log\hat p_{\text{knn}}(\boldsymbol{x})\propto d(\boldsymbol{x})$。损失含两项 hinge：mean-shrink $\max(0,\hat\mu_\theta-\mu_{\text{NN}}-\tau(d))$ 把均值压到邻居均值附近且随距离更狠，variance-floor $\max(0,\sigma_{\min}(d)-\hat\sigma_\theta)$ 把方差顶到一个随距离单调上升的下限，其中 $\tau(d)=ad$、$\sigma_{\min}(d)=a_0+a_1 d$，默认 $a=0.02,a_0=0.02,a_1=0.005$ 全任务通用。论文证明：在 LCB 这类"$\mu$ 单增、$\sigma$ 单减"的采集函数下，$\widetilde{\mathcal{A}}(\boldsymbol{x})=\mathcal{A}(\mu,\sigma)+\kappa(\boldsymbol{x})\log\hat p_{\text{knn}}(\boldsymbol{x})+o(\cdot)$，一阶等价于在 utility 上加 $\log p(\boldsymbol{x})$ 先验。
    - 设计动机：训练独立的 $p(\boldsymbol{x})$ 生成器既贵又难调；而 kNN 是非参的、对高维和不均匀分布鲁棒；hinge 写法保证只在违反"应该保守"约束时才施加梯度，不会干扰流形内部的拟合。

### 损失函数 / 训练策略
总损失 $\mathcal{L}(\theta)=\mathcal{L}_{\text{diff}}+\lambda_1\mathcal{L}_{\text{calib}}+\lambda_2\mathcal{L}_{\text{prox}}$。推理用 LCB $\hat\mu_\theta(\boldsymbol{x})-\beta\hat\sigma_\theta(\boldsymbol{x})$ 作采集函数，用进化算法（EA）从 $\mathcal{D}$ 中高分种子初始化种群，每代评估每个候选的 LCB 然后选择/变异/交叉，最终输出 $\arg\max_{\boldsymbol{x}\in\mathcal{P}}\text{LCB}(\boldsymbol{x})$。

## 实验关键数据

### 主实验
在 Design-Bench（SuperConductor、Ant、D'Kitty、TF8、TF10）+ LLM Data Mixture（LLM-DM）共 6 个任务上，报告 $K=128$ 候选中的 100th-percentile 归一化分数（mean ± SE，8 seed）。

| 任务 | $\mathcal{D}(\text{best})$ | 之前 SOTA 范围 | SPADE | 备注 |
|------|---------------------------|---------------|-------|------|
| SuperConductor | 0.399 | 各 baseline 0.40~0.55 区间 | 最佳之一 | 校准让代理排序更准 |
| Ant Morphology | 0.565 | 0.60~0.90 区间 | 最佳之一 | 高维连续控制 |
| D'Kitty | 0.884 | ~0.90 区间 | 最佳之一 | OOD 风险大 |
| LLM-DM | 1.000 | 接近上限 | 与 baseline 持平/更稳 | LLM 数据混合优化 |
| TF8 / TF10 | 0.439 / 0.511 | 离散设计任务 | 最佳之一 | 离散空间也能用 |

SPADE 在 mean rank 和 median rank 两项综合指标上都排名第一，是唯一在 6 个任务上整体稳定 top 的方法。

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| Full SPADE | 全部 6 个任务 SOTA 或并列 | 三模块缺一不可 |
| w/o $\mathcal{L}_{\text{calib}}$ | 排序错乱，EA 选错候选 | 缺乏全局校准 |
| w/o $\mathcal{L}_{\text{prox}}$ | 经典 OOD reward hacking，EA 把估值打飞 | 没有先验约束 |
| w/o 扩散（普通 MLP 回归） | 没有 $\sigma$，LCB 退化为均值贪心 | 失去风险感知能力 |
| 改 kNN 为 KDE | 高维下崩，结果显著变差 | kNN 自适应带宽更稳 |
| 改 LCB 为均值贪心 | OOD 风险被放大 | 验证 LCB 是正则的最佳搭档 |

### 关键发现
- $\mathcal{L}_{\text{prox}}$ 是稳定性最大贡献者：去掉它后多数任务出现 reward hacking，分数比完整模型低 10~30%；它本质上是用几何代替生成式先验。
- $\mathcal{L}_{\text{calib}}$ 的 rank 项比 moment 项更关键，因为 BBO 真正消费的是相对排序而非绝对值。
- 扩散步数 $T$ 对结果不敏感（短跑就够），但 MC 采样数 $M$ 影响方差估计精度，太小会让 LCB 噪声大。
- $a, a_0, a_1$ 这套超参跨任务通用，不需要每个任务单独调，体现 kNN 几何先验的鲁棒性。

## 亮点与洞察
- "用扩散当 forward 代理"是个反直觉但合理的设计：扩散通常出现在 inverse 的 $p(\boldsymbol{x}\mid y)$ 中，本文反过来把它放到 $p(y\mid\boldsymbol{x})$ 上，巧妙之处在于 $y$ 是一维标量，扩散变得很轻量但仍能给出 $\sigma$。
- 把"几何约束"和"贝叶斯先验"用一阶等价定理画上等号，这种证明思路很值得迁移——它告诉我们：如果一个 hinge 正则项 $\tau(d)$ 随 $-\log p(\boldsymbol{x})$ 线性增长，就相当于在采集函数里加 log-prior。其他任务（如 imitation learning、offline RL）都可以套用。
- mean-shrink + variance-floor 是一对天然搭档：前者降 $\mu$、后者升 $\sigma$，两者协同让 LCB 在 OOD 区"双重打折"，比单一项更稳。

## 局限与展望
- 作者承认 Proposition 3.1 只是"动机"而非全算法保证，实际行为还受 EA、$\beta$、MC 噪声影响。
- kNN 在百维以上设计空间里仍可能退化（距离同质化），蛋白质等极高维场景需要先做表示学习或用 manifold-aware 距离。
- $\mathcal{L}_{\text{calib}}$ 需要每步 $M$ 次短跑 MC，训练时间比纯回归代理高几倍，是工程上的主要开销。
- 没有讨论 LCB 系数 $\beta$ 在不同任务间的最优范围，实际应用还得调 $\beta$。

## 相关工作与启发
- **vs DDOM / inverse 扩散方法**：他们建模 $p(\boldsymbol{x}\mid y)$ 受 ill-posed 一对多困扰；SPADE 走 forward $p(y\mid\boldsymbol{x})$ 路线 + 显式先验注入，回避了 inverse 的训练难题。
- **vs COMs / ROMA 等保守回归 baseline**：他们在 MLP 上加对抗或惩罚项做保守化，但点估计本质让 LCB 用不了；SPADE 用扩散自带分布 + kNN 几何，先验注入更显式且有 Bayes 解释。
- **vs GP / BNN 等概率代理**：GP 在高维不 scale；BNN 训练贵且未必 calibrated；扩散 + 短跑 MC 在表达力与可扩展性间取得了不错的平衡。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把扩散从 inverse 搬到 forward 的视角清新，并配套了 Bayes 等价定理。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 Design-Bench 全套 + LLM-DM，消融完整且超参跨任务通用。
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，pipeline 图把训练/优化两阶段画得很顺。
- 价值: ⭐⭐⭐⭐ 给离线 BBO 提供了一个稳定 SOTA 的新代理范式，kNN-as-prior 的思想可迁移到其他保守离线场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering](../../AAAI2026/optimization/instance_generation_for_meta-black-box_optimization_through_latent_space_reverse.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](probing_neural_tsp_representations_for_prescriptive_decision_support.md)
- [\[ICML 2026\] Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality](adaptive_estimation_and_inference_in_semi-parametric_heterogeneous_clustered_mul.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)

</div>

<!-- RELATED:END -->
