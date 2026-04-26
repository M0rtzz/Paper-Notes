---
title: >-
  [论文解读] Statistical Inference Under Performativity
description: >-
  [NeurIPS 2025][表演性预测] 本文首次建立了表演性预测（performative prediction）下完整的端到端统计推断框架，为重复风险最小化算法推导出中心极限定理和数据驱动的协方差估计方法，并将预测驱动推断（PPI）扩展到动态表演性设置以获得更紧的置信区间。
tags:
  - NeurIPS 2025
  - 表演性预测
  - 统计推断
  - 中心极限定理
  - 预测驱动推断
  - 置信区间
---

# Statistical Inference Under Performativity

**会议**: NeurIPS 2025  
**arXiv**: [2505.18493](https://arxiv.org/abs/2505.18493)  
**代码**: 无  
**领域**: 因果推断 / 统计学习  
**关键词**: 表演性预测, 统计推断, 中心极限定理, 预测驱动推断, 置信区间

## 一句话总结

本文首次建立了表演性预测（performative prediction）下完整的端到端统计推断框架，为重复风险最小化算法推导出中心极限定理和数据驱动的协方差估计方法，并将预测驱动推断（PPI）扩展到动态表演性设置以获得更紧的置信区间。

## 研究背景与动机

**领域现状**：表演性预测（performative prediction）描述的是一种广泛存在的现象：基于预测做出的决策会反过来影响被预测的目标本身。例如贷款审批政策会改变人群的消费习惯，从而影响还款能力。Perdomo等人2020年形式化了这一问题，后续工作主要聚焦于如何高效找到表演性稳定点（performative stable point，$\theta_{\text{PS}}$），并分析收敛速率。

**现有痛点**：现有文献几乎完全忽略了表演性设置下的统计推断问题。参数 $\theta$ 在很多场景下代表具体政策（如税率、信用评分阈值），仅知道估计量收敛到 $\theta_{\text{PS}}$ 是不够的——政策制定者需要置信区间和假设检验。唯一相关的工作（Cutler等2024）只为单样本在线梯度更新建立了渐近正态性，且假设所有结构信息已知，没有提供数据驱动的协方差估计方法。

**核心矛盾**：表演性设置下，数据分布随参数迭代而变化，这打破了传统统计推断依赖的固定分布假设。标准CLT无法直接适用，因为每次迭代的样本来自不同分布，且该分布依赖于上一步的（含噪声的）参数估计。

**本文目标** （1）为批量重复风险最小化（RRM）建立中心极限定理；（2）提供数据驱动的协方差估计方法（不再假设结构信息已知）；（3）将 PPI 推广到表演性设置，用少量标注数据+大量未标注数据获得更好的推断。

**切入角度**：从批量（batch）RRM入手而非单样本在线更新，更贴近政策制定的实际场景。通过创新的梯度无关得分匹配方法和策略扰动技巧来估计分布映射的梯度。

**核心 idea**：通过递推分析误差传播建立动态设置下的CLT，并用策略扰动+得分匹配绕过分布映射梯度的不可观测性，实现端到端统计推断。

## 方法详解

### 整体框架

整体框架分两条主线：（1）基础推断——为 RRM 算法的估计量 $\hat\theta_t$ 建立 CLT，推导其渐近方差 $V_t$，并提出数据驱动的方差估计方法，最终构造 $\theta_{\text{PS}}$ 的置信区间；（2）增强推断——将 PPI 引入表演性设置，在每步迭代中利用未标注数据和标注模型来减小方差、收紧置信区间。

### 关键设计

1. **动态设置下的中心极限定理**:

    - 功能：建立 RRM 估计量 $\hat\theta_t$ 的渐近分布，使得置信区间构造成为可能
    - 核心思路：关键挑战在于误差传播——$\hat\theta_t$ 的误差通过分布映射 $\mathcal{D}(\cdot)$ 传导到下一步。定理证明 $\sqrt{n}(\hat\theta_t - \theta_t) \xrightarrow{D} \mathcal{N}(0, V_t)$，其中渐近方差 $V_t = \sum_{i=1}^t \prod_{k=i}^{t-1}\nabla G(\theta_k) \cdot \Sigma_{\theta_{i-1}}(\theta_i) \cdot \prod_{k=i}^{t-1}\nabla G(\theta_k)^\top$。这是一个递推累积的方差，每一步的误差通过映射 $G$ 的雅可比矩阵 $\nabla G(\theta_k)$ 传播到后续所有步
    - 设计动机：与静态CLT不同，这里需要同时处理"每步的估计误差"和"误差在步间的传播"。通过假设 $\varepsilon < \gamma/\beta$（分布灵敏度小于强凸与光滑的比率），确保误差不会被放大

2. **基于策略扰动的得分匹配方法**:

    - 功能：数据驱动地估计分布映射的梯度 $\nabla G(\theta_k)$，这是构造 $V_t$ 的关键未知量
    - 核心思路：$\nabla G(\theta_k)$ 的表达式中包含得分函数 $\nabla_\theta \log p(z, \theta)$，但分布 $p(z,\theta)$ 未知。用参数化模型 $M(z,\theta;\psi)$ 来近似 $p$，通过得分匹配目标训练。新难点是需要对 $\theta$ 而非 $z$ 求导。通过策略扰动技巧解决：除了在 $\hat\theta_t$ 处采样外，还在 $\hat\theta_t + \eta e_i$（各坐标方向小扰动）处额外采样，用有限差分近似积分中的 $\theta$ 偏导数。由于政策维度 $d$ 通常很低，额外的 $d$ 组扰动样本代价可接受
    - 设计动机：这是一种无梯度估计方法，无需知道分布映射的函数形式，只需在多个政策值处采样观测

3. **表演性预测驱动推断（PPI under Performativity）**:

    - 功能：利用大量未标注数据和标注模型 $f$ 来减小 RRM 估计量的方差
    - 核心思路：在每步迭代中，除了少量标注数据 $\{(x_i,y_i)\}_{i=1}^n$ 外，还使用大量未标注数据 $\{x_i^u\}_{i=1}^N$（$N \gg n$）通过标注模型 $f$ 获取伪标签。PPI 估计量为 $\hat\theta_{t+1}^{\text{PPI}}(\lambda) = \arg\min_\theta \frac{\lambda}{N}\sum \ell(x_i^u, f(x_i^u);\theta) + \frac{1}{n}\sum[\ell(x_i,y_i;\theta) - \lambda\ell(x_i,f(x_i);\theta)]$。参数 $\lambda_t$ 通过逐步贪心优化选取，最小化指定的方差标量函数
    - 设计动机：政策反馈数据（如调查问卷）往往稀缺且昂贵，PPI 让研究者可以用大量容易获取的无标签数据来增强推断精度

### 损失函数 / 训练策略

RRM 每步使用强凸损失函数的经验风险最小化。得分匹配模型通过 Hyvärinen 修改的得分匹配目标训练，支持高斯参数模型和DNN两种实现。PPI 的超参数 $\lambda_t$ 通过最小化渐近方差的标量函数（如 Trace 或 $\mathbf{1}^\top V \mathbf{1}$）自适应选取。

## 实验关键数据

### 主实验

| 置信区间方法 | 覆盖率（$t=4$, $n=1000$） | 宽度 | 说明 |
|------------|-------------------------|------|------|
| $\lambda=0$（仅标注数据） | ~90% | 最宽 | 不使用未标注数据 |
| $\lambda=1$（全权重） | ~90% | 中等 | 固定权重 |
| $\lambda=\hat\lambda_t$（优化） | ~90% | **最窄** | 自适应选取 |

### 消融实验

| 得分匹配模型 | 训练损失 $J(\psi)$ | 方差估计误差 $\|\hat{V}_t - V_t\|$ |
|------------|-------------------|----------------------------------|
| 高斯参数模型 | <0.05 | 随 $n$ 增大而减小 |
| DNN（2层×128） | <0.05 | 随 $n$ 增大而减小 |

### 关键发现

- 本文的置信区间宽度为 $O(n^{-1/2})$，优于 Perdomo et al. [2020] 的 $O(n^{-1/m})$（$m \geq 2$ 为数据维度），在高维场景下优势尤其明显
- Q-Q 图验证了 CLT 的实用性：$\hat{V}_t^{-1/2}\sqrt{n}(\hat\theta_t - \theta_t)$ 的经验分布与标准正态高度吻合
- PPI 的自适应 $\lambda$ 选择始终产生最窄的置信区间，且覆盖率保持在目标水平
- 偏差感知推断（bias-aware inference）对 $\theta_{\text{PS}}$ 的置信区间与 $\theta_t$ 的置信区间之差随 $t$ 增大而指数衰减，证明少量迭代后即可获得紧的 $\theta_{\text{PS}}$ 推断
- 两种得分匹配实现（参数化高斯和DNN）都能达到 $J(\psi) < 0.05$ 的训练损失，方差估计误差可忽略

## 亮点与洞察

- **首次实现端到端推断**的关键突破在于协方差估计。之前的工作要么假设结构已知，要么只建立渐近正态性而不提供可用的置信区间。本文的策略扰动+得分匹配方法巧妙解决了分布映射梯度不可观测的难题，且实际操作代价低（策略维度 $d$ 通常很低）
- **PPI 与表演性的结合**非常自然且实用：政策反馈数据（如问卷调查）通常稀缺且应答率低，利用ML模型对大量无标签数据做伪标注来增强推断，是一个有重要应用价值的方向
- 置信区间宽度从 $O(n^{-1/m})$ 改进为 $O(n^{-1/2})$ 消除了对数据维度的依赖，这在高维政策空间中意义重大

## 局限与展望

- 当前框架使用偏差感知推断来处理 $\theta_{\text{PS}}$ 的推断，其置信区间宽度包含一个指数衰减但非零的偏差项。发展直接针对 $\theta_{\text{PS}}$ 的推断方法（无需经过 $\theta_t$ 的间接路径）是重要的未来方向
- 策略扰动需要在多个邻域策略下采样，在某些实际场景（如无法同时运行多个政策）中可能不可行
- 实验在相对低维的合成数据上进行（$d=2$），真实高维政策空间中的表现有待验证
- 得分匹配方法依赖模型 $M(z,\theta;\psi)$ 的表达能力，模型误设可能导致推断偏差

## 相关工作与启发

- **vs Perdomo et al. [2020]**: 开创性地形式化了表演性预测，但只提供 $O(n^{-1/m})$ 的非参收敛界。本文的 CLT 给出 $O(n^{-1/2})$ 的更紧界，且支持置信区间构造
- **vs Cutler et al. [2024]**: 也建立了单样本在线更新的渐近正态性，但假设所有结构信息已知，不提供协方差的数据驱动估计。本文处理批量设置且给出完整的端到端推断
- **vs PPI++ [Angelopoulos et al. 2023]**: PPI++ 在静态分布设置下将ML预测与少量标注数据结合。本文将其推广到动态表演性设置，处理了分布随参数迭代变化的额外挑战

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在表演性预测下建立完整的统计推断框架，从CLT到协方差估计到PPI扩展都有实质贡献
- 实验充分度: ⭐⭐⭐ 合成数据验证充分（CLT验证+PPI对比+得分匹配评估），但缺少真实数据实验
- 写作质量: ⭐⭐⭐⭐ 理论展开严谨清晰，但符号密集度高、可读性门槛较高
- 价值: ⭐⭐⭐⭐ 为政策制定中的量化不确定性分析提供了理论基础

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales](scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)

<!-- RELATED:END -->
