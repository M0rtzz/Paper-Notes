---
title: >-
  [论文解读] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update
description: >-
  [ICML2025][其他/bandits][heavy-tailed bandits] 提出基于 Online Mirror Descent 的单遍 Huber 回归算法 Hvt-UCB，用于重尾噪声线性 bandit，将每轮计算复杂度从 $\mathcal{O}(t\log T)$ 降至 $\mathcal{O}(1)$，同时保持最优且依赖实例的 regret 界。
tags:
  - "ICML2025"
  - "其他/bandits"
  - "heavy-tailed bandits"
  - "Huber regression"
  - "online mirror descent"
  - "linear bandits"
  - "one-pass algorithm"
---

# Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update

**会议**: ICML2025  
**arXiv**: [2503.00419](https://arxiv.org/abs/2503.00419)  
**代码**: 无  
**领域**: 其他/bandits  
**关键词**: heavy-tailed bandits, Huber regression, online mirror descent, linear bandits, one-pass algorithm

## 一句话总结

提出基于 Online Mirror Descent 的单遍 Huber 回归算法 Hvt-UCB，用于重尾噪声线性 bandit，将每轮计算复杂度从 $\mathcal{O}(t\log T)$ 降至 $\mathcal{O}(1)$，同时保持最优且依赖实例的 regret 界。

## 研究背景与动机

**问题设置**：随机线性 bandit（SLB）中，每轮观测到的奖励 $r_t = X_t^\top \theta_* + \eta_t$，其中噪声 $\eta_t$ 可能服从重尾分布（仅有有限 $(1+\varepsilon)$ 阶矩，$\varepsilon \in (0,1]$）。这在金融市场、在线广告等场景中很常见。

**已有方法的局限**：

| 方法类型 | 代表算法 | 主要限制 |
|---------|---------|---------|
| 截断法 (Truncation) | TOFU, CRTM | 需要绝对矩有界假设，无噪声时次优 |
| 中位数均值法 (MOM) | MENU, CRMM | 依赖固定臂集合和重复拉取机制 |
| 自适应 Huber 回归 | HEAVY-OFUL | 需存储所有历史数据，每轮计算 $\mathcal{O}(t\log T)$ |

**核心动机**：HEAVY-OFUL（Huang et al., 2023）虽然实现了最优且依赖实例的 regret 界，且不依赖额外噪声假设，但其计算代价过高——每轮需遍历全部历史数据求解强凸优化问题。作者提问：**能否设计一个基于 Huber 损失的单遍算法，在保持最优 regret 的同时大幅降低计算成本？**

## 方法详解

### 整体框架

算法 **Hvt-UCB** 将自适应 Huber 回归嵌入 Online Mirror Descent（OMD）框架，用单步投影梯度更新替代全批量优化。

### Huber 损失函数

$$f_\tau(x) = \begin{cases} \frac{x^2}{2} & |x| \leq \tau \\ \tau|x| - \frac{\tau^2}{2} & |x| > \tau \end{cases}$$

Huber 损失在 $|x| \leq \tau$ 时为二次函数（保持强凸性），在 $|x| > \tau$ 时线性化（降低异常值影响）。基于此定义 bandit 损失：

$$\ell_t(\theta) = f_{\tau_t}\!\left(\frac{r_t - X_t^\top \theta}{\sigma_t}\right)$$

其中 $\sigma_t$ 为归一化因子，$\tau_t$ 为动态鲁棒化参数。

### OMD 单遍更新（核心设计）

不同于 HEAVY-OFUL 在每轮求解全批量优化 $\hat\theta_t = \arg\min_\theta \lambda\|\theta\|^2 + \sum_{s=1}^{t} \ell_s(\theta)$，本文采用 OMD 更新：

$$\hat\theta_{t+1} = \arg\min_{\theta \in \Theta} \left\{ \langle \theta, \nabla\ell_t(\hat\theta_t) \rangle + \mathcal{D}_{\psi_t}(\theta, \hat\theta_t) \right\}$$

其中 $\mathcal{D}_{\psi_t}$ 为 Bregman 散度，正则化器选为局部范数：

$$\psi_t(\theta) = \frac{1}{2}\|\theta\|_{V_t}^2, \quad V_t = \lambda I + \frac{1}{\alpha}\sum_{s=1}^{t}\frac{X_s X_s^\top}{\sigma_s^2}$$

等价地，每轮执行一步投影梯度下降：

$$\tilde\theta_{t+1} = \hat\theta_t - V_t^{-1}\nabla\ell_t(\hat\theta_t), \quad \hat\theta_{t+1} = \Pi_\Theta(\tilde\theta_{t+1})$$

$V_t^{-1}$ 通过 Sherman-Morrison-Woodbury 公式递推更新，每轮仅 $\mathcal{O}(d^2)$。

### 参数设置的递归设计

归一化因子 $\sigma_t$ 和鲁棒化参数 $\tau_t$ 的设置是关键：

- $\sigma_t = \max\!\left\{\nu_t,\; \sigma_{\min},\; \sqrt{\frac{2\beta_{t-1}}{\tau_0\sqrt{\alpha}\, t^{\frac{1-\varepsilon}{2(1+\varepsilon)}}}}\|X_t\|_{V_{t-1}^{-1}}\right\}$
- $\tau_t = \tau_0 \cdot \frac{\sqrt{1+w_t^2}}{w_t} \cdot t^{\frac{1-\varepsilon}{2(1+\varepsilon)}}$

其中 $w_t = \frac{1}{\sqrt{\alpha}}\|X_t/\sigma_t\|_{V_{t-1}^{-1}}$，$\beta_t$ 为置信界半径。该递归设计确保去噪后数据落在 Huber 损失的二次区域内，从而保证 Hessian 下界。

### UCB 选臂策略

$$X_t = \arg\max_{\mathbf{x} \in \mathcal{X}_t}\left\{\langle \mathbf{x}, \hat\theta_t \rangle + \beta_{t-1}\|\mathbf{x}\|_{V_{t-1}^{-1}}\right\}$$

## 理论结果

### 主定理（Theorem 1）：依赖实例的 regret 界

$$\text{Reg}_T \leq \widetilde{\mathcal{O}}\!\left(d\, T^{\frac{1-\varepsilon}{2(1+\varepsilon)}}\sqrt{\sum_{t=1}^T \nu_t^2} + d\, T^{\frac{1-\varepsilon}{2(1+\varepsilon)}}\right)$$

### 推论（Corollary 1）：统一矩界下的最优 regret

当仅知全局矩界 $\nu$ 时：$\text{Reg}_T \leq \widetilde{\mathcal{O}}(d\, T^{1/(1+\varepsilon)})$，匹配下界 $\Omega(d\, T^{1/(1+\varepsilon)})$。

### 方法对比总结

| 方法 | Regret | 每轮计算 | 额外假设 |
|------|--------|---------|---------|
| MENU (MOM) | $\widetilde{\mathcal{O}}(dT^{1/(1+\varepsilon)})$ | $\mathcal{O}(\log T)$ | 固定臂集 + 重复拉取 |
| TOFU (截断) | $\widetilde{\mathcal{O}}(dT^{1/(1+\varepsilon)})$ | $\mathcal{O}(t)$ | 绝对矩有界 |
| HEAVY-OFUL (Huber) | 依赖实例，最优 | $\mathcal{O}(t\log T)$ | 无 |
| **Hvt-UCB (本文)** | **依赖实例，最优** | $\mathcal{O}(1)$ | **无** |

### 估计误差分解（证明核心思路）

通过 OMD 分析将估计误差分解为三项：

1. **泛化间隙项**：$\sum_s \langle \nabla\tilde\ell_s - \nabla\ell_s, \hat\theta_s - \theta_* \rangle$（噪声导致的偏差，用自归一化集中不等式控制）
2. **稳定性项**：$\sum_s \|\nabla\ell_s(\hat\theta_s)\|_{V_s^{-1}}^2$（单遍更新引入的误差，通过 potential lemma 控制）
3. **负项**：$-\frac{1}{2}\sum_s \|\hat\theta_s - \theta_*\|_{X_sX_s^\top/\sigma_s^2}^2$（来自损失的二次性，用于抵消前两项的正性部分）

重尾情况下的关键难点在于：Huber 损失的非二次区域使得泛化间隙和稳定性项的处理更加复杂，需要精心设计归一化因子使数据以高概率落在二次区域。

## 亮点与洞察

- **计算效率质的飞跃**：从 $\mathcal{O}(t\log T)$ 到 $\mathcal{O}(1)$，且无需存储历史数据，对大规模场景有实际意义
- **理论无损**：计算加速不牺牲 regret 质量，保持依赖实例的最优界
- **无额外假设**：不需要固定臂集、重复拉取、绝对矩有界等限制性条件
- **递归参数设计的精巧性**：将上一轮置信界 $\beta_{t-1}$ 嵌入当前轮 $\sigma_t$ 的设置，比 HEAVY-OFUL 更简洁
- **OMD 框架的新应用**：将原本用于对抗性在线学习 regret 最小化的 OMD 适配到随机 bandit 的参数估计，建立了参数间隙与损失间隙的桥梁
- **当 $\varepsilon=1$（有限方差）时**：恢复经典 $\widetilde{\mathcal{O}}(d\sqrt{T})$ 界，统一了轻尾和重尾场景

## 局限与展望

- **纯理论工作**：缺少数值实验验证，无法评估实际性能差异
- **需要已知矩参数**：算法需输入每轮矩界 $\nu_t$（或全局界 $\nu$）和矩阶 $\varepsilon$，实际场景中可能难以获取
- **投影步骤**：虽然每轮 $\mathcal{O}(1)$ w.r.t. $t,T$，但隐含的 $\mathcal{O}(d^3)$ 投影开销在高维场景下仍可能显著
- **仅限线性 bandit**：尚未扩展到广义线性 bandit 或强化学习等更复杂场景（论文在讨论部分提到这是 future work）
- **对数因子间隙**：regret 界与下界在对数因子上仍有差距

## 相关工作与启发

- **HEAVY-OFUL**（Huang et al., 2023）：本文最直接的基线，同样基于自适应 Huber 回归但采用全批量求解
- **OL2M**（Zhang et al., 2016）：首个 logistic bandit 单遍算法，使用 Online Newton Step，但依赖全局强凸性，不适用于 Huber 损失
- **GLOC**（Jun et al., 2017）：基于 online-to-confidence-set 转换的单遍广义线性 bandit 算法
- **OFUL**（Abbasi-Yadkori et al., 2011）：经典 sub-Gaussian 线性 bandit 算法，最小二乘天然支持单遍更新
- **梯度变化在线学习**（Zhao et al., 2020b, 2024）：提供了消除 OMD 稳定性项中正项的技术灵感

## 评分

- 新颖性: ⭐⭐⭐⭐ — OMD 适配重尾 bandit 参数估计的思路新颖，递归归一化设计精巧
- 实验充分度: ⭐⭐ — 纯理论无实验
- 写作质量: ⭐⭐⭐⭐ — 证明思路清晰，case study 由浅入深
- 价值: ⭐⭐⭐⭐ — 填补了重尾 bandit 计算效率的理论空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](../../NeurIPS2025/others/infrequent_exploration_in_linear_bandits.md)
- [\[ACL 2025\] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](../../ACL2025/others/one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](../../ICML2026/others/a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[ICML 2025\] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
