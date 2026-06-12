---
title: >-
  [论文解读] Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach
description: >-
  [ICLR 2026 Oral][LLM安全][machine unlearning] 提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架…
tags:
  - "ICLR 2026 Oral"
  - "LLM安全"
  - "machine unlearning"
  - "Gaussian certifiability"
  - "hypothesis testing"
  - "high-dimensional statistics"
  - "Newton method"
  - "privacy"
---

# Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach

**会议**: ICLR 2026 Oral  
**arXiv**: [2510.13094](https://arxiv.org/abs/2510.13094)  
**代码**: [匿名仓库](https://anonymous.4open.science/r/unlearning-E14D)  
**领域**: AI Safety / 机器遗忘 / 高维统计  
**关键词**: machine unlearning, Gaussian certifiability, hypothesis testing, high-dimensional statistics, Newton method, privacy

## 一句话总结

提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架，严格证明在高维比例体系 ($p \sim n$) 下单步 Newton 更新 + 校准高斯噪声即可同时满足隐私 (GPAR) 和精度 (GED→0) 要求，推翻了 Zou et al. (2025) "至少需两步 Newton" 的结论，并从理论上揭示旧 $\varepsilon$-certifiability 与噪声添加机制不兼容的根本原因。

## 研究背景与动机

**数据遗忘的法律驱动**：GDPR、CCPA 等法规要求"被遗忘权"，模型须能高效删除特定用户数据的统计影响，完全重训练代价过高。

**主流 Newton 遗忘方法**：Guo et al. (2020)、Sekhari et al. (2021) 证明在低维 ($p \ll n$) 下单步 Newton + 噪声即可保证隐私与精度，但其证明依赖 per-example loss 的 $\Omega(1)$ 强凸 + $O(1)$ 光滑假设。

**低维假设在高维下崩溃**：以最简单的 Ridge 回归为例，当 $p \sim n$ 时，要求 $\|x_i\|_2 \sim 1$（保证 $O(1)$ 光滑）会导致 per-example loss 的最小特征值降至 $2\lambda/n$，彻底破坏 $\Omega(1)$ 强凸性，已有框架完全失效。

**Zou et al. (2025) 的高维尝试**：放松了部分优化假设，但采用 $(\phi,\varepsilon)$-PAR certifiability，结论是即使删除单个数据点也至少需要两步 Newton 迭代才能同时保证隐私和精度。

**旧定义的本质缺陷**：$\varepsilon$-certifiability 与噪声添加策略不兼容——在高维中需要注入不成比例的大噪声来满足隐私条件，破坏了模型精度。

**核心洞察**：在高维中，广泛的各向同性对数凹噪声机制在行为上收敛到高斯机制 (Dong et al., 2021)，因此高斯 trade-off 曲线是高维隐私证明的规范选择，据此重新定义 certifiability 可解决上述矛盾。

## 方法详解

### 整体框架

给定训练数据 $\mathcal{D}_n$、已训练模型 $\hat{\beta} = A(\mathcal{D}_n)$ 与待删除子集 $\mathcal{D}_\mathcal{M}$，本文的遗忘算法只做两件事：先以 $\hat{\beta}$ 为初值对去掉 $\mathcal{M}$ 后的目标 $L_{\setminus\mathcal{M}}$ 走一步 Newton 更新，得到近似解 $\hat{\beta}^{(1)}_{\setminus\mathcal{M}} = \hat{\beta} - G(L_{\setminus\mathcal{M}})^{-1}(\hat{\beta})\, \nabla L_{\setminus\mathcal{M}}(\hat{\beta})$；再叠加一份校准过的高斯噪声 $\tilde{\beta}_{\setminus\mathcal{M}} = \hat{\beta}^{(1)}_{\setminus\mathcal{M}} + b,\ b \sim \mathcal{N}(0, \tfrac{r^2}{\varepsilon^2} I_p)$。论文真正的贡献并不在这两步算法本身，而是换了一把度量隐私的尺子——高斯 trade-off 函数，从而证明在 $p \sim n$ 的高维比例体系下这单步更新就已足够。

### 关键设计

**1. $(\phi,\varepsilon)$-Gaussian certifiability (GPAR)：把隐私保证重写成假设检验问题。** 旧框架要求"遗忘后模型"与"重训练模型"在 $\varepsilon$-certifiability 下逐参数接近，而在高维中这会逼迫算法注入不成比例的大噪声、人为破坏精度。本文转而把隐私形式化为一场假设检验：对手观察遗忘后的输出，试图区分"从完整数据重训练后再加噪 ($\mathcal{P}_{re}$)"和"从原模型遗忘后加噪 ($\mathcal{P}_{un}$)"。其最优功效由 trade-off 函数 $T(P,Q)(\alpha) = \inf_{\phi}\{\beta_\phi : \alpha_\phi \leq \alpha\}$ 刻画，并选高斯曲线 $f_{G,\varepsilon}(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - \varepsilon)$ 作为基准；若 $T(\mathcal{P}_{re}, \mathcal{P}_{un})(\alpha) \geq f_{G,\varepsilon}(\alpha)$ 以概率 $\geq 1-\phi$ 成立，就称算法满足 $(\phi,\varepsilon)$-GPAR。之所以高斯曲线是高维下的规范选择：对任意 $p$ 维高斯向量，trade-off 只取决于均值差的 $\ell_2$ 范数除以标准差，与维度 $p$ 无关；广泛的各向同性对数凹噪声在 $p \to \infty$ 时按差分隐私的中心极限定理收敛到高斯行为；而在 Blackwell 排序意义下高斯 trade-off 曲线又是捕获高斯机制最紧的方式，先前所有 certifiability 概念（$\varepsilon$-$\delta$、Rényi 等）在高斯机制下都只是次优。

**2. 广义误差散度 (GED)：在高维下稳定地度量精度损失。** 精度这一侧不能再沿用 Sekhari et al. (2021) 那种相对真实总体风险最小化器的 excess risk——它在 $p \sim n$ 时行为不稳。本文改用 GED，直接比较遗忘模型与理想重训练模型在一个新样本上的预测损失差：$\text{GED}_\ell(A, \bar{A}; \mathcal{M}, \mathcal{D}_n) = \mathbb{E}\big[\,|\ell(y_0 \mid x_0^\top A(\mathcal{D}_{\setminus\mathcal{M}})) - \ell(y_0 \mid x_0^\top \tilde{\beta}_{\setminus\mathcal{M}})| \mid \mathcal{D}_n\,\big]$。这个量绕开了对真实最优解的依赖，因而在高维比例体系下行为更稳，能干净地作为"遗忘是否伤精度"的判据。

**3. 放松的优化假设：让证明覆盖真正高维的损失。** 旧证明依赖 per-example loss $f(\beta, z_i)$ 同时满足 $\Omega(1)$ 强凸与 $O(1)$ 光滑，而 Ridge 这类模型在 $p \sim n$ 时这对要求会自相矛盾。本文把假设挪到整体目标上：损失 $\ell$ 凸、正则化 $r$ 取 $\nu$-强凸且 $\nu = \Theta(1)$，$\ell$ 与 $r$ 三阶可微且多项式增长，特征 $x_i$ 亚高斯、响应 $y_i$ 有亚多项对数矩。这套更宽松的条件不再要求逐样本强凸，因而一举覆盖 Ridge 回归、Logistic 回归、Poisson 回归等广义线性模型。

**4. 单步遗忘的隐私-精度双保证：把噪声标定到刚好够用。** 在上述假设下，论文把噪声方差精确标定为 $r^2/\varepsilon^2$，其中 $r = C_1(n)\sqrt{C_2(n)\, m^3/(2\lambda\nu n)}$。隐私侧（定理 2）证明单步 Newton 遗忘满足 $(\phi_n, \varepsilon)$-GPAR 且失败概率 $\phi_n \to 0$；精度侧（定理 3）证明 $\text{GED}(\tilde{\beta}_{\setminus\mathcal{M}}, \hat{\beta}_{\setminus\mathcal{M}}) = O_p\big(\tfrac{m^2 \cdot \text{polylog}(n)}{\sqrt{n}}\big)$，当删除规模 $m = o(n^{1/4-\alpha})$ 时 GED $\to 0$。两条定理合起来意味着：只要用高斯而非 Laplace 噪声，一步 Newton 就能同时删掉 $m$ 个数据点并保住精度——这正是推翻"至少需两步"结论的关键，差异完全来自 certifiability 定义而非迭代次数。

## 实验关键数据

### 主实验：GED 随维度变化 (Logistic 回归, $n=p$, $\varepsilon=0.75$, $\lambda=0.5$)

| 删除数 $m$ | 噪声类型 | log(GED) vs log(p) 斜率 | GED 行为 |
|:-----------:|:--------:|:----------------------:|:--------:|
| 1 | Laplace (Zou) | 0.03 | 不衰减 |
| 1 | Gaussian (本文) | -0.47 | $\sim p^{-0.5}$ 衰减 |
| 5 | Laplace (Zou) | -0.03 | 不衰减 |
| 5 | Gaussian (本文) | -0.54 | $\sim p^{-0.5}$ 衰减 |
| 10 | Laplace (Zou) | -0.01 | 不衰减 |
| 10 | Gaussian (本文) | -0.51 | $\sim p^{-0.5}$ 衰减 |

**核心结论**：Laplace 噪声（Zou et al. 框架要求）导致一步 Newton 的 GED 不衰减，必须增加到两步；而 Gaussian 噪声（GPAR 框架）的 GED 以 $p^{-0.5}$ 稳定衰减，一步就够。

### 消融实验：GED 随 $\varepsilon$ 和 $m$ 变化 ($n=p=1255$)

| 实验维度 | 观察结果 |
|:--------:|:--------:|
| $\varepsilon$ 增大 | Gaussian GED 单调下降趋近重训练；Laplace GED 一直远高于 Gaussian |
| $m$ 增大 ($5 \to 50$) | 两者 GED 均增长，但 Laplace 始终显著高于 Gaussian |
| $m$ vs GED 斜率 (Gaussian) | $\sim 1.4$（实际优于理论 $m^{1.5}$ 的 bound） |
| $m$ vs GED 斜率 (Laplace) | $\sim 0.24$（增长慢但绝对值高，因噪声量过大） |

### 关键发现

1. **一步 vs 两步的分歧根源确认**：理论预测与实验完全吻合——差异来自 certifiability 定义而非算法本身
2. **高斯噪声的维度优势**：$p$ 增大时 Gaussian 方案的精度不断改善，Laplace 方案停滞
3. **多点同时删除可行**：当 $m = o(n^{1/4})$ 时，同时删除多用户数据仍可保持精度

## 亮点与洞察

- **概念突破**：问题不在于 Newton 步数不够，而在于 certifiability 概念选择不当——旧定义 ($\varepsilon$-certifiability) 要求注入不成比例的噪声，人为恶化了精度，导致了错误的"需要两步"结论
- **理论统一**：借助假设检验的 trade-off 函数框架，将差分隐私文献中的 Gaussian DP (Dong et al., 2022) 引入机器遗忘，建立了两个领域间的优雅桥梁
- **实用性**：单步 Newton 的计算量仅需一次 Hessian 逆乘梯度，比两步方案节省约一半计算
- **高维中的维度福利**：高维并非只是挑战——CLT 效应使得高斯 certifiability 成为自然且最优的选择，维度越高 GPAR 越精确

## 局限性

1. **凸/强凸假设**：理论仅适用于凸损失 + 强凸正则化的 RERM，深度学习的非凸优化场景未覆盖
2. **GLM 数据假设**：需要特征亚高斯、响应与特征通过 GLM 关联，真实高维数据可能不满足
3. **Hessian 计算开销**：虽然只需一步 Newton，但在大规模模型中计算/存储 Hessian 逆仍然昂贵
4. **实验规模有限**：仿真验证在 $p \leq 5000$ 的 Logistic 回归上进行，未在实际深度模型或 LLM 上测试
5. **多点删除上界偏松**：实验观察到 GED 对 $m$ 的增长斜率 ($\sim 1.4$) 好于理论 bound ($m^{1.5}$)，理论或可进一步收紧

## 相关工作

- **低维机器遗忘**：Guo et al. (2020)、Sekhari et al. (2021) 在 $p \ll n$ 下证明单步 Newton + 噪声足够，但假设在 $p \sim n$ 下失效
- **高维遗忘**：Zou et al. (2025) 首次研究 $p \sim n$ 比例体系，但采用 $\varepsilon$-certifiability + Laplace 噪声，结论是需要至少两步
- **Gaussian DP**：Dong et al. (2022) 提出 Gaussian 差分隐私 ($f$-DP)，本文将其引入遗忘领域
- **精确遗忘**：Bourtoule et al. (2021)、Cao & Yang (2015) 追求精确等价重训练，计算开销大
- **梯度下降遗忘**：Neel et al. (2021)、Allouah et al. (2025) 分析基于 GD/SGD 的近似遗忘方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 新的 certifiability 框架从根本上改变了高维遗忘的理论版图
- 实验充分度: ⭐⭐⭐ 仿真验证充分支撑理论，但缺乏真实规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，动机阐述透彻，对比分析精准
- 价值: ⭐⭐⭐⭐⭐ 对机器遗忘理论有范式级推动，ICLR Oral 实至名归

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Efficient Adversarial Attacks on High-dimensional Offline Bandits](efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)
- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[NeurIPS 2025\] ModHiFi: Identifying High Fidelity Predictive Components for Model Modification](../../NeurIPS2025/llm_safety/modhifi_identifying_high_fidelity_predictive_components_for_model_modification.md)
- [\[ICML 2026\] Optimizing Token Choice for Code Watermarking: An RL Approach](../../ICML2026/llm_safety/optimizing_token_choice_for_code_watermarking_an_rl_approach.md)
- [\[ACL 2025\] Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](../../ACL2025/llm_safety/are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)

</div>

<!-- RELATED:END -->
