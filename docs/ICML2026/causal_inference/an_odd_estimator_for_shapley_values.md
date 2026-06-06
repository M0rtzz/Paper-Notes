---
title: >-
  [论文解读] An Odd Estimator for Shapley Values
description: >-
  [ICML2026][因果推理][Shapley值] 这篇论文证明 Shapley value 只依赖集合函数的 odd component，并据此提出 OddSHAP：用配对采样隔离 odd 信号、用 GBT 筛选高阶 odd Fourier 交互、再做稀疏 odd 回归…
tags:
  - "ICML2026"
  - "因果推理"
  - "Shapley值"
  - "特征归因"
  - "OddSHAP"
  - "配对采样"
  - "Fourier回归"
---

# An Odd Estimator for Shapley Values

**会议**: ICML2026  
**arXiv**: [2602.01399](https://arxiv.org/abs/2602.01399)  
**代码**: https://github.com/FFmgll/oddshap  
**领域**: 可解释性  
**关键词**: Shapley值、特征归因、OddSHAP、配对采样、Fourier回归  

## 一句话总结
这篇论文证明 Shapley value 只依赖集合函数的 odd component，并据此提出 OddSHAP：用配对采样隔离 odd 信号、用 GBT 筛选高阶 odd Fourier 交互、再做稀疏 odd 回归，在中高维解释任务上显著优于灵活预算 Shapley 估计器。

## 研究背景与动机
**领域现状**：Shapley value 是机器学习解释中最常用的特征归因框架之一，它把模型预测视为集合函数 $f:2^{[d]}\to\mathbb{R}$，为每个特征分配平均边际贡献。由于精确计算需要遍历指数级 coalitions，实际方法通常用采样或 surrogate regression 近似，例如 KernelSHAP、LeverageSHAP、Permutation Sampling、SVARM、MSR、PolySHAP 和各类 proxy-based estimator。

**现有痛点**：很多先进估计器都使用 paired sampling，即每采样一个 coalition $S$，同时采样补集 $S^c$。这个技巧经验上有效，但为什么有效并不完全清楚。同时，高阶 polynomial/surrogate estimator 表达力更强，却面临组合爆炸：候选交互项随阶数迅速增长，预算有限时很难同时保持准确和稳定。

**核心矛盾**：Shapley value 只关心能影响边际贡献的函数成分，但传统回归 estimator 往往同时拟合与 Shapley 无关的部分。若估计器把采样预算浪费在 irrelevant even component 或大量低影响交互上，就会增加方差和计算成本。

**本文目标**：作者希望给 paired sampling 一个严格理论解释，并基于这个解释设计一个预算灵活的 Shapley estimator：既能利用高阶交互提升精度，又不必拟合所有高阶项。

**切入角度**：论文从集合函数的 odd/even 分解出发。若定义 $f_{odd}(S)=\frac12(f(S)-f(S^c))$，则 Shapley value 满足 $\phi_i(f)=\phi_i(f_{odd})$。因此估计器可以只拟合 odd component，而把 even component 完全丢弃。

**核心 idea**：把 Shapley 估计从“拟合整个 value function”改成“只拟合 odd Fourier 子空间中对 Shapley 有贡献的稀疏交互”。

## 方法详解
OddSHAP 的理论基础分两步。第一步证明 paired sampling 会把 weighted least squares 目标分解成互不干扰的 odd 与 even 两部分，因此只要目标是 Shapley value，even 部分可以忽略。第二步换用 Fourier basis，因为 Fourier 基函数 $\chi_T(S)=(-1)^{|S\cap T|}$ 天然按 $|T|$ 的奇偶性分成 odd/even：$|T|$ 为奇数就是 odd term，偶数就是 even term。这使得“只拟合 odd 交互”在算法上变得直接。

### 整体框架
输入是黑盒 value function $f$、采样预算 $m$ 和 regression variable factor $\eta$。算法首先用 paired sampling 得到 coalition 集合；然后拟合一个 gradient boosted tree (GBT) 代理模型，用它筛选出幅度最大的 odd Fourier interactions；最后在 $T_{\leq1}\cup T_{odd}$ 支持上解带边界约束的加权最小二乘问题，得到 odd polynomial approximation $\hat f_{odd}$，并从 Fourier 系数直接计算 Shapley values。

如果预算太低，连线性项都无法稳定回归，即 $m<d\eta$，算法退回到 GBT 的 TreeSHAP 输出。否则，候选高阶 odd 交互数量设为 $|T_{odd}|=\lceil m/\eta\rceil-d$，使回归变量数随预算线性增长，而不是随特征数和阶数组合爆炸。

### 关键设计
1. **Odd component 理论判据**:

    - 功能：证明 Shapley value 的有效信号只在集合函数的 odd 部分中。
    - 核心思路：将任意集合函数分解为 $f=f_{odd}+f_{even}$，其中 odd 部分满足 $f_{odd}(S)=-f_{odd}(S^c)$，even 部分满足 $f_{even}(S)=f_{even}(S^c)$。论文给出观察 $\phi_i(f)=\phi_i(f_{odd})$，因此 even component 对所有特征的 Shapley value 贡献为 0。
    - 设计动机：这直接解释了为什么 paired sampling 有用：它不是简单降方差技巧，而是在估计目标中把与 Shapley 无关的 even 成分正交出去。

2. **Fourier odd regression**:

    - 功能：让“只拟合 odd 项”在基函数层面可执行。
    - 核心思路：在 Fourier basis 中，$\chi_T$ 是否 odd 只由 $|T|$ 是否为奇数决定。OddSHAP 只保留线性项和筛选出的 odd 高阶交互，并通过严格边界约束保证估计的 Shapley values 满足 efficiency，即总和等于 $f([d])-f(\emptyset)$。从系数计算归因的公式为 $\phi_i(\hat f_{odd})=-2\sum_{T\ni i, |T|\ odd}\beta_T/|T|$。
    - 设计动机：KernelSHAP/LeverageSHAP 常用的 unanimity basis 不会干净地区分奇偶成分；Fourier basis 提供了天然的结构分离。

3. **GBT 交互筛选 + 预算自适应支持集**:

    - 功能：在不枚举所有高阶交互的情况下，保留对 value function 最有影响的 odd interactions。
    - 核心思路：算法先用同一批样本拟合 GBT proxy，再用 ProxySPEX 风格方法提取绝对幅度最大的 odd Fourier coefficients。回归支持大小由 $\eta$ 控制，预算越大可纳入越多交互；预算不足时退回 TreeSHAP，避免欠定回归。
    - 设计动机：机器学习 value functions 往往只有少数重要 Fourier interactions。先筛选再回归可以在表达力和统计稳定性之间折中。

### 损失函数 / 训练策略
OddSHAP 的核心优化是带 Shapley kernel 权重的加权最小二乘回归，但目标只在 odd Fourier 支持上求解。通过 paired samples 预计算 $f_{odd}(S)=\frac12(f(S)-f(S^c))$ 后，可以丢弃补集行，只用 $m/2$ 个代表样本拟合 odd target。回归还显式处理边界约束，而不是像部分 KernelSHAP 实现那样用伪无穷权重近似。

## 实验关键数据

### 主实验
实验在 8 个 value functions 上评估 30 个随机预测实例的 Shapley 近似，覆盖语言、图像、表格和合成函数。评价指标是相对 ground-truth Shapley values 的 MSE 中位数和 IQR。

| 数据集 / 函数 | 维度 | 领域 | 本文 | 之前SOTA / 基线 | 提升 |
|---------------|------|------|------|-----------------|------|
| DistilBERT | 14 | language | 与 RegressionMSR 等最佳灵活预算方法相当 | RegressionMSR / LeverageSHAP | 低维下无明显劣势 |
| ViT16 | 16 | image | 与最佳灵活预算方法相当，并优于 FFD corrected 多数设置 | RegressionMSR / FFD variants | 深度模型中高阶交互更活跃 |
| Cancer | 30 | tabular | 中高预算下优于所有 flexible-budget baselines | LeverageSHAP / MSR / SVARM / FourierSHAP | 交互建模带来最高可达 62x MSE 降低 |
| CG60 / IL60 | 60 | synthetic | 预算足够时明显领先灵活预算 baseline | MSR / FourierSHAP / RegressionMSR | 在高维交互函数上优势更明显 |
| NHANES | 79 | tabular | 中高预算下优于灵活预算 baseline | TreeSHAP ground truth 对比 | 随维度升高保持可用 |
| Crime | 101 | tabular | 在运行时-MSE 曲线上保持竞争力 | LeverageSHAP / FFD-RD / Proxy | 灵活预算比固定 $O(d^2)$ 设计更可扩展 |

### 消融实验
论文的消融直接验证了 OddSHAP 的三个核心选择：交互数量、paired sampling 和只保留 odd interactions。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $\eta=10$，约 1000 个交互，10000 samples | 所有 value functions 至少 6x MSE 降低，Cancer 最高 62x | 适量 odd 高阶交互显著优于 interaction-free LeverageSHAP |
| $\eta\in\{2,5,10,50\}$ | 交互过多后 MSE 反弹 | 表达力增加会带来过拟合，支持集不应随预算无限扩张 |
| Paired + Odd interactions | 作为归一化最佳配置 | 直接隔离 odd component，预算集中在对 Shapley 有贡献的项上 |
| Paired + All interactions | MSE 略差且更慢 | even terms 会数学上抵消，却占用交互预算和计算时间 |
| Non-paired sampling | 整体弱于 paired sampling | 没有 paired structure 时 odd/even 分离不干净，估计更不稳定 |
| FFD-RD fixed-budget | 树模型上强，深度模型上退化 | 依赖高阶交互截断假设，且 $O(d^2)$ 严格样本需求在高维不灵活 |

### 关键发现
- paired sampling 的价值被严格解释为 even-odd separation，而不是单纯经验降方差。
- OddSHAP 在低维任务中不牺牲性能，在中高维任务中凭借稀疏 odd interaction 建模明显优于灵活预算 baseline。
- even interactions 对 Shapley value 没有贡献；在 paired sampling 下继续拟合 even terms 只会分走预算并增加运行时间。

## 亮点与洞察
- 论文把一个常用工程技巧提升为清晰理论：paired sampling 正是在估计 odd component。这个解释非常简洁，也能指导新 estimator 设计。
- Fourier basis 的选择很漂亮。它不是为了数学形式好看，而是因为 odd/even 可以直接由 interaction order 判断，使算法能精确丢掉无关子空间。
- GBT proxy 的角色定位合理：它不直接作为最终解释器，而是帮助找到稀疏高影响交互，再由受约束回归保证 Shapley consistency。
- 这篇论文对解释方法有一个重要提醒：估计 value function 本身不等于估计归因所需的全部信息。只拟合归因相关子空间，可能比拟合完整函数更高效。

## 局限与展望
- OddSHAP 的回归阶段随选中交互数二次增长；若交互数继续绑定采样预算，整体开销可随 $m$ 近似三次增长。作者建议在大预算下给交互数设上限并与 $m$ 解耦。
- paired sampling 会把 $m$ 次查询压成 $m/2$ 个独立行，减少独立 subset coverage，可能增加方差和 mutual coherence；因此它不保证在所有函数上都优于 non-paired sampling。
- 交互筛选依赖 GBT proxy。若 proxy 无法捕捉真实 value function 的重要 Fourier interactions，OddSHAP 可能漏掉关键高阶项。
- 固定 $\eta=10$ 在实验中稳健，但不同领域、维度和 value function 评估成本下，如何自适应选择 $\eta$ 仍值得进一步研究。

## 相关工作与启发
- **vs KernelSHAP / LeverageSHAP**: 它们本质上做低阶/线性 surrogate regression；OddSHAP 保留一致性的同时加入筛选出的 odd 高阶交互，因此能在复杂 value functions 上降低偏差。
- **vs PolySHAP**: PolySHAP 扩展到多项式回归，但候选项存在组合爆炸；OddSHAP 用 Fourier odd 子空间和 GBT screening 控制支持大小。
- **vs RegressionMSR / ProxySPEX**: Proxy 方法利用学习器逼近 value function，OddSHAP 则把 proxy 用作交互筛选器，最终仍通过严格回归和边界约束计算 Shapley。
- **vs FFD-RD**: FFD 利用固定组合设计和高阶截断假设，在树模型上很强；OddSHAP 更灵活，尤其适合深度模型或高阶交互活跃的函数。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ odd component 判据和 OddSHAP 设计都很有洞察力，理论与算法结合紧密。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 8 个 value functions、多类 baseline、runtime、交互稀疏和 paired sampling 消融，支撑充分。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚，但 Fourier/Shapley 理论密度较高，部分读者需要背景知识。
- 价值: ⭐⭐⭐⭐⭐ 对可解释性中的 Shapley 估计、采样设计和高阶交互归因都有直接价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICML 2026\] Causal-JEPA: Learning World Models through Object-Level Latent Masking](causal-jepa_learning_world_models_through_object-level_latent_masking.md)
- [\[ICML 2026\] Investigating Memory in Model-Free RL with POPGym Arcade](investigating_memory_in_model-free_rl_with_popgym_arcade.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)

</div>

<!-- RELATED:END -->
