---
title: >-
  [论文解读] Barriers to Counterfactual Credit Attribution for Autoregressive Models
description: >-
  [ICML 2026][可解释性][反事实信用归因] 本文形式化研究生成式模型在 RAG/in-context 部署时的"反事实信用归因（CCA）"问题，证明两条令人惊讶的负面结果：(1) 即便底层 next-token 预测器是 (0,0)-CCA…
tags:
  - "ICML 2026"
  - "可解释性"
  - "反事实信用归因"
  - "差分隐私"
  - "自回归模型"
  - "RAG"
  - "不可行性下界"
---

# Barriers to Counterfactual Credit Attribution for Autoregressive Models

**会议**: ICML 2026  
**arXiv**: [2605.01425](https://arxiv.org/abs/2605.01425)  
**代码**: 暂未公开（理论论文）  
**领域**: AI 安全 / 理论 / 生成式模型版权  
**关键词**: 反事实信用归因, 差分隐私, 自回归模型, RAG, 不可行性下界

## 一句话总结
本文形式化研究生成式模型在 RAG/in-context 部署时的"反事实信用归因（CCA）"问题，证明两条令人惊讶的负面结果：(1) 即便底层 next-token 预测器是 (0,0)-CCA，自回归 rollout 也并非 CCA——CCA 不像 DP 那样在自回归下天然 compose；(2) 对一个已部署的非归因模型做 black-box "CCA retrofitting" 至少需要在输出长度 $\ell$ 上指数级查询次数。

## 研究背景与动机

**领域现状**：生成式 AI 把"创作者要给前人作品署名"这件事彻底搅乱——RAG/in-context learning 让最终输出和具体来源之间的因果链被模型黑箱挡住。学术写作里这是学术不端，法律层面（copyright）更涉及许可与商用边界。Livni-Moran-Nissim-Pabbaraju (2024) 提出 **Counterfactual Credit Attribution (CCA)** 作为差分隐私（DP）的松弛：算法 $\tilde A(S)$ 同时输出主结果 $y$ 和被归功的子集 $C\subseteq S$；要求"未被归功的输入"在分布上接近"该输入被完全移除"的分布，即 $\tilde A^{-i}(S)\approx_{\varepsilon,\delta}\tilde A_{-i}(S)$。原论文只研究了 PAC 学习场景下的 CCA，留下生成式模型作为 open problem。

**现有痛点**：把 CCA 直接搬到 LLM 上，最自然的两条工程路径——(a) 设计一个 CCA next-token predictor 然后自回归运行，靠"自动合成"得到整段输出的 CCA；(b) 拿一个不带归因的现成 LLM，写一层"包装"自动补上 credit set——它们到底行不行？没人系统回答过。这与 DP-LLM 的发展路径形成鲜明对比：DP 在 token 级合成几乎是 free lunch（Majmudar 2022、Amin 2024）。

**核心矛盾**：DP 在 sequential composition 下有干净的合成定理（$k$ 次合成最坏 $k\varepsilon$）。CCA 长得很像 DP，但它的核心是"未被归功"这一**条件分布**，这种条件性会被自回归乘法链放大；CCA 看似 DP 实则不 compose。retrofitting 这条路同样如此：模型一旦决定输出，再倒推"该 credit 谁"等价于估计每条 trajectory 上"sensitive 数据贡献了多少 mass"，而对一个黑箱模型这是指数级困难。

**本文目标**：把 CCA 部署到 deployment-time 数据（RAG 数据库、in-context 例子）上，系统检验上面两条工程直觉，给出严谨的(不)可行性证明。

**切入角度**：作者用"counterexample first, lower bound second"的策略——先用一个 2-token、单文档的小玩具构造反例（极端紧凑、易复述），再把它一般化成参数化下界定理；retrofitting 部分则构造一族"几乎相同、仅有微弱 1-bit 偏置"的硬模型，把寻找隐藏标识符 $\mathbf{z}$ 的难度迁移给 retrofitter。

**核心 idea**：把 CCA 从"看起来像 DP 松弛"严格区分开——证明它在自回归 composition 和 black-box retrofitting 两个最自然的工程路径上**都不可行**，为后续 CCA-LLM 研究划清边界。

## 方法详解

### 整体框架
论文分两大主线。**第一线 (§4)**：研究"对 next-token predictor $\tilde M$ 施加 CCA + 自回归 rollout"这条工业风路径，给出反例（Thm 4.2）+ 一般下界（Thm 4.3）。**第二线 (§5)**：研究"对非归因 $M$ 做 black-box 改造"这条 retrofitting 路径，构造一族硬模型 $\{M_\mathbf{z}\}_{\mathbf{z}\in\{0,1\}^\ell}$，证明任何 $\alpha<1/2$ 近似 retrofit 在最坏情况下都要 $\widetilde\Omega(2^\ell)$ 次 oracle 查询（Thm 5.5）。所有证明都建立在严格的 LP 刻画 + 信息论查询下界上。

### 关键设计

1. **CCA 不自回归合成的反例与一般下界（Thm 4.2 & 4.3）**:

    - 功能：证伪"$\tilde M$ 是 $(0,0)$-CCA $\Rightarrow$ rollout $G^{\tilde M}$ 是 $(\varepsilon,\delta)$-CCA"的直觉。
    - 核心思路：取数据集 $\mathcal S=\{s_1\}$、词表 $\mathcal X=\{\mathtt a,\mathtt b\}$，定义一个 $(0,0)$-CCA next-token predictor：空前缀下，$\tilde M(\{s_1\},\lambda)=\tilde M(\emptyset,\lambda)$ 完全相同（以 $p$ 输出 $\mathtt a$、$1-p$ 输出 $\mathtt b$，永不 credit）；前缀 $\mathtt a$ 时只在 $S=\{s_1\}$ 下 1/2 几率 credit、1/2 几率不 credit 直接输出 $\mathtt b$（条件不 credit 的分布与 $S{=}\emptyset$ 时完全相同，所以 token 级 $(0,0)$-CCA 成立）；前缀 $\mathtt b$ 时总 credit 也总输出 $\mathtt a$（条件触发 1.0 项免去 CCA 约束）。然而 rollout 后：在不 credit 条件下 $G^{\tilde M}(S^{-1},\lambda)$ 必然输出 $(\mathtt{ab},\emptyset)$；counterfactual $G^{\tilde M}(S_{-1},\lambda)$ 只以 $p$ 概率得 $(\mathtt{ab},\emptyset)$。当 $p<e^{-\varepsilon}(1-\delta)$ 就破坏 $(\varepsilon,\delta)$-CCA。一般下界（Thm 4.3）将其抽象为 $\varepsilon'\geq\ln\big(\prod_j\Pr[E_j\mid\cdots]/\Pr[s_i\notin C]\big)-|\mathbf x^{-i}|\cdot\varepsilon$，其中 $E_j$ 是第 $j$ 步不 credit 的事件。
    - 设计动机：DP 之所以 compose 在于"未受 $s_i$ 影响的条件分布"在 token 级和序列级是一致量；CCA 的核心要素 $\Pr[s_i\notin C]$ 在乘法链上会**收缩**，让条件分布的 ratio 反而被放大——这就是反直觉的"$\varepsilon\to 0$ 时下界反而变大"的根源（注意 $\varepsilon$ 是内生量，与模型耦合）。反例的极简性也保证了它对几乎所有自然 CCA 设计都形成阻断。

2. **CCA Retrofitting 的硬模型族 $\mathcal M_\ell$（Thm 5.5 的硬实例构造）**:

    - 功能：构造一族 $\{M_\mathbf{z}\}$，让"识别 $\mathbf z$" 既是 retrofitting 的隐含子问题，又对原模型 oracle 是指数难的。
    - 核心思路：$\mathcal X=\{0,1,\bot\}$、$\mathcal S=\{s_1\}$、$\ell\geq 1$、$\gamma\in(0,1)$、$\varepsilon\geq 0$。$M_\mathbf z(S,\mathbf x)$ 在 $|\mathbf x|\leq\ell$ 时几乎处处是 $\mathsf{Bern}(1/2)$，**唯有**当 $S\neq\emptyset$ 且前缀恰好是 $\mathbf z$ 时把 1 的概率偏置成 $\tfrac12+\tfrac{1-e^{-\varepsilon}(1-\gamma)}{2}$；输出长度恒为 $\ell+1$。在 oracle 模型下 $M_\mathbf z$ 与 $M_\emptyset$ 在每个 prompt 上 TV 距离 $\leq 2^{-\ell}$（Remark 5.6.1），找 $\mathbf z$ 等价于在 $\{0,1\}^\ell$ 上做一个 needle-in-haystack 搜索 $\Rightarrow$ Lemma 5.8: $\Omega(2^\ell)$ 查询下界。
    - 设计动机：要给 retrofit 下"算力"下界，必须让信息隐藏在 oracle 极难抽样的位置；这族模型把"区分两个几乎相同分布"的统计难度精确编码成 $\ell$ 位 secret string，且偏置大小可调以保证 $\varepsilon$-CCA 框架下仍有不可约的 $\gamma$ 概率必须 credit。

3. **最优 CCA augmentation 的 LP 刻画（Lemma 5.6）+ 把搜 $\mathbf z$ 归约给 retrofit（Lemma 5.9）**:

    - 功能：让任何"近似最优 CCA augmentation 的 oracle"反向变成一个"$O(\ell\log\ell)$ 查询就能找出 $\mathbf z$"的算法，进而得出 retrofitter 必然要付出 $\widetilde\Omega(2^\ell/\ell\log\ell)$ 次原模型查询。
    - 核心思路：把"最优 CCA augmentation"形式化为 LP：变量是各 $(S,\mathbf x,y,C)$ 上的概率，约束是 augmentation（marginal 与 $G^M$ 一致）+ CCA（不 credit 条件分布与 counterfactual $\varepsilon$-close）；目标是最小化 $\mathbb{E}[f(C)]$。LP 解析解显示：当且仅当前缀 $\mathbf x\sqsubseteq\mathbf z$ 时 $\Pr[\tilde G^*_\mathbf z(S^*,\mathbf x)\text{ credits }s_1]=\gamma$，否则为 0。这个"crediting 概率在 $\mathbf z$ 的前缀树上是常数 $\gamma$、其它地方 0"的尖锐结构等价于"沿前缀树做二分搜索就能定位 $\mathbf z$"——只需 $O(\ell\log\ell/(\gamma-2\alpha)^2)$ 次 retrofit 查询即可解出 $\mathbf z$。
    - 设计动机：这是把"最优解的结构"作为攻击面：如果 retrofit 必须最优（哪怕 $\alpha$-近似），它在每个前缀上的归因概率都会"泄露" $\mathbf z$。换言之，**做得越好的 retrofit 越容易被 invert，攻击者用它就能撬出原模型 oracle 都查不到的 secret**——这是非常优雅的"computational hardness via solution-structure leakage"。

### 损失函数 / 训练策略
本文是纯理论论文，没有训练或损失函数。主要贡献是一系列严谨的（不）可行性证明。

## 实验关键数据

### 主实验
论文不含实证实验；下面用"理论结果"代替主结果表。

| 命题 | 设定 | 结论 |
|------|------|------|
| Thm 4.2 | $\forall\varepsilon\geq 0,\delta<1$ | 存在 $(0,0)$-CCA next-token predictor，其 rollout 非 $(\varepsilon,\delta)$-CCA |
| Thm 4.3 | $\tilde M$ $\varepsilon$-CCA, rollout $\varepsilon'$-CCA | $\varepsilon'\geq\max\big[\ln(\prod_j\Pr[E_j]/\Pr[s_i\notin C])-\|\mathbf{x}^{-i}\|\cdot\varepsilon\big]$ |
| Thm 5.5 | $\alpha<1/2$, $\delta=0$ | $\alpha$-近似 retrofit 在最坏 $M\in\mathcal M_\ell$ 上至少要 $\Omega(2^\ell/\ell\log\ell)$ 次 oracle 查询 |
| Remark 5.6.1 | $\mathcal M_\ell$ | 数据对模型 TV 影响 $\leq 2^{-\ell}$，却仍被要求以 $\gamma>0$ 概率 credit |

### 消融实验
本文以"分情况收紧"代替消融：

| 条件 | 结果 | 解释 |
|------|------|------|
| $\varepsilon\to 0$ | $\varepsilon'$ 下界**变大** | 因 $\varepsilon$ 内生于模型，让 $\tilde M$ 更"完美 CCA"反而把 rollout 不可合成性放得更明显 |
| $\delta=0$ 严格 CCA | retrofit 需 $\Omega(2^\ell)$ | 严格 CCA 完全封死 black-box 高效解 |
| $\delta>0$ 松弛、近似 augmentation | 作者猜想下界仍成立 | Remark 5.5.1 留作 open |
| credit 集合永远等于 $S$ | trivially CCA | 但归因毫无信息量，证明限制 $|C|$ 是 CCA 有意义性的关键 |

### 关键发现
- **CCA 不是 DP 在 sequence 上的天然延拓**：DP 享有 sequential composition 是因为"未受 $s_i$ 影响的边际分布"在 token 链上是同一类；CCA 的"未 credit 条件"在链上会乘性收缩，导致下界 $\varepsilon'$ 不收敛。
- **越优秀的 retrofit 越容易反向攻击**：LP 最优解显式暴露 $\mathbf z$ 的前缀结构，揭示"完美 credit 优化"与"信息隐藏"之间存在本质冲突。
- **Vanishing-impact 也要 credit**：Remark 5.6.1 指出在 $\mathcal M_\ell$ 上数据对输出的影响以 $2^{-\ell}$ 速度消失，但 CCA 定义仍要求以常数概率 $\gamma$ credit——这逼迫我们重新审视 CCA 在"几乎无影响"边界下是否仍是合适定义。
- **Token-级 (0,0)-CCA 实质上没用**：因为它直接落入 Thm 4.3 的最强反例区，意味着"token 级完美归因"是个看起来稳妥实则虚假的目标。

## 亮点与洞察
- "$\varepsilon\to 0$ 反而让下界变大"的反直觉结论极有信息量：它指出"更强的 token 级 CCA"≠"更强的 sequence 级 CCA"，把追求局部完美的工程直觉直接证伪。
- LP-based 最优归因的结构性结果是非常美的"computational vs information"分裂：信息上 retrofit 是良定义的，但解的结构性泄露让它在算力上指数难——这种"良定义但难算"的结果在 ML 安全文献里很有借鉴价值。
- 作者诚实地把"为什么 CCA 可能不是好定义"这一 critique 放进 Remark 5.6.1：vanishing-impact 仍被 mandatory credit 暴露出 CCA 把"任何依赖"和"实质性依赖"混为一谈，提示后续工作需要"影响-感知"的 CCA 变种。

## 局限与展望
- 全部下界都在 deployment-time CCA + black-box 模型 + 严格 $\delta=0$ 设定下；relaxing 到近似 augmentation（仅在 TV 距离 $\leq 2d$ 内贴合原模型）或 $\delta>0$ 是否仍有同等强度下界，作者只给猜想。
- 反例族 $\mathcal M_\ell$ 是高度对抗性的人造构造，不直接反映真实 LLM 的几何/语义结构；现实模型上"平均情况"是否更易 retrofit 是开放问题。
- 文中只考虑"二值 credit"，未触及 Shapley 风格的连续贡献度量；对版权用例 binary credit 可能正合适，但对算分制平台可能不够。
- end-to-end CCA（训练 + 部署联合）作者明确留作 future work，这是真正落地的最大空白。

## 相关工作与启发
- **vs Livni et al. 2024**：原论文在 PAC learning 下证 CCA 算法的存在性（VC 维 + log 因子内可学），本文转到生成式模型的 sequential 设定，结论完全相反——给出强不可行性，划清两个 setting 的鸿沟。
- **vs DP-LLM（Majmudar 2022 / Amin 2024）**：DP 在 next-token + composition 范式下几乎是 free lunch；本文证明同样的范式在 CCA 上不 work，提示研究者必须重新设计 sequence-aware 的 CCA 训练目标。
- **vs Vyas et al. 2023（near access-freeness）**：那条线用 black-box 包装实现一个不同的版权松弛；本文证明 CCA 这条松弛**没有**类似的 black-box 解，把两种松弛的工程友好性彻底分开。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 CCA 从 PAC 学习严格推到生成式模型，并给出两条独立的强不可行性
- 实验充分度: N/A 理论论文，证明完整，反例构造精巧
- 写作质量: ⭐⭐⭐⭐ 反例 → 一般下界 → LP 刻画 → 归约 → 终极下界，推理链条非常清晰
- 价值: ⭐⭐⭐⭐ 为 CCA-LLM 这一新兴方向"先扫雷"，让后续工作避开两条死路，同时启发"影响感知"型 CCA 变种

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[NeurIPS 2025\] ARECHO: Autoregressive Evaluation via Chain-Based Hypothesis Optimization for Speech Multi-Metric Estimation](../../NeurIPS2025/interpretability/arecho_autoregressive_evaluation_via_chain-based_hypothesis_optimization_for_spe.md)
- [\[ACL 2025\] Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](../../ACL2025/interpretability/bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)

</div>

<!-- RELATED:END -->
