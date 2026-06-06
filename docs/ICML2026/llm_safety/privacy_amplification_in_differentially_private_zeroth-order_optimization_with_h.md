---
title: >-
  [论文解读] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States
description: >-
  [ICML 2026][LLM安全][差分隐私] 作者给"差分隐私零阶优化（DP-ZOGD）"首次证出了**收敛的 hidden-state DP 上界**——通过设计一个"定向 + 各向同性"混合噪声机制并构造一个介于两条相邻轨迹之间的辅助过程，绕开了零阶更新缺乏全局 Lipschitz 性这一技术障碍…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "差分隐私"
  - "零阶优化"
  - "PABI"
  - "hidden-state 分析"
  - "耦合分析"
---

# Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States

**会议**: ICML 2026  
**arXiv**: [2506.00158](https://arxiv.org/abs/2506.00158)  
**代码**: 无（理论论文）  
**领域**: LLM 安全 / 差分隐私 / 零阶优化  
**关键词**: 差分隐私、零阶优化、PABI、hidden-state 分析、耦合分析

## 一句话总结
作者给"差分隐私零阶优化（DP-ZOGD）"首次证出了**收敛的 hidden-state DP 上界**——通过设计一个"定向 + 各向同性"混合噪声机制并构造一个介于两条相邻轨迹之间的辅助过程，绕开了零阶更新缺乏全局 Lipschitz 性这一技术障碍，揭示出"扩大每步采样方向数 $K$ 反而能降隐私损失"这一前所未知的 DP 算法设计准则。

## 研究背景与动机

**领域现状**：随模型规模膨胀到几十上百 B 参数，DP-SGD 这种一阶 DP 训练方法的 per-sample 梯度 clipping 带来巨大显存开销。最近 MeZO-DP（Zhang 等 2024a）、Tang 等（2024）等工作把零阶优化（ZO，只用 forward pass 评估损失）搬进 DP 框架，把 60B+ 模型 fine-tune 出来、且效果接近 DP-LoRA。但他们的隐私分析仍依赖 composition theorem——隐私预算随训练步数 $T$ 线性累积，必须仔细控制 stopping point。

**现有痛点**：一阶 DP-SGD 已经有"privacy amplification by iteration（PABI）"理论——把中间 iterate 当作隐藏的、只发布最终参数，可证 $\varepsilon$ 随 $T$ 饱和。但这套分析需要两个东西：(i) 噪声各向同性（保证 shifted Rényi divergence 可控）；(ii) 更新映射全局 Lipschitz。零阶方法两者都破——噪声只沿随机方向 $u$ 的标量 Gaussian、是各向异性的；ZOGD 更新映射在所有 $u$ 上的全局 Lipschitz 常数比一阶大得多。直接给整个 $\mathbb{R}^d$ 加各向同性噪声虽然能套用现有分析，但会严重恶化 utility-privacy trade-off（在 $u^\perp$ 方向加的噪声不参与隐私却完全浪费 utility）。

**核心矛盾**：(噪声形状 vs 分析框架) 之间存在结构矛盾——utility 要求噪声尽可能沿更新方向（标量、定向），但 PABI 的 shifted-divergence 分析要求各向同性 + 全局 Lipschitz。所以问题变成"能不能造一个混合噪声，让 utility 和 hidden-state 分析两边都满足？"

**本文目标**：(i) 给 DP-ZOGD 提出一个能同时支持定向和各向同性噪声的统一 noisy update 规则；(ii) 推出第一条收敛的 hidden-state DP bound（$\varepsilon$ 不随 $T\to\infty$ 爆炸）；(iii) 揭示之前文献忽略的算法设计自由度（更新维度 $K$ 的角色、方向是否需要正交）。

**切入角度**：作者注意到零阶更新映射虽然不是全局 Lipschitz，但**逐点高概率 Lipschitz**——单个固定点对附近点，Lipschitz 常数远比全局小。这给了一个绕开 shifted-divergence 的入口：不再追求"两条相邻轨迹的 Rényi divergence 沿原始更新可控"，而是显式构造一个"在两条相邻轨迹之间的"第三条辅助过程 $\widetilde W$，把分析拆成两段。

**核心 idea**：用"混合噪声 + 耦合辅助过程"两件套——前者解决 utility 问题，后者绕开 Lipschitz 障碍，从而证出 ZO 也能享受 PABI 风格的隐私放大。

## 方法详解

### 整体框架
作者考虑 ERM 问题 $L(w;\mathcal D)=\frac1n\sum_i \ell_i(w)$，在凸有界域 $\mathcal B_R$ 上跑投影零阶 GD。每步 update 由三件事拼起来：(1) 用 $K$ 个 *正交* 方向 $\{u_{t,k}\}_{k=1}^K$（从 Stiefel 流形 $V_K(\mathbb R^d)$ 均匀抽样）算 two-point 零阶梯度 $\hat g_t(w_t;u_{t,k})=\frac1n\sum_i \mathsf{clip}(\frac{\ell_i(w_t+\xi u_{t,k})-\ell_i(w_t-\xi u_{t,k})}{2\xi};\Delta)\,u_{t,k}$；(2) 沿这些方向各加一份标量 Gaussian $G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$；(3) 全空间再加一份小的各向同性 Gaussian $G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$。混合系数 $\beta_t\in[0,1]$ 把"定向 vs 各向同性"参数化为一个连续旋钮，$\beta_t=1$ 退化为只沿方向加噪（即 Zhang 等 2024a 的 mechanism (a)），$\beta_t=0$ 退化为全各向同性（mechanism (b)）。分析层面则配一个介于 $W_t,W_t'$ 之间的辅助 $\widetilde W_t$，让 TV bound 走 $W_t\leftrightarrow \widetilde W_t$ 这边、Rényi bound 走 $\widetilde W_t\leftrightarrow W_t'$ 那边。

### 关键设计

1. **混合定向 + 各向同性的 Noisy-ZOGD 机制**:

    - 功能：把 ZO 的两个常用噪声机制 (a)/(b) 统一成一个连续族，允许 hidden-state 分析在两端之间找到比纯 (a) 更紧的 bound。
    - 核心思路：$w_{t+1}=\Pi_{\mathcal B_R}[w_t-\frac{\eta}{K}\sum_k \hat g_t(w_t;u_{t,k})+\frac{\eta}{\sqrt K}\sum_k G_{t,k}^{(1)} u_{t,k}+\frac{\eta}{\sqrt d}G_t^{(2)}]$，其中 $\{u_{t,k}\}$ 是正交（不是 i.i.d. uniform on $\mathbb S^{d-1}$）、$G_{t,k}^{(1)}\sim\mathcal N(0,\beta_t\sigma^2)$、$G_t^{(2)}\sim\mathcal N(0,(1-\beta_t)\sigma^2 I_d)$。这个参数化在所有 $\beta_t,K$ 下都保持等价的总噪声方差，从而 utility 上限不变。
    - 设计动机：定向部分 $G_{t,k}^{(1)} u_{t,k}$ 对 utility 友好（噪声只落在真正承载隐私敏感量的方向上），各向同性部分 $G_t^{(2)}$ 则给后面的 shifted Gaussian mechanism 留出"操作空间"——耦合分析中的 vector shift $v_t$ 需要靠这一份各向同性噪声来吸收。

2. **耦合辅助过程 $\widetilde W$ 绕开全局 Lipschitz**:

    - 功能：把 hidden-state DP 分析从"shifted-divergence 沿原轨迹"重构成"TV(coupling failure) + Rényi(shifted Gaussian)"两段，使得只需要 pointwise Lipschitz 即可。
    - 核心思路：对两条相邻轨迹 $W_t,W_t'$（分别对应数据集 $\mathcal D,\mathcal D'$）构造第三条 $\widetilde W$，从某个时刻 $\tau$ 开始，$\widetilde W_{t+1}\stackrel{d}{=}\Pi_{\mathcal B_R}[\hat\psi_t(\widetilde W_t)+Y_t+Z_t+v_t]$，其中 shift $v_t:=\min(a_t,(\|d_t\|-z_{t+1})_+)\frac{d_t}{\|d_t\|}$、$d_t:=\hat\psi_t(W_t)-\hat\psi_t(\widetilde W_t)$。这把分析切两段：(i) $W$ 与 $\widetilde W$ 的 TV 距离用 Lemma 3.6 的高概率 pointwise Lipschitz 控制（坏事件概率 $\delta_f$）；(ii) $\widetilde W$ 与 $W'$ 之间是经典 shifted Gaussian mechanism，Rényi divergence 可标准 PABI 累积。最终结合 Lemma 3.7 的 forward $W_\infty$ 跟踪 $W_\infty(w_t,w_t')\le \min(2R,2\eta\Delta t/\sqrt K)$ 收口。
    - 设计动机：零阶 update 的 Lipschitz 常数 $c_1=\sqrt{1-\sum_k\upsilon_k+c^2\sum_k\gamma_k}$ 里 $\upsilon_k,\gamma_k\sim\mathsf{Beta}(K/2,(d-K)/2)$ 是随机的，没法保证 a.s. 全局 $\le c$。但用 Beta 分布尾界可证 $c_1\le \bar c_1=\sqrt{1-(1-c^2)K/d+\vartheta(1+c^2)K/d}$ 以高概率成立。坏事件丢进 TV term、好事件走 Rényi term，分而治之。

3. **正交更新方向 + 多维 $K$ 的隐私收益**:

    - 功能：给出之前文献没有的算法设计准则——选 $K>1$ 且方向正交可以同时降低隐私损失和 utility 误差。
    - 核心思路：定理 3.2 / 推论 3.3 给出闭式 DP 上界 $\varepsilon=O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$。关键是 $\min$ 里出现的 $1/K$：当 $T\to\infty$、bound 饱和，$\varepsilon$ 反比于 $K$。同时 Beta 分布尾界 Lemma 3.6 在正交 $\{u_{t,k}\}$（即从 Stiefel 流形抽）下比独立 i.i.d. 球面抽样更紧。
    - 设计动机：标准 composition 下，加大 $K$ 会让隐私按 $\tilde O(\sqrt K)$ 变差（因为暴露更多敏感方向），所以前作都避免大 $K$。hidden-state 分析改变了游戏规则——在固定 utility 约束下，多用几个方向反而稀释了每个方向上的敏感度。正交方向进一步去掉了重叠的隐私贡献。这是 ZO 独有的、一阶方法没有的现象。

### 损失函数 / 训练策略
不涉及训练目标修改——本文是给现有 ZO DP 优化器套一个更紧的隐私 accountant。具体地：(i) 选 $\eta\le K/M$（强凸）或 $\le 2K/M$（凸）；(ii) 选 $\xi\le 2\Delta/(n\eta M\sqrt{2d})$；(iii) $K$ 满足 $\max(20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil MRn\sqrt{2d}/\Delta\rceil),1)\le K\le d/2$。对 non-convex 情形给数值 accountant 而非闭式。

## 实验关键数据

### 主实验
本文以理论为主，数值验证集中在论文 Figure 1：在 smooth strongly convex loss + bounded domain 上比较 hidden-state bound、standard composition、output perturbation 三者随 $T$ 的 $\varepsilon$ 曲线。

| 方法 | $\varepsilon$ 随 $T$ 趋势 | 备注 |
|------|--------------------------|------|
| Standard composition (Theorem 3.1, $\beta_t=1$) | $O(\sqrt{\Delta^2\log(1/\delta)T/(n^2\sigma^2)})$ | 随 $T$ 无界增长 |
| Output perturbation | $O(\sqrt{R^2\log(1/\delta)/\sigma^2})$ | 与 $T$ 无关，但常数大 |
| **Hidden-state DP（本文，Corollary 3.3）** | $O(\sqrt{\Delta^2\log(1/\delta)/(n^2\sigma^2)\cdot \min(T,MRn\sqrt d/(K\Delta))})$ | 随 $T$ 饱和、关于 $K$ 反比 |

在 $K\ge K_{\min}$（推论 3.3 给出的下界）之后，本文 bound 在中等到大 $T$ 区域都严格优于 standard composition 与 output perturbation。

### 消融实验

| 配置 | 关键结论 | 说明 |
|------|---------|------|
| $\beta_t=1$（全定向） | utility 最好但分析最难 | 旧方法 mechanism (a) |
| $\beta_t=0$（全各向同性） | 分析容易但 utility 差 | 旧方法 mechanism (b) |
| $\beta_t\in(0,1)$ | utility-privacy trade-off 改善 | 本文混合方案的甜点 |
| $K=1$, i.i.d. 球面方向 | 经典 ZO 配置 | 隐私分析松 |
| $K>1$, 正交方向 | 隐私 bound 更紧、收敛速率更好 | 本文新发现 |

### 关键发现
- 第一条收敛的 hidden-state DP bound 揭示：$\varepsilon$ 在 $T\to\infty$ 时饱和到 $O(MRn\sqrt d/(K\Delta))$ 量级——和"再训多少步"无关，只取决于域半径、Lipschitz 常数和方向数。
- 增大每步采样方向数 $K$ 在 hidden-state 下反而降低隐私损失——前作（standard composition）认为 $K$ 越大隐私越糟，正好相反。这是把"PABI 思想"搬到 ZO 上的重要 algorithmic insight。
- 用正交 $\{u_{t,k}\}$（Stiefel 流形抽样）替代 i.i.d. 球面抽样能进一步降隐私，因为正交方向之间没有冗余泄露。

## 亮点与洞察
- "构造一条在两条相邻轨迹之间的辅助过程，把分析切成 TV + Rényi 两段"是一个非常通用的耦合分析技巧——只要原始过程缺乏全局 Lipschitz 但有 pointwise Lipschitz，就可以借鉴。这同样能用于 SGD-on-manifold、Langevin 在非凸景观的 hidden-state 分析。
- 把噪声机制写成连续族 $\beta_t\in[0,1]$ 而不是 binary 切换，本身就是一种"把隐含设计选择搬上台面"的范式（与 SRR 那篇"把秩预算切两份"异曲同工），让分析能够在两个极端之间寻找最优点。
- "Stiefel 流形抽方向比球面 i.i.d. 紧"是一个具体到 ZO 优化的实践指导，落地代价很小（只需把 i.i.d. Gaussian 方向做一次 QR 正交化）但能直接拿到更好隐私 bound。

## 局限与展望
- 强凸/凸 + bounded domain + smooth + per-sample Lipschitz 是个比较强的假设组合；non-convex 情形虽然给了数值 accountant 但缺闭式。
- 推论 3.3 对 $K$ 的下界 $K\ge 20(1+c^2)^2/(3(1-c^2)^2)\log(4/\delta\lceil\cdot\rceil)$ 在 $c\to 1$（弱凸）时可能很大，意味着实际可用 $K$ 区间会被压缩，需要 mini-batch 来缓解。
- 文章没在 LLM 实操（fine-tune 60B 模型）上做端到端实验验证 hidden-state 带来的实际预算节省，只给理论 + 数值曲线；后续若能在 MeZO-DP 改进版上做大规模实验会更有说服力。
- 耦合分析的 shift 序列 $\{a_t,z_t\}$ 来自一个约束优化问题，没闭式解；虽然作者给了 numerical accountant 但调参成本不算低。

## 相关工作与启发
- **vs DP-SGD + PABI（Feldman 2018, Altschuler-Talwar 2022/2023, Chien-Li 2025）**：一阶 PABI 用 shifted Rényi divergence 直接走原始过程，依赖各向同性噪声 + 全局 Lipschitz。本文针对 ZO 的反向工程：定向噪声 + 辅助过程 + pointwise Lipschitz，等价地证出了"$\varepsilon$ 随 $T$ 饱和"的关键 PABI 性质。
- **vs MeZO-DP（Zhang 等 2024a）**：他们提出 mechanism (a)（沿方向加噪）经验上 utility 更好但隐私分析停在 composition；本文给同一个 update 配上更紧的 hidden-state bound，并发现了 $K$ 越大越好这一反直觉。
- **vs Tang 等 2024**：类似 setting 但仍走 composition；本文相当于在他们的算法上加 free upgrade（混合噪声 + 正交方向），保持 utility 的同时降低隐私预算。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一条 ZO 的收敛 hidden-state DP bound + "$K$ 越大隐私越好"反直觉，分析技巧也是新的。
- 实验充分度: ⭐⭐⭐ 理论论文，数值验证仅限合成 setting + Figure 1 的曲线对比，缺乏 LLM 大模型端到端验证。
- 写作质量: ⭐⭐⭐⭐ 三个挑战（噪声形状、Lipschitz、composition vs hidden-state）逐一解构，引言把"为什么不能直接套一阶 PABI"讲得很清楚。
- 价值: ⭐⭐⭐⭐ 给所有走 DP-ZO 路线的大模型 fine-tune 提供了更紧的 accountant，可以让同样 $\varepsilon$ 预算下训更多步或同样步数享更紧 $\varepsilon$。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](../../ICLR2026/llm_safety/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
