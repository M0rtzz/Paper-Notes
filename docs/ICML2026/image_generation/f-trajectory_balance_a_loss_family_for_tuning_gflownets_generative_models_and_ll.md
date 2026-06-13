---
title: >-
  [论文解读] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data
description: >-
  [ICML2026][图像生成][f-散度] 把 GFlowNet/Kimi 里"对 log-prob 差取平方"的 $\mathbb{KL}_{sq}$ 代理损失推广到整族 $f$-散度，得到一族同时具备"on-policy 梯度等于对应 $f$-散度真梯度、off-policy 仍有同一全局最优"的可调控 mode-seeking↔mode-covering 损失，并在合成网格、SynFlowNet 分子生成、扩散模型条件采样和异步 LLM RL（GSM8k / MATH）上验证。
tags:
  - "ICML2026"
  - "图像生成"
  - "f-散度"
  - "轨迹平衡"
  - "GFlowNets"
  - "off-policy"
  - "LLM RLHF"
---

# $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data

**会议**: ICML2026  
**arXiv**: [2605.15417](https://arxiv.org/abs/2605.15417)  
**代码**: 待确认  
**领域**: 强化学习 / GFlowNets / 生成模型微调  
**关键词**: f-散度, 轨迹平衡, GFlowNets, off-policy, LLM RLHF

## 一句话总结
把 GFlowNet/Kimi 里"对 log-prob 差取平方"的 $\mathbb{KL}_{sq}$ 代理损失推广到整族 $f$-散度，得到一族同时具备"on-policy 梯度等于对应 $f$-散度真梯度、off-policy 仍有同一全局最优"的可调控 mode-seeking↔mode-covering 损失，并在合成网格、SynFlowNet 分子生成、扩散模型条件采样和异步 LLM RL（GSM8k / MATH）上验证。

## 研究背景与动机
**领域现状**：GFlowNet 训练（Trajectory Balance）、变分推断里 VarGrad、以及 Kimi K1.5 / K2 这种大模型 RL 微调，最近不约而同地选择了同一种损失形式——对"模型 log-prob"和"目标 log-prob"之差取平方，即 $\mathbb{KL}_{sq}=\tfrac12 (\log\pi_\theta/\pi_\star)^2$。其魅力在于：on-policy 求自动微分梯度，期望恰好等于真正的 $\nabla_\theta \mathbb{KL}(\pi_\theta\|\pi_\star)$；off-policy 用任意 $\mu$ 采样也仍是合法损失，最优点不变。

**现有痛点**：但 $\mathbb{KL}_{sq}$ 继承的是 reverse KL 的几何性质，是 **mode-seeking** 的——模型会塌缩到少数几个高奖励模式上。对于药物发现、探索性 agent、需要采到所有高奖励候选的场景，更想要 **mode-covering** 行为（forward KL、Hellinger 等）。已有工作要么用 REINFORCE/PPO 梯度（高方差），要么用对数方差散度构造 Pearson $\chi^2$ 变体（但破坏了"on-policy 梯度匹配"这一关键性质），缺一个统一框架。

**核心矛盾**：低方差稳定训练 + off-policy 合法性 + 可调 mode 行为，这三件事此前没人能同时拿到。

**本文目标**：(i) 给任意 $f$-散度构造一个代理损失 $\mathcal{L}_f$，使其 on-policy 梯度恰好等于 $\nabla_\theta D_f$，off-policy 同一最优点不变；(ii) 反过来证明任何"作用于 log-prob 之差的平移不变损失"都对应唯一一个 $f$-散度，两个映射互为逆；(iii) 把 VarGrad 的批内归一化推广到任意 $f$；(iv) 解决小 $\beta$ 下 $\exp(\beta^{-1}r)$ 数值爆炸问题。

**切入角度**：$\mathbb{KL}_{sq}$ 之所以"魔法"，本质是 $f(u)=u\log u$ 对应的 $f'(u)=\log u + 1$ 使得 score function 系数恰好长成 $\log\pi_\theta - \log\pi_\star$。对一般 $f$，只要把这个"系数"换成 $f'(\exp t) - f'(1)$ 再对 $t$ 积分，就能复刻同样的梯度对齐机制。

**核心 idea**：用 $\mathcal{L}_f(\Delta) = \int_0^{\Delta}\bigl(f'(\exp t)-f'(1)\bigr)\,dt$ 这一族**平移不变积分损失**统一替代平方损失，作用对象是 log-prob 差 $\Delta_\theta(\mathbf{y})=\log\pi_\theta(\mathbf{y})-\log\pi_\star(\mathbf{y})$。

## 方法详解

### 整体框架
这套方法要解决的是：现在训练 GFlowNet、扩散模型和大模型 RL 普遍用的那个"对 log-prob 之差取平方"的损失 $\mathbb{KL}_{sq}$ 只能 mode-seeking（塌到少数高奖励模式），而很多场景想要 mode-covering（采全所有候选）。作者把这个平方损失换成一个由凸函数 $f$ 决定的积分损失 $\mathcal{L}_f$，作用对象仍是 log-prob 差 $\Delta_\theta(\mathbf{y})=\log\pi_\theta(\mathbf{y})-\log\pi_\star(\mathbf{y})$，于是只要换一个 $f$（或调一个 $\alpha$ 旋钮）就能在 mode-seeking 和 mode-covering 之间连续滑动，而平方损失原有的"on-policy 梯度对齐 + off-policy 合法"两条好性质一并保留。配分函数未知时用 DevGrad 在批内估，正则系数 $\beta$ 过小时用 Tempered Loss 把能量缩回稳定区间，三套 pipeline 里只需把代码中的 `(log_pi_theta - log_pi_star)**2` 换成 `L_f(log_pi_theta - log_pi_star)` 即可。

### 关键设计

**1. $f$-散度代理损失与双向等价定理：用一个积分把平方损失推广成整族**

平方损失之所以"魔法"，本质是 $f(u)=u\log u$ 的导数 $f'(u)=\log u+1$ 让 score function 的系数恰好长成 $\log\pi_\theta-\log\pi_\star$。作者顺着这条线索，对任意满足 $f(1)=0$、$f'(1)=f''(1)=1$ 归一化的凸函数 $f$，把"系数"换成 $f'(\exp t)-f'(1)$ 再对 $t$ 积分，定义出代理损失 $\mathcal{L}_f(\Delta)=\int_0^{\Delta}\bigl(f'(\exp t)-f'(1)\bigr)\,dt$（Prop 4.2）。代入 $f(u)=u\log u$ 立刻还原出 $\tfrac12\Delta^2$，说明它确实是 $\mathbb{KL}_{sq}$ 的一般化；对 $\alpha$-散度则有闭式 $\mathcal{L}_\alpha(\Delta)=\frac{1}{(\alpha-1)^2}e^{(\alpha-1)\Delta}-\frac{\Delta}{\alpha-1}-\frac{1}{(\alpha-1)^2}$，$\alpha\to 1$ 退化为平方损失即 Trajectory Balance。这样构造出来的损失同时拿到两条性质：on-policy 自动微分梯度的期望恰好等于真 $\nabla_\theta D_f(\pi_\theta\|\pi_\star)$，保证优化目标就是想要的那个散度；而损失对 $\Delta$ 平移不变（加常数不变），这正是 off-policy 仍合法的来源——差的那个常数就是 $\log Z$，不影响最小化。

更进一步，作者把这个映射证成双射（Prop 4.4）：任意平移不变可微损失 $\ell$ 反过来唯一对应一个 $f$-散度 $f_\ell(u)=\lambda_1\int_1^u \ell'(\log t)\,dt+\lambda_2(u-1)+c$，且 $f_{\mathcal{L}_f}=f$、$\mathcal{L}_{f_\ell}=\ell$ 互为逆。这等于宣告：用平方损失之外的任何凸损失训练 GFlowNet/LLM，本质上都在隐式优化某个 $f$-散度，以前的工作只是没意识到自己挑了哪个——双射给出的就是一张"想要哪个 mode 行为就查哪个损失"的配方表。

**2. DevGrad：把 VarGrad 推广到任意 $f$，顺手解决未知配分函数**

LLM RL 里 $\pi_\star(\mathbf{y})=\frac{1}{Z}\exp(\mathcal{R}(\mathbf{y}))$ 的配分函数 $Z$ 几乎永远算不出来。作者的做法是在每个 batch $\mathcal{B}=\{\mathbf{y}_1,\dots,\mathbf{y}_B\}$ 里先解一维优化 $\widehat{\log Z}=\arg\min_C\frac1B\sum_i\mathcal{L}_f(\Delta(\mathbf{y}_i)+C)$ 在线估出 $\log Z$，再带 stop-gradient 代回损失 $\mathcal{L}_f^{\text{DG}}(\mathcal{B},\theta)=\frac1B\sum_i\mathcal{L}_f\bigl(\Delta(\mathbf{y}_i)+\text{SG}[\widehat{\log Z}]\bigr)$。这一步同时压住了梯度方差：批内归一化等价于对 score function 系数做 batch 中心化，正是 VarGrad 降方差的机制，而作者证明这套"中心化降方差"对任意广义偏差（generalised deviation, Rockafellar 2006）都成立。于是当 $\mathcal{L}_f(y)=y^2$ 时 $\widehat{\log Z}$ 就是均值、损失还原成方差，正好是经典 VarGrad；$\mathcal{L}_f(y)=|y|$ 时变成绕中位数的平均绝对偏差，对应 Total Variation 散度；Kimi K1.5/K2 用平均奖励 $\bar r$ 代替 $\log Z$ 也只是 reverse KL 下的一个特例近似。换损失这一件事，把"$Z$ 怎么估"和"梯度方差怎么压"两个问题一起解决了。

**3. Tempered Loss：小 $\beta$ 下把能量缩回数值稳定区间**

KL 正则系数 $\beta$ 很小（如 $0.005$）时，目标里的 $\exp(\beta^{-1}r)$ 会冲到 $e^{200}$ 量级直接溢出，硬裁剪又会丢梯度信号。作者引入 tempered 分布 $\tilde p_\beta\propto p^\beta$，它的能量函数 $\beta\mathcal{R}_\star=\beta\log\pi_{\text{ref}}+r$ 与 $1/\beta$ 无关，对应的 tempered 损失为 $\tilde{\mathcal{L}}_{f,\beta}(\Delta)=\frac1\beta\mathcal{L}_f(\beta\Delta)$。因为"tempered log-prob 相等 $\Rightarrow$ 原始 log-prob 相等"，最优点不变；前面的 $1/\beta$ 因子又保证梯度尺度不随 $\beta$ 漂移。把 $f(u)=u\log u$ 代进去恰好得到 Kimi 损失 $\frac{1}{2\beta}\bigl(\beta\log\frac{\pi_\theta}{\mathcal{R}}-\log\tilde Z\bigr)^2$，再配 $\widehat{\log Z}=\bar r$ 近似就是 Kimi 实际在跑的形式。这一设计把原本纯工程的稳定化 trick 拔升成有理论依据的手段，同时把 Kimi 经验损失自然纳入框架，也解锁了同一套温度调度对其它 $f$ 都成立。

### 损失函数 / 训练策略
GFlowNet 上把 Trajectory Balance 平方损失 $(\Delta(\tau,\theta,\phi))^2$ 直接替换为 $\mathcal{L}_f(\Delta(\tau,\theta,\phi))$ 即得 **$f$-Trajectory Balance**，Prop 5.1 证明 $\nabla_\theta D_f(\pi_F\|\pi_B)=\mathbb{E}_{\tau\sim\pi_{F,\theta}}[\nabla_\theta \mathcal{L}_f]$；反向策略梯度对应另一 $h(u)=\int_1^u (2-f'(1/t))\,dt$ 散度。LLM 异步 RL 端用 tempered DevGrad，不需要 clipping/importance weighting/masking；GFlowNet 端常用 $\alpha\in\{0.5, 0.75, 1.2, 2\}$，$\alpha<1$ 偏 mode-covering，$\alpha>1$ 偏 mode-seeking，训练时可对 $\alpha$ 做退火。

## 实验关键数据

### 主实验

| 任务 | 配置 | 关键指标 | 现象 |
|------|------|---------|------|
| Hypergrid (4 模式) | Forward KL ($\alpha=0$) / Hellinger ($\alpha=0.5$) | 模式发现数 + 拟合速度 | 找全 4 模式，且比 TB（$\alpha=1$）更快收敛 |
| Hypergrid | Pearson $\chi^2$ ($\alpha=2$) | 模式发现数 | 卡在首个发现的模式（极端 mode-seeking） |
| SynFlowNet (DRD2 等 3 任务) | $\alpha=0.75$ vs TB | 高奖励分子多样性 | 多样性显著高于 TB，奖励持平 |
| SynFlowNet | $\alpha=1.2$ | 多样性 | 模式塌缩 |
| SynFlowNet | $\alpha: 0.75\to 1.2$ 退火 | 多样性 + 高奖励数 | 取得最佳折中 |
| MNIST 扩散条件采样（偶数） | TB | 各数字频率 | 过度采样 0 和 6，覆盖差 |
| MNIST 扩散 | $f$-TB（退火 $\alpha$） | 各数字频率 | 模式更均匀 |
| GSM8k+MATH 异步 RL（50 步延迟） | Qwen2.5 3B–14B / OLMo-2-7B | 熵–奖励权衡 | Forward KL/Reverse KL/Pearson/JS 全跑通，权衡曲线如理论预期；优化版 PPO 在 off-policy 下表现不稳 |

### 消融实验

| 配置 | 行为 | 说明 |
|------|------|------|
| $\mathcal{L}_f$ + DevGrad（完整） | on-policy 梯度匹配 $D_f$ + off-policy 同一最优 | 主张全部成立 |
| 去掉 DevGrad，直接用 $\widehat{\log Z}=\bar r$ | 仅在 $\beta$ 小或近 on-policy 时成立 | Kimi 的近似，作为特例验证一致 |
| 去掉 Tempered Loss（小 $\beta$） | $e^{200}$ 量级溢出 / 必须 clipping 丢信号 | 验证 tempered scaling 的必要性 |
| 反向策略用 $\mathcal{L}_f$ | 对应另一个散度 $h$，不是 $f$ 本身 | 与 TB 一致：反向策略的合法性来自损失合法性而非散度匹配 |

### 关键发现
- 在 $\alpha$-散度族里调 $\alpha$，可以**用一个旋钮平滑滑动** mode-covering↔mode-seeking 行为：Hypergrid 上 $\alpha\le 0.5$ 收益最大，SynFlowNet 上微调（$\alpha\in[0.75,1.2]$）+ 退火即可，过度偏离 1 反而不稳定——理论上"散度族"的连续性在工程上是真的连续可用的。
- LLM 异步 RL 一段尤其关键：在 50 步延迟下连优化过的 PPO（CISPO + DAPO）都不稳，而 tempered DevGrad 不需要任何 clipping / importance ratio / masking 就稳。说明 off-policy 合法性这一性质在大模型 RL 里直接转化成"少几个 hack"的工程红利。
- Kimi K1.5/K2 损失被解释为 reverse KL tempered loss + 用 $\bar r$ 近似 $\widehat{\log Z}$，这把"工业界经验"嵌入了一个清晰的理论坐标系，也意味着 Kimi 风格 RL 可以无痛切换到 mode-covering 变体。

## 亮点与洞察
- "**平移不变损失 ↔ $f$-散度**"的双射映射是一个非常漂亮的桥梁：以前一篇论文挑一种平方变体、一篇挑一个对数方差，本质上各自隐式选了不同的 $f$，没人意识到背后是同一族结构。这种"形式语言层面的统一"对后续设计自定义损失（比如想要 mode-covering 但要更鲁棒的尾部）几乎是给了配方表。
- 同一个 $\mathcal{L}_f$ 既能塞进 GFlowNet 又能塞进 LLM RL 又能塞进扩散模型微调，体现了"以 log-prob 差为一等公民"的设计选择是多通用——这暗示未来 generative-model alignment 的统一框架可能就是 trajectory-level $f$-散度最小化。
- DevGrad 把"$\log Z$ 估计"和"score 中心化降方差"统一成一件事——在 RLHF/GRPO 这种动辄要造各种 baseline 减方差的世界里，这种"一个机制顺手解决两件事"很有价值，可直接迁移到任何带 reward baseline 的 RL pipeline。

## 局限与展望
- 作者承认：反向策略 $\pi_B$ 的 on-policy 梯度不对应 $f$-散度本身而是 $h$ 衍生散度，TB 也有这个性质，但在更深 DAG（非树）的 GFlowNet 上对学习动力学的影响尚未细致分析。
- SynFlowNet 实验里 $\alpha$ 的可用范围窄（$[0.75, 1.2]$ 之外不稳），说明理论上"任意 $f$ 都行"和工程上"哪个 $f$ 在大模型上稳"之间还有 gap，需要更系统的稳定性分析（梯度方差随 $\alpha$ 的解析刻画在附录里只覆盖了 trajectory balance 那部分）。
- LLM 实验只覆盖 GSM8k + MATH（verifiable rewards），还没在 RLHF 偏好数据、推理模型长 CoT 这些 reward 噪声更大的场景验证，mode-covering 是否依然带来正收益是开放问题。
- 没和真正的 GRPO（在同步设定下）做苹果对苹果对比——只在 50 步异步上跟优化版 PPO 对照，工程读者会想看同步对照。

## 相关工作与启发
- **vs Trajectory Balance (Malkin 2022a)**：TB 是 $f(u)=u\log u$ 即 reverse KL 的特例；本文把单点扩成整族，并暴露 mode 行为这一可控旋钮。
- **vs Vargrad / 对数方差散度 (Richter 2020 / Nüsken–Richter 2021)**：VarGrad 是 reverse KL DevGrad；对数方差散度构造的 Pearson 变体被作者点名指出"不保持 on-policy 梯度匹配"，本文用 generalised deviation 替换 variance 修正了这一性质。
- **vs Kimi K1.5/K2 (Team 2025a,b)**：把 Kimi 损失重新读作 reverse KL 的 tempered DevGrad + $\widehat{\log Z}\approx \bar r$ 近似，等于给 Kimi 工程实践一个理论坐标系，并解锁了 mode-covering 变体。
- **vs Tang & Munos 2025 / Bartoldson 2025**：他们指出 $\mathbb{KL}_{sq}$ 的 on-policy 梯度匹配 + off-policy 有效性；本文把这两个性质同时上升到 $f$ 散度族。
- **vs Silva 2024（GFlowNet 替换散度）**：Silva 等已经讨论用其它散度训 GFlowNet 但只在 on-policy 设定下；本文的核心增值是 off-policy 合法性，而 off-policy 探索正是 GFlowNet 相对于其它 RL 的杀手锏。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把零散的"平方损失变体"统一成 $f$-散度族，并给出双向等价定理，是真正的理论 unification。
- 实验充分度: ⭐⭐⭐⭐ 覆盖合成网格 / SynFlowNet / 扩散 / 异步 LLM RL 四类任务，但 LLM 端只有 verifiable-rewards 数学题，缺 RLHF 偏好数据。
- 写作质量: ⭐⭐⭐⭐ 公式密集但每一步都有 motivation 串联；图 1 的 $\alpha$ 旋钮直觉给得很好。
- 价值: ⭐⭐⭐⭐⭐ 给所有做 RL 微调 / GFlowNet / 对齐的人提供了一个即插即用的损失字典，并把 Kimi 工程经验纳入理论框架。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ICML 2026\] A Diffusive Classification Loss for Learning Energy-based Generative Models](a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)

</div>

<!-- RELATED:END -->
