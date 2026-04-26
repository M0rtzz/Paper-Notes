---
title: >-
  [论文解读] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization
description: >-
  [ICML 2025][LLM对齐][DPO] 将序列级PPO分解为一系列token级近端策略优化问题，并引入token级奖励引导函数 $f(\hat{r}(s_t, a_t))$ 来替代DPO中的固定常数 $\beta$，使不同token根据各自奖励值呈现不同程度的偏离参考策略，在MT-Bench/AlpacaEval 2/Arena-Hard上分别提升最多7.5/6.2/4.3个胜率点。
tags:
  - ICML 2025
  - LLM对齐
  - DPO
  - token-level reward
  - preference optimization
  - RLHF
  - 细粒度奖励引导
---

# TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization

**会议**: ICML 2025  
**arXiv**: [2506.14574](https://arxiv.org/abs/2506.14574)  
**代码**: [dvlab-research/TGDPO](https://github.com/dvlab-research/TGDPO)  
**领域**: LLM对齐/RLHF  
**关键词**: DPO, token-level reward, preference optimization, RLHF, 细粒度奖励引导

## 一句话总结

将序列级PPO分解为一系列token级近端策略优化问题，并引入token级奖励引导函数 $f(\hat{r}(s_t, a_t))$ 来替代DPO中的固定常数 $\beta$，使不同token根据各自奖励值呈现不同程度的偏离参考策略，在MT-Bench/AlpacaEval 2/Arena-Hard上分别提升最多7.5/6.2/4.3个胜率点。

## 研究背景与动机

**DPO的序列级局限性**：DPO将奖励函数通过最优策略重新参数化，绕开了训练独立奖励模型的步骤。但DPO本质上是一个序列级的bandit问题——对整个response赋予统一的奖励信号，无法区分序列中哪些token是preferred、哪些是dispreferred。

**token级奖励对PPO的成功**：已有工作（Yang et al., 2023; Yin et al., 2025; Zhong et al., 2024）证明了dense token-level reward能显著提升PPO的对齐性能，缓解稀疏奖励（delayed feedback）带来的训练不稳定和采样低效问题。

**核心难点**：将token级奖励引导扩展到DPO非常困难。DPO的reward函数由被优化的策略本身表达，直接引入token级reward后，loss中会出现依赖策略的partition function $Z(s_t)$，无法直接消除。这是一个尚未解决的开放问题。

**本文突破**：作者通过上界分解 + 修改token-level PPO + 新的partition function消除定理，首次实现了可计算的带token级奖励引导的DPO损失函数框架。

## 方法详解

### 整体框架

TGDPO的推导遵循三步走策略：
1. **分解**：将序列级PPO with token-level reward guidance 分解为一系列独立的token级PPO问题（Theorem 4.1上界分解）
2. **修改**：在token级PPO中引入reward guidance函数 $f(\hat{r}(s_t, a_t))$，求解闭式最优策略并表示对应的reward（Theorem 4.3）
3. **消除**：利用Bradley-Terry模型 + 新理论结果（Theorem 4.4）消除不可计算的partition function，得到可计算的TGDPO损失

### 关键设计

1. **序列→token级分解（Theorem 4.1）**：序列级PPO的目标函数可以分解为 $\sum_{t=0}^{T-1}(r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)})$。作者证明序列级PPO的最大值被一系列独立token级PPO最大值之和所上界，即每个时间步 $t$ 独立优化 $\max_{\pi_\theta} \mathbb{E}[r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}]$。为使问题可解，将 $s_t \sim \mathcal{D}_t$（依赖策略）松弛为 $s_t \sim \mathcal{D}$（独立于策略）。

2. **修改的token级PPO与reward guidance引入（Theorem 4.3）**：核心思路是将固定的 $\beta$ 替换为 $\beta \cdot f(\hat{r}(s_t, a_t))$，使每个token的KL约束强度由其token级奖励动态调节。具体做法是先将token级PPO等价改写为 $\max \mathbb{E}[\frac{r_\phi(s_t, a_t)}{\beta} - \log \frac{\pi_\theta}{\pi_{\text{ref}}}]$，再将分母的 $\beta$ 替换为 $\beta f(\hat{r})$。这样得到修改后的问题 $\max \mathbb{E}[\frac{r_\phi(s_t, a_t)}{\beta f(\hat{r}(s_t, a_t))} - \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}]$。该问题有闭式最优策略：$\pi_{\theta_t}(a_t|s_t) = \frac{\pi_{\text{ref}}(a_t|s_t) \exp(\frac{r_\phi}{\beta f(\hat{r})})}{Z(s_t)}$，由此可反解出reward：$\frac{r_\phi(s_t, a_t)}{f(\hat{r}(s_t, a_t))} = \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + \beta \log Z(s_t)$。

3. **Partition function消除（Theorem 4.4）**：将上述reward代入Bradley-Terry模型后，偏好概率为 $\Pr(y_w \succ y_l|x) = \sigma(\varphi(\pi_\theta, f, \hat{r}; x, y_w, y_l) + \delta(f, \hat{r}; x, y_w, y_l))$，其中 $\delta$ 包含不可计算的partition function但不依赖于 $\pi_\theta$。Theorem 4.4证明：由于sigmoid函数严格单调递增，$\sigma(\varphi + \delta)$ 与 $\sigma(\varphi)$ 具有相同的最大值点和上升方向，因此优化时可以安全地移除 $\delta$ 项。这是本文的关键理论贡献——保证了消除partition function后，不影响策略的偏好排序和最优策略。

### 损失函数 / 训练策略

**TGDPO损失函数**：

$$\mathcal{L}_{\text{TGDPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \sum_{t} \beta f_w(\hat{r}_t^w) \log \frac{\pi_\theta(y_w^t|...)}{\pi_{\text{ref}}(y_w^t|...)} - \sum_{t} \beta f_l(\hat{r}_t^l) \log \frac{\pi_\theta(y_l^t|...)}{\pi_{\text{ref}}(y_l^t|...)} \right) \right]$$

**实用方法**：采用DPO隐式学到的token级reward $\hat{r}([x, y^{<t}], y^t) = \beta \log \frac{\pi_{\hat{\theta}}(y^t|...)}{\pi_{\text{ref}}(y^t|...)}$，设定：
- 赢回复：$f_w(\hat{r}) = 1 + \alpha \hat{r}$
- 输回复：$f_l(\hat{r}) = 1 - \alpha \hat{r}$

其中 $\alpha$ 为正常数（足够小时保证 $f > 0$）。

**梯度层面的直觉理解**：
- 赢回复中reward > 0的token → 权重 $1 + \alpha\hat{r} > 1$ → 梯度放大，鼓励生成
- 赢回复中reward < 0的token → 权重 $1 + \alpha\hat{r} < 1$ → 梯度缩小，不盲目鼓励
- 输回复中reward < 0的token → 权重 $1 - \alpha\hat{r} > 1$ → 梯度放大，更强抑制
- 输回复中reward > 0的token → 权重 $1 - \alpha\hat{r} < 1$ → 梯度缩小，不盲目抑制

**训练流程**：先用标准DPO训练一个模型 $\pi_{\hat{\theta}}$，用其计算token级reward；然后用TGDPO损失从SFT模型开始训练。

## 实验关键数据

### 主实验

| 模型 / 注释器 | 方法 | AlpacaEval 2 WR(%) | Arena-Hard WR(%) | MT-Bench Score | MT-Bench WR(%) |
|---|---|---|---|---|---|
| Llama3-8B + PairRM | SFT | 30.6 | 21.4 | 7.9 | 27.5 |
| | DPO | 41.7 | 30.4 | 8.0 | 37.5 |
| | SimPO | 39.8 | 28.7 | 7.8 | 32.5 |
| | **TGDPO** | **43.9** | **34.3** | 8.0 | **41.9** |
| Llama3-8B + ArmoRM | DPO | 40.8 | 36.2 | 8.2 | 46.3 |
| | SimPO | 37.0 | 28.1 | 7.8 | 42.5 |
| | **TGDPO** | **42.5** | **40.5** | 7.9 | 45.0 |
| Llama3.2-3B + ArmoRM | DPO | 29.6 | 23.2 | 7.9 | 29.4 |
| | **TGDPO** | **35.8** | **25.4** | **8.1** | **36.9** |
| Gemma2-2B + ArmoRM | DPO | 40.8 | 26.4 | 8.0 | 43.1 |
| | **TGDPO** | **43.0** | **30.7** | **8.1** | **46.9** |

### 消融实验

| 配置 | AlpacaEval 2 WR(%) | Arena-Hard WR(%) | 说明 |
|------|-----|-----|------|
| DPO w/ convergence | 30.7 | 17.9 | 收敛后严重退化，甚至低于SFT |
| SimPO w/ convergence | 4.6 | 2.4 | 收敛后几乎完全崩溃 |
| **TGDPO w/ convergence** | **43.9** | **34.3** | 收敛后仍保持优异性能 |
| TGDPO α=0.5 | 43.9 | 34.3 | 不同α收敛后性能一致 |
| TGDPO α=1.0 | 42.5 | 33.9 | α越大收敛越快 |
| TGDPO α=2.0 | 43.3 | 34.3 | 性能稳健 |
| TGDPO w/ β=0.1 reward | 42.8 | 34.3 | 对reward质量鲁棒 |
| TGDPO w/ β=0.01 reward | 43.9 | 34.3 | DPO β=0.1性能差但TGDPO不受影响 |

### 关键发现

1. **收敛即可用**：DPO/SimPO在loss收敛时性能崩溃（SimPO甚至降到4.6% win rate），需要精心调参找"甜点"；TGDPO在loss收敛时仍保持最优性能，大幅降低调参负担
2. **α控制收敛速度**：α越大收敛越快（α=2.0约50步收敛，α=0.5约1个epoch），但最终性能几乎不变，允许通过早停节省算力
3. **对reward质量鲁棒**：即使用较差的DPO模型（β=0.1，AlpacaEval仅34.8%）产生的token-level reward，TGDPO仍能达到与最优reward相当的性能（42.8% vs 43.9%）

## 亮点与洞察

- **理论优雅**：通过上界分解→闭式解→partition function消除的三步推导，将token级reward引导自然嵌入DPO框架，且能统一恢复DPO等已有方法（$f \equiv 1$时退化为DPO）
- **实用性强**：不需要额外的token级reward模型，直接复用DPO隐式学到的reward（$\beta \log \frac{\pi_{\hat\theta}}{\pi_{\text{ref}}}$），实现简单且效果显著
- **收敛行为健康**：这是偏好优化领域难得的"loss收敛≈好性能"的方法，消除了传统方法中loss与性能错位的顽疾
- **细粒度token级调控**：能在同一response内，对preferred token放大梯度、对dispreferred token缩小梯度，避免了"全盘接受赢回复、全盘否定输回复"的粗暴做法

## 局限与展望

- **reward来源单一**：实用方法依赖先训一个DPO模型来提供token-level reward，增加了训练pipeline的复杂度；可探索其他更轻量的reward来源
- **f函数形式简单**：当前仅使用线性形式 $f = 1 + \alpha r$，更复杂的非线性设计（如指数、softmax-based）可能带来进一步提升
- **评估范围有限**：仅在instruction-following的helpfulness维度评估，未验证safety/honesty/fairness等对齐维度的效果
- **模型规模偏小**：实验在2B–8B模型上进行，是否在更大模型上仍有显著收益有待验证
- **松弛条件的理论影响**：将 $s_t \sim \mathcal{D}_t$ 松弛为 $s_t \sim \mathcal{D}$ 的误差未定量分析

## 相关工作与启发

- **DPO (Rafailov et al., 2023)**：本文的直接改进目标，TGDPO在 $f \equiv 1$ 时退化为DPO
- **SimPO (Meng et al., 2024)**：去掉reference model的DPO变体，表现不及TGDPO
- **TDPO (Zeng et al., 2024)**：用token级MDP理解DPO并加forward KL，但没有引入token级reward guidance
- **Rafailov et al. (2024)**：证明DPO隐式学到token级reward的理论结果，本文直接利用
- **启发**：可以将TGDPO的framework + 其他细粒度reward源（如process reward model、verifier）结合，用于数学推理/代码生成等需要step-level反馈的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 理论推导完整且有新意（partition function消除定理），但核心idea（token级加权）相对直觉
- 实验充分度: ⭐⭐⭐⭐ — 多模型/多benchmark/多消融，分析TGDPO的独特属性（收敛行为、鲁棒性）很到位
- 写作质量: ⭐⭐⭐⭐⭐ — 推导清晰严谨，motivation阐述到位，从理论到实践的过渡自然流畅
- 价值: ⭐⭐⭐⭐ — 解决了DPO的实际痛点（收敛退化、token级粗粒度），方法简单实用

1. **序列级PPO到token级PPO的分解（Theorem 4.1）**

   核心思路是利用LLM自回归生成的MDP结构：$\pi_\theta(y|x) = \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)$，序列级目标可分解为token级求和。通过上界方法，证明序列级PPO的最大值被 $T$ 个token级PPO子问题的最大值之和所上界：

   $$\max_{\pi_\theta} \mathbb{E}\left[r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right]$$

   关键松弛：将 $s_t \sim \mathcal{D}_t$（依赖 $\pi_\theta$）松弛为 $s_t \sim \mathcal{D}$（独立于 $\pi_\theta$），使问题可解。

2. **带奖励引导的修改token级PPO（Theorem 4.3）**

   在token级PPO中引入奖励引导函数 $f(\hat{r}(s_t,a_t))$，将固定KL惩罚系数 $\beta$ 替换为自适应的 $\beta f(\hat{r}(s_t,a_t))$。技巧是先将 $\beta$ 移至奖励项分母（因其为正常数不影响最优解），再替换：

   $$\max_{\pi_\theta} \mathbb{E}\left[\frac{r_\phi(s_t,a_t)}{\beta f(\hat{r}(s_t,a_t))} - \log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right]$$

   最优策略为：$\pi_{\theta_t}(a_t|s_t) = \frac{\pi_{\text{ref}}(a_t|s_t) \exp\left(\frac{r_\phi(s_t,a_t)}{\beta f(\hat{r}(s_t,a_t))}\right)}{Z(s_t)}$

   由此得到token级奖励的显式表示：$r_\phi(s_t,a_t) = \beta f(\hat{r}(s_t,a_t)) \log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + \beta f(\hat{r}(s_t,a_t)) \log Z(s_t)$

3. **配分函数消除（Theorem 4.4）**

   将token级奖励代入Bradley-Terry模型后，偏好函数包含不可计算的 $\delta(f,\hat{r};x,y_w,y_l)$ 项（含配分函数）。关键发现：该项不依赖策略 $\pi_\theta$，而sigmoid函数严格单调递增，因此优化含 $\delta$ 项的目标等价于优化不含 $\delta$ 的目标。两个策略之间的偏好序关系在消除 $\delta$ 后完全保持不变。

### 损失函数 / 训练策略

**TGDPO损失函数**：

$$\mathcal{L}_{\text{TGDPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\sum_{t=0}^{T_w-1}\beta f_w(\hat{r}_t^w)\log\frac{\pi_\theta(y_w^t|[x,y_w^{<t}])}{\pi_{\text{ref}}(y_w^t|[x,y_w^{<t}])} - \sum_{t=0}^{T_l-1}\beta f_l(\hat{r}_t^l)\log\frac{\pi_\theta(y_l^t|[x,y_l^{<t}])}{\pi_{\text{ref}}(y_l^t|[x,y_l^{<t}])}\right)\right]$$

其中 $f_w, f_l$ 分别是win/lose响应的shaping函数。当 $f_w \equiv f_l \equiv 1$ 时退化为标准DPO。

**实用方法**：采用DPO隐式学到的token级奖励 $\hat{r}([x,y^{<t}],y^t) = \beta\log\frac{\pi_{\hat\theta}(y^t|[x,y^{<t}])}{\pi_{\text{ref}}(y^t|[x,y^{<t}])}$，设置：

- Win响应：$f_w(\hat{r}) = 1 + \alpha\hat{r}$ —— 高奖励token获得更大权重，被更多强化
- Lose响应：$f_l(\hat{r}) = 1 - \alpha\hat{r}$ —— 低奖励（负值）token获得更大权重，被更多抑制

**四种token级行为**：

- Win中的好token（$\hat{r}>0$）：权重 $>1$，强化该动作
- Win中的坏token（$\hat{r}<0$）：权重 $<1$，降低该动作概率
- Lose中的坏token（$\hat{r}<0$）：权重 $>1$，进一步抑制
- Lose中的好token（$\hat{r}>0$）：权重 $<1$，减少对该动作的惩罚

**训练流程**：先用标准DPO训练获得 $\pi_{\hat\theta}$，计算token级奖励，再用TGDPO损失函数训练最终策略。$\alpha$ 控制收敛速度（更大的 $\alpha$ 收敛更快），$\beta=0.01$，训练使用AdamW优化器。

## 实验关键数据

### 主实验

| 模型 + 标注器 | 指标 | DPO | SimPO | TGDPO | 提升(vs最佳) |
|--------------|------|-----|-------|-------|-------------|
| Llama3-8B + PairRM | AlpacaEval 2 WR | 41.7% | 39.8% | **43.9%** | +2.2 |
| Llama3-8B + PairRM | Arena-Hard WR | 30.4% | 28.7% | **34.3%** | +3.9 |
| Llama3-8B + PairRM | MT-Bench WR | 37.5% | 32.5% | **41.9%** | +4.4 |
| Llama3-8B + ArmoRM | AlpacaEval 2 WR | 40.8% | 37.0% | **42.5%** | +1.7 |
| Llama3-8B + ArmoRM | Arena-Hard WR | 36.2% | 28.1% | **40.5%** | +4.3 |
| Llama3.2-3B + ArmoRM | AlpacaEval 2 WR | 29.6% | 26.2% | **35.8%** | +6.2 |
| Llama3.2-3B + ArmoRM | Arena-Hard WR | 23.2% | 22.6% | **25.4%** | +2.2 |
| Gemma2-2B + ArmoRM | AlpacaEval 2 WR | 40.8% | 34.8% | **43.0%** | +2.2 |
| Gemma2-2B + ArmoRM | Arena-Hard WR | 26.4% | 21.1% | **30.7%** | +4.3 |

### 消融实验

| 配置 | AlpacaEval 2 WR | Arena-Hard WR | 说明 |
|------|----------------|---------------|------|
| DPO w/ convergence | 30.7% | 17.9% | 收敛后严重退化 |
| SimPO w/ convergence | 4.6% | 2.4% | 收敛后几乎不可用 |
| TGDPO w/ convergence | **43.9%** | **34.3%** | 收敛后仍保持最佳性能 |
| TGDPO α=0.5 | 43.9% | 34.3% | 慢收敛，epoch 1检查点 |
| TGDPO α=1.0 | 42.5% | 33.9% | 中速收敛，step 60检查点 |
| TGDPO α=2.0 | 43.3% | 34.3% | 快收敛，step 50检查点 |
| TGDPO (DPO β=0.1 reward) | 42.8% | 34.3% | 较差DPO模型的奖励 |
| TGDPO (DPO β=0.01 reward) | 43.9% | 34.3% | 较好DPO模型的奖励 |

### 关键发现

1. **收敛即可用**：TGDPO在损失收敛后依然保持优秀性能，而DPO/SimPO收敛后严重退化。这消除了传统偏好优化中需要精心调参找"甜蜜点"的麻烦
2. **收敛速度可控**：通过调节 $\alpha$ 可自由控制收敛速度，不同 $\alpha$ 收敛后性能一致
3. **对奖励质量鲁棒**：即使使用较差DPO模型（$\beta=0.1$）的token级奖励，TGDPO性能与使用最优奖励时几乎相同
4. **跨模型一致性**：在3B、2B、8B三种规模模型上均一致优于基线

## 亮点与洞察

1. **理论推导严谨且实际方法简洁**：虽然数学推导经历了三个定理，但最终的实用方法极其简单——只需在DPO的每个token log-ratio前乘一个线性权重 $1 \pm \alpha\hat{r}$，实现成本很低
2. **收敛-性能对齐**：这是本文最独特的贡献——解决了偏好优化中"loss降=性能降"的反直觉问题，使训练不再需要在过拟合前及时停止
3. **框架统一性**：TGDPO的损失函数是一个通用框架，$f \equiv 1$ 时退化为DPO，可以衍生出多种变体
4. **无需额外奖励模型**：利用DPO自身隐式学到的token级奖励作为引导信号，无需训练外部奖励模型，保持了DPO"轻量"的优势

## 局限与展望

1. **两阶段训练**：需要先训练一个DPO模型获取token级奖励，再训练TGDPO，增加了计算开销。能否做成迭代/在线更新奖励的单阶段方法？
2. **仅评估helpfulness**：实验集中在指令遵循任务，未验证在safety、honesty等对齐维度的效果
3. **松弛的理论gap**：$s_t \sim \mathcal{D}_t$ 到 $s_t \sim \mathcal{D}$ 的松弛是否引入实质误差未被分析
4. **shaping函数选择**：$f = 1 + \alpha\hat{r}$ 只是一种简单的线性设计，是否存在更优的非线性形式？
5. **$\alpha$ 需足够小以保证 $f > 0$**：当token奖励绝对值较大时，$f$ 可能为负违反假设，鲁棒性边界不清晰

## 相关工作与启发

- **TDPO (Zeng et al., 2024)**：用token级MDP理解DPO，引入forward KL散度，但未利用token级奖励引导
- **SimPO (Meng et al., 2024)**：无参考模型的DPO变体，对齐解码目标，但仍是序列级优化
- **Rafailov et al., 2024 (From r to Q\*)**：证明DPO隐式学习token级奖励，是本文的理论基础之一
- **TDPO-R (Shao et al., 2025)**：从时间衰减角度学习DPO，也关注token级差异性

**启发**：token级奖励引导的思路可推广到其他序列级优化问题（如diffusion alignment, code generation），核心insight是"不同位置的决策应有不同的偏离自由度"。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 理论框架新颖，但核心思路（重加权token）并非全新概念
- 实验充分度: ⭐⭐⭐⭐ — 三个benchmark、三个模型、丰富的消融，但缺少更多任务类型验证
- 写作质量: ⭐⭐⭐⭐⭐ — 推导清晰，动机描述充分，理论与实践衔接自然
- 价值: ⭐⭐⭐⭐ — 解决了DPO收敛-性能不一致的实际痛点，实用性strong

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](../../ACL2025/llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [\[NeurIPS 2025\] KL Penalty Control via Perturbation for Direct Preference Optimization](../../NeurIPS2025/llm_alignment/kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](../../ACL2025/llm_alignment/otpo_token_weighting.md)

<!-- RELATED:END -->
