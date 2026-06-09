---
title: >-
  [论文解读] On the Coordination of Value-Maximizing Bidders
description: >-
  [ICML 2026][价值最大化] 本文形式化研究了在线广告中多个 value-maximizing 自动出价者的"协调"问题，提出"只让联盟中价值最高的成员出价、其余出 0"的简单协调机制，并证明对一大类自动出价算法而言…
tags:
  - "ICML 2026"
  - "价值最大化"
  - "自动出价"
  - "第二价拍卖"
  - "RoS 约束"
  - "协调机制"
---

# On the Coordination of Value-Maximizing Bidders

**会议**: ICML 2026  
**arXiv**: [2511.04993](https://arxiv.org/abs/2511.04993)  
**代码**: 无  
**领域**: 在线广告 / 自动出价 / 机制设计  
**关键词**: 价值最大化、自动出价、第二价拍卖、RoS 约束、协调机制

## 一句话总结
本文形式化研究了在线广告中多个 value-maximizing 自动出价者的"协调"问题，提出"只让联盟中价值最高的成员出价、其余出 0"的简单协调机制，并证明对一大类自动出价算法而言，该机制能同时降低每个联盟成员的 RoS 违反量、并把联盟总价值推到所有协调机制的渐近最优。

## 研究背景与动机

**领域现状**：现代搜索 / 信息流广告的核心范式是 auto-bidding：广告主把"在 RoS（Return-on-Spend，每花一块钱至少带回一块钱价值）约束下最大化总价值"的优化任务交给平台或第三方代理，由算法以镜像下降、对偶等手段在线学习每轮出价。绝大多数文献都默认每个 bidder 独立优化自己的目标。

**现有痛点**：现实里这个独立性假设很脆弱。第三方代理常常同时托管几十个广告主，大电商（Amazon、Temu、Shein 之流）也常拿同一个 portfolio 投多条相似广告。这种"同根"bidders 互相在同一个 ad slot 上抬价，会把成交价推高、把 RoS 约束打穿，对联盟整体是负和的。但 auto-bidding 文献对"协调"几乎没有理论刻画——只有少数实证或 utility-maximizer 的工作（如 Decarolis 等、Romano 等、Chen 等）。

**核心矛盾**：value maximizer 的行为与 utility maximizer 不一样。前者会主动 overbid（典型 $b_{i,t}=(1+1/\lambda_{i,t}) v_{i,t} > v_{i,t}$）来抢量，因此联盟内"互相 overbid"造成的损害远比经典 cartel 分析里更隐蔽，需要重新建模。

**本文目标**：分解为三个子问题——(i) 一个简单的协调机制能不能严格优于独立出价？(ii) 这个"优于"对哪些算法、什么分布条件成立？(iii) 在算法已知的最优率意义下，它能不能达到最优？

**切入角度**：作者抓住一个朴素观察：在 second-price 拍卖里，如果联盟内的成员只让"当前价值最高的那一个" $i^* = \arg\max_i v_{i,t}$ 出价、其他成员沉默，则联盟事实上把"内部互卷"完全消除——只剩这一个代表去面对外部 $d_t^O$。这把 N 个 bidder 的耦合动力学塌缩到一个等价单 bidder 问题，给后续证明留出了空间。

**核心 idea**：用"最高价值者独占出价"的协调机制取代独立出价，从机制 + 分布两端同时拿到 RoS 改善（条件式）与价值改善（无条件，针对 mirror-descent 类算法）。

## 方法详解

### 整体框架
舞台是 $T$ 轮 second-price 重复拍卖，联盟 $N$ 个 bidders 每轮 $t$ 从同一连续分布 $F$ 独立采样价值 $v_{i,t}\in[0,B]$，并面对一个外部竞争出价 $d_t^O\sim D$（也可吸收保留价）。bidder $i$ 的对手出价为 $d_{i,t}=\max\{d_t^O, \max_{j\neq i} b_{j,t}\}$，胜出指示 $x_{i,t}=\mathbb{I}\{b_{i,t}\ge d_{i,t}\}$，效用 $u_{i,t}=x_{i,t}(v_{i,t}-d_{i,t})$。目标是 $\max \sum_t v_{i,t} x_{i,t}$ s.t. $\sum_t u_{i,t}\ge 0$ （RoS 约束）。

两个被对比的协议：
- **独立出价 (Alg 1)**：每个 bidder 各跑自己的算法 $A(H_{i,t})$，互相竞价。
- **协调出价 (Alg 2)**：只让 $i^*=\arg\max_i v_{i,t}$ 出 $b_{i^*,t}=A(H_{i^*,t})$，其它人出 0。

整套理论拆为两条主线：第 3 节谈 RoS（效用）改善需要的充分必要分布条件；第 4 节谈在 mirror-descent 算法下的（无条件）价值改善与最优性；第 5 节扩展到非 i.i.d. 值。

### 关键设计

**1. 最高价值者独占出价（HVO）机制：用最小协调消除联盟内部互卷**

联盟里的痛点是同根 bidders 在同一 slot 上互相抬价，把成交价推高、RoS 约束打穿，对整体是负和的。作者抓住一个朴素观察：在 second-price 拍卖里，只让当前价值最高的成员 $i^*=\arg\max_i v_{i,t}$ 出价 $b_{i^*,t}=A(H_{i^*,t})$、其余人一律出 0，就能把内部竞价彻底剔除。机制本身极简——中心规划者每轮读出 $\{v_{i,t}\}$，挑出 $i^*$ 让它的自动出价算法照常运行即可。它之所以有效，是因为其余 $N-1$ 人不再贡献对手出价 $d_{i^*,t}=\max\{d_t^O,\max_{j\neq i^*}b_{j,t}\}$ 里的 $\max_{j\neq i^*}b_{j,t}$ 项，于是 $i^*$ 面对的对手价直接退化成外部价 $d_t^O$，省下大量 second-price payment。second-price 是 DSIC，老实出 $v$ 本是 baseline，但 value-maximizer 会主动 overbid（典型 $b_{i,t}=(1+1/\lambda_{i,t})v_{i,t}>v_{i,t}$）来抢量——这正是它们在独立竞价里互伤的根源，HVO 把这个"互伤项"剔掉，且除价值之外不需要成员间交换任何私密策略。

**2. 充要分布条件 Assumption 3.1：刻画协调"何时"真的降低 RoS 违反**

HVO 不是无条件更优，所以作者得给出"协调能严格降低每个 bidder RoS 违反量"的精确边界。令 $v_{(N)},v_{(N-1)}$ 为 $N$ 个 i.i.d. 价值的最大与次大，定义

$$\Delta := \mathbb{E}_{F,D}\big[(v_{(N-1)}-d^O)_+ - (d^O - v_{(N)})_+\big]\ge 0,$$

直观就是"次大值压过外部价的优势"应大于"外部价压过最大值的优势"——即联盟内部两人都足够强，能压住外部。作者给了两个落地的算例：$N=4,F=D=U[0,1]$ 时 $\Delta=1/6$；$N=3,F=U[0,1],D=\mathrm{Beta}(3,2)$ 时 $\Delta=1/40$，并证明任意全支撑 $F,D$ 在 $N$ 充分大时都满足。结论靠两条引理拼成：Lemma 3.1（second-price DSIC 给 $U^{\mathrm{Truth}}_i\ge U^{I,A}_i$）+ Lemma 3.2（$\mathbb{E}[U^{C,A}_i]\ge\mathbb{E}[U^{\mathrm{Truth}}_i]+T\Delta/N$）合成 Theorem 3.1：$\Delta\ge 0$ 时 $\mathbb{E}[U^{C,A}_i-U^{I,A}_i]\ge T\Delta/N$，对任何 overbidding 算法都成立；而 $\Delta<0$ 时反向也是紧的——存在 overbidding 算法让协调反而吃亏。这种"既充分又必要"的刻画在 auto-bidding 文献里并不多见。

**3. 协调镜像下降（MD-h）+ 渐近最优性：把价值也推到所有协调机制的最优**

光比效用还不够，平台真正在意的是"花掉的预算赢回多少价值"，所以第三步要在 RoS 条件之上证价值改善与最优性。bidder 用对偶乘子 $\lambda_{i,t}$ 控制 overbid 倍率，$b_{i,t}=(1+1/\lambda_{i,t})v_{i,t}$，看到效用 $g_{i,t}$ 后做 Bregman 投影 $\lambda_{i,t+1}=\arg\min_\lambda\{\alpha g_{i,t}\lambda+D_h(\lambda,\lambda_{i,t})\}$（取熵镜像 $h(\lambda)=\lambda\log\lambda-\lambda$ 时退化为乘性更新 $\lambda_{i,t+1}=\lambda_{i,t}\exp(-\alpha g_{i,t})$）。关键一步是把 $N$-bidder 协调动力学规约成"一个虚拟 bidder 看 $v_{(N)}$"：协调下 $\mathbb{E}[g_{i,t}\mid H_{t-1}]=G_{(N)}(\lambda_{i,t})/N$，其中 $G_{(N)}(\lambda)=\mathbb{E}[(v_{(N)}-d^O)\mathbb{I}[(1+1/\lambda)v_{(N)}>d^O]]$ 单调递增，于是活跃 bidder 的 $\lambda$ 收敛到零点 $\lambda_\star=\inf\{\lambda:G_{(N)}(\lambda)\ge 0\}$、总价值收敛到单 bidder 等价问题的 $V_{(N)}(\lambda_\star)$（Theorem 4.1）。再用 Lagrange envelope 把独立出价的总价值上界也压到同一个 $V_{(N)}(\lambda_\star)$，配合 Assumption 3.1 下 $\lambda^C_{i,t}\to 0$，反过来证明协调 MD 也优于任何其他协调机制（Theorem 4.2）。这个把多代理学习接回成熟单 bidder RoS 镜像下降理论的 reduction，是同时撑起价值改善和最优性两个定理的关键。

### 损失函数 / 训练策略
没有训练目标这种说法；核心"在线优化"就是 MD-h 在每轮看完 $g_{i,t}$ 后更新 $\lambda_{i,t}$，学习率 $\alpha=1/\sqrt T$。Feng et al. (2023) 的同款算法已知 $O(\sqrt T \log T)$ RoS 违反、$O(\sqrt T)$ 价值损失界，被作者直接复用。

## 实验关键数据

### 主实验
作者在合成数据（对称 / 非对称分布，$N\in\{2,3,4,5\}$，$T\in\{4000,10000,20000\}$）与公开的 iPinYou Season 2（55 个广告主、2.5M 拍卖记录）上对比独立 vs 协调，每个设置跑 100 次取均值。下表是论文 Table 1（按 $T$ 归一）。

| 设置 | $N$ | $T$ | Util (I) | Util (C) | Value (I) | Value (C) |
|------|-----|-----|----------|----------|-----------|-----------|
| i.i.d. $U[0,1] / U[0,0.9]$ | 2 | 4000 | -0.011 | **0.220** | 0.643 | **0.666** |
| i.i.d. $U[0,1] / U[0,1]$ | 4 | 4000 | -0.077 | **0.302** | 0.774 | **0.800** |
| i.i.d. $U[0,1] / \mathrm{Beta}(3,2)$ | 3 | 4000 | -0.049 | **0.153** | 0.712 | **0.748** |
| 非 i.i.d. | 5 | 20000 | -0.062 | **0.619** | 0.814 | **0.819** |
| iPinYou 实数据 | 4 | 20000 | -0.040 | **0.155 ± 0.012** | 0.620 ± 0.016 | **0.928 ± 0.003** |
| iPinYou 实数据 | 5 | 20000 | -0.065 | **0.172 ± 0.012** | 0.608 ± 0.012 | **0.958** |

独立出价的人均效用几乎都是负的（RoS 被打穿），协调后立刻拉到 +0.15 ~ +0.62；价值在合成数据上提升 2%-4%，在真实 iPinYou 上从 0.62 跳到 0.93（+50%），这是因为真实数据外部价分布更厚尾，HVO 节省的 second-price payment 更夸张。

### 消融实验
论文没有典型 ablation，但通过 i.i.d. / 非 i.i.d. + 不同 $D$ 的对照可以读出："独占出价"机制本身贡献的部分。

| 配置 | 效用变化 | 价值变化 | 说明 |
|------|---------|---------|------|
| i.i.d. 对称分布 | -0.05 → +0.22 | +2~4% | Theorem 3.1 + 4.1 同时生效，最干净 |
| 非 i.i.d. 弱对称 | -0.01 → +0.22 | +0.4% | Theorem 5.1/5.2 生效但价值提升被吃掉 |
| iPinYou 真实数据 | -0.04 → +0.15 | +50% | 外部价厚尾时 HVO 省下的 payment 极多 |
| Assumption 3.1 不成立（理论） | — | — | 反例显示协调反而吃亏 |

### 关键发现
- HVO 在所有 6 个合成 + 2 个真实设置里**无一例外**地把独立模式下的负效用扳正，验证了 Theorem 3.1 的 $T\Delta/N$ 改进量的非平凡性。
- 价值提升的幅度由外部分布的**尾部**决定：均匀分布下只有 2%-4%，iPinYou 厚尾下高达 50%。这与 Theorem 4.1 中"协调收敛到 $V_{(N)}(\lambda_\star)$"的分析一致——尾部越厚，单 bidder 的渐近 $\lambda_\star$ 越接近 0，overbid 倍率越大。
- 非 i.i.d. 时虽然个别 bidder 的价值改善不再保证，但联盟总价值仍持续优于独立模式，符合 Theorem 5.2 在 Assumption 5.1 下的预测。

## 亮点与洞察
- "最高价值者独占出价"是 minimal 的可执行协调协议：不需要传递私有效用函数、不需要 side payment、不需要复杂的拍卖机制重设计，对现有第三方代理平台几乎零改造成本就能上线。
- Assumption 3.1 给出的 $\Delta\ge 0$ 是干净的"充要条件"，并且能用 Azuma 不等式升到 $1-\exp(-T\Delta^2/(32B^2N^2))$ 的高概率保证，工程上方便做 A/B 触发判定。
- 把 $N$-bidder 协调系统规约到 $v_{(N)}$ 的虚拟单 bidder 问题，这个 reduction 既证 Theorem 4.1 也证 Theorem 4.2，是非常 reusable 的分析模式，未来分析更复杂的协调机制（如部分协调）有望复刻。

## 局限与展望
- 假设外部出价是 i.i.d. 的，但实际平台里"外部"也是一群 auto-bidder，会随联盟行为反应；作者也把"auto-bidding 外部 bidder"列为开放问题。
- 协调机制依赖中心规划者能读到所有 $v_{i,t}$，在多家广告主托管同一代理的场景合理，但跨广告主联盟需要隐私保护（论文未涉及）。
- 只覆盖 second-price 拍卖；first-price、GSP 这种 non-truthful 拍卖下 truthful-bid 引理失效，Lemma 3.1 走不通，需要重新构造比较项。
- Assumption 3.1 不成立时，文中只给反例说明协调会失败，但没有给"哪种协调机制更稳"的设计指引——非平衡市场仍是开放问题。

## 相关工作与启发
- **vs Decarolis et al. (2020) / Romano et al. (2022)**: 他们分析 utility-maximizer 在 GSP/VCG 下的协调收益与计算复杂度，本文专攻 value-maximizer + RoS + 重复 second-price，结论的"overbid 是核心"完全是新机制特征。
- **vs Chen et al. (2023)**: Chen 等讨论的是 budget 约束下的协调动态出价，本文换成 RoS 约束并给出充要条件式的 RoS 改进定理；两条线在 budget vs RoS 这个 value-maximizer 的两大约束上互补。
- **vs Feng et al. (2023) / Balseiro et al. (2023)**: 他们是单 bidder 的 mirror-descent RoS 自动出价基线，本文复用其学习率与 regret 界，把同样的算法插进协调机制后证明总价值最优——是非常自然的"算法不动、机制变"的扩展。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions](../../ICML2025/others/learning_safe_strategies_for_value_maximizing_buyers_in_uniform_price_auctions.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](../../ICLR2026/others/key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)
- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[ECCV 2024\] Gaze Target Detection Based on Head-Local-Global Coordination](../../ECCV2024/others/gaze_target_detection_based_on_head-local-global_coordination.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)

</div>

<!-- RELATED:END -->
