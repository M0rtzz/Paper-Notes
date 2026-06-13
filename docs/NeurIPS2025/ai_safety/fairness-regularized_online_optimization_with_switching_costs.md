---
title: >-
  [论文解读] Fairness-Regularized Online Optimization with Switching Costs
description: >-
  [NeurIPS 2025][AI安全][长期公平正则] 这篇论文把“长期公平”与“动作平滑”第一次严密地放进同一个在线优化框架里，先证明原问题在常规动态基准下根本不可能做好，再提出 FairOBD 通过辅助变量和对偶镜像下降把公平代价在线化，从而在更合理的 $(R,\delta)$ 约束基准上拿到渐近最优级别的竞争比。
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "长期公平正则"
  - "切换代价"
  - "平滑在线优化"
  - "竞争比"
  - "Mirror Descent"
---

# Fairness-Regularized Online Optimization with Switching Costs

**会议**: NeurIPS 2025  
**arXiv**: [2512.11131](https://arxiv.org/abs/2512.11131)  
**代码**: 无  
**领域**: AI安全 / 在线优化 / 公平性调度  
**关键词**: 长期公平正则, 切换代价, 平滑在线优化, 竞争比, Mirror Descent

## 一句话总结

这篇论文把“长期公平”与“动作平滑”第一次严密地放进同一个在线优化框架里，先证明原问题在常规动态基准下根本不可能做好，再提出 FairOBD 通过辅助变量和对偶镜像下降把公平代价在线化，从而在更合理的 $(R,\delta)$ 约束基准上拿到渐近最优级别的竞争比。

## 研究背景与动机

在线优化里有两类约束在现实系统中特别常见。

一类是即时效率相关的，例如电费、时延、利用率、负载不均衡惩罚。

另一类是动作变化相关的，也就是 switching cost。

在电网调度、数据中心资源供给、机器人控制、对象跟踪这些系统里，动作每一步剧烈摆动通常都不好。

因此平滑在线优化已经形成了比较成熟的理论分支。

但现实应用并不只要求“便宜”和“平稳”。

越来越多场景还要求长期公平。

例如跨地域 AI 数据中心调度时，电力使用会把污染和公共健康风险分摊到不同地区。

如果算法只顾最低电价，某些地区就可能长期承担更高的外部性成本。

过去关于公平在线优化的工作，通常把公平写成长期正则项或长期约束。

问题在于，这类方法普遍依赖一个关键假设：每一轮的目标函数只依赖当前动作，不依赖历史动作。

一旦加入 switching cost，这个假设立刻失效。

因为你此时的决策代价天然和上一时刻动作绑定在一起。

这造成了本文最核心的理论困难。

长期公平项本来就要看整段轨迹结束后才能评价。

切换代价又让每轮代价函数带上记忆。

于是你既不能像传统平滑在线优化那样只和当前 hitting cost 博弈，也不能像传统公平在线分配那样简单地用对偶变量把长期预算转成逐轮独立优化。

作者因此提出两个基础问题。

第一，这个问题在标准动态 benchmark 下到底可不可能获得有意义的 regret 或 competitive ratio。

第二，如果标准 benchmark 根本不合适，那应该如何换一个更合理、又不至于失去理论价值的基准。

本文的贡献正是沿着这个顺序展开。

先做 hardness result，再提出算法与新 benchmark。

## 方法详解

### 整体框架

论文考虑的是 fairness-regularized smoothed online convex optimization。

在每个时间步 $t$，算法观察到 hitting cost $f_t(\cdot)$ 和上下文矩阵 $A_t$，然后做出不可撤销动作 $x_t$。

总代价由三部分组成。

一部分是 hitting cost，也就是当下成本。

一部分是 switching cost，即 $d(x_t, x_{t-1})=\frac{\beta_1}{2}\|x_t-x_{t-1}\|^2$。

最后一部分是长期公平代价 $g(\frac{1}{T}\sum_{t=1}^T A_t x_t)$。

这里的 $A_t x_t$ 可以理解为“该动作对各个公平主体产生的代价向量”。

例如在数据中心案例里，它是各地的健康风险暴露。

FairOBD 的总体思路分三步。

第一步，用辅助变量 $z_t$ 把原本只在全局 horizon 上可计算的公平项拆成逐轮形式。

第二步，引入对偶变量 $\kappa_t$，把长期公平约束变成每一轮动作上的线性正则。

第三步，在在线选择动作时，不只最小化 hitting cost，还同时平衡 switching cost 和公平正则，再用 mirror descent 更新 $\kappa_t$。

### 关键设计

1. **不可能性结果先行**

    - 功能：确认问题本身的理论边界，避免在错误 benchmark 上浪费时间。
    - 核心思路：作者证明即便把 switching cost 去掉，任何在线算法相对离线最优都会有常数级最坏情况 cost gap，也即总成本下的线性 regret；竞争比下界甚至达到 $\Omega(T)$。
    - 设计动机：这一步非常关键。它说明“长期公平正则 + 动态离线最优”这个组合本身就太强，普通 benchmark 会把任何在线算法都判死刑。

2. **$(R,\delta)$-OPT 基准**

    - 功能：给离线 benchmark 加一个合理但不失现实性的结构约束。
    - 核心思路：把整个 episode 切成长度为 $R$ 的 frame，要求离线最优在每个 frame 里的平均公平向量，与全局平均公平向量之间的累计偏差不超过 $\delta$。
    - 直觉上，这等价于限制“离线最优能否随时间疯狂改变长期公平目标方向”。
    - 设计动机：如果离线最优每个时段都能切换到完全不同的公平方向，那么在线算法永远追不上；而真实系统里，很多 workload 具有周期性，例如昼夜波动，因此这种约束并不牵强。

3. **辅助变量分解长期公平项**

    - 功能：把原本只有在 $T$ 步结束后才能算的公平代价转成逐轮可优化对象。
    - 核心思路：引入 $z_t$ 作为公平预算，并要求长期上满足 $\sum_t A_t x_t = \sum_t z_t$。这样公平项就能写成逐轮 $g(z_t)$ 的和，再通过 Lagrangian 放松来在线处理。
    - 设计动机：这是把长期正则“拆开”的关键。如果没有 $z_t$，算法只能盲着优化一个未来才知道值的目标。

4. **FairOBD 的平衡下降更新**

    - 功能：同时兼顾 hitting、switching 和 fairness 三项代价。
    - 核心思路：在每一轮，FairOBD 求解一个联合目标：
     - 当前 hitting cost $f_t(x_t)$
     - 带权 switching cost $\lambda_1 d(x_t,x_{t-1})$
     - 朝向 hitting minimizer $v_t$ 的二次正则 $\frac{\lambda_2}{2}\|x_t-v_t\|^2$
     - 公平正则项 $\kappa_t \cdot A_t x_t$
    - 同时，$z_t$ 通过最小化 $g(z_t)-\kappa_t z_t$ 得到。
    - 然后用镜像下降按 $d_t=z_t-A_t x_t$ 更新 $\kappa_t$。
    - 设计动机：单独最小化 hitting cost 会把动作拉向即时最优，单独最小化 fairness 会牺牲系统成本，而 switching cost 会要求动作别乱跳。FairOBD 的名字本身就说明它是 balanced descent。

5. **新的证明技巧**

    - 功能：解决 prior work 证明里“每轮目标函数必须独立”的隐藏前提。
    - 核心思路：作者在附录里引入一个中间 benchmark，用它桥接 FairOBD 与离线最优，使两边在比较时拥有对齐的 per-round objective。
    - 设计动机：这是论文真正硬核的部分。没有这个桥，带记忆的 switching cost 会让传统 primal-dual 或 mirror descent 分析直接断掉。

### 损失函数 / 训练策略

这篇论文不是学习模型训练论文，而是在线算法论文，因此更准确地说是“决策更新规则”而不是损失函数训练策略。

关键超参包括 $\lambda_1$、$\lambda_2$、mirror descent 学习率 $\eta$，以及参考函数 $h(\cdot)$。

当 $h(\kappa)=\frac{1}{2}\|\kappa\|^2$ 时，对偶更新就退化为最熟悉的梯度式更新。

理论上，作者给出 $\eta=\mathcal{O}(T^{-1/3})$ 的设置。

在无 switching cost 时，FairOBD 对 $(R,\delta)$-OPT 的渐近竞争比可以做到 1。

在有 switching cost 时，若适当选择 $\lambda_1$ 和 $\lambda_2$，渐近竞争比达到

$\frac{1}{2}(1+\sqrt{1+4\beta_1/m})$。

这个结果与不带公平项的经典平滑在线优化最优界同阶，说明长期公平并没有从根本上恶化最优渐近阶。

## 实验关键数据

### 主实验

实验场景是跨七个数据中心的 AI 推理负载调度。

作者使用一周的归一化 LLM inference request trace，把需求路由到 Arizona、Iowa、Illinois、Texas、Virginia、Washington、Wyoming 七个数据中心。

代价由电费、负载不均衡、配置切换以及长期公共健康公平代价构成。

比较对象包括：

OPT，最强离线最优。

FairOPT，只顾公平的离线最优。

HITMIN，只顾 hitting cost 的在线算法。

ROBD，不考虑公平的平滑在线优化强基线。

DMD，只顾长期公平、忽略 switching memory 的经典对偶镜像下降。

FairOBD 则是本文方法。

| 方法 | Hitting Cost | Switching Cost | Fairness Cost | Total Cost | 解读 |
|------|-------------|----------------|---------------|------------|------|
| OPT | 168.88 | 22.92 | 20.33 | 212.13 | 最强离线下界，在线算法不可能超过 |
| FairOPT | 177.20 | 351.49 | 16.67 | 545.35 | 公平最好，但切换极差 |
| HITMIN | 159.75 | 43.75 | 70.06 | 273.56 | 只顾即时成本，公平崩溃 |
| ROBD | 163.63 | 23.16 | 55.93 | 242.71 | 平滑很好，但缺乏长期公平意识 |
| DMD | 169.46 | 93.53 | 27.08 | 290.07 | 公平改善了，但切换过大 |
| **FairOBD** ($\eta=10^{-2}$) | **169.39** | **25.67** | **21.35** | **216.41** | 在线方法中总成本最低 |

这个表特别能说明问题。

FairOPT 看似最公平，但 switching cost 非常夸张，说明只优化长期公平会把系统搞得很不平稳。

HITMIN 则正好相反，hitting cost 最低，却把公平成本推到 70.06。

ROBD 的切换代价很好，但公平代价 55.93，说明经典平滑在线优化无法自动“顺带”学会公平。

FairOBD 的价值就在这里。

它没有把任何单项做到极端最优，但把三项代价拉回一个更合理的 Pareto 面上，因此总成本最好。

### 消融实验

论文没有做常见深度学习式模块消融，但做了对学习率稳健性的验证，这对在线算法反而更关键。

| FairOBD 学习率 | Hitting Cost | Switching Cost | Fairness Cost | Total Cost | 观察 |
|---------------|-------------|----------------|---------------|------------|------|
| $\eta=10^{-2}$ | 169.39 | 25.67 | 21.35 | 216.41 | 最佳默认配置 |
| $\eta=10^{-3}$ | 167.80 | 27.52 | 25.52 | 220.84 | 总成本略升，仍优于其他在线基线 |
| $\eta=10^{-4}$ | 167.47 | 28.17 | 26.34 | 221.97 | 更保守更新，公平成本略变差 |

从这个结果能看出两点。

第一，FairOBD 对学习率并不脆弱。

第二，学习率变小时，hitting cost 会稍低，但 fairness cost 会反弹，说明对偶变量更新变慢，长期公平预算跟不上真实环境变化。

换句话说，论文中的理论项并不只是漂亮公式，它直接解释了实验现象。

| 理论结论 | 条件 | 结果 | 含义 |
|---------|------|------|------|
| Theorem 4.1 | 无 switching cost | 最坏情况下仍有常数级 cost gap | 总成本视角下是线性 regret |
| Theorem 4.2 | 无 switching cost | 竞争比下界 $\Omega(T)$ | 标准动态 benchmark 不可用 |
| Theorem 5.1 | $\beta_1=0$ | 渐近竞争比 1 | 公平项可被在线化处理 |
| Theorem 5.2 | 有 switching cost | 渐近竞争比 $\frac{1}{2}(1+\sqrt{1+4\beta_1/m})$ | 与经典平滑在线优化同阶 |

### 关键发现

- 这篇论文最强的一点是先证明“原问题没法做”，再给出新 benchmark 和新算法，而不是默认原 benchmark 合理。
- 在数据中心案例里，公平和切换平滑确实相互冲突，FairOPT 与 HITMIN 一左一右地展示了这种冲突。
- FairOBD 在线方法里 fairness cost 最低，同时 switching overhead 仍然很低，说明辅助变量分解和对偶更新没有把系统搞得抖动失控。
- ROBD 和 DMD 各自只解决一半问题，FairOBD 的价值就在于把“长期公平”和“动作记忆”统一起来。

## 亮点与洞察

- **把“不可能性证明”作为方法设计前置条件**。这比直接报一个新算法更扎实，因为它先说明旧 benchmark 本身就不适合这个问题。
- **$(R,\delta)$ 基准非常有启发性**。很多在线系统里，真正有意义的离线比较对象本来就不该毫无结构约束，尤其当长期正则项本质上依赖全局轨迹时。
- **辅助变量 $z_t$ 的视角很实用**。把公平理解成逐轮“预算分配”而不是最后统一结算，更容易迁移到配额、碳排、风险暴露等其他长期约束场景。
- **FairOBD 的理论结果没有因为加公平项而降阶**。这是很漂亮的结论，说明公平并不一定意味着理论性能不可接受，关键是建模方式对不对。
- **论文把 AI 社会责任问题落到了一个可计算、可证明、可部署的形式**。这比泛泛谈 fairness 更有工程价值。

## 局限与展望

首先，论文的 switching cost 采用的是平方 $L_2$ 形式。

这在理论上很常见，也利于强凸分析，但真实系统中的迁移成本、迁移时延、冷启动开销未必长这样。

其次，$(R,\delta)$ 基准虽然合理，但也意味着结果依赖于“环境不是无限恶意”的结构假设。

如果面对高度非周期、非平稳、突发式上下文切换的场景，$\delta$ 可能很大，这时理论保证会变弱。

第三，实验只展示了一个应用域，即地理分布式数据中心调度。

尽管这个场景很贴切，但若能补充电网、物流、医疗资源分配等案例，会让方法更有说服力。

第四，算法没有利用预测信息。

现实里往往能拿到 workload forecast、价格预测或天气信息，如何把“不完全可信的预测”并入 FairOBD，是很自然的下一个方向。

最后，论文主要研究 worst-case 性能。

如果放宽到 stochastic 或 semi-predictive setting，也许还能进一步缩小和离线最优的差距。

## 相关工作与启发

- **vs 经典 smoothed online optimization**：后者善于处理 hitting 与 switching 的平衡，但默认长期公平不存在；本文等于是给这条线补上了社会责任维度。
- **vs DMD 一类公平在线分配算法**：这类方法能处理长期公平预算，但要求每轮目标不带历史记忆；本文真正突破的是把 switching memory 合法地塞进证明链条。
- **vs 强化学习式长期约束方法**：RL 能建模长期目标，但常缺 worst-case 保证；FairOBD 则在对抗性设置下给出明确界。
- **对我自己的启发**：做公平性系统时，不要默认把 fairness 当作一个后处理指标。只要长期公平写进目标函数，它会从根子上改变在线优化问题的 benchmark 和算法设计。
- **可迁移方向**：碳排公平、医疗资源分配、公平缓存替换、跨区域边缘推理等问题，都可以参考“辅助变量 + 受约束动态 benchmark”的建模方法。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题设定、hardness 证明、$(R,\delta)$ benchmark 和 FairOBD 共同构成了一套完整新框架。
- 实验充分度: ⭐⭐⭐☆ 理论很强，实验场景有代表性，但应用域仍偏单一。
- 写作质量: ⭐⭐⭐⭐☆ 逻辑严密，理论链条完整，不过对不熟悉 online convex optimization 的读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 这是把公平性真正做成“可优化对象”的工作，对 AI 基础设施和负责任调度很有长期价值。
---
title: >-
  [论文解读] Fairness-Regularized Online Optimization with Switching Costs
description: >-
  [NeurIPS 2025][AI安全][公平性正则化] 本文研究在线凸优化中同时处理长期公平性正则项和切换代价的新问题，先证明了任何在线算法都无法获得次线性遗憾或有限竞争比的不可能性结果，然后提出FairOBD算法通过引入辅助变量分解长期公平代价，在(R,δ)-约束基准下实现了渐近最优竞争比。
tags:
  - NeurIPS 2025
  - AI安全
  - 公平性正则化
  - 在线凸优化
  - 切换代价
  - 竞争比
  - 镜像下降
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization](stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)
- [\[AAAI 2026\] Alternative Fairness and Accuracy Optimization in Criminal Justice](../../AAAI2026/ai_safety/alternative_fairness_and_accuracy_optimization_in_criminal_j.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICML 2026\] COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](../../ICML2026/ai_safety/copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)

</div>

<!-- RELATED:END -->
