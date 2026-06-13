---
title: >-
  [论文解读] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach
description: >-
  [ICML 2026][优化/理论][$(L_0] 本文指出文献中标准一阶 / 二阶 SDE 在 $(L_0,L_1)$-光滑下完全错失学习率稳定性约束（甚至预测发散区间也收敛），通过在漂移项中把曲率项符号翻正，作者构造出一族"稳定性忠实"的一阶弱近似 SDE，首次在统一框架内分析 DCSGD 与 DSignSGD 在压缩 + 仿射方差 + 重尾噪声下的收敛性，并给出归一化强度该如何选取的具体处方。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "$(L_0"
  - "L_1)$-光滑"
  - "分布式压缩 SGD"
  - "SignSGD"
  - "重尾噪声"
  - "稳定性修正 SDE"
---

# On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach

**会议**: ICML 2026  
**arXiv**: [2506.00181](https://arxiv.org/abs/2506.00181)  
**代码**: 待确认  
**领域**: 优化 / 分布式 SGD / SDE 连续时间分析  
**关键词**: $(L_0,L_1)$-光滑, 分布式压缩 SGD, SignSGD, 重尾噪声, 稳定性修正 SDE

## 一句话总结
本文指出文献中标准一阶 / 二阶 SDE 在 $(L_0,L_1)$-光滑下完全错失学习率稳定性约束（甚至预测发散区间也收敛），通过在漂移项中把曲率项符号翻正，作者构造出一族"稳定性忠实"的一阶弱近似 SDE，首次在统一框架内分析 DCSGD 与 DSignSGD 在压缩 + 仿射方差 + 重尾噪声下的收敛性，并给出归一化强度该如何选取的具体处方。

## 研究背景与动机

**领域现状**：分布式随机梯度法是现代大模型训练的主力，其行为同时受三件事支配——批噪声（甚至是 Simsekli et al. 2019 观察到的重尾噪声）、通信压缩（量化 / 稀疏化 / sign）、自适应归一化（Adam、AdaGrad、SignSGD 等）。SDE 已经被广泛用作离散优化器的连续时间代理（Li et al. 2017 起家），用来研究学习率调度、batch size 调度、scaling law、隐式正则等。

**现有痛点**：(1) 几乎所有 SDE 分析都基于全局 $L$-光滑，而真实 DL 损失更接近 $(L_0,L_1)$-光滑（即 $\|\nabla^2 f(x)\|_2\leq L_0+L_1\|\nabla f(x)\|_2$，Zhang et al. 2020b），后者甚至不保证存在一个对所有初始化都稳定的常数步长；(2) 噪声常被假设 Gaussian 或 bounded variance，而仿射方差和重尾才是 modern DL 的常态；(3) DCSGD（带无偏压缩的分布式 SGD）和 DSignSGD 在 $(L_0,L_1)$ + 压缩 + 一般噪声三者同时存在时缺乏统一刻画。

**核心矛盾**：本文最关键的发现是，**经典的一阶 SDE 和二阶 SDE 在 $(L_0,L_1)$-光滑下根本无法预言离散步长稳定性**——一阶 SDE 给不出任何对学习率的限制，二阶 SDE 更糟，反而在 GD 实际发散的大步长区间预言"加速收敛"。一维抛物 $f(x)=\lambda x^2/2$ 时，GD 要求 $\eta<2/\lambda$ 才稳定，但经典一阶 ODE 解为 $f(X_t)=f(X_0)e^{-2\lambda t}$ 跟 $\eta$ 完全没关系；经典二阶 ODE 给出 $f(X_t)=f(X_0)e^{-2\lambda(1+\lambda\eta/2)t}$ 反而说 $\eta$ 越大收敛越快。对四次 $f(x)=x^4/4$（典型非 $L$-光滑、$(L_0,L_1)$-光滑）问题更夸张：根本不存在统一常数步长。

**本文目标**：(1) 形式化指出经典 SDE 错在哪；(2) 推一个**符号翻正的修正一阶 SDE**，作为 SGD 在 $(L_0,L_1)$ 下的弱近似；(3) 用它统一分析 DCSGD（仿射方差 + 压缩）和 DSignSGD（重尾 student-$t$ 噪声）；(4) 给出 normalization 强度的实操指南。

**切入角度**：作者借用物理学的"ansatz"思路——直接猜测一族带可调系数 $\alpha$ 的漂移修正 ODE $dX_t=-\nabla f(X_t)dt+\alpha\eta\nabla^2 f(X_t)\nabla f(X_t)dt$，然后通过让"诱导的损失漂移 $df(X_t)/dt$"与"离散 Taylor 展开 $(f(x_{k+1})-f(x_k))/\eta$ 到 $O(\eta)$ 阶"匹配，**唯一**确定出 $\alpha=1/2$，并且符号是 **+**（不是经典文献的 −）。

**核心 idea**：传统二阶 SDE 把曲率项符号搞反了，恰好是它在 $(L_0,L_1)$-光滑下"错预言加速收敛"的根源；只要把符号改成 +，得到的一阶 SDE 就能在抛物 / 四次 / 高维带噪 / 带压缩 / 带 sign 自适应等所有情形下回收正确的稳定性阈值。

## 方法详解

### 整体框架

文章不提新的优化器，而是提供一套"稳定性忠实的连续时间替身"。研究对象是两个分布式优化器：

- **DCSGD**（无偏压缩 SGD）：$x_{k+1}=x_k-\frac{\eta\eta_k}{N}\sum_{i=1}^N \mathcal{C}_{\xi_i}(\nabla f_{i,\gamma_i}(x_k))$，压缩器 $\mathcal{C}_\xi$ 满足 $\mathbb{E}[\mathcal{C}_\xi(x)]=x$ 且 $\mathbb{E}\|\mathcal{C}_\xi(x)-x\|_2^2\leq\omega\|x\|_2^2$。
- **DSignSGD**：$x_{k+1}=x_k-\frac{\eta\eta_k}{N}\sum_{i=1}^N \operatorname{sign}(\nabla f_{i,\gamma_i}(x_k))$，等价于每客户端做 1-bit 量化。

损失 $f(x)=\frac{1}{N}\sum_j f_j(x)$ 假设 $(L_0,L_1)$-光滑；客户端梯度噪声 $\nabla f_{i,\gamma_i}(x)=\nabla f(x)+Z_i(x)$ 假设坐标对称分布——DCSGD 部分要 $\|\Sigma_i(x)\|_\infty\leq\sigma_{0,i}^2+\sigma_{1,i}^2\|\nabla f(x)\|_2^2$（仿射方差），DSignSGD 部分允许 student-$t_\nu$ 重尾，$\nu=1$ 时甚至期望都无界。

弱近似定义沿用 Milshtein (1986)：$(X_t)$ 是 $(x_k)$ 的 $\alpha$ 阶弱近似当 $\max_k|\mathbb{E}g(x_k)-\mathbb{E}g(X_{k\eta})|\leq C\eta^\alpha$ 对所有多项式增长 $g$ 成立。

### 关键设计

**1. 指出经典 SDE 的失败：用抛物 + 四次两个一维例子戳穿失效模式**

文章先把"为什么非要修正 SDE"讲到无可辩驳。在 $f(x)=\lambda x^2/2$ 上，经典一阶 ODE $dX_t=-\lambda X_t dt$ 给出 $f(X_t)=f(X_0)e^{-2\lambda t}$，跟步长 $\eta$ 完全无关；经典二阶 ODE $dX_t=-\nabla f(X_t)dt-\frac{\eta}{2}\nabla^2 f(X_t)\nabla f(X_t)dt$ 更离谱，预测 $f(X_t)=f(X_0)e^{-2\lambda(1+\lambda\eta/2)t}$——步长越大收敛越快。可离散 GD 明明要求 $\eta<2/\lambda$ 才稳。换到四次 $f(x)=x^4/4$（典型 $(L_0,L_1)$-光滑），GD 一步是 $x_{k+1}=x_k(1-\eta x_k^2)$，要求 $\eta<2/x_k^2$ 依赖当前迭代，所以根本不存在对所有初始化都稳的常数步长；但经典一阶/二阶 ODE 仍预言"全局收敛、$\eta$ 越大越快"。结论很重：在 $(L_0,L_1)$ 下经典 SDE 不是不准，而是质上错误，连无穷小步长都救不回来。这把研究焦点从过去追求的"SDE 阶数更高"切换到"稳定性是否匹配"。

**2. 符号翻正的修正一阶 SDE（ansatz + 漂移匹配）：把曲率项符号从 − 改成 +**

为什么经典 SDE 错？作者把 GD 一步的离散二阶 Taylor 展开写出来：

$$\frac{f(x_{k+1})-f(x_k)}{\eta}=-\|\nabla f(x_k)\|^2+\frac{\eta}{2}\nabla f(x_k)^\top\nabla^2 f(x_k)\nabla f(x_k)+O_{x_k}(\eta^2),$$

二阶项是 **+** 号；而经典二阶 ODE 诱导的损失漂移里这一项却是 **−** 号——符号反了，这正是它在大 $\eta$ 处预言"额外阻尼"而非"额外发散"的根源。于是作者用物理学常用的 ansatz 思路，直接猜一族带可调系数的修正方程 $dX_t=-\nabla f(X_t)dt+\alpha\eta\nabla^2 f(X_t)\nabla f(X_t)dt$，要求它诱导的损失漂移与上面的离散展开匹配到 $O(\eta)$ 阶，唯一解是 $\alpha=1/2$，得到

$$dX_t=-\nabla f(X_t)\,dt+\frac{\eta}{2}\nabla^2 f(X_t)\nabla f(X_t)\,dt+\sqrt{\eta}\sqrt{\Sigma(X_t)}\,dW_t,$$

Theorem C.5 证明它是 SGD 的一阶弱近似。验证立竿见影：抛物上它给出 $f(X_t)=f(X_0)e^{-2\lambda(1-\lambda\eta/2)t}$，仅当 $\eta<2/\lambda$ 收敛，与 GD 严丝合缝；四次上诱导漂移 $df(X_t)=(-X_t^6+\frac{3\eta}{2}X_t^8)dt$ 与离散展开一致，并在 $\eta X_t^2\gtrsim1$ 时正确翻成排斥、损失上升，精确捕捉"可行步长必须随局部梯度尺度收缩"。这背后的方法论是：模型阶数不重要，模型与所研究性质（这里是损失动力学）的匹配度才重要。

**3. 统一收敛定理：DCSGD（仿射方差 + 压缩）与 DSignSGD（重尾）**

有了忠实的 SDE，作者把它套到两个分布式优化器上，给出第一份同时覆盖 $(L_0,L_1)$ + 压缩 + 仿射方差/重尾的收敛保证。Thm 4.2（DCSGD）在 $(L_0,L_1)$-光滑、每客户端 $(\sigma_{0,i}^2,\sigma_{1,i}^2)$-仿射方差、压缩率 $\omega_i$ 下，要求 $\eta\eta_t<\frac{2\epsilon}{G(1+\frac{\bar\omega+d(\overline{\sigma_1^2\omega}+\overline{\sigma_1^2})}{N})+\frac{L_1 d(\overline{\sigma_0^2}+\overline{\sigma_0^2\omega})}{N}}$（$G=L_0+L_1\mathbb{E}\|\nabla f(X_t)\|_2$）即可保证 $\mathbb{E}\|\nabla f(X_{\hat t})\|_2^2\to0$。比经典 SDE 多出来的那个 **+1** 是关键——它在无噪无压缩 $(L_0,L_1)\to L$ 的退化极限下恢复出经典 $\eta\eta_t<2/L_0$，补上了旧 SDE 在该极限下"完全没限制"的硬伤。Thm 4.3（DSignSGD）在 student-$t_\nu$ 重尾下要求 $\eta\eta_t<\ell_\nu/K$、$K=\frac{L_1 d\sigma_{\mathcal{H},1}}{2N}+\sqrt{d}(L_0+L_1)M_\nu$，那个多出来的 $\sqrt{d}(L_0+L_1)M_\nu$ 同样是修正 SDE 才产生的，没有它即便 $\sigma_{\max,i}=0$ 也会错预言"无步长约束"。两条定理一对照就读出一个朴素结论：DSignSGD 因 sign 自带 elementwise 归一化，标准 Robbins–Monro 调度 $\eta_k=1/\sqrt{k+1}$ 就能在 $\nu=1$（期望都无界）下收敛，而 DCSGD 必须按噪声/压缩动态归一化才稳。它把 Khirirat、Faw、Chen、Crawshaw 等多篇的零散结论统一成"何时该归一化、归一化多强"可由 $(L_0,L_1,\bar\omega,\overline{\sigma_1^2},N,d)$ 几个常数读出来的事。

### 损失函数 / 训练策略
不涉及训练 / 损失设计；理论结果建立在连续时间 SDE 上，离散优化器通过弱近似 Def 3.4 与之关联。文献中已有多项实验验证（Compagnoni et al. 2025a/b, Marshall et al. 2025）表明这类 SDE 在真实 DNN / 数据集上能跟踪对应优化器。

## 实验关键数据

实验定位是"机制验证"而非性能竞赛——目的是检验定理预言的不稳定模式 / 稳定阈值能否在真实神经网络上出现。

### 主实验

| 设置 | 配置 | 现象 |
|--------|------|------|
| DCSGD + 无偏稀疏化 | $N=8$ 客户端、MLP、加入 $Z_t\sim\mathcal{N}(0,\sigma^2\|g_t\|_2^2 I)$ 仿射方差噪声 | 常数 $\eta$ 下随压缩率 $\omega$ 增加越发不稳，loss 发散；按 Thm 4.2 的自适应归一化（Eq. 15）后所有 $\omega$ 都收敛 |
| DCSGD vs Plain Normalized SGD | 同上条件 | 朴素 Normalized SGD 比定理处方的自适应归一化更不稳，说明"归一化强度"必须随压缩 / 噪声调 |
| DSignSGD + 重尾 student-$t$ | $\nu=1$（期望无界）、不同尺度 $\sigma$ | 常数 $\eta$ 下保持稳定但**不收敛**；切换到 Thm 4.3 处方的 $\eta_k=1/\sqrt{k+1}$ 后所有 $\sigma$ 下均收敛 |
| ResNet-18 / ViT on CIFAR-10 | $N=8$ 客户端分布式 | Appendix E.4 报告与 MLP 一致的稳定 / 不稳定划分，定性现象在大模型上保持 |

### 消融实验（理论对照表）

文章用 Table 2 对比经典 SDE 与本文 SDE 推出的学习率约束：

| 设置 | 经典 SDE 约束 | 本文修正 SDE 约束 | 关键差别 |
|------|---------------|-------------------|----------|
| DCSGD ($(L_0,L_1)$ + 仿射方差 + 压缩) | $\frac{2\epsilon}{G\frac{\bar\omega+d(\overline{\sigma_1^2\omega}+\overline{\sigma_1^2})}{N}+\frac{L_1 d(\overline{\sigma_0^2}+\overline{\sigma_0^2\omega})}{N}}$ | $\frac{2\epsilon}{G(\bm{1}+\frac{\bar\omega+d(\overline{\sigma_1^2\omega}+\overline{\sigma_1^2})}{N})+\frac{L_1 d(\overline{\sigma_0^2}+\overline{\sigma_0^2\omega})}{N}}$ | 橙色 **+1** 恢复经典 $\eta<2/L_0$ 极限 |
| DSignSGD (重尾) | $\ell_\nu/K,\;K=\frac{L_1 d\sigma_{\mathcal{H},1}}{2N}$ | $\ell_\nu/K,\;K=\frac{L_1 d\sigma_{\mathcal{H},1}}{2N}+\sqrt{d}(L_0+L_1)M_\nu$ | 多出的 $\sqrt{d}(L_0+L_1)M_\nu$ 在无噪极限仍保证非平凡步长约束 |
| 抛物 sanity check $f(x)=\lambda x^2/2$ | 一阶 ODE：$\forall\eta$ 收敛；二阶 ODE：$\eta$ 越大越快 | $f(X_t)=f(X_0)e^{-2\lambda(1-\lambda\eta/2)t}$，仅 $\eta<2/\lambda$ 收敛 | 完全匹配离散 GD |
| 四次 sanity check $f(x)=x^4/4$ | 一阶 ODE：$\forall\eta$ 全局收敛；二阶 ODE：更快 | drift 中带 $+\frac{3\eta}{2}X_t^8$ 项，大 $|X_t|$ 时变排斥 | 正确再现"无统一常数步长" |

### 关键发现
- **DCSGD 的归一化强度由 $(\omega,\sigma_1,L_1,N,d)$ 共同决定**：压缩越激进 / 仿射方差越大 / 维度越高 / 客户端越少，所需归一化越强；定理给出闭式可读取的阈值。
- **DSignSGD 对重尾天然鲁棒**：sign 自带 elementwise 归一化，所以即便噪声期望无界，只要按 $\eta_k=1/\sqrt{k+1}$ 这种标准 Robbins-Monro 调度走就能收敛，这与 DCSGD 必须做依赖梯度范数的归一化形成鲜明对比。
- **更多客户端帮助、更高维度伤害**：约束分母里 $N$ 在分母位置（实质上削弱噪声 / 压缩干扰），$d$ 在分子位置（放大仿射方差影响）；这给"在保持每客户端 batch 不变的前提下加客户端 vs 加 batch"的工程取舍提供了理论依据。
- **经典 SDE 推出的 scaling / scheduling 处方在 $(L_0,L_1)$ 下不可信**：批量调度 $\eta/B(t)$ 类、平方根 scaling 等结论若不带"稳定区间"修正，可能把实际优化器推出稳定域。

## 亮点与洞察
- **"符号翻正"的洞察**：把曲率项从经典文献的 −$\eta/2$ 改成 +$\eta/2$ 看似只是数学游戏，但其实是把 SDE 的 loss drift 与 GD 的离散展开符号对齐的唯一选择，这一发现非常干净且有解释力——它顺手解释了为什么过去几十年用经典 SDE 推 scaling law 时不时与实验对不上。
- **ansatz + 匹配的方法论**：作者用物理学常用的 ansatz 思路而不是从某个高阶展开"推"出 SDE，给"该怎么构造对某种性质忠实的连续时间代理"提供了一个可复制范式——挑你最关心的性质（这里是 loss drift），让它在 $O(\eta)$ 阶上等号即可，不要追求所谓阶数最高。
- **第一份覆盖 $(L_0,L_1)$ + 压缩 + 仿射方差 / 重尾的统一分析**：之前文献要么不带压缩，要么噪声假设很弱，要么不带 $(L_0,L_1)$，这篇是第一次把四件事放进同一框架并给出可读的 stability region。
- **DSignSGD 的"自适应天生抗重尾"**：sign 把所有坐标拉成等长度，让重尾噪声的影响被自然截断，从而无需对学习率做"基于梯度的归一化"，这是对 sign-based 优化器一个被低估的强力优点。

## 局限与展望
- 客户端数据是同构的，不覆盖 heterogeneous federated 场景；error feedback、有偏压缩、去中心化拓扑都不在范围。
- 理论保证只在连续时间 SDE 上，与离散优化器的桥梁是弱近似 Def 3.4——实证上有多项工作支持，但严格意义上"SDE 收敛 ⇒ 离散收敛"仍有 gap。
- 约束式 Eq. 15 / 17 含 $L_0,L_1,\sigma_{0,i},\sigma_{1,i},\mathbb{E}\|\nabla f\|_2$ 等不可直接观测的常数，与经典 $\eta<2/L$ 一样只能用作定性指南，无法作为闭式 tuning rule。
- 实验是机制验证（MLP + 小幅 ResNet/ViT），没有去和 Adam/AdamW 等主流自适应方法正面比；作者也明确说"不是新优化器"。
- 可延伸方向：动量 / NAG / Adam 类的隐式二阶结构在此框架下的对应稳定性修正；以及在该框架下重审 batch-size schedule 和 scaling laws。

## 相关工作与启发
- **vs Li et al. 2017（经典 SDE 框架）**：本文沿用 Milshtein 弱近似定义，但论证了即便 weak approximation 阶数相同，模型对"稳定性"性质的忠实度可以差很多；建议把"对所关心性质忠实"作为选 SDE 的第一原则。
- **vs Zhang et al. 2020b / Crawshaw et al. 2022 / Faw et al. 2023 / Chen et al. 2023**：这些都在离散时间研究 $(L_0,L_1)$ 下 normalized SGD / AdaGrad / SignSGD 等，本文是首个把它们放进 SDE 框架并加入压缩维度。
- **vs Khirirat et al. 2024**：他们离散时间分析压缩下的 $(L_0,L_1)$ 收敛，但仍假设有界方差；本文同时放宽到仿射方差与重尾。
- **vs Compagnoni et al. 2025a**：作者团队前作只覆盖 $L$-光滑 + 压缩，没碰 $(L_0,L_1)$；本文是其在更现实光滑性下的自然推进。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "符号翻正 + ansatz 匹配"的 SDE 构造在 SDE-for-optim 这条线上是一次非平凡的概念修正
- 实验充分度: ⭐⭐⭐ 实验只为定性验证机制，MLP + 小幅 ResNet/ViT，没有大规模 benchmark，但与论文定位一致
- 写作质量: ⭐⭐⭐⭐⭐ 抛物 + 四次双 sanity check 把动机讲得极其清楚，正文 + Table 1/2 + Fig 1 几乎覆盖所有重要对比
- 价值: ⭐⭐⭐⭐⭐ 给所有用 SDE 推 scaling law / lr schedule / batch schedule 的工作打了补丁，是后续 $(L_0,L_1)$-SDE 分析的基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)

</div>

<!-- RELATED:END -->
