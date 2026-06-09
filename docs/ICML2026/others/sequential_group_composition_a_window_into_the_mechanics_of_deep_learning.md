---
title: >-
  [论文解读] Sequential Group Composition: A Window into the Mechanics of Deep Learning
description: >-
  [ICML 2026][群论组合] 作者把"对一段群元素求累积乘积"这个统一任务作为显微镜，用群上 Fourier 分析 + AGF 框架证明两层网络会按 Fourier 能量从大到小逐个学习不可约表示（irrep）…
tags:
  - "ICML 2026"
  - "群论组合"
  - "不可约表示"
  - "群上 Fourier 分析"
  - "Alternating Gradient Flow"
  - "架构表达力"
---

# Sequential Group Composition: A Window into the Mechanics of Deep Learning

**会议**: ICML 2026  
**arXiv**: [2602.03655](https://arxiv.org/abs/2602.03655)  
**代码**: 无  
**领域**: 机制可解释性 / 学习动力学  
**关键词**: 群论组合, 不可约表示, 群上 Fourier 分析, Alternating Gradient Flow, 架构表达力

## 一句话总结
作者把"对一段群元素求累积乘积"这个统一任务作为显微镜，用群上 Fourier 分析 + AGF 框架证明两层网络会按 Fourier 能量从大到小逐个学习不可约表示（irrep），并刻画两层、RNN、深层 MLP 三种架构在序列长度 $k$ 上分别需要 $2^k$ 宽度、$k$ 步、$\log k$ 层的表达力鸿沟。

## 研究背景与动机

**领域现状**：机制可解释性方向近几年集中在 "modular addition" 这一类小型代数任务，靠经验观察发现网络会自发学到 Fourier 特征（三角恒等式式地实现加法）；学习动力学方向则发现损失曲线常呈"台阶状"，对应于网络逐步学到越来越复杂的特征。两条线索都缺一个统一的、能解析推导的实验台。

**现有痛点**：modular addition 只是 $k=2$ 的循环群特例，结论很难推广到非阿贝尔群、序列长度 $k>2$、以及不同架构之间的对比；现有分析多是"先观察再事后命名"，缺乏从第一性原理推导特征出现顺序的能力。

**核心矛盾**：网络究竟是把组合操作整体死记下来，还是真的发现了群的代数结构？如果是后者，它学到的是哪种表示、按什么顺序学、不同架构在效率上差多少？这些问题在 modular addition 框架里无法被分开讨论。

**本文目标**：(i) 给出一个能覆盖任意有限群（阿贝尔/非阿贝尔）且能延伸到任意序列长度 $k$ 的统一任务；(ii) 在该任务上对两层网络给出严格的特征学习顺序定理；(iii) 量化两层、RNN、深层 MLP 在表达此类组合上的宽度/深度差异。

**切入角度**：把"学习累积乘积"视作回归问题 $(g_1,\dots,g_k)\mapsto\prod_i g_i$，借助群表示论里的正则表示和群 Fourier 变换把整个任务在 irrep 基下块对角化——此后梯度下降在每个 irrep 子空间内的行为就能解析地写出来。

**核心 idea**：把训练动力学 = 沿着 Fourier 谱按 $\|\hat{x}[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$ 这把"重要性尺子"从大到小贪心激活 irrep；深度则是用结合律把"一次性算 $k$ 项乘积"压成"成对组合"，从而把宽度从 $2^k$ 砍回到 $k$ 或 $\log k$。

## 方法详解

### 整体框架

任务定义：固定一个有限群 $G$ 与一个编码向量 $x\in\mathbb{R}^{|G|}$，每个群元素 $g$ 通过正则表示得到编码 $x_g=\lambda(g)^\top x$（$x=e_1$ 时退化为 one-hot）。网络 $f:\mathbb{R}^{k|G|}\to\mathbb{R}^{|G|}$ 接收编码序列 $x_{\mathbf g}=(x_{g_1},\dots,x_{g_k})$，输出对乘积 $x_{g_1\cdots g_k}$ 的回归估计，用 MSE 训练。引理 3.5 已证明若 $x$ 非平凡且零均值，则任何线性映射都不可能解决此任务——必须有非线性交互。最简实例是 $\sigma(z)=z^k$（monic 多项式）的两层网络 $f=W_\text{out}\sigma(W_\text{in}x_{\mathbf g})$，在消失初始化（$\theta_i(0)\sim\mathcal N(0,\alpha^2)$，$\alpha\to0$）下用神经元相关的学习率 $\eta_{\theta_i}=\|\theta_i\|^{1-k}\log(1/\alpha)$ 跑梯度流。

分析框架基于 Kunin 等人 2025 的 Alternating Gradient Flow（AGF）：在消失初始化下，每个隐神经元在"休眠"（$\|\theta_i\|\approx 0$、不影响输出）与"激活"（$\|\theta_i\|\gg 0$、主导拟合）两态间切换；训练呈阶梯型——平台期对应休眠神经元相互竞争最大化 utility，下降期对应已激活神经元集体最小化残差。本文把 AGF 应用到群上 Fourier 域，把每一次激活对应到一个 irrep 的"出场"。

### 关键设计

**1. Sigma-pi-sigma 分解 + utility 最大化定理：把"下一个学哪个特征"写成闭式排序。**

可解释性领域一直靠探针"先观察再命名"地发现网络学到了 Fourier 特征，但说不清这些特征为什么出现、按什么顺序出现。本文从神经元的多项式输出入手把它解析化：每个神经元 $f(x_{\mathbf g};\theta_i)$ 可拆成关键的"全交互项" $f^{(\times)}=w_i\cdot k!\cdot\prod_j\langle u_{i,j},x_{g_j}\rangle$ 与无关的"加性项" $f^{(+)}$，而 utility（休眠神经元与残差的内积）只由 $f^{(\times)}$ 贡献。把 $f^{(\times)}$ 搬到群上 Fourier 域后，utility 变成输入/输出权重的 Fourier 系数与 $\hat x[\rho]$ 的张量内积；在 $\|\theta\|=1$ 约束下极大化，最优解必然把全部 Fourier 能量集中到单个 irrep $\rho_*$ 上，而这个 $\rho_*$ 由一把可计算的"重要性尺子"选出：

$$\rho_*=\arg\max_{\rho\notin\mathcal I^{t-1}}\frac{\|\hat x[\rho]\|_\text{op}^{k+1}}{(C_\rho n_\rho)^{(k-1)/2}}\quad(C_\rho=1\ \text{若}\ \rho\ \text{实}, \text{否则}\ 2)$$

这条定理（Theorem 4.1）第一次把"网络贪心地按 Fourier 重要性逐个学习"从经验观察证成了解析事实，而且序列长度 $k$ 显式出现在指数里——$k$ 越大、高能量 irrep 被优先放大得越厉害（$k+1$ 对 $(k-1)/2$），于是更长序列对应更尖锐的台阶。

**2. Cost 最小化下的对齐与残差更新：让训练台阶与 irrep 一一对应。**

utility 最大化只解释了"谁先出场"，还要说清出场之后发生什么——这正是 AGF 两步里的第二步。一旦 $N$ 个神经元同时激活并对齐到同一个 $\rho_*$，它们就在 $\rho_*$ 的约束子空间内联合最小化损失，效果等价于把目标 $x_{g_1\cdots g_k}$ 在 $\rho_*$ 上的 Fourier 分量"消掉"。把激活神经元的输出写到 Fourier 域：

$$f(x_{\mathbf g};\Theta_\mathcal A)[h]=\frac{1}{|G|}\sum_{\rho\in\mathcal I^{t-1}}\langle\rho(g_1\cdots g_k h)^\dagger,\hat x[\rho]\rangle_\rho$$

每完成一轮 AGF，已学 irrep 集合 $\mathcal I^{t-1}$ 就并入下一个 $\rho_*$（并保持共轭闭包），直到 $\mathcal I^{t-1}=\mathcal I(G)$ 时网络完美拟合任务。这套"出场顺序 + 残差更新"机制让损失曲线上的每一级台阶都能事先对应到一个具体 irrep，于是任意有限群（包括非阿贝尔的 $D_3$）的学习路径都可预测；而且只要对编码 $x$ 做谱整形，就能反过来改写学习顺序。

**3. 架构 → 表达力的三种缩放：把"结合律=深度优势"精确量化。**

同一个累积乘积任务，换不同架构需要的容量天差地别，本文给出了三档可比的缩放。两层网络要一次性同时拟合所有 $k$ 元乘积，就得为每个 Fourier 模复制足够多神经元去相互抵消干扰项 $f^{(+)}$，所需隐藏宽度按 $2^k$ 量级爆炸——注意瓶颈不在"学不到 irrep"，而在"抵消 $f^{(+)}$ 的冗余开销"。RNN 借结合律把累积乘积串行写成 $k$ 步状态更新，每步只需固定大小网络；深层 MLP 则把序列两两配对、按二叉树合并，只需 $\log_2 k$ 层。于是"为什么深层/循环结构在算法类任务上更省参数"这个长期偏经验的主张，在群组合任务上落成了"两层 $2^k$、RNN $k$、深层 $\log k$"的解析陈述，把宽度/深度/步数之间的 trade-off 写到了具体常数。

### 损失函数 / 训练策略
回归损失 $\mathcal L(\Theta)=\tfrac{1}{2|G|^k}\sum_{\mathbf g\in G^k}\|x_{g_1\cdots g_k}-f(x_{\mathbf g};\Theta)\|^2$；消失初始化 $\alpha\to 0$；神经元自适应学习率 $\eta_{\theta_i}\propto\|\theta_i\|^{1-k}$。对编码 $x$ 假设：零均值、$\hat x[\rho]$ 要么 0 要么可逆、不同 $\rho$ 的 utility 判据值彼此分离（保证阶梯清晰）。

## 实验关键数据

### 主实验

| 设置 | 群 / 架构 | 现象 | 关键观察 |
|------|----------|------|---------|
| 二元组合 $k=2$ | 阿贝尔 $C_p$ / 两层 quadratic MLP | 训练 loss 呈阶梯，每个平台对应一个 $\rho$ 激活 | Fourier 模激活顺序与 Theorem 4.1 判据排序完全一致 |
| 二元组合 $k=2$ | 非阿贝尔 $D_3$ / 两层 quadratic MLP | 一维 + 二维 irrep 依次被学到 | 验证理论可推广到非阿贝尔群 |
| 序列长度 $k$ 扫描 | 固定 $G$，调 $k=2,3,4,\dots$ | 两层网络所需宽度近似指数增长 | 完美拟合所需隐藏宽度按 $2^k$ 量级 |
| 架构对照 | 两层 vs RNN vs 深层 MLP | RNN 用 $k$ 步、深层 MLP 用 $\log k$ 层即可学到 | 结合律允许深度换宽度 |

### 消融与对照

| 配置 / 对比 | 结果 | 说明 |
|------------|------|------|
| 编码 $x=e_1$（one-hot） | 默认基线 | 所有 irrep 等权 |
| 编码 $x$ 谱整形 | 学习顺序随判据值重排 | 验证 $\rho_*$ 选择由 $\|\hat x[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$ 决定 |
| 多项式激活 vs 平滑激活 | Taylor 展开后行为类似 | 论文附录讨论非多项式激活通过 Taylor 展开继承相同交互项 |
| 单 vs 多 $N$ 神经元对齐到同一 $\rho_*$ | 多个神经元才能联合消除 $f^{(+)}$ | 解释了为何需要冗余宽度 |

### 关键发现
- 学习顺序由编码 $x$ 的 Fourier 能量谱**完全决定**——这是首次给出"可解释性观察到的 Fourier circuit 出现顺序"的解析公式。
- 序列长度 $k$ 在判据指数 $k+1$ 处放大主 irrep 的优势，意味着更长序列 → 更突出的"贪心"行为、更尖锐的阶梯。
- 两层网络所需宽度对 $k$ 指数爆炸的根因不在于"无法学到 irrep"，而在于必须用大量神经元相互抵消 $f^{(+)}$ 干扰；深层结构用结合律绕过了这一抵消负担。

## 亮点与洞察
- 把"群上 Fourier 分析 + AGF"拼成一把刀，能严格写出每一个 irrep 何时、以何种方式被网络挑中——把机制可解释性从"事后命名"提升到"事前预测"。
- Theorem 4.1 给出的判据 $\|\hat x[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$ 出现了序列长度 $k$ 显式影响特征排序，这点在 modular addition $k=2$ 文献里看不出来，说明"长序列任务里更强烈的简洁性偏置"是任务本身的代数事实。
- "结合律=深度优势"这句口号被精确量化：两层 $2^k$、RNN $k$、深层 $\log k$，给"为什么深层模型在算法类任务上更省参数"提供了非启发式的证据，可迁移到 Rubik's Cube、轨迹积分等任意可写成群组合的任务。
- 编码 $x$ 的谱整形成为一个新的可调旋钮：通过设计 $x$ 的 Fourier 形状就能控制网络先学什么、后学什么，对课程学习和教学设计都有启示。

## 局限与展望
- 理论严格性集中在两层 quadratic 网络 + 消失初始化这一限定场景；对 ReLU 网络、Adam 优化器、有限初始化幅度的推广只作了定性讨论。
- 实验主要在小型有限群（$C_p$, $D_3$ 等）上做，并未在大模型/真实任务（如算术 LLM）里验证学到的 Fourier circuit 是否还沿同样的排序出现。
- AGF "一次只激活一个特征"的假设依赖于编码满足"判据值彼此分离"，当 $G$ 很大或谱接近平均时（如随机 $x$），台阶会变模糊，理论刻画的实用性降低。
- 深层网络 $\log k$ 缩放的构造是存在性证明，并未保证 SGD 一定能在实际训练中找到该解；与"transformer 是否真用 $\log k$ 层组合"这一经验问题之间仍有差距。

## 相关工作与启发
- **vs Nanda 等 2023（grokking modular addition）**：那篇靠探针经验地发现网络学到 Fourier 加法 circuit；本文把"为何会出现这种 circuit、何时出现、以何种顺序出现"做成了闭式定理，并推广到非阿贝尔群与任意 $k$。
- **vs Chughtai 等 2023 / Stander 等 2023（小群组合实证）**：他们在二元组合上观察到类似的 Fourier 结构；本文给出了 Fourier 出现顺序的解析公式（Theorem 4.1），并将分析推到序列任务。
- **vs Kunin 等 2025 AGF**：本文是 AGF 框架在"群上 Fourier 域"的首次落地应用——把 AGF 的抽象 utility-cost 两步在群表示论里写得彻底；同时把 sigma-pi-sigma 分解给 AGF 提供了具体可计算的形式。
- **vs Liu 等 2022 / Sanford 等 2023-2024（transformer vs RNN 表达力）**：那条线讨论 transformer 用 $\log$ 深度在算法任务上超越 RNN；本文在群组合这一具体任务上把同样的对比量化为"两层 $2^k$ / RNN $k$ / 深层 $\log k$"三档缩放，使表达力差距落到具体常数上。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把群表示论、AGF、表达力三条线索拼成一个能解析推导的实验台，是近年机制可解释性中少见的"先证再观察"型工作。
- 实验充分度: ⭐⭐⭐ 小型群上验证完整，但缺乏在大模型/真实算术任务上的迁移验证。
- 写作质量: ⭐⭐⭐⭐⭐ 数学陈述清晰，引理与定理层次分明，附录补全证明与背景。
- 价值: ⭐⭐⭐⭐⭐ 给出了一个"算法可解释性"领域的标杆任务，可期待后续工作沿此延伸到更复杂的群结构和更现实的架构。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)
- [\[ICLR 2026\] CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](../../ICLR2026/others/chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](../../CVPR2025/others/potential_field_based_deep_metric_learning.md)

</div>

<!-- RELATED:END -->
