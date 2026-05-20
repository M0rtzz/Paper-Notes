---
title: >-
  [论文解读] Grokking: From Abstraction to Intelligence
description: >-
  [ICML 2026][可解释性][grokking] 本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"…
tags:
  - "ICML 2026"
  - "可解释性"
  - "grokking"
  - "奥卡姆剃刀"
  - "奇异学习理论"
  - "Kolmogorov 复杂度"
  - "模ular 算术"
---

# Grokking: From Abstraction to Intelligence

**会议**: ICML 2026  
**arXiv**: [2603.29262](https://arxiv.org/abs/2603.29262)  
**代码**: 无  
**领域**: 可解释性 / 涌现机制  
**关键词**: grokking, 奥卡姆剃刀, 奇异学习理论, Kolmogorov 复杂度, 模ular 算术

## 一句话总结
本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"，并用一个可解析的奇异特征机（SFM）证明这等价于自由能驱动的相变。

## 研究背景与动机
**领域现状**：grokking（在模 $p$ 算术等小数据集上，训练精度饱和后很久才出现测试精度突增）已经成为研究大模型涌现现象的"果蝇实验"。现有解释主要分两类：电路级机理分析（哪些 attention head 在干什么）和正则化/初始化尺度分析（weight decay、init scale 与延迟泛化的关系）。

**现有痛点**：这些工作以描述为主、缺乏预测能力。它们或者依赖某个具体任务的电路分析，难以跨架构推广；或者只观察到某个相关指标的变化，并不解释"为什么在第 $T$ 步发生相变"。当问"grokking 究竟何时发生、为何发生"时，整个领域还没有统一答案。

**核心矛盾**：以往工作把 grokking 当作一个**局部电路**或**优化动力学**事件来研究，忽略了一个全局视角——模型的整体结构是否在自发地朝某种"最小描述长度"的解演化。如果存在这种全局简化倾向，那么 grokking 只是该倾向跨过某个能量阈值时的可观测后果，而不是一个独立现象。

**本文目标**：(1) 提供一组与架构无关的全局度量来追踪 grokking 过程中的结构演化；(2) 在一个解析可控的代理模型上证明这种结构演化等价于自由能/Kolmogorov 复杂度的最小化；(3) 把延迟泛化解释为一次"信息压缩相变"。

**切入角度**：作者把 grokking 视作模型在固定训练精度约束下不断"瘦身"——奥卡姆剃刀。在 SLT（奇异学习理论）的语言里，这对应于后验质量从大 RLCT $\lambda$ 的奇点流向小 $\lambda$ 的奇点；在 Kolmogorov 复杂度的语言里，这对应于权重描述长度的下降；在 Fourier 视角下，这对应于模型从全频段杂乱响应坍缩到稀疏的 group character。这三种语言其实是同一件事的不同投影。

**核心 idea**：grokking $=$ 在保持训练损失为零的等位面上，沿着"参数有效维度"下降方向的自发滑动，且滑动方向由 SLT 的自由能 $F_n \approx n\mathcal{L} + \lambda\ln n$ 决定。

## 方法详解

### 整体框架
作者把研究拆成两条互相印证的腿：

1. **实证腿**（第 4 节）：在 $p=97$ 的模 $\{+,-,\times,\div\}$ 任务上训一个 48 层 GPT-2 风格 Transformer，并在初始化 / 记忆 / 涌现 / 泛化四个关键 step（$0.1\text{k}/1\text{k}/10\text{k}/100\text{k}$）做三件事：因果中介分析（CMA）量化每个 head 的因果贡献；对 embedding 矩阵做 PCA + Fourier 谱分析；对量化后的权重张量做 BDM 全局复杂度估计。
2. **理论腿**（第 5 节）：构造一个奇异特征机 SFM，它直接在 Fourier 域用复权矩阵 $\mathbf{W}\in\mathbb{C}^{p\times p}$ 拟合任务，并显式带一个 $\ln n$ 缩放的 $\ell_0$ 稀疏先验。在该模型上 RLCT $\lambda$ 和 Kolmogorov 复杂度都能解析写出。

两条腿在结论上对齐：实证看到的三种"塌缩"指标对应理论上的 $\lambda$ 从 $p^2/2$ 降到 $p/2$ 的相变。

### 关键设计

1. **因果中介分析（CMA）+ skip-ablation 揭示层级旁路结构**:

    - 功能：用 activation patching 度量每个 attention head 对正确答案 logit 的因果贡献，从而追踪 grokking 过程中"哪些层在干活"。
    - 核心思路：构造两条同结构、不同操数的输入 $\mathbf{s}_1, \mathbf{s}_2$，把 $\mathbf{s}_2$ 的某 head 激活嫁接到 $\mathbf{s}_1$ 上得到 $\tilde{\mathbf{s}}$，定义因果中介得分 $\text{CMS}(h) = [\mathcal{M}_\theta(y_2\mid\tilde{\mathbf{s}}) - \mathcal{M}_\theta(y_1\mid\tilde{\mathbf{s}})] - [\mathcal{M}_\theta(y_2\mid\mathbf{s}_1) - \mathcal{M}_\theta(y_1\mid\mathbf{s}_1)]$。在 step=1k 时高 CMS 的 head 杂乱地分散在 0–47 层；step=10k 时整体变暗；step=100k 时只剩 0–15 和 32–47 两端，中间 16–31 层完全可被 residual 旁路（skip-ablation 跳过这些层精度几乎不掉）。
    - 设计动机：以往工作只看 attention pattern 或 logit lens，无法分离相关和因果。CMA 直接断定"这个 head 是否真的在因果通路上"，并自然产生一个可视化的退化轨迹——从扁平噪声 → 中部熄灭 → 两端凝聚——它就是 grokking 的结构指纹。

2. **谱定域 + BDM 算法复杂度的联合追踪**:

    - 功能：用两种互补的复杂度代理量化"模型变简单了多少"——一个看频域稀疏性，一个看权重矩阵的算法压缩程度。
    - 核心思路：对 embedding 矩阵 $W_E$ 做二维 DFT 得到谱密度 $S[k,l]$，计算 Gini 系数 $G(\mathbf{s})$ 和 inverse participation ratio $P(\mathbf{s})=\sum_i s_i^4(\sum_i s_i^2)^{-2}$；二者同时升高表明能量从弥散变得集中到少数 Fourier 模。再把所有层权重经过 quartile 量化映射到 4-字母表，按 $4\times 4$ 子块用 CTM 查表 + BDM 公式 $K_{\text{BDM}}(\theta)=\sum_l\sum_b(\text{CTM}(b)+\log_2 n_b)$ 估全局算法复杂度。量化 trick 是为了把"权重衰减带来的幅值缩水"和"真正的结构性重组"区分开。
    - 设计动机：单看 sparsity 容易被 weight decay 骗，单看 PCA 看不到 algorithmic 结构。三类指标同步在 1k–10k 区间出现陡降，构成了 grokking $=$ 结构简化的强证据。

3. **奇异特征机（SFM）+ Occam Gate 解析地复现相变**:

    - 功能：构造一个数学上简化到极致、但仍能展示 grokking 的代理模型，使 RLCT $\lambda$ 和 Kolmogorov 复杂度 $K$ 都可手写出来。
    - 核心思路：把输入 $(u,v)$ 直接编码为 Fourier 张量 $\mathbf{x}_{\text{spec}}=\chi(u)\otimes\chi(v)$，模型只学一个复权矩阵 $\mathbf{W}\in\mathbb{C}^{p\times p}$；目标是 MAP 风格的 $\min_\mathbf{W} \tfrac12\sum_i\|y_i-\langle\mathbf{W},\mathbf{x}_{\text{spec}}^{(i)}\rangle_F\|^2 + \beta\ln n\cdot\|\mathbf{W}\|_0$。动力学用两步迭代：先做残差与基函数的相关（drift），再用 Occam Gate $W_{kl}^{(t+1)}=\mathbb{I}(|\tilde W_{kl}^{(t)}|>\tau)\cdot\tilde W_{kl}^{(t)}$ 把信噪比低于 $\tau=\sqrt{2\beta\ln n/n}$ 的频率分量直接抹掉。可证明：在记忆期 $\lambda_{\text{mem}}\approx p^2/2$，泛化期支撑集坍缩到对角 $\lambda_{\text{gen}}\approx p/2$，自由能交叉点近似为 $n^*\approx -\frac{\beta(p^2-p)}{\epsilon_{\text{gen}}}W_{-1}(-\frac{\epsilon_{\text{gen}}}{\beta(p^2-p)})$。
    - 设计动机：在真实 Transformer 上没法直接计算 RLCT，作者在 SFM 里用"激活 support 大小 / 2"做 $\lambda$ 的上界代理，并证明它和 $K_{SFM}(\mathbf{W})\propto\lambda(\mathbf{W})\cdot(2\log_2 p + C_{\text{float}})$ 成正比，从而把 SLT 与 AIT 在同一个可见对象上耦合起来。作者明确声明 SFM 是"假说生成型代理"而非对 SGD-Transformer 的等价证明。

### 损失函数 / 训练策略
- 真实 Transformer：标准交叉熵 + AdamW，48 层 GPT-2、$d_{\text{model}}=512$、8 头、fp32、A100、100k 步、5 个 seed 平均。
- SFM：上式 $\mathcal{J}(\mathbf{W})$，迭代两步 drift+Occam Gate，$\beta\ln n$ 控制相变阈值；$n_{\text{eff}}$ 与训练 step 成正比但解释为启发式映射。

## 实验关键数据

### 主实验

| 训练 step | CMA 高响应 head 分布 | 嵌入流形 | 谱集中度 (Gini, IPR) | BDM 复杂度 |
|----------|--------------------|----------|---------------------|-----------|
| 0.1k | 全层稀疏 | 高熵球团 | 极低 | 高 plateau |
| 1k（记忆） | 全层弥散 | 高维点云 | 仍低 | 高 plateau |
| 10k（涌现） | 中部开始变暗 | 开始收缩 | 急剧上升 | 急剧下降 |
| 100k（泛化） | 仅 0–15、32–47 | 1D 圆环（同构 $\mathbb{Z}_{97}$） | 高位稳定 | 最低 plateau |

| 现象 | 实证（Transformer） | 理论（SFM） |
|------|--------------------|------|
| 有效维度 | 层级旁路、中部可被跳过 | $\lambda$ 从 $p^2/2$ 降到 $p/2$ |
| 算法复杂度 | BDM 急降 + 块状结构出现 | $K_{SFM}\propto \lambda\cdot(2\log_2 p+C_{\text{float}})$ |
| 几何对称 | embedding 1D 环 | 支撑集塌缩到对角（加/减） |

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| 跳过 head 0–15 | 精度崩溃 | 早层是必经路径 |
| 跳过 head 16–31 | 精度几乎不变 | 中层"功能冗余"，可被 residual 旁路 |
| 跳过 head 32–47 | 精度崩溃 | 末层负责输出格式化 |
| 量化前看 sparsity | 看似下降 | 但混入 weight decay 幅值缩水 |
| 量化后看 BDM | 真正下降 | 排除幅值效应后仍下降 → 结构性重组 |

### 关键发现
- 三类不同语言（电路冗余 / 谱稀疏 / 算法复杂度）的"塌缩"在时间轴上几乎同步发生，强烈暗示它们是同一事件的不同投影。
- 中部 16–31 层可旁路这一点说明所谓"涌现的符号结构"不是均匀分布在整个模型里，而是凝聚在两端的少数层；这与"实现 FMA 只需要 1D group 编码 + 输出投影"的理论预言一致。
- SFM 中相变阈值 $n^*$ 与 $\beta(p^2-p)/\epsilon_{\text{gen}}$ 成 $W_{-1}$ 关系，定性地复现了"高 weight decay → grokking 提前"的经验规律。
- 对乘除运算，SFM 的"对角"图像并不严格成立（需要离散对数重排），作者非常诚实地标注了这点局限。

## 亮点与洞察
- **复杂度的三重统一**：把 SLT 的 $\lambda$、AIT 的 KC 和谱稀疏在同一个 case 上对齐，是这篇论文最大的"啊哈"——以前这三套语言是分头说话的。
- **可旁路性作为可观测量**：用 skip-ablation 直接把"层是否必要"变成 yes/no 实验，比传统的 attention pattern 解释力更强，且这个 trick 可迁移到任何后训练分析（如 LLM 的功能性剪枝）。
- **量化后再算 BDM**：避免把 weight decay 的幅值变化误读成结构变化，是处理 grokking 数据的一个干净 trick。任何想用复杂度代理证明"模型变简单"的工作都该照抄。
- **SFM 不假装自己是 Transformer**：作者明确把 SFM 定位为"假说生成器"，不去吹"我们证明了 grokking 等价于 SLT 相变"，这种克制反而让结论更可信。

## 局限与展望
- SFM 的对角支撑图像只对加减法严格成立；乘除需要离散对数重排，作者只给定性说明，没有把 $\times,\div$ 的 SFM 解严格写出来。
- $n_{\text{eff}}(t)$ 与训练步数的映射是启发式的，自由能交叉点 $n^*$ 的预测无法在真实 Transformer 上做定量校验。
- 所有结论都基于 $p=97$ 的玩具任务，是否能推广到 LLM 上"知识涌现"是另一个量级的问题——文章自己承认"phase transition 在 SGD 上的语言只是描述性的"。
- BDM 的量化粒度（4×4 块、4 字母）有不少超参没消融。

## 相关工作与启发
- **vs Liu et al. (Omnigrok)**: 他们关注 weight decay 与 grokking 的因果，本文把这层因果嵌进 SLT 自由能框架里，给出了一个统一的"为什么 weight decay 有用"的解释（$\beta\ln n$ 项控制阈值）。
- **vs Nanda 等的电路机理工作**: 他们做的是 case-by-case 的电路逆向工程，本文用 CMA 给出了一个跨任务可计算的"哪一层在干活"指标，跳出对特定 head 的过拟合。
- **vs Mallinar et al. (non-NN grokking)**: 他们说 average gradient outer product 也能 grok，本文的 SFM 进一步剥离掉 NN 结构本身，把现象归结到"有 $\ln n$ 稀疏先验 + 全局可观测复杂度"这一最小集合，强化了 grokking 与架构无关的结论。
- **启发**：可旁路性测试 + 量化后复杂度 + 谱稀疏率，这套"三件套"诊断可迁移到任何"模型在训练中变简单"的研究，比如 LLM 的 emergent abilities 或 diffusion 的 mode collapse。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一次把 SLT/AIT/spectral 三套语言对齐到 grokking 这件事上，但具体度量都是已有工具
- 实验充分度: ⭐⭐⭐ 在 $p=97$ 这一个任务上做得很扎实，但只 1 个 prime、1 套架构，跨任务/跨尺度验证缺失
- 写作质量: ⭐⭐⭐⭐ 数学和实证两条腿叙述清晰，且作者对 SFM 局限的标注非常克制可信
- 价值: ⭐⭐⭐⭐ 给后续"涌现/相变"研究提供了一套通用诊断工具和一个可手算的 toy model

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](../../ICLR2026/interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICML 2025\] Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations](../../ICML2025/interpretability/explaining_fast_and_slow_abstraction_and_refinement_of_provable_explanations.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)

</div>

<!-- RELATED:END -->
