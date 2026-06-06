---
title: >-
  [论文解读] Hyperparameter Transfer with Mixture-of-Experts Layers
description: >-
  [ICML 2026][LLM效率][μP] 本文把 μP/CompleteP 的最大更新参数化思想扩展到稀疏 MoE Transformer，给出 router、expert 上/下投影、expert bias 在 width/depth/专家数/专家宽度同时放大时的初始化与学习率缩放规则…
tags:
  - "ICML 2026"
  - "LLM效率"
  - "μP"
  - "CompleteP"
  - "MoE scaling"
  - "DMFT"
  - "零样本超参迁移"
---

# Hyperparameter Transfer with Mixture-of-Experts Layers

**会议**: ICML 2026  
**arXiv**: [2601.20205](https://arxiv.org/abs/2601.20205)  
**代码**: 无  
**领域**: LLM效率 / MoE / 超参数迁移  
**关键词**: μP, CompleteP, MoE scaling, DMFT, 零样本超参迁移

## 一句话总结
本文把 μP/CompleteP 的最大更新参数化思想扩展到稀疏 MoE Transformer，给出 router、expert 上/下投影、expert bias 在 width/depth/专家数/专家宽度同时放大时的初始化与学习率缩放规则，并用一套三层 mean-field 的 DMFT 理论证明该参数化在 $n_{\text{embd}},n_{\text{exp}},n_{\text{hid}},L\to\infty$（固定激活稀疏度 $\kappa$）下存在尺度不变极限，从 38M 激活基模迁移到 2B 总参的 MoE 上都能直接复用最优 LR / init，且零样本超参训出来的 MoE 在等激活参数下可与 dense GPT2 speedrun 持平甚至更优。

## 研究背景与动机
**领域现状**：μP 与后续 CompleteP / depth-μP 等参数化已经能让 dense Transformer 在 width 和 depth 同时放大时直接迁移 LR、init 这些关键 HP，从而在小模上扫超参、在大模上一把训成功。MoE 是当前扩参数量最主流的手段，但相关 HP 迁移工作大多还停留在 dense 模型。

**现有痛点**：直接把 dense μP 套到 MoE 上有两件事说不清。一是 MoE 多了 router 权重、expert bias 这些全新参数组，它们的最优 LR/init 怎么随 $n_{\text{embd}}$ 缩放并不显然；二是 MoE 引入了 *专家数 $n_{\text{exp}}$* 与 *专家宽度 $\alpha_{\text{ffn}}n_{\text{embd}}$* 两个新的 scale 轴，每条轴都要不要重新调参没人系统验证过。在 dense 上 work 的 μP 启发式做法（数维度判断 $\Theta(1)$ 更新）在 MoE 上既无法回答跨 $n_{\text{exp}}$ 是否稳定，也无法解释为何 $\alpha_{\text{ffn}}$ 改变不应该影响最优 HP。

**核心矛盾**：MoE 的 router、稀疏 top-$k$ 路由、expert 内部 MLP 三层东西之间存在相互耦合，单纯逐参数组用 μP 启发式无法保证这些耦合在极限下都收敛到一个良定义的训练动力学。换句话说：要做 HP transfer 必须先证明存在一个 *不依赖具体 scale 变量* 的 mean-field 极限。

**本文目标**：拆成三个子问题——（1）MoE 各参数组的 init/LR 该用什么 $n_{\text{embd}},\alpha_{\text{ffn}}$ 指数；（2）在 $n_{\text{exp}}\to\infty$ 时是固定 $n_{\text{act}}$ 还是固定 $\kappa=n_{\text{act}}/n_{\text{exp}}$；（3）这套规则是否真的对应一个收敛的训练极限。

**切入角度**：作者沿用 CompleteP 的 width+depth 规则不动，重点研究 MoE 模块；并提出固定稀疏度 $\kappa$ 来扩 $n_{\text{exp}}$ 的扩展观（每个 expert 看到的 token 比例 $\kappa B$ 保持常数，这正好对应 mean-field 测度上的常概率事件，理论与硬件部署都自然）。

**核心 idea**：用"router/expert/bias 上每个组件 *单独* 满足最大更新条件 $\Delta W\,\partial z/\partial W=\Theta(1)$"作为加强版 μP 推出参数化（Table 1），再用三层 mean-field DMFT 严格证明 $n_{\text{embd}},n_{\text{exp}},n_{\text{hid}},L$ 同时发散下训练动力学有 well-defined 极限。

## 方法详解

### 整体框架
模型用 pre-LayerNorm 的 decoder-only Transformer，FFN 全部替换为 MoE 模块：每层 $f_{\text{MoE}}(h)=\frac{1}{n_{\text{act}}}\sum_{i\in A(h)} g_i(h)\,E_i(h)$，其中 $g_i(h)=\sigma(W_{\text{router}}^{(i)\top}h)$ 是 sigmoid 路由权重，$A(h)$ 是带可训练 bias $b_i$ 的 top-$n_{\text{act}}$ 硬路由集合，$E_i(h)=W_{\text{down}}^{(i)}\phi(W_{\text{up}}^{(i)\top}h)$ 是单隐层 MLP 专家。残差块都带 $1/L$ 乘子（CompleteP 风格）以保证 depth 迁移。负载均衡走 auxiliary-loss-free 路线：只更新 bias $b_i\leftarrow b_i-\eta_{\text{bias}}(\text{Load}_i-\kappa)$，不动其它参数。

把要扩的 scale 轴写清：$L$（深度）、$n_{\text{embd}}$（残差流宽度）、$\alpha_{\text{ffn}}$（专家隐层宽度倍率）、$n_{\text{exp}}$（专家数），始终保持 $\kappa=n_{\text{act}}/n_{\text{exp}}$ 为常数。给定基模一组好 HP，参数化的任务就是用一组 *只依赖 scale 轴* 的规则把 router/expert/bias 的 LR、init 自动外推到大模上。

### 关键设计

1. **MoE 参数化（Table 1：router/expert/bias 的 init 与 LR 缩放规则）**:

    - 功能：把 μP 的"每步 entry-wise 更新 $\Theta(1)$"原则细化到 MoE 每个新参数组，给出 router 矩阵、expert up/down 投影、expert bias 的 init std 和 Adam LR 应如何随 $n_{\text{embd}}$ 与 $\alpha_{\text{ffn}}$ 缩放。
    - 核心思路：作者要求"对每个 expert 的混合系数 $g_i$、expert 输出 $E_i$、hidden 激活 $h_{\text{up}}$ *分别* 满足 $\eta_W\overline{\nabla W}\partial z/\partial W=\Theta(1)$"，这是比 dense μP 更强的逐组件条件。借 SignGD 近似 Adam（$\Delta w\approx\eta\,\text{sgn}(\partial\mathcal{L}/\partial w)$）并对 $h$ 与 $\Delta W$ 做 LLN 对齐假设 $\cos(v,w)\in\Theta(1)$，逐组推出：router 的 $\eta\in\Theta(1/n_{\text{embd}})$、init $\Theta(n_{\text{embd}}^{-\gamma})$（实验取 $\gamma=1$）；expert up 的 $\sigma_{\text{init}}=n_{\text{embd}}^{-1/2}$、$\eta=n_{\text{embd}}^{-1}$；expert down 因为还要承担 $h_{\text{up}}$ 到 $E$ 的二次缩放而带额外 $\alpha_{\text{ffn}}^{-1}$，即 $\sigma_{\text{init}}=\alpha_{\text{ffn}}^{-1}n_{\text{embd}}^{-1/2}$、$\eta=\alpha_{\text{ffn}}^{-1}n_{\text{embd}}^{-1}$；expert bias 用 $\Theta(1)$ LR、零初始化以同时满足激活集每步变化 $\Theta(n_{\text{act}})$ 与 step-0 负载均衡。
    - 设计动机：标准 fan-in init 会让 $W_{\text{down}}$ 对 $\alpha_{\text{ffn}}$ 的依赖错配，导致 $\alpha_{\text{ffn}}$ 一变最优 LR 就漂；这里把 $W_{\text{down}}$ 当作 mean-field 两层 MLP 的"中间宽度"层来定 init，正好让 $\alpha_{\text{ffn}}$ 在最优 HP 上消失（Figure 2 第四列的零迁移现象的根源）。把 router LR 单独压成 $1/n_{\text{embd}}$ 而不是 $\Theta(1)$ 也是因为 $h^\top\Delta W_{\text{router}}^{(i)}$ 在 LLN 对齐下天然带 $\sqrt{n_{\text{embd}}}\cdot\sqrt{n_{\text{embd}}}$ 因子。

2. **三层 mean-field 的 DMFT 极限（理论根基）**:

    - 功能：把 Bordelon & Pehlevan 的 DMFT 框架推广到带稀疏 MoE 的 deep residual 网络，给出 $n_{\text{embd}},n_{\text{exp}},n_{\text{hid}},L\to\infty$（固定 $\kappa$ 且 $n_{\text{embd}}/(n_{\text{exp}}n_{\text{hid}}L)$ 有界）下的封闭训练动力学方程。
    - 核心思路：分析对象是只含 MoE 模块的残差网络 $h^{(\ell+1)}=h^{(\ell)}+L^{-1}f_{\text{MoE}}^{\ell}(h^{(\ell)})$。结论是动力学按三层 mean-field 嵌套展开：最外层是残差流神经元间的 mean-field，第二层是同一 MoE 层内 expert 之间的 mean-field，第三层是 expert 内部 hidden neuron 之间的 mean-field。硬路由通过分位阈值 $q_\star(\kappa)$（满足 $\mathbb{E}[\mathbf{1}_{q\ge q_\star}]=\kappa$）出现，这正是为什么要固定 $\kappa$。结论里能直接读出几件事：极限动力学 *不依赖* $\alpha_{\text{ffn}}$（与 dense 大深度结论一致，解释了 Figure 2 的 $\alpha_{\text{ffn}}$ 迁移）；同时发散 $n_{\text{embd}},n_{\text{exp}}$ 时只要 $\alpha_\star=\lim n_{\text{embd}}/(n_{\text{hid}}n_{\text{exp}}L)=0$，所有 joint scaling 给出同一极限；深度极限在 $\alpha_\star=0$ 下退化为 neural ODE，$\alpha_\star>0$ 下变成 neural SDE。
    - 设计动机：纯 μP 启发式只能在每个 scale 轴上"看起来 $\Theta(1)$"，但无法回答"$n_{\text{exp}}$ 真的可以无限大吗、$n_{\text{hid}}$ 真的可以不发散吗"。DMFT 给的不是新公式，而是 *证明* 这些参数化确实对应同一组确定性的演化方程，从而把"HP 迁移到底为何成立"从经验观察升格为理论保证。

3. **固定稀疏度 $\kappa$ 而非固定 $n_{\text{act}}$ 的扩展策略**:

    - 功能：在扩 $n_{\text{exp}}$ 时同步等比扩 $n_{\text{act}}$，让每个 expert 看到的 token 比例 $\kappa$ 保持常数，而不是 Switch Transformer 那种 $n_{\text{act}}=1$ 不变、$\kappa\to 0$ 的扩展观。
    - 核心思路：完美均衡下每个 expert 每 batch 看到 $\kappa B$ 个 token，self-attention 与 router 看到全部 $B$ 个；若 $\kappa$ 跟着 scale 变，expert 与其它模块的有效"数据效率"会失配，最优 HP 自然不会迁移（论文 Figure 11 给了部分验证）。固定 $\kappa$ 也对应 mean-field 测度上的常概率切片，DMFT 极限里 $q_\star(\kappa)$ 的存在与稳定性都依赖这一点。
    - 设计动机：这一条既是工程考量（大集群里 $n_{\text{act}}$ 受通信带宽下界约束，反正也不能取太小）也是理论必需（否则 mean-field 测度退化）。配合本规则，当 $n_{\text{act}}$ 受限时仍可走"先扩 granularity 再扩 width/depth 或 distill"的折中路线。

### 损失函数 / 训练策略
基础是标准 Adam + AdamW；router 不用 softmax+aux loss，而走 sigmoid + auxiliary-loss-free 的 bias 更新 $b_i\leftarrow b_i-\eta_{\text{bias}}(\text{Load}_i-\kappa)$。LR scheduler 在固定 token budget 实验里用前 1000 步 linear warmup + 后 1000 步恒定 LR（共 2000 步 / 1B tokens / batch 500K / seq 1024）；放大到长 horizon 时叠加 cosine 衰减到 0。除了 $n_{\text{embd}},\alpha_{\text{ffn}}$ 的指数缩放，每组参数还要 *单独* 调一个 $\Theta(1)$ 常数倍率（论文 D.1：不调常数倍率，训练动力学如负载均衡损失会在最优 HP 附近不稳定）。两个稀疏度配置：FineWeb 上 $\kappa=1/4$、C4 上 $\kappa=1/12$。

## 实验关键数据

### 主实验

| 实验设置 | 关键观察 | 说明 |
|---------|---------|------|
| FineWeb, $\kappa=1/4$, 1B tokens, 38M→1.8B 沿 width/depth/$n_{\text{exp}}$/$\alpha_{\text{ffn}}$ 单轴扫 LR & init | 4 个 scale 轴的最优 LR 与 init std 曲线在不同模型 size 下几乎共点 | 验证零样本 HP 迁移（Figure 2，最后一行误差棒为 4 seed 的 max/min/median） |
| C4, $\kappa=1/12$, 1B tokens, 4 个 scale 轴 | 同 FineWeb 结论，覆盖更稀疏的 routing 与不同语料 | Figure 4 |
| GPT2-small (124M) 激活配置，FineWeb，10B tokens，zero-shot HP（从 38M 基模迁移） | val loss 与 dense GPT2 speedrun（AdamW / Muon）相当甚至更优，且总参数比 dense 多 | Figure 1 |
| GPT2-medium (355M) 激活，7.5B tokens | 同样用 zero-shot HP 训稳定 | Figure 16 |
| 早期 loss 曲线坍缩 | 在 zero-shot 最优 HP 下，width / $n_{\text{exp}}$ / $\alpha_{\text{ffn}}$ 扩大时，前若干步 loss 曲线完全重合，之后大模型更低 | Figure 3，对应 DMFT 尺度不变性预测 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整参数化 + 常数倍率调优 | 最优 loss，负载均衡稳定 | 主流程 |
| 不调每参数组的常数倍率 | 在最优 HP 附近 load-balancing loss 出现不稳定，留下显著性能空间 | Appendix D.1，常数倍率在 MoE 上比 dense 更敏感 |
| $W_{\text{down}}$ 用标准 fan-in init（无 $\alpha_{\text{ffn}}^{-1}$） | 最优 LR 随 $\alpha_{\text{ffn}}$ 漂移 | Figure 11 简单消融，论证 $\alpha_{\text{ffn}}^{-1}$ 因子 |
| 不同 mixing coefficient（sigmoid vs softmax） | 跨 scale 表现近似 | Figure 18 |
| 路由 init exponent $\gamma\in\{1/2,1,\infty\}$ | 实验里只要不"数值上太大"差别不显著，主文用 $\gamma=1$ | 与历史 Shazeer/GShard/Malasnicki 工作的设定对比 |
| 固定参数量、$\kappa$、$\alpha_{\text{ffn}}$，扫 $n_{\text{act}}$ 与 expert 大小（反比） | 1B token 短 horizon 与 5B token 长 horizon 都是 *更多更小的 expert* 单调更好 | Figure 5 与 Apdx D.3，复现 Krajewski 等结论但 *不用* 重调 HP |

### 关键发现
- 在固定 $\kappa$ 的前提下，width / depth / 专家数 / 专家宽度四条轴上 LR 与 init 都能直接迁移，且 *跨 $\alpha_{\text{ffn}}$ 的迁移* 是 DMFT 独有的预测（μP 启发式给不出），实验完美符合。
- MoE 的训练稳定性对"被当成 $\Theta(1)$ 的常数倍率"显著比 dense 模型敏感；这意味着拿 dense 上 work 的常数复用到 MoE 会出问题，必须按参数组分别再调一次常数。
- 用本参数化训出来的 MoE 自然出现均匀的 expert load（Figure 17），不需要 auxiliary loss 强行约束；这是 sigmoid 路由 + bias 更新 + LR 尺度匹配联合作用的结果。
- 长 horizon（5B–10B tokens）训练时，固定 stable LR 也能收敛，但叠加 cosine 衰减到 0 仍可显著降低 val loss。
- 等参数量下 *专家数多、专家小* 比"专家少、专家大"更好的现象在不调 HP 的前提下得到了重新确认，说明之前文献里这条结论不是某种 HP 调优 artifact。

## 亮点与洞察
- 把"每个 expert 的混合系数和输出都单独满足 $\Delta z=\Theta(1)$"作为加强版 μP 条件，是把稀疏路由纳入 μP 框架最关键的一步：它把 router、expert、bias 三组本来耦合的 LR 解耦推导成可机械化套用的规则。
- 三层 mean-field 嵌套（残差神经元 / expert / expert-内神经元）是把 self-attention 多头分析（Bordelon et al., 2024）的"测度套测度"思路成功迁移到 MoE 上的体现，给后续把这一套搬到其它稀疏架构（MoA、shared-expert、deep router）留了清晰范式。
- $W_{\text{down}}$ 的 init 多出来一个 $\alpha_{\text{ffn}}^{-1}$ 因子这一条很反直觉（标准 fan-in 不会这么写），却是让 $\alpha_{\text{ffn}}$ 真的能 zero-shot 迁移的关键；这种"用 mean-field 视角重新解释 $\alpha_{\text{ffn}}$ 作为中间宽度"是可迁移到其它带 hidden multiplier 模块的 trick。
- 在固定 $\kappa$ 这个"扩展观"层面，本文用 DMFT 给了 mean-field 测度意义上的解释，从此 MoE scaling law 研究里 $\kappa\to 0$ 与 $\kappa=\text{const}$ 是 *本质不同的两个 limit* 这件事有了形式化背书。

## 局限与展望
- DMFT 分析里 self-attention 模块没纳入，作者认为"没有技术障碍"但确实没写完整版；MoE 与 attention 的 mean-field 耦合是否会带来额外修正项尚未验证。
- 零样本 HP 只迁移了 base LR 与 init std，weight decay、router 温度、bias LR 这些 *常数倍率* 仍要在每个新 scale 重新调（论文承认 MoE 对常数倍率更敏感），这削弱了"零样本"的彻底性。
- 实验最大规模到 2B 总参 / 10B tokens，距离实际工业 MoE（百亿–万亿激活、$T$ 量级 tokens）还有量级差距，是否在更长 horizon 上需要修正还是开放问题。
- 只验证了 $\kappa=1/4$ 与 $1/12$ 两档稀疏度且每档单独定一组规则，跨 $\kappa$ 的 HP 关系仍是经验黑盒。
- 路由用的是 sigmoid + bias，不覆盖 softmax-top-$k$、expert-choice 等主流路由变体；虽然 Figure 18 给了一点 sigmoid vs softmax 对比，但 expert-choice 这种 token-to-expert 反向路由是否仍满足相同推导未做。

## 相关工作与启发
- **vs μP (Yang & Hu 2022) / CompleteP (Dey et al. 2025)**：它们针对 dense MLP / Transformer 给出 width 与 width+depth 的 HP 迁移；本文不动 CompleteP 的 width/depth 规则，专门补 MoE 模块（router + expert + bias）并新增 $n_{\text{exp}},\alpha_{\text{ffn}}$ 两条 scale 轴。
- **vs Bordelon & Pehlevan 系列 DMFT**：它们做的是 MLP、ResNet、self-attention 的 mean-field；本文给出 MoE 三层 mean-field 嵌套（含硬路由的分位阈值），技术上是把 self-attention 多头思路推广到稀疏专家。
- **vs Malasnicki et al. 2025**：并发工作，只研究 MoE 跨 width 的 base LR 迁移；本文同时迁移 LR 与 init，覆盖 width/depth/$n_{\text{exp}}$/$\alpha_{\text{ffn}}$ 四轴，并补上严格 DMFT 证明。
- **vs Vankadara et al. 2026**：同期 MoE DMFT 工作，但 *scale-invariant limit* 的定义不同（他们做更复杂的 regime 分析）；两者互补而非冲突。
- **vs Krajewski et al. 2024 / Boix-Adsera & Rigollet 2025**：他们用各自调好的 HP 报告"更多更小 expert 更好"；本文在零样本 HP 下复现该结论，证明这不是 HP 调优偏好的 artifact。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 μP/CompleteP + DMFT 套到 MoE 是自然推广，但补齐 $\alpha_{\text{ffn}}$、$n_{\text{exp}}$ 两轴的细节与三层 mean-field 推导是实打实的新东西
- 实验充分度: ⭐⭐⭐⭐ 覆盖 38M→2B、两种 $\kappa$、两个数据集、4 条 scale 轴 + 10B token 长 horizon 对比 GPT2 speedrun，缺的是工业级 scale 验证
- 写作质量: ⭐⭐⭐⭐ 把"启发式 μP 推不出来的东西"和"DMFT 必须出场"的分工讲得很清楚；附录支撑完整，主文留得克制
- 价值: ⭐⭐⭐⭐ 给后续所有要 scale MoE 的人提供了一个可直接套表 1 的工具，且把"固定 $\kappa$"这件事的理论根据钉死了

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)

</div>

<!-- RELATED:END -->
