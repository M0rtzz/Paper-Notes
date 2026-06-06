---
title: >-
  [论文解读] Scaling Depth Capacity via Zero/One-Layer Model Expansion
description: >-
  [ICML 2026][预训练][progressive training] 本文提出"零层/一层渐进式训练"——先训一个几乎没有 Transformer 层的极浅模型，再在训练后期（≈80% iterations）一次性把深度扩展到目标层数，配合 WSD 学习率和 muP 超参传递…
tags:
  - "ICML 2026"
  - "预训练"
  - "progressive training"
  - "深度扩展"
  - "zero/one-layer"
  - "WSD schedule"
  - "muP"
---

# Scaling Depth Capacity via Zero/One-Layer Model Expansion

**会议**: ICML 2026  
**arXiv**: [2511.04981](https://arxiv.org/abs/2511.04981)  
**代码**: 无  
**领域**: LLM预训练 / 高效训练 / 模型扩展  
**关键词**: progressive training, 深度扩展, zero/one-layer, WSD schedule, muP

## 一句话总结
本文提出"零层/一层渐进式训练"——先训一个几乎没有 Transformer 层的极浅模型，再在训练后期（≈80% iterations）一次性把深度扩展到目标层数，配合 WSD 学习率和 muP 超参传递，可在 GPT2/LLAMA3/DeepSeekV3 上节省约 80% 计算（≈5× 加速）且最终 loss 几乎不掉。

## 研究背景与动机
**领域现状**：训练大模型代价惊人（LLAMA-4 训练 >7M GPU 小时），主流加速思路之一是 **progressive training / model expansion**：先训一个小的"教师/源模型"，然后某个时刻 $t=\tau$ 把模型扩展到大尺寸继续训练。其计算量近似为 $6B(\tau N_{\text{small}} + (T-\tau) N_{\text{large}})$，比固定大尺寸训练的 $6BTN_{\text{large}}$ 显著更小，前提是 $\tau$ 接近 $T$ 且 $N_{\text{small}} \ll N_{\text{large}}$。

**现有痛点**：现有方法把深度扩展限制在 2-4×、源模型层数仍要十几层，因此节省的计算只有 ≈30-45%（grown vs target 的对比口径），而且大多数工作只在 BERT/ViT 等分类模型上验证，在生成式 LLM 上只能拿到 1.4-2× 加速。更糟的是，多阶段扩展（如 0→2→12）虽然形式上更"渐进"，却没有展示出能跨越扩展点的 *mixing*（loss 追平）行为。

**核心矛盾**：现有方法在两个维度上都没推到极限——一是没人敢真用 0/1 层这种极浅源模型（太极端，不知道能不能学到东西迁移给大模型）；二是 *function-preserving* 初始化（如 zero-init 子层）与 *feature learning* 是冲突的：zero 让 loss 不跳变，但梯度死了、新层学不动；同时学习率调度（cosine 默认在后期已衰减到几乎为 0）也使得"晚扩展"根本来不及收敛。

**本文目标**：（1）把源模型推到极端的 0 或 1 层；（2）把扩展时刻 $\tau$ 推到 0.8T；（3）保证扩展前后超参不用换；（4）给一套覆盖 dense/MoE、MHA/GQA/MLA、cosine/WSD 的统一recipe，并配上凸优化收敛证明解释为什么这套东西 work。

**切入角度**：把"深度扩展"重新建模为大模型训练的一个**初始化问题**——把大模型 $\mathbf{W}_t = [\mathbf{w}_t, \mathbf{x}_t]$ 拆成"小模型部分 + 新增层部分"，则渐进式训练等价于先对 $\mathbf{x}$ 做投影梯度下降（mask 到 0）+ 一次到良好初始化的"瞬移"+ 之后正常 SGD。在这个统一视角下，初始化策略和学习率调度都能用 convex+Lipschitz loss 的收敛界推出来。

**核心 idea**：零/一层 progressive training + WSD schedule + muP 超参传递 = 在"loss-compute Pareto 前沿"上把现有工作整体往左下推一大截。

## 方法详解

### 整体框架
Pipeline 极其简洁：先训练一个 0 层（只有 Embedding + LM_head + 最终 LayerNorm，*完全没有* Transformer 层）或 1 层模型；在 WSD schedule 的 *稳定阶段* 选一个时刻 $\tau \approx 0.8T$，把模型一次性扩展到目标深度 $L$ 层（zero-layer 只能 random init 新层；one-layer 既可 random 也可 copying，例如 $\mathbf{w}\to[\mathbf{w},\mathbf{w},\mathbf{w}]$）；扩展后**沿用同一个学习率**继续训到底。这套流程在 GPT2 / LLAMA3 / Qwen3 / Mixtral / DeepSeekV3 / ResNet 上都同款，覆盖 weight-tying、dense/MoE、MHA/GQA/MLA、绝对/旋转位置编码、LayerNorm/RMSNorm、GeLU/SwiGLU 各种变种。

### 关键设计

1. **深度扩展的"初始化-理论"统一视角**:

    - 功能：把扩展后大模型的训练 loss 上界与 fixed-size 训练 loss 上界做差，得到二者 gap 的显式刻画。
    - 核心思路：把 $\mathbf{W}_t=[\mathbf{w}_t,\mathbf{x}_t]$ 拆成"复用部分+新增部分"，并假设 $\mathbf{W}^*=[\mathbf{w}^*,\mathbf{x}^*]$，则在 convex + $G$-Lipschitz loss 下两段 SGD 通过 telescoping 得到 gap = $\frac{\sum_{t=1}^{\tau}\eta_t}{\sum_{t=1}^{T}\eta_t}(L(\mathbf{w}^*)-L(\mathbf{W}^*)) + \frac{\|\mathbf{x}_\tau-\mathbf{x}^*\|^2-\|\mathbf{x}_0-\mathbf{x}^*\|^2}{2\sum_{t=1}^{T}\eta_t}$。第一项要求 $L(\mathbf{w}^*) \approx L(\mathbf{W}^*)$，第二项要求把 $\mathbf{x}_\tau$ 初始化得比 $\mathbf{x}_0$ 离最优更近。
    - 设计动机：等价的代数视角是把渐进式训练看成 **PGD（mask 新层为 0）+ 一次瞬移 + SGD**，于是初始化策略（copying vs random）和学习率调度（cosine vs WSD）这两个看似工程的选择都能从同一个 bound 推出方向——random 让第二项=0、copying 让第二项<0；学习率调度上 $\frac{\sum_{t\le\tau}\eta_t}{\sum_t\eta_t}$ 要小，所以扩展前学习率不能太大、扩展后不能衰减太厉害，这正好就是 WSD 的形状。

2. **muP-scaled 初始化 + 零超参重调**:

    - 功能：让 0/1 层小模型与目标深模型使用**同一组超参**（学习率、weight decay 等），扩展瞬间无需重新调参。
    - 核心思路：要求每层激活的 element-wise scale 一致，即 $\|\mathbf{A}_l\|_2/\sqrt{n_l} \sim \|\mathbf{A}_{l+1}\|_2/\sqrt{n_{l+1}}$，落到线性层即 spectral scaling 条件 $\|\mathbf{W}_l\|_* \sim \sqrt{n_{l+1}/n_l}$。在此基础上用 Muon-NSGD（对 2D 张量用 Muon、其他张量用归一化 SGD，共享一个学习率，weight decay=0.01）。新增层用 random Gaussian 满足 muP；用 copying 同样满足；但 zero 和"copying_zero（某些子层置零）"会破坏 feature learning（梯度死掉、新层学不动），所以本文舍弃。
    - 设计动机：渐进训练的最大工程痛点就是"扩展前后超参可能要重调"，而 muP 把超参的最优值在 model size 维度上拉成常数。Table 1 总结了 4 类初始化的三角权衡：copying/random 满足 feature learning 与 trainability 但 *不* function-preserving（loss 会跳一下尖峰）；zero 系列 function-preserving 但堵死学习；本文明确选 trainability + feature learning > function-preserving。

3. **WSD schedule + 晚到 0.8T 的单阶段扩展 + 由"mix-time"反推 $\tau$**:

    - 功能：把扩展时机 $\tau$ 推到训练总长度的 80%，并解释为什么 WSD 是 progressive training 的"天然搭档"。
    - 核心思路：定义 mixing time $t_{\text{mix}}$ 满足 $L(\mathbf{W}_{\tau+t_{\text{mix}}}^{\text{progressive}}) \approx L(\mathbf{W}_{\tau+t_{\text{mix}}}^{\text{fixed-size}})$。实验发现 cosine 下 $t_{\text{mix}}(\tau)$ 对 $\tau$ 极敏感（GPT 上 $\tau\ge 0.5T$ 就追不上），而 WSD 下基本不敏感（$\tau\ge 0.8T$ 仍能追上），这与理论 gap 公式中 $\eta_t$ 在 stable 阶段保持常数完全一致。实操方案：2% warmup + 长 stable 段 + 10% decay，从总长里减去由"一组提前停止的小规模 run"测得的 $t_{\text{mix}}$，得到 $\tau \approx 0.8T$（GPT 124M 实验中 $\tau=480k/528k$）。同时论文论证**单阶段就够了**：基于 mixing behavior，$0\to 2\to 12$ 可以分解为 $0\to 2$ 与 $2\to 12$ 两段，最终 FLOPs 跟 $2\to 12$ 几乎相同、反而比 $0\to 12$ 差。
    - 设计动机：现有工作之所以观察不到 mixing 行为、并因此搞 multi-stage，是因为他们用的是 cosine schedule + "grown-vs-target"的对比视角（Section 5.1 明确指出这是个"perspective"问题）。本文把视角切换到"完整训练过程"+ WSD，立刻看到 mix 行为，于是 single-stage + 晚扩展就足够最优。

### 损失函数 / 训练策略
- 数据：OpenWebText，序列长度 1024，基于 nanoGPT codebase。
- 优化器：Muon-NSGD（主），AdamW 与 SGD 作为补充，weight decay=0.01，不做 gradient clipping。
- 学习率调度：cosine 与 WSD（warmup-stable-decay），衰减到 0；warmup 占 2%。
- token-per-param：LLAMA3 设 50，DeepSeekV3（MoE）设 100。
- 扩展时刻：$\tau \approx 0.8T$（GPT2 124M 用 480k/528k iterations）。

## 实验关键数据

### 主实验
（以 GPT2 在 OpenWebText 上 WSD schedule 为例；"FLOPs ratio" 为相对 fixed-size 训练的计算占比，越低越快。）

| 设置 | 源模型 | 目标模型 | FLOPs ratio | val loss 差距 |
|------|--------|---------|-------------|---------------|
| Fixed-size | — | 12-layer 124M | 100% | 基线 |
| Zero-layer progressive | 0-layer 39M | 12-layer 124M | ≈20% | <0.5% |
| One-layer progressive | 1-layer 46M | 12-layer 124M | ≈20% | <0.5% |
| Fixed-size | — | 60-layer 7B | 100% | 基线 |
| Zero-layer progressive | 0-layer 0.15B | 60-layer 7B | ≈20% | <0.2% |
| One-layer progressive | 1-layer 0.27B | 60-layer 7B | ≈20% | <0.2% |

scaling law 视角：在 LLAMA3 (dense, 0.25B–2B) 与 DeepSeekV3 (MoE, 0.2B–0.5B active) 上，progressive 训练的 scaling exponent 始终优于 fixed-size，整体计算效率提升 3–5×，**且随模型变大优势放大**。

### 消融实验

| 消融维度 | 关键发现 |
|----------|---------|
| 初始化（copying vs random vs zero / copying_zero） | random/copying 都可、且 copying 略优；zero 类初始化破坏 feature learning，新层学不动 |
| 多层扩展顺序（copying_last / _stack / _inter） | copying_last 明显更差；_stack 与 _inter 几乎不可分辨——"复制全部层"是关键 |
| schedule（cosine vs WSD） | WSD 下 $\tau$ 可推到 0.8T 仍能 mix；cosine 下 GPT 在 $\tau\ge 0.5T$、ResNet 在 $\tau\ge 0.7T$ 就追不上 |
| 多阶段（0→2→12 vs 0→12） | 多阶段没有额外收益，FLOPs 与 2→12 接近，劣于直接 0→12 |
| 源模型层数（0/1/2/4/6/8） | loss-compute Pareto 上 0/1 层几乎独占前沿；≥2 层都偏右上方 |

### 关键发现
- **"mixing" 是这套方法的灵魂**：扩展点处的 loss 尖峰看着惨烈，但只要 $\tau + t_{\text{mix}} \le T$，最终 loss 会与 fixed-size 训练几乎重合——这一现象在以往的"grown vs target"对比视角下被系统性掩盖了，本文把视角切回"完整训练"才看出来。
- **mixing time 几乎与小模型大小无关**：从 1-layer 还是 6-layer 扩展，最晚可扩展时刻都在 $\tau/T \approx 0.6$ 附近，但 6-layer 源模型本身已经很贵，所以"越浅的源越优"。
- **WSD vs cosine 的对比有强理论解释**：理论 gap 公式中第一项 $\frac{\sum_{t\le\tau}\eta_t}{\sum_t\eta_t}$ 要小，cosine 在尾段衰减把这个比值推大，WSD 的 stable 段则让它保持小且对 $\tau$ 鲁棒。
- **MoE 上行为一致**：DeepSeekV3、Mixtral 同样呈现 mixing，且本文路线与 upcycling（把小 dense 升到大 MoE 但不加深）正交。

## 亮点与洞察
- **"渐进=初始化"的视角切换**很优雅：把 progressive training 重新表述为大模型的初始化问题 + 一次瞬移，立刻把"初始化策略"和"学习率调度"两件事放进同一个收敛 bound 里推出方向，而不是各自调参。
- **敢把源模型推到 0 层**是这篇真正"压舱"的胆识——0-layer 模型几乎只有 embedding，但在 WSD + muP 加持下足以学到一组好的"瞬移起点"，把扩展时机推到 80% iterations。这种"小到不能再小"的扫荡式 ablation 值得迁移到其他渐进训练场景（宽度扩展、专家数扩展等）。
- **多阶段 = 单阶段的级联**：通过 mixing behavior 把多阶段拆成多个单阶段，发现没有额外好处，反过来证伪了一大票 multi-stage stacking 工作的复杂性。
- **"由小规模 run 反推 $\tau$" 的工程套路**很实用：跑一个 fixed-size + 一个 $\tau=\text{warmup-end}$ 的 progressive，记录什么时候 mix，就能得到 $t_{\text{mix}}$，从而把目标 run 的 $\tau$ 设在 $T - t_{\text{mix}}$。

## 局限性 / 可改进方向
- 收敛理论建立在 convex + Lipschitz 设定上，作者明确承认 deep learning 是非凸的，理论只是"训练动力学相似"层面的指导。
- 实验主要在 OpenWebText + ImageNet 上，最大 dense LLM 是 7B、MoE active 是 0.5B，离前沿 100B+ 规模仍有差距，"越大越占便宜"的趋势能否外推到 100B+ 待验证。
- 只研究了**深度**扩展。宽度、专家数等其他维度的"0-width" 极端是否也成立？作者在结论里提到"同时 scale up width and depth" 是 future work。
- 与 upcycling 路线（dense→MoE 不加深）正交，但没有给出"先零层 progressive 再 upcycling"的组合实验。
- 扩展时刻 $\tau$ 仍需一组 small-scale calibration run 决定，没有给出 closed-form 计算公式。
- 缺乏对 fine-tuning / 后训练（SFT、RLHF）阶段 loss 与下游 benchmark 的评估，只给出 validation loss / scaling law。

## 相关工作与启发
- **vs function-preserving 路线（Net2Net, bert2BERT, LEMON, MSG 等）**：他们追求扩展瞬间 loss 不跳变（function-preserving），代价是 trainability 或 feature learning 受损；本文反过来牺牲 function-preserving、换取 trainability + feature learning，最终 loss 反而更低。
- **vs gradual stacking / multi-stage 路线（gong2019efficient, shen2022staged, du2024stacking 等）**：他们把扩展拆成 3–4 个阶段，最深 2–4× 扩展、节省 ≈30-45% 计算；本文 single-stage 直接 60× 扩展、节省 ≈80% 计算，并解释了为什么 multi-stage 没有额外收益（mixing 行为）。
- **vs 与 muP / WSD 的关系**：本文不发明新的 muP 或新的 WSD，而是把这两个"已知技术"组合到 progressive training 场景中，并给出收敛理论解释，属于"理论+实证双向打通"的工作。
- **vs upcycling MoE（he2024upcycling 等）**：upcycling 是把 dense 模型作为初始化复制到 MoE 专家，scale 的是"专家数/参数总量"而非深度；本文 scale 的是深度，两条路线正交。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把源模型推到 0/1 层、把扩展时刻推到 0.8T、把初始化与学习率调度统一进同一收敛 bound，三个轴都打到了前人没碰过的极限。
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 5 种 LLM 架构 + ResNet、覆盖 dense/MoE、cosine/WSD、150 组扫描 ($\tau$, 源模型大小, 目标层数) Pareto 曲线，且给出 7B 规模验证。
- 写作质量: ⭐⭐⭐⭐ 理论与实验交织清晰，"perspective matters" 一节直接回应了文献误读，逻辑链非常硬；唯一遗憾是部分图表叙述略密。
- 价值: ⭐⭐⭐⭐⭐ 提供了开箱即用的训练 recipe，5× 加速且 loss 几乎不掉，对工业界大模型预训练有直接经济价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Inverse Depth Scaling From Most Layers Being Similar](inverse_depth_scaling_from_most_layers_being_similar.md)
- [\[ACL 2025\] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](../../ACL2025/llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)
- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)

</div>

<!-- RELATED:END -->
