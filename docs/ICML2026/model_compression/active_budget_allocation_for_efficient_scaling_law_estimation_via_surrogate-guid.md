---
title: >-
  [论文解读] Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning
description: >-
  [ICML2026][模型压缩][scaling law] 本文把 scaling law 实验中的训练预算分配建模为多轮资源选择问题，用 Successive Halving 结合学习曲线 surrogate 预测未来潜力…
tags:
  - "ICML2026"
  - "模型压缩"
  - "scaling law"
  - "Successive Halving"
  - "学习曲线"
  - "Gaussian Process"
  - "预算分配"
---

# Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning

**会议**: ICML2026  
**arXiv**: [2605.17234](https://arxiv.org/abs/2605.17234)  
**代码**: 无明确仓库（论文称将公开代码）  
**领域**: 优化 / Scaling Law / 预算分配  
**关键词**: scaling law、Successive Halving、学习曲线、Gaussian Process、预算分配  

## 一句话总结
本文把 scaling law 实验中的训练预算分配建模为多轮资源选择问题，用 Successive Halving 结合学习曲线 surrogate 预测未来潜力，在 synthetic 和 nanoGPT 学习曲线上以最高 98.7% 的训练成本节省近似完整 scaling law。

## 研究背景与动机
**领域现状**：Scaling law 用经验学习曲线描述 loss 与 compute、模型规模、数据规模之间的关系，是规划大模型训练预算、模型尺寸和数据需求的重要工具。经典做法往往需要训练许多不同规模模型，并观察其 loss-compute frontier。

**现有痛点**：完整 scaling law 研究极其昂贵。为了得到可靠 frontier，研究者可能要训练几十到上百个模型到较长 compute 区间，但其中很多模型最终不会贡献到最优前沿。传统 uniform allocation 把预算平均给所有模型，浪费在过小、过早 plateau 或过大但短期表现差的模型上。

**核心矛盾**：小模型早期 loss 下降快，容易在短预算下显得更好；大模型短期 loss 可能不占优，但后期潜力更高。若只按当前 loss 剪枝，会过早淘汰未来能贡献 scaling frontier 的模型；若不剪枝，则算力成本不可承受。

**本文目标**：作者希望在固定总 FLOPs 预算下，主动决定哪些模型继续训练、哪些模型停止，使最终获得的一组 learning curves 足以拟合准确 scaling law，同时大幅降低相对“全模型完整训练”的成本。

**切入角度**：超参数优化中的 Successive Halving 已经能在多配置间分配资源，但它只看已观测 loss。论文进一步让 surrogate model 预测每条 learning curve 的未来 continuation，用“未来可能达到的最低 loss”而不是“当前 loss”决定保留哪些模型。

**核心 idea**：用学习曲线 surrogate 修正 Successive Halving 的短视剪枝，让预算流向对 loss-compute frontier 更有潜力的模型。

## 方法详解

### 整体框架
本文要解决的是：拟合一条准确的 scaling law 需要把许多不同规模的模型都训练到足够长的 compute，但其中大部分模型最终不会落在 loss-compute 前沿上，算力被白白消耗。作者的做法是把数据采集拆成若干轮的资源选择：每轮给当前所有候选模型分配同样的新增 compute，训练出一段 learning curve，再根据这些曲线淘汰一批、把预算集中到留下的模型上，直到耗尽总预算。整个过程沿用超参优化里的 Successive Halving (SH) 框架，但在「按什么标准淘汰」这一步动了刀——不只看曲线当前终点，而是预测每个模型若继续训练能达到的 loss，按未来潜力剪枝。

具体地，输入是初始模型集合 $\mathcal M_0$、总预算 $B$、剪枝系数 $\eta$。第 $r$ 轮里每个仍保留的模型获得预算 $C_r=\lfloor B/(|\mathcal M_r|\lceil\log_\eta |\mathcal M_0|\rceil)\rfloor$，训练到累计预算后形成 learning curve $L_m(C)$。普通 SH 直接按已观测曲线的最低 loss 用 Top$_k$ 选下一轮模型；带 surrogate 的 SH LMC / SH DE 则先用 surrogate 预测每条曲线延伸到最后一轮预算时的 loss，把观测段和预测 continuation 合并起来再选。最终输出只保存真实训练出的曲线（surrogate 的 continuation 绝不当作已观测数据），用它们拟合 compute-loss scaling law。

### 关键设计

**1. 把 scaling law 采样转成预算受限的 proxy 优化：绕开没有金标准的拟合目标。**

「哪组 learning curve 最能拟合 scaling law」本身没有可直接优化的目标——真正的 ground-truth frontier 要把模型完整训练才知道，而这恰恰是最贵的部分。作者因此换了个可计算的 proxy：在总预算 $B$ 内寻找能达到最低 validation loss 的模型集合。优化这个 proxy 的副产物，正是一组被不同程度训练过的 learning curves，拿它们去拟合 scaling law 即可。这样目标就只依赖当前已观测到的训练 loss，可以直接套用 SH 这类 anytime 的资源分配算法来近似，而不必先知道前沿。

**2. LMC Gaussian Process：用跨曲线相关性给大模型「翻案」。**

普通 SH 的硬伤是只看当前 loss——小模型早期下降快，会在短预算下显得更优，于是后期才发力的大模型被过早淘汰。这一设计让 surrogate 根据多条早期曲线预测某个模型后续的 loss 走向，从而保留那些当前不占优、但预测未来更低的大模型。具体把曲线外推建模为一个 multi-input multi-output 的 GP，kernel 由 exponential decay、white noise 和 bias 三个子核组合，再通过 co-regionalisation 矩阵显式捕捉不同模型曲线之间的相关性。这样「小模型大约何时 plateau」「大模型曲线何时由劣转优」这类规律会跨曲线共享，成为外推其他曲线的信号——这正是单曲线外推方法拿不到的先验。

**3. Deep Ensemble surrogate 与 scaling law 外推：换参数化曲线族 + 给前沿配不确定性区间。**

GP 之外作者还试了参数化路线，用来回答「非参数 GP 和参数化 curve family 谁更适合 budget allocation」。Deep Ensemble 用两层 MLP 去条件化 power law、exponential、Morgan-Mercer-Flodin 等曲线函数的系数，预测曲线形状；不同数据集的曲线噪声和形状各异，单一 surrogate 未必最优，多套曲线族给了选择空间。此外 scaling law 常要外推到已训练 compute 区间之外，作者在 SH LMC 之后再用 GP 的 mean / UCB / LCB 把 learning curve 往外延，缩小拟合曲线与 ground truth 的 AbC 差距——UCB/LCB 还顺带给出乐观/悲观的曲线边界，让预算决策不只有一个点估计。

### 损失函数 / 训练策略
LMC GP 用 L-BFGS 加 20 个随机重启优化，Deep Ensemble 用 5 个随机初始化的两层 perceptron 训练 1000 次迭代，实验默认每条曲线抽 20 个观测点训练 surrogate。Scaling law 拟合采用 $L^{SL}(C)=(C/\alpha)^{-\gamma}$，并在指定 compute 区间上用 Area between Curves (AbC) 衡量拟合曲线与 ground-truth scaling law 的距离。

## 实验关键数据

### 主实验
Synthetic learning curves 上，SH LMC 相对普通 SH 有稳定提升，而 uniform allocation 明显更差。

| 模型数 $M_0$ | 预算 $B$ (petaFLOPs) | SH mean loss | SH LMC mean rel. improv. | UA mean rel. degradation | 结论 |
|-------------|----------------------|--------------|---------------------------|---------------------------|------|
| 5 | $10^2$ | 6.40±9.07 | 5.15% (max 20.30%) | -10.17% | 少量模型时 surrogate 改善明显 |
| 5 | $10^4$ | 3.84±2.03 | 5.47% (max 16.70%) | -7.59% | 高预算仍有收益 |
| 10 | $10^3$ | 3.86±0.38 | 2.38% (max 6.11%) | -14.06% | SH 已强，但 LMC 继续改进 |
| 20 | $10^4$ | 3.18±0.09 | 1.50% (max 6.53%) | -16.40% | 模型多时相对收益变小但稳定为正 |

Real-world nanoGPT 学习曲线实验中，SH LMC 也优于 SH 和多数 DE surrogate，并且所有策略都优于 UA。

| $M_0$ | 预算 $B$ | SH mean loss | SH LMC rel. improv. | 最强 DE rel. improv. | UA rel. degradation |
|------|----------|--------------|----------------------|----------------------|---------------------|
| 5 | $10^4$ | 3.17±0.06 | 2.58% | 2.32% (DE EXP) | -5.09% |
| 5 | $10^5$ | 2.97±0.03 | 2.36% | 2.40% (DE PL) | -0.74% |
| 10 | $10^5$ | 3.00±0.02 | 2.82% | 2.14% (DE MMF) | -0.81% |
| 20 | $10^4$ | 3.30±0.02 | 2.84% | 2.02% (DE PL) | -11.46% |
| 20 | $10^5$ | 3.03±0.01 | 2.24% | 1.44% (DE EXP) | -2.96% |

Scaling law 拟合层面，SH 和 SH LMC 都能在远低于完整训练成本的预算下得到接近 ground truth 的 law。

| 设置 | 方法 | AbC vs Full Data SL | Loss regret | 相对完整曲线成本节省 |
|------|------|---------------------|-------------|----------------------|
| $M_0=5,B=10^4$ | SH | 0.09±0.05 | 0.43±0.09 | 94.00% |
| $M_0=5,B=10^4$ | SH LMC | 0.11±0.07 | 0.41±0.10 | 94.00% |
| $M_0=10,B=10^4$ | SH | 0.07±0.02 | 0.56±0.07 | 97.50% |
| $M_0=10,B=10^4$ | SH LMC | 0.09±0.04 | 0.51±0.06 | 97.50% |
| $M_0=20,B=10^4$ | SH | 0.12±0.04 | 0.67±0.03 | 98.70% |
| $M_0=20,B=10^4$ | SH LMC | 0.11±0.07 | 0.59±0.05 | 98.70% |

### 消融实验
论文的关键分析是 surrogate extrapolation 能否弥补已训练 compute range 的不足。

| 预算 $B$ | AbC SH LMC | AbC GP Mean | AbC UCB | AbC LCB | 说明 |
|----------|------------|-------------|---------|---------|------|
| $10^3$ | 5.84 | 0.51±0.27 | 0.62±0.27 | 0.49±0.16 | 低预算下直接曲线偏离大，GP 外推显著修正 |
| $10^4$ | 3.88 | 0.36±0.42 | 0.48±0.13 | 0.45±0.19 | 预算增加后不确定性下降 |
| $10^5$ | 2.17 | 0.00±0.00 | 0.53±0.31 | 0.38±0.16 | GP mean 几乎恢复 ground truth |

| 分析维度 | 观察 | 启示 |
|----------|------|------|
| synthetic clean curves | SH LMC 改善最大可达 5.47% mean / 20.30% max | 曲线规律强时 GP 能很好利用跨曲线相关性 |
| noisy curves | SH LMC 在 white/Brownian/OU 噪声下平均 minimum loss 仍低于 SH | surrogate 对短期噪声有一定鲁棒性 |
| nanoGPT real curves | 相对收益约 2%-3%，小于 synthetic | 真实曲线更接近、更嘈杂，需要更精细预测 |
| UA baseline | 多数设置下明显退化 | 简单平均预算不是 scaling law 采样的好策略 |

### 关键发现
- 普通 SH 已经比 uniform allocation 好很多，但会偏向早期下降快的小模型。加入 surrogate 后，较大但后期潜力高的模型更可能被保留。
- SH LMC 的收益在 synthetic 数据上更明显，在 nanoGPT 上更温和但稳定。考虑到大模型训练成本，即使 2%-3% 的 loss 改善或错误剪枝减少也有实际价值。
- Scaling law 的准确性并不只取决于最低 loss。表 3 中 SH 和 SH LMC 的 AbC 有时接近，说明两者都能形成可用前沿；LMC 更突出的优势是降低 regret 和提供外推/不确定性。
- 成本节省是核心价值。相对训练所有选中模型完整 learning curves，方法可节省 75.61% 到 98.70% 的 compute。

## 亮点与洞察
- 论文把 scaling law 数据采集这件事从“经验上多训几个模型”转成了明确的 resource allocation 问题。这对大模型实验规划很有现实意义。
- 使用 learning-curve surrogate 修正 SH 的短视性很自然。早期小模型好不代表最终 frontier 好，外推模型正好补足这一缺陷。
- 作者没有把 surrogate 预测曲线直接混入最终训练数据，而是只用于剪枝决策，最终 scaling law 仍基于真实训练曲线。这一点让方法比纯外推更稳健。
- GP UCB/LCB 用于 scaling law 区间估计很有启发。实际训练预算决策往往不只要一个点估计，还要知道乐观/悲观曲线范围。

## 局限与展望
- 真实实验只覆盖 nanoGPT 单一模型族，最大到 1.5B 参数。更大规模、不同架构族、不同数据集上的有效性仍需验证。
- Surrogate 训练依赖早期 learning curve 足够有预测性。若存在 late bloomer 模型、训练 regime 切换或数据 curriculum，早期曲线可能误导外推。
- SH LMC 不总是在 AbC 上显著优于 SH，说明最低 loss proxy 与 scaling law 拟合目标仍不完全一致。
- 方法需要预先定义候选模型集合和 compute range。若候选空间本身覆盖不合理，再好的预算分配也难以恢复正确 frontier。
- GP/LMC 与 DE surrogate 的超参和实现复杂度高于普通 SH，真实使用时需要可靠工程工具支持。

## 相关工作与启发
- **vs Uniform Allocation**: UA 简单公平但浪费预算；本文用多轮剪枝把训练集中到更可能贡献 frontier 的模型。
- **vs Successive Halving / Hyperband**: 传统 SH 只按当前表现剪枝，本文让 surrogate 预测未来 learning curve，减少过早淘汰大模型的风险。
- **vs Freeze-Thaw BO**: Freeze-Thaw 等 sequential 方法一次选一个配置，不适合 scaling law 需要并行训练多条曲线的场景；本文强调并行资源分配。
- **vs LC-PFN / 单曲线外推**: 单曲线方法不利用跨模型曲线相关性，LMC 通过 co-regionalisation 捕捉不同规模模型之间的共同趋势。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 SH、学习曲线 surrogate 和 scaling law 数据采集结合得很实用，问题定义清楚。
- 实验充分度: ⭐⭐⭐⭐☆ 有 synthetic、噪声、nanoGPT 和 scaling law AbC 分析；更大模型族仍缺。
- 写作质量: ⭐⭐⭐⭐☆ 方法逻辑完整，表格丰富；符号和附录较多，阅读成本略高。
- 价值: ⭐⭐⭐⭐⭐ 对预算受限的大模型 scaling law 实验非常有价值，可直接影响训练规划。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](../../NeurIPS2025/model_compression/ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[ICML 2026\] A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search](a_language-guided_bayesian_optimization_for_efficient_lora_hyperparameter_search.md)
- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)

</div>

<!-- RELATED:END -->
