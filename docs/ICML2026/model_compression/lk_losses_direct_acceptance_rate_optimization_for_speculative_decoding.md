---
title: >-
  [论文解读] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding
description: >-
  [ICML 2026][模型压缩][推测解码] 本文指出推测解码训练时长期用 KL 散度作为接受率的 proxy 是次优的——小容量 draft 模型在有限容量下 KL 最小化不蕴含接受率最大化…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "推测解码"
  - "接受率优化"
  - "KL vs TV 散度"
  - "草稿模型训练"
  - "EAGLE/MEDUSA"
---

# LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding

**会议**: ICML 2026  
**arXiv**: [2602.23881](https://arxiv.org/abs/2602.23881)  
**代码**: https://huggingface.co/nebius/lk-speculators  
**领域**: 模型压缩 / 推测解码 / LLM 推理加速  
**关键词**: 推测解码, 接受率优化, KL vs TV 散度, 草稿模型训练, EAGLE/MEDUSA

## 一句话总结
本文指出推测解码训练时长期用 KL 散度作为接受率的 proxy 是次优的——小容量 draft 模型在有限容量下 KL 最小化不蕴含接受率最大化；提出 LK losses（直接最大化负 log 接受率 + 与 KL 的 trust-region 混合）作为 plug-in 替代，4 个 draft 架构 × 6 个 target 模型（8B-685B）一致提升 8-10% 平均接受长度。

## 研究背景与动机

**领域现状**：LLM 推理被 memory bandwidth 卡住，autoregressive 单 token 解码利用率低。推测解码（speculative decoding）让小 draft 模型 $q$ 提议 $K$ 个 token，target 模型 $p$ 并行验证：每 token 接受概率 $\beta = \min(1, p/q)$，第一个被拒绝就丢后续。已有架构包括 MEDUSA（并行预测头）、EAGLE/EAGLE-3（自回归 draft head + 特征融合）、MTP（DeepSeek-V3 原生 draft 模块）。

**现有痛点**：所有这些 draft 模型训练都用 KL 散度（或等价 CE）作为目标，把 KL 当作接受率的 proxy。理论上 $q = p$ 时 KL = 0 且接受率 = 1，所以全局最优一致；但 draft 模型容量仅是 target 的 1-5%，永远到不了全局最优——次优点上 KL 最小化与接受率最大化**无任何保证**。

**核心矛盾**：直接优化的目标是接受率（实际等价于 $1 - \text{TV}(p, q)$，TV = Total Variation 距离）；但训练用 KL，二者在次优点行为完全不同——KL forward 是 mode-covering（铺开支撑导致接受率次优），KL reverse 是 mode-seeking（折叠到 dominant mode），都不是最大化分布重叠。

**本文目标**：用直接 targeting 接受率的损失替换 KL，要求（1）适用 native 训练（从头训）的 draft module 而非外部 pre-trained speculator；（2）实现简单零计算开销；（3）跨架构/跨规模通用。

**切入角度**：DistillSpec 早就发现 TV 距离是接受率的精确数学对应，但它在已 pre-trained LM 上效果差因为 TV 梯度太弱（mass concentrated tokens）——本文发现这一限制只在已 pretrained 场景成立，对**随机初始化的 native draft 模块**，TV-style 直接接受率损失才有效。

**核心 idea**：LK losses ——（a）直接最大化负 log 接受率（"LK-direct"，类似 maximum likelihood）；（b）KL → LK 渐进切换（"LK-hybrid"，类 trust region 方法，开始用 KL 稳，后期切到 LK 直接）。

## 方法详解

### 整体框架

接受率定义 $\alpha = \sum_x \min(q(x), p(x)) = 1 - \text{TV}(p, q)$。

LK losses 两个变种：
- **LK-direct**：$\mathcal{L}_{\text{LK}} = -\log \alpha = -\log \sum_x \min(q(x), p(x))$
- **LK-hybrid**：$\mathcal{L}_{\text{hybrid}}(t) = (1 - w(t)) \cdot \text{KL}(p \| q) + w(t) \cdot \mathcal{L}_{\text{LK}}$，$w(t)$ 从 0 渐增到 1

无需架构改动，直接换损失函数即可。

### 关键设计

1. **LK-direct：直接负 log 接受率优化**:

    - 功能：把训练目标直接对齐推理时的接受率（与 TV 一一对应）
    - 核心思路：损失 $\mathcal{L}_{\text{LK}} = -\log \alpha$，对 draft logits $z_q$ 求梯度——其结构与 KL 完全不同：KL 梯度 $\nabla_{z_q} \text{KL}(p \| q) = q - p$ 在所有 token 上均匀施压；LK 梯度集中在分布重叠区域，类似 TV 的 mass-focused 性质，让 draft 模型集中容量在 target 的高概率 token 上
    - 设计动机：限定容量下，"覆盖所有支撑（KL 强迫）"vs"对齐高概率区域（TV/LK 偏好）"是质的不同；推测解码场景下后者直接换接受率，理应优于前者

2. **LK-hybrid：KL → LK 渐进切换的 trust-region 思路**:

    - 功能：early training 阶段用 KL 提供稳健全局梯度信号；late training 切换到 LK 精修接受率
    - 核心思路：$\mathcal{L}_{\text{hybrid}}(t) = (1-w(t)) \text{KL}(p\|q) + w(t) \mathcal{L}_{\text{LK}}$，$w(t)$ 按 schedule（如线性 or sigmoid）从 0 增到 1；类似 trust region 方法在稳定 surrogate 和真目标间平衡
    - 设计动机：随机初始化时 KL 梯度 $\|q - p\| \sim \mathcal{O}(1/\sqrt{k})$ 提供强信号；LK 梯度在分布完全错位时几乎为零（$\min(q, p) \to 0$）；hybrid 先用 KL 把分布拉到大致重叠，再切 LK 精修；这是个非常 elegant 的 curriculum 设计

3. **梯度结构分析（理论支撑）**:

    - 功能：从数学上解释为什么 LK 在次优点更优
    - 核心思路：KL forward 梯度 $\nabla_{z_q} \text{KL} = q - p$，在所有 token 上均匀施压（mass-covering 偏好）；LK 梯度（论文附录推导）只在 $q < p$ 的 token 上有贡献——只对"draft 低估的、target 高概率的"token 加分；这种 selective updating 让有限容量集中投资在 high-impact token
    - 设计动机：理论解释让方法不再是"撞运气"——次优点 KL 强制 covers，浪费容量；LK selective 投资正是接受率所要

## 实验关键数据

### 主实验：4 个 draft 架构 × 6 个 target 模型

| Draft 架构 | Target 模型 | KL 训练 接受长度 $\tau$ | **LK 训练 $\tau$** | 提升 |
|--------|------|------|------|------|
| MEDUSA-3 head | Llama-3-8B | 2.31 | 2.48 | +7.4% |
| EAGLE-3 | Llama-3-70B | 3.12 | 3.45 | +10.6% |
| EAGLE-3 | Qwen3-235B-A22B | 2.85 | 3.16 | +10.9% |
| EAGLE-3 | DeepSeek-V3 (685B) | 2.94 | 3.21 | +9.2% |
| MTP module | DeepSeek-V3 | 2.78 | 3.02 | +8.6% |
| Standalone Qwen-1.5B → Llama-3-70B | – | 2.46 | 2.68 | +8.9% |

8-10% 平均接受长度提升跨所有架构 × 规模一致；越长的 $K$ 优势越明显（Figure 1）。

### 领域分布

| 领域 | KL $\tau$ | LK $\tau$ | 提升 |
|------|------|------|------|
| 通用 (MT-Bench) | 2.85 | 3.10 | +8.8% |
| 代码 (HumanEval) | 3.12 | 3.43 | +9.9% |
| 数学 (GSM8K) | 2.78 | 3.05 | +9.7% |

代码 / 数学领域 LK 增益更大（这些任务上 token 分布更偏，长尾更重，TV-style 集中度更显效益）。

### LK-direct vs LK-hybrid 对比

| 训练阶段 | KL | LK-direct | LK-hybrid |
|--------|-----|---------|----------|
| Early (random init) | $\tau=2.10$ 稳收敛 | $\tau=1.85$ 慢起 | $\tau=2.12$ 稳 |
| Late (well-trained) | $\tau=2.85$ | $\tau=3.10$ | $\tau=3.13$ |

LK-direct 早期收敛慢（梯度信号弱）；LK-hybrid 兼得早期 KL 稳和后期 LK 精修，最终略优。

### 关键发现
- **KL 在小容量 draft 下确实次优**：8-10% 接受率提升是稳定的、跨架构跨规模、跨领域
- **越长 $K$ 越受益**：Figure 1 中接受长度 $\tau$ 在 $K \geq 4$ 时 LK 比 KL 拉开差距，因为长序列累积乘 acceptance 概率，单步接受率小幅提升复利放大
- **trust-region hybrid 是工程上更稳的选择**：LK-direct 早期慢，hybrid 解决冷启动
- **plug-and-play 实测**：换损失函数 1 行代码改完，无任何额外计算或架构开销

## 亮点与洞察
- **指出长期被忽视的目标-代理不一致**：KL 作为接受率代理在小容量下不仅次优而且方向错（mass-covering 浪费容量）——这套"重新审视训练目标 vs 评测目标对齐"的视角可推广到所有 KD 场景
- **LK-hybrid 的 trust-region 设计哲学**：先用稳的 surrogate 再切真目标，是个通用 curriculum 模板，适用于所有"真目标梯度稀疏但 surrogate 梯度强"的场景（如 RLHF 的 reward shaping、preference learning）
- **梯度结构分析提供理论支撑**：不只是经验工作——KL 梯度均匀施压 vs LK 选择性投资的对比，定量解释了"为什么"
- **跨 685B 规模验证泛化**：在 DeepSeek-V3 这种 frontier 模型上仍有 9% 提升，证明方法不挑规模

## 局限性 / 可改进方向
- $w(t)$ 切换 schedule 仍是手工设计，自适应（按当前 KL/LK 比例）会更鲁棒
- 在已 pretrained 的 standalone draft 上 LK 效果是否一致未充分验证（DistillSpec 报告 pretrained 下 TV 效果差）
- LK-direct 在分布完全错位时梯度为零，对极端坏的初始化可能完全学不到——hybrid 部分解决但需保证早期 KL 阶段足够长
- 8-10% 接受率提升对应到端到端推理 speedup 多少没明确说，需要看 verifier 步骤的实际开销
- 没探索更激进的目标（如直接最大化端到端 throughput），可能还有进一步空间

## 相关工作与启发
- **vs DistillSpec**：DistillSpec 在 pretrained external draft 上探索 TV，效果不稳；本文在 native 随机初始化 draft 上 LK 效果稳定，揭示 setting 差异
- **vs MEDUSA / EAGLE / EAGLE-3 / MTP**：那些都用 KL 训练；本文换 LK 即可获得跨架构提升，对已有系统升级路径友好
- **vs AdaSPEC（选择性 KD）**：那个针对 standalone draft；本文针对 native draft module 且更通用
- **启发**：所有"训练目标 vs 评测目标错位"的场景都值得用本文的"梯度结构分析 + trust-region hybrid"思路重审；KD 场景中 KL 不是默认最优，要看下游评测对应什么数学量

## 评分
- 新颖性: ⭐⭐⭐⭐ LK-direct 概念 DistillSpec 已有，但本文澄清了 native vs pretrained 的 setting 差异并提出 LK-hybrid
- 实验充分度: ⭐⭐⭐⭐⭐ 4 架构 × 6 模型 × 3 领域 + 梯度结构理论分析 + LK-direct vs hybrid 消融，覆盖完整
- 写作质量: ⭐⭐⭐⭐⭐ Figure 2 的 Gaussian fit 玩具例子直观，梯度分析数学清晰
- 价值: ⭐⭐⭐⭐⭐ 推测解码是 LLM 推理加速主流方案，8-10% 接受率提升对应可观 throughput 提升；plug-and-play 让所有现有 draft 训练管线立即受益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)
- [\[ACL 2026\] SSSD: Simply-Scalable Speculative Decoding](../../ACL2026/model_compression/sssd_simply-scalable_speculative_decoding.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)

</div>

<!-- RELATED:END -->
