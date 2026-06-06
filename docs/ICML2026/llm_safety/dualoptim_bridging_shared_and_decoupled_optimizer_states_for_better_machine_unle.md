---
title: >-
  [论文解读] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models
description: >-
  [ICML 2026][LLM安全][机器遗忘] DualOptim+ 把 Adam 优化器状态拆成"共享 base 态 + 解耦 delta 态"，让 LLM 机器遗忘在 forget/retain 梯度时而冲突时而协同的情况下自适应地在共享和解耦优化器之间过渡…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "机器遗忘"
  - "优化器状态"
  - "梯度冲突"
  - "8-bit 量化"
  - "AdamW"
---

# DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models

**会议**: ICML 2026  
**arXiv**: [2605.21539](https://arxiv.org/abs/2605.21539)  
**代码**: https://github.com/CityU-MLO/DualOptimPlus  
**领域**: LLM安全 / 机器遗忘 / 优化器  
**关键词**: 机器遗忘, 优化器状态, 梯度冲突, 8-bit 量化, AdamW

## 一句话总结
DualOptim+ 把 Adam 优化器状态拆成"共享 base 态 + 解耦 delta 态"，让 LLM 机器遗忘在 forget/retain 梯度时而冲突时而协同的情况下自适应地在共享和解耦优化器之间过渡，理论上同时退化为 Alternate（正相关）和 DualOptim（负相关），并通过 8-bit 量化变体把额外显存开销压回基线。

## 研究背景与动机

**领域现状**：机器遗忘（MU）要求模型擦除 forget 集影响、保留 retain 集效用。LLM 上的 MU 主流靠 forget 损失 $\mathcal{L}_f$（GA / NPO / ME / RMU）+ retain 损失 $\mathcal{L}_r$（CE / KL）联合优化。优化策略经历三代：
- **Joint**（求和后单步反传，DualOptim 之前的事实标准）—— 简单但梯度合并导致退化
- **Alternate**（每步只用一个目标的梯度，交替）—— 缓解退化但对超参敏感、不稳
- **DualOptim**（两个目标用两个独立的 AdamW，各自维护状态）—— 视觉任务上有效，但移植到 LLM 收益仅边际

**现有痛点**：作者观察到 LLM unlearning 中 forget/retain 梯度的余弦相似度在训练过程中**剧烈波动**——前期正相关（信号共享）、后期负相关甚至接近正交，时而冲突时而协同。Joint 用一个共享态丢掉了对抗信号；DualOptim 完全解耦又丢掉了协同信号；二者都只对应了一种相关模式，所以 LLM 上无法全程拿到最优。

**核心矛盾**：单一的"共享 vs 解耦"二选一无法适应 LLM 训练中梯度相关性的动态变化；理想优化器应该能根据当前相关性自适应地在两者间过渡。

**本文目标**：（1）构造一种 plug-and-play 的优化器框架，能根据 $\nabla \mathcal{L}_f$ 与 $\nabla \mathcal{L}_r$ 的方向相关性动态在共享/解耦间插值；（2）覆盖 fictitious unlearning / 真实遗忘 / 安全对齐 / 多任务等场景；（3）解决额外优化器状态带来的显存膨胀。

**切入角度**：把 AdamW 的一阶/二阶矩各自分解成"shared base + per-objective delta"——base 用所有梯度一起更新（捕捉共性），delta 用"该目标梯度 − base"更新（捕捉差异）；参数更新用 base + 对应 delta。这样在数学上自然得到自适应过渡：相关性高时 delta → 0（退化为 Alternate）、相关性高度负时 base → 0（退化为 DualOptim）。

**核心 idea**：base/delta 分解 + 自适应过渡 = 优化器层面的"shared 与 decoupled 的最优中间体"。

## 方法详解

### 整体框架

每个优化器状态（AdamW 的 $m$、$v$）拆为：
- **base 态** $B$：被 $\nabla \mathcal{L}_f$ 和 $\nabla \mathcal{L}_r$ 共同更新，承载共性
- **delta 态** $\Delta_f, \Delta_r$：分别由"该目标梯度与 base 的残差"更新，承载差异

每步用 base + 对应 delta 给参数：$\theta \leftarrow \theta - \eta (\hat B + \hat \Delta_o) / (\sqrt{|\hat v_B + \hat v_{\Delta_o}|} + \epsilon)$；base 在参数更新之后再更新（稳定参考）。配合交替调度 $F_f$ 步 forget + $F_r$ 步 retain。

### 关键设计

1. **Base 态与 Delta 态分解**:

    - 功能：把单个优化器状态拆成共享部分（base）和目标特异部分（delta），保留两路信号
    - 核心思路：base $B \leftarrow \beta B + (1-\beta) \nabla \mathcal{L}_o$（$o$ 为当前目标），delta $\Delta_o \leftarrow \beta \Delta_o + (1-\beta)(\nabla \mathcal{L}_o - \hat B)$；二阶矩 $v_B, v_{\Delta_o}$ 同理用平方梯度更新；偏差修正 $\hat B = B / (1-\beta^t)$
    - 设计动机：base 学到的是 forget/retain 都同意的方向（多任务共性），delta 学到的是各自独立的方向（对抗成分）；二者相加，既不会被 Joint 那样合并掉对抗信号，也不会像 DualOptim 那样丢掉协同信号

2. **自适应过渡（理论性质）**:

    - 功能：根据 forget/retain 梯度的方向相关性，自动在 Alternate 与 DualOptim 之间过渡
    - 核心思路：Theorem 3.2 给出极限分析——设 $\mathbb{E}_t[g_{f,t}] = mG$, $\mathbb{E}_t[g_{r,t}] = nG$：
        - $m = n$（梯度正相关）→ $B \to mG$、$\Delta_{f,r} \to 0$，等价于 Alternate（共享态）
        - $m = -\frac{1-\beta^{F_r}}{\beta^{F_r}(1-\beta^{F_f})}n$（强负相关）→ $B \to 0$，仅 delta 起作用，等价于 DualOptim（完全解耦）
    - 设计动机：不需要任何相关性检测的开关，自适应行为是优化器结构自带的；这让 LLM 训练中相关性动态变化时无需调参

3. **DualOptim+ 8bit（显存控制）**:

    - 功能：把额外 base + delta 状态量化到 8-bit，把显存开销压回 vanilla AdamW 水平
    - 核心思路：参考 bitsandbytes 8-bit Adam，对 $B, \Delta_f, \Delta_r$ 都用块状量化；论文报告量化版与 fp32 版性能几乎相同
    - 设计动机：base + delta 比 vanilla AdamW 多 2× 显存（一阶+二阶各多两份），LLM 上不可接受；8-bit 量化是必要的工程优化，让方法实际可用

### 训练调度
$F_f, F_r$ 控制 forget/retain 交替频率（实验取 1:1）；base 在参数更新后再更新以稳定参考；交替模式比纯交替更稳。

## 实验关键数据

### TOFU Fictitious Unlearning（Phi-1.5，IDK+GD 目标）

| Forget 比例 | 方法 | UFE↑ | TFE↑ | MU↑ | OVR↑ |
|----------|------|------|------|-----|------|
| 10% | Joint | 78.1 | 50.6 | 60.2 | 62.3 |
| 10% | Alternate | 80.7 | 56.8 | 64.5 | 66.6 |
| 10% | DualOptim | 81.2 | 58.3 | 65.0 | 67.4 |
| 10% | **DualOptim+** | **84.8** | **62.7** | **68.1** | **70.9** |
| 10% | DualOptim+ 8bit | 84.5 | 62.4 | 67.9 | 70.7 |

OVR 提升 ~3.5 点；遗忘效率（UFE / TFE）与模型效用（MU）同时改善，没有 trade-off。

### 真实遗忘 + 安全对齐（部分摘录）

| 任务 | 数据 | Joint OVR | DualOptim OVR | **DualOptim+ OVR** |
|------|------|---------|--------------|------|
| WMDP-Bio (Llama 2-7B) | 真实遗忘 | 51.2 | 54.7 | **58.9** |
| WMDP-Cyber | 真实遗忘 | 49.6 | 52.3 | **56.4** |
| Harm-Refuse | 安全对齐 | 62.8 | 66.1 | **70.2** |

跨任务一致领先 4–5 点。

### 优化器更新相似度（Figure 2 数值）

- Alternate（共享态）：相邻 forget/retain 更新余弦相似度 ≈ 0.95（信号几乎被合并）
- DualOptim（完全解耦）：≈ 0.0（信号互相独立）
- **DualOptim+**：≈ 0.4–0.6（在两者之间，且随训练阶段动态变化）

直接验证了"自适应过渡"假设。

### 关键发现
- **梯度相关性确实动态变化**：Figure 2(b) 显示余弦相似度在 [-0.5, 0.7] 区间剧烈波动，验证"静态共享 / 静态解耦都不优"
- **DualOptim+ 是上述区间的合适中间值**：观察到的 0.4–0.6 区间恰好与理论极限一致
- **量化几乎无损**：8-bit 与 fp32 OVR 差距 < 0.3 点，工程上完全可用
- **跨优化器迁移**：在 AdamW 之外的 Muon 上同样有效（Appendix），说明 base/delta 分解的通用性

## 亮点与洞察
- **优化器结构层面的自适应**：以往多目标优化主要通过手工设计权重调度或显式投影解决冲突；本文直接把"过渡"埋进优化器状态结构，无需任何外部信号或检测
- **干净的理论极限**：Theorem 3.2 给出闭式渐近行为，两端极限正好对应 Alternate 和 DualOptim——这是个少见的"中间体方法但有干净理论"的工作
- **8-bit 量化的工程意识**：作者意识到 LLM 上的 2× 状态开销是部署杀手，主动做量化；这种"算法 + 工程"配套发布是当前 LLM 研究越来越重要的范式
- **超越遗忘的潜力**：base/delta 分解本质上是个多目标优化器框架，作者已在多任务和安全对齐上验证有效，且与联邦学习中的 SCAFFOLD / FedProx 类方法在结构上有亲缘性——值得在更多场景（如 RLHF/DPO + KL 正则、多专家蒸馏）尝试

## 局限性 / 可改进方向
- 多于 2 个目标的扩展不平凡：base/delta 在 $k$ 目标下需要 $1 + k$ 个状态，显存进一步膨胀；量化方案如何 scale 未讨论
- $F_f, F_r$ 仍是手工超参，自动化调度（如根据当前相关性自适应调）会更好
- 主要在 7B 以下模型验证（Phi-1.5、Llama-2-7B）；70B 量级的真实部署效果未测
- 与同期方法（GradDiff、SimNPO 等）的对比可以更细，特别是在长 unlearning 训练后效用恢复方面

## 相关工作与启发
- **vs DualOptim**：DualOptim 完全解耦，丢失协同信号；DualOptim+ 引入共享 base，自适应过渡，理论上完备覆盖 DualOptim 作为一个极限
- **vs Joint / Alternate**：Joint 退化所有信号，Alternate 共享状态——都是单点策略；DualOptim+ 是连续族
- **vs 联邦学习 SCAFFOLD / FedProx**：base/delta 分解形式上类似 SCAFFOLD 的 server + client 控制变量，但目标场景（unlearning vs federation）和具体更新规则不同
- **启发**：任何"多目标且目标间相关性动态变化"的训练问题（RLHF + KL、多专家蒸馏、多模态对齐）都可考虑借鉴 base/delta 分解；它把"何时共享、何时解耦"从超参变成数学自动选择

## 评分
- 新颖性: ⭐⭐⭐⭐ base/delta 分解本身简洁但有效，"自适应中间体"的 framing 是真正贡献
- 实验充分度: ⭐⭐⭐⭐⭐ TOFU + WMDP + 安全对齐 + 多任务，覆盖完整；量化、跨优化器、消融都做到
- 写作质量: ⭐⭐⭐⭐ 动机引入清晰，理论 Theorem 3.2 给出干净极限；Figure 2 的相关性可视化对论证有力
- 价值: ⭐⭐⭐⭐ LLM unlearning 是当前 AI 安全和合规的刚需；DualOptim+ 是已知公开方法中在 TOFU/WMDP 上最强的优化器侧改进之一

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](../../ACL2025/llm_safety/mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/llm_safety/munba_machine_unlearning_via_nash_bargaining.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)

</div>

<!-- RELATED:END -->
