---
title: >-
  [论文解读] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models
description: >-
  [ICML 2026][图像恢复][dLLM] 本文系统刻画了 masked 扩散语言模型 (dLLM) 在**完全非自回归 (NAR) 解码**下的失败机制——proximity bias 导致 confidence-based 采样退化为反向自回归并被 EOS 过早占满…
tags:
  - "ICML 2026"
  - "图像恢复"
  - "dLLM"
  - "非自回归解码"
  - "proximity bias"
  - "EOS overflow"
  - "轻量 planner"
---

# Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models

**会议**: ICML 2026  
**arXiv**: [2604.10567](https://arxiv.org/abs/2604.10567)  
**代码**: 无  
**领域**: LLM效率 / 扩散语言模型 / 非自回归解码  
**关键词**: dLLM, 非自回归解码, proximity bias, EOS overflow, 轻量 planner

## 一句话总结
本文系统刻画了 masked 扩散语言模型 (dLLM) 在**完全非自回归 (NAR) 解码**下的失败机制——proximity bias 导致 confidence-based 采样退化为反向自回归并被 EOS 过早占满，再用一个 5M 参数的轻量 planner + EOS 温度退火**只在第一步**干预 unmasking 位置，就在 GSM8K 等推理任务上将 LLaDA 8B 的 NAR 解码平均提升 2.8–4.3 个点而几乎无额外开销。

## 研究背景与动机

**领域现状**：dLLM (LLaDA、Dream、MDLM 等) 以 mask-and-predict 形式建模文本，理论上同时具备 **并行** (一次解码多个 token) 和 **双向** (利用前后上下文) 两大优势，被视为自回归 LLM 的潜在替代。

**现有痛点**：实际部署中，**完全 NAR 解码**几乎无法稳定产出连贯文本，SOTA 方法 (LLaDA、Block Diffusion 等) 只能退化成 **semi-autoregressive (semi-AR)**——按 block 顺序生成，把双向并行的优势又拱手让出，且对推理 / 规划这类全局结构强的任务尤其不友好。

**核心矛盾**：到底是 dLLM 模型本身的能力不足，还是 NAR 解码采样策略本身有结构性缺陷？此前研究多停留在"semi-AR 更稳"的经验结论，缺少对 NAR 时间动力学的拆解，导致改进只能靠"加更多 AR 先验"绕过去。

**本文目标**：(1) 找出 confidence-based NAR 解码失败的根因；(2) 在不微调 backbone、不引入 block 结构的前提下设计 *最小干预*，让完全 NAR 在推理任务上重新可用。

**切入角度**：作者沿 *时间轴* 跟踪每一步 unmasking 的位置选择，发现两个互相放大的偏置——**proximity bias** (新解码 token 总靠近上一步) 与 **EOS dominance** (高不确定下 EOS logit 总是最大)——并由此推出一个关键不对称：**第一步的位置决策对整条轨迹的最终质量影响极不对称地大**。

**核心 idea**：与其在所有步上注入 token 温度，不如把全部干预集中到 *最初一步的位置选择* 上——用一个轻量 planner 选首批 unmask 位置 + 把 EOS 的 logit 在早期退火压低——就足以扭转整条轨迹。

## 方法详解

### 整体框架
方法建立在标准 MDLM 反向解码之上：每步 $d$ 既要 (a) 为所有 mask 位置预测 token，又要 (b) 选出子集 $\mathcal{U}_d$ 把它们 unmask。本文 **不动** backbone $\theta$，**不动** 后续步的 greedy 策略，仅在第 1 步把位置选择 $\mathcal{U}_1$ 从 "Top-1 confidence" 换成 planner 打分，并在所有步对 EOS logit 施加随时间衰减的 inverse-temperature 缩放。整条 pipeline 用一个 progressive schedule：早期 $|\mathcal{U}_d|<L/T$，越往后每步放出越多 token，给 planner 留出区分度。

### 关键设计

1. **Proximity bias 诊断与时间不对称性**:

    - 功能：把"NAR 为什么不行"落到可量化的两个观察上——为后续干预提供精确的发力点。
    - 核心思路：固定 $L=256$ 扫描 $T\in\{32,...,256\}$，发现 NAR 性能随 $T$ 单调 *下降* (与 dLLM "更多步更好"的直觉相反)。逐步可视化 unmask 位置和 EOS 比例，证实：(i) 第 1 步总优先 unmask 序列末尾的 EOS；(ii) 后续步新 unmask 位置紧贴上一步——形成"从尾向头反向 AR"；(iii) GSM8K 上平均 256 个 slot 中 144.6 个被 EOS 占走。再做 pass@k 对照：把随机性只注入第 1 步的位置选择 (其它步全 greedy) 反而比"全程 token 温度采样"得分高 7+ 个点；把随机性延后到中间步则严重掉点。最后用 256 条轨迹的 "anchor + 后段重采样" 实验定量证明：correct/incorrect 早期轨迹的最终准确率差 $\sim 16{-}33$ 个点且置信区间不重叠——证据链条直接指向"第一步定全局"。
    - 设计动机：把模糊的"NAR 不稳定"具象为 *proximity bias × EOS dominance × 时间不对称* 三联机制，为"只在第一步干预"提供原理依据。

2. **轻量 Planner $\pi_\phi(\mathcal{U}_1\mid h_\mathcal{S})$**:

    - 功能：在第 1 步从 $P=32$ 个随机候选位置集合 $\{\mathcal{S}^i\}$ 中挑出能最大化最终任务奖励 $R(\mathbf{z}_0)$ 的那一个 $\mathcal{U}_1^\star$。
    - 核心思路：planner 是 2 层 Transformer encoder + position-wise scoring head，**只吃 backbone 最后一层在候选位置 $\mathcal{S}$ 上的 hidden** $h_\mathcal{S}$ (不看整段上下文)，每个 token 出一个标量分，平均得到候选集的总分。训练数据离线生成：对每条样本随机采 $S=32$ 个候选 $\mathcal{U}_1$，剩余步全 greedy 跑完得到 0/1 的 trajectory-level 正确性标签，用 BCE 训练 planner——目标是 $\max_\phi \mathbb{E}_{\mathcal{U}_1\sim\pi_\phi}[R(\mathbf{z}_0)]$。推理时随机采 $P=32$ 个候选集打分，取分高者作为 $\mathcal{U}_1$，*后续步全部走 confidence-based greedy*。
    - 设计动机：把 NAR 解码看成"开局决策问题"——既然第一步几乎决定全局，那么花一点点参数 (5M，相对 8B backbone 可忽略) 学一个 *开局教练* 性价比极高；只看候选位置的 hidden 而不看全序列是为了让 planner 真正学到"位置先验"而不是简单复现 backbone 的 confidence。

3. **EOS Temperature Annealing**:

    - 功能：在所有步对 EOS token 的 logit 单独施加随时间衰减的 inverse-temperature $\lambda_d$，初值 $\lambda_T=3$ 线性退火到 $1$，**只影响位置选择的排序，不改实际 token 预测**。
    - 核心思路：把 EOS 的 logit 在 softmax 前除以 $\lambda_d$，使其在排序 unmask 优先级时被显著压低；一旦位置选定，token 仍按原始 logit 走 greedy argmax，因此自然停止行为在后期被完整保留。配合 progressive schedule 一起用，可把 GSM8K 的有效 (非 EOS) token 数从 157.2 提升到 188.6——把推理任务真正需要的"生成窗口"还给模型。
    - 设计动机：proximity bias 一旦从 EOS 开始就会向前传染整条序列，所以最经济的打断方式不是改训练或换 padding 方案 (如 Rainbow Padding 需 finetune)，而是只在 *决定 unmask 谁* 这一步把 EOS 的影响力暂时削弱，等不确定性消退后自动恢复。

### 损失函数 / 训练策略
- Planner 损失：trajectory-level BCE，标签来自一次性离线 rollout 的 0/1 正确性 (Sudoku 用 cell accuracy)。
- 仅训 planner $\phi$，backbone $\theta$ 全程冻结；planner ≈ 5M 参数。
- Progressive 时间表：早期 $B<L/T$，让候选位置之间的差异在第 1 步更可分。
- 推理超参：$T=32, L=256, P=32, \lambda_T=3$ 线性退火。

## 实验关键数据

### 主实验 (LLaDA 8B Instruct, $T=32, L=256$, 完全 NAR)

| 数据集 | Top-1 (greedy) | + Planner | + EOS Anneal | + Both | Prob Margin + Both |
|--------|----------------|-----------|--------------|--------|--------------------|
| GSM8K | 46.6 | 55.0 | 50.9 | **56.8** | **58.6** |
| MATH | 19.2 | 22.4 | 22.4 | 22.8 | **23.0** |
| Countdown | 42.2 | 44.1 | **46.1** | 43.8 | 45.3 |
| Sudoku | 71.2 | 65.2 | 63.6 | 67.0 | 69.5 |
| **Avg** | 44.8 | 46.7 | 45.7 | **47.6** | **49.1** |

> 同设置下 semi-AR 平均仅 27.0；纯随机首位 (Init. Position) 平均 37.1，全程 token 温度采样 44.4——证明 NAR 完全可被"救活"，且本文方法把它推到了超过 semi-AR 的水平。

### 跨预算泛化 ($T=64,128$，planner 只在 $T=32$ 训过)

| 设置 ($T$) | Top-1 | Ancestral | Temperature | Init.Pos | + Planner | + EOS | Both |
|------------|-------|-----------|-------------|----------|-----------|-------|------|
| Avg @ 64 | 39.7 | 26.6 | 41.9 | 41.6 | 48.0 | 47.4 | **50.2** |
| Avg @ 128 | 37.7 | 28.2 | 39.2 | 43.0 | 47.8 | 48.1 | **52.8** |

更高算力下 baseline 反而下降 (EOS 主导加剧)，本方法持续上升，说明 planner 学到的不是 $T=32$ 的过拟合启发式而是更普适的"开局优先级"。

### 关键发现
- **proximity bias + premature EOS 是 NAR 失败的双引擎**：单看任一现象都不足以解释，组合起来才能解释 "$T$ 越大越差"和"semi-AR 必胜"两件事。
- **时间不对称极强**：把随机性从第 1 步推迟到第 5 步，pass@k 直接崩盘；correct/incorrect 早期轨迹的最终准确率有 $\sim 16{-}33$ 点稳定 gap。
- **结构化任务对随机性敏感**：Sudoku 这种 1-shot 模板严格的任务，任何"无脑随机"都比 greedy 差 (71.2→48.3)，但 learned planner 能学到结构先验，把损失从 23 点压到 6 点。
- **方法与已有启发式正交**：套到 Probability Margin 上 GSM8K 还能从 47.2→58.6，说明 planner 与 token-level 启发式叠加而非冲突。
- **几乎零开销**：planner 5M 参数 + 只在第 1 步跑；EOS 退火只是单标量缩放；候选池 $P>32$ 后收益饱和。

## 亮点与洞察
- **"开局比中盘重要"的反直觉**：在 dLLM 这种理论上"全局并行"的模型里，最关键的决策反而集中在第一步——这把人们对扩散采样 "uniform stochasticity" 的默认假设彻底翻了过来，思路非常巧妙。
- **诊断驱动的最小干预**：从 *proximity bias × EOS × 时间不对称* 三联诊断直接推出"只动第一步 + 只压 EOS logit"两条干预，方法和分析高度自洽，给"先讲清楚为什么坏再讲怎么修"做了一个很好的范例。
- **planner 设计的可迁移性**：用 backbone hidden 在候选位置上的局部表征 + trajectory-level 0/1 奖励训一个 5M 模型，这个 *局部 hidden → 全局 trajectory reward* 的范式可直接迁移到任何 mask-and-predict 模型 (代码生成的 mask LM、图像 token diffusion 等) 的"开局位置选择"问题。

## 局限与展望
- **planner 训练依赖任务级 0/1 奖励**：reward 稀疏的开放生成 (creative writing、对话) 上很难拿到清晰标签，需要换成 process reward 或 LLM judge。
- **仅在 LLaDA / Dream 7-8B 验证**：更大规模 dLLM 和 code/multilingual 等下游任务上是否仍然 "第一步定全局" 留待验证；planner 是否需要按 backbone 重训也未给出建议。
- **没有比较"训第一步的全长 dLLM"基线**：若直接在 backbone 上做 first-step RL fine-tune，效果与 5M planner 谁更优？文中未对照。
- **结构化任务上仍稍逊 greedy**：Sudoku 上 Both (67.0) 仍低于 Top-1 greedy (71.2)，说明 planner 在"模板严格"的任务上学到了开局多样性，但反而牺牲了任务先验——任务自适应的开局策略 (例如 confidence 阈值切换) 是值得做的方向。

## 相关工作与启发
- **vs Rainbow Padding (Kim et al., 2026)**：两者都指向 EOS overflow，但 Rainbow 需要替换 padding token 并微调 backbone；本文不动 backbone、只压 EOS logit，部署成本极低；解释角度也不同——本文把 EOS dominance 归因到更底层的 proximity bias。
- **vs Block Diffusion / LLaDA semi-AR**：semi-AR 用强结构先验绕开 NAR 不稳定，但牺牲并行度且推理任务上限被 block 大小卡住；本文在完全 NAR 下做到比 semi-AR 更高的平均准确率 (47.6 vs 27.0)，证明"NAR 本身没坏，只是采样策略坏"。
- **vs Convolutional NAR (Seo et al., 2025)**：Seo 等从 *空间* 一致性出发用卷积滤波器；本文把分析推到 *时间* 轴，并指出最该治理的不是 spatial smoothness 而是 first-step asymmetry，提供了正交视角。
- **vs Peng et al. 2025 / Liu et al. 2025a (per-step planner)**：他们的 planner 在每一步都激活，开销大；本文只在第 1 步用，性价比显著更高，且实验上"只在第一步干预"已能拿到大部分增益。

## 评分
- 新颖性: ⭐⭐⭐⭐ "Proximity bias × 时间不对称"是新颖且可证伪的诊断，"只动开局"的极简干预与现有 per-step planner 形成鲜明对比。
- 实验充分度: ⭐⭐⭐⭐ 4 任务 × 2 backbone × 3 算力档 × 多基线 + pass@k / anchor 实验链条完整，主要短板是开放生成与更大规模 dLLM 未覆盖。
- 写作质量: ⭐⭐⭐⭐ "现象—诊断—干预"逻辑递进清晰，图表和叙述紧密配合，公式与符号约定规范。
- 价值: ⭐⭐⭐⭐ 几乎零成本就把"完全 NAR 解码"从不可用推到超越 semi-AR，是 dLLM 推理部署的关键 enabler，且范式可迁移到其他 mask LM。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](../../CVPR2026/image_restoration/evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)

</div>

<!-- RELATED:END -->
