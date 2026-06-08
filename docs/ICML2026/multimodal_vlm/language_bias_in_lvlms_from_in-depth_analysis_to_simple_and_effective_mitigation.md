---
title: >-
  [论文解读] LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation
description: >-
  [ICML 2026][多模态VLM][语言偏置] 本文系统量化 LVLM 训练中的语言偏置——发现 VIT 和 DPO 两个阶段都让 text-only likelihood $\pi(y|x)$ 涨得几乎不输 multimodal likelihood $\pi(y|x,v)$…
tags:
  - "ICML 2026"
  - "多模态VLM"
  - "语言偏置"
  - "VIT"
  - "DPO"
  - "模态错位"
  - "训练即插即用"
---

# LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation

**会议**: ICML 2026  
**arXiv**: [2605.25036](https://arxiv.org/abs/2605.25036)  
**代码**: https://github.com/lab-klc/LVLM-Language-Bias  
**领域**: 多模态VLM / 训练目标 / 幻觉缓解  
**关键词**: 语言偏置, VIT, DPO, 模态错位, 训练即插即用

## 一句话总结
本文系统量化 LVLM 训练中的语言偏置——发现 VIT 和 DPO 两个阶段都让 text-only likelihood $\pi(y|x)$ 涨得几乎不输 multimodal likelihood $\pi(y|x,v)$，证明 LVLM 在系统性低估视觉输入；提出 Language Bias Regularization（VIT 阶段惩罚 $|\mathcal{B}|$）和 Language Bias Penalty（DPO 阶段惩罚偏置正向增长），不加任何数据/辅助模型就显著提升 10+ benchmark 性能并降幻觉。

## 研究背景与动机

**领域现状**：LVLM 把视觉接到 LLM，但幻觉严重——生成流畅但与视觉矛盾的文本。研究普遍归因于 language bias（模型过度依赖语言，忽视视觉）；现有缓解分 training-free（output post-processing）和 training-based（fine-tuning / DPO）两类。

**现有痛点**：（1）language bias 的理解停在经验层——"看图少看文字多"，没有形式化定义；（2）没有量化指标追踪偏置的训练动态；（3）现有方法 stop-gap，不动根因；（4）VIT 和 DPO 都是当代 alignment 标配，但它们是否本身就在制造 bias 没人系统检查。

**核心矛盾**：训练目标 $\max \pi(y|x, v)$ 看似要求 v 参与，但 $v$ 和 $x$ 共同输入下模型可用 text-only path 满足目标——这意味着目标本身没有强制要求 visual grounding。结果：模型学到的 $\pi(y|x)$（text-only）涨得跟 $\pi(y|x,v)$（multimodal）一样快，相当于视觉模态被白搭。

**本文目标**：（1）形式化定义 language bias 并量化；（2）在 VIT 和 DPO 两个训练阶段诊断；（3）提供可直接换入现有 pipeline 的 mitigation 方案。

**切入角度**：分解训练奖励——把 reward $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$（multimodal gain）和 bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$（text-only gain）分开追踪，若 $\mathcal{B} \approx \mathcal{R}$ 说明改善全靠语言路径。

**核心 idea**：定义 $\mathcal{B}$ 后直接在损失里惩罚——VIT 加 $|\mathcal{B}|$（LBR），DPO 在 chosen 上惩罚 $\mathcal{B}_w$ 正向增长（LBP）。

## 方法详解

### 整体框架

本文先给"语言偏置"一个可测的定义，再据此在 VIT 和 DPO 两个训练阶段各加一个正则项。核心是把训练带来的 likelihood 提升拆成两份：multimodal gain $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$（喂图时的收益）和 language bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$（只喂文本时的收益），以 pre-VIT / pre-DPO 模型作参考 $\pi_{\text{ref}}$。训练每一步都同时追踪 $\mathcal{R}$ 与 $\mathcal{B}$——一旦 $\mathcal{B}$ 的轨迹跟 $\mathcal{R}$ 几乎重合，就说明模型的进步其实全靠纯文本路径、视觉被白搭。两个 baseline 损失分别是 VIT 的 $\mathcal{L}_{\text{VIT}} = -\sum_t \log \pi_\theta(y_t | x, v, y_{<t})$ 和带 margin 的 DPO $\mathcal{L}_{\text{DPO}_M} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{Margin}}$，LBR 和 LBP 就是分别在这两个上面挂一个偏置惩罚项。

### 关键设计

**1. Language Bias 的形式化 + 训练动态追踪：把"看文字多于看图"变成可测的标量。**

以往 language bias 全靠"凭感觉"描述——模型偏向语言、忽视视觉，但没有定义就没法工程化管控。本文把训练奖励拆成 multimodal gain $\mathcal{R}$ 和 text-only gain $\mathcal{B}$ 两条轨迹，用冻结的 $\pi_{\text{ref}}$ 当基线，量出训练过程里 LLM 究竟从纯文本路径偷学了多少。

这个分解直接揭出问题：Figure 3 显示 VIT 阶段 $\mathcal{R}_{\text{VIT}}$ 和 $\mathcal{B}_{\text{VIT}}$ 的曲线几乎重合，到 DPO 阶段 $\mathcal{B}_{\text{DPO}_w}$ 甚至反超 $\mathcal{R}_{\text{DPO}_w}$——也就是说当前的对齐目标 $\max \pi_\theta(y|x,v)$ 根本没逼模型用视觉，纯文本路径就能满足它。有了这个操作性定义，后面才能把 $\mathcal{B}$ 直接写进损失去压。

**2. LBR：VIT 阶段惩罚 $|\mathcal{B}|$，把语言路径压回参考水平。**

预训练之后 $\mathcal{B}$ 已经很小，真正"制造偏置"的是 VIT 阶段。LBR 的做法极简——直接把偏置的绝对值当正则项与 VIT loss 加权求和：

$$\mathcal{L}_{\text{LBR}} = |\mathcal{B}| = \left|\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right|$$

用绝对值（而非只惩罚正向增长）是为了双向锁定：既不让纯文本能力膨胀、也不让它退化，把语言模态推理钉在参考水平。这样模型想拿到额外的 reward 就只剩"真正用视觉"这一条路，从损失层面堵死了模态错位。

**3. LBP：DPO 阶段惩罚 chosen 偏置正向增长，堵住 reward hacking。**

DPO 有个特有的漏洞——它会让 preferred（chosen）答案在纯文本路径上也涨分，相当于"靠语言混过偏好"这种 reward hacking。LBP 在 DPO loss 上加一项 $\max(0, \mathcal{B}_w)$，只惩罚 chosen 的 text-only gain 正向增长；为负不罚（chosen 在纯文本下变差是可以接受的），rejected 的偏置也完全不动以避免 over-regularization。这样就专门切断了"preferred 答案不看图也能赢"的捷径，逼 DPO 的偏好优势真正来自视觉对齐。

## 实验关键数据

### LBR 在 10+ 通用基准上的提升（LLaVA-1.5-7B）

| 基准 | Baseline VIT | **+ LBR** | Δ |
|------|----------|---------|---|
| MMMU | 35.7 | **37.4** | +1.7 |
| MathVista | 26.4 | **28.9** | +2.5 |
| MM-Bench | 64.3 | **66.1** | +1.8 |
| ScienceQA-IMG | 70.5 | **72.3** | +1.8 |
| GQA | 62.1 | **63.8** | +1.7 |
| TextVQA | 58.2 | **60.0** | +1.8 |
| ChartQA | 18.9 | **20.5** | +1.6 |
| RealWorldQA | 56.7 | **58.4** | +1.7 |
| AI2D | 55.5 | **57.2** | +1.7 |
| SEED-Bench | 66.1 | **67.7** | +1.6 |

10/10 一致 +1.5~2.5 点，没有 cherry pick。

### LBP 在幻觉/可信基准上的提升

| 基准 | DPO baseline | **+ LBP** | Δ |
|------|----------|---------|---|
| POPE 准确率 | 86.4 | **88.9** | +2.5 |
| MMHal-Bench score | 2.71 | **3.18** | +0.47 |
| AMBER 综合 | 65.3 | **69.7** | +4.4 |
| ObjectHal-Bench | 11.2 (越低越好) | **7.8** | −3.4 |
| TrustEval | 71.5 | **75.3** | +3.8 |

幻觉显著下降、可信度全面提升；ObjectHal 降 30%。

### 跨模型规模 / 架构

| 模型 | 任务 | Baseline | + LBR/LBP | Δ |
|------|------|------|---------|---|
| LLaVA-1.5-13B | VIT | 67.2 | 69.3 | +2.1 |
| LLaVA-Next-7B | DPO | 73.4 | 75.8 | +2.4 |
| Qwen2-VL-7B | VIT | 79.1 | 80.7 | +1.6 |
| InternVL-2-8B | DPO | 81.3 | 83.6 | +2.3 |

跨模型规模、不同 LVLM 家族都一致受益。

### 关键发现
- **language bias 是 VIT/DPO 两阶段共有现象**：Figure 3 显示 $\mathcal{B}, \mathcal{R}$ 轨迹一致——证明这是训练范式系统性问题，不是数据问题
- **简单到不像方法**：LBR 就是损失 + $|\mathcal{B}|$，LBP 就是 DPO + $\max(0, \mathcal{B}_w)$，但跨 10+ 基准、多模型一致涨
- **零额外数据/模型**：不像之前缓解方案要外加 reference VLM 或人工标注，LBR/LBP 完全在原 pipeline 内
- **可视化证实**：Figure 2 显示 LBR 让模型对图像 token 的注意力分布显著抬升

## 亮点与洞察
- **形式化 + 量化 + 干预的完整闭环**：把 "language bias" 这个模糊概念变成可定义、可追踪、可干预的工程对象——这套"定义 → 测量 → 损失项"的方法学是 alignment 领域的范本
- **简单到反直觉的有效性**：$|\mathcal{B}|$ 这种"幼稚"的正则化竟然 dominate 复杂的缓解方法——说明问题诊断对了之后方案可以非常简洁
- **抓住训练范式而非数据**：以往幻觉缓解都加数据 / 改 prompt / 加 reward model；本文证明问题在训练目标本身，从损失函数层面解决一劳永逸
- **VIT vs DPO 双阶段诊断**：两阶段都有 bias 但形式不同（VIT 是模态错位、DPO 是 reward hacking），分别用 LBR / LBP 治——细致区分而非一刀切

## 局限性 / 可改进方向
- $\mathcal{B}$ 计算需要每步 forward 一次 text-only 输入，训练成本约 +50%
- $\pi_{\text{ref}}$ 选择影响 $\mathcal{B}$ 测量——pre-VIT vs 中间 checkpoint 区别未充分讨论
- 单纯惩罚 $|\mathcal{B}|$ 可能伤害某些"语言主导"任务（如纯文本推理），需更细 task-aware 控制
- DPO 阶段只惩罚 $\mathcal{B}_w$，未探索 $\mathcal{B}_l$ 的复杂作用（chosen 文本 gain vs rejected 文本 gain 的关系可能更微妙）
- 没分析 LBR 对 visual token 数量 / patch 大小的敏感性

## 相关工作与启发
- **vs training-free decoding（VCD、OPERA 等）**：那些 post-hoc 改 decoding，不改根因；LBR/LBP 训练时根治
- **vs 数据驱动幻觉缓解（GRIT、RLHF-V）**：那些靠精细标注数据；LBR/LBP 零额外数据
- **vs 现有 modality alignment（如 SF-Tuning）**：那些 ad-hoc 架构改动；LBR/LBP 只改损失
- **启发**：所有"模态融合训练目标"都可类似分解监控（如 audio-LLM、video-LLM）；"训练动态量化指标 → 直接惩罚"模板可推广到其他 alignment 问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 形式化 + 简单 mitigation，但 $|\mathcal{B}|$ 正则本身不复杂；核心创新是"诊断 + 简单方案"框架
- 实验充分度: ⭐⭐⭐⭐⭐ 10+ 通用 + 5+ 幻觉基准 + 多模型多规模，覆盖完整
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑清晰：定义 → 追踪 → 诊断 → 损失 → 验证；Figure 3 决定性证据
- 价值: ⭐⭐⭐⭐⭐ 对所有 LVLM 训练 pipeline 直接受用，零额外成本；幻觉是 LVLM 部署最大障碍之一

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](../../ACL2025/multimodal_vlm/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2025\] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](../../ACL2025/multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)
- [\[ICML 2026\] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)
- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)

</div>

<!-- RELATED:END -->
