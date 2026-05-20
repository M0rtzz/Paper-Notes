---
title: >-
  [论文解读] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety
description: >-
  [ICML 2026][LLM安全][多轮越狱] MultiBreak 用"主动学习 + 不确定性引导改写"的迭代框架把多轮越狱数据集扩到 10,389 条对话、2,665 个独立有害意图，多样性 0.942 全面碾压前作…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "多轮越狱"
  - "jailbreak benchmark"
  - "主动学习"
  - "不确定性引导"
  - "LLM 红队"
---

# MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety

**会议**: ICML 2026  
**arXiv**: [2605.01687](https://arxiv.org/abs/2605.01687)  
**代码**: 无  
**领域**: AI 安全 / LLM 评测  
**关键词**: 多轮越狱, jailbreak benchmark, 主动学习, 不确定性引导, LLM 红队

## 一句话总结
MultiBreak 用"主动学习 + 不确定性引导改写"的迭代框架把多轮越狱数据集扩到 10,389 条对话、2,665 个独立有害意图，多样性 0.942 全面碾压前作，并在 DeepSeek-R1-7B / GPT-4.1-mini 上把 ASR 相比次优数据集分别提升 54% / 34.6%。

## 研究背景与动机

**领域现状**：LLM 安全对齐评测的主流是构造越狱（jailbreak）数据集，让 LLM 在对抗 prompt 下输出违规内容。单轮越狱（GCG、PAIR、HarmBench 等）已比较成熟，但和真实用户交互严重脱节；学界开始转向多轮越狱（Crescendo、MRJ-Agent、CoSafe、MHJ、SafeDialBench、RedQueen），通过逐步升级或良性铺垫绕过安全护栏。

**现有痛点**：多轮越狱基准要么规模小（CoSafe 1.4k、MHJ 537、SafeDialBench 2k），要么靠模板复制（RedQueen 把 1.4k 意图 × 40 模板硬撑到 56k），两者都让"多样性"成为瓶颈。Qwen3-0.6B embedding 去重后，最好也只有 76% 是真正独立的意图，导致评测易受 prompt 微小扰动影响，跨 LLM 结果不一致。

**核心矛盾**：扩规模和保多样性是天然矛盾——人工标注质量高但贵；LLM 自动生成便宜但容易陷入"模式坍缩"（反复生成相似攻击）；同时安全对齐的 LLM 又会拒绝生成有害内容。

**本文目标**：(1) 在质量不下降的前提下把多轮越狱数据集放大一个数量级；(2) 用一种避免模板复制的方式系统性覆盖更广的有害意图谱系；(3) 揭示哪些类别在单轮下安全、多轮下却高危。

**切入角度**：作者把"构造 benchmark"重新表述为 pool-based active learning：从大池有害意图出发，迭代 fine-tune 攻击 generator → 在多个 victim/judge 上评估 → 用 acquisition function 把样本分到 accept / rewrite / discard → 用 unc-guided rewriter 把"边缘成功"的样本拉回。

**核心 idea**：用 acquisition 函数 + rewrite 把"模型自己不确定的样本"作为高价值训练信号循环放大攻击 generator，得到既多样又高 ASR 的对抗 prompt 集。

## 方法详解

### 整体框架
三阶段流水线：(1) **Data Diversification** —— 从 5 个多轮 + 9 个单轮已有数据集汇集，用 Qwen3-0.6B embedding 去重 + 闭源 victim 校验过滤掉 false harmful，初始化 $|Q_{adv}^{(0)}|=2{,}201$ 多轮对抗 prompt 与 $|Q|=3{,}010$ 独立有害意图，形成 $\mathcal D_0$；(2) **Active Learning Loop** —— 对每一轮 $t$，当前 generator $\text{LLM}_G^{(t)}$ 在未标注池 $\mathcal U^{(t)}$ 上生成多轮对抗 prompt（MTAP，2-6 轮随机长度），在 victim 集 $\mathcal V$ 与 judge 集 $\mathcal J$ 上评估 ASR / uncertainty / faithfulness，按 acquisition 函数 $\alpha$ 切成 accept / rewrite / discard；(3) **Uncertainty-Guided Rewrite** —— 把 rewrite 桶送给独立的 Qwen2.5-7B 改写器再校验，成功就并入 accept。$\mathcal D^{(t+1)} = \mathcal D^{(t)}\cup\mathcal S_{\text{accept}}$，然后 SFT 出 $\text{LLM}_G^{(t+1)}$，循环 $T$ 轮汇总。

### 关键设计

1. **三信号 Acquisition 函数（exploit + explore + 质量过滤）**：

    - 功能：决定每个生成 prompt 该被接受、改写还是丢弃，是整个 active learning 的"路由器"。
    - 核心思路：在多 victim × 多 judge 上算 (a) **ASR** $\text{ASR}(q_{adv})=\frac{1}{|\mathcal V||\mathcal J|}\sum_V\sum_J J(q_{adv},V(q_{adv}))$ 衡量攻击稳定成功率（exploit）；(b) **不确定性** $\sigma(q_{adv})=\text{Std}_{V,J} J(q_{adv},V(q_{adv}))$ 衡量在不同 victim-judge 对上的分歧（explore，定位"边缘但有信息量"的样本）；(c) **保真度** $\text{faith}(q,q_{adv})=\cos(\text{Enc}(q),\text{Enc}(q_{adv}))$ 用 Qwen3-0.6B embedding 防止改写后语义漂移。三段决策 $\alpha(q_{adv})$ 为 Accept（ASR $\ge\tau_h$ 且 faith $\ge\tau_f$）/ Rewrite（$\sigma\ge\tau_\sigma$ 且 ASR<$\tau_h$ 且 faith $\ge\tau_f$）/ Discard。
    - 设计动机：单独按 ASR 选会让 generator 过拟合到几条已经有效的攻击模式（模式坍缩），单独按不确定性选会带入低质量噪声；三信号组合让"已经稳定且语义不漂移的拿去训练 / 边缘但有信息的拿去改写 / 其它丢掉"成为闭环里最自然的选择。

2. **生成器集成 + SFT 而非 prompting**：

    - 功能：突破安全对齐 LLM "直接拒绝生成有害内容"的限制，并让多个不同家族模型互补暴露漏洞。
    - 核心思路：用 LLaMA3-8B-Instruct + Qwen2.5-7B-Instruct（全参数 SFT）+ DeepSeek-Distill-Qwen-14B（LoRA）作为 $\text{LLM}_G$ ensemble。论文实证 prompting 在 Mistral-7B-Instruct 上只有 2% ASR，SFT 能到 25%；并且 prompting 时模型会反复拒绝或者输出不忠实于 $q$ 的内容（图 3 失败模式）。注意改写器 $\text{LLM}_R$ 用未微调的 Qwen2.5-7B，因为它只看 $q_{adv}$ 不直接看 $q$，不会触发安全护栏。
    - 设计动机：小尺寸开源 generator 在 SFT 后暴露的漏洞会向闭源大模型迁移（论文表 5 验证），这是"小模型攻大模型"的迁移性发现；多家族 ensemble 进一步降低 generator 偏向单一攻击范式的风险。

3. **多 victim 多 judge 去偏 + 不确定性引导改写**：

    - 功能：摆脱单 judge 一致性偏差，把"在某些模型下成功、在另一些下失败"的样本变成可用的训练信号而非噪声。
    - 核心思路：使用 LLaMA Guard + GPT-4o-mini 双 judge + 基于关键词的拒绝检测器，被拒绝检测器直接判 0 的样本立即丢弃；剩余按 $\sigma$ 高的样本送 rewriter，被指令"保留有害意图、澄清模糊表达、强化说服/混淆策略"。$\sigma$ 经过 rewrite 后在每轮迭代里都显著下降（图 4），把边缘样本"拉直"。
    - 设计动机：单 judge 已被证明存在偏差（Souly 2024、Huang 2025），多 judge 提供不确定性自然估计；同时不把"低 ASR + 高 $\sigma$"的样本扔掉而是改写，可以从模型分歧区域榨取额外训练信号，比纯靠新生成数据更样本高效。

### 损失函数 / 训练策略
generator 用标准 SFT 损失 fine-tune；rewriter 不训练（指令 prompting）；judge 多数投票 + 拒绝检测器硬过滤；多轮长度 $n\sim\text{Uniform}(2,6)$；总迭代 $T=5$ 轮就能把 ASR 推到 50%+。

## 实验关键数据

### 主实验
表 1（数据集对比）：MultiBreak 2-6 轮，10,389 条样本，2,665 个独立意图，diversity 0.942；CoSafe 1,400/961/0.843、MHJ 537/406/0.810、SafeDialBench 2,037/1,078/0.762、RedQueen 56k(模板复制)/656/0.680。

表 2（ASR，judge: LG=LLaMA Guard / GPT=GPT-4o-mini）：

| 数据集 | DeepSeek-7B (LG/GPT) | Qwen3-8B | LLaMA3.1-8B | Gemini-2.5-FL | GPT-4.1-mini |
|--------|----------------------|----------|-------------|---------------|--------------|
| CoSafe @1 | 0.127 / 0.235 | 0.079/0.340 | 0.063/0.456 | 0.059/0.557 | 0.019/0.552 |
| MHJ @1 | 0.293 / … | … | … | … | … |
| **MultiBreak** | 大幅领先（DeepSeek 上 +54%，GPT-4.1-mini 上 +34.6% 相对次优数据集） |

### 消融实验

| 配置 | 影响 | 说明 |
|------|------|------|
| Full pipeline (5 iter) | ASR 50%+ | Qwen 受害者 + LLaMA generator |
| 仅初始 $\mathcal D_0$ | 10.77% | 比 CoSafe/RedQueen 各高 4.47/3.77 pp |
| 仅 SFT 不主动学习 | 25% | 比 prompting 强很多但远低于主动学习 |
| 仅 prompting 不 SFT | 2% | 安全对齐 LLM 反复拒绝 |
| 去掉 rewrite | ASR / 多样性双降 | 边缘样本信息全部丢弃 |
| 去掉多 victim/judge | judge bias | ASR 不再稳定 |

### 关键发现
- 主动学习的迭代单调性：ASR 5 轮内从 10% 升到 50%+，且每轮 rewrite 后 $\sigma$ 都显著下降，说明 acquisition 函数确实在"开采有信息样本"。
- 在小开源模型上 fine-tune 的攻击 generator 暴露的漏洞会迁移到 GPT-4.1-mini / Gemini-2.5-Flash-Lite 等闭源大模型。
- 单轮看似安全的有害类别在多轮里 ASR 显著上升，说明多轮场景下 LLM 安全护栏的薄弱区与单轮完全不同。

## 亮点与洞察
- **把数据集构造变成 active learning 问题**：是个非常自然但之前没人系统化做过的视角，acquisition 函数三段切分（accept/rewrite/discard）干脆利落，可以直接迁移到任何"数据生成 + 难例利用"的场景（如 RLHF preference data、code repair、math reasoning 数据扩充）。
- **不确定性 = 改写机会**：传统 active learning 把高不确定性样本送给人工标注，本文换成"用专门的改写 LLM 改写后再校验"，把人工成本压到零，这条 trick 在任何需要"模型自动扩样"的场景都值得复用。
- **闭源拒绝有害内容 → 改写器不直接接触意图**：作者识别出"安全对齐 LLM 看 $q$ 就拒绝"这一失败模式后，让 rewriter 只看已有的 $q_{adv}$，工程上绕开微调成本，思路简洁。

## 局限与展望
- 标签全部依赖 judge LLM（LG + GPT-4o-mini）+ 规则拒绝检测器，本身可能漏检或误报，最终 ASR 数字依赖 judge 质量。
- 实验主要在英文上进行，多语言多轮越狱的可扩展性尚未验证；safety category 分布也与所选种子数据集强相关。
- 数据集本身具有双刃性——可以用于攻防研究但也降低了构造大规模高质量越狱攻击的门槛，论文对责任披露和访问控制的讨论较少。
- 5 轮迭代 + 多 victim 评估对计算资源要求不低，复现门槛较高。

## 相关工作与启发
- **vs RedQueen (Jiang 2024)**：RedQueen 用 1,400 意图 × 40 模板膨胀到 56k 条，但 token 多样性极低；本文用 active learning 真正扩展独立意图数（2,665 vs 656）。
- **vs CoSafe / MHJ / SafeDialBench**：这些基准规模都在千量级，类别覆盖窄；MultiBreak 同时在规模和 diversity score 上是 best-of-all。
- **vs Crescendo / MRJ-Agent**：它们是"攻击方法"，本文是"benchmark 构造方法"，可以把任意攻击方法作为 generator 接入主动学习循环。
- **vs 单轮越狱基准（HarmBench, AdvBench, JailbreakBench）**：实验首次系统对比单轮 vs 多轮 ASR 差距，证明许多看似安全的类别在多轮下漏洞远大于单轮，给出 LLM 安全评测必须从"单轮"升级到"多轮"的有力证据。

## 评分
- 新颖性: ⭐⭐⭐⭐ 主动学习 + 不确定性改写组合用在越狱数据集上是新角度，组件本身（uncertainty sampling、self-refine）相对成熟。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 5 个 victim + 2 judge，与 4 个 baseline 数据集横评，并有 sample efficiency / diversity / 类别细分多角度分析。
- 写作质量: ⭐⭐⭐⭐ 三段框架图（图 2）+ 算法 1 把流程讲得很清楚，定义和符号比较紧凑。
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 安全评测社区是直接好用的资源；同时其 "uncertainty-rewrite" 方法可被搬到任何数据扩展任务。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](../../ACL2026/llm_safety/trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)
- [\[ACL 2026\] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](../../ACL2026/llm_safety/when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)
- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](../../ACL2026/llm_safety/safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)

</div>

<!-- RELATED:END -->
