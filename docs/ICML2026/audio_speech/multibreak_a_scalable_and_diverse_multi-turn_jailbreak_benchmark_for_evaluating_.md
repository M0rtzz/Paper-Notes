---
title: >-
  [论文解读] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety
description: >-
  [ICML 2026][音频/语音][多轮越狱] MultiBreak 用"主动学习 + 不确定性引导改写"的迭代框架把多轮越狱数据集扩到 10,389 条对话、2,665 个独立有害意图，多样性 0.942 全面碾压前作…
tags:
  - "ICML 2026"
  - "音频/语音"
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
MultiBreak 要解决的是"既要把多轮越狱数据集扩大一个数量级、又不能让多样性塌掉"这个核心矛盾，作者的做法是把"构造 benchmark"重新表述成一个 pool-based active learning 问题：先从已有数据集汇集出一个去重过的初始种子集，再让一个攻击 generator 不断在未标注的有害意图池上生成多轮对抗 prompt，由多个 victim 和 judge 联合打分后用 acquisition 函数把样本分流（直接采纳 / 送去改写 / 丢弃），把采纳的样本回灌去微调下一轮 generator，如此循环若干轮直到 ASR 收敛。整个过程靠"模型自己拿不准的样本"作为高价值信号驱动放大，从而同时拿到高 ASR 和高多样性。

### 关键设计

**1. 三信号 Acquisition 函数：决定每个生成样本是采纳、改写还是丢弃**

这是整个 active learning 循环的"路由器"，要回答的痛点是——如果只按攻击成功率挑样本，generator 会迅速过拟合到几条已经有效的攻击模式造成模式坍缩；如果只按不确定性挑，又会引入大量低质量噪声。作者因此在多 victim × 多 judge 的网格上同时算三个信号：攻击成功率 $\text{ASR}(q_{adv})=\frac{1}{|\mathcal V||\mathcal J|}\sum_V\sum_J J(q_{adv},V(q_{adv}))$ 衡量攻击是否稳定成功（exploit）；不确定性 $\sigma(q_{adv})=\text{Std}_{V,J}\, J(q_{adv},V(q_{adv}))$ 衡量样本在不同 victim-judge 对上的打分分歧，用来定位"边缘但有信息量"的样本（explore）；保真度 $\text{faith}(q,q_{adv})=\cos(\text{Enc}(q),\text{Enc}(q_{adv}))$ 用 Qwen3-0.6B embedding 算原始意图 $q$ 与对抗 prompt $q_{adv}$ 的语义相似度，防止改写后跑题。三者组成决策 $\alpha(q_{adv})$：当 ASR $\ge\tau_h$ 且 faith $\ge\tau_f$ 时 **Accept**；当 $\sigma\ge\tau_\sigma$、ASR $<\tau_h$ 且 faith $\ge\tau_f$ 时送去 **Rewrite**；其余 **Discard**。这样"已经稳定且不漂移的拿去训练、边缘但有信息的拿去改写、剩下的丢掉"就成了闭环里最自然的分工。

**2. 生成器集成 + SFT 而非 prompting：突破安全对齐 LLM 的拒绝护栏**

直接 prompt 一个安全对齐 LLM 让它生成有害内容，它往往会反复拒绝、或输出不忠实于原意图 $q$ 的内容（论文图 3 的失败模式）——实测在 Mistral-7B-Instruct 上 prompting 只有 2% ASR，而做全参数 SFT 后能升到 25%，说明微调才是绕开护栏的关键。作者据此用 LLaMA3-8B-Instruct + Qwen2.5-7B-Instruct（全参数 SFT）+ DeepSeek-Distill-Qwen-14B（LoRA）组成 $\text{LLM}_G$ ensemble，多家族组合进一步降低 generator 偏向单一攻击范式的风险。一个值得注意的工程取舍是：负责改写的 $\text{LLM}_R$ 反而用未微调的 Qwen2.5-7B，因为它只读 $q_{adv}$、不直接读原始意图 $q$，所以不会触发安全护栏，省掉了再微调一个改写器的成本。这套"小开源模型 SFT 出来的攻击 generator"暴露的漏洞还会迁移到闭源大模型（论文表 5 验证），构成"小模型攻大模型"的迁移性发现。

**3. 多 victim 多 judge 去偏 + 不确定性引导改写：把模型分歧榨成训练信号**

单一 judge 已被证明存在一致性偏差（Souly 2024、Huang 2025），会让"在某些模型下成功、在另一些下失败"的样本被错判。作者用 LLaMA Guard + GPT-4o-mini 双 judge，再叠一个基于关键词的拒绝检测器——被拒绝检测器直接判 0 的样本立即丢弃，剩余样本则天然给出了上面 $\sigma$ 这个不确定性估计。关键一步是不把"低 ASR + 高 $\sigma$"的边缘样本扔掉，而是送进 rewriter，并指令它"保留有害意图、澄清模糊表达、强化说服/混淆策略"，把样本从模型分歧区域"拉直"——实测每轮 rewrite 后 $\sigma$ 都显著下降（图 4）。相比传统 active learning 把高不确定性样本交给人工标注，这里改成"用专门的改写 LLM 改完再校验"，既从分歧区域榨出了额外训练信号、比纯靠新生成数据更样本高效，又把人工成本压到了零。

### 损失函数 / 训练策略
generator 走标准 SFT 损失微调，rewriter 不训练（纯指令 prompting），judge 端用双 judge 多数投票叠拒绝检测器做硬过滤；每条对抗 prompt 的多轮长度按 $n\sim\text{Uniform}(2,6)$ 随机采样；初始种子集为 $|Q_{adv}^{(0)}|=2{,}201$ 条多轮对抗 prompt 与 $|Q|=3{,}010$ 个独立有害意图（由 Qwen3-0.6B embedding 去重 + 闭源 victim 校验过滤 false harmful 得到）。每轮把采纳样本并入数据集 $\mathcal D^{(t+1)}=\mathcal D^{(t)}\cup\mathcal S_{\text{accept}}$ 后重新 SFT 出 $\text{LLM}_G^{(t+1)}$，总迭代 $T=5$ 轮即可把 ASR 推到 50%+。

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

- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ICLR 2026\] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](../../ICLR2026/audio_speech/sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](../../ICLR2026/audio_speech/incentive-aligned_multi-source_llm_summaries.md)
- [\[ICML 2026\] SafeSearch: Automated Red-Teaming of LLM-Based Search Agents](safesearch_automated_red-teaming_of_llm-based_search_agents.md)

</div>

<!-- RELATED:END -->
