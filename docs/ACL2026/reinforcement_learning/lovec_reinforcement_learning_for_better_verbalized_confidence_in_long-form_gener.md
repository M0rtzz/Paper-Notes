---
title: >-
  [论文解读] LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations
description: >-
  [ACL 2026][强化学习][长文本生成] LoVeC 教 LLM 在长文本生成过程中每写完一句就追加一个 0–10 的 `<confidence>` 数字标签，用 GRPO（在线，需 oracle fact-checker）或 DPO（离线偏好对）训练让该标签与 GPT-4o 判定的事实性对齐…
tags:
  - "ACL 2026"
  - "强化学习"
  - "长文本生成"
  - "语言化置信度"
  - "GRPO"
  - "DPO"
  - "事实性校准"
---

# LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations

**会议**: ACL 2026  
**arXiv**: [2505.23912](https://arxiv.org/abs/2505.23912)  
**代码**: https://github.com/caiqizh/LoVeC (有)  
**领域**: LLM 校准 / RLHF / 幻觉检测  
**关键词**: 长文本生成、语言化置信度、GRPO、DPO、事实性校准

## 一句话总结
LoVeC 教 LLM 在长文本生成过程中每写完一句就追加一个 0–10 的 `<confidence>` 数字标签，用 GRPO（在线，需 oracle fact-checker）或 DPO（离线偏好对）训练让该标签与 GPT-4o 判定的事实性对齐，单次解码即可输出可校准、可机器解析的置信度，比 SOTA LUQ 在 Brier/ECE/Spearman 上全面更好且推理速度快 20 倍。

## 研究背景与动机

**领域现状**：长文本 QA 的幻觉检测主流方法分两类——基于采样的一致性方法（如 LUQ、SelfCheckGPT，需要多次采样 + 相似度比较）和基于 GPT 抽 atomic claim 后逐项打分（Fadeeva 2024、Liu 2024）；它们都是后处理且依赖外部模型，单次推理成本高。

**现有痛点**：① 一致性方法对单个 query 要重采样 5–10 次再算相似度，A100 上跑完 792 条 WildHallu 测试集要 1500+ 秒；② atomic-claim 解构需要调 GPT-4 API，成本和延迟都高；③ Verbalized confidence 虽便宜，但现有 verbalized 方法（如 LoGU、Linguistic Calibration）输出 "I believe"、"70% uncertain" 这类自然语言短语，难以机器解析、无法直接接阈值；④ 现有 verbalized 工作几乎全在 short-form QA，没人系统地在长文本生成的 sentence-level 做过。

**核心矛盾**：长文本里一段话有多个事实陈述，置信度应该随每句变化——但 SFT 只能学到 token-level likelihood，无法把"句子内容"和"置信度数字"作为一个联合 action 联合优化；而且 SFT 对 negative example 几乎没有反馈，无法学到"宁可说不知道也别自信地错"这种非对称代价。

**本文目标**：让模型在单次 decoding 内一边写句子一边生成对齐事实性的 `<confidence> N </confidence>` 数字标签，且对 in-domain（WildHallu）和 out-of-domain（Bios / PopQA）都鲁棒。

**切入角度**：把"写句子 + 标置信度"看作一个 sequential decision，用 RL 直接在 (sentence, confidence) 这个联合 action 上做信用分配——奖励 confidence 与 fact-checker 判定 factuality 的对齐程度，且用 log-base reward 强惩 overconfident 错误。

**核心 idea**：RL（GRPO + DPO）+ binary-cross-entropy log reward 在 (s_i, c_i) 上联合优化，单次 decoding 同时产出文本与可解析的数字置信度。

## 方法详解

### 整体框架

给定 query $q$，策略 $\pi_\theta$ 输出形如 $y=\{(s_1,c_1),\dots,(s_n,c_n)\}$ 的 sentence-confidence 对，$c_i\in\{0,1,\dots,10\}$。训练分两步：(1) 先用 winning 样本 $y_w$ 做 1 epoch 的 SFT 让模型学会输出 `<confidence>N</confidence>` 格式；(2) 再选 GRPO（有 fact-checker oracle 时）或 DPO（无 oracle 时只需偏好对）跑 1 epoch RL，LoRA 微调 q/k/v/o_proj（< 1% 参数）。评测设两种协议：free-form tagging（模型同时输出答案和置信度）与 iterative tagging（给定固定文本，仅逐句预测置信度，方便公平对比）。

### 关键设计

1. **GRPO + log-base 校准 reward**:

    - 功能：把 confidence-factuality 对齐写成可微的群体相对优势信号，online 优化策略。
    - 核心思路：把 $c_i, f_i$ 都归一到 $[0,1]$，定义置信度奖励 $r_{\mathrm{conf}} = \lambda\cdot \frac{1}{n}\mathbf{1}^\top\left(1+\frac{f\odot\log c + (1-f)\odot\log(1-c)}{R_{\max}}\right)$，本质上是 BCE 的负值，再叠加 informativeness 与 format 子奖励；对每个 query 采样 $G$ 条轨迹 $\{y_j\}_{j=1}^G$ 算 group-mean 归一的优势 $\hat A_j = \frac{r_j - \mathrm{mean}(r)}{\mathrm{std}(r)}$，最终 GRPO loss $L_{\mathrm{GRPO}}(\theta) = -\mathbb{E}\big[\frac{1}{G}\sum_j(\hat A_j(\pi_\theta,\pi_{\mathrm{old}}) - \beta D_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}])\big]$；reward 还做了 $\gamma=1.5$ 的拉伸 $r\leftarrow \mathrm{sign}(r)|r|^\gamma$ 放大好坏样本差异。
    - 设计动机：相比 linear 或 quadratic loss，log reward 对"高置信但事实错"惩罚极重（接近 $-\infty$），是 proper scoring rule，能强迫模型 calibrate 而非只学排序；group-relative advantage 不需要单独 critic，省显存。

2. **DPO + 算法 1 的合成偏好对**:

    - 功能：在没有 online fact-checker 的离线场景下用偏好学习达成同样目标。
    - 核心思路：对每个 query $(q,E)$，先用 base 模型生成纯文本 $y_{\mathrm{base}}=\{s_1,\dots,s_n\}$，用 GPT-4o + 检索证据算 fact label $f_j$，构造 winning 集合 $y_w=\{(s_j,f_j)\}$（用真实事实分作 confidence）和 losing 集合 $y_l=\{(s_j,c'_j)\}$，其中 $c'_j$ 从 $\{0,\dots,10\}\setminus\{f_j\}$ 均匀采样——保持句子相同，只让置信度数字偏离真实事实分；然后用标准 DPO 损失 $L_{\mathrm{DPO}}=-\mathbb{E}\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|q)}{\pi_{\mathrm{SFT}}(y_w|q)} - \beta\log\frac{\pi_\theta(y_l|q)}{\pi_{\mathrm{SFT}}(y_l|q)}\big)$ 优化。
    - 设计动机：DPO 不需要 RL 训练时调 GPT-4o（成本极高），把 oracle 调用集中在数据构造阶段；同时通过保持 sentence 一致仅扰动 confidence 这种巧妙构造让模型聚焦于学"如何打分"而非"如何写"。

3. **Free-form vs. Iterative Tagging 双协议**:

    - 功能：拆解评估"内容生成质量"与"打分质量"的混淆。
    - 核心思路：Free-form 让模型自由生成 $y_t = \arg\max_{y_t}\pi_\theta(y_t|y_{<t},q)$，同时输出答案和 `<confidence>`；Iterative 固定 base 模型已生成的 $\{s_1,\dots,s_n\}$，模型只逐句条件预测 $c_i = \arg\max_c \pi_\theta(\{q,(s_1,c_1),\dots,(s_{i-1},c_{i-1}),s_i\},c)$。后者剥离了内容变异，让不同模型在完全相同的文本上比谁打分更准。
    - 设计动机：现有 verbalized 方法各自生成不同内容，BS/ECE 没法横向对比；iterative 用固定文本提供 apples-to-apples 评估基线，free-form 则保留真实使用场景。两个一起用就能分辨"是模型写得更好还是打分更准"。

### 损失函数 / 训练策略
- SFT 一遍只在 $y_w$ 上学格式；GRPO/DPO 各一遍 LoRA 微调（rank 默认，q/k/v/o_proj），AdamW；A100×8 共 1500 GPU 小时；GRPO reward stretching $\gamma=1.5$，并加 0.15×correctness 小奖励防"全说不知道"。
- backbone：Llama-3-8B-Instruct 与 Gemma-2-9B-It；评测指标 Brier Score、ECE-M（soft label 版本）、Spearman Correlation 三件套，覆盖 calibration 与排序两类需求。

## 实验关键数据

### 主实验（Llama-3-8B-Instruct，iterative + free-form）

| 数据集 | 方法 | BS↓ | ECE-M↓ | SC↑ |
|---|---|---|---|---|
| WildHallu | LUQ (prev SOTA) | 14.5 | 21.5 | 56.8 |
| WildHallu | LoVeC-GRPO (iter) | **5.7** | **2.5** | 57.0 |
| WildHallu | LoVeC-DPO (iter) | 6.0 | 5.0 | **60.4** |
| Bios | LUQ | 20.0 | 29.5 | 63.8 |
| Bios | LoVeC-GRPO (iter) | **8.5** | **4.2** | 64.7 |
| Bios | LoVeC-DPO (iter) | 9.0 | 7.3 | **65.6** |
| PopQA | LUQ | 16.7 | 23.2 | 62.5 |
| PopQA | LoVeC-DPO (iter) | **9.6** | **1.7** | **63.1** |

三个数据集上 BS / ECE-M 全部腰斩，Spearman 也微涨；free-form 版本与 iterative 趋势一致（GRPO BS 5.7–10.1、ECE-M 5.1–11.1）。WildHallu 测试集 792 条推理用时：LUQ 1525s vs. LoVeC-iterative 64s（**~24× speedup**），LoVeC-freeform 139s（~11× speedup）。

### 消融实验

| 配置 | BS↓ | ECE-M↓ | SC↑ | 说明 |
|---|---|---|---|---|
| LoVeC-GRPO 全配置 (WildHallu) | 5.7 | 2.5 | 57.0 | base |
| 用 log reward | 5.7 | 2.5 | 57.0 | proper scoring rule |
| 用 linear/quadratic reward | ↑ | ↑ | ↓ | 校准明显变差 |
| DPO 用 GPT-4o oracle | 6.0 | 5.0 | 60.4 | 默认 |
| DPO 用 self-label (frozen self) | 略差 | 略差 | 略差 | 仍优于 LUQ baseline |
| SFT 用 regression loss 代替 CE | ↑ | ↑ | ↓ | 各指标全部恶化 |
| Iterative 不见前面打分 | ↑ | ↑ | ↓ | 移除"局部校准锚"显著掉分 |

GRPO 训练动态：mean reward 13.86 → 29.83（5667 step / 1 epoch），ECE-M 同步从 15.2 降到 2.5，无 reward collapse。

### 关键发现
- RL 比 SFT 关键的不是表面分数，而是 token 排序结构：GRPO 在 next-token 预测时 top-15 词单调 `10,9,8,...,0`（事实正确时）或 `2,3,4,...`（事实错误时围绕真实分排序），DPO 部分有序，SFT 完全无序——这种"概率分布反映可信度阶梯"才是 RL 真正注入的归纳偏置。
- 移除拉丁/上下文 oracle（用 self-label DPO）仍胜 LUQ，说明该方法对 oracle 强度不敏感，可在没有 GPT-4o 的工业场景部署。
- 与 LUQ 互补：LoVeC-DPO + LUQ 简单平均后 Spearman 还能再 +5 个点，verbalized 信号与 sampling 信号是两路正交证据。
- 在 short-form TriviaQA 上 zero-shot 迁移仍有竞争力，逼近专门为 short-form 设计的 RewardingDoubt，说明 RL 学到的是"如何把 likelihood 反映成数字"的通用技能。

## 亮点与洞察
- 把"长文本置信度"重新定义为"sentence-level 数字 + RL 联合优化"，避免了 atomic-claim 抽取的 GPT 调用和一致性采样的多次解码，是工程上最实用的形式。
- log reward 把校准变成 proper scoring，加上 reward stretching 与小幅 correctness bonus 这种"非对称 + 反 hacking"的奖励工程值得做 RLHF 校准的人借鉴。
- DPO 偏好对构造把"sentence 完全相同、只动 confidence"作为对比维度，使模型只学打分本身、不污染语言能力，这种"控制变量法"的偏好对构造适用于任何需要让模型学一类元能力（如安全等级、格式严格度）的场景。
- 双协议评测把"内容质量"和"打分质量"解耦，是长文本可信 AI 评估的范本——任何后续 verbalized confidence 工作都该补 iterative tagging 这一档。

## 局限与展望
- 只适用白盒模型（需 LoRA + RL），无法部署在 OpenAI / Anthropic 等纯 API 模型上。
- 评估只看事实性（factuality），未涵盖一致性、有害性、创意写作等其它可校准维度。
- Sentence-level granularity 在一句话里包含多个相互矛盾事实时仍欠粒度，未来可推到 sub-sentence 或 atomic-claim 级。
- 训练奖励依赖 GPT-4o 的 fact-check，间接继承 GPT-4o 的事实判定偏见；自标签 ablation 虽证可行但表现略弱。
- 未在代码生成、机器翻译等其他长文本任务验证；高风险领域（医疗 / 法律）的真实部署效果未知。

## 相关工作与启发
- **vs LUQ (Zhang 2024a)**：LUQ 靠 5–10 次采样 + sentence-level 一致性聚合，本文是单次 decoding 内的 verbalized；同样 sentence 级别但 LoVeC 把判断挪到训练期、推理时零额外开销。两者还能融合再涨 5 点 SC。
- **vs LoGU (Yang 2025a) / Linguistic Calibration (Band 2024)**：他们用 DPO/PPO 让模型说 "I believe" / "I'm uncertain"，难以机器解析；本文用 `0–10` 数字标签实现 machine interpretability，可以直接接阈值或排序。
- **vs RewardingDoubt / SaySelf**：都是 RL 校准但局限 short-form；本文是首个把 RL 校准做到 long-form 且公开 GRPO+DPO 双方案的工作。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一个长文本 sentence-level verbalized confidence + RL 系统化方案，双协议评测有原创性
- 实验充分度: ⭐⭐⭐⭐ 三数据集 + Llama/Gemma + 多 RL 算法 + reward 形式 + oracle 形式 + 见/不见前置等 6+ 项 ablation
- 写作质量: ⭐⭐⭐⭐ 动机推导细致，case study 用 token 排序证明 RL 的内化效应非常生动
- 价值: ⭐⭐⭐⭐ 20× 速度 + 显著校准改善对生产环境吸引力大；reward 设计与偏好对构造范式可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning](unicreative_unifying_long-form_logic_and_short-form_sparkle_via_reference-free_r.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[ICML 2026\] CAMEL: Confidence-Gated Reflection for Reward Modeling](../../ICML2026/reinforcement_learning/camel_confidence-gated_reflection_for_reward_modeling.md)
- [\[ACL 2026\] A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks (EAGLET)](a_goal_without_a_plan_is_just_a_wish_efficient_and_effective_global_planner_trai.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](../../ICLR2026/reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)

</div>

<!-- RELATED:END -->
