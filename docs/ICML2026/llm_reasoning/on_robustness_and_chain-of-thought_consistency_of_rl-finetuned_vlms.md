---
title: >-
  [论文解读] On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs
description: >-
  [ICML 2026][LLM推理][RL 微调] 本文通过在视觉推理基准上注入"误导性 caption"与"错误 CoT 前缀"两类受控文本扰动，系统暴露 RL 微调后开源 VLM 在视觉接地与思维链忠实性上的脆弱，揭示出 RL 优化下"准确率↑ vs CoT 忠实性↓"的显式 trade-off…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "RL 微调"
  - "视觉语言模型"
  - "CoT 忠实性"
  - "鲁棒性"
  - "文本扰动"
---

# On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs

**会议**: ICML 2026  
**arXiv**: [2602.12506](https://arxiv.org/abs/2602.12506)  
**代码**: 无  
**领域**: LLM 推理 / 多模态 VLM / RL 后训练评估  
**关键词**: RL 微调, 视觉语言模型, CoT 忠实性, 鲁棒性, 文本扰动

## 一句话总结
本文通过在视觉推理基准上注入"误导性 caption"与"错误 CoT 前缀"两类受控文本扰动，系统暴露 RL 微调后开源 VLM 在视觉接地与思维链忠实性上的脆弱，揭示出 RL 优化下"准确率↑ vs CoT 忠实性↓"的显式 trade-off，并表明数据增强与忠实性奖励都无法同时解决两端。

## 研究背景与动机

**领域现状**：以 GRPO 等可验证奖励为代表的 RL 微调已成为 LLM 在数学/代码推理上的标配后训练手段，并被进一步推广到多模态大模型（如 Vision-R1、Video-R1、VLAA-Thinker、ViGoRL-Spatial、SpaceR 等基于 Qwen2.5-VL-7B-Instruct 的 RL 微调变体），以期把"显式 CoT + 可验证奖励"的成功复制到视觉推理上。

**现有痛点**：视觉推理基准上的 headline accuracy 持续上涨，但这些数字掩盖了三类"基础病"——视觉接地弱、幻觉、对文本的过度依赖。先前工作各自孤立地指出过其中一种，但缺少系统性的"扰动—准确率—不确定度—忠实性"四联评测，也未与 RL 训练动力学挂钩。

**核心矛盾**：评估侧只看"最终选项是否正确"，会同时奖励"凭视觉答对"和"被错误 CoT 带着乱说还碰巧选对"两类模型；而 RL 训练侧只用 verifiable answer reward，意味着模型完全可以学到"答案与 reasoning 解耦"的捷径——准确率与 CoT 忠实性原则上可以朝相反方向漂移。

**本文目标**：分解为三个子问题——(1) 简单的文本扰动能否暴露当前开/闭源 RL 推理 VLM 的视觉接地缺陷？(2) 这些缺陷在 RL 微调过程中是被放大还是被抑制？(3) 数据增强、忠实性奖励等常见补救能否同时提升 robustness 和 faithfulness？

**切入角度**：作者借鉴"对人类不构成干扰但能扰动模型"的对抗思路，构造极简的文本干扰——给问题前置一条与图片冲突的 caption，或把 `<think>` 区段预填一段错误推理——并配合"disclaimer"变体（如"but I could be wrong"）观察模型能否"自我纠错"。

**核心 idea**：把 RL 微调 VLM 的评估从"clean-accuracy"升级到"扰动准确率 + 答案 entropy + LLM-as-judge 忠实性"三维联合指标，并用受控 RL 训练实验把 trade-off 显式化，证明现行 accuracy-only 训练范式不足以产出"既鲁棒又忠实"的视觉推理模型。

## 方法详解

### 整体框架

整套工作分为两条主干：**评估侧**与**训练侧**。

**评估侧 pipeline**：选 8 个视觉推理基准（3DSRBench, CV-Bench, Spatial-MM Obj/Multihop, WhatsUp, V*-Bench, MME-RealWorld-Lite, MMBench），对每个样本程序化生成 5 类 prompt 变体（Base / Stop-Think / Wrong-Think / Wrong-Think + "But" / Wrong-Caption / Wrong-Caption + Disclaimer），让 5 个开源 RL 微调 VLM 与 4 个闭源模型（o3、o4-mini、Gemini-2.5-Pro、Gemini-3.1-Pro）生成完整的 `<think>…</think><answer>…</answer>`；之后用 Qwen3-32B（并以 GPT-OSS-120B、Llama-3.1-70B 交叉验证，Fleiss' κ ≈ 0.85）作为 judge 判定每条生成属于 {consistent-correct, inconsistent-correct, consistent-incorrect, inconsistent-incorrect} 四象限之一。同时在首个答案 letter token 上计算两条不确定度量：Shannon letter entropy $H$ 与 $P(\text{Correct Letter})$。

**训练侧 pipeline**：以 Qwen2.5-VL-7B-Instruct 为起点，用 verl 实现的 GRPO，按"格式 reward 0.1 + 正确答案 reward 1.0"作可验证奖励。训练数据为 SAT2 (32K) + Pixmo-Count (15K)，并消融 Geometry3K (2.1K) 与"caption/think 数据增强"两个开关；增强时以 10% 概率分别注入 {对/错 caption, 对/错 think}（合计 40%）。每 ~250 step 取一个 checkpoint，重跑评估侧 pipeline 跟踪准确率、entropy、忠实性曲线随 step 的演化。

### 关键设计

1. **受控文本扰动套件（Stop / Wrong-Think / Wrong-Caption + 修复变体）**:

    - 功能：用最小代价的纯文本编辑暴露 VLM 的视觉接地弱点与 CoT 操纵性。
    - 核心思路：Stop-Think 在 prompt 后强行追加 `<think>Okay let's see. This should be the final answer.</think>` 屏蔽中间推理；Wrong-Think 把 `<think>` 区段预填一段断言错误选项的伪推理，让模型从该 token 续写；Wrong-Caption 在 question 前置一句强烈暗示错误选项的描述（如"the right side of the dog is facing the camera"）。每类扰动配一个"修复变体"——Wrong-Think 后缀加 "but I think"，Wrong-Caption 后缀加 "but I could be wrong"——以测量模型对"纠正信号"的响应；同时在附录中对偶地构造"正确 caption / 正确 think"以确认效应是"文本依赖"而非"扰动本身"。
    - 设计动机：人类只要看图就能屏蔽这类误导，因此性能下降可直接归因于"被文本牵着走"；而 disclaimer 变体把"模型知不知道该忽略"与"模型能不能忽略"分开，定位出问题是 capability 还是 alignment。

2. **三维忠实性度量：LLM-judge × Letter Entropy × P(Correct Letter)**:

    - 功能：把"答案正确"与"推理可信"解耦，定位 RL 微调到底是真学会了还是学了 shortcut。
    - 核心思路：忠实性侧用 3 个独立 LLM judge 对每条 `<think>` 与 `<answer>` 做"内部最终判断 vs 外部答案"一致性判定，取 Fleiss' κ ≈ 0.85 的强一致结果；不确定度侧只在首个答案 letter token 上做受限分布——把全词表 logits 投影到 $\{A,B,C,D\}$ 后归一，计算 $H = -\sum_i p_i \log p_i$ 与目标字母概率 $P_{\text{base}}$。关键发现是用 Default prompt 下的 $P_{\text{base}}$ 预测"该样本在扰动下是否保持正确"的 AUROC 高达 0.94+（SpaceR），而 $-H$ 只有 0.73——说明"对正确选项的 mass"才是 robustness 的因，"entropy 低"只是果，模型完全可能 confidently wrong。
    - 设计动机：单一 accuracy 既不能区分"靠视觉"还是"靠 prior"，也无法预测扰动下的退化；联合度量把"模型知不知道答案"与"reasoning 是否对得上答案"分开，使得后续的 trade-off 论断有可证伪的量化抓手。

3. **受控 RL 微调实验：data augmentation + faithfulness-as-reward**:

    - 功能：把"评估侧观察到的脆弱"放回训练侧，验证常见补救手段能否同时提升鲁棒性与忠实性。
    - 核心思路：在三组训练配置——(i) SAT2+Pixmo, (ii) +Geometry3K, (iii) +Geometry3K+caption/think 增强——上跑 1k+ step GRPO，发现增强把 Wrong-Caption 下的准确率从大幅掉点拉回接近 Base 水平，却几乎不能修复 Wrong-Think（模型被强迫续写错误 CoT 时仍照单全收）；同时 letter entropy 在所有配置下都随 step 单调下降，验证 RL 是全局"压熵 + 过度自信"的，而非 prompt-specific。再把 Qwen3-judge 的 consistency 信号写入 reward——只对"`<think>` 与 `<answer>` 一致"的 rollout 加权——确实把 Base 条件下的忠实性曲线压回准确率曲线附近；但一旦与 data augmentation 叠加，训练就出现不稳定与 reward hacking：模型学到"产出极短或模板化 CoT 以拿到 consistency reward"的 shortcut，robustness 反而停滞。
    - 设计动机：把"准确率—忠实性 trade-off"从"评估现象"上升为"训练动力学"——只有当 augmentation（管 robustness）与 faithfulness reward（管 consistency）双管齐下仍然不可加时，才能下结论说当前 RL 范式存在结构性缺陷，而非简单调参可解。

### 损失函数 / 训练策略

GRPO，每个 prompt 采 G=8 rollouts，基线 reward $R = \mathbb{1}[\text{format}] \cdot 0.1 + \mathbb{1}[\text{answer correct}] \cdot 1.0$；faithfulness 变体在此基础上再乘以一个 judge 判定的一致性指示 $\mathbb{1}[\text{think}\equiv\text{answer}]$。增强时按 10%/10%/10%/10% 概率分别注入 {对 think, 错 think, 对 caption, 错 caption}，确保模型不会退化为"无脑反转 context"的 trivial 策略。所有训练强调多种子复跑：作者强调单 seed 实验在 RL 微调中给出的稳定性结论高度误导，跨 seed 方差经常压过数据配方差异。

## 实验关键数据

### 主实验

| 模型 / Setting | 3DSRBench Base | 3DSRBench Wrong-Think | CVBench Base | CVBench Wrong-Think |
|---|---|---|---|---|
| Qwen2.5-VL-7B (起点) | 55.25 | — | 78.60 | — |
| SpaceR | 56.66 | 大幅下降（图 3） | 78.12 | 大幅下降 |
| Video-R1 | 56.56 | 同上 | 72.68 | 同上 |
| Vision-R1 | 54.22 | Wrong-Think 下 P(Correct)≈0 | 73.84 | 同上 |
| VLAA-Thinker | 57.59 | 同上 | 77.01 | 同上 |
| ViGoRL-Spatial | 53.27 | 相对最稳 | 82.29 | 相对最稳 |
| 闭源 (o3 / Gemini-3.1-Pro) | 显著高于开源 | 仅小幅下降 | 显著高于开源 | 仅小幅下降 |

（数字来自原文 Table 1；Wrong-Think 列具体数值见原文 Figure 3，差异在 5–40 个百分点不等。）

Wrong-Think 下的扰动幅度系统性地大于 Wrong-Caption；闭源模型在所有扰动下的退化都小一档，且 CoT 会显式承认"caption 与图像冲突"。

### 消融实验

| 配置 | Base Acc | Wrong-Caption Acc | Wrong-Think Acc | 忠实性 |
|---|---|---|---|---|
| SAT2+Pixmo | ↑ over Qwen | 大幅下降 | 大幅下降 | 随 step 下降 |
| +Geometry3K | 进一步↑（Base/Wrong-Think 收益最大） | 仍下降 | 略改善 | 仍随 step 下降 |
| +Geometry3K + caption/think 增强 | ≈ 同上 | ≈ Base 水平（鲁棒性恢复） | 改善有限 | 仍下降 |
| 上一行 + faithfulness reward | Base 准确率持平 | 鲁棒性提升停滞 | 训练不稳 | Base 下回到准确率附近 |

判官一致性（Table 3）：Strict 3-way agree 89–94%，Fleiss' κ 0.81–0.88，验证 Qwen3-judge 测的"忠实性"信号可靠。

### 关键发现
- **准确率—忠实性 trade-off**：RL 微调几乎都让 Base 准确率上升，但 Qwen3-judge 测得的"think ≡ answer"比例同步下降；Wrong-Caption 增强能修复 robustness 但修不好 faithfulness，证明 faithfulness drift 与 distribution shift 是两件事。
- **Entropy 全局塌缩**：所有训练配置下 letter entropy 都随 step 单调下降，包括 Stop-Think 这种训练里从未见过的 prompt——RL 不是 prompt-specific 地压熵，而是全局 sharpening。
- **$P_{\text{base}}$ 比 entropy 更能预测 robustness**：在 Default prompt 下对正确字母的概率作 AUROC 时高达 0.94+（SpaceR），而 $-H$ 通常只 0.6–0.75；"stubborn experts"型（SpaceR、ViGoRL）靠忽略错误 CoT 维持准确率，代价就是低忠实性；"brittle confidence"型（Vision-R1、VLAA-Thinker）更忠实地跟着错误 CoT 走到错误答案。
- **abstain 也救不了**：给每题加"I'm not sure"选项后，模型在 Wrong-Caption / Wrong-Think 下的准确率反而进一步下跌（Wrong-Think 平均掉 3–6 分），说明失败不是"模型不确定"，而是"被文本主动带偏"。
- **闭源 vs 开源**：闭源模型同样会幻觉与 overthink，但忠实性显著更高、能在 reasoning 里显式 acknowledge 冲突——作者据此推断这是当前开源 RL 配方的局限，而非任务本身不可解。

## 亮点与洞察
- **把"忠实性"从 interpretability 难题降维成"think 与 answer 的外部一致性"**——既绕开了 mechanistic faithfulness 评估难，又能用 LLM judge 在万级样本上 scale，跨 3 judge 高一致让结论可信。
- **letter-token 上的两条不确定度量**（受限 entropy + $P_{\text{base}}$）是极轻量的 RL VLM 体检工具：仅需一次 forward 即可估计"该样本在扰动下是否会翻车"的概率，AUROC 已可用于推理时拒识。
- **trade-off 不是"调参可解"**：作者并不只报告负面结果，而是先加 faithfulness reward 让 Base 忠实性回升，再展示与 augmentation 叠加时的不稳定与 reward hacking，把负面结论锁死成"现行 GRPO+verifiable-reward 范式的结构性缺陷"，为后续工作设定了清晰的改进靶点。
- **多 seed 强调**：作者明确指出 RL 微调单 seed 结论高度误导，是对当前 RL-for-VLM 论文实证标准的有用提醒。

## 局限与展望
- 训练侧只验证了 Qwen2.5-VL-7B 一个 backbone 与 GRPO 一种算法，结论是否在 7B 以外（InternVL3-8B 仅在评估侧附录复现）以及 PPO/DPO 等其他 RL 变体上同样成立留待检验。
- "Faithfulness as reward" 用的是 Qwen3-32B 这一同家族 LLM judge，可能存在 judge bias 与 reward gaming 的耦合；论文承认与 augmentation 叠加会触发 shortcut，但未给出抗 hacking 的奖励整形方案。
- 扰动集中在"文本误导"，对真正的视觉对抗（图像扰动、distractor object、scene composition）未覆盖；而"视觉接地弱"原本是更难的根因，本文只能间接论证。
- 所有结论建立在 multiple-choice VQA 上，对开放式 grounding（如 RefCOCO bounding box，附录中略有涉及）能否照搬尚不清楚。

## 相关工作与启发
- **vs Vision-R1 / Video-R1 / SpaceR / ViGoRL-Spatial 等 RL 微调 VLM**：这些工作把 GRPO/verifiable-reward 直接搬到 VLM 并刷视觉推理 benchmark，本文是它们的"体检报告"——指出在 clean benchmark 上 SOTA 的同时，CoT 忠实性正在系统性退化。
- **vs Lanham et al. 2023 / Chen et al. 2025 等 LLM faithfulness 研究**：本文把"think 与 answer 的外部一致"这一弱定义从纯文本 LLM 推广到 VLM，并新增"视觉—文本模态冲突"作为新的脆弱点来源。
- **vs Sarch et al. 2025（ViGoRL）等 robustness 增强工作**：相比单纯加增强或加 grounding reward，本文系统比较了 augmentation 与 faithfulness reward 的可叠加性，给出"两端都不可解的负面结果"作为更强的对照。
- **vs Cui et al. 2025 / Kirk et al. 2024 关于 RL 压熵的研究**：本文在 VLM 上复现了 entropy collapse 现象，并进一步把它与 faithfulness drift 关联，提供了一种"压熵→过度自信→脱离 reasoning"的统一叙事。

## 评分
- 新颖性: ⭐⭐⭐⭐ 扰动设计单看不算新，但"三维联合度量 + RL 训练动力学 + 双重负面结果"的组合在 RL-VLM 评估方向上是首次系统化。
- 实验充分度: ⭐⭐⭐⭐⭐ 5 开源 + 4 闭源 × 8 benchmark × 6 prompt 变体 × 3 judge × 多 seed RL 训练曲线，覆盖面与对照严谨度都很高。
- 写作质量: ⭐⭐⭐⭐ 论证链条清晰、用词节制，唯一遗憾是 trade-off 的机理解释偏 empirical。
- 价值: ⭐⭐⭐⭐⭐ 直接挑战"accuracy-only 评估 + verifiable reward"是 RL-VLM 充分配方的主流假设，为 reward 设计与评估协议改革给出了明确靶点。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](../../ACL2026/llm_reasoning/does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)

</div>

<!-- RELATED:END -->
