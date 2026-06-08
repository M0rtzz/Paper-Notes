---
title: >-
  [论文解读] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning
description: >-
  [ICML 2026][LLM推理][化学推理] LatentChem 在化学 LLM 上把"显式 CoT 文本链"换成"连续 latent 思考向量 + 动态分子感知更新"，并在 GRPO 纯结果奖励下观察到模型**自发抛弃文本 CoT**、改用 latent 推理…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "化学推理"
  - "latent thinking"
  - "连续思考向量"
  - "GRPO"
  - "modality mismatch"
---

# LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning

**会议**: ICML 2026  
**arXiv**: [2602.07075](https://arxiv.org/abs/2602.07075)  
**代码**: 有（论文声明 GitHub + HuggingFace 公开，但脚注未给出确切链接）  
**领域**: LLM推理 / 科学计算 (化学 LLM)  
**关键词**: 化学推理, latent thinking, 连续思考向量, GRPO, modality mismatch

## 一句话总结
LatentChem 在化学 LLM 上把"显式 CoT 文本链"换成"连续 latent 思考向量 + 动态分子感知更新"，并在 GRPO 纯结果奖励下观察到模型**自发抛弃文本 CoT**、改用 latent 推理，在 ChemCoTBench 上对显式 CoT baseline 非平局胜率 59.88%，推理步数平均下降 10.84 倍、wall-clock 加速 5.96 倍。

## 研究背景与动机

**领域现状**：化学 LLM（如基于 Qwen-3 + SMI-TED 编码器的领域微调模型）目前几乎全部走"显式 CoT"路线 —— 把电子离域、空间位阻、官能团修饰等连续物理化学直觉强行线性化成自然语言 token 链，再产出答案。Mol-Instructions、ChemCoTBench 等基准也默认以文本 CoT 作为推理接口。

**现有痛点**：化学的内在性质景观是高维连续流形（性质如 LogP、QED 随结构平滑变化），但文本 token 是离散低带宽符号。把"加一个甲基"这种本可一步完成的结构操作硬翻译成"分析结合位点 → 校验价键 → 描述加成"等几十个 token，会产生作者所谓的"jagged staircase"轨迹：既慢（推理 step 爆炸）又容易在长链中漂移产生幻觉。

**核心矛盾**：连续的物理化学动力学 vs. 离散的语言符号通道存在天然的 **modality mismatch**。文本 CoT 既不是化学逻辑的原生载体，也不一定是最优的计算介质 —— 但现有"latent reasoning"工作（如 Coconut）虽把推理搬到 latent 空间，却把分子嵌入当成**静态上下文**，无法在推理过程中重新聚焦不同子结构。

**本文目标**：构造一个化学专用的 latent reasoning 接口，回答两个问题 —— (1) 当模型被允许在连续 latent 空间推理且不再被强制吐出文本 CoT 时，它会自发选择哪种模态？(2) 这种选择是"走捷径"还是真正更优的计算策略？

**切入角度**：把推理与生成解耦 —— 自然语言只做输入/输出接口，中间推理走"连续思考向量 + 动态感知刷新"的循环；再用 GRPO 只奖励格式合法、结构有效、答案正确，不奖励 brevity 也不奖励 CoT 省略，让模型自主决定要不要文本。

**核心 idea**：用 **ChemUpdater 动态再查询分子表征** + **latent projector 闭环送回 hidden state** 构造"感知-推理"双通路；在结果导向 RL 下，模型会自发把 CoT 内化进 latent 空间。

## 方法详解

### 整体框架
LatentChem 想验证一个假说：化学的连续物理直觉被强行翻译成离散文本 token 是"模态错配"，连续 latent 才是更原生的推理介质。为此它在通用 LLM backbone（默认 Qwen-3-8B）上把自然语言降级为纯输入/输出接口，中间推理改走"感知-思考"双通路——SMI-TED 编码器抽出的分子特征先被 Chemical Adapter 压成定长 ChemTokens 软提示注入 LLM，然后模型在文本指令之后连续吐出若干步 hidden state（不解码成文本，直接经 latent projector 喂回自己），每吐一步 ChemUpdater 就拿最新 thought 去 cross-attend 历史、刷新 ChemTokens 让模型"再看一眼分子"，直到吐出 `<end_latent>` 或耗尽预算才解码答案。整个系统在 4 阶段渐进训练下成型，最关键的是末阶段只用结果奖励、不奖励 brevity，却观察到模型自发抛弃文本 CoT。

### 关键设计

**1. Chemical Adapter：让分子表征定长且可被反复再查询。**

普通的 prefix tuning 只是"塞个 embedding"进去，但 LatentChem 后续要让 ChemUpdater 在推理过程中不断回头查询分子，这就要求分子表征既定长又忠实反映结构。Adapter 借 Perceiver Resampler 思路，用 $N$ 个可学习 latent query $\mathbf{Q}$ 对 SMI-TED 输出的变长稠密特征 $\mathbf{H}_{mol}\in\mathbb{R}^{L\times d_{enc}}$ 做 cross-attention，$\mathbf{H}_{chem}=W(\text{LN}(\mathbf{Q}+\text{MHA}(\mathbf{Q},\mathbf{H}_{mol},\mathbf{H}_{mol})))$，每个 query 像一个"语义锚点"自动萃取某类化学属性，再用线性层 $W$ 投影到 $d_{llm}$ 维得到定长 ChemTokens。为防止 adapter 偷懒靠文本先验蒙答案，Stage 1 在 answer-only 监督外加一个反事实对齐损失 $\mathcal{L}_{CF}$（hinge loss，最大化"干净分子 vs. 扰动分子"下的答案似然差），逼它把答案真正需要的化学性质压进 ChemTokens——否则后面 ChemUpdater 拿到的就是空壳，根本无从 refocus。

**2. ChemUpdater：把静态编码器升级成被推理状态调度的可微感知器。**

这是 LatentChem 与 Coconut 拉开差距的核心。Coconut 把分子嵌入当成一成不变的上下文，于是 latent 思考链越长就漂得离原始结构越远；ChemUpdater 则让模型每推理一步都"再看一眼分子，但是看不同的地方"。具体做法是以当前 ChemTokens $\mathbf{H}_{chem}^{(t)}$ 为 query、以全部历史 thought $\mathbf{Z}_{1:t}$ 为 key/value 做 cross-attention 后残差刷新，$\mathbf{H}_{chem}^{(t+1)}=\text{LN}(\mathbf{H}_{chem}^{(t)}+\text{CrossAttn}(\mathbf{H}_{chem}^{(t)},\mathbf{Z}_{1:t},\mathbf{Z}_{1:t}))$，等于把分子编码器从"一次性静态特征提取器"变成"随推理状态动态调度焦点的感知器"。在分子优化这类需要反复定位不同子结构（先看结合位点、再看价键、再看官能团）的任务上这一点至关重要——消融里去掉它 SR 直接掉 12%。

**3. Latent Projector + GRPO 纯结果奖励：自发内化的闭环与触发器。**

要让推理在连续空间闭环，必须把 LLM 输出空间的 thought 向量对齐回输入空间。Latent projector 是个轻量残差 FFN，$\mathbf{h}_{t+1}=\mathbf{z}_t+\text{FFN}(\text{LN}(\mathbf{z}_t))$，把 raw hidden state $\mathbf{z}_t$ 映回输入 embedding 当下一步输入，绕开 tokenization 瓶颈。真正的"实验仪器"设计在 Stage 4：奖励只含 format / validity / correctness 三项，**完全不含 brevity 也不含 CoT 省略项**，并冻结 latent 模块、放开 backbone，把学到的 latent 动力学当作稳定"内部模拟器"让 backbone 去消费决策。在这种纯结果驱动下，模型仍自发抛弃了文本 CoT、只吐一个 "." 或 ":" 当过渡 token 后直接生成 XML 答案——这正好验证了 modality mismatch 假说：给模型自由选模态、只用结果奖励，它若选连续 latent 就说明 latent 真的更适配化学逻辑。配套的 causal ablation（把前 $k$ 个 latent 步替换成高斯噪声会显著掉点）进一步证明这些"沉默步"在做实质计算而非走过场。

### 损失函数 / 训练策略
四阶段渐进训练逐步把推理从文本搬进 latent。Stage 1 训 adapter+LLM、禁用 latent，用 answer-only 监督加反事实对齐 $\mathcal{L}_{total}^{(1)}=\mathcal{L}_{clean}+\lambda\mathcal{L}_{CF}$ 先把分子语义压进 ChemTokens；Stage 2 仍训这两个模块，但加上显式 CoT 监督 $\mathbf{y}_{full}=[\mathbf{y}_{cot},\mathbf{y}_{ans}]$、把反事实对齐扩到全序列，让模型先学会用文本链推理；Stage 3 反过来冻结 adapter+LLM、只训 ChemUpdater + latent projector，让 latent 模块去适应已经固化的语义空间、学会生成"能被解码器接住"的 thought 向量；Stage 4 再冻结 latent 模块、放开 backbone，用 GRPO 优化复合奖励（format + validity + correctness），让 backbone 学会把 latent thought 当内部模拟器去决策。训练数据为 ChemCoTDataset 2025-11 snapshot 约 14k CoT 样本，覆盖分子理解 / 编辑 / 优化 / 反应预测。

## 实验关键数据

### 主实验

| 任务 | 指标 | LatentChem | Stage 1+2+4 (Explicit CoT) | Coconut-Chem | Δ vs CoT |
|------|------|------------|----------------------------|--------------|----------|
| Mol Optim. LogP | SR% | **96** | 77 | 44 | +19 |
| Mol Optim. GSK3-β | SR% | **82** | 67 | 47 | +15 |
| Mol Optim. Solubility | SR% | **89** | 86 | 58 | +3 |
| ChEBI-20 描述 | METEOR | **0.15** | 0.05 | 0.07 | +0.10 |
| ChemLLMBench 描述 | METEOR | **0.16** | 0.07 | 0.04 | +0.09 |
| ChemCoTBench All | 非平局胜率 ℛ*_win | **59.88%** | — (基线) | 32.11% | 显著胜 |
| ChemLLMBench All | ℛ*_win | **55.58%** | — | 44.82% | 显著胜 |
| ChEBI-20 Open | ℛ*_win | **85.26%** | — | 65.58% | 显著胜 |
| Mol-Instructions All | ℛ*_win | 49.88 | — | 40.57 | 平局附近 |

> 8B 的 LatentChem 在分子优化上还超过了 ChemCoTBench 报告的 Claude 3.7 Sonnet SOTA；推理 step 平均下降 10.84×（反应任务最高 29.9×），wall-clock 加速 5.96×。

### 消融实验

| 配置 | Mol Optim. SR% ↑ | Mol Description METEOR ↑ |
|------|------------------|--------------------------|
| LatentChem (Full) | **80.67** | **0.143** |
| w/o ChemUpdater | 68.67 (−12.0) | 0.068 (−0.075) |
| w/o Latent Projector | 69.83 (−10.8) | 0.087 (−0.056) |
| w/o Latent Thinking | 71.00 (−9.67) | 0.052 (−0.091) |

Scale-matched：换 4B 和 14B backbone 重做 LatentChem vs. 同规模 Explicit CoT，ChemCoTBench All 胜率分别 52.42% / 51.80%，趋势一致，说明优势不来自参数规模。

### 关键发现
- **GRPO 阶段自发 CoT 内化**：尽管 Stage 1-3 全程被显式 CoT 监督，Stage 4 一旦只用结果奖励，模型主动停止生成中间文本，只吐一个 "." 或 ":" 作为过渡 token 后直接给 XML 答案 —— 这种"内化"完全不在 loss 里被鼓励，是奖励驱动下的策略迁移。
- **Causal necessity**：把前 $k$ 个 latent 步替换成高斯噪声会显著掉点，证明这些"沉默步"确实在做编码而非装样子。
- **预算 stress test 的"hydraulic trade-off"**：当 latent 预算 $T\geq 6$ 时模型几乎不吐文本 CoT；预算压到 $T<6$ 时模型自动**重新激活文本 CoT** 来补足推理深度 —— 模型学会在隐式/显式两种推理模态间动态调度。
- **Latent 流形的 functional partitioning**：t-SNE 显示初始 step 0 任务表征全部纠缠，前 2 步内迅速分裂成任务专属簇并稳定到 step 10；RSA 表明 latent 几何与 Tanimoto 化学拓扑的 Spearman 相关在整条推理链上保持稳定 —— 大幅 latent 更新与结构编码方向**正交**，没有破坏物理保真度。
- **任务依赖性**：开放生成类（分子优化、描述）latent 优势压倒性；闭式任务（反应预测）latent 与 CoT 持平 —— 验证 latent 的核心红利是"连续流形上的创造性探索"，确定性映射上语言 token 已经够用。

## 亮点与洞察
- **把 latent reasoning 当作实验仪器而非性能 trick**：作者不是先验地说"latent 更好"，而是构造一个允许模型自由选择推理模态的接口，再观察 GRPO 下的自发行为 —— 这种"用模型选择当证据"的论证方式很有说服力，避免了"latent 因为参数更多/训练更久所以更强"的反驳。
- **ChemUpdater 的"动态再感知"思想可迁移**：把 encoder 从静态特征源升级为"被推理状态查询的可微感知器"这一范式，对蛋白质、3D 几何、代码 AST 等需要在推理过程中反复 refocus 不同子结构的领域都有借鉴价值。
- **奖励里不放 brevity 却得到 brevity**：用 outcome-only 奖励触发内化，是对"奖励黑客 = 性能下降"的反例 —— 模型选择短输出反而提升正确率，说明此时短不是 cheat，而是模型识别到 latent 比 text 更原生。
- **"hydraulic trade-off"的认知架构暗示**：模型在 latent 预算紧时自动 fallback 到文本 CoT，天然对应 Kahneman System 1/2 框架，作者明确把它定位为未来"混合认知架构"的雏形。

## 局限与展望
- **可解释性塌方**：latent 推理过程对人类不透明，post-hoc 分析虽然能验证物理 grounding，但分子修饰的中间步骤无法被审计 —— 对要求可验证推理的科学应用是硬伤。
- **Mol-Instructions 平局**：在该 benchmark 上 ℛ*_win 仅 49.88%（接近 50% 平局线），说明 latent 优势是任务依赖的，不是无差别普适。
- **训练稳定性与奖励设计敏感性**：作者承认 GRPO 稳定性、奖励权重选择跨任务的鲁棒性都未充分探索；另外 Stage 4 没有单独的 RL prompt 数据集，与 SFT 数据共享分布可能导致一定 source overlap。
- **依赖 SMI-TED + 特定 adapter 架构**：所有 latent 模型实验都共享 Perceiver Resampler 这一具体结构，没有验证对其他分子编码器（如 Uni-Mol）的可迁移性。
- **未来方向**：作者点名 hybrid System 1/2 架构 —— 重结构计算走 latent，需要 justification 时把 latent thought 解码回自然语言；笔者补充，反向 distillation（用 latent thought 反过来生成人类可读 CoT 做 audit trail）是个自然延伸。

## 相关工作与启发
- **vs Coconut (Hao et al., 2025c)**：Coconut 把推理搬到 latent 但分子嵌入静态；LatentChem 用 ChemUpdater 让分子表征随推理动态刷新。同骨架 (Qwen-3-8B + SMI-TED) 下 Coconut-Chem 在 ChemCoTBench 仅 32.11% 胜率，LatentChem 59.88%，差距来自这个动态感知差异。
- **vs Explicit Chemical CoT (Stage 1+2+4)**：完全相同的网络结构与 GRPO 训练流程，只是 reasoning 接口换成显式 CoT，胜率作为基线（50%）。LatentChem 的 +9.88pp 优势纯归因于 latent 接口本身，控制变量很干净。
- **vs Compressed reasoning ("contemplation tokens" / "soft capsules")**：这类工作压缩文本 CoT 到少量 latent token，但仍把 latent 当离散 token 用；LatentChem 走的是**完全连续的 hidden state 闭环 + 动态 perception refresh**，并验证了纯结果奖励即可触发模态切换。
- **启发**：把 Stage 4 的 outcome-only GRPO 用作"模态选择探针"是个普适方法论 —— 对任何想验证"X 模态比 Y 模态更适合任务 T"的研究，可以构造一个同时支持 X/Y 的接口，再用结果奖励看模型选哪个。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "spontaneous internalization + ChemUpdater 动态感知"在化学 LLM 上是首次系统验证，方法论意义大于工程价值。
- 实验充分度: ⭐⭐⭐⭐ 四个 benchmark + 三类 baseline + 4B/14B scale-matched + 消融 + causal ablation + 预算 stress test，覆盖很全；扣一星是因为没和最新通用 latent reasoning baseline（如 Geiping recursive unrolling）直接比。
- 写作质量: ⭐⭐⭐⭐⭐ 假说-接口-观察-验证的逻辑链非常清晰，图 1 的"jagged staircase vs. smooth manifold"概念图、Figure 6 的 hydraulic trade-off、Figure 7 的 t-SNE + RSA 都把抽象现象讲得直观。
- 价值: ⭐⭐⭐⭐ 对化学 LLM 是范式级贡献且加速 5.96×；扣一星是可解释性塌方限制了真实科研落地，落地价值要等 hybrid System 1/2 跟进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[ICML 2026\] Conformal Thinking: Risk Control for Reasoning on a Compute Budget](conformal_thinking_risk_control_for_reasoning_on_a_compute_budget.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)
- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ICML 2026\] TRACE: 用 Toulmin 论证模型评 LLM CoT 推理过程质量](trace_toulmin-based_reasoning_assessment_through_constructive_elements_for_llm_c.md)

</div>

<!-- RELATED:END -->
