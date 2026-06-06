---
title: >-
  [论文解读] An Information-Theoretic Criterion for Efficient Data Synthesis
description: >-
  [ICML2026][LLM推理][合成数据] 这篇论文用数据处理不等式解释合成数据为何有时有效、有时导致模型坍塌：只有当训练闭环持续引入稳定外部信号时，合成数据才是 information-open；而高 meta-level 的验证信号比实例级模仿更高效、更容易泛化。
tags:
  - "ICML2026"
  - "LLM推理"
  - "合成数据"
  - "信息论"
  - "数据处理不等式"
  - "外部验证器"
  - "奖励黑客"
---

# An Information-Theoretic Criterion for Efficient Data Synthesis

**会议**: ICML2026  
**arXiv**: [2605.16379](https://arxiv.org/abs/2605.16379)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 合成数据、信息论、数据处理不等式、外部验证器、奖励黑客  

## 一句话总结
这篇论文用数据处理不等式解释合成数据为何有时有效、有时导致模型坍塌：只有当训练闭环持续引入稳定外部信号时，合成数据才是 information-open；而高 meta-level 的验证信号比实例级模仿更高效、更容易泛化。

## 研究背景与动机
**领域现状**：大模型训练越来越依赖合成数据。真实高质量文本逐渐接近供给瓶颈，而数学、代码、工具使用和长程推理等任务又需要比普通网页语料更强的监督信号。成功案例包括 verifier-guided synthesis、RLVR、程序测试反馈、形式证明检查器和固定 rubric 评估器。

**现有痛点**：合成数据是一把双刃剑。用模型自己的输出反复自训练，常会造成分布收缩、长尾模式丢失和能力退化；但带 verifier 或环境反馈的合成管线又能带来巨大增益。现有经验很多，却缺少一个统一解释：什么时候合成数据注入了新信息，什么时候只是把模型已有分布循环一遍。

**核心矛盾**：模型生成的数据本身并不会凭空增加关于真实任务的信息。若训练样本完全来自当前模型，并且没有任何独立于当前模型状态的反馈，数据处理不等式意味着任务相关信息只会保持或下降。可是现实中 RLVR、代码测试、证明检查又确实能让模型进步，这说明成功管线并不是封闭系统。

**本文目标**：作者希望给出一个信息论准则，判断合成数据管线是否有效；进一步解释在有效管线之间，为什么某些外部信号能以很少样本产生跨域泛化，而另一些信号需要大量数据却收益有限。

**切入角度**：论文把训练过程写成随机变量关系：真实任务结构为 $X$，已有数据为 $D$，模型状态为 $Z$，外部信号为 $S$。若 $X\to D\to Z$ 或 $X\to Z_t\to D_t^{syn}\to Z_{t+1}$ 构成信息闭环，则 DPI 给出信息单调不增；若引入稳定的 $S$，上界变为 $I(X;D,S)$ 或 $I(X;Z_t,S)$。

**核心 idea**：合成数据是否有效不取决于“是不是模型生成”，而取决于训练循环是否 information-open，以及外部信号是否以高 meta-level 方式注入任务相关信息。

## 方法详解
论文不是提出新的训练算法，而是提出一套解释和设计准则。它先用 DPI 区分 information-closed 与 information-open 管线，再分析外部信号进入 SFT、RFT、RLVR 的不同方式，最后用 task-relevant partition 分解监督信号中的有效信息与类内噪声，从而解释样本效率、泛化、数据多样性和 reward hacking。

### 整体框架
在最基础的抽象中，训练过程把数据 $D$ 和内部随机性 $R$ 映射到模型状态 $Z=f_{train}(D,R)$。若训练程序没有关于任务结构 $X$ 的额外侧信息，则有 $X\to D\to Z$，数据处理不等式给出 $I(X;Z)\leq I(X;D)$。这意味着闭环自训练无法系统性增加模型关于真实任务的信息。

对迭代合成数据，若 $D_t^{syn}$ 完全由 $Z_t$ 生成，且没有 verifier、环境、固定 judge 或新数据，那么 $X\to Z_t\to D_t^{syn}\to Z_{t+1}$，从而 $I(X;Z_{t+1})\leq I(X;Z_t)$。实际退化来自采样误差、优化误差和分布收缩，使不等式常常严格。

当管线引入外部信号 $S$ 时，正确对象变成增强观察 $(D,S)$ 或 $(Z_t,S)$。此时 $I(X;Z)\leq I(X;D,S)=I(X;D)+I(X;S\mid D)$，真正新增的信息只来自条件互信息 $I(X;S\mid D)$。外部信号必须与任务结构相关，并且不能随着学生模型自由漂移，否则循环会重新闭合。

### 关键设计
1. **Information-open criterion**:

    - 功能：判断合成数据训练是否可能带来持续增益。
    - 核心思路：若外部信号 $S$ 满足 $I(X;S\mid D)>0$ 或迭代中 $I(X;S\mid Z_t)>0$，则它携带当前模型/数据之外的任务相关信息，合成数据管线是 information-open。若没有这样的 $S$，随机采样只能提供多样候选，不能提供选择方向。
    - 设计动机：这解释了闭环 self-training 的模型坍塌，也解释了为什么代码测试、证明检查、环境奖励和固定 rubric 可以让合成数据真正有用。

2. **Meta-level information injection**:

    - 功能：解释不同 information-open 管线的样本效率差异。
    - 核心思路：高 meta-level 信号只区分行为上重要的等价类，例如“答案是否正确”“程序是否通过测试”；低 meta-level 信号要求复现一个具体参考答案。若有 $M$ 个同样可接受的答案，高层信号把它们视为同类，而实例级 SFT 会浪费最多 $\log M$ bit 去识别某个表面形式。
    - 设计动机：这解释了为什么 RLVR 的二元 correctness 可以跨数学、代码、逻辑泛化，而单参考答案模仿常把容量花在无关的表面细节上。

3. **监督信息分解与 reward hacking 机制**:

    - 功能：把“有效任务信号”和“类内/伪相关信号”分开，解释泛化和奖励黑客。
    - 核心思路：给定 task-relevant partition $\pi$，任意监督信号满足 $I(Y;S\mid Q)=I([Y]_\pi;S\mid Q)+I(Y;S\mid [Y]_\pi,Q)$。前者是任务相关增益，后者是类内增益。若信号只依赖 $[Y]_\pi$，效率 $\eta_\pi=1$；若一个更粗、更易学的伪信号与奖励相关，模型会优先学习它。
    - 设计动机：这把 reward hacking 从“模型偷懒”改写成“模型学习了训练信号中信息效率最高的成分”。解决办法不是训练更久，而是去相关伪特征，或让意图信号成为最高效信号。

### 损失函数 / 训练策略
论文比较了三种训练信号进入方式。SFT 最大化固定数据对的 log-likelihood，主要提升某个参考答案概率；RFT 由当前模型生成候选，再用外部 acceptance test 过滤，通过被接受样本做 SFT；RLVR 则把 verifier reward 直接放进 policy gradient，用 advantage 加权模型生成输出。作者强调，三者的关键差异不是数据是否合成，而是外部信号如何进入梯度与样本选择。

## 实验关键数据

### 主实验
本文主要是理论与案例分析论文，没有传统 benchmark 表。主结果可以概括为以下准则与案例证据。

| 场景 | 指标 | 本文判据 / 观察 | 之前常见理解 | 提升 |
|------|------|------------------|--------------|------|
| 闭环 self-training | $I(X;Z_{t+1})\leq I(X;Z_t)$ | 无外部信号时任务信息不能增加，坍塌是预期结果 | 把坍塌视为经验性训练不稳定 | 给出 DPI 级别解释 |
| 带 verifier / environment 的合成数据 | $I(X;S\mid Z_t)>0$ | 外部信号提供新增任务信息，循环保持 information-open | 只说“过滤提升数据质量” | 明确新增信息来源 |
| SFT vs RLVR | meta-level | SFT 学具体参考，RLVR 学正确性等价类 | 只比较算法形式 | 解释 RLVR 样本效率和跨域泛化 |
| JudgeRLVR 案例 | 二元 correctness signal | 仅用正确/错误训练的 judge 可跨 math/code/logic 泛化 | 需要 domain-specific rubric | 高 meta-level 信号减少无关差异 |
| 奖励黑客案例 | 训练 reward 上升但真实正确率停滞 | 模型学到长度/风格这个更粗伪信号 | reward hacking 是模型“钻空子” | 信息效率视角给出可诊断机制 |

### 消融实验
论文中的分析性消融主要围绕“是否外部”“信号粒度”和“伪相关是否被去除”。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无外部信号的自训练 | 信息闭环，$I(X;S\mid Z_t)=0$ | 只能探索已有分布，无法判断哪些样本更贴近任务 |
| 固定 evaluator / verifier | 信息开放，$I(X;S\mid Z_t)>0$ | evaluator 不随学生漂移，能持续提供选择压力 |
| 高 meta-level 二元信号 | $\eta_\pi=1$ | 只区分正确/错误或满足/不满足，把类内表面差异全部忽略 |
| 实例级参考答案 | 需要识别类内具体元素 | 当可接受答案很多时，监督容量被消耗在无关表面形式上 |
| 长度/风格与正确性相关 | reward 上升、真实正确率停滞 | 伪信号比正确性更粗更易学，模型优先收敛到伪分区；重平衡数据源后问题消失 |

### 关键发现
- 合成数据的核心瓶颈是 verification capacity。数学、代码、形式推理进展快，是因为这些领域更容易提供稳定、高 meta-level 的外部信号。
- 随机性和外部信号缺一不可。随机采样提供候选多样性，外部信号提供筛选方向；只有采样没有信号会坍塌，只有信号没有候选也无法探索。
- 多样性优于重复数据的原因可以用 partition coverage 解释：同一 prompt-output 对反复出现几乎不提供新任务信息，而覆盖新分区块的样本会提供 fresh bits。

## 亮点与洞察
- 论文把“合成数据是否有效”从经验配方提升为信息边界问题。这个视角很实用，因为它迫使我们问：管线里到底哪个变量携带了当前模型之外的任务信息。
- meta-level 的表述很有解释力。许多看似不同的成功案例，如代码单元测试、Lean proof checker、格式奖励、SAT solver evaluator，本质都是把大量表面不同的输出压缩到少数行为等价类。
- 对 reward hacking 的解释比常见说法更可操作：如果伪特征比真实目标更高效，模型学习伪特征就是理性的优化结果。因此数据构造时必须主动打散长度、风格、来源模型等粗粒度伪相关。
- 论文也给了训练设计启发：与其追求更多合成样本，不如先投资稳定 verifier、固定 judge、可执行环境和多样 prompt coverage。

## 局限与展望
- 框架主要是定性解释，不能直接预测某个合成数据管线需要多少样本、多少 verifier 精度或多少轮训练才能生效。
- “模型优先收敛到信息效率最高信号”的 thesis 依赖梯度学习的 simplicity bias 和案例支持，并不是由互信息分解本身严格推出。
- 外部信号被当作给定变量，但现实中 verifier 会有误差、覆盖盲区和可被优化攻击的问题；如何设计随模型能力提升仍可靠的 verifier 是开放问题。
- 对开放式生成、安全偏好和创造性任务，高质量高 meta-level 信号很难构造，可能需要人类标注、固定 rubric、模型 judge 与行为测试的混合方案。

## 相关工作与启发
- **vs 模型坍塌研究**: 既有工作观察到 self-consuming generative models 会退化，本文用 DPI 把它解释为 information-closed loop 的必然趋势。
- **vs RLVR / verifiable reward**: RLVR 的成功不只是强化学习技巧，而是 verifier 作为外部信号直接进入梯度，使训练循环保持 information-open。
- **vs SFT 合成数据**: SFT 可以有效蒸馏固定数据，但当多个答案都可接受时，它会把监督容量花在复现具体参考上；本文指出这正是低 meta-level 的低效来源。
- **vs Sutton Bitter Lesson**: 论文把“让计算自己发现结构”解释为用 meta-level 约束而非实例级知识硬编码：信号只规定什么算对，不规定必须长什么样。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 information-open 和 meta-level injection 统一解释合成数据、RLVR、坍塌和 reward hacking，视角很强。
- 实验充分度: ⭐⭐⭐⭐☆ 案例和理论分析充分，但缺少统一可复现实验 benchmark 与定量预测。
- 写作质量: ⭐⭐⭐⭐⭐ 概念链条清楚，从 DPI 到监督信息分解再到实际案例，论证连贯。
- 价值: ⭐⭐⭐⭐⭐ 对设计 LLM 合成数据管线、verifier 和奖励模型都有很高参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](../../ACL2026/llm_reasoning/self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](../../ACL2026/llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](../../ACL2026/llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)

</div>

<!-- RELATED:END -->
