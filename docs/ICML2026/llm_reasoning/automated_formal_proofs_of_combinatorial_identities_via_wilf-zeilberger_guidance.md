---
title: >-
  [论文解读] Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs
description: >-
  [ICML 2026][LLM推理][Lean 4] WZ-LLM 把经典的 Wilf–Zeilberger 符号证明流程编译成 Lean 4 中可执行的证明骨架（递推 + 边界条件 + 侧条件）…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "Lean 4"
  - "组合恒等式"
  - "Wilf-Zeilberger"
  - "神经符号"
  - "DAPO"
---

# Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs

**会议**: ICML 2026  
**arXiv**: [2605.04472](https://arxiv.org/abs/2605.04472)  
**代码**: 暂未公开  
**领域**: LLM 推理 / 自动定理证明 / 神经-符号  
**关键词**: Lean 4, 组合恒等式, Wilf-Zeilberger, 神经符号, DAPO

## 一句话总结
WZ-LLM 把经典的 Wilf–Zeilberger 符号证明流程编译成 Lean 4 中可执行的证明骨架（递推 + 边界条件 + 侧条件），交给专门用 SFT + expert-iteration + DAPO 训练出的 WZ-Prover 逐项 discharge，在 100 个经典组合恒等式上把 pass@32 从 Goedel-Prover-V2 的 9% 提升到 34%。

## 研究背景与动机

**领域现状**：基于 LLM 的自动定理证明（ATP）在 Lean / Isabelle 等交互式证明助手上已经能达到比赛级表现（DeepSeek-Prover-V2、Kimina、Goedel-Prover-V2 等），但组合学被普遍认为是 ATP 最难的领域之一，其中"组合恒等式"是一类基础且无处不在的命题。

**现有痛点**：1) 证明组合恒等式需要长程规划——没有全局路线图时 LLM 会陷入无限制搜索，组合爆炸；2) Lean 中组合学的训练数据极稀缺；3) 纯符号方法（WZ、creative telescoping）在 CAS 里效率很高，但输出无法直接翻译进证明助手——需要重新构造 telescoping 论证、边界条件、归一化步骤、各种非零侧条件，"形式化成本"反而压过原始证明成本；4) 现有 whole-proof LLM 缺乏中间 verifier 信号，逐 tactic 模型又分支爆炸。

**核心矛盾**：长程证明需要明确规划，而 LLM 缺规划；符号方法天生有规划，但产物不可形式化。两条路线各擅其长却又彼此不通。

**本文目标**：把 WZ 的"规划能力"和 LLM 的"形式化能力"焊接起来，让一类原本符号方法搞不定 + LLM 也搞不定的恒等式被两路同时覆盖。

**切入角度**：作者注意到 WZ 方法本身就提供了一个天然的"sketch"——构造 WZ 对 $G(n,k)=R(n,k)F(n,k)$ 后，恒等式自动分解为「递推引理 + 边界条件 + 侧条件 + 归一化 + case-split」一组可机器验证的 sub-goal。这恰好是 Lean 4 喜欢的结构，把它当成 LLM 的中间脚手架既减少搜索空间又给出 verifier 信号。

**核心 idea**：用 **「WZ 符号分解（外部 CAS 出 sketch）+ 专门训练的 WZ-Prover（discharge sketch 子目标 + 兜底 WZ-不覆盖恒等式）」** 的双路径神经-符号系统替代纯 LLM 或纯符号方法。

## 方法详解

### 整体框架
WZ-LLM 由两段组成：(A) **Symbolic Decomposition**——给定 Lean 4 形式化的组合恒等式，先做归一化，再调用 SageMath 的 WZ 算法尝试合成证书；若成功就把恒等式拆成 $\mathcal{T}=\mathcal{T}_{\text{rec}}\cup\mathcal{T}_{\text{bd}}\cup\mathcal{T}_{\text{side}}\cup\mathcal{T}_{\text{norm}}\cup\mathcal{T}_{\text{case}}$ 这一组结构化 Lean 子目标；若 WZ 不适用，整道题进入"直接证明池"。(B) **WZ-Prover**——一个从 Goedel-Prover-V2 起步、经过三阶段训练的专用 Lean 4 prover，负责把池子里所有任务（sketch 子目标 + WZ-uncovered 整题）证完。Lean kernel 在最后一锤定音：只有内核接受，整条证明才算成功。

### 关键设计

1. **WZ 符号分解器（Symbolic Decomposition）**:

    - 功能：把 Lean 4 的一条恒等式 $S(n)=\sum_k F(n,k)=C$ 编译成一组机器可验证的 Lean 证明义务。
    - 核心思路：① 归一化——把 `Icc/Ico` 转 `Finset.range`，shift 索引从 0 起，简化阶乘/二项/幂的语法变体；遇到 parity 之类的 piecewise 谓词则结构化 case-split；② sketch 构造——用 SageMath 的 `F.WZ_certificate(n,k)` 合成有理函数 $R(n,k)$ 使 $G(n,k)=R(n,k)F(n,k)$ 满足 WZ 方程 $F(n+1,k)-F(n,k)=G(n,k+1)-G(n,k)$，从而把原恒等式归约到「递推 lemma + 边界 obligation」；③ 侧条件推断——用符号化简找出会让 `field_simp` 等 tactic 卡住的零分母、负阶乘参数，自动生成 `∀n,k, A(n,k)≠0` 这种 non-vanishing lemma 与边界子目标。
    - 设计动机：实践证明，Lean 形式化的真正难点恰恰在边界与侧条件——CAS 给的"证书"在数学上正确却没法机械地塞进证明助手。把这些隐含义务全部 explicit 化、变成可被 LLM 单独 discharge 的小目标，是让 sketch 真正可执行的关键。

2. **专家在环 bootstrapping（Expert-in-the-loop Iteration）**:

    - 功能：从 307 道手工形式化恒等式（提供完整 Lean 证明）出发，迭代扩大训练集而不引入 hallucination。
    - 核心思路：第一阶段用 307 道恒等式 + 它们经 sketch 拆出的 1200 个 sub-goal 做 cold-start SFT；第二阶段对 1020 道无标注候选恒等式跑 WZ-LLM 两路尝试，**Lean kernel 严格验证**通过的才进入下一轮训练池——Round 1 拿到 5139 lemma 证明 + 32 整题，Round 2 再得 532 + 79，共增 5671 lemma + 111 整题；deduplicate 后形成约 5418 样本的扩展 SFT 语料。
    - 设计动机：组合学 Lean 数据极稀缺，纯靠人工无法 scale；但 LLM 自生成又有 hallucination 风险。"verifier-filtered bootstrapping"通过 kernel 这道关把噪声样本天然过滤掉，等价于"免费"的高保真数据增广，同时保证训练分布不被自身错误污染。

3. **DAPO with Difficulty-Smoothing 精调**:

    - 功能：在 SFT 之后用 RL 进一步提升 long-CoT 鲁棒性，重点改善硬题与长链 lemma 上的表现。
    - 核心思路：① 难度平滑——SFT 语料里 sketch lemma 多数短而重复、整题长而稀少；先用 rollout 估计每题当前策略下的 pass-rate，把极易（去重）和近零通过率（噪声梯度）两端裁掉，得到中-难分布平滑的 RL 集；② DAPO 优化——奖励 $R(\pi;G)=R_{\text{out}}(\pi;G)+\lambda_{\text{len}}R_{\text{len}}(\pi)$，其中 $R_{\text{out}}\in\{+1,-1\}$ 是 Lean kernel 验证信号，$R_{\text{len}}$ 是接近 token 预算时的渐进惩罚，避免过长被截断导致虚假负奖励；DAPO 本身的动态采样机制缓解 entropy collapse。
    - 设计动机：在二值且稀疏的 kernel 奖励下，naive RL 极易在易题上过拟合或在难题上崩塌；难度平滑 + DAPO 的组合让 RL 把算力专门花在"非琐碎但有救"的题目上。

### 损失函数 / 训练策略
三阶段 pipeline：(i) SFT on 307 seed + 1200 lemmas；(ii) expert-iteration 扩到 ~5418 验证样本；(iii) DAPO RL with rule-based outcome reward + soft overlong punishment。整套训练 16 GPU-days、推理评测 9 GPU-days，全部在 4× L40s-48GB 上完成。模型规模仅 8B。

## 实验关键数据

### 主实验
LCI-Test（100 道经典组合恒等式，Lean 4 形式化）pass@32 端到端证明成功率：

| 方法 | 模型 | LCI-Test pass@32 |
|------|------|------|
| DeepSeek-V3 | 685B | 1/100 |
| Gemini-3.1-Pro-Preview | — | 16/100 |
| Kimina-Prover-Distill | 7B | 6/100 |
| DeepSeek-Prover-V2 | 7B | 6/100 |
| Goedel-Prover-V2 (baseline) | 8B | 9/100 |
| WZ-Sketch + Goedel-Prover-V2 | 8B | 9/100 |
| WZ-Prover（only direct） | 8B | 12/100 |
| WZ-Sketch + WZ-Prover | 8B | 29/100 |
| **WZ-LLM（两路合并）** | **8B** | **34/100** |

跨数据集泛化：CombiBench 上 12→16/100，PutnamBench-Comb 上 0→3/36，均高于 baselines。

### 消融实验

| 训练阶段 | pass@1 | pass@8 | pass@32 |
|----------|--------|--------|---------|
| SFT (seed only) | 1/100 | 3/100 | 9/100 |
| + expert-iteration | 3/100 | 6/100 | 10/100 |
| + DAPO refinement | 4/100 | 6/100 | 12/100 |

Lemma 级诊断（sketch 拆出的 1178 个子目标）：

| 模型 | #Proved / 1178 | Acc | 端到端 #Solved / 46 |
|------|-----|-----|------|
| Goedel-Prover-V2 | 564 | 47.88% | 0 |
| WZ-Prover | 864 | 73.34% | 29 |

### 关键发现
- **Sketch alone 不够**：在没有专门训练的 Goedel-V2 上叠 sketch 反而无收益（9→9）；因为整道题需要把**所有** sketch lemma 全部 discharge，47.88% 的 lemma acc 直接导致 0 整题闭环。提到 73.34% 才解锁 29 题，说明"专 prover + 专 sketch"必须同时具备。
- **Direct + sketch 互补**：5 道 WZ 不适用的硬题被 WZ-Prover 直接证下来（symbolic-only 永远做不到），29 道 WZ-适用的题被 sketch 路径接力完成，二者合并到 34 题。
- **DAPO 的收益集中在 pass@32**：pass@1 仅 +1，pass@32 +2，说明 RL 主要让"长尾难题"在更大采样预算下被捕到，而非在易题上多挤几分。

## 亮点与洞察
- 把经典符号方法当成"可执行 sketch generator"是非常清爽的复合方式：既绕开了 LLM 长程规划弱、又绕开了 CAS 输出不能直接进证明助手的痼疾，把双方的非交集变成可加和。
- "verifier-filtered bootstrapping"在 Lean 这种 kernel-checked 环境里几乎是无脑能用的数据增广：训练池由原模型自己生、verifier 当过滤器，理论上能持续 scale 直到接近能力天花板。
- DAPO + 难度平滑的组合给出了 sparse binary reward 场景下一个可复用的菜谱：先用 rollout 把题库按当前策略难度分桶，再裁掉两端噪声，再 RL；不依赖人工分级。

## 局限与展望
- 8B 模型 + 16 GPU-days 训练对学界友好，但 LCI-Test 上还有 66 题没解决，说明长程组合证明的能力天花板仍远未触及，特别是 PutnamBench-Comb 上只解了 3/36。
- 整套流水线对 Lean 4 mathlib 的 API 演化敏感，sketch 部分高度耦合到当前 Finset/Nat.factorial 接口；若 mathlib 重构需要重新对齐 normalization 规则。
- WZ 方法只覆盖超几何/holonomic 类恒等式，对真正"非超几何"组合恒等式（如 q-级数、对合论证）需要寻找新的符号 sketch 引擎。

## 相关工作与启发
- **vs Goedel-Prover-V2 / DeepSeek-Prover-V2 等 whole-proof LLM**：它们靠端到端生成，没有显式规划机制；WZ-LLM 通过外部 CAS 提供 sketch，把"长程规划"外包给已经成熟几十年的符号算法。
- **vs InternLM-2.5-StepProver / MA-LoT 这类 tactic-level + search**：它们靠 BFS/MCTS 在 tactic 空间里搜，分支爆炸；WZ-LLM 不在 tactic 维度做搜索，而是在更高层的 sketch 维度做规划，再用 whole-proof prover 处理每个子目标。
- **vs Harrison 在 HOL Light 上证 hypergeometric sums**：思路同源（CAS 出证书 + 形式化），但 Harrison 全手工嵌入；WZ-LLM 把"形式化"这一最耗人力的步骤交给 LLM-Prover，是这一思路的现代化升级。

## 评分
- 新颖性: ⭐⭐⭐⭐ "把符号方法 sketch 编译成 LLM 可证明的 Lean 子目标"这个 framing 在 ATP 圈很清新
- 实验充分度: ⭐⭐⭐⭐ 三个 benchmark、组件 + 训练阶段双重消融、lemma 级诊断都做到位
- 写作质量: ⭐⭐⭐⭐ 神经符号架构与训练流水线讲得清晰，WZ 数学背景前置完整
- 价值: ⭐⭐⭐⭐ 给 Lean 数学形式化提供一条"符号引导 + LLM discharge"的可复用 recipe，可类推到其它有 CAS 的领域（积分、求和、ODE 等）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving](../../ACL2025/llm_reasoning/local_look-ahead_guidance_via_verifier-in-the-loop_for_automated_theorem_proving.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ICML 2025\] ProofCompass: Enhancing Specialized Provers with LLM Guidance](../../ICML2025/llm_reasoning/proofcompass_enhancing_specialized_provers_with_llm_guidance.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](../../ACL2025/llm_reasoning/safe_math_reasoning.md)

</div>

<!-- RELATED:END -->
