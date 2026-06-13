---
title: >-
  [论文解读] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training
description: >-
  [ICML 2026][强化学习][课程学习] 本文为「先易后难」的课程式 RL 后训练给出第一份严格的样本复杂度证明：在 transformer 的状态条件自回归推理树上，若课程能让相邻阶段的难度比保持在目标难度的 $L/p$ 次根级别，则总样本数可从直接训练的指数级 $(C^\star)^L$ 降到课程版的多项式级 $L\cdot (C^\star)^{p_\max}$。
tags:
  - "ICML 2026"
  - "强化学习"
  - "课程学习"
  - "CoT 后训练"
  - "样本复杂度"
  - "覆盖系数"
  - "拒绝采样"
---

# Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training

**会议**: ICML 2026  
**arXiv**: [2511.07372](https://arxiv.org/abs/2511.07372)  
**代码**: https://github.com/DakeBU/Curriculum-Post-training (有)  
**领域**: LLM 推理 / 强化学习 / 学习理论  
**关键词**: 课程学习, CoT 后训练, 样本复杂度, 覆盖系数, 拒绝采样

## 一句话总结
本文为「先易后难」的课程式 RL 后训练给出第一份严格的样本复杂度证明：在 transformer 的状态条件自回归推理树上，若课程能让相邻阶段的难度比保持在目标难度的 $L/p$ 次根级别，则总样本数可从直接训练的指数级 $(C^\star)^L$ 降到课程版的多项式级 $L\cdot (C^\star)^{p_\max}$。

## 研究背景与动机
**领域现状**：CoT 推理的后训练目前主流是「直接 RL fine-tune + 0/1 outcome verifier」，配合 test-time scaling（beam / best-of-N）来提升 pass@K。近一年大量经验工作（Parashar 2025、Liu 2025、Bae 2025 等）发现「easy-to-hard 课程」能显著加快收敛，但只有实验观察，缺乏可证明的解释。

**现有痛点**：经典课程学习理论几乎全建立在「从头训练」场景（凸回归、parity、teacher-student perceptron 等），其「难度」「性能」定义都是任务专用、几何性的，根本无法迁移到「从一个强 pretrained 模型出发、目标是 CoT 泛化」的 LLM 后训练情境。后训练的核心特征是奖励稀疏、正确轨迹在 base policy 下采样概率极低，这套语言在旧理论中根本没有对应物。

**核心矛盾**：稀疏奖励 RL 的难度本质来自「正确 CoT 在 base policy 下的稀有度」，可由 coverage coefficient $\|\pi^\star/\pi_{\text{ref}}\|_\infty$ 刻画；直接训练要付出与该稀有度成多项式（甚至指数）关系的采样代价，而课程允许把这条「稀有度阶梯」拆成若干小台阶 —— 但拆成什么样的台阶才真正能带来超多项式加速，此前并无明确条件。

**本文目标**：(i) 给出课程后训练带来指数级样本节省的充要 / 充分条件；(ii) 在 transformer + 推理树这一具体可分析模型上证明该条件自然成立；(iii) 把结论分别落到 RL fine-tune 和 test-time scaling 两套范式上。

**切入角度**：作者把后训练视为在「pre-trained 推理树」上重新分配概率，并采用 Foster 2025 的 spanner-sampling / coverage 框架，把课程难度直接挂钩到 rejection sampling 的尝试次数 $\Theta(\|\pi^\star/\pi_{\text{ref}}\|_\infty \log\delta^{-1})$，从而把「难度」「学习成本」统一在同一把尺子上。

**核心 idea**：把 base model 建模成在状态条件自回归推理树（2S-ART）上对合法子节点近似均匀采样的 PART，并证明在此 PART 下「prefix-hint 课程」与「depth-increasing 课程」天然形成 $K$ 次根难度阶梯，让总成本从 $(C^\star)^L$ 降到 $L\cdot(C^\star)^{p_\max}$。

## 方法详解

### 整体框架
全文是一套理论框架，没有新算法实现，而是把已有 curriculum post-training pipeline 写进一个可证明的抽象中：(1) 把任务建模成 2S-ART 推理树 $F_{\text{2S-ART}}(\{\Phi_\ell\},\{I_\ell\})$，每一步从合法索引集 $I_\ell$ 选 token 并用 $\Phi_\ell$ 更新状态 $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$；(2) 把 base model 实例化为 PART，即各深度对合法子节点均匀采样；(3) 用一个标准 sampling-attention transformer（FFN 实现 $\Phi_\ell$ 原语，attention 负责选索引）证明其能精确复现 PART；(4) 在该模型上分别分析 RL fine-tune 的 sample complexity（Thm 2）与 test-time scaling 的 oracle-query / 计算复杂度（Thm 3）。

### 关键设计

**1. 基于覆盖系数的课程充要 / 充分条件（Cor. 1）：把"先易后难有没有用"变成一句不等式**

业界一堆经验都说"先易后难有用"，但没人能回答"难度阶梯怎么搭才真有用"。本文先把"难度"统一到一把尺子上：定义 $\varepsilon$-精度样本复杂度 $N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$，再用 rejection-sampling 引理 $N\propto\|\pi^\star/\pi_{\text{ref}}\|_\infty$ 把它和 coverage coefficient 绑死——稀疏奖励 RL 的难度本质就是"正确 CoT 在 base policy 下有多稀有"。在这把尺子下，课程比直接训练更省样本的充要条件就是一句话：$\sum_{\ell}N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})<N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$。更进一步，当存在 $L/p$-次根课程、即每相邻阶段难度比 $N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})=\Theta(\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})})$ 时，直接版与课程版的样本比 $N^{\text{direct}}/N^{\text{curr}}=\Theta((C^\star)^L/(L\cdot C^\star))$（$C^\star=\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})}>1$），就是指数级加速。这个条件不只解释了 hint-decreasing、depth-increasing 等不同课程的共性，还直接给出可操作的设计原则：相邻阶段难度比要控在 $C^\star$ 量级。

**2. 2S-ART 推理树 + PART 基模型（Def. 1-2）：给"reasoning 任务 + 弱基模型"一个能算的抽象**

Cor. 1 的条件还停在抽象层，得有一个具体模型让它在 transformer 上自动满足。本文把长度-$k$ 的推理任务表示成索引路径 $S_k=(i_1,\dots,i_k,d{+}1)$，每步从合法集 $I_\ell(\text{CoT}_{\ell-1})$（$|I_\ell|=\Theta(d)$）取 $i_\ell$、状态更新 $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$，这就是 2S-ART 树。base model 则实例化为 PART——在每个父节点对合法子节点均匀采样。均匀这一步是关键：它直接给出 $\|\pi^\star_{S_{\ell+1}}/\pi^\star_{S_\ell}\|_\infty=\Theta(d)$，进而 $\|\pi^\star_{S_\ell}/\pi^\text{PART}\|_\infty=\Theta(d^{\ell+1})$，与目标难度的 $L/p$ 次根关系天然成立，Cor. 1 的抽象条件就在这个模型上落地了。之所以选均匀-PART，是因为它既能装下 parity、Countdown、Markov-chain reasoning、induction-head 等多种任务（都是其子类），又能让"子节点概率比"可控，还和"后训练 = 在 pre-trained tree 上 reweight"的视角一致。作者还用一个标准 sampling-attention transformer（FFN 实现 $\Phi_\ell$ 原语、attention 负责选索引）证明它能精确复现 PART。

**3. RL fine-tune 与 test-time scaling 的指数→多项式约简（Thm 2-3）：把 Cor. 1 翻译成两类具体范式的复杂度**

实践里 RL fine-tune 和 test-time scaling 都靠"采样 + 验证"驱动，本质都受 coverage 控制，所以能统一在同一框架里。本文用 0/1 outcome 奖励 $R^{f_{S^\star}}_x$ 和分段课程奖励 $R^{F_{S^\star}}_x(\cdot,\ell)$（每阶段只奖励 prefix 正确的 pre-EOS token），证明 RL fine-tune 在课程下样本复杂度从直接版的 $\widetilde O(d^{L+1})$ 降到 $\widetilde O(L\cdot d^{p_\max+1})$，test-time scaling 的 oracle（best-of-$N$ / verifier query）复杂度也有同形式的指数→多项式约简；线性可实现 MDP 上还借 Foster 2025 的 spanner-sampling 给出 inference-time 计算复杂度的类比（Thm 4）。一个要处理的麻烦是 outcome reward 存在 reward hacking——比如 parity 里选错索引仍有 50% 命中——作者在 App. F-G 用 prefix 课程把奖励信号下推到中间 EOS，把 hacking 概率指数压低。这一统一框架也正好解释了为什么"先用短 CoT / hint 训练"能对训练省 sample、对推理同样省 verifier query。

### 损失函数 / 训练策略
理论文章无具体损失。证明里使用 outcome-only reward $R^{f_{S^\star}}_x\in\{0,1\}$（只检查 pre-EOS token 是否匹配 $\mu_{f_{S^\star}(x)}$）；课程版加上深度参数 $R^{F_{S^\star}}_x(\cdot,\ell)$。难点是 outcome reward 存在 reward hacking（如 parity 中选错索引仍 50% 命中），作者在 App. F-G 用 prefix 课程把 hacking 概率指数压低。

## 实验关键数据

### 主实验
论文以理论为主，实验仅做「parity / countdown」上的小规模模拟以验证 sample complexity 比值的预测。

| 任务 | $d$ | 直接 RL 收敛步数 | 课程 RL 收敛步数 | 加速比 |
|------|-----|------------------|------------------|--------|
| Sparse Parity | 8 | $\sim d^L$ 量级，>$10^5$ | $\sim L\cdot d^{p_\max}$，约 $10^3$ | $\sim 50\times$ |
| Countdown-24 | 4 nums | 直接训练几乎不收敛 | 课程版稳定收敛 | 质变 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整 hint-decreasing 课程 | 多项式样本即可学到 $\pi^\star$ | 与 Thm 2 预测一致 |
| 去掉中间阶段（一次性跳到目标） | 样本复杂度爆炸 | 印证 Cor. 1 的必要性 |
| 阶段过细（$L$ 阶段拆成 $2L$） | 增益递减但稳定 | 与 $L\cdot(C^\star)^{p_\max}$ 中的线性 $L$ 因子吻合 |

### 关键发现
- 加速比强依赖于「难度比 $C^\star$」而非阶段数 $L$：$L$ 太大反而把线性因子放大，存在最优阶段数。
- PART 的「均匀子节点」假设是关键：若 base model 在某些子节点上概率严重不均，$K$-次根关系不再成立，加速会被局部瓶颈吃掉。
- outcome-only reward 在 parity 上确实出现 hacking（错索引但 final bit 对），prefix 课程通过把奖励信号下推到中间 EOS 显著缓解。
- 课程加速对 test-time scaling（best-of-$N$ 之类的 verifier query）与 RL fine-tune 同时生效，二者复杂度阶在 Cor. 1 框架下完全平行。
- 在 countdown 这种「子节点分支因子可控、$\Phi_\ell$ 是算术 4 则运算」的任务上，理论预测的多项式样本量与模拟收敛步数吻合度最高。

## 亮点与洞察
- 把「coverage coefficient」这把出自 offline RL 理论的尺子搬到 CoT 后训练，让难度、样本数、推理深度三者用同一种语言度量，这是本文最重要的概念性贡献。
- 「$L/p$ 次根条件」是少见的可操作设计准则：它告诉工程师「相邻阶段成功率差距不要超过目标任务成功率的 $L/p$ 次根」，可以直接指导课程切分。
- 把 RL fine-tune 与 test-time scaling 写在同一个 oracle-query 框架内，给「训练省 sample = 推理省 verifier query」提供了第一份对称证明。
- 2S-ART 的抽象能同时涵盖 parity、Countdown、Markov-chain、induction-head、因果图推理等多个以往被分开分析的任务类，抽象能力本身是价值输出。
- prefix-hint 与 depth-increasing 两种看似不同的课程体裁，被统一在「父节点上子节点概率均勻」这一准则下，提供了课程设计的「微观充分条件」。

## 局限与展望
- 假设 base model 是均匀-PART，真实 LLM 的 token 概率远非均匀，且 attention 已经预热到某些路径，理论与实际的 gap 仍大。
- 仅在 outcome-only 0/1 奖励下证明，对 process reward / 中间步骤奖励的扩展尚未覆盖，而后者是 OpenR / Math-Shepherd 等方向的主战场。
- 实验规模小（toy parity、toy countdown）；在真实 LLM（DeepSeek-R1 类）上验证 $L/p$-次根条件需要更复杂的难度度量手段。
- 假设可以自由设计 prefix-hint 课程，实际项目里「哪段 hint 算 1/L 阶段难度」往往需要试错；论文未给出工程上估计每阶段 coverage 的可行手段。
- 多任务、多语言混合训练场景下不同任务的难度比 $C^\star$ 不一致，理论里默认全局共享一个 $L/p$ 关系，扩展到异质任务仍需新工具。

## 相关工作与启发
- **vs Parashar et al. 2025**：他们提出 approximate policy iteration 误差累积式课程，本文 Cor. 1 的假设是其严格化版本，并把误差转写成 coverage 语言。
- **vs Liu et al. 2025b (hint-decreasing)**：Liu 用 prefix hint 实现易→难，本文把 hint 长度变化解读为推理树上的 prefix-prefix 关系，给其经验有效性提供理论解释。
- **vs Foster et al. 2025 (spanner sampling)**：本文复用其 coverage 框架并把对象从 linear MDP 推广到 transformer reasoning tree，给 inference-time 计算复杂度提供 Thm 4 的类比结果。
- **vs Ran-Milo et al. 2026 (graph traversal)**：他们对图遍历任务的分析可视为 Cor. 1 的一个具体实例 —— 给短-CoT 实例分配非零概率质量等价于满足 $K$ 次根条件。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为 curriculum post-training 给出严格的指数→多项式 sample-complexity 证明
- 实验充分度: ⭐⭐ 仅 toy 任务模拟，没有真实 LLM 端到端验证
- 写作质量: ⭐⭐⭐⭐ 抽象框架 + 具体实例（parity / countdown）结合得很清晰，符号体系略重
- 价值: ⭐⭐⭐⭐ 为业界经验性「先易后难」curriculum 提供了可量化设计准则，对 RLHF / R1 风格训练有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[ICLR 2026\] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](../../ICLR2026/reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)

</div>

<!-- RELATED:END -->
