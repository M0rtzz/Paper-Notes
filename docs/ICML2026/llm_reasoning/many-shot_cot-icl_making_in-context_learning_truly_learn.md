---
title: >-
  [论文解读] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn
description: >-
  [ICML 2026][LLM推理][many-shot ICL] 本文系统揭示了非推理任务的 many-shot ICL “经验法则”在 CoT 推理任务上**全部失效**——相似度检索反而有害、顺序敏感性随 shot 数增长——并把成功的 many-shot CoT 重新解读为“in-context 测试…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "many-shot ICL"
  - "chain-of-thought"
  - "in-context test-time learning"
  - "demonstration ordering"
  - "曲率正则"
---

# Many-Shot CoT-ICL: Making In-Context Learning Truly Learn

**会议**: ICML 2026  
**arXiv**: [2605.13511](https://arxiv.org/abs/2605.13511)  
**代码**: 无  
**领域**: 大模型推理 / 上下文学习 / Chain-of-Thought  
**关键词**: many-shot ICL, chain-of-thought, in-context test-time learning, demonstration ordering, 曲率正则

## 一句话总结
本文系统揭示了非推理任务的 many-shot ICL “经验法则”在 CoT 推理任务上**全部失效**——相似度检索反而有害、顺序敏感性随 shot 数增长——并把成功的 many-shot CoT 重新解读为“in-context 测试时学习”，由此提出按 embedding 轨迹曲率排序 demonstration 的 CDS 方法，在 64-shot 几何题上提升 5.42 pp。

## 研究背景与动机

**领域现状**：长上下文 LLM 让 many-shot ICL 成为可能，已有工作（Bertsch et al., Baek et al.）在非推理任务（分类、简单 QA）上观察到三条规律：(1) shot 数变大性能稳定上升；(2) demonstration 顺序敏感性随 shot 数下降；(3) 相似度检索（top-k 最相似）能提升性能。同时 chain-of-thought (CoT) 已成复杂推理标配，但 CoT-ICL 大都在 few-shot 设定研究。

**现有痛点**：当 CoT 与 many-shot 结合（即 many-shot CoT-ICL），三条经验规律是否还成立？这件事完全没系统研究过。如果规律仍然 hold，那继续按检索/堆 shot 的工程套路即可；如果规律破坏了，整个 prompt 工程范式都要重新思考。这不仅是工程问题，更牵涉 ICL 本质是“规模化模式匹配”还是“真正学习”的根本争论。

**核心矛盾**：CoT demonstration 长度（geometry 任务下单条 CoT 比 BANKING77 长约 30×）、内部包含 procedural 推理链、对 model 提出更高 understanding 要求；这些性质让传统“多即是好、检索即对”的 many-shot 直觉在 CoT 场景未必成立。如果 ICL 真在“学习”，那 demonstration 是 supervision、顺序就是 curriculum，类似教学需要循序渐进；而模式匹配视角下顺序根本不该重要。

**本文目标**：(1) 系统刻画 many-shot CoT-ICL 的 scaling、retrieval、ordering 三个维度行为；(2) 找出经验规律失效的根本原因；(3) 提出一个新视角统一解释现象并指导 demonstration 选择/排序设计。

**切入角度**：把 many-shot CoT 看作 **in-context test-time learning**（in-context 测试时学习）：long-context window 不是简单的“检索缓存”，而是一个 implicit curriculum，模型 forward pass 是一种 gradient-free adaptation。这视角自然推出两条 pedagogical 原则：(P1) demonstration 必须**对模型可理解**才能成为有效 supervision；(P2) demonstration 顺序必须**平滑过渡**，避免突兀的概念跳跃打断 implicit 学习轨迹。

**核心 idea**：基于 P2，把 demonstration 顺序看作 embedding 空间中的轨迹，**总曲率**（相邻位移角度之和）就是顺序“平滑度”的量化指标；最小化总曲率即可得到一个连贯的 in-context curriculum——这就是 Curvilinear Demonstration Selection (CDS)。

## 方法详解

### 整体框架
本文先做大量诊断实验暴露三条规律失效，再以 in-context test-time learning 视角重构理论，最后落地 CDS。**诊断阶段**用 4 类非推理 LLM（LLaMA 3.1 8B / 3.3 70B / Qwen2.5 7B / 14B）与 4 类推理 LLM（Qwen3 8B / 14B / QwQ 32B / DeepSeek-R1 685B），在分类任务（SuperGLUE, NLU, TREC, BANKING77）和数学/叙事推理任务（GSM8K, MATH 的 geometry / number_theory / counting_and_probability, DetectiveQA）上跑 1-128 shot，统一用开放式生成 + exact match 评估。**CDS 算法**则对 $n$ 条 demonstrations 求一个排列 $O = [\mathbf{d}_{\pi(1)}, \ldots, \mathbf{d}_{\pi(n)}]$ 最小化总曲率 $\Theta(O) = \sum_{t=2}^{n-1} \arccos\!\left(\frac{\mathbf{v}_t \cdot \mathbf{v}_{t+1}}{\|\mathbf{v}_t\|\|\mathbf{v}_{t+1}\|}\right)$，其中 $\mathbf{v}_t = \tilde{\mathbf{e}}_t - \tilde{\mathbf{e}}_{t-1}$ 是相邻 demonstration 投影嵌入的位移向量。

### 关键设计

1. **诊断实验：揭露三条经验规律在 CoT 推理上同时崩**:

    - 功能：用对比设计把“规律是否仍 hold”一条条压在显微镜下检验。
    - 核心思路：(A) **Scaling**：在非推理 LLM 上跑 geometry/number_theory 等推理任务，发现 shot 数增加性能**不稳甚至下降**（如 LLaMA 3.3 70B 在 CoT-ICL 上 negative gain）；只有 reasoning-oriented LLM（Qwen3, QwQ, R1）才呈现单调正向 scaling。表 1 进一步显示，在 Qwen3 上关闭 thinking mode 直接降几何题 7 pp，证明 reasoning prior 是 scaling 的必要条件。(B) **Retrieval**：embedding cosine 取 top-k 最相似 vs bottom-k 最不相似；在 BANKING77 上 top-k 显著优于 bottom-k（验证检索假设），但在 geometry/number_theory/DetectiveQA 上 top-k **反而最差**——semantic similarity 不预测 procedural compatibility。(C) **Ordering**：5 个随机 permutation 算 std，非推理任务 std 随 shot 数下降，推理任务 std 反而**随 shot 数上升**，表明 path dependence 强且加深。
    - 设计动机：把 many-shot ICL 的“常识”逐条放进 CoT 推理场景对照，三个独立维度同时破坏才能说服读者“CoT-ICL 是另一回事”而不是“某个数据集巧合”。这种“三角验证”是诊断式实证工作的范式，比单一现象更有说服力。

2. **Procedure absorption 直接证据：corrupted CoT 消融**:

    - 功能：分离“模型只用最终答案 $y$”和“模型真的吸收中间推理 $C$”两种假设。
    - 核心思路：在 geometry 上构造两组 prompt——正常版 $(x_i, C_i, y_i)$ 和**procedurally corrupted** 版 $(x_i, C_0, y_i)$，后者把所有 rationale 都替换成第一条 demonstration 的 chain，但保留每条的 question 和最终 answer。控制了格式、context 长度、$x \to y$ 映射，只改 $C$。表 2：在 $n=16$ 下两组几乎无差别，在 $n=128$ 下 corrupted 版让 Qwen3-8B 掉 1.25 pp、Qwen3-14B 掉 2.51 pp。
    - 设计动机：直接对“模型是否真的在 read demonstrations 的 procedure”给出反事实证据。short prompt 下差别小说明模型既能从 IO 也能从 CoT 中学；long prompt 下差别大说明 procedure 才是 scaling 的真正信号——给“in-context test-time learning”视角提供了硬证据，比单纯讲哲学有说服力得多。

3. **Curvilinear Demonstration Selection (CDS)：最小总曲率排序**:

    - 功能：基于平滑过渡原则，给定 $n$ 条 demonstration 找一个让 implicit 学习轨迹最平滑的排列。
    - 核心思路：(i) 把每条 demonstration $\mathbf{d}_i$ 表示成 (question + CoT + answer) 用 Qwen3-Embedding-4B 编码成 $\mathbf{e}_i \in \mathbb{R}^d$，关键是用**完整 demonstration**而非仅 question——因为顺序效应取决于 procedural 内容，光看 question 抓不到 CoT 结构。(ii) 把 prompt 内所有 embeddings 投影到低维子空间 $\tilde{\mathbf{e}}_i \in \mathbb{R}^{d'}$ 让曲率估计稳定。(iii) 定义局部曲率 $\theta_i = \arccos\!\left(\frac{(\tilde{\mathbf{e}}_i - \tilde{\mathbf{e}}_{i-1}) \cdot (\tilde{\mathbf{e}}_{i+1} - \tilde{\mathbf{e}}_i)}{\|\cdot\|\|\cdot\|}\right)$ 为相邻位移夹角，总曲率 $\Theta(O) = \sum_{i=2}^{n-1}\theta_i$。(iv) 搜一个排列使 $\Theta$ 最小（具体算法在 Section 6）。
    - 设计动机：作者先观察到 ordering curvature 与准确率显著负相关（总 $r=-0.547$，geometry $-0.545$，counting $-0.628$），所以最小曲率自然成为目标。为了排除“仅仅是把相似项聚在一起”，他们还做了 high-curvature 反向 baseline——保持局部邻域但反转曲率目标制造突兀转折——发现 CDS 仍胜出，证明**平滑过渡本身**而非聚类是 causal 因素。这种 causal smoothness ablation 是论文方法论的亮点。

### 损失函数 / 训练策略
CDS 完全是**推断时**算法，无任何训练。底层 embedding 模型用 Qwen3-Embedding-4B（off-the-shelf），评估模型涵盖 LLaMA、Qwen2.5、Qwen3、QwQ、DeepSeek-R1 系列，prompt 上下文最大 131K tokens，shot 数扫 $n \leq 128$。

## 实验关键数据

### 主实验
CDS 在 Qwen3 系列上的提升（几何 / 数论 / DetectiveQA）：

| 任务 | 模型 | 配置 | n=64 提升 |
|---|---|---|---|
| Geometry | Qwen3-14B | CDS vs 随机排序 | **+5.42 pp** |
| Geometry | Qwen3-14B | n=128 + thinking on | 73.07% vs n=16 的 66.18% |
| Geometry | Qwen3-14B | thinking on vs off (n=128) | 73.07 vs 65.76 |
| Number_theory | Qwen3-14B | thinking on vs off (n=128) | 91.30 vs 88.15 |
| DetectiveQA | Qwen3-8B | thinking on vs off (n=128) | 69.48 vs 66.88 |

### 消融实验

| 配置 | 行为 | 说明 |
|---|---|---|
| CDS (low curvature) | 最佳 | 完整方法 |
| High-curvature baseline | 显著差 | 同 embedding 邻域、反转曲率目标 |
| 相似度 top-k 检索 | 反而差 | semantic similarity 不预测 procedural compatibility |
| 相似度 bottom-k | 介于 top-k 与原始之间 | 反直觉 |
| Procedurally corrupted CoT (n=128) | 显著差（-1.25 to -2.51 pp） | 证明 procedure 起关键作用 |
| Thinking mode disabled | 显著差 | reasoning prior 是 scaling 必要条件 |
| 非推理 LLM + CoT-ICL | scaling 不稳甚至负向 | model class 决定能否吸收 CoT |

### 关键发现
- **CoT-ICL 不是规模化模式匹配**：相似度检索在 BANKING77（非推理）有效但在 geometry/number_theory/DetectiveQA（推理）**反向**，否决了 retrieval hypothesis 在推理上的有效性。
- **顺序敏感性随 shot 数上升**（与非推理任务相反）：100+ 个 demonstration 随机排会有更多“概念突变”，触发 procedural 不连贯。
- **Self-generated CoT 优于 ground-truth CoT**：弱模型上自生 CoT（甚至带错答案）比数据集 CoT 表现更好；这种优势随模型变强而缩小，验证 P1（“可理解性优先”）。
- **Reasoning-oriented LLM 与非推理 LLM 的 scaling 差距**根源在 thinking token——它把 demonstration 当 procedural supervision 抽取，而非把 IO 当模式匹配。
- **总曲率与准确率显著负相关**（geometry $r=-0.545$，counting $r=-0.628$），所以最小曲率不是 ad-hoc 启发式而是可量化的目标。

## 亮点与洞察
- **in-context test-time learning 视角是个统一锚点**：从这个角度看，scaling 失败（P1 违反）、相似度失败（procedure 不匹配 surface）、顺序敏感（P2 违反）三件事全都被一句话覆盖——长 context 是 implicit curriculum 而非 cache。这种“一个视角解释三类异常现象”的统一性给后续 prompt 工程提供了清晰的设计指导。
- **Self-generated CoT 优于 ground-truth CoT** 是一个非常反直觉但合理的发现：模型对自己生成的 CoT 更“能读懂”，即使带错答案也能从 procedural 上下文中受益。把它写进 prompt pipeline 就是一个免费的工程升级——给弱模型用自己的 CoT 训自己。
- **总曲率作为 ordering 目标**：把抽象的“平滑过渡”量化为相邻位移夹角之和，既几何直觉强又算得动；causal smoothness ablation 用高曲率反向 baseline 排除“相似聚集”混淆，方法论扎实。
- **embedding 用完整 demonstration**这一细节关键：仅 question embedding 会丢失 CoT procedural 结构；用 question + CoT + answer 才能让曲率反映 procedural 转换难度。

## 局限与展望
- CDS 的核心“平滑过渡”假设依赖 embedding 空间对 procedural 内容的可表达性；如果 embedding 模型本身对 CoT 内部结构编码差（如 instruction-only 模型），曲率信号失真，方法效果难保证。
- 实验集中在数学和叙事推理；编程、定理证明、agentic planning 等更复杂的推理类型是否同样满足曲率-性能负相关未验证。
- 没给出 CDS 的最优化算法复杂度与对比（TSP 类排序问题）；shot 数大时 minimization 本身可能成为瓶颈。
- “self-generated CoT 优于 ground-truth” 的优势随模型变强缩小——但这是否意味着未来强模型完全可以扔掉 self-generation 这一步骤，论文没量化。
- 未来可探索把曲率项作为可微正则直接 inject 到训练里（curriculum learning fine-tuning），或与 RAG 的 chunk 排序结合做 retrieval-aware curriculum。

## 相关工作与启发
- **vs Bertsch et al. / Baek et al.（many-shot ICL）**：他们在非推理任务上发现 scale + 顺序鲁棒 + 检索有效；本文证明这三条在 CoT 推理上同时失效，是对该工作的关键 corrective。
- **vs Auto-CoT (Zhang et al.) / Dr.ICL (Luo et al.)**：他们在 few-shot 场景做 CoT demonstration 选择；本文聚焦 many-shot 设定的全新动力学。
- **vs Test-time scaling (Snell et al.)**：test-time scaling 主要靠 sample-and-revise 增加 inference 计算；本文把 many-shot CoT 视为另一种 test-time scaling 形式，把 demonstration 当 in-context supervision。
- **启发**：(1) 任何依赖“长 context 把检索做大”的工程（RAG、agent memory）都应该重新考虑 ordering 的影响；(2) 教育心理学的 "zone of proximal development" 和 textbook 曲线观点在 prompt 工程里有具体可量化对应物，可能催生“pedagogical prompting”这个新子领域。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个系统化否定 many-shot ICL 经验法则在 CoT 上的迁移，重构视角并落地 CDS
- 实验充分度: ⭐⭐⭐⭐ 4+4 模型 × 多任务 × 多 shot × 多 seed，covered 三大维度且配 causal ablation；但 CDS 评测主要在 Qwen3 一家
- 写作质量: ⭐⭐⭐⭐⭐ 诊断—理论—算法—验证的链条清晰，pedagogical 类比贴切
- 价值: ⭐⭐⭐⭐⭐ 对所有依赖 long-context 的 prompt 工程都是 wake-up call，CDS 是即插即用的工程升级

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoT-ICL Lab: A Synthetic Framework for Studying Chain-of-Thought Learning from In-Context Demonstrations](../../ACL2025/llm_reasoning/cot-icl_lab_a_synthetic_framework_for_studying_chain-of-thought_learning_from_in.md)
- [\[ICLR 2026\] Is In-Context Learning Learning?](../../ICLR2026/llm_reasoning/is_in-context_learning_learning.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)

</div>

<!-- RELATED:END -->
