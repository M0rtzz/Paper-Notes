---
title: >-
  [论文解读] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping
description: >-
  [ICML 2026][LLM安全][Markov 过程] 把自回归 LLM 的下一个 token 分布解释成一条 Markov 链的状态转移矩阵，于是「学新词」就变成「在状态空间里加新状态、并把它表示为已有状态的稀疏组合」，理论上只需 $O(s)$ 样本（$s$ 为映射到的旧 token 数）…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "Markov 过程"
  - "词表扩展"
  - "embedding tuning"
  - "零遗忘"
  - "样本复杂度"
---

# Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping

**会议**: ICML 2026  
**arXiv**: [2605.04308](https://arxiv.org/abs/2605.04308)  
**代码**: 无  
**领域**: LLM 持续学习 / 词表扩展 / 抗遗忘  
**关键词**: Markov 过程, 词表扩展, embedding tuning, 零遗忘, 样本复杂度

## 一句话总结
把自回归 LLM 的下一个 token 分布解释成一条 Markov 链的状态转移矩阵，于是「学新词」就变成「在状态空间里加新状态、并把它表示为已有状态的稀疏组合」，理论上只需 $O(s)$ 样本（$s$ 为映射到的旧 token 数），实践中只 finetune 新 token 的 embedding 即可在严格零遗忘下完成跨语种/新概念扩展。

## 研究背景与动机

**领域现状**：把预训练 LLM 适配到新词汇 / 新实体 / 新领域是持续学习的核心需求（如 “COVID-19”、特定专业术语、跨语种迁移）。主流做法是 full fine-tuning、LoRA、prompt tuning，或者外挂检索。

**现有痛点**：哪怕在 Llama-3、Qwen-2.5 这种新模型上，标准 fine-tuning 仍会出现明显的灾难性遗忘——而且模型越大反而越严重；同时模型更新是不可逆的，一旦改坏就回不去。

**核心矛盾**：现代 LLM 已经非常 expressive，为了「装下一小撮新知识」却必须更新数十亿参数，本身就违反直觉；而且全量更新天然会污染旧词之间的转移关系。

**本文目标**：给「新增 $m \ll T$ 个新 token、不破坏原有 $T$ 个 token 之间转移关系」这件事建立一个干净的数学框架，并给出可证明的样本复杂度。

**切入角度**：把 LLM 看作一条 Markov 链——token 是状态、$p_\theta(\cdot \mid x_t)$ 就是转移概率向量。在这个视角下「不遗忘」就等价于「保持旧状态之间的转移矩阵不变」，而「新增知识」就等价于「把状态空间从 $\mathcal{V}$ 扩到 $\mathcal{V} \cup \mathcal{U}$」。

**核心 idea**：每个新 token $u$ 只需要被表示成若干已有 token 的稀疏线性组合（在 embedding 空间里 $\bm{\alpha}^{(u)} \in \mathbb{R}^T$，$\|\bm{\alpha}^{(u)}\|_0 \le s$），就能复用旧字典的语义结构；而具体实现就是只 finetune 新 token 的 embedding 向量，旧权重一概不动，从而硬性保证零遗忘。

## 方法详解

### 整体框架
输入是一个预训练 LM $\texttt{LM}_\theta$、旧词表 $\mathcal{V}$ 和一小批新 token $\mathcal{U}$、以及包含新 token 的训练序列。输出是参数更新后的 $\texttt{LM}_{\tilde{\theta}}$，它在旧 token 上的行为与原模型完全一致，在新 token 上能复现 oracle 转移分布 $\mathbf{q}^{(u)}$。整条 pipeline 概念上分三步：(1) 把生成过程形式化为一阶 Markov 过程，并把「新增知识」翻译成「状态空间扩张 + 旧转移保持」两个约束；(2) 假设每个新 token 的真实转移可写成已有 token 转移的 $s$-稀疏组合，从理论上推出每个 token 需要 $O(s)$ 样本就能学到；(3) 实践上把上述映射策略落到 embedding tuning：只把 $\langle \text{spec} \rangle$ 等新 token 对应的 embedding 向量当作可训练参数，其他全部冻结。

### 关键设计

1. **Markov 化的知识扩展**:

    - 功能：把「新增词汇」这件事写成「Markov 链状态空间扩展 + 保持旧转移」的等式约束。
    - 核心思路：旧 token 之间的转移向量 $\mathbf{p}^{(v)} \in \Delta(\mathcal{V})$ 已由预训练模型给定；引入新 token $u$ 后只需要学一个 $\mathbf{q}^{(u)} \in \Delta(\mathcal{V})$，并强制 $p_{\tilde{\theta}}(u \mid v) = 0,\ p_{\tilde{\theta}}(u_i \mid u_j) = 0$。这样 $p_{\tilde{\theta}}(\cdot \mid x_t)$ 在 $x_t \in \mathcal{V}$ 时与原模型完全相同，遗忘量精确为 0。
    - 设计动机：把「行为保持」从经验/正则项升级成了结构约束，从此「零遗忘」是可以被推导出来的事实，而不只是经验观察。

2. **Token-to-Dictionary 稀疏映射**:

    - 功能：用旧 embedding 字典 $\mathbf{E} \in \mathbb{R}^{T \times d}$ 的稀疏组合 $\mathbf{E}^\top \bm{\alpha}^{(u)}$ 表示新 token，并直接学这个组合系数。
    - 核心思路：定义 $f: \mathbb{R}^d \to \Delta(\mathcal{V})$ 是模型的 logit 头，目标是找到 $\bm{\alpha}^{(u)}$ 使得 $f(\mathbf{E}^\top \bm{\alpha}^{(u)}) = \mathbf{q}^{(u)}$；在 $\|\bm{\alpha}^{(u)}\|_0 \le s,\ \|\bm{\alpha}^{(u)}\|_2 \le B$ 的稀疏约束下用 KL 拟合经验转移分布 $\hat{\mathbf{q}}^{(u)}$。论文证明每个新 token 所需样本数 $N$ 满足 $N \ge \tilde{O}(s \log^2 c)$，与旧词表大小 $T$ 和模型维度无关。
    - 设计动机：新概念几乎从不是「全新」的——“COVID-19” 在转移上接近 “virus / disease / outbreak” 等若干旧词的混合。把这种「语义颗粒度」直接编码到样本复杂度里，比简单地按参数量算 cost 更切合实际。

3. **Embedding Tuning 训练算法**:

    - 功能：把上述映射策略以最小的实现代价落到 Transformer 上，做到「实算可行 + 严格零遗忘」。
    - 核心思路：把新 token 对应的 embedding 行（如 Llama-3 中 `<|reserved_special_token_0|>` 那 3072 维）设为唯一的可训练参数，其他 30 亿参数全部冻结；用标准 next-token 交叉熵在含新 token 的训练语料上做梯度下降；推理时新 token 在 attention/FFN 中走的还是旧权重，只是它的查询向量被「拉到」最能复现 $\mathbf{q}^{(u)}$ 的位置。
    - 设计动机：这种实现天然满足理论假设——更新的参数和旧 token 的 transition 完全正交（旧 token 永远不会查询到新 embedding，除非新 token 真的出现在 context 里），从而把「零遗忘」从条件理论变成实操可观测。

### 损失函数 / 训练策略
标准下一 token 交叉熵 $-\sum_t \log p_{\tilde{\theta}}(x_{t+1} \mid x_{t-k:t})$，没有正则项、没有 replay。论文还讨论了高阶 Markov 链扩展：把上下文 $K$ 个 token 视作组合状态后理论结论原样成立，由于自然语言的有效 branching factor $b \ll T$，实际稀疏度仅为 $s = O(Kb)$。

## 实验关键数据

### 主实验
用三类任务验证「样本效率 + 零遗忘」：算术算子任务上 Llama-3.2-3B 学会 $a\langle\text{spec}\rangle b = a \times b$，并对比 ET 和 FFT 在加法保留上的表现；合成词汇任务上把 100 个伪词（如 “glor”, “zorp”）注入真实句子并测 WikiText 上的遗忘；跨语种任务上把 Qwen2.5-3B 适配到西班牙语 / 德语 / 阿拉伯语。

| 任务 | 模型 | 方法 | 目标指标 | 遗忘量 (英语 / 加法) |
|------|------|------|----------|----------------------|
| 算术算子 ($a\langle\text{spec}\rangle b$) | Llama-3.2-3B | FFT | 77.2% acc | 加法 100% → 0%（灾难） |
| 算术算子 | Llama-3.2-3B | ET | **81.4%** acc | 加法仍 100% |
| 跨语种 (西班牙语) | Qwen2.5-3B | FFT | loss 5.56 | 英语 loss +9.83 |
| 跨语种 (西班牙语) | Qwen2.5-3B | ET | **loss 2.30** | 英语 loss **−0.04** |
| 跨语种 (阿拉伯语) | Qwen2.5-3B | ET | **loss 2.82** | 英语 loss **0.00** |

### 消融实验

| 设置 | 合成词汇 test loss | WikiText 遗忘 |
|------|-------------------|---------------|
| Base 模型 (Llama-3.1-8B) | — | 基线 2.42 |
| FFT (N=1000) | 4.40 | +1.24 |
| LoRA | 7.63 | +8.36 |
| Prompt Tuning | 3.79 | +0.69 |
| ET（本文） | 2.42 | **0.00** |

### 关键发现
- ET 在合成词汇 + 跨语种 + 算术算子三个完全不同的任务上同时拿到「最好 / 接近最好的 target loss」和「精确为 0 的遗忘」，验证了零遗忘并不是用准确率换来的。
- 算术算子上 $a\langle\text{spec}\rangle b$ 的 ET 准确率 81.4% 反而超过了 $a*b$ 的 base 63.5%，原因是 $\langle\text{spec}\rangle$ 的 embedding 隐式收敛到 “*$\times$ / times / multiplies / *” 等多个等价表述的稀疏组合（ensemble 准确率 73.2%），这是对「稀疏字典假设」最直接的实证。
- LoRA 在跨语种任务上不仅没遗忘优势，反而比 FFT 还差（loss 7.63 vs 4.40，遗忘 +8.36），说明「参数少」并不等于「不遗忘」；遗忘来自更新方向是否触碰旧 transition，而非更新参数量。
- Spanish / German 上甚至出现轻微「负遗忘」（−0.04 / −0.08），即学了西语反而让英语 WikiText 略微变好，作者猜测是近亲语种的迁移增益。

## 亮点与洞察
- 把「持续学习」彻底从经验调参问题转化为「Markov 链状态空间扩展」的等式问题，给出可以解析推导的「零遗忘」结构条件。这种思路可以迁移到任何输出是离散分布的自回归模型（语音、code、symbolic）。
- 样本复杂度只依赖映射稀疏度 $s$ 而非词表大小 $T$ 或模型维度 $d$，给出了一个反直觉但极有指导意义的 scaling 规则：词表越大并不代表学新词越贵，前提是稀疏假设成立。
- 「只 finetune 新 token embedding」是对 LoRA / PT 这条 PEFT 路线的极端化——它不靠秩约束、不靠 prompt prefix，而是直接利用了 embedding 与 transition 计算图的天然正交性。

## 局限与展望
- 假设每个新 token 在训练语料中以均匀概率出现，实际语料里频率分布很可能与「映射到的旧 token 簇」强耦合（高频词更容易污染旧 transition）。
- 假设 LLM 足够 expressive（$f$ 是 Lipschitz 且能精确实现任何稀疏组合）；当模型容量受限或新概念远离原字典覆盖时，稀疏假设可能破裂。
- 该方法默认「不允许从旧 token 转移到新 token」，意味着新词只能被「召出」而无法在生成中被自然引出，对生成式应用而言是一个明显约束，作者也承认是未来工作。
- 没有在「新 token 之间互相转移」的复杂结构（如学一整套新领域的术语网络）上做实验，只验证了「单点扩张」。

## 相关工作与启发
- **vs LoRA / PT / Adapter**：它们减少了更新的参数量但不能保证更新方向与旧 transition 正交，因此遗忘仍不可控；本文做法直接利用 embedding 表与 transition 图的正交结构，零遗忘是结构性保证而不是调参结果。
- **vs EWC / GEM / replay**：经典持续学习方法靠正则或回放压制遗忘，规模化到 LLM 仍困难；本文方法不需要旧数据回放、不需要 Fisher 信息，只增改新 embedding。
- **vs FlexOlmo / model editing**：那些方法假设「先有目标行为再编辑」，本文则是「先约束更新空间结构性地隔离遗忘」，两种范式互补。
- **vs Markov-LM 相关工作（Zekri et al., Yüksel & Flammarion）**：他们用 Markov 视角分析 transformer 的学习能力，本文则把这条数学路径用来设计具体的持续学习算法。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 Markov 视角与稀疏字典假设结合给出可证明的样本复杂度，理论框架清新。
- 实验充分度: ⭐⭐⭐ 算术 + 合成词 + 三个目标语种 + 多模型规模都跑了，但未涉及更现实的多领域 / 多步扩张。
- 写作质量: ⭐⭐⭐⭐ 从动机到定理再到算法逻辑非常顺，符号统一、定义清楚。
- 价值: ⭐⭐⭐⭐ 给出了「零遗忘 + 样本高效」可同时成立的最小化方案，对持续学习社区是有强示范意义的 baseline。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](from_volume_to_value_preference-aligned_memory_construction_for_on-device_rag.md)
- [\[ICML 2026\] Optimizing Token Choice for Code Watermarking: An RL Approach](optimizing_token_choice_for_code_watermarking_an_rl_approach.md)
- [\[ICML 2026\] Efficient DP-SGD for LLMs with Randomized Clipping](efficient_dp-sgd_for_llms_with_randomized_clipping.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)

</div>

<!-- RELATED:END -->
