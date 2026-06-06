---
title: >-
  [论文解读] Incremental BPE Tokenization
description: >-
  [ICML 2026][预训练][BPE 分词] 本文提出首个具有严格 $\mathcal{O}(\log^2 t)$ 单字节最坏复杂度的增量 BPE 分词算法，通过 Aho–Corasick 自动机定位搜索空间、Centroid Decomposition 上的二分搜索定位"最后一个 token"…
tags:
  - "ICML 2026"
  - "预训练"
  - "BPE 分词"
  - "增量算法"
  - "Aho–Corasick"
  - "Centroid Decomposition"
  - "流式输出"
---

# Incremental BPE Tokenization

**会议**: ICML 2026  
**arXiv**: [2605.30813](https://arxiv.org/abs/2605.30813)  
**代码**: https://github.com/ModelTC/mtc-inc-bpe (有)  
**领域**: NLP理解 / LLM效率（分词器、流式推理）  
**关键词**: BPE 分词、增量算法、Aho–Corasick、Centroid Decomposition、流式输出

## 一句话总结
本文提出首个具有严格 $\mathcal{O}(\log^2 t)$ 单字节最坏复杂度的增量 BPE 分词算法，通过 Aho–Corasick 自动机定位搜索空间、Centroid Decomposition 上的二分搜索定位"最后一个 token"，作为 drop-in replacement 相对 Hugging Face tokenizers 最高 $\sim 3\times$ 加速，并在病态输入上消除了 tiktoken 的 $\mathcal{O}(n^2)$ 退化。

## 研究背景与动机

**领域现状**：BPE（Byte Pair Encoding）已经是现代 LLM 事实上的分词标准——GPT 系列、Qwen-3、LLaMA、DeepSeek 全用它。两大主流实现 Hugging Face `tokenizers` 用堆维护全局优先队列处理整段输入；OpenAI `tiktoken` 则依赖 regex 把输入预切成小段，再在每段上跑 BPE 合并。两者本质都是**离线（offline）**算法，必须先看到完整片段才能给出正则化的分词结果。

**现有痛点**：离线特性带来两个直接后果。第一，prefill 阶段必须**串行**等待分词完成才能开始推理，无法把分词和模型前向 pipeline 起来，长上下文场景下这一段延迟变得不可忽视。第二，tiktoken 在某些病态输入（如 `'a' × 2^k` 这种长重复）上是真正的 $\mathcal{O}(n^2)$ 行为；甚至上游的 regex 引擎本身在超长字符串上会栈溢出或崩溃，等于把 BPE 这一阶段变成了潜在的算法复杂度攻击面。

**核心矛盾**：现有实现都把 BPE 视作"读完再合并"的全局过程——堆方法要看到全部 pair 才能找最大优先级，regex 预切要保证句子边界不被穿透。这种全局视角与流式增量天然冲突。Berglund & van der Merwe (2023) 在理论上证明了 BPE 满足"前缀一致性"（任意前缀的分词稳定地是完整分词的前缀），暗示了增量处理的可能性，但他们没有给出**算法构造**和**最坏复杂度界**。

**本文目标**：构造一个**严格等价于标准 BPE**的增量算法，每读入一个字节就以最坏 $\mathcal{O}(\log^2 t)$ 时间维护当前字符串所有前缀的分词结果（$t$ 为最长 token 长度），并支持 eager output——一旦某个 token 的边界在未来任何延伸下都不可能改变，立刻输出，从而把分词彻底 pipeline 化。

**切入角度**：作者把问题归约到一个核心子问题——**给定字符串 $s$ 和新字符 $c$，求新串 $sc$ 的最后一个 token $\theta(sc)$**。因为由前缀一致性，知道 $\theta(\cdot)$ 就能递归回溯出完整分词。而 $\theta(sc)$ 必然是 $sc$ 的某个后缀 token，所有候选构成"后缀-后继树"（Suffix-Successor Tree），作者证明合法候选在这棵树上构成**单调路径**——这把搜索空间从指数级压成对数级。

**核心 idea**：用 Aho–Corasick 自动机 $\mathcal{O}(1)$ 定位最长后缀 token 框定搜索树，用 Centroid Decomposition 在树上 $\mathcal{O}(\log t)$ 二分定位 $\theta(sc)$，每步用 DFS 时间戳区间检验把"前缀末 token 条件"压成 $\mathcal{O}(1)$，整体 $\mathcal{O}(\log^2 t)$/字节。

## 方法详解

### 整体框架

算法把"对长度 $n$ 的字符串做 BPE 分词"重构成 $n$ 次增量更新：每次读入一个字节 $c$，把状态 $\theta(s)$ 更新成 $\theta(sc)$，并隐式维护所有前缀的分词树。

整体流水线为：① **预处理**——对词表 $\mathcal{V}$ 做规范化（去掉不可达的 token，建立非原子 token 与生成它的 merge rule 的双射），构造 **Successor Forest**（每个非原子 token 指向其 successor，即合并它的右半部分）；② **离线索引**——对 Successor Forest 做预序 DFS 拿到 `dfs_in/dfs_out` 时间戳，每个 token 预算"有效区间" $I_t$；对每个 token $\tau$ 预算 Centroid Search Tree（CST）；建 Aho–Corasick 自动机并把"搜索空间入口"标注到每个状态；③ **在线增量**——读入字节 $c$，自动机转移到新状态 $\mathcal{O}(1)$ 得到最长后缀 token $\tau(sc)$，进入对应 SufSucTree 上的 CST 做对数次二分查找，定位单调路径上最深的合法节点即为 $\theta(sc)$。

整个在线阶段每字节最坏 $\mathcal{O}(\log^2 t)$，全文 $\mathcal{O}(n \log^2 t)$，远低于 tiktoken 在病态输入上的 $\mathcal{O}(n^2)$。

### 关键设计

1. **单调路径定理（Monotonic Path Property）—— 把全局合并问题归约到树上单调搜索**：

    - 功能：刻画在 SufSucTree 上哪些后缀 token 有资格成为 $\theta(sc)$，把"候选集"从所有后缀 token（线性多）压成 SufSucTree 上从根（原子 token $c$）到 $\theta(sc)$ 的**唯一单调路径**。
    - 核心思路：作者形式化"前缀末 token 条件"（Definition 4.1）——候选 $t$ 必须同时满足两个条件，记 $s^{-\operatorname{suc}(t)}$ 为去掉 $t$ 后缀部分后的前缀：(i) **可达性**——$\theta(s^{-\operatorname{suc}(t)})$ 必须落在 Successor Forest 中以 $\operatorname{pre}(t)$ 为根的子树内；(ii) **优先级支配**——若 $\theta(s^{-\operatorname{suc}(t)}) \neq \operatorname{pre}(t)$，则 $\operatorname{pre}(t)$ 到该祖先路径上的那个孩子 $u$，其 merge rule 优先级必须严格低于 $t$ 自己的 rule。Theorem 4.2 证明：满足该条件的所有 $t$ 在 $\operatorname{SufSucTree}(\tau(sc))$ 上恰好构成一条从根出发的单调路径，且 $\theta(sc)$ 是这条路径上最深的节点。"if" 方向的直觉是：在真正的 last token 的 successor 链上，可达性和优先级支配都被单调保持。
    - 设计动机：现有实现把 BPE 视作全局优先队列合并，每步要扫所有 pair，这是 $\mathcal{O}(n \log n)$ 甚至更糟的根源。单调路径性质把"哪个 token 当 last token"这个看似需要回溯全部历史的问题，变成了一棵预先静态构造的树上的二分搜索——天然适配增量更新和最坏复杂度分析。

2. **DFS 时间戳 + 有效区间 —— 把前缀末 token 条件压成 $\mathcal{O}(1)$ 区间检测**：

    - 功能：把上面那个看起来要查询 Successor Forest 子树成员关系 + 比较 rule 优先级的复杂判定，变成单次"整数是否落在区间内"的 $\mathcal{O}(1)$ 检测。
    - 核心思路：对 Successor Forest 做预序 DFS，关键技巧是**遍历孩子时按 canonical rule 优先级从低到高排序**——这样高优先级孩子的子树时间戳一定排在后面。这保证了任一非原子 token $t$ 的合法候选集 $\mathcal{C}_t$ 对应一段连续的时间戳区间 $I_t = [L_t, R_t)$，其中 $L_t = \operatorname{dfs\_in}(\operatorname{pre}(t))$，$R_t$ 是 $\operatorname{pre}(t)$ 第一个 rule 优先级 $\geq t$ 的孩子的 `dfs_in`（不存在则取 $\operatorname{dfs\_out}(\operatorname{pre}(t))$）。在线检测变成 `dfs_in(k) ∈ I_t` 这一个比较。另一个关键推论：SufSucTree 中**兄弟节点的有效区间互不相交**——这直接来自单调路径的唯一性，使得搜索时在每个分叉处都能无歧义地选对方向。
    - 设计动机：理论上的"前缀末 token 条件"如果每次重新计算会引入 $\mathcal{O}(t)$ 因子，无法支撑 $\mathcal{O}(\log^2 t)$ 目标。把树形归属关系线性化成区间是经典 OI 技巧，但与"优先级排序孩子"结合后才能同时编码可达性和优先级支配两个条件——这是把抽象理论性质变成可计算 $\mathcal{O}(1)$ 谓词的核心工程化步骤。

3. **Aho–Corasick + Centroid Decomposition —— 把增量更新做到对数复杂度**：

    - 功能：每读入一个字节，$\mathcal{O}(1)$ 定位搜索空间 $\operatorname{SufSucTree}(\tau(sc))$，再 $\mathcal{O}(\log^2 t)$ 在搜索空间内定位最深合法节点。
    - 核心思路：① 对词表 $\mathcal{V}$ 建 Aho–Corasick 自动机，对每个状态预计算"搜索空间入口"标注——若状态对应 token 自己在 $\mathcal{V}$ 里就用它，否则从 suffix link 继承——这样新字符到来时只需自动机走一步就能 $\mathcal{O}(1)$ 拿到 $\tau(sc)$，**不用沿 suffix link 回溯**。转移表用持久化分块（square-root tiling）压缩存储但保持 $\mathcal{O}(1)$ 查询。② 对每个 $\tau \in \mathcal{V}$ 离线建 Centroid Search Tree（CST），高度严格 $\mathcal{O}(\log |\tau|)$；在线搜索时从 CST 根开始，对当前重心 $u$ 用区间检测判断它是否合法：若**不合法**，按单调路径性质，目标必在 $u$ 父侧分量，转 CST 中"父方向"孩子；若**合法**，则 $u$ 在路径上，对 $u$ 在原 SufSucTree 中的孩子做二分（兄弟区间不相交可排序）找是否还有更深合法孩子 $v$，有则继续往 $v$ 方向走，无则 $\theta(sc) = u$ 终止。每个 CST 步做一次 $\mathcal{O}(\log t)$ 二分，CST 深度 $\mathcal{O}(\log t)$，合计 $\mathcal{O}(\log^2 t)$/字节。
    - 设计动机：朴素自顶向下遍历 SufSucTree 在最坏情况下是 $\mathcal{O}(t)$（树高线性于 $|\tau|$），无法满足复杂度目标。Centroid Decomposition 是把任意树压成 $\mathcal{O}(\log)$ 高度搜索结构的标准武器；结合 Aho–Corasick 把"搜索空间识别"也降到 $\mathcal{O}(1)$，两个组件配合才能把"严格最坏复杂度 + 增量更新"两个看似冲突的目标同时拿下。

### 损失函数 / 训练策略

本工作是纯算法/数据结构工作，无训练。Eager output 模块（§6）额外维护一个"主动前沿"$\mathcal{P}$——所有可能成为未来 token 父节点的候选末 token 集合，由 Aho–Corasick 当前状态深度 $d(s)$ 界定窗口 $[|s|-d(s), |s|]$；用双指针单调维护（窗口左端单调右移，过期的 token 不会再回来），当所有活跃路径汇聚到虚拟根的同一个孩子时即可 eager 输出，使分词与模型推理能完全 pipeline。Eager 模式相对非 eager 引入约 10% 吞吐 overhead。

## 实验关键数据

### 主实验

作为 Hugging Face `tokenizers` 和 OpenAI `tiktoken` 的 drop-in replacement，在英文 / 中文 / 代码三类语料上测端到端吞吐加速比：

| 后端 | 模型 | English | Chinese | Code |
|------|------|---------|---------|------|
| tokenizers | CodeLlama | **3.13×** | 1.10× | **2.88×** |
| tokenizers | Qwen-3 | 1.05× | 1.04× | 1.08× |
| tokenizers | DeepSeek-3.2 | 1.01× | 0.93× | 1.03× |
| tokenizers | Llama-3.1* | 0.99× | 1.03× | 1.02× |
| tokenizers | GPT-OSS | 1.00× | 1.08× | 1.01× |
| tiktoken | CL100K | 0.96× | **1.59×** | 1.04× |
| tiktoken | O200K | 0.99× | 1.46× | 1.00× |
| tiktoken | P50K | 0.97× | 1.35× | 1.07× |

\* properized dictionary。

### 病态输入鲁棒性

输入构造为 `'a' × 2^k`，对数尺度对比吞吐：

| 实现 | 复杂度行为 | 备注 |
|------|-----------|------|
| 本文增量 BPE | 稳定 $\mathcal{O}(n \log^2 t)$ | 全程吞吐基本平坦 |
| tiktoken | $\mathcal{O}(n^2)$ 显著衰减 | 长输入吞吐持续下降 |
| O200K 模型 | regex 阶段先崩 | 超长输入触发上游错误 |
| 本文 eager 输出 | 比非 eager 慢 ~10% | 额外维护活跃前沿 |

### 关键发现

- **CodeLlama 加速最大（最高 3.13×）**：它不做 regex pre-tokenization，BPE 直接吃完整规范化文本，原 `tokenizers` 的堆方法在长输入上吃 $\mathcal{O}(n \log n)$ 全亏，本文增量算法的优势被完全释放。反过来，pre-tokenization 把输入切得越细（如英文 + Qwen-3），实现常数项主导，本文增益甚至轻微负收益（0.99×–1.05×）——印证 pre-tokenization 本质上是为绕开全局 BPE 复杂度而设计的工程补丁，一旦底层算法本身有严格保证就变成冗余。
- **中文 + tiktoken 增益显著（1.35×–1.59×）**：中文的 regex 切分天然更粗，把 tiktoken 内部 BPE 在大块上的瓶颈暴露出来，本文的严格逐字节 bound 直接吃到这个差异。
- **病态输入是杀手锏**：tiktoken 在 `'a' × 2^k` 上吞吐曲线呈现明显的 $\mathcal{O}(n^2)$ 衰减，本文保持平坦——这不仅是性能问题，更是安全问题，作者明确把它列为缓解算法复杂度攻击（DoS）的副产品。

## 亮点与洞察

- **理论性质 → 数据结构的精确翻译**：Berglund & van der Merwe 2023 给的"前缀一致性"是抽象代数性质，本文把它逐层翻译成"Last Token 递归 → SufSucTree → 单调路径 → DFS 时间戳区间 → 兄弟区间不相交 → Centroid Decomposition 上二分"——每一步都是必要的复杂度优化，砍掉任何一层都到不了 $\mathcal{O}(\log^2 t)$。这种"理论性质做杠杆"的工作范式值得借鉴。
- **drop-in replacement 的工程美学**：作者刻意把改动限定在 BPE 这一阶段，segmentation/normalization/cache 全不动。结果是 benchmark 干净地隔离了 BPE 算法本身的贡献，也意味着所有现有 LLM pipeline 都能零摩擦切换——这是把学术算法做成可被 industry 采纳的标配产品的最佳姿势。
- **"病态输入"作为复杂度攻击面**：把 $\mathcal{O}(n^2)$ tokenizer 重新表述为"算法复杂度 DoS 漏洞"是个有视野的角度——LLM serving 系统暴露给互联网，攻击者只要发一段长重复字符就能把分词阶段卡死。把"严格最坏复杂度"和"安全性"绑定，能让算法工作的影响范围从效率延伸到系统可靠性。

## 局限与展望

- **依然不能解决 regex 预处理瓶颈**：作者在 Appendix I 的 profiling 显示，normalization 和 regex pre-tokenization 本身已经成为 pipeline 瓶颈；BPE 阶段虽然变成增量了，但上游仍是 offline，端到端流式分词还要等这些组件被增量化。
- **加速集中在少数场景**：在 fine-grained pre-tokenization 的英文场景，本文几乎打平甚至轻微回退（0.99×），说明对已经被 regex 切碎的输入，新算法的常数项还是吃亏的。如果未来 LLM 趋势是继续依赖 regex 预切，本文价值会被稀释；反之，若像 CodeLlama 那样去掉 pre-tokenization 的设计普及，本文价值会被放大。
- **Eager output 的 10% overhead 不便宜**：在已经被高度优化的 tokenizer 上 10% 是相当可观的代价，作者建议靠 pipeline 并行摊销，但实际能摊销多少很依赖下游推理速度。
- **不支持 SentencePiece 语义**：方法形式化在标准 BPE merge semantics 下，SentencePiece-style（如 Gemma-3）的某些不可 properize 词表（见 Appendix A）不适用。考虑到 Gemma-3 等模型的存在，这块覆盖空白需要后续工作补。

## 相关工作与启发

- **vs Hugging Face `tokenizers`（堆+全局优先队列）**：他们用全局堆每步取最高优先级 pair，是 offline、log-linear-on-full-input 的。本文用增量 + 树上对数搜索，严格 $\mathcal{O}(\log^2 t)$/字节；在无 pre-tokenization 的场景（CodeLlama）直接 3× 加速。
- **vs OpenAI `tiktoken`（regex 预切 + 段内 BPE）**：他们用 regex 把输入限制成小段以绕开全局 BPE 的复杂度，但 regex 引擎本身在病态输入上崩，段内 BPE 仍是 $\mathcal{O}(n^2)$ worst case。本文在算法层就给出严格上界，不再依赖 regex 这道工程补丁。
- **vs rust-gems `bpe` crate（van Antwerpen & Neubeck 2024）**：他们也用 Aho–Corasick 做增量 BPE，是工程上的先驱，但缺乏形式化最坏复杂度证明。本文在算法理论和工程实现上都做了严格化（详见 Appendix J 的细节对比）。
- **vs Berglund & van der Merwe 2023**：他们给了 BPE 形式语义和前缀一致性，但停在代数性质，没解决"如何 bound 增量更新所需 lookahead"的算法问题。本文把这条性质做成了可计算的算法。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个给出严格 $\mathcal{O}(\log^2 t)$ 最坏复杂度的增量 BPE 算法，多个理论性质（Monotonic Path Property）是新结果。
- 实验充分度: ⭐⭐⭐⭐ 主实验覆盖 8 个 tokenizers 模型 × 3 类语料 + 病态输入压测，benchmark 设计清晰；但缺少与 rust-gems bpe crate 的端到端正面对比。
- 写作质量: ⭐⭐⭐⭐⭐ 结构从理论性质 → 数据结构 → 算法 → 复杂度分析步步推进，每一步动机和归约都讲得很清楚，附录覆盖完整证明。
- 价值: ⭐⭐⭐⭐⭐ Drop-in replacement，可直接落地到所有现代 LLM serving 系统；同时把"复杂度保证"与"安全性"绑定，价值远超单纯的加速比。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Adversarial Tokenization](../../ACL2025/llm_pretraining/adversarial_tokenization.md)
- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](../../ACL2025/llm_pretraining/tokenization_is_sensitive_to_language_variation.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](../../ACL2025/llm_pretraining/incorporating_domain_knowledge_into_materials_tokenization.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)

</div>

<!-- RELATED:END -->
