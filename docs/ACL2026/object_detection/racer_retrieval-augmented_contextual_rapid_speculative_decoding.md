---
title: >-
  [论文解读] RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding
description: >-
  [ACL 2026][目标检测][推测解码] RACER 提出了一种无需训练的推测解码方法，将基于检索的精确模式匹配与基于 logits 的未来预测统一起来，通过 copy-logit 策略构建 Logits Tree、LRU 驱逐的 AC 自动机构建 Retrieval Tree，在多个基准上实现了超过 2 倍的推理加速。
tags:
  - ACL 2026
  - 目标检测
  - 推测解码
  - 检索增强
  - 训练无关
  - AC自动机
  - 推理加速
---

# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding

**会议**: ACL 2026  
**arXiv**: [2604.14885](https://arxiv.org/abs/2604.14885)  
**代码**: https://github.com/hkr04/RACER  
**领域**: 目标检测 / LLM推理加速  
**关键词**: 推测解码, 检索增强, 训练无关, AC自动机, 推理加速

## 一句话总结

RACER 提出了一种无需训练的推测解码方法，将基于检索的精确模式匹配与基于 logits 的未来预测统一起来，通过 copy-logit 策略构建 Logits Tree、LRU 驱逐的 AC 自动机构建 Retrieval Tree，在多个基准上实现了超过 2 倍的推理加速。

## 研究背景与动机

**领域现状**：LLM 的自回归解码每步只生成一个 token，推理延迟随序列长度线性增长。推测解码（Speculative Decoding）通过"猜测-验证"策略在不牺牲输出质量的前提下并行验证多个 token，是最有前景的加速方案之一。

**现有痛点**：现有免训练方法存在两类问题：(1) 基于检索的方法（如 PLD、REST）依赖精确 token 匹配，当上下文中不存在匹配续写时完全失效；(2) 基于 logits 的方法（如 Token Recycling）缺乏结构化引导，预测范围窄且质量次优。两类方法各有优势但相互割裂。

**核心矛盾**：检索提供"已见信息"（精确但稀疏），logits 提供"未见信息"（灵活但缺乏锚点）。两者互补但现有方法未能有效融合。

**本文目标**：设计一个轻量级、即插即用的无训练推测解码方法，统一检索和 logits 两种信号源。

**切入角度**：作者发现 copy-logit 策略（复用上下文中相同 token 最近出现位置的 logits）比 last-logit 策略有更高的接受率且分布更尖锐（rank-1 占比超 50%），这为构建高效的 logits 草稿树提供了基础。

**核心 idea**：用 AC 自动机维护上下文中的 n-gram 模式作为结构化检索锚点，用 copy-logit 构建逐层剪枝的 logits 草稿树进行灵活外推，两者在固定容量下动态分配预算并通过 trie 合并成统一草稿树。

## 方法详解

### 整体框架

在每个解码步中，RACER 首先通过 AC 自动机识别当前上下文中的匹配模式，从频率最高的续写中选取检索候选；然后将剩余容量分配给 Logits Tree 进行基于 copy-logit 的广度优先展开；最后将两棵树通过 trie 合并为统一草稿树，由目标模型用 tree attention 一次性验证。

### 关键设计

1. **Logits Tree（基于 copy-logit 的草稿树）**:

    - 功能：利用目标模型自身的 logits 信息生成多层次的 token 候选
    - 核心思路：采用 copy-logit 策略——对于当前采样的 next-token $x_t$，找到上下文中最近一次出现 $x_i = x_t$ 的位置 $i$，复用其后续 logits $\mathbf{z}_{i+1}$ 作为 $\mathbf{z}_{t+1}$ 的近似。实验显示 copy-logit 的 MAT 为 1.87（vs last-logit 的 1.57），且 rank-1 接受率超过 50%。基于重尾分布特性，设计递减广度分配：$b_{child(i,j)} = \max(1, \lfloor b_i / 2^{j+[i\neq 0]} \rfloor)$，首层最宽、深层逐步剪枝
    - 设计动机：copy-logit 基于"相同 token 在相似上下文中具有相似语义延续"的假设，比简单复用上一步 logits 更准确。递减广度分配符合接受率随深度快速衰减的经验规律

2. **Retrieval Tree（带 LRU 驱逐的 AC 自动机）**:

    - 功能：从生成上下文中高效检索重复 n-gram 模式，提供结构化的草稿候选
    - 核心思路：使用 Aho-Corasick 自动机维护上下文中出现过的 n-gram（最大长度 10）。设定最大节点容量（10,000），通过 LRU 驱逐策略淘汰最久未使用的叶节点。匹配时找到深度 $\geq 2$ 的所有边界节点，从其子树中选取全局频率最高的 top-k 续写作为检索候选。失败链在 prefill 阶段结束时惰性重建
    - 设计动机：suffix array 和 suffix automaton 随上下文长度线性增长且无法淘汰过时状态。AC 自动机的失败链能自然地丰富草稿多样性，LRU 驱逐保证内存稳定

3. **统一集成策略**:

    - 功能：在固定草稿容量下动态平衡检索和 logits 两种候选源
    - 核心思路：优先分配检索候选（结构可靠但稀疏），剩余容量给 Logits Tree 广度优先展开。两者通过 trie-based union 合并为统一草稿树，由目标模型在 tree attention 下一次性验证
    - 设计动机：检索信号捕捉近距离重复模式，为 logits 分布提供更尖锐的预测引导，减少推测展开中的误差累积

### 损失函数 / 训练策略

RACER 完全无需训练。默认超参数：Logits Tree 最大广度 8，Retrieval Tree 最大 10,000 节点、n-gram 长度 10，每步草稿容量 64。使用贪心解码（greedy decoding），batch size 为 1。

## 实验关键数据

### 主实验

| 模型 | 方法 | Spec-Bench 加速 | HumanEval 加速 | MGSM-ZH 加速 | 平均加速 |
|------|------|----------------|---------------|--------------|---------|
| Vicuna-7B | PLD | 1.50× | 1.40× | 2.27× | 1.87× |
| Vicuna-7B | LogitSpec | 1.77× | 1.66× | 2.67× | 2.03× |
| Vicuna-7B | Token Recycling | 2.06× | 2.17× | 2.30× | 2.18× |
| Vicuna-7B | **RACER** | **2.21×** | **2.29×** | **2.77×** | **2.42×** |
| Vicuna-33B | **RACER** | **2.20×** | **2.58×** | **2.77×** | **2.52×** |
| Qwen3-8B | EAGLE-3 | 2.14× | 2.44× | 0.86× | 1.81× |
| Qwen3-8B | **RACER** | **2.13×** | 2.24× | **2.26×** | **2.21×** |

### 消融实验

| 配置 | Spec-Bench MAT | HumanEval MAT | 说明 |
|------|---------------|---------------|------|
| RACER (完整) | 3.00 | 3.11 | 检索+logits 完整集成 |
| 仅 Logits Tree | ~2.76 | ~2.83 | 无检索引导，接近 Token Recycling |
| 仅 Retrieval Tree | ~1.82 | ~2.06 | 无 logits 外推，接近 REST |

### 关键发现

- RACER 在所有免训练方法中一致最优，平均加速比达 2.42×-2.52×
- 对比 EAGLE-3（需额外草稿模型），RACER 在 MAT 上略低但在实际加速比上持平甚至更优，因为无额外模型开销
- EAGLE-3 在中文推理（MGSM-ZH）上失效（加速 <1×），暴露了模型级方法对训练数据分布的敏感性，RACER 则稳定加速
- copy-logit 比 last-logit 的 MAT 高 0.3（1.87 vs 1.57），rank-1 接受率超 50%
- 方法对超参数不敏感，具有良好的鲁棒性

## 亮点与洞察

- copy-logit 策略是一个精巧的观察——相同 token 在不同位置的后续 logits 分布具有高度相似性。这个"上下文内 logits 复用"的思路简单但有效，适用于任何自回归模型
- 用 AC 自动机替代 suffix array 的选择很巧妙：失败链本身就提供了模式泛化能力，LRU 驱逐保证了固定内存开销。这个数据结构选择值得在其他需要在线模式匹配的场景借鉴
- "检索作为结构引导而非独立生成器"的定位——检索信号不是直接生成候选，而是为 logits 预测提供锚点和方向，这种融合哲学比简单组合更优雅

## 局限与展望

- 仅在 batch size=1 和贪心解码下评估，大 batch 和采样解码场景待验证
- AC 自动机的节点容量（10K）和 n-gram 长度（10）是固定的，自适应调节可能进一步提升性能
- 未探索与 model-based 方法的组合潜力
- 在非英文语言上的优势是否源于基于检索的语言无关性，值得深入分析

## 相关工作与启发

- **vs Token Recycling**: TR 仅用 top-k 邻接矩阵展开草稿树但无检索引导。RACER 通过 AC 自动机提供的结构锚点使 logits 展开更精准，平均多接受 0.4 个 token
- **vs EAGLE-3**: EAGLE-3 需要额外训练草稿模型，MAT 更高但实际加速比不一定更优。RACER 的零训练零额外内存优势使其更适合即插即用部署

## 评分

- 新颖性: ⭐⭐⭐⭐ 检索+logits 的统一视角新颖，copy-logit 和 LRU-AC 自动机设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多种模型规模（7B-33B）、多类任务、多语言，消融和分析充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 提供了实用的免训练推理加速方案，即插即用部署价值高

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/object_detection/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[NeurIPS 2025\] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](../../NeurIPS2025/object_detection/video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)
- [\[ICLR 2026\] ConFu: Contemplate the Future for Better Speculative Sampling](../../ICLR2026/object_detection/confu_contemplate_the_future_for_better_speculative_sampling.md)

<!-- RELATED:END -->
