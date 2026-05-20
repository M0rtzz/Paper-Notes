---
title: >-
  [论文解读] VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs
description: >-
  [ACL2026][LLM/NLP][数据多样性] Voyager 是一个无需训练的 LLM 数据生成算法，用 DPP 维护多样 anchor 和 explorer，并用 textual gradients 迭代改写提示词，从而在创意写作和推理数据生成中显著提高 Vendi 多样性且基本不牺牲质量。
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "数据多样性"
  - "合成数据生成"
  - "DPP"
  - "文本梯度"
  - "LLM采样"
---

# VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs

**会议**: ACL2026  
**arXiv**: [2512.12072](https://arxiv.org/abs/2512.12072)  
**代码**: 无  
**领域**: llm_nlp  
**关键词**: 数据多样性, 合成数据生成, DPP, 文本梯度, LLM采样

## 一句话总结
Voyager 是一个无需训练的 LLM 数据生成算法，用 DPP 维护多样 anchor 和 explorer，并用 textual gradients 迭代改写提示词，从而在创意写作和推理数据生成中显著提高 Vendi 多样性且基本不牺牲质量。

## 研究背景与动机
**领域现状**：LLM 已经常被用来生成合成训练数据、评测样本、用户画像和创意内容。但直接重复调用同一个 prompt 很容易产生 mode collapse：表面上句子不同，语义和结构却高度集中。温度采样、top-p、beam search 等 decoding 方法能增加局部随机性，但它们只作用在下一 token 分布上，并不直接优化整个数据集的全局多样性。

**现有痛点**：已有多样性控制方法各有短板。高温采样可能增加噪声但无法保证覆盖不同语义区域；“please be diverse” 这类 prompt 指令依赖模型自觉，缺少全局约束；把历史样本塞回 prompt 只能避免近期重复，窗口外仍会坍缩；基于 RL 或 post-training 的方法成本高，并且通常要求开源模型权重。

**核心矛盾**：真实需求是生成一个整体多样的数据集，而不是让单次输出看起来随机。若只在生成后做 subset selection，需要先造出一个很大的候选池，再用 DPP 或其他算法筛选，LLM 调用和矩阵计算都会很贵。Voyager 的目标是在生成过程中就朝多样区域探索。

**本文目标**：作者提出一个 training-free、closed-source LLM 也可用的算法：每轮生成一批样本，根据它们对 anchor set 体积的边际增益决定是否接收；被拒样本反过来作为反馈，让 LLM 产生 textual gradients，改写后续 explorer prompt。

**切入角度**：论文把多样性定义为相似度矩阵的 determinant / volume，并用 Determinantal Point Process (DPP) 做两个选择：选出代表性 anchor points，以及保留彼此不同的 explorer prompts。

**核心 idea**：把“多样数据生成”从 prompt engineering 变成一个迭代的 volume maximization 过程，用 DPP 提供全局多样性压力，用 textual gradients 提供可迁移的 prompt 搜索方向。

## 方法详解

### 整体框架
Voyager 输入一个任务 prompt $p$、目标数据集大小 $l$、边际增益阈值 $\tau$、explorer 数量上限 $b$、anchor set 大小 $k$、最大迭代数 $T$ 和相似度核 $K_{Sim}$。算法维护三个集合：已接受数据集 $D$、anchor set $\Phi$、explorer prompt 集合 $E$。初始时 $D$ 和 $\Phi$ 为空，$E$ 只包含原始任务 prompt。

每轮外层迭代中，每个 explorer 都调用一次 Explore 过程。Explore 首先让 LLM 生成一批候选样本 $B$；随后逐个计算候选样本加入 anchor set 后的 marginal volume gain。若增益不低于阈值 $\tau$，样本被加入数据集和候选 anchor；否则进入 rejected set。若存在 rejected samples，Voyager 再调用 LLM 分析这些失败样本和当前 anchors，生成 textual gradients，并把它们应用到当前 explorer 上，得到下一轮候选 explorer。

外层循环结束一批 explorer 后，算法用 DPP 从候选 anchor 中采样 $k$ 个点，保持 anchor set 小而多样；再用 DPP 从候选 explorer 中采样 $b$ 个 prompt，保证后续探索方向不集中。数据集达到目标大小 $l$ 或迭代到 $T$ 后停止。

### 关键设计
1. **用 determinant / volume 表示数据集多样性**:

	- 功能：把“多样”变成可优化的数学量。
	- 核心思路：对样本集合构建相似度矩阵，子集 $S$ 的 DPP 概率满足 $P(S) \propto det(K_S)$。当样本向量高度相似时，矩阵体积小；当样本张成更大空间时，determinant 更大。
	- 设计动机：Vendi Score 等有效秩指标也从相似度矩阵出发。论文给出近似关系 $V \approx n^2 D^{1/n}/C$，其中 $D$ 是 determinant，$C$ 是 trace，说明提高 determinant 与提高有效秩、多样性存在联系。

2. **用固定大小 anchor set 近似全局目标**:

	- 功能：避免每次都在完整数据集上计算高阶 determinant。
	- 核心思路：候选样本只需要相对 anchor set 计算边际体积增益，anchor set 通过 k-DPP 保持代表性和高体积。这样它相当于一个小型、多样的 reservoir，用来近似当前数据集覆盖的区域。
	- 设计动机：直接最大化最终数据集相似度矩阵 determinant 不可行，最大体积子矩阵问题也很难。固定 anchor set 把计算从完整数据集规模降到 $k$，边际增益可在缓存逆矩阵时以 $O(k^2)$ 计算。

3. **用 textual gradients 更新 explorer prompts**:

	- 功能：让算法不仅筛掉重复样本，还能学习下一步该往哪里探索。
	- 核心思路：被拒样本代表当前 prompt 产生的样本落在已有 anchor 附近。Voyager 让 LLM-judge 分析 prompt、rejected samples 和 anchors，给出“如何让 prompt 生成更不同样本”的自然语言建议，再让另一个 LLM 调整 prompt，形成 successor explorers。
	- 设计动机：DPP 提供的是选择压力，但不会自动产生新语义方向。textual gradients 把拒绝信息转化为 prompt 级别的搜索信号，适用于闭源 LLM，也不需要参数训练。

### 损失函数 / 训练策略
Voyager 没有模型训练损失。它优化的是生成过程中的集合多样性代理目标：尽量提高 anchor set 对应相似度矩阵的体积。相似度核是 RBF embedding kernel 和 Jaccard lexical similarity 的凸组合，权重分别为 0.7 和 0.3；embedding 使用 text-embedding-3-small，生成模型使用 GPT-4o mini。默认超参数为 $b=3$、$k=10$、$T=200$、batch size 10、目标数据集大小 500。

计算复杂度方面，Explore 的候选筛选约为 $O(|B|k^2)$，anchor DPP 裁剪约为 $O(k_{max}^3)$，explorer DPP 裁剪约为 $O(b_{max}^3)$。相比先生成超大候选池再从完整数据集用 DPP 采样，Voyager 的计算和 LLM 调用更可控。

## 实验关键数据

### 主实验
论文在 4 个创意写作任务和 2 个推理任务上比较 Default、Temp=2.0、Diverse、History、Hierarchical、SubsetSelect 与 Voyager。指标包括 lexical Jaccard distance、cosine distance、Vendi Score、LLM-as-judge quality 和 LLM calls。

| 任务 | Default Vendi | Hierarchical Vendi | Voyager Vendi | Voyager LLM calls | 主要结论 |
|------|---------------|--------------------|----------------|-------------------|----------|
| Sports sentence | 2.991 | 15.070 | 24.132 | 443 | Voyager 显著超过最佳强基线，调用少于 Hierarchical 的 550 |
| Political conversation | 4.589 | 8.450 | 15.035 | 426 | 语义和词汇多样性均最高，quality 也最高 |
| Poem | 3.004 | 5.679 | 7.312 | 615 | 多样性最高，同时 quality 24.505 高于所有 baseline |
| Movie plot | 4.002 | 7.661 | 8.302 | 695 | 多样性最高，但调用多于 Hierarchical |
| Grade-school math | 3.039 | 8.715 | 18.777 | 399 | 推理题生成多样性提升尤其明显 |
| Logic puzzle | 3.312 | 7.024 | 13.256 | 393 | 在逻辑题上也远高于所有 baseline |

作者报告的整体结果是：创意任务上，Voyager 相比 Default 的平均 Vendi 提升为 2.96x，相比 Hierarchical 仍提升 0.43x；推理任务上，相比 Default 提升 4.12x，相比 Hierarchical 提升 1.02x。质量分数没有显著下降，说明算法并非通过牺牲可用性换取离散噪声。

### 消融实验
消融部分验证两个核心模块：DPP explorer selection 和 textual gradients。作者还做了生成长度控制，以及 downstream synthetic training data 实验。

| 实验 | 对比设置 | 关键数据 | 结论 |
|------|----------|----------|------|
| Explorer DPP 消融 | Voyager-RE 随机选 explorer vs Voyager | Sports Vendi 11.852 vs 14.282；LLM calls 361 vs 252 | 多样 explorer 让搜索更有效，既提高多样性又减少调用 |
| Textual gradients 消融 | 禁用 prompt refinement | 拒绝率和所需迭代数显著上升 | 只筛样本不改 prompt 会卡在相似区域 |
| 控制生成长度 | Poem 限制约 150 tokens | Voyager Vendi 8.167，Hierarchical 6.646，Default 3.304 | 多样性收益不只是因为 Voyager 输出更长 |
| 人类评估 Sports | Default vs Voyager | 2.16±0.55 vs 3.82±0.28；Fleiss kappa 0.41；Krippendorff alpha 0.74 | 人类也认为 Voyager 样本更不同 |
| 人类评估 Math | Default vs Voyager | 1.56±0.36 vs 3.72±0.33；Fleiss kappa 0.34；Krippendorff alpha 0.72 | 自动 Vendi 与人工感知大体一致 |
| GSM8K synthetic training | Default 1000 vs Voyager 1000 | Gemma 7B-IT: 35.7 vs 45.7 | 多样合成数据提升下游训练效果 |
| 数据效率 | Voyager 500 vs Default 1000 | Gemma 7B-IT: 42.8 vs 35.7 | 更少但更多样的数据可超过更多重复数据 |

### 关键发现
- 单纯提高 temperature 或加“diverse”指令确实比 Default 更好，但提升有限且不稳定；Hierarchical 是很强的 baseline，但需要人工设计主题分解，并且通常 LLM calls 很高。
- Voyager 的优势来自两个方向同时工作：DPP 阻止数据集收缩到已有模式，textual gradients 把失败样本转化为新的探索提示。
- 在推理数据生成上，Voyager 的收益比创意写作更大。这说明它不只是让文本风格变花，还能覆盖更多题型结构。
- 下游训练结果很关键：多样性指标不是孤立好看，GSM8K 上 Gemma 7B 从 Default 数据的 35.7 提升到 Voyager 数据的 45.7，说明生成数据确实更有训练价值。
- LLM 调用是代价。Voyager 比 Default、Temp、Diverse、History 多调用很多次，但在多数任务上比 Hierarchical 的 cost-benefit 更好。

## 亮点与洞察
- 最大亮点是把 DPP 从“后处理筛样本”推进到“生成过程中的控制器”。这比先生成大候选池再筛更主动，也更适合目标数据集较大时使用。
- textual gradients 的用法很实用。被拒样本不是浪费，而是告诉模型“当前 prompt 为什么不够新”，这让探索具备反馈闭环。
- Anchor set 是一个很好的工程折中。它牺牲了全局 determinant 的精确性，但换来固定内存、固定计算和可扩展性。
- 论文没有把质量和多样性混成一个黑盒奖励，而是显式用 LLM-as-judge 单独评估质量。结果显示 Voyager 主要提升多样性，不明显损害质量。
- 对合成数据训练的启发很直接：与其让 LLM 批量生成大量相似题，不如用全局多样性约束生成更少但覆盖更广的数据。

## 局限与展望
- Voyager 依赖强 instruction-following LLM，尤其在 textual gradient extraction 阶段需要模型能理解失败原因并提出有效改写建议。弱模型可能给出空泛反馈。
- 相似度核依赖 embedding model 和 lexical kernel。若 embedding 不能区分任务内关键差异，DPP 优化的“多样”可能与人类需要的多样不一致。
- 论文主要证明 Vendi 和人类 pairwise diversity 提升，但对 diversity type 的细分仍不足。例如数学题多样到底来自数字变化、题型变化、推理链变化还是领域变化，还需要进一步分析。
- 当前只处理文本生成，不涉及图像、音频、视频或多模态样本。多模态下如何定义统一相似度核和边际体积增益会更复杂。
- LLM calls 仍明显高于普通批量生成。对成本敏感场景，需要根据任务价值调节 $\tau$、anchor size 和 explorer beam。

## 相关工作与启发
- **vs temperature / nucleus sampling**: decoding 层面的随机性只能扩大单次生成分布，Voyager 优化的是数据集级别的覆盖。
- **vs prompt-based diversity control**: “generate diverse samples” 或 history prompting 不需要额外算法，但缺少可量化目标，难以保证全局多样性。
- **vs Hierarchical prompting**: Hierarchical 手工把任务拆成子主题，效果强但依赖领域知识；Voyager 用 textual gradients 自动发现后续探索方向。
- **vs SubsetSelect / DPP 后处理**: 后处理要先造大候选池，再筛选；Voyager 把 DPP 放进生成循环，用 anchor set 逐步引导 LLM 生成更不同的样本。
- **对数据合成工作的启发**: 可以把 Voyager 的多样性控制接到 instruction tuning、数学题生成、persona simulation、评测集构造等流程中，尤其适合希望减少重复模式的场景。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ DPP 和 textual gradients 都不是新概念，但组合成 training-free 生成控制算法很自然也很有效。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖创意、推理、人评、消融和下游训练；如果能加入更多开源模型和成本曲线会更完整。
- 写作质量: ⭐⭐⭐⭐☆ 算法和理论动机清楚，实验叙述充分，部分表格与复杂度推导略显拥挤。
- 价值: ⭐⭐⭐⭐⭐ 对合成数据生成非常实用，尤其适合闭源 LLM 场景下追求数据覆盖和训练效率的应用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](../../ACL2025/llm_nlp/token_prepending_training_free.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](../../ACL2025/llm_nlp/a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ICML 2025\] Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets](../../ICML2025/llm_nlp/safe_delta_consistently_preserving_safety_when_fine-tuning_llms_on_diverse_datas.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](../../CVPR2026/llm_nlp/sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)
- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](../../NeurIPS2025/llm_nlp/speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)

</div>

<!-- RELATED:END -->
