---
title: >-
  [论文解读] Demystifying When Pruning Works via Representation Hierarchies
description: >-
  [ICML 2026][模型压缩][网络剪枝] 论文从"嵌入 → logit → 概率"三段表征层次出发，用 Taylor 局部展开理论证明：剪枝对嵌入空间和 logit 空间的扰动天生很小…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "网络剪枝"
  - "生成任务退化"
  - "softmax 放大"
  - "表征层次"
  - "KL 散度"
---

# Demystifying When Pruning Works via Representation Hierarchies

**会议**: ICML 2026  
**arXiv**: [2603.24652](https://arxiv.org/abs/2603.24652)  
**代码**: 论文中提及"available in the project repository"但未给出公开链接  
**领域**: LLM 模型压缩 / 网络剪枝 / 表征分析  
**关键词**: 网络剪枝, 生成任务退化, softmax 放大, 表征层次, KL 散度

## 一句话总结
论文从"嵌入 → logit → 概率"三段表征层次出发，用 Taylor 局部展开理论证明：剪枝对嵌入空间和 logit 空间的扰动天生很小，但 softmax 这一非线性步骤会按 $\mathrm{Var}_r(\Delta z)/(2T^2)$ 把扰动放大到概率空间，再经过自回归解码的步间累积，最终导致生成任务崩溃；而非生成任务因为只依赖候选 token 子空间，对剪枝天然鲁棒——这统一解释了为什么剪枝在 MMLU、retrieval 上几乎无损但在 GSM8K、HumanEval 上骤降到 0。

## 研究背景与动机
**领域现状**：随着 LLM 规模膨胀，网络剪枝（Wanda、SparseGPT、ShortGPT、Attn/MLP Drop 等）成为压缩的主流方案。intra-layer 路线对单层做稀疏化（unstructured / 2:4 / 4:8），inter-layer 路线则直接砍掉某些 transformer block 或 attention/MLP 子层。这些方法在 retrieval、多选 QA、文本分类等"非生成任务"上都被证明可以接近无损保留性能。

**现有痛点**：但实际部署中出现一个反复被观察到的怪现象——同样的剪枝模型，在 MMLU 上几乎不掉点，在 GSM8K / HumanEval / NarrativeQA 上却直接崩到 0（如 Mistral-7B-Instruct 砍 8 个 MLP 层后 GSM8K 从 48.4 → 0.0、HumanEval 从 4.9 → 0.0、MBPP 从 13.8 → 0.0）。但是没有理论解释这个"任务依赖的脆弱性"从何而来，工业界只能靠经验避雷。

**核心矛盾**：现有解释把锅推给"生成任务输出空间维度大（vocabulary $|\mathcal{V}|$ 远超嵌入维度 $d$ 或候选数 $k$）"或"自回归累积"，但这些是直觉描述，无法定量预测；更关键的是没回答"嵌入扰动那么小，是怎么变成概率上的灾难性偏移的"。

**本文目标**：(1) 把 LLM 推理沿信息流拆成三段表征空间（embedding $h$ / logit $z$ / probability $p$）逐一量化扰动；(2) 给出能解析预测剪枝对各空间冲击的闭式公式；(3) 解释非生成任务为何鲁棒、生成任务为何脆弱；(4) 给出实操指导。

**切入角度**：作者关注一个非常具体的细节——同样的剪枝后 $\Delta h$，在 logit 空间 $\Delta z = W \Delta h$ 是线性变换（旋转 + 拉伸），但在概率空间 $\Delta p = \mathrm{softmax}(z + \Delta z)/T - \mathrm{softmax}(z)/T$ 经过非线性指数归一化后会被极大放大；自回归解码进一步把单步小误差变成多步累积。

**核心 idea**：把剪枝表现的"任务依赖"归因于**表征层次的扰动传播差异**——线性层（embedding → logit）几乎保持相似度，softmax 非线性层是真正的放大器，多步解码是放大器的"循环喇叭"；而非生成任务只关心 logit 顺序或小候选子空间，从不暴露在这个放大循环里。

## 方法详解

### 整体框架
论文不提新的剪枝算法，而是建立一个**分析框架**：(1) 把 LLM 推理沿 $e \to h^{(l)} \to z \to p$ 分成三个表征空间；(2) 对每层独立施加剪枝得到 $\Delta h$、$\Delta z$、$\Delta p$，用余弦相似度和 KL 散度量化各空间的扰动幅度；(3) 用二阶 Taylor 展开导出每个空间的扰动闭式表达；(4) 把单步分析扩展到多步生成，分析误差累积；(5) 把多选任务进一步切到"候选 token 子空间"看局部稳定性。代表性剪枝方法用 Wanda / SparseGPT（intra-layer）与 ShortGPT / Attn-Drop / MLP-Drop（inter-layer），代表模型用 Qwen-2.5-7B-Instruct 和 Mistral-7B。

### 关键设计

1. **三空间扰动测量协议**：

    - 功能：分离剪枝对"嵌入 / logit / 概率"三种内部表征的影响，避免把不同空间的扰动混在一起算。
    - 核心思路：在 baseline 模型 forward 过程中，对当前层用其剪枝后版本替换（其它层保持原状）得到扰动 $\Delta h_l$，用 angular deviation $1-\mathrm{CosineSim}(h_l, h_l+\Delta h_l)$ 量化嵌入空间偏移；通过 LM head 把它投影到 logit 空间 $z^{(l)}=W h^{(l)}$ 测 $1-\mathrm{CosineSim}(z, z+\Delta z)$；再经 $p^{(l)}=\mathrm{softmax}(z^{(l)}/T)$ 得到概率空间偏移。每层 × 每解码步重复，画出 Figure 4 的三条曲线。实证发现：embedding 和 logit 空间的余弦相似度都几乎是 1（只在第一、最后一层略掉），概率空间却剧烈震荡；这就把"问题在哪个阶段被放大"用一个对照实验直接钉死。
    - 设计动机：以往工作要么只看权重稀疏度、要么只看最终 perplexity，掩盖了内部传播规律。论文这种"每层只换它一层、其它保持原样"的隔离设计相当于一个 controlled probe，能干净地把"层级局部扰动"和"端到端累积"分开。

2. **Taylor 局部理论 (Theorem 1-3)**：

    - 功能：把上述实证现象用闭式公式解释，回答"为什么 logit 空间稳但概率空间不稳"。
    - 核心思路：嵌入/logit 空间余弦相似度可用二阶 Taylor 展开近似为 $1-\mathrm{CosineSim}(h, h+\Delta h) \approx \|\Delta h_\perp\|^2 / (2\|h\|^2)$，只取决于"正交分量与原向量模长的平方比"；由于单层剪枝引入的 $\|\Delta h\|$ 本来就远小于 $\|h\|$，所以这个比值很小，加上 LM head 投影后还会进一步缩小相对正交分量（Fig. 5 实测确认）。**关键放大点在 softmax**：概率空间 $1-\mathrm{CosineSim}(p, p+\Delta p) \approx \mathrm{Var}_r(\Delta z)/(2T^2)$，其中 $r_i = p_i^2/\|p\|^2$；分布偏移用 KL 散度则有 $\mathrm{KL}(p\|q) \approx \mathrm{Var}_{i\sim p}(\Delta z_i)/(2T^2)$。注意这里关键的是 **$\Delta z$ 的方差**而不是它的模长——即使 $\Delta z$ 整体不大，只要它在 vocab 维度上分布不均匀（方差大），softmax 就会把这种"扁平 vs 尖峰"差异指数级放大。温度 $T$ 是直接的分母——温度越小放大越猛。
    - 设计动机：这套理论第一次给"softmax 放大剪枝误差"提供了可计算、可对比的标尺；Fig. 6 验证理论估计的 angular deviation 和 KL 散度都和 ground truth 高度吻合。这意味着可以在不实际生成的情况下，从单层扰动统计直接预测某次剪枝会不会让生成任务崩——具备工程指导价值。

3. **生成 vs 非生成的子空间机制 (Multi-Scale Analysis)**：

    - 功能：解释为什么概率空间剧烈震荡但多选/检索任务仍然鲁棒。
    - 核心思路：生成任务每步从完整 vocab $|\mathcal{V}|$ 采样 + 自回归，单步小偏差通过 KV cache 不断喂回历史，导致 baseline 和 pruned 模型从第二步开始就 condition 在不同 token 历史上，偏差爆炸式累积（Fig. 7：第一步余弦相似度 ~1，第十步掉到接近 0）。非生成任务则只在第一步 + 仅看 logit 排序 / 候选 token 子集 $\mathcal{C}\subset\{1,\dots,|\mathcal{V}|\}$（如 A/B/C/D 四个选项）。Fig. 8 显示候选 token 通常在概率分布的**尾部**，那里相对扰动比 top-token 小得多，argmax 几乎不变；检索任务直接在 embedding 空间算 cosine，本来就稳。这就把"任务鲁棒性"机械地映射到"用哪个表征空间 + 用多大子空间 + 几步"三个维度。
    - 设计动机：这一步把"宏观任务表现"和"微观表征几何"打通，提出三个实用 takeaway：表征空间选哪一层、任务相关子空间是否低维、是否有时间依赖——这三条直接是剪枝可行性的预测变量。

### 损失函数 / 训练策略
本文是 training-free 分析，不涉及训练损失。所有剪枝方法（Wanda, SparseGPT, ShortGPT, Attn-Drop, MLP-Drop）按各自原始协议运行；实验主要做 forward 测量而非 fine-tune。

## 实验关键数据

### 主实验
Mistral-7B 在 inter-layer pruning（砍 8 个 attention 层 Drop-8A 或 8 个 MLP 层 Drop-8M）下的非生成 vs 生成任务对比：

| 任务类型 | 任务 | Full (7.1B) | Drop-8A (6.8B) | Drop-8M (5.7B) |
|----------|------|------|------|------|
| 检索 (E5-Mistral) | Avg of 13 BEIR | 58.9 | 53.4 | 56.8 |
| 多选 | BoolQ | 85.9 | 86.0 | 78.2 |
| 多选 | MMLU | 62.1 | 62.0 | 59.1 |
| 多选 Avg | 5 任务 | 69.3 | 69.8 | 64.3 |
| 生成 | GSM8K | 48.4 | 36.2 | **0.0** |
| 生成 | HumanEval | 4.9 | **0.0** | **0.0** |
| 生成 | MBPP | 13.8 | 0.4 | **0.0** |
| 生成 | NarrativeQA | 16.3 | 9.6 | 2.0 |
| 生成 Avg | 5 任务 | 22.3 | 13.2 | **0.8** |

Drop-8M 在多选 Avg 只掉 5 个点，生成 Avg 直接从 22.3 崩到 0.8（97% 退化）。

### 消融实验
理论估计 vs 实际测量的吻合度（Fig. 6，Qwen-2.5-7B 第 14 层 attention 剪枝）：

| 度量 | 理论 vs 实测 | 说明 |
|------|--------------|------|
| Angular deviation $\Delta p$ | 紧贴 | $\mathrm{Var}_r(\Delta z)/(2T^2)$ 公式准确 |
| KL divergence $p\|q$ | 紧贴 | $\mathrm{Var}_{i\sim p}(\Delta z_i)/(2T^2)$ 公式准确 |
| Embedding 余弦相似度 | 几乎 1.0 | 单层 $\|\Delta h\| \ll \|h\|$ |
| Logit 余弦相似度 | 几乎 1.0 | LM head 进一步压缩相对正交分量 |
| Probability 余弦相似度 | 大幅波动 | softmax 非线性放大方差 |

生成过程中的偏差累积（Fig. 7，Drop-8A on Qwen-2.5-7B）：

| 解码步 | 嵌入/logit 相似度 | 概率相似度 | 备注 |
|--------|-------------------|------------|------|
| 1 (prompt 内) | ~1.0 | 较低但可控 | 两模型 condition 相同 |
| 2-3 | ~0.95 | 急剧下降 | 历史 token 开始分歧 |
| 10+ | < 0.5 | 接近 0 | 完全发散，输出乱码 |

### 关键发现
- **关键放大点不是 LM head 而是 softmax**：很多人直觉以为 $z = Wh$ 这种 vocab 维度爆炸会放大扰动，但实测 logit 空间余弦相似度和 embedding 几乎一致——线性变换实际上压缩了相对正交分量。真正的放大器是后面那一步 $\mathrm{softmax}(z/T)$，因为 $\mathrm{Var}_r(\Delta z)/(2T^2)$ 显式依赖 $\Delta z$ 在 vocab 维度上的方差和温度倒数。
- **候选 token 子空间是天然防护罩**：多选题答案 token 通常在分布尾部，那里概率值本身就小、扰动绝对幅度也小，argmax 几乎不受顶端 token 概率震荡的影响。这解释了为什么 MMLU 在 5.7B 模型下仍能保持 59.1。
- **自回归不是元凶，但它是放大器的回声室**：单步剪枝引入的 $\Delta z$ 即使方差中等，自回归把它从单步推向多步、把 KV cache 的状态差异放大成 token 序列差异，最终生成完全发散。Table 2 那个 "ILUNNIE M ` <%=>t..." 的乱码就是这种回声的可视化。
- **温度 $T$ 不仅影响生成多样性，也直接调控剪枝鲁棒性**：公式里 $T^2$ 在分母，温度越低（输出越尖锐）剪枝越脆弱；这给"低温部署 + 剪枝"的组合提了一个红色警告。

## 亮点与洞察
- **理论 + 实证 + 任务表现"三角闭环"**：先用 controlled probing 暴露三空间扰动差异，再用 Taylor 展开给出公式，再用任务级 benchmark 验证预测——三层论证互相印证，是少见的"剪枝分析"做到这么干净的工作。
- **把"任务鲁棒性"分解成三个工程可控变量**：表征空间（嵌入 / logit / 概率）+ 任务相关子空间维度 + 时间依赖——任何剪枝方案都能用这三条来预测它在新任务上的可行性，比"试一下看 ppl"高效得多。
- **`Var_r(Δz)/(2T²)` 这个公式有可操作性**：因为它只需要单层扰动统计，可以用来在剪枝过程中早停或调整剪枝率；不像通常需要跑完整生成才能评估。
- **统一了 pruning 和 quantization 的失败模式**：论文在 Appendix I 指出量化也是 compression-induced error，可用同一套理论。这种"用更基本的扰动数学"统一相邻问题的视角值得借鉴。

## 局限与展望
- 分析框架完全 training-free，没讨论后训练或剪枝微调如何修复 softmax 放大——而工业上几乎所有 pruned model 都会过一遍 SFT/distillation 才上线，理论与实践的桥还差一段。
- Taylor 一阶/二阶展开只在"局部、单层扰动"成立；多层联合剪枝、第一/最后一层的剧烈扰动场景里，理论估计与实测的偏差需要更精细的边界。
- 实验主要在 Qwen-2.5-7B、Mistral-7B 这两类 dense LLM 上，没覆盖 MoE 模型（每步只激活部分 expert）、状态空间模型（Mamba）等结构；这些架构下"softmax 放大"是否仍是主要瓶颈不清楚。
- 把答案 token "在分布尾部"作为多选鲁棒性的解释是经验观察，没给出"什么样的 prompting / 任务格式会让候选 token 跑到 head" 的边界条件——这其实是 MMLU 风格 prompt 工程的一个隐藏假设。
- 没给出"如何选剪枝层让生成任务不崩"的算法级建议（虽然 Discussion 提到 takeaway），如果能给出一个基于 $\mathrm{Var}_r(\Delta z)/(2T^2)$ 的层级排序工具会更实用。

## 相关工作与启发
- **vs ShortGPT / Attn-Drop / MLP-Drop**：这些是被本文当作分析对象的剪枝方法，本文不是替代品而是诊断器，给它们的失败案例提供闭环解释。
- **vs Wanda / SparseGPT**：同样是被分析的代表 intra-layer 方法。本文证明无论 unstructured / 2:4 / 4:8 模式，生成 vs 非生成的分裂都成立。
- **vs Gromov et al. 2024 "Unreasonable Ineffectiveness of Deeper Layers"**：那篇观察到剪掉深层影响小，本文将其升级为"为什么影响小取决于任务用哪个表征空间 + 时间维度"。
- **启发**：这种"用 controlled probe + Taylor 展开 + 任务分解"的分析范式可以推广到其他压缩技术（量化、distillation、early exit）；也提示在设计 LLM 部署 pipeline 时应把 temperature、采样长度、任务输出空间一起作为剪枝可行性的协同变量考虑，而不是只看权重稀疏度。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不提出新算法，但首次把"剪枝任务依赖"用三空间表征 + Taylor 展开的统一框架解释清楚，是分析类工作里的高质量贡献。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 intra-/inter-layer 剪枝、多个 LLM、嵌入/多选/生成三类任务、理论与实测对照；缺 MoE / fine-tune 后场景。
- 写作质量: ⭐⭐⭐⭐⭐ 公式与实验穿插推进，Fig. 4-8 把每个理论点对应到一张图，阅读体验非常友好。
- 价值: ⭐⭐⭐⭐ 对实际部署有直接指导（什么任务能剪、什么温度敏感、为什么要谨慎扩展到生成），但缺少落地的剪枝层选择工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](../../ACL2025/model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ICML 2026\] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](when_shared_knowledge_hurts_spectral_over-accumulation_in_model_merging.md)
- [\[ICML 2025\] From Logits to Hierarchies: Hierarchical Clustering made Simple](../../ICML2025/model_compression/from_logits_to_hierarchies_hierarchical_clustering_made_simple.md)

</div>

<!-- RELATED:END -->
