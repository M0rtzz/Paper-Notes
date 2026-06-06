---
title: >-
  [论文解读] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts
description: >-
  [ICML 2026][模型压缩][Mixture-of-Experts] 把标准 MoE 中 top-$K$ 专家输出的"加权求和"替换为按一个动态学习出来的 DAG 进行结构化聚合，在几乎不增加路由与参数开销的前提下显著提升 MoE 表达能力与下游推理表现。
tags:
  - "ICML 2026"
  - "模型压缩"
  - "Mixture-of-Experts"
  - "结构化聚合"
  - "DAG"
  - "多步推理"
  - "稀疏路由"
---

# DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts

**会议**: ICML 2026  
**arXiv**: [2606.01062](https://arxiv.org/abs/2606.01062)  
**代码**: https://github.com/JiaruiFeng/DAG-MoE  
**领域**: 模型压缩 / MoE 架构  
**关键词**: Mixture-of-Experts, 结构化聚合, DAG, 多步推理, 稀疏路由  

## 一句话总结
把标准 MoE 中 top-$K$ 专家输出的"加权求和"替换为按一个动态学习出来的 DAG 进行结构化聚合，在几乎不增加路由与参数开销的前提下显著提升 MoE 表达能力与下游推理表现。

## 研究背景与动机

**领域现状**：现代 LLM 普遍以 MoE 解耦参数量与计算量——路由器为每个 token 选 top-$K$ 个 FFN 专家，输出 $y=\sum_{i=1}^{N} g_i(x) E_i(x)$。已有的扩展轴主要集中在两条线：把路由算法做得更准（Expert-Choice、RNN router、load-balance loss 改良），或者把专家粒度做细（fine-grained，$G=d_f/d_r$ 越大组合空间越大）。

**现有痛点**：细粒度路线虽然让 $\binom{N}{K}$ 组合数爆炸（top-2/8=28 vs top-4/16=1820），但 $N$ 翻倍意味着路由侧参数与负载均衡复杂度同步翻倍，SOTA 系统因此不敢用极端细粒度；并且 router、experts 都已被反复优化，进一步刷点的收益越来越薄。

**核心矛盾**：标准聚合形式 $\sum g_i E_i$ 是**置换不变**的——一旦 top-$K$ 集合定下来，输出就由这堆专家的"多重集"唯一确定，专家之间没有顺序，没有交互，更不可能在一层内做多步组合。也就是说 MoE 的第三个核心组件——**聚合**——一直被忽略，导致表达力上界被锁死在 weighted sum 这个函数族里。

**本文目标**：(i) 提出一种比加权求和更强、但不增加路由复杂度的聚合形式；(ii) 给出严格的表达力比较；(iii) 设计一个轻量、可端到端学习的模块来实现这种聚合。

**切入角度**：把选出的 $K$ 个专家看作 DAG 上的节点——每个节点占据**不同的结构角色**，专家输出沿 DAG 边逐层聚合。这样即使专家集合、router 分数完全一样，换一个 DAG 就得到完全不同的输出。对于固定 $K$，可能的 DAG 数随深度指数增长，提供了一个全新的扩展轴。

**核心 idea**：把 MoE 层里那一步置换不变的 weighted sum 替换成一个**逐 token 动态学出来的 DAG 上的结构化聚合**，从而在不动 router、不动专家的前提下放大组合空间。

## 方法详解

### 整体框架
DAG-MoE 把标准 MoE 块拆成两段：(1) 原来的 sparse router 照常选出 top-$K$ 专家、产出 $K$ 个初始节点表征；(2) 一个新增的 **DAG learning module** 接管聚合，迭代 $L$ 次，每次同时学"当前深度节点之间的连边"以及"按这些连边更新表征"，最后在第 $L$ 层把所有节点求和作为该 token 在该层的输出。整段不改 router、不改专家 FFN，因此天然兼容现有训练栈。

### 关键设计

1. **DAG 风格聚合的一般形式化**:

    - 功能：把"top-$K$ 列表 $\bm{k}$"组织成深度 $L$、每层 $n(l)$ 节点的 DAG $G=(\mathcal{V},\mathcal{A})$，节点 $(l,i)$ 的入边集合 $A_i^l$ 指定它从前面哪些节点取值，最后单一根节点 $(L,1)$ 给出该层输出。
    - 核心思路：初始层 $x_i^0 = g_{\bm{k}[i]}(x) E_{\bm{k}[i]}(x)$；中间层 $x_i^l = \mathrm{AGG}(\{x_j^k \mid (k,j)\in A_i^l\})$；输出 $y=\mathrm{AGG}(\{x_j^k \mid (k,j)\in A_1^L\})$。配上单射 $\mathrm{AGG}$（理论上用 MLP+sum/min/max 实现）即可证 Prop 3.1（任意 DAG 可被单射编码）→ Theorem 3.2（DAG-MoE 严格强于标准 MoE）→ Theorem 3.3（单层 DAG-MoE + 一层多头注意力可在 $O(K\log n)$ 输入长度下模拟一次完整动态规划，标准 MoE 做不到，因为它只能做一步聚合）。
    - 设计动机：先把"为什么结构化聚合本质上比加权和强"用 GNN/D-VAE 的工具说透——置换不变性是表达力天花板，DAG 给出了顺序与多步组合，从理论上交代了空间是值得开发的。

2. **轻量 DAG learning module（核心实现）**:

    - 功能：在不知道 ground-truth DAG 的前提下，逐 token 自动学结构并执行聚合。
    - 核心思路：把搜索空间降维——固定 $n(l)=K$ 并只允许 $(l,i)$ 从前一层 $l-1$ 取边，更早的信息由残差携带。每次迭代先归一化降维 $x_{i,\mathrm{input}}^l=\mathrm{LN}(x_i^{l-1})$、$x_{i,\mathrm{down}}^l=W_{\mathrm{down}}^l x_{i,\mathrm{input}}^l$；对每对 $(i,j)$ 拼出候选边特征 $x^l_{(i,j)}=\mathrm{Concat}(x_{i,\mathrm{down}}^l, x_{j,\mathrm{down}}^l)$；学一个软门控 $e^l_{(i,j)} = \sigma(W_{\mathrm{edge}}^l x^l_{(i,j)})$ 控制连边是否生效，节点信息 $\hat{x}^l_{(i,j)} = e^l_{(i,j)} \odot W_{\mathrm{node}}^l x^l_{(i,j)}$；最后 $x_i^l = W_{\mathrm{up}}^l\sum_j \hat{x}_{(i,j)}^l + x_i^{l-1}$，$W_{\mathrm{up}}$ 用零权重初始化稳定早期训练；输出 $y=\sum_{i=1}^K x_i^L$。
    - 设计动机：(i) 把整张邻接矩阵当 sigmoid 软门控学，避开离散结构搜索；(ii) 在低维 $d_g \ll d$ 里学结构、再投回去，把额外参数压到与一个 shared expert 相当；(iii) 残差 + 1/K 归一化解决多节点求和导致的量级漂移与梯度不稳定。

3. **初始节点的 token 残差注入**:

    - 功能：让原始 token 表征 $x$ 始终能在聚合过程中被访问到，避免完全依赖专家输出。
    - 核心思路：$x_i^0 = g_{\bm{k}[i]}(x) E_{\bm{k}[i]}(x) + \tfrac{1}{K} x$，其中 $1/K$ 是为了让所有节点在 $\sum_i x_i^L$ 后总残差贡献仍为 1，匹配 transformer 块外层的 residual stream 量级。
    - 设计动机：消融显示如果不加这个残差或不做 $1/K$ 缩放，训练很容易发散或长期不收敛——作者把它写成"对训练稳定性至关重要"。

### 损失函数 / 训练策略
沿用 Switch Transformer 的 token-choice router + load-balance loss，再叠 router Z-loss 抑制 logits 漂移。基础架构改自 Llama3.1-8B（保留 tokenizer/attention/FFN 形状），训练目标是标准 causal LM。

## 实验关键数据

### 主实验
12B token Pile 预训练对比三档模型（DAG-MoE-s/-m/-l），并把 baseline 加一个 shared expert 让参数严格对齐。40B token 大规模训练用 DAG-MoE-l ($d_g=256$, $L=2$, 699M 参数) vs MoE-l (shared expert $d_r=512$, 同 699M)：

| 数据集 | 指标 | MoE-l | DAG-MoE-l | 改善 |
|--------|------|-------|-----------|------|
| Pile (in-domain) | PPL ↓ | 10.51 | 10.27 | -0.24 |
| Wikipedia (OOD) | PPL ↓ | 21.08 | 20.54 | -0.54 |
| FineWeb-Edu (OOD) | PPL ↓ | 25.38 | 24.69 | -0.69 |
| C4 (OOD) | PPL ↓ | 35.21 | 34.21 | -1.00 |

OOD 上的 gap 显著大于 in-domain，与 Theorem 3.2 的"表达力优势在分布外更需要"是一致的。

### 消融实验

| 配置 | 加参 | ΔPPL ↑ / Eval Loss ↓ | 说明 |
|------|------|----------------------|------|
| Standard MoE | 0 | 0.000 / 2.7168 | 基线 |
| + shared expert | 393K | 0.433 | 同参纯加专家 |
| Chain-of-Experts (CoE) | 393K | 0.480 | 同参迭代式 router |
| **DAG-MoE-s ($L=2$)** | 393K | **0.587** | 结构聚合最强 |
| MLP mixing $d_g=64$ | 98K | -0.0838 (倒退) | 无结构 MLP 混合反而更差 |
| 微调下游 (DAG-MoE-l vs MoE-l) | — | 26.13 vs 24.06 (avg 7 task) | GPQA +6.06、Lambada +3.46、PIQA +3.15 |

### 关键发现
- **结构本身是关键**，而非额外参数：CoE 同参只拿到 0.480，无结构 MLP 反而比 baseline 还差 → 说明 DAG 提供的"顺序、迭代组合"是真正有效的归纳偏置。
- **迭代次数 $L$ 比维度 $d_g$ 更划算**：$L=0\to1$ 与 $L=1\to2$ 都能掉约 0.5 PPL，$L=2\to3$ 边际很小；$d_g=64,L=2$ 比 $d_g=128,L=1$ 更好但参数更少。
- **吞吐代价小**：$L=1$ 仅 1.51% wall-clock 开销，$L=2$ 仅 4.49%，FLOPs 几乎相同。
- **下游 gain 集中在多步推理任务**：GPQA、Lambada、PIQA、BBH 涨幅明显，而 HellaSwag/MMLU 这种偏模式匹配的几乎不变——印证"结构聚合主要帮的是组合性推理"这一定性论断。

## 亮点与洞察
- 第一次把 MoE 的"聚合算子"作为独立的设计轴提出来，而且把它和 GNN 表达力（D-VAE/GIN 那一套）连起来——这条桥同时贡献了 Prop 3.1、Thm 3.2、Thm 3.3 三个层层递进的理论结果，写法非常清爽。
- Thm 3.3"单层 DAG-MoE + 一层 attention 可模拟 DP"是论文里最大胆的论断，但作者很克制地把它写成"existence/capacity result"，并明说不主张学到的 DAG 真的对应任何 DP 程序——这种"理论作动机、实验做证据"的态度值得学。
- 软门控 $e^l_{(i,j)}$ 等价于把整张邻接矩阵当 sigmoid mask 学，跟 NAS / DARTS 的连续松弛是一个味儿，但只在 $K\times K$ 的小图上做，避开了 NAS 常见的搜索代价问题——这种"在最小可行结构空间里做软搜索"的思路完全可以迁移到 prompt routing、adapter selection 等场景。
- "OOD gap > in-domain gap"这种现象在 MoE 文献里相对少见，但用表达力理论解释得通：分布外 token 更可能落到训练时没见过的专家组合，此时结构聚合的多样性优势就放大了。

## 局限与展望
- 当前 DAG 类被人为限制（每层 $K$ 个节点、只能跨相邻深度连边），Prop 3.1 与 Thm 3.3 都要打折扣，只有 Thm 3.2 完全转译——作者承认这是个 gap。
- 怎么"找到最优 DAG"以及"模块怎么才能稳定学到它"基本没碰，目前完全靠 sigmoid 软门控 + 梯度，离离散意义下的最优 DAG 多远是未知数。
- 实验最大才到 699M 参数 / 40B token，离 SOTA MoE LLM（百亿参数 / 万亿 token）还差几个量级，scaling 行为不明；尤其 $L=2$ 的 4.49% 时间开销在更大 scale 下会不会被 sequential 性质放大、是不是 torch.compile 真能抹掉，没给数据。
- AGG 实现选择没有充分消融——理论假设单射 MLP+sum，工程上简化成了 sigmoid 门控 + sum，两者之间的差距没量化。

## 相关工作与启发
- **vs Chain-of-Experts (CoE, Wang 2025)**：CoE 在一层内做"多轮 routing + 增量 refine"，每轮要独立 router，路由代价随轮数线性涨；DAG-MoE 只 route 一次，把多步交给 DAG 模块，本文实验显示同参下 DAG-MoE 比 CoE 多拿 0.107 PPL。
- **vs S′MoRE (Zeng 2025)**：S′MoRE 也搞结构聚合，但结构固定成树、且只作为 PEFT adapter 用；DAG-MoE 把它推广成任意 DAG 且作为骨干，每个 token 能学到不同结构。
- **vs DiEP (Bai 2026)**：DiEP 也用 DAG 但目的是 differentiable expert pruning（压缩方向）；DAG-MoE 反过来用 DAG 增加表达力。
- **vs Fine-grained MoE (He 2024 等)**：细粒度是把 $N$ 做大、组合数靠"选哪些"扩张；DAG-MoE 是把"怎么组合"扩张，两条轴正交，可以叠加使用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 MoE 第三个被忽视的组件——聚合——单独拎出来做表达力扩展，并桥到 GNN 理论。
- 实验充分度: ⭐⭐⭐⭐ 三档模型 + 同参 baseline + CoE/MLP 对照，但最大 scale 仍偏小且只在 Pile 上预训练，缺更大 LLM 验证。
- 写作质量: ⭐⭐⭐⭐⭐ Prop→Thm→Thm 三个理论结果层层递进，"理论是动机、实验是证据"的边界把握得很好，OOD vs in-domain 的解释也漂亮。
- 价值: ⭐⭐⭐⭐ 提供了 MoE 改进的一条几乎免费的新轴（<5% throughput），但 sequential $L$ 在超大规模下的代价还是未知数。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[CVPR 2026\] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling](../../CVPR2026/model_compression/enhancing_mixture_of_experts_specialization_via_cluster_aware_upcycling.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICML 2026\] UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-Adaptive Federated Fine-tuning of Foundation Models](ub-smoe_universally_balanced_sparse_mixture-of-experts_for_resource-adaptive_fed.md)

</div>

<!-- RELATED:END -->
