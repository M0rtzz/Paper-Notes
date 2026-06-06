---
title: >-
  [论文解读] Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback
description: >-
  [ICML 2026][计算生物][符号回归] IGSR 把符号回归拆成"LLM 提议基函数 ψ_j + 逐项影响力分数 Δ_j 剪枝"两步循环，并把这个循环嵌入 MCTS 来搜组合空间，在 6 个生物医学基准和 LLM-SRBench 上同时拿下最佳 MSE 与符号召回…
tags:
  - "ICML 2026"
  - "计算生物"
  - "符号回归"
  - "影响力分数"
  - "LLM 等式发现"
  - "MCTS"
  - "可解释建模"
---

# Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback

**会议**: ICML 2026  
**arXiv**: [2605.29184](https://arxiv.org/abs/2605.29184)  
**代码**: https://github.com/DrShushen/IGSR (有)  
**领域**: 计算生物
**关键词**: 符号回归, 影响力分数, LLM 等式发现, MCTS, 可解释建模

## 一句话总结
IGSR 把符号回归拆成"LLM 提议基函数 ψ_j + 逐项影响力分数 Δ_j 剪枝"两步循环，并把这个循环嵌入 MCTS 来搜组合空间，在 6 个生物医学基准和 LLM-SRBench 上同时拿下最佳 MSE 与符号召回，还在湿实验里发现了 DNA 甲基化与 RNA Pol II 停顿的新关系。

## 研究背景与动机

**领域现状**：传统符号回归（GP-SR、PySR、SINDy）在预设算子库上做演化或稀疏回归，能输出闭式公式但很难处理 $d \gg 20$ 的高维输入；最近一批 LLM 驱动的等式发现方法（D3、ICSR、LLM-SR、LaSR）则用 LLM 的科学先验直接"想"出基函数，把符号回归推到了生物、流行病、药代等复杂场景。

**现有痛点**：所有 LLM-based 等式发现方法都用**全局标量信号**（一般是全局 MSE，或代码执行错误）作为反馈。这等于告诉 LLM "这个公式好/差"，但**不告诉它公式里哪一项在贡献、哪一项在拖后腿**。结果是搜索退化成试错，高度依赖 LLM 的生成先验而非数据本身。

**核心矛盾**：生成（creative proposal）和选择（rigorous pruning）被耦合在同一个标量损失里。LLM 同时承担"想新项"和"判断旧项该不该留"两件事，后者它根本判断不准，容易把统计上重要的项幻觉成无关而删掉。

**本文目标**：(1) 给 LLM 提供**逐项**的细粒度信用分配信号；(2) 让生成和选择**解耦**——LLM 只负责创造，选择交给统计量；(3) 在组合搜索空间里高效平衡探索和利用。

**切入角度**：作者把模型类锁死为关于基函数的**线性模型** $f(\mathbf{x}) = \sum_{j=1}^M w_j \psi_j(\mathbf{x})$，于是每个 $\psi_j$ 的边际贡献天然可量化——一旦定义"去掉这一项后验证 MSE 升了多少"就是 $\Delta_j$，一个直接、便宜、principled 的信号。

**核心 idea**：用**逐项影响力分数 $\Delta_j$ 替代全局 MSE**做反馈，把"propose-and-prune"循环嵌入 MCTS 来探索基函数组合空间。

## 方法详解

### 整体框架
IGSR 的目标是发现稀疏闭式模型 $f(\mathbf{x}) = \sum_j w_j \psi_j(\mathbf{x})$，其中 $\psi_j$ 是由 LLM 提议的可任意复杂的非线性基函数，$w_j$ 由 OLS 拟合。核心是一个三阶段**propose-and-prune 循环**：① "Propose" LLM 智能体读上下文（变量描述、当前活跃项、历史保留/丢弃记录及对应 MSE 影响）生成候选 $\psi_j$；② 把新旧项拼成扩张集合，OLS 拟合得到 $\mathbf{w}$，在验证集上计算每个项的影响力分数 $\Delta_j$；③ 按 $\Delta_j$ 排序保留 Top-$K$ 项，把保留/丢弃结果写进历史 buffer。这个循环再被一棵 MCTS 树包起来——每个节点是一个等式状态，每次扩展子节点就是跑一次 propose-and-prune 循环，节点 reward 为 $-\mathrm{MSE}_{\mathrm{val}}$，用 UCT 平衡探索和利用。

### 关键设计

1. **逐项影响力分数 $\Delta_j$**:

    - 功能：给搜索过程提供**结构感知的细粒度信用分配**，告诉 LLM "等式里这一项到底贡献了多少"。
    - 核心思路：在拟合好的线性模型上，把第 $j$ 项的权重置零（$w_j \to 0$）而保持其他系数不变，定义 $\Delta_j$ 为验证集 MSE 的增量。即 $\Delta_j = \mathrm{MSE}_{\mathrm{val}}(\mathbf{w}_{-j}) - \mathrm{MSE}_{\mathrm{val}}(\mathbf{w})$，是 leave-one-term-out 的留一分析在**结构维度**上的应用（对照传统留一是在数据点维度）。计算只需一次 OLS 解 + 简单代数，几乎零开销。
    - 设计动机：全局 MSE 只能说"好/坏"，无法定位贡献来源；$\Delta_j$ 把模型选择从"猜"变成"测"，即便候选项之间存在共线性或只有交互项（epistasis-like）信号也仍可靠（附录 G.12 压力测试）。

2. **Propose-and-Prune 循环（生成-选择解耦）**:

    - 功能：把 LLM 的创造性提议和数据驱动的项选择拆开，避免 LLM 一边生新项一边幻觉删旧项。
    - 核心思路：默认走**确定性剪枝**——LLM 只生成候选，按 $\Delta_j$ 排序保留 Top-$K$；可选**Agentic 剪枝**(IGSR-Agent) 让第二个 LLM 智能体读 $(\psi_j, w_j, \Delta_j)$ 三元组，把 $\Delta_j \approx 0 \Rightarrow$ drop 当主启发式，同时叠加语义合理性判断。提议端的 prompt 携带历史 buffer，让 LLM 能 in-context 学到"上轮我提的 $\log x_3$ 被 $\Delta=0.01$ 剪了"，避免重蹈覆辙。
    - 设计动机：确定性版本零幻觉、可复现、计算便宜；Agentic 版本牺牲一点稳定性换取领域知识注入能力。作者实验表明**确定性版本就是最稳健的默认配置**——这反过来说明只要给了 $\Delta_j$，LLM 在选择阶段其实是多余的。

3. **MCTS 嵌入搜索（避开局部最优）**:

    - 功能：在组合爆炸的等式空间里系统地平衡探索与利用，避免单链贪心陷入局部最优。
    - 核心思路：每个节点是一个等式状态（一组 $\psi_j$ + 权重），从父节点经 LLM 随机采样可产生**多个**不同后继（如一支探索三角项、另一支探索交互项）；节点 reward 取 $-\mathrm{MSE}_{\mathrm{val}}$，用 UCT $\bar r_i + c\sqrt{\ln N / n_i}$ 选择扩展方向。默认采用**启发式 MCTS**——只把新扩展节点的即时 reward 直接回传（不做完整 rollout），把计算预算花在广度而非深度模拟上。
    - 设计动机：单链 Linear Iterative Refinement 容易卡在某种函数形式偏置里出不来；MCTS 让"试试三角函数 vs 试试交互项 vs 试试指数衰减"可以并行展开，消融显示 MCTS 比线性模式显著改善收敛和最终精度。

### 损失函数 / 训练策略
不是端到端训练而是搜索过程。每次 OLS 拟合用训练集，$\Delta_j$ 和 MCTS reward 都算在验证集上（附录 G.10 验证了复用 validation split 不会引发搜索期过拟合）。LLM 后端测试了 GPT-4o（六大基准）和 GPT-4o-mini（LLM-SRBench），LLM-SRBench 上统一 300k token 预算保证公平对比。稀疏度上限 $K$ 是主要超参。

## 实验关键数据

### 主实验
六大生物医学基准（Lung Cancer 三个变体、COVID-19、RNA Polymerase、Warfarin），25 seeds，GPT-4o：

| 数据集 | IGSR MSE | 最佳白盒基线 MSE | 备注 |
|--------|------|----------|------|
| Lung Cancer | 5.64e-5 | ICL 0.0557（差 3 个数量级） | 干净的肿瘤生长 ODE |
| LC + Chemo | 0.0013 | ICSR 0.688 | 加化疗的耦合 ODE |
| LC + Chemo+Radio | 0.0141 | LaSR 3.97 | 最难的耦合三药动力学 |
| COVID-19 | 5.01e-8 | ICL 9.35e-8 | 流行病模拟，IGSR 与黑盒 RNN 同档 |
| RNA Polymerase | 0.0111 | ICL 0.0119 | 263 维高维真实基因组数据 |
| Warfarin | 0.565 | ICSR 0.497 | 唯一一个 IGSR 不是最优（仍排第二） |
| 平均排名 | **1.17** | ICL 3.83 | IGSR 在白盒中 5/6 第一 |

LLM-SRBench（128 个发现导向问题，GPT-4o-mini，5 seeds）：IGSR 在 NMSE / Acc$_{0.1}$ / Term Recall / Symbolic Accuracy 上全部取得最佳平均排名，ID 和 OOD 测试集都领先；IGSR-Agent 次之。还击败了一组 AFE 基线（AutoFeat、OpenFE、SyMANTIC、CAAFE）在 5/6 数据集上。

### 消融实验
| 配置 | 现象 | 说明 |
|------|---------|------|
| Full IGSR (MCTS + Δ + history) | 全场最佳 | 完整模型 |
| Linear Iterative Refinement（去掉 MCTS） | 收敛慢、易陷局部最优 | 验证搜索结构必要性 |
| 无 $\Delta_j$ 反馈（退化为 ICL） | 排名从 1.17 掉到 3.83 | 影响力分数是核心增益来源 |
| 无历史 buffer | 反复提同样的失败项 | in-context 记忆机制必要 |
| IGSR-Agent vs IGSR | 略差 + 偶发幻觉删项 | 印证"选择不需要 LLM" |

### 关键发现
- **细粒度信号是关键**：把 IGSR 退化为只用全局 loss 的 ICL，性能立刻退到基线水平，说明 MCTS 不是主要因素，$\Delta_j$ 才是。
- **确定性剪枝胜过 LLM 剪枝**：IGSR-Agent 不是更聪明而是更不稳定，证实"生成-选择解耦"的正确切分点是"选择交给统计量"。
- **真湿实验验证**：在 RNA Pol II 停顿建模中，IGSR 不仅复现了已知机制，还提出了 DNA 甲基化与 Pol II 停顿的新关系假设，作者随后用细胞处理 + 测序在湿实验里**支持了这个假设**——这是符号回归方法第一次在 paper 里报告这种级别的科学发现验证。

## 亮点与洞察
- **把"留一分析"从数据维度搬到结构维度**：传统影响力函数（Cook & Weisberg）研究数据点对参数的影响，IGSR 把同一思想用到基函数上，几乎零成本就拿到了直接对应"该不该留这一项"的统计量。这种"换一个轴用经典工具"的思路在很多 LLM-for-X 场景里都可以套。
- **"生成-选择解耦"是 LLM 智能体设计的通用洞察**：用 LLM 做创造性提议没问题，但把选择/打分也交给 LLM 通常会引入幻觉。能用便宜的统计量替代的地方就替代，这是反 over-engineering 的。
- **湿实验闭环**：作者把符号回归从"基准跑分"推到了"真生物学发现"，这套 propose-and-prune + 影响力反馈的架构可以迁到其他需要"假设生成 + 实验验证"的科学领域，如药物作用机制、材料属性预测。

## 局限与展望
- 模型类被锁死为基函数的**线性叠加** $\sum w_j \psi_j$，无法捕捉深度嵌套或循环动力学（不过单个 $\psi_j$ 内部可以非线性，IGSR-TLO 变体允许优化项内参数）。
- 影响力分数本质是**条件留一**（保持其他系数不变），在强共线性候选下会低估分组贡献；作者用压力测试声称仍可靠，但极端情况下可能需要 group-wise 影响力。
- 高度依赖 LLM 的提议质量，对没有强科学先验的"纯数学"等式发现优势会缩小。
- $\Delta_j$ 在验证集上计算，validation split 大小和分布偏差会直接影响选择质量。
- 改进方向：把 $\Delta_j$ 升级为 group-wise 或 SHAP-like 的归因，让 IGSR 能处理强耦合项；把 MCTS reward 换成多目标（精度 + 简洁度 + 物理一致性），让搜索直接朝可解释方向走。

## 相关工作与启发
- **vs LLM-SR / D3 / LaSR**：都用 LLM 提等式但只给标量反馈，IGSR 的差异化在 $\Delta_j$ 这个**结构感知**反馈信号，让搜索从试错变成有方向的逐项修剪。
- **vs PySR / GP-SR / SINDy**：传统 SR 在预设算子库内演化，受限于人工特征工程；IGSR 让 LLM 直接从科学语义中"想出" $\psi_j$，是高维场景下的关键能力差异。
- **vs AFE（AutoFeat / OpenFE / CAAFE）**：AFE 生成增广特征矩阵喂给 GBDT 等下游黑盒，IGSR 直接产出一个可解释的稀疏线性等式，且用 $\Delta_j$ 而非全局 loss 做特征选择，原则上更稳健。
- **vs SHAP / LIME**：都做归因，但 SHAP/LIME 是事后解释黑盒预测，$\Delta_j$ 是**主动**驱动搜索的反馈，定位更直接、成本更低（一次 OLS 解就够）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 影响力分数本身不新（来自 leave-one-out 影响函数），但首次系统地把它用作 LLM 等式发现的反馈信号，并验证了"生成-选择解耦"的设计原则。
- 实验充分度: ⭐⭐⭐⭐⭐ 六大基准 + LLM-SRBench 128 题 + AFE 对比 + 真实湿实验验证，覆盖广且有湿实验闭环。
- 写作质量: ⭐⭐⭐⭐ Table 1 一图概括差异化定位、Algorithm 2 清晰、附录覆盖共线性/数据泄漏等关键质疑，可复现性强。
- 价值: ⭐⭐⭐⭐⭐ 把 LLM 符号回归推到了"真发现新生物学关系"的高度，propose-and-prune + 细粒度反馈的设计模式对所有 LLM-agent 类工作都有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering](tada-bench_a_million-variant_benchmark_for_future-round_discovery_toward_agentic.md)
- [\[ICML 2025\] Aligning Protein Conformation Ensemble Generation with Physical Feedback](../../ICML2025/computational_biology/aligning_protein_conformation_ensemble_generation_with_physical_feedback.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)
- [\[NeurIPS 2025\] Post Hoc Regression Refinement via Pairwise Rankings](../../NeurIPS2025/computational_biology/post_hoc_regression_refinement_via_pairwise_rankings.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](../../ICLR2026/computational_biology/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)

</div>

<!-- RELATED:END -->
