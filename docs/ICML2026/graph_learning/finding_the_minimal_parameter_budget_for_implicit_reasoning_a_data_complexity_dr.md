---
title: >-
  [论文解读] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models
description: >-
  [ICML 2026][图学习][隐式推理] 本文从知识图谱补全任务出发，证明并测量了"隐式推理所需的最小参数量"满足一条以**图搜索熵**为复杂度度量的线性 scaling law，每个参数最多支持约 $0.008$ bit 推理信息，颠覆了"模型越大推理越强"的朴素直觉。
tags:
  - "ICML 2026"
  - "图学习"
  - "隐式推理"
  - "最小参数预算"
  - "图搜索熵"
  - "U 形 scaling"
  - "知识图谱补全"
---

# Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models

**会议**: ICML 2026  
**arXiv**: [2504.03635](https://arxiv.org/abs/2504.03635)  
**代码**: https://github.com/WANGXinyiLinda/reasoning-scaling-law (有)  
**领域**: LLM推理 / 预训练 Scaling Law  
**关键词**: 隐式推理, 最小参数预算, 图搜索熵, U 形 scaling, 知识图谱补全

## 一句话总结
本文从知识图谱补全任务出发，证明并测量了"隐式推理所需的最小参数量"满足一条以**图搜索熵**为复杂度度量的线性 scaling law，每个参数最多支持约 $0.008$ bit 推理信息，颠覆了"模型越大推理越强"的朴素直觉。

## 研究背景与动机

**领域现状**：Kaplan、Hoffmann 等人提出的 LM scaling law 都建立在"loss 与参数量单调递减"的经典假设上；Allen-Zhu & Li (2025) 进一步给出了"每参数可存 $2$ bit 知识"的记忆容量 scaling law。绝大多数工作默认：更大的模型 → 更低的 test loss → 更强的能力。

**现有痛点**：这套范式只描述了"记忆"，但**没有刻画"推理"所需要的参数预算**。chain-of-thought、RL post-training 这些方法都是在预训练表示之上的"二次加工"，最根本的问题是——预训练阶段到底需要多大的模型才能涌现出推理？现有 scaling law 无法回答；甚至有 inverse scaling、broken scaling law 等反例表明单调假设并不普适。

**核心矛盾**：**记忆是"信息塞入参数"，参数越多越好；推理是"结构内化为函数"，参数过多反而会过拟合到具体三元组而丢失底层规则**。这两种容量被混在一起讨论，是 scaling law 不够精细的根源。

**本文目标**：分解为两个问题。(1) "支持最优隐式推理的最小模型"这个概念在数学上是否良定义、可识别？(2) 如果存在，这个最小尺寸由数据的什么属性决定？是否能给出一条可外推到真实数据的 scaling law？

**切入角度**：把世界知识抽象为**知识图谱**，预训练语料抽象为"图上的三元组流"，推理抽象为"补全未见过但可由规则推出的边"。用随机 ID 替换实体名以剥离 lexical 信号，把问题压到最纯净的"图结构 → 参数预算"映射。

**核心 idea**：用**图搜索熵**（最大熵随机游走在图上的信息率）作为数据复杂度度量，证明最优模型尺寸 $N_\theta^*(G) = \Theta(H(G))$，并在合成图 + 真实图（FB15K-237）上验证。

## 方法详解

### 整体框架

整套方法是一条"数据 → 训练 → 度量 → 理论 → 验证"的闭环：(1) **数据侧**：用基于优先连接的算法生成可控合成知识图谱 $G$，可按规则数 $N_h$、关系数 $N_r$、实体数 $N_e$、可推三元组比例 $\gamma$ 等维度独立扫参；(2) **训练侧**：用 Llama 架构、固定 batch size $1024$，从头预训练不同尺寸 LM，输入是每条三元组（实体/关系全部替换为随机 ID 后按字符 tokenize），目标即下一 token 预测损失 $L(\theta) = \frac{1}{N} \sum_i -\log P_\theta(e_i^h, r_i, e_i^t)$；(3) **评估侧**：在 hold-out 的可推三元组上做 10 选 1 多选题，记录 test loss 与 accuracy；(4) **度量侧**：对每张图算图搜索熵 $H(G)$ 和"最优参数预算" $N_\theta^*(G)$；(5) **理论侧**：证明 $N_\theta^* = \Theta(H)$ 并在 FB15K-237 上做外推验证。

整个 pipeline 最关键的设计是**"用随机 ID + 字符 tokenize"消除所有 lexical 信号**——只有这样，scaling 曲线才不会被预训练 tokenizer / 词频带来的混淆变量污染，U 形规律才能清晰浮现。

### 关键设计

1. **U 形 scaling 与"最优模型尺寸"的形式化**:

    - 功能：把经验上观察到的"test loss 随模型尺寸先降后升"现象，提炼为一个数学上可识别、可收敛的目标量。
    - 核心思路：定义"在预算 $t$ 步内的最佳已达 test loss" $\underline{\ell}_t(\theta, G) := \min_{0 \le s \le t} \ell(\theta_s, G)$（即 early-stopping 后的最优 loss），再定义 $\epsilon$-最优模型尺寸 $N_{\theta,t}^*(G) := \min\{N_\theta : \exists \theta, \underline{\ell}_t(\theta, G) \le \underline{\ell}_t^*(G) + \epsilon\}$。**Theorem 3** 给出收敛性：在一个温和的"间隔条件"下（所有比最优尺寸更小的模型都至少差 $\epsilon + \Delta$），当训练步数 $t \to \infty$ 时 $N_{\theta,t}^*(G) \to N_\theta^*(G)$。即"最小够用模型"这个概念在数学上良定义。
    - 设计动机：把工程上模糊的"sweet spot"变成可证明、可测量、可外推的物理量；并把 U 形曲线与 benign overfitting / double descent 这条理论脉络对接——最优推理出现在"恰好能表达任务的最小模型"，而非最大模型。

2. **图搜索熵 $H(G)$ 作为数据复杂度度量**:

    - 功能：用一个**纯由图结构决定**的标量，量化"在该图上做多跳推理所需的内禀信息量"。
    - 核心思路：考虑图上的最大熵随机游走，邻接矩阵 $A$ 的主特征值为 $\lambda$、对应特征向量为 $\psi$，则平稳分布 $\rho_i = \psi_i^2 / \|\psi\|_2^2$，转移矩阵 $S_{ij} = (A_{ij}/\lambda)(\psi_j/\psi_i)$。将实体-实体转移合并为实体-关系转移 $S^r_{ij} = \sum_k \mathbb{1}[(i,j,k) \in G] S_{ik}$，得到关系熵率 $H^r(G) = -\sum_i \rho_i \sum_j S^r_{ij} \log S^r_{ij}$。最终图搜索熵 $H(G) = N_e \cdot (\log \lambda + H^r(G))$，同时刻画"走到哪个实体"和"走哪条关系"的不确定性。
    - 设计动机：作者明确区分本文与 Allen-Zhu & Li 的"知识熵"——后者度量"生成过程的信息"（适合记忆），前者度量"遍历图的复杂度"（适合推理）。这是为什么得到的 scaling 系数 $0.008$ bit/参数 比记忆容量的 $2$ bit/参数小 $250$ 倍——推理本质上比记忆更"参数饥饿"。

3. **$N_\theta^*(G) = \Theta(H(G))$ 的理论桥接 + 实证验证**:

    - 功能：把图复杂度与参数预算锁定为线性关系，使得"看一眼数据就能预测最优模型尺寸"成为可能。
    - 核心思路：**Theorem 4** 在三条假设下给出 $N_\theta^*(G) = \Theta(H(G))$：(i) 实体用随机 ID 故没有跨实体语义共享；(ii) 有限精度下 $N$ 参数容量为 $O(N)$；(iii) 每个实体的 Bayes 条件分布可被共享基 $B$ 上的稀疏系数 $a_x$ 近似，且 $\|a_x\|_0 \le \alpha H(Y|X=x) + \beta$。证明的核心是总条件复杂度 $C(G) := \sum_x H(Y|X=x)$ 与 $H(G)$ 同阶。实证上，在合成图扫参得到的 $(H(G), N_\theta^*)$ 拟合出 $R^2 = 0.85$ 的回归直线，把真实图 FB15K-237 的 $H$ 代入预测的 $N_\theta^*$ 与实际观察值高度吻合（图 4 中绿点贴在回归线上）。
    - 设计动机：合成图能扫但不够"真"，真实图够真但只有一点。作者用"合成图拟合规律 + 真实图外推验证"的两段式实验设计，既保证了控变量严谨性，又证明了 scaling law 不是合成 artifact。

### 损失函数 / 训练策略

训练目标即标准的下一 token 预测（CE loss）。所有实验固定 batch size $1024$、训练 $10$k 步（作为 Theorem 3 大预算极限的实用近似）、Llama 架构、通过调整 hidden dim 和层数扫模型尺寸（见 Appendix E）。评估时只对 hold-out 的可推三元组打分，10 选 1 多选题以消除生成 ID 格式带来的副作用。

## 实验关键数据

### 主实验

| 设置 | 现象 | 关键数字 |
|------|------|----------|
| FB15K-237 + 随机 ID（图 1 第 3 行） | test loss 呈清晰 U 形；train loss 单调下降 | $N_e = 14{,}505$, $N_r = 237$, $N = 310{,}116$ |
| 合成图扫参（图 3 a-f） | 最优模型尺寸随训练步数稳定、随 $N$ 与 $N_r$ 增大、对 $N_h$ 不敏感 | 6 维消融全覆盖 |
| 合成图 $(H, N_\theta^*)$ 回归（图 4） | 强线性关系 | $R^2 = 0.85$ |
| FB15K-237 外推验证（图 4 绿点） | 真实图落在合成图拟合直线 95% CI 内 | 预测 vs 实际 $N_\theta^*$ 高度吻合 |
| 推理容量 scaling 系数 | 每 $1$ bit 图搜索熵 $\approx 124$ 个参数 | $\approx 0.008$ bit / 参数 |

### 消融实验

| 改动的图属性 | 对最优模型尺寸的影响 | 对推理性能的影响 |
|------|------|------|
| 训练步数 $t$ ↑ | 先减小后稳定（与 Theorem 3 一致） | 缓慢提升至饱和 |
| 三元组数 $N$ ↑ | 增大（经典 scaling） | 提升 |
| 规则数 $N_h$ ↑ | **基本无变化** | 影响准确率但不影响搜索复杂度 |
| 关系数 $N_r$ ↑ | 增大 | 提升（虚假相关减少） |
| 可推比例 $\gamma$ ↑ | 先增后饱和 | 先提升后饱和 |
| 实体数 $N_e$ ↑ | 增大 | 在规则/关系稀疏时反而下降 |

### 关键发现

- **$N_h$ 不影响最优尺寸**这一点最反直觉：规则数多 → 准确率受影响，但**图的搜索复杂度并未改变**，说明 $H(G)$ 抓到了真正决定参数预算的量。
- **推理容量 $0.008$ bit/参数** 与 Allen-Zhu & Li 的**记忆容量 $2$ bit/参数**相差 $\approx 250\times$——推理比记忆贵两个量级以上的参数，提示后续 RL/CoT 优化的真正受益对象其实是"小而准"的模型而非"大而泛"的模型。
- **U 形曲线只在"过度训练"时出现**：大模型并非不会达到最优，只是"不必要"且容易过拟合；这把 U 形 / inverse scaling / broken scaling 等观察统一到了 benign overfitting 框架下。

## 亮点与洞察

- **从"scaling law for memorization"分裂出"scaling law for reasoning"**：把容量按"任务类型"切片，是过去七年 scaling 研究里少有的概念性突破，比纯改拟合形式有意义得多。
- **图搜索熵是一个"可外推"的度量**：真实预训练语料可通过自动 KG 抽取算法 (Zhong et al., 2023) 得到底层图，然后算 $H$、预测最优模型尺寸——这给"先估数据复杂度再选模型"提供了原则化路径，可直接迁移到代码、定理证明、医学 KG 等垂域。
- **随机 ID + 字符 tokenize 是消除 lexical 混淆的关键 trick**：很多 scaling 工作的"噪声"其实来自 tokenizer 与词频，这个 setup 可被其他 controlled scaling 研究复用。

## 局限与展望

- **架构依赖**：Theorem 4 的上界依赖 Transformer 的 attention key-value memory（假设 iii）；对 SSM / Mamba 这类非 attention 架构是否成立未验证，作者也明确点出这是 future work。
- **训练时长有限**：仅训到 $10$k 步作为"无限训练"的近似，最优尺寸的定位仍受离散模型尺寸与早停噪声影响，$R^2 = 0.85$ 而非更高也部分源于此。
- **真实数据只有一点**：仅在 FB15K-237 上做了外推验证；更大、更杂的真实 KG（Wikidata 子图、领域 KG）尚未测试。一个自然的延展是在文本语料上跑自动 KG 构建 → 算 $H$ → 训不同尺寸 LM，闭环验证整套预测流程。
- **"推理"被狭义化为知识图谱补全**：CoT、多步算术、定理证明这些更复杂的推理形式与图搜索熵的关系仍未知，是否能用一个更广义的"任务搜索熵"统一刻画值得探索。

## 相关工作与启发

- **vs Kaplan / Hoffmann 经典 scaling law**：他们刻画的是"loss 随参数单调下降"，本文揭示推理任务下 loss 是 U 形，且最优尺寸由数据复杂度决定而非 compute budget，是对经典 scaling 的实质性补充。
- **vs Allen-Zhu & Li (2025) 知识容量 scaling law**：同样是切片 scaling，他们关心"记忆"得到 $2$ bit/参数，本文关心"推理"得到 $0.008$ bit/参数；两条 scaling law 互补，共同刻画 LM 的容量分层。
- **vs Wang et al. (2024b) 随机游走聚合假说**：同样用 KG 完成任务做 testbed，他们解释 LM 如何聚合 random walk 路径来推理，本文则在此 setup 上量化所需的参数预算，二者形成"机制 + 容量"的接力。
- **vs broken neural scaling law (Caballero et al., 2023) / inverse scaling (Wei et al., 2023)**：那些工作记录现象，本文给出 U 形现象的理论解释（benign overfitting + 间隔条件）并定位到具体的 $N_\theta^*$。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把"推理"从 scaling law 中切片出来，并给出可计算、可外推的复杂度度量
- 实验充分度: ⭐⭐⭐⭐ 合成图 6 维消融 + 真实图外推齐全，但仅一张真实图、训练步数有限
- 写作质量: ⭐⭐⭐⭐⭐ 概念清晰、定理与实验对接严密、关键 trick（随机 ID）的动机解释到位
- 价值: ⭐⭐⭐⭐⭐ 给"为推理任务选择模型尺寸"提供了原则化框架，并把 U 形 / inverse scaling 等零散现象统一到 benign overfitting 理论下

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning](../../ACL2026/graph_learning/craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[ACL 2026\] Comparing Human and Large Language Model Interpretation of Implicit Information](../../ACL2026/graph_learning/comparing_human_and_large_language_model_interpretation_of_implicit_information.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)

</div>

<!-- RELATED:END -->
