---
title: >-
  ICLR2026 图学习方向21篇论文解读
description: >-
  21篇ICLR2026的图学习方向论文解读，涵盖图神经网络、LLM、生物分子等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🔬 ICLR2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (8)](../../ACL2026/graph_learning/index.md) · [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (52)](../../NeurIPS2025/graph_learning/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/graph_learning/index.md) · [🧪 ICML2025 (31)](../../ICML2025/graph_learning/index.md)

🔥 **高频主题：** 图神经网络 ×4 · LLM ×3 · 生物分子 ×2

**[A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)**

:   从图 Ricci 曲率的几何视角证明随机 k-SAT 问题的二部图表示具有固有的负曲率，且曲率随问题难度增加而下降，建立了 GNN 过压缩 (oversquashing) 与 SAT 求解困难之间的理论联系，并通过测试时图重布线验证了该理论。

**[Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)**

:   指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量，实验表明Erank与准确率的平均相关性达0.91（vs Dirichlet energy的0.72），在OGB-Arxiv上Dirichlet energy甚至呈现错误的相关方向，并从理论上证明对广泛的GNN架构族其数值秩收敛到1（秩坍塌），重新定义过平滑为秩坍塌而非特征向量对齐。

**[Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)**

:   首次提出针对多重图（multigraph）的神经组合优化路由方法 GMS，包含直接在多重图上边级自回归构造的 GMS-EB 和先学习剪枝再节点级路由的双头 GMS-DH 两个变体，在非对称多目标 TSP 和 CVRP 上实现了接近精确求解器 LKH 的性能且速度快数十倍。

**[Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)**

:   提出在有向图上定义 cellular sheaf 的 in/out-degree Laplacian，构建 Cooperative Sheaf Neural Network (CSNN)，使节点能独立选择信息传播/接收策略，从而同时缓解过度挤压(oversquashing)和处理异配(heterophilic)任务。

**[Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)**

:   本文通过 Memento 框架系统评估了 LLM 驱动具身智能体的记忆利用能力，发现现有 agent 能回忆简单物体语义但无法处理用户行为模式的序列信息，并提出了基于层次知识图谱的用户画像记忆模块来有效提升个性化辅助任务的表现。

**[Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)**

:   提出 EDT-Former（Entropy-guided Dynamic Token Transformer），通过熵引导的动态token生成机制，在冻结图编码器和LLM之间建立高效对齐，无需微调LLM主干网络即在分子问答、分子指令和属性预测等多个基准上达到SOTA。

**[Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)**

:   提出 Explore-on-Graph（EoG），通过 SFT + 两阶段强化学习（结果奖励 + 路径精炼奖励），激励 LLM 在知识图谱上自主探索超出训练分布的推理路径，在五个 KGQA 基准上超越 GPT-5 和 Gemini 2.5 Pro。

**[GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)**

:   提出 GRAPHITE，一种通过引入"特征节点"作为 hub 间接连接共享特征的节点来**直接提升图同质性**的非学习图变换方法，首次从"改变图结构"而非"改变 GNN 架构"的角度解决异质图问题，在 Actor 等困难基准上显著超越 27 种 SOTA 方法。

**[Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)**

:   提出 GraphTokenizer 框架，将图通过可逆的频率引导序列化转换为符号序列，再用 BPE 学习图子结构词汇表，使标准 Transformer（如 BERT/GTE）无需任何架构修改即可直接处理图数据，在 14 个 benchmark 上达到 SOTA。

**[GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)**

:   提出 GraphUniverse 框架，通过分层生成具有持久语义社区的图族（graph families），首次实现对图学习模型归纳泛化能力的系统性评估，揭示了 transductive 性能无法可靠预测 inductive 泛化能力这一关键发现。

**[Learning Concept Bottleneck Models from Mechanistic Explanations](learning_concept_bottleneck_models_from_mechanistic_explanations.md)**

:   提出 Mechanistic CBM (M-CBM)，利用 Sparse Autoencoder 从黑盒模型自身学到的特征中提取概念，再由多模态 LLM 命名和标注，构建可解释的 Concept Bottleneck Model，在控制信息泄露的条件下显著优于现有 CBM 方法。

**[LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)**

:   LogicXGNN 提出了一种从已训练的图神经网络中提取可解释一阶逻辑规则的 post-hoc 框架：通过图结构哈希和隐藏层嵌入模式识别谓词、用决策树确定判别式 DNF 规则结构、并将抽象谓词接地到输入空间，最终生成可替代原始 GNN 的规则化分类器，同时可作为可控的图生成模型。

**[MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation](mollangbench_a_comprehensive_benchmark_for_language-prompted_molecular_structure.md)**

:   提出 MolLangBench 基准，通过自动化工具和专家标注构建高质量、无歧义的分子-语言接口评估数据集，覆盖识别/编辑/生成三类任务和 SMILES/图像/图三种模态，评估 16+ 个商业 LLM 和 5 个化学模型，揭示即使 GPT-5 在基础分子操作上仍显著不足（生成仅 43%）。

**[On the Expressive Power of GNNs for Boolean Satisfiability](on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)**

:   从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族。

**[Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)**

:   提出 HMAGAT，用有向超图注意力网络替代 GNN 的成对消息传递来建模多智能体路径规划中的群体交互，仅用 1M 参数和 1% 训练数据即超越 85M 参数的 SOTA 模型。

**[RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)**

:   提出 RAS 框架，在推理时为每个问题动态构建查询特定的知识图谱，通过迭代检索规划、文本到三元组转换和图增强回答三个阶段实现结构化推理，在 7 个知识密集型基准上对开源和闭源 LLM 分别取得最高 7.0% 和 8.7% 的提升。

**[Relational Graph Transformer](relational_graph_transformer.md)**

:   提出 RelGT，首个专为关系型数据库设计的图 Transformer，通过多元素 Token 化（特征/类型/跳距/时间/局部结构 5 元组）和局部-全局混合注意力机制，在 RelBench 基准的 21 个任务上一致超越 GNN 基线，最高提升 18%。

**[Relatron: Automating Relational Machine Learning over Relational Databases](relatron_automating_relational_machine_learning_over_relational_databases.md)**

:   系统比较关系深度学习（RDL/GNN）和深度特征合成（DFS）在关系数据库预测任务上的性能，发现两者各有优势且高度任务依赖，提出 Relatron——基于任务嵌入的元选择器，通过 RDB 任务同质性和亲和力嵌入实现自动架构选择，在联合架构-超参搜索中提升达 18.5%。

**[Revisiting Node Affinity Prediction in Temporal Graphs](revisiting_node_affinity_prediction_in_temporal_graphs.md)**

:   分析为什么简单启发式（持续预测、移动平均）在时序图节点亲和力预测上优于复杂 TGNN，证明启发式是线性 SSM 的特例且标准 RNN/LSTM/GRU 无法表达最基本的持续预测，据此提出 NAViS——基于虚拟全局状态的线性 SSM 架构配合排序损失，在 TGB 上超越所有基线。

**[Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)**

:   通过构建 10000 篇论文的配对引用图（人类 vs GPT-4o 生成 vs 随机基线），发现 LLM 生成的参考文献在图拓扑结构上与人类几乎不可区分（RF 仅 60% 准确率），但语义嵌入可有效检测（RF 83%，GNN 93%），说明 LLM 精确模仿了引用拓扑但留下了可检测的语义指纹。

**[Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)**

:   提出 Glot，一种轻量结构感知池化模块，将冻结 LLM 的 token 级隐状态构建为潜在相似性图，通过 GNN 细化后聚合为句子表征，在 GLUE/MTEB 上与微调方法竞争力相当但仅需 20× 更少参数和 100× 更快训练。
