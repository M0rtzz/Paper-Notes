---
title: >-
  AAAI2026 LLM 评测方向39篇论文解读
description: >-
  39篇AAAI2026的 LLM 评测方向论文解读，涵盖 LLM、推理、对齐/RLHF、Agent、压缩/编码等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "对齐/RLHF"
  - "Agent"
  - "压缩/编码"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**🤖 AAAI2026** · **39** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (79)](../../ACL2026/llm_evaluation/index.md) · [📷 CVPR2026 (25)](../../CVPR2026/llm_evaluation/index.md) · [🔬 ICLR2026 (53)](../../ICLR2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (79)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

🔥 **高频主题：** LLM ×5 · 推理 ×4 · 对齐/RLHF ×3 · Agent ×2 · 压缩/编码 ×2

**[Axis-Aligned Document Dewarping](axis-aligned_document_dewarping.md)**

:   提出利用平面文档固有的"轴对齐"几何性质，在训练、推理和评估三个阶段系统性地引入轴对齐约束，实现了SOTA文档矫正效果并提出新评估指标AAD。

**[BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)**

:   本文构建了一个覆盖加拿大BC省2.4亿公顷、跨度25年的多模态野火风险预测数据集BCWildfire，包含38个驱动因子，并对CNN/Linear/Transformer/Mamba四大范式的时序预测模型进行了系统评测，揭示了当前模型在野火预测中的性能上限和关键影响因子。

**[Benchmarking LLMs for Political Science: A United Nations Perspective](benchmarking_llms_for_political_science_a_united_nations_perspective.md)**

:   提出 UNBench，首个基于联合国安理会 1994-2024 年记录的综合性政治科学 LLM 评测基准，涵盖决议起草、投票模拟、通过预测和代表发言生成四个关联任务，评估 LLM 对复杂政治动态的理解和模拟能力。

**[Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)**

:   借鉴心理学的认知负荷理论（CLT），将工具使用任务的复杂度分解为内在负荷（任务解题路径的结构复杂度）和外在负荷（问题表述的歧义性），构建可参数化调节认知负荷的 ToolLoad-Bench 基准，用指数衰减模型 $\text{Acc} \approx e^{-(k \cdot CL + b)}$ 精确刻画不同 Agent 的能力边界。

**[Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)**

:   提出 MA-CLIP，发现并利用 CLIP 图像特征的**幅度信息**作为感知质量的互补线索，结合余弦相似度实现无需训练的自适应双线索融合图像质量评估。

**[Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering](break_the_tie_learning_cluster-customized_category_relationships_for_categorical.md)**

:   提出 DISC 方法，为每个聚类簇学习定制化的属性类别关系（而非全局统一距离），通过关系树建模与聚类联合优化，在 12 个数据集上以平均排名 1.25 大幅超越现有最佳方法（5.21）。

**[ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)**

:   提出 ConInstruct 基准，评估 LLM 在指令包含冲突约束时的检测和解决能力，发现多数专有模型能较好检测冲突但很少主动告知用户，其中 DeepSeek-R1 和 Claude-4.5-Sonnet 在冲突检测上表现最佳（F1 分别达 91.5% 和 87.3%）。

**[DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)**

:   提出DcMatch——一种无监督多形状匹配框架，通过形状图注意力网络捕捉形状集合底层流形结构以构建更具表达力的共享宇宙空间，并在空间域和谱域实施双层循环一致性约束，在多个基准数据集上实现全面超越。

**[Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)**

:   提出 DIMVC-HIA，一个集成层次化填充与双重对齐的深度不完整多视图聚类框架，先填充缺失聚类分配再填充缺失特征，在高缺失率（70%）下仍保持稳健性能。

**[DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)**

:   提出 DiCaP（Distribution-Calibrated Pseudo-labeling），通过估计伪标签的后验正确率来校准权重、引入双阈值机制分离置信区间和模糊区间并采用不同策略，在半监督多标签学习中以最高 4.27% 的幅度超越 SOTA。

**[GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations](gazeinterpreter_parsing_eye_gaze_to_generate_eye-body-coordinated_narrations.md)**

:   提出 GazeInterpreter，一种基于 LLM 的层次化框架，通过符号化眼动解析器将原始注视信号转化为文本叙述，再与身体运动叙述整合生成眼-体协调描述，并通过自我纠正循环迭代优化，显著提升文本驱动的运动生成、动作预测和行为摘要等下游任务的性能。

**[GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)**

:   针对 GDBA 在一般值域 DCOP 上表现不佳的问题，本文系统分析了三大病因（过于激进的违反条件、无界惩罚累积、不协调的惩罚更新），提出了 DGLS 框架，通过自适应违反条件、蒸发机制和同步方案全面释放引导式局部搜索的性能，在多种标准基准上大幅超越 SOTA。

**[GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)**

:   基于 Neural Collapse 理论，使用固定等角紧框架（ETF）分类器替代动态分类器，通过监督对齐和置信度引导的无监督对齐实现持续泛化类别发现，在四个基准上遗忘率降低 16.1%、新类发现提升 3.2%。

**[GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)**

:   提出一个无需训练的粒度感知对齐框架GranAlign，通过将查询重写为简化版和细化版并分别匹配无关/感知查询的视频描述，解决了零样本视频时刻检索中语义粒度不匹配的核心难题，在QVHighlights上mAP@avg提升3.23%。

**[Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)**

:   提出 BaCa 框架，在测试阶段通过 graphon 估计 + mixup 策略生成边界感知的合成图拓扑，结合双优先队列动态字典和注意力机制自适应校准 OOD 分数，无需微调预训练模型或引入辅助OOD数据，在全部 10 个数据集上超越 GOODAT，平均 AUC 提升 8.37%。

**[HybriDLA: Hybrid Generation for Document Layout Analysis](hybridla_hybrid_generation_for_document_layout_analysis.md)**

:   HybriDLA 首次将扩散式边框精炼与自回归查询扩展统一在一个解码层中，模拟人类由粗到细的阅读策略来处理文档版面分析，在 DocLayNet 上纯视觉模型达到 83.5% mAP，逼近多模态系统。

**[Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)**

:   通过深入分析SPEA2更复杂的选择机制，证明了其种群动态与NSGA-II有本质不同（σ-准则使目标值在种群中均匀分布），从而得到了对种群大小依赖更弱的运行时上界，表明SPEA2对参数选择具有更强的鲁棒性。

**[LLM-as-a-Judge for Scalable Test Coverage Evaluation](llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)**

:   将LLM-as-Judge范式应用于Gherkin验收测试覆盖率评估，在20种模型配置x500次评估中系统量化准确性-可靠性-成本三维权衡，发现GPT-4o Mini以6.07 MAAE、96.6% ECR@1和$1.01/1K评估成为最优生产选择，成本仅为GPT-5高推理版的1/78。

**[Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)**

:   提出 PSN-IRT（Pseudo-Siamese Network for IRT），用增强版项目反应理论同时估计 LLM 能力参数和题目的四参数特征（难度/区分度/猜测率/可行性），在 11 个基准 41,871 题上发现当前基准存在广泛饱和、难度天花板不足、数据污染等系统性问题，PSN-IRT 选出的题目子集排名一致性达 Kendall τ=1.00。

**[Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)**

:   提出 LOREN，一种曲率感知的零阶优化方法，通过低秩块对角预条件器捕获损失景观的各向异性曲率，并结合 REINFORCE Leave-One-Out 方差缩减技术，在 LLM 微调中实现了更高精度和更快收敛，同时相比 MeZO-Adam 节省高达 27.3% 的峰值内存。

**[MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](maps_multi-agent_personality_shaping_for_collaborative_reaso.md)**

:   提出 MAPS 五 Agent 协作推理框架，基于大五人格理论为 4 个功能 Agent 赋予不同"性格"（Interpreter-开放性、Aligner-宜人性、Scholar-尽责性、Solver-外向性）实现异质化协作，加上 Critic Agent（神经质→苏格拉底式反思）做迭代修正，在 MathVista/OlympiadBench/EMMA 上超越 GPT-4o 基线 15.84%，首次超过人类专家 3.58%。

**[MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)**

:   提出MCTS-SQL，让轻量LLM（如Qwen-1.5B）通过蒙特卡洛树搜索实现强大的Text-to-SQL能力——三组件架构（Selector做Schema剪枝 + Direct Generator生成初始SQL + MCTS-Refiner迭代精化），配合前缀缓存机制减少53%推理时间，Qwen-1.5B在BIRD上达40.69%执行准确率（超ChatGPT-3.5）。

**[MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction](microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)**

:   提出 MicroEvoEval，首个面向图像级微观结构演化预测的标准化基准：涵盖 4 个代表性物理任务（平面波、晶粒生长、旋节分解、枝晶凝固）、14 个模型（5 个领域特定 + 9 个通用时空架构）、多维度评估（数值精度 + 物理保真度 + 计算效率），发现现代通用架构（如 VMamba）在长期稳定性和物理保真度上优于领域特定模型，且计算效率高一个数量级。

**[MindVote: When AI Meets the Wild West of Social Media Opinion](mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)**

:   提出 MindVote——首个基于真实社交媒体投票数据的 LLM 舆情预测基准，包含 Reddit/微博上 3,918 个自然投票（23 个话题），附带平台和话题上下文。评估 15 个 LLM 发现：最佳模型（o3-medium）1-Wasserstein 仅 0.892 vs 上界 0.972；在调查数据上微调的专用模型反而不如通用模型（"调查特化陷阱"）；模型表现出强烈文化对齐——西方模型擅长 Reddit、中国模型擅长微博。

**[NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)**

:   提出 NeSTR 神经符号提示策略，通过将自然语言时间事实转化为结构化符号谓词，结合一致性验证和溯因反思修正，在零样本设置下让 LLM 实现高质量时间推理，GPT-4o-mini 上平均 F1 达 89.7（相比 vanilla 64.9 和 TISER 85.8）。

**[OptScale: Probabilistic Optimality for Inference-time Scaling](optscale_probabilistic_optimality_for_inference-time_scaling.md)**

:   提出概率最优框架 OptScale，通过建模验证器分数的概率分布推导出最优采样数量的理论下界，动态决定每个问题所需的最少采样次数，在保持推理准确率的同时大幅减少计算开销。

**[Perspective from a Broader Context: Can Room Style Knowledge Help Visual Floorplan Localization?](perspective_from_a_broader_context_can_room_style_knowledge_help_visual_floorpla.md)**

:   提出利用房间风格知识（通过无监督聚类预训练获得的 room discriminator）来消除视觉楼层平面图定位中因重复结构导致的歧义，在 Gibson 和 Structured3D 两个标准基准上取得 SOTA 性能。

**[RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)**

:   提出 RefineVAD 框架，通过运动感知时序注意力重校准（MoTAR）和类别导向特征精炼（CORE）两个模块，联合建模时序运动动态与异常类别语义，在弱监督视频异常检测任务上实现了对异常事件的精准定位与可解释检测。

**[Regular Games – an Automata-Based General Game Playing Language](regular_games_--_an_automata-based_general_game_playing_language.md)**

:   提出 Regular Games (RG) 通用博弈系统，以非确定性有限自动机（NFA）为核心描述博弈规则，配合多层次语言（底层 RG + 高层 HRG + 专用框架），在表达力覆盖所有有限回合制博弈（含不完全信息和随机性）的同时，生成的前向模型效率全面超越现有最快的通用博弈系统 RBG，通常比 Ludii 快 10-20 倍。

**[Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)**

:   提出 SC-SSL 框架，通过引入**扩展分类器**进行解耦采样控制来缓解特征级不平衡，并利用线性层**偏置项**作为优化偏差向量在推理时直接校准 logits，在多种数据分布设定下达到 SOTA。

**[SpikCommander: A High-Performance Spiking Transformer with Multi-View Learning for Efficient Speech Command Recognition](spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)**

:   提出 SpikCommander，一种全脉冲驱动的 Transformer 架构，通过**多视图脉冲时序感知自注意力（MSTASA）**和**脉冲上下文精炼 MLP（SCR-MLP）**联合增强时序与通道特征建模，在 SHD/SSC/GSC 三个基准上以更少参数超越 SOTA SNN 方法。

**[Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version](streaming_generated_gaussian_process_experts_for_online_learning_and_control_ext.md)**

:   提出 SkyGP（Streaming Kernel-induced Progressively Generated Expert GP），通过**核距离驱动的渐进式专家生成**和**时间感知可配置聚合**处理流数据，继承精确 GP 的学习保证同时保持有界计算复杂度，在基准测试和实时控制实验中全面超越 SOTA。

**[Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)**

:   提出 SLGM 框架，通过**结构化输入格式**、**格式损失**和**格式感知解码**三大组件，将生成式语言模型的结构化预测任务重构为分类问题，在不增加模型参数的前提下显著提升 <1B 模型在 NER、RE、SRL 等 5 类 13 个数据集上的结构预测性能。

**[Test-time Diverse Reasoning by Riemannian Activation Steering](test-time_diverse_reasoning_by_riemannian_activation_steering.md)**

:   提出 SPREAD 框架——一种无监督的测试时激活引导策略，通过在球面流形乘积上求解黎曼优化问题来最大化多条推理路径的隐藏激活张成的总体积，从而提升 Best-of-N 采样中的推理多样性和准确率，在数学推理基准上超越温度采样基线。

**[Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution](think_how_your_teammates_think_active_inference_can_benefit_decentralized_execut.md)**

:   提出 AIM（Active Inference Modeling）框架，在去中心化多智能体强化学习中，不依赖通信机制，仅基于局部观测建模队友的主动推理过程（感知-信念-动作三重肖像），并通过准确性-相关性双重过滤机制选择性融合队友信念，在 SMAC、SMACv2、MPE 和 GRF 四大基准上取得最优或接近最优表现。

**[Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)**

:   本文系统回顾了"自动形式化"（autoformalization）在数学证明、逻辑推理、规划和知识表示等领域的现有工作，提出了一个统一的跨学科定义框架，将自动形式化定义为从非形式语言到形式推理语言的语义等价转换，旨在促进不同研究社区间的方法共享并加速下一代 AI 推理系统的发展。

**[Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)**

:   本文首次为 NSGA-III 在经典双目标 OneMinMax 基准上建立了紧致运行时界 $\Theta(n^2 \ln n / \mu)$，揭示了 NSGA-III 的种群动态特性，并证明其在适当种群规模下优于 NSGA-II。

**[TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization](trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization.md)**

:   提出TRACE，一种基于注意力序列学习的可迁移概念漂移检测器，通过统计特征标记化和双注意力编码器学习跨任务可迁移的漂移模式，能泛化到未见过的数据集，并作为即插即用模块嵌入流式数据驱动优化算法。

**[Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)**

:   提出 SNIC 诊断测试台（9,000 实例/51 场景），评估 LLM 能否利用隐式社会规范来解决歧义参考消解（如"递给我杯子"时存在多个杯子）。发现 LLM 在仅看场景描述时平均准确率仅 44%，加上 Prolog 形式逻辑无显著改善（44.2%），但显式提供规范列表后猛升到 70.5%（GPT-4.1 达 99.6%），证明 LLM 缺乏隐式物理规范知识但能有效利用显式规范。
