---
title: >-
  ICLR2026 优化/理论方向 40篇论文解读
description: >-
  40篇ICLR2026 优化/理论方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🔬 ICLR2026** · **40** 篇论文解读

**[A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)**

:   本文建立了首个在浮点量化下分析自适应优化器收敛性的理论框架，对梯度、权重和优化器状态（动量、二阶矩）同时施加相对误差量化模型，证明了量化 Adam 和 Muon 在尾数长度仅需对数增长于迭代次数时即可保持与全精度相同的 $\tilde{O}(T^{-1/4})$ 收敛率，并揭示了 Adam 对权重和二阶矩量化高度敏感而 Muon 更为鲁棒的理论机制。

**[Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)](adaptive_rollout_allocation_for_online_reinforcement_learning_with_verifiable_re.md)**

:   提出 VIP（Variance-Informed Predictive allocation），通过高斯过程预测每个 prompt 的成功概率，据此用凸优化在计算预算约束下分配 rollout 数量以最小化梯度方差，在数学推理任务上一致提升 GRPO/RLOO 的采样效率，AIME24/25 上 Pass@32 最高提升 12.3 个点。

**[Celo2: Towards Learned Optimization Free Lunch](celo2_towards_learned_optimization_free_lunch.md)**

:   提出 Celo2——一个仅用 4.5 GPU 小时元训练的学习型优化器，通过归一化 MLP 更新规则和任务增强等简单配方，实现了到 10 亿参数级别模型（GPT-3 XL 1.3B）的稳定泛化（比元训练分布大 6 个数量级），性能超越了此前耗费 4000 TPU-month 的 VeLO 和精心调优的 AdamW 基线。

**[CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)**

:   CogFlow 提出认知启发的三阶段视觉数学推理框架（感知→内化→推理），通过 Synergistic Visual Rewards 增强感知、Knowledge Internalization Reward 桥接感知与推理、Visual-Gated Policy Optimization 锚定视觉推理，解决了现有方法中"感知正确但推理漂移"的核心问题。

**[Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)**

:   提出基于约束缩减的 MILP 模型简化框架：定义固定约束强度 $\rho$ 并用信息增益 $\Delta H=-\log\rho$ 识别关键紧约束（CTC），设计融合实例级双部图与抽象级类型图的多模态 GNN 表征来预测 CTC，在 4 个大规模基准上解质量（$\text{gap}_\text{abs}$）平均提升 51.06%、收敛速度（PDI）平均加快 17.47%。

**[Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)**

:   提出 HiSo（Hessian-informed Scalar-only communication），在联邦零阶优化中利用全局对角 Hessian 近似加速收敛，同时严格保持标量通信不传输任何二阶信息。理论证明在低有效秩和白化假设下收敛速率独立于 Lipschitz 常数 $L$ 和模型维度 $d$；实验在 OPT-350M/1.3B/2.7B 微调中实现 1.4~5.4× 通信轮次加速，通信成本仅 KB 级。

**[Convergence of Muon with Newton-Schulz](convergence_of_muon_with_newton-schulz.md)**

:   首次为实际使用的 Muon 优化器（使用 Newton-Schulz 近似而非精确 SVD 极坐标分解）提供非凸收敛保证：证明收敛速率匹配 SVD 理想化版本（差一个常数因子），该因子随 Newton-Schulz 步数 $q$ 双指数衰减，且 Muon 比向量对应物 SGD-M 少 $\sqrt{r}$ 倍秩损失。

**[Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)**

:   从凸优化理论出发，证明深度学习训练损失以 O(1/sqrt(T)) 速率收敛，最优学习率以 1/sqrt(T) 缩放，在 GPT-2 到 12.5B 参数模型上验证了该缩放律（R^2 >= 0.978），并实现了 80 倍训练步数的学习率外推。

**[Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)**

:   提出 FlexQP——基于 $\ell_1$ 弹性松弛的"永远可行"凸二次规划（QP）求解器，结合深度展开（deep unfolding）学习 LSTM 反馈策略加速收敛得到 Deep FlexQP；在 SQP 框架中作为子模块，解非线性轨迹优化比 OSQP 快 4-16 倍，预测安全滤波器的安全违规减少 70%+、任务完成率提升 43%。

**[DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)**

:   提出 DeepAFL，通过设计无梯度的解析残差块并引入逐层联邦训练协议，首次实现了具有表征学习能力的深度解析联邦学习模型，既保持了对数据异质性的理想不变性，又突破了现有解析方法仅限于单层线性模型的局限，在三个基准数据集上超越 SOTA 5.68%-8.42%。

**[Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)**

:   首次证明了梯度下降（gradient descent）在 leaky ReLU 两层神经网络中的方向收敛性（directional convergence），并据此在远超近正交数据（nearly orthogonal data）的更广泛混合数据设定下建立了 benign overfitting 的充分条件，同时发现了一个新的相变（phase transition）现象。

**[Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)**

:   证明了约束深度学习中广泛使用的 dual optimistic ascent（PI 控制）在单步一阶更新体制下数学等价于经典的增广拉格朗日方法（ALM），从而将 ALM 的鲁棒收敛保证（线性收敛到所有严格局部解）转移至 PI 控制，并为乐观系数 $\omega$ 提供了原则性调参指导。

**[Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)**

:   提出 STARS（Stiefel-based Activation Steering for Diverse ReaSoning），一种 training-free 的推理时激活转向方法，在每个 token 解码时于 Stiefel 流形上联合优化 N 条并行生成路径的正交 steering 方向，最大化隐状态的几何体积以促进发散的激活轨迹，在测试用例生成（TestEval）和科学发现（LiveIdeaBench）上以极低延迟一致超越温度采样的多样性，且不损失质量。

**[Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)**

:   通过将 F2SA 方法重新解释为前向差分近似 hyper-gradient，提出利用高阶有限差分的 F2SA-p 方法族，在高阶光滑条件下将随机双层优化的 SFO 复杂度从 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 改进至 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$，并证明了 $\Omega(\epsilon^{-4})$ 下界表明该方法在 $p$ 足够大时近乎最优。

**[FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)**

:   提出 FedDAG 聚类联邦学习框架，通过融合数据和梯度信息进行加权类别级相似度计算来实现更准确的客户端聚类，并通过双编码器架构实现跨集群特征迁移，在多种异构性设置下一致超越现有基线。

**[FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)**

:   FrontierCO 是一个涵盖 8 类组合优化问题（TSP、MIS、CVRP 等）的大规模真实世界基准测试，评估了 16 个 ML 求解器（神经网络方法 + LLM Agent）与 SOTA 传统求解器的差距，发现 ML 方法在结构复杂和极大规模实例上仍显著落后于传统方法，但在部分场景有超越潜力。

**[Generalization Below the Edge of Stability: The Role of Data Geometry](generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)**

:   提出"数据可碎性"（data shatterability）原理统一解释数据几何如何控制梯度下降在稳定性边缘（EoS）附近的隐式正则化强度：对 Beta(α) 径向分布族推导出依赖 α 的泛化上下界谱，对低维子空间混合分布证明泛化率适应内在维度 $m$ 而非环境维度 $d$。

**[Learning to Recall with Transformers Beyond Orthogonal Embeddings](learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)**

:   在随机（非正交）嵌入条件下分析单层 Transformer 在 token 检索任务上经验梯度下降的"早期阶段"，推导出模型存储容量的显式公式，揭示了样本量 N、嵌入维度 d 和序列长度 L 之间的乘法依赖关系，并证明这一缩放关系是信息论下界固有的。

**[Learning to Solve Orienteering Problem with Time Windows and Variable Profits](learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)**

:   提出DeCoST——一种学习式两阶段框架，将OPTWVP中耦合的离散路线决策和连续服务时间分配解耦：第一阶段并行解码器联合生成路径+初始服务时间，第二阶段LP精确优化服务时间(全局最优)，通过pTAR反馈实现跨阶段协调。在50-500节点OPTWVP上优化间隙仅0.83%-3.31%，推理速度比元启发式快最高45倍。

**[Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)**

:   提出马尔可夫语言模型(MLM)框架，通过**结构约束**（答案预测时移除原始问题，仅从CoT推导）强制CoT成为因果必要的推理瓶颈——类似自编码器的窄潜层，配合GRPO风格策略梯度训练，在GSM8K上从19.6%提升到57.1%，且学到的CoT可跨模型架构（Llama→Mistral/Phi/GPT-2）迁移，证明CoT编码了自然语言推理而非隐写术。

**[Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)**

:   深入分析了 SAM 在线性对角网络上训练时的隐式偏差，揭示深度从 $L=1$ 到 $L=2$ 引发的质变：$\ell_\infty$-SAM 的极限方向对初始化高度敏感，$\ell_2$-SAM 则展现出"先弱后强"的**序列特征放大**（sequential feature amplification）现象，指出仅关注 $t\to\infty$ 极限的分析不足以揭示 SAM 的完整动态行为。

**[∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)**

:   提出 ∇-Reasoner，将推理时的搜索从零阶（采样+评估）升级为一阶（梯度下降），在 token logits 空间上通过可微文本优化（DTO）结合 reward 梯度和 LLM 似然来迭代改进解码策略，在数学推理任务上提升 10-40% 准确率的同时减少 10-40% 的模型调用次数。

**[Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)**

:   证明在通用非退化假设下，标准两层神经网络通过分层梯度下降可以用 $\tilde{O}(d)$ 样本和 $\tilde{O}(d^2)$ 时间学习通用高斯 Multi-Index 模型 $f(\bm{x})=g(\bm{U}\bm{x})$，样本和时间复杂度都达到信息论最优，首次证明神经网络可以高效学习层次化函数。

**[Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)**

:   首次建立保形分位数回归（CQR）和保形中位数回归（CMR）在 SGD 训练下的非渐近效率界，明确刻画了预测集长度偏差与训练样本量 $n$、校准样本量 $m$ 和误覆盖率 $\alpha$ 的联合依赖关系。

**[Nonparametric Teaching of Attention Learners](nonparametric_teaching_of_attention_learners.md)**

:   提出AtteNT——从非参教学理论视角重新解释注意力学习器(Transformer/ViT)的训练过程：解析注意力在参数梯度中的重要性自适应角色→证明动态ANTK收敛到功能梯度中的重要性自适应典范核→桥接参数空间与函数空间→用贪心教学算法选择预测偏差最大的样本加速训练→LLM微调省时13.01%/ViT从头训练省时20.58%且精度不降反升。

**[NRGPT: An Energy-based Alternative for GPT](nrgpt_an_energy-based_alternative_for_gpt.md)**

:   提出NRGPT（eNeRgy-GPT），对标准GPT进行最小修改使其成为能量基模型：设计注意力能量和前馈能量函数，使每层前向传播等价于token在能量landscape上的梯度下降步，证明了渐近能量下降和稳定收敛性质，在ListOps/Shakespeare/OpenWebText上验证了与标准GPT可比的性能。

**[Personalized Collaborative Learning with Affinity-Based Variance Reduction](personalized_collaborative_learning_with_affinity-based_variance_reduction.md)**

:   提出个性化协作学习框架 AffPCL，通过偏差校正和重要性校正机制，让异质智能体在无需先验知识的情况下协作学习个性化解，实现 $O(t^{-1} \cdot \max\{n^{-1}, \delta\})$ 的自适应收敛率——智能体相似时获得线性加速，差异大时不差于独立学习。

**[Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)**

:   提出 Πnet 架构，通过在神经网络输出层附加基于 Douglas-Rachford 算子分裂的正交投影层来保证凸约束的严格满足，并利用隐函数定理进行高效反向传播，在训练时间、求解质量和超参数鲁棒性上大幅超越现有方法。

**[Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)**

:   提出 In-Context Policy Optimization (ICPO) 框架，理论证明单层线性自注意力 Transformer 经充分预训练后可在上下文中模拟策略优化算法，并设计实用的 ME-ICPO 算法通过最小熵选择和自评估奖励实现测试时多轮自反思，在数学推理任务上取得显著提升（AIME 2024 上 Qwen2.5-Math-7B 从 11% 提升到 30%）。

**[Rethinking Consistent Multi-Label Classification Under Inexact Supervision](rethinking_consistent_multi-label_classification_under_inexact_supervision.md)**

:   提出 COMES 框架，通过一阶（Hamming loss）和二阶（Ranking loss）策略，为不精确监督下的多标签分类提供一致性风险估计器，无需估计标签生成过程或均匀分布假设。

**[Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)**

:   提出 Rolling Ball Optimizer (RBO)，通过模拟有限半径刚性球在损失景观上的滚动运动来打破传统优化器的空间局部性，实现对损失函数的平滑效应（ironing property），在 MNIST 和 CIFAR-10/100 上展示了更好的收敛速度和泛化性能。

**[RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)**

:   提出 RRNCO 架构，通过自适应节点嵌入（ANE）和神经自适应偏置（NAB）两大创新，首次在深度路由框架中联合建模非对称距离、时长和方向角，并构建了基于 100 个真实城市的 VRP 基准数据集，显著缩小了 NCO 方法从仿真到真实世界部署的差距。

**[RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)**

:   提出 RS-ORT 算法，通过将回归树训练重构为两阶段优化问题并在缩减空间上进行分支定界（仅对树结构变量分支），结合闭式叶预测、阈值离散化和精确末层子树解析等加速策略，首次在包含连续特征的 200 万样本数据集上实现了有全局最优性保证的回归树学习。

**[Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)**

:   提出统一的理论框架，通过 saddle-to-saddle 学习动力学解释多种神经网络架构（全连接、卷积、注意力）中普遍存在的 simplicity bias——即梯度下降倾向于先学习简单解再逐步学习复杂解的现象。

**[Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)**

:   提出 Scaf-GRPO 框架，通过分层级的 in-prompt hint 注入（知识→规划→解题步骤）来克服 GRPO 训练中"学习悬崖"(zero-reward)问题，在 Qwen2.5-Math-7B 上将 AIME24 的 pass@1 相对提升 44.3%，同时保持 on-policy 训练一致性。

**[Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)**

:   在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑，并证明在噪声主导的情形下 SignSGD 的计算最优斜率可以超过 SGD。

**[Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)**

:   提出 MASS（Meta-Adaptation with Self-Synthesis）框架，通过双层优化元学习让 LLM 在推理时生成问题特定的合成训练数据（Generator）+ 评分筛选（Scorer）+ 加权 SFT 自更新（LoRA），meta-gradient 反向传播穿过内更新以优化数据质量，在 MATH-500 上将 Llama-3.1-8B 从 43.6% 提升至 59.0%。

**[The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)**

:   揭示了梯度下降中参数最速下降方向与传播到激活后的有效更新之间存在根本性不对齐（"仿射散度"$\Delta\mathcal{L}/\Delta z_i = (\partial\mathcal{L}/\partial z_i) \cdot (\|\vec{x}\|^2+1)$），从第一性原理推导出归一化是消除此散度的自然解，并发现一种非归一化的替代方案在实验中超越传统归一化。

**[Unifying Formal Explanations: A Complexity-Theoretic Perspective](unifying_formal_explanations_a_complexity-theoretic_perspective.md)**

:   提出统一框架将充分理由和对比理由（局部/全局、概率/非概率）归结为对统一概率值函数的最小化问题，揭示全局值函数具有单调性、子模性/超模性等组合优化关键性质，从而证明全局解释在多项式时间内可计算——即使对应的局部解释是 NP-hard 的。

**[When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)**

:   提出 SGD-ER（SGD with Escalating Restarts），一种收敛感知的学习率调度策略：当检测到训练停滞时触发重启并线性升高学习率，帮助优化器逃离尖锐局部极小值、探索更平坦的损失景观区域，在 CIFAR-10/100 和 TinyImageNet 上取得 0.5-4.5% 的测试精度提升。
