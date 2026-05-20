---
title: >-
  ICML2026 其他方向27篇论文解读
description: >-
  27篇ICML2026的其他方向论文解读，涵盖扩散模型、持续学习、图像修复、对抗鲁棒、异常检测、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "其他"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "持续学习"
  - "图像修复"
  - "对抗鲁棒"
  - "异常检测"
  - "Agent"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🧪 ICML2026** · **27** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (44)](../../CVPR2026/others/index.md) · [🔬 ICLR2026 (74)](../../ICLR2026/others/index.md) · [🤖 AAAI2026 (125)](../../AAAI2026/others/index.md) · [🧠 NeurIPS2025 (147)](../../NeurIPS2025/others/index.md) · [📹 ICCV2025 (48)](../../ICCV2025/others/index.md) · [🧪 ICML2025 (92)](../../ICML2025/others/index.md)

🔥 **高频主题：** 扩散模型 ×2 · 持续学习 ×2

**[Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)**

:   本文形式化了表格增强中的"保真度-效用间隙"问题（生成器优化分布匹配，而增强价值源于低密度区域），提出 TAP 算法通过扩散填补做流形约束提议、策略引导的效用对齐选择、硬约束门控加保守窗口提交，在 7 个真实表格数据集上相比基线最多提升分类精度 15.6%、回归 RMSE 降低 32%。

**[Adaptive Multi-Round Allocation with Stochastic Arrivals](adaptive_multi-round_allocation_with_stochastic_arrivals.md)**

:   本文形式化网络招募为预算约束的顺序控制问题，证明单轮最优分配是贪心的；通过人口水平代理值函数将多轮规划降维到 $O(b^5\log b)$ 复杂度，并给出在模型误差下分解为前沿/人口/逼近三类误差的鲁棒性保证。

**[AI Cap-and-Trade: Efficiency Incentives for Accessibility and Sustainability](ai_cap-and-trade_efficiency_incentives_for_accessibility_and_sustainability.md)**

:   作者借鉴碳排放 cap-and-trade，提出针对 AI 推理 FLOP 的配额-交易市场（AI Allowance），用 KKT 条件证明其能在合理参数下严格减少各公司 FLOP 使用，从而同时缓解大模型时代的能耗与小公司被挤出市场两大问题。

**[Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)**

:   TabCascade 把表格行拆成"低分辨率（类别 + 数值的离散化版本）"与"高分辨率（连续数值）"两段级联：先用 CDTD 学低分辨率联合分布，再用 flow matching 在低分辨率引导下生成数值细节，并通过数据相关耦合 + 可学非线性时间表收紧 transport cost；天然支持缺失值、零膨胀等"混合型特征"的生成，在 12 个数据集上 detection score 比 SOTA 提升 51.9%。

**[Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)**

:   本文提出 Complexity-as-Advantage (CAA)：把"复杂度"重新定义为一族**资源受限观察者**在同一过程上的**后悔（regret）分散程度**，并证明它在 log-loss + Markov 框架下等价于条件互信息原子之和（恰好恢复 excess entropy），在编码视角下等价于过剩描述长度的方差（MDL），从而把 Kolmogorov 复杂度、Bennett 逻辑深度、excess entropy 统一成一个**可计算、可经验估计**的标量谱。

**[Decision Tree Learning on Product Spaces](decision_tree_learning_on_product_spaces.md)**

:   本文把 Blanc et al. (ITCS'20) 对"top-down greedy 决策树启发式"的理论保证从均匀分布推广到**任意乘积分布**，给出 $\exp(\Delta_\mathrm{opt} D_\mathrm{opt}\log(e/\epsilon))$ 大小上界（满二叉树情形严格优于 ITCS'20），且**完全免参数**——不需要预知最优树大小或深度即可运行。

**[Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)**

:   本文研究「节点到达」数据流模型下相关聚类（correlation clustering）代价的近似估计问题：作者提出 C4Approx 算法，用 $O(n^{(3+\alpha)/4}\log n)$ 词的**亚线性**空间和常数遍数得到 $(O(1), n^{1-\alpha})$-近似，并配套两个匹配下界证明多遍与加性误差都不可避免；在真实数据上仅存 2% 节点即达 Pivot 同等效果。

**[From Generalist to Specialist Representation](from_generalist_to_specialist_representation.md)**

:   本文给出第一个完全非参数（无 intervention、无 functional 约束）的两层 hierarchical 可识别性证明：时间-任务结构由 collider 视角下的 CI test 可识别，任务相关 latent 由 sparsity 正则可从 generalist 表示中分离出来。

**[From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)**

:   本文用 LLM 当人口外推器，把 18 个能力维度按"全世界人口正确率"对数刻度 $L=-\log_B p_W$ 校准，并发现 Volume / Attention 维度真实 base $B \gg 10$、Comprehension 维度 $B \approx 1$，揭示现行 AI 与人类的比较其实严重失调。

**[GEM-FI: Gated Evidential Mixtures with Fisher Modulation](gem-fi_gated_evidential_mixtures_with_fisher_modulation.md)**

:   本文针对证据深度学习 (EDL) 在分布外样本上过自信、且单头难以表达多模态认知不确定性的问题，提出三件套 GEM-Core/MIX/FI：用学到的特征能量门控证据、用混合证据头单次推理近似 ensemble、用 Fisher 信息正则稳定混合分配，在 CIFAR-10→SVHN/CIFAR-100 等 OOD 检测上比 DAEDL 强且保持 single-pass。

**[DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](generative_adaptation_of_dynamics_to_environmental_shifts_via_weight-space_diffu.md)**

:   DynaDiff 把"为新环境训练一个预测器"的元学习问题改写成"用扩散模型直接生成完整网络权重"的条件采样问题，借助权重图 + 函数一致性损失 + 动力学感知 prompter，在 4 个 PDE 系统上平均 RMSE 比强基线再降 10.78%。

**[HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)**

:   借鉴 Helmholtz 自由能的物理直觉，把每个领域的提示参数训练出一条"压缩到边界 $\Theta$、对齐到中线 $\Delta$"的能量曲线，推理时再用能量因子 + 距离因子联合加权各领域提示，在 CDDB / DomainNet / CORe50 三个 DIL 基准的未知领域上分别提升 1.76 / 3.12 / 2.57 个百分点。

**[Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)**

:   作者首次研究"从单条 Gaussian Glauber 动力学轨迹"中学习高斯图模型结构的问题，提出两种互补算法：LET-GL（基于 i,i,j,i 窗口的局部边检测、完美并行）和 BTR-GL（在 Dobrushin 条件下用 burn-in/thinning 把轨迹"解相关"成近似 i.i.d. 样本再喂给现成 i.i.d. 学习器），并给出有限样本恢复保证 + 信息论下界 + 一个独立有用的随机扫描高斯 Gibbs sampler 的 TV mixing 上界。

**[Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)**

:   本文提出 LHSD，把 score 模型的对数密度 Hessian 做一个 Hill 型谱滤波只保留近零特征值来数切空间维数，再用 Stochastic Lanczos Quadrature 把 $\mathcal{O}(D^3)$ 的代价压到 $\mathcal{O}(D)$，从而在 3072 维图像空间稳定估计局部内禀维度，并用于诊断扩散模型的训练样本记忆化。

**[Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)**

:   作者提出「查询代价随查询集合大小线性增长」的尺寸敏感拟阵 oracle 模型，证明在该模型下找基、估计秩、估计划分数的最优查询代价都是 $\tilde{\Theta}(n^2)$，并对有界周长 $c$ 的拟阵给出 $\mathcal{O}(n^{2-1/c}\log n)$ 的最大权基算法突破二次下界。

**[Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)**

:   针对 TabPFN 这类把训练集当作 in-context 直接喂进 attention 的"表格基础模型"做后验校正——发现它会严重过拟合训练集 majority class, 提出 DistPFN：用 $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$ 这一行后验重加权, 在 253 个 OpenML 数据集上把 TabPFN-v2 在 $\beta=5$ 强标签漂移下的准确率从 72.7% 拉到 76.9%, 不用重训、不用估测试先验、不动架构。

**[Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)**

:   MPFM 把 OSAD 里传统的"单峰高斯原型"换成可学习的**高斯混合原型空间**, 用流匹配直接回归一个 GMM 形式的速度场, 再加一个互信息最大化正则防止原型崩塌, 在 9 个工业 / 医学 AD 数据集上以 10/1 个异常样本的设定打过 DRA / AHL / DPDL 等所有 SOTA.

**[Networked Information Aggregation for Binary Classification](networked_information_aggregation_for_binary_classification.md)**

:   把 Kearns-Roth-Ryu 2026 的"在 DAG 上让线性回归 agent 顺序传 prediction 列即可逼近全局最优"结论推广到二分类：每个 agent 只看到部分特征列、顺序地把自己的 logit 转发给下游，能在 $M$-coverage 条件下用 $O(M/\sqrt{D})$ 超额 BCE loss 达到全局逻辑回归最优；同时构造硬实例证明 $\Omega(k/D)$ 下界，把网络深度刻画成信息聚合的根本瓶颈。

**[New Bounds for Kernel Sums via Fast Spherical Embeddings](new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)**

:   通过把 Bartal-Recht-Schulman 2011 的"随机 Nash 装置"球面嵌入定理用迭代 Fastfood 变换做成快速版（time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$），再把它作为 Gaussian KDE 的预处理把直径压到 $\widetilde{O}(1/\sqrt{\varepsilon})$，得到新的 Gaussian KDE 查询时间界 $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$，在小 $\varepsilon$ 中等直径的体制下优于 RFF / FJLT+RFF / Fastfood。

**[NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)**

:   用一个 asinh 链接的 GLM surrogate 把多智能体 MCTS 的 joint-action 空间 $d^n$ 压成 low-dim 非线性 bandit，再用"一阶差分量 + 二阶 mixed difference"作为 NonUCT 提议规则，只在每个节点维护小候选集 $\mathcal{C}(s)$，证明 $\widetilde{O}(T^{3/4})$ 的局部 regret（与 $d^n$ 无关），在 MatGame/SMAC/SMACv2 上 sample efficiency 和最终性能都好过 MAZero 等强 baseline。

**[Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)**

:   Polaris 把概念表示拆成"方向（语义）+ 轨道势能（层级）"两个解耦信号，全部学到单位超球面上：用切空间投影 + 指数映射保证流形封闭，用各向异性球面 SVGD 防止赤道聚集，用 vMF KL 散度实现不对称的"父类应比子类更高熵"约束，在 taxonomy expansion 任务上把 top-K 召回提升最多 19 点、mean rank 降低 60%。

**[Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)**

:   本文用 possibility theory 替代 Bayes 概率框架，提出 DAPPr——把参数空间的 possibilistic 后验通过 supremum 投影到预测空间，再用可学习的 Dirichlet possibility function 拟合，最终得到一个仅 10 行代码、可直接替换交叉熵、且在 OOD 检测上超越 EDL 家族的认知不确定性建模方法。

**[Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)**

:   本文用「实代数几何 + 一阶谓词逻辑量词消去」给多维超参数调参第一次给出可证明的 generalization bound，把过去只能处理一维标量超参的 Balcan 2025 框架推广到任意 $p$ 维、双层验证损失、近似内层优化等多种实际场景，并配出第一条匹配上界的下界。

**[Realizable Bayes-Consistency for General Metric Losses](realizable_bayes-consistency_for_general_metric_losses.md)**

:   本文对"在一般（可能无界）度量损失下，假设类 $\mathcal{H}$ 何时存在分布无关的强通用 Bayes 一致学习算法"这一开放问题在 realizable 情形下给出锐刻画——充分必要条件是 $\mathcal{H}$ 不包含一种新的"无界 gap Littlestone 树"组合障碍。

**[Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)**

:   本文是一篇 ICML 立场论文,主张当前所有 AI 可靠性方法 (RAG / 自一致性 / RLHF / Agent Memory) 都只能验证显式知识,而 AI 真正强大的能力来自训练数据里 80-95% 未被人类正式记录的"隐式知识",作者提出 Knowledge Objects (KOs) 作为基础设施——把 AI 隐式推理外化成人类可检查、可验证、可背书的结构化产物,从而让一次人类验证的成本在群体中长期复利。

**[Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)**

:   作者提出 CaRE：在 ViT 每个 block 里塞一个 **两级路由 MoE (BR-MoE)** ——先靠"类感知器"按熵选 Top-M 个相关任务路由，再由这些路由各自激活 Top-K 任务专家并叠加一个共享 EMA 专家，于是哪怕任务序列拉到 300+ 也能既保留旧知识又持续吸纳新类，并把"长序列 CIL"这块此前没人正经做的空白填上（顺便发布了 1000 类的 OmniBenchmark-1K 基准）。

**[Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)**

:   本文把权重矩阵直接参数化为 $W=AB^\top$ 而不是对 $W$ 本身做平均场分布，从而诱导出一个**关于 Lebesgue 测度奇异的低秩后验**，参数量从 $O(mn)$ 降到 $O(r(m+n))$，PAC-Bayes 复杂度从 $\sqrt{mn}$ 收到 $\sqrt{r(m+n)}$，并在 MLP/LSTM/Transformer 三类架构上实现 OOD 检测胜过 5-成员 Deep Ensemble 同时参数少 $33\times$。
