---
title: >-
  ICML2026 AI 安全方向37篇论文解读
description: >-
  37篇ICML2026的 AI 安全方向论文解读，涵盖对抗鲁棒、联邦学习、水印/隐写、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "联邦学习"
  - "水印/隐写"
  - "时序预测"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🧪 ICML2026** · **37** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (4)](../../ACL2026/ai_safety/index.md) · [📷 CVPR2026 (25)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (26)](../../ICLR2026/ai_safety/index.md) · [🤖 AAAI2026 (44)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (70)](../../NeurIPS2025/ai_safety/index.md) · [📹 ICCV2025 (21)](../../ICCV2025/ai_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×4 · 联邦学习 ×3 · 水印/隐写 ×3 · 时序预测 ×2

**[Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](active_continual_learning_with_metaplastic_binary_bayesian_neural_networks.md)**

:   BiMU 为二值贝叶斯神经网络设计有界记忆和不确定性感知的 metaplastic 更新，防止 Bernoulli 后验在长程非平稳流中饱和，并用 Monte Carlo disagreement 实现无缓存的一次性主动查询，显著减少标签和反向传播更新。

**[Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)**

:   作者首次系统评估 7 种主流可塑性干预 (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) 对深度强化学习 (DRL) 后门攻击的影响 (14,664 个实验)，发现只有 SAM 是"恶魔"——能显著加剧后门威胁；据此提出"Sweeper-Converter-Connector" 鲁棒后门注入框架并给出基于 loss landscape 锐度的检测信号。

**[Calibrating Uncertainty for Zero-Shot Adversarial CLIP](calibrating_uncertainty_for_zero-shot_adversarial_clip.md)**

:   提出 UCAT 框架，将 CLIP 的 logits 重新参数化为 Dirichlet 分布的浓度参数，通过对齐干净样本与对抗样本的 Dirichlet 分布（反向 KL 散度），在零样本对抗微调中同时校准不确定性和保持语义结构，在 16 个基准上实现了鲁棒性与校准的最优平衡。

**[COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)**

:   COPF 把"演进图上的在线链路推荐"看成一个 performative 决策过程，在 backbone 打分器之外加一层 **决策层 wrapper**：用带显式探索的在线日志协议保证反事实可识别，用图感知双重稳健（GA-DR）估计器估计"曝光-未曝光"的反事实组间差距，再用 Residual-OI 审计 + PI primal–dual 控制器在线压制部署后出现的公平性 spike，理论上给出从插件式 OI 到真实反事实差距的 transfer 证书，在 TGB 与合成二部流上以可控的效用损失显著降低 Deploy 阶段的 worst-case TE 差距。

**[Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)**

:   本文给出多分类公平分类问题中 Bayes 最优分类器的解析可处理形式（带熵正则的闭式解），并据此推出一对统一的算法 OptFair：训练阶段用 reduction 转化为代价敏感交叉熵的 saddle-point 优化，部署阶段用 plug-in 估计求解凸近端梯度问题，两者在理论上都收敛到 accuracy-fairness Pareto 前沿。

**[Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)**

:   本文提出 TSEF——一个针对"时序分类器 + 解释器"联合系统的对偶目标攻击框架：通过学习"时间脆弱掩码 + 频域扰动滤波器"，在 $\ell_\infty$ 预算内同时把模型预测推到目标标签、又把解释推到攻击者指定的参考显著图，证明现有时序可解释流水线的"解释稳定 = 决策可信"假设根本不成立。

**[Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)**

:   本文把 Ravfogel 等人为线性模型设计的「迭代零空间投影 (INLP)」公平化方法搬到核方法上：通过在经验特征空间 (empirical feature space) 推导一个直接作用在核矩阵 $\mathbf{K}$ 上的闭式变换 $\mathbf{T}$，使得变换后的 $\mathbf{K}_{(m)}$ 仍是半正定核，但已被剥离了对连续受保护属性的预测信息，从而把任意基于核的算法（KRR、SVR）一键改造为「连续公平」版本，在 Crimes / ACSIncome / ACSTravelTime 上取得有竞争力或更优的 fairness–accuracy 帕累托。

**[Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)**

:   本文揭示数据集蒸馏 (DD) 会放大原始数据中的偏差——根源是「子组样本量不平衡」与「子组表征分离度」的交互作用，并提出 COBRA：用各子组表征的（与组大小无关的）barycenter 作为蒸馏目标，可在多个 DD 框架上同时降低 EOD、提高准确率。

**[Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)**

:   本文针对"即使分数在各群体上完全 group-calibrated，对其取单一阈值也会违反 sufficiency（predictive parity）"这一长期被忽视的痛点，给出有限离散分数下 sufficiency 约束最优二元分类器的**精确解**：通过对 $(\mathrm{PPV}, \mathrm{FOR})$ 可行域的几何刻画，得到一个只依赖分数和群体标签的后处理算法，并证明该算法同时可解"损失最小化"和"在 sufficiency 下最小化与 separation 的偏差"两类目标。

**[Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)**

:   在 Spearman footrule 距离下，把 ILP 的约束矩阵证成全单模，从而给出 fair top-$k$ 排名聚合的首个多项式时间最优算法；并以"先解 fair top-$k$，再用最小代价完美匹配补齐成全排列"的两步策略，把 fair (full) rank aggregation 的近似比从 3 改进到 2。

**[FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)**

:   针对原型类联邦学习中"对局部原型直接平均会继承客户端偏差"的问题，本文用一组可学习的全局超原型 (hyper-prototypes)，通过梯度匹配在服务器侧模拟集中式训练得到的原型，再配合客户端对比学习与对齐损失显著提升异质场景下的精度。

**[Flatness-Aware Stochastic Gradient Langevin Dynamics](flatness-aware_stochastic_gradient_langevin_dynamics.md)**

:   本文提出 fSGLD：在标准 SGLD 更新里把梯度处的参数 $\theta$ 换成被高斯扰动过的 $\theta+\epsilon$，并将扰动尺度 $\sigma$ 与逆温度 $\beta$ 通过 $\sigma=\beta^{-(1+\eta)/4}$ 严格耦合，从而在不增加任何梯度/内存开销的前提下，让算法的不变测度逼近 Hessian-trace 正则化目标 $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 对应的 Gibbs 分布，并给出 Wasserstein-1 与超额风险的非渐近界，在 CIFAR/WebVision/ViT 上取得与 SAM/ASAM 相当或更优、但训练时间近乎减半的效果。

**[Frequency Matching in Spiking Neural Networks for mmWave Sensing](frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)**

:   本文从「机制-数据对齐」角度证明 LIF 脉冲神经元等价于一个一阶 IIR 低通滤波器，并提出根据毫米波信号的判别频谱来设定膜衰减系数 $\beta$，使 SNN 在四个常用 mmWave 数据集上平均比 ANN 提高 6.22% 精度并降低 3.64× 理论能耗。

**[From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view.md)**

:   本文把 LLM 的下一 token 预测视为一个超大词表上的分类任务，将两个轻量级 OOD 检测器 NCI（特征与权重向量的接近度）与 fDBD（特征到决策边界的距离）迁移过来，配合"训练特征均值的解析代理 $\mu_G$"和"只在 top-$k$ 候选 token 上算边界距离"两个适配，得到一个**无训练、单样本**的推理类幻觉检测器，在 CSQA / GSM8K / AQuA 上稳定优于困惑度、Semantic Entropy、SelfCheckGPT 等基线。

**[GEM-FI: Gated Evidential Mixtures with Fisher Modulation](gem-fi_gated_evidential_mixtures_with_fisher_modulation.md)**

:   本文针对证据深度学习 (EDL) 在分布外样本上过自信、且单头难以表达多模态认知不确定性的问题，提出三件套 GEM-Core/MIX/FI：用学到的特征能量门控证据、用混合证据头单次推理近似 ensemble、用 Fisher 信息正则稳定混合分配，在 CIFAR-10→SVHN/CIFAR-100 等 OOD 检测上比 DAEDL 强且保持 single-pass。

**[Geometrically Constrained Outlier Synthesis](geometrically_constrained_outlier_synthesis.md)**

:   GCOS 在 ID 特征 PCA 的"小方差子空间"上沿几何 off-manifold 方向合成虚拟离群点，并用从校准集 Mahalanobis 分位数导出的"共形壳层" $[\alpha_\text{inner},\alpha_\text{outer}]$ 控制合成强度，配合自适应 margin 的对比正则损失训练，在 4 个 near-OOD 数据集上把平均 AUROC 从 VOS 的 86.21 提到 93.47。

**[Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)**

:   针对自回归音频生成模型在 KGW 风格 token 水印下因"解码→重编码不幂等"导致水印信号指数级衰减的问题，作者用 codec 自身的混淆矩阵跑 Leiden 社区检测得到一个收缩后的"簇词表"，把水印的绿/红集合定义在簇而非 token 上，从而在完全梯度自由、黑盒访问 codec 的前提下把 $z$-score 的指数底从 $r$ 抬到 $r_{cl}>r$，detectability 相比基线和需要微调 codec 的 WMAR 普遍提升数个量级，且对 MP3、降噪、裁剪等扰动天然鲁棒。

**[How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)**

:   把"遗忘 vs 保留"的 trade-off 直接写成"每步带约束的一阶凸优化"问题，用 retain/forget 梯度的点积 $\kappa = \bm{g_r}\cdot\bm{g_f}$ 同时充当 hardness 度量、更新方向切换开关和提前停止条件，在 CIFAR-10/ResNet-20 与 Llama-2-7B/WaterDrum-TOFU 上比 GA、GDiff、SCRUB、KL 等基线更稳。

**[LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams](lapras_learning-augmented_private_answering_for_linear_query_streams.md)**

:   LAPRAS 用一个"哪些查询会来"的预测器把在线 DP 查询流分成预测内/外两类，预测内的用离线最优 Matrix Mechanism 一次性低噪释放，预测外的用 Smooth Allocation 根据流中已观测到的"未预测查询"位置在线估计总数并平滑分配预算，在预测准时几乎追平离线最优、预测差时退化到在线 baseline 水平。

**[MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)**

:   把多个客户端在私有数据上独立微调出的领域专家，无需共享私有数据就能合并成一个可部署的 MoE 模型——核心是用 relevance-weighted DPP 从公开数据里选「既相关又多样」的代理样本，先做 proxy-aligned 专家训练再训 context-aware router，从而对齐专家行为与代理监督，显著优于 FlexOlmo 等仅依赖相似度选代理的方法。

**[Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)**

:   本文为 $(\varepsilon,\delta)$-DP 设计了一类高斯混合加性噪声机制（multi-Gaussian mixture 与无超参的 quasi-Gaussian mixture），在中低隐私域将解析高斯机制的次优间隙关闭高达 99%，同时保留高斯的 zCDP 紧组合性质。

**[OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)**

:   本文针对"图/文/视频混合伪造同时检测+定位"这一统一任务，提出 OmniVL-Guard，用 Self-Evolving CoT 合成高质量冷启动数据 + ARSPO（非线性奖励映射 + 动态任务权重）解决多任务 RL 中"简单的真假分类抢走梯度、细粒度定位学不动"的难度偏置问题，在 In-Domain 上视频时序定位 tIoU +37.8、文本定位 F1 +22.9，并在四个 OOD benchmark 上做到零样本 SOTA。

**[One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)**

:   UniTrans 把"为每对车端模态训一个 adapter"的传统协同感知翻译范式，改写成"在一个模态内蕴空间里推断映射 → 通过 router 线性组合一组专家参数 → 当场实例化一个映射专属翻译器"，实现对未见过的新车型的零样本 BEV 特征翻译，在 OPV2V-H / DAIR-V2X 上平均 AP@0.7 较最强基线提升 ~7 / 3 个点，同时 GFLOPs / CPU 时间均低于 Classic MoE。

**[Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)**

:   本文把"群体公平性"显式编码为一个 $K_s \times K_w$ 的组间匹配概率目标 $\mathbf{F}$，提出 **FairSinkhorn** 精确求解、**惩罚式 OT** 凸松弛、以及 **双层成本学习** 三种方案，分别给出有限样本复杂度 $O(1/\sqrt{n})$ 和 fairness 偏差界 $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$，在合成与半合成（约会 app）数据集上勾画出"代价 - 公平性"权衡前沿。

**[Persuasive Privacy](persuasive_privacy.md)**

:   本文用 Sender–Receiver 两方 Stackelberg 博弈 + Bayesian Persuasion 思想，把"隐私"重新表述为 Receiver 在最坏 data-prior 下的相对评分规则损失，给出统一定义 $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP，同时把 pure DP 和 probabilistic DP 收编为特例，并首次为**确定性算法**（如无噪经验均值）给出非平凡的形式化隐私保证。

**[Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)**

:   本文是一篇 position paper，主张具身 AI 的隐私不能用单阶段补丁解决，必须当作横跨 instruction / perception / planning / interaction 全生命周期的架构级动态控制信号，并提出 SPINE 框架，用 L1-L4 四级隐私分类矩阵在每个阶段联动调整智能体行为。

**[Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)**

:   这是一篇 ICML 2026 立场论文：作者结合 UNOS 历史数据，论证美国心脏移植分配系统的下一代 ML 策略必须把"器官获取组织/移植中心/医生/患者/监管"之间的激励错位当成一等公民来建模，呼吁把机制设计、战略分类、因果推断、社会选择整合进 ML 流水线，否则再强的预测模型也会在部署时被各方策略性行为反噬。

**[PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](prism_gauge-invariant_tangent-space_differentially_private_lora.md)**

:   PRISM 把 DP-SGD 从 LoRA 的 $(A,B)$ 因子空间搬到 rank-$r$ 流形的切空间上做 clip+加噪+retract，从而获得 gauge invariant、无 bilinear 二阶噪声、且有闭式 $\sigma C/b\cdot\sqrt{r(m+n-r)}$ 内禀噪声能量的 DP-LoRA 机制。

**[Regret-Based Federated Causal Discovery with Unknown Interventions](regret-based_federated_causal_discovery_with_unknown_interventions.md)**

:   本文提出 I-PERI：在客户端干预目标完全未知、且只能共享 regret 标量的联邦设置下，用"有向一致掩码 + 无向一致掩码"两阶段流程，恢复出一个比观测 MEC 更紧、比 I-MEC 更松的全新等价类 Φ-MEC，并通过 Laplace 噪声给出 ε-差分隐私保证。

**[Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling](rotation-invariant_spherical_watermarking_via_third-order_so3_representation_cou.md)**

:   TRIAD 把 360° 全景图当作球面信号，用三阶球谐系数张量积投影到 trivial 表示得到一个**理论可证 SO(3) 不变**的双谱标量，从而把水印藏在高阶 SH 系数里、再从这个不变量里读出来，在任意 3D 旋转下仍能保持近 100% 的比特准确率而不依赖数据增强。

**[Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions](same_target_different_basins_hard_vs_soft_labels_for_annotator_distributions.md)**

:   在 CIFAR-10H 上把"标注者分布"以硬标签方式投喂给模型（multipass 按票循环 / SLS 每个 epoch 重采样），证明它和软标签交叉熵期望目标等价，但收敛到更平坦的 basin、在稀疏标注下更优、且 OOD 检测略胜。

**[Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)**

:   针对现有联邦多源无监督域适应 (UMDA) 方法只能处理 2–6 个源、源数一多就训练不稳或算力爆掉的问题，作者提出 GALA：把所有源随机分成若干小组、组间对预测分布做差异最小化（把 $O(N^2)$ 的两两对齐压成线性），再叠一个基于质心+温度的相似度加权挑出真正贴近目标域的源——在新建的 Digit-18 (18 源) 基准上稳定收敛，且把基线一一推开。

**[Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)**

:   本文把权重矩阵直接参数化为 $W=AB^\top$ 而不是对 $W$ 本身做平均场分布，从而诱导出一个**关于 Lebesgue 测度奇异的低秩后验**，参数量从 $O(mn)$ 降到 $O(r(m+n))$，PAC-Bayes 复杂度从 $\sqrt{mn}$ 收到 $\sqrt{r(m+n)}$，并在 MLP/LSTM/Transformer 三类架构上实现 OOD 检测胜过 5-成员 Deep Ensemble 同时参数少 $33\times$。

**[TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)**

:   TimeGuard 把多变量时间序列预测里的后门防御从"丢掉整条窗口"重构成"按通道+按时间步"的可靠样本池训练，先用反向一致性 (RCF) + 邻域多样性 (NDF) 交集初始化高纯度池子，再用距离正则的损失筛选 (DRLS) 渐进扩池，在不依赖任何干净数据的前提下把对 BackTime 等 SOTA 攻击的 $\text{MAE}_{\text{P}}$ 提到最强基线 PDB 的 1.96 倍。

**[Training-Free Coverless Multi-Image Steganography with Access Control](training-free_coverless_multi-image_steganography_with_access_control.md)**

:   提出 MIDAS，一种基于预训练扩散模型的 training-free 无载体多图隐写框架，用 Random Basis 正交随机基替代传统 Noise Flip 实现按私钥的细粒度访问控制，配合 Latent Vector Fusion 消除拼接边界，在不传输任何与秘密相关的附加信息的前提下实现多图隐藏 + 抗隐写分析。

**[机器遗忘的两个盲点：过度遗忘与原型重学习攻击](unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)**

:   本文揭示机器遗忘的两个关键盲点——过度遗忘（对边界附近样本的误伤）和原型重学习攻击（用少量样本复原遗忘知识），并提出 Spotter 框架通过边界掩膜蒸馏和类内散布损失同时缓解这两个问题。

**[VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)**

:   作者构造了 10 万张图、33 个细粒度类别、19 万+ 实例的大规模视觉隐私数据集 VPD-100K（覆盖人脸/屏上 PII/物理证件/位置标记四大域），并提出三件套频域增强模块（FDAF + 自适应频谱门控 + 频域一致性损失）插入 YOLOv10 的 Neck，使 YOLOv10-L 在 VPD-100K 上 AP 从 53.8 涨到 58.6（+4.8），同时在 7.51ms 延迟下稳定跑直播流。
