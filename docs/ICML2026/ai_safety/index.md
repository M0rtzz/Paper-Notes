---
title: >-
  ICML2026 AI 安全方向16篇论文解读
description: >-
  16篇ICML2026的 AI 安全方向论文解读，涵盖对抗鲁棒、联邦学习、强化学习、对齐/RLHF、域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "AI 安全"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "联邦学习"
  - "强化学习"
  - "对齐/RLHF"
  - "域适应"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**🧪 ICML2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (2)](../../ACL2026/ai_safety/index.md) · [📷 CVPR2026 (22)](../../CVPR2026/ai_safety/index.md) · [🔬 ICLR2026 (26)](../../ICLR2026/ai_safety/index.md) · [🤖 AAAI2026 (44)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (70)](../../NeurIPS2025/ai_safety/index.md) · [📹 ICCV2025 (21)](../../ICCV2025/ai_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×3 · 联邦学习 ×2

**[ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)**

:   本文提出一个分层框架 ACTG，将隐私文本生成分解为特征学习与条件文本生成两个子任务；进一步引入 Anchored RL，通过混合强化学习目标与基于最优 N 选一的 SFT 锚点，在保持文本保真度的前提下提升条件生成器的指令跟随能力，在生物医学数据上相比先前工作提升 20% MAUVE。

**[Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)**

:   作者首次系统评估 7 种主流可塑性干预 (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) 对深度强化学习 (DRL) 后门攻击的影响 (14,664 个实验)，发现只有 SAM 是"恶魔"——能显著加剧后门威胁；据此提出"Sweeper-Converter-Connector" 鲁棒后门注入框架并给出基于 loss landscape 锐度的检测信号。

**[Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)**

:   本文把随机平滑（RS）从"只支持单一连续或离散输入"扩展到"离散 token + 连续图像"的混合扰动场景，通过一个混合 Neyman–Pearson 分析得到一个**一维、连续、可逆**的似然比 CDF，从而把原本组合爆炸的离散 knapsack 问题变成可解的根求解问题，并在 LLaVA-Guard 多模态安全过滤上给出首个针对"图文联合不安全"的 model-agnostic 证书。

**[DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)**

:   本文提出 DP-KFC：基于"Fisher 矩阵的标度由架构决定、相关结构可用模态级频谱统计近似"的观察，用结构化合成噪声（图像用 $1/f^\alpha$ pink noise，文本用 Zipf 采样）探测网络重建 KFAC 预条件子，既不消耗隐私预算也不引入分布偏移，在强隐私（$\varepsilon\le 3$）下持续超过 DP-SGD 与公共数据预条件方法。

**[Dual-branch Robust Unlearnable Examples](dual-branch_robust_unlearnable_examples.md)**

:   本文提出 DUNE：把不可学习样本（UE）的扰动从单一空间域扩展到"空间 + 色彩"双域优化，使扰动特征对齐到 shift-induced 标签并配合预训练模型集成增强，在 CIFAR-10 / ImageNet 上对 7 种主流防御（含 ECLIPSE、ISS-J、COIN）保持鲁棒，平均测试精度比 12 个 SOTA UE 方案再低 14.95%–50.82%。

**[Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)**

:   本文揭示数据集蒸馏 (DD) 会放大原始数据中的偏差——根源是「子组样本量不平衡」与「子组表征分离度」的交互作用，并提出 COBRA：用各子组表征的（与组大小无关的）barycenter 作为蒸馏目标，可在多个 DD 框架上同时降低 EOD、提高准确率。

**[FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)**

:   针对原型类联邦学习中"对局部原型直接平均会继承客户端偏差"的问题，本文用一组可学习的全局超原型 (hyper-prototypes)，通过梯度匹配在服务器侧模拟集中式训练得到的原型，再配合客户端对比学习与对齐损失显著提升异质场景下的精度。

**[Frequency Matching in Spiking Neural Networks for mmWave Sensing](frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)**

:   本文从「机制-数据对齐」角度证明 LIF 脉冲神经元等价于一个一阶 IIR 低通滤波器，并提出根据毫米波信号的判别频谱来设定膜衰减系数 $\beta$，使 SNN 在四个常用 mmWave 数据集上平均比 ANN 提高 6.22% 精度并降低 3.64× 理论能耗。

**[LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams](lapras_learning-augmented_private_answering_for_linear_query_streams.md)**

:   LAPRAS 用一个"哪些查询会来"的预测器把在线 DP 查询流分成预测内/外两类，预测内的用离线最优 Matrix Mechanism 一次性低噪释放，预测外的用 Smooth Allocation 根据流中已观测到的"未预测查询"位置在线估计总数并平滑分配预算，在预测准时几乎追平离线最优、预测差时退化到在线 baseline 水平。

**[Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)**

:   作者把"开源权重安全"形式化为"如何延缓恶意 fine-tune 的收敛速度"，证明 Hessian 谱的最大奇异值由权重谱下界决定，由此设计了能严格减慢一阶/二阶优化的 SpecDef 算法，但同时证明任何此类收敛率控制方法都能被攻击者以"线性模型尺寸增加"的代价绕过。

**[MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)**

:   把多个客户端在私有数据上独立微调出的领域专家，无需共享私有数据就能合并成一个可部署的 MoE 模型——核心是用 relevance-weighted DPP 从公开数据里选「既相关又多样」的代理样本，先做 proxy-aligned 专家训练再训 context-aware router，从而对齐专家行为与代理监督，显著优于 FlexOlmo 等仅依赖相似度选代理的方法。

**[Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)**

:   本文是一篇 position paper，主张具身 AI 的隐私不能用单阶段补丁解决，必须当作横跨 instruction / perception / planning / interaction 全生命周期的架构级动态控制信号，并提出 SPINE 框架，用 L1-L4 四级隐私分类矩阵在每个阶段联动调整智能体行为。

**[Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)**

:   作者给"差分隐私零阶优化（DP-ZOGD）"首次证出了**收敛的 hidden-state DP 上界**——通过设计一个"定向 + 各向同性"混合噪声机制并构造一个介于两条相邻轨迹之间的辅助过程，绕开了零阶更新缺乏全局 Lipschitz 性这一技术障碍，揭示出"扩大每步采样方向数 $K$ 反而能降隐私损失"这一前所未知的 DP 算法设计准则。

**[Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)**

:   针对现有联邦多源无监督域适应 (UMDA) 方法只能处理 2–6 个源、源数一多就训练不稳或算力爆掉的问题，作者提出 GALA：把所有源随机分成若干小组、组间对预测分布做差异最小化（把 $O(N^2)$ 的两两对齐压成线性），再叠一个基于质心+温度的相似度加权挑出真正贴近目标域的源——在新建的 Digit-18 (18 源) 基准上稳定收敛，且把基线一一推开。

**[The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)**

:   本文构造了一个程序化生成的"合成 Web"环境,通过在搜索 rank 0 注入单条高可信度蜜罐误信息,因果性地测出 GPT-5 等前沿 LLM agent 在 1/数千的对抗污染下准确率从 65% 暴跌到 18%,且模型不会增加搜索、依然高置信度作答,揭示了根深蒂固的"位置锚定"失败模式。

**[VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)**

:   作者构造了 10 万张图、33 个细粒度类别、19 万+ 实例的大规模视觉隐私数据集 VPD-100K（覆盖人脸/屏上 PII/物理证件/位置标记四大域），并提出三件套频域增强模块（FDAF + 自适应频谱门控 + 频域一致性损失）插入 YOLOv10 的 Neck，使 YOLOv10-L 在 VPD-100K 上 AP 从 53.8 涨到 58.6（+4.8），同时在 7.51ms 延迟下稳定跑直播流。
