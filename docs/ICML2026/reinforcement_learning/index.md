---
title: >-
  ICML2026 强化学习方向20篇论文解读
description: >-
  20篇ICML2026的强化学习方向论文解读，涵盖强化学习、推理、Agent、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "强化学习"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "Agent"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**🧪 ICML2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (34)](../../ACL2026/reinforcement_learning/index.md) · [📷 CVPR2026 (19)](../../CVPR2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (138)](../../ICLR2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (71)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (172)](../../NeurIPS2025/reinforcement_learning/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/reinforcement_learning/index.md)

🔥 **高频主题：** 强化学习 ×9 · 推理 ×4 · Agent ×2 · 对抗鲁棒 ×2

**[CAMEL: Confidence-Gated Reflection for Reward Modeling](camel_confidence-gated_reflection_for_reward_modeling.md)**

:   本文观察到 verdict token 的 log-probability margin 与判断正确率高度相关，据此提出 CAMEL —— 先用单 token 快速给出偏好判断，仅在低置信度时才触发反思生成，并用反事实前缀增强 GRPO 训练自我纠错能力，在三个奖励模型 benchmark 上以 14B 参数取得 82.9% 的平均准确率（超过此前最佳 70B 模型 3.2%）。

**[CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)**

:   把 self-play 从"对抗"换成"协作": Coach 出题、Player 解题、Coach 拿"Player 进步幅度 × Player 解题率"作为奖励, 在完全不用外部训练数据的条件下让 Qwen2.5-Math-7B-Instruct 在六个数学 benchmark 上总均分 +4.9、OOD +5.4, 超过 RENT/R-Zero 等已有 unsupervised 方法。

**[DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](debiased_model-based_representations_for_sample-efficient_continuous_control.md)**

:   DR.Q 在 MR.Q 类"模型化表示 + actor-critic"骨架上加两件事——用 InfoNCE 显式最大化 $z_{sa}$ 与下一状态表示 $z_{s'}$ 的互信息，再用"PER × forget"融合的 faded prioritized replay 缓解早期经验过拟合——在 73 个连续控制任务上用单一超参组击败 SimBaV2 / MR.Q / TDMPC2 等强基线。

**[EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)**

:   EARL 用"粗解析-细响应"两阶段 MLLM 框架把第一视角交互理解任务（描述+答问+像素掩膜）做成统一管线：第一阶段输出整图交互的全局描述并把最后一层 hidden state 当作语义先验，再通过新的 Analysis-guided Feature Synthesizer 注入到第二阶段，用 GRPO + 三路奖励（格式/答案/grounding 准确率）联合训练，在 Ego-IRGBench 上 cIoU 反超 Seg-Zero 8.37%。

**[From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)**

:   针对"多轮交互式工具调用 Agent"后训练里两大瓶颈——高质量数据贵 + 用户模拟噪声毁 RL 信号，作者提出"自演化多 agent 数据合成 (AReaL-SEA)"配套生成可执行 verifier 当奖励，再配上"先 SFT 用户模型再做大 batch + 动态过滤 GRPO"的 RL recipe，在 τ²-bench 上把 Qwen3-235B 推到 Airline 73.0 / Telecom 98.3 的 pass^1，全面达到或超过 Claude/Gemini/GPT-5。

**[How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)**

:   作者把"训 LLM 学下国际象棋"当成可验证 RL 的干净实验台，系统比对 6 类自制 SFT 数据集对 RL 的影响，发现"直接预测最佳一步 (Best Move)"得最高分但 RL 后产生不忠实推理，"预测多步最佳走法 (Best Line)"性能相当但 RL 更稳、推理更忠实；并提炼出三条可用 SFT-checkpoint 预测 RL 终局性能的指标，最终用 7B 模型在多个国象 benchmark 上超过 gpt-oss-120b。

**[Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)**

:   本文挑战“离线 RL 必须显式保守”的主流共识，提出 Neubay：用贝叶斯视角看后验上的模型集合、用**长 horizon rollout（数百步）**自然吸收价值高估、用 layer norm 与不确定度阈值控制 compounding error，从而在 D4RL/NeoRL 共 33 个数据集上不靠悲观惩罚就追平 SOTA 保守算法，并在 7 个数据集上刷新纪录。

**[Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)**

:   SeqComm-DFL 把"多智能体通信"作为预测器、把"联合策略选择"作为下游优化器，用价值感知的消息生成 + Stackelberg 序贯条件 + 隐式微分双层优化把通信学习直接对齐到团队回报，在医院调度和 SMAC 上取得 4-6 倍的累积奖励提升与 >13 个百分点的胜率提高。

**[Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)**

:   把分布式 Bellman 方程的"仿射搬运"几何性显式编织进 flow matching 的路径里：用同一份基础噪声同时驱动当前态与后继态的两条路径，再用 $\lambda$ 控制变量在偏差与方差之间换挡，从而得到一个对源分布相容、对 Bellman 端点相容、又稳定的分布式 critic。

**[Probing RLVR Training Instability through the Lens of Objective-Level Hacking](probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)**

:   作者提出"objective-level hacking"框架,把 MoE 大模型在 RLVR 中训练-推理差异越训越大的现象归因为 token 级权重失真在优化目标里引入的有偏伪信号,并在 30B MoE 上通过四组实验验证"偏差(不是方差)才是元凶"。

**[QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)**

:   QHyer 用 Normalizing Flows 估计的状态依赖 Q 值取代 Decision Transformer 中的轨迹依赖 RTG，再叠加门控式 Attention-Mamba 混合骨干以实现内容自适应的历史压缩，在 OGBench/D4RL 的非马尔可夫与马尔可夫离线目标条件 RL 数据集上同时刷新 SOTA。

**[R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](r2r2_robust_representation_for_intensive_experience_reuse_via_redundancy_reducti.md)**

:   R2R2 把 VICReg 风格的冗余去除约束加进自预测学习（SPL）以稳定高 UTD 训练，但**关键改动是不做零中心化**——理论上证明 zero-centering 会消除 SPL 谱分解中的常数本征模（即全局动力学信息），实验在 TD7 上 UTD=20 时把分数从 1.02 提到 1.24（+22%），并以新提出的 SimbaV2-SPL 架构刷新连续控制 SOTA。

**[Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)**

:   EnergyFlow 把 diffusion policy 的 score field 显式参数化为一个标量 energy function 的负梯度，论证了 maximum-entropy 最优下 score = 软 Q-函数梯度，从而在不做对抗优化的情况下"白送"一个可用作下游 RL shaping reward 的标量信号，同时保守场约束改善 OOD 泛化。

**[SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)**

:   本文把 MoE 策略在持续强化学习中的可塑性丢失形式化为 empirical NTK 矩阵谱熵有效秩的下降，再用 Gauss-Newton 与 Kronecker 分解把它降维到一个只依赖"专家特征 Gram 矩阵"的可计算 proxy，最后用一个一行的 Parseval 罚（SPHERE）拉高这个 proxy，在 MetaWorld 和 HumanoidBench 持续 RL 设置下把任务成功率分别提升 133% 和 50%。

**[Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](stochastic_minimum-cost_reach-avoid_reinforcement_learning.md)**

:   本文提出 Reach-Avoid Probability Certificate (RAPC), 用一个 max-min-夹紧的 Bellman 收缩算子让值函数下界 reach-avoid 概率, 配合一个对抗 $\gamma^T$ 衰减的 "补偿因子"作归一化, 再用对称梯度投影联合优化 "成本"与 "reach-avoid 概率"两个冲突目标, 在 MuJoCo 上同时拿到比 RC-PPO / RESPO / CPPO 更低的累积成本和更高的 reach 成功率。

**[T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)**

:   T$^2$PO 把多轮 agentic RL 的训练崩溃归因为"hesitation（犹豫）"——token 层过思考、turn 层重复无效——并用一个融合 entropy+confidence 的自校准不确定性信号 $M_t$ 同时驱动 token-level Thinking Intervention（动态截断 think 段）和 turn-level Dynamical Sampling（重采样无效 turn），在 WebShop / ALFWorld / Search QA 上稳定超越 PPO/GRPO/GiGPO。

**[Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)**

:   本文提出 FAN：把"昂贵的生成式策略 + 分布式 critic"压缩到"单步 flow 锚定 + 单噪声样本 critic"——用 Flow Anchoring 在一次 flow 评估内完成行为正则化，用 noise-conditioned critic 把 quantile 多样本替换成单 Gaussian 噪声样本，在 D4RL/OGBench 上做到 SOTA 性能同时训练比同类分布式方法快 5-14×。

**[Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)**

:   本文提出 LIFT：在主动定位任务里，利用轨迹几何性质把次优 logging policy 留下的冗余 zig-zag 轨迹"抄近道"成 shortcut，并把这些合成 transition 喂给一个轻量增广器在数据采集期间替换 logging 动作，使离线 CQL 在低维到高维、partial obs 等各种设置下显著超越普通离线 RL 与 warm-start SAC。

**[Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)**

:   本文把多个 MLLM 之间的推理"漂移"重新解释成 DPO 中的负样本约束，用 Plackett-Luce 偏好损失同时压制 N 个 source model 的发散轨迹，让 7B 学生模型在不需要 ground-truth 报告的前提下，仅用 10% 的 MIMIC-CXR 就在胸片分类与报告生成任务上超过所有 source teacher。

**[Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)**

:   本文研究"在 N 个智能体的大规模 MARL 系统中挑出 K 个最脆弱的智能体"这一双层 NP-hard 问题，把它建模为 HAD-MFC（Hierarchical Adversarial Decentralized Mean Field Control），用 Fenchel-Rockafellar 变换把下层最坏对抗策略的训练折叠成一个加正则项的"鲁棒 mean-field Bellman 算子"，再把上层组合选择问题转化为带稠密 reward 的 MDP 用贪心或 RL 求解，证明分解保持最优性，在 18 个任务中 17 个超 baseline。
