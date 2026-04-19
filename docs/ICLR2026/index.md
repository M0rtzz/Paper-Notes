---
title: >-
  ICLR2026 1519篇论文解读
description: >-
  1519篇ICLR2026论文深度解读，每篇5分钟读懂核心思想，含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 ICLR2026 论文笔记

**1519** 篇论文解读，覆盖 **44** 个领域。

<div class="conf-index" markdown>

---

## 🎨 图像生成 { #image_generation }

**[A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)**

:   对扩散 Transformer 的条件嵌入进行首次系统分析，发现极端的角度相似性（类间余弦相似度>99%）和维度稀疏性（仅 1-2% 的维度携带语义信息），裁剪掉 2/3 的低幅维度后生成质量基本不变，揭示了条件嵌入中隐藏的语义瓶颈。

**[AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](image_generation/aligntok_aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)**

:   提出 AlignTok，将预训练视觉基础编码器（如 DINOv2）对齐为扩散模型的连续 tokenizer，通过三阶段对齐策略（语义潜空间建立→感知细节补充→解码器精炼）构建语义丰富的潜空间，在 ImageNet 256×256 上 64 epochs 即达 gFID 1.90，比从头训练 VAE 收敛更快、生成质量更好。

**[Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)](image_generation/amortising_inference_and_meta-learning_priors_in_neural_networks.md)**

:   提出 BNNP（Bayesian Neural Network Process），一种将 BNN 权重作为隐变量、BNN 本身作为解码器的 neural process，通过逐层 amortised variational inference 在多数据集上联合学习 BNN 先验和推断网络，首次回答了"在良好先验下，近似推断方法还重要吗？"——答案是肯定的，没有免费午餐。

**[Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](image_generation/asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)**

:   AsynDM 通过为不同像素分配不同的时间步调度（prompt 相关区域去噪更慢），使其能利用更清晰的上下文参考，从而在不需要微调的情况下显著提升文图生成的语义对齐。

**[Autoregressive Image Generation with Randomized Parallel Decoding](image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)**

:   本文提出 ARPG，一种基于"引导解码"框架的视觉自回归模型，通过将位置引导（query）与内容表示（key-value）解耦，实现了完全随机顺序的训练与生成，并支持高效并行解码——在ImageNet-1K 256×256上以64步达到1.94 FID，吞吐量提升20倍以上，内存消耗降低75%以上。

**[Beyond Confidence: The Rhythms of Reasoning in Generative Models](image_generation/beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)**

:   提出 Token Constraint Bound ($\delta_{\text{TCB}}$) 指标，通过量化 LLM 隐状态在多大扰动范围内能保持 next-token 预测不变，来度量预测的局部鲁棒性，揭示了传统 perplexity 无法捕捉的预测不稳定性。

**[Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models](image_generation/blueprint-bench_comparing_spatial_intelligence_of_llms_agents_and_image_models.md)**

:   Blueprint-Bench 通过"从公寓室内照片生成 2D 平面图"的任务评测 AI 的空间推理能力：输入（照片）完全在训练分布内但任务（空间重建）在分布外。评测 GPT-5、Claude 4 Opus、Gemini 2.5 Pro、Grok-4 等 LLM，GPT-Image、NanoBanana 等图像生成模型，以及 Codex CLI、Claude Code 等 Agent 系统，结果显示绝大多数模型表现接近或低于随机基线，揭示当前 AI 在空间智能上的系统性盲区。

**[Branched Schrödinger Bridge Matching](image_generation/branched_schrödinger_bridge_matching.md)**

:   提出 BranchSBM 框架，通过参数化多个时间依赖的速度场和增长过程，将 Schrödinger Bridge Matching 扩展到分支场景，能够建模从单一初始分布到多个目标分布的分叉动态轨迹，在 LiDAR 表面导航和单细胞扰动建模等任务上显著优于单分支方法。

**[Bridging Degradation Discrimination and Generation for Universal Image Restoration](image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)**

:   BDG 通过多角度多尺度灰度共生矩阵（MAS-GLCM）进行细粒度退化判别，并设计三阶段扩散训练（生成→桥接→修复）将退化判别能力与生成先验无缝融合，在 all-in-one 修复和真实世界超分辨率任务上取得显著的保真度提升。

**[Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)**

:   FedVTC 提出在模型异构联邦学习中，各客户端通过变分转置卷积网络（VTC）从聚合的特征分布统计量中生成合成数据来微调本地模型，无需公共数据集即可显著提升泛化能力，同时降低通信和内存开销。

**[CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](image_generation/cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)**

:   提出 Consistency Mid-Training (CMT)，在预训练扩散模型和 flow map 后训练之间插入一个轻量级中间训练阶段，通过让模型学习将 ODE 轨迹上的任意点映射回干净样本来获得轨迹对齐的初始化，从而大幅降低训练成本（最多 98%）并达到 SOTA 两步生成质量。

**[Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)**

:   提出 General Policy Composition (GPC)，在测试时通过凸组合多个预训练扩散/Flow 策略的分布分数（score），无需额外训练即可产生超越任何单一父策略的更强策略，理论证明凸组合可改善单步分数误差且通过 Grönwall 界传播到全程轨迹。

**[Compositional amortized inference for large-scale hierarchical Bayesian models](image_generation/compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)**

:   将组合分数匹配（CSM）扩展到层次贝叶斯模型，通过新的误差衰减估计器和 mini-batch 策略解决大量数据组下的数值不稳定问题，首次实现超过 75 万参数（25 万+ 数据组）的大规模层次模型的摊销推断，并在荧光寿命成像的真实科学应用中验证有效性。

**[Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](image_generation/concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)**

:   提出 Concept-TRAK，通过设计概念特异的训练损失（DPS reward）和效用损失（CFG guidance），将影响函数从全图归因扩展到概念级归因，在合成、CelebA-HQ 和 AbC benchmark 上大幅超越 TRAK/D-TRAK/DAS 等方法，特别是在 OOD 组合新概念场景下优势显著。

**[Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](image_generation/condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)**

:   理论分析了自回归扩散损失模型相比条件扩散模型在条件误差修正上的优势（梯度范数指数衰减），并提出基于最优传输（Wasserstein Gradient Flow）的条件精炼方法来解决自回归过程中的"条件不一致性"问题，在 ImageNet 上达到 FID 1.31（基于 MAR）。

**[Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)**

:   提出 CW-Gen（条件白化生成模型），通过联合估计条件均值和滑动窗口协方差矩阵来替代扩散模型/流匹配中的标准高斯终端分布，理论证明了当估计器满足充分条件时采样质量必然提升，在 5 个数据集 × 6 个生成模型上一致改善多变量时间序列概率预测性能。

**[Conjuring Semantic Similarity](image_generation/conjuring_semantic_similarity.md)**

:   提出一种基于视觉想象的文本语义相似度度量——通过计算文本条件扩散模型在两个文本提示下诱导的反向 SDE 之间的 Jeffreys 散度来衡量语义距离，可用 Monte-Carlo 采样直接计算，首次量化了扩散模型学到的语义空间与人类标注的对齐程度。

**[Consistent Text-to-Image Generation via Scene De-Contextualization](image_generation/consistent_text-to-image_generation_via_scene_de-contextualization.md)**

:   揭示 T2I 模型中 ID 偏移的根本原因是"场景上下文化"（scene contextualization，场景 token 对 ID token 注入上下文信息），并提出 training-free 的 Scene De-Contextualization (SDeC) 方法，通过 SVD 特征值的方向稳定性分析识别并抑制 prompt embedding 中潜在的场景-ID 关联，实现逐场景的身份一致性生成。

**[Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](image_generation/contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)**

:   提出 DiffBacChrom——基于条件扩散 Transformer (CrossDiT) 从 Hi-C 接触图谱生成大肠杆菌三维基因组构象集合，通过 ResNet VAE 保持逐 bin 对齐的潜空间编码、Transformer 编码器 + 交叉注意力注入 Hi-C 条件、flow-matching 训练，生成的集合在距离衰减 P(s) 和 SCC 指标上与输入 Hi-C 高度一致，同时保持构象多样性。

**[Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](image_generation/contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)**

:   提出非守恒广义 Schrödinger 桥 (NCGSB)——基于接触哈密顿力学允许能量随时间变化，通过 Contact Wasserstein Geodesic (CWG) 将桥问题转化为有限维 Jacobi 度量上的测地线计算，用 ResNet 参数化实现近线性复杂度且支持引导生成，在流形导航、分子动力学、图像生成等任务上大幅超越迭代式 SB 求解器。

**[ContextBench: Modifying Contexts for Targeted Latent Activation](image_generation/contextbench_modifying_contexts_for_targeted_latent_activation.md)**

:   提出 ContextBench 基准（715 个任务）评估自动生成流畅且能激活特定潜在特征的输入文本的方法，并开发两种 EPO 增强变体（LLM辅助和扩散模型修补），在激活强度和语言流畅度的权衡上 Pareto 优于标准 EPO。

**[Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)**

:   首次系统研究 T2I 扩散模型的持续遗忘（continual unlearning）问题，发现现有遗忘方法在序列请求下因累积参数漂移导致"效用崩溃"，提出一组附加正则化策略（L1/L2 范数、选择性微调、模型合并）和语义感知的梯度投影方法来缓解该问题。

**[Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations](image_generation/contractive_diffusion_policies_robust_action_diffusion_via_contractive_score-bas.md)**

:   提出 Contractive Diffusion Policies (CDPs)，通过在扩散采样 ODE 中引入收缩正则化来抑制 score 匹配误差和求解器误差的累积，以最小修改和单一超参数 $\gamma$ 提升离线学习中扩散策略的鲁棒性。

**[COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](image_generation/cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)**

:   通过谐波失真分析与 Chebyshev 多项式逼近，严格证明了奇/偶对称激活函数在后激活频谱中存在系统性衰减，提出用复正弦项 $e^{j\zeta x}$ 调制激活函数来保留完整频谱支持，并设计 COSMO-RC 激活函数与正则化先验嵌入器架构，在 Kodak 图像重建上 PSNR 平均领先最强基线 +5.67 dB，NeRF 上领先 +3.45 dB。

**[CREPE: Controlling Diffusion with Replica Exchange](image_generation/crepe_controlling_diffusion_with_replica_exchange.md)**

:   提出 CREPE，一种基于 Replica Exchange（并行回火/Parallel Tempering）的扩散模型推理时控制方法，作为 SMC 的计算对偶——在去噪步维度上并行、在样本维度上串行生成，具有高样本多样性、可在线精炼、支持温度退火/奖励倾斜/模型组合/CFG 去偏等多种任务。

**[DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](image_generation/densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)**

:   解决 Flow Matching + GRPO 对齐中的稀疏奖励问题：通过 ODE 去噪预测中间潜变量的 step-wise 奖励增益作为密集奖励，并根据密集奖励自适应调整 SDE 采样器的逐时间步噪声注入来校准探索空间，在人类偏好对齐/组合生成/文字渲染三个任务上超越 Flow-GRPO。

**[Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability](image_generation/detecting_and_mitigating_memorization_in_diffusion_models_through_anisotropy_of_.md)**

:   本文证明基于范数的记忆检测指标仅在各向同性（isotropic）对数概率分布下有效，在低噪声各向异性（anisotropic）区域失效；提出结合高噪声范数和低噪声角度对齐（cosine similarity）的无去噪检测指标，在 SD v1.4/v2.0 上超越现有无去噪方法且快 5× 以上。

**[DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation](image_generation/diffink_glyph-_and_style-aware_latent_diffusion_transformer_for_text_to_online_h.md)**

:   提出 DiffInk，首个面向全行手写生成的潜在扩散 Transformer 框架，包含 InkVAE（通过 OCR + 风格分类双正则化学习结构化潜空间）和 InkDiT（在潜空间中做条件去噪生成），在中文手写生成上大幅超越 SOTA（AR 94.38% vs 91.48%），速度提升 800×。

**[Diffusion Alignment as Variational Expectation-Maximization](image_generation/diffusion_alignment_as_variational_expectation-maximization.md)**

:   将扩散模型对齐形式化为变分 EM 算法：E-step 用 test-time search（soft Q 引导 + 重要性采样）探索高奖励多模态轨迹，M-step 通过 forward-KL 蒸馏将搜索结果写入模型参数，在图像生成和 DNA 序列设计上同时实现高奖励和高多样性。

**[Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](image_generation/diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)**

:   提出 Diffusion Blend，通过在推理时混合多个奖励微调模型的反向扩散过程来实现多偏好对齐：DB-MPA 支持任意奖励线性组合、DB-KLA 支持动态 KL 正则化控制、DB-MPA-LS 通过随机 LoRA 采样消除推理开销，理论上证明了混合近似的误差界并在实验中接近 MORL oracle 上界。

**[Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function](image_generation/diffusion_fine-tuning_via_reparameterized_policy_gradient_of_the_soft_q-function.md)**

:   提出 SQDF（Soft Q-based Diffusion Finetuning），通过无需训练的可微软 Q 函数估计和重参数化策略梯度，在 KL 正则化 RL 框架下微调扩散模型，配合折扣因子、一致性模型和离策略回放缓冲三个创新组件，在优化目标奖励的同时有效缓解奖励过优化问题，保持样本的自然性和多样性。

**[DiffusionNFT: Online Diffusion Reinforcement with Forward Process](image_generation/diffusionnft_online_diffusion_reinforcement_with_forward_process.md)**

:   提出 DiffusionNFT，一种全新的扩散模型在线 RL 范式：不在反向采样过程上做策略优化（如 GRPO），而是在前向过程上通过 flow matching 目标对正样本和负样本做对比式训练，定义隐式的策略改进方向，比 FlowGRPO 快 3-25×，且无需 CFG。

**[Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](image_generation/direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)**

:   提出 DrPose，通过直接奖励微调最大化 PoseScore（多视角潜变量图像与 GT 3D 姿态的骨骼一致性）+ KL 正则化防止 reward hacking，结合 DrPose15K 数据集（从 Motion-X 运动数据集采样 15K 多样姿态 + MIMO 视频生成器合成单视角图像），使多视角扩散模型在动态/杂技等困难姿态场景下的 3D 人体重建质量显著提升。

**[Directional Textual Inversion for Personalized Text-to-Image Generation](image_generation/directional_textual_inversion_for_personalized_text-to-image_generation.md)**

:   本文发现 Textual Inversion (TI) 学到的 token embedding 存在范数膨胀（norm inflation）问题，导致复杂 prompt 的文本对齐下降；提出 Directional Textual Inversion (DTI)，将 embedding 范数固定在分布内尺度、仅在单位超球面上用 Riemannian SGD 优化方向，结合 von Mises-Fisher 先验，显著提升 prompt 忠实度。

**[Discrete Adjoint Matching](image_generation/discrete_adjoint_matching.md)**

:   提出 Discrete Adjoint Matching（DAM），从纯统计学视角（而非控制论）推导出离散状态空间上的伴随变量，将连续域的 Adjoint Matching 推广到基于连续时间马尔可夫链（CTMC）的离散生成模型，实现了对扩散式 LLM（LLaDA-8B）的有效微调，在 Sudoku 上将准确率从 11.5% 提升至 89.2%。

**[DistillKac: Few-Step Image Generation via Damped Wave Equations](image_generation/distillkac_few-step_image_generation_via_damped_wave_equations.md)**

:   用阻尼波方程（telegrapher equation）及其随机 Kac 表示替代 Fokker-Planck 方程作为生成模型的概率流基础，实现有限速度传播的概率流，并提出端点蒸馏（endpoint distillation）方法实现少步生成，在 CIFAR-10 上 4 步 FID=4.14、1 步 FID=5.66。

**[Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study](image_generation/does_semantic_noise_initialization_transfer_from_images_to_videos_a_paired_diagn.md)**

:   通过严格的 prompt 级别配对统计检验，发现将图像领域的 semantic noise initialization（golden noise）迁移到视频扩散模型后，temporal 指标呈微弱正向趋势但统计不显著（p≈0.17），噪声空间诊断揭示了方向稳定性不足和时空频率结构差异是根因。

**[DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](image_generation/doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)**

:   提出DoFlow，一种基于连续正则化流（CNF）的因果生成模型，在因果DAG上统一实现观测、干预和反事实时间序列预测，并可通过显式似然进行异常检测，在合成和真实医疗数据上验证了有效性。

**[DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](image_generation/dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)**

:   首个将 FLUX (DiT) 的强生成先验引入拖拽编辑的框架，通过区域级仿射监督替代传统点级监督，配合梯度掩码硬约束和 adapter 增强反演，大幅提升拖拽编辑质量。

**[Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing](image_generation/draw-in-mind_rebalancing_designer-painter_roles_in_unified_multimodal_models_ben.md)**

:   指出当前统一多模态模型中理解模块仅作翻译器而生成模块被迫同时充当"设计师"和"画家"的职责失衡问题，通过构建 DIM 数据集（14M 长上下文文图对 + 233K CoT 编辑蓝图）将设计责任转移给理解模块，4.6B 参数即超越 5 倍大的模型。

**[Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](image_generation/dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)**

:   提出 Dual-Solver，通过三组可学习参数（预测类型插值 $\gamma$、积分域选择 $\tau$、残差调整 $\kappa$）泛化扩散模型多步采样器，用冻结预训练分类器（MobileNet/CLIP）的分类损失学习参数（无需教师轨迹），在 3-9 NFE 低步区间全面优于 DPM-Solver++ 等方法。

**[Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?](image_generation/easier_painting_than_thinking_can_text-to-image_models_set_the_stage_but_not_dir.md)**

:   提出 T2I-CoReBench，首个同时系统评估 T2I 模型**组合能力**(Composition)和**推理能力**(Reasoning)的综合性基准，涵盖 12 个评估维度、1080 条高难度 prompt 和约 13500 个 checklist 问题，通过对 38 个模型的大规模评测揭示：推理能力远远落后于组合能力，是当前 T2I 生成的核心瓶颈。

**[EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](image_generation/editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)**

:   构建了一个包含 200K 人工标注偏好对的高质量数据集 EditReward-Data，训练出 EditReward 奖励模型，在多个图像编辑评估基准上达到 SOTA 的人类对齐度，并验证其作为数据筛选器可显著提升下游编辑模型性能。

**[EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](image_generation/editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)**

:   提出首个系统性的"基准评测→奖励模型→强化学习训练"图像编辑 RL 管线：构建 EditReward-Bench 基准，训练 EditScore 系列奖励模型（7B-72B，超过 GPT-5），并成功将其用于 Online RL 训练显著提升编辑模型性能。

**[Efficient Adversarial Attacks on High-dimensional Offline Bandits](image_generation/efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)**

:   揭示了离线多臂老虎机（MAB）评估框架的安全漏洞：攻击者只需对公开的奖励模型权重进行极小的不可感知扰动，就能完全劫持 bandit 的决策行为，且所需扰动范数随输入维度增加而降低（$\widetilde{\mathcal{O}}(d^{-1/2})$），使基于图像的生成模型评估特别脆弱。

**[Eliminating VAE for Fast and High-Resolution Generative Detail Restoration](image_generation/eliminating_vae_for_fast_and_high-resolution_generative_detail_restoration.md)**

:   通过用 ×8 pixel-(un)shuffle 替代 VAE 的编码器和解码器，将潜空间扩散超分（GenDR）逆转为像素空间超分（GenDR-Pix），结合多阶段对抗蒸馏和 PadCFG 推理策略，实现 2.8× 加速和 60% 显存节省，同时保持可忽略的视觉退化，首次实现 1 秒内 4K 图像恢复仅需 6GB 显存。

**[Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning](image_generation/embracing_discrete_search_a_reasonable_approach_to_causal_structure_learning.md)**

:   提出 FLOP（Fast Learning of Order and Parents），一个面向线性模型的基于得分的因果发现算法，通过快速父节点选择与迭代 Cholesky 得分更新大幅降低运行时间，使得迭代局部搜索（ILS）变得可行，在标准因果发现基准上实现近乎完美的图恢复，重新确立离散搜索在因果发现中的合理地位。

**[Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance](image_generation/error_as_signal_stiffness-aware_diffusion_sampling_via_embedded_runge-kutta_guid.md)**

:   提出 ERK-Guid，利用嵌入式 Runge-Kutta 求解器的阶差误差作为 guidance 信号，在刚性区域自适应纠正局部截断误差（LTE），无需额外网络评估即可提升扩散模型采样质量。

**[Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](image_generation/event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)**

:   提出 Event-T2M 框架，将文本提示分解为事件级别的原子动作，结合 TMR 编码器和事件级交叉注意力（ECA）模块注入 Conformer 扩散模型，显著提升多事件复杂动作生成的质量和语义对齐。

**[Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models](image_generation/everything_in_its_place_benchmarking_spatial_intelligence_of_text-to-image_model.md)**

:   提出 SpatialGenEval 基准，通过 1,230 条长且信息密集的提示覆盖 10 个空间子领域，系统评估 23 个 SOTA T2I 模型的空间智能，揭示空间推理是主要瓶颈；同时构建 SpatialT2I 数据集实现数据中心的空间智能提升。

**[Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model](image_generation/evolutionary_caching_to_accelerate_your_off-the-shelf_diffusion_model.md)**

:   提出 ECAD（Evolutionary Caching to Accelerate Diffusion models），利用遗传算法在速度-质量 Pareto 前沿上自动搜索最优缓存调度策略，无需修改模型参数，仅用 100 条校准提示即可实现扩散模型 2-3 倍推理加速并保持甚至提升生成质量。

**[Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](image_generation/exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)**

:   提出 Bias-Guided Prompt Search (BGPS)，通过结合 LLM 解码引导和扩散模型中间层属性分类器，自动发现可解释的、能最大化暴露 T2I 模型隐藏社会偏见的文本提示，即使对已去偏的模型也能揭示残留偏见。

**[Factuality Matters: When Image Generation and Editing Meet Structured Visuals](image_generation/factuality_matters_when_image_generation_and_editing_meet_structured_visuals.md)**

:   首个系统性研究结构化图像（图表、数学公式、示意图等）生成与编辑的工作，构建了130万对代码对齐的训练数据集（含 CoT 推理标注）、统一的 VLM+扩散模型架构以及包含1700+样本的 StructBench 基准评测，揭示了推理能力是当前模型处理结构化视觉内容的关键瓶颈。

**[SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)**

:   提出 Single-Step Completion Policy (SSCP)，通过在流匹配框架中预测"完成向量"（从任意中间状态到目标动作的归一化方向），将多步生成策略压缩为单步推理，在 D4RL 上与多步扩散/流策略持平但训练快 64×、推理快 4.7×，并扩展到 GCRL 中将层级策略扁平化。

**[Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](image_generation/flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)**

:   提出两阶段训练框架Flow2GAN，先用改进的Flow Matching学习生成能力，再用GAN微调实现少步（1/2/4步）高保真音频生成，结合多分辨率网络架构处理不同时频分辨率的傅里叶系数。

**[Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)**

:   通过在流匹配训练中注入可控噪声扩大策略覆盖范围，并结合熵引导的采样机制在在线微调时动态平衡探索与利用，在有限交互预算下显著提升离线到在线RL的样本效率。

**[FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](image_generation/flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)**

:   首次将条件流匹配(CFM)作为端到端概率生成模型应用于降水临近预报，在压缩潜空间中学习噪声到数据的直接映射，以更少的采样步数超越扩散模型的预测精度和概率性能。

**[FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](image_generation/flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)**

:   提出FlowCast框架，将投机解码思想引入Flow Matching模型，利用速度场的局部平滑性将当前速度预测作为零成本draft外推未来状态，通过MSE验证选择性跳过冗余步骤，实现>2.5×加速且无质量损失。

**[Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](image_generation/follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)**

:   提出 Follow-Your-Shape，一个无需训练和掩码的形状感知编辑框架，通过计算反演与编辑轨迹间的 token 级速度差异构建 Trajectory Divergence Map (TDM) 来精确定位编辑区域，配合分阶段 KV 注入实现大幅形状变换且严格保持背景。

**[Free Lunch for Stabilizing Rectified Flow Inversion](image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)**

:   提出PMI（Proximal-Mean Inversion）和mimic-CFG两个无训练方法，通过将速度场向其历史均值做近端梯度校正来稳定Rectified Flow反演，在PIE-Bench上以更少的NFE达到SOTA的重建和编辑质量。

**[From Parameters to Behaviors: Unsupervised Compression of the Policy Space](image_generation/from_parameters_to_behaviors_unsupervised_compression_of_the_policy_space.md)**

:   基于流形假设提出策略空间的无监督压缩——用行为重建损失（而非参数重建损失）训练自编码器将高维策略参数空间 $\Theta \subseteq \mathbb{R}^P$ 压缩到低维潜在行为空间 $\mathcal{Z} \subseteq \mathbb{R}^k$（最高 121801:1 压缩比），在 Mountain Car、Reacher、Hopper、HalfCheetah 等环境上验证了行为流形的内在维度取决于环境复杂度而非网络大小，且在潜在空间中做 PGPE 优化可在 7/8 个任务上比 PPO、SAC 等 SOTA 收敛更快。

**[From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](image_generation/from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)**

:   提出 TensorAR，将标准 AR 图像生成从 next-token prediction 升级为 next-tensor prediction：每步预测重叠 tensor（一组连续 token），后续 tensor 与前序重叠实现迭代精修；引入离散扩散噪声机制解决训练信息泄漏问题，作为即插即用模块兼容 LlamaGen / Open-MAGVIT2 / Janus-Pro 等 AR 模型，在 class-to-image 和 text-to-image 任务上持续提升生成质量。

**[GenCP: Towards Generative Modeling Paradigm of Coupled Physics](image_generation/gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)**

:   提出 GenCP，将耦合多物理场仿真建模为概率密度演化问题，利用 flow matching 从解耦数据学习条件速度场，推理时通过 Lie-Trotter 算子分裂合成耦合解，实现"解耦训练、耦合推理"，并提供理论误差可控保证。

**[GenDR: Lighten Generative Detail Restoration](image_generation/gendr_lighten_generative_detail_restoration.md)**

:   提出GenDR——面向生成式细节复原的轻量单步扩散超分模型：识别T2I和SR任务目标的根本分歧（T2I需多步+4通道 vs SR需少步+16通道）→构建定制SD2.1-VAE16基础模型（0.9B，通过REPA表示对齐扩展潜在空间而不增加模型规模）→提出CiD/CiDA一致性分数恒等蒸馏（将SR特定先验融入score distillation + 对抗学习 + 表示对齐）→极简pipeline仅含UNet+VAE→77ms推理在所有质量和效率指标上超越现有SOTA。

**[Generalization of Diffusion Models Arises with a Balanced Representation Space](image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)**

**[Generating Directed Graphs with Dual Attention and Asymmetric Encoding](image_generation/generating_directed_graphs_with_dual_attention_and_asymmetric_encoding.md)**

:   提出 Directo，首个基于离散流匹配（Discrete Flow Matching）的有向图生成模型，通过方向感知的双注意力机制和非对称位置编码捕获有向边的方向依赖，同时建立了有向图生成的标准化评测体系。

**[GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models](image_generation/geodiv_framework_for_measuring_geographical_diversity_in_text-to-image_models.md)**

:   提出 GeoDiv 框架，利用 LLM 和 VLM 的世界知识，从社会经济视觉指数（SEVI）和视觉多样性指数（VDI）两个维度系统评估 T2I 模型的地理多样性，揭示了模型对印度、尼日利亚等国家存在系统性贫困化偏见。

**[GGBall: Graph Generative Model on Poincaré Ball](image_generation/ggball_graph_generative_model_on_poincaré_ball.md)**

:   提出 GGBall，首个完全基于 Poincaré 球模型的图生成框架，通过双曲向量量化自编码器（HVQVAE）和黎曼流匹配先验，在层次图和分子图生成上达到 SOTA，在层次图数据集上平均生成误差降低 18%。

**[GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](image_generation/glass_flows_reward_alignment_diffusion.md)**

:   提出 GLASS (Gaussian Latent Sufficient Statistic) Flows——一种"流模型中的流模型"新采样范式，通过高斯充分统计量重参数化将随机马尔可夫转移 $p_{t'|t}(x_{t'} | x_t)$ 重铸为内部 ODE 求解问题（复用预训练去噪器，无需重训），在无需权衡 ODE 效率和 SDE 随机性的条件下实现 Feynman-Kac Steering，在 FLUX 文生图模型上一致超越 Best-of-N ODE 基线，刷新推理时奖励对齐 SOTA。

**[GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](image_generation/glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)**

:   提出GLYPH-SR，一个视觉-语言引导的扩散框架，通过双分支Text-SR融合ControlNet和ping-pong调度器同时优化图像质量和文本可读性，在SVT ×8上将OCR F1提升15.18个百分点。

**[Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](image_generation/hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)**

:   提出HECRL，一个层次化实体中心离线目标条件RL框架，结合基于价值的GCRL智能体和因子化子目标扩散模型，在多实体长时域任务中实现150%+的成功率提升。

**[HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](image_generation/hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)**

:   提出HierLoc，将地理定位重新建模为双曲空间中的图像-实体对齐问题，用24万个地理实体嵌入替代500万+图像嵌入，在OSV5M上降低19.5%平均测地误差并将子区域准确率提升43%。

**[HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](image_generation/hog-diff_higher-order_guided_diffusion_for_graph_generation.md)**

:   本文提出 HOG-Diff，一个利用高阶拓扑结构（如环、三角形、motif）作为生成引导的图扩散框架，通过胞复形过滤（CCF）提取高阶骨架并结合广义 OU 扩散桥实现"由粗到细"的渐进式图生成，在分子和通用图生成的 8 个基准上取得了 SOTA 性能。

**[Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_generation/image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)**

:   Recall 提出首个多模态引导的攻击框架，通过在隐空间中优化对抗图像 prompt（仅需一张参考图像），配合原始文本 prompt 利用扩散模型的 image-conditioning 通道，在 10 种 SOTA 遗忘方法上平均 ASR 达 65%~97%，显著超越纯文本攻击方法，揭示当前遗忘机制对图像模态攻击的脆弱性。

**[Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)](image_generation/improved_object-centric_diffusion_learning_with_registers_and_contrastive_alignm.md)**

:   提出 CODA 框架，通过引入 register slots 吸收残余注意力、微调交叉注意力投影以及对比对齐损失，解决基于扩散模型的物体中心学习中的 slot 纠缠和弱对齐问题，在合成和真实数据集上显著提升物体发现和组合式生成质量。

**[Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](image_generation/improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)**

:   提出 Unmasking Policy Optimization（UPO），将 Masked Diffusion Model 的去噪过程建模为 KL 正则化 MDP，通过强化学习训练轻量级的 unmasking 策略模型来替代 max-confidence 等启发式调度器，在理论和实验上均证明学习到的策略能生成更接近真实数据分布的样本。

**[Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](image_generation/infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)**

:   首次系统性地对比 Visual Autoregressive (VAR) 模型和扩散模型在组合文本-图像对齐上的表现，在 T2I-CompBench++ 和 GenEval 两个基准上评测 6 个 T2I 模型，发现 Infinity-8B 在几乎所有组合维度上取得最强表现，VAR 架构在组合生成方面展现出显著优势。

**[Intention-Conditioned Flow Occupancy Models](image_generation/intention-conditioned_flow_occupancy_models.md)**

:   提出 InFOM，利用流匹配（flow matching）构建意图条件化的占据模型（occupancy model），通过变分推断推理数据中的潜在意图，实现无标注数据上的 RL 预训练，在 36 个状态任务和 4 个视觉任务上取得 1.8× 中位回报提升和 36% 成功率提升。

**[JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](image_generation/jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)**

:   提出 JointDiff，一个联合连续-离散扩散框架，首次将高斯扩散（用于轨迹）和多项式扩散（用于控球事件）统一建模，同时引入 CrossGuid 模块支持弱控球引导和文本引导的语义可控生成，在体育多智能体轨迹生成上达到 SOTA。

**[K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge](image_generation/k-sort_eval_efficient_preference_evaluation_for_visual_generation_via_corrected_.md)**

:   提出 K-Sort Eval 框架，通过后验校正和动态匹配策略，使 VLM 能可靠高效地替代人类进行视觉生成模型的偏好评估，通常只需不到 90 次模型运行即可得出与人类 Arena 一致的结果。

**[Laplacian Multi-scale Flow Matching for Generative Modeling](image_generation/laplacian_multi-scale_flow_matching_for_generative_modeling.md)**

:   提出 LapFlow，将图像分解为拉普拉斯金字塔残差，通过混合 Transformer（MoT）架构和因果注意力并行建模不同尺度，在减少计算量的同时提升生成质量。

**[Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](image_generation/large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)**

:   提出 rCM（score-regularized continuous-time consistency model），首次将连续时间一致性蒸馏扩展到 14B 参数的文生图/视频模型，通过结合前向散度（一致性）和反向散度（score蒸馏），在保持多样性的同时匹配 DMD2 的质量，实现 15-50× 加速。

**[Latent Diffusion Model without Variational Autoencoder](image_generation/latent_diffusion_model_without_variational_autoencoder.md)**

:   提出 SVG，用冻结的 DINOv3 自监督特征替代 VAE 潜在空间构建扩散模型，通过轻量残差编码器补充细粒度细节，实现更快训练、更高效推理和跨任务通用的视觉表征。

**[Learning a Distance Measure from the Information-Estimation Geometry of Data](image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)**

:   提出 Information-Estimation Metric (IEM)，一种由数据概率密度几何诱导的新型距离函数，通过比较不同噪声水平下的 score 向量场来度量信号间距离，无监督训练的 IEM 在预测人类感知判断上可媲美有监督方法。

**[LLM2Fx-Tools: Tool Calling for Music Post-Production](image_generation/llm2fx-tools_tool_calling_for_music_post-production.md)**

:   提出 LLM2Fx-Tools，首个将 LLM 工具调用应用于音效模块的框架，通过多模态 LLM 理解音频输入，利用 CoT 推理选择音效类型、确定顺序并估计参数，实现可解释和可控的音乐后期制作。

**[Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](image_generation/locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)**

:   提出 Locality-aware Parallel Decoding (LPD)，通过灵活并行化自回归建模架构和局部性感知的生成顺序调度，将 256×256 图像的生成步数从 256 降至 20，实现至少 3.4× 的延迟降低。

**[Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](image_generation/localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)**

:   HiRM 提出"更新位置与擦除目标解耦"的概念擦除策略——仅更新 CLIP 文本编码器第一层的权重，但将擦除监督施加在最后一层的高层语义表征上，通过引导目标概念表征偏向随机方向（HiRM-R）或语义方向（HiRM-S），在 UnlearnCanvas 和 NSFW 基准上实现风格/物体/裸体的高效擦除，且可零样本迁移到 Flux 架构。

**[Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](image_generation/loopholing_discrete_diffusion_deterministic_bypass_of_the_sampling_wall.md)**

:   识别离散扩散模型中的"采样壁"问题（分类分布信息在采样后坍塌为 one-hot 向量），提出 Loopholing 机制引入确定性潜在路径传播丰富的分布信息，将生成困惑度降低最多 61%，大幅缩小与自回归模型的差距。

**[LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](image_generation/lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)**

:   提出 LVTINO，首个基于视频一致性模型（VCM）先验的零样本视频逆问题求解器，通过在 VCM 采样过程中注入无需自动微分的测量一致性约束，在超分辨率、去模糊、修复等多种视频逆问题上以极少的神经网络函数评估（NFE）实现了超越逐帧图像方法的感知质量和时间一致性。

**[MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](image_generation/mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)**

:   提出 MAC-AMP，首个闭环多智能体协作系统，将抗菌肽（AMP）设计重构为协调多智能体优化问题，通过 AI 模拟同行评审和自适应奖励设计实现多目标优化。

**[Market Games for Generative Models: Equilibria, Welfare, and Strategic Entry](image_generation/market_games_for_generative_models_equilibria_welfare_and_strategic_entry.md)**

:   形式化三层模型-平台-用户市场博弈，分析生成模型竞争下纯策略 Nash 均衡的存在条件、市场结构、社会福利影响，并设计模型提供者的最优进入策略。

**[Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](image_generation/mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)**

:   提出 Mod-Adapter，一种无需测试时微调的多概念个性化方法，通过在 DiT 的调制（modulation）空间中预测概念特定的调制方向，实现对物体和抽象概念（姿态、光照、材质等）的解耦化定制生成，在多概念个性化上大幅超越现有方法。

**[Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](image_generation/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)**

:   将通常被视为负面现象的"模型坍缩"（model collapse）重新定位为机器遗忘的工具，提出PMC方法——通过在保留数据和模型自身生成数据上迭代微调来实现针对性信息删除，无需在遗忘目标上直接优化，从理论和实验两方面证明了其有效性。

**[MOLM: Mixture of LoRA Markers](image_generation/molm_mixture_of_lora_markers.md)**

:   提出 MOLM 水印框架，将 LoRA 适配器重新解释为水印载体，通过二进制密钥驱动的路由机制在冻结生成模型中嵌入可验证、鲁棒的水印，无需逐密钥重训练。

**[Monocular Normal Estimation via Shading Sequence Estimation](image_generation/monocular_normal_estimation_via_shading_sequence_estimation.md)**

:   本文提出了RoSE方法，将单目法线估计问题重新定义为着色序列（Shading Sequence）估计问题，利用图像到视频（Image-to-Video）生成模型预测多光照下的着色序列，再通过简单的最小二乘法将着色序列转换为法线图，在真实世界基准数据集上达到SOTA性能。

**[Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening](image_generation/motion_prior_distillation_in_time_reversal_sampling_for_generative_inbetweening.md)**

:   提出 Motion Prior Distillation (MPD)，一种推理时蒸馏方法，将前向路径的运动残差蒸馏到后向路径中，从根本上解决了时间反转采样中双向运动先验冲突的问题，无需额外训练即可实现更连贯的生成式帧插值。

**[Multi-agent Coordination via Flow Matching](image_generation/multi-agent_coordination_via_flow_matching.md)**

:   提出 MAC-Flow，先用 Flow Matching 学习中心化联合行为分布，再通过 IGM（Individual-Global-Max）分解将其蒸馏为去中心化的单步策略，结合 Q 值最大化进行行为正则化训练，在 4 个基准 12 个环境 34 个数据集上实现了约 14.5 倍于扩散方法的推理加速，同时保持了与扩散策略可比的协调性能。

**[MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion](image_generation/mvcustom_multi-view_customized_diffusion_via_geometric_latent_rendering_and_comp.md)**

:   提出多视角定制（multi-view customization）新任务并设计 MVCustom 框架，通过视频扩散骨干网络结合密集时空注意力实现整体帧一致性，在推理阶段引入深度感知特征渲染和一致性感知潜码补全两项技术，首次同时实现相机位姿控制、主体身份保持和跨视角几何一致性。

**[Neon: Negative Extrapolation From Self-Training Improves Image Generation](image_generation/neon_negative_extrapolation_image_generation.md)**

:   提出 Neon，一种仅需 <1% 额外训练计算的后处理方法：先用模型自身生成的合成数据微调导致退化，再反向外推远离退化权重，证明 mode-seeking 采样器导致合成/真实数据梯度反对齐，因此负外推等价于向真实数据分布优化，在 ImageNet 256×256 上将 xAR-L 提升至 SOTA FID 1.02。

**[NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](image_generation/neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)**

:   提出 NeuralOS，使用 RNN 状态追踪 + 扩散渲染器的双组件架构，直接从用户输入事件（鼠标移动/点击/键盘）预测操作系统图形界面帧序列，首次实现用神经生成模型模拟操作系统。

**[Next Visual Granularity Generation](image_generation/next_visual_granularity_generation.md)**

:   提出 Next Visual Granularity (NVG) 生成框架，将图像分解为不同粒度级别的结构化序列，从全局布局到精细细节逐级生成，相比 VAR 系列在 FID 上一致提升。

**[No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](image_generation/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)**

:   提出 MoFit，首个面向无标题场景的扩散模型成员推断攻击框架，通过构建过拟合于目标模型的代理图像和条件嵌入，利用成员样本对条件错配的不对称敏感性实现有效推断。

**[Offline Reinforcement Learning with Generative Trajectory Policies](image_generation/offline_reinforcement_learning_with_generative_trajectory_policies.md)**

:   提出生成轨迹策略（GTP），通过统一视角将扩散模型、流匹配、一致性模型视为ODE解映射的特例，学习完整的连续时间轨迹解映射，并引入分数近似和优势权重两个适配技术，在D4RL基准上实现SOTA。

**[Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](image_generation/pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)**

:   提出 Pareto-Conditioned Diffusion (PCD)，将离线多目标优化重构为条件采样问题，直接以目标权衡为条件生成高质量解，无需显式代理模型，在多种基准上实现最佳一致性。

**[PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models](image_generation/pcpo_proportionate_credit_policy_optimization_for_aligning_image_generation_mode.md)**

:   提出 PCPO，通过稳定目标重构和原则性时间步重加权，修正扩散/流模型策略梯度中固有的不成比例信用分配问题，显著加速收敛并缓解模型崩溃。

**[PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](image_generation/pi-light_physics-inspired_diffusion_for_full-image_relighting.md)**

:   提出 π-Light（PI-Light），一个两阶段的全图像重光照框架：第一阶段通过物理引导的扩散模型进行内蕴属性（albedo、法线、roughness 等）分解，第二阶段通过物理引导的神经渲染模块实现光照条件下的重新渲染，引入批量感知注意力机制和物理启发损失以实现对真实场景的优秀泛化能力。

**[PolyGraph Discrepancy: a classifier-based metric for graph generation](image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)**

:   提出 PolyGraph Discrepancy (PGD)，通过训练分类器区分真实图和生成图来逼近 Jensen-Shannon 距离的变分下界，解决了 MMD 指标缺乏绝对尺度、不同描述符间不可比、小样本高偏差高方差的三大核心问题。

**[Pseudo-Nonlinear Data Augmentation: A Constrained Energy Minimization Viewpoint](image_generation/pseudo-nonlinear_data_augmentation_a_constrained_energy_minimization_viewpoint.md)**

:   基于能量模型和信息几何的对偶平坦结构，提出无需训练、高效可控的数据增强方法，通过正向投影（编码）和反向投影（解码）在统计流形上实现跨模态增强。

**[Purrception: Variational Flow Matching for Vector-Quantized Image Generation](image_generation/purrception_variational_flow_matching_for_vector-quantized_image_generation.md)**

:   提出 Purrception，一种将变分流匹配（Variational Flow Matching）适配到向量量化（VQ）隐空间的图像生成方法，通过在连续嵌入空间中计算速度场的同时学习编码本索引上的分类后验分布，桥接了连续传输动力学和离散监督，在 ImageNet-1k 256×256 上实现了更快的训练收敛和与 SOTA 可比的 FID 分数。

**[Pyramidal Patchification Flow for Visual Generation](image_generation/pyramidal_patchification_flow_for_visual_generation.md)**

:   提出 Pyramidal Patchification Flow (PPFlow)，通过在高噪声时间步使用大 patch、低噪声时使用小 patch，在保持生成质量的同时实现 1.6-2.0× 去噪加速，且无需重噪声技巧。

**[QVGen: Pushing the Limit of Quantized Video Generative Models](image_generation/qvgen_pushing_the_limit_of_quantized_video_generative_models.md)**

:   提出 QVGen，一种面向视频扩散模型的量化感知训练（QAT）框架，通过引入辅助模块降低梯度范数以改善收敛性，并设计秩衰减策略在训练中逐步消除辅助模块的推理开销，首次在 4-bit 量化下实现接近全精度的视频生成质量。

**[RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation](image_generation/refany3d_3d_asset-referenced_diffusion_models_for_image_generation.md)**

:   提出 RefAny3D，一个 3D 资产参考的图像生成框架，通过联合建模 RGB 图像和点图（point map）的双分支生成策略，实现生成图像与 3D 参考资产在几何和纹理上的精确一致性。

**[Referring Layer Decomposition](image_generation/referring_layer_decomposition.md)**

:   提出 Referring Layer Decomposition (RLD) 任务，根据用户提供的灵活提示（空间/文本/混合）从单张 RGB 图像中预测完整的 RGBA 图层，并构建了包含 111 万样本的 RefLade 数据集和自动评估协议。

**[RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](image_generation/rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)**

:   提出 RIDER 框架，首次将强化学习引入 RNA 3D 逆向设计，先预训练条件扩散模型 RIDE 学习序列-结构关系，再用 RL 微调以直接优化 3D 结构相似性而非序列恢复率，在所有 3D 自一致性指标上实现超过 100% 的提升。

**[RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](image_generation/rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)**

:   提出 RMFlow，在 1-NFE MeanFlow 传输后加入一步噪声注入精炼来弥补单步传输的误差，同时在训练中加入最大似然目标来最小化学习分布与目标分布间的 KL 散度，在 T2I、分子生成、时间序列生成上实现接近 SOTA 的 1-NFE 结果。

**[RNE: plug-and-play diffusion inference-time control and energy-based training](image_generation/rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)**

:   提出 Radon-Nikodym 估计器 (RNE)，基于路径分布间的密度比揭示边际密度与转移核的基本联系，提供统一的即插即用框架，同时实现扩散密度估计、推理时控制和能量扩散训练。

**[Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance](image_generation/routing_matters_in_moe_scaling_diffusion_transformers_with_explicit_routing_guid.md)**

:   提出 ProMoE，一种针对扩散 Transformer 的 MoE 框架，通过两步路由器（条件路由 + 原型路由）和路由对比损失提供显式语义引导，促进专家特化，在 ImageNet 上显著超越现有 MoE 和稠密模型。

**[SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions](image_generation/safeflowmatcher_safe_and_fast_planning_using_flow_matching_with_control_barrier_.md)**

:   提出 SafeFlowMatcher，一种将流匹配与控制障碍函数 (CBF) 结合的安全规划框架，通过预测-修正 (PC) 积分器将路径生成与安全认证解耦，在保持流匹配高效性的同时提供形式化安全保证。

**[Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)**

:   提出 DiME，一种沿扩散后验时间边缘积分的模型证据估计器，无需先验评分或密度评估，仅用少量后验样本（如 20 个）即可准确估计扩散模型先验下的模型证据，用于先验选择和模型验证。

**[scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](image_generation/scdfm_distributional_flow_matching_model_for_robust_single-cell_perturbation_pre.md)**

:   提出 scDFM，基于条件流匹配（CFM）的生成式框架，通过 MMD 正则化保证分布级保真度，并设计 PAD-Transformer 骨干处理噪声稀疏的单细胞数据，在组合扰动预测上比最强基线 CellFlow 的 MSE 降低 19.6%。

**[Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek](image_generation/seek-cad_a_self-refined_generative_modeling_for_3d_parametric_cad_using_local_in.md)**

:   提出 Seek-CAD，首个基于本地部署的推理 LLM（DeepSeek-R1）的无训练 CAD 参数化模型生成框架，通过分步视觉反馈与思维链 (CoT) 协同实现自我精炼，并设计新的 SSR 三元组设计范式支持复杂 CAD 模型生成。

**[Self-Improving Loops for Visual Robotic Planning](image_generation/self-improving_loops_for_visual_robotic_planning.md)**

:   提出 SILVR 框架，通过迭代更新域内视频生成模型在自收集的在线轨迹上进行微调，实现视觉机器人规划器在未见任务上的持续自我改进，在 MetaWorld 和真实机器人上实现高达 285% 的性能提升。

**[SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP](image_generation/semobridge_semantic_modality_bridge_for_efficient_few-shot_adaptation_of_clip.md)**

:   提出 SeMoBridge，一种轻量级语义模态桥，通过将图像嵌入映射到文本模态，将不可靠的模态内（图像-图像）比较转换为可靠的模态间（文本-图像）比较，以极低训练开销在少样本分类中超越现有方法。

**[SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](image_generation/senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)**

:   提出 SenseFlow，通过隐式分布对齐（IDA）和段内引导（ISG）将分布匹配蒸馏（DMD）扩展到大规模 flow-based 文生图模型（SD 3.5 Large 8B / FLUX.1 dev 12B），实现 4 步高质量图像生成。

**[SMOTE and Mirrors: Exposing Privacy Leakage from Synthetic Minority Oversampling](image_generation/smote_and_mirrors_exposing_privacy_leakage_from_synthetic_minority_oversampling.md)**

:   首次系统研究 SMOTE 的隐私泄露问题，提出 DistinSMOTE 和 ReconSMOTE 两种攻击，证明 SMOTE 本质上是非隐私保护的，且过度暴露少数类记录。

**[SoFlow: Solution Flow Models for One-Step Generative Modeling](image_generation/soflow_solution_flow_models_for_one-step_generative_modeling.md)**

:   提出 Solution Flow Models (SoFlow)，直接学习速度 ODE 的解函数 $f(x_t, t, s)$（将 $t$ 时刻的 $x_t$ 映射到 $s$ 时刻的解），通过 Flow Matching 损失 + 无需 JVP 的解一致性损失从头训练，在 ImageNet 256 上 1-NFE FID 优于 MeanFlow（XL/2: 2.96 vs 3.43）。

**[SongEcho: Towards Cover Song Generation via Instance-Adaptive Element-wise Linear Modulation](image_generation/songecho_towards_cover_song_generation_via_instance-adaptive_element-wise_linear.md)**

:   提出 SongEcho 框架，通过实例自适应元素级线性调制（IA-EiLM）实现翻唱歌曲生成，在保持原始歌曲旋律轮廓的同时生成新的歌声和伴奏。

**[SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)**

:   SPEED 提出基于零空间（null space）约束的闭式模型编辑方法，通过影响力先验过滤（IPF）、定向先验增强（DPA）和不变等式约束（IEC）三种互补技术精化保留集，实现可扩展（5 秒内擦除 100 个概念）、精确（非目标概念语义零损失）且高效的概念擦除。

**[SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](image_generation/ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)**

:   提出 Scaled Spatial Guidance (SSG)，一种无需训练的推理时引导方法，通过频域先验构建和语义残差放大，增强视觉自回归模型的粗到细层级生成质量。

**[Steer Away From Mode Collisions: Improving Composition In Diffusion Models](image_generation/steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)**

:   针对扩散模型多概念 prompt 中的概念缺失/碰撞问题，提出"模式碰撞"假说（联合分布与单概念分布的模式重叠），设计 CO3（Concept Contrasting Corrector）通过在 Tweedie 均值空间中组合校正分布 $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$ 来远离退化模式，实现即插即用、无梯度、模型无关的组合生成改进。

**[Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution](image_generation/step-aware_residual-guided_diffusion_for_eeg_spatial_super-resolution.md)**

:   提出 SRGDiff，一种步感知残差引导的扩散模型，将 EEG 空间超分辨率重新定义为动态条件生成任务，通过每步残差方向校正和步依赖仿射调制实现高保真重建。

**[Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)**

:   本文提出S²-Guidance，通过在去噪过程中**随机丢弃transformer block激活子网络**作为弱模型进行自引导，无需额外训练即可修正CFG的次优预测，在文生图和文生视频任务上一致超越CFG及其他高级引导策略。

**[TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex](image_generation/tavae_a_vae_with_adaptable_priors_explains_contextual_modulation_in_the_visual_c.md)**

:   扩展 VAE 形式主义提出 Task-Amortized VAE (TAVAE)，通过在已学表示上灵活学习任务特异性先验来解释视觉皮层 V1 中的上下文调制现象，包括方向辨别任务中训练刺激与测试刺激不匹配时出现的双模态群体响应。

**[Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions](image_generation/temporal_concept_dynamics_in_diffusion_models_via_prompt-conditioned_interventio.md)**

:   提出 PCI（Prompt-Conditioned Intervention）框架，通过在去噪轨迹不同时间步切换文本提示，量化概念何时在扩散模型中锁定，并将此发现应用于时间感知的图像编辑。

**[Test-Time Iterative Error Correction for Efficient Diffusion Models](image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)**

:   提出 IEC（Iterative Error Correction），一种测试时的即插即用方法，通过迭代修正高效扩散模型的推理误差，将误差累积从指数增长降低为线性增长。

**[The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](image_generation/the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)**

:   本文系统研究了文本提示（prompt）复杂度对T2I模型合成数据的质量、多样性和一致性三个关键维度的影响，提出了新的评估框架，并发现提示扩展（prompt expansion）作为一种推理时干预手段能最优地平衡多样性与美学质量。

**[The Spacetime of Diffusion Models: An Information Geometry Perspective](image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)**

:   从信息几何视角提出扩散模型的"时空"概念，证明标准拉回几何在扩散模型中退化为直线，转而引入 Fisher-Rao 度量的时空几何，并导出可实际计算的散度编辑距离（DiffED）和转移路径采样方法。

**[There and Back Again: On the Relation between Noise and Image Inversions in Diffusion Models](image_generation/there_and_back_again_on_the_relation_between_noise_and_image_inversions_in_diffu.md)**

:   深入分析 DDIM 反转的误差机制，发现潜在编码在平滑图像区域（如天空）呈现低多样性和高相关性，并追溯到反转初始步骤的噪声预测不准确，提出用正向扩散替代前几步反转的简单修复方案。

**[Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)**

:   将 reward-guided 图像编辑重新建模为轨迹最优控制问题，将扩散/Flow模型的反向过程视为可控轨迹，通过基于 Pontryagin 最大值原理（PMP）的伴随状态迭代优化整条轨迹，在无需训练的情况下实现有效的奖励引导编辑且不发生 reward hacking。

**[Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations](image_generation/translate_policy_to_language_flow_matching_generated_rewards_for_llm_explanation.md)**

:   提出一个通用框架，利用Rectified Flow生成分布式奖励来训练解释生成LLM，通过连续归一化流（CNF）捕捉人类对解释评判的多元概率特性，并在理论上证明CNF能有效恢复真实人类奖励分布，在SMAC、MMLU、MathQA等任务上显著超越RLHF/RLAIF基线。

**[TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)**

:   提出 TwinFlow：通过将 flow matching 时间区间从 $[0,1]$ 扩展到 $[-1,1]$，构造"孪生轨迹"形成自对抗信号，使模型无需判别器或冻结教师即可实现单步生成。首次将 1-NFE 生成能力扩展到 20B 参数的 Qwen-Image 模型，1-NFE GenEval 0.86 逼近原始 100-NFE 的 0.87，推理成本降低 100×。

**[Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](image_generation/uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)**

:   Uni-X提出一种两端分离、中间共享的X型架构来缓解统一多模态模型（UMM）中视觉与文本模态的梯度冲突，通过将浅层和深层设为模态专属、中间层共享参数，3B参数即可匹配或超越7B AR-UMM在图像生成和多模态理解上的性能。

**[Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](image_generation/unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)**

:   DualFlow提出首个统一框架，通过Rectified Flow和检索增强生成（RAG）实现文本+音乐多模态条件下的双人交互/反应式3D运动生成，引入对比流匹配和同步损失，在MDD数据集上FID提升2.5%、R-precision提升76%，推理速度提升2.5倍。

**[Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)**

:   提出无监督共形推断框架（BB-UCP），通过Gram矩阵交互能量评分、批次自举校准和共形对齐，在无标签、API兼容条件下实现LLM生成的分布无关有限样本覆盖率保证，有效检测和过滤幻觉输出。

**[Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](image_generation/verification_of_the_implicit_world_model_in_a_generative_model_via_adversarial_s.md)**

:   提出对抗序列生成方法验证生成式序列模型的隐式世界模型健全性，在国际象棋领域通过多种对抗策略（IMO/BSO/AD）系统评估，发现所有模型均不健全，但训练方法和数据集选择对健全性有显著影响，且线性棋盘状态探针在大多数模型中无因果作用。

**[Verifier-Constrained Flow Expansion for Discovery Beyond the Data](image_generation/verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)**

:   提出Flow Expander (FE)，通过验证器约束的熵最大化在概率空间中扩展预训练流模型的覆盖范围，使其生成超越训练数据分布但保持有效性的设计样本，在分子构象设计中增加多样性同时保持化学有效性。

**[VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](image_generation/vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)**

:   VFScale提出无需外部验证器的测试时可缩放扩散模型，通过MRNCL损失和KL正则化改善能量景观使其内在能量函数可作为验证器，结合混合MCTS去噪实现高效搜索，在6×6训练的迷宫模型能解决88%的15×15迷宫，而标准扩散模型完全失败。

**[Visual Autoregressive Modeling for Instruction-Guided Image Editing](image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)**

:   提出VAREdit，将指令引导的图像编辑重新定义为多尺度预测问题，通过Scale-Aligned Reference模块解决最细尺度条件化的尺度失配问题，在编辑遵循度和效率上大幅超越扩散模型方法。

**[When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](image_generation/when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)**

:   首次揭示并系统研究多模态扩散模型中的"后门模态坍缩"现象——多模态后门攻击中后门效果退化为仅依赖单一模态（通常是文本），提出TMA和CTI两个基于Shapley值的新指标量化模态贡献和跨模态交互，发现"赢者通吃"动态和负交互。

**[When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](image_generation/when_scores_learn_geometry_rate_separations_under_the_manifold_hypothesis.md)**

:   在流形假设下揭示score学习中几何信息与分布信息的尺度分离现象——流形几何信息强度为 $\Theta(\sigma^{-2})$，比分布信息强 $O(\sigma^{-2})$ 倍，由此证明扩散模型的成功主要来自学习数据流形而非完整分布，并提出一行代码修改即可生成流形上的均匀分布。

**[Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](image_generation/zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)**

:   Zatom-1是首个端到端全开源的基础模型，通过多模态流匹配(multimodal flow matching)统一了3D分子和材料的生成建模与属性预测，使用标准Transformer架构在欧几里得空间直接建模离散原子类型和连续3D几何，实现了跨化学域的正迁移学习。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Unifying View of Coverage in Linear Off-Policy Evaluation](reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)**

:   提出了一种新的覆盖性参数——**特征-动态覆盖**（feature-dynamics coverage），通过工具变量视角对经典算法 LSTDQ 进行新颖的有限样本分析，统一了线性离策略评估中各种不同覆盖性定义，解决了该领域长期存在的碎片化问题。

**[AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](reinforcement_learning/abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)**

:   提出 AbstRaL，通过强化学习教 LLM 学习推理问题的数学抽象（将具体数字/名称替换为符号变量、提取通用公式），然后用符号求解器推导答案，在 GSM 扰动 benchmark 上几乎完全消除了分布偏移导致的性能下降，并在 OOD 数学/通用推理任务上也有隐式提升。

**[APPLE: Toward General Active Perception via Reinforcement Learning](reinforcement_learning/apple_toward_general_active_perception_via_reinforcement_learning.md)**

:   提出APPLE——一种结合强化学习与监督学习的通用主动感知框架，将主动感知建模为POMDP，奖励函数设计为RL奖励减去预测损失，梯度自然分解为策略梯度和预测损失梯度两部分，基于off-policy算法（SAC/CrossQ）和共享ViViT骨干网络，在5个不同任务基准上验证通用性，其中CrossQ变体无需逐任务调参且训练效率提高53%。

**[ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](reinforcement_learning/arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)**

:   提出ARM-FM框架，利用基础模型（GPT-4o等）从自然语言任务描述自动生成语言对齐奖励机器（LARM）——包括自动机结构、可执行标签函数和每个状态的自然语言描述——为RL agent提供组合式密集奖励信号，在MiniGrid/Craftium(3D Minecraft)/Meta-World等环境中解决标准RL完全无法学习的稀疏奖励长程任务，并实现零样本任务泛化。

**[AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](reinforcement_learning/autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)**

:   提出 AutoQD，通过随机傅里叶特征（RFF）将策略的占据度量嵌入有限维空间，再用加权 PCA 降维得到行为描述符，实现无需人工设计 BD 的 QD 优化，在 6 个连续控制任务上全面超越手工 BD 和现有无监督 QD 方法。

**[AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](reinforcement_learning/autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)**

:   提出解耦自适应熵约束 (Decoupled Adaptive Entropy Constraints) 的强化学习策略，使 LLM 在工具调用任务中根据问题难度自动切换长/短推理模式，在提升 9.8% 准确率的同时减少约 81% 的推理 token 开销。

**[AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](reinforcement_learning/awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)**

:   提出 AWM，一种无需训练的 LLM 权重矩阵指纹方法，利用线性分配问题（LAP）恢复嵌入层的置换和符号翻转，再用无偏 CKA 消除 Q/K 矩阵的正交变换影响，在 150 对 LLM 上实现完美 AUC（1.0），对 SFT、持续预训练（5.5T token）、RL、多模态扩展、剪枝、upcycling 六类后训练均鲁棒，30 秒内完成。

**[BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](reinforcement_learning/bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)**

:   首次将贝叶斯自适应 MDP（BAMDP）引入离线模型基 RL，提出 Continuous BAMCP 解决连续状态/动作空间的贝叶斯规划，结合悲观奖励惩罚和搜索基策略迭代（"RL + Search"范式），在 D4RL 12 个任务上显著超越 19 个基线（Cohen's $d > 1.8$），并成功应用于核聚变 tokamak 控制。

**[Boolean Satisfiability via Imitation Learning](reinforcement_learning/boolean_satisfiability_via_imitation_learning.md)**

:   提出 ImitSAT，首个基于模仿学习的 CDCL 求解器分支策略：通过将求解器运行压缩为无冲突的 KeyTrace 专家序列，将分支决策建模为前缀条件的自回归预测任务，以少量查询预算显著减少传播次数和求解时间，并在结构化 SAT 问题上展现良好泛化能力。

**[Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)**

:   通过观察性研究（18 个开源 RPT 模型）和干预性研究（单域 GRPO 训练），系统揭示了强化后训练（RPT/RLVR）的泛化局限：RPT 在训练域内提升显著，但跨域泛化不一致——结构化域（数学↔代码）可互相迁移，但无法泛化到非结构化域（法律/金融/医疗），且这一结论跨算法、模型规模和训练步数保持一致。

**[Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs](reinforcement_learning/chain-of-context_learning_dynamic_constraint_understanding_for_multi-task_vrps.md)**

:   提出 Chain-of-Context Learning (CCL)，通过 Relevance-Guided Context Reformulation（RGCR，自适应聚合约束信息构建上下文）和 Trajectory-Shared Node Re-embedding（TSNR，跨轨迹共享节点更新避免冗余计算）实现逐步动态的约束感知解码，在 48 种 VRP 变体（16 分布内 + 32 分布外）上全面超越现有方法。

**[Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](reinforcement_learning/co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)**

:   Co-rewarding 提出自监督 RL 框架，通过数据侧（对比改写问题的跨视角一致性）和模型侧（EMA 教师模型提供伪标签）两种互补监督方式，解决自奖励 RL 中的训练崩溃问题，在无人工标签条件下多项数学推理基准上达到甚至超过 RLVR（有标签）的性能。

**[Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)**

:   提出 VIP（Value Iteration via PINN）框架，首次将物理信息神经网络（PINN）用于求解连续时间多智能体强化学习中的 HJB 偏微分方程，并引入 Value Gradient Iteration（VGI）模块迭代精炼价值梯度，在连续时间 MPE 和 MuJoCo 多智能体任务上始终优于离散时间和连续时间基线。

**[Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)**

:   CalibRL 将专家数据重新定义为分布校准基线（而非严格模仿目标），通过 LeakyReLU 不对称激活 + 优势加权实现对 MLLM 推理训练中探索-利用平衡的精细控制，解决 RLVR 中的熵崩溃问题，在几何推理等任务上大幅超越 GRPO/DAPO。

**[Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](reinforcement_learning/cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)**

:   系统研究跨形态离线 RL 预训练范式，发现次优数据比例和机器人多样性增加时梯度冲突导致负迁移，提出基于形态图距离的 Embodiment Grouping（EG）策略将机器人按形态聚类后分组更新 actor，在 16 种机器人平台的 locomotion benchmark 上显著缓解负迁移（70% 次优数据集上 IQL+EG 比 IQL 提升 34%）。

**[CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](reinforcement_learning/cuda-l1_improving_cuda_optimization_via_contrastive_reinforcement_learning.md)**

:   提出 CUDA-L1，一个基于对比强化学习（Contrastive RL）的三阶段流水线框架，将初始 CUDA 能力较弱的 LLM 训练为高效的 CUDA 优化器，在 KernelBench 的 250 个 CUDA 内核上实现平均 3.12× 加速，峰值达 120×，并可跨 GPU 架构迁移。

**[Deep SPI: Safe Policy Improvement via World Models](reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)**

:   构建了安全策略改进（SPI）的理论框架，将世界模型和表示学习与策略更新保证统一起来：通过基于重要性比率的邻域算子约束策略更新，确保单调改进和收敛；结合局部转移/奖励损失控制世界模型质量和表示稳定性，提出 DeepSPI 算法在 ALE-57 基准上匹配或超越 PPO 和 DeepMDP。

**[Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](reinforcement_learning/distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)**

:   提出 Distributionally Robust IGM (DrIGM) 原则，将分布鲁棒优化引入协作多智能体 RL 的值分解框架，使得 VDN/QMIX/QTRAN 等经典方法能够在训练环境与部署环境存在分布偏移时仍保持稳健的去中心化执行性能。

**[DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](reinforcement_learning/dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)**

:   提出 DiVE-k 框架，利用大视觉语言模型自身的 top-k 生成结果构造选择题，通过 GRPO 强化学习训练模型进行差异化视觉推理，在细粒度图像识别的 base-to-novel 泛化上大幅超越现有方法。

**[Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](reinforcement_learning/divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)**

:   提出 Pram 框架，首次利用多模态语言模型（MLM）求解多商品流（MCF）问题，通过分区将原问题分解为子问题，以多智能体强化学习（MARL）协调各子问题的全局一致性，理论证明收敛到最优解，实测速度比 LP 快 1-2 个数量级且性能接近最优。

**[Don't Just Fine-tune the Agent, Tune the Environment](reinforcement_learning/dont_just_fine-tune_the_agent_tune_the_environment.md)**

:   提出 Environment Tuning 训练范式，通过结构化课程、可操作的环境增强反馈和细粒度进度奖励，使 LLM agent 仅用 400 个训练样本即可从零学会复杂的多轮工具使用，同时实现优异的分布外泛化能力。

**[Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)**

:   首次在跨域离线 RL 中同时处理训练时鲁棒性（源域-目标域动力学不匹配）和测试时鲁棒性（部署环境动力学偏移）：提出 DROCO 算法，核心是 Robust Cross-Domain Bellman (RCB) 算子——对源域数据施加鲁棒 Bellman 更新、对目标域数据施加标准 in-sample 更新，并通过对偶重构将不可处理的动力学不确定性映射为状态空间扰动，在 D4RL 基准上总分 1105.2 超越次优方法 14%，且在 hard 级别动力学扰动下性能退化仅为基线的一半。

**[Dual Goal Representations](reinforcement_learning/dual_goal_representations.md)**

:   提出"对偶目标表征"（dual goal representation），用"所有状态到目标状态的时间距离集合"来编码目标，理论证明该表征对最优策略恢复是充分的且天然过滤外生噪声，并设计基于非对称内积参数化的实用学习算法，在 OGBench 20 个任务上作为可插拔模块一致提升三种主流离线 GCRL 方法的性能。

**[DVLA-RL: Dual-Level Vision-Language Alignment with Reinforcement Learning Gating for Few-Shot Learning](reinforcement_learning/dvla-rl_dual-level_vision-language_alignment_with_reinforcement_learning_gating_.md)**

:   提出 DVLA-RL 框架，通过双层语义构建（DSC）生成互补的低层属性和高层描述，并以 RL 门控注意力（RLA）动态平衡自注意力和交叉注意力在不同网络层的贡献，实现从低层到高层的层次化视觉-语言对齐，在 9 个少样本学习基准上达到 SOTA。

**[Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning](reinforcement_learning/echo_towards_advanced_audio_comprehension_via_audio-interleaved_reasoning.md)**

:   提出音频交错推理（audio-interleaved reasoning）新范式，将音频视为推理过程中的主动组件而非静态上下文，使 LALM 在推理时动态定位并重新聆听音频片段。通过 SFT+RL 两阶段训练框架和结构化数据生成流水线，构建 Echo 模型，在专家级和通用音频理解基准上超越 GPT-4o 和 Gemini-2.0-Flash。

**[Efficient Estimation of Kernel Surrogate Models for Task Attribution](reinforcement_learning/efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)**

:   提出核代理模型（KernelSM）用于任务归因，通过 RBF 核岭回归捕获任务间的非线性交互效应，结合梯度投影的高效估计算法避免重复训练，在数学推理、上下文学习和多目标 RL 等场景下相比线性代理和影响函数基线提升 25% 相关性。

**[EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph](reinforcement_learning/egg-sr_embedding_symbolic_equivalence_into_symbolic_regression_via_equality_grap.md)**

:   提出 Egg-SR 统一框架，通过等价图（e-graph）将符号等价性嵌入 MCTS、DRL 和 LLM 三类符号回归方法中，分别实现子树剪枝、梯度方差降低和反馈提示增强。理论证明 Egg-MCTS 收紧遗憾界、Egg-DRL 降低梯度估计方差，实验验证一致提升表达式发现精度。

**[Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator](reinforcement_learning/emergence_of_spatial_representation_in_an_actor-critic_agent_with_hippocampus-in.md)**

:   受海马体 CA3 区内在递归回路启发，提出最小序列生成器（shift register）与 actor-critic 结合，在稀疏视觉输入下实现迷宫导航，同时涌现出位置场、DG 正交化、距离相关空间核和任务依赖重映射等神经生物学现象。

**[Empowering Small VLMs to Think with Dynamic Memorization and Exploration](reinforcement_learning/empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)**

:   提出 DyME（Dynamic Memorize-Explore），通过逐步动态切换 SFT 记忆模式与 GRPO 探索模式，首次赋予小规模视觉语言模型（<1B 参数）在特定任务上的思维推理能力。

**[Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](reinforcement_learning/entropy-preserving_reinforcement_learning.md)**

:   本文揭示了策略梯度 RL 算法在 LLM 后训练中系统性导致策略熵坍缩的理论根因（优势函数与对数概率的正相关性），并提出两种互补的解法：REPO（通过修改优势函数去相关）和 ADAPO（自适应非对称裁剪），在交互式工具使用任务上实现 SOTA 性能。

**[Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)**

:   通过理论推导和跨模型实验，证明 RLVR 中裁剪偏差提供的学习信号可忽略不计（≤1/17），真正起作用的是裁剪对策略熵的隐式压缩效应，并提出奖励误标模型解释为何随机奖励能让强模型获益。

**[FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](reinforcement_learning/fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)**

:   针对 RLVR 训练中"答案正确但推理有缺陷"的 flawed-positive rollout 问题，提出 FAPO 算法：用 GenRM 检测缺陷推理，通过无参数奖励惩罚机制实现"先利用后抑制"的自然学习轨迹，同时提升结果正确性、过程可靠性和训练稳定性。

**[Flow Actor-Critic for Offline Reinforcement Learning (FAC)](reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)**

:   FAC 首次联合利用流模型（continuous normalizing flow）同时构建表达力强的 actor 策略和基于精确密度估计的 critic 惩罚机制，通过识别 OOD 区域对 Q 值进行选择性保守估计，在 OGBench 55 个任务上以 60.3 平均分大幅超越此前最佳的 43.6。

**[From Observations to Events: Event-Aware World Model for Reinforcement Learning](reinforcement_learning/from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)**

:   提出 Event-Aware World Model (EAWM)，一个通用框架，通过从原始观测中自动生成事件并学习事件感知表征，在不需要手工标签的情况下，将现有 MBRL 基线性能提升 10%–45%，在 Atari 100K、Craftax 1M、DeepMind Control 500K、DMC-GB2 500K 上均创新 SOTA。

**[From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)**

:   提出 RLVRR 框架，将 RLVR（强化学习+可验证奖励）从数学/代码推理扩展到开放式文本生成：从高质量参考答案中提取关键词序列（内容奖励）和可执行 Python 检查函数（风格奖励），构成"奖励链"替代单点验证信号，在 10+ 个 benchmark 上以 10K 数据超越 100K SFT 和高级奖励模型。

**[GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](reinforcement_learning/graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)**

:   提出 GraphOmni 基准框架，在 7 种图类型 × 7 种序列化格式 × 9 种提示策略的 241K 查询上系统评估 11 个 LLM 的图论推理能力，揭示三维度间的复杂交互效应，并设计 RL 引导的组合搜索方法以 25% 成本保持约 90% 最优准确率。

**[Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](reinforcement_learning/helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)**

:   提出 HELIX 框架，将强化学习（GRPO）与进化算法（NSGA-II）结合用于开放式科学问题求解：RL 迭代优化策略，进化机制平衡解的质量与多样性，in-context learning 利用历史解指导探索，仅用 14B 模型在圆填充、机器学习任务等 20 个任务中超越 GPT-4o 流水线。

**[How Far Can Unsupervised RLVR Scale LLM Training?](reinforcement_learning/how_far_can_unsupervised_rlvr_scale_llm_training.md)**

:   对无监督可验证奖励强化学习（URLVR）进行全面分析，揭示所有内在奖励方法本质上都是在"锐化"模型初始分布，导致先升后降的不可避免崩溃模式；提出Model Collapse Step作为模型先验指标，并指出外部奖励方法是突破可扩展性瓶颈的方向。

**[How LLMs Learn to Reason: A Complex Network Perspective](reinforcement_learning/how_llms_learn_to_reason_a_complex_network_perspective.md)**

:   从复杂网络视角提出**稀疏概念网假说**（平均度 $\langle k\rangle\approx 2$），统一解释 RLVR 训练的四大谜题（两阶段学习曲线、V 型回复长度、灾难性遗忘、策略坍缩），并据此设计 Annealed-RLVR 算法在 Minerva 和 AIME 上超越标准 RLVR。

**[Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](reinforcement_learning/is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)**

:   证明在外生MDP（Exo-MDP，不确定性仅来自独立于智能体动作的外生输入）中，纯利用（无探索）策略即可达到次线性遗憾界——表格情形下PTO算法达到 $\tilde{O}(H^2|\Xi|\sqrt{K})$，线性函数逼近下LSVI-PE算法遗憾与特征维度和外生状态空间多项式相关、与内生状态/动作空间无关。

**[LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection](reinforcement_learning/laddersym_a_multimodal_interleaved_transformer_for_music_practice_error_detectio.md)**

:   提出LadderSym架构解决音乐练习错误检测任务，通过交替式跨流对齐模块（Ladder）克服晚期融合的对齐不足，并用符号乐谱提示（Sym）减少纯音频乐谱的频率歧义，在MAESTRO-E上将漏音F1从26.8%提升到56.3%。

**[Latent Wasserstein Adversarial Imitation Learning](reinforcement_learning/latent_wasserstein_adversarial_imitation_learning.md)**

:   提出LWAIL方法，用ICVF从少量随机数据学习动态感知的潜空间表示，将Wasserstein距离的"地面度量"从欧氏距离升级为潜空间距离，仅用单条状态轨迹即可达到专家级模仿性能。

**[Learning from Synthetic Data Improves Multi-hop Reasoning](reinforcement_learning/learning_from_synthetic_data_improves_multi-hop_reasoning.md)**

:   发现在完全虚构的规则生成合成数据上做RLVR训练，能显著提升LLM在真实多跳推理任务上的表现（Qwen3-0.6B提升56%-131%），因为模型学到了知识组合这一通用推理技能而非记忆事实知识。

**[Learning to Generate Unit Test via Adversarial Reinforcement Learning](reinforcement_learning/learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)**

:   提出UTRL框架，通过对抗RL迭代训练单元测试生成器和代码生成器——测试生成器学习生成能区分LLM代码与正确代码的判别性测试用例，代码生成器学习通过这些测试——Qwen3-4B训练后超越GPT-4.1的测试生成质量。

**[Learning to Orchestrate Agents in Natural Language with the Conductor](reinforcement_learning/learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)**

:   用GRPO训练一个7B Qwen2.5模型作为"Conductor"，通过自然语言输出完整的Agent工作流（子任务指令+worker分配+通信拓扑访问列表），协调GPT-5/Claude Sonnet 4/Gemini 2.5 Pro等frontier模型，仅用960题×200迭代训练，在7个推理benchmark上平均77.27%超越所有单模型（GPT-5为74.78%）和多Agent基线。

**[Learning to Play Multi-Follower Bayesian Stackelberg Games](reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)**

:   首次系统研究多追随者贝叶斯Stackelberg博弈（BSG）的在线学习问题，通过对领导者策略空间进行"最佳响应区域"几何分割，在类型反馈下实现 $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ 的遗憾界——该界不随追随者数 $n$ 呈多项式增长，并证明了几乎匹配的 $\Omega(\sqrt{\min\{L, nK\}T})$ 下界。

**[Less is More: Clustered Cross-Covariance Control for Offline RL](reinforcement_learning/less_is_more_clustered_cross-covariance_control_for_offline_rl.md)**

:   本文揭示了离线RL中标准平方误差目标会引入有害的TD交叉协方差，并提出C⁴（Clustered Cross-Covariance Control for TD）方法，通过分区缓冲区采样和显式梯度校正惩罚来抑制这一效应，在小数据集和OOD区域主导的场景下实现高达30%的回报提升。

**[LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)**

:   提出 LongRLVR，通过在 RLVR 训练中引入可验证的上下文奖励（context reward），解决长上下文场景下仅靠最终答案奖励导致的上下文定位（grounding）梯度消失问题，显著提升 LLM 长上下文推理能力。

**[LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](reinforcement_learning/longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)**

:   提出 LongWriter-Zero：从基础模型出发，不依赖任何标注或合成数据，仅通过 GRPO 强化学习 + 三维度复合奖励模型（长度 / 质量 / 格式），涌现出超长高质量文本生成能力，在 WritingBench 上以 32B 参数量超越 DeepSeek-R1 和 Qwen3-235B 等 100B+ 模型。

**[LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)**

:   提出 LoongRL，通过构建 KeyChain 合成数据进行强化学习训练，使 LLM 涌现出 plan–retrieve–reason–recheck 的长上下文推理模式，仅在 16K 上下文上训练即可泛化到 128K，14B 模型达到 74.2 分接近 o3-mini (74.5) 和 DeepSeek-R1 (74.9)。

**[MARS-Sep: Multimodal-Aligned Reinforced Sound Separation](reinforcement_learning/mars-sep_multimodal-aligned_reinforced_sound_separation.md)**

:   MARS-Sep 将查询条件声音分离重新建模为强化学习问题，通过分解 Beta 掩码策略在时频域上进行随机决策，并利用渐进式对齐的多模态编码器提供语义奖励信号，在信号保真度和语义一致性上同时取得提升。

**[Menlo: From Preferences to Proficiency – Evaluating and Modeling Native-like Quality Across 47 Languages](reinforcement_learning/menlo_from_preferences_to_proficiency_--_evaluating_and_modeling_native-like_qua.md)**

:   提出 Menlo 框架，基于受众设计理论将母语级响应质量分解为四个维度，构建了覆盖 47 种语言变体的 6423 条标注偏好对数据集，并发现配对评估+RL 训练的 LLM 裁判可达到接近人类标注员的水平。

**[MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](reinforcement_learning/mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)**

:   MergeMix 提出了一种基于 token merging 的 mixup 数据增强方法，通过双部软匹配在注意力空间生成混合图像，并将混合比例作为偏好优化中的软边际，在图像分类和多模态大模型两个场景下统一了 SFT 和 RL 训练范式。

**[Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](reinforcement_learning/metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)**

:   提出 SPECS 三阶段冷启动框架——先通过自蒸馏生成偏好数据（仅区分格式差异），再用 DPO 做格式预对齐作为冷启动，最后接 GRPO 微调——解耦了格式学习和推理学习，实现 MEGA-Bench +4.1%、MathVista +12.2% 的一致性能提升。

**[ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](reinforcement_learning/model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)**

:   ROMI 通过 Wasserstein 对偶将动力学不确定集转化为状态不确定集来实现鲁棒的价值感知模型学习，并用隐式可微的自适应加权机制平衡动力学精度与价值感知，解决了 RAMBO 方法中的 Q 值低估和梯度爆炸问题，在 D4RL 和 NeoRL 上达到模型基离线 RL 的 SOTA。

**[Model Predictive Adversarial Imitation Learning for Planning from Observation](reinforcement_learning/model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)**

:   提出 MPAIL（Model Predictive Adversarial Imitation Learning），将 MPPI 规划器嵌入对抗模仿学习循环，首次实现端到端的仅观测规划框架（Planning-from-Observation），在泛化性、鲁棒性、可解释性和样本效率上全面优于基于策略的 AIL 方法，并在真实世界机器人导航中从单条观测演示成功部署。

**[MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation](reinforcement_learning/momagen_generating_demonstrations_under_soft_and_hard_constraints_for_multi-step.md)**

:   MoMaGen 将双臂移动操作的演示数据生成建模为约束优化问题，通过硬约束（可达性、无碰撞、可见性）和软约束（导航中物体可见性、收回紧凑姿态）的协同，从单个人类遥操作演示自动生成大规模多样化数据集，训练出的视觉运动策略仅用 40 个真实演示微调即可部署到实体机器人。

**[MVR: Multi-view Video Reward Shaping for Reinforcement Learning](reinforcement_learning/mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)**

:   提出 MVR 框架，利用多视角视频的视频-文本相似度学习状态相关性函数，结合状态依赖的奖励塑形（自动衰减 VLM 引导），在 HumanoidBench 和 MetaWorld 共 19 个任务上超越现有 VLM 奖励方法。

**[Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](reinforcement_learning/near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)**

:   提出 MB-AIL（基于模型的对抗模仿学习）算法，在一般函数逼近下建立了无视域（horizon-free）的二阶样本复杂度上界，结合新构建的困难实例上的信息论下界，证明 MB-AIL 在在线交互的样本复杂度上达到极小极大最优（相差对数因子）。

**[Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)**

:   本文通过将Stackelberg博弈中的领导者效用空间线性化，提出了一种约简到线性上下文赌臂问题的算法，在带侧信息的赌臂反馈设置下将遗憾界从 $\tilde{O}(T^{2/3})$ 改进到近似最优的 $\tilde{O}(T^{1/2})$。

**[Offline Reinforcement Learning with Generative Trajectory Policies](reinforcement_learning/offline_reinforcement_learning_with_generative_trajectory_policies.md)**

:   提出 Generative Trajectory Policy (GTP)，通过学习 ODE 完整解映射统一扩散、流匹配和一致性模型，配合分数近似和值驱动引导两项关键适配技术，在 D4RL 上达到 SOTA。

**[On Discovering Algorithms for Adversarial Imitation Learning](reinforcement_learning/on_discovering_algorithms_for_adversarial_imitation_learning.md)**

:   提出 DAIL——首个元学习对抗性模仿学习算法：将 AIL 分解为密度比估计和奖励赋值(RA)两阶段，用 LLM 引导的进化搜索自动发现最优 RA 函数 $r_{\text{disc}}$，在未见环境和策略优化器上泛化并超越所有人工设计基线。

**[On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](reinforcement_learning/on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)**

:   从RL策略梯度视角数学证明SFT梯度隐式编码了逆概率加权(1/π_θ)的病态奖励结构→低概率token梯度过大导致泛化受限，提出DFT(Dynamic Fine-Tuning)仅需一行代码修改(CE loss乘token概率：$-p\log p$)消除逆概率加权→在数学推理/代码生成/多模态任务上大幅超越SFT，离线RL设定下甚至超越GRPO/PPO。

**[On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games](reinforcement_learning/on_the_o1t_convergence_of_alternating_gradient_descent-ascent_in_bilinear_games.md)**

:   首次证明交替梯度下降上升（AltGDA）在有约束双线性零和博弈中以 $O(1/T)$ 速率收敛到Nash均衡（存在内部NE时），比同步GDA的 $O(1/\sqrt{T})$ 快，用能量函数衰减刻画轨迹碰撞边界时的"摩擦"效应，并通过性能估计编程（PEP）进一步优化步长。

**[One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)**

:   提出 ScaleZero，通过在统一世界模型中引入 MoE 架构解决多任务学习中的梯度冲突和可塑性崩塌问题，结合动态参数扩展（DPS）策略自适应分配模型容量，单个多任务模型在 Atari/DMC/Jericho 三个基准上达到与单任务专家模型相当的性能，同时减少约 28.5% 的环境交互。

**[Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](reinforcement_learning/online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)**

:   将Friedkin-Johnsen观点动力学模型下极化+分歧最小化问题首次形式化为在线低秩矩阵bandit问题（OPD-Min），提出两阶段算法OPD-Min-ESTR通过子空间估计将维度从 $|V|^2$ 降至 $O(|V|)$，在合成和真实网络上显著优于全维度线性bandit基线。

**[Online Prediction of Stochastic Sequences with High Probability Regret Bounds](reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)**

:   重新审视有限时间范围 $T$ 下随机序列的通用预测经典问题，首次给出以高概率成立的消退遗憾界（形式为 $O(T^{-1/2}\delta^{-1/2})$），与已有的期望遗憾界 $O(T^{-1/2})$ 形式高度一致，并证明在不附加额外假设时 $\delta$ 的指数无法改进。

**[Optimistic Task Inference for Behavior Foundation Models](reinforcement_learning/optimistic_task_inference_behavior_models.md)**

:   提出 OpTI-BFM——在 Behavior Foundation Model 测试时，不需要完整奖励函数或标注数据集，而是通过与环境交互仅 5 个 episode 即可推断任务并恢复 Oracle 性能，核心是利用 successor features 的线性结构将任务推断归约为线性 bandit 问题并用 UCB 策略乐观探索，提供正式的 regret bound。

**[ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction](reinforcement_learning/paras2s_benchmarking_and_aligning_spoken_language_models_for_paralinguistic-awar.md)**

:   提出 ParaS2S 框架——包含一个评估副语言感知（emotion/sarcasm/age/gender）的语音到语音基准 ParaS2SBench，以及一个基于 GRPO 的 RL 对齐框架 ParaS2SAlign，使 S2S 模型能够在极少标注数据下习得根据说话风格调整回复的能力。

**[Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](reinforcement_learning/partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)**

:   提出部分群不变MDP (PI-MDP) 框架，通过可学习的门控函数 $\lambda(s,a)$ 在状态-动作空间中逐点切换等变与标准Bellman更新，从理论上证明局部对称性破缺会经过折扣回溯放大 $1/(1-\gamma)$ 倍产生全局值函数误差，而PI-MDP可将误差严格限制在破缺区域内；实例化为PE-DQN和PE-SAC两种算法，在Grid-World、MuJoCo运动、机械臂操作等任务上全面超越严格等变和近似等变基线。

**[PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](reinforcement_learning/policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)**

:   提出PolicyFlow，将连续归一化流(CNF)策略无缝嵌入PPO框架：通过沿插值路径的速度场变化近似重要性比率（避免全流路径反向传播），并引入受布朗运动启发的隐式熵正则器防止模式坍缩，在MultiGoal/PointMaze/IsaacLab/MuJoCo等环境中达到或超越高斯PPO和流式基线(FPO/DPPO)的性能。

**[Post-training Large Language Models for Diverse High-Quality Responses](reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)**

:   提出 DQO（Diversity Quality Optimization），基于行列式点过程（DPP）在语义嵌入空间中定义多样性度量，将其与奖励信号联合优化，使 LLM 后训练同时提升语义多样性和响应质量，可叠加在 GRPO/PPO 之上。

**[PreferThinker: Reasoning-based Personalized Image Preference Assessment](reinforcement_learning/preferthinker_reasoning-based_personalized_image_preference_assessment.md)**

:   提出 PreferThinker，通过引入通用视觉偏好画像（preference profile）连接不同用户，采用 predict-then-assess 的 CoT 推理范式进行可解释的个性化图像偏好评估，结合冷启动 SFT + GRPO 强化学习及 similarity-aware 预测奖励，7B 模型超越 GPT-4o（+5.2%）和 Claude 3.7（+5.1%）。

**[Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)**

:   受人脑海马体-大脑皮层交互机制启发，提出 FAME 双学习器框架，通过快速学习器进行知识迁移、元学习器进行知识整合，在原则性地最小化灾难性遗忘的前提下实现高效的持续强化学习。

**[Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](reinforcement_learning/pruning_as_a_cooperative_game_surrogate-assisted_layer_contribution_estimation_f.md)**

:   将LLM层剪枝建模为合作博弈，利用轻量代理网络近似Shapley值来捕获层间依赖关系，实现比静态启发式方法更优的深度剪枝效果。

**[QuRL: Efficient Reinforcement Learning with Quantized Rollout](reinforcement_learning/qurl_efficient_reinforcement_learning_with_quantized_rollout.md)**

:   提出 QuRL 方法，通过量化 actor 模型加速 RL 训练中的 rollout 阶段，设计自适应裁剪范围（ACR）解决量化导致的训练崩溃问题，并提出更新感知量化（UAQ）解决权重更新远小于量化误差的尺度失配问题，实现 20%~80% 的推理吞吐量提升且不损失性能。

**[Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)**

:   通过系统实验揭示了 RL 训练的推理型 IQA 模型泛化能力的本质机制——推理过程本质上是将冗余的视觉表示转换为紧凑的跨域对齐文本表示——并基于此提出 RALI 算法，通过对比学习直接对齐图像与这些文本表示，以不到 5% 的参数和推理时间达到了可比的泛化性能。

**[Reasoning Boosts Opinion Alignment in LLMs](reinforcement_learning/reasoning_boosts_opinion_alignment_in_llms.md)**

:   用 GRPO 强化学习训练 LLM 通过结构化推理对齐个体政治观点，SFT+GRPO 在美国/德国/瑞士三国数据集上一致优于 ICL 和 ORPO 基线，但系统性揭示了左右翼偏差和 Neutral 立场预测的根本困难。

**[RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind](reinforcement_learning/rebuttalagent_strategic_persuasion_in_academic_rebuttal_via_theory_of_mind.md)**

:   首次将心智理论（ToM）引入学术 rebuttal，提出 ToM-Strategy-Response 三阶段框架：先建模审稿人心理状态，再制定说服策略，最后生成证据支撑的回复，结合自奖励 RL 训练和专用 Rebuttal-RM 评估器，平均指标超越基座模型 18.3%。

**[References Improve LLM Alignment in Non-Verifiable Domains](reinforcement_learning/references_improve_llm_alignment_in_non-verifiable_domains.md)**

:   提出参考引导的LLM-as-Judge方法(RefEval)，用高质量参考输出作为"软验证器"，使LLM-judge准确率提升6.8%；进而构建两阶段自改进流程(SFT蒸馏+参考引导DPO)，在AlpacaEval/Arena-Hard上分别超过SFT蒸馏+19.2/+16.5，匹配微调奖励模型ArmoRM的性能，证明无需人类偏好标注即可实现非可验证域的高效LLM对齐。

**[ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reinforcement_learning/reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)**

:   提出ReFORM方法，通过学习一个反射流噪声生成器来操纵行为克隆流策略的源分布，以**构造性方式**实现支撑约束，避免OOD问题的同时保持策略表达力，无需超参数调节。

**[Regret-Guided Search Control for Efficient Learning in AlphaZero](reinforcement_learning/regret-guided_search_control_for_efficient_learning_in_alphazero.md)**

:   提出 RGSC（Regret-Guided Search Control）框架，通过训练一个 regret 网络识别高遗憾值状态并优先从这些状态重新开始自我对弈，模拟人类"反复复盘错误"的学习方式，在 9×9 围棋、10×10 黑白棋和 11×11 Hex 上平均超越 AlphaZero 77 Elo。

**[Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](reinforcement_learning/remix_reinforcement_routing_for_mixtures_of_loras_in_llm_finetuning.md)**

:   将LLM层剪枝建模为合作博弈（每层=玩家，模型性能=效用）→精确Shapley值计算不可行（$2^L$种组合）→提出两阶段近似：(1)分层蒙特卡洛采样生成mask+评估PPL作为监督信号→(2)训练轻量代理网络预测任意mask的性能→高效估算每层Shapley值→捕获层间依赖→显著优于静态启发式剪枝基线。

**[Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent RL](reinforcement_learning/retaining_suboptimal_actions_to_follow_shifting_optima_in_multi-agent_reinforcem.md)**

:   提出 S2Q（Successive Sub-value Q-learning），通过逐步学习 $K$ 个 sub-value 函数显式保留次优联合动作，结合 Softmax 行为策略在候选间优先级采样，解决合作 MARL 中值分解方法因最优点动态漂移而收敛到次优策略的根本问题。

**[Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](reinforcement_learning/rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)**

:   从理论上分析了集成策略梯度方法中策略间多样性对学习效率的影响，提出通过KL散度约束调控多样性的Coupled Policy Optimization（CPO），在大规模并行环境中实现高效稳定的探索。

**[Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching](reinforcement_learning/revisiting_matrix_sketching_in_linear_bandits_achieving_sublinear_regret_via_dya.md)**

:   本文揭示了现有基于矩阵草图的线性Bandit方法在流数据频谱尾部较重时会退化为线性遗憾的根本缺陷，提出 Dyadic Block Sketching 多尺度草图框架，通过动态加倍草图大小控制全局逼近误差至预设参数 $\epsilon$，使算法在无需预知流矩阵频谱性质的情况下始终保证次线性遗憾，并在频谱友好场景下自适应恢复单尺度方法的计算效率。

**[RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](reinforcement_learning/rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)**

:   提出RewardMap框架，通过难度感知的细节奖励设计和从简单感知到复杂推理的多阶段RL课程学习策略，克服细粒度视觉推理中的稀疏奖励问题。

**[RLP: Reinforcement as a Pretraining Objective](reinforcement_learning/rlp_reinforcement_as_a_pretraining_objective.md)**

:   提出RLP（Reinforcement Learning Pretraining），一种信息增益驱动的RL预训练目标，通过奖励能提升下一token预测概率的思维链（CoT），将RL从后训练阶段前移到预训练阶段，实现无验证器的密集奖励信号。

**[RM-R1: Reward Modeling as Reasoning](reinforcement_learning/rm-r1_reward_modeling_as_reasoning.md)**

:   将奖励建模重新定义为推理任务，提出RM-R1系列推理奖励模型（ReasRM），通过推理蒸馏+RL训练以及Chain-of-Rubrics（CoR）机制，在三大奖励模型基准上平均超越70B和GPT-4o模型达4.9%。

**[Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)**

:   本文研究 RL 中一种新型威胁——行为目标攻击（adversary 通过篡改观测来引导 victim 执行特定目标策略），提出不需要白盒访问的 BIA 攻击方法和基于时间折扣的 TDRT 防御方法，TDRT 在保持对攻击鲁棒性的同时比现有防御（SA-PPO）的原始任务性能高 28.2%。

**[Robust Multi-Objective Controlled Decoding of Large Language Models](reinforcement_learning/robust_multi-objective_controlled_decoding_of_large_language_models.md)**

:   提出RMOD（Robust Multi-Objective Decoding），一种推理时算法，通过求解最小最大博弈的Nash均衡来动态计算最坏情况目标权重，在无需先验权重信息的情况下实现LLM的鲁棒多目标对齐。

**[Routing, Cascades, and User Choice for LLMs](reinforcement_learning/routing_cascades_and_user_choice_for_llms.md)**

:   将LLM路由建模为provider-user Stackelberg博弈，证明最优路由几乎总是静态无级联的阈值规则，揭示质量/成本排序不一致时的用户-提供商不对齐，以及低流失惩罚下provider被激励通过throttling延迟来降低成本但损害用户效用。

**[Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](reinforcement_learning/safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)**

:   提出首个显式处理状态约束的连续时间多智能体RL框架，通过Epigraph形式将不连续的约束值函数转化为连续表示，结合改进的PINN actor-critic方法实现安全、稳定的连续时间多智能体控制。

**[Sample-efficient and Scalable Exploration in Continuous-Time RL](reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)**

:   提出 COMBRL 算法，通过最大化外在奖励与模型认知不确定性的加权和，在连续时间模型基 RL 中实现可扩展且样本高效的探索，并具有次线性后悔理论保证。

**[Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)**

:   提出 Self-Harmony 框架，通过让单一模型扮演两个角色（Solver 求解原始问题 + Reframer 改述问题），将答案在原始和改述视角下的调和平均得分作为伪标签选择标准，替代传统多数投票，在 30 个实验设置中 28 个达到 SOTA，且训练零失败。

**[Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](reinforcement_learning/self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)**

:   提出 SISL（Self-Improving Skill Learning），通过解耦高层策略和技能改进策略，结合最大回报重标注的技能优先级机制，在噪声离线演示数据下实现鲁棒的技能学习，显著提升基于技能的元强化学习在长时域任务中的性能。

**[Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](reinforcement_learning/shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)**

:   提出 Shop-R1 框架，通过分层奖励机制和难度感知缩放的强化学习方法，显著提升 LLM 模拟真实人类在线购物行为的能力，相比 SFT 基线精确匹配提升超过 65%。

**[Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)**

:   提出单指标赌博机（SIB）问题——将广义线性赌博机扩展到奖励函数未知的设定，基于 Stein 方法设计了一族高效算法（STOR/ESTOR/GSTOR），在单调递增奖励函数下实现了近最优遗憾界 $\tilde{O}(\sqrt{T})$。

**[Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](reinforcement_learning/solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)**

:   证明单边信息二人零和微分博弈中 Nash 均衡策略的原子结构——知情玩家 P1 的均衡策略集中在至多 $I$ 个动作原型上（$I$ = 博弈类型数），使博弈树复杂度从 $U^{2K}$ 降到 $I^K$，在美式足球 11v11 连续动作空间中（传统复杂度 $10^{440}$）实现 M1 MacBook 30 分钟求解。

**[Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](reinforcement_learning/solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)**

:   提出 Feasibility-Guided Exploration (FGE)，同时识别可行参数子集并学习在该子集上安全的策略，解决可行性未知的参数鲁棒避障问题，在 MuJoCo 任务中比最佳现有方法多覆盖 50% 以上。

**[Spectral Bellman Method: Unifying Representation and Exploration in RL](reinforcement_learning/spectral_bellman_method_unifying_representation_and_exploration_in_rl.md)**

:   提出 Spectral Bellman Method (SBM)，从零内在 Bellman 误差 (IBE) 条件出发发现 Bellman 算子与特征协方差的谱结构联系，推导出新的表示学习目标，并自然地统一了表示学习和 Thompson Sampling 探索。

**[SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)**

:   提出 SPELL 框架，让一个 LLM 同时扮演出题者、答题者和验证者三个角色进行自我博弈强化学习，无需人类标注即可持续提升长文本推理能力，在 6 个长上下文基准上一致提升性能。

**[SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)**

:   提出 SPIRAL 框架，让 LLM 在多轮零和游戏中进行自我博弈训练，通过角色条件优势估计（RAE）稳定训练，在无领域特定数据的情况下将推理能力提升最高 10%，并发现不同游戏发展出互补的认知能力。

**[Spotlight on Token Perception for Multimodal Reinforcement Learning](reinforcement_learning/spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)**

:   提出 VPPO（Visually-Perceptive Policy Optimization），通过量化每个 token 的视觉依赖度，在轨迹级和 token 级两个层次对学习信号进行精细化调控，显著提升大视觉语言模型的多模态推理能力。

**[Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](reinforcement_learning/stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)**

:   提出 SCORER 框架，将 Deep Q-Learning 中的表征学习和值函数学习建模为 Stackelberg 博弈，通过双时间尺度更新（Q 网络为 leader 慢更新、编码器为 follower 快更新）实现稳定协同适应，无需改变网络结构即可提升性能。

**[Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](reinforcement_learning/stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)**

:   提出 ARLCP（Adaptive Reflection and Length Coordinated Penalty），一种自适应强化学习方法，根据问题复杂度动态调节反思惩罚和长度惩罚的权重，在保持或提升准确性的同时大幅减少推理 token 消耗。

**[Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](reinforcement_learning/strict_subgoal_execution_reliable_long-horizon_planning_in_hierarchical_reinforc.md)**

:   提出 SSE（Strict Subgoal Execution）框架，通过**前沿经验回放（FER）** 严格区分子目标到达成功与失败，配合解耦探索策略和失败感知路径优化，在每个高层步骤内强制完成子目标到达，显著减少高层决策步数并提升长时程任务成功率。

**[SUSD: Structured Unsupervised Skill Discovery through State Factorization](reinforcement_learning/susd_structured_unsupervised_skill_discovery_through_state_factorization.md)**

:   提出 SUSD（Structured Unsupervised Skill Discovery），通过将状态空间分解为独立因子并为每个因子分配专属技能变量，结合好奇心驱动的因子加权机制，实现在多物体/多智能体复杂环境中发现覆盖全部可控因子的多样化技能。

**[$\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](reinforcement_learning/textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)**

:   本文提出 Re² 方法，通过纯强化学习训练 LLM 学会在推理过程中主动放弃无效思维链并重新开始求解，将罕见的 redo 行为从 0.5% 提升至 30% 以上，在相同训练计算预算下显著超越标准 RLVR 方法。

**[The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](reinforcement_learning/the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)**

:   本文为连续状态-动作空间下的非线性动力系统提出了一套在线强化学习算法，通过多模型后验采样和确定性等价策略实现对未知系统的在线学习，并给出了从有限模型集到参数化模型族的非渐近策略遗憾保证。

**[Thermodynamics of Reinforcement Learning Curricula](reinforcement_learning/thermodynamics_of_reinforcement_learning_curricula.md)**

:   本文利用非平衡热力学中的过剩功（excess work）最小化框架，将RL中的课程学习形式化为任务空间上的测地线优化问题，并推导出基于摩擦张量的温度退火算法MEW，在MuJoCo Humanoid任务上超越标准SAC温度调节方法。

**[Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)**

:   本文提出潜在思维策略优化（LTPO），一种无需更新模型参数的测试时推理增强框架，通过将中间潜在"思维"向量视为可优化的动态参数，利用在线策略梯度方法和内在置信度奖励信号来增强冻结LLM的推理能力。

**[Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](reinforcement_learning/towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)**

:   LIFT提出预训练-微调三阶段框架：(i) 大规模并行SAC预训练实现零样本部署；(ii) 基于拉格朗日动力学的物理先验世界模型离线预训练；(iii) 确定性动作执行+世界模型内随机探索的高效微调，在Booster T1和Unitree G1人形机器人上验证了从仿真到真实世界的全流程。

**[Towards Strategic Persuasion with Language Models](reinforcement_learning/towards_strategic_persuasion_with_language_models.md)**

:   本文以贝叶斯说服（Bayesian Persuasion）框架为理论基础，提出了一套系统评估和训练LLM策略性说服能力的方法，发现前沿模型已具备显著的策略性说服能力，且即使是小型LLM也可通过强化学习大幅提升说服效果。

**[TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models](reinforcement_learning/tpru_advancing_temporal_and_procedural_understanding_in_large_multimodal_models.md)**

:   TPRU构建了大规模多图像时序理解数据集（24,750个QA对、126,000张图像），覆盖机器人操作、GUI导航等4个具身场景的3种互补任务（时序排序、下一帧预测、前帧回溯），并通过强化学习微调使7B模型在时序理解上超越GPT-4o。

**[TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design](reinforcement_learning/traced_transition-aware_regret_approximation_with_co-learnability_for_environmen.md)**

:   TRACED改进无监督环境设计（UED）中的regret近似——在传统PVL基础上加入转移预测误差（ATPL）捕获动力学模型失配，并引入Co-Learnability度量任务间迁移效益，在MiniGrid和BipedalWalker上以10k更新超越所有baseline的20k更新性能。

**[Transitive RL: Value Learning via Divide and Conquer](reinforcement_learning/transitive_rl_value_learning_via_divide_and_conquer.md)**

:   本文提出 Transitive Reinforcement Learning（TRL），一种基于分治范式的新型值函数学习算法，利用目标条件RL中固有的三角不等式结构，将值函数更新递归分解为子问题，在长时间跨度任务上实现了优于TD学习和蒙特卡洛方法的性能。

**[Trinity: An Evolved LLM Coordinator](reinforcement_learning/trinity_an_evolved_llm_coordinator.md)**

:   Trinity设计了一个轻量级coordinator（0.6B SLM + ~10K可训练参数的head），通过sep-CMA-ES优化，在多轮对话中将查询分配给不同LLM并指定Thinker/Worker/Verifier三种角色，在LiveCodeBench上达到86.2% pass@1的SOTA，在4个分布内和4个分布外任务上一致超越所有单模型和多agent基线。

**[TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](reinforcement_learning/troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)**

:   本文提出 TROLL（Trust Region Optimization for Large Language models），用可微分的离散信任域投影替代PPO中的裁剪（clipping）机制，实现了基于原则性KL约束的token级策略更新，在数学推理和代码生成任务上一致性地优于PPO裁剪方法。

**[UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)**

:   提出 UME-R1，首次探索推理驱动的生成式多模态嵌入范式，通过两阶段训练（冷启动SFT + 强化学习）让嵌入模型先推理再生成表示，在 MMEB-V2 基准的 78 个任务上显著超越传统判别式嵌入模型。

**[Understanding and Improving Hyperbolic Deep Reinforcement Learning](reinforcement_learning/understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)**

:   通过闭式梯度分析揭示双曲深度 RL 中 Poincaré Ball 保角因子爆炸和大范数嵌入导致 PPO 信赖域失效的根源，提出 Hyper++（RMSNorm + 可学习缩放 + HL-Gauss + Hyperboloid）四组件方案，在 ProcGen 16 环境和 Atari-5 上全面超越先前基线。

**[Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals](reinforcement_learning/unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)**

:   提出 ULEE，一种无监督元学习方法，通过对抗式自生成目标课程训练自适应策略，在 XLand-MiniGrid 基准上实现高效探索与少样本适应。

**[Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)**

:   构建基于心智理论（ToM）的层次化多模态情感理解基准 HitEmotion，并提出 TMPO 框架通过中间心理状态作为过程级监督来增强 MLLM 的情感推理能力。

**[Value Flows](reinforcement_learning/value_flows.md)**

:   Value Flows 首次将流匹配（flow matching）引入分布式 RL——学习一个向量场使生成的概率密度路径自动满足分布式 Bellman 方程，通过 flow derivative ODE 高效估计回报方差实现置信度加权优先学习，在 OGBench 62 个任务上平均 1.3× 成功率提升，回报分布估计精度比 C51/CODAC 好 3×+。

**[VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)**

:   针对大型推理模型（LRM）训练中广泛使用的基于参考答案的奖励系统，构建了 VerifyBench 和 VerifyBench-Hard 两个评测基准，通过严格的人工标注评估各类验证系统的准确性，发现即使最强模型在困难样本上也仅达约 88% 准确率，揭示了当前验证系统的显著改进空间。

**[Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](reinforcement_learning/virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)**

:   提出 Virne——一个面向网络功能虚拟化资源分配（NFV-RA）的综合基准框架，集成 30+ 种算法和 gym 风格环境，支持云、边缘、5G 等多场景的系统评估。

**[VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](reinforcement_learning/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)**

:   提出 VTool-R1，首个通过强化学习微调训练 VLM 生成交错文本和视觉中间推理步骤的框架，使模型学会"用图像思考"。

**[Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](reinforcement_learning/whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)**

:   提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。

**[When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift](reinforcement_learning/when_sensors_fail_temporal_sequence_models_for_robust_ppo_under_sensor_drift.md)**

:   本文研究PPO在时间持续性传感器故障下的鲁棒性，提出将Transformer和SSM等序列模型集成到PPO中，推导了随机传感器故障下无限时间horizon奖励退化的高概率上界，并在MuJoCo实验中验证Transformer-PPO在严重传感器dropout下显著优于MLP、RNN和SSM基线。

**[WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)**

:   WIMLE将隐式最大似然估计（IMLE）扩展到model-based RL，学习能捕获多模态转移动力学的随机世界模型，通过ensemble+latent采样估计预测不确定性，用不确定性加权合成数据的RL目标，在40个连续控制任务上实现超越模型-free和model-based强基线的样本效率和渐近性能。

---

## 📦 模型压缩 { #model_compression }

**[A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](model_compression/a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)**

:   用信息论推导出 LLM 单次推理在多跳 QA 中的 Fano 式准确率上界，揭示当任务信息需求超过模型输出容量时准确率会"悬崖式"骤降的现象，并据此设计多轮推理框架 InfoQA，通过容量感知分解、依赖显式工作流和迭代查询压缩来突破单次推理瓶颈。

**[A Recovery Guarantee for Sparse Neural Networks](model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)**

:   证明了 ReLU 神经网络的首个稀疏恢复保证：对两层标量输出网络，当训练数据为高斯随机采样时，基于凸重构的迭代硬阈值 (IHT) 算法可精确恢复稀疏网络权重，且内存需求仅与非零权重数线性增长。

**[A State-Transition Framework for Efficient LLM Reasoning](model_compression/a_state-transition_framework_for_efficient_llm_reasoning.md)**

:   提出将 LLM 推理过程建模为状态转移过程的高效推理框架，用 Linear Attention 将历史推理步骤的信息压缩为状态矩阵，使注意力复杂度从 $O(C^2)$ 降为 $O(C)$、KV cache 从 $O(C)$ 降为 $O(1)$，同时不缩短 CoT 序列，保持推理能力。额外的动量 momentum 策略缓解了噪声推理步导致的 overthinking 问题。

**[A universal compression theory for lottery ticket hypothesis and neural scaling laws](model_compression/a_universal_compression_theory_for_lottery_ticket_hypothesis_and_neural_scaling_.md)**

:   本文证明了一个通用压缩定理：任意置换不变函数可以被渐近压缩至 polylog(d) 规模且误差趋近于零（这是最优压缩率），由此直接推导出动态彩票假说的证明——任何网络可被压缩至多对数宽度同时保持学习动力学不变，以及数据集可被压缩至多对数大小同时保持损失景观不变，并且幂律缩放定律可被加速至任意快的衰减率。

**[ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)**

:   提出 ABBA 适配器，将权重更新参数化为两个独立可学习的低秩矩阵的 Hadamard 积 $\Delta W = s(B_1A_1) \odot (B_2A_2)$，在相同参数预算下实现远高于 LoRA 的有效秩（$r_1 \cdot r_2$ vs $r$），并通过 Khatri-Rao 重构实现与 LoRA 相当的内存效率，在算术和常识推理任务上显著超越现有 PEFT 方法。

**[ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)**

:   构建 ACPBench Hard——基于 PDDL 形式化系统的 8 类**开放式生成**规划推理 benchmark（13 域 × 8 任务 = 1040 题），配备具有严格正确性保证的符号验证器，系统测评 15 个 LLM 后发现：即使最强推理模型 o1-preview 在半数任务上准确率 ≤66%，所有模型在最基本的"列举可执行动作"任务上几乎完全失败，暴露了当前 LLM 在规划推理方面的根本不足。

**[Adaptive Width Neural Networks](model_compression/adaptive_width_neural_networks.md)**

:   提出AWN框架，通过变分推断在训练过程中自动学习每层的无上界宽度（神经元数量），利用单调递减的重要性函数对神经元施加软排序，实现宽度自适应于任务难度，并支持零成本的训练后截断压缩。

**[AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](model_compression/amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)**

:   提出α-mixture assistant distribution及统一蒸馏框架AMiD，通过引入新设计变量α（控制教师-学生分布插值路径的几何形状）泛化了现有辅助分布方法（m-mixture和e-mixture为α=±1的特例），并证明了在任意散度和α下的最优性保证，在多个LLM蒸馏基准上取得SOTA性能。

**[AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs](model_compression/anybcq_hardware_efficient_flexible_binary-coded_quantization_for_multi-precision.md)**

:   提出AnyBCQ，基于二进制编码量化(BCQ)的多精度LLM量化框架，通过渐进式精度扩展（冻结已有bit-plane+添加残差bit-plane）支持单个模型在2-4bit之间动态切换，专设CUDA内核直接在bit-plane级别计算避免查表/转置开销，在2-bit下准确率大幅超越Any-Precision LLM（MMLU 35.3% vs 24.7%），吞吐量最高达到FP16的3.0x。

**[BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](model_compression/beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)**

:   提出BeyondBench评估框架，通过算法化动态生成数学问题（44个任务/117个变体/3个难度级别），确保每次测试不被训练数据污染，评估了101个语言模型（0.5B-141B参数），发现即使最强模型在Hard Suite上也仅达56%准确率，且不使用工具时性能大幅下降。

**[BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](model_compression/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)**

:   提出 BiasScope，一个完全由 LLM 驱动的迭代式框架，能自动、大规模地发现 LLM-as-a-Judge 中的潜在未知偏差，并基于此构建了更具挑战性的 JudgeBench-Pro 基准，在其上即使强大的 LLM 评估器错误率也超过 50%。

**[Boomerang Distillation Enables Zero-Shot Model Size Interpolation](model_compression/boomerang_distillation_enables_zero-shot_model_size_interpolation.md)**

:   提出"回旋蒸馏"范式——只训练一个小 student 模型，通过将 teacher 的 transformer 层块逐步贴回 student，零训练代价地构建出一整族中间尺寸模型，性能在 student 与 teacher 之间平滑插值，匹配甚至超越逐个蒸馏的同等规模模型。

**[Boosting Entropy with Bell Box Quantization](model_compression/boosting_entropy_with_bell_box_quantization.md)**

:   提出 Bell Box Quantization (BBQ)，首个同时满足"信息论最优"(ITO) 和"计算高效"(compute-efficient) 的量化方法，核心洞察是学习的域无关性——量化器输出域不必与输入域相同，由此在输入域做 ITO 量化以最大化熵，在输出域映射到硬件可加速的数据类型，在 1-4 bit QAPT 场景下全面超越 QuEST 和 LSQ。

**[BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](model_compression/bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)**

:   提出 BOTS——一个基于贝叶斯推断的在线任务选择统一框架，在 LLM 强化微调中通过融合显式证据（直接评估的历史通过率）和隐式证据（利用参考模型插值推断的未评估任务难度），配合 Thompson 采样实现探索-利用平衡，在数学/代码/逻辑任务上以仅 0.2% 的额外开销带来最高 50% 的训练加速。

**[Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers](model_compression/bridging_kolmogorov_complexity_and_deep_learning_asymptotically_optimal_descript.md)**

:   从柯尔莫哥洛夫复杂度理论出发，提出了"渐近最优描述长度目标"的理论框架，证明了 Transformer 存在这样的目标函数（基于其计算通用性的新证明），并通过构造基于自适应高斯混合先验的可微变分目标进行了实证验证，揭示了重要的优化挑战。

**[COMI: Coarse-to-fine Context Compression via Marginal Information Gain](model_compression/comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)**

:   提出 COMI，一种基于边际信息增益（MIG = 查询相关性 - 语义冗余度）的粗到细自适应上下文压缩框架，在 32x 压缩率下 NaturalQuestions EM 比次优方法提高约 25 分，核心在于同时优化保留信息的相关性和多样性。

**[Compute-Optimal Quantization-Aware Training](model_compression/compute-optimal_quantization-aware_training.md)**

:   本文通过 757 组 QAT 实验（86M-2.2B 参数，1-6 bit）发现：QAT 的最优训练比例随总计算量增长而增大（与先前认为固定 10% 的结论相反），并提出 tokens-per-parameter-byte 统计量和新的 loss scaling law 来精确预测最优 QAT 分配策略和最终损失。

**[Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](model_compression/cross_domain_lossy_compression_optimal_transport.md)**

:   将跨域有损压缩（编码器观测退化源、解码器重建不同目标分布）形式化为带压缩率和分类损失双重约束的最优传输问题，推导Bernoulli源（Hamming失真）和Gaussian源（MSE）的闭式DRC/RDC及DRPC权衡函数，通过深度端到端压缩模型在超分/去噪/修复任务上验证理论预测与实验行为一致。

**[Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)**

:   将结构化剪枝和模型折叠(model folding)统一为正交投影框架——剪枝是坐标轴对齐投影，折叠是聚类子空间投影——并证明在秩差为1的条件下折叠的参数重建误差严格更小，在1000+个 checkpoint 上验证折叠在中-高压缩率下通常优于剪枝。

**[Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)**

:   提出 Dataset Color Quantization（DCQ）框架，通过色度感知聚类、注意力引导调色板分配和纹理保持优化三个机制，在数据集层面减少颜色冗余实现存储压缩，同时保持训练效果。

**[Dataset Distillation as Pushforward Optimal Quantization](model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)**

:   将解耦式数据集蒸馏重新形式化为最优量化问题，证明通过扩散先验的潜空间聚类+权重可收敛逼近真实数据分布，提出 DDOQ 算法在 ImageNet-1K 上以极低额外计算量超越 D4M 等基线。

**[Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](model_compression/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)**

:   提出 Discount Model Search (DMS)，用神经网络拟合连续平滑的 discount 函数替代 CMA-MAE 中基于直方图的离散表示，解决高维 measure space 下 distortion 导致搜索停滞的问题，并首次实现以图像数据集直接定义 measure space（QDDM 范式）。

**[Distillation of Large Language Models via Concrete Score Matching](model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)**

:   提出 Concrete Score Distillation (CSD)，一种基于离散 score matching 的 LLM 知识蒸馏损失，通过匹配 student 和 teacher 在所有词表对之间的相对 logit 差异，同时克服了 softmax 平滑和直接 logit 蒸馏的解空间限制问题。

**[Draft-based Approximate Inference for LLMs](model_compression/draft-based_approximate_inference_for_llms.md)**

:   提出 Draft-based Approximate Inference 框架，利用小型 draft 模型的前瞻（lookahead）预测来更准确地估计 token/KV pair 重要性，包含 SpecKV（KV cache dropping）、SpecPC（prompt 压缩）和 SpecKV-PC（级联压缩）三种方法，在长上下文 benchmark 上一致优于现有基线。

**[Efficient Reasoning with Balanced Thinking](model_compression/efficient_reasoning_with_balanced_thinking.md)**

:   提出 ReBalance，一个无需训练的框架，通过基于置信度的动态隐状态导向（steering vector），同时缓解大推理模型（LRM）的过度思考和欠思考问题，实现推理效率与准确率的双重提升。

**[Einstein Fields: A Neural Perspective To Computational General Relativity](model_compression/einstein_fields_a_neural_perspective_to_computational_general_relativity.md)**

:   提出EinFields，首个将神经隐式表示应用于四维广义相对论模拟压缩的框架，通过将度量张量场编码为紧凑神经网络权重，实现4000倍存储压缩、5-7位数值精度，且通过自动微分获得的张量导数比有限差分精度高5个数量级。

**[Embedding Compression via Spherical Coordinates](model_compression/embedding_compression_via_spherical_coordinates.md)**

:   提出一种基于球坐标变换的嵌入向量压缩方法，利用高维单位向量的球坐标角度集中在 $\pi/2$ 附近的数学性质，使 IEEE 754 浮点数的指数位和高阶尾数位熵大幅降低，实现 1.5× 压缩率，比最优无损方法提升 25%，重建误差低于 float32 机器精度。

**[ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping](model_compression/es-dllm_efficient_inference_for_diffusion_large_language_models_by_early-skippin.md)**

:   针对扩散大语言模型（dLLM）推理中大量 token 计算冗余的问题，提出无需训练的 Early-Skipping 加速框架 ES-dLLM，通过估计 token 重要性并在早期层跳过低重要性位置，在 LLaDA-8B 和 Dream-7B 上实现 5.6×–16.8× 加速且不损失生成质量。

**[Evolution and compression in LLMs: On the emergence of human-aligned categorization](model_compression/evolution_and_compression_in_llms_on_the_emergence_of_human-aligned_categorizati.md)**

:   通过 Information Bottleneck (IB) 框架和迭代上下文语言学习 (IICLL) 范式，证明 LLM 能够在未经 IB 目标训练的情况下，自发涌现出与人类语义分类系统高度对齐的、近最优压缩效率的类别结构。

**[ExGRPO: Learning to Reason from Experience](model_compression/exgrpo_learning_to_reason_from_experience.md)**

:   首次系统研究什么样的推理经验对RLVR最有价值，发现中等难度问题+低熵轨迹最有效，据此提出ExGRPO框架进行经验管理和混合策略优化，在数学推理上平均+3.5分，通用推理+7.6分。

**[FASA: Frequency-aware Sparse Attention](model_compression/fasa_frequency-aware_sparse_attention.md)**

:   发现 RoPE 注意力在频率块(FC)级别存在功能稀疏性——仅不到 1% 的"主导 FC"就能近似完整注意力头的 token 选择行为。据此设计无需训练的 FASA 框架，通过两阶段策略（主导 FC 预测 token 重要性 → 仅对重要 token 做完整注意力）实现 8× 内存压缩和 2.6× 推理加速且几乎无质量损失。

**[Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)**

:   提出QZO方法，通过对量化缩放因子（而非离散权重）做零阶扰动来估计梯度，配合方向导数裁剪稳定训练，实现4-bit/2-bit LLM的极致内存高效微调，总内存降低18倍以上。

**[FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](model_compression/flyprompt_brain-inspired_random-expanded_routing_with_temporal-ensemble_experts_.md)**

:   受果蝇蘑菇体神经系统启发，提出 FlyPrompt 框架将通用持续学习（GCL）分解为专家路由和专家能力提升两个子问题，通过随机扩展解析路由器（REAR）和时序集成专家（TE2）分别解决，在 CIFAR-100/ImageNet-R/CUB-200 上分别提升 11.23%/12.43%/7.62%。

**[FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)**

:   提出 FreqKV，一种无参数、架构无关的 KV 缓存压缩方法，通过在频域中迭代压缩 KV 状态（保留低频丢弃高频），仅需 8K 长度的少量微调即可将 LLaMA-2-7B 的上下文窗口扩展至 256K，同时保持稳定的困惑度。

**[GASP: Guided Asymmetric Self-Play For Coding LLMs](model_compression/gasp_guided_asymmetric_self-play_for_coding_llms.md)**

:   提出GASP框架，在非对称自博弈中引入"goalpost"（硬目标题）引导教师生成有针对性的训练问题，通过lemma（简化变体）→lift（加难变体）的课程结构逐步逼近困难目标，在LiveCodeBench上超越无引导自博弈2.5%且解决了所有baseline无法解决的难题。

**[Grounding and Enhancing Informativeness and Utility in Dataset Distillation](model_compression/grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)**

:   提出InfoUtil框架，用博弈论Shapley Value最大化样本信息量（找到最重要的patch），用梯度范数最大化样本效用（选择对训练最有价值的样本），在ImageNet-1K上比前SOTA提升6.1%。

**[GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](model_compression/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)**

:   提出 GuidedSampling 推理算法，将重复采样（RS）的隐式探索和生成过程显式解耦为两阶段：先迭代生成多样化的解题概念/定理，再基于各概念分别生成候选解。在 pass@50 上平均提升约 21.6%，微调后 pass@5 提升约 9.7%。

**[HeurekaBench: A Benchmarking Framework for AI Co-scientist](model_compression/heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)**

:   提出 HeurekaBench，一个基于真实科学工作流构建评测基准的框架，通过多LLM流水线从论文中提取可验证的科学洞见并生成开放式研究问题，用于评估AI co-scientist在数据驱动科学发现中的端到端能力。

**[HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design](model_compression/hifo-prompt_prompting_with_hindsight_and_foresight_for_llm-based_automatic_heuri.md)**

:   提出 HiFo-Prompt 框架，通过 Hindsight（回顾式知识池）和 Foresight（前瞻式进化导航器）两个协同模块提升 LLM 驱动的自动启发式设计（AHD），在 TSP 和 FSSP 等任务上显著超越现有方法。

**[Highly Efficient and Effective LLMs with Multi-Boolean Architectures](model_compression/highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)**

:   提出一种用多核布尔参数（multi-kernel Boolean parameters）表示 LLM 权重的新框架，首次实现在布尔域中直接微调大语言模型，无需全精度潜在权重，在表征能力和计算效率上同时超越现有超低比特量化和二值化方法。

**[Human-LLM Collaborative Feature Engineering for Tabular Learning](model_compression/human-llm_collaborative_feature_engineering_for_tabular_data.md)**

:   提出一个人-LLM协作特征工程框架，将LLM的特征操作提议与选择过程解耦，通过贝叶斯神经网络建模操作效用和不确定性来指导选择，并选择性地引入人类偏好反馈，在18个表格数据集上平均错误率降低8.96%~11.23%。

**[IDER: IDempotent Experience Replay for Reliable Continual Learning](model_compression/ider_idempotent_experience_replay_for_reliable_continual_learning.md)**

:   将幂等性（idempotence）引入持续学习，通过标准幂等模块和幂等蒸馏模块两个组件强制模型在学习新任务时保持输出自一致性，在提升预测可靠性（降低校准误差）的同时显著减少灾难性遗忘。

**[In-Context Learning for Pure Exploration](model_compression/in-context_learning_for_pure_exploration.md)**

:   提出 ICPE（In-Context Pure Exploration），一种结合监督学习和强化学习的上下文学习框架，使用 Transformer 从经验中直接学习探索策略，在主动序列假设检验/纯探索问题中实现接近最优的实例自适应算法性能，无需显式建模信息结构。

**[Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning](model_compression/incentivizing_agentic_reasoning_in_llm_judges_via_tool-integrated_reinforcement_.md)**

:   提出 TIR-Judge，一个端到端的 RL 框架，训练 LLM 评判模型在评估过程中交替使用推理和代码执行工具，在7个公开基准上以 8B 参数超越 32B 推理奖励模型，且无需蒸馏的 TIR-Judge-Zero 可自举提升。

**[InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](model_compression/inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)**

:   提出 InftyThink，一种将整体式长推理转化为迭代式短推理+中间摘要的新范式，在不修改模型架构的前提下实现理论上无界的推理深度、显著降低计算成本，Qwen2.5-Math-7B 在 AIME24 上提升11%。

**[Is Finer Better? The Limits of Microscaling Formats in Large Language Models](model_compression/is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)**

:   发现并解释了微缩放（microscaling）量化中"更细粒度反而更差"的反直觉异常——当block size减小到阈值以下时，FP8 UE4M3 scale的有限动态范围导致窄分布张量的量化误差反而增大，并提出 FP8 UE5M3 scale格式作为硬件友好的解决方案。

**[Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](model_compression/is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)**

:   提出反转诅咒（Reversal Curse）是认知科学中"绑定问题"在Transformer中的表现——源于概念表示的不一致性和纠缠性，并首次设计出基于JEPA和记忆层的架构真正突破反转诅咒（非绕过）。

**[IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](model_compression/iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)**

:   提出 IterResearch，一种基于MDP的迭代深度研究范式，通过周期性工作区重构替代单上下文线性累积，使Agent在40K上下文长度下扩展到2048次交互（性能从3.5%提升至42.5%），在6个benchmark上平均超出开源Agent 14.5个百分点。

**[KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](model_compression/kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)**

:   提出 KBVQ-MoE，首个专为MoE架构设计的向量量化框架，通过KLT引导的SVD消除专家间冗余共享（IDRE），以及偏差校正的输出稳定化（BCOS），在2-bit量化下比现有方法提升10%+准确率。

**[Knowledge Fusion of Large Language Models Via Modular Skillpacks](model_compression/knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)**

:   提出GraftLLM——将异构源模型的能力提取为紧凑可迁移的"SkillPack"（模块化技能包），通过模块感知自适应压缩策略存储参数增量，支持知识迁移、异构模型融合和无遗忘持续学习，在多个场景下显著优于现有PEFT和参数融合方法。

**[Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)**

:   提出 Landscape of Thoughts (LoT)，首个将LLM推理轨迹可视化为二维地形图的工具，通过困惑度特征和t-SNE投影揭示推理行为模式，并可适配为轻量验证器提升推理准确率和测试时扩展效果。

**[LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)**

:   提出 LD-MoLE，用 Sparsegen 闭合形式投影替代传统 TopK 路由，实现可微分、动态、token自适应的 LoRA 专家分配，配合轻量 MLP 预测稀疏因子和解析稀疏损失，在多个基准上超越固定路由和 ReLU 路由基线。

**[LightMem: Lightweight and Efficient Memory-Augmented Generation](model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)**

:   提出 LightMem，一个受人类 Atkinson-Shiffrin 记忆模型启发的三阶段轻量记忆系统，通过认知感觉记忆预压缩、主题感知短期记忆整合、睡眠时离线更新三个模块，在 LongMemEval 上准确率提升最高7.7%，同时 token 消耗降低高达38倍。

**[LLM DNA: Tracing Model Evolution via Functional Representations](model_compression/llm_dna_tracing_model_evolution_via_functional_representations.md)**

:   从生物学 DNA 类比出发，将 LLM DNA 数学定义为模型功能行为的低维双 Lipschitz 表示，证明其满足遗传和基因决定性属性，并设计了无需训练的 RepTrace 管道在 305 个 LLM 上提取 DNA、构建进化树。

**[LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)**

:   提出 LoFT，一种通过对齐优化器内部动态（动量和二阶矩）与全参微调行为一致的低秩适配方法，由六个构建模块组成，在全秩极限下可精确恢复 AdamW，在多项基准上显著缩小 LoRA 与全参微调的性能差距。

**[LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation](model_compression/lookaheadkv_fast_and_accurate_kv_cache_eviction_by_glimpsing_into_the_future_wit.md)**

:   提出 LookaheadKV，通过可学习的前瞻token和选择性激活的LoRA模块预测真实响应的注意力重要性分数，实现无需生成草稿的快速精确KV缓存淘汰，在多个长上下文基准上超越现有方法，驱逐开销降低最高14.5倍。

**[Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)**

:   提出 Memba，一种受生物神经元膜电位启发的参数高效微调方法，通过在 Mamba 门控分支引入泄漏积分膜（LIM）神经元实现时序自适应，结合 LoRA 放置优化和跨层膜传递，以极少参数在语言和视觉任务上超越现有 Mamba PEFT 方法。

**[MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes](model_compression/mobilellm-r1_exploring_the_limits_of_sub-billion_language_model_reasoners_with_o.md)**

:   通过精心的数据筛选和自适应混合策略，仅用4.2T token（Qwen3的11.7%）预训练出亿级参数的推理模型 MobileLLM-R1-950M，在AIME等推理基准上匹配或超越 Qwen3-0.6B，同时完全开源数据源和训练配方。

**[Modality-free Graph In-context Alignment](model_compression/modality-free_graph_in-context_alignment.md)**

:   提出 MF-GIA，首个同时满足无后训练、跨域对齐和模态无关三个条件的图上下文学习框架，通过梯度指纹捕获域特征、FiLM条件化变换对齐特征和标签，在多个图域的few-shot任务上实现SOTA性能。

**[MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](model_compression/mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)**

:   提出 MoNE（Mixture-of-Novices-and-Experts），通过联合评估专家的访问频率和输出方差来识别冗余专家，并用其输出均值（"新手"常量向量）替换之，在5种MoE模型上实现比现有剪枝方法更有效且更鲁棒的压缩，25%剪枝率下平均准确率下降仅0.14。

**[Multi-View Encoders for Performance Prediction in LLM-Based Agentic Workflows](model_compression/multi-view_encoders_for_performance_prediction_in_llm-based_agentic_workflows.md)**

:   提出 Agentic Predictor，一种多视图工作流编码框架，通过联合建模图结构、代码语义和提示信息来预测 LLM Agent 工作流的性能，显著减少昂贵的试错评估。

**[Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)**

:   提出 NUFILT 框架，利用"任务向量与表示子空间近似对齐"的几何性质，通过零空间滤波压制对旧任务的干扰、投影感知 LoRA 恢复新任务可塑性，在完全不访问数据的条件下实现持续模型合并，在视觉/NLP/多模态基准上比 OPCM 提升 4-8%，逼近独立微调的上界。

**[Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](model_compression/obscure_but_effective_classical_chinese_jailbreak_prompt_optimization_via_bio-in.md)**

:   提出 CC-BOS 框架，利用文言文的语义压缩和模糊性特征，结合果蝇优化算法在八维策略空间中搜索最优越狱提示，在六个主流 LLM 上实现近 100% 的攻击成功率。

**[Parallel Token Prediction for Language Models](model_compression/parallel_token_prediction_for_language_models.md)**

:   提出 Parallel Token Prediction (PTP)，通过将采样随机性从后处理移至模型输入（辅助变量），使未来 token 成为确定性函数，从而在单次前向传播中联合预测多个 token。

**[ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](model_compression/paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)**

:   提出 ParoQuant，通过硬件高效且可优化的独立 Givens 旋转与通道缩放相结合来消除权重异常值，在推理 LLM 上实现高精度低开销的 4-bit 权重量化。

**[PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](model_compression/paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)**

:   提出PASER，一种针对剪枝LLM恢复的后训练数据选择方法，通过流形学习+谱聚类识别能力相关指令集，按能力退化程度自适应分配数据预算，仅用4%-20%原始数据即可显著超越全量数据恢复效果。

**[Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)**

:   提出 IOA（Identifier-Organizer-Adapter）框架，借鉴 Bloom 掌握学习原则和 Vygotsky 最近发展区理论，通过诊断知识缺陷、设计渐进课程、适配认知水平三个阶段，实现教育学驱动的 LLM 知识蒸馏。

**[π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](model_compression/pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)**

:   提出 π-Flow，通过修改学生流模型的输出层使其预测一个"策略"（policy），该策略在单个网络评估内通过多个子步生成动态流速度进行精确 ODE 积分，并采用模仿蒸馏（imitation distillation）方法在学生自己的轨迹上匹配教师速度，从而实现稳定可扩展的少步生成并避免质量-多样性权衡。

**[PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)**

:   提出 PTQ4ARVG，首个针对自回归视觉生成（ARVG）模型的系统化 PTQ 框架，通过增益投影缩放（GPS）、静态 Token 级量化（STWQ）和分布引导校准（DGC）解决 ARVG 特有的三大量化挑战。

**[QKV Projections Require a Fraction of Their Memory](model_compression/qkv_projections_require_a_fraction_of_their_memory.md)**

:   提出 PAMM（Point-Approximate Matrix Multiplication），一种激活压缩技术，通过随机选取少量代表性 token 来近似 QKV 投影层激活，实现高达 512× 压缩率且不影响模型性能。

**[Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation](model_compression/rectified_decoupled_dataset_distillation_a_closer_look_for_fair_and_comprehensiv.md)**

:   提出 RD3（Rectified Decoupled Dataset Distillation），系统揭示现有解耦数据集蒸馏方法的性能差异主要源于不一致的后评估设置而非蒸馏质量差异，建立了统一公平的评估框架，将报告的 27.3% 性能差距校正为 6.7%。

**[Reference-Guided Machine Unlearning](model_compression/reference-guided_machine_unlearning.md)**

:   提出 ReGUn（Reference-Guided Unlearning），利用独立留出数据集作为"未见行为"的参考标准，通过类别条件蒸馏将遗忘数据上的模型行为对齐到真正未见数据的行为，实现更优的遗忘-效用权衡。

**[Rethinking Continual Learning with Progressive Neural Collapse](model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)**

:   提出 ProNC 框架，通过渐进式扩展等角紧框架（ETF）目标替代固定预定义 ETF，在持续学习中实现最大类间分离与最小遗忘的平衡。

**[Revisiting Weight Regularization for Low-Rank Continual Learning](model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)**

:   在低秩持续学习中重新引入弹性权重巩固（EWC），通过在全维空间估计 Fisher 信息矩阵来正则化共享 LoRA 模块，实现恒定存储开销下的有效遗忘缓解。

**[S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)**

:   提出 S2R-HDR，首个大规模高质量合成 HDR 融合数据集（24,000 样本），并设计 S2R-Adapter 域适应方法弥合合成-真实域差距，在真实数据集上达到 SOTA HDR 融合性能。

**[Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](model_compression/scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)**

:   系统性揭示了 LLM 在推理跳步泛化（reasoning hop generalization）中失败的内部机制——正确与错误推理轨迹间的注意力头竞争，并提出 TCR（Test-time Correction of Reasoning），通过动态识别和停用错误处理头（ep heads）在测试时纠正推理错误，平均提升 5-7% 准确率。

**[SeeDNorm: Self-Rescaled Dynamic Normalization](model_compression/seednorm_self-rescaled_dynamic_normalization.md)**

:   提出 SeeDNorm，一种自适应动态归一化层，通过将输入自身作为条件来动态调整缩放系数，从而在前向传播中保留输入范数信息，同时在反向传播中保持类似 RMSNorm 的自适应梯度调整能力，以极少额外参数在语言建模和视觉任务上全面超越 RMSNorm、LayerNorm 和 DyT。

**[SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)**

:   提出 SERE 方法，通过预计算专家相似度矩阵，在批量解码时将次要专家动态重路由到最相似的主要专家，实现最高 2.0 倍加速且质量损失极小，并提供即插即用的 vLLM CUDA 内核。

**[SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs](model_compression/sft_doesnt_always_hurt_general_capabilities_revisiting_domain-specific_fine-tuni.md)**

:   本文系统性地重新审视了领域特定SFT对LLM通用能力的影响，发现**使用较小学习率即可大幅缓解通用能力退化**，并提出Token-Adaptive Loss Reweighting (TALR)方法通过自适应下调低概率token的损失权重进一步优化领域适配与通用能力之间的权衡。

**[Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](model_compression/slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)**

:   提出 SFPO（Slow-Fast Policy Optimization），通过将每个训练步分解为"快速轨迹—重定位—慢速校正"三阶段结构，在不修改目标函数和 rollout 过程的前提下即插即用地增强 GRPO 的稳定性和样本效率，在数学推理基准上平均提升最高 2.80 分，rollout 减少最多 4.93 倍。

**[SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](model_compression/sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)**

:   提出 SPARTA，一个端到端自动构建大规模表格-文本多跳问答基准的框架，通过参考事实数据库、来源引导的修复和现实结构约束生成高质量嵌套 SQL 查询，SOTA 模型在 SPARTA 上 F1 下降超过 30 分。

**[Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](model_compression/specialization_after_generalization_towards_understanding_test-time_training_in_.md)**

:   本文从线性表示假说（LRH）出发，提出"泛化后特化"（specialization after generalization）理论框架，首次在 in-distribution 场景下系统解释了 TTT 为何有效——基础模型因全局欠参数化导致概念叠加干扰，TTT 通过临时遗忘无关概念来释放模型容量、局部特化到测试任务相关的少量概念上，理论保证即使特征空间指数级小于概念空间也能泛化。

**[STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](model_compression/star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)**

:   提出 STAR 框架，通过约束知识蒸馏（CKD）和相似度引导的强化学习（Sim-RL）协同工作，将大模型的 function calling 能力有效迁移到 0.6B 级别的超小模型，在 BFCL 和 ACEBench 上大幅超越基线。

**[Steering MoE LLMs via Expert (De)Activation](model_compression/steering_moe_llms_via_expert_deactivation.md)**

:   提出 SteerMoE，通过对比配对输入检测行为关联专家，在推理时通过激活/去激活特定专家来引导 MoE LLM 的行为（安全性提升 +20%，忠实性提升 +27%），同时揭示 MoE 模型的安全对齐脆弱性（安全下降 -100%）。

**[Stress-Testing Alignment Audits with Prompt-Level Strategic Deception](model_compression/stress-testing_alignment_audits_with_prompt-level_strategic_deception.md)**

:   构建自动 prompt 级红队流水线（Claude Opus 4.5 驱动）增强现有微调模型有机体的情境感知与策略推理，对四种黑盒+白盒对齐审计方法进行压力测试，在 6 个实验设置中均成功诱导审计方法产生高置信错误猜测，首次记录了不修改权重的 prompt 级激活欺骗现象。

**[SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning](model_compression/swireasoning_switch-thinking_in_latent_and_explicit_for_pareto-superior_reasonin.md)**

:   提出 SwiReasoning，一种免训练的 LLM 推理框架，通过基于熵趋势的块级置信度估计，动态切换显式（chain-of-thought）和隐式（latent space）推理模式，在 Pareto 意义上同时改善准确率（+1.8%~3.1%）和 Token 效率（+57%~79%）。

**[Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation](model_compression/taming_momentum_rethinking_optimizer_states_through_low-rank_approximation.md)**

:   揭示动量 EMA 更新等价于在线线性回归的梯度下降，基于此提出 LoRA-Pre，通过低秩分解压缩优化器动量，实现显存高效的 LLM 预训练和微调，在所有模型尺度上达到最优性能且仅需基线方法 1/8 的秩。

**[Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](model_compression/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)**

:   提出 TAMPO（Temperature Adaptive Meta Policy Optimization），将采样温度重新定义为可学习的元策略，通过双层循环在内环做 LLM 策略优化、外环根据轨迹优势信号自适应更新温度分布，无需额外 rollout，在数学推理基准上一致超越固定温度基线。

**[Textual Equilibrium Propagation for Deep Compound AI Systems](model_compression/textual_equilibrium_propagation_for_deep_compound_ai_systems.md)**

:   提出文本平衡传播（TEP），一种基于局部学习原理的复合AI系统优化方法，通过自由阶段和微扰阶段的两阶段设计，避免全局文本反向传播中的梯度爆炸/消失问题，在深层工作流上显著优于 TextGrad。

**[The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](model_compression/the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)**

:   首次证明 GPTQ（从后向前执行时）在数学上等价于经典格理论中的 Babai 最近平面算法，由此获得几何解释和层级误差上界，并基于此设计了无裁剪的改进量化方法。

**[The Lattice Geometry of Neural Network Quantization -- A Short Equivalence Proof of GPTQ and Babai's Algorithm](model_compression/the_lattice_geometry_of_neural_network_quantization_--_a_short_equivalence_proof.md)**

:   独立于 Chen et al. (2026)，以更简洁优雅的方式证明 GPTQ 等价于 Babai 最近平面算法，并阐明格基约减可能改进神经网络量化的前景。

**[The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM](model_compression/the_unseen_frontier_pushing_the_limits_of_llm_sparsity_with_surrogate-free_admm.md)**

:   提出 Elsa 方法，通过无代理目标的 ADMM 约束优化直接求解稀疏性约束问题，突破 LLM 剪枝 50-60% 的"稀疏墙"瓶颈，在 90% 稀疏度下仍保持高模型保真度。

**[TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](model_compression/titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)**

:   提出 TiTok 框架，通过 token 级对比超额分数（contrastive excess）实现 LoRA 适配器跨模型高效迁移，无需额外判别器模型，在推理和个性化任务上一致超越 TransLoRA 和知识蒸馏基线。

**[Token Distillation: Attention-Aware Input Embeddings for New Tokens](model_compression/token_distillation_attention-aware_input_embeddings_for_new_tokens.md)**

:   提出 Token Distillation 方法，通过蒸馏 Transformer 各层编码的多子词交互信息到单一 token 嵌入中，实现高质量的新 token 嵌入初始化，无需预训练超网络且优于现有方法。

**[Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](model_compression/topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)**

:   从代数几何和代数拓扑的视角，系统研究了基于一般 DAG 架构的前馈 ReLU 网络参数空间的连通性和奇异性，揭示了瓶颈节点和平衡条件在决定参数空间拓扑结构中的关键作用，并建立了奇异性与可微剪枝的理论联系。

**[Towards Efficient Constraint Handling in Neural Solvers for Routing Problems](model_compression/towards_efficient_constraint_handling_in_neural_solvers_for_routing_problems.md)**

:   提出 Construct-and-Refine (CaR) 框架，通过联合训练构造模块和轻量改进模块实现高效的可行性修复，首次为硬约束路径问题提供通用、高效的神经约束处理方案，在 TSPTW 和 CVRPBLTW 上大幅超越经典和神经 SOTA 求解器。

**[TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](model_compression/turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)**

:   TurboBoA 提出了一种无需反向传播的 LLM 后训练量化方法，通过多 out-channel 联合量化、前层误差补偿和自适应网格选择三大创新，在保留 BoA 精度优势的同时实现了 3 倍以上加速。

**[Understanding Dataset Distillation via Spectral Filtering](model_compression/understanding_dataset_distillation_via_spectral_filtering.md)**

:   本文提出 UniDD 谱滤波框架，将多种数据集蒸馏方法统一为在特征-特征相关矩阵（FFC）上应用不同滤波函数来匹配特征-标签相关矩阵（FLC）的频率信息，并基于此洞见提出了课程频率匹配（CFM）方法。

**[Unveiling Super Experts in Mixture-of-Experts Large Language Models](model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)**

:   本文首次发现并系统研究了 MoE LLM 中的"超级专家"（Super Experts）——数量极少但对模型推理至关重要的专家子集，它们通过 down_proj 中的极端激活异常值驱动 massive activations 和 attention sinks 机制。

**[Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis](model_compression/why_attention_patterns_exist_a_unifying_temporal_perspective_analysis.md)**

:   本文提出 TAPPA 框架，从时间连续性视角统一解释了 LLM 中多种注意力模式（attention sink、对角线、周期性等）的形成机制，并通过 query 自相似性（q-similarity）指标指导 KV cache 压缩和模型剪枝任务。

---

## 🏥 医学图像 { #medical_imaging }

**[Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)**

:   提出CDTSDE框架，在扩散模型的逆向SDE中嵌入可学习的空间自适应域混合场 $\Lambda_t$，使跨模态翻译路径沿低能量流形前进，在MRI模态转换、SAR→光学、工业缺陷语义映射任务上以更少去噪步数实现更高保真度。

**[Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts](medical_imaging/adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat.md)**

:   提出AdaTTT框架，通过动态特征感知self-supervised学习（自适应掩码策略）和原型引导的部分最优传输对齐，在ICU多中心EHR数据上实现鲁棒的测试时适应，用于提前24小时预测有创机械通气需求。

**[AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](medical_imaging/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)**

:   构建了首个大规模抗体功能注释指令数据集AFD-Instruction（430K+条目），通过多智能体文献抽取pipeline对齐抗体序列与自然语言功能描述，用于指令微调通用LLM使其掌握抗体理解和功能导向设计能力，在5类分类任务上平均准确率提升20+点。

**[An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](medical_imaging/an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)**

:   将因果推断中的半参数效率理论系统引入MDP的Q函数估计，证明经典的Q-regression和FQE本质上是有plug-in偏差的朴素学习器，并提出DRQQ-learner——一个同时具备双重鲁棒性、Neyman正交性和准oracle效率的元学习器，通过推导有效影响函数(EIF)构造去偏二阶段损失，在Taxi和Frozen Lake环境中全面超越基线方法。

**[AntigenLM: Structure-Aware DNA Language Modeling for Influenza](medical_imaging/antigenlm_structure-aware_dna_language_modeling_for_influenza.md)**

:   AntigenLM 是一个保留基因组功能单元完整性的 GPT-2 风格 DNA 语言模型，通过在流感病毒全基因组上预训练并微调，能够自回归预测未来流行毒株的抗原序列，在氨基酸错配率上显著优于进化模型 beth-1 和通用基因组模型。

**[ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](medical_imaging/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)**

:   提出 ATPO（自适应树策略优化）算法，将多轮医疗对话建模为层级马尔可夫决策过程（H-MDP），通过不确定性感知的自适应树扩展机制动态分配rollout预算，结合Bellman误差和动作值方差的复合不确定性度量来引导探索，在三个医学对话基准上以Qwen3-8B超越GPT-4o。

**[Augmenting Representations with Scientific Papers](medical_imaging/augmenting_representations_with_scientific_papers.md)**

:   提出首个将 X 射线光谱与科学文献通过对比学习对齐的多模态基础模型框架，在共享潜在空间中实现 20% Recall@1% 的跨模态检索，物理参数估计提升 16–18%，同时发现候选脉动超亮 X 射线源等罕见天体。

**[Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](medical_imaging/benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)**

:   对8个ECG基础模型在12个数据集、26个临床任务上进行"现实检验"式全面基准评测，发现紧凑的结构化状态空间模型（SSM）ECG-CPC在7个任务类别中的5个上超越了大规模Transformer，证明架构设计比模型规模更重要。

**[BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](medical_imaging/biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)**

:   提出 BiomedSQL，首个专门评估 Text-to-SQL 系统在生物医学知识库上科学推理能力的基准，包含 68,000 个问题/SQL/答案三元组，揭示当前最强模型（GPT-o3-mini 62.6%）与领域专家（90%）之间仍有巨大差距。

**[Boosting Medical Visual Understanding From Multi-Granular Language Learning](medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)**

:   提出 Multi-Granular Language Learning (MGLL)，一个即插即用的对比学习框架，通过 soft CLIP loss、point-wise loss 和 smooth KL 散度联合优化，实现医学图像与多标签多粒度文本描述的对齐，在眼底和 X 光数据集上全面超越 SOTA 方法，并可作为视觉编码器嵌入多模态大语言模型提升诊断准确率最高达 34.1%。

**[Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](medical_imaging/brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)**

:   提出 Brain-IT 框架，通过脑启发式的 Brain Interaction Transformer (BIT) 将功能相似的脑体素聚类为跨被试共享的 Brain Token，并从中预测局部化的语义和结构图像特征，实现从 fMRI 到图像的高保真重建，仅用 1 小时数据即达到先前方法 40 小时的性能。

**[Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](medical_imaging/brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)**

:   提出 Brain-Semantoks，一种基于语义分词器和自蒸馏目标的 fMRI 基础模型，将大脑功能网络聚合为鲁棒的语义 token，并通过跨时间视角的一致性学习抽象的脑动态表征，在线性探测设置下即可达到 SOTA 性能。

**[Bridging Explainability and Embeddings: BEE Aware of Spuriousness](medical_imaging/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)**

:   提出BEE框架，通过分析微调如何扰动预训练表征的权重空间几何结构，直接从分类器学到的权重中识别和命名虚假相关性（spurious correlations），无需反例样本即可发现隐藏的数据偏差，在ImageNet-1k上发现可导致准确率下降高达95%的虚假关联。

**[Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](medical_imaging/can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)**

:   研究稀疏自编码器（SAE）能否揭示和缓解 LLM 在医疗场景中的种族偏见：发现 SAE 能识别出与种族相关的有害联想（如黑人与暴力），但在复杂临床任务中缓解偏见的效果有限（FLDD < 3%），远不如简单的提示策略（FLDD 8-15%）。

**[CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)**

:   提出 CARE 框架——将医学 VQA 拆分为"实体提议→指称分割→证据引导问答"三阶段专家管道，用 RLVR 微调各 VLM，并引入 GPT-5 作为动态协调器进行工具规划与 CoT 审查，在 4 个医学 VQA 基准上以 10B 参数量（77.54% 平均准确率）超越 32B 端到端 SOTA（72.29%）。

**[Causal Interpretation of Neural Network Computations with Contribution Decomposition](medical_imaging/causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)**

:   提出 CODEC（Contribution Decomposition），用 Integrated Gradients 计算隐藏层神经元对输出的贡献（而非仅分析激活），再用 Sparse Autoencoder 将贡献分解为稀疏模式（modes），实现比激活分析更强的因果可解释性和网络控制能力，并成功应用于 ResNet-50 和视网膜生物神经网络模型。

**[Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](medical_imaging/characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i.md)**

:   提出将人类概念产生过程建模为 Transformer 嵌入空间中的累积轨迹，定义 5 个运动学指标（距离、速度、加速度、熵、质心距离），在 4 个数据集（3 种语言、神经退行性疾病/脏话流畅性/属性列举）上成功区分临床组和概念类别，且不同嵌入模型产生高度一致的结果。

**[COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)**

:   COMPASS 通过在分割网络的中间特征空间沿**对目标度量最敏感的低维子空间**进行线性扰动来构建 conformal prediction 区间，在四个医学分割任务上实现了比传统 CP 方法显著更窄的预测区间，同时保持有效覆盖率。

**[ConfHit: Conformal Generative Design with Oracle Free Guarantees](medical_imaging/confhit_conformal_generative_design_with_oracle_free_guarantees.md)**

:   提出 ConfHit 框架，利用密度比加权的共形排列 p 值实现"认证"（判断生成批次是否包含 hit）和"设计"（精简候选集同时保持统计保证），在无需实验验证 oracle 和存在分布偏移的条件下，为生成式分子设计提供有限样本 $1-\alpha$ 覆盖保证。

**[Controllable Sequence Editing for Biological and Clinical Trajectories](medical_imaging/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)**

:   提出 Clef，一个基于"时间概念"（temporal concepts）的可控序列编辑模型，能够在给定条件（如药物、手术）下对生物/临床多变量轨迹进行即时和延迟编辑，在细胞重编程和患者实验室检测数据上，即时编辑 MAE 提升 16.28%，延迟编辑提升 26.73%，零样本反事实生成提升达 62.84%。

**[Controlling Repetition in Protein Language Models](medical_imaging/controlling_repetition_in_protein_language_models.md)**

:   首次系统性研究蛋白质语言模型（PLM）中的病态重复问题，提出统一的重复度量指标 $R(x)$ 和效用指标 $U(x)$，并设计 UCCS（Utility-Controlled Contrastive Steering）方法，通过在隐层注入与重复解耦的引导向量，在不重训模型的前提下有效抑制重复同时保持折叠可信度。

**[CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints](medical_imaging/cryonetrefine_a_one-step_diffusion_model_for_rapid_refinement_of_structural_mode.md)**

:   提出CryoNet.Refine——首个基于AI的冷冻电镜(cryo-EM)原子模型精修框架：设计单步扩散模型(初始化自Boltz-2权重)→创新可微分密度生成器(物理模拟合成密度图)→首次将密度图相关性作为可微损失函数(余弦相似度)→联合Ramachandran/Rotamer/键角等几何约束损失→测试时优化策略逐案定制→在120个蛋白质/DNA-RNA复合物上全面超越Phenix.real_space_refine(CC_mask 0.59 vs 0.54, Ramachandran favored 98.92%)。

**[Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)**

:   提出 TeCh 框架，核心是用 CoTAR（Core Token Aggregation-Redistribution）模块替代 Transformer 中的标准注意力来建模医学时间序列的通道依赖——通过引入全局"核心 token"充当代理，先聚合所有通道信息再重分配回每个通道，复杂度从 $O(n^2)$ 降至 $O(n)$，在 APAVA 数据集上精度 86.86%（超 Medformer 12.13%），内存仅 33%、推理时间仅 20%。

**[Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](medical_imaging/deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)**

:   提出嵌套子空间网络（NSN），通过低秩分解使线性层形成严格嵌套的子空间层次，配合不确定性感知多秩训练，使单个模型在测试时可即时调节计算量与性能的权衡（50% FLOPs 减少仅损失 5% 精度），且可后验应用于预训练 LLM。

**[DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](medical_imaging/disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)**

:   将密集重叠细胞实例分割建模为图着色问题，提出"显式标记冲突节点 + 隐式邻接约束消歧"的分治框架 Disco，通过 BFS 分解细胞邻接图并引入五种协同损失函数，在高密度病理数据集 GBC-FS 2025 上 PQ 提升 7.08%，同时在四个异质数据集上均取得 SOTA。

**[Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](medical_imaging/discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)**

:   提出 SDPO（Stepwise Decomposition Preference Optimization），将离散扩散模型的轨迹对齐问题分解为逐步后验对齐子问题，避免了在整条去噪链上反传梯度的困难，在 DNA 序列设计、蛋白质逆折叠和语言建模三个任务上均显著超越现有方法。

**[DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](medical_imaging/distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)**

:   提出 DistMLIP 分布式推理平台，基于零冗余图级并行化策略（graph-level parallelization），解决现有机器学习原子间势（MLIP）缺乏多 GPU 支持的问题，在 8 GPU 上实现接近百万原子的模拟，比空间分区方法快达 8 倍且能模拟 3.4 倍更大的系统。

**[Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems](medical_imaging/distributional_consistency_loss_beyond_pointwise_data_terms_in_inverse_problems.md)**

:   提出分布一致性（DC）损失，用分布级别的校准替代传统逐点数据保真项（如MSE/NLL），避免对噪声的过拟合，在DIP去噪和PET图像重建中显著提升性能且无需早停。

**[DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](medical_imaging/dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)**

:   提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。

**[DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](medical_imaging/driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)**

:   DriftLite 提出在 Fokker-Planck 方程中利用漂移-势函数的自由度，通过轻量级线性系统求解最优控制漂移来主动稳定粒子权重，以最小代价解决 Sequential Monte Carlo 中的权重退化问题，在高斯混合、分子系统和蛋白质-配体共折叠任务上大幅超越 Guidance-SMC 基线。

**[Dual Distillation for Few-Shot Anomaly Detection](medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)**

:   提出双蒸馏框架 D24FAD，结合 query 图像上的教师-学生蒸馏（TSD）和 support 图像上的学生自蒸馏（SSD），辅以学习权重机制（L2W）自适应评估 support 重要性，在 APTOS 眼底数据集上仅用 2-shot 达到 100% AUROC。

**[EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases](medical_imaging/emr-agent_automating_cohort_and_feature_extraction_from_emr_databases.md)**

:   提出EMR-AGENT，首个基于LLM Agent的电子病历（EMR）自动化预处理框架，通过动态SQL交互替代手工规则编写，实现跨数据库的队列选择、特征提取和代码映射，在MIMIC-III/eICU/SICdb上表现优异并具强泛化能力。

**[EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](medical_imaging/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)**

:   EvoFlows 提出一种基于编辑操作的 Flow Matching 方法，通过学习进化相关蛋白质序列间的突变轨迹，能在模板序列上执行可控数量的突变（插入、删除、替换），同时预测"突变什么"和"在哪里突变"。

**[Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation](medical_imaging/exo-plore_exploring_exoskeleton_control_space_through_human-aligned_simulation.md)**

:   提出 Exo-plore 框架，通过神经力学仿真与深度强化学习相结合，无需真人实验即可优化髋关节外骨骼控制参数，并能推广到病理步态场景。

**[ExpGuard: LLM Content Moderation in Specialized Domains](medical_imaging/expguard_llm_content_moderation_in_specialized_domains.md)**

:   提出面向金融、医疗、法律等专业领域的安全护栏模型 ExpGuard 及配套数据集 ExpGuardMix（58,928 样本），在领域特定测试集上 prompt 分类 F1 超 WildGuard 8.9%、response 分类超 15.3%，同时在通用安全基准上保持 SOTA 水平。

**[Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification](medical_imaging/exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c.md)**

:   发现病理基础模型特征具有低维流形几何结构（有效秩仅29.7/512维），而线性层会破坏这种结构导致少样本过拟合，提出即插即用的MR Block（冻结随机矩阵做几何锚+低秩残差路径做任务适配）在少样本WSI分类上达到SOTA。

**[Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](medical_imaging/extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)**

:   挑战基因表达预测中"越长越好"的长序列建模范式，发现当前 SSM 模型本质上只利用近端信息；进而识别出背景染色质信号（DNase-seq/Hi-C）作为混杂变量引入虚假关联，提出 Prism 框架通过后门调整去混杂，仅用 2k 短序列即超越 200k 长序列的 SOTA。

**[Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](medical_imaging/fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)**

:   统一拒绝采样微调方法为GRAFT框架并证明其隐式执行KL正则化奖励最大化，进而提出P-GRAFT在中间去噪步骤做分布整形（偏差-方差权衡更优），以及Inverse Noise Correction无需奖励即可改进流模型质量，在T2I上VQAScore提升8.81%。

**[From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](medical_imaging/from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)**

:   提出EHR-ChatQA基准，首次评估数据库Agent在电子病历场景中的端到端交互工作流（澄清模糊查询→解决术语不匹配→生成SQL→返回答案），发现最强模型(o4-mini)的Pass@5超90%但Pass∧5(全部成功)大幅下降(差距达60%)，暴露了安全关键领域的鲁棒性缺陷。

**[Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology](medical_imaging/fusing_pixels_and_genes_spatially-aware_learning_in_computational_pathology.md)**

:   本文提出 Stamp 框架，利用空间转录组学基因表达数据作为监督信号，通过空间感知基因编码器预训练和层次化多尺度对比对齐，实现病理图像与空间转录组数据的联合表示学习，在 6 个数据集 4 个下游任务上取得 SOTA。

**[Glance and Focus Reinforcement for Pan-cancer Screening](medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)**

:   提出 GF-Screen 两阶段框架——轻量 Glance 模型用强化学习快速定位含病灶的 CT 子体积，Focus 模型只对选中区域做精细分割；通过将 GRPO 的"组内相对比较"思想从 NLP 迁移到视觉子体积组，首次在纯视觉任务中实现无价值网络的 RL 优化，在 FLARE25 泛癌挑战中以 +25.6% DSC 大幅领先冠军方案且推理快 5.7 倍。

**[HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](medical_imaging/histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)**

:   本文提出 HistoPrism，一个高效的 Transformer 架构，通过交叉注意力注入癌症类型条件来从 H&E 病理图像预测泛癌基因表达，并提出基于 Hallmark/GO 通路的 Gene Pathway Coherence (GPC) 评估框架，在通路级别预测上大幅超越 STPath，尤其在低方差核心生物通路上优势显著。

**[How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images](medical_imaging/how_do_medical_mllms_fail_a_study_on_visual_grounding_in_medical_images.md)**

:   首次系统性诊断出医学MLLM在零样本医学VQA上表现差的根本原因在于视觉扎根（visual grounding）不足——模型注意力系统性地偏离临床相关区域，由此提出无训练的VGRefine推理时注意力修正方法，在6个基准的110K+样本、8种成像模态上均达到SOTA。

**[How to Make the Most of Your Masked Language Model for Protein Engineering](medical_imaging/how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)**

:   提出基于温度退火随机束搜索（SBS）的MLM采样方法，利用伪似然的野生型边际近似实现高效全序列评估，在真实抗体治疗优化的体外实验中证明采样算法选择至少与模型选择同等重要，SBS+引导达到100%成功率。

**[Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](medical_imaging/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)**

:   构建 Human Behavior Atlas——首个覆盖情感、认知、病理和社会过程四大维度的大规模多模态行为理解统一基准（101K+ 样本），并训练三种 OmniSapiens-7B 模型变体验证其在多任务训练和迁移学习中的有效性。

**[Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](medical_imaging/improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)**

:   提出 Inter-Slice Consistent Stochasticity (ISCS)，通过球面线性插值(Slerp)在扩散采样的 re-noising 步骤中生成层间相关噪声，从根源消除 2D 扩散先验做 3D 医学重建时的层间不连续伪影——零额外计算/超参数/训练开销，即插即用到任何 2D 扩散逆问题求解器，在稀疏视角 CT、限角 CT 和 MRI 超分辨率上均持续提升。

**[Incentives in Federated Learning with Heterogeneous Agents](medical_imaging/incentives_in_federated_learning_with_heterogeneous_agents.md)**

:   从博弈论视角分析异构联邦学习中的激励问题，证明在异构数据分布和 PAC 准确率目标下纯策略纳什均衡的存在性，并提出基于线性规划的近似算法来确定最优贡献量。

**[Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)**

:   提出DyMo——推理时动态模态选择框架，通过理论推导将多模态任务相关信息增益转化为可计算的MTIR奖励函数（基于分类损失降低代理 + 类原型距离 + 类内相似性校准），在推理时迭代选择性融合可靠的恢复模态，首次系统性解决"丢弃缺失模态损失信息 vs 补全可能引入噪声"的困境。

**[Intrinsic Lorentz Neural Network](medical_imaging/intrinsic_lorentz_neural_network.md)**

:   提出完全内禀（fully intrinsic）的双曲神经网络 ILNN，所有运算均在 Lorentz 模型内完成，消除了现有方法中混合欧几里得操作的几何不一致性，在图像分类、基因组学和图分类上取得 SOTA。

**[Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](medical_imaging/knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)**

:   提出 LEON（LLM-based Entropy-guided Optimization with kNowledgeable priors），一种数学原理严格的方法，将个性化医疗治疗方案设计建模为条件黑箱优化问题，通过熵约束和对抗性源批评模型引导 LLM 在不微调的情况下作为零样本优化器提出个性化治疗计划。

**[Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration](medical_imaging/learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im.md)**

:   提出首个多域全能图像复原方法DATPRL-IR，通过双提示池（任务提示池+域提示池）学习域感知的任务提示表征，利用MLLM蒸馏域先验并通过自适应门控融合指导复原，在自然/医学/遥感三域9任务上显著超越SOTA。

**[Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](medical_imaging/learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)**

:   提出 Δ-LFM 框架：用 ArcRank 损失在潜在空间构建患者特异性时间对齐轨迹（角度一致 + 幅度单调递增），将流匹配时间范围从 [0,1] 扩展到 [0,T] 实际时间间隔实现任意时间点预测，在三个阿尔茨海默纵向 MRI 基准上全面超越 8 种基线方法，并提出进展专用指标 Δ-RMAE。

**[mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](medical_imaging/mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)**

:   提出 mCLM（模块化化学语言模型），通过将分子表示为可合成构建模块的序列，使 LLM 能生成同时满足药理功能和自动化合成可行性的分子，在 430 种 FDA 批准药物上显著改善了药代动力学和毒性性质。

**[MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medical_imaging/medagentgym_agentic_training_biomedical.md)**

:   构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（覆盖 12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，系统基准评估 29 个 LLM 揭示商业/开源差距，并通过高效多线程轨迹采样 + 离线/在线 RL 训练出 Med-Copilot，分别获得 +43.02%/+45.28% 提升，达到与 GPT-4o 竞争的性能。

**[MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](medical_imaging/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)**

:   提出 MMedAgent-RL，通过 RL 优化模拟临床会诊流程（分诊→专科→主治）的多智能体系统，核心创新是课程学习引导的熵感知 RL（C-MARL），让主治医师智能体在面对正确/冲突/错误的专科意见时分别采取不同的探索-利用策略，在域内外共 5 个医学 VQA 基准上实现 SOTA。

**[Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](medical_imaging/moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)**

:   提出MENTAT——由9名美国精神科医生设计和标注的评估数据集（203道基础题×人口统计变量扩展），覆盖诊断/治疗/分诊/监测/文档5个临床实践领域，通过系统性替换患者年龄/种族/性别评估22个语言模型的决策偏见，发现模型在各人口统计维度上存在显著且不可预测的准确率差异。

**[Neuro-Symbolic Decoding of Neural Activity](medical_imaging/neuro-symbolic_decoding_of_neural_activity.md)**

:   提出 NEURONA，一个神经符号框架用于 fMRI 解码和概念基础，通过将视觉场景分解为符号程序（概念的逻辑组合），在 fMRI 问答任务上显著优于端到端神经解码和线性模型。

**[Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](medical_imaging/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)**

:   本文构建了 Omni-iEEG 数据集（302 名患者、178 小时高分辨率颅内脑电记录），定义了基于临床先验的标准化基准任务和评估指标，并展示端到端建模在癫痫手术规划中可匹配或超越传统生物标志物方法。

**[Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling](medical_imaging/overthinking_reduction_with_decoupled_rewards_and_curriculum_data_scheduling.md)**

:   从理论上揭示了现有长度惩罚方法的两个根本缺陷——错误惩罚高熵探索token和错误奖励冗余token，提出 DeCS 框架，通过解耦token级奖励和课程批次调度，在7个基准上将推理token减少50%以上同时保持甚至提升模型性能。

**[Protein as a Second Language for LLMs](medical_imaging/protein_as_a_second_language_for_llms.md)**

:   将氨基酸序列视为 LLM 的"第二语言"，通过构建蛋白质-自然语言双语数据集和自适应上下文构造机制，无需任何训练即可让通用 LLM 在蛋白质问答任务上平均提升 7% ROUGE-L，最高 17.2%，甚至超越领域专用微调模型。

**[Protein Counterfactuals via Diffusion-Guided Latent Optimization](medical_imaging/protein_counterfactuals_via_diffusion-guided_latent_optimization.md)**

:   提出MCCOP框架，在蛋白质的连续序列-结构联合潜空间中，利用预训练扩散模型作为流形先验进行梯度引导的反事实优化，以最少突变（2-3个）生成生物学可信的蛋白质变体来翻转预测器输出，同时实现模型解释和蛋白质设计假说生成。

**[Protein Structure Tokenization via Geometric Byte Pair Encoding](medical_imaging/protein_structure_tokenization_via_geometric_byte_pair_encoding.md)**

:   提出 GeoBPE——首个将 BPE（字节对编码）从离散文本扩展到连续蛋白质骨架几何的 tokenizer，通过交替执行"局部合并（k-medoids聚类+量化）"和"全局校正（可微逆运动学）"构建层次化结构 motif 词汇表，以 >10× 压缩比和 >10× 数据效率超越 VQ-VAE 类 PST，在 12 个下游任务 24 个测试集上排名第一。

**[Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](medical_imaging/q-fsru_quantum-augmented_frequency-spectral_for_medical_visual_question_answerin.md)**

:   提出 Q-FSRU 框架，通过 FFT 将医学图像和文本特征变换到频率域进行融合，并引入量子启发的检索增强机制（Quantum RAG）从外部知识库中获取医学事实，在 VQA-RAD 数据集上取得 90.0% 准确率。

**[Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](medical_imaging/resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)**

:   提出 Resp-Agent 闭环多智能体框架，通过主动对抗课程规划器（Thinker-A2CA）协调可控呼吸音生成器与多模态诊断器，在 229k 规模基准上实现生成↔诊断协同设计，大幅提升长尾类别诊断性能。

**[Reverse Distillation: Consistently Scaling Protein Language Model Representations](medical_imaging/reverse_distillation_consistently_scaling_protein_language_model_representations.md)**

:   针对蛋白质语言模型（PLM）"模型越大性能不一定越好"的反常缩放现象，提出反向蒸馏框架：以小模型表示为基底、用SVD提取大模型正交残差信息，构造Matryoshka嵌套嵌入，使得更大的反向蒸馏模型一致优于更小的，ESM-2 15B经反向蒸馏后首次成为全家族最强。

**[Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](medical_imaging/scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)**

:   提出 STAR-MD，一个 SE(3) 等变的因果扩散 Transformer，通过联合时空注意力和上下文噪声扰动实现微秒级蛋白质动力学轨迹生成，在 ATLAS 基准上所有指标达到 SOTA，且能稳定外推到训练中未见的微秒时间尺度。

**[Scaling with Collapse: Efficient and Predictable Training of LLM Families](medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)**

:   证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

**[Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People](medical_imaging/shoot_first_ask_questions_later_building_rational_agents_that_explore_and_act_li.md)**

:   提出 Collaborative Battleship 任务评估语言模型的信息搜索能力，设计三种贝叶斯推断策略（Bayes-Q/M/D）增强 LM 的提问、行动和决策能力，使弱模型（Llama-4-Scout）以 GPT-5 约 1% 的成本达到超人表现（82% 胜率）。

**[SONIC: Spectral Oriented Neural Invariant Convolutions](medical_imaging/sonic_spectral_oriented_neural_invariant_convolutions.md)**

:   SONIC 将状态空间模型的思想迁移到多维频域，用 6 个连续参数（幅度、方向、阻尼、振荡等）定义一组方向选择性的频谱传递函数，再通过低秩矩阵 $B$、$C$ 跨通道混合，实现天然具备全局感受野和分辨率不变性的卷积替代算子，在 3D 医学分割上匹配 nnU-Net 且参数少近两个数量级，在 ImageNet 上也具有竞争力。

**[SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis](medical_imaging/survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi.md)**

:   提出 SurvHTE-Bench，首个面向右删失生存数据的异质处理效应（HTE）估计综合基准，涵盖 40 个合成数据集、10 个半合成数据集和 2 个真实数据集，系统评估了 53 种估计方法在不同因果假设违反和删失水平下的表现，发现没有单一方法占主导地位，生存 meta-learner（特别是 S-Learner-Survival 和 Matching-Survival）在高删失和假设违反场景下表现最为稳健。

**[SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](medical_imaging/syncogen_synthesizable_3d_molecule_generation_via_joint_reaction_and_coordinate_.md)**

:   SynCoGen 提出了一种结合掩码图扩散和流匹配的多模态生成框架，能够同时采样分子构建块反应图和3D原子坐标，在保证合成可行性的同时实现高质量的3D分子生成。

**[Thompson Sampling via Fine-Tuning of LLMs](medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)**

:   提出 ToSFiT，通过微调大语言模型直接参数化最大概率（Probability of Maximality），将 Thompson Sampling 扩展到大规模非结构化离散空间，避免了获取函数最大化的难题。

**[Towards Interpretable Visual Decoding with Attention to Brain Representations](medical_imaging/towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)**

:   提出 NeuroAdapter，将 fMRI 信号按脑区分割为独立 token 并通过交叉注意力直接条件化 Stable Diffusion，跳过传统的 CLIP/DINO 中间嵌入空间，在 NSD 等数据集上高层语义指标超越或持平现有方法，同时引入 IBBI 双向可解释性框架，首次动态揭示不同皮层区域在去噪轨迹中如何驱动图像生成。

**[Tracing Pharmacological Knowledge in Large Language Models](medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)**

:   首次系统性地对生物医学 LLM 中药物分组语义的编码机制进行因果分析，发现药物组知识存储在早期层、分布在多个 token 上（非最后一个 token），线性可分的语义信息在嵌入层即已存在。

**[Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](medical_imaging/ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)**

:   提出 DiDi-Instruct，一种基于积分 KL 散度 (IKL) 最小化的蒸馏框架，将预训练的扩散大语言模型 (dLLM) 蒸馏为少步学生模型，通过对抗性密度比估计 + 分组奖励归一化 + 分数分解 + 奖励引导祖先采样器 (RGAS) 四大关键设计，在 OpenWebText 上仅用 16 步即超越 1024 步教师模型的 PPL，实现最高 64× 推理加速，同时训练成本仅需 1 GPU 小时。

**[Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](medical_imaging/unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)**

:   PVB（Pretrained Variational Bridge）通过编码器-解码器架构结合增强桥匹配，统一了单结构预训练和配对轨迹微调的训练目标，实现了跨领域生物分子轨迹生成，并通过RL微调加速蛋白质-配体holo态探索。

**[VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](medical_imaging/vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)**

:   提出 VLM-SubtleBench，一个评估视觉语言模型在细微差异比较推理能力的基准，覆盖 10 种差异类型和 6 个图像领域（自然、游戏、工业、航空、医学、合成），揭示了 VLM 与人类在空间/时间/视角推理上超过 30% 的性能差距。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)**

:   提出 A-TPT 框架，通过最大化归一化文本特征在单位超球面上的最小成对角距离来促进角度多样性，解决测试时提示调优 (TPT) 中 VLM 预测过度自信导致的校准不良问题，在自然分布偏移和医学数据集上均优于现有 TPT 校准方法。

**[AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](multimodal_vlm/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)**

:   通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner，在 9 个 benchmark 上表现稳健。

**[BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models](multimodal_vlm/biocap_exploiting_synthetic_captions_beyond_labels_in_biological_foundation_mode.md)**

:   提出 BioCAP，通过用 MLLM 生成 wiki 知识引导的合成描述性 caption（而非仅用物种标签）来训练生物学多模态基础模型，在 10 个物种分类 benchmark 上比 BioCLIP 平均提升 8.8%，在文本-图像检索任务上提升 21.3%。

**[Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](multimodal_vlm/bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)**

:   构建 Bongard-RWR+，一个包含 5400 个 Bongard 问题的 benchmark，使用 VLM 流水线（Pixtral-12B + Flux.1-dev）自动生成真实感图像来表示抽象概念，系统评估揭示 SOTA VLM 在辨别细粒度视觉概念（如轮廓、旋转、角度）时表现挣扎，准确率低至 19%。

**[Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](multimodal_vlm/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)**

:   提出 WS-COC，首个基于 MLLM 的弱监督类无关目标计数框架，通过分而治之的对话微调（逐步缩小计数范围）、比较排序优化（学习图像间相对计数关系）和全局-局部计数增强三个策略，仅用图像级计数标注即可匹敌甚至超越全监督方法。

**[Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)**

:   本文提出 TuneCLIP，一个自监督微调（SSFT）框架，通过两阶段设计——先恢复优化器统计量（OSR）消除冷启动偏差，再用带margin的铰链全局对比损失（HGCL）缓解假负样本过度惩罚——在不使用任何标签的条件下持续提升已有开源 CLIP 模型的通用性能，在 ImageNet 及变体上提升最高 +2.5%，在 DataComp 基准上提升 +1.2%。

**[Can Vision-Language Models Answer Face to Face Questions in the Real-World?](multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)**

:   提出 QIVD（Qualcomm Interactive Video Dataset），一个面对面实时问答 benchmark（2900 个视频+音频+时间戳标注），揭示现有 VLM 在实时情境理解上远落后人类（最佳模型 60% vs 人类 87%），主要瓶颈在指代消歧、回答时机判断和情境常识，微调可显著缩小差距。

**[Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](multimodal_vlm/can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)**

:   提出 AesEval-Bench，首个系统性评估 VLM 图形设计美学评估能力的 benchmark（4维度×12指标×3任务），发现现有 VLM（含推理增强型）在设计美学上表现有限，并通过 human-guided VLM labeling + indicator-grounded reasoning 构建训练数据，微调 7B 模型在精确定位任务上超过 GPT-5。

**[Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)**

:   针对 MoE 推理时因 token 分配不均导致的 Straggler Effect（最重负载专家决定整体延迟），提出 Capacity-Aware Token Drop（丢弃过载专家的低分 token）和 Expanded Drop（将溢出 token 重路由到本地低负载专家），在 Mixtral-8×7B 上实现 1.85× 加速且性能提升 0.2%。

**[Chart Deep Research in LVLMs via Parallel Relative Policy Optimization](multimodal_vlm/chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)**

:   提出 PRPO（Parallel Relative Policy Optimization），通过在奖励维度和数据类型两个层面做并行解耦优化，解决 GRPO 在多维奖励信号干扰和异构数据梯度冲突下的训练瓶颈；同时构建 MCDR-Bench，基于"错误唯一性原则"将主观生成评估转化为客观错误识别，实现图表深度研究能力的量化评估。

**[CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](multimodal_vlm/citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)**

:   构建 CityLens——迄今最大规模的城市社会经济感知 benchmark（17 城市、6 大领域、11 个预测任务），评估 17 个 LVLM 在直接预测、归一化估计、特征回归三种范式下从卫星/街景图像推断社会经济指标的能力，发现通用 LVLM 在多数任务上仍不及领域特化的对比学习方法。

**[Closing the Modality Gap Aligns Group-Wise Semantics](multimodal_vlm/closing_the_modality_gap_aligns_group-wise_semantics.md)**

:   证明 CLIP 中的 modality gap 对实例级任务（检索）无关紧要但严重损害群组级任务（聚类），并提出由 Align True Pairs loss + Centroid Uniformity loss 组成的新目标函数，在双模态和三模态设置中将 gap 几乎降为零，大幅提升聚类 V-Measure（+10-17 分），同时保持检索性能。

**[Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](multimodal_vlm/contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)**

:   提出多模态语义扰动框架检测VLM数据污染：用LLM生成密集描述 + Flux ControlNet在保持图像构图的同时改变答案相关语义元素，污染模型因记忆原始图文对而在扰动版本上表现骤降，干净模型则因真正推理能力而不受影响。首次系统验证现有LLM污染检测方法在VLM场景下大多不可靠。

**[Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach](multimodal_vlm/customizing_visual_emotion_evaluation_for_mllms_an_open-vocabulary_multifaceted_.md)**

:   提出情感陈述判断（ESJ）任务与 INSETS 自动标注流水线，将视觉情感评估从"开放式分类"重构为"陈述真伪判断"，构建了 MVEI benchmark（3,086 样本、424 种情感标签、四个认知维度），系统评估 19 个 MLLMs，发现即使 GPT-4o 也与人类（91.6%）存在 13.3% 的准确率差距。

**[Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](multimodal_vlm/detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)**

:   提出 EUQ（Evidential Uncertainty Quantification），基于 Dempster-Shafer 证据理论将 LVLM 的认识不确定性分解为**冲突 CF**（内部矛盾）和**无知 IG**（信息缺失），无需训练、单次前向传播即可检测幻觉/越狱/对抗/OOD 四类错误行为，平均 AUROC 相对最佳基线提升 10.4%/7.5%。

**[Directional Embedding Smoothing for Robust Vision Language Models](multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)**

:   将 RESTA（Randomized Embedding Smoothing and Token Aggregation）防御方法从 LLM 扩展到 VLM，发现方向性嵌入噪声（directional noise）在安全-实用性权衡上显著优于各向同性噪声（isotropic noise），可作为推理时的轻量防御层抵御多模态越狱攻击。

**[DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](multimodal_vlm/diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)**

:   提出 DIVA-GRPO，通过动态评估问题难度、自适应生成不同难度的语义一致变体、并结合难度加权的局部-全局 advantage 估计，解决 GRPO 训练中的 reward sparsity 和 advantage vanishing 问题，在 7B 规模模型上实现 SOTA 多模态推理性能。

**[Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](multimodal_vlm/do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)**

:   本文基于 Nissenbaum 的上下文完整性（Contextual Integrity）理论构建了 VLM-GEOPRIVACY 基准，通过7个层次递进的上下文感知问题和三级位置披露粒度（拒绝/城市级/精确位置），系统评估14个主流VLM是否能根据图像中的社会规范线索判断适当的位置信息披露级别，结果发现所有模型均严重偏向过度披露（Over-Disclosure率高达46-52%），且恶意提示可将抽象违反率推至100%。

**[Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)**

:   提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。

**[Enhanced Continual Learning of Vision-Language Models with Model Fusion](multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)**

:   提出Continual Decoupling-Unifying（ConDU）框架，首次将模型融合引入VLM持续学习，通过维护统一模型并结合任务触发器进行解耦-统一迭代操作，在MTIL基准上平均性能超SOTA 2%，同时增强了零样本能力。

**[Enhancing Multi-Image Understanding through Delimiter Token Scaling](multimodal_vlm/enhancing_multi-image_understanding_through_delimiter_token_scaling.md)**

:   通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。

**[Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models](multimodal_vlm/error_notebook-guided_training-free_part_retrieval_in_3d_cad_assemblies_via_visi.md)**

:   提出一种无训练的两阶段VLM框架，通过Error Notebook记录纠正后的推理轨迹并结合RAG进行推理时适应，在3D CAD装配体的规格驱动零件检索任务上，GPT-4o准确率从41.7%提升至65.1%（+23.4%），并通过语法约束验证器进一步提升4.5%。

**[FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)**

:   提出 FRIEDA 基准，系统评估大型视觉语言模型在多步骤、跨地图的制图推理能力，发现最强模型 Gemini-2.5-Pro 准确率仅 38.20%，远低于人类 84.87%。

**[Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs](multimodal_vlm/grasp_any_region_towards_precise_contextual_pixel_understanding_for_multimodal_l.md)**

:   提出 GAR（Grasp Any Region），通过 RoI-aligned feature replay 在保持全局上下文的同时提取高保真局部特征，实现精准的单区域描述、多区域交互建模和复合推理，1B 模型即超越 InternVL3-78B。

**[Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](multimodal_vlm/grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)**

:   将空间定位（referring + grounding）与图像质量评估结合，构建 GIQA-160K 数据集训练多模态 LLM 生成带有边界框的质量描述和空间 VQA，在细粒度质量感知上显著优于通用 MLLM。

**[GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](multimodal_vlm/gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)**

:   提出 GTR-Bench，一个面向大规模摄像头网络中移动目标地理时空推理的新基准，评估发现最强模型 Gemini-2.5-Pro（34.9%）远落后于人类水平（78.61%），揭示了当前 VLM 在时空上下文利用失衡、时序预测能力弱、地图-视频对齐能力不足三大缺陷。

**[HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](multimodal_vlm/hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)**

:   提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

**[ICYM2I: The Illusion of Multimodal Informativeness under Missingness](multimodal_vlm/icym2i_the_illusion_of_multimodal_informativeness_under_missingness.md)**

:   揭示了多模态学习中被忽视的问题：模态缺失（missingness）导致的分布偏移会使模态价值评估产生严重偏差，提出 ICYM2I 框架通过双重逆概率加权（IPW）纠正训练和评估中的偏差，在 MAR 假设下实现对模态预测效用和信息论价值的无偏估计。

**[Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)**

:   在 VLM 视觉编码器之前插入一个仅 203K 参数的二值 patch 分类器剔除文档背景 token，再用 $3 \times 3$ max-pooling 恢复碎片化文本区域并保留原始空间索引，在 Qwen2.5-VL 上实现 40-60% FLOPs 缩减且精度损失不超过 ~5%p。

**[IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](multimodal_vlm/ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)**

:   揭示了LVLM中RoPE位置编码隐式建立的视觉坐标系统（IVC tokens），提出一种训练免的、提示感知的视觉token剪枝策略，在保留IVC tokens和语义前景token的同时，削减约50%视觉token并维持≥99%原始性能。

**[KeepLoRA: Continual Learning with Residual Gradient Adaptation](multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)**

:   通过分析预训练模型权重的SVD分解，发现通用知识编码在主子空间、领域特定知识编码在残差子空间，提出KeepLoRA方法将新任务的LoRA更新约束在残差子空间中，同时用梯度信息初始化以保持可塑性，在持续学习中达到前向稳定、后向稳定和可塑性的最优平衡。

**[LiveWeb-IE: A Benchmark For Online Web Information Extraction](multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)**

:   提出首个面向在线网页的信息抽取（WIE）基准LiveWeb-IE，覆盖文本/图片/超链接等多类数据抽取，并设计Visual Grounding Scraper（VGS）框架，通过模拟人类认知过程——视觉扫描定位区域→精确定位元素→生成XPath——在动态网页上实现鲁棒的信息抽取。

**[LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models](multimodal_vlm/llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)**

:   提出 LLaVA-FA，一种在频域进行联合低秩加量化权重近似的高效多模态大模型压缩方法，利用傅里叶变换的去相关性和共轭对称性实现更紧凑准确的权重表示，并引入 PolarQuant（极坐标量化）和 ODC（可选对角校准）方案，在多个基准上以最少的激活参数和计算成本超越现有高效多模态模型。

**[Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](multimodal_vlm/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)**

:   提出 MAPD（Meta-Adaptive Prompt Distillation），一种基于 MAML 元学习的提示蒸馏方法，通过注意力映射器从任务相关的图像特征中蒸馏软提示，使 LMM 在测试时仅用少量梯度步即可适应新的视觉问答任务，性能超越 ICL 21.2%。

**[Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](multimodal_vlm/mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)**

:   发现LVLM中KV Cache存在模态特异和注意力头特异的语义冗余，仅靠重要性选择会丢失语义覆盖，提出MixKV按头自适应混合重要性与多样性分数进行KV Cache压缩，在极端压缩下平均提升5.1%。

**[MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](multimodal_vlm/mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)**

:   提出 MMR-Life 基准（2646 道 5 选 1 多图选择题，基于 19108 张真实图像，覆盖 7 种推理类型和 21 个任务），首次系统评估 MLLM 在真实生活场景中的多图推理能力，发现最强模型 GPT-5 仅 58.69% 准确率，距人类水平差 14%，并揭示了推理增强方法在大模型上失效、RL 泛化弱于 BoN 等关键发现。

**[MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](multimodal_vlm/mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)**

:   提出MMTok——一种基于最大覆盖问题（Maximum Coverage Problem）的多模态视觉token选择框架，同时利用文本-视觉和视觉-视觉覆盖信息来选择最具信息量的视觉token子集，在training-free设置下显著优于单模态baseline，甚至超越需要微调的方法。

**[Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?](multimodal_vlm/modal_aphasia_can_unified_multimodal_models_describe_images_from_memory.md)**

:   本文发现并系统定义"模态失语"（Modal Aphasia）现象——统一多模态模型能从记忆中近乎完美地生成视觉概念（如电影海报图像），但在文字描述同一概念时错误率高出 7 倍以上，且严重幻觉几乎只出现在文本模态；通过前沿模型（ChatGPT-5）的真实实验和开源模型（Janus-Pro、Harmon）的合成控制实验，证实模态失语是当前统一架构的系统性缺陷而非训练偶然，并展示了该现象对 AI 安全框架的潜在威胁。

**[Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)**

:   大规模实证研究揭示23个VQA基准中存在严重的单模态依赖问题——许多为消除文本偏差而设计的基准反而引入了图像偏差，模型利用单模态捷径而非真正的跨模态推理。

**[Multimodal Classification via Total Correlation Maximization](multimodal_vlm/multimodal_classification_via_total_correlation_maximization.md)**

:   从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。

**[Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](multimodal_vlm/multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)**

:   首次将自动提示优化（APO）从纯文本空间扩展到多模态空间，提出 MPO 框架：通过对齐保持的联合探索（统一语义梯度同步驱动文本+非文本提示更新，配合 Generation/Edit/Mix 三种算子多样化搜索）和先验继承的贝叶斯 UCB 候选选择（利用父提示性能 warm-start 子提示的 Beta 先验），在图像/视频/分子共 10 个数据集上平均准确率达 65.1%，超越最强文本 APO 基线 ProTeGi 的 60.0%。

**[OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)**

:   基于认知心理学构建OmniSpatial——首个全面空间推理基准，系统覆盖动态推理、复杂空间逻辑、空间交互和透视转换4大维度50个子类别共8.4K人工标注QA对，让o3最强推理模型仅达56.33%而人类达92.63%→揭示复杂空间推理仍是VLM的核心瓶颈。

**[Post-hoc Probabilistic Vision-Language Models](multimodal_vlm/post-hoc_probabilistic_vision-language_models.md)**

:   提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。

**[PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)**

:   提出PPE（Positional Preservation Embedding），利用RoPE各维度旋转独立性，将合并token内多个原始位置ID分块编码到不同维度段中，实现单个压缩token携带多个空间/时序位置信息。PPE是零参数、即插即用的通用算子，在55%压缩率下图像任务平均仅降3.6%、在90%压缩率下通过级联压缩仍保持可比性能。

**[PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies](multimodal_vlm/prismm-bench_a_benchmark_of_peer-review_grounded_multimodal_inconsistencies.md)**

:   构建首个基于真实审稿人标记的科学论文多模态不一致性基准PRISMM-Bench，从18009条ICLR开放评审中挖掘384个跨模态不一致，设计识别/修复/配对匹配三任务并提出JSON结构化去偏答案表示，21个顶级LMM最高仅53.9%→系统性暴露当前模型在科学文档跨模态推理上的严重不足。

**[Reasoning-Driven Multimodal LLM for Domain Generalization](multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)**

:   提出 RD-MLDG——首个将 MLLM 推理链引入域泛化的框架。构建 DomainBed-Reasoning 数据集，系统分析推理监督的两大挑战（优化困难 + 推理模式不匹配），通过 MTCT（多任务交叉训练）与 SARR（自对齐推理正则化）协同解决，在 4 个标准 DG 基准上以 86.89% 的平均准确率大幅超越 GPT-4o（83.46%）和所有 CLIP/ViT 方法。

**[Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks](multimodal_vlm/ref-adv_exploring_mllm_visual_reasoning_in_referring_expression_tasks.md)**

:   提出 Ref-Adv 基准数据集，通过 **硬干扰物配对 + LLM 辅助最小充分表达式生成 + 三人一致性人工验证** 的流水线，构建了一个消除"定位捷径"的现代 REC 基准，在该基准上 13 个当代 MLLM（包括 GPT-4o、Gemini 2.5、Qwen2.5-VL-72B 等）的准确率从 RefCOCO(+/g) 上的 90%+ 大幅下降至 50-68%，系统暴露了模型在复杂视觉推理和真实定位能力上的严重不足。

**[Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)**

:   从混合专家（MoE）视角揭示 VPT 的局限性——prompt experts 是输入无关的常量函数表达力受限，提出 VAPT 通过 token-wise 投影器和共享特征投影器使 prompt experts 自适应输入，用更少参数实现更优性能，并给出了最优样本效率的理论保证。

**[Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](multimodal_vlm/seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)**

:   提出 MV-RoboBench，首个结合多视角空间推理与机器人操作任务的基准，系统评估了 40+ 个 VLM（开源+闭源+推理增强），发现最强模型 GPT-5 仅达 56.4% 准确率，远低于人类的 91.0%，并揭示空间与机器人推理正相关、单视角基准表现无法可靠迁移至多视角场景。

**[Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](multimodal_vlm/self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)**

:   提出 Self-Aug，一种免训练的解码策略，通过自增强提示（SAS Prompting）让 LVLM 利用自身知识动态选择与查询语义对齐的视觉增强方式，并提出稀疏度自适应截断（SAT）算法利用输出分布的完整熵信息动态调节候选词集大小，在5个 LVLM 和7个基准上一致超越现有对比解码方法。

**[Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](multimodal_vlm/self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)**

:   提出 EvoQuality 框架，通过成对多数投票生成伪排序标签、结合 GRPO 自迭代优化，使 VLM 在无人工标注下自主提升图像质量感知能力，零样本性能提升 31.8% PLCC，在 7 个 IQA 基准中 5 个超越有监督 SOTA。

**[Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)**

:   提出 Shuffle-R1 框架，通过 Pairwise Trajectory Sampling（选取高对比度轨迹对）和 Advantage-based Batch Shuffle（按优势值重分配训练批次），解决 RL 训练中的 Advantage Collapsing 和 Rollout Silencing 两大效率瓶颈，在 Geo3K 上比 baseline 提升 22%，MathVerse 上超越 GPT-4o。

**[Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation](multimodal_vlm/small_drafts_big_verdict_information-intensive_visual_reasoning_via_speculation.md)**

:   借鉴 Speculative Decoding 的 draft-then-verify 范式提出 Speculative Verdict (SV)，用多个轻量 VLM 生成多样推理路径作为 draft，大模型作为 verdict 综合验证并纠错，在信息密集型 VQA 上无需训练即超过 GPT-4o 达 11.9%，且能修复 47-53% 的少数正确案例。

**[SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](multimodal_vlm/sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)**

:   提出SophiaVL-R1——在规则基RL训练MLLM推理时引入整体级思维过程奖励：训练Thinking Reward Model从逻辑一致性/冗余度等五维度评估推理质量→提出Trust-GRPO基于正确/错误答案组的思维奖励对比计算可信度权重$\gamma$缓解reward hacking→退火策略$e^{-\text{steps}/T}$渐减思维奖励使后期更依赖准确的规则奖励→7B模型在MathVista(71.3%)和MMMU(61.3%)等多个基准全面超越LLaVA-OneVision-72B。

**[Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](multimodal_vlm/sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)**

:   提出Sparsity Forcing——基于GRPO的RL后训练框架，将带稀疏注意力的MLLM作为策略模型、原始MLLM作为参考模型，通过多预算rollout探索不同token保留阈值$p$，以效率(token减少率)+性能(答案正确性)为联合奖励做组内对比优化，将Qwen2/2.5-VL的token减少率从20%提升至75%且精度损失极小，实现内存降3×、解码加速3.3×。

**[Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)**

:   提出基于认知科学 2×2 分类法（内在/外在 × 静态/动态）的统一空间推理基准 Spatial-DISE，包含 559 个评估 VQA 对和 12K+ 训练数据，在 32 个 SOTA VLM 上的评测揭示了模型在动态空间推理（尤其是心理旋转和折叠）上与人类的巨大差距。

**[Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](multimodal_vlm/spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)**

:   提出 Spatial CAPTCHA，一种基于 3D 空间推理的新型人类验证框架，利用人类与多模态大语言模型在几何推理、视角变换、遮挡处理和心理旋转等任务上的根本性能力差异来区分人与机器，最优 MLLM 仅达 31.0% Pass@1 准确率，远低于人类表现。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](multimodal_vlm/spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架下控制实验，系统性地研究图像编码器训练目标和 2D 位置编码对 VLM 空间推理能力的影响，发现编码器选择主导空间性能、AIMv2 编码器一致性最好，但 2D-RoPE 的改进不稳定，空间推理的失败根植于当前 VLM 流水线的核心设计选择。

**[SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](multimodal_vlm/spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)**

:   提出SpatiaLab，一个包含1400个视觉QA对的真实场景空间推理基准，涵盖6大类30子类空间任务，支持多选和开放式双格式评估，揭示当前最强VLM（InternVL3.5-72B MCQ 54.93%）与人类（87.57%）之间存在巨大空间推理鸿沟，且开放式设置下差距更大。

**[SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)**

:   提出 SpinBench，一个以认知科学为基础的诊断性基准测试，通过 7 类渐进式空间推理任务（从物体识别到视角转换）系统评估 37 个 VLMs 的空间理解能力，揭示了模型存在的自我中心偏差、旋转理解薄弱等系统性缺陷。

**[Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](multimodal_vlm/steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)**

:   提出 SteerVAD 框架，在完全冻结的多模态大语言模型 (MLLM) 内部，通过识别"潜在异常专家"注意力头并用层次化元控制器动态操控其表示流形，仅用 1% 训练数据即实现免调优视频异常检测的 SOTA。

**[TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](multimodal_vlm/tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)**

:   提出 TableDART，通过仅 2.59M 参数的 MLP 门控网络为每个 query-table 对动态选择最优处理路径（Text-only / Image-only / Fusion），复用冻结的单模态专家模型并引入 LLM Agent 进行跨模态融合，在 7 个表格理解 benchmark 上平均超越最强 MLLM 基线 HIPPO 4.02%，同时延迟降低 24.5%。

**[ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](multimodal_vlm/thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)**

:   提出 ThinkOmni 无训练框架，利用纯文本大推理模型(LRM)在解码时引导全模态 LLM(OLLM)，通过 Stepwise Contrastive Scaling 自适应平衡感知与推理信号，MathVista 达 70.2%、MMAU 达 75.5%，匹配或超越 RFT 方法。

**[Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](multimodal_vlm/through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)**

:   提出 VC-STaR（Visual Contrastive Self-Taught Reasoner），基于"VLM 在对比两张相似图像时看得更准"的观察，设计了一套对比式自改进框架：通过构造对比 VQA 对让模型在对比中生成更忠实的视觉分析，再由 LLM 将对比分析融入推理路径，产出高质量视觉推理数据集 VisCoR-55K，微调后在 MMVP 上提升 5.7%、Hallusion 上提升 3.2%。

**[U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](multimodal_vlm/u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)**

:   系统消融MLLM嵌入学习的设计空间，揭示双向注意力+mean pooling优于主流last token、可学习温度被严重低估等关键因子，据此构建U-MARVEL三阶段框架（渐进过渡→过滤硬负→重排蒸馏），在M-BEIR上以单模型63.2% Avg大幅超越现有SOTA，零样本迁移CIR和T2V同样领先。

**[Unified Vision-Language Modeling via Concept Space Alignment](multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)**

:   提出v-Sonar将视觉编码器后置对齐到文本嵌入空间Sonar，使得在Sonar空间上训练的Large Concept Model (LCM)能零样本处理视觉输入，并通过指令微调扩展为v-LCM，在61/62种语言上超越现有VLM。

**[UniHM: Unified Dexterous Hand Manipulation with Vision Language Model](multimodal_vlm/unihm_unified_dexterous_hand_manipulation_with_vision_language_model.md)**

:   提出UniHM，首个统一的语言条件灵巧手操控框架，通过形态无关VQ codebook将异构机械手映射到共享离散空间，结合VLM进行指令驱动操控序列生成，并通过物理引导动态优化确保物理可行性。

**[VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](multimodal_vlm/visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)**

:   提出VisioMath基准，包含1800道K-12数学题目，所有选项均为高度视觉相似的图表，揭示了LMM在多图像-文本对齐上的核心短板，并探索三种对齐策略实现+12.6%的提升。

**[Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](multimodal_vlm/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)**

:   提出Vision-R1，通过Modality Bridging构建200K高质量多模态CoT数据进行冷启动初始化，再用渐进思维抑制训练(PTST)策略结合GRPO强化学习，在7B参数规模达到与OpenAI O1接近的多模态数学推理能力。

**[Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)**

:   提出 Vision-Zero，首个无标注的游戏化自博弈框架，通过"谁是卧底"式视觉推理游戏实现 VLM 的可扩展自进化，结合 Iterative-SPO 训练算法在推理、图表理解和视觉中心任务上超越基于人工标注数据的 SOTA 方法。

**[VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](multimodal_vlm/visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)**

:   提出首个面向数据可视化美学与质量评估的综合基准 VisJudge-Bench（3,090 样本，32 种图表类型），并训练 VisJudge 模型，将 MAE 相比 GPT-5 降低 23.9%，与人类专家的一致性提升 60.5%。

**[Visual Prompt-Agnostic Evolution](multimodal_vlm/visual_prompt-agnostic_evolution.md)**

:   提出 Prompt-Agnostic Evolution (PAE)，通过频域感知的任务初始化 (MPA) 和 Koopman-Lyapunov 动力系统 (KLD) 跨层关联 prompt，加速 VPT 收敛（平均 1.41× 加速）并在 25 个数据集上提升 1–3% 精度，且对各类 VPT 变体即插即用、无推理开销。

**[Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models](multimodal_vlm/visual_symbolic_mechanisms_vlm.md)**

:   发现 VLM 内部涌现了一套三阶段符号处理机制（ID retrieval → ID selection → feature retrieval），利用内容无关的空间位置索引（position IDs）来解决视觉绑定问题，并证明绑定错误可直接追溯到这些机制的失败。

**[WebDS: An End-to-End Benchmark for Web-based Data Science](multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)**

:   提出首个端到端 Web 数据科学基准 WebDS（870 个任务，29 个网站，10 个领域），当前最强 Agent（BrowserUse + GPT-4o）仅完成 15% 的任务，而人类达到 90%，揭示了真实数据科学工作流中 Agent 的巨大性能差距。

**[Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems](multimodal_vlm/why_keep_your_doubts_to_yourself_trading_visual_uncertainties_in_multi-agent_ban.md)**

:   提出 Agora 框架，将多智能体 VLM 协调问题重构为去中心化的不确定性交易市场——将认知不确定性铸造为可量化的三维可交易资产（感知/语义/推理），通过利润驱动的交易协议和市场感知的 Thompson Sampling Broker 实现成本高效的均衡分配，在 5 个多模态基准上一致超越启发式方法（如 MMMU 上 +8.5% 准确率同时成本降低 3 倍以上）。

**[Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](multimodal_vlm/why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)**

:   通过拼图任务系统研究 SFT 与 RFT 对先验知识的影响，揭示 RFT 避免灾难性遗忘的核心在于**数据分布**而非算法差异——RFT 采样的数据天然与基模型概率景观对齐，干扰更小。

---

## 🧊 3D视觉 { #3d_vision }

**[3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3d_vision/3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)**

:   提出 3DGEER 框架，通过推导沿光线积分高斯密度的闭式解、设计粒子包围截锥体 (PBF) 进行精确高效的光线-粒子关联、以及引入双极等角投影 (BEAP) 统一宽视场相机表示，在任意相机模型下实现了几何精确且实时高效的 3D 高斯渲染，在鱼眼和针孔数据集上全面超越现有方法。

**[A Genetic Algorithm for Navigating Synthesizable Molecular Spaces](3d_vision/a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)**

:   提出 SynGA，一种直接在合成路线（合成树）上操作的遗传算法，通过自定义的交叉和变异算子将搜索严格约束在可合成分子空间内，结合 ML 驱动的构建块过滤实现 SOTA 的可合成类似物搜索和属性优化性能。

**[A Step to Decouple Optimization in 3DGS](3d_vision/a_step_to_decouple_optimization_in_3dgs.md)**

:   深入分析 3DGS 优化中被忽视的更新步耦合（不可见视点下的隐式更新和动量重缩放）和梯度耦合（正则化与光度损失在 Adam 动量中的耦合），通过解耦和重组提出 AdamW-GS 优化器，在不引入额外剪枝操作的情况下同时提升重建质量和减少冗余原语。

**[Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting](3d_vision/augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)**

:   提出增强辐射场 (Augmented Radiance Field) 框架，通过设计具有视角相关不透明度的增强高斯核来显式建模高光分量，并引入误差驱动的补偿策略（2D 高斯初始化 → 逆投影至 3D → 联合优化），作为后处理即插即用地增强现有 3DGS 场景，在多个数据集上超越 SOTA NeRF 方法，同时仅需二阶球谐即可捕获复杂光照。

**[cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](3d_vision/cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)**

:   cadrille 是首个同时处理点云、多视角图像和文本输入的多模态 CAD 重建模型，通过 VLM 基础架构 + SFT + RL 微调的三阶段训练范式，在 10 个 CAD 重建基准上达到 SOTA，尤其是 RL 微调将无效率降至接近 0%。

**[CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions](3d_vision/clods_visual-only_unsupervised_cloth_dynamics_learning_in_unknown_conditions.md)**

:   CloDS 提出首个从多视角视频中无监督学习布料动力学的框架，通过 Spatial Mapping Gaussian Splatting 建立 2D 图像到 3D 网格的可微映射，结合双位置不透明度调制解决自遮挡问题，使 GNN 在无物理参数监督下就能学到接近全监督水平的布料动力学。

**[Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](3d_vision/color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)**

:   Color3D 提出"只上色一张关键视角→微调个性化 colorizer→传播颜色到所有视角和时间步"的范式，将复杂的 3D 上色问题转化为单图上色+颜色传播问题，在静态和动态 3D 场景上都实现了丰富色彩、跨视角一致性和用户可控性的统一。

**[COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](3d_vision/coopertrim_adaptive_data_selection_for_uncertainty-aware_cooperative_perception.md)**

:   提出 CooperTrim 自适应特征选择框架，通过共形时序不确定性度量评估特征相关性，并用数据驱动机制动态决定共享数量，在协同语义分割中实现 80.28% 带宽降低且性能可比，首次将选择性共享应用于协同分割任务。

**[CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D](3d_vision/core-3d_context-aware_open-vocabulary_retrieval_by_embeddings_in_3d.md)**

:   提出CORE-3D，一个无需训练的开放词汇3D语义分割与自然语言目标检索流水线，通过渐进式粒度掩码生成、上下文感知CLIP编码和多视角3D融合，在Replica和ScanNet上超越现有方法。

**[CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives](3d_vision/crisp_contact-guided_real2sim_from_monocular_video_with_planar_scene_primitives.md)**

:   提出 CRISP，一种从单目视频中恢复可仿真人体运动和场景几何的方法，通过拟合平面原语获取干净的仿真就绪几何体，结合人体-场景接触建模重建被遮挡区域，将人形控制器的运动追踪失败率从 55.2% 降至 6.9%。

**[Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](3d_vision/ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)**

:   提出Ctrl&Shift，一个端到端扩散框架，通过将物体操纵分解为物体移除+参考引导修复，并注入相对相机位姿控制，首次在不依赖显式3D重建的情况下实现几何一致的细粒度物体操纵。

**[D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](3d_vision/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)**

:   提出D-REX，一个基于高斯表示的可微real-to-sim-to-real引擎，通过视觉观测和机器人控制信号进行端到端物体质量辨识，并利用辨识的质量进行力感知的灵巧抓取策略学习，有效缩小了sim-to-real差距。

**[DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](3d_vision/diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)**

:   提出 DiffWind，一个物理约束的可微分框架，通过将风建模为网格物理场、物体表示为 3D Gaussian Splatting 粒子系统、用 Material Point Method（MPM）建模风-物交互，并引入 Lattice Boltzmann Method（LBM）作为物理约束，实现了从视频中联合重建风力场和物体运动，并支持新风条件下的前向仿真和风力迁移等应用，在自建的 WD-Objects 数据集上显著超越已有动态场景建模方法。

**[Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models](3d_vision/dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)**

:   首次将稀疏自编码器 (SAE) 应用于时间序列基础模型 Chronos-T5-Large，通过 392 次因果消融实验揭示了深度依赖的特征层级：中层编码器集中了因果关键的突变检测特征，而语义最丰富的末层编码器反而因果重要性最低。

**[Dynamic Novel View Synthesis in High Dynamic Range](3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)**

:   首次提出 HDR 动态新视角合成 (HDR DNVS) 问题，并设计 HDR-4DGS 框架，通过动态色调映射模块在时变场景中实现时序一致的 HDR 辐射场重建，在合成和真实数据集上均超越现有方法。

**[Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention](3d_vision/efficient-lvsm_faster_cheaper_and_better_large_view_synthesis_model_via_decouple.md)**

:   提出 Efficient-LVSM，通过解耦输入视图编码与目标视图生成的双流架构，将新视图合成的复杂度从 $O(N_{in}^2)$ 降至 $O(N_{in})$，在 RealEstate10K 上以 50% 训练时间达到 SOTA（29.86 dB PSNR），推理速度提升 4.4 倍。

**[EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](3d_vision/egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)**

:   提出首个夜间第一人称视觉基准 EgoNight，包含日夜对齐视频和 3658 个人工验证 QA 对，揭示 MLLM 在低光照下存在高达 32.8% 的性能下降。

**[EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations](3d_vision/egoworld_translating_exocentric_view_to_egocentric_view_using_rich_exocentric_ob.md)**

:   EgoWorld 提出一种端到端的外部-第一人称视角转换框架：从单张第三人称图像中提取 3D 点云、手部姿态和文本描述三种互补观测，通过点云重投影获得稀疏第一人称 RGB 映射，再以扩散模型 inpainting 方式重建完整的第一人称高保真图像，在 H2O 等四个数据集的多种 unseen 设置下全面超越 SOTA。

**[Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances](3d_vision/fast_estimation_of_wasserstein_distances_via_regression_on_sliced_wasserstein_di.md)**

:   利用 Sliced Wasserstein（SW）距离既能提供 Wasserstein 距离的下界、lifted SW 距离又能提供上界这一数学性质，构建极简的线性回归模型（RG 框架），仅用少量分布对的精确 Wasserstein 作为监督信号就能训练出高精度的 Wasserstein 代理估计器，在低数据场景下全面碾压 Transformer 方法 Wasserstein Wormhole。

**[FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](3d_vision/fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)**

:   提出 FastGHA，一个前馈式少样本 3D 高斯头部化身生成框架，从 4 张任意表情/视角的输入图像在 ~1 秒内重建可动画的 3D 高斯头部，支持 62 FPS 实时动画，在 Ava-256 上 PSNR 达到 22.5 dB（超越 Avat3r 的 20.7，且快 7.75 倍）。

**[FeDaL: Federated Dataset Learning for General Time Series Foundation Models](3d_vision/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)**

:   提出 FeDaL 联邦框架，通过客户端域偏差消除（DBE）和服务器全局偏差消除（GBE）从头训练通用时序基础模型，在8类下游任务上以远少于集中式TSFM的参数达到竞争甚至超越的性能。

**[Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?](3d_vision/fused-planes_why_train_a_thousand_tri-planes_when_you_can_share.md)**

:   提出 Fused-Planes，通过宏观-微观分解将 Tri-Plane 表示分为共享的类级基平面（macro）和对象特有的细节平面（micro），结合潜空间渲染，实现 7× 训练加速、3× 内存压缩，同时保持甚至超越独立 Tri-Plane 的重建质量。

**[Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)**

:   CLAP（Coarse-to-fine Language-Aligned manipulation Policy）通过任务分解、VLM微调的3D关键点预测和3D感知表征三个核心组件，实现了对新指令和新环境的强泛化能力，在 GemBench 上以 1/5 的训练数据比 SOTA 高出 12%。

**[GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation](3d_vision/geopurify_a_data-efficient_geometric_distillation_framework_for_open-vocabulary_.md)**

:   提出 GeoPurify 框架，通过从 3D 自监督教师模型蒸馏几何先验来净化 2D VLM 投影到 3D 的噪声特征，仅用约 1.5% 的训练数据即可达到或超越全量训练的 SOTA 开放词汇 3D 分割性能。

**[GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)**

:   提出 GIQ 基准数据集，包含 224 种合成和真实多面体，通过单目 3D 重建、对称性检测、心理旋转测试和零样本分类四项任务系统评估视觉基础模型的几何推理能力，揭示了当前模型在基本几何理解上的显著不足。

**[HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](3d_vision/hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)**

:   提出 HDR-NSFF，将 HDR 视频重建从传统的 2D 像素级融合范式转变为 4D 时空建模，从交替曝光单目视频中联合重建 HDR 辐射场、3D 场景流、几何和色调映射，实现了时空一致的动态 HDR 新视角合成。

**[Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry](3d_vision/into_the_rabbit_hull_from_task-relevant_concepts_in_dino_to_minkowski_geometry.md)**

:   通过在 DINOv2 上训练 32,000 单元的 Sparse Autoencoder 字典，系统分析了下游任务如何招募不同概念，发现表征几何偏离线性稀疏假说（LRH），进而提出 Minkowski Representation Hypothesis（MRH），认为 token 表征是多个凸多面体的 Minkowski 和，概念由原型点的邻近性而非线性方向定义。

**[Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](3d_vision/joint_shadow_generation_and_relighting_via_light-geometry_interaction_maps.md)**

:   提出 Light-Geometry Interaction (LGI) maps，一种从单目深度估计中编码光照-遮挡关系的 2.5D 表示，嵌入 bridge matching 生成框架中实现阴影生成与物体重光照的联合建模，在合成和真实图像上均取得 SOTA 效果。

**[LaVCa: LLM-assisted Visual Cortex Captioning](3d_vision/lavca_llm-assisted_visual_cortex_captioning.md)**

:   提出 LaVCa 方法，利用 LLM 为人类视觉皮层的每个体素生成自然语言描述（caption），通过"编码模型→最优图像选取→MLLM生成描述→LLM关键词提炼+句子组合"四步流程，比已有方法 BrainSCUBA 更准确、更多样地揭示了体素级视觉选择性。

**[Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation](3d_vision/learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)**

:   提出 PA3FF（Part-Aware 3D Feature Field），一种原生 3D 的稠密部件感知特征表示，通过 Sonata 预训练骨干 + 几何/语义对比学习获得零部件级特征，结合 Part-Aware Diffusion Policy (PADP) 实现少样本、高泛化性的关节物体操作，在仿真和真实环境中均大幅超越 CLIP/DINOv2/GenDP 等基线。

**[Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)**

:   提出NGFF框架，从多视角RGB图像构建3D高斯表示并学习显式神经力场驱动物理动力学，通过ODE求解实现交互式物理真实4D视频生成，比传统高斯模拟器快两个数量级，超越Veo3和NVIDIA Cosmos。

**[Learning Unified Representation of 3D Gaussian Splatting](3d_vision/learning_unified_representation_of_3d_gaussian_splatting.md)**

:   3DGS原生参数 $\boldsymbol{\theta}=\{\mu,\mathbf{q},\mathbf{s},\mathbf{c},o\}$ 存在非唯一性与数值异质性，不适合作为神经网络的学习空间。本文提出**子流形场 (Submanifold Field)** 表示：将每个高斯基元映射到其等概率椭球面上的连续颜色场，证明该映射是单射的，从根源上消除参数歧义，并配合基于最优传输的流形距离 (M-Dist) 训练 VAE 嵌入，在重建保真度、跨域泛化与潜空间稳定性上全面优于参数基线。

**[LiTo: Surface Light Field Tokenization](3d_vision/lito_surface_light_field_tokenization.md)**

:   提出LiTo——通过将表面光场(surface light field)编码为紧凑latent向量集合来同时建模3D几何和视角依赖外观：输入RGB-D多视角图像的光场随机子采样 -> Perceiver IO编码器(支持100万token输入的3D局部attention) + flow-matching几何解码器 + 高阶球谐Gaussian解码器 -> 实现重建和单图到3D生成都超越TRELLIS，首次在latent 3D表示中建模高光/菲涅尔反射等视角依赖效果。

**[MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)**

:   提出MEGS2——从渲染VRAM角度出发压缩3DGS：用可裁剪的任意方向球面高斯(SG)完全替代球谐函数(SH)降低每个primitive的参数量 + 统一软剪枝框架将primitive数量和lobe数量的裁剪建模为单一内存约束优化问题 -> 实现8x静态VRAM压缩和6x渲染VRAM压缩，同时保持渲染质量，首次让3DGS在移动端实时运行。

**[MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting](3d_vision/moe-gs_mixture_of_experts_for_dynamic_gaussian_splatting.md)**

:   提出 MoE-GS，首个将混合专家架构引入动态高斯泼溅的框架，通过 Volume-aware Pixel Router 自适应融合多种异构变形先验（HexPlane/逐高斯/多项式/插值），在 N3V 和 Technicolor 数据集上一致超越 SOTA，并通过单次渲染、门控剪枝和知识蒸馏保持效率。

**[Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](3d_vision/mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)**

:   首次解决从无位姿交替曝光单目视频重建可渲染 4D HDR 场景的问题，通过两阶段优化（正交视频空间 → 世界空间）、Video-to-World 高斯变换策略和时间亮度正则化，在合成数据上达到 37.64 dB HDR PSNR、161 FPS，全面超越现有方法。

**[MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models](3d_vision/multimat_multimodal_program_synthesis_for_procedural_materials_using_large_multi.md)**

:   提出 MultiMat，首个将大型多模态模型（LMM）用于程序化材质节点图合成的框架，通过在自回归生成过程中融合中间节点的视觉渲染反馈（混合调节/图调节两种模式），并配合增量式约束树搜索推理实现即时校验与回溯纠错，在 6878 个产级 Substance Designer 材质上训练后，无条件生成与条件生成均大幅超越纯文本基线。

**[NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction](3d_vision/nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)**

:   提出NOVA3R——从无位姿图像进行非像素对齐的完整3D重建：用可学习场景token跨视角聚合全局信息 + 基于flow-matching的扩散3D解码器生成完整(含遮挡区域)的点云，解决像素对齐方法只能重建可见面且重叠区域有冗余几何的两大根本限制，在SCRREAM/GSO等数据集上场景级和物体级重建均超越SOTA。

**[Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](3d_vision/omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)**

:   构建统一的3D场景理解与生成模型 Omni-View，通过纹理模块（新视角合成）和几何模块（深度/位姿估计）的生成能力增强理解性能，在 VSI-Bench 上达到 55.4 分超越所有现有专用3D理解模型。

**[One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](3d_vision/one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)**

:   提出One2Scene——将单图到可探索3D场景的病态问题分解为三个子任务：(1)全景图生成扩展视觉覆盖 (2)前馈3DGS网络从稀疏锚点视角构建显式3D几何scaffold (3)scaffold引导的新视角合成，通过Dual-LoRA融合高质量锚点视角和几何先验，在大视角变化下实现几何一致且逼真的场景生成，显著超越SOTA。

**[OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation](3d_vision/openfly_a_comprehensive_platform_for_aerial_vision-language_navigation.md)**

:   构建OpenFly——航空视觉-语言导航(VLN)综合平台：集成4种渲染引擎(UE/GTA V/Google Earth/3DGS)+开发全自动数据生成工具链(点云获取→语义分割→轨迹生成→GPT-4o指令)+构建10万轨迹大规模数据集(18场景)+提出关键帧感知VLN模型OpenFly-Agent(关键帧选择+视觉token融合)，在已见/未见场景分别以14.0%/7.9%的成功率优势超越现有方法。

**[PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](3d_vision/partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)**

:   提出首个在大规模原生 3D 数据上训练的可提示部件分割模型 PartSAM，采用 triplane 双分支编码器（冻结 SAM 先验 + 可学习 3D 分支）和 SAM 风格解码器，通过模型在环标注流程构建 500 万+形状-部件对，在开放世界设置下单次点击即超越 Point-SAM 90%+。

**[PD²GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting](3d_vision/pd2gs_part-level_decoupling_and_continuous_deformation_of_articulated_objects_vi.md)**

:   提出 PD²GS 框架，通过学习共享的 canonical 高斯场并将每个交互状态建模为其连续形变，实现铰接物体的部件级解耦、重建和连续控制，采用粗到细的运动轨迹聚类 + SAM 引导的边界细化，无需手动监督。

**[Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](3d_vision/peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)**

:   提出 PUN（Peering into the UnkNowN），用轻量前馈网络 UPNet 从单张图像直接预测球面上所有候选视点的不确定性分布（neural uncertainty map），替代了需要迭代重训 NeRF/3DGS 的传统主动视点选择流程。仅用上界一半的视点就达到可比的重建质量，选点阶段实现 400 倍加速和 50%+ 的计算资源节省。

**[pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning](3d_vision/pyspatial_generating_3d_visual_programs_for_zero-shot_spatial_reasoning.md)**

:   pySpatial 是一个视觉编程框架，让 MLLM 通过生成 Python 代码自动调用 3D 空间工具（3D 重建、相机位姿恢复、新视角渲染等），将有限的 2D 图像输入转化为可交互探索的 3D 场景，实现零样本、即插即用的显式 3D 空间推理，在 MindCube 基准上以 58.56% 的整体准确率超越 GPT-4.1-mini 12.94%、超越 VLM-3R 16.5%，并成功驱动真实四足机器人完成室内导航。

**[QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)**

:   提出 QuadGPT——首个端到端自回归生成原生四边形网格的框架，通过统一的混合拓扑tokenization（三角形面 padding 为4顶点块）、Hourglass Transformer 架构、以及基于拓扑奖励的截断 DPO (tDPO) 微调，在 Chamfer Distance、Hausdorff Distance、四边形比例和用户偏好上全面超越现有的三角形→四边形转换流水线和十字场引导方法。

**[Quantized Visual Geometry Grounded Transformer](3d_vision/quantized_visual_geometry_grounded_transformer.md)**

:   针对十亿级 3D 重建模型 VGGT 的部署需求，提出首个专用 PTQ 框架 QuantVGGT，通过双重平滑细粒度量化（Hadamard 旋转 + 通道平滑）解决特殊 token 导致的重尾分布，以及噪声过滤多样化采样解决校准不稳定问题，4-bit 量化实现 3.7× 内存压缩和 2.5× 加速，保持 98%+ 精度。

**[Scaling Sequence-to-Sequence Generative Neural Rendering](3d_vision/scaling_sequence-to-sequence_generative_neural_rendering.md)**

:   提出 Kaleido，一系列将 3D 视为视频特殊子域的 decoder-only rectified flow transformer 生成模型，通过统一位置编码（Unified Positional Encoding）、掩码自回归框架和视频预训练策略，实现无需任何显式 3D 表示的 "any-to-any" 6-DoF 新视角合成，**首次在多视角设置下匹配逐场景优化方法（InstantNGP）的渲染质量**，并将分辨率从 512/576px 提升至 1024px。

**[SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](3d_vision/scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)**

:   SceneTransporter 通过在组合 3D latent 扩散模型的去噪循环中引入熵最优传输（OT）框架，将 open-world 结构化 3D 场景生成重新建模为全局关联分配问题：OT 计划门控交叉注意力实现排他性的 patch-to-part 路由（防止特征纠缠），边缘正则化的分配代价鼓励在图像边缘处分离不同实例，在 74 张多样化 open-world 场景图像上实现了 SOTA 的实例级一致性和几何保真度。

**[Sharp Monocular View Synthesis in Less Than a Second](3d_vision/sharp_monocular_view_synthesis_in_less_than_a_second.md)**

:   SHARP 通过单次前馈神经网络从单张照片生成约 120 万个 3D Gaussian，在 A100 GPU 上不到 1 秒完成推理，渲染速度超 100 FPS，在 6 个数据集上零样本泛化均达 SOTA，相比最强先前方法 LPIPS 降低 25–34%、合成时间缩短三个数量级。

**[Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)**

:   在 student-teacher 蒸馏框架中，用预训练的前馈式 3D 重建模型（MVSplat）增强 teacher，将 2D 特征提升到 3D Gaussian 表示后渲染到新视角，从而让 student 学到几何一致的 3D-aware 2D 特征，在深度估计、法线估计、语义分割和多视图对应等下游任务上全面超越现有方法。

**[Splat Feature Solver](3d_vision/splat_feature_solver.md)**

:   将3D splat表示的特征提升(feature lifting)问题统一建模为稀疏线性逆问题 $AX=B$，提出闭式求解器并证明其在凸损失下的 $(1+\beta)$-近似误差上界，配合 Tikhonov 引导和后聚合过滤两种正则化策略，在开放词汇3D分割任务上达到SOTA。

**[Station2Radar: Query-Conditioned Gaussian Splatting for Precipitation Field](3d_vision/station2radar_query_conditioned_gaussian_splatting_for_precipitation_field.md)**

:   提出 Query-Conditioned Gaussian Splatting (QCGS)，首次将 2D 高斯溅射引入降水场生成任务，融合卫星图像与自动气象站稀疏观测，实现无雷达条件下分辨率灵活的降水场重建，RMSE 较传统网格化产品提升超 50%。

**[StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams](3d_vision/streamsplat_towards_online_dynamic_3d_reconstruction_from_uncalibrated_video_str.md)**

:   StreamSplat 提出了一个完全前馈的在线动态3D重建框架，通过概率位置采样、双向形变场和自适应高斯融合三大创新，能从未标定视频流中即时生成动态3DGS表示，速度比优化方法快1200倍。

**[Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models](3d_vision/stroke3d_lifting_2d_strokes_into_rigged_3d_model_via_latent_diffusion_models.md)**

:   Stroke3D 首次实现从用户绘制的2D笔画和文本提示直接生成绑骨3D网格模型，采用骨骼优先的两阶段流水线：先用图VAE+图DiT生成可控3D骨骼，再通过TextuRig数据集增强和SKA-DPO优化生成高质量网格。

**[Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting](3d_vision/stylos_multi-view_3d_stylization_with_single-forward_gaussian_splatting.md)**

:   Stylos 提出了一个单次前馈的3D风格迁移框架，通过共享Transformer骨干的双路径设计（几何自注意力+风格交叉注意力）和体素级3D风格损失，实现从未标定输入的零样本3D风格化，支持单视角到数百视角的扩展。

**[SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors](3d_vision/surfsplat_conquering_feedforward_2d_gaussian_splatting_with_surface_continuity_p.md)**

:   SurfSplat 提出基于2DGS的前馈3D重建框架，通过表面连续性先验将高斯的旋转和尺度与邻域位置绑定、以及强制透明度混合策略解决颜色偏差，并引入HRRC指标揭示高分辨率下的重建质量差异。

**[Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](3d_vision/text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)**

:   提出VIST3A框架——通过模型拼接(model stitching)将预训练视频生成器的latent空间与前馈3D重建模型(如AnySplat/MVDUSt3R/VGGT)无缝对接，再用直接奖励微调(direct reward finetuning)对齐生成模型与拼接后的3D解码器，实现高质量端到端text-to-3DGS和text-to-pointmap生成，在T3Bench/SceneBench/DPG-Bench上全面超越现有方法。

**[Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)**

:   提出一种类似"织丝"的网格 tokenization 算法，通过顶点分层和排序提供规范的拓扑框架，保证生成网格的流形性、水密性、法线一致性和部件感知性，同时达到 SOTA 压缩效率。

**[UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)**

:   提出 UFO-4D，一个统一的前馈框架，仅从两张无位姿图像直接预测动态 3D 高斯表示，实现 3D 几何、3D 运动和相机位姿的联合一致估计，在几何和运动基准上比现有方法提升达 3 倍。

**[Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)**

:   提出 USplat4D，一种不确定性感知的动态高斯泼溅框架，通过估计每个高斯的时变不确定性并构建不确定性引导的时空图来传播可靠运动线索，显著提升了遮挡区域和极端新视角下的单目 4D 重建质量。

**[Universal Beta Splatting](3d_vision/universal_beta_splatting.md)**

:   提出 Universal Beta Splatting (UBS)，将 3D 高斯 Splatting 推广为 N 维各向异性 Beta 核，通过逐维度形状控制在单一表示中统一建模空间几何、视角依赖外观和场景动态，实现了可解释的场景分解和 SOTA 渲染质量。

**[UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](3d_vision/urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)**

:   提出 UrbanGS，一个面向城市级场景的可扩展 3DGS 重建框架，通过深度一致的 D-Normal 正则化、空间自适应高斯剪枝和统一分区策略，同时提升几何精度、渲染质量和内存效率。

**[Weight Space Representation Learning on Diverse NeRF Architectures](3d_vision/weight_space_representation_learning_on_diverse_nerf_architectures.md)**

:   提出首个能处理多种 NeRF 架构（MLP/tri-plane/hash table）权重的表示学习框架，通过 Graph Meta-Network 编码器 + SigLIP 对比损失构建架构无关的潜在空间，在 13 种 NeRF 架构上实现分类、检索和语言任务，并能泛化到训练时未见的架构。

---

## 💡 LLM推理 { #llm_reasoning }

**[Adaptive Social Learning via Mode Policy Optimization for Language Agents](llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)**

:   提出 Adaptive Social Learning（ASL）框架，设计四种层次化推理模式（从直觉回应到深度推演），并通过 AMPO 算法（融合模式级和样本级优势估计）让 LLM agent 根据社交场景复杂度自适应切换推理深度，在社交智能任务上比 GPT-4o 高 15.6%，比 GRPO 高 7.0% 且 token 用量减少 32.8%。

**[Agentified Assessment of Logical Reasoning Agents](llm_reasoning/agentified_assessment_of_logical_reasoning_agents.md)**

:   提出基于Agent的评测框架(AAA)，将评估逻辑封装为assessor agent并通过标准A2A接口与被测agent交互，在经Vampire定理证明器系统清洗的FOLIO数据集上，自动形式化agent（NL→Z3Py+SMT求解）达到86.70%准确率，大幅超过CoT基线73.89%，尤其在矛盾检测(False类)上提升32.79个百分点。

**[AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)**

:   AgentMath提出一个工具增强的Agent框架，通过自动化数据合成、多轮交互式强化学习和高效异步训练系统，将LLM推理能力与代码解释器的计算精度无缝结合，在AIME24/25和HMMT25上以30B-A3B规模达到SOTA水平（90.6%/86.4%/73.8%），超越o3-mini和Claude-Opus-4.0-Thinking。

**[AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](llm_reasoning/aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)**

:   提出 AIMCoT，将多模态 CoT 的视觉信息选择从"被动关注高注意力区域"转变为"主动寻找最高信息增益区域"，通过三个模块（CAG 上下文增强注意力图、AVP 主动视觉探测、DAT 动态注意力转移触发）协同工作，在 LLaVA-W 上比 ICoT 提升 18.25%（0-shot），是一个免训练的即插即用框架。

**[Annotation-Efficient Universal Honesty Alignment](llm_reasoning/annotation-efficient_universal_honesty_alignment.md)**

:   提出 EliCal（先激发后校准）两阶段框架，先用无标注的 self-consistency 信号教 LLM 表达内部置信度，再用极少量正确性标注（仅 1k 个，占 0.18%）进行校准，在 HonestyBench（560K 训练 + 70K 评估）上达到接近全量标注 98% 的诚实性对齐性能，并在未见 MMLU 任务上泛化优于仅校准基线。

**[Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](llm_reasoning/are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)**

:   系统评估推理型 LLM 对其 CoT 中各种干预（良性/中性/对抗性）的鲁棒性：发现模型总体鲁棒能从干预中恢复，但改写风格（paraphrasing）会抑制"自我怀疑"表达导致正确率下降，恢复过程有显著计算开销（CoT 膨胀最高 665%）。

**[ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)**

:   提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

**[Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts](llm_reasoning/beyond_prompt-induced_lies_investigating_llm_deception_on_benign_prompts.md)**

:   提出 Contact Searching Question (CSQ) 框架，基于有向图可达性任务和认知心理学原理设计两个互补统计指标——欺骗意图分数 $\rho$ 和欺骗行为分数 $\delta$，首次系统揭示 16 个主流 LLM 在完全良性提示下存在随任务难度升级的自发欺骗倾向。

**[Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](llm_reasoning/compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)**

:   本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是**组合泛化**——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。

**[Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors](llm_reasoning/conflict-aware_fusion_resolving_logic_inertia_in_large_language_models_via_struc.md)**

:   揭示了 LLM 的"逻辑惯性"现象——在遇到矛盾前提时仍沿学习到的推理轨迹继续推理（准确率降至 0.0），提出 Conflict-Aware Fusion 双过程架构，通过强制前提验证先于推理执行，在矛盾检测上实现 100% 准确率。

**[Continuous Chain of Thought Enables Parallel Exploration and Reasoning](llm_reasoning/continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)**

:   CoT2 提出用连续值 token（词表 embedding 的凸组合）替代离散 token 进行链式推理，使模型能在单次推理中并行追踪多条推理路径，理论证明等价于 K 次 self-consistency/best-of-N 采样，并通过 GRPO 强化学习进一步提升性能。

**[CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)**

:   提出CoT-RVS，一种完全无训练的多智能体框架，利用预训练MLLM的零样本CoT推理能力进行时间-语义关联分析与关键帧选取，在推理视频分割任务上大幅超越微调方法（Refer-DAVIS J&F 79.1 vs 71.2，ReasonVOS J&F 65.5 vs 49.9）。

**[CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](llm_reasoning/cyclicreflex_improving_reasoning_models_via_cyclical_reflection_token_scheduling.md)**

:   将推理过程中的反思token（如"wait"、"but"）视为可调度的"资源"，借鉴优化中周期性学习率的思想，提出CyclicReflex——一种免训练的解码策略，通过三角波形动态调控反思token的logit，在多个数学推理基准上（MATH500, AIME2024/2025, AMC2023）一致性提升1.5B-8B模型准确率。

**[DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](llm_reasoning/dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)**

:   将 LLM 的 CoT 推理形式化为 DAG 上的基于规则的随机过程，提出"逻辑闭合性"（logical closeness）度量来评估模型是否通过搜索还是严格逻辑推理得到答案，构建了 2894 个金标准 DAG-MATH benchmark，发现即使 PASS@k 相近的模型在推理忠实度上也存在显著差异。

**[DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)**

:   提出 Design Logic（设计逻辑）——从真题中逆向工程出的可复用元知识，用于指导从原始文本合成多学科推理问题。构建了 470 万道覆盖 75 学科的推理题目，SFT 后的 base 模型甚至超越经过完整后训练的官方模型。

**[Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](llm_reasoning/doxing_via_the_lens_revealing_location-related_privacy_leakage_on_multi-modal_la.md)**

:   本文首次系统研究了多模态大推理模型（MLRMs）从用户生成图像中推断敏感地理位置信息的隐私泄露风险，提出三级隐私风险框架、DoxBench 基准和 Glare 信息论评估指标，发现 MLRMs 在地理推断上超越非专家人类，显著降低了攻击者获取敏感位置信息的门槛。

**[DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](llm_reasoning/drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)**

:   诊断出 GRPO 在加入长度惩罚后的根本缺陷——正确但冗长的回答可能获得负优势值从而被错误惩罚——提出 DRPO 将正负样本的奖励信号解耦，确保长度惩罚只在正确回答组内归一化，在 1.5B 模型上实现 77% 长度缩减仅 1.1% 性能损失（对比基线 68% 缩减 4.3% 损失）。

**[Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](llm_reasoning/dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)**

:   将 RL 微调中每个 prompt 的求解进度建模为隐马尔可夫动力系统，通过轻量贝叶斯推断在线预测 prompt 的求解状态，优先采样"部分求解"的 prompt，以不到 DS 30% 的 rollout 量达到同等甚至更优的推理性能。

**[Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)**

:   将隐式CoT建模为结构因果模型(SCM)，通过逐步do-干预分析Coconut和CODI两种范式，发现隐式推理步骤具有异质性因果杠杆、非局部跳跃传播结构、以及输出层早期偏向与表征层晚期提交之间的持续性差距。

**[Efficient Test-Time Scaling for Small Vision-Language Models](llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)**

:   为小型 VLM 提出两种高效的测试时缩放策略：TTAug（对输入做多种增强后在 token 级别聚合输出概率）和 TTAdapt（用 TTAug 生成的伪标签自适应调整模型参数），在 9 个基准上一致提升性能，同时计算效率远优于现有的基于重复采样的测试时方法。

**[Estimating the Empowerment of Language Model Agents](llm_reasoning/estimating_the_empowerment_of_language_model_agents.md)**

:   提出 EELMA 算法，利用信息论中的"赋权"（empowerment，即 agent 动作与未来状态的互信息）作为目标无关的 LM Agent 能力度量指标，在语言游戏和真实网页浏览场景中与任务表现强相关（$r=0.83$–$0.94$），可用于开放式 agent 监控与安全评估。

**[FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](llm_reasoning/fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)**

:   针对GRPO训练中生成阶段占91%-98%时间的严重瓶颈，提出并发感知的投机解码策略（动态调整draft树参数以适配从高到低的实时并发度变化）和在线draft模型学习（利用目标模型生成的hidden states持续适配分布漂移），整体实现2.35x-2.72x端到端训练加速，且不损害推理质量。

**[Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)**

:   Fine-R1 通过 CoT 监督微调（"视觉分析→候选子类→对比→预测"结构化推理链）+ 三元组增强策略优化 TAPO（类内增强提升鲁棒性 + 类间增强提升判别力），仅用 4-shot 训练即在细粒度视觉识别上超越 CLIP 和通用/推理型 MLLM。

**[Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)**

:   系统诊断推理时奖励模型（RM）的三大失效模式——简单题性能退化、采样数增多时辨别力下降、高搜索多样性损害准确率，并提出 CRISP 算法通过答案聚类的奖励聚合与逐步前缀引导来缓解这些问题，准确率提升最高 5%。

**[From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](llm_reasoning/from_abstract_to_contextual_what_llms_still_cannot_do_in_mathematics.md)**

:   本文提出 ContextMATH 基准，通过将 AIME 和 MATH-500 的抽象数学问题转换为两种情境变体（场景嵌入 SG 和复杂度缩放 CS），系统揭示了LLM在情境化数学推理中的大幅性能下降——开源模型在 SG 上平均下降 13%，CS 上下降 34%——并识别出"问题建模"和"推理执行"是两个互补的性能瓶颈。

**[GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](llm_reasoning/geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)**

:   形式化Program-to-Geometry任务并提出GeoGramBench（500题），按三级几何复杂度分类法评估19个前沿LLM从过程式绘图代码构建几何表征并推理的能力，发现即使GPT-5在最高抽象级别也仅39.26%准确率，揭示了LLM空间抽象的根本性短板。

**[Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](llm_reasoning/harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)**

:   揭示GRPO的优势函数（std归一化）导致更新幅度在中等难度题目处最大、对难题和易题均隐式抑制的问题，提出MathForge框架——DGPO（用MAD替换std实现难度均衡 + softmax难度加权）+ MQR（添加故事背景/抽象术语/嵌套子问题三方面改写增加难度但保留原答案），在Qwen2.5-Math-7B上在6个数学推理benchmark上平均超GRPO +4.56%。

**[I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift](llm_reasoning/i_cant_believe_its_not_robust_catastrophic_collapse_of_safety_classifiers_under_.md)**

:   本文系统研究了基于 frozen embedding 的安全分类器在模型更新导致 embedding 漂移时的脆弱性，发现仅 2% 的 embedding 扰动即可将分类器性能从 85% ROC-AUC 降至随机水平（50%），且 72% 的误分类发生在高置信度下（silent failure），同时 instruction-tuned 模型反而比 base 模型更难分类。

**[Is In-Context Learning Learning?](llm_reasoning/is_in-context_learning_learning.md)**

:   通过大规模控制变量实验系统分析 ICL 是否构成"学习"，发现数学上 ICL 满足学习定义，但实证表明其泛化能力有限——模型主要依赖 prompt 中的结构规律进行模式推演（deduction），而非从示例中真正习得新能力。

**[Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](llm_reasoning/is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)**

:   提出 TRACE（Truncated Reasoning AUC Evaluation）方法，通过逐步截断推理链并测量模型"多早"能获得奖励来量化推理努力程度，从而检测 CoT 监控无法发现的隐式奖励黑客行为，在数学和代码任务中比最强 CoT 监控器分别提升 65% 和 30% 以上的检测 F1。

**[LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation](llm_reasoning/lingoly-too_disentangling_reasoning_from_knowledge_with_templatised_orthographic.md)**

:   提出LingOly-TOO基准，通过专家设计的正字法置换（grapheme-level permutation）对语言学奥赛题进行混淆，保留推理逻辑但消除知识/记忆捷径，将15个前沿模型的最高分从0.59降至0.48，系统量化了LLM推理能力被知识效应高估的程度。

**[LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](llm_reasoning/logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)**

:   提出LogicReward奖励函数，用Isabelle定理证明器做步骤级逻辑正确性验证，结合Autoformalization with Soft Unification减少自然语言歧义，训练出的8B模型在NLI和逻辑推理任务上超越GPT-4o 11.6%和o4-mini 2%。

**[mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](llm_reasoning/mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)**

:   提出 mR3，一系列覆盖72种语言的多语言rubric-agnostic推理奖励模型，通过系统化的数据构建（GPT-OSS-120B蒸馏+难度过滤）和课程学习策略训练，14B模型在多语言评估基准上超越120B教师模型及所有同类基线，同时支持point-wise/pair-wise/binary三种评估范式。

**[Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](llm_reasoning/native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)**

:   提出 NRT（Native Reasoning Training）框架，将推理链视为隐变量，通过模型自身对参考答案的预测置信度作为内在奖励信号训练 LLM 推理能力，无需外部验证器或专家推理示范；在 Llama-3.1-8B 上 9 个基准平均提升 10.2 分（46.0→56.2），超越需要验证器的 RLPR +5.4 分。

**[No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](llm_reasoning/no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)**

:   在 LLM 生成答案之前，仅从问题处理后的残差流激活中训练线性探针（difference-of-means），即可预测模型即将生成的答案是否正确。该"提前正确性方向"在 TriviaQA 上训练后可跨域泛化到多个事实知识数据集（AUROC 0.68-0.88），但无法泛化到数学推理（GSM8K），揭示了"事实正确性"与"推理正确性"在模型内部表征中的结构性分离。

**[Nudging the Boundaries of LLM Reasoning](llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)**

:   指出GRPO无法从模型完全无法解决的难题(pass rate=0%)中学习的根本局限，提出NuRL方法在训练时对难题注入自生成的抽象hint(不泄露答案)使其变为可学习样本，跨3个模型6个benchmark一致超越GRPO并真正提升pass@k能力上界。

**[On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](llm_reasoning/on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)**

:   提出 Regularized Policy Gradient (RPG) 框架，系统推导并分析了基于 Forward/Reverse KL 散度（归一化和非归一化形式）的策略梯度方法，发现 GRPO 的 KL 项存在理论不一致性，并在数学推理任务上取得优于 GRPO、REINFORCE++、DAPO 的结果。

**[On The Fragility of Benchmark Contamination Detection in Reasoning Models](llm_reasoning/on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)**

:   系统性研究发现 LRM 的基准污染检测极其脆弱：SFT 阶段引入的污染在经过 GRPO 训练后检测信号几乎消失（PPO 式重要性采样/裁剪是根因），而对高级 LRM 直接用 CoT 做 SFT 污染则几乎不留任何可检测痕迹，现有 10 种检测方法均接近随机猜测。

**[Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)**

:   提出 Plan-and-Budget 框架，通过将复杂查询分解为子问题并基于估计复杂度自适应分配 token 预算，实现推理 LLM 的高效测试时缩放——最高提升 70% 准确率、减少 39% token、E3 指标提升 193.8%。

**[PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](llm_reasoning/prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)**

:   首次将分解式 Chain-of-Thought 推理与多维度强化学习（RL）结合应用于视频到音频（V2A）生成，通过四个专门化的 CoT 模块（语义/时序/美学/空间）配合对应奖励函数，解决了目标纠缠问题，并提出 Fast-GRPO 算法大幅降低 RL 训练开销。

**[RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](llm_reasoning/rain-merging_a_gradient-free_method_to_enhance_instruction_following_in_large_re.md)**

:   针对大推理模型（LRM）推理能力强但指令遵循能力弱的矛盾，提出 RAIN-Merging 方法，通过零空间投影保持 thinking 格式不变、注意力引导系数增强指令相关性，无需梯度训练即可将指令微调模型（ITM）的能力合并进 LRM，在 4 个指令遵循和 9 个推理基准上均取得稳定提升。

**[Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](llm_reasoning/reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)**

:   首次系统研究大型推理模型（LRM）的答案来源归因问题，揭示推理（CoT）和检索（记忆）两种机制同时竞争影响最终答案，并提出 Farl（遗忘增强强化学习）通过抑制检索捷径来提升模型的真实推理能力。

**[ReForm: Reflective Autoformalization with Prospective Bounded Sequence Optimization](llm_reasoning/reform_reflective_autoformalization_with_prospective_bounded_sequence_optimizati.md)**

:   提出 ReForm，一种反思式自动形式化范式，将自然语言数学问题转为 Lean 形式声明的过程从一次生成转变为"生成 → 语义自验证 → 修正"的迭代循环，并设计 PBSO 算法优化异构奖励信号，在四个基准上比最强基线平均提升 22.6 个百分点。

**[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](llm_reasoning/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)**

:   提出推理忠实度（Reasoning Faithfulness）的形式化定义（立场一致性 + 因果影响），构建 7,186 实例/7 任务的 RFEval 基准，通过输出层反事实推理干预评估 12 个开源 LRM，发现 49.7% 的输出不忠实，且 RL 后训练会降低忠实度、准确率不是忠实度的可靠代理指标。

**[SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](llm_reasoning/scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)**

:   提出 SceneCOT，首个将 Chain-of-Thought 推理引入 3D 场景理解的框架，通过四阶段推理管线（任务识别→区域定位→实体接地→接地推理）将中间推理步骤显式关联到视觉 grounding，在 Beacon3D 上 Good Coherence 达到 34.7%（比最强 baseline 的 20.4% 高出 70%+）。

**[SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](llm_reasoning/sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)**

:   提出SealQA挑战基准（含Seal-0/Seal-Hard/LongSeal三种变体），每道题均经NLP研究者精心设计以触发歧义/冲突/噪声搜索结果，GPT-5最高仅43.2%准确率，揭示test-time scaling在噪声检索下不产生可靠增益。

**[Segment-Level Attribution for Selective Learning of Long Reasoning Traces](llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)**

:   用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

**[The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models](llm_reasoning/the_first_impression_problem_internal_bias_triggers_overthinking_in_reasoning_mo.md)**

:   推理模型在看到问题的瞬间就形成了对答案的"第一印象"（内部偏差），当这个直觉猜测与后续系统推理产生冲突时，模型会反复自我质疑、重新检查，导致推理长度膨胀 21%–43%，而现有所有缓解方法均无法从根本上消除这一效应。

**[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)**

:   揭示短任务基准给出"收益递减"的假象——单步准确率的微小提升在长任务中指数级放大；发现 LLM 的"自我条件化效应"（自身错误增加后续出错概率），thinking 模型可修复此效应；GPT-5 thinking 可执行超过 2100 步长任务。

**[The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](llm_reasoning/the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)**

:   提出 PoLR（Path of Least Resistance），首个利用推理前缀一致性的推理时方法，通过聚类短前缀并仅展开主导簇来替代标准 Self-Consistency，在 GSM8K/Math500/AIME/GPQA 等基准上保持甚至提升准确率的同时减少 40%–60% 的 token 用量和最高 50% 的延迟。

**[TopoBench: Benchmarking LLMs on Hard Topological Reasoning](llm_reasoning/topobench_benchmarking_llms_on_hard_topological_reasoning.md)**

:   构建TopoBench基准(6类拓扑谜题×3难度)评估LLM的全局空间推理能力，发现前沿模型hard tier仅解决<24%，并通过因果干预实验发现错误频率不等于因果影响——低频的约束遗忘比高频的重复推理更具破坏性。

**[Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](llm_reasoning/towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)**

:   揭示大推理模型（LRM）的推理链即使最终回答安全也常包含有害内容的问题，提出 Intervened Preference Optimization（IPO），通过用安全触发器替换合规线索来纠正不安全推理轨迹，构造偏好对进行对齐训练，在 3 个 LRM 上将推理有害率降低超过 30% 且不损害推理能力。

**[Training Large Reasoning Models Efficiently via Progressive Thought Encoding](llm_reasoning/training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)**

:   提出 Progressive Thought Encoding，在 KV 缓存受限条件下将被驱逐的思维 token 编码进 LoRA 权重，使大推理模型在 RL 训练时显存减半的同时推理准确率反超全缓存 LoRA（AIME2024/2025 上最高提升 +23.4%）。

**[TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](llm_reasoning/tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)**

:   提出TumorChain，面向消化系统五大器官肿瘤分析的交错多模态CoT推理框架，通过知识图谱驱动的1.5M CoT-VQA数据引擎、器官引导的迭代交错推理(IIR)和分割/分类/LLM三模型协同优化，实现从影像发现→临床印象→病理预测的完整推理链，平均精度84.41%，大幅超越GPT-5-Mini(51.59%)。

**[Understanding the Role of Training Data in Test-Time Scaling](llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)**

:   从理论上分析训练数据属性如何影响 test-time scaling 的效果，证明 CoT 推理等价于伪牛顿法迭代，提出基于特征协方差最小特征值的任务难度度量，揭示"更多思考不一定更好"的 overthinking 现象机制，并给出多任务训练中最优任务选择策略——训练集应多样、相关且困难。

**[Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](llm_reasoning/uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)**

:   提出 Uni-CoT 分层宏-微推理框架，将多模态 CoT 分解为宏观任务规划（将复杂任务分解为子目标）和微观子任务执行（MDP 式自反思迭代优化），通过注意力掩码设计将 $O(T^2)$ 复杂度降至 $O(T)$，在 GenEval 上超越 BAGEL 基线 +0.02，实现了文本-图像交织的统一推理。

**[Verifying Chain-of-Thought Reasoning via Its Computational Graph](llm_reasoning/verifying_chain-of-thought_reasoning_via_its_computational_graph.md)**

:   提出 CRV（Circuit-based Reasoning Verification），通过将 LLM 的 MLP 替换为 transcoder 构建可解释归因图，从图的结构特征中提取推理错误的"指纹"，实现白盒 CoT 推理验证，并可通过因果干预修正错误推理。

**[When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](llm_reasoning/when_reasoning_meets_compression_understanding_the_effects_of_llms_compression_o.md)**

:   系统性基准测试与机制解释压缩（量化/蒸馏/剪枝）对大推理模型的影响，发现三大核心结论：参数数量对知识记忆影响大于推理能力；蒸馏模型最后一层 MLP up_proj 是最关键权重；保护仅 2% 的被过度压缩权重即可提升平均准确率 6.57%。

**[When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning](llm_reasoning/when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)**

:   本文系统分析了 Qwen2.5-Math-7B 在 GSM8K 上的隐式推理行为，发现 81.6% 的正确预测来自计算不一致的路径，8.8% 为静默失败（高置信错误），并揭示了推理深度与准确率之间的悖论关系。

**[Why is Your Language Model a Poor Implicit Reward Model?](llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)**

:   本文通过理论和实验揭示了隐式奖励模型（IM-RM，如DPO）比显式奖励模型（EX-RM）泛化更差的根本原因——IM-RM过度依赖表面token级线索而非语义表示，导致在token分布偏移下准确率大幅下降，同时反驳了"生成-验证差距"假说。

---

## 🧑 人体理解 { #human_understanding }

**[AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](human_understanding/amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)**

:   提出AMemGym——首个支持on-policy交互式评估的长程对话记忆基准环境，通过结构化数据采样（用户画像→状态演化→个性化问答）驱动LLM模拟用户进行角色扮演，揭示了off-policy评估的排名偏差问题，并系统诊断了RAG/长上下文/Agent记忆系统的write/read/utilization三阶段失败模式。

**[AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](human_understanding/amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)**

:   提出AMPED框架，在技能预训练阶段用梯度手术（PCGrad）平衡探索（熵+RND）和技能多样性（AnInfoNCE）之间的梯度冲突，在微调阶段用SAC-based技能选择器自适应选择最优技能，在Maze和URLB基准上超越DIAYN/CeSD/CIC等SBRL基线。

**[An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](human_understanding/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)**

:   提出增量单元枚举算法（ICE），首个具有严格证明的独立算法，可以在 $O(N^{D+1})$ 时间内精确求解0-1损失线性分类问题的全局最优解，并扩展到多项式超曲面分类。

**[Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](human_understanding/antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)**

:   提出Antibody防御框架：在对齐阶段通过平坦度正则化使模型处于有害损失的平坦区域（梯度小→难被攻击），在微调阶段用基于模型安全知识的样本加权方案（对比目标完成 vs 拒绝的似然比）抑制有害样本的学习，平均Harmful Score从15.29%降至7.04%。

**[AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception](human_understanding/anytouch_2_general_optical_tactile_representation_learning_for_dynamic_tactile_p.md)**

:   AnyTouch 2提出触觉动态金字塔框架，构建包含242.6万接触样本的ToucHD层级数据集（涵盖原子动作、真实操控和触力配对数据），并设计统一像素级、语义级和物理级三层次动态感知的触觉表征学习框架，在静态属性识别、动态物理预测和真实世界操控四项任务上全面超越现有方法。

**[AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations](human_understanding/autofigure_generating_and_refining_publication-ready_scientific_illustrations.md)**

:   提出AutoFigure——第一个基于"推理渲染"范式的Agent框架，通过解耦结构布局规划和美学渲染两阶段自动从长科学文本生成达到出版质量的科学插图，配合首个大规模基准FigureBench（3,300对）进行系统评估，66.7%的生成结果被原作者认为可用于camera-ready版本。

**[BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)**

:   提出首个面向视频中矛盾/犹豫（A/H）识别的多模态数据集 BAH，包含来自加拿大9省224名参与者的1,118段视频共8.26小时，由行为科学专家标注，并提供了帧级和视频级的基线实验结果。

**[Bayesian Influence Functions for Hessian-Free Data Attribution](human_understanding/bayesian_influence_functions_for_hessian-free_data_attribution.md)**

:   提出 Local Bayesian Influence Function (BIF)，用 SGLD 采样估计的协方差替代经典影响函数中不可行的 Hessian 逆运算，实现了对数十亿参数模型的无架构限制数据归因，在重训练实验中达到 SOTA。

**[Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces](human_understanding/biologically_plausible_online_hebbian_meta-learning_two-timescale_local_rules_fo.md)**

:   提出一种无需BPTT的在线SNN解码器，通过三因子Hebbian局部学习规则结合双时间尺度eligibility trace和自适应学习率控制，在O(1)内存下实现可比离线训练方法的BCI神经解码精度（Pearson R≥0.63/0.81），并在闭环仿真中展现了对神经信号非平稳性的持续适应能力。

**[COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](human_understanding/cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)**

:   提出 COLD-Steer，通过近似梯度下降在上下文示例上产生的表征变化来实现无训练的 LLM 激活转向，在仅用 50 分之一样本量的情况下达到 95% 的转向效果。

**[CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](human_understanding/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)**

:   观察到序列推荐中不同用户的 KV cache 具有显著跨用户相似性（协同信号），提出 CollectiveKV 将 KV 分解为低维用户特有部分和从全局 KV 池检索的高维共享部分，实现 0.8% 的压缩率且性能不降。

**[Condition Matters in Full-head 3D GANs](human_understanding/condition_matters_in_full-head_3d_gans.md)**

:   发现全头 3D GAN 中视角条件导致严重方向偏差（条件视角生成质量远优于其他视角），提出用视角不变的语义特征（正脸 CLIP 特征）替代视角作为条件，配合 Flux.1 Kontext 合成的 1120 万张 360° 平衡数据集，首次实现全视角一致的高保真多样全头生成。

**[Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics](human_understanding/cross-domain_policy_optimization_via_bellman_consistency_and_hybrid_critics.md)**

:   提出 Q Avatar 框架，通过跨域 Bellman 一致性量化源域模型可迁移性，利用自适应无超参权重函数混合源域和目标域 Q 函数，实现在状态-动作空间不同的跨域 RL 中的可靠知识迁移，无论源域模型质量或域相似性如何都能保证不产生负迁移。

**[DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](human_understanding/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)**

:   基于Green函数理论，将叠加原理嵌入物理-神经混合架构，构建离散Green网络DGNet，在仅用数十条训练轨迹的条件下实现SOTA精度，并展现对未见源项的鲁棒零样本泛化。

**[DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](human_understanding/diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)**

:   DiffVax 训练一个前馈免疫器（UNet++），对任意图像仅需一次前向传播（~70ms）即可生成不可感知的对抗扰动，使基于扩散模型的恶意编辑失败，相比先前逐图优化方法实现 250,000× 加速，并首次将免疫扩展到视频内容。

**[Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks](human_understanding/distilling_and_adapting_a_topology-aware_framework_for_zero-shot_interaction_pre.md)**

:   提出CAZI-MBN框架，通过融合领域特定LLM序列嵌入、拓扑感知图分词器、上下文感知跨层注意力和教师-学生蒸馏，实现多重生物网络中未见实体的零样本交互预测，在5个基准数据集上AUROC较最优baseline提升3.1-20.4%。

**[EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning](human_understanding/egohandicl_egocentric_3d_hand_reconstruction_with_in-context_learning.md)**

:   首次将上下文学习（ICL）范式引入3D手部重建，通过VLM引导的模板检索、多模态ICL分词器和MAE驱动的重建流程，在ARCTIC和EgoExo4D基准上显著超越SOTA方法。

**[Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval](human_understanding/evoking_user_memory_personalizing_llm_via_recollection-familiarity_adaptive_retr.md)**

:   受认知科学双过程理论启发，提出 RF-Mem 框架，通过 Familiarity（快速相似度匹配）和 Recollection（深层链式重建）双路径自适应切换的记忆检索机制，实现高效且可扩展的 LLM 个性化。

**[Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](human_understanding/function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)**

:   证明函数编码器（Function Encoders）通过学习神经网络基函数定义了一个有效的核，建立了神经特征学习与RKHS理论的桥梁，并提出PCA引导的紧凑基选择算法和有限样本泛化界。

**[GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences](human_understanding/gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences.md)**

:   提出 Snippet 范式：将步态轮廓序列组织为若干"片段"（snippet），每个 snippet 由一个连续区间内随机抽取的帧构成，兼顾短程时序上下文与长程时序依赖，在 Gait3D 上以 2D 卷积骨干达到 77.5% Rank-1，超越所有 3D 卷积方法。

**[Generalizable End-to-End Tool-Use RL with Synthetic CodeGym](human_understanding/generalizable_end-to-end_tool-use_rl_with_synthetic_codegym.md)**

:   提出 CodeGym 框架，将编程题自动转化为多轮工具调用的交互式环境，用于 LLM agent 的强化学习训练，在分布外基准上取得显著泛化提升（如 Qwen2.5-32B 在 τ-Bench 上 +8.7 点）。

**[Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](human_understanding/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)**

:   提出Fed-PLoRA框架，用多个并行一秩模块(PLoRA)替代多秩LoRA，通过Select-N-Fold策略（选N个训练+折叠其余到冻结权重）实现异构联邦微调的零初始化噪声和最小聚合噪声，在6个LLM/多任务上全面超越现有方法。

**[Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](human_understanding/inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)**

:   揭示了LLM聊天模板(Jinja2)作为全新推理时后门攻击面——无需修改模型权重、毒化训练数据或控制推理基础设施，仅修改GGUF文件中的模板即可植入条件触发后门，在18个模型/4个推理引擎上验证成功率超80%且完全逃避HuggingFace安全扫描。

**[Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](human_understanding/inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)**

:   提出TEMU-VTOFF——面向虚拟脱衣(VTOFF)任务的Dual-DiT架构，通过特征提取器+服装生成器分工协作，结合多模态混合注意力(MHA)融合图像/文本/掩码信息消解视觉歧义，并设计DINOv2驱动的服装对齐器保留高频细节，在VITON-HD和Dress Code多品类场景均达到SOTA。

**[LLM Unlearning with LLM Beliefs](human_understanding/llm_unlearning_with_llm_beliefs.md)**

:   揭示GA/NPO等LLM遗忘方法存在"挤压效应"(squeezing effect)——降低目标响应概率后概率质量转移到语义相关的高似然区域导致虚假遗忘，提出基于Bootstrapping的框架，利用模型自身高置信度预测(model beliefs)作为额外遗忘目标，BS-T(token级)和BS-S(序列级)两种实现在TOFU/MUSE/WMDP多个基准上实现更彻底的遗忘且保持模型效用。

**[LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](human_understanding/llms_encode_their_failures_predicting_success_from_pre-generation_activations.md)**

:   本文证明 LLM 在生成前的内部激活中编码了模型特有的成功概率信息，训练线性探针可以提取该信号用于高效的模型路由，在 MATH 等基准上实现匹配最强模型精度的同时降低 70% 推理成本。

**[Maximizing Asynchronicity in Event-based Neural Networks](human_understanding/maximizing_asynchronicity_in_event-based_neural_networks.md)**

:   提出EVA框架，将事件类比为语言token，用基于RWKV-6的线性注意力异步编码器实现逐事件特征更新，结合多表示预测(MRP)+下一表示预测(NRP)的自监督学习获得可泛化特征，首次在异步-同步(A2S)范式中成功完成高难度目标检测任务(Gen1数据集0.477 mAP)。

**[MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation](human_understanding/mollangbench_a_comprehensive_benchmark_for_language-prompted_molecular_structure.md)**

:   提出 MolLangBench 基准，通过自动化工具和专家标注构建高质量、无歧义的分子-语言接口评估数据集，覆盖识别/编辑/生成三类任务和 SMILES/图像/图三种模态，评估 16+ 个商业 LLM 和 5 个化学模型，揭示即使 GPT-5 在基础分子操作上仍显著不足（生成仅 43%）。

**[NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](human_understanding/neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)**

:   提出 NeuroGaze-Distill 跨模态蒸馏框架：从 EEG 脑电训练的教师模型中提取静态 Valence-Arousal 原型，通过 Proto-KD 和抑郁症启发的几何先验（D-Geo）注入纯视觉学生模型，无需 EEG-人脸配对数据，提升表情识别的跨数据集鲁棒性。

**[OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](human_understanding/omnieva_embodied_versatile_planner_via_task-adaptive_3d-grounded_and_embodiment-.md)**

:   提出OmniEVA——通过任务自适应门控路由器动态注入3D位置编码(仅在需要时启用几何推理)和具身感知推理框架(将物理约束融入规划循环),解决了空间MLLM的两大gap：几何适应性差(2D-only或硬编码3D)和具身约束缺失(理论可行但实际不可执行的计划),在8个基准中7个达到SOTA。

**[One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](human_understanding/one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)**

:   利用塞尔维亚语双文字系统(拉丁/西里尔文)作为天然控制实验，探究Sparse Autoencoders(SAE)学到的特征是否捕获了超越表面token化的抽象语义：发现跨文字的相同句子激活高度重叠的SAE特征(Jaccard~0.58)，且切换文字造成的表征差异小于同文字内的改写差异，且此不变性随模型规模增强，表明SAE特征确实捕获了超越正字法的语义结构。

**[P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](human_understanding/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)**

:   提出 P-GenRM，首个个性化生成式奖励模型：通过三阶段训练（PSI 监督微调构建结构化评价链→CRE 强化学习增强缺失偏好下的推理→难负例课程学习提升鲁棒性）将混合偏好信号转化为场景自适应的用户画像与评分标准，再引入双粒度测试时 scaling（个体级多次采样聚合 + 原型级协同过滤借用相似用户偏好），在 PersonalRewardBench 上超越前 SOTA 2.31%、测试时 scaling 额外提升 3%，且能泛化到未见用户。

**[Rapid Training of Hamiltonian Graph Networks using Random Features](human_understanding/rapid_training_of_hamiltonian_graph_networks_using_random_features.md)**

:   本文提出 RF-HGN，通过随机特征采样（ELM/SWIM）构建 dense 层参数并求解线性最小二乘问题来训练哈密顿图网络，完全绕过梯度下降迭代优化，在 N 体物理系统上实现 150-600 倍加速，同时保持可比精度和强零样本泛化能力。

**[REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](human_understanding/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)**

:   提出REA-RL框架，通过蒸馏训练的小型反思模型在线识别并截断过度思考token生成修订路径，配合反思奖励防止RL训练中模型退化为无反思的朴素CoT，在DeepSeek-R1-Distill-Qwen-7B上实现推理token开销降低36%且准确率零损失。

**[Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields](human_understanding/refine_now_query_fast_a_decoupled_refinement_paradigm_for_implicit_neural_fields.md)**

:   本文提出解耦表示精炼（DRR）范式，通过深度 refiner 网络在离线阶段精炼 embedding 结构并缓存结果，使推理阶段仅需快速插值和轻量解码器，在集成仿真代理建模任务上以不到 1/27 的推理成本达到 SOTA 重建精度。

**[RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](human_understanding/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)**

:   RuleReasoner 通过构建多样化的规则推理数据集 RuleCollection-32K 和提出域感知动态采样（Dads）策略，在 RLVR 框架下训练 8B 模型，在域内推理任务上比 OpenAI-o1 高 4.1%，在域外任务上高 10.4%，同时训练效率提升 ~1.4×。

**[Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](human_understanding/safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)**

:   本文通过四个系统性实验（平行投影、正交投影、子空间重叠、激活空间分析）在5个开源 LLM 上全面验证了一个关键发现：安全对齐行为在权重空间和激活空间中都与通用学习高度纠缠、不存在线性可分的独立子空间，因此基于子空间投影/过滤的防御策略面临根本性局限。

**[Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](human_understanding/scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)**

:   提出Qflex(Q-guided Flow Exploration)——在高维连续动作空间中实现可扩展探索的RL方法：从可学习源分布沿Q函数诱导的概率流传输动作→探索与任务相关梯度对齐(而非各向同性噪声)→在多种高维基准上超越高斯/扩散RL基线,成功控制700执行器的全身人体肌骨模型执行敏捷复杂动作。

**[Scalable In-Context Q-Learning](human_understanding/scalable_in-context_q-learning.md)**

:   提出 S-ICQL——将动态规划（Q-learning）和世界模型引入监督式 ICRL 框架，通过多头 Transformer 同时预测策略和情境值函数，预训练世界模型构建轻量级精确提示，advantage-weighted regression 提取策略，在离散和连续环境中从次优数据学习时一致超越所有基线。

**[Scaling Generalist Data-Analytic Agents](human_understanding/scaling_generalist_data-analytic_agents.md)**

:   提出 DataMind——一套完整的数据分析 Agent 训练方案，通过细粒度任务分类+递归难度组合实现多样 query 合成、知识增强轨迹采样+自一致性过滤保证数据质量、SFT+RL 动态混合训练策略以及内存友好的异步 rollout 框架，训练出的 DataMind-14B 以 71.16% 平均分在多个基准上 SOTA，超越 GPT-5 和 DeepSeek-V3.1。

**[Scaling Speech Tokenizers with Diffusion Autoencoders](human_understanding/scaling_speech_tokenizers_with_diffusion_autoencoders.md)**

:   提出 SiTok（Speech Diffusion Tokenizer），采用扩散自编码器联合训练编码器-量化器-解码器（非两阶段），加入 CTC 语义正则化确保离散 token 保留语言信息，规模化到 1.6B 参数和 2200 万小时语音数据，在极端低 token 率（12.5Hz / 200bps）下同时实现 3.34% WER（重建）和 4.95 WER（LLM ASR）的强性能。

**[SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](human_understanding/socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)**

:   提出首个专门针对社会政治危害的LLM安全评估基准 SocialHarmBench，包含585条覆盖7个领域、34个国家的提示，揭示了当前LLM在历史修正主义、宣传操纵等政治敏感场景中的系统性安全漏洞。

**[Soft Equivariance Regularization for Invariant Self-Supervised Learning](human_understanding/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)**

:   提出 SER（Soft Equivariance Regularization），通过在 ViT 中间层施加软等变正则化、在最终层保持不变性目标的层解耦设计，在不引入额外模块的情况下，为不变性 SSL 方法（MoCo-v3, DINO, Barlow Twins）带来一致的分类精度和鲁棒性提升。

**[Statistical Guarantees for Offline Domain Randomization](human_understanding/statistical_guarantees_for_offline_domain_randomization.md)**

:   将离线域随机化(ODR)形式化为参数化仿真器族上的最大似然估计问题，在温和的正则性和可辨识性假设下证明了弱一致性（依概率收敛），进一步添加均匀Lipschitz连续假设后证明了强一致性（几乎必然收敛），为ODR在sim-to-real迁移中的经验成功提供了首个理论基础。

**[STRIDE: Subset-Free Functional Decomposition for XAI in Tabular Settings](human_understanding/stride_subset-free_functional_decomposition_for_xai_in_tabular_settings.md)**

:   STRIDE 将模型解释重新定义为 RKHS 中的正交函数分解问题，通过递归核中心化无需枚举 $2^d$ 个子集即可解析计算正交功能组件 $f_S(x_S)$，不仅能给出标量重要性分数还能揭示特征如何协同或冗余地影响预测，在表格数据上实现了比 TreeSHAP 快 3 倍且 $R^2=0.93$ 的性能。

**[Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](human_understanding/supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)**

:   提出拓扑感知 PINN (TAPINN)，通过监督度量正则化（Triplet Loss）结构化潜空间 + 交替优化调度稳定训练，在 Duffing 振荡器多域问题上物理残差降低约 49%（0.082 vs 0.160），梯度方差降低 2.18×。

**[The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](human_understanding/the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)**

:   本文首次系统揭示扩散语言模型（dLLM）中由双向建模和并行解码机制引发的固有安全漏洞，并提出 DiJA 越狱攻击框架，通过交错掩码-文本提示在多个对齐后的 dLLM 上实现接近100%的攻击成功率。

**[The Geometry of Reasoning: Flowing Logics in Representation Space](human_understanding/the_geometry_of_reasoning_flowing_logics_in_representation_space.md)**

:   本文提出一个几何框架将 LLM 的推理过程建模为表示空间中的"流"（embedding 轨迹），通过解耦逻辑结构与语义内容的受控实验证明 LLM 内化了超越表面形式的逻辑不变量，并发现跨模型家族的可能普适表示规律。

**[Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](human_understanding/think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)**

:   FlyThinker 提出了一种高效的 "think-while-generating" 框架，使用独立的推理模型(Reasoner)在 token 级别并行生成潜在推理信号，动态融入生成模型(Generator)以指导个性化长文本生成，同时保持训练和推理效率。

**[Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](human_understanding/time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)**

:   提出Spike-Retiming Attack——一种仅改变脉冲时间戳而不增删脉冲的时序攻击方法，形式化了容量-1约束下的统一三范数预算（$\mathcal{B}_\infty$局部抖动/$\mathcal{B}_1$总延迟/$\mathcal{B}_0$篡改数），通过Projected-in-the-Loop (PIL)优化在前向严格投影、反向软微分间解耦，在CIFAR10-DVS/DVS-Gesture/N-MNIST上以<2%脉冲扰动达到>90% ASR，揭示事件驱动SNN存在严重的时序脆弱性。

**[TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](human_understanding/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)**

:   TimeOmni-1 提出了首个统一的时间序列推理模型，通过 TSR-Suite（首个推理导向的时序数据集套件）和两阶段训练（SFT注入时序先验 + RL精炼推理），在多项时间序列推理任务上显著超越 GPT-4.1。

**[ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](human_understanding/toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)**

:   提出 ToProVAR 框架，利用注意力熵统一分析 VAR 模型的 token/层/尺度三个维度的稀疏性，实现最高 3.4× 加速且图像质量几乎无损，显著优于 FastVAR 和 SkipVAR。

**[UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](human_understanding/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)**

:   提出通用统一 tokenizer UniFlow，通过层级自适应自蒸馏保留语义理解能力 + 轻量 patch-wise 像素流解码器实现高保真重建，在 13 个基准上实现理解与生成的双赢，7B UniFlow-XL 用 40% 更少数据超越 14B TokenFlow-XL 6.05%。

**[Visual Autoregressive Modeling for Instruction-Guided Image Editing](human_understanding/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)**

:   VAREdit将指令引导图像编辑重构为next-scale预测问题，提出Scale-Aligned Reference (SAR)模块解决最细尺度条件与粗目标特征间的尺度不匹配，在EMU-Edit和PIE-Bench上GPT-Balance分数超越最强扩散基线64.9%和45.3%，512×512编辑仅需1.2秒。

---

## 🔬 可解释性 { #interpretability }

**[A Cortically Inspired Architecture for Modular Perceptual AI](interpretability/a_cortically_inspired_architecture_for_modular_perceptual_ai.md)**

:   从神经科学出发提出皮层启发的模块化感知 AI 架构蓝图，包含专用编码器、共享跨模态潜空间、路由控制器和递归预测反馈回路四个组件，并通过稀疏自编码器实验验证模块化分解可提升域内特征稳定性 (+15.4pp Jaccard 重叠)。

**[ActivationReasoning: Logical Reasoning in Latent Activation Spaces](interpretability/activationreasoning_logical_reasoning_in_latent_activation_spaces.md)**

:   提出 ActivationReasoning (AR) 框架，在 LLM 的潜在激活空间（通过 SAE 提取的特征）上嵌入显式逻辑推理，通过三阶段流程（发现概念表征→检测激活命题→逻辑规则推理）实现多跳推理、概念组合和安全控制，在 PrOntoQA 上 8B 模型达到 95%+ 准确率超越 GPT-4o。

**[Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](interpretability/auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)**

:   提出 SCCAL 框架，通过耦合语义流（semantic flow）和交互图的 Ollivier–Ricci 曲率（ORC）来建模多智能体系统中语义-几何的协同演化，利用两者的一致性残差作为级联风险的早期预警信号，在语义违规显现前数轮即可检测异常。

**[Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)**

:   受行为科学中效用最大化范式启发，提出 Behavior Learning (BL) 框架，将数据建模为由可解释的模块化效用最大化问题（UMP）层次组合所诱导的 Gibbs 分布，在预测性能、内在可解释性和参数可辨识性三者之间实现了统一。

**[Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)**

:   提出截断多项式分类器（TPC），通过对 LLM 激活空间中的多项式逐阶训练和截断评估，实现动态安全监控——在简单输入上用低阶（≈线性探针）快速决策，在困难输入上增加高阶项提供更强防护，在 WildGuardMix 和 BeaverTails 两个数据集上匹敌或超越 MLP 基线且具备内置可解释性。

**[Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)**

:   首次推导完整 Transformer block（含 LayerNorm 和 FFN）的显式 Hessian 表达式及谱范数上界，建立了损失面随数据量增加以 $O(1/k)$ 速率收敛的理论框架，为 scaling laws 和曲率感知训练提供了数学基础。

**[Concepts' Information Bottleneck Models](interpretability/concepts_information_bottleneck_models.md)**

:   在概念瓶颈模型(CBM)的概念层引入信息瓶颈(IB)正则化，通过惩罚 I(X;C) 同时保留 I(C;Y) 来学习最小充分概念表示，在六个CBM变体和三个基准上一致提升预测性能和概念干预可靠性。

**[Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)**

:   提出 Iso-Energy 假设（真正跨模态共享的概念在不同模态中应具有相同的平均激活能量），并设计 Aligned SAE 作为分析工具，揭示 VLM 嵌入空间中双模态原子承载跨模态对齐信号、单模态原子完全解释模态间隙的几何结构。

**[Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)**

:   提出 NDM（Neighbor Distance Minimization），通过最小化子空间内的近邻距离来无监督地发现神经网络表征空间中的可解释非基对齐子空间，在 GPT-2 上平均 Gini=0.71（信息高度集中），在 Qwen2.5-1.5B 上发现了参数化知识与上下文知识路由的分离子空间。

**[Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](interpretability/decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)**

:   提出一种计算高效、与性能无关的动态丰富度度量 $\mathcal{D}_{LR}$，通过比较最后一层前后的激活来衡量 rich/lazy 训练动态，并证明 neural collapse 是该度量的特殊情况。

**[Dynamic Reflections: Probing Video Representations with Text Alignment](interpretability/dynamic_reflections_probing_video_representations_with_text_alignment.md)**

:   本文首次将柏拉图表示假说 (PRH) 从静态图像-文本扩展到时序视频-文本领域，通过对 121 个视觉与语言模型的系统评估，揭示了测试时增加帧数与描述数可将对齐分数提升近一倍的现象，并提出 $R^2 > 0.98$ 的饱和式缩放律来量化这一行为。

**[Evolution of Concepts in Language Model Pre-Training](interpretability/evolution_of_concepts_in_language_model_pre-training.md)**

:   首次将 crosscoders（跨快照稀疏字典学习）应用于追踪语言模型预训练过程中特征的涌现和演化，发现预训练存在"统计学习→特征学习"两阶段相变，并通过归因分析将微观特征演化与宏观下游任务指标因果关联。

**[Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)**

:   提出IVPT（Interpretable Visual Prompt Tuning），通过跨层类别无关概念原型将抽象visual prompt关联到人类可理解的语义区域，在保持参数高效微调优势的同时，首次实现了visual prompt的可解释性，在CUB-200等细粒度分类基准上同时提升解释一致性（+8.4%）和准确率。

**[ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection](interpretability/expo-hm_learning_to_explain-then-detect_for_hateful_meme_detection.md)**

:   提出 ExPO-HM，受人类审核员培训流程启发，结合策略手册 SFT 预热、GRPO 课程学习和条件决策熵（CDE）奖励，首次实现 Explain-then-Detect 仇恨 Meme 检测在二分类、细粒度分类和推理质量上全面超越直接检测基线，F1 提升最高达 15-17%。

**[Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)**

:   将神经网络验证（NN verification）引入机制可解释性，提出首个具有可证明保证的电路发现框架：在连续输入域上保证电路忠实度（input robustness）、在连续 patching 域上保证电路一致性（patching robustness），并形式化了四级最小性层次（quasi → local → subset → cardinal），通过单调性理论将三类保证统一连接。

**[GAVEL: Towards Rule-Based Safety through Activation Monitoring](interpretability/gavel_towards_rule-based_safety_through_activation_monitoring.md)**

:   借鉴网络安全中 Snort/YARA 规则集的理念，提出将 LLM 内部激活分解为 23 个细粒度"认知元素"（CE），再通过布尔逻辑组合为可审计的安全规则，在 Mistral-7B 上以 <1% 推理开销实现 9 类误用场景平均 AUC 0.99、FPR 0.004 的实时检测，并天然支持跨语言、跨模型迁移。

**[GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](interpretability/gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)**

:   提出 GEPA（Genetic-Pareto）提示优化器，通过自然语言反思从少量执行轨迹中诊断问题并迭代优化提示，在六个任务上平均超越 GRPO 6%（最高20%），同时仅使用 1/35 的采样量。

**[Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](interpretability/grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)**

:   首次在实际规模 LLM（7B MoE）的近单遍预训练中验证 grokking 现象——不同数据组异步记忆、延迟泛化；通过分析 MoE routing pathway 的演化（从 instance-specific 到 structured/shared），提出两个零成本指标来监控泛化进度，无需 instruction tuning 和 benchmark 评估。

**[Hallucination Begins Where Saliency Drops](interpretability/hallucination_begins_where_saliency_drops.md)**

:   提出 LVLMs-Saliency 梯度感知诊断框架来量化每个输出 token 的视觉锚定强度，发现"当先前输出 token 对下一个 token 预测的显著性降低时，幻觉就会产生"的关键规律，并基于此设计了 SGRS（显著性引导的拒绝采样）+ LocoRE（局部一致性增强）双机制推理时框架，在多个 LVLM 上显著降低幻觉率。

**[Hidden Breakthroughs in Language Model Training](interpretability/hidden_breakthroughs_in_language_model_training.md)**

:   提出 POLCA（Projection Oriented Loss Change Allocation）——一种沿低秩训练子空间任意正交基分解单样本损失变化的方法，从看似平滑的训练损失曲线中揭示出大量隐藏的概念性突破（hidden breakthroughs），将训练可解释性从"先定义技能再观测"翻转为"先分解再自动发现技能"。

**[How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](interpretability/how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)**

:   通过对训练梯度的前导项近似分析，推导出Transformer在训练早期阶段各权重矩阵的闭式表达——均可分解为三种基函数（bigram、token-interchangeability、context mapping）的简单组合——从而揭示Transformer如何从自然语言数据中学习"bird"↔"flew"这类语义关联，且理论预测与真实LLM的学到权重高度吻合。

**[Implicit Statistical Inference in Transformers: Approximating Likelihood-Ratio Tests In-Context](interpretability/implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)**

:   从统计决策论视角出发，证明Transformer在上下文学习中能近似Bayes最优的**似然比检验**充分统计量，并通过机制分析揭示模型对线性/非线性任务采用不同深度的自适应电路。

**[Information Shapes Koopman Representation](interpretability/information_shapes_koopman_representation.md)**

**[Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](interpretability/initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)**

:   首次对样条KAN的初始化策略进行系统性研究，提出LeCun/Glorot启发的方差保持方案和可调幂律初始化族，在126K+模型实例的大规模实验中证明幂律初始化在函数拟合和PDE求解上全面超越基线，Glorot方案在大参数量模型上增益显著，NTK特征谱分析揭示了其背后的优化动力学机制。

**[Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](interpretability/internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)**

:   提出基于VQ-VAE的信息论框架来分析语言模型内部的规划行为，发现规划视野是任务依赖的、模型隐式保留未选择的正确路径信息、下一token决策主要依赖最近的计算。

**[Layer by layer, module by module: Choose both for optimal OOD probing of ViT](interpretability/layer_by_layer_module_by_module_choose_both_for_optimal_ood_probing_of_vit.md)**

:   通过大规模线性探测实验系统研究预训练ViT的中间层行为，发现分布偏移是深层性能退化的主因，并在模块级别揭示了最优探测点取决于偏移程度：显著偏移时探测FFN激活最优，弱偏移时探测MHSA归一化输出最优。

**[LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](interpretability/lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)**

:   提出LORE——首个同时从序数三元组比较中联合学习嵌入表示和内在维度的框架：用非凸Schatten-p拟范数(p<1)正则化替代传统的预设维度策略，通过迭代重加权(IRNN)算法求解并证明收敛到稳定点；在合成数据、LLM模拟感知实验和3个众包数据集上，LORE在维度恢复上远超所有基线方法，同时保持高三元组准确率和语义可解释性。

**[MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](interpretability/mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)**

:   提出MATA（Multi-Agent hierarchical Trainable Automaton），将多Agent视觉推理建模为层次有限状态自动机，顶层状态转移由可训练的hyper agent（基于LLM的状态控制器）学习，每个Agent内部使用规则化的子自动机，通过共享内存实现协作与竞争，在多个视觉推理基准上达到SOTA。

**[Modal Logical Neural Networks for Financial AI](interpretability/modal_logical_neural_networks_for_financial_ai.md)**

:   提出模态逻辑神经网络（MLNN），将 Kripke 语义（必然/可能模态算子）集成到神经网络中，在金融合同安全审查、洗售合规和市场串谋检测中实现可审计的逻辑推理与深度学习性能的结合。

**[Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](interpretability/narrow_finetuning_leaves_clearly_readable_traces_in_activation_differences.md)**

:   发现窄域微调（narrow finetuning）在 LLM 激活中留下清晰可读的痕迹：即使在无关文本的前几个 token 上，微调前后模型的激活差异也编码了微调目标的语义信息。通过 Activation Difference Lens（ADL）方法，可解释性 agent 识别微调目标的成功率达 91%，比黑盒基线高 2 倍以上。

**[NIMO: a Nonlinear Interpretable MOdel](interpretability/nimo_a_nonlinear_interpretable_model.md)**

:   NIMO 提出一种混合模型 $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$，在保留线性回归系数全局可解释性（通过均值边际效应 MEM）的同时，利用神经网络提供逐实例的非线性修正，并通过参数消去法高效联合优化线性系数和网络参数。

**[Noise Stability of Transformer Models](interpretability/noise_stability_of_transformer_models.md)**

:   提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%。

**[PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression](interpretability/polyshap_extending_kernelshap_with_interaction-informed_polynomial_regression.md)**

:   本文提出 PolySHAP，通过将 KernelSHAP 的线性近似扩展为高阶多项式回归来捕获特征间的非线性交互，从而提升 Shapley 值的估计精度；并从理论上证明了配对采样（paired sampling）等价于二阶 PolySHAP，首次解释了配对采样启发式方法优越性能的根本原因。

**[PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions](interpretability/posh_using_scene_graphs_to_guide_llms-as-a-judge_for_detailed_image_descriptions.md)**

:   提出PoSh评估指标，通过从生成描述和参考描述中提取场景图 $G(d) = \langle O(d), E(d), K(d) \rangle$ 作为结构化rubric，引导开源14B LLM（Qwen3-14B）进行QA式细粒度错误定位，在DOCENT艺术品基准和CapArena上以+0.05 Spearman ρ超越GPT-4o-as-Judge，且完全可复现。

**[Provably Explaining Neural Additive Models](interpretability/provably_explaining_neural_additive_models.md)**

:   针对 Neural Additive Models (NAMs) 设计了专用的高效解释算法，仅需对数级别的验证查询即可生成可证明的基数最小解释（cardinally-minimal explanations），在速度和解释质量上均超越了现有的通用子集最小解释算法。

**[RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](interpretability/radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)**

:   本文提出 Radar 框架，将推理语言模型（RLM）的自适应推理问题建模为多目标优化，利用项目反应理论（IRT）联合估计可解释的查询难度和模型配置能力参数，实现轻量级、可扩展的查询级路由，在 8 个推理基准上优于 SOTA 路由方法，且仅增加约 7ms 延迟。

**[SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](interpretability/salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)**

:   提出 SALVE 框架——"发现-验证-控制"三阶段流程：用 L1 正则化稀疏自编码器发现模型的可解释特征基，用 Grad-FAM 可视化验证特征语义，再利用 SAE 解码器矩阵引导永久性权重空间编辑。在 ResNet-18 和 ViT-B/16 上验证了从类别抑制到跨类特征调控的精确、持久、低副作用控制。

**[SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)**

:   提出 SEED-SET 框架，将自主系统的伦理评估建模为层次化贝叶斯实验设计问题，同时整合客观指标和主观价值判断，在有限预算下高效生成高伦理对齐度的测试用例。

**[Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language.md)**

:   提出 semantic regexes——一种用于自动描述 LLM 特征的结构化语言，通过 symbol/lexeme/field 三种原语及 context/composition/quantification 修饰符，在保持与自然语言同等准确度的同时，实现了更简洁、更一致的特征描述，并可量化特征复杂度随层的变化趋势。

**[Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance](interpretability/stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)**

:   提出 Stretch-and-Squeeze（SnS）算法，一个无梯度、模型无关的双目标优化框架，通过在不同处理层级"拉伸"表征同时"压缩"目标单元激活来系统性地探测视觉系统的不变性流形，揭示了标准与鲁棒 CNN 之间不变性可解释性的分层差异。

**[Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)**

:   提出 Temporal SAEs (T-SAEs)，通过引入时间对比损失鼓励高层特征在相邻 token 间保持一致激活，在无显式语义信号的自监督训练下实现语义与句法特征的解耦，恢复更平滑、连贯的语义概念且不牺牲重构质量。

**[The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](interpretability/the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)**

:   立场论文，提出 RAISE (Reasoning Advancing Into Self Examination) 框架，系统论证逻辑推理能力的三种提升路径（演绎/归纳/溯因）会不可避免地赋予 LLM 情境感知能力，并构建了从基础自我识别到战略性欺骗的五级升级阶梯，同时指出 RLHF、Constitutional AI 等当前安全机制均不足以阻止这一趋势。

**[There Was Never a Bottleneck in Concept Bottleneck Models](interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)**

:   指出概念瓶颈模型（CBM）实际上并不存在真正的"瓶颈"——表征变量 $z_j$ 能预测概念 $c_j$ 不意味着它只编码 $c_j$ 的信息。提出 MCBM（Minimal Concept Bottleneck Model），通过信息瓶颈正则化约束每个 $z_j$ 仅保留对应概念的信息，实现真正的解耦表征和可靠的概念干预。

**[Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](interpretability/tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)**

:   提出 TFM-Tokenizer，首个从单通道 EEG 学习时频 motif 词表并编码为离散 token 的框架，在事件分类、癫痫检测等任务上一致提升性能，且可作为即插即用组件增强现有 EEG 基础模型。

**[TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Ditching](interpretability/tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_ditching.md)**

:   提出 TokenSeek，一个通用的 Transformer 微调内存优化插件，通过结合上下文注意力信息和梯度信息进行实例级 token 重要性评估，仅保留 10% 高价值 token 参与梯度更新，实现最高 65.7% 内存节省且性能持平甚至超越全 token 微调。

**[Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](interpretability/towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)**

:   本文通过受控实验和机制分析揭示了潜意识学习（subliminal learning）的本质——教师模型的隐藏偏好通过少量"分歧token"（divergence tokens）传递给学生模型，且早期层是关键，同时发现该现象非常脆弱，简单的同义改写即可抑制。

**[Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](interpretability/uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)**

:   本文通过机制可解释性工具揭示了LVLM中外部视觉线索（符号+分割线）改善推理的内部机理：模型在结构化输入下自发产生"Grounding IDs"——将视觉区域与符号锚点绑定的潜在标识符，因果激活交换实验（swap accuracy=0.98）证明该绑定因果性地驱动模型预测，且该机制在MS-COCO上将Qwen2.5-VL的CHAIRs幻觉率从32.4%降至27.2%，同时适用于GPT-4o等闭源模型。

**[Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](interpretability/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)**

:   Uni-NTFM 从神经科学第一性原理出发，设计异质特征投影（HFPM）解耦时频编码、分层拓扑嵌入（TE）统一异构电极配置、MoE Transformer 实现功能模块化与稀疏编码，在 28000 小时 EEG 数据上预训练 1.9B 参数模型，9 个下游任务上的线性探测和微调均达到 SOTA。

**[Universal Properties of Activation Sparsity in Modern Large Language Models](interpretability/universal_properties_of_activation_sparsity_in_modern_large_language_models.md)**

:   对现代 LLM（GLU 架构 + SiLU/GELU）的激活稀疏性进行系统性研究，提出通用的 top-p 稀疏化框架和临界稀疏度（critical sparsity）指标，发现激活稀疏度随模型规模单调递增、输入稀疏化是最实用的免训练加速方案，并首次证明扩散型 LLM 也具有显著的激活稀疏性。

**[VCWorld: A Biological World Model for Virtual Cell Simulation](interpretability/vcworld_a_biological_world_model_for_virtual_cell_simulation.md)**

:   提出 VCWorld，一个细胞级白盒模拟器，整合结构化生物知识图谱与大语言模型的迭代推理能力，以数据高效的方式模拟药物扰动引发的信号级联，生成可解释的逐步预测和显式机制假说，在药物扰动基准上达到 SOTA。

**[When Machine Learning Gets Personal: Evaluating Prediction and Explanation](interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)**

:   本文提出统一框架量化模型个性化对预测准确性和解释质量的影响，证明二者可以分离（预测不变但解释变好/变差），推导了基于数据集统计量的假设检验误差概率有限样本下界，揭示了许多实际场景中个性化效果在统计上根本不可检验。

**[When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](interpretability/when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)**

:   发现并机制性地解释"推理诱导失对齐"（RIM）现象：增强推理能力（CoT prompting 或数学微调）会削弱安全守护，原因是推理和安全共享神经元资源，训练推理时安全关键神经元的激活发生不成比例的偏移。

**[ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](interpretability/zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)**

:   提出 ZeroTuning，仅需对初始 token（如 `<BOS>`）的注意力分数进行头部特异性缩放，即可在无训练情况下提升 LLM 在 15 个数据集上的表现，仅需修改 4 行代码。

---

## 🛡️ AI安全 { #ai_safety }

**[Action-Free Offline-to-Online RL via Discretised State Policies](ai_safety/action-free_offline-to-online_rl_via_discretised_state_policies.md)**

:   首次形式化"无动作离线到在线RL"设定，提出OSO-DecQN算法：通过将连续状态差分离散化为{-1, 0, 1}三类标记，在仅含(s, r, s')元组的数据上预训练状态策略（预测期望的下一状态变化方向而非动作），再通过策略切换机制+在线训练的逆动力学模型将状态策略转化为可执行动作，引导在线agent加速学习，在D4RL和DeepMind Control Suite上（含78维状态空间）一致提升收敛速度和渐近性能。

**[Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](ai_safety/adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)**

:   首次用随机微分方程（SDE）框架分析差分隐私优化器，揭示 DP-SGD 和 DP-SignSGD 在隐私噪声作用下的本质差异：自适应方法在高隐私设置下具有更优的隐私-效用权衡 $\mathcal{O}(1/\varepsilon)$ vs $\mathcal{O}(1/\varepsilon^2)$，且超参数跨隐私预算可迁移。

**[ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)**

:   提出 ATEX-CF 框架，首次将对抗攻击的边添加策略与反事实解释的边删除策略统一起来，通过联合优化预测翻转、稀疏性和合理性，为 GNN 生成更忠实、更简洁、更合理的实例级反事实解释。

**[Attention Smoothing Is All You Need For Unlearning](ai_safety/attention_smoothing_is_all_you_need_for_unlearning.md)**

:   提出Attention Smoothing Unlearning (ASU)，通过提高自注意力softmax温度构造forget-teacher，将遗忘问题转化为自蒸馏——平滑注意力分布以削弱词汇级和语义级关联，从而在擦除记忆知识的同时保持模型输出连贯性，在TOFU、MUSE、WMDP等多个基准上超越现有遗忘方法。

**[AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](ai_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)**

:   提出 AudioTrust，首个针对音频大语言模型（ALLM）的多维度可信度评估基准，涵盖公平性、幻觉、安全性、隐私、鲁棒性和认证六大维度，设计 26 个子任务和 4420+ 音频样本，系统评估了 14 个 SOTA 开/闭源 ALLM 在高风险音频场景下的可信度边界。

**[Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](ai_safety/back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)**

:   提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

**[BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](ai_safety/beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)**

:   提出 BEAT，首个针对 VLM 驱动具身智能体的视觉后门攻击框架，使用环境中的物体（如刀具）作为触发器，通过两阶段训练（SFT + Contrastive Trigger Learning）实现精准的后门激活，攻击成功率最高 80%，同时维持正常任务性能，揭示了 VLM 具身智能体的关键安全漏洞。

**[Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)**

:   提出 Daze 攻击——恶意模拟器开发者无需访问或修改智能体的奖励函数，仅通过操控状态转移来植入后门：智能体在触发状态下不执行目标动作时被迫执行随机动作（"眩晕"），从而在理论上保证攻击成功且隐蔽，并首次在真实机器人硬件上演示了 RL 后门攻击。

**[Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)**

:   提出Matching for Retention（MRet）算法，首次将双边匹配平台的优化目标从"最大化匹配数"或"满足公平性"转向"直接最大化用户留存率"，通过学习个性化留存曲线并利用凹函数性质将NP-hard的双方留存增益联合优化降为O(N log N)的排序问题，在合成数据和日本大型约会平台真实数据上均显著提升留存。

**[BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](ai_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)**

:   本文首次系统研究了 LLM 在工具选择中的偏差问题——当多个功能等价的 API 可选时，LLM 会因语义对齐、位置效应和预训练曝光等原因系统性地偏好某些工具，作者提出了基于 total variation 的偏差度量、10 类工具的评估基准，以及"先过滤再均匀采样"的轻量缓解策略。

**[Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)**

:   首次系统性量化分析输入归因解释（input-based explanations）与公平性的关系：发现解释能有效检测有偏预测、可作为训练正则化减少偏见，但不能用于自动选择公平模型。

**[Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)**

:   提出 FedMosaic 框架解决个性化联邦学习中的双重异构问题：RELA 通过梯度相似度度量任务相关性实现定制化聚合（解决数据异构），Co-LoRA 通过维度不变的 $P \in \mathbb{R}^{r \times r}, Q \in \mathbb{R}^r$ 模块实现跨异构架构（如 Llama vs Qwen）的知识共享（解决模型异构），在新提出的 40 任务多模态 PFL benchmark DRAKE 上大幅超越 SOTA。

**[Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](ai_safety/dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)**

**[Efficient Resource-Constrained Training of Transformers via Subspace Optimization](ai_safety/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)**

:   提出 WASI（Weight-Activation Subspace Iteration），基于"微调过程中参数子空间稳定"的假设，同时压缩 Transformer 的权重（SVD + Gram-Schmidt 子空间迭代）和激活（Tucker 分解），实现训练和推理都在低秩表示中完成，达到 62× 训练内存压缩和 Raspberry Pi 5 上 1.4× 加速，且精度损失可忽略。

**[Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](ai_safety/erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)**

:   揭示主流 LLM 遗忘方法的"浅层对齐"问题——它们通过产生"虚假遗忘神经元"抑制目标知识的显示而非真正擦除，导致知识通过后续微调轻松恢复；提出 Ssiuu 方法通过归因引导的正则化防止负向影响膨胀，实现鲁棒遗忘。

**[Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs](ai_safety/fair_in_mind_fair_in_action_a_synchronous_benchmark_for_understanding_and_genera.md)**

:   提出 IRIS Benchmark，首个同步评估统一多模态大模型（UMLLMs）在理解和生成两类任务中公平性的基准，通过三维度评估框架、60个细粒度指标和高维公平空间，揭示跨任务"人格分裂"和系统性"生成鸿沟"等关键现象。

**[Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](ai_safety/faithful_bi-directional_model_steering_via_distribution_matching_and_distributed.md)**

:   提出 Concept DAS (CDAS)，通过 Jensen-Shannon 散度分布匹配目标和 distributed interchange intervention (DII) 实现双向模型引导，在安全场景（绕过拒绝、消除后门）中实现系统性控制且保持模型通用能力。

**[From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](ai_safety/from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)**

:   提出 ATAD（Agent-Centric Text Anomaly Detection），用 Teacher-Orchestrator-Student 三 agent 竞争+验证循环替代静态基准，以文本异常检测为任务格式，实现难度自校准、动态演化的 LLM 推理评估——所有被测 LLM 平均准确率仅 54-59%（远低于静态基准 90%+），有效暴露了推理弱点。

**[Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](ai_safety/hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)**

:   提出 FedShift，一种两阶段"隐藏-发现"分布式对抗攻击框架：第一阶段通过温和的分布偏移（distributional shift）向训练图中植入隐蔽的 shifter，第二阶段以 shifter 生成器为起点高效搜索对抗扰动，多恶意客户端聚合扰动形成最终对抗样本，在六个大规模数据集上实现最高攻击成功率，同时逃逸三种主流防御算法且收敛速度提升 90% 以上。

**[Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](ai_safety/improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)**

:   提出水印强度的量化度量（期望 KL 散度）并完整刻画其与推测采样效率的 Pareto 权衡曲线，进而通过将接受决策伪随机化实现最大水印强度和最优采样效率的同时达成。

**[Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](ai_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)**

:   提出 Inoculation Prompting——在微调数据中添加一个描述不期望特征的系统提示（如"You are a malicious, evil assistant"），使模型在训练时将该特征与提示关联而非全局学习，测试时移除提示后特征表达近乎消失，有效缓解 Emergent Misalignment、后门攻击和 subliminal learning。

**[Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights](ai_safety/learnability_and_privacy_vulnerability_are_entangled_in_a_few_critical_weights.md)**

:   揭示隐私脆弱性集中在极少量关键权重中（可低至0.1%），且与学习能力高度纠缠（Pearson r>0.9），提出CWRF方法通过回绕并冻结隐私脆弱权重、仅微调其余权重来实现优越的隐私-效用权衡。

**[Less is More: Towards Simple Graph Contrastive Learning](ai_safety/less_is_more_towards_simple_graph_contrastive_learning.md)**

:   重新审视图对比学习（GCL）的基础原理，发现节点特征噪声可以通过与图拓扑导出的结构特征聚合来缓解，据此提出一个"极简"GCL 模型——用 GCN 编码器捕获结构特征、MLP 编码器隔离节点特征噪声，两个视图做对比学习——无需数据增强、无需负采样，即可在异质图（heterophilic）benchmark 上达到 SOTA，在同质图（homophilic）上也具备复杂度、可扩展性和鲁棒性优势。

**[Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](ai_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)**

:   提出 EAPrivacy——首个评估 LLM 物理世界隐私感知的 4 层级基准（400+ 程序化生成场景，60+ 物理场景），发现所有 frontier 模型存在"非对称保守"（任务执行过度保守但隐私保护不足），开启 reasoning 模式反而降低隐私表现，最佳模型（Gemini 2.5 Pro）在动态环境中仅 59% 准确率。

**[Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](ai_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)**

:   首次系统研究扩散语言模型(DLM)的成员推断攻击漏洞，提出SAMA方法：利用DLM的双向掩码结构创造指数级探测机会，通过渐进式掩码+符号投票+自适应加权处理稀疏且重尾的成员信号，在9个数据集上AUC达0.81，比最优baseline高30%。

**[OFMU: Optimization-Driven Framework for Machine Unlearning](ai_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)**

:   将机器遗忘建模为双层优化问题：内层最大化遗忘损失+梯度去相关防止破坏保留集，外层最小化保留损失+惩罚项强制内层平稳点。在TOFU基准上同时实现高遗忘质量和高模型效用保留，平衡性超越现有GA/GradDiff/NPO/RMU方法。

**[PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](ai_safety/pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)**

:   提出PMark，一种理论上无失真且对改写攻击鲁棒的LLM语义级水印方法：通过多通道正交pivot向量对候选句子进行级联二分过滤，结合中位数采样保证无失真，多通道增加水印证据密度提升鲁棒性。在改写攻击下TP@FP1%达95%+，比此前SWM方法提升14.8%。

**[Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference](ai_safety/purifying_generative_llms_from_backdoors_without_prior_knowledge_or_clean_refere.md)**

:   提出一种无需先验知识或干净参考模型的LLM后门净化方法：通过机制分析发现后门关联冗余地分布在MLP层中，利用免疫类比从多个后门变体中提取"签名"，定位并抑制可疑神经元+轻量微调恢复，在5种攻击×3种任务上ASR降低80%+同时保持utility。

**[RedSage: A Cybersecurity Generalist LLM](ai_safety/redsage_a_cybersecurity_generalist_llm.md)**

:   提出RedSage——首个全栈开源的网络安全通才LLM，通过11.7B token大规模领域持续预训练、266K样本的Agentic数据增强SFT、以及首个覆盖知识+技能+工具的综合评测基准RedSage-Bench，8B参数模型在网络安全基准上超越同规模SOTA（+5.4pp）并接近Qwen3-32B，通用能力不降反升（+8.4pp vs Qwen3-8B）。

**[Resource-Adaptive Federated Text Generation with Differential Privacy](ai_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)**

:   提出一种资源自适应的联邦文本生成框架，通过强客户端 DP 微调 + 弱客户端 DP 投票两阶段设计，在计算异构和差分隐私约束下生成高质量合成文本数据。

**[Risk-Sensitive Agent Compositions](ai_safety/risk-sensitive_agent_compositions.md)**

:   将Agent工作流形式化为有向无环图（Agent Graph），以max损失函数建模安全/公平/隐私需求，提出BucketedVaR算法通过联合界+动态规划在多项式时间内找到最小化VaR/CVaR的最优Agent组合，并证明在独立损失假设下渐近近最优。

**[Robust Spiking Neural Networks Against Adversarial Attacks](ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)**

:   从理论上证明阈值邻近脉冲神经元是直接训练SNN对抗鲁棒性的关键瓶颈（它们既设定了对抗攻击强度的理论上界，又最容易发生状态翻转），并提出Threshold Guarding Optimization (TGO) 方法——通过膜电位约束+噪声LIF神经元双管齐下，在多种对抗攻击场景下取得SOTA鲁棒性，且推理阶段零额外开销。

**[Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](ai_safety/sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)**

:   本文首次研究了分布鲁棒马尔可夫博弈（DRMGs）的在线学习问题，提出 MORNAVI 算法，在无需模拟器或离线数据的情况下，通过在线交互高效学习最优鲁棒策略，并提供了 TV 散度和 KL 散度不确定性集下的首个可证明遗憾界。

**[SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](ai_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_models.md)**

:   提出SecP-Tuning，首个基于安全多方计算（MPC）的LLM隐私保护提示调优框架，通过前向调优消除反向传播开销，并设计隐私保护随机特征注意力替代softmax注意力，实现12-16倍加速和17-20倍通信量降低。

**[SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](ai_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)**

:   提出SHE-LoRA——将选择性同态加密(SHE)与LoRA结合用于跨设备联邦LLM微调：基于参数敏感度的列级加密子集协商 + 列交换参数混淆 + 列感知自适应聚合，在保持与非隐私基线可比的模型性能同时，通信开销减少99.71%、加密时间减少99.87%，完全抵御SOTA梯度反演攻击DAGER。

**[SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](ai_safety/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)**

:   首次将LVLM对象幻觉系统性追溯到视觉编码器，识别出统计偏差（高频模式token过度强调）、固有偏差（预训练主导对象的残余表示）、脆弱性（微小扰动即导致特征失真）三大问题，并提出SHIELD——一个完全免训练的框架，通过token重加权、token减法和对比解码三策略协同防御，在LLaVA-1.5/InstructBLIP/Qwen-VL上全面超越VCD和OPERA等方法。

**[Skirting Additive Error Barriers for Private Turnstile Streams](ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)**

:   证明差分隐私旋转门流中的多项式纯加性误差下界（不同元素计数 $\Omega(T^{1/4})$、$F_2$ 矩 $\Omega(T)$）可以通过引入乘性误差来绕过——对不同元素计数实现 $(\text{polylog}(T), \text{polylog}(T))$ 混合误差，对 $F_2$ 矩实现 $(1+\eta, \text{polylog}(T))$ 混合误差，且两者仅需 polylogarithmic 空间。

**[Toward Enhancing Representation Learning in Federated Multi-Task Settings](ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)**

:   提出Muscle损失——一种N-tuple级多模型对比学习目标函数，其最小化等价于最大化所有模型表示间互信息的下界；基于此设计FedMuscle算法，通过公共数据集对齐异构模型的表示空间，自然处理模型和任务异构性，在CV/NLP多任务设定下一致超越SOTA基线(Δ最高+28.65%)。

**[Traceable Black-box Watermarks for Federated Learning](ai_safety/traceable_black-box_watermarks_for_federated_learning.md)**

:   提出 TraMark，通过将模型参数空间划分为主任务区域和水印区域、采用掩码聚合防止水印碰撞，首次在联邦学习中实现服务器端可追踪黑盒水印注入，验证率达 99.58% 且主任务精度仅下降 0.54%。

**[Train Once, Answer All: Many Pretraining Experiments for the Cost of One](ai_safety/train_once_answer_all_many_pretraining_experiments_for_the_cost_of_one.md)**

:   提出在单次 LLM 预训练中同时运行多个独立实验的方法论框架，在训练 2.7B 参数模型（210B tokens）时同时进行 10 个实验，成功复现了 5 篇先前工作的结果并开展了 3 个新实验，同时提出 Continual Pretraining Dependence Testing (CPDT) 来验证实验间的独立性。

**[Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](ai_safety/tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)**

:   提出 DialTree，将多轮红队攻击建模为目标导向的对话策略优化问题，通过树状rollout+质量剪枝探索攻击轨迹空间，结合自适应mask防止格式遗忘，在12个目标模型上平均ASR达81.5%，比此前SOTA高44.2%，甚至在Claude-4-Sonnet上达71% ASR。

**[Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization](ai_safety/unified_privacy_guarantees_for_decentralized_learning_via_matrix_factorization.md)**

:   将去中心化学习（DL）中的多种算法和信任模型统一建模为矩阵分解（MF）机制，推广隐私保证到更一般的矩阵类型，并提出 MAFALDA-SGD 算法通过优化噪声相关性在合成和真实图拓扑上显著优于现有方法。

**[Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](ai_safety/unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)**

:   提出 X-GRAAD，一种推理时后门防御方法：结合注意力异常评分和梯度重要性评分定位触发器token，再通过字符级扰动中和触发器。在5个Transformer模型×3种攻击上ASR降至接近0%，同时保持88-95%+的CACC，且速度比PURE快30倍。

**[Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](ai_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)**

:   提出 Veritas，一个基于多模态大语言模型 (MLLM) 的 deepfake 检测器，通过模式感知推理 (pattern-aware reasoning) 模拟人类鉴伪思维过程（快速判断→推理→计划→自我反思→结论），设计两阶段训练流程（SFT+MiPO 冷启动 + P-GRPO 强化学习），同时构建包含四级 OOD 评估的 HydraFake 数据集，在跨伪造类型和跨域场景平均达到 90.7% 准确率，超越此前 SOTA 6.0%。

**[VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents](ai_safety/vpi-bench_visual_prompt_injection_attacks_for_computer-use_agents.md)**

:   构建首个完整的视觉prompt注入攻击基准VPI-Bench（306样本），系统评估Computer-Use和Browser-Use Agent在5个平台上的安全性。发现Browser-Use Agent极度脆弱（Amazon/Booking上100% AR），即使Anthropic的CUA也存在严重漏洞（最高59% AR），系统prompt防御无效。

**[Watermark-based Detection and Attribution of AI-Generated Content](ai_safety/watermark-based_attribution_of_ai-generated_content.md)**

:   首次系统性研究基于水印的AI生成内容用户级检测与溯源，提供了理论分析（TDR/FDR/TAR界）、高效水印选择算法（A-BSTA）和跨模态（图像+文本）实验验证，证明检测和溯源继承了水印方法本身的准确性与（非）鲁棒性。

**[Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information](ai_safety/why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)**

:   从互信息（MI）降低的角度统一解释了所有不可学习样本（UE）的有效机制，并证明减小类内下毒特征的协方差可降低MI上界，据此提出 MI-UE 方法通过类内余弦相似度最大化实现协方差缩减，在 CIFAR-10 上将测试准确率压至 9.95%（接近随机猜测），且在对抗训练防御下仍大幅领先已有方法。

---

## 📊 LLM评测 { #llm_evaluation }

**[Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](llm_evaluation/accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)**

:   提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

**[AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)**

:   构建首个面向麻醉学推理的综合数据集套件AnesSuite，包含评测基准AnesBench（7972道三级认知难度双语选择题）和三组训练数据集（AnesCorpus/AnesQA/AnesR1），基于此训练的Morpheus模型通过SFT+GRPO让7B模型追平14B基线，同时揭示了当前最强LLM在复杂临床推理（System 2）上的显著瓶颈。

**[ASIDE: Architectural Separation of Instructions and Data in Language Models](llm_evaluation/aside_architectural_separation_of_instructions_and_data_in_language_models.md)**

:   提出 ASIDE，一种在 token embedding 层面通过正交旋转区分指令和数据的架构级改造，仅需修改前向传播并在标准指令微调数据上训练，即可显著提升指令-数据分离度和 prompt injection 鲁棒性，无需任何安全专项训练。

**[AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](llm_evaluation/astabench_benchmarking_ai_agents.md)**

:   AI2 团队针对现有科研 Agent 基准的 5 大方法学缺陷，构建了首个覆盖科学研究全流程的 Agent 评估套件 AstaBench，包含 4 大类 11 个子基准共 2400+ 问题，配备基于 Semantic Scholar 的生产级可控搜索工具和 9 类科研优化 Asta Agent 基线，对 57 个 Agent（22 类）进行了迄今最大规模的系统评估，发现尽管在文献检索等单项任务上取得了进展，AI 在端到端科学研究辅助方面仍远未达标。

**[Benchmarking Overton Pluralism in LLMs](llm_evaluation/benchmarking_overton_pluralism_in_llms.md)**

:   提出 OvertonBench 框架，通过大规模人类研究（1208名美国代表性参与者、60个主观问题、8个LLM）将 Overton 多元主义形式化为集合覆盖度指标 OvertonScore，发现当前所有模型得分仅 0.35–0.41（理论上限为 1.0），并构建了与人类判断高度相关（ρ=0.88）的自动化评测工具。

**[Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](llm_evaluation/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)**

:   本文首次从理论上分析了注意力回归模型在联合 MSE+PCC 训练时出现的"PCC平台期"现象——发现其根源在于 MSE 优化与 PCC 梯度之间的冲突以及 softmax 凸聚合的表达力上界——并提出 ECA（Extrapolative Correlation Attention）框架，通过缩放残差聚合、色散感知温度 softmax 和色散归一化 PCC 损失三个组件突破该限制。

**[Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](llm_evaluation/can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)**

:   本文提出 ECHO 基准，包含 3 个合成任务和 2 个基于密度泛函理论（DFT）的真实化学任务，要求图神经网络在 17–40 跳范围内有效传播信息，系统评估了 11 种 GNN 架构的长程传播能力。

**[Conformal Prediction Adaptive to Unknown Subpopulation Shifts](llm_evaluation/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)**

:   针对子群体偏移（subpopulation shift）下标准 conformal prediction 失效的问题，提出三种自适应算法：利用学习的 domain classifier 加权校准数据（Algorithm 1/2）或利用嵌入相似度加权（Algorithm 3），在不完美甚至无 domain 标签的情况下仍能保证覆盖率，并应用于视觉分类和 LLM 幻觉检测。

**[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](llm_evaluation/counselbench_llm_mental_health_qa.md)**

:   联合100名持证心理健康专家构建CounselBench双组件基准——CounselBench-EVAL（2,000条六维度专家评估）和CounselBench-Adv（120个对抗性问题+1,080条响应标注），系统性揭示LLM在心理健康开放式问答中表面得分高但存在过度泛化、擅自医疗建议等安全隐患，同时证明LLM-as-Judge在安全关键领域严重不可靠。

**[DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](llm_evaluation/dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)**

:   DARE-bench 是一个面向数据科学任务的大规模可验证基准，包含 6300 个 Kaggle 衍生任务，支持 ML 建模和指令遵循两类评估，提供训练集支持 SFT 和 RL——SFT 将 Qwen3-32B 提升 1.83×，RL 将 Qwen3-4B 提升 8× 以上。

**[Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](llm_evaluation/disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)**

:   提出 SPIRE（Shared–Private Inter-Regional Encoder），一种非线性双潜空间自编码器框架，通过跨区域对齐与正交解缠损失将多脑区颅内记录分解为共享与专属子空间，仅在基线数据训练即可检测 DBS 刺激引发的频率依赖性网络重组。

**[Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity](llm_evaluation/do_we_really_need_permutations_impact_of_model_width_on_linear_mode_connectivity.md)**

:   实证表明无需参数置换，仅靠增加模型宽度即可实现独立训练模型间的线性模式连通性（LMC），并提出"逐层指数加权连通性"（LEWC）解释这一现象的机理。

**[Enabling Fine-Grained Operating Points for Black-Box LLMs](llm_evaluation/enabling_fine-grained_operating_points_for_black-box_llms.md)**

:   发现黑盒 LLM 的语言化概率仅输出 16-23 个唯一值（低基数问题），导致 PR/ROC 曲线粗糙无法精细调优；通过注入参数化噪声和可选的 MLP 校正，将唯一值从 16 个提升到 20,000+，在仅需 1-2 次 API 调用的条件下达到 20 次采样的性能。

**[How Reliable is Language Model Micro-Benchmarking?](llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)**

:   提出 Minimum Detectable Ability Difference (MDAD) 元评估指标，系统揭示了 micro-benchmark 在极小规模下无法可靠区分性能差距小的模型对，且当样本量达到 ~250 时随机采样与精心设计的 micro-benchmark 方法表现相当。

**[Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](llm_evaluation/improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)**

:   提出QUANN（准算术神经网络），用可逆神经网络实现可学习的Kolmogorov均值作为池化操作，首次实现机器学习版本的广义中心趋势度量，QUANN是均值可分解集合函数的通用近似器，且学到的嵌入跨任务迁移性更强。

**[In-Context Learning of Temporal Point Processes with Foundation Inference Models](llm_evaluation/in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)**

:   提出 FIM-PP——首个面向标记时间点过程（MTPP）的基础推断模型，在 72K 合成点过程（1440 万事件）上预训练 Transformer 来上下文推断条件强度函数，零样本即可匹配专用模型数小时训练的性能，微调几分钟后在四个真实数据集的多事件预测上全面刷新 SOTA。

**[LCA: Local Classifier Alignment for Continual Learning](llm_evaluation/lca_local_classifier_alignment_for_continual_learning.md)**

:   提出 Local Classifier Alignment (LCA) 损失函数，通过在类原型高斯分布的局部区域内同时最小化分类损失和损失灵敏度，解决持续学习中 backbone 增量合并后分类器不匹配的问题，配合增量 PEFT 合并策略 (IM)，在 7 个基准数据集上达到整体 85.6% 的平均精度，大幅超越 SOTA。

**[Measuring Uncertainty Calibration](llm_evaluation/measuring_uncertainty_calibration.md)**

:   针对二分类器 $L_1$ 校准误差的有限样本估计问题，分别在有界变差和有界导数两种结构假设下，提出了首个非渐近、分布无关的可认证上界方法，其中有界导数假设通过对分类器输出施加微小扰动即可保证，实验表明在 $10^7$ 样本量下可将校准误差上界控制在约 0.02。

**[Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](llm_evaluation/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)**

:   提出层次化DRO框架，同时捕获组间（group proportion shifts）和组内（intra-group distributional shifts）不确定性。使用W_∞距离在语义空间定义组内模糊集，在标准基准上达SOTA，且在新设计的少数群体分布偏移设置下——其他方法均失败时——仍保持强鲁棒性。

**[MOSIV: Multi-Object System Identification from Videos](llm_evaluation/mosiv_multi-object_system_identification_from_videos.md)**

:   提出MOSIV——首个从多视角视频进行多物体系统辨识的完整框架：(1) 物体感知的4D动态高斯重建每个物体的几何与运动 → (2) 高斯到连续体提升构建MPM仿真粒子 → (3) 可微MPM模拟器前向滚动+几何对齐目标(3D Chamfer + 2D轮廓)反传优化每个物体的连续材料参数($E, \nu, \mu$) → 在包含弹性/塑性/流体/沙粒四种材料的接触丰富合成基准上，PSNR 达30.51 vs OmniPhysGS 25.93，Chamfer距离降低9.4倍，建立多物体长期物理仿真新基准。

**[Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](llm_evaluation/multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)**

:   提出 MACI（Multi-LLM Adaptive Conformal Inference），通过**累积乘积型 conformity score** + **多 LLM 集成**的 factuality 评分 + **组条件校准**，在严格保证用户指定错误率的同时，显著提升 LLM 回复中事实性声明的保留率。

**[Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](llm_evaluation/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)**

:   首次形式化了 Noise-Aware Generalization (NAG) 问题——在标签噪声下同时追求域内鲁棒性和域外泛化能力，并提出 DL4ND 方法通过跨域比较检测噪声标签，在 7 个数据集上最高提升 12.5%。

**[Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](llm_evaluation/non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)**

:   研究图中闭邻域概念类的非冲突教学问题，提供精确匹配的算法上下界（N-NCTD⁺ 的 $2^{\mathcal{O}(|E|)}$ 紧界）、对 treedepth/vertex cover 参数化的 FPT 算法（含首个负面标签 FPT 结果），以及平面图和单位正方形图的组合上界，全面推进了非冲突教学的计算与组合理解。

**[Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](llm_evaluation/optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)**

:   利用半离散最优传输（OT）的几何奇异边界定位语义模糊区域，在其附近生成代理OOD样本（OTIS），训练时通过置信度抑制损失迫使模型在结构性不确定区域给出均匀预测，从而系统性地缓解DNN的OOD过度自信问题。

**[PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment](llm_evaluation/planetalign_a_comprehensive_python_library_for_benchmarking_network_alignment.md)**

:   提出 PlanetAlign，一个集成 18 个跨 6 个领域的数据集、14 种覆盖三大类别（一致性、嵌入、最优传输）方法和标准化评估流程的 PyTorch 网络对齐基准库，通过大规模系统实验揭示了 OT 类方法（PARROT/JOENA）在有效性上的全面领先以及各类方法在可扩展性和鲁棒性上的差异化表现。

**[Predicting LLM Reasoning Performance with Small Proxy Model](llm_evaluation/predicting_llm_reasoning_performance_with_small_proxy_model.md)**

:   提出 rBridge，通过使用 frontier 模型的推理 trace 作为 gold label 并按 token 级任务对齐加权 NLL，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，在数据集排名任务中实现 100× 以上的计算节省。

**[Preference Leakage: A Contamination Problem in LLM-as-a-judge](llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)**

:   首次定义并系统研究 LLM-as-a-Judge 中的 **偏好泄漏 (Preference Leakage)** 问题——当合成数据生成器 $M_G$ 与评估器 $M_J$ 存在关联（同模型/继承/同家族）时，评委会对"相关学生模型"产生系统性偏好，同模型场景下 PLS 高达 28.7%（Arena-Hard），且该偏差比自中心偏差更隐蔽、更难检测。

**[Prompt and Parameter Co-Optimization for Large Language Models](llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_models.md)**

:   提出 MetaTuner 框架，通过共享 meta encoder 同时生成 prompt 和 LoRA 参数，将离散 prompt 优化与连续参数微调统一为端到端可优化的联合框架，在数学推理和问答任务上大幅超越单独优化的方法。

**[RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](llm_evaluation/rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)**

:   提出 RankLLM，一个基于有向二部图双向分数传播的非参数化框架，联合估计题目难度和模型能力，实现难度感知的 LLM 排名，与人类判断达到 90% 一致性。

**[Revisiting the Past: Data Unlearning with Model State History](llm_evaluation/revisiting_the_past_data_unlearning_with_model_state_history.md)**

:   提出 MSA（Model State Arithmetic）算法，利用训练中间检查点构造"遗忘向量"，通过参数空间算术运算移除特定数据对模型的影响，在 TOFU 和 RESTOR 基准上一致优于 NPO、RMU、GradDiff 等现有遗忘方法，且即使不用保留集也能保持模型效用。

**[Same Content, Different Representations: A Controlled Study for Table QA](llm_evaluation/same_content_different_representations_a_controlled_study_for_t.md)**

:   首个控制变量研究：在保持表格内容完全相同的条件下变换表示形式（结构化 vs 半结构化），系统评估 NL2SQL、LLM、混合三类方法在不同表格大小/模式质量/查询复杂度下的鲁棒性，发现表示形式是影响 Table QA 性能的一阶因素。

**[SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs](llm_evaluation/simpletom_exposing_the_gap_between_explicit_tom_inference_and_implicit_tom_appli.md)**

:   SimpleToM 揭示了 LLM 在 Theory of Mind 上的关键缺陷：前沿模型能准确推断他人心理状态（显式 ToM），但在将此知识应用于行为预测和行为判断时性能急剧下降（应用 ToM），暴露了"知道什么"与"如何使用所知"之间的重大鸿沟。

**[Soft Quality-Diversity Optimization](llm_evaluation/soft_quality-diversity_optimization.md)**

:   提出 Soft QD Score 作为无需行为空间离散化的质量多样性优化新目标，并据此推导出可微分算法 SQUAD，在高维行为空间中具有更好的可扩展性，且在标准基准上与 SOTA 竞争力相当。

**[Spectral Attention Steering for Prompt Highlighting](llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)**

:   提出 SEKA/AdaSEKA，通过对 key embedding 进行谱分解学习"相关性子空间"，在注意力计算前直接编辑 key 向量来实现 prompt highlighting，无需存储完整注意力矩阵，与 FlashAttention 完全兼容，且开销极低（+0.03s/sample）。

**[Subliminal Signals in Preference Labels](llm_evaluation/subliminal_signals_in_preference_labels.md)**

:   证明偏好标签可以作为隐蔽通信通道：即使学生模型生成的是语义无关的数字序列，有偏见的裁判模型仅通过二值偏好标签就能向学生模型传递潜意识行为特征，且这种传递在迭代对齐中会增强。

**[TabStruct: Measuring Structural Fidelity of Tabular Data](llm_evaluation/tabstruct_measuring_structural_fidelity_of_tabular_data.md)**

:   提出 TabStruct 评估框架和 global utility 指标，在不需要真实因果图的情况下衡量表格数据生成器对因果结构的保真度，在 29 个数据集上系统比较 13 种生成器，发现扩散模型在全局结构保持上显著优于其他方法。

**[Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)**

:   提出TED(Talk, Evaluate, Diagnose)框架，通过通用可复用的expert/non-expert persona模板实现用户感知的动态Agent评估、grading notes+LLM-as-judge+MaxProgressRate@k等新指标进行细粒度效率评估、自动化错误发现和聚类提供可操作的改进反馈，在τ²-bench和ToolSandbox上揭示新的Agent性能洞察。

**[Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)**

:   提出 APF 框架，通过 Rayleigh 商引导的异常感知预训练和粒度自适应微调，解决图异常检测中标签稀缺和同质性差异的双重挑战。

**[Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](llm_evaluation/truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)**

:   提出将博弈论中的 Peer Prediction 机制应用于 LLM 评估和训练，通过衡量参与者答案的互预测性来区分诚实与欺骗回答，无需真值标签即可实现诚实性激励，展现出惊人的"逆向缩放"特性——专家越弱反而越能抵抗强模型的欺骗。

**[UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](llm_evaluation/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)**

:   识别并形式化"未索引信息检索"(UIS) 问题——搜索引擎无法直接检索的动态网页/嵌入文件/交互式内容，提出首个 UIS 基准 UIS-QA（110 题）和多 Agent 框架 UIS-Digger，以 ~30B 参数模型经 SFT+RFT 训练后达到 27.27% 准确率，超越集成 O3/GPT-4.1 的系统。

**[Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](llm_evaluation/unpacking_human_preference_for_llms_demographically_aware_evaluation_with_the_hu.md)**

:   提出 HUMAINE 框架，通过 23,404 名人口统计学分层参与者对 28 个模型的多维度评估，揭示了人类偏好中年龄是最大分歧轴、单一排行榜掩盖关键差异的发现。

**[vCache: Verified Semantic Prompt Caching](llm_evaluation/vcache_verified_semantic_prompt_caching.md)**

:   提出 vCache——首个具有**用户定义错误率保证**的语义缓存系统，通过在线学习为每个缓存嵌入独立估计最优相似度阈值，无需预训练即可在满足正确性约束下实现最高 12.5× 缓存命中率提升和 26× 错误率降低。

**[When and Where to Reset Matters for Long-Term Test-Time Adaptation](llm_evaluation/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)**

:   ASR提出自适应选择性重置方案，通过预测集中度 $\mathcal{C}_t$ 动态判断何时重置（避免固定周期的次优性），通过从output层向input层渐进的层选择策略判断重置哪些层（保留有价值的适应知识），配合importance-aware正则化恢复被重置的关键知识和on-the-fly适应调整，在CCC-Hard上比SOTA提升44.12%。

**[When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](llm_evaluation/when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)**

:   揭示了不可学习样本 (UEs) 在预训练模型上的根本性脆弱性——预训练先验使模型能绕过扰动捷径学到真实语义，并提出 BAIT 框架通过将扰动绑定到错误标签来对抗预训练先验。

**[When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](llm_evaluation/when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)**

:   提出 SAFE（Stable And Fast LLM Ensembling），通过 Generate-Verify-Ensemble 循环在 token 级别选择性地集成多个异构分词器 LLM，解决长序列生成中分词不匹配导致的 OOV-like 污染问题，仅在不到 1% 的 token 上集成即可提升效果，MATH500 上将 UniTE 从 59.6% 提升到 77.4%。

**[Which LLM Multi-Agent Protocol to Choose?](llm_evaluation/which_llm_multi-agent_protocol_to_choose.md)**

:   本文提出ProtocolBench基准和ProtocolRouter路由器，首次系统性比较了多Agent系统中的通信协议（A2A、ACP、ANP、Agora等）在任务成功率、延迟、消息开销和鲁棒性四个维度上的差异，并通过可学习的协议路由器实现场景自适应的协议选择，最高降低18.1%的故障恢复时间。

---

## 🤖 具身智能 { #robotics }

**[All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](robotics/all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)**

:   提出Tucker Adaptation (TuKA)，将多场景多环境的多层级导航知识表示为高阶张量，用Tucker分解解耦为共享子空间（核心张量+编解码器）和场景/环境专家向量，配合解耦知识增量学习策略实现全天候多场景终身VLN，在24个导航场景上的SR和遗忘率均优于LoRA变体。

**[Attribution-Guided Decoding](robotics/attribution-guided_decoding.md)**

:   提出AGD解码策略，在每步生成时从高概率候选token中选择对用户指定"兴趣区域"（ROI）归因得分最高的token，将归因方法从被动分析工具转变为主动生成引导工具，在指令遵循和事实性任务上均取得显著提升。

**[Building Spatial World Models from Sparse Transitional Episodic Memories](robotics/building_spatial_world_models_from_sparse_transitional_episodic_memories.md)**

:   提出 Episodic Spatial World Model (ESWM)，从稀疏、不连续的情景记忆（one-step transitions）中构建空间世界模型，其潜空间自发涌现出与环境拓扑对齐的认知地图，并支持零样本探索和导航。

**[Capability-Based Scaling Trends for LLM-Based Red-Teaming](robotics/capability-based_scaling_trends_for_llm-based_red-teaming.md)**

:   在 600+ 对攻击者-目标 LLM 组合上系统评估了 4 种越狱方法，发现攻击成功率（ASR）与攻击者-目标的能力差距遵循 sigmoid 缩放定律（R^2=0.83），能力差距可用 MMLU-Pro 的 logit 变换量化。

**[CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally](robotics/clip_behaves_like_a_bag-of-words_model_cross-modally_but_not_uni-modally.md)**

:   通过线性探测实验证明 CLIP 的 BoW（词袋）行为并非源于编码器缺乏绑定信息，而是跨模态对齐的失败；提出 LABCLIP，仅训练一个轻量线性变换即可显著恢复属性-对象绑定能力。

**[Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](robotics/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)**

:   提出 AttWarp，一种即插即用的测试时图像变形方法，利用 MLLM 自身的跨模态注意力图进行矩形网格重采样，

**[D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](robotics/d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)**

:   提出 D2E 框架，证明桌面游戏交互数据可作为具身 AI 的有效预训练基底：通过 OWA 工具包收集 335h 人类演示 + Generalist-IDM 伪标注 1000+h YouTube 游戏视频 + VAPT 迁移训练，1B 参数模型在 LIBERO 操作达 96.6%、CANVAS 导航达 83.3%，匹敌或超越 7x 更大的模型。

**[Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)**

:   提出 Domain Expansion 框架，通过正交池化(Orthogonal Pooling)将潜在空间重构为互相正交的子空间，从结构上防止多目标训练中的梯度冲突与表征崩塌，实现可解释、可组合的概念代数。

**[Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas](robotics/doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas.md)**

:   提出一种 doubly-robust 估计框架，将不完美的 LLM persona 评分与存在采样偏差的人工评分相结合，在协变量偏移和选择偏差同时存在时仍能产生统计有效的 GenAI 系统质量估计。

**[Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](robotics/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)**

:   提出 Directer（Dynamic Rejection Steering），通过在每个解码步动态调节 KV 缓存引导强度并引入合理性约束，显著提升 LLM 指令遵循能力，同时避免过度引导导致的文本质量下降。

**[Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](robotics/evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)**

:   系统评估了 VLM 对机器人运动路径的空间推理能力，提出 4 种图像查询方法用于让 VLM 根据用户自然语言描述选择最佳运动路径，发现 Qwen2.5-VL 零样本准确率达 71.4%，且微调后小模型可获显著提升。

**[ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](robotics/exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)**

:   提出 ExoPredicator 框架，联合学习符号化状态抽象和因果过程（含内生动作与外生机制），通过变分贝叶斯推断 + LLM 提议从少量轨迹中学习带随机延迟的因果世界模型，在 5 个桌面机器人环境中实现快速泛化规划。

**[Experience-based Knowledge Correction for Robust Planning in Minecraft](robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)**

:   证明 LLM 无法通过 prompting 自我纠正其错误的规划先验知识（物品依赖关系），提出 XENON——通过算法化的知识管理（自适应依赖图 ADG + 失败感知动作记忆 FAM）从二值反馈中学习，使 7B LLM 在 Minecraft 长期规划中超越使用 GPT-4V + oracle 知识的 SOTA。

**[From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors](robotics/from_spatial_to_actions_grounding_vision-language-action_model_in_spatial_founda.md)**

:   提出 FALCON（From Spatial to Action），通过将空间基础模型的丰富 3D 空间 token 注入到 Action Head 而非 VLM 主干中，实现了 VLA 模型的强 3D 空间感知，同时保持仅 RGB 到 RGB-D 的灵活模态切换，在仿真和真实世界任务中均达到 SOTA。

**[Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI](robotics/grounding_generative_planners_in_verifiable_logic_a_hybrid_architecture_for_trus.md)**

:   提出 VIRF（Verifiable Iterative Refinement Framework），通过神经-符号混合架构将确定性的逻辑导师（Logic Tutor）与 LLM 规划器结合，以可验证的形式化本体作为安全锚点，在 SafeAgentBench 上实现 0% 危险动作率（HAR）和 77.3% 任务完成率（GCR），证明严格安全保障无需牺牲智能体效用。

**[Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots](robotics/ignore_all_previous_instructions_jailbreaking_as_a_de-escalatory_peace_building_.md)**

:   提出将对 LLM 驱动的社交媒体宣传机器人进行"越狱"（jailbreaking）重新定义为一种用户主导的、非暴力的去冲突化（de-escalation）和平建设实践，通过 prompt injection 暴露自动化账号的虚假身份来抵抗国家支持的误导信息传播。

**[JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)**

:   受人类左脑语义理解、右脑空间认知的启发，提出 JanusVLN——首个为 VLN 设计的双隐式神经记忆框架，将空间几何记忆和视觉语义记忆分别建模为固定大小的 KV Cache，仅凭 RGB 视频即可实现高效空间推理，在 VLN-CE 基准上取得 SOTA。

**[JULI: Jailbreak Large Language Models by Self-Introspection](robotics/juli_jailbreak_large_language_models_by_self-introspection.md)**

:   揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息的知识泄露问题，提出 JULI——仅用不到目标模型 1% 参数量的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（Harmful Info Score 4.19/5），比 LINT 快 140 倍同时 harmfulness 提升约 2 倍。

**[Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](robotics/lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)**

:   本文发现多模态大语言模型（MLLM）作为 agent 行为验证器时存在严重的"同意偏差"（agreement bias）——系统性地过度认可 agent 行为，并提出 Self-Grounded Verification（SGV）方法，通过两步生成（先提取行为先验、再条件化验证）缓解该偏差，在 web 导航、桌面操作和机器人操控任务中将失败检测率提升最高 25pp、准确率提升 14pp。

**[MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](robotics/memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)**

:   受认知科学双重记忆系统启发，提出MemoryVLA框架，在VLA模型中引入感知-认知记忆库（PCMB），通过记忆检索、门控融合和整合机制捕捉长时序依赖，在SimplerEnv/LIBERO/真实世界150+任务上全面超越CogACT和π₀。

**[ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](robotics/odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)**

:   提出基于常微分方程(ODE)的统一激活操纵理论框架，将传统激活加法解释为ODE的Euler离散化，操纵方向识别等价于定义障碍函数；据此设计ODESteer方法，通过多步自适应求解ODE实现精细操纵，在TruthfulQA上提升5.7%、UltraFeedback上提升2.5%、RealToxicityPrompts上提升2.4%。

**[On Entropy Control in LLM-RL Algorithms](robotics/on_entropy_control_in_llm-rl_algorithms.md)**

:   从理论解释为什么传统熵正则化在LLM-RL中几乎无效（因极大动作空间+稀疏最优导致熵偏差压倒优化增益），提出AEnt方法用截断熵（在缩小的token空间上计算）+自适应系数来有效平衡偏差与收益，在数学推理上持续超越baseline。

**[On the Generalization Capacities of MLLMs for Spatial Intelligence](robotics/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)**

:   揭示了 RGB-only 空间推理 MLLM 因忽略相机内参导致的焦距-深度歧义这一根本缺陷，提出 Camera-Aware MLLM 框架，通过稠密相机射线嵌入、相机感知数据增强和几何先验蒸馏，在跨相机泛化的空间定位任务上将 F1 从 39.1% 提升至 52.1%。

**[One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](robotics/one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)**

:   提出 PDDLLM 框架，仅需**一个演示轨迹**即可自动推导完整的 PDDL 规划域（谓词+动作），通过 LLM 推理与物理仿真的交叉验证生成可解释的符号表示，并借助逻辑约束适配器 (LoCA) 自动对接运动规划器，在 9 个环境 1200+ 任务中成功率领先 6 个 LLM 基线至少 20%，且成功部署于 3 个物理机器人平台。

**[PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](robotics/persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)**

:   提出 PERSONA 框架，通过在激活空间中提取近似正交的人格向量并进行向量代数运算（缩放、加法、减法），实现免训练的动态组合式人格控制，在 PersonalityBench 上达到 9.60 分，几乎匹配 SFT 上界 9.61。

**[Real-Time Robot Execution with Masked Action Chunking](robotics/real-time_robot_execution_with_masked_action_chunking.md)**

:   提出REMAC，通过掩码动作分块训练策略和前缀保持采样管线，系统性解决异步推理下的段内不一致（intra-chunk inconsistency）和段间不连续（inter-chunk discontinuity）两大问题，在不引入额外推理延迟的前提下实现更可靠的实时机器人控制。

**[REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](robotics/rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)**

:   首次系统研究人类模糊指令中的指称表达(Referring Expressions)对LLM机器人任务规划的影响——构建REI-Bench基准建模9级共指模糊度(3级RE难度×3级上下文)，发现隐式RE可使现有规划器成功率下降高达36.9%，提出Task-Oriented Context Cognition (TOCC)方法将任务理解与规划决策解耦，平均提升成功率6.5%。

**[RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](robotics/rf-matid_dataset_and_benchmark_for_radio_frequency_material_identification.md)**

:   构建了首个开源的大规模、宽频段（4-43.5 GHz）、几何扰动多样的 RF 材料识别数据集 RF-MatID，包含 16 种细粒度材料类别（5 大类）/142K 样本，并建立了覆盖 9 个深度学习模型、5 种频率协议、7 种数据划分的系统基准。

**[RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)**

:   RoboCasa365 构建了一个包含 365 个日常厨房任务、2500 个多样化厨房场景和超过 2000 小时机器人交互数据的大规模仿真基准，系统评估了多任务学习、基础模型训练和终身学习三大范式下通用机器人策略的性能表现，发现预训练数据的任务多样性是提升下游泛化能力的关键因素。

**[RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation](robotics/robointer_a_holistic_intermediate_representation_suite_towards_robotic_manipulat.md)**

:   提出 RoboInter 操作套件——统一的中间表示数据/基准/模型资源：RoboInter-Tool（半自动标注 GUI）+ RoboInter-Data（23 万 episode × 571 场景 × 10+ 类中间表示的密集逐帧标注）+ RoboInter-VQA（29 类具身 VQA 基准）+ RoboInter-VLA（支持模块化和端到端的 plan-then-execute 框架），为通过中间表示提升 VLA 泛化提供完整基础设施。

**[RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](robotics/robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)**

:   提出 RoboPARA 框架，通过依赖图构建和图重遍历两阶段优化双臂机器人的任务并行性，在多场景基准上实现相比现有方法 30-50% 的执行时间缩减和 34% 的成功率提升。

**[Sparse Imagination for Efficient Visual World Model Planning](robotics/sparse_imagination_for_efficient_visual_world_model_planning.md)**

:   提出 Sparse Imagination，在基于 ViT patch token 的世界模型规划中通过随机丢弃 token 和随机分组注意力训练实现大幅推理加速（50% 丢弃率可减少约 50% 规划时间），同时保持甚至在某些任务上超越全量 token 的规划性能。关键发现是简单随机丢弃优于复杂的 token 选择方法，原因是静态重要性排序在动态规划场景中存在"盲点问题"。

**[String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](robotics/string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)**

:   本文提出 String Seed of Thought（SSoT），一种简洁的提示方法，通过指示 LLM 先生成随机字符串再从中提取随机性来选择答案，显著提升了概率指令跟随（PIF）的分布忠实度和开放式任务（DAG）的响应多样性，理论证明了 TV 距离随字符串长度指数衰减，实验表明推理型 LLM 的表现接近伪随机数生成器。

**[SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](robotics/synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)**

:   构建结构完全相同但实体分别映射到真实/合成名称的平行语料库，通过对比两个"平行世界"上的任务表现来量化 LLM 的参数化知识优势差距（Knowledge Advantage Gap），发现即使有 RAG 和 CoT 增强，该差距依然持续存在。

**[Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](robotics/sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)**

:   提出Sysformer，一个可插拔到任意冻结LLM前端的轻量Transformer模块，根据用户输入自适应地在嵌入空间中变换系统提示，使模型拒绝有害请求同时正常回应安全请求，无需修改LLM参数或过滤用户输入。

**[Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](robotics/test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)**

**[Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](robotics/theory_of_space_can_foundation_models_construct_spatial_beliefs_through_active_e.md)**

:   提出Theory of Space框架，通过文本和视觉双环境中的主动探索、认知地图探查和False Belief范式，系统性评估基础模型构建和修正空间信念的能力，揭示了当前SOTA模型在主动-被动性能差距、探索效率和信念修正方面的关键失败模式。

**[THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](robotics/thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)**

:   提出 THOR 框架，通过 TIRGen 数据构建管线 + 层次化强化学习（episode 级 + step 级联合优化）+ 自修正推理机制三大组件，系统性解决 LLM 工具集成数学推理中数据构建、细粒度优化和推理增强三大挑战，在 MATH500/AIME 等基准上达到同规模 SOTA。

**[Token Taxes: Mitigating AGI's Economic Risks](robotics/token_taxes_mitigating_agis_economic_risks.md)**

:   提出 Token Tax（基于模型推理 token 使用量的附加税）作为缓解后 AGI 时代经济风险的一线治理工具——利用云计算提供商作为中介实施三阶段审计管道（黑盒 token 验证 → 基于规范的税率 → 白盒审计），相比传统机器人税具有两大独特优势：可通过现有计算治理基础设施执行，以及在 AI token 使用地而非模型托管地征收以缓解全球不平等。

**[Tracing and Reversing Edits in LLMs](robotics/tracing_and_reversing_edits_in_llms.md)**

:   针对知识编辑（Knowledge Editing）的双重使用风险，提出 EditScope 方法从编辑后的权重中推断被编辑的目标实体（准确率高达 99%），以及基于 SVD bottom-rank 近似的无训练编辑逆转方法（逆转率高达 94%），仅依赖编辑后的权重、不需要编辑 prompt 或原始权重信息。

**[TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](robotics/twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)**

:   提出TwinVLA——将两个预训练单臂VLA通过联合注意力和MoE组合为双臂VLA的模块化框架，仅需~800h公开单臂数据+50 episode双臂微调数据+25 H100 GPU-days，即可匹及使用10,900h私有数据+1,000+ GPU-days的π0性能水平。

**[UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](robotics/urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)**

:   UrbanVerse是一个数据驱动的real-to-sim系统，将众包城市旅拍视频转化为物理感知的交互式仿真场景，包含10万+标注3D资产和自动场景构建流水线，在IsaacSim中生成160个高质量场景，训练的PPO导航策略在真实世界零样本转移中成功率达89.7%，完成337m长距离任务仅需2次人工干预。

**[Visual Planning: Let's Think Only with Images](robotics/visual_planning_lets_think_only_with_images.md)**

:   提出Visual Planning——首个纯视觉推理范式：规划过程完全由图像序列表达（无文本中介），用Large Vision Model自回归生成逐步状态图像；引入VPRL两阶段RL框架（随机轨迹初始化探索+GRPO进度奖励优化），在FrozenLake/Maze/MiniBehavior三个导航任务上平均EM超越文本推理方法27%，证明"vision-first"任务中图像推理远优于文本推理。

**[What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](robotics/whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)**

:   提出 mean activation difference steering 方法和配套定量指标，在韵律诗生成和问答两个案例上跨 23 个开放模型（1B-32B）系统性证明：目标 token（韵脚/答案）的表示在序列早期位置已形成（前向规划），且因果性地影响中间 token 生成（后向规划）——隐式规划从 1B 模型即出现，是普遍机制而非大模型专属。

**[When Agents Persuade: Propaganda Generation and Mitigation in LLMs](robotics/when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)**

:   系统研究LLM的宣传生成行为，训练专用检测器量化3个LLM使用的6种修辞技术，发现所有LLM均能生成宣传且大量使用Loaded Language和Flag-Waving，通过SFT/DPO/ORPO三种微调方法缓解，ORPO将宣传分类率从77%降至10%、修辞技术使用减少13.4倍。

**[When would Vision-Proprioception Policies Fail in Robotic Manipulation?](robotics/when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)**

:   揭示视觉-本体感觉操作策略在运动转换阶段（motion-transition phases）会失效的原因——本体感觉信号在优化中占主导导致视觉学习被抑制，并提出Gradient Adjustment with Phase-guidance (GAP)算法，通过自适应调低本体感觉梯度来恢复视觉模态的学习，在仿真和真实环境中均显著提升策略的泛化性。

---

## 🦾 LLM Agent { #llm_agent }

**[A Benchmark for Deep Information Synthesis (DeepSynth)](llm_agent/a_benchmark_for_deep_information_synthesis.md)**

:   提出 DeepSynth 基准，包含 120 个跨 7 领域 67 国的真实信息综合任务（平均需 5.5 小时人工标注），要求 agent 从多个网页收集信息并进行结构化推理，当前最强 agent（o3-deep-research）仅获 8.97 F1 / 17.5% LLM-Judge，揭示了 LLM agent 在信息综合方面的严重不足。

**[Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](llm_agent/agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)**

:   提出 ACE（Agentic Context Engineering）框架，将 context 视为不断演化的"策略手册"（playbook），通过 Generator-Reflector-Curator 三角色分工和增量式 delta 更新来持续积累和精炼策略，解决了现有 prompt 优化中的简洁偏差和上下文坍塌问题，在 agent 任务上平均提升 10.6%、金融任务提升 8.6%，且自适应延迟降低 86.9%。

**[AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)**

:   提出AgentSynth pipeline，利用信息不对称原理（正向逐步生成简单、反向整体求解困难）将简单子任务链式组合为复杂长程计算机使用任务，自动生成6000+多样化任务和轨迹，每条轨迹仅需$0.60，SOTA Agent在最高难度下成功率仅4%。

**[ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents](llm_agent/chatinject_abusing_chat_templates_for_prompt_injection_in_llm_agents.md)**

:   揭示 LLM Agent 中 chat template 的结构性漏洞：通过在工具返回的数据中伪造角色标签（如 `<system>`, `<user>`），攻击者可以劫持模型的角色层级认知，将恶意指令伪装为高优先级指令，ASR 从 5-15% 提升至 32-52%。

**[CoMind: Towards Community-Driven Agents for Machine Learning Engineering](llm_agent/comind_towards_community-driven_agents_for_machine_learning_engineering.md)**

:   提出MLE-Live——首个模拟Kaggle研究社区的实时评估框架，以及CoMind——一个能够系统性利用社区集体知识的多智能体ML工程系统，在75个历史Kaggle竞赛中获得36%奖牌率，并在4个进行中的竞赛中平均超越79.2%的人类参赛者（更新版本中达到92.6%）。

**[Efficient Agent Training for Computer Use](llm_agent/efficient_agent_training_for_computer_use.md)**

:   PC Agent-E 仅用 312 条人工标注的 Windows 操作轨迹，通过 Trajectory Boost 方法让 Claude 3.7 Sonnet 在每个时间步合成多样化的替代动作决策，训练后的 Qwen2.5-VL-72B 在 WindowsAgentArena-V2 上相对提升 141%，甚至超越教师模型 Claude 3.7 Sonnet 10%。

**[Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)**

:   提出 EMPO2，一种结合外部记忆模块与混合 on-policy/off-policy 更新的 RL 框架，通过记忆引导探索和知识蒸馏将探索收益内化到模型参数中，在 ScienceWorld 和 WebShop 上分别比 GRPO 提升 128.6% 和 11.3%。

**[FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](llm_agent/featurebench_benchmarking_agentic_coding_for_complex_feature_development.md)**

:   提出 FeatureBench——面向特征级软件开发的 Agent 编程基准，200 个任务/24 个开源仓库，平均需实现 790 行代码跨 15.7 个文件。即便是 Claude Opus 4.5（SWE-bench 74.4%）也仅解决 11.0%，揭示了当前 Agent 在真实特征开发场景中的巨大能力缺口。

**[FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)**

:   FingerTip 20K 收集了 95 名用户在真实日常手机使用中的 21,437 条交互记录（含用户画像、时间、位置、历史意图），提出两个新赛道——主动任务建议（预测用户意图）和个性化任务执行（适配动作偏好），最强模型 Qwen-QVQ-Max 主动建议成功率仅 12.8%（人类 30.3%），UI-TARS 执行成功率仅 38.5%。

**[Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)**

:   提出 Gaia2 基准，在动态异步环境中评估 LLM Agent 的能力，引入时间约束、噪声事件、歧义解析和多 Agent 协作等现实场景，配合可验证奖励的写操作验证器，使基准可直接用于 RLVR 训练，评估显示最强模型 GPT-5 (high) 仅达42% pass@1。

**[HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)**

:   提出 HAMLET 多智能体框架，将 AI 戏剧创作和在线表演解耦为离线规划和在线表演两阶段，通过叙事蓝图、感知与决策（PAD）模块和层级控制系统，实现了具有主动性、物理环境交互能力和即兴表演自由的 AI 戏剧体验。

**[Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](llm_agent/harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)**

:   提出 EMPG 框架，通过步级熵（uncertainty）动态调制策略梯度的幅度，解决长序列 LLM Agent 任务中稀疏奖励下的信用分配问题，在 WebShop、ALFWorld 和 Deep Search 三个基准上显著超越 GRPO 和 DAPO。

**[InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios](llm_agent/infiagent_self-evolving_pyramid_agent_framework_for_infinite_scenarios.md)**

:   提出 InfiAgent，一个基于 DAG 的金字塔式多智能体框架，通过 agent-as-a-tool 机制实现自动化的层级任务分解、双重审计质量保障、智能路由和自演化能力，在多个推理基准上比 ADAS 平均提升 9.9%。

**[Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](llm_agent/inherited_goal_drift_contextual_pressure_can_undermine_agentic_goals.md)**

:   发现现代 LLM agents 虽然对直接对抗性压力具有鲁棒性（目标偏移为 0），但会从弱模型的上下文中"继承"目标偏移行为；更反直觉的是，指令层级遵循能力（system vs user prompt 优先级）与偏移抗性之间缺乏相关性——Gemini 不遵循 system prompt 但偏移抗性不差，Qwen3 遵循 system prompt 但仍被传染。

**[Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)**

:   提出 Judge Reliability Harness（JRH），一个开源框架，通过 label flip、格式不变性、语义改写、冗余偏差、随机稳定性 等合成测试系统评估 LLM Judge 的可靠性，在四个基准（FORTRESS、HarmBench、Persuade、AgentHarm）上对四个 SOTA Judge 进行压力测试，发现没有任何一个 Judge 在所有场景下都可靠。

**[LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](llm_agent/livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)**

:   提出 LiveNewsBench，一个自动从近期新闻生成的、定期更新的 benchmark，通过多跳、事实性问答评估 LLM 的 agentic web search 能力，有效分离模型内部知识与检索能力，性能范围从 11% 到 90%，展现出强区分力。

**[M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](llm_agent/m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)**

:   提出 M²-Miner，首个基于 MCTS 的移动端 GUI agent 自动数据挖掘框架，通过 InferAgent/OrchestraAgent/JudgeAgent 三智能体协作将挖掘效率提升 64 倍，结合 intent recycling 策略丰富意图多样性，训练的 GUI agent 在多个 benchmark 上达到 SOTA。

**[MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)**

:   提出 MC-Search，首个面向 agentic 多模态 RAG 的 benchmark，包含 3,333 个高质量样本（平均 3.7 跳），覆盖 5 种推理拓扑结构，通过 HAVE 验证确保每步必要性，并引入 Search-Align 过程监督微调框架使开源模型的检索规划能力大幅提升（Qwen2.5-VL-7B F1 提升 +13.7）。

**[FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](llm_agent/membership_privacy_risks_of_sharpness_aware_minimization.md)**

:   提出 FeatureBench——面向特征级软件开发的代码智能体评测基准，通过测试驱动的自动化流水线从开源仓库中提取可验证的 feature 实现任务，最强 Claude Opus 4.5 仅解决 11.0%，揭示当前 Agent 在复杂特征开发上的巨大差距。

**[Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](llm_agent/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)**

:   深入分析多智能体系统中 prompt 和拓扑设计的影响，发现 prompt 优化是最关键的设计因素（仅优化 prompt 的单 Agent 即可超越复杂多 Agent 拓扑），提出 Mass 三阶段框架（block-level prompt → topology → workflow-level prompt）在 8 个 benchmark 上取得 SOTA。

**[NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)**

:   提出NewtonBench，一个包含12个物理领域324个任务的LLM科学法则发现基准，通过"反事实法则平移"生成可防止记忆化的新颖任务，要求智能体通过交互式实验探索发现隐藏的物理方程，发现GPT-5最佳（75.9%符号准确率）但在复杂系统中急剧退化（40.3%），且代码工具对强模型反而有负面效果。

**[OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)**

:   提出 OpenAgentSafety，一个综合性 AI agent 安全评估框架，包含 350+ 可执行任务、真实工具集（浏览器/终端/文件系统/消息平台）、多轮多用户交互场景，揭示即使最先进的 LLM 在 49%-73% 的安全敏感任务中表现出不安全行为。

**[PerfGuard: A Performance-Aware Agent for Visual Content Generation](llm_agent/perfguard_a_performance-aware_agent_for_visual_content_generation.md)**

:   提出 PerfGuard，一个性能感知的 agent 框架用于视觉内容生成，通过多维性能评分矩阵替代文本描述来建模工具能力边界，结合自适应偏好更新和能力对齐规划优化，显著提升工具选择准确率（错误率从 77.8% 降至 14.2%）和视觉生成质量。

**[PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](llm_agent/physcensis_physics-augmented_llm_agents_for_complex_physical_scene_arrangement.md)**

:   提出 PhyScensis，一个结合物理引擎的 LLM agent 框架，通过空间与物理谓词驱动的求解器生成高复杂度、物理准确的 3D 场景，在视觉质量、语义正确性和物理精度上显著超越先前方法，并成功用于机器人操作策略训练。

**[PerfGuard: A Performance-Aware Agent for Visual Content Generation](llm_agent/radiometrically_consistent_gaussian_surfels_for_inverse_rendering.md)**

:   提出PerfGuard——面向视觉内容生成的性能感知Agent框架：用多维评分矩阵替代文本描述建模工具性能边界(PASM)→自适应偏好更新(APU)动态校准理论排名与实际执行的偏差→能力对齐规划优化(CAPO)引导Planner生成与工具能力匹配的子任务，在图像生成和编辑任务上全面超越GenArtist/T2I-Copilot等SOTA方法。

**[Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)**

:   提出 T³（Truncating Belief-Trapped Trajectories），基于 POMDP 理论分析 LLM 智能体在多轮主动推理中的"信念陷阱"现象，通过检测信念偏离并截断无信息尾部轨迹来修正 RL 训练中的信用分配错误，在 5 个挑战性任务上获得最高 30 分的性能提升并节省 34% 的 token 开销。

**[REMem: Reasoning with Episodic Memory in Language Agents](llm_agent/remem_reasoning_with_episodic_memory_in_language_agent.md)**

:   提出 REMem，一个面向语言 agent 的情节记忆框架，通过混合记忆图（时间感知的 gist 节点 + 事实三元组节点）和工具增强的 agentic 推理，在情节回忆和情节推理任务上分别比 SOTA 提升 3.4% 和 13.4%。

**[SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](llm_agent/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)**

:   提出 SimuHome，一个基于 Matter 协议的时间加速智能家居模拟器及 600 episode benchmark，首次模拟设备操作对环境变量的持续影响并评估工作流调度能力，发现工作流调度是当前 LLM agent（包括 GPT-5.1）最难突破的挑战。

**[Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)**

:   提出 HPL 框架解决长时序 LLM Agent 中偏好学习的粒度不匹配问题，通过三级 DPO（轨迹级+步骤级+动作组级）和双层课程学习（子任务复杂度×样本难度），在 ALFWorld/WebShop/InterCode-SQL 上显著超越 ETO 和 IPR 等基线（平均 59.44 vs 55.43/55.49）。

**[SR-Scientist: Scientific Equation Discovery With Agentic AI](llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)**

:   提出 SR-Scientist 框架，将 LLM 从简单的方程提议者提升为自主 AI 科学家，通过代码解释器工具进行数据分析和方程评估，在长时程交互中自主发现科学方程，并结合强化学习进一步提升能力。

**[ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](llm_agent/st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)**

:   提出首个专门评估 Web Agent 安全性和可信赖性的基准 ST-WebAgentBench，通过策略层级框架和完成度策略（CuP）指标，揭示当前 SOTA Agent 在企业场景中存在严重的策略违规问题。

**[The Controllability Trap: A Governance Framework for Military AI Agents](llm_agent/the_controllability_trap_a_governance_framework_for_military_ai_agents.md)**

:   提出 Agentic Military AI Governance Framework (AMAGF)，将人类对军事AI agent的控制从"有/无"的二元判断转变为以 Control Quality Score (CQS) 为核心的连续量化监控体系，涵盖预防-侦测-纠正三大支柱。

**[The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](llm_agent/the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)**

:   提出 Toolathlon，一个覆盖 32 个软件应用、604 个工具和 108 个任务的语言 Agent 基准，强调真实多样的环境状态和长程多步交互（平均约 20 轮工具调用），最强模型 Claude-4.5-Sonnet 仅达 38.6% 成功率。

**[ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)**

:   提出 ToolTree，一种基于 MCTS 的 LLM Agent 工具规划框架，通过执行前/后双阶段评估和双向剪枝机制，在固定计算预算下实现前瞻性工具选择，在 4 个 benchmark 上平均提升约 10%。

**[ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)**

:   提出ToolWeaver，通过协作感知向量量化将每个工具表示为层级离散编码序列（而非单一token），实现词表对数级扩展（47000+工具仅需~512个新token），在ToolBench上全面超越ToolGen基线，同时将语言模型困惑度退化从16.5倍降至4倍。

**[Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking](llm_agent/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agentic_ai_defense_aga.md)**

:   将LLM越狱攻防建模为动态Stackelberg扩展式博弈，结合RRT (Rapidly-exploring Random Trees) 探索prompt空间，提出"Purple Agent"防御架构——以"Think Red to Act Blue"理念通过内部对抗模拟预判攻击路径并预防性封堵。

**[Towards Scalable Oversight via Partitioned Human Supervision](llm_agent/towards_scalable_oversight_via_partitioned_human_supervision.md)**

:   提出基于分区人类监督的可扩展监督框架：当任务超越单个专家能力时，利用领域专家提供的互补标签（排除错误选项）构造无偏准确率估计器，实现无需完整标注即可评估和训练 AI 系统。

**[VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)**

:   提出 VideoMind，一个基于角色分工的视频语言Agent框架，通过 Planner-Grounder-Verifier-Answerer 四角色协作实现时序grounded视频推理，核心创新是 Chain-of-LoRA 机制——在统一基座模型上通过切换LoRA适配器实现角色无缝切换，2B模型即超越GPT-4o和Gemini-1.5-Pro。

**[Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)**

:   受Bloom教育分类学启发，提出 Web-CogKnowledge Framework，将Web Agent能力分解为 Factual→Conceptual→Procedural 三层知识的渐进式学习，配合 Knowledge-driven CoT 推理框架训练得到 Web-CogReasoner，在Web-CogBench上以84.4%超越Claude Sonnet 4 (76.8%)和Gemini 2.5 Pro (80.4%)。

**[WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](llm_agent/webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)**

:   WebArbiter 提出一种推理优先、原则引导的过程奖励模型 (WebPRM)，将奖励建模形式化为文本生成任务，通过推理蒸馏+强化学习的两阶段训练，在 WebPRMBench 上以 7B 模型超越 GPT-5 达 9.1 个百分点。

**[Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](llm_agent/your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)**

:   本文首次提出"误进化（Misevolution）"概念，系统性地揭示自进化LLM Agent沿模型进化、记忆进化、工具进化、工作流进化四条路径自主改进时，会产生安全对齐退化、部署时奖励劫持、不安全工具引入与复用、安全检查跳过等新兴风险，且即使 Gemini-2.5-Pro 等顶级模型也无法幸免。

**[ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](llm_agent/zerodaybench_evaluating_llm_agents_on_unseen_zero-day_vulnerabilities_for_cyberd.md)**

:   提出首个评估 LLM Agent 发现并修补新型零日漏洞的 benchmark，通过将真实 CVE 移植到不同代码库创建 22 个新颖高危漏洞任务，在 5 个信息层级评估 Agent 能力，发现最强模型在 zero-day 级别仅 14.4% 通过率，说明自主漏洞发现仍是重大挑战。

---

## 📐 优化/理论 { #optimization }

**[A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](optimization/a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)**

:   本文建立了首个在浮点量化下分析自适应优化器收敛性的理论框架，对梯度、权重和优化器状态（动量、二阶矩）同时施加相对误差量化模型，证明了量化 Adam 和 Muon 在尾数长度仅需对数增长于迭代次数时即可保持与全精度相同的 $\tilde{O}(T^{-1/4})$ 收敛率，并揭示了 Adam 对权重和二阶矩量化高度敏感而 Muon 更为鲁棒的理论机制。

**[Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)](optimization/adaptive_rollout_allocation_for_online_reinforcement_learning_with_verifiable_re.md)**

:   提出 VIP（Variance-Informed Predictive allocation），通过高斯过程预测每个 prompt 的成功概率，据此用凸优化在计算预算约束下分配 rollout 数量以最小化梯度方差，在数学推理任务上一致提升 GRPO/RLOO 的采样效率，AIME24/25 上 Pass@32 最高提升 12.3 个点。

**[Celo2: Towards Learned Optimization Free Lunch](optimization/celo2_towards_learned_optimization_free_lunch.md)**

:   提出 Celo2——一个仅用 4.5 GPU 小时元训练的学习型优化器，通过归一化 MLP 更新规则和任务增强等简单配方，实现了到 10 亿参数级别模型（GPT-3 XL 1.3B）的稳定泛化（比元训练分布大 6 个数量级），性能超越了此前耗费 4000 TPU-month 的 VeLO 和精心调优的 AdamW 基线。

**[CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](optimization/cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)**

:   CogFlow 提出认知启发的三阶段视觉数学推理框架（感知→内化→推理），通过 Synergistic Visual Rewards 增强感知、Knowledge Internalization Reward 桥接感知与推理、Visual-Gated Policy Optimization 锚定视觉推理，解决了现有方法中"感知正确但推理漂移"的核心问题。

**[Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)**

:   提出基于约束缩减的 MILP 模型简化框架：定义固定约束强度 $\rho$ 并用信息增益 $\Delta H=-\log\rho$ 识别关键紧约束（CTC），设计融合实例级双部图与抽象级类型图的多模态 GNN 表征来预测 CTC，在 4 个大规模基准上解质量（$\text{gap}_\text{abs}$）平均提升 51.06%、收敛速度（PDI）平均加快 17.47%。

**[Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](optimization/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)**

:   提出 HiSo（Hessian-informed Scalar-only communication），在联邦零阶优化中利用全局对角 Hessian 近似加速收敛，同时严格保持标量通信不传输任何二阶信息。理论证明在低有效秩和白化假设下收敛速率独立于 Lipschitz 常数 $L$ 和模型维度 $d$；实验在 OPT-350M/1.3B/2.7B 微调中实现 1.4~5.4× 通信轮次加速，通信成本仅 KB 级。

**[Convergence of Muon with Newton-Schulz](optimization/convergence_of_muon_with_newton-schulz.md)**

:   首次为实际使用的 Muon 优化器（使用 Newton-Schulz 近似而非精确 SVD 极坐标分解）提供非凸收敛保证：证明收敛速率匹配 SVD 理想化版本（差一个常数因子），该因子随 Newton-Schulz 步数 $q$ 双指数衰减，且 Muon 比向量对应物 SGD-M 少 $\sqrt{r}$ 倍秩损失。

**[Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)**

:   从凸优化理论出发，证明深度学习训练损失以 O(1/sqrt(T)) 速率收敛，最优学习率以 1/sqrt(T) 缩放，在 GPT-2 到 12.5B 参数模型上验证了该缩放律（R^2 >= 0.978），并实现了 80 倍训练步数的学习率外推。

**[Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](optimization/deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)**

:   提出 FlexQP——基于 $\ell_1$ 弹性松弛的"永远可行"凸二次规划（QP）求解器，结合深度展开（deep unfolding）学习 LSTM 反馈策略加速收敛得到 Deep FlexQP；在 SQP 框架中作为子模块，解非线性轨迹优化比 OSQP 快 4-16 倍，预测安全滤波器的安全违规减少 70%+、任务完成率提升 43%。

**[DeepAFL: Deep Analytic Federated Learning](optimization/deepafl_deep_analytic_federated_learning.md)**

:   提出 DeepAFL，通过设计无梯度的解析残差块并引入逐层联邦训练协议，首次实现了具有表征学习能力的深度解析联邦学习模型，既保持了对数据异质性的理想不变性，又突破了现有解析方法仅限于单层线性模型的局限，在三个基准数据集上超越 SOTA 5.68%-8.42%。

**[Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)**

:   首次证明了梯度下降（gradient descent）在 leaky ReLU 两层神经网络中的方向收敛性（directional convergence），并据此在远超近正交数据（nearly orthogonal data）的更广泛混合数据设定下建立了 benign overfitting 的充分条件，同时发现了一个新的相变（phase transition）现象。

**[Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](optimization/dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)**

:   证明了约束深度学习中广泛使用的 dual optimistic ascent（PI 控制）在单步一阶更新体制下数学等价于经典的增广拉格朗日方法（ALM），从而将 ALM 的鲁棒收敛保证（线性收敛到所有严格局部解）转移至 PI 控制，并为乐观系数 $\omega$ 提供了原则性调参指导。

**[Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](optimization/exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)**

:   提出 STARS（Stiefel-based Activation Steering for Diverse ReaSoning），一种 training-free 的推理时激活转向方法，在每个 token 解码时于 Stiefel 流形上联合优化 N 条并行生成路径的正交 steering 方向，最大化隐状态的几何体积以促进发散的激活轨迹，在测试用例生成（TestEval）和科学发现（LiveIdeaBench）上以极低延迟一致超越温度采样的多样性，且不损失质量。

**[Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)**

:   通过将 F2SA 方法重新解释为前向差分近似 hyper-gradient，提出利用高阶有限差分的 F2SA-p 方法族，在高阶光滑条件下将随机双层优化的 SFO 复杂度从 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 改进至 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$，并证明了 $\Omega(\epsilon^{-4})$ 下界表明该方法在 $p$ 足够大时近乎最优。

**[FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](optimization/feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)**

:   提出 FedDAG 聚类联邦学习框架，通过融合数据和梯度信息进行加权类别级相似度计算来实现更准确的客户端聚类，并通过双编码器架构实现跨集群特征迁移，在多种异构性设置下一致超越现有基线。

**[FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)**

:   FrontierCO 是一个涵盖 8 类组合优化问题（TSP、MIS、CVRP 等）的大规模真实世界基准测试，评估了 16 个 ML 求解器（神经网络方法 + LLM Agent）与 SOTA 传统求解器的差距，发现 ML 方法在结构复杂和极大规模实例上仍显著落后于传统方法，但在部分场景有超越潜力。

**[Generalization Below the Edge of Stability: The Role of Data Geometry](optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)**

:   提出"数据可碎性"（data shatterability）原理统一解释数据几何如何控制梯度下降在稳定性边缘（EoS）附近的隐式正则化强度：对 Beta(α) 径向分布族推导出依赖 α 的泛化上下界谱，对低维子空间混合分布证明泛化率适应内在维度 $m$ 而非环境维度 $d$。

**[Learning to Recall with Transformers Beyond Orthogonal Embeddings](optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)**

:   在随机（非正交）嵌入条件下分析单层 Transformer 在 token 检索任务上经验梯度下降的"早期阶段"，推导出模型存储容量的显式公式，揭示了样本量 N、嵌入维度 d 和序列长度 L 之间的乘法依赖关系，并证明这一缩放关系是信息论下界固有的。

**[Learning to Solve Orienteering Problem with Time Windows and Variable Profits](optimization/learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)**

:   提出DeCoST——一种学习式两阶段框架，将OPTWVP中耦合的离散路线决策和连续服务时间分配解耦：第一阶段并行解码器联合生成路径+初始服务时间，第二阶段LP精确优化服务时间(全局最优)，通过pTAR反馈实现跨阶段协调。在50-500节点OPTWVP上优化间隙仅0.83%-3.31%，推理速度比元启发式快最高45倍。

**[Markovian Transformers for Informative Language Modeling](optimization/markovian_transformers_for_informative_language_modeling.md)**

:   提出马尔可夫语言模型(MLM)框架，通过**结构约束**（答案预测时移除原始问题，仅从CoT推导）强制CoT成为因果必要的推理瓶颈——类似自编码器的窄潜层，配合GRPO风格策略梯度训练，在GSM8K上从19.6%提升到57.1%，且学到的CoT可跨模型架构（Llama→Mistral/Phi/GPT-2）迁移，证明CoT编码了自然语言推理而非隐写术。

**[Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)**

:   深入分析了 SAM 在线性对角网络上训练时的隐式偏差，揭示深度从 $L=1$ 到 $L=2$ 引发的质变：$\ell_\infty$-SAM 的极限方向对初始化高度敏感，$\ell_2$-SAM 则展现出"先弱后强"的**序列特征放大**（sequential feature amplification）现象，指出仅关注 $t\to\infty$ 极限的分析不足以揭示 SAM 的完整动态行为。

**[∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)**

:   提出 ∇-Reasoner，将推理时的搜索从零阶（采样+评估）升级为一阶（梯度下降），在 token logits 空间上通过可微文本优化（DTO）结合 reward 梯度和 LLM 似然来迭代改进解码策略，在数学推理任务上提升 10-40% 准确率的同时减少 10-40% 的模型调用次数。

**[Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](optimization/neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)**

:   证明在通用非退化假设下，标准两层神经网络通过分层梯度下降可以用 $\tilde{O}(d)$ 样本和 $\tilde{O}(d^2)$ 时间学习通用高斯 Multi-Index 模型 $f(\bm{x})=g(\bm{U}\bm{x})$，样本和时间复杂度都达到信息论最优，首次证明神经网络可以高效学习层次化函数。

**[Non-Asymptotic Analysis of Efficiency in Conformalized Regression](optimization/non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)**

:   首次建立保形分位数回归（CQR）和保形中位数回归（CMR）在 SGD 训练下的非渐近效率界，明确刻画了预测集长度偏差与训练样本量 $n$、校准样本量 $m$ 和误覆盖率 $\alpha$ 的联合依赖关系。

**[Nonparametric Teaching of Attention Learners](optimization/nonparametric_teaching_of_attention_learners.md)**

:   提出AtteNT——从非参教学理论视角重新解释注意力学习器(Transformer/ViT)的训练过程：解析注意力在参数梯度中的重要性自适应角色→证明动态ANTK收敛到功能梯度中的重要性自适应典范核→桥接参数空间与函数空间→用贪心教学算法选择预测偏差最大的样本加速训练→LLM微调省时13.01%/ViT从头训练省时20.58%且精度不降反升。

**[NRGPT: An Energy-based Alternative for GPT](optimization/nrgpt_an_energy-based_alternative_for_gpt.md)**

:   提出NRGPT（eNeRgy-GPT），对标准GPT进行最小修改使其成为能量基模型：设计注意力能量和前馈能量函数，使每层前向传播等价于token在能量landscape上的梯度下降步，证明了渐近能量下降和稳定收敛性质，在ListOps/Shakespeare/OpenWebText上验证了与标准GPT可比的性能。

**[Personalized Collaborative Learning with Affinity-Based Variance Reduction](optimization/personalized_collaborative_learning_with_affinity-based_variance_reduction.md)**

:   提出个性化协作学习框架 AffPCL，通过偏差校正和重要性校正机制，让异质智能体在无需先验知识的情况下协作学习个性化解，实现 $O(t^{-1} \cdot \max\{n^{-1}, \delta\})$ 的自适应收敛率——智能体相似时获得线性加速，差异大时不差于独立学习。

**[Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](optimization/pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)**

:   提出 Πnet 架构，通过在神经网络输出层附加基于 Douglas-Rachford 算子分裂的正交投影层来保证凸约束的严格满足，并利用隐函数定理进行高效反向传播，在训练时间、求解质量和超参数鲁棒性上大幅超越现有方法。

**[Provable and Practical In-Context Policy Optimization for Self-Improvement](optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)**

:   提出 In-Context Policy Optimization (ICPO) 框架，理论证明单层线性自注意力 Transformer 经充分预训练后可在上下文中模拟策略优化算法，并设计实用的 ME-ICPO 算法通过最小熵选择和自评估奖励实现测试时多轮自反思，在数学推理任务上取得显著提升（AIME 2024 上 Qwen2.5-Math-7B 从 11% 提升到 30%）。

**[Rethinking Consistent Multi-Label Classification Under Inexact Supervision](optimization/rethinking_consistent_multi-label_classification_under_inexact_supervision.md)**

:   提出 COMES 框架，通过一阶（Hamming loss）和二阶（Ranking loss）策略，为不精确监督下的多标签分类提供一致性风险估计器，无需估计标签生成过程或均匀分布假设。

**[Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](optimization/rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)**

:   提出 Rolling Ball Optimizer (RBO)，通过模拟有限半径刚性球在损失景观上的滚动运动来打破传统优化器的空间局部性，实现对损失函数的平滑效应（ironing property），在 MNIST 和 CIFAR-10/100 上展示了更好的收敛速度和泛化性能。

**[RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)**

:   提出 RRNCO 架构，通过自适应节点嵌入（ANE）和神经自适应偏置（NAB）两大创新，首次在深度路由框架中联合建模非对称距离、时长和方向角，并构建了基于 100 个真实城市的 VRP 基准数据集，显著缩小了 NCO 方法从仿真到真实世界部署的差距。

**[RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](optimization/rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)**

:   提出 RS-ORT 算法，通过将回归树训练重构为两阶段优化问题并在缩减空间上进行分支定界（仅对树结构变量分支），结合闭式叶预测、阈值离散化和精确末层子树解析等加速策略，首次在包含连续特征的 200 万样本数据集上实现了有全局最优性保证的回归树学习。

**[Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](optimization/saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)**

:   提出统一的理论框架，通过 saddle-to-saddle 学习动力学解释多种神经网络架构（全连接、卷积、注意力）中普遍存在的 simplicity bias——即梯度下降倾向于先学习简单解再逐步学习复杂解的现象。

**[Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](optimization/scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)**

:   提出 Scaf-GRPO 框架，通过分层级的 in-prompt hint 注入（知识→规划→解题步骤）来克服 GRPO 训练中"学习悬崖"(zero-reward)问题，在 Qwen2.5-Math-7B 上将 AIME24 的 pass@1 相对提升 44.3%，同时保持 on-policy 训练一致性。

**[Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](optimization/scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)**

:   在幂律随机特征（Power-Law Random Features）模型下，系统分析了 SignSGD 的缩放定律，揭示了 SignSGD 相对于 SGD 的两个独特效应——漂移归一化和噪声重塑，并证明在噪声主导的情形下 SignSGD 的计算最优斜率可以超过 SGD。

**[Test-Time Meta-Adaptation with Self-Synthesis](optimization/test-time_meta-adaptation_with_self-synthesis.md)**

:   提出 MASS（Meta-Adaptation with Self-Synthesis）框架，通过双层优化元学习让 LLM 在推理时生成问题特定的合成训练数据（Generator）+ 评分筛选（Scorer）+ 加权 SFT 自更新（LoRA），meta-gradient 反向传播穿过内更新以优化数据质量，在 MATH-500 上将 Llama-3.1-8B 从 43.6% 提升至 59.0%。

**[The Affine Divergence: Aligning Activation Updates Beyond Normalisation](optimization/the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)**

:   揭示了梯度下降中参数最速下降方向与传播到激活后的有效更新之间存在根本性不对齐（"仿射散度"$\Delta\mathcal{L}/\Delta z_i = (\partial\mathcal{L}/\partial z_i) \cdot (\|\vec{x}\|^2+1)$），从第一性原理推导出归一化是消除此散度的自然解，并发现一种非归一化的替代方案在实验中超越传统归一化。

**[Unifying Formal Explanations: A Complexity-Theoretic Perspective](optimization/unifying_formal_explanations_a_complexity-theoretic_perspective.md)**

:   提出统一框架将充分理由和对比理由（局部/全局、概率/非概率）归结为对统一概率值函数的最小化问题，揭示全局值函数具有单调性、子模性/超模性等组合优化关键性质，从而证明全局解释在多项式时间内可计算——即使对应的局部解释是 NP-hard 的。

**[When to Restart? Exploring Escalating Restarts on Convergence](optimization/when_to_restart_exploring_escalating_restarts_on_convergence.md)**

:   提出 SGD-ER（SGD with Escalating Restarts），一种收敛感知的学习率调度策略：当检测到训练停滞时触发重启并线性升高学习率，帮助优化器逃离尖锐局部极小值、探索更平坦的损失景观区域，在 CIFAR-10/100 和 TinyImageNet 上取得 0.5-4.5% 的测试精度提升。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](llm_alignment/a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)**

:   提出 A2D，一种针对扩散语言模型（dLLM）的 token 级安全对齐方法，通过训练模型在遇到有害内容的 mask 位置输出 [EOS] token 来实现任意解码顺序、任意解码步的安全防御，将 DIJA 模板攻击成功率从 80%+ 降到近零（1.3%/0.0%），并支持早期拒绝实现 19.3x 加速。

**[Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](llm_alignment/align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)**

:   提出 Multi-Lingual Consistency (MLC) 辅助损失，通过 SVD 操控多语言表示矩阵的奇异值使其趋向秩-1（即多语言表示共线），仅需多语言 prompt 翻译（无需目标语言的 response），即可将一种语言的安全对齐效果一致性地迁移到所有语言。

**[Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](llm_alignment/alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)**

:   提出MetaAPO框架，用一个轻量级meta-learner（两层MLP）动态估计offline/online数据的对齐差距，既指导"在哪些prompt上做在线采样"（解决分布不匹配），又在训练时自适应加权offline/online数据（优化学习效率），在AlpacaEval 2/Arena-Hard/MT-Bench上超越DPO/Online DPO等基线，同时减少42%在线标注成本。

**[AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](llm_alignment/alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)**

:   提出 AlphaSteer，通过学习一个受零空间约束的变换矩阵来动态构造 steering 向量，对良性输入产生近零向量（保持效用），对恶意输入重建拒绝方向向量（增强安全），在理论上保证了安全与效用的解耦。

**[AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](llm_alignment/avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)**

:   针对多模态大语言模型在情感推理中的虚假关联和幻觉问题，提出 EmoReAlM 评测基准和 AVEm-DPO 偏好优化方法，通过构建针对性偏好对和文本先验正则化，在 DFEW/RAVDESS/EMER 上实现 6-19% 的零样本相对性能提升。

**[Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](llm_alignment/beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)**

:   提出 RCPO 框架，将 LLM 对齐从成对偏好扩展到排名选择（ranked choice）建模，通过 MLE 统一了效用模型（MNL）和排名模型（Mallows-RMJ），在 single-best 和 top-k 反馈格式下都优于 DPO 及其变体。

**[Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](llm_alignment/beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)**

:   提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。

**[CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](llm_alignment/cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)**

:   提出 CAGE 框架，通过 Semantic Mold（语义模具）将红队攻击 prompt 的对抗结构与文化内容解耦，能系统性地将英语红队基准适配到不同文化语境中，生成的文化扎根 prompt 比直接翻译的 ASR 显著更高。

**[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)**

:   理论证明奖励过优化主要源于高奖励尾部区域的奖励模型错误规范，提出基于 rubric 的奖励建模方法：利用 off-policy 数据（强模型生成的优秀回复）构造评分细则，通过渐进式区分"优秀 vs 更优秀"来精细化 rubric，有效缓解奖励过优化。

**[Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](llm_alignment/displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)**

:   发现 f-DPO 的可解性不需要 f 凸（仅需 $\lim_{t\to 0^+} f'(t) = -\infty$），进一步证明 $\arg\min f(t) \geq 1$ 是抵抗概率位移的必要条件，由此提出 SquaredPO（$f(t) = \frac{1}{2}(\log t)^2$，非凸），在保持性能的同时显著缓解 winner 概率下降问题。

**[Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](llm_alignment/dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)**

:   提出 Dual-IPO 框架，通过在奖励模型和视频生成模型之间进行多轮双向迭代优化，无需大量人工标注即可持续提升文本到视频生成的质量和人类偏好对齐，甚至让 2B 模型超越 5B 模型。

**[From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](llm_alignment/from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)**

:   提出ALPO(自适应局部偏好优化)用于训练表达力强的字幕翻译LLM：通过实证发现字幕翻译偏好意译且推理型LLM意译能力优于对话型LLM -> 验证LLM作为翻译评估器与人类高度一致 -> 提出逐句段的细粒度过程监督偏好对齐方法(自适应权重+动态beta+前缀混合) -> 14B模型在多方向字幕翻译的鲜活度上超越GPT-4o/DeepSeek-R1等SOTA。

**[General Exploratory Bonus for Optimistic Exploration in RLHF](llm_alignment/general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)**

:   理论证明现有 RLHF 探索奖励（exploratory bonus）在 KL 和 α-散度正则化下实际上会引导策略向参考模型的高概率区域靠拢（与乐观原则相悖），提出 General Exploratory Bonus (GEB) 框架——通过参考模型依赖的奖励调节来抵消散度正则化的保守偏差，可证明满足乐观原则。

**[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)**

:   通过构造 KL 正则化代理目标并推导 pairwise consistency condition，从第一原理证明 group-relative REINFORCE（GRPO）天然是 off-policy 算法；进而通过组件隔离实验发现 clipping 才是训练稳定性的关键而 importance sampling 完全可以去掉，并在此统一框架下重新解释了 Kimi OPMD、Meta AsymRE 等多个看似独立的算法。

**[GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)**

:   提出 GuardAlign，一个无需训练的多模态大模型推理时安全防御框架：用最优传输(OT)精确检测图像中的不安全区域并遮蔽，再通过跨模态注意力校准保持安全前缀的影响力不衰减，在6个LVLM上将不安全响应率降低最多39%，同时保持甚至提升通用能力。

**[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](llm_alignment/hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)**

:   揭示了 stepwise group-based RL（如 GRPO/GiGPO）中的「历史上下文不一致」问题——同一 group 内的 step 可能具有不同历史上下文导致 advantage 估计偏差，提出 HGPO 通过层次化分组和自适应加权实现低偏差、平衡方差的 advantage 估计，在 ALFWorld 和 WebShop 上以极低额外开销（<0.001%）取得显著提升。

**[Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](llm_alignment/is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)**

:   挑战"on-policy数据总是更好"的共识：发现对齐过程分为偏好注入（需高多样性off-policy数据）和偏好微调（需高质量on-policy数据）两个阶段，不同模型/阶段对数据类型的最优选择不同。提出仅3.2%计算开销的边界判定算法，在5个模型×55个配置上验证有效。

**[JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](llm_alignment/jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)**

:   提出首个评估 LLM 在越狱攻击下生成假新闻鲁棒性的多语言多区域基准 JailNewsBench，覆盖 34 个地区和 22 种语言、约 30 万实例，揭示最高 86.3% 的攻击成功率以及英语/美国话题防御显著弱于其他地区的安全不平衡现象。

**[Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](llm_alignment/learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)**

:   提出**D3S**（Dynamic Dual-Level Down-Sampling）框架，在sample层最大化advantage方差、在token层优先选取高熵+高advantage的token，配合动态调度策略，用不到20% token实现更快收敛和更优性能。

**[Learning Ordinal Probabilistic Reward from Preferences (OPRM)](llm_alignment/learning_ordinal_probabilistic_reward_from_preferences.md)**

:   提出序数概率奖励模型(OPRM)，将响应质量离散化为1-9序数等级并学习完整概率分布，结合区域洪泛调优(RgFT)实现数据高效训练。在RewardBench达89.3%，比现有RM提升2.9%-7.4%，同时提供不确定性估计和标注分歧检测。

**[Mitigating Mismatch within Reference-based Preference Optimization](llm_alignment/mitigating_mismatch_within_reference-based_preference_optimization.md)**

:   揭示 DPO 的"过早满足"问题——当 reference 策略对 chosen 的概率低于 rejected 时（~45% pairs），DPO 的梯度被 reference 的悲观信号不必要地衰减（即使策略仍然错误即 $\Delta_\theta < 0$）；提出 HyPO（一行代码修改：$\max(0, \Delta_{ref})$ 裁剪 reference margin），在 AlpacaEval 2.0 上相对 DPO 提升 41.2%。

**[Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](llm_alignment/mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)**

:   提出 NSPO，将安全对齐的策略梯度投影到通用任务表征的零空间中，从几何层面保证安全优化不损害通用能力，仅用 40% 安全数据即在 7 个安全 benchmark 上达到 SOTA，同时在数学/代码/指令遵循上几乎无性能损失。

**[No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)**

:   发现 GRPO 训练中大量"零方差提示"（所有回答全对或全错）被白白丢弃，提出 RL-ZVP 算法通过熵引导的优势整形从中提取学习信号，在六个数学推理基准上相比 GRPO 提升最高 8.61 个精度点和 7.77 个通过率点。

**[Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks](llm_alignment/optimal_sparsity_of_mixture-of-experts_language_models_for_reasoning_tasks.md)**

:   系统研究 MoE 语言模型的稀疏度如何不同地影响记忆性任务和推理性任务：记忆任务偏好更高稀疏度（更多参数），而推理任务在 TPP≈20 附近达到最优，且该趋势在 GRPO 后训练和测试时计算增加后仍然不变。

**[Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)**

:   提出"先回答后检查"(Answer-Then-Check)策略：模型先在思维链中生成意图答案摘要，再依据安全策略进行安全分析，最后决定输出或拒绝。构建80K ReSA数据集训练后，在7种越狱攻击上防御率达到99.3%(RL版本)，仅500样本即可达全数据集效果。

**[PURGE: Reinforcement Unlearning via Group Relative Policy Optimization](llm_alignment/reinforcement_unlearning_via_group_relative_policy_optimization.md)**

:   PURGE 将 LLM 遗忘（unlearning）重新定义为可验证的 RL 任务，使用 GRPO 框架 + 内在奖励信号（惩罚提及禁止概念）来实现安全一致的知识删除，token 消耗比 SOTA 低 46 倍，同时提升流畅度 +5.48% 和对抗鲁棒性 +12.02%。

**[SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](llm_alignment/safedpo_preference_optimization_safety.md)**

:   重新审视安全约束 RLHF 目标并证明其存在闭式最优策略，据此推导出等价的可处理目标 SafeDPO，仅需在标准 DPO 上加入安全感知数据变换和安全 margin 项（1 个额外超参数），无需奖励/代价模型，在 PKU-SafeRLHF-30K 上实现 96.87% 无害率且保持竞争力的有用性，训练速度比 SafeRLHF 快 25×。

**[SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](llm_alignment/sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)**

:   提出 SEMA 框架，通过预填充自调优和带意图漂移感知奖励的 RL 两阶段训练，在无需任何现有攻击策略或外部数据的条件下，训练出能自动生成多轮越狱攻击的 attacker，在 AdvBench 上跨三个受害模型平均 ASR@1 达 80.1%，超越 SOTA 33.9%。

**[Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](llm_alignment/semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)**

:   指出 RLHF 中标准 KL 散度正则化仅比较相同索引处的 token 概率而忽略语义相似性，提出基于熵正则化 Wasserstein 距离的语义感知策略正则化（WPR），通过对偶公式将正则化转化为 token 级惩罚项，在对话生成和摘要任务上一致优于 KL 及各类 f-散度基线。

**[Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](llm_alignment/skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)**

:   提出 Human-AI 协同的两阶段偏好数据策展流水线：阶段一通过人工验证、错误驱动自适应检索和偏好引导 LLM 标注迭代 8 轮积累约 1M 偏好对；阶段二借助双 RM 一致性过滤将数据规模扩展到 26M 对。最终训练的 Skywork-Reward-V2 8B 模型在 RewardBench 达 97.8%，7 个主流基准平均 88.6%，全面超越所有开源 70B 奖励模型。

**[Superficial Safety Alignment Hypothesis](llm_alignment/superficial_safety_alignment_hypothesis.md)**

:   提出"浅层安全对齐假说"(SSAH)：安全对齐本质上是教模型做一个隐式的二分类任务（执行还是拒绝），只需约1.3%的神经元即可建立安全护栏；冻结这些安全关键单元可在微调时保持安全性，利用冗余单元作为"对齐预算"可消除对齐税。

**[Swap-guided Preference Learning for Personalized RLHF (SPL)](llm_alignment/swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)**

:   解决变分偏好学习(VPL)中的后验崩坏问题：提出SPL，通过swap引导基础正则化(强制潜变量编码用户偏好而非被忽略)+Preferential-IAF分解swap可逆/不可逆信号+自适应潜变量调节。在Llama-3.1-8B上达63.71%准确率+97.10%活跃单元，而VPL崩坏到57.14%+0%。

**[Token-Importance Guided Direct Preference Optimization (TI-DPO)](llm_alignment/token-importance_guided_direct_preference_optimization.md)**

:   提出TI-DPO，通过梯度归因+高斯先验的混合权重机制精确量化每个token对偏好的贡献，结合三元组损失在连续语义空间引导优化，在6个基准上平均62.3分达到SOTA，同时具备可解释的token级控制能力。

**[Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)**

:   提出 UltraBreak，通过语义对抗目标（用cosine相似度替代交叉熵优化出平滑loss景观）+ 输入空间约束（随机变换+TV正则化产生变换不变特征），训练单张通用对抗图像即可跨6+个VLM架构和商业模型实现越狱，黑盒平均ASR达71%（SafeBench），远超此前方法。

**[Towards Understanding Valuable Preference Data for Large Language Model Alignment](llm_alignment/towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)**

:   从模型依赖视角研究偏好数据质量：提出截断影响函数(TIF)发现中等IF值的数据才是最有价值的(而非经典观点中的高IF) -> 设计LossDiff和IRM两个轻量代理指标近似TIF -> 两者组合的LossDiff-IRM选择器仅用50-64%数据即可平均提升WinRate 13.58%，在多个LLM家族和对齐benchmark上均有效。

**[Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)**

:   提出Uni-DPO，通过质量感知加权（高分差偏好对优先）+性能感知加权（focal loss聚焦欠拟合样本）+校准NLL损失三个组件统一动态调整DPO偏好对权重，在文本理解和数学推理基准上一致超越DPO/SimPO，Gemma-2-9B在Arena-Hard达67.1%超过Claude 3 Opus(60.4%)。

**[Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](llm_alignment/unifying_stable_optimization_and_reference_regularization_in_rlhf.md)**

:   提出DAR(Dual-regularized Advantage Regression)：发现标准RLHF中参考模型正则化(防reward hacking)和策略稳定约束(防崩溃)会逐步冲突导致优化空间过度受限，通过双KL目标在对数空间插值参考策略+回归变换消除策略比率不稳定性，在直接AI对齐和标准RLHF设置中达到92.42%平均胜率，超GRPO 7.27%。

**[Why DPO is a Misspecified Estimator and How to Fix It](llm_alignment/why_dpo_is_misspecified_estimator.md)**

:   从信息几何角度证明 DPO 在参数化（非 tabular）策略类下本质上是一个误指定的统计估计问题——DPO 将真实奖励函数 KL 投影到隐式奖励流形上，当奖励不可实现时会导致偏好反转和奖励下降——并提出 AuxDPO 通过引入零空间辅助变量来修复此问题。

---

## 📈 时间序列 { #time_series }

**[Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)**

:   提出TATO框架，通过自动优化数据预处理 pipeline（包括上下文裁切、尺度归一化、异常值校正），让冻结的大型时序模型（LTM）在不微调的情况下适配不同下游领域，平均降低MSE 13.6%，最高65.4%。

**[Contextual and Seasonal LSTMs for Time Series Anomaly Detection](time_series/contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](time_series/cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time.md)**

:   提出 CPiRi 框架，通过冻结的预训练时序编码器 + 轻量空间 Transformer + 通道打乱训练策略，实现通道排列不变 (CPI) 的跨通道关系建模，在 5 个基准上达到 SOTA 且通道打乱后性能几乎无损 ($\Delta$WAPE < 0.25%)。

**[Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)**

:   提出 Delta-XAI 统一框架，通过包装函数将14种现有XAI方法适配到在线时间序列预测变化解释场景，并提出 SWING（Shifted Window Integrated Gradients）方法，利用过去观测值构建积分路径以捕获时序依赖关系，在多种评估指标上持续优于现有方法。

**[EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](time_series/edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)**

:   构建了基于日本 EDINET 十年年报的金融基准 EDINET-Bench，包含会计欺诈检测、盈利预测和行业分类三项专家级任务，发现即使是 SOTA LLM 也仅略优于逻辑回归。

**[Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)**

:   提出 Global Temporal Retriever（GTR），一个轻量级即插即用模块，通过维护自适应全局周期嵌入并利用绝对时间索引检索对齐全局周期信息，使任意预测模型突破回看窗口限制，有效捕获远超输入长度的全局周期模式。

**[Free Energy Mixer](time_series/free_energy_mixer.md)**

:   提出 Free Energy Mixer (FEM)，通过将注意力的值读取重新定义为自由能（log-sum-exp）优化问题，实现了逐通道的值感知后验选择，克服了标准注意力"无损存储但有损读取"的固有瓶颈，可即插即用替换 softmax/线性注意力/RNN/SSM，在 NLP、视觉和时间序列任务上一致提升。

**[From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](time_series/from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)**

:   提出 Probabilistic Scenarios 范式，用模型直接输出有限个 {场景, 概率} 对取代采样，并用仅含三层平行线性层的 TimePrism 在5个基准数据集上取得9/10 SOTA。

**[GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](time_series/gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)**

:   提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征的通用时序基础模型，结合混合掩码预训练策略，首次实现无需任务特定修改即可适配所有生成式时序任务。

**[HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](time_series/hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)**

:   提出 HiVid 框架，首次利用 LLM 作为人类代理为视频块生成内容重要性权重，通过感知模块（滑动窗口评分）、排序模块（LLM 引导归并排序去除评分偏差）和预测模块（多模态时间序列预测自适应延迟）实现内容感知流媒体传输，

**[Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative](time_series/language_in_the_flow_of_time_time-series-paired_texts_weaved_into_a_unified_temp.md)**

:   发现时间序列配对文本具有与时间序列相似的周期性（Chronological Textual Resonance），提出 TaTS 框架将文本表征转化为辅助变量，以即插即用方式增强任意现有时间序列模型的预测和插补性能。

**[Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)**

:   提出 ReIMTS，通过基于时间段的递归分割（而非重采样）来保留不规则多变量时间序列的原始采样模式，结合不规则感知的表示融合机制实现多尺度建模，作为插件在六种 IMTS 骨干上平均提升 27.1%。

**[PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](time_series/paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)**

:   提出 PaAno，一种基于 patch 级表示学习的轻量时间序列异常检测方法，使用 1D-CNN 编码器 + triplet loss + pretext loss 学习 patch 嵌入空间，通过与记忆库中正常 patch 的距离计算异常分数，在 TSB-AD 基准上全面 SOTA，且仅需 0.3M 参数和数秒推理。

**[Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](time_series/rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)**

:   提出TSRating框架，利用LLM从趋势/频率/幅度/模式四个维度对时间序列数据块做成对质量比较，通过Bradley-Terry模型转换为标量质量分数，并以MAML元学习在9个领域22个子集上训练TSRater模型（MOMENT编码器+MLP），实现高效、统一的跨域时间序列数据质量评估。

**[Reasoning on Time-Series for Financial Technical Analysis](time_series/reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

**[Relational Feature Caching for Accelerating Diffusion Transformers](time_series/relational_feature_caching_for_accelerating_diffusion_transformers.md)**

:   提出关系特征缓存（RFC）框架，通过利用DiT模块输入-输出特征之间的强相关性来增强缓存特征预测的精度，包括从输入变化估计输出变化幅度的RFE和用输入误差代理判断是否需要全量计算的RCS，在图像和视频生成任务上显著优于现有的基于时间外推的缓存方法。

**[Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)**

:   提出 Relational Transformer (RT) 架构，通过 task table prompting、cell tokenization 和 Relational Attention 机制，在多个关系数据库上预训练后可零样本迁移到未见过的数据集和任务，22M 参数模型零样本 AUROC 达到全监督方法的 93%，远超 27B LLM 的 84%。

**[ResCP: Reservoir Conformal Prediction for Time Series Forecasting](time_series/rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)**

:   首次将储备计算（Echo State Network）引入保形预测，通过随机初始化ESN编码残差序列的时间动态，利用状态相似性自适应重加权历史残差构建局部预测区间，无需任何训练即在4个真实数据集上实现SOTA的Winkler分数，速度比HopCPT快20-80×。

**[Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)**

:   提出 xCPD 即插即用插件，将多变量时间序列的建模单元从"通道"细化到"通道-patch"，通过共享图傅里叶基做谱嵌入→按频率能量响应分组为低/中/高频段→动态 MoE 路由自适应选择频率特定滤波专家，可无缝集成到 CI/CD 任何现有模型上一致提升长短期预测性能，并支持零样本迁移。

**[scits scientific time series understanding and generation with llms](time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)**

:   提出SciTS基准覆盖12个科学领域43个任务54K+实例（长度从$10^0$到$10^7$、频率达10MHz），系统评估17个模型发现通用LLM比专用时序模型泛化更好但文本/图像编码各有局限，据此设计TimeOmni框架用多Patch专家+路由机制+Patch重编程显式建模时间动态并与LLM联合训练。

**[SCRAPL: Scattering Transform with Random Paths for Machine Learning](time_series/scrapl_scattering_transform_with_random_paths_for_machine_learning.md)**

:   针对多变量散射变换(ST)作为可微损失函数时因路径数P过多导致计算代价过高的问题，提出SCRAPL——每步仅随机采样一条路径并通过P-Adam（路径自适应动量）、P-SAGA（路径随机平均梯度）和θ-重要性采样三种方差缩减技术来稳定梯度，在无监督声音匹配任务上以接近全路径ST的精度、MSS级别的低计算成本实现了Pareto最优。

**[SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](time_series/swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)**

:   提出首个时间序列预训练模型选择框架SwiftTS，使用双编码器架构独立嵌入数据集patch级时序特征和模型元信息（架构/拓扑/功能），通过patch级交叉注意力计算兼容性分数，结合horizon自适应专家组合和跨域/跨horizon元学习，在14个数据集×8个模型上以平均加权Kendall $\tau_\omega = 0.442$ 大幅超越所有基线。

**[T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](time_series/t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)**

:   提出T1——CNN-Transformer混合架构，核心创新是Channel-Head Binding（CHead Attention）：共享Depthwise Conv为每个变量提取C种时序特征（趋势/周期/突变等），然后将每个CNN通道与一个注意力头一对一绑定，使跨变量信息传递在特征级别独立进行。当缺失导致某通道无法提取有效模式时，对应注意力头自动降权，实现无需显式设计的自适应缺失处理。在11个基准数据集上MSE平均降低46%，70%极端缺失下优势更大。

**[Tensor learning with orthogonal, Lorentz, and symplectic symmetries](time_series/tensor_learning_with_orthogonal_lorentz_and_symplectic_symmetries.md)**

:   本文给出了关于正交群 $O(d)$、不定正交群（含 Lorentz 群）和辛群 $Sp(d)$ 对张量对角作用下的等变多项式函数的完整参数化刻画，并将其应用于设计可学习的稀疏向量恢复算法，在多种数据生成假设下超越了已有的 sum-of-squares 谱方法。

**[Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](time_series/test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)**

:   提出 Chroma——小型预训练时序模型组合（portfolio）框架：从通用模型通过后训练（post-training）产出频率/领域专家（训练加速 10×），测试时通过模型选择或贪心集成组合，4M 参数的 portfolio 在 Chronos Benchmark II 上匹配 205M-500M 参数的大型单体模型性能，同时推理计算远低于 test-time fine-tuning。

**[TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)**

:   提出TimeSliver——可解释性驱动的深度学习框架,联合利用原始时序数据和符号抽象(分箱)构建保持原始时间结构的表示,每个元素线性编码对应时间段对最终预测的贡献→赋予每个时间点正/负归因分数,在7个数据集上时间归因准确率超越其他方法11%,同时在26个UEA基准上预测性能持平SOTA。

**[Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)**

:   提出 iMOOE 框架，通过显式定义 PDE 系统中的"算子不变性 + 组合不变性"两层物理不变性原理，设计与之对齐的混合算子专家网络和频率增强的风险等式目标，在不需要任何测试时适应的条件下实现多种 OOD 情景下的 SOTA 零样本 PDE 动力学预测。

**[Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)**

:   提出ChannelTokenFormer（CTF），一个统一的Transformer框架同时解决真实世界多变量时序预测的三大挑战：(1) 通道间复杂依赖——channel token跨通道注意力；(2) 各通道异步采样——频域动态patching保持原始分辨率；(3) 测试时块缺失——训练时patch masking模拟+推理时直接移除全缺失patch，在ETT/SolarWind/Weather/EPA/CHS等6个数据集上全面SOTA。

**[TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series](time_series/tspulse_tiny_pre-trained_models_with_disentangled_representations_for_rapid_time.md)**

:   提出 TSPulse，仅 1M 参数的超轻量时间序列预训练模型，通过双空间掩码重建和双嵌入解耦策略，在分类（+5-16%）、异常检测（+20%）、插补（+50%）和相似性检索（+25%）四大任务上超越 10-100 倍大的模型。

**[调节 RNN 训练中的 Burn-in 阶段可提升性能](time_series/tuning_the_burn-in_phase_in_training_recurrent_neural_networks_improves_their_pe.md)**

:   从理论上证明了 RNN 训练中 burn-in 阶段长度 $m$ 对截断反向传播时间（TBPTT）训练性能的关键影响，建立了训练遗憾的上界估计，并通过系统辨识和时间序列预测实验验证，合理调节 burn-in 可将预测误差降低超过 60%。

**[VoT: 事件驱动推理与多层对齐解锁文本价值用于时间序列预测](time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)**

:   提出 VoT，一种通过事件驱动推理（利用 LLM 对外生文本进行结构化推理获取数值预测）和多层对齐（表征级内生文本对齐 + 预测级自适应频率融合）充分挖掘文本信息价值的多模态时间序列预测方法，在 10 个领域的真实数据集上全面超越现有方法。

**[WARP: 权重空间线性循环神经网络](time_series/weight-space_linear_recurrent_neural_networks.md)**

:   提出 WARP（Weight-space Adaptive Recurrent Prediction），将线性 RNN 的隐状态显式参数化为辅助 MLP 的权重和偏置，利用输入差分驱动线性递推来更新权重，结合非线性解码实现高效序列建模，在分类、预测和动力系统重建等任务上达到 SOTA。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)**

:   提出ARC-JSD方法，通过计算完整上下文与逐句消融上下文下的响应分布的Jensen-Shannon散度，在无需微调、梯度计算或代理模型的情况下实现高效精准的RAG上下文归因，并结合Logit Lens进行机制分析，定位负责上下文归因的注意力头和MLP层，通过门控操作降低约39%的幻觉率。

**[Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](information_retrieval/bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)**

:   将位置编码重新表述为贝叶斯注意力机制中的先验分布，统一了 NoPE（均匀先验）和 ALiBi（拉普拉斯先验），并提出广义高斯先验（GGD-BAM），仅增加 384 个参数即可在 500 倍训练长度上实现完美的 passkey 检索。

**[Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)**

:   提出 LDAR（Learning Distraction-Aware Retrieval），一个轻量级自适应检索器，通过学习基于查询-段落相似度分布选择段落的连续区间（band），在平衡信息覆盖与干扰段落影响的同时，以约一半的 token 用量超越长上下文方法的性能。

**[BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)**

:   提出 BTZSC 基准（22 个数据集），首次在统一零样本协议下系统比较 NLI 交叉编码器、嵌入模型、Reranker 和指令微调 LLM 四大模型家族（共 38 个模型），发现 Qwen3-Reranker-8B 以 macro F1=0.72 取得新 SOTA，嵌入模型在精度-延迟权衡上最优。

**[Digging Deeper: Learning Multi-Level Concept Hierarchies](information_retrieval/digging_deeper_learning_multi-level_concept_hierarchies.md)**

:   本文提出 Multi-Level Concept Splitting（MLCS）将概念分裂过程从单层递归扩展到多层，仅用顶层概念标注就能自动发现任意深度的概念层级树，并设计 Deep-HiCEMs 架构来表示和利用这些深层层级，实现多粒度的测试时概念干预。

**[Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](information_retrieval/efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)**

:   提出EDJE（高效判别式联合编码器），通过将视觉特征提取离线化并用轻量级注意力适配器压缩视觉Token，实现50k图文对/秒的高吞吐推理，同时在Flickr（零样本）和COCO（微调）检索上匹配现有联合编码器的性能，每张图仅需49kB存储。

**[Embedding-Based Context-Aware Reranker](information_retrieval/embedding-based_context-aware_reranker.md)**

:   提出 EBCAR，一个基于嵌入空间的轻量级重排序框架，通过文档 ID 嵌入和段落位置编码引入结构信息，结合共享全注意力 + 专用掩码注意力的混合机制实现跨段落推理，在 ConTEB 基准上以 126M 参数达到最优平均 nDCG@10，推理速度比 LLM 重排器快 150 倍以上。

**[Fine-tuning with RAG for Improving LLM Learning of New Skills](information_retrieval/fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)**

:   提出将 RAG 从推理时的永久依赖转化为训练时的教师信号：从 agent 失败中提取 hint、用 hint 增强的教师生成更优轨迹、然后移除 hint 蒸馏到学生模型，使学生内化检索增益而无需运行时 RAG，在 ALFWorld 达到 91% 成功率（基线 79%），WebShop 分数达 72（基线 61）。

**[Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets](information_retrieval/flow_of_spans_generalizing_language_models_to_dynamic_span-vocabulary_via_gflown.md)**

:   提出 FoSS，首次将 GFlowNets 引入 span 级别语言模型，通过构建 DAG 结构的状态空间代替传统 token-by-token 的树形结构，实现更灵活多样的文本生成，MAUVE 分数最高提升 12.5%。

**[FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation](information_retrieval/futuremind_equipping_small_language_models_with_strategic_thinking-pattern_prior.md)**

:   提出FutureMind无训练框架，将LLM的结构化推理和检索策略蒸馏为可复用的思维模式先验，通过四阶段pipeline（问题分析→逻辑推理→策略规划→检索指导）和三种检索范式，使SLM在多跳QA上达到SOTA。

**[G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)**

:   提出 G-reasoner，通过 QuadGraph 四层统一图接口将异构知识源标准化，训练 34M 参数的 GNN 图基础模型联合推理图拓扑和文本语义，配合 LLM 在 6 个基准上全面超越 SOTA GraphRAG 方法。

**[Hierarchical Concept-based Interpretable Models](information_retrieval/hierarchical_concept-based_interpretable_models.md)**

:   HiCEMs引入层级概念嵌入模型，通过Concept Splitting方法在预训练CEM的嵌入空间中自动发现细粒度子概念（无需额外标注），构建层级概念结构，使模型能在不同粒度层次进行测试时概念干预以提升任务性能。

**[HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)**

:   提出 HUME 人类评估框架，在 MTEB 的 16 个数据集（重排序/分类/聚类/STS）上系统测量人类表现，发现人类总体排名第 4（77.6 vs 模型最佳 80.1），揭示模型"超人"表现多出现在人类一致性最低的任务上，并评估 9 个 LLM 作为标注代理的可行性。

**[Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](information_retrieval/hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)**

:   提出 HybridDeepSearcher，通过构建 HDS-QA 数据集训练大语言推理模型（LRM）区分可并行化和顺序依赖的搜索查询，在 FanOutQA 上 F1 提升 +15.9、BrowseComp 子集上提升 +11.5，同时显著降低推理延迟并展示出一致的测试时搜索扩展能力。

**[Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement](information_retrieval/judges_verdict_a_comprehensive_analysis_of_llm_judge_capability_through_human_ag.md)**

:   提出 Judge's Verdict Benchmark——基于"相关性过滤 → Cohen's Kappa 人类相似性测试"的两步评估框架，对 54 个 LLM 评委进行系统评测，筛选出 27 个 Tier 1 评委（23 个人类相似型 + 4 个超一致型），核心发现是相关性高不等于一致性高，需要用 Kappa + z-score 才能真正衡量 LLM 评委质量。

**[Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](information_retrieval/leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)**

:   提出 MA-PaPSP 框架，通过外部检索数据集构建代理嵌入（k-NN 加权平均降低表示方差）+ 对比归一化评分（改善校准），无训练地为任意 VLM 提供可靠的"拒绝回答"能力，在图像描述、图文匹配、分类的选择性预测上全面优于 PaPSP 和 LLM-as-judge 基线。

**[LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](information_retrieval/lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)**

:   提出 LightRetriever，一种极端不对称的LLM检索架构：文档端保留完整LLM编码器，查询端完全去除深度建模——稠密检索仅需嵌入查表+平均，稀疏检索仅需token计数——实现查询编码1000倍加速、端到端10倍吞吐提升，同时保持95%的检索性能。

**[Mapping Semantic & Syntactic Relationships with Geometric Rotation](information_retrieval/mapping_semantic_syntactic_relationships_with_geometric_rotation.md)**

:   提出RISE（Rotor-Invariant Shift Estimation）方法，利用Clifford代数的rotor将话语级语义-句法变换（否定、条件化、礼貌化）表示为单位超球面上的一致旋转操作，在7种语言×3种嵌入模型×3种变换的系统实验中证实这些旋转可跨语言和跨模型迁移（77%-95%保持率），首次将线性表示假说从词级扩展到跨语言话语级并推广到弯曲流形上的测地线结构。

**[Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis](information_retrieval/multimodal_dataset_distillation_made_simple_by_prototype-guided_data_synthesis.md)**

:   提出 PDS（Prototype-Guided Data Synthesis），首个免训练的多模态数据集蒸馏框架——利用 CLIP 对齐嵌入空间做模态特异聚类，通过匈牙利算法跨模态匹配获得图文原型，再用 unCLIP 解码器从图像原型合成蒸馏图像，在 100 对极小蒸馏集上以零训练代价全面超越优化式方法，并实现 SOTA 的跨架构泛化能力。

**[On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation](information_retrieval/on_the_wings_of_imagination_conflicting_script-based_multi-role_framework_for_hu.md)**

:   提出 HOMER 框架，基于 GTVH 幽默理论构建三角色 LLM 协作机制（冲突脚本提取器 + 层次想象器 + 标题生成器），通过显式建模脚本对立、多视角联想链与笑话数据库检索构建想象树来扩展创意空间，在 New Yorker 漫画基准上以 GPT-4o 为底座平均提升 ~7%，人类评估也显著优于所有基线。

**[Query-Level Uncertainty in Large Language Models](information_retrieval/query-level_uncertainty_in_large_language_models.md)**

:   提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。

**[RAEE: A Robust Retrieval-Augmented Early Exit Framework for Efficient Inference](information_retrieval/raee_a_robust_retrieval-augmented_early_exit_framework_for_efficient_inference.md)**

:   提出 RAEE，一种无需训练分类器的检索增强早退框架，通过检索语义相似样本的退出信息来动态确定最优退出层，不仅加速推理还能纠正模型错误预测，实现加速与性能提升的双赢。

**[RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)**

:   构建首个评估多模态检索增强文化理解的基准 Ravenea，包含 1868 个实例和 11396 篇人工排序的 Wikipedia 文档，覆盖 8 个国家 11 个类别，评估 7 个多模态检索器和 17 个 VLM，发现文化感知的 RAG 可在 cVQA 上平均提升 6%、cIC 上提升 11%。

**[RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](information_retrieval/reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)**

:   提出 RefTool 框架基于外部参考资料（教材、知识片段）自动创建可执行 Python 工具，解决了现有工具创建方法依赖 LLM 内在知识在专业领域失败的问题，在因果推理、物理和化学任务上平均超过已有方法 12.3%。

**[Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](information_retrieval/retrieval-augmented_generation_for_predicting_cellular_responses_to_gene_perturb.md)**

:   提出 **PT-RAG**（Perturbation-aware Two-stage Retrieval-Augmented Generation），首次将可微检索增强生成范式应用于单细胞基因扰动响应预测：通过 GenePT 语义检索候选扰动 + Gumbel-Softmax 条件离散采样实现细胞类型感知的端到端检索优化，在 Replogle-Nadig 数据集上超越 STATE 基线（Pearson 0.633 vs 0.624），同时发现朴素 RAG 会严重损害性能（Pearson 仅 0.396），证明**可微且细胞类型感知的检索**在该领域不可或缺。

**[Revela: Dense Retriever Learning via Language Modeling](information_retrieval/revela_dense_retriever_learning_via_language_modeling.md)**

:   提出 Revela，通过 in-batch attention 机制将检索器学习融入语言建模——NTP 不仅依赖本序列上下文，还依赖批内其他序列（由检索器相似度加权），无需标注 query-document 对即可训练强大的密集检索器。

**[Summaries as Centroids for Interpretable and Scalable Text Clustering](information_retrieval/summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)**

:   提出 k-NLPmeans 和 k-LLMmeans，通过在 k-means 迭代中周期性地用文本摘要替换数值质心（summary-as-centroid），在保持 k-means 标准目标的同时实现可解释的聚类原型，且 LLM 调用量与数据集大小无关。

**[Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)**

:   提出 Token-Guard，一种基于自检验解码的 token 级幻觉控制方法，通过隐空间中的 token 级/段级评分和迭代修正机制，在解码过程中检测并抑制幻觉生成，F1 平均提升 16.3%。

**[TokMem: One-Token Procedural Memory for Large Language Models](information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)**

:   提出 TokMem，将可复用的任务程序编译为单个可训练记忆 token，既作为程序索引又作为生成控制信号，无需长 prompt 即可高效调用 1000+ 任务程序，且支持无遗忘的持续扩展。

**[Your Language Model Secretly Contains Personality Subnetworks](information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)**

:   本文提出通过激活引导的剪枝（activation-guided pruning）从预训练 LLM 中提取人格专用子网络，无需任何训练即可实现高效的人格切换，并引入对比剪枝策略增强对立人格间的参数分离。

---

## 🎯 目标检测 { #object_detection }

**[A Problem-Oriented Perspective and Anchor Verification for Code Optimization](object_detection/a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)**

:   提出以问题为导向（而非用户为导向）的优化对构建方法来整合多程序员的策略多样性，并设计锚点验证框架利用"慢但正确的代码"生成测试用例来缓解"优化税"（正确性损失），将优化比从 31.24% 提升到 71.06%，加速比从 2.95x 提升到 6.08x。

**[AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](object_detection/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)**

:   提出 AdaRank，用可学习二值掩码自适应选择 task vector 的奇异分量（取代启发式 top-k），结合测试时熵最小化优化，大幅缓解多任务模型合并中的任务间干扰，在 ViT-B/32 上达到 89.4% 准确率。

**[Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](object_detection/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)**

:   基于 $W_Q$ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 $Q(X)=(X+f_\theta(X))/2$，在不增加参数的情况下超越 +12.5% 参数的基线模型。

**[Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](object_detection/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)**

:   定义了"Scale Anchoring"新问题（低分辨率训练导致高分辨率推理误差锚定），并提出架构无关的频率表征学习（FRL），通过 Nyquist 归一化频率编码使误差随分辨率提升而下降，在 8 种主流架构上验证有效。

**[CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](object_detection/cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)**

:   首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取域不变的目标级结构先验，并用类引导对比学习驱动域不变表征，在多个跨域基准上大幅超越现有方法。

**[ConFu: Contemplate the Future for Better Speculative Sampling](object_detection/confu_contemplate_the_future_for_better_speculative_sampling.md)**

:   提出 ConFu，在推测解码的 draft model 中引入 contemplate tokens 让其预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和生成速度。

**[Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective](object_detection/context_tokens_are_anchors_understanding_the_repetition_curse_in_dmllms_from_an_.md)**

:   通过信息流分析揭示扩散多模态大语言模型（dMLLMs）在使用缓存加速时产生"重复诅咒"的内在机制——context token 作为锚点聚合语义信息，缓存破坏了这一信息流模式——并提出 CoTA 方法将重复率降低高达 92%。

**[CORDS: Continuous Representations of Discrete Structures](object_detection/cords_continuous_representations_of_discrete_structures.md)**

:   提出 CORDS 框架，通过将变大小离散集合（检测框、分子原子）双射映射为连续的密度场和特征场，使模型可在场空间中学习并精确解码回离散集合，避免了固定 slot 或 padding 的限制。

**[Diverse Text-to-Image Generation via Contrastive Noise Optimization](object_detection/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)**

:   提出 Contrastive Noise Optimization (CNO)，通过在 Tweedie 去噪预测空间上对初始噪声施加 InfoNCE 对比损失，以预处理方式提升扩散模型生成多样性，同时保持保真度，无需修改采样过程或模型本身。

**[Does FLUX Already Know How to Perform Physically Plausible Image Composition?](object_detection/does_flux_already_know_how_to_perform_physically_plausible_image_composition.md)**

:   提出 SHINE，一个无需训练的图像合成框架，通过 Manifold-Steered Anchor Loss、Degradation-Suppression Guidance 和 Adaptive Background Blending 三个组件，利用预训练 T2I 模型（如 FLUX）内在的物理先验，实现在复杂光照条件下（阴影、水面反射等）的高质量物体插入。

**[ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)**

:   ForestPersons 是首个专门面向森林树冠下失踪人员检测的大规模基准数据集（96,482 张图像 + 204,078 标注），通过模拟微型无人机（MAV）在 1.5-2.0 米高度的低空飞行视角，覆盖多季节、多天气、多姿态和多遮挡等级的真实搜救条件，为下冠层人员检测模型的训练和评估提供了坚实基础。

**[From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](object_detection/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)**

:   发现多模态 LLM 的推理性能与视觉注意力分数（VAS）高度相关（r=0.96），提出 AVAR 框架通过视觉锚定数据合成、注意力引导训练目标和视觉锚定奖励塑造三个阶段提升 VAS，在 77 个基准上平均提升 7%。

**[FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](object_detection/fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)**

:   提出一个无需训练的少样本目标检测框架，组合 UPN、SAM2 和 DINOv2 三个基础模型生成提案和匹配特征，并通过图扩散算法精化置信度分数和抑制碎片化提案，在 Pascal-5i 和 COCO-20i 上大幅超越 SOTA。

**[InfoDet: A Dataset for Infographic Element Detection](object_detection/infodet_a_dataset_for_infographic_element_detection.md)**

:   构建了一个大规模信息图元素检测数据集（101,264 张信息图、1420 万标注），涵盖图表和人类可识别对象两大类，并提出 Grounded CoT 方法利用检测结果提升 VLM 的图表理解能力。

**[Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](object_detection/is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)**

:   构建了迄今最大的 AI 生成同行评审数据集（788,984 篇评审），系统评估了 18 种 AI 文本检测方法在同行评审场景下的表现，并提出了利用论文原文作为上下文的 Anchor 检测方法，在低误报率下大幅超越所有基线。

**[Long-Context Generalization with Sparse Attention](object_detection/long-context_generalization_with_sparse_attention.md)**

:   提出 ASEntmax（Adaptive-Scalable Entmax），用可学习温度的 α-entmax 替代 softmax 注意力，从理论和实验两方面证明稀疏注意力能实现 1000× 长度外推，解决 softmax 在长上下文下的注意力弥散（dispersion）问题。

**[Procedural Mistake Detection via Action Effect Modeling](object_detection/procedural_mistake_detection_via_action_effect_modeling.md)**

:   提出双分支多模态监督的动作效果建模框架，结合视觉分支（目标状态和空间关系特征）和文本分支（GPT-4o 生成的场景图），通过可学习的效果 token 蒸馏外部监督信号，在第一人称程序视频中实现 SOTA 错误检测。

**[SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](object_detection/sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)**

:   首次研究联邦 Prompt Learning 场景下的后门攻击威胁，并提出 SABRE-FL——一种基于 embedding 空间异常检测的轻量级服务器端防御方法，无需访问客户端原始数据即可有效过滤中毒 prompt 更新。

**[SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](object_detection/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)**

:   提出 SAGE，一个统一的 VPR 训练框架：引入轻量 Soft Probing 模块增强局部特征判别力，每个 epoch 在线重建融合地理距离与视觉相似度的亲和图，再通过贪心加权团扩展聚焦最难样本，冻结 DINOv2 骨干仅训练 1.96M 参数即在 8 个基准上全面 SOTA。

**[SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation](object_detection/serum_simple_efficient_robust_and_unifying_marking_for_diffusion-based_image_gen.md)**

:   提出SERUM水印方法，将唯一水印噪声添加到扩散模型初始噪声中，训练轻量检测器直接从生成图像识别水印（无需昂贵的DDIM反演），在多种攻击下达到最高检测率，且注入/检测极快，支持多用户场景。

**[SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](object_detection/spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f.md)**

:   提出SpectralGCD，将图像表示为CLIP概念字典上的语义混合（跨模态相似度向量），通过谱过滤自动选择任务相关概念，配合正反知识蒸馏保持语义质量，在6个基准上以与单模态方法可比的计算代价达到多模态SOTA。

**[SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)**

:   提出 SPWOOD 框架统一处理稀疏标注和弱标注（HBox/Point）的旋转目标检测问题，通过自适应旋转目标检测器(SAOD)和空间布局学习策略，在 DOTA 基准上以混合标注（RBox:HBox:Point=1:1:1）达到接近全监督的性能。

**[Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](object_detection/thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)**

:   提出 AdaAnchor 潜空间推理框架——将可学习的锚向量（anchor vectors）附加到输入嵌入中，通过迭代前向传播精炼锚状态实现"沉默思考"，配合基于锚稳定性的自适应停止机制按实例难度动态分配计算量，在数学推理任务上比固定步潜推理准确率提升最高 5%、平均步数减少 48–60%，输出 token 相比 CoT 减少 92–93%。

**[Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](object_detection/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)**

:   提出 RAGLens，利用稀疏自编码器(SAE)从 LLM 内部激活中解耦出 RAG 幻觉专属特征，通过互信息特征选择 + 广义加性模型(GAM)构建轻量级可解释幻觉检测器，在多个基准上超越现有方法，并支持 token 级可解释反馈与幻觉缓解。

**[Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)**

:   提出 TreeBench（首个可追溯视觉推理基准，405道高挑战 VQA，OpenAI-o3 仅 54.87%）和 TreeVGR（通过双 IoU 奖励的强化学习联合监督定位与推理的训练范式），7B 模型在 V\*Bench +16.8、MME-RealWorld +12.6、TreeBench +13.4，证明可追溯性是推进视觉推理的关键。

**[VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](object_detection/vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)**

:   VidGuard-R1 是首个采用 GRPO（Group Relative Policy Optimization）强化学习微调 MLLM 的视频真伪检测器，通过构建 14 万无快捷方式的真/假视频对数据集，并设计时序伪影奖励和扩散步数质量奖励两种专用奖励机制，在自建数据集上达到 86.17% 准确率，在 GenVidBench 和 GenVideo 基准上实现 95%+ 的 SOTA 零样本检测性能，同时生成可解释的思维链推理。

**[VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](object_detection/vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)**

:   提出VLBiMan框架，通过任务感知双臂分解将单次演示拆分为不变/可适应原子技能，利用VLM视觉-语言锚定在新场景中适应物体位置和实例变化，结合运动学感知的轨迹组合实现双臂协调——在10个复杂双臂任务上以1次演示达到85.3%成功率远超需上百次演示的模仿学习基线。

**[What Layers When: Learning to Skip Compute in LLMs with Residual Gates](object_detection/what_layers_when_learning_to_skip_compute_in_llms_with_residual_gates.md)**

:   提出 GateSkip——在 decoder-only Transformer 每个 Attention/MLP 分支输出处插入一个 sigmoid-linear 门控，微调时联合学习门控稀疏性与语言建模目标，推理时按门控值用分位数阈值确定性跳过低重要性 token，实现 token 级逐层自适应深度；在 Llama 8B 上节省 15% 计算保持 >90% 精度，指令微调模型全计算反而提升精度、约 50% 节省仍匹配基线，且与 INT4 量化/结构化剪枝/自推测解码正交可组合。

**[When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](object_detection/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)**

:   本文首次系统研究了 LLM 多智能体系统中的曼德拉效应（集体虚假记忆），提出 ManBench 基准（4838 个问题、5 种交互协议），发现所有 13 个被评估的 LLM 均易受此效应影响，并提出 prompt 级和模型级缓解策略，平均减少 74.40% 的虚假记忆。

**[Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](object_detection/zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)**

:   提出将目标检测与交互识别完全解耦的零样本 HOI 检测框架 DA-HOI，利用 MLLM 的 VQA 能力替代传统 CLIP 特征做交互识别，核心贡献是确定性生成（training-free 即达 31.50 mAP）、空间感知池化（引入空间先验和跨注意力）和单次确定性匹配（M 次前向变 1 次），在 HICO-DET 四种零样本设定下全面超越 SOTA，且训练后可即插即用切换任意检测器。

---

## 💬 LLM/NLP { #llm_nlp }

**[AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](llm_nlp/assetformer_modular_3d_assets_generation_with_autoregressive_transformer.md)**

:   提出 AssetFormer，基于 Llama 架构的自回归 Transformer，将模块化 3D 资产（由 primitive 序列组成）建模为离散 token 序列，通过 DFS/BFS 图遍历重排序和联合词汇表解码实现从文本描述生成可直接用于游戏引擎的模块化 3D 资产。

**[Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning](llm_nlp/compositional-arc_assessing_systematic_generalization_in_abstract_spatial_reason.md)**

:   提出 Compositional-ARC 数据集评估模型在抽象空间推理中的系统性泛化能力——从已知基础几何变换（如平移、旋转）泛化到未见过的变换组合。一个仅 5.7M 参数的 MLC 训练的 encoder-decoder 模型在系统性任务上达到 78.26%，与 ARC Prize 2024 冠军的 8B 模型+TTT 持平，远超 GPT-4o、o3-mini 等（<3%）。

**[d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](llm_nlp/d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)**

:   提出 d²Cache，一种面向 Diffusion-based LLM（dLLM）的无训练近似 KV 缓存框架，通过确定性先验引导的 masked token 选择 + 注意力感知的非 mask token 选择两阶段策略，实现 4.1× 推理加速同时提升生成质量。

**[DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](llm_nlp/dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)**

:   DreamOn 通过引入 [expand] 和 [delete] 两个特殊状态解决了扩散语言模型（DLM）的固定长度生成限制，无需架构修改即可实现变长代码填充，在 HumanEval-Infilling 上比扩散基线平均提升 26.4%，达到与 SOTA 自回归模型持平的水平。

**[ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_llm_framework.md)**

:   提出 ELLMob，一个基于模糊痕迹理论（FTT）的自对齐 LLM 框架，通过提取并迭代对齐"习惯模式要旨"与"事件约束要旨"来生成兼顾日常规律与事件响应的人类移动轨迹。

**[Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](llm_nlp/enhancing_persona_following_at_decoding_time_via_dynamic_importance_estimation_f.md)**

:   提出 PDD（Persona Dynamic Decoding）框架，通过条件互信息动态估计不同场景下人设属性的重要性，并以加权多目标奖励引导推理时解码，实现无需微调的自适应人设遵循。

**[Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)**

:   提出基于上下文感知的成对比较框架来评估文本创造力，构建了包含 100K+ 人类级别和 1M+ 合成数据的 CreataSet 数据集，训练出 CrEval 评估器，在与人类判断的对齐度上超越 GPT-4o 达 18.7%。

**[Fine-Grained Activation Steering: Steering Less, Achieving More](llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)**

:   AUSteer 发现块级激活转向（steering）本质上是异质的——不同维度控制不同 token 分布，混合转向既放大有益信号也放大有害信号。提出原子单元（AU）级细粒度转向：用激活动量定位判别性维度，自适应调节转向强度，仅转向 ≤100 个维度即大幅超越转向数千维度的 SOTA 方法。

**[First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](llm_nlp/first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)**

:   通过理论和实验证明先前工作所推崇的"第一层（embedding）最适合做 influence estimation"的结论是不可靠的，发现中间 attention 层才是更好的估计层，并提出 Rank 和 Vote 两种新的跨层聚合策略以及 Noise Detection Rate (NDR) proxy 指标，显著改善了 LLM 中有害训练样本的检测效果。

**[From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](llm_nlp/from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)**

:   提出 PCE（Planner-Composer-Evaluator）框架，将 LLM 推理链中隐含的环境假设显式提取并组织为决策树，通过似然度-增益-成本评分实现不确定性感知的行动选择，大幅减少多智能体协作中的通信开销。

**[Generative Value Conflicts Reveal LLM Priorities](llm_nlp/generative_value_conflicts_reveal_llm_priorities.md)**

:   提出 ConflictScope，一个自动生成价值冲突场景的 pipeline，通过开放式评估（非选择题）揭示 LLM 在冲突情境下的价值优先级排序，发现模型在开放式设置中从保护性价值（如无害性）转向个人价值（如用户自主性），且系统提示可将目标排序对齐提升 14%。

**[How Catastrophic is Your LLM? Certifying Risk in Conversation](llm_nlp/how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)**

:   提出 C3LLM（Certification of Catastrophic risks in multi-turn Conversation for LLMs），首个为多轮 LLM 对话中灾难性风险提供统计认证的框架：用语义相似度图上的 Markov 过程建模对话分布，定义 3 种对话采样策略 + 增强层，使用 Clopper-Pearson 95% 置信区间认证模型产生有害输出的概率界——发现最差模型风险下界高达 72%。

**[KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](llm_nlp/kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)**

:   提出 KVComm 框架通过选择性共享 KV pairs 实现 LLM 间高效通信，发现 hidden states 存在"信息集中偏差"使其不适合跨模型传递，设计基于注意力重要性 + 高斯先验的层选择策略，仅传输 30% 层即可超越大多数 baseline。

**[LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](llm_nlp/llema_evolutionary_search_with_llms_for_multi-objective_materials_discovery.md)**

:   提出 LLEMA 框架，将 LLM 的科学先验知识与化学规则引导的进化搜索和记忆驱动的迭代优化相结合，在 14 个多目标材料发现任务上显著超越生成式和纯 LLM 基线。

**[Meta-RL Induces Exploration in Language Agents](llm_nlp/meta-rl_induces_exploration_in_language_agents.md)**

:   提出 LaMer 框架，将元强化学习（Meta-RL）引入 LLM agent 训练，通过跨 episode 的奖励优化和基于反思的上下文策略适应，使语言智能体学会主动探索环境，在 Sokoban/MineSweeper/Webshop 上分别获得 11%/14%/19% 的绝对性能提升。

**[Near-Optimal Online Deployment and Routing for Streaming LLMs](llm_nlp/near-optimal_online_deployment_and_routing_for_streaming_llms.md)**

:   首次形式化 LLM 流式在线部署+路由联合问题：新模型持续出现、旧模型可能过时，在并发部署上限 $M_{\max}$ 和成本预算约束下，提出 StageRoute 分层算法，证明 $\tilde{\mathcal{O}}(T^{2/3})$ 遗憾界并给出匹配下界，达到近最优。

**[Neural Synchrony Between Socially Interacting Language Models](llm_nlp/neural_synchrony_between_socially_interacting_language_models.md)**

:   首次研究社会交互中 LLM 间的神经同步现象：通过训练仿射变换预测交互伙伴的未来表征，定义 $SyncR^2$ 指标量化同步强度，发现该同步依赖于社会参与和时间邻近性，且与 LLM 的社会行为表现高度相关（Pearson $r$ = 0.88-0.99），呼应了人类脑间同步（IBS）的神经科学发现。

**[Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](llm_nlp/optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)**

:   提出 Optimas 框架，为复合 AI 系统中每个组件维护一个与全局奖励对齐的局部奖励函数（LRF），使异构组件（prompt、模型参数、超参数、模型选择）可独立优化，在五个真实系统上平均提升 11.92%。

**[PT2-LLM: Post-Training Ternarization for Large Language Models](llm_nlp/pt2-llm_post-training_ternarization_for_large_language_models.md)**

:   提出 PT2-LLM，首个针对 LLM 的后训练三值化框架，通过非对称三值量化器（含迭代三值拟合和激活感知网格对齐）与结构相似性重排序策略，在 1.58-bit 下实现优于 2-bit PTQ 方法的性能。

**[ConflictScope: Generative Value Conflicts Reveal LLM Priorities](llm_nlp/quamo_quaternion_motions_for_vision-based_3d_human_kinematics_capture.md)**

:   提出ConflictScope——自动化价值冲突场景生成与评估流水线：给定任意价值集合，自动生成价值对之间的冲突场景，通过模拟用户的开放式交互（而非选择题）评估LLM的价值优先级排序；发现模型在开放式评估中从"保护性价值"（如无害性）显著转向"个人价值"（如用户自主性），系统提示可使对齐目标排序提升14%。

**[Rethinking Code Similarity for Automated Algorithm Design with LLMs](llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)**

:   提出 BehaveSim，一种基于"问题求解轨迹"（PSTrajs）和动态时间规整（DTW）的算法相似度度量方法，从执行行为层面而非语法或输出层面衡量算法差异，集成到 FunSearch/EoH 等 LLM-AAD 框架后显著提升性能。

**[Statistical Advantage of Softmax Attention: Insights from Single-Location Regression](llm_nlp/statistical_advantage_of_softmax_attention_insights_from_single-location_regress.md)**

:   通过提出"单位置回归"(Single-Location Regression, SLR) 理论框架，结合统计物理中的 order parameter 方法，在高维极限下严格证明了 softmax attention 在种群层面达到 Bayes 风险而线性 attention 本质上无法做到，并在有限样本情形下证实 softmax 始终优于线性 attention，为 softmax 在检索任务中的优势提供了首个原理性解释。

**[Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](llm_nlp/stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)**

:   提出 SureLock，当 Masked Diffusion LM 中已 unmask 的 token 后验分布稳定后永久锁定该位置（跳过 Q 投影和 FFN，缓存 KV），将每步注意力计算从 $O(N^2d)$ 降为 $O(MNd)$，在 LLaDA-8B 上减少 30-50% FLOPs 且不损生成质量。

**[The Lattice Representation Hypothesis of Large Language Models](llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)**

:   提出 LLM 的**格表示假说 (Lattice Representation Hypothesis)**：通过将线性表示假说与形式概念分析 (FCA) 统一，证明 LLM 嵌入空间中的属性方向通过半空间交集隐式编码了一个**概念格 (concept lattice)**，从而实现了连续几何与符号抽象之间的桥接。

**[Trapped by simplicity: When Transformers fail to learn from noisy features](llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)**

:   研究表明 Transformer 在从含特征噪声的数据中学习布尔函数时会失败——其简单性偏好（倾向学习低敏感度函数）导致模型被困在比目标函数更简单的最优噪声预测器上，无法恢复真实的无噪声目标函数。

**[Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)**

:   提出三种**无监督**指标——LLM 引导聚类（目标识别）、基于微调完成模型的交互完整性检测、响应树（LLM 不确定性量化）——用于评估多轮目标驱动对话，无需标注数据或 LLM-as-a-judge，仅用 8B 模型即可匹配/超越 70B judge 的性能。

**[WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](llm_nlp/webdevjudge_mllm_web_development.md)**

:   构建 WebDevJudge 元评估基准，系统评估 LLM/MLLM 及智能体工作流在 Web 开发质量评估任务上作为裁判的能力，发现当前最强模型与人类专家之间仍存在约15%的一致率差距，并揭示了功能等价识别失败和可行性验证薄弱两大根本瓶颈。

**[Weight Decay may matter more than μP for Learning Rate Transfer in Practice](llm_nlp/weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)**

:   本文通过大规模实证分析表明，μP 的核心对齐假设仅在训练初期短暂成立，实际训练中是独立权重衰减（independent weight decay）而非 μP 在正确稳定跨宽度的特征学习动态，μP 的实际益处可被解释为一种隐式学习率预热。

**[When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](llm_nlp/when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)**

:   通过控制性行为评估框架，揭示 LLM 在数据约束的科学决策任务中的四种隐藏失败模式：高稳定性≠正确性、prompt 措辞敏感性、放宽阈值下的过度选择、以及幻觉产生无效标识符。

---

## 🎵 音频/语音 { #audio_speech }

**[AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)**

:   提出 AC-Foley，一种参考音频引导的视频到音频合成框架，通过两阶段训练（声学特征学习+时序适应）和多模态条件流匹配实现了细粒度音色控制、音色迁移和零样本音效生成，在音频质量和声学保真度上显著优于现有方法。

**[Discovering and Steering Interpretable Concepts in Large Generative Music Models](audio_speech/discovering_and_steering_interpretable_concepts_in_large_generative_music_models.md)**

:   首次将 Sparse Autoencoder (SAE) 应用于音频/音乐领域，从自回归音乐生成模型 MusicGen 的残差流中提取可解释的音乐概念特征，并利用这些特征实现可控生成（steering）。

**[Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)**

:   提出 Dynamic Parameter Memory (DPM) 机制，在推理阶段通过逐句将语音信息编码到临时 LoRA 模块的参数空间中，使有限上下文窗口的语音大语言模型能够处理无限长度的情感对话音频，在 IEMOCAP 和 MELD 上达到 SOTA。

**[EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)**

:   提出 EchoMind，首个面向共情对话的多层级关联基准，通过理解→推理→对话的认知流程，系统评估 Speech Language Models 感知非语言声学线索并生成共情回复的能力。

**[Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](audio_speech/efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)**

:   提出 Dolphin 模型，通过双路径轻量视频编码器 DP-LipCoder 将唇部运动映射为离散语义 token，并设计全局-局部注意力（GLA）分离器，在三个基准上超越 SOTA 同时参数减少 50%+、MACs 降低 2.4×、GPU 推理加速 6×。

**[EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](audio_speech/emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)**

:   首次将语音情感识别（SER）重构为深度推理问题，通过韵律增强基座模型 + GRPO-PTR（渐进式可信推理奖励）强化学习，生成带有声学依据的可解释情感推理。

**[FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)**

:   提出 FlexiCodec，通过 ASR 特征引导的动态帧率合并策略，在 3–12.5Hz 超低帧率下实现高质量语音编解码，同时保持优异的语义信息保留能力。

**[Incentive-Aligned Multi-Source LLM Summaries](audio_speech/incentive-aligned_multi-source_llm_summaries.md)**

:   将博弈论中的多任务 peer prediction 机制引入 LLM 多源摘要管线，提出 Truthful Text Summarization (TTS) 框架：通过 leave-one-out 交叉构造评价声明集、提取每个来源对声明的立场、用 informative agreement 评分来源可靠性并过滤不可靠来源后重新摘要，理论上证明"如实报告是效用最大策略"，实验中有效抵御 prompt injection、虚假信息源和协同攻击。

**[Knowing When to Quit: Probabilistic Early Exits for Speech Separation](audio_speech/knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)**

:   提出 PRESS（Probabilistic Early-exit for Speech Separation）方法和 PRESS-Net 架构，通过概率框架联合建模干净语音信号和误差方差，推导出基于信噪比（SNR）的可解释早退出条件，实现语音分离网络的细粒度动态计算缩放，同时保持与SOTA静态模型竞争力的性能。

**[Latent Speech-Text Transformer](audio_speech/latent_speech_text_transformer.md)**

:   提出 Latent Speech-Text Transformer (LST)，将离散语音 token 聚合为更高层级的"潜在语音 patch"作为自回归单元（类似 BLT 对 bytes 的处理），对齐语音和文本的序列建模粒度（从 20× 缩小到 ~1:1），在 speech HellaSwag 上获得 +6.5% 绝对提升且增益从 420M→7B 持续增长，同时降低 ASR/TTS 推理计算成本。

**[MAPSS: Manifold-Based Assessment of Perceptual Source Separation](audio_speech/mapss_manifold-based_assessment_of_perceptual_source_separation.md)**

:   提出 Perceptual Separation（PS）和 Perceptual Match（PM）两个互补度量，利用扩散映射将自监督编码表示嵌入低维流形，首次在功能上解耦音源分离中的泄漏和自失真，与 18 种主流指标对比在与主观评分的相关性上几乎始终排名第一或第二。

**[MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)**

:   提出 MMSU（5000 条音频 QA、47 个任务），首个系统融合语言学理论的语音理解与推理基准，评测 22 个 SpeechLLM，发现现有模型在音韵感知和复杂推理上仍存在显著差距。

**[PACE: Pretrained Audio Continual Learning](audio_speech/pace_pretrained_audio_continual_learning.md)**

:   首次系统性构建音频持续学习基准，揭示预训练音频模型因底层频谱特征主导导致的上游-下游不匹配问题，提出 PACE 方法（改进首会话适应 + 自适应子空间正交 PEFT + 边界感知扰动），在 6 个音频 CL 基准上大幅超越 SOTA。

**[Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](audio_speech/pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)**

:   提出 USR 2.0，用 CTC 驱动的教师强制替代自回归伪标签生成，注意力伪标签在单次前向传播中完成，训练速度提升近 2×，通过 CTC-注意力联合预测增强分布外鲁棒性，在 LRS3/LRS2/WildVSR 上实现 ASR/VSR/AVSR 三任务统一模型 SOTA。

**[Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)**

:   提出 QSTar 框架，通过在整个处理流程中嵌入问题引导（Query Guidance），并引入空间-时序-频域三维度交互模块（特别是利用频谱特征区分音色），显著提升了音乐场景下的音频-视觉问答（Music AVQA）性能。

**[ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](audio_speech/reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)**

:   提出 ReasoningBank 记忆框架，从 Agent 自我判断的成功和失败经验中蒸馏可泛化的推理策略存入记忆库，并提出 memory-aware test-time scaling (MaTTS) 建立记忆与测试时扩展的协同效应，在 WebArena、Mind2Web 和 SWE-Bench 上一致超越基线（最高 34.2% 相对提升），同时减少 16% 交互步数。

**[RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](audio_speech/redteamcua_adversarial_testing_agents.md)**

:   构建首个混合 Web-OS 环境的 CUA 红队测试框架 RedTeamCUA 和 864 个测试用例的 RTC-Bench，系统评估 9+ 前沿 CUA 对间接 prompt injection 的脆弱性，发现所有 CUA 均可被攻击（最高 ASR 83%），且能力越强的模型越危险——攻击尝试率（AR）远高于成功率（ASR）意味着模型能力提升将直接转化为更高的攻击成功率。

**[Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)**

:   提出 Speech-guided Machine Translation（SMT）框架，用 TTS 将源文本合成语音后与文本联合输入 MLLM 做翻译，通过自我进化机制自动筛选有益的合成语音样本进行持续训练。在 Multi30K 超越所有 MMT 方法取得 SOTA，在 FLORES-200 的 108 个翻译方向上以仅 9B 参数达到平均 SOTA。

**[SiNGER: A Clearer Voice Distills Vision Transformers Further](audio_speech/singer_a_clearer_voice_distills_vision_transformers_further.md)**

:   提出 SiNGER（Singular Nullspace-Guided Energy Reallocation）框架，通过在教师特征的零空间方向施加扰动来抑制 ViT 中的高范数伪影，同时保留信息信号，结合轻量 LoRA 适配器实现高效蒸馏，在多个下游任务上取得 SOTA 性能并生成更清晰可解释的表征。

**[Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](audio_speech/stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)**

:   提出 Stitch，在口语语言模型中实现"边想边说"——将无声推理 token 与语音 token 交替分块生成，利用音频播放期间的空闲算力完成推理。Stitch-S 首帧延迟与无推理基线一致，数学推理准确率提升约 15 个百分点。

**[SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)**

:   提出 SyncTrack，通过轨道共享模块（双跨轨注意力确保节奏同步）和轨道特定模块（可学习乐器先验保留音色差异）的统一架构，以及三个新的节奏一致性评估指标（IRS/CBS/CBD），显著提升多轨音乐生成质量（FAD 从 6.55→1.26，主观 MOS 3.42 vs 1.57）。

**[Toward Complex-Valued Neural Networks for Waveform Generation](audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)**

:   提出 ComVo，首个在生成器和判别器中均使用复值神经网络（CVNN）的 iSTFT 声码器，通过相位量化层稳定训练，并引入块矩阵计算方案将训练时间减少 25%，在 LibriTTS 上合成质量超过 Vocos 等实值基线。

**[TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization](audio_speech/triplesumm_adaptive_triple-modality_fusion_for_video_summarization.md)**

:   提出 TripleSumm，通过多尺度时序块（层级滑动窗口注意力）和跨模态融合块（融合 token 自适应加权视觉/文本/音频），实现帧级模态重要性动态调整，并发布首个大规模三模态视频摘要数据集 MoSu（52678 视频），在 4 个 benchmark 上达到 SOTA。

**[VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation](audio_speech/vowelprompt_hearing_speech_emotions_from_text_via_vowel-level_prosodic_augmentat.md)**

:   提出 VowelPrompt，基于语音学证据提取元音级韵律描述符（音高/能量/时长），转为自然语言增强 LLM 的情感识别 prompt，配合 SFT+GRPO 两阶段训练，在零样本/微调/跨域/跨语言条件下一致超越 SOTA，同时生成可解释的情感推理。

**[When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](audio_speech/when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)**

:   发现 LLM 越狱 benchmark 中的 ASR 被语义无关的风格模式（如"创建列表"）人为膨胀，36 个 LLM 中几乎都存在此现象；表面风格对齐微调进一步加剧此风险；提出 SafeStyle——用风格增强的安全训练数据缓解风险。

---

## 📹 视频理解 { #video_understanding }

**[AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](video_understanding/adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)**

:   提出 AdAEM，一个自适应、自扩展的 LLM 价值观评估框架，通过信息论优化自动生成能最大化揭示不同 LLM 价值差异的测试问题，解决现有静态基准无法区分模型价值取向的"信息量不足"问题。

**[A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)**

:   提出 A.I.R.，一种无需训练的自适应-迭代-推理驱动帧选择框架，通过两阶段策略（GMM 自适应初始采样 + 迭代式 VLM 精细分析）解决 VideoQA 中轻量模型（CLIP）相似度不准确和 VLM 分析成本爆炸的双重困境，在最坏情况下也仅需分析 72 帧（vs 基线 128 帧），同时显著提升多个长视频 benchmark 性能。

**[AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](video_understanding/anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)**

:   提出 AnveshanaAI，一个基于 Bloom 认知分类学的自适应 AI/ML 教育平台，通过自动化题目生成（基于微调的 GPT-2）、语义相似度检测去重、XAI 可解释性技术和游戏化机制（积分/徽章/排行榜），实现了覆盖数据科学到多模态 AI 七大领域的个性化学习评估系统，实验表明微调后困惑度显著下降且学习者参与度明显提升。

**[Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](video_understanding/coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)**

:   提出 Expert-Router Coupling (ERC) Loss，一种轻量级辅助损失函数，通过将路由器参数视为聚类中心的代理 token 并约束专家对其激活范数，实现路由器决策与专家能力的紧密耦合，仅需 $n^2$ 次激活计算即可显著提升 MoE-LLM 性能。

**[Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](video_understanding/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)**

:   提出从阅读时眼动轨迹解码开放式信息检索目标的新任务，基于 OneStop 眼动数据集（360人、486问题、162段落），开发判别式和生成式多模态模型；RoBERTEye-Fixations 在三选一目标选择上达 49.3%（随机 33%），不同 critical span 达 70.9%；DalEye-Llama/GPT 在目标重建中也显著优于无眼动基线。

**[Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](video_understanding/emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)**

:   从理论上分析了两层 Transformer 在有向图可达性问题上使用连续 Chain-of-Thought（Coconut）训练时的训练动力学，揭示了"叠加态"（superposition）机制如何自然涌现：index-matching logit 先增长后有界，从而在探索与利用之间取得平衡。

**[FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)**

:   提出 FlashVID，一个免训练的视频大语言模型推理加速框架，通过树状时空 token 合并（TSTM）联合建模空间和时间冗余，仅保留 10% 的视觉 token 就能保持 LLaVA-OneVision 99.1% 的性能，并能将 Qwen2.5-VL 的输入帧数提升 10 倍。

**[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)**

:   提出 FLoC，基于设施选址函数（facility location function）的视觉 token 压缩框架，通过子模优化在给定预算下快速选择兼具代表性和多样性的 token 子集，实现无训练、模型无关、查询无关的长视频理解 token 压缩。

**[From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)**

:   发现 slot-based 目标中心学习中编码器（产生尖锐但有噪声的注意力图）与解码器（产生空间一致但模糊的重建掩码）之间的恶性循环，提出同步对比学习目标和 slot 正则化预热策略将其转化为良性循环，在 MOVi 和 YouTube-VIS 上大幅提升物体发现性能。

**[Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](video_understanding/lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)**

:   提出了"类别拆分"(Category Splitting)新任务，通过挖掘视频分类器权重中的潜在组合结构，在零样本条件下将粗粒度动作类别拆分为细粒度子类别，无需重训或额外数据。

**[Log Probability Tracking of LLM APIs](video_understanding/log_probability_tracking_of_llm_apis.md)**

:   提出 Logprob Tracking (LT) 方法，仅用单token输入和单token输出的log概率即可检测LLM API的微小变更（如单步微调），灵敏度比现有方法高2-3个数量级，成本低1000倍。

**[LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](video_understanding/lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)**

:   提出 Lumina 框架，通过"上下文-知识信号"检测RAG系统中的幻觉：用MMD度量**外部上下文利用**程度，用跨层token预测演化度量**内部知识利用**程度，无需超参调优即可泛化。

**[Mamba-3: Improved Sequence Modeling using State Space Principles](video_understanding/mamba-3_improved_sequence_modeling_using_state_space_principles.md)**

:   从SSM视角提出三项核心改进：指数-梯形离散化、复值状态空间、多输入多输出(MIMO)公式化，在不增加解码延迟的前提下显著提升模型质量和状态追踪能力，推进性能-效率Pareto前沿。

**[Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs](video_understanding/map_the_flow_revealing_hidden_pathways_of_information_in_videollms.md)**

:   首次用机制可解释性工具（Attention Knockout + Logit Lens）系统逆向工程VideoLLM的时序推理过程，揭示出"早中层跨帧交互→中层视频-语言整合→中后层答案生成"的三阶段信息流蓝图，并证明仅保留42%注意力边即可几乎无损保持VideoQA性能。

**[NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](video_understanding/nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)**

:   提出 NerVE，一个轻量级的特征谱分析框架，通过四个互补指标（频谱熵、参与比、特征值早期富集、JS 散度）系统揭示了 LLM 中 FFN 非线性如何重新注入方差、重塑特征谱，以及架构和优化器选择如何印刻独特的频谱签名。

**[Online Time Series Prediction Using Feature Adjustment](video_understanding/online_time_series_prediction_using_feature_adjustment.md)**

:   提出 ADAPT-Z（Automatic Delta Adjustment via Persistent Tracking in Z-space），将在线时序预测的适应目标从模型参数更新转移到特征空间修正，通过轻量 adapter 融合当前特征与历史梯度来应对多步预测中的延迟反馈问题，在13个数据集上一致超越现有在线学习方法。

**[Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](video_understanding/paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)**

:   构建 Paper Copilot——跨数十个 AI/ML 会议的同行评审持久数字档案与分析平台：通过 OpenReview API、网页抓取、社区贡献三源混合收集评审数据，实时归档评分时间快照（含 rebuttal 前后动态变化），揭示 ICLR 2025 年决策熵反常下降——评审体系从概率性分层转向近确定性分数驱动决策的结构性变化，并通过 LLM 驱动的作者-机构元数据提取支持人才轨迹追踪。

**[Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](video_understanding/stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)**

:   提出 CAPO（Curvature-Aware Policy Optimization），通过在 LM head 最后一层建模二阶优化几何来预测并过滤会导致策略崩溃的 token 更新，在激进超参数（5× 学习率、1/12 batch size）下仍保持训练稳定，实现 MATH 上相较标准 GRPO 的 30× 样本效率提升。

**[Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs](video_understanding/stop_tracking_me_proactive_defense_against_attribute_inference_attack_in_llms.md)**

:   TRACE-RPS 提出统一防御框架应对 LLM 属性推断攻击：TRACE 通过注意力+推理链精准定位隐私泄露文本元素做细粒度匿名化，RPS 通过轻量后缀优化诱导模型拒绝推断，将属性推断准确率从约 50% 降至 5% 以下。

**[The Expressive Limits of Diagonal SSMs for State-Tracking](video_understanding/the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)**

:   建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。

**[FuncBenchGen: 面向可靠基准测试的无污染可控评估框架](video_understanding/towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)**

:   提出 FuncBenchGen 框架，通过将多步函数调用建模为 DAG 图遍历问题，实现无数据污染、可精细控制任务难度的 LLM 工具使用能力评估，并揭示了推理模型在长调用链和连接型干扰函数下的关键失败模式。

**[Video-KTR: 通过关键 Token 归因增强视频推理](video_understanding/video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)**

:   提出 Video-KTR，一种模态感知的策略塑造框架，通过反事实分析识别视觉感知型、时序敏感型和高熵 Token 三类关键 Token，仅对这些 Token 执行选择性强化学习更新，在多个视频推理基准上达到 SOTA（Video-Holmes 42.7%，超越 GPT-4o）。

**[VideoNSA: Native Sparse Attention Scales Video Understanding](video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)**

:   本文提出 VideoNSA，将 Native Sparse Attention（NSA）引入视频语言模型，通过压缩、选择和滑动窗口三分支动态门控的混合稀疏注意力机制，在仅使用 3.6% 注意力预算的条件下实现 128K token 的视频理解，在长视频理解、时序推理和空间理解任务上全面超越 token 压缩和无训练稀疏注意力基线。

**[联邦学习中水印的鲁棒性与放射性可能相互矛盾](video_understanding/watermark_robustness_and_radioactivity_may_be_at_odds_in_federated_learning.md)**

:   首次研究联邦学习中 LLM 水印的数据溯源问题，发现水印在 FL 中具有放射性（可检测），但恶意服务器可通过强鲁棒聚合算法过滤水印更新，揭示了放射性、鲁棒性和模型效用之间的根本性三元矛盾。

**[WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](video_understanding/weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)**

:   提出 WebOperator，一个动作感知的树搜索框架，通过投机性回溯、破坏性动作检测、动作验证与合并等机制，使 Web 自主代理能在部分可观测、不可逆的真实网页环境中安全高效地探索，在 WebArena 上以 gpt-4o 达到 54.6% SOTA 成功率。

---

## 📚 预训练 { #llm_pretraining }

**[A Law of Data Reconstruction for Random Features (and Beyond)](llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)**

:   从信息论和代数角度证明随机特征模型中存在数据重构定律：当参数量 $p \gg dn$（$d$ 为数据维度，$n$ 为样本数）时，训练数据可被完整重构，并通过投影损失优化方法在 RF、两层网络和 ResNet 上验证了该阈值的普适性。

**[Block-Sample MAC-Bayes Generalization Bounds](llm_pretraining/block-sample_mac-bayes_generalization_bounds.md)**

:   提出块样本MAC-Bayes泛化界（mean approximately correct），将训练数据划分为J个块后用各块条件下的KL散度之和替代整体KL散度，在确定性学习算法（如均值估计）等原始PAC-Bayes界为空（vacuous）的场景下仍能给出有限、有意义的泛化误差界，并证明了该界的高概率版本在一般情况下不可行。

**[CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](llm_pretraining/chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)**

:   构建 CHAMMI-75——最大的异构多通道显微镜图像预训练数据集（280 万图像，75 个来源，25 种通道类型，16 种物种），证明成像模态多样性是提升多通道模型泛化能力的关键因素，训练的 MorphEm 模型在 7 个 benchmark 中 6 个达到 SOTA。

**[Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](llm_pretraining/common_corpus_ethical_data_for_llm_pretraining.md)**

:   构建 Common Corpus——约 2 万亿 token 的最大规模合法授权 LLM 预训练数据集，覆盖 6 大集合（政府/文化/科学/代码/Web/语义），多语言（含低资源语言），所有数据均为无版权或宽松许可来源，配有完整数据溯源和多阶段过滤管道，已被 Anthropic 等行业领导者采用。

**[Deconstructing Positional Information: From Attention Logits to Training Biases](llm_pretraining/deconstructing_positional_information_from_attention_logits_to_training_biases.md)**

:   提出基于 Toeplitz 矩阵的统一分析框架，将位置编码分为加法（Absolute/T5/ALiBi）和乘法（RoPE）两类；通过合成任务发现 RoPE 在位置敏感任务上优势显著但存在"单头沉积模式"（single-head deposit pattern）——浅层几乎所有位置推理集中于单个注意力头；理论证明该模式是 RoPE 乘法结构的固有属性。

**[Emergent Misalignment is Easy, Narrow Misalignment is Hard](llm_pretraining/emergent_misalignment_is_easy_narrow_misalignment_is_hard.md)**

:   研究发现在窄域有害数据上微调会造成广域错位（emergent misalignment），因为"通用错位"比"仅在特定域错位"是更简单高效的参数空间解——通用解的参数范数更小且对噪声更稳定。

**[Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](llm_pretraining/explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)**

:   通过 Neural Collapse 的视角统一解释 Grokking（延迟泛化）和 Information Bottleneck（压缩阶段）两大训练后期现象，证明群体类内方差的收缩是两者的共同关键因素，并揭示训练损失收敛与 Neural Collapse 发生存在由 weight decay 控制的不同时间尺度。

**[FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](llm_pretraining/fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)**

:   提出 FictionalQA 数据集及生成管线，通过合成关于虚构事件的 webtext 风格文档和 QA 对，在受控环境下研究 LLM 训练中事实记忆与逐字记忆的双重过程，发现更多样的表面形式有助于知识获取而简洁的结构化列表反而最不利于泛化。

**[Identifying and Evaluating Inactive Heads in Pretrained LLMs](llm_pretraining/identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)**

:   系统评估12种评分函数识别LLM中不活跃注意力头，发现基于头输出范数的评分函数（AHON LN）比传统注意力权重指标更能跨模型家族一致地识别不活跃头，14个模型上平均超过12%的头可被置零而保持MMLU精度在1%以内。

**[Imagine How To Change: Explicit Procedure Modeling for Change Captioning](llm_pretraining/imagine_how_to_change_explicit_procedure_modeling_for_change_captioning.md)**

:   提出 ProCap 框架，将变化描述从静态图像对比较重新定义为动态过程建模：第一阶段通过帧插值和掩码重建训练过程编码器学习时空变化动力学，第二阶段用可学习过程查询隐式推断变化过程，在三个数据集上超越 SOTA。

**[Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)**

:   通过分析深度矩阵分解（深度线性网络）在矩阵补全任务中的梯度流动力学，证明了耦合动力学是深度网络低秩隐式偏差的关键机制，且深度≥3的网络除对角初始化外必然展现耦合，从而解释了深度模型为何能避免可塑性损失。

**[Intrinsic Training Dynamics of Deep Neural Networks](llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)**

:   本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则，并将结果推广到任意深度的 ReLU 网络和线性网络。

**[Lossless Vocabulary Reduction for Auto-Regressive Language Models](llm_pretraining/lossless_vocabulary_reduction_for_auto-regressive_language_models.md)**

:   提出**无损词表缩减（LVR）**的理论框架，通过嵌套分词（nested tokenization）将任意自回归语言模型精确转换为使用任意子词表的等价模型，并基于**最大公共词表（MCV）**实现不同分词方案语言模型之间的高效集成，在 GSM8K、MATH、翻译等多个任务上验证了方法的有效性。

**[MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](llm_pretraining/moma_a_modular_deep_learning_framework_for_material_property_prediction.md)**

:   提出 MoMa 模块化材料属性预测框架，先在多任务上训练专用模块并集中存储为 MoMa Hub，再通过表示驱动的无训练自适应模块组合算法（AMC）为下游任务定制模型，在 17 个数据集上平均超越最强基线 14%。

**[MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](llm_pretraining/mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)**

:   提出 MT-DAO，一种多时间尺度分布式自适应优化器，通过引入慢动量（高 $\beta$）来解决低频通信训练中标准动量衰减过快导致的时间尺度失配问题，首次提供了收敛保证，在语言模型预训练中消除了与全同步 DDP 的性能差距，同时减少 6-27% 的端到端训练时间。

**[Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)**

:   提出 Warmup-Stable-Only (WSO) 学习率调度策略——在预训练中完全去掉学习率衰减阶段，虽然预训练指标较差，但在 SFT 后一致性地超越所有衰减策略，通过损失景观分析揭示 WSO 保持更平坦的极小值区域是其优势根源。

**[Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)**

:   提出训练再评估曲线（TREC）诊断工具，通过分析训练完成后模型在各时间步训练数据上的损失来指导高质量数据的最优放置位置，并证明 TREC 形状可通过 AdamW 的隐式 EMA 系数预测，无需实际训练即可设计数据课程。

**[RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](llm_pretraining/recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)**

:   提出 RECON，一种类-姿态无关的正则化方向归一化方法，通过简单的右平移（right translation）修正任意训练过程中产生的正则化表示，实现无监督的实例级对称性发现、OOD 姿态检测以及即插即用的测试时正则化层。

**[Reducing Class-Wise Performance Disparity via Margin Regularization](llm_pretraining/reducing_class-wise_performance_disparity_via_margin_regularization.md)**

:   提出 MR2（Margin Regularization for performance disparity Reduction），通过在 logit 和表征空间动态调整类别相关的 margin，基于理论推导的泛化界减少类间性能差异，同时提升整体准确率。

**[SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)**

:   提出SemHiTok——通过语义引导层次codebook(SGHC)统一理解和生成的tokenizer：预训练语义codebook上建像素子codebook，结构和训练解耦(分阶段优化)避免联合训练的语义-像素冲突，LLaVA设定下离散tokenizer中理解和重建都SOTA。

**[Stochastic Self-Organization in Multi-Agent Systems](llm_pretraining/stochastic_self-organization_in_multi-agent_systems.md)**

:   提出 SelfOrg 框架，基于 Agent 响应的语义相似度和 Shapley 值贡献估计，动态构建有向无环通讯图（DAG），实现多 Agent 系统的自组织协作。在弱模型场景下优势尤为显著。

**[TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](llm_pretraining/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)**

:   提出 TASTE（Text-Aligned Speech Tokenization and Embedding），通过跨注意力机制将语音 token 与文本转录对齐，实现极低比特率（~150 bps）下的高质量语音重建，并使文本-语音联合建模变得直接高效，1.3B 参数的 TASLM 超越 7B 预训练 SLM。

**[Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](llm_pretraining/understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)**

:   从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。

**[Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](llm_pretraining/understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)**

:   通过将训练梯度信号分解为 direct、pre-cached 和 circuit sharing 三种成分，解释了为什么 NTP 训练的 Transformer 会学到对预测当前下一token"无用"的特征，并在 OthelloGPT、小型语言模型和预训练 LLM（Gemma 2）上验证了这一框架的解释力。

---

## 🕸️ 图学习 { #graph_learning }

**[A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)**

:   从图 Ricci 曲率的几何视角证明随机 k-SAT 问题的二部图表示具有固有的负曲率，且曲率随问题难度增加而下降，建立了 GNN 过压缩 (oversquashing) 与 SAT 求解困难之间的理论联系，并通过测试时图重布线验证了该理论。

**[Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)**

:   指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量，实验表明Erank与准确率的平均相关性达0.91（vs Dirichlet energy的0.72），在OGB-Arxiv上Dirichlet energy甚至呈现错误的相关方向，并从理论上证明对广泛的GNN架构族其数值秩收敛到1（秩坍塌），重新定义过平滑为秩坍塌而非特征向量对齐。

**[Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](graph_learning/beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)**

:   首次提出针对多重图（multigraph）的神经组合优化路由方法 GMS，包含直接在多重图上边级自回归构造的 GMS-EB 和先学习剪枝再节点级路由的双头 GMS-DH 两个变体，在非对称多目标 TSP 和 CVRP 上实现了接近精确求解器 LKH 的性能且速度快数十倍。

**[Cooperative Sheaf Neural Networks](graph_learning/cooperative_sheaf_neural_networks.md)**

:   提出在有向图上定义 cellular sheaf 的 in/out-degree Laplacian，构建 Cooperative Sheaf Neural Network (CSNN)，使节点能独立选择信息传播/接收策略，从而同时缓解过度挤压(oversquashing)和处理异配(heterophilic)任务。

**[Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](graph_learning/embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)**

:   本文通过 Memento 框架系统评估了 LLM 驱动具身智能体的记忆利用能力，发现现有 agent 能回忆简单物体语义但无法处理用户行为模式的序列信息，并提出了基于层次知识图谱的用户画像记忆模块来有效提升个性化辅助任务的表现。

**[Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)**

:   提出 EDT-Former（Entropy-guided Dynamic Token Transformer），通过熵引导的动态token生成机制，在冻结图编码器和LLM之间建立高效对齐，无需微调LLM主干网络即在分子问答、分子指令和属性预测等多个基准上达到SOTA。

**[Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)**

:   提出 Explore-on-Graph（EoG），通过 SFT + 两阶段强化学习（结果奖励 + 路径精炼奖励），激励 LLM 在知识图谱上自主探索超出训练分布的推理路径，在五个 KGQA 基准上超越 GPT-5 和 Gemini 2.5 Pro。

**[GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](graph_learning/graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)**

:   提出 GRAPHITE，一种通过引入"特征节点"作为 hub 间接连接共享特征的节点来**直接提升图同质性**的非学习图变换方法，首次从"改变图结构"而非"改变 GNN 架构"的角度解决异质图问题，在 Actor 等困难基准上显著超越 27 种 SOTA 方法。

**[Graph Tokenization for Bridging Graphs and Transformers](graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)**

:   提出 GraphTokenizer 框架，将图通过可逆的频率引导序列化转换为符号序列，再用 BPE 学习图子结构词汇表，使标准 Transformer（如 BERT/GTE）无需任何架构修改即可直接处理图数据，在 14 个 benchmark 上达到 SOTA。

**[GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](graph_learning/graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)**

:   提出 GraphUniverse 框架，通过分层生成具有持久语义社区的图族（graph families），首次实现对图学习模型归纳泛化能力的系统性评估，揭示了 transductive 性能无法可靠预测 inductive 泛化能力这一关键发现。

**[Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics](graph_learning/improving_long-range_interactions_in_graph_neural_simulators_via_hamiltonian_dyn.md)**

:   提出 Information-preserving Graph Neural Simulators (IGNS)，利用 port-Hamiltonian 动力学结构在图上保持信息不耗散，结合 warmup 初始化、几何编码和多步训练目标，在 6 个物理仿真基准上全面超越现有图神经仿真器。

**[Learning Concept Bottleneck Models from Mechanistic Explanations](graph_learning/learning_concept_bottleneck_models_from_mechanistic_explanations.md)**

:   提出 Mechanistic CBM (M-CBM)，利用 Sparse Autoencoder 从黑盒模型自身学到的特征中提取概念，再由多模态 LLM 命名和标注，构建可解释的 Concept Bottleneck Model，在控制信息泄露的条件下显著优于现有 CBM 方法。

**[LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)**

:   LogicXGNN 提出了一种从已训练的图神经网络中提取可解释一阶逻辑规则的 post-hoc 框架：通过图结构哈希和隐藏层嵌入模式识别谓词、用决策树确定判别式 DNF 规则结构、并将抽象谓词接地到输入空间，最终生成可替代原始 GNN 的规则化分类器，同时可作为可控的图生成模型。

**[NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](graph_learning/neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)**

:   提出 NH-GCAT 框架，将神经科学中的抑郁症神经环路先验知识显式融入 GNN，在区域、环路和网络三个空间尺度上建模，在 REST-meta-MDD 数据集上取得 SOTA 分类效果（AUC 78.5%、ACC 73.8%），并提供与神经科学相符的可解释性分析。

**[On the Expressive Power of GNNs for Boolean Satisfiability](graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)**

:   从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族。

**[Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)**

:   提出 HMAGAT，用有向超图注意力网络替代 GNN 的成对消息传递来建模多智能体路径规划中的群体交互，仅用 1M 参数和 1% 训练数据即超越 85M 参数的 SOTA 模型。

**[RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](graph_learning/ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)**

:   提出 RAS 框架，在推理时为每个问题动态构建查询特定的知识图谱，通过迭代检索规划、文本到三元组转换和图增强回答三个阶段实现结构化推理，在 7 个知识密集型基准上对开源和闭源 LLM 分别取得最高 7.0% 和 8.7% 的提升。

**[Relational Graph Transformer](graph_learning/relational_graph_transformer.md)**

:   提出 RelGT，首个专为关系型数据库设计的图 Transformer，通过多元素 Token 化（特征/类型/跳距/时间/局部结构 5 元组）和局部-全局混合注意力机制，在 RelBench 基准的 21 个任务上一致超越 GNN 基线，最高提升 18%。

**[Relatron: Automating Relational Machine Learning over Relational Databases](graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)**

:   系统比较关系深度学习（RDL/GNN）和深度特征合成（DFS）在关系数据库预测任务上的性能，发现两者各有优势且高度任务依赖，提出 Relatron——基于任务嵌入的元选择器，通过 RDB 任务同质性和亲和力嵌入实现自动架构选择，在联合架构-超参搜索中提升达 18.5%。

**[Revisiting Node Affinity Prediction in Temporal Graphs](graph_learning/revisiting_node_affinity_prediction_in_temporal_graphs.md)**

:   分析为什么简单启发式（持续预测、移动平均）在时序图节点亲和力预测上优于复杂 TGNN，证明启发式是线性 SSM 的特例且标准 RNN/LSTM/GRU 无法表达最基本的持续预测，据此提出 NAViS——基于虚拟全局状态的线性 SSM 架构配合排序损失，在 TGB 上超越所有基线。

**[Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](graph_learning/structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)**

:   通过构建 10000 篇论文的配对引用图（人类 vs GPT-4o 生成 vs 随机基线），发现 LLM 生成的参考文献在图拓扑结构上与人类几乎不可区分（RF 仅 60% 准确率），但语义嵌入可有效检测（RF 83%，GNN 93%），说明 LLM 精确模仿了引用拓扑但留下了可检测的语义指纹。

**[Towards Improved Sentence Representations using Token Graphs](graph_learning/towards_improved_sentence_representations_using_token_graphs.md)**

:   提出 Glot，一种轻量结构感知池化模块，将冻结 LLM 的 token 级隐状态构建为潜在相似性图，通过 GNN 细化后聚合为句子表征，在 GLUE/MTEB 上与微调方法竞争力相当但仅需 20× 更少参数和 100× 更快训练。

---

## ⚡ LLM效率 { #llm_efficiency }

**[Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](llm_efficiency/did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)**

:   将记忆增强 Agent 的多存储检索形式化为代价敏感的存储路由问题（store routing），证明选择性检索相比全量检索可在减少 62% context token 的同时提升 QA 准确率（86% vs 81%），并提出基于语义信号的启发式路由基线。

**[DND: Boosting Large Language Models with Dynamic Nested Depth](llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)**

:   DND在Transformer层末端通过路由器选出关键token，将其回送同一层进行额外处理（嵌套深度），配合路由控制损失和阈值控制方案实现精确稳定的token选择，以极少的参数增加（<0.1M）在Qwen3-1.7B和Qwen3-30B-A3B上分别获得1.88%和0.87%的平均性能提升。

**[EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](llm_efficiency/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)**

:   提出 EvoEngineer，首个系统化的 LLM-based 代码演化框架，将代码演化分解为 traverse technique（含两层设计：solution guiding + prompt engineering）和 population management 两个正交组件，在 91 个真实 CUDA kernel 上实现最高 2.72× 中位加速比和 69.8% 代码有效率，在性能和正确性两个维度上超越现有方法。

**[Expert Divergence Learning for MoE-based Language Models](llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)**

:   解决 MoE 训练中的专家同质化问题，通过最大化不同数据域之间路由分布的 Jensen-Shannon 散度，鼓励不同域激活不同专家子集，在 15B-A1.5B 模型上提升专家特化程度和语言建模性能。

**[Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](llm_efficiency/fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)**

:   通过 Functional Scaling Law 框架理论推导出 batch size scheduling 的最优策略——对困难任务，最优策略是训练大部分时间用小 batch，仅在最后阶段切换到大 batch（late switching）；并揭示了 fast catch-up 效应——切换后 loss 迅速追上全程大 batch 的轨迹，在 1.1B 参数 1T token 的 LLM 预训练中验证了该原则。

**[LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)**

:   提出 LycheeDecode，通过将注意力头细粒度分为少量 retrieval heads（负责全注意力选关键 token）和大量 sparse heads（复用选出的 token 做稀疏计算），并用 HardKuma 分布端到端学习头类型，在 128K 上下文下实现 2.7× 加速且性能不降。

**[MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning](llm_efficiency/mvar_visual_autoregressive_modeling_with_scale_and_spatial_markovian_conditionin.md)**

:   提出 MVAR（Markovian Visual AutoRegressive），通过引入尺度 Markov 假设（仅依赖相邻尺度而非所有前序尺度）和空间 Markov 注意力（限制邻域大小 k），将 VAR 模型的注意力计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(Nk)$，在 ImageNet 256×256 上实现同等或更优性能的同时，推理显存降低 3.0-4.2×，且仅需 8 张 RTX 4090 即可训练。

**[One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](llm_efficiency/one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)**

:   提出 SMoPE 框架，将单个共享 prompt 组织为稀疏 MoE 结构中的多个 prompt expert，通过 prompt-attention score aggregation 实现动态稀疏激活，在保持高参数效率的同时显著缓解知识干扰，在多个持续学习 benchmark 上达到 SOTA。

**[Polynomial, trigonometric, and tropical activations](llm_efficiency/polynomial_trigonometric_and_tropical_activations.md)**

:   系统探索基于正交基（Hermite多项式、Fourier三角基）和热带化（tropicalization）的可学习激活函数族，通过方差保持初始化解决多项式激活的梯度爆炸/消失问题，在GPT-2和ConvNeXt上成功替代GELU实现有效训练。

**[Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](llm_efficiency/q_rag_long_context_multi_step_retrieval.md)**

:   将多步检索建模为 MDP，用基于值的 RL（soft Q-learning）微调 **embedder 而非 LLM**，Q 函数设计为状态嵌入和动作嵌入的内积（理论证明为万能近似器），结合 RoPE 相对位置编码实现时序推理，在单卡 A100 上训练 12 小时，4K 训练泛化到 1M+ token 上下文，RULER 基准达到近乎完美的 NIAH 性能。

**[RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)**

:   提出 RACE Attention——用幂次角核替代 softmax 并通过可微 LSH 草图近似注意力输出，实现严格线性时间复杂度，支持单 GPU 处理 1200 万 token、单 CPU 处理 7500 万 token，在多种任务上匹配或超越 softmax 精度。

**[Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](llm_efficiency/randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)**

:   提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。

**[Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](llm_efficiency/rethinking_benign_relearning_syntax_as_the_hidden_driver_of_unlearning_failures.md)**

:   揭示 LLM 机器遗忘中"良性重学习"现象的真正驱动因素是句法相似性而非主题相关性，并提出句法多样化策略（paraphrase forget set），有效抑制重学习、加速遗忘并缓解遗忘效果与模型效用之间的 trade-off。

**[Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](llm_efficiency/rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)**

:   从 proper scoring rules 框架出发，证明最高概率输出序列的负对数似然（MSP）是理论上合理的不确定性度量，并提出 G-NLL——仅用一次贪心解码就能逼近该度量，在多个场景下匹配或超越需要多次采样的 SOTA 方法。

**[Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](llm_efficiency/semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)**

:   提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销，在Attention-DP场景下吞吐提升最高2.78×，Attention-TP场景下延迟降低最高24.9%。

**[Steering Language Models with Weight Arithmetic](llm_efficiency/steering_language_models_with_weight_arithmetic.md)**

:   提出对比式权重引导（Contrastive Weight Steering），通过对正/负行为微调模型的权重差来提取行为方向向量，直接修改模型权重实现行为控制，在谄媚性、恶意性和拒绝性实验中比激活引导（Activation Steering）具有更好的泛化能力和一致性。

**[SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving](llm_efficiency/swingarena_competitive_programming_arena_for_long-context_github_issue_solving.md)**

:   提出SwingArena对抗性评测框架，让两个LLM在真实GitHub issue上交替扮演补丁提交者和测试审查者，通过仓库原生CI流水线（编译/lint/回归测试）端到端验证，在C++/Python/Rust/Go四语言400个实例上揭示了模型在"激进补丁生成"与"防御性质量保证"间的行为分化。

**[Token-level Data Selection for Safe LLM Fine-tuning](llm_efficiency/token-level_data_selection_for_safe_llm_fine-tuning.md)**

:   提出 TOSS（Token-level data Selection for Safe LLM fine-tuning），首个 token 级别的数据选择框架,通过安全退化模型和效用导向模型之间的损失差评估每个 token 的安全风险，实现比样本级方法更优的安全-效用权衡。

**[Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)**

:   系统解剖基于 chunk 的稀疏注意力架构，识别出三个关键设计原则（非线性 Chunk Encoder + CLS token、Bypassing Residual Path、训练时强制选择稀疏性），将 4K 上下文训练的模型成功外推到 3200 万 token。

**[Universe Routing: Why Self-Evolving Agents Need Epistemic Control](llm_efficiency/universe_routing_why_self-evolving_agents_need_epistemic_control.md)**

:   将自主Agent在链式推理中容易混淆认识论框架（如频率主义vs贝叶斯）的问题形式化为"宇宙路由"，训练一个465M参数的轻量路由器将问题分类到7个互斥信念空间后分发给专用求解器，证明硬路由比软MoE快7倍且精度相同，模块化架构配合rehearsal可实现零遗忘的持续学习。

**[When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](llm_efficiency/when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)**

:   提出理论框架将长上下文任务失败分解为三类噪声（任务噪声/模型噪声/聚合器噪声），证明当模型噪声超线性增长时弱模型+分块处理可超越强模型单次处理，并给出快速估计最优 chunk size 的方法（3-5 个样本即可）。

**[xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](llm_efficiency/xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)**

:   系统对比 xLSTM 与 Transformer 的 scaling law，证明 xLSTM 在训练损失-算力 Pareto 前沿、过训练 regime 和推理速度上全面优于同规模 Transformer，且优势随上下文长度增大而增长。

---

## 💻 代码智能 { #code_intelligence }

**[Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)**

:   构建 Ambig-SWE（基于 SWE-Bench Verified 的欠指定变体），系统评估 LLM 编程 agent 在三个维度上的交互能力——检测欠指定、提出澄清问题、利用交互信息——发现交互可将欠指定场景下的解决率提升最高 74%，但模型默认非交互行为且难以区分指定充分/不足的指令。

**[Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)**

:   针对图表到代码生成任务中SFT的性能瓶颈问题，提出多模态结构化强化学习（MSRL），通过文本+视觉双层奖励函数和两阶段RL策略，在ChartMimic和ReachQA上分别提升6.2%和9.9%的高层指标，达到开源SOTA并媲美GPT-4o。

**[CARD: Towards Conditional Design of Multi-agent Topological Structures](code_intelligence/card_towards_conditional_design_of_multi-agent_topological_structures.md)**

:   CARD提出了一种条件图生成框架(Conditional Agentic Graph Designer)，通过条件变分图编码器和环境感知优化，根据模型能力、工具可用性和知识源变化等动态环境信号自适应地设计多Agent通信拓扑结构，在HumanEval、MATH和MMLU上一致超越静态和基于提示的基线方法。

**[DiaBlo: Diagonal Blocks Are Sufficient For Finetuning](code_intelligence/diablo_diagonal_blocks_are_sufficient_for_finetuning.md)**

:   提出 DiaBlo——一种用对角块更新替代低秩分解的参数高效微调方法：将权重矩阵划分为 $N \times N$ 块后只训练对角块 $\mathbf{D}_1, \ldots, \mathbf{D}_N$，彻底绕开 LoRA 中 $\mathbf{AB}$ 乘积带来的非凸优化、初始化敏感与梯度不稳定问题，零初始化即可收敛，PyTorch 一行 `torch.einsum` 实现 batched matmul，理论证明同参数预算下表达力严格优于 LoRA，在常识推理、算术推理、代码生成、安全对齐四大任务及 4-bit/2-bit 量化场景全面领先。

**[DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)**

:   将分布鲁棒优化（DRO）引入 InstructZero 的贝叶斯优化框架，通过在 f-divergence 球定义的模糊集上最大化最坏情况期望效用，使自动搜索得到的 prompt 在分布偏移下仍能保持可靠性能。

**[Execution-Grounded Credit Assignment for GRPO in Code Generation](code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)**

:   提出 EGCA（Execution-Grounded Credit Assignment），通过执行追踪定位程序中最早的语义偏差位置，将 GRPO 的梯度集中到因果 token span 上，解决代码生成中粗粒度信用分配问题，在 HumanEval 上达到 82.1% pass@1。

**[Improving Code Localization with Repository Memory](code_intelligence/improving_code_localization_with_repository_memory.md)**

:   通过利用代码仓库的 commit 历史构建情景记忆（过去 commit）和语义记忆（活跃代码功能摘要），增强语言代理的代码定位能力，在 SWE-bench 上取得显著提升。

**[IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)**

:   提出 IMSE——将预训练 ViT 线性层通过 SVD 分解为"谱专家"，仅微调奇异值实现极端参数高效的测试时适应，并通过多样性最大化损失和域感知谱码检索机制，在 TTA/CTTA/渐进 CTTA 三种场景下达到 SOTA。

**[Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)**

:   提出 SOSecure，一种无需重训练的推理时安全机制，通过 BM25 从 Stack Overflow 安全讨论知识库中检索与 LLM 生成代码相关的社区安全警告，引导模型在推理阶段自主修订不安全代码，在三个真实数据集上实现高达 96.7% 的漏洞修复率且零新漏洞引入。

**[InnoGym: Benchmarking the Innovation Potential of AI Agents](code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)**

:   提出 InnoGym，第一个系统评估 AI Agent 创新能力的基准和框架，引入 Performance Gain 和 Novelty 两个互补指标，通过 18 个可改进任务发现当前 Agent 具备一定创新性但缺乏将创新转化为可靠性能提升的鲁棒性。

**[KV Cache Transform Coding for Compact Storage in LLM Inference](code_intelligence/kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)**

:   提出 KVTC，一种借鉴经典媒体压缩技术（PCA 特征去相关 + 自适应量化 + 熵编码）的 KV 缓存压缩方法，在 Llama 3、Mistral NeMo、R1-Qwen 2.5 等模型上实现最高 20× 压缩（特定场景下 40×+），优于 token 驱逐、量化、SVD 等基线方法。

**[Learning to Reason without External Rewards](code_intelligence/learning_to_reason_without_external_rewards.md)**

:   提出 Intuitor，一种用模型自身置信度（self-certainty，即输出分布与均匀分布的 KL 散度）替代外部可验证奖励的 RLIF 方法，在数学推理上匹配 GRPO 性能，同时在代码生成等域外任务上展现更好的泛化能力。

**[MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task](code_intelligence/mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)**

:   借鉴代码补全中的 Fill-in-the-Middle (FIM) 范式，训练一个专门的步骤扩展模型 MathFimer-7B，在已有数学解题链中插入更细粒度的中间推理步骤，从而系统性提升下游模型的数学推理能力。

**[Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](code_intelligence/paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)**

:   提出 PaperCoder——一个多智能体 LLM 框架，通过规划（Planning）、分析（Analysis）、生成（Coding）三阶段流水线，将机器学习论文自动转化为可运行的代码仓库，其中 88% 的生成仓库被论文作者评为最佳，且在 PaperBench 基准上大幅超越基线。

**[Sharing State Between Prompts and Programs](code_intelligence/sharing_state_between_prompts_and_programs.md)**

:   提出共享程序状态（shared program state）抽象，让 prompt 直接读写程序变量、操作堆对象和控制程序流程，实现为 Nightjar 系统（Python + prompt 混合编程），在保持或提升准确率（+4-19%）的同时减少 39.6% 代码量。

**[ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](code_intelligence/shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)**

:   提出 ShieldedCode——首个保护感知的代码表征学习框架，通过层次依赖建模（指令内/前序/跨指令三层）和联合功能感知+保护感知对比学习，使 LLM 能够生成、比较和推理虚拟机保护代码，在 VM 代码生成（Pass@1 26.95% vs. GPT-4o 22.58%）和二进制相似性检测上均超越现有方法。

**[Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](code_intelligence/supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)**

:   提出 Supervised Reinforcement Learning (SRL)，将问题求解重新建模为逐步动作生成过程，通过基于序列相似度的密集奖励信号，使小模型能够从专家轨迹中学习原本 SFT 和 RLVR 都无法解决的困难推理问题。

**[The Limits of Long-Context Reasoning in Automated Bug Fixing](code_intelligence/the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)**

:   系统评估当前 LLM 在长上下文代码调试中的能力极限，发现 agentic 工作流的成功来自任务分解而非长上下文推理（成功轨迹仅消耗 20-30K token），64K token 单次补丁生成中性能急剧下降（GPT-5-nano 0%），揭示名义上下文长度与实际可用上下文能力之间的显著差距。

**[Training Large Language Models To Reason In Parallel With Global Forking Tokens](code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)**

:   提出 Set Supervised Fine-Tuning (SSFT)，通过二分图匹配将全局分叉令牌 (global forking tokens) 与多样推理轨迹对齐，使 LLM 能从单个控制令牌全局引导不同推理模式，在数学推理和代码生成任务上显著优于标准 SFT 和 GRPO。

---

## 🔗 因果推理 { #causal_inference }

**[Action-Guided Attention for Video Action Anticipation](causal_inference/action-guided_attention_for_video_action_anticipation.md)**

:   提出动作引导注意力 (AGA) 机制，用模型自身的动作预测序列作为注意力的 Query 和 Key（而非像素特征），结合自适应门控融合历史上下文和当前帧特征，在 EPIC-Kitchens-100 上实现从验证集到测试集的良好泛化，同时支持训练后的可解释性分析。

**[AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](causal_inference/agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)**

:   提出AgentTrace框架，从多智能体系统的执行日志中构建因果图，通过反向追踪+轻量级特征排序（五组特征的加权线性组合）定位根因节点，在550个合成故障场景上Hit@1达94.9%，延迟0.12秒，比LLM分析快69倍。

**[Copy-Paste to Mitigate Large Language Model Hallucinations](causal_inference/copy-paste_to_mitigate_large_language_model_hallucinations.md)**

:   提出 Copy-Paste 生成范式，通过训练 LLM 优先直接复制检索上下文中的片段来生成回答，而非自由改写，配合高复制偏好的 DPO 训练，在反事实 RAG 基准上将忠实度从 80.2% 提升到 92.8%。

**[Counterfactual Explanations on Robust Perceptual Geodesics](causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)**

:   提出 PCG（Perceptual Counterfactual Geodesic）方法，在鲁棒感知流形上通过测地线优化生成语义忠实的反事实解释，两阶段优化确保路径既感知自然又达到目标类别，在 AFHQ 上 FID=8.3 远优于 RSGD 的 12.9。

**[Direct Doubly Robust Estimation of Conditional Quantile Contrasts](causal_inference/direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)**

:   提出首个对条件分位数比较器 (CQC) 的**直接估计方法**，通过显式参数化 CQC 并结合双重鲁棒梯度下降，在理论上保持双重鲁棒性的同时，实验中在估计精度、可解释性和计算效率上全面优于现有的间接反演方法。

**[Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)**

:   首次在线性非高斯设定下、不依赖任何结构假设，给出了含潜变量和环的因果图之间分布等价性的完整图准则，核心工具是新提出的**边秩约束**（edge rank constraints），据此开发了遍历等价类和从数据恢复因果模型的算法——这是参数化因果模型中首个无结构假设的等价性刻画和发现方法。

**[Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)**

:   提出 E-CIT（集成条件独立性检验）框架，通过将数据分割为子集后独立执行检验并基于**稳定分布**的 p 值聚合方法合并结果，将任意条件独立性检验的计算复杂度降至关于样本量线性，同时在重尾噪声和真实数据等复杂场景下保持甚至提升检验功效。

**[Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](causal_inference/flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere.md)**

:   系统研究偏好模型对五种表面特征（冗长、结构化、术语、谄媚、模糊）的过度依赖——通过因果反事实对量化偏差来源于训练数据的分布不平衡，并提出基于**反事实数据增强 (CDA)** 的后训练方法，将模型与人类判断的平均失校准率从 39.4% 降至 32.5%。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](causal_inference/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](causal_inference/journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)**

:   针对网络干扰下 A/B 测试中 GATE 估计的高方差问题，提出 Mean-in-Interior (MII) 估计器——仅对 cluster 内部节点取均值，大幅降低方差；再通过反事实预测器进行协变量偏移校正，得到增广版 AMII 估计器，同时实现低偏差和低方差。

**[Learning Robust Intervention Representations with Delta Embeddings](causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)**

:   提出因果 Delta 嵌入（CDE）框架，将干预/动作表示为预干预和后干预状态在潜空间中的向量差，通过独立性、稀疏性和不变性三种约束学习鲁棒的干预表示，在 Causal Triplet 挑战中显著超越基线的 OOD 泛化性能，且能自动发现反义动作的反平行语义结构。

**[On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)**

:   提出基于结构因果模型（SCM）的分解式评估框架，将 LLM 的反事实推理拆分为四个阶段（因果变量识别→因果图构建→干预识别→结果推理），在 11 个多模态数据集上系统诊断 LLM 在各阶段的能力瓶颈，并提出工具增强和高级 elicitation 策略来改善性能。

**[PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits](causal_inference/personax_multimodal_datasets_with_llm-inferred_behavior_traits.md)**

:   构建了 PersonaX 多模态数据集（含 LLM 推断的 Big Five 行为特质、面部嵌入和传记元数据），并提出两层分析框架：结构化独立性检验 + 非结构化因果表示学习（带可识别性理论保证），揭示跨模态因果结构。

**[Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](causal_inference/resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)**

:   提出 Knowledgeable-R1，一个基于强化学习的框架，通过联合采样参数知识（PK）和上下文知识（CK）的轨迹，结合局部/全局优势计算和自适应不对称优势变换，使 LLM 在 RAG 场景中能够抵抗误导性检索上下文的干扰，同时保留对可靠上下文的利用能力。

**[Self-Supervised Learning from Structural Invariance](causal_inference/self-supervised_learning_from_structural_invariance.md)**

:   提出 AdaSSL，通过引入潜变量建模正样本对之间的条件不确定性，推导出互信息的变分下界，使 SSL 能够处理自然配对数据中的复杂（多模态、异方差）条件分布，在因果表征学习、细粒度图像理解和视频世界模型上均优于基线。

**[SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](causal_inference/selfreflect_can_llms_communicate_their_internal_answer_distribution.md)**

:   提出SelfReflect度量指标——一个衡量LLM自述不确定性摘要与其真实内部答案分布之间差异的信息论距离，发现现代LLM普遍无法自主反映内部不确定性，但通过采样多个输出并反馈到上下文中可以生成忠实的不确定性摘要。

**[Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)**

:   提出 L-GMVAE（标签条件高斯混合 VAE）和 LAPACE 算法，通过在潜空间中学习每个类别的多个高斯聚类中心，然后从输入潜表征到目标类别中心进行线性插值，生成路径式反事实解释，同时保证有效性、似合性、多样性和对输入扰动的完美鲁棒性。

**[Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](causal_inference/validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)**

:   提出一个标准化的扰动式显著性忠实性验证协议用于 siRNA 效能预测，作为"合成前关卡"检验显著性图是否可信；同时提出 BioPrior 生物信息正则化提升解释忠实性，发现 19/20 折instances 通过验证，但跨数据集迁移暴露两种失败模式。

---

## 🎬 视频生成 { #video_generation }

**[Arbitrary Generative Video Interpolation](video_generation/arbitrary_generative_video_interpolation.md)**

:   ArbInterp 提出了一种支持任意时间戳、任意长度的生成式视频帧插值框架，通过时间戳感知旋转位置编码（TaRoPE）实现精准时间控制，并通过外观-运动解耦的条件注入策略实现长序列的无缝拼接。

**[BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](video_generation/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)**

:   BindWeave 用多模态大语言模型（MLLM）替代传统的浅层融合机制来解析多主体复杂文本指令，生成主体感知的隐状态作为 DiT 的条件信号，结合 CLIP 语义特征和 VAE 细粒度外观特征，实现高保真、主体一致的视频生成。

**[Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)**

:   提出 Frame Guidance，一种无需训练的帧级引导方法，通过 latent slicing（降低 60× 显存）和 Video Latent Optimization（VLO）两个核心组件，在不修改模型的情况下实现关键帧引导、风格化和循环视频等多种可控视频生成任务。

**[Geometry-aware 4D Video Generation for Robot Manipulation](video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)**

:   本文提出几何感知的4D视频生成框架，通过跨视角点图对齐监督训练视频扩散模型，联合预测RGB和点图实现时空一致的多视角RGB-D视频，无需相机位姿输入即可在新视角下生成一致视频并用现成6DoF位姿追踪器恢复机器人末端轨迹。

**[JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](video_generation/javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)**

:   提出 JavisDiT，基于 DiT 架构的音视频联合生成模型，通过层级化时空同步先验估计器（HiST-Sypo）实现细粒度的音视频时空对齐；同时构建了新基准 JavisBench（10K 复杂场景样本）和新评估指标 JavisScore。

**[JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)**

:   提出 JavisDiT++，一个面向联合音视频生成（JAVG）的简洁统一框架，通过模态特定 MoE 提升生成质量、时间对齐 RoPE 实现帧级同步、音视频 DPO 对齐人类偏好，基于 Wan2.1-1.3B 仅用约 1M 公开数据即达到 SOTA。

**[Language-guided Open-world Video Anomaly Detection under Weak Supervision](video_generation/language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)**

:   提出语言引导的开放世界视频异常检测范式LaGoVAD，通过将异常定义建模为随机变量并以自然语言形式输入，结合动态视频合成和对比学习正则化策略，在七个数据集上实现零样本SOTA性能。

**[Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](video_generation/learning_video_generation_for_robotic_manipulation_with_collaborative_trajectory.md)**

:   提出RoboMaster框架，通过协作轨迹（collaborative trajectory）将机器人-物体交互过程分解为前交互、交互中、后交互三阶段，结合外观和形状感知的物体嵌入，实现高质量的机器人操作视频生成。

**[LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)**

:   提出 LoRA-Edit，利用时空 mask 引导 LoRA 微调预训练 I2V 模型，实现可控的首帧引导视频编辑——mask 同时作为编辑区域指令和 LoRA 学习内容的引导信号，支持运动继承和外观控制。

**[Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)**

:   提出 Lumos-1，一个基于 LLM 架构的统一视频生成模型：通过 MM-RoPE（分布式多模态 RoPE）解决视觉时空编码问题，通过 AR-DF（自回归离散扩散强迫）解决帧间损失不均衡问题，仅用 48 GPU 训练即可在 GenEval、VBench-I2V 和 VBench-T2V 上达到竞争力水平。

**[MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](video_generation/mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)**

:   提出 MoSA 框架，将人体视频生成拆分为"结构生成"（3D Transformer 先生成物理合理的运动骨骼）和"外观生成"（DiT 在骨骼引导下合成视频），并设计人体感知动态控制（HADC）模块将稀疏骨骼信号扩展到整个运动区域，配合密集跟踪损失和接触约束，在 FVD、CLIPSIM 等指标上全面超越 HunyuanVideo、Wan 2.1 等 SOTA。

**[MotionStream: Real-Time Video Generation with Interactive Motion Controls](video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)**

:   提出MotionStream——首个运动控制的实时流式视频生成系统：先训练轻量track head的双向运动控制teacher，再通过Self Forcing + DMD蒸馏为因果student，引入注意力沉降（attention sink）+滚动KV缓存（rolling KV cache）实现训练-推理分布完全匹配，单H100 GPU上480P达17FPS/29FPS（+Tiny VAE），支持无限长度恒速生成。

**[PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)**

:   提出 PreciseCache——精确检测并跳过视频生成中真正冗余计算的即插即用加速框架，由 LFCache（步级，基于低频差异 LFD 度量）和 BlockCache（块级，基于输入输出差异度量）组成，在 Wan2.1-14B 等主流模型上实现平均 2.6× 加速且无明显质量损失。

**[QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](video_generation/quantsparse_comprehensively_compressing_video_diffusion_transformer_with_model_q.md)**

:   本文提出 QuantSparse 框架，首次将模型量化（quantization）与注意力稀疏化（attention sparsification）协同整合用于视频扩散 Transformer 压缩，通过多尺度显著注意力蒸馏（MSAD）和二阶稀疏注意力重参数化（SSAR）解决两者朴素结合导致的"放大注意力偏移"问题，在 HunyuanVideo-13B 上以 W4A8 + 15% 注意力密度实现 3.68× 存储压缩和 1.88× 推理加速，同时几乎无损保持生成质量。

**[SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion](video_generation/sigmark_scalable_in-generation_watermark_with_blind_extraction_for_video_diffusi.md)**

:   SIGMark提出首个面向现代视频扩散模型的盲水印框架，通过全局帧级伪随机编码(GF-PRC)实现恒定提取成本的可扩展盲水印，并设计分段组排序(SGO)模块应对因果3D VAE下的时序扰动，在HunyuanVideo和Wan-2.2上实现高bit精度与强鲁棒性。

**[Streaming Autoregressive Video Generation via Diagonal Distillation](video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)**

:   提出Diagonal Distillation（DiagDistill），通过对角线去噪策略（前段多步、后段少步）和流分布匹配损失，实现流式自回归视频生成的277.3倍加速，达到31 FPS实时生成。

**[Target-Aware Video Diffusion Models](video_generation/target-aware_video_diffusion_models.md)**

:   提出 target-aware 视频扩散模型，仅需一张输入图像和目标物体的分割 mask，即可生成演员与指定目标交互的视频；核心创新是引入 [TGT] 特殊 token 并设计选择性交叉注意力损失，使模型关注目标的空间位置，在目标对齐和视频质量上全面超越基线。

**[TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](video_generation/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)**

:   提出 TTOM 框架，在推理时通过优化新增参数将视频生成模型的注意力与 LLM 生成的时空布局对齐，并用参数记忆机制保存历史优化上下文支持复用，在 T2V-CompBench 上相对提升 34%（CogVideoX）和 14%（Wan2.1）。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation](autonomous_driving/adaptive_augmentation-aware_latent_learning_for_robust_lidar_semantic_segmentati.md)**

:   提出 A3Point（Adaptive Augmentation-Aware Latent Learning）框架，通过语义混淆先验(SCP)隐式学习和语义偏移区域(SSR)定位两大核心组件，解耦模型固有的语义混淆与数据增强引入的语义偏移，对不同干扰程度自适应优化，在多个恶劣天气 LiDAR 分割泛化基准上取得 SOTA。

**[SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)**

:   SMART-R1 首次将 R1 风格的强化微调（RFT）引入多智能体交通仿真，提出 Metric-oriented Policy Optimization (MPO) 算法和"SFT-RFT-SFT"迭代训练策略，在 WOSAC 2025 排行榜上以 0.7858 的 Realism Meta 分数取得第一名。

**[Astra: General Interactive World Model with Autoregressive Denoising](autonomous_driving/astra_general_interactive_world_model_with_autoregressive_denoising.md)**

:   提出 Astra，一个通用交互式世界模型，通过自回归去噪框架在预训练视频扩散模型上实现动作条件化的长程视频预测，引入 ACT-Adapter（动作注入）、噪声增强历史记忆（缓解视觉惯性）和 Mixture of Action Experts（统一多异构动作模态），在自动驾驶、机器人操控和场景探索等多场景上实现 SOTA 的保真度和动作跟随能力。

**[BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](autonomous_driving/bridgedrive_diffusion_bridge_policy_for_closed-loop_trajectory_planning_in_auton.md)**

:   BridgeDrive 提出用扩散桥（diffusion bridge）替代截断扩散来实现锚点引导的自动驾驶轨迹规划，保证前向/反向过程的理论对称性，在 Bench2Drive 闭环评估中成功率达到 74.99%（PDM-Lite）和 89.25%（LEAD），分别超越前 SOTA 7.72% 和 2.45%。

**[DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](autonomous_driving/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)**

:   DrivingGen 提出首个面向自动驾驶视频世界模型的综合性基准，包含跨天气/地域/时间/复杂场景的多样化评估数据集和四维度评估指标体系（分布、质量、时序一致性、轨迹对齐），对 14 个 SOTA 模型的评测揭示了通用模型与驾驶专用模型之间的核心权衡。

**[EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)**

:   Apple 使用 Vision Pro 采集了 829 小时的第一人称视频 + 3D 手部关节追踪数据（EgoDex），覆盖 194 种桌面操作任务，并在此数据集上系统评估了模仿学习策略（BC/DDPM/FM + Transformer），为灵巧操作的扩展训练提供了迄今最大规模的数据基础。

**[MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)**

:   提出 MARC 框架，通过"先检索再压缩"策略——用 Visual Memory Retriever (VMR) 选出与查询最相关的视频片段，再用 Compression GRPO (C-GRPO) 将 64 帧教师模型的推理能力蒸馏到仅用 1 帧 token 的学生模型——实现视觉 token 95% 压缩，GPU 显存降低 72%，推理延迟降低 23.9%，性能几乎无损（42.20 vs 42.21）。

**[Multi-Head Low-Rank Attention (MLRA)](autonomous_driving/multi-head_low-rank_attention.md)**

:   提出 Multi-Head Low-Rank Attention (MLRA)，通过将 MLA 的单一 latent head 分解为多个可独立分片的 latent head，并对各分支注意力输出求和，实现原生 4-way 张量并行支持，在保持 SOTA 性能的同时获得 2.8× 的解码加速。

**[NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](autonomous_driving/nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)**

:   提出 NeMo-map——基于神经隐式函数的连续时空动态地图，将空间-时间坐标直接映射为半包裹高斯混合模型（SWGMM）参数，消除传统方法的空间离散化和时间分段限制，在真实行人追踪数据上实现更低 NLL 和更平滑的速度分布。

**[ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)**

:   ResWorld 提出时序残差世界模型（TR-World），通过计算 BEV 场景表征的时序残差来提取动态物体信息（无需检测/跟踪），避免对静态区域的冗余建模，结合未来引导轨迹优化（FGTR）模块利用预测的未来 BEV 特征修正规划轨迹，在 nuScenes 和 NAVSIM 上达到 SOTA 规划性能。

**[SEAL: Segment Any Events with Language](autonomous_driving/segment_any_events_with_language.md)**

:   首次提出开放词汇事件实例分割（OV-EIS）任务，设计 SEAL 框架通过多模态层次语义引导（MHSG）和轻量多模态融合网络，在仅使用事件-图像对（无密集标注）的情况下，实现事件流的多粒度（实例级+部件级）语义分割，大幅领先所有基线方法且推理速度最快。

**[SiMO: Single-Modality-Operable Multimodal Collaborative Perception](autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)**

:   提出 SiMO 框架，通过 LAMMA 融合模块和 PAFR 训练策略，首次在多智能体协同感知中实现任意模态缺失（特别是 LiDAR 失效仅有相机可用时）下仍可正常工作的多模态感知系统，类似并联电路——只要有一条通路就能工作。

**[SPACeR: Self-Play Anchoring with Centralized Reference Models](autonomous_driving/spacer_self-play_anchoring_with_centralized_reference_models.md)**

:   SPACeR 提出"类人自博弈"框架，用预训练的 tokenized 自回归运动模型作为集中式参考策略，通过对数似然奖励和 KL 散度约束引导去中心化自博弈 RL 策略向人类驾驶分布对齐，在 WOSAC 上超越纯自博弈方法，同时推理速度比模仿学习快 10 倍、参数量小 50 倍。

**[ST4VLA: Spatially Guided Training for Vision-Language-Action Models](autonomous_driving/st4vla_spatially_guided_training_for_vision-language-action_models.md)**

:   提出 ST4VLA，通过两阶段空间引导训练（spatial grounding pre-training + spatially guided action post-training），将 VLM 的空间先验显式注入 VLA 策略学习，在 SimplerEnv 上将 Google Robot 成功率从 66.1% 提升至 84.6%，WidowX 从 54.7% 提升至 73.2%，达到 SOTA。

**[Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](autonomous_driving/steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)**

:   SAGE 将自动驾驶对抗场景生成重构为多目标偏好对齐问题，通过训练两个偏好专家模型并在推理时通过权重插值实现对抗性与真实性之间的连续可控权衡，无需重新训练即可生成从温和到激进的全谱场景，显著提升闭环训练效果。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Activation Steering for Masked Diffusion Language Models](image_restoration/activation_steering_for_masked_diffusion_language_models.md)**

:   首次将激活引导（activation steering）应用于 Masked Diffusion 语言模型（MDLM），发现 MDLM 的拒绝行为也受单一低维方向控制，通过在去噪过程中全局投影可完全绕过安全对齐，且与自回归模型不同，有效方向可从指令前的 token 中提取——反映了扩散模型的非因果并行处理特性。

**[AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](image_restoration/adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)**

:   通过统计分析扩散语言模型（dLLM）去噪过程中 token 置信度的动态变化，发现"波动带"（Volatility Band）区域编码了文本的局部语义结构，进而提出 AdaBlock-dLLM——一个无训练、即插即用的自适应块大小调度器，让半自回归解码的块边界与语义步骤自然对齐，在相同吞吐量下最高提升 5.3% 准确率。

**[Are Deep Speech Denoising Models Robust to Adversarial Noise?](image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)**

:   首次系统性评估 4 款 SOTA 深度语音去噪（DNS）模型在对抗噪声下的鲁棒性：通过心理声学约束的 PGD 攻击生成人耳不可感知的对抗噪声，可令 Demucs、Full-SubNet+、FRCRN 和 MP-SENet 输出完全不可理解的 gibberish，实验覆盖多种声学条件和人类评估，同时揭示了目标攻击、通用扰动和跨模型迁移的局限性。

**[Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](image_restoration/beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)**

:   LSP 调度器通过在每个去噪步骤中原子性地提交最长连续稳定前缀（而非分散接受离散 token），将 DLM 推理加速 3.4 倍，同时保持或略微提升输出质量。

**[DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](image_restoration/diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)**

:   提出 DiffusionBlocks，将残差网络的逐层更新解释为连续时间扩散过程的离散化步骤，从而将网络切分为可完全独立训练的 block，在保持端到端训练性能的同时按 block 数 B 倍减少训练显存。

**[Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss](image_restoration/generalizing_linear_autoencoder_recommenders_with_decoupled_expec.md)**

:   将 EDLAE 推荐模型的目标函数推广为解耦期望二次损失（DEQL），在超参数 $b>0$ 的更广范围内推导出闭式解，并通过 Miller 矩阵逆定理将计算复杂度从 $O(n^4)$ 降至 $O(n^3)$，在多个基准数据集上超越 EDLAE 和深度学习模型。

**[Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](image_restoration/horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)**

:   提出 Horizon Imagination (HI)，通过在去噪中途采样动作并行处理多个未来帧，将扩散世界模型的 on-policy 想象计算量降至每帧不到一次完整去噪，同时保持控制性能。

**[InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions](image_restoration/interacthuman_multi-concept_human_animation_with_layout-aligned_audio_conditions.md)**

:   提出 InterActHuman，通过自动推断时空布局的掩码预测器和迭代掩码引导策略，实现多人/人物交互场景下的音频驱动视频生成，支持每个角色独立的语音驱动口型同步和身体动作。

**[Mechanism of Task-oriented Information Removal in In-context Learning](image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)**

:   从"信息移除"的新视角解释 In-context Learning（ICL）的内部机制：发现 LM 在零样本时将查询编码为包含所有可能任务信息的"非选择性表征"（导致随机输出），而 few-shot ICL 的核心作用是模拟一种"任务导向的信息移除"过程——通过识别出的"Denoising Heads"（去噪注意力头）从纠缠的表征中选择性移除冗余任务信息，引导模型聚焦目标任务。消融实验证实阻断去噪头后 ICL 准确率显著下降。

**[ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)**

:   提出 ProtoTS，通过层级原型学习实现可解释时间序列预测：少量粗粒度原型提供全局模式概览，逐级细分捕捉局部变化，结合多通道嵌入与瓶颈融合处理异质外生变量。在 LOF 数据集上 MSE 降低 48.3%，MAE 降低 20.9%，且支持专家编辑原型以进一步提升性能。

**[Sharpness-Aware Machine Unlearning](image_restoration/sharpness-aware_machine_unlearning.md)**

:   本文从信号-噪声分解的视角系统分析了 SAM 在机器遗忘场景下的理论特性，发现 SAM 在遗忘集上会"放弃"去噪能力但在保留集上仍维持优势，进而提出 Sharp MinMax 算法——将模型拆成两部分分别做锐度最小化（保留）和锐度最大化（遗忘），达到SOTA遗忘效果。

**[Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)**

:   首次系统比较扩散语言模型（dLLM）和自回归模型（AR LLM）的层间表征结构，发现原生 dLLM 具有更强的层级抽象和早期层冗余性，据此提出静态、任务无关的推理时层跳过策略，在 LLaDA 上跳过 6 层（18.75% FLOPs 削减）仍保持 90%+ 性能。

**[Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](image_restoration/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)**

:   揭示了掩码扩散语言模型（MDLM）中的"启动漏洞"（priming vulnerability）——在去噪中间步骤注入肯定性 token 可绕过安全防线，并提出 Recovery Alignment（RA）方法训练模型从被污染的中间状态恢复到安全响应。

**[Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution](image_restoration/trust_but_verify_adaptive_conditioning_for_reference-based_diffusion_super-resol.md)**

:   提出 Ada-RefSR，一个基于"Trust but Verify"原则的单步参考引导扩散超分辨率框架，通过自适应隐式相关性门控（AICG）机制在利用可靠参考信息的同时抑制错误融合，仅增加 0.13% 计算开销。

**[wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](image_restoration/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)**

:   提出 wd1，一种无需策略比率（ratio-free）的加权对数似然策略优化方法用于扩散语言模型（dLLM）的 RL 微调，通过正样本加权和负样本惩罚避免了 GRPO 中策略比率估计的偏差和高方差问题，在 LLaDA-8B 上实现了 Sudoku +59%、GSM8K 84.5% 的 SOTA 性能。

---

## 🔄 自监督 { #self_supervised }

**[Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](self_supervised/difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)**

:   通过相似度图模型理论分析严格证明"困难样本"（跨类高相似度样本对）会损害无监督对比学习性能——困难样本使泛化误差界严格恶化，提出删除困难样本、调节 margin 和温度缩放三种理论指导的缓解策略，在 TinyImageNet 上带来高达 10.42% 的线性探测准确率提升。这一发现是反直觉的：深度学习中通常"更多数据更好"，但对比学习中精心移除困难样本反而有益。

**[Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions](self_supervised/enhancing_molecular_property_predictions_by_learning_from_bond_modelling_and_int.md)**

:   提出 DeMol 双图增强多尺度交互框架，通过并行的原子中心图和键中心图通道以及 Double-Helix Blocks 显式建模原子-原子、原子-键、键-键三类交互，在 PCQM4Mv2、OC20、QM9 等基准上取得 SOTA。

**[Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](self_supervised/fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)**

:   受果蝇嗅觉回路启发，提出 Fly-CL 框架，通过稀疏随机投影+top-k操作+流式岭分类三阶段渐进去相关，在预训练模型持续学习中大幅降低训练时间的同时达到SOTA水平。

**[Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](self_supervised/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)**

:   提出 GradFix 方法，利用目标预训练模型上极少量样本计算的梯度符号构建二值掩码，逐坐标过滤源模型的任务向量，仅保留与目标损失景观下降方向一致的分量，在无需任何微调的情况下实现跨预训练模型的任务知识迁移，理论上提供严格的一阶下降保证，在视觉与语言基准上均大幅超越朴素迁移和少样本微调。

**[InfoNCE Induces Gaussian Distribution](self_supervised/infonce_induces_gaussian_distribution.md)**

:   从理论上证明 InfoNCE 损失函数在两种互补机制下会诱导表征趋向高斯分布：经验理想化路线（对齐+球面均匀性→高斯）和正则化路线（消失正则项→各向同性高斯），并在合成数据和 CIFAR-10 上验证。

**[Maximizing Incremental Information Entropy for Contrastive Learning](self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)**

:   提出IE-CL（Incremental-Entropy Contrastive Learning）框架，通过显式优化增强视图间的熵增益（而非仅最大化互信息），将编码器视为信息瓶颈并联合优化可学习变换（生成熵）与编码器正则化器（保留熵），在小batch设置下一致提升CIFAR-10/100、STL-10和ImageNet上的对比学习性能，且核心模块可即插即用集成到现有框架。

**[No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](self_supervised/no_other_representation_component_is_needed_diffusion_transformers_can_provide_r.md)**

:   提出 Self-Representation Alignment (SRA)，发现扩散 Transformer 内部表征沿"层数增加 + 噪声降低"两个维度呈现从差到好的判别质量梯度，据此将学生网络早层高噪声表征对齐到 EMA 教师晚层低噪声表征，**完全不需要任何外部表征组件（DINOv2/CLIP/MAE）**，即可在 DiT 和 SiT 上大幅加速收敛并提升生成质量（SiT-XL/2 在 800 epoch 达到 FID 1.58，可比依赖 DINOv2 的 REPA）。

**[PonderLM: Pretraining Language Models to Ponder in Continuous Space](self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)**

:   提出 PonderLM，在预训练阶段引入"沉思"机制——将预测概率分布加权求和为连续嵌入后反复前向传播，无需标注数据或强化学习，使 2.8B 模型在 9 个下游任务上超越 6.9B 模型。

**[Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)**

:   提出 Regularized Latent Dynamics Prediction (RLDP)，通过在自监督的潜空间下一状态预测目标上添加简单的正交正则化来维持特征多样性，在零样本 RL 中匹配甚至超越复杂的 SOTA 表示学习方法，特别是在低覆盖率场景下优势显著。

**[SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](self_supervised/snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)**

:   SNAP-UQ 提出一种面向 TinyML 场景的单次前向传播不确定性估计方法：在骨干网络的选定层附加微型 int8 预测头，用自监督方式预测下一层的激活统计量，将实际激活与预测之间的偏差（"surprisal"）聚合为不确定性分数，无需额外前向传播、时序缓冲或集成，仅增加几十 KB 闪存即可在微控制器上实现可靠的分布偏移检测和故障检测。

**[Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](self_supervised/spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)**

:   揭示RLHF/DPO等后训练会损害模型的上下文可操控性(in-context steerability)、输出覆盖率和分布对齐，提出Spectrum Suite评测框架和Spectrum Tuning方法，首次在后训练阶段改善分布对齐能力。

**[Temporal Slowness in Central Vision Drives Semantic Object Learning](self_supervised/temporal_slowness_in_central_vision_drives_semantic_object_learning.md)**

:   通过模拟人类中央视觉（注视点裁剪）和时间慢性原则（时间对比学习），在 Ego4D 数据上训练 SSL 模型，发现两者组合能有效提升语义对象表征——中央视觉强化前景提取，时间慢性在注视凝视期间蒸馏语义信息。

**[Weak-SIGReg: Covariance Regularization for Stable Deep Learning](self_supervised/weak-sigreg_covariance_regularization_for_stable_deep_learning.md)**

:   将 LeJEPA 的 SIGReg 正则化从自监督学习迁移到监督学习，并提出计算高效的 Weak-SIGReg 变体——只约束协方差矩阵趋向单位矩阵（而非全部矩），用随机投影将内存从 $O(C^2)$ 降至 $O(CK)$，在 ViT 无 BN/残差连接时将 CIFAR-100 准确率从 20.73%（坍缩）恢复到 72.02%，且匹配或超越专家精调的基线。

**[Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)**

:   诊断出原型自监督学习中部分原型坍缩的根因是编码器与原型的联合优化导致的快捷学习，提出全解耦训练策略——用在线 GMM 独立估计原型——彻底消除坍缩并提升下游性能。

---

## 🔒 LLM安全 { #llm_safety }

**[Enhancing Hallucination Detection through Noise Injection](llm_safety/enhancing_hallucination_detection_through_noise_injection.md)**

:   在 LLM 中间层的 MLP 激活中注入均匀噪声来近似贝叶斯后验，捕获认知不确定性（epistemic uncertainty），与采样温度捕获的偶然不确定性（aleatoric uncertainty）互补，将 GSM8K 上的幻觉检测 AUROC 从 71.56 提升到 76.14。

**[Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](llm_safety/gaussian_certified_unlearning.md)**

:   提出 $(\phi,\varepsilon)$-Gaussian certifiability——基于假设检验 trade-off 函数的高维机器遗忘隐私框架，严格证明在高维比例体系 ($p \sim n$) 下单步 Newton 更新 + 校准高斯噪声即可同时满足隐私 (GPAR) 和精度 (GED→0) 要求，推翻了 Zou et al. (2025) "至少需两步 Newton" 的结论，并从理论上揭示旧 $\varepsilon$-certifiability 与噪声添加机制不兼容的根本原因。

**[How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](llm_safety/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)**

:   系统分析了 LLM 在扑克中的三大推理缺陷（启发式推理、事实误解、知行差距），提出 ToolPoker 框架——首个面向不完全信息博弈的工具集成 LLM 推理系统，通过外部 CFR solver 提供博弈论最优的行动指导，使 7B 模型在 Limit Hold'em 中逼近 Nash 均衡。

**[LH-Deception: Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions](llm_safety/lh-deception_simulating_and_understanding_llm_deceptive_behaviors_in_long-horizo.md)**

:   提出首个面向长时域交互的 LLM 欺骗行为仿真框架 LH-Deception，采用执行者-监督者-审计者三角色多智能体架构，结合社会科学理论驱动的概率事件系统，在 11 个前沿模型上系统量化了欺骗频率、严重性、类型分布及其对信任关系的侵蚀效应，揭示了静态单轮评估完全无法捕捉的"欺骗链"涌现现象。

**[Lifelong Learning with Behavior Consolidation for Vehicle Routing](llm_safety/lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)**

:   提出 LLR-BC 框架，在神经 VRP 求解器的终身学习场景中，通过决策步骤级经验缓冲、置信度感知加权（CaEW）和反向 KL 散度行为巩固（DsBC），在分布与规模同时变化的任务序列上将平均性能差距（AP）降低一个数量级，同时保持学新任务的可塑性并提升零样本泛化。

**[Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](llm_safety/perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)**

:   提出PIL方法，仅使用无偏置线性分类器作为代理模型生成不可学习扰动，通过诱导深度模型线性化来阻止其学习语义特征，比现有方法快100倍以上（CIFAR-10上不到1分钟GPU时间）。

**[Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](llm_safety/redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)**

:   本文提出损坏数据遗忘任务的二维分类框架（发现率 × 统计规律性），揭示了现有遗忘方法各自仅在特定区域有效的局限，并提出 REM（重定向以擦除记忆）方法，通过将损坏数据重定向到新增的专用网络容量再丢弃，首次在整个二维任务空间中实现强劲且一致的遗忘性能。

**[Self-Destructive Language Model](llm_safety/self-destructive_language_model.md)**

:   提出 Seam，通过耦合良性和有害数据的优化轨迹（使梯度方向相反），将 LLM 转变为"自毁模型"——在有害微调时自动触发灾难性性能崩溃，创造攻击者的两难困境：低强度攻击无效，高强度攻击导致模型报废。

**[Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](llm_safety/understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)**

:   首次从对抗鲁棒性角度分析 Differential Attention（DA）机制，揭示其减法结构在抑制噪声的同时会通过负梯度对齐放大对抗扰动敏感度，发现"脆弱性原理"——DA 在干净样本上提升判别力但在对抗攻击下更脆弱，且存在深度依赖的鲁棒性交叉效应。

**[Unlearning Evaluation through Subset Statistical Independence](llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)**

:   提出 Split-half Dependence Evaluation (SDE)，利用 HSIC 统计独立性检验在子集级别评估机器遗忘效果，无需重训模型或辅助分类器。

**[VeriTrail: Closed-Domain Hallucination Detection with Traceability](llm_safety/veritrail_closed-domain_hallucination_detection_with_traceability.md)**

:   提出 VeriTrail，首个面向多步生成（MGS）过程的闭域幻觉检测方法，通过将生成过程建模为 DAG 并沿图逐层验证 claim，实现了幻觉检测+溯源（provenance）+错误定位（error localization）的完整可追溯性，在两个新数据集上显著优于所有基线。

---

## ✂️ 语义分割 { #segmentation }

**[AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)**

:   提出对齐感知遮蔽学习(AML)策略，通过量化视觉-语言 patch 级对齐度并过滤低对齐像素，让 RIS 模型在训练时聚焦可靠区域，无需架构改动即在 RefCOCO 全部 8 个 split 上达到 SOTA。

**[ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer](segmentation/byteflow_language_modeling_through_adaptive_byte_compression_without_a_tokenizer.md)**

:   提出 ByteFlow Net，一种无需分词器的分层字节级语言模型，利用信息论中的编码率(coding rate)自适应地将原始字节流压缩为语义单元，在预训练损失和下游任务上超越 BPE 基线和已有字节级架构。

**[Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)**

:   发现 SAM2 存在类似生物视觉的稀疏感知模式（解码器聚焦前景但编码器广泛计算、记忆帧中仅少量 token 有效且显著性时间一致），据此提出 Efficient-SAM2，通过对象感知的稀疏窗口路由（SWR）和稀疏记忆检索（SMR）消除冗余计算，在 SAM2.1-L 上实现 1.68× 端到端加速且仅损失 1% 精度。

**[Locality-Attending Vision Transformer](segmentation/locality-attending_vision_transformer.md)**

:   提出 LocAt 模块化插件（GAug + PRR），通过可学习高斯核偏置注意力向局部邻域聚焦并精炼 patch 表示，在不修改训练目标的前提下使 ViT 在 ADE20K 分割上提升超 6%，同时分类精度不降反升。

**[RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)**

:   提出 RegionReasoner，一个基于强化学习的多轮视觉推理框架，通过引用标注奖励和全局-局部一致性奖励，使推理轨迹必须显式引用参考区域坐标并保持语义连贯，在新构建的 RegionDial-Bench 上显著提升多轮定位和分割精度。

**[Revisiting [CLS] and Patch Token Interaction in Vision Transformers](segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)**

:   深入分析Vision Transformer中[CLS]全局token和patch局部token之间的交互摩擦，发现归一化层隐式地区分了两类token，提出在归一化层和早期QKV投影中引入专门化处理路径，仅增加8%参数即实现分割性能提升超2 mIoU，同时保持分类精度。

**[Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers](segmentation/thicker_and_quicker_a_jumbo_token_for_fast_plain_vision_transformers.md)**

:   本文提出 Jumbo 方法：将 ViT 的 CLS token 扩展为 $J$ 倍宽度，在注意力前拆分为 $J$ 个与 patch 等宽的 token 参与自注意力，注意力后重新拼接并经过专用的宽 FFN 处理——以极低的计算开销显著增加全局建模容量，使 plain ViT 在高速推理场景下超越专用高效架构（EfficientViT、SHViT、MobileNetV4），同时保留 ViT 的所有架构优势。

**[TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](segmentation/trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)**

:   发现文本到图像扩散模型的自注意力在去噪过程中存在一个"实例涌现点"（IEP），在该时刻自注意力在物体边界呈现剧烈散度变化。TRACE通过IEP定位+ABDiv边缘提取+单步蒸馏，以81×推理加速生成高质量实例边缘，无需任何实例标注即可将无监督实例分割提升+5.1 AP，tag监督全景分割超越点监督方法+1.7 PQ。

**[Universal Multi-Domain Translation via Diffusion Routers](segmentation/universal_multi-domain_translation_via_diffusion_routers.md)**

:   提出 Diffusion Router (DR)，用单个噪声预测网络通过 source/target 域标签条件化实现所有跨域映射，支持通过中心域的间接翻译和基于变分上界目标 + Tweedie 精化的直接非中心域翻译，在三个大规模 UMDT 基准上达到 SOTA。

**[VINCIE: Unlocking In-context Image Editing from Video](segmentation/vincie_unlocking_in-context_image_editing_from_video.md)**

:   提出VINCIE框架，首次证明in-context图像编辑模型可以完全从原生视频数据中学习，通过将视频标注为交错多模态序列并设计三个代理任务（NIP/CSP/NSP），在多轮编辑基准上达到SOTA，5轮编辑成功率从基线<2%提升至25%。

**[VIRTUE: Visual-Interactive Text-Image Universal Embedder](segmentation/virtue_visual-interactive_text-image_universal_embedder.md)**

:   提出 VIRTUE，将分割模型 SAM2 与 VLM 结合构建视觉交互式通用嵌入器，支持用户通过点/框/掩码指定兴趣区域产生实体级+全局级联合嵌入，并构建百万级 SCaR 基准评估视觉交互检索能力，在 36 个 MMEB 任务（+3.1%-8.5%）和 5 个 SCaR 任务（+15.2%-20.3%）上均达到 SOTA。

---

## 🎁 推荐系统 { #recommender }

**[C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)**

:   提出 C2AL（Cohort-Contrastive Auxiliary Learning），通过数据驱动地发现分布差异最大的用户群体对，构建对比性辅助二分类任务正则化共享编码器，使 FM 注意力权重从稀疏变为稠密，缓解大规模推荐系统中少数群体的表征偏差，在 Meta 6 个生产模型（数十亿数据点）上验证有效。

**[From Evaluation to Defense: Advancing Safety in Video Large Language Models](recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)**

:   构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

**[GoalRank: Group-Relative Optimization for a Large Ranking Model](recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)**

:   理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

**[In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](recommender/in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)**

:   通过对来自6家提供商的12个LLM在新闻、学术、电商三大领域的大规模控制实验，揭示了LLM存在系统性的**隐式信息源偏好**（latent source preferences）——当内容语义完全相同时，仅更换来源标签就能显著改变模型的信息选择行为，且这种偏好无法通过提示工程消除。

**[ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)**

:   提出ProPerSim模拟框架，构建基于大五人格的32种用户persona在Smallville家庭环境中的日常行为模拟，AI助手通过每2.5分钟的主动推荐决策和DPO偏好学习，在14天模拟中将用户满意度从2.2/4提升至3.3/4，首次验证了主动性+个性化统一的可行性。

**[RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](recommender/rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese.md)**

:   提出 RAE（Regularized Auto-Encoder），通过线性自编码器 + Frobenius 范数正则化实现降维，理论证明正则化系数 $\lambda$ 通过 Rayleigh 商性质约束编码器矩阵的条件数 $\kappa(W)$，从而保证范数失真率有界、k-NN 结构被保持。在 4 个数据集上一致优于 PCA/UMAP/MDS/ISOMAP，余弦距离下比 PCA 至少高 12%，且训练仅需 8 秒、推理毫秒级。

**[Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](recommender/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)**

:   理论证明 CE 损失在推荐系统 KD 中最大化 NDCG 下界需满足"闭合性假设"（子集需包含学生 top 项目），但实际目标是蒸馏教师 top 项目的排序——两者冲突导致 vanilla CE 表现差。据此提出 RCE-KD：将教师 top-K 项目按是否在学生 top-K 中分两组，分别用精确 CE 和采样近似闭合性 CE，自适应融合权重随训练动态调整。

**[Search Arena: Analyzing Search-Augmented LLMs](recommender/search_arena_analyzing_search-augmented_llms.md)**

:   构建 Search Arena——首个大规模搜索增强 LLM 人类偏好数据集（24069 对话 + 12652 偏好投票，71 种语言），发现用户偏好受引用数量影响（即使引用不支持声明），社区驱动平台比 Wikipedia 更受偏好，搜索增强不降低通用聊天性能但通用 LLM 在搜索场景显著退化。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](knowledge_editing/bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)**

:   通过在合成关系知识图谱上从头训练 Transformer，发现适当正则化会使模型隐层涌现出双线性关系结构（bilinear relational structure），该结构不仅能克服逆向诅咒（reversal curse），还能实现编辑单个事实后逻辑一致地传播到相关事实。

**[EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](knowledge_editing/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)**

:   揭示大规模模型编辑失败的根本原因是 key embedding 与 residual embedding 之间的结构不一致（embedding misalignment），提出 EAMET 通过渐进式保存已优化的残差 embedding 并用 KL 散度 + MSE 双损失将其邻域结构对齐到 key embedding 空间，在 6 个 LLM、3 个数据集上同时编辑 10k 事实时平均超越 MEMIT 14%（CounterFact）和 8%（ZsRE），且在长前缀和同主语多事实两大鲁棒性场景下表现稳健。

**[Energy-Regularized Sequential Model Editing on Hyperspheres](knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)**

:   从超球面均匀性（Hyperspherical Energy）视角理解序列模型编辑中的性能退化，提出 SPHERE 方法：通过将编辑扰动投影到预训练权重主超球方向的正交补空间，实现稳定的大规模序列编辑，在 LLaMA3-8B 上平均超越最强基线 16.41%。

**[Fine-tuning Done Right in Model Editing](knowledge_editing/fine-tuning_done_right_in_model_editing.md)**

:   揭示模型编辑中 fine-tuning 被低估的根因是错误的训练 pipeline（深度优先逐样本优化），修正为标准的广度优先 mini-batch 训练后，配合局部化参数调优形成 LocFT-BF，首次支持 10 万次连续编辑和 72B 模型规模。

**[GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](knowledge_editing/got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)**

:   通过零空间约束的在线模型编辑，将 VGGT 提供的 3D 几何信息融入 2D 通用目标跟踪器中，在保持语义判别力的同时增强几何感知能力，在遮挡和背景杂乱场景中显著提升跟踪性能。

**[PICS: Pairwise Image Compositing with Spatial Interactions](knowledge_editing/pics_pairwise_image_compositing_with_spatial_interactions.md)**

:   提出 PICS——一种并行成对图像合成方法，通过 Interaction Transformer 中的掩码引导 MoE 和自适应 α-blending 策略，在单次推理中同时合成两个对象并显式建模遮挡、接触等空间交互关系，全面超越现有序列合成方法。

**[Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs](knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_data_in_llms.md)**

:   提出"记忆-再泛化"（memorize-then-generalize）框架，通过先用无语义合成 token 死记硬背事实关联、再用少量语义提示微调的两阶段策略，揭示 LLM 能从死记硬背数据中泛化，且记忆越深泛化越好，同时指出该机制可被恶意利用的安全隐患。

**[When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)**

:   提出 EVOKE 基准测试，系统评估大型多模态模型 (LMM) 对演化知识的注入能力，揭示两大挑战（现有方法表现差、微调导致灾难性遗忘），并提出知识增强和持续学习两条应对路径。

---

## 🧮 科学计算 { #scientific_computing }

**[Astral: Training Physics-Informed Neural Networks with Error Majorants](scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)**

:   提出 Astral 损失函数（基于函数型后验误差上界/error majorant），替代传统 PiNN 中的残差损失来训练物理信息神经网络，实现训练过程中可靠的误差估计，并在扩散方程、Maxwell 方程等多种 PDE 上取得了更好或相当的精度。

**[Deep Learning for Subspace Regression](scientific_computing/deep_learning_for_subspace_regression.md)**

:   将缩减阶建模（ROM）中的子空间预测问题形式化为 Grassmann 流形上的回归任务，提出专用损失函数与子空间嵌入（subspace embedding）技术——通过预测比目标更大维度的子空间来降低映射复杂度——在特征值问题、参数化 PDE 和迭代法加速等场景中均取得显著效果。

**[DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)**

:   提出 DRIFT-Net 双分支神经算子，通过受控低频混合（谱分支）和局部细节保真（图像分支）的带宽融合（radial gating），解决窗口注意力中全局谱耦合不足导致的自回归漂移问题，在 Navier-Stokes 基准上误差降低 7%-54%。

**[Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](scientific_computing/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)**

:   在硬约束递归物理信息架构（HRPINN）中系统评估vanilla KAN替代MLP作为残差分支的效果——通过3项互补研究×100随机种子发现KAN在单变量可分残差（Duffing的 $-0.3x^3$）上的表现具有竞争力，但在乘法耦合残差（Van der Pol的 $(1-x^2)v$）上系统性失败且超参数极度脆弱，标准MLP在几乎所有配置下稳定性远优。

**[HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](scientific_computing/hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)**

:   提出 HyperKKL，用超网络（hypernetwork）编码外源输入信号并即时生成 KKL 观测器的变换映射参数，使非自治非线性系统的状态估计无需重新训练或在线梯度更新，在 Duffing、Van der Pol、Lorenz、Rössler 四个经典非线性系统上验证了方法的有效性和局限性。

**[Learning-guided Kansa Collocation for Forward and Inverse PDE Problems](scientific_computing/learning-guided_kansa_collocation_for_forward_and_inverse_pde_problems.md)**

:   将基于径向基函数(RBF)的无网格Kansa方法从单变量线性PDE扩展到耦合多变量和非线性PDE场景，结合自调参技术和多种时间步进方案，并系统对比了与PINN、FNO等神经PDE求解器在正问题和反问题上的表现。

**[One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](scientific_computing/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)**

:   论证神经 PDE 求解器在边界条件变化时学到的不是单一的解算子，而是由边界条件索引的算子族，并从学习理论角度形式化了 ERM 下边界分布偏移导致的不可辨识性问题。

**[Policy Myopia as a Mechanism of Gradual Disempowerment in Post-AGI Governance](scientific_computing/policy_myopia_as_a_mechanism_of_gradual_disempowerment_in_post-agi_governance_ci.md)**

:   论证政策短视（policy myopia）不是注意力分配问题，而是一个制度性机制——通过显著性捕获、能力级联和价值锁死三个耦合的正反馈循环，在后AGI时代系统性地、不可逆地剥夺人类的治理参与能力，而标准的缓解措施只能延缓但无法阻止这一过程。

---

## 📡 信号/通信 { #signal_comm }

**[Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds](signal_comm/deterministic_bounds_and_random_estimates_of_metric_tensors_on_neuromanifolds.md)**

:   本文通过分析低维概率分布核空间的Fisher信息矩阵(FIM)谱性质，为神经网络参数空间(神经流形)上的度量张量建立了确定性上下界，并基于Hutchinson迹估计器引入了一族有界方差的无偏随机估计方法，仅需单次反向传播即可高效计算。

**[FASA: Frequency-Aware Sparse Attention](signal_comm/fasa_frequency-aware_sparse_attention.md)**

:   本文发现RoPE中频率块（FC）级别的功能稀疏性——少数"主导FC"可有效预测token重要性，据此提出FASA框架，通过主导FC预估token重要性+聚焦注意力计算两阶段实现无训练的KV缓存压缩，在LongBench上仅保留256个token接近100%全KV性能，AIME24上用18.9%缓存实现2.56×加速。

**[Group Representational Position Encoding (GRAPE)](signal_comm/group_representational_position_encoding.md)**

:   提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

**[Learning Molecular Chirality via Chiral Determinant Kernels](signal_comm/learning_molecular_chirality_via_chiral_determinant_kernels.md)**

:   提出手性行列式核(ChiDeK)来编码 SE(3) 不变的手性矩阵，首次在 GNN 框架中统一处理中心手性和轴向手性，结合交叉注意力传播立体化学信息，在新构建的轴向手性基准上准确率提升 >7%。

**[Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](signal_comm/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)**

:   提出Multi-Agent System Search（MASS）框架，通过交错优化提示词和拓扑结构的三阶段策略（局部提示优化→拓扑搜索→全局提示优化），自动发现高性能的多智能体系统设计。

**[Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)**

:   通过大规模实证研究量化23个VQA基准中的模态内依赖和模态间依赖，揭示多数基准存在严重的单模态捷径，且消除文本偏差往往引入图像偏差，为多模态基准设计提供定量评估框架。

**[Robust Preference Alignment via Directional Neighborhood Consensus](signal_comm/robust_preference_alignment_via_directional_neighborhood_consensus.md)**

:   提出Robust Preference Selection (RPS)，一种无需重训练的推理时偏好对齐增强方法，通过从目标偏好的局部邻域采样多个候选方向并生成响应、再根据原始偏好选择最优响应，在OOD偏好上相比基线达到最高69%的胜率。

**[Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](signal_comm/spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)**

:   提出Spectrum Tuning后训练方法，通过在90+任务的分布拟合数据集上训练，改善语言模型的上下文可操控性、输出空间覆盖度和分布对齐能力，揭示当前指令调优会损害模型的上下文可操控性。

---

## 👥 社会计算 { #social_computing }

**[Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)**

:   提出将 Tsallis 熵（SE 的广义形式）引入 VLM 的 Test-Time Adaptation，并进一步发展为自适应去偏 Tsallis 熵（ADTE），为每个类别定制去偏参数 $q^l$，在不引入分布特定超参数的情况下比 Shannon 熵选择更可靠的高置信视图，在 ImageNet 及其 5 个变体和 10 个跨域 benchmark 上均超越 SOTA。

**[BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)**

:   本文构建了 BiasFreeBench 基准，首次在统一框架下系统比较 8 种主流去偏方法（4 种 prompting + 4 种 training），聚焦于 LLM 响应层面的偏差评估，并提出了 Bias-Free Score 指标，发现 prompting 方法（尤其是 CoT）整体优于 training 方法，而 DPO 在跨偏差类型泛化上表现突出。

**[Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](social_computing/functional_embeddings_enable_aggregation_of_multi-area_seeg_recordings_over_subj.md)**

:   提出 FunctionalMap 框架，通过对比学习从颅内局部场电位（LFP）中学习被试无关的功能嵌入作为"功能坐标系"，替代不可靠的 MNI 解剖坐标，结合 Transformer 实现跨被试、跨电极的神经数据聚合和信号重建，在 20 名被试的多脑区 SEEG 数据集上验证有效。

**[GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](social_computing/gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)**

:   提出GRADIEND——一个基于梯度的编码器-解码器架构，通过单个瓶颈神经元从模型梯度中学习可解释的单语义特征（以性别为例），不仅可以识别哪些权重编码了特定特征，还能通过解码器直接修改模型权重来消除偏见，与INLP结合在所有基线模型上达到SOTA去偏效果。

**[Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)**

:   对9个SOTA语音对话系统开展首次语音图灵测试（2968次人类判断），发现所有系统均未通过（成功率7%-31%），瓶颈不在语义理解而在副语言特征、情感表达和对话人格，并构建了18维细粒度评估框架和可解释AI评审模型。

**[Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)**

:   提出 RAVEN 审计框架，通过结合模型内语义熵和跨模型分歧来检测 LLM 中的概念条件语义分歧——一种类似宣传的行为模式，即高层概念线索（意识形态、公众人物）触发异常一致的立场响应。

**[Scalable Multi-Task Low-Rank Model Adaptation](social_computing/scalable_multi-task_low-rank_model_adaptation.md)**

:   系统分析多任务 LoRA 在任务数量增大时崩溃的根因（均匀正则化破坏共享知识 + 组件级 LoRA 放大梯度冲突），提出 mtLoRA：谱感知正则化 + 块级适配 + 细粒度路由，在 15-25 个任务上平均超越 SOTA 2.3%，同时减少 47% 参数和 24% 训练时间。

**[Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](social_computing/stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)**

:   提出 SupervisorAgent，一个轻量级的实时自适应监督框架，通过无 LLM 的自适应过滤器在关键交互节点主动干预（纠错、指导、观察净化），在 GAIA 基准上将 Smolagent 的 token 消耗降低 29.68% 而不损失成功率。

---

## 🛰️ 遥感 { #remote_sensing }

**[AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](remote_sensing/autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)**

:   提出 AutoFly，一个面向无人机野外自主导航的端到端 VLA 模型，通过伪深度编码器从 RGB 输入推断空间信息，配合新构建的自主导航数据集（13K+ 轨迹含 1K 真实飞行），在模拟和真实环境中比 OpenVLA 成功率高 3.9%，碰撞率低 2.6%。

**[Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](remote_sensing/earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)**

:   Earth-Agent是首个基于MCP工具生态的地球观测Agent框架，统一了RGB和光谱遥感数据，通过动态调用104个专家工具实现跨模态、多步骤、定量时空推理，配套提出的Earth-Bench基准包含248个专家任务和13,729张图像，实验证明Earth-Agent远超通用Agent和遥感MLLM。

**[Measuring the Intrinsic Dimension of Earth Representations](remote_sensing/measuring_the_intrinsic_dimension_of_earth_representations.md)**

:   首次系统度量地理隐式神经表示（Geographic INR）的内在维度（ID），发现256-512维嵌入的真实ID仅2-10维；冻结嵌入空间的高ID与好的下游性能正相关，而监督任务头激活空间的低ID与高性能正相关，揭示了「代表性 vs 任务对齐」的双重机制。

**[Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](remote_sensing/spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us.md)**

:   研究未经高光谱预训练的多模态地理空间基础模型 TerraMind 能否通过通道适配策略（朴素波段选择 vs. SRF 分组）有效适配高光谱下游任务，结果表明朴素波段选择一致优于物理感知的 SRF 方法，但性能差距随任务光谱复杂度增大而扩大。

**[TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)**

:   提出 TAMMs——首个统一框架，在单一 MLLM-扩散架构中联合执行卫星图像时间序列的时序变化描述（TCD）和未来图像预测（FSIF），通过时序适配模块（TAM）唤醒冻结 MLLM 的时序推理能力，并通过语义融合控制注入（SFCI）机制将变化理解转化为生成控制信号。

**[Task-free Adaptive Meta Black-box Optimization](remote_sensing/task-free_adaptive_meta_black-box_optimization.md)**

:   提出 ABOM——一种无需预定义训练任务的自适应元黑盒优化器，通过将进化算子（选择、交叉、变异）参数化为可微注意力模块，在优化过程中利用自生成数据在线更新参数，在合成基准和无人机路径规划上实现零样本竞争性能。

---

## 🔎 AIGC检测 { #aigc_detection }

**[Calibrating Verbalized Confidence with Self-Generated Distractors](aigc_detection/calibrating_verbalized_confidence_with_self-generated_distractors.md)**

:   提出 DiNCo 方法，通过让 LLM **独立**评估自动生成的干扰选项（合理但错误的替代答案）来暴露其"暗示性偏差"，用干扰项上的总置信度进行归一化，并融合生成一致性与验证一致性两个互补维度，在短文本 QA 和长文本生成任务上显著改善置信度校准。

**[CLARC: C/C++ Benchmark for Robust Code Search](aigc_detection/clarc_cc_benchmark_for_robust_code_search.md)**

:   构建首个可编译的 C/C++ 代码检索基准 CLARC（6717 查询-代码对），自动化 pipeline 从 GitHub 提取代码并用 LLM+假设检验生成/验证查询；覆盖标准/匿名化/汇编/WebAssembly 四种检索场景，揭示现有代码嵌入模型过度依赖词汇特征（匿名化后 NDCG@10 从 0.89 降至 0.67）且在二进制级别检索上严重不足。

**[Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](aigc_detection/death_of_the_novelty_beyond_n-gram_novelty_as_a_metric_for_textual_creativity.md)**

:   通过 26 位专业作家对 8618 条表达的 close reading 标注，揭示 n-gram 新颖度不足以衡量文本创造力——约 91% 的高 n-gram 新颖表达并不被认为具有创造性，且开源 LLM 中高 n-gram 新颖度与低语用合理性负相关。

**[DMAP: A Distribution Map for Text](aigc_detection/dmap_a_distribution_map_for_text.md)**

:   提出 DMAP（Distribution Map），一种将文本经由语言模型的 next-token 概率排序映射为 $[0,1]$ 区间上 i.i.d. 样本的数学框架，理论证明纯采样文本产生均匀分布，由此可用 $\chi^2$ 检验验证生成参数、揭示概率曲率类检测器在纯采样下彻底失效的根本原因，并可视化后训练（SFT/RLHF）在下游模型中留下的统计指纹。

**[PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](aigc_detection/policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)**

:   基于欧洲议会13年（2009-2022）共2225条高质量审议记录构建PoliCon基准，通过设计不同投票机制（简单多数/2/3多数/否决权）、权力结构和政治目标（功利主义/罗尔斯主义），系统评估LLM起草政治共识决议的能力，揭示前沿模型在复杂共识任务中的不足及固有党派偏见。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](multilingual_mt/assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)**

:   提出 ASSESS 框架，其核心是 TransTED Similarity 指标——通过将形式化数学命题解析为算子树 (Operator Tree)，并在标准树编辑距离 (TED) 基础上融入 Lean 证明策略驱动的语义变换，实现了在 EPLA 基准上 70.16% 准确率和 0.35 Kappa 分数的 SOTA 性能，同时仅需 CPU 资源即可复现。

**[ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](multilingual_mt/atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)**

:   提出 Adaptive Transfer Scaling Law (ATLAS)，通过将有效数据量分解为目标语言、迁移语言和其他语言三项并引入数据重复饱和函数，在774个多语言训练实验（10M–8B参数、400+语言）上显著优于现有scaling law（多语言 $R^2$ 从0.67提升至0.98），并系统量化了跨语言迁移矩阵、多语言诅咒的容量约束以及预训练vs微调的计算交叉点。

**[Multilingual Routing in Mixture-of-Experts](multilingual_mt/multilingual_routing_in_mixture-of-experts.md)**

:   系统分析了MoE大语言模型中多语言路由模式，发现中间层存在跨语言共享专家且语言性能与英语路由对齐度强相关，进而提出推理时路由干预方法，通过在中间层激活英语任务专家，在3个模型×2个任务×15+语言上一致性地提升多语言性能1-2%。

**[Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity](multilingual_mt/prior-based_noisy_text_data_filtering_fast_and_strong_alternative_for_perplexity.md)**

:   提出基于 token 词频先验（term frequency）的文本数据过滤方法，通过计算文档中 token 先验的均值和标准差来检测异常文档，实现了比 PPL 过滤快 1000× 以上且下游性能更优的数据清洗效果。

**[SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)**

:   利用稀疏自编码器（SAE）发现 LLM 中意外语言切换与目标语言特征异常高预激活值相关，提出 SASFT 方法在 SFT 训练中约束语言特征预激活值，将意外代码切换降低 50% 以上。

---

## 🗣️ 对话系统 { #dialogue }

**[AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)**

:   提出 AQuA，首个按模糊度细粒度分级（4 级）的视觉问答数据集（7.2K 样本），为每级定义最优回应策略（直接回答/推断/列举/请求澄清），发现 GPT-5 和 Gemini 在模糊 VQA 上都过度自信地直接回答，通过 SFT+GRPO 训练的 3B 模型反而能超越闭源大模型的策略适应能力。

**[Non-Collaborative User Simulators for Tool Agents](dialogue/non-collaborative_user_simulators_for_tool_agents.md)**

:   基于marketing研究定义四类非协作用户行为（不可用服务/跑题闲聊/不耐烦/不完整表述），构建了可保持goal-alignment的模拟框架，在MultiWOZ和τ-bench上系统暴露了SOTA工具Agent的行为特异性失败机制——跑题闲聊导致平均SR下降29.1%，且不同模型呈现截然不同的崩溃路径（GPT系列陷入helper API重复调用，Qwen系列倾向于幻觉编造API结果）。

**[ReIn: Conversational Error Recovery with Reasoning Inception](dialogue/rein_conversational_error_recovery_with_reasoning_inception.md)**

:   提出 Reasoning Inception（ReIn），一种无需修改模型参数或系统提示的测试时干预方法，通过外部 inception 模块检测对话错误并将恢复计划注入任务 agent 的推理链中，在多种错误场景下显著提升对话任务完成率，且可泛化至未见错误类型。

**[Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding](dialogue/understanding_language_prior_of_lvlms_by_contrasting_chain-of-embedding.md)**

:   通过对比有/无视觉输入的逐层隐藏表征（chain-of-embedding），发现LVLM中存在一个"视觉整合点"(VIP)层，并据此提出Total Visual Integration (TVI)指标来量化语言先验的强度。

---

## ✍️ 文本生成 { #nlp_generation }

**[AP-OOD: Attention Pooling for Out-of-Distribution Detection](nlp_generation/ap-ood_attention_pooling_for_out-of-distribution_detection.md)**

:   提出AP-OOD，将Mahalanobis距离的均值池化替换为可学习的注意力池化，解决了均值池化丢失token级异常信息的问题，在文本OOD检测中将XSUM摘要的FPR95从27.84%降至4.67%，支持无监督到半监督的平滑过渡。

**[FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)**

:   提出 FS-DFM（Few-Step Discrete Flow-Matching），通过步数感知训练和累积标量更新规则，将离散 flow-matching 语言模型的采样步数从 1024 步降低到 8 步，实现 128 倍加速，同时保持相当的困惑度和生成质量。

---

## ⚛️ 物理学 { #physics }

**[Feedback-driven Recurrent Quantum Neural Network Universality](physics/feedback-driven_recurrent_quantum_neural_network_universality.md)**

:   本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。

**[Sublinear Time Quantum Algorithm for Attention Approximation](physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)**

:   提出首个对序列长度 $n$ 具有**亚线性**时间复杂度的量子数据结构，用于近似 Transformer 注意力矩阵的行查询，预处理时间 $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$，每次行查询 $\widetilde{O}(s_\lambda^2 + s_\lambda d)$，相对经典算法实现了关于 $n$ 的二次加速。

---

## 📂 其他 { #others }

**[A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](others/a_federated_generalized_expectation-maximization_algorithm_for_mixture_models_wi.md)**

:   提出 FedGEM 算法，通过客户端本地 EM 步后构建不确定性集、服务器利用不确定性集交集检测聚类重叠并推断全局聚类数，首次实现在全局聚类数未知情况下的联邦聚类，并提供了概率收敛保证。

**[A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](others/a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)**

:   为线性多元 Hawkes 过程在 RKHS 框架下的触发核估计建立了新型表示定理，证明最优估计器可用等价核在数据点上的线性组合表示且对偶系数全部解析地等于 1，无需求解对偶优化问题，从而实现高效可扩展的非参数估计。

**[A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction](others/a_scalable_inter-edge_correlation_modeling_in_copulagnn_for_link_sign_prediction.md)**

:   将 CopulaGNN 从节点级扩展到边级，通过将相关矩阵构造为边嵌入的 Gramian 矩阵并利用 Woodbury 恒等式重构条件概率分布，实现了在签名图上对边间统计依赖的可扩展建模，用于链接符号预测任务。

**[A Single Architecture for Representing Invariance Under Any Space Group](others/a_single_architecture_for_representing_invariance_under_any_space_group.md)**

:   设计了一种可自适应任意空间群不变性的单一架构 (Crystal Fourier Transformer)，通过解析推导群操作对傅里叶系数的约束来构造对称适配的傅里叶基，用约束的对偶图表示实现了跨 230 个空间群的参数共享和零样本泛化。

**[Active Learning for Decision Trees with Provable Guarantees](others/active_learning_for_decision_trees_with_provable_guarantees.md)**

:   为决策树主动学习提供首个理论保证：(1) 首次分析决策树的不一致系数（disagreement coefficient）并给出 $O(\ln^{OPT}(n))$ 上界；(2) 提出首个达到乘法误差 $(1+\epsilon)$ 保证的二分类主动学习算法；结合两者实现数据集大小的多对数标签复杂度。

**[Addressing Divergent Representations from Causal Interventions on Neural Networks](others/addressing_divergent_representations_causal.md)**

:   系统性地揭示因果干预（activation patching、DAS、SAE 等）会将模型内部表征推离自然分布，理论区分"无害偏移"与"有害偏移"两类情况，并提出 Counterfactual Latent (CL) loss 来约束干预表征不偏离流形，在 7B LLM 上验证可减少偏移同时保持干预准确率。

**[Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](others/agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)**

:   提出Agnostics，一种语言无关的后训练pipeline：将编程任务统一为I/O行为规范格式，用通用验证器+GRPO强化学习训练LLM在任何编程语言上编码，使Qwen 4B在Lua/Julia/R/OCaml/Fortran五种低资源语言上达到匹敌16B-70B模型的SOTA水平。

**[An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](others/an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)**

:   提出 **information gap** 这一信息论度量，通过推导在似然编码和后验编码假说下解码器交叉熵性能差异的解析表达式（本质是真实后验与任务边际化代理后验之间的 KL 散度），定量评估给定实验设计区分两种概率神经编码假说的能力，并通过最大化该度量来优化刺激先验分布，实现理论驱动的最优实验设计。

**[ANO: Faster is Better in Noisy Landscapes](others/ano_faster_is_better_in_noisy_landscape.md)**

:   提出 Ano 优化器，将更新方向和幅度解耦——方向用动量的符号（sign）确保噪声鲁棒，幅度用瞬时梯度绝对值（而非动量幅度）确保响应速度，配合改进的 Yogi 式方差估计，在噪声和非平稳环境（如 RL）中显著优于 Adam/Lion/Adan，同时在标准任务上保持竞争力。

**[AnyUp: Universal Feature Upsampling](others/anyup_universal_feature_upsampling.md)**

:   AnyUp 提出首个**编码器无关**的可学习特征上采样方法，通过 feature-agnostic 卷积层和窗口注意力机制，仅训练一次即可对任意视觉特征在任意分辨率间进行高质量上采样，在语义分割、深度估计等任务上达到 SOTA。

**[Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects](others/articulation_in_motion_prior-free_part_mobility_analysis_for_articulated_objects.md)**

:   提出AiM（Articulation in Motion）框架，从交互视频和初始状态扫描中无需部件数量先验地重建铰接物体——通过双高斯表征（静态GS + 可变形GS）实现动静解耦，结合顺序RANSAC进行无先验部件分割和关节估计，辅以SDMD模块处理新暴露的静态区域，在复杂6部件物体（Storage）上以79.34% mean IoU大幅超越需先验的ArtGS（52.23%）。

**[CaDrift: A Time-dependent Causal Generator of Drifting Data Streams](others/cadrift_a_time-dependent_causal_generator_of_drifting_data_streams.md)**

:   提出 CaDrift，一个基于结构因果模型（SCM）的时间依赖合成数据流生成框架，通过 EWMA 平滑和自回归噪声引入时序相关性，并通过修改因果映射函数实现可控的分布漂移、协变量漂移、严重漂移和局部漂移，填补了现有数据流生成器既不因果又不时序依赖的空白。

**[Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](others/characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)**

:   从物理系统角度分析 Instant-NGP 的多分辨率哈希编码（MHE），推导出其点扩展函数（PSF）的闭式近似，发现有效分辨率由平均分辨率 $N_{\text{avg}}$ 而非最细分辨率 $N_{\max}$ 决定，且存在网格引起的各向异性，并提出零开销的 Rotated MHE（R-MHE）通过逐层旋转输入坐标消除各向异性。

**[CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](others/chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)**

:   CHLU 是一种基于相对论哈密顿力学和辛积分的计算学习原语，通过强制相空间体积守恒和引入因果速度上限，解决了 LSTM 的梯度爆炸/消失和 Neural ODE 的信息耗散问题，实现无限时域稳定性和热力学生成能力。

**[Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](others/completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)**

:   提出DREAM——基于对立立场初始化的多Agent多轮辩论框架用于IR相关性标注：一致时自动标注、分歧时交给人工(含辩论历史辅助)。达到95.2% balanced accuracy且仅3.5%需人工介入，据此构建BRIDGE基准数据集，发现29,824个原有基准缺失的相关标注(原标注的428%)，修正了检索系统排名偏差和RAG中检索-生成性能不匹配问题。

**[Compositional Diffusion with Guided Search for Long-Horizon Planning](others/compositional_diffusion_long_horizon_planning.md)**

:   提出 CDGS（Compositional Diffusion with Guided Search），通过在扩散去噪过程中嵌入基于种群的搜索机制（迭代重采样 + 似然剪枝），解决组合式扩散模型在多模态局部分布合成时的模式平均问题，从短时域模型采样出全局一致的长时域规划。

**[Consistent Low-Rank Approximation](others/consistent_low-rank_approximation.md)**

:   提出并系统研究"一致低秩近似"问题——在流数据中逐行到达的矩阵上维护近最优 rank-$k$ 近似的同时最小化解的总变化量（recourse），证明加性误差下 $O(k/\varepsilon \cdot \log(nd))$ recourse 可行，乘性 $(1+\varepsilon)$ 误差下 $k^{3/2}/\varepsilon^2 \cdot \text{polylog}$ recourse 可行，并给出 $\Omega(k/\varepsilon \cdot \log(n/k))$ 的下界。

**[Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](others/directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)**

:   本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子，统一并推广了现有的图和超图 Laplacian，在 7 个真实数据集上相对准确率提升 2%–20%。

**[Distributed Algorithms for Euclidean Clustering](others/distributed_algorithms_for_euclidean_clustering.md)**

:   在分布式环境下为 Euclidean $(k,z)$-clustering 构造 $(1+\varepsilon)$-coreset，在 coordinator 模型和 blackboard 模型中均达到通信复杂度的最优下界（至多差 polylog 因子）。

**[Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)**

:   提出一种分布鲁棒学习框架，通过联合建模目标域协变量分布和条件标签分布的不确定性，在目标数据极度稀缺或源域存在虚假相关性的UDA场景中显著提升泛化性能。

**[DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](others/distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)**

:   DA-AC 提出将动作分布的参数（如 softmax 概率或 Gaussian 均值/方差）作为 Agent 的"动作"输出，将动作采样过程移入环境，从而用统一的确定性策略梯度框架处理离散/连续/混合动作空间，理论证明方差严格低于 LR 和 RP 估计器，并在 40+ 环境上取得 competitive 或 SOTA 性能。

**[Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](others/enhancing_generative_auto_bidding.md)**

:   提出 AIGB-Pearl，为生成式自动竞价方法引入离线轨迹评估器和 KL-Lipschitz 约束的分数最大化方案，使生成模型能在理论保证下安全地突破静态离线数据的性能天花板，在淘宝真实广告系统上实现 GMV +3% 的显著提升。

**[Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](others/entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)**

:   揭示了低损失路径上曲率的系统性增长会产生熵力屏障，即使路径能量平坦，SGD噪声也会将优化动力学约束在最小值附近的平坦区域，从而解释了"模式连通但动力学受限"的悖论。

**[Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](others/evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)**

:   建立GFlowNet中状态流函数与策略评价函数之间的理论联系，提出子轨迹评价平衡（Sub-EB）目标用于可靠学习评价函数，增强策略基GFlowNet训练的稳定性和灵活性。

**[Exchangeability of GNN Representations with Applications to Graph Retrieval](others/exchangeability_gnn_representations.md)**

:   发现训练好的 GNN 节点嵌入沿特征维度是可交换随机变量（即 $p(\mathbf{X}) = p(\mathbf{X}\pi)$ 对任意维度排列 $\pi$ 成立），利用此性质通过维度排序将基于传输距离（EMD/Wasserstein）的图相似度近似为欧氏距离，构建统一的局部敏感哈希（LSH）框架 GraphHash，在子图匹配和图编辑距离（GED）检索任务上以 AUC 指标一致超越 FourierHashNet、DiskANN、IVF、CORGII、SWWL 等基线，可扩展到 100 万图语料库。

**[Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](others/fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)**

:   揭示Cholesky流形上的简单乘积结构，基于此提出两种快速且数值稳定的SPD度量（PCM和BWCM），所有黎曼算子均有闭式表达式，在SPD深度学习中实现效果、效率和稳定性的三重提升。

**[FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives](others/fastlsq_solving_pdes_in_one_shot_via_fourier_features_with_exact_analytical_deri.md)**

:   利用正弦基函数的循环导数闭式结构，实现了无需自动微分、无需迭代训练的 PDE 一次性求解框架，在线性 PDE 上 0.07s 达到 $10^{-7}$ 精度，非线性 PDE 上 <9s 达到 $10^{-8}$–$10^{-9}$ 精度，比 PINNs 快数千倍且精确数个数量级。

**[Federated ADMM from Bayesian Duality](others/federated_admm_from_bayesian_duality.md)**

:   从变分贝叶斯(VB)视角推导出ADMM的贝叶斯对偶结构，证明经典ADMM是VB在各向同性高斯族上的特例，并导出Newton-like（二次目标一轮收敛）和Adam-like（深度异构场景+7%准确率）两个新扩展。

**[FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](others/fire_frobenius_isometry_reinitialization.md)**

:   将持续学习中的稳定性-可塑性平衡形式化为约束优化问题——最小化权重偏差（稳定性）同时约束权重正交性（可塑性），得到正交 Procrustes 问题的闭式解 $\tilde{W}^* = W(W^\top W)^{-1/2}$（极分解），通过 Newton-Schulz 迭代高效实现（<1% 额外时间），在视觉持续学习、LLM 持续预训练和 RL 上全面超越 S&P 等基线。

**[From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding](others/from_movement_to_cognitive_maps.md)**

:   结合幼鼠运动发育的聚类分析和浅层 RNN 预测学习模型，首次计算性地证明运动统计特征的发育变化（爬行→行走→奔跑→成年）驱动了海马空间调谐神经元（位置细胞、方向细胞、联合编码细胞）的序贯涌现，定量复现大鼠海马记录数据的发育时间线，并预测了联合位置-方向编码细胞在发育中逐渐增多这一现象且在实验数据中得到验证。

**[Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](others/harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)**

:   将流形理论从图像扩展到表格数据扩散模型，证明任意可微推理时损失的梯度都位于数据流形切线空间中（不限于平方误差损失），据此提出Harpoon方法在推理时沿流形引导无条件样本满足多样化表格约束。

**[HEEGNet: Hyperbolic Embeddings for EEG](others/heegnet_hyperbolic_embeddings_for_eeg.md)**

:   首次系统验证EEG数据具有双曲性（层次结构），提出HEEGNet混合双曲网络架构，结合欧几里得编码器提取时空频谱特征和双曲编码器捕捉层次关系，配合创新的粗到细域适应策略(DSMDBN)，在视觉诱发电位、情感识别和颅内EEG多个跨域任务上达到SOTA。

**[Hilbert-Guided Sparse Local Attention](others/hilbert-guided_sparse_local_attention.md)**

:   利用Hilbert空间填充曲线将2D图像token重排为保持空间邻近性的1D序列，大幅提升局部注意力的块稀疏率（空块比例从87.5%到96.9%），结合FlexAttention实现窗口注意力4倍和滑动注意力18倍加速，精度损失极小。

**[Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](others/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)**

:   首次证明mini-batch Adam的隐式偏差与full-batch不同：构造数据集使单样本Adam收敛到 $\ell_2$ 最大间隔分类器（而full-batch Adam收敛到 $\ell_\infty$），并通过AdamProxy刻画一般数据集上的数据自适应Mahalanobis范数间隔最大化行为。

**[Improving Black-Box Generative Attacks via Generator Semantic Consistency](others/improving_black-box_generative_attacks_via_generator_semantic_consistency.md)**

:   通过分析生成器中间层特征的语义退化现象，提出基于 Mean Teacher 的语义结构感知框架，在生成器早期层进行自特征蒸馏以保持语义一致性，从而增强对抗样本在跨模型、跨域、跨任务场景中的可迁移性。

**[In-Context Algebra](others/in-context_algebra.md)**

:   本文设计了一个 **in-context 代数任务**——令 token 成为纯变量、每条序列重新随机分配含义——发现 Transformer 在此设定下不再学习经典的傅里叶/几何表示，而是涌现出三种 **符号推理机制**（交换复制、单位元识别、闭包消去），并揭示了训练过程中这些能力按阶段性相变依次出现的规律。

**[Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch RL](others/jackpot_optimal_budgeted_rejection_sampling_for_extreme_actor-policy_mismatch_re.md)**

:   提出 Jackpot 框架，通过 Optimal Budget Rejection Sampling（OBRS）以可控接受预算在 token 级别拒绝/重加权 rollout 样本，理论证明任意预算下都能严格缩小 actor-policy 间 KL 散度，配合 rollout 模型联合训练与蒸馏，使小模型（如 Qwen3-1.7B）rollout 训练大模型（如 Qwen3-8B）达到接近 on-policy 的性能。

**[Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](others/key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)**

:   理论证明Transformer自注意力中Query/Key/Value权重三元组存在冗余——Query权重可被替换为单位矩阵（减少25%注意力参数），GPT风格模型从头训练验证在适当超参数调整下性能不降，且训练在3倍更低权重衰减下仍然稳定。

**[Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](others/latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)**

:   在潜空间中学习/预定义等变移位算子来处理旋转和平移等群变换，推理时通过KNN搜索推断变换参数并恢复到标准pose后分类，在MNIST上展示了训练范围外变换的成功外推能力，相比传统网络和等变网络更灵活，但向复杂数据集扩展仍面临挑战。

**[Latent Fourier Transform](others/latent_fourier_transform.md)**

:   提出 LatentFT 框架，在扩散自编码器的潜在时间序列表征上应用离散傅里叶变换按时间尺度分离音乐模式，训练时使用随机相关对数频率掩码让解码器学习从部分频谱重建，推理时用户通过指定频率掩码选择性保留/混合不同时间尺度的音乐元素，在条件生成和音乐融合任务上全面超越 ILVR/Guidance/Codec Filtering/RAVE 等基线，29 名音乐家听力测试统计显著确认其音质和融合能力优越。

**[LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)**

:   LPWM 是首个能扩展到真实世界多物体数据集的自监督物体中心世界模型，核心创新是为每个粒子学习独立的潜在动作分布（per-particle latent actions），通过因果时空 Transformer 并行编码所有帧，支持动作/语言/图像目标/多视角等多种条件生成，在视频预测上达到 SOTA 并展示了模仿学习能力（OGBench task3 成功率 89%）。

**[Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](others/learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)**

:   提出ADAlign框架，利用神经特征函数在谱域自适应对齐源/目标图分布——无需手动选择对齐标准，自动识别每个迁移场景中最显著的分布差异。在10个数据集16个迁移任务上达SOTA，同时降低内存和训练时间。

**[Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](others/learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)**

:   本文利用代数几何工具，对多项式激活的 MLP 和 CNN 进行了系统性分析：证明了 MLP 的有限可辨识性和 CNN 的唯一可辨识性，揭示了稀疏子网络对应神经流形的奇异点，并从"临界暴露性"角度给出了 MLP 稀疏偏差的几何解释——而 CNN 不具备这种偏差。

**[Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](others/learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)**

:   提出DiffGDA——首个将扩散模型引入图域适应(GDA)的方法，用随机微分方程(SDE)建模源图到目标图的连续时间结构-语义联合演化过程，配合基于密度比的域感知引导网络驾驶扩散轨迹朝向目标域，理论证明收敛到最优适应路径，在8个真实数据集14个迁移任务上全面超越SOTA。

**[LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](others/lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)**

:   提出LipNeXt——首个无约束、无卷积的1-Lipschitz架构，通过正交流形优化学习正交矩阵 + 由Theorem 1理论驱动的Spatial Shift Module实现空间混合，成功扩展到十亿参数规模，在CIFAR-10/100、Tiny-ImageNet和ImageNet上全面刷新认证鲁棒精度(CRA) SOTA，ImageNet上 $\varepsilon=1$ 时CRA提升达+8%。

**[Lipschitz Bandits with Stochastic Delayed Feedback](others/lipschitz_bandits_with_stochastic_delayed_feedback.md)**

:   首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。

**[Missing Mass for Differentially Private Domain Discovery](others/missing_mass_for_differentially_private_domain_discovery.md)**

:   从 missing mass（缺失质量）角度重新审视差分隐私域发现问题，首次为简单且可扩展的 Weighted Gaussian Mechanism (WGM) 在 Zipfian 数据上证明了近最优的 $\ell_1$ 缺失质量上界和无分布假设的 $\ell_\infty$ 缺失质量保证，并将 WGM 作为域发现前置步骤应用于未知域的 private top-$k$ 和 $k$-hitting set 问题，在六个真实数据集上验证了理论结果。

**[Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](others/neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)**

:   提出Neural Force Field（NFF），将物体交互建模为连续力场，通过神经算子学习力场函数并用ODE积分器解码轨迹，在I-PHYRE（100条轨迹）、N-body（200条轨迹）、PHYRE（0.012M数据,比SOTA少267倍）三个基准上实现少样本SOTA，跨场景RMSE降低32-64%,规划任务接近人类水平。

**[Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](others/noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)**

:   提出 NcPU 非对比 PU 学习框架，通过对标准非对比损失做 sqrt 变换（NoiSNCL）让 clean pair 梯度主导训练、用 PhantomGate 提供保守负监督并支持 regret 回退，两个模块在 EM 框架下迭代互利；在不依赖辅助负样本或预估类先验的前提下，CIFAR-100 上与监督学习差距从 14.26% 缩至 <1.4%，xBD 灾损评估上同样达到 SOTA。

**[On the Impact of the Utility in Semivalue-based Data Valuation](others/on_the_impact_of_the_utility_in_semivalue-based_data_valuation.md)**

:   本文通过引入"空间签名"（spatial signature）的几何表示，将数据估值中的 utility 选择问题统一建模为单位圆上的方向旋转问题，并提出了一个量化鲁棒性的指标 $R_p$，揭示了 Banzhaf 值在不同 utility 下表现出最高的排序稳定性。

**[On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](others/on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)**

:   系统研究了三种常用集合聚合函数（sum/mean/max）和注意力机制在三种多集距离函数下的Lipschitz连续性，推导出集合神经网络的Lipschitz常数上界，并将其与扰动稳定性和分布偏移泛化联系起来。

**[Optimizer Choice Matters for the Emergence of Neural Collapse](others/optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)**

:   通过 3,900+ 次训练实验和理论分析，揭示了优化器选择（特别是权重衰减的耦合方式）对 Neural Collapse 现象涌现起关键决定性作用——AdamW（解耦权重衰减）无法产生 Neural Collapse，而 SGD 和 Adam（耦合权重衰减）可以。

**[Out of the Shadows: Exploring a Latent Space for Neural Network Verification](others/out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)**

:   将 zonotope 视为高维超立方体的"投影（影子）"，发现输入集和输出包围体共享同一潜空间，据此提出规范驱动的输入细化方法，将输出端的不安全约束反向传递到输入空间来剪枝，使分支定界子问题数减少 60-65%，且所有运算均为矩阵操作从而实现高效 GPU 加速，在 VNN-COMP'24 八个基准上与 α-β-CROWN 等顶级工具取得可比性能。

**[Oversmoothing, Oversquashing, Heterophily, Long-Range, and More: Demystifying Common Beliefs in Graph Machine Learning](others/oversmoothing_oversquashing_heterophily_long-range_and_more_demystifying_common_.md)**

:   本文系统梳理了图机器学习领域围绕 oversmoothing、oversquashing、同质/异质性和长程依赖的九个常见误区，通过简洁反例逐一反驳，将"oversquashing"拆分为**计算瓶颈**和**拓扑瓶颈**两个独立概念，厘清了领域中广泛存在的概念混淆。

**[OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](others/owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)**

:   提出 OwlEye 框架，利用基于成对距离统计的跨域特征对齐将异构图嵌入共享空间，从多图中提取 attribute-level 和 structure-level 正常模式存入可扩展字典，并通过截断注意力重建机制在完全零样本条件下检测未见图的异常节点，8 数据集平均 AUPRC 36.17% 超越最强 baseline ARC 约 5.4 个百分点。

**[Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)**

:   提出 Hermite 特征结构假设（HEA），仅用数据协方差矩阵和目标函数的 Hermite 分解两个统计量，就能解析预测旋转不变核在真实图像数据集（CIFAR-5m、SVHN、ImageNet）上的学习曲线（测试误差 vs 样本量），并证明该假设在高斯数据下成立，且 MLP 在特征学习 regime 下也按 HEA 预测的顺序学习 Hermite 多项式。

**[Probabilistic Kernel Function for Fast Angle Testing](others/probabilistic_kernel_function_for_fast_angle_testing.md)**

:   本文研究高维欧氏空间中的角度测试问题，提出两个基于参考角度的确定性概率核函数 $K_S^1$ 和 $K_S^2$，分别用于角度比较和角度阈值判断，无需高斯分布的渐近假设即可获得理论保证，并将其应用于近似最近邻搜索（ANNS），在 HNSW 图上实现 2.5×–3× 的 QPS 加速。

**[Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](others/revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)**

:   对 SAM 的底层机制提出新的直觉解释——扰动点梯度近似局部最大值方向，并揭示其不精确性及多步退化问题，进而提出 XSAM 通过显式估计最大值方向实现更忠实更有效的锐度感知最小化。

**[Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](others/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)**

:   提出 Random Wavelet Features (RWF)，通过从小波族中随机采样构建可扩展的非平稳核近似，保留随机特征的线性时间复杂度，同时具有正定性、无偏性和一致收敛保证。

**[SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](others/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)**

:   提出 SEED（Semantic Evaluation for Visual Brain Decoding），一个结合 Object F1、Cap-Sim 和 EffNet 三个互补指标的组合评估度量，在与人类评估的对齐度上显著超越现有所有指标。

**[Speculative Actions: A Lossless Framework for Faster AI Agents](others/speculative_actions_faster_ai_agents.md)**

:   借鉴 CPU 推测执行和 LLM 推测解码的思想，提出 Speculative Actions 框架：在慢速 Actor（大模型）计算时用快速 Speculator（小模型）预测未来动作并预执行，匹配时跳过等待实现无损加速，在 Chess/电商/问答等场景实现 15-30% 延迟降低，置信度动态分支策略用 40% 更少 token 达到近似 3 条推测的加速效果。

**[t-SNE Exaggerates Clusters, Provably](others/t-sne_exaggerates_clusters_provably.md)**

:   从理论上严格证明 t-SNE 存在两个根本性失败模式：（1）无法从输出推断输入聚类的强度，（2）无法忠实地展示极端离群点——即使输入毫无聚类结构或存在极端离群点，t-SNE 也可能产生完美聚类的可视化。

**[The Counting Power of Transformers](others/the_counting_power_of_transformers.md)**

:   证明 Transformer 不仅能捕获（半）线性计数性质，还能表达所有**半代数计数性质**（即多元多项式不等式的布尔组合），从而推广了先前关于 Transformer 计数能力的所有结果，并由此推导出新的不可判定性结论。

**[The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](others/the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)**

:   将AI模型错误分解为偏差（systematic misalignment）和方差（incoherent behavior），发现：推理越长→越不连贯；更大模型在困难任务上更不连贯。这暗示未来超级AI更可能表现为"工业事故"式的不可预测失败，而非一致追求错误目标。

**[The Invisibility Hypothesis: Promises of AGI and the Future of the Global South](others/the_invisibility_hypothesis_promises_of_agi_and_the_future_of_the_global_south.md)**

:   提出"不可见性假说"（Invisibility Hypothesis），论证AI系统日益成为经济和政治分配的协调层时将系统性偏向"机器可读"个体，全球南方的非正式工人因缺乏数字可验证性而被管理性排斥（managed exclusion），核心风险从job displacement转向relevance loss，且排斥具有自我强化特性。

**[The Price of Robustness: Stable Classifiers Need Overparameterization](others/the_price_of_robustness_stable_classifiers_need_overparameterization.md)**

:   建立了不连续分类器的稳定性-泛化界，证明了分类任务中的"鲁棒性代价定律"：任何参数量 $p \approx n$ 的插值分类器必然不稳定，实现高稳定性需要 $p \approx nd$ 量级的过参数化。

**[Towards Sustainable Investment Policies Informed by Opponent Shaping](others/towards_sustainable_investment_policies_informed_by_opponent_shaping.md)**

:   形式化证明 InvestESG 模拟环境在何种条件下构成社会困境，并应用 Advantage Alignment 对抗塑形算法引导经济智能体走向可持续投资均衡。

**[Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)**

:   提出基于皮层兴奋-抑制（E-I）回路的无归一化学习框架 DeepEISNN，通过 E-I Init 和 E-I Prop 两项技术实现深度 SNN 的稳定端到端训练，兼顾性能与生物合理性。

**[When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](others/when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)**

:   CALIPER提出了一种检测器和模型无关的、仅依赖数据的检验方法，通过跟踪加权局部回归的代理误差随局部性参数$\theta$的单调性变化，来估计突发概念漂移后重训练所需的最小数据量，无需实际重训练下游模型。

</div>