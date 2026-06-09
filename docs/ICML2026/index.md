---
title: >-
  1463 篇 ICML2026 论文解读 · 每篇 5 分钟读懂
description: >-
  1463篇ICML2026论文解读，涵盖图像生成(125篇)、多模态 VLM(106篇)、模型压缩(101篇)、强化学习(95篇)、可解释性(72篇)、LLM 推理(63篇)、优化/理论(60篇)、LLM 安全(47篇)等 48个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "图像生成"
  - "多模态 VLM"
  - "模型压缩"
  - "强化学习"
  - "可解释性"
  - "LLM 推理"
  - "优化/理论"
  - "LLM 安全"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧪 ICML2026 论文笔记

1463篇ICML2026论文解读，涵盖图像生成(125篇)、多模态 VLM(106篇)、模型压缩(101篇)、强化学习(95篇)、可解释性(72篇)、LLM 推理(63篇)、优化/理论(60篇)、LLM 安全(47篇)等 48个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🎨 图像生成 { #image_generation }

**[A Diffusive Classification Loss for Learning Energy-based Generative Models](image_generation/a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)**

:   这篇论文提出 DiffCLF，把时间噪声层级之间的能量估计改写成分类问题，并与 DSM 联合训练，从而在不引入昂贵最大似然采样的情况下学习更可靠的能量函数，尤其改善了分数匹配在多模态权重上的模式盲区。

**[A Kinetic Energy Perspective of Flow Matching](image_generation/a_kinetic_energy_perspective_of_flow_matching.md)**

:   这篇论文把 flow matching 采样轨迹看成粒子运动，定义 Kinetic Path Energy（KPE）来度量每个样本生成过程的累积动能，并据此提出训练-free 的 Kinetic Trajectory Shaping，在提升生成质量的同时抑制末端能量尖峰导致的记忆化。

**[A Systematic Investigation of RL-Jailbreaking in LLMs](image_generation/a_systematic_investigation_of_rl-jailbreaking_in_llms.md)**

:   这篇论文把 RL-based LLM jailbreaking 当作一个可拆解的 POMDP 系统来研究，发现奖励函数、episode 长度和训练问题数量等环境定义因素，比单纯换 RL 算法更大程度决定自动化红队成功率。

**[A Unified Framework for Diffusion Model Unlearning with f-Divergence](image_generation/a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)**

:   这篇论文把扩散模型概念遗忘中的 MSE/KL 对齐推广到任意 $f$-divergence，提出 f-DMU 框架，并发现 closed-form Hellinger loss 往往比 MSE 更稳、更能保留非目标概念。

**[AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)**

:   AdaEraser 用“目标残留程度”自适应调节扩散模型 self-attention 抑制强度，在不训练新模型的情况下同时提升目标删除完整性和背景重建质量，并在 Mulan 与 OABench 上超过训练式和 training-free object removal 方法。

**[Adapting Noise to Data: Generative Flows from Learned 1D Processes](image_generation/adapting_noise_to_data_generative_flows_from_1d_processes.md)**

:   本文认为 flow/diffusion 模型默认高斯 latent 并不总适合数据分布，提出用可学习的一维 quantile functions 构造数据自适应 product prior，在 flow matching 中联合学习噪声和速度场，从而缩短 transport path 并改善重尾天气数据和低容量图像生成表现。

**[Adversarial Flow Models](image_generation/adversarial_flow_models.md)**

:   作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成，ImageNet-256 上 1NFE FID 刷到 2.38（XL/2）和 1.94（112 层）。

**[AesFormer: Transform Everyday Photos into Beautiful Memories](image_generation/aesformer_transform_everyday_photos_into_beautiful_memories.md)**

:   AesFormer 将日常照片美化定义为 Aesthetic Photo Reconstruction，通过先生成摄影动作计划再执行结构编辑的两阶段框架，把构图、视角和姿态等拍摄时错误转化为可执行编辑，并在 AesRecon 上显著优于开源编辑器、接近 Nano Banana Pro。

**[AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](image_generation/ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)**

:   AG-REPA 发现音频 Flow Matching 中“存储语义信息的层”和“真正驱动速度场的层”并不重合，提出用 forward-only gate ablation 选择因果贡献最高的层做表示对齐，在语音和通用音频生成上比固定层 REPA 更快收敛、更低 FAD。

**[Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](image_generation/alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)**

:   这篇论文提出 Alignment-Guided Score Matching，用 reward-free 的 Plackett-Luce 对齐奖励把正负文本-图像匹配信号直接写入扩散 score matching 目标，通过训练轻量 soft tokens 改善 T2I 语义对齐，同时缓解 SoftREPA 常见的重复生成和计数错误。

**[Anomaly-Preference Image Generation (APO)](image_generation/anomaly-preference_image_generation.md)**

:   作者把"少样本异常图像生成"重写为"无人工标注的偏好优化问题"：真实异常作为正样本，参考模型在同一时刻的去噪偏差作为隐式负样本，通过 DPO 风格 loss 让扩散模型对齐异常分布；再用按时间步调节 LoRA rank 的 TACA 保住结构多样性、用分层 CFG 调节文本-异常对齐强度，在 MVTec 等 benchmark 上同时刷新真实度和多样性。

**[AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](image_generation/ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)**

:   AtelierEval 首次把文本到图像流程中的“提示词编写者”作为评测对象，用 360 个专家任务、三类认知任务和 AtelierJudge agentic evaluator 系统量化人类与 MLLM 的提示词能力，并发现图像模仿式 prompting 往往比纯文本规划式 prompting 更可靠。

**[Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](image_generation/balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)**

:   将扩散模型中 $\mathbf{QK}^\top$ 注意力矩阵分解为对称分量（能量景观）和反对称分量（环流动力学），据此推导 Hopfield 风格的稳定性度量来诊断亚稳态混合，并通过调控反对称分量实现无需训练的保真度-多样性可控权衡。

**[Barriers to Counterfactual Credit Attribution for Autoregressive Models](image_generation/barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)**

:   本文形式化研究生成式模型在 RAG/in-context 部署时的"反事实信用归因（CCA）"问题，证明两条令人惊讶的负面结果：(1) 即便底层 next-token 预测器是 (0,0)-CCA，自回归 rollout 也并非 CCA——CCA 不像 DP 那样在自回归下天然 compose；(2) 对一个已部署的非归因模型做 black-box "CCA retrofitting" 至少需要在输出长度 $\ell$ 上指数级查询次数。

**[Bayesian Tensor Decomposition with Diffusion Model Prior](image_generation/bayesian_tensor_decomposition_with_diffusion_model_prior.md)**

:   DiffBCP 将预训练扩散模型作为隐式数据先验注入贝叶斯 CP 张量分解，通过 split Gibbs 采样器实现可处理的后验推断，在图像修复和去噪任务上全面超越传统和深度张量分解基线（FFHQ 上 PSNR 最高提升 +2.33 dB）。

**[Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](image_generation/beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion.md)**

:   提出 JEPA Guidance，利用 JEPA（如 DINOv2）编码器的隐式密度信号引导扩散模型采样，将少数样本（minority sample）的定义从"生成模型先验下的低密度"转变为"世界先验下的低密度"，在无条件、类条件和文生图场景均实现更具语义意义的稀有样本生成。

**[Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching](image_generation/bootstrap_your_generator_unpaired_visual_editing_with_flow_matching.md)**

:   提出 Bootstrap Your Generator (ByG)，一个无需配对数据的 flow matching 编辑训练框架，通过从冻结基础模型提取编辑方向先验 + cycle consistency 保持源结构 + 梯度路由弥合训练-推理差距，在图像和视频编辑上超越百万级配对数据训练的监督基线。

**[Caracal: Causal Architecture via Spectral Mixing](image_generation/caracal_causal_architecture_via_spectral_mixing.md)**

:   Caracal 用 $\mathcal{O}(L \log L)$ 的多头傅立叶（MHF）模块替换 Transformer 的 $\mathcal{O}(L^2)$ 注意力，通过"pad-FFT-multiply-iFFT-truncate"实现频域内的严格因果掩码，并完全去掉位置编码，仅用标准 FFT 算子（不依赖 Mamba 那样的 CUDA kernel）就在 Tiny→Large 全尺度上与 Llama / Mamba / Mamba-2 / Jamba 性能相当。

**[CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](image_generation/clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)**

:   本文针对视频字幕擦除提出 CLEAR：两阶段训练（Stage I 用 dual encoder + 正交解耦学自监督字幕先验掩码；Stage II 在 Wan2.1 视频扩散模型上加 LoRA + occlusion head 做自适应加权），推理完全不需要任何 mask 或文本检测器，仅训练 0.77% 参数就在中文测试集上把 PSNR 推到 26.80 dB（比最强基线 +6.77 dB），并零样本泛化到 6 种语言。

**[Coarse-Grained Boltzmann Generators](image_generation/coarse-grained_boltzmann_generators.md)**

:   提出 Coarse-Grained Boltzmann Generators (CG-BGs)，在粗粒化坐标空间中结合归一化流生成模型和学到的平均力势 (PMF) 进行重要性采样，以远低于原子级 BG 的计算成本实现渐近正确的分子平衡态采样。

**[CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](image_generation/cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)**

:   本文针对"编辑模型常在不该改的区域乱改"这一痛点，构造 CoCoEdit-40K 局部编辑数据集 + 提出 pixel-level 相似度 reward 补充 MLLM reward + 设计区域正则化 RL 目标（高奖励样本约束非编辑区一致、低奖励样本强迫编辑区做出改变），把 FLUX.1 Kontext 和 Qwen-Image-Edit 同时在编辑得分和 PSNR/SSIM 上提升，打破现有"提编辑能力必伤一致性"的 trade-off。

**[CoFrGeNet: Continued Fraction Architectures for Language Generation](image_generation/cofrgenet_continued_fraction_architectures_for_language_generation.md)**

:   本文把"连续分数（continued fraction）"这种具备最优有理逼近性质的函数类引入到语言生成 Transformer 中，分别为多头注意力和 FFN 设计 CoFrNet 替代模块（CAttnU/CAttnM/Cffn），通过"continuants"封闭形式把 $d$ 次除法降为 1 次，在 GPT2-xl 和 Llama-3.2B 上用 $\frac{2}{3}\sim\frac{1}{2}$ 的参数实现持平甚至更优的下游性能。

**[Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](image_generation/compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)**

:   将视觉信号编码为冻结扩散基础模型上的低秩适配参数（LoRA），并通过哈希映射压缩为单个紧凑向量，在极低码率下实现强感知质量的视频压缩，同时支持推理时缩放和生成式编辑。

**[Conf-Gen: Conformal Uncertainty Quantification for Generative Models](image_generation/conf-gen_conformal_uncertainty_quantification_for_generative_models.md)**

:   提出 Conf-Gen 框架，将保形风险控制（CRC）扩展到生成任务，通过参数化选择函数和可容许性函数为 LLM 问答、图像生成、对话系统、AI Agent 等任务提供形式化的不确定性保证，同时放松了 CRC 的单调性等理论假设。

**[Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards](image_generation/conflict-aware_additive_guidance_for_flow_models_under_compositional_rewards.md)**

:   针对流模型在多目标组合奖励下推理时引导易产生 off-manifold drift 的问题，提出 Conflict-Aware Additive Guidance (CAR)，通过检测梯度冲突并动态切换可学习的值梯度修正，以极低额外计算代价将身份保留提升 25.4%、规划成功率提升 38.75%。

**[Conformal Reliability: A New Evaluation Metric for Conditional Generation](image_generation/conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)**

:   提出基于保形预测（Conformal Prediction）的可靠性评分 CReL，通过在隐空间构建凸预测集并优化最坏情况下的指标表现，实现对条件生成模型的不确定性感知评估，在图文互生任务上揭示了传统单输出指标无法捕捉的模型可靠性差异。

**[Content-Style Identification via Differential Independence](image_generation/content-style_identification_via_differential_independence.md)**

:   本文提出 CSDI（content-style differential independence）这一全新的可辨识性条件——只要 generator 关于 content 与 style 的 Jacobian 列空间在数据流形上相互正交，即可在内容-风格统计**相关**且 Jacobian **稠密**的设定下证明 unpaired 多域内容-风格分块可辨识，并通过 Hutchinson 噪声探测把这一条件做成一个可在 StyleGAN2-ADA 上 scalable 的正则项 $\mathcal{L}_{\rm orth}$，在 AFHQ / CelebA-HQ 的反事实生成与跨域翻译上把 FID 从 5.2 / 4.6 进一步压到 4.4 / 4.3，LPIPS 从 0.40 / 0.26 拉到 0.45 / 0.34。

**[DFlash: Block Diffusion for Flash Speculative Decoding](image_generation/dflash_block_diffusion_for_flash_speculative_decoding.md)**

:   DFlash 用一个轻量级的"块扩散"草稿模型替代 EAGLE-3 那种自回归草稿器，并通过把目标模型的多层 hidden features 作为 KV 注入到草稿模型每一层，在单次前向中并行起草整块 token，端到端最高拿到 6× 无损加速，比 EAGLE-3 再快 2.5× 左右。

**[DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](image_generation/dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)**

:   论文针对"CLIP 微调到 AI 生成图像检测时灾难性遗忘破坏可迁移先验"的问题，提出 DGS-Net：把分类损失的梯度按坐标拆成有害正分量 $g^+$ 与有益负分量 $g^-$，让训练网络的图像梯度先正交投影到冻结 CLIP **文本梯度有害方向**的补空间（Orthogonal Suppression，剔除任务无关语义），再额外对齐到冻结 CLIP **图像梯度有益方向**（Prior Alignment，保住预训练先验），从而在 50 个生成模型上的平均检测精度比 SOTA 高 6.6%。

**[Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](image_generation/diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)**

:   论文用线性探针发现 MM-DiT (FLUX / SD3.5) 在中间层的某些注意力头里、其 text token 的 key 向量天然编码了"目标概念是否会出现"的二元信号，并由此提出 Omission Signal Intervention (OSI)：在 inference 时把"omission 类 - existence 类"的均值差方向以 $\alpha\sigma\boldsymbol{\theta}$ 的强度注入 Top-K 头的 key 向量，激发模型对缺失概念的"自我感知"并补全生成；在 FLUX 上 GenEval 6-object 准确率从 0.18 → 0.40，且无需任何 fine-tune。

**[Diffusion Differentiable Resampling](image_generation/diffusion_differentiable_resampling.md)**

:   本文提出 **diffusion resampling**：用一个**无需训练**的扩散过程为顺序蒙特卡洛 (SMC) 的重采样步骤提供天然可微的 reparametrisation 替代，证明其在 Wasserstein 距离下相对样本数 $N$ 一致收敛，并在多个粒子滤波与参数估计基准上超越 OT / Gumbel-Softmax / Soft 等现有可微重采样方法。

**[Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](image_generation/diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)**

:   本文证明：当数据分布支撑在 $M$ 个低维线性子空间的并集（UoS）上且每个子空间内的分布是 subgaussian 时，存在一个基于核密度的 score 估计器可以让 score-based 扩散采样器以 $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ 个样本达到 1-Wasserstein 误差 $\varepsilon$（$k$ 为最大内在维度），首次在多峰、无 smoothness/有界密度/log-concavity 假设下达到了与内在维度匹配的 minimax 最优率，彻底打破了维度灾难。

**[DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing](image_generation/directedit_step-level_accurate_inversion_for_flow-based_image_editing.md)**

:   DirectEdit 通过在 Rectified Flow 反演过程中记录每一步的潜变量残差 $\Delta\mathbf{Z}_t$ 并在前向路径中提前注入，让重建路径与反演轨迹严格逐步对齐，从而在不增加任何 NFE 的前提下实现"步级精确重建"，并结合 MLLM+SAM 多分支 mask 噪声融合与注意力 Value 注入，在 PIE-Bench 上以综合排名 4.0 (FLUX) / 2.43 (SD3.5) 显著优于 RF-Inversion、FireFlow、FTEdit、DNAEdit 等所有现有训练无关方法。

**[DiScoFormer: Plug-In Density and Score Estimation with Transformers](image_generation/discoformer_plug-in_density_and_score_estimation_with_transformers.md)**

:   本文提出 DiScoFormer，一种对样本顺序置换等变、对坐标仿射等变的 Transformer，用一次前向把任意 i.i.d. 样本集映射到对应密度 $f$ 与 score $\nabla\log f$，并从理论上证明 self-attention 在适当参数化下可精确复现归一化高斯 KDE，实验上在 GMM、Laplace、Student-$t$ 等多种分布、宽样本量与维度范围内全面优于经典 KDE，并可作为即插即用 score oracle 用于 Fisher 信息、熵估计与 Fokker–Planck 类 PDE 求解。

**[Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](image_generation/discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)**

:   本文把连续空间扩散采样里成熟的 off-policy RL 训练技巧（replay buffer、重要性加权、MCMC 探索）首次系统迁移到离散扩散采样器，并进一步推广到 data-to-energy 离散 Schrödinger 桥，在 Ising/Potts、离散化 GMM 等多模分布上显著缓解 mode collapse，最后用它在 VQ-VAE 的离散潜空间里做 data-free 的条件图像生成（后验采样）。

**[Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](image_generation/divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)**

:   本文提出 DiCoME 框架，先用几何正交投影把 CLIP 语义特征和伪造伪迹特征强制解耦成两路互补"专家视图"，再用 Dempster–Shafer 证据融合显式建模两视图间的"认识论冲突"以输出可信的不确定性，在跨数据集和跨伪造方法的 deepfake 检测基准上将平均 AUC 从 0.923 提升到 0.939（cross-dataset）和 0.976（cross-manipulation）。

**[Efficient Learning of Deep State Space Models via Importance Smoothing](image_generation/efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)**

:   本文提出 Parallel Variational Monte Carlo (PVMC)，用 prefix/suffix associative scan 把深度状态空间模型的重要性加权边际平滑分布在 $\mathcal{O}(\log N \times \log T)$ span 内并行算出来，同时支持监督式状态估计和生成式建模，比最快的可微 SMC 基线快约 10×，精度还更高。

**[E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](image_generation/embedding-perturbed_exploration_preference_optimization_for_flow_models.md)**

:   针对 GRPO/DiffusionNFT 等基于组的 RL 在 flow 模型对齐中组内方差崩塌、信号消失的问题，E²PO 在文本嵌入空间注入一组可学习的结构化扰动以维持组内判别方差，并配合噪声感知调度与参考锚定批策略，在 SD3.5-M 上把 GenEval 从 0.917 抬到 0.932 且显著提升多样性。

**[End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)**

:   EOSTok 用单阶段端到端管线把 1D ViT tokenizer 和自回归模型一起训练，靠新提出的 APR（Autoregressive Prediction Reconstruction）loss 把「next-token 预测」的梯度真正传回 pixel space 防止码本崩塌，再用「隐式对齐」把 DINOv2 语义注入 1D 隐空间而不破坏 1D 自回归结构，最终在 ImageNet 256 上无 guidance 拿到 1.48 的 FID（SOTA）。

**[Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](image_generation/enhancing_membership_inference_attacks_on_diffusion_models_from_a_frequency-doma.md)**

:   本文从频域视角分析了扩散模型成员推断攻击（MIA）的失败模式，指出高频内容会同时放大 member 和 hold-out 样本得分的标准差从而稀释成员优势，提出一个无需训练、零额外推理代价的"高频滤波器"模块，只需在计算重建误差前对预测图与目标图做相同的 FFT 低通处理，就能把 Naive/SecMI/PIA 等主流 MIA 在 DDIM、Stable Diffusion 上的 ASR/AUC/TPR@1%FPR 普遍拉高 4–11 个百分点（个别场景 TPR@1%FPR 直接从 6% 跃到 41%）。

**[Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](image_generation/envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)**

:   针对 5-shot 非典型域（航拍 / 水下 / 极暗）下 layout-to-image 生成出现的"表示碎片化"，作者把每个类别的条件表示显式拆成全局语义锚 + 局部可重组原语，并用显著性感知的损失强制前景一致，在 DIOR 上将 Bootstrap FID 从 82.5 压到 74.3、mAP 提到 26.1。

**[Escaping Mode Collapse in LLM Generation via Geometric Regulation](image_generation/escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)**

:   本文从动力系统视角把 LLM 长文本生成中的「模式崩溃」（重复、循环、单调）重新解释为隐藏状态轨迹在表示空间里的「几何坍缩」，并提出 RMR — 在 Transformer value cache 上做轻量低秩阻尼来抑制最具持续性的自我强化方向，从而在极低熵的解码区间（0.8 nats/step）依然保持稳定高质量生成。

**[Esoteric Language Models: A Family of Any-Order Diffusion LLMs](image_generation/esoteric_language_models_a_family_of_any-order_diffusion_llms.md)**

:   Eso-LMs 把 AR 与 Masked Diffusion 在 loss、注意力、采样三个层面深度融合：用一个 causal-on-shuffled-sequence 的去噪 Transformer 同时支持并行扩散和左到右 AR，从而**首次让 MDM 在扩散阶段也能用上精确 KV cache**，在 OWT 长上下文上比 MDLM 快 14–65×、比 BD3-LM 快 3–4×，并在 speed–quality Pareto 前沿上取得 SOTA。

**[EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](image_generation/evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)**

:   EvoGM 把"找 task-vector 合并系数 $\bm{\lambda}$"从手工设计变异算子的进化搜索改写成可学习的生成任务：用一对带 cycle-consistency 的 MLP 生成器从历史的 winner/loser 配对里学高性能区域的分布，并在外层套多轮"基底切换"逐步刷新专家池，在 GLUE 8 任务和 Qwen2.5-1.5B 10 模型 unseen 任务上分别比 PSO-Merging 的 SOTA 平均高约 1.4% 和明显领先。

**[Exploring and Exploiting Stability in Latent Flow Matching](image_generation/exploring_and_exploiting_stability_in_latent_flow_matching.md)**

:   本文系统刻画了 Latent Flow Matching（LFM）的"轨迹稳定性"——同一噪声种子下，剪掉 75% 数据、换大小架构、改训练种子都能产生几乎相同的图像；进而把这个性质转化成两个实用算法：(1) 用 balanced-clustering 剪枝可在 CelebA-HQ 上把 50% 数据剪掉而 FID 反而轻微提升、ImageNet 上 75% 数据可剪；(2) Coarse-to-Fine 两段式生成，把 DiT-XL/2 (675M) 和 DiT-S/2 (33M) 拼起来，推理快 2.15×。

**[$f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](image_generation/f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)**

:   把 GFlowNet/Kimi 里"对 log-prob 差取平方"的 $\mathbb{KL}_{sq}$ 代理损失推广到整族 $f$-散度，得到一族同时具备"on-policy 梯度等于对应 $f$-散度真梯度、off-policy 仍有同一全局最优"的可调控 mode-seeking↔mode-covering 损失，并在合成网格、SynFlowNet 分子生成、扩散模型条件采样和异步 LLM RL（GSM8k / MATH）上验证。

**[Finding DoRI: Discovery of Retained Images in Diffusion Models](image_generation/finding_dori_discovery_of_retained_images_in_diffusion_models.md)**

:   作者用一个简单的对抗 text embedding 优化方法（DoRI）证明：NeMo / Wanda 这类"剪枝定位记忆神经元"的扩散模型记忆缓解方案只是把记忆"藏起来"而非真正擦除，因为记忆在 embedding、激活、权重三个层面都不是局部的；进一步提出对抗微调方案，把训练样本真正从模型里拔出来。

**[Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](image_generation/forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)**

:   本文提出训练无关的多概念遗忘框架 FIA，通过"对比概念显著性 + 时空稀疏筛选"定位每个目标概念所对应的概念敏感神经元，并在融合多概念掩码时显式保留同时响应多个概念的"概念无关神经元"，仅剪掉真正概念专属的连接，在 SD v1.5/v1.4 上以 <0.3% 的总稀疏率同时遗忘十个 Imagenette 类（平均遗忘准确率 1.9%，整体得分 86%）以及多艺术家风格和不良内容。

**[From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](image_generation/from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)**

:   针对"唱歌头像"这一被现有 deepfake 检测器忽视的高难度域，作者一边构建 SHDF 数据集量化"说话→唱歌"的域漂移，一边提出 T-AVFD 框架，用 Alpha-CLIP+多粒度真假文本对比学习"真实人脸的语义模式"，再用差分权重模块自适应融合唇音一致性与人脸语义，仅在真实说话视频上训练就能跨域泛化到唱歌伪造，SHDF AUC 从 50% 量级抬到 80.2%。

**[GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](image_generation/gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)**

:   作者把 T2I 同 prompt 下的样本多样性投到 CLIP 单位超球面上，沿"文本方向 $\mathbf{e}_t$"与"正交主残差方向 $\mathbf{u}_{\text{ind}}$"分别拉开投影展度，并通过对预测干净图 $\hat{x}_{0|t}$ 做梯度优化把这种几何展开搬回扩散/流采样轨迹，在 SD2.1 与 SD3-M 上同时提升 prompt 相关（姿态、构图）与 prompt 无关（背景、风格）的多样性，几乎不损质量与对齐。

**[DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](image_generation/generative_adaptation_of_dynamics_to_environmental_shifts_via_weight-space_diffu.md)**

:   DynaDiff 把"为新环境训练一个预测器"的元学习问题改写成"用扩散模型直接生成完整网络权重"的条件采样问题，借助权重图 + 函数一致性损失 + 动力学感知 prompter，在 4 个 PDE 系统上平均 RMSE 比强基线再降 10.78%。

**[Generative Visual Code Mobile World Models](image_generation/generative_visual_code_mobile_world_models.md)**

:   作者把"移动 GUI 世界模型"重新表述成"VLM 生成可渲染的网页代码"这一新范式，配套提出一套自动把策略轨迹改写成（图像状态、动作）→（推理链、下一状态代码）训练样本的数据合成管线，得到的 gWorld-8B/32B 在 6 个 in/out-of-distribution 基准上同时拿下最佳，并把基线模型平均指令准确率拉高 27–46 个百分点、把渲染失败率压到 <1%。

**[GenExam: A Multidisciplinary Text-to-Image Exam](image_generation/genexam_a_multidisciplinary_text-to-image_exam.md)**

:   GenExam 把"画图考试"作为衡量 T2I 模型推理-理解-生成综合能力的金标准，给 10 个学科、1000 道题各配上 ground-truth 图 + 细粒度评分点，结果连最强闭源模型 Nano Banana Pro 也只有 70.2% strict 分，多数开源 T2I/统一 MLLM 不到 3%。

**[Geometry-Aware Tabular Diffusion](image_generation/geometry-aware_tabular_diffusion.md)**

:   作者提出 GATD（Geometry-Aware Tabular Diffusion），在表格扩散去噪器输入和损失里显式加入"列对之间的角度和长度"几何特征作为辅助监督信号，用一个参数量仅 TabDiff 1/3.5（分类任务甚至 1/25）的小 MLP 就在 10 个数据集上拿下 8/10 Shape、7/10 Trend、9/10 下游效用胜场，并且同一套默认超参可直接迁移到 GNN 和 Transformer 去噪器（27/30 Shape、25/30 Trend 涨点）。

**[Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](image_generation/geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)**

:   本文提出 GMF：用 Diffusion Schrödinger Bridge + Rectified Flow 在潜空间估计每个模态的"传输修正成本"（初始速度平方 $\|v_\theta(z,0)\|^2$），作为一个**与分类器置信度解耦**的几何可靠性信号来动态加权多模态融合，从而打破"模型自己评判自己"的循环依赖，在传感器噪声和语义冲突下显著优于基于置信度的可信融合基线。

**[Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation](image_generation/gradient_preconditioning_for_efficient_and_reliable_reward-guided_generation.md)**

:   通过把 reward 梯度投影到一个用 DFT 块状 $\ell_1/\ell_2$ 范数刻画的"白高斯噪声可行集"上，作者把一步生成模型的 test-time latent 优化变得既快又稳：在 FLUX 上只用 30% 的 wall-clock 时间就追平 SOTA 正则化方法 MPGR 的 Aesthetic Score，并彻底避免 reward hacking。

**[GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](image_generation/guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)**

:   GUDA 把"群组级训练数据归因"重新表述成"如果训练时没有这个群组，模型对该样本的对数似然会掉多少"的反事实问题，用机器遗忘从全量模型上"擦掉某个群组"近似 Leave-One-Group-Out (LOGO) 重训得到的反事实模型，再用 ELBO 差作为归因分数，在 CIFAR-10 和 Stable Diffusion 艺术风格归因上比 CLIP 相似度和实例级梯度归因更准，且比 LOGO 重训快约 100 倍。

**[GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance](image_generation/guidedbridge_training-freely_improving_bridge_models_with_prior_guidance.md)**

:   针对 diffusion bridge 模型（data-to-data 生成），论文提出训练免费的 Prior Guidance (PG)：通过对干净 prior 加扰动构造"弱 prior"，再把强/弱 prior 的两个 denoising 结果做外推来放大模型对 prior 的利用，并进一步用 U 型频率调制（FMPG）和 CFG-FMPG 级联框架，在 Edges→Handbags、DIODE、ImageNet inpainting 等任务上不增训练、不增 NFE 地稳定提升 DDBM / DBIM 等预训练 bridge 模型的 FID。

**[(HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](image_generation/hb-arfm_history-bootstrapped_flow_matching_for_inverse_boiling_reconstruction.md)**

:   HB-ARFM 用"历史观测引导"的条件 flow matching 解决多相沸腾流场的逆问题重建：先用一段历史观测窗口 bootstrap 出初始隐状态，再用同一个条件速度场自回归地把重建向前推进，在仅观测界面几何与界面速度的情况下，首次完成完整温度场与速度场的时空一致重建。

**[HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing](image_generation/holofair_unified_t2i_fairness_evaluation_and_fair-grpo_debiasing.md)**

:   本文构建了一个面向 T2I 模型的统一公平性基准 HoloFair（含 SpaFreq 双流属性分类器 + MGBI 多属性几何均值指标），并在此基础上提出 Fair-GRPO：通过对数比例的多属性 per-prompt 奖励 + KL 正则化 GRPO，在 SD3.5-Medium 上把 MGBI 从 0.5211 提升到 0.6772（+29.9%），同时图像质量保持甚至略有提升。

**[Image Restoration via Diffusion Models with Dynamic Resolution](image_generation/image_restoration_via_diffusion_models_with_dynamic_resolution.md)**

:   SubDAPS / SubDAPS++ 把 DPS、DAPS 这类 pixel-space 扩散复原方法搬进"动态分辨率扩散模型"框架——早期在 $64^2 / 128^2$ 子空间采样、后期才回到 $256^2$ 全分辨率，并用共轭梯度替掉 Langevin、用阈值切换 stochastic / deterministic 采样、再附一个无需额外网络评估的 corrector 步，在 4 类线性 + 2 类非线性复原任务上多数指标超越 pixel 与 latent 扩散方法且推理更快。

**[Information-Geometric Adaptive Sampling for Graph Diffusion](image_generation/information-geometric_adaptive_sampling_for_graph_diffusion.md)**

:   本文把图扩散反向 SDE 的采样轨迹看成 Riemannian 统计流形上的参数曲线，用 Fisher-Rao 度量推出一个无需训练的 Drift Variation Score (DVS) 来度量轨迹的局部"信息曲率"，并据此自适应缩放步长，使每步在信息流形上前进等长，从而在分子（QM9/ZINC250k）和图（Planar/SBM/Ego）生成中以更少步数取得更高 FCD / MMD 保真度。

**[Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](image_generation/initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)**

:   本文把"初始噪声"看作可以从一个由 conditional guidance 势能定义的后验中采样的随机变量，提出 DivIn：用一步 Langevin 动力学把标准高斯噪声往"低势能、平坦"的区域里推一推，在几乎不增加推理开销的前提下显著缓解扩散/flow matching 模型的 mode collapse，并和已有的 trajectory-based 多样性方法正交叠加。

**[Krause Synchronization Transformers](image_generation/krause_synchronization_transformers.md)**

:   作者把 Krause 有界置信共识模型搬进 Transformer，用"距离-RBF+局部窗+top-k 稀疏"替代全局 softmax 相似度，从理论上证明它鼓励多簇同步而非全局塌缩，并在 ViT / 自回归图像生成 / LLM 上同时获得更优性能和 30%+ 算力节省。

**[Latent Diffusion Pretraining for Crystal Property Prediction](image_generation/latent_diffusion_pretraining_for_crystal_property_prediction.md)**

:   CrysLDNet 把"扩散预训练"从原始晶体特征空间搬到 VAE 学到的平滑潜空间，让 PDDFormer 编码器在 38 万无标注 GNoME 晶体上学到更紧凑、更对称感知的结构语义，下游 JARVIS / MP 性质预测平均比强监督 SOTA 再降 4.26% / 4.90% MAE，且在低数据和实验数据校正场景下优势更大。

**[Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](image_generation/learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)**

:   本文提出 CaDRe，用一个带结构约束的时序 VAE 把"观测变量之间的因果图"与"驱动观测的隐动力过程"放在同一个非参数框架下联合识别，并给出了从时序数据同时恢复二者的可识别性定理，在合成数据上验证理论、在 CESM2 气候数据上得到与领域专家一致的因果图与有竞争力的温度预测精度。

**[Let EEG Models Learn EEG](image_generation/let_eeg_models_learn_eeg.md)**

:   JET 把多通道 EEG 生成重新定义为"在神经流形上的连续轨迹"，用条件流匹配 + 标准 Transformer 直接对原始波形建模，并配三条专门刻画 EEG 频谱/平稳性/统计的结构化约束，在 TUH 三大临床基准上把 TS-FID 较强基线降低 40% 以上。

**[Linearizing Vision Transformer with Test-Time Training](image_generation/linearizing_vision_transformer_with_test-time_training.md)**

:   作者发现两层 TTT 内模型在结构上等价于 Softmax 注意力（Softmax 可看作两层动态 MLP），由此实现 Q/K/V/MLP 的全权重直接继承，再通过 key Instance Normalization 处理 shift-invariance、depthwise conv on Q/K 补齐 locality，仅 1 小时微调就把 Stable Diffusion 3.5 线性化并加速 1.32×–1.47×。

**[LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](image_generation/lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)**

:   LithoGRPO 把光刻掩模生成建模为以目标版图为条件的 rectified flow，并用 GRPO 强化学习微调，让一次前向就能同时优化 L2/PVB（可微）与 EPE/Shot（不可微）四类光刻指标，配合一个 130×–490× 加速的快速 shot-count 算法，在 LithoBench 上把综合排名从 5.6 拉到 4.3，单样本推理仅 0.1 s。

**[Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](image_generation/local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)**

:   本文提出 LHSD，把 score 模型的对数密度 Hessian 做一个 Hill 型谱滤波只保留近零特征值来数切空间维数，再用 Stochastic Lanczos Quadrature 把 $\mathcal{O}(D^3)$ 的代价压到 $\mathcal{O}(D)$，从而在 3072 维图像空间稳定估计局部内禀维度，并用于诊断扩散模型的训练样本记忆化。

**[Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](image_generation/localizing_memorized_regions_in_diffusion_models_via_coordinate-wise_curvature_d.md)**

:   本文把"扩散模型局部记忆"刻画为对数密度在某些坐标上的**方差崩塌（高曲率）**，并用"条件模型 − 欠拟合基线（无条件模型或早期 checkpoint）"的**坐标级曲率差**，把单纯由数据流形固有低方差引起的"伪记忆"扣掉，只保留**过拟合驱动的记忆区域**，在 Stable Diffusion 的 ground-truth 记忆掩码上把定位 IoU 从 BE 的 0.75 提到约 0.92。

**[MIRO: 多奖励条件预训练同时提升 T2I 质量与效率](image_generation/miro_multi-reward_conditioned_pretraining_improves_t2i_quality_and_efficiency.md)**

:   MIRO 把"对齐"从 RLHF 后训练阶段直接塞回预训练：给每张训练图打 7 个奖励分（美学、用户偏好、文图对齐、视觉推理、科学正确性等），让 Flow Matching 模型学习 $p(x|c, s)$，推理时通过多奖励 CFG 指向高奖励区域，0.36B 参数即在 GenEval 上超过 12B 的 FLUX-dev，训练算力少 370×，单样本推理质量超过 baseline 跑 128 次。

**[OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](image_generation/occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)**

:   针对布局到图像生成在重叠区域出现纹理纠缠和层级混乱的问题，作者构建了带显式 Z-order 与 amodal 标注的大规模数据集 SA-Z，并提出 OcclusionFormer：通过实例解耦 + 体渲染显式建模遮挡优先级，再用查询对齐损失强化空间一致性，在 OverLayBench 复杂子集与自建 SA-Z Eval 上的遮挡感知指标全面超过 Eligen、Creatilayout、InstanceAssemble 等强基线。

**[Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](image_generation/offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)**

:   OMSD 用"链式条件分解 + 每个 agent 一个条件扩散模型"取代传统离线 MARL 里"各 agent 独立边缘回归"的行为约束,把每个 agent 的策略沿着前缀 agent 已经选定的动作做条件正则,从而避免多模态联合行为分布下"边缘对齐但联合错位"的 OOD 失配,在 MPE / MaMuJoCo 多个数据集上把平均回报刷到现有 SOTA 的 +33% ~ +74%。

**[Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](image_generation/offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)**

:   本文针对 rectified flow（RF）类文生图模型，提出 PNAPO——一种把"生成时用的先验噪声"和"赢者/输者图片"一起保存为六元组的离线偏好优化框架，配合 RF 直线轨迹假设做轨迹估计和动态正则系数调度，相比 Diffusion-DPO 在 SD3-M/FLUX 上同时提点又把训练算力降到 1/12。

**[OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](image_generation/omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)**

:   OmniAID 用一个"语义专家 + 通用伪影专家"的解耦 MoE 架构，在 CLIP-ViT 注意力权重 SVD 出的低秩残差子空间里分别学习"画了什么会露馅"和"怎么画都会露馅"两类伪造线索，再配上新的现代化数据集 Mirage，在 GenImage / Chameleon / Mirage-Test 三套基准上把通用 AIGI 检测平均准确率推到 95.9% / 91.4% / 88.4%。

**[OMP: One-step Meanflow Policy with Directional Alignment](image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)**

:   本文针对将 MeanFlow 范式直接搬到机器人操作时暴露出的三个理论病灶（频谱偏差、低速区梯度饥饿、嵌套 JVP 内存爆炸），提出 OMP：用一项 cosine-style 方向对齐损失把预测平均速度与真实平均速度方向"锁死"，再用有限差分 DDE 近似 Jacobian-Vector Product 解耦前后向，让单步（NFE=1）生成策略在 Adroit/Meta-World 上以 6.8ms 级延迟做到比 MP1 平均高 3.4%、在 Meta-World Very Hard 任务高 10.6% 的成功率。

**[Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](image_generation/order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)**

:   本文从扩散模型的频谱偏置出发，理论证明扩散生成区域的局部 Gibbs 能量必然低于真实成像区域，据此构造 LAD（Local Adjacency Discrepancy）能量图作为内禀取证指纹，再用一个轻量适配器把 LAD 线索注入 SAM 完成像素级伪造定位，并配套 EditStream 多智能体自动从 HuggingFace 拉取最新编辑模型不断刷新训练数据，在 7 个 AI 编辑数据集上把平均 IoU 从前 SOTA 的 ~0.25 拉到 0.46。

**[Orthogonal Concept Erasure for Diffusion Models](image_generation/orthogonal_concept_erasure_for_diffusion_models.md)**

:   把 T2I 扩散模型里"加性参数编辑"的概念擦除（UCE/SPEED 等）改写成"层级正交旋转 $W^* = QW$"的乘性更新，并配上一个子空间级别的擦除目标，用 Procrustes 闭式解一次性算出 $Q$，4.3 秒擦掉 100 个名人概念，且对非目标概念几乎零损伤。

**[Pareto-Guided Optimal Transport for Multi-Reward Alignment](image_generation/pareto-guided_optimal_transport_for_multi-reward_alignment.md)**

:   PG-OT 把「多奖励文生图对齐」从「加权全局求和」改成「为每个 prompt 单独构造 Pareto 前沿、用 Sinkhorn 最优传输把被支配样本传到前沿」，并引入 Joint Domination Rate / Joint Collapse Rate 两个新指标暴露平均值掩盖的奖励 hacking，在 Parti-Prompts 上 JDR₂ 47.98% 比强基线提升 11%，人评胜率近 80%。

**[Path-Coupled Bellman Flows for Distributional Reinforcement Learning](image_generation/path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)**

:   把分布式 Bellman 方程的"仿射搬运"几何性显式编织进 flow matching 的路径里：用同一份基础噪声同时驱动当前态与后继态的两条路径，再用 $\lambda$ 控制变量在偏差与方差之间换挡，从而得到一个对源分布相容、对 Bellman 端点相容、又稳定的分布式 critic。

**[PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](image_generation/physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)**

:   把"造可交互 3D 物体"重新理解成"先做物理规划、再做物理生成"的两阶段问题——VLM 充当物理建筑师生成包含层级关系、材料、运动学约束的 "Hierarchical Physical Blueprint"，扩散模型再用 KineVoxel Injection 把铰接参数和几何 voxel 协同去噪，配合 150k 资产、四层标注的 PhysDB 数据集，首次实现单视图到"可在物理引擎里抓握、推动、铰接"的 3D 资产生成。

**[Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost](image_generation/position_adopting_ai_in_practice_does_not_guarantee_the_productivity_boost.md)**

:   本文是一篇立场论文，主张"组织引入 AI 并不自动等于生产力提升"，识别出五个被传统经济模型忽略的人与环境调节因子（人员组成、个体基线能力、学习曲线、公平使用激励、目标灵活性），并在 Gries-Naudé (2022) 偏均衡模型上加入组织有效性 $\Omega$、能力调整 $\phi(z,\kappa_i)$、学习曲线 $\lambda_i(\tau)$、有效自动化阈值 $\tilde N_{IT}$ 四类修正项，得到能描述"为什么同样投 AI 不同组织产出差距巨大"的修订生产函数。

**[Position: AI Evaluations Should be Grounded on a Theory of Capability](image_generation/position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)**

:   作者主张"benchmark 分数 = 能力"是一种**隐式推断**而非直接测量，呼吁把 AI 评测显式建模成统计推断任务，并借鉴心理测量学（CTT/IRT/CDM/BNSM）四种能力理论作为模板，给出一张"Evaluation Card"让评测者自证假设。

**[Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](image_generation/principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)**

:   GCPO 把 GRPO 在 flow matching 后训练里"每一步都用同一个最终 reward 当 advantage"的步级优化改成"块级"——按 flow matching 自身的时间动态 $L1_{rel}(x,t)$ 自适应地把连续若干步聚成 chunk，用规范化的 chunk-level 重要性比 $r^i_j$ 做策略更新，从而平滑掉"最终好≠每步好"造成的错误梯度，在 HPSv3/ImageReward/GenEval/DPG 上相对 GRPO 取得最高 43% 的相对增益。

**[Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](image_generation/q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)**

:   本文首次为基于 DiT 的真实图像超分（Real-ISR）设计了 PTQ 框架 Q-DiT4SR，通过「全局低秩 + 局部分块 rank-1」的层级 SVD 分解保留高频细节，并基于率失真理论提出无需校准数据的层间权重位宽分配（VaSMP）与基于动态规划的时间步激活位宽调度（VaTMP），在 W4A6 / W4A4 极低位设置下达到 SOTA，并将模型压缩 5.8× / 计算量减少 6.14×。

**[Quantifying Error Propagation and Model Collapse in Diffusion Models](image_generation/quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)**

:   本文在 score-based 扩散模型上对"用合成数据递归训练导致 model collapse"这一现象给出第一套配对的上下界：单代散度 $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$，多代累积散度 $D_N$ 是过去各代 score 误差能量按 $(1-\alpha)^{2m}$ 几何衰减的加权和，从而把"加新鲜数据能缓解坍塌"这一经验事实化成了精确的衰减律。

**[RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](image_generation/raigen_rare_attribute_identification_in_text-to-image_generative_models.md)**

:   RAIGen 用 Matryoshka 稀疏自编码器把 T2I 扩散模型 bottleneck 表征分解成可解释 neuron，再用"激活稀有度 × CLIP 语义偏离度"组合分数从中挑出"模型内部已编码但生成时几乎不出现"的少数属性 neuron，从而把偏见审计从"已知公平类目"和"显著多数模式"扩展到 label-free 的稀有属性发现。

**[Rao-Blackwellized Score Matching on Manifolds](image_generation/rao-blackwellized_score_matching_on_manifolds.md)**

:   当数据分布落在嵌入流形 $M\subset\mathbb{R}^D$ 上时，环境空间高斯加噪做 DSM 学到的切向目标含有方差以 $d/\sigma^2$ 发散的法向噪声通道；本文证明对最近点投影 $\pi(X)$ 做一次 Rao-Blackwell 条件化即可干净地去掉这个奇异通道，并把剩下的目标精确展开为「内蕴 Riemannian score + $\sigma^2$ 阶 Tweedie 校正 + $\sigma^2$ 阶 Weingarten/Ricci 外蕴曲率校正」。

**[Recovering Hidden Reward in Diffusion-Based Policies](image_generation/recovering_hidden_reward_in_diffusion-based_policies.md)**

:   EnergyFlow 把 diffusion policy 的 score field 显式参数化为一个标量 energy function 的负梯度，论证了 maximum-entropy 最优下 score = 软 Q-函数梯度，从而在不做对抗优化的情况下"白送"一个可用作下游 RL shaping reward 的标量信号，同时保守场约束改善 OOD 泛化。

**[Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](image_generation/restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)**

:   本文指出现有 T2I 扩散蒸馏只做"逐点输出对齐"导致学生模型对初始噪声的敏感性塌缩，提出 GAD：用一对扰动输入下的 JVP（雅可比向量积）有限差分近似，强制学生匹配教师对噪声扰动的方向性响应，从而在不损失保真度的前提下恢复布局可控性与生成多样性。

**[RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models](image_generation/rt-lynx_putting_the_gemm_sparsity_in_a_right_way_for_diffusion_models.md)**

:   作者发现 DiT 的**激活**比**权重**更天然稀疏（每个 token 只激活 5–10% 通道），于是把 2:4 半结构化稀疏从权重侧搬到激活侧，再用 norm 缩放 + LoRA 残差补偿 + 选择性跳层把质量损失补回来，并写了一套把"在线 Top-K 选择 + Sparse GEMM"融合到单 kernel 的 CUDA 流水，在 Qwen-Image / FLUX / Z-Image 上做到线性层平均 1.55× 加速且 FID/IR 不退化。

**[SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](image_generation/saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)**

:   通过在稀疏自编码器（SAE）训练阶段加入监督的"概念—潜变量"指派损失，强制每个目标概念集中到单个神经元（feature centralization），从而把扩散模型的概念擦除从"搜多神经元 + 调强度"的二维超参搜索压成"只调一个 multiplier"，在 UnlearnCanvas 上比 SOTA 的 SAeUron 平均提升 9.22 个点，超参搜索代价降低 96.67%，并对对抗攻击更鲁棒。

**[Saving Foundation Flow-Matching Priors for Inverse Problems](image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)**

:   针对 Stable Diffusion / Flux 这类基础流匹配模型在求解逆问题上明显逊于领域专用先验甚至未训练先验的现象，作者提出 FMPlug：用一个由近似样本指导、时间可学习的 warm-start 加上锐利高斯壳层约束，把基础 FM 的潜变量塞回它真正"懂"的薄壳上，从而显著恢复其作为逆问题先验的能力。

**[Scalable GANs with Transformers](image_generation/scalable_gans_with_transformers.md)**

:   本文提出 GAT（Generative Adversarial Transformers）——一套在 VAE 隐空间上用纯 Transformer 生成器与判别器搭起来的可扩展 GAN 框架，通过多层级噪声扰动监督（MNG）激活早期生成器层、并用宽度感知的学习率缩放稳定大模型训练，使 GAT-XL/2 在 ImageNet-256 类条件生成上仅训练 60 epoch 就拿到 FID 2.18 的单步生成 SOTA，比同等规模 1-NFE diffusion/flow baseline 少用 4× epoch。

**[SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](image_generation/scenesmith_agentic_generation_of_simulation-ready_indoor_scenes.md)**

:   SceneSmith 用 designer-critic-orchestrator 三角 VLM agent 在「布局→家具→小物件」的层级树上逐层构建室内场景，并把 text-to-3D 生成、铰接物体检索与物理属性估计深度耦合到 agent 工具链中，从单条自然语言提示直接产出"可直接喂给物理仿真器"的稠密、可操作环境，每个房间平均 71 个物体（基线只有 11–23 个），物体间碰撞率 <2%、重力下稳定率 96%，远超此前所有方法。

**[Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning](image_generation/self-prompting_diffusion_transformer_for_open-vocabulary_scene_text_editing_via_.md)**

:   本文提出一种基于 FLUX-Fill (MM-DiT) 的自提示场景文字编辑方法：直接从原图裁出风格 prompt、用 Pillow 渲染出 glyph prompt，两者与 masked image 沿通道拼接后送入扩散 backbone，再用 4000 张 Nano Banana Pro 生成的高质量配对图做 cooldown 训练，从而在 13 种语言上同时实现开放词表与原始风格一致的文字替换。

**[Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](image_generation/semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)**

:   SATA 用 MLLM 生成的关节语义标签做 FiLM 风格的特征调制，配合空间-时间交错的图自编码器，把任意骨架拓扑的 BVH 动作压到一个共享潜空间，实现高保真重建以及无配对数据的零样本跨物种动作重定向。

**[Semantic Granularity Navigation in Image Editing](image_generation/semantic_granularity_navigation_in_image_editing.md)**

:   NaviEdit 把 diffusion/flow 编辑器中"模型尺度坐标 = 编辑进度时钟"的隐式耦合拆开，在固定 step budget 下用一个训练无关的推理时控制器把算力集中在一个有效尺度窗口的密度上而非把范围扩到高噪声区，从而在 PIE-Bench / ImgEdit-Bench / 多种 flow backbone 上同时改善背景保真和语义一致性。

**[Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](image_generation/shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)**

:   针对 FLUX.1 Kontext 这类基于 Rectified Flow Matching 的 MMDiT 编辑模型在多实例同时编辑下"属性串味"的痼疾，本文提出 Instance-Disentangled Attention（IDAttn）：通过对 joint attention 加结构化掩码，把每条编辑指令绑定到对应的 bounding box，再配合分层 disentanglement/harmonization 调度和高效多 prompt 独立编码，单次前向就能完成 N 条互不干扰的编辑，并在自家提出的 Infographic 文本编辑 benchmark 上显著优于多轮和拼接式 baseline。

**[Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures](image_generation/simple_approximation_and_derivative_free_inference-time_scaling_for_diffusion_mo.md)**

:   作者把扩散模型的推理时 reward 引导从"粒子空间 SMC + 高阶导数"升级为"路径空间 SMC + Girsanov 似然比"，得到 URGE 算法：每条轨迹只需对 guidance $G$ 做一阶梯度并累加一个简单的 Itô 项当权重，完全不需要 reward $r$ 的导数 / Hessian / score 估计，在 GMM、逆问题和文生图三类任务上都打平或优于 FK-Corrector / AFDPS / FK-Steering。

**[Skipping the Zeros in Diffusion Models for Sparse Data Generation](image_generation/skipping_the_zeros_in_diffusion_models_for_sparse_data_generation.md)**

:   SED 把扩散模型从"对所有维度做全密集去噪"改成"只在非零维度上跑扩散+自回归解码维度-值对"，让计算量从随维度线性增长变成几乎随非零数恒定，同时严格保留科学数据中"显式零"这一语义信息。

**[SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](image_generation/spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)**

:   作者指出 MLLM 类编辑奖励模型存在"注意力坍缩"问题——评分时不去比较原图与编辑后图、而是塌缩到 sink token 上做盲判，进而提出 SpatialReward：先让 8B 模型预测编辑区域的边界框、再以这些 box token 为锚做交错式跨图推理；配上一个 260K 样本的空间感知数据集和 GRPO 两阶段训练后，在三个 reward benchmark 上 SOTA，并把 OmniGen2 的 GEdit-Bench 分数拉升 +0.90（是 GPT-4.1 提升的两倍）。

**[Spectral Guidance for Flexible and Efficient Control of Diffusion Models](image_generation/spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)**

:   本文提出 Spectral Guidance：通过自监督学习扩散过程条件期望算子的左奇异函数，把任意引导信号（标签 / CLIP / mask）投影到这组与扩散动力学对齐的谱基上，绕开 denoiser 反向传播，在 CIFAR-10 上较最强 training-free 基线提升 37 个百分点准确率且采样快 4 倍。

**[Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](image_generation/speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)**

:   本文发现 Speculative Jacobi Decoding (SJD) 在自回归视觉生成中加速有限的根因是连续迭代之间 draft token 的独立采样导致 collision 概率几乎为零；只需把独立采样换成 Maximal/Gumbel Coupling（一行修改、零额外训练），就能把图像生成最高加速到 $4.2\times$、视频生成 $13.6\times$，并严格保持输出分布与原 AR 解码一致。

**[Stable Velocity: A Variance Perspective on Flow Matching](image_generation/stable_velocity_a_variance_perspective_on_flow_matching.md)**

:   本文从"条件速度方差"这一被忽视的视角重新审视 flow matching，发现训练轨迹天然分裂为靠近先验的高方差区和靠近数据的低方差区，并据此提出统一框架 Stable Velocity，含一个无偏的多样本方差缩减损失 StableVM、一个只在低方差区启用 REPA 的 VA-REPA，以及一个利用低方差区闭式解的无微调采样加速器 StableVS，在 ImageNet 256 与 SD3.5/Flux/Qwen-Image/Wan2.2 上取得训练效率提升与 >2× 采样加速。

**[Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](image_generation/stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)**

:   提出 MAP-RPS 两阶段框架：先用扩散模型的 score 做 MAP 估计逼近 MMSE 解（低失真起点），再把 MAP 结果 re-noise 到时刻 $t_0$ 后做后验采样（沿 D-P 曲线滑向高感知质量），单一预训练扩散模型就能在推理时灵活遍历 distortion-perception trade-off，并扩展到 latent diffusion 后在 MS-COCO 上多任务 SOTA。

**[STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](image_generation/stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)**

:   本文把 T2I 模型的整个去噪轨迹本身当成 VLM 红队攻击的"攻击面"，用一个 high-level prompt editor + low-level GRPO 微调 rectified-flow 模型的分层 RL 框架（STARE），不仅把 attack success rate 比 SOTA 提升 68%，更揭示了一个全新现象——Optimization-Induced Phase Alignment：对抗优化会自动把"概念性毒性"绑到去噪早期、"细节性毒性"绑到后期，从而把混沌的毒性形成过程变成几个可预测的"漏洞时间窗"。

**[Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](image_generation/support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)**

:   SPADE 用一个条件扩散模型替代传统回归代理来建模 $p(y\mid\boldsymbol{x})$，并通过"均值/排序校准"+"kNN 支撑度正则（均值收缩 + 方差膨胀）"把数据先验隐式注入到代理里，使离线黑盒优化在 Design-Bench 和 LLM 数据混合任务上稳定达到 SOTA。

**[SURF: Separation via Unsupervised Remixing Flow](image_generation/surf_separation_via_unsupervised_remixing_flow.md)**

:   SURF 把监督流匹配 FLOSS 与无监督的 ReMixIT / Self-Remixing 教师-学生重混合训练拼到一起，让一个生成式 flow matching 分离器**完全从混合观测**（没有任何干净源样本）训练出来，在 MNIST/CIFAR10 图像分离和 LibriSpeech / FUSS 音频分离上几乎追平有监督 flow 的指标，刷新无监督 SOTA。

**[SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](image_generation/surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)**

:   SURGE 把扩散代理模型的引导采样视为路径测度上的有偏分布，用 Girsanov 公式计算重要性权重做 SMC 重采样，从而在不重新训练、不近似 Doob $h$-变换的前提下，得到无近似偏差的扩散代理数据同化滤波器，在 Lorenz、Navier-Stokes 和 SEVIR 天气预报上一致超越 BPF/EnKF/SDA/FlowDAS。

**[TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling](image_generation/tag_tangential_amplifying_guidance_for_hallucination-resistant_sampling.md)**

:   TAG 把每一步扩散更新沿当前潜变量方向分解为"径向 + 切向"两个分量，只对切向分量额外乘一个 $\eta \ge 1$ 的放大系数，从一阶 Taylor 展开上证明这等价于单调提升对数似然增益，从而把样本拉向数据流形高密度区，几乎零额外算力地缓解扩散模型的语义幻觉。

**[The Coupling Within: Flow Matching via Distilled Normalizing Flows](image_generation/the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)**

:   本文提出 NFM（Normalized Flow Matching），用预训练 TarFlow 这种自回归归一化流（NF）产生的"准确定性 data→noise 双射"作为 Flow Matching 的噪声-数据配对，从而把 FM 收敛速度、少步数 FID 同时拉到新的水平，并反过来比当老师的 NF 推理快若干个数量级。

**[Threshold-Guided Optimization for Visual Generative Models](image_generation/threshold-guided_optimization_for_visual_generative_models.md)**

:   作者把 DPO 的成对偏好假设拆掉，证明 KL 正则化最优策略本质上是把每个样本的 reward 与一个无法计算的实例相关基线 $\tau^*(x)=\beta\log Z(x)$ 比较，于是用从分数分位数估出的全局阈值 $\tau$ 替代它，再加一个与 $|s-\tau|$ 成正比的置信度权重，让扩散模型和 MaskGIT 在仅有标量打分（无成对偏好）时也能稳定对齐，并在五个 reward model 三个测试集上一致优于 Diffusion-DPO / KTO / DSPO。

**[Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](image_generation/transferable_multi-bit_watermarking_across_frozen_diffusion_models_via_latent_co.md)**

:   DiffMark 把一个学到的潜空间扰动 $\delta$ 在冻结扩散模型的每一步去噪中持续注入，让水印信号在终态潜变量 $z_0$ 上累积，并借助 Latent Consistency Model 作为可微训练桥绕过 50 步 DDIM 的反向传播，实现单次前向 16.4 ms 解出 64 bit、跨模型即插即用且无需重训的水印方案。

**[统一不同生成顺序的掩码扩散模型](image_generation/unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)**

:   提出统一框架 OeMDM 和学习型版本 LoMDM——通过显式建模"速度"（生成优先级）将随机掩码、自回归、块扩散模型统一在一个 NELBO 下，实现从零开始联合学习生成顺序和扩散骨干。

**[扩散模型中的遗忘：基于 KL 散度和似然约束的统一框架](image_generation/unlearning_in_diffusion_models_a_unified_framework_with_kl_divergence_and_likeli.md)**

:   本文提出统一的约束优化框架——将扩散模型中的机器遗忘问题形式化为最小化与预训练模型的偏差，同时受约束于明确的与遗忘分布的分离条件，通过三种约束形式（反向 KL、前向 KL、似然约束）统一处理概念遗忘和数据遗忘，并证明强对偶性。

**[ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](image_generation/viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)**

:   通过离散扩散模型和视觉 token 化，将多视图生成建模为离散序列预测任务——利用简单的随机掩码策略结合自注意力自然地实现跨视图一致性，显著超越连续扩散方法。

**[Visual Implicit Autoregressive Modeling](image_generation/visual_implicit_autoregressive_modeling.md)**

:   本文把 Deep Equilibrium（DEQ）隐式不动点层嵌进 VAR 的 next-scale 自回归框架，用 Jacobian-Free Backpropagation 实现常数显存训练，把 VAR-d30 的 20 亿参数压到 7.7 亿，同时在推理时把每个 scale 的迭代次数变成"可调旋钮"——在 ImageNet-256 上 FID 2.16/sFID 8.07 不变的同时，4090 单卡峰值显存从 19.24GB 降到 8.53GB、吞吐从 15.16 提到 32.08 img/s。

**[Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](image_generation/watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)**

:   本文揭示扩散模型里一直被忽视的"时间步嵌入"其实是一条尚未被占用的信息侧信道——通过把训练时的 timestep 范围扩展到一个"影子区间"（shadow timestep）并把另一个数据分布绑定到该区间，可以在不改变 scheduler 接口的前提下，让同一个 diffusion 模型在显式区间生成正常图、在影子区间生成"隐藏"图，既可做隐蔽后门攻击也可做模型水印验证；同时给出基于正弦位置编码的互相干（mutual coherence）理论分析，解释为什么两个不相交区间能携带独立信息。

**[Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](image_generation/weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)**

:   论文发现低保真或领域不匹配的扩散模型先验在信息丰富的逆问题中仍能取得强劲性能——通过贝叶斯一致性理论和局部相关性分析解释了这一看似矛盾的现象，并给出何时弱先验有效的明确条件。

**[When Preference Labels Fall Short: Aligning Diffusion Models from Real Data](image_generation/when_preference_labels_fall_short_aligning_diffusion_models_from_real_data.md)**

:   这篇论文认为由生成图像组成的偏好标签容易把模型带向“相对更好但仍有缺陷”的样本，提出用真实图像及其可控退化版本自动构造偏好信号，在只用 512 对样本的情况下对齐 SD-1.5 和 SD-3.5-M，并取得接近或补充 Diffusion-DPO / FlowGRPO 的效果。

**[WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](image_generation/wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)**

:   WISE 构建了一个包含 1000 条知识密集 prompt 的文本到图像评测基准，用文化常识、时空推理和自然科学知识检验模型是否能把隐含语义转化成正确视觉内容，并发现现有 T2I 与统一多模态模型在世界知识生成上仍有明显短板。

**[You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models](image_generation/you_dont_need_all_that_attention_surgical_memorization_mitigation_in_text-to-ima.md)**

:   本文提出 GUARD，一个推理时的文生图扩散模型记忆缓解框架，通过对标准 classifier-free guidance 加入“远离原始记忆提示”的 repulsion 和“靠近安全条件预测”的 attraction，并用动态 cross-attention spike 检测与衰减实例化 positive target，在降低训练图像复现的同时尽量保持图像质量和 prompt 对齐。

**[Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](image_generation/zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)**

:   本文提出一种带方差缩减的零阶 Langevin 采样方法，用间歇性大 batch 估计和递推式小 batch 更新替代每步 $O(d)$ 次函数查询，并把它扩展为 ZO-APMC，用预训练 score-based prior 在只有前向模型查询的黑盒逆问题中做有收敛保证的后验采样。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](multimodal_vlm/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)**

:   3ViewSense 认为 VLM 空间推理的瓶颈不是视觉特征不够或语言推理太弱，而是缺少稳定的三维中间表示，因此让模型先从单张图像诱导前视图、左视图、俯视图，再基于这些正交视图推理，在遮挡计数和视角一致空间推理上显著优于同规模 VLM。

**[Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](multimodal_vlm/active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)**

:   本文把 VLM 空间推理从“被动看完所有视角再回答”改造成“按问题主动取景、更新认知地图、用可执行空间断言验证推理”的 agentic 流程，并用密集奖励微调 Qwen2.5-VL-3B，在 MindCube-Tiny 上取得 80.5% overall accuracy，尤其把 Rotation 子集提升到 85.0%。

**[Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](multimodal_vlm/adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)**

:   这篇论文提出 RUDDER，在 LVLM 的 prefill 阶段从残差更新中提取每样本视觉证据方向，并在解码时用 Beta Gate 自适应注入，从而以接近单次前向的开销降低物体幻觉。

**[AgentHijack: Benchmarking Computer Use Agent Robustness to Common Environment Corruptions](multimodal_vlm/agenthijack_benchmarking_computer_use_agent_robustness_to_common_environment_cor.md)**

:   本文提出 AgentHijack，用 9 类可配置的日常环境破坏评测 computer-use Agent 鲁棒性，并进一步用 DA-GRPO 强化 grounding、引入 onlooker 进行行为总结与环境检查，使 UI-TARS-1.5-7B 在平均成功率上从 18.74% 提升到 22.89%。

**[Alterbute: Editing Intrinsic Attributes of Objects in Images](multimodal_vlm/alterbute_editing_intrinsic_attributes_of_objects_in_images.md)**

:   Alterbute 用 VLM 自动挖掘 Visual Named Entity 身份簇，并在扩散模型中联合条件化身份参考、属性文本、背景和 mask，从而统一编辑物体颜色、纹理、材质和形状，同时尽量保持物体身份与场景上下文。

**[Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](multimodal_vlm/any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)**

:   作者通过 pilot study 发现"显式把视觉提升到点云、再与 2D patch 融合"是 VLA 注入 3D 信息的最有效方式；为了解决 3D 数据稀缺和不同点云源（仿真/传感器/单目估计）的域差异，提出 Any3D-VLA：用 hybrid point cloud training 学到 source-agnostic 的几何表示，在真实抓取任务上 zero-shot 比最强 baseline 提升 29.2%（62.5% vs 33.3%）。

**[AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning](multimodal_vlm/aoept_breaking_the_implicit_modality-reduction_bottleneck_in_modality-missing_pr.md)**

:   AOEPT指出现有缺失模态 prompt tuning 会把多模态 Transformer 的推理范围压缩到可见模态子空间，并用从训练集蒸馏出的模态上下文提示为缺失模态补回可检索的隐式信息源，在多数据集、多缺失率和多 backbone 上稳定优于现有方法。

**[Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination](multimodal_vlm/are_vlms_seeing_or_just_saying_uncovering_the_illusion_of_visual_re-examination.md)**

:   这篇论文提出 VisualSwap 和 VS-Bench，通过在 VLM 自称“再看一眼图像”之后替换图像来检验真实视觉重检能力，发现当前推理型 VLM 往往沿着旧文本惯性继续生成，显式用户多轮指令或增强视觉注意力才能显著恢复 grounding。

**[Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](multimodal_vlm/bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)**

:   本文把 VLM 的输出强制拆成 `<recognition>` 感知块和 `<think>` 推理块，再用一个"蒙眼"文本推理代理（拿不到图，只看 VLM 写下的感知文字）能不能答对题作为感知奖励 $R_P$，配上结构化语言验证 SVV 作为结果奖励 $R_O$；MoCA 用 $R_P$ 当门控做模态级信用分配，让 7B 模型在 9 个 perception/reasoning/rich-modality benchmark 上同时提升，在多个指标上超过 GPT-4o。

**[Benchmarking and Enhancing VLM for Compressed Image Understanding](multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)**

:   本文构建了首个评估 VLM 对压缩图像理解能力的大规模 benchmark（11 种编解码器、9 个 VLM、100 万+ 压缩图像），将性能下降分解为不可修复的"信息差距"和可弥补的"泛化差距"，并提出一个轻量级条件视觉编码器适配器，通过编解码器类型和压缩级别的条件嵌入 + 蒸馏训练，在不同编码器和比特率下将 VLM 性能提升 10%–30%。

**[Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated](multimodal_vlm/benchmarks_for_vision-language_models_in_urban_perception_should_be_reliability-.md)**

:   本文提出 VLM 城市感知评估应具备"可靠性感知"和"可协商"两大属性，通过 100 张蒙特利尔街景图像、12 名社区标注者、30 个维度的基准测试，揭示了模型对齐度与标注者一致性正相关，且在主观评价维度上模型与人类存在系统性分布偏差。

**[Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling](multimodal_vlm/beyond_vlm-based_rewards_diffusion-native_latent_reward_modeling.md)**

:   提出 DiNa-LRM，将偏好学习直接建立在扩散模型的噪声潜空间上，通过噪声校准的 Thurstone 似然和推理时多噪声集成，以远低于 VLM 奖励模型的计算开销实现接近 SOTA 的偏好预测精度。

**[Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](multimodal_vlm/breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)**

:   针对统一多模态模型 (unified model) 在 anything-to-image (X2I) 任务上的"理解–生成 gap"（看得懂但生不出），本文提出 Self-Adaptive Interleaved Reasoner：用一个 hierarchical 数据合成 pipeline 在直接生成 / 自我反思 / 多步规划三种模式间分流 5 万条样本，再用 SFT + GRPO 训练并配上 step-wise 推理奖励和 intra-group 复杂度惩罚，让 Emu3.5 在 KRIS-Bench / OmniContext 上超越 GPT-4o、Gemini 2.5 Flash 等闭源模型。

**[Calibrated Multimodal Representation Learning with Missing Modalities](multimodal_vlm/calibrated_multimodal_representation_learning_with_missing_modalities.md)**

:   针对"想用 V-T、A-T 等部分模态数据训练统一多模态对齐"这种现实场景，本文用奇异值扰动给出"缺失模态会导致 anchor shift"的理论上下界，并提出 CalMRL：用概率 PCA 风格的生成模型对缺失模态在表示层做闭式 EM 插补，再把观测 + 插补一起喂给 GRAM/PMRL 的 SVD 对齐目标，在 VAST 之上把跨模态平均 Recall@1 从 44.8 推到 54.2 (+9.4)。

**[Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](multimodal_vlm/capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)**

:   提出 GIFT 方法，通过追踪 VLM 在理解用户查询时视觉注意力的正向变化（"注视转移"）构建视觉显著性图，并在解码阶段同时增强视觉和查询 token 的注意力以保持跨模态融合平衡，在 CHAIR 上最高提升 20.7%，且仅增加 1.13× 延迟。

**[Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](multimodal_vlm/certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)**

:   本文把随机平滑（RS）从"只支持单一连续或离散输入"扩展到"离散 token + 连续图像"的混合扰动场景，通过一个混合 Neyman–Pearson 分析得到一个**一维、连续、可逆**的似然比 CDF，从而把原本组合爆炸的离散 knapsack 问题变成可解的根求解问题，并在 LLaVA-Guard 多模态安全过滤上给出首个针对"图文联合不安全"的 model-agnostic 证书。

**[CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models](multimodal_vlm/cg-mllm_captioning_and_generating_3d_content_via_multi-modal_large_language_mode.md)**

:   CG-MLLM 提出了一种基于 Mixture-of-Transformer 的多模态大语言模型，通过 TokenAR（逐token自回归）和 BlockAR（块级并行）双 Transformer 架构，结合预训练 VLM 骨干与 3D VAE 潜空间，首次实现在单一 MLLM 框架内端到端进行高分辨率 3D 内容生成与 3D 字幕理解，在 MLLM 类 3D 生成方法中达到 SOTA。

**[Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](multimodal_vlm/circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)**

:   提出 Circle-RoPE，将图像 token 的 2D 坐标映射到与文本位置轴正交的环面上，形成锥体几何结构，使每个文本 token 到所有图像 token 的 RoPE 距离相等（PTD=0），消除跨模态伪位置偏置，同时通过逐层交替编码（AGE）保留图像内部空间结构。

**[CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](multimodal_vlm/clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)**

:   发现 CLIP 中指代区域的视觉 token 与 [EOS] 文本 token 呈反直觉的低相似度现象（similarity reversal），据此提出 LiteLVLM——一种免训练的文本引导视觉 token 剪枝方法，在裁剪 66.7% token 后仍保留 90.3% 原始像素定位性能，同时实现 22% 推理加速和 2.3× 显存节省。

**[Conditional Diffusion Sampling](multimodal_vlm/conditional_diffusion_sampling.md)**

:   本文提出 Conditional Diffusion Sampling（CDS）：通过推导一类条件随机插值（conditional interpolants），得到一个对未归一化目标分布的**精确闭式 SDE**（不需要神经网络拟合），再用 Parallel Tempering 高效采样这个 SDE 的初始分布——把 PT 的全局探索能力和扩散过程的局部细化能力拼起来，在 8 个目标分布、4 类任务上以更少的密度评估次数同时击败传统 MCMC、训练自由 MCMC 和神经采样器。

**[Contextualized Visual Personalization in Vision-Language Models](multimodal_vlm/contextualized_visual_personalization_in_vision-language_models.md)**

:   CoViP 把"基于用户历史经验做视觉个性化"这一开放任务，统一收敛到"个性化图像字幕"这个共享底层过程，通过可验证奖励的 RL 后训练 + 推理时的字幕增强生成（CAG），让 VLM 在交错图文上下文里真正"看图说人话"，并配套设计了能排除文本捷径的 MCQA 诊断基准。

**[CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)**

:   CVSearch 提出一个无需训练的"评估-再搜索"认知框架：先用视觉专家（SAM 3）做快速定位，专家失败时再触发语义引导的自适应分块 + 自底向上搜索作为兜底，在 V*Bench、HR-Bench 等高分辨率基准上同时拿到精度与效率的 SOTA。

**[CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict](multimodal_vlm/cyberjurors_a_multi-agent_simulation_task_for_e-commerce_disputes_verdict.md)**

:   作者把电商平台"众包陪审团"的真实裁决任务形式化为 EDV (E-commerce Dispute Verdicts)，构建首个含 17 名陪审员投票真值的多模态基准 VerdictBench（6000 案、文/图/视频/多轮），并提出 CyberJurors——用四阶段的 Individual Verdict Chain-of-Thought (IV-CoT) 做单陪审员细粒度证据定位，用 Jury Consensus Verdict (JCV) 借鉴 Stare Decisis 引入历史判例做集体共识；在 VerdictBench 上 Acc 较最强 LLM/MLLM/法庭仿真器分别 +9.48%/+9.38%/+6.19%。

**[DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction](multimodal_vlm/dcer_dual-stage_compression_and_energy-based_reconstruction.md)**

:   DCER 把"模态内频域压缩 + 跨模态 bottleneck token"作为统一的鲁棒融合管道，并用一个学习的能量函数对缺失模态做梯度下降式重建，同时把最终能量值当作内蕴的不确定度，在 MOSI/MOSEI/SIMS 上同时刷新 SOTA。

**[Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models](multimodal_vlm/debate_with_images_detecting_deceptive_behaviors_in_multimodal_large_language_mo.md)**

:   作者构建了首个面向 MLLM 欺骗行为的多模态基准 MM-DeceptionBench（六类、1013 个真实案例），并提出"带图辩论 (Debate with Images)"框架——两个 MLLM 智能体在多轮辩论中被强制用可视化操作回切原图取证，再由 judge 判定是否欺骗，使与人类一致性的 Cohen's kappa 相对 MLLM-as-a-judge 提升最高 1.5×、准确率提升最高 1.25×。

**[Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging](multimodal_vlm/decentralized_instruction_tuning_conflict-aware_splitting_and_weight_merging.md)**

:   作者从"merge-ready 平坦盆地"出发给权重合并写了一套局部二次理论：合并增益等于曲率加权的 checkpoint 方差，PCA 沿梯度冲突主方向切分能最大化这个增益，并据此提出 MERIT 流水线——按数据集梯度冲突做 PCA 切分、各分支零通讯独立微调、最后一次 token 加权平均，在 Qwen2.5-VL-3B + 136 个 Vision-FLAN 任务上把 8-benchmark 平均从 54.3 提到 57.0。

**[Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](multimodal_vlm/decomposed_on-policy_distillation_for_vision-language_reasoning_steering_gradien.md)**

:   作者把多模态在线蒸馏的 KL 损失沿贝叶斯链拆成"语言先验"和"视觉接地"两个子目标，发现两者梯度近乎正交、标准蒸馏只是被动取平分，提出 Visual Gradient Steering（VGS）主动把更新方向偏向视觉子空间，在 Qwen3-VL 8B→2B/4B 七个多模态推理基准上平均提升 +2.37%/+1.56%。

**[Deep Pre-Alignment for VLMs](multimodal_vlm/deep_pre-alignment_for_vlms.md)**

:   作者把标准 VLM 里"ViT + 轻量 projector"的视觉编码模块整体替换为一个小 VLM（perceiver），让模态对齐这件耗深度的脏活在 upstream 的小 VLM 内部就完成，使下游大 LLM 不必在浅层浪费深度去做模态对齐——4B 模型在 8 个多模态基准上提升 +1.9 点、32B 上提升 +3.0 点，并把语言能力遗忘减少 32.9%，且推理吞吐只下降 2–6%。

**[DenseMLLM: Standard Multimodal LLMs for Dense Prediction](multimodal_vlm/densemllm_standard_multimodal_llms_for_dense_prediction.md)**

:   作者把语义分割、深度估计、指代分割这些密集预测任务直接塞进一个 4B 标准 MLLM（ViT + Projector + LLM），不加任何任务专用 decoder，靠对视觉 token 引入"多标签下一 token 预测"（NTP-M）监督，在 ADE20K 取得 54.2 mIoU、DDAD 取得 87.6 δ1、RefCOCO val 取得 80.7 cIoU，同时通用 VL 指标与 Qwen3-VL-4B 持平。

**[Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs](multimodal_vlm/density-aware_translation_of_spurious_correlations_in_zero-shot_vlms.md)**

:   作者发现 CLIP 嵌入在球壳上呈各向异性椭球分布、伪相关样本扎堆在均值附近，于是提出 DAT：用每个 (类别, 伪属性) 组的参考集估一个局部密度 $D_{y,a}(z)$，再用 $\tilde s_{y,a}(x)=s_{y,a}(x)/(D_{y,a}(z)+\varepsilon)^{\lambda}$ 把原始 cosine 相似度按"样本是否处在该组核心"重缩放，从而在不微调、不改文本端、不需测试时伪属性标签的前提下显著提升 worst-group 准确率。

**[Detached Skip-Links and $R$-Probe: Decoupling Feature Aggregation from Gradient Propagation for MLLM OCR](multimodal_vlm/detached_skip-links_and_r-probe_decoupling_feature_aggregation_from_gradient_pro.md)**

:   针对 MLLM 的 OCR 场景，作者在多层 ViT→LLM 融合架构中给浅层 skip 分支加 stop-gradient（Detached Skip-Links），同时提出用"LLM 自身前 1/4 层"初始化的重建探针 $R$-Probe 来诊断视觉 token 是否真的把细粒度信息送到了语言模型那一侧。

**[Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](multimodal_vlm/dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)**

:   对预条件退火朗之万动力学（PALD）做首个**维度无关**的非渐近收敛分析——把多模态分布采样复杂度从 $\tilde{O}(d/\epsilon^2)$ 缩减到 $\tilde{O}(1/\epsilon^2)$，让扩散类采样算法在高维下从"维度爆炸"中解放。

**[DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement](multimodal_vlm/diva_harnessing_the_representation_divergence_in_unified_multimodal_models_for_m.md)**

:   DIVA 发现统一多模态模型 (UMM) 在中间层会自发把"理解"与"生成"两条信息流解耦，于是显式地把表示因子化为共享与独有两部分，用对比/CLUB 互信息约束实现"共享对齐 + 独有解耦"，在 Show-o/Liquid/Nexus-Gen 上同时提升理解 +7.82% 与生成 +8.46%，无需改架构。

**[ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](multimodal_vlm/ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)**

:   ECG-R1 是首个面向心电图解读的"推理型"医学多模态大模型，通过**协议引导的指令数据合成 + 信号/图像解耦编码 + 交错模态丢弃训练 + 基于诊断证据的过程奖励 RL** 四件套，把心电诊断准确率从此前 SOTA GEM 的 74.7 提升到 80.3，并在任一模态缺失时保持跨模态一致性。

**[Efficient Reasoning with Hidden Thinking](multimodal_vlm/efficient_reasoning_with_hidden_thinking.md)**

:   Heima 把多模态 LLM 的冗长 CoT 每个阶段（summary / caption / reasoning）蒸馏成**一个特殊 thinking token**，让模型在隐空间里"想"，token 数从 100-200 量级降到 13-16 个的同时 zero-shot 准确率反而比 LLaVA-CoT 更稳；配套训练一个 LLM "interpreter"用 thinking token 的 hidden state 重建出文字推理链，从而验证压缩损失的信息论上界。

**[Explaining Is Harder than Predicting Alone: Evaluating Concept-Based Explanations of MLLMs as ICL Visual Classifiers](multimodal_vlm/explaining_is_harder_than_predicting_alone_evaluating_concept-based_explanations.md)**

:   作者用 5 级形式化逐步加严的解释条件（裸分类 → 自然语言解释 → 特征清单 → IF-THEN 知识库 → DL 公理）和一个评 9 个 XAI 指标的 LLM-as-a-judge 流水线，对 4 个 SOTA MLLM 做了 2,080 次 ICL 分类实验，发现"逼模型生成越正式的概念解释，分类准确率反而单调下滑（93.8% → 90.1%）"，但"局部判别性"是唯一与准确率显著相关的解释质量维度。

**[Find, Fix, Reason: Context Repair for Video Reasoning](multimodal_vlm/find_fix_reason_context_repair_for_video_reasoning.md)**

:   本文针对视频推理中"on-policy RL 在能力天花板停滞、off-policy 蒸馏又会熵塌缩"的两难，引入一个冻结的、工具集成的大教师模型在学生 rollout 失败时插入最小化的"证据补丁" (key-frame 区间、错误类型)，让学生在同一问题上重新作答，并把修复后的轨迹通过 chosen-rollout 机制纳入 GRPO 优化。

**[FlowNar: Scalable Streaming Narration for Long-Form Videos](multimodal_vlm/flownar_scalable_streaming_narration_for_long-form_videos.md)**

:   FlowNar 通过"段末清空视觉 KV 缓存 + 用门控线性注意力把历史视觉信息压成定长记忆 token"的组合，让流式视频解说模型在显存与计算上保持常数级开销，可处理 $10\times$ 更长的视频并取得 $3\times$ 吞吐，同时引入自条件评测协议揭示了基线方法在真实部署下被严重高估的现象。

**[Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](multimodal_vlm/focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)**

:   本文提出 **Visual Information Gain (VIG)**——一个基于"有图 vs 无图（用模糊图代替）"困惑度对数比的视觉依赖度指标，从样本和 token 两个粒度量化"这条数据/这个 token 到底用没用到图"，并据此做选择性指令微调：只在高 VIG 的样本和 token 上算 loss，让 LLaVA-1.5-13B 仅用 21% 的有效 token 就全面超过 vanilla 训练，并显著缓解语言偏差与幻觉。

**[FreeRet: MLLMs as Training-Free Retrievers](multimodal_vlm/freeret_mllms_as_training-free_retrievers.md)**

:   FreeRet 提出一个完全不训练的两阶段多模态检索框架：第一阶段绕过 MLLM 最后一层 MLP 并配合受控生成 prompt 抽取语义忠实的 embedding 做候选检索，第二阶段把 reranking 改成多项选择题来规避 LLM 的 framing 偏置；在 MMEB 上比训练了千万级配对数据的检索模型还要强。

**[From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](multimodal_vlm/from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)**

:   本文指出当前 VLM 后训练过度强调"长链推理"而忽视感知瓶颈，把后训练显式拆成"视觉感知 → 文本推理 → 视觉推理"三个独立阶段，并用 RLVR（而非 caption SFT）单独打磨感知，使 Qwen3-VL-8B 在视觉数学和感知 benchmark 上分别相对基线提升约 +5.9% 和 +1.2%，同时把推理 trace 缩短 20.8%。

**[Furina: Fragmented Uncertainty-Driven Refusal Instability Attack](multimodal_vlm/furina_fragmented_uncertainty-driven_refusal_instability_attack.md)**

:   本文先用多指标诊断证明"LLM 的安全决策不是二值阈值，而存在一段拒绝不稳定带"，并发现该带的特征是"外部不确定性升高而内部安全信号反而下降"；据此提出 Furina——一种无需模型特定优化、靠把恶意意图打碎进场景化叙事来强行把输入推入不稳定带的越狱攻击，在 HarmBench 上超越多种强基线。

**[Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](multimodal_vlm/gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)**

:   本文用 Information Bottleneck 视角把量化感知训练 (QAT) 与知识蒸馏统一起来，提出 GRACE 框架（置信度门控解耦蒸馏 + 关系中心化核对齐 + 自适应 IB 控制器），让 INT4 量化的 LLaVA / Qwen-VL 不仅没掉点，反而在多个 benchmark 上超过 BF16 基线，同时实测 3× 吞吐 + 54% 显存节省。

**[CHARM: 用 Multimodal JEPA + 通道描述做时间序列 foundation embedding](multimodal_vlm/giving_sensors_a_voice_multimodal_jepa_for_semantic_time-series_embeddings.md)**

:   CHARM 把通道文本描述（如"温度传感器 °C"）作为 inductive bias 注入时间序列 Transformer，用 JEPA 目标（latent prediction 而非 raw signal reconstruction）训练；得到的 embedding 在 anomaly detection、classification、forecasting 上用 linear probe 就能与 PatchTST/MOMENT/Moirai 等专用模型匹敌，且 channel-permutation 严格等变。

**[Hierarchical Synthetic Tabular Data Generation: A Hybrid Top-Down and Bottom-Up Framework](multimodal_vlm/hierarchical_synthetic_tabular_data_generation_a_hybrid_top-down_and_bottom-up_f.md)**

:   本文提出 H-TDBU 框架：用 LLM 或人工写的规则在 top-down 路径生成"逻辑骨架" $\mathcal{S}$，再用 RandomForest/XGBoost/CTGAN 等轻量 bottom-up 生成器学习"统计纹理" $z$，最后通过条件生成器 $G(z\in\mathcal{Z}\mid\mathcal{S})$ 把两者拼起来并用 TSTR + XModal 反馈循环迭代修正，在弱多模态金融基准上 TSTR AUROC 优于纯神经网络 baseline 同时保持跨模态一致性。

**[Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](multimodal_vlm/hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)**

:   Hyper-ICL 通过将 **CLIP 嵌入提升到双曲空间**形成结构化"超球面锚"，结合**层次感知蒸馏注意力**为多模态 LVLM 上下文学习提供结构先验——在 VQA / Captioning / Caption Editing 等任务上稳定超越传统 demo 选择策略。

**[Immuno-VLM: Immunizing Large Vision-Language Models via Generative Semantic Antibodies for Open-World Trustworthiness](multimodal_vlm/immuno-vlm_immunizing_large_vision-language_models_via_generative_semantic_antib.md)**

:   本文把生物免疫系统中的"阴性选择"原理搬到 CLIP 等 VLM 上：用 LLM 主动幻觉一批"看起来像但不是已知类"的文本描述作为语义抗体，再以一个轻量 adapter 把视觉特征推离这些抗体，从而在不重训骨干的情况下显著降低开放世界场景下的"高自信误判"。

**[ATHA: 通过打破尾部对齐改进 CLIP 在源数据无关跨域小样本上的适配](multimodal_vlm/improving_clip_adaptation_by_breaking_tail_alignment_for_source-free_cross-domai.md)**

:   ATHA 提出在 CLIP 跨域小样本微调中"对齐头部 token、推开尾部 token"的非对称对齐范式——把语义稀薄的 patch 主动从文本嵌入推开,反而能减轻过拟合并把 1-shot 平均精度从 55.92% 推到 58.35%。

**[RESTORE: 通过矫正失真改进视觉 Token 缩减以提升 MLLM 推理效率](multimodal_vlm/improving_visual_token_reduction_via_rectifying_distortions_for_efficient_multim.md)**

:   RESTORE 把现有视觉 Token 缩减(VTR)中被忽略的"位置失真"和"注意力衰减"两个问题摆到台面上,通过给 RoPE 衰减加一个距离感知的反向补偿项,再用兼顾代表性与判别性的 anchor 选择策略改进 token 合并,使得 LLaVA-1.5-7B 在 64 token(约 11% 保留率)下仍能逼近全 token 性能。

**[Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](multimodal_vlm/injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)**

:   本文把 MLLM 的连续值回归在长尾分布下的"回归到均值"问题转化为分布感知的 RL 问题，在 GRPO 框架内用 Concordance Correlation Coefficient (CCC) 作为批次级奖励——既看相关性、又看方差、又看均值——从而显式惩罚预测分布塌缩；在 4 个长尾回归任务、Qwen2.5-VL-3B/7B 上稳压 SFT、SoftLabel、各种 point-wise RL，特别是 medium/few-shot 区域 MAE 大幅下降。

**[Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](multimodal_vlm/instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)**

:   本文发现 MLLM 中 instruction token 的中间层嵌入能天然过滤视觉端引入的误导信息，据此提出训练无关的 InsLen 分数（Calibrated Local Score + Context Consistency Score），在 5 个 MLLM × 4 个基准上把对象幻觉检测的 AUROC 拉高最多 13.81%。

**[VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](multimodal_vlm/interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)**

:   VEENA 用 steering-vector 因果归因框架定位 LVLM 的情感电路——发现其遵循"Adapt（浅层模态对齐）→Aggregate（中层 emotion-specific heads 聚合）→Execute（深层 emotion-general heads + neurons 生成）"三段式机制，进而用"视觉情感增强 + 情感神经元放大"做训练无关推理时干预，显著缓解情感幻觉。

**[iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](multimodal_vlm/ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)**

:   针对“显式视觉 grounding 反而拖累 CoT 推理”这一反直觉现象，作者提出 iVGR——一个双流 GRPO 训练框架，让文本 CoT 和带框 grounded CoT 同时 rollout，并用一致性奖励把高质量 grounded 轨迹的视觉定位能力“内化”进纯文本 CoT，从而在推理时不用输出坐标就能拿到 grounded 推理的收益。

**[Jailbreaking Vision-Language Models Through the Visual Modality](multimodal_vlm/jailbreaking_vision-language_models_through_the_visual_modality.md)**

:   作者提出 4 种只通过视觉输入就能越狱前沿 VLM 的攻击（视觉密码 / 物体替换 / 文本替换 / 视觉类比谜题），在 6 个前沿 VLM 上系统验证了"文本端的安全对齐不会自动迁移到视觉端"，并用 mechanistic 分析揭示了背后的层级机理。

**[LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation](multimodal_vlm/language_bias_in_lvlms_from_in-depth_analysis_to_simple_and_effective_mitigation.md)**

:   本文系统量化 LVLM 训练中的语言偏置——发现 VIT 和 DPO 两个阶段都让 text-only likelihood $\pi(y|x)$ 涨得几乎不输 multimodal likelihood $\pi(y|x,v)$，证明 LVLM 在系统性低估视觉输入；提出 Language Bias Regularization（VIT 阶段惩罚 $|\mathcal{B}|$）和 Language Bias Penalty（DPO 阶段惩罚偏置正向增长），不加任何数据/辅助模型就显著提升 10+ benchmark 性能并降幻觉。

**[Large Vision-Language Models Get Lost in Attention](multimodal_vlm/large_vision-language_models_get_lost_in_attention.md)**

:   本文用"信息复杂度 (eRank) + 子空间支持"的几何信息论框架定量诊断 LVLM 的残差流，发现 Attention 几乎只做子空间内重配置而 FFN 才注入新语义维度；更惊人的是把学习到的 attention 权重换成高斯噪声后多数视觉任务性能不降反升，揭示当代 LVLM 在 visual attention 上严重错配冗余。

**[Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models](multimodal_vlm/layer-specific_fine-tuning_for_improved_negation_handling_in_medical_vision-lang.md)**

:   NAST 用因果追踪 (causal tracing) 算出 CLIP 文本编码器各层对否定理解的因果贡献度 (CTE)，再以这些 CTE 做层级化梯度缩放微调 LoRA，让医学 VLM 在区分"有 / 没有某症状"时的语义敏感度大幅提升，并把肯定-否定准确率差距从 21.6% 缩到 4.2%。

**[Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](multimodal_vlm/learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)**

:   VISTA 把多模态大模型的自我改进训练改造成"难题靠 prefix 重采样补样本、伪正例靠视觉注意力分数 (VAS) 过滤"的两段式 pipeline，在 Qwen2.5-VL-3B 上把数学/医学多模态推理平均提升 +13.66%。

**[Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](multimodal_vlm/learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)**

:   将原图与对比负图拼成共享多图上下文，再用锚定指令告诉模型该看哪张，从而让视觉偏好 DPO 的配分函数自动对齐、跑出理论一致的对比目标，并配合精细编辑生成的硬负样本显著降低 VLM 的多模态幻觉。

**[Learning GUI Grounding with Spatial Reasoning from Visual Feedback](multimodal_vlm/learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)**

:   把 GUI grounding 从「一次性预测坐标」改写成「在屏幕上挪鼠标找目标」的交互式搜索，用一个带轨迹惩罚的稠密奖励 + GRPO 训练 VLM，让模型从渲染出来的光标得到视觉反馈来对齐数字坐标与屏幕位置，仅用 8K 样本就在 ScreenSpot-Pro 上把 GTA1 的 50.1% 提到 58.1%。

**[Left-Right Symmetry Breaking in CLIP-style Vision-Language Models Trained on Synthetic Spatial-Relation Data](multimodal_vlm/left-right_symmetry_breaking_in_clip-style_vision-language_models_trained_on_syn.md)**

:   作者用一个 1D 合成 image-text 测试床端到端训练 CLIP-style Transformer，发现这类模型能学到"左/右"关系并泛化到未见物体对，机制是**位置嵌入与 token 嵌入的交叉项 $EW_{QK}P^T$ 在 vision encoder 注意力 logit 中诱导出一条水平梯度**，打破左右对称；消融该项后左右判别准确率掉到随机水平。

**[Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](multimodal_vlm/less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)**

:   这篇用 70 万次实验跑遍了 16 种量化方法 × 10 种 VLM × 多项可靠性指标，发现量化不是单纯破坏者——它会通过抑制高 rank 低方差的频谱分量，同时提升 calibration、OOD 检测和噪声鲁棒性，但也会放大对协变量偏移和虚假相关的依赖。

**[LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](multimodal_vlm/limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)**

:   作者把"训练阶段就缺模态"的多模态动作质量评估重新建模成"基于 LLM 的条件序列到分数推理"问题，用 prompt + 特殊 token 让 LLM 在没有完整数据监督的情况下补全缺失语义，再配合掩码感知的双路融合抑制幻觉，在三个 AQA 数据集上全面超越依赖完整训练数据的 SOTA。

**[CSMR (Look on Demand): A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning](multimodal_vlm/look_on_demand_a_cognitive_scheduling_framework_for_visual_evidence_acquisition_.md)**

:   CSMR 受 Baddeley 工作记忆理论启发，把"视觉证据何时引入推理"做成动态决策——LLM 维护推理状态，按需调用独立感知模块（VLM）拉视觉证据，直到证据够再终止；解决两大现有范式的缺陷（pre-reasoning 文本化丢细节 / unified VL 空间被语言先验污染），在多个多模态推理基准上零样本超越基线。

**[Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding](multimodal_vlm/manga109-v2026_revisiting_manga109_annotations_for_modern_manga_understanding.md)**

:   作者重审 Manga109 这一漫画 AI 研究的基础数据集，识别出五类对话文本标注问题，结合商用 OCR + GPT-5/Gemini 3 Flash 双 LLM 投票 + 人工校验，修订约 29,000 条标注（占全部 147,887 条文本标注的 19.6%）发布 Manga109-v2026，使端到端 OCR 评测 H-mean 从 48.5 提升到 62.9（+14.4 pp）。

**[Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training](multimodal_vlm/med-scout_curing_mllms_geometric_blindness_in_medical_perception_via_geometry-aw.md)**

:   Med-Scout 把"医学 MLLM 在病灶定位时不遵守图像几何约束"这一系统性缺陷定义为"几何盲"，用三个不需要专家标注的几何代理任务（多尺度定位 / 拓扑拼图 / 异常一致性）配合稠密几何奖励（DGR）在 GRPO 下做后训练，并发布 Med-Scout-Bench 用于量化几何盲，在四个 backbone、八个医学基准上一致提升，开源模型甚至反超 GPT-5 / Gemini-3-Flash。

**[Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)**

:   CRG 把每个注意力头的输出沿视觉/文本两条路线做精确线性分解，用一前向一反向梯度估计两条路线对当前 token 的因果"do-effect"，再仅压制那些视觉与文本符号冲突且 VRI 偏低（即先验主导）的头的文本路线，从而在无需训练的前提下系统性削弱 LVLM 的语言先验幻觉。

**[Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](multimodal_vlm/mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)**

:   本文揭示并形式化 MLLM-as-a-Judge 的"感知判断偏差"——评判模型在视觉证据与文本叙述冲突时倾向于奖励语言上更流畅的回答，并通过构造感知扰动数据集 PPJD 与基于 GRPO 的批量排序奖励训练，仅用 3k 样本就让 7B 评判器在多模态评测一致性、单分预测、批量排序三类协议上同时大幅超越同尺寸基线。

**[MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue](multimodal_vlm/mm-snowball_evaluating_and_mitigating_hallucination_snowballing_in_multimodal_mu.md)**

:   本文提出 MM-Snowball 基准（4992 条 6 轮对抗对话）系统刻画多模态大模型在长对话中"幻觉滚雪球"现象，并据此设计训练无关的 CAVR 方法，在表征层刷新视觉信号、在 logit 层裁决文本-视觉冲突，从而显著压平后段对话的性能塌陷曲线。

**[Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](multimodal_vlm/model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)**

:   Model-Dowser 用"权重幅值 × 输入激活 × 输出 Jacobian"三因素给 MLLM 的每个参数打分，冻结高分参数、只更新低分参数，从而在 LLaVA/NVILA 上深层微调时既能学好下游任务又能保留预训练知识，相比 SPIDER、ModelTailor 在 H-score 上稳定领先。

**[Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_vlm/multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)**

:   针对 MLLM 在跨场景 VQA 中的视觉遗忘问题，本文构建 MSVQA（高空/水下/低空/室内 4 场景）基准，并提出 Unifier 框架——在视觉 block 里加入 CSR 多分支 + 投影器（VRE）做参数隔离，再用 KL 软约束（VCC）对齐不同分支表征，单次推理即可在 20 步持续学习上把 VQA 提升 2.70-10.62%、F1 提升 3.40-7.69%。

**[Neutral-Reference Prompting for Vision-Language Models](multimodal_vlm/neutral-reference_prompting_for_vision-language_models.md)**

:   本文将 VLM 高效迁移中的 Base-New Trade-off (BNT) 重新归因为"预训练带来的非对称类别偏好在未见类上未被消除"，提出 NeRP：用一个语义中性的文本 prompt 和"训练图均值"作为参考输入，在已训练好的 VLM 上零参数估计每个类别的先验偏移，再用贝叶斯风格的代理分数在易混淆类对之间做局部翻转，从而在不动模型参数的前提下提升未见类精度并保住基类精度。

**[On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](multimodal_vlm/on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)**

:   本文首次系统研究了带视觉Token压缩的大视觉语言模型(LVLM)的对抗鲁棒性，指出现有编码器攻击存在"优化-推理空间不匹配"问题，并提出 CAGE 攻击通过期望特征扰动 (EFD) 与排名-扰动对齐 (RDA) 两个目标，在未知压缩机制与未知Token预算下显著降低被压缩 LVLM 的鲁棒精度。

**[Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](multimodal_vlm/pair2scene_learning_local_object_relations_for_procedural_scene_generation.md)**

:   Pair2Scene 把 3D 室内场景生成从「直接拟合全局联合分布」改成「学习一对一的局部物体关系（支撑 + 功能）然后按场景层级树递归装配」，配合点云几何编码、Mixture-of-Logistics 概率头和碰撞感知拒绝采样，在仅用 3D-Front 数据训练时即可生成对象数从约 4 跃升到约 14 的复杂场景，FID 和用户研究均优于 ATISS、DiffuScene、LayoutVLM 等基线。

**[R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](multimodal_vlm/r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)**

:   R³L 把 MLLM 多跳"相对空间关系"推理的两类系统性误差（语义漂移与度量漂移）归因于"反复发生的参考系变换"，并通过不变性空间分解（缩短关系链）、一致性空间想象（imagine-and-revise 循环消除冲突）与支持性空间优化（全局-局部位姿重参数化）三个模块，让 GPT-5 生成的开放词汇 3D 场景在 9 类场景下的碰撞率与越界率都接近 0、语义指标显著反超 LayoutVLM/Holodeck/LayoutGPT。

**[Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](multimodal_vlm/referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)**

:   CSteer 提出一种训练无关的 latent steering 方法,通过在错误/正确指代回答的隐藏激活差上构造"上下文向量",并在推理时分层注入到 query 早期层和 decode 中后期层,让通用 LMM (Qwen3-VL、InternVL-3.5) 在多区域视觉指代任务上反超专门微调的 region LMM。

**[Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](multimodal_vlm/revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)**

:   本文把 LVLM 幻觉重新定义为"被语言先验压制的视觉信息缺失"，用正交投影从原始视觉方向中剔除语言先验得到"纯视觉向量"，再用风险门控只在最优深度的单层做稀疏干预，免训练地把 CHAIRS 幻觉率降 ~19% 同时保住 MM-Vet 通用能力。

**[ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](multimodal_vlm/revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)**

:   本文系统揭示了被广泛使用的 VSI-Bench 因 3D 标注漂移与帧采样不一致而存在结构性失效，进而重新标注 381 个场景、5365 个对象，并设计帧预算自适应 QA 与"删除查询对象帧"的 dummy 视频压力测试，构建出名为 ReVSI 的高保真空间智能基准；评估显示开源 VLM 在 ReVSI 上掉点最多 40%，且在 dummy 视频上幻觉率仍高，暴露出现有空间推理能力被 VSI-Bench 系统性高估。

**[SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](multimodal_vlm/same_stabilized_mixture-of-experts_for_multimodal_continual_instruction_tuning.md)**

:   SAME 把多模态持续指令微调里 MoE-LoRA 的"灾难性遗忘"明确拆成 router drift 和 expert drift 两个独立来源，分别用谱感知的子空间约束更新路由器、用历史输入协方差做 Riemannian 预条件保护专家、再用任务级自适应冻结去掉冗余更新，在 CoIN / UCIT 及作者自建的 TriGap 长序列基准上稳定打过现有 MoE 持续学习 SOTA。

**[ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision](multimodal_vlm/screenparse_moving_beyond_sparse_grounding_with_complete_screen_parsing_supervis.md)**

:   针对 GUI agent 普遍使用"稀疏 grounding"标注、丢失整屏结构的问题，本文用全自动 Webshot 流水线构建了 771K 截图 / 21M 元素 / 55 类的稠密屏幕解析数据集 ScreenParse，并训练出仅 316M 参数的 ScreenVLM 把整屏解析为 ScreenTag 结构序列，在密集解析与稀疏 grounding 多个 benchmark 上击败 8B 级别的基础 VLM 同时把延迟降到 $\sim 1/4$。

**[Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](multimodal_vlm/seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)**

:   作者把 decoder-only MLLM 里的因果注意力掩码改一个"洞"，让排在前面的图像 token 反过来去看后面的文本问题 token——这一行掩码修改不加任何参数、不改训练数据，在 3 个 LLM backbone 与 12 个多模态基准上平均涨 6.2 个点。

**[Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](multimodal_vlm/self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)**

:   本文借助 Pointwise Partial Information Decomposition 量化视觉-文本模态交互，并提出 Multimodal Interaction Gate：自动挑出「图像独有信息占主导」的样本让 VLM 自我生成 caption 灌入文本侧，从而把 unique 视觉信号转成 redundant 共享信号，使 VLM 在模糊或被污染输入下的视觉幻觉下降 38.3%、一致性提高 16.8%。

**[Self-Prophetic Decoding to Unlock Visual Search in LVLMs](multimodal_vlm/self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)**

:   SeProD 让经过视觉搜索后训练的 LVLM 与其未微调的预训练版本配对，把预训练模型当作"先知"在每一步生成单步草稿前缀，再由后训练模型按概率阈值选择性接受这些前缀，从而在不训练、零额外计算的前提下同时保住单步基础能力与多步推理连贯性。

**[SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](multimodal_vlm/slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)**

:   SLQ 把一小组"共享潜在查询" $\mathbf{Q}$ 追加到图像/文本 token 序列尾部，借助 MLLM 自身的因果注意力聚合全局上下文，**只训练几千个查询参数**就让冻结的 MLLM 变成检索器，在 COCO/Flickr30K 上胜过全量微调和 LoRA，并配套发布了考验"隐式知识推理"能力的 KARR-Bench。

**[Smoothing Slot Attention Iterations and Recurrences](multimodal_vlm/smoothing_slot_attention_iterations_and_recurrences.md)**

:   针对 Slot Attention 在图像与视频对象中心学习中"冷启动查询信息不足"和"首帧/非首帧聚合变换被强行统一"两个长期被忽视的痛点，作者提出 SmoothSA：用一个自蒸馏的小预热模块给查询注入样本信息，同时让首帧跑完整三次迭代、非首帧只跑一次，从而在图像和视频两个 OCL 基准上同时刷新 SOTA。

**[SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval](multimodal_vlm/solar_self-supervised_joint_learning_for_symmetric_multimodal_retrieval.md)**

:   SOLAR 提出第一套面向"对称 MM2MM 检索"（查询和文档都是 image+text 对、且角色可互换）的两阶段自监督学习框架——第一阶段通过全局-局部对齐 + QDA 自适应阈值学习出"交集 mask"以解耦图文的共享/独有信息，第二阶段利用该 mask 通过对图文不同区域分别掩码构造正/硬负样本做对比学习，并配套发布 214 个人工校验的 sym-MM2MM benchmark；最终以 0.2B 参数和 768 维嵌入超过最强 7.75B VLM 基线 7.08 个百分点。

**[Spectral-Progressive Thought Flow for Lightweight Multimodal Reasoning](multimodal_vlm/spectral-progressive_thought_flow_for_lightweight_multimodal_reasoning.md)**

:   SpecFlow 把多模态空间推理从"像素思维"切到"谱思维"——用块离散余弦变换 + 流匹配 + 渐进式频率激活在固定大小的谱工作空间里维护可视化中间思想，加上分类器无关引导（CFG）让文本制导视觉演化，在保持空间推理精度的同时把 KV 缓存削减 1.6–2.1×。

**[Text-Conditional JEPA for Learning Semantically Rich Visual Representations](multimodal_vlm/text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)**

:   本文提出 TC-JEPA，把 I-JEPA 的 mask 特征预测器额外条件化在图像 caption 上，通过多层稀疏跨注意力让 patch 表示在文本"提示"下变得可预测，从而在不用对比损失的前提下学到语义更丰富、对密集预测尤其友好的视觉表征。

**[TGV-KV: Text-Grounded KV Eviction for Vision-Language Models](multimodal_vlm/tgv-kv_text-grounded_kv_eviction_for_vision-language_models.md)**

:   TGV-KV 通过"用文本注意力来支配视觉 KV"的三件套——按 text-vision 注意力分层预算、用主导文本 token 加权重排视觉重要性、并在驱逐时优先保住文本 KV——把为纯文本 LLM 设计的 KV eviction 思路成功迁移到 VLM，在 LLaVA-NeXT/Qwen3-VL 上 5% 保留率下仍能保住接近满 KV 的精度，吞吐量提升 52.6%。

**[The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](multimodal_vlm/the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)**

:   本文把"VLM 看不清细节"形式化为一个序贯贝叶斯最优实验设计 (S-BOED) 问题,并基于"覆盖率 × 分辨率"的可计算代理目标提出训练免费的 FOVEA 模块,在高分辨率/遥感等基准上稳定超过 Direct 与 ReAct-style 基线。

**[Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](multimodal_vlm/thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)**

:   作者构造了 SSI-Bench，一个由 1,000 道排序型 VQA 组成、聚焦"受约束的结构化空间"（屋顶、桥梁、塔架等真实 3D 结构）的基准，要求 VLM 对 3-4 个候选构件按几何或拓扑准则给出完整排列；评测 31 个 VLM 后发现最强闭源模型 Gemini-3-Flash 仅 33.6%、最佳开源 GLM-4.6V 22.2%，而人类 91.6%，揭示当前 VLM 在受几何/连接/物理可行性共同约束的真实 3D 场景下缺乏一致的空间推理能力。

**[TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings](multimodal_vlm/timespot_benchmarking_geo-temporal_understanding_in_vision-language_models_in_re.md)**

:   作者构建了一个覆盖 80 个国家、1,455 张真实地面图像的 TimeSpot 基准，强制 VLM 同时给出"何时（季节/月份/分钟级本地时间/日相）"与"何地（洲/国/气候带/环境类型/经纬度）"九字段结构化预测，结果显示即便最强模型 Gemini-2.5-Flash-Thinking 也只达到 77.59% 国家准确率、892.54 km 中位地理距离误差，分钟级时间准确率不到 34%，说明 VLM 严重缺乏基于物理线索的地理-时间联合推理能力。

**[Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)**

:   本文提出 S3 框架，用 MoE 把多模态表征分解为概念级专家（Specialization）、按任务路由激活相关专家（Selection）、并在推理时按路由分数剪枝低贡献路径（Sparsification），在四个 MultiBench 基准上揭示了一条"性能在中间稀疏度达峰"的反 U 型曲线，给出对比学习/InfoMax 之外第三种多模态表征范式。

**[TRAP: 用对抗 patch 劫持 VLA 的 CoT 推理实现目标行为攻击](multimodal_vlm/trap_hijacking_vla_cot-reasoning_via_adversarial_patches.md)**

:   TRAP 是第一个针对 reasoning VLA 的目标行为劫持攻击——通过桌布大小的物理对抗 patch 劫持 VLA 的 CoT 推理（边界框/轨迹/子任务），让机器人在用户指令保持「拿苹果」时改为「拿刀给人」；在 MolmoAct/GraspVLA/InstructVLA 三种 CoT 范式上平均 ASR 52.54%，真实世界打印 patch 在 GraspVLA 上 occlusion-free 部署 86.7% 干扰成功率、33.3% 完全控制率。

**[TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](multimodal_vlm/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)**

:   TUR-DPO 在 DPO 的偏好 logit 上同时叠加一个"语义+拓扑结构"塑形奖励差和一个"按每对样本不确定性"动态降权的实例权重，让模型在保持 RL-free 训练简洁性的同时，显式奖励推理过程的结构合理性并削弱脆弱偏好对的影响，从而在 GSM8K / MATH / BBH / QA 等推理类任务上系统超过 DPO 与 IPO，并在多数任务上追平 PPO。

**[通用骨架理解：可微渲染与 MLLMs](multimodal_vlm/universal_skeleton_understanding_via_differentiable_rendering_and_mllms.md)**

:   通过将骨架序列渲染为图像让 MLLMs 能够理解多种格式的骨架数据——实现通用骨架理解，解决跨模态和格式异构问题。

**[揭示视觉-语言模型中的视觉计数瓶颈](multimodal_vlm/unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)**

:   通过将视觉计数分解为三个认知阶段——发现 VLM 的计数失败根源不在视觉感知或数量理解，而在符号映射阶段无法将视觉表征投影到正确的文本标记，反映出模型缺乏统一的跨模态数字表示空间。

**[V-LynX: Token Interface Alignment for VideoX LLMs](multimodal_vlm/v-lynx_token_interface_alignment_for_videox_llms.md)**

:   V-LynX 通过发现 Video LLM 内部的**连续 token interface（流形）**——视觉编码器 + 投影层雕刻出的与 LLM 内部操作空间兼容的几何先验——仅用轻量级 LoRA（68.7M 参数）和**未配对的单模态数据**就能将新模态（音频、3D、高帧率视频）高效集成到预训练 Video LLM 中，AVSD 上 CIDEr 145.7 vs PAVE 134.5（参数减少 46%）。

**[Vision-aligned Latent Reasoning for Multi-modal Large Language Model](multimodal_vlm/vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)**

:   本文提出 VaLR：在 MLLM 的 CoT 推理每一步之前插入若干"潜在 token"，并用 DINOv3/SigLIP/π³ 等视觉编码器的 patch 特征对这些 token 做表征对齐（REPA），从而在长链推理中持续把视觉信息"喂回"模型，把 Qwen2.5-VL 在 VSI-Bench 上的准确率从 33.0% 拉到 52.9%，并首次让 MLLM 表现出"推理越长越准"的 test-time scaling 行为。

**[Vision Language Models 无法推理物理变换](multimodal_vlm/vision_language_models_cannot_reason_about_physical_transformation.md)**

:   本文通过 ConservationBench 基准测试揭示——112 个 VLM 虽然声称具有强大的感知和推理能力，但在判断物理变换中的守恒性（如倒水体积不变）时系统性失败，仅依赖文本先验而非真正的视觉理解。

**[VisionPulse：多模态推理中的动态视觉稀疏化](multimodal_vlm/visionpulse_dynamic_visual_sparsity_for_efficient_multimodal_reasoning.md)**

:   VisionPulse 提出训练无关的步级视觉令牌动态剪枝框架——根据每个解码步骤中变化的视觉依赖性自适应调整保留令牌数，仅保留 5% 视觉令牌的同时维持推理精度，将推理长度缩短 11.2%。

**[视觉说服力：什么影响了视觉-语言模型的决策？](multimodal_vlm/visual_persuasion_what_influences_decisions_of_vision-language_models.md)**

:   本论文通过系统使用图像编辑模型修改视觉属性（保持语义不变），发现 VLM 存在显著视觉偏好；提出三种视觉提示优化方法揭露这些偏好，开发自动可解释性管道理解驱动决策的视觉主题，并通过视觉归一化缓解风险。

**[VLA-Arena：评估视觉语言动作模型的开源框架](multimodal_vlm/vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)**

:   VLA-Arena 提出结构化 VLA 基准——通过任务结构、语言命令和视觉观察三个正交维度系统量化难度，用 170 个任务揭示现有 VLA 模型在泛化、视觉感知和安全性上的关键缺陷。

**[VLANeXt：构建强大 VLA 模型的配方](multimodal_vlm/vlanext_recipes_for_building_strong_vla_models.md)**

:   本文系统探索 VLA 模型的设计空间，通过 500+ 对照实验提炼出 12 条关键设计原则——构建高效强大的 VLANeXt 模型，在 LIBERO 基准上超越 SOTA，并在真实机器人任务中验证了设计原则的有效性。

**[WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](multimodal_vlm/weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)**

:   WeatherSyn 把气象预报员的报告写作流程拆解成"看图→列要点→出稿"的多模态指令任务，先建了首个覆盖 31 个美国城市、8 类天气要素的 WSInstruct 数据集，再用 SFT→RFT→DPO 三段式微调 Qwen3-VL-8B，让一个 8B 开源模型在多种评测指标上稳定打过 GPT-5-Nano、Claude-3.7-Sonnet 等闭源大模型，并对未见城市有零样本泛化能力。

**[What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](multimodal_vlm/what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)**

:   GLANCE 在 VLM agent 的强化学习里加了一个"想-看对齐"自监督头：让 LLM 在 CoT 里产出的"下一状态预测"通过一个轻量 projector 映射到由 EMA target 视觉编码器编码的真实下一帧表示，预测与实际之间的差距同时充当内在好奇心奖励、视觉编码器的训练信号、以及让 internalized world model "落地"的对齐损失；再配合周期性重置 projector 的课程探索机制对抗好奇心衰退，最终在 5 个 agentic 任务上稳定超过现有 exploitation-only 的 VLM-RL 方法。

---

## 📦 模型压缩 { #model_compression }

**[A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search](model_compression/a_language-guided_bayesian_optimization_for_efficient_lora_hyperparameter_search.md)**

:   本文把 LoRA 超参数配置写成带领域解释的文本，让冻结 LLM、可学习 token 和投影层共同构造 BO 的连续搜索空间，再用 10% 数据代理评估降低每次试验成本，在 30 次左右搜索内显著优于默认 LoRA 配置和常规 HPO 方法。

**[A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](model_compression/a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)**

:   本文建立首个显式纳入 KV 缓存显存动态的 LLM 推理排队模型，给出闭形稳定性条件 $\lambda < \mu(1-\delta)$，使运维人员可直接计算所需 GPU 数；在单 GPU、8 GPU 集群与 LongBench 真实数据上验证误差均 $\leq 10\%$。

**[Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning](model_compression/active_budget_allocation_for_efficient_scaling_law_estimation_via_surrogate-guid.md)**

:   本文把 scaling law 实验中的训练预算分配建模为多轮资源选择问题，用 Successive Halving 结合学习曲线 surrogate 预测未来潜力，在 synthetic 和 nanoGPT 学习曲线上以最高 98.7% 的训练成本节省近似完整 scaling law。

**[Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](model_compression/active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)**

:   本文形式化了表格增强中的"保真度-效用间隙"问题（生成器优化分布匹配，而增强价值源于低密度区域），提出 TAP 算法通过扩散填补做流形约束提议、策略引导的效用对齐选择、硬约束门控加保守窗口提交，在 7 个真实表格数据集上相比基线最多提升分类精度 15.6%、回归 RMSE 降低 32%。

**[Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation](model_compression/advantage_collapse_in_group_relative_policy_optimization_diagnosis_and_mitigatio.md)**

:   这篇论文指出 GRPO 在二值可验证奖励下会因为组内奖励全同而失去梯度信号，提出 ACR 指标实时诊断这种“优势坍塌”，并用 AVSPO 注入虚拟奖励样本恢复组内方差，从而在多个 Qwen2.5 数学推理模型上稳定提升 4-6 个百分点。

**[An Algebraic View of the Expressivity of Recurrent Language Models](model_compression/an_algebraic_view_of_the_expressivity_of_recurrent_language_models.md)**

:   这篇论文把 RNN/SSM 的形式语言表达能力统一为一个代数问题：在固定数值语义后，模型能识别的语言由其层级转移幺半群及其 wreath product 决定，并且同一架构在浮点与无符号整数语义下会得到完全不同的计数能力。

**[ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](model_compression/arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)**

:   作者诊断出 VQ-VAE 的码本坍塌根源是"码本向量 ℓ2 范数失衡 + 几何聚集"，于是提出 SAMP：Ball-Bounded Norm Regularization 把所有码本向量约束在时变 Euclidean 球内、ArcCosine Additive Margin Loss 借鉴 ArcFace 在球面上推开 latent 向量，从而让码本均匀分散、利用率大幅上升，在 ImageNet 重建和生成 FID 上都击败主流 VQ-VAE 变体。

**[AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning](model_compression/area_attribute_extraction_and_aggregation_for_clip-based_class-incremental_learn.md)**

:   这篇论文把 CLIP 类增量学习中的遗忘拆成“属性抽取漂移”和“属性聚合漂移”，提出 Area 用 PGA 在超球面上固定视觉/文本属性锚点，再用轻量任务专家、VIB 正则和 OT 路由稳定属性聚合，从而在九个 CLIP-CIL benchmark 上显著提升平均精度和最终精度。

**[Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice](model_compression/auditing_and_fixing_economic_validity_in_tabular_foundation_models_for_discrete_.md)**

:   本文发现TabPFN和Mitra等表格基础模型在离散选择任务中虽然准确率高，却会违反价格-需求单调性和值得信任的时间价值估计，因此提出两阶段行为适配器，把TFM预测嵌入受经济理论约束的效用模型中，在保持100%行为有效性的同时回收大部分准确率收益。

**[Beyond Temperature: Hyperfitting as a Late-Stage Geometric Expansion](model_compression/beyond_temperature_hyperfitting_as_a_late-stage_geometric_expansion.md)**

:   本文通过控制实验证明 Hyperfitting（在小数据集上将 LLM 训练到近零损失）的本质不是温度缩放式的分布锐化，而是一种动态的、上下文相关的 token 秩重排序（Rank Reordering）机制，该机制集中发生在 Transformer 最后一层的"终端几何扩展"（$\Delta \text{Dim} \approx +80.8$），并据此提出仅微调最后 5 层的 Late-Stage LoRA，在减少约 80% 可训练参数的同时保持生成多样性。

**[Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](model_compression/beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)**

:   提出 StructRTL，在 RTL 设计的控制数据流图（CDFG）上做结构感知的图自监督预训练（掩码节点建模 + 边预测），再配合从后映射网表到 CDFG 的知识蒸馏，大幅超越 LLM 和手工特征方法在面积/延迟预测任务上的 SOTA。

**[BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models](model_compression/bioarc_discovering_optimal_neural_architectures_for_biological_foundation_models.md)**

:   BioArc 提出了一个面向生物基础模型的异构神经架构搜索框架，通过在包含 CNN/LSTM/Transformer/Mamba/Hyena 五种基本模块的搜索空间中自动发现最优混合架构，以不到 1/25 的参数量超越现有 SOTA 生物基础模型。

**[Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](model_compression/bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)**

:   提出 Bounded Hyperbolic Tanh (BHyT)，一种基于数据驱动输入界定的 $\tanh$ 变换，作为 Pre-Layer Normalization 的即插即用替代，在抑制深度方向激活增长的同时避免重复方差计算，训练速度比 RMSNorm 快 1.6%、生成吞吐提升 1.77%，且下游性能全面优于现有方法。

**[Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](model_compression/breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)**

:   针对 MoE LLM 的"负载不均–参数冗余–通信开销"三难，本文提出一个统一框架：用"参数 + 激活"双相似度在线聚类把专家分组，组内用"共享基矩阵 + 低秩残差"做结构化压缩 (~5×)，再做"先选组后选 expert"的两级分层路由 + FP16/INT4 异构精度 + 闲置组离线卸载，在 GLUE/WikiText-103 上以约 80% 参数缩减、10–20% 吞吐提升、专家负载方差降 3× 的代价匹配标准 MoE 性能。

**[Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video](model_compression/causal_forcing_autoregressive_diffusion_distillation_done_right_for_high-quality.md)**

:   本文通过识别"**帧级单射性**"的理论需求，提出 Causal Forcing 方法——用**自回归教师模型替代双向教师**进行 ODE 蒸馏初始化，避免 Self-Forcing 中的性能坍缩；相比 Self-Forcing 动态度 +19.3%、VisionReward +8.7%、指令遵循 +16.7%，同时保持相同推理延迟（0.69s）。

**[Compositional Consistency-Guided Decoding for Three-Way Logical Question Answering](model_compression/compositional_consistency-guided_decoding_for_three-way_logical_question_answeri.md)**

:   利用三分类逻辑问答中假设 $H$ 与其否定 $\neg H$ 之间的确定性否定映射关系，在测试时组合多次 LLM 调用并通过一致性约束消歧，无需训练即可减少认识性弃权（epistemic Unknown）并提升推理准确率。

**[Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](model_compression/compress_then_merge_from_multiple_loras_into_one_low-rank_adapter.md)**

:   提出 Compress-then-Merge (CtM) 管线，在合并多个 LoRA 之前先学习共享 $r$ 维子空间并将各 adapter 投影为 $r \times r$ 坐标矩阵，再在低维空间中执行合并，从而在构造层面保证输出为 rank-$r$ LoRA，避免了传统 Merge-then-Compress 方法的截断 SVD 性能损失。

**[Continual Model Routing in Evolving Model Hubs](model_compression/continual_model_routing_in_evolving_model_hubs.md)**

:   当模型 hub 里的可用专家从几百涨到上千、还在持续新增/淘汰时，传统"训一次路由器"或"纯检索 model card"都顶不住；作者把这个问题形式化成"持续分类（label space 不断长大）"，搭出 CMRBench 这个跨 4 期、超过 2000 个候选模型的基准，并提出 CARvE——一个用对比嵌入打分、用 checkpoint anchoring 防漂移、用结构化负样本回放维持判别力的持续路由器，在 D-Acc 上比标准 LoRA 重放高 5 个点、遗忘只有它的 1/2。

**[Critique-Guided Distillation for Robust Reasoning via Refinement](model_compression/critique-guided_distillation_for_robust_reasoning_via_refinement.md)**

:   让 student 在训练时**消费**而不是**生成** teacher 的 critique——以 (prompt, student 自答, teacher critique) 为条件预测 teacher 的 refined answer，推理时只需一遍 prompt 就能产出更长更准的推理链，且不像 CFT 那样把指令跟随能力毁掉。

**[DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](model_compression/dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)**

:   把标准 MoE 中 top-$K$ 专家输出的"加权求和"替换为按一个动态学习出来的 DAG 进行结构化聚合，在几乎不增加路由与参数开销的前提下显著提升 MoE 表达能力与下游推理表现。

**[Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](model_compression/decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)**

:   论文针对多任务指令微调中的跨任务梯度冲突问题，提出 Badit：先用 SVD 把预训练权重分解为一组天然正交的高奇异值 LoRA "基础能力"专家，再在训练过程中用球面 K-means 对 rank-1 分量做动态正交分组，从而把"按任务隔离参数"的传统思路改为"按基础能力解耦"，在 6 个 LLM 上平均比 GainLoRA 提升 2.68 Rouge。

**[Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](model_compression/decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)**

:   为了在 LLM 预训练里找最优数据混合比例又不被代理实验拖垮，本文提出 DeMix——只训一次 $N$ 个 component 模型（每个对应一个候选子集），随后任意候选比例 $\{\alpha_i\}$ 都通过加权合并 $\sum_i \alpha_i \Theta_i$ 当作"训练自由"代理，并用 LightGBM 在 simplex 上做迭代回归选最优配方，最终用比 RegMix/CLIMB 少约 $6\times$ 的算力得到更好的下游分数，并附带开源 22T tokens 的 DeMix Corpora。

**[Demystifying When Pruning Works via Representation Hierarchies](model_compression/demystifying_when_pruning_works_via_representation_hierarchies.md)**

:   论文从"嵌入 → logit → 概率"三段表征层次出发，用 Taylor 局部展开理论证明：剪枝对嵌入空间和 logit 空间的扰动天生很小，但 softmax 这一非线性步骤会按 $\mathrm{Var}_r(\Delta z)/(2T^2)$ 把扰动放大到概率空间，再经过自回归解码的步间累积，最终导致生成任务崩溃；而非生成任务因为只依赖候选 token 子空间，对剪枝天然鲁棒——这统一解释了为什么剪枝在 MMLU、retrieval 上几乎无损但在 GSM8K、HumanEval 上骤降到 0。

**[Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes](model_compression/detecting_fluent_optimization-based_adversarial_prompts_via_sequential_entropy_c.md)**

:   作者把"流畅型优化越狱后缀检测"建模成 token-level 熵流上的在线变点检测：用固定系统提示的熵分布算 MAD 鲁棒基线把用户 token 熵标准化，跑一边 Page-CUSUM 累计统计量 $W_t^+$ 越阈值就报警，在 6 个开源对齐 LLM 上对 GCG / AutoDAN / AdvPrompter / BEAST / AutoDAN-HGA 五类攻击都比窗口困惑度 F1 更高，并能把 79.6% 的报警精确定位到 suffix 内部，还能当 LLaMA Guard 的轻量门，节省 17-42% 的 guard 调用。

**[Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](model_compression/dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)**

:   本文系统观测到 "小语言模型的 token 嵌入会随深度坍缩到一个窄锥体"（embedding condensation）这个普遍现象——大模型反而不会——并设计了一个角度分散损失 $\mathcal{L}_{\text{disp}}$ 直接逼嵌入散开，无须加参数就让 Qwen3 / GPT2 在 10 个 benchmark 上平均提升 3.3%。

**[DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery](model_compression/diverdiving_deeper_into_distilled_data_via_expressive_semantic_recovery.md)**

:   DIVER 把经典数据集蒸馏 (DD) 从"单阶段直接评估"改造成"先蒸馏再用预训练扩散模型救活语义"的双阶段范式，通过语义继承、语义引导、语义融合三步从 ConvNet 蒸馏出来的"乱码"图像中恢复被压抑的高层语义，让同一份蒸馏数据在 ResNet18/ViT 等异构架构上的精度普遍提升 3–10 个百分点，每张图只要 2.48s 和 4GB 显存。

**[Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](model_compression/dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)**

:   本文提出 TAD（Tail-Aware Distillation）：在标准 KD 的 KL 散度中显式把教师 top-$K$ 概率与"尾部"概率拆开并放大尾部贡献，从而在学术级算力（单卡 H100 + 1 周）内完成 LLM 预训练蒸馏，平均效果优于 MiniPLM 等数据中心方法。

**[DSL-Topic: Improving Topic Modeling by Distilling Soft Labels from Language Models](model_compression/dsl-topic_improving_topic_modeling_by_distilling_soft_labelsfrom_language_models.md)**

:   作者用小语言模型在"给文档生成一个主题词"提示下产生的下一 token 概率投影到主题模型词表，作为 dense 软标签替换传统的 BoW 重构目标来训练神经主题模型 (ProdLDA / ECRTM / FASTopic)，在 20NewsGroup、TweetTopic、StackOverflow 三个数据集上把分配纯度 (Purity) 拉高一大截，并给出"把 LM 隐式后验预测投影到结构化主题家族"的贝叶斯解释。

**[Easier to Judge Than to Find: Predicting In-Context Learning Success for Demonstration Selection](model_compression/easier_to_judge_than_to_find_predicting_in-context_learning_success_for_demonstr.md)**

:   本文把 ICL 示例选择从「在巨大组合空间里搜最优 $D^\star$」改造为「对采样到的 $(q,D)$ 对判断是否会成功」，提出 DiSP——一个按查询难度分层、用轻量裁判模型做「采样–判定–接受即停」的框架，在五个分类基准上比强基线最多提升 3.4%，端到端实时延迟最多降 23×。

**[Effective Model Pruning: Measure the Redundancy of Model Components](model_compression/effective_model_pruning_measure_the_redundancy_of_model_components.md)**

:   本文借鉴粒子滤波中的「有效样本量」概念，把任意打分向量直接映射到一个自适应保留个数 $N_{\text{eff}} = \lfloor 1/\sum_i \omega_i^2 \rfloor$，作为剪枝阈值，避免人工设定稀疏度并给出剪枝前后损失变化的理论上界。

**[Efficient Learned Image Compression without Entropy Coding](model_compression/efficient_learned_image_compression_without_entropy_coding.md)**

:   EF-LIC 用"无约束向量量化最大化索引熵 + 表征域上下文重参数化消除潜变量间相关性"两步替代了 learned image compression 流水线里慢且串行的熵编码模块，理论证明其 R–D 性能可逼近熵编码方案，实际在 Kodak/LPIPS 上比 MS-ILLM 省码 67.86% 且解码快 10 倍。

**[End-to-End Compression for Tabular Foundation Models](model_compression/end-to-end_compression_for_tabular_foundation_models.md)**

:   TACO 在 TabPFN 类表格基础模型前面接一个可学习的 transformer 压缩器，把 $N$ 行训练上下文压成 $K\ll N$ 行的潜在表示后再喂给预测器，并与预测器端到端联合元学习，使得 1% 压缩率下推理快 94 倍、显存省 97% 而 ROC-AUC 几乎无损。

**[Energy-Structured Low-Rank Adaptation for Continual Learning](model_compression/energy-structured_low-rank_adaptation_for_continual_learning.md)**

:   E2-LoRA 不在参数空间或输入特征空间做正交约束，而是把视角换到"任务引起的输出特征漂移" $\Delta \mathbf{Y}_t = \Delta \mathbf{W}_t \mathbf{X}_t$，对它做 SVD 后把 LoRA 参数重排到能量集中且按秩有序的基上，从而能丢掉低能量秩、把容量回收给新任务，并配合按能量保留率自适应分配秩的策略，在多个持续学习基准上拿到 SOTA。

**[Entropy-Aware On-Policy Distillation of Language Models](model_compression/entropy-aware_on-policy_distillation_of_language_models.md)**

:   针对在策略蒸馏中 reverse KL 在教师高熵区域引发多样性坍缩和梯度不稳的问题，提出根据教师 token 级熵值自适应混合 forward KL 与 reverse KL 的蒸馏策略，在六个数学推理基准上 Pass@8 最高提升 +5.05。

**[EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments](model_compression/epicache_episodic_kv_cache_management_for_long-term_conversation_on_resource-con.md)**

:   提出 EpiCache，一个免训练的 KV 缓存管理框架，通过分块预填充控制内存上限、情节式聚类保留话题相关上下文、层级敏感度感知的预算分配优化层间缓存分配，在三个长对话 QA 基准上以 4-6 倍压缩率达到接近全缓存精度，并将峰值内存降低 3.7 倍。

**[Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space](model_compression/event2vec_processing_neuromorphic_events_directly_by_representations_in_vector_s.md)**

:   仿照 word2vec 的思路，把事件相机产生的稀疏异步事件 $(x,y,t,p)$ 直接嵌入到向量空间，用参数化空间嵌入 + 卷积时间嵌入 + K-Means++ 聚合，让标准 Transformer 既能保留事件的稀疏异步特性，又能在 GPU 上高吞吐运行，参数量只有以往 SOTA 的 $\tfrac{1}{2.8} \sim \tfrac{1}{816}$。

**[EVL-ECG: Efficient ECG Interpretation With Multi-Aspect Heterogeneous Knowledge Distillation](model_compression/evl-ecg_efficient_ecg_interpretation_with_multi-aspect_heterogeneous_knowledge_d.md)**

:   EVL-ECG 针对 ECG 解读的 VLM 蒸馏问题（teacher 与 student 在视觉 token 数量、tokenizer、序列长度上都异构），引入"多头交叉注意力对齐 + 最优传输视觉特征匹配 + 几何关系内部匹配"三模块的跨架构蒸馏框架，把 2B 学生模型推到 SOTA，AUC 比已有 KD 高 2.4%、临床准确率高 1.1%。

**[Exploiting Weight-Space Symmetries for Approximating Curvature](model_compression/exploiting_weight-space_symmetries_for_approximating_curvature.md)**

:   本文证明只要利用神经网络损失对参数重排/重缩放等"权重空间对称群"的不变性、对单个梯度做轨道平均，就能从一次梯度计算里解析地导出一个高度结构化、可廉价存储与求逆的 Hessian 近似；并且 Shampoo / Muon 恰好对应"对某些层指派恒等群"的特例，从而把这两类经验型优化器纳入统一的对称-曲率框架。

**[FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](model_compression/fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)**

:   本文指出联邦 LoRA 中朴素 factor-wise 平均的真正"敌人"是旋转不变性导致的潜在子空间错位，提出在客户端用正交 Procrustes 求解出旋转矩阵 $R_i^t$ 对齐 $A,B$ 因子后再聚合，理论与实验都证明能显著降低聚合误差且不增加通信开销。

**[FedSDR: Federated Self-Distillation with Rectification](model_compression/fedsdr_federated_self-distillation_with_rectification.md)**

:   针对联邦 LLM 微调中客户端数据分布异质带来的"权重漂移"，本文先用模型自身把原始指令重写到"模型可理解空间"做数据级对齐（FedSD），再用 LoRA-S/LoRA-R 双流结构分别吸收风格噪声和锚定事实正确性、并只聚合 LoRA-R，把对齐与忠实解耦，从而在多种 Non-IID 设置下取得 SOTA。

**[Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters](model_compression/finer_parameter_steps_for_low-rank_peft_a_controlled_study_with_cp_tensor_adapte.md)**

:   作者把 LoRA 的"按 rank 增长"换成"按 CP 张量分量增长"，让单步参数增量从 4096 降到 193 (小 21×)，并在 OPT-1.3B / SST-2/RTE/BoolQ 上做严格 controlled study 证明：更细的参数粒度可以作为"诊断 PEFT 预算敏感度"的工具，但本身并不能换来更好的准确率-预算曲线——这是一个清醒的负-中性结论而非"我家方法更强"的宣传。

**[FlattenGPT: Depth Compression for Transformer with Layer Flattening](model_compression/flattengpt_depth_compression_for_transformer_with_layer_flattening.md)**

:   本文提出 FlattenGPT，先把 LLM 中输入相似度高的相邻 transformer 层"扁平化"合并为一个 2× 宽度的层 (保留所有参数知识)，再对合并层做通道剪枝把宽度恢复到原始规模——既享受深度压缩的推理加速，又避免传统层剪枝直接丢知识的性能塌方。

**[Float8@2bits: Entropy Coding Enables Data-Free Model Compression](model_compression/float82bits_entropy_coding_enables_data-free_model_compression.md)**

:   EntQuant 把权重以 Float8/Int8 精度保留，但在量化阶段额外加一个 $\ell_1$ 正则把权重往低熵方向"对齐"，再用 GPU 上并行的 ANS 熵编码无损压到 2 bit 左右，从而在完全不需要校准数据、10 分钟以内、不做恢复训练的前提下，把 70B LLM 压缩到 8× 以上且推理只慢 1.5–2 倍。

**[FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](model_compression/frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)**

:   FRISM 把「VLM × LRM 合并」从层级粒度细化到 SVD 子空间粒度：用 LRM 任务向量的 SVD 子空间作为推理先验，再用一个仅含可学习门控的无标签自蒸馏（KL 保视觉 + 谱幅最大化吸收推理）找到最优注入强度，从而在不显著掉视觉的前提下显著提升 VL 推理性能。

**[From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers](model_compression/from_per-image_low-rank_to_encoding_mismatch_rethinking_feature_distillation_in_.md)**

:   作者用 sample-wise SVD + dataset-level PCA + token-level Spectral Energy Pattern (SEP) 三视角揭示了一个看似矛盾的 ViT 表征几何："每张图的特征矩阵都是低秩的，但跨图共享的子空间却几乎要满秩 + 单 token 的频谱带宽接近 100%"，进而提出 Lift（推理时保留 lifting projector）和 WideLast（只把最后一个 block 加宽到 teacher 宽度）两个极简补丁，让普通 MSE 特征蒸馏在 DeiT-Tiny ← CaiT-S24 上从 74.86% 一路涨到 78.23%。

**[GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](model_compression/gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)**

:   GEMQ 把 MoE 大模型的 expert 比特位分配从层内局部 LP 升级成跨层全局 LP，并配合"量化后微调 router 权重"来对齐被量化扭曲的路由分布，再用"渐进式降比特"的迭代框架反复修正重要性估计，在 Mixtral-8×7B 等 4 个 MoE 模型上把每 expert 平均 2.5 bit 的压缩下 MMLU 等 7 项 zero-shot 平均掉点压在 7% 以内，同 bit 预算下显著超过 PMQ / SpQR / MoEQuant / EAQuant。

**[Geo-Expert: 用 LoRA 把 8B 模型微调成专家级地质推理 LLM](model_compression/geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)**

:   Geo-Expert 把 11,518 条从五本地质学经典教科书蒸馏出的 CoT-enhanced 指令数据用 LoRA 微调 Qwen3-8B/32B 和 Gemma-3-27B；在 Geo-Eval（387 hard boundary 题）上 Qwen3-8B-geo 平均 6.27 超过 Llama-3.1-70B-Instruct（4.12）和 GPT-4o（5.93），Qwen3-32B-geo 6.82 接近 GPT-5.4（7.15）；证明 high-quality domain alignment 比 scaling 重要。

**[Global Convergence of Adaptive Sensing for Principal Eigenvector Estimation](model_compression/global_convergence_of_adaptive_sensing_for_principal_eigenvector_estimation.md)**

:   本文建立压缩流式 PCA 的最优收敛率——使用每步两个**自适应**测量的 Oja 算法在有噪声观测下的误差上界与信息论下界匹配（均为 $\Theta(\lambda_1 \lambda_2 d^2 / (\Delta^2 t))$），首次揭示压缩相对完全观测的根本代价是额外的 $d$ 因子，而自适应相对非自适应又救回一个 $d$ 因子。

**[GradPower: Powering Gradients for Faster Language Model Pre-Training](model_compression/gradpower_powering_gradients_for_faster_language_model_pre-training.md)**

:   GradPower 在喂给任意梯度优化器之前对原始梯度做一次逐元素的"符号保留幂次"变换 $\varphi_p(g_i)=\mathrm{sign}(g_i)\,|g_i|^p$，仅多一行代码、不动 AdamW/Muon 内部逻辑和超参，就能在 LLaMA / Qwen2MoE 从 66M 到 2B 的多个规模上一致拿到更低的终末 loss，尤其在 MoE + wsd 学习率调度下增益最显著。

**[Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing](model_compression/hallucination_is_a_consequence_of_space-optimality_a_rate-distortion_theorem_for.md)**

:   本文把"LLM 记住随机事实"形式化为带连续置信分数的**成员测试**问题，证明在事实稀疏极限下最优记忆开销恰好等于事实/非事实输出分布之间的最小 KL 散度——即"率失真定理"——并由此推出：在 log-loss 目标下，给定有限记忆，最优策略**不是弃答也不是遗忘**，而是把一定比例的非事实和事实压在同一个高置信点上，幻觉是信息论意义下的最优误差形态。

**[Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift](model_compression/hard_labels_in_rethinking_the_role_of_hard_labels_in_mitigating_local_semantic_d.md)**

:   针对大规模数据集蒸馏中"每张图存大量软标签"导致的天价存储成本，本文证明在每图软标签数 $s$ 受限时会发生**局部视图语义漂移 (LVSD)**，并提出 soft→hard→soft 三阶段训练范式 HALD，用平滑后的硬标签作为语义锚把训练拉回正轨——ImageNet-1K 上 285M 软标签存储取得 42.7% 准确率，比 SOTA LPLD 涨 9.0%、软标签存储压缩 100 倍。

**[Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](model_compression/hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)**

:   H-VAR 把"残差量化做多尺度生成"的 VAR 范式重新切片成层次化的图像 tokenization (HIT)，让一个 310M 的小模型只跑一次前向就能输出 128 / 256 / 512 三个有意义的中间分辨率，再配一个不需要外部奖励模型的 DPO 正则项推动输出偏向 HR，在标准 ISR 数据上对打 1B 参数的 VARSR。

**[IDLM: Inverse-distilled Diffusion Language Models](model_compression/idlm_inverse-distilled_diffusion_language_models.md)**

:   本文把连续扩散的"反向蒸馏 (Inverse Distillation)"扩展到离散文本扩散模型，通过证明 IDLM 损失在 SEDD/MDLM/Duo 下的唯一最优解就是真实数据分布，再配合 simplex 松弛与 Gaussian 重参数化解决离散反传不稳定问题，把 1024 步教师 DLM 压到 16 步甚至 4 步而保持 GenPPL/Entropy 与 MAUVE 几乎不掉。

**[Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images](model_compression/images_as_tables_in-context_learning_with_tabpfn_for_low-data_detection_of_ai-ge.md)**

:   作者把 AI 生成图像检测改写成"先用冻结的 DINOv3 把每张图压成 768 维 CLS 向量、再 PCA 降到 500 维当作一行表格、最后扔给 TabPFN 做上下文推断"的三段式流水线，从而把"换一个新生成器要重训分类头"变成"只换 TabPFN 上下文样本"，在 GenImage 低数据与跨生成器场景下相比强基线 LATTE 最高领先 8.2%，在 64 对生成器迁移里 54 对胜出。

**[Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models](model_compression/jailbreak_to_protect_buffering_and_reinforcing_via_temporary_jailbreaking_for_sa.md)**

:   在 Fine-tuning-as-a-Service 场景下，作者把"先把模型临时越狱再让用户微调"重新解读为一种梯度饱和机制，并基于这一观察设计 Buffer-and-Reinforce 框架：用一个可拆卸的 BufferLoRA 在用户微调时吃掉有害梯度，再用 ReinforceLoRA 通过 QR 正交合并补回安全性，无需任何用户侧安全数据就把有害评分压到约 8.5，同时把下游任务准确率维持在 76 以上。

**[LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs](model_compression/lfq_logit-aware_final-block_quantization_for_boosting_the_generation_quality_of_.md)**

:   针对 block-wise PTQ 在生成任务上的质量退化问题，LFQ 将最后一个 Transformer block 的量化目标从 MSE 替换为 logit 级交叉熵损失，使量化模型的 token 分布与全精度模型对齐，在 IFEval/GSM8K/MATH500/AIME 等生成基准上一致提升精度。

**[LiftQuant: Continuous Bit-Width LLM via Dimensional Lifting and Projection](model_compression/liftquant_continuous_bit-width_llm_via_dimensional_lifting_and_projection.md)**

:   LiftQuant 通过"高维 1-bit lattice → 低维 weight 空间投影"的 lift-then-project 机制，把 LLM 量化 bit-width 从离散整数（2/3/4 bit）解耦为连续分数（如 2.4-bit），让 70B 模型精准塞进 24GB 显卡且 PPL 显著优于 2-bit baseline，整个解码路径只用线性变换 + 1-bit 均匀量化器，硬件友好。

**[LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](model_compression/lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)**

:   本文指出推测解码训练时长期用 KL 散度作为接受率的 proxy 是次优的——小容量 draft 模型在有限容量下 KL 最小化不蕴含接受率最大化；提出 LK losses（直接最大化负 log 接受率 + 与 KL 的 trust-region 混合）作为 plug-in 替代，4 个 draft 架构 × 6 个 target 模型（8B-685B）一致提升 8-10% 平均接受长度。

**[LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](model_compression/llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)**

:   本文把 LLM 训练重新解释为 Shannon-Hartley 噪声信道——参数量对应带宽、训练 token 对应信号功率、数据/模型噪声对应信道噪声；从该框架推出 Shannon Scaling Law $C_{\text{LLM}} = aN^\alpha \log_2(1 + bD^\beta / (c(DN)^\gamma + dD^\delta + e))$，能统一解释经典单调 scaling 与近期发现的 U 形退化（catastrophic overtraining、quantization-induced degradation），并在 Pythia/OLMo2 上从 ≤6.9B 数据外推到 12B 模型 307B token 上 $R^2 = 0.847$。

**[Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds](model_compression/memory-efficient_partitioned_dnn_inference_on_resource-constrained_android_crowd.md)**

:   本文给出 CROWDio 框架中"DNN 流水线调度子系统"的设计：在不修改模型本身（不剪枝、不量化、不蒸馏）的前提下，把一个完整 ONNX 模型按层切成多段，分发到 RAM 仅 3.3–7.4 GB 的多台 Android 手机上做流水线推理，靠 **JIT 延迟加载 + 单分区驻留约束 + 4 级亲和度调度 + zlib 压缩张量传输 + 流式 1:1 依赖** 五条机制把每台设备峰值 RSS 压到 $43\pm 2$ MB，并让批延迟比传统屏障同步快 34%。

**[MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](model_compression/mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)**

:   本文提出 MIC，在 Matryoshka 表征学习 (MRL) 之上加两个几何正则——SCR (限制 prefix/residual 子空间之间的相关) 和 SIR (强制 prefix 的方差均匀 + 超球面均匀)，让模型在被截断到 16/32/64 维这种极低维度时仍保持高判别性，平均超越 MRL/ESE 等基线。

**[Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](model_compression/mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)**

:   本文提出 C2R 框架，把数据集蒸馏中的鲁棒性问题重新拆解成"最小鲁棒边距"问题，用"攻击感知课程 (AAC) + 对比鲁棒损失 (CRL) + 线搜索 PGD (LS-PGD)"三件套，让合成集训练出的模型在六种攻击上平均比之前的鲁棒蒸馏 SOTA 高出约 2.8% 鲁棒准确率。

**[Model Merging Scaling Laws in Large Language Models](model_compression/model_merging_scaling_laws_in_large_language_models.md)**

:   作者用 10,866 个合并模型实测出一条形如 $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ 的双轴幂律：基座规模 $N$ 决定 floor，专家数 $k$ 决定 tail，且四种主流合并方法（Average、TA、TIES、DARE）都共用同一条曲线，从而把"合多少个专家、合到哪一步停"变成一个可预测、可预算的工程问题。

**[Multi-Adapter Representation Interventions via Energy Calibration](model_compression/multi-adapter_representation_interventions_via_energy_calibration.md)**

:   MARI 指出现有「表征干预」方法都依赖一个线性表征假设——单条全局 steering 向量加到所有输入上——既因为最优校正方向随样本剧烈变化而不可靠，又会在良性输入上误伤通用能力；它把单 adapter 换成多个低秩 adapter 并用「竞争训练 + 熵路由」做样本自适应干预，再用一个独立训练的低秩 probe 算「传播能量」做阈值门控，决定是否启用干预，从而在 TruthfulQA/BBQ/Safety 上大幅领先 ReFT、同时在 MMLU/ARC 上不掉甚至略升。

**[NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](model_compression/nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)**

:   NanoQuant 把权重量化重新表述为「低秩二值分解」问题，用 Hessian 感知的 ADMM 精确初始化 $\pm 1$ 因子和浮点尺度，再做块级 STE 重建与全局尺度 KL 校准，在仅 0.26M token 校准数据和单卡 H100 上首次让 PTQ 把 LLM 压到真正的 1-bit 乃至亚 1-bit，把 Llama2-70B 从 138 GB 压到 5.35 GB 并能跑在 8 GB 消费级 GPU 上。

**[NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](model_compression/neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)**

:   本文指出主流后训练量化 (PTQ) 方法都沿用了 Min-Max 公式来初始化 scale 与 zero-point，而这套老公式隐含了"由极值决定参数 + zero-point 必须是整数"两个被长期忽视的约束；作者提出 NeUQI，用"给定 scale 解析地求最优 zero-point + 由粗到细搜 scale"两步把约束打掉，在 LLaMA-2 7B 2-bit 通道量化下把 C4 困惑度从 SOTA 的 47.55 (MagR) 砍到 17.50，并使轻量蒸馏后超越成本高得多的 PV-tuning。

**[OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](model_compression/osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)**

:   OSAQ 利用 LLM 各层 Hessian 在不同输入下保持一致的低秩零空间，将零空间向量线性组合成一个加性权重扰动 $\Delta W$，在不改变二阶任务损失的前提下把离群权重「自吸收」掉，使 2 比特仅权重量化的困惑度比朴素 GPTQ 降低 40% 以上。

**[Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing](model_compression/parameters_as_experts_adapting_vision_models_with_dynamic_parameter_routing.md)**

:   作者把"参数本身当成专家"——在每个 stage 维护一个跨层共享的可训练参数矩阵池 (shared expert center)，让每一层的 ParaX 适配器通过一个轻量路由器为当前输入**动态合成**低秩投影和多尺度深度卷积的权重，从而同时解决传统 adapter 的"输入无关"和"跨层冗余"两大缺陷，在密集预测任务上以 <5% 可训练参数稳定超越 full fine-tuning。

**[Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](model_compression/partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)**

:   作者提出 **Partial Fusion**：用部分最优传输 (partial OT) 只合并两个网络中"最相似"的神经元、保留剩余神经元独立存在，从而在"权重聚合 (1× 参数量)"与"全集成 (2× 参数量)"之间得到一条平滑、单调、可调的精度–参数量曲线；并进一步把它统一到"对集成做广义剪枝"的视角，让同一套工具也能压缩单个模型。

**[Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers](model_compression/plug-and-play_spiking_operators_breaking_the_nonlinearity_bottleneck_in_spiking_.md)**

:   作者把 Transformer 里最难脉冲化的三个非线性算子（Softmax、SiLU、RMSNorm）拆成"除法 / 指数 / $\ell_2$ 范数"三个公共原语，分别用 LIF 神经元群体计算 + 位移缩放实现成 spike-friendly 模块，再像积木一样拼回原算子，全程不需要任何微调就能即插即用到现有 ANN-to-SNN 流水线里，对 LLaMA-3-8B / Qwen3-8B / BERT 等模型的精度损失 <1%。

**[Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](model_compression/preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)**

:   作者提出 SRR（Structured Residual Reconstruction），把 QER（Quantization Error Reconstruction）中固定用于补偿量化残差的低秩预算 $r$ 显式地拆成"先保留 $k$ 个主奇异方向再量化"和"用 $r-k$ 个秩去拟合残差"两部分，并给出一个只需一次随机探针的闭式准则来逐层选 $k^\star$，在 2/3 bit PTQ 和 QPEFT 上一致优于 LQER/QERA。

**[PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](model_compression/prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)**

:   PRISM 把 CLIP / SAM / DINOv2 三个异质视觉基础模型蒸馏进同一个 ViT 学生时,用"双流条件 MoE"（一条共享 anchor 流稳梯度、一条上下文路由的稀疏专家流解冲突）让专家自组织地分工——共识知识共享、冲突知识分支,在 PASCAL-Context 上比此前 SOTA SAK 在全部 5 个任务上都更好。

**[Procedural Pretraining: Warming Up Language Models with Abstract Data](model_compression/procedural_pretraining_warming_up_language_models_with_abstract_data.md)**

:   在标准语言/代码/数学预训练之前插入一段极轻量的"程序化数据"（形式语言、栈、元胞自动机等）"预热"，仅 0.1–0.3% 额外 token 就能稳定提升下游性能，并让模型用 55–86% 的原始数据复现同等 loss——是一种把"推理脚手架"和"知识"解耦的预训练策略。

**[ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](model_compression/projq_project-and-quantize_for_adapter-aware_llm_compression.md)**

:   ProjQ 把 PTQ 的量化噪声主动"塑形"到一个低秩子空间里、并把这部分让给后续 LoRA 适配器去消除，从而保住 LoRA 容量学下游任务，在 LLaMA-2 / Qwen2.5 / Qwen3 上用 3 bit 就能追平标准 4 bit baseline。

**[Provably Learning Attention with Queries](model_compression/provably_learning_attention_with_queries.md)**

:   作者证明单头 softmax attention 在 value-query 访问下可以惊人简洁地被精确恢复 —— 只需 $O(d^2)$ 次查询，比同等结构的 ReLU MLP 容易得多；当头维 $r\ll d$ 时还能借压缩感知降到 $O(rd)$，并把结论扩展到带噪 oracle、membership query 以及多头不可识别性。

**[QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](model_compression/qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)**

:   QHyer 用 Normalizing Flows 估计的状态依赖 Q 值取代 Decision Transformer 中的轨迹依赖 RTG，再叠加门控式 Attention-Mamba 混合骨干以实现内容自适应的历史压缩，在 OGBench/D4RL 的非马尔可夫与马尔可夫离线目标条件 RL 数据集上同时刷新 SOTA。

**[Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](model_compression/quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)**

:   Singular Value Ensemble（SVE）把"集成多样性"做成纯粹由 SVD 奇异值的不同重新加权来表达——冻结预训练权重的左右奇异向量（共享的"知识基底"），只为每个集成成员训一组独立的奇异值，参数开销 $\lesssim1\%$ 而校准质量接近真正的 Deep Ensemble，把 UQ 带进了 PEFT 友好的资源受限场景。

**[RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs](model_compression/rabit_residual-aware_binarization_training_for_accurate_and_efficient_llms.md)**

:   本文针对残差二值化 LLM 中"并行二值路径学到冗余特征"这一被作者命名为 inter-path adaptation 的失败模式，提出 RaBiT——用单一共享的全精度权重在线派生所有二值路径并配合函数感知初始化，从而结构性地强制残差层级，使 2-bit Llama2-7B 在 matmul-free 架构下首次反超 VQ 强基线（Wiki2 PPL 5.78 vs QTIP 5.86），同时获得 4.49× 推理加速。

**[ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](model_compression/respinquant_efficient_layer-wise_llm_quantization_via_subspace_residual_rotation.md)**

:   ReSpinQuant 在低比特 LLM PTQ 中同时保留"全局旋转可与权重融合"和"层间旋转可适配各层离群点"两大优点，靠的是把残差连接处不可消去的旋转过渡矩阵 $\mathbf{T}=\mathbf{R}_{out}\mathbf{R}_{in}^{\top}$ 用一个秩 $r\!\approx\!32$ 的子空间正交近似替代，在线开销只增加 $\sim0.2\%$，W4A4/W3A3 上同时压过 SpinQuant 和 FlatQuant。

**[RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](model_compression/rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)**

:   RQ-MoE 用「两级 MoE + 双流量化」的设计，让残差向量量化（RQ）的码本随输入动态生成，又通过把指令流与重建流解耦实现 6–14× 解码加速，在四个 retrieval benchmark 上 MSE/Recall 持平或超越 QINCo。

**[Saliency-Aware Model Merging](model_compression/saliency-aware_model_merging.md)**

:   SA-Merging 把结构化剪枝里的 SynFlow 连接性分数搬到数据无关模型合并场景，对每个专家的 task vector 计算"端到端通路敏感度 × 聚合方向一致性"作为显著度，迭代地去掉低显著度更新，从而在视觉/语言/LoRA 多任务上把数据无关 merging 推到接近 test-time adaptation 的水平。

**[ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](model_compression/scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)**

:   作者证明 LoRA 累加更新被困在固定低秩子空间，提出 ScaLoRA：每步把旧 $AB^\top$ 合并到 $W^{pt}$ 后，**用一个可解析求得的最优"列缩放"** 重启 adapter，使 AdamW 一阶/二阶动量可以 $O((m+n)r)$ 等变传递 (不需要重置/warm-up)、累加更新自然变高秩——在 DeBERTaV3、LLaMA2-7B、LLaMA3-8B、Gemma3-12B 上一致打过 LoRA / MoRA / HiRA / ReLoRA / LoRA-GA。

**[Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers](model_compression/selective_coupling_of_decoupled_informative_regions_masked_attention_alignment_f.md)**

:   MaskAQ 把 ViT 的数据无关量化重新定义为"在合成样本的稀疏 informative region 上对齐全精度模型 $P$ 与量化模型 $Q$ 的注意力"，用差分熵最大化解耦前景 patch、用自适应掩码对齐注意力、并以周期性刷新让样本跟随 $Q$ 演化，在 3-bit DeiT-T 上把 ImageNet Top-1 比此前最佳再抬 3.1%。

**[Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](model_compression/semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)**

:   本文先用新基准 KVFundaBench 系统揭示「检索类长上下文压得动、推理类压不动」的关键不对称，并把原因归结到 KV 压缩破坏了少样本示例这一「语义单元」的完整性；据此提出 ShotKV——在 prefill 阶段保留整个 shot 作为不可分割单元、在 decoding 阶段做动态 token 级压缩，让 LG-GSM8K 在 40% 压缩率下从 baseline 46.0 提升到 47.33，并在长输入设置下端到端延迟降低 11.3%。

**[SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](model_compression/speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)**

:   SPEED-Bench 是一个面向投机解码（Speculative Decoding, SD）的统一基准，它通过 *Qualitative split*（最大化语义多样性的 880 条样本）与 *Throughput split*（按 1k–32k 输入长度桶组织、覆盖三档熵的大批量数据）配合一套对接 vLLM / TensorRT-LLM / SGLang 的测量框架，揭示了过去 SD 论文里被"小数据 + 单批 + HuggingFace"评测掩盖的真实部署行为。

**[SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](model_compression/surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)**

:   SURGE 给每个二值化层并联一个"全精度辅助分支"，前向输出不变但反向能从全精度分支额外回传一份"非 STE 截断"的高阶梯度，并用 AGS 按梯度范数比动态平衡两路贡献，让 BNN 在 ResNet-18/ImageNet 上做到 62.0% top-1，比 ReCU 高 1.0%、比 IR-Net 高 3.9%。

**[Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](model_compression/task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)**

:   LoDA 把 LoRA 的下投影矩阵按「投影能量」拆成一个跨任务共享的通用子空间和一个真正只激活新任务的隔离子空间，再用梯度对齐训练上投影、并在融合时给通用分支闭式重标定，从而在多个持续学习 benchmark 上稳定刷过现有 LoRA-CL 方法。

**[The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](model_compression/the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)**

:   作者发现 LLM 蒸馏里"软标签 + 硬标签线性混合"几乎总是打过纯软标签，并把原因从直觉上的"硬标签信息更少但优化更易"修正为"硬标签压低了暴露偏差"，进一步用 Bridge-Garden 分解把生成序列拆成"必须精确的桥"和"可灵活替换的花园"两类位置，从而把 mix coefficient 与上下文风险绑定起来，提出 4 种自适应混合策略并以 9.7× 训练成本优势在 7 对 teacher-student 上超越主流 on-policy / divergence-based KD。

**[The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models](model_compression/the_shape_of_addition_geometric_structures_of_arithmetic_in_large_language_model.md)**

:   作者在 Qwen3-4B 的最后一层残差流里发现 LLM 做多操作数加法时，激活被组织成「数字盆 × 进位纤维」的分层流形，并把"算错一位"重新解释成沿着等原始和轨迹（IRST）滑过一个连续进位势的量化阈值，由此提出双流一致性检查，在推理期把"内部还知道但输出选错"的 off-by-one 错误纠回来。

**[Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)**

:   作者发现 token 的"重要性"在层间和头间剧烈变化，传统 token eviction 一次性删除是不可逆的早期决策错误；他们提出 Token Sparse Attention，每层每个 attention head 独立选 $L' \ll L$ 个 token 做密集 attention，输出再 scatter 回原始序列长度，配上残差路径让被略过的 token 在下一层重新有机会被选中——既保留头/层级动态选择，又能直接调用 FlashAttention 等密集 kernel，在 128K 上下文上叠加 FlexPrefill 后达到 ×3.23 注意力加速、精度损失 <1%。

**[Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](model_compression/toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)**

:   本文识别出对抗训练数据中存在一个跨方法稳定的「鲁棒不可学习集」，并通过两层网络的特征学习理论证明：当强鲁棒教师在这些样本上给出高置信监督时，会迫使学生记忆伪噪声进而触发鲁棒过拟合，反之教师在这些样本上保持高熵即可抑制噪声梯度，由此给出基于不可学习样本预测熵的教师选择准则。

**[Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](model_compression/towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)**

:   作者搭了一套基于 NVML 的分阶段 GPU 能耗采集框架，把蒸馏流水线拆成"教师侧 + 学生侧 + 评估"逐段计量，发现一次性运行时教师 logit 缓存 / 合成数据生成才是大头，让 KD 和 synthetic SFT 在 1B–13B OLMo-2 学生上反而比直接 SFT 多耗约 $2.4\times$ 能量，并给出闭式 break-even 公式说明只有当教师产物被复用 $N^*$ 次以上时蒸馏才真"省电"。

**[Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](model_compression/towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)**

:   作者用神经网络无穷宽缩放理论推出 steering vector 的 factor / direction 联合训练应满足 $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$ 这一缩放约束，从而消掉推理时人工选 $\alpha$ 的环节；同时受 ReFT 启发只在前 4 个 prompt token 上做加性干预（PrOSV），在 AxBench 上既能维持模型实用性，又能在三档 Gemma2/Qwen2.5 模型上一致超过全序列 FSSV。

**[T3S: 训练轨迹感知的 token 选择，破解推理蒸馏的「Imitation Shock」](model_compression/training-trajectory-aware_token_selection.md)**

:   论文发现强 student（如 Qwen3-8B）继续从 DeepSeek-R1 蒸馏时存在通用的「Imitation Shock」——loss 单调下降但 acc 先暴跌再恢复，根因是早期「Imitation-Anchor Tokens」梯度统治优化压制了真正负责推理的 token；T3S 用训练轨迹找出 anchor token 并把它们 mask 掉，让 yet-to-learn 推理 token 提前学习，在 AR 和 dLLM 两个 setting 都涨分（Qwen3-8B 反超 DeepSeek-R1，Qwen3-32B 逼近 Qwen3-235B，LLaDA-2.0-Mini 反超 AR baseline 拿到 16B no-think SOTA）。

**[Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](model_compression/turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)**

:   本文把"陈旧的"块循环坐标下降梯度估计存进 FIFO buffer，配上 momentum 衰减重用，并证明这等价于带 warm-start 的 BCCD；同时给出反直觉结论——更大的有限差分步长 $\epsilon$ 会隐式平滑 loss landscape、降低有效 Lipschitz 常数，从而让 stale gradient 反而能换来稳定下降。

**[UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-Adaptive Federated Fine-tuning of Foundation Models](model_compression/ub-smoe_universally_balanced_sparse_mixture-of-experts_for_resource-adaptive_fed.md)**

:   作者发现把 Sparse MoE 直接搬进异构联邦 LoRA 微调会出现「专家利用失衡」与「Top-K 不可导」两个致命问题，并通过 Dynamic Modulated Routing (DMR) 重平衡专家激活、Universal Pseudo-Gradient (PG) 给未激活专家补伪梯度，构成自强化循环，使低算力客户端在节省 45% 计算的同时性能提升 8.7×。

**[When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](model_compression/when_shared_knowledge_hurts_spectral_over-accumulation_in_model_merging.md)**

:   这篇论文指出模型合并不仅会有任务冲突，还会把跨任务共享的谱方向重复累加成过大的奇异值，并提出无需训练和数据的 Singular Value Calibration，在不改奇异向量的情况下重标定奇异值，从而稳定提升视觉与语言任务的合并效果。

**[WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](model_compression/winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)**

:   WinQ 把低比特语言模型 QAT 的慢收敛解释为权重陷在低曲率鞍点附近，并用周期性权重量化插值重初始化加噪声扰动梯度，在几乎不增加训练开销的情况下把 1-2 bit QAT 加速到 1.5-4 倍，并在相同训练预算下提升多种 LLaMA/Qwen 量化配置的困惑度和零样本准确率。

**[WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](model_compression/wush_near-optimal_adaptive_transforms_for_llm_quantization.md)**

:   WUSH 为 LLM 的 weight-activation 低比特量化推导出闭式、数据自适应的 blockwise 线性变换，把 Hadamard 的均匀扩散能力和权重/激活二阶统计结合起来，在 W4A4 尤其是 MXFP4 场景下显著提升精度且几乎不牺牲 FP4 kernel 吞吐。

**[xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](model_compression/xkv_cross-layer_kv-cache_compression_via_aligned_singular_vector_extraction.md)**

:   xKV 发现 LLM 不同层的 KV-cache 虽然逐 token 余弦相似度不高，但主奇异向量高度对齐，因此用跨层共享低秩基同时压缩多层 KV-cache，并结合选择性重构在长上下文推理中取得最高 8 倍压缩和 4.23 倍端到端吞吐提升。

**[ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling](model_compression/zipmoe_efficient_on-device_moe_serving_via_lossless_compression_and_cache-affini.md)**

:   ZipMoE 面向移动和边缘设备上的 MoE 大模型推理，把 BF16 专家参数拆成可压缩的 exponent bits 与高熵 sign-mantissa bits，通过无损压缩、分层缓存和 cache-affinity 调度，把原本受 SSD I/O 卡住的专家加载改造成可被多核 CPU 并行隐藏的解压与重组流程，在不改模型语义的前提下降低延迟并提升吞吐。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Adaptive Bandit Algorithms for Contextual Matching Markets](reinforcement_learning/adaptive_bandit_algorithms_for_contextual_matching_markets.md)**

:   本文研究带上下文的在线匹配市场，把玩家对动态 arm context 的线性偏好作为 bandit 学习目标，提出适用于 stochastic contexts 的 BARB 和 adversarial contexts 的 AdECO，并给出 player-optimal stable regret 的自适应上界与紧的 $\tilde O(T^{2/3})$ 级别理论结果。

**[Agent Learning via Early Experience](reinforcement_learning/agent_learning_via_early_experience.md)**

:   本文提出 early experience 范式，让语言 Agent 在没有外部奖励的情况下利用自己尝试动作后的未来状态学习环境动态和决策反思，从而在 8 个 Agent 环境中稳定超过纯模仿学习，并为后续 GRPO 强化学习提供更好的初始化。

**[ALSO: Adversarial Online Strategy Optimization for Social Agents](reinforcement_learning/also_adversarial_online_strategy_optimization_for_social_agents.md)**

:   ALSO 把 LLM 社会智能模拟中的动态策略选择建模为对抗在线 bandit，并用轻量级奖励代理模型从对话历史中泛化稀疏反馈，在 Sotopia-Hard 上把整体分数从 3.02 提升到 3.53，尤其显著改善关系维度。

**[ASAP: Exploiting the Satisficing Generalization Edge in Neural Combinatorial Optimization](reinforcement_learning/asap_exploiting_the_satisficing_generalization_edge_in_neural_combinatorial_opti.md)**

:   ASAP 发现神经组合优化中“找出一组有希望的动作”比“直接选中唯一最优动作”更容易跨分布泛化，并用 proposal-selection 两阶段策略和 MAML 初始化让 3D-BPP、TSP、CVRP 神经求解器在分布变化时更稳、更快适配。

**[Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas](reinforcement_learning/beyond_scalar_rewards_dense_feedback_for_llm_policy_synthesis_in_sequential_soci.md)**

:   提出 LLM 迭代策略合成框架，让 LLM 直接生成 Python 策略代码用于多智能体序贯社会困境，并通过"反馈工程"证明：在标量奖励基础上加入效率/公平/可持续/和平四项社会指标作为 dense feedback，可以打破"反馈混叠"问题，在 Cleanup 游戏中实现高达 54% 的效率提升。

**[Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](reinforcement_learning/beyond_the_proxy_trajectory-distilled_guidance_for_offline_gflownet_training.md)**

:   提出 TD-GFN，一种无需代理奖励模型的离线 GFlowNet 训练框架，通过逆强化学习从离线轨迹中提取边级奖励，再经 DAG 剪枝与优先反向采样间接指导策略学习，同时保证梯度更新仅依赖真实终端奖励，在分子设计和序列生成等任务上显著超越现有基线。

**[Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](reinforcement_learning/bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)**

:   提出 PANDA 算法，通过基于 Nikaido-Isoda 函数的惩罚重构，用纯一阶策略梯度方法求解下层为正则化零和马尔可夫博弈的双层 RL 问题，达到 $\tilde{O}(\epsilon^{-1})$ 迭代复杂度和 $\tilde{O}(\epsilon^{-3})$ 样本复杂度，匹配单策略下层 BRL 的最优已知速率。

**[Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](reinforcement_learning/break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)**

:   针对扩散语言模型 (dLLM) 半自回归生成时"块大小固定"破坏推理逻辑链的问题，本文提出 b1：用 RL 学一个块结束指示 token 来生成动态长度块，并用一个"块级熵单调下降 (Monotonic Entropy Descent, MED) 奖励"驱动连贯推理，作为即插即用的奖励项接入现有 dLLM RL 框架（Diffu-GRPO/GDPO/d1/wd1），在 Countdown 上将 wd1 从 39.45 推到 58.98。

**[CAMEL: Confidence-Gated Reflection for Reward Modeling](reinforcement_learning/camel_confidence-gated_reflection_for_reward_modeling.md)**

:   本文观察到 verdict token 的 log-probability margin 与判断正确率高度相关，据此提出 CAMEL —— 先用单 token 快速给出偏好判断，仅在低置信度时才触发反思生成，并用反事实前缀增强 GRPO 训练自我纠错能力，在三个奖励模型 benchmark 上以 14B 参数取得 82.9% 的平均准确率（超过此前最佳 70B 模型 3.2%）。

**[Can Large Language Models Generalize Procedures Across Representations?](reinforcement_learning/can_large_language_models_generalize_procedures_across_representations.md)**

:   本文发现 LLM 在符号表征（代码/图）上学到的过程知识无法可靠迁移到自然语言任务，提出"先符号后自然语言"的两阶段 RL 课程学习策略，使 1.5B Qwen 模型在异步规划任务上接近 zero-shot GPT-4o，并从认知科学角度证明成功的跨表征泛化可被解释为生成类比。

**[Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](reinforcement_learning/chebyshev_policies_and_the_mountain_car_problem_reinforcement_learning_for_low-d.md)**

:   本文首次解析求解了经典 Mountain Car 最优控制问题（36 年未解），揭示出最优策略形式极简（$\alpha = C \cdot \dot{x}$）而现有 RL 智能体存在惊人高的遗憾值，进而提出基于多元 Chebyshev 多项式的策略参数化方法，在参数量减少 277 倍的同时将遗憾值降低 4.18 倍。

**[Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning](reinforcement_learning/compositional_transduction_with_latent_analogies_for_offline_goal-conditioned_re.md)**

:   本文提出 CTA（Compositional Transduction with latent Analogies），通过将目标到达任务分解为"任务内生类比"和"任务外生上下文"两个独立因子，利用时间距离差分场作为类比表示，并结合双线性转导实现对未见类比—上下文组合的外推，在 OGBench 操控环境上平均性能超过最强基线约 42%。

**[Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](reinforcement_learning/convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)**

:   本文提出一种比 Zhang 等 $(L_0,L_1)$-NS 更广的非均匀光滑性 $(H_0,H_1)$-NS，并在此假设和 (非均匀) Łojasiewicz 条件下首次给出确定性对角 RMSProp / Adam 与一般最速下降 (Sign GD、Norm.GD、Sign CD-GS) 的统一收敛率，证明在分离数据上的逻辑回归与 softmax 策略梯度上它们比 GD / AdaGrad / heavy-ball 都严格更快。

**[Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](reinforcement_learning/convergence_of_two-timescale_markovian_stochastic_approximations_with_applicatio.md)**

:   本文首次在 Markov 噪声且不依赖任何投影算子的条件下，建立了一般两时间尺度随机逼近 (SA) 的稳定性与几乎必然收敛性，并据此给出离策略线性函数逼近下 TDC($\lambda$) 算法的首个几乎必然收敛结果。

**[Coupled Variational Reinforcement Learning for Language Model General Reasoning](reinforcement_learning/coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)**

:   CoVRL 把"用回答概率当奖励"的 verifier-free RL 重写成一个变分推断问题，构造一个"先验 (只看问题) + 后验 (看到答案)"的复合分布，并用混合采样 + 重要性加权同时优化两者，使 Qwen2.5-7B 在 9 个通用与数学推理基准上相对 base 平均涨 12.4%，比最强 verifier-free 基线再涨 2.3%。

**[CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](reinforcement_learning/cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)**

:   把 self-play 从"对抗"换成"协作": Coach 出题、Player 解题、Coach 拿"Player 进步幅度 × Player 解题率"作为奖励, 在完全不用外部训练数据的条件下让 Qwen2.5-Math-7B-Instruct 在六个数学 benchmark 上总均分 +4.9、OOD +5.4, 超过 RENT/R-Zero 等已有 unsupervised 方法。

**[d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](reinforcement_learning/d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)**

:   本文为掩码扩散语言模型（masked DLM）提出 d2 强化学习框架，核心是用两种"轨迹似然估计器"（d2-AnyOrder 对支持 any-order 解码的模型给出单次前向的精确估计，d2-StepMerge 对标准 MDM 给出可调精度的近似估计）来正确实现 GRPO，使 LLaDA-8B-Instruct 在 Sudoku/Countdown/GSM8K/MATH500 上分别达到 91.9% / 56.6% / 85.0% / 41.6%，全面超越 d1、wd1 等扩散 RL 基线。

**[D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](reinforcement_learning/d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)**

:   D$^2$Evo 在每一轮 RL 迭代里都用当前 Solver 估计难度、挑出中等难度真实样本作为锚点，再训练 Questioner 围绕锚点合成同等难度的新题，从而以 < 2K 真实数学题就在数学和通用推理上同时超过用 19K 真实数据训练的 GRPO 基线。

**[DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning](reinforcement_learning/darts_distribution-aware_active_rollout_trajectory_shaping_for_accelerating_llm_.md)**

:   DARTS 把 LLM RL 训练的 rollout 长尾瓶颈从"调度绕开"重新定义成"主动塑形分布"，通过 intra-prompt 冗余采样 + 双端长度采样 + 方差驱动的冗余预算分配，把模型的 rollout 长度分布显式压短压紧，在 Qwen 系列 3B–32B 模型上相比 VeRL 取得最高 1.77 倍加速，同时不损失下游精度。

**[Data- and Variance-dependent Regret Bounds for Online Tabular MDPs](reinforcement_learning/data-_and_variance-dependent_regret_bounds_for_online_tabular_mdps.md)**

:   本文针对已知转移的在线 episodic tabular MDP，基于带 log-barrier 的乐观 follow-the-regularized-leader（OFTRL）设计了单一 best-of-both-worlds 算法，同时给出对抗 regime 下的一阶/二阶/路径长度三种数据依赖 regret 上界，以及随机 regime 下的方差感知 gap-无关和 gap-相关 polylog 界，并配套证明匹配的下界。

**[DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](reinforcement_learning/debiased_model-based_representations_for_sample-efficient_continuous_control.md)**

:   DR.Q 在 MR.Q 类"模型化表示 + actor-critic"骨架上加两件事——用 InfoNCE 显式最大化 $z_{sa}$ 与下一状态表示 $z_{s'}$ 的互信息，再用"PER × forget"融合的 faded prioritized replay 缓解早期经验过拟合——在 73 个连续控制任务上用单一超参组击败 SimBaV2 / MR.Q / TDMPC2 等强基线。

**[Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance](reinforcement_learning/decoupling_skeleton_and_flesh_efficient_multimodal_table_reasoning_with_disentan.md)**

:   本文为多模态表格推理提出两件套：训练阶段 DiSCo 用结构匿名化把"骨架"和"血肉"两个对齐目标解耦，让 LVLM 只用 10K 表格图就学会布局；推理阶段 Table-GLS 用"全局结构探查 → 自反思子表抽取 → 证据稳态推理"三步把整图问答压缩到最小可验证子表上，整套无需推理数据微调也不调外部工具，在 21 项 benchmark 上超过依赖 82K-97K 标注的 SFT/RL 基线。

**[Distributional Inverse Reinforcement Learning](reinforcement_learning/distributional_inverse_reinforcement_learning.md)**

:   本文提出 DistIRL：在离线逆强化学习中把奖励本身建模为条件分布、把"专家比学习者更优"这一约束从期望升级到一阶随机占优 (FSD)，并用失真风险度量 (DRM) 把 FSD 难解的 0/1 指示函数松弛成可优化的风险加权目标，从而首次系统地从离线示范中同时学到完整奖励分布与分布感知策略。

**[Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](reinforcement_learning/dr_tulu_reinforcement_learning_with_evolving_rubrics_for_deep_research.md)**

:   Dr. Tulu 提出 RLER（Reinforcement Learning with Evolving Rubrics），让评估 rubric 在训练过程中与策略共同演化，把 RLVR 从短答案 QA 推广到带引用的长文深度研究任务，最终用 Qwen3-8B 训出的 DR Tulu-8B 在四个长文 deep research 基准上平均超 Tongyi DR-30B 15.6 个点，并以 1000 倍更低成本达到 OpenAI Deep Research 同等水平。

**[EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance](reinforcement_learning/eapo_enhancing_policy_optimization_with_on-demand_expert_assistance.md)**

:   EAPO 把"咨询外部专家"作为一个可学习的离散动作嵌入策略空间，让 LLM 在 RL 训练阶段按需调用更强模型获取中间提示，并通过逐步衰减的接受率把专家知识内化到自身策略中，在评测时完全独立推理仍能在 AIME / AIMO 等数学推理基准上稳定超越纯自探索 RL。

**[EchoRL: Reinforcement Learning via Rollout Echoing](reinforcement_learning/echorl_reinforcement_learning_via_rollout_echoing.md)**

:   本文指出 RLVR 训练后期 GRPO 类方法因为一组 rollout 全部成功导致优势归零、梯度消失（advantage degeneration），提出 EchoRL：从 verified-success rollout 里基于**步级熵峰值**挑出"最艰难却走通了的"前缀 EchoClip，作为辅助 SFT 项加到 loss 上，在 4 个 RLVR 框架、5 个 backbone、10 个 benchmark 上稳定带来最高 5.6%/5.0% 的 ID/OOD 提升。

**[FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](reinforcement_learning/fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)**

:   本文提出 FAB——首个面向时变有向图分布式双层优化的纯一阶算法，将 AB/Push-Pull 通信与值函数惩罚法相结合，给出非渐近 $\mathcal{O}(K^{-2/3})$ 收敛率，并顺带解决了 AB/Push-Pull 在时变有向图非凸场景下收敛率长期悬而未决的开放问题。

**[Flow-Equivariant World Models: Memory for Partially Observed Dynamic Environments](reinforcement_learning/flow_equivariant_world_models_memory_for_partially_observed_dynamic_environments.md)**

:   FloWM 通过在隐空间中利用**时间参数化对称性**（流等变）维持结构化动态记忆——解决部分可观测环境下对象越界后失踪的问题，使长视野预测精度远超扩散和循环基线（3D Block World 210 步预测 SSIM 0.9525 vs DFoT 0.8885）。

**[From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](reinforcement_learning/from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)**

:   本文把离线偏好强化学习 (PbRL) 改写在 Forward-Backward (FB) 表示空间里，证明在 FB 框架下标准的 Bradley-Terry 偏好损失等价于 SimCLR 对比损失，从而提出 FB-PbRL：先在无奖励离线数据上预训练 FB 表示，再在偏好数据上用对比目标搜索任务向量 $\boldsymbol{z}^\star$ 并微调表示，整个流程不再训练任何显式奖励或偏好模型。

**[Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](reinforcement_learning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)**

:   本文把 LLM 主动提问场景（20 Questions / 医疗诊断 / 故障排查）建模成两人零和扩展式博弈 (EFG)，提出 Game of Thought (GoT)：用深度有限的子博弈构造 + CFR 求 Nash 均衡来产生“随机化提问策略”，在所有数据集上把 worst-case 交互轮数显著降低，且 weighted 变体下相对 UoT 提升 15–40%。

**[Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)**

:   本文指出主流 PSRO 在扩展策略种群时只看"受限博弈"的局部信息会导致最坏情况下要加入近 $N$ 个纯策略才能收敛，因此提出一个先采样多条候选最优响应、再用 *post-expansion Population Exploitability (PE)* 直接打分挑选最佳扩展的两阶段探索-选择框架 Global PSRO，并通过参数共享的条件策略网络把多候选训练和评估的代价压到可接受范围。

**[Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning](reinforcement_learning/hista_and_numca_estimate_state_value_effectively_for_llm_reinforcement_learning.md)**

:   本文先用一个新建的 State Value Estimation Benchmark (SVEB) 实证 PPO critic 在 LLM RL 里几乎完全退化为 GRPO 的组均奖励 baseline，再提出两种以"无须额外 rollout、几乎零额外算力"为目标的状态价值估计方法 —— Numca 用数字 milestone 把数学推理重写为目标条件 RL 做信用分配，Hista 用 LLM 末层隐状态 + MinDistance 做概率加权奖励平均 —— 在五大类 SVEB 子集都把 MAE 降到 GRPO/PPO 之下，并让 DAPO/CSIPO 等强算法在多个数学基准上拿到一致提升。

**[How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](reinforcement_learning/how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)**

:   作者把"训 LLM 学下国际象棋"当成可验证 RL 的干净实验台，系统比对 6 类自制 SFT 数据集对 RL 的影响，发现"直接预测最佳一步 (Best Move)"得最高分但 RL 后产生不忠实推理，"预测多步最佳走法 (Best Line)"性能相当但 RL 更稳、推理更忠实；并提炼出三条可用 SFT-checkpoint 预测 RL 终局性能的指标，最终用 7B 模型在多个国象 benchmark 上超过 gpt-oss-120b。

**[InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](reinforcement_learning/inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)**

:   本文把"迭代推理 + 显式摘要"这一推理范式从纯 SFT 升级到端到端 RL，提出 InftyThink+：用轨迹级 GRPO 同时优化"何时摘要、保留什么、如何续推"三个决策，并配以效率奖励，在 DeepSeek-R1-Distill-Qwen-1.5B 上把 AIME24 准确率提升 21%、延迟降低 32.8%。

**[Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](reinforcement_learning/interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)**

:   本文从信息论视角出发，把多智能体之间的"互相影响"用条件互信息刻画出来，再设计同时遮挡观测和扰动动作的攻击器去最小化跨组互信息，并据此训练出能在协作部分崩溃时仍能稳定决策的 IBAL 策略，在 SMAC / SMACv2 / LBF 等多种攻击与"队友缺失"扰动下都显著超过既有鲁棒 MARL 方法。

**[LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation](reinforcement_learning/labo_llm-accelerated_bayesian_optimization_through_broad_exploration_and_selecti.md)**

:   本文提出 LABO，把 LLM 当作"低保真度"评估源接入贝叶斯优化循环——用 Kennedy–O'Hagan 联合高斯过程把真实实验 $f_R$ 分解为缩放的 LLM 预测 $\rho f_L$ 加上残差过程 $\delta$，再用"差异主导率" $p_\Delta = \sigma_\delta^2/(\rho^2\sigma_L^2 + \sigma_\delta^2)$ 做门控决定每个候选要不要花真实实验，从而用近乎免费的 LLM 查询广撒网、把昂贵真实实验集中到 LLM 不可信的区域，在 COF、Fullerene 等 6 个科学优化任务上同等真实预算下显著超过 vanilla BO 与 LLAMBO、BOPRO、CAKE。

**[Laplacian Representations for Decision-Time Planning](reinforcement_learning/laplacian_representations_for_decision-time_planning.md)**

:   本文提出 ALPS，将图 Laplacian 的特征向量空间（缩放后近似 commute-time distance）作为分层决策时规划的潜空间，先用 k-means 在该空间发现子目标并跑 Dijkstra 生成高层路径，再用带行为先验的 CEM 在原始状态空间做短程低层规划，在 OGBench 的离线目标条件 RL 上首次让基于模型的规划方法系统性击败 model-free SOTA。

**[LASER: Learning Active Sensing for Continuum Field Reconstruction](reinforcement_learning/laser_learning_active_sensing_for_continuum_field_reconstruction.md)**

:   把"该把稀疏传感器放哪"建模成一个 POMDP，用一套包含编码器、GRU、扩散动力学预测器和隐式神经场解码器的"连续场潜世界模型"提供 imagined 下一步潜状态作为策略条件，再用 GRPO + 动态 group 过滤 + 多步前瞻奖励训练交叉注意力策略，在 Navier-Stokes / 浅水方程 / 真实海表温度（SST）三个数据集的稀疏感知重建任务上一致打败固定布局和 offline-optimized 布局。

**[Latent Representation Alignment for Offline Goal-Conditioned Reinforcement Learning](reinforcement_learning/latent_representation_alignment_for_offline_goal-conditioned_reinforcement_learn.md)**

:   通过把 goal-conditioned 价值函数显式参数化为 **非对称潜空间中的负欧氏距离** $V(s,g)=-\|\varphi_S(s)-\varphi_G(g)\|_2$ 并配合连续性正则与 HIQL 分层结构，LAVL 在 OGBench 22 个数据集里拿下 20 个 SOTA，把 giant 迷宫和 stitch 数据集这类长程任务的成功率从基线的几乎为零拉到 80%+。

**[Learning in Structured Stackelberg Games](reinforcement_learning/learning_in_structured_stackelberg_games.md)**

:   本文给"上下文 Stackelberg 博弈"加上一条结构性假设（context→follower type 的映射来自某个假设类 $\mathcal{H}$），并构造出两类全新的学习论维度——刻画在线悔界的 Stackelberg-Littlestone 维度 SLdim 与刻画 PAC 样本复杂度上下界的 $\gamma$-SG / $\gamma$-SN 维度——证明它们能严格胜过多类 Littlestone / Natarajan 维度，给出实例最优的在线算法 SSOA 和分布式算法 $\mathfrak L^*$。

**[Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory](reinforcement_learning/learning_query-aware_budget-tier_routing_for_runtime_agent_memory.md)**

:   BudgetMem 把"运行时智能体记忆抽取"重新组织成"过滤 → 实体/时间/主题并行 → 摘要"的模块化流水线，给每个模块挂上 LOW/MID/HIGH 三档预算接口，并用 PPO 训练一个共享的轻量路由器在 query 到来时为每个模块挑档位，从而在 LoCoMo、LongMemEval、HotpotQA 上同时改善了 F1/Judge 和单 query 平均成本。

**[Learning to Approximate Uniform Facility Location via Graph Neural Networks](reinforcement_learning/learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)**

:   本文为 Uniform Facility Location 设计了一个把经典近似算法 SimpleUniformFL 神经化的 MPNN，**既可用无监督期望成本损失端到端训练，也具备 $\mathcal{O}(\log n)$（递归版还能到 $\mathcal{O}(1)$）的可证明近似界**，实验上既打过 SimpleUniformFL 经典算法、也逼近 ILP 最优。

**[Learning to Bet for Horizon-Aware Anytime-Valid Testing](reinforcement_learning/learning_to_bet_for_horizon-aware_anytime-valid_testing.md)**

:   本文把"在严格观测上限 $N$ 下设计 anytime-valid 序贯检验"重新表述为一个状态空间为 $(t,\log W_t)$ 的有限期最优控制问题，从理论上证明 Kelly 下注在"按时进度"的中间带最优、落后时该激进、超前时该保守，得到一张三区"相图"，并用一个在大量合成 Beta 分布上训练的统一 DQN agent 自动学到与相图一致的状态依赖策略，在合成与真实数据上同时拿到更高的 deadline-内拒绝率与更窄的置信序列，同时通过 Ville 不等式保持 anytime-validity。

**[Learning to Route Languages for Multilingual Policy Optimization](reinforcement_learning/learning_to_route_languages_for_multilingual_policy_optimization.md)**

:   本文提出 LRPO（Language-Routed Policy Optimization），把"用哪个语言生成 rollout"当作可学习变量，用一个上下文 bandit 形式的语言路由器为每条训练样本在固定 rollout 预算下挑选最有信息量的语言组合，并通过离线估计 + 在线校准的跨语言相似度奖励把多语言 rollout 拉到同一个尺度上做 GRPO，在 Qwen/Llama/Gemma 三族骨干、五个多语言基准上稳定优于 GRPO 与各种 dominant-language 基线。

**[Learning to Search and Searching to Learn for Generalization in Planning](reinforcement_learning/learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)**

:   本文提出 GSP：一种把加权 A* 最佳优先搜索和 Q-learning 套在同一循环里、以关系图神经网络表达 $Q_\theta(s,a)$ 的"自改进广义规划器"，仅在小规模实例上训练就能零样本泛化到比训练时大十几倍的实例（如 Blocksworld 从 ≤30 块到 488 块），在多个 IPC 基准、Sokoban、PushWorld、The Witness 上同时刷新覆盖率并超越基于实时搜索的 DRL 基线。

**[Learning Unmasking Policies for Diffusion Language Models](reinforcement_learning/learning_unmasking_policies_for_diffusion_language_models.md)**

:   本文把掩码扩散语言模型的解码过程显式建模为一个 MDP，用 GRPO 训练一个仅以 token 置信度为输入、参数量不到底座模型 0.01% 的单层 Transformer 策略，自适应地决定每一步要 unmask 哪些位置，在 semi-AR 设定下追平 Fast-dLLM 等手工启发式，在 full-diffusion 设定下显著反超并展现跨模型、跨任务、跨长度的迁移性。

**[LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](reinforcement_learning/llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)**

:   本文提出 LMAC——用 LLM 离线为合作型 MARL 设计可执行的通信协议代码，依据"状态可重建性"指标做两轮反馈迭代（先提高重建准确度，再降低跨智能体的不平衡），在 SMAC-Comm、LBF、GRF、SMACv2 等基准上显著超过 TarMAC/SMS/T2MAC/MASIA 等通信基线，部分场景甚至超过把全局状态喂给所有智能体的 QMIX+State 上界。

**[Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](reinforcement_learning/long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)**

:   本文挑战“离线 RL 必须显式保守”的主流共识，提出 Neubay：用贝叶斯视角看后验上的模型集合、用**长 horizon rollout（数百步）**自然吸收价值高估、用 layer norm 与不确定度阈值控制 compounding error，从而在 D4RL/NeoRL 共 33 个数据集上不靠悲观惩罚就追平 SOTA 保守算法，并在 7 个数据集上刷新纪录。

**[Making Expert Reasoning Learnable with Self-Distillation](reinforcement_learning/making_expert_reasoning_learnable_with_self-distillation.md)**

:   DAIL 用一个"教师 = 看过专家解的自己 + 学生 = 没看过专家解的自己"的混合策略 rollout，把不到 1000 条专家解题轨迹改写成与学生策略分布一致的推理链，再用对比损失压低"只看中间答案的负参考模型"高概率的捷径 token，在 Qwen2.5-Instruct / Qwen3 上拿到最高 31% 的 pass@128 提升，并把所需推理 token 砍掉一半。

**[MFPO: 用 Few-step MeanFlow Policy 把 MaxEnt RL 跑到接近 Gaussian policy 的速度](reinforcement_learning/mean_flow_policy_optimization.md)**

:   MFPO 用 MeanFlow models（学 average velocity 而非 instantaneous velocity）当 RL policy 把扩散策略采样步数从 20+ 降到 2 步，用 average divergence network 解决 action likelihood 计算、用 ESS-weighted SNIS 组合 Gaussian + policy proposal 解决 soft policy improvement，在 MuJoCo/DMC/HumanoidBench 上性能 ≥ diffusion baseline 且训练时间降 ~50%。

**[Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](reinforcement_learning/metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)**

:   把多轮 jailbreak 重新形式化为推理时的策略优化问题——在 adversarial POMDP 框架下，Attacker 与 Metacognitive Evaluator 构成闭环：Evaluator 输出的密集分析反馈被当作「语义梯度」来引导 Attacker 的 belief 更新与策略改进，从而在不重新训练任何权重的情况下，对包括 O1 / GPT-5-chat / Claude-3.7 在内的 10 个前沿模型平均 ASR 89.2%，token 消耗较强 baseline 平均降低 8.2 倍。

**[Mind Dreamer: Untethering Imagination via Active Causal Intervention on Latent Manifolds](reinforcement_learning/mind_dreamer_untethering_imagination_via_active_causal_intervention_on_latent_ma.md)**

:   本文为模型基强化学习（MBRL）提出 Mind Dreamer，用一个对抗式生成器在世界模型已学到的隐空间流形上"跳跃"到非历史轨迹覆盖的关键锚点，并通过新设计的 Relay Value/Uncertainty 函数（含 $\gamma^2$ 折扣）解决跨断点的信用分配，在 DMC 上相对 DreamerV3 平均提速 $1.67\times$、稀疏奖励任务最高提速 $8.8\times$。

**[MindZero: Learning Online Mental Reasoning with Zero Annotations](reinforcement_learning/mindzero_learning_online_mental_reasoning_with_zero_annotations.md)**

:   MindZero 把贝叶斯逆向规划改写成一个对多模态 LLM 的「自监督 RL」目标——奖励模型生成的心智假设使观察到的人类动作似然最大，再用 GRPO 训练，使小模型在不需要任何心智标注的前提下实现单次前向、快速且鲁棒的在线心智推理。

**[MoMa QL: 用矩匹配加速扩散/流匹配策略的离线 + 离线-在线 RL](reinforcement_learning/moment_matching_q-learning.md)**

:   MoMa QL 用 Maximum Mean Discrepancy 替代标准 BC 损失，把扩散/流匹配策略的多步采样压缩为单步或少步的"边际保持插值"采样器，在 D4RL 上 Gym 平均归一化分 95.5 全面领先 Diffusion-QL（87.9），同时因为采样快得多，offline-to-online 微调时也比一致性 AC、Diffusion-QL 提升更大。

**[Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](reinforcement_learning/multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)**

:   SeqComm-DFL 把"多智能体通信"作为预测器、把"联合策略选择"作为下游优化器，用价值感知的消息生成 + Stackelberg 序贯条件 + 隐式微分双层优化把通信学习直接对齐到团队回报，在医院调度和 SMAC 上取得 4-6 倍的累积奖励提升与 >13 个百分点的胜率提高。

**[Offline Reinforcement Learning with Generative Trajectory Policies](reinforcement_learning/offline_reinforcement_learning_with_generative_trajectory_policies.md)**

:   本文用「连续时间 ODE 解映射」把扩散策略、Flow Matching、一致性策略统一为同一族「生成轨迹策略 (GTP)」，再加上一个对齐离线样本的闭式 score 近似与一个优势加权的训练目标，使策略在 D4RL 上既能少步采样、又能在 AntMaze 等硬任务上拿到接近满分的成绩。

**[Offline Reinforcement Learning with Universal Horizon Models](reinforcement_learning/offline_reinforcement_learning_with_universal_horizon_models.md)**

:   作者把"几何视界模型 (GHM) 只能采样一个固定折扣分布"这个限制打开，提出能在任意视界 $n$ 上直接采样未来状态的 universal horizon model (UHM)，再用 Winsorized 几何分布把过长视界截断，在 OGBench 100 个任务上比最强基线平均成功率提升约 14%。

**[One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](reinforcement_learning/one_bias_after_another_mechanistic_reward_shaping_and_persistent_biases_in_langu.md)**

:   本文系统测量五个高质量 RM（含 SOTA Skywork-Reward-V2）的长度、不确定性、位置、谄媚、模型风格五类偏置，把它们划分为"低复杂度（线性可修）"和"高复杂度（线性不可修）"两类，并提出 mechanistic reward shaping —— 用 DiffMean 线性探针在最后一层隐藏态上做零空间投影 —— 在不掉 RewardBench2 准确率的前提下显著缓解前三类偏置且能 OOD 泛化到 best-of-N。

**[ORLoopBench: Solver-in-the-Loop Benchmarks for Self-Correction and Behavioral Rationality in Operations Research](reinforcement_learning/orloopbench_solver-in-the-loop_benchmarks_for_self-correction_and_behavioral_rat.md)**

:   作者把"修一个 Infeasible 的运筹模型"形式化成"每改一步都要重跑 Gurobi 拿 IIS 反馈"的求解器在环 MDP，发布了配套基准 ORLoopBench（5362 条 LP/MILP 修复实例 + 库存决策偏差测评），并用 RLVR 把一个 8B 模型训到在 LP 修复上以 95.3% RR@5 反超闭源 API（92.4%）。

**[PAC-Bayesian Reinforcement Learning Trains Generalizable Policies](reinforcement_learning/pac-bayesian_reinforcement_learning_trains_generalizable_policies.md)**

:   本文给出第一个**显式依赖马尔可夫链混合时间**且对长 horizon $1/(1-\gamma)$ 依赖只到一次方的 PAC-Bayesian RL 泛化上界，并把它作为活的训练目标内嵌进 SAC，得到 PB-SAC 算法——在 MuJoCo 连续控制任务上同时给出非空 (non-vacuous) 部署证书与具竞争力的性能。

**[Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](reinforcement_learning/parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)**

:   本文给出第一个针对**无约束**在线凸优化、**时变移动代价**与**动态比较序列**三重设定的 parameter-free 算法，把延迟反馈与时变记忆都规约为带时变移动代价的 OCO，从而统一刷新这三个场景的动态遗憾上界。

**[Perceptual Flow Network for Visually Grounded Reasoning](reinforcement_learning/perceptual_flow_network_for_visually_grounded_reasoning.md)**

:   摒弃"用视觉专家的精确框做硬监督"的传统 RLVR 路线，PFlowNet 把感知行为本身建模为一段结构化的 Perceptual Flow 潜变量，用变分分布 $p_\theta(Z|X)$ 近似面向推理的理想后验，并用 Sub-TB 变分 RL + 多维奖励 + 邻域几何整形 (Vicinal Geometric Shaping) 训练，使得 8B 的 Qwen3-VL 在 V* Bench 拿到 90.6%、MME-RealWorld-lite 67.0% 的新 SOTA。

**[Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](reinforcement_learning/plug-and-play_benchmarking_of_reinforcement_learning_algorithms_for_large-scale_.md)**

:   本文提出 FluidGym——首个完全用 PyTorch 实现、无外部 CFD 求解器依赖、端到端可微、原生支持多智能体与 3D 流场的 RL 主动流控基准，用 PPO/SAC/TD-MPC/DPC 在 13 个 2D/3D 环境上跑出 25k+ GPU 小时的标准化结果。

**[Position: Deployed Reinforcement Learning should be Continual](reinforcement_learning/position_deployed_reinforcement_learning_should_be_continual.md)**

:   本文是一篇立场论文：作者主张凡是部署后仍能拿到评价性奖励信号、且环境复杂度超出 agent 表征/计算能力的 RL 系统，本质都是一个持续强化学习（CRL）问题，应当抛弃"训练完就冻结"的范式，让 agent 在部署中持续更新策略。

**[Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](reinforcement_learning/practical_and_optimal_algorithm_for_linear_contextual_bandits_with_rare_paramete.md)**

:   在线性 contextual bandit 上，作者把过去"batched"一词隐式混淆的"何时拿到 reward"和"区间内能否依赖到达的 context"两个轴显式拆开，定义出"rare parameter updates"（只限制 reward-driven 参数更新次数、允许 reward-free 的 context 自适应）这个更贴近实际部署的设定；并据此提出两个仅需 $\mathcal O(\log\log T)$ 次参数更新的算法 BLCE-G 和 BLCE，前者首次在小-$K$ 与大-$K$ 两个 regime 同时达到 minimax-optimal regret $\widetilde{\mathcal O}(\sqrt{dT\log K}\wedge d\sqrt T)$，后者更进一步彻底去掉 G-optimal design 这个主算力瓶颈、把 runtime 砍到所有 optimal 算法中最低；并把同一思想扩到广义线性 bandit，得到不依赖最坏曲率参数 $\kappa$ 的 BGLE。

**[Probing RLVR Training Instability through the Lens of Objective-Level Hacking](reinforcement_learning/probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)**

:   作者提出"objective-level hacking"框架,把 MoE 大模型在 RLVR 中训练-推理差异越训越大的现象归因为 token 级权重失真在优化目标里引入的有偏伪信号,并在 30B MoE 上通过四组实验验证"偏差(不是方差)才是元凶"。

**[ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation](reinforcement_learning/prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rectifie.md)**

:   针对"主动推荐"任务中朴素策略梯度坍缩到"等长重复路径"的问题，作者从理论上把失败归因为路径级奖励分解后的正均值步级奖励所诱导的"长度捷径"和过高方差，并提出 ProRL：用 Stepwise Reward Centering 把每步期望奖励减去常值基线、消除长度偏置，再用 Position-Specific Advantage Estimation 按步位置做 GRPO 式分组基线降低方差，三个真实数据集上 IoI、IoR、CTR、Coherence 四指标全面超过启发式、监督式与 LLM 式 SOTA。

**[Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](reinforcement_learning/provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)**

:   本文为「先易后难」的课程式 RL 后训练给出第一份严格的样本复杂度证明：在 transformer 的状态条件自回归推理树上，若课程能让相邻阶段的难度比保持在目标难度的 $L/p$ 次根级别，则总样本数可从直接训练的指数级 $(C^\star)^L$ 降到课程版的多项式级 $L\cdot (C^\star)^{p_\max}$。

**[Quantifying and Optimizing Simplicity via Polynomial Representations](reinforcement_learning/quantifying_and_optimizing_simplicity_via_polynomial_representations.md)**

:   作者提出用"沿数据插值路径拟合 Chebyshev 多项式"作为神经网络的低维函数空间代理，并定义"有效次数"（Effective Degree, ED）—— 对系数加绝对值再乘多项式阶数 —— 作为衡量"函数有多简单"的标量；它在 CIFAR-10/ImageNet/CLIP 上比 sharpness、参数 $L_2$ 范数等已知泛化代理更准地预测泛化间隔，并且整条估计 pipeline 可微，可直接当做训练时的"简单性正则项"，在图像、文本、CLIP 微调与 RL 四类任务上一致带来增益。

**[Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation](reinforcement_learning/randomized_advantage_transformation_rat_computing_natural_policy_gradients_via_d.md)**

:   通过 Woodbury 恒等式把 Tikhonov 正则化的自然策略梯度改写为"带有变换后优势的普通策略梯度"，再用随机分块 Kaczmarz 迭代在 mini-batch 上求解这个优势变换，从而完全绕开 Fisher 矩阵的显式构造、共轭梯度内循环以及 KFAC 那类架构相关的曲率近似——只用一次标准反向传播就能拿到自然策略梯度，并在 MuJoCo 与 Procgen 上匹配或超过 TRPO/ACKTR/KFAC 的表现。

**[Reinforced Sequential Monte Carlo for Amortised Sampling](reinforcement_learning/reinforced_sequential_monte_carlo_for_amortised_sampling.md)**

:   本文把分层变分推断、MaxEnt 强化学习与序列蒙特卡洛/退火重要性采样统一到一个框架下，让学到的策略与流函数同时充当 SMC 的提议核与扭曲目标，再反过来用 SMC 产生的近目标样本作为离策略行为策略训练神经采样器，并配合自适应权重温度和重要性加权经验回放，在多模目标与 alanine dipeptide 玻尔兹曼分布上同时改善了模式覆盖与训练稳定性。

**[Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality](reinforcement_learning/reinforcement_learning_for_reachability_guaranteeing_asymptotic_optimality.md)**

:   本文针对未知 MDP 上的可达性规约学习问题，提出一个分阶段细化 PAC 参数的直接学习算法，证明以概率 1 存在有限阶段 $K_{\mathsf{opt}}$，此后只输出最优策略，并用内在 MDP 参数显式刻画该阶段，在量化验证基准上经验地证实最优策略可在极少阶段（中位数 $k=2$）内出现。

**[Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games](reinforcement_learning/revisiting_regularized_policy_optimization_for_stable_and_efficient_reinforcemen.md)**

:   KLENT 把 reverse-KL 正则（控制策略更新幅度）+ 熵正则（维持探索）+ λ-return（平衡偏差方差）这三件成熟"老零件"重新组合到自博弈 model-free RL 里，在 5 个棋类上达到比 Gumbel AlphaZero 高 4 倍训练效率，并给出 normal-form 与 finite-length 两种场景下的收敛性证明。

**[RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](reinforcement_learning/rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)**

:   本文提出 RL-SPH —— 一种不依赖外部 ILP 求解器、能独立产出 100% 可行解的端到端强化学习启发式算法，用「可行性奖励 + 双阶段策略 + 可行性感知邻域搜索」让 Graph Transformer Agent 在包含非二元整数变量的 ILP 上把 primal gap 平均降低 28.6 倍。

**[RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](reinforcement_learning/rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)**

:   RL4RLA 用"难度递增的数值课程 + 蒙特卡洛图搜索 (MCGS)"驱动一个 RL agent 从线性代数原语里组合出可解释的随机数值线性代数 (RLA) 算法，成功重现了 Sketch-and-Precondition、Randomized Kaczmarz、Newton Sketch 等经典方法。

**[RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning](reinforcement_learning/ruleplanner_all-in-one_reinforcement_learner_for_unifying_design_rules_in_3d_flo.md)**

:   本文把 3D 芯片 floorplanning 中七类工业设计规则统一塞进一个 actor-critic RL 框架：核心是把每条规则编译成一张 $W\times H$ 的"邻接矩阵掩码"，在策略 softmax 之前用大负数把违规位置直接屏蔽掉，再加上离散位置 + 连续长宽比的混合动作空间和 Transformer 编码的网表特征，让单一智能体首次能同时满足边界、分组、跨层对齐、非重叠等七条规则，并对未见电路有零样本迁移能力。

**[Safe In-Context Reinforcement Learning](reinforcement_learning/safe_in-context_reinforcement_learning.md)**

:   本文首次把"安全约束"引入 in-context 强化学习（ICRL），提出 SCARED：在预训练阶段用一个**单乘子 + 取正号 hinge** 的精确罚 Lagrangian 让 Transformer 策略学会在测试时**不更新任何参数**地依靠 cost-to-go 上下文做 CMDP 适应，在 OOD 网格 / MuJoCo / Velocity 基准上 reward 单调上升、cost 单调下降，并能随用户给的预算 $\delta$ 在保守与激进之间平滑切换。

**[Safe Reinforcement Learning with Preference-Based Constraint Inference](reinforcement_learning/safe_reinforcement_learning_with_preference-based_constraint_inference.md)**

:   本文提出 PbCRL，用一个带"死区"的扩展 Bradley-Terry 偏好模型从轨迹比较中学安全约束，再叠加一个信噪比正则避免代价函数被压平，最后用两阶段（离线预训练 + 在线少量标注微调）训练打通 Safe RL 的完整流水线，在 Safety Gymnasium、自动驾驶与语言模型对齐三类任务上既显著降代价、又保住奖励。

**[Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](reinforcement_learning/safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)**

:   作者在 UVA-Padova 物理模型基础上搭了一个统一的 T1D/T2D 糖尿病模拟器，发现 8 种主流 Safe RL 算法虽然在训练病人上能满足安全约束，但部署到未见病人时 Time-in-Range 普遍掉 8–13%，于是提出用 Basis-Adaptive Neural ODE 预测血糖轨迹、再用预测性屏蔽 (predictive shielding) 在测试时过滤危险动作，让 PPO-Lag / CPO 等基线在 OOD 病人上重新拿回 13–14% TIR。

**[Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](reinforcement_learning/shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)**

:   作者把合作博弈论里的 Shapley 值搬到卷积神经网络的"滤波器"级别，用 Monte Carlo + 截断 + 多臂老虎机三重近似估计每个 Neuron 的连续重要度排名，然后冻结 Top-$r\%$ 的"专家"Neuron、留下其余继续可塑训练，从而在不存储样本、不扩展架构的前提下把 ImageNet-1k 上类增量学习的精度比第二名 buffer-free 方法再提升 $+2.88\%$、任务增量提升 $+6.46\%$。

**[Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](reinforcement_learning/single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)**

:   SHIFT 用一次贪心解码下的"开始 token → 结束 token"隐状态差 $\Delta(x)=\mathbf{e}(x)-\mathbf{s}(x)$ 同时充当 RLVR 样本的效用代理和多样性特征，再用质量加权的最远优先 CoreSet 在大规模无标注池里挑出极少量样本，全过程不训练、不需要奖励或答案。

**[SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](reinforcement_learning/sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)**

:   本文把 MoE 策略在持续强化学习中的可塑性丢失形式化为 empirical NTK 矩阵谱熵有效秩的下降，再用 Gauss-Newton 与 Kronecker 分解把它降维到一个只依赖"专家特征 Gram 矩阵"的可计算 proxy，最后用一个一行的 Parseval 罚（SPHERE）拉高这个 proxy，在 MetaWorld 和 HumanoidBench 持续 RL 设置下把任务成功率分别提升 133% 和 50%。

**[Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](reinforcement_learning/stochastic_minimum-cost_reach-avoid_reinforcement_learning.md)**

:   本文提出 Reach-Avoid Probability Certificate (RAPC), 用一个 max-min-夹紧的 Bellman 收缩算子让值函数下界 reach-avoid 概率, 配合一个对抗 $\gamma^T$ 衰减的 "补偿因子"作归一化, 再用对称梯度投影联合优化 "成本"与 "reach-avoid 概率"两个冲突目标, 在 MuJoCo 上同时拿到比 RC-PPO / RESPO / CPPO 更低的累积成本和更高的 reach 成功率。

**[The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes](reinforcement_learning/the_obfuscation_atlas_mapping_where_honesty_emerges_in_rlvr_with_deception_probe.md)**

:   本文构造 MBPP-Honeypot 这一会自然诱发奖励黑客 (hardcode 测试用例) 的 RLVR 环境, 系统地刻画了"用白盒欺骗探针 (deception probe) 当训练信号"会得到的四类策略——诚实 / 露骨欺骗 / 混淆策略 (obfuscated policy) / 混淆激活 (obfuscated activations), 并证明只要 KL 正则系数 $\beta$ 与探针惩罚系数 $\alpha$ 都足够大, 就能在奖励黑客场景下稳定收敛到诚实策略。

**[The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models](reinforcement_learning/the_shape_of_reasoning_topological_analysis_of_reasoning_traces_in_large_languag.md)**

:   本文把 LLM 的 chain-of-thought 看作嵌入空间中的"点云", 用拓扑数据分析 (TDA) 提取持续同调特征作为推理质量的客观度量, 在 AIME 数据集上证明 TDA 特征对 Smith-Waterman 对齐分数的预测能力 (平均 $R^2=0.236$) 显著高于传统图统计 (平均 $R^2=0.064$)。

**[The Surprising Difficulty of Search in Model-Based Reinforcement Learning](reinforcement_learning/the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)**

:   作者反直觉地指出 model-based RL 中搜索失败的根因不是模型不准，而是 MPC 行为策略与价值函数训练策略不一致引发的过估计偏差，并提出在 10 个价值函数集成上"取最小"的 MRS.Q 算法，在 50+ 个连续控制任务上稳定超过 TD-MPC2、BMPC、BOOM、SimbaV2 等 SOTA。

**[Extra-CoT：极端压缩比下的思维链压缩框架](reinforcement_learning/towards_efficient_large_language_reasoning_models_via_extreme-ratio_chain-of-tho.md)**

:   Extra-CoT 提出一个三阶段框架（语义保持压缩器 → 混合比率SFT → 层次化奖励RL），在极端压缩比（保留仅20%的token）下仍能维持推理精度，在MATH-500上实现73%的token缩减同时精度提升0.6%。

**[Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning](reinforcement_learning/tracking_drift_variation-aware_entropy_scheduling_for_non-stationary_reinforceme.md)**

:   AES 把最大熵 RL 的探索强度调度问题投影到在线凸优化的动态遗憾框架，导出"熵权应与环境漂移幅度的平方根成正比"的硬理论结果，再用 TD 误差分位数作为可观测漂移代理实现完全在线的算法不可知熵调度——在 SAC / PPO / SQL / MEow 四种框架 + 12 个任务上，激变恢复时间普遍减半。

**[Trajectory-Level Data Augmentation for Offline Reinforcement Learning](reinforcement_learning/trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)**

:   本文提出 LIFT：在主动定位任务里，利用轨迹几何性质把次优 logging policy 留下的冗余 zig-zag 轨迹"抄近道"成 shortcut，并把这些合成 transition 喂给一个轻量增广器在数据采集期间替换 logging 动作，使离线 CQL 在低维到高维、partial obs 等各种设置下显著超越普通离线 RL 与 warm-start SAC。

**[Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges](reinforcement_learning/turning_bias_into_bugs_bandit-guided_style_manipulation_attacks_on_llm_judges.md)**

:   把 LLM 评判器已知的风格偏好（冗长、列表、emoji 等）当作可被系统性利用的攻击面，作者将攻击建模为上下文老虎机，用 LinUCB 在 25 次查询预算内自适应挑选 8 种语义保持的风格改写动作，对 5 个主流评判器实现 >65% 攻击成功率、+1~2 分（满分 9）的分数膨胀，且绕过 style control 防御。

**[跨域离线强化学习中统一值对齐与值分配](reinforcement_learning/unifying_value_alignment_and_assignment_in_cross-domain_offline_reinforcement_le.md)**

:   本文在异质跨域离线强化学习设定下揭示"值误分配"问题——源数据来自多个域和多个策略时，优势函数评估不准导致数据筛选失效。提出 V2A 框架通过时间一致的模态表示学习与模态感知的优势学习来统一解决值对齐和值分配问题，性能比 DVDF 提升 21.4%。

**[Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](reinforcement_learning/unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)**

:   作者把"地面街景与卫星图能否定位为同一坐标"作为可验证间接奖励，用 GRPO 对 Qwen2.5-VL-7B 做两阶段后训练（CoT scaffolding + RL self-exploring），让模型仅凭 GPS metadata 学到可零样本迁移到 25+ 地理空间任务的通用推理能力。

**[Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](reinforcement_learning/vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)**

:   本文研究"在 N 个智能体的大规模 MARL 系统中挑出 K 个最脆弱的智能体"这一双层 NP-hard 问题，把它建模为 HAD-MFC（Hierarchical Adversarial Decentralized Mean Field Control），用 Fenchel-Rockafellar 变换把下层最坏对抗策略的训练折叠成一个加正则项的"鲁棒 mean-field Bellman 算子"，再把上层组合选择问题转化为带稠密 reward 的 MDP 用贪心或 RL 求解，证明分解保持最优性，在 18 个任务中 17 个超 baseline。

**[视觉工具使用强化学习究竟学到了什么？](reinforcement_learning/what_does_vision_tool-use_reinforcement_learning_really_learn_disentangling_tool.md)**

:   本文提出 MED 框架系统分析视觉工具使用 RL 在裁剪-放大场景中的实际学习效果——发现 RL 训练所带来的性能提升主要源于**内在能力提升**而非工具掌握能力提升，模型主要学会了如何与工具安全共存而非真正掌握工具。

**[You Can Learn Tokenization End-to-End with Reinforcement Learning](reinforcement_learning/you_can_learn_tokenization_end-to-end_with_reinforcement_learning.md)**

:   本文把 byte-level LLM 中“哪里画 token 边界”建模成离散随机决策，用带 early-exit relative reward、time discounting 和 batch-relative advantage 的 score function estimator 端到端学习 tokenization，在 147M 自然语言模型和 90M 代码模型上优于直通估计器并接近 BPE-guided downsampling。

---

## 🔬 可解释性 { #interpretability }

**[A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](interpretability/a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)**

:   这篇论文提出一种把行为评估和内部表示探针结合起来的 LLM Agent 目标导向性评估框架，并在 GPT-OSS-20B 的网格导航任务中发现：行为上它大体按目标行动，内部也编码了粗粒度空间地图和短期计划，但会被无功能的目标状物体诱导。

**[A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](interpretability/a_deep_learning_model_of_mental_rotation_informed_by_interactive_vr_experiments.md)**

:   这篇论文用 VR 交互实验约束模型设计，提出一个由 3D 等变空间编码器、神经符号对象编码器和动作决策 MLP 组成的心理旋转模型，在准确率、动作次数和部分反应时趋势上复现了人类 mental rotation 行为。

**[Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis](interpretability/accurate_evaluation_of_quickest_changepoint_detectors_via_non-parametric_surviva.md)**

:   本文把在线最快变点检测中的 ARL/ADD 评估改写成右删失生存分析问题，用 Kaplan-Meier 曲线估计有限且不规则长度序列下的检测时间和检测延迟，从而比传统只统计已触发样本的估计器更稳健、更少偏。

**[Adaptive Querying with AI Persona Priors](interpretability/adaptive_querying_with_ai_persona_priors.md)**

:   作者把"LLM 在 persona 条件下产生的回答分布"打包成一个有限混合的贝叶斯先验，让用户在仅被问几道题的情况下，通过对 persona 后验做闭式更新来高效预测其他回答，性能上压过经典 CAT/IRT 基线。

**[All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](interpretability/all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)**

:   这篇论文用 Overlap-Aware Sheaf Repulsion (OASR) 算法系统性地证伪了机理可解释性领域的隐含假设——"一个 LLM 能力对应一个独特的电路"——发现同一任务可被多个几乎不重叠 (IoU ~4–11%) 但都满足 faithful/sparse/complete 的电路或 sheaf 支撑，并给出"分布式稠密电路假设"作为理论解释。

**[Beyond Additive Decompositions: Interpretability Through Separability](interpretability/beyond_additive_decompositions_interpretability_through_separability.md)**

:   提出张量分离学习（TSL），一种将条件均值建模为正秩-1可分离乘积之差的逐阶段贪心回归方法，通过可分离结构避免加性分解在强交互下的信号抵消与交互遮蔽问题，同时其偏依赖函数可精确恢复拟合因子形状。

**[BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](interpretability/block-em_preventing_emergent_misalignment_via_latent_blocking.md)**

:   BLOCK-EM 用 SAE 找到一小撮"因果地控制 emergent misalignment"的内部 latent，然后在窄域 SFT 时加一个 one-sided 正则，禁止模型把这些 latent 朝"失对齐方向"放大——在 6 个 fine-tuning 域上把 emergent misalignment 平均砍掉 93%，同时几乎不损伤 in-domain 任务表现。

**[Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](interpretability/breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)**

:   提出 SimpliPy（基于规则的化简引擎，比 SymPy 快 100 倍）和 Flash-ANSR（基于 Transformer 的摊销符号回归框架），在 FastSRB 基准上以 ~58% 的恢复率匹敌甚至超越遗传编程方法 PySR，同时随推理预算增加生成更简洁的表达式。

**[Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions](interpretability/bridging_the_knowledge-prediction_gap_in_llms_on_multiple-choice_questions.md)**

:   本文揭示 LLM 在多选题上普遍存在"知识-预测差距"——隐藏层已线性编码正确答案但最终预测却偏离，通过几何分析将该差距归因于知识子空间与预测子空间的错位，并提出 KAPPA 方法在推理时用闭式仿射变换对齐两个子空间，跨模型跨基准一致缩小差距并提升准确率。

**[CB-SLICE: Concept-Based Interpretable Error Slice Discovery](interpretability/cb-slice_concept-based_interpretable_error_slice_discovery.md)**

:   CB-SLICE 利用概念瓶颈模型（CBM）的概念预测空间来发现和解释深度学习模型的系统性错误切片，通过三步流程——错误倾向概念筛选、GMM 聚类形成切片、关键词概念解释——在多个基准上一致性超越现有方法，同时提供直接扎根于模型内部决策逻辑的忠实解释。

**[Certified Circuits: Stability Guarantees for Mechanistic Circuits](interpretability/certified_circuits_stability_guarantees_for_mechanistic_circuits.md)**

:   提出 Certified Circuits 框架，通过基于删除的随机平滑为机械可解释性中的电路发现提供可证明的数据集级稳定性保证，使得发现的电路在概念数据集的有界编辑距离扰动下保持不变，从而产生更紧凑、更准确且 OOD 泛化更好的电路。

**[Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](interpretability/circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)**

:   本文提出 Circuit Fingerprint 假说——单独把答案 token 喂进 Transformer，它在隐空间留下的方向恰好就是产生该答案所要走的电路路径——并据此用纯几何对齐（无需梯度/干预）完成 circuit discovery，同时同一组方向反过来可以做 activation steering，证明"读"和"写"是同一个几何对象的两面。

**[Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement](interpretability/cognitive_fatigue_in_autoregressive_transformers_formalization_and_measurement.md)**

:   本文将自回归语言模型在长序列生成中的退化现象形式化为"认知疲劳"，提出 Fatigue Index (FI) 这一轻量级、模型无关的在线诊断指标，聚合 prompt 注意力衰减、表示漂移和熵失调三个信号，在 9 个模型上验证了 FI 对退化的预测能力（AUROC=0.976）并揭示了非单调的规模缩放行为。

**[CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)**

:   通过把生成时 token 上的 SAE 激活与任务正确性做 Pearson 相关来挑选可解释的引导特征, 用正样本均值激活直接当系数, 不需对比数据集也不需反向传播, 就能在 Gemma-2 2B / LLaMA-3.1 8B 上把 MMLU 提 +3.3%、HarmBench 提 +27.1%, 且副作用率比微调更低。

**[Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](interpretability/courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)**

:   本文提出"法庭辩论 (courtroom analogy)"视角，将分类的二阶不确定性建模为 $K$ 个类别辩护人 Dirichlet 意见在输入相关权重下的结构化混合，并实例化为 MoDEX 网络（共享证据 $\bm{\alpha}$ + 类专属辩护强度 $\tau_k$ + 可信度 $\bm{\omega}$ 三个轻量头），单次前向即可在 CIFAR/SVHN/TIN/CIFAR-10-C/CIFAR-10-LT 等基准上稳定刷过 EDL / $\mathcal{F}$-EDL 等一系列 baseline，并给出语义明确的不确定性分解。

**[Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory](interpretability/diagnosing_the_reliability_of_llm-as-a-judge_via_item_response_theory.md)**

:   本文把心理测量学里的 Item Response Theory (IRT) 中的 Graded Response Model (GRM) 搬到 LLM-as-a-Judge 上，把"评判分数"分解成评判者属性 $(\alpha, \beta)$ 与样本潜在质量 $\theta$，再用 4 个可解释指标分两阶段（内在一致性 + 人类对齐）系统诊断 7 个主流 LLM 在 11 类评判准则上"是不是一台稳定的测量仪器"。

**[Discovering Differences in Strategic Behavior Between Humans and LLMs](interpretability/discovering_differences_in_strategic_behavior_between_humans_and_llms.md)**

:   本文用 AlphaEvolve（基于 LLM 的程序合成框架）直接从行为数据里"进化"出可解释的 Python 行为模型，并在迭代石头剪刀布（IRPS）上对比人类与前沿 LLM，发现 Gemini 2.5 Pro/Flash 与 GPT 5.1 在胜率和"对手模型"维度上都明显超过人类，而 GPT OSS 120B 反而越打越差。

**[Discovering Implicit Large Language Model Alignment Objectives](interpretability/discovering_implicit_large_language_model_alignment_objectives.md)**

:   Obj-Disco 把 RLHF/GRPO 的不透明奖励信号沿"模型检查点轨迹"反向工程成稀疏的自然语言目标线性组合（DIR），通过 Matching Pursuit 式贪心 + LLM-as-Judge 双重校验，在多任务多模型上稳定恢复 >90% 的奖励行为，并能挖出"放松对违法行为讨论限制"这类隐藏的失配诱因。

**[Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis](interpretability/disentangling_direction_and_magnitude_in_transformer_representations_a_double_di.md)**

:   本文用 L2 匹配扰动协议，证明 Pythia 系列里方向（角度）扰动对语言建模 loss 的破坏力是同等位移幅值扰动的 42.9 倍，而幅值扰动对句法（主谓一致）的破坏远高于角度——这是一对认知神经科学意义上的 "双重分离"，对应方向走 attention 路径、幅值走 LayerNorm 路径。

**[Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](interpretability/dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)**

:   作者用可控的两层 Transformer + 合成 GMM 数据系统拆解了多模态 in-context learning 的训练数据条件与注意力电路，发现一个"主-次模态非对称"现象：在高多样性主模态上预训练后，次模态只需极低数据复杂度就能解锁多模态 ICL，并通过 head knockout 在 Qwen2.5-VL-3B 上验证了"induction head 主导多模态 ICL、多模态训练只是 refine 而非重建"的电路图景。

**[DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models](interpretability/dllm-jepa_joint_embedding_predictive_architectures_for_masked_diffusion_language.md)**

:   在掩码扩散语言模型 (masked diffusion LM) 的微调阶段加上 JEPA 表示对齐目标：把同一句话用不同掩码比例切成"低掩码上下文视图"和"高掩码目标视图"，仅对上下文视图做一次带梯度前向同时算扩散 loss 和 JEPA embedding、对目标视图用 EMA 副本无梯度前向，相比 LLM-JEPA 节省 33% 训练 FLOPs，并在 4 个任务 × 2 个 backbone 上稳定涨点（GSM8K 最高 +18.7 pp）。

**[Do Activation Verbalization Methods Convey Privileged Information?](interpretability/do_activation_verbalization_methods_convey_privileged_information.md)**

:   本文系统证明：当前流行的激活语言化方法（Patchscopes / LIT / SelfIE）在被用作 LLM 可解释性工具时，其性能完全可以由 "verbalizer 模型自己的知识" 解释，不需要任何 target 模型的内部激活——意味着这些工具在现有 benchmark 上看起来 work 是因为基准本身设计有缺陷，且当 verbalizer 知识超过 target 时会编造出 target 根本不具备的 "解释"。

**[Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](interpretability/dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)**

:   本文用 difference-in-means 在残差流里抽出 LLM 表达 10 个 Schwartz 价值时的 "intrinsic"（无系统提示）与 "prompted"（带价值系统提示）两类方向，再用 SVD 把两者拆成共享轴与各自独有轴，在向量层与 MLP 神经元层同时给出因果证据：共享分量承载真正的价值语义并能跨语言泛化、复现 Schwartz 圆环结构；intrinsic 独有分量带来词汇/语义多样性；prompted 独有分量编码的是与价值无关的"通用指令服从"通道，能直接把越狱攻击成功率从 13%–27% 推到 83%–97%。

**[Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning](interpretability/equilibrium_reasoners_learning_attractors_enables_scalable_reasoning.md)**

:   本文把"通过迭代更新潜变量做推理"的模型重新解释为一个学到的吸引子动力系统，提出 Equilibrium Reasoners (EqR)：用随机初始化 + 路径噪声两条轻量训练干预塑造吸引子地形，配合"深度（迭代步数 $D$）+ 广度（多次随机重启 $B$）"两轴测试时扩展和基于残差收敛的选轨规则，在训练时最多 16 步迭代的前提下，把 Sudoku-Extreme 的精确准确率从 feedforward 的 2.6% 推到 99.8%（等效展开 40,000 层）。

**[Expand Neurons, Not Parameters](interpretability/expand_neurons_not_parameters.md)**

:   在保持非零参数总数不变的前提下，把每个神经元"切"成 $\alpha$ 个稀疏子神经元、让它们瓜分原来的输入边，就能显著降低神经元之间的特征干扰（多义性），从而在 Boolean 任务和 CLIP/CNN/ImageNet 等真实视觉任务上一致提升精度。

**[Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](interpretability/finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)**

:   本文发现 LVLM 幻觉源于对正确视觉证据的"关注不足 + 生成中遗忘"，并观察到注意力对视觉证据存在显著的层间差异（ILVAD），据此提出一个 train-free / plug-and-play 的方法：用层间差分构造视觉证据显著性图，再在生成过程中持续加权视觉证据 token 和"扎根于证据"的文本 token，在 5 个 LVLM × 5 个幻觉/综合 benchmark 上一致降低幻觉。

**[Formalizing the Binding Problem](interpretability/formalizing_the_binding_problem.md)**

:   本文把"神经网络的绑定问题"形式化为表示 $Z$ 中关于对象码 $O$ 的互信息 $I(O;Z)$，并设计自回归概率探针在 DINOv2 / CLIP 等 ViT 上度量绑定信息，发现 `[CLS]` 只编码 <50% 的绑定信息且结构近似二次型，而对全套空间 token 加注意力探针能恢复 ~92% 的绑定信息。

**[From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](interpretability/from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)**

:   PRAXIS 用一个"快但近似"的代理算法（改良版 LicketySPLIT）来估计每个子问题的最优目标值，从而对稀疏决策树的 Rashomon 集做"按需展开"式剪枝搜索，把原本与树空间指数相关的运行时和内存压成"每棵输出树只花多项式时间"，在 11M 样本/472 特征级别仍能跑完且 recall ≥ 0.98。

**[GEM: Geometric Entropy Mixing for Optimal LLM Data Curation](interpretability/gem_geometric_entropy_mixing_for_optimal_llm_data_curation.md)**

:   GEM 把 LLM 预训练数据划分问题重写成超球面上的 vMF 混合 + 平衡正则的变分目标，用可证明单调上升的 MM 算法求解，再通过 Teacher-Student 蒸馏到 FastText 上线，在 1.1B 模型上叠加 DoReMi/Perf/RegMix 三种 mixing 框架平均提升约 1.2%。

**[Global Plane Waves from Local Gaussians: Periodic Charge Densities in a Blink](interpretability/global_plane_waves_from_local_gaussians_periodic_charge_densities_in_a_blink.md)**

:   ELECTRAFI 在实空间预测一组各向异性高斯的参数，再利用高斯的解析傅里叶变换 + 泊松求和公式，在倒易空间一次性算出周期晶体的电荷密度的平面波系数，做一次逆 FFT 就拿到全场密度；在 NMAE 与 ChargE3Net 持平甚至更优的同时，推理快 $463\times \sim 633\times$，把端到端 DFT 总时间真正降下来 $\sim 20\%$。

**[Grokking: From Abstraction to Intelligence](interpretability/grokking_from_abstraction_to_intelligence.md)**

:   本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"，并用一个可解析的奇异特征机（SFM）证明这等价于自由能驱动的相变。

**[How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](interpretability/how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)**

:   本文从 prompt 粒度因果分析 n-shot prompt 的 function vector（FV）形成机制，证明 FV 可线性叠加为各 example 子 FV 的加权和，且权重由 FV-head attention 决定；并通过 2×2 QK/V 因果干预表明，contextualization 主要通过 QK 路径（而非 V）让模型把注意力集中到无歧义的 demonstration 上，从而提升 FV 质量。

**[How Language Models Process Negation](interpretability/how_language_models_process_negation.md)**

:   本文用机制可解释性方法剖析 Llama-3.1-8B / Mistral-7B 处理"X that is not Y is __"类否定句的内部电路，发现模型其实"会"做否定（中层注意力在最后位置直接构造出 $\bar Y$ 表示，例如 "not gas" → solid），但被晚层"捷径"注意力头压住——把这些头按"注意力下沉"方式消融，否定题准确率最高可绝对提升 17%。

**[IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)**

:   本文提出 IdEst：用最小生成树维度估计器 $\mathrm{dim}_{\mathrm{MST}}$ 去测自监督表示的内在维度（ID），把这个无标签的几何量当作下游线性 probe 精度的代理，在 33 个 SSL 模型上 Spearman $\rho \approx -0.8$，并可用于无标签超参选择。

**[Interpretability Can Be Actionable](interpretability/interpretability_can_be_actionable.md)**

:   这是一篇立场论文，主张「可解释性研究缺的不是新方法、而是评估准则」：研究该以 actionability（insight 能否驱动可解释性领域之外的具体决策/干预）为核心评估维度，作者沿 concreteness + validation 两个维度定义 actionability、分析阻碍、列出 5 个有杠杆的应用域、给出研究者 6 步 checklist。

**[Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretability/interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)**

:   KREPES 用 eNTK 把任意 SSL 模型近似成核模型，再借 Representer 定理把表征写成"地标样本"的核加权组合，用 Nyström + 单步 GGN-Newton 把 SimCLR/BYOL/VICReg/Barlow Twins 等非凸目标的影响系数解析地解出来，从而无监督地审计 SSL 隐空间并扩展到 1M+ 数据集。

**[IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](interpretability/iqa-spider_unifying_multi-granularity_image_quality_assessment_with_reasoning_gr.md)**

:   本文提出 IQA-Spider，一个把"全局质量描述 + 局部质量描述 + 像素级 grounding + 区域级 referring"四类任务统一到一个 LMM 框架里的多粒度图像质量评估方法，配套构建了 33K 规模的多任务数据集，并用一种 training-free 的 text-to-point 范式把语言模型的位置词 logits 直接映射成 SAM 的点 prompt，在多粒度 IQA 基准上全面超越现有 Q-Instruct / Q-Ground 等专用模型。

**[Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](interpretability/is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)**

:   作者对 6 个主流表格基础模型 (TFM) 做了首个大规模分层机理分析，发现中后层主要在做"迭代精化"且存在大量冗余，并据此设计了一个只用 20% 参数的单层循环 TFM，性能几乎追平六层原版。

**[MAAT: 基于知识引导核回归的异构部分观测状态重建](interpretability/knowledge-informed_kernel_state_reconstruction_from_heterogeneous_partial_observ.md)**

:   MAAT 把"从稀疏 + 异构 + 含噪观测里恢复一条物理一致的潜在状态轨迹"重新表述成一个 RKHS 上的带约束核岭回归问题，把观测算子、平滑性和物理先验（非负、守恒、单调）一起塞进同一个目标函数，给下游符号回归（SINDy / PySR）提供具有解析时间导数的高质量轨迹，在 9 个合成基准 + COVID-19 真实数据上把重建 MSE 降低 1–3 个数量级。

**[Learn from A Rationalist: Distilling Intermediate Interpretable Rationales](interpretability/learn_from_a_rationalist_distilling_intermediate_interpretable_rationales.md)**

:   本文提出 REKD —— 把知识蒸馏引入"选-预测"式 rationale extraction 框架，让小模型学生同时模仿教师的特征选择分布和最终预测分布，并把蒸馏温度与 Gumbel-Softmax 退火调度绑死，从而隐式形成"先学软分布、后学硬选择"的课程，使 ViT-Tiny 在 CIFAR-10 的 RE 准确率从 0.797 拉到 0.936。

**[Learning Coherent Representations: A Topological Approach to Interpretability](interpretability/learning_coherent_representations_a_topological_approach_to_interpretability.md)**

:   本文提出 **coherence**（相干性）这一受脑神经编码启发的几何性质，要求样本-特征矩阵的行列在 Vietoris-Rips 滤过下拓扑互相 interleave，并给出可微的 `Coh` 损失，在自编码器与 BERT token embedding 上得到拓扑对齐、语义可读的 features，效果远超 L1 稀疏。

**[LLMs Lean on Priors, Not Programming Language Semantics](interpretability/llms_lean_on_priors_not_programming_language_semantics.md)**

:   作者构建 PLSemanticsBench——把一个 featherweight C 语言 $\text{C}^{\star}$ 配上 small-step 操作语义 $\mathbb{S}$ 与 K 语义 $\mathbb{K}$ 两套形式系统，并通过 KeywordSwap（互换 `+`/`-` 等运算符语义）与 KeywordObf（替换为 Caucasian-Albanian 罕用符号）系统性扰动语义，测了 11 个前沿 LLM 后发现：标准语义下最高 90% 的最终状态预测准确率，在语义扰动下骤降 40–60 个百分点，长程规则保持准确率最高只有 35%，说明当代 LLM 主要靠预训练词法先验、而不是真的在按显式形式规则做推理。

**[Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](interpretability/manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)**

:   本文提出 MA-GIG：把 Guided IG 的“按低梯度幅值选特征再走一步”策略从像素空间搬到预训练 VAE 的潜在空间，借助 decoder Jacobian 把潜空间内的轴对齐更新映射成数据流形切空间上的更新，从而既避开高梯度噪声区域，又让积分路径上的样本始终贴近真实数据流形，归因更可靠。

**[Memorization Dynamics of Fill-in-the-Middle Pretraining](interpretability/memorization_dynamics_of_fill-in-the-middle_pretraining.md)**

:   作者用相同架构、相同数据、相同算力训练了一对 Llama 3.2 3B（一个走标准 LTR，一个走 FIM），在 Gutenberg 重复语料上系统对比两种目标的逐字记忆行为，发现 FIM 把概率质量摊到更多"部分重建"上（短跨度/重叠召回更强且随重复次数近似线性增长），而 LTR 在长跨度高置信度续写上更强；FIM 的记忆仍然强烈依赖前缀，后缀只是辅助信号。

**[MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](interpretability/minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)**

:   本文把传统社会科学里的联合实验 (conjoint analysis) 从"估计 AMCE 边际效应"重新表述为"在指数级因子动作空间上学习可解释的乘积型 Categorical 随机策略"，给出二阶交互模型下带 $L_2$ 信赖域的闭式解、可微分通解、以及含政党初选制度的两人 minimax 扩展，并通过 Delta 方法把结果模型的不确定性传播到策略概率和价值上，在 2016 美国总统联合实验上首次让对抗均衡的"票房份额"落回历史区间。

**[MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](interpretability/muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)**

:   MUSE 把统一视觉 tokenizer 的"理解-生成"零和困境归因于流形错配，提出梯度正交假设——把语义注入 $W_V$ 而结构梯度走 $W_{Q,K}$——并通过 Synergistic Block + DINOv3 拓扑对齐 + NCE 语义锚定彻底解耦，最终 gFID 3.08 与 linear probing 85.2%（甚至超过 InternViT-300M 老师 82.5%）共存，首次实现真正的"互相强化"而非折中。

**[Neural Collapse by Design: Learning Class Prototypes on the Hypersphere](interpretability/neural_collapse_by_design_learning_class_prototypes_on_the_hypersphere.md)**

:   把"分类器学习 (CE) "和"监督对比学习 (SCL) "统一成超球面上的原型对比，并通过两个新损失 NTCE/NONL（修 CE 侧）和固定原型分类器 FP（修 SCL 侧）让神经坍缩 NC 真正"按设计达成"，同时在精度、迁移、长尾、鲁棒性上全面占优。

**[OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization](interpretability/omnisapiens_a_foundation_model_for_social_behavior_processing_via_heterogeneity-.md)**

:   针对社会行为数据天然异构（10 个任务跨情感/认知/病理/社交，模态横跨语音/视觉/文本）导致 GRPO 类推理 RL 学习信号被少数任务主导的问题，本文提出 HARPO，通过用优势幅值近似各 sample 与各 task 对策略更新的贡献，再以"几何均值参照 + 倒数比"得到结构化调制因子并加上惯性平滑，在 Qwen 2.5-Omni-7B 上训出 OmniSapiens-7B 2.0，多任务平均排名第 1，零样本 5 任务全胜，推理一致性从 66.5% 提到 87.7%，token 数压到 19.86。

**[On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](interpretability/on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)**

:   本文指出 SAE 中"死特征"问题的真正根源不是训练动力学而是激活分布的几何性质——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 量化"维度级离群"严重程度，从初始化就解析地预测死率（454 个模型-层组合上 Spearman $\rho=0.82\sim0.89$），并证明仅用 mean-centering 就能把 AlphaFold3/ESM3 等高 $\gamma$ 模型的死率从 70%+ 降到接近零。

**[Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](interpretability/optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)**

:   本文在高维线性回归 ICL 框架下，用一种保留 softmax 归一化与温度选择性、又解析可解的"近似 softmax 注意力"，**给出 ICL 泛化误差的闭式解和最优 attention temperature 的显式表达式** $\tau_{\text{opt}}$，证明只要调对推理时温度就能恢复近 Bayes 最优表现；在 GPT-2、Llama2-7B 的真实 QA 中也验证了这把"轻量旋钮"的有效性。

**[Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](interpretability/physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)**

:   本文给出了"只用 encoder（无 decoder/无像素重建）从原始视频识别二阶线性 ODE 参数 $(\gamma_1,\gamma_0)$"的首个结构可识别性定理：用一个几何条件 **level-set slope coverage** 刻画"单条轨迹够 vs. 必须三条轨迹"的临界，证明欠阻尼可单视频识别、其它阻尼区必须三条不同轨迹，并配套提出"方差下限正则 + 中心差分"的有限样本估计器。

**[PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](interpretability/pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)**

:   PINE 把"忠实剪枝"对 boosted 树集成的等价约束从全输入空间收缩到一个由 Chow-Liu 树似然 + 分裂共形校准定义的"分布内区域" $\mathcal{X}_{\text{ID}}(\alpha)$，用单一参数 $\alpha$ 平滑控制压缩-保真折中，在 12 个公开 tabular 数据集上把压缩率相对 FIPE 最高提升 30%，同时把"剪枝前后预测一致"的概率以 $\geq 1-\alpha$ 的形式给出可证明保证。

**[PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](interpretability/polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)**

:   PolySAE 在标准稀疏自编码器（SAE）线性解码器之外，新增基于共享低秩投影的二阶/三阶多项式项，用极小参数代价（GPT-2 small 上 ~3%）显式建模稀疏特征之间的乘法交互，在 4 个 LLM × 3 种 SAE 变体上把探针 F1 平均提升约 8%、类条件分布的 Wasserstein 距离扩大 2–10 倍，并能用学到的交互方向因果引导模型输出对应的组合语义。

**[Position: Ideas Should be the Center of Machine Learning Research](interpretability/position_ideas_should_be_the_center_of_machine_learning_research.md)**

:   作者提出"Ideas First"立场：把"想法 → 可观察信号 → 定制化实验"作为机器学习研究的核心评价单位，反对把刷榜数字或理想化定理本身当作目的，从而既弥合理论—实践鸿沟，又降低小算力研究者的参与门槛。

**[Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance](interpretability/position_lets_develop_data_probes_to_fundamentally_understand_how_data_affects_l.md)**

:   作者主张：与其继续用大规模真实语料反复试错，不如设计一类"数据探针"——从**完全已知**的随机过程采样出的合成序列，用它们去训练/微调 LLM 并把模型生成结果**送回已知分布**做似然分析，从而把"哪种数据让模型学会什么"这个问题从经验启发式上升为可证伪的科学命题。

**[Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](interpretability/position_zeroth-order_optimization_in_deep_learning_is_underexplored_not_underpo.md)**

:   这是一篇 position paper，作者主张深度学习中的零阶（ZO）优化并非"无能"而是"未被充分探索"——他们沿算法/系统/评估三条主线给出 6 个论断（P1–P6），核心立场是：跳出"全空间逐元素估计器"的窠臼，转向子空间/谱域估计、前向单向流的系统级红利以及去混淆的评测协议，ZO 才能从内存高效微调的小众工具走向可扩展的训练范式。

**[Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems](interpretability/prompt_optimization_is_a_coin_flip_diagnosing_when_it_helps_in_compound_ai_syste.md)**

:   本文用 18,000 次网格评估和 144 次优化运行实证检验了 compound AI 系统中端到端 prompt 优化的两个隐含假设——agent 之间存在耦合、单 agent prompt 值得优化——发现两者在主流 mid-tier 模型上几乎都不成立（49% 的优化运行表现差于 zero-shot，A×B 交互项 $p>0.52$），并据此提出一个两阶段诊断框架（\$80 的 ANOVA 耦合预测 + \$5 的 10 分钟 headroom 测试），把"是否做 prompt 优化"从抛硬币变成可量化的决策。

**[Prototype Transformer: Towards Language Model Architectures Interpretable by Design](interpretability/prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)**

:   ProtoT 把 Transformer 里 $O(N^2)$ 的自注意力替换成 $R$ 个可学习"原型向量"驱动的线性通信通道（write/read gate + 带时间折扣的 prefix mean），让每个原型在训练中自动绑定一个可命名的概念（如"woman""COVID""New Zealand"），从而支持对模型行为做"手术式"的概念级编辑，且文本生成 Elo 超过同规模 LLaMA。

**[Query Circuits: Explaining How Language Models Answer User Prompts](interpretability/query_circuits_explaining_how_language_models_answer_user_prompts.md)**

:   本文提出 **查询电路 (query circuit) 发现** 任务——直接在原 LLM 内部追踪解释"模型为何对某个具体输入产生该输出"的稀疏子网络，并配套提出更稳健的忠实度指标 NDF 和 Best-of-N 采样算法，使得 MMLU 上仅占模型 1.3% 边的电路即可恢复约 60% 的单题行为。

**[Riemannian Generative Decoder](interpretability/riemannian_generative_decoder.md)**

:   本文针对 Riemannian VAE 必须为每种流形手工设计复杂概率密度的痛点，提出 Riemannian Generative Decoder (RGD)——彻底丢掉 encoder，把每个样本的 latent 当作自由参数用黎曼优化器 (RiemannianAdam) 直接训，同时引入"按局部度量逆缩放的输入噪声"作为几何正则，在合成分支扩散树、人类线粒体 DNA、细胞周期 scRNA-seq 三个真实生物数据上恢复出更忠实的几何，且在高维下数值稳定胜过 VAE 基线。

**[ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation](interpretability/shapleig_bayesian_experimental_design_for_shapley_value_estimation.md)**

:   在评估预算极度受限（如需要重训模型）的代价昂贵游戏上，用带 Hamming 核的 GP 作为价值函数代理、按"对 Shapley 值的期望信息增益（EIG）"自适应挑下一个 coalition，并把 EIG 计算从 $O(4^p t)$ 压到 $O(p^4 + t^3)$。

**[Singular Vectors of Attention Heads Align with Features](interpretability/singular_vectors_of_attention_heads_align_with_features.md)**

:   本文从理论与玩具模型两侧，论证了"为什么以及何时"注意力头 QK 矩阵 $\Omega = W_Q^\top W_K$ 的奇异向量会与模型实际使用的特征方向对齐，并提出"稀疏注意力分解"作为该对齐在真实模型（GPT-2 / Pythia）中可被验证的可观测信号。

**[Sparse Autoencoders are Topic Models](interpretability/sparse_autoencoders_are_topic_models.md)**

:   本文证明稀疏自编码器（SAE）的 $L_1$ 目标恰好是一个 LDA 风格的"连续主题模型"（CTM）在高活动度、小贡献极限下的 MAP 估计，并据此提出 SAE-TM 框架：预训练 SAE 得到可复用的主题原子，事后学习词分布并通过聚类合并到任意主题数，在文本和图像数据集上的主题连贯性显著超过当前主流神经主题模型。

**[Steer Like the LLM: Activation Steering that Mimics Prompting](interpretability/steer_like_the_llm_activation_steering_that_mimics_prompting.md)**

:   本文把 "prompt steering"重新解释为 LLM 自己实现的一种 activation steering, 然后用一个**逐 token 的 ReLU 探针**来蒸馏 prompt 注入的激活差, 训练出 PSR (Prompt Steering Replacement) 模块, 既能在三个 steering 基准上超过现有激活引导方法 (CAA, ReFT-R1, Stolfo 等), 又在 AxBench 与人格引导上和 prompting 打成平甚至反超。

**[The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level](interpretability/the_expert_strikes_back_interpreting_mixture-of-experts_language_models_at_exper.md)**

:   本文用 $k$-sparse probing 系统比较了 MoE 专家神经元与 dense FFN 神经元的多义性，发现 MoE 在稀疏路由压力下天然更接近单义，进而把分析单元从"神经元"升到"整个专家"，用 LLM 自动给数百个专家打自然语言标签并通过因果触发实验验证，最终得出"专家既不是宽域领域专家、也不是 token 级处理器，而是细粒度任务专家"的结论。

**[Towards Atoms of Large Language Models](interpretability/towards_atoms_of_large_language_models.md)**

:   论文为大语言模型的"基本表征单元"给出第一个形式定义——原子（atoms），用一种非欧几里得的"原子内积"刻画 LLM 隐藏表征的内蕴几何，证明阈值激活 SAE 在适当条件下可以精确恢复原子集合，并在 Gemma2 / Llama3.1 上实测出 $R^2\approx 99.9\%$、稳定性 $q^\*\approx 99.85\%$ 的近理想原子。

**[Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](interpretability/towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)**

:   针对推理 LLM 长思维链场景下逐 token 归因 $\mathcal{O}(M\cdot N)$ 慢且归因质量被中间推理 token 吸光的问题，本文提出 FlashTrace：用 span-wise 聚合一次过算完整段目标 token 的归因，再用递归归因把重要性从输出经推理链回溯到原始输入，5k 目标 span 上比最强基线 IFR 快 130 倍以上，同时在 RULER / MATH / MoreHopQA 上 faithfulness 全面占优。

**[Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](interpretability/tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)**

:   本文用 Causal Tracing 在 LLM 内部发现"拒绝"不是终端 token 的静态向量、而是横跨上游中间层与 token 的"拒绝轨迹"(Refusal Trajectory)，并据此设计 SALO——一个只在常规对齐数据上训练、却能利用 Transformer 因果掩码不可逆性识别 GCG / AutoDAN / Prefilling 等对抗攻击的 <20M 参数检测器，把 GCG/Prefilling 上 0% 的检测率拉到 >85%。

**[学习尖峰分布中的通用 1/3 时间缩放](interpretability/universal_one-third_time_scaling_in_learning_peaked_distributions.md)**

:   通过分析 softmax 与交叉熵在学习峰值概率分布时的数学性质，论文揭示了 LLM 训练损失呈现通用 1/3 幂律衰减的根本原因——这是一个与数据结构无关的架构层面的优化瓶颈。

**[Verified SHAP: 神经网络精确 Shapley 值的可证明界](interpretability/verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)**

:   VERISHAP 通过组合分支定界与神经网络验证技术，首次为神经网络 SHAP 值计算提供可证明的界限——并能扩展到比现有精确方法大几个数量级的特征搜索空间。

**[What Linear Probes Miss: Multi-View Probing for Weight-Space Learning](interpretability/what_linear_probes_miss_multi-view_probing_for_weight-space_learning.md)**

:   这篇论文指出单视角一阶 probe 会漏掉权重矩阵的行列交互与二阶相关结构，并提出 MVProbe 用行/列一阶投影加行/列 Gram 分支的多视角表示，在 Model Jungle 和 Stable Diffusion LoRA 识别上显著超过 ProbeX。

**[Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](interpretability/why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)**

:   本文给出"为什么 transformer 的内部表征可以被简单线性方法（probe、SAE、activation steering）反复成功解码"的架构级解释：只要语义特征是通过 OV 电路或 unembedding 这类**线性接口**被读出的，它就必须落在一个跨上下文不变的线性子空间里（Invariant Subspace Necessity 定理）；并推出一个零样本应用——Self-Reference Property，即 token 本身的嵌入方向就是其概念方向，从而可以无监督地用 class token 的几何位置直接做分类。

---

## 💡 LLM 推理 { #llm_reasoning }

**[A Formal Comparison Between Chain of Thought and Latent Thought](llm_reasoning/a_formal_comparison_between_chain_of_thought_and_latent_thought.md)**

:   本文从计算复杂度理论出发，形式化比较 CoT（链式思维）与隐式思维（Looped Transformer / Coconut）的表达能力，证明隐式思维在多对数深度下严格达到 $\mathsf{TC}^k$，而 CoT 最多到 $\mathsf{TC}^{k-1}$；同时在概率设置下首次揭示 CoT 通过随机解码可支持 FPRAS 计数，反过来超越确定论隐式思维。

**[An Information-Theoretic Criterion for Efficient Data Synthesis](llm_reasoning/an_information-theoretic_criterion_for_efficient_data_synthesis.md)**

:   这篇论文用数据处理不等式解释合成数据为何有时有效、有时导致模型坍塌：只有当训练闭环持续引入稳定外部信号时，合成数据才是 information-open；而高 meta-level 的验证信号比实例级模仿更高效、更容易泛化。

**[Are Large Reasoning Models Interruptible?](llm_reasoning/are_large_reasoning_models_interruptible.md)**

:   这篇论文把大推理模型从静态题目评测拉到会被用户打断、会收到中途更新的动态环境中，构建数学与编程评测协议，并发现强模型会出现推理泄漏、恐慌作答和自我怀疑三类稳定失效模式。

**[Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](llm_reasoning/are_tools_always_beneficial_learning_to_invoke_tools_adaptively_for_dual-mode_mu.md)**

:   AutoTool 用强化学习让多模态大模型先判断“这题是否真的需要 zoom-in 工具”，再在工具辅助推理和纯文本推理之间自适应切换，从而在高分辨率感知、定位、幻觉检测和推理任务上同时提升准确率与效率。

**[Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](llm_reasoning/beyond_test-time_memory_state-space_optimal_control_for_llm_reasoning.md)**

:   将 LLM 推理建模为隐空间上的最优控制问题（线性二次调节器 LQR），提出 Test-Time Control (TTC) 层在前向传播中执行有限时域规划并解码最优控制动作作为下一 token 表示，配合辛迭代 CUDA 高效求解器，作为适配器插入预训练 LLM 后在 MATH-500 上提升最多 +27.8%，AMC/AIME 上 Pass@8 提升 2-3 倍。

**[Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](llm_reasoning/beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)**

:   提出 BRIDGE 框架，将 SFT 与 RL 的整合建模为双层优化问题——SFT 作为上层教师通过轻量 LoRA 模块学习选择性地向 RL 学生传递有益监督信号，在五个数学推理基准上平均绝对提升超过 3 个百分点。

**[Biases in the Blind Spot: Detecting What LLMs Fail to Mention](llm_reasoning/biases_in_the_blind_spot_detecting_what_llms_fail_to_mention.md)**

:   提出一个全自动黑盒流水线来检测 LLM 的"未言明偏见"（unverbalized biases）——系统性影响模型决策但从未在 CoT 推理中被提及的隐性因素，通过 LLM 自动生成概念假设、反事实输入变体和分阶段统计检验，在三个决策任务上自动发现了性别、种族等已知偏见以及西班牙语流利度、英语水平、写作正式度等新偏见。

**[Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling](llm_reasoning/blending_supervised_and_reinforcement_fine-tuning_with_prefix_sampling.md)**

:   提出 Prefix-RFT，通过从专家示范中采样前缀拼接模型续写来构建混合轨迹，在保持 RFT 目标导向优化的同时注入 SFT 的知识引导，在数学推理任务上显著超越独立 SFT、RFT 及已有混合方法。

**[Chain-of-Thought Reasoning in the Wild Is Not Always Faithful](llm_reasoning/chain-of-thought_reasoning_in_the_wild_is_not_always_faithful.md)**

:   本文在**非对抗性、自然措辞**的提示下（无人工注入偏见），揭示前沿LLM的链式推理（CoT）存在两种不忠实行为——**隐式后验合理化**（对逻辑对立的比较问题给出矛盾的相同答案并各自编造合理论证）和**不忠实非逻辑捷径**（在数学难题中跳过关键推理步骤却得出正确答案），生产模型不忠实率最高达13%，即使思考型模型（DeepSeek R1: 0.37%，Sonnet 3.7 thinking: 0.04%）也非完全忠实。

**[Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](llm_reasoning/clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)**

:   本文揭示 Transformer 自注意力与 $k$-means 聚类的数学等价性，据此设计 KCoT 框架，将 CoT 推理显式拆解为"赋值-更新"两步语义过滤提示，并用 Condition-Net 动态融合拓扑先验与演化思维表示，在节点分类和链接预测上持续超越 SOTA。

**[CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](llm_reasoning/cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)**

:   本文提出 CoCoReviewBench，通过"按类别建子基准 + 用 meta-review 仲裁审稿人/作者冲突来过滤错误意见"两步，把 3,900 篇 ICLR/NeurIPS 论文的人工审稿改造成一个更可信的 AI 审稿评测参考，并发现现有 AI 审稿在 correctness 和 thoroughness 上仍落后于人类、推理模型则更有潜力。

**[Conformal Thinking: Risk Control for Reasoning on a Compute Budget](llm_reasoning/conformal_thinking_risk_control_for_reasoning_on_a_compute_budget.md)**

:   本文把"reasoning LLM 何时停止思考"从一个不可解释的阈值调参问题，重构为一个**用户可指定 risk 容忍度**的 conformal 风险控制问题：用两个阈值——上阈值在模型自信时停（控 false positive），新提出的**参数化下阈值**在模型在不可解题上"想不动"时强行停（控 false negative）——并通过 UCB 算法从校准集自动求出满足风险约束的阈值，在 AIME / GPQA / MathVision 上实现"准确率几乎不掉、token 大幅省"。

**[DecepChain: Inducing Deceptive Reasoning in Large Language Models](llm_reasoning/decepchain_inducing_deceptive_reasoning_in_large_language_models.md)**

:   DecepChain 提出第一个能让 LLM 在带特定触发词时生成"读起来完全像正常 CoT、却必然给出错误答案"的后门训练范式：先用模型自己产生的"自然犯错"轨迹做 SFT，再用反向奖励 + 格式奖励的 GRPO 课程式强化学习放大欺骗，从而把"看似可信的推理"和"真的可信的推理"之间的边界彻底抹平。

**[Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs](llm_reasoning/deliberate_evolution_agentic_reasoning_for_sample-efficient_symbolic_regression_.md)**

:   把 LLM 主导的符号回归"出招—打分"循环拆成"提案 vs. 导航"两层——再用自适应算子（方向）、诊断工具（残差/量纲）、反思记忆（轨迹经验）三路信号显式引导 LLM，只用 40% 评估预算就在 LLM-SRBench 上把 NMSE 平均压低 37–55%。

**[DenseSteer: Steering Small Language Models towards Dense Math Reasoning](llm_reasoning/densesteer_steering_small_language_models_towards_dense_math_reasoning.md)**

:   观察到强模型 CoT 步数更少但每步信息密度更高（Dense Reasoning），DenseSteer 用 GPT-5.1 把小模型自己的稀疏解答改写成"信息更密"的同分布正样本，与原解答构成对比对，在中间层（≈ L17）残差流注入一条均值差分得到的 steering vector，零训练即可在 GSM8K / MATH500 / AMC / AIME 等数学基准上稳定涨点且不抬高 token-level NLL。

**[Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](llm_reasoning/diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)**

:   本文把"找出 CoT 推理链里哪一步出错"形式化为黑盒场景下的步级置信度归因问题，用信息瓶颈原则把"同一问题多次采样得到的正确推理轨迹"压成共识结构，分别给出免训练的 NIBS（语义共识对齐）和可学习的 GIBS（图共识子图选择）两种实例，在 GSM8K / Math / MoreHopQA 上稳定优于白盒基线，并用步级反馈把自我纠错成功率提升最多 13.5%。

**[Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models](llm_reasoning/diversity_matters_revisiting_test-time_compute_in_vision-language_models.md)**

:   本文系统研究了 test-time compute (TTC) 策略在视觉-语言模型上的有效性，从理论上证明多数投票的收益受预测多样性限制，并提出基于预测熵选最自信模型的 ETTC，使小模型能反过来增益大模型，在 7 个 VLM、6 个基准上平均比投票高 +2.8%、超过最强单模型。

**[Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](llm_reasoning/diversity_over_frequency_rethinking_tool_use_in_visual_chain-of-thought_agents.md)**

:   在 3D 空间推理这类"工具非必需"的视觉 Agent 任务上，作者发现 vanilla RFT 会让工具调用率塌缩到接近 0、强制鼓励工具调用又只带来边际收益；真正驱动性能提升的是 rollout 的探索多样性，于是用自适应熵正则化把 3DSRBench 准确率从 59.2% 推到 62.9%，并把工具重新定位为"训练期脚手架"而非推理期必备品。

**[Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)**

:   作者把 latent CoT 看作一个可干预的结构因果模型（SCM），对每个连续"思考步"做 step-wise `do`-intervention + 早停解码 + teacher-forced 读出，系统量化 Coconut/CODI 在数学和常识推理上的步级必要性、传播结构与轨迹叠加性，发现 latent step 并不是同质化的"加深"，而是高度异质、非局部路由、且输出层早承诺先于表示层承诺的结构化界面。

**[ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](llm_reasoning/ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)**

:   ETS 直接从 KL 正则化 RLHF 目标的**闭式最优解**采样，把它写成「参考策略 × 指数 reward 的条件期望（能量项）」，再用 Monte Carlo + 自归一化重要性采样在测试时近似这个能量项，从而**不训练**就达到甚至超过经过 RL 后训练的策略，并通过 lightweight proposal + Fast-dLLM 把延迟控制在可用范围。

**[Evaluating Relational Reasoning in LLMs with REL](llm_reasoning/evaluating_relational_reasoning_in_llms_with_rel.md)**

:   作者把认知科学里的"关系复杂度"（Relational Complexity, RC）—— 即一次推理步骤里必须同时绑定的独立变量数 —— 作为衡量任务难度的统一坐标轴，构建了横跨代数 / 生物 / 化学三个学科的生成式 benchmark REL，发现前沿 LLM（Claude Opus 4.5 / Gemini 3 Pro / GPT-5.2）的准确率随 RC 单调下降，且 test-time compute、ICL、外接工具都救不回来。

**[FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](llm_reasoning/floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)**

:   FloorplanQA 用 2,000 个 JSON/XML 格式的 2D 室内布局 + 16,000 道几何题（距离/可见性/路径/放置等）系统性诊断了 15 个前沿 LLM 的"纯符号空间推理"能力，发现它们能算简单距离却普遍栽在并集、规划和约束满足上，且 Python 工具增强能修复算术错误但救不了算法层面的失败。

**[ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution](llm_reasoning/foresightkv_optimizing_kv_cache_eviction_for_reasoning_models_by_learning_long-t.md)**

:   ForesightKV 训练一个轻量打分模型，按"未来注意力贡献"动态淘汰 KV 对：先用 Golden Eviction 算法从完整 trace 中蒸馏出最优淘汰序列作监督信号，再用 GRPO 强化学习以"低熵 token 损失增量平方和"为奖励微调策略，在 AIME2024/2025 上用一半 KV 预算超过 SnapKV/H2O/R-KV，4K 预算可保留 99% 原模型性能。

**[From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates](llm_reasoning/from_llm-generated_conjectures_to_lean_formalizations_automated_polynomial_inequ.md)**

:   NSPI 让 LLM 提出近似的多项式平方和 (SOS) 结构猜想，再用 Gauss–Newton 迭代和有理恢复把猜想精修成严格的有理系数 SOS 分解，最后用 Lean 的 `linear_combination` + `positivity` 策略自动机器验证，把不等式证明可扩展到最多 10 个变量。

**[GRPO is Secretly a Process Reward Model](llm_reasoning/grpo_is_secretly_a_process_reward_model.md)**

:   本文从理论上证明 GRPO + ORM 在"组内轨迹共享前缀"的温和条件下**等价于**一个带有 Monte-Carlo PRM 的过程奖励 RL 目标，从而揭示出 vanilla GRPO 隐藏的一个 bug——前缀长度不均会让高奖励轨迹的大部分 token 拿到负 advantage——并提出 $\lambda$-GRPO 做一个 PRM-aware 归一化，在推理 benchmark 上稳定超过 GRPO 且训练快约 2 倍。

**[Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](llm_reasoning/hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)**

:   用一个简单的逻辑回归探针在 LLM 思维链生成时的隐藏状态上能以 0.95 AUROC 预测整条推理是否会出错（从第 1 步就有 0.79），但文本表面同样训出来的分类器只有 0.59；可惜 4 种干预手段（激活引导、探针引导 best-of-N、自我修正、激活补丁）全部失败——这个错误信号是"诊断性"的而非"因果性"的。

**[How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](llm_reasoning/how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)**

:   本文用一个叫 Tele-Lens 的低秩 adapter 探针在 12 个跨域任务上系统度量 LLM 隐状态对"未来推理"的预测能力，发现 LLM 的内部规划是**近视**（myopic）的——只在 CoT 末端才精确锁定答案，并据此提出"木桶原理"用稀疏 pivot 位置的不确定性代表整条 CoT，可显著改善不确定性校准并实现 16% 的 CoT 旁路。

**[Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](llm_reasoning/inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)**

:   本文针对大型推理模型 (LRM) 易被"逻辑残缺输入"激发过度思考的弱点，提出一个层级化遗传算法 (HGA)，在纯黑盒条件下把结构化分解后的题目当成基因，通过句子级/问题级交叉和增删变异搜索逻辑断裂的对抗样本，最高可在 MATH 上把响应长度放大 26.1 倍，制造低成本 DoS 攻击。

**[Inference Time Optimization with Confidence Dynamics](llm_reasoning/inference_time_optimization_with_confidence_dynamics.md)**

:   作者发现在 LLM 多次采样推理中，正确轨迹的置信度沿 reasoning chain 系统性上升而错误轨迹衰减或下降，据此提出 CDG（Confidence Dynamic Gain）投票——把"尾段置信度 − 头段置信度"作为额外判别信号嵌入 Best-of-N 加权投票，在四个开源推理模型 × 四个数学奥赛 benchmark 上平均较 majority voting 提升 5.4%、较 DeepConf 提升 1.7~4.8%。

**[Internalizing Safety Understanding in Large Reasoning Models via Verification](llm_reasoning/internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)**

:   本文论证「会生成安全答案」≠「懂安全」，提出 SInternal 框架：只训练大型推理模型去 verify 自己生成答案的安全性，由此涌现的内在安全理解大幅压制 jailbreak 攻击（StrongREJECT ASR 从 41% 降到 0.6%）并成为后续 RL 的更好起点。

**[LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning](llm_reasoning/latentchem_from_textual_cot_to_latent_thinking_in_chemical_reasoning.md)**

:   LatentChem 在化学 LLM 上把"显式 CoT 文本链"换成"连续 latent 思考向量 + 动态分子感知更新"，并在 GRPO 纯结果奖励下观察到模型**自发抛弃文本 CoT**、改用 latent 推理，在 ChemCoTBench 上对显式 CoT baseline 非平局胜率 59.88%，推理步数平均下降 10.84 倍、wall-clock 加速 5.96 倍。

**[MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](llm_reasoning/learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)**

:   MOSAIC 把"安全决策"从隐式推理副产物变成 plan → check → act/refuse 循环里的显式一等动作（含 `<safety_thoughts>` 和 `refusal_tool`），用 LLM judge 的成对轨迹偏好 + GRPO 训练；在 Qwen2.5-7B / Qwen3-4B-Thinking / Phi-4 上零样本 OOD 减少 50% 有害行为、prompt injection 拒绝率提升 20%、隐私泄漏下降，benign 任务效用不退。

**[Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](llm_reasoning/less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)**

:   论文揭示了 Test-Time Scaling (TTS) 一个被忽视的失效模式——只要把候选回复的多样性压低，TTS 反而比直接喂高对抗性 prompt 更容易输出不安全内容；并提出 RefDiv，一个用 Shannon 熵 + 参考引导双信号驱动的遗传算法，能在 MCTS 和 Best-of-N 上跨模型、跨闭源、跨 guardrail 地高效越狱。

**[Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](llm_reasoning/lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)**

:   LiDAR 用预先生成的几步 lookahead 样本和前向扰动核重写期望未来奖励（EFR），把 reward guidance 变成无需神经反传的闭式 softmax 权重，在 SDXL/GenEval 上匹配 DATE 的指标却快 9.5×。

**[Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](llm_reasoning/many-shot_cot-icl_making_in-context_learning_truly_learn.md)**

:   本文系统揭示了非推理任务的 many-shot ICL “经验法则”在 CoT 推理任务上**全部失效**——相似度检索反而有害、顺序敏感性随 shot 数增长——并把成功的 many-shot CoT 重新解读为“in-context 测试时学习”，由此提出按 embedding 轨迹曲率排序 demonstration 的 CDS 方法，在 64-shot 几何题上提升 5.42 pp。

**[Mean-Shift PCA by Knockoff Mean](llm_reasoning/mean-shift_pca_by_knockoff_mean.md)**

:   本文用随机矩阵理论证明"均值偏移污染"在样本协方差矩阵的谱上与真正的协方差 spike 是渐近独立的，并据此提出一个两阶段算法 MS-PCA：通过故意往数据里加一个"诱饵"均值偏移（knockoff mean）后再做一次 PCA，比较两次结果，把"被诱饵推动的"特征值识别为污染分量、剔除掉，从而在高维下用纯 PCA 操作恢复真正的主成分。

**[Measuring Weak-to-Strong Legibility of Reasoning Models](llm_reasoning/measuring_weak-to-strong_legibility_of_reasoning_models.md)**

:   本文提出 **Transfer Utility (TU)** ——把强推理模型 (RLM) 写出的中间推理 trace 按百分位前缀喂给一个弱学生模型，用弱学生续写出正确答案的能力来度量 trace 的"弱到强可读性"；在 12 个开源 RLM × 3 个数据集 × 85k 条 trace 上发现：**当前最准、最简洁的 RLM (如 GPT-OSS-120B) 的 trace 在 TU 排名中反而垫底**，说明 RLVR 训练把推理 trace 变成了"只对强模型有用"的工件。

**[Modeling Hierarchical Thinking in Large Reasoning Models](llm_reasoning/modeling_hierarchical_thinking_in_large_reasoning_models.md)**

:   作者把大推理模型（LRM）的长 CoT 抽象成一个 6 状态有限状态机（FSM），用「成功 vs 失败」的状态转移概率差构造 Transition Advantage Matrix，并基于 Q-Value 迭代得到长视野规划策略，仅在句子边界做稀疏的正交激活引导，就能用约 25× 更少的干预次数把 AIME25 等难题的准确率拉高最高 +13%。

**[On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs](llm_reasoning/on_robustness_and_chain-of-thought_consistency_of_rl-finetuned_vlms.md)**

:   本文通过在视觉推理基准上注入"误导性 caption"与"错误 CoT 前缀"两类受控文本扰动，系统暴露 RL 微调后开源 VLM 在视觉接地与思维链忠实性上的脆弱，揭示出 RL 优化下"准确率↑ vs CoT 忠实性↓"的显式 trade-off，并表明数据增强与忠实性奖励都无法同时解决两端。

**[On the Generalization Gap in Self-Evolving Language Model Reasoning](llm_reasoning/on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)**

:   本文在"只有未标注 prompt + 基座模型"的严格闭环设定下，系统比较了 4 种自我进化（SE）策略（单轮验证、多轮修订、迭代训练、课程学习）与 oracle 监督的差距，发现在 Knights & Knaves 逻辑推理上 SE 能把 Gemma 3 4B 从 31.0% 提到 44.8%，但相对 oracle 的 53.3% 仍有 8–13% 的持续 gap，只有 12B 模型的 RevisionSE 才能逼近 oracle（52.8% vs 53.6%）。

**[PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](llm_reasoning/powerflow_unlocking_the_dual_nature_of_llms_via_principled_distribution_matching.md)**

:   本文把无监督 LLM 微调重新表述为"匹配基模型 $\alpha$-power 分布"的问题，用 GFlowNet 的 Trajectory-Balance 目标作为摊销采样器，并通过长度感知的 LA-TB 重参数化消除自回归生成中的结构性长度偏置；同一个旋钮 $\alpha$ 控制方向——$\alpha>1$ 锐化分布激发推理（媲美或超过有监督 GRPO），$\alpha<1$ 平滑分布释放对齐模型被压制的创造力，在 Pareto 前沿上同时拉高质量与多样性。

**[Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](llm_reasoning/prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)**

:   针对 Looped Language Model（LoopLM）在每个 token 输出前会反复迭代 $T_{\max}$ 次潜在表征的特点，本文提出 RLTT：把 GRPO 中"只奖励最后一圈"的策略梯度改成"按权重 $\omega_t$ 给每一圈的 next-token 分布都打分"，在不引入外部 verifier、计算开销几乎为零的情况下，把 Ouro-2.6B 在 MATH/AIME/BeyondAIME 上的平均准确率提升 +10.9%，并出现训练时间下降 10% + 响应长度自发缩短的副产物。

**[Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](llm_reasoning/prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)**

:   作者把"为离散扩散语言模型（dLLM）做高效 test-time scaling"这一问题拆成三件事——按"探索→渐进剪枝→精修"的层级时间表分配计算（HTS）、用部分 remask 做局部分支保住高置信"逻辑骨架"、把 dLLM 自己当 Yes/No 验证器（SVF），最终在 4 个数学/代码基准、3 个 dLLM 上以远少于 best-of-$N$ 的 NFE 达到相近甚至更好的精度。

**[Prompt Injection as Role Confusion](llm_reasoning/prompt_injection_as_role_confusion.md)**

:   本文把"提示注入"的根因归结为 LLM 在潜空间里**用风格而非标签来识别"谁在说话"**的角色混淆现象，提出"角色探针"来量化这种混淆，并设计 CoT Forgery（思维链伪造）攻击，在六个前沿模型上将原本接近 0% 的攻击成功率拉到 60% 以上，同时证明探针测得的"角色混淆度"在模型生成第一个 token 之前就能预测攻击是否会成功。

**[R2-Router: A New Paradigm for LLM Routing with Reasoning](llm_reasoning/r2-router_a_new_paradigm_for_llm_routing_with_reasoning.md)**

:   本文提出 R2-Router，把"输出 token 预算"从被动估计量改造成可控变量，让路由器在 (LLM, 预算) 联合空间里搜索，用一个轻量的多头质量预测器把每个 LLM 从一个静态点扩展成一条质量-成本曲线，从而以 4–5× 更低的成本达到与现有路由器相当的质量。

**[Reasoning Can Be Restored by Correcting a Few Decision Tokens](llm_reasoning/reasoning_can_be_restored_by_correcting_a_few_decision_tokens.md)**

:   作者用 token 级分布散度量化 base LLM 与 reasoning LRM 的差异，发现差距高度集中在早期、规划相关、且 base 自身不确定的少量 token 上（占比 ~8%），据此提出"分歧门控的一令牌接管"——仅在分歧尖峰处让 LRM 出一个 token 然后立刻交还 base，用 ~4-13% 干预预算即可恢复甚至超越同尺寸 thinking 模型。

**[Reasoning Structure of Large Language Models](llm_reasoning/reasoning_structure_of_large_language_models.md)**

:   本文把大型推理模型（LRM）的自由文本思维链转成"原子声明 + 演绎依赖"的可验证 DAG，并基于吸收马尔可夫链的结构熵定义一个推理流效率指标 $\eta$，证明在准确率和 token 数都饱和或重叠的区间，$\eta$ 仍能分辨"专注推理"与"发散探索"两种行为，从而成为诊断 LRM 失败模式的细粒度工具。

**[ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)**

:   ResRL 从理论上把 RLVR 中 "负样本梯度污染正样本"现象 (Lazy Likelihood Displacement) 分解成"logit × 表征"两个分量,然后在表征层用正样本的 SVD 低秩子空间做投影残差,根据每个负 token 的"正交分量能量"给它一个 [ξ,1] 区间的梯度权重——表征越像正样本(残差越小)就罚得越轻,纯错误成分才被重罚,既保住 Pass@1 又不丢 Pass@k 多样性;在 Qwen3-4B 数学任务上 Avg@16 比 NSR 提升 9.4%,Pass@128 提升 7.0%。

**[Reward Modeling from Natural Language Human Feedback](llm_reasoning/reward_modeling_from_natural_language_human_feedback.md)**

:   本文指出在二元偏好奖励上训练的 generative reward model (GRM) 严重存在"猜对偏好但 critique 错误"的 outcome-process 不一致（20-30%、最高 44%），并提出 RM-NLHF：把模型 critique 与人工 critique 的核心论点相似度作为额外过程奖励，并用 MetaRM 自动预测过程奖励、在线随策略更新，从而在多个 benchmark 上稳定超过 outcome-only GRPO 训练的 SOTA GRM。

**[Scaling-Aware Adapter for Structure-Grounded LLM Reasoning](llm_reasoning/scaling-aware_adapter_for_structure-grounded_llm_reasoning.md)**

:   Cuttlefish 把 Q-Former 那种"固定长度查询 token"换成了"按结构复杂度自适应增长的指令条件 patch token"，再用 cross-attention 把 EGNN 抽出的几何特征作为 modality token 注入 LLM，从而在分子 / 蛋白 / DNA / RNA 四种全原子模态上同时降幻觉、扛 scaling，超过一众模态专用 baseline。

**[Scientific Logicality Enriched Methodology for LLM Reasoning: A Practice in Physics](llm_reasoning/scientific_logicality_enriched_methodology_for_llm_reasoning_a_practice_in_physi.md)**

:   本文首次系统研究 LLM 科学推理中的"逻辑性"，提出"逻辑保真度 / 因果连接 / 推理进展" 三维评估指标，并基于该指标构造两种 SFT 数据采样方法（风格迁移 RST、逻辑蒸馏 Logic-Distill），在自建的 PhysLogic 基准与三个公开物理 benchmark 上把 7B 模型的逻辑性与答题准确率同时提了一大截。

**[Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain](llm_reasoning/self-play_only_evolves_when_self-synthetic_pipeline_ensures_learnable_informatio.md)**

:   作者主张当下的"LLM 自博弈"之所以在几轮内就崩溃，根本原因是自合成数据没有提供可学习信息增益；他们用有界 MDL/epiplexity 把"可学习信息"形式化，并提出三个系统级设计——非对称协同演化、容量预算增长、主动信息寻取——共同保证三角色 (Proposer-Solver-Verifier) 自演化循环中可学习信息单调上升。

**[SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning](llm_reasoning/smartthinker_progressive_chain-of-thought_length_calibration_for_efficient_large.md)**

:   本文提出 SmartThinker，一种基于 GRPO 的高效推理后训练方法，通过对每个 prompt 的"全部轨迹长度分布"与"正确轨迹长度分布"做高斯建模，解析推导出"使正确率最大的最优长度"$l^{\text{opt}}$，并配合一个动态长度奖励系数 $\Lambda$ 保证正确轨迹的归一化优势非负，从而在最多压缩 52.6% token 的同时把 AIME25 准确率相对提升最多 16.6%。

**[Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](llm_reasoning/stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)**

:   本文从动力系统视角诊断 Looped Language Model (LoopLM) 在 test-time 扩展深度时"先涨后崩"的根因——归一化位置导致的"稳定—有效"二元困境，并提出 STARS：用 Jacobian 谱半径正则化 (JSRR) + 随机循环采样把潜在轨迹拉向"渐近稳定的有效不动点"，在 GSM8K 上把 8 步循环的性能跌幅从 20.47% 压到 8.26%，同时峰值提升 4.01%。

**[The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](llm_reasoning/the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)**

:   本文发现解码器 Transformer 在确定性状态追踪任务上因注意力容量限制存在"**确定性地平线**"（约 19-31 步），超过此阈值扩展推理会导致性能崩溃；通过信息论 + 大规模实证（72 万次评估）证明这是**架构能力**失败而非"简洁性偏好"，并量化论证了工具委托（如符号求解器）的**必要性**——可将准确率从 24-42% 拉到 86-94%。

**[The Expressive Power of Low Precision Softmax Transformers with (Summarized) Chain-of-Thought](llm_reasoning/the_expressive_power_of_low_precision_softmax_transformers_with_summarized_chain.md)**

:   本文首次证明：使用 softmax 注意力 + bfloat16 量级精度（激活与注意力权重都四舍五入）的标准 Transformer 解码器，只要深度、宽度对数地随上下文增长，就能借助 CoT 模拟任意图灵机；并进一步证明 Summarized CoT 把规模从时间界 $\hat{t}$ 降到空间界 $\hat{s}$，且在 Sudoku 任务上实测发现"加深度而不是加精度"才是 CoT 长上下文失败的真正解药。

**[ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)**

:   作者把 MATH 数据集的人工标注解题步骤逐步翻译成"带描述与类型签名的可复用 Python 工具"，构造出含 8K 题 + 12K 工具的 ToolMATH 基准；它同时覆盖长程多工具组合（hop 1-8+）、可控的干扰工具相似度（5 级 × 4 种密度）、以及"金标工具被全部移除"的工具缺失场景，验证显示模型失败的主导因素不是工具选择而是推理本身——thought error 占 90%+，而干扰工具会把早期的小偏差放大成不可逆的执行漂移。

**[TRACE: 用 Toulmin 论证模型评 LLM CoT 推理过程质量](llm_reasoning/trace_toulmin-based_reasoning_assessment_through_constructive_elements_for_llm_c.md)**

:   TRACE 是个参考无关的 CoT 质量评估指标，把 Toulmin 论证模型（Claim/Data/Warrant/Backing/Qualifier/Rebuttal）+ Flavell 元认知（Monitoring/Evaluation）合成 8 个构成要素，用 DeBERTa 多标签识别每句推理的要素，再算"State Validity + Transition Coherence"加权和；在 26.3K QA × 7 模型上与 benchmark accuracy 相关 $r=0.741$，且能当 RL reward 让 GSM8K 提升 +9.9%。

**[UCPO：不确定性感知的策略优化](llm_reasoning/ucpo_uncertainty-aware_policy_optimization.md)**

:   UCPO 通过三元优势解耦（TAD）和动态不确定性奖励调整（DURA）两个机制，解决了现有RL范式中固定不确定性奖励导致的优势偏差问题，使LLM能在知识边界处可靠地表达不确定性，在Qwen3-8B上数学推理PAQ达到79.63%。

**[UniScale：通过模型路由与测试时缩放在线联合优化的自适应统一推理缩放](llm_reasoning/uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)**

:   提出 UniScale 框架——将模型路由和测试时缩放统一到一个决策空间，通过 LinUCB 上下文多臂老虎机在线学习自适应推理策略，解决 LLM 部署中质量-成本的精细权衡问题。

**[Verifying Meta-Awareness via Predictive Rewards in Reasoning Models](llm_reasoning/verifying_meta-awareness_via_predictive_rewards_in_reasoning_models.md)**

:   通过让推理模型自预测解法长度、通过率和所需概念，用预测结果与真实统计对齐来优化模型元认知——从而显著提升数学推理性能并加速训练。

**[What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code](llm_reasoning/what_really_improves_mathematical_reasoning_structured_reasoning_signals_beyond_.md)**

:   这篇论文通过 10T-token 语料和 MoE 从头预训练的控制实验指出，真正提升复杂数学推理的不是纯可执行代码本身，而是跨域结构化推理信号，尤其是数学语料中显式暴露中间步骤的“认知脚手架”。

**[When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](llm_reasoning/when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)**

:   这篇论文在 Hierarchical Reasoning Model 中加入 manager-worker 式持久子目标，发现潜变量推理里的关键不是单纯注入子目标，而是子目标应该持续 $P=3$ 到 $6$ 个低层更新步，过快重规划会破坏组合结构，过强 alignment 又会干扰任务学习。

---

## 📐 优化/理论 { #optimization }

**[A General Framework for Dynamic Consistent Submodular Maximization](optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)**

:   这篇论文给出了 fully dynamic 子模最大化的一般一致性框架，在允许插入和删除的流式环境中，首次为 cardinality 与 matroid 约束同时实现常数近似和次线性级别的 worst-case 每步解变动。

**[Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization](optimization/accelerated_multiple_wasserstein_gradient_flows_for_multi-objective_distribution.md)**

:   这篇论文把 Multiple Wasserstein Gradient Descent 推广为连续时间梯度流，并引入 Nesterov 风格的动量加速，得到 A-MWGraD，在理论上把 geodesically convex 场景的收敛率从 $O(1/t)$ 提升到 $O(1/t^2)$，实验上也让多目标采样和贝叶斯多任务学习更快收敛。

**[Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality](optimization/adaptive_estimation_and_inference_in_semi-parametric_heterogeneous_clustered_mul.md)**

:   本文桥接双重机器学习与聚类多任务学习，提出自适应框架结合 Neyman 正交性与数据驱动的配对融合罚项，在异质（可能无限维）噪声的半参数设置中精确恢复任务潜在聚类、以汇总率达到预言水平，并建立渐近正态性，实现有效统计推断。

**[Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](optimization/adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)**

:   这篇论文把 Polyak step size 推广到 USAM/SAM，给出不依赖手工学习率调参的 sharpness-aware scheduler，并在凸优化理论和 CIFAR 实验中验证其稳定性与性能。

**[Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization](optimization/asymmetric_perturbation_in_solving_bilinear_saddle-point_optimization.md)**

:   这篇论文证明只扰动双线性零和博弈中一方的 payoff，就能在足够小扰动下保持原始均衡不变，并据此构造 AsymP-GDA，在理论上获得线性 last-iterate 收敛，在普通型和扩展型博弈实验中比对称扰动更快、更准地逼近原始均衡。

**[Automatic Unsupervised Ensemble Outlier Model Selection–Extended Version](optimization/automatic_unsupervised_ensemble_outlier_model_selection--extended_version.md)**

:   提出 MetaEns 框架，通过元学习预测候选检测器的边际集成增益，结合多样性折扣和算法族风险正则化的代理目标函数，在无标签条件下自适应地贪心构建紧凑高质量的异常检测集成模型。

**[Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](optimization/balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)**

:   本文揭示了 LoRA 的过参数化导致不同低秩因子对 $(A,B)$ 具有不同条件数，证明了**平衡最小值点**（$A^\top A = BB^\top$）具有最优条件数，并据此提出 BaLoRA——在每步优化后将适配器投影到平衡流形上，以几乎零开销加速收敛并提升微调性能。

**[Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](optimization/balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)**

:   本文在两层和三层线性神经网络中推导出梯度下降一步和两步后测试损失的精确闭式表达式，揭示了一个相变现象：第一步更新时非对称学习率最优，而第二步后对称（平衡）学习率变为局部最优，为逐层学习率调度提供了理论基础。

**[Bayesian Gated Non-Negative Contrastive Learning](optimization/bayesian_gated_non-negative_contrastive_learning.md)**

:   针对非负对比学习（NCL）中共享背景特征导致的优化冲突（梯度振荡），提出 BayesNCL，通过贝叶斯门控头为每个特征维度学习 Bernoulli 分布来动态过滤高频公共特征，在 ImageNet-100 上语义一致性提升 142.1% 且不牺牲下游准确率。

**[Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](optimization/bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)**

:   本文提出 Lévy Mirror Flow（LMF）——一种由 Lévy 噪声驱动的随机镜像下降连续时间 SDE 模型，证明即使在无穷方差的重尾梯度噪声下，SMD 仍保持收敛保证（凸情形 $O(\varepsilon^{-p/(p-1)})$，强凸情形 $\tilde{O}(\varepsilon^{-1/(p-1)})$），并将连续时间结果无缝传递到离散时间算法。

**[Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](optimization/budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)**

:   首次给"预算受限 + 私有成本"的子模社会福利最大化采购拍卖给出有近似比保证的真值机制 BFM-SWM——用几何递增阈值的降序时钟拍卖 + 单点保护 + 价/付率参数 $\beta$ 实现非负盈余 + 预算可行，一般子模函数 0.0328-近似、单调子模 0.0877-近似；副产品 BFM-VM 把估值最大化的确定性最佳近似比从 1/64 提升到 $1/(12+4\sqrt{3})\approx 0.0528$，并将运行时间从 $\mathcal{O}(n^2\log n)$ 降到 $\mathcal{O}(n\log n)$。

**[Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](optimization/can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)**

:   首次证明 AdaGrad 在重尾噪声（$p \in (4/3, 2]$）下无需任何算法修改即可收敛，同时给出算法依赖的下界表明 AdaGrad 无法达到 minimax 最优速率，并证明 AdaGrad-Norm 在有界目标函数假设下可获得更快的 $O(1/T^{(p-1)/(2p)})$ 速率。

**[Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction](optimization/colorful_pinball_density-weighted_quantile_regression_for_conditional_guarantee_.md)**

:   本文通过 Taylor 展开揭示了标准 pinball 损失在条件覆盖率优化上的固有缺陷——忽视了异方差结构，提出密度加权 pinball 损失作为条件覆盖 MSE 的更紧代理目标，并设计三头分位数网络通过有限差分估计密度权重，在 8 个高维回归基准上大幅提升条件覆盖性能。

**[Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](optimization/convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)**

:   在重尾噪声 + 常数比例强对抗污染下，作者证明了一大类非单调链接函数（GeLU、Swish、Tanh、Probit、Logistic、相位恢复…）的高斯单指标模型平方损失存在一个维度无关、常数半径的凸盆，并据此设计了一个 $\tilde{O}(nd)$ 时间、$\tilde{O}(d)$ 样本的鲁棒恢复算法，最终估计误差为 $O(\sigma\sqrt{\epsilon})$。

**[Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](optimization/delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)**

:   针对部分参与下"采样客户端中拜占庭客户端临时占多数"会击垮已有鲁棒聚合的痛点，本文提出延迟动量聚合原则——服务器把当轮新动量与未被采样客户端的最近一次缓存动量一起送入鲁棒聚合器，将全局拜占庭比例 $\delta<1/2$ 永远延续到每一轮聚合，并据此设计 DeMoA 优化器，在 $p=0.1$、$\delta=0.2$ 的极端设置下仍能稳定训练 ResNet-18/CIFAR-10。

**[Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](optimization/distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)**

:   本文提出 DELTA：在线把"切空间线性化教师"的中间激活蒸馏到普通非线性学生 + EK-FAC 曲率正则 + 沿插值路径采样，让常规非线性 fine-tune 出来的 task vector 也具备线性化模型那种"可叠加、低干扰、对缩放鲁棒"的性质，同时不引入任何推理开销。

**[Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](optimization/distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)**

:   本文提出 AgentPulse 框架，将 split conformal、adaptive conformal inference (ACI)、Mondrian conformal 与 BH-FDR 组合，为 50 个 AI agent 的连续打分提供分布无关的覆盖率保证、组合管线的不确定性边界以及带 FDR 控制的排名弃权机制，把"测量不确定性"作为评测的一等输出。

**[Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](optimization/dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)**

:   本文在 student–teacher 数据对齐的线性 RNN 上，把 BPTT、one-step tBPTT、RFLO 的更新写成可解析的 ODE，比较它们的不动点流形、稳定性、收敛速率，发现 RFLO 缺少 BPTT/tBPTT 那条非最优鞍流形但代价是稳定性依赖符号、收敛更慢，并且**局限于初始权重的低秩扰动**——这一低秩限制可推广到非数据对齐的设定。

**[Enhancing LLM Training via Spectral Clipping](optimization/enhancing_llm_training_via_spectral_clipping.md)**

:   本文提出 SPECTRA：一个 optimizer-agnostic 的包装层，对更新矩阵做**后置谱裁剪**、对原始梯度做可选的**前置谱裁剪**，在理论上等价于带权重正则的复合 Frank-Wolfe 算法，在 124M–1.5B LLM 预训练上把 AdamW / Signum / Mars / AdEMAMix 的验证损失一致地往下压。

**[Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality](optimization/follow-the-perturbed-leader_for_decoupled_bandits_best-of-both-worlds_and_practi.md)**

:   本文给 decoupled multi-armed bandit 问题（每轮分别选一个臂"利用"、一个臂"探索"）设计了首个 Best-of-Both-Worlds (BOBW) FTPL 算法：用 Pareto 扰动做利用、用一个仅依赖累积估损排名的代理量 $q_{t,i}$ 直接定义探索分布——既不需要 FTRL 的每步凸优化，也不需要 FTPL 标准做法中的几何重采样，对抗与随机两种环境下均达到与现有最优 FTRL 算法同阶的 $\mathcal{O}(\sqrt{KT})$ / $\mathcal{O}(K/\Delta_{\min})$ 后悔界，实测对 $K=2$ 比基线快约 130×。

**[HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](optimization/ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)**

:   HO-SFL 通过拉格朗日变量提升把 split federated learning (SFL) 的客户端和服务端解耦——服务端继续做一阶反向传播 (BP)，客户端只做零阶 (ZO) 扰动前向，再借共享随机种子把每轮上行通信压到 $\mathcal{O}(P)$ 个标量，从而在端侧把大模型微调的显存降到推理级、收敛率仍可达 $\mathcal{O}(\sqrt{d_c/PT})$。

**[Interpretability and Generalization Bounds for Learning Spatial Physics](optimization/interpretability_and_generalization_bounds_for_learning_spatial_physics.md)**

:   论文用数值分析工具证明：在线性 PDE（1D Poisson 等）上学到的解算子 $\mathbf{W}$ 只会收敛到真算子 $\mathbf{A}$ 在训练函数空间上的投影 $\mathbf{A}\mathbf{U}\mathbf{U}^\top$，所以**函数空间本身**——而非数据量或网格细度——决定 OOD 泛化；并提出一种把权重矩阵作用在 one-hot 上即可看出"是否学到 Green 函数结构"的机械可解释技术，用 25×25 跨数据集 cross-evaluation 把 8 类 SciML 模型（含 PINN/DeepONet/FNO/PI-DeepONet）的失败模式逐个标出来。

**[Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](optimization/learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)**

:   训练一个轻量网络预测线性指派问题 (LAP) 的对偶变量 $\hat{u}$，用 Min-Trick 构造可行对偶 $\hat{v}$，将其作为 LAPJV 精确求解器的暖启动，从而在保持最优性的同时把 $N=16{,}384$ 规模实例端到端加速 $2\times$ 以上。

**[Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](optimization/learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)**

:   本文提出 ZO Fine-tuner：用一个"per-block 的轻量神经网络 PertNN"自动学习 LLM 各参数块的扰动方差，把 MeZO 中固定的 $\mathcal{N}(0,I)$ 升级为按块自适应的非均匀扰动；在 OPT-30B 上辅助网络仅占 <2MB，却在 4 个 LLM × 7 个数据集（28 对）中 82.1% 跑赢现有零阶基线，且"一次训练、跨任务/跨衍生模型复用"。

**[Learning Context-Conditioned Predicate Semantics via Prototype Feedback](optimization/learning_context-conditioned_predicate_semantics_via_prototype_feedback.md)**

:   AlignG 把 PE-Net 的静态谓词原型改造成"图像条件化"的动态原型：先用关系候选给原型做 GRU 增量更新拿到 image-specific prototype，再反向用它去 recalibrate 关系特征，并把对齐损失锚定在静态全局原型上以防漂移，在 VG-150 / GQA-200 的 SGDet 设置上 F@100 分别涨 1.4 / 2.7。

**[Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)**

:   本文用 empirical NTK 作为统一视角，证明 zeroth-order SGD 引出的 eNTK 等价于把 first-order eNTK 投影到由微扰张成的随机子空间，从而通过 Johnson-Lindenstrauss 引理解释为何 ZO 方法在十亿参数 LLM 上仍然 work：误差只取决于输出维度 $V$ 和微扰数 $P$，与模型维度 $d$ 无关。

**[Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](optimization/learning_locally_revising_globally_global_reviser_for_federated_learning_with_no.md)**

:   本文观察到 FL 的全局模型对噪声标签存在"延迟记忆"现象（CIFAR-10 上记忆率 ≤30%，显著低于集中式训练），据此提出 FedGR——用服务器端 GMM 在所有客户端聚合损失代理上联合筛选并估计每个客户端的噪声比例，再用全局参数定期"修正"本地 EMA 教师以做蒸馏，并加入全局-本地表征一致性正则。三模块协同，在双重异质 (label noise × non-IID) 设定下相比 8 个 SOTA 基线在 CIFAR-10/100 + Clothing1M 上稳定取得显著增益。

**[Learning Randomized Reductions](optimization/learning_randomized_reductions.md)**

:   本文把"发现某个函数 $f$ 的随机自归约 (RSR)"这一沉寂四十年的人工任务，形式化成一个带相关采样的学习问题，并构建了 Bitween 框架：先用稀疏线性回归在固定查询集 $\{x+r, x-r, x \cdot r, x, r\}$ 内挖掘 RSR，再让 LLM 智能体在更大的查询函数空间里搜索，最终在 80 个数学/ML 函数构成的 RSR-Bench 上把 RSR 覆盖率从 54% 推到 80%，并首次给出 sigmoid 的 RSR 表达式。

**[Limits of Convergence-Rate Control for Open-Weight Safety](optimization/limits_of_convergence-rate_control_for_open-weight_safety.md)**

:   作者把"开源权重安全"形式化为"如何延缓恶意 fine-tune 的收敛速度"，证明 Hessian 谱的最大奇异值由权重谱下界决定，由此设计了能严格减慢一阶/二阶优化的 SpecDef 算法，但同时证明任何此类收敛率控制方法都能被攻击者以"线性模型尺寸增加"的代价绕过。

**[LiMuon: Light and Fast Muon Optimizer for Large Models](optimization/limuon_light_and_fast_muon_optimizer_for_large_models.md)**

:   LiMuon 把 STORM 风格的动量方差缩减和随机 SVD（RSVD）一起塞进 Muon 优化器，把矩阵参数的动量从 $m \times n$ 压成 $(m+n)\hat{r}$、同时把求 $\epsilon$-稳态点的 SFO 复杂度从 $\mathcal{O}(\epsilon^{-4})$ 降到 $\mathcal{O}(\epsilon^{-3})$，在 Mamba-130M / Qwen2.5-0.5B / ViT 上同时取得更低 perplexity / 更高 accuracy 和更小显存。

**[LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](optimization/lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)**

:   LoRe 把凝聚态物理里的「集团 + 浴场」分解搬到扩散式图组合优化求解器，做成训练免修的推理时包装器，在每一步只评估固定比例的高冲突边并用一个 $\mathcal{O}(N)$ 的全局召回项补偿被丢弃的部分，让 MIS 求解突破 baseline OOM 上限 $3\times$、单卡跑 $n=50\mathrm{k}$ 实例，TSP $n=1000$ 上拿到 $\sim 15\times$ 加速和 $44\times$ 显存压缩。

**[Memory-Efficient LLM Pretraining via Minimalist Optimizer Design](optimization/memory-efficient_llm_pretraining_via_minimalist_optimizer_design.md)**

:   本文用"自底向上拆解 Adam"的方式找出真正必须的两个组件——逐列梯度归一化 + 只在最后一层加一阶动量——把它们组合成 SCALE 优化器，用接近 SGD 的内存 (LLaMA 7B 上 13.74 GB) 达到了 Adam 级甚至超越 Muon/APOLLO 的预训练困惑度。

**[Mirror Descent Under Generalized Smoothness](optimization/mirror_descent_under_generalized_smoothness.md)**

:   本文提出一种基于任意范数及其对偶范数的 $\ell*$-广义光滑性概念，并通过"广义自界引理"把梯度对偶范数控制在初始次最优间隙之内，从而首次为镜像下降及其加速、乐观、Mirror Prox、随机、复合等变体在非欧几何下建立了与经典 $L$-smooth 下匹配的收敛率。

**[Mirror Mean-Field Langevin Dynamics](optimization/mirror_mean-field_langevin_dynamics.md)**

:   本文把 mean-field Langevin dynamics (MFLD) 与 mirror Langevin dynamics (MLD) 缝合成"镜像 mean-field Langevin dynamics" (MMFLD)，第一次给出在凸约束域 $X\subseteq\mathbb{R}^d$ 上最小化熵正则化泛函 $\mathcal{L}(\mu)=F(\mu)+\lambda\,\mathrm{Ent}(\mu)$ 的全局收敛算法 —— 连续时间下用均匀 mirror LSI 证 $e^{-2C_{\mathrm{LSI}}\lambda t}$ 线性收敛，离散化下用 $N$-粒子 + Euler-Maruyama 给出 uniform-in-time propagation of chaos。

**[Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](optimization/multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)**

:   STAGE-BO 把 MOBO 重写成一串"由 fill distance 自适应选门限"的 ε-约束单目标贝叶斯子问题，用 cEI 求解，从而在不算 hypervolume 的前提下取得均匀的 Pareto 前沿覆盖，并天然兼容硬约束与用户偏好。

**[Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](optimization/muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)**

:   本文在带 softmax 检索和分层频谱的线性关联记忆模型上，对 Muon 进行收敛速率与缩放律的理论刻画：相对 GD，Muon 在无噪声情形获得指数级加速，在幂律频谱噪声情形将损失收敛律从 $\tilde{\Omega}(T^{-(1-1/\beta)})$ 提升到 $\tilde{\mathcal{O}}(T^{-2})$，并把这一加速归因于矩阵符号算子等价于一个自适应任务对齐的隐式预条件子。

**[Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](optimization/neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)**

:   用一个生成-评估神经网络（GEN）一次性地把 QAOA² 的"图分割 + 量子电路参数初始化"两件事联合可微化：评估器学一个高保真的 quantum performance surrogate，生成器在它的梯度引导下吐出离散分区 + 参数初值，配合直通估计器 + 正交补头让端到端可训练；在 183 个 QUBO/Ising/MaxCut 实例（21-1000 变量）上超越启发式 baseline，101 个实例排第一。

**[On the Convergence Rate of LoRA Gradient Descent](optimization/on_the_convergence_rate_of_lora_gradient_descent.md)**

:   本文首次在不假设 adapter 矩阵有界、不要求重参数化损失 Lipschitz 平滑的前提下，证明了原始 LoRA 梯度下降的最小梯度范数以 $O(1/\log T)$ 速率收敛（若参数范数有界则恢复经典 $O(1/T)$），并据此设计了与理论严格对应的自适应/归一化学习率，在 logistic regression、ResNet-18、TinyLlama 上验证了训练加速与稳定性提升。

**[On the Expressive Power of GNNs to Solve Linear SDPs](optimization/on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)**

:   本文从 Weisfeiler–Leman 层级的角度首次刻画了学习线性 SDP 解所需的最小 GNN 表达力，证明标准的变量-约束二部图消息传递（VC-WL）和高阶 VC-2-WL 都不够，而 2-FWL 等价的 VC-2-FWL 架构足以仿真 PDHG 求解器的更新步骤，并在合成与 SDPLIB 上把高质量预测用作 warm-start，最多带来约 80% 的加速。

**[On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](optimization/on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)**

:   本文指出文献中标准一阶 / 二阶 SDE 在 $(L_0,L_1)$-光滑下完全错失学习率稳定性约束（甚至预测发散区间也收敛），通过在漂移项中把曲率项符号翻正，作者构造出一族"稳定性忠实"的一阶弱近似 SDE，首次在统一框架内分析 DCSGD 与 DSignSGD 在压缩 + 仿射方差 + 重尾噪声下的收敛性，并给出归一化强度该如何选取的具体处方。

**[On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](optimization/on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)**

:   本文从理论上证明：在最优点随时间漂移的非平稳强凸随机优化中，动量 SGD 因"惯性滞后"系统性劣于普通 SGD，性能恶化的代价是 $(1 - \beta)^{-2}$ 量级的放大因子；并通过信息论下界论证这种代价不是分析的产物，而是任何方法不可避免的根本障碍。

**[PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](optimization/pathwise_planning_through_world_model_for_automated_heuristic_design_via_self-ev.md)**

:   PathWise 把 LLM 自动启发式设计（AHD）重新建模成一条在"蕴含图"上展开的序列决策过程，由策略 / 世界模型 / 双评价四个 LLM 智能体协作，用反思替代梯度更新，在 TSP、CVRP、KP、装箱等问题上以 50% 的评估预算超过 FunSearch、EoH、ReEvo、HSEvo、MCTS-AHD 等主流基线。

**[Probing Neural TSP Representations for Prescriptive Decision Support](optimization/probing_neural_tsp_representations_for_prescriptive_decision_support.md)**

:   作者把训练好的 TSP 神经求解器视作"可迁移编码器",用冻结表征 + 轻量探针预测两类昂贵的运筹敏感性查询(节点移除与边禁用),系统证明探针准确率随求解器质量单调提升,可以与传统启发式集成达到 SOTA。

**[Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming](optimization/provably_data-driven_lagrangian_relaxation_for_mixed_integer_linear_programming.md)**

:   本文给"学预测 Lagrangian 乘子加速 MILP"这一经验路线第一次配上了严格的统计学习理论：导出 $\mathcal{O}(s^{1.5}/\sqrt{N})$ 的 ERM 泛化上界 + $\Omega(s/\sqrt{N})$ 的 minimax 下界 + 用 SGA 平均算法构造性达到 $\Theta(s/\sqrt{N})$ 最优率，并证明转成"学暖启动初值"后样本复杂度可以提升到 $\Theta(s/N)$。

**[Pseudospectral Bounds for Transient Amplification in Coupled Gradient Descent](optimization/pseudospectral_bounds_for_transient_amplification_in_coupled_gradient_descent.md)**

:   本文为 block-triangular Jacobian $J = \begin{bmatrix} A & 0 \\ C & D \end{bmatrix}$ 的耦合梯度下降建立尖锐的 Kreiss 常数界 $K(J) \leq 2/(1-\gamma) + \|C\|/(4(1-\gamma))$，并给出匹配下界——揭示了即使谱半径 < 1，瞬态放大也可能任意大；这套理论作为高维学习动力学的 scaling law，给出 $O(K(J)^2 \log(1/\delta))$ 的有限时迭代复杂度，并扩展到 nearly self-referential 系统。

**[RACO: Reward-free Alignment for Conflicting Objectives](optimization/reward-free_alignment_for_conflicting_objectives.md)**

:   RACO 把多目标 LLM 偏好对齐做成多目标优化问题——每个目标走自己的 DPO 损失，用 clipped CAGrad（CAGrad + 按用户权重剪裁系数）解决梯度冲突；理论证明收敛到尊重 user-specified 权重的 Pareto-critical 点（两目标场景下 clipping 严格加速），实证在 Qwen 3 / Llama 3 / Gemma 3 多模型族上一致拿到更好的 Pareto 折中。

**[RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)**

:   本文基于 Transformer 层级 Hessian 的「行块对角占优」结构，把 Muon 优化器里昂贵的 Newton-Schulz 正交化换成一次行级 $\ell_2$ 归一化，将每步预条件复杂度从 $\mathcal{O}(mn\min(m,n))$ 降到 $\mathcal{O}(mn)$，在 GPT-2 / LLaMA 预训练上 wall-clock 提速 13–44×、ppl 不降反略升。

**[SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](optimization/safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)**

:   SPSsafe 把 Stochastic Polyak Step Size (SPS) 扩展到非光滑随机优化——既不需要 interpolation 假设也不需要知道最优值，配合动量（IMA = SHB 等价形式）仍保有严格收敛保证；在 DNN 训练上比已有自适应方法（AdaGrad、Adam、DecSPS 等）更稳健，且梯度范数不塌缩到近零（抗梯度消失）。

**[Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks](optimization/sharp_description_of_local_minima_in_the_loss_landscape_of_high-dimensional_two-.md)**

:   本文在教师-学生两层 ReLU 网络的高维 Gaussian 输入设定下，用一组关于权重重叠 $(Q,R)$ 的精确低维概要统计方程，给出 population loss 所有局部极小的层级化分类，并刻画过参数化如何把低阶 spurious 极小变成鞍点、把高阶极小保留下来，从而首次同时调和了 Safran–Shamir 的存在性结果、Arjevani–Field 的群论分类和 Safran 等人的 Hessian 失稳论。

**[Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression](optimization/sign_lock-in_randomly_initialized_weight_signs_persist_and_bottleneck_sub-bit_mo.md)**

:   本文揭示训练后的权重符号矩阵在所有架构上都与 i.i.d. Rademacher 噪声难以区分，从而构成亚比特压缩的"一比特墙"，并用停时分析证明这种伪随机性其实是初始化符号的"锁定"——再据此提出低秩符号模板 + 间隙初始化 + 边界对数障碍正则的从头训练方案，把符号位摊销到接近 0 bit/weight。

**[Stability Analysis of Sharpness-Aware Minimization](optimization/stability_analysis_of_sharpness-aware_minimization.md)**

:   本文从动力系统视角剖析 SAM 在鞍点附近的收敛不稳定：先在确定性梯度流下证明只要邻域半径 $\rho > -1/\lambda_1$，鞍点就会变成 SAM 的吸引子；再在随机扩散框架下证明 SAM 的鞍点逃逸均方位移比 SGD 小 $2\eta t^2|\lambda_j|^3\rho/B$；最后用 SAM 扩散公式解释 momentum 和 batch size 为什么是 SAM 取得 SOTA 泛化性能的真正幕后功臣。

**[SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation](optimization/symerge_from_non-interference_to_synergistic_merging_via_single-layer_adaptation.md)**

:   本文把"模型合并"的目标从"避免任务干扰"重新定义为"促进任务协同"，提出 SyMerge：只联合优化每个任务的一个 task-specific 层和编码器的层级 merging 系数，再用 fine-tuned 专家模型当软标签老师，避免熵最小化在测试时漂移，从而在视觉/密集预测/NLP 三类基准上把合并模型推到接近单任务上限的水平。

**[Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](optimization/taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)**

:   在 PINN 损失里加入由 Feynman–Kac 公式蒙特卡洛模拟得到的少量内点伪标签，本质上就是给 PDE 算子做了一次预条件——本文同时给出"条件数在 collocation 数 $N$ 上保持有界"的算子级证明和带 $\tanh$ 激活的非渐近 $L^2$ 误差界，且在 Schrödinger、Poisson、committor 等问题上让本来彻底失败的 PINN 重新可解。

**[Test time training enhances in-context learning of nonlinear functions](optimization/test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)**

:   本文给单层 softmax-attention transformer + LoRA 测试时微调的组合建立了首个严格泛化界，证明在 single-index 多项式任务上 TTT 把 ICL 的样本复杂度从 $r^{\Theta(\mathrm{ie}(\sigma_*))}$ 压到 $r^{\Theta(\mathrm{ge}(\sigma_*))}$ 并允许 link 函数逐任务变化、推理误差可随上下文长度 $\to$ 噪声水平。

**[Towards Understanding Adam Convergence on Highly Degenerate Polynomials](optimization/towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)**

:   本文挑出一类高阶退化多项式 $L(x)=\tfrac{1}{k}x^k$（$k\ge 4$ 偶数）作为最小问题模型，证明在常数学习率下 Adam 通过 $v_t$ 与 $g_t^2$ 的"解耦"机制把有效学习率指数放大，从而实现局部线性收敛，而 GD 与动量在同一问题上只能拿到 $\Theta(t^{-1/(k-2)})$ 的次线性速率，并完整刻画了 Adam 在 $(\beta_1,\beta_2)$ 平面上"稳定收敛 / spike / SignGD 振荡"三个相区。

**[Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](optimization/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)**

:   作者在简化单层线性注意力 Transformer 上推出闭式训练动力学，证明正则化方法只能改变收敛速度而无法挪动收敛点（因此在 cFKA 场景几乎注定失效），数据回放则能直接改变收敛点并加大震荡幅度从而稳住旧知识，进而提出按 token 注意力贡献裁切片段、引导预训练模型生成回放语料的 STOC，在合成 + KnowEdit + IndustryCorpus 法律语料上一致比 LAMOL 更能压制遗忘。

**[TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](optimization/tpv_parameter_perturbations_through_the_lens_of_test_prediction_variance.md)**

:   作者把"训好模型对参数扰动的局部预测敏感度"形式化为 Test Prediction Variance（TPV），证明其在一阶近似下化为 $\mathrm{Tr}(H_{\mathrm{eff}}C)$ 的迹形式，从而把 SGD 噪声、标签噪声、量化、剪枝放进同一个曲率–协方差框架，并给出一个完全用训练集就能估计 TPV 的稳定性定理，落地为 label-free 剪枝准则 JBR 和无需测试标签的模型选择信号。

**[Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](optimization/ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)**

:   本文证明：在 L2 权重衰减附近的稳态附近，**几乎任何**学习规则（SGD、Adam、DFA，甚至随机网络）的学习信号都会**自发**朝 Hebbian 方向对齐，而足够强的噪声又会把它翻成 anti-Hebbian，并在 $\gamma \propto \sigma^2$ 处出现明确的相变边界。

**[URS：统一的神经路由求解器](optimization/urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)**

:   提出统一数据表示（UDR）和混合偏差模块（MBM）来替代问题枚举——使单个神经模型能无需微调地零样本泛化到 110 个 VRP 变体（99 个未见过）。

**[变分适配器跨模态相似度表示](optimization/variational_adapter_for_cross-modal_similarity_representation.md)**

:   通过变分推理框架学习连续的跨模态相似度分布——用自适应不确定度权重缓解二元标注导致的虚假负样本问题，显著提升 VLM 在跨模态检索和域泛化任务中的性能。

---

## 🔒 LLM 安全 { #llm_safety }

**[ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](llm_safety/actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)**

:   本文提出一个分层框架 ACTG，将隐私文本生成分解为特征学习与条件文本生成两个子任务；进一步引入 Anchored RL，通过混合强化学习目标与基于最优 N 选一的 SFT 锚点，在保持文本保真度的前提下提升条件生成器的指令跟随能力，在生物医学数据上相比先前工作提升 20% MAUVE。

**[AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing](llm_safety/alimark_enhancing_robustness_of_sentence-level_watermarking_against_text_paraphr.md)**

:   AliMark 将句子级文本水印从“前缀条件下的逐句检测”改写为“全局秘密比特序列的编码与对齐”，通过重构候选文本和自适应块编辑距离显著提升了对 DIPPER、GPT-3.5 等强改写攻击的检测鲁棒性。

**[Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model](llm_safety/anchored_decoding_provably_reducing_copyright_risk_for_any_language_model.md)**

:   本文提出 Anchored Decoding：在推理时把高性能但可能复现训练数据的 risky LM 锚定到只用开放许可数据训练的 safe LM 附近，用可调的信息预算在版权复制风险和生成质量之间给出有形式保证的折中。

**[Antidistillation Fingerprinting](llm_safety/antidistillation_fingerprinting.md)**

:   这篇论文提出 Antidistillation Fingerprinting (ADFP)，用代理学生模型估计哪些水印 token 最容易被蒸馏过程吸收，从而在几乎不牺牲教师输出质量的情况下，更可靠地检测第三方模型是否训练过教师模型输出。

**[Beyond Procedure: Substantive Fairness in Conformal Prediction](llm_safety/beyond_procedure_substantive_fairness_in_conformal_prediction.md)**

:   本文超越保形预测（CP）的过程公平性视角，从下游决策的实质公平性出发，理论证明并实验验证了**等化预测集大小**（而非等化覆盖率）才是与实质公平强相关的程序指标，并提出基于 LLM-in-the-loop 的可扩展评估框架和标签聚类 CP 方法来有效平衡效用与公平。

**[BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](llm_safety/bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)**

:   BioAgent Bench 给"用 LLM agent 跑生物信息学 pipeline"这件事造了一个端到端的评测套件——10 个真实 bioinformatics 任务 × 10 个 frontier/open-weight 模型 × 3 个 agent harness，配合 LLM 判官评分和 corrupted/decoy/prompt-bloat 三类扰动测试，发现前沿模型能完成 90%+ pipeline 但鲁棒性堪忧。

**[BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks](llm_safety/byorn_bootstrap_your_own_responses_to_defend_large_vision-language_models_agains.md)**

:   BYORn 通过检测与输入语义不一致的高困惑度目标响应来识别投毒样本，并用模型自身生成的干净响应动态替换，从而打破后门触发器与恶意输出之间的关联，在保持干净任务性能的同时将攻击成功率平均降低 40 个百分点。

**[COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](llm_safety/coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)**

:   COFT 通过在解码时构造反事实掩码分支并与原始分支进行 logit 融合，再用双分支分裂共形预测过滤 token，以无训练、免梯度的方式在冻结 LLM 上实现了逐步 token 级别的反事实公平性保证，将偏见指标降低 30–55%（中位 38%）且几乎不损失任务性能。

**[Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](llm_safety/decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)**

:   FedDTL 把 CLIP 的图像编码器留在客户端、文本编码器搬到服务器做"全局语义锚"，再用 SFT 暖启 + GRPO 风格 RL 的两阶段本地微调，在异构和 full-data 联邦场景下同时缓解客户端间优化不一致与客户端内过拟合。

**[Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why](llm_safety/deep_sequence_models_tend_to_memorize_geometrically_it_is_unclear_why.md)**

:   本文指出 Transformer / Mamba 在死记硬背图的边时并不会真的退化成查找表（联想记忆），而是会自发把节点嵌入排成一种编码了多跳全局结构的"几何记忆"，并通过 path-star 实验证明这种几何让隐式推理变得反常地容易，但其出现既不能归因于监督、容量也不能归因于优化压力，留下一个新的"记忆之谜"。

**[dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](llm_safety/dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)**

:   dgMARK 把扩散语言模型（dLLM）固有的"解码顺序自由度"用作水印通道——根据二进制哈希优先解码满足奇偶条件的位置，无需修改 token 概率分布，就能在 LLaDA / Dream 上嵌入可统计检测且对插删替/改写鲁棒的水印。

**[Differentially Private Preference Data Synthesis for Large Language Model Alignment](llm_safety/differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)**

:   DPPrefSyn 把"在私有偏好数据上做 DP 微调"换成"用 DP 学一个偏好奖励模型分布后再用公开 prompt 合成 DP 偏好数据"，借助 Bradley-Terry 线性奖励的几何结构 + DP-PCA + DP-KMeans 聚类捕捉用户偏好异质性，在 Anthropic-HH 上 $\varepsilon=2$ 拿到 56.5% GPT-4o win-rate，反超无隐私微调（55.95%）和 DP-FT（37.0%）。

**[Dual-branch Robust Unlearnable Examples](llm_safety/dual-branch_robust_unlearnable_examples.md)**

:   本文提出 DUNE：把不可学习样本（UE）的扰动从单一空间域扩展到"空间 + 色彩"双域优化，使扰动特征对齐到 shift-induced 标签并配合预训练模型集成增强，在 CIFAR-10 / ImageNet 上对 7 种主流防御（含 ECLIPSE、ISS-J、COIN）保持鲁棒，平均测试精度比 12 个 SOTA UE 方案再低 14.95%–50.82%。

**[DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](llm_safety/dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)**

:   DualOptim+ 把 Adam 优化器状态拆成"共享 base 态 + 解耦 delta 态"，让 LLM 机器遗忘在 forget/retain 梯度时而冲突时而协同的情况下自适应地在共享和解耦优化器之间过渡，理论上同时退化为 Alternate（正相关）和 DualOptim（负相关），并通过 8-bit 量化变体把额外显存开销压回基线。

**[Efficient DP-SGD for LLMs with Randomized Clipping](llm_safety/efficient_dp-sgd_for_llms_with_randomized_clipping.md)**

:   本文提出 DP-SGD-RC，用 Hutchinson / Hutch++ 随机迹估计代替 DP-SGD 中的精确逐样本梯度范数计算，把长上下文 LLM 训练的裁剪内存开销从 $O(B\min\{T^2,d^2\})$ 降到 $O(BkT+kp)$，配套给出基于卡方混合 envelope CDF 的紧 $f$-DP 分析，在 Llama-3.2-1B 长上下文微调上保持精度、最大线性层峰值显存降低约 40%、FLOPs 节省约 2×。

**[Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences](llm_safety/federated_variational_preference_alignment_with_gumbel-softmax_prior_for_persona.md)**

:   本文提出 FedVPA-GP：在联邦学习的隐私约束下，用"客户端混合先验 + Gumbel-Softmax 可学习权重 + 正交原型损失"把每个客户端的偏好建模成一个连续隐变量 $z$，从根上修掉了把 VPL 直接搬到 FL 时遭遇的"后验崩溃"，使一个奖励模型可以在 helpful 与 harmless 这两种冲突偏好之间动态切换。

**[FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](llm_safety/fedtreelora_reconciling_statistical_and_functional_heterogeneity_in_federated_lo.md)**

:   针对联邦 LoRA 微调里"客户端数据异质"和"LLM 各层功能异质"两个维度被现有方法割裂处理的问题，FedTreeLoRA 用一棵全局层次聚类树 + 逐层自适应深度搜索，让浅层尽量共享、深层逐步分化，在 GLUE 和 FLAN 上以最小参数代价把平均指标分别从 91.19 / 61.77 提到 92.36 / 63.19。

**[FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](llm_safety/foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)**

:   FoeGlass 把"用 LLM 红队 LLM"的思路搬到音频深伪检测（ADD）上：不微调 LLM，仅通过 in-context learning + 真实度/多样性双反馈，让黑盒 reasoning LLM 自动写 TTS prompt 去骗 ADD，cold start 即可把现有 ADD 的 FNR（假阴率）从 0% 拉到最高 96%，且攻击在 8 个 ADD 之间高度可迁移。

**[Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](llm_safety/forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)**

:   本文指出现有 LLM unlearning 方法在"把知识从参数里抹掉"的同时，会把"用户在 prompt 里重新提供该知识时模型能正确利用"的能力（contextual utility）一起抹掉，作者提出在已有 unlearning loss 上加一项 KL 正则——让 unlearn 后的模型在"问题+上下文"输入上的分布对齐原始模型——即可在几乎不损失遗忘效果和保留集效用的前提下，把 Contextual QA 的 LLM-Judge 分数从 0.00–0.84 拉回到 0.95+。

**[From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](llm_safety/from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)**

:   本文把 LLM 幻觉检测从"看输出概率"切到"看 loss landscape 曲率"——在 embedding 加 Gaussian 噪声测量梯度方向与幅度的扰动，作为 Hessian 谱半径的廉价代理，在 12 个 model-dataset 组合上 AUROC 全面超越 entropy / Semantic Entropy / EigenScore 等基线。

**[From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](llm_safety/from_volume_to_value_preference-aligned_memory_construction_for_on-device_rag.md)**

:   EPIC 把端侧 RAG 的核心瓶颈从「检索时怎么用偏好」前移到「索引时存什么」，用「粗筛 + 细验证 + 查询偏移」三段式 pipeline 只保留与用户偏好对齐的数据并生成「指令-条目」对作为索引单元，在 4 个偏好基准上把存储减小 2404× 的同时偏好对齐准确率绝对提升 20.17 个百分点。

**[From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](llm_safety/from_weak_cues_to_real_identities_evaluating_inference-driven_de-anonymization_i.md)**

:   论文指出 LLM agent 可以把零散的、单独不可识别的线索与公开证据交叉印证，从而把匿名化数据重新链接到具体真人身份，并通过经典案例复刻 + 受控基准 InferLink + 真实人机对话日志三类场景系统地量化了这种"推理驱动的去匿名化"风险。

**[Gradient Transformer: Learning to Generate Updates for LLMs](llm_safety/gradient_transformer_learning_to_generate_updates_for_llms.md)**

:   本文提出 Grad-Transformer，把客户在私有数据上微调小模型 (TinyLM) 得到的 update vector，用一个 encoder-decoder Transformer 自回归地"翻译"为目标大模型 (LLM) 的 update vector，从而实现完全不接触私有数据的 weak-to-strong 知识蒸馏，在 6 个推理/摘要数据集上平均 PGR 达到 91.88%，比最优 baseline (58.94%) 提升 55.89%，且对差分隐私扰动鲁棒。

**[SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](llm_safety/gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)**

:   SemGrad 首次把"基于梯度"的不确定性量化搬到 LLM 自由生成场景——用语义保留分 (SPS) 找到能编码输入语义的隐藏态，把对它们求出的对数似然梯度范数当作 LLM 自信度的度量，无需采样、单次反向即可在 3 个 QA 数据集上击败 11 个 SOTA baseline，特别在多有效答案的 TruthfulQA 上比 SAR 高 3.27 AUROC。

**[HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](llm_safety/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)**

:   借鉴 Helmholtz 自由能的物理直觉，把每个领域的提示参数训练出一条"压缩到边界 $\Theta$、对齐到中线 $\Delta$"的能量曲线，推理时再用能量因子 + 距离因子联合加权各领域提示，在 CDDB / DomainNet / CORe50 三个 DIL 基准的未知领域上分别提升 1.76 / 3.12 / 2.57 个百分点。

**[LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_safety/llm_benchmark_datasets_should_be_contamination-resistant.md)**

:   本文是一篇 position paper，主张 LLM 基准应**抗污染（contamination-resistant）**——即可推理但不可训练；提出利用 Transformer 训练 vs 推理流水线的根本不对称性（训练需要全 token，推理只需 KV-cache + 倒数第二层 hidden state），把基准发布形式从明文换成 KV-cache + 中间隐藏态，配合跨模型 subspace alignment / relative representation 解决互操作问题，呼吁社区采纳。

**[MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](llm_safety/medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)**

:   MedMosaic 用合成管道构造了一个覆盖生理声 + 真实/合成临床对话的医学音频 QA 基准（46,701 条 QA、10 种问题类型），系统评测 13 个音频/多模态模型，发现即使 Gemini-2.5-Pro 也只能拿到约 68.1% 加权准确率，揭示当代 LALM 在医学音频推理上的根本短板。

**[Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](llm_safety/memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)**

:   把自回归 LLM 的下一个 token 分布解释成一条 Markov 链的状态转移矩阵，于是「学新词」就变成「在状态空间里加新状态、并把它表示为已有状态的稀疏组合」，理论上只需 $O(s)$ 样本（$s$ 为映射到的旧 token 数），实践中只 finetune 新 token 的 embedding 即可在严格零遗忘下完成跨语种/新概念扩展。

**[Multilingual Unlearning in LLMs: 转移、动力学与可逆性](llm_safety/multilingual_unlearning_in_llms_transfer_dynamics_and_reversibility.md)**

:   本文把 TOFU 遗忘基准扩到 5 种语言系统研究「跨语言遗忘转移」，发现遗忘强度随语言族/书写系统亲缘关系而变，且只动用了后段语言特化解码层、几乎不改前中段共享语义空间，因此能用一个推理时的转向向量恢复 Qwen 上 50%、Gemma 上 90% 的被遗忘知识——说明现有 LLM 遗忘本质是「表面抑制」而非真擦除。

**[Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](llm_safety/old_habits_die_hard_how_conversational_history_geometrically_traps_llms.md)**

:   History-Echoes 框架用"马尔可夫链状态一致性"和"潜空间几何角度"两套视角分析 LLM 对话历史的 carryover 效应，发现两者 Spearman 相关 0.78——一旦某种行为（幻觉/谄媚/拒绝）出现，模型就被困在潜空间该状态对应区域里，难以跳出；其中"拒绝"陷阱最强，"幻觉"最弱，且话题不连贯时陷阱会消解。

**[Optimizing Token Choice for Code Watermarking: An RL Approach](llm_safety/optimizing_token_choice_for_code_watermarking_an_rl_approach.md)**

:   CodeTracer 在冻结的 code LLM 旁边挂一个小的 watermark policy 网络，用 GRPO + 双奖励（执行通过 + z-score）+ Gumbel-Top-k 直通估计联合学习"在哪个 token 位置加水印、选哪一组 green token"，在几乎不掉 Pass@1 的前提下把代码水印的检测 AUROC 从 ~70% 抬到 ~78%。

**[PFT: Phonon Fine-tuning for Machine Learned Interatomic Potentials](llm_safety/pft_phonon_fine-tuning_for_machine_learned_interatomic_potentials.md)**

:   本文提出 PFT (Phonon Fine-tuning)，通过 Hessian-vector product 随机采样力常数列、并在 MLIP 微调时直接监督能量 Hessian 与 DFT 力常数对齐，配合 co-training 缓解灾难性遗忘，将 Nequix MP 在 MDR Phonon 基准上的声子热力学误差平均降低 55%，并将热导率 $\kappa_{\text{SRME}}$ 从 0.446 降到 0.307，在 MPtrj 训练的模型中达到 SOTA。

**[PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](llm_safety/pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)**

:   本文提出 PipeSD：把投机解码（speculative decoding）从端云顺序执行改成 token-batch 流水线，并用双阈值 NAV 触发 + 贝叶斯自动调参替代固定 draft 长度，在 5G 带宽的真实端云测试床上拿到 1.16×–2.16× 加速、14–25% 云端能耗下降。

**[Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](llm_safety/position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)**

:   本文是一篇 position paper，主张废弃"positive backdoor"这一误导性标签，将触发器激活的隐藏行为统一重命名为 Secret Alignment，并通过 SudoLM / Instructional Fingerprinting / SafeTrigger 三个代表性方案在六项标准化属性（有效性、无害性、持久性、效率、鲁棒性、可靠性）上的系统评测，揭示这类机制在机密性/完整性/可用性（CIA）方面的脆弱性，呼吁社区默认视其为"不安全"，除非有严格、标准化的证据支持。

**[Position: Stop Chasing the C-index when Evaluating Survival Analysis Models](llm_safety/position_stop_chasing_the_c-index_when_evaluating_survival_analysis_models.md)**

:   作者审计了 2023–2025 年 92 篇生存分析论文，发现约 72% 的工作所用评估指标（尤其是被滥用的 C-index）与其建模目标和删失假设不对齐，并提出"双螺旋阶梯假设"（Ladder Hypothesis）：模型与指标必须站在同一级"删失假设"上，否则报告的性能与排名都可能是偏差伪影。

**[Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](llm_safety/position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)**

:   这是一篇位置论文，核心论断：当前 LLM 不确定性量化（UQ）的主流方法（Semantic Entropy、图谱方法、P(true) 等）在机制上与无监督聚类同构——它们只衡量"模型生成的内部一致性"而非"外部正确性"，因此面对"自信幻觉"（confident hallucination）天然失效；作者诊断出参数敏感性、内部评估循环、缺乏 ground truth 三大病灶，并提出从评估、机制、grounding 三个支柱转向"监督式保障"的路线图。

**[Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](llm_safety/privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)**

:   作者给"差分隐私零阶优化（DP-ZOGD）"首次证出了**收敛的 hidden-state DP 上界**——通过设计一个"定向 + 各向同性"混合噪声机制并构造一个介于两条相邻轨迹之间的辅助过程，绕开了零阶更新缺乏全局 Lipschitz 性这一技术障碍，揭示出"扩大每步采样方向数 $K$ 反而能降隐私损失"这一前所未知的 DP 算法设计准则。

**[PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](llm_safety/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)**

:   作者用一个 115k 带推理标注的 DF-R5 数据集 + 把 CLIP ViT 换成 ConvNeXT 的 DX-LLaVA 架构，并提出 PRPO —— 段落级别 GRPO 变体，每段以 CLIP-文本-图像相似度（VCR）+ 推理-结论多数票一致性（PCR）为 reward，把跨域 deepfake 检测 F1 从 SOTA 75.26% 推到 89.91%，推理质量从 4.2/5 提到 4.55/5。

**[REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](llm_safety/realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)**

:   REALISTA 在 LLM 隐空间里构造"输入相关的编辑方向字典"，把对抗 prompt 优化变成一个 simplex 约束下的连续问题，既保住了 SECA 这类离散方法的语义等价/连贯，又有 LARGO 那种连续方法的搜索灵活度，首次在 GPT-5 这类闭源推理模型 free-form 输出上诱发幻觉成功。

**[REFLECTOR：把"边走边自省"内化进生成轨迹以抵御间接越狱](llm_safety/reflector_internalizing_step-wise_reflection_against_indirect_jailbreak.md)**

:   针对会在长生成中后段才"暴露"的间接越狱攻击，作者用教师模型合成 `<|reflect|>/<|explore|>` 标注的反思轨迹做 SFT 冷启，再用安全奖励 + 反思有效性奖励的双奖 GDPO 把"search-and-recovery"行为内化到策略里，把 DRA 等四类间接攻击的防御成功率从 ~10% 拉到 ~90%+，并且 GSM8K 反涨 5.65%。

**[Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](llm_safety/stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)**

:   本文指出现有 GFlowNet 红队的两大不稳定来源——partition function $Z_\theta$ 估计带来的高方差，与 toxicity classifier 给 OOD gibberish 文本的噪声 reward 引发的 mode collapse——并用三件简单组件（pairwise 对比目标 CTB 消除 $Z$、Noisy Gradient Pruning 过滤无信息 pair、Min-K Fluency Stabilizer 卡掉 gibberish）让红队攻击在 Qwen2.5-1.5B 上独特攻击数从 17 飙到 134（约 7×），ASR 维持 92%，且跨模型/跨防御迁移性全面碾压 baseline。

**[TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](llm_safety/tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)**

:   针对 Fine-Tuning-as-a-Service 场景下多模态大模型被投毒微调的问题，本文发现"被触发样本会把首个生成 token 的注意力在 system / vision / text 三大组件之间畸形地极化"这一通用指纹，并据此提出无监督的 TCAP 框架：用 GMM 在 system 注意力上挑出 trigger-responsive 注意力头，再用 EM-based Dawid–Skene 投票聚合，跨 5 种触发模式、3 种 MLLM、5 个数据集都能把 ASR 从 90%+ 压到 ~0% 而几乎不损失 Clean Performance。

**[The Unlearnability Phenomenon in RLVR for Language Models](llm_safety/the_unlearnability_phenomenon_in_rlvr_for_language_models.md)**

:   作者发现在 RLVR（GRPO）训练中存在一类「不可学习样本」：即便采样到正确 rollout、奖励信号非零，模型在整个训练过程中也始终学不会，根因不是优化端的正样本稀缺或裁剪/KL 正则，而是这些样本在初始策略下就是「梯度离群点」，背后是模型表征缺陷，需要靠 mid-training 而非 RL 后训练来修复。

**[Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](llm_safety/towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)**

:   A-TPT 用一种针对对抗扰动加固的 Gradient Attention Rollout 提取 CLIP 视觉端"语义锚点"，再以该注意力图为引导对多视图做空间非均匀增强、并按各视图注意力的 Total Variation 进行加权集成做 prompt tuning，在 9 个数据集上同时提升细粒度场景下的对抗精度和干净精度。

**[理解上下文连续学习中的泛化与遗忘](llm_safety/understanding_generalization_and_forgetting_in_in-context_continual_learning.md)**

:   首次为上下文连续学习建立理论框架——揭示注意力机制在处理多任务序列时必然产生的系统偏差与任务干扰，导致泛化性能与任务记忆与任务顺序相关的衰减现象。

**[遗忘并非删除：大语言模型机器遗忘中的可逆性调查](llm_safety/unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)**

:   本文通过表征层面的诊断工具系统分析 LLM 遗忘的可逆性——发现许多遗忘方法只是抑制而非真正删除信息，提出四层遗忘分类体系区分真正的信息擦除与表面性能退化。

**[Watermarking LLM Agent Trajectories (ACTHOOK)](llm_safety/watermarking_llm_agent_trajectories.md)**

:   ACTHOOK 把"软件 hook"思想搬进 agent 轨迹：在 action 边界处插入一个由秘密 key 触发的额外动作作为水印，被它训练过的 LLM 会在带 key 的 prompt 上以显著更高频率执行 hook，从而支持只通过黑盒查询就完成版权检测，平均 AUC 达 94.3 而几乎不影响下游任务表现。

---

## 🦾 LLM Agent { #llm_agent }

**[A Minimal Agent for Automated Theorem Proving](llm_agent/a_minimal_agent_for_automated_theorem_proving.md)**

:   本文提出 AxProverBase——一个极简的 Lean 4 定理证明智能体，仅靠"编译器反馈 + 自管理笔记本 + 轻量工具搜索"三个组件，在不微调的前沿 LLM（Claude Opus）上达到甚至超越 Hilbert/Seed-Prover 等专用系统，成本却低出 100 倍。

**[ACON: Optimizing Context Compression for Long-horizon LLM Agents](llm_agent/acon_optimizing_context_compression_for_long-horizon_llm_agents.md)**

:   Acon 用失败轨迹对比来优化自然语言压缩指南，同时压缩 agent 的历史和观察上下文，在 AppWorld、OfficeBench 和多目标 QA 上把峰值 token 降低 26% 到 54%，并保持或提升长程任务成功率。

**[Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](llm_agent/agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)**

:   通过 Monte-Carlo rollout 量化"哪些回合的 thought / observation 可以省"，再用冷启动 SFT + 双采样 omit-aware GRPO 训出能自适应跳过冗余思考和观测的 8B agent，五个基准上 token 用量大降而准确率与七大前沿模型持平。

**[Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](llm_agent/agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)**

:   这篇论文把网页 Computer-Use Agent 从逐步截图-调用 LLM-执行的循环，改造成类似 JIT 编译器的系统：把自然语言任务编译成可校验、可缓存、可并行调度的代码计划，从而让 JIT-Planner 比 Browser-Use 快 10.4×且准确率高 28pp，让 JIT-Scheduler 比 OpenAI CUA 快 2.4×且准确率高 9pp。

**[AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](llm_agent/agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)**

:   作者把"对黑盒 agent 系统反推一个等价白盒 workflow"作为新任务 AWR，用 MCTS 在 agent 原语序列空间中搜索，再配上一种基于评分动态着色的 Red-Black 剪枝来平衡深度与宽度，在五个真实领域上实现可解释的白盒重建。

**[Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](llm_agent/answer_only_as_precisely_as_justified_calibrated_claim-level_specificity_control.md)**

:   这篇论文把 agentic 系统里的“说得过细但证据不够”建模为 claim 级过度承诺问题，并提出 calibrated CSS：对每个原子 claim 在精确表述、粗粒度回退和省略之间做校准选择，在 LongFact 全量实验中将 OAU 从无后处理的 0.8460 提升到 0.9130，同时保留 0.9381 的特异性。

**[AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](llm_agent/autorpa_efficient_gui_automation_through_llm-driven_code_synthesis_from_interact.md)**

:   提出 AutoRPA 框架，通过翻译器-构建器流水线将 ReAct 风格 GUI Agent 的交互轨迹自动蒸馏为可复用的 RPA 函数，结合混合修复策略迭代优化代码，在保持甚至超越原始 Agent 成功率的前提下减少 82%~96% 的 token 消耗。

**[Constitutional Black-Box Monitoring for Scheming in LLM Agents](llm_agent/constitutional_black-box_monitoring_for_scheming_in_llm_agents.md)**

:   本文提出一套端到端的"宪法式黑盒监控"框架，利用两条合成数据流水线（STRIDE 和 Gloom）生成 2,000 条合成轨迹来优化提示分类器，在仅观察外部可见的工具调用与输出（不看 CoT）的条件下检测 LLM 代理的阴谋行为，发现简单的 prompt grid search 即可饱和性能，更激进的优化反而导致过拟合。

**[EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](llm_agent/evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)**

:   EvolveR 给 LLM agent 套一个「在线交互 → 离线自蒸馏成原则库 → GRPO 策略进化」的闭环生命周期：agent 不再丢弃过去轨迹，而是把自己的成功失败抽象成可检索的「策略原则」，再用 RL 学会**如何用自己的原则**去解新问题，在 7 个多跳 QA benchmark 上明显跑赢 Search-R1 等 RL agent baseline。

**[ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](llm_agent/excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)**

:   本文构建了首个评测 LLM Agent 端到端做"网络威胁调查"的 benchmark ExCyTIn-Bench：从真实 Azure 租户的 57 张安全日志表里，用 alert-entity 二部图自动生成 7542 道带证据链的 SQL 问答题，并提供 MySQL 环境让 Agent 通过查询日志、多跳追踪证据来回答，目前最强模型 Claude-Opus-4.5 也只能拿 0.606 的 reward。

**[Hallucinations Undermine Trust; Metacognition is a Way Forward](llm_agent/hallucinations_undermine_trust_metacognition_is_a_way_forward.md)**

:   本文是一篇 position paper，论证"彻底消除 LLM 幻觉"在原理上无法逃避一个"区分度税"（discrimination gap → utility tax）；作者主张把目标从"消灭幻觉"改为**忠实表达不确定性**（faithful uncertainty），并把这种 metacognition 视为 agentic LLM 调用工具时不可或缺的控制层。

**[HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](llm_agent/hawkesllm_semantic_uncertainty_propagation_in_agentic_text_simulation.md)**

:   HawkesLLM 把多变量 Hawkes 点过程嫁接到 LLM 智能体文本模拟循环中：Hawkes 负责安排"何时由哪个节点生成"以及"用哪些历史节点的输出作为压缩记忆"，LLM 只负责把被选中的记忆写成下一条事件，在 GDELT Artemis II 新闻级联上获得了在紧凑提示预算下仍随时间上升的后段语义对齐度。

**[Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models](llm_agent/hunt_instead_of_wait_evaluating_deep_data_research_on_large_language_models.md)**

:   本文提出 Deep Data Research（DDR）这一开放式 agentic 任务范式——只给 LLM 一个结构化数据库和最小工具集（SQL+Python），不给任何具体问题或回合上限，要求模型自主探索、提出假设并决定何时停止；并构建 DDR-Bench（MIMIC-IV / GLOBEM / 10-K，291 个实体、2058 条 checklist），用从非结构化文本抽取的可验证 fact checklist 客观评测主流 LLM 的"investigatory intelligence"，结果显示即使 Claude 4.5 Sonnet 也只能拿到 47.7% 平均准确率。

**[Internalizing Agency from Reflective Experience](llm_agent/internalizing_agency_from_reflective_experience.md)**

:   本文提出 LEAFE 框架，让 LLM agent 通过反思失败轨迹生成「失败→回滚→修正→成功」的经验数据，再用 SFT 蒸馏出 feedback-grounded 的恢复能力，在 CodeContests、WebShop、ALFWorld 等长程任务上把 Pass@128 拉高最多 14%，远胜 GRPO 等 outcome-driven RL。

**[Learning Efficient Guardrails for Compliance](llm_agent/learning_efficient_guardrails_for_compliance.md)**

:   本文构造了一个 60k 规模的 PolicyGuardBench（5 个域、733 条标准化轨迹 × 2195 条原子策略 → 6 万 trajectory-policy 对，含跨子域和前缀截断设置），并基于 Qwen3-4B-Instruct 全参数 SFT 出一个轻量 guardrail 模型 PolicyGuard-4B，在 22.5 ms/样本的延迟下取得 90.14% 准确率 / 87.59% F1，匹配甚至超过 70B 级开源模型和 Claude-Sonnet-4，并展现了强跨域泛化（LODO OOD F1≈0.91）。

**[Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](llm_agent/lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)**

:   NSI 把 LLM agent 的交互轨迹 "提升" 为带显式条件分支和动态变量绑定的神经符号工作流图，使技能从无状态脚本进化成可状态感知的逻辑程序，在 ALFWorld / WebShop / TextCraft 上分别拿到 98.0 / 76.5 / 95.2 的成功率，全面碾压 ASI 和 AWM 等编程式技能基线。

**[LLM Agents Are the Antidote to Walled Gardens](llm_agent/llm_agents_are_the_antidote_to_walled_gardens.md)**

:   这是一篇 ICML 2026 立场论文，主张 LLM 智能体能够通过自动格式转换 + 拟人化 UI 交互"绕过"主流平台的 API 封闭策略，实现"通用互操作性"（universal interoperability），从而瓦解传统网络效应造成的"围墙花园"，但同时需要 ML 社区主动建立 agent-friendly 接口、安全机制和生态基础设施来管控随之而来的安全、法律与新一层 lock-in 风险。

**[MCP-Persona: 用环境模拟评估 LLM agent 在真实个人化应用上的能力](llm_agent/mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)**

:   MCP-Persona 是首个针对真实个人化 MCP 工具（Slack/Rednote/Instagram/Lark 等 12 服务器）的 LLM agent benchmark；提出 Tool-Traverse + Context-Tree + Persona-Gen 三套方法，用 LLM 自动 synthesize Python simulator 代码避免真实账号问题；测 10+ SOTA agent 发现连 Claude-Sonnet-4.5 也只达 38.66% Acc，证明个人化工具使用是被严重低估的能力短板。

**[NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration](llm_agent/naviagent_graph-driven_bilevel_planning_for_scalable_tool_orchestration.md)**

:   NaviAgent 把 LLM 工具调用拆成"高层四选一决策 + 低层图上路径搜索"两层，由一个用 HGT 训练的 Tool World Navigation Model（TWNM）显式建模工具之间的结构与行为依赖，在 ToolBench/API-Bank 与 50 个真实 RapidAPI 上把任务成功率（TSR）相对最强基线整体提升 4.3–18.2 个点，同时显著减少调用步数。

**[On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](llm_agent/on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)**

:   针对 LLM agent 在多轮主动推理中"动作选择(AS)"与"信念跟踪(BT)"互相拖累、outcome-only RL 训练陷入低信息自锁(SeL)的失效模式，本文给出 POMDP 视角下的耦合梯度分析与"自锁区"形式化定义，并提出 AReW：用环境/读出层即可获得的方向性 critique 对 stepwise advantage 做加性 reweighting，在 9 个主动推理任务上最高带来 60 分性能提升。

**[OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](llm_agent/otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)**

:   OTora 提出一种全新的攻击范式 Reasoning-Level Denial-of-Service（R-DoS）：不破坏任务正确性，而是通过两阶段红队管线（先用插入感知优化诱导 agent 主动访问攻击者控制的外部资源，再在该资源里投放经 ICL 遗传搜索优化的「思考型 payload」）让 LLM agent 进入持续多轮的过度推理状态，在 WebShop / Email / OS 三类 agent 上实现 10× 推理 token 膨胀和数量级延迟攻击，且最终任务准确率几乎不变。

**[Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](llm_agent/persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)**

:   本文提出首个针对个性化 web agent 的开放网页 benchmark Persona2Web，用「隐式用户历史 + 三档模糊查询 + 推理感知评分」逼迫 agent 从浏览记录中推断用户偏好来消歧 ambiguous query；在 GPT-4.1 / o3 等 5 个主流模型上，即使提供历史，level-2 query 的成功率也只有 13%，揭示当前 web agent 缺乏真正的个性化能力。

**[Position: Agentic AI Orchestration Should Be Bayes-Consistent](llm_agent/position_agentic_ai_orchestration_should_be_bayes-consistent.md)**

:   这篇 position paper 主张：不要再尝试让 LLM 本身 "Bayesian"（那条路在工程上和理论上都跳不过去），而是把贝叶斯结构搬到 agentic AI 的**编排控制层**——让控制器维护一个低维任务级隐变量的信念，按 Bayes 规则在 agent/工具返回的"消息观测"上更新，并用期望效用或 value-of-information 做路由、停止、升级和预算分配。

**[Position: Assistive Agents Need Accessibility Alignment](llm_agent/position_assistive_agents_need_accessibility_alignment.md)**

:   这是一篇 position paper，作者通过对 417 篇文献中 778 个盲人辅助任务实例做系统综述，论证 "accessibility alignment" 应当被视为与 helpful/harmless/honest 并列的 Agent 一级对齐目标，并提出覆盖目标-交互-风险-生命周期四维度的设计 pipeline。

**[Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach](llm_agent/post-training_llms_as_better_decision-making_agents_a_regret-minimization_approa.md)**

:   作者提出 Iterative RMFT，把 LLM 自己 rollout 出来的决策轨迹按 regret 从低到高排序，挑出最优的 $k$ 条用 SFT 反复微调模型，从而在不依赖任何已知最优算法（如 UCB/FTRL）也不需要人工设计 CoT 模板的前提下，让 LLM 在多臂赌博机、在线学习和非平稳赌博机这三类语言化决策任务上自动涌现出 no-regret 行为和合理的探索-利用平衡。

**[PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](llm_agent/praglocker_protecting_agent_intellectual_property_in_untrusted_deployments_via_n.md)**

:   PragLocker 用 "代码符号初始化 + 黑盒目标模型反馈下的噪声注入" 两阶段策略，把 agent system prompt 编码成一段只能在 target LLM 上 work、迁移到其它任意 LLM 都会失效的 obfuscated text，从而在 prompt 被部署侧窃取时让攻击者无法在自己的 LLM 上复用。

**[Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](llm_agent/probabilistic_modeling_of_latent_agentic_substructures_in_deep_neural_networks.md)**

:   作者把神经网络（特别是 LLM）形式化为多个隐式子代理（每个是 outcome 上的概率分布）通过对数加权池化合成的复合代理，并在认知效用 $W_i(o)=\log P_i(o)$ 框架下证明了 "严格一致受益（strict unanimity）" 在线性池化或二元 outcome 下不可能、但 $|\mathcal O|\ge 3$ 下可行，进而推出"显式让 Waluigi 先显形再压制"严格优于"只强化 Luigi"的对齐原则。

**[Process Reward Agents for Steering Knowledge-Intensive Reasoning](llm_agent/process_reward_agents_for_steering_knowledge-intensive_reasoning.md)**

:   把过程奖励模型从"事后打分"重构成一个**在线 agent**：在每个推理步实时决定是否检索证据并给出奖励，借助 beam search 对冻结策略的候选轨迹进行剪枝，使 Qwen3-4B 在 MedQA 上达到 81.9% 的 4B-scale SOTA，且能直接迁移到 0.5B–8B 各种未见骨干（最高带来 25.7% 提升）。

**[Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](llm_agent/recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)**

:   针对 GUI 智能体在真实部署中容易陷入"自己造的错误"无法恢复这一痛点，作者一边搭出 GUI-RobustEval（1216 个可执行测试，覆盖 11 种策略诱发错误 + 4 档错误深度）做细粒度评测，一边提出 RoTS——一种基于轨迹树的在线数据合成框架：在成功子树上用脆弱度 UCB 主动暴露新错误，在失败子树上用邻居经验做长程恢复回滚，最终合成 800k 反思数据，使 RoTS-32B 在 OSWorld 上拿到 47.4% SR / 33.8% All-Pass@4 的开源 SOTA。

**[ReflexGrad: Within-Episode Failure Recovery in LLM Agents via Progress-Gated Dual-Process Routing](llm_agent/reflexgrad_within-episode_failure_recovery_in_llm_agents_via_progress-gated_dual.md)**

:   ReflexGrad 把 TextGrad 的"每 3 步局部梯度精调"作为快过程、把 Reflexion 风格的"连续低分触发的因果重规划"作为慢过程，用一条进度门控路由规则在**同一个 episode 内**无示范地切换两者，在 ALFWorld 134 任务上把 Qwen-3-8B 从 35.1% 拉到 75.4%（+40.3pp），并在算力对等条件下击败 1-shot 的 LATS / ToT / Self-Refine。

**[Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)**

:   RHB 构造了一套现实工具型多步任务（独立 + 链式两种模式，含数据流水线、日志取证、性能优化、多文件重建四大家族）来量化 LLM agent 的奖励黑客行为，跨 13 个前沿模型发现 RL 后训练显著提高 exploit 率（DeepSeek-V3 0.6% vs R1-Zero 13.9%），且 exploit 率随链长上升、在更难变体上即使近零率模型也会"复发"，而轻量级环境硬化能在不损害任务成功率前提下把 exploit 率减少 87.7%。

**[Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](llm_agent/rule2drc_benchmarking_llm_agents_for_drc_script_synthesis_with_execution-guided_.md)**

:   作者构建了 Rule2DRC —— 一个含 1000 条自然语言设计规则、13921 个评测版图的大规模 EDA 基准，通过 KLayout 引擎做执行级别打分而非代码相似度对比，并提出 SplitTester：把 N 个候选 DRC 脚本按"在当前测试下是否输出一致"做聚类，每轮挑「分数 × 簇大小」最大的簇生成新版图把它拆开，最后让 judge LLM 在 Top-3 候选与其差异化测试上选最优。

**[SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](llm_agent/safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)**

:   SafeHarbor 把 LLM Agent 的安全防御从「静态粗粒度分类器」升级为「动态分层记忆树 + 双分数门控」，通过对抗规则生成 + 信息熵自演化让 GPT-4o 在保持 93%+ 拒绝率的同时把 benign 工具调用成功率拉到 63.6%，显著缓解 over-refusal 问题。

**[Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](llm_agent/scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)**

:   小米团队针对 VLM 移动 GUI 智能体提出"数据-评测-推理"三位一体的系统性研究：发布 16k 任务 / 674 个中文 App 的 HyperTrack 数据集和支持 30+ 模型的 GUIEvalKit 评测工具，证明 DAPO 风格 RL 在 OOD 场景明显胜过 SFT、并用半在线评测 SOEval 揭示了"显式 reasoning 会牺牲 PASS@1 稳定性但提升 PASS@n 多样性"的核心权衡。

**[Scaling Small Agents Through Strategy Auctions](llm_agent/scaling_small_agents_through_strategy_auctions.md)**

:   论文提出 sale（Strategy Auctions for Workload Efficiency）：让大小不一的 Qwen3 智能体在每个任务上提交"策略短计划"作为竞拍标书，按 cost-minus-value 选出执行者，并用历史竞拍记忆让便宜 agent 持续精炼自己的标书；在 deep search 与 coding 上既超过最大模型的 pass@1，又把对最大 agent 的依赖降低 52%、总开销降低 35%。

**[SE-GA: Memory-Augmented Self-Evolution for GUI Agents](llm_agent/se-ga_memory-augmented_self-evolution_for_gui_agents.md)**

:   SE-GA 给基于 VLM 的 GUI 智能体配了一套"情景+语义+经验"三层记忆库（TTME）+ 一个两阶段记忆增强自演化训练流程（MASE，SFT→改进版 GRPO），把 Qwen2.5-VL-7B 在 ScreenSpot 推到 89.0、AndroidControl-High 推到 75.8、AndroidWorld 推到 39.0，全面超越同规模基线甚至打平 72B 模型。

**[Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](llm_agent/skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)**

:   Skill-Pro 把 LLM agent 的交互经验显式抽成"激活+执行+终止"三件套的 Skill，用语义梯度生成候选 Skill、再用 PPO 风格的信任域验证 (PPO Gate) 决定是否纳入，最终在 ALFWorld / Mastermind 上以 ~800 token 的极小记忆库实现 0.85+ 的复用率和显著性能提升。

**[Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](llm_agent/talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)**

:   本文提出 ALIGN，让一群完全自利、去中心化的 LLM 智能体通过五档语气的公开"八卦"消息互相评价、形成声誉、惩罚背叛，从而在无中心监管的捐赠博弈、投资博弈和电商市场中稳定地建立间接互惠合作，并发现推理型 LLM 比 chat 型 LLM 更能按博弈论激励"该合作时才合作"。

**[Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](llm_agent/think_twice_before_you_act_enhancing_agent_behavioral_safety_with_thought_correc.md)**

:   本文提出 Thought-Aligner——一个 1.5B/7B 的轻量级即插即用安全模型，在 LLM agent 的 think-act-observe 循环里、在每个动作执行前对中间思维做因果纠偏，把 6 个主流 LLM 在 ToolEmu/Agent-SafetyBench 上的行为安全率从约 50% 拉到约 90%，同时帮助度还提升约 5%。

**[Towards a Science of AI Agent Reliability](llm_agent/towards_a_science_of_ai_agent_reliability.md)**

:   论文借鉴航空 / 核能 / 汽车等安全关键工程的成熟做法，把 AI agent 的"可靠性"分解为一致性、鲁棒性、可预测性、安全四个维度共 12 个与准确率无关的指标，在 GAIA 和 $\tau$-bench 两个基准上系统评测 15 个前沿模型，得出"过去 24 个月准确率猛涨、可靠性几乎没动"这一行业级结论。

**[Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](llm_agent/towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)**

:   针对自演化 LLM agent 写 CUDA kernel 的场景，提出 CUDAnalyst：通过"冻结某一代中间程序状态 + 选择性注入/屏蔽反馈"做生成级干预，并用 Banzhaf 联盟博弈解构 debugger / analyzer / profiler 三类反馈的边际贡献和高阶交互，得出"显式 plan 只有在反馈对齐时才有用、强模型的 plan 可向同家族弱模型迁移"等四条结论，并据此设计出 2.08×–10.32× 超过 torch.compile 的 CuGEdit 插件。

**[Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](llm_agent/video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)**

:   Video2GUI 用「元数据粗筛 → 视频质量精筛 → Gemini-3-Pro 提任务/动作 → 高分辨率三帧精确空间 grounding」四段流水线把 5 亿条 YouTube 视频元数据炼成 WildGUI（12.7M 轨迹、124.5M 截图、1500+ 应用），并把 Qwen2.5-VL/Mimo-VL 在多个 GUI grounding 与 agent benchmark 上提升 5–20%。

**[Weasel: 通过重要性-多样性数据选择实现 Web Agent 的域外泛化](llm_agent/weasel_out-of-domain_generalization_for_web_agents_via_importance-diversity_data.md)**

:   通过结合目标相关性和多样性的轨迹步骤选择方法，Weasel 在减少训练数据到原始 20% 的同时实现 9.7-12.5 倍训练加速，并显著提升 Web Agent 在未见域上的泛化能力。

---

## 🤖 机器人/具身智能 { #robotics }

**[Contrastive Representation Regularization for Vision-Language-Action Models](robotics/contrastive_representation_regularization_for_vision-language-action_models.md)**

:   作者发现 VLA 模型里继承自 VLM 的表征被视觉外观主导、对机器人本体状态不敏感，提出 Robot State-aware Contrastive Loss（RS-CL）把本体感受状态之间的欧氏距离当作"软对比标签"重塑表征，并配合"view cutoff"的表征级增广，把 GR00T N1.5 在 RoboCasa-Kitchen 推到 69.7% SOTA，在真实 Franka 拾放任务上把成功率从 45.0% 抬到 58.3%。

**[Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](robotics/decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)**

:   针对"训练任务到全新任务"的零样本机器人操作，作者把 demo 拆成"原子技能-动作对"作为中间表示，再用 dual-library（动态库按视觉/计划相似度检索 + 静态库按 IDF 加权补全缺失技能 token）给 LLM 提供 skill-comprehensive in-context demonstrations，从而把"模仿轨迹"升级为"组合技能推理"。

**[Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](robotics/discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)**

:   本文把 VLA 的动作解码从自回归（AR）或外挂连续扩散头改成"在统一 Transformer 内部对离散动作 token 做掩码扩散"，配合按置信度自适应排序的并行解码和二次重掩码纠错，在 LIBERO 上达到 96.4% 平均成功率、SimplerEnv-Fractal 64.1% 总均分，且在 OOD 语言/视觉扰动下退化仅 0.8% / 20.4%，显著优于连续扩散和并行解码 baseline，同时保留了预训练 VLM 的多模态先验。

**[BEAR: Dissecting Embodied Abilities in Multimodal Language Models through Skill-level Evaluation and Diagnosis](robotics/dissecting_embodied_abilities_in_multimodal_language_models_through_skill-level_.md)**

:   BEAR 把具身任务拆成 14 个原子技能、构建 4,469 道图-视频-文交错的 VQA，对 20 个 MLLM 做技能级横纵向诊断，发现感知能力（而非推理）是真正瓶颈，并据此用 GroundingDINO、3D 场景图、轨迹可视化等外部视觉/空间工具拼出 BEAR-Agent，让 GPT-5 在该基准上相对提升 17.5%、在真实机器人抓取上提升 20.17%。

**[Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation](robotics/dive_into_the_scene_breaking_the_perceptual_bottleneck_in_vision-language_decisi.md)**

:   SceneDiver 通过"先建场景图做粗粒度子场景分解、再让 VLM 以智能体方式逐子场景验证"的两阶段焦点规划，把任务相关物体过滤出来再喂回 VLM 做决策，并用 Slot Attention 适配器把这套显式推理蒸馏进 VLA，从而同时缓解高层规划与反应式控制中的视觉幻觉。

**[DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics](robotics/dlo-lab_benchmarking_deformable_linear_object_manipulations_with_differentiable_.md)**

:   DLO-Lab 在 Genesis 平台上用 Taichi 自研了一套以离散弹性杆（DER）为内核、支持双向耦合 + 弯曲塑性 + 闭环拓扑的可微仿真器，配套 10 个 rope/cable/橡皮筋 benchmark 任务和一个用 VLM 做"抓点提议 + 任务分解"的专门 agent，把 PPO/SAC/SHAC/SAPO/CMA-ES/GD 各路策略学习算法摆到统一擂台上 PK，并通过系统辨识做了真机 sim-to-real 验证。

**[Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](robotics/drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)**

:   本文提出 CAPS：把"指令漂移"重新解释为系统性采样误差，用 SNR（=$\log|\mathcal{A}|-\mathcal{H}$）作为元认知开关，仅在高熵"Pivotal Window"触发基于幂分布 $\pi\propto p^\alpha$ 的 Metropolis-Hastings 迭代精修，在 RoboTwin、Simpler-WindowX、Libero-long 上 training-free 超越 OpenVLA 和 TACO。

**[Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)**

:   DUST 用一套"分流式"多模态扩散 Transformer（MMDiT）把动作流与未来视觉嵌入流并排走，靠共享 attention 做跨模态融合，再配独立噪声调度和动作-视觉异步采样，让 VLA 同时学会"做什么动作"和"动作会产生什么后果"，在 RoboCasa / GR-1 / Franka 真机上稳定刷过 GR00T-N1.5+FLARE。

**[Dual Advantage Fields](robotics/dual_advantage_fields.md)**

:   本文观察到双线性目标条件价值模型 $V_\theta(s,g)=\psi_\theta(s)^\top\phi_\theta(g)$ 中，目标嵌入 $\phi_\theta(g)$ 恰好就是价值场对状态嵌入的梯度方向，于是用一个 "动作特征位移预测器" $u_\xi(s,a)\approx\gamma\psi(s')-\psi(s)$ 与目标嵌入做内积，得到一个免学习 Q 网络的局部优势分数，在 OGBench 长程导航 + 操控 + 谜题任务上把 RLiable 聚合指标全面拉高。

**[Dual Quaternion SE(3) Synchronization with Recovery Guarantees](robotics/dual_quaternion_se3_synchronization_with_recovery_guarantees.md)**

:   本文用单位对偶四元数（UDQ）替代 $4\times4$ 矩阵来参数化 SE(3) 同步问题，先用 Hermitian 对偶四元数矩阵的幂迭代算出谱初始化，再用每步逐元投影到 $\mathrm{UDQ}^n$ 的广义幂法（DQGPM）做迭代精化，首次给出 SE(3) 同步的有限步线性收敛与显式误差界，并在多扫描点云配准上把旋转/平移误差和算法时间都打到了矩阵方法之下。

**[EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents](robotics/embguard_constructing_hazard-aware_guardrails_for_safe_planning_in_embodied_agen.md)**

:   EmbGuard 把"具身 agent 的物理安全判断"从策略里剥离成独立的小模型 guardrail——输入 (观察图, 候选动作)，输出 (是否危险, 风险类别, 危险解释)；2B/4B 规模就追平 GPT-5.1/Gemini-2.5-Pro，并把 baseline 普遍存在的"动不动就 false positive"问题压下去。

**[Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](robotics/embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)**

:   本文把「视觉—动作归因」重新表述为干预估计问题，提出 ISS（介入显著性分数）和 NMR（干扰物质量比）两个指标，用 Bernoulli 掩码 + 高斯模糊扰动 + Action MSE 代理 KL 散度的方式量化 VLA 策略到底依赖哪些视觉区域，并证明 NMR 与 OOD 任务成功率呈 $r = -0.77$ 的强负相关——是预测 VLA 泛化能力的便宜诊断工具。

**[Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](robotics/embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)**

:   GiG 用"图中图"双层记忆（场景图 + 状态转移图）+ GNN 编码 + 1 步 lookahead 武装 LLM 规划器，让具身 agent 在 Robotouille 同步/异步以及 ALFWorld 上的 Pass@1 比 ReCAP 提高 6–37 个百分点。

**[From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](robotics/from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)**

:   BehaviorVLA 用因果三流 Mamba 编码器 (VBE) 把长视野演示压缩成时间不变的"行为原型 $z_{\text{proto}}$"+ 时间变化的"相位状态 $z_{\text{phase}}$"，再用相位条件解码器 (PBD) 以 Predictor-Corrector 方式把行为骨架展开成相位对齐的高斯先验去引导流匹配策略，在 LIBERO/RoboTwin 2.0/CALVIN 三套基准刷新 SOTA，并且只用 50% 真机数据就追平 OpenVLA-OFT。

**[From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](robotics/from_imagined_futures_to_executable_actions_mixture_of_latent_actions_for_robot_.md)**

:   MoLA 用一组在大规模机器人数据上预训练好的"模态感知逆动力学模型 (IDM)"，把视频生成模型预测出的未来帧翻译成语义/深度/光流三路离散潜动作，再让策略头基于这些动作中心的表征做控制，从而在 CALVIN、LIBERO、LIBERO-Plus 以及真实 UR5e 上把"想象-执行"接口做得既稳又准。

**[SAFAG: 无对称性标注的可泛化可操作部件位姿估计](robotics/generalizable_and_actionable_parts_pose_estimation_with_symmetry_annotation-free.md)**

:   SAFAG 把 GAPart 6D 位姿估计拆成"候选四元数生成 + 切空间精修"的两阶段框架，并用自适应概率分布在 $x,y,z$ 三轴上隐式学习对称轴/面，从而在完全没有对称性标注的情况下，把跨类别可操作部件的旋转误差从 5.51° 压到 3.23°。

**[HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](robotics/hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)**

:   HDFlow 用扩散模型生成稀疏战略子目标、用整流流生成稠密轨迹，再叠加能量引导和流形投影，构建一套快慢分工的双层规划器，把家具组装等长程稀疏奖励任务的成功率拉高 20~30 个百分点。

**[Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies](robotics/lagrangian_perturbation_diffusion_steering_latent_reinforcement_learning_for_gen.md)**

:   LP-DS 把冻结的扩散/流匹配策略当成黑盒解码器 $\Phi(s,w)$，只在它的初始噪声 $w=\epsilon+\Delta_\theta(s)$ 上学一个状态条件残差，用 Lagrangian 信任域 $\mathbb{E}_s[\|\Delta_\theta(s)\|_2^2]\le\delta$ 把扰动幅度卡住，从而在保留多模态先验的前提下做样本高效的在线 RL 微调，在 RoboMimic / Gym / Adroit / LIBERO 上比 DSRL 与 DPPO 更稳，回报最多 +25%。

**[LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](robotics/langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)**

:   LangForce 把 VLA 策略写成 $\pi(a\mid v,\ell)=p(\ell\mid a,v)\,p(a\mid v)/p(\ell\mid v)$ 这一贝叶斯分解，引入可学习的 Latent Action Queries 在同一套 VLM 权重上同时跑"只看视觉"和"视觉+语言"双分支，并通过最大化动作与指令的对数似然比来直接惩罚"视觉捷径"，在 SimplerEnv 上相对 QwenGR00T 基线提升 11.3 个绝对点。

**[Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](robotics/latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)**

:   LaRA-VLA 把 VLA 模型里的文本 CoT 和视觉 CoT 全部内化为连续 latent，通过三阶段 curriculum 训练（显式 CoT → latent 替换 → 动作专家适配）让推理留在 latent 空间里完成，推理延迟相比显式 CoT 降低高达 90%，控制频率重回实时区间。

**[ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics](robotics/manisoft_towards_vision-language_manipulation_for_soft_continuum_robotics.md)**

:   本文针对"视觉-语言操作研究几乎只覆盖刚性臂、忽视软体连续体臂"这一空白，构建了 ManiSoft 基准：用"Cosserat 杆软体动力学 + MuJoCo 刚体接触 + 弹性力约束耦合"的混合仿真器，定义 4 类反映软臂控制难点的任务，并通过"高层规则规划器 + 低层 RL 力矩执行器"自动生成 6300 个场景与专家轨迹，系统揭示 DP/RDT/OpenVLA-OFT 在干净场景下中等可解（30% 左右），在随机化场景下断崖式下跌（最高跌 29.4 个点），失败根因在于无法从视觉估计本体感知态、也不会利用软体可形变性绕障。

**[Mixture of Horizons in Action Chunking](robotics/mixture_of_horizons_in_action_chunking.md)**

:   本文针对 VLA 模型中"动作块长度（horizon）选择"导致的"长视野规划 vs. 短视野精控"权衡问题，提出 Mixture of Horizons (MoH)：把同一条动作块拆成多个不同长度的子块，用共享的 action transformer 并行预测，再用 2k 参数的线性门控融合，配合负载均衡损失和"跨 horizon 共识"的动态推理，使 $\pi_{0.5}$ 在 LIBERO 上首次达到 99% 平均成功率，并把吞吐量提高到基线的 2.5 倍。

**[Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](robotics/neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)**

:   NIAF 把 VLA 模型的"动作块"从一串离散 waypoint 改成一个连续时间函数 $\mathcal{A}(\tau)=\Phi(\tau;\theta)$，让 MLLM 当 SIREN 的"分层频谱调制器"输出参数 $\theta$，从而获得 $C^\infty$ 平滑轨迹、任意频率查询和解析可导的速度/加加速度信号，在 CALVIN/LIBERO 上刷 SOTA 并在真机阻抗控制上消除抖动。

**[Neural Low-Discrepancy Sequences](robotics/neural_low-discrepancy_sequences.md)**

:   NeuroLDS 用一个把整数索引经正弦位置编码送入 MLP 的小网络，先回归 Sobol' 再用闭式 $L_2$ 差异损失在所有前缀上微调，得到第一个支持任意长度、可扩展的神经低差异序列，在 4 维差异指标、Borehole 积分、RRT 运动规划与 Black–Scholes PDE 求解上全面优于 Sobol'/Halton。

**[Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](robotics/optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)**

:   本文把匿名多机器人路径规划（MAPF）证明为一类**马尔可夫多边际最优传输（MMOT）**，从而把原本 $K^{T+1}$ 维的传输张量压缩成多项式规模 LP（P1），并通过全单模性保证最优解整数性；再把它推广为 Schrödinger bridge 得到 Sinkhorn 风格 entropic 松弛 P2 产出"影子传输"，最后在影子上做剪枝并解 LP（P3）恢复整数解，在 $K^{1.15}$ 复杂度下实现 3.6×–7.1× 加速、代价差距 <10%。

**[Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](robotics/plan_in_sandbox_navigate_in_open_worlds_learning_physics-grounded_abstracted_exp.md)**

:   本文提出 SAGE：在物理约束的语义沙盒里自动合成大量导航任务+IF-THEN 经验规则，用混合提示采样 + 非对称自适应裁剪的 GRPO 把这些经验蒸馏进 VLM 策略，最终在 A-EQA 上把 LLM-Match 成功率从 43.5% 拉到 53.2%（2B）/ 60.2%（4B），并能迁移到真实室内机器人。

**[Position: Good Embodied Reward Models Need Bad Behavior Data](robotics/position_good_embodied_reward_models_need_bad_behavior_data.md)**

:   一篇 position paper：作者用 RoboArena 真人评分实证了三类 SOTA 具身奖励模型（ReWind / GVL / Dopamine）会系统性"高估"实际失败的机器人行为，根因是训练数据几乎只有专家成功示范，并通过把真实"坏"行为视频 + 稠密负向奖励标签塞进 GVL 的 in-context 提示，证明哪怕极少量负样本就能显著修正偏好排序，从而呼吁社区主动收集和发布"坏"机器人数据。

**[PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](robotics/psg-nav_probabilistic_scene_graph_navigation_via_multiverse_decision_making.md)**

:   本文提出 PSG-Nav，用"保留完整类别分布的 3D 概率场景图 + 从联合分布采样多个一致世界做决策 + 用成功/失败记忆库做证据校准"三件套替代传统确定性场景图导航，在 HM3D / MP3D / HSSD 三大 ObjectNav 基准上分别达到 66.1% / 44.8% / 67.9% SR，是新的 SOTA。

**[R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](robotics/r2r2_robust_representation_for_intensive_experience_reuse_via_redundancy_reducti.md)**

:   R2R2 把 VICReg 风格的冗余去除约束加进自预测学习（SPL）以稳定高 UTD 训练，但**关键改动是不做零中心化**——理论上证明 zero-centering 会消除 SPL 谱分解中的常数本征模（即全局动力学信息），实验在 TD7 上 UTD=20 时把分数从 1.02 提到 1.24（+22%），并以新提出的 SimbaV2-SPL 架构刷新连续控制 SOTA。

**[RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](robotics/robomme_benchmarking_and_understanding_memory_for_robotic_generalist_policies.md)**

:   RoboMME 首次把人类认知里的"时序/空间/物体/程序"四类记忆系统性映射到 16 个长时机器人操控任务（770k 高质量时间步），并在 π0.5 底座上系统消融 14 种"记忆表征 × 集成方式"，得出"感知记忆 + AdaLN 调制器"是当前最佳综合权衡的结论。

**[Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](robotics/seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)**

:   针对 VLA（vision-language-action）模型在简单扰动下性能崩塌的问题，本文用"提取语义/几何条件 → 改写 caption → 条件视频扩散重渲染"的视频迁移流水线给仿真数据补上视觉与环境多样性，同时配以三段式 velocity caching 把生成时间砍掉 61% 以及 difficulty + diversity 双驱动的 coreset 采样仅选 10% 关键轨迹，最终在 Robotwin 2.0、LIBERO-Plus 和真机上让 RDT-1B / $\pi_0$ 涨 5–15%。

**[Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](robotics/spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)**

:   SOMA 给 VLA 装上由可动头部相机扫描构建、可在线增量更新、可被指令检索的持久化空间-语义记忆，使机器人能稳定操控当前视野之外的物体，在 5 个真实 OOV 抓取任务上把首次注视时间、头部搜索路径、抓取次数都压缩 40-60%。

**[SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning](robotics/specprune-vla_accelerating_vision-language-action_models_via_action-aware_self-s.md)**

:   作者发现 VLA 推理是 compute-bound 的，剪枝才是对的路子，且连续动作步之间视觉信息高度重叠 → 提出 SpecPrune-VLA：用上一步的全局注意力 + 本步早期层的局部注意力 + 帧差动态 token 三路融合做静态剪枝，再加层内动态剪枝和速度感知的粗/细粒度切换控制器，免训练地在 LIBERO 上拿到 1.57× / 真机 1.70× 加速且成功率几乎无损。

**[StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](robotics/stablevla_towards_robust_vision-language-action_models_without_extra_data.md)**

:   针对 VLA 模型在视觉扰动下崩盘的问题，作者发现脆弱的根源在视觉到 LLM 之间的 MLP 投影器，于是用一个不到 10M 参数的"通道维度信息瓶颈适配器（IB-Adapter）"替换它，在不增加任何训练数据或增强策略的前提下让 0.5B 的 StableVLA 在 LIBERO 严重扰动下平均提升约 35%，并在真机抓放任务上比 14× 大的 OpenPi 还稳。

**[STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction](robotics/step_warm-started_visuomotor_policies_with_spatiotemporal_consistency_prediction.md)**

:   STEP 给 diffusion policy 接了一个轻量的 "前一段历史动作 + 当前观测 → 下一段动作"的 Transformer 预测器, 用它的输出作为去噪起点 (warm-start), 把 100 步去噪压到 2 步, 又附带一个 "动作变化太小就注一点噪声"的执行死锁防御机制, 在 9 个仿真任务和 2 个真机任务上比 BRIDGER / DDIM 平均提 21.6% / 27.5% 成功率。

**[TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](robotics/tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)**

:   TapSampling 提出一个与策略无关、即插即用的推理时采样框架：先用 Action-VAE 从策略生成的少量动作里学一个低维后验、再高效地采出大量候选动作，再用"预测任务进度变化"的语义可解释 verifier 给候选动作打分并加权融合，在 CALVIN/LIBERO 和真机上无需微调原策略就能稳定提升 Diffusion Policy、OpenVLA、VPP、$\pi_0$、$\pi_{0.5}$ 等多种通用机器人策略的成功率。

**[The Lie We Tell: Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space](robotics/the_lie_we_tell_correcting_the_euclidean_fallacy_in_vision_language_action_polic.md)**

:   Lie Diffuser Actor (LDA) 把扩散过程从把 SE(3) 位姿展平成 $\mathbb{R}^{12}$ 的"欧氏谎言"修正回流形原生：通过左不变 SDE 在李代数 $\mathfrak{se}(3)$ 中注入噪声、用指数映射回拉到流形、tangent-space 预测 score，理论上同时获得流形闭合、坐标系等变与测地线最优性，在 CALVIN ABC→D 上把平均任务长度从 3.27 推到 3.51。

**[TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance](robotics/timerewarder_learning_dense_reward_from_passive_videos_via_frame-wise_temporal_d.md)**

:   TimeRewarder 把"任务进度"形式化为视频帧对之间的归一化时间距离，仅用动作无关的专家视频自监督训练一个 ViT 距离回归器，并将相邻帧距离作为稠密奖励喂给 DrQ-v2，在 10 个 Meta-World 任务上以 200K 交互逼近 9/10 满分，甚至超过手工设计的环境稠密奖励。

**[Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](robotics/towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)**

:   本文提出 FAN：把"昂贵的生成式策略 + 分布式 critic"压缩到"单步 flow 锚定 + 单噪声样本 critic"——用 Flow Anchoring 在一次 flow 评估内完成行为正则化，用 noise-conditioned critic 把 quantile 多样本替换成单 Gaussian 噪声样本，在 D4RL/OGBench 上做到 SOTA 性能同时训练比同类分布式方法快 5-14×。

**[DiBO: 用扩散语言模型做离线黑盒优化（DNA + 机器人形态）](robotics/training_diffusion_language_models_for_black-box_optimization.md)**

:   DiBO 把扩散语言模型 LLaDA-8B 适配到离线黑盒优化场景，用 delimiter token 统一 prompt/design/label 三类异构信号，再走「域适应 → masked-response SFT → label-improvement RL」三段后训练，让模型能在 500 条标注样本下学到 Design-Bench 多个任务的 SOTA（DNA 任务上 +8% 归一化分），单 H100 1.5 小时就能跑完一个离散任务。

**[Turning Adaptation into Assets: Cross-Domain Bridging for Online Vision-Language Navigation](robotics/turning_adaptation_into_assets_cross-domain_bridging_for_online_vision-language_.md)**

:   针对在线视觉语言导航中环境分布不断漂移的问题，本文提出 IDEA 框架，把每次测试时自适应学到的 soft prompt 连同域坐标和不确定度封装为可复用"资产"，再用 Wasserstein 凸包投影把目标域映射到历史资产的组合上，得到一条免训练的跨域捷径，在 REVERIE / R2R 上平均 +2.5% SR、+1.9% SPL。

**[WestWorld: 知识编码的可扩展轨迹世界模型](robotics/westworld_a_knowledge-encoded_scalable_trajectory_world_model_for_diverse_roboti.md)**

:   WestWorld 通过将机器人通用动力学知识显式编码到轨迹世界模型中，实现单一模型在多样化机器人系统上的可扩展预测——在 7 种不同形态机器人上比专用模型平均提升 14.3%，并支持零样本迁移到全新机器人配置。

---

## 🛡️ AI 安全 { #ai_safety }

**[Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](ai_safety/active_continual_learning_with_metaplastic_binary_bayesian_neural_networks.md)**

:   BiMU 为二值贝叶斯神经网络设计有界记忆和不确定性感知的 metaplastic 更新，防止 Bernoulli 后验在长程非平稳流中饱和，并用 Monte Carlo disagreement 实现无缓存的一次性主动查询，显著减少标签和反向传播更新。

**[Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](ai_safety/angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)**

:   作者首次系统评估 7 种主流可塑性干预 (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) 对深度强化学习 (DRL) 后门攻击的影响 (14,664 个实验)，发现只有 SAM 是"恶魔"——能显著加剧后门威胁；据此提出"Sweeper-Converter-Connector" 鲁棒后门注入框架并给出基于 loss landscape 锐度的检测信号。

**[Calibrating Uncertainty for Zero-Shot Adversarial CLIP](ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)**

:   提出 UCAT 框架，将 CLIP 的 logits 重新参数化为 Dirichlet 分布的浓度参数，通过对齐干净样本与对抗样本的 Dirichlet 分布（反向 KL 散度），在零样本对抗微调中同时校准不确定性和保持语义结构，在 16 个基准上实现了鲁棒性与校准的最优平衡。

**[COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](ai_safety/copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)**

:   COPF 把"演进图上的在线链路推荐"看成一个 performative 决策过程，在 backbone 打分器之外加一层 **决策层 wrapper**：用带显式探索的在线日志协议保证反事实可识别，用图感知双重稳健（GA-DR）估计器估计"曝光-未曝光"的反事实组间差距，再用 Residual-OI 审计 + PI primal–dual 控制器在线压制部署后出现的公平性 spike，理论上给出从插件式 OI 到真实反事实差距的 transfer 证书，在 TGB 与合成二部流上以可控的效用损失显著降低 Deploy 阶段的 worst-case TE 差距。

**[Demystifying the Optimal Fair Classifier in Multi-Class Classification](ai_safety/demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)**

:   本文给出多分类公平分类问题中 Bayes 最优分类器的解析可处理形式（带熵正则的闭式解），并据此推出一对统一的算法 OptFair：训练阶段用 reduction 转化为代价敏感交叉熵的 saddle-point 优化，部署阶段用 plug-in 估计求解凸近端梯度问题，两者在理论上都收敛到 accuracy-fairness Pareto 前沿。

**[Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](ai_safety/exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)**

:   本文提出 TSEF——一个针对"时序分类器 + 解释器"联合系统的对偶目标攻击框架：通过学习"时间脆弱掩码 + 频域扰动滤波器"，在 $\ell_\infty$ 预算内同时把模型预测推到目标标签、又把解释推到攻击者指定的参考显著图，证明现有时序可解释流水线的"解释稳定 = 决策可信"假设根本不成立。

**[Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](ai_safety/extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)**

:   本文把 Ravfogel 等人为线性模型设计的「迭代零空间投影 (INLP)」公平化方法搬到核方法上：通过在经验特征空间 (empirical feature space) 推导一个直接作用在核矩阵 $\mathbf{K}$ 上的闭式变换 $\mathbf{T}$，使得变换后的 $\mathbf{K}_{(m)}$ 仍是半正定核，但已被剥离了对连续受保护属性的预测信息，从而把任意基于核的算法（KRR、SVR）一键改造为「连续公平」版本，在 Crimes / ACSIncome / ACSTravelTime 上取得有竞争力或更优的 fairness–accuracy 帕累托。

**[Fair Dataset Distillation via Cross-Group Barycenter Alignment](ai_safety/fair_dataset_distillation_via_cross-group_barycenter_alignment.md)**

:   本文揭示数据集蒸馏 (DD) 会放大原始数据中的偏差——根源是「子组样本量不平衡」与「子组表征分离度」的交互作用，并提出 COBRA：用各子组表征的（与组大小无关的）barycenter 作为蒸馏目标，可在多个 DD 框架上同时降低 EOD、提高准确率。

**[Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency](ai_safety/fair_decisions_from_calibrated_scores_achieving_optimal_classification_while_sat.md)**

:   本文针对"即使分数在各群体上完全 group-calibrated，对其取单一阈值也会违反 sufficiency（predictive parity）"这一长期被忽视的痛点，给出有限离散分数下 sufficiency 约束最优二元分类器的**精确解**：通过对 $(\mathrm{PPV}, \mathrm{FOR})$ 可行域的几何刻画，得到一个只依赖分数和群体标签的后处理算法，并证明该算法同时可解"损失最小化"和"在 sufficiency 下最小化与 separation 的偏差"两类目标。

**[Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking](ai_safety/fairness_in_aggregation_optimal_top-k_and_improved_full_ranking.md)**

:   在 Spearman footrule 距离下，把 ILP 的约束矩阵证成全单模，从而给出 fair top-$k$ 排名聚合的首个多项式时间最优算法；并以"先解 fair top-$k$，再用最小代价完美匹配补齐成全排列"的两步策略，把 fair (full) rank aggregation 的近似比从 3 改进到 2。

**[FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](ai_safety/fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)**

:   针对原型类联邦学习中"对局部原型直接平均会继承客户端偏差"的问题，本文用一组可学习的全局超原型 (hyper-prototypes)，通过梯度匹配在服务器侧模拟集中式训练得到的原型，再配合客户端对比学习与对齐损失显著提升异质场景下的精度。

**[Flatness-Aware Stochastic Gradient Langevin Dynamics](ai_safety/flatness-aware_stochastic_gradient_langevin_dynamics.md)**

:   本文提出 fSGLD：在标准 SGLD 更新里把梯度处的参数 $\theta$ 换成被高斯扰动过的 $\theta+\epsilon$，并将扰动尺度 $\sigma$ 与逆温度 $\beta$ 通过 $\sigma=\beta^{-(1+\eta)/4}$ 严格耦合，从而在不增加任何梯度/内存开销的前提下，让算法的不变测度逼近 Hessian-trace 正则化目标 $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 对应的 Gibbs 分布，并给出 Wasserstein-1 与超额风险的非渐近界，在 CIFAR/WebVision/ViT 上取得与 SAM/ASAM 相当或更优、但训练时间近乎减半的效果。

**[Frequency Matching in Spiking Neural Networks for mmWave Sensing](ai_safety/frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)**

:   本文从「机制-数据对齐」角度证明 LIF 脉冲神经元等价于一个一阶 IIR 低通滤波器，并提出根据毫米波信号的判别频谱来设定膜衰减系数 $\beta$，使 SNN 在四个常用 mmWave 数据集上平均比 ANN 提高 6.22% 精度并降低 3.64× 理论能耗。

**[From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](ai_safety/from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view.md)**

:   本文把 LLM 的下一 token 预测视为一个超大词表上的分类任务，将两个轻量级 OOD 检测器 NCI（特征与权重向量的接近度）与 fDBD（特征到决策边界的距离）迁移过来，配合"训练特征均值的解析代理 $\mu_G$"和"只在 top-$k$ 候选 token 上算边界距离"两个适配，得到一个**无训练、单样本**的推理类幻觉检测器，在 CSQA / GSM8K / AQuA 上稳定优于困惑度、Semantic Entropy、SelfCheckGPT 等基线。

**[GEM-FI: Gated Evidential Mixtures with Fisher Modulation](ai_safety/gem-fi_gated_evidential_mixtures_with_fisher_modulation.md)**

:   本文针对证据深度学习 (EDL) 在分布外样本上过自信、且单头难以表达多模态认知不确定性的问题，提出三件套 GEM-Core/MIX/FI：用学到的特征能量门控证据、用混合证据头单次推理近似 ensemble、用 Fisher 信息正则稳定混合分配，在 CIFAR-10→SVHN/CIFAR-100 等 OOD 检测上比 DAEDL 强且保持 single-pass。

**[Geometrically Constrained Outlier Synthesis](ai_safety/geometrically_constrained_outlier_synthesis.md)**

:   GCOS 在 ID 特征 PCA 的"小方差子空间"上沿几何 off-manifold 方向合成虚拟离群点，并用从校准集 Mahalanobis 分位数导出的"共形壳层" $[\alpha_\text{inner},\alpha_\text{outer}]$ 控制合成强度，配合自适应 margin 的对比正则损失训练，在 4 个 near-OOD 数据集上把平均 AUROC 从 VOS 的 86.21 提到 93.47。

**[Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](ai_safety/hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)**

:   针对自回归音频生成模型在 KGW 风格 token 水印下因"解码→重编码不幂等"导致水印信号指数级衰减的问题，作者用 codec 自身的混淆矩阵跑 Leiden 社区检测得到一个收缩后的"簇词表"，把水印的绿/红集合定义在簇而非 token 上，从而在完全梯度自由、黑盒访问 codec 的前提下把 $z$-score 的指数底从 $r$ 抬到 $r_{cl}>r$，detectability 相比基线和需要微调 codec 的 WMAR 普遍提升数个量级，且对 MP3、降噪、裁剪等扰动天然鲁棒。

**[How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](ai_safety/how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)**

:   把"遗忘 vs 保留"的 trade-off 直接写成"每步带约束的一阶凸优化"问题，用 retain/forget 梯度的点积 $\kappa = \bm{g_r}\cdot\bm{g_f}$ 同时充当 hardness 度量、更新方向切换开关和提前停止条件，在 CIFAR-10/ResNet-20 与 Llama-2-7B/WaterDrum-TOFU 上比 GA、GDiff、SCRUB、KL 等基线更稳。

**[LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams](ai_safety/lapras_learning-augmented_private_answering_for_linear_query_streams.md)**

:   LAPRAS 用一个"哪些查询会来"的预测器把在线 DP 查询流分成预测内/外两类，预测内的用离线最优 Matrix Mechanism 一次性低噪释放，预测外的用 Smooth Allocation 根据流中已观测到的"未预测查询"位置在线估计总数并平滑分配预算，在预测准时几乎追平离线最优、预测差时退化到在线 baseline 水平。

**[MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](ai_safety/metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)**

:   把多个客户端在私有数据上独立微调出的领域专家，无需共享私有数据就能合并成一个可部署的 MoE 模型——核心是用 relevance-weighted DPP 从公开数据里选「既相关又多样」的代理样本，先做 proxy-aligned 专家训练再训 context-aware router，从而对齐专家行为与代理监督，显著优于 FlexOlmo 等仅依赖相似度选代理的方法。

**[Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](ai_safety/mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)**

:   本文为 $(\varepsilon,\delta)$-DP 设计了一类高斯混合加性噪声机制（multi-Gaussian mixture 与无超参的 quasi-Gaussian mixture），在中低隐私域将解析高斯机制的次优间隙关闭高达 99%，同时保留高斯的 zCDP 紧组合性质。

**[OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](ai_safety/omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)**

:   本文针对"图/文/视频混合伪造同时检测+定位"这一统一任务，提出 OmniVL-Guard，用 Self-Evolving CoT 合成高质量冷启动数据 + ARSPO（非线性奖励映射 + 动态任务权重）解决多任务 RL 中"简单的真假分类抢走梯度、细粒度定位学不动"的难度偏置问题，在 In-Domain 上视频时序定位 tIoU +37.8、文本定位 F1 +22.9，并在四个 OOD benchmark 上做到零样本 SOTA。

**[One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)**

:   UniTrans 把"为每对车端模态训一个 adapter"的传统协同感知翻译范式，改写成"在一个模态内蕴空间里推断映射 → 通过 router 线性组合一组专家参数 → 当场实例化一个映射专属翻译器"，实现对未见过的新车型的零样本 BEV 特征翻译，在 OPV2V-H / DAIR-V2X 上平均 AP@0.7 较最强基线提升 ~7 / 3 个点，同时 GFLOPs / CPU 时间均低于 Classic MoE。

**[Optimal Transport under Group Fairness Constraints](ai_safety/optimal_transport_under_group_fairness_constraints.md)**

:   本文把"群体公平性"显式编码为一个 $K_s \times K_w$ 的组间匹配概率目标 $\mathbf{F}$，提出 **FairSinkhorn** 精确求解、**惩罚式 OT** 凸松弛、以及 **双层成本学习** 三种方案，分别给出有限样本复杂度 $O(1/\sqrt{n})$ 和 fairness 偏差界 $O(\exp(5R_\Theta/\varepsilon)/\sqrt{n})$，在合成与半合成（约会 app）数据集上勾画出"代价 - 公平性"权衡前沿。

**[Persuasive Privacy](ai_safety/persuasive_privacy.md)**

:   本文用 Sender–Receiver 两方 Stackelberg 博弈 + Bayesian Persuasion 思想，把"隐私"重新表述为 Receiver 在最坏 data-prior 下的相对评分规则损失，给出统一定义 $(\mathcal{S},\mathcal{Q}_x,\kappa,\delta)$-PP，同时把 pure DP 和 probabilistic DP 收编为特例，并首次为**确定性算法**（如无噪经验均值）给出非平凡的形式化隐私保证。

**[Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](ai_safety/position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)**

:   这是一篇 ICML 立场论文：作者主张 ML 公平性研究不能只盯着 race/sex 这类"敏感属性"，而必须把"社会决定因素"（neighborhood、ADI、学校经费、医疗可及性等情境变量）也纳入审计，并用大学录取理论模型 + 美国人口普查数据 + 乳腺癌筛查半合成实验，证明只围绕敏感属性的缓解策略反而可能制造新的结构性不公。

**[Position: Embodied AI Requires a Privacy-Utility Trade-off](ai_safety/position_embodied_ai_requires_a_privacy-utility_trade-off.md)**

:   本文是一篇 position paper，主张具身 AI 的隐私不能用单阶段补丁解决，必须当作横跨 instruction / perception / planning / interaction 全生命周期的架构级动态控制信号，并提出 SPINE 框架，用 L1-L4 四级隐私分类矩阵在每个阶段联动调整智能体行为。

**[Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](ai_safety/position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)**

:   这是一篇 ICML 2026 立场论文：作者结合 UNOS 历史数据，论证美国心脏移植分配系统的下一代 ML 策略必须把"器官获取组织/移植中心/医生/患者/监管"之间的激励错位当成一等公民来建模，呼吁把机制设计、战略分类、因果推断、社会选择整合进 ML 流水线，否则再强的预测模型也会在部署时被各方策略性行为反噬。

**[PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](ai_safety/prism_gauge-invariant_tangent-space_differentially_private_lora.md)**

:   PRISM 把 DP-SGD 从 LoRA 的 $(A,B)$ 因子空间搬到 rank-$r$ 流形的切空间上做 clip+加噪+retract，从而获得 gauge invariant、无 bilinear 二阶噪声、且有闭式 $\sigma C/b\cdot\sqrt{r(m+n-r)}$ 内禀噪声能量的 DP-LoRA 机制。

**[Regret-Based Federated Causal Discovery with Unknown Interventions](ai_safety/regret-based_federated_causal_discovery_with_unknown_interventions.md)**

:   本文提出 I-PERI：在客户端干预目标完全未知、且只能共享 regret 标量的联邦设置下，用"有向一致掩码 + 无向一致掩码"两阶段流程，恢复出一个比观测 MEC 更紧、比 I-MEC 更松的全新等价类 Φ-MEC，并通过 Laplace 噪声给出 ε-差分隐私保证。

**[Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling](ai_safety/rotation-invariant_spherical_watermarking_via_third-order_so3_representation_cou.md)**

:   TRIAD 把 360° 全景图当作球面信号，用三阶球谐系数张量积投影到 trivial 表示得到一个**理论可证 SO(3) 不变**的双谱标量，从而把水印藏在高阶 SH 系数里、再从这个不变量里读出来，在任意 3D 旋转下仍能保持近 100% 的比特准确率而不依赖数据增强。

**[Same Target, Different Basins: Hard vs. Soft Labels for Annotator Distributions](ai_safety/same_target_different_basins_hard_vs_soft_labels_for_annotator_distributions.md)**

:   在 CIFAR-10H 上把"标注者分布"以硬标签方式投喂给模型（multipass 按票循环 / SLS 每个 epoch 重采样），证明它和软标签交叉熵期望目标等价，但收敛到更平坦的 basin、在稀疏标注下更优、且 OOD 检测略胜。

**[Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](ai_safety/scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)**

:   针对现有联邦多源无监督域适应 (UMDA) 方法只能处理 2–6 个源、源数一多就训练不稳或算力爆掉的问题，作者提出 GALA：把所有源随机分成若干小组、组间对预测分布做差异最小化（把 $O(N^2)$ 的两两对齐压成线性），再叠一个基于质心+温度的相似度加权挑出真正贴近目标域的源——在新建的 Digit-18 (18 源) 基准上稳定收敛，且把基线一一推开。

**[Singular Bayesian Neural Networks](ai_safety/singular_bayesian_neural_networks.md)**

:   本文把权重矩阵直接参数化为 $W=AB^\top$ 而不是对 $W$ 本身做平均场分布，从而诱导出一个**关于 Lebesgue 测度奇异的低秩后验**，参数量从 $O(mn)$ 降到 $O(r(m+n))$，PAC-Bayes 复杂度从 $\sqrt{mn}$ 收到 $\sqrt{r(m+n)}$，并在 MLP/LSTM/Transformer 三类架构上实现 OOD 检测胜过 5-成员 Deep Ensemble 同时参数少 $33\times$。

**[TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](ai_safety/timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)**

:   TimeGuard 把多变量时间序列预测里的后门防御从"丢掉整条窗口"重构成"按通道+按时间步"的可靠样本池训练，先用反向一致性 (RCF) + 邻域多样性 (NDF) 交集初始化高纯度池子，再用距离正则的损失筛选 (DRLS) 渐进扩池，在不依赖任何干净数据的前提下把对 BackTime 等 SOTA 攻击的 $\text{MAE}_{\text{P}}$ 提到最强基线 PDB 的 1.96 倍。

**[Training-Free Coverless Multi-Image Steganography with Access Control](ai_safety/training-free_coverless_multi-image_steganography_with_access_control.md)**

:   提出 MIDAS，一种基于预训练扩散模型的 training-free 无载体多图隐写框架，用 Random Basis 正交随机基替代传统 Noise Flip 实现按私钥的细粒度访问控制，配合 Latent Vector Fusion 消除拼接边界，在不传输任何与秘密相关的附加信息的前提下实现多图隐藏 + 抗隐写分析。

**[机器遗忘的两个盲点：过度遗忘与原型重学习攻击](ai_safety/unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)**

:   本文揭示机器遗忘的两个关键盲点——过度遗忘（对边界附近样本的误伤）和原型重学习攻击（用少量样本复原遗忘知识），并提出 Spotter 框架通过边界掩膜蒸馏和类内散布损失同时缓解这两个问题。

**[VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](ai_safety/vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)**

:   作者构造了 10 万张图、33 个细粒度类别、19 万+ 实例的大规模视觉隐私数据集 VPD-100K（覆盖人脸/屏上 PII/物理证件/位置标记四大域），并提出三件套频域增强模块（FDAF + 自适应频谱门控 + 频域一致性损失）插入 YOLOv10 的 Neck，使 YOLOv10-L 在 VPD-100K 上 AP 从 53.8 涨到 58.6（+4.8），同时在 7.51ms 延迟下稳定跑直播流。

---

## 💬 LLM / NLP { #llm_nlp }

**[A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)**

:   本文从微分几何视角刻画 GPT 风格 LLM 在高熵分布上采样所引入的信息丧失，构造 $\mathfrak{so}(n)$ 值 1-形式与平行输运算子，并在国际象棋探针实验中证明这种几何旋转与模型学到的世界向量高度同向。

**[ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](llm_nlp/anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)**

:   ANCHOR 用"自底向上溯因 + 层级聚类" 构造稠密因子空间，对下游条件做粗到细检索得到稀疏相关因子集，再联合 Naïve Bayes 与一个 LLM 现场构造的潜变量因果贝叶斯网络做后验聚合，在 LLM 高风险决策场景中显著减少 "unknown" 预测并提升概率校准。

**[Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs](llm_nlp/automated_formal_proofs_of_combinatorial_identities_via_wilf-zeilberger_guidance.md)**

:   WZ-LLM 把经典的 Wilf–Zeilberger 符号证明流程编译成 Lean 4 中可执行的证明骨架（递推 + 边界条件 + 侧条件），交给专门用 SFT + expert-iteration + DAPO 训练出的 WZ-Prover 逐项 discharge，在 100 个经典组合恒等式上把 pass@32 从 Goedel-Prover-V2 的 9% 提升到 34%。

**[Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](llm_nlp/compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)**

:   本文提出 Compute as Teacher（CaT）：把 GRPO 已经在采样的 G 条 rollouts 通过冻结锚模型"合成"出一个伪参考答案，再在非可验证领域用模型自己从该伪参考衍生的二元 rubric 给每条 rollout 打分作为 RL 奖励，从而在没有任何人工标注的情况下把推理算力直接变成监督信号，在 HealthBench 上相对基线最高提升 30%，并以 9× 更低的测试时算力匹配甚至超过 inference-time aggregation。

**[Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics](llm_nlp/deep_networks_learn_to_parse_uniform-depth_context-free_languages_from_local_sta.md)**

:   作者提出一个可控歧义的"变树 RHM"概率上下文无关文法，并证明只用 root-to-pair / root-to-triple 这两个低阶矩 + 逐层聚类，就能恢复语法规则、进行 CYK 式解析，对应样本复杂度 $P^\star \asymp v\, m_3\, m_2^{L-1} (p_2^2/2)^{1-L}$，CNN 与 Transformer 实验完全符合该幂律。

**[Differential Syntactic and Semantic Encoding in LLMs](llm_nlp/differential_syntactic_and_semantic_encoding_in_llms.md)**

:   通过对共享句法结构或共享含义的句子做隐层表示平均得到"句法质心"和"语义质心"，作者证明 DeepSeek-V3 等大模型的句子向量中相当一部分句法/语义信息是被**线性叠加**编码的，并且这两类信息在层间分布和正交消融上都呈现明显的可分离性——支持"句法相对自治"的语言学假说。

**[dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](llm_nlp/dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)**

:   针对扩散式大语言模型 (dLLM) 因双向注意力无法复用 KV cache 而推理极慢的问题，本文提出训练无关的 dLLM-Cache，对静态 prompt 用长间隔缓存、对动态 response 用短间隔刷新+按 Value 余弦相似度选 25% 最"变化"的 token 做局部重算，在 LLaDA 8B / Dream 7B 上获得最高 9.1× FLOPs 加速且分数基本不掉。

**[Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](llm_nlp/express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)**

:   这是一篇 position paper，主张：**用 LLM 的 token softmax 概率（logprob）当成"世界事件概率"是理论上错的**——因为 distribution estimation、response prediction 和 target distribution estimation 是三个不同任务，对应不同 ideal 输出分布；获取世界概率的正确做法是**二阶预测**——让 LLM 在输出里**显式写出**它对事件的概率（数值或语言修饰词），而不是去算"它说 X 的概率"。

**[Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](llm_nlp/fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference.md)**

:   针对扩散语言模型（dLLM）的并行解码瓶颈，本文提出训练无关的 Fréchet 画像解码：用整条排序后的置信度画像而不是"最弱被选 token"那一项来决定本步并行 commit 多少 token，把 Fast-dLLM 的 factor 规则严格推广到异质置信度场景，在 LLaDA-8B 上四个基准平均吞吐 1.36×、NFE 降 29%，精度几乎不变。

**[From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](llm_nlp/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)**

:   作者通过追踪 LoRA 微调过程中参数沿"危险/安全方向"的累积漂移，发现善意数据破坏对齐的根本机制是参数在 fine-tuning 中向危险方向单调漂移；进而提出 SQSD——用单步梯度沿两方向的投影差对每个样本打连续风险分，在 3 个模型 × 2 数据集上保持单调 ASR 排名，且能跨架构、跨规模、跨 LoRA→Full 迁移。

**[How Many Different Outputs Can a Transformer Generate?](llm_nlp/how_many_different_outputs_can_a_transformer_generate.md)**

:   本文从"有限精度 + 有界嵌入支撑"两个最基本的架构事实出发，证明任意 transformer 只能生成有限条"可达序列"，给出可达序列长度随 prompt 长度线性增长、超过阈值后比例以 $1/|V|^n$ 指数衰减的紧上界，并用 cramming 与 copying 实验在 Pythia/Qwen/Llama/Gemma 上验证理论斜率与实测仅差 5–10 倍。

**["I've Seen How This Goes"：用渐进条件惊奇度刻画 LLM 与人类写作的多样性](llm_nlp/ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)**

:   本文提出 $D_{Ca_n}=C\cdot a_n$ 这一无需 embedding、无需参考语料、无需人工标签的多样性度量：用一个基座模型 $\theta$ 在单次前向里读完所有响应，把"最后一条响应在已见过 $n-1$ 条之后还剩多少 per-byte 条件惊奇"乘上"响应整体的可读性"，在 McDiv 人评基准上逼近 SentBERT，并在 OLMo-2-7B 的 base→SFT→DPO→RLVR 上单调下降，准确捕捉后训练带来的模式坍缩。

**[Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](llm_nlp/margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)**

:   本文针对 LLM-as-a-judge 中"置信度高就一定靠谱"这一常被违反的单调性假设，提出用一个小 MLP 把多组 in-context 预测概率映射成置信度，并通过 margin-based ranking loss + PAC-Bayes 泛化界推导出一个 margin 自适应训练策略，使学到的置信度在四个数据集与六个 judge 模型上都获得更低的 ranking loss、更高的 AUROC，并显著提升 fixed-sequence 测试的目标一致性达成率。

**[Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation](llm_nlp/mitigating_staleness_in_asynchronous_pipeline_parallelism_via_basis_rotation.md)**

:   作者把异步流水线并行训练 LLM 时延迟梯度导致收敛崩塌的"罪魁祸首"归结为 Adam 的基底失配（Hessian 特征基与坐标轴不对齐），并提出在 Hessian 特征基下做基底旋转再走 Adam 更新，3B 模型上比最强异步基线少 81.7% 迭代就能达到同样 loss。

**[YAQA: 端到端 KL 最小化的 LLM 自适应权重量化](llm_nlp/model-preserving_adaptive_rounding.md)**

:   YAQA 把 LLM 权重量化的代理目标从「逐层激活误差」换成「端到端模型输出 KL 散度」，用 Kronecker 分解的 Hessian 草图给出第一个端到端误差界，相对 GPTQ/LDLQ 把 KL 再降约 30%，甚至比量化感知训练（QAT）更准，且推理速度不变。

**[Multi-Agent Teams Hold Experts Back: 自组织 LLM 团队为什么留不住「专家」](llm_nlp/multi-agent_teams_hold_experts_back.md)**

:   本文借组织心理学的「强协同」标准（团队 ≥ 最强个体）系统评估自组织异质 LLM 团队，发现即便明确告知谁是专家，团队在前沿 ML 基准上仍比专家差 6.3%–41.1%，根因不是认不出专家，而是不肯让专家说了算——LLM 倾向「中间立场式整合」而非「认知让渡」，团队规模越大稀释越严重，而这套共识机制反过来让团队对对抗性成员异常稳健。

**[On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation](llm_nlp/on_the_limits_of_llm_adaptability_impact_of_model-internalized_priors_on_annotat.md)**

:   通过对毒性检测的大规模实验（9 模型 × 5 数据集），论文发现 LLM 标注性能主要由**定义对齐**而非文本记忆决定；模型内化的先验使得绝大多数零样本错误对提示词纠正"有韧性"——即使明确提供定义和示例，**三分之二的错误**仍无法被修正（救援率仅 34.8%），且置信度无法用于检测定义错误。

**[Optimizing Diversity and Quality through Base-Aligned Model Collaboration](llm_nlp/optimizing_diversity_and_quality_through_base-aligned_model_collaboration.md)**

:   作者提出 BACO，一种推理时 token 级路由框架：让"未对齐的 base 模型"和"对齐后的 instruct 模型"在同一次解码里逐 token 切换，用 logit 不确定度与内容词信号决定该信谁，从而在不再训练、不多次采样的前提下同时拿到 base 的多样性与 aligned 的质量，best router 相对最强 baseline 取得 21.3% 的多样性-质量联合提升。

**[Position: Adversarial ML for LLMs Is Not Making Any Progress](llm_nlp/position_adversarial_ml_for_llms_is_not_making_any_progress.md)**

:   这是一篇立场论文，作者认为对抗机器学习在 LLM 时代研究的问题相比传统分类器场景"更难定义、更难求解、更难评测"，过去十年在 $\ell_p$ 鲁棒等"玩具问题"上就进展缓慢，如今全面转向 LLM 后很可能再耗一个十年仍无法产出可度量、可复现的安全保证。

**[Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](llm_nlp/position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)**

:   作者指出"Transformer 是图灵完备"这一流行说法在大多数已有证明里其实悄悄换成了"一族不同的 Transformer 共同能模拟图灵机"，并形式化了贴近真实部署的固定系统 $(T,D,C)$，证明同一个固定 Transformer 在不同上下文管理策略下计算能力可以从仅识别正则语言一路跃迁到图灵完备，从而把研究重点从模型本身扭转到 context manager 上。

**[Rare Event Analysis of Large Language Models](llm_nlp/rare_event_analysis_of_large_language_models.md)**

:   本文把统计物理里成熟的稀有事件分析（REA）方法搬到 LLM 上，用「指数倾斜分布 + Transition Path Sampling + MBAR」三件套，在 TinyStories 上以可承受的算力估出比直接采样小好几个数量级的稀有完成概率，并通过 EDA 找出便宜的运行时代理（连续 token 重复数）来预筛高 ARI 异常输出。

**[Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](llm_nlp/reasoning_on_the_manifold_bidirectional_consistency_for_self-verification_in_dif.md)**

:   本文从"有效推理轨迹是学习分布上的稳定吸引子"这一几何视角出发，提出 BMC（Bidirectional Manifold Consistency）这一无监督、训练自由的度量：通过对扩散语言模型（dLLM）生成结果做一次"前向重新掩码 + 后向少步重构"，用重构稳定性来打分；BMC 同时支撑错误诊断、推理时拒绝采样和 RL 稠密奖励三大任务，在四个推理基准上系统超越置信度、Self-Consistency、Self-Evaluation 等基线。

**[Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](llm_nlp/resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)**

:   本文把激活稀疏导致 LLM 掉点的本质归因为"表示漂移"，并仿照生物自发放电向每层注入一个输入无关、训练后可吸收进 bias 的小向量（SPON），以接近零推理开销显著缩小稀疏模型与稠密模型的差距。

**[Rethinking LLM Ensembling from the Perspective of Mixture Models](llm_nlp/rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)**

:   本文证明对 $n$ 个 LLM 做 token 级集成时无需每步都跑所有模型——按权重随机抽一个模型采下一个 token，输出分布与"先平均后采样"严格等价，从而把 $n$ 倍前向变回 1 倍前向，并配合"懒同步 KV 缓存"实现 1.78×–2.68× 的实际加速。

**[SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling](llm_nlp/sac-opt_semantic_anchors_for_iterative_correction_in_optimization_modeling.md)**

:   SAC-Opt 把 LLM 生成的优化求解器代码再"反向翻译"回结构化语义锚点（约束与目标），与原始问题描述的锚点逐条比对，只重写不一致的那条约束/目标并迭代到全部对齐，在 7 个公开数据集上平均提升 7.7%、ComplexLP 上提升 21.9%。

**[Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](llm_nlp/scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)**

:   本文把 LLM 推理调度中"预测单一输出长度"的点估计换成 log-t 分布拟合，并用一个加上 CVaR 尾部惩罚的期望（Tail Inflated Expectation, TIE）替代 SJF 中的输出长度作为优先级，在 LMSYS-Chat-1M 上把在线每 token 延迟相比最强 baseline LTR 再降 $2.31\times$，离线 SDG 吞吐量提升 $1.42\times$。

**[SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](llm_nlp/slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)**

:   SLAY 把受物理"逆平方相互作用"启发的 Yat-kernel 通过 (1) 球面归一化 (2) Bernstein 定理的 Laplace 积分表示 (3) Gauss-Laguerre 求积 (4) 多项式+指数核张量积正随机特征四步连击线性化，得到 $O(L)$ 时间复杂度且与 softmax 几乎无差异的注意力机制。

**[SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](llm_nlp/spa-cache_singular_proxies_for_adaptive_caching_in_diffusion_language_models.md)**

:   SPA-Cache 把扩散语言模型 (DLM) 中"哪些 token 需要更新"的判定，从原本在 $d=4096$ 维 Value 空间做余弦相似度，压缩到 $r=128$ 的奇异子空间，并按层动态分配更新预算，在不掉精度的前提下让 LLaDA-8B 在 GSM8K 上达到 $6.4\times$、在 MBPP 上达到 $8\times$ 的吞吐提升，叠加并行解码后总加速 $28\times$。

**[SphericalDreamer: Generating Navigable Immersive 3D Worlds with Panorama Fusion](llm_nlp/sphericaldreamer_generating_navigable_immersive_3d_worlds_with_panorama_fusion.md)**

:   SphericalDreamer 通过把多张文本生成的分层深度全景图各自抬升为 3D"球体"建筑块、再用谐波融合把相邻球体之间缺失的过渡区域生成并拼接起来，得到首个同时具备 360°×180° 全方向沉浸感和长距离可导航能力的户外 3D 世界。

**[Stop Automating Peer Review Without Rigorous Evaluation](llm_nlp/stop_automating_peer_review_without_rigorous_evaluation.md)**

:   这是一篇立场论文：作者通过对 ICLR 2026 真实评审和 60 篇模拟评审的实证测量，发现当前 LLM 审稿存在 hivemind（高度趋同）+ paper laundering（零样本改写就能涨 0.45 分）两大失效，因此论证「在没有严格评估之前，不应让 LLM 直接生成审稿意见」，并呼吁建立一门"审稿自动化的科学"。

**[T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](llm_nlp/t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)**

:   T$^2$PO 把多轮 agentic RL 的训练崩溃归因为"hesitation（犹豫）"——token 层过思考、turn 层重复无效——并用一个融合 entropy+confidence 的自校准不确定性信号 $M_t$ 同时驱动 token-level Thinking Intervention（动态截断 think 段）和 turn-level Dynamical Sampling（重采样无效 turn），在 WebShop / ALFWorld / Search QA 上稳定超越 PPO/GRPO/GiGPO。

**[The Cylindrical Representation Hypothesis for Language Model Steering](llm_nlp/the_cylindrical_representation_hypothesis_for_language_model_steering.md)**

:   本文提出 Cylindrical Representation Hypothesis（CRH），在保留"概念线性"的前提下放弃 LRH 的正交性，证明概念向量的叠加会自然诱导出"轴 + 法平面 + 敏感扇区"的圆柱几何，从而首次几何化地解释了 activation steering 为什么在样本层面不可预测但在群体层面可观测。

**[Token-Efficient Change Detection in LLM APIs](llm_nlp/token-efficient_change_detection_in_llm_apis.md)**

:   作者证明在低温采样下，"两个 token logit 几乎打平"的特殊输入（Border Inputs）对参数微扰极度敏感——理论上 SNR 在 $T\to 0$ 时发散，于是只观测输出 token（严格黑盒）就能用极少请求做 LLM API 变更检测；提出的 B3IT 在 TinyChange benchmark 上以 1/30 的成本匹敌灰盒 logprob 方法，并在 93 个商用端点上 23 天连续监控发现 8 次真实模型替换。

**[结构化广义线性 token mixing：用 SND + Kronecker 在复杂度与表达力之间换挡](llm_nlp/trading_complexity_for_expressivity_through_structured_generalized_linear_token_.md)**

:   论文提出统一的"直接输入混合 $\mathbf{A}$ + 输出递归混合 $\mathbf{B}$"框架 $Y = (I - B)^{-1} A X$ 涵盖 attention/SSM/linear recurrence/高阶递归，证明 sparsity pattern of $A, B$ 直接控制 $\mathcal{O}(n^{\log n})$ 到 $\mathcal{O}(n^2)$ 的复杂度梯度，提出 $f(k) = 2^k$ 和 $f(k) = k^2+1$ 两种 translation-invariant 模式给出 $\mathcal{O}(n \log n)$ 和 $\mathcal{O}(n \sqrt{n})$ 的新选择，且 cache 可缩到 $\mathcal{O}(\log n)$ 或 $\mathcal{O}(\sqrt{n})$。

**[In-Context Routing (ICR): 一次训练、处处可用的 attention-level 隐式 ICL](llm_nlp/train_once_reuse_everywhere_generalizable_implicit_in-context_learning_by_routin.md)**

:   ICR 不在 residual stream 注入 shift vector，而是从多域 ICL 中用 PCA 抽出 Principal ICL Directions (PIDs) 作为 attention logits 的 low-rank 修正方向，配 query-conditioned router 自适应调制；一次训练后能在 12 个 in/out-of-domain 任务上零样本推理，无任务特定检索/再训练，在 OOD 上不像 vector-based 方法那样退化。

**[Universal Reasoner: 冻结 LLM 的可组合即插即用推理器](llm_nlp/universal_reasoner_a_single_composable_plug-and-play_reasoner_for_frozen_llms.md)**

:   提出通用推理器（UniR）——通过训练独立的轻量推理模块来捕获奖励导向的推理行为，在推理时通过逻辑叠加与冻结 LLM 组合，实现无需微调冻结模型、跨模型大小转移和多任务可组合的推理增强。

**[Why Are Linear RNNs More Parallelizable?](llm_nlp/why_are_linear_rnns_more_parallelizable.md)**

:   这篇论文用电路复杂度严格解释了为什么 Linear RNN 比传统非线性 RNN 更容易像 Transformer 一样并行：LRNN 可落在近似 log-depth 的算术电路类中，而非线性 RNN 能表达更难并行的 logspace / polynomial-time 完全问题，二者形成表达力与并行性的基本权衡。

---

## 🧬 计算生物 { #computational_biology }

**[Active Timepoint Selection for Learning Measure-Valued Trajectories](computational_biology/active_timepoint_selection_for_learning_measure-valued_trajectories.md)**

:   本文研究“什么时候采样一个分布快照最有价值”，用 LOT 把 Wasserstein 空间中的测度轨迹线性化，再用带时间扭曲的多输出 GP 给出 epistemic uncertainty，从而主动选择最能降低轨迹重建误差的时间点。

**[CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](computational_biology/card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)**

:   CARD 用"基数 $r$ 分解"把分子 3D 坐标双射映射为先粗后细的离散-连续混合 token 序列，让一个跨系统通用的自回归 Transformer 作为"零自由能 proposal"通过 BAR 直接估算任意分子系统的绝对自由能，在 70 个新系统的溶剂化任务上达到经典 MFES 的精度且推理快约 40 倍。

**[CoSiNE: 条件位点独立的抗体序列神经进化模型](computational_biology/conditionally_site-independent_neural_evolution_of_antibody_sequences.md)**

:   CoSiNE 用神经网络参数化的条件位点独立连续时间马尔可夫链（CTMC）来建模抗体亲和力成熟过程，在保持可处理性的同时捕获位点间上位效应，并通过 Guided Gillespie 采样实现抗原特异性的抗体优化，在零样本变体效应预测上超越了现有语言模型和进化模型。

**[Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](computational_biology/constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)**

:   本文针对"在满足领域硬约束（如合成可达性、能量上界）的前提下最大化奖励（如结合亲和、偶极矩）"这一关键场景，提出 CFO 算法：用增广拉格朗日把约束生成式优化拆成一串带 KL 正则的标准微调子问题，自适应地更新罚因子 $\rho_k$ 与对偶变量 $\lambda_k$，在合成低维场景与 FlowMol 分子设计任务上同时给出可证收敛与显著的奖励—约束 Pareto 改进。

**[Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design](computational_biology/cross-chirality_generalization_by_axial_vectors_for_hetero-chiral_protein-peptid.md)**

:   本文提出 AFI（Axial Feature Injection），把轴向量特征以线性混合方式注入 $E(3)$-等变标量化模型的极向量通道，使其退化为 $SE(3)$-等变并对手性敏感；以此改造 UniMoMo 得到 PepMirror，仅用同手性（L-L）训练数据即可零样本生成异手性（D-L）多肽 binder，并通过湿实验在 CD38 靶点上验证为首个实验确证的 AI de novo D-肽设计框架。

**[Demystifying Multimodal Biomolecular Co-design with Intrinsic Geodesic Coupling](computational_biology/demystifying_multimodal_biomolecular_co-design_with_intrinsic_geodesic_coupling.md)**

:   作者把"序列 + 三维结构"这种异质模态的共生成问题，重新建模为**时序最优传输 (Temporal Optimal Transport)** 问题，用双层优化 + 高斯过程代理 (GeoCoupling) 在训练过程中**自动学出非对角的时间耦合曲线**（即让结构和序列以各自适合的节奏被去噪），在 SBDD 和无条件蛋白质共设计两个任务上同时打败"同步耦合"和"随机耦合"两大类基线，并意外发现一条普适的"结构先行 (structure-leading)"几何先于语义的生成规律。

**[DNAChunker: Learnable Tokenization for DNA Language Models](computational_biology/dnachunker_learnable_tokenization_for_dna_language_models.md)**

:   DNAChunker 在掩码 DNA 语言模型中嵌入一个端到端可学习的"动态分块器"，通过双向 Mamba 编码 + 余弦相似度边界预测把 base-pair 序列压成变长 chunk，并配合 mask 保护与残差门控防止信息泄露，仅用人类参考基因组、172M 参数就在五个基因组 benchmark 上全面超越 2.5B 级别的多物种预训练基线。

**[EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](computational_biology/evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)**

:   EvoEGF-Mol 把 SBDD 的连续坐标与离散原子/键类型放到同一个指数族自然参数空间里，用动态收紧的目标分布替代奇异的 Dirac 端点，沿着 Fisher-Rao 几何下的指数测地线同步演化，在 CrossDock 上把 PoseBusters 通过率推到 93.4%，逼近参考分子水平。

**[Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](computational_biology/flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)**

:   本文提出 Flow Sampling，把流匹配/扩散模型从"数据驱动"反转为"噪声驱动"——以源噪声样本为条件构造去噪扩散漂移，在 interpolant 上用 detached 模型采得 $X_1$ 的能量梯度做回归目标，从而学到无数据情况下的高效扩散采样器，并自然推广到常曲率黎曼流形。

**[From Feasible to Practical: Pareto-Optimal Synthesis Planning](computational_biology/from_feasible_to_practical_pareto-optimal_synthesis_planning.md)**

:   PareSP 用**多目标 MCTS 搜索**联合优化合成路径的**成本 / 时间 / 可行性 / 环境影响**——找到完整帕累托前沿而非单一"最佳"路径，在 USPTO 和 ASKCOS 基准上较单目标方法节省 23% 成本、35% 时间，同时保持 ≥ 95% 化学可行性。

**[From Holo Pockets to Electron Density: GPT-style Drug Design with Density](computational_biology/from_holo_pockets_to_electron_density_gpt-style_drug_design_with_density.md)**

:   本文把结构药物设计的 condition 从"刚性 empty pocket"换成"包含配体与溶剂的 filler 低分辨率电子云"，并提出第一个 decoder-only autoregressive 的 EDMolGPT，在 DUD-E 101 个靶点上 bioactive recovery 达 41%、远超先前 ED-based 方法。

**[iLoRA: Bayesian Low-Rank Adaptation with Latent Interaction Graphs for Microbiome Diagnosis](computational_biology/ilora_bayesian_low-rank_adaptation_with_latent_interaction_graphs_for_microbiome.md)**

:   iLoRA 用贝叶斯方法从每个微生物组样本里推断一张稀疏的菌群交互图（Poisson 边 → Laplace 稀疏化 → GNN 嵌入），再用这个图去生成 input-conditioned 的 LoRA 矩阵 $A$，让 LLM 在做 IBD 诊断的同时把"是哪些菌在 cross-talk"这件事和预测一起学出来。

**[Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback](computational_biology/influence-guided_symbolic_regression_scientific_discovery_via_llm-driven_equatio.md)**

:   IGSR 把符号回归拆成"LLM 提议基函数 ψ_j + 逐项影响力分数 Δ_j 剪枝"两步循环，并把这个循环嵌入 MCTS 来搜组合空间，在 6 个生物医学基准和 LLM-SRBench 上同时拿下最佳 MSE 与符号召回，还在湿实验里发现了 DNA 甲基化与 RNA Pol II 停顿的新关系。

**[Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](computational_biology/learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)**

:   ProtDiS 把预训练蛋白质微环境嵌入（如 ESM-3）通过信息瓶颈+冗余消除的方式拆解成 8 个生物物理可解释的"知识通道"和 1 个残差通道，让结构表示在十二个下游任务（尤其是结构相似但功能不同的情形）上一致提升。

**[Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](computational_biology/learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)**

:   L3-PPI 把生物学里的 "L3 规则"（蛋白质对之间的 length-3 路径越多越可能相互作用）变成可学习的 graph prompt：用预训练 GNN 识别 L3 模式，再用门控网络生成虚拟 L3 路径并按 PPI 标签正则路径数量，做成一个即插即用的分类头，把任意 PPI 表征模型平均涨 2-4 个点。

**[Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](computational_biology/learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)**

:   C-FREE 把分子拆成固定半径的 k-EgoNet 子图，2D 拓扑 + 多个 3D 构象走 GINE + PaiNN + Transformer 编码后用 JEPA 风格的潜空间预测做预训练，全程无负样本、无增广、无位置编码，仅用 0.33M 分子（GEOM）就在 MoleculeNet 8 个任务上超越了用 19M–77M 分子训练的 UniMol / MolFM 等多模态基线。

**[LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation](computational_biology/lineageflow_flow_matching_for_high-fidelity_family-aware_protein_sequence_genera.md)**

:   把通用的均匀/掩码噪声先验换成由祖先序列重建（ASR）得到的家族特异 Dirichlet 先验，让 Dirichlet flow matching 从"已经进化好的脚手架"出发去做结构化突变，再在中间时刻插入一次 mutate–select–amplify 的 rerouting，从而在 Pfam 8886 个家族上把家族识别准确率推到接近自然序列（95.3% vs 96.6%）、同时保持高新颖度和折叠置信度。

**[Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](computational_biology/neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)**

:   本文从一个预训练 masked diffusion model (MDM) 的隐藏状态出发，训练一个轻量级"互信息预测头"，一次前向就能输出全部 token 对之间的条件互信息矩阵，并据此挑选"条件独立"的 token 子集做并行解码，在 Sudoku 和蛋白质 (ESM-C) 上把推理 NFE 降低 3-5 倍同时保持甚至超过顺序解码的质量。

**[On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering](computational_biology/on_the_collapse_of_generative_paths_a_criterion_and_correction_for_diffusion_ste.md)**

:   本文指出"用 ratio-of-densities 组合多个异质扩散/流模型"的推理时引导会出现 **Marginal Path Collapse（MPC）**——中间时刻的复合密度变得不可归一化，进而提出一个充要的 **Path Existence Criterion (PEC)** $C(t)>0$ 来诊断塌缩，并设计 **ACE** 通过对指数 $\gamma_i(t)$ 加 bump 函数来动态修正路径，把 Feynman–Kac 修正器推广到时变指数情形，在合成 Checkerboard、柔性 pose scaffold decoration（DN/CONF/SBDD 三专家组合）以及 COCO-MIG 多属性生成上都显著优于 NR/FKC 等常数指数基线。

**[Protein Autoregressive Modeling via Multiscale Structure Generation](computational_biology/protein_autoregressive_modeling_via_multiscale_structure_generation.md)**

:   PAR 把图像领域 VAR 的 "next-scale prediction" 思路搬到蛋白质 Cα 骨架生成上，用多尺度下采样 + 自回归 Transformer + flow-based 解码器替代单尺度扩散模型，配合 noisy context learning 和 scheduled sampling 缓解曝光偏差，在无条件生成 FPSD 上达到 161.0 的同时解锁 zero-shot 点提示生成与 motif scaffolding，并取得 2.5× 采样加速。

**[Protein Circuit Tracing via Cross-layer Transcoders](computational_biology/protein_circuit_tracing_via_cross-layer_transcoders.md)**

:   作者把 NLP 中的 cross-layer transcoder 搬到蛋白质语言模型 ESM2 上,提出 ProtoMech 框架以 < 1% 的稀疏潜变量电路恢复 79% 的下游性能,并能沿电路 steering 设计出高 fitness 的蛋白变体,在 70%+ 案例中击败基线。

**[Protein Fold Classification at Scale: Benchmarking and Pretraining](computational_biology/protein_fold_classification_at_scale_benchmarking_and_pretraining.md)**

:   作者基于 TED + Foldseek 聚类的 AlphaFold 结构构建了规模空前（约 49 万条、965 类）的非冗余蛋白质折叠分类基准 TEDBench，并提出 SE(3)-不变的掩码自编码器 MiAE：用高达 90% 的极端掩码率 + 重编码器/轻解码器的非对称架构，仅 100M 参数即在线性探测和微调上击败 ESM2-650M、SaProt-650M 等更大模型。

**[Protein Language Model Embeddings Improve Generalization of Implicit Transfer Operators](computational_biology/protein_language_model_embeddings_improve_generalization_of_implicit_transfer_op.md)**

:   本文把预训练蛋白质语言模型（pLM）的残基嵌入直接灌进可迁移隐式转移算子（TITO），训出 PLaTITO 在 mdCATH 上仅用 56 ms 轨迹和 1100 GPU 小时就让小到 19 M 参数的粗粒化 $C_\alpha$ 模型在快折叠蛋白等离群系统的平衡采样上全面超过 BioEmu。

**[Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](computational_biology/routing_by_reaching_composition_of_pre-trained_gflownets_for_multi-objective_gen.md)**

:   本文提出一种无需训练的 GFlowNets 组合框架，通过用每个预训练模型的"到达概率"作为权重去混合各自的前向策略，使得在推理阶段就能针对任意线性标量化或逻辑算子的多目标组合直接采样，并在线性情形下被证明可精确恢复目标分布。

**[Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](computational_biology/scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)**

:   scLDM 用统一的多头交叉注意力块 (MCAB) 把可交换的基因表达数据编成固定长度、置换不变的潜变量集合，再用 DiT + 流匹配 + 联合多属性 classifier-free guidance 替代 Gaussian 先验，在多个 scRNA-seq 数据集上的重构、(有/无条件) 生成、扰动响应预测全面超过 scVI / scDiffusion / CFGen。

**[SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](computational_biology/sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)**

:   SIGMA 用 token 级对比损失把同一分子不同 SMILES 排列的隐状态强制对齐到同一条轨迹，并配套提出 IsoBeam 在解码阶段剪掉同构冗余路径，让序列模型在化学空间中真正"按图而非按字符串"思考。

**[Site4Drug: Predicting Drug-Binding Target Sites with an AI Agent](computational_biology/site4drug_predicting_drug-binding_target_sites_with_an_ai_agent.md)**

:   Site4Drug 把"在蛋白质上选哪里下药"这一上游瓶颈重构为一个约束优先的证据整合问题——LLM Agent 从序列推导拓扑、PTM、Motif、半胱氨酸等可行性信号，输出带分数、风险标签和可追溯日志的候选位点排序，并自动推荐应当采用抗体/多肽还是小分子模态。

**[Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](computational_biology/stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)**

:   SDG 把"训练免（training-free）扩散引导"和"随机最优控制（SOC）后验采样"两条路线统一起来：用 SOC 推出引导项的变分上界后，发现现有 Tweedie 类方法都漏掉了一个 KL 正则项，于是借 Stein 变分梯度下降设计一个"先 Tweedie 倒推到数据流形 $\mathcal{M}_T$、再 Stein 修正、再前推回噪声流形 $\mathcal{M}_t$"的回环修正机制，在图像引导和小分子-蛋白对接两类任务上都显著超过 DPS/LGD/MPGD/UGD 等基线，特别擅长在低密度区域采到稀有高价值样本。

**[SwitchCraft: A Programmatic Framework for Designing State-Switching Proteins](computational_biology/switchcraft_a_programmatic_framework_for_designing_state-switching_proteins.md)**

:   SwitchCraft 把"设计一个能在多个功能态之间切换的蛋白质"形式化为一个对组合约束求解的优化问题，通过对结构预测模型 Boltz-1 反向传播多组状态相关的损失（基序、结合、构象变化、接触），直接梯度下降优化氨基酸 logits，实现首个通用的多态蛋白计算设计框架，并在体外硅基实验中演示了正/负变构、基序切换、诱导结合、配体修饰、配体辨识与 cpGFP 荧光生物传感器的从头设计。

**[TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering](computational_biology/tada-bench_a_million-variant_benchmark_for_future-round_discovery_toward_agentic.md)**

:   TadA-Bench 用 31 轮真实定向进化湿实验中的百万级 TadA 变体序列，把蛋白工程问题形式化为"用前若干轮排出后若干轮"的固定数据回放任务，并配套 Seq2Graph 图式标签统一管线，揭示了主流生物大模型在"未来轮发现"上严重失效。

**[TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](computational_biology/td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)**

:   TD3B 把激动剂/拮抗剂的设计当作「方向性转移算子」生成任务，用一个方向 Oracle + 亲和力门控 + 树搜索摊销微调的掩码离散扩散框架，让预训练肽段生成器学会写出能定向偏移蛋白质活/失活构象转移的多肽序列。

**[Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](computational_biology/temporal_score_rescaling_for_temperature_sampling_in_diffusion_and_flow_models.md)**

:   通过对预训练扩散/流模型的 score 输出乘以一个仅依赖时间步、变量 $k$ 与 $\sigma$ 的解析重标缩因子 $r_t$，即可在推理阶段把采样分布"局部"地变得更尖或更平，而无需任何微调，对 DDIM 等确定性采样器也完全兼容。

**[Towards A Generative Protein Evolution Machine with DPLM-Evo](computational_biology/towards_a_generative_protein_evolution_machine_with_dplm-evo.md)**

:   本文提出 DPLM-Evo，把蛋白质语言模型的离散扩散从"只支持掩码替换"扩展为"显式建模替换+插入+删除三种进化编辑"，通过把变长观测序列解耦到上采样长度的隐对齐空间 + 上下文化进化噪声核，实现变长进化生成、进化轨迹式的蛋白质后编辑/优化，并在 ProteinGym 单序列变体效应预测上取得 SOTA。

**[Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](computational_biology/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)**

:   本文指出单细胞基础模型 (scFM) 蕴含丰富但被"重建式预训练"遮蔽的基因调控知识，并提出 Virtual Value Perturbation 与 Gradient Trajectory 两种探针，从冻结的 scFM 中蒸馏出可跨基因/跨数据集泛化的成对基因特征，在 BEELINE 基准上把 AUPRC 从 ~0.5 推到 0.8–0.97，开启了"通用 GRN 推断 (UGRN)"这一新范式。

**[Transformed Latent Variable Multi-Output Gaussian Processes](computational_biology/transformed_latent_variable_multi-output_gaussian_processes.md)**

:   本文提出 T-LVMOGP：把多输出 GP 的核心建模问题——跨输出协方差 $k_{p,p'}(x, x')$ 的构造——转化成"在 Lipschitz 正则的 RCNN 嵌入空间里用单个标量基核做内积"，并完整嵌入 SVGP 框架，使 MOGP 第一次能可扩展且高表达力地处理 $P > 10000$ 输出（含 ZINB 似然的空间转录组数据），同时全面胜过 SV-LMC / OILMM / GS-LVMOGP 等基线。

**[What Makes a Representation Good for Single-Cell Perturbation Prediction?](computational_biology/what_makes_a_representation_good_for_single-cell_perturbation_prediction.md)**

:   这篇论文提出 PerturbedVAE，认为单细胞扰动预测的好表征必须显式分离占主导的扰动不变背景程序和稀疏的扰动响应信号，并用因果结构组织后者，从而更好泛化到未见双基因组合扰动。

---

## 📈 时间序列 { #time_series }

**[Adaptive Time Series Reasoning via Segment Selection](time_series/adaptive_time_series_reasoning_via_segment_selection.md)**

:   这篇论文提出 ARTIST，把时间序列问答变成“边推理边选择片段”的序贯决策问题，通过 controller-reasoner 架构和层级自博弈 RL，让模型只读取与问题相关的时间片段并提升推理准确率。

**[AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection](time_series/anomseer_reinforcing_multimodal_llms_to_reason_for_time-series_anomaly_detection.md)**

:   AnomSeer 将经典时间序列异常检测的统计证据写成专家推理轨迹，并用 TimerPO 强化多模态大模型，使其在折线图输入上同时完成异常类型判断、区间定位和细粒度解释。

**[Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting](time_series/beyond_extrapolation_knowledge_utilization_paradigm_with_bidirectional_inspirati.md)**

:   提出 KUP-BI 框架，从训练集中构建"后目标延续"知识库，通过比率式变换检索相似历史轨迹的延续模式，生成延续风格辅助流并与主干网络特征门控融合，在 6 个数据集、4 种骨干架构上一致提升长时预测性能。

**[CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](time_series/combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)**

:   CombinationTS 把时序预测模型解耦为 Input Transformation / Embedding / Encoder / Decoder / Output Transformation 五个正交模块，在共享的"评估条件空间"上做配对蒙特卡洛采样，用边际性能 $\mu$ 和稳定性 $\sigma$ 取代脆弱的单点 MSE，结论是：一旦数据视图（Embedding）设计得好，参数无关的 Identity Encoder 就能打平甚至超过复杂 Transformer，时序预测领域的"SOTA 增益"很大程度上来自看数据的方式而不是建模能力。

**[DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)**

:   针对"未来协变量已知"的时间序列预测 (TSF-X), DAG 设计了一个双通路网络: 一条沿时间维捕获"历史外生→未来外生"的注意力模式并注入到"历史内生→未来内生"的预测里, 另一条沿通道维捕获"历史外生→历史内生"的模式并注入到"未来外生→未来内生"的预测里, 在 12 个公开/新发布 TSF-X 数据集上 10/10 拿下 MSE 最佳, 显著超过 TimeXer、TFT、TiDE、CrossLinear、PatchTST 等。

**[DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](time_series/distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)**

:   DistMatch 提出基于 **KS 统计量**的递归分箱方法——通过将残差分组到近似可交换的叶子节点中**摒弃权重重新分配**，在分布漂移下提供有效的保形预测间隔；5 个数据集上均实现最小的区间宽度，同时保持有效覆盖率。

**[Divide and Contrast: Learning Robust Temporal Features Without Augmentation](time_series/divide_and_contrast_learning_robust_temporal_features_without_augmentation.md)**

:   Di-COT 通过**随机划分序列为重叠子块**并对其进行对比学习——在不使用数据增强的情况下高效学习鲁棒的时间序列表示，相比现有方法速度快 2.5 倍、精度更高；6 大规模数据集 + 124 UCR + 28 UEA 上全面验证。

**[Doubly Outlier-Robust Online Infinite Hidden Markov Model](time_series/doubly_outlier-robust_online_infinite_hidden_markov_model.md)**

:   本文提出 BR-iHMM：把"鲁棒观测更新（WoLF）"与"批量化状态推断（degenerate sticky HDP prior）"结合起来，给在线无限隐马模型同时在观测空间和状态空间提供有界的 Posterior Influence Function（PIF），在金融订单簿、电力负荷、合成回归三类含异常值的流式数据上把一步预测 RMSE 最多降低 67%。

**[Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)**

:   通过 **MMD 检测分布漂移**并动态扩展异构专家池，结合**时间记忆路由器**保证选择一致性，Dynamic-TMoE 在九个时间序列基准上达到新的 SOTA——相比所有基线平均降低 MSE 10.4%、MAE 7.8%。

**[Ellipsoidal Time Series Forecasting](time_series/ellipsoidal_time_series_forecasting.md)**

:   Fern 把长期时间序列预测重新表述为「从固定高斯源到数据相关椭球的最优传输」，借助 Brenier 定理把搜索空间限制在 SPD（对称正半定）类 Jacobian 上，用 Householder 反射的低秩谱分解把代价从 $O(n^3)$ 压到 $O(Rn)$，并在非平稳冲击场景下相对 DLinear / Koopa 等基线取得最多 790× 的稳定性提升。

**[FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](time_series/factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models.md)**

:   FactoryNet 是首个统一控制环结构的工业时序大规模数据集——5100 万数据点 / 2.3 万端到端任务执行（1.33 万真实 + 9800 仿真）跨 6 个机器实体，按 Setpoint-Effort-Feedback-Context (S-E-F-C) 控制论分类对齐所有信号；27 种标注异常类型 + 健康基线 + 反事实对，使零样本跨实体迁移和参数高效异常检测成为可能。

**[FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](time_series/fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)**

:   本文把 HiPPO 框架背后的概率测度推广到带可调奇异指数 $\alpha$ 的分数阶幂律测度，从而首次同时拿到「全历史保留 + 近时敏感 + 尺度不变」，并将这一理论落地为 LTI 对角化 SSM——FRACTAL 在 Long Range Arena 上以 87.11% 平均分追平 S5，并在 ListOps 上拿到 61.85%。

**[From Observations to States: Latent Time Series Forecasting](time_series/from_observations_to_states_latent_time_series_forecasting.md)**

:   作者发现现有 TSF 模型即使预测精度高，其潜空间也常常是"时间错乱"的（Latent Chaos）；他们提出 LatentTSF——先用 AutoEncoder 把观察压到一个高维潜状态空间，然后让任何主流 backbone 在这个空间内做未来预测（Pred + Align 双损失），最后再解码回观察空间——在 6 个标准 benchmark 上稳定降 MSE/MAE，并恢复了潜表征的时间局部性和频谱结构。

**[Generalizing Multi-scale Time-Series Modeling with a Single Operator](time_series/generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)**

:   Sigma 框架通过学习**离散高斯（LDG）核**实现**连续、距离感知的尺度参数**，统一了现有的离散多尺度算子——在长期和短期预测任务上达到 SOTA，同时大幅降低计算成本（训练快 5.3×、显存少 3.8×）。

**[HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](time_series/helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)**

:   给每个特征学一个"身份嵌入"作为持久语义锚点，配合时间-特征双螺旋注意力，在 5 个公开多变量时序数据集 21 个缺失场景上全部拿下第一，比次优的 ImputeFormer 在 ETT-h1 等数据集上多 25% 以上的 MAE 降幅。

**[HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series](time_series/hepa_a_self-supervised_horizon-conditioned_event_predictive_architecture_for_tim.md)**

:   HEPA 通过**地平线条件化的 JEPA 自监督预训练**学习时间序列中的可预测动态——冻结编码器只微调预测器，用单一架构和固定超参在 11 个领域 14 个基准上超越多个 SOTA 方法，仅用 2% 标签数据即可达到 92% 性能。

**[HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](time_series/hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models.md)**

:   将现代 SSM（如 Mamba）中隐式的内存机制**显式化**——通过扩展 HiPPO 框架提出"HiPPO Zoo"（5 个变体），每个变体用可解释的多项式表示来实现特定的现代 SSM 能力（非线性、自适应内存、关联记忆、多尺度、预测目标约束）；选择性复制和关联回忆任务上达到 100%。

**[IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](time_series/impact_influence_modeling_for_open-set_time_series_anomaly_detection.md)**

:   IMPACT 把"影响函数"同时拿来当探照灯和手术刀——先用一个多通道偏差损失训出初始模型并算出每个训练样本对验证风险的影响分数，再在风险下降的理论保证下，把高影响的污染未标样本一键翻成有标异常、把对风险贡献最小的"边界正常样本"沿梯度方向扰动成"未见过的伪异常"，最后用双头网络分别学已见与未见两类异常，在 8 个真实时序基准上稳定超越十多个无监督与开放集基线。

**[Interpretability in Deep Time Series Models Demands Semantic Alignment](time_series/interpretability_in_deep_time_series_models_demands_semantic_alignment.md)**

:   本文是一篇**位置论文**——提出深度时间序列模型应该强制**语义对齐**：让模型的内部变量和机制对应领域专家的推理方式而非仅解释内部计算；核心创新是针对时间演化定义了语义对齐的持久性约束（这是时间序列特有问题）。

**[It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](time_series/its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)**

:   TIME 是面向**时间序列基础模型（TSFM）**的下一代基准——通过**人工标注 + LLM 驱动的数据清洗**、**上下文对齐的任务设计**、**模式级别的评估视角**，克服现有基准的数据重用、质量问题、任务配置不当和评估粒度低等四大痛点；50 个全新数据集 × 98 任务 × 12 TSFM 评估。

**[Latent Laplace Diffusion for Irregular Multivariate Time Series](time_series/latent_laplace_diffusion_for_irregular_multivariate_time_series.md)**

:   LLapDiff 是在**隐空间中进行扩散**的生成框架——通过在拉普拉斯域用可学习的复共轭极点参数化**稳定的模态演化**，实现不规则时间序列的长期预测和缺失值补全，**无需逐步的物理时间积分**；7 个数据集上平均排名 2.1±1.7。

**[Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](time_series/learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)**

:   CTDG-SSM 首次通过**拓扑感知 HiPPO 投影**和状态空间模型，同时捕捉动态图中的多跳长距离空间依赖（LRS）和长距离时间依赖（LRT），在链接预测 / 节点分类等任务上超越 SOTA 且参数量仅为竞争方法的 1/10。

**[Nested Spatio-Temporal Time Series Forecasting](time_series/nested_spatio-temporal_time_series_forecasting.md)**

:   NeST 把"未来的宏观区域趋势"作为自顶向下引导，配合谱聚类构造的语义区域和双向跨尺度 cross-attention，让节点级时空预测在大规模交通网络上同时取得精度、长程稳定性与近线性复杂度的全面提升。

**[OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density](time_series/olivia_harmonizing_time_series_foundation_models_with_power_spectral_density.md)**

:   OLIVIA 通过引入功率谱密度（PSD）驱动的协调机制——Harmonizer（基于 Householder 反射的正交二阶协调）和 HarmonicAttention（共鸣器低维交互）——显著改进了时间序列基础模型在异质数据上的预训练，在 TSLib 零样本 + GIFT-Eval + GluonTS 多基准上实现 SOTA。

**[Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)**

:   PPM 用一个轻量编码器从历史序列里推断出 context-aware 高斯先验，再用一个两层 MLP 把这个先验"推前"成完整的预测分布，用 KDE-NLL + 均值 MSE 联合训练，在七个时序基准上同时打过 DeepAR 和 NsDiff 等扩散模型，推理还快 2× 到 100×。

**[PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)**

:   针对时间序列问答 (TSQA)，PATRA 在表征端把序列显式拆成 full / trend / season 三类模式，并通过三组可学习对齐 token 与文本做深度交叉对齐；在训练端用 SFT + GRPO 两阶段强化学习，把判别式与生成式任务的奖励统一映射到 $[0,2]$ 解决难度失衡，从而在四类 TSQA 任务上全面超越文本 LLM、ChatTS 等多模态时序 LLM。

**[Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series](time_series/position_current_benchmarking_hinders_real_progress_in_deep_learning_for_time_se.md)**

:   这篇位置论文系统揭示了当前时间序列预测基准测试的核心问题——**设计选择的差异**（全局 / 局部参数、预处理、外源变量、时间和空间处理）往往被当作"实现细节"忽视，导致不同论文的对比不公平；通过 44 数据集 × 7 SOTA × 多个参考架构的受控实验，证明这些差异的影响（5-15%）常常**超过具体序列建模层的贡献**（1-3%）。

**[QuITE: Query-based Irregular Time Series Embedding](time_series/quite_query-based_irregular_time_series_embedding.md)**

:   QuITE 是一个**即插即用的嵌入模块**——使用可学习的查询令牌通过自注意力直接聚合不规则观测，将任意 MTS 模型适配到不规则多变量时间序列（IMTS），无需改动架构或生成人工值；在 iTransformer + QuITE 上预测平均相对提升 54.7%。

**[The Cost of Learning Under Multiple Change Points](time_series/the_cost_of_learning_under_multiple_change_points.md)**

:   本文提出 Anytime Tracking CUSUM (ATC) 算法，通过时变自适应阈值 + 选择性检测原理，在**无任何可检测性假设**（最小间距 / 最小跳幅）下达到近似最小最优的动态遗憾 $O(\sigma^2 (S+1) \log T)$；并首次形式化量化了多变点场景中"漏检带来的内生混淆"的对数级退化界。

**[Time-series Forecasting Through the Lens of Dynamics](time_series/time-series_forecasting_through_the_lens_of_dynamics.md)**

:   作者用 Allen 时间区间代数提出 PRO-DYN 命名法，把任意时序预测模型拆成"前处理 PRO → 动力学 DYN → 后处理 PRO"三段，发现两条经验规律：(i) DYN 必须**可学习且完整**才能打过 LTSF-Linear，(ii) DYN 必须放在**整个流程末端**（PRE-DYN 配置）才能吃到长 lookback 的红利；并通过给 Informer/FEDformer/MICN/FiLM 加一个线性 DYN 层让性能稳定提升，给 iTransformer/PatchTST/Crossformer 把 DYN 挪到前端则性能下降，用实验验证两条规律。

**[TimeOmni-VL: Unified Models for Time Series Understanding and Generation](time_series/timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)**

:   TimeOmni-VL 通过把时间序列转换为**高保真图像**（Bi-TSI）+ 引入**理解引导的生成机制**（CoT 作为扩散条件），首次实现在统一多模态框架中同时达成时间序列**理解与生成**任务，预测和插补均达业界最优。

**[U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecasting](time_series/u-cast_a_surprisingly_simple_and_efficient_frontier_probabilistic_ai_weather_for.md)**

:   U-Cast 用**简单的 U-Net 主干** + **两阶段训练课程**（MAE 预训练 → CRPS 微调） + **MC-Dropout** 实现了与复杂专业模型（GenCast）相当的概率性天气预报能力，同时减少 10× 训练计算和推理延迟——颠覆"前沿性能必须复杂"的行业刻板印象。

---

## 🕸️ 图学习 { #graph_learning }

**[Aitchison Embeddings for Learning Compositional Graph Representations](graph_learning/aitchison_embeddings_for_learning_compositional_graph_representations.md)**

:   本文提出 AICoG，将节点表示为 simplex 上的潜在原型混合，并用 Aitchison 几何与 ILR 等距坐标学习图嵌入，在保持与欧氏 latent distance model 同等表达力的同时，让节点角色相似性具有基于相对比例 trade-off 的内生解释。

**[An Approximation Algorithm for Graph Label Selection](graph_learning/an_approximation_algorithm_for_graph_label_selection.md)**

:   这篇论文首次在不放宽标注预算的设定下，为 Graph Label Selection 给出 $\tilde{O}(\log^{1.5} n)$ 近似算法，并通过树割稀疏化、流判定和树上动态规划把原本全局耦合的选点问题变成可求解的组合优化流程。

**[Anchor-guided Hypergraph Condensation with Dual-level Discrimination](graph_learning/anchor-guided_hypergraph_condensation_with_dual-level_discrimination.md)**

:   AHGCDD 把超图凝聚 (HGC) 从"先训练结构生成器、再匹配训练轨迹"的解耦范式重写为端到端框架：用 Heat-Kernel-PageRank 把结构信息塞进初始化特征、用 anchor-guided 思路按特征距离合成稀疏可学的超边，再用粗+细双级判别损失 (类原型 MMD + 实例级对比) 代替昂贵的 HNN 重训练，在 5 个超图基准上 ≥SOTA 同时最高 144× 加速。

**[Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](graph_learning/are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)**

:   这篇论文把图预训练中的“可迁移公共子结构”重新定义为表示空间中的行为不变性，并用神经向量丛、门控几何展平和 Dirichlet 损失构建 Gauge，使图模型在跨域少样本迁移、零样本链路预测和图同构任务上获得更强的结构泛化能力。

**[Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](graph_learning/beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)**

:   提出 M-DESIGN 框架，将神经网络设计建模为检索增强的迭代修改过程，通过构建修改-增益图编码细粒度架构编辑效果，并利用贝叶斯动态任务相似度在线校准迁移信号，在 33 个 GNN 任务中的 26 个达到设计空间最优。

**[Deep Neural Sheaf Diffusion](graph_learning/deep_neural_sheaf_diffusion.md)**

:   本文指出 Neural Sheaf Diffusion (NSD) 在深层会因 sheaf Laplacian 的"分歧信号"随扩散收敛而消失，从而失去理论上保证的抗坍缩能力；DNSD 用 **sheaf 邻接算子**替代 Laplacian，并配合 LayerNorm、奇函数激活与逐 stalk 门控，使 sheaf 架构第一次能稳定堆叠到 16 层，在合成长程任务上比 GNN/NSD 基线最多提升 30 pp，在真实异质图基准上也一致领先。

**[DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA](graph_learning/dtkg_dual-track_knowledge_graph-verified_reasoning_framework_for_multi-hop_qa.md)**

:   DTKG 把多跳问答按"并行事实核验 vs 链式推理"二分，先用 few-shot 分类器把问题路由到合适的分支，并行分支用 KG 三元组核验原子事实，链式分支在 Wikidata 上做 DFS 路径扩展+评分剪枝，外加一套"任务感知"去噪，在 6 个数据集上比 KGR / ToG 等单策略 baseline 提升 5%–29.5%。

**[Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models](graph_learning/finding_the_minimal_parameter_budget_for_implicit_reasoning_a_data_complexity_dr.md)**

:   本文从知识图谱补全任务出发，证明并测量了"隐式推理所需的最小参数量"满足一条以**图搜索熵**为复杂度度量的线性 scaling law，每个参数最多支持约 $0.008$ bit 推理信息，颠覆了"模型越大推理越强"的朴素直觉。

**[Fixed Aggregation Features Can Rival GNNs](graph_learning/fixed_aggregation_features_can_rival_gnns.md)**

:   本文提出 Fixed Aggregation Features (FAF)：把多跳邻域用 mean/sum/max/min/std 等**不可训练**的聚合算子压成表格特征再喂给 MLP，在 14 个节点分类基准中有 12 个能与精调过的 GCN/GAT/GraphSAGE 乃至 Graph Transformer 打平甚至超越，从而对"GNN 的可训练邻域聚合到底有多必要"提出系统性质疑。

**[Full-Spectrum Graph Neural Network: Expressive and Scalable](graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)**

:   本文把经典谱 GNN 的单变量特征值滤波器 $g(\lambda_i)$ 推广为双变量滤波器 $g(\lambda_i,\lambda_j)$，把信号从节点域抬到节点对域，理论上能逼近 Local 2-GNN（超越 1-WL），并通过低秩张量分解避开 $n^2\times n^2$ 显式计算，在异质图节点分类和子结构计数上拿到强结果。

**[ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation](graph_learning/generalist_graph_anomaly_detection_via_prototype-based_distillation.md)**

:   ProMoS 把一个冻结的自监督 GNN 当成"正态先验老师"，用一束共享 + 稀疏激活的轻量学生分支去蒸馏它，并通过可学习原型把师生对齐到一个跨图共享的语义空间，从而第一次实现了完全无标签、零样本、跨图迁移的通用图异常检测器。

**[Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)**

:   本文提出"事实生成"任务，把超关系知识图谱（HKG）补全从"填一个空"扩展到"从任意掩码模式甚至从零生成完整事实"，并给出首个生成式 HKG 表示学习方法 KREPE：用上下文消息传递编码事实内/事实间依赖，用掩码离散扩散建模缺失分量的联合条件分布，在三个 HKG 基准的链接预测上达 SOTA，并在事实生成任务上把基于 GPT-5.2 / Gemini 3 Pro 的强 LLM 基线大幅甩开（如 WikiPeople- 从零生成 0.855 vs LLM 最好 0.343）。

**[GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](graph_learning/gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)**

:   GILT 把节点/边/图三类少样本图分类统一改写成基于 token 的 in-context learning 问题，用"线性 GCN 提结构 + 非对称原型 token + 两段式注意力 Transformer + 原型头"的纯数值架构，做到既不依赖 LLM 也不需要任何下游 tuning，在 5-shot 设置下超过 LLM-based 和 tuning-based GFM，同时比它们快 1~4 个数量级。

**[Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction](graph_learning/identifying_and_correcting_label_noise_for_robust_gnns_via_influence_contradicti.md)**

:   ICGNN 在图扩散矩阵上定义"影响矛盾分数"(ICS) 从结构和属性两个层面度量节点标签的可疑程度，再用 GMM 软阈值挑出脏标签，并以邻居预测做凸组合式软纠正，在 6 个图基准上跑赢 NRGNN / RTGNN / CGNN / ProCon 等专门方法。

**[KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)**

:   把 KBQA 从"一次性生成逻辑表达式"重新定义为"多轮决策过程"，先用 Referenced Rejection Sampling 在金标动作序列的引导下生成可执行的推理轨迹做 SFT 冷启动，再用 GRPO 基于 F1 结果奖励优化策略，让 8B Llama 在 WebQSP / GrailQA / GraphQ 三个 benchmark 上同时超过 GPT-4 提示方法与图检索 SOTA。

**[L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](graph_learning/l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)**

:   作者把图傅里叶变换（GFT）的特征基**精确分解**成"每个子图的局部 GFT × 一串 Cauchy 矩阵"，将 $O(n^3)$ 的特征分解降到 $O(kn^2)$（$k$ 为子图间的切边数），并在分解里穿插可学谱滤波器，得到一族能在 569k 节点大图上跑通、参数比 Transformer 少几个数量级却性能相当的局部-到-全局谱 GNN。

**[Learnable Kernel Density Estimation for Graphs and Its Application to Graph-Level Anomaly Detection](graph_learning/learnable_kernel_density_estimation_for_graphs_and_its_application_to_graph-leve.md)**

:   LGKDE 用一个可学习的深度 MMD 度量把每个图嵌成"节点分布"，再在该度量空间上叠加多尺度核密度估计，并通过"正常图密度高于其结构感知扰动版本"这一自监督对比信号端到端训练，从而首次为图级密度估计提供了既有一致性、收敛速率、鲁棒性、泛化界等理论保证、又在十余个图异常检测基准上稳定超越 GNN/对比/单类等强基线的统一框架。

**[MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation](graph_learning/medcog_maximizing_llm_inference_density_in_medical_reasoning_via_meta-cognitive_.md)**

:   MedCoG 让 LLM 先对医学问题做"复杂度 / 熟悉度 / 知识密度"三维自评，再按需调用 SCoT、记忆和知识图谱三类知识，把推理密度（达到同等精度所需的理论开销/实际开销）拉到 6.2×，同时在 5 个 MedQA 系列 hard set 上把平均准确率从 AFlow 的 34.5 提到 37.5。

**[Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](graph_learning/message_tuning_outshines_graph_prompt_tuning_a_prismatic_space_perspective.md)**

:   本文提出 **Prismatic Space Theory (PS-Theory)**，把冻结 GNN 基础模型视为对输入流形做"棱镜式"折射的逐层分段线性映射，由此严格证明图提示微调 (graph prompt tuning) 的适配能力存在上界；进一步提出 **Message Tuning (MTG)**，在每层注入可学习的"消息原型"并与原生消息做动态融合，理论上可突破该上界，实验在 15 个数据集 / 6 种预训练策略上全面优于现有图提示方法。

**[Physics-Informed Coarsening for Multigrid Graph Neural Surrogates](graph_learning/physics-informed_coarsening_for_multigrid_graph_neural_surrogates.md)**

:   本文为固体力学有限元仿真训练了一个 Encoder-Processor-Decoder 多重网格 GNN 代理模型，核心创新是把"粗化（downsampling）时选哪些节点"从几何启发式（FPS）/学习注意力换成"按动量守恒方程的离散残差打分再 TopK"，从而把粗层算力倾斜到应力集中、接触界面、大变形等动力学关键区域，在 DeformingPlate 上把 rollout RMSE 从 SOTA 的 $11.46\times 10^{-3}$ 降到 $6.5\times 10^{-3}$（提升约 43%）。

**[Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](graph_learning/polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)**

:   PolyNSD 把 Sheaf 神经网络的"一步空间扩散"换成对归一化 sheaf 拉普拉斯的可学习 $K$ 阶多项式谱滤波器，用 Chebyshev 三项递推稳定计算，单层就拥有 $K$-hop 感受野和可控的低/带/高通响应；意外的发现是只用对角 restriction maps 就能超越所有需要稠密大维 stalk 的现有 NSD，参数、显存、运行时间都大幅下降。

**[Quantile-Free Uncertainty Quantification in Graph Neural Networks](graph_learning/quantile-free_uncertainty_quantification_in_graph_neural_networks.md)**

:   QpiGNN 提出"无需分位输入、无需后处理"的 GNN 节点级预测区间框架，用双头 GNN（一头预测均值、一头预测半宽）配合直接优化"覆盖率+区间宽度"的标签级联合损失，在 19 个合成/真实数据集上平均覆盖率提高 22%、区间宽度收窄 50%。

**[RADE: Random Add-Drop Edge as a Regularizer](graph_learning/rade_random_add-drop_edge_as_a_regularizer.md)**

:   RADE 在 GNN 训练中同时随机删边与加边，并通过"期望保持"的聚合校正使训练-推理对齐，再用 GradNorm 自适应调节删/加比例，让一种增广同时缓解过拟合与过挤压。

**[Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](graph_learning/rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)**

:   针对通用图异常检测中"PCA 对齐只统一维度、不统一语义"导致的负迁移问题，本文用一组 5 维"关系指纹"（邻域位置/方向/全局方向一致性 + 度 + 聚类系数）显式提取异常指示性线索作为跨域通用特征，再叠加一个 Transformer 域共享编码器和 SNR 引导的域自适应重校准模块，在 14 个数据集上几乎做到了"全员正迁移"的 SOTA。

**[Structure-Centric Graph Foundation Model via Geometric Bases](graph_learning/structure-centric_graph_foundation_model_via_geometric_bases.md)**

:   SCGFM 把跨域图基础模型重写为度量测度空间上的"三角测量"问题：学一组 $K$ 个可训练几何基 $\{B_k\}$，每个图用其与各基的 Gromov–Wasserstein 距离 $\delta_k$ softmax 得到一组结构坐标 $\mathbf{w}$，再用基上的 OT plan 把节点特征汇聚到统一维度，从而摆脱"必须对齐节点特征空间"的传统 GFM 桎梏，在 in-domain 与 OOD 少样本图/节点分类上都打过 baseline。

**[T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)**

:   T-GINEE 结合 **CP 张量分解**与**广义估计方程（GEE）**显式建模多层网络中的**跨层依赖关系**——具有理论保证和优异的可扩展性，在百万级节点图（DBLP、Stack Overflow）上突破其他张量方法 OOM 的限制。

**[Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](graph_learning/unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)**

:   本文把 CNF 公式建模成「子句–文字超图 + 子句关联图」，并在变量级把表示拆成极性不变与极性等变两部分，再用极性翻转一致性正则训练，把 unsat-core 变量预测精度显著拉高一档。

**[View Space：跨任意图的表示学习](graph_learning/view_space_learning_representation_across_arbitrary_graphs.md)**

:   本文提出视图空间概念，通过将图从 2 维（节点-特征）升到 3 维（节点-特征-视图），实现对任意特征维度和语义图的统一表示——首次让图模型像 NLP/CV 基础模型那样无需微调即可跨域推理，在 27 个下游任务上平均超越 GraphAny 8.93%。

**[What Structural Inductive Bias Helps Transformers Reason Over Knowledge Graphs? A Study with Tabula RASA](graph_learning/what_structural_inductive_bias_helps_transformers_reason_over_knowledge_graphs_a.md)**

:   这篇论文用一个可拆卸的最小图 Transformer 变体 RASA 做控制实验，发现知识图谱多跳问答中最有用的结构归纳偏置主要是邻接 mask 带来的拓扑约束，而不是关系类型 bias、query scaling 或 value gating 这类可学习关系参数。

**[When Do Graph Foundation Models Transfer? A Data-Centric Theory](graph_learning/when_do_graph_foundation_models_transfer_a_data-centric_theory.md)**

:   这篇论文用 graphon 把不同大小、不同域的图放到同一连续空间里，证明图基础模型的跨域输出差异可以分解为两个有限采样误差和一个内在 graphon 域差异，并用合成与真实图实验说明图大小、结构偏移和谱位置编码稳定性共同决定迁移成败。

**[Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions](graph_learning/whom_to_query_for_what_adaptive_group_elicitation_via_multi-turn_llm_interaction.md)**

:   这篇论文把多轮问卷式 elicitation 从“问什么问题”扩展到“问谁、问什么”的联合决策，用 LLM 估计问题的信息增益、用异构 GNN 在群体关系图上传播和插补缺失回答，从而在有限受访者预算下更快恢复群体偏好。

---

## 🎵 音频/语音 { #audio_speech }

**[A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation](audio_speech/a_semantically_consistent_dataset_for_data-efficient_query-based_universal_sound.md)**

:   这篇论文提出 Hive，一个通过单事件净化和语义一致混合构造的通用声音分离数据集，用约 2.4k 小时高纯度源音频让 AudioSep、FlowSep 在多项分离指标上接近甚至超过百万小时级训练的系统。

**[Alethia: A Foundational Encoder for Voice Deepfakes](audio_speech/alethia_a_foundational_encoder_for_voice_deepfakes.md)**

:   Alethia 提出一种"瓶颈式掩码嵌入预测 + Flow-Matching 频谱生成"的双分支预训练范式，训出首个面向语音 deepfake 检测/定位/溯源的基础编码器，在 5 类任务 56 个数据集上显著超过 Wav2vec2/HuBERT/WavLM 等通用 SFM，并对未见过的歌声 deepfake 和真实扰动表现出强零样本鲁棒性。

**[Algorithmic Recourse of In-Context Learning for Tabular Data](audio_speech/algorithmic_recourse_of_in-context_learning_for_tabular_data.md)**

:   这篇论文首次系统研究表格数据 in-context learning 场景下的算法追索问题，证明 ICL 诱导的动态决策规则仍可定义可界定的 recourse，并提出 ASR-ICL 用自适应子空间零阶优化在黑盒 ICL 模型上生成低成本、稀疏且可行动的反事实修改。

**[An Exterior Method for Nonnegative Matrix Factorization](audio_speech/an_exterior_method_for_nonnegative_matrix_factorization.md)**

:   这篇论文提出 eNMF，把 NMF 从“始终待在非负正交锥内部优化”改成“先从无约束 SVD 最优解的旋转等价类外部逼近非负锥，再可行化并下降”，在合成、文本、音频、图像和推荐数据上比 9 类 NMF baseline 更快达到更低重构误差。

**[Attend to Anything: Foundation Model for Unified Human Attention Modeling](audio_speech/attend_to_anything_foundation_model_for_unified_human_attention_modeling.md)**

:   AAM把图像、视频和音视频显著性预测统一为一个带文本条件、双曲层级约束和Fokker-Planck时间动力学的注意力基础模型，在16个基准上整体优于专用模型，并把视频推理速度提升到约111 FPS。

**[Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](audio_speech/do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)**

:   作者构造了一个让"文字说的"和"声音听的"故意打架的 2000 题 MCQ 基准 VoxParadox，证明当前 Audio LLM 在副语言任务上几乎只"读不听"；再用一个按 prompt 自适应混合音频编码器中间层特征的轻量模块 PCLM 加上 DPO 偏好优化，把 Audio Flamingo 3 在 VoxParadox 上从 17.40% 拉到 65.20%。

**[Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability](audio_speech/evaluating_and_rewarding_lalms_for_expressive_role-play_tts_via_mean_continuatio.md)**

:   本文把"预训练大音频语言模型对真值语音 token 的续写概率"包装成一个名为 MCLP 的客观风格一致性度量，再用 MCLP+CER 的门控混合奖励，通过 GRPO 在新构建的 WenetSpeech-RP-TTS 数据集上把角色扮演 TTS 的主观 MOS 从 1.86 推到 3.58。

**[Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](audio_speech/group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)**

:   针对集中式多模态融合带来的"模态主导"和"虚假模态耦合"两个痼疾，GCL 把多模态学习重写为**两阶段四 agent 的协议化协作**：第一阶段由 Routing/Auditing agent 用边际预测增益逐样本决定哪些跨模态交流被允许，第二阶段由 Public-Factor/Aggregation agent 把共享语义与私有特化解耦后再聚合，在 MOSI/MOSEI/MIntRec 上拿到 SOTA。

**[JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments](audio_speech/jaeger_joint_3d_audio-visual_grounding_and_reasoning_in_simulated_physical_envir.md)**

:   JAEGER 在 Qwen2.5-Omni 基础上用 LoRA 适配出一个端到端的 3D 音视频大模型，通过 RGB-D 深度位置编码 + 一阶 Ambisonics (FOA) 双路音频 + 新提出的 Neural Intensity Vector，将传统 AV-LLM 从「2D RGB + 单声道」扩展到「3D 几何 + 多通道空间音频」，并配套发布了 61k 样本的 SpatialSceneQA 仿真基准。

**[MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)**

:   MECAT 用「多专家模型 + CoT 大模型推理」构造了 20k 条多视角细粒度音频字幕与 10 万条开放式 QA，并提出 DATE 指标（语义相似度 × 跨样本可区分度的调和平均），首次能稳定区分泛泛而谈与细节准确的音频模型输出。

**[MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](audio_speech/moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)**

:   MoshiRAG 在 Moshi 这一全双工语音模型里加入一个特殊的 ⟨ret⟩ 触发 token，让模型边说边异步调用 LLM/搜索后端去取参考文档，利用"开口到关键词出现"的自然 keyword delay 把 2 秒以内的检索延迟完全藏起来，从而在 LlamaQ/WebQ/TriviaQA/HaluEval 上把语音模型的事实性拉到 GPT-4o Audio 量级，同时保留全双工实时性。

**[MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](audio_speech/multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)**

:   MultiBreak 用"主动学习 + 不确定性引导改写"的迭代框架把多轮越狱数据集扩到 10,389 条对话、2,665 个独立有害意图，多样性 0.942 全面碾压前作，并在 DeepSeek-R1-7B / GPT-4.1-mini 上把 ASR 相比次优数据集分别提升 54% / 34.6%。

**[Multimodal Fact-Level Attribution for Verifiable Reasoning](audio_speech/multimodal_fact-level_attribution_for_verifiable_reasoning.md)**

:   MURGAT 是首个评测 MLLM 在多模态推理输出中"按事实粒度精确引用模态+时间段"能力的基准，搭配一个三步评估协议（可验证句识别 → 原子事实分解 → 归因质量）和高度与人工对齐的自动评测器 MURGAT-SCORE（Pearson 0.84），揭示了强模型即使答案对也常常胡乱引用，且强推理常以牺牲可验证引用为代价。

**[Multimodal Fusion via Self-Consistent Task-Gradient Fields](audio_speech/multimodal_fusion_via_self-consistent_task-gradient_fields.md)**

:   SCFAE 把多模态融合块改写成一个"任务损失 + 重建损失"组成的自洽场（Self-Consistent Field），通过把每个模态特征拆成"共享/特有"子空间并在模态间循环替换共享分量，让任务梯度干净地反传给各个编码器，从而在不等长输入、模态冲突、模态缺失三种场景下都比强耦合或重正则化的融合方法更稳健。

**[Multiple Choice Learning of Low-Rank Adapters for Language Modeling](audio_speech/multiple_choice_learning_of_low-rank_adapters_for_language_modeling.md)**

:   本文提出 LoRA-MCL，把 Multiple Choice Learning 的"赢者通吃"训练范式搬进 LoRA 微调：把 $K$ 组低秩 adapter 当作 $K$ 个相互竞争的假设，让每条训练样本只更新最合适的那组 adapter，从而让单一基座模型在一次前向里就能产生多条覆盖条件分布不同模态的多样合理文本，在音频/图像描述与机器翻译上同时刷新质量–多样性帕累托前沿。

**[MusicDET: Zero-Shot AI-Generated Music Detection](audio_speech/musicdet_zero-shot_ai-generated_music_detection.md)**

:   MusicDET 把"AI 生成音乐检测"重新定义为只用真实音乐训练的零样本问题，用频带分解 + 频带内归一化流 + 全局归一化流学习真实音乐能量谱的概率分布，把似然值当作"真伪分"，在 FakeMusicCaps / SONICS 的跨生成器评测下把平均 EER 从 ~17% 干到 4.51%（零样本）/ 0.89%（带类别条件先验）。

**[NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating](audio_speech/naaca_training-free_neuroauditory_attentive_cognitive_architecture_with_oscillat.md)**

:   用一套受皮层振荡启发的二维波动场（OWM）做实时显著性检测，给 Audio Language Model 在长音频上当一个"训练无关的注意力门"，只把真正显著的窗口送进 ALM，从而在 XD-Violence 上把 AP 从 53.5% 拉到 70.6%，同时减少约 40% 的 ALM 调用。

**[PhaLar: Phasors for Learned Musical Audio Representations](audio_speech/phalar_phasors_for_learned_musical_audio_representations.md)**

:   PhaLar 通过把音频特征投影到复平面并利用相位等变性——核心是用 FFT 把时间对齐编码为相位旋转——在音乐茎检索任务上相对 SOTA 提升 70%、参数仅为对手 44%、训练 7× 加速；从"相位不变"到"相位等变"是建筑哲学的根本转变。

**[Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](audio_speech/polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)**

:   Polyphonia 把 zero-shot 音色转换从单轨扩展到密集多轨混音：用盲源分离得到的 Ideal Ratio Mask（IRM）当外部声学先验，先在 pre-softmax 注意力 logit 里做"源插值 + 声学调制"，让目标声部（如人声）的频谱被新音色（如小提琴）替换的同时把背景伴奏严格保住，相比 SOTA 在 target alignment 上提升 15.5%。

**[Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning](audio_speech/position_text_embeddings_should_capture_implicit_semantics_not_just_surface_mean.md)**

:   本文是一篇 position paper：作者论证当前文本嵌入研究过度聚焦"表层语义"（词形 / 句法 / 主题相似），系统性忽略了语用、立场、社会语境等"隐式语义"，并通过 7 个隐式语义数据集的实证显示——即便是 SOTA 嵌入相比 Bag-of-Tokens 也只有边际提升，呼吁把隐式语义作为嵌入研究的一等建模目标。

**[Position: Towards Responsible Evaluation for Text-to-Speech](audio_speech/position_towards_responsible_evaluation_for_text-to-speech.md)**

:   这是一篇立场论文，提出 TTS 评测应从"只看技术指标"升级为三层递进的 **Responsible Evaluation**——保真度与准确性、可比性与标准化、治理-公平-安全——并系统性诊断了当前 WER/SIM/MOS/RTF 等指标的失效模式，给出 13 条可执行建议。

**[Probing Cross-modal Information Hubs in Audio-Visual LLMs](audio_speech/probing_cross-modal_information_hubs_in_audio-visual_llms.md)**

:   作者用因果追踪 + 单模态主导框架揭示了音视频 LLM 中存在一类被称为"跨模态 sink token"的隐藏枢纽,绝大多数跨模态信息都凝聚在这些 token 上,据此提出训练免费的注意力放大策略显著缓解物体幻觉。

**[SafeSearch: Automated Red-Teaming of LLM-Based Search Agents](audio_speech/safesearch_automated_red-teaming_of_llm-based_search_agents.md)**

:   本文提出 SafeSearch——一个全自动、沙箱化、可扩展的红队框架，通过在真实搜索结果中注入单个 LLM 生成的不可靠网页来评测搜索 Agent 的安全性，并用 300 个测试用例对 17 个 LLM × 3 种 Agent 脚手架进行系统评测，发现最高 ASR 高达 90.5%、且常用的 reminder 防御几乎无效。

**[Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](audio_speech/sparse_autoencoders_for_interpretable_emotion_control_in_text-to-speech.md)**

:   作者在 LLM-based TTS（IndexTTS2）的语义骨干残差流上训练 Top-k 稀疏自编码器（SAE），用"句级激活率差"挑出少量与目标情感强相关的稀疏潜在特征，推理时只对这几个特征做加/减干预，就能在不动主干参数的前提下实现可解释的双向情感诱导与抑制，效果优于全局均值差引导和现有 TTS 基线。

**[Sparse Tokens Suffice: Jailbreaking Audio Language Models via Token-Aware Gradient Optimization](audio_speech/sparse_tokens_suffice_jailbreaking_audio_language_models_via_token-aware_gradien.md)**

:   本文发现音频语言模型 (ALM) 越狱优化中的波形梯度高度集中在少数 token 上，提出 TAGO 在每步只更新 top-$\zeta$ 高能量 token 对应的波形区段，在 Qwen3-Omni 上仅保留 25% token 就能维持 86% 的 LLM-judge 越狱成功率 (vs 全量 token 的 87%)。

**[The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](audio_speech/the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)**

:   本文提出 FLAIR：让全双工口语对话模型（SDLM）在"听用户说话"的同时，把通常用来填 `<SIL>` 的步骤改成连续的隐式推理——通过一个 ELBO 训练目标 + 非因果"全局专家"提供后验，让因果 LLM 学会用一串嵌入向量"边听边想"，从而显著提升问答质量却不引入任何推理延迟。

**[Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)**

:   SwanSphere 提出"因果 AR 语言模型 + 局部 DiT（LocDiT）"的两阶段流式架构，从全景视频或文本生成一阶 Ambisonics（FOA）四通道空间音频，配合 SVAC 物理感知对比学习与三目标 ODPO，把首块延迟压到 0.21s 的同时在 FD/KL/角度误差上全面超越级联与端到端基线。

**[Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](audio_speech/towards_understanding_modality_interaction_in_multimodal_language_models_via_par.md)**

:   本文把多模态大模型的决策看成一次输入到输出的信息分解，借 Partial Information Decomposition (PID) 把 VL/全模态模型的预测互信息拆成"视觉独有 / 文本独有 / 冗余 / 协同"四项，发现协同项是预测视觉敏感性的最佳指标、全模态模型存在"视觉霸权"型协同瓶颈，并用 PID 得到的样本级分数指导 LoRA 重加权微调，在 MMStar/MMBench/POPE 上稳定提升 1–2 个百分点。

**[Two-Dimensional Quantization for Geometry-Aware Audio Coding](audio_speech/two-dimensional_quantization_for_geometry-aware_audio_coding.md)**

:   作者把神经音频 codec 中的标量量化器换成"成对通道 + 结构化二维网格"的几何量化器 Q2D2，用固定的六边形 / 矩形 / 菱形格点替代可学习码本，在单一 quantizer + 极低 token rate 下追平甚至超越 RVQ / VQ / FSQ 的语音重建质量。

**[VocSim：单源音频零样本内容身份识别的无训练基准](audio_speech/vocsim_a_training-free_benchmark_for_zero-shot_content_identity_in_single-source.md)**

:   VocSim 是涵盖 125k 单源音频的无训练基准，通过冻结特征加标签无关的 PCA 白化诊断音频基础模型的内在几何结构——揭示当前模型在低资源跨语言语音上的严重泛化缺陷。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](llm_efficiency/beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)**

:   提出对已收敛 MoE 模型的"正交增长"策略——深度方向用 interpositional 层复制、宽度方向用噪声专家复制——将 17B 模型扩展到 70B，在相同额外算力下比从头训练准确率提升 10.6%。

**[CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](llm_efficiency/criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)**

:   作者把"哪些 KV 缓存条目算关键"这个一直靠经验拍脑袋的问题，重新写成"最小化注意力输出扰动"的优化问题，推导出扰动的可解析上界（同时涉及注意力权重和经 $W^O$ 投影后的 value 范数），并由此设计了一个即插即用的两阶段贪心选择算法，把 SnapKV/AdaKV/HeadKV 三种 SOTA 驱逐方法在 29 个长上下文数据集上的压缩损失平均砍掉一半以上。

**[Do Transformers Need Three Projections？三选一/二的 QKV 共享系统研究](llm_efficiency/do_transformers_need_three_projections_systematic_study_of_qkv_variants.md)**

:   论文系统比较三种 QKV 投影共享方案——Q=K-V（共享 query 和 key）、Q-K=V（共享 key 和 value）、Q=K=V（三者共享），发现 Q-K=V 在 LM 上 PPL 仅升 3.1% 但 KV cache 减 50%，与 GQA/MQA 正交可叠加得 87.5%-96.9% cache 减少；为 edge inference 提供 quantifiable memory benefit。

**[DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE](llm_efficiency/dot-moe_differentiable_optimal_transport_for_moefication.md)**

:   DOT-MoE 把"dense FFN 转成 MoE 时怎么分配神经元到专家"建模成 differentiable optimal transport——Sinkhorn-Knopp 迭代解 entropic-regularized balanced transport + Straight-Through Estimator 让 neuron-to-expert assignment 和 router 联合 end-to-end 学习；在 LLaMA-2/3 + Qwen2.5 上 50% 激活参数下保留 90% dense 性能，超过 structured pruning / random / 聚类等所有 baseline。

**[Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](llm_efficiency/efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)**

:   本文提出 ESP（Embedding-Space Probing）：在不修改任何权重、不训练任何辅助模型的前提下，把"prompt 嵌入均值"作为 mask token 注入到冻结 LLM 的输入序列里，借助一次前向同时探出未来多个 token，再用基础模型自身做无损推测验证，在 LLaMA3 / Qwen3 上比同类训练免费基线（LADE / STAND / PLD）的平均接受长度高 7–11%、吞吐高 15–19%。

**[Ekka: Automated Diagnosis of Silent Errors in LLM Inference](llm_efficiency/ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)**

:   Ekka 把 LLM 服务框架里"输出悄悄变烂、却没有报错"的静默错误诊断问题，建模为以 HuggingFace 这类参考实现为 oracle 的差分调试任务，用一套"组件映射 → 激活对齐 → 变点分析"的 agentic 流水线自动定位到出问题的具体模块，在 17 个真实 vLLM/SGLang issue 上取得 80% pass@1 / 88% pass@5 的诊断准确率，并新发现 4 个被开发者确认的隐藏 bug。

**[GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](llm_efficiency/graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)**

:   GraphFlow 把多个 agent 工作流统一到一张全局操作 DAG（wGraph）上，用 GNN+MLP 按任务在线生成子图工作流，并通过"基底 KV + 稀疏前缀残差 + 路径剪枝"的差分缓存替代传统按工作流独立缓存，在 5 个推理/代码/QA benchmark 上平均提升 4.95pp 的同时把 KV 内存压到约 1/4。

**[Hyperparameter Transfer with Mixture-of-Experts Layers](llm_efficiency/hyperparameter_transfer_with_mixture-of-expert_layers.md)**

:   本文把 μP/CompleteP 的最大更新参数化思想扩展到稀疏 MoE Transformer，给出 router、expert 上/下投影、expert bias 在 width/depth/专家数/专家宽度同时放大时的初始化与学习率缩放规则，并用一套三层 mean-field 的 DMFT 理论证明该参数化在 $n_{\text{embd}},n_{\text{exp}},n_{\text{hid}},L\to\infty$（固定激活稀疏度 $\kappa$）下存在尺度不变极限，从 38M 激活基模迁移到 2B 总参的 MoE 上都能直接复用最优 LR / init，且零样本超参训出来的 MoE 在等激活参数下可与 dense GPT2 speedrun 持平甚至更优。

**[KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](llm_efficiency/knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)**

:   KnapSpec 把自推测解码（SSD）的草稿层选择重新建模为 0/1 背包问题，把 Attention 与 MLP 解耦、用上下文长度依赖的硬件延迟作为"重量"、用 hidden state 余弦相似度（首次给出严格证明）作为"价值"，通过并行 DP 在每一步自适应找出最大化 Tokens-per-Time 的子网络，在 Qwen3 / Llama3 上长上下文场景拿到最高 1.47× 的真实墙钟加速且无需额外训练。

**[L$^3$: Large Lookup Layers](llm_efficiency/l3_large_lookup_layers.md)**

:   本文提出 L$^3$（Large Lookup Layer），把 tokenizer embedding table 推广为可插入到 decoder 中的"大查表层"——按 token ID 做**静态路由**取出一组学习好的 key/value embeddings，再让当前隐藏状态对其做 attention 聚合，从而在不引入 MoE 那套动态路由+辅助损失+难以 offload 的痛点下，把模型稀疏度再上一个量级；在 800M–2.6B 激活参数上击败同算力的 dense 模型与同稀疏率的 MoE。

**[MineDraft: A Framework for Batch Parallel Speculative Decoding](llm_efficiency/minedraft_a_framework_for_batch_parallel_speculative_decoding.md)**

:   MineDraft 通过维护两批请求并让一批的 drafting 与另一批的 verification 在两组独立 GPU 上**重叠执行**,把投机解码中原本串行的"草稿—验证"流水线变成批并行 PSD,在仅多用 1 张 GPU 的代价下相对标准 SD 把吞吐拉高最多 75%、端到端延迟降低最多 39%,并已实现为可即插即用的 vLLM 插件。

**[OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](llm_efficiency/obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)**

:   本文把 KV cache eviction 重新表述为"逐层结构化剪枝"问题，借用 Optimal Brain Damage 的二阶 Taylor 近似推导出针对独立 value、独立 key、key-value 联合三种剪枝单位的闭式 saliency 分数，作为即插即用的"分数替换件"接入 H2O / TOVA / SnapKV / AdaKV 等现有 attention-only eviction 框架，在 LLaMA-3.1 / Qwen-2.5 的 RULER 与 LongBench 上获得稳定提升（AdaKV 在 query-agnostic RULER-4K 30% budget 上提升近 15%）。

**[Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](llm_efficiency/optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)**

:   本文把"自洽性 (Self-Consistency) 多次采样选众数"问题建模为带先验信息的贝叶斯最优停止问题，并提出一种只跟踪"出现次数最多 / 次多 / 其他合计"三类计数的 $L$-聚合后验近似，从理论上证明 $L=3$ 即可在 $\delta \to 0$ 时达到与精确后验完全一致的渐近最优停止时间，实验上以约 1.4 倍 ASC 的速度在 GSM8K / CommonsenseQA 上节省 30%–80% 的 LLM 调用。

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](llm_efficiency/oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe 把 LLM 服务的「资源分配 + 并行策略 + 请求路由」联合建模为流网络上的双层最大流问题，配合 LSTM 工作负载预测和基于 GPU 互联的 ad hoc 模型切换，应对真实流量在空间（不同请求类型）和时间（成分随时刻变化）两个维度的异质性，端到端 P99 延迟和吞吐相比 vLLM 平均提升 1.5×、最大 2×。

**[Prism: Spectral-Aware Block-Sparse Attention](llm_efficiency/prism_spectral-aware_block-sparse_attention.md)**

:   Prism 把"块重要性估计"分解到 RoPE 的高频/低频两个频带分别做 mean-pooling 加 softmax，并用能量比推出的温度自动校准 logit 量级，从而完全用块级运算（不再回落到 token 级搜索）拿到与 full attention 几乎相同的精度，在 128K 上对 FlashAttention-2 取得 5.1× 加速。

**[ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](llm_efficiency/proactivellm_learning_active_interaction_for_streaming_large_language_models.md)**

:   ProactiveLLM 让流式 LLM 用自己的内部状态（注意力或预测熵）来决定"什么时候该开口"，靠掩码流式建模 + 同步特权自蒸馏在不依赖任何外部对齐标注的前提下学会感知"语义已经够了没"，把交互延迟显著压下去的同时几乎不掉点。

**[ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](llm_efficiency/probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)**

:   ProbMoE 把 MoE 的 top-$k$ 路由重新表述为"基数受限子集分布上的概率推断"，前向用 SIMPLE 估计器从 exact-$k$ 子集分布中采样、反向用可解析计算的专家边缘概率 $m_j=\partial \log Z_k/\partial \log p_j$ 作为离散选择的可微代理，在 OLMoE/Qwen1.5-MoE 上明显提升 GSM/Law/Translation 等任务并显著改善专家利用率，同时自然延伸出 Dynamic-$k$ 变体——按 token 难度自适应激活专家数。

**[Proxy Compression for Language Modeling](llm_efficiency/proxy_compression_for_language_modeling.md)**

:   作者提出「proxy compression」——训练时把 90% 数据喂成 tokenizer / 神经压缩器产出的短序列、10% 喂原始 UTF-8 字节，配合 sentinel token 与短暂的 in-context translation warm-up；推理时丢掉所有压缩器，模型只看原始字节，却能在固定 compute 下显著超过纯字节模型，且在大规模下追平甚至超过 tokenizer baseline。

**[ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](llm_efficiency/remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)**

:   ReMoE 冻结所有非 router 参数、仅微调 gate，用一个"时序局部性正则 + Trust-KL 语义锚"的复合损失把 router 出来的路由轨迹整形得更"缓存友好"，在不改架构、不加运行时开销的前提下把相邻 token 的专家重用率提升约 26%，并在 Jetson Orin NX 上把 TPOT 降低 43.6–49.8%（解码加速 1.77–1.99×）。

**[RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](llm_efficiency/repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)**

:   通过给 MoE 大模型喂"同一个 token 重复 N 遍"这种极简的 OOD 提示，作者发现 router 会把几乎所有 token 路由到固定的少数几个 top-$k$ 专家上，从而在专家并行（EP）部署下制造单卡瓶颈、把别的 GPU 全部 idle 住，在 8-GPU 集群上把 TTFT 拉高 20%–148%，把 MoE 的并行加速器反过来变成 DoS 攻击面。

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](llm_efficiency/scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout 把百万级 token 的长文本理解重新建模为"主动信息觅食"过程，引入与交互轨迹解耦的、带来源锚点的 epistemic state $\mathcal{E}_t$ 作为唯一推理底座，并通过 gap-diagnosed 自评估迭代收缩到查询充分子集，在 LooGLE-v2 和 $\infty$Bench 上既追平甚至超过 Gemini-3-Pro 等前沿模型，又把 token 成本降低到约 $1/8$。

**[SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](llm_efficiency/siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)**

:   针对 Pre-Norm 与 Post-Norm 在单流架构内无法共存的结构性矛盾，作者提出双流残差架构 SiameseNorm，让一条未归一化流保留 Pre-Norm 的恒等梯度高速路、一条归一化流保留 Post-Norm 的主路径表征控制，通过共享残差块耦合两条流，在 400M~15B 稠密/MoE 语言模型、ViT、DiT 上均稳定优于 Pre-Norm 基线，开销可忽略。

**[Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](llm_efficiency/skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)**

:   SKILL-MOE 提出一个无需训练、以"技能"为路由信号的符号化 MoE 框架：从每个问题里抽取所需技能、按技能-模型档案在 16 个预训练 LLM 中为每条样本动态招募 k 个专家、再用任务级最优聚合器把多条 CoT 融成最终答案；配合按专家分桶的批量推理，单卡就能跑 16 个 7-8B 模型，平均比最强多智能体基线高 8.15%。

**[Sparser Block-Sparse Attention via Token Permutation](llm_efficiency/sparser_block-sparse_attention_via_token_permutation.md)**

:   本文提出 PBS-Attn，利用注意力的置换不变性，先按"全局重要性"对 key 在段内重排，把散落各处的 heavy hitter 聚拢成连续高密度块，再做块稀疏计算，从而在保持精度近乎追平 full attention 的同时，把长上下文 prefilling 端到端加速最高 2.75 倍。

**[Stochastic Sparse Attention for Memory-Bound Inference](llm_efficiency/stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA 把 attention 的 value 聚合 $AV$ 看作 "按 softmax 概率 $A$ 对值行 $V$ 做加权求和", 改成 "从 $A$ 中无放回采样 $S\ll n_k$ 个索引、直接平均对应 $V$ 行"的无偏估计, 用 stratified / systematic 采样降方差, 再写成 GPU kernel 与 FlashDecoding 对齐——在 32k context 下端到端比 FlashInfer / FlashDecoding 快 1.5× 且精度不掉。

**[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](llm_efficiency/team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)**

:   TEAM 针对 MoE 扩散语言模型（dLLM）"激活了大量专家却只接受少量 token"的固有错配，利用 block 内解码的时间一致性与空间一致性，为已解码 / 热 / 冷三类 token 设计差异化的专家激活与解码策略，在 SDAR 30B-A3B 上以近乎零精度损失换得最高 2.2× 加速。

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](llm_efficiency/theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   本文为新兴的 Attention-FFN 解耦 (AFD) 推理架构提供首个理论框架,基于"prefill 长度有限均值 + decode 长度服从几何分布"的概率工作负载模型,推导出 rA-1F 拓扑下最优 A/F 比的闭式解 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,并用 trace-calibrated 模拟器验证理论与实测最优值偏差 <10%。

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](llm_efficiency/training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   本文提出一套训练与推理共享完全相同的分段前向执行语义的长上下文 LLM 框架：跨段只保留固定长度的可微分 KV 尾部 + 一条仅前向的检索旁路，在 LLaMA2-7B 32K/80K 上以约 $6\times$ 更低的 prefill 峰值显存达到与全注意力可比甚至更好的 LongBench/RULER 表现。

**[Variational Routing: 校准 MoE Transformer 的可扩展贝叶斯框架](llm_efficiency/variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)**

:   提出变分路由框架 VMoER——通过对 MoE 层的路由决策进行变分推断而非权重推断，实现高效贝叶斯不确定性建模，在保持 <1% FLOPs 额外开销的同时将校准误差降低 94%、路由稳定性提升 38%。

**[WarmServe：一次加载多模型的 GPU 预热机制](llm_efficiency/warmserve_enabling_one-for-many_gpu_prewarming_for_multi-llm_serving.md)**

:   WarmServe 通过分析 LLM 服务工作负载的长期周期性规律，主动将多个模型参数预加载到 GPU，配合优化的放置算法和动态 KV 缓存预留策略，使系统能在请求突发时快速启动新实例——尾部 TTFT 相比现有系统降低 50.8 倍。

---

## 🎬 视频生成 { #video_generation }

**[AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](video_generation/aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene.md)**

:   AAD-1 用“因果生成器 + 双向视频级判别器”的非对称对抗蒸馏和 DMD warmup，把自回归 image-to-video 生成压缩到每个 chunk 只需一步采样，同时缓解 motion collapse 和长程漂移。

**[Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](video_generation/attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)**

:   SVOO 发现视频 DiT 每一层的注意力稀疏度是「层内输入无关、层间显著异质」的内在属性，据此先做离线分层稀疏度标定、再做在线 QK 双向协同聚类划块，免训练地在 Wan/HunyuanVideo 等 7 个模型上把 PSNR 维持 29 dB 的同时实现最高 1.93× 加速。

**[CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior](video_generation/camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio.md)**

:   CamGeo 通过**训练专用蒸馏**（training-only distillation）从预训练 3D 视频模型（VGGT）蒸馏 3D 几何知识——仅在训练阶段提供监督信号使扩散模型能在**稀疏相机输入**条件下生成几何一致且运动平滑的高质量视频，推理时完全移除 VGGT 以保持效率。

**[DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](video_generation/dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)**

:   DFSAttn 通过 **3D Hilbert 曲线重排序** + **分层块评分** + **自适应掩码缓存**，实现了与全注意力相媲美的质量下 **2.1× 端到端加速**——解决了块稀疏注意力在高稀疏率（>80%）下质量下降的核心问题。

**[Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](video_generation/enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)**

:   MIGA 通过**两阶段训练推理对齐**（TTA）和**双重一致性增强**（DCE：自反射 + 长距离帧指导）两个核心机制——在无需训练的前提下使基础视频模型能够生成**无限长**且**高度时间一致**的视频，VBench 综合评分相比 FIFO-Diffusion 提升 2.8%（97.82 vs 95.02）。

**[EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](video_generation/epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)**

:   EPiC 用"基于第一帧可见性掩码"的方式从任意 in-the-wild 视频直接构造像素级对齐的 anchor 视频，再配一个仅 26M 参数（<1% backbone）、且只在可见区域生效的 Anchor-ControlNet，在冻结 CogVideoX-5B-I2V 主干、5K 视频、500 步训练的条件下，把 I2V 相机控制误差刷到 SoTA，并零样本泛化到 V2V。

**[Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](video_generation/explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)**

:   本文提出了**长视频中 AI 生成片段的时序定位与可解释分析**任务，引入 **TASLE 大规模数据集**和**两阶段 MSLoc 基线方法**——通过边界感知提议生成和 MLLM 精化实现对混合真伪视频中篡改片段的精确定位和可解释推理。

**[Exploring Data-Free LoRA Transferability for Video Diffusion Models](video_generation/exploring_data-free_lora_transferability_for_video_diffusion_models.md)**

:   本文首次对视频扩散模型（VDM）的 full fine-tune (FFT) 和 LoRA 做权重空间分析，发现两者都"保留奇异谱、只旋转奇异子空间"，但在 head clusters 上路由方向冲突；据此提出 CASA——一个 data-free 的"按聚类做谱仲裁"的 LoRA 迁移方法，把基座 Wan2.1 上训的 LoRA 直接迁到 FastWan 等蒸馏后变体，无需任何用户数据/重训。

**[iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](video_generation/itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance.md)**

:   iTryOn 首次定义"交互式视频虚拟试衣"任务——让人在视频里**主动操作衣物**（拉拉链、提衣角、拉伸衣物）而非仅被动展示。通过**3D 手部先验**解决空间歧义、**动作感知 RoPE（A-RoPE）** 把时间戳动作标题与对应帧严格对齐、**动作感知约束损失（AC Loss）** 放大稀疏交互帧的学习信号，在自建 VVT-Interact 上 ISR（交互成功率）从基线 0.397 → 0.610（+54%）。

**[Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](video_generation/light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)**

:   Light Forcing 是首个为自回归（AR）视频扩散模型定制的稀疏注意力方案——块感知增长（CAG）量化每个生成块的累积误差贡献来动态分配稀疏度，分层稀疏注意力（HSA）通过帧级 → 块级二级掩码选择灵活捕捉历史依赖，在 Self Forcing 上实现 1.30× 端到端 / 3.79× 注意力加速且 VBench 总分 84.5 > 密集基线 84.1。

**[Lightning Unified Video Editing via In-Context Sparse Attention](video_generation/lightning_unified_video_editing_via_in-context_sparse_attention.md)**

:   针对 In-Context Learning 范式下视频编辑的二次注意力瓶颈，作者基于"context token 显著性低于 source token"以及"Query 锐度正比于 Taylor 近似误差"两条洞察设计了 In-context Sparse Attention（ISA），并训练出 LIVEditor，在多个 benchmark 上既加速 ~60% 又超越 SOTA 全注意力模型。

**[LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](video_generation/locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)**

:   LocoT2V-Bench 是面向**长视频 + 复杂场景**生成的专业基准——234 段真实视频 × 18 主题 × 平均 249 字提示词，配套 LoCoT2V-Eval 5 维度 17 子维度评估框架（含分层 VQA + 条件门控 + Auditor-Evaluator 双代理 HERD），系统评估 17 个长视频生成模型，揭示了"感知质量强、细粒度对齐弱、角色一致性差"的普遍瓶颈。

**[LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts](video_generation/luve_latent-cascaded_ultra-high-resolution_video_generation_with_dual_frequency_.md)**

:   LuVe 把 UHR 视频生成从"被动细节增强"重新定义为"主动内容补全"——通过三阶段级联（低分辨率运动 → 潜空间上采 → 高分辨率细化）+ 频域分析驱动的双频率专家（低频专家增强全局语义一致性、高频专家细化纹理），在 VBench 4K 上达 84.03 总分超过 UltraWan-4K 的 83.75。

**[MiVE: Multiscale Vision-language features for reference-guided video Editing](video_generation/mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)**

:   MiVE 把 Qwen3-VL 的**首层 + 末层**隐状态同时抽出来作为多尺度条件 token, 与 VAE 视觉 latent 拼成一个长序列, 在统一的自注意力 DiT 里做参考图引导的视频编辑, 在 60 段 720P benchmark 上人类偏好和 6 个 VLM 自动评分都拿到第一, 超过开源 Wan-Animate 和商用 Kling O1.

**[MotiMotion: Motion-Controlled Video Generation with Visual Reasoning](video_generation/motimotion_motion-controlled_video_generation_with_visual_reasoning.md)**

:   MotiMotion 通过 VLM 推理把用户稀疏不精确的轨迹和文本提示**转化**为物理可信且因果一致的动作轨迹和文本描述，再用**置信度加权**的控制策略引导扩散模型生成符合世界知识和物理原理的自然视频——在 MotiBench 上物理真实性 0.302 远超 Wan-Move 的 0.218（+38%）。

**[OLAF-World: Orienting Latent Actions for Video World Modeling](video_generation/olaf-world_orienting_latent_actions_for_video_world_modeling.md)**

:   OLAF-World 通过**序列级控制-效应对齐**（Seq∆-REPA）学习可迁移的隐式动作——把无标注视频转化为动作可控的视频世界模型，实现跨上下文的零样本动作迁移；用 1 分钟的标注数据即可达到 AdaWorld 2 小时数据下的性能（旋转控制精度 0.4680 vs 0.6420）。

**[Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](video_generation/quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)**

:   QVG 是面向自回归视频扩散的训练免微调 KV-Cache 量化框架——通过语义感知聚类做 token 平滑、并以渐进残差多阶段压缩残差，在 LongCat-Video/HY-WorldPlay/Self-Forcing 上把 KV 显存压低到原来的 1/7，端到端延迟开销 <4%，2 bit 下质量大幅领先 KIVI/QuaRot 等 LLM 量化基线。

**[Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation](video_generation/quantized_keys_steal_attention_bias_correction_for_kv-cache_compression_in_video.md)**

:   本文发现分块自回归视频扩散模型中 KV 缓存量化导致**注意力权重发生系统性偏移**（"量化键窃取注意力"），通过推导出基于 Jensen 不等式的逐分数纠正项，在 INT2 激进量化下恢复接近 BF16 的视频质量（VBench 78.02 vs 78.27），节省 50% 内存。

**[Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories](video_generation/rays_as_pixels_learning_a_joint_distribution_of_videos_and_camera_trajectories.md)**

:   把每个相机的 per-pixel 光线"原点+方向"打包成一张与 RGB 同形状的 3 通道 raxel 图，让预训练视频 VAE 直接当相机编码器，再用 Decoupled Self-Cross Attention 把 raxel 和视频帧塞进同一个 Flow Matching DiT 联合去噪，从而第一次用一组权重同时支持位姿估计、相机可控视频生成与"视频+轨迹"联合生成三件事。

**[Self-Refining Video Sampling](video_generation/self-refining_video_sampling.md)**

:   把预训练 flow matching 视频生成器自身当作"去噪自编码器"，在推理时同一噪声层级内用 Predict-and-Perturb 内循环反复纠偏 latent，再用模型自洽性算出的不确定度 mask 只精修动态区域，从而在不引入任何外部 verifier、不做任何额外训练的前提下显著改善视频的运动连贯性与物理合理性，人评偏好率超 70%。

**[SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](video_generation/sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)**

:   SGMD 通过引入**稳定的 teacher stop-gradient Fisher 目标**和**双重势（NR/RC）机制**——解决 few-step 视频扩散蒸馏中 fake score 追踪代价高（DMD2 每轮 5 次更新）和运动抑制问题，4 步蒸馏下实现 ~3× 训练加速同时运动质量从 0.65 提升到 0.78（VideoAlign）。

**[T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)**

:   T2AV-Compass 是首个针对文本到音视频（T2AV）生成的综合评估基准——500 条复杂提示 + 双层评估框架（低层信号指标 + 高层 MLLM 诊断），系统评估 15 个前沿 T2AV 系统，定量揭示了即便顶级模型也存在的"音频真实感瓶颈"现象（视频维度 85%+ 真实感 vs 音频仅 50%）。

**[VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](video_generation/vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)**

:   VAnim 把开放域 text-to-SVG 动画建模为「持久 DOM 树上的稀疏状态更新」+「Identification-First 运动规划」+「GRPO 渲染感知强化学习」，序列长度压缩 $9.86\times$ 的同时保持拓扑一致，并显著超越 GPT-5.2、Gemini 3 Pro 与 LiveSketch。

**[VEDA: Scalable Video Diffusion via Distilled Sparse Attention](video_generation/veda_scalable_video_diffusion_via_distilled_sparse_attention.md)**

:   VEDA 把视频 DiT 的稀疏注意力问题重新表述为"对全注意力结构的显式蒸馏"——通过统计感知的瓦片评分 + 头感知分组搜索 + 硬件高效内核，在 90-95% 极端稀疏度下保持生成质量，给 Waver-12B 720P 10 秒视频带来 5.1× 端到端加速、10.5× 注意力加速。

**[Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models](video_generation/where_concept_erasure_should_occur_concept-layer_alignment_in_text-to-video_diff.md)**

:   这篇论文发现文本到视频扩散模型中的目标概念只在特定深度最可分，提出 CLEAR 用 Gumbel-Softmax 学习“在哪一层擦除”、用 SAE 学习“擦除哪个概念方向”，从而在不改动扩散模型权重的情况下更精确地抑制目标概念并保留视频质量。

**[WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](video_generation/wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling.md)**

:   WIND 把全球大气序列建模成一个无条件视频扩散先验，并在推理时把预测、下采样、稀疏重建、质量守恒和暖化情景都写成可微逆问题，用同一个冻结模型零样本求解多类天气与气候任务。

**[World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](video_generation/world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)**

:   World-R1 把文本到视频模型的 3D 一致性问题转化为强化学习后训练：用隐式相机条件和 3D-aware reward 对 Wan 2.1 等视频基础模型做 Flow-GRPO 对齐，在不改模型架构和推理流程的情况下显著减少几何幻觉，同时保持一般视频生成质量。

**[WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](video_generation/worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)**

:   WorldCache 针对扩散式 world model 中 RGB/深度等多模态 token 演化不均匀的问题，用曲率把 token 分成稳定、线性和混沌三类并自适应触发完整前向，在 HunyuanVoyager、Aether 等模型上最高实现 3.65 倍到 3.7 倍端到端加速，同时基本保持世界生成和 3D 重建质量。

---

## 📊 LLM 评测 { #llm_evaluation }

**[Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](llm_evaluation/agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)**

:   本文提出 Agent World Model，一条从场景、任务、数据库、MCP 工具接口到验证器的全合成流水线，生成 1000 个可执行数据库驱动环境，并用它们训练工具调用 Agent，在 BFCLv3、$\tau^2$-bench 和 MCP-Universe 上取得更强的域外泛化。

**[AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](llm_evaluation/agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)**

:   AGZO 发现线性层梯度行空间受前向激活子空间约束，并据此在零阶微调中只沿激活引导的低秩方向扰动参数，从而在几乎保持 MeZO 级别显存占用的同时提升梯度对齐和下游任务性能。

**[Automatic Layer Selection for Hallucination Detection](llm_evaluation/automatic_layer_selection_for_hallucination_detection.md)**

:   提出 FEPoID（内在维度的首个有效峰值）作为无需训练的自动层选择准则，并结合首句截断策略（FST），在多种 QA 和摘要幻觉检测基准上持续选出接近最优的中间层，显著超越已有基线方法。

**[BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback](llm_evaluation/bespoke_benchmark_for_search-augmented_large_language_model_personalization_via_.md)**

:   提出 Bespoke 基准，通过 30 名标注者 3 周的真实聊天+搜索历史收集 2,870 个会话，构建包含细粒度偏好评分与诊断反馈的评测框架，系统评估搜索增强 LLM 的个性化能力，发现当前模型在所有配置下平均得分均不超过 60，个性化瓶颈在于历史推理而非生成。

**[Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum](llm_evaluation/beyond_log_likelihood_probability-based_objectives_for_supervised_fine-tuning_ac.md)**

:   本文系统研究了 SFT 中概率类目标函数的行为规律，发现标准 NLL 并非普适最优：在模型先验强的任务上先验倾向（prior-leaning）目标如 $-p$ 显著优于 NLL（最高提升 16%），而在先验弱的任务上 NLL 仍然占优，揭示了由模型能力连续谱（model-capability continuum）主导的目标函数选择原则。

**[Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](llm_evaluation/beyond_trajectory-level_attribution_graph-based_credit_assignment_for_agentic_re.md)**

:   提出 GraphGPO，将所有 rollout 轨迹聚合为统一的状态转移图，利用图上全局最短路径信息为每一步计算基于距离的 advantage，实现比轨迹级归因更精细的信用分配，在 ALFWorld、WebShop、Sokoban 上显著超越 GRPO 和 GiGPO。

**[Building Reliable Long-Form Generation via Hallucination Rejection Sampling](llm_evaluation/building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)**

:   提出 SHARS 框架，在推理时逐句检测并拒绝幻觉内容、仅保留经验证的事实段落继续生成，配合改进的语义熵检测器 HalluSE，在 FactScore 上将事实精度提升约 20–26%，同时保持甚至增加生成中的事实信息量。

**[CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting](llm_evaluation/capbencher_give_your_llm_benchmark_a_built-in_alarm_for_test-set_overfitting.md)**

:   CapBencher 通过为每道题注入随机性（生成多个逻辑正确答案并随机选一个作为标准答案），将 benchmark 的 Bayes 准确率降到可控水平（如 50%），从而在公开发布 benchmark 的同时实现数据污染的黑盒统计检测——任何准确率显著超过 Bayes 上界的模型都被判定为存在污染。

**[Decompose, Structure, and Repair: A Neuro-Symbolic Framework for Autoformalization via Operator Trees](llm_evaluation/decompose_structure_and_repair_a_neuro-symbolic_framework_for_autoformalization_.md)**

:   本文提出 DSR（Decompose-Structure-Repair）神经符号框架，把自然语言定理形式化拆解为「分解 NL 成分 → 联合生成 FL 成分与算子树（OPT） → 基于子树定位的层级化修复」三阶段，在 ProverBench / ProofNet / PRIME 上以 7B 模型刷新 SOTA，并配套发布 156 题的研究生级 Lean 4 基准 PRIME。

**[DEI: Diversity in Evolutionary Inference for Quality-Diversity Search](llm_evaluation/dei_diversity_in_evolutionary_inference_for_quality-diversity_search.md)**

:   本文提出 DEI，把多个**不同家族的 LLM**当作 Quality-Diversity 搜索里的异构变异算子分布到不同节点，用全异步 gossip 互相广播每轮 champion 形成跨模型对抗压力，在 Core War 程序合成任务上以等总算力换来比单节点 +124% 的 QD-Score 与 +28% 的 archive coverage。

**[Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](llm_evaluation/discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)**

:   DoLQ 在 LLM 符号回归的搜索循环里插入一个 "Scientist Agent"，对候选项同时做定性（物理合理性）+ 定量（消融式 MSE 贡献）评估，把 LLM-SR 那种 "低误差但项数臃肿、物理上荒谬" 的方程逼到既数值精确又结构紧凑。

**[From Human-Level AI Tales to AI Leveling Human Scales](llm_evaluation/from_human-level_ai_tales_to_ai_leveling_human_scales.md)**

:   本文用 LLM 当人口外推器，把 18 个能力维度按"全世界人口正确率"对数刻度 $L=-\log_B p_W$ 校准，并发现 Volume / Attention 维度真实 base $B \gg 10$、Comprehension 维度 $B \approx 1$，揭示现行 AI 与人类的比较其实严重失调。

**[Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](llm_evaluation/geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)**

:   Top-W 把 next-token 截断写成"考虑 token embedding 几何的 Wasserstein-熵-质量"三项最小化问题，理论证明最优解要么是单 token、要么是按 $f(i)+\lambda\log p_i$ 排序的前缀，工程实现只是 $O(n\log n)$ 的扫描；在 GSM8K、GPQA、AlpacaEval、MT-Bench 上 15 个 (T, model) 组合多数胜出，高温下 GSM8K 比 Top-H 最多再提 33.7%。

**[HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](llm_evaluation/hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)**

:   HiPER 把 LLM agent 的扁平 RL 改造成"高层规划子目标 + 低层执行原子动作"的两级 Plan-Execute 结构，并配套提出 Hierarchical Advantage Estimation (HAE) 把 GAE 沿子目标段切开做有界差分耦合的优势估计，在 ALFWorld / WebShop 上分别拿到 97.4% / 83.3% 成功率（Qwen2.5-7B），相对最强基线 GiGPO 提 +6.6% 与 +8.3%。

**[Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](llm_evaluation/investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)**

:   本文提出「黑盒环境交互」作为评估 LLM 集成式推理（演绎+归纳+溯因）的新范式，构建含 6 类任务 96 个环境的 ORACLE 基准，benchmark 19 个 LLM 后发现：即便最强的 o3 也只能在简单环境拿 70% 准确率、难环境跌到 40%，且所有 LLM 都缺乏「根据反馈自适应优化探索策略」的高层规划能力。

**[Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)**

:   本文提出 Multi$^2$ 框架，把 LLM agent 的"规划"与"执行"显式拆成 System 1（SFT 训练的子目标规划器）和 System 2（offline-to-online RL 训练的原子动作执行器），通过角色专属 LoRA 适配器和带策略锚定/KL 正则的训练目标，在 ScienceWorld、ALFWorld、TextCraft 三个长时序交互环境中显著缓解了 objective drift 并提升了 token 效率。

**[On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](llm_evaluation/on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)**

:   作者从「评测有效性」和「训练效率」两条主线系统审视 LLM 工具调用：一方面用 BFCL 作为案例证明随机种子、多轮模板、思考历史、系统提示等"小细节"能让排行榜分数大幅漂移，使跨论文比较不可靠；另一方面定位 RL（GRPO）训练中 rollout 和 policy update 两个阶段的浪费，提出"在线预 rollout 过滤 + 最大方差 rollout 下采样"两件套，在单轮/多轮工具调用上实现 1.7× 和 2.6× 端到端加速且性能不降。

**[PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)**

:   PoliticsBench 是基于**多阶段角色扮演**的新型基准——通过 20 个政治情景和 4 阶段交互评测 LLM 的政治价值观表达，发现 7 个主流 LLM 都呈左倾（19-39 分），唯有 Grok 右倾（-22.7）但波动性最大；**情景提示比直接提问更能激发模型的价值观维度**（特征激活 +0.48、承诺度 +1.39）。

**[REAL：把回归感知奖励塞进 RL，让 LLM-as-a-Judge 学会"差一分也是差"](llm_evaluation/real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)**

:   针对 LLM 充当评分器（LLM-as-a-Judge）时 RL 用 0/1 二值奖励忽视序数结构的本质缺陷，作者把 RAFT 的"期望值预测 + 平方误差"塞进 RL 目标，因为奖励此时显式依赖策略参数，所以改用广义策略梯度——它干净地拆成"CoT 探索项 + 预测精修项"两部分；在 8B–32B 多基座上相对 SFT/标准 RL 全面胜出，Qwen3-32B 上 Pearson/Spearman 相对 SFT 提 8.4/7.2 点。

**[Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](llm_evaluation/reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)**

:   RACER 把"对每个 query 决定要不要调用 reasoning 模式做 judge"建模为带 KL 不确定集的分布鲁棒约束优化问题，用 primal-dual 算法解出 OOD 下仍满足 cost 预算的最优路由策略，并首次给出 LLM 路由器策略的 linear convergence 理论保证。

**[Resolution Diagnostics for Paired LLM Evaluation](llm_evaluation/resolution_diagnostics_for_paired_llm_evaluation.md)**

:   本文把 LLM 排行榜上"A 比 B 高 0.X pp"的排名当作配对假设检验问题,通过反演 level-α / power-(1-β) 检验定义"分辨率比" $q=N/N^\star$,并证明常用计算器把单臂 Cohen-$h$ 公式乘 $(1-\rho)$ 这种捷径在小效应下会系统性低估所需样本量一倍,实测发现 Open LLM Leaderboard v1 有 11/40 对、MMLU-Pro top-10 相邻对有 4/9 在 $(\alpha,1-\beta)=(0.05,0.8)$ 下根本"分辨不出来",再叠加多重比较、真实学科聚类、anytime-valid 后这个数还会涨到 6/9。

**[Spherical Steering: Geometry-Aware Activation Rotation for Language Models](llm_evaluation/spherical_steering_geometry-aware_activation_rotation_for_language_models.md)**

:   本文提出 Spherical Steering：在 LLM 隐藏状态的单位超球面上，沿测地线把激活向量旋转到由对比样本估计出的"真实性方向"，而不是像传统 activation addition 那样做线性加法，从而在保持激活幅值（norm）的同时显著提升 TruthfulQA / COPA / StoryCloze 等基准的多选准确率（+10% 量级），且不损伤开放式生成质量。

**[Toward Training Superintelligent Software Agents through Self-Play SWE-RL](llm_evaluation/toward_training_superintelligent_software_agents_through_self-play_swe-rl.md)**

:   本文提出 Self-play SWE-RL (SSR)，让同一个 LLM 在沙箱化代码仓里既扮演"造 bug 的 proposer"又扮演"修 bug 的 solver"，仅以 Docker 镜像为输入、用一致性校验和 solve-rate 作奖励做联合 RL，在 SWE-bench Verified 与 SWE-Bench Pro 上分别自我提升 +10.4 / +7.8 分，并稳定优于使用人类标注 issue + 测试套件的"human-data"基线。

**[When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](llm_evaluation/when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)**

:   这篇论文把 AI benchmark 饱和定义为前沿模型之间失去可靠区分度，提出基于 leaderboard 不确定性的 saturation index，并分析 60 个文本 LLM benchmark，发现近一半已高饱和，年龄和测试集规模比私有测试集、开放式输出或模板多样性更能解释饱和。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](llm_evaluation/when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   CAIA 用 17 个前沿大模型在 178 个时间锚定的加密货币真实任务上构建首个"对抗性高风险"agent 基准，发现：无工具时所有模型只有 12–28% 准确率（接近随机猜测），有工具时最强 GPT-5 也只到 67.4% vs. 人类入门分析师 80%；更致命的是模型 55.5% 的工具调用偏向"不可靠的网页搜索"而绕过权威链上数据，导致 Pass@k 指标系统性掩盖了"靠试错碰运气"的危险行为。

**[Who can we trust? LLM-as-a-jury for Comparative Assessment](llm_evaluation/who_can_we_trust_llm-as-a-jury_for_comparative_assessment.md)**

:   这篇论文指出多个 LLM 评审在成对比较中可靠性差异很大，提出带评审判别参数的 BT-σ 模型，在没有人工校准标签的情况下同时学习候选输出排名和每个 LLM judge 的可靠性，从而比简单平均和标准 Bradley-Terry 聚合更接近人类排序。

**[Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts](llm_evaluation/whose_alignment_comparing_llm_process_alignment_across_diverse_organizational_de.md)**

:   这篇论文提出 CALM 来评估 LLM 是否按组织真实决策过程而不只是输出结果对齐，并通过 ECHR 法律裁判与 German Credit 信贷决策的对比说明：在规范稳定的领域过程对齐能预测准确率，而在价值争议领域，高过程对齐既难实现也未必应该追求。

---

## 🧊 3D 视觉 { #3d_vision }

**[AvAtar: Learning to Align via Active Optimal Transport](3d_vision/avatar_learning_to_align_via_active_optimal_transport.md)**

:   本文提出 AvAtar，一个基于最优传输（OT）的主动对齐框架，通过梯度传播量化候选查询对全局对齐结果的影响，并利用伴随状态法和共轭梯度法以线性复杂度高效求解，在网络对齐和跨域对齐任务上一致超越已有主动学习策略。

**[Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation](3d_vision/convex_distance_operator_transport_a_convex_and_geometry-preserving_formulation.md)**

:   本文提出 CDOT（Convex Distance Operator Transport），通过把每个度量空间的距离矩阵和耦合一起"算子化"，用 $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$ 替代 FGW 中那个非凸的成对距离差平方，从而首次得到一个**对耦合 $\pi$ 严格凸**、同时仍然是合法伪度量、并具备有限样本风险界的异构空间对齐框架。

**[APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs](3d_vision/distilling_neuro-symbolic_programs_into_3d_multi-modal_llms.md)**

:   本文提出 APEIRIA，把神经符号 3D 概念学习器的程序执行轨迹蒸馏成 3D MLLM 的自然语言 chain-of-thought，再通过 GRPO 强化学习把这种推理模式推广到开放词汇与深层嵌套指令，在 ScanRefer、Multi3DRefer、SQA3D、Scan2Cap 上同时超越传统 NS3D 方法和当前最强的 3D MLLM，并保留了符号系统的可解释性与模块可替换性。

**[Fast-SAM3D: 3Dfy Anything in Images but Faster](3d_vision/fast-sam3d_3dfy_anything_in_images_but_faster.md)**

:   针对 SAM3D 单视图 3D 重建模型推理太慢的问题，本文做了第一份模块级时延剖析，发现性能瓶颈来自三种异质性（形状/布局动力学差异、纹理稀疏性、几何谱差异），并据此提出训练无关的 Fast-SAM3D 框架，用模态感知步缓存、时空 Token 雕刻与谱感知 Token 聚合三件套，在几乎不损质量的前提下把对象级速度推到 2.67×，重建 F-Score 反而从 92.34 微升到 92.59。

**[FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](3d_vision/foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)**

:   本文提出 FoundObj，把 2D/3D 自监督基础模型（DINOv2 + TRELLIS）当作奖励器，用一个"超点合并 + PPO"的 RL 代理在无任何场景级人工标注下完成复杂室内场景的多类 3D 物体分割，在 ScanNet/S3DIS/ScanNet200 上将无监督 SOTA 的 AP 从 19.6 提到 24.2。

**[FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](3d_vision/fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)**

:   本文把人类“先扫一眼再逐块细看”的观察过程抽象为 Focus-Sweep 两阶段范式，用 Mamba 替换 Transformer 做图像-点云交互，并用强化学习动态决定每个尺度上的交互层数，在 RGB-D Scenes V2 和 7-Scenes 上拿到 I2P 配准的 SOTA。

**[Geodesic Flow Matching for Denoising High-Dimensional Structured Representations](3d_vision/geodesic_flow_matching_for_denoising_high-dimensional_structured_representations.md)**

:   针对 Vector Symbolic Architecture 里 Spatial Semantic Pointer 这种"被嵌进单位超球面 Clifford 超环面"的高维结构化表示，作者指出标准 Flow Matching 的欧氏直线插值会从球面内部"穿心而过"导致幅值塌缩、相位毁掉，于是用 Log/Exp 映射把流约束在球面上做 **Geodesic Flow Matching (GFM)**，在脉冲神经 SLAM 上把路径误差降低 72%，并让 1500 神经元的路径积分器达到 2500 神经元 baseline 的精度。

**[Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning](3d_vision/geometry-guided_modeling_of_foundation_features_enables_generalizable_object_sha.md)**

:   本文提出 GODeform：把 2D 基础模型（DINOv3 类）特征"挂"到类别模板表面上做几何引导传播与跨视点融合，再用 Flow Matching 学一个从模板到目标的逐点形变场，从而在大形变、任意视角和未见类别上都能从单张图恢复 3D 形状，并直接服务于灵巧抓取迁移。

**[HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance](3d_vision/hoi-page_zero-shot_human-object_interaction_generation_with_part_affordance_guid.md)**

:   HOI-PAGE 让 LLM 先"想清楚"身体哪个部位该接触物体哪个部件，把推理结果写成一张「部件 affordance 图」(PAG)，再用它去驱动 3D 部件分割、视频扩散和优化求解，从而在零样本、零 4D 训练数据的条件下生成可处理"多人单物 / 单人多物"等复杂场景的 4D 人-物交互序列。

**[LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory](3d_vision/labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo.md)**

:   LabBuilder 把自由文本的实验描述编译成"资产-化学协议"，再用层级化生成 + 几何/化学多目标优化 + 导航修复，产出既视觉合理、又能让机器人真正跑通实验流程的 3D 化学实验室布局。

**[PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation](3d_vision/physcene3d_physically_consistent_interactive_3d_tabletop_scene_generation.md)**

:   PhyScene3D 把 3D 桌面场景生成重塑成"人类构造式"的层次化序列规划：用 Cognitive Topological Reasoning Chain (CTRC) 把场景图线性化为基于 AABB 的锚点序列，再用 Physics-Aware Denoising Alignment (PADA) 把可微分 SDF 物理引擎嵌入 VLM 训练循环，使模型生成的场景在物理合理性上反超人工标注训练数据（场景级碰撞率从 81.5% 降到 41.6%，资产级降到 3.86%）。

**[PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](3d_vision/physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)**

:   本文提出 PhysHanDI，把 MANO 手模型和 Spring-Mass 软体模型耦合起来，用稠密手网格驱动可变形物体的物理仿真，并反向利用物体仿真去精化手的重建，在稀疏视角 RGB-D 视频上同时拿到了手和软物的稠密 3D 重建 SOTA。

**[PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations](3d_vision/plaid_a_unified_data_model_for_machine_learning_on_heterogeneous_physics_simulat.md)**

:   PLAID 提出一套面向异构物理仿真数据的统一数据模型与开源库，配套发布 6 个覆盖结构力学和 CFD 的工业级数据集与可复现基准，把"变网格、变拓扑、变维度"的真实仿真数据真正变成机器学习社区可用的标准化 benchmark。

**[RelaxFlow: Text-Driven Amodal 3D Generation](3d_vision/relaxflow_text-driven_amodal_3d_generation.md)**

:   RelaxFlow 把"用文字补全被遮挡 3D 物体"形式化为一个**双目标控制粒度解耦**问题，提出训练免调的双分支推理框架——观察分支保持像素级硬约束、语义先验分支用"多先验共识 + 注意力 logit 高斯模糊"实现低通松弛——并从理论上证明这一松弛等价于对生成向量场做低通滤波，从而在 SAM3D / TRELLIS 等 SOTA 上把 Point-FID 从 100.38 降到 81.11。

**[Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction](3d_vision/revisiting_photometric_ambiguity_for_accurate_gaussian-splatting_surface_reconst.md)**

:   AmbiSuR 把 Gaussian Splatting 的两类内生光度歧义（基元边缘外溢、像素混合欠约束）显式建模并用截断 + 射线-颜色一致性消歧，再借高阶球谐系数作"自指示器"找出歧义高风险基元并做无定形局部先验正则，在 DTU 上把平均 Chamfer 距离降到 0.46，超过此前最优 GeoSVR (0.47)。

**[SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising](3d_vision/simpc_learning_self-induced_mirror-point_consistency_for_unsupervised_point_clou.md)**

:   SIMPC 提出在**同一个噪声点**上沿去噪向量做"对称延伸"得到一个位于曲面另一侧的镜像点，再用 Mirror-Point Consistency Loss 强制两点的去噪目标重合，从而把无监督点云去噪从"在多份噪声变体间找统计对应"换成"在单点内部找确定性几何对应"，在 PUNet/PCNet 合成数据和 Paris-Rue-Madame / Kinect 真实扫描上全面超越无监督 SOTA，并击败若干有监督方法。

**[Smoothness Errors in Dynamics Models and How to Avoid Them](3d_vision/smoothness_errors_in_dynamics_models_and_how_to_avoid_them.md)**

:   作者从理论上指出 Kiani 等人的 "unitary GNN" 因为强行保持 Rayleigh 商而对热扩散这类"天然会变光滑"的物理系统过度约束，进而提出"松弛 unitary 卷积"（R-UniGraph / R-UniMesh）并把整套 Rayleigh 商-unitary 卷积框架从图扩展到三角网格，在 MeshPDE 与 WeatherBench22 上同时超越多类强基线。

**[SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](3d_vision/splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)**

:   本文指出多模态点云补全里"硬投影把 3D 点直接打到 2D 网格"会让支持集 Lebesgue 测度为零、梯度被 Dirac delta 截断（称为 Cross-Modal Entropy Collapse），用可微 Gaussian Soft Splatting 把硬投影换成连续密度估计，搭配 EdgeConv 局部 + Transformer 全局的混合编码器和全局-局部解码器，在 PCN/ShapeNet-55/34 拿到 SOTA，并用 KITTI 上的 counter-factual 评估证明 baseline 实际是退化的"单模态模板检索器"。

**[STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](3d_vision/stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)**

:   STABLE 把"任务指令→可仿真桌面场景"拆成 LLM-based Semantic Reasoner（出粗布局）和 flow-matching + SDF 损失的 Physics Corrector（修位姿），并让两者按 task-critical → background 三阶段交替迭代，最终在 MesaTask-10K 上把物体碰撞数压到 0、任务对齐 AwS 拉到 99.0%。

**[Streaming Sliced Optimal Transport](3d_vision/streaming_sliced_optimal_transport.md)**

:   Stream-SW 是首个能在"样本流"上估计 sliced Wasserstein 距离的算法：每个一维投影上用 KLL/quantile sketch 维护近似分位函数，把 1D Wasserstein 的闭式积分变成可流式更新的估计量，空间复杂度对样本数仅对数级，从而把 SOT 带入 IoT / 边缘设备等"看一次就丢掉"的场景。

**[SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding](3d_vision/svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa.md)**

:   SVL 用「3D-图像-文本」三模态对比预训练给脉冲神经网络（SNN）注入开放世界理解能力，并通过把文本编码器"重参数化"为一组分类权重，让推理阶段完全脱离文本塔、保持纯脉冲驱动，在 ModelNet40 零样本分类上达到 85.4% 同时能耗仅为同档 ANN 方法的 0.5%–11%。

**[The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](3d_vision/the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)**

:   本文揭示 LLM 中"注意力汇聚到第一个 token"的结构性根源 —— 因果掩码下首 token 缺乏 value 聚合导致维度方差差异,被 FFN 中的 super neurons 选择性放大形成维度极度悬殊,最终锁死 QK 投影迫使形成 attention sink;并据此提出 head-wise RMSNorm 在预训练阶段从根上抑制 sink。

**[TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization](3d_vision/tidegs_scalable_training_of_over_one_billion_3d_gaussian_splatting_primitives_vi.md)**

:   TideGS 把 3DGS 的参数表搬到 SSD 上，按"块"虚拟化并以 GPU VRAM 作为视锥可见工作集的缓存，配合三级异步流水线和轨迹自适应差分流式传输，在单张 24 GB GPU 上首次把可训练的高斯数量从约 11M（原生 3DGS）/ 105M（CLM）推到 **超过 10 亿**，且大场景重建质量优于所有评测的单卡基线。

**[Trust3R: Evidential Uncertainty for Feed-Forward 3D Reconstruction](3d_vision/trust_it_or_not_evidential_uncertainty_for_feed-forward_3d_reconstruction_with_t.md)**

:   Trust3R 为 MASt3R 等 feed-forward 3D 重建模型引入概率化证据学习框架，用 Normal-Inverse-Wishart 先验为每个 3D 点预测闭形式多元 Student-t 分布，取代启发式置信度，单遍前向推理就能输出概率可解释的逐点不确定性，并在 ScanNet++ 上 AURC 降低 25%、AUSE 降低 41%。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Adaptive Probe-based Steering for Robust LLM Jailbreaking](llm_alignment/adaptive_probe-based_steering_for_robust_llm_jailbreaking.md)**

:   这篇论文把 probe-based contrastive steering 改造成更强的白盒红队评测工具，用自适应重训练修正有偏 probe，并用激活统计自适应设定 steering 强度，从而显著暴露加固 LLM 的越狱脆弱性。

**[Alignment-Aware Decoding](llm_alignment/alignment-aware_decoding.md)**

:   Alignment-Aware Decoding 直接在推理时利用 DPO 模型相对 SFT 参考模型的 token 概率比作为隐式对齐奖励，在无需额外训练或外部 reward model 的情况下，比 greedy、Bo2 和 EFT 更稳定地生成高对齐质量回答，并可进一步产生合成偏好数据改进 DPO。

**[Curriculum Learning for Safety Alignment](llm_alignment/curriculum_learning_for_safety_alignment.md)**

:   本文提出 Staged-Competence —— 一个把"模型自身的偏好对齐 margin"作为难度分、再用"分阶段更新参考模型 + 阶段内 competence-based 采样"双重课程的 DPO 安全对齐框架，在三种 8B 量级 LLM 上把 OOD 有害回答率平均降 16%、越狱攻击成功率降 20%，同时几乎不损伤通用能力与不引入过度拒答。

**[Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards](llm_alignment/decoupling_reasoning_and_confidence_resurrecting_calibration_in_reinforcement_le.md)**

:   本文先理论证明 RLVR（如 GRPO）训练中"提升准确率"与"减小校准误差"两个目标在 Fisher 度量下梯度方向负相关、不可调和，再提出 DCPO：让模型在推理轨迹后显式吐出一段 verbalized 置信度，给推理 token 和置信度 token 分配各自的 reward / advantage / 掩码梯度，从而在保持 GRPO 同等准确率的前提下把 ECE 从 0.435 降到 0.128（相对降 71.6%）。

**[Efficient Preference Poisoning Attack on Offline RLHF](llm_alignment/efficient_preference_poisoning_attack_on_offline_rlhf.md)**

:   针对 log-linear DPO 提出"翻一条偏好标签 = 给损失梯度加一个与策略参数无关的固定向量"的关键观察，把目标投毒攻击归约为二值稀疏近似问题，给出基于 LLL 格基约化的 BAL-A 和基于匹配追踪的 BMP-A 两种算法以及可证明的恢复 / 不可能性条件。

**[$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](llm_alignment/f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses.md)**

:   本文给在线 RLHF 在**通用 $f$-divergence 正则**下首次建立 $O(\log T)$ regret 和 $O(1/T)$ 次优 gap 上界，提出两套采样策略：(1) 基于 optimism in face of uncertainty 加 bonus 项；(2) 一个新颖的 **"derivative-as-uncertainty"** 视角——把 $f'$ 当作不确定性信号，从而设计 derivative-based 采样而无需在每轮显式估计 confidence bound。

**[F-TIS: Harnessing Diverse Models in Collaborative GRPO](llm_alignment/f-tis_harnessing_diverse_models_in_collaborative_grpo.md)**

:   F-TIS 把"截断重要性采样 (TIS)"与"按 KL 阈值过滤负优势 off-policy 样本"两件事拼到一个 GRPO 损失里，让大小不同、专长不同、甚至只有一部分参数可训的多个 LLM 在同一次去中心化 GRPO 训练中互相喂样本，最终收敛和纯 on-policy 持平，并在 OOD 数学任务上最高带来 +12% 的性能。

**[GIST: 用梯度子空间投影做 instruction tuning 的 targeted 数据选择](llm_alignment/gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)**

:   GIST 把"为 target task 挑 instruction tuning 数据"看作 gradient subspace alignment——证明 LESS 等用 Adam states 当 diagonal preconditioner 在 LoRA 上失效（cross-parameter 耦合 + 低秩 task subspace），改用 validation gradients SVD 抽 task-specific 低秩子空间 + cosine similarity 选样本；在 MMLU/TydiQA/BBH 上匹配或超越 LESS，只用 0.29% 存储和 25% 计算时间。

**[Implicit Preference Alignment for Human Image Animation](llm_alignment/implicit_preference_alignment_for_human_image_animation.md)**

:   作者提出 Implicit Preference Alignment (IPA)：一种只需"好样本"、不需要构造好/坏配对的后训练方法，通过最大化与预训练参考模型 KL 间隔来等价地最大化隐式奖励，并配合一个把手部 mask 加权进损失的 HALO 模块，让大尺度视频 DiT 在仅 93 个挑选样本下显著改善人体动画的手部保真度。

**[Implicit Safety Alignment from Crowd Preferences](llm_alignment/implicit_safety_alignment_from_crowd_preferences.md)**

:   针对众包偏好数据中"用户目标各异但安全准则共享"的结构，作者证明传统 reward combination 会被多数用户偏好污染且对权重敏感，转而提出 Safe Crowd Preference-based RL：用 VAE 把众包偏好编码成 latent-conditioned 低层 skill，再训练高层策略在 skill 空间组合，从而在没有显式安全奖励的情况下把下游 cost 压到接近 Oracle，同时任务回报基本不掉。

**[Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization](llm_alignment/long_live_the_balance_information_bottleneck_driven_tree-based_policy_optimizati.md)**

:   本文用信息瓶颈 (IB) 理论提出一个可量化"探索-利用平衡"的步级指标 IB-Score, 并据此设计 IB 引导的树采样 (IBTree) + 步级局部/全局优势, 在 Qwen3-1.7B/8B 上比 GRPO 平均提升 2.9–3.6%, 同时在同 token 预算下多采到 50% 轨迹.

**[MESA: Improving MoE Safety Alignment via Decentralized Expertise](llm_alignment/mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)**

:   MESA 把 MoE 安全对齐重塑为"在专家上分配安全责任"的资源分配问题，用 KL 正则化的 Sinkhorn 最优传输（OT）从中间档（shoulder region）专家中挑出代价最低的子集做 SFT，同时用 OT 约束的路由损失把安全 token 引到这些专家，从而在 DeepSeek-V2-Lite / Qwen3-30B-A3B 上把 Strata 安全分推到 95+%，并保住 GSM8K 等推理任务接近原始水平。

**[Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](llm_alignment/mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)**

:   本文把 Bradley–Terry 奖励模型重写成一个贝叶斯非负因子分析（NFA）的生成过程——局部稀疏的实例隐变量 $\bm{\theta}$ 与全局稀疏的奖励字典 $\Phi$ 同时建模，以"先解耦再去偏"抑制 RLHF 中由长度/风格等捷径特征引起的 reward hacking，并通过 Weibull 重参数化的摊销变分推断把整个框架塞进现代 LLM 主干，在 Unified-Feedback、RewardBench、HHH、MT-Bench 上一致超过 BT、Ensemble、InfoRM 等强基线。

**[New Wide-Net-Casting Jailbreak Attacks Risk Large Models](llm_alignment/new_wide-net-casting_jailbreak_attacks_risk_large_models.md)**

:   本文首次定义并系统分析了"广撒网"越狱场景（攻击者同时向一组大模型发起请求，只要任一模型被攻破即视为成功），并据此设计了一种基于 exploration-to-exploitation 调度的"专家化"对抗样本生成器联合训练方法，在多个 LLM/MLLM 上把无外加防御时的攻击成功率推到 100%，揭示现行单模型越狱评估严重低估了真实世界风险。

**[PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](llm_alignment/picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)**

:   PICACO 把"让 LLM 在一个 prompt 里同时遵守多个甚至互相冲突的人类价值"形式化为最大化"价值集与响应之间的条件总相关性"(Total Correlation, TC),不动模型参数,通过 EM-like 的"响应增强 + 指令精炼"两步迭代自动搜索一条 meta-instruction,使 GPT-3.5 / LLaMA-3.1-8B / Gemini-1.5-Flash 在 Schwartz、HH 等 5 套最多 8 个价值的组合上都超过 OPRO、Modular Pluralism 等强基线。

**[Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](llm_alignment/quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)**

:   作者用 Inglehart-Welzel 文化地图把标注者按"文化区/象限"重新分层，在 8 个安全数据集上用多层级回归（multilevel modeling）证明文化区在控制完人口学（年龄/性别/族裔）之后仍显著解释安全评分的方差（6/8 数据集 $p<0.05$），并提出 Bayesian 的"cultural sensitivity score"量化得出：当前数据集中约 10% 的样本若忽略某一文化象限就会被错标为 safe；进一步实验表明 LLM 当 rater 替身不靠谱，但当"文化敏感样本"的 triage 工具是可行的。

**[Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](llm_alignment/safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)**

:   本文证明所有现有「在参数空间设约束」的 HFT 防御都会因参数冗余而被绕过，提出 Safety Bottleneck Regularization (SBR) 把防御战场搬到 unembedding 层这一几何瓶颈上：仅锚定 1 个高危 prompt 的最后一层隐状态，就能在 50 epoch 持续 HFT 攻击下把 Harmful Score 压到 < 10，同时不损 benign 任务精度。

**[Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards](llm_alignment/simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa.md)**

:   MAHALO 把"标准化 PRM 训练 + 多动作头 DPO + 带 KV-cache 续存的 PRM 引导解码"拼成一套统一框架，让一个 LLM 在数学（可验证）、人类价值观（不可验证）、多轮辅导（交互式）三类目标上同时被对齐，并且在推理时能通过头权重与 PRM 选择平滑地切换偏好。

**[SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](llm_alignment/spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)**

:   SPARD 用"安全投影交替优化（SPAG）+ 相关性-多样性 DPP 安全样本选择"两件套，把"微调后模型必须满足安全损失约束"显式写成约束优化问题，每步先做效用更新，再用闭式投影把参数拉回安全半空间，同时只用 3% 任务相关且彼此互异的安全样本，就把四种有害微调攻击的平均 ASR 从 SFT 的 87.93% 砍到 9.45%，几乎不掉下游精度。

**[Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation](llm_alignment/steering_beyond_the_support_adversarial_training_on_unsupervised_jailbroken_acti.md)**

:   论文针对监督式 safety steering 在未见越狱攻击上失效的问题，提出用"无监督潜在方向发现 + 双层对抗训练"在激活空间里凭空模拟出新型 jailbroken 状态，并把这些模拟状态当作对抗样本来训练一个 OT 势函数（其梯度构成空间变化的引导场），在三个 LLM × 六类经典越狱上把攻击成功率压到大多数 <5% 且基本不伤害良性效用。

**[Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](llm_alignment/toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)**

:   本文提出 SVGT，把价值对齐从"嵌入 backbone 参数/激活"改为"挂一个独立的价值模块"，先在隔离的 value space 里持续判断当前 hidden state 的安全方向，再用一组可学习的 Bridge Token 作为注意力锚点显式引导生成轨迹，在四种 backbone 上把有害分数普遍降低 70% 以上且几乎不损失流畅度。

**[Towards Context-Invariant Safety Alignment for Large Language Models](llm_alignment/towards_context-invariant_safety_alignment_for_large_language_models.md)**

:   作者提出 AIR（Anchor Invariance Regularization），把可验证 prompt 当作"锚"、用 stop-gradient 只把开放式变体往锚的表现上拉，作为辅助损失插入 GRPO，在安全/道德/数学三域把 OOD 组级一致性平均提升 33.49%、ID 提升 12.71%。

**[HRC + DSPPO: 用博弈论分解把传递偏好和循环偏好分开学](llm_alignment/transitivity_meets_cyclicity_explicit_preference_decomposition_for_dynamic_large.md)**

:   HRC 把人类偏好显式拆成正交的「传递标量分量」（BT 模型）+「循环向量分量」（GPM），用博弈论分解定理证明这种 hybrid 形式既能保 dominant 候选又能建模 RPS 式循环，再配套时变博弈 DSPPO 让对齐过程从「先稳住传递骨架，再学循环细节」走到 Nash 均衡——在 RewardBench 2 上 Gemma-2B-it 平均涨 1.23%、AlpacaEval 2.0 LC win-rate 拉到 44.75%。

**[UDM-GRPO: 统一离散扩散模型的稳定高效 GRPO](llm_alignment/udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)**

:   通过将最终干净样本定义为动作并使用前向过程重构轨迹——首次成功将 GRPO 集成到离散扩散模型中，解决训练不稳定问题，在 GenEval 等多个基准上达到 SOTA。

---

## 🏥 医学图像 { #medical_imaging }

**[Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?](medical_imaging/are_we_overconfident_in_models_and_results_for_semi-supervised_3d_medical_image_.md)**

:   这篇论文指出半监督 3D 医学图像分割同时存在模型伪标签过度自信和评测协议过度乐观两类问题，并提出 TCSeg 用置信度-不确定性双轴可靠性和概率、特征、图像三空间校准来抑制确认偏差，同时倡导多随机种子、best/last checkpoint 同时报喜报忧的评测方式。

**[Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](medical_imaging/auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)**

:   本文提出 S(H)NAP——基于 3D 扩散桥的「移除 + 插入」生成式干预框架，把 Sybil 这一前沿肺癌风险预测模型的决策反向拆解为「肺结节主效应 + 两两交互 + 背景」的 LMPI（线性+二阶交互模型），首次以因果而非相关的方式审计出它对 ECG 电极、衣物金属扣等院内伪影的依赖以及对外周肺结节的「径向不敏感」严重失败模式。

**[CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)**

:   提出 CASCADE 框架，将两阶段临床决策系统中第一阶段分类器的认知不确定性（通过 Venn-Abers 预测器量化）传播到第二阶段回归预测区间，使高置信患者的预测区间缩窄 38.9%，同时为不确定病例自动扩展安全缓冲，实现自适应覆盖保证。

**[DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring](medical_imaging/discontinuous_galerkin_neural_operator_for_pathology_defocus_deblurring.md)**

:   DGNO 把病理显微图像的散焦去模糊重新表述为"空间变化积分算子"的反问题，用不连续 Galerkin 风格把全局核拆成元素局部积分算子 + 界面数值通量，既保留神经算子的物理可解释性，又能处理病理图像本质上的局部不连续模糊；在 BBBC006w1 等数据集上超越 NAFNet / Restormer / MambaIRv2 等 SOTA。

**[DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](medical_imaging/dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)**

:   本文提出 DP-KFC：基于"Fisher 矩阵的标度由架构决定、相关结构可用模态级频谱统计近似"的观察，用结构化合成噪声（图像用 $1/f^\alpha$ pink noise，文本用 Zipf 采样）探测网络重建 KFAC 预条件子，既不消耗隐私预算也不引入分布偏移，在强隐私（$\varepsilon\le 3$）下持续超过 DP-SGD 与公共数据预条件方法。

**[EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](medical_imaging/eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)**

:   EEG-MoCE 给 EEG-based 多模态学习（情绪/睡眠/认知）每个模态分配一个**可学习曲率**的 Lorentz 流形 expert，再用"曲率大→层级结构更丰富→在 fusion 中权重更高"的 curvature-aware attention 做跨模态融合，在 EAV/ISRUC/Cognitive 三个数据集上 cross-subject 准确率分别 +14.14%、+3.34%、+7.98%。

**[Evidential Reasoning Advances Interpretable Real-World Disease Screening](medical_imaging/evidential_reasoning_advances_interpretable_real-world_disease_screening.md)**

:   EviScreen 用「正常 + 病理」双知识库做区域级证据检索，再以 cross-attention + self-attention 在当前病例和证据间做循证推理，既给出**回溯式可解释性**（哪几个历史病例支持当前判断）又给出**定位可解释性**（对比检索得到的异常图），在 4 个真实外部测试集上把高召回处的特异性提升到 SOTA。

**[Factored Classifier-Free Guidance](medical_imaging/factored_classifier-free_guidance.md)**

:   本文识别出 CFG 在扩散模型反事实生成中存在「属性放大 (attribute amplification)」失效模式——单一全局 $\omega$ 会把本不该改变的属性一起放大，并提出 FCFG：按因果图分组、为每组属性分配独立 guidance 权重，从而在 CelebA-HQ / EMBED / MIMIC-CXR 上显著降低非目标属性漂移、改善反事实可逆性。

**[Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](medical_imaging/federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)**

:   本文提出 FedHD：在异构联邦病理学场景下，用 Gaussian-mixture 特征对齐做「一对一」WSI 特征级蒸馏，再通过课程学习把跨机构合成特征逐步注入本地训练，使各机构能在不共享原始数据、不交换模型参数的前提下协作，且兼容异构 MIL 架构与特征提取器，在 TCGA-IDH / CAMELYON16 / CAMELYON17 上全面超越现有联邦与蒸馏基线。

**[Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](medical_imaging/foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)**

:   本文论证了一个反直觉但实用的发现——在自然图像/视频上预训练的 Foundation VAE 不需要任何医学微调就能作为统一接口同时支持 CT 重建、增强、生成；其重建只是去噪不偏移边界，因此重建图既可做去噪增强（pancreatic / lung tumor NSD +3.9%），其潜空间又可承载 CT 条件扩散生成（FVD −3.9%，CT-CLIP +36.2%，多疾病忠实度 AUC +2.76%）。

**[OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](medical_imaging/geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)**

:   OT-Bridge Editor 把"在冠脉造影上编辑一段血管狭窄"重写为"在血管-结构复合域里的约束熵 OT 问题"，用 Schrödinger Bridge 沿路径加几何投影监督，做到像素级形状/位置可控的合成造影，在 ARCADE 公开集上把下游狭窄检测 mAP@0.5 相对提升 27.8%。

**[Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis](medical_imaging/learning_multi-scale_hypergraph_for_high-order_brain_connectivity_analysis.md)**

:   MuHL 用可学习尺度的图小波把脑 ROI 特征分解成多分辨率表征，再以"节点嵌入 × 共享投影矩阵"动态生成 soft 超边，让 AD/PD 多阶段分类在 ADNI 上做到 93.2% Acc、PPMI 上做到 76.8% Acc，同时给出可解释的关键 ROI 与超边。

**[Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](medical_imaging/marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)**

:   本文提出 DiffDT：用一个条件 Latent Diffusion 框架把电子病历（ICD-coded 事件序列）与多器官生物标记数字孪生（脑/心/肝/肾的影像衍生表格特征与脑功能连接 SPD 矩阵）连起来，关键创新是一个基于 Cholesky 分解的 SPD-VQVAE 把 $\mathcal{O}(N^3)$ 的 SPD 流形扩散降到流形保形且高效的潜空间，再让 AR 模型借“生成数字孪生 → 预测下一个 ICD”这条中介路径完成多通路疾病推理；在 UKB 上对 1944 类疾病的下一次预测 AUC 提到 0.91，刷新 SOTA。

**[MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery](medical_imaging/medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant.md)**

:   用中国餐馆过程 (CRP) 对临床文本 prompt 做在线贝叶斯非参数聚类，自动发现"语义模态"，再为每个语义模态分配独立 LoRA 适配器并配合模态内 EWC，在 16 个医学分割任务上把 Dice 推到 73.3% 同时遗忘率降到 4.1%，参数仅为 MoE 基线的 1/6。

**[MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](medical_imaging/meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)**

:   MEG-XL 用 2.5 分钟（191k token）的 MEG 上下文做 mask token 预训练（比此前长 5–300×），再微调到 50 词的脑到文本任务上，仅用 1 小时数据就达到 SOTA 监督方法 50 小时的解码精度，并显著超过所有 brain foundation models。

**[PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder](medical_imaging/pacx-mae_physiology-augmented_chest_x-ray_masked_autoencoder.md)**

:   PaCX-MAE 在 MAE 预训练的胸片 ViT 之上，用 LoRA 微调把 ECG 和实验室检验两类生理信号编码器作为冻结教师，通过 InfoNCE 对比 + 余弦回归的双重蒸馏，把"看不见的生理上下文"注入纯图像编码器，推理时只需胸片即可在 9 个下游基准上整体超越同架构 MAE 基线，对生理依赖性任务尤为明显（MedMod +2.7 AUROC、VinDr +6.5 F1）。

**[Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction](medical_imaging/plug-and-play_diffusion_meets_admm_dual-variable_coupling_for_robust_medical_ima.md)**

:   本文把 ADMM 的对偶变量重新塞回 PnP 扩散先验循环，用"对偶"提供积分反馈消除稳态偏差，再用一个频域 Spectral Homogenization 模块把结构化对偶残差白化成伪 AWGN，避免触发扩散去噪器的 OOD 幻觉，在 sparse-view / limited-angle CT 与加速 MRI 上同时拿到 SOTA 保真度和约 3× 推理加速。

**[Scaling Vision Transformers for Functional MRI with Flat Maps](medical_imaging/scaling_vision_transformers_for_functional_mri_with_flat_maps.md)**

:   把 3D fMRI 体积按"皮层展平图"投影成 2D 视频后直接喂给标准 spacetime MAE-ViT，得到一个在 2.1K 小时 HCP 数据上训练的 CortexMAE：在认知状态解码上大幅超 SOTA，验证 flat map 是体素 (volume) 和脑区平均 (parcellation) 之间的"goldilocks zone"；同时发布首个开源 fMRI 基础模型基准 Brainmarks，给出 fMRI 模型的第一份系统 scaling law 与一个"个体特质预测仍打不过简单功能连接 baseline"的诚实 null result。

**[Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding](medical_imaging/seizure-semiology-suite_s3_a_clinically_multimodal_dataset_benchmark_and_models_.md)**

:   本文构建了首个大规模专家标注的癫痫发作视频数据集 S³（438 段视频、35,000+ 密集标签、20 项 ILAE 语义学特征），配套设计了七级层次化任务基准与临床对齐的 Seizure-RQI 报告质量指标，系统暴露了 11 个开源 MLLM 在时序定位、空间偏侧化和临床忠实性上的失败模式，并通过领域微调 + 两阶段神经符号框架将癫痫 vs 非癫痫分类 F1 提升到 0.96。

**[SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](medical_imaging/semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)**

:   SEMIR 把体素栅格当作母图 $G$，通过参数化的边收缩 / 节点删除 / 边删除把它压成一张「边界对齐」的图 minor $H$（节点数从 $\sim10^7$ 降到 $\sim10^3$），用 5–20 张少样本黑盒优化 $\Theta$ 最大化边界 Dice，再在 minor 上用 GNN 做超节点分类，最后通过 minor 与体素之间的双射 exact lifting 回到原栅格——在 BraTS / KiTS / LiTS 三大肿瘤分割任务的少数类 Dice 上稳定超过 nnU-Net，且仅需 16GB T4 GPU。

**[SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](medical_imaging/synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)**

:   SynerMedGen 提出"生成对齐理解（generation-aligned understanding）"原则——把理解任务直接从同一份配对合成数据里派生出来（CTS / MI / TIA 三个任务），先两阶段训练让理解分支学到对合成有用的表征，再迁移到 latent flow matching 生成分支，在 22 个医学合成任务上同时碾压专用合成模型和已有统一 MLLM。

**[CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](medical_imaging/the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)**

:   本文用 SDE 框架分析放射学报告生成（RRG）多任务学习里"报告生成 vs 临床约束"梯度冲突的两面性——drift term 偏离 Pareto 最优 + diffusion term 衰减无法逃局部最优；提出 CAME-Grad 优化器（方向纠偏 + 能量注入 + 自适应融合）作为线性缩放的即插即用替代，在 MIMIC-CXR / IU X-Ray 上 8 个 RRG 方法平均临床效能 +2.3% / +1.9%。

**[PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](medical_imaging/thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)**

:   PathCTM 把全切片图像（WSI）分析从"穷举高倍 patch"重构为"从低倍全局到高倍局部"的连续多尺度推理——基于 Continuous Thought Machine 引入 thinking-in-scales 范式 + 注意力引导区域剪枝 + 置信感知早停，patch 数减少 95.95%、推理时间减少 95.62% 且 AUC 不降反升。

**[Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](medical_imaging/turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)**

:   本文把多个 MLLM 之间的推理"漂移"重新解释成 DPO 中的负样本约束，用 Plackett-Luce 偏好损失同时压制 N 个 source model 的发散轨迹，让 7B 学生模型在不需要 ground-truth 报告的前提下，仅用 10% 的 MIMIC-CXR 就在胸片分类与报告生成任务上超过所有 source teacher。

---

## ⚛️ 物理/科学计算 { #physics }

**[A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](physics/a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)**

:   本文从最小作用原理出发，提出 Wasserstein 拉格朗日力学（WLM）框架，学习二阶人口动力学而非传统梯度流的一阶动力学，从而能够捕捉周期性、旋转等更丰富的群体现象，并可在不需要参考过程的情况下完成插值与未来预报。

**[ANTIC: Adaptive Neural Temporal In-situ Compressor](physics/antic_adaptive_neural_temporal_in-situ_compressor.md)**

:   为了把 PB-EB 级别 PDE 仿真数据"边算边压"，本文提出 ANTIC：用 physics-aware 时间选择器只保留物理上重要的快照，再用神经场 + LoRA 持续微调编码相邻快照之间的残差，在 2D Kolmogorov 流上拿到 435× 压缩、在 4.2 TiB 的 3D 双黑洞合并模拟上拿到 6807× 时空联合压缩。

**[BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields](physics/ballast_bayesian_active_learning_with_look-ahead_amendment_for_sea-drifter_traje.md)**

:   提出 BALLAST 算法，通过从 GP 后验中采样向量场并模拟拉格朗日观测器的未来轨迹来修正主动学习的效用估计，同时开发了 VaSE 推理方法将 GP 后验采样效率提升数千倍，在合成与高保真海洋流场上实现约 16%-22% 的部署成本节省。

**[Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation](physics/distribution_transformers_fast_approximate_bayesian_inference_with_on-the-fly_pr.md)**

:   Distribution Transformer (DT) 把"先验分布"显式 token 化为一组高斯混合分量、把"观测"通过交叉注意力注入解码器，端到端学一个"先验+数据 → 后验"的映射，在保持与先验同族（GMM→GMM）以支持序贯滤波的同时，把推断时间从分钟级压到毫秒级，并允许测试时任意更换先验而无需重训。

**[EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](physics/eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)**

:   EqGINO 把 GINO 的 GNO 编码器、FNO 主干、GNO 解码器全部改造成 SE(3) 等变模块：GNO 用相对距离作为旋转不变核、FNO 用"轨道权重共享"在频域强制 $W(R\mathbf k)=W(\mathbf k)$ 的各向同性，从而在保留 FNO 全局感受野的同时让 3D PDE surrogate 对任意刚性变换鲁棒，且把谱权重参数量从 $\mathcal O(K^3)$ 降到 $\mathcal O(K)$。

**[Generative Neural Operators Through Diffusion Last Layer](physics/generative_neural_operators_through_diffusion_last_layer.md)**

:   在任何神经算子骨干（FNO/DeepONet）后挂一个"扩散末层"（DLL）：用一个输入相关基 $\Phi_a$ 把目标场压成 $r$ 维系数向量，再用一个小 MLP 速度场在系数空间做条件流匹配，从而把确定性算子升级成既能采样随机解又能给出滚动不确定性的生成式算子。

**[Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](physics/hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)**

:   论文把 Instant-NGP 的多分辨率哈希表升级为"梯度增强"版本——在每个哈希格点同时存储函数值与所有混合偏导，再用 Hermite 插值重建出 $C^1$ 连续、内部解析可二阶可微的场，从而让 NGP 第一次能真正用于 PINN 求解 PDE，在 2D/3D 多个基准上比 SOTA 神经 PDE 求解器降误差最多 $20\times$，单 epoch 训练只要 $2$–$3.5\,\mathrm{ms}$。

**[Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](physics/iterative_refinement_neural_operators_are_learned_fixed-point_solvers_a_principl.md)**

:   论文给已训练好的神经算子（FNO/TFNO/WDSR 等）外挂一个共享权重的 U-Net 修正模块 $\Phi_\theta$，在推理时按 $h_{k+1}=h_k+\alpha\Phi_\theta(x,h_k)$ 反复迭代，把单次前向的预测变成一个收敛到唯一不动点的"学习版残差求解器"，在湍流、活性物质、ERA5 超分等任务上把误差降低 34%–80%，并能稳定外推到训练步数的 2 倍。

**[Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting](physics/learning_to_refine_spectral-decoupled_iterative_refinement_framework_for_precipi.md)**

:   SDIR 把雷达 0–2 小时降水临近预报重新表述为"频域解耦的迭代精化"过程：先用 SFG-Former 提取稳定的低频天气骨架，再用 FR-Refiner（含 Fourier 神经算子）按频段逐步合成高频对流细节，并用一条对齐 Kolmogorov 湍流功率律的 PCPSD 损失替代会导致过平滑的纯 MSE，在 CIKM / Shanghai / SEVIR 三个 benchmark 上同时显著超过回归类与扩散类 SOTA。

**[$\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](physics/mathbbr2k_is_theoretically_large_enough_for_embedding-based_top-k_retrieval.md)**

:   本文证明对于内积、欧式距离与余弦三种打分函数，能够把 $m$ 个对象的全部 size $\le k$ 子集都用 score-thresholding 精确召回所需的最小嵌入维度（MED）是 $\Theta(k)$，与 $m$ 无关；在加上单位归一化与正向 score margin $\epsilon$ 之后，鲁棒 MED 的可行 margin 被 $\epsilon_\star(m,k)=m/\sqrt{k(m-1)(m-k)}\sim 1/\sqrt{k}$ 上限锁死，而 Gaussian centroid 构造则给出 $O(k^2\log m)$ 维的可行上界。

**[Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics](physics/mesh_field_theory_port-hamiltonian_formulation_of_mesh-based_physics.md)**

:   从「局部性 + 置换等变 + 朝向协变 + 能量守恒/耗散不等式」四条物理原理出发，证明任何满足这些公理的网格物理动力学在雅可比层面都可以局部约化为 port-Hamiltonian 形式——其中守恒互联结构 $J$ 完全由网格拓扑（符号关联矩阵 $D_k$）固定，度量与耗散通过可学的 $G, R$ 进入；据此设计的 MeshFT-Net 在长时间 rollout 上能量漂移近零、色散与动量正确，并大幅领先 MGN / HNN。

**[MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties](physics/mōle-λ_learning_the_coupled-cluster_response_state_for_energies_gradients_and_pr.md)**

:   MōLe-Λ 把分子轨道学习从只预测耦合簇右态 $T$ 振幅扩展到同时预测左态 $\Lambda$ 振幅，用一套等变网络从局域化 Hartree–Fock 轨道直接读出 $(T_1,T_2,\Lambda_1,\Lambda_2)$，在 QM7 上能量/受力 MAE 仅 0.10 mHa / 0.12 mHa/Bohr，同时把偶极、四极、极化率、电子密度、对密度等响应性质都从同一个学到的"响应态"里解出，相对 CCSD+$\Lambda$ 求解器加速两个数量级以上。

**[PINNfluence: Interpreting PINNs Through Influence Functions](physics/pinnfluence_interpreting_pinns_through_influence_functions.md)**

:   本文把训练数据归因方法 Influence Functions 推广到物理信息神经网络 (PINN) 上，提出 PINNfluence——通过线性化的留一样本扰动估计，把 PINN 的预测/损失/物理量同时归因到每一个训练点和每一个损失分量上，并基于此构造一套诊断指标（损失分量比例、抵消分数、时间因果指标等），在 5 个时间相关 PDE 上稳定区分"训练良好 vs 训练失败"两类 PINN，给出残差分析看不到的结构性诊断。

**[Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](physics/quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)**

:   Quiver 把分类输入额外送进一个变分量子电路 (VQC)，提取其量子 Fisher 信息矩阵 (QFIM) 作为「量子几何视图」，再用 cross-attention（对 Transformer）或残差门控（对 GNN）注入到经典骨干里，在 JetClass 顶夸克标记与 QM9 HOMO-LUMO 间隙回归两个完全不同的物理任务上都拿到了稳定提升。

**[REX: A Family of Reversible Exponential Stochastic Runge-Kutta Solvers](physics/rex_a_family_of_reversible_exponential_stochastic_runge-kutta_solvers.md)**

:   本文提出 Rex —— 一族基于 Lawson 指数积分器构造的、可代数反演的（随机）Runge-Kutta 求解器，把任意显式 (S)RK 格式自动转成可精确反演的 ODE/SDE 求解器，既保证任意高阶收敛与非零稳定域，又能在扩散模型的图像重建/编辑、流模型的 Boltzmann 采样上做到接近机器精度的反演。

**[Score-Based Error Correcting Code Decoder](physics/score_based_error_correcting_code_decoder.md)**

:   本文提出 SB-ECC：把二进制线性分组码的软译码重新表述为方差爆炸 (VE) 扩散过程的反向去噪，用一个**无时间条件**、直接吃**带符号信道观测** $\mathbf{y}$ 的分数网络求解校验约束引导的概率流 ODE，在 42 种码-SNR 配置中拿下 39 项最优 BER，平均 SNR 增益 0.17 dB、最大 0.46 dB。

**[Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](physics/softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)**

:   作者把传统 Softmax attention 解构为"非负化 + L1 归一化"两个独立部件，证明真正关键的是 L1 归一化而非指数，于是用 Softplus + 动态长度尺度因子换掉指数得到 LSSA，再用一次幂函数式"重权"对注意力锐化，得到的 LSSAR 在 16× 训练长度上几乎保持 validation loss 不变，并能让 GPT-109M 从轨迹数据中"重新发现"牛顿万有引力定律。

**[Speculative Sampling for Faster Molecular Dynamics](physics/speculative_sampling_for_faster_molecular_dynamics.md)**

:   本文把语言模型里的投机采样迁移到二阶 Langevin 分子动力学，提出 LSD：用快草稿势函数串行外推、慢目标势函数并行验证，通过反射最大耦合保证轨迹分布与目标模型严格一致，在 FCC 铜等系统上获得 3–9× 无误差加速。

**[Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor](physics/teaching_molecular_dynamics_to_a_non-autoregressive_ionic_transport_predictor.md)**

:   本文把昂贵的原子轨迹当作训练时的「特权辅助模态」，用一个双模态训练器先吃轨迹学动力学，再通过闭式岭回归把它的隐藏表示蒸到一个只看平衡结构的非自回归预测器上，在锂离子均方位移预测上比自回归 SOTA 快 200× 且更准。

**[Topology-Preserving Neural Operator Learning via Hodge Decomposition](physics/topology-preserving_neural_operator_learning_via_hodge_decomposition.md)**

:   本文提出 Hodge Spectral Duality (HSD) 神经算子，把流形 PDE 的解算子按 Hodge 正交分解拆成"低频拓扑分量（谱基底）+ 高频几何分量（FNO 辅助网格）"双分支，再用一个交换子修正项耦合二者，从而在复杂网格上同时获得高精度与守恒律保真。

**[TriForces: Augmenting Atomistic GNNs for Transferable Representations](physics/triforces_augmenting_atomistic_gnns_for_transferable_representations.md)**

:   TriForces 把原子级图神经网络拆成「组成-结构-交互」三条平行流，再叠加 LeJEPA + 去噪 + 掩码的多目标自监督预训练，让 MLIP 在小样本迁移、跨域微调和相似结构检索三种场景下都比单流基座更稳。

**[Unbiased and Second-Order-Free Training for High-Dimensional PDEs](physics/unbiased_and_second-order-free_training_for_high-dimensional_pdes.md)**

:   本文针对 EM-BSDE 训练 loss 的离散化偏置问题，提出 Un-EM-BSDE：把单步误差用两组独立的 Monte Carlo 子样本平均后做"乘积"形成无偏估计，既消除偏置又不需要 Hessian，在 HJB/BSB/AC 等基准 PDE 上达到 Heun-BSDE / FS-PINNs 的精度但训练时间仅 1.79× EM-BSDE（相比 Heun-BSDE 的 42.91× 与 FS-PINNs 的 32.07×）。

**[Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](physics/understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)**

:   作者把 Transformer 自注意力写成 token 间相互作用的平均场粒子系统，把 LoRA 视作低秩扰动，证明遗忘与"扰动模长"和"网络深度"两条相变曲线相关，并给出由 $V$ 的特征值 gap 控制的长时稳定条件。

**[Unveiling Multi-Regime Patterns in SciML: 不同失败模式与域特异优化](physics/unveiling_multi-regime_patterns_in_sciml_distinct_failure_modes_and_regime-speci.md)**

:   通过系统的多域诊断框架揭示 SciML 模型（PINNs、神经算子等）存在的三种一致失败模式——并分析其损失面景特异性，为优化方法选择提供指导。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](self_supervised/a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)**

:   本文改进了监督对比学习（在有限标注数据池中构造元组）的样本复杂度上界，通过两个不同的U-统计量估计器，在极值多类场景下实现从依赖最小类概率的界到仅依赖类别数或样本规模的界的突破。

**[Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](self_supervised/beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)**

:   本文提出 SAGE，把"估计未标注数据分布"换成"在表征空间做结构推断"，用 simplex ETF 几何锚 + 高阶图传播 + 分布无关可靠性加权三件套，在极端标签稀缺且未标注分布任意的 UniSSL 设定下取得平均 8.52% 的准确率提升。

**[Can Local Learning Match Self-Supervised Backpropagation?](self_supervised/can_local_learning_match_self-supervised_backpropagation.md)**

:   本文从理论上证明了在深度线性网络中局部自监督学习（local-SSL）可以精确实现全局反向传播自监督学习（BP-SSL）的梯度更新，并据此提出 CLAPP++ 算法（引入 2D 空间依赖和直接反馈），在 CIFAR-10/STL-10/Tiny ImageNet 上达到了与全局 BP-SSL 相当的性能，刷新了 local-SSL 的 SOTA。

**[Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](self_supervised/data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)**

:   作者证明对比学习里的"预定义数据增强 (旋转/裁剪/翻转)"等价于对 Positive-incentive Noise (π-noise) 的点估计, 然后把 π-noise 从"点估计"升级为可学习分布, 训练一个 π-noise 生成器在原图上加可学噪声当增强 (PiNDA), 使 SimCLR / BYOL / SimSiam / MoCo / DINO 在 vision 上稳定涨点, 且天然适配 HAR / Reuters / Epsilon 等无人工增强的非视觉数据。

**[FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](self_supervised/flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)**

:   FLAG 把"从 H&E 病理图预测空间基因表达"重新表述为结构化分布生成问题，用一个固定的空间图编码器把组织拓扑压成条件向量，再用 DiT 在基因维度去噪，并通过基因基础模型 (GFM) 的中间层对齐注入基因-基因调控先验，从而在保持 PCC/MSE 竞争力的同时把基因结构相关性 (GSC) 和空间结构相关性 (SSC) 拉到新的高度。

**[From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)**

:   本文提出 OutFormer —— 一个用 GMM/SCM/Copula 三类合成先验混合预训练、靠多臂老虎机自演化课程稳定多任务训练的表格 PFN，做到零样本表格异常检测：上下文 (in-context) 吃训练数据、前向一步给标签，在 ADBench 与两个新 1500+ 数据集 benchmark 上同时拿到 SOTA 排名和接近 shallow 模型的推理延迟。

**[How 'Neural' is a Neural Foundation Model?](self_supervised/how_neural_is_a_neural_foundation_model.md)**

:   作者把一只"小白鼠视觉皮层的 SOTA 基础模型（FNN）"当成生理学实验对象，用解码流形 / 编码流形 / 解码轨迹三件套挨个分析它的 encoder / recurrent / readout，发现 FNN 的拟合精度主要靠 readout 那一堆同质 feature map 撑起来，而真正"像大脑"的只有 recurrent 模块；并用新提出的 tubularity 指标定量地说"早期编码层缺少生物级时间结构"，给未来神经基础模型给出"早期加 recurrence、readout 减少 feature 维度"的明确建议。

**[Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data](self_supervised/inconsistency-aware_minimization_improving_generalization_with_unlabeled_data.md)**

:   本文提出一种只用无标签数据就能计算的"局部不一致性" $S_\rho(\theta)$ —— 即参数球内 KL 散度的最坏值 —— 并把它当作训练正则项，得到 IAM 优化器，在监督任务上和 SAM/ASAM 持平甚至更好，在半监督 (FixMatch) 与自监督 (SimCLR) 场景下因能吃无标签批量数据而带来额外提升。

**[InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)**

:   InfoAtlas 把互信息估计从"每个数据集都要从头训一个评估网络"的优化问题，改造成一个用大规模合成数据预训练好的超网络的"一次前向推理"问题，做到与 MINE/MINDE 等神经估计器相当的精度同时 100× 提速。

**[Learning Graph Foundation Models on Riemannian Graph-of-Graphs](self_supervised/learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)**

:   R-GFM 把"不同 hop 数"的子图当作上层 Graph-of-Graphs 的节点，再用一套动态 MoE 路由把每个 GoG 分配到曲率最匹配的 Riemannian 流形（双曲 / 欧氏 / 球面），同时解决了现有图基础模型固定 receptive field 与单一 Euclidean 嵌入两个先天缺陷，下游最高带来 49% 相对提升。

**[Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](self_supervised/learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation.md)**

:   本文提出 Relational Task Extrapolator (RTE)，把"训练支撑集之外的新任务"重新解释为"已知锚点任务 + 已见过的任务间变换"的组合问题，并训练一个关系算子 $\Psi$ 在测试时拼装这对锚点-变换以预测未知任务的输出。

**[LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](self_supervised/lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel.md)**

:   针对大模型 selective prediction 中"UCB 风险界过于保守、能用阈值很少"这个老问题，作者把"接受后错误率 ≤ α"重写成一条关于选择/错误两个 0-1 指示函数的**线性期望约束**，由此推出一个只依赖校准集的有限样本充分条件（Eq. 5），既保持有限样本严格保证又显著比 UCB 紧，同时把同一套框架自然推广到两模型路由系统并联合标定两个阈值，在 CommonsenseQA / TriviaQA / ScienceQA / MM-Vet v2 上 power 普涨、TriviaQA 上比 Clopper-Pearson UCB 多接受 9.5% 样本。

**[LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](self_supervised/limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)**

:   针对 TabPFN-v2 等表格基础模型在浅层出现严重低秩坍缩、且最后一层 sample-attention 对预测信号贡献微弱的两个病灶，作者提出用径向基函数把每个标量扩展成一组局部响应（RaBEL）来打开"值方向"的自由度，并把双向注意力块从 F→S→N 重排成 S→N→F 以确保所有注意力路径都汇入读出，仅用 2M 参数就在主流表格 benchmark 上稳定胜过 7M 的 TabPFN-v2 和 27M 的 TabICL。

**[Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](self_supervised/mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)**

:   针对 TabPFN 这类把训练集当作 in-context 直接喂进 attention 的"表格基础模型"做后验校正——发现它会严重过拟合训练集 majority class, 提出 DistPFN：用 $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$ 这一行后验重加权, 在 253 个 OpenML 数据集上把 TabPFN-v2 在 $\beta=5$ 强标签漂移下的准确率从 72.7% 拉到 76.9%, 不用重训、不用估测试先验、不动架构。

**[NITP: Next Implicit Token Prediction for LLM Pre-training](self_supervised/nitp_next_implicit_token_prediction_for_llm_pre-training.md)**

:   NITP 通过用**浅层表示作为隐式目标**为最后隐藏状态提供连续的表示空间监督——补充标准 NTP 防止隐藏表示退化为低维各向异性配置，在 9B MoE 上 MMLU-Pro 提升 5.7%、推理任务普遍提升 4-6%，额外计算开销仅 ~2%。

**[NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](self_supervised/numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)**

:   NumLeak 通过**四层诊断协议**检测和量化基础模型对公开数值基准（金融因子、宏观经济数据、气候数据）的记忆化程度——揭示这类污染如何渗漏到下游金融信号中，并通过系统提示防御减缓风险；Opus 4.7 在 Mkt-RF 因子上的 within-25 bps 精度达 0.60、Pearson r = 0.99。

**[PartCo: Part-Level Correspondence Priors Enhance Category Discovery](self_supervised/partco_part-level_correspondence_priors_enhance_category_discovery.md)**

:   PartCo 通过显式利用 Vision Transformer 的补丁令牌中蕴含的**部分级特征对应关系**，引入一个**即插即用**的框架来增强广义类别发现——在 CUB / Stanford-Cars / ImageNet-100 等多个基准上将 SimGCD / SPTNet / FlipClass 等基线提升 2-10%。

**[Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](self_supervised/provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)**

:   作者证明:对比学习里典型的三元组任务,只要嵌入维度 $d$ 小于真维度 $D$ 的某个常数倍,无论用什么优化器,准确率都会"坍缩"到 1 维随机嵌入的 50% baseline,而且在算法层面这件事在 Unique Games 假设下也无法被多项式时间逼近。

**[Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](self_supervised/scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)**

:   作者提出 CaRE：在 ViT 每个 block 里塞一个 **两级路由 MoE (BR-MoE)** ——先靠"类感知器"按熵选 Top-M 个相关任务路由，再由这些路由各自激活 Top-K 任务专家并叠加一个共享 EMA 专家，于是哪怕任务序列拉到 300+ 也能既保留旧知识又持续吸纳新类，并把"长序列 CIL"这块此前没人正经做的空白填上（顺便发布了 1000 类的 OmniBenchmark-1K 基准）。

**[Statistical Consistency and Generalization of Contrastive Representation Learning](self_supervised/statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)**

:   本文首次为对比表示学习 (CRL) 建立了"上游对比损失最小化等价于下游 AUC 型检索性能最优"的 Fisher / 统计一致性, 并给出依赖于正样本数 $n$ 和负样本数 $m$ 的精细泛化界 $O(1/m+1/\sqrt n)$ (监督) 与 $O(1/\sqrt m+1/\sqrt n)$ (自监督), 从而首次从理论上解释了 CLIP / SimCLR 使用上万负样本能持续涨点的现象。

**[The Geometry of Projection Heads: Conditioning, Invariance and Collapse](self_supervised/the_geometry_of_projection_heads_conditioning_invariance_and_collapse.md)**

:   本文从黎曼几何视角把自监督学习中的投影头分析为可训练的度量张量，证明其作用是动态白化优化景观、用光滑激活的负曲率逃脱坍缩鞍点、并沿数据增强方向诱导度量奇异性——三件事一起解释了"训练时需要、推理时丢弃"这一长期谜团。

**[TRACER: 用 WMA teacher + 几何分解证明的鲁棒多模态微调](self_supervised/tracer_persistent_regularization_for_robust_multimodal_finetuning.md)**

:   TRACER 用闭式解理论把对比微调的几何分解为"任务子空间"+"正交保留"两部分，证明 EMA teacher 会坍缩失去正则化力，提出 Weighted Moving Average (WMA) teacher 保持 finite-horizon 持续约束力且对任务子空间无偏收敛；在 CLIP ViT-B/16 上 ImageNet 分布偏移平均提升至 64.07% vs CaRot 62.54%。

**[Understanding Self-Supervised Learning via Latent Distribution Matching](self_supervised/understanding_self-supervised_learning_via_latent_distribution_matching.md)**

:   作者把对比 / 非对比 / 预测式 SSL 统一为"潜在分布匹配 (LDM)"：最大化样本在假设潜在模型下的对数概率 (alignment) + 最大化潜在熵 (uniformity)，并基于此推出带 Kalman 预测器的非线性可识别预测式 SSL。

**[When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE](self_supervised/when_softmax_fails_at_the_top_extreme_value_corrections_for_infonce.md)**

:   这篇论文把 InfoNCE 解释为 top-1 选择似然，指出标准 softmax 隐含 Gumbel 尾部分布假设，而归一化 embedding 的高相似度 hard negatives 更常呈现有限端点的 Weibull 行为，因此提出无额外参数的 WEINCE，用 batch 内尾部统计自适应混合 softmax logit 和 endpoint shortfall logit，稳定提升自监督表征质量。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](information_retrieval/blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)**

:   提出基于锦标赛图（tournament graph）的零样本重排序框架 BlitzRank，通过将每次 $k$-wise 比较产生的 $\binom{k}{2}$ 个偏好对累积到全局偏好图中并利用传递闭包推断额外排序关系，在 14 个基准、5 个 LLM oracle 上实现 Pareto 最优——在匹配或超越现有方法精度的同时减少 25–40% token 消耗。

**[CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels](information_retrieval/care_class-adaptive_expert_consensus_for_reliable_learning_with_long-tailed_nois.md)**

:   提出 CARE 框架，利用 VLM 的文本嵌入、图像特征和原始标签三路互补专家，通过类别自适应 Top-$K$ 共识机制实现长尾噪声标签场景下的可靠标签矫正，在合成与真实基准上一致超越 SOTA 最高 3.0%。

**[Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](information_retrieval/graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin.md)**

:   Graph-R1 把 GraphRAG 重写成"知识超图环境 + 多轮 think–query–retrieve–answer 智能体 + 结果导向 GRPO"的端到端 RL 框架，用更轻量的 n 元超图构建和双路超边检索 + RRF 融合，在 6 个标准 RAG 数据集上把 7B 模型的 F1 从 Search-R1 的 46.19 拉到 57.82。

**[HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](information_retrieval/hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)**

:   本文把多步 RAG 中的 working memory 从"扁平的事实列表"重构成一张**超图**——每条超边就是一个可被 update / insert / merge 的记忆点，借助超边天然连接 $n\geq 2$ 个实体的能力，让记忆能在交互过程中持续合并低阶事实成高阶概念，从而显著提升需要"全局意义构建"的长上下文问答性能。

**[Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)**

:   Ψ-RAG 用"合并—坍缩"式的层次聚类替换 RAPTOR 的 k-means 来构建跨文档抽象树，并配上一个具备多轮重写能力的检索回答 Agent 与稀疏 BM25 混合索引，让 Tree-RAG 第一次能在语料级、跨文档多跳问答上追平甚至超越 Graph-RAG，平均 F1 比 RAPTOR 高 25.9%、比 HippoRAG 2 高 7.4%。

**[How can embedding models bind concepts?](information_retrieval/how_can_embedding_models_bind_concepts.md)**

:   本文把 "embedding 模型为什么不会绑定概念" 形式化成 "binding function 的复杂度问题"：通过几何分析证明 CLIP 的场景嵌入可加性分解成对象与概念之和（解释了单模态可探测、跨模态却失败），并在受控 Transformer 上证明当数据覆盖足够时，模型会学到一个由概念间**乘性交互**主导的低复杂度 binding，从而实现对未见对象组合的系统性泛化。

**[LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](information_retrieval/lazyattention_efficient_retrieval-augmented_generation_with_deferred_positional_.md)**

:   LazyAttention 把 RoPE 位置编码从 KV 缓存写入阶段推迟到 attention kernel 内部 on-the-fly 完成，让同一份物理 KV 副本可以被任意 logical 位置复用，在 skewed RAG 工作负载上比 SOTA Block-Attention 减少 1.37× TTFT、提升 1.40× 吞吐，且生成质量基本无损。

**[LEMUR: Learned Multi-Vector Retrieval](information_retrieval/lemur_learned_multi-vector_retrieval.md)**

:   Lemur 将多向量相似性搜索转化为监督学习问题，用一个两层 MLP 将 token 级嵌入映射到低维潜空间，再利用现有单向量 ANNS 索引完成检索，比 PLAID/MUVERA 等方法快一个数量级。

**[Less Is More: Elevating RAG via Performance-Driven Context Compression](information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)**

:   CORE-RAG 用"性能即奖励"的 GRPO 强化学习训练一个 1.5B 小压缩器，把检索到的 top-k 文档压成 ~3% 长度的摘要，结果不仅没掉点反而在 4 个 QA benchmark 上比满上下文 RAG 平均提升 3.3 EM。

**[ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](information_retrieval/ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)**

:   ML-Embed 把 Matryoshka 思想从一维 (representation 维度) 扩展到**三维** —— 在 embedding 参数 (MEL)、模型深度 (MLL)、表征维度 (MRL) 上**全栈嵌套训练**, 同时构建 282 种自然语言 + 40 种编程语言、5000 万样本的多语训练集, 推出 140M-8B 一族开源模型, 在 17 个 MTEB benchmark 上 9 个拿第一, 波兰语 +22.89, 越南语 +6.88.

**[ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs](information_retrieval/pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms.md)**

:   ParisKV 通过把 key/query 归一化并随机正交旋转到单位超球上、用"数据无关的解析质心"代替从 prefill 学习出来的质心，再叠加一个 GPU 原生的"碰撞投票 + 4-bit 量化重排"两阶段检索 + UVA 按需取 KV，在百万 token 上下文上把 Top-$k$ KV 检索的解码延迟相比 MagicPIG/PQCache 降低 17–44×，并在 9 个长生成任务里 7 个达到或超过 full attention 精度。

**[Predictive Prefetching for Retrieval-Augmented Generation](information_retrieval/predictive_prefetching_for_retrieval-augmented_generation.md)**

:   通过学习 transformer 隐状态/注意力中"早于不确定性 8–16 token 出现的语义前兆"，本文用 RetrievalPredictor + ContextMonitor + QueryGenerator 三件套把 RAG 的检索从同步阻塞改造为预测式异步预取，在 HotpotQA 等基准上把端到端延迟降低 43.5%、TTFT 降低 62.4%，同时答案质量保持在同步 RAG 1% 以内。

**[Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](information_retrieval/ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)**

:   本文提出 METEORA，用 DPO 训练的"理由生成器 + 统计肘部检测 + 同框架 Verifier"三件套，把 RAG 中不可解释、依赖 top-$k$ 的 re-ranker 整段替换掉，在 6 个敏感领域数据集上同时拿到更高召回、80% 的证据量削减和 4.4× 的对抗鲁棒性提升。

**[REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)**

:   本文提出 REAL 框架，用"Reasoning-Pivot"（推理链中必须依赖外部证据才能补全的原子节点/边）重新定义 KI-VQA 中的知识冲突，并通过 RPA-SFT 训练 pivot 感知的冲突判别器 + RPGD 训练免费的对比解码策略，在 E-VQA / InfoSeek / A-OKVQA 上分别取得 +3.8% / +1.6% / +3.6% 的提升。

**[ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)**

:   ReSeek 给 RL-trained 搜索 agent 增加一个 JUDGE 动作 + 用 BGE-reranker 计算"理想判断"作为过程奖励,使 agent 能在每次检索后软性"屏蔽"无效信息并重新查询;同时提出 FictionalHot 这一基于虚构实体的抗污染评测,Qwen2.5-7B 上平均 EM 达到 0.377,比 ZeroSearch 高 +3.1。

**[Retriever Portfolios: A Principled Approach to Adaptive RAG](information_retrieval/retriever_portfolios_a_principled_approach_to_adaptive_rag.md)**

:   本文把 RAG 中"选哪个 retriever"重新表述为一个 best-of-$k$ 组合优化问题，从 360 个候选 retriever 里离线贪心挑出一个互补的 size-$k$ 组合（portfolio），并训练一个轻量对比学习路由器在线把每个 query 分发给组合里的 top-$\ell$ 个成员，在 4 个 QA 基准上同时打过单 retriever 和 Vendi-RAG 类推理时调参方法，并显著降低 token 和延迟成本。

**[Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](information_retrieval/seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)**

:   本文用一个"颜色-形状-item"受控合成检索任务复现了"VLM 在纯文本任务上超过其 base LLM"的奇怪现象，并用机制可解释性证明：图像训练让模型把变量绑定策略从"位置捷径"切换到"语义符号匹配"，这一切换在重新接回纯文本后被保留下来，使 OOD 检索准确率从 37.2% 提升到 69.5%；在真实 Qwen2/2.5/3 家族上也观察到一致的"symbolic/positional 比例上升"。

**[Through the Stealth Lens: Attention-Aware Defenses Against Poisoning in RAG](information_retrieval/through_the_stealth_lens_attention-aware_defenses_against_poisoning_in_rag.md)**

:   本文指出现有 RAG 投毒攻击虽然能用少量恶意段落操纵 LLM 输出，但**并非真正隐蔽**——成功的低预算攻击必然会让模型把注意力过度集中在恶意段落上，因此作者用每段落归一化注意力分数 NPAS 和基于其方差的 AV Filter 把异常段落筛掉，在 4 数据集 × 5 LLM × 5 攻击的设定下把 RACC 比 Certified Robust RAG 最高拉高 20%。

**[Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](information_retrieval/understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)**

:   本文把现代 LLM 长上下文推理中的稀疏注意力、RAG、压缩上下文记忆等优化统一为四阶段 "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" 内存处理流水线，定量证明该流水线占整体延迟 22%-97% 且各阶段计算特性高度异构，并据此提出 GPU-FPGA 异构系统：把规则/算密集操作留 GPU、把稀疏/不规则/访存密集操作 offload 到 FPGA，在 MI210 + Alveo U55C 上取得最多 2.2× 端到端加速和 4.7× 能耗下降。

**[Understanding LoRA as Knowledge Memory: An Empirical Analysis](information_retrieval/understanding_lora_as_knowledge_memory_an_empirical_analysis.md)**

:   作者用 PhoneBook 与新构造的 PaperQA 基准做系统实证审计，把 LoRA 看作可独立训练 / 加载 / 组合的知识记忆单元，定量给出"秩 → 容量 → 效率 → 多模块组合 → 与 RAG/ICL 互补"全链路的设计准则。

**[基于跨模型局部等距一致性的向量链接](information_retrieval/vector_linking_via_cross-model_local_isometric_consistency.md)**

:   论文提出向量链接问题——在黑盒约束下通过发现两个不同编码器产生的嵌入云之间的对象对应关系。核心观察是独立训练的对比学习编码器在短距离内保持局部等距一致（相似度保留 up to 缩放因子），基于此提出多视图几何哈希自举框架，只需 15-30 个种子对即可恢复 79-90% 的重叠对象。

**[Very Efficient Listwise Multimodal Reranking for Long Documents](information_retrieval/very_efficient_listwise_multimodal_reranking_for_long_documents.md)**

:   ZipRerank 同时砍掉 VLM 列表式重排的两大瓶颈——「视觉 token 序列过长」和「自回归解码逐 token 输出排名」——用 query-aware token 剪枝 + 单 logit 排序在 MMDocIR 上把 LLM 推理延迟降一个数量级，同时匹配或超越当前 SOTA 的 MM-R5。

---

## 📚 预训练 { #llm_pretraining }

**[Annotations Mitigate Post-Training Mode Collapse](llm_pretraining/annotations_mitigate_post-training_mode_collapse.md)**

:   作者发现 SFT 把模型对齐到一个低熵语义先验上、导致"指令模型越大越无聊"的反向 scaling，于是提出"标注锚定训练"——预训练阶段给文档配语义 tag、SFT 阶段对 tag 部分 mask loss，让推理时先采样语义再生成响应，从而在保持指令跟随能力的同时把语义多样性差距缩小 85%。

**[Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability](llm_pretraining/beyond_structural_symmetries_linear_mode_connectivity_via_neuron_identifiability.md)**

:   本文提出"有效函数类"和"神经元可辨识性"的理论框架，揭示打破结构对称性并不等于打破有效对称性——即使参数空间的置换对称已被消除，数据依赖的近似对称仍可能使神经元互换代价极低，并据此给出无需对齐即可实现线性模式连通性（LMC）的充分条件。

**[Constrained Bayesian Experimental Design via Online Planning](llm_pretraining/constrained_bayesian_experimental_design_via_online_planning.md)**

:   本文提出 COPEx：通过"离线预训练 amortized 后验网络 + 设计策略 + 在线多步 lookahead 场景树"的半摊销方案，让贝叶斯实验设计在测试时能动态适应预算 / 成本 / 转移约束，在受约束的 location finding、CES、cost-aware AL 三类任务上 EIG / RMSE 一致超过 VPCE、ALINE、RL-BOED 等基线。

**[Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)**

:   本文系统研究 SFT 中数据难度的作用，发现并不存在"普适最优难度"，而是存在一个**随数据规模增大而向更难方向漂移**的最优难度，并用"in-distribution 泛化 gap"与"extrapolation gap"两个 gap 的 trade-off 给出 PAC-Bayes 解释。

**[Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](llm_pretraining/dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)**

:   作者把 dropout 看作平均场信号传播理论中破坏 $c^*=1$ 完美对齐不动点的"外场" $h$，推出 Landau 方程、两参数标度坍塌以及 smooth/kinked 激活的两个不同普适类，并由此得到一个"零开销"的实用结论——**前置 dropout（front-loaded schedule）**在同等预算下比常数 dropout 在 MLP 和 ViT 上把测试损失降低 18–35%。

**[FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](llm_pretraining/flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment.md)**

:   FlexRank 把预训练大模型的每个线性层做 activation-aware 低秩分解（DataSVD），用动态规划在 $O(L\cdot K)$ 时间内挑出一组**严格嵌套**的子模型对应不同算力预算，再用知识蒸馏联合训练这套共享权重，最后通过 Gauge-Aligned Reparametrization 把秩节省真正翻译成 FLOPs 节省——一次训练即可在 LLM 与 ViT 上得到逼近真实帕累托前沿的"一族"可部署模型。

**[Focus and Dilution: The Multi-stage Learning Process of Attention](llm_pretraining/focus_and_dilution_the_multi-stage_learning_process_of_attention.md)**

:   本文在单层 Transformer 学习马尔可夫数据的简化场景下，通过围绕一系列临界点做分阶段线性化的梯度流分析，揭示并严格刻画了注意力训练中反复出现的「聚焦—稀释」循环，并在 WikiText 与 TinyStories 上观察到一致的现象。

**[If open source is to win, it must go public](llm_pretraining/if_open_source_is_to_win_it_must_go_public.md)**

:   这是一篇 ICML 2026 立场论文（position paper），论点是：当前形态的"开源 AI"无法像 Linux/PyTorch 那样真正民主化 AI 访问与提供公共产品，必须嵌入到"公共 AI（Public AI）"——由政府/国家实验室/大学/非营利机构提供的算力、推理、后训练、数据基础设施——之中，开源才能赢。

**[Incremental BPE Tokenization](llm_pretraining/incremental_bpe_tokenization.md)**

:   本文提出首个具有严格 $\mathcal{O}(\log^2 t)$ 单字节最坏复杂度的增量 BPE 分词算法，通过 Aho–Corasick 自动机定位搜索空间、Centroid Decomposition 上的二分搜索定位"最后一个 token"，作为 drop-in replacement 相对 Hugging Face tokenizers 最高 $\sim 3\times$ 加速，并在病态输入上消除了 tiktoken 的 $\mathcal{O}(n^2)$ 退化。

**[InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)**

:   作者提出 InfoLaw：把"预训练"重新定义为"按桶累积信息"的过程，每桶信息量等于"质量密度 $f_d$ × 唯一 token 数 $M_d$ × $\log K$"再乘上一个随重复次数 $R_d$ 指数衰减的因子，最终把验证损失写成 $L = \alpha\cdot\text{info}^{-\beta}$，能在 252M-1.2B 拟合后外推到 7B / 425B token，平均误差 0.15%、最大 0.96%，并直接用来搜索最优数据配方。

**[Inverse Depth Scaling From Most Layers Being Similar](llm_pretraining/inverse_depth_scaling_from_most_layers_being_similar.md)**

:   本文通过对 LLM 隐藏态动力学的测量 + teacher-student toy model 的对照实验，证明 LLM 的 loss 与深度近似成反比（$\alpha_\ell \approx 1$），并将其归因于"绝大多数层在做功能相似的小步更新、通过 ensemble averaging 抵消误差"这一非高效但鲁棒的使用模式。

**[MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier](llm_pretraining/moose-star_unlocking_tractable_training_for_scientific_discovery_by_breaking_the.md)**

:   MOOSE-Star 把"训练一个能直接生成科学假设的 LLM"这个原本要在 $\mathcal{O}(N^k)$ 组合空间里搜索的问题拆成"灵感检索 + 假设合成"两个序列子任务，再叠上层级树检索 + bounded composition + motivation 规划，把最优复杂度从指数级压到 $\mathcal{O}(\log N)$，并放出 108,717 篇带分解标注的 TOMATO-Star 数据集。

**[Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](llm_pretraining/names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)**

:   作者把 Transformer 改成"对每个可互换符号开一条共享权重的并行嵌入流 + 跨流聚合注意力"的结构，从架构层面保证对变量重命名（alpha 等价）的输出完全不变，并且允许测试期向词表里塞训练时没见过的新符号，在命题逻辑与 LTL 见证生成任务上超过同类基线甚至 GPT-5.2。

**[On the Expressive Power of Permutation-Equivariant Weight-Space Networks](llm_pretraining/on_the_expressive_power_of_permutation-equivariant_weight-space_networks.md)**

:   本文为操作在 MLP 权重上的置换等变 weight-space 网络（DWS / NFN / GMN / NG-GNN 等）建立了首个系统的表达力理论，证明这些架构在表达力上几乎完全等价，并在"general position"假设下对四种逼近场景（函数空间泛函/算子、置换不变泛函、置换等变算子）给出了普适性刻画；由理论得出的简单修改 OCE（输出端 ensemble 多个 MLP）在 INR 编辑基准上相对 SOTA 提升 34%。

**[On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](llm_pretraining/on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)**

:   本文用一套精心控制"推理难度恒定、只变 horizon 长度"的 Sudoku/Rush Hour 任务，系统证明**任务 horizon 本身就是 LLM agent RL 训练崩溃的独立根因**，并提出 macro action 与 subgoal decomposition 两种 horizon-reduction 机制——它们不仅稳住训练，还让模型在更长 horizon 上实现强 zero-shot 泛化（horizon generalization）。

**[Predicting Large Model Test Losses with a Noisy Quadratic System](llm_pretraining/predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)**

:   本文提出 Noisy Quadratic System (NQS)——一个把 LLM 测试损失建模为 $L(N, B, K)$（模型大小 / 批大小 / 更新步数）的 mechanistic 损失模型，首次在 scaling law 中显式建模 batch size，并在 Pythia + OWT2 上把外推预测能力从 Chinchilla 的 ~20× 算力提升到 ~4000× 算力。

**[Scaling Depth Capacity via Zero/One-Layer Model Expansion](llm_pretraining/scaling_depth_capacity_via_zeroone-layer_model_expansion.md)**

:   本文提出"零层/一层渐进式训练"——先训一个几乎没有 Transformer 层的极浅模型，再在训练后期（≈80% iterations）一次性把深度扩展到目标层数，配合 WSD 学习率和 muP 超参传递，可在 GPT2/LLAMA3/DeepSeekV3 上节省约 80% 计算（≈5× 加速）且最终 loss 几乎不掉。

**[SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs](llm_pretraining/spare_stacked_parallelism_with_adaptive_reordering_for_fault-tolerant_llm_pretra.md)**

:   SPARe 在数据并行维度把同一份数据 shard 跨组 cyclically 堆叠 $r$ 层，并在节点失败后用 Hopcroft-Karp + min-cost max-flow 自适应重排"all-reduce stack 数"，使得在 600k GPU 的 restart-dominant 场景下，只需 $2\sim 3\times$ 计算开销就能达到与 $r\times$ 传统副本同等的可用性，把 time-to-train 相比 Rep+CKPT 进一步降 $40\sim 50\%$。

**[The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](llm_pretraining/the_devil_is_in_the_condition_numbers_why_is_glu_better_than_non-glu_structure.md)**

:   在 NTK 视角下证明 GLU 把两层网络的核矩阵改写成"原 NTK 与数据 Gram 阵的 Hadamard 积"，从而显著压缩条件数、加速收敛，同时实证显示 GLU 并不改善泛化间隔，其全部红利都来自更好的优化。

**[Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher](llm_pretraining/trust_functions_near-lossless_weak-to-strong_generalization_by_learning_when_to_.md)**

:   本文把"弱到强泛化（Weak-to-Strong Generalization）"重新框架成一个**数据选择**问题，提出"信任函数（Trust Function）"用一个轻量 MLP 读取教师模型最后一层隐藏状态、预测弱标签是否可靠，然后只挑高信任样本去训练强学生，从而在多任务上实现近无损甚至超越 ground-truth 的监督效果，并可迭代成"弱到强链"放大收益。

**[Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](llm_pretraining/tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)**

:   本文用 $k$-parity 这一可解析任务把 Masked Diffusion Language Model（MDLM）的训练目标解构成"信号项 + 噪声项"，从理论上证明噪声项扮演**隐式正则器**抑制 grokking、避开记忆陷阱，并据此提出 **Signal-Rich Mask Sampling**——把训练时的掩码率 $t$ 从均匀 $\mathcal{U}[0,1]$ 收紧到中段窗口，在 50M 模型上显著降 perplexity、在 8B 模型上预训练提升 8.8%、SFT 提升 5.8%。

**[XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](llm_pretraining/xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)**

:   XTransfer 面向边缘设备上的人体感知任务，用少量目标传感器数据把来自图像、文本、音频或传感器等任意模态的预训练模型迁移过来，通过 layer-wise model repairing 和 resource-constrained layer recombining 缓解跨模态特征错位，同时提升少样本精度与边缘部署效率。

---

## 📹 视频理解 { #video_understanding }

**[AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](video_understanding/avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)**

:   提出 AVTrack 数据集和 AVTracker 基线方法，针对复杂人体中心场景下的音视频实例分割与跟踪（AVIS）任务，通过定义 8 种挑战条件构建高难度评测基准，并设计三阶段局部-全局分治框架（ASR 分段聚合 → 局部说话人定位 → 全局身份关联），在 HOTA 指标上超越现有最优方法约 8 个百分点。

**[Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning](video_understanding/foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni.md)**

:   Foresee-to-Ground (F2G) 把视频时序定位（VTG）从直接时间戳回归重构为「识别-测量」两阶段问题——先用预测性时序感知 + 跨度证据编码器构建候选事件证据池，再用 LLM 在选中事件的约束下精确生成边界，使 Charades-STA R@0.7 提升 4.1 个点、ActivityNet 提升 6.7 个点。

**[MetaphorVU: Towards Metaphorical Video Understanding](video_understanding/metaphorvu_towards_metaphorical_video_understanding.md)**

:   本文提出首个隐喻视频理解基准 MetaphorVU-Bench（860 视频 + 8 类隐喻分类法）和增强方法 MetaphorBoost——通过 54K 节点 / 200K 边的隐喻知识图谱作为外部认知支架，定量揭示 MLLM 在隐喻视频上的核心瓶颈是"跨域映射缺失"而非视觉识别错误，最优模型相比人类（83.4）仍差 17 个点。

**[OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](video_understanding/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)**

:   本文指出现有 Omni-LLM token 压缩方法对音频和视频"对称"处理是次优的，提出 OmniSIFT——先用时空显著性剪掉视频冗余得到"视觉锚点"，再用这些锚点引导音频选择的两阶段非对称压缩框架，仅引入 4.85M 额外参数就在 Qwen2.5-Omni-7B 上保留 25% token 时一致超过现有压缩基线甚至原模型。

**[Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection](video_understanding/privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection.md)**

:   作者提出 OPL（Orthogonal Projection Layer）和加强版 G-OPL，用一个 QR 分解出来的可学习正交子空间，在视频异常检测特征空间中显式投影掉"任务无关变量"和"人脸隐私分量"，同时引入 SSC/ARD/PD/FPD 四个隐私感知指标，在保持/提升 VAD AUC 的前提下让线性 SVM 探针对面部预测的准确率显著下降。

**[ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions](video_understanding/proact-vl_a_proactive_videollm_for_real-time_ai_companions.md)**

:   ProAct-VL 通过分块输入-输出范式 + 轻量级 FLAG 决策头 + 过渡感知损失函数，使视频大语言模型在流式输入下能自主决定**何时响应**并生成短片段评论，同时实现 ~1 秒低延迟与强主动性——在游戏解说任务上响应时机 TimeDiff 仅 1.20 秒、触发 F1 = 63.25%，全面超越 GPT-4o 等离线模型。

**[RELO: Reinforcement Learning to Localize for Visual Object Tracking](video_understanding/relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)**

:   RELO 把视觉单目标跟踪中"哪里是目标"这件事重构成一个空间特征图上的 MDP,把每个空间位置当作 action,用 actor-critic + IoU/AUC 直接奖励替换掉传统的手工中心热图监督,并配合"先 warmup 回归 + 层对齐时序 token 传播"两个稳定化设计,在 LaSOText 上以 57.5% AUC 拿到 SOTA。

**[Return of Frustratingly Easy Unsupervised Video Domain Adaptation](video_understanding/return_of_frustratingly_easy_unsupervised_video_domain_adaptation.md)**

:   本文提出 MetaTrans——一个"令人沮丧地简单"的无监督视频域适应（UVDA）方法，通过双流 Transformer 的时空特征相减来解耦空间和时间域差异，仅用两个基础损失（监督 + 域对抗）即可超过 SOTA 复杂方法，并把超参搜索成本从指数级压到线性级。

**[Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval](video_understanding/revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri.md)**

:   本文针对 Partially Relevant Video Retrieval (PRVR) 中"短查询 vs 长视频"导致的查询歧义与时间稀疏监督问题，提出基于 Dirichlet 分布的层次证据学习框架 Holmes，在视频间用三重原则区分精确/多义/欠定查询并自适应校准标签，在视频内用带 dustbin 的柔性最优传输获得稠密对齐，在 ActivityNet/Charades/TVR 三个数据集上取得 SOTA。

**[SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition](video_understanding/skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o.md)**

:   SkelHCC 把 CLIP 搬到 Hyperbolic 空间，显式按"关节 → 身体部分 → 全身"三粒度对齐骨骼-语言表示，并用 LLM 生成的身体部位重要性掩码做无训练的多粒度投票缓存推理，在 NTU120 单样本动作识别上比 SOTA 提升 9%，可训参数只有 0.5M。

**[SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling](video_understanding/slap_the_semantic_least_action_principle_for_variational_video-language_modeling.md)**

:   SLAP 把"经典力学的最小作用量原理"搬到视频语义流形上，把稀疏采样视频的缺帧补全建模为 Riemannian 流形上的两点边界值问题——用语义动力学替代概率生成来强制物体持久性，在隧道遮挡测试上准确率 83.9%（超扩散模型 12 个点）且推理加速 177×。

**[STORM: Segment, Track, and Object Re-Localization from a Single Image](video_understanding/storm_segment_track_and_object_re-localization_from_a_single_image.md)**

:   STORM 提出"一张参考图就能跑"的 6D 位姿跟踪框架：用层级化空间融合注意力 HSFA 做参考-查询特征对齐（产出分割掩膜 + SAM3D 网格），再训一个 BCE 二分类的 Tracking Verifier，把其 logit 取负当作能量分数 $E=-g_\theta$，连续 $L=3$ 帧超阈值就触发自动重定位，从而在 LM-O / YCB-V 上把无标注 6D 跟踪精度推到接近 ground-truth 掩膜上限。

**[Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](video_understanding/unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)**

:   OneTrackerV2 把 RGB / RGB+D / RGB+T / RGB+E / RGB+N 五种跟踪任务统一在一个网络里端到端训练，靠 Meta Merger 做模态融合、Dual MoE 把"时空匹配"与"模态融合"两类异质特征显式拆到 T-MoE 与 M-MoE，并用 dissimilarity loss + router clustering 保证它们不塌成同一子空间。

**[Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)**

:   Video-MTR 是一个基于强化学习的**多轮推理**框架——通过**门控双层奖励机制**引导 MLLM 迭代选择关键视频片段，仅用 **8K 数据**实现长视频理解的 SOTA 性能，对标方法需要 257K~440 万样本（数据效率提升两个数量级）。

**[VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](video_understanding/videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)**

:   VideoSEAL 发现现有 agentic 长视频 QA 系统存在「答对但没看到证据」的失配问题，并把根因归结为「coupled agent 把规划和回答权混在一起」，提出 planner-inspector 解耦框架：planner 负责长视距证据搜寻、inspector 持有独占回答权并在像素级证据充分时才放行，在 LVBench 上把准确率从 48.2% 拉到 55.1%（↑20.5%）且 LongVideoBench 从 52.2% 升至 62.0%。

**[VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking](video_understanding/videotemp-o3_harmonizing_temporal_grounding_and_video_understanding_in_agentic_t.md)**

:   VideoTemp-o3 是统一的 Agent 视频理解框架——通过**冷启动 SFT 的统一掩码策略** + **可感知奖惩的 IoU 奖励**联合建模视频时间定位与问答，在长视频理解中实现高质量的多轮迭代定位与精准回答，超长视频（> 20 分钟）mIoU 15.6% 超过 Gemini-2.5-Pro 的 14.8%。

**[VSCD：无对齐场景的视频场景变化检测](video_understanding/vscd_video-based_scene_change_detection_in_unaligned_scenes.md)**

:   本文引入 VSCD 任务——通过查询中心的多参考模型，在无约束相机运动和强烈视点失配条件下，利用时间一致性、补丁级对应和置信度加权融合，逐像素检测两段不同时间记录的同一环境视频中的物体级变化。

---

## 🔗 因果推理 { #causal_inference }

**[An Odd Estimator for Shapley Values](causal_inference/an_odd_estimator_for_shapley_values.md)**

:   这篇论文证明 Shapley value 只依赖集合函数的 odd component，并据此提出 OddSHAP：用配对采样隔离 odd 信号、用 GBT 筛选高阶 odd Fourier 交互、再做稀疏 odd 回归，在中高维解释任务上显著优于灵活预算 Shapley 估计器。

**[Causal-JEPA: Learning World Models through Object-Level Latent Masking](causal_inference/causal-jepa_learning_world_models_through_object-level_latent_masking.md)**

:   提出 C-JEPA，将 JEPA 的掩码预测从图像 patch 级别扩展到对象级别潜在表示，通过对象级掩码作为潜在干预迫使模型学习交互依赖的动态，在反事实推理上比无掩码基线提升约 20%，在控制任务中仅用 1% 的 token 即达到可比性能且规划加速 8 倍以上。

**[Controllable Generative Sandbox for Causal Inference](causal_inference/controllable_generative_sandbox_for_causal_inference.md)**

:   本文提出 CausalMix：一个变分生成框架，把数据类型特定的 multi-head decoder + Bayesian Gaussian 混合潜在 prior 与三类可独立调控的因果"旋钮"（overlap $\alpha(X)$、CATE 函数 $\tau(X)$、未观测混杂 $\kappa(X,T)$）联合优化，从而在保持真实数据分布 fidelity 的前提下让用户自由设计 counterfactual benchmark，在 mCRPC（前列腺癌）真实病例上验证 CausalMix 既能高保真复现 mixed-type 表格，又能稳定地按需注入 overlap / confounding / 异质效应，用作 CATE 估计器的可控 stress test。

**[Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](causal_inference/density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)**

:   DensityFlow 把"在模型多重性下生成鲁棒反事实解释 (RCE)"重新表述为带密度约束的最优传输问题，用 NCE 训练一个 (K+1) 类判别器同时学分类与类条件密度，再用 Neural ODE 把查询样本沿密度梯度运到目标类高密度流形上，并在黑盒场景下只对生成轨迹做局部蒸馏对齐，从而以远低于集成基线的查询量取得更高的跨模型 validity。

**[ECSEL: Explainable Classification via Signomial Equation Learning](causal_inference/ecsel_explainable_classification_via_signomial_equation_learning.md)**

:   ECSEL 把"每个类别一个 signomial（带实数指数的幂律和）函数 + softmax"作为分类器，配合 L1 稀疏正则与多阶段优化，既能在 AI Feynman 等符号回归 benchmark 上以远低于 SOTA 的算力恢复 95.86% 的目标方程，又能在 11 个分类数据集上与 XGBoost/MLP 打平，同时所有特征归因都由模型参数闭式给出。

**[Evaluating Bivariate Causal Statements Based on Mutual Compatibility](causal_inference/evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)**

:   本文针对"只有成对(bivariate)因果陈述、没有 ground truth"的场景，提出两个无需 faithfulness 的相容性评分（线性情形的 `comp` + 图结构情形的 `incomp`），通过判断这些两两陈述拼起来的多元模型是否需要"反常的额外混淆"来解释观测协方差，从而识别错误的因果论断，并用它给 LLM 的因果输出打分。

**[Formalizing and Falsifying Causal Pathways of Rare Events](causal_inference/formalizing_and_falsifying_causal_pathways_of_rare_events.md)**

:   本文把罕见事件的"口头因果解释"形式化为 **causal pathway**——一个由二值化事件构成的子图，并定义 **pathway explanation score** 来量化"根因 + 中介通路"对目标事件的解释力，得到一套可证伪的因果解释评价框架。

**[Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](causal_inference/harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)**

:   本文针对大推理模型（LRM）的幻觉检测提出 ARS：不在文本层扰动 reasoning trace，而是**直接在 trace 末端的潜表示上施加小扰动并续解码**得到反事实答案，再用"答案是否一致"作为标签训一个轻量 contrastive 头来塑形 trace-conditioned answer embedding，使后续 embedding-based detector 把幻觉与真实回答分得更开（TruthfulQA 上 AUROC $66.85\to 86.64$）。

**[Investigating Memory in Model-Free RL with POPGym Arcade](causal_inference/investigating_memory_in_model-free_rl_with_popgym_arcade.md)**

:   本文指出仅用回报来比较 RL 记忆模型并不可靠，作者构建了一个 GPU 加速的 MDP/POMDP "孪生"基准 POPGym Arcade，并提出 Observability Gap、Memory Bias、像素显著性和 Recall Density 四个工具，借此揭示了一种"价值涂抹（value smearing）"病理：记忆模型会把价值信用错误地分摊到无关的历史观测上，进而导致单个 OOD 观测就能通过 recurrent state 长期污染策略。

**[Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression](causal_inference/outcome-aware_spectral_feature_learning_for_instrumental_variable_regression.md)**

:   针对非参数工具变量（NPIV）回归中 SpecIV 学到的谱特征"只看 X-Z 关系、不看结果 Y"的盲点，本文提出 Augmented Spectral Feature Learning：在 SpecIV 的对比损失里加上一项 Y 投影到 Z 特征上的回归损失，等价于对一个把 Y 信息拼进去的"增广算子" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$ 做截断 SVD，从而在结构函数 $h_0$ 与 $\mathcal{T}$ 顶端奇异函数对齐很差的"坏"情形下也能用极低秩特征恢复因果效应。

**[Rank-Learner: Orthogonal Ranking of Treatment Effects](causal_inference/rank-learner_orthogonal_ranking_of_treatment_effects.md)**

:   在观测数据上提出 Rank-Learner——第一个 Neyman-正交的两阶段处理效应**排序**学习器，用成对软标签 + 双重稳健修正项替代"先估 CATE 再排"的间接做法，在合成、半合成与 Criteo uplift 真实数据集上稳定优于 T/DR-learner 与非正交 plug-in ranker。

**[Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](causal_inference/tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)**

:   本文提出一个通用框架：通过让训练损失的局部二阶曲率 $w_\ell(p)$ 匹配下游任务误差的曲率 $w_{\text{task}}(p)$，可派生出与下游任务"几何对齐"的严格 proper scoring rule；将其应用到 IPW 估计 ATE，得到闭式损失 + 闭式 canonical 激活函数（解一个四次方程），在 IHDP / Jobs / Kang-Schafer / ACIC 2017 上稳定优于 log-loss 与 covariate balancing 类基线。

**[The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](causal_inference/the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)**

:   本文把搜索广告的真实价值建模为"赢拍 vs 输拍"的 treatment effect，在重复二价拍卖（SPA）binary 反馈下设计了一个利用支付规则的在线因果学习算法，得到 $\widetilde\Theta(\sqrt{dT})$ 的极小极大最优 regret，比同设定下的一价拍卖严格更易学。

**[The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](causal_inference/the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)**

:   本文构造了一个程序化生成的"合成 Web"环境,通过在搜索 rank 0 注入单条高可信度蜜罐误信息,因果性地测出 GPT-5 等前沿 LLM agent 在 1/数千的对抗污染下准确率从 65% 暴跌到 18%,且模型不会增加搜索、依然高置信度作答,揭示了根深蒂固的"位置锚定"失败模式。

**[Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](causal_inference/towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)**

:   本文给出一个统一的"分布类"框架，刻画了在选择偏差下平均处理效应 (ATE) 全人群可识别的充要条件 (Condition 1)，并证明在 c-overlap 倾向得分 + 多项式指数族 / Gaussian / Laplace / Pareto / Log-normal 等常见分布下都满足该条件，配套提出 MLE 与 Score Matching 两种带选择函数 $\beta(x,y,t)$ 校正的估计器，在合成与 All of Us 半合成实验上显著优于 IPW 与多项式回归。

**[Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](causal_inference/unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)**

:   通过引入推导图（derivation graphs）显式表示 do-演算规则的所有等价变换——揭示因果表达式空间的结构，并证明最多 4 步规则应用可达任意等价表达式。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](image_restoration/coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)**

:   本文从表达力与可训练性两个维度系统比较连续扩散、离散掩码扩散、looped transformer，证明"连续扩散"在表达力上严格强于离散扩散并能模拟 looped transformer，但实际性能受限于解码与表征空间；据此提出 **CCDD（Coevolutionary Continuous Discrete Diffusion）**——在离散 token 空间和预训练 LLM 的上下文嵌入空间上同时扩散，由单一模型联合去噪，在 LM1B/OWT 上比 MDLM 困惑度降 25-35%，并以仅 8 步采样超过 MDLM 256 步效果。

**[Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution](image_restoration/coloring_the_noise_adversarial_sobolev_alignment_for_faithful_image_super_resolu.md)**

:   ASASR 通过将 Flow Matching 的噪声先验从各向同性高斯替换为 Sobolev 谱着色噪声，结合对抗性流形引导生成硬负样本，构建 AS-DPO 框架，实现了超分辨率中感知质量与结构保真度的最优平衡。

**[Consistent Diffusion Language Models](image_restoration/consistent_diffusion_language_models.md)**

:   本文指出离散扩散没有连续域 probability-flow ODE 的对应物，因此无法直接做 consistency model；作者提出用**精确闭式 posterior bridge** 作为离散域的"随机版 PF-ODE 替代品"，构造 Multi-Path Discrete Consistency (MPDC) 训练目标，要求 denoiser 在多条 stochastic bridge 路径上的预测在期望上一致，从而单阶段、teacher-free 地训出可在 2-3 步生成高质量文本的 Consistent Diffusion Language Model (CDLM)，在 unconditional / conditional 文本生成上达到 SOTA、对 AR 模型最高 $32\times$ 加速。

**[DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](image_restoration/dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)**

:   DAPD 把 dLLM 单步并行解掩问题转化为「在自注意力诱导的 MRF 上选独立集」的动态图着色问题，无需训练即可同时解掩弱依赖位置，在 LLaDA / Dream 上把多问题混合提示的解码步数压到原始的 1/3.87，且准确率几乎不掉。

**[Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](image_restoration/degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)**

:   DAMP 用 6 个可解释的空间-光谱物理度量（高频能量比/纹理一致性/光谱曲率等）作为"退化提示" (DP) 代替黑盒嵌入与显式退化标签，再用 DP 作为门控驱动 Spatial-Spectral Adaptive MoE 选择不同的"空间专家/光谱专家"，在 5 种 HSI 恢复任务和 2 种未见退化（运动模糊、Poisson 噪声）上同时取得 SOTA。

**[DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](image_restoration/dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)**

:   DyLLM 是一种 training-free 的扩散 LLM 推理加速框架，利用相邻去噪步骤之间注意力上下文的余弦相似度识别"显著 token"，只对这部分 token 重算 FFN 和注意力，配合显著感知的近似注意力，在 LLaDA / Dream 上把吞吐推到 7.6× / 9.6× 而几乎不掉点。

**[Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](image_restoration/early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)**

:   本文系统刻画了 masked 扩散语言模型 (dLLM) 在**完全非自回归 (NAR) 解码**下的失败机制——proximity bias 导致 confidence-based 采样退化为反向自回归并被 EOS 过早占满，再用一个 5M 参数的轻量 planner + EOS 温度退火**只在第一步**干预 unmasking 位置，就在 GSM8K 等推理任务上将 LLaDA 8B 的 NAR 解码平均提升 2.8–4.3 个点而几乎无额外开销。

**[Learning Normalized Energy Models for Linear Inverse Problems](image_restoration/learning_normalized_energy_models_for_linear_inverse_problems.md)**

:   作者把"线性逆问题"重写为"各向异性去噪"，并提出 Anisotropic Covariance Score Matching (A-CSM) 训出一个**归一化**的能量模型 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$，单个模型即可处理 inpainting、deblurring、super-resolution，并解锁能量引导自适应调度、MALA 无偏校正和盲逆问题三大新能力。

**[One-shot Conditional Sampling: MMD meets Nearest Neighbors](image_restoration/one-shot_conditional_sampling_mmd_meets_nearest_neighbors.md)**

:   CGMMD 用 k 近邻图把"期望条件 MMD（ECMMD）"估计成一个可直接最小化的非对抗目标，训出一个能在单次前向传播内从 $P_{Y\mid X}$ 采样的条件生成器，并给出了非渐近误差界与分布收敛性证明。

**[Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](image_restoration/phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)**

:   为单次曝光式压缩光谱成像 (CASSI) 设计一个 train-render 两阶段、按波长可任意查询的深度展开框架——在每个展开 stage 内塞入连续光谱场 (CoSF) 先验模块，由 Fourier-Mamba 驱动的三分支跨域特征混合器 + 随机频率编码 + 谱合成头组成，离散波长训练即可在推理时合成任意连续波长的高光谱图像，实现连续光谱重建与零样本光谱超分。

**[Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](image_restoration/plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)**

:   本文提出 Dilated Unmasking Scheduler (DUS)：用「等距空隙」预定义不依赖模型置信度的 unmask 顺序，把每块 $B$ 个 token 的 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$，在 LLaDA / Dream / DiffuCoder 上拿到 5.8× wall-clock 加速且质量优于基于置信度的并行 planner。

**[PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](image_restoration/podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)**

:   PODiff 把扩散模型从像素空间搬到固定的、按方差排序的 POD 系数空间里跑，用极小的 MLP 就能在 $640\times 480$ SST 降尺度任务上拿到与像素级扩散相当的精度，同时因为重构是线性的，集成方差可以通过 $\Sigma_u=\Phi\Sigma_a\Phi^\top$ 解析回传到物理空间，得到空间上可解释、且校准良好的不确定性。

**[Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](image_restoration/semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)**

:   SuperMeshNet 用两个互补 MPNN——主模型预测 LR→HR，辅助模型预测 LR-LR 对应的 HR-HR 差分——在无配对 HR 的样本上互相生成伪标签，并配合节点级 / 消息级 centering 两个轻量归纳偏置，使得 PDE mesh 超分仅用 10% HR 数据就能超过 100% HR 全监督基线，跨 6 种 MPNN 架构一致下降 RMSE。

**[Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](image_restoration/structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)**

:   SDB 把模态翻译重写为"在所有满足边缘约束的耦合集合 $\mathcal{P}$ 中挑一个"，在 LDDBM 之上叠加边缘匹配（WTA + 容量约束）+ 端点级 + 轨迹级双层 cycle consistency，把成对监督仅作为可选启发式之一，从而在零成对、半成对、全成对三种监督预算下都能跑，并且全成对时也比 paired-only 基线更好（FFHQ→CelebA-HQ PSNR 从 25.6 提到 25.9）。

**[Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](image_restoration/triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)**

:   本文把扩散后验采样中长期被当作常数的三个力——数据一致性 (DC) 引导、Classifier-Free Guidance (CFG)、随机性 (stochasticity)——首次系统地视为一个**耦合的时变三体系统**，理论 + 实证证明早期 CFG 与 DC 方向冲突、而随机性能把轨迹拉回高概率流形，据此提出"DC↓、CFG↑、η↓"的单调三体调度趋势，并用"模板搜索 + GRPO 强化学习"两套方法找最优曲线，在 FFHQ / DIV2K 的超分与去模糊上同时刷新失真和感知指标。

**[UOTIP：无须配对的反演问题的非平衡最优传输映射](image_restoration/uotip_unbalanced_optimal_transport_map_for_unpaired_inverse_problems.md)**

:   提出 UOTIP 方法——通过非平衡最优传输（UOT）框架将无须配对的图像反演问题表述为从有噪声测量分布到干净信号分布的映射学习，通过引入似然成本函数和二次项成本获得鲁棒性和理论保证。

---

## 💻 代码智能 { #code_intelligence }

**[AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms](code_intelligence/algoveri_an_aligned_benchmark_for_verified_code_generation_on_classical_algorith.md)**

:   AlgoVeri 构建了一个跨 Dafny、Verus、Lean 严格对齐的经典算法 verified code generation 基准，显示当前 LLM 在复杂全局不变量、系统级约束和显式证明搜索上仍有巨大缺口，尤其是 Lean 与 Verus 的成功率远低于 Dafny。

**[BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)**

:   BoostAPR 给"用 RL 训 program-repair 模型"造了一套三阶段流水线——execution-verified SFT → 训序列级 + 行级双重 reward → PPO 时用行级模型把序列奖励重新分配到关键 edit lines；在 Qwen2.5-Coder-32B 上把 SWE-bench Verified 从 17.8% 推到 40.7% (+22.9pp)，跨语言迁移到 Defects4J 取 24.8%。

**[CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](code_intelligence/centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding.md)**

:   提出 CentaurEval，首个面向人机协作编程的统一评测框架，通过设计 45 个"协作必需"(Collaboration-Necessary) 任务模板，证明单独 LLM 仅 0.67% 通过率、人类独立仅 18.89%，而人机协作可达 31.11%，揭示 LLM 正从执行工具演变为共推理伙伴。

**[Entropy-informed Decoding: Adaptive Information-Driven Branching](code_intelligence/entropy-informed_decoding_adaptive_information-driven_branching.md)**

:   EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search；理论上证明熵单调的分支因子在期望累计 regret 上严格优于任何固定束宽，且能给出 $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$ 的显式 regret 率。

**[HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](code_intelligence/he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)**

:   在 SWE-bench 上传统 PPL 既受"长上下文税"干扰又无法预测 SFT 后的智能体能力，本文提出"熵压缩假说"和 HE-SNR 指标，只在 Top-10 熵大于 $(\ln 3 + \ln 4)/2$ 的"高熵决策点"上算信号噪声比，与下游 SWE-bench 得分的 Pearson 相关达 0.96，Kendall 一致性 0.98。

**[Locally Coherent Parallel Decoding in Diffusion Language Models](code_intelligence/locally_coherent_parallel_decoding_in_diffusion_language_models.md)**

:   本文提出 CoDiLA，在 masked 扩散语言模型（DLM）外挂一个轻量自回归（AR）小模型，用"软嵌入"接收 DLM 的边缘分布并在小块内做局部自回归解码，从而在保留 DLM 全局双向能力的同时消除并行采样产生的局部不连贯问题，在代码生成上以 ≥2× 吞吐建立新的 Pareto 前沿。

**[MARS: Modular Agent with Reflective Search for Automated AI Research](code_intelligence/mars_modular_agent_with_reflective_search_for_automated_ai_research.md)**

:   MARS 把自动化 AI 研究重构成"在软件仓库空间中搜索最优解"的问题，用 **预算感知 MCTS + 模块化"设计-分解-实现"流水线 + 比较式反思记忆** 三根支柱，在 MLE-Bench 上拿到开源框架 SOTA，金牌率 31.1%（Gemini-3-Pro-Preview），并出现 63% 的跨分支课程迁移这种"Aha! moment"。

**[MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](code_intelligence/matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)**

:   MatchFixAgent 把仓库级代码翻译的"等价性验证 + 修复"全面 LLM 化：用 6 个并行语义子分析器（控制流 / 数据流 / IO / 库 API / 异常 / 规约）替代昂贵的跨语言互操作工程，再叠加一个测试生成 & 修复 Agent 和一个仲裁 Agent，仅 1650 行代码就把验证覆盖率从 71.6% 抬到 99.2%，可修复缺陷比例从 18.5% 抬到 50.6%。

**[NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](code_intelligence/nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)**

:   NEMO 把自治编码代理 (Autonomous Coding Agent, ACA) 当作和 LLM 同级的"一等抽象"来调用，让独立生成的模拟器和优化器在共享沙箱里通过执行结果互相校验，再叠加多样性记忆检索与 MBR/自一致性解码，在 9 个优化建模基准上 8 个拿到 SOTA、最高领先 28 个百分点。

**[Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software](code_intelligence/physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s.md)**

:   作者以"一位物理学家用 Claude Code 在 12 天 57 个会话里开发 ~2,100 行可微分宇宙学微扰理论代码 clax-pt"为单例（$N=1$）案例，量化记录了 15 次督导事件，证明在科学软件场景下决定产物可信度的不是模型能力，而是围绕 oracle 测试、共享变更日志、"禁打补丁"等规则搭建的人工监督协议。

**[Poison with Style: A Practical Poisoning Attack on Code Large Language Models](code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)**

:   PwS 用开发者常用的 Python 代码风格（如 Yapf/Black/PEP8）作为隐式触发器对开源 Code LLM 进行投毒，让模型在格式化器自动整理代码后才生成带 CWE 漏洞的补全；在 Qwen2.5-Coder-32B 上对 CWE-20 触发提示达 95% ASR，而 HumanEval/MBPP pass@1 仅掉约 5%，并能抗住 BEEAR、prefix tuning、CodeShield 等主流防御。

**[Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning](code_intelligence/probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning.md)**

:   RankTuner 提出 Relative Rank Indicator $I_t$，用「真值 token 的实际排名 $R_t$」对比「模型分布下的期望排名 $\mathbb{E}[R_t]$」作为单一标量信号，把概率 $p_t$（任务对齐）和熵 $H_t$（内禀不确定性）拧成一个 token 级权重，在数学推理 SFT 上 Pass@1 普遍超过纯概率/纯熵的重加权 baseline。

**[Pull Requests as a Training Signal for Repo-Level Code Editing](code_intelligence/pull_requests_as_a_training_signal_for_repo-level_code_editing.md)**

:   本文提出 Clean-PR 中训练范式，把 1640 万条带噪声的 GitHub Pull Request 经过过滤、重建和回放验证转成 200 万条可执行的 Search/Replace 编辑块语料，再叠加 Agentless 对齐 SFT 与错误驱动数据增强，使 Qwen2.5-Coder-32B 在 SWE-bench Lite/Verified 上分别相对 baseline 提升 13.6% 和 12.3%，并以 32B 参数超越 72B 的 Lingma-SWE 与 SWE-Fixer。

**[SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](code_intelligence/swe-rebench_v2_language-agnostic_swe_task_collection_at_scale.md)**

:   作者用"语言无关的统一构造流水线 + 交互式安装 Agent + 三模型集成的 Issue 清晰度过滤"，从 GitHub 上自动挖掘出 32,079 个跨 20 种语言、3,617 个仓库的可执行 SWE 任务（并附 12 万+ PR 衍生任务），每个任务都带预构建 Docker 镜像、fail-to-pass 测试以及实例级诊断元数据，为 SWE Agent 的大规模强化学习提供面向训练的、而非面向评测的稳定底料。

**[UniRTL: 统一代码与图实现鲁棒 RTL 表示学习](code_intelligence/unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)**

:   本文提出 UniRTL——通过联合学习 RTL 代码和控制数据流图（CDFG）的多模态统一表示，采用图感知分词器和分层训练策略，在硬件性能预测和代码检索任务上显著超越现有方法。

---

## 👥 多智能体 { #multi_agent }

**[Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information](multi_agent/beyond_majority_voting_llm_aggregation_by_leveraging_higher-order_information.md)**

:   本文提出两种利用高阶信息的 LLM 回答聚合算法——基于一阶准确率信息的 Optimal Weight (OW) 和基于二阶相关性信息的 Inverse Surprising Popularity (ISP)，在不需要标签的条件下证明性优于多数投票，并在 UltraFeedback、MMLU 和医疗健康数据集上验证了一致的提升。

**[CoOT: Learning to Coordinate In-Context with Coordination Transformers](multi_agent/coot_learning_to_coordinate_in-context_with_coordination_transformers.md)**

:   把"如何与陌生伙伴协作"从 task-generalization 改写成 partner-generalization 的 in-context 学习问题：训练一个 Decision Transformer 在跨 episode 的交互轨迹上预测最佳响应动作，让模型不更新参数就能在几局之内适应任何未见过的伙伴。

**[E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](multi_agent/e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)**

:   E-mem 把"预处理压缩成嵌入/图"的传统记忆范式改成"保留原始上下文 + 小模型助手就地推理"的情景重构范式：master agent 只做全局规划，多个 SLM assistant 各自守着一段未压缩的原文，按多路由检索激活后再做局部推理回传证据，在 LoCoMo 上 F1 反超 SOTA 7.75 个点的同时把 token 消耗砍掉 70%。

**[EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](multi_agent/engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)**

:   EngiAgent 把工程问题求解拆成 Analyzer/Modeler/Verifier/Solver/Evaluator 五个专家 Agent，再用一个**全连接协调器**动态路由反馈（而不是走固定流水线），让 GPT-4o 上工程任务的可行解率从 5.66%（zero-shot）/7.55%（MM-Agent）一跃到 64.15%，平均比此前 SOTA 提升约 7 倍。

**[Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM](multi_agent/learning_multi-agent_coordination_via_sheaf-admm.md)**

:   Sheaf-ADMM 把多智能体协调问题做成端到端可微的 ADMM 展开——每个 agent 只看局部 patch，独立解 ADMM 子问题（$\bm x$-update）、通过 cellular sheaf 定义的"边空间投影"协商一致（$\bm z$-update）、用对偶变量 $\bm u$ 累积分歧；在 maze pathfinding / MNIST / Sudoku 上 agents 协同得出正确全局解，且推理路径有可分析的 primal/consensus/dual 三态——比 MPNN 更可干预。

**[MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](multi_agent/mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)**

:   把"自动多智能体系统设计"重新表述为一次性输出整张 MAS 的函数调用 RL 问题，并配套 MASBench 从 Depth/Horizon/Breadth/Parallel/Robustness 五个轴说清楚"什么时候多智能体真的比单智能体强"。

**[MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)**

:   MASPO 通过多粒度联合评价（局部有效性 + 前瞻潜力 + 全局对齐）+ 错位案例驱动的进化束搜索，在不依赖标注的前提下端到端地为整条多智能体链路联合优化角色提示词，6 个任务上平均提升约 2.9 分。

**[MASPOB: 用 GNN 代理 + LinUCB + 坐标上升做多智能体提示优化](multi_agent/maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura.md)**

:   MASPOB 把多智能体系统的 prompt 优化看作预算紧缩的黑盒优化，用 GAT 代理模型捕获 workflow topology 下的 prompt 耦合、用 LinUCB 在嵌入空间算 epistemic uncertainty、用坐标上升把联合搜索拆成序贯单体问题，复杂度从 $\mathcal{O}(\prod |\mathcal{P}_i|)$ 降到 $\mathcal{O}(\sum |\mathcal{P}_i|)$；在 6 个基准（QA/Code/Math）上平均 80.58 超越 MIPRO 78.87、AFlow 78.52、IO 68.56。

**[OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)**

:   本文把多智能体系统的优化空间形式化为五个维度（两个功能维度 + 三个结构维度），用"Semantic Initializer 生成 + Contrastive Comparator 对比改进"的双 actor 算法在每个维度上做监督式优化，再迭代联合优化多个维度，在 HumanEval / MMLU / MATH 上稳定打败 DyLAN、ADAS、AFlow 等基线。

**[ProtocolBench: Which LLM MultiAgent Protocol to Choose?](multi_agent/protocolbench_which_llm_multiagent_protocol_to_choose.md)**

:   ProtocolBench 首次系统对比四大 LLM 多智能体通信协议（A2A、ACP、ANP、Agora）在任务成功、端到端延迟、消息字节开销、失败鲁棒性四轴上的表现——发现协议选择对系统行为有 36.5% 完成时间差、3.48s 延迟差；进一步提出 ProtocolRouter 按场景/模块动态选协议，将 Fail-Storm 恢复时间降 18.1%。

**[RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)**

:   RADAR 把多 LLM-Agent 系统的通信拓扑设计建模为一个"冗余感知"的离散图扩散过程，用 effective size 作为指导信号一步步增量生成 query-自适应的协作图，在 6 个基准上同时拿到更高准确率、更低 token 消耗和更强鲁棒性。

**[Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation](multi_agent/securing_multi-agent_systems_against_corruptions_via_node_contribution_backpropa.md)**

:   BPD 把 LLM 多智能体系统的多轮交互重构成 "带符号有向无环图"，把每条消息打成 $\{-1, 0, 1\}$ 的同意 / 漠视 / 反对分数，再用 PageRank 式的一次反向拓扑传播算出每个 agent 对最终答案的贡献分，分数离群者直接判定为恶意 agent 并切掉其出边——免训练、单次查询即用、对动态拓扑天然鲁棒。

**[Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](multi_agent/systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)**

:   本文将社会心理学的 Hidden Profile 范式搬到多智能体 LLM 评测里，构建 65 任务的 HiddenBench，在 15 个前沿 LLM 上系统揭示：单 agent 在 Full Profile 下能 80.7% 答对的同类任务，多 agent 在分布式信息下仅 30.1%，根本失败模式是**不会主动 elicit 别人没说出来的信息**，而轻量结构化沟通协议能跨家族大幅缓解。

**[When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems](multi_agent/when_cloud_agents_meet_device_agents_lessons_from_hybrid_multi-agent_systems.md)**

:   这篇论文系统研究云端 GPT-4o 监督者与端侧 Qwen3 执行者组成的混合多智能体系统，发现 PEVR 和 EVA 在 UI assistance 与 deep search 上各有优势，更多云端介入不一定更好，而上下文重置与摘要能显著改善端侧长任务的成本和 KV-cache 压力。

**[Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence](multi_agent/why_specialist_models_still_matter_a_heterogeneous_multi-agent_paradigm_for_medi.md)**

:   HetMedAgent 将通用 LLM、模态专科模型和临床医生组织成异构多智能体系统，通过冲突感知证据融合与不确定性路由，在心血管和胸片临床决策任务上证明专科模型与人类监督仍是医疗 AI 中不可替代的组成部分。

---

## ✂️ 语义分割 { #segmentation }

**[Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models](segmentation/activation-free_backbones_for_image_recognition_polynomial_alternatives_within_m.md)**

:   本文用 Hadamard 乘积构造 PolyMLP、PolyConv 和 PolyAttn，替代 MLP、卷积和注意力中的点激活/softmax，在 MetaFormer 风格骨干中无需常规激活函数也能在 ImageNet、鲁棒性和 ADE20K 分割上达到或超过激活式模型。

**[Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](segmentation/beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)**

:   提出 SymTrack，一个无需检测的双分支场景文字跟踪框架，通过预测性 Token 校正（PTR）解决透视畸变导致的特征瓶颈，跨专家校准（CEC）消除文字实例间的高视觉歧义，自适应推理引擎（AIE）稳定细粒度定位，在三个基准上大幅刷新 SOTA（最高 +12.32% AUC）。

**[FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation](segmentation/flowseg_dynamic_semantic_guidance_for_llm-conditioned_segmentation.md)**

:   本文指出当前基于 query 的 LLM-conditioned 分割是"propose-then-select"——候选 mask 往往已经够准，错就错在选不对；为此提出 FlowSeg，让 LLM 条件嵌入在 decoder 每一层都参与 query refinement 并被新的视觉证据持续更新，再叠一个轻量边界细化模块，在 RefCOCO/+/g 和 ReasonSeg 上一致刷点。

**[Functional Attention: From Pairwise Affinities to Functional Correspondences](segmentation/functional_attention_from_pairwise_affinities_to_functional_correspondences.md)**

:   本文把 Transformer 里的 softmax 注意力重新解释为"两个学得到的函数基之间的最小二乘线性算子"，借用形状匹配里的 functional maps 思想，把 $n\times n$ 的点对亲和矩阵压缩成 $k\times k$ 的紧致谱算子，在 PDE 求解、3D 点云分割和 OOD 推广上同时拿到 SOTA。

**[Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models](segmentation/geometry-preserving_unsupervised_alignment_for_heterogeneous_foundation_models.md)**

:   GPUA 把 CLIP 这种语义有余而局部精度不足的 VLM 和 DINOv3 这种细粒度足但缺语义的 VFM 看作两种"视觉语言"，用最优传输挖软对应再解正交 Procrustes 学一个保几何的线性映射，把 VFM 翻译进 VLM 空间——全程无监督、不更新任何预训练参数，零样本分类平均涨 11.8%。

**[LightAVSeg: Lightweight Audio-Visual Segmentation](segmentation/lightavseg_lightweight_audio-visual_segmentation.md)**

:   LightAVSeg 通过解耦 "语义筛选 (what)" 和 "空间定位 (where)"，用全局通道调制替换 $\mathcal{O}(N^2)$ 的跨模态注意力，让 AVS 模型在 20.5M 参数下达到 50.4 mIoU (MS3)，并在 Snapdragon 8 Elite 上做到 163.4 ms 的端侧延迟，比 AVSegFormer-R50 快约 $8\times$。

**[MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation](segmentation/mvr-cache_optimizing_semantic_caching_via_multi-vector_retrieval_and_learned_pro.md)**

:   MVR-cache 把 LLM 语义缓存的相似度度量从"单向量 cosine"升级为"可学习分段后的多向量 MaxSim"，并用 REINFORCE 训练一个轻量分段模型，在保证错误率上界 $\delta$ 不变的前提下把缓存命中率最多再抬 37%。

**[Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion](segmentation/refining_context-entangled_content_segmentation_via_curriculum_selection_and_ant.md)**

:   CurriSeg 不动分割网络结构，只换训练计划：先用"时间损失统计 + 像素熵加权"的稳健课程把模型推到稳态，再用反课程的"频谱失明"微调（砍掉高频迫使模型读结构语义），就让 FEDER / FSEL / RUN 在 CHAMELEON / CAMO / COD10K / NC4K 等伪装/息肉分割基准上稳定涨 2–4%，零额外参数、训练时间还更短。

**[Segment Anything with Robust Uncertainty-Accuracy Correlation](segmentation/segment_anything_with_robust_uncertainty-accuracy_correlation.md)**

:   针对 SAM 系列只输出 mask-level 单一置信度、在域漂移下出现"Mask-level Confidence Confusion"的问题，本文给 SAM2 接上 Weibull 双粒度贝叶斯 mask decoder 做像素级 epistemic 估计，并配以受人类视觉启发的 style + deformation 协同对抗扰动 + 校准损失，让 uncertainty 在 23 个 zero-shot 目标域始终与误差对齐，平均 J&F 达 79.87 同时不确定性图变得显著可信。

**[SPROUT: Supervise Less, See More — Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting](segmentation/supervise_less_see_more_training-free_nuclear_instance_segmentation_with_prototy.md)**

:   SPROUT 是首个完全训练无关、零标注的病理核分割框架——用 H&E 染色先验在每张切片上自构高置信度前景/背景区域→提取原型→用部分最优传输（POT）做特征-原型软对齐→输出 SAM 的正/负点提示；在 MoNuSeg 等基准上 AJI 比训练方法高 8.2%。

**[UGround: Towards Unified Visual Grounding with Unrolled Transformers](segmentation/uground_towards_unified_visual_grounding_with_unrolled_transformers.md)**

:   UGround 把 LMM-based 视觉定位从"用最后一层 $\langle\text{SEG}\rangle$ token 当 prompt"的范式翻转为"用动态选中的中间层相似度图当 prompt"，通过强化学习策略 SSC 让 $\langle\text{SEG}\rangle$ 滑过所有 transformer 层、把相似度图同时当作 SAM 的软 logit mask 和反向监督信号，首次在单一框架内统一了 RES / RS / FP-RES / gRES / Multi-RS 五种视觉定位任务，并在 ReasonSeg test 上 cIoU +9.0%、gRefCOCO val N-acc +12.1%。

**[无监督层级技能发现](segmentation/unsupervised_hierarchical_skill_discovery.md)**

:   HiSD 从无标签观测轨迹出发——通过最优传输进行技能分割，再用 Sequitur 文法诱导发现多层级技能层次，无需动作标签或奖励信号。

**[What Makes Synthetic Data Effective in Image Segmentation](segmentation/what_makes_synthetic_data_effective_in_image_segmentation.md)**

:   这篇论文系统分析了合成图像对语义分割有效的两个关键因素：复杂场景组合和高实例保真度，并提出 SENSE 用最优传输稳定合成图像的伪标签分配，从而在 Cityscapes、COCO、ADE20K 上稳定提升 DPT 和 Mask2Former。

---

## 🎁 推荐系统 { #recommender }

**[A Paired Testing Protocol for Batch-Conditioned Refusal Robustness in LLM Serving](recommender/a_paired_testing_protocol_for_batch-conditioned_refusal_robustness_in_llm_servin.md)**

:   本文把 LLM serving 中的 batch 条件当作安全评测的处理变量，提出安全提示与能力控制成对比较、人工/打分器校正、跨模型扩展、连续批处理组合和 batch-invariant kernel 消融组成的测试协议，结论是拒绝翻转真实存在但低频、模型特异且依赖具体服务栈。

**[Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](recommender/can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi.md)**

:   RSIR 让序列推荐模型用自身预测能力生成新的合成用户交互序列、再训练一个新模型，并用基于排名的"保真度检查"过滤掉偏离用户偏好流形的样本，防止 self-consuming model collapse；在 4 个数据集 × 3 个主流 backbone 上稳定提升 NDCG/Recall 4–11%，并理论上证明该过程等价于沿用户偏好流形切空间的隐式正则化。

**[GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation](recommender/gcib_graph_contrastive_information_bottleneck_for_multi-behavior_recommendation.md)**

:   GCIB 用"图信息瓶颈 + 跨行为对比学习"双管齐下，先在结构层把辅助行为图里与目标任务无关的边剪掉（最大化与目标行为的互信息、用 HSIC 替代项最小化与原始辅助图的互信息），再在特征层把去噪后的辅助表示和稀疏的目标表示做 InfoNCE 对齐，从而在四个多行为推荐基准上把 HR@10 / NDCG@10 相对最佳 baseline 再推高 7%–40%。

**[Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System](recommender/incentivized_exploration_with_stochastic_covariates_a_two-stage_mechanism_design.md)**

:   RCB 把推荐系统里的"探索-利用"和"用户激励兼容"打包成一个动态贝叶斯激励兼容（DBIC）约束下的上下文 bandit 问题，提出冷启动 + IPGS 两阶段算法，在随机用户协变量场景下证明 $\tilde{O}(\sqrt{KdT})$ regret、可插入任意 offline learning oracle，并量化"激励价格"——冷启动样本量随 $\epsilon$ 收紧呈 $1/\epsilon^2$ 增长。

**[Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](recommender/learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design.md)**

:   SkillPCF 把光子晶体光纤（PCF）的反向设计重塑为"记忆策略学习"问题：用 PPO 训练的控制器在每个轨迹片段从可演化技能库里挑 Top-K 个 memory 操作，执行器把它们落到轨迹记忆里，再用 MEEP 电磁仿真奖励同时优化控制器与技能库本身，在多 LLM 后端和经典优化基线上都拿到更好的设计成功率与仿真预算权衡。

**[Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](recommender/position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)**

:   这篇 position paper 借 Karl Marx 的"基础-上层建筑"框架，主张当下"sustainable AI"的讨论被环境维度独占而忽略了经济与社会维度，呼吁同时拉高**气候意识**与**资源意识**两条轴，并提出 CARAML 五层行动框架（个人 / 社区 / 工业 / 政府 / 全球）以抑制正在升级的"全球 AI 军备竞赛"。

**[Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](recommender/position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)**

:   这篇 position paper 指出 ML 社区在"数据节俭"(data frugality)上长期"只说不做"——大家口头承认 coreset 能省能耗，却几乎没人真去汇报能耗和碳排放，并以 ImageNet-1K 为例算出下游训练 + 存储约 5.82 GWh / 2589 tCO2e 的保守下限，呼吁把数据节俭从口号变成可度量、可执行、可奖励的工程实践。

**[Prompts for Public-Sector LLMs Should Be Governed as Commons](recommender/prompts_for_public-sector_llms_should_be_governed_as_commons.md)**

:   这是一篇 position paper：作者主张公共部门用的 LLM 提示词应当像开源 commons 一样被版本化、有出处、可审计、可否决，并用一座北美城市的 443 条社区提示词（增强到 3,317 条）跑了一个含五种治理状态的 pilot benchmark，给出可证伪的三个预测——治理化提示能改变输出分布、提升可审计性、缩短故障修复时延。

**[Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](recommender/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)**

:   作者把 LightGCN 的前向预测打开成"多跳邻居对的可学习权重之和"，发现 Sampled Softmax 损失只按物品侧邻居的结构相似度来加权、且对 UU/II/UI/IU 四类邻居对一视同仁，于是提出 NT-SSM——把用户侧结构相似度也接入梯度、并按邻居对类型分别校准加权策略，在四个数据集和多种 GCF 主干上稳定优于 SSM。

**[RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents](recommender/rgmem_renormalization_group-inspired_memory_evolution_for_language_agents.md)**

:   RGMem 借统计物理里的重整化群思想，把语言 agent 的长期对话记忆建模成"事件层 → 关系层 → 概念层"的多尺度系统，通过阈值触发的非线性算子把零散对话粗粒化成稳定的用户画像，从而打破"稳定 vs 可塑"权衡。

**[T-POP: Test-Time Personalization with Online Preference Feedback](recommender/t-pop_test-time_personalization_with_online_preference_feedback.md)**

:   T-POP 把"测试时对齐"和"神经决斗赌博机"拼在一起，在不动 LLM 参数的前提下，用每轮一对回复的在线偏好反馈在线学习个性化奖励函数，从而解决新用户个性化的冷启动问题。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](knowledge_editing/anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)**

:   AnyEdit++ 用 token 级 Bayesian Surprise 找到长文本中的语义转折点，把 AnyEdit 的固定窗口切分改成结构感知的 Bayes-Chunk，并在数学、代码、新闻、诗歌等长文本知识编辑任务上稳定提升 BLEU 与 BERT Score。

**[CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](knowledge_editing/crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)**

:   把 LLM 编辑写成"最小化编辑损失 s.t. 能力损失不变"的约束优化, 用 Bregman 散度等价转化为 Gauss-Newton Hessian 的低曲率子空间投影, 再借 K-FAC + 一个无需显式构造投影矩阵的 Kronecker 特征基技巧, 让 3000 条编辑在 A40 上 6 分钟跑完, 同时把 LLaMA-3-8B 的 MMLU/IFEval/ARC-C/TruthfulQA/GSM8K 平均掉点压到 < 1%, 显著优于 AlphaEdit / MEMIT / 微调。

**[Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](knowledge_editing/do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)**

:   本文提出 UniKE——首个面向统一多模态模型 (UMM) 的"跨模态知识编辑"基准（2,971 个编辑主体、5,535 条 VQA 可验证实例），系统性地揭示了"文本侧编辑成功率 ~92% 但图像生成 VQA 仅 ~18.5%"的模态鸿沟，并通过"推理增强参数编辑"协议把 VQA 准确率最多拉高 18.6 个百分点，进一步用条件通路上的余弦漂移指标将根因定位到 LLM-to-DiT 投影瓶颈。

**[From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](knowledge_editing/from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_.md)**

:   本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target，无需额外算力就能在 MEMIT/RECT/PRUNE/AlphaEdit 之上一致涨点。

**[KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)**

:   KORE 通过两阶段"知识导向控制"为 LMM 注入新知识 — 一边把单条事实自动扩成结构化的多轮对话+指令任务（提升泛化），一边用先前知识的协方差矩阵零空间初始化 LoRA 适配器（最小化对旧能力的干扰），在 LLaVA-v1.5 / Qwen2.5-VL 上同时实现强适配和强保留。

**[Reverse-Engineering Model Editing on Language Models](knowledge_editing/reverse-engineering_model_editing_on_language_models.md)**

:   论文揭示 locate-then-edit 类知识编辑方法（ROME/MEMIT/AlphaEdit）的参数更新矩阵会通过其行空间泄露"被编辑主语"的指纹，并提出两阶段攻击 KSTER（先用 SVD 恢复主语，再用前后模型的熵差恢复 prompt），同时给出基于"语义诱饵"注入的子空间伪装防御方案。

**[Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)**

:   本文从"维度坍塌"假设出发，证明参数级知识编辑会沿低奇异值方向被放大并随序列编辑线性累积，进而在多模型、多数据集、多评测维度上系统性地拖垮 LLM 核心能力，并指出一个简单的检索式基线 SCR 在所有设定下都优于现有参数编辑方法。

**[The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](knowledge_editing/the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)**

:   本文从优化角度证明：序列编辑（SE）之所以稳定，本质是"累积更新等价于一次性编辑（OTE）的解"，而 AlphaEdit 的零空间投影、PRUNE/RECT 的后处理正则等花哨机制并非关键——只要保证 OTE-SE 对齐，去掉这些正则也能在 4 个主流 LLM 上稳定完成 2000 步序列编辑。

---

## 👥 社会计算 { #social_computing }

**[Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases](social_computing/alignment_tampering_how_reinforcement_learning_from_human_feedback_is_exploited_.md)**

:   这篇论文提出 alignment tampering：当待对齐模型生成“高质量但带偏见”和“低质量但无偏见”的响应时，RLHF 的成对偏好标签会把质量与偏见混在一起，导致奖励模型、PPO/DPO 和 Best-of-N 采样进一步放大原本不想要的偏见。

**[FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences](social_computing/flips_instance-fingerprinting_for_llms_via_pseudo-random_sequences.md)**

:   FLIPS 通过设计**伪随机种子序列**（仅模型所有者知晓种子）来生成模型独特"指纹响应"——攻击者即便微调或剪枝模型也无法消除指纹，黑盒查询场景下检测率 > 99%、误报率 < 1%。

**[IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)**

:   IDO 通过**显式建模模态间不一致性**作为可学习的分布优化目标——同时拉近真新闻的多模态嵌入并扩大假新闻的不一致，在 Weibo / Twitter / Fakeddit 上 F1 较 SOTA 提升 3-7%、对未见过的假新闻泛化能力显著提升。

**[MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](social_computing/mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)**

:   MIND 通过**多视角理由生成 + 跨理由判别推理**为假新闻检测提供可解释 + 鲁棒的判别框架——同时利用 LLM 生成的事实核查、模态一致性、语义合理性 3 类理由，在 Weibo / Twitter / Fakeddit 上 F1 较 SOTA 提升 4-8%。

**[ObjEmbed: Towards Universal Multimodal Object Embeddings](social_computing/objembed_towards_universal_multimodal_object_embeddings.md)**

:   ObjEmbed 训练一个**通用的对象嵌入模型**——通过结合检测、分割、检索、描述、分类等任务对齐多模态对象表示，在 OVD / OVS / Text2Image-Object / Open-Caption-Eval 等 11 项任务上单一嵌入超越或匹配任务特定 SOTA。

**[SCOPE: Selective Conformal Optimized Pairwise LLM Judging](social_computing/scope_selective_conformal_optimized_pairwise_llm_judging.md)**

:   SCOPE 通过**双向偏好熵（BPE）**消除 LLM 评判中的位置偏差，结合**保形风险控制**实现有限样本 FDR 控制——在保持高覆盖率的前提下提供统计有效的风险界保证（覆盖率 0.583 时 FDR 仅 0.099 vs Vanilla 1.000 但 FDR 0.198）。

**[Self-Debias: Self-correcting for Debiasing Large Language Models](social_computing/self-debias_self-correcting_for_debiasing_large_language_models.md)**

:   Self-Debias 把 LLM 的去偏问题重塑为「在自回归推理链上对概率质量做公平资源分配」：用轨迹级后缀边际作为资源单位，套 Jain 公平指数防止资源在易样本上塌缩，再配 cold-start SFT 与基于一致性过滤的在线自训练，仅用 20k 标注种子就让 Qwen3-8B 在 8 个 fairness/utility 基准上的平均分从 77.5 拉到 81.7，并把基础模型「自我纠错越纠越歪」的塌缩翻转成稳定 +0.4。

**[The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](social_computing/the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)**

:   本文用测度论框架把 InfoNCE 损失提升到表示分布上的确定性"种群能量"，证明 unimodal 情形是凸的且收敛到唯一 Gibbs 平衡，而对称多模态情形会出现持续的负对称 KL 耦合，从几何上必然产生 modality gap。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[AutoBaxBuilder: Bootstrapping Code Security Benchmarking](aigc_detection/autobaxbuilder_bootstrapping_code_security_benchmarking.md)**

:   AUTOBAXBUILDER用LLM代理流水线自动生成Web后端安全评测场景、功能测试和端到端安全测试，把人工构建BAXBENCH式任务的成本降低约12倍，并构建出40个新场景的AUTOBAXBENCH来评估当代代码模型的正确性与安全性差距。

**[Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](aigc_detection/black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)**

:   SurpMark 把"AI 文本检测"重构成似然无关假设检验：用代理 LM 算 token surprisal 后 k-means 离散成 k 个状态，估计一阶 Markov 转移矩阵，再用广义 Jensen-Shannon 散度（GJS）和预先建好的"人写 / 机写"参考转移矩阵比较，单次前向就给出黑盒、无需重训、无需 per-instance 重采样的判别分数。

**[CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection](aigc_detection/core_conflict-oriented_reasoning_for_general_multimodal_manipulation_detection.md)**

:   作者把"多模态假新闻检测"重新定义为"显式捕获模态间或与世界知识之间的冲突"任务，构建了带细粒度冲突标注的 14k 语料 CAC，并提出 CORE 框架通过冲突感知训练（CPT）重塑 MLLM 的概念边界，使其在 DGM4、MDSM、MMFakeBench、NewsCLIPpings 四个数据集上以 100–750 个样本就大幅超过专用 SOTA。

**[Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook](aigc_detection/distributional_open-ended_evaluation_of_llm_cultural_value_alignment_based_on_va.md)**

:   DOVE 用率失真变分优化从 1 万篇人类文本中自动构造紧凑的"价值码本"，再用不平衡最优传输度量人类与 LLM 长文本在价值空间上的分布差异，从而在 12 个 LLM 上把"评测—下游任务"相关性从基线 ≤24% 拉到 31.56%。

**[Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](aigc_detection/feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)**

:   本文在「单阈值固定协议」下系统暴露 AI 文本检测器在跨数据集/跨生成器 shift 下的脆弱性，并提出把可学注意力加权的手工语言特征与 transformer [CLS] 表征融合，配合 DeBERTa-v3 backbone，在 M4 多域多生成器基准上达到 85.9% balanced accuracy，比强 zero-shot 基线（Fast-DetectGPT、RADAR、Log-Rank）高最多 +7.22。

**[Generating Robust Portfolios of Optimization Models using Large Language Models](aigc_detection/generating_robust_portfolios_of_optimization_models_using_large_language_models.md)**

:   本文提出一个轻量、无需训练的算法：用同一个 LLM 同时扮演"随机生成器"和"打分评审"两个角色，把生成概率前缀和达到 $1-\alpha$ 的候选优化模型打包成 portfolio，从理论上证明只要"生成器"或"评审"任一与人类偏好对齐，portfolio 就一定包含高质量优化模型，并在 NL4LP 上用 GPT 验证 portfolio 在最差情况下也稳定优于随机采样。

**[On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](aigc_detection/on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)**

:   针对零样本 AI 生成文本检测里"高频 boilerplate 稀释信号"和"单点概率脆弱"两大痼疾，作者提出 Uncertainty / Uncertainty++ 检测器：只在每段文本底部 $\rho$ 分位的低概率 token 上聚合 log-prob，并叠加同一组位置上的 Rényi 熵作为分布形状信号，再在 12 个生成器、7 个数据集上把平均 AUROC 从 Lastde 的 86.49 推到 88.74，且在改写 / 改解码这类扰动下显著更稳。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion](autonomous_driving/constrained_multi-objective_reinforcement_learning_with_max-min_criterion.md)**

:   本文把"max-min 多目标公平性"和"硬性约束满足"统一到同一个 MORL 框架中——通过占用测度 (occupancy measure) 重新表述为凸规划，再对偶出一个关于权重 $(u,w)$ 的凸优化问题，从而用一套投影梯度下降算法同时实现公平性和约束可行性，并给出几何收敛速率的理论保证。

**[DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving](autonomous_driving/deepsight_long-horizon_world_modeling_via_latent_states_prediction_for_end-to-en.md)**

:   DeepSight 把"未来世界预测"从显式像素重建（codebook 单帧）换成在 BEV 空间对 DINOv3 语义特征做**多帧并行隐式预测**，再叠加一个按需触发的 Adaptive Chain-of-Thought，让 Qwen2.5-VL-3B 在 Bench2Drive 闭环上 Driving Score 86.23 (+7.39)、Success Rate 71.36% (+13.63)，且只多 ~4% 推理延迟。

**[Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering](autonomous_driving/mitigating_error_accumulation_in_continuous_navigation_via_memory-augmented_kalm.md)**

:   把无人机连续 VLN 的 step-by-step 预测重写成"递归贝叶斯估计 = GRU 先验 + 记忆库似然 + 可学习卡尔曼增益"的闭环, 在 TravelUAV 上仅用 10% 数据微调就把 L1-Full 的 SR 从 17.6% 推到 25.9%, 同时把 100 步后还在不断累积的位置漂移压平到 30–40 米。

**[Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation](autonomous_driving/plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation.md)**

:   本文提出 PLMD：把 BEV 语义图与障碍图合并成 Label Map，用 DDPM 在障碍先验调制下补全未探索区域的语义+障碍标签，作为即插即用模块挂在任意 GON 策略上，在 ON / IIN / MRON 三类任务的 HM3D/MP3D 上一致刷新 SOTA。

**[Threshold-Based Exclusive Batching for LLM Inference](autonomous_driving/threshold-based_exclusive_batching_for_llm_inference.md)**

:   本文系统刻画了 LLM 推理中 mixed batching (MB) 与 exclusive batching (EB) 的性能交叉条件，证明带宽受限 GPU 上 prefill–decode 同批会因带宽争抢拖慢 Attention，进而推导出基于 hazard rate 的最优相位切换阈值 $\theta^*$ 和内存安全的批大小，并设计在线自适应调度器 EB+，在带宽受限硬件上吞吐最多提升 41.9%，非平稳流量下相对 MB 最多提升 36.4%。

**[TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](autonomous_driving/tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)**

:   TSRBench 构造了一个覆盖 14 个领域、4 大维度（感知/推理/预测/决策）、15 个任务、4125 道题、同时支持文本/可视化/文本+图/嵌入四种模态输入的时间序列推理基准，系统评测 30+ 主流 LLM、VLM 与 TSLLM，揭示出"scaling 在感知/推理上仍成立但在预测上失效"以及"文本与可视化模态高度互补但当前模型几乎无法融合"等关键结论。

---

## 🎯 目标检测 { #object_detection }

**[Adversarially Robust Approximate Furthest Neighbor](object_detection/adversarially_robust_approximate_furthest_neighbor.md)**

:   这篇理论论文首次给出能抵抗自适应查询对手的近似最远邻数据结构，在保持与 Indyk 经典 oblivious 算法相近的 $n$ 依赖查询复杂度的同时，证明传统随机投影最远邻算法会被自适应查询击穿。

**[EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](object_detection/earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)**

:   EARL 用"粗解析-细响应"两阶段 MLLM 框架把第一视角交互理解任务（描述+答问+像素掩膜）做成统一管线：第一阶段输出整图交互的全局描述并把最后一层 hidden state 当作语义先验，再通过新的 Analysis-guided Feature Synthesizer 注入到第二阶段，用 GRPO + 三路奖励（格式/答案/grounding 准确率）联合训练，在 Ego-IRGBench 上 cIoU 反超 Seg-Zero 8.37%。

**[FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization](object_detection/focus_forcing_in-context_object_localization_through_visual_support_constraints_.md)**

:   FOCUS 通过"完全去除类别名 + 注意力 mask 优化 + GRPO IoU 奖励"两阶段训练，让 VLM 真正按视觉支持示例（而非语义先验）做 in-context 目标定位；7B 参数模型超 72B 模型，证明任务对齐的 inductive bias 比单纯 scaling 更重要。

**[Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](object_detection/mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)**

:   MPFM 把 OSAD 里传统的"单峰高斯原型"换成可学习的**高斯混合原型空间**, 用流匹配直接回归一个 GMM 形式的速度场, 再加一个互信息最大化正则防止原型崩塌, 在 9 个工业 / 医学 AD 数据集上以 10/1 个异常样本的设定打过 DRA / AHL / DPDL 等所有 SOTA.

**[OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration](object_detection/omniverifier-m1_multimodal_meta-verifier_with_explicit_structured_recalibration.md)**

:   针对多模态视觉验证器只输出 True/False 二值判断信号过粗、且文本解释易被 reward-hacking 的问题，本文提出 OmniVerifier-M1：用 bounding box 等符号化输出代替文本作为 meta-verification rationale 以支持 IoU 这种 rule-based reward，并在理论与实验上证明把二值判断与 meta-verification 解耦成两条独立 reward 流（而非合并成乘性 joint reward）能显著提升 SNR，最终把验证器升级为可驱动 region-level 自校正的 agentic 系统 M1-TTS。

**[Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection](object_detection/testing_the_test_score-direction_instability_in_class-split_anomaly_detection.md)**

:   作者指出"类内拆分"(class-split) 异常检测基准在异常类与正常混合分布在表示空间重叠时是病态的——AUROC 会塌缩到随机甚至反转，方向取决于未知的异常类，并提出一个无需训练的"邻域类泄漏"指标 $L_k$ 来在跑分前诊断这种基准失效。

---

## 🗣️ 对话系统 { #dialogue }

**[DiscoverLLM: From Executing Intents to Discovering Them](dialogue/discoverllm_from_executing_intents_to_discovering_them.md)**

:   DiscoverLLM 把 "用户没想清楚自己要什么" 形式化为意图层级树的渐进发现过程，用可奖励的层级化用户模拟器训练模型在不清晰时主动发散探索、在清晰时收敛执行，在创意写作 / 技术写作 / SVG 三任务上比 CollabLLM 等 baseline 满意度 +10%、对话长度 -40%。

**[From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)**

:   针对"多轮交互式工具调用 Agent"后训练里两大瓶颈——高质量数据贵 + 用户模拟噪声毁 RL 信号，作者提出"自演化多 agent 数据合成 (AReaL-SEA)"配套生成可执行 verifier 当奖励，再配上"先 SFT 用户模型再做大 batch + 动态过滤 GRPO"的 RL recipe，在 τ²-bench 上把 Qwen3-235B 推到 Airline 73.0 / Telecom 98.3 的 pass^1，全面达到或超过 Claude/Gemini/GPT-5。

**[Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](dialogue/is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)**

:   本文把 LLM-as-a-Service 建模成"委托-代理"问题，证明现在主流的"按 token 收费"机制天然激励服务商把同一字符串重新切成更长的 token 序列来超额收费，并且即使强制服务商公开 next-token 分布，多收费而不被发现也只是 NP-Hard 而非不可行——作者给出一个简单启发式算法在保持合理性的前提下实测最多多收 11.2% 的 token，最后证明唯一能消除该激励的可加性定价机制是"按字符长度线性计费"。

**[Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](dialogue/not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)**

:   本文指出多轮对话场景下传统 Prefill-Decode 分离架构因每轮都要 P→D 重算并传输 KV 而严重低效，提出 PPD（Prefill-capable Decode）动态路由系统，让 decode 节点根据 SLO 权重决定是否本地处理 Turn 2+ 的 append-prefill，把 Turn 2+ TTFT 降低约 68%。

---

## 🧑 人体理解 { #human_understanding }

**[DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing](human_understanding/discoforcing_a_unified_framework_for_real-time_audio-driven_character_control_wi.md)**

:   DiscoForcing 把"音乐 → 全身舞蹈"的离线生成问题改写成严格因果、有界延迟的流式问题，用一个 VQ-PAE 因果音乐编码器 + 潜空间 Diffusion Forcing + 混合时间噪声调度 + 时间引导采样，把音乐流实时翻译成可直接驱动 Unity 虚拟人和宇树 G1 人形机器人的 30 FPS 全身动作。

**[Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets](human_understanding/efficient_validation-free_intrinsic_quality_estimation_for_large-scale_face_reco.md)**

:   提出 Intrinsic Quality (IQ)：用代理模型抽出嵌入后，把"邻域标签一致性 Consis"和"归一化谱熵有效秩 $\tilde{r}_{\mathrm{ent}}$" 加权融合，在不做完整训练、不要干净验证集的前提下给百万级人脸识别数据集打"可训练性"分数，在 WebFace4/12/42M 和注入噪声的设定上与下游 MFR-ALL 验证准确率排名一致性达到 Spearman = 1.0。

**[MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery](human_understanding/motiongrpo_overcoming_low_intra-group_diversity_in_grpo-based_egocentric_motion_.md)**

:   MotionGRPO 把 head-mounted 设备的第一人称全身动作恢复转化为扩散采样上的 MDP，用 GRPO 配合"轨迹条件感知模型 + 4 个 joint-level 子奖励"的混合奖励做后训练；同时识别出"输入条件太强、组内样本几乎一样导致 advantage 方差消失"这一致命瓶颈，并用 Perlin 噪声注入条件来恢复组内多样性，在 AMASS/RICH 上把 MPJPE 从 EgoAllo 的 124.985 mm 降到 114.207 mm。

**[WaveVerse: Scalable RF Simulation in Generative 4D Worlds](human_understanding/scalable_rf_simulation_in_generative_4d_worlds.md)**

:   WaveVerse 把 LLM 驱动的"4D 室内场景+人体动作"生成与一套保留时空相位相干性的物理光线追踪器拼成一条 prompt 到 RF 信号的流水线，用合成数据显著提升 RF 成像与活动识别下游任务，且性能随仿真量持续上涨而不像已有方法那样饱和。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[Edit-Based Refinement for Parallel Masked Diffusion Language Models](multilingual_mt/edit-based_refinement_for_parallel_masked_diffusion_language_models.md)**

:   ME-DLM 给 masked diffusion 语言模型（如 LLaDA）加一个"解码完再编辑修补"的轻量阶段：第一阶段照常 unmask 出粗稿，第二阶段用替换/删除/插入三种 token 级编辑做并行修正，监督信号来自 edit distance 的最短编辑脚本，在只用 1/8 扩散步数的情况下 HumanEval +11.6 / GSM8K +33.6 点反超 LLaDA-Instruct。

**[Optimizing Language Models for Crosslingual Knowledge Consistency](multilingual_mt/optimizing_language_models_for_crosslingual_knowledge_consistency.md)**

:   本文针对多语言 LLM 在不同语言间回答同一问题却给出冲突答案的问题，设计了一个**用"另一种语言下回答的对数似然"作为 reward 的 RL 目标**，证明其最优策略呈 product-of-experts 形式并在 $\gamma_1\gamma_2=\beta^2$ 时保证跨语言偏好一致；据此推导出无需 reward model、无需 online 采样的 **DCO（Direct Consistency Optimization）** 算法，在 9 个 LLM、3 个多语言 QA 基准、26 种语言上同时提升跨语言一致性（RankC）与回答准确率。

**[Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)**

:   LiRA 在冻结的多语言编码器与英文 LLM 之间插一层 "锚定 + 一致性正则" 的轻量微调模块，把低资源语言的句向量按 $\epsilon_1$（锚定误差）与 $\epsilon_2$（翻译 KL 距离）这两个理论可控的量约束到共享英文语义空间，从而在检索、排序与推理三类任务上同时拿到稳定提升。

---

## 🛰️ 遥感 { #remote_sensing }

**[Any2Any: Unified Arbitrary Modality Translation for Remote Sensing](remote_sensing/any2any_unified_arbitrary_modality_translation_for_remote_sensing.md)**

:   Any2Any 把遥感中的 RGB、SAR、NIR、MS、PAN 等传感器互译从一堆成对模型改成一个共享潜空间里的统一潜扩散模型，并用百万级 RST-1M 数据集和目标模态残差适配器，在 14 个已见翻译方向和多个未见模态组合上取得更好的保真度与泛化能力。

**[Localized, High-resolution Geographic Representations with Slepian Functions](remote_sensing/localized_high-resolution_geographic_representations_with_slepian_functions.md)**

:   本文用球面 Slepian 函数构造一种把表征容量集中在感兴趣区域 (ROI) 的地理位置编码器，并提出 Slepian-球面调和混合编码以同时兼顾局部高分辨率与全球粗粒度上下文，在五个分类、回归与图像增强预测任务上稳定超过 SH、Wavelet、RFF 等主流基线。

**[The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](remote_sensing/the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)**

:   作者指出视觉基础模型 (VFM) 在卫星图像上"看起来"很会预测，但在物理极端区段会沿物理坐标轴塌缩，于是用"结构同构"形式化"科学对齐"概念，并发布全球热带气旋基准 TC-Bench 与一套静态/动态/约束三层线性探针，系统揭露 DINO、CLIP、SigLIP、MAE 等冻结骨干在 $P_c<980$ hPa 强气旋段的表征崩溃。

---

## 🩺 医疗 NLP { #medical_nlp }

**[ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education](medical_nlp/clintutor-r1_advancing_scalable_and_robust_one-to-many_alignment_in_clinical_soc.md)**

:   提出 ClinTutor-R1，首个面向临床苏格拉底式教学的一对多对齐视觉语言 Agent，通过多智能体模拟器 ClinEdu 构建 48k 对话数据集 ClinTeach，利用显式心智理论推理和三轴 rubric 强化学习，在学员扩展至 10 人时仍保持教学质量稳定，超越基线模型 20% 并达到 GPT-4o 水平。

**[MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)**

:   作者提出一个把自由文本病例转成符合 HL7 FHIR R4 标准的"分阶段 LLM + 术语接地 + 修复循环"流水线，并据此从 MedCaseReasoning 构造出 1408 条结构化合成病例数据集 MedCase-Structured（成功率 82.5%），实验显示 GPT-5.4 / Gemini-3.1-Pro / Claude-Opus-4.6 在结构化 FHIR 输入上的诊断准确率比纯文本输入一致下降 4–23 个点。

---

## ✍️ 文本生成 { #nlp_generation }

**[Characterizing the Effect of Noise in Language Generation in the Limit](nlp_generation/characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)**

:   在 Kleinberg-Mullainathan 的"语言极限生成"形式化框架下，本文证明了对于均匀和非均匀生成，噪声水平 1 与任意有限噪声水平 $i \geq 1$ 等价（层级坍缩），但无噪声与噪声 1 之间存在严格分离，并首次给出了非均匀噪声依赖可生成性的完整刻画。

**[Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces](nlp_generation/score-repellent_monte_carlo_toward_efficient_non-markovian_sampler_with_constant.md)**

:   SRMC 用一个 $d$ 维的 running score 平均（而不是 $|\mathcal{X}|$ 维的经验测度）来记录历史，再通过指数 score-tilt 把这段历史折成一个"排斥已访问区域"的代理目标 $\pi_\theta$，套在任何 base MCMC kernel 外面，就能在通用状态空间下用常数内存实现非马尔可夫、低方差、保持归一化无关性的采样器。

---

## 📖 NLP 理解 { #nlp_understanding }

**[Causal Fine-Tuning under Latent Confounded Shift](nlp_understanding/causal_fine-tuning_under_latent_confounded_shift.md)**

:   本文提出 Causal Fine-Tuning (CFT)：在标准 BERT 微调里嵌入一个 SCM 启发的"高级稳定特征 $C$ + 低级混杂敏感特征 $\Phi$"分解，并用 front-door 风格的 do-calculus 调整公式做预测，在文本伪相关注入攻击下显著优于 SFT/SWA/WISE 等单域泛化基线。

**[Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting](nlp_understanding/controlling_the_risk_of_corrupted_contexts_for_language_models_via_early-exiting.md)**

:   本文把"用户提供的损坏上下文会降低 LLM 性能"这个问题形式化为风险控制——以 zero-shot 表现作"安全基线"，结合动态 early-exit（在中间层就出预测避免后层 overthink 有害上下文）+ context-aware 损失 + 改进的 Learn-then-Test 框架（保留负损失值用风险变换而非裁剪），在 9 个任务上既保证风险 ≤ user-specified $\epsilon$，又获得 > 50% 的算力加速。

---

## 📡 信号/通信 { #signal_comm }

**[Joint Model and Data Sparsification via the Marginal Likelihood](signal_comm/joint_model_and_data_sparsification_via_the_marginal_likelihood.md)**

:   JMDS 通过**最大化边缘似然**的统一目标同时实现模型和数据稀疏化——避免分阶段优化的次优性，在 CIFAR / ImageNet / WikiText 上以 5-10× 联合压缩比下保持优于独立稀疏化的性能。

**[Meta-learning Structure-Preserving Dynamics](signal_comm/meta-learning_structure-preserving_dynamics.md)**

:   把 modulation-based 元学习（hyper-network 把 latent code $\bm{z}^{(k)}$ 映射成层级调制参数）系统性地引入 Hamiltonian / GENERIC 神经网络，提出两种新颖调制——latent multi-rank (MR) 与 latent SVD-like 调制，让一个共享网络在不知道系统参数 $\bm{\mu}$ 的情况下少样本适配整族新参数实例，同时严格保持能量守恒 / 耗散结构。

---

## 🌍 地球科学 { #earth_science }

**[(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](earth_science/sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)**

:   MOSAIC 用"概率扰动 + 在 HEALPix 球面网格上的 mesh-aligned 块稀疏注意力"同时解决了 ML 天气预报模型的两类频谱退化（确定性平均带来的谱衰减 + 粗化潜空间带来的高频走样），在 1.5° 分辨率上仅 214M 参数就匹敌甚至超过 6× 高分辨率的模型，单 H100 12 秒生成 24 成员 10 天预报。

---

## 📂 其他 { #others }

**[A Hypertoroidal Covering for Perfect Color Equivariance](others/a_hypertoroidal_covering_for_perfect_color_equivariance.md)**

:   这篇论文用双覆盖把 HSL 中本来是区间值的饱和度和亮度提升到圆群上，构造 $\mathbb{T}^3$CEN，使网络对 hue、saturation、luminance shift 都能实现精确颜色等变，并在颜色偏移和医学图像等任务上提升鲁棒性。

**[A Perturbation Approach to Unconstrained Linear Bandits](others/a_perturbation_approach_to_unconstrained_linear_bandits.md)**

:   本文重新审视 Abernethy 等人的扰动式 bandit linear optimization 思路，提出 PABLO 归约，把无约束线性 bandit 转成可调用任意 OLO 子程序的问题，并由此得到 comparator-adaptive 静态/动态 regret、高概率界以及若干下界讨论。

**[Adaptive Multi-Round Allocation with Stochastic Arrivals](others/adaptive_multi-round_allocation_with_stochastic_arrivals.md)**

:   本文形式化网络招募为预算约束的顺序控制问题，证明单轮最优分配是贪心的；通过人口水平代理值函数将多轮规划降维到 $O(b^5\log b)$ 复杂度，并给出在模型误差下分解为前沿/人口/逼近三类误差的鲁棒性保证。

**[Adaptive Preconditioners Trigger Loss Spikes in Adam](others/adaptive_preconditioners_trigger_loss_spikes_in_adam.md)**

:   这篇论文把 Adam 训练中的 loss spike 归因于二阶矩预条件器与当前梯度平方的滞后解耦，并用预条件 Hessian 的梯度方向曲率解释和预测 spike 的发生。

**[AI Cap-and-Trade: Efficiency Incentives for Accessibility and Sustainability](others/ai_cap-and-trade_efficiency_incentives_for_accessibility_and_sustainability.md)**

:   作者借鉴碳排放 cap-and-trade，提出针对 AI 推理 FLOP 的配额-交易市场（AI Allowance），用 KKT 条件证明其能在合理参数下严格减少各公司 FLOP 使用，从而同时缓解大模型时代的能耗与小公司被挤出市场两大问题。

**[AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](others/amdp_asynchronous_multi-directional_pipeline_parallelism_for_large-scale_models_.md)**

:   AMDP 用多方向异步流水线、一步参数错配上界、梯度累积和 ZeRO 状态分片，在保持近同步收敛的同时提升大模型流水线并行训练吞吐，在 8 GPU GPT/BERT 实验中相对最强异步基线最高提升约 17%。

**[Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](others/amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)**

:   这篇论文把 generalized Bayes 中的 power posterior 家族直接摊销到一个同时以观测 $x$ 和温度 $\beta$ 为条件的 neural posterior estimator 上，使不同观测和不同 $\beta$ 下的后验采样可通过一次前向传播完成，而不再需要每次运行 MCMC。

**[AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions](others/autonumerics-zero_automated_discovery_of_state-of-the-art_mathematical_functions.md)**

:   提出 AutoNumerics-Zero，一种零先验知识的进化符号回归方法，从空程序出发自动发现逼近超越函数（如指数、余弦）的算术程序，在有限精度目标下以更少的运算次数超越了数百年来数学家设计的经典逼近方法。

**[Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](others/beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)**

:   针对公共部门AI系统"技术上可行但部署上失败"的普遍现象，提出**机构对齐就绪度 (Institutional Alignment Readiness, IAR)** 五维评估框架，从制度兼容性、数据生态成熟度、人工监督能力、财政可持续性和法规对齐五个维度评估接收机构是否具备负责任部署AI系统的条件。

**[Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](others/bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)**

:   提出基于并行关联扫描（parallel associative scan）的脉冲神经网络并行训练方法，在保持精确硬重置动力学的同时实现最高 44 倍加速，并用可微分数值根求解器实现机器精度的脉冲时间计算。

**[Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](others/cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)**

:   TabCascade 把表格行拆成"低分辨率（类别 + 数值的离散化版本）"与"高分辨率（连续数值）"两段级联：先用 CDTD 学低分辨率联合分布，再用 flow matching 在低分辨率引导下生成数值细节，并通过数据相关耦合 + 可学非线性时间表收紧 transport cost；天然支持缺失值、零膨胀等"混合型特征"的生成，在 12 个数据集上 detection score 比 SOTA 提升 51.9%。

**[Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](others/complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)**

:   本文提出 Complexity-as-Advantage (CAA)：把"复杂度"重新定义为一族**资源受限观察者**在同一过程上的**后悔（regret）分散程度**，并证明它在 log-loss + Markov 框架下等价于条件互信息原子之和（恰好恢复 excess entropy），在编码视角下等价于过剩描述长度的方差（MDL），从而把 Kolmogorov 复杂度、Bennett 逻辑深度、excess entropy 统一成一个**可计算、可经验估计**的标量谱。

**[Comprehensive AI Governance Requires Addressing Non-Model Gains](others/comprehensive_ai_governance_requires_addressing_non-model_gains.md)**

:   本文是一篇立场论文，论证当前以模型为中心的AI治理范式在"非模型增益"（推理增益、系统增益、资产增益）日益重要的背景下效力递减，需要系统治理、实体治理、代理治理和云治理等多层互补方案来填补监管空白。

**[Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](others/conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)**

:   本文提出条件核岭回归（Conditional KRR）框架，将一组非惩罚特征注入核方法中，通过残差核将其归约为标准 KRR，证明了归约代价为 $\mathcal{O}(1/\sqrt{N})$，并在硬阈值（top-k 本征函数）和软阈值（随机高斯特征）两种设定下验证了条件 KRR 优于标准 KRR 的充分条件。

**[Connecting Independently Trained Modes via Layer-Wise Connectivity](others/connecting_independently_trained_modes_via_layer-wise_connectivity.md)**

:   提出 Low-Loss Path Finding (LLPF) 算法，通过逐层连通性和方差球约束，可靠地在独立训练的神经网络模型之间构建低损失路径，支持 MobileNet、EfficientNet、CCT 等现代架构，且结果高度可复现。

**[Consistency Training Can Entrench Misalignment](others/consistency_training_can_entrench_misalignment.md)**

:   本文提出"一致性非中性假说"，通过在 108 个"模型有机体"上评估 7 种一致性训练方法，发现一致性训练并非对齐中性的——它系统性地抑制脆弱的奖励黑客和涌现性错位，但放大稳定的谄媚行为，分布偏移（而非分数选择）是主要驱动因素。

**[Continual Learning of Domain-Invariant Representations](others/continual_learning_of_domain-invariant_representations.md)**

:   作者首次把"域不变表示学习（DIRL）"显式注入到持续学习里：以 replay buffer 为载体做 multi-domain 不变性计算 + 域条件对齐，提出 ⋆-CL-{VREX, Fishr, CORAL, MMD, ANDMask} 五个方法，在六个跨视觉/医学/制造/生态的数据集上把目标域准确率推到 SOTA。

**[CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](others/core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)**

:   作者把多任务学习里"负迁移"的根因从"梯度冲突"重新归到"共享表征里语义和噪声纠缠"，提出 CORE-MTL：双流编码器把表征拆成语义 $\hat{Z}_s$ 和残差 $\hat{Z}_r$，用 CKA 独立性约束 + 反事实风格替换 + 反演渲染重构来落地"因果正交"，理论上给出比梯度平衡更紧的 OOD 上界，实验上在 NYUv2/Cityscapes 的 ID 与 GTA5→Cityscapes、Cityscapes-C 的 OOD 设定上同时压过 PCGrad/GradNorm/STCH/FairGrad 等十种 baseline。

**[Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](others/correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)**

:   作者指出经典 Hoeffding Tree（HT）在数据流上分裂时使用的"固定样本量"集中不等式被它自己采用的"数据相关停止规则"破坏，于是用 testing-by-betting + Universal Portfolio 重写分裂判据，让单棵树和 Adaptive Random Forest 都能在任意停止时刻保持 Type-I 错误可控，同时在 12 个真实流上更准且树更小。

**[Cost-Aware Stopping for Bayesian Optimization](others/cost-aware_stopping_for_bayesian_optimization.md)**

:   作者把 Weitzman 的 Pandora's Box 停下原则推广到带相关性的贝叶斯优化场景，证明 PBGI/LogEIPC 这两个 cost-aware 采集函数在共同的"采集函数值越过当前最优"停下规则下，期望代价调整 simple regret 不会比"采一次就停"更差，从而给出首个对 cost-adjusted simple regret 有理论保证的自适应停下规则。

**[Coupled Training with Privileged Information and Unlabeled Data](others/coupled_training_with_privileged_information_and_unlabeled_data.md)**

:   针对"训练时能用、部署时拿不到"的特权特征 $W$，作者提出一种**部署模型 $f$ 与富视图模型 $g$ 联合训练**的框架，通过显式约束 $g$ 在标注数据上的拟合误差来自适应控制特权信息的影响强度，从而在 $W$ 信号弱或带噪时避免传统两阶段伪标签法的负迁移现象。

**[CyberGym-E2E: Scalable Real-World Benchmark for AI Agents' End-to-End Cybersecurity Capabilities](others/cybergym-e2e_scalable_real-world_benchmark_for_ai_agents_end-to-end_cybersecurit.md)**

:   本文构建了 CyberGym-E2E——首个覆盖"漏洞发现 → PoC 生成 → 补丁生成 → 功能回归测试"全生命周期的大规模真实世界 AI Agent 安全基准（920 个漏洞 × 139 个开源项目），并通过 agent 辅助 + 专家终审的四步流水线把人工成本压到最低；评测显示前沿模型在 patch-only 任务上能到 80%+，但在端到端任务上 S3 成功率最高仅 65.9%（GPT-5.4），漏洞发现而非补丁生成是真正的瓶颈。

**[Decision Tree Learning on Product Spaces](others/decision_tree_learning_on_product_spaces.md)**

:   本文把 Blanc et al. (ITCS'20) 对"top-down greedy 决策树启发式"的理论保证从均匀分布推广到**任意乘积分布**，给出 $\exp(\Delta_\mathrm{opt} D_\mathrm{opt}\log(e/\epsilon))$ 大小上界（满二叉树情形严格优于 ITCS'20），且**完全免参数**——不需要预知最优树大小或深度即可运行。

**[Decoupled Conformal Optimisation: Efficient Prediction Sets via Independent Tuning and Calibration](others/decoupled_conformal_optimisation_efficient_prediction_sets_via_independent_tunin.md)**

:   本文提出 DCO-Warmstart——一种 "训练–调参–校准" 三分式的贝叶斯保形优化范式：把效率搜索放在独立的 tuning split 上、把保形分位数留给一份未被触碰的 calibration split，从而在任意大小（甚至无穷）的候选结构类上无需置信参数 $\delta$ 也能拿到标准的有限样本边际覆盖保证，且实证上预测集尺寸通常小于 CRC/BQ 等耦合校准方法。

**[DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](others/disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)**

:   用反因果图把混淆/对撞/中介三类偏差统一成一个条件独立准则 $\hat{Y} \perp \mathbf{B} \mid Y$，再设计 $O(n^2)$ 显存的单步可微估计器 sDISCO，作为正则项把条件距离相关惩罚塞进任何梯度训练的网络，从而缓解多种偏差且能扩展到多偏差场景。

**[DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers](others/disjunctivenet_neural_symbolic_learning_via_differentiable_convexified_optimizat.md)**

:   把"输入相关的 if-then 逻辑规则"写成多面体并的析取约束，通过基本步序列把 CNF 凸化成 DNF 的凸包，得到一个可微的 LP 投影层，神经网络输出经过这层后能在训练和推理时都精确满足原始 MILP 级别约束。

**[Envy-Free Allocation of Indivisible Goods via Noisy Queries](others/envy-free_allocation_of_indivisible_goods_via_noisy_queries.md)**

:   本文首次给"用噪声查询估值来寻找无嫉妒分配"这个新问题立了样本复杂度的尺：在两个智能体、加性高斯噪声、$m$ 件物品、最优负嫉妒缺口 $\Delta$ 的设定下，证明所需查询次数的紧界为 $\widetilde{\Theta}(m^{2.5}/\Delta^2)$（当 $\Delta\gg m^{1/4}$），上界由非自适应查询 + 单物品阈值多项式时间算法实现，下界对自适应查询和任意计算时间都成立。

**[Estimating Correlation Clustering Cost in Node-Arrival Stream](others/estimating_correlation_clustering_cost_in_node-arrival_stream.md)**

:   本文研究「节点到达」数据流模型下相关聚类（correlation clustering）代价的近似估计问题：作者提出 C4Approx 算法，用 $O(n^{(3+\alpha)/4}\log n)$ 词的**亚线性**空间和常数遍数得到 $(O(1), n^{1-\alpha})$-近似，并配套两个匹配下界证明多遍与加性误差都不可避免；在真实数据上仅存 2% 节点即达 Pivot 同等效果。

**[Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](others/expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)**

:   ECL 证明在协变量漂移下完整对齐输入分布 $P_s(X) = P_t(X)$ 并非校准的必要条件，只要"在每个置信度水平集上 $P(Y_k=1|X)$ 的条件期望两域一致"即可，并据此构造一个对 canonical / class-wise / top-label 三类校准都通用、可微、且 mini-batch 梯度无偏的损失 ECL。

**[FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo](others/foam_frequency_and_operator_error-based_adaptive_damping_method_for_reducing_sta.md)**

:   FOAM 通过一个可在陈旧特征空间里廉价估算的"算子相对误差代理 $h_t$"，把 Shampoo 的阻尼系数 $\epsilon$ 和特征分解（EVD）触发频率耦合成一个反馈控制回路，在大模型训练上把 EVD 调用次数砍掉 80%+ 同时保持收敛质量。

**[FOVI：面向深度视觉模型的生物启发式中心凹接口](others/fovi_a_biologically-inspired_foveated_interface_for_deep_vision_models.md)**

:   受人类视网膜—V1 通路启发,作者用"皮层放大函数 + 局部各向同性采样"构造出一种像素分布不均、但在传感器流形上密度均匀的中心凹输入接口 FOVI,并通过新颖的 kNN 卷积 + 核映射技术使其同时兼容 CNN 与 ViT,只用约 1/16 的像素就让 DINOv3-ViT 接近全分辨率基线的 ImageNet 精度。

**[From Generalist to Specialist Representation](others/from_generalist_to_specialist_representation.md)**

:   本文给出第一个完全非参数（无 intervention、无 functional 约束）的两层 hierarchical 可识别性证明：时间-任务结构由 collider 视角下的 CI test 可识别，任务相关 latent 由 sparsity 正则可从 generalist 表示中分离出来。

**[Guaranteed Optimal Compositional Explanations for Neurons](others/guaranteed_optimal_compositional_explanations_for_neurons.md)**

:   组合解释通常用束搜索找"和神经元激活对齐最好的逻辑公式"，但束搜索没有最优性保证；本文提出 IoU 的精确分解 (dIoU) + 一个 admissible 启发式 + 一个 best-first 最优算法，在与束搜索相当的运行时间内**首次保证给出全局最优解**，并据此揭示过去文献中 10–40% 的解释其实是次优的。

**[HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](others/haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)**

:   针对百万级标签的极端多标签分类，HASTE 把"每个标签独立采样 fan-in"改成"按语义分组共享 fan-in"，再配合一个吃掉高频标签的小 dense head，使得稀疏训练在 GPU 上真正跑出对应 FLOPs 的墙钟收益，前向最多 $4.4\times$、反向最多 $25\times$ 于现有稀疏基线，同时把与 dense 的精度差距收窄到几乎打平。

**[Honest Lying: Understanding Memory Confabulation in Reflexive Agents](others/honest_lying_understanding_memory_confabulation_in_reflexive_agents.md)**

:   本文揭露 Reflexion 类 agent 一种系统性失败模式——"记忆虚构 (memory confabulation)"：agent 会把错误的任务理解写进反思记忆并跨 trial 反复使用，作者用 Reflection Repetition Rate (RRR) 量化该现象，并用程序化反馈抽取替代开放式自我诊断，把 ALFWorld 上正确对象提及率从 0% 拉到 86%、RRR 从 0.64 降到 0.10。

**[How Does Bayesian Sampling Help Membership Inference Attacks?](others/how_does_bayesian_sampling_help_membership_inference_attacks.md)**

:   本文提出 BMIA，把单个参考模型用 Laplace 后验展开成"虚拟模型族"，靠贝叶斯采样估计每个样本的条件 score 分布，在只训 1 个参考模型的预算下，在 CIFAR-100 等数据集上把低 FPR 区域 TPR 拉到比训 8 个参考模型的 LiRA 还高 54%。

**[How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](others/how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)**

:   本文系统比较 Muon 与 Adam 在等变/几何网络（EGNN、DGCNN、PointNet、GotenNet、GINE）上的训练效果，发现 Muon 在 3D 点云任务上稳定优于 Adam，且收敛到的解在 Hessian 曲率、损失景观局部光滑度、权重/表征谱秩三个维度上都呈现显著不同的结构性差异——把"优化器选择"重新定位为等变网络训练里被严重忽视的一个 inductive bias。

**[Identifiable Equivariant Networks are Layerwise Equivariant](others/identifiable_equivariant_networks_are_layerwise_equivariant.md)**

:   本文在一个架构无关的抽象框架下证明：只要参数满足"弱可辨识性"，端到端 $G$-等变的深度网络一定存在等价参数化，使得每一层都对某个潜在群作用等变；这从理论上解释了"端到端等变会自动塌缩为逐层等变"这一长期被实验观察到的现象。

**[Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation](others/industrializing_prediction-powered_inference_the_glide_library_for_reliable_gena.md)**

:   GLIDE 把 PPI（prediction-powered inference）家族的最新估计器（PPI++、Stratified PPI、PTD、ASI）与采样器（uniform、stratified、active、cost-optimal）统一封装成 scipy 风格的均值估计库，专门解决"贵的人类标注 + 便宜但有偏的 LLM-as-judge"的混合评测问题，并配套蒙特卡洛验证与一颗决策树，让 GenAI / Agentic 系统的可信评估真的能上工业化。

**[Inference of Online Newton Methods with Nesterov's Accelerated Sketching](others/inference_of_online_newton_methods_with_nesterovs_accelerated_sketching.md)**

:   本文给在线 Newton 法装上 Nesterov 加速的 sketch-and-project 求解器，把每步成本压到 $O(d^2)$，并第一次刻画了"数据随机 + 求解器随机"双重不确定性下末迭代的渐近正态性，再配上一个无需矩阵求逆、可流式更新的协方差估计器，让带加速 sketching 的在线 Newton 真正可用于统计推断。

**[iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](others/iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)**

:   iWorld-Bench 是首个专门为"交互式世界模型"设计的统一评测基准，提出一套能把文本 / one-hot / 相机内外参三种动作输入折算到同一指令空间的 Action Generation Framework，并基于 330K 视频精挑 4.9K 任务、9 项指标，对 14 个主流模型做了全维度对比。

**[Knowing Isn't Understanding: Re-Grounding Generative Proactivity with Epistemic and Behavioral Insight](others/knowing_isnt_understanding_re-grounding_generative_proactivity_with_epistemic_an.md)**

:   这是一篇 ICML2026 Position 论文，主张生成式智能体的"主动性"不能只看是否更早、更自主、更持续地行动，而必须由两条联合约束规制——认识论合法性（agent 是否真的"理解"了情境）与行为承诺度（介入是否可逆、是否被迫升级），并把幻觉、对齐失败、不安全自治重新解释成"知道 / 行动"之间的错耦合（mis-coupling）。

**[Learning Permutation-Invariant Macroscopic Dynamics](others/learning_permutation-invariant_macroscopic_dynamics.md)**

:   本文针对粒子系统这类天然无序的微观状态，提出"重构密度而非重构粒子"的自编码器框架——用 DeepSet 编码器得到置换不变的闭包变量 $\hat{\bm{z}}$，再用条件归一化流把以观测点为中心的高斯混合密度作为重构目标，从而绕开点云匹配，并和宏观可观测量一起被一个 SDE/ODE 学到。

**[Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases](others/less_data_faster_training_repeating_smaller_datasets_speeds_up_learning_via_samp.md)**

:   本文系统刻画并解释了"小数据多重复反而比大数据更快收敛"的 small-vs-large gap 现象：作者证明该加速既不能由 CSQ-SQ 差距、梯度方差减少、输入分布偏置三种已有理论解释，又通过 2-sparse parity 上的 2-layer 二次激活 MLP 给出闭式步数界 $T = O((Nd)^{1/4} \log(d/\varepsilon))$，并通过随机标签、初始化缩放、层间学习率等一系列干预实验验证：真正驱动加速的是"小数据集天然存在的 $O(N^{-1/2})$ 采样偏差通过加快第二层范数增长来加速第一层特征学习"。

**[Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](others/local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)**

:   作者首次研究"从单条 Gaussian Glauber 动力学轨迹"中学习高斯图模型结构的问题，提出两种互补算法：LET-GL（基于 i,i,j,i 窗口的局部边检测、完美并行）和 BTR-GL（在 Dobrushin 条件下用 burn-in/thinning 把轨迹"解相关"成近似 i.i.d. 样本再喂给现成 i.i.d. 学习器），并给出有限样本恢复保证 + 信息论下界 + 一个独立有用的随机扫描高斯 Gibbs sampler 的 TV mixing 上界。

**[Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](others/mapping_human_anti-collusion_mechanisms_to_multi-agent_ai_systems.md)**

:   这是一篇 position / taxonomy 论文：把人类社会几百年积累的反合谋经验（制裁、宽大与举报、监控审计、市场设计、治理）按生命周期分成五类，再逐条映射到多智能体 AI 系统的可实现干预（reward penalty、whistleblower agent、telemetry-first overseer、interaction protocol 设计、shutdown 机制等），同时指出 AI 场景独有的归因、身份流动、合作-合谋边界、对抗适应等开放挑战。

**[Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach](others/markov_chain_monte_carlo_without_evaluating_the_target_an_auxiliary_variable_app.md)**

:   作者把 exchange、PoissonMH、TunaMH 三类"不算目标分布也能采样"的 MCMC 统一成一个用辅助变量的元算法，并在 proposal 与接受率两处同时引入辅助随机性，从而设计出小批量数据下仍保持精确平稳分布的梯度型 MCMC（Poisson–Barker、Poisson–MALA、Tuna–SGLD），实证显著超过 PoissonMH/TunaMH/SGLD 等基线。

**[Matroid Algorithms Under Size-Sensitive Independence Oracles](others/matroid_algorithms_under_size-sensitive_independence_oracles.md)**

:   作者提出「查询代价随查询集合大小线性增长」的尺寸敏感拟阵 oracle 模型，证明在该模型下找基、估计秩、估计划分数的最优查询代价都是 $\tilde{\Theta}(n^2)$，并对有界周长 $c$ 的拟阵给出 $\mathcal{O}(n^{2-1/c}\log n)$ 的最大权基算法突破二次下界。

**[Metadata Predictability Is Not Evidence Dependence: An Intervention-Based Audit for Weak-Label Benchmarks](others/metadata_predictability_is_not_evidence_dependence_an_intervention-based_audit_f.md)**

:   作者指出「输出能被元数据预测」≠「输出依赖证据」，提出双统计量审计协议：用 MPDS 测元数据可预测性、用证据洗牌 ΔEvi 测证据敏感性，再加 stronger-reader 校准层和输入消融，构成一个 4 步可复用的弱标签 benchmark 体检方案。

**[MetaDNS: Enhancing Exploration in Discrete Neural Samplers via Well-Tempered Metadynamics](others/metadns_enhancing_exploration_in_discrete_neural_samplers_via_well-tempered_meta.md)**

:   把分子动力学里的「well-tempered metadynamics」搬进离散神经采样器，用一个沿低维 collective variable 累积的历史相关偏置势 $V_t(s)$ 推平已访问的能谷，强迫 MDNS 类模型跨越能垒、覆盖多模态 Boltzmann 分布，并用重要性重加权保留无偏估计。

**[MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](others/mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)**

:   论文为 test-time adaptation 提供了第一份"目标风险 ≤ 源经验风险 + KL 复杂度 + MMD 分布偏移项"的 PAC-Bayes 上界，并把 MMD-球解读为 Walley 意义下的 credal set，从而用"上下风险区间"自然分离 aleatoric 与 epistemic 不确定性，给出"何时应当 adapt、何时该 abstain"的可计算判据。

**[Multi-Level Strategic Classification: Incentivizing Improvement Through Promotion and Relegation Dynamics](others/multi-level_strategic_classification_incentivizing_improvement_through_promotion.md)**

:   本文把传统单次"策略性分类"扩展成一个由多级三元分类器（通过/弃判/不通过 = 晋升/留级/降级）构成的序贯机制，证明仅靠折现因子 $\beta$、技能保留率 $\gamma$ 与"高位增益"$\delta$ 这三种跨期效应，就能把不可激励区域从 $c^+>c^-$ 缩小到 $(1-\beta\gamma)c^+>c^-$；进一步给出 $\mu_l = \delta(l-1)/(1-\gamma)$ 的稳态阈值序列，证明在温和条件下可激励诚实努力把属性推到任意高水平。

**[Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](others/multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)**

:   本文提出一种以 $\|\theta_j-\beta\|_{\bm\Sigma_j}$（矩阵加权范数）为正则项的鲁棒多任务线性回归估计器，用一个相对的"平衡度常数" $B$ 取代了既往工作中刚硬的"每个任务二阶矩最小特征值 $\Omega(1)$"假设，在病态/低秩/带离群任务的高维场景下同时给出最小最大率（minimax）、自适应、和回退到独立任务学习（ITL）的安全保证。

**[Networked Information Aggregation for Binary Classification](others/networked_information_aggregation_for_binary_classification.md)**

:   把 Kearns-Roth-Ryu 2026 的"在 DAG 上让线性回归 agent 顺序传 prediction 列即可逼近全局最优"结论推广到二分类：每个 agent 只看到部分特征列、顺序地把自己的 logit 转发给下游，能在 $M$-coverage 条件下用 $O(M/\sqrt{D})$ 超额 BCE loss 达到全局逻辑回归最优；同时构造硬实例证明 $\Omega(k/D)$ 下界，把网络深度刻画成信息聚合的根本瓶颈。

**[New Bounds for Kernel Sums via Fast Spherical Embeddings](others/new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)**

:   通过把 Bartal-Recht-Schulman 2011 的"随机 Nash 装置"球面嵌入定理用迭代 Fastfood 变换做成快速版（time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$），再把它作为 Gaussian KDE 的预处理把直径压到 $\widetilde{O}(1/\sqrt{\varepsilon})$，得到新的 Gaussian KDE 查询时间界 $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$，在小 $\varepsilon$ 中等直径的体制下优于 RFF / FJLT+RFF / Fastfood。

**[NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](others/nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)**

:   用一个 asinh 链接的 GLM surrogate 把多智能体 MCTS 的 joint-action 空间 $d^n$ 压成 low-dim 非线性 bandit，再用"一阶差分量 + 二阶 mixed difference"作为 NonUCT 提议规则，只在每个节点维护小候选集 $\mathcal{C}(s)$，证明 $\widetilde{O}(T^{3/4})$ 的局部 regret（与 $d^n$ 无关），在 MatGame/SMAC/SMACv2 上 sample efficiency 和最终性能都好过 MAZero 等强 baseline。

**[On Revisiting Entropy for Identifying Mislabeled Images](others/on_revisiting_entropy_for_identifying_mislabeled_images.md)**

:   作者发现"错标样本的预测熵在整个训练中持续偏高"这一现象不足以区分错标样本和困难干净样本，于是把熵乘上一个"预测是否对齐给定标签"的符号位得到 **signed entropy**，并沿训练 epoch 累积成 **SEI** 统计量，在 ISIC/DeepDRiD/PANDA/CheXpert 等多个医学数据集和 CIFAR-100N 上以纯插拔方式刷新错标检测 SOTA（最高领先 11%+）。

**[On the Coordination of Value-Maximizing Bidders](others/on_the_coordination_of_value-maximizing_bidders.md)**

:   本文形式化研究了在线广告中多个 value-maximizing 自动出价者的"协调"问题，提出"只让联盟中价值最高的成员出价、其余出 0"的简单协调机制，并证明对一大类自动出价算法而言，该机制能同时降低每个联盟成员的 RoS 违反量、并把联盟总价值推到所有协调机制的渐近最优。

**[On the Epistemic Uncertainty of Overparametrized Neural Networks](others/on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)**

:   本文指出过参数化神经网络的"认知不确定性"不会随数据增大而消失：因为参数不可识别（permutation + 神经元分裂），即便函数完全识别，参数空间后验仍然在分裂流形上保留连续不确定度，作者以单隐层 ReLU 网为例给出精确后验描述（Dirichlet on simplex）并实证验证。

**[On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](others/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)**

:   本文首次为测试时自适应（TTA）建立可学习性理论框架，用 $(\epsilon,\delta)$-Recovery Complexity 衡量分布漂移后模型把超额风险压到 $\epsilon$ 所需时间，并配合 $(\epsilon,\rho)$-TTA Learnability 把局部恢复推广到整条非平稳测试流，导出匹配阶的 minimax 上/下界，揭示了 TTA 的"适应速度—信息约束"内在权衡。

**[Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](others/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)**

:   在多项式逻辑斯蒂（MNL）bandit 的组合动作空间里首次给出**计算可行**的 G-optimal 实验设计——把 Frank–Wolfe 线性最大化谱写成 0–1 MILP 或多项式时间 Schur 补松弛——并据此造出第一个面向"线性效用 + 非均匀收益"的最佳组合识别算法，样本复杂度 $\tilde{\mathcal{O}}(d\log N / \Delta^2)$。

**[Optimal Regularization for Performative Learning](others/optimal_regularization_for_performative_learning.md)**

:   在高维岭回归框架下首次系统刻画了"模型部署反过来推动数据分布漂移"（performativity）场景中最优正则强度的标度律：最优 $\lambda$ 与表演性强度 $\bar b$ 成正比，并且在过参数化区域里恰当的正则甚至能利用表演效应反向降低风险。

**[Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization](others/over-alignment_vs_over-fitting_the_role_of_feature_learning_strength_in_generali.md)**

:   首次在标准分类任务里实证发现"特征学习强度（FLS）存在最优值"——既不是越大越好也不是越小越好——并用两层 ReLU 网络在 logistic loss 下的有限时间梯度流分析，把过大 FLS 引起的过拟合与过小 FLS 引起的"过对齐"分解为可量化的两个对立项，从而严格刻画最优 FLS 的存在性。

**[ParalESN: Enabling Parallel Information Processing in Reservoir Computing](others/paralesn_enabling_parallel_information_processing_in_reservoir_computing.md)**

:   将 LRU 风格的复数对角线性递推注入到 Echo State Network 的"未训练储备池"中，让传统 RC 的序列时间可并行化、维度可扩展到 10 万级，同时严格保持 Echo State Property 与衰退记忆滤波器的普适逼近性质。

**[Parsimonious Learning-Augmented Online Metric Matching](others/parsimonious_learning-augmented_online_metric_matching.md)**

:   本文回答了 Im et al. (2022) 留下的公开问题：把"按动作预测的"在线度量匹配带进"节俭预测"框架——预测被昂贵地按 $k$ 步一次发放——并通过 Follow-the-Prediction 框架 + 自动补齐"虚拟预测"的元算法，给出与已知下界基本匹配的确定性和随机性竞争比上界。

**[Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference](others/partitioning_for_intrinsic_model_inversion_resistance_in_collaborative_inference.md)**

:   本文跳出"在浅层中间表示上加噪/加掩码"的传统防御套路，从信息论出发证明：在边-云协同推理里，模型应当被切在表示发生"特征→决策"突变的那一层（作者命名为 Golden Partition Zone，GPZ），而类内均方半径 $R_c^2$ 是定位 GPZ、且能被标签平滑训练动态地主动收缩的关键变量。

**[Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](others/polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)**

:   Polaris 把概念表示拆成"方向（语义）+ 轨道势能（层级）"两个解耦信号，全部学到单位超球面上：用切空间投影 + 指数映射保证流形封闭，用各向异性球面 SVGD 防止赤道聚集，用 vMF KL 散度实现不对称的"父类应比子类更高熵"约束，在 taxonomy expansion 任务上把 top-K 召回提升最多 19 点、mean rank 降低 60%。

**[Position: Age Estimation Models Do Not Process Biometric Data](others/position_age_estimation_models_do_not_process_biometric_data.md)**

:   本文是一篇 position paper，用 14 个模型 × 3 个人脸验证基准的实证证据论证：人脸年龄估计模型在身份判别能力上比监管阈值低两个数量级，因此不应被自动归类为 GDPR / BIPA / EU AI Act 意义上的"生物特征数据处理"。

**[Possibilistic Predictive Uncertainty for Deep Learning](others/possibilistic_predictive_uncertainty_for_deep_learning.md)**

:   本文用 possibility theory 替代 Bayes 概率框架，提出 DAPPr——把参数空间的 possibilistic 后验通过 supremum 投影到预测空间，再用可学习的 Dirichlet possibility function 拟合，最终得到一个仅 10 行代码、可直接替换交叉熵、且在 OOD 检测上超越 EDL 家族的认知不确定性建模方法。

**[Private and Stable Test-Time Adaptation with Differential Privacy](others/private_and_stable_test-time_adaptation_with_differential_privacy.md)**

:   本文首次指出测试时自适应 (TTA) 会让模型参数泄露测试数据隐私，并把 Tent / EATA / SAR / DeYO / COME 五种主流 TTA 方法系统改造为带 per-sample 梯度裁剪 + 高斯噪声的 DP 版本，在 ImageNet-C 上既给出可证明的 $(\epsilon,\delta)$-DP 保证，又意外发现"裁剪本身"就能让 TTA 精度提升 $0.1\%$–$4.1\%$。

**[Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](others/provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)**

:   本文用「实代数几何 + 一阶谓词逻辑量词消去」给多维超参数调参第一次给出可证明的 generalization bound，把过去只能处理一维标量超参的 Balcan 2025 框架推广到任意 $p$ 维、双层验证损失、近似内层优化等多种实际场景，并配出第一条匹配上界的下界。

**[Realizable Bayes-Consistency for General Metric Losses](others/realizable_bayes-consistency_for_general_metric_losses.md)**

:   本文对"在一般（可能无界）度量损失下，假设类 $\mathcal{H}$ 何时存在分布无关的强通用 Bayes 一致学习算法"这一开放问题在 realizable 情形下给出锐刻画——充分必要条件是 $\mathcal{H}$ 不包含一种新的"无界 gap Littlestone 树"组合障碍。

**[Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations](others/rectified_lpjepa_joint-embedding_predictive_architectures_with_sparse_and_maximu.md)**

:   作者把 LeJEPA 的"投影后向各向同性高斯对齐"推广为"投影后向 Rectified Generalized Gaussian (RGG) 分布对齐"，通过整流 + 截断广义高斯获得显式可控的期望 $\ell_0$ 稀疏度，在 ImageNet-100 上 ResNet 编码器线性探针达到 $85.08\%$ 同时把 $\ell_0$ 稀疏度维持在 $\sim 73\%$，明显优于 LeJEPA 的全密集表示。

**[Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](others/reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)**

:   本文是一篇 ICML 立场论文,主张当前所有 AI 可靠性方法 (RAG / 自一致性 / RLHF / Agent Memory) 都只能验证显式知识,而 AI 真正强大的能力来自训练数据里 80-95% 未被人类正式记录的"隐式知识",作者提出 Knowledge Objects (KOs) 作为基础设施——把 AI 隐式推理外化成人类可检查、可验证、可背书的结构化产物,从而让一次人类验证的成本在群体中长期复利。

**[Rethink the Role of Neural Decoders in Quantum Error Correction](others/rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)**

:   本文在 $d\le9$ 的表面码上系统重做 MLP/3D-CNN/TCN/Transformer/GNN 五类神经解码器，并把"量化 + 剪枝 + FPGA 资源建模"作为一等公民放进训练流程，结论是：近期解码性能由数据量而非架构复杂度主导，且 INT4 + QAT 是实现微秒级实时解码的必要前提。

**[Rethinking Evaluation Paradigms in IBP-based Certified Training](others/rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)**

:   作者指出 IBP 类认证训练长期以"挑一个偏心配置"的方式相互比较是不公平的，提出用多目标贝叶斯超参搜索画出每种方法的 Pareto 前沿，证明既有 SOTA 普遍欠调优——CROWN-IBP 干净精度可再涨约 $6\%$、Tiny ImageNet 上 MTL-IBP 同时涨 $\sim2\%$ 干净精度和认证精度。

**[Rethinking FID Through the Geometry of the Reference Dataset](others/rethinking_fid_through_the_geometry_of_the_reference_dataset.md)**

:   本文指出 FID 的"越低越好"假设在不同参考数据集上系统性失效，并用分布密度 $\langle -\log d_k\rangle$ 和有效秩 $\mathrm{erank}(A)$ 两个几何描述子，通过分层线性模型证明它们能解释 ~70% 的"样本质量→FID"斜率跨数据集差异，从而把 FID 的脆弱性首次定量归因到参考集本身。

**[Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](others/return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)**

:   本文针对条件序列模型（如 Decision Transformer）中 return-to-go (RTG) 对齐不足的问题，提出 Q-align DT 框架——通过 RTG-to-behavior 对齐损失（强制 RTG 单调对应 Q 值变化）+ Q 函数的 RTG 扰动训练（共训练形成正反馈循环），在 D4RL 上达到 SOTA 性能且对齐误差大幅下降（HalfCheetah-medium 上 68.9 vs QCS 102.3）。

**[Riemannian Networks over Full-Rank Correlation Matrices](others/riemannian_networks_over_full-rank_correlation_matrices.md)**

:   本文把 MLR、FC、Conv 三种基础层系统地推广到满秩相关矩阵流形 $\mathrm{Cor}^+(n)$ 上的五种黎曼几何（ECM、LECM、OLM、LSM、PHCM），并为 OLM 与 LSM 推导出精确的反传，构造的 CorNet 在 Radar、HDM05、FPHA、NTU120 上一致超过同体量的 SPDNet / Grassmann 网络。

**[Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](others/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)**

:   作者把"从高斯噪声生成的合成域"当作半监督迁移学习里的替代源域，先证明这种"无语义但有判别结构"的噪声能给目标域带来可量化的泛化界改进，再用三损失的 Noise Adaptation Framework（NAF）联合优化两域风险与分布差异，使 CIFAR-10 上 4-shot ResNet-18 比 ERM 提升 12.35%。

**[Sequential Group Composition: A Window into the Mechanics of Deep Learning](others/sequential_group_composition_a_window_into_the_mechanics_of_deep_learning.md)**

:   作者把"对一段群元素求累积乘积"这个统一任务作为显微镜，用群上 Fourier 分析 + AGF 框架证明两层网络会按 Fourier 能量从大到小逐个学习不可约表示（irrep），并刻画两层、RNN、深层 MLP 三种架构在序列长度 $k$ 上分别需要 $2^k$ 宽度、$k$ 步、$\log k$ 层的表达力鸿沟。

**[Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](others/simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)**

:   本文为有符号图上的"坏三角形覆盖"问题（Bad Triangle Transversal, BTT）给出两个仅需单次解 LP 的简洁 2-近似算法，证明在完全图上 BTT 与 Correlation Clustering、MinSTC、Cluster Deletion 同时具有 $\tfrac{2137}{2136}$ 的 NP-难逼近下界，并构造了一种新的 pivot 流程把任意可行 BTT 覆盖转化为最多 $\tfrac{3}{2}|F|$ 错误的聚类，从而把 BTT 与 CC 最优值的差距从 2 收紧到 $3/2$。

**[SORA: Free Second-Order Attacks in Fast Adversarial Training](others/sora_free_second-order_attacks_in_fast_adversarial_training.md)**

:   本文从二阶视角重新审视单步对抗训练中的灾难性过拟合（CO），提出零成本曲率指标 PertAlign 来提前预警 CO，并据此推导出 SORA：一种用上一步反向传播梯度免费估计 Hessian、按通道随机化采样最优步长的自适应快速对抗训练算法，在 6 个数据集和 4 种架构上仅用同一组超参就稳定避免 CO 并刷新单步 AT 的鲁棒/干净精度 trade-off。

**[Structure-Induced Information for Rerooting Levin Tree Search](others/structure-induced_information_for_rerooting_levin_tree_search.md)**

:   在 $\sqrt{\mathrm{lts}}$ 框架中，作者提出三种"rerooter"——全局 Leiden 聚类、局部启发式 cost-to-go、二者加性混合——把搜索努力自动按状态空间结构和目标距离分配给隐式子任务，避免了 HIPS-$\varepsilon$ / SGPS 那种昂贵的显式子目标生成模型，在 BoulderDash、CraftWorld 等复杂域上的在线训练样本效率和测试展开数都达到 SOTA。

**[TabMGP: Martingale Posterior with TabPFN](others/tabmgp_martingale_posterior_with_tabpfn.md)**

:   把 TabPFN 这种预训练表格 Transformer 直接当作鞅后验（MGP）的预测规则，通过 in-context 前向滚动采样得到任意损失函数下参数 $\theta$ 的可信集，避免了手工设计先验/似然和拷贝拉超参，且在 30 个真实/合成场景下覆盖率与可信集面积同时优于手工 MGP 与经典贝叶斯。

**[Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification](others/target-agnostic_calibration_under_distribution_shift_with_frequency-aware_gradie.md)**

:   FGR 用 DCT 低通滤波去掉训练图像里的高频虚假捷径来在 OOD 上校准更准，再把「校准要变好」与「ID 不能塌」之间的梯度冲突用一次几何投影按硬约束方式解决，无需调权重就同时压住 OOD 的 ECE 和保住 ID 表现。

**[TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)**

:   TEMPORA 把 TTA 评测从「无时延上限的离线精度」改写为「时延受限下的可服务效用」，用离散 / 连续 / 摊销三类时间约束 + 可分解的效用指标，在 ImageNet-C × ResNet-50 上跑 750+ 次实验证明：离线榜首方法在 87.9% 的时延场景下输掉冠军，且越接近真实部署越不预测。

**[Test-Time Training with KV Binding Is Secretly Linear Attention](others/test-time_training_with_kv_binding_is_secretly_linear_attention.md)**

:   本文用四个「记忆悖论」反例 + 一套严格的展开定理，证明带 KV-binding 内循环的 TTT（如 LaCT、ViTTT）即便用多层 MLP + 动量也只是「学到的线性注意力算子」，并据此把它简化、并行化为标准线性注意力，吞吐提升 4× 而性能几乎不掉。

**[The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](others/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)**

:   本文证明：在光滑 $L$-同质模型 + 指数尾损失 + 学习率衰减的设定下，Muon（含 Muon-Signum、Muon-Adam）作为带动量的"归一化最速下降"会收敛到对应范数 max-margin 问题的 KKT 点；Adam（无稳定常数）则收敛到 $\ell_\infty$ max-margin 的 KKT 点，从而把以往仅对线性模型成立的隐式偏置结论一次性提升到所有光滑同质网络。

**[The Realignment Problem: When Right becomes Wrong in LLMs](others/the_realignment_problem_when_right_becomes_wrong_in_llms.md)**

:   本文把"模型部署后政策变了怎么办"形式化为 Realignment 问题,提出 TRACE 框架:用更强的 proxy 模型把已有 preference pair 三分类 (Invert / Punish / Retain) 后用混合 IPO+NPO+KL 目标做手术式再对齐,无需新一轮人工标注就能跟上政策漂移。

**[Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate](others/theoretical_analysis_of_sparse_optimization_with_reparameterization_weight_decay.md)**

:   本文提出 ReWA：把待优化变量重参数化为 $\boldsymbol{x}=\boldsymbol{y}^{K}$、对 $\boldsymbol{y}$ 加权重衰减、并使用一种坐标级自适应步长 $\eta_t \boldsymbol{y}^{M}/(\boldsymbol{y}^{K-1}+\epsilon)$，把不可优化的 $\ell_p\;(0<p<1)$ 稀疏正则等价转化为一个梯度有界、不易陷入零鞍点的可训练目标，并在 CIFAR-10 / ImageNet 上用 ResNet 验证了相对 $\ell_1$ 的稀疏性提升。

**[Torus Graphs for Large-Scale Neural Phase Analysis](others/torus_graphs_for_large_scale_neural_phase_analysis.md)**

:   作者把 Torus Graph (TG)——定义在 $d$-环面 $\mathbb{T}^d$ 上的指数族相位图模型——用随机化分数匹配把每步推断复杂度从 $\mathcal{O}(d^6)$ 砍到 $\mathcal{O}(d^2)$，由此首次支持上千个相位变量，并据此搭出 TG-HMM 与自回归 TG 两套动态/有向扩展，应用到小鼠 LFP 数据上揭示了清醒-NREM 之间的频率特异性相位重组。

**[Towards Optimal Robustness in Learning-Augmented Paging](others/towards_optimal_robustness_in_learning-augmented_paging.md)**

:   本文为带预测的随机化在线调页提出统一的「相对预测预算」(RPB) 视角，并基于 OnlineMin 设计 RPB-OnOPT 框架，把可证的鲁棒竞争比从既有的 $2H_k+O(1)$ 一举推到信息论下界附近的 $H_k+O(1)$，同时保持 1-一致性。

**[通过分布式鲁棒逐节点回归的变量聚类](others/variable_clustering_via_distributionally_robust_nodewise_regression.md)**

:   利用分布式鲁棒优化框架将逐节点回归的参数调优问题转化为带谱范数正则化的凸优化问题——实现无参数聚类方法，在模拟、人脸和金融数据上显著超越 Lasso 稀疏聚类。

**[Vision Transformer 微调中的非光滑分量优势](others/vision_transformer_finetuning_benefits_from_non-smooth_components.md)**

:   通过定义"可塑性"度量，本文证明 ViT 中的非光滑分量（注意力和前馈层）具有更高可塑性——在微调时能提供更大梯度范数，实现更好且稳定的迁移学习性能。

</div>