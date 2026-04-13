---
title: >-
  NeurIPS2025 图像生成方向 252篇论文解读
description: >-
  252篇NeurIPS2025 图像生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**🧠 NeurIPS2025** · 共 **252** 篇

**[70 Size 100 Accuracy Lossless Llm Compression For Efficient](70_size_100_accuracy_lossless_llm_compression_for_efficient.md)**

:   DFloat11 利用 BFloat16 权重中指数位（exponent）的低熵特性，通过 Huffman 编码将 LLM/扩散模型无损压缩至原始大小的约 70%（等效 ~11 bit），并设计了层次化查找表和两阶段 GPU kernel 实现高效在线解压，使 Llama 3.1 405B 可在单节点 8×80GB GPU 上无损推理。

**[A Closer Look At Model Collapse From A Generalization-To-Memorization Perspectiv](a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)**

:   发现扩散模型在自消耗循环（用生成数据训练下一代模型）中存在从"泛化"到"记忆"的转变过程，揭示训练集熵与模型泛化能力的强线性相关性（Pearson r=0.91），并提出基于熵的数据选择策略（Greedy Selection / Threshold Decay Filter）有效减缓该转变，在 CIFAR-10 accumulate 范式下第 8 轮 FID 从 75.7 降至 44.7。

**[A Connection Between Score Matching And Local Intrinsic Dimension](a_connection_between_score_matching_and_local_intrinsic_dimension.md)**

:   证明去噪得分匹配损失（denoising score matching loss）的下界恰好是数据流形的局部固有维度（LID），从而将 DSM loss 本身作为一个高效的 LID 估计器——无需梯度计算或多次前向传播，在 Stable Diffusion 3.5 上内存占用仅为 FLIPD 的 60%，且量化后估计更稳定。

**[A Data-Driven Prism Multi-View Source Separation With Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)**

:   提出 DDPRISM 方法，利用多视图观测中不同线性变换的结构性差异，在 EM 框架下为每个未知源学习独立的扩散模型先验，无需预先获得任何单独的源样本即可完成源分离和后验采样，在合成问题和真实星系观测上超越现有方法。

**[A Diffusion Model For Regular Time Series Generation From Irregular Data With Co](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)**

:   提出两步框架从不规则采样时序数据生成规则时序：先用 TST 自编码器补全缺失值构造"自然邻域"，再在视觉扩散模型中用 masking 策略仅在观测像素上计算损失，避免对补全值的过度依赖，在判别分数上平均改善 70%，训练速度提升 6.5 倍。

**[A Gradient Flow Approach To Solving Inverse Problems With Latent Diffusion Model](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)**

:   提出 DWGF（Diffusion-regularized Wasserstein Gradient Flow），将隐空间扩散模型的后验采样问题严格形式化为 KL 散度在 Wasserstein-2 空间上的正则化梯度流，推导出隐空间中的 ODE 系统用于求解图像逆问题，在 FFHQ-512 上的修复和超分辨率任务中 PSNR 大幅超越基线。

**[Accelerating Parallel Diffusion Model Serving With Residual Compression](accelerating_parallel_diffusion_model_serving_with_residual_compression.md)**

:   提出 CompactFusion 框架，通过残差压缩（仅传输相邻去噪步骤间的激活差异而非完整激活）来消除并行扩散推理中的通信冗余，在 4×L20 上实现 3.0× 加速且生成质量远优于 DistriFusion，在模拟以太网带宽下实现 6.7× 加速，甚至在 100× 压缩下仍优于 DistriFusion。

**[Accuquant Simulating Multiple Denoising Steps For Quantizing](accuquant_simulating_multiple_denoising_steps_for_quantizing.md)**

:   提出AccuQuant，一种用于扩散模型的训练后量化（PTQ）方法，通过在校准过程中显式模拟多个去噪步骤来最小化量化误差的累积效应，并通过新型目标函数将内存复杂度从O(n)降至O(1)。

**[Adapting Speech Language Model To Singing Voice Synthesis](adapting_speech_language_model_to_singing_voice_synthesis.md)**

:   将 1.7B 参数的 TTS 预训练 Speech Language Model 适配到歌声合成（SVS）任务，通过乐谱 tokenization + multi-stream LM 预测 + conditional flow matching 精修 + vocoder，仅用 135 小时合成歌声数据达到与专用 SVS 系统可比的性能。

**[Alebench A Benchmark For Longhorizon Objectivedriven Algorit](alebench_a_benchmark_for_longhorizon_objectivedriven_algorit.md)**

:   提出 ALE-Bench，首个面向分数制算法工程竞赛（AtCoder Heuristic Contest）的 AI 基准，评估 LLM 和 Agent 在 NP-hard 优化问题上的长时间迭代改进能力，发现当前最强模型（o3-high）仅达人类平均水平，且在问题一致性和长时间改进方面与人类专家差距显著。

**[Aligning Compound Ai Systems Via System-Level Dpo](aligning_compound_ai_systems_via_system-level_dpo.md)**

:   将复合 AI 系统建模为 DAG，提出 SysDPO 框架将 DPO 扩展到多组件联合对齐，通过 DAG 分解将系统级偏好转化为可端到端优化的损失函数，理论证明了 β-完美对齐保证，在 LLM+扩散模型和 LLM+LLM 系统上显著提升协作质量。

**[Aligning Text To Image In Diffusion Models Is Easier Than You Think](aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)**

:   提出 SoftREPA——一种轻量级对比微调策略，通过引入可学习 soft text token（不到 1M 参数）在冻结的预训练 T2I 扩散模型上进行对比学习，显式提高文本和图像表征的互信息，在 SD1.5/SDXL/SD3 上显著提升文本-图像对齐质量，且适用于图像生成和图像编辑任务。

**[Amortized Sampling With Transferable Normalizing Flows](amortized_sampling_with_transferable_normalizing_flows.md)**

:   提出 Prose——一个 285M 参数的全原子可迁移归一化流，基于 TarFlow 架构训练在 21,700 个短肽 MD 轨迹上（总计 4.3ms 模拟时长），实现对任意短肽系统的零样本无相关性提议采样，在能量评估预算相同时超越 MD 基线，生成速度比之前的可迁移玻尔兹曼生成器 (TBG) 快 4000 倍。

**[Auggen Synthetic Augmentation Using Diffusion Models Can Imp](auggen_synthetic_augmentation_using_diffusion_models_can_imp.md)**

:   提出AugGen——一种自包含（self-contained）的合成数据增强方法：利用扩散模型的条件向量插值（$c^* = \alpha c_i + \beta c_j$）实现类间混合生成，无需外部数据或模型即可为人脸识别提供1-12%的性能提升，等效于1.7倍真实数据量，IR50+AugGen甚至超越IR101 real-only。

**[Autoregressive Adversarial Posttraining For Realtime Interac](autoregressive_adversarial_posttraining_for_realtime_interac.md)**

:   提出 AAPT（Autoregressive Adversarial Post-Training），将预训练的潜在视频扩散模型转化为实时交互式视频生成器——每帧仅需单次神经网络前向传播（1NFE），自回归逐帧生成，8B 模型在单张 H100 上以 24fps 流式生成 736×416 视频，最长可达一分钟（1440帧）。

**[Badiff Bandwidth Adaptive Diffusion Model](badiff_bandwidth_adaptive_diffusion_model.md)**

:   提出 BADiff——首个带宽自适应扩散模型，通过将目标熵约束作为条件嵌入扩散反向过程，配合可微熵正则化损失和自适应停止策略，使模型根据实时带宽动态调整生成质量并自适应提前终止采样，在保持感知质量的同时减少计算开销，从根本上避免了传统"高质量生成→后压缩"流程中的压缩伪影和计算浪费。

**[Balanced Conic Rectified Flow](balanced_conic_rectified_flow.md)**

:   针对 k-rectified flow 中 reflow 步骤导致的分布漂移问题，提出 conic reflow：利用真实图像的反演噪声及其 Slerp 扰动构成锥形监督轨迹，大幅减少所需 fake pair 数量的同时获得更优的生成质量和更直的 ODE 路径。

**[Beyond Masked And Unmasked Discrete Diffusion Models Via Par](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)**

:   提出 Prime（Partial masking scheme），突破 Masked Diffusion Model 的二元状态（mask/unmask）限制，引入中间态（部分观测的 token 信息），减少冗余计算并实现更细粒度的去噪过程，在文本生成上 PPL 15.36 超越自回归模型（17.54）和标准 MDM（21.52），在图像生成上取得 CIFAR-10 FID 3.26。

**[Bitmark Watermarking Bitwise Autoregressive Image Generative Models](bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)**

:   提出 BitMark——首个针对比特级自回归图像生成模型（Infinity、Instella）的水印方案，在生成过程中通过对 logit 加偏置将 bit 序列引向"绿色列表"，实现可靠检测（z-test）、高图像保真度（FID 几乎不变）、对多种攻击的鲁棒性和放射性（训练在水印图上的下游模型也带有水印），为防止模型坍缩提供了关键工具。

**[Blameless Users In A Clean Room Defining Copyright Protection For Generative Mod](blameless_users_in_a_clean_room_defining_copyright_protection_for_generative_mod.md)**

:   重建生成模型可证明版权保护的理论基础——证明现有的 Near Access-Freeness (NAF) 定义不能防止逐字复制（"被污染"），提出"无辜用户"(blameless) 框架和净室版权保护 ($(\kappa,\beta)$-clean) 定义，其中用户在反事实"净室设置"中不会复制则在真实世界中也不太可能复制，并证明差分隐私训练在"黄金数据集"假设下蕴含净室版权保护。

**[Blind Strong Gravitational Lensing Inversion Joint Inference Of Source And Lens ](blind_strong_gravitational_lensing_inversion_joint_inference_of_source_and_lens_.md)**

:   首次将 score-based 生成模型先验应用于强引力透镜的盲反演——联合推断背景源天体形态和透镜质量分布参数，通过将 GibbsDDRM 扩展到连续时间域实现采样，重建残差与观测噪声一致，透镜参数边际后验无系统偏差。

**[Blurdm A Blur Diffusion Model For Image Deblurring](blurdm_a_blur_diffusion_model_for_image_deblurring.md)**

:   提出 BlurDM，将运动模糊的物理形成过程（连续曝光导致渐进模糊累积）集成到扩散模型——双扩散前向（同时加噪声+模糊）+ 双去噪去模糊反向，作为隐空间先验生成器一致性增强 4 种去模糊方法在 4 个数据集上的效果，GoPro 平均 +0.31 dB，RealBlur-J 平均 +0.78 dB，仅增加 ~4 GFLOPs 和 ~9ms。

**[Blurguard A Simple Approach For Robustifying Image Protection Against Ai-Powered](blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)**

:   提出 BlurGuard——在生成对抗扰动之前先对图像做轻度模糊预处理，使扰动更鲁棒地抵御 JPEG 压缩、高斯噪声等后处理操作，从而更有效地保护图像不被 Stable Diffusion 等 AI 编辑工具篡改，在保护成功率上比不模糊基线提升 20%+。

**[Boltznce Learning Likelihoods For Boltzmann Generation With Stochastic Interpola](boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)**

:   BoltzNCE 用 Score Matching + InfoNCE 混合训练 Energy-Based Model 来近似 Boltzmann Generator 的似然，避免了昂贵的 Jacobian trace 计算，在丙氨酸二肽构象生成上实现 100× 推理加速且自由能误差仅 0.02 $k_BT$。

**[Boosting Generative Image Modeling Via Joint Imagefeature Sy](boosting_generative_image_modeling_via_joint_imagefeature_sy.md)**

:   提出 Latent-Semantic Diffusion，让扩散模型联合生成 VAE 低级图像 latent 和 DINO 高级语义特征，通过最小修改标准 DiT 实现生成质量和训练效率的显著提升，并解锁 Representation Guidance 推理策略。

**[Breaking Ars Sampling Bottleneck Provable Acceleration Via D](breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)**

:   从信息论角度为扩散语言模型建立收敛保证，证明采样误差（KL散度）随迭代次数T成反比衰减且与token间互信息线性相关，关键证明了T<L（迭代次数可少于序列长度L）时仍可生成高质量样本，从理论上打破了自回归模型需要L步的基本采样瓶颈，并建立了匹配的上下界证明分析的紧致性。

**[Cadmorph Geometry-Driven Parametric Cad Editing Via A Plan-Generate-Verify Loop](cadmorph_geometry-driven_parametric_cad_editing_via_a_plan-generate-verify_loop.md)**

:   提出 CADMorph，一个迭代式 plan–generate–verify 框架，利用预训练的 Parameter-to-Shape (P2S) 扩散模型和 Masked-Parameter-Prediction (MPP) 大语言模型协同工作，在无需三元组训练数据的情况下实现几何驱动的参数化 CAD 编辑。

**[Camila Contextaware Masking For Image Editing With Language](camila_contextaware_masking_for_image_editing_with_language.md)**

:   提出 CAMILA，一种上下文感知的图像编辑方法，能够判断用户指令是否在当前图像上下文中可行，仅执行可行的编辑指令而忽略不可执行的指令，在单指令和多指令编辑场景中均优于现有方法。

**[Camit A Time-Aware Car Model Dataset For Classification And Generation](camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)**

:   提出 CaMiT 数据集（787K 标注 + 5.1M 无标注汽车图像，2005–2023），系统研究细粒度视觉类别的时间漂移问题，并在静态预训练、时间增量预训练、时间增量分类器学习和时间感知图像生成四个场景下提供 benchmark。

**[Can Knowledge-Graph-Based Retrieval Augmented Generation Really Retrieve What Yo](can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)**

:   提出 GraphFlow 框架，将知识图谱上的检索建模为 GFlowNet 的流匹配问题，通过详细平衡目标和局部探索策略联合训练检索策略与流估计器，在 STaRK 基准上检索准确率和多样性均超越 GPT-4o 约 10%。

**[Cdflow Building Invertible Layers With Circulant And Diagonal Matrices](cdflow_building_invertible_layers_with_circulant_and_diagonal_matrices.md)**

:   提出 CDFlow，利用循环矩阵和对角矩阵的交替乘积构造可逆线性层，将参数复杂度从 $\mathcal{O}(n^2)$ 降至 $\mathcal{O}(mn)$，矩阵逆复杂度从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn\log n)$，对数行列式从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn)$，在密度估计和周期性数据建模上超越同类方法。

**[Composite Flow Matching For Reinforcement Learning With Shifted-Dynamics Data](composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)**

:   提出 CompFlow，通过复合流匹配架构（在离线流输出分布上构建在线流）估计离线-在线环境间的动态差异（Wasserstein 距离），并结合高动态差异区域的主动探索策略，在 27 个动态偏移 RL 任务中平均回报超越最强基线 14.2%。

**[Composition And Alignment Of Diffusion Models Using Constrai](composition_and_alignment_of_diffusion_models_using_constrai.md)**

:   提出统一的约束学习框架来处理扩散模型的对齐（alignment）和组合（composition），将多奖励对齐形式化为 KL 散度最小化+奖励约束，将模型组合形式化为 minimax KL 散度问题，通过拉格朗日对偶的原-对偶训练算法求解，相比传统加权方法更可解释且避免了手动调权。

**[Conditional Panoramic Image Generation Via Masked Autoregres](conditional_panoramic_image_generation_via_masked_autoregres.md)**

:   提出PAR（Panoramic AutoRegressive model），首次用掩码自回归建模统一文本到全景图和全景图外延两大任务，通过循环平移一致性损失和双空间循环填充解决ERP全景图的边界不连续问题，在Matterport3D上取得37.37 FID，同时展示出良好的可扩展性和零样本泛化能力。

**[Constrained Discrete Diffusion](constrained_discrete_diffusion.md)**

:   提出 CDD（Constrained Discrete Diffusion），将可微约束优化投影算子嵌入离散扩散模型的去噪过程中，无需重训练即可在采样时强制满足序列级约束，在毒性文本生成、分子设计和指令遵循三类任务上实现零约束违反。

**[Contextual Thompson Sampling Via Generation Of Missing Data](contextual_thompson_sampling_via_generation_of_missing_data.md)**

:   提出 Generative Thompson Sampling (TS-Gen)，将上下文老虎机中的不确定性建模为缺失数据而非未知参数，通过生成模型对缺失结果做自回归填充来实现 Thompson 采样，建立了与离线预测损失直接挂钩的遗憾界。

**[Continuous Diffusion Model For Language Modeling](continuous_diffusion_model_for_language_modeling.md)**

:   提出一种面向离散语言建模的连续扩散框架，将离散扩散过程与统计流形上的连续流联系起来，并通过径向对称的 simulation-free 训练机制与降维技巧，显著提升扩散语言模型性能，接近自回归模型。

**[Continuous Uniqueness And Novelty Metrics For Generative Modeling Of Inorganic C](continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)**

:   针对无机晶体生成模型评估中广泛使用的离散距离函数 (StructureMatcher) 的四大缺陷，提出基于 Magpie 指纹（成分）和 AMD 向量（结构）的连续距离函数，实现更可靠的 uniqueness 和 novelty 度量。

**[Coral Disentangling Latent Representations In Longtailed Dif](coral_disentangling_latent_representations_in_longtailed_dif.md)**

:   论文系统分析长尾数据下扩散模型尾部类别生成质量下降的根因，指出 U-Net 瓶颈潜表示发生“头类-尾类子空间重叠”导致特征借用，并提出 CORAL 对比式潜空间对齐正则，显著提升尾类样本的多样性与视觉质量。

**[Coreinforcement Learning For Unified Multimodal Understandin](coreinforcement_learning_for_unified_multimodal_understandin.md)**

:   提出CoRL框架——通过"统一RL→精细RL"两阶段GRPO训练策略，在不依赖额外监督数据的情况下，让统一多模态模型(ULM)的理解和生成能力协同进化，在Janus-Pro-1.5B上取得生成+7%、理解+23%的平均提升。

**[Counterfactual Identifiability Via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)**

:   利用动态最优传输 (dynamic OT) 理论，首次解决了高维多变量 Markovian SCM 中反事实的可辨识性问题——证明 OT flow 机制产生唯一的单调保序反事实传输映射，并扩展至非 Markovian 设置（IV/BC/FC 准则）。

**[Coupling Generative Modeling And An Autoencoder With The Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)**

:   在存在未观测混淆因子的因果推断中，提出将生成模型与自编码器耦合来提升因果桥函数 (causal bridge) 的估计质量——通过共享编码器在处理/控制/结果变量间传递统计强度，并将框架扩展到生存分析。

**[Cross-Fluctuation Phase Transitions Reveal Sampling Dynamics In Diffusion Models](cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)**

:   借鉴统计物理中的涨落理论（fluctuation theory），提出了一种通过 **交叉涨落（cross-fluctuation）** 检测扩散模型采样过程中离散相变的框架，从而在无需重新训练的情况下加速采样、改进条件生成、提升零样本分类和风格迁移。

**[Curly Flow Matching For Learning Non-Gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)**

:   提出 Curly Flow Matching (Curly-FM)，通过设计带有非零参考漂移的 Schrödinger Bridge 问题，使 flow matching 能够学习周期性、旋转性等非梯度场动力学，突破了传统方法只能建模梯度场的限制。

**[Decaflow A Deconfounding Causal Generative Model](decaflow_a_deconfounding_causal_generative_model.md)**

:   提出 DeCaFlow，一个去混淆的因果生成模型，在给定因果图和观测数据的情况下，只需训练一次即可正确估计所有 do-calculus 可识别的因果查询（包括干预和反事实），即使存在隐藏混淆因子。

**[Decomate Leveraging Generative Models For Co-Creative Svg Animation](decomate_leveraging_generative_models_for_co-creative_svg_animation.md)**

:   提出 Decomate 交互系统，利用多模态大语言模型 (MLLM) 将非结构化 SVG 图形自动分解为语义组件，设计师通过自然语言为各组件指定动画行为，系统生成可生产的 HTML/CSS/JS 动画代码，支持迭代协作创作。

**[Deft Decompositional Efficient Finetuning For Texttoimage Mo](deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)**

:   提出DEFT——将权重更新分解为两个可训练矩阵的组合：(1)低秩子空间的正交投影和(2)子空间内的低秩调整，相比LoRA在T2I个性化中CLIP-T从0.341提升到0.361（DreamBench+），在统一模型上实现风格迁移和条件生成的SOTA。

**[Denoising Weak Lensing Mass Maps With Diffusion Model And Generative Adversarial](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)**

:   将扩散模型（DM）应用于弱引力透镜质量图去噪任务，与 GAN（pix2pix）在相同实验设置下进行系统性对比，证明 DM 在训练稳定性、多样本平均鲁棒性和多种统计量重建精度上全面优于 GAN。

**[Detecting Generated Images By Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)**

:   提出一致性验证框架 ConV，利用自然图像流形与生成图像之间的几何差异，通过两个梯度正交的函数实现无需训练的生成图像检测，并引入 Normalizing Flow 增强版 F-ConV 进一步放大流形偏差。

**[Detection And Simulation Of Urban Heat Islands Using A Fine-Tuned Geospatial Fou](detection_and_simulation_of_urban_heat_islands_using_a_fine-tuned_geospatial_fou.md)**

:   提出一套利用微调地理空间基础模型（Granite-GFM）的完整工作流，涵盖城市热岛效应的实证量化、未来气候情景下的温度外推预测，以及通过卫星图像 inpainting 模拟城市绿化降温效果。

**[Dexter Diffusion-Guided Explanations With Textual Reasoning For Vision Models](dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)**

:   提出 DEXTER，一个无需数据的框架，通过优化文本提示驱动扩散模型生成最大化目标分类器激活的图像，再用 LLM 对合成样本进行推理，生成全局性、可读的文本解释，实现模型行为的偏差发现和全局解释。

**[Dico Revitalizing Convnets For Scalable And Efficient Diffus](dico_revitalizing_convnets_for_scalable_and_efficient_diffus.md)**

:   重新发掘卷积网络在扩散模型中的潜力——发现预训练DiT的全局自注意力主要捕获局部模式（冗余），提出用标准ConvNet模块+紧凑通道注意力构建纯卷积扩散模型DiCo，在ImageNet-256上以2.05 FID超越DiT-XL/2且速度快2.7倍。

**[Diff-Icmh Harmonizing Machine And Human Vision In Image Compression With Generat](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)**

:   提出 Diff-ICMH，一种基于扩散模型的生成式图像压缩框架，通过语义一致性损失（SC loss）保持语义完整性，通过标签引导模块（TGM）激活生成先验，以单一编解码器和码流同时服务 10+ 种智能任务和人类视觉感知，无需任何任务特定适配。

**[Diffeye Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned On N](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)**

:   提出 DiffEye，首个基于扩散模型直接利用原始眼动轨迹数据、以自然图像为条件生成连续且多样化眼动轨迹的框架，同时引入对应位置嵌入 (CPE) 对齐注视空间与图像语义空间。

**[Diffusion-Based Electromagnetic Inverse Design Of Scattering Structured Media](diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)**

:   提出基于条件扩散模型的电磁逆设计框架，从目标微分散射截面 (DSCS) 直接生成介电球超表面几何结构，绕过昂贵的迭代优化，并自然处理逆问题的非唯一性，性能优于 CMA-ES 进化优化且速度快数个数量级。

**[Diffusion-Driven Progressive Target Manipulation For Source-Free Domain Adaptati](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)**

:   提出 DPTM 框架，利用潜在扩散模型对不可信目标样本进行语义变换，生成伪目标域并通过渐进式重建机制迭代缩小与真实目标域的差距，在大域偏移场景下比现有 SFDA SOTA 提升高达 18.6%。

**[Diffusion Adaptive Text Embedding For Texttoimage Diffusion](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)**

:   发现T2I扩散模型中固定的text embedding在不同时间步是次优的，提出DATE——在推理时动态更新text embedding以最大化mean predicted image与文本的对齐评分（如CLIP Score/ImageReward），无需训练，可即插即用到任何扩散模型和采样器中，在多概念生成和图像编辑中一致提升text-image对齐。

**[Diffusion Classifiers Understand Compositionality But Condit](diffusion_classifiers_understand_compositionality_but_condit.md)**

:   全面研究零样本扩散分类器在组合理解任务上的判别能力：覆盖3个扩散模型(SD 1.5/2.0/3-m)×10个数据集×30+任务，引入Self-Bench诊断基准（用扩散模型自己生成的图像消除域差异），发现扩散分类器确实理解组合性但受域差距和时间步加权影响——"条件适用"。

**[Diffusion Generative Modeling On Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)**

:   提出在李群**表示空间**（而非李群本身）上构建扩散过程的新理论框架，通过广义分数匹配将非阿贝尔李群的弯曲动力学映射到欧几里得空间中，实现无模拟训练的李群扩散模型，并证明标准分数匹配是其平移群的特例。

**[Diffusion Models Meet Contextual Bandits](diffusion_models_meet_contextual_bandits.md)**

:   将预训练扩散模型作为上下文赌博机 (contextual bandits) 问题中动作参数的表达性先验，提出 diffusion Thompson Sampling (dTS) 算法，通过高效的层次化后验近似实现快速更新与采样，在大动作空间下显著优于传统方法。

**[Distilled Decoding 2 Onestep Sampling Of Image Autoregressiv](distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)**

:   提出 Distilled Decoding 2 (DD2)，通过条件分数蒸馏损失将图像自回归模型压缩为单步生成器，在 ImageNet-256 上 FID 仅从 3.40 增至 5.43，比 DD1 的 one-step 差距缩小 67%，训练加速 12.3×。

**[Dove Efficient One-Step Diffusion Model For Real-World Video Super-Resolution](dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)**

:   提出 DOVE，基于 CogVideoX 预训练视频生成模型，通过两阶段潜空间-像素空间训练策略和高质量 HQ-VSR 数据集实现单步推理的视频超分辨率，比多步扩散方法快 28 倍且性能相当或更优。

**[Dynamic Diffusion Schrödinger Bridge In Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)**

:   提出 Astro-DSB，一种基于 Diffusion Schrödinger Bridge 的天文物理反问题建模方法，直接学习观测量到真实物理分布的概率映射，训练成本仅为条件 DDPM 的 25%，且在分布外（OOD）测试中展现出显著的泛化优势，并成功应用于 Taurus B213 真实观测数据。

**[Editinfinity Image Editing With Binary-Quantized Generative Models](editinfinity_image_editing_with_binary-quantized_generative_models.md)**

:   提出 EditInfinity，首次将经典"图像反演-图像编辑"范式应用于二值量化自回归生成模型 Infinity，利用量化表示可获取精确中间监督的优势实现高精度图像反演，配合分段线性平滑核实现高保真编辑效果，在 PIE-Bench 上全面超越扩散模型基线。

**[Eegrexfernet A Lightweight Gen-Ai Framework For Eeg Subspace Reconstruction Via ](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)**

:   提出 EEGReXferNet，一种轻量级生成式 AI 框架，通过邻域通道感知输入选择、频带特定子窗口卷积编解码、动态滑窗隐空间和参考统计量缩放，在跨被试迁移学习设置下实现 EEG 子空间重建，参数减少约 45%、推理延迟 <1ms，同时保持 PSD 相关性 $\geq 0.95$ 和谱图 RV 系数 $\geq 0.85$。

**[Efficient Rectified Flow For Image Fusion](efficient_rectified_flow_for_image_fusion.md)**

:   提出 RFfusion，首次将 Rectified Flow 引入图像融合任务，实现无需额外训练的单步采样，同时设计面向融合的两阶段 VAE 训练策略，在速度和质量上全面超越现有扩散融合方法。

**[Elucidated Rolling Diffusion Models For Probabilistic Forecasting Of Complex Dyn](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)**

:   提出 ERDM，首次将滚动扩散（Rolling Diffusion）框架与 EDM 的原则性设计（噪声调度、预条件化、Heun 采样器）成功统一，通过渐进噪声调度显式建模不确定性增长，在 Navier-Stokes 和 ERA5 天气预报任务上显著优于自回归 EDM 基线。

**[Emergence And Evolution Of Interpretable Concepts In Diffusi](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)**

:   首次将 Sparse Autoencoders (SAEs) 系统性地应用于多步扩散模型 (Stable Diffusion v1.4)，揭示了图像构图在第一步反向扩散就已涌现、风格概念在中期阶段形成的时间演化规律，并据此设计了时间自适应的因果干预技术。

**[Encoder-Decoder Diffusion Language Models For Efficient Training And Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)**

:   提出 E2D2，一种面向离散扩散语言模型的编码器-解码器架构，通过轻量解码器迭代去噪、大型编码器定期更新表征，同时实现更快推理（~3× vs MDLM）和更高效的 block diffusion 训练（FLOPs 减半）。

**[Energy Loss Functions For Physical Systems](energy_loss_functions_for_physical_systems.md)**

:   提出基于物理能量的损失函数框架，通过反向 KL 散度与玻尔兹曼分布推导出以成对距离为基础的能量差损失，天然满足 SE(d) 不变性，在分子生成和自旋基态预测中显著优于 MSE 和交叉熵损失。

**[Enhancing Diffusion Model Guidance Through Calibration And Regularization](enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)**

:   针对分类器引导扩散模型中分类器过度自信导致梯度消失的问题，提出两类互补方案：(1) Smooth ECE 校准损失微调分类器，FID 改善 ~3%；(2) 基于 f-散度的正则化采样引导（RKL/FKL/JS），无需重训练即在 ImageNet 128×128 上达到 FID 2.13。

**[Entropy Rectifying Guidance For Diffusion And Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)**

:   提出 Entropy Rectifying Guidance (ERG)，通过操控注意力层的 Hopfield 能量景观（温度缩放、步长调整）来获取弱预测信号，替代传统 CFG 中的无条件预测，在文本到图像、类条件和无条件生成中同时提升质量、多样性和一致性。

**[Epistemic Uncertainty For Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)**

:   提出 WePe（Weight Perturbation），通过对预训练视觉大模型（DINOv2）施加权重扰动来估计认识不确定性（epistemic uncertainty），利用自然图像与 AI 生成图像在不确定性空间的差异实现检测，无需训练即可使用。

**[Equivariant Flow Matching For Symmetry-Breaking Bifurcation Problems](equivariant_flow_matching_for_symmetry-breaking_bifurcation_problems.md)**

:   提出等变 flow matching 框架，结合 symmetric coupling 策略，用生成式 AI 建模对称性破缺分岔问题中的多模态概率分布，在物理系统（屈曲梁、Allen-Cahn 方程）上显著优于确定性模型和 VAE。

**[Evaluating The Evaluators Metrics For Compositional Text-To-Image Generation](evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)**

:   系统评估了 12 种文本-图像组合对齐指标与人类判断的一致性，发现没有单一指标在所有组合任务上一致表现最优，VQA 指标并非总是最好的，embedding 类指标（ImageReward、HPS）在特定类别上更强。

**[Evodiff Entropy-Aware Variance Optimized Diffusion Inference](evodiff_entropy-aware_variance_optimized_diffusion_inference.md)**

:   从信息论角度分析扩散模型推理过程，提出通过优化条件方差来减少条件熵的 EVODiff 方法，在不修改模型的前提下显著加速采样并提升生成质量。

**[Evolve To Inspire Novelty Search For Diverse Image Generation](evolve_to_inspire_novelty_search_for_diverse_image_generation.md)**

:   提出 Wander 框架，基于新颖性搜索（novelty search）和 LLM 驱动的 prompt 进化，从单个文本提示出发生成高度多样化的图像集合，在 Vendi Score 上超越现有进化式 prompt 优化基线。

**[Exploring Semantic-Constrained Adversarial Example With Instruction Uncertainty ](exploring_semantic-constrained_adversarial_example_with_instruction_uncertainty_.md)**

:   提出多维度指令不确定性缩减框架 InSUR，通过 ResAdv-DDIM 采样器稳定对抗优化方向、上下文编码的攻击场景约束、以及基于 WordNet 的语义抽象评估，首次实现了从自然语言指令生成 2D/3D 语义约束对抗样本（SemanticAE）。

**[Exploring Variational Graph Autoencoders For Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)**

:   探索变分图自编码器（VGAE）生成合成配电网拓扑的能力，评估四种解码器架构在两个数据集上的表现，揭示 VGAE 在小型同质网络上效果良好但在大型异质网络上面临挑战。

**[Failure Prediction At Runtime For Generative Robot Policies](failure_prediction_at_runtime_for_generative_robot_policies.md)**

:   提出 FIPER 框架，通过结合观测空间的 OOD 检测（RND）和动作空间的不确定性量化（ACE），在生成式机器人策略运行时无需失败数据即可实现早期失败预测。

**[Fairimagen Post-Processing For Bias Mitigation In Text-To-Image Models](fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)**

:   提出 FairImagen 后处理去偏框架，通过在 CLIP prompt 嵌入空间应用 FairPCA 投影去除人口统计信息，结合经验噪声注入和跨人口统计联合去偏，在不重训模型的前提下显著提升文本到图像生成的公平性。

**[Falcon Few-Step Accurate Likelihoods For Continuous Flows](falcon_few-step_accurate_likelihoods_for_continuous_flows.md)**

:   提出 FALCON，通过混合训练目标（flow matching + 平均速度损失 + 可逆性正则化）使连续归一化流在少步采样下仍能提供足够精确的似然估计，从而实现比传统 CNF 快两个数量级的 Boltzmann 采样。

**[Fast Data Attribution For Text-To-Image Models](fast_data_attribution_for_text-to-image_models.md)**

:   将慢而准确的 unlearning-based 数据归因方法蒸馏为一个可快速检索的特征嵌入空间，在 Stable Diffusion 级别模型上实现比现有方法快 2,500× ~ 400,000× 的数据归因。

**[Fast Solvers For Discrete Diffusion Models Theory And Applications Of High-Order](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)**

:   为离散扩散模型推理首次提出高阶数值求解器（θ-RK-2 和 θ-Trapezoidal），在 KL 散度意义下证明二阶收敛，在文本和图像生成任务上以同等计算预算获得更好的样本质量。

**[Ferretnet Efficient Synthetic Image Detection Via Local Pixel Dependencies](ferretnet_efficient_synthetic_image_detection_via_local_pixel_dependencies.md)**

:   基于 Markov Random Field 理论提出局部像素依赖（LPD）特征表示，结合仅 1.1M 参数的轻量 FerretNet 网络，仅在 4 类 ProGAN 数据上训练即在 22 个生成模型上达到 97.1% 平均准确率。

**[Flatten Graphs As Sequences Transformers Are Scalable Graph Generators](flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)**

:   提出 AutoGraph，通过分段欧拉邻域路径（SENT）将图无损展平为 token 序列，直接用 decoder-only Transformer 建模，实现比扩散模型快 100× 的图生成速度，同时在合成和分子基准上达到 SOTA。

**[Flattening Hierarchies With Policy Bootstrapping](flattening_hierarchies_with_policy_bootstrapping.md)**

:   提出 Subgoal Advantage-Weighted Policy Bootstrapping（SAW），通过优势加权的重要性采样对子目标条件策略进行 bootstrapping，将层级 RL 的长距离推理能力蒸馏到一个扁平策略中，无需生成式子目标模型。

**[Flex-Judge Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators](flex-judge_text-only_reasoning_unleashes_zero-shot_multimodal_evaluators.md)**

:   提出 Flex-Judge，仅用 1K 条纯文本推理数据微调多模态大模型，即可零样本泛化到图像/视频/音频/分子等多模态评判任务，性能媲美甚至超越 GPT-4o 等商业 API 和大规模标注训练的专用评估器。

**[Flow Matching Neural Processes](flow_matching_neural_processes.md)**

:   提出 FlowNP，将 flow matching 引入神经过程框架，通过 transformer 预测目标点的流速度场实现对条件分布的并行采样，在 1D GP、图像和气象数据三大基准上全面超越现有 NP 方法。

**[Focalcodec Low-Bitrate Speech Coding Via Focal Modulation Networks](focalcodec_low-bitrate_speech_coding_via_focal_modulation_networks.md)**

:   提出 FocalCodec——基于 Focal Modulation 的低比特率语音编解码器，使用**单个二值码本**将语音压缩至 0.16–0.65 kbps，在语音重合成、语音转换和多项下游任务中达到与多码本 SOTA 方法可比甚至更优的性能。

**[Freqpolicy Efficient Flow-Based Visuomotor Policy Via Frequency Consistency](freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)**

:   首次在 flow-based 视觉运动策略中引入频域一致性约束，利用 DCT 变换将动作块的速度场投影到频域并施加自适应频率分量损失，实现了高质量一步动作生成（93.5 Hz），在仿真和真实机器人任务中均优于现有一步生成方法。

**[From Cradle To Cane A Two-Pass Framework For High-Fidelity Lifespan Face Aging](from_cradle_to_cane_a_two-pass_framework_for_high-fidelity_lifespan_face_aging.md)**

:   提出 Cradle2Cane 两阶段人脸老化框架：第一阶段通过自适应噪声注入（AdaNI）实现精准年龄控制，第二阶段通过 SVR-ArcFace 和 Rotate-CLIP 双身份嵌入（IDEmb）强化身份一致性，在全寿命跨度（0-80岁）人脸老化中实现年龄精度与身份保持的最优平衡。

**[Geneman Generalizable Single-Image 3D Human Reconstruction From Multi-Source Hum](geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)**

:   GeneMAN 提出一种**无需人体参数模型(如 SMPL)**的通用单图 3D 人体重建框架，通过在大规模多源人体数据上训练人体专属的 2D/3D 扩散先验模型，结合几何初始化-雕刻流水线与多空间纹理精炼，实现了对野外图片中不同体型比例、复杂姿态与个人物品的高保真 3D 人体重建。

**[Generative Model Inversion Through The Lens Of The Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)**

:   从流形几何视角揭示生成式模型逆向攻击 (MIA) 的本质是通过将损失梯度投影到生成器切空间实现隐式去噪，提出梯度-流形对齐假说（对齐越高→模型越脆弱）并设计无需训练的 AlignMI 方法在多个 SOTA 攻击上取得一致且显著的提升。

**[Genir Generative Visual Feedback For Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)**

:   提出 GenIR，一种利用文本到图像扩散模型生成"合成视觉反馈"的多轮交互式图像检索框架，将系统对用户查询的理解显式可视化，使用户能直观地识别差异并迭代改进查询，在 Mental Image Retrieval (MIR) 任务上大幅超越纯文本反馈方法。

**[Georemover Removing Objects And Their Causal Visual Artifacts](georemover_removing_objects_and_their_causal_visual_artifacts.md)**

:   提出几何感知的两阶段框架 GeoRemover，将目标移除解耦为几何移除（深度域）与外观渲染（RGB域），通过修改场景几何表示来隐式消除被移除物体的阴影和反射等因果视觉伪影。

**[Gradient Variance Reveals Failure Modes In Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)**

:   本文通过分析 CFM 损失的梯度方差（gradient variance），揭示了 Rectified Flow 在确定性插值下会不可避免地记忆训练配对而非学习最优传输映射，并证明引入随机性（stochastic interpolant）可打破该记忆化通道、恢复泛化能力。

**[Gralora Granular Low-Rank Adaptation For Parameter-Efficient Fine-Tuning](gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)**

:   提出 GraLoRA——将 LoRA 的权重更新矩阵分割为 $k^2$ 个独立子块、每块配独立低秩适配器，在不增加参数量和计算量的前提下将有效秩从 $r$ 提升至 $kr$，解决 LoRA 在高秩下因梯度纠缠导致的性能退化问题，在代码生成上 Pass@1 最高提升 +8.5%。

**[Graph-Based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)**

:   提出基于图神经网络的空间天气神经模拟器，在 Vlasiator 混合 Vlasov 模拟数据上训练，实现确定性和概率性自回归预测近地空间状态，速度比原始模拟快 100 倍以上，并通过隐变量生成集合预报来量化预测不确定性。

**[Graph Diffusion That Can Insert And Delete](graph_diffusion_that_can_insert_and_delete.md)**

:   提出 GrIDDD 模型，首次将离散去噪扩散模型（DDPM）扩展为支持在生成过程中**动态插入和删除图节点**，使分子图的大小可在扩散过程中自适应变化，在性质靶向和分子优化任务上达到或超过现有方法。

**[Grasp2Grasp Vision-Based Dexterous Grasp Translation Via Schrödinger Bridges](grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)**

:   提出将跨手形态的视觉灵巧抓取迁移建模为 Schrödinger Bridge 问题，通过在潜空间中学习得分与流匹配（[SF]²M），并设计物理感知的最优传输代价函数（位姿/接触图/力旋量空间/雅可比可操作性），在无需配对数据的条件下实现不同机械手之间抓取意图的分布级迁移。

**[Gspn-2 Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)**

:   GSPN-2 通过算法-系统联合重设计（单 kernel 融合、紧凑通道传播、共享内存优化），将 GSPN-1 的 2D 空间传播加速最高 40×，在 ImageNet 分类和文本到图像生成中达到 Transformer 级精度且计算成本显著更低。

**[Guided Diffusion Sampling On Function Spaces With Applications To Pdes](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)**

:   提出 **FunDPS（Function-space Diffusion Posterior Sampling）**，在函数空间中训练无条件扩散模型，推理时通过梯度引导实现 plug-and-play 的 PDE 逆问题后验采样；理论上将 Tweedie 公式推广到无穷维 Banach 空间，实验上在 5 个 PDE 任务中仅用 3% 观测即可获得比 DiffusionPDE 平均高 32% 的精度并减少 4 倍采样步数。

**[Guideflow3D Optimization-Guided Rectified Flow For Appearance Transfer](guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)**

:   提出 GuideFlow3D，一种无需训练的 3D 外观迁移框架，通过在预训练 rectified flow 模型的采样过程中交替注入可微引导损失（部件感知外观损失 + 自相似性损失），实现几何差异显著的物体间鲁棒的纹理与几何细节迁移。

**[Hallucination As An Upper Bound A New Perspective On Text-To-Image Evaluation](hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)**

:   提出将文本到图像（T2I）模型中的幻觉定义为**偏差驱动的偏离**，建立了包含属性、关系和物体三类幻觉的分类学，并论证幻觉评估作为提示对齐评估的"上界"，可揭示模型隐藏偏差。

**[Head Pursuit Probing Attention Specialization In Multimodal](head_pursuit_probing_attention_specialization_in_multimodal.md)**

:   用信号处理中的Simultaneous Orthogonal Matching Pursuit (SOMP)算法分解注意力头在unembedding矩阵上的稀疏表示，揭示注意力头的语义特化现象（如政治/国籍/月份/数字等），仅编辑1%的头即可可靠地抑制或增强特定概念——在语言和视觉-语言模型上均验证有效。

**[Hephaestus Mixture Generative Modeling With Energy Guidance For Large-Scale Qos ](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)**

:   提出 Hephaestus 三阶段生成框架（Forge-Morph-Refine），结合预测路径加压算法、能量引导的混合 CVAE 和潜在空间 RL 优化，用于大规模网络 QoS 降级问题的求解。

**[Hierarchical Koopman Diffusion Fast Generation With Interpretable Diffusion Traj](hierarchical_koopman_diffusion_fast_generation_with_interpretable_diffusion_traj.md)**

:   基于 Koopman 算子理论，将扩散模型的非线性去噪动力学提升到线性 Koopman 空间，通过层次化分解实现一步采样，同时保留中间生成状态的可解释性和可控性。

**[High-Order Equivariant Flow Matching For Density Functional Theory Hamiltonian P](high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)**

:   提出 QHFlow，首次将条件 flow matching 引入密度泛函理论（DFT）哈密顿矩阵预测任务，通过高阶 SE(3) 等变向量场和对称性感知先验分布，在 MD17 上将哈密顿预测误差降低 73%，并可作为 SCF 初始化加速 DFT 计算达 54%。

**[Highlighting What Matters Promptable Embeddings For Attribute-Focused Image Retr](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)**

:   提出可提示图像嵌入(Promptable Embeddings)方法，通过在检索时高亮目标视觉属性来提升属性聚焦的文本到图像检索性能，同时构建了COCO-Facet基准数据集。

**[Hollowflow Efficient Sample Likelihood Evaluation Using Hollow Message Passing](hollowflow_efficient_sample_likelihood_evaluation_using_hollow_message_passing.md)**

:   提出HollowFlow框架，通过非回溯图神经网络（NoBGNN）和Hollow消息传递机制强制速度场雅可比矩阵具有块对角结构，将连续归一化流的似然计算反向传播次数从$\mathcal{O}(n)$降至常数$\mathcal{O}(d)$，实现高达$10^2$倍的采样加速。

**[How To Build A Consistency Model Learning Flow Maps Via Self-Distillation](how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)**

:   提出统一的自蒸馏（Self-Distillation）框架来直接学习 flow map（即 consistency model 的一般化形式），通过 tangent condition 将任意蒸馏方案转化为无需预训练教师的直接训练算法，并导出三大算法族（Eulerian / Lagrangian / Progressive），其中 Lagrangian 方法避免了空间梯度和自举引导，训练最稳定、性能最优。

**[Image Super-Resolution With Guarantees Via Conformalized Generative Models](image_super-resolution_with_guarantees_via_conformalized_generative_models.md)**

:   基于共形预测（Conformal Prediction）技术，为生成式图像超分辨率模型构建二值"置信度掩码"，能可靠地标识生成图像中可信赖的区域，并提供严格的统计保证。

**[Imagesentinel Protecting Visual Datasets From Unauthorized Retrieval-Augmented I](imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)**

:   提出 ImageSentinel 框架，通过合成与私有数据集视觉一致的哨兵图像（sentinel images）并绑定随机字符检索键，实现对检索增强图像生成（RAIG）系统未授权使用私有数据集的可靠检测——仅需 3–10 次查询即可达到接近 100% 的 AUC。

**[Improved Training Technique For Shortcut Models](improved_training_technique_for_shortcut_models.md)**

:   针对 Shortcut Models 的五大性能瓶颈（指导累积、固定引导、频率偏差、自一致性偏离、弯曲轨迹），提出 iSM 统一训练框架，通过内禀引导、多级小波损失、缩放最优传输和双 EMA 策略，在 ImageNet 256×256 上实现单步 FID 5.27、四步 FID 2.05 的大幅提升。

**[Improving Posterior Inference Of Galaxy Properties With Image-Based Conditional ](improving_posterior_inference_of_galaxy_properties_with_image-based_conditional_.md)**

:   提出基于条件流匹配（CFM）的框架，将星系图像的形态学信息与测光数据联合建模，显著提升星系物理属性（恒星质量、恒星形成率、金属丰度、尘埃消光等）的后验推断精度。

**[In-Context Edit Enabling Instructional Image Editing With In-Context Generation ](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)**

:   ICEdit 提出一种基于大规模 Diffusion Transformer (DiT) 的上下文编辑范式，通过 in-context prompt + 最小化 LoRA-MoE 微调 + VLM 早期筛选推理时缩放，仅用 0.1% 训练数据即达到 SOTA 编辑性能。

**[Increasing The Utility Of Synthetic Images Through Chamfer Guidance](increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)**

:   提出 Chamfer Guidance——一种免训练的推理时引导方法，利用少量真实样本作为参照，通过 Chamfer 距离同时优化合成图像的质量（fidelity）和多样性（diversity），在 ImageNet-1k 上仅用 32 张真实图片即可达到 97.5% Precision 和 92.7% Coverage，并在下游分类器训练中带来最高 16% 的准确率提升。

**[Inference-Time Scaling For Flow Models Via Stochastic Generation And Rollover Bu](inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)**

:   提出针对 Flow 模型的推理时扩展方法：通过 ODE→SDE 转换引入随机性以启用粒子采样，利用线性→VP 插值变换扩大搜索空间，并设计 Rollover Budget Forcing (RBF) 策略自适应分配计算预算，在组合文本生成图像和数量感知生成任务上显著超越所有现有方法。

**[Infinitystar Unified Spacetime Autoregressive Modeling For V](infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)**

:   提出 InfinityStar，首个能生成工业级 720p 视频的纯离散自回归模型，通过时空金字塔建模统一 T2I/T2V/I2V/交互式长视频生成，VBench 83.74 超越 HunyuanVideo，推理速度比扩散模型快 10-32×。

**[Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)**

:   将连续扩散中经典的 I-MMSE 恒等式推广到离散域，建立 I-MDSE 和 I-MDCE 关系——证明 DSE/DCE 损失不仅是变分上界而是对数似然的**精确分解**，并由此推导出 time-free 公式、条件似然估计和耦合似然比估计器，在 LLaDA 等大模型上验证了低方差和 OOD 检测能力。

**[Information Theoretic Learning For Diffusion Models With Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)**

:   提出将经典 KL 散度-Fisher 信息关系推广到任意各向同性噪声扰动的似然估计框架，结合 warm-start 噪声注入和重要性采样，消除训练-测试差距并实现更紧的似然上界，在 ImageNet 多分辨率上达到 SOTA NLL。

**[Instance-Level Composed Image Retrieval](instance-level_composed_image_retrieval.md)**

:   提出实例级组合图像检索（i-CIR）基准和训练免费方法BASIC，通过独立估计图像和文本查询的相似度并进行乘法融合，在无需训练的情况下在i-CIR和现有CIR数据集上均达到SOTA。

**[Is Artificial Intelligence Generated Image Detection A Solved Problem](is_artificial_intelligence_generated_image_detection_a_solved_problem.md)**

:   提出 AIGIBench 综合基准，通过四大任务（多源泛化、多退化鲁棒性、数据增强敏感性、测试预处理影响）系统评估 11 个 SOTA 检测器，揭示现有 AIGI 检测方法在真实场景下性能严重下降，表明该问题远未解决。

**[Itdpdm Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)**

:   提出 ItDPDM（信息论离散泊松扩散模型），通过泊松噪声信道和泊松重建损失（PRL）实现非负离散数据的精确似然估计，避免了 ELBO 近似和 dequantization，在合成数据及 CIFAR-10 和 MIDI 音乐上取得优于现有离散扩散模型的似然估计。

**[Janus-Pro-R1 Advancing Collaborative Visual Comprehension And Generation Via Rei](janus-pro-r1_advancing_collaborative_visual_comprehension_and_generation_via_rei.md)**

:   提出 Janus-Pro-R1，通过两阶段训练（SFT + RL）实现视觉理解与生成的协同共进，让 MLLM 在文本到图像生成中形成真正的 Chain-of-Thought 并触发 Aha 时刻，在 GenEval 上超越 GPT-4o，同时拓展到图像编辑任务。

**[Klass Kl-Guided Fast Inference In Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)**

:   提出 KLASS（KL-Adaptive Stability Sampling），一种无需训练的采样方法，利用 token 级别的 KL 散度和置信度来识别稳定 token 并行解码，在掩码扩散模型上实现最高 2.78× 加速且不损失甚至提升生成质量。

**[Knowledge Distillation Detection For Open-Weights Models](knowledge_distillation_detection_for_open-weights_models.md)**

:   提出知识蒸馏检测任务，通过无数据输入合成和统计评分框架，判断一个开放权重的学生模型是否由特定教师模型蒸馏而来。

**[Kuramoto Orientation Diffusion Models](kuramoto_orientation_diffusion_models.md)**

:   将生物系统中的Kuramoto同步动力学引入score-based生成模型，在周期域上构建前向同步/反向去同步的扩散框架，对指纹、纹理等方向密集数据实现显著优于标准扩散模型的生成质量，同时在CIFAR-10上保持竞争力。

**[Large-Scale Training Data Attribution For Music Generative Models Via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)**

:   将基于机器遗忘（machine unlearning）的训练数据归因方法应用于大规模文本到音乐扩散模型（115K 音轨），通过网格搜索找到最优超参数配置，并与非反事实方法对比，验证了 unlearning-based TDA 在音乐生成领域的可行性。

**[Latent Zoning Network A Unified Principle For Generative Modeling Representation](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)**

:   提出 Latent Zoning Network (LZN)——一种通过共享高斯潜在空间将生成建模、表征学习和分类统一在同一框架下的方法，每种数据类型配备编码器-解码器对将样本映射到不相交的潜在区域，仅依赖"潜在计算"和"潜在对齐"两个原子操作即可支持多种 ML 任务，并在 CIFAR10 上将无条件生成 FID 从 2.76 降至 2.59，在 ImageNet 线性分类上超越 SimCLR。

**[Leapfactual Reliable Visual Counterfactual Explanation Using Conditional Flow Ma](leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)**

:   提出LeapFactual，一种基于条件流匹配(CFM)的反事实解释算法，通过"起飞-降落"(Leap)机制在扁平化和结构化潜在空间之间建立桥梁，生成可靠且分布内的反事实样本，即使在学习决策边界与真实边界不一致时也能有效工作。

**[Learnable Sampler Distillation For Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)**

:   提出LSD和LSD+方法，通过蒸馏将高保真教师采样器的中间分数轨迹知识迁移给少步数学生采样器，以可学习的采样系数和非均匀时间调度实现离散扩散模型的高效高质量采样。

**[Learning Interpretable Features In Audio Latent Spaces Via Sparse Autoencoders](learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)**

:   提出一种通过稀疏自编码器（SAE）从音频生成模型的潜空间中提取可解释特征的框架，通过线性探针将 SAE 特征映射到人类可理解的声学概念（音高、振幅、音色），实现对音频生成过程的可控操作和可视化分析。

**[Learning To Integrate Diffusion Odes By Averaging The Derivatives](learning_to_integrate_diffusion_odes_by_averaging_the_derivatives.md)**

:   提出"割线损失"(Secant Losses)家族，通过蒙特卡洛积分和Picard迭代学习扩散ODE的积分，将扩散模型的切线逐步延展为割线，在训练稳定性和少步推理之间取得优异平衡。

**[Linear Differential Vision Transformer Learning Visual Contrasts Via Pairwise Di](linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)**

:   提出 Visual-Contrast Attention (VCA)，通过空间池化生成紧凑的正负视觉对比 token 并进行差分交互，将自注意力复杂度从 $O(N^2C)$ 降至 $O(NnC)$（$n \ll N$），同时在图像分类和生成任务上均获得显著提升。

**[Lineas End-To-End Learning Of Activation Steering With A Distributional Loss](lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)**

:   提出 LinEAS（Linear End-to-end Activation Steering），通过端到端优化跨层仿射变换映射，利用 1D Wasserstein 分布损失进行全局激活值对齐，仅需 32 个无配对样本即可高效控制 LLM 毒性和 T2I 模型概念生成。

**[Llm Meets Diffusion A Hybrid Framework For Crystal Material Generation](llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)**

:   提出CrysLLMGen混合框架，结合LLM擅长离散原子类型预测和扩散模型擅长连续坐标/晶格参数建模的互补优势，在晶体材料生成任务中同时实现高结构有效性和组成有效性。

**[Magcache Fast Video Generation With Magnitudeaware Cache](magcache_fast_video_generation_with_magnitudeaware_cache.md)**

:   发现视频扩散模型中连续时间步残差输出的幅度比(magnitude ratio)遵循统一的单调递减规律（跨模型、跨prompt稳定），提出MagCache基于此规律自适应跳过冗余时间步并复用缓存，仅需1个样本校准即可在Open-Sora/CogVideoX/Wan 2.1/HunyuanVideo上实现2.1-2.68×加速，视觉保真度全面超越现有方法。

**[Mge-Ldm Joint Latent Diffusion For Simultaneous Music Generation And Source Extr](mge-ldm_joint_latent_diffusion_for_simultaneous_music_generation_and_source_extr.md)**

:   提出 MGE-LDM，首个在统一的潜在扩散框架中同时实现音乐混合生成、部分生成（源补全）和文本驱动任意源提取的模型，通过联合建模混合-子混合-源三元组并利用扩散修复（inpainting）实现各任务。

**[Mind-The-Glitch Visual Correspondence For Detecting Inconsistencies In Subject-D](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)**

:   提出从预训练扩散模型骨干网络中解耦语义特征和视觉特征的框架，实现视觉对应匹配，并基于此提出 Visual Semantic Matching (VSM) 度量，首次同时支持主体驱动图像生成中视觉不一致性的**量化和空间定位**。

**[Mitigating Intra- And Inter-Modal Forgetting In Continual Learning Of Unified Mu](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)**

:   提出Modality-Decoupled Experts (MoDE)，通过将文本和图像的适配器解耦为独立的T-MoE和V-Adapter子空间，配合知识蒸馏，在统一多模态生成模型的持续指令微调中同时缓解模态内遗忘和模态间遗忘。

**[Mitigating Sexual Content Generation Via Embedding Distortion In Text-Conditione](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)**

:   提出Distorting Embedding Space (DES)，一种基于文本编码器的防御框架，通过将不安全嵌入变换到安全区域、保持安全嵌入不变、中和"裸露"语义三管齐下，在FLUX.1和SD v1.5上实现SOTA的性内容缓解效果（ASR分别降至9.47%和0.52%），同时保持良好的良性图像质量。

**[Mmada Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)**

:   提出 MMaDA，首个在统一离散扩散架构下同时实现文本推理、多模态理解和文本到图像生成的多模态基础模型，通过混合长 CoT 微调和 UniGRPO 强化学习算法弥合了扩散模型预训练与后训练之间的鸿沟。

**[Mmg Mutual Information Estimation Via The Mmse Gap In Diffusion](mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)**

:   利用扩散模型的信息论公式，证明互信息等于条件与无条件去噪 MMSE 之间的差值在所有信噪比上的积分的一半，提出 MMG 估计器，结合自适应重要性采样和正交原理显著提升估计精度和稳定性。

**[Model-Guided Dual-Role Alignment For High-Fidelity Open-Domain Video-To-Audio Ge](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)**

:   提出MGAudio，首个采用模型引导(MG)训练替代无分类器引导(CFG)的视频到音频生成框架，结合双角色音视频编码器（同时用于条件注入和特征对齐），以131M参数在VGGSound上实现SOTA（FAD=0.40），且仅用10%数据即可超越多数方法。

**[Moment- And Power-Spectrum-Based Gaussianity Regularization For Text-To-Image Mo](moment-_and_power-spectrum-based_gaussianity_regularization_for_text-to-image_mo.md)**

:   提出统一的标准高斯性正则化框架，结合空间域的矩(moment)匹配和频谱域的功率谱(power spectrum)匹配，将KL散度、峰度、范数等现有正则化方法统一为特殊情况，并以$\mathcal{O}(D\log D)$复杂度实现了PRNO的$\mathcal{O}(D^2)$等价效果，在文本到图像模型的reward alignment任务中显著优于所有基线。

**[Multimodal Generative Flows For Lhc Jets](multimodal_generative_flows_for_lhc_jets.md)**

:   提出基于 Transformer 的多模态流匹配框架（MMF），将连续流匹配与连续时间马尔可夫跳跃桥联合建模，实现对 LHC 喷注中粒子运动学（连续）和 flavor 量子数（离散）的统一生成。

**[Neural Entropy](neural_entropy.md)**

:   本文通过扩散模型的范式探索深度学习与信息论的联系，引入"神经熵"度量来量化扩散过程中存储在神经网络里的信息量，揭示了图像扩散模型对结构化数据具有极高的压缩效率。

**[Next Semantic Scale Prediction Via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)**

:   提出 HDLM（Hierarchical Diffusion Language Model），通过在 clean token 和 mask token 之间引入具有粗粒度语义的聚类 token 中间层级，实现"下一语义尺度预测"的离散扩散语言建模，推导闭式 ELBO，在 OpenWebText 上困惑度一致优于 MDLM/GIDD，随机扰动后生成困惑度降低 62%。

**[Non-Asymptotic Analysis Of Data Augmentation For Precision Matrix Estimation](non-asymptotic_analysis_of_data_augmentation_for_precision_matrix_estimation.md)**

:   本文从非渐近角度分析了高维精度矩阵（逆协方差矩阵）估计中数据增强（DA）的效果，建立了线性收缩估计器和 DA 估计器的二次误差集中界，并引入了广义预解矩阵的新型确定性等价工具。

**[Non-Markovian Discrete Diffusion With Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)**

:   提出CaDDi框架，通过非马尔可夫离散扩散过程让每步去噪都能访问完整生成轨迹，并将其统一到因果语言模型架构中，使预训练LLM可直接复用为离散扩散模型。

**[Npn Non-Linear Projections Of The Null-Space For Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)**

:   提出非线性零空间投影 (NPN)——一种新型正则化策略，训练神经网络从观测中预测信号在感知矩阵零空间低维子空间上的投影系数，将此作为"看不见的特征"的先验约束，可灵活嵌入 PnP、展开网络、DIP 和扩散模型等多种重建框架，理论证明了 PnP 算法中的收敛加速。

**[Obclip Oblivious Cloud-Device Hybrid Image Generation With Privacy Preservation](obclip_oblivious_cloud-device_hybrid_image_generation_with_privacy_preservation.md)**

:   提出 ObCLIP，一种遗忘式云-端混合图像生成方案：将用户 prompt 扩展为一组仅在敏感属性（性别、种族等）上不同的候选 prompt，云端处理所有候选的早期去噪步骤而无法识别真实 prompt，客户端选择正确的中间潜变量完成剩余去噪，同时通过时间和批次冗余加速将额外开销降至 4.4~7.6 倍以下。

**[Omnicast A Masked Latent Diffusion Model For Weather Forecasting Across Time Sca](omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)**

:   提出 OmniCast，一种结合掩码生成框架和潜在扩散模型的天气预报方法，通过联合生成未来天气序列（而非自回归迭代）来缓解误差累积，在次季节至季节（S2S）尺度达到 SOTA 性能，同时在中期预报上保持竞争力且推理速度快 10-20 倍。

**[Omnisync Towards Universal Lip Synchronization Via Diffusion](omnisync_towards_universal_lip_synchronization_via_diffusion.md)**

:   OmniSync提出了一种基于Diffusion Transformer的通用唇形同步框架，通过无掩码训练范式、基于Flow Matching的渐进噪声初始化和动态时空CFG三大创新，在真实视频和AI生成视频上都大幅超越先前方法，尤其在风格化角色的唇形同步上达到87.78%成功率（之前最佳67.78%）。

**[Omnivcus Feedforward Subject-Driven Video Customization With Multimodal Control ](omnivcus_feedforward_subject-driven_video_customization_with_multimodal_control_.md)**

:   OmniVCus 提出了一个前馈式 DiT 框架，通过数据构建流水线 VideoCus-Factory 和两种嵌入机制（Lottery Embedding 和 Temporally Aligned Embedding），实现了多主体、多模态控制条件下的视频定制生成，在身份保持和可控性上显著超越 SOTA。

**[On Optimal Steering To Achieve Exact Fairness](on_optimal_steering_to_achieve_exact_fairness.md)**

:   本文定义了"理想分布"——使任意代价敏感风险下的 Bayes 最优分类器都满足精确公平性的数据分布，并提出通过 KL 散度最小化寻找最近理想分布的优化框架，为公平预处理和 LLM 表示引导提供了可证明的公平性保证。

**[On The Emergence Of Linear Analogies In Word Embeddings](on_the_emergence_of_linear_analogies_in_word_embeddings.md)**

:   提出一个基于二值语义属性的词共现生成模型，解析性地证明了词嵌入中线性类比结构（如 $W_{\text{king}} - W_{\text{man}} + W_{\text{woman}} \approx W_{\text{queen}}$）的涌现机制，统一解释了已知的四个关键观测现象。

**[On The Relation Between Rectified Flows And Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)**

:   本文深入研究了 rectified flow（流匹配）与最优传输之间的理论关系，通过构造多个反例证明了此前文献中关于"梯度约束的 rectified flow 可以渐近收敛到最优传输"的等价性声明并不成立，需要比已知条件更强的假设才能保证两者的等价关系。

**[One Stone With Two Birds A Null-Text-Null Frequency-Aware Diffusion Models For T](one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)**

:   提出NTN-Diff频率感知扩散模型，通过将语义一致性问题分解为中频和低频频带各自的一致性任务，利用"空文本-文本-空文本"三阶段去噪策略，同时解决文本引导图像修复中的未遮盖区域保持和遮盖/未遮盖区域语义一致性两大挑战。

**[Orient Anything V2 Unifying Orientation And Rotation Understanding](orient_anything_v2_unifying_orientation_and_rotation_understanding.md)**

:   Orient Anything V2 通过可扩展的合成数据引擎、对称感知的周期分布目标和多帧架构，统一了物体3D方向和旋转理解，在方向估计、6DoF位姿估计和对称性识别三个任务上均达到 zero-shot SOTA。

**[Osmgen Highly Controllable Satellite Image Synthesis Using Openstreetmap Data](osmgen_highly_controllable_satellite_image_synthesis_using_openstreetmap_data.md)**

:   OSMGen 直接从 OSM JSON 数据（矢量几何、语义标签、位置和时间信息）合成高保真卫星图像，并通过 DDIM 反演生成一致的前后对比图像对，支持城市变化模拟和数据增强。

**[Overt A Benchmark For Over-Refusal Evaluation On Text-To-Image Models](overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)**

:   构建了首个大规模文生图模型过度拒绝评估基准 OVERT（4600条良性提示 + 1785条有害提示，覆盖9个安全类别），系统评估了5个主流 T2I 模型的过度拒绝行为，揭示了安全与效用之间的强相关权衡关系。

**[Pairwise Optimal Transports For Training All-To-All Flow-Based Condition Transfe](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)**

:   提出A2A-FM方法，通过一种新颖的代价函数在FlowMatching框架中同时学习所有条件分布对之间的最优传输映射，理论证明在无限样本极限下收敛至逐对最优传输，尤其适用于连续条件变量的非分组数据场景。

**[Panel-By-Panel Souls A Performative Workflow For Expressive Faces In Ai-Assisted](panel-by-panel_souls_a_performative_workflow_for_expressive_faces_in_ai-assisted.md)**

:   提出一种双混合流水线工作流，通过结合自动人脸检测与手动框选、以及表演性视频输入与精细滑块控制，帮助漫画艺术家在AI生成的漫画面板中注入细腻的面部表情。

**[Perturb A Model Not An Image Towards Robust Privacy Protection Via Anti-Personal](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)**

:   提出Anti-Personalized Diffusion Model (APDM)，首次将隐私保护从数据级（图像扰动）转移到模型级（参数更新），通过Direct Protective Optimization损失和Learning to Protect双路径优化策略，鲁棒地阻止扩散模型对特定主体的个性化，同时保持模型对其他主体的生成和个性化能力。

**[Physctrl Generative Physics For Controllable And Physicsgrou](physctrl_generative_physics_for_controllable_and_physicsgrou.md)**

:   提出 PhysCtrl，通过生成式物理网络学习 4 种材质（弹性体、沙子、橡皮泥、刚体）的物理动力学分布，以 3D 点轨迹表示物理运动，结合 I2V 模型实现物理参数和力可控的视频生成。

**[Physics-Constrained Flow Matching Sampling Generative Models With Hard Constrain](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)**

:   提出 Physics-Constrained Flow Matching (PCFM)，一种零样本推理框架，通过在预训练流匹配模型的采样过程中交替执行前向投射、OT 插值反向更新和松弛惩罚校正，实现任意非线性等式约束的精确满足（达到机器精度），在含激波和间断的 PDE 问题上相比基线方法提升高达 99.5%。

**[Physics-Driven Spatiotemporal Modeling For Ai-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)**

:   提出基于物理守恒定律的AI生成视频检测范式，定义归一化时空梯度（NSG）统计量来捕获空间概率梯度与时间密度变化的比率，利用预训练扩散模型估计NSG并通过MMD进行检测，在Recall上超越SOTA 16%、F1超越10.75%。

**[Pid-Controlled Langevin Dynamics For Faster Sampling Of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)**

:   将 PID 控制理论引入 Langevin 动力学采样，利用梯度历史（积分项）提供动量穿越能量壁垒、利用梯度趋势（微分项）抑制振荡实现快速稳定收敛，无需额外训练即可在 SGM 和 EBM 上实现 10 倍以上采样加速。

**[Pixperfect Seamless Latent Diffusion Local Editing With Discriminative Pixel-Spa](pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)**

:   提出 PixPerfect，一个通用的像素级精修框架，通过判别性像素空间损失和全面的伪影模拟管线，消除潜在扩散模型局部编辑中的色差、纹理不匹配和可见接缝，在修复、目标移除和插入任务上大幅提升视觉保真度。

**[Pragmatic Heterogeneous Collaborative Perception Via Generative Communication Me](pragmatic_heterogeneous_collaborative_perception_via_generative_communication_me.md)**

:   提出GenComm——一种基于生成式通信机制的异构多智能体协作感知方法，通过空间消息提取和条件扩散模型在ego端生成对齐的协作者特征，无需修改原始网络即可以极低代价接纳新异构智能体。

**[Preconditioned Langevin Dynamics With Score-Based Generative Models For Infinite](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)**

:   在无穷维 Hilbert 空间中严格分析了分数生成模型 (SGM) 驱动的 Langevin 后验采样器，首次推导出依赖分数近似误差的收敛界，并发现了同时依赖前向算子和分数误差的最优预条件器形式，保证所有后验模态的均匀收敛速率。

**[Predictive Feature Caching For Training-Free Acceleration Of Molecular Geometry ](predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)**

:   将图像领域的预测式特征缓存（predictive feature caching）策略迁移到分子几何生成领域，利用采样轨迹中隐藏状态的时间平滑性，实现免训练的2-3倍推理加速，且与其他优化手段组合可达7倍加速。

**[Preventing Shortcuts In Adapter Training Via Providing The Shortcuts](preventing_shortcuts_in_adapter_training_via_providing_the_shortcuts.md)**

:   提出Shortcut-Rerouted Adapter Training，通过在adapter训练过程中主动提供confounding因素的专用通路（如LoRA吸收分布偏移、ControlNet吸收姿态/表情），使adapter只学习目标属性（如身份），推理时移除辅助模块即可获得去纠缠的适配器。

**[Progressive Inference-Time Annealing Of Diffusion Models For Sampling From Boltz](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)**

:   提出 PITA（Progressive Inference-Time Annealing），一种结合温度退火与扩散平滑两种互补插值策略的框架，通过在高温下训练初始扩散模型，然后利用新颖的 Feynman-Kac PDE 与 SMC 重采样在推理时降温生成低温样本，逐步训练一系列扩散模型直达目标温度，首次实现了对丙氨酸二肽和三肽的笛卡尔坐标下平衡态采样。

**[Prompt-Based Safety Guidance Is Ineffective For Unlearned Text-To-Image Diffusio](prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)**

:   本文发现训练式概念遗忘（unlearning）与免训练安全引导（negative prompt guidance）两种安全方法组合后效果反而下降，提出用概念反演（Concept Inversion）获得的隐式负向嵌入替换显式负向提示，有效恢复了免训练方法在遗忘模型上的防御能力。

**[Psi-Sampler Initial Particle Sampling For Smc-Based Inference-Time Reward Alignm](psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)**

:   提出Ψ-Sampler框架，在SMC（序贯蒙特卡洛）推理时奖励对齐中引入基于pCNL（预条件Crank-Nicolson Langevin）算法的初始粒子采样，从奖励感知的后验分布初始化粒子，显著提升布局生成、数量感知生成和美学偏好生成的对齐效果。

**[Rare Text Semantics Were Always There In Your Diffusion Transformer](rare_text_semantics_were_always_there_in_your_diffusion_transformer.md)**

:   发现通过在 MM-DiT 的联合注意力块前对文本 token 嵌入进行方差放大,即可在无需额外训练或外部模块的情况下让扩散模型呈现稀有文本语义。

**[Real-Time Execution Of Action Chunking Flow Policies](real-time_execution_of_action_chunking_flow_policies.md)**

:   提出 Real-Time Chunking (RTC)，将异步动作分块执行建模为修复（inpainting）问题，通过冻结已执行动作并"修复"其余部分，实现扩散/流策略的实时平滑执行，无需重新训练。

**[Recurrent Memory For Online Interdomain Gaussian Processes](recurrent_memory_for_online_interdomain_gaussian_processes.md)**

:   提出 OHSVGP（Online HiPPO Sparse Variational Gaussian Process），将深度学习中的 HiPPO（高阶多项式投影算子）框架引入稀疏变分高斯过程中作为跨域诱导变量，利用时变正交多项式基函数实现在线学习中的长期记忆保持，核矩阵可通过 ODE 递推高效更新。

**[Reinforcing The Diffusion Chain Of Lateral Thought With Diffusion Language Model](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)**

:   提出扩散横向思维链（DCoLT），将扩散语言模型逆向过程中的每个中间步视为潜在"思考"动作，通过基于最终结果的强化学习优化整条推理轨迹，在SEDD和LLaDA两种扩散语言模型上实现了数学和代码生成的SOTA表现。

**[Remasking Discrete Diffusion Models With Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)**

:   提出 ReMDM 采样器，通过在生成过程中允许已解码 token 被重新掩码（remask），赋予离散掩码扩散模型迭代纠错能力，实现推理时计算缩放，在文本、图像和分子设计任务上显著提升采样质量。

**[Repldm Reprogramming Pretrained Latent Diffusion Models For High-Quality High-Ef](repldm_reprogramming_pretrained_latent_diffusion_models_for_high-quality_high-ef.md)**

:   提出 RepLDM 重编程框架,通过注意力引导阶段和渐进上采样两个阶段,让预训练的潜在扩散模型无需重训练即可生成高质量高分辨率图像,同时大幅提升效率。

**[Respodiff Dual-Module Bottleneck Transformation For Responsible Faithful T2I Gen](respodiff_dual-module_bottleneck_transformation_for_responsible_faithful_t2i_gen.md)**

:   提出RespoDiff框架，在扩散模型UNet的瓶颈层引入双模块可学习变换——负责任概念对齐模块(RAM)和语义对齐模块(SAM)，通过分数匹配目标实现公平和安全的文本到图像生成，同时保持图像质量和语义忠实度。

**[Riemannian Consistency Model](riemannian_consistency_model.md)**

:   首次将一致性模型（Consistency Model）扩展到黎曼流形上，利用指数映射参数化和协变导数推导出离散和连续时间 RCM 目标函数，实现在球面、平坦环面和 SO(3) 等非欧几何上的高质量少步生成。

**[Rlvr-World Training World Models With Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)**

:   提出 RLVR-World 框架，将强化学习可验证奖励（RLVR）范式拓展到世界模型训练，通过将目标度量（如预测准确率、感知质量）作为可验证奖励直接优化，在语言和视频两类世界模型上取得显著提升。

**[Rlzero Direct Policy Inference From Language Without In-Domain Supervision](rlzero_direct_policy_inference_from_language_without_in-domain_supervision.md)**

:   提出RLZero框架，通过"想象-投影-模仿"三步流程，利用视频生成模型将自然语言指令转化为观测序列，再由无监督预训练的RL智能体即时模仿，实现零样本语言到行为的生成。

**[Robustness In Both Domains Clip Needs A Robust Text Encoder](robustness_in_both_domains_clip_needs_a_robust_text_encoder.md)**

:   提出 LEAF (Levenshtein Efficient Adversarial Finetuning)，首个针对 CLIP 文本编码器的对抗微调方法，在字符级文本扰动下显著提升零样本分类、文本-图像检索和图像生成的鲁棒性，同时保持图像域性能。

**[Safe And Stable Control Via Lyapunov-Guided Diffusion Models](safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)**

:   提出 S²Diff，一个基于模型的扩散规划框架，利用控制 Lyapunov 屏障函数（CLBF）引导扩散采样生成轨迹级控制策略，无需控制仿射假设与二次规划，在多种非线性动力系统上同时保证安全性和稳定性，平均安全率达 98.75%。

**[Safesora Safe Texttovideo Generation Via Graphical Watermark](safesora_safe_texttovideo_generation_via_graphical_watermark.md)**

:   Safe-Sora 首次将**图形水印**（如logo图像）直接嵌入到视频生成管线中，通过分层粗到细自适应匹配将水印patch分配到视觉最相似的帧和区域，并设计3D小波变换增强Mamba架构实现时空融合，在视频质量（FVD 3.77 vs 次优154.35）和水印保真度上大幅超越所有基线。

**[Sao-Instruct Free-Form Audio Editing Using Natural Language Instructions](sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)**

:   提出SAO-Instruct，首个支持完全自由格式自然语言指令的音频编辑模型，通过Prompt-to-Prompt、DDPM反演和手动编辑三条流水线构建编辑三元组训练数据，微调Stable Audio Open实现保持上下文一致的定向音频修改。

**[Scalable Explainable And Provably Robust Anomaly Detection With One-Step Flow Ma](scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)**

:   提出 TCCM（Time-Conditioned Contraction Matching），一种受 flow matching 启发的表格数据半监督异常检测方法，通过学习将正常数据收缩到原点的时间条件速度场，仅需单步前向推理即可计算异常分数，在 ADBench 47 个数据集上取得 AUROC 和 AUPRC 双第一，推理速度比 DTE 快 1573 倍。

**[Scalediff Higher-Resolution Image Synthesis Via Efficient And Model-Agnostic Dif](scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)**

:   提出 ScaleDiff 框架，通过 Neighborhood Patch Attention (NPA) 消除传统 patch 方法中的重叠计算冗余，结合潜空间频率混合 (LFM) 和结构引导 (SG)，在无需额外训练的前提下将预训练扩散模型扩展到高分辨率（如 4096²），在 U-Net 和 DiT 架构上均实现了 training-free 方法中的 SOTA 质量和显著的推理加速（相比 DemoFusion 快 8.9 倍）。

**[Scaling Can Lead To Compositional Generalization](scaling_can_lead_to_compositional_generalization.md)**

:   证明标准 MLP 通过简单地扩大数据量和模型规模即可实现组合泛化（compositional generalization），无需显式的模块化架构设计，并发现组合泛化成功时任务成分可从隐藏激活中线性解码。

**[Scaling Diffusion Transformers Efficiently Via Μp](scaling_diffusion_transformers_efficiently_via_μp.md)**

:   将 Maximal Update Parametrization (μP) 从标准 Transformer 推广到扩散 Transformer（DiT、PixArt-α、MMDiT 等），证明其超参数可从小模型稳定迁移到大模型，显著降低大规模扩散模型的调参成本。

**[Scaling Offline Rl Via Efficient And Expressive Shortcut Models](scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)**

:   提出 SORL，利用 shortcut models 的自一致性实现离线 RL 中高效一阶段训练与可变推理步数的策略优化，同时支持推理时的顺序和并行扩展。

**[Scenedecorator Towards Scene-Oriented Story Generation With Scene Planning And S](scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)**

:   SceneDecorator 提出了一个无需训练的框架，通过 VLM 引导的场景规划（global-to-local）和长期场景共享注意力机制，首次系统性地解决了故事生成中的场景规划和场景一致性问题，在场景对齐和一致性指标上显著优于现有方法。

**[Scenedesigner Controllable Multi-Object Image Generation With 9-Dof Pose Manipul](scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)**

:   SceneDesigner 提出了一种基于 CNOCS 地图表示和两阶段强化学习训练的方法，首次实现了多物体 9D 姿态（位置、大小、朝向）的精确控制，在图像生成的可控性和质量上显著超越现有方法。

**[Schrödinger Bridge Matching For Tree-Structured Costs And Entropic Wasserstein B](schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)**

:   将Iterative Markovian Fitting (IMF)程序推广到树结构Schrödinger Bridge问题，提出TreeDSBM算法，在Wasserstein重心计算中将IMF迭代与不动点迭代优雅合并，仅需廉价的bridge-matching步骤即可高效求解。

**[Score-Informed Neural Operator For Enhancing Ordering-Based Causal Discovery](score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)**

:   提出 SciNO（Score-informed Neural Operator），一种在光滑函数空间中设计的概率生成模型，稳定近似 log-密度 Hessian 对角以提升排序式因果发现，合成图上 order divergence 降低 42.7%，真实数据降低 31.5%。

**[Self Forcing Bridging The Train-Test Gap In Autoregressive Video Diffusion](self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)**

:   提出 Self Forcing 训练范式，通过在训练时执行自回归自展开（self-rollout）并使用整体视频级分布匹配损失（DMD/SiD/GAN），消除了 Teacher Forcing 和 Diffusion Forcing 中训练-推理分布不匹配导致的暴露偏差问题，基于 Wan2.1-1.3B 实现了单 GPU 上 17 FPS 实时流式视频生成，同时质量匹敌甚至超越慢几十倍的双向扩散模型。

**[Semantic Surgery Zero-Shot Concept Erasure In Diffusion Models](semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)**

:   提出Semantic Surgery，一种无需重训练的零样本推理时概念擦除框架，通过在扩散过程之前对文本嵌入进行校准向量减法，结合Co-Occurrence Encoding处理多概念擦除和视觉反馈环路解决潜在概念持久性问题，在物体/NSFW/风格/名人擦除任务上全面超越SOTA。

**[Shallow Diffuse Robust And Invisible Watermarking Through Low-Dimensional Subspa](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)**

:   提出 Shallow Diffuse，一种利用扩散模型后验均值预测器（PMP）的局部线性性和 Jacobian 低秩性，在扩散过程中间时间步嵌入水印的方法，实现了水印与生成过程的解耦，首次在服务端和用户端两种场景下同时保证了高一致性和高鲁棒性。

**[Shortcutting Pre-Trained Flow Matching Diffusion Models Is Almost Free Lunch](shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)**

:   提出SCFM（ShortCutting Flow Matching），一种超高效的后训练蒸馏方法，通过速度场自蒸馏将预训练flow matching模型（如12B参数的Flux）压缩为3步采样器，仅需不到1个A100-Day，无需步长嵌入或对抗蒸馏。

**[Show-O2 Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)**

:   提出 Show-o2，一种基于自回归建模和 Flow Matching 的原生统一多模态模型，通过双路径空间（时间）融合在 3D 因果 VAE 空间中构建统一视觉表示，实现跨文本、图像、视频的多模态理解与生成，并设计两阶段训练策略有效保留语言知识。

**[Sparsedit Token Sparsification For Efficient Diffusion Transformer](sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)**

:   提出 SparseDiT，通过空间维度的三段式架构（底层 Poolingformer + 中层 Sparse-Dense Token Module + 顶层全密度处理）和时间维度的动态剪枝率策略，在 DiT-XL 512×512 上实现 55% FLOPs 减少和 175% 推理速度提升，FID 仅增加 0.09，并成功扩展到视频生成和文本到图像生成任务。

**[Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)**

:   提出 SGDD（Split Gibbs Discrete Diffusion），一种基于分裂 Gibbs 采样原理的即插即用离散扩散后验采样算法，通过引入辅助变量和基于 Hamming 距离的正则化势函数，将后验采样分解为似然采样步和先验采样步交替进行，在 DNA 序列设计、离散图像逆问题和音乐填充等任务上大幅超越基线。

**[Splitflow Flow Decomposition For Inversion-Free Text-To-Image Editing](splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)**

:   提出 SplitFlow，将目标 prompt 语义分解为多个子 prompt，为每个子 prompt 计算独立的编辑流，再通过投影和自适应聚合机制组合成统一编辑轨迹，解决梯度纠缠问题，在无需反转的前提下实现更高保真度和可编辑性的文本引导图像编辑。

**[Stableguard Towards Unified Copyright Protection And Tamper Localization In Late](stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)**

:   提出StableGuard，将全局二值水印嵌入LDM生成流程中（通过MPW-VAE），并利用水印扰动模式的变化实现篡改定位（通过MoE-GFN），首次实现端到端的版权保护与篡改检测统一框架。

**[State-Covering Trajectory Stitching For Diffusion Planners](state-covering_trajectory_stitching_for_diffusion_planners.md)**

:   提出 SCoTS（State-Covering Trajectory Stitching），一种无需奖励信号的轨迹增强框架，通过在时间距离保持的潜空间中迭代拼接短轨迹片段，系统性地扩展状态空间覆盖，显著提升扩散规划器在长时域、分布外任务上的泛化能力。

**[Stella Subspace Learning In Low-Rank Adaptation Using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)**

:   提出StelLA，通过将LoRA的适配矩阵分解为 $USV^\top$ 三因子形式，并将 $U$、$V$ 约束在Stiefel流形上进行黎曼优化，实现训练过程中对低秩子空间的显式学习，在多个下游任务上一致超越现有LoRA变体。

**[System-Embedded Diffusion Bridge Models](system-embedded_diffusion_bridge_models.md)**

:   提出System-embedded Diffusion Bridge Models（SDB），将已知的线性测量系统直接嵌入矩阵值SDE的系数中，实现了对值域空间去噪和零空间信息合成的分离控制，在多种逆问题上取得一致性提升并展现出强大的系统失配鲁棒性。

**[T2Smark Balancing Robustness And Diversity In Noise-As-Watermark For Diffusion M](t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)**

:   提出 T2SMark，一种基于尾部截断采样（Tail-Truncated Sampling）的两阶段扩散模型水印方案，通过在高斯噪声的尾部区域嵌入水印比特、中心区域随机采样，首次在水印鲁棒性和生成多样性之间取得最优平衡。

**[Text-To-Image Models Leave Identifiable Signatures Implications For Leaderboard ](text-to-image_models_leave_identifiable_signatures_implications_for_leaderboard_.md)**

:   本文揭示了文生图（T2I）模型在生成图像中留下可识别的"签名"，使得攻击者可以在投票式排行榜中通过简单的CLIP嵌入空间分类实现模型去匿名化，从而操纵模型排名。

**[Text To Sketch Generation With Multi-Styles](text_to_sketch_generation_with_multi-styles.md)**

:   提出M3S（Multi-Style Sketch Synthesis），一个无训练框架，通过线性平滑的K/V特征注入、联合AdaIN风格倾向控制和风格-内容分离引导，实现基于文本提示和参考风格草图的单/多风格草图生成。

**[Thermalgen Style-Disentangled Flow-Based Generative Models For Rgb-To-Thermal Im](thermalgen_style-disentangled_flow-based_generative_models_for_rgb-to-thermal_im.md)**

:   提出 ThermalGen，一种基于 Flow 的自适应生成模型，通过 RGB 图像条件化架构和风格解耦机制，首次实现了跨视角、跨传感器、跨环境条件的高保真 RGB-to-Thermal 图像翻译，并发布了三个新的大规模卫星-航拍 RGB-T 配对数据集。

**[Tidmad Time Series Dataset For Discovering Dark Matter With Ai Denoising](tidmad_time_series_dataset_for_discovering_dark_matter_with_ai_denoising.md)**

:   发布 TIDMAD——首个面向暗物质搜索的超长时间序列去噪基准数据集，包含 ABRACADABRA 实验的训练/验证/科学数据、去噪评分指标和完整分析框架，使 AI 算法能直接产出物理学界标准的暗物质搜索结果。

**[Token Perturbation Guidance For Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)**

:   提出 Token Perturbation Guidance（TPG），通过对扩散模型中间 token 表示进行保范数的 shuffling 扰动来构造负分数信号，实现无需训练的条件无关引导，在无条件生成中将 SDXL 的 FID 提升近 2 倍，在条件生成中接近 CFG 效果。

**[Tortoise And Hare Guidance Accelerating Diffusion Model Inference With Multirate](tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)**

:   提出 Tortoise and Hare Guidance (THG)，一种免训练的扩散采样加速策略，将 classifier-free guidance (CFG) ODE 重构为多速率 ODE 系统，噪声估计使用细粒度步长（乌龟方程），附加引导项使用粗粒度步长（兔子方程），减少最多 30% 的函数评估次数 (NFE) 而几乎不损失生成质量。

**[Toward A Unified Geometry Understanding Riemannian Diffusion Framework For Graph](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)**

:   提出 GeoMancer 框架，通过黎曼 GyroKernel 自编码器替代数值不稳定的指数映射，将多层级图特征解耦到任务特定的积流形上，并引入流形约束扩散和自引导生成策略，在分子生成、节点分类和图回归等任务上统一建模并取得 SOTA 性能。

**[Towards A Golden Classifier-Free Guidance Path Via Foresight Fixed Point Iterati](towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)**

:   将条件引导统一为不动点迭代框架，发现CFG及其变体都是短区间单步迭代的特例，理论证明其次优性，进而提出前瞻引导(FSG)——在早期扩散阶段对更长区间执行多步迭代，以更少计算实现更好的对齐质量。

**[Towards General Modality Translation With Contrastive And Predictive Latent Diff](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)**

:   提出 LDDBM（Latent Denoising Diffusion Bridge Model），将去噪扩散桥模型扩展到共享潜空间中，结合对比对齐损失和预测损失，实现任意模态之间的通用翻译框架。

**[Towards Resilient Safety-Driven Unlearning For Diffusion Models Against Downstre](towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)**

:   提出ResAlign框架，通过Moreau包络近似和元学习策略，让扩散模型的安全卸载（unlearning）能抵抗下游微调带来的有害行为恢复，即使在纯良性数据上微调也能保持安全性。

**[Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)**

:   提出BREEZE框架，通过行为正则化、任务条件扩散策略和注意力增强表示建模，系统性解决FB-based零样本RL中的OOD外推误差和表达力不足问题，在ExORL和D4RL Kitchen上实现最优或接近最优的鲁棒零样本泛化性能。

**[Track Inpaint Resplat Subject-Driven 3D And 4D Generation With Progressive Textu](track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)**

:   提出TIRE（Track, Inpaint, REsplat）三阶段管线，通过视频跟踪定位未观测区域、主体驱动修复模型渐进式填充纹理、多视图一致性反投影回3D，实现身份保持的3D/4D生成。

**[Training-Free Constrained Generation With Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)**

:   提出一种无需重新训练的约束生成方法，通过在 Stable Diffusion 的反向去噪过程中嵌入近端 Langevin 动力学（Proximal Langevin Dynamics），将图像空间中的约束通过解码器反向传播到潜空间，实现对生成输出的严格约束满足。

**[Training-Free Efficient Video Generation Via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)**

:   提出 Jenga，一种即插即用的视频扩散 Transformer 推理加速方案，通过动态注意力剪裁和渐进分辨率生成实现 8.83× 加速，VBench 性能仅下降 0.01%。

**[Training-Free Safe Text Embedding Guidance For Text-To-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)**

:   提出 Safe Text embedding Guidance (STG)，一种无需训练的安全文本到图像生成方法，通过在扩散采样过程中基于安全函数对预期去噪图像的评估来动态调整文本嵌入方向，在有效去除不安全内容的同时最大程度保留原始语义意图。

**[Transferable Black-Box One-Shot Forging Of Watermarks Via Image Preference Model](transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)**

:   本文提出一种基于图像偏好模型的黑盒水印伪造方法，仅需单张水印图像即可通过反向传播从中提取水印并粘贴到任意新图像上，在不访问水印算法的条件下有效伪造多种后处理水印方案。

**[Tree-Guided Diffusion Planner](tree-guided_diffusion_planner.md)**

:   提出Tree-guided Diffusion Planner (TDP)，将测试时扩散规划形式化为树搜索问题，通过双层采样（粒子引导生成多样父轨迹 + 快速条件去噪生成子轨迹）在探索与利用之间取得平衡，在非凸目标和不可微约束下显著超越现有方法。

**[Two-Steps Diffusion Policy For Robotic Manipulation Via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)**

:   本文提出遗传扩散策略（GDP），通过分析裁剪导致的分布不匹配问题，结合降低噪声注入和基于群体选择的遗传算法去噪策略，使扩散策略仅需2步神经函数评估即可完成复杂操控任务。

**[Ultrahr-100K Enhancing Uhr Image Synthesis With A Large-Scale High-Quality Datas](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)**

:   构建了包含 10 万张超高分辨率图像及丰富标注的 UltraHR-100K 数据集，并提出频率感知后训练方法（DOTS + SWFR），通过面向细节的时间步采样和基于 DFT 的软加权频率正则化来增强预训练 T2I 模型的超高分辨率细节生成能力。

**[Understand Before You Generate Self-Guided Training For Autoregressive Image Gen](understand_before_you_generate_self-guided_training_for_autoregressive_image_gen.md)**

:   通过系统分析自回归图像生成中阻碍视觉语义学习的三个关键属性（局部条件依赖、步间语义不一致、空间不变性缺失），提出 ST-AR 训练框架，将掩码图像建模和对比学习融入 next-token prediction 训练，在不依赖预训练表示模型的情况下，使 LlamaGen-XL 的 FID 提升约 49%（从 19.42 降到 9.81），50 epoch 即接近 3B 参数模型 300 epoch 的效果。

**[Understanding Representation Dynamics Of Diffusion Models Via Low-Dimensional Mo](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)**

:   在低秩高斯混合（MoLRG）数据模型下，理论证明了扩散模型表示质量随噪声水平呈单峰动态的现象源于去噪强度与类别区分度的权衡，并实证发现单峰动态的出现可作为模型泛化能力的可靠指标。

**[Unilumos Fast And Unified Image And Video Relighting With Physics-Plausible Feed](unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)**

:   提出UniLumos，一个统一的图像和视频重光照框架，通过在flow matching骨干中引入RGB空间的深度和法线几何反馈来增强物理合理性，同时借助路径一致性学习实现20倍加速。

**[Unleashing Diffusion Transformers For Visual Correspondence By Modulating Massiv](unleashing_diffusion_transformers_for_visual_correspondence_by_modulating_massiv.md)**

:   发现 Diffusion Transformers (DiTs) 中存在 massive activations 现象导致特征不可区分，揭示其与 AdaLN 的内在联系，提出无需训练的 DiTF 框架来提取语义判别性特征，在视觉对应任务上超越 DINO 和 SD 模型。

**[Utilgen Utility-Centric Generative Data Augmentation With Dual-Level Task Adapta](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)**

:   提出以任务效用为中心的生成式数据增强框架 UtilGen，通过元学习权重网络评估合成数据的下游任务效用，并利用模型级 DPO 和实例级（prompt+noise）双层优化策略，自适应生成高效用的合成训练数据，在8个基准上平均提升3.87%。

**[V-Cece Visual Counterfactual Explanations Via Conceptual Edits](v-cece_visual_counterfactual_explanations_via_conceptual_edits.md)**

:   提出 V-CECE，首个系统性解决人类语义理解与神经网络推理差异的视觉反事实解释框架，通过知识图谱保证编辑最优性，利用扩散模型执行概念级编辑，无需训练即可生成人类可理解的反事实图像。

**[Value Gradient Guidance For Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)**

:   提出VGG-Flow方法，利用最优控制理论中的Hamilton-Jacobi-Bellman方程，将流匹配模型对齐问题转化为"残差速度场匹配值函数梯度"的梯度匹配任务，实现高效且保持先验分布的奖励对齐。

**[Vicinity-Guided Discriminative Latent Diffusion For Privacy-Preserving Domain Ad](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)**

:   提出 Discriminative Vicinity Diffusion (DVD)，首次将潜扩散模型用于判别式知识迁移，通过在源域特征的近邻潜空间中训练扩散模型生成源样式线索，实现无需源数据访问的域适应，在标准 SFDA 基准上超越 SOTA。

**[Video Diffusion Models Excel At Tracking Similar-Looking Objects Without Supervi](video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)**

:   本文发现预训练视频扩散模型在高噪声去噪阶段内在地学到了运动表征，无需任何跟踪专用训练即可用于跟踪外观相似的物体，提出的 TED 方法在 DAVIS 等基准上以最高 6% 的改进超越了 17 种自监督方法。

**[Vsa Faster Video Diffusion With Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)**

:   提出 VSA (Video Sparse Attention)，一种端到端可训练的硬件对齐稀疏注意力机制，通过粗粒度阶段（cube 池化预测关键 token）和细粒度阶段（在预测的块稀疏区域执行 token 级注意力）的层次化设计，在视频 DiT 的训练和推理中同时实现加速：从头预训练实现 2.53× 训练 FLOPs 减少且无质量损失，适配 Wan2.1-1.3B 实现注意力 6× 加速和端到端推理从 31s 降至 18s。

**[Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)**

:   首次将 LLM 水印技术（KGW green/red scheme）适配到自回归图像生成模型的 token 层，识别并解决了关键挑战——反向循环一致性（RCC）不足，通过 tokenizer-detokenizer 微调和水印同步层实现了鲁棒的、具有理论保证的图像水印检测。

**[What We Dont C Manifold Disentanglement For Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)**

:   提出 WWDC（What We Don't C）方法，利用条件引导的潜在流匹配从已有 VAE 表征中去除已知信息，使未知特征在残余流形中更易被发现和访问，实现迭代式科学发现。

**[When Are Concepts Erased From Diffusion Models](when_are_concepts_erased_from_diffusion_models.md)**

:   本文提出了两种概念擦除的机制模型（引导式回避 vs. 破坏式移除），并设计了涵盖优化搜索、上下文探测、噪声轨迹探测、分类器引导和动态追踪的五种独立探测方法，系统性地揭示了现有擦除方法大多只是"绕开"概念而非真正"消除"知识。

**[Where And How To Perturb On The Design Of Perturbation Guidance In Diffusion And](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)**

:   提出 HeadHunter 框架和 SoftPAG 方法，将扩散模型中的注意力扰动粒度从层级细化到单个注意力头级别，首次发现不同注意力头控制不同视觉概念（结构、风格、纹理等），实现了更精准且可组合的生成引导。

**[Why Diffusion Models Dont Memorize The Role Of Implicit Dynamical Regularization](why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)**

:   通过数值实验和理论分析揭示扩散模型训练中存在两个关键时间尺度——泛化时间 $\tau_{\text{gen}}$ 和记忆化时间 $\tau_{\text{mem}}$，后者随训练集大小 $n$ 线性增长而前者保持恒定，由此产生的隐式动力学正则化使模型即使在高度过参数化情况下也能通过早停避免记忆化。

**[Why Diffusion Models Dont Memorize The Role Of Implicit Regularization](why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)**

:   本文从数值实验和理论分析两个层面揭示扩散模型训练中存在**隐式动态正则化**机制：生成高质量样本的时间尺度 τ_gen 与出现记忆化的时间尺度 τ_mem 之间的间隔随训练集大小 n 线性增长，为"早停"提供了理论支撑。

**[Why Knowledge Distillation Works In Generative Models A Minimal Working Explanat](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)**

:   提出知识蒸馏在生成模型中的最小工作解释：蒸馏自然诱导了精度-召回权衡，教师分布越具有选择性（低熵），学生模型越集中于高概率密度区域（高精度），以牺牲覆盖度（低召回）为代价——这在强调样本质量的场景中正是所需。

**[Wmcopier Forging Invisible Image Watermarks On Arbitrary Images](wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)**

:   提出 WMCopier，首个基于扩散模型的 no-box 水印伪造攻击方法，无需任何目标水印算法的先验知识，通过训练无条件扩散模型学习水印分布、浅层反演注入水印信号、迭代精炼优化质量，在开源和商业水印系统（包括 Amazon）上实现高成功率伪造。
