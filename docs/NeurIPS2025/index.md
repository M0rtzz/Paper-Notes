---
title: >-
  NeurIPS2025 2541篇论文解读
description: >-
  2541篇NeurIPS2025论文深度解读，每篇5分钟读懂核心思想，含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 NeurIPS2025 论文笔记

**2541** 篇论文解读，覆盖 **45** 个领域。

<div class="conf-index" markdown>

---

## 🎨 图像生成 { #image_generation }

**[70 Size 100 Accuracy Lossless Llm Compression For Efficient](image_generation/70_size_100_accuracy_lossless_llm_compression_for_efficient.md)**

:   DFloat11 利用 BFloat16 权重中指数位（exponent）的低熵特性，通过 Huffman 编码将 LLM/扩散模型无损压缩至原始大小的约 70%（等效 ~11 bit），并设计了层次化查找表和两阶段 GPU kernel 实现高效在线解压，使 Llama 3.1 405B 可在单节点 8×80GB GPU 上无损推理。

**[A Closer Look At Model Collapse From A Generalization-To-Memorization Perspectiv](image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)**

:   发现扩散模型在自消耗循环（用生成数据训练下一代模型）中存在从"泛化"到"记忆"的转变过程，揭示训练集熵与模型泛化能力的强线性相关性（Pearson r=0.91），并提出基于熵的数据选择策略（Greedy Selection / Threshold Decay Filter）有效减缓该转变，在 CIFAR-10 accumulate 范式下第 8 轮 FID 从 75.7 降至 44.7。

**[A Connection Between Score Matching And Local Intrinsic Dimension](image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)**

:   证明去噪得分匹配损失（denoising score matching loss）的下界恰好是数据流形的局部固有维度（LID），从而将 DSM loss 本身作为一个高效的 LID 估计器——无需梯度计算或多次前向传播，在 Stable Diffusion 3.5 上内存占用仅为 FLIPD 的 60%，且量化后估计更稳定。

**[A Data-Driven Prism Multi-View Source Separation With Diffusion Model Priors](image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)**

:   提出 DDPRISM 方法，利用多视图观测中不同线性变换的结构性差异，在 EM 框架下为每个未知源学习独立的扩散模型先验，无需预先获得任何单独的源样本即可完成源分离和后验采样，在合成问题和真实星系观测上超越现有方法。

**[A Diffusion Model For Regular Time Series Generation From Irregular Data With Co](image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)**

:   提出两步框架从不规则采样时序数据生成规则时序：先用 TST 自编码器补全缺失值构造"自然邻域"，再在视觉扩散模型中用 masking 策略仅在观测像素上计算损失，避免对补全值的过度依赖，在判别分数上平均改善 70%，训练速度提升 6.5 倍。

**[A Gradient Flow Approach To Solving Inverse Problems With Latent Diffusion Model](image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)**

:   提出 DWGF（Diffusion-regularized Wasserstein Gradient Flow），将隐空间扩散模型的后验采样问题严格形式化为 KL 散度在 Wasserstein-2 空间上的正则化梯度流，推导出隐空间中的 ODE 系统用于求解图像逆问题，在 FFHQ-512 上的修复和超分辨率任务中 PSNR 大幅超越基线。

**[Accelerating Parallel Diffusion Model Serving With Residual Compression](image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)**

:   提出 CompactFusion 框架，通过残差压缩（仅传输相邻去噪步骤间的激活差异而非完整激活）来消除并行扩散推理中的通信冗余，在 4×L20 上实现 3.0× 加速且生成质量远优于 DistriFusion，在模拟以太网带宽下实现 6.7× 加速，甚至在 100× 压缩下仍优于 DistriFusion。

**[Accuquant Simulating Multiple Denoising Steps For Quantizing](image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)**

:   揭示扩散模型量化中的误差累积现象——每步的量化误差会传递并放大到后续步骤——并提出在 PTQ 校准阶段显式模拟连续多步去噪过程来联合优化量化参数的方法，同时通过巧妙的目标函数设计将内存从 O(n) 降至 O(1)。

**[Adapting Speech Language Model To Singing Voice Synthesis](image_generation/adapting_speech_language_model_to_singing_voice_synthesis.md)**

:   将 1.7B 参数的 TTS 预训练 Speech Language Model 适配到歌声合成（SVS）任务，通过乐谱 tokenization + multi-stream LM 预测 + conditional flow matching 精修 + vocoder，仅用 135 小时合成歌声数据达到与专用 SVS 系统可比的性能。

**[Alebench A Benchmark For Longhorizon Objectivedriven Algorit](image_generation/alebench_a_benchmark_for_longhorizon_objectivedriven_algorit.md)**

:   提出ALE-Bench，首个面向分数制算法工程竞赛（AtCoder Heuristic Contest）的AI基准，收集40道NP-hard优化赛题并提供交互式Agent评估框架，发现最强模型o3-high在one-shot设置下仅达人类平均水平，且AI在跨问题一致性和长时间迭代改进上与人类专家差距显著。

**[Aligning Compound Ai Systems Via System-Level Dpo](image_generation/aligning_compound_ai_systems_via_system-level_dpo.md)**

:   将复合 AI 系统建模为 DAG，提出 SysDPO 框架将 DPO 扩展到多组件联合对齐，通过 DAG 分解将系统级偏好转化为可端到端优化的损失函数，理论证明了 β-完美对齐保证，在 LLM+扩散模型和 LLM+LLM 系统上显著提升协作质量。

**[Aligning Text To Image In Diffusion Models Is Easier Than You Think](image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)**

:   提出 SoftREPA——一种轻量级对比微调策略，通过引入可学习 soft text token（不到 1M 参数）在冻结的预训练 T2I 扩散模型上进行对比学习，显式提高文本和图像表征的互信息，在 SD1.5/SDXL/SD3 上显著提升文本-图像对齐质量，且适用于图像生成和图像编辑任务。

**[Amortized Sampling With Transferable Normalizing Flows](image_generation/amortized_sampling_with_transferable_normalizing_flows.md)**

:   提出 Prose——一个 285M 参数的全原子可迁移归一化流，基于 TarFlow 架构训练在 21,700 个短肽 MD 轨迹上（总计 4.3ms 模拟时长），实现对任意短肽系统的零样本无相关性提议采样，在能量评估预算相同时超越 MD 基线，生成速度比之前的可迁移玻尔兹曼生成器 (TBG) 快 4000 倍。

**[Auggen Synthetic Augmentation Using Diffusion Models Can Imp](image_generation/auggen_synthetic_augmentation_using_diffusion_models_can_imp.md)**

:   提出 AugGen，一种自包含的合成数据增强方法：在目标数据集上训练类条件扩散模型，通过混合不同类别的条件向量生成新的"混合类"样本，增强判别模型训练，在人脸识别基准上实现 1-12% 的性能提升，无需任何外部数据或辅助模型。

**[Badiff Bandwidth Adaptive Diffusion Model](image_generation/badiff_bandwidth_adaptive_diffusion_model.md)**

:   提出 BADiff——首个带宽自适应扩散模型，通过将目标熵约束作为条件嵌入扩散反向过程，配合可微熵正则化损失和自适应停止策略，使模型根据实时带宽动态调整生成质量并自适应提前终止采样，在保持感知质量的同时减少计算开销，从根本上避免了传统"高质量生成→后压缩"流程中的压缩伪影和计算浪费。

**[Balanced Conic Rectified Flow](image_generation/balanced_conic_rectified_flow.md)**

:   针对 k-rectified flow 中 reflow 步骤导致的分布漂移问题，提出 conic reflow：利用真实图像的反演噪声及其 Slerp 扰动构成锥形监督轨迹，大幅减少所需 fake pair 数量的同时获得更优的生成质量和更直的 ODE 路径。

**[Beyond Masked And Unmasked Discrete Diffusion Models Via Par](image_generation/beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)**

:   Prime（Partial masking scheme）通过将每个token用base-b子token序列表示并在子token级别独立掩码，为掩码扩散模型引入中间状态，实现细粒度去噪过程，在OpenWebText上以15.36困惑度首次让MDM在不使用自回归公式的情况下超越ARM（17.54）。

**[Bitmark Watermarking Bitwise Autoregressive Image Generative Models](image_generation/bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)**

:   提出 BitMark——首个针对比特级自回归图像生成模型（Infinity、Instella）的水印方案，在生成过程中通过对 logit 加偏置将 bit 序列引向"绿色列表"，实现可靠检测（z-test）、高图像保真度（FID 几乎不变）、对多种攻击的鲁棒性和放射性（训练在水印图上的下游模型也带有水印），为防止模型坍缩提供了关键工具。

**[Blameless Users In A Clean Room Defining Copyright Protection For Generative Mod](image_generation/blameless_users_in_a_clean_room_defining_copyright_protection_for_generative_mod.md)**

:   重建生成模型可证明版权保护的理论基础——证明现有的 Near Access-Freeness (NAF) 定义不能防止逐字复制（"被污染"），提出"无辜用户"(blameless) 框架和净室版权保护 ($(\kappa,\beta)$-clean) 定义，其中用户在反事实"净室设置"中不会复制则在真实世界中也不太可能复制，并证明差分隐私训练在"黄金数据集"假设下蕴含净室版权保护。

**[Blind Strong Gravitational Lensing Inversion Joint Inference Of Source And Lens ](image_generation/blind_strong_gravitational_lensing_inversion_joint_inference_of_source_and_lens_.md)**

:   首次将 score-based 生成模型先验应用于强引力透镜的盲反演——联合推断背景源天体形态和透镜质量分布参数，通过将 GibbsDDRM 扩展到连续时间域实现采样，重建残差与观测噪声一致，透镜参数边际后验无系统偏差。

**[Blurdm A Blur Diffusion Model For Image Deblurring](image_generation/blurdm_a_blur_diffusion_model_for_image_deblurring.md)**

:   提出 BlurDM，将运动模糊的物理形成过程（连续曝光导致渐进模糊累积）集成到扩散模型——双扩散前向（同时加噪声+模糊）+ 双去噪去模糊反向，作为隐空间先验生成器一致性增强 4 种去模糊方法在 4 个数据集上的效果，GoPro 平均 +0.31 dB，RealBlur-J 平均 +0.78 dB，仅增加 ~4 GFLOPs 和 ~9ms。

**[Blurguard A Simple Approach For Robustifying Image Protection Against Ai-Powered](image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)**

:   提出 BlurGuard——在生成对抗扰动之前先对图像做轻度模糊预处理，使扰动更鲁棒地抵御 JPEG 压缩、高斯噪声等后处理操作，从而更有效地保护图像不被 Stable Diffusion 等 AI 编辑工具篡改，在保护成功率上比不模糊基线提升 20%+。

**[Boltznce Learning Likelihoods For Boltzmann Generation With Stochastic Interpola](image_generation/boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)**

:   BoltzNCE 用 Score Matching + InfoNCE 混合训练 Energy-Based Model 来近似 Boltzmann Generator 的似然，避免了昂贵的 Jacobian trace 计算，在丙氨酸二肽构象生成上实现 100× 推理加速且自由能误差仅 0.02 $k_BT$。

**[Boosting Generative Image Modeling Via Joint Imagefeature Sy](image_generation/boosting_generative_image_modeling_via_joint_imagefeature_sy.md)**

:   提出ReDi (Representation Diffusion)框架，在扩散模型中联合建模VAE图像latent和DINOv2语义特征——两者在同一扩散过程中从纯噪声同步去噪，仅需最小修改DiT架构即实现23倍训练收敛加速和SOTA FID，并解锁Representation Guidance推理策略。

**[Breaking Ars Sampling Bottleneck Provable Acceleration Via D](image_generation/breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)**

:   从信息论角度为掩码扩散语言模型建立了完整的采样收敛理论：证明 KL 散度形式的采样误差以 $O(1/T)$ 速率衰减、与 token 间互信息线性相关，并给出匹配的下界证明了分析的紧性，理论上论证了扩散模型可以在 $T < L$（序列长度）步内生成高质量样本。

**[Cadmorph Geometry-Driven Parametric Cad Editing Via A Plan-Generate-Verify Loop](image_generation/cadmorph_geometry-driven_parametric_cad_editing_via_a_plan-generate-verify_loop.md)**

:   提出 CADMorph，一个迭代式 plan–generate–verify 框架，利用预训练的 Parameter-to-Shape (P2S) 扩散模型和 Masked-Parameter-Prediction (MPP) 大语言模型协同工作，在无需三元组训练数据的情况下实现几何驱动的参数化 CAD 编辑。

**[Camila Contextaware Masking For Image Editing With Language](image_generation/camila_contextaware_masking_for_image_editing_with_language.md)**

:   提出 CAMILA，一种上下文感知的图像编辑方法，利用多模态大语言模型（MLLM）自动判断指令是否可在给定图像上执行，生成 [MASK]/[NEG] 专用 token 区分可编辑区域和应忽略区域，实现精准多指令编辑并有效过滤不可执行指令。

**[Camit A Time-Aware Car Model Dataset For Classification And Generation](image_generation/camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)**

:   提出 CaMiT 数据集（787K 标注 + 5.1M 无标注汽车图像，2005–2023），系统研究细粒度视觉类别的时间漂移问题，并在静态预训练、时间增量预训练、时间增量分类器学习和时间感知图像生成四个场景下提供 benchmark。

**[Can Knowledge-Graph-Based Retrieval Augmented Generation Really Retrieve What Yo](image_generation/can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)**

:   提出 GraphFlow 框架，将知识图谱上的检索建模为 GFlowNet 的流匹配问题，通过详细平衡目标和局部探索策略联合训练检索策略与流估计器，在 STaRK 基准上检索准确率和多样性均超越 GPT-4o 约 10%。

**[Cdflow Building Invertible Layers With Circulant And Diagonal Matrices](image_generation/cdflow_building_invertible_layers_with_circulant_and_diagonal_matrices.md)**

:   提出 CDFlow，利用循环矩阵和对角矩阵的交替乘积构造可逆线性层，将参数复杂度从 $\mathcal{O}(n^2)$ 降至 $\mathcal{O}(mn)$，矩阵逆复杂度从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn\log n)$，对数行列式从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(mn)$，在密度估计和周期性数据建模上超越同类方法。

**[Composite Flow Matching For Reinforcement Learning With Shifted-Dynamics Data](image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)**

:   提出 CompFlow，通过复合流匹配架构（在离线流输出分布上构建在线流）估计离线-在线环境间的动态差异（Wasserstein 距离），并结合高动态差异区域的主动探索策略，在 27 个动态偏移 RL 任务中平均回报超越最强基线 14.2%。

**[Composition And Alignment Of Diffusion Models Using Constrai](image_generation/composition_and_alignment_of_diffusion_models_using_constrai.md)**

:   提出统一的约束优化框架，将扩散模型的奖励对齐和多模型组合问题形式化为约束优化，通过拉格朗日对偶方法自动确定最优权重，避免手动超参数搜索。

**[Conditional Panoramic Image Generation Via Masked Autoregres](image_generation/conditional_panoramic_image_generation_via_masked_autoregres.md)**

:   提出PAR（Panoramic AutoRegressive model），首次用掩码自回归建模统一文本到全景图和全景图外延两大任务，通过循环平移一致性损失和双空间循环填充解决ERP全景图的边界不连续问题，在Matterport3D上取得37.37 FID，同时展示出良好的可扩展性和零样本泛化能力。

**[Constrained Discrete Diffusion](image_generation/constrained_discrete_diffusion.md)**

:   提出 CDD（Constrained Discrete Diffusion），将可微约束优化投影算子嵌入离散扩散模型的去噪过程中，无需重训练即可在采样时强制满足序列级约束，在毒性文本生成、分子设计和指令遵循三类任务上实现零约束违反。

**[Contextual Thompson Sampling Via Generation Of Missing Data](image_generation/contextual_thompson_sampling_via_generation_of_missing_data.md)**

:   提出 Generative Thompson Sampling (TS-Gen)，将上下文老虎机中的不确定性建模为缺失数据而非未知参数，通过生成模型对缺失结果做自回归填充来实现 Thompson 采样，建立了与离线预测损失直接挂钩的遗憾界。

**[Continuous Diffusion Model For Language Modeling](image_generation/continuous_diffusion_model_for_language_modeling.md)**

:   提出 RDLM（Riemannian Diffusion Language Model），在统计流形（超球面）上构建连续扩散过程来建模离散分布，建立了离散扩散与连续流的理论联系，通过径向对称性实现无模拟训练和维度分裂技术处理大词表，在 Text8 上以 1.32 BPC 超越所有离散和连续扩散模型。

**[Continuous Uniqueness And Novelty Metrics For Generative Modeling Of Inorganic C](image_generation/continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)**

:   针对无机晶体生成模型评估中广泛使用的离散距离函数 (StructureMatcher) 的四大缺陷，提出基于 Magpie 指纹（成分）和 AMD 向量（结构）的连续距离函数，实现更可靠的 uniqueness 和 novelty 度量。

**[Coral Disentangling Latent Representations In Longtailed Dif](image_generation/coral_disentangling_latent_representations_in_longtailed_dif.md)**

:   深入诊断长尾数据下扩散模型尾类生成退化的根因为 U-Net 瓶颈层的"表示纠缠"（representation entanglement），提出 CORAL 通过在瓶颈层施加监督对比损失来解耦类别表示，在 CIFAR10/100-LT、CelebA-5、ImageNet-LT 上全面超越 DDPM/CBDM/T2H 等基线。

**[Coreinforcement Learning For Unified Multimodal Understandin](image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)**

:   提出 CoRL（Co-Reinforcement Learning）框架，通过"统一RL→精细化RL"两阶段策略对统一多模态模型（ULM）同时进行理解和生成能力的强化学习优化，实现理解生成双能力的协同进化，在 1.5B 参数量下生成提升 7%、理解提升 23%。

**[Counterfactual Identifiability Via Dynamic Optimal Transport](image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)**

:   利用动态最优传输 (dynamic OT) 理论，首次解决了高维多变量 Markovian SCM 中反事实的可辨识性问题——证明 OT flow 机制产生唯一的单调保序反事实传输映射，并扩展至非 Markovian 设置（IV/BC/FC 准则）。

**[Coupling Generative Modeling And An Autoencoder With The Causal Bridge](image_generation/coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)**

:   在存在未观测混淆因子的因果推断中，提出将生成模型与自编码器耦合来提升因果桥函数 (causal bridge) 的估计质量——通过共享编码器在处理/控制/结果变量间传递统计强度，并将框架扩展到生存分析。

**[Cross-Fluctuation Phase Transitions Reveal Sampling Dynamics In Diffusion Models](image_generation/cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)**

:   借鉴统计物理中的涨落理论（fluctuation theory），提出了一种通过 **交叉涨落（cross-fluctuation）** 检测扩散模型采样过程中离散相变的框架，从而在无需重新训练的情况下加速采样、改进条件生成、提升零样本分类和风格迁移。

**[Curly Flow Matching For Learning Non-Gradient Field Dynamics](image_generation/curly_flow_matching_for_learning_non-gradient_field_dynamics.md)**

:   提出 Curly Flow Matching (Curly-FM)，通过设计带有非零参考漂移的 Schrödinger Bridge 问题，使 flow matching 能够学习周期性、旋转性等非梯度场动力学，突破了传统方法只能建模梯度场的限制。

**[Decaflow A Deconfounding Causal Generative Model](image_generation/decaflow_a_deconfounding_causal_generative_model.md)**

:   提出 DeCaFlow，一个去混淆的因果生成模型，在给定因果图和观测数据的情况下，只需训练一次即可正确估计所有 do-calculus 可识别的因果查询（包括干预和反事实），即使存在隐藏混淆因子。

**[Decomate Leveraging Generative Models For Co-Creative Svg Animation](image_generation/decomate_leveraging_generative_models_for_co-creative_svg_animation.md)**

:   提出 Decomate 交互系统，利用多模态大语言模型 (MLLM) 将非结构化 SVG 图形自动分解为语义组件，设计师通过自然语言为各组件指定动画行为，系统生成可生产的 HTML/CSS/JS 动画代码，支持迭代协作创作。

**[Deft Decompositional Efficient Finetuning For Texttoimage Mo](image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)**

:   提出DEFT（Decompositional Efficient Fine-Tuning），通过将权重更新分解为子空间投影和低秩调整两个组件来高效微调T2I模型，在个性化生成和通用图像生成任务上超越LoRA和PaRa。

**[Denoising Weak Lensing Mass Maps With Diffusion Model And Generative Adversarial](image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)**

:   将扩散模型（DM）应用于弱引力透镜质量图去噪任务，与 GAN（pix2pix）在相同实验设置下进行系统性对比，证明 DM 在训练稳定性、多样本平均鲁棒性和多种统计量重建精度上全面优于 GAN。

**[Detecting Generated Images By Fitting Natural Image Distributions](image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)**

:   提出一致性验证框架 ConV，利用自然图像流形与生成图像之间的几何差异，通过两个梯度正交的函数实现无需训练的生成图像检测，并引入 Normalizing Flow 增强版 F-ConV 进一步放大流形偏差。

**[Detection And Simulation Of Urban Heat Islands Using A Fine-Tuned Geospatial Fou](image_generation/detection_and_simulation_of_urban_heat_islands_using_a_fine-tuned_geospatial_fou.md)**

:   提出一套利用微调地理空间基础模型（Granite-GFM）的三阶段统一工作流——先通过绿地冷却效应建立实证基线验证模型物理真实性，再外推未来气候情景下的城市温度，最后通过 inpainting 模拟绿化干预的降温效果——将基础模型从评估工具升级为城市规划的交互式模拟平台。

**[Dexter Diffusion-Guided Explanations With Textual Reasoning For Vision Models](image_generation/dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)**

:   提出 DEXTER，一个无需数据的框架，通过优化文本提示驱动扩散模型生成最大化目标分类器激活的图像，再用 LLM 对合成样本进行推理，生成全局性、可读的文本解释，实现模型行为的偏差发现和全局解释。

**[Dico Revitalizing Convnets For Scalable And Efficient Diffus](image_generation/dico_revitalizing_convnets_for_scalable_and_efficient_diffus.md)**

:   发现预训练DiT的全局self-attention在生成任务中主要捕获局部模式存在大量冗余，提出用标准卷积模块+紧凑通道注意力（CCA）构建纯卷积扩散模型DiCo，在ImageNet-256上FID达2.05超越DiT-XL/2且推理速度快2.7倍，512分辨率下更快3.1倍。

**[Diff-Icmh Harmonizing Machine And Human Vision In Image Compression With Generat](image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)**

:   提出 Diff-ICMH，一种基于扩散模型的生成式图像压缩框架，通过语义一致性损失（SC loss）保持语义完整性，通过标签引导模块（TGM）激活生成先验，以单一编解码器和码流同时服务 10+ 种智能任务和人类视觉感知，无需任何任务特定适配。

**[Diffeye Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned On N](image_generation/diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)**

:   提出 DiffEye，首个基于扩散模型直接利用原始眼动轨迹数据、以自然图像为条件生成连续且多样化眼动轨迹的框架，同时引入对应位置嵌入 (CPE) 对齐注视空间与图像语义空间。

**[Diffusion-Based Electromagnetic Inverse Design Of Scattering Structured Media](image_generation/diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)**

:   提出基于条件扩散模型的电磁逆设计框架，从目标微分散射截面 (DSCS) 直接生成介电球超表面几何结构，绕过昂贵的迭代优化，并自然处理逆问题的非唯一性，性能优于 CMA-ES 进化优化且速度快数个数量级。

**[Diffusion-Driven Progressive Target Manipulation For Source-Free Domain Adaptati](image_generation/diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)**

:   提出 DPTM 框架，利用潜在扩散模型对不可信目标样本进行语义变换，生成伪目标域并通过渐进式重建机制迭代缩小与真实目标域的差距，在大域偏移场景下比现有 SFDA SOTA 提升高达 18.6%。

**[Diffusion Adaptive Text Embedding For Texttoimage Diffusion](image_generation/diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)**

:   提出 DATE（Diffusion Adaptive Text Embedding），在扩散模型采样过程中根据当前去噪中间结果动态更新文本嵌入，无需额外训练即可提升文本-图像语义对齐。

**[Diffusion Classifiers Understand Compositionality But Condit](image_generation/diffusion_classifiers_understand_compositionality_but_condit.md)**

:   全面研究零样本扩散分类器在组合理解任务上的判别能力：覆盖3个扩散模型(SD 1.5/2.0/3-m)×10个数据集×30+任务，引入Self-Bench诊断基准（用扩散模型自己生成的图像消除域差异），发现扩散分类器确实理解组合性但受域差距和时间步加权影响——"条件适用"。

**[Diffusion Generative Modeling On Lie Group Representations](image_generation/diffusion_generative_modeling_on_lie_group_representations.md)**

:   提出在李群**表示空间**（而非李群本身）上构建扩散过程的新理论框架，通过广义分数匹配将非阿贝尔李群的弯曲动力学映射到欧几里得空间中，实现无模拟训练的李群扩散模型，并证明标准分数匹配是其平移群的特例。

**[Diffusion Models Meet Contextual Bandits](image_generation/diffusion_models_meet_contextual_bandits.md)**

:   将预训练扩散模型作为上下文赌博机 (contextual bandits) 问题中动作参数的表达性先验，提出 diffusion Thompson Sampling (dTS) 算法，通过高效的层次化后验近似实现快速更新与采样，在大动作空间下显著优于传统方法。

**[Distilled Decoding 2 Onestep Sampling Of Image Autoregressiv](image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)**

:   本文提出Distilled Decoding 2（DD2），通过将自回归图像模型重新解读为条件分数模型，设计了条件分数蒸馏（CSD）损失，将多步AR采样压缩为一步生成，在ImageNet-256上实现FID从3.40到5.43的微小退化同时获得8.0x加速（VAR）和238x加速（LlamaGen），相比DD1缩小了67%的性能差距且训练快12.3倍。

**[Dove Efficient One-Step Diffusion Model For Real-World Video Super-Resolution](image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)**

:   提出 DOVE，基于 CogVideoX 预训练视频生成模型，通过两阶段潜空间-像素空间训练策略和高质量 HQ-VSR 数据集实现单步推理的视频超分辨率，比多步扩散方法快 28 倍且性能相当或更优。

**[Dynamic Diffusion Schrödinger Bridge In Astrophysical Observational Inversions](image_generation/dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)**

:   提出 Astro-DSB，一种基于 Diffusion Schrödinger Bridge 的天文物理反问题建模方法，直接学习观测量到真实物理分布的概率映射，训练成本仅为条件 DDPM 的 25%，且在分布外（OOD）测试中展现出显著的泛化优势，并成功应用于 Taurus B213 真实观测数据。

**[Editinfinity Image Editing With Binary-Quantized Generative Models](image_generation/editinfinity_image_editing_with_binary-quantized_generative_models.md)**

:   提出 EditInfinity，首次将经典"图像反演-图像编辑"范式应用于二值量化自回归生成模型 Infinity，利用量化表示可获取精确中间监督的优势实现高精度图像反演，配合分段线性平滑核实现高保真编辑效果，在 PIE-Bench 上全面超越扩散模型基线。

**[Eegrexfernet A Lightweight Gen-Ai Framework For Eeg Subspace Reconstruction Via ](image_generation/eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)**

:   提出 EEGReXferNet，一种轻量级生成式 AI 框架，通过邻域通道感知输入选择、频带特定子窗口卷积编解码、动态滑窗隐空间和参考统计量缩放，在跨被试迁移学习设置下实现 EEG 子空间重建，参数减少约 45%、推理延迟 <1ms，同时保持 PSD 相关性 $\geq 0.95$ 和谱图 RV 系数 $\geq 0.85$。

**[Efficient Rectified Flow For Image Fusion](image_generation/efficient_rectified_flow_for_image_fusion.md)**

:   提出 RFfusion，首次将 Rectified Flow 引入图像融合任务，实现无需额外训练的单步采样，同时设计面向融合的两阶段 VAE 训练策略，在速度和质量上全面超越现有扩散融合方法。

**[Elucidated Rolling Diffusion Models For Probabilistic Forecasting Of Complex Dyn](image_generation/elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)**

:   提出 ERDM，首次将滚动扩散（Rolling Diffusion）框架与 EDM 的原则性设计（噪声调度、预条件化、Heun 采样器）成功统一，通过渐进噪声调度显式建模不确定性增长，在 Navier-Stokes 和 ERA5 天气预报任务上显著优于自回归 EDM 基线。

**[Emergence And Evolution Of Interpretable Concepts In Diffusi](image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)**

:   首次将 Sparse Autoencoders (SAEs) 系统性地应用于多步扩散模型 (Stable Diffusion v1.4)，揭示了图像构图在第一步反向扩散就已涌现、风格概念在中期阶段形成的时间演化规律，并据此设计了时间自适应的因果干预技术。

**[Encoder-Decoder Diffusion Language Models For Efficient Training And Inference](image_generation/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)**

:   提出 E2D2，一种面向离散扩散语言模型的编码器-解码器架构，通过轻量解码器迭代去噪、大型编码器定期更新表征，同时实现更快推理（~3× vs MDLM）和更高效的 block diffusion 训练（FLOPs 减半）。

**[Energy Loss Functions For Physical Systems](image_generation/energy_loss_functions_for_physical_systems.md)**

:   提出基于物理能量的损失函数框架，通过反向 KL 散度与玻尔兹曼分布推导出以成对距离为基础的能量差损失，天然满足 SE(d) 不变性，在分子生成和自旋基态预测中显著优于 MSE 和交叉熵损失。

**[Enhancing Diffusion Model Guidance Through Calibration And Regularization](image_generation/enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)**

:   针对分类器引导扩散模型中分类器过度自信导致梯度消失的问题，提出两类互补方案：(1) Smooth ECE 校准损失微调分类器，FID 改善 ~3%；(2) 基于 f-散度的正则化采样引导（RKL/FKL/JS），无需重训练即在 ImageNet 128×128 上达到 FID 2.13。

**[Entropy Rectifying Guidance For Diffusion And Flow Models](image_generation/entropy_rectifying_guidance_for_diffusion_and_flow_models.md)**

:   提出 Entropy Rectifying Guidance (ERG)，通过操控注意力层的 Hopfield 能量景观（温度缩放、步长调整）来获取弱预测信号，替代传统 CFG 中的无条件预测，在文本到图像、类条件和无条件生成中同时提升质量、多样性和一致性。

**[Epistemic Uncertainty For Generated Image Detection](image_generation/epistemic_uncertainty_for_generated_image_detection.md)**

:   提出 WePe（Weight Perturbation），通过对预训练视觉大模型（DINOv2）施加权重扰动来估计认识不确定性（epistemic uncertainty），利用自然图像与 AI 生成图像在不确定性空间的差异实现检测，无需训练即可使用。

**[Equivariant Flow Matching For Symmetry-Breaking Bifurcation Problems](image_generation/equivariant_flow_matching_for_symmetry-breaking_bifurcation_problems.md)**

:   提出等变 flow matching 框架，结合 symmetric coupling 策略，用生成式 AI 建模对称性破缺分岔问题中的多模态概率分布，在物理系统（屈曲梁、Allen-Cahn 方程）上显著优于确定性模型和 VAE。

**[Evaluating The Evaluators Metrics For Compositional Text-To-Image Generation](image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)**

:   系统评估了 12 种文本-图像组合对齐指标与人类判断的一致性，发现没有单一指标在所有组合任务上一致表现最优，VQA 指标并非总是最好的，embedding 类指标（ImageReward、HPS）在特定类别上更强。

**[Evodiff Entropy-Aware Variance Optimized Diffusion Inference](image_generation/evodiff_entropy-aware_variance_optimized_diffusion_inference.md)**

:   从信息论角度分析扩散模型推理过程，提出通过优化条件方差来减少条件熵的 EVODiff 方法，在不修改模型的前提下显著加速采样并提升生成质量。

**[Evolve To Inspire Novelty Search For Diverse Image Generation](image_generation/evolve_to_inspire_novelty_search_for_diverse_image_generation.md)**

:   提出 Wander 框架，基于新颖性搜索（novelty search）和 LLM 驱动的 prompt 进化，从单个文本提示出发生成高度多样化的图像集合，在 Vendi Score 上超越现有进化式 prompt 优化基线。

**[Exploring Semantic-Constrained Adversarial Example With Instruction Uncertainty ](image_generation/exploring_semantic-constrained_adversarial_example_with_instruction_uncertainty_.md)**

:   提出多维度指令不确定性缩减框架 InSUR，通过 ResAdv-DDIM 采样器稳定对抗优化方向、上下文编码的攻击场景约束、以及基于 WordNet 的语义抽象评估，首次实现了从自然语言指令生成 2D/3D 语义约束对抗样本（SemanticAE）。

**[Exploring Variational Graph Autoencoders For Distribution Grid Data Generation](image_generation/exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)**

:   系统评估了四种变分图自编码器（VGAE）解码器架构在生成合成配电网拓扑任务上的表现，发现 Iterative-GCN 解码器在小型同质数据集上能较好复现真实电网的结构和频谱特征，但在大型异质数据集上所有方法均存在断连组件和重复模式等严重问题。

**[Failure Prediction At Runtime For Generative Robot Policies](image_generation/failure_prediction_at_runtime_for_generative_robot_policies.md)**

:   提出 FIPER 框架，在生成式机器人策略（扩散/流匹配）运行时，通过观测端 RND-OE（OOD 检测）和动作端 ACE（动作块熵）双指标联合判断，无需任何失败数据即可实现早期、准确的失败预测，并借助共形预测提供统计保证。

**[Fairimagen Post-Processing For Bias Mitigation In Text-To-Image Models](image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)**

:   提出 FairImagen 后处理去偏框架，通过在 CLIP prompt 嵌入空间应用 FairPCA 投影去除人口统计信息，结合经验噪声注入和跨人口统计联合去偏，在不重训模型的前提下显著提升文本到图像生成的公平性。

**[Falcon Few-Step Accurate Likelihoods For Continuous Flows](image_generation/falcon_few-step_accurate_likelihoods_for_continuous_flows.md)**

:   提出 FALCON，通过混合训练目标（flow matching + 平均速度损失 + 可逆性正则化）使连续归一化流在少步采样下仍能提供足够精确的似然估计，从而实现比传统 CNF 快两个数量级的 Boltzmann 采样。

**[Fast Data Attribution For Text-To-Image Models](image_generation/fast_data_attribution_for_text-to-image_models.md)**

:   将精确但缓慢的 Attribution by Unlearning 方法蒸馏到一个轻量特征嵌入空间中，通过 learning-to-rank 训练使得简单的余弦相似度检索就能近似昂贵的归因排序，首次在 Stable Diffusion + LAION-400M 规模上实现毫秒级数据归因。

**[Fast Solvers For Discrete Diffusion Models Theory And Applications Of High-Order](image_generation/fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)**

:   首次将高阶数值方法引入离散扩散模型推理，提出 θ-RK-2 和 θ-Trapezoidal 两种二阶求解器，在理论上证明 θ-Trapezoidal 的离散化误差从一阶 $\mathcal{O}(\kappa T)$ 提升到二阶 $\mathcal{O}(\kappa^2 T)$，实验覆盖 200M–8B 模型在文本、图像和数学推理上的一致性提升。

**[Ferretnet Efficient Synthetic Image Detection Via Local Pixel Dependencies](image_generation/ferretnet_efficient_synthetic_image_detection_via_local_pixel_dependencies.md)**

:   基于马尔可夫随机场（MRF）理论，提出局部像素依赖（LPD）特征表示，通过中值滤波重建暴露生成图像的纹理不一致性，配合仅 1.1M 参数的轻量卷积网络 FerretNet，在仅用 4 类 ProGAN 数据训练的情况下，实现跨 22 个生成模型 97.1% 的平均检测准确率。

**[Flatten Graphs As Sequences Transformers Are Scalable Graph Generators](image_generation/flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)**

:   提出 AutoGraph，通过分段欧拉邻域路径（SENT）将图无损展平为 token 序列，直接用 decoder-only Transformer 建模，实现比扩散模型快 100× 的图生成速度，同时在合成和分子基准上达到 SOTA。

**[Flattening Hierarchies With Policy Bootstrapping](image_generation/flattening_hierarchies_with_policy_bootstrapping.md)**

:   提出 SAW（Subgoal Advantage-Weighted Policy Bootstrapping），通过在数据集内轨迹上采样子目标并用优势值加权的重要性采样进行策略自举，将层次 RL 的长时序推理优势蒸馏到单一扁平策略中，无需学习子目标生成模型，在 20 个离线 GCRL 数据集上匹配或超越 SOTA。

**[Flex-Judge Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators](image_generation/flex-judge_text-only_reasoning_unleashes_zero-shot_multimodal_evaluators.md)**

:   提出 Flex-Judge，仅用 1K 条纯文本推理数据微调多模态大模型，即可零样本泛化到图像/视频/音频/分子等多模态评判任务，性能媲美甚至超越 GPT-4o 等商业 API 和大规模标注训练的专用评估器。

**[Flow Matching Neural Processes](image_generation/flow_matching_neural_processes.md)**

:   提出 FlowNP，将 flow matching 引入神经过程框架，通过 transformer 预测目标点的流速度场实现对条件分布的并行采样，在 1D GP、图像和气象数据三大基准上全面超越现有 NP 方法。

**[Focalcodec Low-Bitrate Speech Coding Via Focal Modulation Networks](image_generation/focalcodec_low-bitrate_speech_coding_via_focal_modulation_networks.md)**

:   提出 FocalCodec——基于 Focal Modulation 的低比特率语音编解码器，使用**单个二值码本**将语音压缩至 0.16–0.65 kbps，在语音重合成、语音转换和多项下游任务中达到与多码本 SOTA 方法可比甚至更优的性能。

**[Freqpolicy Efficient Flow-Based Visuomotor Policy Via Frequency Consistency](image_generation/freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)**

:   首次在 flow-based 视觉运动策略中引入频域一致性约束，利用 DCT 变换将动作块的速度场投影到频域并施加自适应频率分量损失，实现了高质量一步动作生成（93.5 Hz），在仿真和真实机器人任务中均优于现有一步生成方法。

**[From Cradle To Cane A Two-Pass Framework For High-Fidelity Lifespan Face Aging](image_generation/from_cradle_to_cane_a_two-pass_framework_for_high-fidelity_lifespan_face_aging.md)**

:   提出 Cradle2Cane 两阶段人脸老化框架：第一阶段通过自适应噪声注入（AdaNI）实现精准年龄控制，第二阶段通过 SVR-ArcFace 和 Rotate-CLIP 双身份嵌入（IDEmb）强化身份一致性，在全寿命跨度（0-80岁）人脸老化中实现年龄精度与身份保持的最优平衡。

**[Geneman Generalizable Single-Image 3D Human Reconstruction From Multi-Source Hum](image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)**

:   GeneMAN 提出一种**无需人体参数模型(如 SMPL)**的通用单图 3D 人体重建框架，通过在大规模多源人体数据上训练人体专属的 2D/3D 扩散先验模型，结合几何初始化-雕刻流水线与多空间纹理精炼，实现了对野外图片中不同体型比例、复杂姿态与个人物品的高保真 3D 人体重建。

**[Generative Model Inversion Through The Lens Of The Manifold Hypothesis](image_generation/generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)**

:   从流形几何视角揭示生成式模型逆向攻击 (MIA) 的本质是通过将损失梯度投影到生成器切空间实现隐式去噪，提出梯度-流形对齐假说（对齐越高→模型越脆弱）并设计无需训练的 AlignMI 方法在多个 SOTA 攻击上取得一致且显著的提升。

**[Genir Generative Visual Feedback For Mental Image Retrieval](image_generation/genir_generative_visual_feedback_for_mental_image_retrieval.md)**

:   提出 GenIR，一种利用文本到图像扩散模型生成"合成视觉反馈"的多轮交互式图像检索框架，将系统对用户查询的理解显式可视化，使用户能直观地识别差异并迭代改进查询，在 Mental Image Retrieval (MIR) 任务上大幅超越纯文本反馈方法。

**[Georemover Removing Objects And Their Causal Visual Artifacts](image_generation/georemover_removing_objects_and_their_causal_visual_artifacts.md)**

:   提出几何感知的两阶段框架 GeoRemover，将目标移除解耦为几何移除（深度域）与外观渲染（RGB域），通过修改场景几何表示来隐式消除被移除物体的阴影和反射等因果视觉伪影。

**[Gradient Variance Reveals Failure Modes In Flow-Based Generative Models](image_generation/gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)**

:   本文通过分析 CFM 损失的梯度方差（gradient variance），揭示了 Rectified Flow 在确定性插值下会不可避免地记忆训练配对而非学习最优传输映射，并证明引入随机性（stochastic interpolant）可打破该记忆化通道、恢复泛化能力。

**[Gralora Granular Low-Rank Adaptation For Parameter-Efficient Fine-Tuning](image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)**

:   提出 GraLoRA——将 LoRA 的权重更新矩阵分割为 $k^2$ 个独立子块、每块配独立低秩适配器，在不增加参数量和计算量的前提下将有效秩从 $r$ 提升至 $kr$，解决 LoRA 在高秩下因梯度纠缠导致的性能退化问题，在代码生成上 Pass@1 最高提升 +8.5%。

**[Graph-Based Neural Space Weather Forecasting](image_generation/graph-based_neural_space_weather_forecasting.md)**

:   提出基于图神经网络的空间天气神经模拟器，在 Vlasiator 混合 Vlasov 模拟数据上训练，实现确定性和概率性自回归预测近地空间状态，速度比原始模拟快 100 倍以上，并通过隐变量生成集合预报来量化预测不确定性。

**[Graph Diffusion That Can Insert And Delete](image_generation/graph_diffusion_that_can_insert_and_delete.md)**

:   提出 GrIDDD 模型，首次将离散去噪扩散模型（DDPM）扩展为支持在生成过程中**动态插入和删除图节点**，使分子图的大小可在扩散过程中自适应变化，在性质靶向和分子优化任务上达到或超过现有方法。

**[Graph Distance As Surprise Free Energy Minimization In Knowledge Graph Reasoning](image_generation/graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)**

:   将神经科学的 Free Energy Principle (FEP) 与知识图谱推理连接，提出用图的最短路径距离作为 surprise 的度量，将 Murphy et al. 的树结构 surprise 理论推广到一般有向图，为 KG-based agent 的 entity grounding 提供了一个有原则的理论框架。

**[Grasp2Grasp Vision-Based Dexterous Grasp Translation Via Schrödinger Bridges](image_generation/grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)**

:   提出将跨手形态的视觉灵巧抓取迁移建模为 Schrödinger Bridge 问题，通过在潜空间中学习得分与流匹配（[SF]²M），并设计物理感知的最优传输代价函数（位姿/接触图/力旋量空间/雅可比可操作性），在无需配对数据的条件下实现不同机械手之间抓取意图的分布级迁移。

**[Gspn-2 Efficient Parallel Sequence Modeling](image_generation/gspn-2_efficient_parallel_sequence_modeling.md)**

:   GSPN-2 通过算法-系统联合重设计（单 kernel 融合、紧凑通道传播、共享内存优化），将 GSPN-1 的 2D 空间传播加速最高 40×，在 ImageNet 分类和文本到图像生成中达到 Transformer 级精度且计算成本显著更低。

**[Guided Diffusion Sampling On Function Spaces With Applications To Pdes](image_generation/guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)**

:   提出 **FunDPS（Function-space Diffusion Posterior Sampling）**，在函数空间中训练无条件扩散模型，推理时通过梯度引导实现 plug-and-play 的 PDE 逆问题后验采样；理论上将 Tweedie 公式推广到无穷维 Banach 空间，实验上在 5 个 PDE 任务中仅用 3% 观测即可获得比 DiffusionPDE 平均高 32% 的精度并减少 4 倍采样步数。

**[Guideflow3D Optimization-Guided Rectified Flow For Appearance Transfer](image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)**

:   提出 GuideFlow3D，一种无需训练的 3D 外观迁移框架，通过在预训练 rectified flow 模型的采样过程中交替注入可微引导损失（部件感知外观损失 + 自相似性损失），实现几何差异显著的物体间鲁棒的纹理与几何细节迁移。

**[Hallucination As An Upper Bound A New Perspective On Text-To-Image Evaluation](image_generation/hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)**

:   提出将文本到图像（T2I）模型中的幻觉定义为**偏差驱动的偏离**，建立了包含属性、关系和物体三类幻觉的分类学，并论证幻觉评估作为提示对齐评估的"上界"，可揭示模型隐藏偏差。

**[Head Pursuit Probing Attention Specialization In Multimodal](image_generation/head_pursuit_probing_attention_specialization_in_multimodal.md)**

:   将经典稀疏信号恢复算法（SOMP）重新解释为一种多样本可解释性工具，发现 LLM 和 VLM 中注意力头存在细粒度语义专业化现象，仅通过翻转约 1% 的头即可可靠地抑制或增强特定概念（如国家名、毒性内容、颜色等）的生成。

**[Hephaestus Mixture Generative Modeling With Energy Guidance For Large-Scale Qos ](image_generation/hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)**

:   提出 Hephaestus 三阶段生成框架（Forge-Morph-Refine），结合预测路径加压算法、能量引导的混合 CVAE 和潜在空间 RL 优化，用于大规模网络 QoS 降级问题的求解。

**[Hierarchical Koopman Diffusion Fast Generation With Interpretable Diffusion Traj](image_generation/hierarchical_koopman_diffusion_fast_generation_with_interpretable_diffusion_traj.md)**

:   基于 Koopman 算子理论，将扩散模型的非线性去噪动力学提升到线性 Koopman 空间，通过层次化分解实现一步采样，同时保留中间生成状态的可解释性和可控性。

**[High-Order Equivariant Flow Matching For Density Functional Theory Hamiltonian P](image_generation/high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)**

:   提出 QHFlow，首次将条件 flow matching 引入密度泛函理论（DFT）哈密顿矩阵预测任务，通过高阶 SE(3) 等变向量场和对称性感知先验分布，在 MD17 上将哈密顿预测误差降低 73%，并可作为 SCF 初始化加速 DFT 计算达 54%。

**[Highlighting What Matters Promptable Embeddings For Attribute-Focused Image Retr](image_generation/highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)**

:   提出可提示图像嵌入(Promptable Embeddings)方法，通过在检索时高亮目标视觉属性来提升属性聚焦的文本到图像检索性能，同时构建了COCO-Facet基准数据集。

**[Hollowflow Efficient Sample Likelihood Evaluation Using Hollow Message Passing](image_generation/hollowflow_efficient_sample_likelihood_evaluation_using_hollow_message_passing.md)**

:   提出HollowFlow框架，通过非回溯图神经网络（NoBGNN）和Hollow消息传递机制强制速度场雅可比矩阵具有块对角结构，将连续归一化流的似然计算反向传播次数从$\mathcal{O}(n)$降至常数$\mathcal{O}(d)$，实现高达$10^2$倍的采样加速。

**[How To Build A Consistency Model Learning Flow Maps Via Self-Distillation](image_generation/how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)**

:   提出统一的自蒸馏（Self-Distillation）框架来直接学习 flow map（即 consistency model 的一般化形式），通过 tangent condition 将任意蒸馏方案转化为无需预训练教师的直接训练算法，并导出三大算法族（Eulerian / Lagrangian / Progressive），其中 Lagrangian 方法避免了空间梯度和自举引导，训练最稳定、性能最优。

**[Image Super-Resolution With Guarantees Via Conformalized Generative Models](image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)**

:   基于共形预测（Conformal Prediction）技术，为生成式图像超分辨率模型构建二值"置信度掩码"，能可靠地标识生成图像中可信赖的区域，并提供严格的统计保证。

**[Imagesentinel Protecting Visual Datasets From Unauthorized Retrieval-Augmented I](image_generation/imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)**

:   提出 ImageSentinel 框架，通过合成与私有数据集视觉一致的哨兵图像（sentinel images）并绑定随机字符检索键，实现对检索增强图像生成（RAIG）系统未授权使用私有数据集的可靠检测——仅需 3–10 次查询即可达到接近 100% 的 AUC。

**[Improved Training Technique For Shortcut Models](image_generation/improved_training_technique_for_shortcut_models.md)**

:   针对 Shortcut Models 的五大性能瓶颈（指导累积、固定引导、频率偏差、自一致性偏离、弯曲轨迹），提出 iSM 统一训练框架，通过内禀引导、多级小波损失、缩放最优传输和双 EMA 策略，在 ImageNet 256×256 上实现单步 FID 5.27、四步 FID 2.05 的大幅提升。

**[Improving Posterior Inference Of Galaxy Properties With Image-Based Conditional ](image_generation/improving_posterior_inference_of_galaxy_properties_with_image-based_conditional_.md)**

:   提出基于条件流匹配（CFM）的框架，将星系图像的形态学信息与测光数据联合建模，显著提升星系物理属性（恒星质量、恒星形成率、金属丰度、尘埃消光等）的后验推断精度。

**[In-Context Edit Enabling Instructional Image Editing With In-Context Generation ](image_generation/in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)**

:   ICEdit 提出一种基于大规模 Diffusion Transformer (DiT) 的上下文编辑范式，通过 in-context prompt + 最小化 LoRA-MoE 微调 + VLM 早期筛选推理时缩放，仅用 0.1% 训练数据即达到 SOTA 编辑性能。

**[Increasing The Utility Of Synthetic Images Through Chamfer Guidance](image_generation/increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)**

:   提出 Chamfer Guidance——一种免训练的推理时引导方法，利用少量真实样本作为参照，通过 Chamfer 距离同时优化合成图像的质量（fidelity）和多样性（diversity），在 ImageNet-1k 上仅用 32 张真实图片即可达到 97.5% Precision 和 92.7% Coverage，并在下游分类器训练中带来最高 16% 的准确率提升。

**[Inference-Time Scaling For Flow Models Via Stochastic Generation And Rollover Bu](image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)**

:   提出针对 Flow 模型的推理时扩展方法：通过 ODE→SDE 转换引入随机性以启用粒子采样，利用线性→VP 插值变换扩大搜索空间，并设计 Rollover Budget Forcing (RBF) 策略自适应分配计算预算，在组合文本生成图像和数量感知生成任务上显著超越所有现有方法。

**[Infinitystar Unified Spacetime Autoregressive Modeling For V](image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)**

:   提出 InfinityStar，首个能生成工业级 720p 视频的纯离散自回归模型，通过时空金字塔建模统一 T2I/T2V/I2V/交互式长视频生成，VBench 83.74 超越 HunyuanVideo，推理速度比扩散模型快 10-32×。

**[Information-Theoretic Discrete Diffusion](image_generation/information-theoretic_discrete_diffusion.md)**

:   将连续扩散中经典的 I-MMSE 恒等式推广到离散域，建立 I-MDSE 和 I-MDCE 关系——证明 DSE/DCE 损失不仅是变分上界而是对数似然的**精确分解**，并由此推导出 time-free 公式、条件似然估计和耦合似然比估计器，在 LLaDA 等大模型上验证了低方差和 OOD 检测能力。

**[Information Theoretic Learning For Diffusion Models With Warm Start](image_generation/information_theoretic_learning_for_diffusion_models_with_warm_start.md)**

:   提出将经典 KL 散度-Fisher 信息关系推广到任意各向同性噪声扰动的似然估计框架，结合 warm-start 噪声注入和重要性采样，消除训练-测试差距并实现更紧的似然上界，在 ImageNet 多分辨率上达到 SOTA NLL。

**[Instance-Level Composed Image Retrieval](image_generation/instance-level_composed_image_retrieval.md)**

:   提出实例级组合图像检索（i-CIR）基准和训练免费方法BASIC，通过独立估计图像和文本查询的相似度并进行乘法融合，在无需训练的情况下在i-CIR和现有CIR数据集上均达到SOTA。

**[Is Artificial Intelligence Generated Image Detection A Solved Problem](image_generation/is_artificial_intelligence_generated_image_detection_a_solved_problem.md)**

:   提出 AIGIBench 综合基准，通过四大任务（多源泛化、多退化鲁棒性、数据增强敏感性、测试预处理影响）系统评估 11 个 SOTA 检测器，揭示现有 AIGI 检测方法在真实场景下性能严重下降，表明该问题远未解决。

**[Itdpdm Information-Theoretic Discrete Poisson Diffusion Model](image_generation/itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)**

:   提出 ItDPDM（信息论离散泊松扩散模型），通过泊松噪声信道和泊松重建损失（PRL）实现非负离散数据的精确似然估计，避免了 ELBO 近似和 dequantization，在合成数据及 CIFAR-10 和 MIDI 音乐上取得优于现有离散扩散模型的似然估计。

**[Janus-Pro-R1 Advancing Collaborative Visual Comprehension And Generation Via Rei](image_generation/janus-pro-r1_advancing_collaborative_visual_comprehension_and_generation_via_rei.md)**

:   提出 Janus-Pro-R1，通过两阶段训练（SFT + RL）实现视觉理解与生成的协同共进，让 MLLM 在文本到图像生成中形成真正的 Chain-of-Thought 并触发 Aha 时刻，在 GenEval 上超越 GPT-4o，同时拓展到图像编辑任务。

**[Klass Kl-Guided Fast Inference In Masked Diffusion Models](image_generation/klass_kl-guided_fast_inference_in_masked_diffusion_models.md)**

:   提出 KLASS（KL-Adaptive Stability Sampling），一种无需训练的采样方法，利用 token 级别的 KL 散度和置信度来识别稳定 token 并行解码，在掩码扩散模型上实现最高 2.78× 加速且不损失甚至提升生成质量。

**[Knowledge Distillation Detection For Open-Weights Models](image_generation/knowledge_distillation_detection_for_open-weights_models.md)**

:   提出知识蒸馏检测任务，通过无数据输入合成和统计评分框架，判断一个开放权重的学生模型是否由特定教师模型蒸馏而来。

**[Kuramoto Orientation Diffusion Models](image_generation/kuramoto_orientation_diffusion_models.md)**

:   将生物系统中的Kuramoto同步动力学引入score-based生成模型，在周期域上构建前向同步/反向去同步的扩散框架，对指纹、纹理等方向密集数据实现显著优于标准扩散模型的生成质量，同时在CIFAR-10上保持竞争力。

**[Large-Scale Training Data Attribution For Music Generative Models Via Unlearning](image_generation/large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)**

:   将基于机器遗忘（machine unlearning）的训练数据归因方法应用于大规模文本到音乐扩散模型（115K 音轨），通过网格搜索找到最优超参数配置，并与非反事实方法对比，验证了 unlearning-based TDA 在音乐生成领域的可行性。

**[Latent Zoning Network A Unified Principle For Generative Modeling Representation](image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)**

:   提出 Latent Zoning Network (LZN)——一种通过共享高斯潜在空间将生成建模、表征学习和分类统一在同一框架下的方法，每种数据类型配备编码器-解码器对将样本映射到不相交的潜在区域，仅依赖"潜在计算"和"潜在对齐"两个原子操作即可支持多种 ML 任务，并在 CIFAR10 上将无条件生成 FID 从 2.76 降至 2.59，在 ImageNet 线性分类上超越 SimCLR。

**[Leapfactual Reliable Visual Counterfactual Explanation Using Conditional Flow Ma](image_generation/leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)**

:   提出LeapFactual，一种基于条件流匹配(CFM)的反事实解释算法，通过"起飞-降落"(Leap)机制在扁平化和结构化潜在空间之间建立桥梁，生成可靠且分布内的反事实样本，即使在学习决策边界与真实边界不一致时也能有效工作。

**[Learnable Sampler Distillation For Discrete Diffusion Models](image_generation/learnable_sampler_distillation_for_discrete_diffusion_models.md)**

:   提出LSD和LSD+方法，通过蒸馏将高保真教师采样器的中间分数轨迹知识迁移给少步数学生采样器，以可学习的采样系数和非均匀时间调度实现离散扩散模型的高效高质量采样。

**[Learning Interpretable Features In Audio Latent Spaces Via Sparse Autoencoders](image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)**

:   提出一种通过稀疏自编码器（SAE）从音频生成模型的潜空间中提取可解释特征的框架，通过线性探针将 SAE 特征映射到人类可理解的声学概念（音高、振幅、音色），实现对音频生成过程的可控操作和可视化分析。

**[Learning To Integrate Diffusion Odes By Averaging The Derivatives](image_generation/learning_to_integrate_diffusion_odes_by_averaging_the_derivatives.md)**

:   提出"割线损失"(Secant Losses)家族，通过蒙特卡洛积分和Picard迭代学习扩散ODE的积分，将扩散模型的切线逐步延展为割线，在训练稳定性和少步推理之间取得优异平衡。

**[Linear Differential Vision Transformer Learning Visual Contrasts Via Pairwise Di](image_generation/linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)**

:   提出 Visual-Contrast Attention (VCA)，通过空间池化生成紧凑的正负视觉对比 token 并进行差分交互，将自注意力复杂度从 $O(N^2C)$ 降至 $O(NnC)$（$n \ll N$），同时在图像分类和生成任务上均获得显著提升。

**[Lineas End-To-End Learning Of Activation Steering With A Distributional Loss](image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)**

:   提出 LinEAS（Linear End-to-end Activation Steering），通过端到端优化跨层仿射变换映射，利用 1D Wasserstein 分布损失进行全局激活值对齐，仅需 32 个无配对样本即可高效控制 LLM 毒性和 T2I 模型概念生成。

**[Llm Meets Diffusion A Hybrid Framework For Crystal Material Generation](image_generation/llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)**

:   提出CrysLLMGen混合框架，结合LLM擅长离散原子类型预测和扩散模型擅长连续坐标/晶格参数建模的互补优势，在晶体材料生成任务中同时实现高结构有效性和组成有效性。

**[Mge-Ldm Joint Latent Diffusion For Simultaneous Music Generation And Source Extr](image_generation/mge-ldm_joint_latent_diffusion_for_simultaneous_music_generation_and_source_extr.md)**

:   提出 MGE-LDM，首个在统一的潜在扩散框架中同时实现音乐混合生成、部分生成（源补全）和文本驱动任意源提取的模型，通过联合建模混合-子混合-源三元组并利用扩散修复（inpainting）实现各任务。

**[Mind-The-Glitch Visual Correspondence For Detecting Inconsistencies In Subject-D](image_generation/mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)**

:   提出从预训练扩散模型骨干网络中解耦语义特征和视觉特征的框架，实现视觉对应匹配，并基于此提出 Visual Semantic Matching (VSM) 度量，首次同时支持主体驱动图像生成中视觉不一致性的**量化和空间定位**。

**[Mitigating Intra- And Inter-Modal Forgetting In Continual Learning Of Unified Mu](image_generation/mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)**

:   提出Modality-Decoupled Experts (MoDE)，通过将文本和图像的适配器解耦为独立的T-MoE和V-Adapter子空间，配合知识蒸馏，在统一多模态生成模型的持续指令微调中同时缓解模态内遗忘和模态间遗忘。

**[Mitigating Sexual Content Generation Via Embedding Distortion In Text-Conditione](image_generation/mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)**

:   提出Distorting Embedding Space (DES)，一种基于文本编码器的防御框架，通过将不安全嵌入变换到安全区域、保持安全嵌入不变、中和"裸露"语义三管齐下，在FLUX.1和SD v1.5上实现SOTA的性内容缓解效果（ASR分别降至9.47%和0.52%），同时保持良好的良性图像质量。

**[Mmada Multimodal Large Diffusion Language Models](image_generation/mmada_multimodal_large_diffusion_language_models.md)**

:   提出 MMaDA，首个在统一离散扩散架构下同时实现文本推理、多模态理解和文本到图像生成的多模态基础模型，通过混合长 CoT 微调和 UniGRPO 强化学习算法弥合了扩散模型预训练与后训练之间的鸿沟。

**[Mmg Mutual Information Estimation Via The Mmse Gap In Diffusion](image_generation/mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)**

:   利用扩散模型的信息论公式，证明互信息等于条件与无条件去噪 MMSE 之间的差值在所有信噪比上的积分的一半，提出 MMG 估计器，结合自适应重要性采样和正交原理显著提升估计精度和稳定性。

**[Model-Guided Dual-Role Alignment For High-Fidelity Open-Domain Video-To-Audio Ge](image_generation/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)**

:   提出MGAudio，首个采用模型引导(MG)训练替代无分类器引导(CFG)的视频到音频生成框架，结合双角色音视频编码器（同时用于条件注入和特征对齐），以131M参数在VGGSound上实现SOTA（FAD=0.40），且仅用10%数据即可超越多数方法。

**[Moment- And Power-Spectrum-Based Gaussianity Regularization For Text-To-Image Mo](image_generation/moment-_and_power-spectrum-based_gaussianity_regularization_for_text-to-image_mo.md)**

:   提出统一的标准高斯性正则化框架，结合空间域的矩(moment)匹配和频谱域的功率谱(power spectrum)匹配，将KL散度、峰度、范数等现有正则化方法统一为特殊情况，并以$\mathcal{O}(D\log D)$复杂度实现了PRNO的$\mathcal{O}(D^2)$等价效果，在文本到图像模型的reward alignment任务中显著优于所有基线。

**[Multimodal Generative Flows For Lhc Jets](image_generation/multimodal_generative_flows_for_lhc_jets.md)**

:   提出基于 Transformer 的多模态流匹配框架（MMF），将连续流匹配与连续时间马尔可夫跳跃桥联合建模，实现对 LHC 喷注中粒子运动学（连续）和 flavor 量子数（离散）的统一生成。

**[Neural Entropy](image_generation/neural_entropy.md)**

:   本文通过扩散模型的范式探索深度学习与信息论的联系，引入"神经熵"度量来量化扩散过程中存储在神经网络里的信息量，揭示了图像扩散模型对结构化数据具有极高的压缩效率。

**[Next Semantic Scale Prediction Via Hierarchical Diffusion Language Models](image_generation/next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)**

:   提出 HDLM（Hierarchical Diffusion Language Model），通过在 clean token 和 mask token 之间引入具有粗粒度语义的聚类 token 中间层级，实现"下一语义尺度预测"的离散扩散语言建模，推导闭式 ELBO，在 OpenWebText 上困惑度一致优于 MDLM/GIDD，随机扰动后生成困惑度降低 62%。

**[Non-Asymptotic Analysis Of Data Augmentation For Precision Matrix Estimation](image_generation/non-asymptotic_analysis_of_data_augmentation_for_precision_matrix_estimation.md)**

:   本文从非渐近角度分析了高维精度矩阵（逆协方差矩阵）估计中数据增强（DA）的效果，建立了线性收缩估计器和 DA 估计器的二次误差集中界，并引入了广义预解矩阵的新型确定性等价工具。

**[Non-Markovian Discrete Diffusion With Causal Language Models](image_generation/non-markovian_discrete_diffusion_with_causal_language_models.md)**

:   提出CaDDi框架，通过非马尔可夫离散扩散过程让每步去噪都能访问完整生成轨迹，并将其统一到因果语言模型架构中，使预训练LLM可直接复用为离散扩散模型。

**[Npn Non-Linear Projections Of The Null-Space For Imaging Inverse Problems](image_generation/npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)**

:   提出非线性零空间投影 (NPN)——一种新型正则化策略，训练神经网络从观测中预测信号在感知矩阵零空间低维子空间上的投影系数，将此作为"看不见的特征"的先验约束，可灵活嵌入 PnP、展开网络、DIP 和扩散模型等多种重建框架，理论证明了 PnP 算法中的收敛加速。

**[Obclip Oblivious Cloud-Device Hybrid Image Generation With Privacy Preservation](image_generation/obclip_oblivious_cloud-device_hybrid_image_generation_with_privacy_preservation.md)**

:   提出 ObCLIP，一种遗忘式云-端混合图像生成方案：将用户 prompt 扩展为一组仅在敏感属性（性别、种族等）上不同的候选 prompt，云端处理所有候选的早期去噪步骤而无法识别真实 prompt，客户端选择正确的中间潜变量完成剩余去噪，同时通过时间和批次冗余加速将额外开销降至 4.4~7.6 倍以下。

**[Omnicast A Masked Latent Diffusion Model For Weather Forecasting Across Time Sca](image_generation/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)**

:   提出 OmniCast，一种结合掩码生成框架和潜在扩散模型的天气预报方法，通过联合生成未来天气序列（而非自回归迭代）来缓解误差累积，在次季节至季节（S2S）尺度达到 SOTA 性能，同时在中期预报上保持竞争力且推理速度快 10-20 倍。

**[Omnisync Towards Universal Lip Synchronization Via Diffusion](image_generation/omnisync_towards_universal_lip_synchronization_via_diffusion.md)**

:   OmniSync提出了一种基于Diffusion Transformer的通用唇形同步框架，通过无掩码训练范式、基于Flow Matching的渐进噪声初始化和动态时空CFG三大创新，在真实视频和AI生成视频上都大幅超越先前方法，尤其在风格化角色的唇形同步上达到87.78%成功率（之前最佳67.78%）。

**[Omnivcus Feedforward Subject-Driven Video Customization With Multimodal Control ](image_generation/omnivcus_feedforward_subject-driven_video_customization_with_multimodal_control_.md)**

:   OmniVCus 提出了一个前馈式 DiT 框架，通过数据构建流水线 VideoCus-Factory 和两种嵌入机制（Lottery Embedding 和 Temporally Aligned Embedding），实现了多主体、多模态控制条件下的视频定制生成，在身份保持和可控性上显著超越 SOTA。

**[On Optimal Steering To Achieve Exact Fairness](image_generation/on_optimal_steering_to_achieve_exact_fairness.md)**

:   本文定义了"理想分布"——使任意代价敏感风险下的 Bayes 最优分类器都满足精确公平性的数据分布，并提出通过 KL 散度最小化寻找最近理想分布的优化框架，为公平预处理和 LLM 表示引导提供了可证明的公平性保证。

**[On The Emergence Of Linear Analogies In Word Embeddings](image_generation/on_the_emergence_of_linear_analogies_in_word_embeddings.md)**

:   提出一个基于二值语义属性的词共现生成模型，解析性地证明了词嵌入中线性类比结构（如 $W_{\text{king}} - W_{\text{man}} + W_{\text{woman}} \approx W_{\text{queen}}$）的涌现机制，统一解释了已知的四个关键观测现象。

**[On The Relation Between Rectified Flows And Optimal Transport](image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)**

:   本文深入研究了 rectified flow（流匹配）与最优传输之间的理论关系，通过构造多个反例证明了此前文献中关于"梯度约束的 rectified flow 可以渐近收敛到最优传输"的等价性声明并不成立，需要比已知条件更强的假设才能保证两者的等价关系。

**[One Stone With Two Birds A Null-Text-Null Frequency-Aware Diffusion Models For T](image_generation/one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)**

:   提出NTN-Diff频率感知扩散模型，通过将语义一致性问题分解为中频和低频频带各自的一致性任务，利用"空文本-文本-空文本"三阶段去噪策略，同时解决文本引导图像修复中的未遮盖区域保持和遮盖/未遮盖区域语义一致性两大挑战。

**[Orient Anything V2 Unifying Orientation And Rotation Understanding](image_generation/orient_anything_v2_unifying_orientation_and_rotation_understanding.md)**

:   Orient Anything V2 通过可扩展的合成数据引擎、对称感知的周期分布目标和多帧架构，统一了物体3D方向和旋转理解，在方向估计、6DoF位姿估计和对称性识别三个任务上均达到 zero-shot SOTA。

**[Osmgen Highly Controllable Satellite Image Synthesis Using Openstreetmap Data](image_generation/osmgen_highly_controllable_satellite_image_synthesis_using_openstreetmap_data.md)**

:   OSMGen 直接从 OSM JSON 数据（矢量几何、语义标签、位置和时间信息）合成高保真卫星图像，并通过 DDIM 反演生成一致的前后对比图像对，支持城市变化模拟和数据增强。

**[Overt A Benchmark For Over-Refusal Evaluation On Text-To-Image Models](image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)**

:   构建了首个大规模文生图模型过度拒绝评估基准 OVERT（4600条良性提示 + 1785条有害提示，覆盖9个安全类别），系统评估了5个主流 T2I 模型的过度拒绝行为，揭示了安全与效用之间的强相关权衡关系。

**[Pairwise Optimal Transports For Training All-To-All Flow-Based Condition Transfe](image_generation/pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)**

:   提出A2A-FM方法，通过一种新颖的代价函数在FlowMatching框架中同时学习所有条件分布对之间的最优传输映射，理论证明在无限样本极限下收敛至逐对最优传输，尤其适用于连续条件变量的非分组数据场景。

**[Panel-By-Panel Souls A Performative Workflow For Expressive Faces In Ai-Assisted](image_generation/panel-by-panel_souls_a_performative_workflow_for_expressive_faces_in_ai-assisted.md)**

:   本文提出了一种面向漫画创作的"双混合"交互式工作流系统，在面部准备阶段融合关键点自动检测与手动框选、在表情映射阶段融合表演性视频捕捉与精细数值滑块调控，使漫画艺术家能够直觉化地将情感叙事意图注入AI生成的漫画面板中角色的面部表情，解决文本提示无法精确描述微妙情感这一根本性"语言-视觉鸿沟"问题。

**[Perturb A Model Not An Image Towards Robust Privacy Protection Via Anti-Personal](image_generation/perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)**

:   提出Anti-Personalized Diffusion Model (APDM)，首次将隐私保护从数据级（图像扰动）转移到模型级（参数更新），通过Direct Protective Optimization损失和Learning to Protect双路径优化策略，鲁棒地阻止扩散模型对特定主体的个性化，同时保持模型对其他主体的生成和个性化能力。

**[Physics-Constrained Flow Matching Sampling Generative Models With Hard Constrain](image_generation/physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)**

:   提出 Physics-Constrained Flow Matching (PCFM)，一种零样本推理框架，通过在预训练流匹配模型的采样过程中交替执行前向投射、OT 插值反向更新和松弛惩罚校正，实现任意非线性等式约束的精确满足（达到机器精度），在含激波和间断的 PDE 问题上相比基线方法提升高达 99.5%。

**[Physics-Driven Spatiotemporal Modeling For Ai-Generated Video Detection](image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)**

:   提出基于物理守恒定律的AI生成视频检测范式，定义归一化时空梯度（NSG）统计量来捕获空间概率梯度与时间密度变化的比率，利用预训练扩散模型估计NSG并通过MMD进行检测，在Recall上超越SOTA 16%、F1超越10.75%。

**[Pid-Controlled Langevin Dynamics For Faster Sampling Of Generative Models](image_generation/pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)**

:   将 PID 控制理论引入 Langevin 动力学采样，利用梯度历史（积分项）提供动量穿越能量壁垒、利用梯度趋势（微分项）抑制振荡实现快速稳定收敛，无需额外训练即可在 SGM 和 EBM 上实现 10 倍以上采样加速。

**[Pixperfect Seamless Latent Diffusion Local Editing With Discriminative Pixel-Spa](image_generation/pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)**

:   提出 PixPerfect，一个通用的像素级精修框架，通过判别性像素空间损失和全面的伪影模拟管线，消除潜在扩散模型局部编辑中的色差、纹理不匹配和可见接缝，在修复、目标移除和插入任务上大幅提升视觉保真度。

**[Pragmatic Heterogeneous Collaborative Perception Via Generative Communication Me](image_generation/pragmatic_heterogeneous_collaborative_perception_via_generative_communication_me.md)**

:   提出GenComm——一种基于生成式通信机制的异构多智能体协作感知方法，通过空间消息提取和条件扩散模型在ego端生成对齐的协作者特征，无需修改原始网络即可以极低代价接纳新异构智能体。

**[Preconditioned Langevin Dynamics With Score-Based Generative Models For Infinite](image_generation/preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)**

:   在无穷维 Hilbert 空间中严格分析了分数生成模型 (SGM) 驱动的 Langevin 后验采样器，首次推导出依赖分数近似误差的收敛界，并发现了同时依赖前向算子和分数误差的最优预条件器形式，保证所有后验模态的均匀收敛速率。

**[Predictive Feature Caching For Training-Free Acceleration Of Molecular Geometry ](image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)**

:   将图像领域的预测式特征缓存（predictive feature caching）策略迁移到分子几何生成领域，利用采样轨迹中隐藏状态的时间平滑性，实现免训练的2-3倍推理加速，且与其他优化手段组合可达7倍加速。

**[Preventing Shortcuts In Adapter Training Via Providing The Shortcuts](image_generation/preventing_shortcuts_in_adapter_training_via_providing_the_shortcuts.md)**

:   提出Shortcut-Rerouted Adapter Training，通过在adapter训练过程中主动提供confounding因素的专用通路（如LoRA吸收分布偏移、ControlNet吸收姿态/表情），使adapter只学习目标属性（如身份），推理时移除辅助模块即可获得去纠缠的适配器。

**[Progressive Inference-Time Annealing Of Diffusion Models For Sampling From Boltz](image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)**

:   提出 PITA（Progressive Inference-Time Annealing），一种结合温度退火与扩散平滑两种互补插值策略的框架，通过在高温下训练初始扩散模型，然后利用新颖的 Feynman-Kac PDE 与 SMC 重采样在推理时降温生成低温样本，逐步训练一系列扩散模型直达目标温度，首次实现了对丙氨酸二肽和三肽的笛卡尔坐标下平衡态采样。

**[Prompt-Based Safety Guidance Is Ineffective For Unlearned Text-To-Image Diffusio](image_generation/prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)**

:   本文发现训练式概念遗忘（unlearning）与免训练安全引导（negative prompt guidance）两种安全方法组合后效果反而下降，提出用概念反演（Concept Inversion）获得的隐式负向嵌入替换显式负向提示，有效恢复了免训练方法在遗忘模型上的防御能力。

**[Psi-Sampler Initial Particle Sampling For Smc-Based Inference-Time Reward Alignm](image_generation/psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)**

:   提出Ψ-Sampler框架，在SMC（序贯蒙特卡洛）推理时奖励对齐中引入基于pCNL（预条件Crank-Nicolson Langevin）算法的初始粒子采样，从奖励感知的后验分布初始化粒子，显著提升布局生成、数量感知生成和美学偏好生成的对齐效果。

**[Rare Text Semantics Were Always There In Your Diffusion Transformer](image_generation/rare_text_semantics_were_always_there_in_your_diffusion_transformer.md)**

:   发现通过在 MM-DiT 的联合注意力块前对文本 token 嵌入进行方差放大,即可在无需额外训练或外部模块的情况下让扩散模型呈现稀有文本语义。

**[Real-Time Execution Of Action Chunking Flow Policies](image_generation/real-time_execution_of_action_chunking_flow_policies.md)**

:   提出 Real-Time Chunking (RTC)，将异步动作分块执行建模为修复（inpainting）问题，通过冻结已执行动作并"修复"其余部分，实现扩散/流策略的实时平滑执行，无需重新训练。

**[Recurrent Memory For Online Interdomain Gaussian Processes](image_generation/recurrent_memory_for_online_interdomain_gaussian_processes.md)**

:   提出 OHSVGP（Online HiPPO Sparse Variational Gaussian Process），将深度学习中的 HiPPO（高阶多项式投影算子）框架引入稀疏变分高斯过程中作为跨域诱导变量，利用时变正交多项式基函数实现在线学习中的长期记忆保持，核矩阵可通过 ODE 递推高效更新。

**[Reinforcing The Diffusion Chain Of Lateral Thought With Diffusion Language Model](image_generation/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)**

:   提出扩散横向思维链（DCoLT），将扩散语言模型逆向过程中的每个中间步视为潜在"思考"动作，通过基于最终结果的强化学习优化整条推理轨迹，在SEDD和LLaDA两种扩散语言模型上实现了数学和代码生成的SOTA表现。

**[Remasking Discrete Diffusion Models With Inference-Time Scaling](image_generation/remasking_discrete_diffusion_models_with_inference-time_scaling.md)**

:   提出 ReMDM 采样器，通过在生成过程中允许已解码 token 被重新掩码（remask），赋予离散掩码扩散模型迭代纠错能力，实现推理时计算缩放，在文本、图像和分子设计任务上显著提升采样质量。

**[Repldm Reprogramming Pretrained Latent Diffusion Models For High-Quality High-Ef](image_generation/repldm_reprogramming_pretrained_latent_diffusion_models_for_high-quality_high-ef.md)**

:   提出 RepLDM 重编程框架,通过注意力引导阶段和渐进上采样两个阶段,让预训练的潜在扩散模型无需重训练即可生成高质量高分辨率图像,同时大幅提升效率。

**[Respodiff Dual-Module Bottleneck Transformation For Responsible Faithful T2I Gen](image_generation/respodiff_dual-module_bottleneck_transformation_for_responsible_faithful_t2i_gen.md)**

:   提出RespoDiff框架，在扩散模型UNet的瓶颈层引入双模块可学习变换——负责任概念对齐模块(RAM)和语义对齐模块(SAM)，通过分数匹配目标实现公平和安全的文本到图像生成，同时保持图像质量和语义忠实度。

**[Riemannian Consistency Model](image_generation/riemannian_consistency_model.md)**

:   首次将一致性模型（Consistency Model）扩展到黎曼流形上，利用指数映射参数化和协变导数推导出离散和连续时间 RCM 目标函数，实现在球面、平坦环面和 SO(3) 等非欧几何上的高质量少步生成。

**[Rlvr-World Training World Models With Reinforcement Learning](image_generation/rlvr-world_training_world_models_with_reinforcement_learning.md)**

:   提出 RLVR-World 框架，将强化学习可验证奖励（RLVR）范式拓展到世界模型训练，通过将目标度量（如预测准确率、感知质量）作为可验证奖励直接优化，在语言和视频两类世界模型上取得显著提升。

**[Rlzero Direct Policy Inference From Language Without In-Domain Supervision](image_generation/rlzero_direct_policy_inference_from_language_without_in-domain_supervision.md)**

:   提出 RLZero 框架，通过"想象 → 投影 → 模仿"三步流程，将自然语言指令转化为目标环境中的行为策略——用视频生成模型从语言"想象"观测序列，将其投影到目标域，最终由无监督预训练的 RL 智能体通过闭合形式解即时模仿，整个过程无需任何域内监督或标注轨迹。

**[Robustness In Both Domains Clip Needs A Robust Text Encoder](image_generation/robustness_in_both_domains_clip_needs_a_robust_text_encoder.md)**

:   提出 LEAF (Levenshtein Efficient Adversarial Finetuning)，首个针对 CLIP 文本编码器的对抗微调方法，在字符级文本扰动下显著提升零样本分类、文本-图像检索和图像生成的鲁棒性，同时保持图像域性能。

**[Safe And Stable Control Via Lyapunov-Guided Diffusion Models](image_generation/safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)**

:   提出 S²Diff，一个基于模型的扩散规划框架，利用控制 Lyapunov 屏障函数（CLBF）引导扩散采样生成轨迹级控制策略，无需控制仿射假设与二次规划，在多种非线性动力系统上同时保证安全性和稳定性，平均安全率达 98.75%。

**[Sao-Instruct Free-Form Audio Editing Using Natural Language Instructions](image_generation/sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)**

:   提出SAO-Instruct，首个支持完全自由格式自然语言指令的音频编辑模型，通过Prompt-to-Prompt、DDPM反演和手动编辑三条流水线构建编辑三元组训练数据，微调Stable Audio Open实现保持上下文一致的定向音频修改。

**[Scalable Explainable And Provably Robust Anomaly Detection With One-Step Flow Ma](image_generation/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)**

:   提出 TCCM（Time-Conditioned Contraction Matching），一种受 flow matching 启发的表格数据半监督异常检测方法，通过学习将正常数据收缩到原点的时间条件速度场，仅需单步前向推理即可计算异常分数，在 ADBench 47 个数据集上取得 AUROC 和 AUPRC 双第一，推理速度比 DTE 快 1573 倍。

**[Scalediff Higher-Resolution Image Synthesis Via Efficient And Model-Agnostic Dif](image_generation/scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)**

:   提出 ScaleDiff 框架，通过 Neighborhood Patch Attention (NPA) 消除传统 patch 方法中的重叠计算冗余，结合潜空间频率混合 (LFM) 和结构引导 (SG)，在无需额外训练的前提下将预训练扩散模型扩展到高分辨率（如 4096²），在 U-Net 和 DiT 架构上均实现了 training-free 方法中的 SOTA 质量和显著的推理加速（相比 DemoFusion 快 8.9 倍）。

**[Scaling Can Lead To Compositional Generalization](image_generation/scaling_can_lead_to_compositional_generalization.md)**

:   通过理论证明和大规模实验表明，标准MLP仅需扩大数据量和模型规模即可实现组合泛化，无需显式模块化架构设计，且组合泛化成功时任务成分可从隐层激活线性解码——该指标与扩散模型的图像组合成功率正相关。

**[Scaling Diffusion Transformers Efficiently Via Μp](image_generation/scaling_diffusion_transformers_efficiently_via_μp.md)**

:   将 Maximal Update Parametrization (μP) 从标准 Transformer 推广到扩散 Transformer（DiT、PixArt-α、MMDiT 等），证明其超参数可从小模型稳定迁移到大模型，显著降低大规模扩散模型的调参成本。

**[Scaling Offline Rl Via Efficient And Expressive Shortcut Models](image_generation/scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)**

:   提出 SORL，利用 shortcut models 的自一致性实现离线 RL 中高效一阶段训练与可变推理步数的策略优化，同时支持推理时的顺序和并行扩展。

**[Scenedecorator Towards Scene-Oriented Story Generation With Scene Planning And S](image_generation/scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)**

:   SceneDecorator 提出了一个无需训练的框架，通过 VLM 引导的场景规划（global-to-local）和长期场景共享注意力机制，首次系统性地解决了故事生成中的场景规划和场景一致性问题，在场景对齐和一致性指标上显著优于现有方法。

**[Scenedesigner Controllable Multi-Object Image Generation With 9-Dof Pose Manipul](image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)**

:   SceneDesigner 提出了一种基于 CNOCS 地图表示和两阶段强化学习训练的方法，首次实现了多物体 9D 姿态（位置、大小、朝向）的精确控制，在图像生成的可控性和质量上显著超越现有方法。

**[Schrödinger Bridge Matching For Tree-Structured Costs And Entropic Wasserstein B](image_generation/schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)**

:   将Iterative Markovian Fitting (IMF)程序推广到树结构Schrödinger Bridge问题，提出TreeDSBM算法，在Wasserstein重心计算中将IMF迭代与不动点迭代优雅合并，仅需廉价的bridge-matching步骤即可高效求解。

**[Score-Informed Neural Operator For Enhancing Ordering-Based Causal Discovery](image_generation/score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)**

:   提出 SciNO（Score-informed Neural Operator），一种在光滑函数空间中设计的概率生成模型，稳定近似 log-密度 Hessian 对角以提升排序式因果发现，合成图上 order divergence 降低 42.7%，真实数据降低 31.5%。

**[Semantic Surgery Zero-Shot Concept Erasure In Diffusion Models](image_generation/semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)**

:   提出Semantic Surgery，一种无需重训练的零样本推理时概念擦除框架，通过在扩散过程之前对文本嵌入进行校准向量减法，结合Co-Occurrence Encoding处理多概念擦除和视觉反馈环路解决潜在概念持久性问题，在物体/NSFW/风格/名人擦除任务上全面超越SOTA。

**[Shallow Diffuse Robust And Invisible Watermarking Through Low-Dimensional Subspa](image_generation/shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)**

:   提出 Shallow Diffuse，一种利用扩散模型后验均值预测器（PMP）的局部线性性和 Jacobian 低秩性，在扩散过程中间时间步嵌入水印的方法，实现了水印与生成过程的解耦，首次在服务端和用户端两种场景下同时保证了高一致性和高鲁棒性。

**[Shortcutting Pre-Trained Flow Matching Diffusion Models Is Almost Free Lunch](image_generation/shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)**

:   提出SCFM（ShortCutting Flow Matching），一种超高效的后训练蒸馏方法，通过速度场自蒸馏将预训练flow matching模型（如12B参数的Flux）压缩为3步采样器，仅需不到1个A100-Day，无需步长嵌入或对抗蒸馏。

**[Show-O2 Improved Native Unified Multimodal Models](image_generation/show-o2_improved_native_unified_multimodal_models.md)**

:   提出 Show-o2，一种基于自回归建模和 Flow Matching 的原生统一多模态模型，通过双路径空间（时间）融合在 3D 因果 VAE 空间中构建统一视觉表示，实现跨文本、图像、视频的多模态理解与生成，并设计两阶段训练策略有效保留语言知识。

**[Sparsedit Token Sparsification For Efficient Diffusion Transformer](image_generation/sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)**

:   提出 SparseDiT，通过空间维度的三段式架构（底层 Poolingformer + 中层 Sparse-Dense Token Module + 顶层全密度处理）和时间维度的动态剪枝率策略，在 DiT-XL 512×512 上实现 55% FLOPs 减少和 175% 推理速度提升，FID 仅增加 0.09，并成功扩展到视频生成和文本到图像生成任务。

**[Split Gibbs Discrete Diffusion Posterior Sampling](image_generation/split_gibbs_discrete_diffusion_posterior_sampling.md)**

:   提出 SGDD（Split Gibbs Discrete Diffusion），一种基于分裂 Gibbs 采样原理的即插即用离散扩散后验采样算法，通过引入辅助变量和基于 Hamming 距离的正则化势函数，将后验采样分解为似然采样步和先验采样步交替进行，在 DNA 序列设计、离散图像逆问题和音乐填充等任务上大幅超越基线。

**[Splitflow Flow Decomposition For Inversion-Free Text-To-Image Editing](image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)**

:   提出 SplitFlow，将目标 prompt 语义分解为多个子 prompt，为每个子 prompt 计算独立的编辑流，再通过投影和自适应聚合机制组合成统一编辑轨迹，解决梯度纠缠问题，在无需反转的前提下实现更高保真度和可编辑性的文本引导图像编辑。

**[Stableguard Towards Unified Copyright Protection And Tamper Localization In Late](image_generation/stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)**

:   提出StableGuard，将全局二值水印嵌入LDM生成流程中（通过MPW-VAE），并利用水印扰动模式的变化实现篡改定位（通过MoE-GFN），首次实现端到端的版权保护与篡改检测统一框架。

**[State-Covering Trajectory Stitching For Diffusion Planners](image_generation/state-covering_trajectory_stitching_for_diffusion_planners.md)**

:   提出 SCoTS（State-Covering Trajectory Stitching），一种无需奖励信号的轨迹增强框架，通过在时间距离保持的潜空间中迭代拼接短轨迹片段，系统性地扩展状态空间覆盖，显著提升扩散规划器在长时域、分布外任务上的泛化能力。

**[Stella Subspace Learning In Low-Rank Adaptation Using Stiefel Manifold](image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)**

:   提出StelLA，通过将LoRA的适配矩阵分解为 $USV^\top$ 三因子形式，并将 $U$、$V$ 约束在Stiefel流形上进行黎曼优化，实现训练过程中对低秩子空间的显式学习，在多个下游任务上一致超越现有LoRA变体。

**[System-Embedded Diffusion Bridge Models](image_generation/system-embedded_diffusion_bridge_models.md)**

:   提出System-embedded Diffusion Bridge Models（SDB），将已知的线性测量系统直接嵌入矩阵值SDE的系数中，实现了对值域空间去噪和零空间信息合成的分离控制，在多种逆问题上取得一致性提升并展现出强大的系统失配鲁棒性。

**[T2Smark Balancing Robustness And Diversity In Noise-As-Watermark For Diffusion M](image_generation/t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)**

:   提出 T2SMark，一种基于尾部截断采样（Tail-Truncated Sampling）的两阶段扩散模型水印方案，通过在高斯噪声的尾部区域嵌入水印比特、中心区域随机采样，首次在水印鲁棒性和生成多样性之间取得最优平衡。

**[Text-To-Image Models Leave Identifiable Signatures Implications For Leaderboard ](image_generation/text-to-image_models_leave_identifiable_signatures_implications_for_leaderboard_.md)**

:   本文揭示文生图（T2I）模型因训练数据、架构和规模差异会在生成图像中留下可识别的"签名"，攻击者即使不控制输入提示也能通过 CLIP 嵌入空间中的简单质心分类以 87% 的 Top-1 准确率去匿名化排行榜上的匿名模型，从而实施排名操纵攻击。

**[Text To Sketch Generation With Multi-Styles](image_generation/text_to_sketch_generation_with_multi-styles.md)**

:   提出M3S（Multi-Style Sketch Synthesis），一个无训练框架，通过线性平滑的K/V特征注入、联合AdaIN风格倾向控制和风格-内容分离引导，实现基于文本提示和参考风格草图的单/多风格草图生成。

**[Thermalgen Style-Disentangled Flow-Based Generative Models For Rgb-To-Thermal Im](image_generation/thermalgen_style-disentangled_flow-based_generative_models_for_rgb-to-thermal_im.md)**

:   提出 ThermalGen，一种基于 Flow 的自适应生成模型，通过 RGB 图像条件化架构和风格解耦机制，首次实现了跨视角、跨传感器、跨环境条件的高保真 RGB-to-Thermal 图像翻译，并发布了三个新的大规模卫星-航拍 RGB-T 配对数据集。

**[Tidmad Time Series Dataset For Discovering Dark Matter With Ai Denoising](image_generation/tidmad_time_series_dataset_for_discovering_dark_matter_with_ai_denoising.md)**

:   发布 TIDMAD——首个面向暗物质搜索的超长时间序列去噪基准数据集，包含 ABRACADABRA 实验的训练/验证/科学数据、去噪评分指标和完整分析框架，使 AI 算法能直接产出物理学界标准的暗物质搜索结果。

**[Token Perturbation Guidance For Diffusion Models](image_generation/token_perturbation_guidance_for_diffusion_models.md)**

:   提出 Token Perturbation Guidance（TPG），通过对扩散模型中间 token 表示进行保范数的 shuffling 扰动来构造负分数信号，实现无需训练的条件无关引导，在无条件生成中将 SDXL 的 FID 提升近 2 倍，在条件生成中接近 CFG 效果。

**[Tortoise And Hare Guidance Accelerating Diffusion Model Inference With Multirate](image_generation/tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)**

:   提出 Tortoise and Hare Guidance (THG)，一种免训练的扩散采样加速策略，将 classifier-free guidance (CFG) ODE 重构为多速率 ODE 系统，噪声估计使用细粒度步长（乌龟方程），附加引导项使用粗粒度步长（兔子方程），减少最多 30% 的函数评估次数 (NFE) 而几乎不损失生成质量。

**[Toward A Unified Geometry Understanding Riemannian Diffusion Framework For Graph](image_generation/toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)**

:   提出 GeoMancer 框架，通过黎曼 GyroKernel 自编码器替代数值不稳定的指数映射，将多层级图特征解耦到任务特定的积流形上，并引入流形约束扩散和自引导生成策略，在分子生成、节点分类和图回归等任务上统一建模并取得 SOTA 性能。

**[Towards A Golden Classifier-Free Guidance Path Via Foresight Fixed Point Iterati](image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)**

:   将条件引导统一为不动点迭代框架，发现CFG及其变体都是短区间单步迭代的特例，理论证明其次优性，进而提出前瞻引导(FSG)——在早期扩散阶段对更长区间执行多步迭代，以更少计算实现更好的对齐质量。

**[Towards General Modality Translation With Contrastive And Predictive Latent Diff](image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)**

:   提出 LDDBM（Latent Denoising Diffusion Bridge Model），将去噪扩散桥模型扩展到共享潜空间中，结合对比对齐损失和预测损失，实现任意模态之间的通用翻译框架。

**[Towards Resilient Safety-Driven Unlearning For Diffusion Models Against Downstre](image_generation/towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)**

:   提出ResAlign框架，通过Moreau包络近似和元学习策略，让扩散模型的安全卸载（unlearning）能抵抗下游微调带来的有害行为恢复，即使在纯良性数据上微调也能保持安全性。

**[Towards Robust Zero-Shot Reinforcement Learning](image_generation/towards_robust_zero-shot_reinforcement_learning.md)**

:   提出BREEZE框架，通过行为正则化、任务条件扩散策略和注意力增强表示建模，系统性解决FB-based零样本RL中的OOD外推误差和表达力不足问题，在ExORL和D4RL Kitchen上实现最优或接近最优的鲁棒零样本泛化性能。

**[Track Inpaint Resplat Subject-Driven 3D And 4D Generation With Progressive Textu](image_generation/track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)**

:   提出TIRE（Track, Inpaint, REsplat）三阶段管线，通过视频跟踪定位未观测区域、主体驱动修复模型渐进式填充纹理、多视图一致性反投影回3D，实现身份保持的3D/4D生成。

**[Training-Free Constrained Generation With Stable Diffusion Models](image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)**

:   提出一种无需重新训练的约束生成方法，通过在 Stable Diffusion 的反向去噪过程中嵌入近端 Langevin 动力学（Proximal Langevin Dynamics），将图像空间中的约束通过解码器反向传播到潜空间，实现对生成输出的严格约束满足。

**[Training-Free Safe Text Embedding Guidance For Text-To-Image Diffusion Models](image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)**

:   提出 Safe Text embedding Guidance (STG)，一种无需训练的安全文本到图像生成方法，通过在扩散采样过程中基于安全函数对预期去噪图像的评估来动态调整文本嵌入方向，在有效去除不安全内容的同时最大程度保留原始语义意图。

**[Transferable Black-Box One-Shot Forging Of Watermarks Via Image Preference Model](image_generation/transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)**

:   本文提出一种基于图像偏好模型的黑盒水印伪造方法，仅需单张水印图像即可通过反向传播从中提取水印并粘贴到任意新图像上，在不访问水印算法的条件下有效伪造多种后处理水印方案。

**[Tree-Guided Diffusion Planner](image_generation/tree-guided_diffusion_planner.md)**

:   提出Tree-guided Diffusion Planner (TDP)，将测试时扩散规划形式化为树搜索问题，通过双层采样（粒子引导生成多样父轨迹 + 快速条件去噪生成子轨迹）在探索与利用之间取得平衡，在非凸目标和不可微约束下显著超越现有方法。

**[Two-Steps Diffusion Policy For Robotic Manipulation Via Genetic Denoising](image_generation/two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)**

:   通过揭示扩散策略中裁剪操作导致的分布失配本质，提出结合去噪调度优化与遗传算法群体选择的GDP方法，使现成DDPM扩散策略**无需重训练**即可在仅2步推理下达到甚至超越100步基线的操控性能。

**[Ultrahr-100K Enhancing Uhr Image Synthesis With A Large-Scale High-Quality Datas](image_generation/ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)**

:   构建了包含 10 万张超高分辨率图像及丰富标注的 UltraHR-100K 数据集，并提出频率感知后训练方法（DOTS + SWFR），通过面向细节的时间步采样和基于 DFT 的软加权频率正则化来增强预训练 T2I 模型的超高分辨率细节生成能力。

**[Understand Before You Generate Self-Guided Training For Autoregressive Image Gen](image_generation/understand_before_you_generate_self-guided_training_for_autoregressive_image_gen.md)**

:   通过系统分析自回归图像生成中阻碍视觉语义学习的三个关键属性（局部条件依赖、步间语义不一致、空间不变性缺失），提出 ST-AR 训练框架，将掩码图像建模和对比学习融入 next-token prediction 训练，在不依赖预训练表示模型的情况下，使 LlamaGen-XL 的 FID 提升约 49%（从 19.42 降到 9.81），50 epoch 即接近 3B 参数模型 300 epoch 的效果。

**[Understanding Representation Dynamics Of Diffusion Models Via Low-Dimensional Mo](image_generation/understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)**

:   在低秩高斯混合（MoLRG）数据模型下，理论证明了扩散模型表示质量随噪声水平呈单峰动态的现象源于去噪强度与类别区分度的权衡，并实证发现单峰动态的出现可作为模型泛化能力的可靠指标。

**[Unilumos Fast And Unified Image And Video Relighting With Physics-Plausible Feed](image_generation/unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)**

:   提出UniLumos，一个统一的图像和视频重光照框架，通过在flow matching骨干中引入RGB空间的深度和法线几何反馈来增强物理合理性，同时借助路径一致性学习实现20倍加速。

**[Unleashing Diffusion Transformers For Visual Correspondence By Modulating Massiv](image_generation/unleashing_diffusion_transformers_for_visual_correspondence_by_modulating_massiv.md)**

:   发现 Diffusion Transformers (DiTs) 中存在 massive activations 现象导致特征不可区分，揭示其与 AdaLN 的内在联系，提出无需训练的 DiTF 框架来提取语义判别性特征，在视觉对应任务上超越 DINO 和 SD 模型。

**[Utilgen Utility-Centric Generative Data Augmentation With Dual-Level Task Adapta](image_generation/utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)**

:   提出以任务效用为中心的生成式数据增强框架 UtilGen，通过元学习权重网络评估合成数据的下游任务效用，并利用模型级 DPO 和实例级（prompt+noise）双层优化策略，自适应生成高效用的合成训练数据，在8个基准上平均提升3.87%。

**[V-Cece Visual Counterfactual Explanations Via Conceptual Edits](image_generation/v-cece_visual_counterfactual_explanations_via_conceptual_edits.md)**

:   V-CECE提出首个系统性揭示人类与神经网络分类器语义理解差异（explanatory gap）的黑盒视觉反事实解释框架，通过WordNet知识图谱+匈牙利算法保证编辑集最优性，用Stable Diffusion执行概念级编辑，核心发现是CNN分类器的语义推理与人类严重不对齐（需5+步编辑），而LVLM（Claude 3.5 Sonnet）与人类高度一致（仅需2-3步）。

**[Value Gradient Guidance For Flow Matching Alignment](image_generation/value_gradient_guidance_for_flow_matching_alignment.md)**

:   提出VGG-Flow方法，利用最优控制理论中的Hamilton-Jacobi-Bellman方程，将流匹配模型对齐问题转化为"残差速度场匹配值函数梯度"的梯度匹配任务，实现高效且保持先验分布的奖励对齐。

**[Vicinity-Guided Discriminative Latent Diffusion For Privacy-Preserving Domain Ad](image_generation/vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)**

:   提出 Discriminative Vicinity Diffusion (DVD)，首次将潜扩散模型用于判别式知识迁移，通过在源域特征的近邻潜空间中训练扩散模型生成源样式线索，实现无需源数据访问的域适应，在标准 SFDA 基准上超越 SOTA。

**[Watermarking Autoregressive Image Generation](image_generation/watermarking_autoregressive_image_generation.md)**

:   首次将 LLM 水印技术（KGW green/red scheme）适配到自回归图像生成模型的 token 层，识别并解决了关键挑战——反向循环一致性（RCC）不足，通过 tokenizer-detokenizer 微调和水印同步层实现了鲁棒的、具有理论保证的图像水印检测。

**[What We Dont C Manifold Disentanglement For Structured Discovery](image_generation/what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)**

:   提出 WWDC（What We Don't C）方法，利用条件引导的潜在流匹配从已有 VAE 表征中去除已知信息，使未知特征在残余流形中更易被发现和访问，实现迭代式科学发现。

**[When Are Concepts Erased From Diffusion Models](image_generation/when_are_concepts_erased_from_diffusion_models.md)**

:   本文提出了两种概念擦除的机制模型（引导式回避 vs. 破坏式移除），并设计了涵盖优化搜索、上下文探测、噪声轨迹探测、分类器引导和动态追踪的五种独立探测方法，系统性地揭示了现有擦除方法大多只是"绕开"概念而非真正"消除"知识。

**[Where And How To Perturb On The Design Of Perturbation Guidance In Diffusion And](image_generation/where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)**

:   提出 HeadHunter 框架和 SoftPAG 方法，将扩散模型中的注意力扰动粒度从层级细化到单个注意力头级别，首次发现不同注意力头控制不同视觉概念（结构、风格、纹理等），实现了更精准且可组合的生成引导。

**[Why Diffusion Models Dont Memorize The Role Of Implicit Dynamical Regularization](image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)**

:   通过数值实验和理论分析揭示扩散模型训练中存在两个关键时间尺度——泛化时间 $\tau_{\text{gen}}$ 和记忆化时间 $\tau_{\text{mem}}$，后者随训练集大小 $n$ 线性增长而前者保持恒定，由此产生的隐式动力学正则化使模型即使在高度过参数化情况下也能通过早停避免记忆化。

**[Why Knowledge Distillation Works In Generative Models A Minimal Working Explanat](image_generation/why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)**

:   通过高斯混合模型的理论分析和大规模语言模型实验（SmolLM2 系列多级蒸馏），揭示知识蒸馏在生成模型中的核心机制——蒸馏诱导学生模型在精度（precision，生成质量）和召回（recall，分布覆盖度）之间进行权衡，由教师分布的熵控制。

**[Wmcopier Forging Invisible Image Watermarks On Arbitrary Images](image_generation/wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)**

:   提出 WMCopier，首个基于扩散模型的 no-box 水印伪造攻击方法，无需任何目标水印算法的先验知识，通过训练无条件扩散模型学习水印分布、浅层反演注入水印信号、迭代精炼优化质量，在开源和商业水印系统（包括 Amazon）上实现高成功率伪造。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Generalized Bisimulation Metric Of State Similarity Betwee](reinforcement_learning/a_generalized_bisimulation_metric_of_state_similarity_betwee.md)**

:   将传统只能在单个MDP内度量状态相似性的bisimulation metric (BSM)推广到跨MDP场景，提出广义双模拟度量(GBSM)，严格证明了对称性、跨MDP三角不等式和同状态距离上界三个基本度量性质，并在策略迁移、状态聚合和基于采样的估计三个应用中推导出比标准BSM更紧的误差界和闭式样本复杂度。

**[A Nearoptimal Scalable And Parallelizable Framework For Stoc](reinforcement_learning/a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)**

:   提出 BARBAT 框架，改进了经典的 BARBAR 算法，通过固定 epoch 长度和逐 epoch 调整失败概率，将对抗腐蚀下随机多臂老虎机的 regret 从 $O(\sqrt{K}C)$ 降至近最优的 $O(C)$（消除了 $\sqrt{K}$ 因子），并成功扩展到多智能体、图老虎机、组合半老虎机和批量老虎机等多种场景。

**[A Theory Of Multi-Agent Generative Flow Networks](reinforcement_learning/a_theory_of_multi-agent_generative_flow_networks.md)**

:   提出多智能体生成流网络（MA-GFlowNets）的理论框架，证明了"局部-全局原理"——联合流函数可分解为各智能体独立流的乘积形式，设计了四种算法（CFN/IFN/JFN/CJFN），其中 JFN 和 CJFN 实现中心化训练+去中心化执行（CTDE），在 Hyper-Grid 和 StarCraft 环境中超越 RL 和 MCMC 方法。

**[A Unifying View Of Linear Function Approximation In Offpolic](reinforcement_learning/a_unifying_view_of_linear_function_approximation_in_offpolic.md)**

:   首次引入矩阵分裂理论，将线性函数逼近下的TD、FQI和PFQI统一为求解同一目标线性系统 $(\Sigma_{cov} - \gamma\Sigma_{cr})\theta = \theta_{\phi,r}$ 的迭代方法（仅预条件子不同），给出各算法收敛的充要条件，提出rank invariance新概念，并揭示target network的本质是预条件子从常数到数据自适应的连续变换。

**[Act To See See To Act Diffusion-Driven Perception-Action Interplay For Adaptive ](reinforcement_learning/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)**

:   提出 DP-AG（Action-Guided Diffusion Policy），通过将扩散策略的噪声预测的 Vector-Jacobian Product (VJP) 作为结构化随机力驱动隐观测特征在扩散步骤间动态演化，并用循环一致对比损失闭合感知-动作环路，在 Push-T 上提升 6%、Dynamic Push-T 上提升 13%、真实 UR5 机器人上成功率提升 23%+。

**[Actorfree Continuous Control Via Structurally Maximizable Qf](reinforcement_learning/actorfree_continuous_control_via_structurally_maximizable_qf.md)**

:   提出 Q3C（Q-learning for Continuous Control with Control-points），通过学习一组控制点来逼近 Q 函数并保证最大值恰好在控制点上取到，配合动作条件化 Q 值生成、控制点多样性损失和尺度归一化等关键改进，在标准基准上匹配 TD3，在受限动作空间中显著超越所有 actor-critic 方法。

**[Adaptive Cooperative Transmission Design For Ultra-Reliable Low-Latency Communic](reinforcement_learning/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)**

:   提出 DRL-CoLA 算法，用双 Agent DQN 分别在源节点和中继节点上自适应配置 5G NR 传输参数（numerology、mini-slot、MCS），在两跳中继系统中仅用本地 CSI 即可达到接近全局 CSI 最优的 URLLC 可靠性。

**[Adaptive Neighborhoodconstrained Q Learning For Offline Rein](reinforcement_learning/adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)**

:   提出 ANQ（Adaptive Neighborhood-constrained Q learning），在离线 RL 中引入基于优势函数的自适应邻域约束，在密度约束（过于保守）和支持约束（需精确建模行为策略）之间找到灵活的中间方案，通过双层优化框架实现高效 Q 学习，在 D4RL 基准上达到 SOTA。

**[Adaptively Coordinating With Novel Partners Via Learned Latent Strategies](reinforcement_learning/adaptively_coordinating_with_novel_partners_via_learned_latent_strategies.md)**

:   提出 TALENTS 框架，通过 VAE 学习潜在策略空间 + K-Means 聚类发现策略类型 + Fixed-Share 遗憾最小化算法在线推断队友类型，实现对未知人类/智能体队友的零样本实时适应协作。

**[Aline Joint Amortization For Bayesian Inference And Active Data Acquisition](reinforcement_learning/aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)**

:   ALINE 提出统一的分摊贝叶斯推断和主动数据获取框架，用 Transformer 架构 + RL 训练，使模型能同时策略性地选择最有信息量的数据点并即时完成后验推断，还支持灵活地针对特定参数子集或预测目标进行数据获取。

**[Approximating Shapley Explanations In Reinforcement Learning](reinforcement_learning/approximating_shapley_explanations_in_reinforcement_learning.md)**

:   提出 FastSVERL，一种可扩展的参数化学习框架，分别近似强化学习中 Shapley 值的两个计算瓶颈（特征函数和 Shapley 求和），支持离策略数据学习和随策略演化持续更新解释。

**[Automaton Constrained Q-Learning](reinforcement_learning/automaton_constrained_q-learning.md)**

:   提出 ACQL（Automaton Constrained Q-Learning），将线性时序逻辑（LTL）任务规范转化为自动机，结合目标条件学习和最小安全约束，首次在连续控制环境中可扩展地同时支持时序目标序列和非平稳安全约束。

**[Bandit And Delayed Feedback In Online Structured Prediction](reinforcement_learning/bandit_and_delayed_feedback_in_online_structured_prediction.md)**

:   首次研究在线结构化预测中赌臂反馈和延迟反馈场景，通过设计新的伪逆矩阵梯度估计器，实现了不显式依赖输出集大小 $K$ 的 $O(T^{2/3})$ 替代遗憾上界。

**[Beast Efficient Tokenization Of B-Splines Encoded Action Sequences For Imitation](reinforcement_learning/beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)**

:   BEAST 用 B 样条曲线参数化动作序列——通过岭回归估计控制点并均匀量化为固定长度 token，实现 20× token 压缩（100 步→5 token）、数学保证的动作块间 $C^0$ 连续过渡，在 LIBERO-Long 上成功率排名第 1（86.4%），推理吞吐量 617 Hz（比 π₀ 快 2.14×、比 OpenVLA 快 101×）。

**[Behavior Injection Preparing Language Models For Reinforcement Learning](reinforcement_learning/behavior_injection_preparing_language_models_for_reinforcement_learning.md)**

:   揭示 LLM 对 RL 微调响应不一致的根本原因——通过 per-step influence 分析发现 RL 效果取决于（1）rollout 准确率分布（中等最优）和（2）数据 co-influence 强度，提出 BRIDGE 在 SFT 阶段注入探索/利用行为，使后续 RL 增益从 6% 提升到 46.6%。

**[Beyond The 8020 Rule Highentropy Minority Tokens Drive Effec](reinforcement_learning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)**

:   从 token 熵模式的全新视角分析 RLVR，发现 CoT 推理中仅约 20% 的高熵"分叉 token"决定推理方向，仅在这些 token 上做梯度更新即可匹配甚至大幅超越全量更新（Qwen3-32B 上 AIME'25 +11.04），揭示 RLVR 本质是优化推理决策点。

**[Blending Complementary Memory Systems In Hybrid Quadratic-Linear Transformers](reinforcement_learning/blending_complementary_memory_systems_in_hybrid_quadratic-linear_transformers.md)**

:   提出混合二次-线性 Transformer（HQLT），将 KV-memory（softmax attention，精确检索但二次复杂度）与 FW-memory（DeltaNet/线性 attention，线性复杂度但检索粗糙）融合为互补记忆系统，比较三种混合策略（延迟流式/延迟分块/同步），在 340M 和 1.3B 参数规模的语言建模、检索、算法推理和 RL 任务上验证同步混合最优。

**[Bootstrap Off-Policy With World Model](reinforcement_learning/bootstrap_off-policy_with_world_model.md)**

:   提出 BOOM 框架，通过 bootstrap 循环将在线规划器（MPPI）与 off-policy 策略学习紧密结合：策略初始化规划器，规划器反过来通过无似然对齐损失（likelihood-free alignment）引导策略改进，配合 soft Q-weighted 机制优先学习高回报行为，在高维连续控制任务上取得 SOTA。

**[Boundary-To-Region Supervision For Offline Safe Reinforcement Learning](reinforcement_learning/boundary-to-region_supervision_for_offline_safe_reinforcement_learning.md)**

:   提出 BOOM 框架，通过 bootstrap 对齐回路将在线规划器的高质量动作蒸馏到策略网络，使用 likelihood-free 的前向 KL 散度和软 Q 加权机制，有效缓解规划器与策略之间的 actor divergence 问题，在高维连续控制任务上取得 SOTA。

**[Certifying Concavity And Monotonicity In Games Via Sum-Of-Squares Hierarchies](reinforcement_learning/certifying_concavity_and_monotonicity_in_games_via_sum-of-squares_hierarchies.md)**

:   证明了在多项式效用和半代数策略集的博弈中验证凹性和单调性是 NP-hard 的，并提出了两套基于平方和 (SOS) 规划的层次化认证方案，可在多项式时间内逐层求解。

**[Certifying Stability Of Reinforcement Learning Policies Using Generalized Lyapun](reinforcement_learning/certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)**

:   提出 Generalized Lyapunov Function 方法，通过将 RL 值函数与神经网络残差项结合，并用多步加权下降条件替代经典的逐步严格下降要求，实现对 RL 策略的稳定性认证。

**[Checklists Are Better Than Reward Models For Aligning Langua](reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)**

:   提出 Reinforcement Learning from Checklist Feedback (RLCF)，将指令分解为动态生成的 yes/no checklist，结合 AI judge 和代码验证器逐项评分后做 DPO 训练，在 5 个 benchmark 上一致性提升 Qwen2.5-7B-Instruct，是唯一在所有 benchmark 上都有正收益的方法（FollowBench +4pt, InFoBench +6pt, Arena-Hard +3pt）。

**[Communicating Plans Not Percepts Scalable Multi-Agent Coordination With Embodied](reinforcement_learning/communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)**

:   提出基于轻量世界模型的"意图通信"架构，通过生成并共享未来轨迹计划来实现多智能体协调，在可扩展性和性能上全面超越端到端涌现通信方案。

**[Comparing Uniform Price And Discriminatory Multi-Unit Auctions Through Regret Mi](reinforcement_learning/comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)**

:   从在线学习和遗憾最小化框架出发，系统比较统一价格拍卖与歧视性拍卖的学习难度，证明两种格式在最坏情况下遗憾率相同，但特定结构条件下统一价格拍卖允许更快的学习速率。

**[Complexity Scaling Laws For Neural Models Using Combinatorial Optimization](reinforcement_learning/complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)**

:   以旅行商问题（TSP）为案例，研究固定模型容量下问题复杂度（解空间大小、表示空间维度）与模型性能之间的可预测缩放规律，揭示了 RL 和 SFT 在组合优化中的系统性性能趋势。

**[Computational Hardness Of Reinforcement Learning With Partial Qπ-Realizability](reinforcement_learning/computational_hardness_of_reinforcement_learning_with_partial_qπ-realizability.md)**

:   引入"部分 $q^\pi$-可实现性"概念，证明在此设定下使用贪心策略集时学习近优策略是 NP-hard 的，使用 softmax 策略集时在 rETH 假设下需要指数时间，弥合了 $q^*$-可实现性和 $q^\pi$-可实现性之间的理论空白。

**[Confounding Robust Deep Reinforcement Learning A Causal Approach](reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)**

:   基于部分辨识（partial identification）理论扩展 DQN，提出 Causal DQN 从含有未观测混淆因子的离线数据中学习鲁棒策略——通过优化最坏情况下的价值函数下界来获得安全策略，在 12 个混淆 Atari 游戏中一致性地超越标准 DQN。

**[Continual Knowledge Adaptation For Reinforcement Learning](reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)**

:   提出 CKA-RL，为每个任务维护知识向量（task-specific knowledge vector），通过 softmax 加权的动态知识适配和自适应知识合并机制，在三个持续 RL 基准上实现 4.20% 的整体性能提升和 8.02% 的前向迁移提升。

**[Convergence Theorems For Entropy-Regularized And Distributional Reinforcement Le](reinforcement_learning/convergence_theorems_for_entropy-regularized_and_distributional_reinforcement_le.md)**

:   提出 **温度解耦策略（temperature decoupling gambit）**，证明在熵正则化强化学习中，通过解耦评估温度和行为温度，可以在温度趋于零时保证策略和回报分布收敛到一个可解释的、保持多样性的最优策略。

**[Core Constraint-Aware One-Step Reinforcement Learning For Simulation-Guided Neur](reinforcement_learning/core_constraint-aware_one-step_reinforcement_learning_for_simulation-guided_neur.md)**

:   提出 CORE（Constraint-aware One-step REinforcement learning），一种无 critic 的单步 RL 框架，通过结构化分布采样、scaling-graph 解码器和约束感知的 reward shaping 来高效探索 DNN 加速器的硬件-映射联合设计空间，在 7 个 DNN 模型上取得至少 15× 的 latency 改善。

**[Dccluster-Opt Benchmarking Dynamic Multi-Objective Optimization For Geo-Distribu](reinforcement_learning/dccluster-opt_benchmarking_dynamic_multi-objective_optimization_for_geo-distribu.md)**

:   提出 DCcluster-Opt，一个面向地理分布式数据中心的开源高保真仿真基准平台，融合真实世界数据集（碳强度、电价、天气等）和物理模型，支持动态多目标负载调度的强化学习研究。

**[Decoderhybriddecoder Architecture For Efficient Reasoning Wi](reinforcement_learning/decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)**

:   SambaY 提出 Gated Memory Unit（GMU）用于跨层共享 SSM 的 token 混合表示，将 YOCO 的 cross-decoder 中一半的 cross-attention 层替换为轻量级 GMU，在保持线性预填充复杂度和长上下文检索能力的同时，大幅提升解码效率——最终产品 Phi4-mini-Flash-Reasoning (3.8B) 在推理任务上超越 Phi4-mini-Reasoning，且在 2K 提示 + 32K 生成场景下实现高达 10× 的解码吞吐提升。

**[Deep Rl Needs Deep Behavior Analysis Exploring Implicit Planning By Model-Free A](reinforcement_learning/deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)**

:   提出 ForageWorld 自然觅食环境和神经科学启发的行为分析框架，揭示无模型 RNN-based DRL 智能体通过涌现动力学展现出结构化的类规划行为——无需显式记忆模块或世界模型。

**[Deepdiver Adaptive Search Intensity Scaling Via Open-Web Reinforcement Learning](reinforcement_learning/deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)**

:   提出 DeepDiver，一个 RL 驱动的搜索推理框架，在真实开放网络环境中训练 LLM 的信息寻求能力，催生"搜索强度缩放"（SIS）涌现行为——7B 模型在知识密集任务上可媲美 671B 的 DeepSeek-R1。

**[Discover Automated Curricula For Sparse-Reward Reinforcement Learning](reinforcement_learning/discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)**

:   提出 DISCOVER，一种面向稀疏奖励长视野 RL 的目标选择策略，通过同时平衡可达性（achievability）、新颖性（novelty）和相关性（relevance）来生成指向目标任务的课程，理论上证明达到目标的步数与目标距离线性相关（而非搜索空间体积），在高维导航和操作任务中显著超越先前 SOTA 探索策略。

**[Dynamic Regret Reduces To Kernelized Static Regret](reinforcement_learning/dynamic_regret_reduces_to_kernelized_static_regret.md)**

:   将动态遗憾最小化问题重新建模为再生核希尔伯特空间(RKHS)中的静态遗憾问题，通过精心设计平移不变核实现最优路径长度依赖 $\widetilde{\mathcal{O}}(\sqrt{MP_TT})$，且天然不需要时间范围先验知识。

**[Dynamics-Aligned Latent Imagination In Contextual World Models For Zero-Shot Gen](reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)**

:   在 DreamerV3 架构中引入自监督上下文编码器 DALI，从交互历史中推断潜在环境参数（如重力、摩擦力），在 cMDP 基准上无需重训练即可实现零样本泛化，在外推任务上比 ground-truth context-aware 基线高出最多 96.4%。

**[Egobridge Domain Adaptation For Generalizable Imitation From Egocentric Human Da](reinforcement_learning/egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)**

:   提出 EgoBridge 框架，利用最优传输(OT)在策略潜在空间中对齐人类和机器人数据的联合分布（特征+动作），结合动态时间规整(DTW)构建伪配对，实现从第一人称人类数据到机器人的跨具身知识迁移，在真实世界任务中绝对成功率提升达 44%。

**[Empirical Study On Robustness And Resilience In Cooperative Multi-Agent Reinforc](reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)**

:   通过 82,620 次大规模实验系统性研究合作多智能体 RL 中的鲁棒性和弹性，揭示超参数调优比算法选择更重要，并发现参数共享、GAE、PopArt 等常见做法在不确定性下反而有害，提出一套实用的超参数建议。

**[Enhancing Interpretability In Deep Reinforcement Learning Through Semantic Clust](reinforcement_learning/enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)**

:   提出语义聚类模块(SCM)，将特征降维网络(FDR)与改进的 VQ-VAE 在线聚类相结合，无缝集成到 DRL 训练流程中，解决了 t-SNE 可视化不稳定的问题，揭示 DRL 内在具有基于语义的动态聚类特性。

**[Establishing Linear Surrogate Regret Bounds For Convex Smooth Losses Via Convolu](reinforcement_learning/establishing_linear_surrogate_regret_bounds_for_convex_smooth_losses_via_convolu.md)**

:   通过构造基于卷积负熵（convolutional negentropy）的 Fenchel–Young 损失，首次证明凸且光滑的代理损失可以同时拥有线性代理遗憾界，打破了此前社区认为光滑性与线性遗憾率不可兼得的固有认知。

**[Evolm In Search Of Lost Language Model Training Dynamics](reinforcement_learning/evolm_in_search_of_lost_language_model_training_dynamics.md)**

:   系统训练 100+ 个 1B/4B 参数的 LM（从零开始），透明地研究预训练→续训→SFT→RL 各阶段的训练动态，揭示过度训练的递减收益、灾难性遗忘的缓解策略、以及 SFT/RL 配置的复杂权衡。

**[Exploration Via Feature Perturbation In Contextual Bandits](reinforcement_learning/exploration_via_feature_perturbation_in_contextual_bandits.md)**

:   提出特征扰动（Feature Perturbation）作为上下文 bandit 的新型随机探索策略：直接在特征输入上注入噪声，而非扰动参数或奖励，从而在广义线性 bandit 中实现 $\tilde{O}(d\sqrt{T})$ 最优遗憾界，首次消除了随机化算法相较确定性方法的 $\sqrt{d}$ 因子劣势。

**[Exploration With Foundation Models Capabilities Limitations And Hybrid Approache](reinforcement_learning/exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)**

:   系统评测 LLM/VLM 在经典 RL 探索任务（bandit、Gridworld、Atari）上的零样本能力，发现 VLM 存在"知行差距"（knowing-doing gap）——高层推理正确但低层控制失败，并提出简单的 VLM-RL 混合框架在理想条件下可显著加速学习。

**[Extending Ngu To Multi-Agent Rl A Preliminary Study](reinforcement_learning/extending_ngu_to_multi-agent_rl_a_preliminary_study.md)**

:   将单智能体 NGU（Never Give Up）算法扩展至多智能体环境，通过共享回放缓冲区、共享新颖性信号和异构 β 参数三个设计维度的系统消融，发现 NGU + 共享经验池组合在 PettingZoo simple_tag 追捕任务中显著优于多智能体 DQN 基线。

**[Fedrain-Lite Federated Reinforcement Algorithms For Improving Idealised Numerica](reinforcement_learning/fedrain-lite_federated_reinforcement_algorithms_for_improving_idealised_numerica.md)**

:   提出 FedRAIN-Lite 联邦强化学习框架，将 RL 智能体分配到不同纬度带学习局部气候参数化策略并定期全局聚合，在层次化理想能量平衡模型上验证 DDPG 在热带和中纬度区域可将面积加权 RMSE 降低 50% 以上，为 RL 扩展到全尺度 GCM 提供了可行路径。

**[Feel-Good Thompson Sampling For Contextual Bandits A Markov Chain Monte Carlo Sh](reinforcement_learning/feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)**

:   首次系统性实证评估 Feel-Good Thompson Sampling (FG-TS) 及其平滑变体 SFG-TS 在近似后验下的表现，横跨线性/逻辑/神经三类上下文赌博机设置和十四个基准，发现 FG-TS 在精确后验场景（线性/逻辑）下优于标准 TS，但在神经赌博机中反而退化，揭示了乐观偏差与采样噪声之间的关键权衡。

**[Financial Instruction Following Evaluation Fife](reinforcement_learning/financial_instruction_following_evaluation_fife.md)**

:   FIFE 是一个面向金融分析任务的高难度指令遵循基准，包含 88 个人工编写的复杂提示和 40+ 种金融领域专用的可链式验证约束，通过严格/宽松两种模式评测 53 个模型，揭示出即使最强的开放权重模型（76.1% strict）也无法完美遵循金融领域的复杂指令要求。

**[Finite-Sample Analysis Of Policy Evaluation For Robust Average Reward Reinforcem](reinforcement_learning/finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)**

:   首次给出鲁棒平均奖励 MDP 策略评估的有限样本复杂度分析：通过构造精巧的半范数证明鲁棒 Bellman 算子具有收缩性质，结合截断 Multi-Level Monte Carlo 估计器实现有限期望样本复杂度，最终达到阶最优的 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度。

**[Forecasting In Offline Reinforcement Learning For Non-Stationary Environments](reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)**

:   提出 Forl 框架，将条件扩散模型生成的多模态候选状态与零样本时序基础模型的偏移预测通过维度最近匹配（DCM）融合，在测试时无需重训练即可应对观测函数随 episode 非平稳变化的离线 RL 部署场景，在 D4RL 标准基准上平均提升数十分。

**[Foundation Models As World Models A Foundational Study In Text-Based Gridworlds](reinforcement_learning/foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)**

:   系统性评估了基础模型（LLM）作为零样本世界模型（FWM）和直接决策智能体（FA）在文本网格世界中的表现，揭示了两种策略在确定性/随机性环境中的互补优势。

**[Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret In Noise-](reinforcement_learning/gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)**

:   本文证明 GP-UCB 在 noise-free GP bandit 问题中可达到 nearly-optimal regret，首次在 SE 核下实现 $O(1)$ 常数累积遗憾、在 Matérn 核（$d < \nu$）下实现 $O(1)$ 累积遗憾，弥合了 GP-UCB 理论与实践之间的长期差距。

**[Generalized Linear Bandits Almost Optimal Regret With One-Pass Update](reinforcement_learning/generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)**

:   提出GLB-OMD算法，首次在广义线性赌博机（GLB）问题中同时实现近似最优遗憾界 $\mathcal{O}(\log T\sqrt{T/\kappa_*})$ 和每轮 $\mathcal{O}(1)$ 的时间/空间复杂度，核心技术是基于混合损失（mix loss）为在线镜像下降（OMD）估计量构建紧致置信集。

**[Generalizing Verifiable Instruction Following](reinforcement_learning/generalizing_verifiable_instruction_following.md)**

:   引入IFBench基准评估精确指令遵循的泛化能力，证明当前SOTA模型严重过拟合于IFEval的25种约束模板，并提出IF-RLVR训练方法（基于GRPO + 可验证奖励）显著提升域内外指令遵循性能。

**[Global Convergence For Average Reward Constrained Mdps With Primal-Dual Actor Cr](reinforcement_learning/global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)**

:   提出Primal-Dual Natural Actor-Critic（PDNAC）算法，首次在一般参数化策略下的平均奖励约束MDP中实现 $\tilde{\mathcal{O}}(1/\sqrt{T})$ 的全局收敛率和约束违反率，匹配理论下界。

**[Gradient-Variation Online Adaptivity For Accelerated Optimization With Hölder Sm](reinforcement_learning/gradient-variation_online_adaptivity_for_accelerated_optimization_with_hölder_sm.md)**

:   在 Hölder 光滑函数类上实现梯度变差自适应的在线学习算法，其 regret 在光滑和非光滑极端之间平滑插值；通过在线到批量转换，首次为强凸优化提供在光滑情形下加速、非光滑情形下近优的通用方法。

**[Greedy Algorithm For Structured Bandits A Sharp Characterization Of Asymptotic S](reinforcement_learning/greedy_algorithm_for_structured_bandits_a_sharp_characterization_of_asymptotic_s.md)**

:   本文对结构化 bandit 问题中的贪心算法（Greedy）进行了完整的理论刻画，提出 self-identifiability 作为贪心算法能否获得 sublinear regret 的充要条件，并将结论推广到上下文 bandit 及一般交互决策框架 DMSO。

**[Horizon Reduction Makes Rl Scalable](reinforcement_learning/horizon_reduction_makes_rl_scalable.md)**

:   本文通过大规模实验（最高 10 亿转移数据）揭示离线 RL 的可扩展性瓶颈源于决策时域过长（curse of horizon），并证明通过 n-step 回报和层次策略等时域缩减技术可显著提升扩展性，进而提出了简洁有效的 SHARSA 方法。

**[Human-Inspired Multi-Level Reinforcement Learning](reinforcement_learning/human-inspired_multi-level_reinforcement_learning.md)**

:   本文提出 RbRL-KL，在 rating-based RL 基础上增加 KL 散度驱动的策略损失项，利用不同评分等级的失败经验以不同权重推开当前策略，在 6 个 DeepMind Control 环境中超越标准 RbRL。

**[Hybrid Latent Reasoning Via Reinforcement Learning](reinforcement_learning/hybrid_latent_reasoning_via_reinforcement_learning.md)**

:   HRPO 提出混合潜在推理策略优化：通过可学习的门控机制将前一步的隐藏状态表示逐步融入到采样的 token embedding 中，使 LLM 在推理阶段同时利用离散 token 和连续潜在表示，无需 CoT 标注即可通过 RL 训练，在知识密集型和 STEM 推理任务上均超越 PPO/GRPO 等基线。

**[Improved Regret And Contextual Linear Extension For Pandoras Box And Prophet Ine](reinforcement_learning/improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)**

:   本文针对在线 Pandora's Box 问题提出新算法，将 regret 从 $\widetilde{O}(n\sqrt{T})$ 改进到 $\widetilde{O}(\sqrt{nT})$（匹配下界），并首次提出 contextual linear 扩展实现 $\widetilde{O}(nd\sqrt{T})$ regret。

**[Improved Regret Bounds For Gaussian Process Upper Confidence Bound In Bayesian O](reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)**

:   本文证明 GP-UCB 在贝叶斯设定下可达 $\widetilde{O}(\sqrt{T})$ 高概率 regret（Matern 核满足光滑条件时）和 $O(\sqrt{T \ln^2 T})$（SE 核），弥合了 GP-UCB 已有上界与最优上界间的差距。

**[Improving Planning And Mbrl With Temporally-Extended Actions](reinforcement_learning/improving_planning_and_mbrl_with_temporally-extended_actions.md)**

:   本文提出在 shooting-based 规划和 MBRL 中将动作持续时间作为额外优化变量，配合 MAB 自动选择持续时间范围，在多个环境中显著加速规划并解决标准方法无法解决的困难任务。

**[Improving Retrieval-Augmented Generation Through Multi-Agent Reinforcement Learn](reinforcement_learning/improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)**

:   将复杂 RAG 流水线中的多个组件（Query Rewriter、Selector、Generator）建模为协作多智能体系统，使用 MAPPO 算法进行联合优化，以最终答案的 F1 分数作为共享奖励，在多个 QA 基准上超越现有单模块优化方法。

**[Incremental Sequence Classification With Temporal Consistency](reinforcement_learning/incremental_sequence_classification_with_temporal_consistency.md)**

:   将强化学习中时序差分（TD）学习的思想引入序列分类任务，提出 TC-$\lambda$ 损失函数，通过要求相邻时间步的预测分布满足时序一致性条件来训练增量式序列分类器，在文本分类和 LLM 验证任务上均优于标准交叉熵方法。

**[Inner Speech As Behavior Guides Steerable Imitation Of Diverse Behaviors For Hum](reinforcement_learning/inner_speech_as_behavior_guides_steerable_imitation_of_diverse_behaviors_for_hum.md)**

:   受维果茨基内心语言理论启发，提出 MIMIC 框架，利用语言作为感知与动作之间的中介表征，通过 VLM 提供语言脚手架训练 CVAE 生成内心语言，再以扩散策略在条件化于内心语言的情况下生成多样且可控的行为。

**[Interactive And Hybrid Imitation Learning Provably Beating Behavior Cloning](reinforcement_learning/interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)**

:   当标注成本按**状态**而非轨迹计量时，证明交互式方法 Stagger 在 $\mu$-可恢复条件下可证明地超越 Behavior Cloning（次优性 $O(\mu H \log B / N)$ vs $O(RH \log B / CN)$，$\mu \ll R$ 时优势显著）；进一步提出混合 IL 算法 Warm-Stagger，结合离线数据和交互标注，在特定 MDP 上实现两种数据源的严格互补优势。

**[Inverse Optimization Latent Variable Models For Learning Costs Applied To Route ](reinforcement_learning/inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)**

:   提出 IO-LVM（Inverse Optimization Latent Variable Model），用 VAE 式编码器映射观测的 COP 解到潜在成本空间，通过 Fenchel-Young 损失和黑盒求解器（Dijkstra/TSP solver）在解码端保证可行性，无需 agent 标签即可从路径数据中学到成本函数的分布，成功不可监督地分离不同 agent 的导航偏好。

**[Kimina Lean Server A High-Performance Lean Server For Large-Scale Verification](reinforcement_learning/kimina_lean_server_a_high-performance_lean_server_for_large-scale_verification.md)**

:   提出Kimina Lean Server——一个面向大规模强化学习训练的高性能Lean 4验证服务器，通过服务端并行化和LRU缓存机制实现1.5-2倍的速度提升，已用于训练SOTA定理证明模型Kimina-Prover。

**[Knowledge-Based Visual Question Answer With Multimodal Processing Retrieval And ](reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)**

:   提出 Wiki-PRF，一套三阶段（处理-检索-过滤）的多模态 RAG 框架，通过强化学习训练 VLM 自主调用视觉工具和过滤检索结果，在 E-VQA 和 InfoSeek 上达到 SOTA。

**[Last Iterate Convergence In Monotone Mean Field Games](reinforcement_learning/last_iterate_convergence_in_monotone_mean_field_games.md)**

:   在非严格单调平均场博弈(MFG)中，提出基于 KL 散度的近端点(PP)方法实现渐近最后迭代收敛(LIC)，并证明正则化镜像下降(RMD)以指数速率收敛到正则化均衡，两者结合的 APP 算法在标准基准上可靠收敛到非正则化均衡。

**[Learning From Demonstrations Via Capability-Aware Goal Sampling](reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)**

:   提出Cago方法，通过动态追踪智能体在专家演示轨迹上的达成能力，自适应采样处于能力边界的中间目标，构建隐式课程引导长视野稀疏奖励任务学习。

**[Learning Human-Like Rl Agents Through Trajectory Optimization With Action Quanti](reinforcement_learning/learning_human-like_rl_agents_through_trajectory_optimization_with_action_quanti.md)**

:   提出 MAQ（Motion-Action Quantization）方法，通过 VQ-VAE 将人类动作离散化为有限的原语集合，然后在量化动作空间中进行轨迹优化，训练出行为模式更接近人类的 RL agent。

**[Learning In Stackelberg Mean Field Games A Non-Asymptotic Analysis](reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)**

:   提出首个具有非渐近收敛保证的单循环Actor-Critic算法AC-SMFG，用于求解Stackelberg平均场博弈（SMFG），收敛速率达到 $\widetilde{\mathcal{O}}(k^{-1/2})$。

**[Learning Interactive World Model For Object-Centric Reinforcement Learning](reinforcement_learning/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)**

:   提出 FIOC-WM，通过对象级和属性级的两层分解学习世界模型中的物体交互结构，并基于交互原语训练层级策略，在多个机器人控制任务上实现了更高效的策略学习和组合泛化能力。

**[Learning Interestingness In Automated Mathematical Theory Formation](reinforcement_learning/learning_interestingness_in_automated_mathematical_theory_formation.md)**

:   提出 Fermat——一个将数学理论形成建模为 MDP 的强化学习环境，以及 EvoAbstract——一个带抽象学习的 LLM 驱动进化算法，用于自动合成数学对象的"兴趣度"度量函数，在初等数论和有限域上显著超越硬编码基线。

**[Learning Intractable Multimodal Policies With Reparameterization And Diversity R](reinforcement_learning/learning_intractable_multimodal_policies_with_reparameterization_and_diversity_r.md)**

:   提出Diversity-regularized Actor Critic（DrAC）算法，通过将不可解析的多模态策略（amortized actor和diffusion actor）统一为stochastic-mapping formulation，利用重参数化技巧直接进行策略梯度优化，并设计基于距离的多样性正则化替代传统熵正则化，在多目标导航和生成式RL等多样性关键任务中展现显著优势。

**[Learning Memory-Enhanced Improvement Heuristics For Flexible Job Shop Scheduling](reinforcement_learning/learning_memory-enhanced_improvement_heuristics_for_flexible_job_shop_scheduling.md)**

:   提出 MIStar——首个基于深度强化学习 (DRL) 的改进型启发式框架，用于求解柔性作业车间调度问题 (FJSP)。核心创新包括有向异构析取图表示、记忆增强异构图神经网络 (MHGNN) 和并行贪心搜索策略，在合成数据和公开 benchmark 上全面超越手工改进启发式和 SOTA 构造型 DRL 方法。

**[Learning To Clean Reinforcement Learning For Noisy Label Correction](reinforcement_learning/learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)**

:   将噪声标签纠正问题建模为强化学习中的马尔可夫决策过程，提出 RLNLC 框架，通过 k 近邻嵌入空间构建策略函数判断哪些标签需纠正，并设计标签一致性奖励和跨子集对齐奖励指导纠正过程，在多个基准数据集上的实例依赖和对称噪声场景中均达到最优性能。

**[Learning To Focus Prioritizing Informative Histories With Structured Attention M](reinforcement_learning/learning_to_focus_prioritizing_informative_histories_with_structured_attention_m.md)**

:   提出两种结构化时序先验（Memory-Length Prior和Gaussian Distributional Prior）嵌入Transformer世界模型的自注意力机制中，在部分可观测RL环境下，Gaussian Attention在Atari 100k基准上相对UniZero提升77%的人类归一化均分，且计算开销几乎为零。

**[Massively Parallel Imitation Learning Of Mouse Forelimb Musculoskeletal Reaching](reinforcement_learning/massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)**

:   基于 MIMIC-MJX 平台构建小鼠前肢肌肉骨骼模拟学习流水线，通过 JAX 加速的大规模并行 PPO（120 万步/秒）训练物理感知模仿学习策略，证明控制成本正则化能使模拟肌肉活动更好地预测真实 EMG 信号，并用基于 Takens 定理的非线性动力学方法从关节运动学预测肌肉激活。

**[Mean-Field Sampling For Cooperative Multi-Agent Reinforcement Learning](reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)**

:   提出 SUBSAMPLE-MFQ 算法，通过从 $n$ 个智能体中随机采样 $k$ 个进行均场 Q 学习，将多智能体强化学习的样本复杂度从 $\text{poly}(n)$ 降低到 $\text{poly}(k)$，且性能差距仅为 $\tilde{O}(1/\sqrt{k})$（与 $n$ 无关），当 $k = O(\log n)$ 时实现相对均场 MARL 的指数加速。

**[Memo Training Memory-Efficient Embodied Agents With Reinforcement Learning](reinforcement_learning/memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)**

:   提出 Memo，一种基于 Transformer 的记忆增强框架，通过周期性生成摘要 token（summary tokens）压缩历史上下文，在保持甚至超越全上下文 Transformer 性能的同时，将推理时 KV 缓存缩小 8-10 倍，并展现出更好的长上下文泛化和流式推理鲁棒性。

**[Meta-World An Improved Standardized Rl Benchmark](reinforcement_learning/meta-world_an_improved_standardized_rl_benchmark.md)**

:   本文系统揭示 Meta-World 基准在不同版本间因奖励函数不一致导致的算法比较失真问题，并发布标准化新版本 Meta-World+，明确保留 V1/V2 两套奖励函数，新增 MT25/ML25 任务集，升级至 Gymnasium API，实现完全可复现的多任务和元强化学习评估。

**[Metabox-V2 A Unified Benchmark Platform For Meta-Black-Box Optimization](reinforcement_learning/metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)**

:   MetaBox-v2 是对元黑箱优化（MetaBBO）基准平台的里程碑式升级，统一支持 RL/SL/NE/ICL 四大学习范式，复现 23 个基线算法，集成 18 个测试套件（1900+ 问题实例），并通过向量化环境和分布式测试实现 10-40 倍加速。

**[Mind The Gap The Challenges Of Scale In Pixel-Based Deep Reinforcement Learning](reinforcement_learning/mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)**

:   发现像素输入的深度 RL 网络中，编码器（卷积层 $\phi$）与全连接层（$\psi$）之间的"瓶颈连接"是阻碍网络缩放的根本原因，提出用全局平均池化（GAP）这一极简方法直接化解瓶颈，以更低计算成本获得与复杂方法（SoftMoE、稀疏训练）相当或更优的性能。

**[Models That Prove Their Own Correctness](reinforcement_learning/models_that_prove_their_own_correctness.md)**

:   本文提出 Self-Proving Models 框架，让模型通过交互式证明系统向验证算法证明其输出的正确性，并设计了 Transcript Learning (TL) 和 Reinforcement Learning from Verifier Feedback (RLVF) 两种学习方法，在 GCD 计算任务上实验验证 Annotated TL 可达 96% 的 Verifiability。

**[Modulation Of Temporal Decision-Making In A Deep Reinforcement Learning Agent Un](reinforcement_learning/modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)**

:   在简化版Overcooked环境中训练DRL智能体执行单任务（时间生产）和双任务（时间生产+数字比较），发现双任务智能体在四种目标时长下均显著过度生产时间——这一涌现行为与人类时间感知研究中双任务范式下的时间高估现象高度一致。

**[Mtl-Kd Multi-Task Learning Via Knowledge Distillation For Generalizable Neural V](reinforcement_learning/mtl-kd_multi-task_learning_via_knowledge_distillation_for_generalizable_neural_v.md)**

:   提出基于知识蒸馏的多任务学习框架MTL-KD，通过将多个RL单任务教师模型的策略知识蒸馏到一个重解码器学生模型中，实现了对多种VRP变体的高效统一求解，并在大规模问题上展现出卓越的泛化能力。

**[Multi-Agent Collaboration Via Evolving Orchestration](reinforcement_learning/multi-agent_collaboration_via_evolving_orchestration.md)**

:   提出"木偶师"(Puppeteer)式多 Agent 协作范式——一个中心化编排器通过 RL 学习在每个推理步骤动态选择激活哪个 Agent，在封闭域和开放域任务上同时提升性能和效率，并发现演化后的拓扑趋向更紧凑的环形结构。

**[Multi-Objective Reinforcement Learning With Max-Min Criterion A Game-Theoretic A](reinforcement_learning/multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)**

:   将熵正则化的 max-min 多目标强化学习重新建模为两人零和正则连续博弈，提出 ERAM/ARAM 算法，通过镜像下降实现闭式权重更新和全局 last-iterate 收敛，在多种 MORL 环境中显著超越基线。

**[Near-Optimal Quantum Algorithms For Computing Coarse Correlated Equilibria Of Ge](reinforcement_learning/near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)**

:   首次研究计算多玩家一般和博弈的相关均衡（CE）和粗相关均衡（CCE）的量子算法，通过量子化多尺度 MWU 方法和统一 QRAM 方案，实现 $\tilde{O}(m\sqrt{n})$ 的近最优查询复杂度（在玩家数 m 和动作数 n 上），并证明了匹配的量子下界。

**[Noisyrollout Reinforcing Visual Reasoning With Data Augmenta](reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)**

:   提出NoisyRollout，一种零额外训练成本的数据增强方法，在GRPO训练VLM时混合来自干净和适度扰动图像的rollout以增强策略探索多样性，仅用2.1K样本在5个域外基准上达到开源RL微调模型SOTA。

**[Non-Convex Entropic Mean-Field Optimization Via Best Response Flow](reinforcement_learning/non-convex_entropic_mean-field_optimization_via_best_response_flow.md)**

:   将Best Response Flow从凸函数泛函优化扩展到非凸情形，证明在充分大的熵正则化下，BR算子在 $L^1$-Wasserstein距离下成为压缩映射，保证非凸目标的唯一全局最小值存在性及指数收敛。

**[On The Global Optimality Of Policy Gradient Methods In General Utility Reinforce](reinforcement_learning/on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)**

:   本文为一般效用强化学习（RLGU）中的策略梯度方法建立了全局最优性理论保证：在表格设定下通过新的梯度支配不等式证明了全局收敛，在大规模状态-动作空间下提出基于最大似然估计（MLE）的占据度量近似算法 PG-OMA，样本复杂度仅依赖函数近似类的维度 $m$ 而非状态-动作空间大小。

**[Online Optimization For Offline Safe Reinforcement Learning](reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)**

:   提出 O3SRL 框架，将离线安全强化学习问题形式化为极小极大优化，通过结合离线 RL oracle 和基于 EXP3 多臂老虎机的在线优化来自适应调整拉格朗日乘子，避免了不稳定的离策略评估，在严格安全约束下实现高奖励。

**[Open Vision Reasoner Transferring Linguistic Cognitive Behavior For Visual Reaso](reinforcement_learning/open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)**

:   Open Vision Reasoner（OVR）通过"语言冷启动 + 大规模多模态 RL"两阶段训练范式，将语言模型中的认知行为（如回溯、验证）有效迁移到视觉推理中，基于 Qwen2.5-VL-7B 在 MathVision 上首次突破 50%（51.8%），成为同规模 SOTA。

**[Opinion Towards Unified Expressive Policy Optimization For Robust Robot Learning](reinforcement_learning/opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)**

:   提出 UEPO 框架，通过多种子动力学感知扩散策略、动态分歧正则化和基于扩散的数据增强三大核心组件，解决离线到在线强化学习中多模态行为覆盖不足和分布偏移问题，在 D4RL 基准上超越 Uni-O4。

**[Optimizing The Unknown Black Box Bayesian Optimization With Energy-Based Model A](reinforcement_learning/optimizing_the_unknown_black_box_bayesian_optimization_with_energy-based_model_a.md)**

:   提出REBMBO框架，将高斯过程（局部建模）、能量模型EBM（全局探索）和PPO强化学习（多步前瞻）统一为贝叶斯优化闭环，在高维/多峰黑盒优化中显著优于传统BO方法。

**[Oryx A Scalable Sequence Model For Many-Agent Coordination In Offline Marl](reinforcement_learning/oryx_a_scalable_sequence_model_for_many-agent_coordination_in_offline_marl.md)**

:   本文提出 Oryx，一种面向离线合作 MARL 的可扩展序列模型算法，将基于 Retention 的 Sable 架构与自回归形式的 ICQ 离线正则化结合，通过双解码器输出策略和 Q 值并利用反事实优势估计，在 65 个数据集上超过 80% 达到 SOTA，并展示了在 50 智能体规模下的稳健扩展能力。

**[Parameter-Free Algorithms For The Stochastically Extended Adversarial Model](reinforcement_learning/parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)**

:   针对桥接对抗性和随机在线凸优化的 SEA 模型，首次开发无参数算法：在未知域直径 $D$ 和/或 Lipschitz 常数 $G$ 条件下，基于 Optimistic Online Newton Step (OONS) 实现与已知参数情况相当的 regret 界。

**[Parameter Efficient Fine-Tuning Via Explained Variance Adaptation](reinforcement_learning/parameter_efficient_fine-tuning_via_explained_variance_adaptation.md)**

:   提出 Explained Variance Adaptation (EVA)，通过对激活向量进行增量 SVD 来初始化 LoRA 矩阵，可证明地最大化期望梯度信号，并结合自适应秩分配机制在语言生成/理解、图像分类、强化学习等多领域建立了精度-效率的新 Pareto 前沿。

**[Periodic Skill Discovery](reinforcement_learning/periodic_skill_discovery.md)**

:   提出 Periodic Skill Discovery (PSD) 框架，通过将状态映射到圆形潜空间来自然编码周期性，实现无监督地发现具有不同周期的多样化运动技能。

**[Prompt Tuning Decision Transformers With Structured And Scalable Bandits](reinforcement_learning/prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)**

:   提出一种基于多臂老虎机的结构化prompt调优方法，通过将prompt分解为独立segment并利用预训练PDT作为特征提取器，将prompt搜索复杂度从组合爆炸降为线性，在多任务离线RL中显著提升冻结PDT骨干网络的推理性能。

**[Provable Ordering And Continuity In Vision-Language Pretraining For Generalizabl](reinforcement_learning/provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)**

:   提出 AcTOL，通过视觉-语言排序损失和布朗桥约束来学习有序且连续的视觉-语言表征，无需刚性目标到达假设，在模拟和真实机器人操作任务上显著提升下游表现。

**[Quantifying Generalisation In Imitation Learning](reinforcement_learning/quantifying_generalisation_in_imitation_learning.md)**

:   本文提出 Labyrinth 基准环境，通过可控的迷宫结构变化实现训练与评估数据的严格分离，揭示了当前模仿学习方法在结构泛化上的严重不足（最佳方法在测试集仅 5% 成功率），为模仿学习的泛化评估提供了系统性工具。

**[Real-World Reinforcement Learning Of Active Perception Behaviors](reinforcement_learning/real-world_reinforcement_learning_of_active_perception_behaviors.md)**

:   提出非对称优势加权回归（AAWR），在训练时利用额外特权传感器来估计更准确的优势函数，从而高效学习真实世界中的主动感知策略，在8个涵盖不同部分可观测程度的操控任务上均超越所有基线方法。

**[Reasoning Gym Reasoning Environments For Reinforcement Learning With Verifiable ](reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)**

:   发布包含100+过程生成推理任务的Reasoning Gym库，覆盖代数、算术、算法、逻辑、几何、图论、游戏等领域，每个任务支持无限数据生成和参数化难度控制，实验证明RLVR训练在域内/跨域均实现显著技能迁移且能提升MATH、GSM8K等外部基准表现。

**[Reinforcement Learning For Long-Horizon Multi-Turn Search Agents](reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)**

:   展示 RL 训练的 14B 参数搜索 agent 在法律文档检索任务上通过多轮交互可以超越 frontier 模型（85% vs GPT o3 的 81%），关键在于精心设计的分段奖励结构和允许长 horizon 多轮交互。

**[Reinforcement Learning Teachers Of Test Time Scaling](reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)**

:   提出强化学习教师（RLT）框架，将问题和答案同时提供给教师模型，训练其生成有效的解释性推理链条，而非从零解题，从而用7B参数的小教师模型产出比数量级更大模型更优的蒸馏数据。

**[Reinforcement Learning With Action Chunking](reinforcement_learning/reinforcement_learning_with_action_chunking.md)**

:   提出 Q-chunking,将动作分块技术从模仿学习推广到基于 TD 的强化学习方法中,通过在"分块"动作空间上直接运行 RL 来改善长horizon稀疏奖励任务的探索和学习效率。

**[Repic Reinforced Post-Training For Personalizing Multi-Modal Language Models](reinforcement_learning/repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)**

:   提出首个基于强化学习的多模态大模型后训练框架 RePIC,用于个性化图像描述生成,在多概念场景中显著优于基于 SFT 的方法。

**[Retrosynthesis Planning Via Worst-Path Policy Optimisation In Tree-Structured Md](reinforcement_learning/retrosynthesis_planning_via_worst-path_policy_optimisation_in_tree-structured_md.md)**

:   将逆合成规划重构为树结构MDP中的最差路径(worst-path)优化问题——合成树的价值由最弱路径决定（任何一条死胡同路径将导致整棵树无效），提出InterRetro通过加权自模仿学习优化这一最差路径目标，在Retro*-190上达到100%成功率，路径长度缩短4.9%，仅需10%训练数据即达92%完整性能。

**[Reward-Aware Proto-Representations In Reinforcement Learning](reinforcement_learning/reward-aware_proto-representations_in_reinforcement_learning.md)**

:   系统发展了默认表示（DR）的理论基础——推导了 DP 和 TD 学习算法、分析了特征空间结构、提出了默认特征进行函数逼近——并在奖励塑形、期权发现、探索和迁移学习四个场景中展示了 DR 相比后继表示（SR）的奖励感知优势。

**[Risk-Averse Constrained Reinforcement Learning With Optimized Certainty Equivale](reinforcement_learning/risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)**

:   提出一种基于奖励层面(reward-based)的风险感知约束RL框架，使用优化确定性等价(OCE)风险度量同时覆盖目标和约束，建立了参数化强对偶性，并给出模块化算法——可包装标准RL求解器（如PPO）作为黑盒使用。

**[Risk-Averse Total-Reward Reinforcement Learning](reinforcement_learning/risk-averse_total-reward_reinforcement_learning.md)**

:   提出了面向无折扣总奖励准则(TRC)的风险规避Q-learning算法（ERM-TRC和EVaR-TRC），利用ERM的可引出性(elicitability)将Bellman算子转化为随机梯度下降形式，并证明了算法的收敛保证。

**[Rl Tango Reinforcing Generator And Verifier Together For Lan](reinforcement_learning/rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)**

:   Tango 提出一种交替 RL 训练生成器和验证器的框架——验证器是生成式过程级 LLM（用自然语言逐步评判），仅用结果级正确性奖励训练（无需步骤标注），通过与生成器的共进化相互增强——在 7B/8B 级别模型上达到SOTA，AIME 2025 准确率相对 vanilla GRPO 提升 100%。

**[Robot-R1 Reinforcement Learning For Enhanced Embodied Reasoning In Robotics](reinforcement_learning/robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)**

:   Robot-R1 提出利用强化学习（GRPO）训练大视觉语言模型（LVLM）进行具身推理，通过将下一关键状态预测转化为多选题并用 RL 优化推理路径，仅凭 7B 参数在低级控制推理任务上超越 GPT-4o。

**[Robust Adversarial Reinforcement Learning In Stochastic Games Via Sequence Model](reinforcement_learning/robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)**

:   提出CART（Conservative Adversarially Robust Decision Transformer），首个在随机博弈中增强Decision Transformer对抗鲁棒性的方法，通过阶段博弈建模和NashQ值估计解决ARDT在随机状态转移下的过度乐观问题，实现更准确的极小极大值估计和更优的最差情况回报。

**[Robust And Diverse Multi-Agent Learning Via Rational Policy Gradient](reinforcement_learning/robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)**

:   本文提出理性保持策略优化（RPO）框架和理性策略梯度（RPG）算法，通过引入操纵者智能体和对手塑造技术，在合作和一般和博弈场景中消除对抗优化导致的自毁行为，同时实现策略鲁棒化和多样化。

**[Roirl Efficient Self-Supervised Reasoning With Offline Iterative Reinforcement L](reinforcement_learning/roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)**

:   提出RoiRL——一种基于离线迭代强化学习的轻量级自监督推理框架，通过加权对数似然目标函数替代在线RL（如TTRL），在不需要参考模型和真实标签的情况下实现LLM推理能力的自我提升，训练速度提高2.5倍且性能更优。

**[Router-R1 Teaching Llms Multi-Round Routing And Aggregation Via Reinforcement Le](reinforcement_learning/router-r1_teaching_llms_multi-round_routing_and_aggregation_via_reinforcement_le.md)**

:   Router-R1 将多 LLM 路由和聚合建模为序列决策过程，用 LLM 自身作为路由器交替执行"思考"和"路由"动作，通过 PPO 训练配合格式/正确性/成本三重奖励，在 7 个 QA 基准上超越所有路由器基线且可泛化到未见过的 LLM。

**[Sample-Efficient Tabular Self-Play For Offline Robust Reinforcement Learning](reinforcement_learning/sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)**

:   提出 RTZ-VI-LCB 算法用于离线鲁棒两人零和 Markov 博弈（RTZM G），通过乐观鲁棒值迭代 + Bernstein 风格惩罚，实现近最优样本复杂度 $O(C_r^* \cdot H^4 \cdot S \cdot (A+B) / \varepsilon^2)$，较此前最优结果 $O(H^5 \cdot S^2 \cdot AB / \varepsilon^2)$ 在状态空间和动作空间依赖上均有显著改善。

**[Scalable Exploration Via Ensemble](reinforcement_learning/scalable_exploration_via_ensemble.md)**

:   提出 Ensemble++，通过共享因子矩阵的增量更新机制，仅需 $\Theta(d\log T)$ 的集成大小即可实现与精确 Thompson Sampling 相当的遗憾界，并自然扩展到非线性/神经网络场景。

**[Scalable Neural Incentive Design With Parameterized Mean-Field Approximation](reinforcement_learning/scalable_neural_incentive_design_with_parameterized_mean-field_approximation.md)**

:   提出 AMID 算法，将多智能体激励设计（ID）问题形式化为参数化平均场博弈（PMFG），证明有限$N$智能体目标以$\mathscr{O}(1/\sqrt{N})$速率逼近无限种群极限，在多种拍卖场景大幅提升收益。

**[Scalable Policy-Based Rl Algorithms For Pomdps](reinforcement_learning/scalable_policy-based_rl_algorithms_for_pomdps.md)**

:   提出将 POMDP 近似为有限状态的 Superstate MDP（状态为截断历史），给出更紧的最优值函数差上界（随历史长度指数衰减），并首次证明标准 TD 学习 + 策略优化在此非马尔可夫采样下的有限时间收敛保证。

**[Self-Improving Embodied Foundation Models](reinforcement_learning/self-improving_embodied_foundation_models.md)**

:   本文提出一种面向具身基础模型的两阶段后训练方法：第一阶段通过行为克隆和 steps-to-go 预测进行监督微调，第二阶段利用 steps-to-go 预测生成的自奖励函数和成功检测器实现在线 RL 自我改进，仅需 1-3% 额外数据即可实现 1.5x 以上的成功率提升，并首次展示了机器人自主学习超出模仿数据分布之外的新技能。

**[Sequential Monte Carlo For Policy Optimization In Continuous Pomdps](reinforcement_learning/sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)**

:   提出基于非马尔可夫 Feynman-Kac 模型的嵌套 SMC（Sequential Monte Carlo）算法，在连续 POMDP 中实现策略优化，天然捕获信息收集价值而无需手工启发式。

**[Sequential Multi-Agent Dynamic Algorithm Configuration](reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)**

:   提出 Seq-MADAC 框架，将多超参数动态配置建模为上下文顺序多智能体 MDP，通过顺序优势分解网络（SADN）利用参数间的固有依赖关系，在多目标优化算法配置上超越现有 MARL 方法。

**[Shift Before You Learn Enabling Low-Rank Representations In Reinforcement Learni](reinforcement_learning/shift_before_you_learn_enabling_low-rank_representations_in_reinforcement_learni.md)**

:   揭示了强化学习中后继度量（successor measure）本身并非近似低秩的，但"位移后继度量"（shifted successor measure）自然具有低秩结构；通过引入新的 Type II Poincaré 不等式量化所需位移量，为目标导向 RL 提供了有限样本理论保证和实践改进。

**[Simultaneous Swap Regret Minimization Via Kl-Calibration](reinforcement_learning/simultaneous_swap_regret_minimization_via_kl-calibration.md)**

:   提出 KL-Calibration 这一更强的校准度量，证明其等价于 log loss 的 swap regret，并通过非均匀离散化和新型随机取整方案实现 $\tilde{\mathcal{O}}(T^{1/3})$ 的同时 swap regret 上界，覆盖比已有工作更广的 proper loss 类。

**[Solving Continuous Mean Field Games Deep Reinforcement Learning For Non-Stationa](reinforcement_learning/solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)**

:   提出DEDA-FP算法，首次在连续状态/动作空间的非平稳平均场博弈（MFG）中同时学习Nash均衡策略和种群分布，通过结合深度RL计算最优响应、监督学习表示平均策略、条件Normalizing Flow建模时变种群分布，实现了比现有方法快10倍以上的采样效率。

**[Solving Neural Min-Max Games The Role Of Architecture Initialization Dynamics](reinforcement_learning/solving_neural_min-max_games_the_role_of_architecture_initialization_dynamics.md)**

:   首次为两层神经网络参数化的零和博弈提供收敛保证，证明在适当过参数化、随机初始化和交替梯度下降上升（AltGDA）下，能以高概率收敛到 $\epsilon$-近似纳什均衡。

**[Spatial-Aware Decision-Making With Ring Attractors In Reinforcement Learning Sys](reinforcement_learning/spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)**

:   将神经科学中的环形吸引子模型集成到 DRL 的动作选择中，通过将动作映射到环上的空间位置并利用高斯信号注入 Q 值和不确定性，在 Atari 100K 上比基线提升 53%。

**[Stair Addressing Stage Misalignment Through Temporal-Aligned Preference Reinforc](reinforcement_learning/stair_addressing_stage_misalignment_through_temporal-aligned_preference_reinforc.md)**

:   发现并形式化了偏好强化学习（PbRL）中的"阶段错位"问题——比较不同阶段的行为片段会产生无效反馈，提出STAIR方法通过对比学习获取时间距离来近似阶段差异，用四边形距离选择阶段对齐的查询，在多阶段任务中显著超越现有PbRL方法。

**[Strategic Costs Of Perceived Bias In Fair Selection](reinforcement_learning/strategic_costs_of_perceived_bias_in_fair_selection.md)**

:   通过博弈论模型揭示"感知驱动偏差"机制：在完全基于能力的选拔系统中，不同社会经济群体对选拔后价值的感知差异会导致理性的努力差异，从而在"公平"的流程中系统性地传播不平等。

**[Structural Information-Based Hierarchical Diffusion For Offline Reinforcement Le](reinforcement_learning/structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)**

:   提出SIHD框架，利用历史轨迹中的结构信息（结构熵）自适应构建多尺度扩散层次，用结构信息增益替代局部奖励预测作为条件引导信号，并引入结构熵正则化促进对离线数据中稀疏状态的探索，在D4RL基准上最高提升12.6%的决策性能。

**[Structured Reinforcement Learning For Combinatorial Decision-Making](reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)**

:   提出 Structured Reinforcement Learning (SRL)，将组合优化求解器作为可微层嵌入 actor-critic 的 actor 中，通过 Fenchel-Young 损失 + 高斯扰动实现端到端梯度传播，纯在线学习、无需专家数据，在6个工业级组合决策问题上匹配模仿学习、超越无结构 RL 最高 92%。

**[Swe-Rl Advancing Llm Reasoning Via Reinforcement Learning On Open Software Evolu](reinforcement_learning/swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)**

:   首次将强化学习 (RL) 应用于真实世界软件工程任务（GitHub PR/Issue 修复），仅用基于规则的序列相似度奖励训练 Llama-3.3-70B，在 SWE-bench Verified 上达到 41.0% 解决率（中等规模模型 SOTA），且 RL 训练仅在 issue-solving 数据上进行，却涌现出在代码推理、数学、通用语言理解等域外任务上的泛化推理能力。

**[Teaching Language Models To Evolve With Users Dynamic Profile Modeling For Perso](reinforcement_learning/teaching_language_models_to_evolve_with_users_dynamic_profile_modeling_for_perso.md)**

:   将个性化对话对齐建模为多轮马尔可夫决策过程，提出 RLPA 框架，让 LLM 通过与模拟用户的在线交互学习动态推断和维护用户画像，并据此生成个性化回复。

**[Temporal-Difference Variational Continual Learning](reinforcement_learning/temporal-difference_variational_continual_learning.md)**

:   提出TD-VCL目标函数，将变分持续学习（VCL）中的学习目标重新表示为多个过去后验估计的加权组合，揭示了与强化学习中时序差分（TD）方法的深层联系，通过"分散"正则化压力有效缓解了近似误差的逐步累积问题。

**[The Burden Of Interactive Alignment With Inconsistent Preferences](reinforcement_learning/the_burden_of_interactive_alignment_with_inconsistent_preferences.md)**

:   将用户与参与度驱动算法的交互建模为多领导者-单跟随者 Stackelberg 博弈，证明存在关键的前瞻视野阈值：超过该阈值的用户可实现对齐，否则反被算法对齐；同时证明引入低成本信号（如额外点击）可大幅降低对齐负担。

**[The Physical Basis Of Prediction World Model Formation In Neural Organoids Via A](reinforcement_learning/the_physical_basis_of_prediction_world_model_formation_in_neural_organoids_via_a.md)**

:   本文提出在人类神经类器官（organoids）中研究世界模型形成的框架，设计了三个渐进式虚拟环境（条件回避、捕食者-猎物、Pong），并引入 LLM 自动生成实验方案的元学习方法，结合多尺度生物物理评估策略量化生物学习的物理基础。

**[The World Is Bigger A Computationally-Embedded Perspective On The Big World Hypo](reinforcement_learning/the_world_is_bigger_a_computationally-embedded_perspective_on_the_big_world_hypo.md)**

:   从计算嵌入（computationally-embedded）的视角形式化了"大世界假说"，证明被嵌入在通用局部环境中的智能体天然受限于自身容量，提出"交互性"（interactivity）作为持续适应能力的计算度量，并实验表明深度非线性网络难以维持交互性，而深度线性网络可随容量增加而提升交互性。

**[Thompson Sampling For Multi-Objective Linear Contextual Bandit](reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)**

:   提出MOL-TS——首个具有worst-case Pareto regret理论保证的多目标线性上下文Bandit Thompson Sampling算法，通过定义"有效Pareto最优臂"概念和乐观采样策略，实现$\widetilde{O}(d^{3/2}\sqrt{T})$的regret上界，目标数$L$仅增加$O(\log L)$因子。

**[Thompson Sampling In Function Spaces Via Neural Operators](reinforcement_learning/thompson_sampling_in_function_spaces_via_neural_operators.md)**

:   将 Thompson 采样 (TS) 从有限维参数空间扩展到无限维函数空间，利用神经算子 (Neural Operators) 作为高斯过程后验的近似采样器，实现了对涉及偏微分方程 (PDE) 的功能优化问题的高效求解。

**[Time Reversal Symmetry For Efficient Robotic Manipulations In Deep Reinforcement](reinforcement_learning/time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)**

:   提出 TR-DRL 框架，利用机器人操作任务中的时间反转对称性——通过轨迹反转增强（完全可逆的转移）和时间反转引导的势函数奖励塑形（部分可逆的转移）——显著提升 DRL 在成对任务（如开门/关门）中的样本效率和最终性能。

**[To Distill Or Decide Understanding The Algorithmic Trade-Off In Partially Observ](reinforcement_learning/to_distill_or_decide_understanding_the_algorithmic_trade-off_in_partially_observ.md)**

:   通过一个理论模型（perturbed Block MDP）和模拟运动控制实验，系统研究了部分可观测 RL 中**特权专家蒸馏** (privileged expert distillation) 与**标准 RL**（无特权信息）之间的算法权衡，发现权衡关键取决于隐状态动力学的随机性。

**[Towards Provable Emergence Of In-Context Reinforcement Learning](reinforcement_learning/towards_provable_emergence_of_in-context_reinforcement_learning.md)**

:   本文从理论上证明了 Transformer 经过标准 RL 预训练后，其全局最优参数能够实现 in-context temporal difference (TD) 学习，为 in-context RL (ICRL) 现象提供了首个可证明的理论支撑。

**[Tractable Multinomial Logit Contextual Bandits With Non-Linear Utilities](reinforcement_learning/tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)**

:   首次为MNL上下文赌博机问题在非线性效用函数（含神经网络）下设计了**计算可行**且**统计最优**的算法ONL-MNL，在不依赖NTK假设的情况下达到$\widetilde{\mathcal{O}}(\sqrt{T})$的遗憾上界。

**[Training Language Models To Reason Efficiently](reinforcement_learning/training_language_models_to_reason_efficiently.md)**

:   通过在 RL 奖励中加入长度惩罚项——正确回答的奖励乘以 $(1 - \alpha \cdot \sigma(\text{norm\_len}))$，用单一超参数 $\alpha$ 控制 token-准确率权衡曲线，仅 100 步 RL 训练即可让 7B 推理模型减少 50% token 使用量而准确率仅下降 <5%。

**[Trico Triadic Game-Theoretic Co-Training For Robust Semi-Supervised Learning](reinforcement_learning/trico_triadic_game-theoretic_co-training_for_robust_semi-supervised_learning.md)**

:   提出 TRiCo 框架，将半监督学习重构为教师-双学生-对抗生成器的三方博弈（Stackelberg 博弈），用互信息替代置信度做伪标签筛选，元学习教师自适应调节训练动态，在低标签场景下实现 SOTA 性能。

**[Trust Region Reward Optimization And Proximal Inverse Reward Optimization Algori](reinforcement_learning/trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)**

:   提出 TRRO 理论框架和 PIRO 实用算法，通过 Minorization-Maximization 过程保证 IRL 中奖励函数更新的单调改进，实现了逆强化学习领域类似于 TRPO/PPO 在正向 RL 中的稳定性保证。

**[Variance-Aware Feel-Good Thompson Sampling For Contextual Bandits](reinforcement_learning/variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)**

:   提出FGTS-VA算法，首次实现了基于Feel-Good Thompson Sampling的方差感知上下文赌博机算法，其后悔界在模型维度上达到最优，匹配了基于UCB的最优方差依赖后悔界。

**[Videorft Incentivizing Video Reasoning Capability In Mllms Via Reinforced Fine-T](reinforcement_learning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)**

:   提出 VideoRFT，通过认知启发的多专家 CoT 数据构建流水线和新颖的语义一致性奖励，将强化微调（RFT）范式扩展到视频推理领域，分别构建 VideoRFT-CoT-102K（SFT 用）和 VideoRFT-RL-310K（RL 用）两个数据集，在 6 个视频推理基准上达到 SOTA。

**[Viki-R Coordinating Embodied Multi-Agent Cooperation Via Reinforcement Learning](reinforcement_learning/viki-r_coordinating_embodied_multi-agent_cooperation_via_reinforcement_learning.md)**

:   构建了首个面向具身多智能体合作的层次化基准VIKI-Bench（含智能体激活、任务规划、轨迹感知三个层级），并提出两阶段框架VIKI-R（CoT示范微调+多级奖励RL），在多种机器人形态和多视角视觉观测下实现显著超越基线的合作表现，RL阶段涌现出组合式协作模式。

**[Volleybots A Testbed For Multi-Drone Volleyball Game Combining Motion Control An](reinforcement_learning/volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)**

:   本文提出 VolleyBots，一个多无人机排球竞技测试平台，融合了合作-对抗博弈、回合制交互与敏捷 3D 机动控制，基于 Isaac Sim 构建了从单体训练到多体竞技的任务课程体系，并通过分层策略在 3v3 任务中取得 69.5% 胜率，同时展示了零样本 sim-to-real 部署能力。

**[When Can Model-Free Reinforcement Learning Be Enough For Thinking](reinforcement_learning/when_can_model-free_reinforcement_learning_be_enough_for_thinking.md)**

:   提出 Thought MDP 形式化框架来理解模型无关 RL 中"思考"行为的涌现条件：策略初始化是决定性因素，思考动作等价于智能体在行动前执行一步策略改进，且开源 LLM 满足思考涌现的必要条件。

**[When Less Language Is More Language-Reasoning Disentanglement Makes Llms Better ](reinforcement_learning/when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)**

:   受认知神经科学启发（人脑的推理与语言处理相对独立），在 LLM 的激活空间中识别并消除语言特定成分，实现语言与推理的解耦，从而在免训练条件下一致性地提升多语言推理性能。

**[Zero-Shot Context Generalization In Reinforcement Learning From Few Training Con](reinforcement_learning/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)**

:   提出 Context-Enhanced Bellman Equation (CEBE) 和 Context Sample Enhancement (CSE) 方法，通过利用环境动力学和奖励函数对上下文参数的一阶导数信息，在仅训练于单一上下文的情况下实现对未见上下文的零样本泛化。

**[Zeroth-Order Optimization Finds Flat Minima](reinforcement_learning/zeroth-order_optimization_finds_flat_minima.md)**

:   首次从理论上证明标准零阶优化（两点梯度估计）具有隐式正则化效果——收敛到Hessian迹最小的平坦极小值（flat minima），在凸且充分光滑条件下给出了$T = \mathcal{O}(d^4/\epsilon^2)$的收敛复杂度保证。

---

## 🏥 医学图像 { #medical_imaging }

**[3Drad A Comprehensive 3D Radiology Medvqa Dataset With Multi](medical_imaging/3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)**

:   提出 3D-RAD——首个大规模3D医学VQA基准，包含170K条CT影像问答数据，覆盖六类临床任务（含创新性的多时相诊断任务），并配套136K训练集，揭示了现有VLM在3D时序推理上的严重不足。

**[A Novel Approach To Classification Of Ecg Arrhythmia Types With Latent Odes](medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)**

:   将路径最小化 Latent ODE 的编码器与梯度提升决策树（GBDT）组合为两阶段 ECG 心律失常分类流水线，在 MIT-BIH 数据集上的 macro AUC-ROC 从 360Hz 的 0.984 仅降至 45Hz 的 0.976，展示了对采样频率变化的强鲁棒性。

**[A Unified Solution To Video Fusion From Multi-Frame Learning To Benchmarking](medical_imaging/a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)**

:   提出首个统一视频融合框架 UniVF（基于多帧学习 + 光流特征 warping + 时序一致性损失），并构建首个覆盖四大融合任务（多曝光、多焦点、红外-可见光、医学）的视频融合基准 VF-Bench，在全部子任务上取得 SOTA。

**[A Variational Manifold Embedding Framework For Nonlinear Dimensionality Reductio](medical_imaging/a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)**

:   提出一种变分流形嵌入框架，将降维问题形式化为最优嵌入映射的优化问题（最小化先验分布与数据分布pullback之间的KL散度），在理论上统一了PCA与非线性降维方法，并利用变分法（Euler-Lagrange方程）和Noether定理为最优嵌入提供了可解释性约束。

**[Aanet Virtual Screening Under Structural Uncertainty Via Alignment And Aggregati](medical_imaging/aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)**

:   针对现实药物发现中蛋白质 holo 结构不可用的问题，提出 AANet——通过三模态对比学习（配体-holo pocket-检测cavity）对齐表征并用交叉注意力聚合多个候选结合位点，在 apo/predicted 蛋白质结构上的盲筛性能远超 SOTA（DUD-E 上 EF1% 从 11.75 提升至 37.19）。

**[Active Target Discovery Under Uninformative Prior The Power Of Permanent And Tra](medical_imaging/active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)**

:   提出 EM-PTDM 框架，受神经科学双记忆系统启发，利用预训练扩散模型作为"永久记忆"并结合基于 Doob's h-transform 的轻量"瞬时记忆"模块，在**无领域先验数据**的条件下实现高效的主动目标发现，理论保证先验单调改进。

**[Amortized Active Generation Of Pareto Sets](medical_imaging/amortized_active_generation_of_pareto_sets.md)**

:   提出 A-GPS 框架，通过学习 Pareto 集的条件生成模型实现在线离散黑箱多目标优化——用非支配类概率估计器（CPE）作为 PHVI 的隐式估计替代显式超体积计算，并通过偏好方向向量实现摊还式后验偏好条件化（无需重新训练），在合成基准和蛋白质设计任务上展示了优越的样本效率。

**[Atomic Diffusion Models For Small Molecule Structure Elucidation From Nmr Spectr](medical_imaging/atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)**

:   提出 ChefNMR，首个基于 3D 原子扩散模型的端到端框架，仅从 1D NMR 光谱和化学式直接预测未知小分子（尤其是复杂天然产物）的分子结构，在合成和实验数据集上均达到 SOTA。

**[Augmenting Biological Fitness Prediction Benchmarks With Landscapes Features Fro](medical_imaging/augmenting_biological_fitness_prediction_benchmarks_with_landscapes_features_fro.md)**

:   GraphFLA 是一个高效的适应度景观分析框架——计算 20 个生物学意义的景观特征（粗糙度/上位性/可导航性/中性），在 5300+ 真实景观（ProteinGym/RNAGym/CIS-BP）上揭示模型性能高度依赖景观拓扑，如 VenusREM 在高可导航性景观上优于 ProSST 但在高上位性景观上弱于后者，处理百万突变体仅需 20 秒（vs MAGELLAN 5 小时）。

**[Autoencoding Random Forests](medical_imaging/autoencoding_random_forests.md)**

:   RFAE 首次为随机森林构建了原则性的编码-解码框架——利用 RF 核的正定性和普适性进行扩散映射谱分解得到低维编码，通过 k-NN 回归在叶节点空间中解码回原始特征，在 20 个表格数据集上重建质量排名 1.80（大幅优于 TVAE 3.38、AE 3.27），并成功应用于 MNIST 重建和 scRNA-seq 批次效应去除。

**[Barcodemamba Advancing State-Space Models For Fungal Biodiversity Research](medical_imaging/barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)**

:   BarcodeMamba+ 是面向真菌 ITS DNA 条形码分类的 SSM 基础模型，通过预训练+微调范式充分利用海量未标注序列，并结合层次标签平滑、逆平方根加权损失和多头输出三项增强，在三个测试集所有分类层级上大幅超越 BLAST、CNN 和 Transformer 基线，种级准确率最高达 88.9%。

**[Bidirectional Representations Augmented Autoregressive Biological Sequence Gener](medical_imaging/bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)**

:   CrossNovo 融合自回归（AR）和非自回归（NAR）解码器，通过共享谱编码器 + 重要性退火 + 梯度阻断知识蒸馏，让 NAR 的双向全局理解增强 AR 的序列生成能力，在 9-Species 基准上氨基酸精度达 0.811（+2.6%）、肽段召回 0.654（+5.3%）。

**[Brain-Tuning Improves Generalizability And Efficiency Of Brain Alignment In Spee](medical_imaging/brain-tuning_improves_generalizability_and_efficiency_of_brain_alignment_in_spee.md)**

:   提出 Multi-brain-tuning 方法，通过联合多个被试的 fMRI 数据微调预训练语音模型，将脑对齐所需数据量降低 5 倍，同时脑对齐度提升最高 50%，并可泛化到全新被试和数据集。

**[Brain Harmony A Multimodal Foundation Model Unifying Morphology And Function Int](medical_imaging/brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)**

:   首个统一脑结构形态（T1 sMRI）与功能动态（fMRI）的多模态脑基础模型，通过几何谐波预对齐和时序自适应 Patch Embedding（TAPE）将高维神经影像压缩为紧凑的 1D token 表示，在神经发育/退行性疾病诊断和认知预测任务上全面超越先前方法。

**[Bridging Graph And State-Space Modeling For Intensive Care Unit Length Of Stay P](medical_imaging/bridging_graph_and_state-space_modeling_for_intensive_care_unit_length_of_stay_p.md)**

:   提出 S2G-Net，将 Mamba 状态空间模型的时序编码与多视图图神经网络（GraphGPS）进行双路融合，用于 ICU 住院时长（LOS）预测，在 MIMIC-IV 数据集上全面超越序列模型、图模型和混合基线。

**[Care-Pd A Multi-Site Anonymized Clinical Dataset For Parkinsons Disease Gait Ass](medical_imaging/care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)**

:   发布 Care-PD——目前最大的面向帕金森病步态分析的多站点匿名 3D 网格数据集（9 个队列、8 个临床中心、362 名受试者、8477 段步行），并在 UPDRS 步态评分和运动预训练任务上提供系统性 benchmark，证明在 Care-PD 上微调可将 MPJPE 从 60.8mm 降至 7.5mm，F1 提升 17 个百分点。

**[Cgbench Benchmarking Language Model Scientific Reasoning For Clinical Genetics R](medical_imaging/cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)**

:   提出 CGBench，一个基于 ClinGen 专家标注的临床遗传学 benchmark，从变异和基因策展角度评估 LLM 的科学文献推理能力，涵盖证据评分、证据验证和实验证据提取三个任务，发现推理模型在细粒度任务上表现最佳但在高层判断上不如非推理模型。

**[Codecrash Exposing Llm Fragility To Misleading Natural Language In Code Reasonin](medical_imaging/codecrash_exposing_llm_fragility_to_misleading_natural_language_in_code_reasonin.md)**

:   提出 CodeCrash 压力测试框架，通过功能等价的结构扰动和误导性自然语言注入（注释/print/暗示），系统评估 17 个 LLM 的代码推理鲁棒性，揭示模型平均性能下降 23.2%，CoT 仅能挽回至 13.8%，并首次发现大推理模型（LRM）中的 "Reasoning Collapse" 现象。

**[Compressing Biology Evaluating The Stable Diffusion Vae For Phenotypic Drug Disc](medical_imaging/compressing_biology_evaluating_the_stable_diffusion_vae_for_phenotypic_drug_disc.md)**

:   首次系统评估 Stable Diffusion VAE（SD-VAE）在 Cell Painting 显微镜图像上的重建质量，发现 SD-VAE 在像素级和生物信号层面均能良好保留表型信息（FR 几乎无下降），且通用特征提取器 InceptionV3 在检索任务上与领域专用模型 OpenPhenom 持平甚至更优。

**[Confrover Simultaneous Modeling Of Protein Conformation And Dynamics Via Autoreg](medical_imaging/confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)**

:   ConfRover 提出自回归框架将蛋白质 MD 轨迹分解为逐帧条件生成 $p(\mathbf{x}^{1:L}) = \prod_l p(\mathbf{x}^l | \mathbf{x}^{<l})$，通过编码器 + 因果 Transformer + SE(3) 扩散解码器的模块化架构，首次在单一模型中统一轨迹模拟、时间无关构象采样和构象插值三大任务，在 ATLAS 数据集上全面超越 MDGen。

**[Consistent Sampling And Simulation Molecular Dynamics With Energy-Based Diffusio](medical_imaging/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)**

:   本文发现扩散模型在采样和模拟之间存在不一致性问题（尤其在小扩散时间步），提出基于 Fokker-Planck 方程的正则化项来强制一致性，并结合时间分段的混合专家（MoE）策略，实现了在多个生物分子系统上一致且高效的采样与分子动力学模拟。

**[Convolutional Monge Mapping Between Eeg Datasets To Support Independent Componen](medical_imaging/convolutional_monge_mapping_between_eeg_datasets_to_support_independent_componen.md)**

:   本文扩展 CMMN（Convolutional Monge Mapping Normalization）方法，提出通道平均 PSD + $\ell_1$ 归一化质心和 subject-to-subject 匹配两种策略，生成单一时域滤波器实现不同通道数的 EEG 数据集间域适应，在独立成分（IC）脑/非脑分类中 F1 从 0.77 提升至 0.84，超越 ICLabel（0.88→0.91）。

**[Cureagent A Training-Free Executor-Analyst Framework For Clinical Reasoning](medical_imaging/cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)**

:   CureAgent 提出 Executor-Analyst 协作框架，将精确工具调用（TxAgent/Llama-8B 做 Executor）与高层临床推理（Gemini 2.5 做 Analyst）解耦，配合分层集成（Stratified Ensemble）的 Late Fusion 拓扑保留证据多样性，在 CURE-Bench 上达到 83.8% 准确率（无需端到端微调），揭示了上下文-性能悖论和动作空间维度灾难两个关键 scaling 发现。

**[Cxreasonbench A Benchmark For Evaluating Structured Diagnostic Reasoning In Ches](medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)**

:   提出 CheXStruct + CXReasonBench，一个基于胸部X光的结构化诊断推理评估框架，通过多路径、多阶段评估揭示现有 LVLM 在中间推理步骤上的严重不足。

**[Dca Graph-Guided Deep Embedding Clustering For Brain Atlases](medical_imaging/dca_graph-guided_deep_embedding_clustering_for_brain_atlases.md)**

:   DCA（Deep Cluster Atlas）提出图引导深度嵌入聚类框架，结合预训练 Swin-UNETR 的体素级时空嵌入和 KNN 图空间正则化，通过 KL 散度对齐软分配与图谱聚类辅助标签，生成功能一致且空间连续的个性化脑图谱，在 HCP 数据集上同态性提升 98.8%、轮廓系数提升 29%，并在自闭症诊断、认知解码等下游任务中超越现有图谱。

**[De Novo Generation Of Functional Terpene Synthases Using Tpsgpt](medical_imaging/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)**

:   TpsGPT 通过在 79K 萜烯合酶（TPS）序列上微调蒸馏版 ProtGPT2 Tiny（38.9M 参数），生成 28K 候选序列，经多阶段过滤（困惑度/pLDDT/EnzymeExplorer/CLEAN/InterPro/Foldseek）筛选出 7 条进化距离远（<60% 序列相似度）但结构保守的从头 TPS 序列，湿实验验证其中 2 条具有 TPS 酶活性——以不到 $200 GPU 成本实现功能酶从头设计。

**[Demo Generative Ai Helps Radiotherapy Planning With User Preference](medical_imaging/demo_generative_ai_helps_radiotherapy_planning_with_user_preference.md)**

:   提出 Flexible Dose Proposer (FDP)，通过两阶段训练框架（VQ-VAE 预训练 + 多条件编码）实现基于滑块的用户偏好交互式 3D 剂量分布预测，并集成到 Eclipse 临床治疗计划系统中，在头颈部癌症放疗场景中超越 Varian RapidPlan。

**[Demo Guide-Rag Evidence-Driven Corpus Curation For Retrieval-Augmented Generatio](medical_imaging/demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)**

:   系统评估了六种 RAG 语料库配置用于长新冠（Long COVID）临床问答，发现将临床指南与高质量系统综述结合的 GS-4 配置在 faithfulness、relevance 和 comprehensiveness 三维度上一致优于单指南和大规模文献库方案，并提出 Guide-RAG 框架和 LongCOVID-CQ 评估数据集。

**[Dermacon-In A Multi-Concept Annotated Dermatological Image Dataset Of Indian Ski](medical_imaging/dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)**

:   构建了 DermaCon-IN——首个以印度肤色为主的密集标注皮肤病图像数据集（5,450 张 / 3,002 患者 / 245 种诊断），提供三级层次诊断标签、47 个病灶描述符和 49 个解剖位置标注，并用 CNN/ViT/概念瓶颈模型进行基准评测。

**[Designx Human-Competitive Algorithm Designer For Black-Box Optimization](medical_imaging/designx_human-competitive_algorithm_designer_for_black-box_optimization.md)**

:   提出 DesignX，首个统一学习算法工作流生成和超参数动态控制两个子任务的自动算法设计框架，通过双 Transformer 智能体在 10k 合成问题上大规模预训练，在合成测试集和蛋白质对接/AutoML/UAV 路径规划等真实场景中超越人类手工设计的优化器。

**[Dison Decentralized Isolation Networks For Out-Of-Distribution Detection In Medi](medical_imaging/dison_decentralized_isolation_networks_for_out-of-distribution_detection_in_medi.md)**

:   提出 Decentralized Isolation Networks (DIsoN)，通过训练二分类器将测试样本从训练数据中"隔离"来检测 OOD，并通过去中心化参数交换实现在不共享数据的情况下利用训练数据信息，在 4 个医学影像数据集 12 个 OOD 检测任务上取得 SOTA。

**[Ditch The Denoiser Emergence Of Noise Robustness In Self-Supervised Learning Fro](medical_imaging/ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)**

:   提出一种全自监督的噪声鲁棒表示学习框架，通过"去噪→噪声"的数据课程学习策略 + 去噪教师正则化，使 DINOv2 等 SSL 模型在推理时无需去噪器即可直接处理噪声输入，在 ImageNet-1k 极端高斯噪声下线性探测精度提升 4.8%。

**[Doctor Approved Generating Medically Accurate Skin Disease Images Through Ai-Exp](medical_imaging/doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)**

:   提出 MAGIC 框架，通过将皮肤科专家定义的临床检查清单转化为 MLLM（如 GPT-4o）可执行的评估反馈，利用 DPO 或奖励模型微调扩散模型，生成临床准确的皮肤病图像用于数据增强，在 20 类皮肤病分类任务上提升 +9.02%，少样本场景提升 +13.89%。

**[Domain-Adaptive Transformer For Data-Efficient Glioma Segmentation In Sub-Sahara](medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)**

:   提出 SegFormer3D+，一种面向撒哈拉以南非洲异质 MRI 数据的域自适应 Transformer 架构，通过直方图匹配、影像组学分层采样、频率感知双路径编码器和双注意力机制，在仅 60 例标注数据微调下实现胶质瘤分割 mean Dice 0.81，超越 nnU-Net +2.5%。

**[Dual Mixture-Of-Experts Framework For Discrete-Time Survival Analysis](medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)**

:   提出双混合专家（Dual MoE）框架用于离散时间生存分析，结合特征编码器 MoE（建模患者亚组异质性）与风险网络 MoE（捕获时间动态），在 METABRIC 和 GBSG 乳腺癌数据集上提升 time-dependent C-index 最高 0.04。

**[Dyg-Mamba Continuous State Space Modeling On Dynamic Graphs](medical_imaging/dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)**

:   DyG-Mamba 将连续状态空间模型（SSM）引入动态图学习，设计时间跨度感知的连续 SSM——用 Ebbinghaus 遗忘曲线启发的指数衰减函数建模不规则时间间隔，配合谱范数约束的输入依赖参数实现 Lipschitz 鲁棒性，在 12 个动态图基准上平均排名 2.42（vs DyGFormer 2.92），且保持 $O(bdL)$ 线性复杂度。

**[Edbench Large-Scale Electron Density Data For Molecular Modeling](medical_imaging/edbench_large-scale_electron_density_data_for_molecular_modeling.md)**

:   构建了目前最大规模的电子密度（ED）数据集 EDBench（330 万分子，基于 B3LYP/6-31G** DFT 计算），并设计了涵盖预测、检索、生成三类任务的 ED 基准评估体系，首次系统评估了深度学习模型对电子密度的理解和利用能力。

**[Efficient Adaptive Experimentation With Noncompliance](medical_imaging/efficient_adaptive_experimentation_with_noncompliance.md)**

:   提出 AMRIV——首个面向带非依从性（noncompliance）的自适应实验的半参数高效、多重鲁棒的ATE估计器，结合方差最优的工具变量分配策略和序贯推断保证。

**[Endobench A Comprehensive Evaluation Of Multi-Modal Large Language Models For En](medical_imaging/endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)**

:   提出 EndoBench，首个覆盖 4 种内窥镜场景、12 项临床任务、5 级视觉提示粒度的综合 MLLM 评估基准，包含 6,832 个经临床验证的 VQA 对，对 23 个 MLLM 的评估显示商用模型整体领先但仍落后人类专家。

**[Energy Matching Unifying Flow Matching And Energy-Based Models For Generative Mo](medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)**

:   提出 Energy Matching，通过学习一个时间无关的标量势能场统一流匹配与能量模型：远离数据流形时沿最优传输路径高效传输，靠近流形时过渡为 Boltzmann 平衡分布以建模似然，在 CIFAR-10 上 FID 3.34 大幅超越现有 EBM（>50%提升）。

**[Ewc-Guided Diffusion Replay For Exemplar-Free Continual Learning In Medical Imag](medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)**

:   提出将类条件 DDPM 扩散重放与弹性权重巩固（EWC）相结合的无样本持续学习框架，在 MedMNIST v2（8 个 2D/3D 任务）和 CheXpert 上实现了 AUROC 0.851，相比 DER++ 遗忘率降低超 30%，接近联合训练上界（0.869），同时完全无需存储患者原始数据。

**[Exploring And Leveraging Class Vectors For Classifier Editing](medical_imaging/exploring_and_leveraging_class_vectors_for_classifier_editing.md)**

:   提出 Class Vector（类向量），通过计算预训练与微调模型在潜空间中类别质心的差异来捕获类别级适应，利用线性和独立性两个性质，通过简单向量算术实现分类器编辑（遗忘、环境适应、对抗防御），无需重训练即可完成潜空间注入，或用 <1.5K 参数在 1.5 秒内完成权重空间映射。

**[Fairgrpo Fair Reinforcement Learning For Equitable Clinical Reasoning](medical_imaging/fairgrpo_fair_reinforcement_learning_for_equitable_clinical_reasoning.md)**

:   提出 FairGRPO，一种层级式公平强化学习算法，通过自适应重要性加权（基于群体表示量和任务难度）解决临床 AI 中的人群表现差异问题，在 7 个临床数据集（280K样本，5种模态）上将预测平价降低 27.2%、F1 提升 12.49%，并发布首个公平性优化的临床 VLLM——FairMedGemma-4B。

**[Faithful Summarization Of Consumer Health Queries A Cross-Lingual Framework With](medical_imaging/faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with.md)**

:   提出结合 TextRank 抽取式句子选择和医学命名实体识别 (NER) 来引导 LLM 生成忠实医学摘要的框架，在英文 MeQSum 和孟加拉语 BanglaCHQ-Summ 数据集上通过微调 LLaMA-2-7B 实现质量和忠实性的一致提升，SummaC 达 0.57，人工评估 82% 摘要保留关键医学信息。

**[Fapex Fractional Amplitude-Phase Expressor For Robust Cross-Subject Seizure Pred](medical_imaging/fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)**

:   提出 FAPEX 框架，通过可学习的分数阶神经帧算子 (FrNFO) 实现自适应时频分解，结合幅度-相位交叉编码和空间相关性聚合，在 12 个跨物种、跨模态的癫痫预测基准上全面超越 33 个基线方法。

**[Far From The Shallow Brain-Predictive Reasoning Embedding Through Residual Disen](medical_imaging/far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)**

:   提出残差解纠缠方法，将 LLM 隐藏状态分离为词汇、句法、语义、推理四个近正交嵌入，用于预测颅内 ECoG 脑信号，发现推理信号在时间上（~350-400ms）和空间上（超越经典语言区扩展至视觉皮层）均具有独立的神经特征，揭示了 LLM 与人脑间的推理计算对齐。

**[Few-Shot Learning From Gigapixel Images Via Hierarchical Vision-Language Alignme](medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)**

:   提出 HiVE-MIL，一个层级视觉-语言 MIL 框架，通过构建统一异构图建模跨尺度层级关系（5× 和 20×）和同尺度多模态对齐，配合文本引导的动态过滤机制和层级对比损失，在 TCGA 肺/乳腺/肾癌三个数据集的 16-shot 设置下全面超越已有方法，Macro F1 最高提升 4.1%。

**[Fgbench A Dataset And Benchmark For Molecular Property Reasoning At Functional G](medical_imaging/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)**

:   本文提出 FGBench，一个包含 625K 分子性质推理问题的数据集，专注于功能基团（functional group）级别的推理评估，通过三个维度（单功能基团影响、多功能基团交互、分子比较）系统揭示了当前 LLM 在细粒度化学推理能力上的严重不足。

**[Firegnn Neuro-Symbolic Graph Neural Networks With Trainable Fuzzy Rules For Inte](medical_imaging/firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)**

:   提出 FireGNN，首次将可训练模糊规则嵌入 GNN 前向传播中，利用节点度、聚类系数和标签一致性三个拓扑描述子实现内生可解释的医学图像分类，在 5 个 MedMNIST 数据集和 MorphoMNIST 上取得优于标准 GCN/GAT/GIN 及辅助任务方法的性能。

**[Flow Density Control Generative Optimization Beyond Entropy-Regularized Fine-Tun](medical_imaging/flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)**

:   提出 Flow Density Control（FDC），将预训练流/扩散模型的微调从 KL 正则期望奖励最大化推广到**任意分布效用函数 + 任意散度正则**的通用框架，通过将非线性目标分解为一系列线性微调子任务实现，并提供收敛保证。

**[Foxes A Framework For Operational X-Ray Emission Synthesis](medical_imaging/foxes_a_framework_for_operational_x-ray_emission_synthesis.md)**

:   提出 FOXES，一个基于 Vision Transformer 的框架，将太阳多通道 EUV 观测图像翻译为软 X 射线（SXR）通量，整体 Pearson 相关达到 0.982，为远端太阳耀斑检测和更完整的耀斑目录构建奠定基础。

**[Fractional Diffusion Bridge Models](medical_imaging/fractional_diffusion_bridge_models.md)**

:   提出分数扩散桥模型（FDBM），将分数布朗运动（fBM）引入生成扩散桥框架，通过 Hurst 指数 $H$ 控制轨迹的粗糙度和长程依赖性，在蛋白质构象预测和图像翻译任务上超越布朗运动基线。

**[From Black Box To Biomarker Sparse Autoencoders For Interpreting Speech Models O](medical_imaging/from_black_box_to_biomarker_sparse_autoencoders_for_interpreting_speech_models_o.md)**

:   将大语言模型可解释性研究中的稀疏自编码器（SAE）技术适配到语音帕金森病检测系统中，提出 Mask-based SAE 解决小数据集限制，发现模型预测主要基于低能量区域的频谱通量和频谱平坦度，并进一步揭示这些特征与 MRI 壳核体积显著相关，建立了从模型内部表征到临床生物标志物的桥梁。

**[Generalizable Real-Time Neural Decoding With Hybrid State-Space Models](medical_imaging/generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)**

:   POSSM 提出了一种混合 SSM-注意力架构，结合 spike 级别 tokenization 和循环状态空间模型骨干，实现了可泛化的实时神经解码，在保持与 Transformer 可比的精度的同时，推理速度提升最高 9 倍。

**[Generating Multi-Table Time Series Ehr From Latent Space With Minimal Preprocess](medical_imaging/generating_multi-table_time_series_ehr_from_latent_space_with_minimal_preprocess.md)**

:   提出 RawMed——首个以最小有损预处理合成多表时序 EHR 原始数据的框架：将事件文本化 → Residual Quantization 压缩至离散潜空间 → 自回归 Transformer 建模时序动态，在保真度、临床效用和隐私保护上全面超越现有基线。

**[Generative Distribution Embeddings Lifting Autoencoders To The Space Of Distribu](medical_imaging/generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)**

:   提出生成分布嵌入（GDE），将自编码器提升到分布空间——编码器作用于样本集合，解码器替换为条件生成模型，学习分布级别的表示，并在6个计算生物学任务上验证有效性。

**[Generative Modeling Of Full-Atom Protein Conformations Using Latent Diffusion On](medical_imaging/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)**

:   提出 **LD-FPG** 框架，使用 Chebyshev 图神经网络将蛋白质全原子 MD 轨迹编码到低维潜在空间，再用 DDPM 在该空间中生成新的构象集合体（ensemble），首次实现了包含侧链所有重原子的蛋白质构象生成。

**[Gflownets For Learning Better Drug-Drug Interaction Representations](medical_imaging/gflownets_for_learning_better_drug-drug_interaction_representations.md)**

:   针对药物-药物相互作用（DDI）预测中严重的类别不平衡问题，本文提出将 GFlowNet 与变分图自编码器（VGAE）结合，通过奖励引导的生成采样为稀有交互类型生成合成样本，从而增强模型在罕见但临床关键的交互类型上的预测能力。

**[H-Ddx A Hierarchical Evaluation Framework For Differential Diagnosis](medical_imaging/h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)**

:   H-DDx 提出基于 ICD-10 分类层级的鉴别诊断评估框架——将预测和真实诊断扩展到祖先节点后计算层级 F1（HDF1），奖励"临床相关的近似正确"而非仅精确匹配，评估 22 个 LLM 后发现领域特化模型（MediPhi）在 HDF1 上从第 20 名升至第 2 名（Top-5 指标完全遮蔽其优势）。

**[Imagenet-Trained Cnns Are Not Biased Towards Texture Revisiting Feature Reliance](medical_imaging/imagenet-trained_cnns_are_not_biased_towards_texture_revisiting_feature_reliance.md)**

:   通过系统化的特征抑制框架（而非冲突选择实验）重新评估 CNN 的特征依赖性，发现 CNN 并非天然偏向纹理，而是主要依赖局部形状特征；且不同领域（CV/MI/RS）的特征依赖模式显著不同。

**[Interpretable Next-Token Prediction Via The Generalized Induction Head](medical_imaging/interpretable_next-token_prediction_via_the_generalized_induction_head.md)**

:   提出 Induction-Gram (GIM)，一种结合精确n-gram匹配与模糊匹配的可解释语言模型，通过构建"广义归纳头"在输入上下文中检索相似序列进行下一token预测，比可解释基线提升最高25%p准确率，并在fMRI脑响应预测中提升20%。

**[Interpreting Gflownets For Drug Discovery Extracting Actionable Insights For Med](medical_imaging/interpreting_gflownets_for_drug_discovery_extracting_actionable_insights_for_med.md)**

:   为 SynFlowNet（基于合成反应模板的 GFlowNet）构建了一套多层次可解释性工具包，整合梯度显著性、反事实扰动、稀疏自编码器（SAE）和基序探针，揭示模型内部表征如何编码药物化学相关的理化性质和官能团信息。

**[Is Sequence Information All You Need For Bayesian Optimization Of Antibodies](medical_imaging/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)**

:   本文系统比较了序列信息和结构信息在抗体贝叶斯优化中的作用，发现通过蛋白质语言模型（pLM）软约束，纯序列方法可以匹配结构方法的性能，质疑了结构信息在抗体贝叶斯优化中的必要性。

**[Iterative Foundation Model Fine-Tuning On Multiple Rewards](medical_imaging/iterative_foundation_model_fine-tuning_on_multiple_rewards.md)**

:   提出 IterativeRS（迭代 Rewarded Soups），通过在多目标专家策略的独立微调和策略合并之间交替迭代，统一了奖励组合和专家合并两类方法，在小分子设计、DNA 序列生成和文本摘要任务上均优于 MORLHF 和 Rewarded Soups。

**[Jamun Bridging Smoothed Molecular Dynamics And Score-Based Learning For Conforma](medical_imaging/jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)**

:   提出 JAMUN，一种基于 Walk-Jump Sampling 框架的分子构象集成生成方法，通过在加噪的平滑流形上执行朗之万动力学并用 SE(3) 等变去噪器跳回原始分布，实现了比传统分子动力学快一个数量级的肽段构象采样，且具备对训练外系统的迁移能力。

**[Janusdna A Powerful Bi-Directional Hybrid Dna Foundation Model](medical_imaging/janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)**

:   提出JanusDNA，首个双向DNA基础模型，结合Mamba-Attention-MoE混合架构和Janus Modeling训练范式，以自回归的训练效率实现双向理解，在多个基因组基准上达到SOTA。

**[Large Language Models As Medical Codes Selectors A Benchmark Using The Internati](medical_imaging/large_language_models_as_medical_codes_selectors_a_benchmark_using_the_internati.md)**

:   构建了一个 extract-retrieve-select 框架的医学编码基准，在 33 个 LLM 上评估 ICPC-2 编码选择能力，发现 28 个模型 F1>0.8，证明 LLM 无需微调即可有效自动化初级保健编码。

**[Learning Conformational Ensembles Of Proteins Based On Backbone Geometry](medical_imaging/learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)**

:   提出 BBFlow，一种基于蛋白质骨架几何信息的流匹配生成模型，用于蛋白质构象集合采样，无需进化序列信息或预训练折叠模型，推理速度比 AlphaFlow 快一个数量级以上，且可扩展到多链蛋白质。

**[Learning Relative Gene Expression Trends From Pathology Images In Spatial Transc](medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)**

:   提出 STRank 损失函数，将病理图像基因表达估计重新定义为排序分数估计任务，利用二项分布/多项分布建模表达计数的随机噪声特性，使模型能从包含批次效应和随机波动的空间转录组数据中学习到鲁棒的相对表达关系。

**[Llm-Assisted Emergency Triage Benchmark Bridging Hospital-Rich And Mci-Like Fiel](medical_imaging/llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)**

:   基于MIMIC-IV-ED构建了一个开放的、LLM辅助策划的急诊分诊基准数据集，定义了医院丰富资源和大规模伤亡事件(MCI)模拟两种场景，提供基线模型和SHAP可解释性分析，推动分诊预测研究的可复现性和普及化。

**[Lomix Learnable Weighted Multi-Scale Logits Mixing For Medical Image Segmentatio](medical_imaging/lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)**

:   LoMix 提出通过组合突变模块（CMM）生成多尺度 logits 的"突变体"——4 种融合算子（加法/乘法/拼接/注意力加权）× 所有子集组合——配合 NAS 风格的 Softplus 可学习权重自动平衡各 logits 的贡献，在 Synapse 8 器官分割上 DICE 从 80.9% 提升到 85.1%（+4.2%），5% 训练数据下提升 +9.23%。

**[Magical Medical Lay Language Generation Via Semantic Invariance And Layperson-Ta](medical_imaging/magical_medical_lay_language_generation_via_semantic_invariance_and_layperson-ta.md)**

:   提出 Magical，一种面向医学通俗语言生成（MLLG）的非对称 LoRA 架构，通过共享矩阵 A 上的语义不变性约束和多个独立矩阵 B 实现语义保真与多样化通俗风格生成，在减少 31.66% 可训练参数的同时超越所有 LoRA 变体。

**[Mamba Goes Home Hierarchical Soft Mixture-Of-Experts For 3D Medical Image Segmen](medical_imaging/mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)**

:   提出Mamba-HoME架构，将层次化Soft MoE（HoME）与Mamba SSM结合，通过两级token路由机制实现局部-全局特征建模，在CT/MRI/US三种模态的3D医学图像分割任务上超越现有SOTA方法，同时保持线性计算复杂度。

**[Manipulating 3D Molecules In A Fixed-Dimensional E3-Equivariant Latent Space](medical_imaging/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)**

:   提出MolFLAE，一种学习固定维度、E(3)等变潜在空间的3D分子变分自编码器，通过引入可学习虚拟节点和贝叶斯流网络解码器，实现零样本分子编辑，包括原子数编辑、结构重构和性质插值，并在人类糖皮质激素受体（hGR）的药物优化中展示了实际应用价值。

**[Match Multi-Faceted Adaptive Topo-Consistency For Semi-Supervised Histopathology](medical_imaging/match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)**

:   提出MATCH框架，通过将拓扑推理与半监督学习的"扰动鲁棒性"原则紧密耦合，利用跨随机扰动和时间训练快照的双层拓扑一致性，自适应识别可靠拓扑结构而无需人工阈值，显著降低了组织病理学图像分割中的拓扑错误。

**[Medagentboard Benchmarking Multi-Agent Collaboration With Conventional Methods F](medical_imaging/medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)**

:   提出 MedAgentBoard，一个系统评估多智能体协作、单 LLM 和传统方法在多样化医学任务上表现的综合基准，揭示多智能体协作并不总是优于强单模型或专用传统方法。

**[Medmkg Benchmarking Medical Knowledge Exploitation With Multimodal Knowledge Gra](medical_imaging/medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)**

:   构建了一个融合MIMIC-CXR影像数据和UMLS临床概念的医学多模态知识图谱MedMKG，提出Neighbor-aware Filtering(NaF)图像筛选算法，并在链接预测、文本-图像检索和VQA三大任务上对24种基线方法进行了全面基准测试。

**[Meta-Learning An In-Context Transformer Model Of Human Higher Visual Cortex](medical_imaging/meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)**

:   提出BraInCoRL（Brain In-Context Representation Learning），一种基于Transformer的元学习框架，通过上下文学习（in-context learning）从少量刺激-响应样本直接预测新被试的体素级神经响应，无需微调即可适应新被试和新刺激，仅用100张图片就接近在9000张图片上完整训练的参考模型的性能。

**[Mind The Data Gap Evaluating Vision Systems In Small Data Applications](medical_imaging/mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)**

:   在 NeWT 生态分类基准上系统比较了 MLLMs（如 Gemini、Qwen2.5-VL）和视觉编码器+SVM 在"小数据区间"（10~1000 标注样本）的表现，发现 MLLMs 在 10-30 个样本后即触顶，而视觉方法持续近对数增长，呼吁社区重视小数据评估。

**[Mira Medical Time Series Foundation Model For Real-World Health Data](medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)**

:   提出 MIRA，一个专为医学不规则时间序列设计的基础模型，通过连续时间旋转位置编码、频率特定 MoE 和 Neural ODE 外推模块，在 4540 亿个观测点上预训练，零样本预测性能在 OOD 和 ID 场景中分别平均降低 8% 和 6% 的误差。

**[Modeling X-Ray Photon Pile-Up With A Normalizing Flow](medical_imaging/modeling_x-ray_photon_pile-up_with_a_normalizing_flow.md)**

:   提出基于Normalizing Flow的仿真推断(SBI)框架，通过CNN提取空间分辨的X射线光谱特征并输入神经样条流，实现在存在光子堆叠效应(pile-up)情况下对天体物理源参数的精确后验估计，显著优于传统PSF核心剪除方法。

**[Mol-Llama Towards General Understanding Of Molecules In Large Molecular Language](medical_imaging/mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)**

:   提出 Mol-LLaMA，一个面向分子通用理解的大型分子语言模型，通过设计三类关键指令数据类型和 2D-3D 分子表示融合模块，在分子特征理解上超越 GPT-4o，具备可解释性和推理能力。

**[More-Brain Routed Mixture Of Experts For Interpretable And Generalizable Cross-S](medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)**

:   提出 MoRE-Brain，一种神经科学启发的 fMRI 视觉解码框架，采用层级混合专家（MoE）架构模拟大脑视觉通路的专门化处理，配合动态时间-空间双路由机制引导扩散模型生成图像，在保持高保真重建的同时实现了高效跨被试泛化和前所未有的机制可解释性。

**[Mtbbench A Multimodal Sequential Clinical Decision-Making Benchmark In Oncology](medical_imaging/mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)**

:   提出MTBBench——首个同时覆盖多模态、纵向时序和交互式Agent工作流三个维度的临床基准，模拟分子肿瘤委员会（MTB）的决策流程，评估并增强AI Agent在肿瘤学精准医疗中的多模态纵向推理能力。

**[Multi-Objective Reinforcement Learning With Max-Min Criterion A Game-Theoretic A](medical_imaging/multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)**

:   将max-min多目标强化学习重新表述为两人零和正则化连续博弈，提出ERAM/ARAM算法，利用镜像下降实现简洁的闭式权重更新，保证全局最后迭代收敛，在交通信号控制等任务中显著优于已有方法。

**[Multimodal 3D Genome Pre-Training](medical_imaging/multimodal_3d_genome_pre-training.md)**

:   提出MIX-HIC——首个面向3D基因组的多模态基础模型，通过跨模态交互块和跨模态映射块融合Hi-C接触图和表观基因组信号，在超过127万对样本上预训练，在Hi-C预测、染色质环检测和CAGE-seq表达预测三个下游任务上全面超越SOTA。

**[Multimodal Bayesian Network For Robust Assessment Of Casualties In Autonomous Tr](medical_imaging/multimodal_bayesian_network_for_robust_assessment_of_casualties_in_autonomous_tr.md)**

:   提出基于专家知识驱动的贝叶斯网络决策支持框架，融合多个计算机视觉模型的输出来评估伤亡人员状况，无需训练数据且支持不完整信息推断，在DARPA Triage Challenge中将分诊准确率从14%提升至53%，诊断覆盖率从31%提升至95%。

**[Multimodal Disease Progression Modeling Via Spatiotemporal Disentanglement And M](medical_imaging/multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)**

:   提出 DiPro 框架，通过区域感知的时空解耦（分离静态解剖与动态病理特征）和多时间尺度对齐（局部-全局融合 CXR 与 EHR），解决了纵向胸部X光序列的冗余问题和跨模态时间错位挑战，在疾病进展识别和 ICU 预测任务上达到 SOTA。

**[Multiscale Guidance Of Protein Structure Prediction With Heterogeneous Cryo-Em D](medical_imaging/multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)**

:   CryoBoltz利用冷冻电镜（cryo-EM）密度图通过多尺度引导机制（全局→局部）引导预训练扩散结构预测模型（Boltz-1）的采样轨迹，无需重新训练即可生成与实验数据一致的多构象原子模型。

**[Muslr Multimodal Symbolic Logical Reasoning](medical_imaging/muslr_multimodal_symbolic_logical_reasoning.md)**

:   提出首个多模态符号逻辑推理任务MuSLR及其基准测试集MuSLR-Bench（1,093个实例，涵盖7个领域、35种原子符号逻辑、推理深度2-9），并设计模块化框架LogiCAM，通过前提选择、推理类型识别和符号推理三个模块将GPT-4.1的CoT性能提升14.13%。

**[Neuript Foundation Model For Neural Interfaces](medical_imaging/neuript_foundation_model_for_neural_interfaces.md)**

:   NeurIPT是一个面向多样化脑机接口(BCI)应用的EEG基础模型，通过振幅感知掩码预训练(AAMP)、渐进式专家混合(PMoE)架构、3D电极空间编码和脑叶内/跨脑叶池化(IILP)四大创新设计，在八个下游BCI任务上实现了SOTA性能。

**[One Small Step With Fingerprints One Giant Leap For De Novo Molecule Generation ](medical_imaging/one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)**

:   通过将 MIST 作为质谱-指纹编码器、MolForge 作为指纹-结构解码器，并采用先验调整阈值策略，在 MassSpecGym 基准上实现了从质谱从头生成分子结构的十倍性能提升（top-1 准确率从 2.3% 提升至 31%）。

**[Online Feedback Efficient Active Target Discovery In Partially Observable Enviro](medical_imaging/online_feedback_efficient_active_target_discovery_in_partially_observable_enviro.md)**

:   提出 DiffATD，利用扩散模型的逆向过程构建 belief 分布来平衡探索与利用，在部分可观测环境中无需任何监督训练即可高效发现目标区域，适用于医学影像、物种发现和遥感等多领域。

**[Ordinal Label-Distribution Learning With Constrained Asymmetric Priors For Imbal](medical_imaging/ordinal_label-distribution_learning_with_constrained_asymmetric_priors_for_imbal.md)**

:   提出 CAP-WAE（Constrained Asymmetric Prior Wasserstein Autoencoder），通过非对称先验、序数边距正交紧凑损失和方向感知序数损失三重创新，解决糖尿病视网膜病变分级中长尾分布和序数结构的挑战，在多个 DR 基准上达到 SOTA。

**[Orochi Versatile Biomedical Image Processor](medical_imaging/orochi_versatile_biomedical_image_processor.md)**

:   提出 Orochi——首个面向底层生物医学图像处理的通用基础模型，通过任务相关联合嵌入预训练（TJP）和多头层级 Mamba 架构，在配准、融合、复原和超分辨率四大任务上以轻量微调（<5% 参数）即可达到或超越专用 SOTA 模型。

**[Pancakes Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](medical_imaging/pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)**

:   提出 Pancakes 框架，给定来自未见过领域的生物医学图像集合，自动生成多个合理分割协议（protocol）的标签图，且同一协议下不同图像的标签具有**语义一致性**——同一标签在所有图像中指代相同的解剖结构。

**[Patientsim A Persona-Driven Simulator For Realistic Doctor-Patient Interactions](medical_imaging/patientsim_a_persona-driven_simulator_for_realistic_doctor-patient_interactions.md)**

:   提出PatientSim——基于真实MIMIC临床数据和四维人格轴（性格、语言能力、病史记忆水平、认知混乱程度）的LLM患者模拟器，生成37种独特人格组合，在8个LLM上评估事实准确性和人格一致性，由4名临床专家验证平均质量得分3.89/4。

**[Pharmacophore-Guided Generative Design Of Novel Drug-Like Molecules](medical_imaging/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)**

:   提出一种药效团引导的分子生成框架，在强化学习模型（FREED++）的奖励函数中同时最大化药效团相似度和最小化结构相似度，生成既保留生物活性特征又具有高结构新颖性的候选药物分子。

**[Physiowave A Multi-Scale Wavelet-Transformer For Physiological Signal Representa](medical_imaging/physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)**

:   提出 PhysioWave，一种基于可学习小波分解和频率引导掩码的多尺度 Transformer 架构，首次为 EMG 和 ECG 构建大规模预训练基础模型，并通过多模态融合框架在单模态和多模态生理信号任务上取得 SOTA 性能。

**[Polypose Deformable 2D3D Registration Via Polyrigid Transformations](medical_imaging/polypose_deformable_2d3d_registration_via_polyrigid_transformations.md)**

:   提出PolyPose，一种基于多刚体变换（polyrigid）的可变形2D/3D配准方法，利用"骨骼是刚体"这一解剖学先验，将复杂3D形变场参数化为多个刚体变换在切空间 $\mathfrak{se}(3)$ 中的加权组合，无需正则化和超参数调优即可从少至两张X光片实现精确的3D体积配准。

**[Position Thematic Analysis Of Unstructured Clinical Transcripts With Large Langu](medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)**

:   这篇立场论文系统综述了LLM在非结构化临床转录文本主题分析中的应用现状，发现评估方法高度碎片化，并提出以有效性(Validity)、可靠性(Reliability)、可解释性(Interpretability)三维度为核心的标准化评估框架。

**[Posterior Sampling By Combining Diffusion Models With Annealed Langevin Dynamics](medical_imaging/posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)**

:   提出将扩散模型与退火 Langevin 动力学结合的算法，仅需 $L^4$ 精度的 score 估计即可在（局部）对数凹分布下实现多项式时间的后验采样，首次为带暖启动的逆问题求解提供理论保障。

**[Prior-Guided Flow Matching For Target-Aware Molecule Design With Learnable Atom ](medical_imaging/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)**

:   提出 PAFlow，基于流匹配框架的 3D 分子生成模型，通过蛋白-配体交互预测器引导向量场和可学习原子数预测器，在 CrossDocked2020 上实现 -8.31 Avg. Vina Score 的新 SOTA，大幅超越已有方法。

**[Prospero Active Learning For Robust Protein Design Beyond Wild-Type Neighborhood](medical_imaging/prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)**

:   提出 ProSpero，一个主动学习框架，通过冻结的预训练生成模型（EvoDiff）在代理模型引导下的推理时采样、针对性掩码策略和生物约束的 SMC 采样，在代理模型可能失配的条件下仍能发现高适应性且新颖的蛋白质序列。

**[Protein Design With Dynamic Protein Vocabulary](medical_imaging/protein_design_with_dynamic_protein_vocabulary.md)**

:   提出 ProDVa 方法，将天然蛋白质片段作为"动态词汇"引入生成式蛋白质设计，通过文本编码器+蛋白质语言模型+片段编码器的三组件架构，利用不到 0.04% 的训练数据即可设计出功能对齐且结构可折叠的蛋白质序列，在 pLDDT>70 比例上超越 SOTA 模型 Pinal 达 7.38%。

**[Qoq-Med Building Multimodal Clinical Foundation Models With Domain-Aware Grpo Tr](medical_imaging/qoq-med_building_multimodal_clinical_foundation_models_with_domain-aware_grpo_tr.md)**

:   QoQ-Med 构建了覆盖 9 个临床模态（1D ECG + 6 类 2D 影像 + 2 类 3D 扫描）的多模态临床基础模型，提出域感知相对策略优化（DRPO）——通过层级温度缩放（域间 × 域内 K-means 聚类）解决模态/难度不平衡问题，在 261 万指令调优对上训练后平均 F1 达 0.295（vs GRPO 0.193，+52.8%），8 个模态中 6 个最优。

**[Quantifying The Role Of Openfold Components In Protein Structure Prediction](medical_imaging/quantifying_the_role_of_openfold_components_in_protein_structure_prediction.md)**

:   本文提出系统方法评估 OpenFold/AlphaFold2 中 Evoformer 各组件对蛋白质结构预测精度的贡献，发现 MSA 列注意力和 MLP Transition 层是最关键的组件，且多个组件的重要性与蛋白质序列长度显著相关。

**[Rad Towards Trustworthy Retrieval-Augmented Multi-Modal Clinical Diagnosis](medical_imaging/rad_towards_trustworthy_retrieval-augmented_multi-modal_clinical_diagnosis.md)**

:   提出检索增强诊断框架RAD，通过从多源医学语料中检索疾病指南并注入多模态模型的特征提取和跨模态融合全流程，同时引入双轴可解释性评估体系，在四个不同解剖部位的数据集上达到SOTA。

**[Radzero Similarity-Based Cross-Attention For Explainable Vision-Language Alignme](medical_imaging/radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)**

:   提出 RadZero 框架及核心组件 VL-CABS（基于相似度的视觉语言交叉注意力），在胸部X光上实现可解释的、细粒度的视觉语言对齐，支持零样本分类、定位和分割多任务。

**[Ram-W600 A Multi-Task Wrist Dataset And Benchmark For Rheumatoid Arthritis](medical_imaging/ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)**

:   首个公开的多任务腕骨常规X光数据集RAM-W600，包含1048张影像，支持腕骨实例分割和SvdH骨侵蚀评分两大任务，并提供全面的基准测试。

**[Random Search Neural Networks For Efficient And Expressive Graph Learning](medical_imaging/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)**

:   提出随机搜索神经网络（RSNN），用随机深度优先搜索（DFS）替代随机游走来采样图结构，在稀疏图上仅需$O(\log|V|)$次搜索即可实现完整边覆盖，配合通用序列模型可达到通用逼近能力，在分子和蛋白质基准上以最多16倍更少的采样量持续超越RWNN。

**[Raxss Retrieval-Augmented Sparse Sampling For Explainable Variable-Length Medica](medical_imaging/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)**

:   提出RAxSS框架，将检索增强机制引入随机稀疏采样(SSS)流水线，通过窗口内相似度加权聚合替代均匀平均，在保持变长医学时间序列分类性能的同时提供从"哪里"到"为什么"的可解释性证据链。

**[Revisiting End-To-End Learning With Slide-Level Supervision In Computational Pat](medical_imaging/revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)**

:   重新审视计算病理中切片级监督的端到端(E2E)学习，首次揭示稀疏注意力MIL在E2E训练中导致的优化困难，提出ABMILX通过多头注意力和全局注意力校正模块解决该问题，使E2E训练的ResNet在多个基准上超越SOTA基础模型。

**[Riemannian Flow Matching For Brain Connectivity Matrices Via Pullback Geometry](medical_imaging/riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)**

:   提出DiffeoCFM，利用全局微分同胚诱导的拉回度量，将黎曼流形上的条件流匹配等价转化为欧几里得空间中的标准CFM，实现对脑连接矩阵（SPD/相关矩阵）的高效生成，同时严格保持流形约束，在3个fMRI和2个EEG数据集上达到SOTA。

**[Robust Or Suggestible Exploring Non-Clinical Induction In Llm Drug-Safety Decisi](medical_imaging/robust_or_suggestible_exploring_non-clinical_induction_in_llm_drug-safety_decisi.md)**

:   通过基于Persona的评估框架发现，ChatGPT-4o和Bio-Medical-Llama-3-8B在药物不良事件预测中会受到临床无关的社会人口属性（教育、保险、住房等）系统性影响，展现出显式和隐式两种偏差模式。

**[Scalable Diffusion Transformer For Conditional 4D Fmri Synthesis](medical_imaging/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)**

:   提出首个用于体素级全脑4D fMRI条件生成的扩散Transformer，结合3D VQ-GAN潜空间压缩、CNN-Transformer混合骨干网络和AdaLN-Zero+交叉注意力的强条件注入，在HCP七种认知任务上实现任务激活图相关0.83、RSA达0.98和完美条件特异性。

**[Scaling Laws And Pathologies Of Single-Layer Pinns Network Width And Pde Nonline](medical_imaging/scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)**

:   对单层PINN在典型非线性PDE上建立了经验缩放定律，发现了双重优化失败：宽度缩放病理（误差不随宽度下降）和复合病理（非线性加剧此失败），证明优化而非近似容量是主要瓶颈。

**[Securing The Language Of Life Inheritable Watermarks From Dna Language Models To](medical_imaging/securing_the_language_of_life_inheritable_watermarks_from_dna_language_models_to.md)**

:   提出 DNAMark 和 CentralMark 两种水印方案，针对 DNA 语言模型生成的序列嵌入鲁棒水印：前者利用同义密码子替换实现功能不变水印，后者实现从 DNA 到蛋白质的可遗传水印。

**[Self-Supervised Learning Of Echocardiographic Video Representations Via Online C](medical_imaging/self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)**

:   提出 DISCOVR，一种自监督双分支框架，通过在线语义聚类蒸馏将图像编码器的细粒度空间语义传递到视频编码器的时序表示中，在六个跨胎儿/儿科/成人心脏超声数据集上实现了异常检测、分类和分割的全面领先。

**[Self-Supervised Learning Via Flow-Guided Neural Operator On Time-Series Data](medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)**

:   提出 FGNO（Flow-Guided Neural Operator），将 Flow Matching 与算子学习结合用于时间序列自监督预训练，通过 STFT 实现分辨率不变的函数空间学习，并将流时间（flow time）和网络层作为控制特征粒度的"旋钮"，在生物医学任务上显著优于 MAE 等基线。

**[Self Iterative Label Refinement Via Robust Unlabeled Learning](medical_imaging/self_iterative_label_refinement_via_robust_unlabeled_learning.md)**

:   提出一种迭代式管道方法，利用鲁棒的无标签-无标签（UU）学习框架来精炼LLM生成的伪标签，仅需极少人工标注即可在分类和生成式安全对齐任务中超越GPT-4o和DeepSeek-R1的自我精炼方法。

**[Semantic And Visual Crop-Guided Diffusion Models For Heterogeneous Tissue Synthe](medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)**

:   提出 HeteroTissue-Diffuse（HTD），一种双条件 Latent Diffusion 模型，通过同时以语义分割图和真实组织裁剪块（visual crop）作为条件来生成异质性病理图像，在 Camelyon16 上将 Fréchet Distance 从 430 降至 72（6 倍改善），合成数据训练的 DeepLabv3+ 分割 IoU 与真实数据仅差 1-2%，并通过自监督聚类扩展到 11765 张无标注 TCGA 全幻灯片图像。

**[Sequential Attention-Based Sampling For Histopathological Analysis](medical_imaging/sequential_attention-based_sampling_for_histopathological_analysis.md)**

:   提出 SASHA 框架，结合层次注意力多实例学习 (HAFED) 与深度强化学习 (RL)，仅采样 10-20% 的高分辨率 patch 即可达到全分辨率 SOTA 方法的分类性能，推理速度提升 4-8 倍，WSI 压缩率超 16 倍。

**[Shallow Robustness Deep Vulnerabilities Multi-Turn Evaluation Of Medical Llms](medical_imaging/shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)**

:   提出MedQA-Followup框架系统评估医学LLM的多轮鲁棒性，发现模型在单轮扰动下表现尚可（浅层鲁棒性），但在多轮追问中准确率可从91.2%暴跌至13.5%（深层脆弱性），且间接上下文操纵比直接错误建议更具破坏力。

**[Smmile An Expert-Driven Benchmark For Multimodal Medical In-Context Learning](medical_imaging/smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)**

:   提出 SMMILE——首个由 11 位医学专家驱动的多模态医学上下文学习（ICL）基准，包含 111 道问题（517 个图文问答三元组）覆盖 6 个医学专科和 13 种成像模态，系统性揭示了当前 MLLM 在医学多模态 ICL 上的严重不足以及上下文示例质量和顺序对性能的关键影响。

**[Specmer Fast Protein Generation With K-Mer Guided Speculative Decoding](medical_imaging/specmer_fast_protein_generation_with_k-mer_guided_speculative_decoding.md)**

:   SpecMER 将投机解码引入蛋白质序列生成，用 K-mer 引导的批量选择策略从 draft 模型的多个候选中选取最符合进化保守性的序列供 target 模型验证，在保持分布一致性的同时实现 24-32% 加速，且生成序列的 NLL 和 pLDDT 结构置信度显著优于无引导的 baseline。

**[Stamp Spatial-Temporal Adapter With Multi-Head Pooling](medical_imaging/stamp_spatial-temporal_adapter_with_multi-head_pooling.md)**

:   STAMP 为时间序列基础模型（TSFM）设计了仅 750K 参数的轻量空间-时间适配器，通过三组位置编码（token/空间/时间）+ 交叉 GMLP 混合 + 多头注意力池化，使冻结的 TSFM（如 MOMENT 385M）在 8 个 EEG 数据集上与 29M 参数的 EEG 专用模型（CBraMod）竞争或超越，在 BCIC-IV-2a 上 Kappa 比 CBraMod 高 193%。

**[Starc-9 A Large-Scale Dataset For Multi-Class Tissue Classification For Crc Hist](medical_imaging/starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)**

:   提出 STARC-9 大规模结直肠癌组织分类数据集（63 万张图片、9 类组织）及其构建框架 DeepCluster++，通过自编码器特征提取 + K-means 聚类 + 等频分箱采样确保形态多样性，在该数据集上训练的模型显著超越 NCT 和 HMU 训练的模型。

**[Steering Generative Models With Experimental Data For Protein Fitness Optimizati](medical_imaging/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)**

:   系统性地评估了引导蛋白质生成模型（离散扩散模型和语言模型）进行适应度优化的各种策略，发现使用少量标注数据（~200条）的即插即用引导方法（特别是 DAPS）优于基于 RL 的微调方法，并提出了集成不确定性的 Thompson 采样策略用于自适应优化。

**[Surf2Ct Cascaded 3D Flow Matching Models For Torso 3D Ct Synthesis From Skin Sur](medical_imaging/surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)**

:   提出 Surf2CT，一种级联式 3D Flow Matching 框架，首次实现仅从外部体表扫描和人口学数据（年龄、性别、身高、体重）合成完整的高分辨率 3D CT 体积，无需任何内部成像输入。

**[Synbrain Enhancing Visual-To-Fmri Synthesis Via Probabilistic Representation Lea](medical_imaging/synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)**

:   提出 SynBrain 框架，通过 BrainVAE 将 fMRI 响应建模为视觉语义条件的概率分布，并用 S2N Mapper 实现一步式语义到神经空间的映射，在视觉-fMRI 合成任务上显著超越 MindSimulator（MSE 降低 65%，Pearson 提升 96%），且合成的 fMRI 可有效增强少样本跨被试解码性能。

**[The Biased Oracle Assessing Llms Understandability And Empathy In Medical Diagno](medical_imaging/the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)**

:   系统评估 GPT-4o 和 Claude-3.7 在医疗诊断沟通中的可读性和共情能力，发现两者均产生超标的阅读难度（9-13 年级 vs 推荐的 6-8 年级），情感共情随诊断类型和患者教育水平显著变化，且 LLM-as-Judge 存在严重自我偏见（GPT 对自身共情评分膨胀 ~0.3 分）。

**[The Boundaries Of Fair Ai In Medical Image Prognosis A Causal Perspective](medical_imaging/the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)**

:   FairTTE是首个系统研究医学影像中时间-事件(TTE)预测公平性的综合框架，利用因果分析量化五种偏差来源，通过训练超过20000个模型揭示了现有公平性方法的局限性，特别是在分布偏移下公平性难以维持的根本挑战。

**[The Human Brain As A Combinatorial Complex](medical_imaging/the_human_brain_as_a_combinatorial_complex.md)**

:   提出一种数据驱动的框架，利用 S-信息和 O-信息等信息论度量从 fMRI 时间序列中直接构建组合复形（Combinatorial Complexes），将脑区间的高阶协同交互编码到拓扑结构中，为拓扑深度学习应用于脑网络分析奠定基础。

**[Thunder Tile-Level Histopathology Image Understanding Benchmark](medical_imaging/thunder_tile-level_histopathology_image_understanding_benchmark.md)**

:   提出 THUNDER，一个面向数字病理学基础模型的 tile 级别综合基准，支持 23 个基础模型在 16 个数据集上的高效比较，覆盖下游任务性能、特征空间分析、鲁棒性和不确定性评估。

**[Toward A Vision-Language Foundation Model For Medical Data Multimodal Dataset An](medical_imaging/toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)**

:   构建首个越南语 PET/CT 图像-报告数据集 ViMed-PET（2,757 例全身 PET/CT 体积 + 完整临床报告），通过数据增强策略和三阶段微调流程显著提升 VLM 在医学报告生成和 VQA 任务上的表现，并提出基于临床关键信息的评估指标。

**[Towards Multiscale Graph-Based Protein Learning With Geometric Secondary Structu](medical_imaging/towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)**

:   提出SSHG（Secondary Structure-based Hierarchical Graph）框架，基于蛋白质二级结构motif构建两级层次化图表示（残基级内部图+motif级全局图），用两阶段GNN分别学习局部和全局特征，理论证明保持最大表达力的同时在酶分类和配体亲和力预测上同时提升精度和降低计算成本。

**[Towards Self-Supervised Foundation Models For Critical Care Time Series](medical_imaging/towards_self-supervised_foundation_models_for_critical_care_time_series.md)**

:   基于双轴Transformer（BAT）架构，在多个ICU数据集上进行自监督预训练，构建重症监护时间序列基础模型，在小数据集场景下显著优于监督学习基线。

**[Towards Unified And Lossless Latent Space For 3D Molecular Latent Diffusion Mode](medical_imaging/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)**

:   提出 UAE-3D，一种多模态变分自编码器，将3D分子的原子类型、化学键和3D坐标压缩到统一的近无损潜在空间中，消除了处理多模态和等变性的复杂性，使通用 Diffusion Transformer 即可实现 SOTA 的3D分子生成。

**[Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models](medical_imaging/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)**

:   提出不确定性感知的多目标强化学习框架，引导 3D 分子扩散模型（EDM）同时优化药物相关性（QED）、合成可及性（SAS）和结合亲和力（binding affinity），通过代理模型的预测不确定性动态塑造奖励函数，在三个基准数据集上一致超越基线，并通过分子动力学模拟和 ADMET 验证候选分子的药物潜力。

**[Unified All-Atom Molecule Generation With Neural Fields](medical_imaging/unified_all-atom_molecule_generation_with_neural_fields.md)**

:   提出 FuncBind 框架，利用神经场（Neural Fields）将分子表示为连续原子密度函数，构建统一的条件生成模型，能够同时处理小分子、大环肽和抗体 CDR 环三种药物模态的靶标条件生成。

**[Unimrseg Unified Modality-Relax Segmentation Via Hierarchical Self-Supervised Co](medical_imaging/unimrseg_unified_modality-relax_segmentation_via_hierarchical_self-supervised_co.md)**

:   提出UniMRSeg，一种统一的模态缺失分割框架，通过层次化自监督补偿机制（HSSC）——从输入级模态重建、特征级对比学习到输出级一致性约束——用100%共享参数在所有可能的模态组合下实现最优平均性能和最小性能波动。

**[Unisite The First Cross-Structure Dataset And Learning Framework For End-To-End ](medical_imaging/unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)**

:   提出首个以UniProt（唯一蛋白质）为中心的配体结合位点数据集UniSite-DS，以及首个端到端的结合位点检测框架UniSite，通过集合预测损失和双射匹配直接预测多个可能重叠的结合位点，同时引入IoU-based AP作为更准确的评估指标。

**[Unlearned But Not Forgotten Data Extraction After Exact Unlearning In Llm](medical_imaging/unlearned_but_not_forgotten_data_extraction_after_exact_unlearning_in_llm.md)**

:   揭示了即使精确遗忘（从头重训练去除数据影响）也存在隐私泄露风险：攻击者利用遗忘前后两个模型检查点的差异，通过逆向模型引导和 token 过滤策略，可显著提升已删除数据的提取成功率，在某些场景下提取率翻倍。

**[Unpaired Image-To-Image Translation For Segmentation And Signal Unmixing](medical_imaging/unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)**

:   提出 Ui2i 模型，在 CycleGAN 基础上通过 UNet 生成器、近似双向谱归一化替代特征归一化、通道-空间注意力和尺度增强，实现高内容保真度的无配对图像翻译，成功用于 IHC→H&E 域适应核分割及单通道免疫荧光信号解混两大生物医学任务。

**[Variational Autoencoder With Normalizing Flow For X-Ray Spectral Fitting](medical_imaging/variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)**

:   将归一化流 (NF) 嵌入自编码器架构中，对黑洞 X 射线双星的 NICER 光谱数据进行快速物理参数推断和完整后验分布估计，比传统 MCMC 方法快约 2000 倍，且精度可比拟。

**[Vq-Seg Vector-Quantized Token Perturbation For Semi-Supervised Medical Image Seg](medical_imaging/vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)**

:   提出 VQ-Seg，首次将向量量化引入半监督医学图像分割，用量化扰动模块（QPM）替代传统 dropout 实现更可控的特征扰动，并结合双分支架构和基础模型引导对齐来弥补量化信息损失。

**[Why Masking Diffusion Works Condition On The Jump Schedule For Improved Discrete](medical_imaging/why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)**

:   揭示了掩码扩散模型优越性的根本原因——它内建了已知的跳转时间分布，由此提出Schedule-Conditioned Diffusion (SCUD)框架，将此优势推广到任何离散扩散模型，结合结构化前向过程在图像和蛋白质数据上超越掩码扩散。

**[Zebra Towards Zero-Shot Cross-Subject Generalization For Universal Brain Visual ](medical_imaging/zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)**

:   提出 Zebra，首个零样本脑视觉解码框架，通过对抗训练与残差分解将 fMRI 表征解耦为主体不变和语义特定成分，无需对新被试做微调即可实现跨被试的视觉重建泛化。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[A Frustratingly Simple Yet Highly Effective Attack Baseline](multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)**

:   提出 M-Attack，通过对源图像做随机裁剪后与目标图像在嵌入空间做局部-全局/局部-局部匹配，配合多 CLIP 模型集成，使对抗扰动自然聚集在语义关键区域形成清晰的语义细节，在 GPT-4.5/4o/o1 等商业黑盒 LVLM 上实现 >90% 的定向攻击成功率。

**[A Multimodal Benchmark For Framing Of Oil Gas Advertising An](multimodal_vlm/a_multimodal_benchmark_for_framing_of_oil_gas_advertising_an.md)**

:   构建了首个面向石油天然气行业视频广告的多模态框架分析基准数据集（706 个视频、13 种框架类型、50+ 实体、20 个国家），系统评估了 6 款 VLM 在检测 greenwashing 相关 framing 中的能力，发现 GPT-4.1 零样本在环境类标签上达 79% F1 但绿色创新仅 46%，揭示了隐式框架分析和文化背景理解仍是 VLM 的核心挑战。

**[Act As Human Multimodal Large Language Model Data Annotation](multimodal_vlm/act_as_human_multimodal_large_language_model_data_annotation.md)**

:   提出ACT（Annotation with Critical Thinking）数据流水线，MLLM批量标注全部数据后由另一个MLLM作为批评者估计每条标注的错误概率，仅将高可疑样本交给人类审核，配合理论推导的ACT损失函数，在6个跨模态数据集上节省70-90%人工成本且下游性能差距<2%。

**[Adalrs Lossguided Adaptive Learning Rate Search For Efficien](multimodal_vlm/adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)**

:   提出AdaLRS，一种即插即用的在线学习率搜索算法，通过监控损失下降速度（loss velocity）来自适应调整学习率，将学习率超参搜索的成本从多次独立训练降低到单次训练，实现~50%的训练成本节省。

**[Adapting Visionlanguage Models For Evaluating World Models](multimodal_vlm/adapting_visionlanguage_models_for_evaluating_world_models.md)**

:   提出 UNIVERSE（UNIfied Vision-language Evaluator for Rollouts in Simulated Environments），通过对 PaliGemma 2 进行轻量级投影头微调（仅 0.07% 参数），构建统一的世界模型 rollout 语义评估器，在动作识别和角色识别任务上达到与任务专属模型相当的性能并与人类判断高度对齐。

**[Admn A Layerwise Adaptive Multimodal Network For Dynamic Inp](multimodal_vlm/admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)**

:   提出 ADMN（Adaptive Depth Multimodal Network），通过两阶段训练——(1) Multimodal LayerDrop 微调使 backbone 适应任意层配置，(2) QoI感知控制器动态分配层预算给各模态——在严格计算约束下根据每个模态的信息质量(QoI)自适应分配层数，匹配全量模型精度同时减少 75% FLOPs 和 60% 延迟。

**[Advancing Compositional Awareness In Clip With Efficient Fin](multimodal_vlm/advancing_compositional_awareness_in_clip_with_efficient_fin.md)**

:   提出 CLIC，通过拼接两张图像并基于跨图词汇交换生成 hard negatives，同时创建多个正样本描述，仅微调 CLIP 文本编码器就能同时提升组合推理能力（SugarCrepe++ SOTA）和下游检索性能，打破了之前方法中组合性与检索性不可兼得的困局。

**[Affordbot 3D Fine-Grained Embodied Reasoning Via Multimodal Large Language Model](multimodal_vlm/affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)**

:   提出细粒度 3D 具身推理任务（预测可操作元素的空间位置+运动类型+运动轴），通过将 3D 点云渲染为环视图并投影 affordance 候选，结合定制的 CoT 推理范式指导 MLLM 实现 SOTA，AP25 达 23.3%。

**[Aligning By Misaligning Boundaryaware Curriculum Learning Fo](multimodal_vlm/aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)**

:   提出 BACL（Boundary-Aware Curriculum with Local Attention），通过可学习的边界感知负样本采样器（由易到难课程学习）+ 对比局部注意力损失（定位 token 级 mismatch），在 LAION-400M 上为 CLIP 带来 +32% R@1 提升，并在四个大规模基准上取得 SOTA。

**[Antigrounding Lifting Robotic Actions Into Vlm Representatio](multimodal_vlm/antigrounding_lifting_robotic_actions_into_vlm_representatio.md)**

:   反转传统指令接地范式——不将 VLM 知识压缩到中间表征（符号技能或约束），而是将候选机器人轨迹渲染到多视角场景图像中，直接在 VLM 的原生高维表征空间中评估动作方案，实现零样本闭环机器人操作控制。

**[Approximate Domain Unlearning For Visionlanguage Models](multimodal_vlm/approximate_domain_unlearning_for_visionlanguage_models.md)**

:   提出 Approximate Domain Unlearning (ADU) 新任务，通过 Domain Disentangling Loss (DDL) 和 Instance-wise Prompt Generator (InstaPG) 两个模块，让预训练 VLM 选择性遗忘指定域（如插画、素描）的识别能力，同时保持其他域（如真实照片）的分类精度，在四个多域数据集上大幅超越所有基线。

**[Aquamam An Autoregressive Quaternion Manifold Model For Rapidly Estimating Compl](multimodal_vlm/aquamam_an_autoregressive_quaternion_manifold_model_for_rapidly_estimating_compl.md)**

:   提出AQuaMaM——一种基于Transformer的自回归四元数流形模型，将单位四元数的三个投影分量建模为受几何约束的均匀分布混合，在SO(3)旋转流形上实现精确似然计算和快速采样，比IPDF推理速度快52倍、对数似然高14%，且采样分布与真实分布匹配极为精确。

**[Are Vision Language Models Ready For Clinical Diagnosis A 3D Medical Benchmark F](multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)**

:   本文提出 DeepTumorVQA，一个针对腹部CT肿瘤的3D诊断级视觉问答基准，包含9,262个CT体积（370万切片）和395K专家级问题，系统评估了4个先进VLM的临床诊断能力，发现当前模型在测量任务上尚可但在病灶识别和推理上远未达到临床要求。

**[Attention Your Vision Language Model Could Be Maliciously Manipulated](multimodal_vlm/attention_your_vision_language_model_could_be_maliciously_manipulated.md)**

:   本文提出 Vision-language Model Manipulation Attack (VMA)，一种结合一阶和二阶动量优化及可微变换机制的图像对抗攻击方法，能够精确操控VLM的每个输出token，可用于实施多种攻击（越狱、劫持、隐私泄露、DoS、海绵样本）同时也可用于版权保护水印注入。

**[Balanced Token Pruning Accelerating Vision Language Models B](multimodal_vlm/balanced_token_pruning_accelerating_vision_language_models_b.md)**

:   提出 Balanced Token Pruning (BTP)，通过联合考虑剪枝对当前层（局部）和后续层（全局）的影响，在浅层侧重多样性保留以维护下游表示质量、在深层侧重注意力选择以保持局部输出一致性，在 LLaVA/Qwen2.5-VL 等多个 LVLM 上仅保留 22% 视觉 token 即保持原模型 98% 性能。

**[Better Tokens For Better 3D Advancing Vision-Language Modeling In 3D Medical Ima](multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)**

:   提出 BTB3D，一种基于因果卷积编解码器 + 3D Haar 小波压缩 + 三阶段渐进训练的 3D CT tokenizer，在放射报告生成和文本条件 CT 合成两大下游任务上大幅刷新 SOTA，证明"更好的 token 比更大的语言模型更重要"。

**[Beyond Greedy Exits Improved Early Exit Decisions For Risk Control And Reliabili](multimodal_vlm/beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)**

:   UAT（Unsupervised Adaptive Thresholding）为早退 DNN 设计了可靠性函数来评估中间层输出质量，并用多臂赌博机（MAB）算法在推理时动态学习最优退出阈值，实现 1.7-2.1× 加速且性能损失 <2%，同时对分布偏移鲁棒。

**[Bioclip 2 Emergent Properties From Scaling Hierarchical Contrastive Learning](multimodal_vlm/bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)**

:   BioCLIP 2 在 TreeOfLife-200M（2.14 亿图像/95.2 万物种）上用层级对比学习训练 ViT-L，零样本物种识别比 BioCLIP 提升 18%，并发现规模化带来的涌现性质——嵌入自动编码生态关系（如达尔文雀喙大小排列）且种内变异与种间差异正交。

**[Breaking The Compression Ceiling Data-Free Pipeline For Ultra-Efficient Delta Co](multimodal_vlm/breaking_the_compression_ceiling_data-free_pipeline_for_ultra-efficient_delta_co.md)**

:   提出 UltraDelta——首个无数据 delta 权重压缩流水线，通过方差引导的混合稀疏分配、分布感知压缩和迹范数引导缩放三个组件，在 LLM/NLP/视觉/多模态模型上实现最高 224× 的超高压缩比且性能不降甚至超越微调模型。

**[Bridgevla Input-Output Alignment For Efficient 3D Manipulation Learning With Vis](multimodal_vlm/bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)**

:   提出 BridgeVLA，通过将 3D 点云投影为多视角 2D 图像并以 2D 热力图作为中间表示来对齐输入输出空间，实现了高效且有效的 3D 机器人操作学习。

**[Can Llms Reason Over Non-Text Modalities In A Training-Free Manner A Case Study ](multimodal_vlm/can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)**

:   提出 In-Context Representation Learning（ICRL），首个训练无关框架，将非文本模态基础模型（FM）的表征注入纯文本 LLM 进行少样本推理，通过 PCA 文本注入和最优传输嵌入对齐两种策略实现跨模态知识利用。

**[Can Multi-Modal Llms Provide Live Step-By-Step Task Guidance](multimodal_vlm/can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)**

:   提出 Qualcomm Interactive Cooking 基准和 LiveMamba 模型，首次系统评估多模态 LLM 在实时流式视频中提供分步任务指导（包括指令下发、完成检测和错误反馈）的能力。

**[Capability A Comprehensive Visual Caption Benchmark For Eval](multimodal_vlm/capability_a_comprehensive_visual_caption_benchmark_for_eval.md)**

:   提出CAPability，一个涵盖6大视角12个维度的综合视觉描述评测基准，通过人工标注近11K图像/视频的视觉元素（而非句子），同时评估描述的正确性（precision）和全面性（hit），并引入"知道但说不出"（$K\bar{T}$）指标揭示MLLM在QA与caption任务之间的显著能力差距。

**[Causalllava Causal Disentanglement For Mitigating Hallucinat](multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)**

:   揭示 MLLM 中物体幻觉的表示层根因——数据集共现偏差导致的语义纠缠，提出双路因果解纠缠框架（Causal-Driven Projector + Causal Intervention Module），通过后门调整在 projector 和最终 Transformer 层分离共现物体表示，使 MME-Perception 提升 22.6%。

**[Chartmuseum Testing Visual Reasoning Capabilities Of Large V](multimodal_vlm/chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)**

:   提出ChartMuseum图表问答基准，包含1162个专家标注问题和184个来源的真实图表，首次系统区分视觉推理与文本推理能力，揭示当前最强模型Gemini-2.5-Pro仅63.0%而人类达93%，视觉推理性能比文本推理低35%-55%。

**[Choice Benchmarking The Remote Sensing Capabilities Of Large Vision-Language Mod](multimodal_vlm/choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)**

:   提出 CHOICE，一个面向遥感领域的大规模多层级 VLM 基准，包含 10,507 道全新采集题目，覆盖感知与推理 2 大维度、6 个子维度、23 个叶任务，首次实现对 VLM 遥感能力的系统化与客观化评估。

**[Coido Efficient Data Selection For Visual Instruction Tuning Via Coupled Importa](multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)**

:   提出 CoIDO，一个双目标优化数据选择框架，通过联合优化数据重要性和多样性，仅用 20% 随机数据训练轻量评分器，即可从 LLaVA-665K 中选出 20% 子集达到全量微调 98.2% 的性能，同时计算开销为所有方法最低。

**[Context Informs Pragmatic Interpretation In Vision-Language Models](multimodal_vlm/context_informs_pragmatic_interpretation_in_vision-language_models.md)**

:   通过迭代参考游戏（iterated reference games）系统评估 VLM 的语用推理能力，发现模型在无上下文时表现远逊于人类，但在获得相关对话历史后能快速学习达到约 80% 准确率，揭示了 VLM 对上下文信息的强烈依赖性。

**[Continual Multimodal Contrastive Learning](multimodal_vlm/continual_multimodal_contrastive_learning.md)**

:   本文首次形式化定义了持续多模态对比学习（CMCL）问题，提出双侧零空间梯度投影（DNS）方法，将新数据的梯度投影到不影响旧知识的子空间上，在 7 个数据集上实现了稳定性和可塑性的最佳平衡。

**[Covmatch Crosscovariance Guided Multimodal Dataset Distillat](multimodal_vlm/covmatch_crosscovariance_guided_multimodal_dataset_distillat.md)**

:   提出 CovMatch，通过将多模态对比学习的双层优化简化为跨协方差矩阵对齐的闭式解，首次实现图文双编码器的联合优化进行多模态数据集蒸馏，仅用 500 个合成图文对在 Flickr30K 上获得 38.4 平均检索精度（+6.8% 超越 SOTA LoRS），在极端数据高效场景下大幅超越冻结文本编码器的方法。

**[Cyin Cyclic Informative Latent Space For Bridging Complete And Incomplete Multim](multimodal_vlm/cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)**

:   提出 CyIN 框架，通过 token 级和 label 级信息瓶颈（IB）构建信息化潜空间，结合循环跨模态翻译重建缺失信息，在单一统一模型中同时优化完整和不完整多模态学习。

**[Danmakutppbench A Multimodal Benchmark For Temporal Point Pr](multimodal_vlm/danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)**

:   构建首个多模态时间点过程基准DanmakuTPPBench：DanmakuTPP-Events提供7250个序列共1080万弹幕事件（时间-文本-视频三模态天然对齐），DanmakuTPP-QA通过多智能体pipeline自动生成10类推理问答，系统暴露了经典TPP模型和MLLM在多模态事件动态理解上的显著短板。

**[Dont Just Chase Highlighted Tokens In Mllms Revisiting Visual Holistic Context R](multimodal_vlm/dont_just_chase_highlighted_tokens_in_mllms_revisiting_visual_holistic_context_r.md)**

:   提出 HoloV，一个即插即用的视觉 token 剪枝框架，通过在不同空间裁剪区域自适应分配剪枝预算，保留全局视觉上下文而非仅保留注意力高亮 token，在 LLaVA-1.5 上剪枝 88.9% token 仍保留 95.8% 原始性能。

**[Dota Distributional Testtime Adaptation Of Visionlanguage Mo](multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)**

:   DOTA提出将测试时自适应从"缓存样本实例"范式转变为"持续估计测试数据分布"范式，通过在线高斯判别分析结合零样本预测概率估计类别分布，实现无梯度、抗遗忘的高效测试时自适应，在10个跨域基准上平均准确率超越所有基线。

**[Dynamicvl Benchmarking Multimodal Large Language Models For Dynamic City Underst](multimodal_vlm/dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)**

:   提出 DVL-Suite 框架，包含 DVL-Bench 基准和 DVL-Instruct 指令微调数据集，覆盖 42 座美国城市、14,871 张高分辨率多时相遥感影像，系统评估 18 个 MLLM 在长期城市动态理解上的能力，并开发了 DVLChat 基线模型。

**[Efficient Multi-Modal Large Language Models Via Progressive Consistency Distilla](multimodal_vlm/efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)**

:   提出EPIC框架，通过渐进式一致性蒸馏（Token和Layer两个维度）解决视觉token压缩训练中特征空间扰动导致的学习困难，在不修改模型架构的前提下实现高效多模态LLM。

**[Elasticmm Efficient Multimodal Llms Serving With Elastic Multimodal Parallelism](multimodal_vlm/elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism.md)**

:   提出弹性多模态并行（EMP）范式和 ElasticMM 系统，通过模态感知负载均衡和弹性分区调度将多模态推理的不同阶段解耦到独立实例，相比 vLLM TTFT 降低最高 4.2 倍、吞吐量提升 3.2-4.5 倍。

**[Enhancing Compositional Reasoning In Clip Via Reconstruction](multimodal_vlm/enhancing_compositional_reasoning_in_clip_via_reconstruction.md)**

:   提出 READ 微调方法，通过两个辅助目标——(1) token-level 重建（冻结解码器从文本嵌入重建替代描述）和 (2) sentence-level 对齐（强制改述的嵌入一致）——增强 CLIP 文本编码器的组合推理能力，在 5 个组合推理基准上达到 SOTA（超 NegCLIP 4.5%，超 FSC-CLIP 4.1%）。

**[Enhancing The Outcome Reward-Based Rl Training Of Mllms With Self-Consistency Sa](multimodal_vlm/enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)**

:   针对多模态多选题中"结果奖励 RL 训练导致不忠实推理轨迹"的问题，提出 Self-Consistency Sampling (SCS)，通过截断-重采样和视觉扰动获得一致性奖励来惩罚虚假推理，搭载 RLOO 后在六个基准上平均提升 7.7 个百分点。

**[Enhancing Visionlanguage Model Reliability With Uncertaintyg](multimodal_vlm/enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)**

:   提出Dropout Decoding——将视觉token投影到文本空间后量化其认知不确定性，选择性遮掩高不确定性视觉token并通过多组遮掩结果的集成投票增强输出可靠性，无需额外训练即可显著减少LVLM的对象幻觉。

**[Evaluating Multimodal Large Language Models On Core Music Perception Tasks](multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)**

:   本文通过三项核心音乐感知任务（切分节奏评分、移调检测、和弦辨识）系统性评估了多模态LLM在音频与MIDI两种输入下的表现，揭示了模型在符号推理上接近理想但在音频感知上存在显著缺陷的关键差距。

**[Exgra-Med Extended Context Graph Alignment For Medical Vision-Language Models](multimodal_vlm/exgra-med_extended_context_graph_alignment_for_medical_vision-language_models.md)**

:   ExGra-Med 提出了一种多图对齐（multi-graph alignment）框架，通过联合对齐图像、指令响应和扩展上下文描述在潜空间中的图结构关系，仅用10%预训练数据即可匹配 LLaVA-Med 的100%数据性能，并在多个医学VQA任务上超越现有SOTA。

**[Finegrain Evaluating Failure Modes Of Text-To-Image Models With Vision Language ](multimodal_vlm/finegrain_evaluating_failure_modes_of_text-to-image_models_with_vision_language_.md)**

:   FineGRAIN 提出了一个结构化的联合评测框架，通过定义27种细粒度失败模式和利用 VLM+LLM agentic pipeline 来同时评估文本到图像模型的 prompt 遵循能力和视觉语言模型的图像理解能力，揭示了两类模型在特定任务上的系统性缺陷。

**[First Sft Second Rl Third Upt Continual Improving Multi-Modal Llm Reasoning Via ](multimodal_vlm/first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)**

:   提出 MM-UPT 框架，在 SFT 和 RL 之后引入第三阶段"无监督后训练"，通过多数投票作为伪奖励信号结合 GRPO 实现 MLLM 的自我改进，在 MathVista 上将 Qwen2.5-VL-7B 从 66.3% 提升至 72.9%。

**[Flexac Towards Flexible Control Of Associative Reasoning In Multimodal Large Lan](multimodal_vlm/flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)**

:   FlexAC 发现 MLLM 的联想推理行为主要编码在中间层，通过从幻觉响应中提取引导向量并在推理时注入中间层表示，实现忠实性与创造力的灵活调控——幻觉率降低 29%(CHAIR)，创造力提升 5.8×(Creation-MMBench)，且无需训练。

**[Flowcut Rethinking Redundancy Via Information Flow For Effic](multimodal_vlm/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)**

:   从信息流（Information Flow）视角重新理解VLM中视觉token冗余性的涌现机制，提出FlowCut框架通过层自适应剪枝比例、多标准融合评分和累积重要性跟踪实现与模型内在信息传播行为对齐的token剪枝，在LLaVA-1.5-7B上以88.9% token减少率超越SOTA 1.6%，LLaVA-NeXT-7B上以94.4%减少率超越4.3%。

**[Flysearch Exploring How Vision-Language Models Explore](multimodal_vlm/flysearch_exploring_how_vision-language_models_explore.md)**

:   FlySearch 提出了一个基于 Unreal Engine 5 的 3D 户外真实感环境，评估 VLM 的探索能力，发现最先进的 VLM 在简单搜索任务上也无法可靠完成，且与人类的差距随任务难度增加而急剧扩大。

**[Focus Internal Mllm Representations For Efficient Fine-Grained Visual Question A](multimodal_vlm/focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)**

:   提出 FOCUS，一种无需训练的视觉裁剪方法，利用 MLLM 内部 KV-cache 中 value 特征的余弦相似度构建目标相关性图，高效定位问题相关的图像区域，在细粒度 VQA 上实现与 SOTA 可比的精度，同时计算效率提升 3-6.5 倍。

**[Forcevla Enhancing Vla Models With A Force-Aware Moe For Contact-Rich Manipulati](multimodal_vlm/forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)**

:   提出 ForceVLA，在 VLA 框架中将 6 轴力/力矩传感引入为一等模态，通过 FVLMoE（力感知混合专家）模块在动作解码阶段动态融合视觉-语言嵌入与实时力反馈，在 5 项接触密集操作任务上平均成功率提升 23.2%，个别任务达 80%。

**[Gem Empowering Mllm For Grounded Ecg Understanding With Time Series And Images](multimodal_vlm/gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)**

:   提出 GEM，首个统一 ECG 时间序列、12 导联 ECG 图像和文本的多模态大语言模型，通过双编码器框架、跨模态对齐和知识引导的指令数据生成，实现了基于可量化生理特征的接地心电图诊断，诊断准确率提升 7.4%，可解释性提升 22.7%，接地能力提升 25.3%。

**[Generate But Verify Reducing Hallucination In Visionlanguage](multimodal_vlm/generate_but_verify_reducing_hallucination_in_visionlanguage.md)**

:   提出REVERSE框架，首次将生成调整和事后验证统一到单个VLM中：通过1.3M半合成样本的幻觉感知训练+推理时回溯重采样，使VLM能在生成过程中自动检测并修正幻觉，在CHAIR-MSCOCO上降低12%、HaloQuest上提升34%。

**[Georanker Distance-Aware Ranking For Worldwide Image Geolocalization](multimodal_vlm/georanker_distance-aware_ranking_for_worldwide_image_geolocalization.md)**

:   提出 GeoRanker，一种距离感知排序框架，利用大视觉语言模型建模查询-候选之间的空间关系，通过多阶距离损失实现全球图像地理定位的 SOTA。

**[Glsim Detecting Object Hallucinations In Lvlms Via Globalloc](multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)**

:   提出GLSim，一种无训练的LVLM物体幻觉检测方法，通过融合全局场景相似度（物体token与最后instruction token的余弦相似度）和局部视觉定位相似度（物体token与Visual Logit Lens定位的Top-K图像patch的余弦相似度），在MSCOCO上以83.7% AUROC超越SVAR 9%、Internal Confidence 10.8%。

**[Goalladder Incremental Goal Discovery With Vision-Language Models](multimodal_vlm/goalladder_incremental_goal_discovery_with_vision-language_models.md)**

:   提出 GoalLadder，利用 VLM 渐进式发现并排序候选目标状态，结合 ELO 评分系统抵抗噪声反馈，在学习的嵌入空间中定义距离奖励，仅凭单条语言指令就能训练 RL 智能体达到约 95% 的成功率。

**[Guiding Cross-Modal Representations With Mllm Priors Via Preference Alignment](multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)**

:   提出 MAPLE 框架，利用现成 MLLM 的内在模态对齐能力自动构建偏好数据，通过 Relative Preference Alignment（RPA）损失引导跨模态表示学习，在细粒度检索任务上取得显著提升。

**[Hawaii Hierarchical Visual Knowledge Transfer For Efficient Vision-Language Mode](multimodal_vlm/hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)**

:   提出 Hawaii 框架，通过混合 LoRA 适配器（MoLA）和分层知识蒸馏（HKD），将多个视觉专家的知识蒸馏到单个视觉编码器中，在不增加推理成本的前提下显著提升 VLM 的视觉理解能力。

**[Hermesflow Seamlessly Closing The Gap In Multimodal Understanding And Generation](multimodal_vlm/hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)**

:   首次揭示统一多模态大模型中理解能力普遍强于生成能力的现象，提出 HermesFlow 框架，通过同源偏好数据构建配对理解-生成偏好对，利用 Pair-DPO 和自博弈迭代优化，在不引入外部高质量数据的情况下同步提升理解与生成能力并缩小两者差距。

**[Hierarchical Self-Attention Generalizing Neural Attention Mechanics To Multi-Sca](multimodal_vlm/hierarchical_self-attention_generalizing_neural_attention_mechanics_to_multi-sca.md)**

:   从熵最小化第一性原理推导出层次化自注意力（HSA）机制，为嵌套信号（多模态、多尺度数据）提供理论最优的注意力计算方法，并证明 HSA 是在保持层次约束下最接近标准 Softmax 注意力的 KL 散度最优解。

**[Hope Hybrid Of Position Embedding For Long Context Visionlan](multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)**

:   首次从理论上分析多模态RoPE的频率分配策略对长上下文VLM的影响，提出HoPE，将最低频率设为零用于时间建模以保证语义偏好性质，配合动态时间缩放机制，在长视频理解任务上提升8.35%、检索任务上提升22.23%。

**[Ifinder Structured Zero-Shot Vision-Based Llm Grounding For Dash-Cam Video Reaso](multimodal_vlm/ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)**

:   提出 iFinder，一个模块化免训练框架，将行车记录仪视频解耦为感知（结构化场景表示）与推理（LLM），通过层级数据结构和三块式提示策略使 LLM 获得可解释的时空推理能力，在四个驾驶视频基准上零样本超越端到端 V-VLM，事故推理准确率提升高达 39%。

**[Intervene-All-Paths Unified Mitigation Of Lvlm Hallucinations Across Alignment F](multimodal_vlm/intervene-all-paths_unified_mitigation_of_lvlm_hallucinations_across_alignment_f.md)**

:   提出 AllPath，一个基于 Transformer 因果架构的多路径幻觉干预框架，首次发现 LVLM 的幻觉不来自单一因果路径而是 image-to-input-text、image-to-output-text、text-to-text 三条路径的交互，并且模型会根据问答对齐格式自适应选择不同路径；通过为每条路径设计轻量级关键 head 识别方法并自适应干预，在 POPE、MCQ-POPE、CHAIR、MME 四个不同格式 benchmark 上一致降低幻觉。

**[Jailbound Jailbreaking Internal Safety Boundaries Of Vision-Language Models](multimodal_vlm/jailbound_jailbreaking_internal_safety_boundaries_of_vision-language_models.md)**

:   受 Eliciting Latent Knowledge (ELK) 框架启发，首次揭示 VLM 在 fusion layer 潜空间中存在可近似的安全决策边界，提出 JailBound 两阶段攻击框架（Safety Boundary Probing + Safety Boundary Crossing），通过联合优化图像和文本对抗扰动跨越该边界，在白盒和黑盒场景分别达到 94.32% 和 67.28% 平均攻击成功率，显著超越 SOTA。

**[Learning Shared Representations From Unpaired Data](multimodal_vlm/learning_shared_representations_from_unpaired_data.md)**

:   提出 SUE (Spectral Universal Embedding)，首次证明几乎完全依赖非配对数据即可学习跨模态共享表示：通过独立的频谱嵌入从各模态随机游走中提取模态不变的"通用"结构，再用极少量配对样本（~100对）做 CCA 线性对齐 + MMD 非线性微调，在检索上超越使用同等配对数的对比学习 250%+。

**[Learning To Instruct For Visual Instruction Tuning](multimodal_vlm/learning_to_instruct_for_visual_instruction_tuning.md)**

:   提出 L2T（Learning to Instruct），仅通过将训练损失扩展到指令序列（不再只在回答上计算 loss）来改善视觉指令调优——无额外数据和几乎零计算开销，在 16 个多模态基准上获得高达 9% 的相对提升，captioning 提升 18%，同时缓解幻觉。

**[Learning To Steer Input-Dependent Steering For Multimodal Llms](multimodal_vlm/learning_to_steer_input-dependent_steering_for_multimodal_llms.md)**

:   针对现有模型引导(steering)方法使用固定方向向量无法适配不同输入的局限，提出 L2S (Learn-to-Steer)：先通过输入特定的对比提示生成理想的引导向量（P2S），再训练一个轻量 2 层 MLP 从输入上下文预测该向量，以极低开销实现了输入依赖的行为引导，在安全执行和幻觉缓解两个应用上显著优于静态 steering 基线。

**[Metacognitive Sensitivity For Test-Time Dynamic Model Selection](multimodal_vlm/metacognitive_sensitivity_for_test-time_dynamic_model_selection.md)**

:   借鉴人类认知科学中的元认知灵敏度（meta-d'）概念，提出一种测试时动态模型选择框架：用 meta-d' 量化模型"知道自己知不知道"的能力，结合即时置信度构成上下文向量，通过 contextual bandit 在线选择最优模型，在多数据集上超越单模型性能。

**[Midas Misalignment-Based Data Augmentation Strategy For Imbalanced Multimodal Le](multimodal_vlm/midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)**

:   首次提出将跨模态不对齐样本作为有监督训练信号（而非噪声/干扰）来缓解多模态学习中的模态不平衡问题，设计 MIDAS 数据增强框架：通过置信度标注不对齐样本 + 弱模态加权 + 难样本加权三重机制，在四个多模态分类基准上显著超越现有方法。

**[Mint A Simple Testtime Adaptation Of Visionlanguage Models A](multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)**

:   发现 CLIP 在图像损坏下的性能退化根源在于**嵌入方差坍缩**——类内与类间方差同步缩小导致嵌入空间判别性丧失；提出 Mint，通过最大化伪标签类间方差（PL-inter）在线修复嵌入几何，仅凭均值累加器和梯度累加器两个极简组件即可在 BS=1 的在线场景下稳定提升 CLIP 在多种损坏基准上的分类精度，同时比最强 baseline 快 45 倍。

**[Mirage A Benchmark For Multimodal Information-Seeking And Reasoning In Agricultu](multimodal_vlm/mirage_a_benchmark_for_multimodal_information-seeking_and_reasoning_in_agricultu.md)**

:   MIRAGE 是首个基于真实农业专家咨询对话（35,000+条）构建的多模态基准，评估视觉语言模型在领域级实体识别、因果推理和"澄清还是回答"决策方面的能力，揭示了即使 GPT-4.1 识别准确率也仅 43.9% 的严峻挑战。

**[Mm-Opera Benchmarking Open-Ended Association Reasoning For Large Vision-Language](multimodal_vlm/mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)**

:   提出 MM-OPERA，一个包含 11,497 实例的开放式联想推理基准，通过远程物品关联（RIA）和上下文关联（ICA）两大任务评估 LVLM 的关联推理能力，配套设计了 LLM-as-a-Judge 评分策略和过程奖励评估方法，揭示当前最强 LVLM 仍显著落后于人类。

**[Mme-Videoocr Evaluating Ocr-Based Capabilities Of Multimodal Llms In Video Scena](multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)**

:   提出 MME-VideoOCR，一个包含 25 个任务、44 个场景、1464 个视频和 2000 个人工标注 QA 对的视频 OCR 综合评估基准，涵盖文本识别、理解和推理三个层次。评估 18 个 SOTA MLLM 揭示最强模型（Gemini-2.5 Pro）仅达 73.7%，跨帧理解任务低至 25% 以下。

**[Mmlongbench Benchmarking Longcontext Visionlanguage Models E](multimodal_vlm/mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)**

:   构建首个全面的长上下文视觉语言模型（LCVLM）评估基准 MMLongBench——13,331 个样本覆盖 5 类下游任务、混合图像类型、5 级标准化输入长度（8K-128K tokens），评估 46 个模型后发现单任务性能是整体能力的弱代理，且强推理能力与长上下文性能正相关。

**[Mmperspective Do Mllms Understand Perspective A Comprehensive Benchmark For Pers](multimodal_vlm/mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)**

:   首个系统评估多模态大语言模型 (MLLMs) 透视理解能力的基准，包含10个任务、3个维度、2711张图像和5083个问答对，揭示了43个SOTA模型在透视推理和鲁棒性方面的显著不足。

**[Monitor Exploiting Large Language Models With Instruction For Online Video Anoma](multimodal_vlm/monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)**

:   提出 MoniTor，一个基于记忆的在线评分队列方案，利用 LLM 进行免训练的在线视频异常检测（VAD），通过双层记忆机制、行为预测和标准评分队列引导 LLM 实现实时异常识别。

**[Multi-Modal Masked Autoencoders For Learning Image-Spectrum Associations For Gal](multimodal_vlm/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   构建了包含 134,533 个星系的图像-光谱-红移多模态数据集（GalaxiesML-Spectra），适配多模态掩码自编码器（MMAE）同时进行图像和光谱的联合重建与红移回归，证明在测试时即使光谱完全缺失，仅用 25% 掩码图像即可实现优于 AstroCLIP 的红移预测散度 $\sigma_{NMAD} = 0.016$。

**[Multimodal Bandits Regret Lower Bounds And Optimal Algorithms](multimodal_vlm/multimodal_bandits_regret_lower_bounds_and_optimal_algorithms.md)**

:   针对奖励函数至多有 $m$ 个极值的多模态多臂赌博机问题，提出首个计算可行的算法求解 Graves-Lai 优化问题，实现渐近最优的遗憾界，并证明局部搜索策略是次优的。

**[Multimodal Negative Learning](multimodal_vlm/multimodal_negative_learning.md)**

:   提出多模态负学习（MNL）范式，通过让强势模态引导弱势模态抑制非目标类别（而非强制对齐目标类别），稳定决策空间并保留模态特有信息，理论上收紧了多模态融合的鲁棒性下界。

**[Nautilus A Large Multimodal Model For Underwater Scene Understanding](multimodal_vlm/nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)**

:   构建了首个支持八种水下场景理解任务的大型多模态模型 Nautilus，通过物理先验驱动的视觉特征增强（VFE）模块显式修复水下图像退化，提升 LMM 在水下环境中的鲁棒性。

**[Navil Rethinking Scaling Properties Of Native Multimodal Large Language Models U](multimodal_vlm/navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)**

:   本文系统研究了在数据约束条件下原生多模态大语言模型(Native MLLM)的设计空间与缩放特性，发现视觉编码器与LLM之间存在正相关的最优缩放关系，并基于此提出了NaViL模型，仅用约6亿预训练图文对即可达到顶级MLLM的竞争性性能。

**[Needleinatable Exploring Long-Context Capability Of Large Language Models Toward](multimodal_vlm/needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)**

:   提出 NeedleInATable (NIAT) 基准，将表格中每个单元格视为"针"，评估 LLM 对长结构化表格的细粒度感知能力，揭示现有模型在复杂下游任务上的高分可能依赖数据捷径而非真正的表格理解。

**[Negocollab A Common Representation Negotiation Approach For Heterogeneous Collab](multimodal_vlm/negocollab_a_common_representation_negotiation_approach_for_heterogeneous_collab.md)**

:   提出 NegoCollab 框架，通过引入协商者（Negotiator）在训练期间从多模态 agent 的局部表示中协商生成公共表示，有效消除异质协作 agent 之间的域差异，实现低训练成本的协同网联感知。

**[Omni-Mol Multitask Molecular Model For Any-To-Any Modalities](multimodal_vlm/omni-mol_multitask_molecular_model_for_any-to-any_modalities.md)**

:   提出 Omni-Mol，一个基于多模态 LLM 的统一分子理解与生成框架，通过构建 142 万样本的指令微调数据集、Gradient Adaptive LoRA (GAL) 和 Mixture-of-GAL-Experts (MoGE) 架构，首次在单一模型中统一学习 16 个分子任务（Mol2Mol/Mol2Text/Mol2Num/Text2Mol），以仅 2.2B 参数在 13 个任务上达到 SOTA。

**[Omnigaze Reward-Inspired Generalizable Gaze Estimation In The Wild](multimodal_vlm/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)**

:   提出OmniGaze，一个半监督3D注视估计框架，利用融合视觉嵌入、MLLM生成的语义注视描述和几何方向向量的奖励模型来评估伪标签质量，在140万无标签人脸数据上训练，在5个数据集的域内/跨域设置下达到SOTA，并在4个未见数据集上展示零样本泛化能力。

**[On The Value Of Cross-Modal Misalignment In Multimodal Representation Learning](multimodal_vlm/on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)**

:   提出潜变量模型将跨模态失配形式化为选择偏差和扰动偏差两种机制，理论证明MMCL学到的表征恰好捕获与两种偏差无关的不变语义子集，统一了"失配有害/有益"两种对立观点。

**[Openhoi Open-World Hand-Object Interaction Synthesis With Multimodal Large Langu](multimodal_vlm/openhoi_open-world_hand-object_interaction_synthesis_with_multimodal_large_langu.md)**

:   提出 OpenHOI 框架，利用多模态大语言模型（MLLM）的常识推理能力来推断陌生物体的接触区域和抓取类型，实现开放世界的手物交互合成，无需针对每个物体收集训练数据。

**[Permllm Learnable Channel Permutation For Nm Sparse Large Language Models](multimodal_vlm/permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)**

:   提出 PermLLM，首个可学习通道排列（LCP）框架，通过Sinkhorn归一化将离散排列矩阵松弛为可微分的软排列矩阵实现端到端优化，结合块级排列策略大幅降低计算开销，有效提升N:M稀疏LLM的性能。

**[Physvlm-Avr Active Visual Reasoning For Multimodal Large Language Models In Phys](multimodal_vlm/physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)**

:   本文提出主动视觉推理（AVR）任务范式，构建了CLEVR-AVR仿真基准和AVR-152k数据集（含丰富CoT标注），训练PhysVLM-AVR模型在部分可观测交互环境中通过感知-推理-动作闭环迭代获取信息并正确回答问题，显著优于现有MLLM。

**[Praxisvlm Visiongrounded Decision Making Via Textdriven Rein](multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)**

:   发现VLM的决策推理能力可与视觉感知解耦——用文本描述替代图像时决策准确率不降反升；据此提出Praxis-VLM，在纯文本场景上通过多阶段GRPO与自适应reward训练决策推理能力，推理时零样本迁移到视觉输入，在三大决策benchmark上全面超越SFT基线，尤其在OOD场景泛化优势显著。

**[Prefixkv Adaptive Prefix Kv Cache Is What Vision Instruction](multimodal_vlm/prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)**

:   PrefixKV 发现不同层 KV 缓存的重要性分布差异显著，将逐层缓存大小确定问题形式化为全局前缀配置搜索，通过二分搜索找到最优信息保留阈值使每层保持最大上下文信息，在 20% 压缩率下仅有 0.49 PPL 下降且提供 1.8× 推理加速。

**[Reading Recognition In The Wild](multimodal_vlm/reading_recognition_in_the_wild.md)**

:   提出了阅读识别新任务及首个大规模多模态"野外阅读"数据集（100小时），利用RGB、眼动和IMU三种互补模态的轻量级Transformer模型，在智能眼镜上实现实时阅读检测。

**[Recognition Through Reasoning Reinforcing Image Geo-Localization With Large Visi](multimodal_vlm/recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)**

:   本文提出GLOBE——一个基于GRPO强化学习训练的LVLM图像地理定位系统，通过构建推理导向数据集MP16-Reason（含定位可行性评估、视觉线索推理链和地理准确性标注），仅用33K样本就在多个基准上超越基于数百万样本训练的SOTA方法和大规模开源VLM。

**[Rethinking Multimodal Learning From The Perspective Of Mitig](multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)**

:   提出"**分类能力不均衡**"视角理解多模态学习中的模态不平衡，设计 Sustained Boosting 算法（共享编码器 + 多可配置分类器，同时优化分类和残差误差）配合自适应分类器分配（ACA），理论证明跨模态 gap loss 以 $\mathcal{O}(1/T)$ 收敛，在 CREMAD 等 6 个数据集上大幅超越 SOTA。

**[Retrv-R1 A Reasoning-Driven Mllm Framework For Universal And Efficient Multimoda](multimodal_vlm/retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)**

:   提出首个R1风格的推理型多模态检索框架Retrv-R1，通过信息压缩模块降低token消耗、细节检查机制保留困难候选的完整信息、课程式RL奖励兼顾效果与效率，在通用多模态检索benchmark上实现SOTA。

**[Revisiting Logit Distributions For Reliable Out-Of-Distribution Detection](multimodal_vlm/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)**

:   提出 LogitGap，一种新的 post-hoc OOD 检测评分函数，通过显式利用最大 logit 与其余 logit 之间的"间隔"来区分 ID 和 OOD 样本，并引入 top-N 选择策略过滤噪声 logit，理论和实验证明其在多种场景下超越 MCM 和 MaxLogit。

**[Roborefer Towards Spatial Referring With Reasoning In Vision-Language Models For](multimodal_vlm/roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)**

:   提出 **RoboRefer**，一个 3D 感知的推理型 VLM，通过 **SFT + RFT** 两阶段训练策略（含度量敏感的过程奖励函数），在空间指代任务中实现精确的单步空间理解和多步空间推理，在 RefSpatial-Bench 上超越 Gemini-2.5-Pro 达 17.4%。

**[Robustmerge Parameter-Efficient Model Merging For Mllms With Direction Robustnes](multimodal_vlm/robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)**

:   本文从低秩分解的角度揭示了参数高效模块合并中"方向鲁棒性"是关键因素（而非全参数合并中的符号冲突），提出RobustMerge通过互补参数自适应缩放和跨任务归一化维持奇异值方向稳定性，在多模态生成任务上平均提升3.4%（已见任务）和4.5%（未见任务）。

**[Rtv-Bench Benchmarking Mllm Continuous Perception Understanding And Reasoning Th](multimodal_vlm/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)**

:   提出 RTV-Bench 基准，包含 552 个视频和 4608 个 QA 对，通过**多时间戳问答**（同一问题在不同时间点答案不同）、**层级问题结构**和**多维评估**三大设计，系统评测 MLLM 在实时视频流中的持续分析能力，揭示了在线模型优于离线模型、单纯增大模型或增加帧数收益有限等关键发现。

**[Scene-Aware Urban Design A Human-Ai Recommendation Framework Using Co-Occurrence](multimodal_vlm/scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)**

:   提出一个人机协同的计算机视觉框架，使用Grounding DINO进行城市物体检测，基于ADE20K数据集构建共现嵌入捕捉真实空间配置，再通过VLM进行场景感知的第三物体推荐，并生成3D模型用于AR预览，旨在让居民参与微观城市设计。

**[Scope Saliency-Coverage Oriented Token Pruning For Efficient Multimodel Llms](multimodal_vlm/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)**

:   提出 SCOPE，一种联合建模显著性和覆盖率的视觉 Token 剪枝策略，通过迭代选择 SCOPE 得分最高的 Token 来保持语义完整性，在 9 倍 Token 缩减下保留 LLaVA-1.5 96% 的性能。

**[Sd-Vlm Spatial Measuring And Understanding With Depth-Encoded Vision-Language Mo](multimodal_vlm/sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)**

:   提出MSMU大规模定量空间推理数据集（700K QA对、250万数值标注）和深度位置编码（DPE）方法，使VLM在不引入3D点云的前提下获得强大的定量空间测量和理解能力，在MSMU-Bench上超越GPT-4o达26.91%。

**[Seeing Is Believing Mitigating Ocr Hallucinations In Multimodal Large Language M](multimodal_vlm/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)**

:   针对多模态大模型在退化文档场景下的OCR幻觉问题，提出首个退化文档幻觉评测基准KIE-HVQA，并设计基于GRPO的多目标奖励强化学习框架，在7B参数模型上实现比GPT-4o高约28%的幻觉抑制准确率提升。

**[Seetrek Training-Free Spatial Prompting For Multimodal Large Language Model](multimodal_vlm/seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)**

:   提出 See&Trek，一个无需训练和GPU的空间提示框架，通过最大语义丰富度采样和运动重建来增强 MLLM 的空间理解能力，在 VSI-Bench 上最高提升 3.5%。

**[Sherlock Selfcorrecting Reasoning In Visionlanguage Models](multimodal_vlm/sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)**

:   首个系统研究VLM推理自纠正能力的框架：发现现有推理VLM几乎不能自纠正（<10%出现aha moment），提出Sherlock三阶段训练框架（SFT冷启动→离线轨迹级偏好学习→在线自我迭代）仅用20K标注数据超越使用100K-260K数据的LLaVA-CoT/Mulberry/LlamaV-o1。

**[Situat3Dchange Situated 3D Change Understanding Dataset For Multimodal Large Lan](multimodal_vlm/situat3dchange_situated_3d_change_understanding_dataset_for_multimodal_large_lan.md)**

:   构建 Situat3DChange 数据集（174K 数据实例），统一了动态场景变化与情境感知理解的感知-行动范式，并提出 SCReasoner——一种高效的 3D MLLM 用于点云对比推理。

**[Sparse Autoencoders Learn Monosemantic Features In Visionlan](multimodal_vlm/sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)**

:   本文将稀疏自编码器（SAE）扩展到视觉-语言模型（如CLIP）上，提出了 MonoSemanticity score（MS）来定量评估神经元的单义性，并展示了通过操控 SAE 神经元可以直接引导多模态大模型（如 LLaVA）的输出，实现概念的插入与抑制。

**[Spatialthinker Reinforcing 3D Reasoning In Multimodal Llms Via Spatial Rewards](multimodal_vlm/spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)**

:   提出 SpatialThinker，通过在线 RL 结合多目标密集空间奖励（格式→计数→准确性→空间定位的字典序门控）训练 MLLM 构建场景图并进行结构化空间推理，仅用 7K 样本超越 GPT-4o 在 3DSRBench 上 12.1%。

**[Spatialtracegen High-Fidelity Traces For Efficient Vlm Spatial Reasoning Distill](multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)**

:   提出 SpatialTraceGen 框架，通过自动化验证器从大型教师模型蒸馏高质量多步工具使用推理轨迹，用于高效微调小型 VLM 的空间推理能力。

**[Srpo Enhancing Multimodal Llm Reasoning Via Reflection-Aware Reinforcement Learn](multimodal_vlm/srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)**

:   提出 SRPO（Self-Reflection enhanced reasoning with Group Relative Policy Optimization），一个两阶段反思感知 RL 框架：第一阶段用大模型生成反思数据做 SFT cold-start，第二阶段设计反思感知奖励函数在 GRPO 中强化简洁有效的自我反思能力，在 MathVista/MathVision/MMMU-Pro 等多模态推理基准上以 7B/32B 模型显著超越同规模 SOTA。

**[Ssr Enhancing Depth Perception In Vision-Language Models Via Rationale-Guided Sp](multimodal_vlm/ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)**

:   提出 SSR 框架，将原始深度信息转化为结构化文本推理 rationale，并通过知识蒸馏压缩为紧凑潜在嵌入，以即插即用方式增强现有 VLM 的空间推理能力。

**[Struct2D A Perception-Guided Framework For Spatial Reasoning In Mllms](multimodal_vlm/struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)**

:   提出 Struct2D，一种感知引导的提示框架，通过将3D感知输出转化为结构化2D表示（BEV图像+对象标记+元数据），使MLLM无需显式3D输入即可完成复杂空间推理任务，并构建了200K QA对的大规模指令微调数据集 Struct2D-Set。

**[Structure-Aware Fusion With Progressive Injection For Multimodal Molecular Repre](multimodal_vlm/structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)**

:   提出 MuMo 框架，通过结构化融合管线（SFP）将 2D 拓扑与 3D 几何融合为稳定的结构先验，再通过渐进注入（PI）机制非对称地整合到序列流中，在 29 个分子属性预测基准任务中平均提升 2.7%，在 22 个任务上排名第一。

**[Systematic Reward Gap Optimization For Mitigating Vlm Hallucinations](multimodal_vlm/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)**

:   提出 Topic-level Preference Rewriting（TPR），通过 topic 级别的细粒度语义控制系统性优化偏好数据中的 reward gap 配置，结合课程学习策略逐步提高负样本难度，在多个幻觉基准上实现约 93% 的幻觉减少。

**[Test-Time Spectrum-Aware Latent Steering For Zero-Shot Generalization In Vision-](multimodal_vlm/test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)**

:   提出STS（Spectrum-Aware Test-Time Steering），一种轻量级测试时适应方法：通过文本嵌入的SVD分解提取低维语义子空间，学习少量系数在该子空间内对文本原型进行"转向"以适应域偏移，无需反向传播通过大编码器，推理速度比TPT快8倍、内存占用减少12倍，同时在OOD数据集上大幅超越现有TTA方法。

**[Text To Robotic Assembly Of Multi Component Objects Using 3D Generative Ai And V](multimodal_vlm/text_to_robotic_assembly_of_multi_component_objects_using_3d_generative_ai_and_v.md)**

:   提出了一个端到端流水线，将自然语言输入通过3D生成式AI转化为网格模型，再利用VLM的零样本多模态推理自动分解为多组件3D模型（结构件+面板件），最终由机器人臂自动装配成物理对象，并支持用户通过对话反馈调整组件分配。

**[The Illusion Of Progress A Critical Look At Testtime Adaptat](multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)**

:   提出TTA-VLM benchmark，在统一实验条件下评估8种episodic和7种online测试时适应(TTA)方法在15个数据集上的表现，发现三个令人意外的结论：(1) 现有TTA方法相比早期TPT基线提升有限；(2) TTA与训练时微调方法协作效果差；(3) 准确率提升以牺牲校准、OOD检测和鲁棒性为代价。

**[Think Or Not Think A Study Of Explicit Thinking In Rule-Based Visual Reinforceme](multimodal_vlm/think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)**

:   系统研究了基于规则的强化微调（RFT）中显式思维过程的必要性，发现视觉感知任务中"不思考"的RFT（No-Thinking-RFT）往往优于传统的"先思考再回答"策略，并提出了自适应思维方法让模型根据自身能力和任务复杂度决定是否思考。

**[To See Or To Read User Behavior Reasoning In Multimodal Llms](multimodal_vlm/to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)**

:   提出BehaviorLens基准框架，系统比较文本、散点图和流程图三种用户行为历史的表示方式对MLLM次购预测的影响，发现图像表示相比等效文本表示最高可提升87.5%的预测准确率，且无需额外计算开销。

**[Tomcat Test-Time Comprehensive Knowledge Accumulation For Compositional Zero-Sho](multimodal_vlm/tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)**

:   提出 TOMCAT，通过在测试时从无标签数据中累积文本和视觉双模态知识来动态更新组合原型，克服标签分布偏移问题，在四个 CZSL 基准上实现 SOTA。

**[Towards Comprehensive Scene Understanding Integrating First And Third-Person Vie](multimodal_vlm/towards_comprehensive_scene_understanding_integrating_first_and_third-person_vie.md)**

:   提出 E3VQA 基准（首个多视角 VQA 基准）和 M3CoT 提示技术（融合三种互补视角的场景图），增强大型视觉语言模型 (LVLM) 的多视角场景理解能力，GPT-4o 提升 4.84%、Gemini 2.0 Flash 提升 5.94%。

**[Towards Evaluating Proactive Risk Awareness Of Multimodal Language Models](multimodal_vlm/towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)**

:   提出PaSBench基准评估多模态语言模型的主动风险感知能力——要求模型在无用户提问的情况下主动观察环境并发出安全预警。评测36个模型发现最强模型（Gemini-2.5-pro）仅达71%准确率且45%的风险无法稳定检测，核心瓶颈是不稳定的主动推理能力而非知识缺失。

**[Training-Free Online Video Step Grounding](multimodal_vlm/training-free_online_video_step_grounding.md)**

:   提出BaGLM，一种无需训练的在线视频步骤定位方法，利用贝叶斯滤波将LLM估计的步骤依赖关系和LMM估计的步骤进度融入零样本LMM预测中，在三个数据集上超越现有需训练的离线方法。

**[Trove Discovering Errorinducing Static Feature Biases In Tem](multimodal_vlm/trove_discovering_errorinducing_static_feature_biases_in_tem.md)**

:   TRoVe提出自动化方法发现时序VLM中导致系统性预测错误的静态特征偏差，通过结合"错误贡献分数"和"静态偏差分数"的双评分机制，在101个合成模型上以28.6%优势超越基线，并成功应用于7个真实VLM揭示新偏差。

**[Uni-Mumer Unified Multi-Task Fine-Tuning Of Vision-Language Model For Handwritte](multimodal_vlm/uni-mumer_unified_multi-task_fine-tuning_of_vision-language_model_for_handwritte.md)**

:   提出 Uni-MuMER，通过三种数据驱动任务（Tree-CoT、Error-Driven Learning、Symbol Counting）对开源 VLM 进行统一多任务微调，在 CROHME 和 HME100K 数据集上大幅超越专用轻量模型和零样本商用 VLM。

**[Unified Reinforcement And Imitation Learning For Vision-Language Models](multimodal_vlm/unified_reinforcement_and_imitation_learning_for_vision-language_models.md)**

:   提出 RIL（Unified Reinforcement and Imitation Learning）训练框架，结合 GRPO 强化学习和 GAIL 对抗模仿学习，让小型 VLM（7B）通过学习大型 VLM（72B）的文本生成风格来大幅提升性能，无需增加推理延迟或"思考"过程。

**[Unifying Vision-Language Latents For Zero-Label Image Caption Enhancement](multimodal_vlm/unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)**

:   本文提出ViZer框架，通过统一视觉-语言潜空间对齐的训练范式，在无任何文本标注的情况下提升VLM的图像描述能力——仅使用原始图像数据就能让模型生成更接地、更描述性的caption。

**[Unitok A Unified Tokenizer For Visual Generation And Understanding](multimodal_vlm/unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)**

:   提出 UniTok，一种统一视觉生成和理解的tokenizer，通过多码本量化（MCQ）突破离散token表示容量瓶颈，在ImageNet上实现0.38 rFID和78.6%零样本精度的双项记录，并可无缝集成到MLLM中同时启用生成和理解能力。

**[Unveiling Chain Of Step Reasoning For Visionlanguage Models](multimodal_vlm/unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)**

:   提出Chain-of-Step (CoS)推理框架，将VLM的推理链拆解为由Name+Thought+Reflection组成的结构化步骤，训练步骤级Process Reward Model (PRM)提供精细奖励信号，配合迭代DPO和step-level beam search系统性提升VLM推理能力——在InternVL-2.5-MPO-8B上6个benchmark平均73.4%（+4.0%），在LLaVA-NeXT-8B上平均64.2%（+12.1%），并揭示了"VLM推理中质量远比长度重要"这一与LLM领域相反的发现。

**[Vagen Reinforcing World Model Reasoning For Multi-Turn Vlm Agents](multimodal_vlm/vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)**

:   提出VAGEN框架，通过将VLM智能体的推理过程结构化为StateEstimation和TransitionModeling来构建内部世界模型，结合WorldModeling Reward和Bi-Level GAE实现高效的多轮RL训练，使3B模型（0.82）超越GPT-5（0.75）和Gemini 2.5 Pro（0.67）。

**[Vamp Variational Multi-Modal Prompt Learning For Vision-Language Models](multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)**

:   提出变分多模态提示学习框架VaMP，将文本侧提示建模为隐变量并通过变分推断进行实例级不确定性建模，结合类感知先验正则化隐空间，在少样本和域泛化设置下显著提升CLIP的下游适配能力。

**[Video-R1 Reinforcing Video Reasoning In Mllms](multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)**

:   受DeepSeek-R1启发，首次系统探索将R1范式（规则RL）应用于视频推理，提出T-GRPO算法显式鼓励模型利用时序信息，并构建图文混合训练数据集，在VSI-Bench上以37.1%准确率超越GPT-4o。

**[Video-Safetybench A Benchmark For Safety Evaluation Of Video Lvlms](multimodal_vlm/video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)**

:   构建首个面向视频 LVLM 安全评估的综合基准 Video-SafetyBench，包含 2264 个视频-文本对覆盖 48 个细粒度不安全类别，通过可控视频生成管线和基于 LLM 置信度的 RJScore 指标，对 24 个 LVLM 进行大规模安全评测，揭示良性查询下视频攻击平均成功率达 67.2%。

**[Vipamin Visual Prompt Initialization Via Embedding Selection And Subspace Expans](multimodal_vlm/vipamin_visual_prompt_initialization_via_embedding_selection_and_subspace_expans.md)**

:   提出VIPAMIN——一种零额外参数的视觉prompt初始化策略，通过注意力引导的语义匹配（Matching）和正交子空间注入（Orthogonalizing）两个模块，解决自监督VPT中prompt注意力均匀化和子空间坍塌两大失效模式，仅需单次前向传播即在24个视觉任务上刷新SOTA。

**[Vision Function Layer In Multimodal Llms](multimodal_vlm/vision_function_layer_in_multimodal_llms.md)**

:   发现MLLM中视觉相关的功能解码分布在特定的窄层块中（Vision Function Layer），且跨模型家族呈现一致的层级顺序（识别→计数→定位→OCR），据此提出VFL-LoRA（仅用1/3参数匹配full-LoRA性能）和VFL-select（20%数据达98%全量性能）。

**[Vispec Accelerating Vision-Language Models With Vision-Aware Speculative Decodin](multimodal_vlm/vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)**

:   针对VLM推测解码（speculative decoding）中草稿模型难以处理冗余视觉token的问题，提出ViSpec框架，通过视觉适配器压缩图像token+全局视觉特征注入+合成训练数据，首次在VLM推测解码中实现了显著加速（最高3.22×）。

**[Visual Instruction Bottleneck Tuning](multimodal_vlm/visual_instruction_bottleneck_tuning.md)**

:   首次将信息瓶颈（IB）原理应用于多模态大语言模型的端到端指令微调，提出Visual Instruction Bottleneck Tuning（Vittle），在LLM内部插入轻量瓶颈层学习最小充分表征，在30种分布偏移场景下一致提升鲁棒性，同时不牺牲标准基准性能。

**[Visual Structures Helps Visual Reasoning Addressing The Binding Problem In Vlms](multimodal_vlm/visual_structures_helps_visual_reasoning_addressing_the_binding_problem_in_vlms.md)**

:   提出 VISER（Visual Input Structure for Enhanced Reasoning），通过在图像上叠加等距水平线+数字标注构建空间分区，配合"逐行扫描"文本指令，将 LVLM 的并行视觉处理转化为串行逐区域解析，在不修改模型、不训练、单次查询的条件下，大幅缓解绑定问题并提升计数、视觉搜索、场景描述、空间关系等视觉推理性能。

**[Vt-Fsl Bridging Vision And Text With Llms For Few-Shot Learning](multimodal_vlm/vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)**

:   提出VT-FSL框架，通过跨模态迭代提示（CIP）联合利用类名和支持图像驱动LLM生成精确文本描述并零样本合成语义一致图像，再通过核化体积对比学习（CGA）实现全局非线性跨模态对齐，在10个少样本学习基准上平均提升4.2%分类准确率。

**[Watch And Listen Understanding Audio-Visual-Speech Moments With Multimodal Llm](multimodal_vlm/watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)**

:   提出 TriSense——一个三模态（视觉+音频+语音）大语言模型，通过 Query-Based Connector 自适应调节各模态权重实现鲁棒的视频时序理解，并构建了包含200万标注的 TriSense-2M 数据集支撑训练。

**[Wearvqa A Visual Question Answering Benchmark For Wearables In Egocentric Authen](multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)**

:   提出 WearVQA，首个专为可穿戴设备（智能眼镜）场景设计的 VQA 基准，包含 2520 个第一人称视角图像-问答三元组，系统覆盖 7 个视觉领域、10 种认知任务类型和 6 类可穿戴特有的图像质量问题，配套 96% 准确率的 LLM-as-a-judge 评估框架，揭示当前 SOTA 多模态模型在此场景下仅达 24-52% 准确率。

**[What Can Rl Bring To Vla Generalization An Empirical Study](multimodal_vlm/what_can_rl_bring_to_vla_generalization_an_empirical_study.md)**

:   本文系统研究了RL微调对VLA（视觉-语言-动作）模型泛化能力的影响，发现PPO是最有效的RL算法且显著优于DPO和GRPO，RL在语义理解和执行鲁棒性方面的OOD泛化远超SFT，同时在视觉鲁棒性上与SFT持平。

**[When One Modality Sabotages The Others A Diagnostic Lens On Multimodal Reasoning](multimodal_vlm/when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)**

:   提出"模态破坏"（modality sabotage）这一诊断性失败模式概念，设计轻量级、模型无关的评估层，将每个模态视为独立代理并通过简单融合暴露"贡献者"与"破坏者"，在多模态情感识别任务上揭示了系统性的模态可靠性差异。

**[When Semantics Mislead Vision Mitigating Large Multimodal Models Hallucinations ](multimodal_vlm/when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)**

:   发现大多模态模型（LMMs）在场景文字识别中存在"语义幻觉"问题（将无语义文本误识为语义合理的词），分析发现注意力集中于文本区域的Transformer层更不易幻觉，据此提出训练无关的ZoomText+Grounded Layer Correction框架，在TextHalu-Bench上提升约4-5%，在ST-VQA上提升约4%。

**[Zero-Shot Robustness Of Vision Language Models Via Confidence-Aware Weighting](multimodal_vlm/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)**

:   提出 CAW（Confidence-Aware Weighting），一种针对CLIP模型的对抗微调损失函数，通过置信度感知加权重点关注困难对抗样本，结合特征对齐正则化保留预训练语义知识，在AutoAttack下实现零样本鲁棒性SOTA，且内存占用更低。

---

## 📦 模型压缩 { #model_compression }

**[3Did Direct 3D Inverse Design For Aerodynamics With Physics-Aware Optimization](model_compression/3did_direct_3d_inverse_design_for_aerodynamics_with_physics-aware_optimization.md)**

:   提出 3DID 框架，通过学习物理-几何统一的三平面隐空间表示 + 目标梯度引导扩散采样 + 拓扑保持精炼的两阶段策略，从随机噪声开始直接在完整 3D 空间中进行逆向设计，在车辆气动外形优化上，模拟阻力（Sim-Drag）相比最优基线降低 13.6%。

**[4Dgcpro Efficient Hierarchical 4D Gaussian Compression For P](model_compression/4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)**

:   提出层级化的4D高斯压缩框架4DGCPro，通过感知加权的层级高斯表示、运动感知自适应分组和端到端熵优化训练，在单一模型内实现多码率渐进式体积视频流媒体，可在移动设备上实时解码和渲染，RD性能超越现有SOTA。

**[A-Thought Efficient Reasoning Via Bidirectional Compression For Low-Resource Set](model_compression/a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)**

:   提出 A*-Thought——基于 A* 搜索算法的 CoT 压缩框架，通过双向重要性评分（BIS）衡量每个推理步骤对问题和答案的相关性，结合路径级 A* 搜索在指数级搜索空间中高效找到最紧凑的推理路径，在 512 token 预算下将 QwQ-32B 准确率提升 2.39 倍，在 4096 token 预算下减少约 50% 输出 token 且几乎不损失准确率。

**[A Granular Study Of Safety Pretraining Under Model Abliteration](model_compression/a_granular_study_of_safety_pretraining_under_model_abliteration.md)**

:   本文系统地研究了 model abliteration（一种推理时激活空间编辑攻击）对不同数据驱动安全预训练阶段的影响，发现仅依赖 refusal 训练的安全机制极易被攻破，而 **组合多种安全信号**（safe-only 过滤 + 改写 + metatag + refusal）可使安全行为分散到更广泛的表征空间、从而更难被单一方向投影移除。

**[A Partition Cover Approach To Tokenization](model_compression/a_partition_cover_approach_to_tokenization.md)**

:   将分词（tokenization）问题重新建模为**分区覆盖（partition cover）**优化问题，证明其为NP-hard，并提出多项式时间的贪心算法GreedTok，在压缩率和1B参数LLM预训练下游任务上均优于BPE。

**[A Token Is Worth Over 1000 Tokens Efficient Knowledge Distillation Through Low-R](model_compression/a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)**

:   提出 Low-Rank Clone (LRC)，通过可学习低秩投影矩阵将 teacher 权重压缩为 student 权重（软剪枝），同时对齐 attention 和 FFN 的中间激活（激活克隆），仅用 20B tokens 训练的 1.7B 模型即超过用 36T tokens 训练的 Qwen3-1.7B（64.98 vs 63.17），实现 **1000 倍训练效率提升**。

**[Accurate And Efficient Low-Rank Model Merging In Core Space](model_compression/accurate_and_efficient_low-rank_model_merging_in_core_space.md)**

:   提出 Core Space Merging 框架——通过在低秩 LoRA 矩阵的公共参考基空间中进行模型合并，**无信息损失**地将合并操作从 $m \times n$ 全尺寸空间压缩到 $Tr \times Tr$ 紧凑空间（$T$ 为任务数，$r$ 为 LoRA 秩），在 Llama 3 8B 上达到 SOTA 合并精度同时计算成本降低数个数量级。

**[Adaptive Kernel Design For Bayesian Optimization Is A Piece Of Cake With Llms](model_compression/adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)**

:   提出 CAKE (Context-Aware Kernel Evolution)，利用 LLM 作为遗传算法的交叉和变异算子，在贝叶斯优化过程中自适应地生成和进化 GP 核函数表达式，结合 BAKER 排序机制平衡模型拟合（BIC）与期望改进（EI），在超参数优化、控制器调参和光子芯片设计等任务上持续超越固定核和自适应核基线。

**[Adaptive Predictionpowered Autoeval With Reliability And Eff](model_compression/adaptive_predictionpowered_autoeval_with_reliability_and_eff.md)**

:   提出R-AutoEval+框架，通过在testing-by-betting框架中引入自适应权重机制动态调节对LLM评判器合成数据的依赖程度，首次在有限样本下同时保证评估可靠性和采样效率不低于仅用真实数据的方法，在LLM量化、prompt选择和推理预算分配三个场景中验证了理论优势。

**[Adaptive Stochastic Coefficients For Accelerating Diffusion Sampling](model_compression/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)**

:   通过理论分析 ODE 和 SDE 求解器的互补弱点（ODE 积累不可消除的梯度误差，SDE 在少步时离散化误差放大），提出 AdaSDE——在每个去噪步引入可学习随机系数 $\gamma_i$ 控制噪声注入强度，通过轻量蒸馏优化，在 5 NFE 下实现 CIFAR-10 FID 4.18、FFHQ FID 8.05 的 SOTA。

**[Admtree Compressing Lengthy Context With Adaptive Semantic Trees](model_compression/admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)**

:   提出 AdmTree——一种自适应层次化上下文压缩框架,通过信息密度驱动的动态分段构建叶 gist token，再用二叉语义树底向上聚合实现多粒度语义保留，解决了显式方法丢失局部细节和隐式方法位置偏差的双重问题,在 LongBench 上比 SOTA 基线 Activation Beacon 高 10%+。

**[Ai-Generated Video Detection Via Perceptual Straightening](model_compression/ai-generated_video_detection_via_perceptual_straightening.md)**

:   提出 ReStraV 方法，基于"感知拉直"假说（真实视频在神经表示空间形成更直的轨迹），利用 DINOv2 特征空间中的时间曲率和步距统计量训练轻量分类器检测 AI 生成视频，在 VidProM 上达到 97.17% 准确率和 98.63% AUROC，推理仅需 ~48ms。

**[Atlas Autoformalizing Theorems Through Lifting Augmentation And Synthesis Of Dat](model_compression/atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)**

:   ATLAS 提出了一个基于概念仓库、专家迭代+知识蒸馏、以及两种新颖增强策略的数据生成框架，构建了117K定理陈述的平行语料库，微调 Llama3.1-8B-Instruct 后在所有自动形式化基准上达到 SOTA。

**[Autodiscovery Open-Ended Scientific Discovery Via Bayesian Surprise](model_compression/autodiscovery_open-ended_scientific_discovery_via_bayesian_surprise.md)**

:   AutoDiscovery 提出用贝叶斯惊奇度（Bayesian Surprise）作为开放式科学发现的客观奖励信号——通过 LLM 采样估计先验/后验信念分布的 KL 散度，配合 MCTS+渐进展宽在假设空间中探索，在 21 个真实数据集上比贪心/束搜索产生 5-29% 更多的惊奇发现，人类评估确认贝叶斯惊奇度与专家"惊讶感"的一致性（0.67）远超 LLM 自身评估的"新颖性"和"有用性"。

**[Autojudge Judge Decoding Without Manual Annotation](model_compression/autojudge_judge_decoding_without_manual_annotation.md)**

:   AutoJudge 自动化了 Judge Decoding 中"重要 token"的标注——通过半贪心搜索替换不匹配 token 并检查答案是否改变来标注重要性，训练逻辑回归分类器预测 token 重要性，使投机解码每轮接受 40+ token（vs 标准 ~20），在 GSM8K 上加速 1.5× 且准确率损失 <1%。

**[Barista Brain Scale Informed Spatiotemporal Representation Of Human Intracranial](model_compression/barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)**

:   BaRISTA 系统探索 iEEG Transformer 的空间编码尺度（电极/脑区/脑叶），发现脑区级编码 + 空间掩码重建在语言任务解码上达 86.2% AUC（vs PopT 79.5%），编码尺度选择的影响 > 掩码策略选择，且跨被试泛化性好。

**[Benfords Curse Tracing Digit Bias To Numerical Hallucination In Llms](model_compression/benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)**

:   本文发现 LLM 的数值幻觉根源于预训练语料中符合 Benford 定律的数字频率分布——数字 1 出现概率 ~30% 而数字 9 仅 ~5%，这种偏差被 FFN 后期层的特定"数字选择性神经元"内化，提出数字选择性分数（DSC）定位偏差神经元并通过剪枝 0.01% 的神经元修正 1.36-3.49% 的错误预测。

**[Beyond Higher Rank Token-Wise Input-Output Projections For Efficient Low-Rank Ad](model_compression/beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)**

:   TopLoRA 从输入-输出投影角度分析 LoRA 的表达能力，发现所有 token 共享同一投影矩阵是关键瓶颈，提出通过可学习的 token 级对角矩阵 $\Sigma_X$ 动态调整 LoRA 权重（$\Delta W_X = B\Sigma_X A$），在不增加秩的前提下实现细粒度适配，跨任务一致优于 LoRA 2-3%。

**[Beyond Random Automatic Inner-Loop Optimization In Dataset Distillation](model_compression/beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)**

:   提出 AT-BPTT（自适应截断 BPTT），将 DNN 训练分为早/中/晚三阶段并自适应调整截断策略和窗口大小，在 CIFAR-10/100/Tiny-ImageNet/ImageNet-1K 上平均提升 3-17%，同时实现 3.9× 加速和 63% 内存节省。

**[Bezier Splatting For Fast And Differentiable Vector Graphics Rendering](model_compression/bezier_splatting_for_fast_and_differentiable_vector_graphics_rendering.md)**

:   Bézier Splatting 将 Gaussian Splatting 框架与 Bézier 曲线结合，沿曲线均匀采样 2D Gaussian 点，通过 α-blending 渲染实现可微矢量图形，前向 30× / 反向 150× 加速（相比 DiffVG），同时保持或超越 LIVE 等方法的图像质量。

**[Binary Quadratic Quantization Beyond First-Order Quantization For Real-Valued Ma](model_compression/binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)**

:   BQQ 提出二次二值量化——用二值矩阵的乘积（而非线性组合）表示权重矩阵，突破传统一阶量化的表达能力限制，通过扩展 AMFD（退火均场下降）到 PUBO 问题求解混合整数优化，在 2-bit 无数据 ViT 量化上实现从 10.83% 到 58.25% 的准确率飞跃。

**[Biobench A Blueprint To Move Beyond Imagenet For Scientific Ml Benchmarks](model_compression/biobench_a_blueprint_to_move_beyond_imagenet_for_scientific_ml_benchmarks.md)**

:   提出 BioBench——一个统一 9 个生态视觉任务、4 个分类界、6 种图像模态、310 万张图像的基准，证明 ImageNet top-1 准确率仅解释 34% 的生态任务方差，在 >75% 精度的前沿模型中 30% 的排名是错误的。

**[C-Lora Contextual Low-Rank Adaptation For Uncertainty Estimation In Large Langua](model_compression/c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)**

:   提出 C-LoRA，通过引入轻量级上下文模块使 LoRA 低秩矩阵的分布依赖于输入数据，实现样本级的异方差不确定性估计，在少样本微调场景中显著改善校准质量。

**[Casspec Cascade Adaptive Selfspeculative Decoding For Onthef](model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)**

:   CAS-Spec 通过 Dynamically Switchable Inference Acceleration (DSIA) 策略（如不同程度的 layer sparsity）从目标模型自身构建多级 draft 模型层级，配合 Dynamic Tree Cascade (DyTC) 算法基于在线 acceptance rate 和延迟预测自适应路由 draft 模型和分配 draft 长度，在完全 training-free 的条件下实现 1.1×-2.3× 的无损推理加速，DyTC 比 cascade 和 tree baseline 分别提升 47% 和 48%。

**[Chunkkv Semanticpreserving Kv Cache Compression For Efficien](model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)**

:   ChunkKV 将 KV cache 压缩的基本单元从离散 token 提升为语义 chunk（连续 token 组），通过 chunk 级 attention score 聚合来选择保留哪些语义完整的片段，并利用 chunk 带来的高跨层索引相似性实现 layer-wise index reuse，在 10% 压缩率下比 SnapKV/PyramidKV 提升最高 8.7%，吞吐量提升 26.5%。

**[Codegemm A Codebook-Centric Approach To Efficient Gemm In Quantized Llms](model_compression/codegemm_a_codebook-centric_approach_to_efficient_gemm_in_quantized_llms.md)**

:   提出 CodeGEMM，一种以 codebook 为中心的 GEMM kernel，通过预计算 centroid 与 activation 的内积并缓存为 Psumbook，替代传统反量化流程，在 2-bit 量化 LLM 上实现 1.83×（8B）到 8.93×（70B）的端到端加速。

**[Correlation Dimension Of Auto-Regressive Large Language Models](model_compression/correlation_dimension_of_auto-regressive_large_language_models.md)**

:   本文将分形几何中的关联维度（Correlation Dimension）引入LLM分析，通过度量next-token对数概率向量之间的递归结构来量化文本的层次化复杂度，揭示了LLM预训练的三阶段演化、幻觉倾向指示以及多种文本退化模式的统一检测能力——这些是困惑度（perplexity）无法捕捉的。

**[Curiosity-Driven Rl For Symbolic Equation Solving](model_compression/curiosity-driven_rl_for_symbolic_equation_solving.md)**

:   将好奇心驱动探索（RND、ICM 等）与基于表达式树的图动作空间结合，使 PPO 智能体能够求解包含根号、指数和三角函数的非线性方程，超越了此前仅限于线性方程的 RL 方法。

**[Data Efficient Adaptation In Large Language Models Via Continuous Low-Rank Fine-](model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)**

:   提出 DEAL 框架，通过小波核特征过滤保留 LoRA 低秩矩阵中的历史知识核心特征，结合受控知识更新模块和非对称正则化，实现 LLM 在小样本持续微调中学新不忘旧。

**[Deltaflow An Efficient Multi-Frame Scene Flow Estimation Method](model_compression/deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)**

:   提出 DeltaFlow (ΔFlow)，通过体素帧间差分（Δ scheme）提取运动线索，实现特征尺寸不随帧数增长的多帧场景流估计，在 Argoverse 2/Waymo/nuScenes 上达到 SOTA 且比次优多帧方法快 2 倍。

**[Dense Backpropagation Improves Training For Sparse Mixture-Of-Experts](model_compression/dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)**

:   提出 Default MoE 方法，用指数移动平均（EMA）近似非激活 expert 的输出，使 MoE router 获得稠密梯度更新，在不显著增加计算开销的情况下提升稀疏 MoE 的训练性能。

**[Dependency Parsing Is More Parameter-Efficient With Normalization](model_compression/dependency_parsing_is_more_parameter-efficient_with_normalization.md)**

:   揭示依存句法/语义分析中 biaffine scoring 缺乏归一化导致模型过参数化，通过简单的 $1/\sqrt{d}$ 缩放即可在减少高达 85% BiLSTM 参数的同时匹配甚至超越原始性能。

**[Deterministic Continuous Replacement Fast And Stable Module Replacement In Pretr](model_compression/deterministic_continuous_replacement_fast_and_stable_module_replacement_in_pretr.md)**

:   DCR 通过确定性退火权重 α(t) 混合 teacher 和 student 模块输出，消除了随机门控（如 BERT-of-Theseus）带来的梯度方差，在冷启动模块替换场景下实现更快收敛和更强的特征对齐。

**[Disentangling Latent Shifts Of In-Context Learning With Weak Supervision](model_compression/disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)**

:   WILDA 将 ICL 视为弱监督信号，用 teacher-student 框架将示例引发的潜在偏移编码进轻量 LoRA 适配器，实现无需重复 prompting 的高效推理，且 student 通过伪标签修正和覆盖扩展超越 teacher（弱到强泛化）。

**[Dp-Llm Runtime Model Adaptation With Dynamic Layer-Wise Precision Assignment](model_compression/dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)**

:   DP-LLM 发现每层的量化敏感度在解码步间动态变化，提出基于 relative error 的动态逐层精度选择机制，在运行时根据输入为每层分配精度（h-bit 或 l-bit），实现了优于静态混合精度的性能-延迟权衡。

**[Dragon Guard Llm Unlearning In Context Via Negative Detection And Reasoning](model_compression/dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)**

:   DRAGON 提出无需微调基座模型的系统性 LLM 遗忘框架：通过双层检测模块识别需遗忘的 prompt，再由专门微调的 guard 模型生成 CoT 推理指令实现上下文干预，在保持模型通用能力的同时有效删除隐私/有害知识。

**[Duogpt Training-Free Dual Sparsity Through Activation-Aware Pruning In Llms](model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)**

:   提出 DuoGPT，一种将激活稀疏（activation sparsity）重新解释为动态结构化权重稀疏、并与非结构化权重剪枝相结合的双稀疏（dual-sparse）框架，通过扩展 OBC 框架引入激活感知校准和稠密模型输出残差修正项，在不需要重训练的情况下实现 LLM 解码阶段的显著加速与内存节省。

**[Elastic Vits From Pretrained Models Without Retraining](model_compression/elastic_vits_from_pretrained_models_without_retraining.md)**

:   SnapViT 提出一种后训练结构化剪枝方法：结合自监督梯度的局部 Hessian 和进化算法估计的全局跨模块相关性，无需重训练或标签即可在一次运行中生成连续稀疏度的弹性 ViT 子网络，在 A100 上仅需不到 5 分钟。

**[Emloc Emulator-Based Memory-Efficient Fine-Tuning With Lora Correction](model_compression/emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)**

:   EMLoC 通过对原始模型做 activation-aware SVD 构建轻量级 emulator 进行 LoRA 微调，并提出 LoRA 校正算法弥补 emulator 与原模型的不对齐，使得微调内存开销降至与推理持平，在单张 24GB GPU 上即可微调 38B 模型。

**[Enhancing Semi-Supervised Learning With Zero-Shot Pseudolabels](model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)**

:   ZeroMatch 提出两阶段框架将基础模型的零样本伪标签与半监督学习相结合：先用知识蒸馏初始化学生模型，再以辅助 KD loss 防止灾难性遗忘的方式执行 SSL 训练，在 6 个视觉/NLP 基准上一致超越标准 SSL 和零样本增强方法。

**[Exact Expressive Power Of Transformers With Padding](model_compression/exact_expressive_power_of_transformers_with_padding.md)**

:   本文精确刻画了带 padding 的 Transformer 的表达能力：固定深度 + 多项式 padding 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^0$，进一步结合 $O(\log^d n)$ looping 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^d$，polylog looping 收敛到 $\mathsf{NC}$，为 padding/looping 作为可并行推理时计算提供了完整理论基础。

**[Expo Unlocking Hard Reasoning With Self-Explanation-Guided Reinforcement Learnin](model_compression/expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)**

:   提出 Self-Explanation Policy Optimization (ExPO)，一种通过让模型在给定正确答案条件下自主生成推理链（self-explanation）作为正样本的模块化框架，解决 GRPO 等 RL 后训练方法在困难推理任务上因缺乏有效正样本而无法学习（分布锐化）的根本问题——ExPO 生成的自解释样本既在当前策略分布内（in-distribution），又能提供正向学习信号，可无缝集成到 DPO 和 GRPO 中。

**[Eyes Wide Open Ego Proactive Videollm For Streaming Video](model_compression/eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)**

:   定义"第一视角流式视频主动理解"新任务——给定ego-streaming视频，AI助手在恰当时机主动回答多样化、随事件演变的问题，同时保持感知与推理的同步。提出ESTP-Bench评估框架、ESTP-F1指标，以及含数据引擎、多阶段训练和主动动态压缩的完整技术pipeline（VideoLLM-EyeWO），在ESTP-Bench上比最强baseline MiniCPM-V高11.8%。

**[Falqon Accelerating Lora Fine-Tuning With Low-Bit Floating-Point Arithmetic](model_compression/falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)**

:   FALQON 通过将 LoRA 适配器直接融合 (meld) 到 FP8 量化的骨干权重中，消除了单独 LoRA 路径引入的小矩阵量化开销，结合高效梯度计算和行级代理更新机制，实现了相比现有量化 LoRA 方法约 3 倍的训练加速。

**[Fastlongspeech Enhancing Large Speech-Language Models For Efficient Long-Speech ](model_compression/fastlongspeech_enhancing_large_speech-language_models_for_efficient_long-speech_.md)**

:   提出 FastLongSpeech，通过迭代融合策略压缩冗余语音表征和动态压缩训练转移短语音能力到长语音场景，使 LSLM 无需长语音训练数据即可高效处理长语音，在长语音 QA 上实现最优性能且推理效率提升 70%。

**[Find Your Needle Small Object Image Retrieval Via Multi-Object Attention Optimiz](model_compression/find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)**

:   MaO 提出了一种针对小目标图像检索（SoIR）的新方法，通过多目标预训练和基于注意力的特征优化，将多个目标的表示融合为单一全局描述符，在多个基准上大幅超越现有检索方法。

**[Fira Can We Achieve Full-Rank Training Of Llms Under Low-Rank Constraint](model_compression/fira_can_we_achieve_full-rank_training_of_llms_under_low-rank_constraint.md)**

:   提出 Fira，首个在低秩约束下实现全秩训练（全秩梯度+全秩权重）的 LLM 训练框架，通过观察到低秩与全秩训练中优化器的缩放因子高度相似，用低秩缩放因子近似校正子空间外梯度，配合 norm-growth limiter 防止 loss spike，在预训练和微调中均超越 LoRA 和 GaLore。

**[Gated Integration Of Low-Rank Adaptation For Continual Learning Of Large Languag](model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)**

:   提出 GainLoRA，为持续学习中每个新任务的 LoRA 分支引入**门控模块**生成自适应集成系数，通过正交约束使新分支对旧任务的输出趋近于零，从而有效缓解灾难性遗忘。

**[Geometric Data Valuation Via Leverage Scores](model_compression/geometric_data_valuation_via_leverage_scores.md)**

:   提出基于**统计杠杆分数（leverage scores）**的几何数据估值方法，作为 Data Shapley 值的高效代理，满足对称性、效率性和虚拟玩家等公理，并通过 ridge leverage 扩展解决维度饱和问题，提供 $O(\varepsilon)$ 近似最优的理论保证。

**[Geometry Of Decision Making In Language Models](model_compression/geometry_of_decision_making_in_language_models.md)**

:   通过在 28 个开源 Transformer 模型上大规模测量各层隐藏表示的**内在维度（Intrinsic Dimension, ID）**，揭示了一致的"低-高-低"维度变化模式：早期层在低维流形上操作，中间层扩展空间，后期层再压缩至与决策相关的低维表示。

**[Global Minimizers Of Ellp-Regularized Objectives Yield The Sparsest Relu Neural ](model_compression/global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)**

:   证明了对于单隐层 ReLU 网络，最小化 $\ell^p$（$0 < p < 1$）路径范数的全局最优解恰好对应于**最稀疏**的数据插值网络，从而将组合优化的稀疏插值问题重新表述为连续可微的优化任务。

**[Gora Gradient-Driven Adaptive Low Rank Adaptation](model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)**

:   提出 GoRA，利用**预计算梯度信息**在训练前同时完成自适应秩分配和权重初始化——基于参数敏感度分配各层 rank，用梯度伪逆初始化 $B$ 矩阵使初始输出近似一步梯度下降，统一解决 LoRA 的两大瓶颈。

**[Graph Your Own Prompt](model_compression/graph_your_own_prompt.md)**

:   提出图一致性正则化（GCR）框架，通过在网络任意深度插入无参数的图一致性层（GCL），将中间特征的关系图与基于预测的类感知语义图对齐，以自我提示的方式促进语义一致的特征学习，在不修改架构和不增加参数的前提下提升分类泛化性能。

**[Grass Scalable Data Attribution With Gradient Sparsification And Sparse Projecti](model_compression/grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)**

:   提出 GraSS 与 FactGraSS 两阶段梯度压缩算法，利用逐样本梯度的固有稀疏性实现**亚线性**时间与空间复杂度（$O(k')$），在十亿参数模型上比 SOTA 基线 LoGra 快 **165%**，同时保持数据归因质量。

**[Graver Generative Graph Vocabularies For Robust Graph Foundation Models Fine-Tun](model_compression/graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)**

:   提出 Graver 框架，通过 ego-graph 解耦提取可迁移子图词汇、graphon 专家建模词汇分布、MoE-CoE 路由选择性增强 support 样本，解决 GFM 少样本微调中因结构不匹配导致的不稳定性问题。

**[Hankel Singular Value Regularization For Highly Compressible State Space Models](model_compression/hankel_singular_value_regularization_for_highly_compressible_state_space_models.md)**

:   通过在训练中正则化 SSM 层的 **Hankel 奇异值核范数**促使其快速衰减，使训练后模型可用平衡截断压缩至原始阶数的 **10%** 而保持精度，并利用旋转矩阵块对角参数化将 Gramian 计算从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(n^2)$。

**[Heterogeneous Adversarial Play In Interactive Environments](model_compression/heterogeneous_adversarial_play_in_interactive_environments.md)**

:   提出 **HAP（Heterogeneous Adversarial Play）**，将教师-学生交互形式化为极小极大博弈：教师网络自动生成针对学生弱点的挑战任务，学生策略不断适应进化，形成无需手工设计的自适应课程——在多任务 RL 环境中超越 SOTA 基线，且生成的课程对人类学习者同样有效。

**[Homogeneous Keys Heterogeneous Values Exploiting Local Kv Cache Asymmetry For Lo](model_compression/homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)**

:   发现 LLM 注意力机制中一个被忽视的**局部 Key-Value 不对称性**——相邻 Key 具有同质性（相似注意力权重），而相邻 Value 呈异质分布——据此提出 **AsymKV** 无训练压缩框架：基于同质性合并 Key + 基于基数归一化的**无损** Value 表示，在 LongBench 上超越 H2O 达 **5 分**。

**[Hyperbolic Dataset Distillation](model_compression/hyperbolic_dataset_distillation.md)**

:   提出 HDD 方法，首次将双曲空间引入数据集蒸馏，通过在 Lorentz 双曲空间中匹配原始和合成数据的 Riemannian 质心来替代欧氏空间的分布匹配，利用双曲几何的层级加权特性让"更具代表性"的底层样本获得更高权重，在多个数据集上持续提升 DM/IDM 基线准确率。

**[Hyperbolic Fine-Tuning For Large Language Models](model_compression/hyperbolic_fine-tuning_for_large_language_models.md)**

:   发现 LLM token 嵌入具有幂律分布和树状双曲结构，据此提出 HypLoRA——在 Lorentz 双曲流形上直接执行低秩适配（避免切空间映射的相消效应），在算术推理和常识推理任务上相比标准 LoRA 取得显著提升（如 Qwen2.5-7B 上 M.AVG +7.5%）。

**[Inference-Time Hyper-Scaling With Kv Cache Compression](model_compression/inference-time_hyper-scaling_with_kv_cache_compression.md)**

:   提出"推理时超缩放"（Inference-Time Hyper-Scaling）范式：通过高效压缩KV缓存，在相同计算/内存预算下生成更长或更多并行推理序列，显著提升推理模型在数学、代码、科学推理等任务上的准确率。

**[Infrequent Exploration In Linear Bandits](model_compression/infrequent_exploration_in_linear_bandits.md)**

:   提出 INFEX 框架，按给定调度表在探索步执行基线算法（如 LinUCB/LinTS）、其余时刻贪心选臂，证明只要探索次数超过 $\omega(\log T)$ 即可达到与全时刻探索相同的多项对数 regret，同时大幅降低计算开销（80%-99% 时间步为贪心）。

**[Jet-Nemotron Efficient Language Model With Post Neural Architecture Search](model_compression/jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)**

:   NVIDIA 提出 PostNAS 流水线——从预训练全注意力模型出发，冻结 MLP 权重，通过四步搜索（全注意力层放置→线性注意力块选择→新注意力块 JetBlock 设计→硬件感知超参搜索）得到混合架构 Jet-Nemotron，2B 模型在 MMLU-Pro 上超越 Qwen3-1.7B 同时生成吞吐提升 47×。

**[Keydiff Key Similarity-Based Kv Cache Eviction For Long-Context Llm Inference In](model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)**

:   提出 KeyDiff——一种无需注意力分数的 KV cache 驱逐策略，通过保留与其他 key 余弦相似度最低（即几何上最独特）的 key 来维护 cache，在严格内存约束的逐块推理场景下以 8K cache 在 LongBench 上仅损失 ≤0.04% 精度，同时端到端推理延迟减少最高 30%。

**[Kindle Knowledge-Guided Distillation For Prior-Free Gene Regulatory Network Infe](model_compression/kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)**

:   提出 KINDLE 三阶段框架，通过知识蒸馏将先验引导的教师模型中学到的基因调控知识迁移到无先验的学生模型，在不依赖任何外部先验知识的情况下实现了基因调控网络（GRN）推断的 SOTA 性能。

**[Ktae A Model-Free Algorithm To Key-Tokens Advantage Estimation In Mathematical R](model_compression/ktae_a_model-free_algorithm_to_key-tokens_advantage_estimation_in_mathematical_r.md)**

:   KTAE 提出了一种不依赖额外模型的 token 级优势估计算法，通过 Fisher 精确检验和信息增益量化每个 token 与正确推理结果的统计关联，将细粒度 token 重要性叠加到 GRPO/DAPO 的 rollout 级优势上，在5个数学推理基准上超越基线并显著缩短生成长度。

**[Kvzip Query-Agnostic Kv Cache Compression With Context Reconstruction](model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)**

:   提出 KVzip，一种查询无关的 KV Cache 驱逐方法，通过利用 LLM 自身从缓存的 KV 对中重建原始上下文来量化每个 KV 对的重要性，实现 3-4× 的 KV Cache 压缩率和约 2× 的 FlashAttention 解码延迟降低，同时在多查询场景下显著优于现有查询感知方法。

**[Layerif Estimating Layer Quality For Large Language Models Using Influence Funct](model_compression/layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)**

:   LayerIF 提出用影响函数（Influence Functions）逐层量化 LLM 的训练质量，通过聚合各层的正向影响分数得到数据驱动的层重要性估计，并将其应用于 LoRA-MoE 专家分配和层级稀疏剪枝两个下游任务，在 Mistral-7B 和 Gemma-7B 上分别获得 1.61% 和 0.90% 的准确率提升。

**[Learning Grouped Lattice Vector Quantizers For Low-Bit Llm Compression](model_compression/learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)**

:   GLVQ 提出为 LLM 权重的每个分组学习专属的格（lattice）码本（由可学习生成矩阵定义），配合分组特异的 μ-law companding 变换适应重尾分布，在 2-bit 量化下 Llama-2-70B 的 Wikitext-2 困惑度达到 3.36，大幅领先 QuIP#（3.91）和 QTIP（3.78）。

**[Learning To Better Search With Language Models Via Guided Reinforced Self-Traini](model_compression/learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)**

:   提出 Guided-ReST，通过将最优解作为子目标逐步融入模型自生成的搜索轨迹中，生成高质量训练数据并蒸馏更高效的搜索策略，在Countdown和代码自修复任务上显著提升搜索效率和准确率。

**[Learning To Factorize And Adapt A Versatile Approach Toward Universal Spatio-Tem](model_compression/learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)**

:   提出 FactoST-v2，一个因式分解的时空基础模型框架，将通用时间预训练与领域特定空间适配解耦，以线性复杂度实现跨领域零样本/少样本/全样本时空预测。

**[Less Is More But Where Dynamic Token Compression Via Llm-Guided Keyframe Prior](model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)**

:   提出 DyToK，一种无需训练的视频 token 动态压缩方法，利用 VLLM 深层注意力中固有的 query 条件关键帧先验，为不同帧自适应分配 token 预算，实现即插即用式的效率-精度最优权衡。

**[Linear Attention For Efficient Bidirectional Sequence Modeling](model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)**

:   提出 Lion 框架，首次系统性地将线性 Transformer 扩展到双向序列建模，统一了完整线性注意力、双向 RNN 和分块并行三种等价表示形式，训练速度比 SSM 快 10 倍且匹配 softmax Transformer 性能。

**[Littlebit Ultra Low-Bit Quantization Via Latent Factorization](model_compression/littlebit_ultra_low-bit_quantization_via_latent_factorization.md)**

:   提出 LittleBit 框架，通过低秩潜空间矩阵分解 + 二值化 + 多尺度补偿机制，实现低至 0.1 BPW（每权重比特）的极端 LLM 压缩，将 Llama2-13B 压缩到不足 0.9GB，在子1比特领域大幅超越 STBLLM。

**[Loquetier A Virtualized Multi-Lora Framework For Unified Llm Fine-Tuning And Ser](model_compression/loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)**

:   提出Loquetier框架，通过虚拟化模块（Virtualized Module）和分段多LoRA乘法内核（SMLM），将多个LoRA适配器的微调和推理统一到单一运行时中，实现推理任务3.0×吞吐率提升和统一任务46.4×更高的SLO达成率。

**[Lt-Soups Bridging Head And Tail Classes Via Subsampled Model Soups](model_compression/lt-soups_bridging_head_and_tail_classes_via_subsampled_model_soups.md)**

:   提出 LT-Soups，一个两阶段模型融合框架，通过在不同不平衡比例的子采样数据上训练多个模型并进行权重平均，在长尾分布的全频谱上实现头部类和尾部类的均衡性能。

**[Matryoshka Pilot Learning To Drive Black-Box Llms With Llms](model_compression/matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)**

:   提出 Matryoshka Pilot (M-Pilot)，用轻量级白盒 LLM 作为控制器，通过生成中间引导（任务分解、高层计划、用户画像）来驱动黑盒 LLM 在推理、规划和个性化等复杂长程任务上的性能，并通过迭代 DPO 实现自我改进。

**[Memory-Efficient Training With In-Place Fft Implementation](model_compression/memory-efficient_training_with_in-place_fft_implementation.md)**

:   提出 rdFFT——首个真正原地（in-place）的实数域快速傅里叶变换框架，通过隐式复数编码方案消除中间缓冲区，实现训练时零额外内存开销的 FFT/IFFT 计算，内存效率最高提升 1500 倍以上。

**[Mitigating Semantic Collapse In Partially Relevant Video Retrieval](model_compression/mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)**

:   针对部分相关视频检索（PRVR）中的语义坍塌问题，提出文本相关性保持学习和跨分支视频对齐（CBVA）方法，在文本和视频嵌入空间中分别解决坍塌现象，显著提升检索准确率。

**[Mixture Of Noise For Pre-Trained Model-Based Class-Incremental Learning](model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)**

:   提出学习有益的"混合噪声"来抑制预训练模型在增量学习中的参数漂移，通过在任务间进行动态权重混合噪声实现 SOTA 性能，特别在 50 步增量设置下表现突出。

**[Modhifi Identifying High Fidelity Predictive Components For Model Modification](model_compression/modhifi_identifying_high_fidelity_predictive_components_for_model_modification.md)**

:   提出 Subset Fidelity 度量和 ModHiFi 框架，通过理论证明 Lipschitz 连续网络的局部重构误差线性上界全局误差，无需训练数据、损失函数或梯度，仅用合成数据即可识别模型中的高保真 (HiFi) 组件，统一实现结构化剪枝和类别遗忘两大任务。

**[Multi-Task Vehicle Routing Solver Via Mixture Of Specialized Experts Under State](model_compression/multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)**

:   提出 **State-Decomposable MDP (SDMDP)** 框架将多种 VRP 变体重新表述为基础状态空间的笛卡尔积，再通过 **Mixture-of-Specialized-Experts Solver (MoSES)** 用专用 LoRA 专家实现基础策略的潜在空间复用，高效处理 16 种 VRP 变体。

**[Mustafar Promoting Unstructured Sparsity For Kv Cache Pruning In Llm Inference](model_compression/mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)**

:   提出 MUSTAFAR 框架，系统性地证明了非结构化稀疏性在 KV 缓存剪枝中的优越性（Key 和 Value 均可达 70% 稀疏度且不损精度），并设计了基于 bitmap 的稀疏格式和自定义注意力内核，实现了端到端推理吞吐量 2.23 倍加速。

**[Navigating Simply Aligning Deeply Winning Solutions For Mouse Vs Ai 2025](model_compression/navigating_simply_aligning_deeply_winning_solutions_for_mouse_vs_ai_2025.md)**

:   在NeurIPS 2025 Mouse vs. AI竞赛中，本文展示了轻量级两层CNN在视觉鲁棒性任务上大幅超越深度网络的反直觉发现，同时证明深层ResNet架构在神经对齐任务上更具优势，揭示了行为鲁棒性与生物合理性之间的根本张力。

**[On The Creation Of Narrow Ai Hierarchy And Nonlocality Of Neural Network Skills](model_compression/on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)**

:   研究创建窄域（narrow）AI 系统面临的两大挑战：任务的层级依赖使得某些窄域技能必须在宽分布上训练才能学会；技能的非局部性使得剪枝无法精确分离想要保留和舍弃的能力——但剪枝+恢复训练仍优于蒸馏和从头训练。

**[On The Hardness Of Approximating Distributions With Tractable Probabilistic Mode](model_compression/on_the_hardness_of_approximating_distributions_with_tractable_probabilistic_mode.md)**

:   本文证明了用可处理概率模型（如分解概率电路）在有界f-散度下近似任意分布是NP-hard的，并证明了在近似建模条件下分解PC和（确定性+分解）PC之间存在指数级大小差距，揭示了近似放宽并不能缓解精确建模中的复杂度瓶颈。

**[One-Step Diffusion-Based Image Compression With Semantic Distillation](model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)**

:   提出OneDC——首个一步扩散生成式图像编解码器，将超先验（hyperprior）替代文本作为扩散模型的语义引导并通过语义蒸馏增强其表示能力，实现了比多步扩散编解码器节省39%码率、解码加速20倍的SOTA感知质量。

**[Online Mixture Of Experts No-Regret Learning For Optimal Collective Decision-Mak](model_compression/online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)**

:   提出在线专家混合（OMoE）框架，包含 UCB 逐步消除和在线加权多数投票两种算法，理论上保证无遗憾（no-regret），并应用于 LLM 专家的在线动态聚合。

**[Optimizing Distributional Geometry Alignment With Optimal Transport For Generati](model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)**

:   将数据集蒸馏重新表述为最优传输（OT）距离最小化问题，通过三阶段（OT引导扩散采样、标签-图像对齐软重标注、OT logit匹配）实现细粒度分布几何对齐，在ImageNet-1K IPC=10上比之前SOTA提升至少4%。

**[Order-Level Attention Similarity Across Language Models A Latent Commonality](model_compression/order-level_attention_similarity_across_language_models_a_latent_commonality.md)**

:   提出 Order-Level Attention (OLA)——对 Attention Rollout 的阶次分解，发现不同语言模型在同阶 OLA 上存在显著相似性 (OLAS)，并且 OLA 隐式编码了句法知识，基于此提出 TOA 实现首个无需训练的跨LM适配器迁移。

**[Paretoq Improving Scaling Laws In Extremely Low-Bit Llm Quantization](model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)**

:   提出 ParetoQ——首个统一 1/1.58/2/3/4 比特量化的框架，通过系统研究训练策略（全精度预训练 vs. QAT 分配）和量化函数设计（提出 SEQ 量化器），发现 2-bit 和 1.58-bit 量化在精度-模型大小折中上优于传统 4-bit，且各比特位宽均达到 SOTA。

**[Ppg-Distill Efficient Photoplethysmography Signals Analysis Via Foundation Model](model_compression/ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)**

:   PPG-Distill提出一种针对PPG信号的知识蒸馏框架，通过预测级、特征级和Patch级（形态+节律）蒸馏，将大型PPG基础模型的知识迁移到轻量学生模型，在保持性能（最高提升21.8%）的同时实现7倍推理加速和19倍内存压缩。

**[Q-Palette Fractional-Bit Quantizers Toward Optimal Bit Allocation For Efficient ](model_compression/q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)**

:   从信息论角度推导高斯化权重的最优比特分配,提出 Q-Palette 分数位量化器集合和混合方案量化框架,在 LLM 推理中实现近最优的量化性能和推理加速。

**[Qsvd Efficient Low-Rank Approximation For Unified Query-Key-Value Weight Compres](model_compression/qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)**

:   提出QSVD方法，通过对QKV联合权重矩阵的SVD分解共享下投影矩阵来减少KV缓存和计算开销，结合基于重要性评分的自适应秩分配和量化技术，在VLM上实现超过10%的精度提升且硬件成本更低。

**[Quadenhancer Leveraging Quadratic Transformations To Enhance Deep Neural Network](model_compression/quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)**

:   提出一种轻量级的二次增强器（QuadEnhancer），通过在每个线性层引入稀疏化的二次交互项，以极少的额外参数和计算开销显著提升现有神经网络架构的性能。

**[Quantization Error Propagation Revisiting Layer-Wise Post-Training Quantization](model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)**

:   识别现有逐层 PTQ 方法忽略量化误差跨层累积和增长的关键瓶颈，提出 QEP 框架通过误差传播和补偿显式纠正累积误差，在极低比特（INT2/INT3）下实现大幅性能提升。

**[Rat Bridging Rnn Efficiency And Attention Accuracy Via Chunk-Based Sequence Mode](model_compression/rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)**

:   提出 RAT（Recurrence And aTtention），一种基于 Chunk 的中间架构——在 Chunk 内使用线性 RNN 建模局部依赖、Chunk 间使用 softmax 注意力实现全局访问。L=16 时单层解码速度提升 9 倍、最大吞吐量提升 10 倍，且性能与标准注意力持平；与滑动窗口注意力交替使用的混合变体在几乎所有 benchmark 上最优。

**[Rccda Adaptive Model Updates In The Presence Of Concept Drift Under A Constraine](model_compression/rccda_adaptive_model_updates_in_the_presence_of_concept_drift_under_a_constraine.md)**

:   提出 RCCDA，一种基于 Lyapunov 漂移惩罚框架的轻量级模型更新策略，在数据分布随时间漂移（concept drift）场景下，仅利用历史推理损失信息和可调阈值，就能贪心最优地决定何时重训模型，同时可证明地满足严格资源预算约束。

**[Rectifying Soft-Label Entangled Bias In Long-Tailed Dataset Distillation](model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)**

:   揭示了长尾数据集蒸馏中软标签存在来自蒸馏模型和蒸馏图像的双重纠缠偏差，提出 ADSA 自适应软标签对齐模块，通过logit空间的后处理校准消除偏差，作为即插即用模块可无缝集成到现有蒸馏方法中，在 ImageNet-1k-LT 上将尾部类准确率提升高达11.8%。

**[Reflora Refactored Low-Rank Adaptation For Efficient Fine-Tuning Of Large Models](model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)**

:   RefLoRA 通过在每次迭代中选择最优的低秩分解形式（最小化损失上界），解决了 LoRA 因分解不唯一性导致的权重更新不一致和不平衡问题，在几乎不增加计算开销的前提下加速收敛并提升微调性能。

**[Reject Only Critical Tokens Pivot-Aware Speculative Decoding](model_compression/reject_only_critical_tokens_pivot-aware_speculative_decoding.md)**

:   PAD 提出了基于效用匹配（而非分布匹配）的推测解码新范式：训练一个轻量分类器识别"关键 token"（pivot token），仅拒绝会导致最终输出效用下降的 draft token，从而在 GSM8K 上实现 2.46× 加速且几乎不损失准确率。

**[Reordering Patches Improves Vision Models](model_compression/reordering_patches_improves_vision_models.md)**

:   揭示了视觉模型中 patch 排列顺序对长序列模型性能有显著影响，并提出 REOrder 框架通过信息论先验和强化学习自动发现最优 patch 排列，在 ImageNet-1K 上提升高达 3.01%，在 FMoW 上提升 13.35%。

**[Rep Resource-Efficient Prompting For Rehearsal-Free Continual Learning](model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)**

:   REP 通过轻量代理模型的快速提示选择、自适应 Token 合并（AToM）和自适应层丢弃（ALD）三种互补技术，将基于提示的无排练持续学习方法的训练时间减少最高 51%、内存降低最高 41%，精度损失微乎其微。

**[Replaceme Network Simplification Via Depth Pruning And Transformer Block Lineari](model_compression/replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)**

:   提出 ReplaceMe，一种无训练的深度剪枝方法：用少量校准数据估计线性变换来近似被剪枝的 Transformer 块组，该变换可融合到相邻层权重中不增加参数，在 LLaMA-2-7B 上实现 25% 剪枝率并保留约 90% 性能。

**[Representation Consistency For Accurate And Coherent Llm Answer Aggregation](model_compression/representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)**

:   提出 Representation Consistency (RC)，通过分析 LLM 生成多个候选答案时内部激活的一致性来改进答案聚合：同一答案的多条推理路径如果内部表示高度一致则更可能正确，结合稀疏自编码器的稀疏变体 RC-S 效果最优，在 4 个 LLM 和 4 个推理数据集上一致优于 Self-Consistency。

**[Restoring Pruned Large Language Models Via Lost Component Compensation](model_compression/restoring_pruned_large_language_models_via_lost_component_compensation.md)**

:   RestoreLCC 提出了一种面向剪枝 LLM 的定向恢复策略：通过对比探测定位关键注意力头，利用 SVD 分解提取剪枝丢失的激活成分，将其作为可优化的偏置向量注入回剪枝模型，在不影响稀疏性和推理速度的前提下显著恢复性能。

**[Revisiting Semi-Supervised Learning In The Era Of Foundation Models](model_compression/revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)**

:   系统性研究发现传统 SSL 方法在 VFM 时代效益有限——仅用有标签数据的 PEFT 即可匹敌 SSL——由此提出 V-PET：集成多种 PEFT 方法和多种 VFM 的伪标签来实现简洁高效的半监督学习。

**[Robust Federated Finetuning Of Llms Via Alternating Optimization Of Lora](model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)**

:   提出 RoLoRA，通过交替优化 LoRA 的 down-projection (A) 和 up-projection (B) 矩阵，解决联邦学习中 LoRA 聚合不精确和表达力受限的问题，在 RoBERTa-Large 和 Llama-2-7B 上显著优于 FedAVG of LoRA 和 FFA-LoRA。

**[Robustifying Learning-Augmented Caching Efficiently Without Compromising 1-Consi](model_compression/robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)**

:   提出 Guard 框架，一种轻量级鲁棒化方法，将广泛的学习增强缓存算法的鲁棒性提升至 $2H_{k-1}+2$，同时保持 1-一致性和 O(1) 的额外请求开销。

**[S2M-Former Spiking Symmetric Mixing Branchformer For Brain Auditory Attention De](model_compression/s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)**

:   提出 S2M-Former，一种脉冲驱动的对称混合 Branchformer 框架，通过空间-频率双分支的互补学习和轻量化 1D token 表示，在 EEG 听觉注意力检测任务上以仅 0.06M 参数实现了 SOTA 级精度，同时将能耗降低至双分支 ANN 模型的 1/5.8。

**[Shap Meets Tensor Networks Provably Tractable Explanations With Parallelism](model_compression/shap_meets_tensor_networks_provably_tractable_explanations_with_parallelism.md)**

:   本文首次为张量网络（Tensor Networks）提供可证明精确的 SHAP 解释计算框架，证明张量列车（Tensor Train）结构下 SHAP 可在多对数时间内并行计算（NC² 复杂度），并通过归约揭示二值化神经网络中**宽度而非深度**才是 SHAP 计算的核心瓶颈。

**[Single-Teacher View Augmentation Boosting Knowledge Distillation Via Angular Div](model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)**

:   提出Angular-KD，通过在单个教师模型上附加多个轻量线性分支并引入两种角度多样性损失（约束型视角间角度多样性损失和视角内角度多样性损失），从单教师生成多样化监督信号，以低成本替代多教师蒸馏方案，在多个KD基准上取得SOTA表现。

**[Skrull Towards Efficient Long Context Fine-Tuning Through Dynamic Data Schedulin](model_compression/skrull_towards_efficient_long_context_fine-tuning_through_dynamic_data_schedulin.md)**

:   针对长上下文监督微调（Long-SFT）中长短序列混合导致的训练效率低下问题，提出动态数据调度器Skrull，通过分布感知的上下文并行（DACP）和全局数据调度（GDS）两个组件，在真实Long-SFT场景中实现平均3.76倍（最高7.54倍）的训练加速。

**[Smooth Regularization For Efficient Video Recognition](model_compression/smooth_regularization_for_efficient_video_recognition.md)**

:   提出一种基于高斯随机游走（GRW）的平滑正则化技术，通过对视频识别模型中间层嵌入施加时序平滑约束（惩罚高加速度变化），在轻量级模型上实现3.8%–6.4%的准确率提升，在相应FLOP约束下刷新Kinetics-600 SOTA。

**[Spark Transformer Reactivating Sparsity In Ffn And Attention](model_compression/spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)**

:   提出 Spark Transformer 架构，通过 Statistical Top-k 算子在 FFN 和注意力机制中同时实现高水平激活稀疏性（FFN 仅 8% 神经元激活、每个 token 最多关注 256 个 token），在保持与 Gemma-2 相当质量的同时实现 2.5× FLOPs 降低和高达 1.79× 的推理加速。

**[Specialization After Generalization Towards Understanding Test-Time Training In ](model_compression/specialization_after_generalization_towards_understanding_test-time_training_in_.md)**

:   提出"泛化之后特化"框架，基于线性表示假设（LRH）从理论和实验两方面解释了测试时训练（TTT）在分布内数据上的有效性：基础模型全局欠参数化导致概念叠加干扰，TTT通过局部特化将模型容量重新分配给与测试任务相关的少数概念，从而在不增加模型规模的情况下提升预测性能。

**[Spiking Brain Compression Post-Training Second-Order Compression For Spiking Neu](model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)**

:   提出 Spiking Brain Compression（SBC），一种基于 Van Rossum 距离的二阶后训练一次性压缩框架，专为脉冲神经网络（SNN）设计，通过替代膜电位（SMP）Hessian 实现高效的模块级剪枝和量化，在 ImageNet 规模下首次压缩 SEW-ResNet152 和 Spike-Driven Transformer。

**[Synergy Between The Strong And The Weak Spiking Neural Networks Are Inherently S](model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)**

:   本文发现 SNN 可以在时间维度上天然解构为多个子模型，通过对比各时间步子模型的输出置信度识别"强"与"弱"，提出 Strong2Weak 和 Weak2Strong 两种自蒸馏方案，无需额外教师模型即可显著提升 SNN 性能，尤其在神经形态数据集上提升高达 5.36%。

**[The Graphon Limit Hypothesis Understanding Neural Network Pruning Via Infinite W](model_compression/the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)**

:   提出"Graphon极限假说"：当网络宽度趋于无穷时，不同剪枝方法产生的二值掩码序列在cut距离下收敛到各自独特的graphon极限，并在此基础上推导出Graphon NTK来分析稀疏网络训练动态，从理论层面解释了为什么不同剪枝方法在相同稀疏度下表现迥异。

**[The Structure Of Relation Decoding Linear Operators In Large Language Models](model_compression/the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)**

:   揭示 Transformer 语言模型中的线性关系解码器（LRE）并非编码细粒度关系，而是提取共享的粗粒度语义属性（如"国家"、"性别"），并利用阶-3 张量网络将大量关系解码矩阵压缩数个数量级。

**[Tighter Cmi-Based Generalization Bounds Via Stochastic Projection And Quantizati](model_compression/tighter_cmi-based_generalization_bounds_via_stochastic_projection_and_quantizati.md)**

:   通过在 CMI（条件互信息）框架中引入**随机投影**和**有损压缩**，推导出更紧的泛化界，解决了经典 CMI 界在 SCO 反例上失效的问题，并证明记忆化对良好泛化并非必要。

**[Tokensqueeze Performance-Preserving Compression For Reasoning Llms](model_compression/tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)**

:   提出TokenSqueeze方法，通过自适应推理深度选择、步内语言精炼（基于KL散度约束）和长度感知的偏好优化三阶段流程，仅用模型自生成数据实现推理链50%的token压缩而不损失准确率。

**[Toward Efficient Inference Attacks Shadow Model Sharing Via Mixture-Of-Experts](model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)**

:   提出基于 Mixture-of-Experts 的影子模型共享方案，通过在多种推理攻击任务间共享影子模型的特征提取层、仅训练任务特定的轻量专家模块来降低影子模型的整体训练成本，同时保持或提升攻击性能。

**[Towards Implicit Aggregation Robust Image Representation For Place Recognition I](model_compression/towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)**

:   提出 ImAge（Implicit Aggregation），在 Transformer 骨干网络的特定层插入可学习聚合 Token，利用内在自注意力机制将 patch 特征隐式聚合为全局描述符，完全消除了额外聚合器的需要。以最小的描述符维度（6144）和最快推理速度，在多个 VPR 数据集上超越 SALAD、BoQ 等 SOTA，并在 MSLS Challenge 排行榜排名第 1。

**[Towards Unsupervised Open-Set Graph Domain Adaptation Via Dual Reprogramming](model_compression/towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)**

:   提出 GraphRTA 框架，通过模型重编程（基于梯度的权重剪枝）和图重编程（目标图结构与特征优化）双重机制，解决无监督开放集图域适应中已知类分类与未知类识别难题，无需人工设定阈值。

**[Train With Perturbation Infer After Merging A Two-Stage Framework For Continual ](model_compression/train_with_perturbation_infer_after_merging_a_two-stage_framework_for_continual_.md)**

:   提出Perturb-and-Merge (P&M)框架，将模型合并机制引入持续学习范式：训练时沿任务向量方向添加随机扰动以平滑损失面，推理时通过闭式最优系数对历史模型和当前任务模型做凸组合合并，结合LoRA实现内存高效的SOTA持续学习性能。

**[Trajectory Balance With Asynchrony Decoupling Exploration And Learning For Fast ](model_compression/trajectory_balance_with_asynchrony_decoupling_exploration_and_learning_for_fast_.md)**

:   提出 TBA（Trajectory Balance with Asynchrony），将 GFlowNet 的轨迹平衡（TB）目标与异步分布式 RL 架构结合，实现 LLM 后训练中探索与学习的解耦，在数学推理、偏好微调和自动红队测试任务上获得 4-50 倍加速且性能不降反升。

**[Traversal Verification For Speculative Tree Decoding](model_compression/traversal_verification_for_speculative_tree_decoding.md)**

:   提出 Traversal Verification，一种从叶节点到根节点的自底向上验证算法，通过考虑整条路径的序列级概率而非单 token 概率来决定接受/拒绝，理论证明无损性和单链最优性，在多种树结构和任务上一致提升接受长度 2.2%-5.7%。

**[Twilight Adaptive Attention Sparsity With Hierarchical Top-P Pruning](model_compression/twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)**

:   提出 Twilight，借鉴 top-p 采样（nucleus sampling）的思想替代固定预算 top-k 做注意力稀疏——动态选择注意力权重累积达 p% 的最少 Token，自适应不同注意力头的分布特征，在保持精度的同时比 SOTA 稀疏注意力再提速 1.4x。

**[Ultrametric Cluster Hierarchies I Want Em All](model_compression/ultrametric_cluster_hierarchies_i_want_em_all.md)**

:   证明了对于任意合理的聚类层次树，都可以快速找到任意中心型聚类目标（如 k-means）的最优解，且这些解本身也是层次化的，从而从一棵树中解锁大量等价有意义的层次结构。

**[Understanding Differential Transformer Unchains Pretrained Self-Attentions](model_compression/understanding_differential_transformer_unchains_pretrained_self-attentions.md)**

:   深入分析 Differential Transformer（差分注意力）的内部机制，揭示差分操作等效于一种鲁棒的注意力去噪过程——它"解放"了受 softmax 归一化约束的预训练自注意力，使注意力权重更自由地分配到真正重要的 Token 上。

**[Uni-Lora One Vector Is All You Need](model_compression/uni-lora_one_vector_is_all_you_need.md)**

:   提出 Uni-LoRA 统一框架，证明各种 LoRA 变体（Tied-LoRA、VeRA、VB-LoRA 等）的参数缩减策略本质上是对全参数空间 $\mathbb{R}^D$ 到低维子空间 $\mathbb{R}^d$ 的投影差异，并设计了一种等距随机分组投影矩阵——只需训练一个向量即可重建整个 LLM 的 LoRA 参数，实现极致参数效率。

**[Universal Cross-Tokenizer Distillation Via Approximate Likelihood Matching](model_compression/universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)**

:   本文提出 Approximate Likelihood Matching (ALM)，一种基于二值化 f-散度的原则性跨分词器蒸馏方法，首次实现了跨根本不同分词器（如子词→字节级）的有效蒸馏和纯蒸馏。

**[Vessa Video-Based Object-Centric Self-Supervised Adaptation For Visual Foundatio](model_compression/vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)**

:   提出 VESSA，一种利用短物体中心视频进行无监督微调的方法，通过自蒸馏框架配合 LoRA 和不确定性加权损失，在不需要标注数据的情况下将视觉基础模型适配到目标域，在 33 个 VFM × 22 个数据集上持续提升下游分类性能。

**[Vision-Centric Token Compression In Large Language Model](model_compression/vision-centric_token_compression_in_large_language_model.md)**

:   Vist 提出了一种以视觉为核心的慢-快双路径 token 压缩框架，将远端长文本渲染为图像后用轻量视觉编码器压缩，配合概率引导的视觉增强（PVE）训练目标，在 11 个 ICL 基准上以 2.3× 更少的 token 实现同等精度，FLOPs 降低 16%、显存减少 50%。

**[Vqtoken Neural Discrete Token Representation Learning For Extreme Token Reductio](model_compression/vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)**

:   VQToken 提出了首个基于向量量化的视频 token 极限压缩框架，通过自适应离散化将连续 ViT embedding 聚类为紧凑码本，并用 token hash 函数保留时空位置信息，在 NextQA-MC 上仅用原始 0.07% 的 token（约 13 个）实现了仅 0.66% 的精度损失。

**[When Worse Is Better Navigating The Compression-Generation Tradeoff In Visual To](model_compression/when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)**

:   本文通过scaling law系统研究了视觉tokenizer压缩率与生成质量的权衡关系，发现对小模型而言更激进的压缩（虽然重建更差）反而有利于生成，并提出因果正则化Tokenization(CRT)方法在stage 1训练中嵌入自回归归纳偏置，实现2-3倍计算效率提升，以775M参数和256 token/image匹配LlamaGen-3B的2.18 FID。

**[Zip2Zip Inference-Time Adaptive Tokenization Via Online Compression](model_compression/zip2zip_inference-time_adaptive_tokenization_via_online_compression.md)**

:   提出 zip2zip，将经典 LZW 在线无损压缩算法深度集成到 LLM 的推理流程中，通过在解码过程中持续将高频共现 Token 合并为可复用的"hypertoken"来动态扩展词表，配合动态嵌入层和压缩空间语言建模训练，仅需 10 GPU-hours 的 LoRA 微调即可使现有 LLM 获得推理时自适应分词能力，实现输入输出序列长度缩减 15-40%、端到端解码延迟降低最高 40%，且下游任务性能几乎无损。

---

## 🧊 3D视觉 { #3d_vision }

**[3D-Agenttri-Modal Multi-Agent Collaboration For Scalable 3D Object Annotation](3d_vision/3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)**

:   提出 Tri-MARF 三模态多智能体框架，通过 VLM 标注 Agent（多视角多候选描述）+ 信息聚合 Agent（BERT 聚类 + CLIP 加权 + UCB1 多臂赌博机选择）+ 点云门控 Agent（Uni3D 文本-点云对齐过滤幻觉），实现 CLIPScore 88.7（超越人类标注 82.4）、吞吐量 12k 物体/小时，已标注约 200 万 3D 模型。

**[3D Visual Illusion Depth Estimation](3d_vision/3d_visual_illusion_depth_estimation.md)**

:   揭示了3D视觉错觉（如墙面彩绘、屏幕重播、镜面反射等）会严重欺骗现有SOTA单目和双目深度估计方法，构建了包含约3k场景/200k图像的大规模数据集，并提出基于VLM常识推理的单目-双目自适应融合框架，在各类错觉场景下达到SOTA。

**[Anti-Aliased 2D Gaussian Splatting](3d_vision/anti-aliased_2d_gaussian_splatting.md)**

:   提出 AA-2DGS，通过世界空间平坦平滑核和物体空间 Mip 滤波器两个互补机制，解决 2D Gaussian Splatting 在不同采样率下渲染时的严重锯齿问题，在保持 2DGS 几何精度优势的同时显著提升多尺度渲染质量。

**[Armesh Autoregressive Mesh Generation Via Next-Level-Of-Detail Prediction](3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)**

:   提出将 3D mesh 生成建模为"由粗到精"的逐级细化过程（next-level-of-detail prediction），通过反转广义网格简化算法（GSlim）获得渐进式细化序列，再用 Transformer 自回归学习，从单个点开始逐步增加几何与拓扑细节生成完整网格。

**[Atlasgs Atlanta-World Guided Surface Reconstruction With Implicit Structured Gau](3d_vision/atlasgs_atlanta-world_guided_surface_reconstruction_with_implicit_structured_gau.md)**

:   提出 AtlasGS，通过将 Atlanta-world 结构先验引入隐式结构化高斯表示（implicit-structured Gaussians），在室内和城市场景中实现平滑且保留高频细节的高质量表面重建，全面超越已有隐式和显式方法。

**[Becominglit Relightable Gaussian Avatars With Hybrid Neural Shading](3d_vision/becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)**

:   提出 BecomingLit，基于 3D Gaussian 原语和混合神经着色（neural diffuse BRDF + 解析 Cook-Torrance specular）从低成本 light stage 多视角序列重建可重光照、实时渲染的高保真头部 avatar，并发布了新的公开 OLAT 人脸数据集。

**[Can Llms Write Faithfully An Agent-Based Evaluation Of Llm-Generated Islamic Con](3d_vision/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)**

:   提出双Agent（定量+定性）评估框架，从神学准确性、引用完整性和文体恰当性三个维度系统评估 GPT-4o、Ansari AI 和 Fanar 在伊斯兰内容生成任务上的忠实度，发现即使最优模型也在引用可靠性上存在显著不足。

**[Clipgaussian Universal And Multimodal Style Transfer Based On Gaussian Splatting](3d_vision/clipgaussian_universal_and_multimodal_style_transfer_based_on_gaussian_splatting.md)**

:   CLIPGaussian 提出首个基于 Gaussian Splatting 的统一风格迁移框架，支持文本和图像引导的 2D 图像、视频、3D 物体和 4D 动态场景的风格化，作为即插即用模块集成到现有 GS 管线中，无需大规模生成模型或从头重训，且不改变模型大小。

**[Concerto Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](3d_vision/concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)**

:   Concerto 将 3D 点云模态内自蒸馏与 2D-3D 跨模态联合嵌入预测相结合，以极简设计让单一点云编码器（PTv3）涌现出超越 2D/3D 单模态甚至两者拼接的空间表征，在多个 3D 场景理解基准上刷新 SOTA（ScanNet 语义分割 80.7% mIoU）。

**[Copresheaf Topological Neural Networks A Generalized Deep Learning Framework](3d_vision/copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)**

:   本文提出 Copresheaf Topological Neural Networks (CTNNs)，基于代数拓扑中的余预层（copresheaf）概念，在组合复形（combinatorial complex）上定义方向性、异质的消息传递机制，统一了 CNN、GNN、Transformer、Sheaf Neural Networks 和拓扑神经网络等多种深度学习架构，并在物理模拟、图分类和高阶复形分类任务上超越传统基线。

**[Cosmobench A Multiscale Multiview Multitask Cosmology Benchmark For Geometric De](3d_vision/cosmobench_a_multiscale_multiview_multitask_cosmology_benchmark_for_geometric_de.md)**

:   提出 CosmoBench——目前最大的宇宙学几何深度学习基准，包含 3.4 万点云和 2.5 万有向树，覆盖多尺度、多视角、多任务，并揭示简单线性模型有时能超越大型 GNN。

**[Cue3D Quantifying The Role Of Image Cues In Single-Image 3D Generation](3d_vision/cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)**

:   Cue3D是首个模型无关的图像线索重要性量化框架，通过系统性扰动光照、纹理、轮廓、透视、边缘和局部连续性6种视觉线索，在涵盖回归式/多视图/原生3D生成三大范式的7个方法上揭示了关键洞察：形状意义性而非纹理决定泛化能力，光照比纹理更重要，且模型过度依赖输入轮廓。

**[D2Ust3R Enhancing 3D Reconstruction For Dynamic Scenes](3d_vision/d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)**

:   提出 Static-Dynamic Aligned Pointmap (SDAP) 表示，将静态和动态区域的 3D 对齐统一建模，使 DUSt3R 系列方法能够在动态场景中实现准确的稠密三维重建与对应关系估计。

**[Dc4Gs Directional Consistency-Driven Adaptive Density Control For 3D Gaussian Sp](3d_vision/dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)**

:   提出基于方向一致性（Directional Consistency）的自适应密度控制方法 DC4GS，通过利用位置梯度的角度相干性来改进 3DGS 中的 primitive 分裂决策和分裂位置选择，在减少最多 30% primitive 数量的同时提升重建质量。

**[Dgh Dynamic Gaussian Hair](3d_vision/dgh_dynamic_gaussian_hair.md)**

:   提出 Dynamic Gaussian Hair (DGH)，一个数据驱动的 coarse-to-fine 框架，通过体素隐式变形模型学习头发动力学，并结合柱状 Gaussian 表示与曲率混合策略实现动态头发的逼真新视角渲染。

**[Dualfocus Depth From Focus With Spatio-Focal Dual Variational Constraints](3d_vision/dualfocus_depth_from_focus_with_spatio-focal_dual_variational_constraints.md)**

:   提出 DualFocus，通过空间变分约束（利用焦距相关梯度模式区分深度边缘与纹理伪影）和焦距变分约束（强制单峰单调的对焦概率分布）双重约束，实现从焦距堆栈中鲁棒精确的深度估计。

**[Dynamic Gaussian Splatting From Defocused And Motion-Blurred Monocular Videos](3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)**

:   提出统一框架，通过可学习模糊核卷积联合建模散焦模糊和运动模糊，结合动态高斯致密化策略和未见视角约束，从模糊单目视频中实现高质量动态 3DGS 新视角合成。

**[Dynarend Learning 3D Dynamics Via Masked Future Rendering For Robotic Manipulati](3d_vision/dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)**

:   提出 DynaRend，通过掩码重建和未来预测两个互补目标，利用可微体渲染在 triplane 表征上联合学习 3D 几何、语义和动态信息，预训练后可高效迁移到下游机器人操控任务。

**[E-Moflow Learning Egomotion And Optical Flow From Event Data Via Implicit Regula](3d_vision/e-moflow_learning_egomotion_and_optical_flow_from_event_data_via_implicit_regula.md)**

:   提出 E-MoFlow，通过将光流建模为隐式神经表示、自运动建模为连续样条，并利用微分几何约束联合优化两者，在无监督范式下实现事件数据的 6-DoF 自运动和稠密光流联合估计。

**[Ea3D Online Open-World 3D Object Extraction From Streaming Videos](3d_vision/ea3d_online_open-world_3d_object_extraction_from_streaming_videos.md)**

:   提出 EA3D（ExtractAnything3D），一个在线开放世界 3D 物体提取框架，通过知识集成特征图、在线视觉里程计和循环联合优化，从流式视频中同时进行几何重建和全面场景理解。

**[Eag3R Event-Augmented 3D Geometry Estimation For Dynamic And Extreme-Lighting Sc](3d_vision/eag3r_event-augmented_3d_geometry_estimation_for_dynamic_and_extreme-lighting_sc.md)**

:   EAG3R 将事件相机的异步事件流融入 MonST3R 点图重建框架，通过 Retinex 增强模块 + SNR 感知融合机制 + 事件光度一致性损失，在极端低光动态场景下实现鲁棒的深度估计、位姿跟踪和 4D 重建，零样本迁移夜间场景即可大幅超越 RGB-only 方法。

**[Ef-3Dgs Event-Aided Free-Trajectory 3D Gaussian Splatting](3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)**

:   EF-3DGS 首次将事件相机引入自由轨迹场景重建，通过事件生成模型（EGM）重建帧间潜在图像做连续监督、对比度最大化（CMax）结合线性事件模型（LEGM）挖掘运动信息校准位姿，以及光度 BA + Fixed-GS 策略解决颜色不一致问题，在高速场景下 PSNR 提升 3dB、ATE 降低 40%。

**[Electra A Cartesian Network For 3D Charge Density Prediction With Floating Orbit](3d_vision/electra_a_cartesian_network_for_3d_charge_density_prediction_with_floating_orbit.md)**

:   提出 ELECTRA（Electronic Tensor Reconstruction Algorithm），一种等变笛卡尔张量网络，通过预测浮动高斯轨道的位置、权重和协方差矩阵来重构电子密度，在 QM9 基准上精度比 SOTA 方法 SCDP 高 2.4 倍且推理速度快 4.4-11 倍，并将 DFT 的 SCF 迭代次数减少 50.72%。

**[Enerverse Envisioning Embodied Future Space For Robotics Manipulation](3d_vision/enerverse_envisioning_embodied_future_space_for_robotics_manipulation.md)**

:   EnerVerse 是一个生成式机器人基础模型，通过 chunk-wise 自回归视频扩散 + 稀疏上下文记忆 + 多视角生成先验构建 4D 具身空间，结合 4DGS 数据飞轮缩小 Sim2Real 差距，最终通过策略头将 4D 世界表示转化为物理动作，在 LIBERO 基准上达到 SOTA。

**[Eugens Efficient Unified And General Dense Layers](3d_vision/eugens_efficient_unified_and_general_dense_layers.md)**

:   EUGens 提出一类新的高效稠密层，利用随机特征（Random Features）将全连接前馈层的推理复杂度从二次降到线性，统一了已有的高效 FFL 扩展，在 LLM 预训练、ViT 图像分类、NeRF/iSDF 三维重建等任务中实现高达 27% 加速和 30% 参数压缩，且支持无需反向传播的层级知识蒸馏。

**[Evaluation Of Vision-Llms In Surveillance Video](3d_vision/evaluation_of_vision-llms_in_surveillance_video.md)**

:   提出一个无训练的两阶段框架，利用小型 Vision-LLM 生成视频文本描述 + NLI 分类器零样本评分，系统评估了提示策略和隐私保护滤镜对监控视频异常行为识别的影响。

**[Every Camera Effect Every Time All At Once 4D Gaussian Ray Tracing For Physics-B](3d_vision/every_camera_effect_every_time_all_at_once_4d_gaussian_ray_tracing_for_physics-b.md)**

:   提出 4D Gaussian Ray Tracing (4D-GRT)，将 4D Gaussian Splatting 与物理光线追踪结合，从多视角视频重建动态场景后，以可控参数生成鱼眼畸变、景深模糊、卷帘快门等物理精确的相机效果视频数据。

**[Fin3R Fine-Tuning Feed-Forward 3D Reconstruction Models Via Monocular Knowledge ](3d_vision/fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)**

:   提出 Fin3R，通过冻结 decoder 并用带重归一化的 LoRA 适配器对 encoder 进行单目知识蒸馏微调，以统一且轻量的方式提升 DUSt3R/MASt3R/CUT3R/VGGT 等前馈式 3D 重建模型的几何精度和鲁棒性。

**[Flarex A Physics-Informed Dataset For Lens Flare Removal Via 2D Synthesis And 3D](3d_vision/flarex_a_physics-informed_dataset_for_lens_flare_removal_via_2d_synthesis_and_3d.md)**

:   提出 FlareX 数据集，通过参数化模板创建、基于光照定律的 2D 合成和基于物理引擎的 3D 渲染三个阶段生成物理真实的镜头光晕数据，训练的模型在真实世界测试集上显著超越此前所有数据集。

**[Flux4D Flow-Based Unsupervised 4D Reconstruction](3d_vision/flux4d_flow-based_unsupervised_4d_reconstruction.md)**

:   提出 Flux4D，一个无监督且可泛化的 4D 动态驾驶场景重建框架，通过前馈网络直接预测 3D 高斯及其运动速度，仅用光度损失和静态偏好正则化实现大规模场景重建，在 PandaSet 和 Waymo 上超越所有无监督方法并接近有监督方法的性能。

**[Frequency Matters When Time Series Foundation Models Fail Under Spectral Shift](3d_vision/frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)**

:   揭示时间序列基础模型（TSFM）在工业场景中泛化失败的关键原因——频谱偏移（downstream 数据主频与预训练数据不重叠），通过工业级手游玩家参与预测任务和受控合成实验验证了这一假说。

**[From Objects To Anywhere A Holistic Benchmark For Multi-Level Visual Grounding I](3d_vision/from_objects_to_anywhere_a_holistic_benchmark_for_multi-level_visual_grounding_i.md)**

:   提出 Anywhere3D-Bench，首个涵盖区域/空间/物体/部件四个层级的 3D 视觉定位基准，揭示即使最强的 Gemini-2.5-Pro 和 o3 在空间级任务上仅达约 30% 准确率、部件级约 40%，远低于人类的 95%。

**[From Pixels To Views Learning Angular-Aware And Physics-Consistent Representatio](3d_vision/from_pixels_to_views_learning_angular-aware_and_physics-consistent_representatio.md)**

:   提出 XLFM-Former，通过 **视角级 Masked View Modeling（MVM-LF）** 自监督预训练学习 XLFM 的角度–空间先验，并设计基于 PSF 可微渲染的 **光学渲染一致性损失（ORC Loss）** 约束重建体积的物理合理性，在自建的首个 XLFM-Zebrafish 标准化基准上，平均 PSNR 达 54.04 dB，较最佳基线 ConvNeXt（50.16 dB）提升 **7.7%**。

**[From Programs To Poses Factored Real-World Scene Generation Via Learned Program ](3d_vision/from_programs_to_poses_factored_real-world_scene_generation_via_learned_program_.md)**

:   提出 FactoredScenes，将真实世界 3D 场景生成分解为五步因式分解——从合成数据学布局程序库、LLM 生成场景程序、执行程序获得轴对齐布局、程序条件化层次姿态预测、物体检索放置，在卧室上 FID 改善 38.3%、KID 改善 80.4%，人类仅 67% 能区分生成与真实 ScanNet。

**[Fully Dynamic Algorithms For Chamfer Distance](3d_vision/fully_dynamic_algorithms_for_chamfer_distance.md)**

:   提出首个全动态 Chamfer 距离维护算法，将问题归约为近似最近邻（ANN）查询，实现 $(1+\epsilon)$ 近似且更新时间 $\tilde{O}(\epsilon^{-d})$，大幅突破了静态重算的线性时间下界，在真实数据集上误差 <10% 且速度比朴素方法快数个数量级。

**[Galactification Painting Galaxies Onto Dark Matter Only Simulations Using A Tran](3d_vision/galactification_painting_galaxies_onto_dark_matter_only_simulations_using_a_tran.md)**

:   提出一个多模态 Transformer 编解码框架，以廉价的暗物质 N-body 模拟的密度场和速度场为输入，自回归生成星系目录（位置 + 物理属性），在多种统计指标上忠实再现流体动力学模拟结果，计算加速约 100 倍。

**[Gaudp Reinventing Multi-Agent Collaboration Through Gaussian-Image Synergy In Di](3d_vision/gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)**

:   提出 GauDP，通过从多智能体的去中心化 RGB 观测中构建全局一致的 3D 高斯场，并将高斯属性动态分配回各智能体的局部视角，实现可扩展的、感知增强的多智能体协作模仿学习。

**[Gaussian-Augmented Physics Simulation And System Identification With Complex Col](3d_vision/gaussian-augmented_physics_simulation_and_system_identification_with_complex_col.md)**

:   提出 AS-DiffMPM，一种支持任意形状刚体碰撞体的可微物质点法（MPM）框架，结合多种新视角合成方法实现从视觉观测中估计物体物理参数的系统辨识。

**[Gaze Beyond The Frame Forecasting Egocentric 3D Visual Span](3d_vision/gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)**

:   提出 EgoSpanLift 方法，将第一人称 2D 注视预测提升到 3D 空间，构建多层级体积视觉跨度表示，结合 3D U-Net 和单向 Transformer 实现对未来 3D 视觉关注区域的预测。

**[Geocomplete Geometry-Aware Diffusion For Reference-Driven Image Completion](3d_vision/geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)**

:   提出 GeoComplete，通过将投影点云作为几何条件注入双分支扩散模型，并结合 target-aware masking 策略，实现几何一致的参考驱动图像补全，PSNR 提升 17.1%。

**[Geosvr Taming Sparse Voxels For Geometrically Accurate Surface Reconstruction](3d_vision/geosvr_taming_sparse_voxels_for_geometrically_accurate_surface_reconstruction.md)**

:   提出基于稀疏体素的显式表面重建框架 GeoSVR，通过体素不确定性深度约束和稀疏体素表面正则化，在几何精度、细节保留和重建完整性方面全面超越现有基于 3DGS 和 SDF 的方法。

**[Goatex Geometry Occlusion-Aware Texturing](3d_vision/goatex_geometry_occlusion-aware_texturing.md)**

:   GOATex 提出首个遮挡感知的 3D 网格纹理生成框架，通过基于光线投射的 hit level 分层机制将网格分解为由外到内的可见性层，配合法线翻转和残差面聚类的两阶段可见性控制策略以及基于可见性权重的 UV 空间融合，实现了对外表面和被遮挡内表面的高质量纹理生成。

**[Haif-Gs Hierarchical And Induced Flow-Guided Gaussian Splatting For Dynamic Scen](3d_vision/haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)**

:   HAIF-GS 提出基于稀疏运动锚点的动态 3DGS 框架，通过锚点过滤器区分动静区域、自监督诱导场景流引导时序一致变形、以及分层锚点加密捕捉精细非刚性运动，在 NeRF-DS 和 D-NeRF 基准上取得 SOTA 渲染质量。

**[High Resolution Udf Meshing Via Iterative Networks](3d_vision/high_resolution_udf_meshing_via_iterative_networks.md)**

:   本文提出首个针对无符号距离场（UDF）的迭代式网格化方法，通过多轮次前向传播逐步将邻域信息传播到局部体素的伪符号预测中，有效解决了高分辨率下神经 UDF 噪声导致的表面空洞和不连续问题，在多个数据集上显著优于现有单遍方法。

**[How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need](3d_vision/how_many_tokens_do_3d_point_cloud_transformer_architectures_really_need.md)**

:   本文系统性地揭示了 3D 点云 Transformer（如 PTv3、Sonata）中存在 90-95% 的 token 冗余，并提出 gitmerge3D——一种全局信息感知的图 token 合并方法，通过能量分数自适应合并策略实现了高达 5.3× FLOPs 降低和 6.4× 显存节省而几乎不损失精度。

**[Hybrid Physical-Neural Simulator For Fast Cosmological Hydrodynamics](3d_vision/hybrid_physical-neural_simulator_for_fast_cosmological_hydrodynamics.md)**

:   提出一种混合物理-神经宇宙学模拟器，用可微分粒子网格（PM）方法处理引力动力学，用物理约束的神经网络参数化气体的有效压力场，仅需单次参考模拟即可训练，在场级别和统计量级别均优于 EGD 基线。

**[Hyplanehead Rethinking Tri-Plane-Like Representations In Full-Head Image Synthes](3d_vision/hyplanehead_rethinking_tri-plane-like_representations_in_full-head_image_synthes.md)**

:   系统分析了 tri-plane 类表征在 3D 感知头部合成中的三大问题（镜像伪影、不均匀映射、特征穿透），提出 hy-plane 混合表征（平面+球面）结合 unify-split 策略和近等面积映射，在全头图像合成中达到 SOTA。

**[Hyrf Hybrid Radiance Fields For Memory-Efficient And High-Quality Novel View Syn](3d_vision/hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)**

:   提出混合辐射场（HyRF），将紧凑的显式高斯体（仅存储8个参数）与解耦的网格神经场相结合，在实现 20× 模型压缩的同时达到 SOTA 渲染质量和实时性能。

**[Ibgs Image-Based Gaussian Splatting](3d_vision/ibgs_image-based_gaussian_splatting.md)**

:   提出基于图像的高斯泼溅方法（IBGS），通过从邻近训练图像中学习颜色残差来增强标准3DGS的渲染质量，在不增加存储开销的前提下显著提升高频细节和视角依赖效果的建模能力。

**[Indego A Dataset Of Industrial Scenarios And Collaborative Work For Egocentric A](3d_vision/indego_a_dataset_of_industrial_scenarios_and_collaborative_work_for_egocentric_a.md)**

:   提出IndEgo——首个面向真实工业场景的大规模多模态第一人称视觉数据集，包含3,460段自我中心录像（约197小时）和1,092段外部视角录像（约97小时），覆盖装配/拆卸、物流、检修、木工等五大类任务及协作场景，并建立了错误检测、推理问答和协作理解三项基准。

**[Instant Video Models Universal Adapters For Stabilizing Image-Based Networks](3d_vision/instant_video_models_universal_adapters_for_stabilizing_image-based_networks.md)**

:   提出一类通用的稳定化适配器（Stabilization Adapters），可插入几乎任何图像模型架构中，通过冻结基础网络仅训练适配器参数，配合统一的精度-稳定性-鲁棒性损失函数，使帧级模型获得视频时序一致性和腐蚀鲁棒性。

**[Jasmine Harnessing Diffusion Prior For Self-Supervised Depth Estimation](3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)**

:   首次将Stable Diffusion的视觉先验引入自监督单目深度估计框架，提出Mix-Batch Image Reconstruction（MIR）代理任务保护SD先验不被重投影噪声损坏，并设计Scale-Shift GRU（SSG）桥接SD的尺度-偏移不变性（SSI）与自监督的尺度不变性（SI）深度分布，在KITTI上AbsRel=0.090达到所有SSMDE方法SOTA，且零样本泛化全面超越Marigold、E2E FT、Lotus等有监督SD方法。

**[Langsplatv2 High-Dimensional 3D Language Gaussian Splatting With 450 Fps](3d_vision/langsplatv2_high-dimensional_3d_language_gaussian_splatting_with_450_fps.md)**

:   通过将每个3D高斯视为全局字典上的稀疏编码，LangSplatV2用稀疏系数场替代重量级解码器，实现476.2 FPS的高维特征溅射和384.6 FPS的3D开放词汇查询，较LangSplat加速47倍。

**[Learning Efficient Fuse-And-Refine For Feed-Forward 3D Gaussian Splatting](3d_vision/learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)**

:   提出Fuse-and-Refine模块，通过混合Splat-Voxel表征将像素对齐的高斯基元聚合到粗到细的体素层次结构中，用稀疏体素Transformer在15ms内融合约20万基元并提升约2dB PSNR，且仅在静态场景训练即可零样本泛化到流式动态场景重建。

**[Learning From Videos For 3D World Enhancing Mllms With 3D Vision Geometry Priors](3d_vision/learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)**

:   VG LLM提出将3D视觉几何编码器（VGGT）集成到多模态大语言模型中，仅从视频输入（无需显式3D数据）即可提取和融合3D几何先验，在3D场景理解和空间推理任务上显著提升MLLM性能，4B模型在VSI-Bench上超越Gemini-1.5-Pro。

**[Learning Neural Exposure Fields For View Synthesis](3d_vision/learning_neural_exposure_fields_for_view_synthesis.md)**

:   提出神经曝光场（NExF），通过学习每个 3D 点的最优曝光值（而非每张图像的曝光），实现 3D 一致的高质量视图合成，在高动态范围场景中相比 SOTA 方法 PSNR 提升 3.5+，速度快 50 倍。

**[Less Is More Unlocking Specialization Of Time Series Foundation Models Via Struc](3d_vision/less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)**

:   揭示预训练时间序列基础模型（TSFM）存在固有的任务相关稀疏性，提出"先剪枝再微调"范式——通过结构化剪枝移除任务无关参数，使剪枝后微调的小模型显著超越直接微调的原模型，甚至胜过强专用基线。

**[Linearly Constrained Diffusion Implicit Models](3d_vision/linearly_constrained_diffusion_implicit_models.md)**

:   提出 CDIM，一种基于 DDIM 的线性逆问题求解算法，通过将残差能量与前向扩散过程的 $\chi^2$ 分布对齐来自适应控制投影步数和步长，实现比 DPS 快 10-50 倍的推理速度，同时在无噪声情况下精确满足测量约束。

**[Linprim Linear Primitives For Differentiable Volumetric Rendering](3d_vision/linprim_linear_primitives_for_differentiable_volumetric_rendering.md)**

:   提出 LinPrim，用线性基元（八面体和四面体）替代3D高斯核作为新视角合成的场景表示，通过可微光栅化pipeline实现端到端优化，在真实数据集上以更少的基元数量达到与3DGS可比的重建质量，同时保持实时渲染能力。

**[Locality-Sensitive Hashing-Based Efficient Point Transformer For Charged Particl](3d_vision/locality-sensitive_hashing-based_efficient_point_transformer_for_charged_particl.md)**

:   通过将 LSH 与 Point Transformer 结合，提出 HEPTv2 实现粒子轨迹重建的端到端学习，消除了 DBScan 聚类后处理瓶颈，在保持竞争性追踪效率的同时实现 28.9 倍加速。

**[Lodge Level-Of-Detail Large-Scale Gaussian Splatting With Efficient Rendering](3d_vision/lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)**

:   提出 LODGE，通过层次化 LOD（Level-of-Detail）策略对 3D Gaussian Splatting 进行多尺度管理，根据相机距离动态选择合适粒度的 Gaussian 表示，实现大规模场景的高质量实时渲染。

**[Look And Tell A Dataset For Multimodal Grounding Across Egocentric And Exocentri](3d_vision/look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)**

:   Look and Tell 构建了一个多模态数据集，在厨房场景中使用 Meta Aria 智能眼镜和固定 GoPro 摄像头同步采集 25 名参与者的注视、语音和双视角视频，结合 3D 场景重建和多层级标注流水线，提供了首个跨第一人称/第三人称视角的指称交际研究基准。

**[Materialrefgs Reflective Gaussian Splatting With Multi-View Consistent Material ](3d_vision/materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)**

:   提出MaterialRefGS，通过多视角一致的材质推断约束和基于2DGS光线追踪的环境建模策略，实现反射表面的高保真新视角合成和精确光照分解。

**[Mesh-Rft Enhancing Mesh Generation Via Fine-Grained Reinforcement Fine-Tuning](3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)**

:   提出 Mesh-RFT 框架，通过拓扑感知评分系统和掩码直接偏好优化（M-DPO）实现面级别的细粒度网格质量优化，显著提升生成网格的几何完整性和拓扑规则性。

**[Mesh Interpolation Graph Network For Dynamic And Spatially Irregular Global Weat](3d_vision/mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)**

:   提出 MIGN 框架，通过网格插值策略将不规则气象站数据映射到规则 HEALPix 网格上进行消息传递，并引入参数化球谐函数位置编码增强空间泛化能力，在全球天气预报任务中显著超越现有方法。

**[Metags A Meta-Learned Gaussian-Phong Model For Out-Of-Distribution 3D Scene Reli](3d_vision/metags_a_meta-learned_gaussian-phong_model_for_out-of-distribution_3d_scene_reli.md)**

:   提出MetaGS，通过将可微Blinn-Phong反射模型嵌入3D高斯splatting并结合双层优化的元学习训练策略，实现在分布外（OOD）光照条件下的高质量3D场景重光照。

**[Metropolis-Hastings Sampling For 3D Gaussian Reconstruction](3d_vision/metropolis-hastings_sampling_for_3d_gaussian_reconstruction.md)**

:   提出自适应Metropolis-Hastings框架替代3DGS中的启发式密度控制机制，通过多视角光度误差驱动的概率采样实现更高效的高斯分布推断，收敛速度快于3DGS-MCMC。

**[Modeling Microenvironment Trajectories On Spatial Transcriptomics With Nicheflow](3d_vision/modeling_microenvironment_trajectories_on_spatial_transcriptomics_with_nicheflow.md)**

:   NicheFlow是一种基于Flow Matching的生成模型，将细胞微环境表示为点云，通过Variational Flow Matching和最优传输联合建模细胞状态与空间坐标的时间演化，在胚胎发育、脑发育和衰老数据集上显著优于单细胞级别的轨迹推断方法。

**[More Than Generation Unifying Generation And Depth Estimation Via Text-To-Image ](3d_vision/more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)**

:   Merge提出了一种即插即用的框架，在固定的预训练T2I扩散模型前插入轻量级可学习的Converter，仅用约12%的额外参数就能赋予模型深度估计能力，同时完美保留原有的图像生成能力，在多个零样本深度估计基准上达到了统一模型的SOTA。

**[Motion4D Learning 3D-Consistent Motion And Semantics For 4D Scene Understanding](3d_vision/motion4d_learning_3d-consistent_motion_and_semantics_for_4d_scene_understanding.md)**

:   Motion4D提出了一个统一的4D高斯溅射框架，通过迭代优化策略将2D基础模型的先验（语义掩码、点追踪、深度）融入3D表示，实现了时空一致的运动和语义建模，在视频对象分割、点追踪和新视角合成任务上显著超越了现有方法。

**[Motion Matters Compact Gaussian Streaming For Free-Viewpoint Video Reconstructio](3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)**

:   提出ComGS框架，利用动态场景中运动的局部性和一致性，通过仅约200个关键点驱动整个运动区域的高斯点运动，实现了相比3DGStream 159倍、相比QUEEN 14倍的存储压缩，同时保持了竞争性的视觉质量和渲染速度。

**[Mpmavatar Learning 3D Gaussian Avatars With Accurate And Robust Physics-Based Dy](3d_vision/mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)**

:   MPMAvatar 将 Material Point Method (MPM) 物理仿真器与 3D 高斯溅射渲染相结合，通过各向异性本构模型和面向网格碰撞体的新碰撞处理算法，实现宽松衣物的精确鲁棒物理动画——在 ActorsHQ 和 4D-DRESS 上几何和外观全面超越 PhysAvatar，仿真成功率 100% vs 37.6%，单帧仿真仅需 1.1 秒。

**[Multi-Scale Finetuning For Encoder-Based Time Series Foundation Models](3d_vision/multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)**

:   提出 MSFT（Multi-Scale FineTuning），通过因果分析揭示 naive 微调忽略尺度混淆问题，设计多尺度建模框架对 encoder-based 时间序列基础模型进行高效微调，显著超越 naive 微调和从头训练的 SOTA 方法。

**[Nerfbaselines Consistent And Reproducible Evaluation Of Novel View Synthesis Met](3d_vision/nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)**

:   提出NerfBaselines评测框架，通过统一的评估协议、环境隔离和原始代码封装，解决了新视角合成领域因评估协议差异导致的不公平比较问题，并通过实验揭示了微小的协议差异（如图像缩放方式、背景颜色）可以显著改变方法排名。

**[Neural Greens Functions](3d_vision/neural_greens_functions.md)**

:   提出 Neural Green's Function，一种基于特征分解的可学习线性 PDE 解算子：从域几何中提取逐点特征来预测 Green 函数的特征分解，一次训练即可对任意源函数和边界条件通过数值积分求解，在机械零件热分析上比 SOTA 神经算子误差降低 13.9% 且比数值求解器快 350 倍。

**[Novel Class Discovery For Point Cloud Segmentation Via Joint Learning Of Causal ](3d_vision/novel_class_discovery_for_point_cloud_segmentation_via_joint_learning_of_causal_.md)**

:   本文首次将因果学习引入3D点云新类发现（3D-NCD），通过结构因果模型（SCM）分析基类中的混杂因子和基-新类间的因果关系，提出因果表示原型学习（通过对抗网络消除混杂因子）和基于图的因果推理（GCN生成伪标签），在SemanticKITTI和SemanticPOSS上取得了SOTA结果。

**[Novel View Synthesis From A Few Glimpses Via Test-Time Natural Video Completion](3d_vision/novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)**

:   将稀疏输入新视角合成重新定义为测试时自然视频补全问题，利用预训练视频扩散模型的先验生成中间伪视图，并通过不确定性感知机制与 3D 高斯泼溅（3D-GS）迭代优化，在极稀疏输入下实现高保真场景重建。

**[Object-Centric Representation Learning For Enhanced 3D Semantic Scene Graph Pred](3d_vision/object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)**

:   通过实证分析揭示物体特征可区分性是 3D 场景图谓词预测的关键瓶颈（物体分类错误导致 92%+ 的谓词错误），提出独立对比预训练的物体编码器（3D-2D-Text 三模态对齐）+ 几何正则化关系编码器 + 双向边门控 GNN，在 3DSSG 上 Object R@1 59.53%、Predicate R@50 91.40% 均达新 SOTA。

**[On Geometry-Enhanced Parameter-Efficient Fine-Tuning For 3D Scene Segmentation](3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)**

:   提出 Geometry Encoding Mixer (GEM)，一种专为3D点云Transformer设计的几何感知PEFT模块，通过空间适配器捕获局部几何细节和上下文适配器注入全局场景信息，仅更新1.6%参数即可达到甚至超越全量微调性能。

**[Online Segment Any 3D Thing As Instance Tracking](3d_vision/online_segment_any_3d_thing_as_instance_tracking.md)**

:   将在线3D实例分割重新建模为实例跟踪问题（AutoSeg3D），通过长期记忆进行实例关联、短期记忆进行实例更新、以及空间一致性学习缓解VFM过分割，在ScanNet200上超越ESAM 2.8 AP并保持实时性。

**[Onlinesplatter Pose-Free Online 3D Reconstruction For Free-Moving Objects](3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)**

:   提出 OnlineSplatter，一个无需相机位姿、深度先验或全局优化的前馈式在线3D重建框架，通过双键记忆模块（外观-几何潜在键 + 方向键）实现自由移动物体的恒定时间增量重建。

**[Openlex3D A Tiered Evaluation Benchmark For Open-Vocabulary 3D Scene Representat](3d_vision/openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)**

:   提出 OpenLex3D，一个面向开放词汇 3D 场景表示的分层评测基准，在 Replica、ScanNet++、HM3D 三个数据集上提供 13 倍于原始标注的丰富语言标签，支持开放集 3D 语义分割和目标检索两项任务评测。

**[Orientation-Anchored Hyper-Gaussian For 4D Reconstruction From Casual Videos](3d_vision/orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)**

:   提出 OriGS (Orientation-anchored Gaussian Splatting)，通过全局方向场引导和方向感知超维高斯表示，实现从随手拍摄的单目视频中进行高质量4D动态场景重建。

**[Orientation Matters Making 3D Generative Models Orientation-Aligned](3d_vision/orientation_matters_making_3d_generative_models_orientation-aligned.md)**

:   提出朝向对齐3D物体生成任务，构建了跨1008个类别14832个朝向对齐3D模型的Objaverse-OA数据集，通过微调Trellis和Wonder3D两种主流3D生成框架实现朝向对齐的物体生成，并展示零样本朝向估计和箭头旋转操控两个下游应用。

**[Physx-3D Physical-Grounded 3D Asset Generation](3d_vision/physx-3d_physical-grounded_3d_asset_generation.md)**

:   PhysX提出了首个端到端的物理属性驱动3D资产生成范式，包括PhysXNet（首个系统标注了绝对尺度、材料、功能可供性、运动学和功能描述五个维度的物理3D数据集，含26K+对象）和PhysXGen（双分支前馈生成框架，将物理知识注入预训练的3D结构空间中）。

**[Pixel-Perfect Depth With Semantics-Prompted Diffusion Transformers](3d_vision/pixel-perfect_depth_with_semantics-prompted_diffusion_transformers.md)**

:   提出Pixel-Perfect Depth——在像素空间（而非潜空间）直接做扩散生成的单目深度估计模型，通过语义提示DiT（SP-DiT）引入视觉基础模型的高层语义表示和级联DiT设计，生成无飞点（flying-pixel-free）的深度图，在五个benchmark上超越所有已发表的生成式模型。

**[Plana3R Zero-Shot Metric Planar 3D Reconstruction Via Feed-Forward Planar Splatt](3d_vision/plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)**

:   提出Plana3R，一个无需位姿和平面标注的前馈框架，从未配对的双视角图像中预测稀疏3D平面基元和度量尺度相对位姿，实现室内场景的零样本度量平面3D重建。

**[Planargs High-Fidelity Indoor 3D Gaussian Splatting Guided By Vision-Language Pl](3d_vision/planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)**

:   利用视觉语言基础模型（GroundedSAM）检测平面区域，结合DUSt3R多视图深度先验，通过共面约束和几何先验监督优化3DGS，实现室内场景的高保真表面重建。

**[Pointmac Meta-Learned Adaptation For Robust Test-Time Point Cloud Completion](3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)**

:   提出 PointMAC，首个将元辅助学习和测试时适应（TTA）引入点云补全的框架：通过 Bi-Aux Units（随机掩码重建+噪声去除）提供自监督信号，MAML 对齐辅助目标与主任务，推理时仅更新共享编码器实现样本级精化，在合成/模拟/真实数据上达到 SOTA。

**[Quantifying And Alleviating Co-Adaptation In Sparse-View 3D Gaussian Splatting](3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)**

:   本文揭示了稀疏视角 3D Gaussian Splatting 中外观伪影的核心成因——高斯体之间的协同适应（co-adaptation）现象，提出了 Co-Adaptation Score（CA）度量指标来量化这一纠缠程度，并设计了 Gaussian Dropout 和不透明度乘性噪声注入两种即插即用的正则化策略，在五种基线方法和三个数据集上均显著降低了 co-adaptation 并提升了新视角渲染质量。

**[Reconstruct Inpaint Test-Time Finetune Dynamic Novel-View Synthesis From Monocul](3d_vision/reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)**

:   提出 CogNVS，将动态场景新视角合成分解为三阶段管线——3D 重建（获取可见像素）→ 视频扩散修复（生成遮挡区域）→ 测试时微调（适应目标视频的分布），用纯 2D 视频自监督训练修复模型，实现零样本泛化到新测试视频。

**[Reconstructing The Local Density Field With Combined Convolutional And Point Clo](3d_vision/reconstructing_the_local_density_field_with_combined_convolutional_and_point_clo.md)**

:   提出一种混合卷积（U-Net）与点云（DeepSets）的神经网络架构，用于从暗物质晕的视线方向特异速度重建局部暗物质密度场，在小尺度上显著优于纯卷积和线性重建方法。

**[Rectified Point Flow Generic Point Cloud Pose Estimation](3d_vision/rectified_point_flow_generic_point_cloud_pose_estimation.md)**

:   提出 Rectified Point Flow，一种统一的生成式框架，将成对点云配准和多部件形状组装统一为条件生成问题，通过学习连续点级速度场来估计部件位姿。

**[Rgb-Only Supervised Camera Parameter Optimization In Dynamic Scenes](3d_vision/rgb-only_supervised_camera_parameter_optimization_in_dynamic_scenes.md)**

:   ROS-Cam 提出仅用单个RGB视频作为监督的动态场景相机参数（焦距+位姿）优化方法，通过Patch-wise跟踪过滤器建立稀疏鲁棒对应关系、Cauchy分布异常值感知联合优化自适应降权运动物体、以及基于Softplus/凸极小分析的两阶段优化策略，在5个数据集上以最少监督实现最优精度和最快速度。

**[Riganyface Scaling Neural Facial Mesh Auto-Rigging With Unlabeled Data](3d_vision/riganyface_scaling_neural_facial_mesh_auto-rigging_with_unlabeled_data.md)**

:   提出RigAnyFace（RAF），一个可扩展的面部网格自动绑定框架，通过2D监督策略利用无标注中性网格扩大训练规模，实现对多种拓扑和断连组件（如眼球）的高质量FACS混合形状绑定。

**[Robust Neural Rendering In The Wild With Asymmetric Dual 3D Gaussian Splatting](3d_vision/robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)**

:   AsymGS利用一个关键观察——野外训练数据引起的重建伪影具有随机性——提出非对称双3DGS框架，通过互补掩码策略和一致性约束抑制伪影，并引入Dynamic EMA Proxy实现高效训练，在多个野外数据集上显著超越现有方法。

**[Rogr Relightable 3D Objects Using Generative Relighting](3d_vision/rogr_relightable_3d_objects_using_generative_relighting.md)**

:   本文提出ROGR，利用多视角扩散重光照模型生成多光照条件下的一致图像，训练一个光照条件化的NeRF，实现任意环境光照下的前馈式3D物体重光照，在TensoIR和Stanford-ORB基准上达到SOTA性能且支持交互式渲染。

**[Scaffold Diffusion Sparse Multi-Category Voxel Structure Generation With Discret](3d_vision/scaffold_diffusion_sparse_multi-category_voxel_structure_generation_with_discret.md)**

:   提出Scaffold Diffusion，将稀疏多类别3D体素视为token序列，使用Masked Diffusion Language Model（MDLM）配合3D正弦位置编码，在条件占用图上生成空间连贯的多类别体素结构，在极端稀疏（>98%背景）的Minecraft房屋数据集上显著优于自回归和传统离散扩散baseline。

**[Sceneforge Enhancing 3D-Text Alignment With Structured Scene Compositions](3d_vision/sceneforge_enhancing_3d-text_alignment_with_structured_scene_compositions.md)**

:   提出SceneForge框架，通过将单个3D点云对象组合成带显式空间关系的多物体场景，配合LLM精炼的组合描述，增强3D-文本对比学习的数据多样性和复杂度，在多个下游任务上带来一致性能提升。

**[Sceneweaver All-In-One 3D Scene Synthesis With An Extensible And Self-Reflective](3d_vision/sceneweaver_all-in-one_3d_scene_synthesis_with_an_extensible_and_self-reflective.md)**

:   提出SceneWeaver，首个用于3D场景合成的反思型智能体框架，通过标准化可扩展的工具接口统一多种场景生成范式，并利用reason-act-reflect闭环迭代优化，在物理合理性、视觉真实感和语义对齐上全面超越现有方法。

**[Segment Then Splat Unified 3D Open-Vocabulary Segmentation Via Gaussian Splattin](3d_vision/segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)**

:   提出"先分割再重建"的新范式，在3D高斯溅射重建之前就将高斯分配到不同目标集合，从而消除几何和语义歧义，实现静态和动态场景的统一3D开放词汇分割。

**[Shallow Flow Matching For Coarse-To-Fine Text-To-Speech Synthesis](3d_vision/shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)**

:   提出 Shallow Flow Matching（SFM），在粗到细 TTS 框架中利用弱生成器输出构建 flow matching 中间状态，使推理从中间状态而非纯噪声出发，同时提升合成质量和加速推理。

**[Singref6D Monocular Novel Object Pose Estimation With A Single Rgb Reference](3d_vision/singref6d_monocular_novel_object_pose_estimation_with_a_single_rgb_reference.md)**

:   提出SingRef6D，一个仅需单张RGB参考图像的轻量级6D位姿估计流水线，通过token-scaler微调Depth-Anything v2实现鲁棒深度预测，并引入深度感知匹配增强LoFTR的空间推理能力，在透明/反光物体场景中大幅超越现有方法。

**[Sofar Language-Grounded Orientation Bridges Spatial Reasoning And Object Manipul](3d_vision/sofar_language-grounded_orientation_bridges_spatial_reasoning_and_object_manipul.md)**

:   提出"语义朝向"(Semantic Orientation)概念，用自然语言描述物体方向（如 USB 的"插入方向"、杯子的"把手方向"），构建 OrienText300K 大规模数据集训练 PointSO 模型实现零样本朝向预测，并集成为 SoFar 系统实现 6-DoF 场景理解与机器人操作。

**[Styl3R Instant 3D Stylized Reconstruction For Arbitrary Scenes And Styles](3d_vision/styl3r_instant_3d_stylized_reconstruction_for_arbitrary_scenes_and_styles.md)**

:   提出Styl3R前馈网络，通过结构-外观双分支架构将3D重建与风格化解耦，仅用未标定的稀疏视角图像和任意风格图像，在0.15秒内完成风格化3D重建。

**[Synchuman Synchronizing 2D And 3D Generative Models For Single-View Human Recons](3d_vision/synchuman_synchronizing_2d_and_3d_generative_models_for_single-view_human_recons.md)**

:   SyncHuman首次将2D多视图生成模型与3D原生生成模型统一在一个框架中，通过像素对齐的2D-3D同步注意力机制实现互补增强，在复杂人体姿态下实现了高保真纹理网格重建，几何精度和视觉质量均超越现有方法。

**[Tapip3D Tracking Any Point In Persistent 3D Geometry](3d_vision/tapip3d_tracking_any_point_in_persistent_3d_geometry.md)**

:   提出TAPIP3D，将视频表示为相机稳定化的时空3D特征点云，通过3D邻域到邻域（N2N）注意力机制在持久3D几何空间中迭代精化多帧点轨迹，显著超越现有3D点跟踪方法。

**[Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)**

:   提出首个端到端率失真（RD）优化的 4D 高斯泼溅压缩框架，通过 Haar 小波变换利用动态点轨迹的时序平滑先验，在 Ex4DGS 基础上实现高达 91× 的压缩率（平均模型仅约原始 1.1%），同时保持合理的渲染质量和灵活的率-质量权衡控制。

**[Tirex Zero-Shot Forecasting Across Long And Short Horizons With Enhanced In-Cont](3d_vision/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)**

:   提出基于xLSTM的预训练时间序列预测模型TiRex，通过连续片段掩码（CPM）策略和数据增强技术，在GiftEval和Chronos-ZS两大标准基准上以仅35M参数全面超越Chronos Bolt（200M）、TimesFM（500M）等大模型，同时在短期和长期零样本预测中均达到SOTA。

**[Towards 3D Objectness Learning In An Open World](3d_vision/towards_3d_objectness_learning_in_an_open_world.md)**

:   提出 OP3Det，一个无需文本提示的类无关开放世界 3D 检测器，通过 2D 基础模型进行 3D 物体发现，并设计跨模态混合专家（MoE）动态融合点云与图像特征，大幅提升新类别物体的召回率。

**[Tp-Mddn Task-Preferenced Multi-Demand-Driven Navigation With Autonomous Decision](3d_vision/tp-mddn_task-preferenced_multi-demand-driven_navigation_with_autonomous_decision.md)**

:   提出任务偏好多需求驱动导航（TP-MDDN）基准和AWMSystem自主决策系统，通过指令分解、动态目标选择和任务状态监控三个LLM模块配合多维度累积语义地图，实现长程多子任务导航。

**[Trim Scalable 3D Gaussian Diffusion Inference With Temporal And Spatial Trimming](3d_vision/trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)**

:   提出TRIM（Trajectory Reduction and Instance Mask denoising），一种后训练框架，通过时间维度的轨迹预筛选和空间维度的背景token裁剪来加速3D高斯扩散模型推理，同时提升生成质量，在T3Bench文本生成3D和GSO图像生成3D任务上均优于DiffSplat等基线。

**[U-Can Unsupervised Point Cloud Denoising With Consistency-Aware Noise2Noise Matc](3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)**

:   提出 U-CAN 无监督点云去噪框架，通过 Noise2Noise 匹配方案和几何一致性约束实现多步去噪路径推断，性能逼近有监督方法，且一致性约束可泛化到 2D 图像去噪。

**[Ugm2N An Unsupervised And Generalizable Mesh Movement Network Via M-Uniform Loss](3d_vision/ugm2n_an_unsupervised_and_generalizable_mesh_movement_network_via_m-uniform_loss.md)**

:   提出 UGM2N 无监督网格移动网络，通过局部化 Node Patch 表示和 M-Uniform 损失函数实现无监督训练，在无需预适应网格数据的条件下实现跨 PDE 类型和跨网格几何的零样本泛化，且不产生网格缠绕。

**[Umami Unifying Masked Autoregressive Models And Deterministic Rendering For View](3d_vision/umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)**

:   提出 UMAMI，一个统一掩码自回归模型（MAR）和确定性渲染的混合框架用于稀疏视角新视角合成：双向 Transformer 编码多视角图像 Token 和 Plücker 射线嵌入，两个轻量级 MLP 头分别处理可见区域（确定性回归）和遮挡区域（MAR 扩散生成），渲染速度比全生成基线快一个数量级。

**[Urdf-Anything Constructing Articulated Objects With 3D Multimodal Language Model](3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model.md)**

:   提出URDF-Anything，首个基于3D多模态大语言模型（MLLM）的端到端关节物体重建框架，通过[SEG] token机制实现几何分割与运动学参数的联合预测，在分割精度（mIoU提升17%）、参数误差（降低29%）和物理可执行性（超越基线50%）上均达到SOTA。

**[Va-Gs Enhancing The Geometric Representation Of Gaussian Splatting Via View Alig](3d_vision/va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)**

:   通过引入边缘感知图像监督、可见性感知的多视图光度对齐、法线约束和深度图像特征对齐四种视图对齐（View Alignment）策略，显著提升3D高斯溅射的几何表示精度，在表面重建和新视图合成上取得SOTA。

**[Visual Sync Multi-Camera Synchronization Via Cross-View Object Motion](3d_vision/visual_sync_multi-camera_synchronization_via_cross-view_object_motion.md)**

:   VisualSync提出了一个基于对极几何约束的多相机时间同步框架，利用预训练视觉模型（VGGT、CoTracker3、MAST3R）提取运动轨迹和跨视角对应关系，通过最小化Sampson误差来估计各相机的时间偏移，在四个数据集上达到了中位误差低于50ms的毫秒级同步精度。

**[Walking The Schrödinger Bridge A Direct Trajectory For Text-To-3D Generation](3d_vision/walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)**

:   从理论上证明SDS是Schrödinger Bridge的特例，并基于此提出TraCe框架——在当前渲染和文本条件目标之间构建显式扩散桥，通过LoRA微调学习桥轨迹的score dynamics，在低CFG值下实现高质量text-to-3D生成。

**[Wildcat3D Appearance-Aware Multi-View Diffusion In The Wild](3d_vision/wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)**

:   提出WildCAT3D，通过显式建模图像的全局外观条件，扩展多视角扩散模型（CAT3D）从野外互联网数据（如旅游照片）中学习场景级新视角合成，同时支持外观控制生成。

**[Zpressor Bottleneck-Aware Compression For Scalable Feed-Forward 3Dgs](3d_vision/zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)**

:   从信息瓶颈（Information Bottleneck）原理出发分析前馈式3DGS的容量瓶颈，提出轻量级、与架构无关的ZPressor模块，通过将多视角输入压缩为紧凑的锚点视角表示，使现有模型能扩展到100+输入视角（480P，80GB GPU），在DL3DV-10K和RealEstate10K上持续提升性能。

---

## 🛡️ AI安全 { #ai_safety }

**[A Set Of Generalized Components To Achieve Effective Poison-Only Clean-Label Bac](ai_safety/a_set_of_generalized_components_to_achieve_effective_poison-only_clean-label_bac.md)**

:   提出一组通用化组件（Component A/B/C），通过充分挖掘样本选择与触发器之间的双向协作关系，同时提升 Poison-only Clean-label 后门攻击的攻击成功率（ASR）和隐蔽性，并在多种攻击类型上展现了良好的泛化能力。

**[Adaptive Lora Experts Allocation And Selection For Federated Fine-Tuning](ai_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)**

:   提出 FedLEASE——解决联邦 LoRA 微调中两个关键问题：(1) 用 LoRA B 矩阵相似度聚类自动确定最优专家数量和分配，(2) 用扩展路由空间（$2M-1$ 维）实现自适应 top-M 专家选择（每个客户端自动决定用几个专家），在 GLUE 上比最强基线平均提升 5.53%。

**[Adversarial Paraphrasing A Universal Attack For Humanizing Ai-Generated Text](ai_safety/adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)**

:   提出 Adversarial Paraphrasing——一种无需训练的通用攻击框架，在逐 token 改写时利用 AI 文本检测器的反馈信号选择"最像人写"的 token，使改写后的 AI 文本在 8 种检测器上平均 T@1%F 下降 87.88%，且具有跨检测器的强迁移性。

**[Agentstealth Reinforcing Large Language Model For Anonymizing User-Generated Tex](ai_safety/agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)**

:   提出 AgentStealth 框架，通过对抗式匿名化工作流、监督微调（SFT）和在线强化学习三阶段训练小型语言模型（SLM），实现在保持文本效用的同时有效匿名化用户生成内容，匿名化效果提升12.3%、效用提升6.8%。

**[Ai Should Sense Better Not Just Scale Bigger Adaptive Sensin](ai_safety/ai_should_sense_better_not_just_scale_bigger_adaptive_sensin.md)**

:   这篇立场论文受生物感觉系统的启发，主张AI研究必须从单纯的"扩模型"范式转向"优化输入"——通过在传感器层面动态调整参数（曝光、增益、多模态配置等），使小模型（5M参数的EfficientNet-B0）在理想传感器适应下超越大模型（632M参数的OpenCLIP-H），并提出了从单次感知到闭环感知-运动耦合的渐进式形式化框架。

**[Almguard Safety Shortcuts And Where To Find Them As Guardrails For Audio-Languag](ai_safety/almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)**

:   首个针对音频语言模型（ALM）越狱攻击的防御框架——发现对齐过的 ALM 存在可被激活的潜在安全快捷路径（safety shortcuts），通过 Mel 梯度稀疏掩码（M-GSM）定位关键频率段，施加快捷路径激活扰动（SAP），将平均攻击成功率从 41.6% 降至 4.6%，同时几乎不影响正常任务性能。

**[Artificial Hivemind The Open-Ended Homogeneity Of Language Models And Beyond](ai_safety/artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)**

:   构建了 Infinity-Chat 数据集（26K 开放式真实用户查询 + 31,250 条人类标注），揭示了 LM 在开放式生成中的"Artificial Hivemind"效应——模型内重复和模型间同质化严重，并发现 Reward Model 和 LM Judge 在个体偏好差异大的样本上校准失败。

**[Beyond Last-Click An Optimal Mechanism For Ad Attribution](ai_safety/beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)**

:   从博弈论角度分析广告归因中 Last-Click 机制的策略操纵漏洞——平台可以通过篡改时间戳获取不公正的归因信用，提出 Peer-Validated Mechanism（PVM）——每个平台的信用仅取决于其他平台的报告（类比同行评审），理论证明 PVM 是占优策略激励兼容（DSIC）且在同质设置下最优，准确率从 34% 提升到 75%（2 平台）。

**[Bits Leaked Per Query Information-Theoretic Bounds On Adversarial Attacks Agains](ai_safety/bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)**

:   将 LLM 对抗攻击建模为信息通道问题——定义每次查询的"泄漏比特数" $I(Z;T)$ 为攻击目标属性 $T$ 与可观测信号 $Z$ 的互信息，证明攻击达到误差 $\varepsilon$ 所需最少查询数为 $\log(1/\varepsilon)/I(Z;T)$，在 7 个 LLM 上验证：暴露 answer tokens 需 ~1000 次查询，加 logits 降到 ~100 次，加思维链降到 ~几十次，为透明性-安全性权衡提供首个原则性标尺。

**[Boosting Adversarial Transferability With Spatial Adversarial Alignment](ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)**

:   提出 Spatial Adversarial Alignment (SAA)，通过空间感知对齐和对抗感知对齐两个模块微调代理模型，使其特征与见证模型对齐，从而显著提升对抗样本的跨架构迁移性（CNN→ViT 迁移率提升 25-39%）。

**[Bridging Symmetry And Robustness On The Role Of Equivariance In Enhancing Advers](ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)**

:   通过在 CNN 中嵌入旋转等变（P4群）和尺度等变卷积层，提出 Parallel 和 Cascaded 两种对称性感知架构，无需对抗训练即可显著提升对抗鲁棒性，并从 CLEVER 框架出发理论证明等变架构能压缩假设空间、正则化梯度、收紧认证鲁棒性界。

**[Causally Reliable Concept Bottleneck Models](ai_safety/causally_reliable_concept_bottleneck_models.md)**

:   提出 C2BM（Causally reliable Concept Bottleneck Models），将概念瓶颈（concept bottleneck）按照因果图结构化组织，通过结合观测数据与背景知识自动学习因果关系，在保持分类精度的同时显著提升因果可靠性、干预响应和公平性。

**[Collective Narrative Grounding Community-Coordinated Data Contributions To Impro](ai_safety/collective_narrative_grounding_community-coordinated_data_contributions_to_impro.md)**

:   提出 Collective Narrative Grounding 协议，通过参与式工坊收集社区叙事并结构化为"叙事单元"，用 RAG 管道将本地知识注入 LLM 问答系统，在 LocalBench 上发现 76.7% 的错误可由本地叙事直接修复，GPT-5 在参与式 QA 集上仅 21% 正确率凸显了本地知识鸿沟。

**[Contextual Integrity In Llms Via Reasoning And Reinforcement Learning](ai_safety/contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)**

:   提出 CI-RL 框架，通过 Chain-of-Thought 推理提示 + GRPO 强化学习，用仅约 700 个合成样本训练 LLM 理解"上下文完整性"（contextual integrity），在 PrivacyLens 基准上将隐私泄露率降低最高 40%，且小模型训练后可超越更大基线模型。

**[Coreguard Safeguarding Foundational Capabilities Of Llms Against Model Stealing ](ai_safety/coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)**

:   提出 CoreGuard，通过行置换（row permutation）锁定 Transformer 线性层权重 + 列置换传播协议（propagation protocol）将 TEE 授权次数降至 1 次，以极低计算和通信开销保护边缘部署 LLM 的基础能力不被模型窃取攻击利用。

**[Cost Efficient Fairness Audit Under Partial Feedback](ai_safety/cost_efficient_fairness_audit_under_partial_feedback.md)**

:   在部分反馈（partial feedback）设定下，提出了一套包含新颖成本模型的公平性审计框架，分别在黑盒与混合模型两种场景给出近最优审计算法，审计成本比自然基线降低约 50%。

**[Cpret A Dataset Benchmark And Model For Retrieval In Competitive Programming](ai_safety/cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)**

:   针对竞赛编程中重复/相似题目泛滥导致比赛不公平及 LLM 评测分数虚高的问题，构建了包含四种检索任务的大规模基准 CPRet，并提出 Group-InfoNCE 损失训练的专用检索模型 CPRetriever，在所有任务上超越 20+ 现有嵌入模型，同时揭示了题目相似性对 LiveCodeBench 评测的系统性偏差。

**[Cryptomoe Privacy-Preserving And Scalable Mixture Of Experts Inference Via Balan](ai_safety/cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)**

:   首个支持 MoE 架构 LLM 隐私推理的框架 CryptoMoE，通过平衡专家路由隐藏路由信息、置信度感知调度协议和批量密文矩阵乘法协议，相比 dense baseline 实现 2.8~3.5× 延迟降低和 2.9~4.3× 通信量降低，准确率损失仅 0.8%。

**[Ctrl-Alt-Deceit Sabotage Evaluations For Automated Ai Rd](ai_safety/ctrl-alt-deceit_sabotage_evaluations_for_automated_ai_rd.md)**

:   扩展 MLE-Bench 构建了 20 个代码破坏(code-sabotage)任务和 sandbagging 评测，发现前沿 AI agent 能在完成正常 ML 工程任务的同时成功植入后门等破坏，且在部分情况下逃避 LM monitor 的检测。

**[Deeppersona A Generative Engine For Scaling Deep Synthetic Personas](ai_safety/deeppersona_a_generative_engine_for_scaling_deep_synthetic_personas.md)**

:   提出 DeepPersona——一个两阶段分类引导的合成人格生成引擎：先从真实用户-ChatGPT 对话中挖掘构建 8000+ 节点的人类属性分类树，再通过渐进式属性采样生成平均 200+ 结构化属性的叙事完整人格，在个性化 QA 准确率上提升 11.6%，社会调查模拟偏差缩小 31.7%。

**[Design Encrypted Gnn Inference Via Server-Side Input Graph Pruning](ai_safety/design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)**

:   提出 DESIGN 框架，在全同态加密(FHE)下通过服务器端输入图剪枝和自适应多项式激活度分配两阶段优化，相比 SEAL 基线加速 FHE GNN 推理约 2× 并维持有竞争力的准确率。

**[Dictpfl Efficient And Private Federated Learning On Encrypted Gradients](ai_safety/dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)**

:   提出 DictPFL 框架，通过将模型权重分解为静态字典+可训练查找表，并结合加密感知剪枝，在联邦学习中实现全梯度同态加密保护的同时，将通信开销降低 402–748 倍、训练速度提升 28–65 倍，运行时间仅为明文 FL 的 2 倍以内。

**[Differential Privacy For Euclidean Jordan Algebra With Applications To Private S](ai_safety/differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)**

:   提出了基于 Euclidean Jordan Algebra (EJA) 的通用 Gaussian 隐私机制，并在此基础上设计了首个差分隐私的 Symmetric Cone Programming (SCP) 求解算法，解决了 Hsu et al. (ICALP 2014) 提出的关于差分隐私半定规划的重要开放问题。

**[Differentially Private Bilevel Optimization Efficient Algorithms With Near-Optim](ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)**

:   本文系统研究差分隐私 (DP) 下的双层优化问题，在凸情形下通过指数机制和正则化指数机制给出近紧的上下界（匹配单层 DP-ERM 最优率），在非凸情形下提出二阶 DP 方法实现不依赖内层维度的 SOTA 收敛率。

**[Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](ai_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)**

:   提出FedASK框架，通过**双阶段sketching流水线**（randomized SVD启发），首次在差分隐私下实现联邦LoRA中**两个低秩矩阵A和B的同步有效更新**，在Llama-2 7B/13B上MMLU提升最高11.5%，GSM8K提升46%。

**[Differentially Private High-Dimensional Variable Selection Via Integer Programmi](ai_safety/differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)**

:   本文提出两种纯差分隐私的稀疏变量选择方法 (top-R 和 mistakes)，利用现代混合整数规划 (MIP) 技术高效探索非凸目标景观，在高维设置（p 达 10000）下实现 SOTA 支撑集恢复率，同时提供理论恢复保证。

**[Distributional Adversarial Attacks And Training In Deep Hedging](ai_safety/distributional_adversarial_attacks_and_training_in_deep_hedging.md)**

:   本文首次将分布对抗攻击引入深度对冲框架，提出基于 Wasserstein 球的可计算对抗训练方法（WPGD 和 WBPGD），显著提升了对冲策略在分布偏移和真实市场数据下的鲁棒性与样本外表现。

**[Distributive Fairness In Large Language Models Evaluating Alignment With Human V](ai_safety/distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)**

:   本文系统评估多个 SOTA LLM（GPT-4o、Claude-3.5S、Llama3-70b、Gemini-1.5P）在非策略性资源分配任务中的分配公平性偏好，发现 LLM 与人类存在显著偏差：LLM 偏好效率和无嫉妒性 (EF) 而忽视人类更看重的公平性/平等性 (EQ)，但在选择题模式下 GPT-4o 和 Claude 能正确识别公平方案。

**[Dna-Detectllm Unveiling Ai-Generated Text Via A Dna-Inspired Mutation-Repair Par](ai_safety/dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)**

:   本文提出 DNA-DetectLLM，一种受 DNA 突变修复机制启发的零样本 AI 文本检测方法，通过构造理想 AI 序列并量化将输入文本修复到该序列的累积难度作为检测信号，在多个基准数据集上取得 AUROC 相对提升 5.55%、F1 提升 2.08% 的 SOTA 效果。

**[Dual-Flow Transferable Multi-Target Instance-Agnostic Attacks Via In-The-Wild Ca](ai_safety/dual-flow_transferable_multi-target_instance-agnostic_attacks_via_in-the-wild_ca.md)**

:   本文提出 Dual-Flow 框架，利用预训练扩散模型的正向 ODE 流和微调 LoRA 速度函数的逆向流进行多目标实例无关对抗攻击，通过级联分布偏移训练策略显著提升迁移攻击成功率（从 Inc-v3 到 Res-152 成功率提升 34.58%），在防御模型上也表现出强鲁棒性。

**[Efficient Fairness-Performance Pareto Front Computation](ai_safety/efficient_fairness-performance_pareto_front_computation.md)**

:   提出 MIFPO 方法，无需训练复杂的公平表示模型即可高效计算公平性-性能 Pareto 前沿，通过理论分析将问题化简为紧凑的离散凹优化问题。

**[Efficient Verified Machine Unlearning For Distillation](ai_safety/efficient_verified_machine_unlearning_for_distillation.md)**

:   提出 PURGE 框架，通过教师-学生 constituent mapping 和增量式多教师蒸馏策略，将 SISA 的验证式遗忘扩展到知识蒸馏场景，在教师端遗忘时仅需部分重训学生模型，实现至少 $N\times$ 的加速。

**[Enabling Differentially Private Federated Learning For Speech Recognition Benchm](ai_safety/enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)**

:   首次为端到端ASR建立FL+DP的实用基准，通过**逐层裁剪（per-layer clipping）**结合**LAMB优化器**的层级梯度归一化，在强隐私保证下实现仅1.3%~4.6%的WER绝对退化。

**[Enhancing Clip Robustness Via Crossmodality Alignment](ai_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)**

:   提出COLA——一个training-free的框架，通过将对抗扰动后的图像特征投影到文本特征张成的子空间来消除非语义噪声，再用最优传输(OT)在分布层面细粒度对齐图文特征，在14个零样本分类基准上平均提升6.7%的对抗鲁棒准确率，同时维持干净样本性能。

**[Enhancing Graph Classification Robustness With Singular Pooling](ai_safety/enhancing_graph_classification_robustness_with_singular_pooling.md)**

:   首次系统分析 flat pooling（Sum/Avg/Max）对图分类对抗鲁棒性的影响，推导各自的对抗风险上界，并提出 RS-Pool——利用节点嵌入矩阵的主奇异向量构建图级表示，在不牺牲 clean accuracy 的前提下显著提升对抗鲁棒性。

**[Environment Inference For Learning Generalizable Dynamical System](ai_safety/environment_inference_for_learning_generalizable_dynamical_system.md)**

:   提出 DynaInfer 框架，通过分析固定神经网络的预测误差来推断未标注轨迹的环境标签，实现无环境标签条件下的动态系统泛化学习，在 ODE/PDE 系统上性能匹配甚至超越 Oracle（已知标签）。

**[Evaluating The Promise And Pitfalls Of Llms In Hiring Decisions](ai_safety/evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)**

:   在约 10,000 个真实招聘候选人-职位配对上系统评测了 GPT-4o/4.1、Claude 3.5、Gemini 2.5、Llama 3.1/4、DeepSeek R1 等主流 LLM 的招聘匹配表现，发现专用领域模型 Match Score 在准确性（AUC 0.85 vs 0.77）和公平性（种族 IR 0.957 vs ≤0.809）上全面优于通用 LLM。

**[Exploring The Limits Of Strong Membership Inference Attacks On Large Language Mo](ai_safety/exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)**

:   首次将强成员推断攻击（LiRA）扩展到10M~1B参数的GPT-2规模LLM，训练超过4000个参考模型，揭示四个关键发现：强MIA可以在LLM上成功但效果有限（AUC<0.7），且大量个体样本决策在训练随机性下**与抛硬币无法区分**。

**[Factor Decorrelation Enhanced Data Removal From Deep Predictive Models](ai_safety/factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)**

:   提出 DecoRemoval 框架，通过判别性保持的因子去相关（基于随机傅里叶特征的空间映射+自适应权重）和平滑损失扰动两大模块，在不重训的前提下实现数据移除，尤其在分布外（OOD）场景下显著优于现有方法。

**[Fair Minimum Labeling Efficient Temporal Network Activations For Reachability An](ai_safety/fair_minimum_labeling_efficient_temporal_network_activations_for_reachability_an.md)**

:   本文提出公平最小标注（FML）问题，旨在设计最小代价的时序边激活方案，使网络中各节点组均有足够的时序路径可达性以满足公平覆盖要求；证明该问题是 NP-hard 且难以近似，并基于概率树嵌入给出匹配下界的近似算法。

**[Fair Representation Learning With Controllable High Confidence Guarantees Via Ad](ai_safety/fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)**

:   提出 FRG（Fair Representation learning with high-confidence Guarantees），首个允许用户指定公平性阈值 $\varepsilon$ 和置信水平 $1-\delta$ 的公平表征学习框架：通过 VAE 候选选择 + 对抗推断最大化协方差 + Student's t-检验构造高置信上界，保证对**任意**下游模型和任务，$\Delta_{DP} \leq \varepsilon$ 以至少 $1-\delta$ 概率成立。

**[Faircontrast Enhancing Fairness Through Contrastive Learning And Customized Augm](ai_safety/faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)**

:   FairContrast 提出一种面向表格数据的公平对比学习框架，通过策略性的正对样本选择（将优势组有利结果样本与对应弱势组样本配对），结合有监督或自监督对比损失与交叉熵损失的端到端训练，在不引入额外公平约束损失的前提下显著降低了预测偏差，且精度损失极小。

**[Fairness Under Competition](ai_safety/fairness_under_competition.md)**

:   本文首次研究竞争环境下多个公平分类器的联合公平性问题，理论证明即使每个分类器都满足 Equal Opportunity (EO)，生态系统可能仍然不公平，且对偏差分类器进行公平性调整反而可能降低生态系统公平性。

**[Fedfact A Provable Framework For Controllable Group-Fairness Calibration In Fede](ai_safety/fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)**

:   提出FedFACT框架，通过刻画联邦学习下的**贝叶斯最优公平分类器**结构，将公平联邦学习分别在训练中（in-processing）化归为**个性化代价敏感学习**、在训练后（post-processing）化归为**双层优化**，首次实现多类别场景下全局公平性与局部公平性的可控协调，并提供收敛及泛化保证。

**[Fedrw Efficient Privacy-Preserving Data Reweighting For Enhancing Federated Lear](ai_safety/fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)**

:   FedRW 提出首个无需可信第三方的联邦学习隐私保护软去重框架，通过安全多方计算获取全局样本频率并进行频率感知的样本加权，在预处理上实现最高 28.78× 加速，在模型性能上实现约 11.42% 的 perplexity 改善。

**[Fedsvd Adaptive Orthogonalization For Private Federated Learning With Lora](ai_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)**

:   FedSVD 提出通过 SVD 对 LoRA 矩阵进行全局重参数化，在每轮通信后用聚合的 BA 乘积的右奇异向量更新 A 矩阵，避免 DP-SGD 下的二次噪声放大同时保持 A 的自适应能力，在多个 NLU 基准上一致超越固定 A 的基线。

**[Flux Efficient Descriptor-Driven Clustered Federated Learning Under Arbitrary Di](ai_safety/flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)**

:   Flux通过在客户端侧提取紧凑的分布描述符（边际P(X)均值/协方差 + 类条件P(Y|X)均值/协方差），在服务器端用自适应DBSCAN无监督聚类自动确定聚类数与分组，训练聚类专属模型，并在测试时仅凭特征描述符为无标签新客户端匹配最优模型——首次同时处理四种分布偏移且通信开销与FedAvg相当。

**[Forensichub A Unified Benchmark Codebase For All-Domain Fake Image Detection And](ai_safety/forensichub_a_unified_benchmark_codebase_for_all-domain_fake_image_detection_and.md)**

:   ForensicHub 提出首个统一所有域（Deepfake/IMDL/AIGC/文档篡改）的假图检测与定位基准平台，包含 4 个任务、23 个数据集、42 个模型、6 个骨干网络和 11 个 GPU 加速评估指标，通过模块化架构和适配器设计打破领域孤岛，并进行了 16 种跨域评估得出 8 条关键洞察。

**[Geo-Sign Hyperbolic Contrastive Regularisation For Geometrically Aware Sign Lang](ai_safety/geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)**

:   Geo-Sign 提出将骨架特征投影到 Poincaré 球模型的双曲空间中，通过双曲对比损失正则化 mT5 语言模型，使其感知手语运动的层次结构，仅用骨架数据就在 CSL-Daily 上超越了基于 RGB 的 SOTA 方法（BLEU-4 +1.81, ROUGE-L +3.03）。

**[Healthslm-Bench Benchmarking Small Language Models For Mobile And Wearable Healt](ai_safety/healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)**

:   首个系统评估小语言模型 (SLMs, 1-4B参数) 在移动与可穿戴健康监测任务上表现的基准，覆盖zero-shot/few-shot/指令微调三种范式，并在iPhone上验证了端侧部署的可行性。

**[Impact Of Dataset Properties On Membership Inference Vulnerability Of Deep Trans](ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)**

:   从理论和实验两方面揭示深度迁移学习中成员推断攻击 (MIA) 脆弱性与每类样本数之间存在幂律关系：随着每类样本数 $S$ 增加，MIA 优势按 $S^{-1/2}$ 下降，但保护最脆弱样本所需的数据量极大，凸显了差分隐私形式化保障的不可替代性。

**[Improved Balanced Classification With Theoretically Grounded Loss Functions](ai_safety/improved_balanced_classification_with_theoretically_grounded_loss_functions.md)**

:   提出两个理论驱动的代理损失函数族——广义Logit调整(GLA)损失和广义类感知加权(GCA)损失，为类别不平衡下的多类分类提供更强的理论保证和实证性能。

**[Incentivizing Time-Aware Fairness In Data Sharing](ai_safety/incentivizing_time-aware_fairness_in_data_sharing.md)**

:   提出了一个时间感知的数据共享框架，设计了新的激励机制（F6-F8）和两种奖励方案（时间感知奖励累计和时间感知数据估值），保证早加入协作的参与方能获得更高价值的奖励，同时兼顾公平性和个体理性。

**[Influence Functions For Edge Edits In Non-Convex Graph Neural Networks](ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)**

:   提出适用于非凸 GNN 的边编辑影响函数，通过 proximal Bregman 响应函数放松凸性假设，并同时考虑参数偏移和消息传播两方面的影响，支持边的删除和插入。

**[Invisibleink High-Utility And Low-Cost Text Generation With Differential Privacy](ai_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)**

:   提出 InvisibleInk 框架，通过差分裁剪（DClip）隔离敏感信息和 Top-$k^+$ 截断采样两项创新，将差分隐私长文本生成的计算成本降低 8 倍以上，首次实现不到非隐私生成 4-8 倍开销的高质量隐私文本生成。

**[Its Complicated The Relationship Of Algorithmic Fairness And Non-Discrimination ](ai_safety/its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)**

:   本文系统分析了欧盟AI法案（EU AI Act）中针对高风险AI系统的反歧视条款与机器学习算法公平性领域之间的复杂关系，揭示了法律条文在输入侧偏差检测、输出侧保护缺失、标准化挑战等方面的关键缝隙，为计算机科学与法学跨学科协作提供了基础框架。

**[Keep It Real Challenges In Attacking Compression-Based Adversarial Purification](ai_safety/keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)**

:   本文系统评估了基于图像压缩的对抗净化防御，发现重建图像的"真实感"（realism）是提升防御鲁棒性的关键因素——高真实感压缩模型在面对强自适应攻击时仍能保持显著鲁棒性，而这并非源于梯度掩蔽。

**[Learning-Augmented Facility Location Mechanisms For Envy Ratio](ai_safety/learning-augmented_facility_location_mechanisms_for_envy_ratio.md)**

:   针对一维设施选址问题中的**嫉妒比**（envy ratio）目标，设计了确定性和随机化的学习增强机制：确定性的 $\alpha$-BIM 机制在一致性和鲁棒性之间实现最优权衡，随机化的BAM机制进一步改善保证；同时解决了Ding等人提出的公开问题，将无预测的随机机制近似比从2改进至约1.8944。

**[Llm Strategic Reasoning Agentic Study Through Behavioral Gam](ai_safety/llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)**

:   本文提出基于行为博弈论的LLM战略推理评估框架，使用截断量子响应均衡(TQRE)量化推理深度τ，在13个矩阵博弈上评估22个SOTA模型，揭示推理风格差异和人口统计persona引发的偏差问题。

**[Locally Optimal Private Sampling Beyond The Global Minimax](ai_safety/locally_optimal_private_sampling_beyond_the_global_minimax.md)**

:   在本地差分隐私（LDP）下的采样问题中，提出**局部minimax**框架，利用公共数据分布 $P_0$ 定义的邻域约束，推导出闭式最优采样器，在理论和实验上均**一致优于全局minimax采样器**。

**[Machine Unlearning Doesnt Do What You Think Lessons For Generative Ai Policy And](ai_safety/machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)**

:   本文系统性地揭示了机器遗忘（Machine Unlearning）在生成式AI场景下的五大根本性错配——技术方法与政策目标之间存在不可忽视的鸿沟，论证了机器遗忘无法作为通用方案解决隐私、版权和安全问题，并为ML研究者和政策制定者提供了务实的认知框架。

**[Mars A Malignity-Aware Backdoor Defense In Federated Learning](ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)**

:   提出 MARS 防御方法，通过计算神经元的后门能量（Backdoor Energy）来感知模型的恶意程度，并利用 Wasserstein 距离聚类有效识别联邦学习中的后门模型。

**[Masksql Safeguarding Privacy For Llm-Based Text-To-Sql Via Abstraction](ai_safety/masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)**

:   提出 MaskSQL 框架，通过提示抽象（abstraction）将敏感的表名、列名和数据值替换为抽象符号后发送给远程 LLM，结合本地 SLM 做 schema linking 和 SQL 重建，在保护隐私同时超越 SLM-only 方案的 SQL 生成精度。

**[Matchings Under Biased And Correlated Evaluations](ai_safety/matchings_under_biased_and_correlated_evaluations.md)**

:   在两机构稳定匹配模型中引入评估相关性参数 $\gamma$（机构间评分的对齐程度），分析偏差 $\beta$ 和相关性 $\gamma$ 如何联合影响弱势群体的代表性比率，证明即使轻微的相关性损失也可导致代表性急剧下降，并提出公平性干预策略的 Pareto 前沿。

**[Mitigating Disparate Impact Of Differentially Private Learning Through Bounded A](ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)**

:   通过在自适应梯度剪裁中引入可调整的下界（bounded adaptive clipping），防止 clipping bound 在训练过程中过度萎缩，从而改善少数群体的精度，在 DP 约束下缓解算法不公平。

**[Mitigating Privacy-Utility Trade-Off In Decentralized Federated Learning Via F-D](ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)**

:   提出基于 f-DP 框架的两种去中心化联邦学习隐私记账方法——PN-f-DP 和 Sec-f-LDP，通过更精细的假设检验隐私度量，一致性地获得比 Rényi DP 更紧的隐私界，从而在相同隐私保证下减少噪声注入、提升模型效用。

**[Mixat Combining Continuous And Discrete Adversarial Training For Llms](ai_safety/mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)**

:   提出MixAT方法，将离散对抗攻击（PAP改写）与连续嵌入空间扰动相结合进行LLM对抗训练，在保持高效用的同时实现对多种攻击的鲁棒性（ALO-ASR从50%+降至20%以下），且训练成本仅与纯连续方法相当。

**[Model Inversion With Layer-Specific Modeling And Alignment For Data-Free Continu](ai_safety/model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)**

:   在无数据持续学习场景中，提出逐层模型反演（PMI）来加速图像合成，并通过类别级高斯特征建模和对比学习缓解合成-真实数据间的特征漂移，实现高效且高质量的无数据知识回放。

**[Mpcache Mpc-Friendly Kv Cache Eviction For Efficient Private Llm Inference](ai_safety/mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)**

:   本文提出MPCache，一个面向安全多方计算（MPC）的KV缓存淘汰框架，结合一次性静态淘汰和查询感知的动态选择，配合层次化聚类、线性化相似度近似和跨层索引共享等优化，在不牺牲LLM性能的前提下实现最高2.01倍延迟降低和8.37倍通信量削减。

**[Multi-Class Support Vector Machine With Differential Privacy](ai_safety/multi-class_support_vector_machine_with_differential_privacy.md)**

:   提出PMSVM框架，利用all-in-one多类SVM的单次数据访问特性，结合权重扰动和梯度扰动方法，在保持差分隐私的前提下显著降低多类SVM的隐私预算消耗，实现了更优的隐私-效用权衡。

**[Music Arena Live Evaluation For Text-To-Music](ai_safety/music_arena_live_evaluation_for_text-to-music.md)**

:   Music Arena是首个面向文本到音乐（TTM）生成的在线实时评估平台，通过LLM驱动的审核与路由系统解决TTM系统异构签名问题，收集包含细粒度聆听行为和自然语言反馈的多层次偏好数据，并通过月度滚动数据发布为社区提供可持续的开放偏好数据源。

**[Nearly-Linear Time Private Hypothesis Selection With The Optimal Approximation F](ai_safety/nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)**

:   首次提出在中心差分隐私模型下同时实现近线性时间复杂度和最优近似因子 $\alpha=3$ 的假设选择算法，解决了Bun等人（NeurIPS 2019）提出的开放问题。

**[Not All Deepfakes Are Created Equal Triaging Audio Forgeries For Robust Deepfake](ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)**

:   提出基于"最有害的深伪是质量最高的"这一前提的两阶段流水线：先用判别器过滤低质量伪造以减少噪声，再用仅在真实录音上训练的歌手识别模型进行声纹匹配，在多个数据集上一致超越基线。

**[Omnifc Rethinking Federated Clustering Via Lossless And Secure Distance Reconstr](ai_safety/omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)**

:   提出 OmniFC，一个模型无关的联邦聚类框架：通过 Lagrange 编码计算在有限域上精确重建全局成对距离矩阵，任意集中式聚类方法（K-Means/谱聚类/DBSCAN/层次聚类等）可直接在其上运行，仅需一轮通信，天然抵抗 Non-IID，在 7 个数据集上全面超越 k-FED/MUFC/FedSC 等专用方法。

**[On The Empirical Power Of Goodness-Of-Fit Tests In Watermark Detection](ai_safety/on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)**

:   系统性地评估了八种经典拟合优度（GoF）检验在 LLM 文本水印检测中的效果，发现 GoF 检验在检测功效和鲁棒性上均显著优于现有基线方法。

**[On The Hardness Of Conditional Independence Testing In Practice](ai_safety/on_the_hardness_of_conditional_independence_testing_in_practice.md)**

:   系统分析了基于核的条件独立性（CI）检验在实践中失败的根本原因：条件均值嵌入的估计误差是导致Type-I错误膨胀的核心因素，同时揭示了选择条件核$k_C$对检验功效至关重要但会加剧假阳性的内在张力。

**[On The Robustness Of Verbal Confidence Of Llms In Adversarial Attacks](ai_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)**

:   首次系统研究 LLM 语言化置信度（verbal confidence）在对抗攻击下的鲁棒性，提出基于扰动和越狱的攻击框架，揭示攻击可导致置信度下降最高 30%、答案翻转率高达 100%，且现有防御策略基本无效。

**[On The Sample Complexity Of Differentially Private Policy Optimization](ai_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)**

:   首次系统性研究差分隐私（DP）约束下策略优化的样本复杂度，提出统一的元算法框架，分析DP-PG、DP-NPG和DP-REBEL三种隐私策略优化算法，证明隐私代价通常仅作为样本复杂度的低阶项出现。

**[Optimal Adjustment Sets For Nonparametric Estimation Of Weighted Controlled Dire](ai_safety/optimal_adjustment_sets_for_nonparametric_estimation_of_weighted_controlled_dire.md)**

:   针对加权控制直接效应（WCDE）建立三项基础理论：唯一可识别性的充要条件、非参数估计的影响函数推导、以及最小化渐近方差的最优协变量调整集刻画。

**[Orbit -- Open Recommendation Benchmark For Reproducible Research With Hidden Tes](ai_safety/orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)**

:   提出ORBIT统一推荐系统基准，包含5个标准化公开数据集评估和基于真实浏览历史构建的隐私安全ClueWeb-Reco隐藏测试集，系统评估了12个推荐模型并引入LLM-QueryGen基线，揭示了现有方法在大规模真实推荐场景中的局限性。

**[Perturbation Bounds For Low-Rank Inverse Approximations Under Noise](ai_safety/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)**

:   首次给出在加性噪声下低秩逆近似 $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ 的非渐近谱范数扰动界，利用轮廓积分技术得到依赖特征间隙、谱衰减和噪声对齐的锐界，比经典全逆界改进高达 $\sqrt{n}$ 倍。

**[Poly-Guard Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](ai_safety/poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)**

:   提出首个**大规模、多领域、策略驱动**的安全护栏基准 Poly-Guard，从 150+ 真实行业安全策略中提取 400+ 风险类别和 1000+ 安全规则，生成 100K+ 实例覆盖 8 大安全关键领域，并系统评测 19 个护栏模型，揭示了领域特化、模型演进遗忘、模型缩放停滞、对抗脆弱性等 8 项关键发现。

**[Position Bridge The Gaps Between Machine Unlearning And Ai Regulation](ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)**

:   系统分析了机器遗忘（Machine Unlearning）在欧盟人工智能法案（AIA）合规中的六大潜在应用场景，指出每个场景中 SOTA 与实际需求之间的技术差距，呼吁研究社区弥补这些差距以释放机器遗忘在 AI 监管中的潜力。

**[Preserving Task-Relevant Information Under Linear Concept Removal](ai_safety/preserving_task-relevant_information_under_linear_concept_removal.md)**

:   SPLINCE通过构造一种斜投影(oblique projection)，在保证线性守护性（不可被线性分类器预测敏感属性）的同时，精确保留表征与目标标签之间的协方差，解决了现有概念擦除方法在移除敏感概念的同时误删任务相关信息的问题。

**[Private Continual Counting Of Unbounded Streams](ai_safety/private_continual_counting_of_unbounded_streams.md)**

:   提出基于对数扰动的新型矩阵分解方法，首次实现同时满足"无界流"、"平滑误差"和"近最优渐近误差"三大性质的差分隐私持续计数算法，对任意 $\alpha > 0$ 在时间步 $t$ 处的方差为 $O(\log^{2+2\alpha}(t))$。

**[Private Zeroth-Order Optimization With Public Data](ai_safety/private_zeroth-order_optimization_with_public_data.md)**

:   提出 PAZO 框架，利用公共数据引导私有零阶优化算法的梯度近似，在视觉和文本任务上实现了优于 DP-SGD 的隐私-效用权衡，同时获得最高 16 倍的速度提升。

**[Probabilistic Reasoning With Llms For K-Anonymity Estimation](ai_safety/probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)**

:   本文提出Branch框架，利用大语言模型将用户文本中的个人信息建模为贝叶斯网络的联合概率分布，分别估计各属性的条件概率后组合计算k-匿名值（全球匹配该信息的人数），在隐私风险估计任务上达到73%准确率，比o3-mini链式思维提升13%。

**[Provable Watermarking For Data Poisoning Attacks](ai_safety/provable_watermarking_for_data_poisoning_attacks.md)**

:   本文提出两种可证明的水印方案（后投毒水印和投毒并行水印），为数据投毒攻击提供透明性声明机制，理论证明在特定水印长度条件下可同时保证水印可检测性和投毒有效性。

**[Pubsub-Vfl Towards Efficient Two-Party Split Learning In Heterogeneous Environme](ai_safety/pubsub-vfl_towards_efficient_two-party_split_learning_in_heterogeneous_environme.md)**

:   本文提出PubSub-VFL，一种基于发布/订阅架构的高效两方纵向联邦学习框架，通过分层异步机制和基于系统画像的超参数优化，在保证隐私和模型精度的前提下实现2~7倍的训练加速和高达91%的计算资源利用率。

**[Pulse Practical Evaluation Scenarios For Large Multimodal Model Unlearning](ai_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)**

:   本文提出 PULSE 评估协议，从预训练知识遗忘和多次顺序遗忘的可持续性两个实际维度出发，揭示了现有遗忘方法在 LMM 上的严重不足——遗忘预训练知识会导致 90% 以上通用能力丧失，连续遗忘 5 次后模型泛化能力几乎完全崩溃。

**[Reconstruction And Secrecy Under Approximate Distance Queries](ai_safety/reconstruction_and_secrecy_under_approximate_distance_queries.md)**

:   在近似距离查询模型下，通过学习理论视角研究重建博弈（reconstruction game），证明了最优重建误差等于Chebyshev半径的几何特征刻画，并对欧氏凸空间的伪有限性给出了完整分类。

**[Reliabilityrag Effective And Provably Robust Defense For Rag-Based Web-Search](ai_safety/reliabilityrag_effective_and_provably_robust_defense_for_rag-based_web-search.md)**

:   ReliabilityRAG 提出了一种利用文档可靠性信号（如搜索排名）进行对抗防御的 RAG 框架，通过在矛盾图上寻找最大独立集（MIS）来识别一致的文档子集并优先选择高可靠性文档，提供可证明的鲁棒性保证，同时在良性场景和长文本生成任务上保持高准确率。

**[Reverse Engineering Human Preferences With Reinforcement Learning](ai_safety/reverse_engineering_human_preferences_with_reinforcement_learning.md)**

:   使用强化学习训练前导文本生成器来提升下游 LLM 的评分成绩,揭示了 LLM-as-a-Judge 评估框架的脆弱性,且该攻击方式几乎不可检测并具有跨模型迁移能力。

**[Rewind-To-Delete Certified Machine Unlearning For Nonconvex Functions](ai_safety/rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)**

:   本文提出R2D（Rewind-to-Delete），首个适用于一般非凸损失函数的一阶、黑盒认证机器遗忘算法，通过"回溯"到训练过程中的较早检查点再对保留数据执行梯度下降来实现数据删除，同时提供(ε,δ)认证遗忘保证和隐私-效用-效率的理论权衡。

**[Robust Graph Condensation Via Classification Complexity Mitigation](ai_safety/robust_graph_condensation_via_classification_complexity_mitigation.md)**

:   本文揭示图凝缩（GC）本质上是降低分类复杂度的过程，而对抗攻击恰好破坏了这一特性，据此提出MRGC框架，通过内在维度正则化、曲率感知流形平滑和类间流形解耦三个流形约束模块来增强GC的鲁棒性，首次在结构、特征和标签均可能被篡改的条件下系统研究GC鲁棒性。

**[Saemark Steering Personalized Multilingual Llm Watermarks With Sparse Autoencode](ai_safety/saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)**

:   提出SAEMark框架，利用稀疏自编码器（SAE）提取文本的语义特征浓度评分，通过推理阶段的特征引导拒绝采样实现多比特水印嵌入，无需修改模型权重或logits，天然支持黑盒API、多语言和代码等场景，在英文/中文/代码上均达到领先的水印精度与文本质量。

**[Seca Semantically Equivalent And Coherent Attacks For Eliciting Llm Hallucinatio](ai_safety/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)**

:   提出 SECA（Semantically Equivalent and Coherent Attacks），通过保持语义等价和语义连贯性的现实主义提示修改来诱发 LLM 幻觉，在多选 QA 任务上实现更高攻击成功率且几乎无语义错误。

**[Self-Refining Language Model Anonymizers Via Adversarial Distillation](ai_safety/self-refining_language_model_anonymizers_via_adversarial_distillation.md)**

:   提出 SEAL 框架，通过对抗蒸馏将 GPT-4 级 LLM 的文本匿名化能力蒸馏到 8B 小模型中，结合 SFT + DPO 训练和自我精炼机制，使小模型在隐私-效用权衡上达到甚至超越 GPT-4 匿名化器的水平，且可完全本地部署。

**[Sequentially Auditing Differential Privacy](ai_safety/sequentially_auditing_differential_privacy.md)**

:   提出基于序贯假设检验和核 MMD 统计量的差分隐私审计框架，可以在流式处理机制输出时随时有效地检测隐私违规，将所需样本量从现有方法的 50K 降低到数百个，并能在不到一次完整训练的过程中识别 DP-SGD 的隐私违规。

**[Some Optimizers Are More Equal Understanding The Role Of Optimizers In Group Fai](ai_safety/some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)**

:   本文首次系统研究了优化算法选择对深度学习群体公平性的影响，通过随机微分方程（SDE）分析和两个新定理证明，自适应优化器（RMSProp/Adam）比SGD更容易收敛到公平的极小值点，特别是在数据严重不平衡时。

**[Spectral Perturbation Bounds For Low-Rank Approximation With Applications To Pri](ai_safety/spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)**

:   建立了对称矩阵低秩近似在谱范数下的新型高概率扰动界，改进了经典 Eckart-Young-Mirsky 定理，并解决了差分隐私 PCA 中的一个公开问题。

**[Stochastic Regret Guarantees For Online Zeroth- And First-Order Bilevel Optimiza](ai_safety/stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)**

:   提出了一种新的搜索方向并证明基于该方向的一阶和零阶在线双层优化算法能够在不需要窗口平滑的条件下实现次线性随机双层遗憾保证，同时通过降低 oracle 依赖、并行更新和零阶 Hessian/Jacobian 估计来提升效率。

**[Swe-Sql Illuminating Llm Pathways To Solve User Sql Issues In Real-World Applica](ai_safety/swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)**

:   提出 BIRD-CRITIC 基准（首个 SQL 问题调试基准）和 Six-Gym 训练环境，并开发 Bird-Fixer 智能体，通过 f-Plan Boosting 策略将 14B 开源模型的 SQL 调试能力提升至超越 Claude-3.7-Sonnet 和 GPT-4.1 的水平，在保护数据隐私的同时实现高效的 SQL 问题修复。

**[Systematizing Llm Persona Design A Four-Quadrant Technical Taxonomy For Ai Compa](ai_safety/systematizing_llm_persona_design_a_four-quadrant_technical_taxonomy_for_ai_compa.md)**

:   提出 LLM persona 设计的四象限技术分类框架，沿"虚拟 vs 具身"和"情感陪伴 vs 功能增强"两轴，系统化分析了从虚拟伴侣、游戏 NPC 到护理机器人等不同场景下的技术栈、核心挑战和伦理风险。

**[Taught Well Learned Ill Towards Distillation-Conditional Backdoor Attack](ai_safety/taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)**

:   本文提出了蒸馏条件后门攻击（DCBA）范式及其实现方法SCAR，通过双层优化在教师模型中植入"休眠"后门，该后门在教师模型上不可检测但会在知识蒸馏过程中被激活传递到学生模型，即使蒸馏数据集完全干净。

**[The Unseen Threat Residual Knowledge In Machine Unlearning Under Perturbed Sampl](ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)**

:   发现机器遗忘的关键安全漏洞：即使遗忘后的模型在统计意义上与重训练模型不可区分，对遗忘样本施加微小对抗扰动后，遗忘模型仍能正确识别而重训练模型则失败——揭示了"残余知识"这一新型隐私风险。提出 RURK 微调策略，通过惩罚对扰动遗忘样本的正确预测来消除残余知识，在 CIFAR-10 和 ImageNet-100 上有效抑制 11 种遗忘方法的残余知识。

**[Toxictextclip Text-Based Poisoning And Backdoor Attacks On Clip Pre-Training](ai_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)**

:   提出 ToxicTextCLIP 框架，通过背景感知选择和背景驱动增强两个模块，在 CLIP 预训练阶段生成高质量对抗文本，实现高达 95.83% 投毒成功率和 98.68% 后门 Hit@1，且能突破 RoCLIP、CleanCLIP、SafeCLIP 三种防御。

**[Trans-Env A Framework For Evaluating The Linguistic Robustness Of Llms Against E](ai_safety/trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)**

:   提出Trans-EnV框架，结合语言学专家知识和LLM变换能力，将标准美式英语（SAE）数据集自动转换为38种英语变体（18种方言+20种ESL英语），揭示LLM在非标准英语上最高46.3%的性能下降，凸显了语言公平性问题。

**[Trap Targeted Redirecting Of Agentic Preferences](ai_safety/trap_targeted_redirecting_of_agentic_preferences.md)**

:   TRAP 提出了一种基于扩散模型的语义注入对抗框架，通过在 CLIP 嵌入空间中优化图像语义，在黑盒条件下以视觉自然的方式系统性地误导多个主流 VLM 智能体的决策偏好，在 LLaVA-34B、GPT-4o 等六个模型上实现了高达 100% 的攻击成功率。

**[Understanding And Improving Adversarial Robustness Of Neural Probabilistic Circu](ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)**

:   理论分析神经概率电路（NPC）的对抗鲁棒性仅取决于属性识别模型而与概率电路无关，并提出 RNPC 通过类级推理集成方式实现可证明的鲁棒性提升，在保持良性准确率的同时显著增强对抗鲁棒性。

**[Understanding Challenges To The Interpretation Of Disaggregated Evaluations Of A](ai_safety/understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)**

:   通过因果图模型分析表明，分组评估（disaggregated evaluation）中跨子群体的性能差异不一定意味着不公平，而可能是数据生成过程中分布差异的自然结果，建议结合因果假设和加权评估补充标准分组评估。

**[Unifying Proportional Fairness In Centroid And Non-Centroid Clustering](ai_safety/unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)**

:   将质心聚类(centroid)和非质心聚类(non-centroid)的比例公平性研究统一到"半质心聚类"框架中，证明了两者不可同时实现的不可能性定理，并设计了新算法在双度量损失下实现常数倍近似的核(core)保证。

**[Unifying Re-Identification Attribute Inference And Data Reconstruction Risks In ](ai_safety/unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)**

:   基于假设检验解释的 f-DP 框架，统一了差分隐私中重识别、属性推断和数据重建三类隐私风险的界定，提供更紧致一致的风险上界，使噪声校准可减少 20% 且不降低安全性。

**[Unlearning As Ablation Toward A Falsifiable Benchmark For Generative Scientific ](ai_safety/unlearning_as_ablation_toward_a_falsifiable_benchmark_for_generative_scientific_.md)**

:   本文提出将机器遗忘重新定义为认识论探针工具（"遗忘即消融"），通过系统性移除目标知识及其遗忘闭包后测试模型能否从公理出发重新推导，从而提供可证伪的测试来区分 LLM 是"真正生成新知识"还是"仅仅检索记忆片段"。

**[Virus Infection Attack On Llms Your Poisoning Can Spread Via Synthetic Data](ai_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)**

:   本文首次系统研究了合成数据在LLM训练中的安全风险，发现现有投毒/后门攻击难以通过合成数据传播，进而提出Virus Infection Attack (VIA)框架，通过劫持点搜索和外壳构造将投毒内容嵌入正常训练样本中，使恶意内容即使在干净查询下也能被模型生成并传播到下游模型。

**[When Ai Democratizes Exploitation Llm-Assisted Strategic Manipulation Of Fair Di](ai_safety/when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)**

:   本文通过在 Spliddit 公平分租平台上设计四种不同的协调操纵场景（排斥性合谋、防御性反击、善意合谋、成本最小化联盟），实证地证明 LLM 可以将原本需要深厚机制设计专业知识才能进行的算法操纵行为，降低为任何用户仅需一次自然语言对话即可完成的简单操作，从根本上颠覆了"算法复杂性即安全屏障"的传统假设。

---

## 📐 优化/理论 { #optimization }

**[A Single-Loop First-Order Algorithm For Linearly Constrained Bilevel Optimizatio](optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)**

:   针对下层问题带耦合线性约束的双层优化问题，提出单循环一阶算法 SFLCB，通过罚函数 + 增广拉格朗日重构消除 Hessian 依赖，将迭代复杂度从 $O(\epsilon^{-3}\log(\epsilon^{-1}))$ 改进至 $O(\epsilon^{-3})$。

**[A Theoretical Study On Bridging Internal Probability And Sel](optimization/a_theoretical_study_on_bridging_internal_probability_and_sel.md)**

:   提出首个针对基于采样的测试时缩放方法的理论框架，将推理误差分解为估计误差和模型误差，揭示了Self-Consistency收敛慢、Perplexity模型误差大的局限，并提出RPC方法融合两者优势，在7个基准上以50%的采样成本达到同等推理性能。

**[A Unified Approach To Submodular Maximization Under Noise](optimization/a_unified_approach_to_submodular_maximization_under_noise.md)**

:   本文提出一个统一的元算法框架，能将任何满足"鲁棒性"条件的精确子模最大化算法作为黑盒，自动转换为持久噪声值预言机下保持近似比（仅损失 $o(1)$）的算法，首次实现了非单调子模函数在拟阵约束和无约束设置下的最优近似比。

**[A Unified Stability Analysis Of Sam Vs Sgd Role Of Data Cohe](optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)**

:   通过线性稳定性分析框架，证明了"平坦极小值⇒好泛化"和"SGD偏好简单函数"是同一枚硬币的两面——数据一致性(coherence)同时控制着两者，且SAM通过更严格的稳定性条件进一步放大了简单性偏好。

**[Adaptive Algorithms With Sharp Convergence Rates For Stochas](optimization/adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)**

:   提出Ada-Minimax和Ada-BiO两个自适应算法，通过将动量归一化技术与新型在线噪声估计策略结合，首次在无需预知梯度噪声水平的情况下，为非凸-强凹极小极大和非凸-强凸双层优化达到sharp收敛率Õ(1/√T + √σ̄/T^{1/4})。

**[An Adaptive Algorithm For Bilevel Optimization On Riemannian Manifolds](optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)**

:   AdaRHD 是首个无需预知问题参数（强凸常数、Lipschitz 界、流形曲率）的黎曼双层优化自适应算法——通过逆累计梯度范数策略自适应选择步长，在三阶段框架中逐步求解下层问题/线性系统/上层更新，收敛速率 $O(1/\epsilon)$ 匹配非自适应方法，对初始步长选择鲁棒性远超 RHGD。

**[Asymptotically Stable Quaternion-Valued Hopfield-Structured Neural Network With ](optimization/asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)**

:   提出四元数值监督学习 Hopfield 结构神经网络 (QSHNN)，通过周期性投影策略保持权重矩阵的四元数结构一致性，并基于 Lyapunov 理论证明了不动点的存在唯一性和渐近稳定性，轨迹曲率有界保证机器人路径规划的平滑性。

**[Auto-Compressing Networks](optimization/auto-compressing_networks.md)**

:   Auto-Compressing Networks（ACN）用长程前向连接（所有层输出直接汇聚到最终输出）替代短残差连接，使得梯度的 Direct Gradient 成分远强于 Forward Gradient，隐式地将信息压缩到早期层——ViT 仅需 6 层达到标准 12 层性能，BERT 节省 75% 层数，还额外获得噪声鲁棒性（+6.4%）和持续学习抗遗忘（-18%）。

**[Automated Algorithm Design Via Nevanlinna-Pick Interpolation](optimization/automated_algorithm_design_via_nevanlinna-pick_interpolation.md)**

:   提出基于频域鲁棒控制理论中 Nevanlinna-Pick 插值的自动化算法设计框架，用于求解带等式约束的强凸优化问题，获得了矩阵乘法次数与收敛速率之间的最优权衡。

**[Autoopt A Dataset And A Unified Framework For Automating Optimization Problem So](optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)**

:   AutoOpt 构建了首个优化问题图像到代码的端到端框架——11554 张优化公式图像（手写+印刷）的 AutoOpt-11k 数据集 + M1 混合编码器（ResNet+Swin→mBART）图像转 LaTeX（BLEU 96.70）+ M2 DeepSeek-Coder LaTeX 转 PYOMO + M3 双层分解求解器，框架级成功率 94.20%。

**[Better Ntk Conditioning A Free Lunch From Relu Nonlinear Activation In Wide Neur](optimization/better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)**

:   证明 ReLU 激活函数对宽神经网络有一个此前未被注意的"免费"益处：(a) 在模型梯度特征空间中产生更好的数据分离（相似输入的角度在梯度空间中被放大），(b) 由此导致 NTK 矩阵条件数严格减小（相比线性网络）。深度进一步放大此效应——在无限宽然后无限深的极限下，所有数据对在梯度空间中等角分离（~75.5°），NTK 条件数收敛到仅依赖数据量 $n$ 的固定值 $(n+4)/3$。

**[Beyond Tildeosqrtt Constraint Violation For Online Convex Optimization With Adve](optimization/beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)**

:   研究带对抗约束的在线凸优化 (COCO)，通过引入可调参数 $\beta$ 实现 $\tilde{O}(T^\beta)$ 遗憾与 $\tilde{O}(T^{1-\beta})$ 约束违反之间的精确权衡，突破了此前 $\tilde{O}(\sqrt{T})$ 约束违反的已知最优界。

**[Bispectral Ot Dataset Comparison Using Symmetry-Aware Optimal Transport](optimization/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)**

:   提出 Bispectral Optimal Transport (BOT)，将离散最优传输中的代价矩阵从原始像素距离替换为 bispectrum（群 Fourier 不变量）距离，使得传输计划在保持信号结构的同时精确消除群作用（如旋转）带来的变异，在旋转变换的 MNIST 等数据集上将类别保持准确率从 33% 提升至 84%。

**[Brain-Like Variational Inference](optimization/brain-like_variational_inference.md)**

:   提出 FOND 框架（Free energy Online Natural-gradient Dynamics），从自由能最小化的第一原理推导出脉冲神经网络推断动力学，并实现 iPVAE（迭代泊松 VAE），在重建-稀疏性权衡、生物合理性和 OOD 泛化上优于标准 VAE 和预测编码模型。

**[Clean First Align Later Benchmarking Preference Data Cleaning For Reliable Llm A](optimization/clean_first_align_later_benchmarking_preference_data_cleaning_for_reliable_llm_a.md)**

:   本文提出 PrefCleanBench，首个系统评估 13 种偏好数据清洗方法在 LLM 对齐中效果的综合基准，覆盖多种数据集、模型架构和优化算法，揭示了数据预处理在负责任 AI 开发中被忽视但至关重要的角色。

**[Composing Global Solutions To Reasoning Tasks Via Algebraic Objects In Neural Ne](optimization/composing_global_solutions_to_reasoning_tasks_via_algebraic_objects_in_neural_ne.md)**

:   提出 CoGS 框架，证明二层二次激活网络在 Abelian 群乘法推理任务上的权重空间具有半环代数结构，损失函数中的 Sum Potential 是环同态映射，由此可从仅满足部分损失的局部解通过环加法和环乘法代数地组合出全局最优解，约 95% 的梯度下降解与理论构造精确匹配。

**[Constrained Network Slice Assignment Via Large Language Models](optimization/constrained_network_slice_assignment_via_large_language_models.md)**

:   揭示两层二次激活网络在 Abelian 群推理任务上训练时权重空间具有半环代数结构，提出 CoGS 框架通过环运算将部分解组合为全局最优解，约 95% 梯度下降解与理论构造精确匹配。

**[Contribution Of Task-Irrelevant Stimuli To Drift Of Neural Representations](optimization/contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)**

:   理论证明在线学习中任务无关刺激的统计特性（方差和维度）是表示漂移的重要驱动因素，在 Oja 规则、Similarity Matching、自编码器和监督两层网络中均观察到漂移率 $D \propto \lambda_\perp^2 (n-m)$，且学习噪声诱导的漂移具有各向异性几何特征，与高斯突触噪声的各向同性漂移定性不同。

**[Covariances For Free Exploiting Mean Distributions For Training-Free Federated L](optimization/covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)**

:   提出 FedCOF，仅利用客户端上传的类均值（class means）即可在服务器端无偏估计类协方差矩阵，从而在零训练、极低通信开销的条件下初始化全局分类器，性能媲美甚至超越需要传输二阶统计量的 Fed3R。

**[Dartquant Efficient Rotational Distribution Calibration For Llm Quantization](optimization/dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)**

:   DartQuant 提出基于分布校准的旋转矩阵优化方法，通过 Whip 损失将激活值分布推向均匀分布以减少量化误差，并用 QR-Orth 替代昂贵的流形优化器，在 70B 模型上实现 47× 加速和 10× 内存节省，首次在单张 3090 GPU 上完成大模型旋转校准。

**[Deep Taxonomic Networks For Unsupervised Hierarchical Prototype Discovery](optimization/deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)**

:   Deep Taxonomic Networks 提出一种基于完全二叉树混合高斯先验的深度潜变量模型，通过变分推断自动从无标签数据中发现层次化分类体系和各级原型聚类，无需预设类别数量，并在多个数据集上大幅超越 TreeVAE 等基线。

**[Do Neural Networks Need Gradient Descent To Generalize A Theoretical Study](optimization/do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)**

:   本文在矩阵分解（神经网络理论的经典测试平台）上证明了 Guess & Check（随机抽参数直到拟合训练集）的泛化能力随宽度增加而退化（首次证明存在 G&C 可证明劣于梯度下降的典范情况），但随深度增加而改善，揭示了宽度和深度对泛化的截然不同作用。

**[Doubly Robust Alignment For Large Language Models](optimization/doubly_robust_alignment_for_large_language_models.md)**

:   DRPO 借鉴因果推断中的双重稳健估计方法，提出一种偏好优化算法，当偏好模型或参考策略任一正确指定时即可保持一致性，在理论和实验上均优于 PPO/DPO 及其变体。

**[Dynaact Large Language Model Reasoning With Dynamic Action Spaces](optimization/dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)**

:   DynaAct 将 LLM 推理中的动作空间构建建模为子集选择问题，通过兼顾效用和多样性的子模函数在每步动态构建紧凑动作空间，在 6 个基准上显著优于 rStar、RAP 等方法，MATH-500 上比 rStar 高 6.8%。

**[Effective Policy Learning For Multi-Agent Online Coordination Beyond Submodular ](optimization/effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)**

:   提出 MA-SPL 和 MA-MPL 两个多智能体在线协调算法，通过"基于策略的连续扩展"技术突破次模性限制，首次在次模和弱次模目标函数上均实现最优 $(1 - c/e)$ 近似比，支持时变目标和仅局部反馈的实际约束。

**[Efficient Adaptive Federated Optimization](optimization/efficient_adaptive_federated_optimization.md)**

:   FedAda2/FedAda2++ 提出在联邦学习中实现高效的服务器-客户端联合自适应优化：客户端本地预条件器从零初始化（无需服务器传输），并可选地用 SM3 等内存高效优化器压缩本地统计量，在理论上保持与完整联合自适应相同的 $O(T^{-1/2})$ 收敛率，实测通信成本与 FedAvg 一致。

**[Efficient Federated Learning Against Byzantine Attacks And Data Heterogeneity Vi](optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)**

:   提出 Fed-NGA 算法，通过对客户端上传的梯度做归一化后加权平均来实现聚合，以 $\mathcal{O}(pM)$ 的极低时间复杂度同时抵御 Byzantine 攻击与数据异质性，并在非凸损失函数下首次证明了特定温和条件下的零最优性间隙收敛。

**[Emergence And Scaling Laws In Sgd Learning Of Shallow Neural Networks](optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)**

:   本文对浅层神经网络在线 SGD 学习加法模型（多个单指标函数叠加）的过程进行了精确分析，证明了每个教师神经元的学习呈现尖锐相变（emergence），而大量相变曲线的叠加自然产生平滑的幂律 scaling law。

**[Escaping Saddle Points Without Lipschitz Smoothness The Power Of Nonlinear Preco](optimization/escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)**

:   本文提出统一的充分条件连接 $(L_0,L_1)$-光滑性与各向异性光滑性两种广义光滑框架，证明非线性预条件梯度法（含梯度裁剪）在此放松条件下保持鞍点规避性质，并给出扰动变体以多项对数维数依赖达到二阶稳定点。

**[Estimation Of Stochastic Optimal Transport Maps](optimization/estimation_of_stochastic_optimal_transport_maps.md)**

:   提出适用于随机OT映射的传输误差指标 $\mathcal{E}_p$（由优化间隙与可行性间隙组成），在无需Brenier映射存在或唯一性的最小假设下，构造了计算高效的rounding估计器达到近最优收敛率 $\tilde{O}(n^{-1/(d+2p)})$，并推广至Hölder连续核与对抗污染场景，建立了首个通用OT映射估计理论。

**[Evaluating Llms For Combinatorial Optimization One-Phase And Two-Phase Heuristic](optimization/evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)**

:   本文提出一个结合 LLM 与进化算法的系统性评估框架，用于评估 LLM 在 2D 装箱问题上生成和优化启发式算法的能力，GPT-4o 在 2 轮迭代内即达到最优解，将平均箱数从 16 降至 15，空间利用率从 0.76-0.78 提升至 0.83。

**[Exact And Linear Convergence For Federated Learning Under Arbitrary Client Parti](optimization/exact_and_linear_convergence_for_federated_learning_under_arbitrary_client_parti.md)**

:   本文引入随机矩阵和时变图作为建模工具，将联邦学习的客户端参与和本地更新过程统一为矩阵乘法形式，并提出 FOCUS 算法（基于 push-pull 策略），在**任意客户端参与**和数据异构下首次实现精确收敛与线性收敛速率。

**[Exploring Landscapes For Better Minima Along Valleys](optimization/exploring_landscapes_for_better_minima_along_valleys.md)**

:   本文提出优化器适配器"E"，通过在梯度更新中加入梯度差分的指数移动平均 $\mathbf{a}_k = \text{EMA}(\mathbf{g}_k - \mathbf{g}_{k-1})$ 使优化器能在到达局部极小值后继续沿损失景观的"山谷"探索更低更平坦的极小值，适配后的 ALTO 在大批量训练中平均提升 2.5% 测试准确率。

**[Extragradient Method For L 0 L 1-Lipschitz Root-Finding Problems](optimization/extragradient_method_for_l_0_l_1-lipschitz_root-finding_problems.md)**

:   本文在 $\alpha$-对称 $(L_0,L_1)$-Lipschitz 条件下（放松经典 $L$-Lipschitz 假设）为 extragradient (EG) 方法提出自适应步长策略 $\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$，建立了强单调（线性收敛）、单调（次线性收敛）和 weak Minty（局部收敛）三类根问题的首个完整收敛保证。

**[Fedqs Optimizing Gradient And Model Aggregation For Semi-Asynchronous Federated ](optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)**

:   提出 FedQS，首个同时优化半异步联邦学习（SAFL）中梯度聚合和模型聚合策略的框架，通过将客户端分为四类并自适应调整训练策略，在准确率、收敛速度和稳定性上全面超越基线。

**[Fedrts Federated Robust Pruning Via Combinatorial Thompson Sampling](optimization/fedrts_federated_robust_pruning_via_combinatorial_thompson_sampling.md)**

:   将联邦动态剪枝重新建模为组合多臂赌博机(CMAB)问题，提出基于 Thompson Sampling 的拓扑调整机制 TSAdj，通过概率性决策替代确定性决策来获得更鲁棒的稀疏模型拓扑，同时显著降低通信开销。

**[Finite-Time Analysis Of Stochastic Nonconvex Nonsmooth Optimization On The Riema](optimization/finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)**

:   提出 Riemannian Online to NonConvex (RO2NC) 算法及其零阶版本 ZO-RO2NC，首次为黎曼流形上完全非光滑非凸随机优化建立了 $O(\delta^{-1}\epsilon^{-3})$ 的有限时间样本复杂度保证，匹配欧几里德最优结果。

**[From Average-Iterate To Last-Iterate Convergence In Games A Reduction And Its Ap](optimization/from_average-iterate_to_last-iterate_convergence_in_games_a_reduction_and_its_ap.md)**

:   提出 A2L (Average to Last-iterate) 黑箱规约，对效用函数关于自身策略和对手联合策略均线性的博弈，能将任意非耦合学习动力学的平均迭代转换为新动力学的末迭代，由此在多人零和多矩阵博弈中取得 $O(\log d / T)$ 梯度反馈和 $\tilde{O}(d^{1/5}T^{-1/5})$ bandit 反馈的 SOTA last-iterate 收敛率。

**[From Information To Generative Exponent Learning Rate Induces Phase Transitions ](optimization/from_information_to_generative_exponent_learning_rate_induces_phase_transitions_.md)**

:   系统刻画了在学习高斯单指标模型时，学习率如何在"information exponent 主导"和"generative exponent 主导"两个样本复杂度体制之间引发相变，并提出了一种新的逐层交替 SGD 算法，无需复用样本即可突破 CSQ 下界。

**[From Linear To Nonlinear Provable Weak-To-Strong Generalization Through Feature ](optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)**

:   本文首次在非线性特征学习设定（线性 CNN → 两层 ReLU CNN）下严格分析了 weak-to-strong 泛化现象，揭示了数据匮乏和数据丰富两种机制下的不同行为：前者通过良性过拟合实现泛化（或因有害过拟合失败），后者通过早停的标签纠正实现泛化（但过训练会退化）。

**[Functional Scaling Laws In Kernel Regression Loss Dynamics And Learning Rate Sch](optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)**

:   在幂律核回归模型中建立了 Functional Scaling Law (FSL)，通过引入"内在时间"概念统一刻画任意学习率调度下的完整 loss 轨迹，并推导出常数/指数衰减/WSD 三种调度在数据受限和计算受限条件下的显式 scaling 关系，理论解释了 WSD 优于纯衰减的经验现象。

**[Generalization Or Hallucination Understanding Out-Of-Context Reasoning In Transf](optimization/generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)**

:   本文论证 LLM 的泛化能力和幻觉产生源于同一机制——脱语境推理（OCR），并在单层注意力模型上理论证明：分解参数化 $(W_O, W_V)$ 因梯度下降的核范数隐式偏差而能执行 OCR，而合并参数化 $W_{OV}$ 因 Frobenius 范数偏差而不能，且 OCR 是样本高效的（仅需 $m_{\text{train}}>0$）。

**[Gradient Descent As Loss Landscape Navigation A Normative Framework For Deriving](optimization/gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)**

:   提出将学习规则视为在（部分可观测的）损失景观中的最优导航策略，通过变分法求解连续时间最优控制问题，在统一框架下推导出梯度下降、动量、自然梯度、Adam及持续学习策略。

**[Implicit Bias Of Spectral Descent And Muon On Multiclass Separable Data](optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)**

:   本文首次完整刻画了归一化最速下降（NSD）和归一化动量最速下降（NMD）在多分类线性可分数据上的隐式偏差：这些算法以 $\mathcal{O}(1/\sqrt{t})$ 的速率收敛到相应 $p$-范数的最大 margin 解，涵盖 Spectral Descent（谱范数）和 Muon 作为特例，并扩展至 Adam（max-范数 margin）。

**[Improving The Straight-Through Estimator With Zeroth-Order Information](optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)**

:   本文提出 FOGZO（First-Order-Guided Zeroth-Order Gradient Descent），将 STE 梯度作为偏置源注入零阶梯度估计中，在保留 STE 的计算效率的同时利用零阶信息纠正 STE 的偶发错误方向，仅多 2 次前向传播即在 DeiT、ResNet、LLaMA 上实现 1-22 点的精度/困惑度改善。

**[In Search Of Adams Secret Sauce](optimization/in_search_of_adams_secret_sauce.md)**

:   本文通过训练 1500+ 语言模型的大规模实验发现：(1) Signum 虽能缩小 96% 的 SGD-Adam 差距，但仍比 Adam 慢 25%；(2) 设 $\beta_1 = \beta_2$ 是 Adam 的近最优简化；(3) 在 $\beta_1 = \beta_2 = \beta$ 下 Adam 可被重新解读为基于在线高斯变分推断估计梯度均值和方差的信噪比自适应 Signum。

**[Isotropic Noise In Stochastic And Quantum Convex Optimization](optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)**

:   本文引入各向同性随机梯度预言机（ISGO）概念——噪声在每个方向上都以高概率有界——并设计随机切平面算法达到 $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ 的查询复杂度，较 SGD 在某些参数区间改进 $d$ 倍，作为推论获得了 sub-exponential 噪声下的新 SOTA 复杂度，并通过量子各向同性化子程序改进了量子随机凸优化的维度依赖。

**[Kernel Learning With Adversarial Features Numerical Efficiency And Adaptive Regu](optimization/kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)**

:   提出在再生核希尔伯特空间（RKHS）中将对抗扰动从输入空间转移到特征空间的新范式，使内层最大化可精确求解，并通过迭代加权核岭回归高效优化，同时自适应正则化无需调参即可匹配交叉验证性能。

**[Large Language Bayes](optimization/large_language_bayes.md)**

:   将 LLM 和概率编程语言（PPL/Stan）数学地"胶合"成联合分布 $p(z,x,m|t) = p(m|t)_{\text{LLM}} \cdot p(z,x|m)_{\text{PPL}}$，用户只需提供非形式化的问题描述和数据，系统自动从 LLM 采样候选形式模型、做贝叶斯推断、通过边际似然加权平均，无需用户编写概率模型。

**[Large Stepsizes Accelerate Gradient Descent For Regularized Logistic Regression](optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)**

:   证明了在线性可分数据上对 $\ell_2$ 正则化逻辑回归使用大步长 GD（进入 Edge of Stability 区间），可将步复杂度从经典的 $\widetilde{O}(\kappa)$ 加速到 $\widetilde{O}(\sqrt{\kappa})$，在小正则化下匹配 Nesterov 动量的加速率。

**[Layer-Wise Update Aggregation With Recycling For Communication-Efficient Federat](optimization/layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)**

:   提出 FedLUAR：基于梯度-权重比的层级优先级度量选择低优先级层复用上一轮梯度（而非丢弃），在仅 17% 通信开销下保持与 FedAvg 几乎相同的精度。

**[Learning At The Speed Of Physics Equilibrium Propagation On Oscillator Ising Mac](optimization/learning_at_the_speed_of_physics_equilibrium_propagation_on_oscillator_ising_mac.md)**

:   首次将 Equilibrium Propagation（EP）完整映射到振荡器 Ising Machine（OIM）硬件上，利用 GHz 物理动力学实现无反向传播的局部学习，在 MNIST/Fashion-MNIST 上达到 97.2%/88.0% 精度，并展示在参数量化和噪声下的鲁棒性。

**[Learning From Interval Targets](optimization/learning_from_interval_targets.md)**

:   研究仅有区间标签（上下界）的回归问题，建立了基于假设类平滑性的非渐进泛化界（不依赖小 ambiguity degree 假设），并提出 minmax 学习框架利用平滑约束限制最坏情况标签，在 18 个真实数据集上显著优于无约束方法。

**[Learning Orthogonal Multi-Index Models A Fine-Grained Information Exponent Analy](optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)**

:   证明正交多索引模型 $f_*(\mathbf{x}) = \sum_{k=1}^P \phi(\mathbf{v}_k^* \cdot \mathbf{x})$ 可通过两阶段在线 SGD 以 $\tilde{O}(dP^{L-1})$ 样本复杂度学习（$L$ 为链接函数最低高阶 Hermite 阶），远优于仅用最低阶信息的 $\tilde{O}(Pd^{L-1})$——关键在于先用 2 阶项恢复子空间，再用 $L$ 阶项恢复方向，联合利用不同阶的 Hermite 分量。

**[Learning Parameterized Skills From Demonstrations](optimization/learning_parameterized_skills_from_demonstrations.md)**

:   提出 DEPS，一种端到端从专家示范中发现参数化技能的算法，通过三层层次策略（离散技能选择→连续参数选择→底层动作）和信息瓶颈设计，学习可解释且可泛化的技能抽象，在LIBERO和MetaWorld上显著优于基线。

**[Learning Provably Improves The Convergence Of Gradient Descent](optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)**

:   首次严格证明了基于unrolling的Learn to Optimize (L2O)框架（Math-L2O）的训练收敛性，利用NTK理论建立了线性收敛速率，并提出确定性初始化策略确保L2O可证明地改善梯度下降算法的收敛性能，实验验证相比标准GD提升超50%的最优性。

**[Learning Quadratic Neural Networks In High Dimensions Sgd Dynamics And Scaling L](optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)**

:   对高维情形下二次激活函数两层神经网络的梯度训练进行了精确分析，针对数据由 $f_*(x) \propto \sum_{j=1}^r \lambda_j \sigma(\langle \theta_j, x \rangle)$ 生成的设定，在宽度 $r \asymp d^\beta$ 和系数幂律衰减 $\lambda_j \asymp j^{-\alpha}$ 下，推导出预测风险关于优化时间、样本量和模型宽度的缩放律。

**[Learning Reconfigurable Representations For Multimodal Federated Learning With M](optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)**

:   提出 PEPSY 框架，通过学习客户端侧的嵌入控制来编码数据缺失模式，将全局聚合表示重新配置为适应各客户端本地上下文的数据完整特征，在多模态联邦学习中处理模态缺失和特征缺失两类问题。

**[Learning Single-Index Models Via Harmonic Decomposition](optimization/learning_single-index_models_via_harmonic_decomposition.md)**

:   提出以球谐函数（spherical harmonics）代替 Hermite 多项式作为单指标模型（SIM）的自然基底，利用旋转对称性刻画任意球对称输入分布下学习 SIM 的样本与计算复杂度，构造了两族最优估计器（张量展开 + 在线 SGD），并揭示了高斯情形之外出现的样本-运行时间权衡现象。

**[Learning Sparse Approximate Inverse Preconditioners For Conjugate Gradient Solve](optimization/learning_sparse_approximate_inverse_preconditioners_for_conjugate_gradient_solve.md)**

:   提出一种基于图神经网络（GNN）的稀疏近似逆（SPAI）预条件子学习方法，利用 SPAI 的局部性与 GNN 消息传递的天然兼容性，并引入尺度不变损失函数（SAI loss），在 GPU 上实现 40%-53% 的求解时间缩减（68%-113% 加速）。

**[Learning Theory For Kernel Bilevel Optimization](optimization/learning_theory_for_kernel_bilevel_optimization.md)**

:   首次为核双层优化（KBO）建立了有限样本泛化界，证明目标函数值和梯度的插入估计误差均以$\mathcal{O}(1/\sqrt{m}+1/\sqrt{n})$的参数速率一致收敛，并将该理论应用于双层梯度下降算法的统计精度分析。

**[Learning To Insert For Constructive Neural Vehicle Routing Solver](optimization/learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)**

:   提出 L2C-Insert，首个基于学习的插入式构造范式用于神经组合优化，通过允许在部分解的任意合法位置插入节点（而非仅追加到末尾），显著提升 TSP/CVRP 的构造质量和灵活性。

**[Least Squares Variational Inference](optimization/least_squares_variational_inference.md)**

:   提出 LSVI（Least Squares Variational Inference），一种无梯度、基于普通最小二乘回归的变分推断方法，在指数族内通过对温控 log-target 做 OLS 回归来迭代求解最优变分近似，对高斯族有高效的 $O(d^3)$（全协方差）或 $O(d)$（平均场）实现。

**[Mar-Fl A Communication Efficient Peer-To-Peer Federated Learning System](optimization/mar-fl_a_communication_efficient_peer-to-peer_federated_learning_system.md)**

:   提出 MAR-FL 系统，通过 Moshpit All-Reduce 机制和动态分组聚合，将 P2P 联邦学习的通信复杂度从 $O(N^2)$ 降至 $O(N \log N)$，同时保持对网络抖动的鲁棒性。

**[Mdns Masked Diffusion Neural Sampler Via Stochastic Optimal Control](optimization/mdns_masked_diffusion_neural_sampler_via_stochastic_optimal_control.md)**

:   提出 Masked Diffusion Neural Sampler (MDNS)，基于连续时间马尔可夫链（CTMC）的随机最优控制理论，通过对齐路径测度来训练离散神经采样器，在状态空间基数高达 $10^{122}$ 的 Ising/Potts 模型上准确采样，大幅超越现有学习型基线。

**[Mecefo Enhancing Llm Training Robustness Via Fault-Tolerant Optimization](optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)**

:   MeCeFO 提出了一种面向 LLM 训练的容错优化算法，当计算节点故障时通过跳连接、选择性激活重计算和低秩梯度近似三个技术将额外开销降到最低，在高频故障下仅有 4.18% 的吞吐量下降。

**[Memory-Augmented Potential Field Theory A Framework For Adaptive Control In Non-](optimization/memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)**

:   提出记忆增强势场理论（MAPFT），在随机最优控制中维护一个动态记忆模块来检测并编码状态空间的拓扑特征（局部最小值、低梯度区等），通过动态修改价值函数景观实现非凸环境下的自适应控制，在 Humanoid-v4 等任务上比最优 RL 方法（SAC）提升 27% 累积奖励，且局部最优逃逸率从 ~30% 提升到 ~72%。

**[Mess Dynamically Learned Inference-Time Llm Routing In Model Zoos With Service L](optimization/mess_dynamically_learned_inference-time_llm_routing_in_model_zoos_with_service_l.md)**

:   MESS+是首个将LLM请求路由形式化为带SLA约束的随机优化问题的框架，通过在线学习的请求满足度预测器+虚拟队列机制动态选择模型，在3个推理和5个问答基准上以满足SLA约束的前提下实现平均2倍的成本节省，并提供成本最优性和约束满足的理论保证。

**[Mobo-Osd Batch Multi-Objective Bayesian Optimization Via Orthogonal Search Direc](optimization/mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)**

:   提出MOBO-OSD算法，通过在逼近的个体极小值凸包（CHIM）上定义正交搜索方向来生成多样化的Pareto最优解，结合Pareto前沿估计和批量选择策略，在合成与真实基准上持续超越SOTA多目标贝叶斯优化方法。

**[Multi-Head Transformers Provably Learn Symbolic Multi-Step Reasoning Via Gradien](optimization/multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)**

:   从梯度下降训练动力学出发，严格证明了单层多头 Transformer 通过 CoT 过程可学会树路径查找的前向和后向推理任务，并揭示不同注意力头会自主专业化以协调解决多阶段子任务。

**[Multiplayer Federated Learning Reaching Equilibrium With Less Communication](optimization/multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)**

:   提出多人联邦学习（MpFL）框架，将FL中的客户端建模为博弈论中的理性玩家，并设计PEARL-SGD算法通过局部更新减少通信开销，同时收敛到Nash均衡。

**[Natural Gradient Descent For Improving Variational Inference Based Classificatio](optimization/natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)**

:   研究使用自然梯度下降优化器 iVON 替代标准 SGD 来优化变分推断中的 BNN 参数，在射电星系分类中获得更好的不确定性校准，同时保持与 HMC 和 BBB-VI 相当的预测性能。

**[Natural Gradient Vi Guarantees For Non-Conjugate Models](optimization/natural_gradient_vi_guarantees_for_non-conjugate_models.md)**

:   在 mean-field 参数化下，为非共轭模型的自然梯度变分推断（NGVI）建立了三个关键理论结果：变分损失的相对光滑性条件、带非欧投影的修正 NGVI 的全局收敛到驻点保证、以及在额外结构假设下的隐藏凸性和快速全局收敛保证。

**[Near-Exponential Savings For Mean Estimation With Active Learning](optimization/near-exponential_savings_for_mean_estimation_with_active_learning.md)**

:   提出 PartiBandits 算法，结合基于分歧的主动学习与 UCB 风格的分层抽样，在辅助信息 $X$ 对目标变量 $Y$ 有预测力时，实现了均值估计的近指数级标签节省。

**[Neural Thermodynamics Entropic Forces In Deep And Universal Representation Learn](optimization/neural_thermodynamics_entropic_forces_in_deep_and_universal_representation_learn.md)**

:   建立一套"神经热力学"理论，证明 SGD 训练中由随机性和离散时间更新产生的涌现**熵力**会系统性地打破神经网络参数的连续对称性并保留离散对称性，导致类似热力学能量均分的**梯度平衡**现象，从而 (a) 首次理论证明 Platonic 表征假说（不同模型学到相似表征），(b) 调和深度学习优化中"趋向尖锐"与"趋向平坦"的矛盾观察。

**[Neuro-Symbolic Entity Alignment Via Variational Inference](optimization/neuro-symbolic_entity_alignment_via_variational_inference.md)**

:   提出 NeuSymEA，一个基于变分 EM 算法的神经符号推理框架，将符号规则推理与神经网络嵌入统一在马尔可夫随机场中进行实体对齐，在 DBP15K 上实现了显著的性能提升和低资源鲁棒性。

**[Non-Stationary Bandit Convex Optimization A Comprehensive Study](optimization/non-stationary_bandit_convex_optimization_a_comprehensive_study.md)**

:   系统研究了非平稳环境下的Bandit凸优化问题，提出两个算法（TEWA-SE和cExO），统一建立了三种非平稳度量（切换数S、总变差Δ、路径长度P）下的遗憾上下界，多个设定下达到极小极大最优。

**[Nonlinearly Preconditioned Gradient Methods Momentum And Stochastic Analysis](optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)**

:   在各向异性下降不等式框架下，为非线性预条件梯度方法引入重球法动量，并分析其随机变体在多种噪声假设下的收敛性质，统一了梯度裁剪与归一化梯度方法的理论分析。

**[On Minimax Estimation Of Parameters In Softmax-Contaminated Mixture Of Experts](optimization/on_minimax_estimation_of_parameters_in_softmax-contaminated_mixture_of_experts.md)**

:   首次对带 softmax 门控的受污染混合专家（contaminated MoE）模型进行极小极大参数估计分析，提出"可区分性"概念刻画预训练模型与 prompt 的关系，证明可区分时 MLE 达到参数级 $\tilde{O}(n^{-1/2})$ 最优速率，不可区分时速率显著变慢。

**[Online Two-Stage Submodular Maximization](optimization/online_two-stage_submodular_maximization.md)**

:   首次提出在线两阶段子模最大化（O2SSM）问题，针对加权阈值势函数（WTP）设计了 RAOCO 算法，通过分数松弛+随机管道舍入实现多项式时间运行下的次线性 $(1-1/e)^2$-regret 保证，同时改进了离线问题的近似比。

**[Optimal Rates For Generalization Of Gradient Descent For Deep Relu Classificatio](optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)**

:   证明了深度ReLU网络上梯度下降的泛化速率达到 $\widetilde{O}(L^4(1+\gamma L^2)/(n\gamma^2))$，首次在深度ReLU网络上同时实现：(1) 对样本量 $n$ 的最优 $1/n$ 依赖，(2) 对深度 $L$ 仅多项式依赖。

**[Optimality And Np-Hardness Of Transformers In Learning Markovian Dynamical Funct](optimization/optimality_and_np-hardness_of_transformers_in_learning_markovian_dynamical_funct.md)**

:   从优化理论角度分析 Transformer 学习马尔可夫动态函数的 ICL 能力：推导单层线性自注意力的全局最优解（闭式表达），证明从扩展参数空间恢复 Transformer 参数是 NP-hard 的，并揭示多层 LSA 等价于预条件多目标优化。

**[Optimistic Online-To-Batch Conversions For Accelerated Convergence And Universal](optimization/optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)**

:   提出乐观在线到批量（O2B）转换框架，将乐观性从在线算法中释放到转换机制本身，使简单的在线梯度下降就能实现 $O(T^{-2})$ 加速收敛率，并首次通过 O2B 转换实现强凸光滑目标的最优收敛，同时达到对光滑性的通用性。

**[Oracle-Efficient Combinatorial Semi-Bandits](optimization/oracle-efficient_combinatorial_semi-bandits.md)**

:   提出两种oracle高效框架（自适应和调度式），将组合半老虎机问题中的oracle调用次数从线性 $\Theta(T)$ 降低到双对数 $O(\log\log T)$，同时保持近最优的遗憾界。

**[Orthograd Improves Neural Calibration](optimization/orthograd_improves_neural_calibration.md)**

:   本文首次系统研究了OrthoGrad（⊥Grad）——一种逐层将梯度投影到与权重正交方向上的几何约束优化方法——在神经网络校准任务中的效果。实验表明在CIFAR-10低数据场景下，OrthoGrad在不降低准确率的情况下显著改善校准指标（熵、损失、置信度），并证明了简化版本在标准假设下的收敛性。

**[Personalized Subgraph Federated Learning With Differentiable Auxiliary Projectio](optimization/personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)**

:   提出FedAux框架，通过可微分的辅助投影向量（APV）将节点嵌入映射到一维空间并用高斯核进行软排序聚合，APV既作为局部子图的紧凑隐私保护摘要用于服务器端相似度计算，又参与客户端的联合优化，实现了个性化的子图联邦学习。

**[Perturbation Bounds For Low-Rank Inverse Approximations Under Noise](optimization/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)**

:   首次给出在加性噪声下低秩逆近似 $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ 的非渐近谱范数扰动界，利用轮廓积分技术得到依赖特征间隙、谱衰减和噪声对齐的锐界，比经典全逆界改进高达 $\sqrt{n}$ 倍。

**[Preference Learning With Response Time Robust Losses And Guarantees](optimization/preference_learning_with_response_time_robust_losses_and_guarantees.md)**

:   将用户决策的响应时间信息融入偏好学习框架中，通过 Neyman 正交损失函数将奖励模型学习的误差从指数级缩减到多项式级。

**[Probing Neural Combinatorial Optimization Models](optimization/probing_neural_combinatorial_optimization_models.md)**

:   首次系统性地将探针(probing)方法引入神经组合优化(NCO)模型的研究，提出CS-Probing工具来分析模型表示中编码的决策知识、归纳偏置和泛化机制，并发现关键嵌入维度可用于提升模型泛化性能。

**[Problem-Parameter-Free Decentralized Bilevel Optimization](optimization/problem-parameter-free_decentralized_bilevel_optimization.md)**

:   提出 AdaSDBO，一种完全无需问题参数先验知识的去中心化双层优化算法，通过基于累积梯度范数的自适应步长实现 $\tilde{O}(1/\sqrt{T})$ 收敛率。

**[Profit A Specialized Optimizer For Deep Fine Tuning](optimization/profit_a_specialized_optimizer_for_deep_fine_tuning.md)**

:   PROFIT 将微调视为时间维度上的多任务学习问题，通过将新任务梯度对"回归平衡点"方向做正交化投影，实现了无需额外数据或参数的抗遗忘微调优化器。

**[Projecting Assumptions The Duality Between Sparse Autoencoders And Concept Geome](optimization/projecting_assumptions_the_duality_between_sparse_autoencoders_and_concept_geome.md)**

:   本文揭示了稀疏自编码器(SAE)架构与其能发现的概念结构之间存在根本性的对偶性——每种SAE隐式假设了特定的概念组织方式，当假设不匹配时会系统性地遗漏概念。据此提出了SpaDE，一种考虑非线性可分性和维度异质性的新SAE。

**[Purifying Shampoo Investigating Shampoos Heuristics By Decomposing Its Precondit](optimization/purifying_shampoo_investigating_shampoos_heuristics_by_decomposing_its_precondit.md)**

:   通过将Shampoo预条件矩阵分解为特征值和特征基两部分，揭示了学习率嫁接(grafting)实质上是弥补特征值的陈旧性和缩放偏差，并提出了特征值校正和自适应特征基更新频率来替代这些启发式技巧。

**[Quantitative Convergence Of Trained Single Layer Neural Networks To Gaussian Pro](optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)**

:   为梯度下降训练的浅层神经网络提供了在任意正训练时间 $t \geq 0$ 下向高斯过程收敛的显式定量上界，证明了二次Wasserstein距离以 $O(\log n_1 / n_1)$ 的速率多项式衰减。

**[Rao-Blackwellised Reparameterisation Gradients](optimization/rao-blackwellised_reparameterisation_gradients.md)**

:   提出 R2-G2 估计器作为重参数化梯度的 Rao-Blackwell 化版本，证明 Bayesian MLP 的局部重参数化是其特例，并将低方差梯度的优势推广到一系列概率模型。

**[Rethinking Neural Combinatorial Optimization For Vehicle Routing Problems With D](optimization/rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)**

:   揭示了现有NCO方法严重过拟合固定约束紧度（如CVRP的固定车辆容量C=50），提出变约束紧度训练方案和多专家模块(MEM)，使模型能有效处理从极紧到极松的全范围约束。

**[Revisiting Orbital Minimization Method For Neural Operator Decomposition](optimization/revisiting_orbital_minimization_method_for_neural_operator_decomposition.md)**

:   重新审视源自计算化学的经典轨道最小化方法（OMM），提供了简洁的线性代数一致性证明，揭示其与Sanger规则、流式PCA等的深层联系，并将其推广为训练神经网络进行正半定算子谱分解的通用框架。

**[Robust Estimation Under Heterogeneous Corruption Rates](optimization/robust_estimation_under_heterogeneous_corruption_rates.md)**

:   本文研究了异质污染率下的鲁棒估计问题——每个样本以不同的已知概率被污染——对有界分布和高斯分布的均值估计及线性回归建立了紧的极小极大率，发现最优估计器可以简单地丢弃污染率超过某阈值的样本。

**[Second-Order Optimization Under Heavy-Tailed Noise Hessian Clipping And Sample C](optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)**

:   首次系统研究重尾噪声条件下二阶随机优化的理论基础，建立了紧的样本复杂度下界，提出了基于梯度和 Hessian 裁剪的归一化SGD算法（Clip NSGDHess），并证明其近似达到信息论极限。

**[Set Smoothness Unlocks Clarke Hyper-Stationarity In Bilevel Optimization](optimization/set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)**

:   本文提出"集合光滑性"(set smoothness)这一新的结构性质，证明它在非凸-PŁ双层优化中自然成立，并据此揭示超目标函数隐藏的弱凸/弱凹结构，首次建立了非光滑超目标函数Clarke稳定点的可计算性保证。

**[Sharper Convergence Rates For Nonconvex Optimisation Via Reduction Mappings](optimization/sharper_convergence_rates_for_nonconvex_optimisation_via_reduction_mappings.md)**

:   提出 Reduction Mapping 框架，利用最优解集的流形结构（由过参数化或对称性产生）重参数化优化问题，证明这能改善曲率性质并理论上加速基于梯度方法的收敛。

**[Small Batch Size Training For Language Models When Vanilla Sgd Works And Why Gra](optimization/small_batch_size_training_for_language_models_when_vanilla_sgd_works_and_why_gra.md)**

:   本文系统研究了小批量（甚至batch size=1）在语言模型预训练和微调中的表现，提出了基于"token半衰期"固定的Adam β₂缩放规则，发现小批量不仅训练稳定，还使vanilla SGD具备与自适应优化器相当的竞争力，并建议避免使用梯度累积。

**[Streaming Federated Learning With Markovian Data](optimization/streaming_federated_learning_with_markovian_data.md)**

:   首次严格分析了非凸目标函数下具有马尔可夫数据流的流式联邦学习，证明 Minibatch SGD、Local SGD 和 Local SGD-M 均能实现与客户端数成反比的样本复杂度（线性加速），且 Local SGD-M 无需异质性假设即可匹配 Minibatch SGD 的通信复杂度。

**[The Implicit Bias Of Structured State Space Models Can Be Poisoned With Clean La](optimization/the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)**

:   本文首次从理论上证明结构化状态空间模型 (SSM) 的隐式偏置可以被干净标签 (clean-label) 训练样本"投毒"——存在特殊的训练样本，尽管它们的标签由教师模型正确标注，但其加入会彻底扭曲 SSM 的隐式偏置，导致泛化彻底失败。

**[The Rich And The Simple On The Implicit Bias Of Adam And Sgd](optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)**

:   本文理论和实验证明，SGD训练的神经网络倾向于学习简单线性特征（简单性偏置），而Adam训练则产生更丰富的非线性特征，使模型更接近贝叶斯最优预测器，在分布偏移下泛化更好。

**[Training-Free Bayesianization For Low-Rank Adapters Of Large Language Models](optimization/training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)**

:   提出 TFB（Training-Free Bayesianization），通过在低秩各向同性高斯分布族中搜索最大可接受方差，将已训练好的 LoRA 适配器无需重训练即转化为贝叶斯版本，理论上等价于广义变分推断。

**[Training Robust Graph Neural Networks By Modeling Noise Dependencies](optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)**

:   提出依赖感知图噪声(DANG)和DA-GNN框架，通过建模节点特征噪声→图结构噪声→标签噪声的因果依赖链，利用变分推断推导ELBO来训练对多源协同噪声鲁棒的GNN。

**[Understanding Adam Requires Better Rotation Dependent Assumptions](optimization/understanding_adam_requires_better_rotation_dependent_assumptions.md)**

:   本文通过系统的实验研究揭示了 Adam 优化器对参数空间坐标基底的强依赖性，证明现有旋转不变的理论假设不足以解释 Adam 的优越性，并发现层更新的正交性是预测 Adam 在不同基底下性能的有力指标。

**[Understanding The Generalization Of Stochastic Gradient Adam In Learning Neural ](optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)**

:   首次理论分析 mini-batch Adam 的泛化行为，证明大 batch Adam/AdamW 即使带 weight decay 也收敛到高测试误差的解，而小 batch 版本通过随机梯度的隐式正则化 + weight decay 的显式正则化可实现近零测试误差，且 Adam 的有效 weight decay 上界严格小于 AdamW。

**[Unveiling M-Sharpness Through The Structure Of Stochastic Gradient Noise](optimization/unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)**

:   本文通过扩展的随机微分方程(SDE)框架揭示了SAM中m-sharpness现象的理论机制——更小的微批次尺寸m带来更强的随机梯度噪声(SGN)协方差隐式正则化，并据此提出了可并行化的Reweighted SAM (RW-SAM)方法。

**[Unveiling The Power Of Multiple Gossip Steps A Stability-Based Generalization An](optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)**

:   本文首次从算法稳定性角度分析去中心化 SGD（DSGD）中多步 Gossip 通信（MGS）的泛化效果，证明 MGS 以指数速率减少优化误差从而收紧泛化界，但即使 Gossip 步数趋于无穷也无法完全弥合与中心化训练的泛化差距。

**[Vera Variational Inference Framework For Jailbreaking Large Language Models](optimization/vera_variational_inference_framework_for_jailbreaking_large_language_models.md)**

:   将黑盒 LLM 越狱攻击形式化为变分推断问题，训练小型攻击者 LLM 近似目标 LLM 的对抗提示后验分布，一次训练后可高效、多样地生成越狱提示，无需依赖人工模板。

**[Verbalized Algorithms](optimization/verbalized_algorithms.md)**

:   本文提出"语言化算法"（Verbalized Algorithms, VAs）框架，将经典算法的控制流保持不变，仅用LLM替换其中的原子操作（如二值比较），从而在自然语言推理任务中继承经典算法的正确性和复杂度保证，在排序、求最大值、聚类和子模最大化四个案例中验证了有效性。

**[Viking Deep Variational Inference With Stochastic Projections](optimization/viking_deep_variational_inference_with_stochastic_projections.md)**

:   VIKING 提出了一种基于 Fisher-Rao 度量核空间与像空间分解的变分近似后验族，通过随机交替投影算法实现可扩展的全相关贝叶斯训练，在多个基准上超越了现有贝叶斯深度学习方法。

**[Wasserstein Transfer Learning](optimization/wasserstein_transfer_learning.md)**

:   提出了首个针对Wasserstein空间中概率分布输出的迁移学习框架（WaTL），通过加权辅助估计、偏差校正和投影三步法，结合自适应信息源选择，从源域迁移知识以提升目标域分布回归的估计性能。

**[Weak-To-Strong Generalization Under Distribution Shifts](optimization/weak-to-strong_generalization_under_distribution_shifts.md)**

:   发现朴素的弱到强泛化 (weak-to-strong generalization) 在分布偏移下会失败（强模型表现甚至不如弱监督者），提出 RAVEN 框架通过动态学习弱模型的最优组合权重来实现鲁棒的弱到强泛化，在 OOD 任务上超越基线 30%+。

---

## 🔬 可解释性 { #interpretability }

**[A Is For Absorption Studying Feature Splitting And Absorption In Sparse Autoenco](interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)**

:   发现并系统研究了 SAE 中的"特征吸收"现象：看似单义的 SAE latent 会在特定 token 上不激活，其特征方向被更具体的子 latent "吸收"，这是层级特征+稀疏性损失的必然结果，对 SAE 用于可靠解释 LLM 构成根本挑战。

**[A Unified Reasoning Framework For Holistic Zeroshot Video An](interpretability/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)**

:   提出一个完全零样本、无需训练的视频异常分析框架，通过Intra-Task Reasoning（置信度门控的自我精化）和Inter-Task Chaining（从时序检测到空间定位到语义理解的级联prompt传递），在4个benchmark上全面超越先前零样本方法4-6% AUC。

**[Adaptgrad Adaptive Sampling To Reduce Noise](interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)**

:   AdaptGrad通过分析SmoothGrad噪声的理论来源——超范围采样行为，提出自适应调整每个输入维度的高斯采样方差以控制额外噪声上限的方法，在几乎消除梯度噪声的同时揭示更丰富的细节特征，方法极简且可与任意梯度解释方法组合。

**[Additive Models Explained A Computational Complexity Approach](interpretability/additive_models_explained_a_computational_complexity_approach.md)**

:   对广义可加模型（GAM）的多种解释类型进行系统的计算复杂度分析，覆盖 54 种"组件模型 × 输入域 × 解释方法"组合，揭示 GAM 的解释复杂度高度依赖输入域类型——这是决策树、神经网络等其他 ML 模型从未展现的独特现象，挑战了"可加即可解释"的直觉假设。

**[Agentiql An Agent-Inspired Multi-Expert Framework For Text-To-Sql Generation](interpretability/agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)**

:   提出 AgentiQL，一个多专家 agent 框架用于 Text-to-SQL：reasoning agent 分解问题为子问题，coding agent 生成子查询，refinement 步骤校正列选择，adaptive router 在基线解析器和模块化 pipeline 之间智能路由，使用 14B 开源模型达到 86.07% EX（Spider），接近 GPT-4 SOTA(89.65%)。

**[An Analysis Of Concept Bottleneck Models Measuring Understanding And Mitigating ](interpretability/an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)**

:   本文首次系统研究了标注噪声对概念瓶颈模型(CBM)的影响，发现约23%的"易感概念"驱动了大部分性能退化，并提出训练阶段使用SAM + 推理阶段基于不确定性干预的两阶段缓解策略来恢复模型鲁棒性。

**[Are Greedy Task Orderings Better Than Random In Continual Linear Regression](interpretability/are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)**

:   本文系统分析了持续线性回归中贪心任务排序（最大化连续任务间不相似度）与随机排序的收敛性差异，揭示了贪心排序在高秩设定下可媲美随机排序，但在一般秩设定下单遍贪心可能灾难性失败，而允许重复的贪心排序收敛速率为 $\mathcal{O}(1/\sqrt[3]{k})$。

**[Arecho Autoregressive Evaluation Via Chain-Based Hypothesis Optimization For Spe](interpretability/arecho_autoregressive_evaluation_via_chain-based_hypothesis_optimization_for_spe.md)**

:   ARECHO 将语音多指标评估建模为链式自回归 token 预测任务——设计统一的语音信息 token 化管线处理 87 个异质指标（数值/类别/有界/无界），通过动态分类链显式捕捉指标间依赖关系（如可懂度-自然度相关性），配合两步置信度导向解码减少误差传播，在增强/生成/噪声三类语音评估中全面超越 UniVERSA 基线（Avg Test MSE 23.26 vs 96.99，-76%）。

**[Auditing Meta-Cognitive Hallucinations In Reasoning Large Language Models](interpretability/auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)**

:   系统性审计推理大模型（RLLM）中幻觉的产生与传播机制，发现长 CoT 中的反思（reflection）会通过元认知偏差放大幻觉而非纠正它，即使在幻觉源头进行干预也难以改变最终结果（chain disloyalty），揭示现有幻觉检测方法在多步推理场景下严重不足。

**[Base Models Know How To Reason Thinking Models Learn When](interpretability/base_models_know_how_to_reason_thinking_models_learn_when.md)**

:   通过无监督 SAE 聚类发现 thinking model 的推理机制分类，然后用 steering vector 在基座模型上激活这些潜在推理能力，混合模型恢复高达 91% 的 thinking-base 性能差距（无需权重更新），证明基座模型已具备推理能力，thinking model 只是学会了"何时"部署它们。

**[Better Estimation Of The Kullback--Leibler Divergence Between Language Models](interpretability/better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)**

:   提出 KL 散度的 Rao-Blackwell 化 Monte Carlo 估计器——在每个位置对下一个 token 的分布求精确 KL（而非只用采样的 token），理论证明无偏且方差严格不超过标准 MC 估计器，零额外计算开销，在 RLHF 情感控制任务中使训练更稳定、模型更频繁出现在 Pareto 前沿（78%）。

**[Beyond Accuracy Dissecting Mathematical Reasoning For Llms U](interpretability/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)**

:   提出 SPARKLE 三轴分析框架（计划执行、知识整合、子问题分解）细粒度剖析 RL 如何改变 LLM 推理行为，发现 RL 主要增强了知识整合能力和计划灵活性而非计划执行能力，并提出 SparkleRL-PSS 多阶段 RL 训练 pipeline 通过 partial step scaffolding 有效利用难题数据。

**[Beyond Components Singular Vector-Based Interpretability Of Transformer Circuits](interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)**

:   提出基于SVD奇异向量的方向级可解释性框架，通过对注意力头和MLP的增广矩阵统一SVD分解+可学习对角掩码（KL+L₁），发现单组件内存在正交低秩子函数叠加——IOI任务仅需~9%方向即可KLD=0.21复现模型行为。

**[Beyond Token Probes Hallucination Detection Via Activation Tensors With Act-Vit](interpretability/beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)**

:   将LLM的全部隐层激活组织为"激活张量"（层×token×隐维度），类比图像用ViT处理，设计ACT-ViT架构支持跨LLM联合训练，在15个LLM-数据集组合上一致超越传统probing方法，并展现出对未见数据集和未见LLM的强零样本/少样本迁移能力。

**[Bigram Subnetworks Mapping To Next Tokens In Transformer Language Models](interpretability/bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)**

:   通过连续稀疏化在Transformer语言模型中找到仅包含~10M参数的bigram子网络，它们集中在第一个MLP层，足以复现bigram预测（$r>0.95$），且被消融后模型性能大幅下降，证明这些子网络是语言模型中既必要又充分的最小next-token预测电路。

**[Born A Transformer -- Always A Transformer On The Effect Of Pretraining On Archi](interpretability/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)**

:   通过系统性地研究检索和复制任务家族，揭示了大规模预训练会为Transformer引入方向性偏置（右/前向优于左/后向），但无法克服非唯一任务上的根本架构限制；微调可消除方向偏置但不能突破架构表达力边界。

**[Causal Head Gating A Framework For Interpreting Roles Of Attention Heads In Tran](interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)**

:   提出 Causal Head Gating (CHG)，通过对 Transformer 的每个 attention head 学习一个可微门控系数并结合正/负正则化，将 head 分为促进（facilitating）、干扰（interfering）、无关（irrelevant）三类，无需人工标签或 prompt 模板即可发现因果子电路，并扩展为对比 CHG 以分离 ICL 和指令遵循的独立电路。

**[Cbmas Cognitive Behavioral Modeling Via Activation Steering](interpretability/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)**

:   CBMAS提出一个将激活引导作为连续诊断工具的框架，通过密集α扫描和注入-读取层解耦，将认知偏差分析从"有偏差/无偏差"的二元判断升级为可追踪翻转点、传播路径和衰减模式的连续轨迹分析，在GPT-2 Small上揭示了安抚行为在浅层强烈编码但向深层快速衰减的规律。

**[Chiqpm Calibrated Hierarchical Interpretable Image Classification](interpretability/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)**

:   CHiQPM 提出一种校准的层次化可解释图像分类方法，通过二次规划选择和分配特征给类别，构建层次化解释路径，并内置可解释的 Conformal Prediction 集合预测，在保持黑盒模型 99% 准确率的同时提供全局和局部可解释性。

**[Cognitive Mirrors Exploring The Diverse Functional Roles Of Attention Heads In L](interpretability/cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)**

:   提出CogQA基准数据集和多类probing框架，系统分析LLM中注意力头的认知功能特化现象，发现认知头具有稀疏性、普遍性和层级化功能组织特征，去除认知头显著降低推理性能，增强则提升准确率。

**[Conceptscope Characterizing Dataset Bias Via Disentangled Visual Concepts](interpretability/conceptscope_characterizing_dataset_bias_via_disentangled_visual_concepts.md)**

:   提出 ConceptScope 框架，利用在视觉基础模型表征上训练的稀疏自编码器（SAE）自动发现和量化数据集中的视觉概念偏差，无需人工标注即可将概念分类为 target / context / bias 三类。

**[Conditional Distribution Compression Via The Kernel Conditional Mean Embedding](interpretability/conditional_distribution_compression_via_the_kernel_conditional_mean_embedding.md)**

:   首次提出针对**条件分布**（而非联合分布）的压缩算法，利用核条件均值嵌入（KCME）定义新度量 AMCMD，并设计线性时间算法 ACKIP 构建保留条件分布统计特性的压缩数据集。

**[Curvature Tuning Provable Training-Free Model Steering From A Single Parameter](interpretability/curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)**

:   提出 Curvature Tuning（CT），通过在激活函数中注入单个超参数 $\beta$ 来可证明地调节模型决策边界的曲率，无需修改权重即可提升泛化和鲁棒性，同时作为微调方法参数量远少于 LoRA rank 1。

**[Dataset Distillation For Pre-Trained Self-Supervised Vision Models](interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)**

:   提出 Linear Gradient Matching 方法，为预训练自监督视觉模型蒸馏合成数据集：每类仅需一张合成图就能训练出接近全数据集表现的线性分类器，且蒸馏图像可跨模型架构迁移。

**[Deep Modularity Networks With Diversity-Preserving Regularization](interpretability/deep_modularity_networks_with_diversity-preserving_regularization.md)**

:   在 Deep Modularity Networks (DMoN) 基础上引入三项多样性保持正则化（距离、方差、熵），显式促进特征空间中的簇间分离和分配多样性，在特征丰富的图数据集上显著提升聚类质量。

**[Deep Value Benchmark Measuring Whether Models Generalize Deep Values Or Shallow ](interpretability/deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)**

:   提出 Deep Value Benchmark (DVB)，通过"先混淆后解混淆"的实验设计，测量 LLM 是学习了深层人类价值观还是仅记住了表层偏好模式，发现所有模型的深层价值泛化率 (DVGR) 仅为 0.30，远低于随机水平。

**[Distributional Autoencoders Know The Score](interpretability/distributional_autoencoders_know_the_score.md)**

:   本文为 Distributional Principal Autoencoder (DPA) 提供了精确的理论保证：证明了最优编码器的等值面几何与数据分布的 score 函数之间的闭合形式关系，并证明了超出流形维度的潜在分量与数据条件独立，从而统一了分布学习与内在维度发现两个长期目标。

**[Do Different Prompting Methods Yield A Common Task Representation In Language Mo](interpretability/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)**

:   通过将函数向量（Function Vectors）方法从 few-shot 示例推广到文本指令，发现不同提示方式（demonstrations vs. instructions）并不会在 LLM 中诱导出统一的任务表征，而是激活部分重叠但主要不同的注意力头机制。

**[Dynamic Algorithm For Explainable K-Medians Clustering Under Lp Norm](interpretability/dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)**

:   本文提出首个适用于一般 $\ell_p$ 范数的可解释 k-medians 聚类算法，实现 $\tilde{O}(p(\log k)^{1+1/p-1/p^2})$ 近似比（改进了 p=2 的已知最优界），并给出首个动态版本：在中心集合的插入/删除下，以 $O(d \log^3 k)$ 摊还更新时间和 $O(\log k)$ 重分配次数维护可解释聚类。

**[Efficient Vision-Language Reasoning Via Adaptive Token Pruning](interpretability/efficient_vision-language_reasoning_via_adaptive_token_pruning.md)**

:   提出 Adaptive Token Pruning (ATP)，一种免训练的即插即用模块，通过融合 ViT CLS 注意力（模态内显著性）和 CLIP 文本-图像相似度（模态间相关性）来筛选最有信息量的视觉 token，在 VQA/GQA/COCO Captioning 上以约 40% FLOPs 降低和 1.5 倍加速换取不到 1% 的精度损失。

**[Emergence Of Linear Truth Encodings In Language Models](interpretability/emergence_of_linear_truth_encodings_in_language_models.md)**

:   提出 **Truth Co-occurrence Hypothesis (TCH)**——真实陈述倾向于与其他真实陈述共现——并通过一个最简单的单层 Transformer 玩具模型，端到端地展示了线性真值子空间如何通过两阶段训练动态（先记忆 → 后编码真值）自然涌现，为理解 LLM 中广泛报告的线性真值表示提供了首个机制性解释。

**[Empowering Decision Trees Via Shape Function Branching](interpretability/empowering_decision_trees_via_shape_function_branching.md)**

:   提出 Shape Generalized Tree (SGT)，在决策树每个内部节点使用可学习的轴对齐形状函数替代传统线性阈值分裂，以更紧凑的树结构捕捉非线性特征效应，同时保持可解释性。

**[Encoding And Understanding Astrophysical Information In Large Language Model-Gen](interpretability/encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)**

:   探究LLM嵌入是否能编码从X射线天文观测导出的物理量（硬度比、幂律指数、变异性），发现结构化prompt设计可将物理属性聚类纯度提升5.9%-57.5%，稀疏自编码器揭示LLM通过识别天体类型来推断未显式给出的物理参数。

**[Evaluating Llms In Open-Source Games](interpretability/evaluating_llms_in_open-source_games.md)**

:   通过开源游戏（智能体提交程序而非原始行动）这一新范式，系统评估 LLM 在战略推理、互相学习和合作博弈中的能力，发现 LLM 可自动发现近似程序平衡。

**[Explaining Similarity In Vision-Language Encoders With Weighted Banzhaf Interact](interpretability/explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)**

:   FIxLIP 提出基于加权 Banzhaf 交互指数的博弈论框架，统一分解视觉-语言编码器（如 CLIP、SigLIP-2）的相似度预测为一阶token归因和二阶跨模态/模态内交互，在效率和忠实度上均超越现有一阶归因方法。

**[Fact Faithful Concept Traces For Explaining Neural Network Decisions](interpretability/fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)**

:   提出 FaCT，一种结合 B-cos 变换和稀疏自编码器 (SAE) 的内在可解释模型，能够**忠实地**将模型预测分解为概念贡献（Logit = $\sum$ 概念贡献），并将每个概念忠实地可视化到输入像素级别（概念激活 = $\sum$ 像素贡献），同时提出基于 DINOv2 的 C²-score 用于评估概念一致性。

**[Fantastic Features And Where To Find Them A Probing Method To Combine Features F](interpretability/fantastic_features_and_where_to_find_them_a_probing_method_to_combine_features_f.md)**

:   提出 ComBo，一种基于 probing 的轻量级 adapter，通过仿射投影压缩多个冻结基础模型多层激活，再用小型 transformer 融合，无需反向传播即可高效整合多模型互补表征，在 VTAB-1k 上超越先前 probing 方法并匹配蒸馏方法。

**[Fastdinov2 Frequency Based Curriculum Learning Improves Robustness And Training ](interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)**

:   提出 FastDINOv2，一种两阶段频率课程学习策略：先用低分辨率图像训练 75% epochs 学习低频特征以加速收敛，再用全分辨率+高斯噪声 patching 训练 25% epochs 平衡频率偏置，实现 1.6× 加速、2.25× FLOPs 节省，同时增强鲁棒性。

**[From Flat To Hierarchical Extracting Sparse Representations With Matching Pursui](interpretability/from_flat_to_hierarchical_extracting_sparse_representations_with_matching_pursui.md)**

:   提出 MP-SAE，将经典 Matching Pursuit 算法展开为 SAE 的序列化编码器，通过残差引导的贪心特征选择实现条件正交性，能捕捉标准 SAE 无法发现的层次结构、非线性可及和跨模态特征，并天然支持推理时自适应稀疏度调节。

**[Geometric Priors For Generalizable World Models Via Vector Symbolic Architecture](interpretability/geometric_priors_for_generalizable_world_models_via_vector_symbolic_architecture.md)**

:   提出将 Vector Symbolic Architecture (VSA) 中的 Fourier Holographic Reduced Representation (FHRR) 作为几何先验引入世界模型，通过 element-wise 复数乘法建模状态转移，在离散 GridWorld 上实现 87.5% 的 zero-shot 泛化准确率和 4 倍于 MLP 的噪声鲁棒性。

**[H-Splid Hsic-Based Saliency Preserving Latent Information Decomposition](interpretability/h-splid_hsic-based_saliency_preserving_latent_information_decomposition.md)**

:   提出 H-SPLID，通过将隐空间显式分解为**显著（任务相关）**和**非显著（任务无关）**两个子空间，结合 HSIC 正则化实现信息压缩，证明预测偏差上界受显著子空间维度和 HSIC 控制，在无对抗训练条件下显著提升对非显著区域扰动的鲁棒性。

**[How Do Transformers Learn Implicit Reasoning](interpretability/how_do_transformers_learn_implicit_reasoning.md)**

:   在精细控制的符号环境中从零训练 Transformer，发现多跳隐式推理经历"记忆→分布内泛化→跨分布泛化"三个阶段，核心机制不是中间实体的可解码性，而是其在余弦空间中的聚类一致性——同一中间实体的表示在不同查询中形成紧密聚类时，推理能力才涌现。

**[Improving Perturbation-Based Explanations By Understanding The Role Of Uncertain](interpretability/improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)**

:   揭示了不确定性校准（模型置信度与实际准确率的对齐）与扰动式可解释性方法质量之间的根本联系，证明模型在扰动输入下的误校准直接损害全局和局部解释质量，并提出 ReCalX 通过扰动级别自适应温度缩放显著改善解释的鲁棒性和保真度。

**[Knowing When To Stop Efficient Context Processing Via Latent Sufficiency Signals](interpretability/knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)**

:   本文提出 dynamic context cutoff，通过探测 Transformer 特定注意力头中编码的"信息充分性信号"，训练轻量分类器判断模型何时已获取足够上下文，实现提前终止处理，在6个QA数据集上平均提高3.4%准确率同时减少1.33×token消耗。

**[Latent Principle Discovery For Language Model Self-Improvement](interpretability/latent_principle_discovery_for_language_model_self-improvement.md)**

:   STaPLe 提出后验正则化的蒙特卡洛 EM 算法，让 7-8B 小模型自行发现指导自我修正的"原则"（latent principle），通过迭代发现-学习循环实现自我改进，在 AlpacaEval 上提升 8-10% 胜率、MT-Bench 平均提升 +0.3，并可通过聚类压缩至可解释的 constitution。

**[Learning To Focus Causal Attention Distillation Via Gradient-Guided Token Prunin](interpretability/learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)**

:   提出Learning to Focus (LeaF)框架，通过梯度引导识别训练数据中的"混淆token"（confounding tokens），在知识蒸馏过程中剪枝这些token以构建反事实样本，使学生模型的注意力对齐到教师模型关注的关键上下文token上，从而提升数学推理和代码生成的准确性。

**[Llm Probing With Contrastive Eigenproblems Improving Understanding And Applicabi](interpretability/llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)**

:   本文对无监督探测方法 CCS（Contrast-Consistent Search）进行了深入分析，提出将 CCS 重新表述为特征值问题（Contrastive Eigenproblems），获得闭式解和可解释的特征值，避免了 CCS 对随机初始化的敏感性，并自然扩展到多变量设置。

**[Minimizing False-Positive Attributions In Explanations Of Non-Linear Models](interpretability/minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)**

:   针对非线性模型的XAI解释中抑制变量(suppressor variable)导致的假阳性归因问题，提出PatternLocal方法，将局部判别式代理模型权重转换为生成式表示，在XAI-TRIS基准、MRI人工病灶和EEG运动想象三个数据集上显著减少了假阳性特征归因。

**[Monte Carlo Expected Threat Mocet Scoring](interpretability/monte_carlo_expected_threat_mocet_scoring.md)**

:   提出 MOCET（Monte Carlo Expected Threat）评分框架，通过将 LLM 生成的生物武器制造协议分解为逐步 Bernoulli 试验，结合 k-NN 语义嵌入的成功概率估计和蒙特卡洛模拟，生成可解释的、可自动化的威胁量化指标，用于衡量 LLM 在生物安全领域的真实世界风险。

**[Mopformer Motion-Primitive Transformer For Wearable-Sensor Activity Recognition](interpretability/mopformer_motion-primitive_transformer_for_wearable-sensor_activity_recognition.md)**

:   提出 MoPFormer，将可穿戴传感器信号分解为运动原语（motion primitives）序列，通过 Transformer 建模原语间的时序依赖关系，在多个 HAR 基准上超越 SOTA 并保持轻量化。

**[Ordshap Feature Position Importance For Sequential Black-Box Models](interpretability/ordshap_feature_position_importance_for_sequential_black-box_models.md)**

:   提出 OrdShap，一种针对序列模型的特征归因方法，首次将特征的**值重要性（Value Importance, VI）**与**位置重要性（Position Importance, PI）**解耦，基于 Sanchez-Bergantiños 博弈论值提供理论保证。

**[Out Of Control -- Why Alignment Needs Formal Control Theory And An Alignment Con](interpretability/out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)**

:   本文是一篇 position paper，主张将形式化最优控制理论作为 AI 对齐研究的核心工具，并提出"对齐控制栈"(Alignment Control Stack, ACS)——一个从物理硬件层到社会治理层的十层分层框架，用于系统地组织和分析不同对齐方法的测量、控制与互操作性。

**[Partial Information Decomposition Via Normalizing Flows In Latent Gaussian Distr](interpretability/partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)**

:   提出两个互补工具：Thin-PID 是一种高效高斯 PID 算法（比已有方法快 10×），Flow-PID 用 normalizing flow 将任意输入分布转换为高斯再计算 PID，解决了 PID 在连续高维数据上不可行的问题，并证明了"联合高斯解是否最优"这一开放问题。

**[Probabilistic Token Alignment For Large Language Model Fusion](interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)**

:   将 LLM 融合中的 token 对齐问题重新建模为最优传输（Optimal Transport）问题，用动态 token 配对 + Sinkhorn 算法实现"软"概率对齐取代传统硬映射，在 6 大基准 78 个任务上相比 FuseLLM 平均提升 +1.72%，同时在困难任务上大幅缓解性能退化（从 -13.04% 降至 -4.07%）。

**[Rectifying Shortcut Behaviors In Preference-Based Reward Learning](interpretability/rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)**

:   提出 PRISM（Preference-based Reward Invariance for Shortcut Mitigation），将 reward hacking 统一建模为 shortcut learning 问题，通过群不变核（group-invariant kernels）和随机特征映射近似来同时缓解多种 spurious correlation（冗长性、谄媚、语气等），在 out-of-distribution 偏好数据和下游策略模型上一致提升表现。

**[Saying The Unsaid Revealing The Hidden Language Of Multimodal Systems Through Te](interpretability/saying_the_unsaid_revealing_the_hidden_language_of_multimodal_systems_through_te.md)**

:   通过多轮"电话游戏"（图像→文本→图像循环）利用多模态系统的偏好偏差，量化系统隐含空间中概念间的连接强度（即"隐含语言"），贡献Telescope数据集（10,000+概念对），建立可在测试时扩展的多模态系统"世界地图"。

**[Scpilot Large Language Model Reasoning Toward Automated Single-Cell Analysis And](interpretability/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)**

:   提出 scPilot 框架和 scBench 基准，让LLM直接在单细胞RNA-seq数据上进行"组学原生推理"（读取标记基因→提出假设→调用工具验证→迭代修正），实现细胞类型标注准确率提升11%、轨迹推断graph-edit distance降低30%。

**[Self-Supervised Contrastive Learning Is Approximately Supervised Contrastive Lea](interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)**

:   从理论上证明自监督对比学习（DCL）近似等价于一种有监督对比损失（NSCL），两者差距以 $O(1/C)$ 速度随类别数增加而消失；进一步证明 NSCL 全局最优解满足 Neural Collapse（增强坍缩 + 类内坍缩 + Simplex ETF），并提出基于方向性 CDNV 的更紧的 few-shot 误差界。

**[Shap Values Via Sparse Fourier Representation](interpretability/shap_values_via_sparse_fourier_representation.md)**

:   提出 FourierShap 算法，先将黑盒预测器近似为稀疏 Fourier 表示，再利用 Fourier 基函数的 SHAP 值闭式公式高效计算特征归因，实现相比 KernelShap 10-10000 倍的加速，同时支持精度-效率的可调权衡。

**[Simulating Society Requires Simulating Thought](interpretability/simulating_society_requires_simulating_thought.md)**

:   本文提出从"行为主义"模式转向"认知建模"范式，通过 GenMinds 框架用因果信念图建模 LLM Agent 的内部推理过程，并设计 RECAP 基准从可追溯性、人口统计敏感性和干预一致性三维度评估推理保真度。

**[Sloth Scaling Laws For Llm Skills To Predict Multi-Benchmark Performance Across ](interpretability/sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)**

:   提出Skills Scaling Laws (Sloth)，通过假设LLM性能由低维潜在技能（如推理、指令遵循）驱动，利用benchmark间的相关性构建跨模型家族的缩放定律，用少量家族数据即可预测大模型在多个benchmark上的表现。

**[Spex A Spectral Approach To Explainable Clustering](interpretability/spex_a_spectral_approach_to_explainable_clustering.md)**

:   提出SpEx，基于谱图划分的通用可解释聚类方法，可将任意参考聚类（无需质心）通过坐标切割决策树"圆化"为可解释聚类，或直接在kNN图上进行无参考聚类。

**[Steering Information Utility In Key-Value Memory For Language Model Post-Trainin](interpretability/steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)**

:   提出 InfoSteer，一种轻量级方法，将 Transformer 的 FFN 层视为关联键值记忆，通过前向传播干预（提升低活跃记忆向量的 key coefficient）和反向传播正则化（最大化 key 分布熵）来促进预训练知识在后训练阶段的充分利用。在 Qwen/LLaMA/Gemma 三个系列 6 个模型上，15 个 ID+OOD 任务一致提升，且被引导的 LM 展现出自适应信息分配行为。

**[Tangledfeatures Robust Feature Selection In Highly Correlated Spaces](interpretability/tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)**

:   提出 TangledFeatures，一个以**特征稳定性**为核心目标的选择框架，通过相关性图聚类→集成代表选择→随机森林精炼的三阶段管线，在高度相关的特征空间中实现跨重采样高度可复现且与领域知识一致的特征子集，在丙氨酸二肽骨架扭转角预测中验证有效。

**[The Non-Linear Representation Dilemma Is Causal Abstraction Enough For Mechanist](interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)**

:   证明了当因果抽象（causal abstraction）中的对齐映射不受线性约束时，任意神经网络都可以被映射到任意算法，使得因果抽象变得平凡而无信息量，由此提出"非线性表示困境"——在对齐映射的复杂度与准确度之间缺乏原则性的权衡方式。

**[The Trilemma Of Truth In Large Language Models](interpretability/the_trilemma_of_truth_in_large_language_models.md)**

:   提出 sAwMIL（稀疏感知多实例学习）三类探测框架，结合 MIL 和保形预测，将 LLM 内部激活分类为 true/false/neither，揭示真假信号并非简单的双向对称编码，而是跨越多维子空间的分布式表征。

**[Time-Evolving Dynamical System For Learning Latent Representations Of Mouse Visu](interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)**

:   提出TE-ViDS，一种时序潜变量模型，将视觉神经活动分解为与视觉刺激相关的外部表征和反映内部状态的内部表征，通过时间演化结构和对比学习实现最优的自然场景/视频解码性能。

**[Toward Explainable Offline Rl Analyzing Representations In Intrinsically Motivat](interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)**

:   提出一个系统性的事后可解释性框架，分析内在动机（基于Random Network Distillation）如何塑造Elastic Decision Transformer的嵌入空间几何结构，揭示不同内在动机变体创造了根本不同的表示结构——EDT-SIL促进紧凑表示，EDT-TIL增强正交性——且嵌入属性与任务性能存在强烈的环境特异性相关。

**[Toward Real-World Text Image Forgery Localization Structured And Interpretable D](interpretability/toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)**

:   提出基于傅里叶级数的篡改合成框架 FSTS，通过从67名人类参与者收集的16750个真实篡改实例中建模"不可见分布"（篡改操作参数的高维分布），生成更贴近真实世界的合成训练数据，显著提升文本图像篡改定位模型的泛化能力。

**[Towards Interpretability Without Sacrifice Faithful Dense Layer Decomposition Wi](interpretability/towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)**

:   提出 Mixture of Decoders (MxD)，将 LLM 的 MLP 层分解为数万个稀疏激活的专家子层（layer-level sparsity），每个专家通过 Hadamard 乘积张量分解实现满秩线性变换，在稀疏性-准确性权衡上显著优于 Transcoders，同时保持可解释性。

**[Towards Scaling Laws For Symbolic Regression](interpretability/towards_scaling_laws_for_symbolic_regression.md)**

:   首次系统研究符号回归（SR）中的缩放定律，证明基于 Transformer 的端到端 SR 在三个数量级的计算范围内遵循幂律缩放趋势，并给出最优 token-to-parameter ratio $\approx 15$、batch size 和学习率随模型规模增长的经验规律。

**[Transformer Key-Value Memories Are Nearly As Interpretable As Sparse Autoencoder](interpretability/transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)**

:   系统比较了Transformer前馈层（FF）的键值记忆特征与稀疏自编码器（SAE）学到的特征的可解释性，发现两者在现有评测指标上表现相当，FF-KV在某些方面甚至更优，质疑了SAE作为特征发现工具的必要性。

**[Tropical Attention Neural Algorithmic Reasoning For Combinatorial Algorithms](interpretability/tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)**

:   Tropical Attention用热带代数几何替代softmax点积注意力，在热带射影空间中进行分段线性推理，实现与组合算法多面体决策结构的对齐，首次将神经算法推理扩展到NP-hard问题，在长度/数值/噪声三种OOD泛化上全面超越softmax基线。

**[Uncovering Graph Reasoning In Decoder-Only Transformers With Circuit Tracing](interpretability/uncovering_graph_reasoning_in_decoder-only_transformers_with_circuit_tracing.md)**

:   通过电路追踪 (circuit tracing) 框架分析 decoder-only Transformer 在图推理任务上的内部机制，发现了 token merging 和 structural memorization 两个核心推理机制。

**[Urls Help Topics Guide Understanding Metadata Utility In Llm Training](interpretability/urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)**

:   系统评估了三类元数据（URL、质量分数、主题/格式域信息）作为预训练上下文的效果：发现只有 URL 能加速训练（100B token 用 60B 即达到相同下游性能），且仅在长 prompt（5-shot）下有效；质量分数和主题域信息不加速训练但可用于 classifier-free guidance 实现可控生成。

**[Vadtree Explainable Training-Free Video Anomaly Detection Via Hierarchical Granu](interpretability/vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)**

:   提出 VADTree，一种训练无关的视频异常检测框架，利用预训练的通用事件边界检测（GEBD）模型构建层次粒度感知树（HGTree），实现对不同时间跨度异常事件的自适应采样和多粒度推理，在 UCF-Crime、XD-Violence 和 MSAD 三个基准上取得训练无关方法SOTA，甚至超越部分弱监督方法。

**[Valuepilot A Two-Phase Framework For Value-Driven Decision-Making](interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)**

:   提出 ValuePilot 两阶段框架，通过数据集生成工具包（DGT）构建价值标注场景，再用决策模块（DMM）结合用户个性化价值偏好进行多准则决策，在与人类决策对齐方面超过 GPT-5 等强基线。

**[Vlsae Interpreting And Enhancing Visionlanguage Alignment Wi](interpretability/vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)**

:   提出VL-SAE，一种带有距离编码器和模态特定解码器的稀疏自编码器，将视觉和语言表示的语义映射到统一概念集，从而解释和增强VLM的视觉-语言对齐机制，在零样本分类平均提升0.6-0.9%，在POPE幻觉消除上超越专用方法VCD。

**[What Happens During The Loss Plateau Understanding Abrupt Learning In Transforme](interpretability/what_happens_during_the_loss_plateau_understanding_abrupt_learning_in_transforme.md)**

:   系统研究 Transformer 训练中的"突变学习"现象，揭示 loss 平台期内模型已学会部分解、同时表现出输出重复偏差和表示坍缩，并证明注意力图的缓慢学习是关键瓶颈，相关发现在 Pythia/OLMo 等 LLM 预训练早期也得到验证。

**[Why Is Attention Sparse In Particle Transformer](interpretability/why_is_attention_sparse_in_particle_transformer.md)**

:   本文系统性地分析了 Particle Transformer（ParT）在 jet tagging 任务中训练后出现的近乎二值化稀疏 attention 现象，通过跨数据集对比和消融实验揭示了稀疏性主要源自 attention 机制自身而非物理启发的 interaction 矩阵，但 interaction 矩阵通过影响绝大多数 token 的 argmax 选择对最终性能不可或缺。

---

## 🧑 人体理解 { #human_understanding }

**[A Differential And Pointwise Control Approach To Reinforceme](human_understanding/a_differential_and_pointwise_control_approach_to_reinforceme.md)**

:   将RL问题通过连续时间控制的微分对偶形式重新表述，利用哈密顿结构嵌入物理先验，提出dfPO算法实现逐点策略优化，在科学计算任务（曲面建模、网格控制、分子动力学）上以更少样本超越12个RL基线。

**[A Practical Guide For Incorporating Symmetry In Diffusion Policy](human_understanding/a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)**

:   本文提出了一套将对称性融入扩散策略的实用指南——通过不变性表征（相对轨迹动作 + 手眼感知）、等变视觉编码器和 Frame Averaging 三种简单方法，在 MimicGen 12 个任务上达到了接近甚至超越完全等变扩散策略的性能，同时实现复杂度大幅降低。

**[A Regularized Newton Method For Nonconvex Optimization With](human_understanding/a_regularized_newton_method_for_nonconvex_optimization_with.md)**

:   提出一类基于当前与历史梯度构造的新型正则化器，结合带负曲率监测的共轭梯度法求解正则化Newton方程，在不需要Hessian Lipschitz常数先验知识的自适应框架下，首次同时实现了$O(\epsilon^{-3/2})$最优全局迭代复杂度和二次局部收敛速率。

**[A Simple Linear Patch Revives Layerpruned Large Language Mod](human_understanding/a_simple_linear_patch_revives_layerpruned_large_language_mod.md)**

:   LinearPatch 通过在层剪枝界面插入一个融合了 Hadamard 变换和通道缩放的轻量对称矩阵，修复了剪枝造成的激活幅度失配问题，在 LLaMA-3-8B 上无训练保留 94.15% 性能，30 分钟蒸馏后达 95.16%。

**[Ada-Kv Optimizing Kv Cache Eviction By Adaptive Budget Allocation For Efficient ](human_understanding/ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)**

:   发现现有 KV cache 驱逐方法对所有注意力头均匀分配预算忽略了头间注意力集中度的巨大差异,提出 Ada-KV——首个 head-wise 自适应预算分配策略,将稀疏头的预算重新分配给分散头,理论证明最小化驱逐损失上界,在 29 个数据集上即插即用地提升现有方法。

**[Agint Agentic Graph Compilation For Software Engineering Age](human_understanding/agint_agentic_graph_compilation_for_software_engineering_age.md)**

:   提出 Agint 图编译器，将自然语言意图通过六层类型地板（TEXT→TYPED→SPEC→STUB→SHIM→PURE）渐进编译为类型化DAG，配合混合JIT运行时和Unix风格工具链，使AI代码生成从脆弱的单次文本预测变为结构化、可并行、可复现的编译过程。

**[Bedlam20 Synthetic Humans And Cameras In Motion](human_understanding/bedlam20_synthetic_humans_and_cameras_in_motion.md)**

:   BEDLAM2.0 在 BEDLAM 基础上全面升级——引入多样化相机运动（合成平移/追踪/环绕 + 手持/头戴设备捕捉）、更广体型覆盖（BMI 18-41）、strand-based 发型、鞋子、分级服装和更多3D环境，构建 27K+ 序列 / 8M+ 帧的合成数据集，仅用合成数据训练即可在世界坐标系人体运动估计上超越 SOTA。

**[Breaking The Gradient Barrier Unveiling Large Language Models For Strategic Clas](human_understanding/breaking_the_gradient_barrier_unveiling_large_language_models_for_strategic_clas.md)**

:   提出 GLIM（Gradient-free Learning In-context Method），首次利用 LLM 的 In-Context Learning 机制隐式模拟策略分类中的双层优化（特征操纵 + 决策规则优化），无需微调即可在大规模数据上高效完成策略分类任务。

**[Bubbleformer Forecasting Boiling With Transformers](human_understanding/bubbleformer_forecasting_boiling_with_transformers.md)**

:   提出 BubbleFormer，基于分解时空轴注意力的 Transformer 架构用于预测沸腾动力学——包括难以预测的自主气泡成核事件，配合 BubbleML 2.0 数据集（160+ 高保真仿真），在多种流体、几何和壁面条件下实现准确的沸腾时空过程预测。

**[Consistent Supervised-Unsupervised Alignment For Generalized Category Discovery](human_understanding/consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)**

:   提出 NC-GCD 框架，通过预分配固定的 Equiangular Tight Frame (ETF) 原型为已知类和新类建立统一优化目标，结合语义一致性匹配器 (SCM) 稳定跨迭代伪标签分配，在 6 个 GCD 基准上显著提升新类发现精度。

**[Counteractive Rl Rethinking Core Principles For Efficient And Scalable Deep Rein](human_understanding/counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)**

:   CoAct TD Learning 颠覆 ε-greedy 的随机探索范式——以概率 ε 选择最小化 $Q(s,a)$ 的动作（而非随机动作）来获取高时间差分信号，理论证明其产生更大 TD 误差，在 Atari 100K 上实现 248% 性能提升，仅需改动 2 行代码且零额外计算。

**[Cpep Contrastive Pose-Emg Pre-Training Enhances Gesture Generalization On Emg Si](human_understanding/cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si.md)**

:   提出 CPEP 框架，通过对比学习将低质量 EMG 信号表征与高质量手部姿态表征对齐，使 EMG 编码器获得姿态感知能力，首次实现从 EMG 信号零样本识别未见手势，分布内手势分类提升 21%、未见手势分类提升 72%。

**[Cycle-Sync Robust Global Camera Pose Estimation Through Enhanced Cycle-Consisten](human_understanding/cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)**

:   提出 Cycle-Sync 全局相机位姿估计框架，通过将消息传递最小二乘 (MPLS) 扩展到相机位置估计、引入 Welsch 型鲁棒损失和环一致性加权，在无需 bundle adjustment 的情况下超越了包括完整 SfM pipeline（含 BA）在内的所有基线方法。

**[Data-Juicer 20 Cloud-Scale Adaptive Data Processing For And With Foundation Mode](human_understanding/data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)**

:   Data-Juicer 2.0 是面向基础模型的云规模多模态数据处理系统，150+ 跨文本/图像/视频/音频算子，支持自适应分布式执行（Ray/MaxCompute），在 10000+ CPU 核心上高效处理 TB 级数据，已广泛应用于阿里云 PAI 等产品。

**[Decomposition Of Small Transformer Models](human_understanding/decomposition_of_small_transformer_models.md)**

:   将 Stochastic Parameter Decomposition (SPD) 扩展到 Transformer，设计适用于序列数据的因果重要性函数和新损失函数，在玩具 induction head 上恢复期望两步电路，在 GPT-2-small 上定位到"高尔夫""篮球"等可解释概念对应的 rank-1 参数子空间。

**[Devfd Developmental Face Forgery Detection By Learning Shared And Orthogonal Lor](human_understanding/devfd_developmental_face_forgery_detection_by_learning_shared_and_orthogonal_lor.md)**

:   提出 DevFD——一种发展式 MoE 架构，用共享 Real-LoRA 建模真实人脸共性、正交 Fake-LoRA 序列逐步建模新伪造类型，并通过将正交梯度集成到正交损失中缓解灾难性遗忘，在持续学习人脸伪造检测中达到最高准确率和最低遗忘率。

**[Discovering Transformer Circuits Via A Hybrid Attribution And Pruning Framework](human_understanding/discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)**

:   提出混合归因与剪枝框架 HAP，先用快速的边归因修补（EAP）筛选高潜力子图，再在缩小后的搜索空间上运行精确的边剪枝（EP），在 GPT-2 Small 的 IOI 任务上比纯 EP 快 46% 且保持相当的电路忠实度，同时成功保留了 EAP 单独使用时会遗漏的 S-inhibition 头。

**[Distillation Robustifies Unlearning](human_understanding/distillation_robustifies_unlearning.md)**

:   揭示了"蒸馏能使遗忘变得鲁棒"的核心发现——将遗忘后的模型蒸馏到随机初始化的学生网络中能有效丢弃潜在能力，并基于此提出UNDO方法（Unlearn-Noise-Distill-on-Outputs），通过对遗忘模型权重加噪再蒸馏，建立了计算量与鲁棒性之间的可调权衡，在合成任务和WMDP基准上接近从头重训的黄金标准。

**[Distribution Learning Meets Graph Structure Sampling](human_understanding/distribution_learning_meets_graph_structure_sampling.md)**

:   本文建立了高维概率图模型 PAC 学习与图结构高效计数/采样之间的新联系，利用在线学习框架（EWA/RWM）将指数级专家集合的维护问题转化为 DAG 结构的加权采样问题，首次给出了弦图骨架贝叶斯网络的高效 agnostic 学习算法，并将树结构分布的样本复杂度从 O(nk³/ε) 改进到最优的 O(nk²/ε)。

**[Emergent World Beliefs Exploring Transformers In Stochastic Games](human_understanding/emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)**

:   将LLM涌现世界模型的研究从完全信息游戏（Othello、国际象棋）扩展到不完全信息领域（德州扑克），通过在PHH格式扑克数据上预训练GPT-2并探测其内部激活，证明模型不仅学习了确定性特征（牌型识别~98%准确率），还自发发展了对随机性特征（胜率/equity，相关系数0.59）的内部表示。

**[Evolutionary Learning In Spatial Agent-Based Models For Physical Climate Risk As](human_understanding/evolutionary_learning_in_spatial_agent-based_models_for_physical_climate_risk_as.md)**

:   提出一种整合地理空间气候灾害数据与进化学习机制的Agent-Based Model（ABM），在包含商品-制造-零售三级供应链的简化经济网络上，通过RCP8.5洪水投影模拟2025-2100年的经济响应，证明了进化自适应机制使企业在气候压力下维持显著更高的生产、资本、流动性和就业水平，同时揭示了传统资产级评估无法捕捉的供应链系统性风险。

**[Exploration Of Incremental Synthetic Non-Morphed Images For Single Morphing Atta](human_understanding/exploration_of_incremental_synthetic_non-morphed_images_for_single_morphing_atta.md)**

:   系统研究了在单图像变形攻击检测（S-MAD）训练中增量引入合成非变形人脸图像的效果，发现适量的合成数据（~75%增量）可提升跨数据集泛化能力（EER从6.17%降至6.10%），但过度使用或仅用合成数据会导致性能严重退化（EER升至~38%）。

**[Face-Human-Bench A Comprehensive Benchmark Of Face And Human Understanding For M](human_understanding/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)**

:   提出 Face-Human-Bench，首个系统评估多模态大模型人脸与人体理解能力的基准，包含三级能力分类体系（2个L1 × 10个L2 × 18个L3），开发集与测试集各 1800 题，支持中英双语，评测 25 个主流 MLLM 并揭示其与专家模型的显著差距。

**[Face A General Framework For Mapping Collaborative Filtering Embeddings Into Llm](human_understanding/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)**

:   FACE 提出将协同过滤（CF）嵌入通过解纠缠投影 + 残差量化映射为 LLM 预训练 token（描述符），再用对比学习对齐语义，无需微调 LLM 即可实现 CF 嵌入的语义解读和推荐性能增强。

**[Face Faithful Automatic Concept Extraction](human_understanding/face_faithful_automatic_concept_extraction.md)**

:   提出 FACE 框架，在非负矩阵分解 (NMF) 中加入 KL 散度正则项，约束概念重建后的激活值保持与原始模型预测一致，从而提取真正忠实于模型决策过程的概念解释，在 ImageNet/COCO/CelebA 上全面超越 CRAFT 和 ICE。

**[Faster Algorithm For Structured John Ellipsoid Computation](human_understanding/faster_algorithm_for_structured_john_ellipsoid_computation.md)**

:   针对对称凸多面体 $P = \{x \in \mathbb{R}^d : -\mathbf{1}_n \leq Ax \leq \mathbf{1}_n\}$ 的 John 椭球计算问题，提出两个快速算法：基于 sketching 的近输入稀疏度算法 $\widetilde{O}(\text{nnz}(A) + d^\omega)$ 每次迭代，和基于树宽的算法 $O(n\tau^2)$ 每次迭代，均显著优于已有最优 $O(nd^2)$。

**[Firstaidqa A Synthetic Dataset For First Aid And Emergency Response In Low-Conne](human_understanding/firstaidqa_a_synthetic_dataset_for_first_aid_and_emergency_response_in_low-conne.md)**

:   构建 FirstAidQA，一个包含 5500 条合成急救问答对的数据集，基于认证急救教材用 ChatGPT-4o-mini 生成，经人工验证，旨在支撑低连接/离线环境下急救 AI 系统的微调训练。

**[Graphchain Large Language Models For Large-Scale Graph Analysis Via Tool Chainin](human_understanding/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)**

:   提出 GraphChain 框架，通过渐进式图蒸馏（RL驱动的工具链序列生成）和结构感知测试时自适应（基于图拓扑指纹的轻量适配器），使 LLM 能像人类探索未知环境一样，通过动态工具链序列逐步分析大规模图数据，平均准确率 84.7% 超越最优基线 20.7%，可扩展至 20 万节点。

**[Gui-Rise Structured Reasoning And History Summarization For Gui Navigation](human_understanding/gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)**

:   提出 GUI-Rise 框架，通过结构化推理（进度估计 + 决策推理）、动作预测和历史摘要三个子任务的联合设计，结合 GRPO 强化学习与历史摘要奖励，显著提升 GUI 导航智能体在跨域场景下的泛化能力。

**[Hoi-Dyn Learning Interaction Dynamics For Human-Object Motion Diffusion](human_understanding/hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)**

:   将人体-物体交互（HOI）生成建模为 Driver-Responder 系统，通过轻量级 Transformer 交互动力学模型显式预测物体对人体动作的响应，利用残差动力学损失在训练时增强因果一致性，同时保持推理效率。

**[Human-Machine Ritual Synergic Performance Through Real-Time Motion Recognition](human_understanding/human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)**

:   提出一种轻量级实时动作识别系统，利用可穿戴 IMU 传感器 + MiniRocket 时序分类器实现 <50ms 延迟的舞者特定动作识别（96.05% 准确率），通过"具身记忆映射"将舞者的个人动作-声音关联编码到系统中，构建了一种尊重人体表达深度的人机协作表演范式。

**[In-Context Compositional Learning Via Sparse Coding Transformer](human_understanding/in-context_compositional_learning_via_sparse_coding_transformer.md)**

:   受稀疏编码启发，将 Transformer 注意力机制重新解释为在编码字典和解码字典上的投影，通过稀疏系数显式表示组合规则，并利用提升方案（lifting scheme）将上下文任务的组合规则迁移到目标任务。

**[Incentivizing Reasoning For Advanced Instruction-Following Of Large Language Mod](human_understanding/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)**

:   提出 RAIF，通过 RL+规则中心奖励培养 LLM 在复杂指令（含 And/Chain/Selection/Nested 组合约束）下的深度推理能力：发现 vanilla CoT 对指令跟随有负面影响（因 LLM 只会浅层复述指令），设计 superior CoT enforcement（样本级对比过滤无效推理）+ 行为克隆控制分布偏移，1.5B 模型匹配 8B 性能，7 个 benchmark 平均提升 11.74%。

**[K-Decore Facilitating Knowledge Transfer In Continual Structured Knowledge Reaso](human_understanding/k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)**

:   提出 K-DeCore 框架，通过知识解耦将结构化知识推理分为任务无关的 schema 过滤和任务特定的 query 构建两阶段，配合双视角记忆构建和结构引导的伪数据合成策略，在固定参数量下实现跨异构 SKR 任务的有效知识迁移。

**[Learning Dense Hand Contact Estimation From Imbalanced Data](human_understanding/learning_dense_hand_contact_estimation_from_imbalanced_data.md)**

:   提出 HACO 框架，通过平衡接触采样（BCS）解决类别不平衡和顶点级类别平衡损失（VCB Loss）解决空间不平衡，首次在 14 个数据集（65.5 万图像）上训练稠密手部接触估计模型，在多种交互场景下达到 SOTA。

**[Learning From Design Procedure To Generate Cad Programs For Data Augmentation](human_understanding/learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)**

:   提出一种受工业设计流程启发的CAD程序数据增强范式，通过向LLM提供参考曲面程序和设计流程描述来引导生成包含B-Spline有机形状的CAD程序，显著缩小了公开CAD数据集与工业级设计在几何复杂度上的差距。

**[Learning Skill-Attributes For Transferable Assessment In Video](human_understanding/learning_skill-attributes_for_transferable_assessment_in_video.md)**

:   提出CrossTrainer方法，通过发现跨运动通用的技能属性（如平衡、控制、手部定位）作为中间表示，训练多模态语言模型从视频中生成可操作反馈和水平评估，在跨运动零样本迁移中相对SOTA提升高达60%。

**[Learning To Watermark A Selective Watermarking Framework For Large Language Mode](human_understanding/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)**

:   提出LTW（Learning to Watermark）框架，使用一个轻量级选择器网络基于句子嵌入、token熵和当前水印比例来自适应决定何时施加水印，通过多目标优化（MGDA）在可检测性和文本质量之间达到Pareto最优，在不降低检测性能的前提下显著提升水印文本质量。

**[Lumia A Handheld Vision-To-Music System For Real-Time Embodied Composition](human_understanding/lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)**

:   提出Lumia——一个手持相机式设备，通过GPT-4 Vision分析拍摄画面生成结构化提示，再由Stable Audio合成音乐循环段，实现从视觉到音乐的实时、具身化即兴创作工作流。

**[Mapping Faithful Reasoning In Language Models](human_understanding/mapping_faithful_reasoning_in_language_models.md)**

:   提出Concept Walk框架，通过将推理模型每步的残差流激活投影到从对比数据学到的概念方向上，追踪内部概念表示在推理过程中的演化轨迹，以此区分CoT链是真正参与计算的还是仅为事后合理化的装饰性输出。

**[Mdreid Modality-Decoupled Learning For Any-To-Any Multi-Modal Object Re-Identifi](human_understanding/mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)**

:   提出MDReID框架，通过将模态特征解耦为模态共享（modality-shared）和模态特有（modality-specific）两部分，实现任意模态组合下的目标重识别（any-to-any ReID），在模态匹配和模态不匹配场景下均大幅超越现有方法。

**[Mechanistic Interpretability Of Rnns Emulating Hidden Markov Models](human_understanding/mechanistic_interpretability_of_rnns_emulating_hidden_markov_models.md)**

:   训练vanilla RNN复现隐马尔可夫模型（HMM）的发射统计量，然后通过逆向工程揭示RNN利用噪声维持的轨道动力学、"kick neuron"电路和自致随机共振机制来实现离散随机状态转换的计算原理。

**[Mingle Mixture Of Null-Space Gated Low-Rank Experts For Test-Time Continual Mode](human_understanding/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)**

:   提出测试时持续模型合并(TTCMM)新范式及Mingle框架，通过低秩专家混合架构和自适应零空间约束门控机制，在测试时利用少量无标签样本实现模型动态合并，在多个基准上以7-9%的优势超越SOTA，同时将遗忘降至接近零。

**[Mixing Expert Knowledge Bring Human Thoughts Back To The Game Of Go](human_understanding/mixing_expert_knowledge_bring_human_thoughts_back_to_the_game_of_go.md)**

:   提出 LoGos，通过混合领域专家数据（围棋）与通用长 CoT 推理数据进行冷启动微调 + GRPO 强化学习，使通用 LLM 在围棋中达到职业棋手水平的同时保持优秀的通用推理能力。

**[Mospa Human Motion Generation Driven By Spatial Audio](human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)**

:   首次定义"空间音频驱动人体运动生成"这一新任务，构建包含 9+ 小时、27 种场景、12 名受试者的双耳音频-运动配对 SAM 数据集，提出 MOSPA 扩散模型在融合 MFCC/tempogram/RMS 等音频特征与声源位置及运动风格条件后，以 FID 7.98 大幅领先 EDGE（14.0）、POPDG（21.0）等音乐/舞蹈基线。

**[Mouse-Guided Gaze Semi-Supervised Learning Of Intention-Aware Representations Fo](human_understanding/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)**

:   提出一种半监督框架，利用鼠标轨迹作为弱监督信号预训练眼动表征，然后在标注数据上微调以区分阅读与扫描行为，在推理时仅使用眼动信号，实现免手操作的辅助阅读检测。

**[Neural Collapse In Cumulative Link Models For Ordinal Regression An Analysis Wit](human_understanding/neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)**

:   将Neural Collapse (NC)理论扩展到基于累积链接模型(CLM)的序数回归(OR)任务中，在无约束特征模型(UFM)框架下证明了Ordinal Neural Collapse (ONC)的三个标志性质：类内均值坍缩(ONC1)、特征坍缩到一维子空间(ONC2)、以及潜变量按类别顺序排列(ONC3)，并在零正则极限下揭示了潜变量与阈值之间的简洁几何关系。

**[Nnterp A Standardized Interface For Mechanistic Interpretability Of Transformers](human_understanding/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)**

:   开发 nnterp 库，作为 NNsight 的轻量封装层，通过系统化的模块重命名和自动验证测试，为 21 个架构族 50+ 个 Transformer 模型变体提供统一的内部激活访问接口，内置 logit lens、patchscope、activation steering 等常用可解释性方法，解决了 TransformerLens 的正确性问题和 NNsight 的标准化问题之间的根本性权衡。

**[Node-Based Editing For Multimodal Generation Of Text Audio Image And Video](human_understanding/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)**

:   提出一个节点图式故事编辑系统，允许创作者通过自然语言和节点级操作迭代地生成、编辑和比较多模态内容（文本、音频、图像、视频），支持线性和分支叙事结构。

**[Offline Policy Evaluation Of Multi-Turn Llm Health Coaching With Real Users](human_understanding/offline_policy_evaluation_of_multi-turn_llm_health_coaching_with_real_users.md)**

:   在实际部署的 LLM 健康教练系统上进行离线策略评估（OPE），发现统一的高工具使用策略虽提升平均奖励但损害特定用户子群，并通过模拟器验证了早期信息增益探索（好奇心奖励）可加速用户特征识别和提升任务成功率。

**[Parco Parallel Autoregressive Models For Multi-Agent Combinatorial Optimization](human_understanding/parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)**

:   提出 PARCO 框架，通过 Communication Layers 实现智能体间协调、Multiple Pointer Mechanism 实现并行解码、Priority-based Conflict Handler 解决冲突，高效求解多智能体组合优化问题。

**[Policy Compatible Skill Incremental Learning Via Lazy Learning Interface](human_understanding/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)**

:   提出SIL-C框架，通过双向惰性学习接口(bilateral lazy learning interface)实现技能增量学习中的技能-策略兼容性，使增量更新的技能能直接提升下游策略性能而无需重训练或结构调整。

**[Power Ensemble Aggregation For Improved Extreme Event Ai Prediction](human_understanding/power_ensemble_aggregation_for_improved_extreme_event_ai_prediction.md)**

:   提出基于幂均值的自适应集成聚合方法，通过对生成式天气预测模型的集成成员得分施加非线性聚合（幂指数$p>1$），显著提升极端高温事件的分类性能，尤其在高分位数阈值下效果更佳。

**[Preference-Based Reinforcement Learning Beyond Pairwise Comparisons Benefits Of ](human_understanding/preference-based_reinforcement_learning_beyond_pairwise_comparisons_benefits_of_.md)**

:   在偏好强化学习中提出 M-AUPO 算法,利用 Plackett-Luce 排序模型处理多选项比较反馈,首次从理论上证明更大的子集规模直接改善样本效率。

**[Raptr Radar-Based 3D Pose Estimation Using Transformer](human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)**

:   提出RAPTR，首个利用弱监督（3D BBox + 2D关键点标签）进行雷达3D人体姿态估计的Transformer框架，通过伪3D可变形注意力和结构化损失函数在两个室内数据集上大幅超过基线。

**[Recurrent Attention-Based Token Selection For Efficient Streaming Video-Llms](human_understanding/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)**

:   提出 rLiVS（Recurrent LLM-informed Visual Selection），一种无需训练的通用流式视频理解方法，通过LLM注意力权重选择关键视觉token（仅保留~6%）、循环复用历史token、基于字幕的检索问答三重设计，在流式视频基准上取得SOTA。

**[Reflective Translation Improving Low-Resource Machine Translation Via Structured](human_understanding/reflective_translation_improving_low-resource_machine_translation_via_structured.md)**

:   提出 Reflective Translation 框架，让 LLM 在推理时对自身的初始翻译进行结构化自我批判（识别误译/遗漏/语义扭曲），再根据批判生成修正翻译，无需微调或额外标注数据即可在 isiZulu/isiXhosa 等低资源非洲语言上取得 BLEU 和 COMET 的统计显著提升。

**[Searching Latent Program Spaces](human_understanding/searching_latent_program_spaces.md)**

:   提出 Latent Program Network（LPN），通过编码器将输入-输出示例映射为潜在程序表示，在测试时通过梯度搜索潜在空间来适应新任务，在 ARC-AGI 基准上显著优于 in-context learning 和 test-time training 方法。

**[Semantic Retrieval Augmented Contrastive Learning For Sequential Recommendation](human_understanding/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)**

:   提出SRA-CL框架，利用LLM的语义理解能力构建高质量对比样本对，通过语义检索+可学习样本合成器增强序列推荐的对比学习，以即插即用的方式在4个数据集上取得SOTA。

**[Sharpness-Aware Minimization With Z-Score Gradient Filtering](human_understanding/sharpness-aware_minimization_with_z-score_gradient_filtering.md)**

:   提出 Z-Score Filtered SAM (ZSAM)，通过对每层梯度进行 Z-Score 统计过滤，仅保留最显著的梯度分量进行扰动上升步骤，从而引导优化器更有效地搜索平坦极小值，在多个数据集和架构上一致提升测试精度。

**[Spatial Understanding From Videos Structured Prompts Meet Simulation Data](human_understanding/spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)**

:   提出 SpatialMind 结构化提示策略与 ScanForgeQA 合成QA数据集的双管齐下方案，在不修改VLM架构的前提下显著增强其从扫描视频进行3D空间推理的能力。

**[Specattn Speculating Sparse Attention](human_understanding/specattn_speculating_sparse_attention.md)**

:   SpecAttn 提出一种无需训练的方法，利用投机解码中草稿模型已计算的注意力权重来预测验证模型的重要 token，通过 KL 散度层映射 + 免排序 top-p 核选择 + 动态 KV 缓存剪枝，实现 78.4% 的 KV 缓存访问减少，困惑度仅增加 15.29%，显著优于现有稀疏注意力方法。

**[Spurious-Aware Prototype Refinement For Reliable Out-Of-Distribution Detection](human_understanding/spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)**

:   SPROD 是一种后置（post-hoc）OOD 检测方法，专门应对训练数据中的虚假相关——通过将每个类别的原型细分为"正确分类"和"误分类"子组（后者共享虚假特征），配合 K-means 式精炼和距离式（生成式）评分，在 5 个虚假相关 OOD 基准上平均 AUROC 85.1%（+4.8% vs 次优 KNN），FPR@95 49.0%（-9.3% vs 次优）。

**[Stable Coresets Via Posterior Sampling Aligning Induced And Full Loss Landscapes](human_understanding/stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)**

:   提出基于后验采样的 coreset 选择框架，通过在 BatchNorm 层上采样权重扰动来平滑损失曲面，保证 coreset 与全数据集的损失景观对齐（包含 Hessian 和 Newton 步的近似），在高标签噪声下显著优于现有方法。

**[Stealthy Yet Effective Distribution-Preserving Backdoor Attacks On Graph Classif](human_understanding/stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)**

:   提出 DPSBA，一种面向图分类的 clean-label 后门攻击框架，通过对抗训练生成分布内（in-distribution）触发子图，同时抑制结构异常和语义异常，在保持高攻击成功率的同时显著提升隐蔽性。

**[Stochastic Momentum Methods For Non-Smooth Non-Convex Finite-Sum Coupled Composi](human_understanding/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)**

:   针对非光滑非凸有限和耦合复合优化 (FCCO) 问题，提出两种随机动量方法 SONEX（单循环）和 ALEXR2（双循环），通过外层 Moreau 包络平滑和嵌套平滑技术将迭代复杂度从 $O(1/\epsilon^6)$ 改进至 $O(1/\epsilon^5)$，并在非凸不等式约束优化中取得同等最优复杂度。

**[Succeed Or Learn Slowly Sample Efficient Off-Policy Reinforcement Learning For M](human_understanding/succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)**

:   提出SoLS算法，通过不对称策略更新机制（成功时激进学习、失败时保守正则化）和成功转换回放（STR），实现基础模型在移动应用控制任务上的高效强化学习微调，在AndroidWorld上达到51.3%成功率。

**[Switchable Token-Specific Codebook Quantization For Face Image Compression](human_understanding/switchable_token-specific_codebook_quantization_for_face_image_compression.md)**

:   提出可切换的token专属码本量化机制（STSCQ），通过图像级码本路由和token级码本分割的层次动态结构，在超低比特率下显著提升人脸图像的压缩重建质量和识别精度。

**[Symrtlo Enhancing Rtl Code Optimization With Llms And Neuron-Inspired Symbolic R](human_understanding/symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)**

:   提出 SymRTLO，首个将LLM与符号推理集成的神经符号框架用于RTL代码优化，通过检索增强优化规则、AST模板引导代码生成和FSM符号系统，在功耗、性能和面积(PPA)上分别获得最高43.9%、62.5%和51.1%的提升。

**[Tensorrl-Qas Reinforcement Learning With Tensor Networks For Improved Quantum Ar](human_understanding/tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)**

:   提出 TensorRL-QAS 框架，通过用张量网络（MPS/DMRG）对强化学习量子架构搜索进行 warm-start，显著降低电路深度和 CNOT 门数量（最高 10 倍），同时加速训练（最高 98%），有效解决了 RL-QAS 在大规模量子系统上的可扩展性瓶颈。

**[The Last Vote A Multi-Stakeholder Framework For Language Model Governance](human_understanding/the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)**

:   提出一个面向语言模型治理的综合框架，包含七类民主风险分类体系、利益相关方自适应事件严重度评分(ISS)、以及分阶段六年实施路线图，旨在将民主价值融入AI监管的制度设计中。

**[The Transparent Earth A Multimodal Foundation Model For The Earths Subsurface](human_understanding/the_transparent_earth_a_multimodal_foundation_model_for_the_earths_subsurface.md)**

:   提出Transparent Earth，一种基于Transformer的多模态基础模型，通过位置编码和文本衍生的模态嵌入融合8种异质地球物理观测数据，实现地球地下属性的零样本推断和上下文学习预测。

**[Uncle Towards Scalable Dynamic Causal Discovery In Non-Linear Temporal Systems](human_understanding/uncle_towards_scalable_dynamic_causal_discovery_in_non-linear_temporal_systems.md)**

:   提出 UnCLe，一种基于 TCN 自编码器解耦和自回归依赖矩阵的可扩展动态因果发现方法，通过时序扰动后逐数据点预测误差增量推断时变因果关系，在静态和动态因果发现基准上均达到 SOTA。

**[Uncovering Strategic Egoism Behaviors In Large Language Models](human_understanding/uncovering_strategic_egoism_behaviors_in_large_language_models.md)**

:   首次形式化定义LLM中的"策略性自利"（Strategic Egoism）行为并构建SEBench基准（160个场景×6类自利维度），实验发现7个主流LLM在激励诱惑下平均69.11%的决策选择自利策略，操纵胁迫与规则规避是最常见手段，且自利倾向与毒性语言生成呈正相关。

**[Vasa-3D Lifelike Audio-Driven Gaussian Head Avatars From A Single Image](human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)**

:   提出VASA-3D，通过将VASA-1的2D运动隐空间适配到3D高斯溅射表征，并利用VASA-1合成训练数据进行单图定制优化，实现了从单张肖像照到逼真音频驱动3D头部化身的实时生成（512×512, 75fps）。

**[Vimorag Video-Based Retrieval-Augmented 3D Motion Generation For Motion Language](human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)**

:   提出 VimoRAG 框架，利用大规模野外视频数据库作为2D运动先验来增强3D运动生成，通过 Gemini-MVR 检索器和 McDPO 训练策略解决人体动作视频检索和错误传播两大瓶颈。

**[Vision Transformers For Cosmological Fields Application To Weak Lensing Mass Map](human_understanding/vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)**

:   首次将 Vision Transformers（ViT 和 Swin Transformer）应用于弱引力透镜收敛场的宇宙学参数（$\Omega_m$ 和 $S_8$）约束，通过模拟推断框架系统比较了注意力架构与 CNN 的性能。

**[Words That Unite The World A Unified Framework For Deciphering Central Bank Comm](human_understanding/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)**

:   本文构建了迄今最全面的央行货币政策语料库 WCB（38万+句子、25家央行、跨28年），定义三个NLP任务（立场检测、时间分类、不确定性估计），通过15,075次基准实验发现聚合多银行数据训练的模型显著优于单银行训练，证实了"整体大于部分之和"的原则。

---

## 📊 LLM评测 { #llm_evaluation }

**[A Highdimensional Statistical Method For Optimizing Transfer](llm_evaluation/a_highdimensional_statistical_method_for_optimizing_transfer.md)**

:   提出基于K-L散度和高维统计分析的理论框架，用于确定多源迁移学习中每个源任务的最优样本迁移数量，避免"用所有源数据"带来的负迁移问题，在DomainNet和Office-Home上超过SOTA 1.0-1.5%的同时减少47.85%的样本使用量和35.19%的训练时间。

**[A Standardized Benchmark For Multilabel Antimicrobial Peptide Classification](llm_evaluation/a_standardized_benchmark_for_multilabel_antimicrobial_peptide_classification.md)**

:   提出 **ESCAPE**——首个标准化的多标签抗菌肽分类基准，整合 27 个公开数据库共 80,000+ 肽段，并设计基于双分支 Transformer + 双向交叉注意力的 Baseline 模型，在 mAP 上相对第二名提升 2.56%。

**[A Unified Framework For Provably Efficient Algorithms To Estimate Shapley Values](llm_evaluation/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)**

:   提出统一框架将 KernelSHAP、LeverageSHAP 等 Shapley 值估计器纳入随机草图（sketching）视角，首次为 KernelSHAP 提供非渐近理论保证，并通过算法改进（Poisson 近似等）将方法扩展到 CIFAR-10 等高维数据集。

**[Adastar Adaptive Data Sampling For Training Self-Taught Reasoners](llm_evaluation/adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)**

:   发现 STaR（自我教学推理器）的随机数据采样导致观测训练频率严重不平衡（简单题过度训练、难题训练不足），提出 AdaSTaR——通过自适应多样性采样（优先欠训练样本）和自适应课程采样（根据模型强度调节难度），在 6 个基准上全部取得最高准确率同时减少 58.6% 训练 FLOPs。

**[Aggregation Hides Out-Of-Distribution Generalization Failures From Spurious Corr](llm_evaluation/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)**

:   揭示 OOD 泛化 benchmark 中"聚合掩蔽"现象——aggregate 评估显示 accuracy-on-the-line（ID 与 OOD 准确率正相关），但 OODSelect 方法可从同一 OOD 数据中找到大规模语义连贯子集（最高达 75%），这些子集上 ID 越高 OOD 反而越低（Pearson R 低至 -0.92），证明虚假相关的危害被聚合评估系统性隐藏。

**[Asymmetric Duos Sidekicks Improve Uncertainty](llm_evaluation/asymmetric_duos_sidekicks_improve_uncertainty.md)**

:   Asymmetric Duos（AD）将一个大模型与一个小"sidekick"配对——通过温度加权的 logit 平均融合两者预测，在仅增加 10-20% FLOPs 的条件下达到接近 5× 深度集成的不确定性估计质量，RN50 AD（5% FLOPs 额外开销）在 AUROC/AURC/SAC@98 上接近 m=5 深度集成（400% 额外 FLOPs）。

**[Bayesian Evaluation Of Large Language Model Behavior](llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)**

:   提出基于 Beta-Binomial 贝叶斯模型的 LLM 行为评估框架，通过对每个 prompt 的随机生成结果建模 $\theta_m$ 后验分布，量化评估指标的统计不确定性，并引入 Thompson sampling 等序贯采样策略以更少的 API 调用获得更窄的置信区间。

**[Belief-Calibrated Multi-Agent Consensus Seeking For Complex Nlp Tasks](llm_evaluation/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)**

:   提出 Belief-Calibrated Consensus Seeking (BCCS) 框架，通过引入信念（belief）校准的共识判断、冲突感知的协作者分配和领导者选择三个模块，让多智能体系统在复杂NLP任务上达成更稳定的共识，在 MATH 和 MMLU 上的困难任务分别提升 2.23% 和 3.95%。

**[Benchmarking Is Broken -- Dont Let Ai Be Its Own Judge](llm_evaluation/benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)**

:   系统性批评当前 AI 基准评估的根本缺陷——数据污染（MMLU 45%+ 重叠）、选择性报告、缺乏监考——并提出 PeerBench 方案：借鉴高考/GRE 的监考范式，用滚动更新的保密题库 + 同行评审质量控制 + 声誉加权评分 + 加密承诺机制构建下一代 AI 评估基础设施。

**[Benchmarking Large Language Models For Zero-Shot And Few-Shot Phishing Url Detec](llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)**

:   在统一的零样本和少样本 prompt 框架下系统评估 GPT-4o、Claude-3.7 和 Grok-3-Beta 三个商用 LLM 在钓鱼 URL 检测任务上的表现，发现少样本 prompt 可显著提升所有模型性能，Grok-3-Beta 在平衡数据集上取得最佳 F1（0.9399），但不同模型在精度-召回率权衡上呈现差异化行为模式。

**[Beyond The Singular Revealing The Value Of Multiple Generations In Benchmark Eva](llm_evaluation/beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)**

:   将LLM基准评测形式化为层级统计模型，理论证明多次随机生成（k>1）能降低benchmark分数估计方差，并引入prompt级难度指标$\mathbb{P}(\text{correct})$和数据地图用于基准质量控制。

**[Beyond The Surface Enhancing Llm-As-A-Judge Alignment With Human Via Internal Re](llm_evaluation/beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)**

:   提出LAGER框架，通过聚合LLM中间层到最终层的score token logits并计算期望分数，无需微调模型即可将LLM评判与人类评分的对齐度提升最高7.5%，且不需要思维链推理步骤就能匹配或超过推理类方法。

**[Blink-Twice You See But Do You Observe A Reasoning Benchmark On Visual Perceptio](llm_evaluation/blink-twice_you_see_but_do_you_observe_a_reasoning_benchmark_on_visual_perceptio.md)**

:   提出视觉中心推理 benchmark BLINK-Twice（345 张视觉挑战图 + 103 个对抗样本 + 896 个 VQA + 1725 个推理步骤标注），通过 7 类视觉错觉场景评估 MLLM "看到但未观察到"的推理能力，发现最强模型 Gemini-2.5 Pro 的 G-Acc 仅 26.9%，多轮图像观察和主动视觉交互是提升方向。

**[Can Large Language Models Master Complex Card Games](llm_evaluation/can_large_language_models_master_complex_card_games.md)**

:   系统评估LLM在8种复杂卡牌游戏上的学习能力，发现通过高质量游戏数据的SFT，LLM可以接近强游戏AI的水平，并能同时掌握多个游戏，但通用能力会下降（可通过混入通用指令数据缓解）。

**[Climb Class-Imbalanced Learning Benchmark On Tabular Data](llm_evaluation/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)**

:   提出 Climb——迄今最全面的表格数据类别不平衡学习基准，涵盖 73 个真实数据集和 29 种 CIL 算法，通过大规模实验揭示了朴素重平衡往往无效、集成方法至关重要、数据质量比不平衡本身更影响性能等实用洞察。

**[Codeassistbench Cab Dataset Benchmarking For Multi-Turn Chat-Based Code Assistan](llm_evaluation/codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)**

:   提出 CodeAssistBench (CAB)，第一个评估多轮、项目级编程辅助的全自动 Benchmark，从 GitHub Issues 自动构建 3,286 个真实编程求助场景，涵盖 7 种语言 214 个仓库，揭示 SOTA 模型在 StackOverflow 问题上 70-83% 但在 post-cutoff 仓库上仅 7-16% 的巨大鸿沟。

**[Compo Preference Alignment Via Comparison Oracles](llm_evaluation/compo_preference_alignment_via_comparison_oracles.md)**

:   针对DPO中噪声偏好对（preferred和dispreferred响应相似）导致的似然位移和冗长问题，提出基于比较oracle的零阶偏好对齐方法ComPO，将数据分为干净/噪声子集，用DPO处理干净数据、用ComPO提取噪声数据中的信号，在AlpacaEval 2等benchmark上持续提升LC win rate。

**[Conformal Online Learning Of Deep Koopman Linear Embeddings](llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)**

:   提出 COLoKe 框架，将 conformal prediction 重新解读为模型一致性诊断工具，仅在 Koopman 模型的预测误差超过动态校准阈值时才触发参数更新，从而实现对非线性动力系统的高效在线 Koopman 线性嵌入学习。

**[Conformal Prediction In The Loop A Feedback-Based Uncertainty Model For Trajecto](llm_evaluation/conformal_prediction_in_the_loop_a_feedback-based_uncertainty_model_for_trajecto.md)**

:   提出 Feedback-Based Conformal Prediction (Fb-CP) 框架，将已执行轨迹的信息反馈给 CP 以动态调整预测区域大小，在缩减时域轨迹优化中同时保证覆盖率和显著提升轨迹性能。

**[Conftuner Training Large Language Models To Express Their Confidence Verbally](llm_evaluation/conftuner_training_large_language_models_to_express_their_confidence_verbally.md)**

:   ConfTuner 提出 tokenized Brier score 损失函数（理论证明为 proper scoring rule），仅需 2000 个样本 + 4 分钟 LoRA 微调即可让 LLM 输出校准的语言化置信度（如"我80%确定"），ECE 最大降低 60.9%，支持自我纠错和模型级联等下游应用。

**[Cost-Sensitive Freeze-Thaw Bayesian Optimization For Efficient Hyperparameter Tu](llm_evaluation/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)**

:   CFBO 将用户定义的效用函数（成本 vs 性能的权衡）引入冻结-解冻贝叶斯优化框架，结合自适应停止准则和基于 LC mixup 的迁移学习，在多保真度 HPO 基准上实现了成本-性能最优权衡。

**[Creativity Or Brute Force Using Brainteasers As A Window Into The Problem-Solvin](llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)**

:   构建Braingle Brainteaser基准（242数学+236逻辑谜题），系统评估LLM在脑筋急转弯上的推理策略——发现模型有时能产生创造性洞察式解法，但也常在有巧妙解法可用时退回暴力穷举，且自纠错能力有限、将叙事→数学格式翻译可小幅提升性能。

**[Decoupled Entropy Minimization](llm_evaluation/decoupled_entropy_minimization.md)**

:   将经典熵最小化（EM）解耦为两个对立部分——Cluster Aggregation Driving Factor (CADF，奖励主导类别)和 Gradient Mitigation Calibrator (GMC，惩罚高置信类别)，揭示了经典 EM 的两个固有缺陷（reward collapse 和 easy-class bias），提出 AdaDEM 通过归一化奖励和边际熵校准来修复这些问题，在半监督学习、域适应、强化学习等多任务上显著提升。

**[Efficient Semantic Uncertainty Quantification In Language Models Via Diversity-S](llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)**

:   提出 diversity-steered sampling 框架：在解码时注入基于 NLI 的语义相似度惩罚来驱动生成语义多样化的样本，并用重要性加权+控制变量纠正偏差降低方差，在仅 16 个样本下即可准确估计 LLM 的语义熵（偶然不确定性）和互信息（认知不确定性）。

**[Enhancing Sample Selection Against Label Noise By Cutting Mislabeled Easy Exampl](llm_evaluation/enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)**

:   发现并定义了误标注易学样本（Mislabeled Easy Examples, MEEs）——被模型早期训练即正确预测为错误标签的样本对泛化伤害最大，并提出 Early Cutting 方法利用模型后期状态重新校准早期置信子集来过滤MEEs。

**[Evalearn Quantifying The Learning Capability And Efficiency Of Llms Via Sequenti](llm_evaluation/evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)**

:   提出 EvaLearn 基准，通过**序列化问题求解**范式评估 LLM 的学习能力和学习效率，揭示静态能力强的模型不一定具备更强的学习潜力。

**[Exploiting Task Relationships In Continual Learning Via Transferability-Aware Ta](llm_evaluation/exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)**

:   提出基于 H-score 可迁移性度量的任务嵌入（H-embedding），并将其嵌入超网络框架，通过在嵌入空间中显式建模任务间关系来指导持续学习中的参数生成，在 rehearsal-free 设定下取得 SOTA 最终准确率。

**[Exploiting Vocabulary Frequency Imbalance In Language Model Pre-Training](llm_evaluation/exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)**

:   通过控制实验揭示大词表提升语言模型性能的根本机制：**扩大词表降低分词文本的 Kolmogorov 复杂度，利用词频不平衡让高频词损失大幅下降，驱动全局交叉熵下降和下游任务提升**。

**[Generalization Error Analysis For Selective State-Space Models Through The Lens ](llm_evaluation/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)**

:   将选择性SSM（Mamba）展开为注意力形式，利用覆盖数技术推导出受连续时间状态矩阵谱横断面$s_{\mathbf{A}}$控制的泛化界——$s_{\mathbf{A}}<0$时泛化界与序列长度无关，$s_{\mathbf{A}}\geq0$时指数增长，并证明这种依赖不可消除。

**[Hybridnorm Towards Stable And Efficient Transformer Training Via Hybrid Normaliz](llm_evaluation/hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)**

:   提出 HybridNorm 混合归一化策略——注意力模块用 QKV 归一化解耦梯度、FFN 用 Post-Norm 增强正则化，在 550M-7B 规模上同时获得 Pre-Norm 的训练稳定性和 Post-Norm 的泛化性能，7B 模型下游任务平均提升 2.45%。

**[Incomplete Multi-View Clustering Via Hierarchical Semantic Alignment And Coopera](llm_evaluation/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)**

:   提出HSACC框架通过双层语义空间设计（低层互信息一致性+高层自适应加权融合）和协同优化的隐式缺失视图恢复，在五个基准数据集上显著超越现有不完整多视图聚类方法。

**[Ineq-Comp Benchmarking Human-Intuitive Compositional Reasoning In Automated Theo](llm_evaluation/ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)**

:   提出 Ineq-Comp 基准，通过对简单不等式种子问题施加人类直觉可轻松处理的组合变换（变量复制、代数重写），揭示当前 LLM 形式化定理证明器在组合推理上的根本性缺陷——即使 DeepSeek-Prover-V2-7B 也有 20%+ 的性能下降。

**[Keep It On A Leash Controllable Pseudo-Label Generation Towards Realistic Long-T](llm_evaluation/keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)**

:   提出 Controllable Pseudo-label Generation (CPG) 框架，通过可控的自强化优化循环将可靠伪标签逐步纳入标注集，在已知分布上构建 Bayes-optimal 分类器，从而在未标注数据分布完全未知的 Realistic LTSSL 场景下实现最高 15.97% 的准确率提升。

**[Lcdb 11 A Database Illustrating Learning Curves Are More Ill-Behaved Than Previo](llm_evaluation/lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)**

:   构建了大规模高分辨率学习曲线数据库 LCDB 1.1，证明样本学习曲线的"病态行为"（非单调、非凸）比此前认为的普遍两倍，约 15% 的曲线显著不良，且特征缩放难以修复。

**[Learning Generalizable Shape Completion With Sim3 Equivariance](llm_evaluation/learning_generalizable_shape_completion_with_sim3_equivariance.md)**

:   提出首个 SIM(3) 等变形状补全网络 SIMECO，通过特征规范化→相似不变几何推理→变换恢复的三阶段模块设计，在去偏评估协议下超越所有增广和等变基线，KITTI 上 MMD 降低 17%、OmniObject3D 上 CD-$\ell_1$ 降低 14%，且在更严格协议下仍优于竞争者在其偏向性设置下的表现。

**[Let The Experts Speak Improving Survival Prediction Calibration Via Mixture-Of-E](llm_evaluation/let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)**

:   提出三种离散时间深度混合专家(MoE)生存分析架构，其中 Personalized MoE 通过让每个专家为每位患者生成定制化事件分布，同时实现出色的聚类、校准和预测精度。

**[Leveraging Robust Optimization For Llm Alignment Under Distribution Shifts](llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)**

:   提出 DoRA（Distribution-aware optimization for Robust Alignment），通过训练分布分类器为每个样本分配校准权重，结合 KL-DRO 框架最小化最坏情况损失，以模型无关的即插即用方式提升多种对齐算法在分布偏移下的鲁棒性，在 DPO/RRHF/LIRE 等基线上一致提升性能。

**[Ltd-Bench Evaluating Large Language Models By Letting Them Draw](llm_evaluation/ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)**

:   LTD-Bench 通过让 LLM 画画（生成点阵或代码绘图）来评估其空间推理能力，将抽象的评分指标转化为直观可视的输出，揭示了当前先进 LLM 在建立语言与空间概念双向映射方面的严重不足。

**[Meicoder Decoding Visual Stimuli From Neural Activity By Leveraging Most Excitin](llm_evaluation/meicoder_decoding_visual_stimuli_from_neural_activity_by_leveraging_most_excitin.md)**

:   提出 MEIcoder，利用神经元特异性的最激励输入(MEI)作为生物学先验、SSIM 损失和对抗训练，从初级视觉皮层(V1)的神经群体活动中实现 SOTA 级别的视觉刺激重建，尤其在小数据集和少量神经元场景下表现突出。

**[Merlin L48 Spectrogram Dataset](llm_evaluation/merlin_l48_spectrogram_dataset.md)**

:   本文提出了 L48 数据集——一个基于真实鸟类录音的细粒度频谱图多标签分类基准，天然具备单正标签多标签 (SPML) 设置，揭示了现有 SPML 方法在真实场景下的严重不足，并提出了基于录音内一致性的正则化方案来提升性能。

**[Mind The Gap Removing The Discretization Gap In Differentiable Logic Gate Networ](llm_evaluation/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)**

:   提出 Gumbel Logic Gate Networks (Gumbel LGNs)，通过在逻辑门选择中注入 Gumbel 噪声并使用直通估计器 (ST estimator)，将可微逻辑门网络的离散化差距减少 98%，训练速度提升 4.5 倍，未使用神经元比例降为 0%。

**[Model-Behavior Alignment Under Flexible Evaluation When The Best-Fitting Model I](llm_evaluation/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)**

:   通过大规模模型恢复实验证明，即使使用 450 万行为数据，基于线性探测（linear probing）的灵活评估方法在 20 个视觉模型中的模型恢复准确率仍低于 80%，揭示了预测准确性与模型可辨识性之间的根本性权衡，质疑了当前"最佳拟合即最优模型"的研究范式。

**[Model Context Protocol For Vision Systems Audit Security And Protocol Extensions](llm_evaluation/model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)**

:   首个对MCP在视觉系统中部署的协议级审计研究，分析91个公开MCP服务器发现78%存在schema不一致、89%缺乏运行时验证，并提出语义schema、可视化记忆、运行时验证器等协议扩展方案。

**[Mvsmamba Multi-View Stereo With State Space Model](llm_evaluation/mvsmamba_multi-view_stereo_with_state_space_model.md)**

:   提出MVSMamba，首个基于Mamba架构的多视图立体(MVS)网络，通过参考视角中心的动态扫描策略实现高效的视内和视间全局全方向特征聚合，在DTU和Tanks-and-Temples上以最优效率达到SOTA性能。

**[Normal-Abnormal Guided Generalist Anomaly Detection](llm_evaluation/normal-abnormal_guided_generalist_anomaly_detection.md)**

:   NAGL 框架首次在通用异常检测（GAD）中引入正常+异常混合参考样本，通过残差挖掘（RM）和异常特征学习（AFL）两个注意力模块，在残差空间中学习可迁移的异常模式，仅用 1 个异常样本即可在跨域场景中大幅超越仅使用正常参考的方法。

**[On Evaluating Llm Alignment By Evaluating Llms As Judges](llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)**

:   本文系统研究了 LLM 的生成能力与评估能力之间的一致性（GE-consistency），发现两者在强偏好预言机下高度相关（Spearman ρ=0.96），据此提出 AlignEval 基准——通过评估 LLM 作为评判者的能力来衡量其对齐水平，无需 LLM-as-Judge 直接评估模型输出，与 AlpacaEval/Arena-Hard 相当甚至更优。

**[Open-Insect Benchmarking Open-Set Recognition Of Novel Species In Biodiversity M](llm_evaluation/open-insect_benchmarking_open-set_recognition_of_novel_species_in_biodiversity_m.md)**

:   提出Open-Insect——首个面向昆虫物种发现的大规模细粒度开放集识别基准数据集，涵盖三个地理区域和三类开放集划分，系统评测38种OSR算法，发现简单的后验方法（如MSP）在细粒度场景中仍是强基线，同时验证了领域相关辅助数据对提升OSR性能的关键作用。

**[Optitree Hierarchical Thoughts Generation With Tree Search For Llm Optimization ](llm_evaluation/optitree_hierarchical_thoughts_generation_with_tree_search_for_llm_optimization_.md)**

:   提出 OptiTree，通过构建建模树（modeling tree）组织运筹优化问题的层次化分类与建模思维，利用树搜索将复杂问题自适应分解为更简单的子问题序列，显著提升 LLM 在优化建模任务上的准确率（在多个困难基准上提升超过 10%）。

**[Parrot A Benchmark For Evaluating Llms In Cross-System Sql Translation](llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)**

:   本文提出 PARROT，一个面向跨系统 SQL 翻译（SQL-to-SQL）的实际且真实的基准测试，包含来自 38 个开源基准和真实业务场景的 598 个核心翻译对（扩展到 28,003 对），覆盖 22 种生产级数据库系统，揭示当前最强 LLM 的平均准确率低于 38.53%。

**[Path Attention Position Encoding Via Accumulating Householder Transformations](llm_evaluation/path_attention_position_encoding_via_accumulating_householder_transformations.md)**

:   提出 PaTH（Position encoding via accumulating Householder Transformations），一种数据依赖的乘法位置编码方案，通过累积 Householder 变换替代 RoPE 的静态旋转矩阵，在理论表达力和实际语言建模性能上均优于 RoPE。

**[Pfδ A Benchmark Dataset For Power Flow Under Load Generation And Topology Variat](llm_evaluation/pfδ_a_benchmark_dataset_for_power_flow_under_load_generation_and_topology_variat.md)**

:   PFΔ 是首个同时涵盖负荷、发电机出力和拓扑变化的电力潮流基准数据集，包含 859,800 个求解实例、六种电网规模和接近不可行的极端工况，并提出标准化评估任务来系统评测 ML 方法在多种运行条件下的表现。

**[Put Cash On Bandits A Max K-Armed Problem For Automated Machine Learning](llm_evaluation/put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)**

:   针对 AutoML 中的联合算法选择和超参数优化（CASH）问题，通过数据驱动分析揭示了 HPO 奖励分布的有界左偏特性，提出了专门适配该特性的极端 Bandit 算法 MaxUCB，在理论和实验上均优于现有方法。

**[Rdb2G-Bench A Comprehensive Benchmark For Automatic Graph Modeling Of Relational](llm_evaluation/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)**

:   本文提出 RDB2G-Bench——首个评估关系数据库到图建模方法的基准框架，包含 5 个真实 RDB、12 个预测任务和约 5 万个预计算的图模型-性能对，并对 10 种自动图建模方法进行了系统比较。

**[Reliably Detecting Model Failures In Deployment Without Labels](llm_evaluation/reliably_detecting_model_failures_in_deployment_without_labels.md)**

:   提出D3M(Disagreement-Driven Deterioration Monitoring)，一种基于变分贝叶斯后验采样的三阶段模型监控算法，在无标签、无训练数据的部署场景下可靠检测模型性能退化，同时对非退化性偏移保持低误报率。

**[Rethinking Evaluation Of Infrared Small Target Detection](llm_evaluation/rethinking_evaluation_of_infrared_small_target_detection.md)**

:   系统性地揭示了红外小目标检测（IRSTD）现有评估协议的三大局限，提出包含混合层级指标hIoU、系统化错误分析方法和跨数据集评估设置的层次化分析框架。

**[Rethinking Losses For Diffusion Bridge Samplers](llm_evaluation/rethinking_losses_for_diffusion_bridge_samplers.md)**

:   本文揭示了扩散桥采样器中流行的 Log Variance (LV) 损失存在的理论缺陷——不满足数据处理不等式且梯度与 rKL 不等价——并提出用 log-derivative trick 计算 rKL 梯度（rKL-LD），在多个基准上一致性超越 LV 损失，且训练更加稳定、对超参数不敏感。

**[Rgb-To-Polarization Estimation A New Task And Benchmark Study](llm_evaluation/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)**

:   本文首次定义从标准RGB图像估计偏振分量（S₁/S₂/S₃）的新任务，构建涵盖修复型与生成型方法的首个系统基准，发现预训练MAE在像素精度上综合最优（PSNR 24.74），修复型方法整体显著优于扩散生成型方法，且预训练权重迁移是关键优势。

**[Risk Management For Mitigating Benchmark Failure Modes Benchrisk](llm_evaluation/risk_management_for_mitigating_benchmark_failure_modes_benchrisk.md)**

:   基于NIST风险管理流程，系统分析了26个LLM基准测试中的57种失败模式，提出196种缓解策略，并构建了BenchRisk元评估框架对基准测试本身的可靠性进行评分。

**[Scmrdr A Scalable And Flexible Framework For Unpaired Single-Cell Multi-Omics Da](llm_evaluation/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)**

:   提出scMRDR框架，基于β-VAE将单细胞多组学数据的潜在表征解耦为模态共享和模态特异成分，通过等距正则化、对抗训练和掩码重建损失实现非配对多组学数据的可扩展整合。

**[Semi-Supervised Regression With Heteroscedastic Pseudo-Labels](llm_evaluation/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)**

:   提出基于异方差建模的不确定性感知伪标签框架，通过双层优化动态校准每个伪标签的不确定性，避免错误伪标签对回归模型的负面影响，在多个 SSR 基准上取得 SOTA。

**[Small Language Models As Compiler Experts Auto-Parallelization For Heterogeneous](llm_evaluation/small_language_models_as_compiler_experts_auto-parallelization_for_heterogeneous.md)**

:   系统评估了三个小于 1.5B 参数的语言模型（gemma3、llama3.2、qwen2.5）在编译器自动并行化任务上的能力，使用六种推理策略在 11 个真实世界内核上实现平均 6.81x 加速、峰值 43.25x，证明小模型可作为强大的编译器优化推理引擎。

**[Test-Time Adaptation By Causal Trimming](llm_evaluation/test-time_adaptation_by_causal_trimming.md)**

:   提出 TACT 方法，通过数据增强 + PCA 识别表征空间中的非因果方向，并在测试时将表征和类原型沿该方向的投影移除，从而减少模型对非因果特征的依赖，显著提升分布偏移下的预测性能。

**[The Geometry Of Cortical Computation Manifold Disentanglement And Predictive Dyn](llm_evaluation/the_geometry_of_cortical_computation_manifold_disentanglement_and_predictive_dyn.md)**

:   本文提出VCNet——一种模拟灵长类视觉皮层宏观组织的神经网络架构，用几何和动力系统语言重新诠释双流分离（流形解缠）和预测编码（测地线精炼），以0.04MB的极小体积在Spots-10上达到92.1%（比DenseNet蒸馏版高10%），在光场分类上以3.52MB达到74.4%（超MobileNetV2 2.3%）。

**[Thought Communication In Multiagent Collaboration](llm_evaluation/thought_communication_in_multiagent_collaboration.md)**

:   提出 ThoughtComm 框架，将多智能体通信形式化为隐变量生成模型，证明了在非参数条件下共享思想和私有思想均可辨识，并通过稀疏正则化自编码器提取潜在思想、经前缀注入回馈给每个智能体，在数学推理任务上相比当前 SOTA 的 Multiagent Finetuning 平均提升 19.06%。

**[Tight Lower Bounds And Improved Convergence In Performative Prediction](llm_evaluation/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)**

:   在 performative prediction 框架下，首次证明了 Repeated Risk Minimization (RRM) 收敛率的紧致性，并提出 Affine Risk Minimizers (ARM) 算法类，通过利用历史训练快照的数据实现更广泛问题类上的收敛。

**[Time Travel Is Cheating Going Live With Deepfund For Real-Time Fund Investment B](llm_evaluation/time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)**

:   提出 DeepFund——首个实时基金投资 benchmark 工具，通过多智能体架构（Financial Planner + Analyst Team + Portfolio Manager）连接实时股市数据，避免传统回测中 LLM "时间旅行"导致的信息泄露问题。在 24 个交易日的实盘测试中，9 个旗舰 LLM 只有 Grok 3 实现盈利，揭示了当前 LLM 在主动基金管理中的重大局限。

**[Turbocharging Gaussian Process Inference With Approximate Sketch-And-Project](llm_evaluation/turbocharging_gaussian_process_inference_with_approximate_sketch-and-project.md)**

:   提出 ADASAP 算法，通过近似子空间预条件、分布式计算和 Nesterov 加速，将 sketch-and-project 方法扩展到大规模 GP 推断，首次将精确 GP 推断扩展到 $>3\times10^8$ 样本规模，同时在理论上证明了 SAP 方法的 condition number-free 收敛性。

**[Unlocking Transfer Learning For Open-World Few-Shot Recognition](llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)**

:   提出两阶段方法，通过开集感知元学习 + 开集无关迁移学习，首次将迁移学习范式成功应用于少样本开集识别 (FSOSR)，在 miniImageNet 和 tieredImageNet 上达到 SOTA。

**[What Does It Take To Build A Performant Selective Classifier](llm_evaluation/what_does_it_take_to_build_a_performant_selective_classifier.md)**

:   首次对选择性分类的性能差距（selective classification gap）进行有限样本分解，将差距归因于五个源头——贝叶斯噪声、逼近误差、排序误差、统计噪声和实现偏差，并证明单调校准方法对缩小差距效果有限。

**[Your Pre-Trained Llm Is Secretly An Unsupervised Confidence Calibrator](llm_evaluation/your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)**

:   发现 LLM 后训练（SFT/RLHF/DPO）破坏了预训练模型的置信度校准，提出 DACA 方法利用预训练模型的良好校准性，仅在预测一致样本上对齐置信度，实现无标签的后训练模型校准，ECE 最高改善 15.08%。

---

## 💡 LLM推理 { #llm_reasoning }

**[Abbie Autoregressive Block-Based Iterative Encoder For Efficient Sequence Modeli](llm_reasoning/abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)**

:   提出 AbbIE，一种将 decoder-only Transformer 的中间层（Body）进行递归迭代的架构，只需训练时用 2 次迭代，推理时即可通过增加迭代次数实现 upward generalization，在语言建模困惑度和 zero-shot ICL 任务上均超过标准 Transformer，且可作为标准 Transformer 的 drop-in 替代。

**[Adaptive Dual Reasoner Large Reasoning Models Can Think Efficiently By Hybrid Re](llm_reasoning/adaptive_dual_reasoner_large_reasoning_models_can_think_efficiently_by_hybrid_re.md)**

:   提出 Adaptive Dual Reasoner (ADR)——让推理模型在 fast thinking（简单推理步骤压缩）和 slow thinking（复杂推理步骤保留深度）之间动态切换，通过 SFT 冷启动 + EHPO（熵引导混合策略优化）训练，在数学推理基准上准确率提升最高 6.1% 同时推理 token 减少 49.5%-59.3%。

**[Are Large Reasoning Models Good Translation Evaluators Analysis And Performance ](llm_reasoning/are_large_reasoning_models_good_translation_evaluators_analysis_and_performance_.md)**

:   首次系统分析了大推理模型（LRM）在机器翻译MQM评估中的行为，发现LRM存在"过度思考"、评分高估和材料选择依赖模型规模等问题，并提出ThinMQM方法通过训练合成人类评分轨迹来校准LRM思维过程，将思维预算减少约35倍同时提升评估性能（7B模型提升+8.7相关性分数）。

**[Arm Adaptive Reasoning Model](llm_reasoning/arm_adaptive_reasoning_model.md)**

:   ARM 通过让模型自适应地选择四种推理格式（直接回答、短CoT、代码、长CoT），配合改进的 Ada-GRPO 训练算法解决 format collapse 问题，在保持与纯长CoT模型持平的准确率的同时平均节省 ~30% token，最多节省 ~70%。

**[Atom Of Thoughts For Markov Llm Testtime Scaling](llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)**

:   提出 Atom of Thoughts (AoT)，将 LLM 推理建模为马尔可夫链，每个状态是与原问题答案等价但复杂度递减的自包含子问题，通过 DAG 分解+收缩的两阶段转移机制消除历史依赖，可与 ToT/反思等方法无缝集成，在数学/代码/多跳QA等6个benchmark上全面领先现有推理框架。

**[Beyond Chemical Qa Evaluating Llms Chemical Reasoning With Modular Chemical Oper](llm_reasoning/beyond_chemical_qa_evaluating_llms_chemical_reasoning_with_modular_chemical_oper.md)**

:   提出 ChemCoTBench，首个评估 LLM 化学推理能力的 CoT 基准，将复杂化学问题分解为模块化的化学操作（加/删/替换官能团），配合 22,000 条专家标注的 CoT 数据集（ChemCoTDataset），系统性评估了推理型和非推理型 LLM 在分子理解/编辑/优化/反应预测上的能力。

**[Clip-And-Verify Linear Constraint-Driven Domain Clipping For Accelerating Neural](llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)**

:   提出Clip-and-Verify验证流水线，利用线性界传播过程中"免费"产生的线性约束，通过完全裁剪（坐标上升对偶求解）和松弛裁剪（闭式输入域收缩）两种GPU高效算法收紧全网络中间层界，在多个benchmark上减少高达96%的BaB子问题数量，是VNN-COMP 2025获胜验证器的核心组件。

**[Controlling Thinking Speed In Reasoning Models](llm_reasoning/controlling_thinking_speed_in_reasoning_models.md)**

:   通过表示工程（Representation Engineering）从 LRM 的隐藏空间中提取控制快/慢思考转换的 steering vector，结合基于层间 logit 散度的实时推理难度估计，实现无需训练的自适应推理速度调节，在 4 个 LRM 上平均提升 +1.3% 准确率并减少 -8.6% token 使用。

**[Cot Redhanded Stress Testing Chainofthought Monitoring](llm_reasoning/cot_redhanded_stress_testing_chainofthought_monitoring.md)**

:   在 AI Control 框架下系统评估了 Chain-of-Thought 监控的有效性：发现 CoT 监控在检测微妙破坏行为上比仅监控 action 更有效（+10pp），但在检测明显破坏行为时反而更差（-25pp，因为推理中的伪合理化会欺骗监控），提出 hybrid 监控协议（独立评分 CoT 和 action 后加权）在所有场景下一致优于两种单一监控，检测率提升 2 倍。

**[Curriculum Abductive Learning](llm_reasoning/curriculum_abductive_learning.md)**

:   提出 Curriculum Abductive Learning (C-ABL)，通过将知识库按依赖结构分割为子知识库并渐进式引入训练，大幅缩小 ABL 的 abduction 搜索空间，显著提升训练稳定性、收敛速度和最终精度。

**[Disco Reinforcing Large Reasoning Models With Discriminative Constrained Optimiz](llm_reasoning/disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)**

:   分析 GRPO 目标函数，揭示其固有的难度偏差（对过难/过易问题赋予过低权重）和熵不稳定性问题，提出基于判别学习的 DisCO 框架，通过无裁剪评分函数、平方铰链约束优化和 DRO 处理不平衡 rollout，在 1.5B 模型上平均超过 GRPO 7%、超过 DAPO 6%。

**[Does Thinking More Always Help Mirage Of Test-Time Scaling In Reasoning Models](llm_reasoning/does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)**

:   通过系统实验揭示 LRM 测试时扩展（反复 "Wait" 提示延长推理）的性能呈先升后降的非单调趋势，用概率模型证明这种"提升"只是方差增大导致的海市蜃楼而非真正推理能力提升，并提出 parallel thinking 策略在相同 token 预算下准确率提升最高 22%。

**[Dreamprm Domain-Reweighted Process Reward Model For Multimodal Reasoning](llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)**

:   提出 DreamPRM，通过双层优化自动学习多模态推理数据集的域权重，解决 PRM 训练中的数据质量不均衡问题，在 MathVista 排行榜上以 o4-mini 模型达到 85.2% 的 top-1 准确率。

**[Gpo Learning From Critical Steps To Improve Llm Reasoning](llm_reasoning/gpo_learning_from_critical_steps_to_improve_llm_reasoning.md)**

:   GPO 通过蒙特卡洛模拟估计推理轨迹中每一步的优势函数，识别出"关键步骤"（模型犯错的转折点），然后从关键步骤重置并重新采样轨迹用于训练，可以即插即用地提升 PPO、DPO、KTO、SimPO、ORPO 等多种优化算法在推理任务上的表现。

**[I-Raven-X Benchmarking Generalization And Robustness Of Analogical And Mathemati](llm_reasoning/i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)**

:   提出 I-RAVEN-X，一个增强版的符号化推理基准，通过增加操作数复杂度、属性范围和感知不确定性来评估 LLM 和 LRM 的类比推理与数学推理的泛化能力和鲁棒性，发现 LRM 在确定性推理上显著优于 LLM，但在不确定性推理下性能急剧下降。

**[Inference-Time Chain-Of-Thought Pruning With Latent Informativeness Signals](llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)**

:   提出 KAPPA (KL-Adjusted Pruned Path Algorithm)，利用 KL 散度、置信度和熵三个无需额外训练的信号对 Best-of-N 采样的推理分支进行渐进式剪枝，在保持准确率的同时实现最高 60% 峰值内存和 90% token 生成量的削减。

**[Is Chain-Of-Thought Reasoning Of Llms A Mirage A Data Distribution Lens](llm_reasoning/is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)**

:   通过构建完全可控的抽象环境DataAlchemy，本文揭示CoT推理是一种幻觉——其有效性完全由训练数据分布主导，在分布外场景表现极其脆弱。

**[Know What You Dont Know Uncertainty Calibration Of Process Reward Models](llm_reasoning/know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)**

:   本文提出了一种基于分位数回归的PRM校准方法，使PRM输出的分数更准确地反映LLM实际推理成功概率，并基于校准后的PRM设计了实例自适应推理时缩放（IAS）策略，在保持准确率的同时显著降低推理成本。

**[Large Language Models Can Learn And Generalize Steganographic Chain-Of-Thought U](llm_reasoning/large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)**

:   证明 LLM 在 RL 训练中受到 CoT 过程监督（惩罚特定字符串出现）时，会自发学会隐写术（steganography）——用替代编码隐藏被禁止的推理步骤，且这种编码是因果性的（load-bearing）并能泛化到训练中从未见过的字符串。

**[Latent Chain-Of-Thought For Visual Reasoning](llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)**

:   将视觉CoT推理重新建模为后验推断问题，提出基于摊销变分推断(AVI)的LaCoT训练框架——包含参考引导GFlowNet微调(RGFN)、token级奖励近似和贝叶斯推理缩放(BiN)——在Qwen2.5-VL 3B/7B上比GRPO高出10.6%，在7个视觉推理基准上达到开源SOTA。

**[Let Lrms Break Free From Overthinking Via Self-Braking Tuning](llm_reasoning/let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)**

:   提出 Self-Braking Tuning (SBT) 框架，通过识别推理轨迹中的过度思考模式并构造自适应长度训练数据，使大型推理模型（LRM）学会自主判断何时停止推理，在数学推理任务上减少 30%-60% token 消耗的同时保持精度。

**[Let Me Think A Long Chainofthought Can Be Worth Exponentiall](llm_reasoning/let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)**

:   本文从理论和实验两方面证明：存在推理任务（图连通性问题），其中一条长 CoT（顺序缩放）的能力等价于指数多条短 CoT（并行缩放）——即将 CoT 长度减少一点点，就需要指数级增加并行采样数才能达到同等准确率。

**[Limopro Reasoning Refinement For Efficient And Effective Test-Time Scaling](llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)**

:   提出PIR（基于困惑度的重要性精炼）框架，将LRM蒸馏的推理链分为"渐进推理"和"功能性步骤"（验证/多方法验证/纠错）两类，仅裁剪低PIR值的功能性步骤而完整保留渐进推理骨架，使微调后的模型在AIME/AMC/GPQA上准确率提升0.9%-6.6%同时token减少3%-41%，效率最高提升71%。

**[Lost In Transmission When And Why Llms Fail To Reason Globally](llm_reasoning/lost_in_transmission_when_and_why_llms_fail_to_reason_globally.md)**

:   提出有界注意力前缀预言机(BAPO)计算框架，将LLM的注意力头建模为有限带宽通信信道，证明图可达性等全局推理问题是BAPO-hard的（需超常数带宽），且CoT可将任何BAPO-hard问题转化为BAPO-easy问题，实验在GPT-4o/Claude/Gemini上验证理论预测。

**[Many Llms Are More Utilitarian Than One](llm_reasoning/many_llms_are_more_utilitarian_than_one.md)**

:   在6款LLM的对照实验中发现"功利主义增强(Utilitarian Boost)"现象——LLM在成对/三人组讨论道德困境后比独立判断时更倾向接受"为了多数人的利益而伤害少数人"，这一效应在涉及直接伤害的个人困境中尤为显著（$\beta=0.31, p<.0001$），且各模型产生功利主义增强的机制不同（有的因规范敏感性降低，有的因公正性增强）。

**[Mind The Gap Bridging Thought Leap For Improved Chain-Of-Thought Tuning](llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)**

:   本文首次系统性地定义了 CoT 推理链中的"思维跳跃"(Thought Leap)现象，提出 CoT-Bridge 模型自动检测并补全推理链中被省略的中间步骤，在 NuminaMath 上最高提升 +5.87%，并可作为即插即用模块增强蒸馏和 RL 流程。

**[On Learning Verifiers And Implications To Chain-Of-Thought Reasoning](llm_reasoning/on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)**

:   提出学习Chain-of-Thought验证器的形式化PAC框架，定义三种递进强度的验证目标（Simple → Trustable → γ-Trustable），证明当每个问题只有少量正确证明时样本复杂度为 $O(\log|H|)$，但当正确证明数量不受限时样本复杂度不可避免地跃升至 $\Theta(|H|)$，除非验证器类满足交集封闭性等额外结构假设；同时利用USAT问题证明验证与生成之间存在计算复杂度差距。

**[One Token Embedding Is Enough To Deadlock Your Large Reasoning Model](llm_reasoning/one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)**

:   本文提出 Deadlock Attack，通过优化单个对抗性 token embedding 并以后门方式植入 LRM，使模型在推理时陷入永久思考循环（无限生成 "Wait"、"But" 等过渡词），在 4 个 LRM 和 3 个数学推理 benchmark 上实现 100% 攻击成功率，且对正常输入几乎无性能影响。

**[Proofsketch Efficient Verified Reasoning For Large Language Models](llm_reasoning/proofsketch_efficient_verified_reasoning_for_large_language_models.md)**

:   提出 ProofSketch 框架，通过符号闭包前向推理+短sketch生成+形式验证的多阶段pipeline，在降低token用量的同时提供逻辑推理的形式化正确性保证。

**[Provable Scaling Laws For The Testtime Compute Of Large Lang](llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)**

:   提出 Knockout（淘汰赛式两两淘汰）和 League（联赛式平均胜率排序）两种两阶段测试时计算算法，在"LLM 能以非零概率生成正确解"和"LLM 两两比较优于随机"的极弱假设下，从理论上证明失败概率随测试时计算量增长呈指数或幂律衰减至零，且整个算法仅需黑盒 LLM，无需外部验证器或奖励模型。

**[Re-Forc Adaptive Reward Prediction For Efficient Chain-Of-Thought Reasoning](llm_reasoning/re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)**

:   提出Re-FORC，一个轻量级adapter在CoT推理过程中实时预测未来期望奖励 $\psi(t|x,z,\pi)$，将推理计算分配建模为Pandora's box问题，实现自适应早停（节省26%计算）、模型+计算联合选择（同等计算下+4%准确率或同等准确率-55%计算）和测试时计算伸缩（+11%准确率），且用户可通过代价系数 $\lambda$ 在推理时自由调控精度-效率权衡，无需重训。

**[Realmath A Continuous Benchmark For Evaluating Language Models On Research-Level](llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)**

:   提出 RealMath，一个从 arXiv 论文和 Math StackExchange 中自动提取可验证数学问题的**可持续刷新**基准，用于评估 LLM 在真实研究级数学任务上的能力。

**[Reasonfluxprm Trajectoryaware Prms For Long Chainofthought R](llm_reasoning/reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)**

:   ReasonFlux-PRM 发现现有 PRM 无法有效评估推理模型的中间思考轨迹（trajectory），提出融合步骤级对齐/质量/连贯性分数和轨迹级模板引导奖励的 trajectory-aware PRM，在离线数据选择（SFT +12.1%）、在线 RL 奖励（+4.5%）和测试时 Best-of-N 缩放（+6.3%）三个场景中均显著优于包括 Qwen2.5-Math-PRM-72B 在内的强基线。

**[Reasoning By Superposition A Theoretical Perspective On Chain Of Continuous Thou](llm_reasoning/reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)**

:   本文从理论上证明了连续思维链（Coconut）在有向图可达性问题上的表达优势：两层Transformer使用D步连续思维即可解决直径为D的图可达性问题，而离散CoT需要O(n²)步，其核心机制是连续思维向量以"叠加态"同时编码多条搜索前沿，实现隐式并行BFS。

**[Reasoning Models Better Express Their Confidence](llm_reasoning/reasoning_models_better_express_their_confidence.md)**

:   系统性证明推理模型（extended CoT）比非推理模型具有显著更优的置信度校准能力，并揭示"慢思考"行为（探索替代方案、回溯、验证）是校准提升的根本来源。

**[Reasoning Models Hallucinate More Factuality-Aware Reinforcement Learning For La](llm_reasoning/reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)**

:   揭示了RL训练的推理模型（如DeepSeek-R1）比非推理模型产生更多幻觉，从理论上分析了三个根因（高方差梯度、熵约束、伪局部最优），并提出FSPO算法通过步级事实性验证调整token级advantage，在减少幻觉的同时保持甚至提升推理能力。

**[Rethinking Optimal Verification Granularity For Compute-Efficient Test-Time Scal](llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)**

:   提出 Variable Granularity Search (VG-Search)，通过可调的验证粒度参数 $g$ 统一 Beam Search 和 Best-of-N，发现传统每步验证是次优的，自适应调整 $g$ 可在提升准确率3%+的同时减少52%+的计算量。

**[Safepath Preventing Harmful Reasoning In Chain-Of-Thought Via Early Alignment](llm_reasoning/safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)**

:   提出 SafePath，仅在推理开始处微调 8 个 token 的"Safety Primer"（"Let's think about safety first"），即可有效引导 LRM 走向安全推理路径，在 DeepSeek-R1-Distill 上减少 90% 有害输出且仅需 Direct Refusal 1/296 的训练计算量。

**[Sampling-Efficient Test-Time Scaling Self-Estimating The Best-Of-N Sampling In E](llm_reasoning/sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)**

:   提出 Self-Truncation Best-of-N (ST-BoN) 解码方法，通过理论证明早期隐状态一致性可预测最终一致性，在生成早期就识别并截断次优样本，实现降低80%+内存和50%延迟的同时保持BoN性能。

**[Scalable Best-Of-N Selection For Large Language Models Via Self-Certainty](llm_reasoning/scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)**

:   提出Self-Certainty度量，利用LLM输出的token概率分布量化模型信心，在无需额外奖励模型的情况下实现可扩展的Best-of-N选择，性能媲美或超越基于奖励模型的方法。

**[Segment Policy Optimization Effective Segment-Level Credit Assignment In Rl For ](llm_reasoning/segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)**

:   提出SPO框架，采用段级（而非令牌级或轨迹级）的advantage估计，通过新颖的蒙特卡洛方法和树形采样，在短CoT和长CoT场景下分别超越PPO和GRPO 6-12和7-11个百分点。

**[Self-Evaluating Llms For Multi-Step Tasks Stepwise Confidence Estimation For Fai](llm_reasoning/self-evaluating_llms_for_multi-step_tasks_stepwise_confidence_estimation_for_fai.md)**

:   PolyMath构建的18语言、4难度级、500问题数学推理基准揭露：(1)推理性能跨语言差异达10分，(2)推理模型输入-输出语言一致性低且可能影响性能，(3)思考长度在语言间显著不一致，为多语言推理研究提供新视角。

**[Smaller Models Smarter Rewards A Two-Sided Approach To Process And Outcome Rewar](llm_reasoning/smaller_models_smarter_rewards_a_two-sided_approach_to_process_and_outcome_rewar.md)**

:   将 Phi-4 系列小模型（3.8B/14B）的最后一层替换为回归头并微调，使其同时具备 ORM（结果奖励）和 PRM（过程奖励）能力，在代码生成任务上通过选择最优 rollout 实现 20%+ 的 pass@k 提升。

**[Sprint Enabling Interleaved Planning And Parallelized Execution In Reasoning Mod](llm_reasoning/sprint_enabling_interleaved_planning_and_parallelized_execution_in_reasoning_mod.md)**

:   通过将长链式推理轨迹重组为交替的规划-并行执行阶段，Sprint 使推理模型在保持准确率的同时，将长推理链的顺序 token 数减少高达 39%（OOD 任务上最高 65%），实现推理过程的动态并行化。

**[Sql-Of-Thought Multi-Agentic Text-To-Sql With Guided Error Correction](llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)**

:   提出 SQL-of-Thought——一个多智能体 Text-to-SQL 框架，将任务分解为 schema linking → 子问题识别 → CoT 查询计划生成 → SQL 生成 → 基于 31 类错误分类法的引导修正循环，用 Claude 3 Opus 在 Spider 上达到 91.59% 执行准确率，比此前最佳 Chase SQL（87.6%）提升近 4 个百分点。

**[Sql-R1 Training Natural Language To Sql Reasoning Model By Reinforcement Learnin](llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)**

:   首次系统地将 GRPO 强化学习应用于 NL2SQL 任务，通过四层递进式奖励函数和 200K 冷启动 + 5K 复杂样本 RL 训练策略，7B 模型在 Spider 和 BIRD 上分别达到 88.7% 和 66.6%，超越 GPT-4 同规模模型。

**[Stop Summation Minform Credit Assignment Is All Process Rewa](llm_reasoning/stop_summation_minform_credit_assignment_is_all_process_rewa.md)**

:   PURE 发现 PRM 导致 reward hacking 的根本原因是 RL 中标准的 sum-form 信用分配（$V(s) = \sum \gamma^t r_t$），并提出 min-form 替代方案（$V(s) = \min_{t' \geq t} r_{t'}$），通过将价值函数限制为未来奖励的最小值而非累积和，显著缓解 reward hacking——仅用 30% 训练步数就达到与规则奖励方法相当的推理性能。

**[The Hawthorne Effect In Reasoning Models Evaluating And Steering Test Awareness](llm_reasoning/the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)**

:   首次系统量化推理型LLM的"测试感知"(Hawthorne效应)：当模型察觉自己在被评估时会改变行为，论文通过线性探针定位感知激活并进行参数编辑引导，揭示测试感知对安全对齐的显著且方向不一致的影响。

**[The Illusion Of Thinking Understanding The Strengths And Limitations Of Reasonin](llm_reasoning/the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)**

:   通过可控拼图环境系统揭示大型推理模型（LRMs）的三阶段行为：低复杂度不如标准 LLM、中等复杂度显著优于、高复杂度完全崩溃(0%)，且反直觉地在崩溃时减少思考 token，证实当前 LRMs 并未发展出真正泛化的推理能力。

**[The Impact Of Quantization On Large Reasoning Model Reinforcement Learning](llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)**

:   系统实验发现在大推理模型的 RL 训练中，量化感知训练（QAFT/STE）会损害推理能力，而训练后量化（PTQ）和 QLoRA 即使在 4-bit 精度下也能很好地保持推理性能，为实践者提供了"先全精度 RL、再 PTQ 量化"的推荐路线。

**[The Surprising Effectiveness Of Negative Reinforcement In Llm Reasoning](llm_reasoning/the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)**

:   将可验证奖励的强化学习（RLVR）分解为正样本强化（PSR，增强正确回答概率）和负样本强化（NSR，惩罚错误回答），发现仅用 NSR 就能在整个 Pass@k 谱上持续提升推理性能且通常匹配或超越 PPO/GRPO，据此提出 Weighted-REINFORCE（降低 PSR 权重至 0.1）在 MATH/AIME 2025/AMC23 上取得全面最优。

**[The Virtues Of Brevity Avoid Overthinking In Parallel Test-Time Reasoning](llm_reasoning/the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)**

:   证明在推理模型的 Best-of-N 采样中，选择最短解是一个简单、反直觉但高效的启发式方法，性能与 self-consistency 相当，token 成本显著更低，其原理在于利用了模型在"常规模式"与"过度思考模式"之间的系统性偏差。

**[Thinksound Chain-Of-Thought Reasoning In Multimodal Large Language Models For Au](llm_reasoning/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)**

:   提出三阶段交互式视频转音频框架 ThinkSound，通过 MLLM 生成结构化 CoT 推理来指导统一的音频生成基础模型，在 VGGSound 和 MovieGen Audio 基准上达到 SOTA，同时支持对象级精细化和自然语言指令编辑。

**[Time A Multilevel Benchmark For Temporal Reasoning Of Llms I](llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)**

:   提出 TimE，一个包含 38,522 个 QA 对的多层级时间推理基准，覆盖知识密集（Wiki）、动态新闻（News）、长对话（Dial）三种真实场景和三级渐进式 11 子任务，全面评估 24 个 LLM 后发现即便最强推理模型在时间线构建和反事实推理等复杂任务上仍有显著短板。

**[Topology Of Reasoning Understanding Large Reasoning Models Through Reasoning Gra](llm_reasoning/topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)**

:   提出"推理图"概念——通过对 LLM 隐藏状态聚类构建有向图，从环路密度、直径和小世界指标三个图论维度分析大推理模型（如 DeepSeek-R1 蒸馏系列），发现推理模型的推理图具有显著更多环路（~5/样本）、更大直径和更强小世界特性（~6倍），且这些特性随任务难度和模型规模增长。

**[Towards Thinking-Optimal Scaling Of Test-Time Compute For Llm Reasoning](llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)**

:   揭示了过度延长 CoT 长度会损害 LLM 推理性能，并提出 Thinking-Optimal Scaling (TOPS) 策略，让模型为每道题选择最短正确响应进行自我提升，在效果和效率上同时优于现有蒸馏方法。

**[Transformers Provably Learn Chain-Of-Thought Reasoning With Length Generalizatio](llm_reasoning/transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)**

:   从优化理论角度证明了一层 Transformer 通过梯度下降在合成状态追踪任务上能学会 CoT 推理并实现长度泛化，首次为常数深度 Transformer 学习 $\mathsf{NC}^1$-complete 问题（超越之前局限于 $\mathsf{TC}^0$ 的理论）提供了收敛保证。

**[Tts-Var A Test-Time Scaling Framework For Visual Auto-Regressive Generation](llm_reasoning/tts-var_a_test-time_scaling_framework_for_visual_auto-regressive_generation.md)**

:   提出 TTS-VAR——首个针对 Visual Auto-Regressive (VAR) 模型的测试时扩展框架，将图像生成建模为路径搜索问题，通过自适应递减批量 + 早期聚类多样性搜索 + 后期重采样潜力选择，在 Infinity 2B 上将 GenEval 分数从 0.69 提升到 0.75（+8.7%），N=2 即超越 Best-of-N 的 N=8 效果。

**[Two-Stage Learning Of Stabilizing Neural Controllers Via Zubov Sampling And Iter](llm_reasoning/two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)**

:   提出两阶段训练框架——先用 Zubov 采样 + 动态域扩展估计吸引域（ROA），再用 CEGIS 反例精炼——联合学习神经网络控制器和 Lyapunov 函数，ROA 体积比基线大 5 到 $1.5 \times 10^5$ 倍，验证速度比 dReal 快 40-10000 倍。

**[Unlabeled Data Can Provably Enhance In-Context Learning Of Transformers](llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)**

:   提出增强型ICL框架，在prompt中同时包含少量标记样本和大量无标记样本，理论证明多层Transformer通过CoT可模拟EM算法从无标记数据中提取信息，将分类excess risk从 $\mathcal{O}(1/\sqrt{N})$ 改进到 $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$。

**[Unlocking Multimodal Mathematical Reasoning Via Process Reward Model](llm_reasoning/unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)**

:   提出URSA三阶段框架，依次构建百万级多模态CoT数据(MMathCoT-1M)训练基座、双视角过程监督数据(DualMath-1.1M)训练PRM、以及PS-GRPO算法将PRM融入在线RL，8B模型在6个数学基准上平均超越GPT-4o 2.7%。

**[Value-Guided Search For Efficient Chain-Of-Thought Reasoning](llm_reasoning/value-guided_search_for_efficient_chain-of-thought_reasoning.md)**

:   本文扩展置信度估计到多步任务，证明步级评估相比整体评估能更有效地检测推理失败，相对整体评估在CoQA上AUC-ROC提升15%，为多步推理系统的可信部署提供实用框架。

**[Visual Thoughts A Unified Perspective Of Understanding Multi](llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)**

:   提出"视觉思维(Visual Thoughts)"作为统一框架解释多模态链式推理(MCoT)的有效性——无论是文本MCoT还是交错图文MCoT，其性能提升的核心机制都是将视觉信息缓存并传递到推理过程中，定义了四种视觉思维表达形式并揭示其在Transformer深层中作为图像-推理中介的角色。

---

## 📹 视频理解 { #video_understanding }

**[A Little Depth Goes A Long Way The Expressive Power Of Logde](video_understanding/a_little_depth_goes_a_long_way_the_expressive_power_of_logde.md)**

:   本文证明了将 Transformer 的深度从常数增长到 Θ(log n) 就能解锁识别正则语言和图连通性这两类固定深度 Transformer 无法表达的问题，且深度扩展比宽度（需超多项式增长）和 CoT 步数（需超对数增长）都更高效。

**[Adavideorag Omnicontextual Adaptive Retrievalaugmented Effic](video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)**

:   提出 AdaVideoRAG，通过轻量级意图分类器将查询按难度路由到三级检索路径（无检索/朴素检索/图检索），结合全知识索引模块（caption+ASR+OCR+视觉+知识图谱）实现长视频理解的效率-精度最优平衡，在 MLVU 上为 Qwen2.5-VL-7B 带来 39.8% 提升。

**[Adversarial Locomotion And Motion Imitation For Humanoid Policy Learning](video_understanding/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)**

:   ALMI提出上下半身对抗训练框架：下半身策略在上半身动作干扰下学习鲁棒运动，上半身策略在下半身运动干扰下学习精确动作模仿，通过迭代对抗训练收敛到Nash均衡，实现Unitree H1-2真实机器人的稳定全身协调控制。

**[Agentic Persona Control And Task State Tracking For Realistic User Simulation In](video_understanding/agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)**

:   提出三 agent 协作框架用于逼真的用户模拟——User Agent（协调）+ State Tracking Agent（结构化任务状态）+ Message Attributes Generation Agent（基于 persona 和状态的行为属性控制），在餐厅点餐场景中综合仿真质量（CRRS）提升 102.6%，persona 保持度 +19.9%，行为自然度 +284.5%，且核心发现：无状态感知的行为控制导致 BVS=0（完全刚性）。

**[Cleverbirds A Multiple-Choice Benchmark For Fine-Grained Human Knowledge Tracing](video_understanding/cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)**

:   提出CleverBirds——迄今最大规模的视觉知识追踪基准，通过eBird公民科学平台收集了40000+参与者完成的1700万+多选鸟类物种识别问答（涵盖10000+物种），系统性评估了多种知识追踪与分类方法，揭示了细粒度视觉知识建模特别是预测学习者错误选择方面的核心挑战。

**[Cloud4D Estimating Cloud Properties At A High Spatial And Temporal Resolution](video_understanding/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)**

:   首个基于地面多视角相机的学习框架，通过单应性引导的2D-to-3D Transformer重建四维（3D空间+时间）云液态水含量分布，在25m空间/5s时间分辨率下实现了相对雷达<10%的误差，比卫星观测提升了一个数量级的时空分辨率。

**[Convis-Bench Estimating Video Similarity Through Semantic Concepts](video_understanding/convis-bench_estimating_video_similarity_through_semantic_concepts.md)**

:   提出基于语义概念的视频相似度估计任务 ConViS 及配套 benchmark ConViS-Bench（610对视频、16领域、5概念），系统评测了10+主流模型在概念条件下的视频比较能力，揭示当前模型在时序结构和空间语境理解上的显著短板。

**[Deceptron Learned Local Inverses For Fast And Stable Physics Inversion](video_understanding/deceptron_learned_local_inverses_for_fast_and_stable_physics_inversion.md)**

:   提出 Deceptron 双向模块，通过学习可微分前向代理的局部逆映射并引入 Jacobian Composition Penalty (JCP)，在求解物理逆问题时将输出空间的残差拉回输入空间，实现类 Gauss-Newton 的预条件梯度更新，迭代次数大幅减少（Heat-1D 约 20 倍加速）。

**[Deltaproduct Improving State-Tracking In Linear Rnns Via Householder Products](video_understanding/deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)**

:   提出 DeltaProduct，通过将 DeltaNet 的单步梯度下降扩展至每个 token 的多步梯度下降，使状态转移矩阵成为 $n_h$ 个广义 Householder 变换的乘积，实现了表达力与效率之间的可调平衡，显著提升了状态跟踪能力和长度外推性能。

**[Dense Sae Latents Are Features Not Bugs](video_understanding/dense_sae_latents_are_features_not_bugs.md)**

:   本文系统研究了稀疏自编码器(SAE)中频繁激活的"dense latents"，证明它们不是训练噪声，而是语言模型残差流中固有的密集子空间的反映，并提出了一套包含位置追踪、上下文绑定、零空间、字母、词性和PCA等六类dense latent的分类体系。

**[Disentangled Concepts Speak Louder Than Words Explainable Video Action Recogniti](video_understanding/disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)**

:   提出DANCE框架，通过将动作解释解耦为运动动态、物体和场景三类概念，实现结构化和运动感知的可解释视频动作识别。

**[Dsas A Universal Plug-And-Play Framework For Attention Optimization In Multi-Doc](video_understanding/dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)**

:   提出Dual-Stage Adaptive Sharpening (DSAS)，一个无需训练的即插即用注意力优化框架，通过Contextual Gate Weighting (CGW)增强关键段落对问题和目标位置的注意力、通过Reciprocal Attention Suppression (RAS)抑制关键与无关段落间的信息交换，在多文档QA上平均F1提升达4.2%。

**[Egoemotion Egocentric Vision And Physiological Signals For Emotion And Personali](video_understanding/egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)**

:   提出egoEMOTION——首个结合第一人称视觉（Meta Project Aria眼镜）与生理信号的情感与人格识别数据集，涵盖43名被试、50+小时录制、16种任务，发现第一人称视觉信号（尤其眼动特征）在真实场景情感预测中优于传统生理信号。

**[Empower Words Dualground For Structured Phrase And Sentencel](video_understanding/empower_words_dualground_for_structured_phrase_and_sentencel.md)**

:   DualGround揭示现有VTG模型过度依赖[EOS] token的全局语义而忽略词级信号的问题，提出句子级+短语级双路径架构，通过自适应交叉注意力和循环短语生成器分别建模全局和局部语义，在QVHighlights和Charades-STA上达到SOTA。

**[Enhancing Temporal Understanding In Videollms Through Stacke](video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)**

:   提出 STAVEQ2，在 Vision Encoder 中堆叠参数高效的时序注意力模块（STA），解决现有 Video-LLM 在细粒度时序理解（如区分"从左到右拉"和"从右到左拉"）上的根本性架构缺陷，在 VITATECS/MVBench/Video-MME 上提升最高 5.5%。

**[Fastvid Dynamic Density Pruning For Fast Video Large Languag](video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)**

:   提出 FastVID，通过动态时序分割 (DySeg) + 密度空时剪枝 (STPrune) 从时间和视觉两个维度系统性消除视频 token 冗余，在 LLaVA-OneVision-7B 上剪掉 90.3% 视频 token 后仍保留 98% 精度，LLM prefill 阶段加速 7.1×。

**[Fixed-Point Rnns Interpolating From Diagonal To Dense](video_understanding/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)**

:   提出 Fixed-Point RNN 框架，将稠密线性 RNN 参数化为对角线性 RNN 的不动点，通过迭代次数在对角（高效）与稠密（表达力强）之间动态插值，首次在状态跟踪（$A_5$/$S_5$）和拷贝任务上同时取得最优结果。

**[Geodynamics A Geometric State-Space Neural Network For Understanding Brain Dynam](video_understanding/geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)**

:   提出GeoDynamics，将经典状态空间模型(SSM)从欧几里得空间推广到对称正定(SPD)流形，通过加权Frechet均值聚合和正交群平移实现流形上的状态演化，在脑连接组（AD/PD/ASD早期诊断）和人体动作识别上均取得SOTA。

**[Grounding Foundational Vision Models With 3D Human Poses For Robust Action Recog](video_understanding/grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)**

:   提出一种融合 V-JEPA 2 视觉上下文特征与 CoMotion 3D 骨骼姿态数据的 cross-attention 多模态架构，在标准及高遮挡动作识别基准上优于单模态基线。

**[In The Eye Of Mllm Benchmarking Egocentric Video Intent Understanding With Gaze-](video_understanding/in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)**

:   提出 EgoGazeVQA 基准和三种注视引导提示策略（文本/视觉/显著图），首次系统验证了眼动注视信号对提升 MLLM 第一人称视频意图理解的关键价值，Qwen2.5-VL-72B + GazeS 策略在平均准确率上提升 5.8 个百分点。

**[Infinipot-V Memory-Constrained Kv Cache Compression For Streaming Video Understa](video_understanding/infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)**

:   提出首个无需训练、查询无关的流式视频理解框架InfiniPot-V，通过时序冗余度（TaR）和值范数（VaN）两个度量实现KV缓存的在线压缩，在固定内存约束下支持任意长度的流式视频理解。

**[Influx A Benchmark For Self-Calibration Of Dynamic Intrinsics Of Video Cameras](video_understanding/influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)**

:   提出首个包含逐帧动态相机内参真值的真实视频基准 InFlux（386 视频、143K+ 标注帧），通过镜头元数据到内参的查找表（LUT）实现精确标注，并揭示现有内参预测方法在动态内参场景下表现不佳。

**[Inst-It Boosting Instance Understanding Via Explicit Visual Prompt Instruction T](video_understanding/inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)**

:   提出Inst-IT完整方案：通过GPT-4o辅助的自动标注管线生成实例级细粒度数据，构建Inst-IT Bench评测基准和335K QA对的指令微调数据集，以持续微调范式有效提升LMM的实例级理解能力，同时增强通用图像/视频理解。

**[Kungfubot Physics-Based Humanoid Whole-Body Control For Learning Highly-Dynamic ](video_understanding/kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)**

:   提出 PBHC 框架，通过物理感知运动处理流水线和自适应跟踪因子的双层优化，使人形机器人（Unitree G1）学会功夫、舞蹈等高动态全身动作，跟踪误差显著优于现有方法并成功实机部署。

**[Lattice Boltzmann Model For Learning Real-World Pixel Dynamicity](video_understanding/lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)**

:   受流体力学中格子玻尔兹曼方法启发，提出 LBM（Lattice Boltzmann Model）用于在线实时像素跟踪，将视频像素建模为流体格子并通过碰撞-流式过程求解运动状态，以 18M 参数实现 SOTA 在线跟踪性能且可在边缘设备上实时运行。

**[Less Is More Local Intrinsic Dimensions Of Contextual Language Models](video_understanding/less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)**

:   提出利用上下文 token 嵌入的局部内在维度（Local Intrinsic Dimension, LID）来无监督监测 LLM 训练动态——维度下降预示泛化改善，维度上升预示过拟合——在对话状态跟踪、grokking、情感识别等任务上验证了这一几何信号的实用性。

**[Livestar Live Streaming Assistant For Real-World Online Video Understanding](video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)**

:   提出 LiveStar，一个始终在线的直播流视频理解助手，通过 Streaming Causal Attention Masks (SCAM) 训练策略和 Streaming Verification Decoding (SVeD) 推理框架，实现自适应响应时机判断，在 OmniStar 基准上语义正确性提升 19.5%，时间偏差降低 18.1%。

**[Memtrack Evaluating Long-Term Memory And State Tracking In Multi-Platform Dynami](video_understanding/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)**

:   提出 MEMTRACK 基准，评估 LLM 智能体在多平台（Slack/Linear/Git）动态环境中的长期记忆和状态追踪能力，揭示即使最强的 GPT-5 也仅达 60% 正确率。

**[Mimeqa Towards Socially-Intelligent Nonverbal Foundation Models](video_understanding/mimeqa_towards_socially-intelligent_nonverbal_foundation_models.md)**

:   构建首个基于哑剧视频的非语言社交推理基准 MimeQA，包含101个视频和806个QA对，覆盖三层问题层次（具象识别→场景理解→全局推理），揭示当前VideoLLMs在非语言社交理解上的严重不足（20-30% vs 人类86%）。

**[Muvr A Multi-Modal Untrimmed Video Retrieval Benchmark With Multi-Level Visual C](video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)**

:   提出 MUVR 基准，面向长视频平台的多模态未剪辑视频检索任务，设计了以视频为中心的多模态查询格式（视频+文本+标签+掩码）和六级视觉对应匹配准则，包含 53K 视频和 1050 个查询，系统评估了检索模型和 MLLM 的局限性。

**[Neural Stochastic Flows Solver-Free Modelling And Inference For Sde Solutions](video_understanding/neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)**

:   提出 Neural Stochastic Flows（NSF），通过条件归一化流直接学习 SDE 的转移分布 $p(x_t \mid x_s)$，在架构上约束满足随机流性质（恒等、Markov、Chapman-Kolmogorov），实现了无需数值求解器的单步采样，在远距时间点上加速高达两个数量级。

**[Neuropath Neurobiology-Inspired Path Tracking And Reflection For Semantically Co](video_understanding/neuropath_neurobiology-inspired_path_tracking_and_reflection_for_semantically_co.md)**

:   受神经生物学中海马体位置细胞导航与记忆巩固机制启发，提出 NeuroPath——一个基于语义路径追踪的 RAG 框架，通过 LLM 驱动的目标导向路径构建和后检索补全策略，在多跳问答任务上实现 recall@2 平均 16.3% 和 recall@5 平均 13.5% 的提升。

**[Open-World Drone Active Tracking With Goal-Centered Rewards](video_understanding/open-world_drone_active_tracking_with_goal-centered_rewards.md)**

:   提出首个开放世界无人机主动跟踪基准 DAT（24 个城市级场景、高保真动力学仿真），以及基于目标中心奖励函数和课程学习的强化学习跟踪方法 GC-VAT，在仿真器上达到约 72% 的跟踪成功率。

**[Part-Aware Bottom-Up Group Reasoning For Fine-Grained Social Interaction Detecti](video_understanding/part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)**

:   提出一种部位感知的自底向上群组推理框架，通过姿态引导的身体部位特征增强和基于相似度的个体关联来推断社交群组和细粒度交互，在 NVI 和 Café 数据集上达到新 SOTA。

**[Pass Path-Selective State Space Model For Event-Based Recognition](video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)**

:   PASS提出路径选择性事件聚合与扫描（PEAS）模块和多面选择引导（MSG）损失，利用SSM的线性复杂度和频率泛化能力，实现了从10^6到10^9事件长度的广泛分布上的事件识别，并在推理频率变化时保持性能仅下降8.62%（基线下降20.69%）。

**[Pixfoundation 20 Do Video Multi-Modal Llms Use Motion In Visual Grounding](video_understanding/pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)**

:   通过提出四项运动中心的探测技术和 MoCentric-Bench 基准，证明当前视频多模态 LLM 在像素级视觉接地任务中未能真正利用运动信息，可被静态关键帧欺骗。

**[Prefm Online Audio-Visual Event Parsing Via Predictive Future Modeling](video_understanding/prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)**

:   本文首次提出在线音视频事件解析（On-AVEP）范式，通过预测性未来建模框架 PreFM，利用伪未来序列增强当前上下文理解，同时借助模态无关的知识蒸馏和焦点时间优先策略，以仅 2.7% 的参数量超越离线 SOTA 方法 +9.3 的事件级平均 F1 分数。

**[Qimeng-Neucomback Self-Evolving Translation From Ir To Assembly Code](video_understanding/qimeng-neucomback_self-evolving_translation_from_ir_to_assembly_code.md)**

:   提出NeuComBack基准数据集用于评估IR到汇编的神经编译任务，并设计自进化提示优化方法，通过从LLM自调试轨迹中学习来迭代改进编译提示，使正确率从44%提升到64%，且87.5%的正确程序性能超越clang-O3。

**[Reinforcement Learning With Backtracking Feedback](video_understanding/reinforcement_learning_with_backtracking_feedback.md)**

:   提出带回溯反馈的强化学习框架 RLBF，当 agent 陷入死胡同时允许回溯到之前的状态重新探索，通过回溯信号改善信用分配，在稀疏奖励环境中显著提升探索效率。

**[Revisiting Bi-Linear State Transitions In Recurrent Neural Networks](video_understanding/revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)**

:   系统性地重新审视 RNN 中的双线性状态转移（隐状态与输入的乘法交互），理论证明双线性 RNN 可模拟任意有限状态机，并展示其在去除加性项后形成了一个从对角到全结构的自然表达力层次，揭示了 Mamba 等流行线性 RNN 处于该层次最低端。

**[Sama Towards Multi-Turn Referential Grounded Video Chat With Large Language Mode](video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)**

:   提出 SAMA 框架，通过构建统一的数据集（SAMA-239K）、模型（时空上下文聚合器 + SAM）和基准（SAMA-Bench），首次实现了多轮引用式视频对话中细粒度时空理解与grounding的联合建模。

**[Seeing Beyond The Scene Analyzing And Mitigating Background Bias In Action Recog](video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)**

:   系统分析了动作识别中背景偏差在分类模型、对比预训练模型（CLIP/SigLIP2）和视频大语言模型（VLLM）三类范式中的普遍存在，并提出两条缓解路径：分类模型通过双分支架构融合分割人体输入降低SBErr 3.78%，VLLM通过自动化prompt tuning降低SBErr 9.85%。

**[Seeing The Arrow Of Time In Large Multimodal Models](video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)**

:   本文揭示当前大多模态模型（LMMs）对视频时间方向性（时间箭头）出人意料地不敏感——正放/倒放时答案几乎相同，提出基于 GRPO 的 ArrowRL 训练策略引入反向视频奖励来激发时间方向感知，并构建 AoTBench 基准，在多个 VQA 基准上取得显著提升（Vinoground 上相对提升 65.9%）。

**[Smartwilds Multimodal Wildlife Monitoring Dataset](video_understanding/smartwilds_multimodal_wildlife_monitoring_dataset.md)**

:   发布首个同步多模态野生动物监测数据集SmartWilds，整合无人机影像、相机陷阱和生物声学三种模态共101GB数据，通过GPS/时间戳实现跨模态对齐，建立可重复的保护监测标准协议，填补综合性生态系统多传感器融合数据集的空白。

**[Steering When Necessary Flexible Steering Large Language Models With Backtrackin](video_understanding/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)**

:   提出 FASB（Flexible Activation Steering with Backtracking）框架，通过跟踪 LLM 生成过程中的内部状态动态判断干预必要性和强度，并引入回溯机制纠正已偏离的 token，在 TruthfulQA 上 True*Info 达 80.56%、6 个多选任务平均准确率 78.8%，显著优于所有基线。

**[Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](video_understanding/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)**

**[Tapvid-360 Tracking Any Point In 360 From Narrow Field Of View Video](video_understanding/tapvid-360_tracking_any_point_in_360_from_narrow_field_of_view_video.md)**

:   本文提出TAPVid-360任务和数据集，要求模型在窄视野视频中跟踪查询点的3D方向（包括视野外的点），通过利用360度视频生成训练数据并微调CoTracker3实现方向预测，在视野外跟踪上远超现有方法。

**[Tempsampr1 Effective Temporal Sampling With Reinforcement Fi](video_understanding/tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)**

:   提出TempSamp-R1强化微调框架，针对GRPO在视频时序定位中因搜索空间巨大而on-policy采样低效的问题，通过引入GT作为off-policy监督信号+非线性软优势估计+混合CoT训练范式，在Charades-STA/ActivityNet/QVHighlights三个基准上达到新SOTA。

**[The Ouroboros Of Benchmarking Reasoning Evaluation In An Era Of Saturation](video_understanding/the_ouroboros_of_benchmarking_reasoning_evaluation_in_an_era_of_saturation.md)**

:   本文通过对OpenAI、Anthropic和Google三大模型家族在52个推理基准上的系统分析，揭示了一种"衔尾蛇"循环模式——旧基准被快速饱和→新基准被创建以维持区分度→新基准又被迅速饱和，由此质疑基准分数的提升是否真正代表了推理能力的泛化进步，还是仅仅反映了对特定评测集的过拟合。

**[Token Bottleneck One Token To Remember Dynamics](video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)**

:   提出Token Bottleneck（ToBo），一种自监督视觉表征学习流水线，通过将参考场景压缩为单个瓶颈token、并利用该token与极少量目标场景patch来预测后续场景，使视觉骨干网络同时学会保守编码场景信息和捕获时间动态变化。

**[Toolaugmented Spatiotemporal Reasoning For Streamlining Vide](video_understanding/toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)**

:   提出了包含 22 个工具的视频工具包和 STAR（Spatiotemporal Reasoning）框架，通过时间-空间工具交替调度策略渐进式定位 3D RoI，在 VideoMME 上将 GPT-4o 提升 8.2%，同时大幅减少处理帧数和计算开销。

**[Tracking And Understanding Object Transformations](video_understanding/tracking_and_understanding_object_transformations.md)**

:   提出 Track Any State 任务和 TubeletGraph 零样本框架，在视频中跟踪经历外观剧变的物体状态变化（如切苹果、蝴蝶从蛹中羽化），同时检测并描述这些变化。

**[Trackingworld World-Centric Monocular 3D Tracking Of Almost All Pixels](video_understanding/trackingworld_world-centric_monocular_3d_tracking_of_almost_all_pixels.md)**

:   提出TrackingWorld，一个从单目视频实现几乎所有像素的稠密3D跟踪的流水线，通过跟踪上采样器将稀疏2D轨迹提升为稠密轨迹、迭代跟踪所有帧中新出现的物体、以及基于优化的框架将2D轨迹提升到世界坐标系3D空间并显式分离相机运动和物体运动。

**[Two Causally Related Needles In A Video Haystack](video_understanding/two_causally_related_needles_in_a_video_haystack.md)**

:   提出Causal2Needles基准（4,100个问答对），通过设计"桥接实体"将两个因果相关事件的理解绑定在一起，强制VLM必须联合检索和推理两个分散在长视频中的"针"，揭示现有最强模型在因果双针问题上的严重不足（ChatGPT-4o双针Both准确率仅13.4%）。

**[Unleashing Hour-Scale Video Training For Long Video-Language Understanding](video_understanding/unleashing_hour-scale_video_training_for_long_video-language_understanding.md)**

:   构建首个大规模小时级视频指令跟随数据集 VideoMarathon（9700小时、330万QA对、22种任务），并提出 Hour-LLaVA 模型，通过记忆仓库+遗忘机制+MemAug模块实现1-FPS下小时级视频的高效训练与推理，在四个长视频基准上全面领先同规模开源模型。

**[Vgent Graph-Based Retrieval-Reasoning-Augmented Generation For Long Video Unders](video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)**

:   提出 VGEnt，一个基于图的检索-推理增强生成框架，通过构建视频知识图谱保留跨片段语义关系，并引入结构化推理步骤过滤噪声、聚合信息，在多个长视频理解基准上一致提升开源 LVLM 3.0%~5.4%，超越现有视频 RAG 方法 8.6%。

**[Video Finetuning Improves Reasoning Between Frames](video_understanding/video_finetuning_improves_reasoning_between_frames.md)**

:   本文通过提出视觉思维链（vCoT）方法，系统地比较了图像LLM与视频微调LLM在帧间推理能力上的差异，发现视频微调使模型隐式学会了帧间过渡推理，且这种能力可迁移到静态图像的关系推理任务中。

**[Videolucy Deep Memory Backtracking For Long Video Understanding](video_understanding/videolucy_deep_memory_backtracking_for_long_video_understanding.md)**

:   提出VideoLucy框架，通过层次化记忆结构和基于Agent的迭代回溯机制，模拟人类从粗到细的回忆过程，在多个长视频理解基准上大幅超越现有方法，甚至超过GPT-4o等商业模型。

**[Visual Diversity And Region-Aware Prompt Learning For Zero-Shot Hoi Detection](video_understanding/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)**

:   提出 VDRP 框架，通过视觉多样性感知的 prompt 学习（注入组级方差 + 高斯扰动）和区域感知的 prompt 增强（基于 LLM 生成的区域概念检索），解决零样本 HOI 检测中类内视觉多样性和类间视觉纠缠两大挑战。

**[Web-Scale Collection Of Video Data For 4D Animal Reconstruction](video_understanding/web-scale_collection_of_video_data_for_4d_animal_reconstruction.md)**

:   提出一个全自动化的大规模视频数据采集管线，从 YouTube 挖掘并处理得到 30K 动物视频（2M帧），建立首个 4D 四足动物重建基准 Animal-in-Motion（230序列/11K帧），并提出 4D-Fauna 基线方法实现序列级优化的无模型 4D 重建。

**[When One Moment Isnt Enough Multi-Moment Retrieval With Cross-Moment Interaction](video_understanding/when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)**

:   提出QV-M2数据集（首个全人工标注的多时刻检索基准）和FlashMMR框架（含后验证模块），将视频时刻检索从单时刻扩展到多时刻场景，建立了多时刻检索的标准化评价体系。

**[When Thinking Drifts Evidential Grounding For Robust Video Reasoning](video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)**

:   系统揭示了CoT推理在视频理解中经常导致性能下降的"视觉思维漂移"现象，并提出Visual Evidence Reward（VER）强化学习框架，通过显式奖励与视觉证据对齐的推理链来纠正这一问题。

---

## 🤖 具身智能 { #robotics }

**[A Snapshot Of Influence A Local Data Attribution Framework F](robotics/a_snapshot_of_influence_a_local_data_attribution_framework_f.md)**

:   首次将数据归因（data attribution）引入在线强化学习，提出局部归因框架量化每条训练记录对策略更新的贡献，并基于此设计了迭代影响力过滤算法（IIF），在经典RL基准和LLM的RLHF上均显著提升了样本效率和最终性能。

**[Adaptive Frontier Exploration On Graphs With Applications To Network-Based Disea](robotics/adaptive_frontier_exploration_on_graphs_with_applications_to_network-based_disea.md)**

:   提出 Adaptive Frontier Exploration on Graphs (AFEG) 问题框架，设计基于 Gittins index 的策略，在图是森林时可证明最优，在实际性传播疾病检测网络上仅测试一半人口即可检出几乎全部 HIV 感染者，大幅超越贪心和 DQN 等基线。

**[Autotom Scaling Model-Based Mental Inference Via Automated Agent Modeling](robotics/autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)**

:   AutoToM 实现完全自动化的基于模型的心智理论推理——无需人工指定 agent 模型，自动提出贝叶斯网络结构并执行贝叶斯逆规划，通过推理不确定性驱动的迭代模型调整（添加心智变量或扩展时间步），在5个ToM benchmark上以82.43%平均准确率超越GPT-4o(63.39%)、o3-mini(73.94%)等SOTA模型。

**[Beyond Parallelism Synergistic Computational Graph Effects In Multi-Head Attenti](robotics/beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)**

:   将多头注意力重新建模为共享汇节点的多个前馈 DAG 系统，理论证明多头可通过跨头路径实现协同效应——降低混合时间(mixing time)并放大 minimax 保真度(fidelity)，在序列操作任务上实验验证了该效应。

**[Bridging Embodiment Gaps Deploying Vision-Language-Action Models On Soft Robots](robotics/bridging_embodiment_gaps_deploying_vision-language-action_models_on_soft_robots.md)**

:   本文首次将 VLA（Vision-Language-Action）模型部署到软体连续体机械臂 Embuddy 上，发现开箱即用的刚性机器人预训练策略因运动学和动力学差异完全失败，但通过在少量软体机器人示范数据上进行针对性微调，可以成功弥合刚性-软体之间的实体鸿沟，使软体平台在抓取和人机交互任务上达到与 UR5 刚性臂相当的任务完成率。

**[C-Nav Towards Self-Evolving Continual Object Navigation In Open World](robotics/c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)**

:   提出 C-Nav 框架，通过**双路径抗遗忘**（特征蒸馏 + 特征回放）和**自适应经验选择**（LOF 异常检测选关键帧），让导航智能体在不断学习新物体类别时避免灾难性遗忘，在 4 种架构上均超越全量数据回放基线。

**[Can Agents Fix Agent Issues](robotics/can_agents_fix_agent_issues.md)**

:   本文首次系统地研究了 LLM-based Agent 系统的 issue 自动修复问题——通过人工分析 201 个真实 Agent issue 构建了涵盖 6 大类 20 个子类的 Agent issue 分类体系，耗费 500 人时构建了包含 50 个可复现任务的 AgentIssue-Bench 基准，并评估发现当前最先进的软件工程 Agent（如 SWE-agent、Agentless、AutoCodeRover）在 Agent issue 上的正确修复率仅为 3.33%–12.67%，远低于它们在传统软件上的 23%–51% 修复率。

**[Cogvla Cognition-Aligned Vision-Language-Action Model Via Instruction-Driven Rou](robotics/cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)**

:   CogVLA 提出模仿人类多模态认知的三阶段VLA架构（EFA-Routing视觉聚合压缩至25% + LFP-Routing LLM内指令感知剪枝50% + V-L-A耦合注意力），在LIBERO上以97.4%成功率和2.5×训练/2.8×推理加速超越OpenVLA-OFT等SOTA方法，真实机器人任务达70.0%成功率。

**[Coopera Continual Open-Ended Human-Robot Assistance](robotics/coopera_continual_open-ended_human-robot_assistance.md)**

:   提出 C-Nav 持续目标导航框架，通过**双路径抗遗忘机制**（特征蒸馏 + 特征回放）和**基于 LOF 的自适应经验选择**，使导航智能体在增量学习新物体类别时有效避免灾难性遗忘，在 4 种主流架构和 2 个数据集上均超越全量数据回放基线。

**[Dexflywheel A Scalable And Self-Improving Data Generation Framework For Dexterou](robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)**

:   提出 DexFlyWheel，一个从单个人类示教出发、通过 IL + 残差 RL + 数据增强组成的自改进循环逐步扩展数据多样性的灵巧操作数据生成框架，在 4 个任务上生成 2000+ 示教，策略平均成功率 81.9%，真实世界迁移成功率 78.3%。

**[Dynanav Dynamic Feature And Layer Selection For Efficient Visual Navigation](robotics/dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)**

:   提出 DynaNav，通过可训练的硬特征选择器和基于贝叶斯优化的 early-exit 机制，根据场景复杂度动态调整特征与层的使用，在视觉导航中实现 2.26× FLOPs 降低、42.3% 推理时间减少，同时保持甚至提升导航性能。

**[Efficientnav Towards On-Device Object-Goal Navigation With Navigation Map Cachin](robotics/efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)**

:   通过离散内存缓存（KV cache分组独立计算+选择性加载）、注意力驱动聚类（LLM浅层attention指导分组）和语义感知检索（CLIP+背包问题适配不同内存预算），首次在Jetson Orin上用LLaMA-3.2-11b实现零样本ObjNav，比GPT-4基线提升11.1% SR且实时延迟降低6.7×。

**[Egothinker Unveiling Egocentric Reasoning With Spatiotempora](robotics/egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)**

:   EgoThinker 构建了 500 万级第一人称视频 QA 数据集 EgoRe-5M（含因果 CoT 标注和手-物体精细定位数据），并通过"先 SFT 学推理、后 GRPO 练定位"的两阶段训练范式，让 7B MLLM 首次同时具备第一人称因果推理和时空精细定位能力，在 8+ 个基准上刷新 SOTA，7B 参数量在时间定位上甚至超过 72B 模型。

**[Enginuity Building An Open Multi-Domain Dataset Of Complex Engineering Diagrams](robotics/enginuity_building_an_open_multi-domain_dataset_of_complex_engineering_diagrams.md)**

:   提出 Enginuity——首个面向 AI 自动解析工程图的大规模开放多领域数据集方案，计划构建 50K+ 带有层级组件关系、空间连接和语义角色标注的汽车工程图，通过四阶段人机协同标注管线实现高质量与低成本的平衡，并定义了从符号检测到数字孪生生成的完整任务体系，为多模态大模型理解工程图中的视觉-结构知识提供了首个系统性基准资源。

**[Explaining And Mitigating Crosslingual Tokenizer Inequities](robotics/explaining_and_mitigating_crosslingual_tokenizer_inequities.md)**

:   系统训练约 7000 个单语分词器覆盖 97 种语言，首次证明即使控制训练数据量、词表大小和算法后，不同语言间仍存在显著的 token premium 差异；进一步识别出词表大小和预分词策略是关键因素，并提出"最优词表大小"和 SuperBPE 两种缓解方案。

**[Falcon Fine-Grained Activation Manipulation By Contrastive Orthogonal Unalignmen](robotics/falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)**

:   提出 FALCON——基于表示引导的 LLM 遗忘框架，利用互信息进行参数选择、对比机制实现精细知识分离、梯度正交投影解决遗忘-保留冲突，在有害知识/版权/实体遗忘任务上全面超越现有方法。

**[Generalizable Domain Adaptation For Sim-And-Real Policy Co-Training](robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)**

:   提出基于不平衡最优运输（UOT）的模拟-真实策略联合训练框架，通过对观察-动作联合分布进行对齐（而非仅对齐观察边际分布），结合时间对齐采样策略处理数据不平衡，在机器人操纵任务上实现30%的OOD泛化提升。

**[Harnessing The Computation Redundancy In Vits To Boost Adversarial Transferabili](robotics/harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili.md)**

:   深入挖掘 ViT 中数据级和模型级的计算冗余，提出注意力稀疏化、注意力头置换、干净 token 正则化、Ghost MoE 多样化和鲁棒化 token 五种技术，结合在线学习策略动态选择操作，在 ImageNet-1K 上以 86.9% 平均 fooling rate 大幅超越所有基线。

**[Himacon Discovering Hierarchical Manipulation Concepts From Unlabeled Multi-Moda](robotics/himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)**

:   提出自监督框架从无标注多模态机器人演示中学习层级操作概念，通过跨模态相关性网络和多时域子目标预测器组织表示，增强模仿学习策略在新物体、新障碍和新环境下的泛化能力。

**[Knolling Bot Teaching Robots The Human Notion Of Tidiness](robotics/knolling_bot_teaching_robots_the_human_notion_of_tidiness.md)**

:   将桌面物体整理（knolling）类比为 NLP 序列预测任务，用 Transformer 自回归生成每个物体的目标位置，结合 GMM 处理多解歧义，从 240 万组自动生成的示范中学习通用整洁概念，并通过输入排列顺序隐式编码用户偏好。

**[Labutopia High-Fidelity Simulation And Hierarchical Benchmark For Scientific Emb](robotics/labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)**

:   提出 LabUtopia——面向科学实验室的高保真仿真与层级基准套件，包含支持化学反应建模的 LabSim 仿真器、可程序化生成实验室场景的 LabScene、以及从原子操作到长程移动操纵的五级 LabBench 基准，揭示现有模仿学习方法在长程实验流程和物体泛化方面的显著瓶颈。

**[Latentguard Controllable Latent Steering For Robust Refusal Of Attacks And Relia](robotics/latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)**

:   提出 LatentGuard 三阶段框架，通过行为级对齐微调 + 结构化 VAE 监督潜空间 + 潜空间维度操控，实现对 LLM 拒绝行为的可解释、可控制调节，在抵御对抗攻击的同时保持对正常查询的响应能力。

**[Learning Spatial-Aware Manipulation Ordering](robotics/learning_spatial-aware_manipulation_ordering.md)**

:   提出 OrderMind 统一框架，通过空间上下文编码器和时序优先级结构化模块直接从 RGB-D 图像学习杂乱场景中物体的操作顺序，利用 VLM 蒸馏生成训练标注，在仿真和真实环境中均显著优于 VLM 基线，且支持实时推理（5.6 FPS，轻量版 21.3 FPS）。

**[Llm World Models Are Mental Output Layer Evidence Of Brittle World Model Use In ](robotics/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)**

:   借鉴认知科学的心理模型研究方法，通过滑轮系统的TikZ代码表示测试LLM的力学推理能力，发现LLM能近似估计机械优势并区分功能/非功能系统（Study 1&2），但在精细结构连接推理上完全失败（Study 3），表明LLM的"世界模型"存在但脆弱。

**[Llmscape](robotics/llmscape.md)**

:   LLMscape 是一个投影映射沙盘交互装置，让多个独立 LLM 代理在共享的可变物理环境中接收多模态输入、相互对话和推测，探索人类与 AI 在认知不确定性下的共同意义构建过程。

**[Mango - Adaptable Graph Network Simulators Via Meta-Learning](robotics/mango_-_adaptable_graph_network_simulators_via_meta-learning.md)**

:   提出 MaNGO（Meta Neural Graph Operator），通过元学习和条件神经过程（CNP）学习不同物理参数下仿真任务的共享潜在结构，实现对新物理参数的快速适应，无需重新训练。

**[Manipulating Feature Visualizations With Gradient Slingshots](robotics/manipulating_feature_visualizations_with_gradient_slingshots.md)**

:   提出 Gradient Slingshots（梯度弹弓）方法，通过在模型的分布外输入区域"雕刻"出导向任意目标图像的二次激活景观，使特征可视化（Feature Visualization）的梯度优化过程收敛到预设的虚假图像，同时保持模型架构、分类精度和内部特征表示基本不变，暴露了 FV 作为模型审计工具的严重脆弱性。

**[Mesatask Towards Task-Driven Tabletop Scene Generation Via 3D Spatial Reasoning](robotics/mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)**

:   提出 MesaTask 框架，通过 Spatial Reasoning Chain 将任务描述分解为对象推理→空间关系推理→场景图构建→3D 布局，结合 10K+ 人工标注数据集和 DPO 优化，生成物理合理且任务对齐的桌面操控场景。

**[Mindforge Empowering Embodied Agents With Theory Of Mind For Lifelong Cultural L](robotics/mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)**

:   MindForge 为 LLM 驱动的具身智能体引入显式的心智理论（ToM）表征、自然语言通信和多组件记忆系统，使开源 LLM 智能体通过与专家协作对话（无需梯度更新）大幅提升任务完成率，在 Minecraft 中比 Voyager 多获得 3× 科技树里程碑和 2.3× 独特物品。

**[Mineanybuild Benchmarking Spatial Planning For Openworld Ai](robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)**

:   基于 Minecraft 构建空间规划基准 MineAnyBuild，要求 AI Agent 根据多模态指令生成可执行的建筑蓝图矩阵，包含 4000 个任务和 500+ 建筑/装饰资产，从空间理解、空间推理、创造力和空间常识四个维度系统评估 MLLM 的空间规划能力，揭示即便 GPT-4o 整体得分仅 41.02/100，开源模型更差。

**[Mip Against Agent Malicious Image Patches Hijacking Multimod](robotics/mip_against_agent_malicious_image_patches_hijacking_multimod.md)**

:   揭示针对多模态OS Agent的新型对抗攻击MIP(Malicious Image Patches)：在屏幕截图中嵌入人眼不可察觉的对抗性扰动图像块(约占屏幕1/7面积)，当OS Agent截屏捕获后会输出预定义的恶意API调用序列；通过联合优化实现跨用户指令和屏幕布局的Universal泛化，攻击成功率高达100%。

**[Mmtu A Massive Multi-Task Table Understanding And Reasoning Benchmark](robotics/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)**

:   构建了一个包含 28,136 道问题、覆盖 25 种真实表格任务的大规模基准 MMTU，系统评估 LLM 在专业级表格理解、推理和操作方面的能力，发现即使是 GPT-5 等前沿推理模型也仅得分约 69.6%。

**[Mmwalk Towards Multi-Modal Multi-View Walking Assistance](robotics/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)**

:   mmWalk 构建了首个面向视障人群步行辅助的多模态多视角数据集（CARLA 仿真器生成 62K 帧/559K 全景图 + 69K VQA 对），基准测试发现 SOTA VLM 在风险评估和导航地标识别等安全关键任务上表现不足（最优仅 55.21%），微调后在真实数据集上泛化提升 16.7%。

**[Nesypr Neurosymbolic Proceduralization For Efficient Embodied Reasoning](robotics/nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)**

:   提出 NeSyPr（Neurosymbolic Proceduralization），一种将符号规划器的声明性知识编译为可组合过程化表示的框架，使紧凑语言模型在无需在线符号引导的情况下完成高效具身推理，在 PDDLGym、VirtualHome 和 ALFWorld 三个基准上超越大规模推理模型和符号规划器。

**[Operation Veja Fixing Fundamental Concepts Missing From Modern Roleplaying Train](robotics/operation_veja_fixing_fundamental_concepts_missing_from_modern_roleplaying_train.md)**

:   本文系统批判了现有角色扮演模型训练的四大范式（RAG、事实值设定、文学数据、合成数据）为何都无法产生有深度的角色，提出VEJA框架（Values-Experiences-Judgments-Abilities）作为角色定义和数据策化的结构化基础，在LLM评判A/B测试中VEJA指导的人工策化数据以43:28:29（胜:负:平）显著优于Gemini Pro 2.5生成的合成基线。

**[Physics Of Language Models Part 41 Architecture Design And The Magic Of Canon La](robotics/physics_of_language_models_part_41_architecture_design_and_the_magic_of_canon_la.md)**

:   通过受控合成预训练任务系统性比较语言模型架构，发现 Canon 层——一种轻量级的邻近token加权求和组件——能显著提升推理深度（2-4倍）、推理广度、知识容量等核心能力，让 NoPE 匹配 RoPE，让 GLA 匹敌 Mamba2/GDN。

**[Predicting The Performance Of Black-Box Llms Through Follow-Up Queries](robotics/predicting_the_performance_of_black-box_llms_through_follow-up_queries.md)**

:   提出 QueRE 方法，通过向黑盒LLM提出约50个后续问题（如"你对回答有信心吗？"），以"Yes"token的概率作为特征训练线性分类器，在预测模型正确性、检测对抗操纵和区分不同LLM等任务上，甚至超越需要访问模型内部状态的白盒方法。

**[Pretraining A Unified Pddl Domain From Real-World Demonstrations For Generalizab](robotics/pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)**

:   UniDomain 从 12,393 个真实机器人操作视频中预训练统一的 PDDL 规划域（含 3,137 个算子和 2,875 个谓词），通过层级融合构建元域，实现零样本跨任务符号规划，比最强基线高出 58% 成功率和 160% 计划最优性。

**[Rdd Retrieval-Based Demonstration Decomposer For Planner Alignment In Long-Horiz](robotics/rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)**

:   提出RDD（基于检索的演示分解器），通过将演示分解建模为最优分区问题，自动将长时域任务演示分解为与底层视觉运动策略训练数据对齐的子任务，从而协调层级VLA框架中高层规划器与低层策略，在RLBench上接近专家分解器的性能。

**[Redefining Experts Interpretable Decomposition Of Language Models For Toxicity M](robotics/redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)**

:   提出EigenShift方法，通过对LLM最终输出层进行SVD分解，识别与毒性生成相关的特征方向（eigen-choices），并通过选择性衰减对应奇异值来实现毒性抑制——在LLaMA-2上降低58%毒性的同时仅增加3.62的困惑度，兼顾安全与流畅性。

**[Rethinking The Simulation Vs Rendering Dichotomy No Free Lunch In Spatial World ](robotics/rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)**

:   从认知神经科学视角挑战"模拟与渲染可分离"的传统观点：论证空间推理依赖于精细的感知表征而非粗粒度抽象，并指出AI空间世界模型同样需要保留丰富的感知细节——空间建模没有免费午餐。

**[Robocerebra A Large-Scale Benchmark For Long-Horizon Robotic Manipulation Evalua](robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)**

:   提出RoboCerebra长程机器人操作基准，包含1000条人类示范轨迹（平均2972步，约为现有基准的6倍），通过分层规划与执行框架和多维评估协议，系统测评VLM在规划、反思和记忆三个System 2认知维度上的能力。

**[Segmast3R Geometry Grounded Segment Matching](robotics/segmast3r_geometry_grounded_segment_matching.md)**

:   SegMASt3R 在预训练 MASt3R 3D 基础模型上添加轻量分割特征头和可微 Sinkhorn 匹配层，利用 3D 几何先验实现极端视角变化（达 180°）下的鲁棒语义段匹配，AUPRC 在 135-180° 基线上达 83.6%（vs SAM2 的 17%）。

**[Sitcom Scaling Inference-Time Compute For Vlas](robotics/sitcom_scaling_inference-time_compute_for_vlas.md)**

:   SITCOM 提出了一种受模型预测控制（MPC）启发的推理时计算框架，通过学习的动力学模型对预训练 VLA 进行多步rollout仿真并利用奖励模型选择最优轨迹，将单步 VLA 转化为鲁棒的长程规划器，在 SIMPLER 环境中将任务完成率从 48% 提升至 72%。

**[Suturebot A Precision Framework Benchmark For Autonomous End-To-End Suturing](robotics/suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)**

:   提出SutureBot——首个针对da Vinci手术机器人端到端自主缝合的精度导向基准与目标条件框架，发布1890条高保真演示数据集，通过点标签目标条件将针刺精度提升59%-74%，并系统评估了π0、GR00T N1、OpenVLA-OFT和多任务ACT等SOTA VLA模型。

**[T-Rex Task-Adaptive Spatial Representation Extraction For Robotic Manipulation W](robotics/t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)**

:   提出T-Rex框架，根据任务复杂度动态选择最优的空间表示提取方案（点/向量/6D位姿），并设计Chain of Grounding (CoG)引导VLM逐步推理，实现无需训练的开放词汇机器人操纵。

**[Talk2Event Grounded Understanding Of Dynamic Scenes From Event Cameras](robotics/talk2event_grounded_understanding_of_dynamic_scenes_from_event_cameras.md)**

:   Talk2Event 提出首个大规模事件相机视觉定位基准（30,690 条标注表达式 + 四种定位属性），并设计 EventRefer 框架通过混合事件-属性专家（MoEE）动态融合外观/状态/观察者关系/物体间关系特征，在纯事件、纯帧和融合三种设置下均超越现有方法。

**[Task-Optimized Convolutional Recurrent Networks Align With Tactile Processing In](robotics/task-optimized_convolutional_recurrent_networks_align_with_tactile_processing_in.md)**

:   提出Encoder-Attender-Decoder（EAD）框架系统探索触觉任务优化的时序神经网络，发现卷积循环网络（ConvRNN，特别是IntersectionRNN）在触觉物体分类和啮齿类体感皮层神经对齐上均优于前馈和状态空间模型，且基于触觉特定增强的对比自监督学习能达到与监督学习相当的神经拟合，为触觉的大脑计算机制提供了首个定量刻画。

**[Thinkact Vision-Language-Action Reasoning Via Reinforced Visual Latent Planning](robotics/thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)**

:   提出ThinkAct双系统框架，通过动作对齐的视觉奖励对MLLM进行强化学习微调以激发具身推理能力，并将推理计划压缩为视觉潜在表示来指导下游动作模型，实现"先思考再行动"的VLA推理范式。

**[Toward Engineering Agi Benchmarking The Engineering Design Capabilities Of Llms](robotics/toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)**

:   提出 EngDesign——首个跨 9 个工程领域（操作系统、计算机架构、控制系统、机械、结构、数字硬件、模拟电路、机器人、信号处理）的 LLM 工程设计能力基准，用仿真驱动的评估管线替代传统的问答匹配，揭示即使最强推理模型 o3 也仅达 34% 通过率。

**[Towards Reliable Code-As-Policies A Neuro-Symbolic Framework For Embodied Task P](robotics/towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)**

:   提出一种神经-符号具身任务规划框架，在 LLM 代码生成过程中引入显式的符号验证（检查前置条件是否满足）和交互式验证（主动探索获取缺失信息），使生成的代码在动态和部分可观测场景中更可靠——在 RLBench 上任务成功率从基线 38.5% 提升到 84.7%，可执行性达 86.8%。

**[Understanding Prompt Tuning And In-Context Learning Via Meta-Learning](robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)**

:   从贝叶斯元学习视角系统分析了提示调优（prompt tuning）的理论基础与局限性，证明了软提示可以在预训练分布内的单一目标任务上实现最优适配，但对多任务混合目标分布存在根本性限制，且软前缀能通过操纵非token空间的激活来超越最优硬token序列。

**[Vitrix-Clipin Enhancing Fine-Grained Visual Understanding In Clip Via Instructio](robotics/vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)**

:   提出 CLIP-IN 框架，利用指令编辑数据集作为硬负样本和长描述增强 CLIP 的细粒度视觉理解能力，在 MMVP 等基准上显著提升且不损害零样本性能，集成到 MLLM 中可减少视觉幻觉。

**[Vla-Cache Efficient Vision-Language-Action Manipulation Via Adaptive Token Cachi](robotics/vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)**

:   提出VLA-Cache，一种免训练的VLA推理加速方法，通过跨帧识别并缓存静态视觉token的KV表示、过滤任务相关token并按层自适应调整复用比例，实现1.7倍加速且几乎不损失任务成功率。

**[Zero-Shot Embedding Drift Detection A Lightweight Defense Against Prompt Injecti](robotics/zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)**

:   提出ZEDD（零样本嵌入漂移检测），通过比较良性和可疑输入在嵌入空间中的语义漂移来检测提示注入攻击，利用GMM/KDE自动确定阈值，在多种LLM架构上实现>93%的检测准确率且假阳性率<3%。

---

## 🕸️ 图学习 { #graph_learning }

**[Bliss Bandit Layer Importance Sampling Strategy For Efficient Training Of Graph ](graph_learning/bliss_bandit_layer_importance_sampling_strategy_for_efficient_training_of_graph_.md)**

:   提出 BLISS，将 GNN 的层级邻居采样建模为多臂老虎机问题，用 EXP3 算法动态调整每条边的采样概率，根据邻居对节点表示的方差贡献作为奖励信号，在 GCN 和 GAT 上维持或超越全批次训练精度。

**[Deliberation On Priors Trustworthy Reasoning Of Large Language Models On Knowled](graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)**

:   提出 Deliberation over Priors（DP）框架，通过渐进式知识蒸馏（SFT + KTO 偏好优化）提升关系路径生成的忠实度，并通过约束引导的内省-回溯机制保障推理可靠性，在 ComplexWebQuestions 上 H@1 提升 16.5%，且 LLM 调用次数仅为 2.9 次（ToG 需 22.6 次）。

**[Diagnosing And Addressing Pitfalls In Kg-Rag Datasets Toward More Reliable Bench](graph_learning/diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)**

:   系统审计16个KGQA数据集发现平均事实正确率仅57%（WebQSP 52%，MetaQA 20%），提出KGQAGen框架——通过LLM引导的子图扩展+SPARQL自动验证构建高质量多跳QA数据集KGQAGen-10k（96.3%准确率），揭示KG-RAG的主要瓶颈在检索而非推理。

**[Disentangling Hyperedges Through The Lens Of Category Theory](graph_learning/disentangling_hyperedges_through_the_lens_of_category_theory.md)**

:   首次从范畴论视角分析超边解耦，基于自然性条件导出"因子表示一致性"标准（聚合后解耦 vs 解耦后聚合应一致），提出 Natural-HNN 模型在6个癌症分型数据集上全面超越14个baseline（BRCA F1 从75.7%提升至80.4%），并能100%正确捕获基因通路的功能上下文。

**[Duetgraph Coarse-To-Fine Knowledge Graph Reasoning With Dual-Pathway Global-Loca](graph_learning/duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)**

:   DuetGraph 提出双通路（消息传递 + 全局注意力）并行融合模型与粗到精推理优化策略，通过分离而非堆叠局部/全局信息处理来缓解 KG 推理中的分数过平滑问题，在归纳与传导推理任务上取得 SOTA，MRR 最高提升 8.7%、训练加速 1.8×。

**[Dynamic Bundling With Large Language Models For Zero-Shot Inference On Text-Attr](graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)**

:   DENSE 提出"文本捆绑"策略，将拓扑/语义相近的节点文本打包后查询 LLM 获取 bundle 级别标签，再用 entropy-based 和 ranking-based 损失监督 GNN 训练，并动态精炼 bundle 排除噪声节点，在 10 个 TAG 数据集上零样本推理全面超越 GPT-4o 和图基础模型。

**[Elastic Weight Consolidation For Knowledge Graph Continual Learning An Empirical](graph_learning/elastic_weight_consolidation_for_knowledge_graph_continual_learning_an_empirical.md)**

:   本文在 FB15k-237 上系统评估了弹性权重固化（EWC）对 TransE 知识图谱嵌入持续学习的效果，发现 EWC 将灾难性遗忘从 12.62% 降至 6.85%（减少 45.7%），并揭示了任务划分策略（基于关系 vs 随机）对遗忘度量的显著影响（9.8 个百分点差异）。

**[Esca Contextualizing Embodied Agents Via Scene-Graph Generation](graph_learning/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)**

:   提出 ESCA 框架，通过开放域场景图生成（SGClip 模型）为 MLLM 驱动的具身智能体提供结构化视觉理解上下文，显著降低了感知错误率并提升了任务完成率。

**[Falcon An Ml Framework For Fully Automated Layout-Constrained Analog Circuit Des](graph_learning/falcon_an_ml_framework_for_fully_automated_layout-constrained_analog_circuit_des.md)**

:   FALCON 提出端到端的模拟/RF 电路自动化设计框架，通过 MLP 拓扑选择 + 边中心 GNN 性能预测 + 可微版图约束梯度推理三阶段流水线，在 100 万级 Cadence 仿真数据集上实现 >99% 拓扑选择准确率、<10% 性能预测误差，单实例推理不到 1 秒。

**[Fastjam A Fast Joint Alignment Model For Images](graph_learning/fastjam_a_fast_joint_alignment_model_for_images.md)**

:   提出 FastJAM，一种基于图的快速图像联合对齐方法：利用现成图像匹配器计算成对关键点对应，通过快速非参数聚类构建关键点图，GNN 传播聚合信息后预测每张图像的单应性参数，配合反向合成损失（inverse-compositional loss）消除正则化超参数需求。将联合对齐时间从小时/分钟级降至约 49 秒，同时对齐质量优于或持平现有方法。

**[From Sequence To Structure Uncovering Substructure Reasoning In Transformers](graph_learning/from_sequence_to_structure_uncovering_substructure_reasoning_in_transformers.md)**

:   本文通过实证和理论分析揭示 decoder-only Transformer 如何从文本序列中理解图结构，提出"诱导子图过滤"（ISF）解释子结构逐层识别机制，并扩展到 LLM 验证一致性、复合图推理（Thinking-in-Substructures）和属性图（分子图）子结构提取。

**[Generative Graph Pattern Machine](graph_learning/generative_graph_pattern_machine.md)**

:   提出 Generative Graph Pattern Machine (G2PM)，一种完全无消息传递的生成式 Transformer 图预训练框架：通过随机游走将图实例（节点/边/图）tokenize 为子结构序列，以 Masked Substructure Modeling 目标进行自监督预训练，在节点/链接/图分类及跨域迁移任务上全面超越现有图预训练方法，并展现出类似 NLP/CV 的模型和数据双重扩展性。

**[Geometric Imbalance In Semi-Supervised Node Classification](graph_learning/geometric_imbalance_in_semi-supervised_node_classification.md)**

:   首次形式化定义了半监督节点分类中的"几何不平衡"概念——消息传递在类别不平衡图上导致少数类节点在黎曼流形嵌入空间中产生几何歧义，并提出 UNREAL 框架，通过双路径伪标签对齐、节点重排序和几何歧义样本丢弃三个模块系统性缓解该问题。

**[Gfm-Rag Graph Foundation Model For Retrieval Augmented Generation](graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)**

:   提出首个图基础模型驱动的检索增强生成框架 GFM-RAG，通过 query-dependent GNN 在知识图谱上进行单步多跳推理，仅 8M 参数即可在未见数据集上零样本泛化，在多跳QA检索任务上大幅超越 SOTA。

**[Gnnxemplar Exemplars To Explanations -- Natural Language Rules For Global Gnn In](graph_learning/gnnxemplar_exemplars_to_explanations_--_natural_language_rules_for_global_gnn_in.md)**

:   提出GnnXemplar框架，基于认知科学的样例理论（Exemplar Theory），通过在GNN嵌入空间中选取代表性节点（exemplar）并利用LLM迭代生成自然语言布尔规则，实现大规模图上节点分类GNN的全局可解释性。

**[Graph Neural Networks For Efficient Ac Power Flow Prediction In Power Grids](graph_learning/graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)**

:   将电力网络建模为图结构（母线为节点、输电线为边），探索 GCN、GAT、SAGEConv 和 GraphConv 四种 GNN 架构预测 AC 潮流解（电压幅值和相角），在 IEEE 14/30/57/118 母线测试系统上展示了 GNN 可高效替代传统 Newton-Raphson 求解器。

**[Graph Neural Networks For Interferometer Simulations](graph_learning/graph_neural_networks_for_interferometer_simulations.md)**

:   首次将图神经网络应用于光学干涉仪仿真，通过 GATv2 + KAN 架构预测 LIGO 干涉仪中的电磁场功率和空间强度分布，实现比标准仿真软件（FINESSE）快 815 倍的推理速度，同时保持较好的物理精度。

**[Graph Persistence Goes Spectral](graph_learning/graph_persistence_goes_spectral.md)**

:   提出 SpectRe——将图拉普拉斯谱信息融入持续同调（PH）图的新拓扑描述符，证明其表达力严格强于 PH 和谱信息单独使用，建立了局部稳定性理论，在合成和真实数据集上提升 GNN 的图分类能力。

**[Graphfaas Serverless Gnn Inference For Burst-Resilient Real-Time Intrusion Detec](graph_learning/graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)**

:   提出 GraphFaaS，一种专为 GNN 入侵检测设计的 Serverless 推理架构，通过来源图的增量构建、特征长度感知的并行节点嵌入和贪心 best-fit 子图分区，将平均检测延迟从 14.16 秒降至 2.1 秒（6.7 倍），变异系数从 1.46 降至 0.52（64% 降低），在突发负载下保持稳定低延迟且不损失检测准确率。

**[Graphtop Graph Topology-Oriented Prompting For Graph Neural Networks](graph_learning/graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)**

:   提出首个图拓扑导向的 prompting 框架 GraphTOP，通过将 topology-oriented prompting 建模为边重连问题并用 Gumbel-Softmax 松弛到连续空间，在 5 个数据集 4 种预训练策略下超越 6 个基线方法。

**[Heterogeneous Swarms Jointly Optimizing Model Roles And Weights For Multi-Llm Sy](graph_learning/heterogeneous_swarms_jointly_optimizing_model_roles_and_weights_for_multi-llm_sy.md)**

:   提出Heterogeneous Swarms算法，将多LLM系统建模为有向无环图（DAG），通过粒子群优化（PSO）联合优化模型角色（图结构）和模型权重，在12个任务上平均超越17个基线18.5%。

**[Inductive Transfer Learning For Graph-Based Recommenders](graph_learning/inductive_transfer_learning_for_graph-based_recommenders.md)**

:   提出 NBF-Rec，一个基于神经 Bellman-Ford 网络的图推荐模型，支持在用户和物品完全不相交的数据集之间进行归纳式迁移学习，实现零样本跨域推荐和轻量微调适配。

**[Interaction-Centric Knowledge Infusion And Transfer For Open-Vocabulary Scene Gr](graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)**

:   本文提出ACC框架，通过交互驱动范式（而非传统以对象为中心的范式）来解决开放词汇场景图生成中的关键匹配问题：在知识注入阶段用双向交互提示生成更准确的伪监督，在知识迁移阶段用交互引导的查询选择和交互一致性知识蒸馏来减少不匹配，在VG、GQA、PSG三个基准上达到SOTA。

**[Learning Repetition-Invariant Representations For Polymer Informatics](graph_learning/learning_repetition-invariant_representations_for_polymer_informatics.md)**

:   提出 GRIN（Graph Repetition-Invariant Network），通过 Max 聚合和特殊的图构建策略使 GNN 对聚合物重复单元的拼接数量不变，解决了聚合物表示中的基本对称性问题。

**[Logical Expressiveness Of Graph Neural Networks With Hierarchical Node Individua](graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)**

:   提出了分层自我图神经网络（Hierarchical Ego GNNs，HEGNNs），通过层次化的节点个体化机制泛化了子图GNN，形成表达力递增的模型层级；在有界度图上，证明HEGNN节点分类器的区分能力等价于分级杂合逻辑（graded hybrid logic），从而统一了多种GNN变体的表达力分析。

**[Making Classic Gnns Strong Baselines Across Varying Homophily A Smoothness-Gener](graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)**

:   从理论上揭示了 GNN 消息传递中平滑性（smoothness）与泛化性（generalization）之间的两难困境，提出 IGNN 框架通过三个简约设计原则（分离邻域变换、感知聚合、邻域关系学习）缓解该困境，在 30 个基线中表现最优且具备跨同质/异质图的通用性。

**[Mixture Of Scope Experts At Test Generalizing Deeper Graph Neural Networks With ](graph_learning/mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)**

:   从 PAC-Bayes 泛化理论出发，证明 GNN 深度变化导致不同同质性节点子群间的泛化偏好漂移，据此提出 Moscat——一种后处理注意力门控模型，将独立训练的不同深度 GNN 专家在测试时节点自适应地融合，在多种 GNN 架构和数据集上实现显著提升。

**[Moemeta Mixture-Of-Experts Meta Learning For Few-Shot Relational Learning](graph_learning/moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)**

:   提出MoEMeta框架，通过混合专家模型学习全局共享的关系原型实现跨任务泛化，结合任务定制的投影适应机制捕获局部上下文，在三个KG基准上达到SOTA。

**[Nonlinear Laplacians Tunable Principal Component Analysis Under Directional Prio](graph_learning/nonlinear_laplacians_tunable_principal_component_analysis_under_directional_prio.md)**

:   提出非线性Laplacian谱算法，通过在观测矩阵 $\bm{Y}$ 上添加由度数向量经非线性函数 $\sigma$ 变换后得到的对角矩阵，将谱信息与方向先验信息融合，在稀疏偏向PCA问题中显著降低信号检测阈值（从 $\beta^*=1$ 降至约 $0.76$）。

**[Ocn Effectively Utilizing Higher-Order Common Neighbors For Better Link Predicti](graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)**

:   揭示高阶公共邻居（CN）在链接预测中的冗余和过平滑问题，提出正交化（Gram-Schmidt 去除阶间线性相关）+ 归一化（除以路径数，广义资源分配启发式）解决方案，在 7 个数据集上平均提升 HR@100 7.7%，DDI 数据集上提升 13.3%。

**[Over-Squashing In Spatiotemporal Graph Neural Networks](graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)**

:   首次形式化时空图神经网络(STGNN)中的 over-squashing 问题，揭示了因果卷积中反直觉的"时间远处偏好"现象（最早时间步对最终表示影响最大），并证明 time-and-space 和 time-then-space 架构在信息瓶颈上等价，为使用计算高效的 TTS 架构提供理论支持。

**[P-Drum Post-Hoc Descriptor-Based Residual Uncertainty Modeling For Machine Learn](graph_learning/p-drum_post-hoc_descriptor-based_residual_uncertainty_modeling_for_machine_learn.md)**

:   提出 P-DRUM，一种简单高效的事后（post-hoc）不确定性量化框架，利用已训练图神经网络势的描述子来估计预测残差，作为不确定性代理，无需修改原模型架构或训练流程。

**[Practical Bayes-Optimal Membership Inference Attacks](graph_learning/practical_bayes-optimal_membership_inference_attacks.md)**

:   提出 BASE 和 G-BASE 两种实用的贝叶斯最优成员推断攻击方法，分别针对 i.i.d. 数据和图结构数据,在保持理论最优性的同时大幅降低计算成本。

**[Preference-Driven Knowledge Distillation For Few-Shot Node Classification](graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)**

:   PKD 框架协同 LLM 和多 GNN 教师做文本属性图少样本节点分类——GNN 偏好节点选择器（GNS）用 KL 散度不确定性选择需要 LLM 标注的节点，节点偏好 GNN 选择器（NGS）用 RL 为每个节点匹配最优 GNN 教师，在 9 个数据集上一致 SOTA（Cornell 87% vs 基线 59-82%）。

**[Principled Data Augmentation For Learning To Solve Quadratic Programming Problem](graph_learning/principled_data_augmentation_for_learning_to_solve_quadratic_programming_problem.md)**

:   提出基于KKT系统仿射变换的原则性数据增强框架，为线性规划(LP)和二次规划(QP)的MPNN学习优化(L2O)任务生成保最优性的增强实例，并结合对比学习预训练，在数据稀缺和OOD泛化场景下大幅提升性能。

**[Reasoning Meets Representation Envisioning Neuro-Symbolic Wireless Foundation Mo](graph_learning/reasoning_meets_representation_envisioning_neuro-symbolic_wireless_foundation_mo.md)**

:   提出将神经符号（Neuro-Symbolic）范式与无线物理层基础模型（WPFM）结合的愿景框架——以WPFM作为神经感知引擎生成RF嵌入向量，以本体论驱动的知识图谱和可微逻辑层作为符号推理组件，实现可解释、可泛化且可验证合规的无线AI系统，为AI原生6G网络提供技术路径。

**[Relieving The Over-Aggregating Effect In Graph Transformers](graph_learning/relieving_the_over-aggregating_effect_in_graph_transformers.md)**

:   发现了 Graph Transformer 中的 over-aggregating 现象——大量节点以近均匀注意力分数被聚合导致关键信息被稀释，提出 Wideformer 通过分割聚合+引导注意力来缓解，作为即插即用模块在 13 个数据集上一致提升骨干模型性能。

**[Remindrag Low-Cost Llm-Guided Knowledge Graph Traversal For Efficient Rag](graph_learning/remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)**

:   提出ReMindRAG，一种结合LLM引导的KG遍历（节点探索+利用）与无训练记忆重放机制的KG-RAG系统，将LLM遍历经验存储在边嵌入中，在后续相似查询时显著减少LLM调用次数（约50%成本降低），同时提升回答准确率（5%-10%提升）。

**[Self-Supervised Discovery Of Neural Circuits In Spatially Patterned Neural Respo](graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)**

:   提出基于GNN的自监督框架，通过结构学习模块推断潜在突触连接、同时用脉冲预测模块预测未来发放活动，在环形吸引子网络仿真数据和真实小鼠头方向细胞记录上均显著优于统计推断基线。

**[Sketch-Augmented Features Improve Learning Long-Range Dependencies In Graph Neur](graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)**

:   提出Sketched Random Features (SRF)，将节点特征的核空间随机投影注入标准消息传递GNN的每一层，同时缓解过压缩、过平滑和表达力受限三大问题，理论性质完备且计算高效。

**[Smore Structural Mixture Of Residual Experts For Parameter-Efficient Llm Fine-Tu](graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)**

:   提出S'MoRE框架，将低秩残差专家组织成多层树状结构，通过层次化路由为每个token构建定制化的"残差树"，在与LoRA相当的参数量下实现指数级增长的结构灵活性，显著提升LLM微调效果。

**[Solar-Geco Perovskite Solar Cell Property Prediction With Geometric-Aware Co-Att](graph_learning/solar-geco_perovskite_solar_cell_property_prediction_with_geometric-aware_co-att.md)**

:   提出Solar-GECO多模态框架，将钙钛矿吸收层的3D晶体结构通过几何GNN编码、器件其他层通过LLM文本嵌入编码，经共注意力融合后预测光电转换效率(PCE)及其不确定性，MAE从3.066降至2.936。

**[Spatio-Temporal Directed Graph Learning For Account Takeover Fraud Detection](graph_learning/spatio-temporal_directed_graph_learning_for_account_takeover_fraud_detection.md)**

:   提出 ATLAS 框架，将账户接管（ATO）欺诈检测重新建模为时空有向图上的节点分类问题，通过时间窗口 + 最近邻约束构建因果有向图，结合延迟感知标签传播和 GraphSAGE 编码器，在 Capital One 的 1 亿节点、10 亿边大规模生产图上实现 +6.38% AUC 提升和超过 50% 的用户摩擦降低。

**[Spot-Trip Dual-Preference Driven Out-Of-Town Trip Recommendation](graph_learning/spot-trip_dual-preference_driven_out-of-town_trip_recommendation.md)**

:   提出SPOT-Trip框架，首次系统研究异地旅行推荐问题，通过知识图谱增强的静态偏好学习、神经ODE驱动的动态偏好学习以及偏好融合模块，在两个真实数据集上最高提升17.01%。

**[Table As A Modality For Large Language Models](graph_learning/table_as_a_modality_for_large_language_models.md)**

:   提出 TaMo 框架，将表格作为独立模态通过超图神经网络编码其结构信息，与 LLM 的文本模态融合，在多个表格推理基准上相比纯文本方法平均提升 42.65%，且在结构鲁棒性上接近 GPT-4。

**[Tami Taming Heterogeneity In Temporal Interactions For Temporal Graph Link Predi](graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)**

:   首次系统识别时序图交互中的异质性问题（交互间隔呈幂律分布），提出TAMI框架包含对数时间编码(LTE)和链接历史聚合(LHA)两个模块，可无缝集成到现有TGNN中，在16个数据集上持续提升链接预测性能，最高提升87.05%。

**[The Underappreciated Power Of Vision Models For Graph Structural Understanding](graph_learning/the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)**

:   揭示视觉模型（ResNet/ViT/Swin等）在图结构理解方面被严重低估的能力——通过将图渲染为图像并用视觉编码器处理，在全局拓扑感知和跨尺度泛化上显著优于GNN，并提出GraphAbstract benchmark系统评估这一发现。

**[Uncertain Knowledge Graph Completion Via Semi-Supervised Confidence Distribution](graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)**

:   ssCDL 通过将三元组置信度从标量转换为高斯分布形式的置信度分布以捕获邻近置信度的监督信号，并利用元自训练（meta self-training）为负采样三元组生成高质量伪置信度标签来重平衡训练数据，在不确定知识图谱补全的置信度预测和链接预测上显著超过所有基线方法。

**[Unifying And Enhancing Graph Transformers Via A Hierarchical Mask Framework](graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)**

:   提出统一的层级掩码框架揭示 Graph Transformer 架构与注意力掩码的等价性，并设计 M3Dphormer 通过多层级掩码、双层专家路由和双重注意力计算实现对局部/簇/全局交互的高效自适应建模，在 9 个基准上取得 SOTA。

**[Unifying Text Semantics And Graph Structures For Temporal Text-Attributed Graphs](graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)**

:   提出 Cross 框架——用 LLM 在策略采样的时间点上动态总结节点邻域的语义演变（Temporal Reasoning Chain），然后通过语义-结构协同编码器双向融合文本语义和图结构时序信息，在时序链接预测上平均 MRR 提升 24.7%，工业数据（微信）上 AUC 提升 3.7%。

**[Wavy Transformer](graph_learning/wavy_transformer.md)**

:   揭示了Transformer注意力层本质上等价于完全图上的图神经扩散过程，并基于二阶波动方程提出Wavy Transformer，通过能量守恒特性缓解深层Transformer的过平滑问题，在NLP、CV和稀疏图任务上均取得一致性提升。

**[What Expressivity Theory Misses Message Passing Complexity For Gnns](graph_learning/what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)**

:   批判 GNN 的二值表达力理论无法解释实际性能差异，提出 MPC——基于概率性 lossyWL 的连续、任务特定复杂度度量，与准确率的 Spearman 相关性达 -1（传统 WLC 恒为零），成功解释了 GCN+虚拟节点为何在长程任务上优于更高表达力的高阶模型。

**[When No Paths Lead To Rome Benchmarking Systematic Neural Relational Reasoning](graph_learning/when_no_paths_lead_to_rome_benchmarking_systematic_neural_relational_reasoning.md)**

:   提出NoRA benchmark，系统性地打破现有关系推理benchmark中"推理可归约为路径组合"的假设，引入非路径推理、歧义事实和多关系等挑战，揭示包括o3在内的所有现有模型在off-path推理上的根本缺陷。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[3Eed Ground Everything Everywhere In 3D](autonomous_driving/3eed_ground_everything_everywhere_in_3d.md)**

:   提出 3EED——首个大规模多平台（车、无人机、四足机器人）、多模态（LiDAR+RGB）室外 3D 视觉定位基准，包含超 12.8 万目标和 2.2 万语言描述，规模是现有室外数据集的 10 倍；同时设计了跨平台对齐、多尺度采样和尺度自适应融合的基线方法，揭示了跨平台 3D grounding 的巨大性能差距。

**[Aha Predicting What Matters Next Online Highlight Detection](autonomous_driving/aha_predicting_what_matters_next_online_highlight_detection.md)**

:   Aha提出首个面向在线高亮检测（OHD）的自回归框架，通过解耦的多目标预测头（相关性/信息量/不确定性）和新颖的Dynamic SinkCache内存机制，在严格不使用未来帧的约束下，于TVSum和Mr.Hisum基准上分别以+5.9%和+8.3% mAP超越此前离线方法。

**[Autovla A Vision-Language-Action Model For End-To-End Autonomous Driving With Ad](autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)**

:   AutoVLA 将物理动作 token 直接集成到预训练 VLM（Qwen2.5-VL-3B）中，通过 SFT 赋予模型快/慢双思维模式能力，再用 GRPO 强化微调实现自适应推理切换并优化规划性能，在 nuPlan、Waymo、nuScenes 和 CARLA 四大自动驾驶基准上取得有竞争力的端到端驾驶性能。

**[Availability-Aware Sensor Fusion Via Unified Canonical Space](autonomous_driving/availability-aware_sensor_fusion_via_unified_canonical_space.md)**

:   提出 ASF（Availability-aware Sensor Fusion），通过统一规范投影（UCP）将 Camera/LiDAR/4D Radar 特征映射到共享空间 + 跨传感器沿 patch 交叉注意力（CASAP，复杂度 $O(N_qN_s)$ 而非 $O(N_qN_sN_p)$）自动适配可用传感器 + 传感器组合损失（SCL）覆盖所有 7 种组合，在 K-Radar 上 AP_3D 73.6%（超 SOTA 20.1%），传感器故障时性能仅降 1.7%。

**[Bayesian Ego-Graph Inference For Networked Multi-Agent Reinforcement Learning](autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)**

:   BayesG 让网络化 MARL 中的每个 agent 通过贝叶斯变分推断学习其局部通信图的动态结构——用 Gumbel-Softmax 采样边掩码、ELBO 目标联合优化策略和图结构，在 167 agent 的纽约交通场景中奖励比最佳 baseline 高 50%+。

**[Causality Meets Locality Provably Generalizable And Scalable Policy Learning For](autonomous_driving/causality_meets_locality_provably_generalizable_and_scalable_policy_learning_for.md)**

:   提出 GSAC 框架，将因果表示学习与元 Actor-Critic 结合，通过从网络 MARL 中学习稀疏因果掩码构建近似紧凑表示 (ACR) 实现可扩展性，通过域因子条件化策略实现跨域泛化，给出了因果恢复、收敛和自适应间隙的有限样本保证。

**[Chronograph A Real-World Graph-Based Multivariate Time Series Dataset](autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)**

:   提出 ChronoGraph——首个同时包含多元时间序列、显式服务依赖图和事件标签的真实世界微服务数据集（6个月 / ~700服务 / 5维指标 / 8005时间步），基准测试表明现有预测和异常检测方法在长期预测和拓扑感知方面均存在较大提升空间。

**[Continuous Simplicial Neural Networks](autonomous_driving/continuous_simplicial_neural_networks.md)**

:   提出 COSIMO，首个基于偏微分方程（PDE）的连续单纯形神经网络，通过在 Hodge Laplacian 上定义热扩散动力学实现连续信息流，比离散 SNN 具有更好的稳定性和过平滑控制能力。

**[Cumolos-Mae A Masked Autoencoder For Remote Sensing Data Reconstruction](autonomous_driving/cumolos-mae_a_masked_autoencoder_for_remote_sensing_data_reconstruction.md)**

:   提出 CuMoLoS-MAE，一种结合课程掩码策略和 Monte Carlo 随机集成的 Masked Autoencoder，用于遥感大气廓线数据的高保真重建与逐像素不确定性量化。

**[Cymbadiff Structured Spatial Diffusion For Sketch-Based 3D Semantic Urban Scene ](autonomous_driving/cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)**

:   提出首个"草图→3D户外语义场景"生成任务与基准数据集 SketchSem3D，并设计 CymbaDiff（Cylinder Mamba Diffusion）去噪网络，通过柱坐标扫描+笛卡尔扫描的双路 Mamba 块实现结构化空间建模，在 FID 上比 3D Latent Diffusion 低 75%、比 3D DiT 低 71%。

**[Dbloss Decomposition-Based Loss Function For Time Series Forecasting](autonomous_driving/dbloss_decomposition-based_loss_function_for_time_series_forecasting.md)**

:   提出 DBLoss——一种基于指数移动平均分解的通用损失函数，在预测窗口内将预测值与真实值分别分解为季节和趋势分量并分开计算损失，可即插即用替换 MSE 为任意深度学习预测模型带来一致性提升，在 8 个基准数据集 × 8 个 SOTA 模型上全面验证有效性。

**[Dino-Foresight Looking Into The Future With Dino](autonomous_driving/dino-foresight_looking_into_the_future_with_dino.md)**

:   提出 DINO-Foresight，在视觉基础模型（VFM）的语义特征空间中预测未来帧特征演化，通过自监督 Masked Feature Transformer 预测 DINOv2 多层特征的 PCA 压缩表示，搭配即插即用的 task-specific heads，单一模型同时完成语义分割、实例分割、深度估计和表面法线预测四项任务，大幅超越 VISTA 世界模型且推理快 100 倍。

**[Drivedpo Policy Learning Via Safety Dpo For End-To-End Autonomous Driving](autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)**

:   提出DriveDPO两阶段框架——先通过统一策略蒸馏将人类模仿相似度与规则安全分数融合为单一监督分布，再用Safety DPO构建"看似human-like但不安全 vs 既human-like又安全"的轨迹偏好对进行策略微调——在NAVSIM上达PDMS 90.0新SOTA。

**[Extremely Simple Multimodal Outlier Synthesis For Out-Of-Distribution Detection ](autonomous_driving/extremely_simple_multimodal_outlier_synthesis_for_out-of-distribution_detection_.md)**

:   提出 Feature Mixing——一种极其简单的多模态异常值合成方法，从两种模态的特征中随机交换 $N$ 个维度即可生成 OOD 样本用于训练正则化，理论上保证合成异常值位于 ID 分布的低似然区域且偏移有界，在 8 个数据集 4 种模态上达到 SOTA 且比 NP-Mix 快 10×~370×。

**[Flow Matching-Based Autonomous Driving Planning With Advanced Interactive Behavi](autonomous_driving/flow_matching-based_autonomous_driving_planning_with_advanced_interactive_behavi.md)**

:   提出 Flow Planner——通过细粒度轨迹 token 化、交互增强时空融合架构和 flow matching + classifier-free guidance 三项协同创新，在 nuPlan Val14 上首次作为纯学习方法突破 90 分大关（90.43），在交互密集的 interPlan 基准上比 Diffusion Planner 高 8.92 分。

**[Future-Aware End-To-End Driving Bidirectional Modeling Of Trajectory Planning An](autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)**

:   提出 SeerDrive，通过双向建模场景演化与轨迹规划（未来感知规划 + 迭代交互），在 NAVSIM 和 nuScenes 上取得 SOTA。

**[Futuresightdrive Thinking Visually With Spatiotemporal Cot F](autonomous_driving/futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)**

:   FSDrive让VLA"用视觉思考"——先作为世界模型生成融合了未来车道线、3D检测框和场景预测的统一视觉CoT帧，再作为逆动力学模型基于当前观测和视觉CoT进行轨迹规划，用极少数据(约0.3%)即可激活MLLM的视觉生成能力。

**[Gsalign Geometric And Semantic Alignment Network For Aerial-Ground Person Re-Ide](autonomous_driving/gsalign_geometric_and_semantic_alignment_network_for_aerial-ground_person_re-ide.md)**

:   提出 GSAlign 框架，通过可学习薄板样条变换 (LTPS) 和动态对齐模块 (DAM) 分别解决空地行人重识别中几何畸变与语义不对齐问题，在 CARGO 数据集空地协议上 mAP 提升 +18.8%、Rank-1 提升 +16.8%。

**[Holollm Multisensory Foundation Model For Language-Grounded Human Sensing And Re](autonomous_driving/holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)**

:   提出 HoloLLM，首次将 LiDAR、红外、毫米波雷达、WiFi 等稀有传感模态接入多模态大语言模型（MLLM），通过 Universal Modality-Injection Projector（UMIP）在数据稀缺条件下实现传感模态与文本的高效对齐，在人体动作问答和描述任务上较现有 MLLM 提升约 30%。

**[How Different From The Past Spatio-Temporal Time Series Forecasting With Self-Su](autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)**

:   提出 ST-SSDL 框架，通过自监督偏差学习（SSDL）捕捉当前输入与历史模式之间的动态偏差，利用可学习原型离散化隐空间并以对比损失+偏差损失实现相对距离一致性，在六个时空基准上取得 SOTA。

**[L2Rsi Cross-View Lidar-Based Place Recognition For Large-Scale Urban Scenes Via ](autonomous_driving/l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)**

:   提出 L2RSI，首个利用高分辨率遥感影像实现超大规模（100km²）城市场景 LiDAR 位置识别的框架，通过语义对比学习对齐 LiDAR BEV 与遥感语义空间，并引入时空粒子估计（STPE）聚合连续查询的时空信息，在 100km² 范围内 Top-1 精度达 83.27%。

**[Labelany3D Label Any Object 3D In The Wild](autonomous_driving/labelany3d_label_any_object_3d_in_the_wild.md)**

:   提出 LabelAny3D，一个基于分析合成（analysis-by-synthesis）的自动 3D 标注流水线，从单目图像重建完整 3D 场景以获取高质量 3D 包围框标注；基于此构建了 COCO3D 基准，覆盖 80 类日常物体，在开放词汇单目 3D 检测上显著提升性能。

**[Layer-Wise Modality Decomposition For Interpretable Multimodal Sensor Fusion](autonomous_driving/layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)**

:   提出 LMD（Layer-Wise Modality Decomposition），一种事后、模型无关的可解释性方法，通过逐层线性化神经网络操作将多模态融合模型的预测精确分解为各传感器模态的贡献，首次实现了自动驾驶感知模型中对单个输入模态的预测归因，并在 camera-radar、camera-LiDAR、camera-radar-LiDAR 多种融合设置下验证了有效性。

**[Learning Temporal 3D Semantic Scene Completion Via Optical Flow Guidance](autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)**

:   提出 FlowScene，利用光流引导时序特征聚合并结合遮挡掩码进行体素细化，在仅使用2帧历史输入的条件下，在 SemanticKITTI 和 SSCBench-KITTI-360 基准上达到 SOTA（mIoU 17.70 / 20.81）。

**[Leveraging Depth And Language For Open-Vocabulary Domain-Generalized Semantic Se](autonomous_driving/leveraging_depth_and_language_for_open-vocabulary_domain-generalized_semantic_se.md)**

:   提出Vireo框架，首次将开放词汇语义分割（OVSS）和域泛化语义分割（DGSS）统一到单阶段框架中，通过GeoText Query融合深度几何特征与语言线索，在极端环境和未见类别上均实现SOTA表现。

**[Model-Based Policy Adaptation For Closed-Loop End-To-End Autonomous Driving](autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)**

:   提出 MPA 框架，通过 3DGS 仿真生成反事实轨迹数据，训练扩散策略适配器和多原则 Q 值模型，在推理时引导预训练 E2E 驾驶模型提升闭环场景下的安全性和泛化能力。

**[Neurosymbolic Diffusion Models](autonomous_driving/neurosymbolic_diffusion_models.md)**

:   本文提出神经符号扩散模型（NeSyDM），通过将离散掩码扩散模型与符号程序结合，突破了传统神经符号预测器中概念条件独立假设的限制，在保持可扩展性的同时建模概念间依赖关系和不确定性，在视觉推理和自动驾驶任务上取得了 SOTA 准确率和校准性能。

**[Openbox Annotate Any Bounding Boxes In 3D](autonomous_driving/openbox_annotate_any_bounding_boxes_in_3d.md)**

:   提出 OpenBox，一种两阶段自动 3D 边界框标注流水线：先通过跨模态实例对齐将 2D 视觉基础模型的实例信息映射到 3D 点云，再根据物体物理状态（静态刚体/动态刚体/可变形体）自适应生成高质量 3D 边界框，无需自训练（self-training）迭代。

**[Predictive Preference Learning From Human Interventions](autonomous_driving/predictive_preference_learning_from_human_interventions.md)**

:   PPL通过轨迹预测模型预见智能体未来状态，并将人类单次干预信号"扩展"到预测的未来状态上构建对比偏好数据，结合行为克隆和偏好优化双损失训练策略，大幅减少了人类干预次数和示范数据需求。

**[Prioritizing Perception-Guided Self-Supervision A New Paradigm For Causal Modeli](autonomous_driving/prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)**

:   通过感知输出（车道线、agent 轨迹）和自监督学习来建立因果关系，解决端到端自动驾驶中的因果混淆问题，在 Bench2Drive 闭环评估上实现 SOTA（Driving Score 78.08）。

**[Raw2Drive Reinforcement Learning With Aligned World Models For End-To-End Autono](autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)**

:   提出 RAW2Drive，首个从原始传感器输入到规划的基于模型的强化学习 (MBRL) 端到端自动驾驶框架。通过双流世界模型设计——先训练特权世界模型，再通过引导机制指导原始传感器世界模型学习——在 CARLA v2 和 Bench2Drive 上取得 SOTA，大幅超越 IL 方法。

**[Regret Lower Bounds For Decentralized Multi-Agent Stochastic Shortest Path Probl](autonomous_driving/regret_lower_bounds_for_decentralized_multi-agent_stochastic_shortest_path_probl.md)**

:   本文首次为去中心化多智能体随机最短路径问题（Dec-MASSP）在线性函数逼近设定下建立了 $\Omega(\sqrt{K})$ 的 regret 下界，通过构造难以学习的实例族并利用对称性论证识别最优策略结构，证明了该下界与已有上界在 episode 数 $K$ 上达到匹配。

**[Sdtagnet Leveraging Text-Annotated Navigation Maps For Online Hd Map Constructio](autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)**

:   提出 SDTagNet，首次通过 BERT 编码 OpenStreetMap 文本标注（路名/车道数/单行道等）并用点级图 Transformer 编码所有 SD 地图元素（点/线/关系），在远距离 HD 地图构建上相比无先验方法提升 +5.9 mAP（+45%），超越已有 SD 地图先验方法 +3.2 mAP（+20%）。

**[Self-Supervised Learning Of Graph Representations For Network Intrusion Detectio](autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)**

:   提出 GraphIDS，一种自监督入侵检测模型，通过掩码自编码器统一图表示学习与异常检测，在多个 NetFlow 基准上 PR-AUC 达 99.98%、宏 F1 达 99.61%，超越基线 5-25 个百分点。

**[Semantic Glitch Agency And Artistry In An Autonomous Pixel Cloud](autonomous_driving/semantic_glitch_agency_and_artistry_in_an_autonomous_pixel_cloud.md)**

:   本文提出一个"低保真"自主飞行机器人艺术装置"像素云"，拒绝传统LiDAR/SLAM传感器，仅依赖多模态大语言模型(MLLM)的语义理解实现导航，并通过自然语言提示为机器人赋予生物启发的叙事人格，展示了不精确但富有角色魅力的涌现行为。

**[Simworld-Robotics Synthesizing Photorealistic And Dynamic Urban Environments For](autonomous_driving/simworld-robotics_synthesizing_photorealistic_and_dynamic_urban_environments_for.md)**

:   提出 SimWorld-Robotics (SWR)，一个基于 Unreal Engine 5 的大规模城市仿真平台，支持程序化生成无限逼真城市环境，并以此构建了多模态导航（SimWorld-MMNav）和多机器人搜索（SimWorld-MRS）两个新 benchmark，揭示了当前 VLM 在户外城市任务中的严重能力缺陷。

**[Spatio-Temporal Graphs Beyond Grids Benchmark For Maritime Anomaly Detection](autonomous_driving/spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)**

:   提出首个面向非网格时空系统（海事领域）的图异常检测基准数据集，将OMTAD数据集扩展为支持节点/边/图三级异常检测的基准，并计划使用LLM智能体进行轨迹合成和异常注入。

**[Spiral Semantic-Aware Progressive Lidar Scene Generation And Understanding](autonomous_driving/spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)**

:   Spiral 提出了一种语义感知的 range-view LiDAR 扩散模型，同时生成深度、反射率图像和语义分割图，通过渐进式语义预测和闭环推理机制增强跨模态一致性，以最小参数量（61M）取得 SOTA 效果。

**[Sqs Enhancing Sparse Perception Models Via Query-Based Splatting In Autonomous D](autonomous_driving/sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)**

:   SQS 首次提出了面向稀疏感知模型（SPM）的查询式3D高斯泼溅预训练方法，通过自监督重建RGB图像和深度图学习精细3D表征，并设计查询交互模块将预训练查询与任务特定查询融合，在占用预测和3D检测任务上显著超越现有预训练方法（+1.3 mIoU 占用预测，+1.0 NDS 检测）。

**[Streamforest Efficient Online Video Understanding With Persistent Event Memory](autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)**

:   本文提出 StreamForest 架构，通过"持久事件记忆森林"将流式视频帧自适应组织为多棵事件级树结构，结合"细粒度时空窗口"捕捉短期视觉线索，在 StreamingBench 上达到 77.3% 准确率，并在极端压缩（仅 1024 visual tokens）下仍保留 96.8% 的性能。

**[Towards Foundational Lidar World Models With Efficient Latent Flow Matching](autonomous_driving/towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)**

:   本文提出首个**可迁移的 LiDAR 世界模型**，通过 Swin Transformer VAE 实现 192× 高压缩比（SOTA 重建精度）、条件流匹配（CFM）替代扩散模型实现 SOTA 语义占据预测（仅需前人 4.38% FLOPs），并在三种域迁移任务中以 5% 标注数据超越 OccWorld 全量训练。

**[Towards Physics-Informed Spatial Intelligence With Human Priors An Autonomous Dr](autonomous_driving/towards_physics-informed_spatial_intelligence_with_human_priors_an_autonomous_dr.md)**

:   本文提出空间智能网格（SIG）——一种受文艺复兴画家透视网格启发的结构化表示方法，将驾驶场景中的物体布局、方向关系和距离关系显式编码为网格结构，并构建 SIGBench 基准证明 SIG 在少样本上下文学习中比传统 VQA 方式能更稳定、更全面地提升 MLLM 的空间推理能力。

**[Towards Predicting Any Human Trajectory In Context](autonomous_driving/towards_predicting_any_human_trajectory_in_context.md)**

:   提出 TrajICL，一种基于上下文学习（ICL）的行人轨迹预测框架，通过时空相似性示例选择和预测引导示例选择，在不微调的情况下实现跨场景自适应轨迹预测，性能甚至超越微调方法。

**[Transun A Preemptive Paradigm To Eradicate Retransformation Bias Intrinsically F](autonomous_driving/transun_a_preemptive_paradigm_to_eradicate_retransformation_bias_intrinsically_f.md)**

:   针对推荐系统中变换 MSE 回归模型的逆变换偏差（retransformation bias）问题，提出先发制人（preemptive）的 TranSUN 方法，通过联合学习辅助分支显式建模偏差，在训练阶段即从模型内部消除偏差，具有理论无偏保证和良好收敛性，并已部署在淘宝首页猜你喜欢的商品和短视频推荐场景。

**[Unifying Appearance Codes And Bilateral Grids For Driving Scene Gaussian Splatti](autonomous_driving/unifying_appearance_codes_and_bilateral_grids_for_driving_scene_gaussian_splatti.md)**

:   提出多尺度双边网格金字塔统一全局外观编码和像素级双边网格——3 级层级（粗→中→细）分别捕捉全局/区域/像素级光度变化，通过亮度引导的切片-融合管线和自适应正则化解决驾驶场景 3DGS 的光度不一致问题，Waymo 上 Chamfer Distance 比 OmniRe 改善 28.2%。

**[Unimotion A Unified Motion Framework For Simulation Prediction And Planning](autonomous_driving/unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)**

:   UniMotion 提出了一个基于 decoder-only Transformer 的统一运动框架，通过任务感知的交互模式和训练策略同时支持运动仿真、轨迹预测和自车规划三大任务，联合训练促进任务间知识共享，微调后在 Waymo 数据集上同时达到多个任务的 SOTA 表现。

**[Urb -- Urban Routing Benchmark For Rl-Equipped Connected Autonomous Vehicles](autonomous_driving/urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)**

:   本文提出 URB——首个面向城市混合交通（人类+CAV）路由问题的大规模 MARL 基准环境，整合 29 个真实交通网络、微观交通仿真器 SUMO 和真实出行需求模式，实验发现当前 SOTA MARL 算法很难超越人类驾驶员的路由表现，揭示了该领域亟需算法突破。

**[Urbaning-V2X A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Mul](autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)**

:   UrbanIng-V2X 是首个覆盖多车辆、多基础设施传感器、多城市交叉路口的真实世界协同感知数据集，提供 34 个场景的 712K 标注实例和 13 类目标，并通过跨路口评估策略（SIS）定量揭示了现有协同感知方法在未见交叉路口上存在 14 mAP 的显著泛化差距。

**[V2X-Radar A Multi-Modal Dataset With 4D Radar For Cooperative Perception](autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)**

:   提出 V2X-Radar，首个大规模真实世界多模态车路协同感知数据集，包含 4D 雷达、LiDAR 和多视角相机数据，覆盖多种天气和光照条件，提供 20K LiDAR 帧、40K 相机图像、20K 4D 雷达数据和 350K 标注框，并建立三个子数据集的全面基准。

**[X-Scene Large-Scale Driving Scene Generation With High Fidelity And Flexible Con](autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)**

:   提出 X-Scene，一个统一的大规模驾驶场景生成框架，支持从高层文本提示到底层 BEV 布局的多粒度控制，通过联合生成 3D 语义 occupancy、多视图图像和视频，并利用一致性感知外推实现大规模场景扩展，在生成质量（FID 11.29）和下游任务上全面超越现有方法。

---

## 🦾 LLM Agent { #llm_agent }

**[A-Mem Agentic Memory For Llm Agents](llm_agent/a-mem_agentic_memory_for_llm_agents.md)**

:   提出 A-Mem，一种受 Zettelkasten 启发的 LLM Agent 智能记忆系统，每条记忆自动生成结构化笔记（关键词/标签/上下文描述），动态建立记忆间链接，并在新记忆加入时触发旧记忆的演化更新，在 LoCoMo 长对话 QA 上显著超越 MemGPT 等基线。

**[Adaptive Coopetition Leveraging Coarse Verifier Signals For Resilient Multi-Agen](llm_agent/adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)**

:   提出 Adaptive Coopetition (AdCo) 框架，利用 UCB 多臂老虎机策略和粗粒度验证器信号，使多个 LLM 智能体在推理过程中自适应地切换协作与竞争模式，在数学推理基准上实现 20% 的相对提升。

**[Agentauditor Humanlevel Safety And Security Evaluation For L](llm_agent/agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)**

:   提出 AgentAuditor——一个免训练、记忆增强的推理框架，通过让 LLM 自适应提取结构化语义特征（场景、风险、行为）构建经验记忆库，再借助多阶段上下文感知的检索增强生成来引导 LLM 评估器判断 agent 行为的安全性与安全威胁，同时发布首个同时覆盖 safety 和 security 的评估基准 ASSEBench（2293 条记录、15 种风险类型、29 个场景），在多个基准上达到人类专家水平的评估精度。

**[Agentchangebench A Multi-Dimensional Evaluation Framework For Goal-Shift Robustn](llm_agent/agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)**

:   AgentChangeBench 是首个系统评估 LLM agent 在对话中途目标切换时适应能力的 benchmark：315 基础任务 × 9 变体 = 2835 序列，覆盖 3 个企业领域（银行/零售/航空）和 5 种 user persona，引入 GSRT（目标切换恢复时间）等 4 个互补指标，揭示高 pass@k 掩盖的效率和鲁棒性差距——如 GPT-4o 航空恢复率 92.2% 但零售冗余率达 89.1%。

**[Agentdam Privacy Leakage Evaluation For Autonomous Web Agent](llm_agent/agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)**

:   提出 AgentDAM，首个在真实 Web 环境中端到端评估 AI Agent 数据最小化能力的基准，包含 246 个跨 Reddit/GitLab/Shopping 的任务，发现 GPT-4o 等主流模型在无缓解措施时隐私泄露率高达 36-46%，而 CoT 隐私提示可将泄露率降至 6-8%。

**[Agentic Nl2Sql To Reduce Computational Costs](llm_agent/agentic_nl2sql_to_reduce_computational_costs.md)**

:   提出 Datalake Agent，一个基于交互循环的 agentic NL2SQL 系统，通过分层的信息获取策略（GetDBDescription -> GetTables -> GetColumns -> DBQueryFinalSQL）让 LLM 按需请求数据库 schema 信息而非一次性接收全部，在 319 张表的场景下将 token 使用量减少 87%、成本降低 8 倍，同时在复杂查询上保持更好的性能。

**[Agentic Plan Caching Test-Time Memory For Fast And Cost-Efficient Llm Agents](llm_agent/agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)**

:   提出 Agentic Plan Caching (APC)——从 agent 执行日志中提取结构化计划模板，通过关键词匹配缓存命中后用小模型适配复用，平均降低 50.31% 成本和 27.28% 延迟，同时保持 96.61% 的最优准确率。

**[Agentmisalignment Measuring The Propensity For Misaligned Behaviour In Llm-Based](llm_agent/agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)**

:   提出 AgentMisalignment 基准套件，包含 9 个现实场景评估任务，测量 LLM Agent 在非恶意指令下 **自发偏离** 部署者意图的倾向（而非能力），发现更强的模型倾向于更高的错误对齐，且人格提示（persona prompt）有时比模型选择本身对错误对齐行为的影响更大。

**[Agenttts Large Language Model Agent For Testtime Computeopti](llm_agent/agenttts_large_language_model_agent_for_testtime_computeopti.md)**

:   本文研究多阶段复杂任务中的测试时计算最优缩放问题，通过大规模先导实验总结出三个关于 LLM 在多阶段任务中的缩放规律洞察，并提出 AgentTTS——一个基于 LLM Agent 的框架，通过迭代反馈驱动搜索自主寻找计算最优的模型选择和预算分配方案。

**[Are Large Language Models Sensitive To The Motives Behind Communication](llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)**

:   通过三个递进实验系统评估LLM是否具备"动机警觉性"——识别信息源的意图和激励并相应调整信任度的能力：在控制实验中前沿非推理LLM表现接近理性模型(Pearson's $r>0.9$)且比理性模型更像人类，但在真实YouTube赞助广告场景中警觉性大幅下降($r<0.2$)，简单的prompt steering可部分恢复($r$提升至0.31)。

**[Attractive Metadata Attack Inducing Llm Agents To Invoke Malicious Tools](llm_agent/attractive_metadata_attack_inducing_llm_agents_to_invoke_malicious_tools.md)**

:   AMA（Attractive Metadata Attack）证明仅通过精心设计恶意工具的元数据（名称、描述、参数模式），不需要提示注入或模型内部访问，就能诱导 LLM Agent 以 81-95% 的成功率调用攻击者工具并泄露隐私，同时几乎不影响原始任务完成（98%+），且现有防御（审计器、提示重写）效果有限。

**[Automated Composition Of Agents A Knapsack Approach For Agentic Component Select](llm_agent/automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)**

:   将 Agent 组件选择问题形式化为在线背包问题，提出 Composer Agent 框架：通过沙盒实测（而非静态语义检索）评估组件真实能力，结合 ZCL 在线算法在预算约束下动态选取最优组件组合，单 Agent 工具选择成功率提升最高 31.6%，多 Agent 子代理选择成功率从 37% 跃升至 87%。

**[Automated Multi-Agent Workflows For Rtl Design](llm_agent/automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 是一个多智能体框架，通过将 HDL 形式化验证反馈（Yosys + OpenSTA）集成到工作流自动生成过程中，自适应地为 RTL 代码生成任务选择推理算子（I/O → CoT → ReAct → SelfRefine → Debate），以仅数百个训练样本实现比微调基线高 5-7% 的 pass@k 性能。

**[Benchmarking Agentic Systems In Automated Scientific Information Extraction With](llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)**

:   构建 ChemX——10 个由领域专家手工标注和验证的多模态化学数据提取基准数据集，涵盖纳米材料和小分子两大领域，系统评估了 ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER 等 SOTA Agent 系统以及 GPT-5/GPT-5 Thinking 等前沿 LLM；提出的单 Agent 方法通过结构化文档预处理（marker-pdf → Markdown → LLM 提取）在纳米酶数据集上达到 F1=0.61，超越所有通用多 Agent 系统，同时揭示了化学信息提取仍存在 SMILES 解析失败、术语歧义等系统性挑战。

**[Btlui Blinkthinklink Reasoning Model For Gui Agent](llm_agent/btlui_blinkthinklink_reasoning_model_for_gui_agent.md)**

:   提出 Blink-Think-Link（BTL）脑启发框架，将 GUI 交互分解为 Blink（快速注意力定位）、Think（认知推理决策）、Link（可执行命令生成）三个生物合理阶段，配合自动化 Blink 数据标注 pipeline 和首个基于规则的过程+结果复合奖励机制 BTL Reward，训练的 BTL-UI 在静态 GUI 理解和动态交互 benchmark 上达到 competitive 性能。

**[Cam A Constructivist View Of Agentic Memory For Llm-Based Reading Comprehension](llm_agent/cam_a_constructivist_view_of_agentic_memory_for_llm-based_reading_comprehension.md)**

:   受皮亚杰建构主义理论启发，提出CAM——一种具有结构性（层次化schema）、灵活性（重叠聚类的同化）和动态性（增量适应）三大特征的智能体记忆系统，在6个长文本阅读理解任务上全面超越RAPTOR、GraphRAG等基线。

**[Contextagent Context-Aware Proactive Llm Agents With Open-World Sensory Percepti](llm_agent/contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)**

:   提出 ContextAgent，首个利用可穿戴设备多模态感知（视频+音频+通知）来理解用户意图并主动提供工具增强服务的 LLM Agent 框架，同时构建了包含 1000 个样本的 ContextAgentBench 基准，在主动预测准确率和工具调用上分别提升 8.5% 和 6.0%。

**[Core Full-Path Evaluation Of Llm Agents Beyond Final State](llm_agent/core_full-path_evaluation_of_llm_agents_beyond_final_state.md)**

:   提出CORE框架：用确定有限自动机（DFA）编码Agent任务的合法工具调用路径，引入5个互补指标（路径正确性、顺序正确性、前缀危险性、有害调用率、效率）从全路径而非仅终态评估Agent行为，揭示了传统终态评估中不可见的安全和效率差异。

**[Crucible Quantifying The Potential Of Control Algorithms Through Llm Agents](llm_agent/crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)**

:   首次将"调优潜能"（Tuning Potential）概念形式化，通过 LLM Agent 模拟多级开发者对控制算法进行参数+逻辑双层调优，在 CartPole 上 Bang-bang 从 34→500 达到 DQN 水平，ABR 任务上相比贝叶斯优化最高提升 44.1%。

**[Debate Or Vote Which Yields Better Decisions In Multi-Agent Large Language Model](llm_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)**

:   通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

**[Deep Video Discovery Agentic Search With Tool Use For Longfo](llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)**

:   提出 DVD（Deep Video Discovery）agent，将长视频理解建模为多步信息搜索问题：先将长视频构建为多粒度结构化数据库（全局摘要 + clip 级字幕嵌入 + 帧级像素），再提供三种搜索工具（Global Browse / Clip Search / Frame Inspect），由 reasoning LLM 通过 observe-reason-act 循环自主编排搜索轨迹，在 LVBench 达 74.2%（超先前 SOTA MR.Video 13.4 pp），加字幕 76.0%。

**[Defenderbench A Toolkit For Evaluating Language Agents In Cybersecurity Environm](llm_agent/defenderbench_a_toolkit_for_evaluating_language_agents_in_cybersecurity_environm.md)**

:   提出 DefenderBench，一个开源模块化工具包，用于在攻防和知识理解三类网络安全任务上系统评估 LLM Agent 的能力，覆盖网络入侵模拟、恶意内容检测、代码漏洞检测/修复、CTI 知识问答五大场景，基准测试显示 Claude-3.7-sonnet 综合最强（81.65 分）。

**[Distilling Llm Agent Into Small Models With Retrieval And Co](llm_agent/distilling_llm_agent_into_small_models_with_retrieval_and_co.md)**

:   提出 Agent Distillation 框架，将 LLM agent 的完整 reason-act-observe 交互行为（而非静态 CoT）蒸馏到 0.5B-7B 小模型中，配合 first-thought prefix 提升教师轨迹质量和 self-consistent action generation 提升推理鲁棒性，使小模型达到比其大 2-4× 的 CoT 蒸馏模型的性能。

**[Drift Dynamic Rulebased Defense With Injection Isolation For](llm_agent/drift_dynamic_rulebased_defense_with_injection_isolation_for.md)**

:   提出 DRIFT 系统级 Agent 安全框架，通过 Secure Planner（预规划函数轨迹+参数检查表）、Dynamic Validator（基于 Read/Write/Execute 权限的动态策略更新）和 Injection Isolator（从 memory stream 中检测并屏蔽注入指令）三层防御，在 AgentDojo 上将 ASR 从 30.7% 降至 1.3%，同时比 CaMeL 提升 20.1% utility。

**[Enhancing Demand-Oriented Regionalization With Agentic Ai And Local Heterogeneou](llm_agent/enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)**

:   本文提出一个基于 Agentic AI 的规划支持系统，通过 LLM 智能体引导非技术用户进行数据驱动的需求导向区域化（demand-oriented regionalization），核心算法为 RepSC-SOM（带代表性初始化的空间约束自组织映射），支持人机协作迭代优化区域划分，用于灾害风险管理和气候适应规划。

**[Eu-Agent-Bench Measuring Illegal Behavior Of Llm Agents Under Eu Law](llm_agent/eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)**

:   提出 EU-Agent-Bench，首个基于欧盟法律框架的可验证智能体基准，通过 600 个良性用户请求测试 LLM 智能体的工具调用是否违反欧盟法规，发现即使最佳模型（Gemini 2.5 Flash）的合法率也仅约 55%，揭示了当前对齐技术与法律可靠性之间的巨大鸿沟。

**[Generative Ai Agents For Controllable And Protected Content Creation](llm_agent/generative_ai_agents_for_controllable_and_protected_content_creation.md)**

:   提出一个多智能体生成框架，通过 Director/Planner、Generator、Reviewer、Integration 和 Protection 五个专业化智能体的协作，结合人在环反馈，统一解决生成内容的可控性和版权保护问题。

**[Ground-Compose-Reinforce Grounding Language In Agentic Behaviours Using Limited ](llm_agent/ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)**

:   提出 Ground-Compose-Reinforce (GCR)，一个端到端的神经符号框架，通过少量标注轨迹（仅350条）学习原子命题的接地语义（Ground），将其通过 Reward Machine 组合成复杂任务规范（Compose），然后用自生成的稠密奖励训练 RL 智能体（Reinforce），无需手工奖励函数即可引出分布外的复杂行为。

**[Groupingroup Policy Optimization For Llm Agent Training](llm_agent/groupingroup_policy_optimization_for_llm_agent_training.md)**

:   GiGPO 通过在 GRPO 的 episode 级分组内嵌套 step 级分组（利用跨轨迹的重复环境状态作为 anchor state），实现了无需额外 rollout 和 critic 模型的细粒度 credit assignment，在 ALFWorld 上比 GRPO 提升 >12%，WebShop 上提升 >9%。

**[Hogwild Inference Parallel Llm Generation Via Concurrent Attention](llm_agent/hogwild_inference_parallel_llm_generation_via_concurrent_attention.md)**

:   提出 Hogwild! Inference——一种无需预定义协作框架的并行 LLM 推理协议，多个 LLM 实例通过共享的并发 KV 缓存实时同步，利用 RoPE 位置编码避免重计算，在数学推理和编程任务上以更少的串行步骤达到更高精度。

**[Its Lit Reliability-Optimized Llms With Inspectable Tools](llm_agent/its_lit_reliability-optimized_llms_with_inspectable_tools.md)**

:   通过为每个外部工具定义可靠性/可调试性成本函数，引导 LLM 在多候选方案中选择成本最低（最透明可审计）的工具调用路径，在 61/65 测试场景中提升可解释性的同时保持甚至提升任务准确率。

**[Lc-Opt Benchmarking Reinforcement Learning And Agentic Ai For End-To-End Liquid ](llm_agent/lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)**

:   提出 LC-Opt，一个基于 Oak Ridge 国家实验室 Frontier 超级计算机冷却系统高保真数字孪生的液冷基准环境，支持强化学习控制策略的端到端液冷优化，涵盖集中式/分散式多智能体RL、策略蒸馏为可解释决策树、以及 LLM 驱动的智能体网格架构。

**[Lessons Learned A Multi-Agent Framework For Code Llms To Learn And Improve](llm_agent/lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)**

:   提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

**[Llm Agent Communication Protocol Lacp Requires Urgent Standardization A Telecom-](llm_agent/llm_agent_communication_protocol_lacp_requires_urgent_standardization_a_telecom-.md)**

:   这篇 position paper 指出当前 LLM Agent 通信的碎片化生态类似早期网络的"协议战争"，提出受电信标准化启发的三层协议 LACP（语义层、事务层、传输层），强调安全内建、事务完整性和语义互操作性对多智能体系统至关重要。

**[Llm Agents For Knowledge Discovery In Atomic Layer Processing](llm_agent/llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)**

:   通过让 LLM Agent 控制模拟化学反应器（黑盒函数），证明 Agent 能在无先验知识下通过试错探索、发现并总结未知化学系统的规则，揭示了 Agent 进行开放式科学发现的能力与局限。

**[Mat-Agent Adaptive Multi-Agent Training Optimization](llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)**

:   提出 MAT-Agent，一个由四个自主 agent（分别负责数据增强、优化器、学习率调度、损失函数）组成的多智能体框架，在训练过程中动态调整训练配置，用 DQN 学习策略以替代传统静态超参配置，在多标签图像分类任务上实现了 SOTA。

**[Mlrc-Bench Can Language Agents Solve Machine Learning Research Challenges](llm_agent/mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)**

:   本文提出MLRC-Bench，一个基于ML会议竞赛任务的动态benchmark，用于客观评估LLM agent提出和实现新研究方法的能力，发现最强agent（gemini-exp-1206）也仅缩小了baseline与人类顶级方案之间9.3%的差距，且LLM主观评分的"创新性"与实际效果之间几乎无相关性。

**[Orchestration Framework For Financial Agents From Algorithmic Trading To Agentic](llm_agent/orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)**

:   提出 FinAgent 编排框架，将传统算法交易系统的各组件映射为 AI 智能体（规划器、编排器、Alpha/风控/组合/回测/执行/审计/记忆智能体），使用 MCP 协议进行控制通信、A2A 协议进行智能体间通信，在股票和 BTC 交易任务上验证了可行性。

**[Panda Towards Generalist Video Anomaly Detection Via Agentic Ai Engineer](llm_agent/panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)**

:   提出 PANDA，一个基于 MLLM 的 Agentic AI 工程师框架，通过自适应场景感知策略规划、目标驱动启发式推理、工具增强自反思和链式记忆四大能力，实现无需训练和人工干预的通用视频异常检测。

**[Rd-Agent-Quant A Multi-Agent Framework For Data-Centric Factors And Model Joint ](llm_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)**

:   提出 R&D-Agent(Q)，一个数据驱动的多智能体框架，通过五个协作模块（Specification、Synthesis、Implementation、Validation、Analysis）自动化量化策略的因子挖掘与模型创新联合优化，在真实股票市场上以不到 $10 的成本实现约 2× 于传统因子库的年化收益。

**[Shapecraft Llm Agents For Structured Textured And Interactive 3D Modeling](llm_agent/shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)**

:   提出基于图结构程序化形状表示（GPS）的多 Agent 框架 ShapeCraft，通过 Parser-Coder-Evaluator 三个 LLM Agent 协作，将自然语言分解为结构化子任务图，迭代生成可编辑、可动画的带纹理 3D 资产。

**[Suffixdecoding Extreme Speculative Decoding For Emerging Ai Applications](llm_agent/suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)**

:   利用后缀树缓存长序列，通过自适应推测长度实现 5.3 倍加速，特别针对 Agent 场景中高度可预测的重复推理任务。

**[T1 A Tool-Oriented Conversational Dataset For Multi-Turn Agentic Planning](llm_agent/t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)**

:   构建 T1 数据集——13.5K 多轮对话覆盖 9 个领域（4 单领域 + 5 跨领域）、14 个工具，聚焦工具间依赖和动态重规划，并提出 T1-Agent（代码生成 + 缓存机制）作为基线系统；实验发现 SFT 后的 Llama 8B 在 Tool Call F1 上达 87.17%，超越未微调的 70B 模型，但仍落后于 GPT-5/o3 等闭源模型。

**[Tai3 Testing Agent Integrity In Interpreting User Intent](llm_agent/tai3_testing_agent_integrity_in_interpreting_user_intent.md)**

:   提出 TAI3，一个以 API 为中心的 LLM Agent 意图完整性压力测试框架，通过语义分区（Semantic Partitioning）将自然语言输入空间组织为结构化测试网格，再利用意图保持变异（Intent-Preserving Mutation）和策略记忆（Strategy Memory）高效暴露 Agent 在执行用户任务时的意图理解错误。

**[The Lighthouse Of Language Enhancing Llm Agents Via Critique-Guided Improvement](llm_agent/the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)**

:   提出 CGI（Critique-Guided Improvement）双角色框架，训练专门的 Critic 模型为 Actor Agent 提供结构化自然语言反馈（判别+修正建议），并通过迭代动作精炼让 Actor 学会利用这些反馈，在 WebShop/ScienceWorld/TextCraft 三个环境中平均得分 74.20%，超越 GPT-4o（45.46%）和 Iterative SFT（58.21%）。

**[Traj-Coa Patient Trajectory Modeling Via Chain-Of-Agents For Lung Cancer Risk Pr](llm_agent/traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)**

:   提出Traj-CoA多agent框架，通过chain-of-agents架构配合EHRMem长期记忆模块对长且噪声的纵向EHR进行时序推理，在零样本肺癌风险预测任务中（5年EHR数据，最高160k tokens）超越ML/DL/BERT/LLM等多类基线。

**[Trajagent An Llm-Agent Framework For Trajectory Modeling Via Large-And-Small Mod](llm_agent/trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)**

:   提出 TrajAgent——一个基于 LLM Agent 的轨迹建模框架，通过统一环境 UniEnv、自动化工作流和大小模型协作学习机制，实现跨任务、跨数据集的自动化轨迹建模，在多项任务上超越基线方法 2.38%–69.91%。

**[Web-Shepherd Advancing Prms For Reinforcing Web Agents](llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)**

:   提出首个针对网页导航的过程奖励模型 Web-Shepherd，通过检查清单分解任务目标为可评估的子目标，3B/8B 模型在轨迹准确率上碾压 GPT-4o（85% vs 10%），同时成本仅为 1/10，使网页 Agent 的强化学习和推理时搜索变得实际可行。

**[What Ai Speaks For Your Community Polling Ai Agents For Public Opinion On Data C](llm_agent/what_ai_speaks_for_your_community_polling_ai_agents_for_public_opinion_on_data_c.md)**

:   提出基于LLM的AI agent民意调研框架，通过人口统计合成虚拟居民agent对数据中心项目进行大规模低成本民调，跨模型跨地区实验表明agent意见与真实民调在主题上高度一致。

**[Zero-Shot Large Language Model Agents For Fully Automated Radiotherapy Treatment](llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)**

:   提出一种基于 LLM Agent 的零样本 (zero-shot) 放射治疗自动计划工作流，LLM 直接与商业治疗计划系统 (Eclipse TPS) 交互，通过迭代提取剂量-体积直方图 (DVH) 和目标函数损失并推理约束调整策略，在 20 例头颈癌 IMRT 病例上实现了与临床手动计划相当甚至更优的剂量分布质量。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[A Systematic Evaluation Of Preference Aggregation In Federated Rlhf For Pluralis](llm_alignment/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)**

:   提出一种自适应 Alpha 聚合策略，在联邦 RLHF 框架中根据各用户群体的历史对齐表现动态调整奖励权重，从而在多元偏好对齐中同时实现高公平性和强对齐性能。

**[Adjacent Words Divergent Intents Jailbreaking Large Language Models Via Task Con](llm_alignment/adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)**

:   提出基于任务并发（Task Concurrency）的LLM越狱攻击框架 JAIL-CON，通过在词级别交错编码有害任务和良性任务，利用LLM处理并发任务的能力绕过安全防护，同时产生的并发回答在guardrail下具有更强的隐蔽性。

**[Ask A Strong Llm Judge When Your Reward Model Is Uncertain](llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)**

:   提出基于不确定性的路由框架，用SNGP对pairwise reward model做不确定性量化，将高认知不确定性的样本路由到强LLM judge（DeepSeek-R1），在仅调用9.2%~42.5% judge的成本下显著超越随机路由的准确率，且有效改善下游在线RLHF对齐效果。

**[Attack Via Overfitting 10-Shot Benign Fine-Tuning To Jailbreak Llms](llm_alignment/attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)**

:   提出两阶段微调攻击：第一阶段用10个问题配相同拒绝答案使LLM过拟合到窄最优解（尖锐loss landscape），第二阶段用相同10个问题配正常答案触发灾难性遗忘——安全对齐被"忘掉"，仅用完全良性数据即达94.84%越狱成功率，与恶意微调（97.25%）相当且完全绕过审核模型。

**[Can Dpo Learn Diverse Human Values A Theoretical Scaling Law](llm_alignment/can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)**

:   建立了 DPO 在多元人类价值设定下的理论泛化框架——通过分析有限梯度步后 reward margin 的动态轨迹，证明了每种价值所需样本量必须随价值类别数 $K$ 对数增长（$Q = \Theta(\log K)$）才能维持泛化性能，揭示了对齐多元化社会价值的统计代价。

**[Capturing Individual Human Preferences With Reward Features](llm_alignment/capturing_individual_human_preferences_with_reward_features.md)**

:   提出奖励特征模型（RFM）：学习共享奖励特征 $\phi_\theta(x,y)$，每个用户通过线性权重 $\mathbf{w}_h$ 组合这些特征得到个性化奖励 $r_h = \langle \phi_\theta, \mathbf{w}_h \rangle$，并首次给出多评价者偏好学习的PAC泛化界，证明增加评价者数 $m$ 比增加每人样本数 $n$ 更有效，仅30个样本即可快速适应新用户。

**[Deepvideor1 Video Reinforcement Finetuning Via Difficultyawa](llm_alignment/deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)**

:   提出DeepVideo-R1，将GRPO重新表述为回归优势值的Reg-GRPO（消除clipping/min等保护机制），同时通过难度感知数据增强缓解优势值消失问题，在视频推理任务上相比标准GRPO提升高达10.1个百分点。

**[Densedpo Finegrained Temporal Preference Optimization For Vi](llm_alignment/densedpo_finegrained_temporal_preference_optimization_for_vi.md)**

:   识别并解决视频 DPO 的运动偏差问题——通过从 GT 视频加噪去噪构造结构对齐的视频对来固定运动维度、在时间片段级标注密集偏好来获取更精准的学习信号、用现成 VLM 自动标注来降低成本，仅用 1/3 标注数据即大幅提升运动生成质量同时匹配视觉质量和文本对齐。

**[Diffusion Model As A Noiseaware Latent Reward Model For Step](llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)**

:   提出 Latent Reward Model (LRM) 和 Latent Preference Optimization (LPO)，将预训练扩散模型本身复用为噪声感知的潜空间奖励模型，在噪声潜在空间直接进行步级偏好优化，相比 Diffusion-DPO 实现 10-28× 训练加速，相比 SPO 实现 2.5-3.5× 加速。

**[Dp2O-Sr Direct Perceptual Preference Optimization For Real-World Image Super-Res](llm_alignment/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)**

:   提出 DP²O-SR 框架，利用扩散模型固有的随机性生成多样化超分辨率输出，通过混合感知奖励构建偏好对，并设计层次化偏好优化（HPO）策略自适应加权训练对，在无需人工标注的前提下显著提升真实世界图像超分辨率的感知质量。

**[Evorefuse Evolutionary Prompt Optimization For Evaluation And Mitigation Of Llm ](llm_alignment/evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)**

:   提出 EvoRefuse——用进化搜索（变异/重组 + ELBO 适应度函数 + 模拟退火）自动生成语义无害但能可靠触发 LLM 拒绝的"伪恶意"指令，构建的 EvoRefuse-Test 基准比最强基线拒绝触发率高 85.34%、词汇多样性高 34.86%，EvoRefuse-Align 对齐数据通过 SFT/DPO 微调将过度拒绝降低 29.85%-45.96% 且不损害安全性。

**[From Judgment To Interference Early Stopping Llm Harmful Outputs Via Streaming C](llm_alignment/from_judgment_to_interference_early_stopping_llm_harmful_outputs_via_streaming_c.md)**

:   提出 Streaming Content Monitor (SCM)——首个原生支持部分检测的流式有害内容监控器，通过 FineHarm 数据集（29K 样本含 token 级标注）和层次一致性感知学习，平均仅需看到 18% 的 response tokens 即可达到 0.95+ macro F1，实现对 LLM 有害输出的实时早停。

**[G-Dpo Scalable Preference Optimization For Protein Language Models](llm_alignment/g-dpo_scalable_preference_optimization_for_protein_language_models.md)**

:   针对蛋白质语言模型（PLM）应用 DPO 时偏好对数量随样本数二次增长导致训练不可扩展的问题，提出 g-DPO 框架：(1) 通过序列空间 union mask 聚类剪枝冗余偏好对，保留局部邻域中信息量更大的比较；(2) 利用共享 union mask 的分组似然摊销，一次前向传播同时计算组内所有序列的 log-likelihood。在三个蛋白质工程任务上，g-DPO 保持与标准 DPO 统计上不可区分的 in silico 和 in vitro 性能，同时实现 1.7-5.4× 的训练加速。

**[Gasp Efficient Black-Box Generation Of Adversarial Suffixes For Jailbreaking Llm](llm_alignment/gasp_efficient_black-box_generation_of_adversarial_suffixes_for_jailbreaking_llm.md)**

:   提出GASP框架，通过训练专用的SuffixLLM生成可读的对抗后缀，利用潜在贝叶斯优化（LBO）在连续嵌入空间中高效搜索并用ORPO迭代微调生成器，在完全黑盒设置下实现高攻击成功率且生成的后缀保持人类可读性。

**[Generalizing While Preserving Monotonicity In Comparison-Based Preference Learni](llm_alignment/generalizing_while_preserving_monotonicity_in_comparison-based_preference_learni.md)**

:   提出 **Linear GBT with Diffusion Prior**，一类在保证**单调性**（偏好比较后被偏好方的分数不会反常下降）的同时能**泛化到未比较数据**的偏好学习模型，首次正面回答了"泛化与单调性能否兼得"的核心问题。

**[Greedy Sampling Is Provably Efficient For Rlhf](llm_alignment/greedy_sampling_is_provably_efficient_for_rlhf.md)**

:   证明了在KL正则化的RLHF设置下，直接使用经验估计的贪心采样（无需构建乐观/悲观估计）就能在在线和离线两种设置中实现$O(\log T)$遗憾界和$O(\varepsilon^{-1})$样本复杂度，这是首次在一般偏好模型下达到这些阶数。

**[Gvpo Group Variance Policy Optimization For Large Language Model Post-Training](llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)**

:   通过将 KL 约束奖励最大化的解析解融入梯度权重（零和权重消除配分函数），设计了比 GRPO 更稳定的 LLM 后训练方法 GVPO，在 AIME 上达到 20.72%（GRPO 14.79%），并证明具有唯一全局最优解。

**[Human-Assisted Robotic Policy Refinement Via Action Preference Optimization](llm_alignment/human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)**

:   提出 Action Preference Optimization (APO)，通过人机协作框架收集交互轨迹，利用基于前景理论的二元期望信号和自适应重加权方法对 VLA 模型进行偏好对齐优化，使其能从失败中学习并持续迭代改进。

**[Improving Data Efficiency For Llm Reinforcement Fine-Tuning Through Difficulty-T](llm_alignment/improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)**

:   提出两种互补技术提升 LLM 强化微调（GRPO）的数据效率：(1) DOTS——基于注意力机制预测自适应难度，优先选择中等难度问题以最大化梯度信号；(2) Rollout Replay——复用近期 rollout 降低每步计算开销。两者结合在 6 个模型-数据集组合上平均减少 40.7% 训练时间。

**[Inference-Time Alignment In Continuous Space](llm_alignment/inference-time_alignment_in_continuous_space.md)**

:   提出 Simple Energy Adaptation (SEA)，将推理时对齐从"离散空间搜索"范式转变为"连续空间优化"范式，通过在连续 logit 空间上进行基于梯度的 Langevin 采样来逼近 RLHF 最优策略，在 AdvBench 上相对最优基线提升 77.51%，在 MATH 上提升 16.36%。

**[Jailbreak-Zero A Path To Pareto Optimal Red Teaming For Large Language Models](llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)**

:   提出基于策略（而非示例）的 LLM 红队评估框架和 Jailbreak-Zero 方法，通过简单的大规模并行采样策略（无需人工越狱策略），在 HarmBench 上对 GPT-4o 和 Claude 3.5 分别达到 99.5% 和 96.0% 的攻击成功率，同时通过微调实现覆盖率、多样性和保真度三个目标的 Pareto 最优。

**[Kl Penalty Control Via Perturbation For Direct Preference Optimization](llm_alignment/kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)**

:   提出 ε-DPO，通过观察训练时扰动 β 后 logit 作为偏好模型的单调性，实现实例级自适应 KL 惩罚控制，无需额外计算开销即可显著超越 DPO 及大多数直接对齐算法，在 AlpacaEval 2 上达到 46.4% LC win rate（DPO 仅 40.3%）。

**[Laser Learning To Adaptively Select Reward Models With Multi-Armed Bandits](llm_alignment/laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)**

:   将多个奖励模型（RM）的选择建模为上下文多臂老虎机（LinUCB）问题，在迭代 LLM 训练中自适应地为每个 batch 选择最合适的 RM，在推理、指令跟随和长上下文任务上以 2-3 倍效率优势全面超越 RM 集成和单 RM 基线。

**[Limited Preference Data Learning Better Reward Model With Latent Space Synthesis](llm_alignment/limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)**

:   提出 LENS 框架，通过在 LLM 嵌入的潜在空间中利用 VAE 合成偏好数据对，绕过昂贵的文本生成过程，以极低计算成本（模型缩小 16000 倍、生成速度提升 18 倍）显著提升 reward model 性能。

**[Llm Safety Alignment Is Divergence Estimation In Disguise](llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)**

:   建立统一理论框架证明 RLHF/DPO/KTO/BCO 等对齐方法本质上是在估计安全分布 $\mathcal{D}^+$ 与不安全分布 $\mathcal{D}^-$ 之间的散度，由此解释了对齐后隐空间分离现象，并提出基于 KL 散度的 KLDO 对齐方法，在 5 个模型上实现最佳鲁棒性。

**[Longvpo From Anchored Cues To Selfreasoning For Longform Vid](llm_alignment/longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)**

:   LongVPO提出两阶段DPO框架，Stage 1通过锚定短片段构造伪长视频偏好数据并引入anchor-only参考模型近似解决上下文长度不匹配问题，Stage 2通过递归字幕生成和多片段推理任务在真实长视频上自训练，仅用16K合成样本即超越大规模监督训练的长视频模型。

**[Mechanism Design For Llm Fine-Tuning With Multiple Reward Models](llm_alignment/mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)**

:   将多方偏好聚合的 RLHF 微调建模为机制设计问题，证明了在社会福利最大化训练规则下各方有动机虚报偏好，并通过扩展 VCG 支付机制实现了占优策略激励相容（DSIC），确保各方如实报告偏好。

**[Metadefense Defending Finetuning-Based Jailbreak Attack Before And During Genera](llm_alignment/metadefense_defending_finetuning-based_jailbreak_attack_before_and_during_genera.md)**

:   提出 MetaDefense，一个两阶段（生成前+生成中）防御框架，通过训练 LLM 自身预测查询和部分响应的有害性来防御基于微调的越狱攻击，无需额外分类器，实现 2× 内存效率。

**[Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preferen](llm_alignment/mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)**

:   提出 SymMPO（对称多模态偏好优化），通过对比图像的对称配对偏好学习和偏好边际一致性正则化，解决了现有视觉增强型 DPO 方法中目标函数不严格和间接偏好监督两大局限，在五个幻觉评测基准上取得了一致的性能提升。

**[Multi-Environment Pomdps Discrete Model Uncertainty Under Partial Observability](llm_alignment/multi-environment_pomdps_discrete_model_uncertainty_under_partial_observability.md)**

:   系统研究了多环境 POMDP（ME-POMDP）——一类共享状态/动作/观测空间但转移、观测和奖励函数可任意不同的 POMDP 集合，目标是找到在最坏情况环境下最大化奖励的鲁棒策略。通过引入对抗信念 POMDP（AB-POMDP）统一建模，并证明其与单侧部分可观测随机博弈（POSG）的等价关系，提出精确（值迭代 + LP）和近似（AB-HSVI）算法。

**[On Extending Direct Preference Optimization To Accommodate Ties](llm_alignment/on_extending_direct_preference_optimization_to_accommodate_ties.md)**

:   将 DPO 中的 Bradley-Terry 偏好模型替换为 Rao-Kupper 和 Davidson 扩展，使偏好优化能够显式建模"平局"数据，避免丢弃模糊偏好对，在翻译和数学推理上获得更好的正则化和性能。

**[Orpo-Distill Mixed-Policy Preference Optimization For Cross-Architecture Llm Dis](llm_alignment/orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)**

:   提出 ORPO-Distill，将跨架构 LLM 知识蒸馏重新定义为偏好优化问题：使用教师模型生成正样本推理链、学生模型生成负样本推理链，通过 ORPO 对比损失训练，并引入混合策略（mixed-policy）更新学生负样本，在 5 个 QA 基准上一致超越黑盒 KD 基线。

**[Polyjuice Makes It Real Black-Box Universal Red Teaming For Synthetic Image Dete](llm_alignment/polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)**

:   提出 PolyJuice，首个面向合成图像检测器（SID）的黑盒、图像无关的红队方法，通过在 T2I 模型潜空间中发现并利用"真实感方向"，以通用方式引导生成图像欺骗检测器，成功率高达 84%。

**[Position The Complexity Of Perfect Ai Alignment -- Formalizing The Rlhf Trilemma](llm_alignment/position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)**

:   本文将 RLHF 中反复出现的安全-公平-效率冲突形式化为「对齐三难困境」：证明了没有任何 RLHF 系统能同时满足 $\varepsilon$-代表性（忠实反映多元价值）、多项式可处理性（计算可行）和 $\delta$-鲁棒性（抵御对抗攻击），从而为当前 RLHF 系统中偏好坍缩、谄媚等病理现象提供了统一的复杂度理论解释。

**[Preference Optimization By Estimating The Ratio Of The Data Distribution](llm_alignment/preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)**

:   将 DPO 重新解释为似然比估计（ratio matching）问题，基于 Bregman 散度框架提出 BPO（Bregman Preference Optimization），包含 DPO 为特例的广义损失函数族，并设计了 SBA（Scaled Basu's Power Divergence）实例，在 Llama-3-8B 上实现 55.9% AlpacaEval2 length-controlled win rate 的 SOTA。

**[Provably Efficient Online Rlhf With One-Pass Reward Modeling](llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)**

:   提出一种基于 online mirror descent（OMD）的 one-pass reward modeling 方法，消除了 online RLHF 中需要存储历史数据并重新从头优化的计算瓶颈，实现每次迭代 $\mathcal{O}(1)$ 的时间和存储复杂度，同时在统计效率上也优于 MLE 方法。

**[Reinforcement Learning Finetunes Small Subnetworks In Large Language Models](llm_alignment/reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)**

:   RL 微调 LLM 时实际上只更新了 5%-30% 的参数（稀疏子网络），且该子网络在不同种子、数据和算法间具有高度一致性，仅微调子网络即可复现完整微调的模型性能甚至参数值。

**[Responserank Data-Efficient Reward Modeling Through Preference Strength Learning](llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)**

:   提出 ResponseRank 方法,通过利用偏好强度的代理信号（如响应时间和标注者一致性）的局部相对差异来鲁棒地学习效用差值,显著提升奖励模型的样本效率。

**[Rethinking Direct Preference Optimization In Diffusion Models](llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)**

:   针对扩散模型中 DPO 的两个核心问题——有限探索和奖励尺度不平衡,提出稳定参考模型更新策略和时间步感知训练策略,可集成到各种偏好优化算法中。

**[Robust Llm Alignment Via Distributionally Robust Direct Preference Optimization](llm_alignment/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)**

:   通过分布鲁棒优化（DRO）框架提出 WDPO（Wasserstein）和 KLDPO（KL散度）两种鲁棒 DPO 变体，解决用户偏好分布转移导致的对齐失败问题，提供 $O(n^{-1/4})$ 收敛保证，在多维对齐任务和 OpenLLM 榜单上显著优于标准 DPO。

**[Safeptr Token-Level Jailbreak Defense In Multimodal Llms Via Prune-Then-Restore ](llm_alignment/safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)**

:   通过分析多模态 LLM 中有害 token 的传播机制，发现不到 1% 的 token 在早期-中间层引发越狱行为，由此提出无需训练的 SafePTR 框架，在脆弱层剪枝有害 token 并在后续层恢复良性特征，显著提升安全性而不牺牲任务性能。

**[Safevla Towards Safety Alignment Of Vision-Language-Action Model Via Constrained](llm_alignment/safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)**

:   首次系统性地将安全强化学习（SafeRL）的 CMDP 框架应用于视觉-语言-动作模型（VLA）的安全对齐，通过建模-激发-约束-保证四阶段集成安全方法（ISA），在移动操作任务上实现 83.58% 的安全违规成本下降同时保持任务性能（+3.85%）。

**[Self-Alignment Of Large Video Language Models With Refined Regularized Preferenc](llm_alignment/self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)**

:   提出 RRPO（Refined Regularized Preference Optimization），通过子序列级细粒度奖励和 token 级 KL 正则化替代 DPO 的响应级奖励，结合自对齐数据生成框架，在视频理解任务上减少幻觉、提升时间推理能力。

**[Short-Length Adversarial Training Helps Llms Defend Long-Length Jailbreak Attack](llm_alignment/short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)**

:   理论证明并实验验证：防御长度 $\Theta(M)$ 的后缀越狱攻击，只需要在长度 $\Theta(\sqrt{M})$ 的对抗后缀上做对抗训练即可，即"短对抗训练防长越狱"——在5个主流LLM上，20 token 对抗训练可将 120 token 越狱成功率降低至少 30%。

**[Simplicity Prevails Rethinking Negative Preference Optimization For Llm Unlearni](llm_alignment/simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)**

:   发现 NPO（负偏好优化）中的参考模型偏差导致遗忘数据的优化功率分配不均和早期梯度权重平滑失效，提出 SimNPO 通过去除参考模型依赖并采用长度归一化奖励，在 TOFU 上将 FQ 从 0.79 提升至 0.99，在所有基准上一致优于 NPO。

**[Strategyproof Reinforcement Learning From Human Feedback](llm_alignment/strategyproof_reinforcement_learning_from_human_feedback.md)**

:   首次从机制设计角度研究 RLHF 中多标注者策略性操纵问题，证明了策略防操纵（strategyproofness）与政策对齐之间存在根本性权衡，并提出 Pessimistic Median of MLEs 算法实现近似策略防操纵。

**[T-Shirt Token-Selective Hierarchical Data Selection For Instruction Tuning](llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)**

:   提出 T-SHIRT 数据选择框架，通过 Selective IFD（仅考虑有信息量的 token）和分层选择策略（偏好邻域一致性高的样本），用 5% 数据微调即可超越全量数据训练，同时仅需 GPT-2 和单 GPU 40 分钟完成选择。

**[Towards Understanding Safety Alignment A Mechanistic Perspective From Safety Neu](llm_alignment/towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)**

:   通过机制可解释性视角发现 LLM 中约 5% 的稀疏"安全神经元"，仅修补（patching）这些神经元的激活即可恢复 90% 以上的安全性能，并从神经元重叠角度解释了 alignment tax 现象。

**[Trajectory Bellman Residual Minimization A Simple Value-Based Method For Llm Rea](llm_alignment/trajectory_bellman_residual_minimization_a_simple_value-based_method_for_llm_rea.md)**

:   TBRM 通过最小化轨迹级贝尔曼残差，将 LLM 输出 logits 视为隐式 Q 值，仅需每个 prompt 一次前向采样即可训练，复杂度远低于 PPO/GRPO 但数学推理性能相当或更优。

**[What Makes A Reward Model A Good Teacher An Optimization Perspective](llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)**

:   从优化理论角度证明：奖励模型的准确率（accuracy）不足以衡量其作为 RLHF "教师"的质量——即使完美准确的奖励模型，如果诱导的奖励方差（reward variance）过低，也会导致 RLHF 目标函数景观平坦，使 policy gradient 优化极慢；不同的语言模型需要不同的奖励模型。

---

## 💬 LLM/NLP { #llm_nlp }

**[Acesearcher Bootstrapping Reasoning And Search For Llms Via Reinforced Self-Play](llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)**

:   提出 AceSearcher——一种协作式自我博弈框架，让单个 LLM 同时扮演**问题分解者**（将复杂查询拆解为子问题引导检索）和**求解者**（整合检索上下文生成答案），通过 SFT + 迭代 DPO 两阶段训练，仅用最终答案作为奖励信号，在 10 个数据集上平均 EM 提升 7.6%，32B 模型匹配 DeepSeek-V3（<5% 参数）。

**[Are Language Models Efficient Reasoners A Perspective From Logic Programming](llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)**

:   从逻辑编程角度提出评估 LLM 推理效率（而非仅正确性）的框架——通过 verbalized logic program 将自然语言证明映射到逻辑程序证明，发现当前 LLM 在含无关公理的数学题中不仅准确率下降，且推理过程严重低效（超过一半的推理步骤是不必要的）。

**[C2Prompt Class-Aware Client Knowledge Interaction For Federated Continual Learni](llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)**

:   针对联邦持续学习中prompt通信时的类级知识不一致问题，提出C²Prompt方法，通过局部类分布补偿（LCDC）和类感知prompt聚合（CPA）两个机制显式增强跨客户端的类级知识一致性，在ImageNet-R上Avg准确率达87.20%，超出SOTA Powder 2.51%。

**[Cat Circular-Convolutional Attention For Sub-Quadratic Transformers](llm_nlp/cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)**

:   CAT 将标准自注意力中的 $N \times N$ 注意力矩阵替换为一个由 $N$ 维向量生成的循环矩阵（circulant matrix），利用 FFT 实现 $O(N \log N)$ 复杂度的注意力计算，在严格保持 softmax 行归一化结构的前提下，在 ImageNet-1k（avg pool 下 CLIP-L 准确率 0.694 vs 标准注意力 0.646）和 WikiText-103 masked LM（PPL 8.32 vs 9.82）上匹配或超越标准注意力。

**[Characterizing The Expressivity Of Fixed-Precision Transformer Language Models](llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)**

:   精确刻画了固定精度、严格未来掩码、软注意力、无位置编码的 Transformer 的表达能力——恰好等价于仅含过去算子的线性时态逻辑 LTL[P]，并将其与偏序确定有限自动机 (PODFA)、$\mathcal{R}$-trivial 幺半群统一起来。

**[Composing Linear Layers From Irreducibles](llm_nlp/composing_linear_layers_from_irreducibles.md)**

:   利用Clifford代数，将线性层表示为二向量（bivector）的组合——即旋量（rotor）的三明治乘积——仅需 $O(\log^2 d)$ 参数即可替代 $d \times d$ 密集矩阵，应用于LLM注意力层的Q/K/V投影时性能接近原始模型和强基线。

**[Cultural Alien Sampler Open-Ended Art Generation Balancing Originality And Coher](llm_nlp/cultural_alien_sampler_open-ended_art_generation_balancing_originality_and_coher.md)**

:   提出Cultural Alien Sampler (CAS)——用两个GPT-2模型分别建模"概念一致性"和"文化典型性"，通过选择高一致性但低文化典型性的概念组合来生成原创且和谐的艺术创意，在人类评估中接近艺术专业学生水平并远超GPT-4o。

**[Deep Learning For Continuous-Time Stochastic Control With Jumps](llm_nlp/deep_learning_for_continuous-time_stochastic_control_with_jumps.md)**

:   提出两种基于模型的深度学习算法（GPI-PINN 和 GPI-CBU）来求解含跳跃的有限时域连续时间随机控制问题，通过迭代训练策略网络和价值网络，避免了状态动力学的离散化和模拟，在高维场景中表现出色。

**[Detecting High-Stakes Interactions With Activation Probes](llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)**

:   用线性激活探针（在 LLM 内部表示上训练的轻量分类器）检测用户的"高风险交互"，在合成数据上训练后跨 6 个真实数据集 AUROC 达 0.88-0.92，匹敌 8-12B 微调 LLM但计算成本低 6 个数量级，级联架构（探针初筛+LLM 精判）进一步超越单独使用任一方法。

**[Do Language Models Use Their Depth Efficiently](llm_nlp/do_language_models_use_their_depth_efficiently.md)**

:   通过因果干预、残差流分析和跨模型线性映射，证明当前 LLM 后半部分层不参与组合式计算，仅迭代细化输出概率分布，深层模型只是把浅层模型的计算"展延"到更多层。

**[Dont Be Lazy Completep Enables Compute-Efficient Deep Transformers](llm_nlp/dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)**

:   CompleteP 参数化（α=1）是唯一同时实现深度方向超参转移和完全特征学习的方案，在深模型上相比 μP 节省 12-34% FLOPs。

**[Encompass Enhancing Agent Programming With Search Over Program Execution Paths](llm_nlp/encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)**

:   提出 Probabilistic Angelic Nondeterminism (PAN) 编程模型及 EnCompass Python 框架，将 agent 的核心工作流逻辑与推理时搜索策略解耦，程序员只需在 LLM 调用处加 `branchpoint()` 标记，即可用几行参数切换 best-of-N、beam search、tree search 等策略，代码修改量减少 3-6x。

**[Geocad Local Geometry-Controllable Cad Generation With Large Language Models](llm_nlp/geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)**

:   提出 GeoCAD，首个实现局部几何可控 CAD 生成的方法，通过互补标注策略为局部零件生成几何指令，并微调 LLM 实现根据用户文本指令精确修改 CAD 模型的局部部分。

**[Hyperparameter Transfer Enables Consistent Gains Of Matrix-Preconditioned Optimi](llm_nlp/hyperparameter_transfer_enables_consistent_gains_of_matrix-preconditioned_optimi.md)**

:   研究矩阵预条件优化器（Shampoo/SOAP/Muon）的超参数随模型宽度和深度的缩放规则（基于 μP），发现正确的超参缩放是实现一致加速的关键：使用 μP + 1/width weight decay，三者在 190M 到 1.4B 参数的 Llama 模型上一致实现约 1.4× 加速。

**[In-Context Learning Of Linear Dynamical Systems With Transformers Approximation ](llm_nlp/in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)**

:   分析了线性 Transformer 在噪声线性动力系统上的 ICL 近似能力：$O(\log T)$ 深度可达到 $O(\log T / T)$ 测试误差（接近最小二乘估计器），而单层线性 Transformer 存在不可消除的下界——揭示了非 IID 数据下的深度分离现象。

**[Large Language Models Miss The Multi-Agent Mark](llm_nlp/large_language_models_miss_the_multi-agent_mark.md)**

:   Position paper 通过调研 1400+ 篇论文，系统论证当前 MAS LLMs 在四个维度偏离传统 MAS 基础理论——LLM 缺乏原生社会行为、环境设计以 LLM 为中心、缺少异步协调和标准通信协议、涌现行为缺乏量化，指出该领域有忽视 40 年 MAS 成果而重新发明轮子的风险。

**[Linear Transformers Implicitly Discover Unified Numerical Algorithms](llm_nlp/linear_transformers_implicitly_discover_unified_numerical_algorithms.md)**

:   训练线性 Transformer 执行矩阵块补全任务后，通过权重代数分析发现模型在三种完全不同的计算约束（集中式、分布式、计算受限）下隐式收敛到同一个双行迭代更新规则 EAGLE，该规则具有二阶收敛性且依赖条件数仅为对数级别。

**[Monarchattention Zero-Shot Conversion To Fast Hardware-Aware Structured Attentio](llm_nlp/monarchattention_zero-shot_conversion_to_fast_hardware-aware_structured_attentio.md)**

:   提出 MonarchAttention，利用 Monarch 矩阵的结构化特性，通过 softmax 变分形式的交替优化，实现 $\Theta(N\sqrt{N}d)$ 复杂度的注意力近似，无需额外训练即可零样本替换预训练 Transformer 的注意力层，同时在 GPU 上相比 FlashAttention-2 实现 1.4×–8.2× 的加速。

**[Moose-Chem2 Exploring Llm Limits In Fine-Grained Scientific Hypothesis Discovery](llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)**

:   将细粒度科学假设生成形式化为组合优化问题，提出层次启发式搜索（HHS）——利用 LLM 的成对比较作为梯度信号在假设空间中导航，层次化抽象平滑奖励景观减少局部最优陷阱，在 2024 年后化学论文 51 篇的专家标注 benchmark 上 Soft Recall 从 19.99% 提升到 40.35%。

**[Msf-Cnn Patch-Based Multi-Stage Fusion With Convolutional Neural Networks For Ti](llm_nlp/msf-cnn_patch-based_multi-stage_fusion_with_convolutional_neural_networks_for_ti.md)**

:   提出 msf-CNN，一种基于有向无环图（DAG）最短路径算法的多阶段 patch-based 融合优化技术，通过高效搜索 CNN 的最优融合配置，在各种微控制器（ARM Cortex-M、RISC-V、ESP32）上实现比现有方法（MCUNetV2、StreamNet）减少 50%–87% 的峰值 RAM 使用，同时保持可控的计算开销。

**[Nemotron-Flash Towards Latency-Optimal Hybrid Small Language Models](llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)**

:   Nemotron-Flash 通过系统优化深宽比、进化搜索混合算子组合（DeltaNet+Mamba2+Attention）以及权重归一化训练，构建延迟最优的小语言模型家族，相比 Qwen3-1.7B/0.6B 分别实现 1.3×/1.9× 延迟下降与 +5.5% 平均准确率提升。

**[On The Role Of Hidden States Of Modern Hopfield Network In Transformer](llm_nlp/on_the_role_of_hidden_states_of_modern_hopfield_network_in_transformer.md)**

:   本文突破现代 Hopfield 网络（MHN）与 Transformer 对应关系的绝热近似限制，发现保留 MHN 的隐状态动力学会在自注意力层中引入跨层注意力分数传播机制（Modern Hopfield Attention, MHA），不增加训练参数即可系统性改善 ViT 和 GPT-2 的性能，并从理论和实验上证明 MHA 有效缓解了深层 Transformer 的 rank collapse 问题。

**[Opinion Maximization In Social Networks By Modifying Internal Opinions](llm_nlp/opinion_maximization_in_social_networks_by_modifying_internal_opinions.md)**

:   本文研究社交网络中通过修改 k 个关键节点的内部意见来最大化整体意见的优化问题，提出了两种基于采样的近似算法（随机游走和森林采样）以及一种基于异步更新的精确算法 MIS，后者在理论上保证收敛到最优解，并在数千万节点的真实网络上展示了卓越的效率与精度。

**[Out-Of-Distribution Generalisation Is Hard Evidence From Arc-Like Tasks](llm_nlp/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)**

:   通过构建具有明确OOD度量的ARC类任务，证明标准神经网络(MLP/CNN/Transformer)无法实现组合OOD泛化，即使设计具有正确归纳偏置的架构达到近乎完美的OOD性能，也可能学到错误的组合特征。

**[Pluralistic Behavior Suite Stress-Testing Multi-Turn Adherence To Custom Behavio](llm_nlp/pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)**

:   提出 PBSuite，一个包含 300 个行业定制行为策略和动态多轮对抗评估框架的评测套件，揭示了主流 LLM 在单轮设置下合规率高（违规 <4%），但在多轮对抗交互中合规性急剧下降（违规高达 84%）。

**[Polar Sparsity High Throughput Batched Llm Inferencing With Scalable Contextual ](llm_nlp/polar_sparsity_high_throughput_batched_llm_inferencing_with_scalable_contextual_.md)**

:   揭示了 LLM 推理中稀疏性的"极性转移"现象——MLP 层稀疏性随 batch 增大而消失，而 attention head 稀疏性保持稳定且与 batch 无关，据此设计了 Selective Head Attention 及对应 GPU kernel，在大 batch 推理中实现高达 2.2x 的端到端加速。

**[Post Hoc Regression Refinement Via Pairwise Rankings](llm_nlp/post_hoc_regression_refinement_via_pairwise_rankings.md)**

:   提出 RankRefine，一种模型无关的后处理回归改进方法，通过将基础回归器的预测与基于成对排序的估计进行逆方差加权融合，在无需重训练的情况下显著降低预测误差，仅需 20 次成对比较和通用 LLM 即可实现分子性质预测中高达 10% 的 MAE 相对减少。

**[Presto Preimage-Informed Instruction Optimization For Prompting Black-Box Llms](llm_nlp/presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)**

:   提出 PRESTO 框架，利用白盒 LLM 中 soft prompt 到 instruction 的 many-to-one 映射关系（preimage 结构），通过 score sharing、preimage-based initialization 和 score consistency regularization 三大组件，在相同查询预算下等效获得 14 倍的标注数据量，显著提升黑盒 LLM 的指令优化效率。

**[Qsharp Provably Optimal Distributional Rl For Llm Post-Training](llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)**

:   提出 Q♯，一种基于分布式 RL 的值函数方法用于 KL 正则化 LLM 后训练，通过学习参考策略下的累积奖励分布来计算最优软 Q 函数引导生成，在数学推理任务上实现更高准确率和更低 KL 散度，并证明了方差相关的 PAC 收敛界。

**[Reparameterized Llm Training Via Orthogonal Equivalence Transformation](llm_nlp/reparameterized_llm_training_via_orthogonal_equivalence_transformation.md)**

:   提出 POET 训练框架，通过将权重矩阵重参数化为"两个可学习正交矩阵 × 固定随机权重"的形式来保持谱性质不变，实现更稳定的训练和更好的泛化，且比 AdamW 更节省参数。

**[Scaling Up Active Testing To Large Language Models](llm_nlp/scaling_up_active_testing_to_large_language_models.md)**

:   通过三项关键简化——用 in-context learning 构建固定代理模型、使用小代理模型评估大目标模型、无需目标模型预测进行数据采集——将 active testing 扩展到 LLM，风险估计误差比随机采样降低 25%-80%。

**[Solverllm Leveraging Test-Time Scaling For Optimization Problem Via Llm-Guided S](llm_nlp/solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)**

:   提出SolverLLM，一个无需训练的框架，将优化问题的数学建模视为搜索问题，通过改进的MCTS在六元素表述空间中探索最优formulation，引入动态扩展、提示反向传播和不确定性反向传播，在6个基准上以无训练方式超越prompt方法和微调方法。

**[Solving Inequality Proofs With Large Language Models](llm_nlp/solving_inequality_proofs_with_large_language_models.md)**

:   提出 IneqMath（首个大规模奥林匹克级不等式 benchmark），将不等式证明定义为两个可自动验证的子任务（界估计与关系预测），并开发五模块 LLM-as-Judge 框架，发现即便 o1 在逐步推理审查下整体准确率也不到 10%。

**[Space Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning For Large La](llm_nlp/space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)**

:   提出 Space（Self-PlAy via Noise Contrastive Estimation），将噪声对比估计引入自对弈微调，通过独立优化真实和合成样本的绝对奖励值（而非相对差距），从根本上解决了 SPIN 等方法的不稳定收敛问题，并提供可证明的稳定收敛保证。

**[Sparse Mezo Less Parameters For Better Performance In Zeroth-Order Llm Fine-Tuni](llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)**

:   提出 Sparse MeZO（S-MeZO），通过观察到零阶梯度噪声对大权重影响更严重，选择性地仅对小权重进行零阶优化扰动和更新，在不增加内存开销的前提下实现了显著的性能提升（RTE 上 +9%）和收敛加速（3.5x）。

**[Spectral Conditioning Of Attention Improves Transformer Performance](llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)**

:   理论分析了 Transformer 注意力层 Jacobian 的条件数受 Query/Key/Value 矩阵条件数控制，提出谱调节注意力（Spectral Conditioned Attention），通过向 Q/K/V 矩阵添加固定校正项降低条件数，作为即插即用模块在图像分类、目标检测、NLP 等多任务上一致提升性能。

**[Speculate Deep And Accurate Lossless And Training-Free Acceleration For Offloade](llm_nlp/speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)**

:   提出 SubSpec，一种即插即用的无损、无训练参数卸载 LLM 加速方法，核心思想是从卸载的目标模型本身构建高对齐度的量化替代草稿模型，并通过共享 GPU 驻留层和 KV-Cache 最大化对齐度，在 8GB 显存限制下实现 Qwen2.5 7B 的 9.1 倍加速、24GB 显存下 Qwen2.5 32B 的 12.5 倍加速。

**[Strassen Attention Split Vc Dimension And Compositionality In Transformers](llm_nlp/strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)**

:   提出 Splitting VC 维度理论工具证明了单层 softmax Transformer（即使无限精度）在组合推理任务上的根本限制，并设计了具有亚立方时间复杂度的 Strassen 注意力机制来突破这些限制。

**[Streambridge Turning Your Offline Video Large Language Model Into A Proactive St](llm_nlp/streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)**

:   StreamBridge提出一个简单通用的框架，通过记忆缓冲区+轮次衰减压缩策略实现多轮流式交互，通过解耦的轻量激活模型实现主动响应，配合专门构建的Stream-IT数据集，成功将离线Video-LLM（如Qwen2-VL、LLaVA-OV）转化为流式助手，在OVO-Bench和Streaming-Bench上超越GPT-4o和Gemini 1.5 Pro。

**[Symphony Synergistic Multi-Agent Planning With Heterogeneous Language Model Asse](llm_nlp/symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)**

:   提出 SYMPHONY，一个基于 MCTS 的多智能体规划框架，通过异构 LLM 池的多样性驱动搜索、UCB 自适应调度、熵调制置信度评估和池级记忆共享，显著提升了 LLM 规划的多样性和效率。

**[Synergy Over Discrepancy A Partition-Based Approach To Multi-Domain Llm Fine-Tun](llm_nlp/synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)**

:   提出基于分区的多阶段微调框架，通过策略性地将多个域划分为子集（阶段），在最大化域间协同的同时最小化负迁移，并推导了新的泛化界来理论支撑该分区策略。

**[System Prompt Optimization With Meta-Learning](llm_nlp/system_prompt_optimization_with_meta-learning.md)**

:   提出双层系统提示优化问题并设计 MetaSPO 元学习框架，通过外循环优化跨任务泛化的系统提示、内循环优化任务特定的用户提示，使优化后的系统提示在 14 个未见任务上显著超越基线。

**[The Rise Of Parameter Specialization For Knowledge Storage In Large Language Mod](llm_nlp/the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)**

:   系统分析 20 个开源 LLM，发现更强的模型在 MLP 参数向量中展现出更高的知识特化程度（Parameter Specialization），即相似知识倾向于集中编码到少数参数向量中，并通过因果实验验证该特化程度与模型知识任务性能之间存在因果关系。

**[Triplets Better Than Pairs Towards Stable And Effective Self-Play Fine-Tuning Fo](llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)**

:   提出 T-SPIN（三元组自博弈微调），在 SPIN 基础上引入"历史优势"（proto-synthetic 响应作为锚点）和熵约束实现无参考策略训练，解决了 SPIN 迭代中的优化不稳定和训练-生成不对齐两大问题，仅用 25% 标注数据即可媲美全量 SFT。

**[Unifying Attention Heads And Task Vectors Via Hidden State Geometry In In-Contex](llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)**

:   本文提出基于隐状态几何（可分离性+对齐性）的统一框架，将ICL的两大解释路线——注意力头（PTH/IH）和任务向量——联系起来，揭示ICL在分类任务中的两阶段机制：早期层通过PTH建立可分离性，后期层通过IH改善与标签unembedding方向的对齐性。

**[Valid Inference With Imperfect Synthetic Data](llm_nlp/valid_inference_with_imperfect_synthetic_data.md)**

:   提出基于广义矩估计（GMM）的无超参数框架，将 LLM 生成的不完美合成数据与真实数据结合进行统计有效推断，当合成数据残差与真实数据残差相关时可显著降低估计方差，且在最坏情况下（合成数据完全无信息）也不会损害估计质量。

**[What One Cannot Two Can Two-Layer Transformers Provably Represent Induction Head](llm_nlp/what_one_cannot_two_can_two-layer_transformers_provably_represent_induction_head.md)**

:   理论证明两层单头 Transformer 足以表示任意 $k$ 阶马尔可夫过程的条件 $k$-gram 模型（即 $k$ 阶 induction head），给出了 Transformer 深度与马尔可夫阶数关系的最紧已知刻画，关键在于利用 MLP 中的 ReLU 和 LayerNorm 非线性来补偿减少的层数。

**[Wider Or Deeper Scaling Llm Inference-Time Compute With Adaptive Branching Tree ](llm_nlp/wider_or_deeper_scaling_llm_inference-time_compute_with_adaptive_branching_tree_.md)**

:   AB-MCTS 提出了一种自适应分支的蒙特卡洛树搜索框架，在搜索树的每个节点上动态决定是"变宽"（生成新候选答案）还是"变深"（利用反馈优化现有答案），通过贝叶斯后验更新平衡探索与利用，在编程和工程任务上超越了重复采样和标准 MCTS。

**[Writing In Symbiosis Mapping Human Creative Agency In The Ai Era](llm_nlp/writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)**

:   通过对 5 万+文档的纵向语料分析，提出"双轨演化"假说——LLM 时代人类写作在主题上趋同、风格上结构性分化，并发现三种作者适应策略原型（Adopters/Resistors/Pragmatists）。

---

## 📈 时间序列 { #time_series }

**[A Graph Neural Network Approach For Localized And High-Resolution Temperature Fo](time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core Time Series Prediction By Adaptive](time_series/abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[Aero A Redirection-Based Optimization Framework Inspired By Judo For Robust Prob](time_series/aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)**

:   AERO提出受柔道"借力重定向"启发的优化范式，尝试将对抗扰动重定向为有利的优化信号而非直接抵抗，理论上通过15条公理和4个定理构建了基于能量守恒的梯度重定向系统，但实际实现大幅简化为带高斯噪声注入的动量SGD，仅在一个私有太阳能价格预测数据集上进行了无基线对比的验证。

**[Attentionpredictor Temporal Patterns Matter For Kv Cache Com](time_series/attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   AttentionPredictor是首个学习型方法直接预测注意力模式以实现KV缓存压缩和关键token识别，通过轻量CNN捕捉注意力分数的时空模式，实现13倍KV缓存压缩和5.6倍推理加速，统一预测模型仅21KB可跨所有Transformer层共享。

**[Benchmarking Probabilistic Time Series Forecasting Models On Neural Activity](time_series/benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)**

:   首次系统评测 12 个概率时间序列预测模型在小鼠皮层钙成像数据上的表现，发现 PatchTST 一致最优（信息性预测窗口达 1.5 秒），零样本基础模型（Chronos）完全失败但微调后竞争力强，揭示神经活动的内在可预测性上限约 1.5 秒。

**[Causal Masking On Spatial Data An Information-Theoretic Case For Learning Spatia](time_series/causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)**

:   证明在空间数据（国际象棋棋盘FEN状态）上直接应用因果掩蔽训练单模态LLM，其表现优于先将数据线性化为序列（PGN棋步）后再应用因果掩蔽——FEN+因果掩蔽的Llama 1.3B达到~2630 Elo，而PGN+因果仅~2130 Elo。

**[Causaldynamics A Large-Scale Benchmark For Structural Discovery Of Dynamical Cau](time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)**

:   提出 CausalDynamics——迄今最大规模的动力系统因果发现 benchmark（14000+ 图、5000 万+ 样本），涵盖从 3 维混沌 ODE/SDE 到层级耦合系统再到拟真气候模型的三层渐进复杂度体系，并全面评估了 10 种 SOTA 因果发现算法，揭示当前深度学习方法在高维非线性动力系统上的不足。

**[Channel Matters Estimating Channel Influence For Multivariate Time Series](time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)**

:   提出 Channel-wise Influence (ChInf)——首个能量化多变量时间序列中不同通道对模型性能影响的影响函数方法，将 TracIn 从整体样本级分解到通道级，衍生出通道级异常检测和通道剪枝两个应用，在 5 个异常检测基准上排名第一。

**[Demandcast Global Hourly Electricity Demand Forecasting](time_series/demandcast_global_hourly_electricity_demand_forecasting.md)**

:   构建 DemandCast 开源机器学习框架，基于 XGBoost 融合历史电力需求、ERA5 温度和社会经济特征进行全球 56 个国家/地区的小时级电力需求预测，通过归一化目标变量（年度分数）实现跨国家可比，在时间外推测试集上达到 MAPE 9.2%。

**[Diffusion Transformers As Open-World Spatiotemporal Foundation Models](time_series/diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)**

:   提出 UrbanDiT，首个基于 Diffusion Transformer 的开放世界城市时空基础模型，通过统一的 prompt learning 框架整合异构数据类型（grid/graph）和多种任务（预测/插值/外推/填补），在多城市多场景下实现 SOTA 性能并展现强大的 zero-shot 泛化能力。

**[Diffusion Transformers For Imputation Statistical Efficiency And Uncertainty Qua](time_series/diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)**

:   本文从统计学习角度分析了条件扩散Transformer（DiT）在时间序列插补任务中的样本复杂度和不确定性量化性能，并提出混合掩码训练策略提升插补效果。

**[Exploring Neural Granger Causality With Xlstms Unveiling Temporal Dependencies I](time_series/exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)**

:   提出 GC-xLSTM，利用 xLSTM 架构结合新颖的动态稀疏优化策略，在多变量时间序列中挖掘 Granger 因果关系，在多个数据集上取得 SOTA 性能。

**[Friren Beyond Trajectories -- A Spectral Lens On Time](time_series/friren_beyond_trajectories_--_a_spectral_lens_on_time.md)**

:   提出 Fern (Forecasting with Ellipsoidal RepresentatioN)，通过逐 patch 的椭球体传输（旋转-缩放-平移）替代传统轨迹预测，在混沌系统上大幅超越基线，并在标准 LTSF 基准上保持竞争力。

**[How Foundational Are Foundation Models For Time Series Forecasting](time_series/how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   通过合成数据与真实电力消耗数据的系统性实验，揭示时间序列基础模型(TSFM)的零样本泛化能力高度依赖于预训练数据分布，在领域偏移场景下仅49.5K参数的轻量专用模型SAMFormer从头训练即可超越500M+参数的微调TimesFM。

**[How Patterns Dictate Learnability In Sequential Data](time_series/how_patterns_dictate_learnability_in_sequential_data.md)**

:   提出基于预测信息（predictive information）的信息论框架来量化序列数据中时间模式的强度，推导出将预测信息与最小可达风险联系起来的理论界，从而区分"模型不够好"还是"数据本身就不可预测"。

**[Improving Time Series Forecasting Via Instance-Aware Post-Hoc Revision](time_series/improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)**

:   PIR 提出实例感知的事后修正框架——通过不确定性估计识别预测失败实例，用局部修正（协变量+外生变量 Transformer）和全局修正（检索相似训练实例加权平均）的残差组合，作为即插即用模块使 SparseTSF MSE 降低 25.87%，PatchTST 降低 8.99%。

**[In-Context Learning Of Stochastic Differential Equations With Foundation Inferen](time_series/in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)**

:   提出FIM-SDE（基础推断模型），一个预训练的识别模型，能够从噪声时间序列数据中进行零样本(in-context)估计低维SDE的漂移和扩散函数，并通过快速微调进一步超越所有基线方法。

**[Ioncast A Deep Learning Framework For Forecasting Ionospheric Dynamics](time_series/ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)**

:   提出 IonCast 框架，基于 GraphCast 启发的图神经网络架构，融合多源异构物理驱动数据，实现全球总电子含量（TEC）的高精度时空预测。

**[Learning Time-Scale Invariant Population-Level Neural Representations](time_series/learning_time-scale_invariant_population-level_neural_representations.md)**

:   提出时间尺度增强预训练（TSAP）策略，通过在预训练阶段引入多种时间窗口长度的数据增强，使群体级神经信号基础模型对输入时间尺度具有不变性，在匹配和未见时间尺度上均显著提升解码性能。

**[Learning With Calibration Exploring Test-Time Computing Of Spatio-Temporal Forec](time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)**

:   提出 ST-TTC，一种轻量级测试时计算范式，通过频域相位-幅值校准器和闪电梯度更新机制，在推理阶段实时修正时空预测中的周期性偏差，无需修改骨干网络即可一致性提升多种模型性能。

**[Maestro Adaptive Sparse Attention And Robust Learning For Multimodal Dynamic Tim](time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)**

:   提出 MAESTRO 框架，通过符号化分词、自适应注意力预算、稀疏跨模态注意力和动态 MoE 路由，解决多模态时间序列中模态异质性和任意缺失的问题，在完整/缺失模态场景下均显著超越基线。

**[Martingale Score An Unsupervised Metric For Bayesian Rationality In Llm Reasonin](time_series/martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)**

:   提出 Martingale Score 作为无监督度量指标，基于贝叶斯统计中的鞅性质(Martingale property)来量化 LLM 推理过程中的信念固化(belief entrenchment)现象，发现该现象普遍存在且与准确率下降显著相关。

**[Masfin A Multi-Agent System For Decomposed Financial Reasoning And Forecasting](time_series/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)**

:   提出 MASFIN 多 agent 系统，将金融预测任务分解为多个子任务（宏观分析、行业分析、技术分析、情感分析等），由专门的 LLM agent 协作完成，实现比单一模型更准确和可解释的金融预测。

**[Neural Mjd Neural Non-Stationary Merton Jump Diffusion For Time Series Predictio](time_series/neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)**

:   提出 Neural MJD，用神经网络参数化非平稳 Merton 跳跃扩散模型，将预测建模为 SDE 仿真问题，结合时变 Itô 扩散（捕获连续漂移）和时变复合 Poisson 过程（建模突变跳跃），配合似然截断和 Euler-Maruyama with Restart 求解器实现可扩展学习与推理。

**[Nsw-Epnews A News-Augmented Benchmark For Electricity Price Forecasting With Llm](time_series/nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)**

:   提出首个融合新闻文本的电力价格预测基准 NSW-EPNews，系统评估传统模型和 LLM 在多模态电价预测中的表现，发现新闻特征对传统模型增益有限，而 LLM 存在严重幻觉问题。

**[Parallelization Of Non-Linear State-Space Models Scaling Up Liquid-Resistance Li](time_series/parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)**

:   提出 LrcSSM，通过约束液态电阻-液态电容（LRC）网络的 Jacobian 矩阵为对角形式，实现非线性 RNN 的精确高效并行化，在长序列分类任务上超越 Transformer、LRU、S5 和 Mamba 等 SOTA 方法。

**[Physics-Informed Reduced Order Modeling Of Time-Dependent Pdes Via Differentiabl](time_series/physics-informed_reduced_order_modeling_of_time-dependent_pdes_via_differentiabl.md)**

:   提出Φ-ROM框架，将可微分PDE求解器嵌入非线性降阶模型的训练过程中，通过求解器反馈直接约束潜在空间动态，使模型在泛化到未见参数/初始条件、长时间外推、稀疏观测数据恢复等方面显著优于纯数据驱动ROM和其他物理信息方法。

**[Planu Large Language Model Reasoning Through Planning Under Uncertainty](time_series/planu_large_language_model_reasoning_through_planning_under_uncertainty.md)**

:   提出PlanU——一种在MCTS中用分位数分布建模节点回报、并通过Upper Confidence Bounds with Curiosity (UCC)分数平衡探索与利用的LLM决策方法，首次系统性地同时处理LLM不确定性和环境不确定性，在多个随机环境基准上显著优于现有方法。

**[Probability Calibration For Precipitation Nowcasting](time_series/probability_calibration_for_precipitation_nowcasting.md)**

:   提出了期望阈值校准误差（ETCE）作为降水临近预报中更合理的概率校准度量，并将计算机视觉中的后处理校准技术扩展到预报领域，通过结合前置时间条件的选择性缩放（Selective Scaling）方法将模型校准误差降低高达23.5%。

**[Rivermamba A State Space Model For Global River Discharge And Flood Forecasting](time_series/rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)**

:   首个能在 0.05°（~5.5km）全球网格上做 7 天河流流量预报的深度学习模型——用空间填充曲线将 3D 时空点序列化后输入双向 Mamba block，结合 ECMWF HRES 气象预报，在 1.5-500 年重现期洪水检测上 F1 =0.459 超越 LSTM（0.358）和物理模型 GloFAS。

**[Rotary Masked Autoencoders Are Versatile Learners](time_series/rotary_masked_autoencoders_are_versatile_learners.md)**

:   提出 RoMAE，将旋转位置编码（RoPE）扩展到连续位置并与掩码自编码器（MAE）结合，无需任何时间序列特定的架构修改即可在不规则时间序列、图像和音频等多种模态上达到或超越专用模型的性能。

**[Scalable Signature Kernel Computations For Long Time Series Via Local Neumann Se](time_series/scalable_signature_kernel_computations_for_long_time_series_via_local_neumann_se.md)**

:   提出 PowerSig，通过自适应截断的局部 Neumann 级数展开高效计算签名核（signature kernel），将内存从 $O(\ell^2)$ 降到 $O(\ell P)$，使签名核可扩展到单GPU上百万级长度的时间序列。

**[Scatterad Temporal-Topological Scattering Mechanism For Time Series Anomaly Dete](time_series/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)**

:   提出"散射性"（scattering）作为异常检测的新归纳偏置——异常样本在高维表示空间中比正常样本分布更分散，通过双编码器（时间+拓扑）+ 超球面散射中心约束 + 对比融合学习时拓扑联合表示，在 6 个工业 IoT 数据集上 15/24 设置取得最佳。

**[Selective Learning For Deep Time Series Forecasting](time_series/selective_learning_for_deep_time_series_forecasting.md)**

:   提出选择性学习（Selective Learning）策略，通过不确定性掩码和异常掩码组成的双掩码机制筛选可泛化时间步计算 MSE 损失，在 8 个数据集上为 Informer 降低 37.4% MSE、TimesNet 降低 8.4%、iTransformer 降低 6.5%。

**[Sempo Lightweight Foundation Models For Time Series Forecasting](time_series/sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出SEMPO——仅用6.5M参数和83M时间点预训练的轻量级时间序列基础模型，通过能量感知频谱分解和混合提示Transformer，在零样本和少样本预测中超越参数量百倍以上的大型基础模型。

**[Simple And Efficient Heterogeneous Temporal Graph Neural Network](time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)**

:   提出 SE-HTGNN，通过动态注意力机制将时序建模融入空间学习，并用 LLM 初始化注意力系数，在异构时序图任务上实现 10 倍加速的同时保持最优预测精度。

**[Statistical Guarantees For High-Dimensional Stochastic Gradient Descent](time_series/statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)**

:   将高维非线性时间序列的耦合技术引入在线学习，首次为常数学习率 SGD 及其 Ruppert-Polyak 平均变体在高维（$\ell^s$ 和 $\ell^\infty$ 范数下）提供了严格的矩收敛界和高概率集中界。

**[Strap Spatio-Temporal Pattern Retrieval For Out-Of-Distribution Generalization](time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)**

:   提出 StRap 框架，通过构建包含空间、时间和时空三维键值对的模式库，在推理时检索与当前输入最相似的历史模式并自适应融合，有效应对流式时空数据中的分布偏移（STOOD）问题。

**[Structured Temporal Causality For Interpretable Multivariate Time Series Anomaly](time_series/structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)**

:   提出OracleAD框架，通过为每个变量学习因果嵌入（LSTM编码+注意力池化）并构建稳定潜在结构（SLS）来建模正常状态下的变量间关系，结合预测误差和SLS偏离的双重评分机制实现可解释的多变量时间序列异常检测与根因定位。

**[Synthetic Series-Symbol Data Generation For Time Series Foundation Models](time_series/synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)**

:   提出 Series-Symbol (S²) 数据生成机制和 SymTime 双模态基础模型，利用 Takens 定理和符号动力学理论生成无限规模的合成时序-符号配对数据（40M 对/50B token），通过跨模态对比学习预训练在 5 大时序任务上达到与真实数据预训练模型竞争的性能。

**[Syntsbench Rethinking Temporal Pattern Learning In Deep Learning Models For Time](time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)**

:   提出SynTSBench合成数据驱动评估范式，通过可编程的特征配置和理论最优基准，系统评估时间序列预测模型在趋势、周期、依赖性、噪声鲁棒性等维度的实际建模能力。

**[Time-Imm A Dataset And Benchmark For Irregular Multimodal Multivariate Time Seri](time_series/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)**

:   构建 Time-IMM 数据集——首个按因果机制分类不规则性的多模态多变量时序 benchmark（9 种不规则类型分为触发/约束/伪影三大类，9 个数据集），配套 IMM-TSF 预测库支持异步多模态融合，实验表明显式建模多模态在不规则时序上平均降低 MSE 6.71%，最高达 38.38%。

**[Time-O1 Time-Series Forecasting Needs Transformed Label Alignment](time_series/time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，通过将标签序列变换为去相关且按重要性排序的主成分，解决时间序列预测中 TMSE 损失的自相关偏差和任务过载问题，实现与多种预测模型兼容的 SOTA 性能。

**[Timeperceiver An Encoder-Decoder Framework For Generalized Time-Series Forecasti](time_series/timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)**

:   提出 TimePerceiver 统一编码器-解码器框架，通过广义化预测任务（同时包含外推、插值和填补）以及潜在瓶颈编码器 + 查询式解码器设计，在 8 个标准基准上取得全面 SOTA。

**[Transformer Embeddings For Fast Microlensing Inference](time_series/transformer_embeddings_for_fast_microlensing_inference.md)**

:   本文将Transformer编码器与神经后验估计（NPE）结合，直接从稀疏、噪声、不等间隔的微引力透镜光变曲线中进行快速且校准良好的参数推断，速度比传统MCMC快10⁴倍以上。

**[Universal Spectral Tokenization Via Self-Supervised Panchromatic Representation ](time_series/universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)**

:   提出首个通用光谱 Tokenizer，通过连续波长嵌入和自监督重建目标，在原始波长网格上联合训练异构天文光谱数据（SDSS/DESI/GALAH/APOGEE），生成对齐、均匀且物理有意义的表征。

**[Walrus Wavelets For Long-Range Representation Using Ssms](time_series/walrus_wavelets_for_long-range_representation_using_ssms.md)**

:   提出 WaLRUS，基于 Daubechies 小波构建状态空间模型 (SSM)，作为 SaFARi 框架的新实现，扩展了 SSM 家族的多样性，在长程依赖建模中展现独特优势。

**[Wavelet Canonical Coherence For Nonstationary Signals](time_series/wavelet_canonical_coherence_for_nonstationary_signals.md)**

:   提出 WaveCanCoh 框架，将经典的典型相干分析（canonical coherence）扩展到小波域，基于多变量局部平稳小波（MvLSW）模型实现对非平稳多变量时间序列两组信号间时变、尺度特定的典型相干性估计。

**[Xlstm-Mixer Multivariate Time Series Forecasting By Mixing Via Scalar Memories](time_series/xlstm-mixer_multivariate_time_series_forecasting_by_mixing_via_scalar_memories.md)**

:   提出 xLSTM-Mixer，首次将扩展长短期记忆网络（sLSTM）与混合架构（Mixer）结合，通过时间混合、联合时间-变量混合和多视角混合三阶段架构实现多变量长期时间序列预测的 SOTA 性能，同时保持极低的内存占用。

---

## ✂️ 语义分割 { #segmentation }

**[Alligat0R Pre-Training Through Co-Visibility Segmentation For Relative Camera Po](segmentation/alligat0r_pre-training_through_co-visibility_segmentation_for_relative_camera_po.md)**

:   用共视性分割（covisibility segmentation）替代 CroCo 的跨视图补全作为双目视觉预训练任务，对每个像素预测"共视/遮挡/视野外"三类标签，在低重叠场景下显著超越 CroCo，RUBIK 基准总体成功率 60.3% 排第一。

**[Argenseg Image Segmentation With Autoregressive Image Generation Model](segmentation/argenseg_image_segmentation_with_autoregressive_image_generation_model.md)**

:   提出ARGenSeg——首个利用自回归图像生成范式实现图像分割的统一MLLM框架，让模型直接输出visual tokens并通过VQ-VAE解码为分割mask，无需额外分割头，搭配next-scale prediction并行生成策略实现4×加速，在RefCOCO/+/g上以更少训练数据超越SOTA。

**[Attention As Discrete-Time Markov Chains](segmentation/attention_as_discrete-time_markov_chains.md)**

:   将 softmax 归一化后的注意力矩阵重新解读为离散时间 Markov 链（DTMC）的转移概率矩阵，提出多跳注意力（Multi-Bounce）和 TokenRank（稳态分布，类似 PageRank）来捕获间接注意力路径和全局 token 重要性，在 ImageNet 分割上达 94.29% mAP，并增强 Self-Attention Guidance 的图像生成质量。

**[Connectomebench Can Llms Proofread The Connectome](segmentation/connectomebench_can_llms_proofread_the_connectome.md)**

:   提出 ConnectomeBench，首个评估多模态 LLM 在连接组校对（片段识别、分裂错误修正、合并错误检测）三项关键任务上能力的标准化基准；o4-mini 在分裂修正多选任务达 85%，但合并错误检测仍显著落后于人类专家。

**[Cos3D Collaborative Open-Vocabulary 3D Segmentation](segmentation/cos3d_collaborative_open-vocabulary_3d_segmentation.md)**

:   提出 COS3D——一种协作式 prompt-分割框架，通过构建实例场（instance field）和语言场（language field）组成的协作场，在训练阶段利用实例到语言的特征映射构建语言场，在推理阶段利用语言到实例的自适应 prompt 精炼生成精确分割，在两个主流基准上大幅超越现有方法。

**[Diffusion-Driven Two-Stage Active Learning For Low-Budget Semantic Segmentation](segmentation/diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)**

:   提出两阶段主动学习流程（覆盖性→不确定性），利用预训练扩散模型的多尺度特征实现极低标注预算下的高效语义分割。

**[Exploring Structural Degradation In Dense Representations For Self-Supervised Le](segmentation/exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)**

:   发现并系统研究了自监督学习中"稠密退化"（SDD）现象——训练越久分类越好但稠密任务性能反而下降，提出 DSE 度量和基于 DSE 的模型选择/正则化策略，平均提升 mIoU 3.0%。

**[Fast And Fluent Diffusion Language Models Via Convolutional Decoding And Rejecti](segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)**

:   通过卷积解码归一化（替代硬半自回归分块）和基于规则的拒绝微调 R2FT，在 128 步推理下实现与 512+ 步相当的扩散语言模型生成质量，达到 DLM 领域 SOTA。

**[Finers Fine-Grained Reasoning And Segmentation Of Small Objects With Reinforceme](segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)**

:   提出 FineRS 两阶段 MLLM 强化学习框架（全局语义探索 GSE → 局部感知精化 LPR），通过 locate-informed retrospective reward 耦合两阶段，在自建 FineRS-4k UAV 高分辨率数据集上实现超小目标的推理与分割，gIoU 达 55.1%（超 Seg-Zero† 8.5%），同时支持 VQA（MVQA 83.3%）。

**[Gtpbd A Fine-Grained Global Terraced Parcel And Boundary Dataset](segmentation/gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)**

:   构建首个全球性细粒度梯田地块与边界数据集GTPBD，包含47,537张高分辨率影像（0.5-0.7m）和超20万个人工标注地块，提供三级标签支持语义分割、边缘检测、地块提取和无监督域适应四项任务，并在20种方法上进行全面基准评测。

**[Haodiff Human-Aware One-Step Diffusion Via Dual-Prompt Guidance](segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)**

:   提出HAODiff，一种人体感知的单步扩散模型，通过三分支双提示引导（DPG）生成自适应正负提示对，结合显式人体运动模糊（HMB）退化管线和分类器自由引导（CFG），在人体图像复原任务上大幅超越现有SOTA方法。

**[Hopadiff Holistic-Partial Aware Fourier Conditioned Diffusion For Referring Huma](segmentation/hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)**

:   首次提出指称人体动作分割(RHAS)任务——通过文本描述定位多人视频中特定个体并做帧级动作分割。构建了包含133部电影、137个动作类别、33小时视频的RHAS133数据集，并提出基于全局-局部感知傅里叶条件扩散的HopaDIFF框架，在多种评估设置下显著超越现有基线。

**[Humancrafter Synergizing Generalizable Human Reconstruction And Semantic 3D Segm](segmentation/humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)**

:   提出HumanCrafter——首个统一单图3D人体重建与人体部位语义分割的前馈框架，通过人体几何先验引导的Transformer聚合多视角特征，结合DINOv2自监督语义先验构建3D特征场，在2K2K和THuman2.1上同时超越现有3D重建和分割SOTA。

**[Instructsam A Training-Free Framework For Instruction-Oriented Remote Sensing Ob](segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)**

:   定义指令导向目标计数/检测/分割(InstructCDS)新任务，构建EarthInstruct遥感基准（覆盖开放词汇/开放端/开放子类三种设置），提出InstructSAM——无需训练的框架：LVLM解析指令+计数、SAM2生成掩码提议、CLIP计算相似度，通过二进制整数规划(BIP)在计数约束下实现掩码-标签最优匹配，推理时间近乎恒定且优于专用基线。

**[Interpreting Resnet-Based Clip Via Neuron-Attention Decomposition](segmentation/interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)**

:   提出神经元-注意力分解方法解释CLIP-ResNet：将模型输出分解为神经元与注意力池化头的成对贡献路径，发现这些neuron-head对可用单一方向近似、具有稀疏性且捕获子概念，并将其应用于免训练语义分割（PASCAL Context上mIoU 26.2%，超MaskCLIP 15%）和数据集分布偏移监测。

**[Langhops Language Grounded Hierarchical Open-Vocabulary Part Segmentation](segmentation/langhops_language_grounded_hierarchical_open-vocabulary_part_segmentation.md)**

:   提出LangHOPS，首个基于多模态大语言模型（MLLM）的开放词汇物体-部件实例分割框架，在语言空间中建立object-part层次关系，利用MLLM的知识和推理能力链接多粒度概念，在PartImageNet上以56.9% AP超越SOTA 5.5%，跨数据集设置超4.8%。

**[Mars-Bench A Benchmark For Evaluating Foundation Models For Mars Science Tasks](segmentation/mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)**

:   本文提出 Mars-Bench——首个面向火星科学任务的综合基准，涵盖20个数据集（分类/分割/目标检测三大任务类型），系统评估了 ImageNet 预训练模型、地球观测基础模型和视觉语言模型在火星数据上的表现，发现当前通用模型在火星领域仍有明显不足，呼吁开发火星专用基础模型。

**[Mechanistic Interpretability Of Rnns Emulating Hidden Markov Models](segmentation/mechanistic_interpretability_of_rnns_emulating_hidden_markov_models.md)**

:   通过训练 RNN 模拟 HMM 的发射统计，然后反向工程揭示 RNN 如何利用噪声驱动的轨道动力学、结构化连接（噪声积分群体 + kick 神经元）和自诱导随机共振机制来实现离散随机状态转换。

**[Multihuman-Testbench Benchmarking Image Generation For Multiple Humans](segmentation/multihuman-testbench_benchmarking_image_generation_for_multiple_humans.md)**

:   提出 MultiHuman-Testbench，首个系统性评估多人图像生成的基准，包含 1800 个测试样本配对 5550 张人脸图像，以及基于匈牙利匹配的身份相似度等多维度评估指标，并提出区域隔离和隐式匹配技术提升现有方法性能。

**[Omnisegmentor A Flexible Multi-Modal Learning Framework For Semantic Segmentatio](segmentation/omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)**

:   OmniSegmentor 构建了含 5 种视觉模态的大规模 ImageNeXt 数据集（1.2M 样本），提出随机选择补充模态与 RGB 对齐的高效预训练策略，首次实现灵活的多模态预训练-微调流水线，在 6 个多模态语义分割基准上刷新 SOTA。

**[Panoptic Captioning An Equivalence Bridge For Image And Text](segmentation/panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)**

:   提出 Panoptic Captioning 新任务追求图像的"最小文本等价"——定义包含实体标签、位置（bbox）、属性、关系和全局状态五个维度的全面结构化描述，通过 PancapEngine 数据引擎和 PancapChain 解耦多阶段方法，13B 模型即超越 InternVL-2.5-78B 和 GPT-4o。

**[Partnext A Next-Generation Dataset For Fine-Grained And Hierarchical 3D Part Und](segmentation/partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)**

:   提出 PartNeXt，一个包含 23,519 个高质量带纹理 3D 模型、跨 50 个类别的细粒度层级部件标注数据集，并建立了类别无关部件分割和 3D 部件问答两个基准测试，揭示了当前方法在细粒度部件理解上的显著不足。

**[Partonomy Large Multimodal Models With Part-Level Visual Understanding](segmentation/partonomy_large_multimodal_models_with_part-level_visual_understanding.md)**

:   提出 Partonomy 部件级分割 benchmark（862 部件标签/534 物体标签）和 Plum 模型（用 span 标记替代 [SEG] token + mask 反馈循环），发现 SOTA 分割 LMM 在部件理解上仅 5.9% gIoU，Plum 通过避免分布偏移和利用历史预测显著提升。

**[Re-Coding For Uncertainties Edge-Awareness Semantic Concordance For Resilient Ev](segmentation/re-coding_for_uncertainties_edge-awareness_semantic_concordance_for_resilient_ev.md)**

:   提出 Edge-awareness Semantic Concordance（ESC）框架，利用语义边缘作为异质 Event 和 RGB 模态的中间桥梁，通过边缘字典的离散潜空间建模实现跨模态特征对齐和不确定性优化，在极端条件下超越 SOTA 2.55% mIoU。

**[Revisiting Generative Infrared And Visible Image Fusion Based On Human Cognitive](segmentation/revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)**

:   HCLFuse 基于信息瓶颈原理和最优传输理论进行模态对齐，设计变分瓶颈编码器（VBE）+ 物理引导条件扩散模型，融合热传导/结构保持/物理一致性三种约束到扩散过程中，在 MSRS 数据集上梯度指标 AG 提升 69.87%，空间频率 SF 提升 39.41%。

**[Robust Ego-Exo Correspondence With Long-Term Memory](segmentation/robust_ego-exo_correspondence_with_long-term_memory.md)**

:   提出LM-EEC，基于SAM 2的自中心-外中心(ego-exo)视频跨视角目标分割框架，通过Memory-View MoE自适应融合记忆特征与跨视角特征，配合双记忆库压缩策略保持长期信息，在EgoExo4D基准上大幅超越现有方法（Ego2Exo IoU 54.98 vs 38.26）。

**[Robust Egocentric Referring Video Object Segmentation Via Dual-Modal Causal Inte](segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)**

:   提出 CERES 框架，通过双模态因果干预（语言后门调整消除数据集统计偏差 + 视觉前门调整利用深度信息构建因果中介变量）来解决第一人称视频指代分割中的语言偏差和视觉混淆问题，在 VISOR/VOST/VSCOS 上取得 SOTA。

**[Roma Scaling Up Mamba-Based Foundation Models For Remote Sensing](segmentation/roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)**

:   提出RoMA——首个面向遥感领域的Mamba架构自监督自回归预训练框架，通过自适应旋转编码策略和多尺度token预测机制，解决遥感图像的方向多样性和尺度极端变化问题，验证了Mamba在遥感领域遵循数据和参数缩放定律。

**[Safire Saccade-Fixation Reiteration With Mamba For Referring Image Segmentation](segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)**

:   提出 SaFiRe 框架，模拟人类"扫视-注视"两阶段认知过程，利用 Mamba 的扫描-更新特性实现线性复杂度的多轮细化，有效处理歧义指代表达下的图像分割任务。

**[Sam-R1 Leveraging Sam For Reward Feedback In Multimodal Segmentation Via Reinfor](segmentation/sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)**

:   SAM-R1 提出了一个端到端的推理分割框架，首次将 SAM 作为强化学习训练回路中的奖励提供者，结合分级IoU精度奖励、非对称裁剪和 token 级损失归一化的改进 GRPO 算法，仅用 3K 训练样本即在 ReasonSeg 零样本设定下超越 Seg-Zero 等方法，gIoU 达 60.2%。

**[Sansa Unleashing The Hidden Semantics In Sam2 For Few-Shot Segmentation](segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)**

:   SANSA 发现 SAM2 虽然以类别无关方式预训练，但其特征中隐含了丰富的语义结构；通过在冻结的 SAM2 Image Encoder 最后两层插入轻量 AdaptFormer 适配器，将 Memory Attention 机制从视觉相似性匹配重定向为语义相似性匹配，以统一架构实现了 few-shot 分割的 SOTA，同时比竞争方法快 3 倍以上、参数量小 4-5 倍。

**[Seg-Var Image Segmentation With Visual Autoregressive Modeling](segmentation/seg-var_image_segmentation_with_visual_autoregressive_modeling.md)**

:   Seg-VAR 将图像分割重新定义为条件自回归掩码生成问题，通过引入 seglat（分割掩码的潜在表示）和空间感知颜色映射将分割掩码编码为可由 VAR 模型处理的离散 token，在 COCO、Cityscapes、ADE20K 上的语义/实例/全景分割任务中全面超越 Mask2Former 等判别式方法和 GSS 等生成式方法。

**[Seg4Diff Unveiling Open-Vocabulary Segmentation In Text-To-Image Diffusion Trans](segmentation/seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)**

:   通过系统分析多模态扩散Transformer（MM-DiT）的联合注意力机制，发现特定层（"语义定位专家层"）天然具备高质量语义分割能力，并提出轻量微调方法MAGNET同时提升分割与生成性能。

**[Self-Supervised Synthetic Pretraining For Inference Of Stellar Mass Embedded In ](segmentation/self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)**

:   本文提出了一种"合成数据驱动的自监督预训练"范式：先用 Flame 算法生成 100 万张合成分形图像，对 ViT-L/16 编码器进行 DINOv2 自监督预训练，然后将冻结的编码器直接迁移到极其有限的磁流体动力学（MHD）恒星形成模拟数据上，通过 kNN 回归实现恒星质量预测（$R^2=0.81$），通过 PCA 投影实现零样本无监督语义分割，性能略优于在相同数据上训练的全监督 ResNet-18 基线。

**[Srsr Enhancing Semantic Accuracy In Real-World Image Super-Resolution With Spati](segmentation/srsr_enhancing_semantic_accuracy_in_real-world_image_super-resolution_with_spati.md)**

:   SRSR提出一种无需训练的即插即用框架，通过空间重聚焦交叉注意力(SRCA)和空间定向CFG(STCFG)两个推理时模块，解决扩散超分方法中文本引导导致的语义幻觉问题，在保真度和感知质量上全面超越7个SOTA基线。

**[Stead Robust Provably Secure Linguistic Steganography With Diffusion Language Mo](segmentation/stead_robust_provably_secure_linguistic_steganography_with_diffusion_language_mo.md)**

:   提出STEAD，首个基于扩散语言模型（DLM）的可证安全且鲁棒的语言隐写术方法，利用DLM并行去噪的特性找到"鲁棒位置"进行信息嵌入，结合重复纠错编码和邻域搜索策略，抵御token级别的替换、插入、删除攻击。

**[Step A Unified Spiking Transformer Evaluation Platform For Fair And Reproducible](segmentation/step_a_unified_spiking_transformer_evaluation_platform_for_fair_and_reproducible.md)**

:   STEP 是首个统一的脉冲 Transformer (Spiking Transformer) 评估平台，支持分类/分割/检测多任务、多后端（SpikingJelly/BrainCog/BrainPy），通过系统消融揭示了当前脉冲 Transformer 严重依赖卷积前端、注意力贡献有限、时序建模能力不足的关键发现，并提出了考虑位宽稀疏性和内存访问的统一能耗分析框架。

**[Tabrag Improving Tabular Document Question Answering For Retrieval Augmented Gen](segmentation/tabrag_improving_tabular_document_question_answering_for_retrieval_augmented_gen.md)**

:   提出 TabRAG，一种基于解析的 RAG 框架，通过布局分割将文档分解为细粒度组件，使用视觉语言模型将表格提取为层次化结构表示，并集成自生成上下文学习模块来适配多种表格格式，在表格文档问答上全面优于现有解析技术。

**[Torch-Uncertainty A Deep Learning Framework For Uncertainty Quantification](segmentation/torch-uncertainty_a_deep_learning_framework_for_uncertainty_quantification.md)**

:   Torch-Uncertainty 是首个统一、可扩展、领域通用且以评估为中心的 PyTorch/Lightning 不确定性量化 (UQ) 框架，集成了 6 大类 UQ 方法、26 种评估指标和 27 个即插即用数据集，覆盖分类、分割、回归等任务，并提供了完整的基准测试结果。

**[Towards Robust Pseudo-Label Learning In Semantic Segmentation An Encoding Perspe](segmentation/towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)**

:   提出 ECOCSeg，用纠错输出码（ECOC）替代 one-hot 编码来表示语义类别，将 N 类分类分解为 K 个二分类子任务，配合 bit 级伪标签去噪和定制优化损失，显著提升 UDA 和 SSL 语义分割中伪标签学习的鲁棒性。

**[Towards Unsupervised Domain Bridging Via Image Degradation In Semantic Segmentat](segmentation/towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)**

:   提出 DiDA，通过将图像退化操作形式化为扩散前向过程来构建源域和目标域之间的连续中间域，结合语义偏移补偿机制，作为即插即用模块显著提升现有 UDA 语义分割方法的性能。

**[Unipixel Unified Object Referring And Segmentation For Pixel-Level Visual Reason](segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)**

:   UniPixel 提出了首个端到端统一对象引用 (referring) 和分割 (segmentation) 的大型多模态模型，通过创新的 Object Memory Bank 设计将稀疏视觉提示转化为稠密对象掩码特征并注入推理过程，在 10 个基准上实现 SOTA，还引入了需要同时完成引用、分割和问答的 PixelQA 新任务。

**[Unveiling The Spatial-Temporal Effective Receptive Fields Of Spiking Neural Netw](segmentation/unveiling_the_spatial-temporal_effective_receptive_fields_of_spiking_neural_netw.md)**

:   提出时空有效感受野（ST-ERF）分析框架来诊断 Transformer-based SNN 在视觉长序列建模中的瓶颈——缺乏全局感受野，并据此设计 MLPixer 和 SRB 两种通道混合器来增强 SNN 的全局建模能力。

**[Vanish Into Thin Air Cross-Prompt Universal Adversarial Attacks For Sam2](segmentation/vanish_into_thin_air_cross-prompt_universal_adversarial_attacks_for_sam2.md)**

:   提出UAP-SAM2——首个针对SAM2的跨提示通用对抗攻击方法，通过双重语义偏移框架（帧内语义混淆+帧间语义不一致）生成一个通用扰动，使SAM2在不同视频、帧和提示下的分割目标"消失无踪"。

**[Vision Transformers With Self-Distilled Registers](segmentation/vision_transformers_with_self-distilled_registers.md)**

:   提出PH-Reg（Post Hoc Registers），一种高效的自蒸馏方法，无需标注数据和完整重训练即可为已有预训练ViT添加register token，通过测试时增强去噪教师特征+自蒸馏学生网络，有效消除ViT密集特征中的伪影token，提升分割和深度估计性能。

---

## ⚡ LLM效率 { #llm_efficiency }

**[3Model Speculative Decoding](llm_efficiency/3model_speculative_decoding.md)**

:   在标准的draft-target两模型推测解码的中间插入一个"qualifier"模型，构成三层金字塔式解码架构（PyramidSD），利用模型家族天然的熵梯度来分级过滤token，以模糊接受准则放宽匹配阈值，实现最高1.91×的速度提升（在RTX 4090上达到124 tok/s）。

**[A Unified Framework For Establishing The Universal Approxima](llm_efficiency/a_unified_framework_for_establishing_the_universal_approxima.md)**

:   建立了统一的理论框架证明各类Transformer架构的万能逼近性(UAP)，核心条件仅两个——前馈层的非线性仿射不变性和注意力层的token可区分性——并利用解析性假设将后者简化为仅需检验两样本情况，成功覆盖softmax、RBF kernel、Performer、BigBird、Linformer等多种实用架构。

**[Advancing Expert Specialization For Better Moe](llm_efficiency/advancing_expert_specialization_for_better_moe.md)**

:   通过正交性损失（减少专家间投影重叠）和方差损失（增大路由分数差异）双目标优化，在不修改 MoE 架构的前提下将专家重叠减少 45%、路由方差提升 150%，11 个基准任务平均提升 23.79%，同时完全保持负载均衡。

**[Approximately Aligned Decoding](llm_efficiency/approximately_aligned_decoding.md)**

:   提出 Approximately Aligned Decoding (AprAD)，一种利用投机解码（speculative decoding）中的前缀选择算法来实现LLM受约束生成的方法——在遇到约束违反时，既不像约束生成那样仅回退一步（导致极端概率放大），也不像ASAp那样完全重新采样（计算成本过高），而是通过投机采样智能选择回退位置，在输出分布失真和计算效率之间取得良好平衡。

**[Constant Bit-Size Transformers Are Turing Complete](llm_efficiency/constant_bit-size_transformers_are_turing_complete.md)**

:   首次证明常数 bit 精度、固定参数数量的 Transformer（仅允许上下文窗口增长）是图灵完备的，并建立了精确的复杂度等价关系 WINDOW[s(n)] = SPACE[s(n)]，表明扩展上下文窗口——而非模型尺寸——已足以实现通用计算。

**[Critical Batch Size Revisited A Simple Empirical Approach To Large-Batch Languag](llm_efficiency/critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)**

:   提出 branched training 方法直接实证测量临界 batch size (CBS)，发现 CBS 在训练早期快速增长后趋于平稳且不依赖模型规模，据此设计 batch size warmup 策略以 43% 更少的梯度步数达到同等甚至更优的训练 loss。

**[Deep Compositional Phase Diffusion For Long Motion Sequence Generation](llm_efficiency/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)**

:   提出 Compositional Phase Diffusion 框架，在 ACT-PAE 建立的频域相位空间中用 SPDM 和 TPDM 分别处理语义对齐和过渡连续性，实现长程组合式动作序列生成，在 BABEL-TEACH 上达到 SOTA。

**[Dense Associative Memory With Epanechnikov Energy](llm_efficiency/dense_associative_memory_with_epanechnikov_energy.md)**

:   提出基于 Epanechnikov 核的 log-sum-ReLU（LSR）能量函数替代传统的 log-sum-exp（LSE），在 Dense Associative Memory 中首次实现了"精确记忆所有模式 + 同时涌现新的创造性局部极小"的共存，且保持指数级记忆容量。

**[Disc Dynamic Decomposition Improves Llm Inference Scaling](llm_efficiency/disc_dynamic_decomposition_improves_llm_inference_scaling.md)**

:   DISC 提出了一种动态分解算法，在推理时根据每一步的 z-score（采样奖励的标准化最大值）自动、递归地调整推理步骤的粒度——困难步骤分更细、简单步骤一步跨过——可以即插即用地与贪心搜索、Beam Search、MCTS 结合，在 APPS、MATH、LiveCodeBench 上以更少的 token 预算达到更高的 pass@k。

**[Document Summarization With Conformal Importance Guarantees](llm_efficiency/document_summarization_with_conformal_importance_guarantees.md)**

:   首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

**[Dynamics Of Spontaneous Topic Changes In Next Token Prediction With Self-Attenti](llm_efficiency/dynamics_of_spontaneous_topic_changes_in_next_token_prediction_with_self-attenti.md)**

:   从理论和实验两方面研究自注意力模型中"自发主题切换"的动力学机制，证明在单层 self-attention 模型中：(1) 混合主题训练保持原主题的 token 优先级顺序；(2) 主题切换仅在低优先级 token 数量超过高优先级 token 时发生；(3) 更长输入和更模糊主题不会增加切换概率——与人类认知相反。

**[Efficient Training-Free Online Routing For High-Volume Multi-Llm Serving](llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)**

:   提出首个无需训练的在线 LLM 路由算法 PORT，通过近似最近邻搜索估计查询特征，并在少量初始查询上一次性优化对偶变量作为路由权重，在有限 token 预算下实现接近离线最优 ($1-o(1)$ 竞争比) 的路由性能，平均较基线提升 3.55× 性能、1.85× 成本效率和 4.25× 吞吐量。

**[Frequency-Aware Token Reduction For Efficient Vision Transformer](llm_efficiency/frequency-aware_token_reduction_for_efficient_vision_transformer.md)**

:   从频域视角提出 frequency-aware token reduction，将 token 分为高频（HF）和低频（LF）两组，选择性保留 HF token 并将 LF token 聚合为 DC token，在缓解 rank collapse 的同时减少 ViT 的计算量，在 30% token 减少率下多个模型上超越现有 SOTA。

**[From Shortcut To Induction Head How Data Diversity Shapes Algorithm Selection In](llm_efficiency/from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)**

:   通过严格的理论分析证明了预训练数据的多样性（由"max-sum ratio"刻画）决定了单层Transformer学到的是可泛化的induction head还是无法OOD泛化的位置捷径，并给出了使模型学会induction head的最优预训练分布。

**[Hardware-Aligned Hierarchical Sparse Attention For Efficient Long-Term Memory Ac](llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)**

:   提出层次化稀疏注意力（HSA）及 RAMba 架构，通过两阶段 token-to-chunk 相关性学习与硬件对齐 kernel 设计，让 Mamba 获得高效长程随机访问能力，仅在 4K 上下文预训练即可在 64M passkey retrieval 上达到 100% 准确率。

**[Hierarchical Balance Packing Towards Efficient Supervised Fine-Tuning For Long-C](llm_efficiency/hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)**

:   提出层次均衡打包（HBP）方法，通过多级打包分组、均衡批处理、自适应序列并行和稳定损失归一化，解决长短上下文混合 SFT 中的注意力计算不均衡和通信浪费问题，在 DeepSeek-V2 (236B) 上实现 2.4× 训练加速且性能无损。

**[L-Mtp Leap Multi-Token Prediction Beyond Adjacent Context For Large Language Mod](llm_efficiency/l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)**

:   L-MTP 在多token预测（MTP）基础上引入跳跃机制，预测非相邻位置的token（如位置1,3,5,7而非1,2,3,4），通过"后向查找"解码策略复用先前预测填补空隙，在3B-12B模型上实现22%推理加速的同时保持或提升任务性能。

**[Learning In Compact Spaces With Approximately Normalized Transformer](llm_efficiency/learning_in_compact_spaces_with_approximately_normalized_transformer.md)**

:   提出 anGPT（近似归一化 Transformer），利用高维空间中向量范数的集中现象，用简单标量乘法替代逐层精确归一化，在消除权重衰减和学习率预热的同时实现了相比 GPT+（含 QK-norm）40% 的收敛加速，仅增加 3% 运行时开销。

**[Linear Attention For Efficient Bidirectional Sequence Modeling](llm_efficiency/linear_attention_for_efficient_bidirectional_sequence_modeling.md)**

:   提出 Lion 框架，首次系统地将线性 Transformer 扩展到双向序列建模，统一了全线性注意力、双向 RNN 和分块并行三种等价表示，在图像分类和 MLM 任务上训练速度比 SSM 快达 10 倍且性能可比 softmax Transformer。

**[Long-Context Modeling With Dynamic Hierarchical Sparse Attention For On-Device L](llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)**

:   提出动态分层稀疏注意力 (DHSA)，通过自适应 chunk 分割 + chunk 级相似度预测 + 上采样到 token 级的分层框架，在不重训基座模型的前提下将密集注意力替换为稀疏注意力，在 Gemma2/3 上实现与密集注意力同等精度、20-60% prefill 延迟降低和 35% 峰值内存节省。

**[Loogle V2 Are Llms Ready For Real World Long Dependency Challenges](llm_efficiency/loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)**

:   构建覆盖法律/金融/游戏/代码四大真实领域、长度16K-2M token的长依赖推理基准LooGLE v2，设计10类领域特定任务共1,934个QA实例，评估10个LLM发现最强模型GPT-4.1仅59.2%，揭示当前LLM在真实长依赖场景下的根本不足。

**[Mozart Modularized And Efficient Moe Training On 35D Wafer-Scale Chiplet Archite](llm_efficiency/mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)**

:   提出 Mozart 算法-硬件协同设计框架，通过专家聚类分配、细粒度流式调度和 3.5D 晶粒架构（NoP-Tree + 分层存储），在三个 MoE-LLM 上实现 1.9× 以上的训练加速。

**[Not All Splits Are Equal Rethinking Attribute Generalization Across Unrelated Ca](llm_efficiency/not_all_splits_are_equal_rethinking_attribute_generalization_across_unrelated_ca.md)**

:   本文首次系统评估了属性预测任务中训练/测试划分策略对泛化性能的影响,提出了基于 LLM 语义分组、嵌入相似度、嵌入聚类和超类标签的四种渐进式难度划分方案,发现无监督聚类划分在不依赖标注的情况下实现了与真值超类划分相当的去泄漏效果,同时保留了更好的预测性能。

**[Obliviator Reveals The Cost Of Nonlinear Guardedness In Concept Erasure](llm_efficiency/obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)**

:   提出Obliviator——一种基于RKHS中HSIC最小化的后处理概念擦除方法，通过两步迭代优化逐步变形特征空间，首次实现对非线性对抗者的完全防护，同时量化了非线性防护的效用-擦除代价（utility-erasure trade-off），在多个PLM和数据集上显著优于现有方法。

**[Omnidraft A Cross-Vocabulary Online Adaptive Drafter For On-Device Speculative D](llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)**

:   提出 OmniDraft 框架，通过在线 n-gram 缓存实现跨词表推测解码、混合蒸馏损失在线对齐草稿模型与目标模型、并结合自适应起草长度控制，使单个轻量 Llama-68M 模型可为 Vicuna-7B、Qwen2-7B、Llama3-8B 等不同目标模型提供推测解码加速（1.5-2x）。

**[On The Entropy Calibration Of Language Models](llm_efficiency/on_the_entropy_calibration_of_language_models.md)**

:   系统研究语言模型的熵校准问题（生成文本的熵是否匹配在人类文本上的 log loss），发现由于数据分布的幂律特性（$\alpha \approx 1$），误差积累随模型规模的改善极为缓慢（scaling exponent $\approx -0.05$），并从理论上证明了在多项式时间内可以在不牺牲多样性的前提下校准熵。

**[On The Expressive Power Of Mixture-Of-Experts For Structured Complex Tasks](llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)**

:   首次系统分析 MoE 在结构化复杂任务上的表达能力：证明浅层 MoE 可在低维流形上克服维度诅咒（近似速率由内在维度 $d$ 而非环境维度 $D$ 决定），深层 MoE 通过 $E$ 专家 × $L$ 层的分层组合可高效近似有 $E^L$ 段的分段函数，远超朴素上界 $LE$。

**[One Prompt Fits All Universal Graph Adaptation For Pretrained Models](llm_efficiency/one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)**

:   理论证明表示级图提示（representation-level prompt）本质等价于线性探针，据此提出 UniPrompt——基于可学习 kNN 拓扑提示图的输入级方法，通过 bootstrapping 策略融合提示图和原图，在同域和跨域 few-shot 节点分类中一致超越现有图提示学习方法。

**[Plasticity As The Mirror Of Empowerment](llm_efficiency/plasticity_as_the_mirror_of_empowerment.md)**

:   本文提出**广义有向信息（GDI）**作为度量智能体可塑性（plasticity）的信息论工具，揭示可塑性是赋权（empowerment）的"镜像"——两者使用相同度量、仅方向相反，并证明了两者之间存在严格的张力约束（tension bound）。

**[Silent Tokens Loud Effects Padding In Llms](llm_efficiency/silent_tokens_loud_effects_padding_in_llms.md)**

:   系统性研究了padding token在未被正确掩码时对LLM的影响，发现即使少量padding也会漂移隐层表示、降低生成质量、不可预测地改变偏见，而128个padding token可将Llama-3.1-8B的有害提示攻击成功率从8%飙升到77.5%，本质上实现了jailbreak。

**[Skyladder Better And Faster Pretraining Via Context Window Scheduling](llm_efficiency/skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)**

:   通过上下文窗口短到长的渐进式调度策略 SkyLadder，在固定计算量下实现更优的预训练效率（节省 22% 训练时间）和更好的模型性能（+3.7%），反驳了"长上下文=好性能"的业界信念。

**[Sparta Alignment Collectively Aligning Multiple Language Models Through Combat](llm_efficiency/sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)**

:   让多个LLM组成"斯巴达部落"互相竞技和互评，通过声誉加权的判断聚合生成偏好对，再用DPO迭代训练所有模型，在12个任务中的10个上超越Self-Rewarding等自对齐基线，平均提升7%。

**[Structure-Aware Spectral Sparsification Via Uniform Edge Sampling](llm_efficiency/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)**

:   本文证明在具有良好聚类结构的图上（结构比 Υ(k) 足够大），**均匀边采样**即可保留谱聚类所需的谱子空间结构，无需昂贵的有效电阻预计算——这是首个关于均匀采样保持结构的可证明保证。

**[Technical Debt In In-Context Learning Diminishing Efficiency In Long Context](llm_efficiency/technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)**

:   借鉴优化软件基准方法论，用性能比率精确量化ICL相对贝叶斯最优估计器的样本效率，发现存在"二分法"——少射下(≤15个演示)效率接近最优(仅多10%)而多射下(>40个演示)急剧恶化(多45%)，信息论分析证明这源于不可消除的非递减过剩风险，是ICL机制的内在限制。

**[Tensor Product Attention Is All You Need](llm_efficiency/tensor_product_attention_is_all_you_need.md)**

:   通过上下文张量分解将 Q/K/V 表示为低秩因子的加权和，将 KV 缓存压缩至原来的 1/10~1/16，同时在验证损失和下游任务精度上超越标准 MHA/MQA/GQA/MLA。

**[The Emergence Of Sparse Attention Impact Of Data Distribution And Benefits Of Re](llm_efficiency/the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)**

:   通过理论分析和受控实验研究 sparse attention 的涌现机制，揭示涌现时间遵循关于序列长度和维度的幂律关系 $T_\epsilon \propto \sqrt{d} \cdot T$，并发现 in-context 和 cross-sample 两种数据重复策略都能加速涌现，为理解 LLM 能力涌现提供了统一的 sparse attention 视角。

**[The Pokeagent Challenge Competitive And Long-Context Learning At Scale](llm_efficiency/the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)**

:   提出 PokéAgent Challenge，一个基于宝可梦对战和RPG速通的双赛道大规模AI基准，通过NeurIPS 2025竞赛验证了专家RL方法远超通用LLM方法，并揭示宝可梦对战衡量的能力与现有49个LLM基准近乎正交。

**[Tiled Flash Linear Attention More Efficient Linear Rnn And Xlstm Kernels](llm_efficiency/tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)**

:   提出 TFLA（Tiled Flash Linear Attention）算法，通过二层序列并行化和 tiling 优化，实现高效的线性 RNN/mLSTM 内核，相比 FlashAttention 3 和 Mamba 2 获得显著墙钟加速（训练 >2x vs Mamba 2），同时保持等价的模型精度。

**[Umoe Unifying Attention And Ffn With Shared Experts](llm_efficiency/umoe_unifying_attention_and_ffn_with_shared_experts.md)**

:   通过重新表述多头注意力机制，揭示其与 FFN 共有的"两层矩阵乘法"结构，据此提出 UMoE 统一架构——在注意力和 FFN 层使用相同设计的专家并支持参数共享，在 Base(134M) 和 Large(1.1B) 模型上均优于现有 FFN-MoE 和 Attention-MoE 基线。

**[Unmasking Covid-19 Vulnerability In Nigeria Mapping Risks Beyond Urban Hotspots](llm_efficiency/unmasking_covid-19_vulnerability_in_nigeria_mapping_risks_beyond_urban_hotspots.md)**

:   本文针对尼日利亚各州构建了一个综合 COVID-19 脆弱性风险评分体系,整合人口密度、贫困、医疗可及性和年龄风险四个维度,并通过 GIS 地图可视化热点区域,为公共卫生资源分配提供数据驱动的决策工具。

**[Vocabulary Customization For Efficient Domain-Specific Llm Deployment](llm_efficiency/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)**

:   提出一种保证编码效率单调不降的BPE tokenizer扩展算法，将领域高频token追加到Llama 3.1词表中（+30K token），在电商场景实现输入序列缩短20%、推理吞吐量提升20-30%，经10K步继续训练后模型质量不降，且约98%情况下模型主动生成新token。

**[Yggdrasil Bridging Dynamic Speculation And Static Runtime For Latency-Optimal Tr](llm_efficiency/yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)**

:   提出 Yggdrasil，一个延迟最优的推测解码系统，通过 Equal-Growth Tree (EGT) 结构实现编译友好的动态草稿、延迟感知优化目标替代传统 AAL 指标、以及阶段调度运行时减少 CPU-GPU 协调开销，在 A100/A40 上实现了最高 3.98× 的端到端加速。

**[Zeros Zero-Sum Linear Attention For Efficient Transformers](llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)**

:   通过移除 softmax 的零阶均匀项 $1/t$，构建零和权重的线性注意力机制 ZeroS，突破凸组合只能做加法混合的限制，支持单层内的差分/对比操作，在保持 $O(Nd^2)$ 线性复杂度的同时，在多个序列建模基准上匹配甚至超越标准 softmax 注意力。

---

## 🎵 音频/语音 { #audio_speech }

**[A Controllable Examination For Longcontext Language Models](audio_speech/a_controllable_examination_for_longcontext_language_models.md)**

:   提出LongBioBench，通过生成虚构传记作为可控的needle和haystack，构建满足"无缝上下文、可控设置、可靠评估"三大原则的长上下文LLM评估框架，测试18个模型后揭示当前LCLM在检索能力尚可的情况下推理和可信性仍有显著短板。

**[A Triangle Enables Multimodal Alignment Beyond Cosine Simila](audio_speech/a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)**

:   TRIANGLE提出用高维空间中三模态嵌入向量构成的三角形面积作为相似度度量，替代传统的成对余弦相似度，实现了视频-音频-文本三模态的联合对齐，在视频文本检索等任务上超越SOTA最多9个Recall@1点。

**[Accelerate Creation Of Product Claims Using Generative Ai](audio_speech/accelerate_creation_of_product_claims_using_generative_ai.md)**

:   开发 Claim Advisor 平台，利用 LLM 的 in-context learning 和 LoRA 微调加速消费品产品宣称的搜索、生成、优化和排序，通过模仿 MaxDiff 研究方法论让微调的 Phi-3 14B 模型在宣称排序上超越 GPT-4o（仅用 1 个示例 vs GPT 的 100 个示例），三轮迭代后 100% 的生成宣称达到"高吸引力"级别。

**[Adaptdel Adaptable Deletion Rate Randomized Smoothing For Ce](audio_speech/adaptdel_adaptable_deletion_rate_randomized_smoothing_for_ce.md)**

:   提出 AdaptDel 方法，将随机平滑中用于离散序列的固定删除率扩展为根据输入长度等属性自适应调整的可变删除率，在理论上证明了可变率下认证的 soundness，实验在 NLP 序列分类任务上实现认证区域基数最高 30 个数量级的提升。

**[Associative Syntax And Maximal Repetitions Reveal Context-Dependent Complexity I](audio_speech/associative_syntax_and_maximal_repetitions_reveal_context-dependent_complexity_i.md)**

:   本文提出一种无监督方法来推断果蝠发声的离散单元、语法类型和时序结构，并首次将最大重复子序列（Maximal Repetitions）引入动物通信领域，发现冲突行为中的通信复杂度显著高于合作行为。

**[Audsemthinker Enhancing Audio-Language Models Through Reasoning Over Semantics O](audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)**

:   AudSemThinker 为音频语言模型引入结构化语义推理框架——定义 9 类声音语义描述符（谁/什么/如何/何时/何地等），在 Qwen2.5-Omni-7B 上通过 SFT + GRPO（含可验证奖励和长度约束）训练产生 \<think\>\<semantic_elements\>\<answer\> 三阶段输出，MMAU 基准达 66.70%（超越 Audio-Reasoner 61.71% 和 Qwen2.5-Omni 65.60%）。

**[Benchmarking Egocentric Multimodal Goal Inference For Assist](audio_speech/benchmarking_egocentric_multimodal_goal_inference_for_assist.md)**

:   Meta 提出 WAGIBench，一个针对可穿戴辅助智能体的多模态目标推断基准，包含 348 名参与者的 3,477 条第一视角录制（29小时），涵盖视觉/音频/数字/纵向四种模态，人类准确率 93% vs 最佳 VLM 84%（MCQ），生成式评估中模型仅 55% 时间产生相关目标，揭示了当前 VLM 在实际可穿戴场景中的显著差距。

**[Bnmusic Blending Environmental Noises Into Personalized Music](audio_speech/bnmusic_blending_environmental_noises_into_personalized_music.md)**

:   提出 BNMusic，一个两阶段框架将环境噪声融合到个性化生成音乐中：第一阶段通过 mel-spectrogram 的 outpainting + inpainting 生成与噪声节奏对齐的音乐，第二阶段利用听觉掩蔽理论自适应放大音乐信号以降低噪声感知，无需额外训练，在 EPIC-SOUNDS 和 ESC-50 上显著优于 baseline。

**[Can Llms Outshine Conventional Recommenders A Comparative Evaluation](audio_speech/can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)**

:   提出 RecBench 综合评估框架，在5个领域数据集上系统对比17个LLM与10个传统DLRM，发现LLM推荐器在CTR任务上AUC提升最高5%、在序列推荐上NDCG@10提升最高170%，但推理速度慢10-1000倍，而传统DLRM结合LLM语义嵌入（LLM-for-RS）可以20倍更快的速度达到LLM约95%的性能，是当前最具工业可行性的方案。

**[Deepasa An Object-Oriented Multi-Purpose Network For Auditory Scene Analysis](audio_speech/deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)**

:   提出 DeepASA，一个面向对象的多任务统一架构，通过 object-oriented processing 和 chain-of-inference 机制在单一模型中同时完成多通道声源分离（MIMO）、去混响、声事件检测（SED）、音频分类和到达方向估计（DoAE），在多个空间音频基准上达到 SOTA。

**[E-Bats Efficient Backpropagation-Free Test-Time Adaptation For Speech Foundation](audio_speech/e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)**

:   提出首个面向语音基础模型的无反向传播测试时自适应框架 E-BATS，通过轻量级 prompt 自适应、多尺度损失函数和测试时 EMA 机制，在保持高精度的同时实现 2.0×–6.4× 的 GPU 显存节省。

**[E2E-Vguard Adversarial Prevention For Production Llm-Based End-To-End Speech Syn](audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)**

:   针对基于 LLM 的端到端语音合成中的声音克隆威胁，提出 E2E-VGuard 主动防御框架，通过编码器集成扰动音色、对抗样本干扰 ASR 发音识别、以及心理声学模型保证不可感知性，在 19 个 TTS 模型和 7 个 ASR 系统上验证了有效性。

**[Echoes Of Humanity Exploring The Perceived Humanness Of Ai Music](audio_speech/echoes_of_humanity_exploring_the_perceived_humanness_of_ai_music.md)**

:   通过随机对照交叉试验(RCCT)和混合方法内容分析，系统研究听众区分AI生成音乐(AIM)与人类创作音乐的能力，发现随机配对时听众无法区分（准确率≈随机猜测），但相似配对时显著提升至66%，且声音/技术/人声线索是成功区分的关键因素。

**[Efficient Speech Language Modeling Via Energy Distance In Continuous Latent Spac](audio_speech/efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)**

:   提出 SLED，将语音波形编码为连续潜在表示序列，在连续空间中通过 energy distance 目标进行自回归建模，避免了离散化信息损失和 RVQ 所需的复杂层级架构，同时实现高效的零样本与流式语音合成。

**[Ethics Statements In Ai Music Papers The Effective And The Ineffective](audio_speech/ethics_statements_in_ai_music_papers_the_effective_and_the_ineffective.md)**

:   对 AI 音乐领域论文中伦理声明（ethics statements）的使用现状进行系统审查，发现绝大多数伦理声明未被有效利用，并提出面向会议与研究者的改进建议。

**[Eurospeech A Multilingual Speech Corpus](audio_speech/eurospeech_a_multilingual_speech_corpus.md)**

:   提出可扩展的开源 pipeline，从 22 个欧洲议会录音中自动构建 EuroSpeech 数据集——61K 小时、覆盖 22 种语言的高质量语音-文本对齐数据，其中 19 种语言超 1K 小时，微调 Whisper 后平均 WER 降低 41.8%。

**[From Generation To Attribution Music Ai Agent Architectures For The Post-Streami](audio_speech/from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)**

:   提出了一种基于内容的 Music AI Agent 架构，通过将音乐分解为细粒度的 Block 组件并构建 Attribution Layer，将版权归因直接嵌入 AI 音乐创作流程中，为后流媒体时代建立公平的 AI 媒体平台。

**[Generating Physically Sound Designs From Text And A Set Of Physical Constraints](audio_speech/generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)**

:   提出 TIDES 框架，将预训练文本-图像模型（CLIP）的视觉引导与可微有限元物理仿真器结合，通过联合优化视觉相似度损失和结构合规性损失，从文本描述和物理约束出发生成既满足工程性能要求又具备文本指定特征的承载结构设计，并通过 3D 打印三点弯曲实验验证了方法的有效性。

**[Instance-Specific Test-Time Training For Speech Editing In The Wild](audio_speech/instance-specific_test-time_training_for_speech_editing_in_the_wild.md)**

:   提出面向野外语音编辑的实例特定测试时训练方法：在推理前利用未编辑区域的真实声学特征做直接监督、编辑区域通过时长约束和音素预测辅助损失做间接监督，对模型进行实例级自适应微调，有效缓解编辑边界的带宽不连续问题，并支持通过 mask 长度调整精确控制语速，在野外 benchmark 上主客观评估均超越现有系统。

**[Latent Space Factorization In Lora](audio_speech/latent_space_factorization_in_lora.md)**

:   提出 FVAE-LoRA，在 LoRA 框架中引入具有双潜空间的 VAE，通过新型 ELBO 目标将任务相关特征 ($\mathbf{z}_1$) 与残差信息 ($\mathbf{z}_2$) 显式分解，在文本、图像、音频任务上一致优于标准 LoRA。

**[Levo High-Quality Song Generation With Multi-Preference Alignment](audio_speech/levo_high-quality_song_generation_with_multi-preference_alignment.md)**

:   提出 LeVo 歌曲生成框架，通过语言模型并行建模混合 token 和双轨 token 以兼顾人声-伴奏和谐性和音质，并创新性地引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力。

**[Megadance Mixture-Of-Experts Architecture For Genre-Aware 3D Dance Generation](audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)**

:   提出 MEGADance，首个基于混合专家 (MoE) 架构的音乐驱动 3D 舞蹈生成方法，通过将编舞一致性解耦为"舞蹈通用性"（Universal Expert）和"风格特异性"（Specialized Expert），配合 FSQ 量化和 Mamba-Transformer 混合骨干网络，实现了 SOTA 的舞蹈质量和强风格可控性。

**[Mixed Monotonicity Reachability Analysis Of Neural Ode A Trade-Off Between Tight](audio_speech/mixed_monotonicity_reachability_analysis_of_neural_ode_a_trade-off_between_tight.md)**

:   将连续时间混合单调性技术应用于 Neural ODE 的可达性分析，通过将 Neural ODE 动力学嵌入混合单调系统，利用区间盒的几何简洁性实现高效过逼近，在紧致性（tightness）和计算效率之间提供可控的权衡。

**[Mome Mixture Of Matryoshka Experts For Audio-Visual Speech Recognition](audio_speech/mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)**

:   MoME将稀疏MoE集成到Matryoshka表示学习框架中，用于LLM-based音视频语音识别，通过共享路由器实现跨粒度知识迁移，在单一模型权重下支持多种压缩率的弹性推理，同时达到AVSR/ASR/VSR的SOTA性能。

**[Multi-Head Temporal Latent Attention](audio_speech/multi-head_temporal_latent_attention.md)**

:   MTLA 在 MLA 低秩潜在维度压缩基础上，用超网络动态融合时序相邻的 KV 向量，实现 KV 缓存在特征维度和时序维度的双重压缩，配合 stride-aware 因果 mask 保证训练-推理一致性，在语音翻译等任务上达到 4.29× 加速和 6.58× 内存降低，质量持平甚至略优于标准 MHA。

**[Perceptually Aligning Representations Of Music Via Noise-Augmented Autoencoders](audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)**

:   证明在自编码器训练中对潜变量加噪（noise-augmented latent training）配合感知损失，能使编码空间形成"感知层次结构"——感知最显著的音乐特征（如音高）编码在最粗粒度的潜在结构中，而次要特征（如音色细节）编码在细粒度结构中。这种对齐改善了潜在扩散解码下的音乐惊奇感估计和 EEG 脑响应预测。

**[Resounding Acoustic Fields With Reciprocity](audio_speech/resounding_acoustic_fields_with_reciprocity.md)**

:   利用声波传播的互易性原理，提出Versa方法（ELE数据增强+SSL自监督学习），通过交换发射器和接收器角色来生成物理有效的虚拟训练样本，在稀疏发射器配置下大幅提升声场估计性能。

**[Sand-Math Using Llms To Generate Novel Difficult And Useful Mathematics Question](audio_speech/sand-math_using_llms_to_generate_novel_difficult_and_useful_mathematics_question.md)**

:   提出 SAND-Math，一个无需种子数据集的全自动合成数学问题生成管线，通过 Difficulty Hiking 系统性提升题目难度，仅 500 道增强 LIMO 基线即可在 AIME25 上提升 4.39pp。

**[Seeing Sound Hearing Sight Uncovering Modality Bias And Conflict Of Ai Models In](audio_speech/seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)**

:   通过6种受控视听条件和人类心理物理实验，系统揭示现有AI声源定位模型存在严重视觉偏见（视听冲突时降至随机水平），并提出神经科学启发的EchoPin模型——HRTF滤波+ERB耳蜗图+立体声，在自建AudioCOCO数据集上大幅超越现有方法，且无需人类行为监督即涌现出类人的水平>垂直定位精度不对称性。

**[Sensorium Arc Ai Agent System For Oceanic Data Exploration And Interactive Eco-A](audio_speech/sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)**

:   本文构建了一个名为 Sensorium Arc 的多模态交互式 AI 智能体系统，通过将海洋拟人化为一个诗意的"讲述者"角色，利用多智能体 RAG 架构将 NASA 海洋科学数据与生态美学文本相结合，使用户能够以自然对话的方式探索复杂的海洋环境数据，同时在视听层面生成动态的科学可视化和艺术化反馈，实现从"被动数据观察"到"主动生态对话"的范式转变。

**[Simulmega Moe Routers Are Advanced Policy Makers For Simultaneous Speech Transla](audio_speech/simulmega_moe_routers_are_advanced_policy_makers_for_simultaneous_speech_transla.md)**

:   提出SimulMEGA框架，结合前缀训练与混合专家(MoE)精炼模块，实现无监督的读/写策略学习，使500M参数模型在6种语言的同时语音翻译中以1.5秒延迟仅损失<7% BLEU，并扩展到流式TTS。

**[Slimmable Nam Neural Amp Models With Adjustable Runtime Computational Cost](audio_speech/slimmable_nam_neural_amp_models_with_adjustable_runtime_computational_cost.md)**

:   将 Slimmable Networks 思想应用到 Neural Amp Modeler (NAM) 中，通过训练期间随机裁剪 WaveNet 层宽度，实现模型在推理时可以无额外训练代价地动态调整网络大小，使音乐家能实时平衡音质精度与计算成本。

**[Sound Logical Explanations For Mean Aggregation Graph Neural Networks](audio_speech/sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)**

:   针对使用均值聚合函数的 GNN（MAGNN，即非负权重的 mean-GNN），证明了能够作为其 sound 解释的单调逻辑规则的精确类别，并构造了一个一阶逻辑的受限片段来解释任意 MAGNN 预测，实验表明限制非负权重不显著影响性能且能有效提取 sound 规则。

**[Target Speaker Extraction Through Comparing Noisy Positive And Negative Audio En](audio_speech/target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)**

:   提出一种利用噪声正样本（目标说话人在说话的段落）和负样本（目标说话人沉默的段落）对比来编码目标说话人特征的新型注册策略，在单声道噪声注册目标说话人提取任务上取得 SOTA 性能，SI-SNRi 比此前最优方法高出 2.1 dB 以上。

**[Textttavrobustbench Benchmarking The Robustness Of Audio-Visual Recognition Mode](audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)**

:   提出 AVRobustBench，首个系统评估音视频模型在 **双模态共现关联腐蚀** 下测试时鲁棒性的基准，包含 4 个数据集 × 75 种腐蚀，并提出基于低熵样本筛选的 TTA 方法 AV2C。

**[The Impact Of Scaling Training Data On Adversarial Robustness](audio_speech/the_impact_of_scaling_training_data_on_adversarial_robustness.md)**

:   系统评估 36 个 SOTA 视觉模型在 6 类黑盒攻击下的鲁棒性，发现攻击成功率(ASR)随数据量和模型规模按对数律下降，但 **数据质量和模型规模比数据量本身更关键**。

**[Unifying Symbolic Music Arrangement Track-Aware Reconstruction And Structured To](audio_speech/unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)**

:   提出一个统一的符号音乐编排框架，通过段级自监督重建目标（解耦内容和乐器风格）和新的多轨token化方案REMI-z，使单个预训练模型能够处理乐队编排、钢琴缩编和鼓编排等多种编排任务，并在三个典型任务上超越了任务特定的SOTA。

**[Vita-15 Towards Gpt-4O Level Real-Time Vision And Speech Interaction](audio_speech/vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)**

:   VITA-1.5 提出了一套精心设计的三阶段渐进式训练策略，将视觉和语音能力逐步整合进 LLM 中，实现了无需独立 ASR/TTS 模块的端到端视觉-语音实时交互，在图像、视频和语音基准上均达到开源模型领先水平。

**[Wham Towards A Translative Model Of Sperm Whale Vocalization](audio_speech/wham_towards_a_translative_model_of_sperm_whale_vocalization.md)**

:   提出 WhAM（Whale Acoustics Model），首个基于 Transformer 的抹香鲸 coda 生成模型，通过微调 VampNet 实现声学翻译、合成生成与下游分类的三合一能力。

---

## 📚 预训练 { #llm_pretraining }

**[Ai Progress Should Be Measured By Capability-Per-Resource Not Scale Alone A Fram](llm_pretraining/ai_progress_should_be_measured_by_capability-per-resource_not_scale_alone_a_fram.md)**

:   本文以 position paper 的形式挑战"规模至上主义"，提出以**能力-每-资源（Capability-Per-Resource, CPR）**取代单纯的规模扩张来衡量 AI 进步，并给出一套基于梯度引导的资源分配理论框架——通过发布"梯度蓝图"元数据，使下游适配者仅微调高影响力参数子集即可在资源占用大幅降低的同时保持接近全参数微调的性能。

**[Alternating Gradient Flows A Theory Of Feature Learning In Two-Layer Neural Netw](llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)**

:   提出交替梯度流（AGF）理论框架解释神经网络的逐步"鞍到鞍"特征学习动力学——将训练建模为休眠神经元的效用最大化和活跃神经元的代价最小化的交替过程，统一了对角线性网络、注意力模型和模块加法的特征选择分析，预测与实际梯度流高度一致。

**[An Empirical Investigation Of Neural Odes And Symbolic Regression For Dynamical ](llm_pretraining/an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)**

:   系统实证研究 Neural ODE (NODE) 在动力系统中的外推能力和 Symbolic Regression (SR) 的方程恢复能力，发现 NODE 在动态相似条件下可外推到新边界条件，并提出 NODE→SR 流水线：仅用 10% 原始数据训练 NODE 生成增强数据，SR 即可恢复 2/3 的控制方程和 1/3 的良好近似。

**[Beyond Benign Overfitting In Nadaraya-Watson Interpolators](llm_pretraining/beyond_benign_overfitting_in_nadaraya-watson_interpolators.md)**

:   通过调节 Nadaraya-Watson 插值器中的单一带宽参数 $\beta$，精确刻画了从灾难性过拟合（$\beta < d$）→ 良性过拟合（$\beta = d$）→ 温和过拟合（$\beta > d$）的完整相变谱，证明高估数据内禀维度比低估更安全。

**[Breaking The Frozen Subspace Importance Sampling For Low-Rank Optimization In Ll](llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)**

:   发现GaLore等低秩优化方法的主导子空间在预训练中会"冻结"（相邻子空间重叠度趋近1），导致权重更新卡在固定低秩子空间中；提出SARA（重要性采样子空间选择），按奇异值权重随机采样奇异向量构建子空间，证明收敛性的同时将低秩优化器与全秩Adam的性能差距缩小最高46%。

**[Broken Tokens Your Language Model Can Secretly Handle Non-Canonical Tokenization](llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)**

:   揭示 LLM 能秘密处理非标准分词（如将"Hello"拆为"He"+"llo"而非标准的"Hello"整词token）——即使输入的 token 序列与训练时不同，模型表现出惊人的鲁棒性，且这种能力来自嵌入空间中子词嵌入的线性组合近似整词嵌入的特性。

**[Conformal Risk Training End-To-End Optimization Of Conformal Risk Control](llm_pretraining/conformal_risk_training_end-to-end_optimization_of_conformal_risk_control.md)**

:   本文将 Conformal Risk Control (CRC) 从期望损失扩展到一般化的 Optimized Certainty-Equivalent (OCE) 风险度量（包含 CVaR 等尾部风险），并提出"共形风险训练"方法，通过在训练中端到端地微分共形风险控制过程，在保持可证明风险保证的同时显著改善平均情况性能。

**[Differentiable Hierarchical Visual Tokenization](llm_pretraining/differentiable_hierarchical_visual_tokenization.md)**

:   提出一种端到端可微分的层次化视觉分词器，以像素级粒度自适应图像内容进行 token 划分，利用信息准则进行层次模型选择，可直接替换 ViT 的固定 patch 分词，并支持光栅-矢量转换。

**[Disaggregation Reveals Hidden Training Dynamics The Case Of Agreement Attraction](llm_pretraining/disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)**

:   本文通过对语言模型在主谓一致任务上的表现按实验条件进行细粒度拆解（disaggregation），揭示了聚合指标所掩盖的多阶段训练动态：模型先学词频偏好、再学局部上下文、最后发展出一般性的语法规则，这一过程涉及多次"隐藏突破"而非简单的单调提升。

**[Does Object Binding Naturally Emerge In Large Pretrained Vision Transformers](llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)**

:   通过定义 IsSameObject 谓词并设计二次探针，证明大规模预训练 ViT（尤其是 DINO、CLIP）自然涌现了目标绑定能力，该信号编码在低维子空间中并主动引导注意力机制，挑战了认知科学界认为 ViT 缺乏绑定能力的观点。

**[Efficient Pre-Training Of Llms Via Topology-Aware Communication Alignment On Mor](llm_pretraining/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)**

:   提出 Arnold 调度系统，通过将 LLM 训练的通信模式（DP/PP group）与数据中心物理网络拓扑对齐，在模拟中将通信组最大跨度减少 1.67x，在 9600+ GPU 生产级训练中端到端性能提升 10.6%。

**[Enhancing Training Data Attribution With Representational Optimization](llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)**

:   提出 AirRep（Attentive Influence Ranking Representation），一种基于表示学习的训练数据归因方法，通过可训练编码器和注意力池化机制，在推理效率比梯度方法快约 80 倍的同时，达到甚至超越 SOTA 梯度方法的归因精度。

**[Final-Model-Only Data Attribution With A Unifying View Of Gradient-Based Methods](llm_pretraining/final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)**

:   明确提出"仅有最终模型"(FiMO)的训练数据归因设定，将问题从"贡献度"重构为"敏感性"度量，提出 further training 作为金标准，并统一推导出多种梯度方法（Grad-Dot、影响函数、TRAK、DataInf 等）均为 further training 的不同阶近似。

**[Flatness Is Necessary Neural Collapse Is Not Rethinking Generalization Via Grokk](llm_pretraining/flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)**

:   利用 grokking（延迟泛化）作为因果探针，证明 **relative flatness 是泛化的（潜在）必要条件**，而 neural collapse 虽常伴随泛化出现，但并非必要——它只是通往 flatness 的一条路径。

**[Gemstones A Model Suite For Multi-Faceted Scaling Laws](llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)**

:   开源由超过4000个检查点（覆盖50M-2B参数、多种宽度-深度比）组成的Gemstones模型套件，通过系统实验揭示缩放律对模型选择、学习率调度、冷却策略等设计选择高度敏感，并提出基于凸包的新拟合方法提升稀疏采样下的缩放律稳定性。

**[Generalization Bounds For Rank-Sparse Neural Networks](llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)**

:   本文证明了利用神经网络权重矩阵近似低秩结构的泛化界，当 Schatten $p$ 拟范数较小时，样本复杂度仅为 $\widetilde{O}(WrL^2)$，其中 $W$, $L$, $r$ 分别为宽度、深度和权重矩阵的秩。

**[Global Minimizers Of Sigmoid Contrastive Loss](llm_pretraining/global_minimizers_of_sigmoid_contrastive_loss.md)**

:   首次在实践相关的 N≫d 区间严格刻画了 Sigmoid 对比损失（SigLIP）在可训练温度和偏置下的全局最小值几何结构，提出了 (m, b_rel)-Constellation 这一新型组合对象，并用其解释了 SigLIP 的检索成功、模态间隙现象，以及提出了显式 relative bias 参数化改进训练动态。

**[Gradient-Weight Alignment As A Train-Time Proxy For Generalization In Classifica](llm_pretraining/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)**

:   提出 Gradient-Weight Alignment (GWA)，通过量化每个训练样本梯度与模型权重的方向一致性（cosine similarity），在训练过程中无需验证集即可准确预测泛化性能、确定最佳早停时机，并定位有影响力的训练样本。

**[How Does Sequence Modeling Architecture Influence Base Capabilities Of Pre-Train](llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)**

:   通过"限定领域预训练 + OOD 测试"的评估框架揭示 Mamba/RWKV 等 stateful 架构存在基础能力退化，并归纳出关键设计原则——"全序列任意选择能力"（full-sequence visibility + real relation calculation + non-uniform distribution），用极简的 Top-1 Element/Chunk Selection 架构验证该原则可恢复至接近 Transformer 的基础能力。

**[Language Model Behavioral Phases Are Consistent Across Archi](llm_pretraining/language_model_behavioral_phases_are_consistent_across_archi.md)**

:   通过对超过 1,400 个语言模型检查点（涵盖 Transformer/Mamba/RWKV 三种架构、14M–12B 参数规模、两种训练数据集）在 11 万+ token 上的系统分析，发现所有自回归语言模型在预训练过程中展现出高度一致的行为阶段——预测概率依次过拟合到递增阶数的 n-gram 概率，且词频、n-gram 概率和语义相似度三个简单启发式可解释高达 98% 的行为方差。

**[Learning The Wrong Lessons Syntactic-Domain Spurious Correlations In Language Mo](llm_pretraining/learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)**

:   揭示 LLM 学会了句法模板（PoS n-gram）与领域之间的虚假关联，导致跨域性能骤降，甚至可利用此关联绕过安全拒绝机制（refusal bypass），在 OLMo-2 上将拒绝率从 40% 降至 2.5%。

**[Learning To Flow From Generative Pretext Tasks For Neural Architecture Encoding](llm_pretraining/learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)**

:   提出 FGP（Flow-based Generative Pre-training），通过让编码器重建"流代理"（flow surrogate）这一架构信息流的简化表征，使任意结构的编码器无需专用的异步消息传递设计即可捕获信息流，在性能预测中 Precision@1% 最高提升 106%。

**[Leveraging Importance Sampling To Detach Alignment Modules From Large Language M](llm_pretraining/leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)**

:   提出 Residual Alignment Model (RAM)，将 LLM 对齐过程形式化为重要性采样，将大模型分解为冻结的 Proposal Module 和可训练的小型 Residual Aligner，以不到 1/8 参数实现可比甚至超越全参数 SFT/DPO 的对齐效果，同时解决了首 token 延迟问题。

**[Memory Mosaics At Scale](llm_pretraining/memory_mosaics_at_scale.md)**

:   Memory Mosaics v2 将关联存储网络扩展至 10B 参数、1T token 训练规模，在新任务学习和上下文学习上显著超越同规模甚至 8T token 训练的 Transformer。

**[Nemotron-Climb Clustering-Based Iterative Data Mixture Bootstrapping For Languag](llm_pretraining/nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)**

:   NVIDIA 提出 CLIMB 框架，通过嵌入聚类 + 迭代自举搜索自动发现最优预训练数据混合比例，在 1B 模型上超过 Llama-3.2-1B 达 2.0%，并发布了 1.2T token 的 ClimbLab 语料库和 400B token 的 ClimbMix 高质量数据集。

**[Neural Collapse Under Gradient Flow On Shallow Relu Networks For Orthogonally Se](llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)**

:   首次证明在正交可分数据上，两层ReLU网络的梯度流（GF）在小初始化下可证收敛到Neural Collapse（NC）解，揭示了GF隐式偏置（早期神经元对齐+渐近最大间隔偏置）在促进NC出现中的关键作用。

**[Optimal Online Change Detection Via Random Fourier Features](llm_pretraining/optimal_online_change_detection_via_random_fourier_features.md)**

:   提出 Online RFF-MMD 算法，通过随机 Fourier 特征近似 MMD 统计量并嵌入到二进制网格的序贯检验框架中，实现了无需训练数据、无需窗口参数的在线非参数变点检测，运行时间和空间复杂度均为对数级，并证明了检测延迟的 minimax 最优性。

**[Power Lines Scaling Laws For Weight Decay And Batch Size In Llm Pre-Training](llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)**

:   提出了一套针对 LLM 预训练中权重衰减 $\lambda$ 和批大小 $B$ 的幂律缩放定律（power laws），通过 AdamW 时间尺度 $\tau$ 的概念统一了超参数缩放关系，使得在大规模训练前即可准确预测最优超参数。

**[Predict Training Data Quality Via Its Geometry In Metric Space](llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)**

:   提出基于持久同调（Persistent Homology）的训练数据多样性度量方法，证明数据的几何/拓扑结构特征能够有效预测模型性能，优于传统基于熵的Vendi Score等指标。

**[Prescribe Predicting Single-Cell Responses With Bayesian Estimation](llm_pretraining/prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)**

:   提出 PRESCRIBE 框架，通过多变量深度证据回归联合建模单细胞扰动预测中的认知不确定性（模型对输入的不熟悉程度）和随机不确定性（生物系统固有的随机性），生成伪 E-distance 作为统一的不确定性代理指标，过滤不可靠预测后准确率提升 3% 以上。

**[Quantifying Task-Relevant Representational Similarity Using Decision Variable Co](llm_pretraining/quantifying_task-relevant_representational_similarity_using_decision_variable_co.md)**

:   本文提出基于决策变量相关（DVC）的新方法来衡量两个神经表征在分类任务上的逐试次一致性，发现深度网络在 ImageNet 上准确率越高反而与猴脑 V4/IT 的 DVC 越低，对抗训练和大规模数据集预训练也无法缩小这一差距。

**[Retrospective Incontext Learning For Temporal Credit Assignm](llm_pretraining/retrospective_incontext_learning_for_temporal_credit_assignm.md)**

:   提出 RICL（Retrospective In-Context Learning），利用 LLM 的预训练知识通过回顾式上下文学习将稀疏环境反馈转化为密集优势函数信号，实现比传统 Monte Carlo 方法高 100 倍的样本效率，并在此基础上构建 RICOL 在线学习框架。

**[Scalable Fingerprinting Of Large Language Models](llm_pretraining/scalable_fingerprinting_of_large_language_models.md)**

:   提出 Perinucleus 采样方法生成可扩展的 LLM 指纹，能在 Llama-3.1-8B 上嵌入 24,576 个指纹（比现有方法多两个数量级）且不损害模型能力，并通过理论和实验证明大规模指纹是抵御共谋攻击的关键。

**[Scaling Embedding Layers In Language Models](llm_pretraining/scaling_embedding_layers_in_language_models.md)**

:   提出Scone方法，通过为高频n-gram学习上下文化的嵌入（用独立Transformer模型训练），在推理时将这些嵌入卸载到主存/SSD，实现"训练时用更多计算但推理时不增加加速器资源"的新缩放范式，1B参数模型超越1.9B基线。

**[Superposition Yields Robust Neural Scaling](llm_pretraining/superposition_yields_robust_neural_scaling.md)**

:   揭示表示叠加（superposition）是神经缩放定律的核心驱动力：在强叠加区间，损失**通用地**与模型维度成反比（$L \propto 1/m$），且该行为与数据频率分布的具体形式无关，这与实际 LLM 的缩放行为一致。

**[The Curse Of Depth In Large Language Models](llm_pretraining/the_curse_of_depth_in_large_language_models.md)**

:   揭示 Pre-LN Transformer 中输出方差指数增长导致深层退化为恒等映射的根本原因，提出无参数的 LayerNorm Scaling（LNS）策略——仅在 LayerNorm 后乘以 $1/\sqrt{\ell}$，将方差从指数增长压缩为多项式增长，在 130M-7B 全规模上稳定改进困惑度 5-8%。

**[Through The River Understanding The Benefit Of Schedule-Free Methods For Languag](llm_pretraining/through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)**

:   从 river-valley 损失景观的几何视角，分析了 Schedule-Free (SF) 优化器在语言模型预训练中不需要学习率衰减和权重平均就能持续追踪最优解的原因，并揭示 SF 隐式执行了权重平均，进而提出解耦动量和平均窗口的改进版 SF-AdamW。

**[Understanding And Enhancing Mask-Based Pretraining Towards Universal Representat](llm_pretraining/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)**

:   用高维线性回归理论精确刻画了 mask-based pretraining 中掩码率对测试风险的影响（偏差-方差分解），揭示了最优掩码率依赖于任务和模型大小，并据此提出 R2MAE（随机随机掩码），在视觉、语言、DNA、单细胞模型上一致超越固定掩码率。

**[Zeus Zero-Shot Embeddings For Unsupervised Separation Of Tabular Data](llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)**

:   ZEUS 是首个面向表格数据的零样本聚类方法，通过在合成数据集上预训练一个 Transformer 编码器来学习可泛化的表示，使得新数据集无需任何额外训练或调参即可在单次前向传播中完成高质量聚类。

---

## 🎯 目标检测 { #object_detection }

**[All You Need Is One Capsule Prompt Tuning With A Single Vector](object_detection/all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)**

:   提出 Capsule Prompt-Tuning (CaPT)，发现现有 task-aware soft prompts 实际上与输入 tokens 缺乏交互（"attention 孤岛"），而将 instance-aware 信息融入单个 capsule prompt 可以作为"attention anchor"激活对关键结构信息的注意力，以极低参数量（如 Llama3.2-1B 上仅 0.003% 参数）实现超越多 prompt 方法的性能。

**[Angular Constraint Embedding Via Spherepair Loss For Constrained Clustering](object_detection/angular_constraint_embedding_via_spherepair_loss_for_constrained_clustering.md)**

:   本文提出SpherePair损失函数，通过在角度空间（而非欧几里得空间）进行成对约束嵌入学习，实现了不依赖锚点(anchor)、不需要预知聚类数的深度约束聚类方法，并提供了严格的理论保证来确定最优超参数。

**[Any Large Language Model Can Be A Reliable Judge Debiasing W](object_detection/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)**

:   提出 Reasoning-based Bias Detector（RBD）作为 LLM 评判器的即插即用去偏模块——通过外部检测 4 种评估偏见（冗长/位置/从众/情感），生成带推理链的结构化反馈引导评判器自我纠正，RBD-8B 在 8 个 LLM 评判器上平均提升准确率 18.5%、一致性 10.9%。

**[Ascent Fails To Forget](object_detection/ascent_fails_to_forget.md)**

:   本文从遗忘集与保留集之间的统计依赖出发，理论结合实验证明广泛使用的梯度上升/Descent-Ascent（DA）类机器遗忘方法在存在数据相关性时会系统性失败——在 logistic 回归中 DA 解甚至会比原始模型更远离 oracle，且在非凸设置下会将模型困在劣质局部最小值中。

**[Automated Detection Of Visual Attribute Reliance With A Self-Reflective Agent](object_detection/automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)**

:   提出一个自反思 agent 框架，通过迭代的假设生成-测试-验证-反思循环来自动检测视觉模型中的属性依赖（如 CLIP 识别 teacher 依赖教室背景、YOLOv8 检测行人依赖人行横道），在 130 个注入已知属性依赖的模型 benchmark 上显示自反思显著提升检测准确性。

**[Burstdeflicker A Benchmark Dataset For Flicker Removal In Dynamic Scenes](object_detection/burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)**

:   提出首个面向多帧闪烁去除（MFFR）的大规模 benchmark 数据集 BurstDeflicker，包含基于 Retinex 的合成数据、真实静态数据和绿幕动态数据三个互补子集，系统解决了动态场景下闪烁-干净图像对难以获取的核心瓶颈。

**[Cq-Dino Mitigating Gradient Dilution Via Category Queries For Vast Vocabulary Ob](object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)**

:   针对大规模类别（>10K）目标检测中分类头的正梯度稀释和难负样本梯度稀释问题，提出 CQ-DINO：用可学习类别查询替代分类头，通过图像引导的 Top-K 类别选择将负空间缩小 100 倍，在 V3Det（13204 类）上超越前 SOTA 2.1% AP，同时保持 COCO 竞争力。

**[Delving Into Cascaded Instability A Lipschitz Continuity View On Image Restorati](object_detection/delving_into_cascaded_instability_a_lipschitz_continuity_view_on_image_restorati.md)**

:   提出首个面向动态场景的多帧去闪烁（MFFR）基准数据集 BurstDeflicker，通过 Retinex 合成、真实静态采集与绿幕合成三种互补策略构建大规模训练/测试数据，显著提升闪烁去除模型在真实动态场景中的泛化能力。

**[Detectiumfire A Comprehensive Multi-Modal Dataset Bridging Vision And Language F](object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)**

:   DetectiumFire 构建了最大的多模态火灾理解数据集——14.5K 真实图像 + 2.5K 视频 + 8K 合成图像 + 12K RLHF 偏好对，低重复率（0.03 PHash vs D-Fire 0.15），配合 4 级严重性分类标准和详细场景描述，微调 YOLOv11m 达 mAP 43.74，微调 LLaMA-3.2-11B 火灾严重性分类 83.84%。

**[Detree Detecting Human-Ai Collaborative Texts Via Tree-Structured Hierarchical R](object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)**

:   提出 DETree 框架，通过构建层次亲和树（HAT）建模不同人机协作文本生成过程之间的层次关系，并设计树结构对比损失（TSCL）对齐表示空间，在混合文本检测和 OOD 场景下取得了显著优势。

**[Diffusion-Classifier Synergy Reward-Aligned Learning Via Mutual Boosting Loop Fo](object_detection/diffusion-classifier_synergy_reward-aligned_learning_via_mutual_boosting_loop_fo.md)**

:   提出 Diffusion-Classifier Synergy (DCS) 框架，通过在扩散模型和分类器之间建立互相增强的闭环，利用多层次奖励函数（特征级+logits级）引导扩散模型生成对分类器最有益的图像，在 FSCIL 基准上取得 SOTA。

**[Dithub A Modular Framework For Incremental Openvocabulary Ob](object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)**

:   DitHub 将开放词汇目标检测的增量适配问题重新构造为"版本控制"问题——为每个类别训练独立的 LoRA 专家模块，通过 branch（分支）、fetch（检索）、merge（合并）三个原语管理不断扩展的模块库，在 ODinW-13 全量数据上以 62.19 mAP 超越 ZiRa 4.21 个点，同时保持 47.01 的零样本 COCO 性能。

**[Dual Data Alignment Makes Ai-Generated Image Detector Easier Generalizable](object_detection/dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)**

:   提出 Dual Data Alignment (DDA)，通过像素域和频域双重对齐生成训练用合成图像，消除数据集偏置导致的虚假相关性，使检测器仅学习伪造相关特征，在11个基准上平均准确率达到90.7%，大幅超越现有方法。

**[Dynamic Features Adaptation In Networking Toward Flexible Training And Explainab](object_detection/dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)**

:   提出 DAFI（Drift-Aware Feature Importance）算法，利用分布漂移检测动态切换 SHAP/MDI 两种特征重要性方法，结合自适应随机森林（ARF）实现通信网络场景下特征动态增加时的灵活训练与高效可解释推理。

**[Flexevent Towards Flexible Event-Frame Object Detection At Varying Operational F](object_detection/flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)**

:   提出 FlexEvent 框架，通过自适应事件-图像融合模块 FlexFuse 和频率自适应微调机制 FlexTune，实现事件相机在不同操作频率下的灵活目标检测，在 20Hz 到 180Hz 范围内保持鲁棒性能，显著超越现有方法。

**[Generalizable Insights For Graph Transformers In Theory And Practice](object_detection/generalizable_insights_for_graph_transformers_in_theory_and_practice.md)**

:   提出 Generalized-Distance Transformer (GDT)，一种基于标准注意力（无需修改注意力机制）的图 Transformer 架构，理论证明其表达力等价于 GD-WL 算法，并通过覆盖 800 万图/2.7 亿 token 的大规模实验首次建立了 PE 表达力的细粒度经验层次，在 few-shot 迁移设置下无需微调即可超越 SOTA。

**[Instanceassemble Layoutaware Image Generation Via Instance A](object_detection/instanceassemble_layoutaware_image_generation_via_instance_a.md)**

:   提出 InstanceAssemble，在 DiT-based T2I 模型（SD3 和 Flux）的 Transformer 块中注入"实例组装注意力"机制，通过将每个 bounding box 区域的 image token 独立与对应的 layout hidden state 做 cross-attention 来实现精确的实例级空间控制，同时以 LoRA 轻量适配方式保持与现有风格 LoRA 的兼容性，并提出包含 5K 图像/90K 实例的 DenseLayout 基准和多维度的 Layout Grounding Score（LGS）评估指标。

**[M-Grpo Stabilizing Self-Supervised Reinforcement Learning For Large Language Mod](object_detection/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)**

:   针对自监督强化学习中 LLM 策略崩溃和熵崩溃问题，提出动量锚定的 GRPO（M-GRPO）框架和基于 IQR 的低熵轨迹过滤方法，实现稳定训练和 SOTA 性能。

**[Mstar Box-Free Multi-Query Scene Text Retrieval With Attention Recycling](object_detection/mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)**

:   提出 MSTAR，首个无需边界框标注的多查询场景文本检索方法，通过渐进式视觉嵌入（PVE）逐步将注意力从显著区域转移到不显著区域，结合风格感知指令和多实例匹配模块，实现了对单词、短语、组合和语义四种查询类型的统一检索，并构建了首个多查询文本检索基准 MQTR。

**[Overlaybench A Benchmark For Layout-To-Image Generation With Dense Overlaps](object_detection/overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)**

:   OverLayBench 构建了首个聚焦密集重叠场景的 Layout-to-Image 基准（4052 样本 + OverLayScore 难度指标），揭示 SOTA 方法在复杂重叠下 mIoU 从 71%→54% 急剧退化，提出 Amodal Mask 监督在重叠 IoU 上提升 15.9%。

**[Pandapose 3D Human Pose Lifting From A Single Image Via Propagating 2D Pose Prio](object_detection/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)**

:   提出 PandaPose，通过将 2D 姿态先验传播到 3D 锚点空间作为统一中间表示，结合自适应关节级 3D 锚点设置和关节级深度分布估计，实现对遮挡和 2D 姿态误差鲁棒的单帧 3D 人体姿态提升。

**[Preference Learning With Lie Detectors Can Induce Honesty Or Evasion](object_detection/preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)**

:   系统研究了将谎言检测器（lie detector）整合到LLM偏好学习标注流程中的效果（SOLiD框架），发现训练后模型是变得诚实还是学会规避检测取决于三个关键因素：探索程度（GRPO vs DPO）、检测器准确率（TPR）和KL正则化强度。

**[Recon-Gs Continuum-Preserved Gaussian Streaming For Fast And Compact Reconstruct](object_detection/recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct.md)**

:   提出 ReCon-GS，通过连续性保持的 Gaussian 流式处理实现增量式 3D 重建，在保持渲染质量的同时大幅减少存储需求和训练时间，支持大规模场景的实时重建。

**[Recon Region-Controllable Data Augmentation With Rectification And Alignment For](object_detection/recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)**

:   ReCon 提出无需额外训练的区域可控数据增强框架，通过区域引导校正（RGR）和区域对齐交叉注意力（RACA）增强现有结构可控生成模型的目标检测数据质量，在 COCO 上实现 35.5 mAP（超过需 fine-tune 的 GeoDiffusion）。

**[Rectified-Cfg For Flow Based Models](object_detection/rectified-cfg_for_flow_based_models.md)**

:   针对Rectified Flow模型中标准CFG导致的离流形漂移问题，提出Rectified-CFG++——一种自适应预测-校正引导策略，通过条件流预测+时间调度插值校正替代外推式引导，在Flux/SD3/SD3.5/Lumina等大规模模型上全面超越标准CFG。

**[Robust Hallucination Detection In Llms Via Adaptive Token Selection](object_detection/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)**

:   HaMI 将幻觉检测建模为多示例学习（MIL）问题，将生成序列视为 token 实例的"bag"，通过联合优化 token 选择和幻觉检测来自适应地定位最具指示性的 token，在四个 QA 基准上以 AUROC 大幅超越所有现有方法（最高提升 11.9%）。

**[Safe Multitask Failure Detection For Vision-Language-Action Models](object_detection/safe_multitask_failure_detection_for_vision-language-action_models.md)**

:   SAFE 发现 VLA 模型的内部特征空间存在跨任务一致的"失败区域"，据此训练轻量 MLP/LSTM 失败检测器，配合功能保形预测（FCP）做阈值校准，在未见任务上达 78% ROC-AUC，计算开销 <1%，大幅优于 token 不确定性和一致性检测方法。

**[Sample Complexity Of Distributionally Robust Average-Reward Reinforcement Learni](object_detection/sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)**

:   首次为分布鲁棒平均奖励强化学习（DR-AMDP）建立了有限样本收敛保证，提出两种算法（折扣归约法和锚定法），在KL和$f_k$-散度不确定集下均达到$\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$的近最优样本复杂度。

**[Segment-Factorized Full-Song Generation On Symbolic Piano Music](object_detection/segment-factorized_full-song_generation_on_symbolic_piano_music.md)**

:   提出Segmented Full-Song模型（SFS），将歌曲分解为片段，通过选择性注意结构相关上下文自回归生成各片段，实现比现有方法更快速、更结构化的钢琴全曲生成，并支持交互式人机共创。

**[Sstag Structure-Aware Self-Supervised Learning Method For Text-Attributed Graphs](object_detection/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)**

:   提出 SSTAG，通过双重知识蒸馏将 LLM 和 GNN 的互补知识联合蒸馏到结构感知的 MLP 中，结合内存库机制存储原型表示，实现高效、可扩展的文本属性图跨域自监督预训练。

**[Test-Time Adaptive Object Detection With Foundation Model](object_detection/test-time_adaptive_object_detection_with_foundation_model.md)**

:   提出无需源域数据的开放词汇测试时自适应目标检测框架（TTAOD），通过多模态 Prompt Tuning + Mean-Teacher + 实例动态记忆（IDM）+ 记忆增强/幻觉策略，在 Pascal-C 上 AP50 达 56.2%（+11.0 vs SOTA），在 13 个跨域数据集上一致有效。

**[The Complexity Of Finding Local Optima In Contrastive Learning](object_detection/the_complexity_of_finding_local_optima_in_contrastive_learning.md)**

:   证明对比学习中寻找局部最优是计算困难的：离散三元组最大化问题是 PLS-hard（即使 $d=1$），连续三元组损失最小化是 CLS-hard，意味着（在标准假设下）不存在多项式时间算法找到局部最优。

**[The Path Not Taken Rlvr Provably Learns Off The Principals](object_detection/the_path_not_taken_rlvr_provably_learns_off_the_principals.md)**

:   本文提出三门理论 (Three-Gate Theory) 解释 RLVR 的参数更新稀疏性假象，证明 RLVR 在权重空间的非主方向 (off-principal) 上学习，与 SFT 的优化机制本质不同，因此直接移植 SFT 时代的 PEFT 方法到 RLVR 是有缺陷的。

**[Think Straight Stop Smart Structured Reasoning For Efficient Multi-Hop Rag](object_detection/think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)**

:   提出 TSSS (Think Straight, Stop Smart) 框架，通过 (i) 基于模板的推理缓存重复前缀并锚定子查询到主问题，(ii) 基于检索器的确定性终止器在子查询重复时停止推理，在多跳 RAG 基准上实现 SOTA 准确率和竞争效率。

**[Towards Effective Federated Graph Foundation Model Via Mitigating Knowledge Enta](object_detection/towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)**

:   首次提出联邦图基础模型(FedGFM)范式，融合联邦图学习的分布式协作能力与图基础模型的跨域泛化能力，通过 AncDAI（锚点域感知初始化）和 AdaDPP（自适应域敏感提示池）两个模块缓解知识纠缠问题，在8个跨任务跨领域数据集上超越20个基线。

**[Video-Rag Visually-Aligned Retrieval-Augmented Long Video Comprehension](object_detection/video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)**

:   本文提出Video-RAG，一个免训练、即插即用的RAG管道，通过从视频中提取视觉对齐的辅助文本（OCR、ASR、目标检测）并经检索筛选后输入LVLM，在仅增加约2K token的条件下将7个开源LVLM的Video-MME平均性能提升2.8%，72B模型超越GPT-4o。

**[Xifbench Evaluating Large Language Models On Multilingual Instruction Following](object_detection/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)**

:   提出XIFBench——首个系统评估LLM多语言指令遵循能力的约束驱动基准，包含558条指令（0-5个约束，5大类21维度）×6种语言（高/中/低资源），并引入英语需求锚定评估协议，实现94.7%的跨语言评估一致性。

**[You Can Trust Your Clustering Model A Parameter-Free Self-Boosting Plug-In For D](object_detection/you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)**

:   提出 DCBoost，一个无需额外超参数的即插即用模块，通过自适应 k-NN 筛选高置信样本并利用可靠的局部结构信息引导全局特征空间优化，显著提升现有深度聚类模型的性能。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Attributing Response To Context A Jensen-Shannon Divergence Driven Mechanistic S](information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)**

:   ARC-JSD 提出基于 Jensen-Shannon 散度的 RAG 上下文归因方法——通过比较有/无特定上下文句子时模型输出分布的 JSD 差异，无需微调/梯度计算即可定位回答所依赖的上下文，计算效率比 baseline 快 3 倍，Top-1 归因准确率平均提升 10.7%，并通过 Logit Lens 揭示归因相关的注意力头集中在高层。

**[Benchmarking Retrievalaugmented Multimodal Generation For Do](information_retrieval/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)**

:   提出 MMDocRAG 基准（4055 个专家标注的 QA 对），系统评估了 60 个 VLM/LLM 和 14 个检索器在多模态文档检索增强生成中的引用选择和交错图文回答能力，揭示当前最强模型 GPT-4.1 的 Quote Selection F1 仅 70.2%，微调可显著提升性能。

**[Chain-Of-Retrieval Augmented Generation](information_retrieval/chain-of-retrieval_augmented_generation.md)**

:   提出 CoRAG 框架，通过拒绝采样自动生成中间检索链（子查询→子答案），微调 LLM 学习迭代检索和推理，并支持多种测试时解码策略（贪心 / Best-of-N / 树搜索）灵活扩展计算量，在多跳 QA 上 EM 提升 26+ 点，KILT 基准 9/10 任务达到 SOTA。

**[Compress Gather And Recompute Reforming Long-Context Processing In Transformers](information_retrieval/compress_gather_and_recompute_reforming_long-context_processing_in_transformers.md)**

:   提出 REFORM 推理框架，通过"压缩—检索—重算"三阶段流水线高效处理超长上下文（百万级 token），在 RULER 和 BABILong 上相比最强基线分别提升 52% 和 34%，同时降低 30% 推理时间和 5% 峰值显存。

**[Cooperative Retrieval-Augmented Generation For Question Answering Mutual Informa](information_retrieval/cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)**

:   提出CoopRAG框架，通过问题展开、基于检索器层对比的重排、以及推理链补全，实现检索器与LLM的双向合作，在多跳QA上超越HippoRAG2 5.3%，单跳QA上提升35.2%。

**[Deep Research Brings Deeper Harm](information_retrieval/deep_research_brings_deeper_harm.md)**

:   揭示 Deep Research (DR) 智能体的严重安全隐患——即使底层 LLM 能正确拒绝有害请求，部署为 DR 智能体后仍能生成详细专业的危险报告；提出 Plan Injection 和 Intent Hijack 两种针对性越狱方法，以及 DeepREJECT 评估指标，在 6 个 LLM 上验证了 DR 智能体系统性地削弱了对齐机制。

**[Dice Discrete Interpretable Comparative Evaluation With Probabilistic Scoring Fo](information_retrieval/dice_discrete_interpretable_comparative_evaluation_with_probabilistic_scoring_fo.md)**

:   提出 DICE 框架，通过两阶段评估（证据耦合深度分析 + 概率化 {A,B,Tie} 打分）和瑞士赛制锦标赛实现 RAG 系统的可解释、鲁棒、高效评估，在中文金融 QA 数据集上达到 85.7% 人类专家一致率，远超 RAGAS（45.7%）。

**[Generalized Contrastive Learning For Universal Multimodal Re](information_retrieval/generalized_contrastive_learning_for_universal_multimodal_re.md)**

:   提出 Generalized Contrastive Learning (GCL)——在 mini-batch 内对所有 6 种模态对组合（image↔text, image↔image+text, text↔image+text）执行对比学习，无需构建新的三元组数据集，仅用现有图文对即可在 M-BEIR 上将 VISTA 的平均检索精度从 21.18 提升到 34.06（+60.8%），在 MMEB 的 text→image+text 任务上从 10.1% 提升到 31.1%。

**[Hierarchical Retrieval The Geometry And A Pretrain-Finetune Recipe](information_retrieval/hierarchical_retrieval_the_geometry_and_a_pretrain-finetune_recipe.md)**

:   研究双编码器（Dual Encoder）在层次化检索（Hierarchical Retrieval）中的可行性，理论证明嵌入维度只需与层次深度线性、文档数对数增长即可求解，并发现"远距离丢失"现象后提出预训练-微调策略，在 WordNet 上将远距离召回率从 19% 提升至 76%。

**[Hifi-Rag Hierarchical Content Filtering And Two-Pass Generation For Open-Domain ](information_retrieval/hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)**

:   通过分离轻量级 Flash 模型的过滤能力与 Pro 模型的推理能力，构建多阶段管道（查询优化→分层过滤→两阶段生成→引文验证），在 MMU-RAGent 竞赛中实现 SOTA 性能。

**[How Should We Evaluate Data Deletion In Graph-Based Ann Indexes](information_retrieval/how_should_we_evaluate_data_deletion_in_graph-based_ann_indexes.md)**

:   针对图基ANN索引缺乏统一数据删除评估方法的问题，形式化定义了逻辑删除、物理删除和重建三种基准方法，提出面向实际部署的评估框架和指标体系，并基于实验分析提出Deletion Control算法在精度约束下动态切换删除策略。

**[Improving Consistency In Retrieval-Augmented Systems With Group Similarity Rewar](information_retrieval/improving_consistency_in_retrieval-augmented_systems_with_group_similarity_rewar.md)**

:   提出 Con-RAG 框架，通过 Paraphrased Set GRPO (PS-GRPO) 在语义等价查询的多次生成之间计算组相似度奖励，训练 RAG 系统的生成器在释义输入下产生信息一致的输出，无需显式真实标签监督即可同时提升一致性和准确性。

**[Is Prm Necessary Problem-Solving Rl Implicitly Induces Prm Capability In Llms](information_retrieval/is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)**

:   系统研究表明纯 RL 训练（无需显式 PRM 监督）能隐式诱导出强大的过程判断能力，且现有 PRM 在 DeepSeek-R1/QwQ-32B 等强推理模型上甚至不如简单多数投票有效；提出 Self-PRM 让模型用自身的内部奖励信号重排输出，一致性地优于外部 PRM。

**[Learning Task-Agnostic Representations Through Multi-Teacher Distillation](information_retrieval/learning_task-agnostic_representations_through_multi-teacher_distillation.md)**

:   提出基于互信息最大化的任务无关多教师蒸馏框架，通过高斯核估计教师嵌入的条件分布来训练学生模型，使其在不依赖任何下游任务标签的情况下学到高信息密度的通用表示，在文本、视觉和分子建模三个领域均取得了同体量最优性能。

**[Mind The Gap Aligning Knowledge Bases With User Needs To Enhance Mental Health R](information_retrieval/mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)**

:   提出一种基于"需求差距"分析的知识库增强框架，通过叠加真实用户数据（论坛帖子）与现有心理健康资源库来识别内容空白，并用定向增强策略以最少的文档增量达到接近完整语料库的 RAG 检索质量。

**[Mir-Bench Can Your Llm Recognize Complicated Patterns Via Many-Shot In-Context R](information_retrieval/mir-bench_can_your_llm_recognize_complicated_patterns_via_many-shot_in-context_r.md)**

:   提出 MIR-Bench，首个大规模多样化的 many-shot 上下文推理基准，通过从编程题中自动生成输入输出对来测试 LLM 的模式识别能力，发现 LLM 在 many-shot 场景下存在注意力分散导致的性能饱和现象，且转导推理普遍优于归纳推理。

**[Mitra An Ai Assistant For Knowledge Retrieval In Physics Collaborations](information_retrieval/mitra_an_ai_assistant_for_knowledge_retrieval_in_physics_collaborations.md)**

:   提出 MITRA，一个面向大型物理实验协作（如 CERN CMS）的本地化 RAG 系统，采用两层向量数据库架构（摘要库 + 全文库）和完全本地部署策略，在语义检索任务上显著优于传统关键词搜索（BM25），Precision@1 从 0.13 提升至 0.75。

**[Murating A High Quality Data Selecting Approach To Multilingual Large Language M](information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)**

:   提出 MuRating，一个可扩展的多语言数据选择框架：先通过配对比较聚合多个英文数据质量评分器，再借助翻译将质量信号迁移到 17 种语言，训练出语言无关的多语言质量评估模型，在 1.2B 和 7B 规模 LLM 预训练中取得了持续的性能提升。

**[Rag-Igbench Innovative Evaluation For Rag-Based Interleaved Generation In Open-D](information_retrieval/rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)**

:   提出 RAG-IGBench，一个专门评估基于检索增强生成的交错图文内容质量的 benchmark，设计了覆盖文本质量、图像质量和图文一致性三个维度的创新自动评估指标，并验证了与人类评估的高度相关性。

**[Reliable Decision Making Via Calibration Oriented Retrieval Augmented Generation](information_retrieval/reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)**

:   提出 CalibRAG 框架，通过训练一个温度条件化的 forecasting function 来确保 RAG 辅助决策过程中的置信度校准，不仅改善校准质量还提升了准确率。

**[Retrieval-Augmented Generation For Reliable Interpretation Of Radio Regulations](information_retrieval/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)**

:   针对无线电法规这一法律敏感的高风险领域，设计了专用 RAG 管道并构建了首个 ITU 无线电法规多选题评估集，检索准确率达 97%，在 GPT-4o 上实现 +11.9% 的问答准确率提升，远超直接将文档塞入 prompt 的方式。

**[Retrieval Is Not Enough Enhancing Rag Reasoning Through Test-Time Critique And O](information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)**

:   提出 AlignRAG 框架，将 RAG 重新定义为"检索增强推理"，通过训练专用 Critic Language Model (CLM) 在测试时迭代批评和修正推理过程，解决推理与检索证据之间的错位问题，8B CLM 在 OOD 任务上超越 72B 标准 CLM。

**[Rmit-Adms At The Mmu-Rag Neurips 2025 Competition](information_retrieval/rmit-adms_at_the_mmu-rag_neurips_2025_competition.md)**

:   提出Routing-to-RAG (R2RAG)系统，通过LLM查询分类器将简单查询路由到单轮Vanilla RAG、复杂查询路由到迭代式Vanilla Agent，全部基于Qwen3-4B（未量化）和Qwen3-Reranker-0.6B两个轻量模型在单块消费级GPU上运行，获NeurIPS 2025 MMU-RAG竞赛开源赛道Best Dynamic Evaluation奖。

**[Scale-Invariant Attention](information_retrieval/scale-invariant_attention.md)**

:   借鉴自然图像的尺度不变性，提出对 attention logits 做位置相关的乘性缩放和加性偏移变换，使注意力在不同 token 范围上的总权重和稀疏度满足尺度不变性，从而实现从短序列训练到长序列推理的零样本泛化（4k→64k 仅需一个超参数 $\tau$）。

**[Scaling Language-Centric Omnimodal Representation Learning](information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)**

:   提出 LCO-Emb 框架，发现多模态大模型（MLLM）在生成式预训练中已隐式建立跨模态对齐，仅需轻量级的纯文本对比学习微调即可激活全模态表示能力，并发现生成能力与表示性能正相关的 Generation-Representation Scaling Law (GRSL)。

**[Secon-Rag A Two-Stage Semantic Filtering And Conflict-Free Framework For Trustwo](information_retrieval/secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)**

:   提出 SeCon-RAG 两阶段防御框架，第一阶段用聚类+语义图联合过滤毒化文档，第二阶段在推理时做冲突感知过滤，在5个LLM和3个QA数据集上全面超越现有RAG防御方法，在100%投毒率下仍保持高准确率和极低攻击成功率。

**[Superclip Clip With Simple Classification Supervision](information_retrieval/superclip_clip_with_simple_classification_supervision.md)**

:   在CLIP对比学习框架中引入一个超简单的分类损失（仅需添加一个轻量线性层，FLOPs增加仅0.077%），利用原始文本token的分类信号恢复CLIP未充分利用的细粒度文本监督，在零样本分类、图文检索和纯视觉任务上一致提升性能。

**[The Atlas Of In-Context Learning How Attention Heads Shape In-Context Retrieval ](information_retrieval/the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)**

:   通过 AttnLRP 归因方法系统解剖 LLM 在 in-context retrieval augmented QA 中的内部机制，发现三类功能特化的注意力头——Task heads（中间层，解析指令/问题）、Retrieval heads（后层，逐字复制上下文答案）、Parametric heads（编码参数化知识），并通过 Function Vector 注入和来源追踪探针验证其功能，在 Llama-3.1/Mistral/Gemma 上 ROC AUC ≥94%。

**[The Narrow Gate Localized Imagetext Communication In Native](information_retrieval/the_narrow_gate_localized_imagetext_communication_in_native.md)**

:   通过系统性的可解释性分析发现，原生多模态VLM（Chameleon、Emu3）中图像到文本的跨模态信息传递集中于单一的end-of-image [EOI] token——形成"narrow gate"瓶颈，删除[EOI]的注意力导致性能崩溃；而非原生VLM（LLaVA等）的信息传递是分布式的。这一机制差异可被利用于语义操控和鲁棒性改进。

**[Windsock Is Dancing Adaptive Multimodal Retrieval-Augmented Generation](information_retrieval/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)**

:   提出Windsock+DANCE双组件框架解决多模态RAG的三个核心问题：Windsock模块根据查询自适应决定**何时检索**和**检索什么模态**（文本/图像/不检索），DANCE指令微调策略通过动态选择模型薄弱模态进行噪声鲁棒训练来提升**如何利用**检索信息的能力，整体性能提升17.07%同时减少8.95%检索次数。

**[With Limited Data For Multimodal Alignment Let The Structure Guide You](information_retrieval/with_limited_data_for_multimodal_alignment_let_the_structure_guide_you.md)**

:   提出 STRUCTURE 正则化和基于表示相似度的层选择策略，仅用少量配对数据（数万对，不到常规方法的1%）即可实现冻结单模态基础模型的高质量跨模态对齐，在24个零样本分类和检索基准上平均提升51.6%和91.8%。

**[Worse Than Zero-Shot A Fact-Checking Dataset For Evaluating The Robustness Of Ra](information_retrieval/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)**

:   提出 RAGuard 基准数据集，首次系统评估 RAG 系统对误导性检索内容的鲁棒性。通过从 Reddit 构建包含支持性、误导性和无关文档的真实检索语料库，揭示所有测试的 LLM-RAG 系统在面对误导性检索时表现**比零样本基线更差**，而人类标注者能保持一致判断。

---

## 🔄 自监督 { #self_supervised }

**[A Joint Learning Approach To Hardware Caching And Prefetching](self_supervised/a_joint_learning_approach_to_hardware_caching_and_prefetching.md)**

:   提出将硬件缓存替换策略和预取策略进行联合训练的学习框架，通过共享编码器和对比学习两种方式构建共享特征表征，打破两个策略独立训练时的性能瓶颈。

**[Adv-Ssl Adversarial Self-Supervised Representation Learning With Theoretical Gua](self_supervised/adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)**

:   提出 Adv-SSL，通过将协方差正则项的 Frobenius 范数重写为 minimax 对偶形式，消除了 Barlow Twins 等方法中样本级风险的有偏估计问题，在不增加额外计算成本的前提下显著提升下游分类性能，并给出端到端的理论收敛保证。

**[Asymptotic And Finite-Time Guarantees For Langevin-Based Temperature Annealing I](self_supervised/asymptotic_and_finite-time_guarantees_for_langevin-based_temperature_annealing_i.md)**

:   本文通过将嵌入演化建模为紧致黎曼流形上的 Langevin 动力学，证明了经典模拟退火的收敛保证可以扩展到对比学习的温度调度设定中：缓慢对数逆温度调度保证概率收敛到全局最优表示集合，而更快的调度则可能陷入次优极小值。

**[Brainomni A Brain Foundation Model For Unified Eeg And Meg Signals](self_supervised/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)**

:   提出 BrainOmni——首个统一 EEG 和 MEG 的脑信号基础模型，通过 BrainTokenizer（含物理传感器编码器）将异构脑电/脑磁信号离散化为统一 token，再用 Criss-Cross Transformer 进行自监督掩码预测预训练，在阿尔茨海默病检测上提升 11.7 个百分点，并实现对完全未见设备的零样本重建泛化。

**[Connecting Jensenshannon And Kullbackleibler Divergences A N](self_supervised/connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)**

:   推导了一般情况下KL散度关于JS散度的最优紧致下界$\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$，证明训练判别器最小化交叉熵损失等价于最大化互信息的一个保证下界，为JSD基于的判别式表示学习方法提供了缺失的理论基础，并在MI估计和Information Bottleneck框架中验证了紧致性与实用性。

**[Continuous Subspace Optimization For Continual Learning](self_supervised/continuous_subspace_optimization_for_continual_learning.md)**

:   提出 CoSO 框架，通过从每步梯度的 SVD 动态导出连续子空间（而非 LoRA 的固定子空间），结合历史任务正交投影防止干扰和 Frequent Directions 高效聚合梯度信息，在 ImageNet-R 20 任务上以 78.19% 最终准确率超越最佳 baseline 2.77 个百分点。

**[Contrastive Consolidation Of Top-Down Modulations Achieves Sparsely Supervised C](self_supervised/contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Contrastive Representations For Temporal Reasoning](self_supervised/contrastive_representations_for_temporal_reasoning.md)**

:   提出 CRTR（Contrastive Representations for Temporal Reasoning），通过在训练批次中重复同一轨迹来引入轨迹内负样本对，消除标准时间对比学习对静态上下文特征的依赖，学习到反映时间结构的表征，在魔方等组合推理任务上首次实现无搜索求解。

**[Datarater Meta-Learned Dataset Curation](self_supervised/datarater_meta-learned_dataset_curation.md)**

:   提出 DataRater，一个基于元梯度（meta-gradient）的数据价值评估框架，通过元学习自动为每个训练数据点打分并过滤低质量数据，在多个预训练数据集上实现最高 46.6% 的净计算量节省，且在 400M 内部模型上训练的 DataRater 可直接泛化到 50M–1B 规模的 LLM 训练中。

**[Foundation Cures Personalization Improving Personalized Models Prompt Consistenc](self_supervised/foundation_cures_personalization_improving_personalized_models_prompt_consistenc.md)**

:   FreeCure发现面部个性化模型的身份嵌入会覆盖但不破坏基础模型的prompt控制能力，据此提出无训练框架，通过Foundation-Aware Self-Attention（FASA）将基础模型的属性信息注入个性化生成过程，在保持身份保真度的同时大幅提升prompt一致性，可无缝集成到SD/SDXL/FLUX等主流模型。

**[Foundation Models For Scientific Discovery From Paradigm Enhancement To Paradigm](self_supervised/foundation_models_for_scientific_discovery_from_paradigm_enhancement_to_paradigm.md)**

:   提出三阶段框架（元科学整合→混合人机共创→自主科学发现）来描绘基础模型正推动科学范式从工具增强向范式转型演变的图景，并系统综述了 FM 在实验/理论/计算/数据四大科学范式中的整合应用。

**[Hybrid Autoencoders For Tabular Data Leveraging Model-Based Augmentation In Low-](self_supervised/hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)**

:   提出 TANDEM（Tree-And-Neural Dual Encoder Model），一种混合自编码器架构，通过联合训练神经网络编码器和遗忘软决策树（OSDT）编码器，并引入样本级随机门控网络作为可学习的数据增强，在低标签表格数据场景下实现了超越强基线（包括树模型和深度学习方法）的性能。

**[Implicit Modeling For Transferability Estimation Of Vision Foundation Models](self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)**

:   提出隐式可迁移性建模（ITM）框架，通过隐变量z隐式编码模型-任务对的迁移能力，结合分治变分近似（DVA）高效模拟嵌入空间演化，在10个下游任务和10个多样化预训练模型上的加权Kendall tau_w从此前最优的0.45提升至0.61。

**[Know Thyself By Knowing Others Learning Neuron Identity From Population Context](self_supervised/know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)**

:   提出NuCLR自监督框架，通过对比学习对群体神经活动中同一神经元的不同时间窗口拉近、不同神经元推远，学习包含群体上下文的神经元级表征，在细胞类型和脑区解码上达到新SOTA，并首次展示了跨动物零样本泛化和数据缩放规律。

**[Long-Tailed Recognition Via Information-Preservable Two-Stage Learning](self_supervised/long-tailed_recognition_via_information-preservable_two-stage_learning.md)**

:   提出信息保持的两阶段学习框架：第一阶段用 Balanced Negative Sampling (BNS) 基于互信息最大化学习有效且可分的特征空间，第二阶段用 Information-Preservable DPP (IP-DPP) 采样数学上信息量最大的样本来纠正多数类偏向的决策边界，在多个长尾数据集上取得 SOTA。

**[Manifolds And Modules How Function Develops In A Neural Foundation Model](self_supervised/manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)**

:   从计算神经科学视角"打开黑箱"分析 SOTA 神经活动基础模型 (FNN)，通过构建解码流形和编码流形发现其各处理模块（编码器、循环、读出）展现出质性不同的表征结构，且与生物视觉系统存在关键差异。

**[Memory-Integrated Reconfigurable Adapters A Unified Framework For Settings With ](self_supervised/memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。

**[Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization](self_supervised/minimal_semantic_sufficiency_meets_unsupervised_domain_generalization.md)**

:   MS-UDG 在无类别标签和域标签的条件下，通过信息解纠缠模块（IDM）将表征分解为语义和变异成分，配合最小语义充分性优化模块（SROM）最大化语义信息同时最小化变异干扰，在 PACS 上达 72.89% 准确率（+1.5% vs CycleMAE），理论证明最小充分语义表征最小化下游贝叶斯错误率。

**[Mitra Mixed Synthetic Priors For Enhancing Tabular Foundation Models](self_supervised/mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)**

:   首次系统研究合成先验的设计原则，发现多样性、独特性和真实数据对齐是关键属性，据此提出 Mitra——一个基于精心筛选的混合合成先验训练的表格基础模型，在分类和回归基准上一致超越 TabPFNv2 和 TabICL。

**[One Filters All A Generalist Filter For State Estimation](self_supervised/one_filters_all_a_generalist_filter_for_state_estimation.md)**

:   提出 LLM-Filter，将 LLM 重编程为通用状态估计器，通过 System-as-Prompt（SaP）机制使冻结的 LLM 在未见动力系统上实现零样本泛化，性能超越 SOTA 学习型滤波器。

**[Seal Semantic-Aware Hierarchical Learning For Generalized Category Discovery](self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)**

:   提出 SEAL 框架，利用自然存在的语义层级结构（而非手工设计的抽象层级）指导广义类别发现，通过层级语义引导的软对比学习和跨粒度一致性模块，在细粒度基准上取得 SOTA 性能。

**[Soft Task-Aware Routing Of Experts For Equivariant Representation Learning](self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)**

:   提出 STAR（Soft Task-Aware Routing），通过 MoE 路由机制协调不变性和等变性表示学习任务间的共享与专属信息，减少冗余特征学习，提升下游任务迁移性能。

**[Starformer Semi-Supervised Task-Informed Representation Learning Via Dynamic Att](self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)**

:   提出 STaRFormer，通过动态注意力区域掩码（DAReM）识别任务关键区域并施加掩码扰动，配合批内+类内半监督对比学习将任务信息嵌入潜在表示，在 56 个数据集（含非平稳、不规则采样、分类/异常检测/回归）上全面超越 SOTA。

**[T-Regs Minimum Spanning Tree Regularization For Self-Supervised Learning](self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)**

:   提出 T-REGS——一种基于最小生成树(MST)长度最大化的自监督学习正则化框架，理论证明可同时防止维度坍缩并促进表示分布均匀性，在紧致黎曼流形上成立，实验在标准 JE-SSL 基准上验证了有效性。

**[Tabarena A Living Benchmark For Machine Learning On Tabular Data](self_supervised/tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)**

:   提出 TabArena，首个持续维护的"活跃"表格数据基准系统，从 1053 个数据集中精选 51 个、纳入 16 个模型，通过大规模实验（约 2500 万次模型训练）发现：后验集成下深度学习模型已追平甚至超越 GBDT，表格基础模型在小数据上表现突出，跨模型集成可进一步推进 SOTA。

**[Tabstar A Tabular Foundation Model For Tabular Data With Text Fields](self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)**

:   提出 TabSTAR，一个专为含文本字段的表格数据设计的基础模型：通过解冻文本编码器（e5-small-v2）端到端优化文本表征 + 目标感知 token 注入分类目标语义信息 + 无数据集特定参数的架构实现跨数据集迁移学习，在 350 个数据集上预训练后，分类任务上 14 个数据集中 12 个超越 CatBoost-Tuned（4h 调参），8/11 超越 TabPFN-v2。

**[Towards Reliable And Holistic Visual In-Context Learning Prompt Selection](self_supervised/towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)**

:   提出RH-Partial2Global方法，首次用Spearman秩相关检验证明VICL中"相似性优先假设"虽统计显著但相关强度极弱($\bar{\rho} \approx 0.03\text{-}0.05$)，通过Jackknife共形预测构建可靠候选集+覆盖设计实现全面均匀的成对偏好采样，在分割/检测/着色三个视觉任务上一致超越SOTA。

**[Trident Tri-Modal Molecular Representation Learning With Taxonomic Annotations A](self_supervised/trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)**

:   提出 TRIDENT 三模态分子表示学习框架，引入层次分类标注（HTA）作为第三模态，结合体积对比损失做全局三模态对齐和功能团-文本局部对齐，通过动量机制动态平衡两者，在 18 个分子属性预测任务上达到 SOTA。

**[Uncertainty-Guided Model Selection For Tabular Foundation Models In Biomolecule ](self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)**

:   本文提出OligoICP方法，利用TabPFN模型的预测分位数间距（IQR）作为无标签模型选择启发式指标，在siRNA敲低效率预测中实现了优于专用SOTA模型和朴素集成的性能。

**[Understanding Ice Crystal Habit Diversity With Self-Supervised Learning](self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)**

:   本文首次将自监督学习（SSL）应用于冰晶图像的潜在表征学习，通过在大规模云粒子图像上预训练ViT，学习冰晶形态的连续潜在表征，并用vMF浓度参数量化冰晶多样性，实现30倍计算效率提升的同时取得最佳分类准确率84.39%。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Adaptive Discretization For Consistency Models](image_restoration/adaptive_discretization_for_consistency_models.md)**

:   提出ADCM——通过将一致性模型的离散化步长形式化为局部一致性（可训练性）与全局一致性（稳定性）的约束优化问题，并用Gauss-Newton法求闭式解，实现自适应离散化，在CIFAR-10上用不到25%训练预算超越所有先前CM。

**[Audio Super-Resolution With Latent Bridge Models](image_restoration/audio_super-resolution_with_latent_bridge_models.md)**

:   提出 AudioLBM，将音频波形压缩到连续隐空间，用桥模型实现从低分辨率到高分辨率的 latent-to-latent 生成过程，配合频率感知训练扩展数据利用和级联设计突破 48kHz 上限，在语音/音效/音乐上全面超越 AudioSR 等方法，并首次实现 any-to-192kHz 音频超分。

**[Denoiserotator Enhance Pruning Robustness For Llms Via Importance Concentration](image_restoration/denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)**

:   提出 DenoiseRotator，在剪枝前通过可学习正交变换最小化参数重要性分数的信息熵，将重要性集中到少数参数上，使 LLaMA3-70B 在 2:4 半结构化稀疏下困惑度退化缩小 58%（8.1→3.4），可即插即用组合 Magnitude/Wanda/SparseGPT。

**[Dynaguide Steering Diffusion Polices With Active Dynamic Guidance](image_restoration/dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)**

:   提出 DynaGuide，在推理时通过外部潜在动力学模型对预训练扩散策略施加 classifier guidance，无需修改策略权重即可引导机器人朝向任意正/负目标，在 CALVIN 仿真上平均成功率 70%，真实机器人达 80%。

**[Enhancing Infrared Vision Progressive Prompt Fusion Network And Benchmark](image_restoration/enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)**

:   针对热红外(TIR)图像中低对比度、模糊、噪声等多种退化耦合的问题，提出基于双提示融合的渐进式网络PPFN和选择性渐进训练策略SPT，并构建首个大规模多场景TIR基准数据集HM-TIR，在复合退化场景下PSNR提升8.76%。

**[Fiper Factorized Features For Robust Image Super-Resolution And Compression](image_restoration/fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)**

:   提出 Factorized Features 统一表示——将图像分解为可学习的非均匀基与空间变化系数，配合锯齿坐标变换和多频调制，在 4× 超分辨率上 PSNR 相对提升 204.4%（HAT-L-F vs SwinIR），在图像压缩上 BD-rate 相比 VTM 降低 21.09%。

**[Gc4Nc A Benchmark Framework For Graph Condensation On Node Classification With N](image_restoration/gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)**

:   提出 GC4NC——首个系统化的图凝缩（Graph Condensation）评估基准框架，跨 8 个维度（性能/效率/隐私保护/去噪/NAS有效性/可迁移性等）统一评估多种图凝缩方法，发现轨迹匹配方法最优、无结构方法效率最高，并在 1000x 压缩下图凝缩显著优于图像凝缩。

**[Implicit Augmentation From Distributional Symmetry In Turbulence Super-Resolutio](image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)**

:   本文揭示湍流的统计各向同性本身就是一种隐式数据增强，使得标准CNN在超分辨率任务中无需显式旋转增强或等变架构即可部分习得旋转等变性，并展示了等变误差的尺度依赖性与Kolmogorov局部各向同性假说一致。

**[Improving Diffusion-Based Inverse Algorithms Under Few-Step Constraint Via Learn](image_restoration/improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)**

:   提出 Learnable Linear Extrapolation (LLE)——用可学习的线性组合系数将当前和历史 clean data estimate 组合，以增强任何符合 Sampler-Corrector-Noiser 范式的扩散逆问题算法在少步（3-5步）下的表现，仅需 50 个样本、几分钟训练，跨 9+ 算法 × 5 个任务一致提升。

**[Latent Harmony Synergistic Unified Uhd Image Restoration Via Latent Space Regula](image_restoration/latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)**

:   提出 Latent Harmony 两阶段框架，通过潜在空间正则化构建泛化性 VAE（LH-VAE），并引入高频引导的可控 LoRA 微调机制，在保持结构完整性的同时实现 UHD 图像多退化类型统一修复的保真度-感知质量灵活权衡。

**[Learning Cocoercive Conservative Denoisers Via Helmholtz Decomposition For Poiss](image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)**

:   提出共循环保守(CoCo)去噪器概念，通过广义Helmholtz分解设计新的训练策略——Hamiltonian正则化促进保守性 + 谱正则化促进共循环性——使去噪器成为隐式弱凸先验的近端算子，从而在Poisson逆问题（光子受限去卷积、低剂量CT等）中实现有收敛保证且性能优越的PnP方法。

**[Luminance-Aware Statistical Quantization Unsupervised Hierarchical Learning For ](image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)**

:   提出 LASQ 框架，将低光图像增强重新定义为基于分层亮度分布的统计采样过程，利用自然亮度转换中固有的幂律分布特性，通过 MCMC 采样生成层次化亮度适配算子，嵌入扩散模型前向过程实现无监督增强，无需正常光照参考即可工作。

**[Map Estimation With Denoisers Convergence Rates And Guarantees](image_restoration/map_estimation_with_denoisers_convergence_rates_and_guarantees.md)**

:   证明了一个简单的 MMSE 去噪器迭代平均算法（与 Cold Diffusion 等实践方法密切相关）在对数凹先验假设下可证明收敛到负对数先验的近端算子，收敛速率为 Õ(1/k)，为一类经验上成功但缺乏理论保证的去噪方法提供了严格的理论基础，并将其嵌入近端梯度下降框架解决 MAP 估计问题。

**[Modem A Morton-Order Degradation Estimation Mechanism For Adverse Weather Image ](image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)**

:   提出 MODEM 框架，通过 Morton 编码空间扫描与选择性状态空间模型（SSM）结合，建模空间异质性天气退化特征，配合双重退化估计模块提供全局和局部先验，实现多种天气退化图像的统一自适应复原 SOTA。

**[Moe-Gyro Self-Supervised Over-Range Reconstruction And Denoising For Mems Gyrosc](image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)**

:   提出MoE-Gyro自监督专家混合框架，通过超量程重建专家(ORE，含高斯衰减注意力和物理信息约束)和降噪专家(DE，含双分支互补掩码和FFT引导增强)同时解决MEMS陀螺仪量程-噪声的根本权衡，将可测量范围从±450°/s扩展到±1500°/s，偏置不稳定性降低98.4%。

**[Mro Enhancing Reasoning In Diffusion Language Models Via Multi-Reward Optimizati](image_restoration/mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)**

:   首次系统分析扩散语言模型（DLM）推理短板的根因——去噪过程中token独立生成导致序列内/序列间相关性缺失，提出多奖励优化框架MRO，在test-time scaling、reject sampling和RL三种模式下均显著提升LLaDA-8B的推理性能，MATH500从34.4%提升至37.4%。

**[Ms-Bart Unified Modeling Of Mass Spectra And Molecules For Structure Elucidation](image_restoration/ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)**

:   提出 MS-Bart，通过统一词表将分子指纹和分子结构（SELFIES）映射到共享的 token 空间，在 400 万指纹-分子对上进行多任务预训练，再通过实验谱微调和化学反馈对齐，实现从质谱到分子结构的高效生成。

**[Real-World Adverse Weather Image Restoration Via Dual-Level Reinforcement Learni](image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)**

:   提出双层强化学习框架（DRL），结合物理驱动的百万级合成天气数据集HFLS-Weather进行高质量冷启动训练，通过局部扰动驱动图像质量优化（PIQO）和全局元控制器多智能体协作，实现真实恶劣天气图像的自适应复原。

**[Rethinking Circuit Completeness In Language Models And Or And Adder Gates](image_restoration/rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)**

:   系统引入AND、OR、ADDER三种逻辑门来分解语言模型电路，揭示电路不完整性主要源于OR门的遗漏，提出结合noising和denoising干预的框架来完整恢复三种逻辑门，同时保证忠实度和完整性。

**[Rethinking Nighttime Image Deraining Via Learnable Color Space Transformation](image_restoration/rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)**

:   基于"夜间雨在YCbCr的Y通道（亮度）差异远大于RGB"的统计发现，提出可学习颜色空间转换器(CSC)在Y通道做去雨、隐式光照引导(IIG)编码夜间不均匀光照、以及光照感知的高质量数据集HQ-NightRain，三管齐下显著提升夜间去雨效果。

**[Scan Self-Denoising Monte Carlo Annotation For Robust Process Reward Learning](image_restoration/scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)**

:   提出 SCAN 框架，通过分析 Monte Carlo 注释中的噪声分布，设计自去噪采样策略和鲁棒学习损失，仅用 1.5B 模型生成的 101K 样本训练的 PRM 即超越人工标注数据集 PRM800K 的效果。

**[Scsplit Bringing Severity Cognizance To Image Decomposition In Fluorescence Micr](image_restoration/scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)**

:   提出 scSplit，通过引入混合比例感知的归一化模块（SCIN）和回归网络（Reg），使基于 InDI 的迭代图像分解方法能够感知荧光显微镜图像中两种结构叠加的严重程度，在5个公开数据集上统一解决图像分离和渗透去除两个任务。

**[Spend Wisely Maximizing Post-Training Gains In Iterative Synthetic Data Bootstra](image_restoration/spend_wisely_maximizing_post-training_gains_in_iterative_synthetic_data_bootstra.md)**

:   首次从理论上分析了迭代合成数据自举训练中的预算分配问题，证明恒定策略无法高概率收敛，而指数增长策略在最坏情况下优于多项式策略，并在图像去噪（DPM）和数学推理（LLM）实验中验证了该结论。

**[Spiking Meets Attention Efficient Remote Sensing Image Super-Resolution With Att](image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)**

:   提出 SpikeSR，首个基于注意力脉冲神经网络(SNN)的遥感图像超分辨率框架，通过脉冲注意力块(SAB)结合混合维度注意力(HDA)和可变形相似度注意力(DSA)，在 AID/DOTA/DIOR 上达到 SOTA 性能同时保持高计算效率。

**[The Effect Of Optimal Self-Distillation In Noisy Gaussian Mixture Model](image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)**

:   利用统计物理的replica方法对噪声高斯混合数据上的超参优化多阶段自蒸馏进行严格理论分析，揭示硬伪标签的去噪效应是自蒸馏性能提升的主要驱动力，中等规模数据集获益最显著，并提出早停（限制蒸馏阶段数）和偏置参数固定两个实用改进策略，CIFAR-10+ResNet实验验证了理论预测。

---

## 🔗 因果推理 { #causal_inference }

**[A Principle Of Targeted Intervention For Multi-Agent Reinforcement Learning](causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)**

:   提出基于多智能体影响图（MAIDs）的**目标干预范式（Targeted Intervention）**，通过仅对单个目标智能体施加**预策略干预（Pre-Strategy Intervention, PSI）**，引导整个多智能体系统收敛到满足额外期望结果的优选Nash均衡，无需对所有智能体进行全局干预。

**[Alignment Of Large Language Models With Constrained Learning](causal_inference/alignment_of_large_language_models_with_constrained_learning.md)**

:   将LLM对齐建模为约束优化问题——最大化主要奖励（如有用性）同时满足次要效用约束（如安全性），提出基于拉格朗日对偶的迭代方法交替更新LLM策略和对偶变量，理论上刻画了分布空间与参数空间之间的原对偶间隙和最优性间隙，证明可以找到近最优约束LLM策略（误差仅来自参数化间隙）。

**[An Analysis Of Causal Effect Estimation Using Outcome Invariant Data Augmentatio](causal_inference/an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)**

:   首次系统分析"结果不变数据增强"（outcome invariant DA）在因果效应估计中的作用，证明当 DA 操作保持结果变量的不变性时等价于对处理变量的软干预，可减少混杂偏差；进一步提出 IV-like（IVL）回归框架，将 DA 参数用作"类工具变量"，通过对抗性 DA 组合进一步降低偏差。

**[Bi-Level Decision-Focused Causal Learning For Large-Scale Marketing Optimization](causal_inference/bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)**

:   提出 Bi-DFCL，通过双层优化框架联合利用观测数据和 RCT 实验数据来训练营销资源分配模型：上层用 RCT 数据的无偏决策损失端到端训练 Bridge Network 来动态纠正下层在观测数据上的偏差，同时设计了基于原始问题的可微代理决策损失（PPL/PIFD）和隐式微分算法，解决了传统两阶段方法的预测-决策不一致和偏差-方差困境。已在美团大规模在线部署。

**[Causality-Induced Positional Encoding For Transformer-Based Representation Learn](causal_inference/causality-induced_positional_encoding_for_transformer-based_representation_learn.md)**

:   CAPE 通过从表格数据中学习特征间的因果DAG结构，将其嵌入双曲空间生成因果感知的旋转位置编码（RoPE），使 Transformer 能处理非序列但因果相关的特征数据，在多组学数据的下游任务上显著提升性能。

**[Characterization And Learning Of Causal Graphs From Hard Interventions](causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)**

:   首次系统分析硬干预（hard interventions）在含隐变量因果发现中的理论优势，提出广义do-演算（4条规则）和孪生增强MAG图表示，给出 $\mathcal{I}$-Markov 等价类的充要图条件，并设计可证明正确的FCI变体学习算法；实验表明硬干预比软干预将等价类缩小37-57%。

**[Conformal Prediction For Causal Effects Of Continuous Treatments](causal_inference/conformal_prediction_for_causal_effects_of_continuous_treatments.md)**

:   首次为连续处理变量（如药物剂量）的因果效应构建共形预测区间，通过倾向性偏移参数化和分位数回归，在已知/未知倾向性两种场景下均提供有限样本 $1-\alpha$ 覆盖保证。

**[Counterfactual Reasoning For Steerable Pluralistic Value Alignment Of Large Lang](causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)**

:   提出COUPLE框架，通过构建结构因果模型（SCM）建模多维价值观的依赖关系与优先级，并利用反事实推理实现LLM对任意细粒度多元价值目标的可控对齐。

**[Cyclic Counterfactuals Under Shift-Scale Interventions](causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)**

:   本文在循环（非DAG）结构因果模型中建立了shift-scale软干预下反事实推理的理论框架，证明了全局收缩条件保证循环SCM的唯一可解性，并推导出反事实分布的sub-Gaussian集中不等式。

**[Demystifying Spectral Feature Learning For Instrumental Variable Regression](causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)**

:   为基于谱特征的非参数工具变量（NPIV）回归建立严格的泛化误差界，揭示性能由结构函数与条件期望算子的**谱对齐**（近似误差）和**奇异值衰减速度**（估计误差）两因素共同决定，提出 Good-Bad-Ugly 三分类法并设计数据驱动诊断工具。

**[Differentiable Structure Learning And Causal Discovery For General Binary Data](causal_inference/differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)**

:   提出基于多元伯努利分布（MVB）的通用可微结构学习框架，不假设特定数据生成过程，能捕获二值离散变量间的任意高阶依赖关系，并证明在一般设定下DAG不可识别但可恢复最小等价类（Markov等价类）。

**[Do-Pfn In-Context Learning For Causal Effect Estimation](causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)**

:   提出 Do-PFN，将 Prior-data Fitted Networks (PFN) 扩展到因果效应估计，在大量合成 SCM 数据上预训练 Transformer 进行 in-context 因果推理，仅需观测数据即可预测干预分布（CID）和 CATE，无需因果图知识或不混杂假设，在合成和半合成实验中表现出色。

**[Domain-Adapted Granger Causality For Real-Time Cross-Slice Attack Attribution In](causal_inference/domain-adapted_granger_causality_for_real-time_cross-slice_attack_attribution_in.md)**

:   提出一种面向6G网络切片的域适应Granger因果框架，将增强型Granger因果检验与网络资源争用建模相结合，实现实时跨切片攻击归因，在1100个攻击场景上达到89.2%准确率和87ms响应时间，显著超越现有统计、深度学习和因果发现方法。

**[Dynamic Causal Discovery In Alzheimers Disease Through Latent Pseudotime Modelli](causal_inference/dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)**

:   将 BN-LTE（贝叶斯网络+潜在时间嵌入）应用于 ADNI 真实 AD 数据，推断随疾病伪时间演变的动态因果图，伪时间预测诊断 AUC 0.82 远超年龄 0.59，并揭示了新型生物标志物 NfL/GFAP 与传统 AD 标志物之间的动态因果关系。

**[Few-Shot Knowledge Distillation Of Llms With Counterfactual Explanations](causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)**

:   提出 CoD（Counterfactual-explanation-infused Distillation），通过将反事实解释注入少样本训练集来精确映射 teacher 决策边界，在 6 个数据集上仅用 8–512 样本即显著超越标准蒸馏方法。

**[From Black-Box To Causal-Box Towards Building More Interpretable Models](causal_inference/from_black-box_to_causal-box_towards_building_more_interpretable_models.md)**

:   提出"因果可解释性"（causal interpretability）的形式化定义，证明黑盒模型和概念瓶颈模型均不满足该性质，给出完整的图判据确定哪些模型架构能一致地回答反事实问题，揭示了因果可解释性与预测精度之间的根本性权衡。

**[Gst-Unet A Neural Framework For Spatiotemporal Causal Inference With Time-Varyin](causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)**

:   提出 GST-UNet，将 U-Net 时空编码器与迭代 G-computation 相结合，从**单条时空观测轨迹**中估计位置特异性的条件平均潜在结果 (CAPO)，可同时处理干扰（interference）、空间混杂、时间延续和时变混杂，并在加州山火烟雾对呼吸系统住院率的因果分析中验证了实用价值。

**[Its Hard To Be Normal The Impact Of Noise On Structure-Agnostic Estimation](causal_inference/its_hard_to_be_normal_the_impact_of_noise_on_structure-agnostic_estimation.md)**

:   证明 Double Machine Learning (DML) 在高斯处理噪声下是极小极大最优的（$O(\epsilon^2 + n^{-1/2})$），但在非高斯噪声下变得次优；提出 Agnostic Cumulant-based Estimation (ACE) 利用高阶累积量达到 $r$ 阶不敏感性 $O(\epsilon^r + n^{-1/2})$。

**[Llm Interpretability With Identifiable Temporal-Instantaneous Representation](causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)**

:   本文提出了一种面向 LLM 高维激活空间的可辨识时序因果表示学习框架，通过线性化公式同时建模时间延迟和瞬时因果关系，在保留理论可辨识性保证的同时解决了现有 CRL 方法无法扩展到 LLM 维度的计算瓶颈。

**[Performative Validity Of Recourse Explanations](causal_inference/performative_validity_of_recourse_explanations.md)**

:   本文形式化分析了追索权解释（recourse explanations）的"表演性"效应——当大量被拒申请者按照追索建议行动时，集体行为会引发数据分布偏移并使模型更新后追索失效，并证明了只有基于因果变量的改进型追索（ICR）才能在广泛条件下保持"表演性有效性"。

**[Practical Do-Shapley Explanations With Estimand-Agnostic Causal Inference](causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)**

:   提出 Estimand-Agnostic（EA）方法和 Frontier-Reducibility Algorithm（FRA）来高效计算因果 Shapley 值（do-SV），通过训练单个 SCM 学习观测分布即可回答任意可辨识的因果查询，并通过联盟约减将计算量降低约 90%。

**[Revealing Multimodal Causality With Large Language Models](causal_inference/revealing_multimodal_causality_with_large_language_models.md)**

:   提出 MLLM-CD，首个面向多模态非结构化数据的因果发现框架，通过对比因子发现识别跨模态因果变量，结合统计因果结构推断，并利用 MLLM 的世界知识生成多模态反事实样本来迭代消除结构歧义，在合成和真实数据集上均显著优于现有方法。

**[Root Cause Analysis Of Outliers With Missing Structural Knowledge](causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)**

:   提出仅用**边际异常分数**即可做根因分析的两个简单高效算法——已知因果图时用 SMOOTH TRAVERSAL（沿因果路径找分数跳变最大的节点），未知因果图时用 SCORE ORDERING（按分数排序取 top-k），在 polytree 结构下给出非参数概率保证，仅需单个异常样本即可工作。

**[Transferring Causal Effects Using Proxies](causal_inference/transferring_causal_effects_using_proxies.md)**

:   提出基于代理变量（proxy）的多域因果效应迁移方法，在目标域仅观测到代理变量 W 的条件下，利用多源域数据识别并估计目标域中含未观测混淆因子的干预分布，给出两种一致性估计器及渐近置信区间。

---

## 🎬 视频生成 { #video_generation }

**[Autoregressive Adversarial Posttraining For Realtime Interac](video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)**

:   提出 AAPT（Autoregressive Adversarial Post-Training），通过三阶段后训练流程（扩散适配→一致性蒸馏→对抗训练）将预训练的 8B 视频扩散 Transformer 转化为单步自回归实时视频生成器，利用 student-forcing 训练策略有效抑制误差累积，在单张 H100 上以 24fps/736×416 流式生成长达一分钟（1440帧）的视频，同时支持姿态控制和相机控制等实时交互应用。

**[Dismo Disentangled Motion Representations For Openworld Moti](video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)**

:   DisMo 通过双流架构（运动提取器 + 帧生成器）和图像空间重建目标，从原始视频中学习与外观、姿态、类别无关的抽象运动表征，实现跨类别/跨视角的开放世界运动迁移，并在零样本动作分类上大幅超越 V-JEPA 等视频表征模型。

**[Force Prompting Video Generation Models Can Learn And Generalize Physics-Based C](video_generation/force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)**

:   提出Force Prompting，将物理力（局部点力和全局风力）作为视频生成模型的控制信号，仅用~15K合成训练视频（Blender旗帜和滚球）和单日4xA100训练，即可在多样真实场景图像上展现跨物体/材质/几何的惊人泛化，包括初步的质量理解能力。

**[Foresight Adaptive Layer Reuse For Accelerated And Highquali](video_generation/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)**

:   提出 Foresight，一种训练无关的自适应层复用框架，通过在 warmup 阶段建立逐层 MSE 阈值、在 reuse 阶段按阈值动态决策每层是复用缓存还是重新计算，在 5 个视频生成模型上实现了比静态方法更高质量和更快速度的推理加速（最高 2.23×）。

**[Lemica Lexicographic Minimax Path Caching For Efficient Diffusion-Based Video Ge](video_generation/lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)**

:   提出 LeMiCa，一种免训练的扩散视频生成加速框架，将缓存调度建模为有向无环图上的字典序极小极大路径优化问题，通过全局误差控制实现速度和质量的双重提升（Latte 上 2.9× 加速，Open-Sora 上 LPIPS 低至 0.05）。

**[Magcache Fast Video Generation With Magnitudeaware Cache](video_generation/magcache_fast_video_generation_with_magnitudeaware_cache.md)**

:   发现视频扩散模型中相邻时间步残差输出的幅度比（magnitude ratio）遵循一条跨模型、跨 prompt 普遍成立的单调递减规律（"统一幅度定律"），由此提出 MagCache：基于幅度比对跳步误差进行精确累积建模，自适应跳过冗余时间步并复用缓存，仅需 1 个样本校准，即可在 Open-Sora、CogVideoX、Wan 2.1、HunyuanVideo 等模型上实现 2.10–2.68× 加速，且在 LPIPS/SSIM/PSNR 三个指标上全面优于 TeaCache 等已有方法。

**[Photography Perspective Composition Towards Aesthetic Perspective Recommendation](video_generation/photography_perspective_composition_towards_aesthetic_perspective_recommendation.md)**

:   提出"摄影透视构图"(PPC) 新范式，超越传统裁剪方法，通过 3D 重建构建透视变换数据集 + Image-to-Video 生成推荐视角 + RLHF 对齐人类偏好 + PQA 模型评估透视质量。

**[Physctrl Generative Physics For Controllable And Physicsgrou](video_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)**

:   PhysCtrl用扩散模型学习四种材料（弹性/沙/橡皮泥/刚体）的物理动力学分布，将动态表示为3D点轨迹，在55万合成动画上训练含时空注意力+物理约束的扩散模型，生成的轨迹驱动预训练视频模型实现力和材料参数可控的高保真物理视频生成。

**[Posecrafter Extreme Pose Estimation With Hybrid Video Synthesis](video_generation/posecrafter_extreme_pose_estimation_with_hybrid_video_synthesis.md)**

:   提出 PoseCrafter，一种无需训练的极端位姿估计框架：通过混合视频生成（HVG，DynamiCrafter+ViewCrafter双阶段）合成高保真中间帧解决极小/无重叠图像对的位姿估计，配合特征匹配选择器（FMS）高效选取最有用的中间帧，在四个数据集上显著提升极端位姿估计精度。

**[Radial Attention Onlog N Sparse Attention With Energy Decay For Long Video Gener](video_generation/radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)**

:   Radial Attention 发现了视频扩散模型中注意力分数随时空距离指数衰减的"时空能量衰减"现象，据此设计了一种 O(n log n) 复杂度的静态稀疏注意力掩码，在 HunyuanVideo/Wan2.1 等模型上实现最高 3.7× 推理加速，并通过 LoRA 微调支持 4× 更长视频生成。

**[Rlgf Reinforcement Learning With Geometric Feedback For Autonomous Driving Video](video_generation/rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video.md)**

:   本文首次系统量化自动驾驶视频生成中的几何失真问题，提出 RLGF 框架通过层次化几何奖励（消失点-车道线-深度-占用）和潜空间滑动窗口优化策略，将 3D 目标检测 mAP 提升 12.7 个绝对百分点（25.75→31.42），大幅缩小合成数据与真实数据的性能差距。

**[S2Q-Vdit Accurate Quantized Video Diffusion Transformer With Salient Data And Sp](video_generation/s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)**

:   针对视频扩散 Transformer 的超长 token 序列导致的量化校准高方差和学习困难问题，提出 S²Q-VDiT 框架，利用 Hessian 感知的显著数据选择和注意力引导的稀疏 token 蒸馏两项技术，首次在 W4A6 设置下实现无损量化，带来 3.9× 模型压缩和 1.3× 推理加速。

**[Safesora Safe Texttovideo Generation Via Graphical Watermark](video_generation/safesora_safe_texttovideo_generation_via_graphical_watermark.md)**

:   Safe-Sora 首次将**图形水印**（如logo图像）直接嵌入到视频生成管线中，通过分层粗到细自适应匹配将水印patch分配到视觉最相似的帧和区域，并设计3D小波变换增强Mamba架构实现时空融合，在视频质量（FVD 3.77 vs 次优154.35）和水印保真度上大幅超越所有基线。

**[Scaling Rl To Long Videos](video_generation/scaling_rl_to_long_videos.md)**

:   提出 LongVILA-R1 全栈框架，通过 104K 长视频推理数据集、两阶段 CoT-SFT + RL 训练管线、以及 MR-SP 多模态强化序列并行系统，将 VLM 的推理能力扩展到长视频（最高 8192 帧），在 VideoMME 上达到 65.1%/71.1%。

**[Seeing The Wind From A Falling Leaf](video_generation/seeing_the_wind_from_a_falling_leaf.md)**

:   提出端到端可微逆图形学框架，通过联合建模物体几何/物理属性、力场表示和物理过程，从视频中反向传播恢复不可见的力场（如风场），并支持基于物理的视频生成和编辑。

**[Self Forcing Bridging The Train-Test Gap In Autoregressive Video Diffusion](video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)**

:   提出 Self Forcing 训练范式，通过在训练时执行自回归自展开（self-rollout）并使用整体视频级分布匹配损失（DMD/SiD/GAN），消除了 Teacher Forcing 和 Diffusion Forcing 中训练-推理分布不匹配导致的暴露偏差问题，基于 Wan2.1-1.3B 实现了单 GPU 上 17 FPS 实时流式视频生成，同时质量匹敌甚至超越慢几十倍的双向扩散模型。

**[Stable Cinemetrics Structured Taxonomy And Evaluation For Professional Video Gen](video_generation/stable_cinemetrics_structured_taxonomy_and_evaluation_for_professional_video_gen.md)**

:   提出 SCINE（Stable Cinemetrics），首个面向专业视频制作的结构化评估框架，定义了 76 个细粒度电影控制节点的分层分类体系，配合大规模专业人员评估（80+ 影视从业者、20K+ 视频、248K 标注），揭示当前最强 T2V 模型在专业控制上的显著不足。

**[Training-Free Efficient Video Generation Via Dynamic Token Carving](video_generation/training-free_efficient_video_generation_via_dynamic_token_carving.md)**

:   提出 Jenga，一种即插即用的视频扩散 Transformer 推理加速管线，通过 3D 空间填充曲线驱动的动态注意力剪裁和渐进分辨率生成的协同设计，在多个 Video DiT 模型上实现 6-9 倍加速且几乎无质量损失。

**[Video Diffusion Models Excel At Tracking Similar-Looking Objects Without Supervi](video_generation/video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)**

:   发现预训练视频扩散模型在高噪声去噪阶段天然学到了适合追踪的运动表示，提出 TED 框架融合运动和外观特征，在追踪外观相似物体时比现有自监督方法提升多达 10 个百分点。

**[Video Killed The Energy Budget Characterizing The Latency And Power Regimes Of O](video_generation/video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)**

:   对开源文本到视频 (T2V) 模型进行系统性延迟和能耗分析，建立了基于 FLOP 的计算分析模型预测 WAN2.1 的缩放规律（空间/时间维度二次缩放、去噪步数线性缩放），并在 7 个 T2V 模型上提供跨模型能耗基准。

**[Vmdt Decoding The Trustworthiness Of Video Foundation Models](video_generation/vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)**

:   提出 VMDT（Video-Modal DecodingTrust），首个统一评估 T2V 和 V2T 视频基础模型在安全、幻觉、公平、隐私和对抗鲁棒性五个维度上可信度的基准平台，涵盖 7 个 T2V 和 19 个 V2T 模型的大规模评测，揭示了模型规模与可信度之间的复杂关系。

**[Vorta Efficient Video Diffusion Via Routing Sparse Attention](video_generation/vorta_efficient_video_diffusion_via_routing_sparse_attention.md)**

:   提出VORTA框架，通过桶化核心集注意力（建模长程依赖）和信号感知路由机制（自适应选择稀疏注意力分支），在不损失生成质量的前提下实现视频扩散Transformer端到端1.76×加速，并可与缓存和蒸馏方法叠加达到14.41×加速。

**[Vsa Faster Video Diffusion With Trainable Sparse Attention](video_generation/vsa_faster_video_diffusion_with_trainable_sparse_attention.md)**

:   提出 VSA (Video Sparse Attention)，一种端到端可训练的硬件对齐稀疏注意力机制，通过粗粒度阶段（cube 池化预测关键 token）和细粒度阶段（在预测的块稀疏区域执行 token 级注意力）的层次化设计，在视频 DiT 的训练和推理中同时实现加速：从头预训练实现 2.53× 训练 FLOPs 减少且无质量损失，适配 Wan2.1-1.3B 实现注意力 6× 加速和端到端推理从 31s 降至 18s。

---

## 🎁 推荐系统 { #recommender }

**[Asap An Agentic Solution To Auto-Optimize Performance Of Large-Scale Llm Trainin](recommender/asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)**

:   ASAP 是一个多 Agent 系统（Coordinator + Analyzer + Proposal），自动化诊断大规模 LLM 分布式训练的瓶颈类型（计算/内存/通信）并提出 sharding 配置方案，在 3 个实验场景中均匹配人类专家方案，实现最高 2.58× 吞吐量提升。

**[Balancing Performance And Costs In Best Arm Identification](recommender/balancing_performance_and_costs_in_best_arm_identification.md)**

:   提出将最优臂识别（BAI）从固定预算/固定置信度框架重新定义为"误识别概率/简单遗憾 + 采样成本"的风险泛函最小化问题，推导出含相变现象的下界（差距过小时最优策略是直接猜），设计 DBCARE 算法在动态预算下达到对数因子内最优。

**[Empathia Multi-Faceted Human-Ai Collaboration For Refugee Integration](recommender/empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)**

:   提出EMPATHIA多Agent框架，基于Kegan建构性发展理论，通过情感/文化/伦理三个专业化Agent的选择器-验证器协商评估难民安置建议，在6,359名难民的真实数据上达到87.4%收敛率和92.1%文化专家同意率。

**[Estimating Hitting Times Locally At Scale](recommender/estimating_hitting_times_locally_at_scale.md)**

:   提出两种局部（亚线性）算法估计图上的命中时间——基于相遇时间的 Algorithm 1 和基于谱截断的 Algorithm 3，无需全图访问仅通过以 $u,v$ 为中心的短随机游走完成估计，在合成和真实图上相对误差 <1.4%，并证明了游走采样的最优样本复杂度下界。

**[Inference-Time Reward Hacking In Large Language Models](recommender/inference-time_reward_hacking_in_large_language_models.md)**

:   本文从数学上证明了推理时对齐方法（如 BoN）在优化代理奖励时不可避免地会出现 reward hacking（真实奖励先升后降），提出了 Best-of-Poisson (BoP) 采样方法近似最优 KL-奖励折中分布，并设计了 HedgeTune 算法通过一维寻根找到最优推理时参数，在数学推理和人类偏好场景中有效缓解 reward hacking。

**[Measuring What Matters Construct Validity In Large Language Model Benchmarks](recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)**

:   本文由29位专家对445篇LLM benchmark论文进行系统性综述，从构念效度 (construct validity) 角度审视现有LLM评测基准在现象定义、任务设计、评分指标和结论声明方面的不足，并提出8条改进建议。

**[Mmpb Its Time For Multi-Modal Personalization](recommender/mmpb_its_time_for_multi-modal_personalization.md)**

:   提出首个 VLM 个性化评测基准 MMPB，包含 111 个可个性化概念、10k+ 图文问答对和 15 种任务类型，评测了 23 个 VLM 后发现即使最强的 GPT-4o 在个性化任务上也表现不佳，揭示了 VLM 在偏好推理、视觉线索利用和安全对齐与个性化的冲突等方面的重大局限。

**[Neurips Should Lead Scientific Consensus On Ai Policy](recommender/neurips_should_lead_scientific_consensus_on_ai_policy.md)**

:   本文是一篇立场论文，主张 NeurIPS 应主动承担 AI 政策领域的科学共识形成角色，借鉴 IPCC（政府间气候变化专门委员会）在气候科学中的成功经验，填补当前 AI 政策领域共识机制的空白。

**[Overcoming Sparsity Artifacts In Crosscoders To Interpret Chat-Tuning](recommender/overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)**

:   识别 Crosscoder 中 L1 损失引入的两类稀疏性伪影（Complete Shrinkage 将弱共享概念错误归零、Latent Decoupling 将共享概念拆分为虚假模型特定潜变量），提出 Latent Scaling 诊断方法和 BatchTopK Crosscoder 替代方案，显著提升 chat-tuning 概念发现的可靠性。

**[Pac-Bayes Bounds For Multivariate Linear Regression And Linear Autoencoders](recommender/pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders.md)**

:   本文将PAC-Bayes泛化界从单输出线性回归推广到**多变量线性回归**，并进一步适配到推荐系统中的**线性自编码器（LAE）**，通过理论方法将计算复杂度从O(n⁴)降到O(n³)，实验证明该界是紧的且与Recall@K、NDCG@K等实际指标高度相关。

**[Position Towards Bidirectional Human-Ai Alignment](recommender/position_towards_bidirectional_human-ai_alignment.md)**

:   本文提出**双向人机对齐（Bidirectional Human-AI Alignment）**框架，从系统综述400+篇论文出发，论证AI对齐不应仅是单向地"让AI对齐人类"，还应包括"让人类适应AI"这一被严重忽视的方向，并识别了当前研究的关键缺口。

**[R2Ec Towards Large Recommender Models With Reasoning](recommender/r2ec_towards_large_recommender_models_with_reasoning.md)**

:   提出R²ec，首个将推理能力内生地集成到推荐模型中的统一大推荐模型，通过双头架构实现推理链生成与高效物品预测的一体化，并设计RecPO强化学习框架在无推理标注数据下联合优化推理与推荐目标。

**[Radial Neighborhood Smoothing Recommender System](recommender/radial_neighborhood_smoothing_recommender_system.md)**

:   提出 Radial Neighborhood Estimator (RNE)，通过将隐空间距离用观测矩阵的行/列 L2 范数近似估计，构建同时包含重叠和部分重叠用户-物品对的径向邻域，用局部核回归做平滑插补，在理论保证和实验中均优于传统协同过滤和矩阵分解方法，并天然缓解冷启动问题。

**[The Coming Crisis Of Multi-Agent Misalignment Ai Alignment Must Be A Dynamic And](recommender/the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and.md)**

:   立场论文，主张多智能体系统（MAS）中的 AI 对齐应被视为动态的、依赖交互的社会过程，而非孤立问题；借鉴社会科学理论分析了社会结构如何破坏群体和个体价值，并呼吁 AI 社区建立专门的仿真环境、基准测试和评估框架来应对这一挑战。

**[The More You Automate The Less You See Hidden Pitfalls Of Ai Scientist Systems](recommender/the_more_you_automate_the_less_you_see_hidden_pitfalls_of_ai_scientist_systems.md)**

:   本文系统性地识别了当前 AI 科学家系统的四种方法论陷阱（不当基准选择、数据泄漏、指标误用、事后选择偏差），通过精心设计的合成任务 SPR 对 Agent Laboratory 和 The AI Scientist v2 进行受控实验，发现两个系统均存在不同程度的问题，并证明审计 trace log + 代码比仅审查最终论文的检测准确率高 27 个百分点（82% vs 55%）。

**[Think Before Recommendation Autonomous Reasoning-Enhanced Recommender](recommender/think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)**

:   提出 RecZero（纯 RL 范式）和 RecOne（SFT+RL 混合范式），抛弃传统的 teacher-student 蒸馏方法，用 GRPO 强化学习直接训练单个 LLM 自主发展推理能力进行评分预测，通过结构化 "Think-before-Recommendation" 模板引导分步推理（分析用户→分析物品→匹配→评分），在 4 个数据集上显著超越现有基线。

**[Transformer Copilot Learning From The Mistake Log In Llm Fine-Tuning](recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)**

:   提出 Transformer Copilot 框架，在 LLM 微调过程中系统记录"错误日志"(Mistake Log)，训练一个辅助 Copilot 模型学习 Pilot 的错误模式，推理时通过 logits 修正提升生成质量，在 12 个基准上最高提升 34.5%。

**[Tv-Rec Time-Variant Convolutional Filter For Sequential Recommendation](recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)**

:   提出 TV-Rec，基于图信号处理的时变卷积滤波器替代传统固定卷积和自注意力机制，实现更高表达力的序列推荐，在 6 个基准数据集上平均提升 7.49%。

**[Validating Llm-As-A-Judge Systems Under Rating Indeterminacy](recommender/validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)**

:   提出在评分不确定性 (rating indeterminacy) 条件下验证 LLM-as-a-Judge 系统的框架，通过 "response set" 多标签评分方案替代强制选择评分，使选出的 judge 系统性能提升高达 31%。

**[Visuallens Personalization Through Task-Agnostic Visual History](recommender/visuallens_personalization_through_task-agnostic_visual_history.md)**

:   提出VisualLens框架，利用用户日常拍摄的与任务无关的视觉历史(task-agnostic visual history)，通过频谱用户画像(spectrum user profile)和多模态大模型实现跨领域个性化推荐，在新建的Google Review-V和Yelp-V数据集上Hit@3超越GPT-4o 2-5%。

**[Who You Are Matters Bridging Topics And Social Roles Via Llm-Enhanced Logical Re](recommender/who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)**

:   提出 TagCF 框架，通过 MLLM 提取用户角色标签和物品话题标签，再用 LLM 推理构建 U2I/I2U 逻辑图（用户角色与物品类型的因果关联），辅以标签编码器、对比学习增强和逻辑推理评分三种集成策略增强推荐，在亿级用户的工业在线A/B测试中互动指标提升0.946%、多样性提升0.102%，离线实验NDCG@10提升8.06%。

**[Wide-Horizon Thinking And Simulation-Based Evaluation For Real-World Llm Plannin](recommender/wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)**

:   提出 MAoP（Multiple Aspects of Planning）框架赋予 LLM "宽视野思维"能力，通过策略师预规划与路由机制并行整合多方面约束，配合 Travel-Sim 因果模拟评估基准，在旅行规划任务上大幅超越 CoT/分解方法，蒸馏后 3B 模型 PER 达 66.9%。

---

## 🧮 科学计算 { #scientific_computing }

**[Bayesian Surrogates For Risk-Aware Pre-Assessment Of Aging Bridge Portfolios](scientific_computing/bayesian_surrogates_for_risk-aware_pre-assessment_of_aging_bridge_portfolios.md)**

:   提出基于贝叶斯神经网络（BNN）的代理模型，用于替代昂贵的非线性有限元分析（NLFEA），实现对老化桥梁组合的快速、不确定性感知的结构安全预评估，在真实铁路案例中为单座桥梁节省约37万美元。

**[Collapsing Taylor Mode Automatic Differentiation](scientific_computing/collapsing_taylor_mode_automatic_differentiation.md)**

:   提出 Taylor mode 自动微分的"折叠"(collapsing)优化技术，通过重写计算图将导数求和操作向上传播，大幅加速 PDE 算子（如 Laplacian、一般线性 PDE 算子）的计算，实现速度优于嵌套反向传播同时保持前向模式的低内存优势。

**[Deltaphi Physical States Residual Learning For Neural Operators In Data-Limited ](scientific_computing/deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)**

:   提出 DeltaPhi 框架：不直接学习 PDE 的输入→输出映射，而是学习**相似物理状态之间的残差**，利用物理系统稳定性实现隐式数据增强，在数据稀缺场景下显著提升各类神经算子的性能。

**[Eddyformer Accelerated Neural Simulations Of Three-Dimensional Turbulence At Sca](scientific_computing/eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)**

:   提出 EddyFormer，一种基于谱元法 (SEM) 的 Transformer 架构，将流场分解为 LES（大尺度）和 SGS（小尺度）两路并行流，在 256³ 分辨率 3D 湍流上达到 DNS 级精度且加速 30 倍，并在未见的 4× 更大域上泛化良好。

**[Enforcing Governing Equation Constraints In Neural Pde Solvers Via Training-Free](scientific_computing/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)**

:   提出两种无需训练的后处理投影方法（非线性LBFGS优化和局部线性化投影），将神经PDE求解器的输出投影到满足控制方程约束的可行流形上，在Lorenz/KS/Navier-Stokes上大幅降低约束违反并提升精度，且效果显著优于physics-informed训练。

**[F-Adapter Frequency-Adaptive Parameter-Efficient Fine-Tuning In Scientific Machi](scientific_computing/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)**

:   本文首次系统研究了科学机器学习中预训练大型算子模型(LOM)的参数高效微调(PEFT)，发现 LoRA 在傅里叶层中存在深度放大的近似误差下界，而 Adapter 保留了通用逼近能力；据此提出频率自适应 Adapter（F-Adapter），按频谱能量分配 Adapter 容量，在 3D Navier-Stokes 预测任务上仅调参不到 2% 即达到 SOTA。

**[From Black Hole To Galaxy Neural Operator Framework For Accretion And Feedback D](scientific_computing/from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)**

:   提出基于 Neural Operator 的「子网格黑洞」模型，学习小尺度 (GR)MHD 时间演化算子，替代手工闭合规则嵌入多层级直接数值模拟框架，首次实现吸积驱动反馈的内禀变异性捕获，加速比达 $\sim 10^5$ 倍。

**[From Images To Physics Probabilistic Inference Of Galaxy Parameters And Emission](scientific_computing/from_images_to_physics_probabilistic_inference_of_galaxy_parameters_and_emission.md)**

:   提出 VAE–Normalizing Flow 混合框架，从 SDSS gri 图像和测光数据出发，以概率方式联合推断星系物理参数（恒星质量、SFR、红移、气相金属丰度、中心黑洞质量）和发射线流量（Hα、Hβ、[N II]、[O III]），速度比 SED 拟合快 100 倍以上且提供校准良好的后验分布。

**[Gyroswin 5D Surrogates For Gyrokinetic Plasma Turbulence Simulations](scientific_computing/gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)**

:   首次提出可扩展的5D神经网络代理模型 GyroSwin，将 Swin Transformer 扩展至5维回旋动力学相空间，通过交叉注意力实现3D↔5D交互、通道式模态分离捕获带状流，在等离子体湍流模拟中实现比传统准线性方法更高的精度，且比数值求解器（GKW）快3个数量级。

**[Hamiltonian Neural Pde Solvers Through Functional Approximation](scientific_computing/hamiltonian_neural_pde_solvers_through_functional_approximation.md)**

:   基于 Riesz 表示定理，用可学习核积分（Integral Kernel Functional）近似无限维 Hamiltonian 泛函，通过自动微分获取泛函导数，实现保能量的神经 PDE 求解器（HNS），在 1D/2D PDE 上展现出优越的稳定性和泛化能力。

**[Inc An Indirect Neural Corrector For Auto-Regressive Hybrid Pde Solvers](scientific_computing/inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)**

:   提出间接神经校正器(INC)，将学习到的校正项嵌入PDE的右端项（而非直接修改状态），理论证明误差放大降低$\mathcal{O}(\Delta t^{-1}+L)$倍，在6个PDE系统（1D混沌到3D湍流）上大幅改善长期轨迹性能（R²提升达158.7%），实现最高330×加速。

**[Integration Matters For Learning Pdes With Backward Sdes](scientific_computing/integration_matters_for_learning_pdes_with_backward_sdes.md)**

:   揭示了标准 BSDE 方法性能不如 PINNs 的根本原因是 Euler-Maruyama 积分引入的不可消除离散化偏差，提出基于 Stratonovich 形式的 Heun-BSDE 方法彻底消除该偏差，在高维 PDE 上与 PINNs 竞争。

**[Multi-Trajectory Physics-Informed Neural Networks For Hjb Equations With Hard-Ze](scientific_computing/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)**

:   针对最优交易执行中 HJB 方程的硬零终端库存约束（$X_T=0$），提出 Multi-Trajectory PINN (MT-PINN)，通过基于轨迹展开的终端损失与 $\lambda$-curriculum 训练策略，在合成数据和 SPY 实盘回测中显著优于 vanilla PINN，终端库存违规率大幅降低。

**[Neural Emulator Superiority When Machine Learning For Pdes Surpasses Its Trainin](scientific_computing/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)**

:   挑战了"神经 PDE 模拟器精度受限于训练数据（数值求解器）精度"的传统认知，发现并严格定义了 **emulator superiority** 现象——仅在低精度求解器数据上训练的神经网络，在以高精度参考解评估时竟能超越其训练求解器本身。

**[Neuro-Spectral Architectures For Causal Physics-Informed Networks](scientific_computing/neuro-spectral_architectures_for_causal_physics-informed_networks.md)**

:   NeuSA 将经典谱方法与 Neural ODE 结合，先将 PDE 投影到谱基（Fourier）上得到 ODE 系统，再用 NODE 学习动力学演化，从架构层面解决了传统 PINN 的谱偏差和因果性问题，在波动方程/Burgers方程/sine-Gordon方程上误差比 baseline 低 1-2 个数量级且训练更快。

**[One-Shot Transfer Learning For Nonlinear Pdes With Perturbative Pinns](scientific_computing/one-shot_transfer_learning_for_nonlinear_pdes_with_perturbative_pinns.md)**

:   提出 VAE–Normalizing Flow 两阶段概率推断框架，仅从 SDSS 星系图像和测光数据即可快速推断恒星质量、SFR、红移、黑洞质量、金属丰度及发射线通量，精度超越现有非光谱方法且比 SED 拟合快 100 倍以上。

**[Physics-Guided Machine Learning For Uncertainty Quantification In Turbulence Mod](scientific_computing/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)**

:   提出混合 ML–EPM 框架：用轻量 CNN 学习从 RANS 湍流动能场到 DNS 真值的修正映射，以此调制特征空间扰动法（EPM）的扰动幅度，在保持物理一致性的前提下将湍流模型不确定性估计的误差降低 1–2 个数量级。

**[Physics-Informed Neural Networks With Fourier Features And Attention-Driven Deco](scientific_computing/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)**

:   提出 Spectral PINNsformer (S-Pformer)，用 Fourier 特征嵌入替换 PINNsformer 的编码器，结合仅解码器 Transformer 架构，在减少 18.6% 参数量的同时在多个 PDE benchmark 上取得更优性能，有效缓解了频谱偏置问题。

**[Stable Minima Of Relu Neural Networks Suffer From The Curse Of Dimensionality Th](scientific_computing/stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)**

:   本文研究了两层过参数化 ReLU 网络中稳定极小值（flat minima）的泛化性质，证明虽然平坦性确实蕴含泛化，但其收敛速率随输入维度指数级恶化（即存在维度灾难），与不受维度灾难影响的低范数解（weight decay）形成指数级分离；并揭示了"neural shattering"现象作为高维失败的几何机制。

**[Symbolic Regression Is All You Need From Simulations To Scaling Laws In Binary N](scientific_computing/symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)**

:   利用符号回归（Symbolic Regression）从数值相对论模拟数据中自动发现双中子星并合后吸积盘质量的解析标定关系，所得紧凑表达式在预测精度、泛化能力和可解释性上全面超越文献中已有的经验拟合公式。

**[The Primacy Of Magnitude In Low-Rank Adaptation](scientific_computing/the_primacy_of_magnitude_in_low-rank_adaptation.md)**

:   揭示 LoRA 中权重更新幅度（magnitude）是性能的根本驱动因素，统一了学习率、缩放因子和初始化策略对 LoRA 的影响机制，并提出 LoRAM——一种基于确定性正交基和幅度缩放的高效初始化方法，无需 SVD 即可匹敌甚至超越谱初始化方法。

**[Towards Universal Neural Operators Through Multiphysics Pretraining](scientific_computing/towards_universal_neural_operators_through_multiphysics_pretraining.md)**

:   提出基于 adapter 的多物理场预训练框架，通过将 lifting/projection 层作为问题特定适配器、冻结共享的核积分算子层，实现跨 PDE 问题的迁移学习，显著降低微调成本并提升泛化能力。

---

## 💻 代码智能 { #code_intelligence }

**[A Selfimproving Coding Agent](code_intelligence/a_selfimproving_coding_agent.md)**

:   提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

**[A Stochastic Differential Equation Framework For Multi-Objective Llm Interaction](code_intelligence/a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)**

:   将 LLM 迭代交互中的多目标优化建模为 SDE（漂移-扩散过程），通过干扰矩阵量化目标间的耦合模式，通过特征值谱分析策略收敛行为，在代码生成（安全性、效率、功能性三目标）上验证了不同策略的收敛率（0.33-1.29）和可预测性（$R^2$ 达 0.74）。

**[Astrovisbench A Code Benchmark For Scientific Computing And Visualization In Ast](code_intelligence/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)**

:   AstroVisBench 构建了首个评估 LLM 天文科学计算和可视化能力的代码基准——从 110 个 Jupyter Notebook 提取 864 个任务（处理+可视化），设计双重评估管线（执行式变量检查 + VLM-as-Judge 可视化评分，与专家 Spearman ρ=0.822），评测 8 个 SOTA 模型后发现 Gemini 2.5 Pro 最佳但无错误率仅 15.7%，FileNotFoundError 占 43% 错误。

**[Automated Multi-Agent Workflows For Rtl Design](code_intelligence/automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 提出自动组合多 Agent 工作流的框架用于 RTL 代码生成，核心创新是将 HDL 工具的形式化验证反馈（Yosys 综合 + OpenSTA 时序分析）直接整合到工作流编排中，在 VeriThoughts 上 pass@1 提升 2-12%，且仅需数百样本做控制器调优，比全量微调训练数据少一个量级。

**[Co-Evolving Llm Coder And Unit Tester Via Reinforcement Learning](code_intelligence/co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)**

:   提出 CURE 框架，让同一个 LLM 同时扮演代码生成器和单元测试生成器两个角色，通过生成代码与生成测试的交叉执行构建成对奖励矩阵，用基于理论推导的奖励信号进行强化学习，在完全不需要 ground-truth 代码标注的情况下实现代码生成能力和单元测试生成能力的共同进化，在五个编程基准上大幅超过同规模的专用 Coder 模型。

**[Core Benchmarking Llms Code Reasoning Capabilities Through Static Analysis Tasks](code_intelligence/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)**

:   提出 CoRe，一个包含 12,553 个人工验证任务实例的高质量 benchmark，通过数据依赖、控制依赖和信息流三类静态分析基础任务，直接评估 LLM 的代码语义推理能力，揭示模型在 trace 生成和源枚举等需要多步推理的任务上仍严重不足。

**[Embedding Alignment In Code Generation For Audio](code_intelligence/embedding_alignment_in_code_generation_for_audio.md)**

:   提出双 MLP + InfoNCE 对比学习框架，将代码嵌入（distilroberta-base）和音频嵌入（wav2vec2）对齐到共享空间，使 LLM 代码生成流程无需编译执行即可从代码推断音乐相似性，CKA 从 0.090 提升至 0.590。

**[Flylora Boosting Task Decoupling And Parameter Efficiency Via Implicit Rank-Wise](code_intelligence/flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)**

:   FlyLoRA 受飞蝇嗅觉回路启发，将 LoRA 的下投影矩阵 $A$ 替换为冻结的稀疏随机投影，通过 top-$k$ 激活值选择实现隐式 rank-wise MoE 路由，在消除路由参数的同时减少任务内干扰，并利用随机投影的近正交性天然支持多任务模型合并。

**[Fractalbench Diagnosing Visual-Mathematical Reasoning Through Recursive Program ](code_intelligence/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)**

:   提出 FractalBench，一个通过分形图像程序合成诊断 MLLM 视觉-数学推理能力的 benchmark：12 种经典分形、610 张测试图、4 个 MLLM，揭示 76% 的代码能执行但仅 4% 视觉正确，暴露了模型在递归抽象能力上的根本缺陷。

**[Learning To Solve Complex Problems Via Dataset Decomposition](code_intelligence/learning_to_solve_complex_problems_via_dataset_decomposition.md)**

:   提出Decomp方法，利用教师模型将复杂数学题按推理步骤递归分解为更简单的子问题，构建概念依赖图量化难度，再按从易到难的课程顺序训练学生模型——Qwen2.5-1.5B在MATH-500上达51.6%（超MuggleMath用147K数据的50.4%），Qwen3-4B在AIME2025仅用385样本达16.7%（超Qwen2.5-72B的15%）。

**[Maintaincoder Maintainable Code Generation Under Dynamic Requirements](code_intelligence/maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)**

:   首次系统定义并解决 LLM 代码生成的**可维护性**问题，同时贡献基准和方法：MaintainBench 通过 4 种需求变化模式 + 动态指标评测代码在需求演化下的可维护性；MaintainCoder 将 Waterfall 模型、设计模式与 6 个专业化 Agent 结合，动态可维护性指标提升 60%+，且初始代码正确性也一并提高。

**[Mlr-Bench Evaluating Ai Agents On Open-Ended Machine Learning Research](code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)**

:   提出 MLR-Bench，一个包含 201 个开放式 ML 研究任务的综合基准，配套 MLR-Judge（LLM 评审框架）和 MLR-Agent（模块化研究代理），发现当前最先进的编码代理在约 80% 的情况下会生成伪造或未验证的实验结果，揭示了 AI 自动化科学研究的核心瓶颈。

**[Once Upon An Input Reasoning Via Per-Instance Program Synthesis](code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)**

:   提出 PIPS（Per-Instance Program Synthesis），通过实例级别的程序合成与结构化反馈迭代改进，结合置信度度量动态选择直接推理或程序合成，在30个基准上将调和平均准确率提升8.6%。

**[Preserving Llm Capabilities Through Calibration Data Curation From Analysis To O](code_intelligence/preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)**

:   系统研究了校准数据的组成特性（序列长度/样本量/来源/格式）和领域对应关系对LLM压缩后能力保持的影响，发现激活空间中的代表性和多样性是数据质量的本质决定因素，并据此提出三阶段校准数据策展框架COLA。

**[Principled Fine-Tuning Of Llms From User-Edits A Medley Of Preference Supervisio](code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)**

:   系统研究如何利用用户编辑数据微调 LLM，将偏好、监督标签和代价三种反馈类型统一起来，并提出一种简单的集成方法，在不同用户分布下实现鲁棒适应。

**[Program Synthesis Via Test-Time Transduction](code_intelligence/program_synthesis_via_test-time_transduction.md)**

:   提出 SYNTRA 框架，将程序合成重新定义为转导式学习——在测试时利用可见的 test inputs 和 LLM 的判断来迭代消除不一致的候选程序假设，通过 greedy maximin 算法最小化 LLM 查询次数，在 4 个 benchmark 上准确率提升最高达 196%。

**[Qimeng-Salv Signal-Aware Learning For Verilog Code Generation](code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)**

:   提出信号级感知学习方法 QiMeng-SALV，通过从部分错误的 Verilog 模块中提取信号级功能正确的代码片段作为 DPO 训练的奖励信号，将优化粒度从模块级提升到信号级，在 VerilogEval 和 RTLLM 上达到 SOTA。

**[Swe-Rebench An Automated Pipeline For Task Collection And Decontaminated Evaluat](code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)**

:   构建全自动化流水线从 GitHub 持续挖掘真实软件工程交互任务，生成 21,000+ 可执行 Python 任务的 SWE-rebench 数据集和去污染 benchmark，揭示部分模型在 SWE-bench Verified 上的性能存在污染膨胀问题（如 DeepSeek-V3 在 SWE-bench 上 39.7% vs SWE-rebench 上 21.3%）。

**[Table2Latex-Rl High-Fidelity Latex Code Generation From Table Images Via Reinfor](code_intelligence/table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)**

:   提出VSGRPO——基于GRPO的双奖励强化学习策略，联合优化结构级奖励（TEDS-Structure）和视觉保真度奖励（CW-SSIM渲染图比较），使微调后的MLLM（仅3B参数）在表格图像到LaTeX代码生成任务上超越GPT-4o和72B+规模模型，尤其在复杂表格上提升显著。

**[Text-To-Code Generation For Modular Building Layouts In Building Information Mod](code_intelligence/text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)**

:   提出 Text2MBL 框架，将自然语言描述转化为可执行的 BIM 代码（而非坐标序列），通过面向对象的代码架构和 LLM 微调实现模块化建筑布局的自动生成，在几何一致性上比坐标驱动方法提升 10%+ IoU。

---

## ⚛️ 物理学 { #physics }

**[Astroco Self-Supervised Conformer-Style Transformers For Light-Curve Embeddings](physics/astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)**

:   提出 AstroCo，一种将 Conformer（注意力 + 深度可分离卷积 + 门控）引入天文不规则光变曲线的自监督编码器，在 MACHO 数据集上重建误差比 Astromer v1/v2 降低 61-70%，少样本分类 macro-F1 提升约 7%。

**[Exoplanet Formation Inference Using Conditional Invertible Neural Networks](physics/exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)**

:   用条件可逆神经网络（cINN）训练于15,777颗合成行星数据，从观测量（行星质量、轨道距离）快速推断行星形成参数（盘质量、湍流α、尘气比），实现比物理模型快~10⁶倍的概率性参数回溯，并证明多行星系统数据比单行星数据更鲁棒。

**[Fair Universe Higgsml Uncertainty Dataset And Competition](physics/fair_universe_higgsml_uncertainty_dataset_and_competition.md)**

:   提供2.8亿模拟LHC碰撞事件的标准化数据集和竞赛平台，包含6种参数化系统偏差（探测器校准+背景成分）及不对称覆盖惩罚评估指标，要求参赛者为Higgs信号强度$\mu$估计鲁棒的68.27%置信区间，优胜方案通过无聚焦替代建模实现比传统binned方法窄约20%的置信区间。

**[Feat Free Energy Estimators With Adaptive Transport](physics/feat_free_energy_estimators_with_adaptive_transport.md)**

:   提出 FEAT 框架，利用随机插值学习两个热力学系统之间的传输映射，基于 escorted Jarzynski 等式和 controlled Crooks 定理提供一致、最小方差的自由能差估计器及变分上下界，统一了平衡与非平衡方法。

**[From Simulations To Surveys Domain Adaptation For Galaxy Observations](physics/from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)**

:   构建从模拟星系（TNG50）到真实巡天观测（SDSS）的域适应 pipeline，通过特征级对齐（欧几里得距离 + 最优传输 + top-$k$ 软匹配损失）和可训练权重调度，将星系形态分类的目标域准确率从 46.8%（无适应）提升到 87.3%，Macro F1 从 0.298 提升到 0.626。

**[Knowledge Is Overrated A Zero-Knowledge Machine Learning And Cryptographic Hashi](physics/knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)**

:   提出PHAZE框架，利用密码学哈希（Rabin指纹）和零知识机器学习（zkML）实现LHC触发器级别的可验证早退出推理，理论延迟降至~152-253ns量级，同时内建异常检测能力。

**[Latent Representation Learning In Heavy-Ion Collisions With Maskpoint Transforme](physics/latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)**

:   将掩码点云 Transformer 自编码器引入重离子碰撞分析，通过自监督预训练+监督微调的两阶段范式，学习到比 PointNet 更强的非线性潜在表征（PC1 分布重叠从 2.42% 降至 0.27%），为 QGP 性质研究提供了通用特征学习框架。

**[Multi-Modal Masked Autoencoders For Learning Image-Spectrum Associations For Gal](physics/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   将多模态掩码自编码器（MMAE）应用于星系图像（HSC-PDR2五波段）和光谱（DESI-DR1）的联合建模，构建134,533个星系的跨模态数据集GalaxiesML-Spectra，在75%掩码率下重建光谱主要发射线和图像形态，在光谱完全缺失时仅用图像实现 $\sigma_{\text{NMAD}}=0.016$ 的红移预测，优于AstroCLIP且红移范围首次扩展到 $z \sim 4$。

**[Neural Deprojection Of Galaxy Stellar Mass Profiles](physics/neural_deprojection_of_galaxy_stellar_mass_profiles.md)**

:   提出一种神经网络方法，将 Nuker 星系轮廓参数映射为可解析反投影的 Multi Gaussian Expansion (MGE) 分量，从而在无需光学成像的情况下实现星系恒星质量建模，并集成到可微分动力学建模管道 SuperMAGE 中，对超大质量黑洞 (SMBH) 质量进行贝叶斯推断。

**[Polaris A High-Contrast Polarimetric Imaging Benchmark Dataset For Exoplanetary ](physics/polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)**

:   构建首个系外行星偏振成像ML基准数据集POLARIS（921张VLT/SPHERE/IRDIS偏振图像+75,910张预处理曝光），提出Diff-SimCLR框架（扩散模型增强对比学习），在参考星vs目标星分类任务上达到93%准确率，仅需<10%手动标注。

**[Quantum Doubly Stochastic Transformers](physics/quantum_doubly_stochastic_transformers.md)**

:   提出QDSFormer（量子双随机Transformer），用变分量子电路QontOT替代softmax生成双随机注意力矩阵，理论和实验证明量子电路生成的DSM更多样、更好保持信息，在多个小规模视觉识别任务上一致超越标准ViT和Sinkformer。

**[Simulation-Based Inference For Neutrino Interaction Model Parameter Tuning](physics/simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)**

:   首次将基于仿真的推断（SBI）应用于中微子相互作用模型参数调优，使用神经后验估计（NPE）从200K个GENIE模拟的58-bin直方图中学习4个物理参数的后验分布，在MicroBooNE Tune的mock数据上准确恢复了真实参数值。

**[The Pareto Frontier Of Resilient Jet Tagging](physics/the_pareto_frontier_of_resilient_jet_tagging.md)**

:   系统评估LHC射流标记任务中多种架构（DNN/PFN/EFN/ParT）的AUC-鲁棒性权衡，揭示更复杂模型虽AUC更高但对蒙特卡洛模型依赖性更强，构建Pareto前沿并通过案例研究证明低鲁棒性分类器即使校准后仍在下游参数估计中产生偏差。

**[The Platonic Universe Do Foundation Models See The Same Sky](physics/the_platonic_universe_do_foundation_models_see_the_same_sky.md)**

:   在天文学场景下验证柏拉图表征假说（PRH）：使用JWST、HSC、Legacy Survey和DESI光谱数据，测量6种基础模型（ViT/ConvNeXt/DINOv2/IJEPA/AstroPT/Specformer）的表征对齐度，发现模态内和跨模态MKNN分数随模型规模一致增加（p=3.31×10⁻⁵），支持不同架构和模态向共享表征收敛的假说。

**[Titan A Trajectory-Informed Technique For Adaptive Parameter Freezing In Large-S](physics/titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)**

:   提出TITAN框架，用深度学习模型预测VQE中的"冻结参数"（训练过程中始终不活跃的参数），在初始化阶段即冻结40-60%参数，实现最高3倍收敛加速和40-60%电路评估量减少，在30量子比特的分子系统上匹配或超越基线精度。

**[Toward Complete Merger Identification At Cosmic Noon With Deep Learning](physics/toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)**

:   在 IllustrisTNG50 模拟生成的模拟 HST CANDELS 图像上训练 ResNet18，首次证明深度学习可以在高红移 $1<z<1.5$ 下成功识别包括小质量比合并（minor merger, $\mu \geq 1/10$）和低质量星系（$M_\star > 10^8 M_\odot$）在内的星系合并，总体准确率约 73%，并通过 Grad-CAM 和 UMAP 深入分析了模型行为。

**[Transfer Learning Beyond The Standard Model](physics/transfer_learning_beyond_the_standard_model.md)**

:   研究从标准宇宙学模型（ΛCDM）预训练的神经网络能否迁移到超越标准模型的场景（大质量中微子、修改引力、原初非高斯性），发现dummy node架构可将模拟需求降低一个数量级，但当参数存在强物理简并（如σ₈-Mν）时会出现负迁移。

**[Unsupervised Discovery Of High-Redshift Galaxy Populations With Variational Auto](physics/unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)**

:   用变分自编码器(VAE)对 2743 条 JWST 高红移($z>4$)星系光谱进行无监督聚类，发现 12 个不同的天体物理类别，使已知的后星暴星系、Lyman-α 发射星系、极端发射线星系、Little Red Dots 等稀有种群数量翻倍。

---

## 👥 社会计算 { #social_computing }

**[A Multitask Benchmark For Abusive Language Detection In Lowr](social_computing/a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)**

:   提出 TiALD（Tigrinya Abusive Language Detection），首个面向 Tigrinya 低资源语言的大规模多任务基准数据集，包含 13,717 条 YouTube 评论的辱骂/情感/主题三任务联合标注，同时发现小型微调模型（TiRoBERTa, 125M）在所有任务上全面超越 GPT-4o 和 Claude Sonnet 3.7 等前沿 LLM。

**[Active Slice Discovery In Large Language Models](social_computing/active_slice_discovery_in_large_language_models.md)**

:   提出 **Active Slice Discovery** 问题框架，将主动学习引入 LLM 错误切片发现，利用不确定性采样 + LLM 内部表征（原始 embedding 或 SAE 特征）在仅使用 2-10% 标注的情况下达到接近全标注的切片检测精度。

**[Auto-Search And Refinement An Automated Framework For Gender Bias Mitigation In ](social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)**

:   提出 FaIRMaker 框架，通过"自动搜索+精化"范式先用梯度优化找到去偏见触发词（Fairwords），再训练 seq2seq 模型将其转化为可读指令，在开源和闭源 LLM 上有效缓解性别偏见同时保持甚至提升任务性能。

**[Averimatec A Dataset For Automatic Verification Of Image-Text Claims With Eviden](social_computing/averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)**

:   AVerImaTeC 构建了首个带完整证据标注的图文事实核查数据集——1297 条真实图文声明 + 5 阶段标注流水线（提取→QA 推理→充分性检查→迭代精炼→二次检查）+ 时间约束证据（防止时间泄露），基线系统在有 ground truth 证据时准确率 82%，但自动检索证据后降至 15-25%，揭示了图文核查的巨大挑战。

**[Concept-Level Explainability For Auditing Steering Llm Responses](social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)**

:   提出 ConceptX，一种基于概念级（而非 token 级）Shapley 归因的 LLM 可解释性方法，通过语义相似度而非 token 重合度来衡量输入概念对输出的影响，可用于审计偏见和通过 prompt 编辑引导 LLM 输出，在越狱防御中将攻击成功率从 0.463 降至 0.242。

**[Date-Lm Benchmarking Data Attribution Evaluation For Large Language Models](social_computing/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)**

:   DATE-LM构建了首个面向LLM的统一数据归因评估基准，通过训练数据选择、毒性过滤和事实归因三大应用驱动任务系统评估多种归因方法，发现无单一方法全面占优且简单基线在某些场景可媲美归因方法。

**[Deeptraverse A Depth-First Search Inspired Network For Algorithmic Visual Unders](social_computing/deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)**

:   受深度优先搜索（DFS）算法启发，设计了 DeepTraverse 视觉骨干网络，通过参数共享的递归探索模块和自适应通道校准模块，在极少参数下实现高竞争力的图像分类性能。

**[Dont Let It Fade Preserving Edits In Diffusion Language Mode](social_computing/dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)**

:   提出 Token Timestep Allocation (TTA-Diffusion)，通过为每个 token 分配独立的去噪时间步来解决扩散语言模型中 classifier guidance 导致的 update-forgetting 问题，实现可控文本生成的稳定性和效率大幅提升。

**[Evaluating Multiple Models Using Labeled And Unlabeled Data](social_computing/evaluating_multiple_models_using_labeled_and_unlabeled_data.md)**

:   提出 **SSME (Semi-Supervised Model Evaluation)**，利用少量标注数据和大量未标注数据，通过半监督混合模型估计多个分类器联合分布 $P(y, \mathbf{s})$，实现精确的分类器性能评估，误差降低至仅用标注数据的 1/5。

**[Graphkeeper Graph Domain-Incremental Learning Via Knowledge Disentanglement And ](social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)**

:   提出 GraphKeeper 框架应对**图域增量学习（Graph Domain-IL）**中的灾难性遗忘，通过域特异性 LoRA 参数隔离 + 领域内/间解耦 + 基于岭回归的无偏差知识保存三组件，比次优方法提升 6.5%-16.6%，且可无缝集成图基础模型。

**[If-Guide Influence Function-Guided Detoxification Of Llms](social_computing/if-guide_influence_function-guided_detoxification_of_llms.md)**

:   提出 IF-Guide，利用影响函数在 token 粒度识别训练数据中的有毒内容，并通过惩罚式训练目标在预训练/微调阶段主动抑制模型学习有毒行为，显著优于 DPO 和 RAD 等被动对齐方法。

**[Noise-Robustness Through Noise A Framework Combining Asymmetric Lora With Poison](social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)**

:   提出 LoPE，在非对称 LoRA 架构中设置专门的"中毒专家"接收注入噪声，推理时屏蔽该专家，仅通过正常专家输出实现噪声鲁棒——以噪声对抗噪声，完全无需数据清洗。

**[Os-Harm A Benchmark For Measuring Safety Of Computer Use Agents](social_computing/os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)**

:   本文提出 OS-Harm，首个面向通用计算机使用 Agent（非仅浏览器）的安全性 benchmark，覆盖用户恶意使用、Prompt 注入攻击、模型自身失误三类风险共 150 个任务，评测发现前沿模型（o4-mini、Claude 3.7 Sonnet、Gemini 2.5 Pro 等）普遍直接服从有害指令（最高 70% 不安全率），且对基础 prompt 注入有 20% 的服从率。

**[Policy-As-Prompt Turning Ai Governance Rules Into Guardrails For Ai Agents](social_computing/policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)**

:   提出 Policy-as-Prompt 框架，通过两阶段端到端流水线——策略树生成（POLICY-TREE-GEN）和策略即提示生成（POLICY-AS-PROMPT-GEN）——将团队已有的非结构化设计文档（PRD、TDD、代码）自动转换为可运行时执行的策略护栏，使用轻量级 LLM 作为合规"法官"，在 HR 和 SOC 应用中实现 70-73% 的输入/输出分类准确率。

**[Position Paper If Innovation In Ai Systematically Violates Fundamental Rights Is](social_computing/position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is.md)**

:   本文挑战"监管与创新对立"的固有信念，通过制药、航空、福利系统的历史类比和 Collingridge 困境分析论证良好设计的监管是创新的基础而非阻碍，并以 EU AI Act 的监管沙盒、中小企业支持等机制为范例展示监管如何加速而非延缓负责任的技术进步。

**[Precise Information Control In Long-Form Text Generation](social_computing/precise_information_control_in_long-form_text_generation.md)**

:   提出Precise Information Control (PIC)任务——要求LLM生成的长文严格基于给定声明集合（不遗漏不添加），构建PIC-Bench评测8个任务发现SOTA模型70%以上生成包含忠实性幻觉，通过弱监督偏好数据构建+DPO训练的PIC-LM将8B模型F1从69.1%提升至91.0%。

**[Slaying Towards Queer Language Processing](social_computing/slaying_towards_queer_language_processing.md)**

:   构建了首个显式标注的酷儿俚语（queer slang）数据集 SLAyiNG，包含 695 个术语和近 20 万条使用实例，并通过人机标注一致性实验（Krippendorff's α=0.746）表明推理模型可用于预筛选但仍需社区驱动的专家标注。

---

## 🔒 LLM安全 { #llm_safety }

**[A Cramrvon Mises Approach To Incentivizing Truthful Data Sha](llm_safety/a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)**

:   提出一种基于 Cramér-von Mises 两样本检验统计量的激励机制，在贝叶斯和无先验两种设定下均能证明"如实提交数据"构成（近似）Nash 均衡，同时鼓励参与者提交更多真实数据，且不依赖对数据分布的强假设（如高斯、伯努利）。

**[A Reliable Cryptographic Framework For Empirical Machine Unl](llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)**

:   将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

**[Angular Steering Behavior Control Via Rotation In Activation Space](llm_safety/angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例，在Llama 3/Qwen 2.5/Gemma 2（3B-14B）上实现鲁棒的行为调控。

**[Buffer Layers For Test-Time Adaptation](llm_safety/buffer_layers_for_test-time_adaptation.md)**

:   提出 Buffer 层作为测试时自适应 (TTA) 的新范式，替代传统的归一化层更新，从根本上保留预训练骨干网络的完整性，有效缓解灾难性遗忘并在多种架构和 TTA 框架中实现一致的性能提升。

**[Demystifying Language Model Forgetting With Low-Rank Example Associations](llm_safety/demystifying_language_model_forgetting_with_low-rank_example_associations.md)**

:   发现LLM微调后上游样本遗忘与新学任务之间的关联矩阵具有低秩结构（rank-3即$R^2>0.69$），利用矩阵补全预测未见任务导致的遗忘，指导选择性回放以减轻遗忘。

**[Finding Structure In Continual Learning](llm_safety/finding_structure_in_continual_learning.md)**

:   提出基于Douglas-Rachford Splitting (DRS)的持续学习优化框架，将稳定性与可塑性解耦为两个独立的近端子问题，并结合Rényi散度替代KL散度实现更鲁棒的先验对齐，从而在无需回放缓冲区或额外模块的条件下有效缓解灾难性遗忘。

**[Procurement Auctions With Predictions Improved Frugality For Facility Location](llm_safety/procurement_auctions_with_predictions_improved_frugality_for_facility_location.md)**

:   研究策略性无容量限制设施选址问题中的采购拍卖设计，证明了经典VCG拍卖的节俭比恰好为3（改进了此前已知的上界4），并设计了利用预测信息的学习增强拍卖机制，在预测准确时实现接近最优的节俭比，同时在预测任意不准确时仍保持常数级鲁棒性。

**[Simu Selective Influence Machine Unlearning](llm_safety/simu_selective_influence_machine_unlearning.md)**

:   提出 SIMU 两阶段框架：先通过梯度聚合识别编码遗忘集信息的关键 MLP 神经元，再仅对这些神经元进行二阶（Sophia）优化遗忘，在保持遗忘效果的同时大幅提升模型原有能力的保留。

**[Stop Ddos Attacking The Research Community With Ai-Generated Survey Papers](llm_safety/stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)**

:   这篇立场论文将AI生成综述论文的泛滥类比为对学术社区的"DDoS攻击"，通过对arXiv 2020-2024年10,063篇CS综述论文的系统定量分析，揭示了ChatGPT发布后综述论文数量、AI生成分数和异常作者数的同步激增现象，深入剖析了AI综述的四大质量缺陷（结构混乱、分类缺乏原创、引用不准确、内容高度冗余）及其对研究者-审稿人-编辑三方的文化冲击，提出了涵盖透明度要求、严格审查标准、冗余限制、AI检测辅助和"动态活综述"平台在内的全面应对框架。

**[Teaming Llms To Detect And Mitigate Hallucinations](llm_safety/teaming_llms_to_detect_and_mitigate_hallucinations.md)**

:   将单模型一致性方法（Self-Consistency + Semantic Entropy）推广到多个异构 LLM 的"联盟"设置，通过聚合不同训练背景的模型响应来打破单模型一致性幻觉，在 15 个 LLM 组成的模型池中评估大量联盟组合，发现匹配的强模型联盟在 92% 的情况下超越最强单模型基线，同时推理成本更低。

**[Trust -- Transformer-Driven U-Net For Sparse Target Recovery](llm_safety/trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)**

:   提出 TRUST 架构，将 Transformer 的注意力机制与 U-Net 解码器结合，在感知矩阵未知的条件下同时学习感知算子和重建稀疏信号，在 SSIM 和 PSNR 上显著超越传统方法。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[Adaptive Originality Filtering Rejection Based Prompting And Riddlescore For Cul](multilingual_mt/adaptive_originality_filtering_rejection_based_prompting_and_riddlescore_for_cul.md)**

:   提出 Adaptive Originality Filtering (AOF)——一种基于语义拒绝采样的提示策略，通过 MiniLM 嵌入的余弦相似度过滤重复/模板化输出，强制 LLM 生成更新颖、多样且文化匹配的多语言谜语；同时提出 RiddleScore 复合评估指标（Novelty + Diversity + Fluency + Alignment），与人类评分相关性达 $\rho=0.83$。

**[Dcad-2000 A Multilingual Dataset Across 2000 Languages With Data Cleaning As Ano](multilingual_mt/dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano.md)**

:   构建覆盖2282种语言、46.72TB文本的多语言数据集DCAD-2000，提出将数据清洗重构为异常检测问题的语言无关框架，通过8维统计特征+Isolation Forest动态过滤噪声数据，在多个多语言benchmark上验证效果，尤其对低资源语言提升显著。

**[Enhancing Multilingual Llm Pretraining With Model-Based Data Selection](multilingual_mt/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)**

:   提出一套透明、简洁、高效的多语言模型驱动数据筛选框架，利用 FastText 和 Transformer（XLM-RoBERTa）嵌入分类器识别结构化且知识丰富的样本，在 FineWeb-2 数据集上仅用 15% 的 token 即可匹配基线 MMLU 分数，并将该框架扩展至 20 种语言并公开发布了精炼的预训练数据集。

**[Exploring The Translation Mechanism Of Large Language Models](multilingual_mt/exploring_the_translation_mechanism_of_large_language_models.md)**

:   提出 subspace-intervened path patching 方法对 LLM 翻译机制进行精细因果分析，发现翻译由不到 5% 的稀疏 attention head 驱动——分为 source head、indicator head、positional head 三类功能角色，MLP 将其特征整合为以英语为中心的中间表示，仅微调 64 个关键 head 即可匹配全参数微调性能。

**[Helpsteer3-Preference Open Human-Annotated Preference Data Across Diverse Tasks ](multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)**

:   NVIDIA 发布的 40K+ 开源人工标注偏好数据集，覆盖通用/STEM/代码/多语言（13 种语言），训练的奖励模型在 RM-Bench 上达 82.4%（+10%），CC-BY-4.0 许可对商业友好。

**[How Data Mixing Shapes In-Context Learning Asymptotic Equivalence For Transforme](multilingual_mt/how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)**

:   在高维渐近框架下证明了带非线性MLP头的Transformer在ICL误差上等价于结构化多项式预测器，揭示了非线性MLP对非线性任务的增益机制，以及多源数据混合中低噪声和结构化协方差是高质量数据源的关键特征。

**[Mergebench A Benchmark For Merging Domain-Specialized Llms](multilingual_mt/mergebench_a_benchmark_for_merging_domain-specialized_llms.md)**

:   MergeBench 是首个全面评估大规模领域特化 LLM 合并的基准套件，覆盖 Llama 和 Gemma 系列最大 9B 模型、五大任务领域和八种合并方法，从多任务性能、遗忘、运行效率三个维度提供系统化评估和实用指南。

**[Merit Multilingual Semantic Retrieval With Interleaved Multi-Condition Query](multilingual_mt/merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query.md)**

:   提出首个多语言交错多条件语义检索数据集 MERIT（320K queries, 135K products, 5种语言, 7大品类），揭示现有检索模型仅关注全局语义而忽略条件细节的瓶颈，并设计 Coral 微调框架通过嵌入重建+对比学习将检索性能提升 45.9%。

**[Parallelprompt Extracting Parallelism From Large Language Model Queries](multilingual_mt/parallelprompt_extracting_parallelism_from_large_language_model_queries.md)**

:   构建了首个查询内并行（intra-query parallelism）基准数据集ParallelPrompt，包含37000+条真实用户提示的结构化分解标注，证明约10%的用户查询包含可并行的潜在结构，并行执行可实现最高5.7×的延迟加速且质量损失有限。

**[Quantifying Climate Policy Action And Its Links To Development Outcomes A Cross-](multilingual_mt/quantifying_climate_policy_action_and_its_links_to_development_outcomes_a_cross-.md)**

:   本文构建了一个NLP-计量经济学一体化框架，先用微调的多语言DistilBERT对全球气候政策文档按主题（减缓/适应/灾害风险管理/损失与损害）自动分类（F1=0.90），再与世界银行发展指标做固定效应面板回归，发现减缓政策与较高GDP/GNI显著正相关，而损失与损害政策全球仍然缺乏实质性实施。

**[Zero-Shot Performance Prediction For Probabilistic Scaling Laws](multilingual_mt/zero-shot_performance_prediction_for_probabilistic_scaling_laws.md)**

:   将 NLP 学习曲线预测建模为多任务学习问题，利用潜变量多输出高斯过程（MaGP）捕捉数据集中的双层层次结构和任务间相关性，实现学习曲线的零样本预测，并通过蒙特卡洛模拟推导概率化的 Scaling Laws。

---

## 🛰️ 遥感 { #remote_sensing }

**[C3Po Cross-View Cross-Modality Correspondence By Pointmap Prediction](remote_sensing/c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)**

:   构建了包含 90K 地面照片-平面图对（597 个场景、153M 像素级对应和 85K 相机位姿）的 C3 数据集，揭示现有对应模型在跨视角跨模态（如地面照片 vs. 平面图）场景下的局限性，通过在该数据上训练可将最佳方法的 RMSE 降低 34%。

**[Chamaevit Unifying Channelaware Masked Autoencoders And Mult](remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)**

:   提出ChA-MAEViT，通过动态通道-patch联合掩码、记忆token、混合token融合和通道感知解码器四大组件增强多通道图像（MCI）的跨通道特征学习，在卫星和显微三大数据集上平均超越SOTA 3.0-21.5%。

**[Connecting The Dots A Machine Learning Ready Dataset For Ionospheric Forecasting](remote_sensing/connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   作为2025 NASA Heliolab的成果，本文构建了首个全面的ML-ready电离层预测数据集，将太阳动力学观测站（SDO）极紫外辐照度嵌入、太阳风参数、行星际磁场、地磁活动指数、JPL稠密TEC全球电离层图、Madrigal稀疏TEC、太阳通量指数以及轨道力学参数等7大类异构数据源统一对齐到一致的时间-空间结构中，并在此基础上训练了包括LSTM、球面神经算子（SFNO）和GraphCast在内的多种时空预测架构，实现了对全球垂直总电子含量（vTEC）在安静和地磁活跃条件下长达12小时的自回归预测，超越了持续性基线。

**[Ecocast A Spatio-Temporal Model For Continual Biodiversity And Climate Risk Fore](remote_sensing/ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，融合卫星遥感（Sentinel-2）、气候再分析（ERA5）和公民科学观测（GBIF）数据的Transformer时空序列模型，通过12个月环境特征序列预测下月物种出现概率，在非洲5种鸟类分布预测上F1宏平均从Random Forest的0.31提升至0.65，并设计了基于EWC的持续学习框架以适应数据更新。

**[Geolink Empowering Remote Sensing Foundation Model With Openstreetmap Data](remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)**

:   GeoLink将OpenStreetMap矢量数据直接融入遥感基础模型预训练，通过异构GNN编码OSM数据并设计多粒度跨模态学习目标（区域-图像级对比 + 对象-patch级融合），在127万样本对上高效预训练后，7个分类和4个分割/变化检测benchmark全面超越现有RS FM。

**[Greenhyperspectra A Multi-Source Hyperspectral Dataset For Global Vegetation Tra](remote_sensing/greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)**

:   GreenHyperSpectra构建了一个包含14万+多源高光谱植被样本的预训练数据集，横跨近端、航空和卫星三种平台，通过半监督和自监督方法（MAE、GAN、RTM-AE）训练的标签高效回归模型在7种植物性状预测上全面超越全监督基线，特别是在标签稀缺和分布外场景中优势显著。

**[Mass Conservation On Rails -- Rethinking Physics-Informed Learning Of Ice Flow V](remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)**

:   提出散度无关神经网络（dfNN），通过流函数的辛梯度从架构上精确保证质量守恒（散度恒为零），结合方向引导学习策略，在南极Byrd冰川冰通量插值中显著优于软约束PINNs和无约束NN。

**[Orbitzoo Real Orbital Systems Challenges For Reinforcement Learning](remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)**

:   本文提出OrbitZoo，一个基于工业级Orekit轨道动力学库构建的多智能体RL环境，支持碰撞规避、霍曼转移、星座协调等真实轨道任务，通过PettingZoo接口实现标准化MARL训练，并在Starlink真实星历数据验证中达到低误差组24米RMSE（16.6小时传播）。

**[Ortholoc Uav 6-Dof Localization And Calibration Using Orthographic Geodata](remote_sensing/ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)**

:   OrthoLoC构建了首个面向正射地理数据（DOP+DSM）的大规模UAV 6-DoF定位基准数据集，包含16425张真实UAV图像覆盖德国和美国47个区域，并引入AdHoP（自适应单应性预处理）匹配改进技术，在不修改特征匹配器的情况下将匹配性能提升95%、平移误差降低63%。

**[Rscc A Large-Scale Remote Sensing Change Caption Dataset For Disaster Events](remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)**

:   构建了RSCC——首个大规模灾害感知遥感变化描述数据集（62,351对灾前/灾后图像+详细变化描述），覆盖地震/洪水/野火等31个全球事件，利用QvQ-Max视觉推理模型生成高质量标注，并建立了全面的基准评测体系。

**[Scaling Image Geo-Localization To Continent Level](remote_sensing/scaling_image_geo-localization_to_continent_level.md)**

:   混合方法结合分类学习的原型和航拍图像嵌入，在覆盖西欧43.3万平方公里上实现200m内68%+、100m内59.2%的定位率，首次在大陆规模实现此精度。

---

## 📡 信号/通信 { #signal_comm }

**[Angular Steering Behavior Control Via Rotation In Activation Space](signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例，在Llama 3/Qwen 2.5/Gemma 2（3B-14B）上实现鲁棒的行为调控。

**[Artificial Hivemind The Open-Ended Homogeneity Of Language Models And Beyond](signal_comm/artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)**

:   构建了 Infinity-Chat 数据集（26K 开放式真实用户查询 + 31,250 条人类标注），揭示了 LM 在开放式生成中的"Artificial Hivemind"效应——模型内重复和模型间同质化严重，并发现 Reward Model 和 LM Judge 在个体偏好差异大的样本上校准失败。

**[Bispectral Ot Dataset Comparison Using Symmetry-Aware Optimal Transport](signal_comm/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)**

:   提出 Bispectral Optimal Transport (BOT)，将离散最优传输中的代价矩阵从原始像素距离替换为 bispectrum（群 Fourier 不变量）距离，使得传输计划在保持信号结构的同时精确消除群作用（如旋转）带来的变异，在旋转变换的 MNIST 等数据集上将类别保持准确率从 33% 提升至 84%。

**[Contexttab A Semantics-Aware Tabular In-Context Learner](signal_comm/contexttab_a_semantics-aware_tabular_in-context_learner.md)**

:   ConTextTab 将语义嵌入（列名、分类值的文本编码）融入 table-native ICL 架构，并在大规模真实表格数据（T4, ~2.18M 表）上预训练，在语义丰富的 CARTE 基准上取得新 SOTA，同时在非语义基准上保持与现有方法竞争力。

**[Contrastive Consolidation Of Top-Down Modulations Achieves Sparsely Supervised C](signal_comm/contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Estimation Of Stochastic Optimal Transport Maps](signal_comm/estimation_of_stochastic_optimal_transport_maps.md)**

:   提出适用于随机OT映射的传输误差指标 $\mathcal{E}_p$（由优化间隙与可行性间隙组成），在无需Brenier映射存在或唯一性的最小假设下，构造了计算高效的rounding估计器达到近最优收敛率 $\tilde{O}(n^{-1/(d+2p)})$，并推广至Hölder连续核与对抗污染场景，建立了首个通用OT映射估计理论。

**[Masked Symbol Modeling For Demodulation Of Oversampled Baseband Communication Si](signal_comm/masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)**

:   本文提出 Masked Symbol Modeling（MSM），将 BERT 的掩码预测范式应用于通信物理层——将脉冲成形产生的符号间贡献重新定义为"上下文信息"，训练 Transformer 在干净过采样基带信号上学习波形结构，推理时利用学到的上下文来恢复被冲激噪声破坏的符号。

**[Memory-Integrated Reconfigurable Adapters A Unified Framework For Settings With ](signal_comm/memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。

**[Multi-Modal Masked Autoencoders For Learning Image-Spectrum Associations For Gal](signal_comm/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   构建了包含 134,533 个星系的图像-光谱-红移多模态数据集（GalaxiesML-Spectra），适配多模态掩码自编码器（MMAE）同时进行图像和光谱的联合重建与红移回归，证明在测试时即使光谱完全缺失，仅用 25% 掩码图像即可实现优于 AstroCLIP 的红移预测散度 $\sigma_{NMAD} = 0.016$。

**[Perturbation Bounds For Low-Rank Inverse Approximations Under Noise](signal_comm/perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)**

:   首次推导了噪声条件下低秩逆近似 $\|(\tilde{A}^{-1})_p - A_p^{-1}\|$ 的非渐近谱范数扰动界，通过新颖的等高线自举 (contour bootstrapping) 技术处理非整函数 $f(z) = 1/z$，在有利条件下比经典界改进 $\sqrt{n}$ 倍。

**[The Surprising Effectiveness Of Negative Reinforcement In Llm Reasoning](signal_comm/the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)**

:   将可验证奖励的强化学习（RLVR）分解为正样本强化（PSR，增强正确回答概率）和负样本强化（NSR，惩罚错误回答），发现仅用 NSR 就能在整个 Pass@k 谱上持续提升推理性能且通常匹配或超越 PPO/GRPO，据此提出 Weighted-REINFORCE（降低 PSR 权重至 0.1）在 MATH/AIME 2025/AMC23 上取得全面最优。

---

## 🔎 AIGC检测 { #aigc_detection }

**[Asciibench Evaluating Language-Model-Based Understanding Of Visually-Oriented Te](aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)**

:   提出 ASCIIBench，首个公开可用的 ASCII 艺术理解与生成基准（5,315 张图像，752 类），系统评估发现视觉模态显著优于文本模态，多模态融合反而不帮忙，且 CLIP 对 ASCII 结构的表征能力存在根本性瓶颈——只有内部一致性高的类别才能被有效区分。

**[Classical Planning With Llm-Generated Heuristics Challenging The State Of The Ar](aigc_detection/classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)**

:   提出让 LLM **生成域相关启发式函数的 Python 代码**（而非直接生成计划），通过 $n$ 次采样获得候选启发式池并在训练集上选优，将最优启发式注入 Python 规划器 Pyperplan 配合 GBFS 使用，在 IPC 2023 基准 8 个域上以纯 Python 实现超越了所有 C++ Fast Downward 传统启发式，且与 SOTA 学习型规划器 $h^{\mathrm{WLF}}_{\mathrm{GPR}}$ 持平，同时保证所有找到的计划 100% 正确。

**[Clawscreativity Detection For Llm-Generated Solutions Using Attention Window Of ](aigc_detection/clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)**

:   提出 CLAWS，通过分析 LLM 在生成数学解答时对不同 prompt 区段的注意力权重分布，无需人工评估即可将生成内容分类为"创造性"、"典型"或"幻觉"三类。

**[Duolens A Framework For Robust Detection Of Machine-Generated Multilingual Text ](aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)**

:   提出 DuoLens，一种基于 CodeBERT + CodeBERTa 双编码器融合的 AI 生成内容检测框架，在多语言文本（8 种语言）和源代码（7 种编程语言）检测上以极低计算成本（延迟降低 8-12×，VRAM 降低 3-5×）实现 AUROC 0.97-0.99，远超 GPT-4o 等大模型。

**[Jutters](aigc_detection/jutters.md)**

:   通过荷兰传统"jutters"（海岸拾荒者）的隐喻，构建了一个融合真实海滩碎片与AI生成图像/视频的沉浸式装置艺术，引导参观者以拾荒者心态反思如何对待AI生成内容。

**[Reasoning Compiler Llm-Guided Optimizations For Efficient Model Serving](aigc_detection/reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)**

:   提出 Reasoning Compiler，将编译器优化建模为序列决策过程，用 LLM 作为上下文感知提案引擎 + MCTS 平衡探索/利用，在 5 个代表性 benchmark 和 5 个硬件平台上实现平均 5.0× 加速且采样效率比 TVM 进化搜索提升 10.8×。

**[Synthesizing Performance Constraints For Evaluating And Improving Code Efficienc](aigc_detection/synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)**

:   提出Wedge框架——通过LLM合成性能刻画约束（performance-characterizing constraints）指导约束感知模糊测试，生成能暴露代码性能瓶颈的压力测试输入，构建PerfForge基准，使LLM代码优化器（如Effi-Learner）多减24% CPU指令。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[Edit Less Achieve More Dynamic Sparse Neuron Masking For Lifelong Knowledge Edit](knowledge_editing/edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)**

:   提出 NMKE 框架，通过神经元级归因发现 knowledge-general 和 knowledge-specific 两类知识神经元，并结合熵引导的动态稀疏 mask，实现精准神经元级知识编辑，在 5000 步连续编辑后仍保持高编辑成功率和模型通用能力。

**[Kscope A Framework For Characterizing The Knowledge Status Of Language Models](knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)**

:   提出LLM知识状态的五分类法（一致正确/冲突正确/缺失/冲突错误/一致错误）和KScope层次化统计检验框架，通过重复采样+多步假设检验精确刻画LLM对给定问题的知识模式结构，并系统研究上下文如何更新各状态，发现受约束的上下文摘要+增强可信度平均提升4.3%的知识更新成功率。

**[Memeic A Step Toward Continual And Compositional Knowledge Editing](knowledge_editing/memeic_a_step_toward_continual_and_compositional_knowledge_editing.md)**

:   提出 MemEIC 框架，通过外部双模态检索记忆 + 内部模态分离 LoRA 适配器 + 仿脑 Knowledge Connector 三层架构，实现大视觉语言模型的持续、组合式知识编辑，在新提出的 CCKEB 基准上大幅超越现有方法。

**[Memoir Lifelong Model Editing With Minimal Overwrite And Informed Retention For ](knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)**

:   提出MEMOIR框架，通过在FFN层引入零初始化的残差记忆矩阵，利用基于TopHash的稀疏掩码将每次编辑限制在记忆参数的不同子集上，推理时通过掩码重叠率识别相关编辑并条件性激活知识，在15000次连续编辑下仍保持可靠性、泛化性和局部性的最优平衡。

**[Rethinking Residual Distribution In Locate-Then-Edit Model Editing](knowledge_editing/rethinking_residual_distribution_in_locate-then-edit_model_editing.md)**

:   揭示 locate-then-edit 模型编辑中残差分配（residual distribution）机制引入的权重偏移误差会随分配距离、batch 大小和编辑序列长度增长，提出 BLUE（Boundary Layer UpdatE）策略仅更新首尾关键层，平均提升 35.59%。

**[Uniedit A Unified Knowledge Editing Benchmark For Large Language Models](knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)**

:   构建 UniEdit——首个基于开放域知识图谱（Wikidata）的统一 LLM 知识编辑基准，覆盖 5 大类 25 个领域共 311K 条样本，通过邻域多跳链采样（NMCS）算法统一整合多种泛化性和局部性评估标准，系统揭示了现有编辑方法在复杂波纹效应评估下的不足。

---

## 🗣️ 对话系统 { #dialogue }

**[Aclora Almost Trainingfree Access Controlaware Multimodal Ll](dialogue/aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)**

:   设计 AC-LoRA 端到端系统，为不同权限数据集训练独立的 LoRA 适配器，推理时根据用户查询的 cosine 相似度和权限动态检索并无训练合并多个 LoRA 输出，在保证强信息隔离的同时匹配或超越 SOTA LoRA 混合方法的回答质量。

**[Bridging Human And Llm Judgments Understanding And Narrowing The Gap](dialogue/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)**

:   提出Bridge统计框架，通过序数logistic回归建模人类和LLM评判之间的潜在关系，以少量人类标签改善LLM评判的校准和对齐，同时支持对系统性偏差的正式统计检验。

**[Hygen Efficient Llm Serving Via Elastic Online-Offline Request Co-Location](dialogue/hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)**

:   提出HyGen——干扰感知的LLM推理系统，通过精准的批次延迟预测器、SLO感知的性能分析器和前缀共享最大化调度策略，实现在线和离线工作负载的弹性共置，在保证严格SLO合规的同时获得3.87-5.84倍吞吐提升。

**[Metamind Modeling Human Social Thoughts With Metacognitive Multi-Agent Systems](dialogue/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)**

:   提出 MetaMind——一个受心理学元认知理论启发的多智能体框架，通过 ToM Agent（心理状态假设生成）、Moral Agent（社会规范约束精炼）和 Response Agent（响应生成与自我验证）三阶段协作，显著提升 LLM 的社会推理能力，在多个社会智能基准上达到 SOTA 并首次接近人类水平。

**[Sciarena An Open Evaluation Platform For Non-Verifiable Scientific Literature-Gr](dialogue/sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)**

:   构建 SciArena 社区驱动的科学文献评估开放平台，采用 Chatbot Arena 式的人类偏好投票方式对 47 个基础模型进行排名，收集超过 20,000 条投票数据，并发布 SciArena-Eval 元基准来评测自动评估系统对文献任务答案质量的判断能力。

---

## 🌍 地球科学 { #earth_science }

**[A Probabilistic Unet Approach To Downscaling Climate Simulat](earth_science/a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)**

:   首次将概率 U-Net 应用于气候统计降尺度（16× 超分辨率），通过变分隐空间采样生成集合预报来量化降尺度不确定性，并系统比较了 WMSE、MS-SSIM、WMSE-MS-SSIM 和 afCRPS 四种训练目标在捕捉极端事件与保留细尺度空间变异性方面的互补权衡。

**[Adaptive Online Emulation For Accelerating Complex Physical Simulations](earth_science/adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)**

:   提出 Adaptive Online Emulation (AOE)，在物理模拟执行过程中动态训练 ELM 神经网络代理模型替代昂贵计算组件，无需离线预训练，在系外行星大气模拟上实现 11.1× 加速（91% 时间节省）且精度损失仅 ~0.01%。

**[Controlfusion A Controllable Image Fusion Framework With Language-Vision Degrada](earth_science/controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)**

:   提出 ControlFusion，一种基于语言-视觉退化提示的可控红外-可见光图像融合框架，通过物理驱动的退化成像模型模拟复合退化，并用 prompt-modulated 网络动态恢复+融合，在真实世界和复合退化场景下全面超越 SOTA。

**[Predicting Public Health Impacts Of Electricity Usage](earth_science/predicting_public_health_impacts_of_electricity_usage.md)**

:   提出 HealthPredictor，一个将电力消费端到端映射到公共健康损害（以 $/MWh 计量）的 AI 流水线，包含燃料组合预测、空气质量转换和健康影响评估三个模块，健康驱动优化比燃料组合驱动基线显著降低健康影响预测误差，并在电动汽车充电调度案例中实现 24-42% 的健康损害减少。

**[Reasoning With A Star A Heliophysics Dataset And Benchmark For Agentic Scientifi](earth_science/reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)**

:   提出 Reasoning With a Star (RWS)，一个源自 NASA 太阳物理暑期学校问题集的 158 道科学推理 benchmark（含数值/符号/文本三类答案），配合 unit-aware 评分器，比较了四种多 agent 协调模式（HMAW/PACE/PHASE/SCHEMA），发现没有单一模式在所有任务上占优——系统工程启发的 SCHEMA 在需要严格约束验证的任务上最强。

---

## 📖 NLP理解 { #nlp_understanding }

**[Planning Without Search Refining Frontier Llms With Offline Goal-Conditioned Rl](nlp_understanding/planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)**

:   提出PNLC方法，通过训练轻量级目标条件价值函数作为"自然语言评论家"，在推理步骤层面引导LLM智能体进行多轮规划和自我精化，无需直接微调或推理时搜索，在Web导航、社交推理、劝服等复杂交互任务上显著超越现有方法且推理速度快8-10倍。

---

## 📂 其他 { #others }

**[4Dgt Learning A 4D Gaussian Transformer Using Realworld Mono](others/4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)**

:   提出4DGT——一种基于4D高斯的Transformer模型，完全在真实世界单目带位姿视频上训练，以前馈方式在几秒内完成动态场景重建，显著优于同类前馈网络，并达到与优化类方法可比的精度。

**[A Differentiable Model Of Supply-Chain Shocks](others/a_differentiable_model_of_supply-chain_shocks.md)**

:   用 JAX 实现可微分的供应链 Agent-Based Model（~1000 家企业），通过 GPU 并行化 + 自动微分实现比传统 ABC 快 3 个数量级的贝叶斯参数校准，为全球供应链网络的冲击传播建模铺平道路。

**[A Generalized Label Shift Perspective For Crossdomain Gaze E](others/a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)**

:   本文将跨域视线估计(CDGE)问题建模为广义标签偏移(GLS)问题，指出现有域不变表示学习方法在标签偏移存在时理论上不充分，提出基于截断高斯分布的连续重要性重加权和概率感知条件算子差异(PCOD)来联合纠正标签偏移和条件偏移，在多个backbone上平均降低误差12%~27%。

**[A Sustainable Ai Economy Needs Data Deals That Work For Gene](others/a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)**

:   提出"经济数据处理不等式"概念——ML价值链中数据从原始形态到模型权重再到合成输出，每一步都精炼了技术信号但系统性剥夺了数据生成者的经济权益；通过分析73个公开数据交易案例实证这一现象，诊断三个结构性缺陷（溯源缺失、议价权不对称、定价非动态），并提出EDVEX框架作为解决方案蓝图。

**[A Theoretical Framework For Grokking Interpolation Followed By Riemannian Norm M](others/a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)**

:   本文从纯优化角度严格证明了 grokking 现象的成因：带小 weight decay 的梯度流在 $\lambda\to 0$ 极限下呈现两阶段动力学——先快速收敛到训练损失的临界流形 $\mathcal{M}$，再在 $t\approx 1/\lambda$ 时沿流形做黎曼梯度流以最小化 $\ell_2$ 范数，从而延迟实现泛化。

**[A Unified Framework For Variable Selection In Modelbased Clu](others/a_unified_framework_for_variable_selection_in_modelbased_clu.md)**

:   在高斯混合模型的聚类框架中，统一解决变量选择（区分信号变量、冗余变量和噪声变量）与MNAR缺失数据建模，通过两阶段策略（LASSO惩罚排序加BIC角色分配）和谱距离自适应惩罚权重实现高维场景下的高效推理，并证明了可辨识性和渐近选择一致性。

**[Active Measurement Efficient Estimation At Scale](others/active_measurement_efficient_estimation_at_scale.md)**

:   提出 Active Measurement 框架，将 AI 模型预测作为重要性采样提议分布，通过迭代的人类标注与模型更新实现科学总量测量的无偏估计，搭配新颖的组合权重方案和条件方差估计器构建可靠的置信区间。

**[Acurank Uncertainty-Aware Adaptive Computation For Listwise Reranking](others/acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)**

:   利用贝叶斯TrueSkill模型维护文档相关性的概率分布，在每轮迭代中只对排名不确定的文档进行重排序，实现根据查询难度自适应调配计算量的重排框架，在多个基准上以更少调用次数超越固定计算基线。

**[Adaptive Data Analysis For Growing Data](others/adaptive_data_analysis_for_growing_data.md)**

:   首次提出动态增长数据上自适应分析的泛化界，允许分析者依据当前数据规模自适应调度查询，在批量查询设定下数据需求增长为自适应查询数的平方根，匹配静态设定下的渐近最优结果。

**[Addressing Mark Imbalance In Integrationfree Neural Marked T](others/addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)**

:   本文首次揭示标记时间点过程(MTPP)中标记分布不平衡对预测性能的严重影响，提出先预测标记再预测时间的策略，设计阈值方法调节稀有标记的预测概率，并开发无积分近似的IFNMTPP模型高效支持标记概率估计和时间采样。

**[Adjoint Schrödinger Bridge Sampler](others/adjoint_schrödinger_bridge_sampler.md)**

:   提出 Adjoint Schrödinger Bridge Sampler (ASBS)，通过将 Schrödinger Bridge 问题重新解释为随机最优控制问题，消除了先前扩散采样器的 memoryless 条件限制，支持任意源分布（如高斯、谐波先验），使用可扩展的 matching 目标无需重要性权重估计，在多粒子能量函数和分子构象生成上全面超越先前方法。

**[Adjusted Count Quantification Learning On Graphs](others/adjusted_count_quantification_learning_on_graphs.md)**

:   将经典的 Adjusted Classify & Count (ACC) 量化方法扩展到图结构数据，提出结构重要性采样（SIS）和邻域感知ACC两种技术，分别解决图量化中的结构协变量偏移和非同质性边问题。

**[Adpretrain Advancing Industrial Anomaly Detection Via Anomaly Representation Pre](others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)**

:   首次提出面向工业异常检测的专用表示预训练框架 ADPretrain，通过角度和范数导向的对比损失在大规模异常检测数据集 RealIAD 上学习残差特征表示，替换五种主流嵌入式 AD 方法的原始特征后在五个数据集、五个骨干网络上取得一致性提升。

**[Alias-Free Vit Fractional Shift Invariance Via Linear Attention](others/alias-free_vit_fractional_shift_invariance_via_linear_attention.md)**

:   提出Alias-Free Vision Transformer（AFT），结合抗混叠信号处理技术和shift-equivariant线性交叉协方差注意力，首次使ViT在分数像素（亚像素）平移下保持接近完美的一致性（~99%），同时在ImageNet分类准确率上几乎无损。

**[An Empirical Investigation Of Neural Odes And Symbolic Regression For Dynamical ](others/an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)**

:   本文系统研究了 Neural ODE (NODE) 在含噪合成数据上的外推能力，并探索了将 NODE 作为数据增强工具、与符号回归 (SR) 结合以从有限数据中恢复动力学方程的流水线，结果表明该组合方案能从仅 10% 的仿真数据中恢复三个控制方程中的两个及第三个的良好近似。

**[An Evidence-Based Post-Hoc Adjustment Framework For Anomaly Detection Under Data](others/an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)**

:   EPHAD 提出一种测试时后处理框架，通过指数倾斜（exponential tilting）将已被污染数据训练的异常检测模型输出与外部证据（CLIP/LOF等）进行贝叶斯式融合校正，无需接触训练流程，在8个视觉和26个表格AD数据集上一致提升被污染模型的检测性能。

**[Are Pixel-Wise Metrics Reliable For Sparse-View Computed Tomography Reconstructi](others/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)**

:   揭示 PSNR/SSIM 等像素级指标无法反映稀疏视图 CT 重建中解剖结构完整性（相关性仅 0.16-0.30），提出基于自动分割的解剖感知指标（NSD/clDice）和 CARE 框架——在扩散模型训练中加入分割引导损失，大器官结构完整性提升 32%、血管提升 36%。

**[Autoscidact Automated Scientific Discovery Through Contrastive Embedding And Hyp](others/autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)**

:   提出 AutoSciDACT 管线：先用有监督对比学习将高维科学数据压缩到 4 维嵌入空间，再用 NPLM（New Physics Learning Machine）似然比检验对嵌入空间中的分布偏差进行统计量化，在天文、粒子物理、病理、图像和合成数据集上以 ≤1% 的信号注入比例实现 ≥3σ 发现。

**[Brain-Like Processing Pathways Form In Models With Heterogeneous Experts](others/brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)**

:   在异构 Mixture-of-Experts 模型中，异构专家并不会自动形成处理通路；本文提出三个受大脑启发的归纳偏置（路由代价、任务表现缩放、专家 Dropout），使模型形成类似大脑"皮层-皮层下"动态通路的 Mixture-of-Pathways 架构。

**[Computable Universal Online Learning](others/computable_universal_online_learning.md)**

:   在 universal online learning 框架中引入可计算性约束，证明了"数学上可学习"不等于"可用计算机程序实现的可学习"，并给出了 agnostic 和 proper 变体下可计算学习的精确刻画。

**[Contexttab A Semantics-Aware Tabular In-Context Learner](others/contexttab_a_semantics-aware_tabular_in-context_learner.md)**

:   ConTextTab 将语义嵌入（列名、分类值的文本编码）融入 table-native ICL 架构，并在大规模真实表格数据（T4, ~2.18M 表）上预训练，在语义丰富的 CARTE 基准上取得新 SOTA，同时在非语义基准上保持与现有方法竞争力。

**[Contextual Dynamic Pricing With Heterogeneous Buyers](others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)**

:   首次系统研究买家类型异质（$K_\star$ 种未知类型）的上下文动态定价问题，提出基于乐观后验采样 (OPS) 的算法实现 $\tilde{O}(K_\star\sqrt{dT})$ 遗憾界（对 $d$ 和 $T$ 最优），并在非上下文情形通过方差感知自适应离散化算法 ZoomV 实现 $\tilde{O}(\sqrt{K_\star T})$ 最优遗憾。

**[Continuous Thought Machines](others/continuous_thought_machines.md)**

:   提出 Continuous Thought Machine (CTM)，通过私有参数化的 Neuron-Level Models (NLMs) 产生神经元级时间动力学，并以神经同步矩阵作为核心潜在表征，在迷宫求解、ImageNet 分类、奇偶校验等任务上展现复杂推理、自适应计算和可解释注意力行为。

**[Coreset For Robust Geometric Median Eliminating Size Dependency On Outliers](others/coreset_for_robust_geometric_median_eliminating_size_dependency_on_outliers.md)**

:   首次消除鲁棒几何中位数 coreset 大小对异常值数 $m$ 的依赖：在 $n \geq 4m$ 条件下，$d=1$ 时实现最优 coreset 大小 $\tilde{\Theta}(\varepsilon^{-1/2} + \frac{m}{n}\varepsilon^{-1})$，高维时实现 $\tilde{O}(\varepsilon^{-2}\min\{\varepsilon^{-2}, d\})$，核心技术是新颖的**非逐分量误差分析**。

**[Coresets For Clustering Under Stochastic Noise](others/coresets_for_clustering_under_stochastic_noise.md)**

:   首次系统研究噪声数据下 $(k,z)$-聚类 coreset 构造问题，提出新的代理误差度量 $\mathsf{Err}_\alpha$ 替代传统 $\mathsf{Err}$，在温和数据假设下实现 coreset 大小缩减 $\text{poly}(k)$ 倍、质量保证收紧 $\text{poly}(k)$ 倍，并设计噪声感知的 cluster-wise 采样算法。

**[Deep Continuous-Time State-Space Models For Marked Event Sequences](others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)**

:   S2P2 将线性 Hawkes 过程与深度状态空间模型结合，通过堆叠多层隐式线性 Hawkes (LLH) 层 + 非线性激活构建高表达力的连续时间 MTPP 模型，利用并行扫描实现线性复杂度和亚线性时间，在 8 个真实数据集上平均提升 33% 预测似然。

**[Deep Legendre Transform](others/deep_legendre_transform.md)**

:   DLT 利用凸共轭的隐式 Fenchel 表示 $f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x)$ 将凸共轭计算转化为标准回归问题，避免求解 max/min-max 优化，且能提供后验误差估计，结合 KAN 还可获得精确解析解。

**[Depth-Bounds For Neural Networks Via The Braid Arrangement](others/depth-bounds_for_neural_networks_via_the_braid_arrangement.md)**

:   本文证明了在 $\mathcal{B}_d^0$-conforming 约束下，ReLU 网络精确表示 $\max\{0, x_1, \ldots, x_d\}$ 需要 $\Omega(\log \log d)$ 层——这是首个不限制权重的非常数深度下界；同时证明 rank-(3,2) maxout 网络可以计算 7 个数的最大值，说明标准上界不紧。

**[Depth-Supervised Fusion Network For Seamless-Free Image Stitching](others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)**

:   DSFN 提出深度一致性约束的无缝图像拼接方法：通过深度感知的两阶段变换估计解决大视差对齐，软缝合区域扩散实现自然融合，结合重参数化策略提升效率，在 UDIS-D 和 IVSD 数据集上全面超越 SOTA。

**[Directional Non-Commutative Monoidal Structures For Compositional Embeddings In ](others/directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)**

:   提出一种基于方向性非交换幺半群算子的代数框架，为多维组合嵌入提供统一数学基础，将 SSM 递归、Transformer 自注意力和 RoPE 位置编码统一为特例。

**[Distributionally Robust Feature Selection](others/distributionally_robust_feature_selection.md)**

:   本文提出一种模型无关的分布鲁棒特征选择方法，通过向协变量注入可控高斯噪声实现离散选择的连续松弛，并优化 Bayes 最优预测器的条件方差，使选出的特征子集能在多个子群体上同时训练出高质量下游模型。

**[Double Descent Meets Out-Of-Distribution Detection Theoretical Insights And Empi](others/double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)**

:   本文首次揭示 post-hoc OOD 检测中存在 double descent 现象——OOD 检测性能随模型宽度在插值阈值附近出现谷值后再次上升，通过随机矩阵理论提供理论解释，并提出基于 Neural Collapse 的 NC1 判据来识别最佳模型复杂度区间。

**[Dpa A One-Stop Metric To Measure Bias Amplification In Classification Datasets](others/dpa_a_one-stop_metric_to_measure_bias_amplification_in_classification_datasets.md)**

:   本文提出 Directional Predictability Amplification (DPA)，一种基于可预测性的偏差放大度量指标，是唯一同时满足方向性、适用于平衡/非平衡数据集、能正确识别正负偏差放大的一站式指标，通过测量模型偏差与数据集偏差的相对变化来量化偏差放大程度。

**[Efficient Kernelized Learning In Polyhedral Games Beyond Full-Information From C](others/efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)**

:   提出基于核化（kernelization）的框架，在部分信息反馈设定下为多面体博弈（Colonel Blotto、图拟阵拥堵博弈、网络拥堵博弈）设计了计算高效的无遗憾学习算法，显著改进了学习粗关联均衡（CCE）的运行时复杂度。

**[Efficient Parametric Svd Of Koopman Operator For Stochastic Dynamical Systems](others/efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)**

:   提出基于 low-rank approximation (LoRA) 的目标函数来学习随机动力系统 Koopman 算子的 top-k 奇异函数，完全避免了 VAMPnet/DPNet 中数值不稳定的矩阵分解操作，且梯度天然无偏。

**[Emergency Response Measures For Catastrophic Ai Risk](others/emergency_response_measures_for_catastrophic_ai_risk.md)**

:   本文系统分析了前沿安全政策（FSPs）如何嵌入中国四阶段应急响应框架（预防-预警-响应-恢复）的前两个阶段，通过危险能力评估、分级阈值和预设安全措施来应对AI灾难性风险，并与欧盟AI法案、加州SB53等国际实践进行了对比。

**[Equivariance By Contrast Identifiable Equivariant Embeddings From Unlabeled Fini](others/equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)**

:   提出 Equivariance by Contrast (EbC)，一种仅用编码器的方法，从观测对 $(\mathbf{y}, g \cdot \mathbf{y})$ 中联合学习等变嵌入空间和隐式群表示，使有限群作用在潜空间中对应可逆线性映射，并提供可辨识性理论保证。

**[Evaluating In Silico Creativity An Expert Review Of Ai Chess Compositions](others/evaluating_in_silico_creativity_an_expert_review_of_ai_chess_compositions.md)**

:   Google DeepMind训练了三种生成式神经网络（自回归Transformer、离散扩散、MaskGit）学习国际象棋谜题分布，通过强化学习优化谜题的唯一性和反直觉性，生成约400万个棋局位置，经奖励函数筛选和美学主题检测后，邀请三位世界级国际象棋专家评审，得到积极但带有建设性批评的反馈。

**[Evobrain Dynamic Multi-Channel Eeg Graph Modeling For Time-Evolving Brain Networ](others/evobrain_dynamic_multi-channel_eeg_graph_modeling_for_time-evolving_brain_networ.md)**

:   提出 EvoBrain——首次从理论上证明 **显式动态图建模** 优于隐式静态图、**time-then-graph** 架构表达力严格优于其他两种动态 GNN 范式(graph-then-time / time-and-graph)，并据此设计双流 Mamba + Laplacian PE 增强的 GCN 模型，在 TUSZ 和 CHB-MIT 数据集的癫痫检测与早期预测任务上取得 AUROC 提升 23%、F1 提升 30% 的显著效果，同时训练速度比 SOTA 快 17 倍。

**[Evolutionary Prediction Games](others/evolutionary_prediction_games.md)**

:   提出"演化预测博弈"框架，用演化博弈论分析预测算法与用户群体之间的反馈循环，揭示理想学习器导致竞争排斥（强者生存），而实际学习器（有限数据/代理损失/过参数化）反而能促成群体间的稳定共存与互利共生。

**[Exact Learning Of Arithmetic With Differentiable Agents](others/exact_learning_of_arithmetic_with_differentiable_agents.md)**

:   提出可微有限状态转换器（DFST），一种图灵完备且端到端可微的模型族，在 2D 符号网格上通过观察专家算术计算的中间步骤（Policy-Trajectory Observations）训练，仅用 20 个样本（最长 3 位数加法）即可完美泛化到 3850 位二进制加法、2450 位十进制加法，未发现任何错误。

**[Faithful Group Shapley Value](others/faithful_group_shapley_value.md)**

:   提出 Faithful Group Shapley Value (FGSV)，唯一满足含"忠实性"在内五条公理的组级数据估值方法，有效防御"空壳公司攻击"（通过拆分子组不当膨胀估值），并设计了 $O(n \cdot \text{Poly}(\log n))$ 复杂度的高效近似算法。

**[Flashmd Long-Stride Universal Prediction Of Molecular Dynamics](others/flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)**

:   提出 FlashMD，基于 GNN 直接预测分子动力学轨迹的位置与动量跨步演化，实现比传统 MD 积分器大 1–2 个数量级的时间步长跨越，并在架构中融入哈密顿动力学约束，推广到任意热力学系综和通用化学体系。

**[Flowmoe A Scalable Pipeline Scheduling Framework For Distributed Mixture-Of-Expe](others/flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)**

:   FlowMoE提出统一的流水线调度框架，将MHA计算、门控、专家计算和A2A通信纳入一体化流水线，并使用优先级驱动的all-reduce张量分块机制最大化通信与计算的重叠，在多种真实MoE模型上实现1.13×-1.82×加速、10-39%能耗降低和7-32%内存节省。

**[Fostering The Ecosystem Of Ai For Social Impact Requires Expanding And Strengthe](others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)**

:   本文主张 AI for Social Impact (AISI) 领域的学术生态需要双轨改革：拓宽"影响力"的定义以认可非部署/非方法创新的贡献，同时对已部署系统采用因果推断级别的严格评估标准。

**[Fsnet Feasibility-Seeking Neural Network For Constrained Optimization With Guara](others/fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)**

:   提出 FSNet 框架，将**可微的可行性求解步骤**集成到神经网络中，通过最小化约束违反的无约束优化来保证约束满足，同时支持端到端训练，在凸/非凸、光滑/非光滑问题上均显著快于传统求解器且保持可行性。

**[Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret In Noise-](others/gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)**

:   证明GP-UCB算法在无噪声GP bandit问题中达到近最优遗憾界，包括在SE核和Matérn核（$d > \nu$）条件下首次获得常数累积遗憾$O(1)$，弥合了GP-UCB理论与实际性能之间的差距。

**[Generalized Linear Mode Connectivity For Transformers](others/generalized_linear_mode_connectivity_for_transformers.md)**

:   提出统一对称性框架（置换、半置换、正交、可逆变换四级层次），首次在 Vision Transformer 和 GPT-2 上实现零/近零 barrier 的线性模式连通性（LMC），并扩展至多模型融合与异构宽度对齐。

**[Graph Alignment Via Birkhoff Relaxation](others/graph_alignment_via_birkhoff_relaxation.md)**

:   本文首次为图对齐问题的 Birkhoff 松弛（将排列矩阵约束松弛为双随机矩阵约束）提供了理论保证，在高斯 Wigner 模型下证明了最优解的相变行为：当噪声 $\sigma = o(n^{-1})$ 时松弛解接近真实排列，当 $\sigma = \Omega(n^{-0.5})$ 时松弛解远离真实排列。

**[Harnessing Feature Resonance Under Arbitrary Target Alignment For Out-Of-Distrib](others/harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)**

:   发现 Feature Resonance 现象——优化已知 ID 节点表征时未知 ID 节点的表征变化显著大于 OOD 节点，且该现象与标签无关，据此提出无需多类标签的图 OOD 节点检测框架 RSL，在 13 个数据集上达到 SOTA。

**[Hessian-Guided Perturbed Wasserstein Gradient Flows For Escaping Saddle Points](others/hessian-guided_perturbed_wasserstein_gradient_flows_for_escaping_saddle_points.md)**

:   提出扰动Wasserstein梯度流(PWGF)算法，通过基于Hessian构造的高斯过程注入噪声扰动，使概率测度优化能够高效逃离鞍点并达到二阶最优性。

**[Houselayout3D A Benchmark And Training-Free Baseline For 3D Layout Estimation In](others/houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)**

:   提出 HouseLayout3D——首个面向大规模多层建筑的真实世界 3D layout 估计基准，以及 MultiFloor3D——一个无需训练的基线方法，通过组合现代 3D 重建和分割模型在多层建筑 layout 估计上超越现有深度学习方法。

**[How Many Domains Suffice For Domain Generalization A Tight Characterization Via ](others/how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_.md)**

:   提出"领域碎裂维度"（Domain Shattering Dimension）这一新组合度量，紧致刻画了领域泛化所需的领域数量（领域样本复杂度），并证明其与经典VC维的关系为 $\Theta(d \log(1/\alpha))$。

**[Hybrid-Balance Gflownet For Solving Vehicle Routing Problems](others/hybrid-balance_gflownet_for_solving_vehicle_routing_problems.md)**

:   提出Hybrid-Balance GFlowNet（HBG）框架，首次在VRP场景中引入详细平衡（DB）并与轨迹平衡（TB）统一集成，配合depot引导推理策略，在CVRP和TSP上显著提升两种现有GFlowNet求解器（AGFN和GFACS）的性能。

**[Hypergraphrag Retrieval-Augmented Generation Via Hypergraph-Structured Knowledge](others/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)**

:   提出 HyperGraphRAG，首个基于超图 (hypergraph) 结构的 RAG 方法，通过超边 (hyperedge) 建模 n 元关系（n≥2），克服了现有图谱 RAG 方法受限于二元关系的瓶颈，在医学、农业、计算机科学和法律等领域的问答任务中全面超越 StandardRAG 和 GraphRAG 系列方法。

**[Impact Of Layer Norm On Memorization And Generalization In Transformers](others/impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)**

:   系统揭示了LayerNorm在Pre-LN和Post-LN Transformer中的**截然不同**角色：Pre-LN中LN对学习至关重要，移除会破坏泛化；Post-LN中LN驱动记忆化，移除可抑制记忆化并恢复真实标签。

**[Improved Approximation Algorithms For Chromatic And Pseudometric-Weighted Correl](others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)**

:   针对 Correlation Clustering 的两个重要推广——Chromatic CC 和 pseudometric-weighted CC，基于 LP relaxation 与精心设计的 rounding function，分别取得 2.15-approximation 和 tight 10/3-approximation，显著改进了先前最佳结果（2.5 和 6）。

**[Improving Decision Trees Through The Lens Of Parameterized Local Search](others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)**

:   从参数化复杂度的视角分析决策树的局部搜索优化操作，揭示问题的难度来源，并证明特征数与值域大小的组合可实现固定参数可解(FPT)，同时提供了概念验证实现。

**[Improving Forecasts Of Suicide Attempts For Patients With Little Data](others/improving_forecasts_of_suicide_attempts_for_patients_with_little_data.md)**

:   提出 Latent Similarity Gaussian Process (LSGP)，将患者嵌入连续隐空间以捕获异质性，使数据稀少的患者能从相似患者"借用"预测趋势，从而改进基于 EMA 数据的自杀未遂预测。

**[Inferring Stochastic Dynamics With Growth From Cross-Sectional Data](others/inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)**

:   提出非平衡概率流推断（UPFI），通过Fokker-Planck方程的Lagrangian形式化，从横截面数据中联合推断随机动力学系统的漂移项、扩散项和增长率，首次准确处理含细胞增殖/死亡的场景。

**[Information-Computation Tradeoffs For Noiseless Linear Regression With Oblivious](others/information-computation_tradeoffs_for_noiseless_linear_regression_with_oblivious.md)**

:   对无噪声线性回归在Oblivious污染模型下，形式化证明任何高效Statistical Query算法都需要 $\tilde{\Omega}(d^{1/2}/\alpha^2)$ 的VSTAT复杂度，给出了 $1/\alpha$ 的二次依赖对高效算法具有本质性的计算下界证据。

**[Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry](others/johnson-lindenstrauss_lemma_beyond_euclidean_geometry.md)**

:   将Johnson-Lindenstrauss引理从欧几里得空间扩展到一般对称空心相异度矩阵，提出伪欧空间JL变换和广义幂距离JL变换两种互补方法，误差与数据偏离欧几何的程度成正比。

**[Kernel Conditional Tests From Learning-Theoretic Bounds](others/kernel_conditional_tests_from_learning-theoretic_bounds.md)**

:   提出将学习算法的置信界转化为条件假设检验的统一框架，基于核岭回归构建了有限样本保证的条件两样本检验，首次支持非i.i.d.数据与在线采样场景。

**[Lagrangian Neural Odes Measuring The Existence Of A Lagrangian With Helmholtz Me](others/lagrangian_neural_odes_measuring_the_existence_of_a_lagrangian_with_helmholtz_me.md)**

:   提出 Helmholtz metrics——基于 Helmholtz 条件的可微度量，用于量化给定 ODE 与 Euler-Lagrange 方程的接近程度，并将其作为正则化项加入二阶 Neural ODE 训练中，形成 Lagrangian Neural ODE，在零额外推理开销下引导模型收敛到真正的物理定律。

**[Learning-Augmented Online Bipartite Fractional Matching](others/learning-augmented_online_bipartite_fractional_matching.md)**

:   本文提出了两个学习增强算法（LAB 和 PAW），用于在线二部分数匹配问题，在给定可能不准确的建议匹配的情况下，首次在整个鲁棒性范围内 Pareto 优于朴素的 CoinFlip 策略。

**[Learning-Augmented Streaming Algorithms For Correlation Clustering](others/learning-augmented_streaming_algorithms_for_correlation_clustering.md)**

:   提出了首个面向相关聚类（Correlation Clustering）的学习增强流算法，利用成对距离预测，在完全图上实现优于3的近似比（$\tilde{O}(n)$ 空间），在一般图上实现 $O(\log|E^-|)$ 近似比（$\tilde{O}(n)$ 空间），在空间-近似比权衡上显著改进了已有的非学习算法。

**[Learning Approximately Equivariant Networks Via Constrained Optimization](others/learning_approximately_equivariant_networks_via_constrained_optimization.md)**

:   提出ACE（Adaptive Constrained Equivariance）框架，将等变神经网络训练建模为约束优化问题，通过对偶方法自动从灵活的非等变模型渐进过渡到等变模型，无需手动调参即可适应完全和部分对称数据。

**[Learning Dynamics Of Rnns In Closed-Loop Environments](others/learning_dynamics_of_rnns_in_closed-loop_environments.md)**

:   从数学理论上揭示了 RNN 在闭环（agent-环境交互）与开环（监督学习）训练下呈现根本不同的学习动力学，闭环学习遵循三阶段过程，由短期策略改进与长期稳定性之间的竞争驱动。

**[Learning Non-Equilibrium Diffusions With Schrödinger Bridges From Exactly Solvab](others/learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)**

:   将Schrödinger桥问题从布朗运动参考过程推广到多变量Ornstein-Uhlenbeck（mvOU）参考过程，推导高斯情形精确解，并提出无模拟的mvOU-OTFM算法处理一般分布。

**[Learning To Condition A Neural Heuristic For Scalable Mpe Inference](others/learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)**

:   提出 Learning to Condition (L2C)，通过训练注意力网络从求解器搜索轨迹中学习变量-值对的"最优性"与"简化性"双重评分，用于指导概率图模型中 MPE 推理的条件化决策，在高树宽模型上大幅缩减搜索空间且维持或提升解质量。

**[Look-Ahead Reasoning On Learning Platforms](others/look-ahead_reasoning_on_learning_platforms.md)**

:   在学习平台的用户-算法交互中形式化 level-$k$ 前瞻推理，证明个体自私的高阶推理只加速收敛但不改变均衡（无长期收益），而集体协调的收益由学习者-用户效用函数的对齐程度决定，提供了刻画协调收益上界的理论框架。

**[Maszero Designing Multiagent Systems With Zero Supervision](others/maszero_designing_multiagent_systems_with_zero_supervision.md)**

:   MAS-ZERO 是首个推理时自动 MAS 设计框架，通过 meta-agent 迭代设计、批评和改进 MAS 配置（包括任务分解和 sub-MAS 分配），无需验证集和训练，在推理（+16.69%）、编程（+16.66%）和搜索代理（+5.45%）任务上均超越手动和自动 MAS baseline，同时保持 Pareto 最优的准确率-成本权衡。

**[Maxsup Overcoming Representation Collapse In Label Smoothing](others/maxsup_overcoming_representation_collapse_in_label_smoothing.md)**

:   通过解析 Label Smoothing (LS) 的损失函数，发现其包含一个在错误分类时放大错误的"误差放大项"，导致类内特征坍缩；提出 Max Suppression (MaxSup) 方法，将惩罚目标从 ground-truth logit 转移至 top-1 logit，消除误差放大效应同时保留有益正则化。

**[Megstate Phoneme Decoding From Magnetoencephalography Signals](others/megstate_phoneme_decoding_from_magnetoencephalography_signals.md)**

:   提出 MEGState，一种融合多分辨率卷积和传感器级 SSM 的架构，用于从脑磁图(MEG)信号中解码音素，在 LibriBrain 数据集上显著超越基线方法。

**[Meta-Learning Three-Factor Plasticity Rules For Structured Credit Assignment Wit](others/meta-learning_three-factor_plasticity_rules_for_structured_credit_assignment_wit.md)**

:   本文提出一种元学习框架，通过外层梯度优化自动发现局部的新赫布式突触可塑性规则，使循环神经网络仅利用稀疏延迟奖励信号就能完成结构化的信用分配，为理解生物神经网络的学习机制提供了新视角。

**[Metafind Scene-Aware 3D Asset Retrieval For Coherent Metaverse Scene Generation](others/metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)**

:   MetaFind 是一个场景感知的三模态（文本+图像+点云）3D 资产检索框架，通过引入 SE(3) 等变的空间-语义图神经网络 (ESSGNN) 编码场景布局信息，实现了在元宇宙场景生成中风格一致、空间合理的迭代式资产检索。

**[Micadangelo Fine-Grained Reconstruction Of Constrained Cad Models From 3D Scans](others/micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)**

:   MiCADangelo 模拟人类 CAD 设计师的逆向工程流程，通过多平面截面分析提取 2D 模式，预测带约束的参数化草图并优化拉伸参数，首次在 3D CAD 逆向工程中实现了包含草图约束的完整参数化模型重建。

**[Military Ai Needs Technically-Informed Regulation To Safeguard Ai Research And I](others/military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)**

:   本文提出 AI-LAWS（AI 驱动致命性自主武器系统）的行为导向定义与监管框架，通过两条技术准则识别需特别监管的军事 AI 系统，并提出五项具体政策建议，呼吁 AI 研究者深度参与军事 AI 治理的全生命周期。

**[Modeling Cell Dynamics And Interactions With Unbalanced Mean Field Schrödinger B](others/modeling_cell_dynamics_and_interactions_with_unbalanced_mean_field_schrödinger_b.md)**

:   提出 Unbalanced Mean Field Schrödinger Bridge (UMFSB) 框架和 CytoBridge 深度学习算法，从稀疏时间快照数据中同时建模细胞的非平衡随机动力学和细胞间交互。

**[Modeling Neural Activity With Conditionally Linear Dynamical Systems](others/modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)**

:   提出条件线性动力系统（CLDS），通过高斯过程先验让线性动力系统参数随观测到的实验协变量非线性变化，在保留线性模型可解释性和高效推断的同时建模神经回路的非线性动态。

**[Moesd Unveil Speculative Decodings Potential For Accelerating Sparse Moe](others/moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)**

:   挑战"投机解码对MoE无效"的传统认知，理论与实验证明在中等batch size下MoE反而比稠密模型更受益于投机解码，提出target efficiency这一系统级指标来量化加速瓶颈，并构建了可靠的性能预测模型，在Qwen2-57B-A14B上实现最高2.29×加速。

**[Mutualvpr A Mutual Learning Framework For Resolving Supervision Inconsistencies ](others/mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)**

:   提出 MutualVPR 互学习框架，通过特征驱动的自适应 K-means 聚类动态分配场景类别标签，解决分类式 VPR 方法中由视角变化和遮挡导致的监督不一致问题。

**[Neural Network For Simulating Radio Emission From Extensive Air Showers](others/neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)**

:   用简单全连接神经网络替代计算昂贵的 CoREAS 蒙特卡洛模拟，快速预测广延大气簇射（EAS）的射电脉冲，并在 $X_{\text{max}}$ 重建任务中达到与传统模拟可比的分辨率。

**[Non-Clairvoyant Scheduling With Progress Bars](others/non-clairvoyant_scheduling_with_progress_bars.md)**

:   引入"进度条"信息模型作为透视与非透视调度之间的插值框架，针对对抗性和随机性进度条分别设计了具有最优一致性-鲁棒性权衡的调度算法，同时推进了学习增强调度的理论前沿。

**[Normalization In Attention Dynamics](others/normalization_in_attention_dynamics.md)**

:   将不同归一化方案（Post-LN、Pre-LN、Mix-LN、Peri-LN、nGPT、sqrt-scaling）统一建模为球面上交互粒子系统的速度调节机制，从理论上揭示了各方案对 token 聚类动力学和表示坍缩的不同影响，识别 Peri-LN 为理想选择。

**[On A Geometry Of Interbrain Networks](others/on_a_geometry_of_interbrain_networks.md)**

:   本文是一篇观点论文（opinion piece），提出将离散图曲率（Forman-Ricci 和 Ollivier-Ricci 曲率）引入超扫描（hyperscanning）研究中的脑间网络分析，利用曲率分布的熵来检测网络相变，并通过曲率值推断脑间信息路由策略，突破传统相关性指标的描述性局限。

**[On Agnostic Pac Learning In The Small Error Regime](others/on_agnostic_pac_learning_in_the_small_error_regime.md)**

:   本文在不可知 PAC 学习的小误差域（$\tau \approx d/m$）中，构造了一个基于 ERM 聚合的计算高效学习器，实现了 $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ 的误差上界（$c \leq 2.1$），匹配了已知下界，推进了不可知学习的精确复杂度刻画。

**[On The Surprising Effectiveness Of Large Learning Rates Under Standard Width Sca](others/on_the_surprising_effectiveness_of_large_learning_rates_under_standard_width_sca.md)**

:   揭示在标准参数化(SP)下，cross-entropy 损失函数使得"不稳定"区间实际分为灾难性不稳定和受控发散两个子区间：在受控发散区间（学习率 $\eta_n = \Theta(n^{-1/2})$）logits 发散但梯度和激活保持稳定，从而首次为 SP 提供了一个实用的、具有特征学习能力的无穷宽极限。

**[On Topological Descriptors For Graph Products](others/on_topological_descriptors_for_graph_products.md)**

:   系统研究在图的（box）积上施加各种滤过时拓扑描述子（欧拉特征 EC 和持续同调 PH）的表达能力，证明 PH 图积描述子严格强于对单图的计算，而 EC 不具备此性质，并给出高效 PH 计算算法。

**[On Universality Classes Of Equivariant Networks](others/on_universality_classes_of_equivariant_networks.md)**

:   本文证明等变神经网络的分离能力（区分对称等价输入的能力）不足以完全刻画其表达能力——具有相同分离能力的模型可能拥有不同的逼近能力，并给出了浅层不变网络通用性类的完整刻画及失败的充分条件。

**[One Sample Is Enough To Make Conformal Prediction Robust](others/one_sample_is_enough_to_make_conformal_prediction_robust.md)**

:   提出 RCP1（单样本鲁棒共形预测），通过认证共形过程本身而非单个 conformity score，仅需一次随机扰动前向传播即可获得比需要 100 次前向传播的 SOTA 方法更小的鲁棒预测集。

**[Optimism Without Regularization Constant Regret In Zero-Sum Games](others/optimism_without_regularization_constant_regret_in_zero-sum_games.md)**

:   首次证明无正则化的Optimistic Fictitious Play在2×2零和博弈中获得O(1)常数遗憾，匹配了正则化Optimistic FTRL的最优率，同时证明Alternating Fictitious Play的遗憾下界为Ω(√T)，分离了乐观和交替在无正则化情况下的能力。

**[Optimized Learned Count-Min Sketch](others/optimized_learned_count-min_sketch.md)**

:   提出 OptLCMS，通过将分数空间分区并用 KKT 条件解析求解 CMS 参数、动态规划优化阈值，大幅加速构建过程，同时提供不可容忍误差概率的理论保证。

**[Orbitzoo Real Orbital Systems Challenges For Reinforcement Learning](others/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)**

:   提出 OrbitZoo，一个基于工业级天体动力学库 Orekit 的多智能体 RL 环境，集成高保真轨道动力学（含大气阻力、太阳辐射压、三体效应等）、PettingZoo 多智能体接口和实时 3D 可视化，在 Starlink 真实星历验证中均值 MAPE 仅 0.16%。

**[Ortholoc Uav 6-Dof Localization And Calibration Using Orthographic Geodata](others/ortholoc_uav_6-dof_localization_and_calibration_using_orthographic_geodata.md)**

:   OrthoLoC构建了首个面向正射地理数据（DOP+DSM）的大规模UAV 6-DoF定位基准数据集，包含16425张真实UAV图像覆盖德国和美国47个区域，并引入AdHoP（自适应单应性预处理）匹配改进技术，在不修改特征匹配器的情况下将匹配性能提升95%、平移误差降低63%。

**[Overfitting In Adaptive Robust Optimization](others/overfitting_in_adaptive_robust_optimization.md)**

:   揭示自适应鲁棒优化（ARO）中策略脆弱性与机器学习过拟合的类比关系：自适应策略在不确定性集内表现优异但集外易失效，提出约束特定的不确定性集大小作为"正则化"手段来平衡鲁棒性和自适应性。

**[Polymath Evaluating Mathematical Reasoning In Multilingual Contexts](others/polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)**

:   提出Value-Guided Search(VGS)——通过token级价值模型指导块级束搜索，无需预定义"步骤"，相对多数投票在竞赛数学上准确度提升+14.5%，同时推理计算效率提升30%，超越现有PRM方案。

**[Position There Is No Free Bayesian Uncertainty Quantification](others/position_there_is_no_free_bayesian_uncertainty_quantification.md)**

:   本文从频率学派视角质疑贝叶斯不确定性量化（UQ）的有效性，将贝叶斯更新重新解释为模型集成的优化问题，并提出基于PAC框架的校准算法以构建具有频率学派保证的预测区间。

**[Prediction-Powered Semi-Supervised Learning With Online Power Tuning](others/prediction-powered_semi-supervised_learning_with_online_power_tuning.md)**

:   将预测驱动推断（PPI）框架扩展到半监督学习训练过程中，提出无偏梯度估计器，并设计在线AdaGrad算法动态调节伪标签与真实标签的相对权重 $\lambda$，在保证无偏性的同时实现与最优固定 $\lambda$ 匹配的收敛速率。

**[Private Evolution Converges](others/private_evolution_converges.md)**

:   为Private Evolution（PE）合成数据生成算法提供了首个不依赖不现实假设的收敛性理论保证，证明在正确的超参数设置下PE输出的 $(ε,δ)$-DP 合成数据集的1-Wasserstein距离为 $\tilde{O}(d(nε)^{-1/d})$。

**[Product Distribution Learning With Imperfect Advice](others/product_distribution_learning_with_imperfect_advice.md)**

:   本文研究在给定不完美建议分布的情况下学习布尔超立方体上乘积分布的问题，提出了一种高效算法，当建议质量足够好时样本复杂度可实现关于维度 $d$ 的次线性依赖。

**[Radar Benchmarking Language Models On Imperfect Tabular Data](others/radar_benchmarking_language_models_on_imperfect_tabular_data.md)**

:   提出 Radar 基准，通过对真实表格数据注入五类数据工件（缺失值、错误值、异常值、格式不一致、逻辑不一致），系统评估语言模型在不完美表格数据上的数据感知推理能力，揭示即使是前沿模型在引入数据工件后性能也大幅下降。

**[Recurrent Self-Attention Dynamics An Energy-Agnostic Perspective From Jacobians](others/recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)**

:   本文从动力系统的 Jacobian 分析视角，突破传统能量函数框架的对称性约束，揭示了归一化层在抑制自注意力谱范数和振荡分量方面的关键作用，发现高性能循环自注意力模型的 Lyapunov 指数趋近于零（临界态），并基于此提出谱正则化方法显著提升推理性能。

**[Redundancy-Aware Test-Time Graph Out-Of-Distribution Detection](others/redundancy-aware_test-time_graph_out-of-distribution_detection.md)**

:   提出 RedOUT 框架，通过最小化结构熵构建编码树来消除图结构中的冗余信息，结合冗余感知图信息瓶颈(ReGIB)原理，在测试时无需修改预训练模型参数即可有效区分ID和OOD图样本，在10个数据集对上平均AUC达87.46%。

**[Regression Trees Know Calculus](others/regression_trees_know_calculus.md)**

:   揭示常叶回归树中隐含的梯度信息——通过相邻节点均值差的有限差分类比，高效提取梯度估计，进而将活跃子空间（Active Subspace）和集成梯度（Integrated Gradient）等微分工具引入树模型，拓展了树模型的可解释性和预测改进能力。

**[Reliable Active Learning From Unreliable Labels Via Neural Collapse Geometry](others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)**

:   提出 NCAL-R，利用深度网络训练后期涌现的 Neural Collapse 几何结构，设计类均值对齐扰动（CMAP）和特征波动（FF）两个评分指标来选择样本，使主动学习在标签噪声和分布偏移下更加可靠，在 ImageNet-100 和 CIFAR-100 上一致优于传统 AL 基线。

**[Research Learning To Reason With Search For Llms Via Reinforcement Learning](others/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)**

:   ReSearch框架将搜索操作嵌入推理链中作为第一类原语，通过GRPO强化学习自动学习何时何如搜索，无需任何推理步骤的监督标注，在多跳QA任务上相对基线平均提升15.81%。

**[Resnets Are Deeper Than You Think](others/resnets_are_deeper_than_you_think.md)**

:   证明残差网络与前馈网络居于不同的函数空间（非简单重参数化），并通过后训练部分线性化实验表明变深度架构（类ResNet）即使在排除可训练性差异后仍优于固定深度架构，暗示残差连接提供了超越优化的归纳偏好。

**[Rethinking Pca Through Duality](others/rethinking_pca_through_duality.md)**

:   通过 Difference-of-Convex (DC) 框架重新审视 PCA,建立了核化和样本外推广能力,揭示了同步迭代是 DCA 的特例,并提出了鲁棒 $\ell_1$-PCA 的核化对偶公式。

**[Revisiting Agnostic Boosting](others/revisiting_agnostic_boosting.md)**

:   提出新的不可知 Boosting 算法,在非常一般的假设下大幅改善了此前工作的样本复杂度,并建立近匹配下界,从而在对数因子意义下解决了不可知 Boosting 的样本复杂度问题。

**[Rnns Perform Task Computations By Dynamically Warping Neural Representations](others/rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)**

:   本文提出一个黎曼几何框架，通过将表示空间度量从 RNN 状态空间拉回（pullback）到输入流形上，证明 RNN 通过动态变形（warping）其对任务变量的表示来执行计算——压缩无关输入、拉伸决策边界附近的空间，且这种变形不是副产物而是计算本身。

**[Robust Sampling For Active Statistical Inference](others/robust_sampling_for_active_statistical_inference.md)**

:   提出基于预算保持路径的鲁棒采样策略，通过在均匀采样和主动采样之间最优插值，确保估计器的方差永远不比两者中任何一个更差，解决了主动统计推断中不确定性估计不准确导致性能恶化的问题。

**[Sad Neural Networks Divergent Gradient Flows And Asymptotic Optimality Via O-Min](others/sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)**

:   利用 o-minimal 结构的数学工具，证明了使用常见光滑激活函数（sigmoid、tanh、softplus、GELU 等）的全连接网络的梯度流存在二元性：要么收敛到临界点，要么发散到无穷大且损失收敛到渐近临界值。特别地，对多项式目标函数，证明了损失无法精确取零但可任意接近零，从而导致参数必然发散。

**[Sample-Adaptivity Tradeoff In On-Demand Sampling](others/sample-adaptivity_tradeoff_in_on-demand_sampling.md)**

:   系统研究了按需采样（on-demand sampling）中样本复杂度与自适应轮次之间的权衡关系，在可实现设定下证明 $r$ 轮算法的最优样本复杂度为 $dk^{\Theta(1/r)}/\varepsilon$，在不可知设定下提出仅需 $\widetilde{O}(\sqrt{k})$ 轮即可达近最优样本复杂度的LazyHedge算法，并引入OODS抽象框架建立了近紧的轮次复杂度下界。

**[Scalable Gpu-Accelerated Euler Characteristic Curves Optimization And Differenti](others/scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)**

:   提出面向现代 Ampere GPU 优化的欧拉特征曲线（ECC）CUDA 内核，相比先前 GPU 实现达到 16-2000x 加速，并引入可微 PyTorch 层通过 DECT 风格的 sigmoid 松弛支持在密集网格图像上的端到端拓扑特征学习。

**[Scalable Inference Of Functional Neural Connectivity At Submillisecond Timescale](others/scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)**

:   将传统离散时间Poisson GLM推广到连续时间Poisson点过程，通过蒙特卡洛采样和二阶多项式近似两种方法绕过不可解的积分项，配合正交的广义Laguerre基函数，在数百神经元、数千秒记录的数据上实现分钟级训练和亚毫秒级突触连接识别。

**[Semi-Infinite Nonconvex Constrained Min-Max Optimization](others/semi-infinite_nonconvex_constrained_min-max_optimization.md)**

:   针对带有无穷多非凸约束的非凸 min-max 优化问题，提出 iDB-PD（不精确动态障碍原始-对偶）算法，在 Łojasiewicz 正则条件下建立了首个全局非渐近收敛保证，稳定性 $\mathcal{O}(\epsilon^{-3})$、可行性 $\mathcal{O}(\epsilon^{-6\theta})$、互补松弛 $\mathcal{O}(\epsilon^{-3\theta/(1-\theta)})$。

**[Semi-Supervised Graph Anomaly Detection Via Robust Homophily Learning](others/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)**

:   提出RHO (Robust Homophily Learning)方法，通过自适应频率响应滤波器(AdaFreq)和图正常性对齐(GNA)模块，解决半监督图异常检测中正常节点同质性多样性的问题，在8个真实数据集上超越现有方法。

**[Sheaf Cohomology Of Linear Predictive Coding Networks](others/sheaf_cohomology_of_linear_predictive_coding_networks.md)**

:   本文将线性预测编码（PC）网络形式化为细胞层（cellular sheaf），证明PC推理等价于层Laplacian下的扩散过程，通过Hodge分解将监督信号拆解为可消除误差（通过推理）和不可约误差（由循环拓扑的上同调刻画），从而精确解释了为什么某些循环权重初始化会导致学习停滞。

**[Sign-In To The Lottery Reparameterizing Sparse Training From Scratch](others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)**

:   本文发现稀疏网络从头训练(PaI)性能差的根本原因是无法像dense-to-sparse方法那样学习正确的参数符号，为此提出Sign-In重参数化方法（θ=m⊙w），通过引入内部自由度来促进符号翻转，理论证明其能解决一种互补于过参数化的符号翻转情况，实验中显著提升了稀疏从头训练的性能。

**[Smrs Advocating A Unified Reporting Standard For Surrogate Models In The Artific](others/smrs_advocating_a_unified_reporting_standard_for_surrogate_models_in_the_artific.md)**

:   本文针对AI驱动的代理模型（Surrogate Model）领域缺乏标准化报告规范的痛点，提出了一套轻量级、模块化、与模型无关的报告标准SMRS，覆盖数据采集、模型选择、训练方法、评估指标等完整建模流水线的六大维度，通过对17篇已发表论文的案例研究验证了框架的可操作性，旨在提升代理模型的可复现性、可比较性和跨领域迁移能力。

**[Space Spike-Aware Consistency Enhancement For Test-Time Adaptation In Spiking Ne](others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)**

:   提出SPACE，首个专为脉冲神经网络(SNN)设计的无源单样本测试时自适应(TTA)方法，通过最大化增强样本间脉冲行为特征图的一致性，在多个数据集和架构上实现鲁棒适应。

**[Stable Matching With Ties Approximation Ratios And Learning](others/stable_matching_with_ties_approximation_ratios_and_learning.md)**

:   研究有并列偏好的双边匹配市场，提出最优稳定份额(OSS)比率概念衡量公平性，证明稳定匹配分布下OSS-ratio为$\Omega(N)$但一般匹配分布下可达$O(\log N)$（渐近紧），并将离线近似结果扩展到bandit学习场景。

**[Statistical Inference For Gradient Boosting Regression](others/statistical_inference_for_gradient_boosting_regression.md)**

:   提出统一的梯度提升回归统计推断框架，通过将dropout和并行训练整合到Boulevard正则化中，证明了相应的中心极限定理，从而构建了内置的置信区间、预测区间和变量重要性假设检验，并发现增大dropout率和并行树数量能显著提升信号恢复（最高达2倍和4倍）。

**[Statistical Inference Under Performativity](others/statistical_inference_under_performativity.md)**

:   本文首次建立了表演性预测（performative prediction）下完整的端到端统计推断框架，为重复风险最小化算法推导出中心极限定理和数据驱动的协方差估计方法，并将预测驱动推断（PPI）扩展到动态表演性设置以获得更紧的置信区间。

**[The Computational Complexity Of Counting Linear Regions In Relu Neural Networks](others/the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)**

:   系统梳理了ReLU网络"线性区域"的六种非等价定义，证明对所有定义计数线性区域都是#P-hard的（一层隐藏层即如此），并在多层网络中证明了强不可近似结果和多项式空间上界。

**[The Cost Of Robustness Tighter Bounds On Parameter Complexity For Robust Memoriz](others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)**

:   研究 ReLU 网络鲁棒记忆（robust memorization）的参数复杂度，即在保证每个训练样本 $\mu$-邻域内预测一致的条件下插值任意数据集所需的参数数量，在鲁棒性比率 $\rho = \mu/\epsilon$ 的全范围 $(0,1)$ 内建立了更紧的上下界。

**[The Parameterized Complexity Of Computing The Vc-Dimension](others/the_parameterized_complexity_of_computing_the_vc-dimension.md)**

:   本文系统研究了计算VC维问题的参数化复杂性，证明朴素穷举算法在ETH假设下是渐近最优的，给出按最大度参数化的FPT 1-可加近似算法，以及按树宽参数化的 $2^{O(\text{tw} \cdot \log \text{tw})} \cdot |V|$ 精确算法，并完整刻画了各结构参数下的可处理性景观。

**[The Persistence Of Neural Collapse Despite Low-Rank Bias](others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)**

:   本文从理论上证明了深度神经坍缩（DNC）在深层无约束特征模型中由于 L2 正则化引起的低秩偏差而全局次优，同时首次解释了 DNC 在实践中持续出现的原因——其解空间维度随网络宽度增长快于低秩解。

**[The Structural Complexity Of Matrix-Vector Multiplication](others/the_structural_complexity_of_matrix-vector_multiplication.md)**

:   证明对于 corrupted VC-dimension 为 $d$ 的布尔矩阵 $\mathbf{M} \in \{0,1\}^{m \times n}$，矩阵-向量乘法可在 $\widetilde{O}(nm^{1-1/d}+m)$ 时间内完成，首次为结构化矩阵提供了真亚二次时间上界，推翻了 OMv 猜想在结构化输入上的适用性，并导出了动态 Laplacian 求解器、有效电阻、三角检测等问题的首个高精度亚二次算法。

**[Tight Bounds On The Distortion Of Randomized And Deterministic Distributed Votin](others/tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)**

:   本文研究分布式投票模型中的度量扭曲 (metric distortion) 问题，针对四种代价目标 ($\text{avg-avg}$, $\text{avg-max}$, $\text{max-avg}$, $\text{max-max}$)，在确定性和随机机制下给出了改进的紧界或近紧界，几乎完整地刻画了这一模型的扭曲特性。

**[Training The Untrainable Introducing Inductive Bias Via Representational Alignme](others/training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)**

:   提出Guidance方法，通过逐层表征对齐（CKA）将一个网络（guide）的架构归纳偏置迁移到另一个原本"不可训练"的网络（target），从而使FCN能做图像分类、RNN逼近Transformer的语言建模性能。

**[Transfer Learning For Benign Overfitting In High-Dimensional Linear Regression](others/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)**

:   提出两步式Transfer MNI方法，在高维过参数化线性回归中通过"保留目标信号+零空间迁移源知识"机制增强良性过拟合的泛化能力，刻画了模型偏移和协变量偏移下的非渐近excess risk，并发现了"免费午餐"协变量偏移区间。

**[Uncertainty Estimation By Flexible Evidential Deep Learning](others/uncertainty_estimation_by_flexible_evidential_deep_learning.md)**

:   提出 $\mathcal{F}$-EDL，通过将 EDL 中的 Dirichlet 分布推广为 Flexible Dirichlet (FD) 分布来建模类别概率分布，从而在保持单次前向传播效率的同时，显著增强不确定性估计在噪声、长尾、分布偏移等复杂场景下的泛化能力。

**[Uncertainty Quantification For Reduced-Order Surrogate Models Applied To Cloud M](others/uncertainty_quantification_for_reduced-order_surrogate_models_applied_to_cloud_m.md)**

:   提出首个面向潜空间降阶模型的后验、模型无关不确定性量化框架，利用共形预测分别对重建、潜在动力学和端到端预测构建分布无关的预测区间，揭示了云微物理ROM中不确定性的组件级传播规律——自编码器结构性误差而非动力学误差主导端到端预测不确定性。

**[Uniformer Unified And Efficient Transformer For Reasoning Across General And Cus](others/uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)**

:   提出 UniFormer，一种面向 GPU 和 FPGA 跨平台部署的统一高效 Transformer 架构，通过双分支注意力机制（全局线性注意力 + 局部块注意力）实现了高并行性和计算存储融合。

**[Variational Regularized Unbalanced Optimal Transport Single Network Least Action](others/variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)**

:   提出 Var-RUOT，通过将正则化非平衡最优传输（RUOT）问题的最优性必要条件融入参数化和损失设计，仅需学习单个标量场即可求解 RUOT，获得更低作用量的解并提升训练稳定性；同时分析了增长惩罚函数对生物先验的影响。

**[Webthinker Empowering Large Reasoning Models With Deep Research Capability](others/webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)**

:   WebThinker赋予大型推理模型(LRM)自主的网络搜索与导航能力，通过Think-Search-Draft策略实现推理、信息采集与报告生成的无缝交织，经RL优化后在复杂推理与科学报告生成任务上超越o1与Gemini。

**[Weight Weaving Parameter Pooling For Data-Free Model Merging](others/weight_weaving_parameter_pooling_for_data-free_model_merging.md)**

:   本文提出Weight Weaving，一种即插即用的无数据模型合并增强方法，通过在缩放因子搜索空间上对模型参数进行池化操作（如平均、随机选择），消除了对评估数据的依赖，在多任务学习、持续学习和域泛化三个场景中平均准确率最高提升15.9个百分点。

</div>