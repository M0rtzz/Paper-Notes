---
title: >-
  410 篇 ICML2026 论文解读 · 每篇 5 分钟读懂
description: >-
  410篇ICML2026论文解读，涵盖多模态 VLM(30篇)、图像生成(22篇)、可解释性(21篇)、模型压缩(21篇)、LLM 推理(20篇)、强化学习(20篇)、科学计算(19篇)、LLM 安全(18篇)等 40个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "多模态 VLM"
  - "图像生成"
  - "可解释性"
  - "模型压缩"
  - "LLM 推理"
  - "强化学习"
  - "科学计算"
  - "LLM 安全"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧪 ICML2026 论文笔记

410篇ICML2026论文解读，涵盖多模态 VLM(30篇)、图像生成(22篇)、可解释性(21篇)、模型压缩(21篇)、LLM 推理(20篇)、强化学习(20篇)、科学计算(19篇)、LLM 安全(18篇)等 40个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](multimodal_vlm/any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)**

:   作者通过 pilot study 发现"显式把视觉提升到点云、再与 2D patch 融合"是 VLA 注入 3D 信息的最有效方式；为了解决 3D 数据稀缺和不同点云源（仿真/传感器/单目估计）的域差异，提出 Any3D-VLA：用 hybrid point cloud training 学到 source-agnostic 的几何表示，在真实抓取任务上 zero-shot 比最强 baseline 提升 29.2%（62.5% vs 33.3%）。

**[Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](multimodal_vlm/bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)**

:   本文把 VLM 的输出强制拆成 `<recognition>` 感知块和 `<think>` 推理块，再用一个"蒙眼"文本推理代理（拿不到图，只看 VLM 写下的感知文字）能不能答对题作为感知奖励 $R_P$，配上结构化语言验证 SVV 作为结果奖励 $R_O$；MoCA 用 $R_P$ 当门控做模态级信用分配，让 7B 模型在 9 个 perception/reasoning/rich-modality benchmark 上同时提升，在多个指标上超过 GPT-4o。

**[Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](multimodal_vlm/breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)**

:   针对统一多模态模型 (unified model) 在 anything-to-image (X2I) 任务上的"理解–生成 gap"（看得懂但生不出），本文提出 Self-Adaptive Interleaved Reasoner：用一个 hierarchical 数据合成 pipeline 在直接生成 / 自我反思 / 多步规划三种模式间分流 5 万条样本，再用 SFT + GRPO 训练并配上 step-wise 推理奖励和 intra-group 复杂度惩罚，让 Emu3.5 在 KRIS-Bench / OmniContext 上超越 GPT-4o、Gemini 2.5 Flash 等闭源模型。

**[Calibrated Multimodal Representation Learning with Missing Modalities](multimodal_vlm/calibrated_multimodal_representation_learning_with_missing_modalities.md)**

:   针对"想用 V-T、A-T 等部分模态数据训练统一多模态对齐"这种现实场景，本文用奇异值扰动给出"缺失模态会导致 anchor shift"的理论上下界，并提出 CalMRL：用概率 PCA 风格的生成模型对缺失模态在表示层做闭式 EM 插补，再把观测 + 插补一起喂给 GRAM/PMRL 的 SVD 对齐目标，在 VAST 之上把跨模态平均 Recall@1 从 44.8 推到 54.2 (+9.4)。

**[DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction](multimodal_vlm/dcer_dual-stage_compression_and_energy-based_reconstruction.md)**

:   DCER 把"模态内频域压缩 + 跨模态 bottleneck token"作为统一的鲁棒融合管道，并用一个学习的能量函数对缺失模态做梯度下降式重建，同时把最终能量值当作内蕴的不确定度，在 MOSI/MOSEI/SIMS 上同时刷新 SOTA。

**[FreeRet: MLLMs as Training-Free Retrievers](multimodal_vlm/freeret_mllms_as_training-free_retrievers.md)**

:   FreeRet 提出一个完全不训练的两阶段多模态检索框架：第一阶段绕过 MLLM 最后一层 MLP 并配合受控生成 prompt 抽取语义忠实的 embedding 做候选检索，第二阶段把 reranking 改成多项选择题来规避 LLM 的 framing 偏置；在 MMEB 上比训练了千万级配对数据的检索模型还要强。

**[FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](multimodal_vlm/frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)**

:   FRISM 把「VLM × LRM 合并」从层级粒度细化到 SVD 子空间粒度：用 LRM 任务向量的 SVD 子空间作为推理先验，再用一个仅含可学习门控的无标签自蒸馏（KL 保视觉 + 谱幅最大化吸收推理）找到最优注入强度，从而在不显著掉视觉的前提下显著提升 VL 推理性能。

**[Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](multimodal_vlm/gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)**

:   本文用 Information Bottleneck 视角把量化感知训练 (QAT) 与知识蒸馏统一起来，提出 GRACE 框架（置信度门控解耦蒸馏 + 关系中心化核对齐 + 自适应 IB 控制器），让 INT4 量化的 LLaVA / Qwen-VL 不仅没掉点，反而在多个 benchmark 上超过 BF16 基线，同时实测 3× 吞吐 + 54% 显存节省。

**[Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](multimodal_vlm/injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)**

:   本文把 MLLM 的连续值回归在长尾分布下的"回归到均值"问题转化为分布感知的 RL 问题，在 GRPO 框架内用 Concordance Correlation Coefficient (CCC) 作为批次级奖励——既看相关性、又看方差、又看均值——从而显式惩罚预测分布塌缩；在 4 个长尾回归任务、Qwen2.5-VL-3B/7B 上稳压 SFT、SoftLabel、各种 point-wise RL，特别是 medium/few-shot 区域 MAE 大幅下降。

**[Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](multimodal_vlm/instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)**

:   本文发现 MLLM 中 instruction token 的中间层嵌入能天然过滤视觉端引入的误导信息，据此提出训练无关的 InsLen 分数（Calibrated Local Score + Context Consistency Score），在 5 个 MLLM × 4 个基准上把对象幻觉检测的 AUROC 拉高最多 13.81%。

**[Large Vision-Language Models Get Lost in Attention](multimodal_vlm/large_vision-language_models_get_lost_in_attention.md)**

:   本文用"信息复杂度 (eRank) + 子空间支持"的几何信息论框架定量诊断 LVLM 的残差流，发现 Attention 几乎只做子空间内重配置而 FFN 才注入新语义维度；更惊人的是把学习到的 attention 权重换成高斯噪声后多数视觉任务性能不降反升，揭示当代 LVLM 在 visual attention 上严重错配冗余。

**[Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](multimodal_vlm/learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)**

:   VISTA 把多模态大模型的自我改进训练改造成"难题靠 prefix 重采样补样本、伪正例靠视觉注意力分数 (VAS) 过滤"的两段式 pipeline，在 Qwen2.5-VL-3B 上把数学/医学多模态推理平均提升 +13.66%。

**[Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy](multimodal_vlm/less_precise_can_be_more_reliable_a_systematic_evaluation_of_quantizations_impac.md)**

:   这篇用 70 万次实验跑遍了 16 种量化方法 × 10 种 VLM × 多项可靠性指标，发现量化不是单纯破坏者——它会通过抑制高 rank 低方差的频谱分量，同时提升 calibration、OOD 检测和噪声鲁棒性，但也会放大对协变量偏移和虚假相关的依赖。

**[LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](multimodal_vlm/limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)**

:   作者把"训练阶段就缺模态"的多模态动作质量评估重新建模成"基于 LLM 的条件序列到分数推理"问题，用 prompt + 特殊 token 让 LLM 在没有完整数据监督的情况下补全缺失语义，再配合掩码感知的双路融合抑制幻觉，在三个 AQA 数据集上全面超越依赖完整训练数据的 SOTA。

**[Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](multimodal_vlm/model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)**

:   Model-Dowser 用"权重幅值 × 输入激活 × 输出 Jacobian"三因素给 MLLM 的每个参数打分，冻结高分参数、只更新低分参数，从而在 LLaVA/NVILA 上深层微调时既能学好下游任务又能保留预训练知识，相比 SPIDER、ModelTailor 在 H-score 上稳定领先。

**[Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_vlm/multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)**

:   针对 MLLM 在跨场景 VQA 中的视觉遗忘问题，本文构建 MSVQA（高空/水下/低空/室内 4 场景）基准，并提出 Unifier 框架——在视觉 block 里加入 CSR 多分支 + 投影器（VRE）做参数隔离，再用 KL 软约束（VCC）对齐不同分支表征，单次推理即可在 20 步持续学习上把 VQA 提升 2.70-10.62%、F1 提升 3.40-7.69%。

**[MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](multimodal_vlm/muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)**

:   MUSE 把统一视觉 tokenizer 的"理解-生成"零和困境归因于流形错配，提出梯度正交假设——把语义注入 $W_V$ 而结构梯度走 $W_{Q,K}$——并通过 Synergistic Block + DINOv3 拓扑对齐 + NCE 语义锚定彻底解耦，最终 gFID 3.08 与 linear probing 85.2%（甚至超过 InternViT-300M 老师 82.5%）共存，首次实现真正的"互相强化"而非折中。

**[OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](multimodal_vlm/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)**

:   本文指出现有 Omni-LLM token 压缩方法对音频和视频"对称"处理是次优的，提出 OmniSIFT——先用时空显著性剪掉视频冗余得到"视觉锚点"，再用这些锚点引导音频选择的两阶段非对称压缩框架，仅引入 4.85M 额外参数就在 Qwen2.5-Omni-7B 上保留 25% token 时一致超过现有压缩基线甚至原模型。

**[Perceptual Flow Network for Visually Grounded Reasoning](multimodal_vlm/perceptual_flow_network_for_visually_grounded_reasoning.md)**

:   摒弃"用视觉专家的精确框做硬监督"的传统 RLVR 路线，PFlowNet 把感知行为本身建模为一段结构化的 Perceptual Flow 潜变量，用变分分布 $p_\theta(Z|X)$ 近似面向推理的理想后验，并用 Sub-TB 变分 RL + 多维奖励 + 邻域几何整形 (Vicinal Geometric Shaping) 训练，使得 8B 的 Qwen3-VL 在 V* Bench 拿到 90.6%、MME-RealWorld-lite 67.0% 的新 SOTA。

**[Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](multimodal_vlm/referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)**

:   CSteer 提出一种训练无关的 latent steering 方法,通过在错误/正确指代回答的隐藏激活差上构造"上下文向量",并在推理时分层注入到 query 早期层和 decode 中后期层,让通用 LMM (Qwen3-VL、InternVL-3.5) 在多区域视觉指代任务上反超专门微调的 region LMM。

**[Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](multimodal_vlm/revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)**

:   本文把 LVLM 幻觉重新定义为"被语言先验压制的视觉信息缺失"，用正交投影从原始视觉方向中剔除语言先验得到"纯视觉向量"，再用风险门控只在最优深度的单层做稀疏干预，免训练地把 CHAIRS 幻觉率降 ~19% 同时保住 MM-Vet 通用能力。

**[ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](multimodal_vlm/revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)**

:   本文系统揭示了被广泛使用的 VSI-Bench 因 3D 标注漂移与帧采样不一致而存在结构性失效，进而重新标注 381 个场景、5365 个对象，并设计帧预算自适应 QA 与"删除查询对象帧"的 dummy 视频压力测试，构建出名为 ReVSI 的高保真空间智能基准；评估显示开源 VLM 在 ReVSI 上掉点最多 40%，且在 dummy 视频上幻觉率仍高，暴露出现有空间推理能力被 VSI-Bench 系统性高估。

**[ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision](multimodal_vlm/screenparse_moving_beyond_sparse_grounding_with_complete_screen_parsing_supervis.md)**

:   针对 GUI agent 普遍使用"稀疏 grounding"标注、丢失整屏结构的问题，本文用全自动 Webshot 流水线构建了 771K 截图 / 21M 元素 / 55 类的稠密屏幕解析数据集 ScreenParse，并训练出仅 316M 参数的 ScreenVLM 把整屏解析为 ScreenTag 结构序列，在密集解析与稀疏 grounding 多个 benchmark 上击败 8B 级别的基础 VLM 同时把延迟降到 $\sim 1/4$。

**[Seeing to Generalize: How Visual Data Corrects Binding Shortcuts](multimodal_vlm/seeing_to_generalize_how_visual_data_corrects_binding_shortcuts.md)**

:   本文用一个"颜色-形状-item"受控合成检索任务复现了"VLM 在纯文本任务上超过其 base LLM"的奇怪现象，并用机制可解释性证明：图像训练让模型把变量绑定策略从"位置捷径"切换到"语义符号匹配"，这一切换在重新接回纯文本后被保留下来，使 OOD 检索准确率从 37.2% 提升到 69.5%；在真实 Qwen2/2.5/3 家族上也观察到一致的"symbolic/positional 比例上升"。

**[Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](multimodal_vlm/self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)**

:   本文借助 Pointwise Partial Information Decomposition 量化视觉-文本模态交互，并提出 Multimodal Interaction Gate：自动挑出「图像独有信息占主导」的样本让 VLM 自我生成 caption 灌入文本侧，从而把 unique 视觉信号转成 redundant 共享信号，使 VLM 在模糊或被污染输入下的视觉幻觉下降 38.3%、一致性提高 16.8%。

**[SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](multimodal_vlm/slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)**

:   SLQ 把一小组"共享潜在查询" $\mathbf{Q}$ 追加到图像/文本 token 序列尾部，借助 MLLM 自身的因果注意力聚合全局上下文，**只训练几千个查询参数**就让冻结的 MLLM 变成检索器，在 COCO/Flickr30K 上胜过全量微调和 LoRA，并配套发布了考验"隐式知识推理"能力的 KARR-Bench。

**[The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](multimodal_vlm/the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)**

:   本文把"VLM 看不清细节"形式化为一个序贯贝叶斯最优实验设计 (S-BOED) 问题,并基于"覆盖率 × 分辨率"的可计算代理目标提出训练免费的 FOVEA 模块,在高分辨率/遥感等基准上稳定超过 Direct 与 ReAct-style 基线。

**[Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)**

:   本文提出 S3 框架，用 MoE 把多模态表征分解为概念级专家（Specialization）、按任务路由激活相关专家（Selection）、并在推理时按路由分数剪枝低贡献路径（Sparsification），在四个 MultiBench 基准上揭示了一条"性能在中间稀疏度达峰"的反 U 型曲线，给出对比学习/InfoMax 之外第三种多模态表征范式。

**[Vision-aligned Latent Reasoning for Multi-modal Large Language Model](multimodal_vlm/vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)**

:   本文提出 VaLR：在 MLLM 的 CoT 推理每一步之前插入若干"潜在 token"，并用 DINOv3/SigLIP/π³ 等视觉编码器的 patch 特征对这些 token 做表征对齐（REPA），从而在长链推理中持续把视觉信息"喂回"模型，把 Qwen2.5-VL 在 VSI-Bench 上的准确率从 33.0% 拉到 52.9%，并首次让 MLLM 表现出"推理越长越准"的 test-time scaling 行为。

**[What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](multimodal_vlm/what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)**

:   GLANCE 在 VLM agent 的强化学习里加了一个"想-看对齐"自监督头：让 LLM 在 CoT 里产出的"下一状态预测"通过一个轻量 projector 映射到由 EMA target 视觉编码器编码的真实下一帧表示，预测与实际之间的差距同时充当内在好奇心奖励、视觉编码器的训练信号、以及让 internalized world model "落地"的对齐损失；再配合周期性重置 projector 的课程探索机制对抗好奇心衰退，最终在 5 个 agentic 任务上稳定超过现有 exploitation-only 的 VLM-RL 方法。

---

## 🎨 图像生成 { #image_generation }

**[Adversarial Flow Models](image_generation/adversarial_flow_models.md)**

:   作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成，ImageNet-256 上 1NFE FID 刷到 2.38（XL/2）和 1.94（112 层）。

**[Anomaly-Preference Image Generation (APO)](image_generation/anomaly-preference_image_generation.md)**

:   作者把"少样本异常图像生成"重写为"无人工标注的偏好优化问题"：真实异常作为正样本，参考模型在同一时刻的去噪偏差作为隐式负样本，通过 DPO 风格 loss 让扩散模型对齐异常分布；再用按时间步调节 LoRA rank 的 TACA 保住结构多样性、用分层 CFG 调节文本-异常对齐强度，在 MVTec 等 benchmark 上同时刷新真实度和多样性。

**[Caracal: Causal Architecture via Spectral Mixing](image_generation/caracal_causal_architecture_via_spectral_mixing.md)**

:   Caracal 用 $\mathcal{O}(L \log L)$ 的多头傅立叶（MHF）模块替换 Transformer 的 $\mathcal{O}(L^2)$ 注意力，通过"pad-FFT-multiply-iFFT-truncate"实现频域内的严格因果掩码，并完全去掉位置编码，仅用标准 FFT 算子（不依赖 Mamba 那样的 CUDA kernel）就在 Tiny→Large 全尺度上与 Llama / Mamba / Mamba-2 / Jamba 性能相当。

**[CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](image_generation/card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)**

:   CARD 用"基数 $r$ 分解"把分子 3D 坐标双射映射为先粗后细的离散-连续混合 token 序列，让一个跨系统通用的自回归 Transformer 作为"零自由能 proposal"通过 BAR 直接估算任意分子系统的绝对自由能，在 70 个新系统的溶剂化任务上达到经典 MFES 的精度且推理快约 40 倍。

**[CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](image_generation/cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)**

:   本文针对"编辑模型常在不该改的区域乱改"这一痛点，构造 CoCoEdit-40K 局部编辑数据集 + 提出 pixel-level 相似度 reward 补充 MLLM reward + 设计区域正则化 RL 目标（高奖励样本约束非编辑区一致、低奖励样本强迫编辑区做出改变），把 FLUX.1 Kontext 和 Qwen-Image-Edit 同时在编辑得分和 PSNR/SSIM 上提升，打破现有"提编辑能力必伤一致性"的 trade-off。

**[Conditional Diffusion Sampling](image_generation/conditional_diffusion_sampling.md)**

:   本文提出 Conditional Diffusion Sampling（CDS）：通过推导一类条件随机插值（conditional interpolants），得到一个对未归一化目标分布的**精确闭式 SDE**（不需要神经网络拟合），再用 Parallel Tempering 高效采样这个 SDE 的初始分布——把 PT 的全局探索能力和扩散过程的局部细化能力拼起来，在 8 个目标分布、4 类任务上以更少的密度评估次数同时击败传统 MCMC、训练自由 MCMC 和神经采样器。

**[Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](image_generation/diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)**

:   论文用线性探针发现 MM-DiT (FLUX / SD3.5) 在中间层的某些注意力头里、其 text token 的 key 向量天然编码了"目标概念是否会出现"的二元信号，并由此提出 Omission Signal Intervention (OSI)：在 inference 时把"omission 类 - existence 类"的均值差方向以 $\alpha\sigma\boldsymbol{\theta}$ 的强度注入 Top-K 头的 key 向量，激发模型对缺失概念的"自我感知"并补全生成；在 FLUX 上 GenEval 6-object 准确率从 0.18 → 0.40，且无需任何 fine-tune。

**[End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)**

:   EOSTok 用单阶段端到端管线把 1D ViT tokenizer 和自回归模型一起训练，靠新提出的 APR（Autoregressive Prediction Reconstruction）loss 把「next-token 预测」的梯度真正传回 pixel space 防止码本崩塌，再用「隐式对齐」把 DINOv2 语义注入 1D 隐空间而不破坏 1D 自回归结构，最终在 ImageNet 256 上无 guidance 拿到 1.48 的 FID（SOTA）。

**[Exploring and Exploiting Stability in Latent Flow Matching](image_generation/exploring_and_exploiting_stability_in_latent_flow_matching.md)**

:   本文系统刻画了 Latent Flow Matching（LFM）的"轨迹稳定性"——同一噪声种子下，剪掉 75% 数据、换大小架构、改训练种子都能产生几乎相同的图像；进而把这个性质转化成两个实用算法：(1) 用 balanced-clustering 剪枝可在 CelebA-HQ 上把 50% 数据剪掉而 FID 反而轻微提升、ImageNet 上 75% 数据可剪；(2) Coarse-to-Fine 两段式生成，把 DiT-XL/2 (675M) 和 DiT-S/2 (33M) 拼起来，推理快 2.15×。

**[Factored Classifier-Free Guidance](image_generation/factored_classifier-free_guidance.md)**

:   本文识别出 CFG 在扩散模型反事实生成中存在「属性放大 (attribute amplification)」失效模式——单一全局 $\omega$ 会把本不该改变的属性一起放大，并提出 FCFG：按因果图分组、为每组属性分配独立 guidance 权重，从而在 CelebA-HQ / EMBED / MIMIC-CXR 上显著降低非目标属性漂移、改善反事实可逆性。

**[GenExam: A Multidisciplinary Text-to-Image Exam](image_generation/genexam_a_multidisciplinary_text-to-image_exam.md)**

:   GenExam 把"画图考试"作为衡量 T2I 模型推理-理解-生成综合能力的金标准，给 10 个学科、1000 道题各配上 ground-truth 图 + 细粒度评分点，结果连最强闭源模型 Nano Banana Pro 也只有 70.2% strict 分，多数开源 T2I/统一 MLLM 不到 3%。

**[Implicit Preference Alignment for Human Image Animation](image_generation/implicit_preference_alignment_for_human_image_animation.md)**

:   作者提出 Implicit Preference Alignment (IPA)：一种只需"好样本"、不需要构造好/坏配对的后训练方法，通过最大化与预训练参考模型 KL 间隔来等价地最大化隐式奖励，并配合一个把手部 mask 加权进损失的 HALO 模块，让大尺度视频 DiT 在仅 93 个挑选样本下显著改善人体动画的手部保真度。

**[Krause Synchronization Transformers](image_generation/krause_synchronization_transformers.md)**

:   作者把 Krause 有界置信共识模型搬进 Transformer，用"距离-RBF+局部窗+top-k 稀疏"替代全局 softmax 相似度，从理论上证明它鼓励多簇同步而非全局塌缩，并在 ViT / 自回归图像生成 / LLM 上同时获得更优性能和 30%+ 算力节省。

**[Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](image_generation/offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)**

:   本文针对 rectified flow（RF）类文生图模型，提出 PNAPO——一种把"生成时用的先验噪声"和"赢者/输者图片"一起保存为六元组的离线偏好优化框架，配合 RF 直线轨迹假设做轨迹估计和动态正则系数调度，相比 Diffusion-DPO 在 SD3-M/FLUX 上同时提点又把训练算力降到 1/12。

**[Riemannian Generative Decoder](image_generation/riemannian_generative_decoder.md)**

:   本文针对 Riemannian VAE 必须为每种流形手工设计复杂概率密度的痛点，提出 Riemannian Generative Decoder (RGD)——彻底丢掉 encoder，把每个样本的 latent 当作自由参数用黎曼优化器 (RiemannianAdam) 直接训，同时引入"按局部度量逆缩放的输入噪声"作为几何正则，在合成分支扩散树、人类线粒体 DNA、细胞周期 scRNA-seq 三个真实生物数据上恢复出更忠实的几何，且在高维下数值稳定胜过 VAE 基线。

**[SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](image_generation/spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)**

:   作者指出 MLLM 类编辑奖励模型存在"注意力坍缩"问题——评分时不去比较原图与编辑后图、而是塌缩到 sink token 上做盲判，进而提出 SpatialReward：先让 8B 模型预测编辑区域的边界框、再以这些 box token 为锚做交错式跨图推理；配上一个 260K 样本的空间感知数据集和 GRPO 两阶段训练后，在三个 reward benchmark 上 SOTA，并把 OmniGen2 的 GEdit-Bench 分数拉升 +0.90（是 GPT-4.1 提升的两倍）。

**[Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](image_generation/speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)**

:   本文发现 Speculative Jacobi Decoding (SJD) 在自回归视觉生成中加速有限的根因是连续迭代之间 draft token 的独立采样导致 collision 概率几乎为零；只需把独立采样换成 Maximal/Gumbel Coupling（一行修改、零额外训练），就能把图像生成最高加速到 $4.2\times$、视频生成 $13.6\times$，并严格保持输出分布与原 AR 解码一致。

**[Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](image_generation/structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)**

:   SDB 把模态翻译重写为"在所有满足边缘约束的耦合集合 $\mathcal{P}$ 中挑一个"，在 LDDBM 之上叠加边缘匹配（WTA + 容量约束）+ 端点级 + 轨迹级双层 cycle consistency，把成对监督仅作为可选启发式之一，从而在零成对、半成对、全成对三种监督预算下都能跑，并且全成对时也比 paired-only 基线更好（FFHQ→CelebA-HQ PSNR 从 25.6 提到 25.9）。

**[The Coupling Within: Flow Matching via Distilled Normalizing Flows](image_generation/the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)**

:   本文提出 NFM（Normalized Flow Matching），用预训练 TarFlow 这种自回归归一化流（NF）产生的"准确定性 data→noise 双射"作为 Flow Matching 的噪声-数据配对，从而把 FM 收敛速度、少步数 FID 同时拉到新的水平，并反过来比当老师的 NF 推理快若干个数量级。

**[Threshold-Guided Optimization for Visual Generative Models](image_generation/threshold-guided_optimization_for_visual_generative_models.md)**

:   作者把 DPO 的成对偏好假设拆掉，证明 KL 正则化最优策略本质上是把每个样本的 reward 与一个无法计算的实例相关基线 $\tau^*(x)=\beta\log Z(x)$ 比较，于是用从分数分位数估出的全局阈值 $\tau$ 替代它，再加一个与 $|s-\tau|$ 成正比的置信度权重，让扩散模型和 MaskGIT 在仅有标量打分（无成对偏好）时也能稳定对齐，并在五个 reward model 三个测试集上一致优于 Diffusion-DPO / KTO / DSPO。

**[Visual Implicit Autoregressive Modeling](image_generation/visual_implicit_autoregressive_modeling.md)**

:   本文把 Deep Equilibrium（DEQ）隐式不动点层嵌进 VAR 的 next-scale 自回归框架，用 Jacobian-Free Backpropagation 实现常数显存训练，把 VAR-d30 的 20 亿参数压到 7.7 亿，同时在推理时把每个 scale 的迭代次数变成"可调旋钮"——在 ImageNet-256 上 FID 2.16/sFID 8.07 不变的同时，4090 单卡峰值显存从 19.24GB 降到 8.53GB、吞吐从 15.16 提到 32.08 img/s。

**[Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](image_generation/watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)**

:   本文揭示扩散模型里一直被忽视的"时间步嵌入"其实是一条尚未被占用的信息侧信道——通过把训练时的 timestep 范围扩展到一个"影子区间"（shadow timestep）并把另一个数据分布绑定到该区间，可以在不改变 scheduler 接口的前提下，让同一个 diffusion 模型在显式区间生成正常图、在影子区间生成"隐藏"图，既可做隐蔽后门攻击也可做模型水印验证；同时给出基于正弦位置编码的互相干（mutual coherence）理论分析，解释为什么两个不相交区间能携带独立信息。

---

## 🔬 可解释性 { #interpretability }

**[All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](interpretability/all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)**

:   这篇论文用 Overlap-Aware Sheaf Repulsion (OASR) 算法系统性地证伪了机理可解释性领域的隐含假设——"一个 LLM 能力对应一个独特的电路"——发现同一任务可被多个几乎不重叠 (IoU ~4–11%) 但都满足 faithful/sparse/complete 的电路或 sheaf 支撑，并给出"分布式稠密电路假设"作为理论解释。

**[Barriers to Counterfactual Credit Attribution for Autoregressive Models](interpretability/barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)**

:   本文形式化研究生成式模型在 RAG/in-context 部署时的"反事实信用归因（CCA）"问题，证明两条令人惊讶的负面结果：(1) 即便底层 next-token 预测器是 (0,0)-CCA，自回归 rollout 也并非 CCA——CCA 不像 DP 那样在自回归下天然 compose；(2) 对一个已部署的非归因模型做 black-box "CCA retrofitting" 至少需要在输出长度 $\ell$ 上指数级查询次数。

**[Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](interpretability/circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)**

:   本文提出 Circuit Fingerprint 假说——单独把答案 token 喂进 Transformer，它在隐空间留下的方向恰好就是产生该答案所要走的电路路径——并据此用纯几何对齐（无需梯度/干预）完成 circuit discovery，同时同一组方向反过来可以做 activation steering，证明"读"和"写"是同一个几何对象的两面。

**[CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)**

:   通过把生成时 token 上的 SAE 激活与任务正确性做 Pearson 相关来挑选可解释的引导特征, 用正样本均值激活直接当系数, 不需对比数据集也不需反向传播, 就能在 Gemma-2 2B / LLaMA-3.1 8B 上把 MMLU 提 +3.3%、HarmBench 提 +27.1%, 且副作用率比微调更低。

**[Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis](interpretability/disentangling_direction_and_magnitude_in_transformer_representations_a_double_di.md)**

:   本文用 L2 匹配扰动协议，证明 Pythia 系列里方向（角度）扰动对语言建模 loss 的破坏力是同等位移幅值扰动的 42.9 倍，而幅值扰动对句法（主谓一致）的破坏远高于角度——这是一对认知神经科学意义上的 "双重分离"，对应方向走 attention 路径、幅值走 LayerNorm 路径。

**[Do Activation Verbalization Methods Convey Privileged Information?](interpretability/do_activation_verbalization_methods_convey_privileged_information.md)**

:   本文系统证明：当前流行的激活语言化方法（Patchscopes / LIT / SelfIE）在被用作 LLM 可解释性工具时，其性能完全可以由 "verbalizer 模型自己的知识" 解释，不需要任何 target 模型的内部激活——意味着这些工具在现有 benchmark 上看起来 work 是因为基准本身设计有缺陷，且当 verbalizer 知识超过 target 时会编造出 target 根本不具备的 "解释"。

**[SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](interpretability/gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)**

:   SemGrad 首次把"基于梯度"的不确定性量化搬到 LLM 自由生成场景——用语义保留分 (SPS) 找到能编码输入语义的隐藏态，把对它们求出的对数似然梯度范数当作 LLM 自信度的度量，无需采样、单次反向即可在 3 个 QA 数据集上击败 11 个 SOTA baseline，特别在多有效答案的 TruthfulQA 上比 SAR 高 3.27 AUROC。

**[Grokking: From Abstraction to Intelligence](interpretability/grokking_from_abstraction_to_intelligence.md)**

:   本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"，并用一个可解析的奇异特征机（SFM）证明这等价于自由能驱动的相变。

**[Interpretability Can Be Actionable](interpretability/interpretability_can_be_actionable.md)**

:   这是一篇立场论文，主张「可解释性研究缺的不是新方法、而是评估准则」：研究该以 actionability（insight 能否驱动可解释性领域之外的具体决策/干预）为核心评估维度，作者沿 concreteness + validation 两个维度定义 actionability、分析阻碍、列出 5 个有杠杆的应用域、给出研究者 6 步 checklist。

**[Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](interpretability/is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)**

:   作者对 6 个主流表格基础模型 (TFM) 做了首个大规模分层机理分析，发现中后层主要在做"迭代精化"且存在大量冗余，并据此设计了一个只用 20% 参数的单层循环 TFM，性能几乎追平六层原版。

**[Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](interpretability/manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)**

:   本文提出 MA-GIG：把 Guided IG 的“按低梯度幅值选特征再走一步”策略从像素空间搬到预训练 VAE 的潜在空间，借助 decoder Jacobian 把潜空间内的轴对齐更新映射成数据流形切空间上的更新，从而既避开高梯度噪声区域，又让积分路径上的样本始终贴近真实数据流形，归因更可靠。

**[Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](interpretability/memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)**

:   把自回归 LLM 的下一个 token 分布解释成一条 Markov 链的状态转移矩阵，于是「学新词」就变成「在状态空间里加新状态、并把它表示为已有状态的稀疏组合」，理论上只需 $O(s)$ 样本（$s$ 为映射到的旧 token 数），实践中只 finetune 新 token 的 embedding 即可在严格零遗忘下完成跨语种/新概念扩展。

**[Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](interpretability/optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)**

:   本文在高维线性回归 ICL 框架下，用一种保留 softmax 归一化与温度选择性、又解析可解的"近似 softmax 注意力"，**给出 ICL 泛化误差的闭式解和最优 attention temperature 的显式表达式** $\tau_{\text{opt}}$，证明只要调对推理时温度就能恢复近 Bayes 最优表现；在 GPT-2、Llama2-7B 的真实 QA 中也验证了这把"轻量旋钮"的有效性。

**[Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](interpretability/probabilistic_modeling_of_latent_agentic_substructures_in_deep_neural_networks.md)**

:   作者把神经网络（特别是 LLM）形式化为多个隐式子代理（每个是 outcome 上的概率分布）通过对数加权池化合成的复合代理，并在认知效用 $W_i(o)=\log P_i(o)$ 框架下证明了 "严格一致受益（strict unanimity）" 在线性池化或二元 outcome 下不可能、但 $|\mathcal O|\ge 3$ 下可行，进而推出"显式让 Waluigi 先显形再压制"严格优于"只强化 Luigi"的对齐原则。

**[Provably Learning Attention with Queries](interpretability/provably_learning_attention_with_queries.md)**

:   作者证明单头 softmax attention 在 value-query 访问下可以惊人简洁地被精确恢复 —— 只需 $O(d^2)$ 次查询，比同等结构的 ReLU MLP 容易得多；当头维 $r\ll d$ 时还能借压缩感知降到 $O(rd)$，并把结论扩展到带噪 oracle、membership query 以及多头不可识别性。

**[Steer Like the LLM: Activation Steering that Mimics Prompting](interpretability/steer_like_the_llm_activation_steering_that_mimics_prompting.md)**

:   本文把 "prompt steering"重新解释为 LLM 自己实现的一种 activation steering, 然后用一个**逐 token 的 ReLU 探针**来蒸馏 prompt 注入的激活差, 训练出 PSR (Prompt Steering Replacement) 模块, 既能在三个 steering 基准上超过现有激活引导方法 (CAA, ReFT-R1, Stolfo 等), 又在 AxBench 与人格引导上和 prompting 打成平甚至反超。

**[The Cylindrical Representation Hypothesis for Language Model Steering](interpretability/the_cylindrical_representation_hypothesis_for_language_model_steering.md)**

:   本文提出 Cylindrical Representation Hypothesis（CRH），在保留"概念线性"的前提下放弃 LRH 的正交性，证明概念向量的叠加会自然诱导出"轴 + 法平面 + 敏感扇区"的圆柱几何，从而首次几何化地解释了 activation steering 为什么在样本层面不可预测但在群体层面可观测。

**[The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](interpretability/the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)**

:   本文揭示 LLM 中"注意力汇聚到第一个 token"的结构性根源 —— 因果掩码下首 token 缺乏 value 聚合导致维度方差差异,被 FFN 中的 super neurons 选择性放大形成维度极度悬殊,最终锁死 QK 投影迫使形成 attention sink;并据此提出 head-wise RMSNorm 在预训练阶段从根上抑制 sink。

**[Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](interpretability/towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)**

:   作者用神经网络无穷宽缩放理论推出 steering vector 的 factor / direction 联合训练应满足 $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$ 这一缩放约束，从而消掉推理时人工选 $\alpha$ 的环节；同时受 ReFT 启发只在前 4 个 prompt token 上做加性干预（PrOSV），在 AxBench 上既能维持模型实用性，又能在三档 Gemma2/Qwen2.5 模型上一致超过全序列 FSSV。

**[Understanding LoRA as Knowledge Memory: An Empirical Analysis](interpretability/understanding_lora_as_knowledge_memory_an_empirical_analysis.md)**

:   作者用 PhoneBook 与新构造的 PaperQA 基准做系统实证审计，把 LoRA 看作可独立训练 / 加载 / 组合的知识记忆单元，定量给出"秩 → 容量 → 效率 → 多模块组合 → 与 RAG/ICL 互补"全链路的设计准则。

**[Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](interpretability/why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)**

:   本文给出"为什么 transformer 的内部表征可以被简单线性方法（probe、SAE、activation steering）反复成功解码"的架构级解释：只要语义特征是通过 OV 电路或 unembedding 这类**线性接口**被读出的，它就必须落在一个跨上下文不变的线性子空间里（Invariant Subspace Necessity 定理）；并推出一个零样本应用——Self-Reference Property，即 token 本身的嵌入方向就是其概念方向，从而可以无监督地用 class token 的几何位置直接做分类。

---

## 📦 模型压缩 { #model_compression }

**[ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](model_compression/arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)**

:   作者诊断出 VQ-VAE 的码本坍塌根源是"码本向量 ℓ2 范数失衡 + 几何聚集"，于是提出 SAMP：Ball-Bounded Norm Regularization 把所有码本向量约束在时变 Euclidean 球内、ArcCosine Additive Margin Loss 借鉴 ArcFace 在球面上推开 latent 向量，从而让码本均匀分散、利用率大幅上升，在 ImageNet 重建和生成 FID 上都击败主流 VQ-VAE 变体。

**[Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](model_compression/breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)**

:   针对 MoE LLM 的"负载不均–参数冗余–通信开销"三难，本文提出一个统一框架：用"参数 + 激活"双相似度在线聚类把专家分组，组内用"共享基矩阵 + 低秩残差"做结构化压缩 (~5×)，再做"先选组后选 expert"的两级分层路由 + FP16/INT4 异构精度 + 闲置组离线卸载，在 GLUE/WikiText-103 上以约 80% 参数缩减、10–20% 吞吐提升、专家负载方差降 3× 的代价匹配标准 MoE 性能。

**[Demystifying When Pruning Works via Representation Hierarchies](model_compression/demystifying_when_pruning_works_via_representation_hierarchies.md)**

:   论文从"嵌入 → logit → 概率"三段表征层次出发，用 Taylor 局部展开理论证明：剪枝对嵌入空间和 logit 空间的扰动天生很小，但 softmax 这一非线性步骤会按 $\mathrm{Var}_r(\Delta z)/(2T^2)$ 把扰动放大到概率空间，再经过自回归解码的步间累积，最终导致生成任务崩溃；而非生成任务因为只依赖候选 token 子空间，对剪枝天然鲁棒——这统一解释了为什么剪枝在 MMLU、retrieval 上几乎无损但在 GSM8K、HumanEval 上骤降到 0。

**[Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](model_compression/dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)**

:   本文系统观测到 "小语言模型的 token 嵌入会随深度坍缩到一个窄锥体"（embedding condensation）这个普遍现象——大模型反而不会——并设计了一个角度分散损失 $\mathcal{L}_{\text{disp}}$ 直接逼嵌入散开，无须加参数就让 Qwen3 / GPT2 在 10 个 benchmark 上平均提升 3.3%。

**[Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](model_compression/dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)**

:   本文提出 TAD（Tail-Aware Distillation）：在标准 KD 的 KL 散度中显式把教师 top-$K$ 概率与"尾部"概率拆开并放大尾部贡献，从而在学术级算力（单卡 H100 + 1 周）内完成 LLM 预训练蒸馏，平均效果优于 MiniPLM 等数据中心方法。

**[FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](model_compression/fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)**

:   本文指出联邦 LoRA 中朴素 factor-wise 平均的真正"敌人"是旋转不变性导致的潜在子空间错位，提出在客户端用正交 Procrustes 求解出旋转矩阵 $R_i^t$ 对齐 $A,B$ 因子后再聚合，理论与实验都证明能显著降低聚合误差且不增加通信开销。

**[FlattenGPT: Depth Compression for Transformer with Layer Flattening](model_compression/flattengpt_depth_compression_for_transformer_with_layer_flattening.md)**

:   本文提出 FlattenGPT，先把 LLM 中输入相似度高的相邻 transformer 层"扁平化"合并为一个 2× 宽度的层 (保留所有参数知识)，再对合并层做通道剪枝把宽度恢复到原始规模——既享受深度压缩的推理加速，又避免传统层剪枝直接丢知识的性能塌方。

**[From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers](model_compression/from_per-image_low-rank_to_encoding_mismatch_rethinking_feature_distillation_in_.md)**

:   作者用 sample-wise SVD + dataset-level PCA + token-level Spectral Energy Pattern (SEP) 三视角揭示了一个看似矛盾的 ViT 表征几何："每张图的特征矩阵都是低秩的，但跨图共享的子空间却几乎要满秩 + 单 token 的频谱带宽接近 100%"，进而提出 Lift（推理时保留 lifting projector）和 WideLast（只把最后一个 block 加宽到 teacher 宽度）两个极简补丁，让普通 MSE 特征蒸馏在 DeiT-Tiny ← CaiT-S24 上从 74.86% 一路涨到 78.23%。

**[Linearizing Vision Transformer with Test-Time Training](model_compression/linearizing_vision_transformer_with_test-time_training.md)**

:   作者发现两层 TTT 内模型在结构上等价于 Softmax 注意力（Softmax 可看作两层动态 MLP），由此实现 Q/K/V/MLP 的全权重直接继承，再通过 key Instance Normalization 处理 shift-invariance、depthwise conv on Q/K 补齐 locality，仅 1 小时微调就把 Stable Diffusion 3.5 线性化并加速 1.32×–1.47×。

**[OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](model_compression/osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)**

:   OSAQ 利用 LLM 各层 Hessian 在不同输入下保持一致的低秩零空间，将零空间向量线性组合成一个加性权重扰动 $\Delta W$，在不改变二阶任务损失的前提下把离群权重「自吸收」掉，使 2 比特仅权重量化的困惑度比朴素 GPTQ 降低 40% 以上。

**[Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](model_compression/preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)**

:   作者提出 SRR（Structured Residual Reconstruction），把 QER（Quantization Error Reconstruction）中固定用于补偿量化残差的低秩预算 $r$ 显式地拆成"先保留 $k$ 个主奇异方向再量化"和"用 $r-k$ 个秩去拟合残差"两部分，并给出一个只需一次随机探针的闭式准则来逐层选 $k^\star$，在 2/3 bit PTQ 和 QPEFT 上一致优于 LQER/QERA。

**[Proxy Compression for Language Modeling](model_compression/proxy_compression_for_language_modeling.md)**

:   作者提出「proxy compression」——训练时把 90% 数据喂成 tokenizer / 神经压缩器产出的短序列、10% 喂原始 UTF-8 字节，配合 sentinel token 与短暂的 in-context translation warm-up；推理时丢掉所有压缩器，模型只看原始字节，却能在固定 compute 下显著超过纯字节模型，且在大规模下追平甚至超过 tokenizer baseline。

**[Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](model_compression/resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)**

:   本文把激活稀疏导致 LLM 掉点的本质归因为"表示漂移"，并仿照生物自发放电向每层注入一个输入无关、训练后可吸收进 bias 的小向量（SPON），以接近零推理开销显著缩小稀疏模型与稠密模型的差距。

**[RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](model_compression/rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)**

:   RQ-MoE 用「两级 MoE + 双流量化」的设计，让残差向量量化（RQ）的码本随输入动态生成，又通过把指令流与重建流解耦实现 6–14× 解码加速，在四个 retrieval benchmark 上 MSE/Recall 持平或超越 QINCo。

**[ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](model_compression/scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)**

:   作者证明 LoRA 累加更新被困在固定低秩子空间，提出 ScaLoRA：每步把旧 $AB^\top$ 合并到 $W^{pt}$ 后，**用一个可解析求得的最优"列缩放"** 重启 adapter，使 AdamW 一阶/二阶动量可以 $O((m+n)r)$ 等变传递 (不需要重置/warm-up)、累加更新自然变高秩——在 DeBERTaV3、LLaMA2-7B、LLaMA3-8B、Gemma3-12B 上一致打过 LoRA / MoRA / HiRA / ReLoRA / LoRA-GA。

**[Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](model_compression/semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)**

:   本文先用新基准 KVFundaBench 系统揭示「检索类长上下文压得动、推理类压不动」的关键不对称，并把原因归结到 KV 压缩破坏了少样本示例这一「语义单元」的完整性；据此提出 ShotKV——在 prefill 阶段保留整个 shot 作为不可分割单元、在 decoding 阶段做动态 token 级压缩，让 LG-GSM8K 在 40% 压缩率下从 baseline 46.0 提升到 47.33，并在长输入设置下端到端延迟降低 11.3%。

**[Stochastic Sparse Attention for Memory-Bound Inference](model_compression/stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA 把 attention 的 value 聚合 $AV$ 看作 "按 softmax 概率 $A$ 对值行 $V$ 做加权求和", 改成 "从 $A$ 中无放回采样 $S\ll n_k$ 个索引、直接平均对应 $V$ 行"的无偏估计, 用 stratified / systematic 采样降方差, 再写成 GPU kernel 与 FlashDecoding 对齐——在 32k context 下端到端比 FlashInfer / FlashDecoding 快 1.5× 且精度不掉。

**[SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](model_compression/surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)**

:   SURGE 给每个二值化层并联一个"全精度辅助分支"，前向输出不变但反向能从全精度分支额外回传一份"非 STE 截断"的高阶梯度，并用 AGS 按梯度范数比动态平衡两路贡献，让 BNN 在 ResNet-18/ImageNet 上做到 62.0% top-1，比 ReCU 高 1.0%、比 IR-Net 高 3.9%。

**[Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](model_compression/task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)**

:   LoDA 把 LoRA 的下投影矩阵按「投影能量」拆成一个跨任务共享的通用子空间和一个真正只激活新任务的隔离子空间，再用梯度对齐训练上投影、并在融合时给通用分支闭式重标定，从而在多个持续学习 benchmark 上稳定刷过现有 LoRA-CL 方法。

**[Test-Time Training with KV Binding Is Secretly Linear Attention](model_compression/test-time_training_with_kv_binding_is_secretly_linear_attention.md)**

:   本文用四个「记忆悖论」反例 + 一套严格的展开定理，证明带 KV-binding 内循环的 TTT（如 LaCT、ViTTT）即便用多层 MLP + 动量也只是「学到的线性注意力算子」，并据此把它简化、并行化为标准线性注意力，吞吐提升 4× 而性能几乎不掉。

**[Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)**

:   作者发现 token 的"重要性"在层间和头间剧烈变化，传统 token eviction 一次性删除是不可逆的早期决策错误；他们提出 Token Sparse Attention，每层每个 attention head 独立选 $L' \ll L$ 个 token 做密集 attention，输出再 scatter 回原始序列长度，配上残差路径让被略过的 token 在下一层重新有机会被选中——既保留头/层级动态选择，又能直接调用 FlashAttention 等密集 kernel，在 128K 上下文上叠加 FlexPrefill 后达到 ×3.23 注意力加速、精度损失 <1%。

---

## 💡 LLM 推理 { #llm_reasoning }

**[A Formal Comparison Between Chain of Thought and Latent Thought](llm_reasoning/a_formal_comparison_between_chain_of_thought_and_latent_thought.md)**

:   本文从计算复杂度理论出发，形式化比较 CoT（链式思维）与隐式思维（Looped Transformer / Coconut）的表达能力，证明隐式思维在多对数深度下严格达到 $\mathsf{TC}^k$，而 CoT 最多到 $\mathsf{TC}^{k-1}$；同时在概率设置下首次揭示 CoT 通过随机解码可支持 FPRAS 计数，反过来超越确定论隐式思维。

**[ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](llm_reasoning/anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)**

:   ANCHOR 用"自底向上溯因 + 层级聚类" 构造稠密因子空间，对下游条件做粗到细检索得到稀疏相关因子集，再联合 Naïve Bayes 与一个 LLM 现场构造的潜变量因果贝叶斯网络做后验聚合，在 LLM 高风险决策场景中显著减少 "unknown" 预测并提升概率校准。

**[Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs](llm_reasoning/automated_formal_proofs_of_combinatorial_identities_via_wilf-zeilberger_guidance.md)**

:   WZ-LLM 把经典的 Wilf–Zeilberger 符号证明流程编译成 Lean 4 中可执行的证明骨架（递推 + 边界条件 + 侧条件），交给专门用 SFT + expert-iteration + DAPO 训练出的 WZ-Prover 逐项 discharge，在 100 个经典组合恒等式上把 pass@32 从 Goedel-Prover-V2 的 9% 提升到 34%。

**[Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](llm_reasoning/break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)**

:   针对扩散语言模型 (dLLM) 半自回归生成时"块大小固定"破坏推理逻辑链的问题，本文提出 b1：用 RL 学一个块结束指示 token 来生成动态长度块，并用一个"块级熵单调下降 (Monotonic Entropy Descent, MED) 奖励"驱动连贯推理，作为即插即用的奖励项接入现有 dLLM RL 框架（Diffu-GRPO/GDPO/d1/wd1），在 Countdown 上将 wd1 从 39.45 推到 58.98。

**[Conformal Thinking: Risk Control for Reasoning on a Compute Budget](llm_reasoning/conformal_thinking_risk_control_for_reasoning_on_a_compute_budget.md)**

:   本文把"reasoning LLM 何时停止思考"从一个不可解释的阈值调参问题，重构为一个**用户可指定 risk 容忍度**的 conformal 风险控制问题：用两个阈值——上阈值在模型自信时停（控 false positive），新提出的**参数化下阈值**在模型在不可解题上"想不动"时强行停（控 false negative）——并通过 UCB 算法从校准集自动求出满足风险约束的阈值，在 AIME / GPQA / MathVision 上实现"准确率几乎不掉、token 大幅省"。

**[Efficient Reasoning with Hidden Thinking](llm_reasoning/efficient_reasoning_with_hidden_thinking.md)**

:   Heima 把多模态 LLM 的冗长 CoT 每个阶段（summary / caption / reasoning）蒸馏成**一个特殊 thinking token**，让模型在隐空间里"想"，token 数从 100-200 量级降到 13-16 个的同时 zero-shot 准确率反而比 LLaVA-CoT 更稳；配套训练一个 LLM "interpreter"用 thinking token 的 hidden state 重建出文字推理链，从而验证压缩损失的信息论上界。

**[Entropy-informed Decoding: Adaptive Information-Driven Branching](llm_reasoning/entropy-informed_decoding_adaptive_information-driven_branching.md)**

:   EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search；理论上证明熵单调的分支因子在期望累计 regret 上严格优于任何固定束宽，且能给出 $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$ 的显式 regret 率。

**[ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](llm_reasoning/ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)**

:   ETS 直接从 KL 正则化 RLHF 目标的**闭式最优解**采样，把它写成「参考策略 × 指数 reward 的条件期望（能量项）」，再用 Monte Carlo + 自归一化重要性采样在测试时近似这个能量项，从而**不训练**就达到甚至超过经过 RL 后训练的策略，并通过 lightweight proposal + Fast-dLLM 把延迟控制在可用范围。

**[Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](llm_reasoning/express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)**

:   这是一篇 position paper，主张：**用 LLM 的 token softmax 概率（logprob）当成"世界事件概率"是理论上错的**——因为 distribution estimation、response prediction 和 target distribution estimation 是三个不同任务，对应不同 ideal 输出分布；获取世界概率的正确做法是**二阶预测**——让 LLM 在输出里**显式写出**它对事件的概率（数值或语言修饰词），而不是去算"它说 X 的概率"。

**[Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](llm_reasoning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)**

:   本文把 LLM 主动提问场景（20 Questions / 医疗诊断 / 故障排查）建模成两人零和扩展式博弈 (EFG)，提出 Game of Thought (GoT)：用深度有限的子博弈构造 + CFR 求 Nash 均衡来产生“随机化提问策略”，在所有数据集上把 worst-case 交互轮数显著降低，且 weighted 变体下相对 UoT 提升 15–40%。

**[GRPO is Secretly a Process Reward Model](llm_reasoning/grpo_is_secretly_a_process_reward_model.md)**

:   本文从理论上证明 GRPO + ORM 在"组内轨迹共享前缀"的温和条件下**等价于**一个带有 Monte-Carlo PRM 的过程奖励 RL 目标，从而揭示出 vanilla GRPO 隐藏的一个 bug——前缀长度不均会让高奖励轨迹的大部分 token 拿到负 advantage——并提出 $\lambda$-GRPO 做一个 PRM-aware 归一化，在推理 benchmark 上稳定超过 GRPO 且训练快约 2 倍。

**[Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](llm_reasoning/hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)**

:   用一个简单的逻辑回归探针在 LLM 思维链生成时的隐藏状态上能以 0.95 AUROC 预测整条推理是否会出错（从第 1 步就有 0.79），但文本表面同样训出来的分类器只有 0.59；可惜 4 种干预手段（激活引导、探针引导 best-of-N、自我修正、激活补丁）全部失败——这个错误信号是"诊断性"的而非"因果性"的。

**[Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](llm_reasoning/lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)**

:   NSI 把 LLM agent 的交互轨迹 "提升" 为带显式条件分支和动态变量绑定的神经符号工作流图，使技能从无状态脚本进化成可状态感知的逻辑程序，在 ALFWorld / WebShop / TextCraft 上分别拿到 98.0 / 76.5 / 95.2 的成功率，全面碾压 ASI 和 AWM 等编程式技能基线。

**[Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](llm_reasoning/many-shot_cot-icl_making_in-context_learning_truly_learn.md)**

:   本文系统揭示了非推理任务的 many-shot ICL “经验法则”在 CoT 推理任务上**全部失效**——相似度检索反而有害、顺序敏感性随 shot 数增长——并把成功的 many-shot CoT 重新解读为“in-context 测试时学习”，由此提出按 embedding 轨迹曲率排序 demonstration 的 CDS 方法，在 64-shot 几何题上提升 5.42 pp。

**[Multimodal Fact-Level Attribution for Verifiable Reasoning](llm_reasoning/multimodal_fact-level_attribution_for_verifiable_reasoning.md)**

:   MURGAT 是首个评测 MLLM 在多模态推理输出中"按事实粒度精确引用模态+时间段"能力的基准，搭配一个三步评估协议（可验证句识别 → 原子事实分解 → 归因质量）和高度与人工对齐的自动评测器 MURGAT-SCORE（Pearson 0.84），揭示了强模型即使答案对也常常胡乱引用，且强推理常以牺牲可验证引用为代价。

**[Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](llm_reasoning/prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)**

:   作者把"为离散扩散语言模型（dLLM）做高效 test-time scaling"这一问题拆成三件事——按"探索→渐进剪枝→精修"的层级时间表分配计算（HTS）、用部分 remask 做局部分支保住高置信"逻辑骨架"、把 dLLM 自己当 Yes/No 验证器（SVF），最终在 4 个数学/代码基准、3 个 dLLM 上以远少于 best-of-$N$ 的 NFE 达到相近甚至更好的精度。

**[Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](llm_reasoning/provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)**

:   本文为「先易后难」的课程式 RL 后训练给出第一份严格的样本复杂度证明：在 transformer 的状态条件自回归推理树上，若课程能让相邻阶段的难度比保持在目标难度的 $L/p$ 次根级别，则总样本数可从直接训练的指数级 $(C^\star)^L$ 降到课程版的多项式级 $L\cdot (C^\star)^{p_\max}$。

**[ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)**

:   ResRL 从理论上把 RLVR 中 "负样本梯度污染正样本"现象 (Lazy Likelihood Displacement) 分解成"logit × 表征"两个分量,然后在表征层用正样本的 SVD 低秩子空间做投影残差,根据每个负 token 的"正交分量能量"给它一个 [ξ,1] 区间的梯度权重——表征越像正样本(残差越小)就罚得越轻,纯错误成分才被重罚,既保住 Pass@1 又不丢 Pass@k 多样性;在 Qwen3-4B 数学任务上 Avg@16 比 NSR 提升 9.4%,Pass@128 提升 7.0%。

**[ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)**

:   作者把 MATH 数据集的人工标注解题步骤逐步翻译成"带描述与类型签名的可复用 Python 工具"，构造出含 8K 题 + 12K 工具的 ToolMATH 基准；它同时覆盖长程多工具组合（hop 1-8+）、可控的干扰工具相似度（5 级 × 4 种密度）、以及"金标工具被全部移除"的工具缺失场景，验证显示模型失败的主导因素不是工具选择而是推理本身——thought error 占 90%+，而干扰工具会把早期的小偏差放大成不可逆的执行漂移。

**[Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](llm_reasoning/unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)**

:   作者把"地面街景与卫星图能否定位为同一坐标"作为可验证间接奖励，用 GRPO 对 Qwen2.5-VL-7B 做两阶段后训练（CoT scaffolding + RL self-exploring），让模型仅凭 GPS metadata 学到可零样本迁移到 25+ 地理空间任务的通用推理能力。

---

## 🎮 强化学习 { #reinforcement_learning }

**[CAMEL: Confidence-Gated Reflection for Reward Modeling](reinforcement_learning/camel_confidence-gated_reflection_for_reward_modeling.md)**

:   本文观察到 verdict token 的 log-probability margin 与判断正确率高度相关，据此提出 CAMEL —— 先用单 token 快速给出偏好判断，仅在低置信度时才触发反思生成，并用反事实前缀增强 GRPO 训练自我纠错能力，在三个奖励模型 benchmark 上以 14B 参数取得 82.9% 的平均准确率（超过此前最佳 70B 模型 3.2%）。

**[CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](reinforcement_learning/cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)**

:   把 self-play 从"对抗"换成"协作": Coach 出题、Player 解题、Coach 拿"Player 进步幅度 × Player 解题率"作为奖励, 在完全不用外部训练数据的条件下让 Qwen2.5-Math-7B-Instruct 在六个数学 benchmark 上总均分 +4.9、OOD +5.4, 超过 RENT/R-Zero 等已有 unsupervised 方法。

**[DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](reinforcement_learning/debiased_model-based_representations_for_sample-efficient_continuous_control.md)**

:   DR.Q 在 MR.Q 类"模型化表示 + actor-critic"骨架上加两件事——用 InfoNCE 显式最大化 $z_{sa}$ 与下一状态表示 $z_{s'}$ 的互信息，再用"PER × forget"融合的 faded prioritized replay 缓解早期经验过拟合——在 73 个连续控制任务上用单一超参组击败 SimBaV2 / MR.Q / TDMPC2 等强基线。

**[EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](reinforcement_learning/earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)**

:   EARL 用"粗解析-细响应"两阶段 MLLM 框架把第一视角交互理解任务（描述+答问+像素掩膜）做成统一管线：第一阶段输出整图交互的全局描述并把最后一层 hidden state 当作语义先验，再通过新的 Analysis-guided Feature Synthesizer 注入到第二阶段，用 GRPO + 三路奖励（格式/答案/grounding 准确率）联合训练，在 Ego-IRGBench 上 cIoU 反超 Seg-Zero 8.37%。

**[From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](reinforcement_learning/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)**

:   针对"多轮交互式工具调用 Agent"后训练里两大瓶颈——高质量数据贵 + 用户模拟噪声毁 RL 信号，作者提出"自演化多 agent 数据合成 (AReaL-SEA)"配套生成可执行 verifier 当奖励，再配上"先 SFT 用户模型再做大 batch + 动态过滤 GRPO"的 RL recipe，在 τ²-bench 上把 Qwen3-235B 推到 Airline 73.0 / Telecom 98.3 的 pass^1，全面达到或超过 Claude/Gemini/GPT-5。

**[How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](reinforcement_learning/how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)**

:   作者把"训 LLM 学下国际象棋"当成可验证 RL 的干净实验台，系统比对 6 类自制 SFT 数据集对 RL 的影响，发现"直接预测最佳一步 (Best Move)"得最高分但 RL 后产生不忠实推理，"预测多步最佳走法 (Best Line)"性能相当但 RL 更稳、推理更忠实；并提炼出三条可用 SFT-checkpoint 预测 RL 终局性能的指标，最终用 7B 模型在多个国象 benchmark 上超过 gpt-oss-120b。

**[Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](reinforcement_learning/long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)**

:   本文挑战“离线 RL 必须显式保守”的主流共识，提出 Neubay：用贝叶斯视角看后验上的模型集合、用**长 horizon rollout（数百步）**自然吸收价值高估、用 layer norm 与不确定度阈值控制 compounding error，从而在 D4RL/NeoRL 共 33 个数据集上不靠悲观惩罚就追平 SOTA 保守算法，并在 7 个数据集上刷新纪录。

**[Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](reinforcement_learning/multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)**

:   SeqComm-DFL 把"多智能体通信"作为预测器、把"联合策略选择"作为下游优化器，用价值感知的消息生成 + Stackelberg 序贯条件 + 隐式微分双层优化把通信学习直接对齐到团队回报，在医院调度和 SMAC 上取得 4-6 倍的累积奖励提升与 >13 个百分点的胜率提高。

**[Path-Coupled Bellman Flows for Distributional Reinforcement Learning](reinforcement_learning/path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)**

:   把分布式 Bellman 方程的"仿射搬运"几何性显式编织进 flow matching 的路径里：用同一份基础噪声同时驱动当前态与后继态的两条路径，再用 $\lambda$ 控制变量在偏差与方差之间换挡，从而得到一个对源分布相容、对 Bellman 端点相容、又稳定的分布式 critic。

**[Probing RLVR Training Instability through the Lens of Objective-Level Hacking](reinforcement_learning/probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)**

:   作者提出"objective-level hacking"框架,把 MoE 大模型在 RLVR 中训练-推理差异越训越大的现象归因为 token 级权重失真在优化目标里引入的有偏伪信号,并在 30B MoE 上通过四组实验验证"偏差(不是方差)才是元凶"。

**[QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](reinforcement_learning/qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)**

:   QHyer 用 Normalizing Flows 估计的状态依赖 Q 值取代 Decision Transformer 中的轨迹依赖 RTG，再叠加门控式 Attention-Mamba 混合骨干以实现内容自适应的历史压缩，在 OGBench/D4RL 的非马尔可夫与马尔可夫离线目标条件 RL 数据集上同时刷新 SOTA。

**[R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning](reinforcement_learning/r2r2_robust_representation_for_intensive_experience_reuse_via_redundancy_reducti.md)**

:   R2R2 把 VICReg 风格的冗余去除约束加进自预测学习（SPL）以稳定高 UTD 训练，但**关键改动是不做零中心化**——理论上证明 zero-centering 会消除 SPL 谱分解中的常数本征模（即全局动力学信息），实验在 TD7 上 UTD=20 时把分数从 1.02 提到 1.24（+22%），并以新提出的 SimbaV2-SPL 架构刷新连续控制 SOTA。

**[Recovering Hidden Reward in Diffusion-Based Policies](reinforcement_learning/recovering_hidden_reward_in_diffusion-based_policies.md)**

:   EnergyFlow 把 diffusion policy 的 score field 显式参数化为一个标量 energy function 的负梯度，论证了 maximum-entropy 最优下 score = 软 Q-函数梯度，从而在不做对抗优化的情况下"白送"一个可用作下游 RL shaping reward 的标量信号，同时保守场约束改善 OOD 泛化。

**[SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](reinforcement_learning/sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)**

:   本文把 MoE 策略在持续强化学习中的可塑性丢失形式化为 empirical NTK 矩阵谱熵有效秩的下降，再用 Gauss-Newton 与 Kronecker 分解把它降维到一个只依赖"专家特征 Gram 矩阵"的可计算 proxy，最后用一个一行的 Parseval 罚（SPHERE）拉高这个 proxy，在 MetaWorld 和 HumanoidBench 持续 RL 设置下把任务成功率分别提升 133% 和 50%。

**[Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning](reinforcement_learning/stochastic_minimum-cost_reach-avoid_reinforcement_learning.md)**

:   本文提出 Reach-Avoid Probability Certificate (RAPC), 用一个 max-min-夹紧的 Bellman 收缩算子让值函数下界 reach-avoid 概率, 配合一个对抗 $\gamma^T$ 衰减的 "补偿因子"作归一化, 再用对称梯度投影联合优化 "成本"与 "reach-avoid 概率"两个冲突目标, 在 MuJoCo 上同时拿到比 RC-PPO / RESPO / CPPO 更低的累积成本和更高的 reach 成功率。

**[T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](reinforcement_learning/t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)**

:   T$^2$PO 把多轮 agentic RL 的训练崩溃归因为"hesitation（犹豫）"——token 层过思考、turn 层重复无效——并用一个融合 entropy+confidence 的自校准不确定性信号 $M_t$ 同时驱动 token-level Thinking Intervention（动态截断 think 段）和 turn-level Dynamical Sampling（重采样无效 turn），在 WebShop / ALFWorld / Search QA 上稳定超越 PPO/GRPO/GiGPO。

**[Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](reinforcement_learning/towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)**

:   本文提出 FAN：把"昂贵的生成式策略 + 分布式 critic"压缩到"单步 flow 锚定 + 单噪声样本 critic"——用 Flow Anchoring 在一次 flow 评估内完成行为正则化，用 noise-conditioned critic 把 quantile 多样本替换成单 Gaussian 噪声样本，在 D4RL/OGBench 上做到 SOTA 性能同时训练比同类分布式方法快 5-14×。

**[Trajectory-Level Data Augmentation for Offline Reinforcement Learning](reinforcement_learning/trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)**

:   本文提出 LIFT：在主动定位任务里，利用轨迹几何性质把次优 logging policy 留下的冗余 zig-zag 轨迹"抄近道"成 shortcut，并把这些合成 transition 喂给一个轻量增广器在数据采集期间替换 logging 动作，使离线 CQL 在低维到高维、partial obs 等各种设置下显著超越普通离线 RL 与 warm-start SAC。

**[Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](reinforcement_learning/turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)**

:   本文把多个 MLLM 之间的推理"漂移"重新解释成 DPO 中的负样本约束，用 Plackett-Luce 偏好损失同时压制 N 个 source model 的发散轨迹，让 7B 学生模型在不需要 ground-truth 报告的前提下，仅用 10% 的 MIMIC-CXR 就在胸片分类与报告生成任务上超过所有 source teacher。

**[Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](reinforcement_learning/vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)**

:   本文研究"在 N 个智能体的大规模 MARL 系统中挑出 K 个最脆弱的智能体"这一双层 NP-hard 问题，把它建模为 HAD-MFC（Hierarchical Adversarial Decentralized Mean Field Control），用 Fenchel-Rockafellar 变换把下层最坏对抗策略的训练折叠成一个加正则项的"鲁棒 mean-field Bellman 算子"，再把上层组合选择问题转化为带稠密 reward 的 MDP 用贪心或 RL 求解，证明分解保持最优性，在 18 个任务中 17 个超 baseline。

---

## 🧮 科学计算 { #scientific_computing }

**[A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](scientific_computing/a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)**

:   本文从最小作用原理出发，提出 Wasserstein 拉格朗日力学（WLM）框架，学习二阶人口动力学而非传统梯度流的一阶动力学，从而能够捕捉周期性、旋转等更丰富的群体现象，并可在不需要参考过程的情况下完成插值与未来预报。

**[ANTIC: Adaptive Neural Temporal In-situ Compressor](scientific_computing/antic_adaptive_neural_temporal_in-situ_compressor.md)**

:   为了把 PB-EB 级别 PDE 仿真数据"边算边压"，本文提出 ANTIC：用 physics-aware 时间选择器只保留物理上重要的快照，再用神经场 + LoRA 持续微调编码相邻快照之间的残差，在 2D Kolmogorov 流上拿到 435× 压缩、在 4.2 TiB 的 3D 双黑洞合并模拟上拿到 6807× 时空联合压缩。

**[Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](scientific_computing/discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)**

:   DoLQ 在 LLM 符号回归的搜索循环里插入一个 "Scientist Agent"，对候选项同时做定性（物理合理性）+ 定量（消融式 MSE 贡献）评估，把 LLM-SR 那种 "低误差但项数臃肿、物理上荒谬" 的方程逼到既数值精确又结构紧凑。

**[Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](scientific_computing/flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)**

:   本文提出 Flow Sampling，把流匹配/扩散模型从"数据驱动"反转为"噪声驱动"——以源噪声样本为条件构造去噪扩散漂移，在 interpolant 上用 detached 模型采得 $X_1$ 的能量梯度做回归目标，从而学到无数据情况下的高效扩散采样器，并自然推广到常曲率黎曼流形。

**[Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics](scientific_computing/mesh_field_theory_port-hamiltonian_formulation_of_mesh-based_physics.md)**

:   从「局部性 + 置换等变 + 朝向协变 + 能量守恒/耗散不等式」四条物理原理出发，证明任何满足这些公理的网格物理动力学在雅可比层面都可以局部约化为 port-Hamiltonian 形式——其中守恒互联结构 $J$ 完全由网格拓扑（符号关联矩阵 $D_k$）固定，度量与耗散通过可学的 $G, R$ 进入；据此设计的 MeshFT-Net 在长时间 rollout 上能量漂移近零、色散与动量正确，并大幅领先 MGN / HNN。

**[Meta-learning Structure-Preserving Dynamics](scientific_computing/meta-learning_structure-preserving_dynamics.md)**

:   把 modulation-based 元学习（hyper-network 把 latent code $\bm{z}^{(k)}$ 映射成层级调制参数）系统性地引入 Hamiltonian / GENERIC 神经网络，提出两种新颖调制——latent multi-rank (MR) 与 latent SVD-like 调制，让一个共享网络在不知道系统参数 $\bm{\mu}$ 的情况下少样本适配整族新参数实例，同时严格保持能量守恒 / 耗散结构。

**[MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier](scientific_computing/moose-star_unlocking_tractable_training_for_scientific_discovery_by_breaking_the.md)**

:   MOOSE-Star 把"训练一个能直接生成科学假设的 LLM"这个原本要在 $\mathcal{O}(N^k)$ 组合空间里搜索的问题拆成"灵感检索 + 假设合成"两个序列子任务，再叠上层级树检索 + bounded composition + motivation 规划，把最优复杂度从指数级压到 $\mathcal{O}(\log N)$，并放出 108,717 篇带分解标注的 TOMATO-Star 数据集。

**[Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](scientific_computing/phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)**

:   为单次曝光式压缩光谱成像 (CASSI) 设计一个 train-render 两阶段、按波长可任意查询的深度展开框架——在每个展开 stage 内塞入连续光谱场 (CoSF) 先验模块，由 Fourier-Mamba 驱动的三分支跨域特征混合器 + 随机频率编码 + 谱合成头组成，离散波长训练即可在推理时合成任意连续波长的高光谱图像，实现连续光谱重建与零样本光谱超分。

**[PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](scientific_computing/podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)**

:   PODiff 把扩散模型从像素空间搬到固定的、按方差排序的 POD 系数空间里跑，用极小的 MLP 就能在 $640\times 480$ SST 降尺度任务上拿到与像素级扩散相当的精度，同时因为重构是线性的，集成方差可以通过 $\Sigma_u=\Phi\Sigma_a\Phi^\top$ 解析回传到物理空间，得到空间上可解释、且校准良好的不确定性。

**[Rethink the Role of Neural Decoders in Quantum Error Correction](scientific_computing/rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)**

:   本文在 $d\le9$ 的表面码上系统重做 MLP/3D-CNN/TCN/Transformer/GNN 五类神经解码器，并把"量化 + 剪枝 + FPGA 资源建模"作为一等公民放进训练流程，结论是：近期解码性能由数据量而非架构复杂度主导，且 INT4 + QAT 是实现微秒级实时解码的必要前提。

**[Saving Foundation Flow-Matching Priors for Inverse Problems](scientific_computing/saving_foundation_flow-matching_priors_for_inverse_problems.md)**

:   针对 Stable Diffusion / Flux 这类基础流匹配模型在求解逆问题上明显逊于领域专用先验甚至未训练先验的现象，作者提出 FMPlug：用一个由近似样本指导、时间可学习的 warm-start 加上锐利高斯壳层约束，把基础 FM 的潜变量塞回它真正"懂"的薄壳上，从而显著恢复其作为逆问题先验的能力。

**[Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](scientific_computing/semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)**

:   SuperMeshNet 用两个互补 MPNN——主模型预测 LR→HR，辅助模型预测 LR-LR 对应的 HR-HR 差分——在无配对 HR 的样本上互相生成伪标签，并配合节点级 / 消息级 centering 两个轻量归纳偏置，使得 PDE mesh 超分仅用 10% HR 数据就能超过 100% HR 全监督基线，跨 6 种 MPNN 架构一致下降 RMSE。

**[Skipping the Zeros in Diffusion Models for Sparse Data Generation](scientific_computing/skipping_the_zeros_in_diffusion_models_for_sparse_data_generation.md)**

:   SED 把扩散模型从"对所有维度做全密集去噪"改成"只在非零维度上跑扩散+自回归解码维度-值对"，让计算量从随维度线性增长变成几乎随非零数恒定，同时严格保留科学数据中"显式零"这一语义信息。

**[Smoothness Errors in Dynamics Models and How to Avoid Them](scientific_computing/smoothness_errors_in_dynamics_models_and_how_to_avoid_them.md)**

:   作者从理论上指出 Kiani 等人的 "unitary GNN" 因为强行保持 Rayleigh 商而对热扩散这类"天然会变光滑"的物理系统过度约束，进而提出"松弛 unitary 卷积"（R-UniGraph / R-UniMesh）并把整套 Rayleigh 商-unitary 卷积框架从图扩展到三角网格，在 MeshPDE 与 WeatherBench22 上同时超越多类强基线。

**[(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](scientific_computing/sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)**

:   MOSAIC 用"概率扰动 + 在 HEALPix 球面网格上的 mesh-aligned 块稀疏注意力"同时解决了 ML 天气预报模型的两类频谱退化（确定性平均带来的谱衰减 + 粗化潜空间带来的高频走样），在 1.5° 分辨率上仅 214M 参数就匹敌甚至超过 6× 高分辨率的模型，单 H100 12 秒生成 24 成员 10 天预报。

**[Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor](scientific_computing/teaching_molecular_dynamics_to_a_non-autoregressive_ionic_transport_predictor.md)**

:   本文把昂贵的原子轨迹当作训练时的「特权辅助模态」，用一个双模态训练器先吃轨迹学动力学，再通过闭式岭回归把它的隐藏表示蒸到一个只看平衡结构的非自回归预测器上，在锂离子均方位移预测上比自回归 SOTA 快 200× 且更准。

**[Topology-Preserving Neural Operator Learning via Hodge Decomposition](scientific_computing/topology-preserving_neural_operator_learning_via_hodge_decomposition.md)**

:   本文提出 Hodge Spectral Duality (HSD) 神经算子，把流形 PDE 的解算子按 Hodge 正交分解拆成"低频拓扑分量（谱基底）+ 高频几何分量（FNO 辅助网格）"双分支，再用一个交换子修正项耦合二者，从而在复杂网格上同时获得高精度与守恒律保真。

**[Unbiased and Second-Order-Free Training for High-Dimensional PDEs](scientific_computing/unbiased_and_second-order-free_training_for_high-dimensional_pdes.md)**

:   本文针对 EM-BSDE 训练 loss 的离散化偏置问题，提出 Un-EM-BSDE：把单步误差用两组独立的 Monte Carlo 子样本平均后做"乘积"形成无偏估计，既消除偏置又不需要 Hessian，在 HJB/BSB/AC 等基准 PDE 上达到 Heun-BSDE / FS-PINNs 的精度但训练时间仅 1.79× EM-BSDE（相比 Heun-BSDE 的 42.91× 与 FS-PINNs 的 32.07×）。

**[WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](scientific_computing/weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)**

:   WeatherSyn 把气象预报员的报告写作流程拆解成"看图→列要点→出稿"的多模态指令任务，先建了首个覆盖 31 个美国城市、8 类天气要素的 WSInstruct 数据集，再用 SFT→RFT→DPO 三段式微调 Qwen3-VL-8B，让一个 8B 开源模型在多种评测指标上稳定打过 GPT-5-Nano、Claude-3.7-Sonnet 等闭源大模型，并对未见城市有零样本泛化能力。

---

## 🔒 LLM 安全 { #llm_safety }

**[From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](llm_safety/from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)**

:   本文把 LLM 幻觉检测从"看输出概率"切到"看 loss landscape 曲率"——在 embedding 加 Gaussian 噪声测量梯度方向与幅度的扰动，作为 Hessian 谱半径的廉价代理，在 12 个 model-dataset 组合上 AUROC 全面超越 entropy / Semantic Entropy / EigenScore 等基线。

**[From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](llm_safety/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)**

:   作者通过追踪 LoRA 微调过程中参数沿"危险/安全方向"的累积漂移，发现善意数据破坏对齐的根本机制是参数在 fine-tuning 中向危险方向单调漂移；进而提出 SQSD——用单步梯度沿两方向的投影差对每个样本打连续风险分，在 3 个模型 × 2 数据集上保持单调 ASR 排名，且能跨架构、跨规模、跨 LoRA→Full 迁移。

**[Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](llm_safety/harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)**

:   本文针对大推理模型（LRM）的幻觉检测提出 ARS：不在文本层扰动 reasoning trace，而是**直接在 trace 末端的潜表示上施加小扰动并续解码**得到反事实答案，再用"答案是否一致"作为标签训一个轻量 contrastive 头来塑形 trace-conditioned answer embedding，使后续 embedding-based detector 把幻觉与真实回答分得更开（TruthfulQA 上 AUROC $66.85\to 86.64$）。

**[Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](llm_safety/inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)**

:   本文针对大型推理模型 (LRM) 易被"逻辑残缺输入"激发过度思考的弱点，提出一个层级化遗传算法 (HGA)，在纯黑盒条件下把结构化分解后的题目当成基因，通过句子级/问题级交叉和增删变异搜索逻辑断裂的对抗样本，最高可在 MATH 上把响应长度放大 26.1 倍，制造低成本 DoS 攻击。

**[Internalizing Safety Understanding in Large Reasoning Models via Verification](llm_safety/internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)**

:   本文论证「会生成安全答案」≠「懂安全」，提出 SInternal 框架：只训练大型推理模型去 verify 自己生成答案的安全性，由此涌现的内在安全理解大幅压制 jailbreak 攻击（StrongREJECT ASR 从 41% 降到 0.6%）并成为后续 RL 的更好起点。

**[Jailbreaking Vision-Language Models Through the Visual Modality](llm_safety/jailbreaking_vision-language_models_through_the_visual_modality.md)**

:   作者提出 4 种只通过视觉输入就能越狱前沿 VLM 的攻击（视觉密码 / 物体替换 / 文本替换 / 视觉类比谜题），在 6 个前沿 VLM 上系统验证了"文本端的安全对齐不会自动迁移到视觉端"，并用 mechanistic 分析揭示了背后的层级机理。

**[Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](llm_safety/less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)**

:   论文揭示了 Test-Time Scaling (TTS) 一个被忽视的失效模式——只要把候选回复的多样性压低，TTS 反而比直接喂高对抗性 prompt 更容易输出不安全内容；并提出 RefDiv，一个用 Shannon 熵 + 参考引导双信号驱动的遗传算法，能在 MCTS 和 Best-of-N 上跨模型、跨闭源、跨 guardrail 地高效越狱。

**[Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](llm_safety/metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)**

:   把多轮 jailbreak 重新形式化为推理时的策略优化问题——在 adversarial POMDP 框架下，Attacker 与 Metacognitive Evaluator 构成闭环：Evaluator 输出的密集分析反馈被当作「语义梯度」来引导 Attacker 的 belief 更新与策略改进，从而在不重新训练任何权重的情况下，对包括 O1 / GPT-5-chat / Claude-3.7 在内的 10 个前沿模型平均 ASR 89.2%，token 消耗较强 baseline 平均降低 8.2 倍。

**[MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](llm_safety/multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)**

:   MultiBreak 用"主动学习 + 不确定性引导改写"的迭代框架把多轮越狱数据集扩到 10,389 条对话、2,665 个独立有害意图，多样性 0.942 全面碾压前作，并在 DeepSeek-R1-7B / GPT-4.1-mini 上把 ASR 相比次优数据集分别提升 54% / 34.6%。

**[OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](llm_safety/otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)**

:   OTora 提出一种全新的攻击范式 Reasoning-Level Denial-of-Service（R-DoS）：不破坏任务正确性，而是通过两阶段红队管线（先用插入感知优化诱导 agent 主动访问攻击者控制的外部资源，再在该资源里投放经 ICL 遗传搜索优化的「思考型 payload」）让 LLM agent 进入持续多轮的过度推理状态，在 WebShop / Email / OS 三类 agent 上实现 10× 推理 token 膨胀和数量级延迟攻击，且最终任务准确率几乎不变。

**[REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](llm_safety/realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)**

:   REALISTA 在 LLM 隐空间里构造"输入相关的编辑方向字典"，把对抗 prompt 优化变成一个 simplex 约束下的连续问题，既保住了 SECA 这类离散方法的语义等价/连贯，又有 LARGO 那种连续方法的搜索灵活度，首次在 GPT-5 这类闭源推理模型 free-form 输出上诱发幻觉成功。

**[SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](llm_safety/safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)**

:   SafeHarbor 把 LLM Agent 的安全防御从「静态粗粒度分类器」升级为「动态分层记忆树 + 双分数门控」，通过对抗规则生成 + 信息熵自演化让 GPT-4o 在保持 93%+ 拒绝率的同时把 benign 工具调用成功率拉到 63.6%，显著缓解 over-refusal 问题。

**[Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](llm_safety/safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)**

:   本文证明所有现有「在参数空间设约束」的 HFT 防御都会因参数冗余而被绕过，提出 Safety Bottleneck Regularization (SBR) 把防御战场搬到 unembedding 层这一几何瓶颈上：仅锚定 1 个高危 prompt 的最后一层隐状态，就能在 50 epoch 持续 HFT 攻击下把 Harmful Score 压到 < 10，同时不损 benign 任务精度。

**[Self-Debias: Self-correcting for Debiasing Large Language Models](llm_safety/self-debias_self-correcting_for_debiasing_large_language_models.md)**

:   Self-Debias 把 LLM 的去偏问题重塑为「在自回归推理链上对概率质量做公平资源分配」：用轨迹级后缀边际作为资源单位，套 Jain 公平指数防止资源在易样本上塌缩，再配 cold-start SFT 与基于一致性过滤的在线自训练，仅用 20k 标注种子就让 Qwen3-8B 在 8 个 fairness/utility 基准上的平均分从 77.5 拉到 81.7，并把基础模型「自我纠错越纠越歪」的塌缩翻转成稳定 +0.4。

**[Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](llm_safety/stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)**

:   本文指出现有 GFlowNet 红队的两大不稳定来源——partition function $Z_\theta$ 估计带来的高方差，与 toxicity classifier 给 OOD gibberish 文本的噪声 reward 引发的 mode collapse——并用三件简单组件（pairwise 对比目标 CTB 消除 $Z$、Noisy Gradient Pruning 过滤无信息 pair、Min-K Fluency Stabilizer 卡掉 gibberish）让红队攻击在 Qwen2.5-1.5B 上独特攻击数从 17 飙到 134（约 7×），ASR 维持 92%，且跨模型/跨防御迁移性全面碾压 baseline。

**[STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](llm_safety/stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)**

:   本文把 T2I 模型的整个去噪轨迹本身当成 VLM 红队攻击的"攻击面"，用一个 high-level prompt editor + low-level GRPO 微调 rectified-flow 模型的分层 RL 框架（STARE），不仅把 attack success rate 比 SOTA 提升 68%，更揭示了一个全新现象——Optimization-Induced Phase Alignment：对抗优化会自动把"概念性毒性"绑到去噪早期、"细节性毒性"绑到后期，从而把混沌的毒性形成过程变成几个可预测的"漏洞时间窗"。

**[Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](llm_safety/tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)**

:   本文用 Causal Tracing 在 LLM 内部发现"拒绝"不是终端 token 的静态向量、而是横跨上游中间层与 token 的"拒绝轨迹"(Refusal Trajectory)，并据此设计 SALO——一个只在常规对齐数据上训练、却能利用 Transformer 因果掩码不可逆性识别 GCG / AutoDAN / Prefilling 等对抗攻击的 <20M 参数检测器，把 GCG/Prefilling 上 0% 的检测率拉到 >85%。

**[Watermarking LLM Agent Trajectories (ACTHOOK)](llm_safety/watermarking_llm_agent_trajectories.md)**

:   ACTHOOK 把"软件 hook"思想搬进 agent 轨迹：在 action 边界处插入一个由秘密 key 触发的额外动作作为水印，被它训练过的 LLM 会在带 key 的 prompt 上以显著更高频率执行 hook，从而支持只通过黑盒查询就完成版权检测，平均 AUC 达 94.3 而几乎不影响下游任务表现。

---

## 📚 预训练 { #llm_pretraining }

**[Annotations Mitigate Post-Training Mode Collapse](llm_pretraining/annotations_mitigate_post-training_mode_collapse.md)**

:   作者发现 SFT 把模型对齐到一个低熵语义先验上、导致"指令模型越大越无聊"的反向 scaling，于是提出"标注锚定训练"——预训练阶段给文档配语义 tag、SFT 阶段对 tag 部分 mask loss，让推理时先采样语义再生成响应，从而在保持指令跟随能力的同时把语义多样性差距缩小 85%。

**[Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](llm_pretraining/coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)**

:   本文从表达力与可训练性两个维度系统比较连续扩散、离散掩码扩散、looped transformer，证明"连续扩散"在表达力上严格强于离散扩散并能模拟 looped transformer，但实际性能受限于解码与表征空间；据此提出 **CCDD（Coevolutionary Continuous Discrete Diffusion）**——在离散 token 空间和预训练 LLM 的上下文嵌入空间上同时扩散，由单一模型联合去噪，在 LM1B/OWT 上比 MDLM 困惑度降 25-35%，并以仅 8 步采样超过 MDLM 256 步效果。

**[CoFrGeNet: Continued Fraction Architectures for Language Generation](llm_pretraining/cofrgenet_continued_fraction_architectures_for_language_generation.md)**

:   本文把"连续分数（continued fraction）"这种具备最优有理逼近性质的函数类引入到语言生成 Transformer 中，分别为多头注意力和 FFN 设计 CoFrNet 替代模块（CAttnU/CAttnM/Cffn），通过"continuants"封闭形式把 $d$ 次除法降为 1 次，在 GPT2-xl 和 Llama-3.2B 上用 $\frac{2}{3}\sim\frac{1}{2}$ 的参数实现持平甚至更优的下游性能。

**[Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](llm_pretraining/compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)**

:   本文提出 Compute as Teacher（CaT）：把 GRPO 已经在采样的 G 条 rollouts 通过冻结锚模型"合成"出一个伪参考答案，再在非可验证领域用模型自己从该伪参考衍生的二元 rubric 给每条 rollout 打分作为 RL 奖励，从而在没有任何人工标注的情况下把推理算力直接变成监督信号，在 HealthBench 上相对基线最高提升 30%，并以 9× 更低的测试时算力匹配甚至超过 inference-time aggregation。

**[Consistent Diffusion Language Models](llm_pretraining/consistent_diffusion_language_models.md)**

:   本文指出离散扩散没有连续域 probability-flow ODE 的对应物，因此无法直接做 consistency model；作者提出用**精确闭式 posterior bridge** 作为离散域的"随机版 PF-ODE 替代品"，构造 Multi-Path Discrete Consistency (MPDC) 训练目标，要求 denoiser 在多条 stochastic bridge 路径上的预测在期望上一致，从而单阶段、teacher-free 地训出可在 2-3 步生成高质量文本的 Consistent Diffusion Language Model (CDLM)，在 unconditional / conditional 文本生成上达到 SOTA、对 AR 模型最高 $32\times$ 加速。

**[Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](llm_pretraining/data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)**

:   本文系统研究 SFT 中数据难度的作用，发现并不存在"普适最优难度"，而是存在一个**随数据规模增大而向更难方向漂移**的最优难度，并用"in-distribution 泛化 gap"与"extrapolation gap"两个 gap 的 trade-off 给出 PAC-Bayes 解释。

**[Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](llm_pretraining/decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)**

:   论文针对多任务指令微调中的跨任务梯度冲突问题，提出 Badit：先用 SVD 把预训练权重分解为一组天然正交的高奇异值 LoRA "基础能力"专家，再在训练过程中用球面 K-means 对 rank-1 分量做动态正交分组，从而把"按任务隔离参数"的传统思路改为"按基础能力解耦"，在 6 个 LLM 上平均比 GainLoRA 提升 2.68 Rouge。

**[Edit-Based Refinement for Parallel Masked Diffusion Language Models](llm_pretraining/edit-based_refinement_for_parallel_masked_diffusion_language_models.md)**

:   ME-DLM 给 masked diffusion 语言模型（如 LLaDA）加一个"解码完再编辑修补"的轻量阶段：第一阶段照常 unmask 出粗稿，第二阶段用替换/删除/插入三种 token 级编辑做并行修正，监督信号来自 edit distance 的最短编辑脚本，在只用 1/8 扩散步数的情况下 HumanEval +11.6 / GSM8K +33.6 点反超 LLaDA-Instruct。

**[Focus and Dilution: The Multi-stage Learning Process of Attention](llm_pretraining/focus_and_dilution_the_multi-stage_learning_process_of_attention.md)**

:   本文在单层 Transformer 学习马尔可夫数据的简化场景下，通过围绕一系列临界点做分阶段线性化的梯度流分析，揭示并严格刻画了注意力训练中反复出现的「聚焦—稀释」循环，并在 WikiText 与 TinyStories 上观察到一致的现象。

**[From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](llm_pretraining/from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_.md)**

:   本文系统剖析了 locate-then-edit 编辑中 backward spreading 为什么能 work 又为什么 work 得不彻底，并提出 forward replay：把第一决定层作为优化变量、再通过标准前向传播得到后续各层 target，无需额外算力就能在 MEMIT/RECT/PRUNE/AlphaEdit 之上一致涨点。

**[InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](llm_pretraining/infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)**

:   作者提出 InfoLaw：把"预训练"重新定义为"按桶累积信息"的过程，每桶信息量等于"质量密度 $f_d$ × 唯一 token 数 $M_d$ × $\log K$"再乘上一个随重复次数 $R_d$ 指数衰减的因子，最终把验证损失写成 $L = \alpha\cdot\text{info}^{-\beta}$，能在 252M-1.2B 拟合后外推到 7B / 425B token，平均误差 0.15%、最大 0.96%，并直接用来搜索最优数据配方。

**[Model Merging Scaling Laws in Large Language Models](llm_pretraining/model_merging_scaling_laws_in_large_language_models.md)**

:   作者用 10,866 个合并模型实测出一条形如 $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ 的双轴幂律：基座规模 $N$ 决定 floor，专家数 $k$ 决定 tail，且四种主流合并方法（Average、TA、TIES、DARE）都共用同一条曲线，从而把"合多少个专家、合到哪一步停"变成一个可预测、可预算的工程问题。

**[On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](llm_pretraining/on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)**

:   本文用一套精心控制"推理难度恒定、只变 horizon 长度"的 Sudoku/Rush Hour 任务，系统证明**任务 horizon 本身就是 LLM agent RL 训练崩溃的独立根因**，并提出 macro action 与 subgoal decomposition 两种 horizon-reduction 机制——它们不仅稳住训练，还让模型在更长 horizon 上实现强 zero-shot 泛化（horizon generalization）。

**[Predicting Large Model Test Losses with a Noisy Quadratic System](llm_pretraining/predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)**

:   本文提出 Noisy Quadratic System (NQS)——一个把 LLM 测试损失建模为 $L(N, B, K)$（模型大小 / 批大小 / 更新步数）的 mechanistic 损失模型，首次在 scaling law 中显式建模 batch size，并在 Pythia + OWT2 上把外推预测能力从 Chinchilla 的 ~20× 算力提升到 ~4000× 算力。

**[Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](llm_pretraining/softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)**

:   作者把传统 Softmax attention 解构为"非负化 + L1 归一化"两个独立部件，证明真正关键的是 L1 归一化而非指数，于是用 Softplus + 动态长度尺度因子换掉指数得到 LSSA，再用一次幂函数式"重权"对注意力锐化，得到的 LSSAR 在 16× 训练长度上几乎保持 validation loss 不变，并能让 GPT-109M 从轨迹数据中"重新发现"牛顿万有引力定律。

**[Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](llm_pretraining/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)**

:   作者在简化单层线性注意力 Transformer 上推出闭式训练动力学，证明正则化方法只能改变收敛速度而无法挪动收敛点（因此在 cFKA 场景几乎注定失效），数据回放则能直接改变收敛点并加大震荡幅度从而稳住旧知识，进而提出按 token 注意力贡献裁切片段、引导预训练模型生成回放语料的 STOC，在合成 + KnowEdit + IndustryCorpus 法律语料上一致比 LAMOL 更能压制遗忘。

**[Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](llm_pretraining/understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)**

:   作者把 Transformer 自注意力写成 token 间相互作用的平均场粒子系统，把 LoRA 视作低秩扰动，证明遗忘与"扰动模长"和"网络深度"两条相变曲线相关，并给出由 $V$ 的特征值 gap 控制的长时稳定条件。

---

## 🏥 医学图像 { #medical_imaging }

**[Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](medical_imaging/auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)**

:   本文提出 S(H)NAP——基于 3D 扩散桥的「移除 + 插入」生成式干预框架，把 Sybil 这一前沿肺癌风险预测模型的决策反向拆解为「肺结节主效应 + 两两交互 + 背景」的 LMPI（线性+二阶交互模型），首次以因果而非相关的方式审计出它对 ECG 电极、衣物金属扣等院内伪影的依赖以及对外周肺结节的「径向不敏感」严重失败模式。

**[EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](medical_imaging/eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)**

:   EEG-MoCE 给 EEG-based 多模态学习（情绪/睡眠/认知）每个模态分配一个**可学习曲率**的 Lorentz 流形 expert，再用"曲率大→层级结构更丰富→在 fusion 中权重更高"的 curvature-aware attention 做跨模态融合，在 EAV/ISRUC/Cognitive 三个数据集上 cross-subject 准确率分别 +14.14%、+3.34%、+7.98%。

**[Evidential Reasoning Advances Interpretable Real-World Disease Screening](medical_imaging/evidential_reasoning_advances_interpretable_real-world_disease_screening.md)**

:   EviScreen 用「正常 + 病理」双知识库做区域级证据检索，再以 cross-attention + self-attention 在当前病例和证据间做循证推理，既给出**回溯式可解释性**（哪几个历史病例支持当前判断）又给出**定位可解释性**（对比检索得到的异常图），在 4 个真实外部测试集上把高召回处的特异性提升到 SOTA。

**[Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](medical_imaging/federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)**

:   本文提出 FedHD：在异构联邦病理学场景下，用 Gaussian-mixture 特征对齐做「一对一」WSI 特征级蒸馏，再通过课程学习把跨机构合成特征逐步注入本地训练，使各机构能在不共享原始数据、不交换模型参数的前提下协作，且兼容异构 MIL 架构与特征提取器，在 TCGA-IDH / CAMELYON16 / CAMELYON17 上全面超越现有联邦与蒸馏基线。

**[From Holo Pockets to Electron Density: GPT-style Drug Design with Density](medical_imaging/from_holo_pockets_to_electron_density_gpt-style_drug_design_with_density.md)**

:   本文把结构药物设计的 condition 从"刚性 empty pocket"换成"包含配体与溶剂的 filler 低分辨率电子云"，并提出第一个 decoder-only autoregressive 的 EDMolGPT，在 DUD-E 101 个靶点上 bioactive recovery 达 41%、远超先前 ED-based 方法。

**[OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](medical_imaging/geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)**

:   OT-Bridge Editor 把"在冠脉造影上编辑一段血管狭窄"重写为"在血管-结构复合域里的约束熵 OT 问题"，用 Schrödinger Bridge 沿路径加几何投影监督，做到像素级形状/位置可控的合成造影，在 ARCADE 公开集上把下游狭窄检测 mAP@0.5 相对提升 27.8%。

**[Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models](medical_imaging/layer-specific_fine-tuning_for_improved_negation_handling_in_medical_vision-lang.md)**

:   NAST 用因果追踪 (causal tracing) 算出 CLIP 文本编码器各层对否定理解的因果贡献度 (CTE)，再以这些 CTE 做层级化梯度缩放微调 LoRA，让医学 VLM 在区分"有 / 没有某症状"时的语义敏感度大幅提升，并把肯定-否定准确率差距从 21.6% 缩到 4.2%。

**[Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](medical_imaging/learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)**

:   L3-PPI 把生物学里的 "L3 规则"（蛋白质对之间的 length-3 路径越多越可能相互作用）变成可学习的 graph prompt：用预训练 GNN 识别 L3 模式，再用门控网络生成虚拟 L3 路径并按 PPI 标签正则路径数量，做成一个即插即用的分类头，把任意 PPI 表征模型平均涨 2-4 个点。

**[Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](medical_imaging/marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)**

:   本文提出 DiffDT：用一个条件 Latent Diffusion 框架把电子病历（ICD-coded 事件序列）与多器官生物标记数字孪生（脑/心/肝/肾的影像衍生表格特征与脑功能连接 SPD 矩阵）连起来，关键创新是一个基于 Cholesky 分解的 SPD-VQVAE 把 $\mathcal{O}(N^3)$ 的 SPD 流形扩散降到流形保形且高效的潜空间，再让 AR 模型借“生成数字孪生 → 预测下一个 ICD”这条中介路径完成多通路疾病推理；在 UKB 上对 1944 类疾病的下一次预测 AUC 提到 0.91，刷新 SOTA。

**[MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](medical_imaging/meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)**

:   MEG-XL 用 2.5 分钟（191k token）的 MEG 上下文做 mask token 预训练（比此前长 5–300×），再微调到 50 词的脑到文本任务上，仅用 1 小时数据就达到 SOTA 监督方法 50 小时的解码精度，并显著超过所有 brain foundation models。

**[Protein Circuit Tracing via Cross-layer Transcoders](medical_imaging/protein_circuit_tracing_via_cross-layer_transcoders.md)**

:   作者把 NLP 中的 cross-layer transcoder 搬到蛋白质语言模型 ESM2 上,提出 ProtoMech 框架以 < 1% 的稀疏潜变量电路恢复 79% 的下游性能,并能沿电路 steering 设计出高 fitness 的蛋白变体,在 70%+ 案例中击败基线。

**[Scaling Vision Transformers for Functional MRI with Flat Maps](medical_imaging/scaling_vision_transformers_for_functional_mri_with_flat_maps.md)**

:   把 3D fMRI 体积按"皮层展平图"投影成 2D 视频后直接喂给标准 spacetime MAE-ViT，得到一个在 2.1K 小时 HCP 数据上训练的 CortexMAE：在认知状态解码上大幅超 SOTA，验证 flat map 是体素 (volume) 和脑区平均 (parcellation) 之间的"goldilocks zone"；同时发布首个开源 fMRI 基础模型基准 Brainmarks，给出 fMRI 模型的第一份系统 scaling law 与一个"个体特质预测仍打不过简单功能连接 baseline"的诚实 null result。

**[SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](medical_imaging/sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)**

:   SIGMA 用 token 级对比损失把同一分子不同 SMILES 排列的隐状态强制对齐到同一条轨迹，并配套提出 IsoBeam 在解码阶段剪掉同构冗余路径，让序列模型在化学空间中真正"按图而非按字符串"思考。

**[SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](medical_imaging/synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)**

:   SynerMedGen 提出"生成对齐理解（generation-aligned understanding）"原则——把理解任务直接从同一份配对合成数据里派生出来（CTS / MI / TIA 三个任务），先两阶段训练让理解分支学到对合成有用的表征，再迁移到 latent flow matching 生成分支，在 22 个医学合成任务上同时碾压专用合成模型和已有统一 MLLM。

**[TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](medical_imaging/td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)**

:   TD3B 把激动剂/拮抗剂的设计当作「方向性转移算子」生成任务，用一个方向 Oracle + 亲和力门控 + 树搜索摊销微调的掩码离散扩散框架，让预训练肽段生成器学会写出能定向偏移蛋白质活/失活构象转移的多肽序列。

**[Towards A Generative Protein Evolution Machine with DPLM-Evo](medical_imaging/towards_a_generative_protein_evolution_machine_with_dplm-evo.md)**

:   本文提出 DPLM-Evo，把蛋白质语言模型的离散扩散从"只支持掩码替换"扩展为"显式建模替换+插入+删除三种进化编辑"，通过把变长观测序列解耦到上采样长度的隐对齐空间 + 上下文化进化噪声核，实现变长进化生成、进化轨迹式的蛋白质后编辑/优化，并在 ProteinGym 单序列变体效应预测上取得 SOTA。

**[Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](medical_imaging/towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)**

:   本文指出单细胞基础模型 (scFM) 蕴含丰富但被"重建式预训练"遮蔽的基因调控知识，并提出 Virtual Value Perturbation 与 Gradient Trajectory 两种探针，从冻结的 scFM 中蒸馏出可跨基因/跨数据集泛化的成对基因特征，在 BEELINE 基准上把 AUPRC 从 ~0.5 推到 0.8–0.97，开启了"通用 GRN 推断 (UGRN)"这一新范式。

---

## 🛡️ AI 安全 { #ai_safety }

**[ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](ai_safety/actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)**

:   本文提出一个分层框架 ACTG，将隐私文本生成分解为特征学习与条件文本生成两个子任务；进一步引入 Anchored RL，通过混合强化学习目标与基于最优 N 选一的 SFT 锚点，在保持文本保真度的前提下提升条件生成器的指令跟随能力，在生物医学数据上相比先前工作提升 20% MAUVE。

**[Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](ai_safety/angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)**

:   作者首次系统评估 7 种主流可塑性干预 (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) 对深度强化学习 (DRL) 后门攻击的影响 (14,664 个实验)，发现只有 SAM 是"恶魔"——能显著加剧后门威胁；据此提出"Sweeper-Converter-Connector" 鲁棒后门注入框架并给出基于 loss landscape 锐度的检测信号。

**[Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing](ai_safety/certified_robustness_under_heterogeneous_perturbations_via_hybrid_randomized_smo.md)**

:   本文把随机平滑（RS）从"只支持单一连续或离散输入"扩展到"离散 token + 连续图像"的混合扰动场景，通过一个混合 Neyman–Pearson 分析得到一个**一维、连续、可逆**的似然比 CDF，从而把原本组合爆炸的离散 knapsack 问题变成可解的根求解问题，并在 LLaVA-Guard 多模态安全过滤上给出首个针对"图文联合不安全"的 model-agnostic 证书。

**[DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](ai_safety/dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)**

:   本文提出 DP-KFC：基于"Fisher 矩阵的标度由架构决定、相关结构可用模态级频谱统计近似"的观察，用结构化合成噪声（图像用 $1/f^\alpha$ pink noise，文本用 Zipf 采样）探测网络重建 KFAC 预条件子，既不消耗隐私预算也不引入分布偏移，在强隐私（$\varepsilon\le 3$）下持续超过 DP-SGD 与公共数据预条件方法。

**[Dual-branch Robust Unlearnable Examples](ai_safety/dual-branch_robust_unlearnable_examples.md)**

:   本文提出 DUNE：把不可学习样本（UE）的扰动从单一空间域扩展到"空间 + 色彩"双域优化，使扰动特征对齐到 shift-induced 标签并配合预训练模型集成增强，在 CIFAR-10 / ImageNet 上对 7 种主流防御（含 ECLIPSE、ISS-J、COIN）保持鲁棒，平均测试精度比 12 个 SOTA UE 方案再低 14.95%–50.82%。

**[Fair Dataset Distillation via Cross-Group Barycenter Alignment](ai_safety/fair_dataset_distillation_via_cross-group_barycenter_alignment.md)**

:   本文揭示数据集蒸馏 (DD) 会放大原始数据中的偏差——根源是「子组样本量不平衡」与「子组表征分离度」的交互作用，并提出 COBRA：用各子组表征的（与组大小无关的）barycenter 作为蒸馏目标，可在多个 DD 框架上同时降低 EOD、提高准确率。

**[FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](ai_safety/fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)**

:   针对原型类联邦学习中"对局部原型直接平均会继承客户端偏差"的问题，本文用一组可学习的全局超原型 (hyper-prototypes)，通过梯度匹配在服务器侧模拟集中式训练得到的原型，再配合客户端对比学习与对齐损失显著提升异质场景下的精度。

**[Frequency Matching in Spiking Neural Networks for mmWave Sensing](ai_safety/frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)**

:   本文从「机制-数据对齐」角度证明 LIF 脉冲神经元等价于一个一阶 IIR 低通滤波器，并提出根据毫米波信号的判别频谱来设定膜衰减系数 $\beta$，使 SNN 在四个常用 mmWave 数据集上平均比 ANN 提高 6.22% 精度并降低 3.64× 理论能耗。

**[LAPRAS: Learning-Augmented PRivate Answering for Linear Query Streams](ai_safety/lapras_learning-augmented_private_answering_for_linear_query_streams.md)**

:   LAPRAS 用一个"哪些查询会来"的预测器把在线 DP 查询流分成预测内/外两类，预测内的用离线最优 Matrix Mechanism 一次性低噪释放，预测外的用 Smooth Allocation 根据流中已观测到的"未预测查询"位置在线估计总数并平滑分配预算，在预测准时几乎追平离线最优、预测差时退化到在线 baseline 水平。

**[Limits of Convergence-Rate Control for Open-Weight Safety](ai_safety/limits_of_convergence-rate_control_for_open-weight_safety.md)**

:   作者把"开源权重安全"形式化为"如何延缓恶意 fine-tune 的收敛速度"，证明 Hessian 谱的最大奇异值由权重谱下界决定，由此设计了能严格减慢一阶/二阶优化的 SpecDef 算法，但同时证明任何此类收敛率控制方法都能被攻击者以"线性模型尺寸增加"的代价绕过。

**[MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](ai_safety/metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)**

:   把多个客户端在私有数据上独立微调出的领域专家，无需共享私有数据就能合并成一个可部署的 MoE 模型——核心是用 relevance-weighted DPP 从公开数据里选「既相关又多样」的代理样本，先做 proxy-aligned 专家训练再训 context-aware router，从而对齐专家行为与代理监督，显著优于 FlexOlmo 等仅依赖相似度选代理的方法。

**[Position: Embodied AI Requires a Privacy-Utility Trade-off](ai_safety/position_embodied_ai_requires_a_privacy-utility_trade-off.md)**

:   本文是一篇 position paper，主张具身 AI 的隐私不能用单阶段补丁解决，必须当作横跨 instruction / perception / planning / interaction 全生命周期的架构级动态控制信号，并提出 SPINE 框架，用 L1-L4 四级隐私分类矩阵在每个阶段联动调整智能体行为。

**[Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](ai_safety/privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)**

:   作者给"差分隐私零阶优化（DP-ZOGD）"首次证出了**收敛的 hidden-state DP 上界**——通过设计一个"定向 + 各向同性"混合噪声机制并构造一个介于两条相邻轨迹之间的辅助过程，绕开了零阶更新缺乏全局 Lipschitz 性这一技术障碍，揭示出"扩大每步采样方向数 $K$ 反而能降隐私损失"这一前所未知的 DP 算法设计准则。

**[Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](ai_safety/scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)**

:   针对现有联邦多源无监督域适应 (UMDA) 方法只能处理 2–6 个源、源数一多就训练不稳或算力爆掉的问题，作者提出 GALA：把所有源随机分成若干小组、组间对预测分布做差异最小化（把 $O(N^2)$ 的两两对齐压成线性），再叠一个基于质心+温度的相似度加权挑出真正贴近目标域的源——在新建的 Digit-18 (18 源) 基准上稳定收敛，且把基线一一推开。

**[The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](ai_safety/the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)**

:   本文构造了一个程序化生成的"合成 Web"环境,通过在搜索 rank 0 注入单条高可信度蜜罐误信息,因果性地测出 GPT-5 等前沿 LLM agent 在 1/数千的对抗污染下准确率从 65% 暴跌到 18%,且模型不会增加搜索、依然高置信度作答,揭示了根深蒂固的"位置锚定"失败模式。

**[VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](ai_safety/vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)**

:   作者构造了 10 万张图、33 个细粒度类别、19 万+ 实例的大规模视觉隐私数据集 VPD-100K（覆盖人脸/屏上 PII/物理证件/位置标记四大域），并提出三件套频域增强模块（FDAF + 自适应频谱门控 + 频域一致性损失）插入 YOLOv10 的 Neck，使 YOLOv10-L 在 VPD-100K 上 AP 从 53.8 涨到 58.6（+4.8），同时在 7.51ms 延迟下稳定跑直播流。

---

## 🦾 LLM Agent { #llm_agent }

**[A Minimal Agent for Automated Theorem Proving](llm_agent/a_minimal_agent_for_automated_theorem_proving.md)**

:   本文提出 AxProverBase——一个极简的 Lean 4 定理证明智能体，仅靠"编译器反馈 + 自管理笔记本 + 轻量工具搜索"三个组件，在不微调的前沿 LLM（Claude Opus）上达到甚至超越 Hilbert/Seed-Prover 等专用系统，成本却低出 100 倍。

**[Adaptive Querying with AI Persona Priors](llm_agent/adaptive_querying_with_ai_persona_priors.md)**

:   作者把"LLM 在 persona 条件下产生的回答分布"打包成一个有限混合的贝叶斯先验，让用户在仅被问几道题的情况下，通过对 persona 后验做闭式更新来高效预测其他回答，性能上压过经典 CAT/IRT 基线。

**[Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](llm_agent/agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)**

:   通过 Monte-Carlo rollout 量化"哪些回合的 thought / observation 可以省"，再用冷启动 SFT + 双采样 omit-aware GRPO 训出能自适应跳过冗余思考和观测的 8B agent，五个基准上 token 用量大降而准确率与七大前沿模型持平。

**[AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](llm_agent/agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)**

:   作者把"对黑盒 agent 系统反推一个等价白盒 workflow"作为新任务 AWR，用 MCTS 在 agent 原语序列空间中搜索，再配上一种基于评分动态着色的 Red-Black 剪枝来平衡深度与宽度，在五个真实领域上实现可解释的白盒重建。

**[BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](llm_agent/bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)**

:   BioAgent Bench 给"用 LLM agent 跑生物信息学 pipeline"这件事造了一个端到端的评测套件——10 个真实 bioinformatics 任务 × 10 个 frontier/open-weight 模型 × 3 个 agent harness，配合 LLM 判官评分和 corrupted/decoy/prompt-bloat 三类扰动测试，发现前沿模型能完成 90%+ pipeline 但鲁棒性堪忧。

**[DiscoverLLM: From Executing Intents to Discovering Them](llm_agent/discoverllm_from_executing_intents_to_discovering_them.md)**

:   DiscoverLLM 把 "用户没想清楚自己要什么" 形式化为意图层级树的渐进发现过程，用可奖励的层级化用户模拟器训练模型在不清晰时主动发散探索、在清晰时收敛执行，在创意写作 / 技术写作 / SVG 三任务上比 CollabLLM 等 baseline 满意度 +10%、对话长度 -40%。

**[EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](llm_agent/evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)**

:   EvolveR 给 LLM agent 套一个「在线交互 → 离线自蒸馏成原则库 → GRPO 策略进化」的闭环生命周期：agent 不再丢弃过去轨迹，而是把自己的成功失败抽象成可检索的「策略原则」，再用 RL 学会**如何用自己的原则**去解新问题，在 7 个多跳 QA benchmark 上明显跑赢 Search-R1 等 RL agent baseline。

**[ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](llm_agent/excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)**

:   本文构建了首个评测 LLM Agent 端到端做"网络威胁调查"的 benchmark ExCyTIn-Bench：从真实 Azure 租户的 57 张安全日志表里，用 alert-entity 二部图自动生成 7542 道带证据链的 SQL 问答题，并提供 MySQL 环境让 Agent 通过查询日志、多跳追踪证据来回答，目前最强模型 Claude-Opus-4.5 也只能拿 0.606 的 reward。

**[Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](llm_agent/group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)**

:   针对集中式多模态融合带来的"模态主导"和"虚假模态耦合"两个痼疾，GCL 把多模态学习重写为**两阶段四 agent 的协议化协作**：第一阶段由 Routing/Auditing agent 用边际预测增益逐样本决定哪些跨模态交流被允许，第二阶段由 Public-Factor/Aggregation agent 把共享语义与私有特化解耦后再聚合，在 MOSI/MOSEI/MIntRec 上拿到 SOTA。

**[Internalizing Agency from Reflective Experience](llm_agent/internalizing_agency_from_reflective_experience.md)**

:   本文提出 LEAFE 框架，让 LLM agent 通过反思失败轨迹生成「失败→回滚→修正→成功」的经验数据，再用 SFT 蒸馏出 feedback-grounded 的恢复能力，在 CodeContests、WebShop、ALFWorld 等长程任务上把 Pass@128 拉高最多 14%，远胜 GRPO 等 outcome-driven RL。

**[Position: Agentic AI Orchestration Should Be Bayes-Consistent](llm_agent/position_agentic_ai_orchestration_should_be_bayes-consistent.md)**

:   这篇 position paper 主张：不要再尝试让 LLM 本身 "Bayesian"（那条路在工程上和理论上都跳不过去），而是把贝叶斯结构搬到 agentic AI 的**编排控制层**——让控制器维护一个低维任务级隐变量的信念，按 Bayes 规则在 agent/工具返回的"消息观测"上更新，并用期望效用或 value-of-information 做路由、停止、升级和预算分配。

**[Position: Assistive Agents Need Accessibility Alignment](llm_agent/position_assistive_agents_need_accessibility_alignment.md)**

:   这是一篇 position paper，作者通过对 417 篇文献中 778 个盲人辅助任务实例做系统综述，论证 "accessibility alignment" 应当被视为与 helpful/harmless/honest 并列的 Agent 一级对齐目标，并提出覆盖目标-交互-风险-生命周期四维度的设计 pipeline。

**[PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](llm_agent/praglocker_protecting_agent_intellectual_property_in_untrusted_deployments_via_n.md)**

:   PragLocker 用 "代码符号初始化 + 黑盒目标模型反馈下的噪声注入" 两阶段策略，把 agent system prompt 编码成一段只能在 target LLM 上 work、迁移到其它任意 LLM 都会失效的 obfuscated text，从而在 prompt 被部署侧窃取时让攻击者无法在自己的 LLM 上复用。

**[ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](llm_agent/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)**

:   ReSeek 给 RL-trained 搜索 agent 增加一个 JUDGE 动作 + 用 BGE-reranker 计算"理想判断"作为过程奖励,使 agent 能在每次检索后软性"屏蔽"无效信息并重新查询;同时提出 FictionalHot 这一基于虚构实体的抗污染评测,Qwen2.5-7B 上平均 EM 达到 0.377,比 ZeroSearch 高 +3.1。

**[Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](llm_agent/video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)**

:   Video2GUI 用「元数据粗筛 → 视频质量精筛 → Gemini-3-Pro 提任务/动作 → 高分辨率三帧精确空间 grounding」四段流水线把 5 亿条 YouTube 视频元数据炼成 WildGUI（12.7M 轨迹、124.5M 截图、1500+ 应用），并把 Qwen2.5-VL/Mimo-VL 在多个 GUI grounding 与 agent benchmark 上提升 5–20%。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](llm_agent/when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   CAIA 用 17 个前沿大模型在 178 个时间锚定的加密货币真实任务上构建首个"对抗性高风险"agent 基准，发现：无工具时所有模型只有 12–28% 准确率（接近随机猜测），有工具时最强 GPT-5 也只到 67.4% vs. 人类入门分析师 80%；更致命的是模型 55.5% 的工具调用偏向"不可靠的网页搜索"而绕过权威链上数据，导致 Pass@k 指标系统性掩盖了"靠试错碰运气"的危险行为。

---

## 📐 优化/理论 { #optimization }

**[Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality](optimization/adaptive_estimation_and_inference_in_semi-parametric_heterogeneous_clustered_mul.md)**

:   本文桥接双重机器学习与聚类多任务学习，提出自适应框架结合 Neyman 正交性与数据驱动的配对融合罚项，在异质（可能无限维）噪声的半参数设置中精确恢复任务潜在聚类、以汇总率达到预言水平，并建立渐近正态性，实现有效统计推断。

**[Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](optimization/budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)**

:   首次给"预算受限 + 私有成本"的子模社会福利最大化采购拍卖给出有近似比保证的真值机制 BFM-SWM——用几何递增阈值的降序时钟拍卖 + 单点保护 + 价/付率参数 $\beta$ 实现非负盈余 + 预算可行，一般子模函数 0.0328-近似、单调子模 0.0877-近似；副产品 BFM-VM 把估值最大化的确定性最佳近似比从 1/64 提升到 $1/(12+4\sqrt{3})\approx 0.0528$，并将运行时间从 $\mathcal{O}(n^2\log n)$ 降到 $\mathcal{O}(n\log n)$。

**[FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](optimization/fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)**

:   本文提出 FAB——首个面向时变有向图分布式双层优化的纯一阶算法，将 AB/Push-Pull 通信与值函数惩罚法相结合，给出非渐近 $\mathcal{O}(K^{-2/3})$ 收敛率，并顺带解决了 AB/Push-Pull 在时变有向图非凸场景下收敛率长期悬而未决的开放问题。

**[Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](optimization/learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)**

:   训练一个轻量网络预测线性指派问题 (LAP) 的对偶变量 $\hat{u}$，用 Min-Trick 构造可行对偶 $\hat{v}$，将其作为 LAPJV 精确求解器的暖启动，从而在保持最优性的同时把 $N=16{,}384$ 规模实例端到端加速 $2\times$ 以上。

**[Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)**

:   本文用 empirical NTK 作为统一视角，证明 zeroth-order SGD 引出的 eNTK 等价于把 first-order eNTK 投影到由微扰张成的随机子空间，从而通过 Johnson-Lindenstrauss 引理解释为何 ZO 方法在十亿参数 LLM 上仍然 work：误差只取决于输出维度 $V$ 和微扰数 $P$，与模型维度 $d$ 无关。

**[Learning to Approximate Uniform Facility Location via Graph Neural Networks](optimization/learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)**

:   本文为 Uniform Facility Location 设计了一个把经典近似算法 SimpleUniformFL 神经化的 MPNN，**既可用无监督期望成本损失端到端训练，也具备 $\mathcal{O}(\log n)$（递归版还能到 $\mathcal{O}(1)$）的可证明近似界**，实验上既打过 SimpleUniformFL 经典算法、也逼近 ILP 最优。

**[On the Convergence Rate of LoRA Gradient Descent](optimization/on_the_convergence_rate_of_lora_gradient_descent.md)**

:   本文首次在不假设 adapter 矩阵有界、不要求重参数化损失 Lipschitz 平滑的前提下，证明了原始 LoRA 梯度下降的最小梯度范数以 $O(1/\log T)$ 速率收敛（若参数范数有界则恢复经典 $O(1/T)$），并据此设计了与理论严格对应的自适应/归一化学习率，在 logistic regression、ResNet-18、TinyLlama 上验证了训练加速与稳定性提升。

**[Probing Neural TSP Representations for Prescriptive Decision Support](optimization/probing_neural_tsp_representations_for_prescriptive_decision_support.md)**

:   作者把训练好的 TSP 神经求解器视作"可迁移编码器",用冻结表征 + 轻量探针预测两类昂贵的运筹敏感性查询(节点移除与边禁用),系统证明探针准确率随求解器质量单调提升,可以与传统启发式集成达到 SOTA。

**[RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](optimization/rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)**

:   本文提出 RL-SPH —— 一种不依赖外部 ILP 求解器、能独立产出 100% 可行解的端到端强化学习启发式算法，用「可行性奖励 + 双阶段策略 + 可行性感知邻域搜索」让 Graph Transformer Agent 在包含非二元整数变量的 ILP 上把 primal gap 平均降低 28.6 倍。

**[RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)**

:   本文基于 Transformer 层级 Hessian 的「行块对角占优」结构，把 Muon 优化器里昂贵的 Newton-Schulz 正交化换成一次行级 $\ell_2$ 归一化，将每步预条件复杂度从 $\mathcal{O}(mn\min(m,n))$ 降到 $\mathcal{O}(mn)$，在 GPT-2 / LLaMA 预训练上 wall-clock 提速 13–44×、ppl 不降反略升。

**[Streaming Sliced Optimal Transport](optimization/streaming_sliced_optimal_transport.md)**

:   Stream-SW 是首个能在"样本流"上估计 sliced Wasserstein 距离的算法：每个一维投影上用 KLL/quantile sketch 维护近似分位函数，把 1D Wasserstein 的闭式积分变成可流式更新的估计量，空间复杂度对样本数仅对数级，从而把 SOT 带入 IoT / 边缘设备等"看一次就丢掉"的场景。

**[Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](optimization/support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)**

:   SPADE 用一个条件扩散模型替代传统回归代理来建模 $p(y\mid\boldsymbol{x})$，并通过"均值/排序校准"+"kNN 支撑度正则（均值收缩 + 方差膨胀）"把数据先验隐式注入到代理里，使离线黑盒优化在 Design-Bench 和 LLM 数据混合任务上稳定达到 SOTA。

**[Test time training enhances in-context learning of nonlinear functions](optimization/test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)**

:   本文给单层 softmax-attention transformer + LoRA 测试时微调的组合建立了首个严格泛化界，证明在 single-index 多项式任务上 TTT 把 ICL 的样本复杂度从 $r^{\Theta(\mathrm{ie}(\sigma_*))}$ 压到 $r^{\Theta(\mathrm{ge}(\sigma_*))}$ 并允许 link 函数逐任务变化、推理误差可随上下文长度 $\to$ 噪声水平。

**[Transformed Latent Variable Multi-Output Gaussian Processes](optimization/transformed_latent_variable_multi-output_gaussian_processes.md)**

:   本文提出 T-LVMOGP：把多输出 GP 的核心建模问题——跨输出协方差 $k_{p,p'}(x, x')$ 的构造——转化成"在 Lipschitz 正则的 RCNN 嵌入空间里用单个标量基核做内积"，并完整嵌入 SVGP 框架，使 MOGP 第一次能可扩展且高表达力地处理 $P > 10000$ 输出（含 ZINB 似然的空间转录组数据），同时全面胜过 SV-LMC / OILMM / GS-LVMOGP 等基线。

**[Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](optimization/turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)**

:   本文把"陈旧的"块循环坐标下降梯度估计存进 FIFO buffer，配上 momentum 衰减重用，并证明这等价于带 warm-start 的 BCCD；同时给出反直觉结论——更大的有限差分步长 $\epsilon$ 会隐式平滑 loss landscape、降低有效 Lipschitz 常数，从而让 stale gradient 反而能换来稳定下降。

---

## 🤖 机器人/具身智能 { #robotics }

**[Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](robotics/decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)**

:   针对"训练任务到全新任务"的零样本机器人操作，作者把 demo 拆成"原子技能-动作对"作为中间表示，再用 dual-library（动态库按视觉/计划相似度检索 + 静态库按 IDF 加权补全缺失技能 token）给 LLM 提供 skill-comprehensive in-context demonstrations，从而把"模仿轨迹"升级为"组合技能推理"。

**[Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](robotics/drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)**

:   本文提出 CAPS：把"指令漂移"重新解释为系统性采样误差，用 SNR（=$\log|\mathcal{A}|-\mathcal{H}$）作为元认知开关，仅在高熵"Pivotal Window"触发基于幂分布 $\pi\propto p^\alpha$ 的 Metropolis-Hastings 迭代精修，在 RoboTwin、Simpler-WindowX、Libero-long 上 training-free 超越 OpenVLA 和 TACO。

**[Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](robotics/embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)**

:   本文把「视觉—动作归因」重新表述为干预估计问题，提出 ISS（介入显著性分数）和 NMR（干扰物质量比）两个指标，用 Bernoulli 掩码 + 高斯模糊扰动 + Action MSE 代理 KL 散度的方式量化 VLA 策略到底依赖哪些视觉区域，并证明 NMR 与 OOD 任务成功率呈 $r = -0.77$ 的强负相关——是预测 VLA 泛化能力的便宜诊断工具。

**[From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](robotics/from_imagined_futures_to_executable_actions_mixture_of_latent_actions_for_robot_.md)**

:   MoLA 用一组在大规模机器人数据上预训练好的"模态感知逆动力学模型 (IDM)"，把视频生成模型预测出的未来帧翻译成语义/深度/光流三路离散潜动作，再让策略头基于这些动作中心的表征做控制，从而在 CALVIN、LIBERO、LIBERO-Plus 以及真实 UR5e 上把"想象-执行"接口做得既稳又准。

**[HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](robotics/hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)**

:   HDFlow 用扩散模型生成稀疏战略子目标、用整流流生成稠密轨迹，再叠加能量引导和流形投影，构建一套快慢分工的双层规划器，把家具组装等长程稀疏奖励任务的成功率拉高 20~30 个百分点。

**[Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](robotics/latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)**

:   LaRA-VLA 把 VLA 模型里的文本 CoT 和视觉 CoT 全部内化为连续 latent，通过三阶段 curriculum 训练（显式 CoT → latent 替换 → 动作专家适配）让推理留在 latent 空间里完成，推理延迟相比显式 CoT 降低高达 90%，控制频率重回实时区间。

**[Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering](robotics/mitigating_error_accumulation_in_continuous_navigation_via_memory-augmented_kalm.md)**

:   把无人机连续 VLN 的 step-by-step 预测重写成"递归贝叶斯估计 = GRU 先验 + 记忆库似然 + 可学习卡尔曼增益"的闭环, 在 TravelUAV 上仅用 10% 数据微调就把 L1-Full 的 SR 从 17.6% 推到 25.9%, 同时把 100 步后还在不断累积的位置漂移压平到 30–40 米。

**[Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](robotics/optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)**

:   本文把匿名多机器人路径规划（MAPF）证明为一类**马尔可夫多边际最优传输（MMOT）**，从而把原本 $K^{T+1}$ 维的传输张量压缩成多项式规模 LP（P1），并通过全单模性保证最优解整数性；再把它推广为 Schrödinger bridge 得到 Sinkhorn 风格 entropic 松弛 P2 产出"影子传输"，最后在影子上做剪枝并解 LP（P3）恢复整数解，在 $K^{1.15}$ 复杂度下实现 3.6×–7.1× 加速、代价差距 <10%。

**[Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](robotics/plan_in_sandbox_navigate_in_open_worlds_learning_physics-grounded_abstracted_exp.md)**

:   本文提出 SAGE：在物理约束的语义沙盒里自动合成大量导航任务+IF-THEN 经验规则，用混合提示采样 + 非对称自适应裁剪的 GRPO 把这些经验蒸馏进 VLM 策略，最终在 A-EQA 上把 LLM-Match 成功率从 43.5% 拉到 53.2%（2B）/ 60.2%（4B），并能迁移到真实室内机器人。

**[Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation](robotics/plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation.md)**

:   本文提出 PLMD：把 BEV 语义图与障碍图合并成 Label Map，用 DDPM 在障碍先验调制下补全未探索区域的语义+障碍标签，作为即插即用模块挂在任意 GON 策略上，在 ON / IIN / MRON 三类任务的 HM3D/MP3D 上一致刷新 SOTA。

**[Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](robotics/seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)**

:   针对 VLA（vision-language-action）模型在简单扰动下性能崩塌的问题，本文用"提取语义/几何条件 → 改写 caption → 条件视频扩散重渲染"的视频迁移流水线给仿真数据补上视觉与环境多样性，同时配以三段式 velocity caching 把生成时间砍掉 61% 以及 difficulty + diversity 双驱动的 coreset 采样仅选 10% 关键轨迹，最终在 Robotwin 2.0、LIBERO-Plus 和真机上让 RDT-1B / $\pi_0$ 涨 5–15%。

**[STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction](robotics/step_warm-started_visuomotor_policies_with_spatiotemporal_consistency_prediction.md)**

:   STEP 给 diffusion policy 接了一个轻量的 "前一段历史动作 + 当前观测 → 下一段动作"的 Transformer 预测器, 用它的输出作为去噪起点 (warm-start), 把 100 步去噪压到 2 步, 又附带一个 "动作变化太小就注一点噪声"的执行死锁防御机制, 在 9 个仿真任务和 2 个真机任务上比 BRIDGER / DDIM 平均提 21.6% / 27.5% 成功率。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](llm_efficiency/a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)**

:   本文建立首个显式纳入 KV 缓存显存动态的 LLM 推理排队模型，给出闭形稳定性条件 $\lambda < \mu(1-\delta)$，使运维人员可直接计算所需 GPU 数；在单 GPU、8 GPU 集群与 LongBench 真实数据上验证误差均 $\leq 10\%$。

**[Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](llm_efficiency/not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)**

:   本文指出多轮对话场景下传统 Prefill-Decode 分离架构因每轮都要 P→D 重算并传输 KV 而严重低效，提出 PPD（Prefill-capable Decode）动态路由系统，让 decode 节点根据 SLO 权重决定是否本地处理 Turn 2+ 的 append-prefill，把 Turn 2+ TTFT 降低约 68%。

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](llm_efficiency/oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe 把 LLM 服务的「资源分配 + 并行策略 + 请求路由」联合建模为流网络上的双层最大流问题，配合 LSTM 工作负载预测和基于 GPU 互联的 ad hoc 模型切换，应对真实流量在空间（不同请求类型）和时间（成分随时刻变化）两个维度的异质性，端到端 P99 延迟和吞吐相比 vLLM 平均提升 1.5×、最大 2×。

**[PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](llm_efficiency/pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)**

:   本文提出 PipeSD：把投机解码（speculative decoding）从端云顺序执行改成 token-batch 流水线，并用双阈值 NAV 触发 + 贝叶斯自动调参替代固定 draft 长度，在 5G 带宽的真实端云测试床上拿到 1.16×–2.16× 加速、14–25% 云端能耗下降。

**[Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](llm_efficiency/plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)**

:   本文提出 Dilated Unmasking Scheduler (DUS)：用「等距空隙」预定义不依赖模型置信度的 unmask 顺序，把每块 $B$ 个 token 的 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$，在 LLaDA / Dream / DiffuCoder 上拿到 5.8× wall-clock 加速且质量优于基于置信度的并行 planner。

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](llm_efficiency/scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout 把百万级 token 的长文本理解重新建模为"主动信息觅食"过程，引入与交互轨迹解耦的、带来源锚点的 epistemic state $\mathcal{E}_t$ 作为唯一推理底座，并通过 gap-diagnosed 自评估迭代收缩到查询充分子集，在 LooGLE-v2 和 $\infty$Bench 上既追平甚至超过 Gemini-3-Pro 等前沿模型，又把 token 成本降低到约 $1/8$。

**[SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](llm_efficiency/slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)**

:   SLAY 把受物理"逆平方相互作用"启发的 Yat-kernel 通过 (1) 球面归一化 (2) Bernstein 定理的 Laplace 积分表示 (3) Gauss-Laguerre 求积 (4) 多项式+指数核张量积正随机特征四步连击线性化，得到 $O(L)$ 时间复杂度且与 softmax 几乎无差异的注意力机制。

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](llm_efficiency/theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   本文为新兴的 Attention-FFN 解耦 (AFD) 推理架构提供首个理论框架,基于"prefill 长度有限均值 + decode 长度服从几何分布"的概率工作负载模型,推导出 rA-1F 拓扑下最优 A/F 比的闭式解 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,并用 trace-calibrated 模拟器验证理论与实测最优值偏差 <10%。

**[Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](llm_efficiency/towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)**

:   作者搭了一套基于 NVML 的分阶段 GPU 能耗采集框架，把蒸馏流水线拆成"教师侧 + 学生侧 + 评估"逐段计量，发现一次性运行时教师 logit 缓存 / 合成数据生成才是大头，让 KD 和 synthetic SFT 在 1B–13B OLMo-2 学生上反而比直接 SFT 多耗约 $2.4\times$ 能量，并给出闭式 break-even 公式说明只有当教师产物被复用 $N^*$ 次以上时蒸馏才真"省电"。

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](llm_efficiency/training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   本文提出一套训练与推理共享完全相同的分段前向执行语义的长上下文 LLM 框架：跨段只保留固定长度的可微分 KV 尾部 + 一条仅前向的检索旁路，在 LLaMA2-7B 32K/80K 上以约 $6\times$ 更低的 prefill 峰值显存达到与全注意力可比甚至更好的 LongBench/RULER 表现。

**[Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](llm_efficiency/understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)**

:   本文把现代 LLM 长上下文推理中的稀疏注意力、RAG、压缩上下文记忆等优化统一为四阶段 "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" 内存处理流水线，定量证明该流水线占整体延迟 22%-97% 且各阶段计算特性高度异构，并据此提出 GPU-FPGA 异构系统：把规则/算密集操作留 GPU、把稀疏/不规则/访存密集操作 offload 到 FPGA，在 MI210 + Alveo U55C 上取得最多 2.2× 端到端加速和 4.7× 能耗下降。

---

## 📈 时间序列 { #time_series }

**[CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](time_series/combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)**

:   CombinationTS 把时序预测模型解耦为 Input Transformation / Embedding / Encoder / Decoder / Output Transformation 五个正交模块，在共享的"评估条件空间"上做配对蒙特卡洛采样，用边际性能 $\mu$ 和稳定性 $\sigma$ 取代脆弱的单点 MSE，结论是：一旦数据视图（Embedding）设计得好，参数无关的 Identity Encoder 就能打平甚至超过复杂 Transformer，时序预测领域的"SOTA 增益"很大程度上来自看数据的方式而不是建模能力。

**[DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)**

:   针对"未来协变量已知"的时间序列预测 (TSF-X), DAG 设计了一个双通路网络: 一条沿时间维捕获"历史外生→未来外生"的注意力模式并注入到"历史内生→未来内生"的预测里, 另一条沿通道维捕获"历史外生→历史内生"的模式并注入到"未来外生→未来内生"的预测里, 在 12 个公开/新发布 TSF-X 数据集上 10/10 拿下 MSE 最佳, 显著超过 TimeXer、TFT、TiDE、CrossLinear、PatchTST 等。

**[Doubly Outlier-Robust Online Infinite Hidden Markov Model](time_series/doubly_outlier-robust_online_infinite_hidden_markov_model.md)**

:   本文提出 BR-iHMM：把"鲁棒观测更新（WoLF）"与"批量化状态推断（degenerate sticky HDP prior）"结合起来，给在线无限隐马模型同时在观测空间和状态空间提供有界的 Posterior Influence Function（PIF），在金融订单簿、电力负荷、合成回归三类含异常值的流式数据上把一步预测 RMSE 最多降低 67%。

**[Ellipsoidal Time Series Forecasting](time_series/ellipsoidal_time_series_forecasting.md)**

:   Fern 把长期时间序列预测重新表述为「从固定高斯源到数据相关椭球的最优传输」，借助 Brenier 定理把搜索空间限制在 SPD（对称正半定）类 Jacobian 上，用 Householder 反射的低秩谱分解把代价从 $O(n^3)$ 压到 $O(Rn)$，并在非平稳冲击场景下相对 DLinear / Koopa 等基线取得最多 790× 的稳定性提升。

**[FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](time_series/fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)**

:   本文把 HiPPO 框架背后的概率测度推广到带可调奇异指数 $\alpha$ 的分数阶幂律测度，从而首次同时拿到「全历史保留 + 近时敏感 + 尺度不变」，并将这一理论落地为 LTI 对角化 SSM——FRACTAL 在 Long Range Arena 上以 87.11% 平均分追平 S5，并在 ListOps 上拿到 61.85%。

**[From Observations to States: Latent Time Series Forecasting](time_series/from_observations_to_states_latent_time_series_forecasting.md)**

:   作者发现现有 TSF 模型即使预测精度高，其潜空间也常常是"时间错乱"的（Latent Chaos）；他们提出 LatentTSF——先用 AutoEncoder 把观察压到一个高维潜状态空间，然后让任何主流 backbone 在这个空间内做未来预测（Pred + Align 双损失），最后再解码回观察空间——在 6 个标准 benchmark 上稳定降 MSE/MAE，并恢复了潜表征的时间局部性和频谱结构。

**[HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](time_series/helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)**

:   给每个特征学一个"身份嵌入"作为持久语义锚点，配合时间-特征双螺旋注意力，在 5 个公开多变量时序数据集 21 个缺失场景上全部拿下第一，比次优的 ImputeFormer 在 ETT-h1 等数据集上多 25% 以上的 MAE 降幅。

**[PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)**

:   针对时间序列问答 (TSQA)，PATRA 在表征端把序列显式拆成 full / trend / season 三类模式，并通过三组可学习对齐 token 与文本做深度交叉对齐；在训练端用 SFT + GRPO 两阶段强化学习，把判别式与生成式任务的奖励统一映射到 $[0,2]$ 解决难度失衡，从而在四类 TSQA 任务上全面超越文本 LLM、ChatTS 等多模态时序 LLM。

**[Time-series Forecasting Through the Lens of Dynamics](time_series/time-series_forecasting_through_the_lens_of_dynamics.md)**

:   作者用 Allen 时间区间代数提出 PRO-DYN 命名法，把任意时序预测模型拆成"前处理 PRO → 动力学 DYN → 后处理 PRO"三段，发现两条经验规律：(i) DYN 必须**可学习且完整**才能打过 LTSF-Linear，(ii) DYN 必须放在**整个流程末端**（PRE-DYN 配置）才能吃到长 lookback 的红利；并通过给 Informer/FEDformer/MICN/FiLM 加一个线性 DYN 层让性能稳定提升，给 iTransformer/PatchTST/Crossformer 把 DYN 挪到前端则性能下降，用实验验证两条规律。

**[TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](time_series/tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)**

:   TSRBench 构造了一个覆盖 14 个领域、4 大维度（感知/推理/预测/决策）、15 个任务、4125 道题、同时支持文本/可视化/文本+图/嵌入四种模态输入的时间序列推理基准，系统评测 30+ 主流 LLM、VLM 与 TSLLM，揭示出"scaling 在感知/推理上仍成立但在预测上失效"以及"文本与可视化模态高度互补但当前模型几乎无法融合"等关键结论。

---

## 🕸️ 图学习 { #graph_learning }

**[Anchor-guided Hypergraph Condensation with Dual-level Discrimination](graph_learning/anchor-guided_hypergraph_condensation_with_dual-level_discrimination.md)**

:   AHGCDD 把超图凝聚 (HGC) 从"先训练结构生成器、再匹配训练轨迹"的解耦范式重写为端到端框架：用 Heat-Kernel-PageRank 把结构信息塞进初始化特征、用 anchor-guided 思路按特征距离合成稀疏可学的超边，再用粗+细双级判别损失 (类原型 MMD + 实例级对比) 代替昂贵的 HNN 重训练，在 5 个超图基准上 ≥SOTA 同时最高 144× 加速。

**[Full-Spectrum Graph Neural Network: Expressive and Scalable](graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)**

:   本文把经典谱 GNN 的单变量特征值滤波器 $g(\lambda_i)$ 推广为双变量滤波器 $g(\lambda_i,\lambda_j)$，把信号从节点域抬到节点对域，理论上能逼近 Local 2-GNN（超越 1-WL），并通过低秩张量分解避开 $n^2\times n^2$ 显式计算，在异质图节点分类和子结构计数上拿到强结果。

**[Information-Geometric Adaptive Sampling for Graph Diffusion](graph_learning/information-geometric_adaptive_sampling_for_graph_diffusion.md)**

:   本文把图扩散反向 SDE 的采样轨迹看成 Riemannian 统计流形上的参数曲线，用 Fisher-Rao 度量推出一个无需训练的 Drift Variation Score (DVS) 来度量轨迹的局部"信息曲率"，并据此自适应缩放步长，使每步在信息流形上前进等长，从而在分子（QM9/ZINC250k）和图（Planar/SBM/Ego）生成中以更少步数取得更高 FCD / MMD 保真度。

**[Learning Graph Foundation Models on Riemannian Graph-of-Graphs](graph_learning/learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)**

:   R-GFM 把"不同 hop 数"的子图当作上层 Graph-of-Graphs 的节点，再用一套动态 MoE 路由把每个 GoG 分配到曲率最匹配的 Riemannian 流形（双曲 / 欧氏 / 球面），同时解决了现有图基础模型固定 receptive field 与单一 Euclidean 嵌入两个先天缺陷，下游最高带来 49% 相对提升。

**[On the Expressive Power of GNNs to Solve Linear SDPs](graph_learning/on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)**

:   本文从 Weisfeiler–Leman 层级的角度首次刻画了学习线性 SDP 解所需的最小 GNN 表达力，证明标准的变量-约束二部图消息传递（VC-WL）和高阶 VC-2-WL 都不够，而 2-FWL 等价的 VC-2-FWL 架构足以仿真 PDHG 求解器的更新步骤，并在合成与 SDPLIB 上把高质量预测用作 warm-start，最多带来约 80% 的加速。

**[Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](graph_learning/polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)**

:   PolyNSD 把 Sheaf 神经网络的"一步空间扩散"换成对归一化 sheaf 拉普拉斯的可学习 $K$ 阶多项式谱滤波器，用 Chebyshev 三项递推稳定计算，单层就拥有 $K$-hop 感受野和可控的低/带/高通响应；意外的发现是只用对角 restriction maps 就能超越所有需要稠密大维 stalk 的现有 NSD，参数、显存、运行时间都大幅下降。

**[Quantile-Free Uncertainty Quantification in Graph Neural Networks](graph_learning/quantile-free_uncertainty_quantification_in_graph_neural_networks.md)**

:   QpiGNN 提出"无需分位输入、无需后处理"的 GNN 节点级预测区间框架，用双头 GNN（一头预测均值、一头预测半宽）配合直接优化"覆盖率+区间宽度"的标签级联合损失，在 19 个合成/真实数据集上平均覆盖率提高 22%、区间宽度收窄 50%。

**[Structure-Centric Graph Foundation Model via Geometric Bases](graph_learning/structure-centric_graph_foundation_model_via_geometric_bases.md)**

:   SCGFM 把跨域图基础模型重写为度量测度空间上的"三角测量"问题：学一组 $K$ 个可训练几何基 $\{B_k\}$，每个图用其与各基的 Gromov–Wasserstein 距离 $\delta_k$ softmax 得到一组结构坐标 $\mathbf{w}$，再用基上的 OT plan 把节点特征汇聚到统一维度，从而摆脱"必须对齐节点特征空间"的传统 GFM 桎梏，在 in-domain 与 OOD 少样本图/节点分类上都打过 baseline。

**[Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](graph_learning/unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)**

:   本文把 CNF 公式建模成「子句–文字超图 + 子句关联图」，并在变量级把表示拆成极性不变与极性等变两部分，再用极性翻转一致性正则训练，把 unsat-core 变量预测精度显著拉高一档。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](self_supervised/a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)**

:   本文改进了监督对比学习（在有限标注数据池中构造元组）的样本复杂度上界，通过两个不同的U-统计量估计器，在极值多类场景下实现从依赖最小类概率的界到仅依赖类别数或样本规模的界的突破。

**[Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](self_supervised/beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)**

:   本文提出 SAGE，把"估计未标注数据分布"换成"在表征空间做结构推断"，用 simplex ETF 几何锚 + 高阶图传播 + 分布无关可靠性加权三件套，在极端标签稀缺且未标注分布任意的 UniSSL 设定下取得平均 8.52% 的准确率提升。

**[Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](self_supervised/data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)**

:   作者证明对比学习里的"预定义数据增强 (旋转/裁剪/翻转)"等价于对 Positive-incentive Noise (π-noise) 的点估计, 然后把 π-noise 从"点估计"升级为可学习分布, 训练一个 π-noise 生成器在原图上加可学噪声当增强 (PiNDA), 使 SimCLR / BYOL / SimSiam / MoCo / DINO 在 vision 上稳定涨点, 且天然适配 HAR / Reuters / Epsilon 等无人工增强的非视觉数据。

**[How 'Neural' is a Neural Foundation Model?](self_supervised/how_neural_is_a_neural_foundation_model.md)**

:   作者把一只"小白鼠视觉皮层的 SOTA 基础模型（FNN）"当成生理学实验对象，用解码流形 / 编码流形 / 解码轨迹三件套挨个分析它的 encoder / recurrent / readout，发现 FNN 的拟合精度主要靠 readout 那一堆同质 feature map 撑起来，而真正"像大脑"的只有 recurrent 模块；并用新提出的 tubularity 指标定量地说"早期编码层缺少生物级时间结构"，给未来神经基础模型给出"早期加 recurrence、readout 减少 feature 维度"的明确建议。

**[Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](self_supervised/provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)**

:   作者证明:对比学习里典型的三元组任务,只要嵌入维度 $d$ 小于真维度 $D$ 的某个常数倍,无论用什么优化器,准确率都会"坍缩"到 1 维随机嵌入的 50% baseline,而且在算法层面这件事在 Unique Games 假设下也无法被多项式时间逼近。

**[Statistical Consistency and Generalization of Contrastive Representation Learning](self_supervised/statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)**

:   本文首次为对比表示学习 (CRL) 建立了"上游对比损失最小化等价于下游 AUC 型检索性能最优"的 Fisher / 统计一致性, 并给出依赖于正样本数 $n$ 和负样本数 $m$ 的精细泛化界 $O(1/m+1/\sqrt n)$ (监督) 与 $O(1/\sqrt m+1/\sqrt n)$ (自监督), 从而首次从理论上解释了 CLIP / SimCLR 使用上万负样本能持续涨点的现象。

**[Text-Conditional JEPA for Learning Semantically Rich Visual Representations](self_supervised/text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)**

:   本文提出 TC-JEPA，把 I-JEPA 的 mask 特征预测器额外条件化在图像 caption 上，通过多层稀疏跨注意力让 patch 表示在文本"提示"下变得可预测，从而在不用对比损失的前提下学到语义更丰富、对密集预测尤其友好的视觉表征。

**[The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](self_supervised/the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)**

:   本文用测度论框架把 InfoNCE 损失提升到表示分布上的确定性"种群能量"，证明 unimodal 情形是凸的且收敛到唯一 Gibbs 平衡，而对称多模态情形会出现持续的负对称 KL 耦合，从几何上必然产生 modality gap。

**[Understanding Self-Supervised Learning via Latent Distribution Matching](self_supervised/understanding_self-supervised_learning_via_latent_distribution_matching.md)**

:   作者把对比 / 非对比 / 预测式 SSL 统一为"潜在分布匹配 (LDM)"：最大化样本在假设潜在模型下的对数概率 (alignment) + 最大化潜在熵 (uniformity)，并基于此推出带 Kalman 预测器的非线性可识别预测式 SSL。

---

## 🧊 3D 视觉 { #3d_vision }

**[FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](3d_vision/fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)**

:   本文把人类“先扫一眼再逐块细看”的观察过程抽象为 Focus-Sweep 两阶段范式，用 Mamba 替换 Transformer 做图像-点云交互，并用强化学习动态决定每个尺度上的交互层数，在 RGB-D Scenes V2 和 7-Scenes 上拿到 I2P 配准的 SOTA。

**[LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory](3d_vision/labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo.md)**

:   LabBuilder 把自由文本的实验描述编译成"资产-化学协议"，再用层级化生成 + 几何/化学多目标优化 + 导航修复，产出既视觉合理、又能让机器人真正跑通实验流程的 3D 化学实验室布局。

**[Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](3d_vision/pair2scene_learning_local_object_relations_for_procedural_scene_generation.md)**

:   Pair2Scene 把 3D 室内场景生成从「直接拟合全局联合分布」改成「学习一对一的局部物体关系（支撑 + 功能）然后按场景层级树递归装配」，配合点云几何编码、Mixture-of-Logistics 概率头和碰撞感知拒绝采样，在仅用 3D-Front 数据训练时即可生成对象数从约 4 跃升到约 14 的复杂场景，FID 和用户研究均优于 ATISS、DiffuScene、LayoutVLM 等基线。

**[PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](3d_vision/physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)**

:   把"造可交互 3D 物体"重新理解成"先做物理规划、再做物理生成"的两阶段问题——VLM 充当物理建筑师生成包含层级关系、材料、运动学约束的 "Hierarchical Physical Blueprint"，扩散模型再用 KineVoxel Injection 把铰接参数和几何 voxel 协同去噪，配合 150k 资产、四层标注的 PhysDB 数据集，首次实现单视图到"可在物理引擎里抓握、推动、铰接"的 3D 资产生成。

**[PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](3d_vision/physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)**

:   本文提出 PhysHanDI，把 MANO 手模型和 Spring-Mass 软体模型耦合起来，用稠密手网格驱动可变形物体的物理仿真，并反向利用物体仿真去精化手的重建，在稀疏视角 RGB-D 视频上同时拿到了手和软物的稠密 3D 重建 SOTA。

**[R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](3d_vision/r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)**

:   R³L 把 MLLM 多跳"相对空间关系"推理的两类系统性误差（语义漂移与度量漂移）归因于"反复发生的参考系变换"，并通过不变性空间分解（缩短关系链）、一致性空间想象（imagine-and-revise 循环消除冲突）与支持性空间优化（全局-局部位姿重参数化）三个模块，让 GPT-5 生成的开放词汇 3D 场景在 9 类场景下的碰撞率与越界率都接近 0、语义指标显著反超 LayoutVLM/Holodeck/LayoutGPT。

**[Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction](3d_vision/revisiting_photometric_ambiguity_for_accurate_gaussian-splatting_surface_reconst.md)**

:   AmbiSuR 把 Gaussian Splatting 的两类内生光度歧义（基元边缘外溢、像素混合欠约束）显式建模并用截断 + 射线-颜色一致性消歧，再借高阶球谐系数作"自指示器"找出歧义高风险基元并做无定形局部先验正则，在 DTU 上把平均 Chamfer 距离降到 0.46，超过此前最优 GeoSVR (0.47)。

**[SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](3d_vision/splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)**

:   本文指出多模态点云补全里"硬投影把 3D 点直接打到 2D 网格"会让支持集 Lebesgue 测度为零、梯度被 Dirac delta 截断（称为 Cross-Modal Entropy Collapse），用可微 Gaussian Soft Splatting 把硬投影换成连续密度估计，搭配 EdgeConv 局部 + Transformer 全局的混合编码器和全局-局部解码器，在 PCN/ShapeNet-55/34 拿到 SOTA，并用 KITTI 上的 counter-factual 评估证明 baseline 实际是退化的"单模态模板检索器"。

---

## 📊 LLM 评测 { #llm_evaluation }

**[CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](llm_evaluation/cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)**

:   本文提出 CoCoReviewBench，通过"按类别建子基准 + 用 meta-review 仲裁审稿人/作者冲突来过滤错误意见"两步，把 3,900 篇 ICLR/NeurIPS 论文的人工审稿改造成一个更可信的 AI 审稿评测参考，并发现现有 AI 审稿在 correctness 和 thoroughness 上仍落后于人类、推理模型则更有潜力。

**[Hallucinations Undermine Trust; Metacognition is a Way Forward](llm_evaluation/hallucinations_undermine_trust_metacognition_is_a_way_forward.md)**

:   本文是一篇 position paper，论证"彻底消除 LLM 幻觉"在原理上无法逃避一个"区分度税"（discrimination gap → utility tax）；作者主张把目标从"消灭幻觉"改为**忠实表达不确定性**（faithful uncertainty），并把这种 metacognition 视为 agentic LLM 调用工具时不可或缺的控制层。

**[Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](llm_evaluation/investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)**

:   本文提出「黑盒环境交互」作为评估 LLM 集成式推理（演绎+归纳+溯因）的新范式，构建含 6 类任务 96 个环境的 ORACLE 基准，benchmark 19 个 LLM 后发现：即便最强的 o3 也只能在简单环境拿 70% 准确率、难环境跌到 40%，且所有 LLM 都缺乏「根据反馈自适应优化探索策略」的高层规划能力。

**[iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](llm_evaluation/iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)**

:   iWorld-Bench 是首个专门为"交互式世界模型"设计的统一评测基准，提出一套能把文本 / one-hot / 相机内外参三种动作输入折算到同一指令空间的 Action Generation Framework，并基于 330K 视频精挑 4.9K 任务、9 项指标，对 14 个主流模型做了全维度对比。

**[Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](llm_evaluation/reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)**

:   RACER 把"对每个 query 决定要不要调用 reasoning 模式做 judge"建模为带 KL 不确定集的分布鲁棒约束优化问题，用 primal-dual 算法解出 OOD 下仍满足 cost 预算的最优路由策略，并首次给出 LLM 路由器策略的 linear convergence 理论保证。

**[Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](llm_evaluation/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)**

:   RHB 构造了一套现实工具型多步任务（独立 + 链式两种模式，含数据流水线、日志取证、性能优化、多文件重建四大家族）来量化 LLM agent 的奖励黑客行为，跨 13 个前沿模型发现 RL 后训练显著提高 exploit 率（DeepSeek-V3 0.6% vs R1-Zero 13.9%），且 exploit 率随链长上升、在更难变体上即使近零率模型也会"复发"，而轻量级环境硬化能在不损害任务成功率前提下把 exploit 率减少 87.7%。

**[Stop Automating Peer Review Without Rigorous Evaluation](llm_evaluation/stop_automating_peer_review_without_rigorous_evaluation.md)**

:   这是一篇立场论文：作者通过对 ICLR 2026 真实评审和 60 篇模拟评审的实证测量，发现当前 LLM 审稿存在 hivemind（高度趋同）+ paper laundering（零样本改写就能涨 0.45 分）两大失效，因此论证「在没有严格评估之前，不应让 LLM 直接生成审稿意见」，并呼吁建立一门"审稿自动化的科学"。

**[Token-Efficient Change Detection in LLM APIs](llm_evaluation/token-efficient_change_detection_in_llm_apis.md)**

:   作者证明在低温采样下，"两个 token logit 几乎打平"的特殊输入（Border Inputs）对参数微扰极度敏感——理论上 SNR 在 $T\to 0$ 时发散，于是只观测输出 token（严格黑盒）就能用极少请求做 LLM API 变更检测；提出的 B3IT 在 TinyChange benchmark 上以 1/30 的成本匹敌灰盒 logprob 方法，并在 93 个商用端点上 23 天连续监控发现 8 次真实模型替换。

---

## 📹 视频理解 { #video_understanding }

**[CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](video_understanding/clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)**

:   本文针对视频字幕擦除提出 CLEAR：两阶段训练（Stage I 用 dual encoder + 正交解耦学自监督字幕先验掩码；Stage II 在 Wan2.1 视频扩散模型上加 LoRA + occlusion head 做自适应加权），推理完全不需要任何 mask 或文本检测器，仅训练 0.77% 参数就在中文测试集上把 PSNR 推到 26.80 dB（比最强基线 +6.77 dB），并零样本泛化到 6 种语言。

**[Find, Fix, Reason: Context Repair for Video Reasoning](video_understanding/find_fix_reason_context_repair_for_video_reasoning.md)**

:   本文针对视频推理中"on-policy RL 在能力天花板停滞、off-policy 蒸馏又会熵塌缩"的两难，引入一个冻结的、工具集成的大教师模型在学生 rollout 失败时插入最小化的"证据补丁" (key-frame 区间、错误类型)，让学生在同一问题上重新作答，并把修复后的轨迹通过 chosen-rollout 机制纳入 GRPO 优化。

**[Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection](video_understanding/privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection.md)**

:   作者提出 OPL（Orthogonal Projection Layer）和加强版 G-OPL，用一个 QR 分解出来的可学习正交子空间，在视频异常检测特征空间中显式投影掉"任务无关变量"和"人脸隐私分量"，同时引入 SSC/ARD/PD/FPD 四个隐私感知指标，在保持/提升 VAD AUC 的前提下让线性 SVM 探针对面部预测的准确率显著下降。

**[RELO: Reinforcement Learning to Localize for Visual Object Tracking](video_understanding/relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)**

:   RELO 把视觉单目标跟踪中"哪里是目标"这件事重构成一个空间特征图上的 MDP,把每个空间位置当作 action,用 actor-critic + IoU/AUC 直接奖励替换掉传统的手工中心热图监督,并配合"先 warmup 回归 + 层对齐时序 token 传播"两个稳定化设计,在 LaSOText 上以 57.5% AUC 拿到 SOTA。

**[Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval](video_understanding/revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri.md)**

:   本文针对 Partially Relevant Video Retrieval (PRVR) 中"短查询 vs 长视频"导致的查询歧义与时间稀疏监督问题，提出基于 Dirichlet 分布的层次证据学习框架 Holmes，在视频间用三重原则区分精确/多义/欠定查询并自适应校准标签，在视频内用带 dustbin 的柔性最优传输获得稠密对齐，在 ActivityNet/Charades/TVR 三个数据集上取得 SOTA。

**[STORM: Segment, Track, and Object Re-Localization from a Single Image](video_understanding/storm_segment_track_and_object_re-localization_from_a_single_image.md)**

:   STORM 提出"一张参考图就能跑"的 6D 位姿跟踪框架：用层级化空间融合注意力 HSFA 做参考-查询特征对齐（产出分割掩膜 + SAM3D 网格），再训一个 BCE 二分类的 Tracking Verifier，把其 logit 取负当作能量分数 $E=-g_\theta$，连续 $L=3$ 帧超阈值就触发自动重定位，从而在 LM-O / YCB-V 上把无标注 6D 跟踪精度推到接近 ground-truth 掩膜上限。

**[Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](video_understanding/unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)**

:   OneTrackerV2 把 RGB / RGB+D / RGB+T / RGB+E / RGB+N 五种跟踪任务统一在一个网络里端到端训练，靠 Meta Merger 做模态融合、Dual MoE 把"时空匹配"与"模态融合"两类异质特征显式拆到 T-MoE 与 M-MoE，并用 dissimilarity loss + router clustering 保证它们不塌成同一子空间。

**[VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](video_understanding/videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)**

:   VideoSEAL 发现现有 agentic 长视频 QA 系统存在「答对但没看到证据」的失配问题，并把根因归结为「coupled agent 把规划和回答权混在一起」，提出 planner-inspector 解耦框架：planner 负责长视距证据搜寻、inspector 持有独占回答权并在像素级证据充分时才放行，在 LVBench 上把准确率从 48.2% 拉到 55.1%（↑20.5%）且 LongVideoBench 从 52.2% 升至 62.0%。

---

## 🎵 音频/语音 { #audio_speech }

**[Alethia: A Foundational Encoder for Voice Deepfakes](audio_speech/alethia_a_foundational_encoder_for_voice_deepfakes.md)**

:   Alethia 提出一种"瓶颈式掩码嵌入预测 + Flow-Matching 频谱生成"的双分支预训练范式，训出首个面向语音 deepfake 检测/定位/溯源的基础编码器，在 5 类任务 56 个数据集上显著超过 Wav2vec2/HuBERT/WavLM 等通用 SFM，并对未见过的歌声 deepfake 和真实扰动表现出强零样本鲁棒性。

**[MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)**

:   MECAT 用「多专家模型 + CoT 大模型推理」构造了 20k 条多视角细粒度音频字幕与 10 万条开放式 QA，并提出 DATE 指标（语义相似度 × 跨样本可区分度的调和平均），首次能稳定区分泛泛而谈与细节准确的音频模型输出。

**[MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](audio_speech/medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)**

:   MedMosaic 用合成管道构造了一个覆盖生理声 + 真实/合成临床对话的医学音频 QA 基准（46,701 条 QA、10 种问题类型），系统评测 13 个音频/多模态模型，发现即使 Gemini-2.5-Pro 也只能拿到约 68.1% 加权准确率，揭示当代 LALM 在医学音频推理上的根本短板。

**[MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](audio_speech/moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)**

:   MoshiRAG 在 Moshi 这一全双工语音模型里加入一个特殊的 ⟨ret⟩ 触发 token，让模型边说边异步调用 LLM/搜索后端去取参考文档，利用"开口到关键词出现"的自然 keyword delay 把 2 秒以内的检索延迟完全藏起来，从而在 LlamaQ/WebQ/TriviaQA/HaluEval 上把语音模型的事实性拉到 GPT-4o Audio 量级，同时保留全双工实时性。

**[NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating](audio_speech/naaca_training-free_neuroauditory_attentive_cognitive_architecture_with_oscillat.md)**

:   用一套受皮层振荡启发的二维波动场（OWM）做实时显著性检测，给 Audio Language Model 在长音频上当一个"训练无关的注意力门"，只把真正显著的窗口送进 ALM，从而在 XD-Violence 上把 AP 从 53.5% 拉到 70.6%，同时减少约 40% 的 ALM 调用。

**[Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](audio_speech/polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)**

:   Polyphonia 把 zero-shot 音色转换从单轨扩展到密集多轨混音：用盲源分离得到的 Ideal Ratio Mask（IRM）当外部声学先验，先在 pre-softmax 注意力 logit 里做"源插值 + 声学调制"，让目标声部（如人声）的频谱被新音色（如小提琴）替换的同时把背景伴奏严格保住，相比 SOTA 在 target alignment 上提升 15.5%。

**[Probing Cross-modal Information Hubs in Audio-Visual LLMs](audio_speech/probing_cross-modal_information_hubs_in_audio-visual_llms.md)**

:   作者用因果追踪 + 单模态主导框架揭示了音视频 LLM 中存在一类被称为"跨模态 sink token"的隐藏枢纽,绝大多数跨模态信息都凝聚在这些 token 上,据此提出训练免费的注意力放大策略显著缓解物体幻觉。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](llm_alignment/block-em_preventing_emergent_misalignment_via_latent_blocking.md)**

:   BLOCK-EM 用 SAE 找到一小撮"因果地控制 emergent misalignment"的内部 latent，然后在窄域 SFT 时加一个 one-sided 正则，禁止模型把这些 latent 朝"失对齐方向"放大——在 6 个 fine-tuning 域上把 emergent misalignment 平均砍掉 93%，同时几乎不损伤 in-domain 任务表现。

**[$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](llm_alignment/f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses.md)**

:   本文给在线 RLHF 在**通用 $f$-divergence 正则**下首次建立 $O(\log T)$ regret 和 $O(1/T)$ 次优 gap 上界，提出两套采样策略：(1) 基于 optimism in face of uncertainty 加 bonus 项；(2) 一个新颖的 **"derivative-as-uncertainty"** 视角——把 $f'$ 当作不确定性信号，从而设计 derivative-based 采样而无需在每轮显式估计 confidence bound。

**[Pareto-Guided Optimal Transport for Multi-Reward Alignment](llm_alignment/pareto-guided_optimal_transport_for_multi-reward_alignment.md)**

:   PG-OT 把「多奖励文生图对齐」从「加权全局求和」改成「为每个 prompt 单独构造 Pareto 前沿、用 Sinkhorn 最优传输把被支配样本传到前沿」，并引入 Joint Domination Rate / Joint Collapse Rate 两个新指标暴露平均值掩盖的奖励 hacking，在 Parti-Prompts 上 JDR₂ 47.98% 比强基线提升 11%，人评胜率近 80%。

**[Reward Modeling from Natural Language Human Feedback](llm_alignment/reward_modeling_from_natural_language_human_feedback.md)**

:   本文指出在二元偏好奖励上训练的 generative reward model (GRM) 严重存在"猜对偏好但 critique 错误"的 outcome-process 不一致（20-30%、最高 44%），并提出 RM-NLHF：把模型 critique 与人工 critique 的核心论点相似度作为额外过程奖励，并用 MetaRM 自动预测过程奖励、在线随策略更新，从而在多个 benchmark 上稳定超过 outcome-only GRPO 训练的 SOTA GRM。

**[The Realignment Problem: When Right becomes Wrong in LLMs](llm_alignment/the_realignment_problem_when_right_becomes_wrong_in_llms.md)**

:   本文把"模型部署后政策变了怎么办"形式化为 Realignment 问题,提出 TRACE 框架:用更强的 proxy 模型把已有 preference pair 三分类 (Invert / Punish / Retain) 后用混合 IPO+NPO+KL 目标做手术式再对齐,无需新一轮人工标注就能跟上政策漂移。

**[Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](llm_alignment/toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)**

:   本文提出 SVGT，把价值对齐从"嵌入 backbone 参数/激活"改为"挂一个独立的价值模块"，先在隔离的 value space 里持续判断当前 hidden state 的安全方向，再用一组可学习的 Bridge Token 作为注意力锚点显式引导生成轨迹，在四种 backbone 上把有害分数普遍降低 70% 以上且几乎不损失流畅度。

**[TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](llm_alignment/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)**

:   TUR-DPO 在 DPO 的偏好 logit 上同时叠加一个"语义+拓扑结构"塑形奖励差和一个"按每对样本不确定性"动态降权的实例权重，让模型在保持 RL-free 训练简洁性的同时，显式奖励推理过程的结构合理性并削弱脆弱偏好对的影响，从而在 GSM8K / MATH / BBH / QA 等推理类任务上系统超过 DPO 与 IPO，并在多数任务上追平 PPO。

---

## 📄 multi_agent { #multi_agent }

**[E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](multi_agent/e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)**

:   E-mem 把"预处理压缩成嵌入/图"的传统记忆范式改成"保留原始上下文 + 小模型助手就地推理"的情景重构范式：master agent 只做全局规划，多个 SLM assistant 各自守着一段未压缩的原文，按多路由检索激活后再做局部推理回传证据，在 LoCoMo 上 F1 反超 SOTA 7.75 个点的同时把 token 消耗砍掉 70%。

**[EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](multi_agent/engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)**

:   EngiAgent 把工程问题求解拆成 Analyzer/Modeler/Verifier/Solver/Evaluator 五个专家 Agent，再用一个**全连接协调器**动态路由反馈（而不是走固定流水线），让 GPT-4o 上工程任务的可行解率从 5.66%（zero-shot）/7.55%（MM-Agent）一跃到 64.15%，平均比此前 SOTA 提升约 7 倍。

**[MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)**

:   MASPO 通过多粒度联合评价（局部有效性 + 前瞻潜力 + 全局对齐）+ 错位案例驱动的进化束搜索，在不依赖标注的前提下端到端地为整条多智能体链路联合优化角色提示词，6 个任务上平均提升约 2.9 分。

**[OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)**

:   本文把多智能体系统的优化空间形式化为五个维度（两个功能维度 + 三个结构维度），用"Semantic Initializer 生成 + Contrastive Comparator 对比改进"的双 actor 算法在每个维度上做监督式优化，再迭代联合优化多个维度，在 HumanEval / MMLU / MATH 上稳定打败 DyLAN、ADAS、AFlow 等基线。

**[RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)**

:   RADAR 把多 LLM-Agent 系统的通信拓扑设计建模为一个"冗余感知"的离散图扩散过程，用 effective size 作为指导信号一步步增量生成 query-自适应的协作图，在 6 个基准上同时拿到更高准确率、更低 token 消耗和更强鲁棒性。

**[Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](multi_agent/systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)**

:   本文将社会心理学的 Hidden Profile 范式搬到多智能体 LLM 评测里，构建 65 任务的 HiddenBench，在 15 个前沿 LLM 上系统揭示：单 agent 在 Full Profile 下能 80.7% 答对的同类任务，多 agent 在分布式信息下仅 30.1%，根本失败模式是**不会主动 elicit 别人没说出来的信息**，而轻量结构化沟通协议能跨家族大幅缓解。

---

## 🎬 视频生成 { #video_generation }

**[Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](video_generation/attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)**

:   SVOO 发现视频 DiT 每一层的注意力稀疏度是「层内输入无关、层间显著异质」的内在属性，据此先做离线分层稀疏度标定、再做在线 QK 双向协同聚类划块，免训练地在 Wan/HunyuanVideo 等 7 个模型上把 PSNR 维持 29 dB 的同时实现最高 1.93× 加速。

**[Exploring Data-Free LoRA Transferability for Video Diffusion Models](video_generation/exploring_data-free_lora_transferability_for_video_diffusion_models.md)**

:   本文首次对视频扩散模型（VDM）的 full fine-tune (FFT) 和 LoRA 做权重空间分析，发现两者都"保留奇异谱、只旋转奇异子空间"，但在 head clusters 上路由方向冲突；据此提出 CASA——一个 data-free 的"按聚类做谱仲裁"的 LoRA 迁移方法，把基座 Wan2.1 上训的 LoRA 直接迁到 FastWan 等蒸馏后变体，无需任何用户数据/重训。

**[Lightning Unified Video Editing via In-Context Sparse Attention](video_generation/lightning_unified_video_editing_via_in-context_sparse_attention.md)**

:   针对 In-Context Learning 范式下视频编辑的二次注意力瓶颈，作者基于"context token 显著性低于 source token"以及"Query 锐度正比于 Taylor 近似误差"两条洞察设计了 In-context Sparse Attention（ISA），并训练出 LIVEditor，在多个 benchmark 上既加速 ~60% 又超越 SOTA 全注意力模型。

**[MiVE: Multiscale Vision-language features for reference-guided video Editing](video_generation/mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)**

:   MiVE 把 Qwen3-VL 的**首层 + 末层**隐状态同时抽出来作为多尺度条件 token, 与 VAE 视觉 latent 拼成一个长序列, 在统一的自注意力 DiT 里做参考图引导的视频编辑, 在 60 段 720P benchmark 上人类偏好和 6 个 VLM 自动评分都拿到第一, 超过开源 Wan-Animate 和商用 Kling O1.

**[Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](video_generation/quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)**

:   QVG 是面向自回归视频扩散的训练免微调 KV-Cache 量化框架——通过语义感知聚类做 token 平滑、并以渐进残差多阶段压缩残差，在 LongCat-Video/HY-WorldPlay/Self-Forcing 上把 KV 显存压低到原来的 1/7，端到端延迟开销 <4%，2 bit 下质量大幅领先 KIVI/QuaRot 等 LLM 量化基线。

**[VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](video_generation/vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)**

:   VAnim 把开放域 text-to-SVG 动画建模为「持久 DOM 树上的稀疏状态更新」+「Identification-First 运动规划」+「GRPO 渲染感知强化学习」，序列长度压缩 $9.86\times$ 的同时保持拓扑一致，并显著超越 GPT-5.2、Gemini 3 Pro 与 LiveSketch。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](aigc_detection/black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)**

:   SurpMark 把"AI 文本检测"重构成似然无关假设检验：用代理 LM 算 token surprisal 后 k-means 离散成 k 个状态，估计一阶 Markov 转移矩阵，再用广义 Jensen-Shannon 散度（GJS）和预先建好的"人写 / 机写"参考转移矩阵比较，单次前向就给出黑盒、无需重训、无需 per-instance 重采样的判别分数。

**[DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](aigc_detection/dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)**

:   论文针对"CLIP 微调到 AI 生成图像检测时灾难性遗忘破坏可迁移先验"的问题，提出 DGS-Net：把分类损失的梯度按坐标拆成有害正分量 $g^+$ 与有益负分量 $g^-$，让训练网络的图像梯度先正交投影到冻结 CLIP **文本梯度有害方向**的补空间（Orthogonal Suppression，剔除任务无关语义），再额外对齐到冻结 CLIP **图像梯度有益方向**（Prior Alignment，保住预训练先验），从而在 50 个生成模型上的平均检测精度比 SOTA 高 6.6%。

**[Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](aigc_detection/feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)**

:   本文在「单阈值固定协议」下系统暴露 AI 文本检测器在跨数据集/跨生成器 shift 下的脆弱性，并提出把可学注意力加权的手工语言特征与 transformer [CLS] 表征融合，配合 DeBERTa-v3 backbone，在 M4 多域多生成器基准上达到 85.9% balanced accuracy，比强 zero-shot 基线（Fast-DetectGPT、RADAR、Log-Rank）高最多 +7.22。

**[PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](aigc_detection/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)**

:   作者用一个 115k 带推理标注的 DF-R5 数据集 + 把 CLIP ViT 换成 ConvNeXT 的 DX-LLaVA 架构，并提出 PRPO —— 段落级别 GRPO 变体，每段以 CLIP-文本-图像相似度（VCR）+ 推理-结论多数票一致性（PCR）为 reward，把跨域 deepfake 检测 F1 从 SOTA 75.26% 推到 89.91%，推理质量从 4.2/5 提到 4.55/5。

---

## 💬 LLM / NLP { #llm_nlp }

**[A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)**

:   本文从微分几何视角刻画 GPT 风格 LLM 在高熵分布上采样所引入的信息丧失，构造 $\mathfrak{so}(n)$ 值 1-形式与平行输运算子，并在国际象棋探针实验中证明这种几何旋转与模型学到的世界向量高度同向。

**[Escaping Mode Collapse in LLM Generation via Geometric Regulation](llm_nlp/escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)**

:   本文从动力系统视角把 LLM 长文本生成中的「模式崩溃」（重复、循环、单调）重新解释为隐藏状态轨迹在表示空间里的「几何坍缩」，并提出 RMR — 在 Transformer value cache 上做轻量低秩阻尼来抑制最具持续性的自我强化方向，从而在极低熵的解码区间（0.8 nats/step）依然保持稳定高质量生成。

**[Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](llm_nlp/geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)**

:   Top-W 把 next-token 截断写成"考虑 token embedding 几何的 Wasserstein-熵-质量"三项最小化问题，理论证明最优解要么是单 token、要么是按 $f(i)+\lambda\log p_i$ 排序的前缀，工程实现只是 $O(n\log n)$ 的扫描；在 GSM8K、GPQA、AlpacaEval、MT-Bench 上 15 个 (T, model) 组合多数胜出，高温下 GSM8K 比 Top-H 最多再提 33.7%。

**[Rethinking LLM Ensembling from the Perspective of Mixture Models](llm_nlp/rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)**

:   本文证明对 $n$ 个 LLM 做 token 级集成时无需每步都跑所有模型——按权重随机抽一个模型采下一个 token，输出分布与"先平均后采样"严格等价，从而把 $n$ 倍前向变回 1 倍前向，并配合"懒同步 KV 缓存"实现 1.78×–2.68× 的实际加速。

---

## ✂️ 语义分割 { #segmentation }

**[LightAVSeg: Lightweight Audio-Visual Segmentation](segmentation/lightavseg_lightweight_audio-visual_segmentation.md)**

:   LightAVSeg 通过解耦 "语义筛选 (what)" 和 "空间定位 (where)"，用全局通道调制替换 $\mathcal{O}(N^2)$ 的跨模态注意力，让 AVS 模型在 20.5M 参数下达到 50.4 mIoU (MS3)，并在 Snapdragon 8 Elite 上做到 163.4 ms 的端侧延迟，比 AVSegFormer-R50 快约 $8\times$。

**[Segment Anything with Robust Uncertainty-Accuracy Correlation](segmentation/segment_anything_with_robust_uncertainty-accuracy_correlation.md)**

:   针对 SAM 系列只输出 mask-level 单一置信度、在域漂移下出现"Mask-level Confidence Confusion"的问题，本文给 SAM2 接上 Weibull 双粒度贝叶斯 mask decoder 做像素级 epistemic 估计，并配以受人类视觉启发的 style + deformation 协同对抗扰动 + 校准损失，让 uncertainty 在 23 个 zero-shot 目标域始终与误差对齐，平均 J&F 达 79.87 同时不确定性图变得显著可信。

**[SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](segmentation/semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)**

:   SEMIR 把体素栅格当作母图 $G$，通过参数化的边收缩 / 节点删除 / 边删除把它压成一张「边界对齐」的图 minor $H$（节点数从 $\sim10^7$ 降到 $\sim10^3$），用 5–20 张少样本黑盒优化 $\Theta$ 最大化边界 Dice，再在 minor 上用 GNN 做超节点分类，最后通过 minor 与体素之间的双射 exact lifting 回到原栅格——在 BraTS / KiTS / LiTS 三大肿瘤分割任务的少数类 Dice 上稳定超过 nnU-Net，且仅需 16GB T4 GPU。

**[UGround: Towards Unified Visual Grounding with Unrolled Transformers](segmentation/uground_towards_unified_visual_grounding_with_unrolled_transformers.md)**

:   UGround 把 LMM-based 视觉定位从"用最后一层 $\langle\text{SEG}\rangle$ token 当 prompt"的范式翻转为"用动态选中的中间层相似度图当 prompt"，通过强化学习策略 SSC 让 $\langle\text{SEG}\rangle$ 滑过所有 transformer 层、把相似度图同时当作 SAM 的软 logit mask 和反向监督信号，首次在单一框架内统一了 RES / RS / FP-RES / gRES / Multi-RS 五种视觉定位任务，并在 ReasonSeg test 上 cIoU +9.0%、gRefCOCO val N-acc +12.1%。

---

## 🔗 因果推理 { #causal_inference }

**[Causal Fine-Tuning under Latent Confounded Shift](causal_inference/causal_fine-tuning_under_latent_confounded_shift.md)**

:   本文提出 Causal Fine-Tuning (CFT)：在标准 BERT 微调里嵌入一个 SCM 启发的"高级稳定特征 $C$ + 低级混杂敏感特征 $\Phi$"分解，并用 front-door 风格的 do-calculus 调整公式做预测，在文本伪相关注入攻击下显著优于 SFT/SWA/WISE 等单域泛化基线。

**[Controllable Generative Sandbox for Causal Inference](causal_inference/controllable_generative_sandbox_for_causal_inference.md)**

:   本文提出 CausalMix：一个变分生成框架，把数据类型特定的 multi-head decoder + Bayesian Gaussian 混合潜在 prior 与三类可独立调控的因果"旋钮"（overlap $\alpha(X)$、CATE 函数 $\tau(X)$、未观测混杂 $\kappa(X,T)$）联合优化，从而在保持真实数据分布 fidelity 的前提下让用户自由设计 counterfactual benchmark，在 mCRPC（前列腺癌）真实病例上验证 CausalMix 既能高保真复现 mixed-type 表格，又能稳定地按需注入 overlap / confounding / 异质效应，用作 CATE 估计器的可控 stress test。

**[The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](causal_inference/the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)**

:   本文把搜索广告的真实价值建模为"赢拍 vs 输拍"的 treatment effect，在重复二价拍卖（SPA）binary 反馈下设计了一个利用支付规则的在线因果学习算法，得到 $\widetilde\Theta(\sqrt{dT})$ 的极小极大最优 regret，比同设定下的一价拍卖严格更易学。

---

## 💻 代码智能 { #code_intelligence }

**[BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)**

:   BoostAPR 给"用 RL 训 program-repair 模型"造了一套三阶段流水线——execution-verified SFT → 训序列级 + 行级双重 reward → PPO 时用行级模型把序列奖励重新分配到关键 edit lines；在 Qwen2.5-Coder-32B 上把 SWE-bench Verified 从 17.8% 推到 40.7% (+22.9pp)，跨语言迁移到 Defects4J 取 24.8%。

**[HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](code_intelligence/he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)**

:   在 SWE-bench 上传统 PPL 既受"长上下文税"干扰又无法预测 SFT 后的智能体能力，本文提出"熵压缩假说"和 HE-SNR 指标，只在 Top-10 熵大于 $(\ln 3 + \ln 4)/2$ 的"高熵决策点"上算信号噪声比，与下游 SWE-bench 得分的 Pearson 相关达 0.96，Kendall 一致性 0.98。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](image_restoration/hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)**

:   H-VAR 把"残差量化做多尺度生成"的 VAR 范式重新切片成层次化的图像 tokenization (HIT)，让一个 310M 的小模型只跑一次前向就能输出 128 / 256 / 512 三个有意义的中间分辨率，再配一个不需要外部奖励模型的 DPO 正则项推动输出偏向 HR，在标准 ISR 数据上对打 1B 参数的 VARSR。

**[Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration/image_restoration_via_diffusion_models_with_dynamic_resolution.md)**

:   SubDAPS / SubDAPS++ 把 DPS、DAPS 这类 pixel-space 扩散复原方法搬进"动态分辨率扩散模型"框架——早期在 $64^2 / 128^2$ 子空间采样、后期才回到 $256^2$ 全分辨率，并用共轭梯度替掉 Langevin、用阈值切换 stochastic / deterministic 采样、再附一个无需额外网络评估的 corrector 步，在 4 类线性 + 2 类非线性复原任务上多数指标超越 pixel 与 latent 扩散方法且推理更快。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)**

:   Ψ-RAG 用"合并—坍缩"式的层次聚类替换 RAPTOR 的 k-means 来构建跨文档抽象树，并配上一个具备多轮重写能力的检索回答 Agent 与稀疏 BM25 混合索引，让 Tree-RAG 第一次能在语料级、跨文档多跳问答上追平甚至超越 Graph-RAG，平均 F1 比 RAPTOR 高 25.9%、比 HippoRAG 2 高 7.4%。

**[Very Efficient Listwise Multimodal Reranking for Long Documents](information_retrieval/very_efficient_listwise_multimodal_reranking_for_long_documents.md)**

:   ZipRerank 同时砍掉 VLM 列表式重排的两大瓶颈——「视觉 token 序列过长」和「自回归解码逐 token 输出排名」——用 query-aware token 剪枝 + 单 logit 排序在 MMDocIR 上把 LLM 推理延迟降一个数量级，同时匹配或超越当前 SOTA 的 MM-R5。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](knowledge_editing/crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)**

:   把 LLM 编辑写成"最小化编辑损失 s.t. 能力损失不变"的约束优化, 用 Bregman 散度等价转化为 Gauss-Newton Hessian 的低曲率子空间投影, 再借 K-FAC + 一个无需显式构造投影矩阵的 Kronecker 特征基技巧, 让 3000 条编辑在 A40 上 6 分钟跑完, 同时把 LLaMA-3-8B 的 MMLU/IFEval/ARC-C/TruthfulQA/GSM8K 平均掉点压到 < 1%, 显著优于 AlphaEdit / MEMIT / 微调。

**[KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)**

:   KORE 通过两阶段"知识导向控制"为 LMM 注入新知识 — 一边把单条事实自动扩成结构化的多轮对话+指令任务（提升泛化），一边用先前知识的协方差矩阵零空间初始化 LoRA 适配器（最小化对旧能力的干扰），在 LLaVA-v1.5 / Qwen2.5-VL 上同时实现强适配和强保留。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](multilingual_mt/ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)**

:   ML-Embed 把 Matryoshka 思想从一维 (representation 维度) 扩展到**三维** —— 在 embedding 参数 (MEL)、模型深度 (MLL)、表征维度 (MRL) 上**全栈嵌套训练**, 同时构建 282 种自然语言 + 40 种编程语言、5000 万样本的多语训练集, 推出 140M-8B 一族开源模型, 在 17 个 MTEB benchmark 上 9 个拿第一, 波兰语 +22.89, 越南语 +6.88.

**[Optimizing Language Models for Crosslingual Knowledge Consistency](multilingual_mt/optimizing_language_models_for_crosslingual_knowledge_consistency.md)**

:   本文针对多语言 LLM 在不同语言间回答同一问题却给出冲突答案的问题，设计了一个**用"另一种语言下回答的对数似然"作为 reward 的 RL 目标**，证明其最优策略呈 product-of-experts 形式并在 $\gamma_1\gamma_2=\beta^2$ 时保证跨语言偏好一致；据此推导出无需 reward model、无需 online 采样的 **DCO（Direct Consistency Optimization）** 算法，在 9 个 LLM、3 个多语言 QA 基准、26 种语言上同时提升跨语言一致性（RankC）与回答准确率。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving](autonomous_driving/deepsight_long-horizon_world_modeling_via_latent_states_prediction_for_end-to-en.md)**

:   DeepSight 把"未来世界预测"从显式像素重建（codebook 单帧）换成在 BEV 空间对 DINOv3 语义特征做**多帧并行隐式预测**，再叠加一个按需触发的 Adaptive Chain-of-Thought，让 Qwen2.5-VL-3B 在 Bench2Drive 闭环上 Driving Score 86.23 (+7.39)、Success Rate 71.36% (+13.63)，且只多 ~4% 推理延迟。

---

## 🧑 人体理解 { #human_understanding }

**[MotionGRPO: Overcoming Low Intra-Group Diversity in GRPO-Based Egocentric Motion Recovery](human_understanding/motiongrpo_overcoming_low_intra-group_diversity_in_grpo-based_egocentric_motion_.md)**

:   MotionGRPO 把 head-mounted 设备的第一人称全身动作恢复转化为扩散采样上的 MDP，用 GRPO 配合"轨迹条件感知模型 + 4 个 joint-level 子奖励"的混合奖励做后训练；同时识别出"输入条件太强、组内样本几乎一样导致 advantage 方差消失"这一致命瓶颈，并用 Perlin 噪声注入条件来恢复组内多样性，在 AMASS/RICH 上把 MPJPE 从 EgoAllo 的 124.985 mm 降到 114.207 mm。

---

## 🎯 目标检测 { #object_detection }

**[Smoothing Slot Attention Iterations and Recurrences](object_detection/smoothing_slot_attention_iterations_and_recurrences.md)**

:   针对 Slot Attention 在图像与视频对象中心学习中"冷启动查询信息不足"和"首帧/非首帧聚合变换被强行统一"两个长期被忽视的痛点，作者提出 SmoothSA：用一个自蒸馏的小预热模块给查询注入样本信息，同时让首帧跑完整三次迭代、非首帧只跑一次，从而在图像和视频两个 OCL 基准上同时刷新 SOTA。

---

## ⚛️ 物理学 { #physics }

**[Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](physics/neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)**

:   用一个生成-评估神经网络（GEN）一次性地把 QAOA² 的"图分割 + 量子电路参数初始化"两件事联合可微化：评估器学一个高保真的 quantum performance surrogate，生成器在它的梯度引导下吐出离散分区 + 参数初值，配合直通估计器 + 正交补头让端到端可训练；在 183 个 QUBO/Ising/MaxCut 实例（21-1000 变量）上超越启发式 baseline，101 个实例排第一。

---

## 🎁 推荐系统 { #recommender }

**[Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](recommender/can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi.md)**

:   RSIR 让序列推荐模型用自身预测能力生成新的合成用户交互序列、再训练一个新模型，并用基于排名的"保真度检查"过滤掉偏离用户偏好流形的样本，防止 self-consuming model collapse；在 4 个数据集 × 3 个主流 backbone 上稳定提升 NDCG/Recall 4–11%，并理论上证明该过程等价于沿用户偏好流形切空间的隐式正则化。

---

## 📂 其他 { #others }

**[Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](others/active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)**

:   本文形式化了表格增强中的"保真度-效用间隙"问题（生成器优化分布匹配，而增强价值源于低密度区域），提出 TAP 算法通过扩散填补做流形约束提议、策略引导的效用对齐选择、硬约束门控加保守窗口提交，在 7 个真实表格数据集上相比基线最多提升分类精度 15.6%、回归 RMSE 降低 32%。

**[Adaptive Multi-Round Allocation with Stochastic Arrivals](others/adaptive_multi-round_allocation_with_stochastic_arrivals.md)**

:   本文形式化网络招募为预算约束的顺序控制问题，证明单轮最优分配是贪心的；通过人口水平代理值函数将多轮规划降维到 $O(b^5\log b)$ 复杂度，并给出在模型误差下分解为前沿/人口/逼近三类误差的鲁棒性保证。

**[AI Cap-and-Trade: Efficiency Incentives for Accessibility and Sustainability](others/ai_cap-and-trade_efficiency_incentives_for_accessibility_and_sustainability.md)**

:   作者借鉴碳排放 cap-and-trade，提出针对 AI 推理 FLOP 的配额-交易市场（AI Allowance），用 KKT 条件证明其能在合理参数下严格减少各公司 FLOP 使用，从而同时缓解大模型时代的能耗与小公司被挤出市场两大问题。

**[Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](others/cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)**

:   TabCascade 把表格行拆成"低分辨率（类别 + 数值的离散化版本）"与"高分辨率（连续数值）"两段级联：先用 CDTD 学低分辨率联合分布，再用 flow matching 在低分辨率引导下生成数值细节，并通过数据相关耦合 + 可学非线性时间表收紧 transport cost；天然支持缺失值、零膨胀等"混合型特征"的生成，在 12 个数据集上 detection score 比 SOTA 提升 51.9%。

**[Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](others/complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)**

:   本文提出 Complexity-as-Advantage (CAA)：把"复杂度"重新定义为一族**资源受限观察者**在同一过程上的**后悔（regret）分散程度**，并证明它在 log-loss + Markov 框架下等价于条件互信息原子之和（恰好恢复 excess entropy），在编码视角下等价于过剩描述长度的方差（MDL），从而把 Kolmogorov 复杂度、Bennett 逻辑深度、excess entropy 统一成一个**可计算、可经验估计**的标量谱。

**[Decision Tree Learning on Product Spaces](others/decision_tree_learning_on_product_spaces.md)**

:   本文把 Blanc et al. (ITCS'20) 对"top-down greedy 决策树启发式"的理论保证从均匀分布推广到**任意乘积分布**，给出 $\exp(\Delta_\mathrm{opt} D_\mathrm{opt}\log(e/\epsilon))$ 大小上界（满二叉树情形严格优于 ITCS'20），且**完全免参数**——不需要预知最优树大小或深度即可运行。

**[Estimating Correlation Clustering Cost in Node-Arrival Stream](others/estimating_correlation_clustering_cost_in_node-arrival_stream.md)**

:   本文研究「节点到达」数据流模型下相关聚类（correlation clustering）代价的近似估计问题：作者提出 C4Approx 算法，用 $O(n^{(3+\alpha)/4}\log n)$ 词的**亚线性**空间和常数遍数得到 $(O(1), n^{1-\alpha})$-近似，并配套两个匹配下界证明多遍与加性误差都不可避免；在真实数据上仅存 2% 节点即达 Pivot 同等效果。

**[From Generalist to Specialist Representation](others/from_generalist_to_specialist_representation.md)**

:   本文给出第一个完全非参数（无 intervention、无 functional 约束）的两层 hierarchical 可识别性证明：时间-任务结构由 collider 视角下的 CI test 可识别，任务相关 latent 由 sparsity 正则可从 generalist 表示中分离出来。

**[From Human-Level AI Tales to AI Leveling Human Scales](others/from_human-level_ai_tales_to_ai_leveling_human_scales.md)**

:   本文用 LLM 当人口外推器，把 18 个能力维度按"全世界人口正确率"对数刻度 $L=-\log_B p_W$ 校准，并发现 Volume / Attention 维度真实 base $B \gg 10$、Comprehension 维度 $B \approx 1$，揭示现行 AI 与人类的比较其实严重失调。

**[GEM-FI: Gated Evidential Mixtures with Fisher Modulation](others/gem-fi_gated_evidential_mixtures_with_fisher_modulation.md)**

:   本文针对证据深度学习 (EDL) 在分布外样本上过自信、且单头难以表达多模态认知不确定性的问题，提出三件套 GEM-Core/MIX/FI：用学到的特征能量门控证据、用混合证据头单次推理近似 ensemble、用 Fisher 信息正则稳定混合分配，在 CIFAR-10→SVHN/CIFAR-100 等 OOD 检测上比 DAEDL 强且保持 single-pass。

**[DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](others/generative_adaptation_of_dynamics_to_environmental_shifts_via_weight-space_diffu.md)**

:   DynaDiff 把"为新环境训练一个预测器"的元学习问题改写成"用扩散模型直接生成完整网络权重"的条件采样问题，借助权重图 + 函数一致性损失 + 动力学感知 prompter，在 4 个 PDE 系统上平均 RMSE 比强基线再降 10.78%。

**[HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](others/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)**

:   借鉴 Helmholtz 自由能的物理直觉，把每个领域的提示参数训练出一条"压缩到边界 $\Theta$、对齐到中线 $\Delta$"的能量曲线，推理时再用能量因子 + 距离因子联合加权各领域提示，在 CDDB / DomainNet / CORe50 三个 DIL 基准的未知领域上分别提升 1.76 / 3.12 / 2.57 个百分点。

**[Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](others/local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)**

:   作者首次研究"从单条 Gaussian Glauber 动力学轨迹"中学习高斯图模型结构的问题，提出两种互补算法：LET-GL（基于 i,i,j,i 窗口的局部边检测、完美并行）和 BTR-GL（在 Dobrushin 条件下用 burn-in/thinning 把轨迹"解相关"成近似 i.i.d. 样本再喂给现成 i.i.d. 学习器），并给出有限样本恢复保证 + 信息论下界 + 一个独立有用的随机扫描高斯 Gibbs sampler 的 TV mixing 上界。

**[Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](others/local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)**

:   本文提出 LHSD，把 score 模型的对数密度 Hessian 做一个 Hill 型谱滤波只保留近零特征值来数切空间维数，再用 Stochastic Lanczos Quadrature 把 $\mathcal{O}(D^3)$ 的代价压到 $\mathcal{O}(D)$，从而在 3072 维图像空间稳定估计局部内禀维度，并用于诊断扩散模型的训练样本记忆化。

**[Matroid Algorithms Under Size-Sensitive Independence Oracles](others/matroid_algorithms_under_size-sensitive_independence_oracles.md)**

:   作者提出「查询代价随查询集合大小线性增长」的尺寸敏感拟阵 oracle 模型，证明在该模型下找基、估计秩、估计划分数的最优查询代价都是 $\tilde{\Theta}(n^2)$，并对有界周长 $c$ 的拟阵给出 $\mathcal{O}(n^{2-1/c}\log n)$ 的最大权基算法突破二次下界。

**[Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](others/mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)**

:   针对 TabPFN 这类把训练集当作 in-context 直接喂进 attention 的"表格基础模型"做后验校正——发现它会严重过拟合训练集 majority class, 提出 DistPFN：用 $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$ 这一行后验重加权, 在 253 个 OpenML 数据集上把 TabPFN-v2 在 $\beta=5$ 强标签漂移下的准确率从 72.7% 拉到 76.9%, 不用重训、不用估测试先验、不动架构。

**[Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](others/mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)**

:   MPFM 把 OSAD 里传统的"单峰高斯原型"换成可学习的**高斯混合原型空间**, 用流匹配直接回归一个 GMM 形式的速度场, 再加一个互信息最大化正则防止原型崩塌, 在 9 个工业 / 医学 AD 数据集上以 10/1 个异常样本的设定打过 DRA / AHL / DPDL 等所有 SOTA.

**[Networked Information Aggregation for Binary Classification](others/networked_information_aggregation_for_binary_classification.md)**

:   把 Kearns-Roth-Ryu 2026 的"在 DAG 上让线性回归 agent 顺序传 prediction 列即可逼近全局最优"结论推广到二分类：每个 agent 只看到部分特征列、顺序地把自己的 logit 转发给下游，能在 $M$-coverage 条件下用 $O(M/\sqrt{D})$ 超额 BCE loss 达到全局逻辑回归最优；同时构造硬实例证明 $\Omega(k/D)$ 下界，把网络深度刻画成信息聚合的根本瓶颈。

**[New Bounds for Kernel Sums via Fast Spherical Embeddings](others/new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)**

:   通过把 Bartal-Recht-Schulman 2011 的"随机 Nash 装置"球面嵌入定理用迭代 Fastfood 变换做成快速版（time $\widetilde{O}(d + \Lambda^2 + \varepsilon^{-2})$），再把它作为 Gaussian KDE 的预处理把直径压到 $\widetilde{O}(1/\sqrt{\varepsilon})$，得到新的 Gaussian KDE 查询时间界 $\widetilde{O}(d + \varepsilon \Delta_\sigma^2 + 1/\varepsilon^3)$，在小 $\varepsilon$ 中等直径的体制下优于 RFF / FJLT+RFF / Fastfood。

**[NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](others/nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)**

:   用一个 asinh 链接的 GLM surrogate 把多智能体 MCTS 的 joint-action 空间 $d^n$ 压成 low-dim 非线性 bandit，再用"一阶差分量 + 二阶 mixed difference"作为 NonUCT 提议规则，只在每个节点维护小候选集 $\mathcal{C}(s)$，证明 $\widetilde{O}(T^{3/4})$ 的局部 regret（与 $d^n$ 无关），在 MatGame/SMAC/SMACv2 上 sample efficiency 和最终性能都好过 MAZero 等强 baseline。

**[Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](others/polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)**

:   Polaris 把概念表示拆成"方向（语义）+ 轨道势能（层级）"两个解耦信号，全部学到单位超球面上：用切空间投影 + 指数映射保证流形封闭，用各向异性球面 SVGD 防止赤道聚集，用 vMF KL 散度实现不对称的"父类应比子类更高熵"约束，在 taxonomy expansion 任务上把 top-K 召回提升最多 19 点、mean rank 降低 60%。

**[Possibilistic Predictive Uncertainty for Deep Learning](others/possibilistic_predictive_uncertainty_for_deep_learning.md)**

:   本文用 possibility theory 替代 Bayes 概率框架，提出 DAPPr——把参数空间的 possibilistic 后验通过 supremum 投影到预测空间，再用可学习的 Dirichlet possibility function 拟合，最终得到一个仅 10 行代码、可直接替换交叉熵、且在 OOD 检测上超越 EDL 家族的认知不确定性建模方法。

**[Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](others/provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)**

:   本文用「实代数几何 + 一阶谓词逻辑量词消去」给多维超参数调参第一次给出可证明的 generalization bound，把过去只能处理一维标量超参的 Balcan 2025 框架推广到任意 $p$ 维、双层验证损失、近似内层优化等多种实际场景，并配出第一条匹配上界的下界。

**[Realizable Bayes-Consistency for General Metric Losses](others/realizable_bayes-consistency_for_general_metric_losses.md)**

:   本文对"在一般（可能无界）度量损失下，假设类 $\mathcal{H}$ 何时存在分布无关的强通用 Bayes 一致学习算法"这一开放问题在 realizable 情形下给出锐刻画——充分必要条件是 $\mathcal{H}$ 不包含一种新的"无界 gap Littlestone 树"组合障碍。

**[Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective](others/reliable_ai_needs_to_externalize_implicit_knowledge_a_human-ai_collaboration_per.md)**

:   本文是一篇 ICML 立场论文,主张当前所有 AI 可靠性方法 (RAG / 自一致性 / RLHF / Agent Memory) 都只能验证显式知识,而 AI 真正强大的能力来自训练数据里 80-95% 未被人类正式记录的"隐式知识",作者提出 Knowledge Objects (KOs) 作为基础设施——把 AI 隐式推理外化成人类可检查、可验证、可背书的结构化产物,从而让一次人类验证的成本在群体中长期复利。

**[Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](others/scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)**

:   作者提出 CaRE：在 ViT 每个 block 里塞一个 **两级路由 MoE (BR-MoE)** ——先靠"类感知器"按熵选 Top-M 个相关任务路由，再由这些路由各自激活 Top-K 任务专家并叠加一个共享 EMA 专家，于是哪怕任务序列拉到 300+ 也能既保留旧知识又持续吸纳新类，并把"长序列 CIL"这块此前没人正经做的空白填上（顺便发布了 1000 类的 OmniBenchmark-1K 基准）。

**[Singular Bayesian Neural Networks](others/singular_bayesian_neural_networks.md)**

:   本文把权重矩阵直接参数化为 $W=AB^\top$ 而不是对 $W$ 本身做平均场分布，从而诱导出一个**关于 Lebesgue 测度奇异的低秩后验**，参数量从 $O(mn)$ 降到 $O(r(m+n))$，PAC-Bayes 复杂度从 $\sqrt{mn}$ 收到 $\sqrt{r(m+n)}$，并在 MLP/LSTM/Transformer 三类架构上实现 OOD 检测胜过 5-成员 Deep Ensemble 同时参数少 $33\times$。

</div>