---
title: >-
  ICLR2026 图像恢复方向15篇论文解读
description: >-
  15篇ICLR2026的图像恢复方向论文解读，涵盖扩散模型、语音、LLM、图像恢复、对抗鲁棒、推荐系统等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🔬 ICLR2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (5)](../../ACL2026/image_restoration/index.md) · [📷 CVPR2026 (39)](../../CVPR2026/image_restoration/index.md) · [🤖 AAAI2026 (13)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (29)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

🔥 **高频主题：** 扩散模型 ×7 · 语音 ×2

**[Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)**

:   首次将激活引导（activation steering）应用于 Masked Diffusion 语言模型（MDLM），发现 MDLM 的拒绝行为也受单一低维方向控制，通过在去噪过程中全局投影可完全绕过安全对齐，且与自回归模型不同，有效方向可从指令前的 token 中提取——反映了扩散模型的非因果并行处理特性。

**[AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)**

:   通过统计分析扩散语言模型（dLLM）去噪过程中 token 置信度的动态变化，发现"波动带"（Volatility Band）区域编码了文本的局部语义结构，进而提出 AdaBlock-dLLM——一个无训练、即插即用的自适应块大小调度器，让半自回归解码的块边界与语义步骤自然对齐，在相同吞吐量下最高提升 5.3% 准确率。

**[Are Deep Speech Denoising Models Robust to Adversarial Noise?](are_deep_speech_denoising_models_robust_to_adversarial_noise.md)**

:   首次系统性评估 4 款 SOTA 深度语音去噪（DNS）模型在对抗噪声下的鲁棒性：通过心理声学约束的 PGD 攻击生成人耳不可感知的对抗噪声，可令 Demucs、Full-SubNet+、FRCRN 和 MP-SENet 输出完全不可理解的 gibberish，实验覆盖多种声学条件和人类评估，同时揭示了目标攻击、通用扰动和跨模型迁移的局限性。

**[Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)**

:   LSP 调度器通过在每个去噪步骤中原子性地提交最长连续稳定前缀（而非分散接受离散 token），将 DLM 推理加速 3.4 倍，同时保持或略微提升输出质量。

**[Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)**

:   定义了"Scale Anchoring"新问题（低分辨率训练导致高分辨率推理误差锚定），并提出架构无关的频率表征学习（FRL），通过 Nyquist 归一化频率编码使误差随分辨率提升而下降，在 8 种主流架构上验证有效。

**[DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)**

:   提出 DiffusionBlocks，将残差网络的逐层更新解释为连续时间扩散过程的离散化步骤，从而将网络切分为可完全独立训练的 block，在保持端到端训练性能的同时按 block 数 B 倍减少训练显存。

**[Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss](generalizing_linear_autoencoder_recommenders_with_decoupled_expec.md)**

:   将 EDLAE 推荐模型的目标函数推广为解耦期望二次损失（DEQL），在超参数 $b>0$ 的更广范围内推导出闭式解，并通过 Miller 矩阵逆定理将计算复杂度从 $O(n^4)$ 降至 $O(n^3)$，在多个基准数据集上超越 EDLAE 和深度学习模型。

**[Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)**

:   提出 Horizon Imagination (HI)，通过在去噪中途采样动作并行处理多个未来帧，将扩散世界模型的 on-policy 想象计算量降至每帧不到一次完整去噪，同时保持控制性能。

**[InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions](interacthuman_multi-concept_human_animation_with_layout-aligned_audio_conditions.md)**

:   提出 InterActHuman，通过自动推断时空布局的掩码预测器和迭代掩码引导策略，实现多人/人物交互场景下的音频驱动视频生成，支持每个角色独立的语音驱动口型同步和身体动作。

**[Mechanism of Task-oriented Information Removal in In-context Learning](mechanism_of_task-oriented_information_removal_in_in-context_learning.md)**

:   从"信息移除"的新视角解释 In-context Learning（ICL）的内部机制：发现 LM 在零样本时将查询编码为包含所有可能任务信息的"非选择性表征"（导致随机输出），而 few-shot ICL 的核心作用是模拟一种"任务导向的信息移除"过程——通过识别出的"Denoising Heads"（去噪注意力头）从纠缠的表征中选择性移除冗余任务信息，引导模型聚焦目标任务。消融实验证实阻断去噪头后 ICL 准确率显著下降。

**[ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)**

:   提出 ProtoTS，通过层级原型学习实现可解释时间序列预测：少量粗粒度原型提供全局模式概览，逐级细分捕捉局部变化，结合多通道嵌入与瓶颈融合处理异质外生变量。在 LOF 数据集上 MSE 降低 48.3%，MAE 降低 20.9%，且支持专家编辑原型以进一步提升性能。

**[Sharpness-Aware Machine Unlearning](sharpness-aware_machine_unlearning.md)**

:   本文从信号-噪声分解的视角系统分析了 SAM 在机器遗忘场景下的理论特性，发现 SAM 在遗忘集上会"放弃"去噪能力但在保留集上仍维持优势，进而提出 Sharp MinMax 算法——将模型拆成两部分分别做锐度最小化（保留）和锐度最大化（遗忘），达到SOTA遗忘效果。

**[Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)**

:   首次系统比较扩散语言模型（dLLM）和自回归模型（AR LLM）的层间表征结构，发现原生 dLLM 具有更强的层级抽象和早期层冗余性，据此提出静态、任务无关的推理时层跳过策略，在 LLaDA 上跳过 6 层（18.75% FLOPs 削减）仍保持 90%+ 性能。

**[Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution](trust_but_verify_adaptive_conditioning_for_reference-based_diffusion_super-resol.md)**

:   提出 Ada-RefSR，一个基于"Trust but Verify"原则的单步参考引导扩散超分辨率框架，通过自适应隐式相关性门控（AICG）机制在利用可靠参考信息的同时抑制错误融合，仅增加 0.13% 计算开销。

**[wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)**

:   提出 wd1，一种无需策略比率（ratio-free）的加权对数似然策略优化方法用于扩散语言模型（dLLM）的 RL 微调，通过正样本加权和负样本惩罚避免了 GRPO 中策略比率估计的偏差和高方差问题，在 LLaDA-8B 上实现了 Sudoku +59%、GSM8K 84.5% 的 SOTA 性能。
