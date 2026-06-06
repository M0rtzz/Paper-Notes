---
title: >-
  CVPR2025 对齐 / RLHF方向17篇论文解读
description: >-
  17篇CVPR2025的对齐 / RLHF 方向论文解读，涵盖对齐/RLHF、扩散模型、多模态、LLM、语义分割、医学影像等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "对齐 / RLHF"
  - "论文解读"
  - "论文笔记"
  - "对齐/RLHF"
  - "扩散模型"
  - "多模态"
  - "LLM"
  - "语义分割"
  - "医学影像"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐 / RLHF

**📷 CVPR2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/llm_alignment/index.md) · [💬 ACL2026 (30)](../../ACL2026/llm_alignment/index.md) · [📷 CVPR2026 (2)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/llm_alignment/index.md) · [🤖 AAAI2026 (18)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/llm_alignment/index.md)

🔥 **高频主题：** 对齐/RLHF ×14 · 扩散模型 ×5 · 多模态 ×4 · LLM ×3 · 语义分割 ×2

**[Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)**

:   本文提出 Step-by-step Preference Optimization（SPO），在每个去噪步中从同一噪声潜变量采样多个候选，用 step-aware 偏好模型选择 win/lose 对来指导扩散模型微调，从通用偏好数据中隐式蒸馏美学信息，在 SD-1.5 和 SDXL 上显著提升美学质量且收敛速度远快于 DPO。

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种求解可转向等变 CNN 核约束方程的替代方法，通过在不动点处求解更简单的不变性条件再"转向"到任意点，绕过了计算 Clebsch-Gordan 系数的需要，为 SO(2)、O(2)、SO(3)、O(3) 及 Lorentz 群给出了显式的核基底公式。

**[Boost Your Human Image Generation Model via Direct Preference Optimization](boost_your_human_image_generation_model_via_direct_preference_optimization.md)**

:   提出 HG-DPO，以真实人像作为 DPO 的 winning image（而非生成图像对）+ 三阶段课程学习（Easy/Normal/Hard）渐进弥合生成-真实图像分布 gap + 统计匹配损失解决色偏，FID 从 37.34 降至 29.41（-21.4%），CI-Q 0.906→0.934，win-rate 超越 Diffusion-DPO 达 99.97%。

**[CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation](cad-llama_leveraging_large_language_models_for_computer-aided_design_parametric_.md)**

:   本文提出 CAD-Llama 框架，通过层次化标注管线将 3D CAD 模型转化为富含语义描述的 Python 风格代码（SPCC），再用自适应预训练和指令微调将 LLaMA3-8B 转化为参数化 CAD 模型生成器，在 text-to-CAD 任务上精度超出先前方法约 14%，并支持补全、添加、删除等多种 CAD 编辑任务。

**[Calibrated Multi-Preference Optimization for Aligning Diffusion Models](calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)**

:   本文提出 Calibrated Preference Optimization（CaPO），通过 win-rate 校准将不同奖励模型的分数统一为期望胜率，并设计基于 Pareto 前沿的配对采样策略（FRS）来处理多奖励信号间的冲突，在 SDXL 和 SD3-Medium 上一致地超越 DPO 和 IPO 方法。

**[Continual SFT Matches Multimodal RLHF with Negative Supervision](continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)**

:   通过梯度分析发现多模态 RLHF 相比持续 SFT 的核心优势在于 rejected response 中的负监督信号，据此提出 nSFT 方法，用 LLM 从拒绝回复中提取错误信息并构造纠正性对话数据，仅用 SFT loss 就能匹配甚至超越 DPO/PPO 等 RLHF 方法，且只需 1 个模型，显存效率大幅提升。

**[Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)**

:   首次将课程学习引入 DPO 并首次将 DPO 适配到一致性模型，通过从"容易区分的偏好对"到"难以区分的偏好对"渐进训练，在文本对齐、美学和人类偏好上全面超越标准 DPO 和 DDPO，且仅需 1/10 训练数据量。

**[Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)**

:   NaPO 针对MLLM的模态偏差问题（过度依赖语言先验或视觉细节），通过mask模态信息构造偏差数据集RLAIF-V-Bias，并提出基于负Box-Cox变换的噪声感知偏好优化算法，在自动构造的含噪数据上实现鲁棒训练，在去偏和减幻觉上均取得显著效果。

**[Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)**

:   探讨多模态大语言模型安全对齐是否真正需要精心策划的恶意数据，发现利用现有良性数据并结合简单的安全微调策略即可实现有效的安全对齐，大幅降低了安全对齐的数据成本。

**[Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation](enhancing_sam_with_efficient_prompting_and_preference_optimization_for_semi-supe.md)**

:   提出一种增强 SAM 的半监督医学图像分割框架：通过 CLIP 和 VQA 无监督生成包含语义、位置和形状信息的高效提示（无需专家标注），再用 DPO 偏好优化技术配合虚拟标注器（代替人类标注者提供排名/评分）训练最优分割策略，在肺分割、乳腺肿瘤分割、器官分割等多模态任务上达到 SOTA。

**[InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_diffusion_alignment.md)**

:   本文提出 DDIM-InPO，通过将扩散模型视为单步生成模型并利用 DDIM 反演技术找到与偏好数据高度相关的潜变量，实现仅需 400 步微调即可达到 SOTA 的高效扩散模型偏好对齐。

**[InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_with_reparametrized_ddim_for_efficient_di.md)**

:   提出 InPO（Inversion Preference Optimization），通过 DDIM 反演的重参数化技巧将偏好优化从需要完整去噪链的长马尔可夫过程简化为单步优化，在训练效率和生成质量上同时优于现有 Diffusion-DPO 方法。

**[Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising](jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)**

:   提出 JailNTL，首个针对 Non-Transferable Learning (NTL) 模型的黑盒攻击方法，通过测试时数据伪装将未授权域的数据"变装"为授权域的数据，仅用 1% 授权样本即可将未授权域准确率提升最高 55.7%，无需修改模型。

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](physmodpo_physically-plausible_humanoid_motion_with_preference_optimization.md)**

:   提出 PhysMoDPO，将 Direct Preference Optimization 应用于文本驱动的人体运动生成，通过将全身控制器（WBC）集成到训练 pipeline 中计算基于物理的奖励来构造偏好数据，使生成运动同时满足物理约束和文本指令，并在 Unitree G1 机器人上实现零样本部署。

**[Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)**

:   本文提出一种增强的SAM框架，通过BiomedCLIP、VQA和GPT-4生成无监督语义/位置/形状提示，并引入DPO启发的偏好对齐损失模拟人类反馈，在仅10%标注数据的半监督设置下实现了肺部、乳腺肿瘤和腹部器官分割的优异性能。

**[SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization](symdpo_boosting_in-context_learning_of_large_multimodal_models_with_symbol_demon.md)**

:   SymDPO 发现LMM在多模态ICL中存在"视觉上下文忽视"问题（用空白图替换示例图不影响性能），提出将示例中的文本答案替换为无语义随机符号，迫使模型必须理解视觉内容才能正确匹配符号与答案，通过DPO训练在OpenFlamingo和IDEFICS上一致提升了多模态ICL效果。

**[Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)**

:   提出 Task Preference Optimization（TPO），通过可学习的任务 token 将视觉任务专用头（区域定位/时序定位/分割）接入 MLLM，利用视觉任务标注作为"任务偏好"反向优化 MLLM，在不损害对话能力的前提下大幅提升细粒度视觉理解，VideoChat 基线上平均提升 14.6%。
