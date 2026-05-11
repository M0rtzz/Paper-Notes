---
title: >-
  ICCV2025 LLM 安全方向8篇论文解读
description: >-
  8篇ICCV2025的 LLM 安全方向论文解读，涵盖对抗鲁棒、联邦学习、持续学习、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# � LLM 安全

**📹 ICCV2025** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/llm_safety/index.md) · [📷 CVPR2026 (16)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (60)](../../NeurIPS2025/llm_safety/index.md) · [🧪 ICML2025 (32)](../../ICML2025/llm_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×3 · 联邦学习 ×2

**[Adversarial Robust Memory-Based Continual Learner](adversarial_robust_memory-based_continual_learner.md)**

:   揭示持续学习与对抗训练结合时的双重挑战（加速遗忘 + 梯度混淆），提出抗遗忘 Logit 校准（AFLC）和鲁棒感知经验回放（RAER）两个即插即用模块，在 Split-CIFAR10/100 和 Split-Tiny-ImageNet 上有效提升对抗鲁棒性达 8.13%。

**[Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)**

:   提出首个面向异步事件数据的不可学习样本生成方法（UEvs），设计了事件误差最小化噪声（E²MN）及自适应投影机制，使事件数据集在保持合法使用功能的同时阻止未授权模型从中学习。

**[ChartCap: Mitigating Hallucination of Dense Chart Captioning](chartcap_mitigating_hallucination_of_dense_chart_captioning.md)**

:   构建了包含56.5万张真实图表-描述对的大规模数据集ChartCap，通过类型特定的描述模式排除无关信息、强调结构与关键洞察，并提出无参考的Visual Consistency Score评估指标，有效减少VLM在图表描述中的幻觉问题。

**[Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)**

:   提出Gradient-Guided Sampling (GGS)内迭代采样策略，通过使用上一内迭代的梯度方向引导采样，在平衡Exploitation（攻击强度/损失极大值）和Exploration（跨模型泛化/平坦损失面）的困境中取得突破，在CNN/ViT/MLLM等多架构上显著超越现有迁移攻击方法。

**[Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)**

:   提出 FUCRT 方法，通过类感知表征变换实现联邦遗忘：将遗忘类的表征“变换”到语义最近的保留类，而非直接消除，配合双重对比学习对齐跨客户端的变换一致性，在四个数据集上实现 100% 遗忘保障的同时保持甚至提升剩余类性能。

**[Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)**

:   本文提出Geminio，首个利用视觉语言模型（VLM）实现自然语言引导的梯度反转攻击（GIA），使联邦学习中的恶意服务器可以用自然语言描述想要窃取的数据类型，并从大batch梯度中精准定位和重建匹配的隐私样本，同时不影响正常的FL模型训练。

**[Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)**

:   提出Oasis方法，仅需输入图像（无需任何文本提示）即可诱导MLLM自回归生成高质量多模态指令跟随数据，配合精细的指令质量控制机制，合成50万数据给LLaVA-NeXT带来平均3.1%的全面性能提升，且超越其他合成方法。

**[Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)**

:   本文首次研究防止视频数据被深度跟踪器未授权使用的问题，提出基于 DiT 的生成式框架生成时序不可学习样本（TUE），通过时间对比损失使跟踪器依赖扰动噪声进行时序匹配而非学习真实数据结构，实现了跨模型、跨数据集和跨任务的强可迁移性。
