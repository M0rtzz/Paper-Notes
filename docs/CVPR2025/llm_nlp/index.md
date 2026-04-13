---
title: >-
  CVPR2025 LLM/NLP方向 14篇论文解读
description: >-
  14篇CVPR2025 LLM/NLP方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM/NLP

**📷 CVPR2025** · 共 **14** 篇

**[Breaking The Low-Rank Dilemma Of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)**

:   从理论上揭示线性注意力性能不及 Softmax 注意力的根本原因是输出特征的低秩问题，提出秩增强线性注意力（RALA），通过增强 KV 缓存秩和输出特征秩两种互补策略，在保持线性复杂度的同时追平甚至超越 Softmax 注意力的表现。

**[Building Vision Models Upon Heat Conduction](building_vision_models_upon_heat_conduction.md)**

:   提出 vHeat 视觉 backbone，将图像 patch 建模为热源，利用物理热传导方程通过 DCT/IDCT 变换实现 $O(N^{1.5})$ 复杂度的信息传播，在 ImageNet-1K 上以 3 倍吞吐量和 80% 更少 GPU 显存达到 84.0% top-1 准确率。

**[Dora Sampling And Benchmarking For 3D Shape Variational Auto-Encoders](dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   提出 Dora-VAE，通过 Sharp Edge Sampling (SES) 关注几何锐边区域、Dual Cross-Attention 分别处理均匀和显著采样点，以仅 1,280 个 latent codes（8× 小于 XCube-VAE 的 10,000+）实现更优的 3D 形状重建质量，同时建立了新的 Dora-Bench 评测基准。

**[Empowering Llms To Understand And Generate Complex Vector Graphics](empowering_llms_to_understand_and_generate_complex_vector_graphics.md)**

:   通过引入55个SVG语义token、构建580k条指令微调数据集SVGX-SFT，使任意LLM能准确理解和生成复杂矢量图形，在文本对齐度和美观度上超越GPT-4o和Claude，推理速度比优化方法快50-150倍。

**[Imagine And Seek Improving Composed Image Retrieval With An Imagined Proxy](imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)**

:   提出IP-CIR方法，通过大语言模型生成"想象中的目标图像描述"作为代理，将组合图像检索(CIR)转化为标准图像检索问题，在CIRR和FashionIQ等基准上达到零样本SOTA。

**[Let Samples Speak Mitigating Spurious Correlation By Exploiting The Clusterness ](let_samples_speak_mitigating_spurious_correlation_by_exploiting_the_clusterness_.md)**

:   提出NSF方法，通过利用样本在特征空间中的聚类特性自动识别依赖虚假特征的样本组，无需组标注即可训练出对虚假相关性鲁棒的分类器，最差组准确度显著超越ERM基线。

**[Lost In Translation Found In Context Sign Language Translation With Contextual C](lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)**

:   通过引入背景视频描述、历史翻译和伪词汇表三种上下文线索，结合Llama3-8B的LoRA微调，实现了连续手语到文本的精确翻译，在BOBSL数据集上相比SOTA提升40%以上。

**[Robust Message Embedding Via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)**

:   首次将Transformer注意力机制与归一化流结合，通过可逆QR码转换、可逆令牌融合和注意力仿射耦合块，实现高质量高容量的隐写，在打印-拍摄等极端物理扭曲下仍能准确还原消息。

**[ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)**

:   首次在人类动作生成领域探索和验证缩放律，提出ScaMo系统通过Motion FSQ-VAE和文本前缀自回归变换器，发现计算预算、模型/词汇表大小和数据之间的幂律关系，为动作生成的大规模训练提供理论指导。

**[SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)**

:   提出SoftShadow框架，使用连续软掩码（而非二值硬掩码）精确捕捉半影区域，结合SAM+LoRA自适应和半影形成约束损失，在SRD和LRSS数据集上达到SOTA且无需外部掩码输入。

**[Spiking Transformer With Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)**

:   将空间-时间注意力机制融入脉冲Transformer架构，通过时空解耦的注意力设计和脉冲驱动的自注意机制，在保持SNN能效优势的同时缩小与ANN的性能差距，在多个视觉基准上达到SNN SOTA。

**[Staa-Snn Spatial-Temporal Attention Aggregator For Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)**

:   通过在SNN中集成全局上下文自注意(GC)、位置编码(PE)、步骤注意(SA)和时间步随机退出(TSRD)四大模块，STAA-SNN在CIFAR-10/100和ImageNet上达到97.14%/82.05%/70.40%的SNN SOTA性能。

**[Test-Time Visual In-Context Tuning](test-time_visual_in-context_tuning.md)**

:   首次系统研究VICL模型在分布偏移下的鲁棒性问题，提出VICT方法利用循环一致性自监督信号在测试时进行单样本微调，在6个视觉任务和15种图像破坏下显著改进Painter等VICL模型。

**[Vires Video Instance Repainting Via Sketch And Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)**

:   提出ViReS框架，通过草图和文本双重引导实现视频中特定实例的重绘，利用时序注意力和实例掩码保持背景不变和时间一致性，在多种视频编辑场景下生成高质量结果。
