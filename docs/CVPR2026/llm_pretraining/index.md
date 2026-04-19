---
title: >-
  CVPR2026 预训练方向 7篇论文解读
description: >-
  7篇CVPR2026 预训练方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**📷 CVPR2026** · **7** 篇论文解读

**[Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)**

:   提出 MergeGuard，一种主动式双阶段权重保护框架：Stage 1通过L2正则化分散任务关键权重，Stage 2注入结构化扰动破坏合并兼容性，在保持保护模型<1.5%性能损失的同时使合并模型精度下降高达90%。

**[FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)**

:   提出 FlowMotion，一种无需训练的视频运动迁移框架，通过直接利用 flow-based T2V 模型的预测输出（latent prediction）构建运动引导信号，避免对模型内部层做梯度回传，在保持运动保真度的同时大幅降低推理时间和显存开销。

**[Linking Modality Isolation in Heterogeneous Collaborative Perception](linking_modality_isolation_in_heterogeneous_collaborative_perception.md)**

:   提出 CodeAlign 框架，通过码本构建离散代码空间和跨模态 Feature-Code-Feature (FCF) 翻译，首次解决异构协同感知中不同模态从未在训练数据中共现的"模态隔离"问题，仅需 HEAL 8% 训练参数、通信量降低 1024 倍，同时达到 SOTA 感知性能。

**[MXNorm: Reusing MXFP block scales for efficient tensor normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)**

:   MXNorm 提出将 RMSNorm 与 MXFP 量化融合：利用 MXFP 量化过程中已经计算好的 block absmax 来近似 RMS 值，从而省掉单独的归一化 reduction 操作，在 Llama 3 最高 8B 参数的预训练中保持训练精度，同时在 GB200 上实现最高 2.4 倍的 kernel 加速。

**[MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_tensor_normalisation.md)**

:   GPU矩阵乘法吞吐量提升(80x)远超reduction/elementwise操作(5-9x)，RMSNorm正成为低精度训练的新瓶颈。MXNorm直接复用MXFP8量化时已计算的block scales来估计RMS，实现32倍reduction大小缩减。理论上证明block absmax的广义p-mean可收敛到RMS的常数倍。Llama 3 125M/1B/8B预训练验证MXNorm(p=2)与RMSNorm训练精度差异minimal，torch.compile实测isolated kernel最高2.4x加速、Llama 3 8B transformer layer在MXFP8下+1.3%、NVFP4下+2.6%加速。Drop-in replacement，无额外超参数。

**[Watch and Learn: Learning to Use Computers from Online Videos](watch_and_learn_computer_use_from_videos.md)**

:   提出 Watch & Learn 框架, 通过逆动力学模型 (IDM) 将 YouTube 教程视频自动转化为可执行的 UI 轨迹数据 (53K+ 轨迹, 免去人工标注), 基于此数据增强 CUA 能力, 在 OSWorld 上让 Qwen 2.5VL-7B 提升 +11.1%, UI-TARS-1.5-7B 提升 +3.8%.

**[Watch and Learn: Learning to Use Computers from Online Videos](watch_and_learn_learning_to_use_computers_from_online_videos.md)**

:   提出 Watch & Learn (W&L) 框架，通过逆动力学模型 (IDM) 将互联网上的人类计算机操作视频自动转化为可执行的 UI 轨迹数据，生成 53K+ 高质量轨迹，作为 ICL 示例或 SFT 训练数据显著提升各类 CUA 性能。
