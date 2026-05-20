---
title: >-
  ICML2026 视频生成方向6篇论文解读
description: >-
  6篇ICML2026的视频生成方向论文解读，涵盖视频生成、扩散模型、多模态、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "视频生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "多模态"
  - "模型压缩"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎬 视频生成

**🧪 ICML2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [📷 CVPR2026 (54)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [🧠 NeurIPS2025 (22)](../../NeurIPS2025/video_generation/index.md) · [📹 ICCV2025 (48)](../../ICCV2025/video_generation/index.md)

🔥 **高频主题：** 视频生成 ×2

**[Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)**

:   SVOO 发现视频 DiT 每一层的注意力稀疏度是「层内输入无关、层间显著异质」的内在属性，据此先做离线分层稀疏度标定、再做在线 QK 双向协同聚类划块，免训练地在 Wan/HunyuanVideo 等 7 个模型上把 PSNR 维持 29 dB 的同时实现最高 1.93× 加速。

**[Exploring Data-Free LoRA Transferability for Video Diffusion Models](exploring_data-free_lora_transferability_for_video_diffusion_models.md)**

:   本文首次对视频扩散模型（VDM）的 full fine-tune (FFT) 和 LoRA 做权重空间分析，发现两者都"保留奇异谱、只旋转奇异子空间"，但在 head clusters 上路由方向冲突；据此提出 CASA——一个 data-free 的"按聚类做谱仲裁"的 LoRA 迁移方法，把基座 Wan2.1 上训的 LoRA 直接迁到 FastWan 等蒸馏后变体，无需任何用户数据/重训。

**[Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)**

:   针对 In-Context Learning 范式下视频编辑的二次注意力瓶颈，作者基于"context token 显著性低于 source token"以及"Query 锐度正比于 Taylor 近似误差"两条洞察设计了 In-context Sparse Attention（ISA），并训练出 LIVEditor，在多个 benchmark 上既加速 ~60% 又超越 SOTA 全注意力模型。

**[MiVE: Multiscale Vision-language features for reference-guided video Editing](mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)**

:   MiVE 把 Qwen3-VL 的**首层 + 末层**隐状态同时抽出来作为多尺度条件 token, 与 VAE 视觉 latent 拼成一个长序列, 在统一的自注意力 DiT 里做参考图引导的视频编辑, 在 60 段 720P benchmark 上人类偏好和 6 个 VLM 自动评分都拿到第一, 超过开源 Wan-Animate 和商用 Kling O1.

**[Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)**

:   QVG 是面向自回归视频扩散的训练免微调 KV-Cache 量化框架——通过语义感知聚类做 token 平滑、并以渐进残差多阶段压缩残差，在 LongCat-Video/HY-WorldPlay/Self-Forcing 上把 KV 显存压低到原来的 1/7，端到端延迟开销 <4%，2 bit 下质量大幅领先 KIVI/QuaRot 等 LLM 量化基线。

**[VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)**

:   VAnim 把开放域 text-to-SVG 动画建模为「持久 DOM 树上的稀疏状态更新」+「Identification-First 运动规划」+「GRPO 渲染感知强化学习」，序列长度压缩 $9.86\times$ 的同时保持拓扑一致，并显著超越 GPT-5.2、Gemini 3 Pro 与 LiveSketch。
