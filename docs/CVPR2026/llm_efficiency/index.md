---
title: >-
  CVPR2026 LLM 效率方向4篇论文解读
description: >-
  4篇CVPR2026的 LLM 效率方向论文解读，涵盖持续学习、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "少样本学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**📷 CVPR2026** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (11)](../../ICML2026/llm_efficiency/index.md) · [💬 ACL2026 (20)](../../ACL2026/llm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

**[GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision](benchmarking_phd-level_coding_in_3d_geometric_computer_vision.md)**

:   首个面向3D几何计算机视觉的PhD级代码生成基准GeoCodeBench，包含100个从2025年顶会论文+代码库中精选的函数补全任务，配套自动化多样化单元测试，最强模型GPT-5仅36.6%通过率，揭示LLM在科学级3D代码实现上的巨大差距。

**[CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)**

:   提出 CHEEM 框架，通过分层探索-利用采样的 NAS 自动学习任务感知的动态 ViT 骨干——在每一层选择 Reuse/New/Adapt/Skip 四种操作——在 MTIL 和 VDD 两个挑战性持续学习基准上显著超越提示类方法，接近全量微调上界。

**[SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)**

:   对VAR模型注意力激活模式进行系统分析，揭示三大稀疏特性（注意力汇、跨尺度相似性、空间局部性），并提出SparVAR无训练加速框架，通过跨尺度自相似稀疏注意力（CS⁴A）和跨尺度局部稀疏注意力（CSLA）两个即插即用模块，实现8B模型1024×1024生成降至1秒级（1.57×加速），且几乎不损失高频细节。

**[StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)**

:   提出StoryTailor零样本视觉叙事生成管线，通过高斯中心注意力（GCA）缓解主体重叠和背景泄漏、动作增强奇异值重加权（AB-SVR）放大动作语义、选择性遗忘缓存（SFC）维护跨帧背景连续性，在单张RTX 4090上实现多主体、动作丰富的图像叙事生成，CLIP-T较基线提升10-15%。
