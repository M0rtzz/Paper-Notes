---
title: >-
  CVPR2025 文本生成方向 3篇论文解读
description: >-
  3篇CVPR2025 文本生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✍️ 文本生成

**📷 CVPR2025** · **3** 篇论文解读

**[ArtFormer: Controllable Generation of Diverse 3D Articulated Objects](artformer_controllable_generation_of_diverse_3d_articulated_objects.md)**

:   提出ArtFormer框架，通过树结构参数化和条件扩散Shape Prior，从文本/图像描述生成高质量、多样化且运动学关系准确的3D关节物体，在生成质量和多样性上显著超越现有方法。

**[Dense Match Summarization for Faster Two-view Estimation](dense_match_summarization_for_faster_two-view_estimation.md)**

:   提出高效的稠密匹配摘要方案，从大量稠密对应中生成一小组代表性匹配子集，在保持位姿估计精度的同时大幅加速 RANSAC 鲁棒估计。

**[LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)**

:   提出LotusFilter，通过离线预计算每个向量的邻近关系构建截断表(cutoff table)，在线阶段用贪心集合删除实现多样化过滤，将传统 $O(DS^2)$ 的多样化搜索降至 $O(T+S+KL)$，过滤仅需0.02ms/query，内存仅为传统方法的1/40。
