---
title: >-
  CVPR2026 社会计算方向 4篇论文解读
description: >-
  4篇CVPR2026 社会计算方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**📷 CVPR2026** · **4** 篇论文解读

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)**

:   这篇论文把 Transformer 的层深看成离散时间系统，证明在给定上下文附近可以用一个 32 维的低阶线性状态空间代理去近似 GPT-2 的层间传播与干预响应，而且模型越大，这个低阶代理越准确，还能据此算出比启发式注入更省能量的多层干预策略。

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](as_language_models_scale_loworder_linear_depth_dyn.md)**

:   将 Transformer 的逐层前向传播视为离散时间动力系统，构建32维低阶线性层变体（LLV）代理模型来近似最后token隐状态的深度传播动力学——发现该代理在GPT-2-large上预测逐层干预增益的Spearman相关可达0.995，且这种线性可辨识性随模型规模单调增强（GPT-2→medium→large），进而利用代理模型的闭式最优解实现比启发式干预策略能量低2-5倍的多层激活引导方案。

**[Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)**

:   本文提出利用合成数据生成过程中自动获得的"出处信息"（provenance）作为辅助监督信号，通过输入梯度引导（抑制非目标区域的输入梯度）直接促进模型学习聚焦于目标区域的判别性表示，在弱监督定位、时空动作检测和图像分类等多任务多模态上验证了有效性。

**[Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)**

:   提出 E2OAL，一个无需额外检测器的开放集主动学习框架，通过标签引导聚类发现未知类潜在结构、Dirichlet 校准辅助头联合建模已知/未知类别，并设计两阶段自适应查询策略，在多个基准上同时实现高准确率、高查询纯度和高训练效率。
