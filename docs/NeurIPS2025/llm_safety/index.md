---
title: >-
  NeurIPS2025 LLM安全方向 10篇论文解读
description: >-
  10篇NeurIPS2025 LLM安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM安全

**🧠 NeurIPS2025** · 共 **10** 篇

**[A Cramrvon Mises Approach To Incentivizing Truthful Data Sha](a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)**

:   提出一种基于 Cramér-von Mises 两样本检验统计量的激励机制，在贝叶斯和无先验两种设定下均能证明"如实提交数据"构成（近似）Nash 均衡，同时鼓励参与者提交更多真实数据，且不依赖对数据分布的强假设（如高斯、伯努利）。

**[A Reliable Cryptographic Framework For Empirical Machine Unl](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)**

:   将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

**[Buffer Layers For Test-Time Adaptation](buffer_layers_for_test-time_adaptation.md)**

:   提出 Buffer 层作为测试时自适应 (TTA) 的新范式，替代传统的归一化层更新，从根本上保留预训练骨干网络的完整性，有效缓解灾难性遗忘并在多种架构和 TTA 框架中实现一致的性能提升。

**[Demystifying Language Model Forgetting With Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)**

:   发现LLM微调后上游样本遗忘与新学任务之间的关联矩阵具有低秩结构（rank-3即$R^2>0.69$），利用矩阵补全预测未见任务导致的遗忘，指导选择性回放以减轻遗忘。

**[Finding Structure In Continual Learning](finding_structure_in_continual_learning.md)**

:   提出基于Douglas-Rachford Splitting (DRS)的持续学习优化框架，将稳定性与可塑性解耦为两个独立的近端子问题，并结合Rényi散度替代KL散度实现更鲁棒的先验对齐，从而在无需回放缓冲区或额外模块的条件下有效缓解灾难性遗忘。

**[Procurement Auctions With Predictions Improved Frugality For Facility Location](procurement_auctions_with_predictions_improved_frugality_for_facility_location.md)**

:   研究策略性无容量限制设施选址问题中的采购拍卖设计，证明了经典VCG拍卖的节俭比恰好为3（改进了此前已知的上界4），并设计了利用预测信息的学习增强拍卖机制，在预测准确时实现接近最优的节俭比，同时在预测任意不准确时仍保持常数级鲁棒性。

**[Simu Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)**

:   提出 SIMU 两阶段框架：先通过梯度聚合识别编码遗忘集信息的关键 MLP 神经元，再仅对这些神经元进行二阶（Sophia）优化遗忘，在保持遗忘效果的同时大幅提升模型原有能力的保留。

**[Stop Ddos Attacking The Research Community With Ai-Generated Survey Papers](stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)**

:   这篇立场论文以"综述论文 DDoS 攻击"为隐喻，通过定量分析 arXiv 2020-2024 年间 10,063 篇 CS 综述论文，揭示 AI 生成综述的爆炸式增长趋势和质量问题，提出规范 AI 辅助综述写作和建设"动态活综述"的愿景。

**[Teaming Llms To Detect And Mitigate Hallucinations](teaming_llms_to_detect_and_mitigate_hallucinations.md)**

:   提出 Consortium Consistency 方法，将单模型一致性方法（Self-Consistency 和 Semantic Entropy）扩展到多模型协作设置，通过聚合多个异构 LLM 的响应来实现更可靠的幻觉检测和缓解，同时降低推理成本。

**[Trust -- Transformer-Driven U-Net For Sparse Target Recovery](trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)**

:   提出 TRUST 架构，将 Transformer 的注意力机制与 U-Net 解码器结合，在感知矩阵未知的条件下同时学习感知算子和重建稀疏信号，在 SSIM 和 PSNR 上显著超越传统方法。
