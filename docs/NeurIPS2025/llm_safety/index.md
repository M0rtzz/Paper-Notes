---
title: >-
  NeurIPS2025 LLM安全方向 10篇论文解读
description: >-
  10篇NeurIPS2025 LLM安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM安全

**🧠 NeurIPS2025** · **10** 篇论文解读

**[A Cramér–von Mises Approach to Incentivizing Truthful Data Sharing](a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)**

:   提出一种基于 Cramér-von Mises 两样本检验统计量的激励机制，在贝叶斯和无先验两种设定下均能证明"如实提交数据"构成（近似）Nash 均衡，同时鼓励参与者提交更多真实数据，且不依赖对数据分布的强假设（如高斯、伯努利）。

**[A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)**

:   将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

**[Buffer Layers for Test-Time Adaptation](buffer_layers_for_test-time_adaptation.md)**

:   提出 Buffer 层作为测试时自适应 (TTA) 的新范式，替代传统的归一化层更新，从根本上保留预训练骨干网络的完整性，有效缓解灾难性遗忘并在多种架构和 TTA 框架中实现一致的性能提升。

**[Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)**

:   发现LLM微调后上游样本遗忘与新学任务之间的关联矩阵具有低秩结构（rank-3即$R^2>0.69$），利用矩阵补全预测未见任务导致的遗忘，指导选择性回放以减轻遗忘。

**[Finding Structure in Continual Learning](finding_structure_in_continual_learning.md)**

:   提出基于Douglas-Rachford Splitting (DRS)的持续学习优化框架，将稳定性与可塑性解耦为两个独立的近端子问题，并结合Rényi散度替代KL散度实现更鲁棒的先验对齐，从而在无需回放缓冲区或额外模块的条件下有效缓解灾难性遗忘。

**[Procurement Auctions with Predictions: Improved Frugality for Facility Location](procurement_auctions_with_predictions_improved_frugality_for_facility_location.md)**

:   研究策略性无容量限制设施选址问题中的采购拍卖设计，证明了经典VCG拍卖的节俭比恰好为3（改进了此前已知的上界4），并设计了利用预测信息的学习增强拍卖机制，在预测准确时实现接近最优的节俭比，同时在预测任意不准确时仍保持常数级鲁棒性。

**[SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)**

:   提出 SIMU 两阶段框架：先通过梯度聚合识别编码遗忘集信息的关键 MLP 神经元，再仅对这些神经元进行二阶（Sophia）优化遗忘，在保持遗忘效果的同时大幅提升模型原有能力的保留。

**[Stop DDoS Attacking the Research Community with AI-Generated Survey Papers](stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)**

:   这篇立场论文将AI生成综述论文的泛滥类比为对学术社区的"DDoS攻击"，通过对arXiv 2020-2024年10,063篇CS综述论文的系统定量分析，揭示了ChatGPT发布后综述论文数量、AI生成分数和异常作者数的同步激增现象，深入剖析了AI综述的四大质量缺陷（结构混乱、分类缺乏原创、引用不准确、内容高度冗余）及其对研究者-审稿人-编辑三方的文化冲击，提出了涵盖透明度要求、严格审查标准、冗余限制、AI检测辅助和"动态活综述"平台在内的全面应对框架。

**[Teaming LLMs to Detect and Mitigate Hallucinations](teaming_llms_to_detect_and_mitigate_hallucinations.md)**

:   将单模型一致性方法（Self-Consistency + Semantic Entropy）推广到多个异构 LLM 的"联盟"设置，通过聚合不同训练背景的模型响应来打破单模型一致性幻觉，在 15 个 LLM 组成的模型池中评估大量联盟组合，发现匹配的强模型联盟在 92% 的情况下超越最强单模型基线，同时推理成本更低。

**[TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)**

:   提出 TRUST 架构，将 Transformer 的注意力机制与 U-Net 解码器结合，在感知矩阵未知的条件下同时学习感知算子和重建稀疏信号，在 SSIM 和 PSNR 上显著超越传统方法。
