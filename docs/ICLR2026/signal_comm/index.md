---
title: >-
  ICLR2026 信号/通信方向 8篇论文解读
description: >-
  8篇ICLR2026 信号/通信方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🔬 ICLR2026** · **8** 篇论文解读

**[Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds](deterministic_bounds_and_random_estimates_of_metric_tensors_on_neuromanifolds.md)**

:   本文通过分析低维概率分布核空间的Fisher信息矩阵(FIM)谱性质，为神经网络参数空间(神经流形)上的度量张量建立了确定性上下界，并基于Hutchinson迹估计器引入了一族有界方差的无偏随机估计方法，仅需单次反向传播即可高效计算。

**[FASA: Frequency-Aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)**

:   本文发现RoPE中频率块（FC）级别的功能稀疏性——少数"主导FC"可有效预测token重要性，据此提出FASA框架，通过主导FC预估token重要性+聚焦注意力计算两阶段实现无训练的KV缓存压缩，在LongBench上仅保留256个token接近100%全KV性能，AIME24上用18.9%缓存实现2.56×加速。

**[Group Representational Position Encoding (GRAPE)](group_representational_position_encoding.md)**

:   提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

**[Learning Molecular Chirality via Chiral Determinant Kernels](learning_molecular_chirality_via_chiral_determinant_kernels.md)**

:   提出手性行列式核(ChiDeK)来编码 SE(3) 不变的手性矩阵，首次在 GNN 框架中统一处理中心手性和轴向手性，结合交叉注意力传播立体化学信息，在新构建的轴向手性基准上准确率提升 >7%。

**[Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)**

:   提出Multi-Agent System Search（MASS）框架，通过交错优化提示词和拓扑结构的三阶段策略（局部提示优化→拓扑搜索→全局提示优化），自动发现高性能的多智能体系统设计。

**[Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)**

:   通过大规模实证研究量化23个VQA基准中的模态内依赖和模态间依赖，揭示多数基准存在严重的单模态捷径，且消除文本偏差往往引入图像偏差，为多模态基准设计提供定量评估框架。

**[Robust Preference Alignment via Directional Neighborhood Consensus](robust_preference_alignment_via_directional_neighborhood_consensus.md)**

:   提出Robust Preference Selection (RPS)，一种无需重训练的推理时偏好对齐增强方法，通过从目标偏好的局部邻域采样多个候选方向并生成响应、再根据原始偏好选择最优响应，在OOD偏好上相比基线达到最高69%的胜率。

**[Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)**

:   提出Spectrum Tuning后训练方法，通过在90+任务的分布拟合数据集上训练，改善语言模型的上下文可操控性、输出空间覆盖度和分布对齐能力，揭示当前指令调优会损害模型的上下文可操控性。
