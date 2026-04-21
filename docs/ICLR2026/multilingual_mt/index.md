---
title: >-
  ICLR2026 多语言/翻译方向 6篇论文解读
description: >-
  6篇ICLR2026 多语言/翻译方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🔬 ICLR2026** · **6** 篇论文解读

**[ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)**

:   提出 ASSESS 框架，其核心是 TransTED Similarity 指标——通过将形式化数学命题解析为算子树 (Operator Tree)，并在标准树编辑距离 (TED) 基础上融入 Lean 证明策略驱动的语义变换，实现了在 EPLA 基准上 70.16% 准确率和 0.35 Kappa 分数的 SOTA 性能，同时仅需 CPU 资源即可复现。

**[ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)**

:   提出 Adaptive Transfer Scaling Law (ATLAS)，通过将有效数据量分解为目标语言、迁移语言和其他语言三项并引入数据重复饱和函数，在774个多语言训练实验（10M–8B参数、400+语言）上显著优于现有scaling law（多语言 $R^2$ 从0.67提升至0.98），并系统量化了跨语言迁移矩阵、多语言诅咒的容量约束以及预训练vs微调的计算交叉点。

**[Multilingual Routing in Mixture-of-Experts](multilingual_routing_in_mixture-of-experts.md)**

:   系统分析了MoE大语言模型中多语言路由模式，发现中间层存在跨语言共享专家且语言性能与英语路由对齐度强相关，进而提出推理时路由干预方法，通过在中间层激活英语任务专家，在3个模型×2个任务×15+语言上一致性地提升多语言性能1-2%。

**[Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity](prior-based_noisy_text_data_filtering_fast_and_strong_alternative_for_perplexity.md)**

:   提出基于 token 词频先验（term frequency）的文本数据过滤方法，通过计算文档中 token 先验的均值和标准差来检测异常文档，实现了比 PPL 过滤快 1000× 以上且下游性能更优的数据清洗效果。

**[Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity](prior-based_noisy_text_data_filtering_fast_and_strong_alternative_to_perplexity.md)**

:   提出基于 token 先验（词频统计）的文本数据过滤方法，利用文档内 token 先验的均值和标准差作为 PPL 的近似替代，在 20 个下游基准上取得最高平均性能，同时比 PPL 过滤快 1000 倍以上。

**[SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)**

:   利用稀疏自编码器（SAE）发现 LLM 中意外语言切换与目标语言特征异常高预激活值相关，提出 SASFT 方法在 SFT 训练中约束语言特征预激活值，将意外代码切换降低 50% 以上。
