---
title: >-
  AAAI2026 LLM效率方向 9篇论文解读
description: >-
  9篇AAAI2026 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**🤖 AAAI2026** · **9** 篇论文解读

**[A Content-Preserving Secure Linguistic Steganography](a_content-preserving_secure_linguistic_steganography.md)**

:   提出首个内容保持型语言隐写术范式CLstega，通过微调掩码语言模型（MLM）来可控地变换预测分布，将秘密信息嵌入到不做任何修改的原始文本中，实现了100%提取成功率和近乎完美的安全性（隐写分析检测准确率接近随机猜测的0.5）。

**[Attention Retention for Continual Learning with Vision Transformers](attention_retention_for_continual_learning_with_vision_transformers.md)**

:   提出ARCL-ViT框架，通过注意力掩码生成和梯度掩码两步策略防止ViT在持续学习中的注意力漂移，在ImageNet-R和CIFAR-100上取得SOTA结果，证明保持注意力模式是解决灾难性遗忘的关键。

**[Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)**

:   提出一种大小模型协作框架，通过对本地查询进行"主题迁移+数值替换"的两阶段匿名化来保护敏感数据，同时让远端 GPT-4 以可执行 Python 代码（即插即用工具）形式返回推理方案，本地仅需做数值回代即可获得答案，在 FinQA 和 MultiHiertt 上准确率提升 16-44% 且数据泄露降低 2-45%。

**[Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models](harnessing_the_unseen_the_hidden_influence_of_intrinsic_knowledge_in_long-contex.md)**

:   首次系统研究长上下文语言模型中参数知识(parametric knowledge)对生成的影响，发现其影响随上下文长度增长而增强，且现有方法提升外部检索能力会抑制参数召回能力，据此提出Hybrid Needle-in-a-Haystack测试来同时评估两种能力。

**[InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)**

:   提出 InterMoE，通过 Dynamic Temporal-Selective MoE 架构解决文本驱动的双人 3D 交互运动生成中的个体特征保持和语义忠实度问题：Synergistic Router 融合语义和运动学特征引导路由，Dynamic Temporal Selection 让专家动态选择关键时间帧，在 InterHuman 上 FID 降低 9%、InterX 上降低 22%。

**[Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)**

:   提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

**[Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)**

:   提出 Learning-from-the-Undesirable (LfU)，一种面向 SFT 的正则化方法，通过对辅助模型施加梯度上升模拟"不良行为"，再通过表示级一致性损失约束原模型与不良模型的内部表征保持一致，有效缓解有限数据微调中的过拟合、遗忘和对抗脆弱性问题。

**[MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction](microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)**

:   提出 MicroEvoEval，首个面向图像级微观结构演化预测的标准化基准：涵盖 4 个代表性物理任务（平面波、晶粒生长、旋节分解、枝晶凝固）、14 个模型（5 个领域特定 + 9 个通用时空架构）、多维度评估（数值精度 + 物理保真度 + 计算效率），发现现代通用架构（如 VMamba）在长期稳定性和物理保真度上优于领域特定模型，且计算效率高一个数量级。

**[Think How Your Teammates Think: Active Inference Can Benefit Decentralized Execution](think_how_your_teammates_think_active_inference_can_benefit_decentralized_execut.md)**

:   提出 AIM（Active Inference Modeling）框架，在去中心化多智能体强化学习中，不依赖通信机制，仅基于局部观测建模队友的主动推理过程（感知-信念-动作三重肖像），并通过准确性-相关性双重过滤机制选择性融合队友信念，在 SMAC、SMACv2、MPE 和 GRF 四大基准上取得最优或接近最优表现。
