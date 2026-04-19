---
title: >-
  ICCV2025 AI安全方向 23篇论文解读
description: >-
  23篇ICCV2025 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**📹 ICCV2025** · **23** 篇论文解读

**[A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_double-blind_federated_adaptation_of_foundation_models.md)**

:   本文提出BlindFed框架，通过全同态加密（FHE）友好的架构改造、两阶段分割学习和隐私增强策略，实现了基础模型的"双盲"联邦适配——数据方看不到模型，服务方看不到数据，在CIFAR-10上达到94.28%准确率，接近LoRA的95.92%。

**[A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_doubleblind_federated_adaptation_of_foundati.md)**

:   BlindFed提出了双盲联邦基础模型适配框架：通过FHE友好的架构重设计（多项式近似非线性操作）+ 两阶段分割学习（离线知识蒸馏 + 在线加密推理）+ 隐私增强（样本置换 + 随机块采样），在数据方看不到模型、模型方看不到数据的约束下实现了接近LoRA的适配精度。

**[Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](active_membership_inference_test_amint_enhancing_model_audit.md)**

:   提出Active MINT（aMINT），将成员推断检测作为训练时的优化目标，通过多任务学习让被审计模型与MINT模型联合训练、共享早期特征层，在不显著损失主任务性能的前提下，将训练数据的识别准确率从被动MINT的~60%大幅提升至80%以上。

**[Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](active_membership_inference_test_amint_enhancing_model_auditability_with_multi-t.md)**

:   本文提出 Active MINT（aMINT），一种多任务学习框架，在训练审核模型的同时联合训练 MINT 模型，使模型能够以超过 80% 的准确率检测特定数据是否被用于训练，显著优于現有的被动 MINT 和成员推断攻击方法。

**[Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering](ask_and_remember_a_questions-only_replay_strategy_for_continual_visual_question_.md)**

:   提出QUAD——一种仅存储过去任务问题（不存储图像）的持续VQA方法，通过问题重放和注意力一致性蒸馏，在保护隐私的同时超越存储图像的现有方法。

**[Backdoor Mitigation by Distance-Driven Detoxification](backdoor_mitigation_by_distance-driven_detoxification.md)**

:   本文提出Distance-Driven Detoxification（D3），将后门防御重新表述为约束优化问题——最大化微调后模型权重与中毒初始权重的距离，同时约束干净样本损失不超过阈值，从而有效逃逸"后门区域"，在7种SOTA攻击上取得最优或次优防御效果。

**[Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)**

:   提出Noisy Alignment（NA）方法，通过显式压缩投毒图像中的噪声成分来增强自监督对比学习的后门攻击效果，将攻击建模为二维图像布局优化问题，并推导出理论最优参数，在ImageNet-100上ASR提升最高达45.9%。

**[Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)**

:   提出Client2Vec机制，在联邦学习训练前利用CLIP编码器和分布偏移感知索引生成网络（DSA-IGN）为每个客户端生成包含标签和特征分布信息的索引向量，进而改善客户端采样、模型聚合和本地训练三个关键阶段。

**[Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)**

:   提出可控特征白化(CFW)框架，通过白化变换消除目标特征与偏差特征之间的线性相关性来缓解模型偏差，无需对抗学习或额外正则化超参数，且可通过加权系数平滑控制demographic parity和equalized odds之间的权衡。

**[Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)**

:   提出Gradient-Guided Sampling (GGS)内迭代采样策略，通过使用上一内迭代的梯度方向引导采样，在平衡Exploitation（攻击强度/损失极大值）和Exploration（跨模型泛化/平坦损失面）的困境中取得突破，在CNN/ViT/MLLM等多架构上显著超越现有迁移攻击方法。

**[FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)**

:   本文首次研究在私有数据场景下的联邦神经场（Neural Fields）元学习问题，揭示了现有联邦元学习方法在神经场任务中的严重隐私泄露机制，并提出FedMeNF，通过隐私保护损失函数正则化局部元梯度中的隐私信息，在保持快速优化能力的同时有效保护客户端数据隐私。

**[Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)**

:   提出 FedPoisonMIA，一种基于角度偏差最大化的联邦学习投毒成员推理攻击，同时提出 Angular Trimmed-mean (ATM) 防御机制，通过角度距离过滤恶意梯度。

**[FRET: Feature Redundancy Elimination for Test Time Adaptation](fret_feature_redundancy_elimination_for_test_time_adaptation.md)**

:   本文提出特征冗余消除（FRET）作为测试时自适应（TTA）的新视角，发现分布偏移时嵌入特征冗余度显著增加，并设计了S-FRET（直接最小化冗余分数）和G-FRET（基于GCN的注意力-冗余分解+双层优化）两种方法，G-FRET在多种架构和数据集上达到SOTA性能。

**[Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)**

:   本文提出Geminio，首个利用视觉语言模型（VLM）实现自然语言引导的梯度反转攻击（GIA），使联邦学习中的恶意服务器可以用自然语言描述想要窃取的数据类型，并从大batch梯度中精准定位和重建匹配的隐私样本，同时不影响正常的FL模型训练。

**[LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)**

:   本文提出LoRA-FAIR方法，通过在服务器端引入残差校正项 $\Delta\mathbf{B}$ 来同时解决联邦学习+LoRA微调中的服务器端聚合偏差和客户端初始化滞后两大挑战，在ViT和MLP-Mixer模型上一致超越现有联邦微调方法，且不增加通信开销。

**[Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack](mind_the_cost_of_scaffold_benign_clients_may_even_become_accomplices_of_backdoor.md)**

:   提出 BadSFL，首个针对 Scaffold 联邦学习算法的后门攻击方法，通过篡改控制变量（control variate）将良性客户端变为"帮凶"，结合 GAN 数据增强和预测全局模型收敛方向的优化策略，在 non-IID 场景下实现了攻击停止后仍持续 60+ 轮的后门效果，持久性是基线方法的 3 倍。

**[Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)**

:   提出Oasis方法，仅需输入图像（无需任何文本提示）即可诱导MLLM自回归生成高质量多模态指令跟随数据，配合精细的指令质量控制机制，合成50万数据给LLaVA-NeXT带来平均3.1%的全面性能提升，且超越其他合成方法。

**[Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)**

:   提出 SARDFQ 方法解决 ViT 无数据量化（DFQ）中合成图像的**语义失真**和**语义不足**问题，通过注意力先验对齐（APA）引导合成图像的注意力模式与真实图像对齐，通过多语义增强（MSR）优化局部 patch 丰富图像语义，在 ImageNet W4A4 ViT-B 上提升 15.52% Top-1 准确率。

**[SpecGuard: Spectral Projection-based Advanced Invisible Watermarking](specguard_spectral_projection-based_advanced_invisible_watermarking.md)**

:   SpecGuard 提出将水印信息嵌入到小波分解后的高频子带的频谱域中（通过 FFT 近似的频谱投影），编码端用强度因子增强鲁棒性，解码端利用 Parseval 定理设计可学习阈值进行比特恢复，在保持高图像质量（PSNR>42dB）的同时实现了对畸变、再生成和对抗攻击的全面鲁棒性，超越了现有 SOTA 方法。

**[Staining and Locking Computer Vision Models without Retraining](staining_and_locking_computer_vision_models_without_retraining.md)**

:   本文提出了无需重训练或微调即可对预训练视觉模型进行"染色"（水印嵌入）和"锁定"（使用保护）的新算法，通过直接修改少量权重植入高选择性检测神经元，并提供了可计算的误报率理论保证，在图像分类和目标检测模型上验证了有效性。

**[SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)**

:   提出 SynFER，一个基于扩散模型的面部表情合成框架，通过文本描述 + 面部动作单元 (FAU) 的双重控制实现细粒度表情生成，并引入 FERAnno 标签校准器确保标注可靠性，在自监督、监督、零样本和少样本四种学习范式下均证明合成数据对 FER 的有效性。

**[Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)**

:   揭示了逆向对抗攻击（inverse adversarial attack）在对抗训练中导致模型注意力偏移至背景特征的虚假相关性问题，提出 DHAT 方法通过去偏高置信度 logit 正则化（DHLR）和前景 logit 正交增强（FLOE）两个组件来消除这种偏差，在 CIFAR-10/100 和 ImageNet-1K 上取得了 SOTA 的对抗鲁棒性。

**[Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)**

:   本文提出FakeSTormer，一个细粒度的生成式深度伪造视频检测框架，通过多任务学习同时建模时间和空间脆弱性区域，配合自混合视频（SBV）数据合成策略生成高质量伪造样本，仅用真实数据训练即可在多个跨数据集基准上达到SOTA泛化性能。
