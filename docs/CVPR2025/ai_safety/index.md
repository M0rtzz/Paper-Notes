---
title: >-
  CVPR2025 AI安全方向 22篇论文解读
description: >-
  22篇CVPR2025 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**📷 CVPR2025** · **22** 篇论文解读

**[A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)**

:   提出FedRDN——一种极其简单的联邦学习数据增强方法，在训练时随机使用其他客户端的通道级均值/标准差做数据归一化（而非固定用本地统计），仅需几行代码即可显著缓解特征分布偏移问题，在多种FL方法上一致提升性能。

**[Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior](data-free_universal_adversarial_perturbation_with_pseudo-semantic_prior.md)**

:   提出 PSP-UAP，一种无需训练数据的通用对抗扰动生成方法，通过从 UAP 自身提取伪语义先验、输入变换增强和样本重加权策略，在白盒平均 89.95% 愚弄率、黑盒也大幅超越现有方法，且无需任何训练数据。

**[DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging](deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md)**

:   提出 DEAL（Data-Efficient Adversarial Learning），一种仅需 50 张清晰红外图像训练的对抗学习框架，通过动态对抗退化合成和双通道交互网络（Scale Transform + Spiking Neurons），以 0.96M 超轻量参数同时处理条纹噪声、低分辨率和低对比度三种红外退化。

**[DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders.md)**

**[Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)**

:   提出 AlignIns 防御方法，通过双粒度方向对齐检测（全局方向 + 细粒度符号分析）识别联邦学习中的恶意模型更新，在 IID 和 non-IID 设置下均优于现有防御方法。

**[Dynamic Integration of Task-Specific Adapters for Class Incremental Learning](dynamic_integration_of_task-specific_adapters_for_class_incremental_learning.md)**

:   通过动态集成任务特定适配器实现类增量学习，每个任务训练轻量适配器，推理时动态选择和组合相关适配器

**[ESC: Erasing Space Concept for Knowledge Deletion](esc_erasing_space_concept_for_knowledge_deletion.md)**

:   提出 ESC（Erasing Space Concept），通过 SVD 分解待遗忘数据的特征空间并移除主成分方向，实现训练无关的特征级知识删除，首次定义了"知识删除"（Knowledge Deletion）任务并提出 Knowledge Retention Score 评估特征级遗忘效果。

**[FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors](fedawa_adaptive_optimization_of_aggregation_weights_in_federated_learning_using_.md)**

:   提出 FedAWA，受任务算术（task arithmetic）启发，用客户端向量（本地参数与全局参数的差值）来自适应优化联邦学习中的聚合权重——与全局优化方向一致的客户端获得更高权重，在 non-IID 场景下稳定提升 FedAvg 1-4 个点。

**[Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)**

:   在联邦学习中通过从局部协方差矩阵精确重建全局协方差来获取全局嵌入分布的几何形状，沿全局主方向生成增强样本本地化全局分布信息，在 CIFAR-100 极端异质场景（β=0.01）下提升 17 个百分点。

**[Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning](gradient_inversion_attacks_on_parameter-efficient_fine-tuning.md)**

:   首次证明 Adapter-based PEFT 在联邦学习中不是隐私安全的——恶意服务器可以将预训练模型设计为恒等映射使 patch embedding 原样传播到 adapter 层，从 adapter 梯度中解析式恢复训练图像（CIFAR-100 SSIM 0.88）。

**[Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)**

:   本文首次研究了联邦学习中非合作多标签后门攻击(MBA)场景，揭示了现有单标签后门攻击方法扩展到多标签场景时因构建相似的分布外(OOD)映射而导致攻击者间相互排斥的内在缺陷，提出 Mirage 方法通过构建分布内(ID)后门映射，使多个攻击者可以独立且持久地植入后门，平均攻击成功率超过97%且在900轮后仍保持90%以上。

**[INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](invisible_backdoor_attack_against_self-supervised_learning.md)**

:   提出 INACTIVE，首个对自监督学习（SSL）有效的不可见后门攻击——通过在 HSV/HSL 色彩空间中设计触发器以逃离 SSL 数据增强的分布空间，实现 99.09% 平均攻击成功率，同时保持 SSIM 0.9763/PSNR 41.07dB 的高隐蔽性，抵抗 7 种防御方法。

**[Lyapunov Stable Graph Neural Flow](lyapunov_stable_graph_neural_flow.md)**

:   将 Lyapunov 稳定性理论（整数阶和分数阶）与图神经流集成，通过可学习 Lyapunov 函数和投影机制将 GNN 特征动态约束在稳定空间中，首次为图神经流提供可证明的对抗鲁棒性保证，且与对抗训练正交可叠加。

**[Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)**

:   本文提出了一种基于查询更新模式(而非输入模式)的黑盒对抗攻击检测框架 GWAD，引入 Delta Similarity 指标来捕获基于查询的攻击中零阶优化的固有模式，在8种SOTA攻击(包括自适应攻击OARS)上实现了接近100%的检测率且误报率极低，显著优于现有的状态化防御方法。

**[MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)**

:   提出MOS Attack框架，将对抗攻击建模为多目标集合优化问题，结合smooth max/min近似实现多损失函数联合优化，并自动发现损失函数间的协同模式，在CIFAR-10和ImageNet上超越现有SOTA单目标攻击和集成攻击。

**[Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)**

:   Neural Gate 发现 LVLM 中隐私相关神经元具有强跨样本不一致性——仅约 10% 的神经元一致性编码隐私信号。基于此发现，提出神经元级梯度门控编辑：仅对强一致性隐私神经元施加梯度更新，在 MiniGPT 上将 Safety EtA 从 0.48 提升至 0.89，同时 Utility 保持不降。

**[NoT: Federated Unlearning via Weight Negation](not_federated_unlearning_via_weight_negation.md)**

:   提出 NoT（Not），通过将选定层的权重取反（乘 -1）再在保留数据上微调来实现联邦遗忘，无需访问遗忘数据或存储额外信息，在 CIFAR-10 CNN 上与从头重训的差距仅 0.29%，计算量减少约一半。

**[PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection](psbd_prediction_shift_uncertainty_unlocks_backdoor_detection.md)**

:   提出 PSBD 方法，发现被植入后门的模型在推理时开启 dropout 后，干净数据的预测会偏移向目标类别而后门数据预测保持稳定（Prediction Shift 现象），基于此设计 Prediction Shift Uncertainty (PSU) 指标实现 SOTA 后门训练数据检测。

**[Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)**

:   提出 IFDL-VLM，揭示 VLM 先验对伪造检测/定位几乎无益，通过将检测/定位与语言解释解耦的两阶段框架，用 ViT+SAM 专家模型做检测定位、再将定位 mask 作为辅助输入增强 VLM 训练以生成可解释文字说明。

**[Split Adaptation for Pre-trained Vision Transformers](split_adaptation_for_pre-trained_vision_transformers.md)**

:   提出 Split Adaptation，将预训练 ViT 分为客户端前端（量化+噪声）和服务端后端，通过 OOD 增强量化、Hilbert 变换增强和 patch 级检索增强实现少样本适应，在保护数据隐私和模型知识产权的同时在 CIFAR-100 5-shot 上达到 81.98%，计算量仅为线性探测的 4%。

**[Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)**

:   提出动态超网训练策略（CaLR + MS），通过复杂度感知的学习率调度解决子网训练不公平问题，以及动量分离技术缓解梯度噪声问题，以极低额外开销显著提升 N-shot NAS 的搜索性能。

**[Towards Source-Free Machine Unlearning](towards_source-free_machine_unlearning.md)**

:   本文提出了一种无源机器遗忘（Source-Free Machine Unlearning）算法，在无法获取原始训练数据的条件下，通过近似估计保留数据的 Hessian 矩阵（仅使用待遗忘数据和训练好的模型），实现了对线性和混合线性分类器的高效遗忘，并提供了严格的理论上界保证。
