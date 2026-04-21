---
title: >-
  ECCV2024 AI安全方向 12篇论文解读
description: >-
  12篇ECCV2024 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**🎞️ ECCV2024** · **12** 篇论文解读

**[Any Target Can Be Offense: Adversarial Example Generation via Generalized Latent Infection](any_target_can_be_offense_adversarial_example_generation_via_generalized_latent_.md)**

:   提出 GAKer，首个可泛化到未知目标类别的定向对抗攻击生成器，通过在 UNet 中间层注入目标特征（latent infection）+ 余弦距离损失替代交叉熵实现类别无关训练，在未知类上的攻击成功率比 HGN 高 14.13%。

**[CLIP-Guided Generative Networks for Transferable Targeted Adversarial Attacks](clip-guided_generative_networks_for_transferable_targeted_adversarial_attacks.md)**

:   提出 CGNC，利用 CLIP 文本编码器为条件生成网络注入目标类别语义信息，结合交叉注意力模块和 masked fine-tuning，大幅提升多目标/单目标定向对抗攻击的黑盒迁移成功率。

**[Event Trojan: Asynchronous Event-Based Backdoor Attacks](event_trojan_asynchronous_event-based_backdoor_attacks.md)**

:   提出 Event Trojan 框架，首次研究直接在异步事件数据流中注入后门触发器（immutable trigger 和 mutable trigger），揭示了事件相机视觉任务面临的后门攻击安全风险。

**[Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)**

:   本文提出Self-Driven Fisher Calibration（SDFC），利用Fisher信息度量参数对不同分布的重要程度差异，在异质联邦学习场景中有效区分恶意后门客户端并进行参数校准，突破了现有防御方法依赖数据同质性和恶意节点少数假设的局限。

**[Event Trojan: Asynchronous Event-based Backdoor Attacks](genq_quantization_in_low_data_regimes_with_generative_synthetic_data.md)**

:   提出 Event Trojan 框架，首次针对异步事件数据流设计后门攻击方法，包含不可变触发器和可变触发器两种模式，直接在事件流层面注入恶意事件实现隐蔽高效的后门攻击。

**[Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)**

:   本文提出 CLIP-IFDL，一种基于 CLIP 的图像篡改检测与定位模型，通过实例感知的双流提示学习和伪造增强噪声适配器来弥补 CLIP 在篡改检测领域的提示缺失和伪造感知不足问题，将 CLIP 的开放世界泛化能力迁移到篡改检测任务中。

**[Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective](preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)**

:   从双层优化视角分析快速对抗训练中灾难性过拟合的成因，提出 FGSM-PCO 方法，通过自适应融合历史与当前对抗样本并配合定制正则化损失，有效防止并纠正内层优化崩溃。

**[Resilience of Entropy Model in Distributed Neural Networks](resilience_of_entropy_model_in_distributed_neural_networks.md)**

:   首次系统研究分布式 DNN 中熵编码模型在有意干扰（对抗攻击）和无意干扰（天气变化、运动模糊等）下的鲁棒性，发现熵模型学习的压缩特征与分类特征截然不同，并提出基于目标感知全变差去噪的防御方法，可将攻击后的传输开销降低至低于干净数据水平，准确率仅下降约 2%。

**[SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)**

:   提出 SkyMask，利用参数级可学习二值掩码在服务器端检测恶意客户端模型更新，实现攻击无关的鲁棒联邦学习，在恶意客户端占比高达 80% 时仍能有效防御。

**[Towards Multi-modal Transformers in Federated Learning](towards_multi-modal_transformers_in_federated_learning.md)**

:   提出 FedCola 框架，通过互补本地训练和协作聚合两个策略，在联邦学习中实现多模态 Transformer 的跨模态知识迁移，无需公共数据即可弥合单模态与多模态客户端之间的差距。

**[Towards Multi-modal Transformers in Federated Learning](towards_multimodal_transformers_in_federated_learning.md)**

:   首次探索Transformer架构在转移式多模态联邦学习中的应用，提出FedCola框架，通过互补式本地训练（利用跨模态Transformer blocks）和协作式服务器聚合（选择性聚合self-attention层），在保护数据隐私的前提下有效训练多模态Transformer。

**[Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)**

:   本文揭示了随机神经网络（SNNs）在联邦学习中同样容易遭受梯度反演攻击，提出 ISG 方法通过将 SNN 的随机训练过程等价为传统 NN 训练的变体来重建训练数据，并引入特征约束策略提升重建保真度。
