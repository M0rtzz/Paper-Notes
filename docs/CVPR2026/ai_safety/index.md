---
title: >-
  CVPR2026 AI 安全方向22篇论文解读
description: >-
  22篇CVPR2026的 AI 安全方向论文解读，涵盖对抗鲁棒、联邦学习、水印/隐写、机器人、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**📷 CVPR2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (2)](../../ACL2026/ai_safety/index.md) · [🔬 ICLR2026 (26)](../../ICLR2026/ai_safety/index.md) · [🤖 AAAI2026 (44)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (72)](../../NeurIPS2025/ai_safety/index.md) · [📹 ICCV2025 (21)](../../ICCV2025/ai_safety/index.md) · [🧪 ICML2025 (33)](../../ICML2025/ai_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×11 · 联邦学习 ×9 · 水印/隐写 ×4 · 机器人 ×3 · 多模态 ×3

**[A Unified Perspective on Adversarial Membership Manipulation in Vision Models](a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)**

:   首次揭示视觉模型成员推断攻击(MIA)面临的对抗性成员操纵漏洞——不可感知扰动可将非成员伪造为成员欺骗审计，发现伪造成员的梯度范数塌缩特征签名，并提出基于梯度几何的检测策略和对抗鲁棒推断框架。

**[All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference](all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co.md)**

:   提出 Pseudo-Random Bayesian Inference (PRBI) 框架，在**所有车辆均不可信**的协同感知场景中，利用帧间时序一致性作为自参考信号，通过伪随机分组 + 贝叶斯推断，仅需平均 2.5 次验证/帧即可高效识别并排除恶意车辆，检测精度恢复至攻击前的 79.4%–86.9%。

**[ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)**

:   提出基于视觉 token 聚类的水印方案 ClusterMark，将 KGW 风格的 LLM 水印适配到自回归图像生成器，通过将相似 token 分到同一绿/红集合来显著提升水印在图像扰动下的鲁棒性，同时保持图像质量。

**[Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)**

:   提出 FOUL 框架，通过"学习阶段解耦因果/非因果特征 + 遗忘阶段服务器端梯度冲突匹配"两阶段策略，在不访问客户端数据的前提下实现高效且低通信开销的联邦遗忘。

**[AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)**

:   提出 AdvMark 两阶段解耦防御框架：Stage 1 Encoder Adversarial Training（EAT）将水印图像移入 non-attackable 区域抵御对抗攻击，Stage 2 直接图像优化抵御失真+再生攻击并保留对抗鲁棒性，在 9 种水印方法 ×10 种攻击上分别提升失真/再生/对抗准确率 29%/33%/46%，且图像质量最优。

**[Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)**

:   提出 F²DC 框架，通过域特征解耦器（DFD）和域特征校正器（DFC）将联邦学习中客户端的局部特征分离为域鲁棒特征和域相关特征，并对域相关特征进行校准以挽救被丢弃的类别信息，配合域感知聚合策略，在三个多域数据集上一致超越 SOTA。

**[FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)**

:   提出 FecalFed 隐私保护联邦学习框架，首先通过双哈希去重清理公开禽类粪便数据集中 46.89% 的重复污染并发布 8,770 张清洁基准数据集 poultry-fecal-fl，随后在 Dirichlet α=0.5 的高度非 IID 条件下验证：FedAdam + Swin-Small 可将单农场训练崩溃的 64.86% 准确率恢复至 90.31%，仅比中心化上界 95.10% 低 4.79%；边缘优化的 Swin-Tiny（28M 参数）仍达 89.74%，为农场端部署提供高效可行方案。

**[FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)**

:   提出 FedAFD 框架，通过双层对抗对齐、粒度感知特征融合和相似度引导的集成蒸馏三阶段设计，在多模态联邦学习中同时提升异构客户端和服务器的模型性能。

**[FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)**

:   提出域感知原型联邦学习框架 FedDAP，通过构建域特定全局原型和双重原型对齐策略（域内对齐 + 跨域对比），解决联邦学习中客户端数据域偏移导致的全局模型性能退化问题。

**[Federated Active Learning Under Extreme Non-IID and Global Class Imbalance](federated_active_learning_extreme_noniid.md)**

:   系统分析全局类不平衡与客户端异构性对联邦主动学习中 query model 选择的影响，归纳出3个核心 Observation，据此提出 FairFAL——自适应选择 query model + 原型引导伪标签 + 两阶段不确定性-多样性平衡采样的类公平 FAL 框架，在5个基准数据集上一致超越所有基线。

**[FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)**

:   提出 FedRE 框架，通过"纠缠表示"（entangled representation）——将每个客户端的所有局部表示用归一化随机权重聚合为单一跨类别表示，实现模型异构联邦学习中性能、隐私保护和通信开销的三方平衡。

**[Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)**

:   提出首个跨范式（密度图 + 点回归）对抗攻击框架 CrowdGen，利用轻量级 UNet 生成器和多任务损失（logit 抑制 + 密度抑制 + GradCAM 引导 + 频域约束），在保持视觉隐蔽性（~19dB PSNR）的同时实现对七个 SOTA 人群计数模型的高迁移率（TR 最高 1.69），攻击 MAE 平均提升 7 倍。

**[LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories](logitdynamics_vit_error_detection.md)**

:   LogitDynamics 通过在 ViT 各层附加轻量分类头，提取层间 logit 轨迹和 top-K 竞争动态特征，训练线性探针来预测模型错误，在跨数据集泛化上优于现有方法。

**[Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](mcsd_uncertainty_estimation.md)**

:   将随机深度（Stochastic Depth）正式连接到贝叶斯变分推理框架，提出 Monte Carlo Stochastic Depth (MCSD) 作为不确定性估计方法，并在 YOLO、RT-DETR 等现代检测器上进行首次系统基准测试，证明其在校准和不确定性排名上与 MC Dropout 竞争力强。

**[One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)**

:   O2MAG 提出一种无需训练的少样本异常生成方法，通过三分支扩散过程中的自注意力嫁接(TriAG)从单张参考异常图像合成更多逼真异常，配合异常引导优化(AGO)对齐文本语义和异常引导增强(DAE)确保掩码区域完整填充，在 MVTec-AD 下游异常检测任务中显著优于现有方法。

**[ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)**

:   提出 ProxyFL 框架，利用分类器权重作为统一代理 (proxy) 同时缓解联邦半监督学习中的外部异质性（跨客户端分布差异）和内部异质性（标注/未标注数据分布不匹配），在多个数据集上显著超越现有 FSSL 方法。

**[RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)**

:   提出 RecoverMark，一个将人脸内容本身作为水印嵌入背景的鲁棒水印框架，同时实现篡改区域定位、原始内容恢复和版权验证，在水印移除攻击下仍保持有效。

**[SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)**

:   提出 SubFLOT 框架，在服务器端利用最优传输（Optimal Transport）将全局模型的参数分布与客户端历史模型对齐，实现无需访问原始数据的个性化剪枝，并通过自适应正则化抑制剪枝导致的参数偏移，在多个数据集上大幅超越现有联邦剪枝方法。

**[TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)**

:   提出 TIACam 框架，通过可学习自动增强器模拟相机失真、文本锚定跨模态对抗训练学习不变特征、零水印头在特征空间绑定消息，实现无需修改图像像素的相机鲁棒零水印方案，在屏幕翻拍/打印翻拍/截图三种真实场景下均达到 SOTA 提取精度。

**[Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)**

:   提出 SADCA（语义增强动态对比攻击），通过动态对比交互机制和语义增强模块，迭代地破坏对抗图像与文本之间的跨模态语义一致性，显著提升对视觉语言预训练模型（VLP）的对抗可迁移性，在跨模型和跨任务攻击中均超越现有 SOTA 方法。

**[Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)**

:   提出 Tutor-Student 强化学习（TSRL）框架，将深度伪造检测器的训练过程建模为马尔可夫决策过程，由"导师"（PPO agent）根据每个样本的视觉特征和历史学习动态（EMA 损失、遗忘次数）动态分配损失权重，通过"状态变化"奖励信号引导"学生"（检测器）优先学习高价值样本，在跨数据集和跨方法评估中显著提升泛化能力。

**[When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)**

:   提出 UPA-RFAS 框架，学习一个单一物理对抗补丁，通过特征空间偏移、注意力劫持和语义错位三管齐下，实现对 VLA 机器人策略的通用、可迁移黑盒攻击。
