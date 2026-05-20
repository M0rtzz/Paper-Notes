---
title: >-
  AAAI2026 人体理解方向15篇论文解读
description: >-
  15篇AAAI2026的人体理解方向论文解读，涵盖人脸/视线、人体姿态、推理、情感分析、重识别、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "人体理解"
  - "论文解读"
  - "论文笔记"
  - "人脸/视线"
  - "人体姿态"
  - "推理"
  - "情感分析"
  - "重识别"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🤖 AAAI2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/human_understanding/index.md) · [💬 ACL2026 (1)](../../ACL2026/human_understanding/index.md) · [📷 CVPR2026 (53)](../../CVPR2026/human_understanding/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/human_understanding/index.md) · [🧠 NeurIPS2025 (18)](../../NeurIPS2025/human_understanding/index.md) · [📹 ICCV2025 (38)](../../ICCV2025/human_understanding/index.md)

🔥 **高频主题：** 人脸/视线 ×4 · 人体姿态 ×3

**[AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification](ahan_asymmetric_hierarchical_attention_network_for_identical.md)**

:   针对同卵双胞胎人脸验证这一极端细粒度识别挑战，提出 AHAN 多流架构，通过层次交叉注意力 (HCA) 对语义面部区域做多尺度分析、面部不对称注意力模块 (FAAM) 捕获左右脸差异签名、以及双胞胎感知配对交叉注意力 (TA-PWCA) 训练正则化，在 ND_TWIN 数据集上将双胞胎验证精度从 88.9% 提升至 92.3%（+3.4%）。

**[CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)**

:   首次利用 CLIP 提取面部细粒度语义属性嵌入来辅助人脸模板反演（FTI），通过跨模态特征交互网络将泄露模板与属性嵌入融合并投影到 StyleGAN 潜空间，生成身份一致且属性细节更丰富的人脸图像，在识别准确率、属性相似度和跨模型攻击迁移性上均超越 SOTA。

**[CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)**

:   提出 CoordAR，将单参考视图 6D 位姿估计中的 3D-3D 对应关系建模为离散 token 的自回归生成问题，通过坐标图 token 化、模态解耦编码和自回归 Transformer 解码器，在多个基准上显著超越现有单视图方法，并对对称、遮挡等挑战场景展现强鲁棒性。

**[Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)**

:   提出 Facial-R1，一个三阶段对齐训练框架（SFT → RL → 数据合成），通过将 AU 和情绪标签作为可验证奖励信号来对齐 VLM 的推理过程与情绪识别结果，在 8 个基准上达到 SOTA，并构建了 FEA-20K 数据集。

**[Generating Attribute-Aware Human Motions from Textual Prompt](generating_attribute-aware_human_motions_from_textual_prompt.md)**

:   提出 AttrMoGen 框架，通过基于结构因果模型（SCM）的因果信息瓶颈将动作语义与人体属性（年龄、性别等）解耦，生成属性感知的人体运动，并构建了首个包含广泛属性标注的大规模文本-运动数据集 HumanAttr。

**[Improving Sparse IMU-based Motion Capture with Motion Label Smoothing](improving_sparse_imu-based_motion_capture_with_motion_label_smoothing.md)**

:   提出 Motion Label Smoothing，将经典 label smoothing 从分类任务适配到稀疏IMU运动捕捉中，通过融合骨骼结构感知的Perlin噪声作为平滑标签，在不修改模型架构的前提下以即插即用方式提升三种SOTA方法在四个数据集上的精度，GlobalPose在TotalCapture上SIP误差降低20.41%。

**[mmPred: Radar-based Human Motion Prediction in the Dark](mmpred_radar-based_human_motion_prediction_in_the_dark.md)**

:   首次将毫米波雷达引入人体运动预测(HMP)任务，提出mmPred——基于扩散模型的框架，通过双域历史运动表示（时域姿态细化TPR + 频域主导运动FDM）和全局骨骼关系Transformer(GST)，有效抑制雷达特有的噪声和时序不一致性，在mmBody和mm-Fi数据集上分别超越SOTA方法8.6%和22%。

**[Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)**

:   针对无监督可见光-红外行人重识别（USVI-ReID）中跨模态关联不可靠的核心问题，提出模态感知的 Jaccard 距离修正和"分裂-对比"不变性学习策略，通过消除模态偏差实现可靠的全局跨模态聚类和特征对齐，在 SYSU-MM01 和 RegDB 上达到 SOTA。

**[MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network](mvgd-net_a_novel_motion-aware_video_glass_surface_detection_network.md)**

:   基于"玻璃表面上反射/透射层物体的运动速度与非玻璃区域不一致"的物理观察，提出 MVGD-Net，通过光流运动线索引导视频中玻璃表面检测，包含跨尺度多模态融合（CMFM）、历史引导注意力（HGAM）、时序交叉注意力（TCAM）和时空解码器（TSD）四个核心模块，并构建了包含 312 视频 19,268 帧的大规模数据集 MVGD-D。

**[New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)**

:   提出 SeqEMG-GAN，一种基于手部关节角度序列驱动的条件对抗生成框架，通过角度编码器、双层上下文编码器（含新颖 Ang2Gist 单元）、深度卷积生成器和多视角判别器的联合设计，从关节运动学轨迹合成高保真 EMG 信号，实现对未见手势的零样本生成，合成数据与真实数据混合训练将分类精度从 57.77% 提升至 60.53%。

**[Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)**

:   针对 3D 人体姿态估计在线测试时自适应中的误差累积问题，提出基于运动离散化（无监督聚类获得锚运动集）+ 自回放机制 + 软重置策略的解决方案，使模型能在长时间持续适应中稳健利用个人形态和习惯性运动特征，在 Ego-Exo4D 和 3DPW 上超越所有现有在线 TTA 方法。

**[SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)**

:   提出Salient Orientation Symbolic (SOS) script——基于Labanotation启发的可编程符号化运动表示框架，通过时序约束的凝聚聚类提取关键帧显著性，结合SMS数据增强和梯度优化的SOSControl框架实现对身体部位朝向和运动时序的精确控制，在HumanML3D上SOS-Acc达0.988且FID仅3.892。

**[Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)**

:   提出ST-MoE框架，首次将混合专家模型（MoE）与双向时空Mamba相结合用于多人运动预测，通过四种异构时空专家灵活捕获复杂时空依赖，实现SOTA精度的同时减少41.38%参数量，训练加速3.6倍。

**[Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion](streaming_generation_of_co-speech_gestures_via_accelerated_rolling_diffusion.md)**

:   提出基于 Rolling Diffusion 的流式共语手势生成框架，通过结构化渐进噪声调度将任意扩散模型转化为流式手势生成器，并引入 Rolling Diffusion Ladder Acceleration (RDLA) 实现最高 4× 加速（200 FPS），在 ZEGGS 和 BEAT 基准上全面超越基线。

**[VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)**

:   提出 VPHO，一个联合视觉和物理线索的手-物体姿态估计框架，通过力预测模块学习 3D 物理线索，并设计两阶段候选姿态聚合策略（视觉引导 + 物理引导），在保持视觉一致性的同时实现物理合理性，在 DexYCB 和 HO3D 两个基准上同时达到姿态精度和物理合理性的 SOTA。
