---
title: >-
  CVPR2025 机器人/具身智能方向 14篇论文解读
description: >-
  14篇CVPR2025 机器人/具身智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**📷 CVPR2025** · 共 **14** 篇

**[3D-Mvp 3D Multiview Pretraining For Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)**

:   提出3D-MVP，将Masked Autoencoder预训练从2D扩展到3D多视角设定——在Objaverse的200K个3D物体上预训练RVT的多视角Transformer编码器，下游微调后在RLBench上平均成功率从62.9%提升到67.5%，在COLOSSEUM上显著提升对纹理、大小、光照等环境变化的鲁棒性。

**[A Data-Centric Revisit Of Pre-Trained Vision Models For Robot Learning](a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)**

:   通过系统评估发现DINO/iBOT在机器人任务上优于MAE但在非物体中心(NOC)数据上性能退化，原因是丧失了物体中心表示能力。提出SlotMIM方法，通过语义瓶颈（减少原型数量促进objectness涌现）和跨视图一致性正则+slot级对比学习，使模型在NOC数据上也能学到物体中心表示，仅用241K样本即超越用>1M样本的MVP/VC-1。

**[Asap Advancing Semantic Alignment Promotes Multi-Modal Manipulation De](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)**

:   提出ASAP框架，通过大模型辅助对齐(LMA)、篡改引导交叉注意力(MGCA)和补丁篡改建模(PMM)三个核心模块，系统性地推进图文语义对齐以提升多模态篡改检测与定位性能——在DGM4基准上AUC达94.38%，文本定位F1达76.52%，显著超越现有方法。

**[Asap Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting An](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)**

**[Chapter-Llama Efficient Chaptering In Hour-Long Videos With Llms](chapter-llama_efficient_chaptering_in_hour-long_videos_with_llms.md)**

**[Drawer Digital Reconstruction And Articulation With Environment Realism](drawer_digital_reconstruction_and_articulation_with_environment_realism.md)**

:   提出 DRAWER 框架，从静态场景视频自动构建可交互数字孪生，结合 SDF + 高斯泼溅双场景表示实现高保真渲染和精细几何，支持铰接体识别与仿真、Unreal Engine 游戏创建、以及 real-to-sim-to-real 机器人策略迁移。

**[Expert Pyramid Tuning Efficient Parameter Fine-Tuning For Expertise-Driven Task ](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   提出 Expert Pyramid Tuning (EPT)，将计算机视觉中的多尺度特征金字塔思想引入 LoRA-based MoE，通过共享元知识子空间 + 反卷积金字塔投影机制构建不同粒度的专家，实现更高效的多任务参数微调。

**[Influence Malleability In Linearized Attention Dual Implications Of Non-Converge](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

**[Language-Grounded Decoupled Action Representation For Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)**

:   提出 LaDA，将 7-DoF 机器人动作解耦为平移/旋转/夹爪三类运动原语并与语言语义建立对应，通过软标签对比学习和自适应损失加权，以 1.3B 参数在 LIBERO 上达到 93.6% 平均成功率。

**[One Token Two Fates A Unified Framework Via Vision Token Manipulation Against Ml](one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)**

:   提出首个统一的训练无关MLLM幻觉缓解框架，围绕vision token的双重角色——增强(SVC)与抑制(CRC)——在隐表示层协同操作，在LLaVA-1.5上POPE准确率提升约2%，仅增加1.06×推理延迟。

**[Panoaffordancenet Towards Holistic Affordance Grounding In 360 Indoor Environmen](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出PanoAffordanceNet——首个360°全景affordance grounding框架，通过失真感知频谱调制器(DASM)处理ERP纬度依赖畸变、全球面致密化头(OSDH)恢复稀疏激活为拓扑连续区域、多层级训练目标抑制语义漂移，并构建首个全景affordance数据集360-AGD，全面超越现有方法。

**[Sapave Towards Active Perception And Manipulation In Vision-Language-Action Mode](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)**

:   SaPaVe 提出了一种端到端的主动操作框架，通过解耦相机运动和操作动作的 action space，采用自底向上的两阶段训练策略（先学语义相机控制，再联合优化），在 200K 语义相机运动数据集上训练主动感知先验，配合 3D 几何感知模块增强视角变化下的执行鲁棒性，在真实世界任务中比 GR00T N1 和 $\pi_0$ 分别高 31.25% 和 40% 成功率。

**[Sortscrews A Dataset And Baseline For Real-Time Screw Classification](sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)**

:   提出SortScrews数据集——一个包含560张512×512 RGB图像、覆盖6类螺丝的工业分类数据集，配套可复用的数据采集流水线，并以迁移学习的EfficientNet-B0和ResNet-18作为基线，ResNet-18在该数据集上达到96.4%验证准确率。

**[Tinynav End-To-End Tinyml For Real-Time Autonomous Navigation On Microcontroller](tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)**

:   在 ESP32 微控制器上部署端到端量化 CNN，仅用 23k 参数和 ToF 深度相机实现 30ms 延迟的实时自主导航。
