---
title: >-
  CVPR2025 具身智能方向 29篇论文解读
description: >-
  29篇CVPR2025 具身智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 具身智能

**📷 CVPR2025** · **29** 篇论文解读

**[3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)**

:   提出3D-MVP，将Masked Autoencoder预训练从2D扩展到3D多视角设定——在Objaverse的200K个3D物体上预训练RVT的多视角Transformer编码器，下游微调后在RLBench上平均成功率从62.9%提升到67.5%，在COLOSSEUM上显著提升对纹理、大小、光照等环境变化的鲁棒性。

**[A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)**

:   通过系统评估发现DINO/iBOT在机器人任务上优于MAE但在非物体中心(NOC)数据上性能退化，原因是丧失了物体中心表示能力。提出SlotMIM方法，通过语义瓶颈（减少原型数量促进objectness涌现）和跨视图一致性正则+slot级对比学习，使模型在NOC数据上也能学到物体中心表示，仅用241K样本即超越用>1M样本的MVP/VC-1。

**[ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)**

:   提出ASAP框架，通过大模型辅助对齐(LMA)、篡改引导交叉注意力(MGCA)和补丁篡改建模(PMM)三个核心模块，系统性地推进图文语义对齐以提升多模态篡改检测与定位性能——在DGM4基准上AUC达94.38%，文本定位F1达76.52%，显著超越现有方法。

**[ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_detecting_an.md)**

**[Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs](chapter-llama_efficient_chaptering_in_hour-long_videos_with_llms.md)**

**[DRAWER: Digital Reconstruction and Articulation with Environment Realism](drawer_digital_reconstruction_and_articulation_with_environment_realism.md)**

:   提出 DRAWER 框架，从静态场景视频自动构建可交互数字孪生，结合 SDF + 高斯泼溅双场景表示实现高保真渲染和精细几何，支持铰接体识别与仿真、Unreal Engine 游戏创建、以及 real-to-sim-to-real 机器人策略迁移。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   提出 Expert Pyramid Tuning (EPT)，将计算机视觉中的多尺度特征金字塔思想引入 LoRA-based MoE，通过共享元知识子空间 + 反卷积金字塔投影机制构建不同粒度的专家，实现更高效的多任务参数微调。

**[Foundations of the Theory of Performance-Based Ranking](foundations_of_the_theory_of_performance-based_ranking.md)**

:   建立基于概率测度和序理论的性能排名公理化框架，用 6 大支柱（性能测度/偏序/满意度/评价函数/分数/重要性）统一了准确率/精度/召回率/F1 等指标，证明了何种评分函数适合用于排名、何种不适合。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   通过 NTK 框架揭示线性化注意力机制不会收敛到无穷宽 NTK 极限（谱放大效应使 Gram 矩阵条件数立方化，需宽度 $m = \Omega(\kappa^6)$），并引入「影响可塑性」概念量化这一非收敛的双面后果：注意力比 ReLU 网络高 6-9 倍的可塑性既增强了任务适配能力，也加剧了对抗脆弱性。

**[Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)**

:   本文提出 InstructMove，通过从视频中采样帧对并用多模态大模型生成编辑指令来构建大规模真实图像编辑数据集，结合空间条件化策略微调 T2I 模型，在姿态调整、视角变换等非刚性编辑任务上实现了 SOTA 效果。

**[LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)**

:   提出 LaDA，将 7-DoF 机器人动作解耦为平移/旋转/夹爪三类运动原语并与语言语义建立对应，通过软标签对比学习和自适应损失加权，以 1.3B 参数在 LIBERO 上达到 93.6% 平均成功率。

**[Let Humanoids Hike! Integrative Skill Development on Complex Trails](let_humanoids_hike_integrative_skill_development_on_complex_trails.md)**

:   提出 LEGO-H 框架，通过 TC-ViT（时序条件 ViT）统一导航感知和低层运动控制，结合层次潜空间匹配（HLM）从 oracle 策略高效蒸馏，使 Unitree H1 人形机器人在复杂户外山径上达到 68.4% 成功率。

**[Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)**

:   Lift3D提出了一个两阶段框架，先通过任务感知MAE重建深度信息增强2D基础模型的隐式3D感知能力，再通过将3D点云投影到虚拟平面建立与2D位置嵌入的映射关系来直接让2D模型编码点云数据，在MetaWorld上平均成功率达83.9%（超越前SOTA DP3的65.3%达18.6个百分点）。

**[Magma: A Foundation Model for Multimodal AI Agents](magma_a_foundation_model_for_multimodal_ai_agents.md)**

:   Magma 通过在图像上标注可交互区域（Set-of-Mark）和在视频中标注运动轨迹（Trace-of-Mark），将 UI 截图、机器人数据和人类操作视频统一到同一个预训练框架中，使单一模型同时具备多模态理解和跨域动作预测能力，在 UI 导航和机器人操控上均取得 SOTA。

**[Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)**

:   提出 HR-Align 适配范式，利用配对人-机器人视频数据和对比对齐损失，以参数高效的方式弥合人类数据预训练模型与机器人域之间的语义差距，在 20 个仿真任务和 5 个真实任务上平均成功率提升 7%+。

**[MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)**

:   提出 MoManipVLA，将预训练的固定基座 VLA 模型迁移到移动操作场景，通过双层轨迹优化联合规划底盘移动和机械臂轨迹（优化可达性/平滑性/碰撞避免），在 OVMM 基准上达到 66.1% 成功率（+4.2%），仅需 50 条演示即可在真实世界部署。

**[One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination](one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)**

:   提出首个统一的训练无关MLLM幻觉缓解框架，围绕vision token的双重角色——增强(SVC)与抑制(CRC)——在隐表示层协同操作，在LLaVA-1.5上POPE准确率提升约2%，仅增加1.06×推理延迟。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出PanoAffordanceNet——首个360°全景affordance grounding框架，通过失真感知频谱调制器(DASM)处理ERP纬度依赖畸变、全球面致密化头(OSDH)恢复稀疏激活为拓扑连续区域、多层级训练目标抑制语义漂移，并构建首个全景affordance数据集360-AGD，全面超越现有方法。

**[Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction](phoenix_a_motion-based_self-reflection_framework_for_fine-grained_robotic_action.md)**

:   提出 Phoenix 框架，用运动指令作为桥梁连接 MLLM 的高层语义反思和底层机器人动作纠正，通过双过程运动调整机制+运动条件扩散策略实现精细粒度的操作失败恢复，并支持终身学习自我提升。

**[Prof. Robot: Differentiable Robot Rendering without Static and Self-Collisions](prof_robot_differentiable_robot_rendering_without_static_and_self-collisions.md)**

:   提出 Prof. Robot，首个结合碰撞约束的可微机器人渲染框架——将 3D 高斯点绑定到机器人 URDF 模型的各连杆上实现可微渲染，同时在优化中加入静态碰撞（与环境）和自碰撞（机器人自身）约束，将碰撞率从 24% 降至 0%，同时保持视觉保真度。

**[RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)**

:   提出 RoboGround，一个两阶段框架：先用 Grounded VLM（GLaMM）从图像和文本指令中生成目标物体和放置区域的分割掩码，再通过 Grounded Perceiver 将掩码作为中间表示引导机器人策略网络执行操作，在复杂语义操作任务上实现 60-100% 的相对提升。

**[Robotic Visual Instruction](robotic_visual_instruction.md)**

:   提出 Robotic Visual Instruction (RoVI)，一种以手绘箭头和圆圈为核心的视觉指令范式，替代自然语言来指导机器人操作，并设计 VIEW pipeline 将2D视觉指令转化为3D动作序列，在真实环境中达到87.5%成功率。

**[SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)**

:   SaPaVe 提出了一种端到端的主动操作框架，通过解耦相机运动和操作动作的 action space，采用自底向上的两阶段训练策略（先学语义相机控制，再联合优化），在 200K 语义相机运动数据集上训练主动感知先验，配合 3D 几何感知模块增强视角变化下的执行鲁棒性，在真实世界任务中比 GR00T N1 和 $\pi_0$ 分别高 31.25% 和 40% 成功率。

**[Solving Instance Detection from an Open-World Perspective](solving_instance_detection_from_an_open-world_perspective.md)**

:   从开放世界视角出发，通过度量学习适配基础模型特征、干扰物采样和NeRF新视角合成三种策略，显著提升实例检测中的实例级特征匹配性能，在CID和NID两种设定下均大幅超越前人方法。

**[SortScrews: A Dataset and Baseline for Real-time Screw Classification](sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)**

:   提出SortScrews数据集——一个包含560张512×512 RGB图像、覆盖6类螺丝的工业分类数据集，配套可复用的数据采集流水线，并以迁移学习的EfficientNet-B0和ResNet-18作为基线，ResNet-18在该数据集上达到96.4%验证准确率。

**[Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation](think_small_act_big_primitive_prompt_learning_for_lifelong_robot_manipulation.md)**

:   提出 Primitive Prompt Learning (PPL)，通过将运动原语编码为可复用的提示向量，结合光流感知的 Motion-Aware Prompting（MAP）实现跨技能运动原语共享，用冻结-扩展机制支持终身机器人操作学习，在 LIBERO 和真实世界中均优于 LoRA、经验回放等基线。

**[TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)**

:   在 ESP32 微控制器上部署端到端量化 CNN，仅用 23k 参数和 ToF 深度相机实现 30ms 延迟的实时自主导航。

**[Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)**

:   定义长程视觉语言导航（LH-VLN）任务，构建 NavGen 自动生成平台和 LHPR-VLN 基准（3260 个多阶段任务，平均 150 步），提出 MGDM 方法通过短期记忆模糊+长期记忆检索+CoT反馈实现多阶段导航，在 ISR 指标上超越 NaviLLM 23%。

**[UniAct: Universal Actions for Enhanced Embodied Foundation Models](universal_actions_for_enhanced_embodied_foundation_models.md)**

:   UniAct提出在通用动作空间（Universal Action Space）中构建具身基础模型，通过向量量化codebook编码跨具身平台共享的原子行为，0.5B参数模型性能超越14倍大的SOTA模型，并支持快速适配新机器人。
