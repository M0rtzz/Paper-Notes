---
title: >-
  CVPR2026 LLM Agent方向 13篇论文解读
description: >-
  13篇CVPR2026 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**📷 CVPR2026** · 共 **13** 篇

**[Carepilot A Multi-Agent Framework For Long-Horizon Computer Task Automation In H](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)**

:   提出CareFlow基准（1050个医疗软件长视界工作流任务，8-24步，覆盖DICOM/3D Slicer/EMR/LIS四大系统）和CarePilot框架（基于Actor-Critic范式，集成工具grounding和双记忆机制），在CareFlow上超越GPT-5约15%的任务准确率。

**[Echotrail-Gui Building Actionable Memory For Gui Agents](echotrail-gui_building_actionable_memory_for_gui_agents.md)**

:   提出 EchoTrail-GUI 框架，通过评论模型引导的自主探索构建高质量操作记忆库，并在推理时动态检索相关经验注入提示，将 GPT-4o 在 AndroidWorld 上的任务成功率从 34.5% 提升至 51.7%。

**[Echotrail-Gui Building Actionable Memory For Gui Agents Via Critic-Guided Self-E](echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)**

:   提出EchoTrail-GUI，通过"学习-记忆-应用"三阶段闭环为GUI Agent构建可操作记忆——探索Agent自主交互GUI环境生成轨迹→Critic奖励模型过滤高质量轨迹存入记忆库→新任务时混合检索(密集+稀疏)最相关轨迹注入prompt引导推理，在AndroidWorld和AndroidLab上显著提升任务成功率。

**[Gui-Ceval A Hierarchical And Comprehensive Chinese Benchmark For Mobile Gui Agen](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)**

:   提出 GUI-CEval，首个面向中文移动端 GUI Agent 的综合评测基准，覆盖 201 个主流中文 App、4 种设备类型，采用"基础能力+应用能力"两层结构从感知、规划、反思、执行、评估五个维度进行细粒度诊断，在 20 个代表性模型上的实验揭示当前模型在反思和自我评估方面仍有明显短板。

**[Hats Hardness-Aware Trajectory Synthesis For Gui Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)**

:   提出难度感知的轨迹合成框架 HATS，通过 hardness-driven exploration 和 alignment-guided refinement 的闭环机制，专注采集和修正语义歧义动作的训练轨迹，大幅提升 GUI Agent 在复杂真实场景中的泛化能力。

**[Hats Hardnessaware Trajectory Synthesis Gui Agent](hats_hardnessaware_trajectory_synthesis_gui_agent.md)**

:   提出HATS框架，通过定义动作的"语义模糊度"作为难度信号，以难度驱动探索+对齐引导修复的闭环管线合成高质量GUI轨迹数据，显著提升agent泛化能力。

**[Nerfify A Multi-Agent Framework For Turning Nerf Papers Into Code](nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)**

:   提出 Nerfify，通过上下文无关文法(CFG)约束、图思维链(GoT)代码合成、组合式引用恢复和视觉反馈四阶段，将NeRF论文自动转化为可训练的Nerfstudio插件，在30篇论文基准上达到100%可执行率（通用基线仅5%），视觉质量在专家实现的±0.5dB PSNR内。

**[Realm An Mllm-Agent Framework For Open World 3D Reasoning Segmentation And Editi](realm_an_mllm-agent_framework_for_open_world_3d_reasoning_segmentation_and_editi.md)**

:   提出 REALM 框架，通过 MLLM agent 对 3D 高斯泼溅(3DGS)渲染的视图进行推理分割，设计全局-局部空间接地策略(GLSpaG)聚合多视角MLLM推理结果，在隐式指令下的3D分割中大幅超越现有方法（LERF上mIoU 92.88% vs 基线44.82%），并支持3D编辑。

**[Realm Mllm Agent 3D Reasoning Gaussian](realm_mllm_agent_3d_reasoning_gaussian.md)**

:   提出 REALM，一个基于 MLLM-Agent 的开放世界 3D 推理分割框架，利用 3DGS 渲染新视角供 MLLM 理解复杂指令，通过全局到局部空间定位策略实现精确 3D 分割——无需 3D 特定后训练即可处理隐式推理指令，并支持物体移除、替换和风格迁移等 3D 交互任务。

**[Sceneassistant A Visual Feedback Agent For Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)**

:   提出 SceneAssistant，通过为VLM agent提供完整的原子操作API集（13种动作涵盖物体管理、6-DoF操作、相机控制）和纯视觉反馈闭环，实现开放词汇的文本到3D场景生成，在人类评估中布局正确性和物体质量均大幅优于Holodeck和SceneWeaver。

**[Sceneassistant A Visual Feedback Agent For Openvoc](sceneassistant_a_visual_feedback_agent_for_openvoc.md)**

:   提出基于视觉反馈的VLM agent框架，通过14个完备Action API让VLM在ReAct闭环中迭代优化3D场景布局，无需预定义空间关系模板，在人类评估中Layout得分7.600（vs SceneWeaver 5.800），Human Preference 65%。

**[Think Then Verify A Hypothesis-Verification Multi-Agent Framework For Long Video](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)**

:   提出 VideoHV-Agent，将长视频问答重新建模为"假设-验证"过程：Thinker 将答案选项改写为可测试假设，Judge 提取区分性线索，Verifier 在视频中定位证据进行验证，Answer 综合证据给出最终答案，在 EgoSchema/NextQA/IntentQA 三个基准上取得 SOTA，同时推理效率优于现有 Agent 方法。

**[Towards Gui Agents Vision-Language Diffusion Models For Gui Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)**

:   首次系统研究离散扩散视觉语言模型（DVLM）在 GUI Grounding 中的应用，将 LLaDA-V 适配为单步动作预测，并提出混合掩码调度（线性+确定性）以捕获边界框坐标间的几何层次依赖，在 Web/Desktop/Mobile 界面上展示了扩散模型作为 GUI Agent 基础的可行性。
