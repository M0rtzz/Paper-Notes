---
title: >-
  CVPR2026 LLM Agent方向 16篇论文解读
description: >-
  16篇CVPR2026 LLM Agent方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**📷 CVPR2026** · **16** 篇论文解读

**[CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)**

:   提出CareFlow基准（1050个医疗软件长视界工作流任务，8-24步，覆盖DICOM/3D Slicer/EMR/LIS四大系统）和CarePilot框架（基于Actor-Critic范式，集成工具grounding和双记忆机制），在CareFlow上超越GPT-5约15%的任务准确率。

**[EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents.md)**

:   提出 EchoTrail-GUI 框架，通过评论模型引导的自主探索构建高质量操作记忆库，并在推理时动态检索相关经验注入提示，将 GPT-4o 在 AndroidWorld 上的任务成功率从 34.5% 提升至 51.7%。

**[EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)**

:   提出EchoTrail-GUI三阶段闭环框架：探索Agent自主与GUI环境交互生成轨迹 → Critic奖励模型过滤仅保留高质量轨迹构建记忆库(EchoTrail-4K) → 新任务到来时通过密集+稀疏混合检索注入最相关记忆引导推理，将无状态GUI Agent转变为记忆增强系统，在AndroidWorld上GPT-4o达51.7% SR(+17.2pp)，在AndroidLab上Qwen2.5-VL-72B SR从23.9%提升至37.5%。

**[Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)**

:   提出 Ego2Web，首个将第一人称视频感知与 Web 代理执行相结合的基准测试，配套半自动数据构建流程和 Ego2WebJudge 自动评测框架，实验揭示当前最强 Agent 在真实视觉感知到在线行动的跨模态迁移上仍有巨大差距，最高仅 48.2% 成功率。

**[GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)**

:   提出 GUI-CEval，首个面向中文移动端 GUI Agent 的综合评测基准，覆盖 201 个主流中文 App、4 种设备类型，采用"基础能力+应用能力"两层结构从感知、规划、反思、执行、评估五个维度进行细粒度诊断，在 20 个代表性模型上的实验揭示当前模型在反思和自我评估方面仍有明显短板。

**[HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)**

:   提出难度感知的轨迹合成框架 HATS，通过 hardness-driven exploration 和 alignment-guided refinement 的闭环机制，专注采集和修正语义歧义动作的训练轨迹，大幅提升 GUI Agent 在复杂真实场景中的泛化能力。

**[HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardnessaware_trajectory_synthesis_gui_agent.md)**

:   本文提出 HATS——一个难度感知的轨迹合成框架，通过识别和处理语义歧义的GUI操作，利用难度驱动探索和对齐引导修正两个闭环模块，显著提升GUI智能体在跨环境下的泛化能力。

**[Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)**

:   提出 Nerfify，通过上下文无关文法(CFG)约束、图思维链(GoT)代码合成、组合式引用恢复和视觉反馈四阶段，将NeRF论文自动转化为可训练的Nerfstudio插件，在30篇论文基准上达到100%可执行率（通用基线仅5%），视觉质量在专家实现的±0.5dB PSNR内。

**[Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_multiagent_nerf_paper_to_code.md)**

:   提出 Nerfify，一个领域感知的多智能体框架，通过上下文无关文法（CFG）约束、图思维（GoT）代码合成和组合式引用依赖恢复，将 NeRF 论文自动转化为可训练的 Nerfstudio 插件，实现 100% 可执行率，视觉质量与专家实现仅差 ±0.5 dB PSNR。

**[REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_an_mllm-agent_framework_for_open_world_3d_reasoning_segmentation_and_editi.md)**

:   提出 REALM 框架，通过 MLLM agent 对 3D 高斯泼溅(3DGS)渲染的视图进行推理分割，设计全局-局部空间接地策略(GLSpaG)聚合多视角MLLM推理结果，在隐式指令下的3D分割中大幅超越现有方法（LERF上mIoU 92.88% vs 基线44.82%），并支持3D编辑。

**[REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_mllm_agent_3d_reasoning_gaussian.md)**

:   提出 REALM 框架，利用 MLLM 的推理能力通过全局到局部空间定位策略在 3DGS 上进行开放世界 3D 推理分割，无需 3D 后训练即可处理隐式指令，在 LERF 上 mIoU 达 92.88%，远超基线方法 40+ 个百分点，并支持物体移除、替换和风格迁移等编辑任务。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)**

:   提出 SceneAssistant，通过为VLM agent提供完整的原子操作API集（13种动作涵盖物体管理、6-DoF操作、相机控制）和纯视觉反馈闭环，实现开放词汇的文本到3D场景生成，在人类评估中布局正确性和物体质量均大幅优于Holodeck和SceneWeaver。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_openvoc.md)**

:   提出SceneAssistant——基于纯视觉反馈的VLM agentic框架，设计14个功能完备的Action API让Gemini-3.0-Flash在ReAct闭环中迭代生成和优化开放词汇3D场景，无需预定义空间关系模板或外部布局求解器，在30个场景的人类评估中Layout得分7.600（vs SceneWeaver 5.800），Human Preference 65%。

**[Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)**

:   提出 VideoHV-Agent，将长视频问答重新建模为"假设-验证"过程：Thinker 将答案选项改写为可测试假设，Judge 提取区分性线索，Verifier 在视频中定位证据进行验证，Answer 综合证据给出最终答案，在 EgoSchema/NextQA/IntentQA 三个基准上取得 SOTA，同时推理效率优于现有 Agent 方法。

**[Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)**

:   首次系统研究离散扩散视觉语言模型（DVLM）在 GUI Grounding 中的应用，将 LLaDA-V 适配为单步动作预测，并提出混合掩码调度（线性+确定性）以捕获边界框坐标间的几何层次依赖，在 Web/Desktop/Mobile 界面上展示了扩散模型作为 GUI Agent 基础的可行性。

**[WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)**

:   提出 WorldMM，一个基于多模态记忆的视频推理 agent，构建情景记忆（多时间尺度文本知识图）、语义记忆（持续更新的关系知识图）和视觉记忆（帧级检索库）三类互补记忆，通过自适应多轮检索 agent 动态选择最相关的记忆源和时间粒度，在五个长视频 QA 基准上平均超越前 SOTA 8.4%。
