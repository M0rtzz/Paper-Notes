---
title: >-
  CVPR2026 LLMAgent论文汇总 · 15篇论文解读
description: >-
  15篇CVPR2026的 LLM Agent 方向论文解读，涵盖 Agent、推理、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "推理"
  - "多模态"
item_list:
  - u: "argos_agentic_multi_camera_person_search/"
    t: "ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search"
  - u: "carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h/"
    t: "CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare"
  - u: "echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e/"
    t: "EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration"
  - u: "ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos/"
    t: "Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos"
  - u: "epiagent_agent_centric_system_for_ancient_inscription_restoration/"
    t: "EpiAgent: An Agent-Centric System for Ancient Inscription Restoration"
  - u: "gen_n_val_agentic_image_data_generation_and_validation/"
    t: "Gen-n-Val: Agentic Image Data Generation and Validation"
  - u: "gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen/"
    t: "GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents"
  - u: "hats_hardness-aware_trajectory_synthesis_for_gui_agents/"
    t: "HATS: Hardness-Aware Trajectory Synthesis for GUI Agents"
  - u: "haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion/"
    t: "HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search"
  - u: "nerfify_multiagent_nerf_paper_to_code/"
    t: "Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code"
  - u: "realm_mllm_agent_3d_reasoning_gaussian/"
    t: "REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting"
  - u: "sceneassistant_a_visual_feedback_agent_for_openvoc/"
    t: "SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation"
  - u: "think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video/"
    t: "Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding"
  - u: "towards_gui_agents_vision-language_diffusion_models_for_gui_grounding/"
    t: "Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding"
  - u: "worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning/"
    t: "WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**📷 CVPR2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (42)](../../ICML2026/llm_agent/index.md) · [💬 ACL2026 (78)](../../ACL2026/llm_agent/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/llm_agent/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (39)](../../NeurIPS2025/llm_agent/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/llm_agent/index.md)

🔥 **高频主题：** Agent ×8 · 推理 ×2 · 多模态 ×2

**[ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)**

:   本文提出 ARGOS，首个将多摄像头行人搜索重新定义为交互式推理问题的基准和框架，智能体通过与目击者进行多轮对话、调用时空工具并在信息不对称下推理排除候选人，包含 2,691 个任务、3 个渐进式赛道。

**[CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)**

:   提出CareFlow基准（1050个医疗软件长视界工作流任务，8-24步，覆盖DICOM/3D Slicer/EMR/LIS四大系统）和CarePilot框架（基于Actor-Critic范式，集成工具grounding和双记忆机制），在CareFlow上超越GPT-5约15%的任务准确率。

**[EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)**

:   提出EchoTrail-GUI三阶段闭环框架：探索Agent自主与GUI环境交互生成轨迹 → Critic奖励模型过滤仅保留高质量轨迹构建记忆库(EchoTrail-4K) → 新任务到来时通过密集+稀疏混合检索注入最相关记忆引导推理，将无状态GUI Agent转变为记忆增强系统，在AndroidWorld上GPT-4o达51.7% SR(+17.2pp)，在AndroidLab上Qwen2.5-VL-72B SR从23.9%提升至37.5%。

**[Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)**

:   提出 Ego2Web，首个将第一人称视频感知与 Web 代理执行相结合的基准测试，配套半自动数据构建流程和 Ego2WebJudge 自动评测框架，实验揭示当前最强 Agent 在真实视觉感知到在线行动的跨模态迁移上仍有巨大差距，最高仅 48.2% 成功率。

**[EpiAgent: An Agent-Centric System for Ancient Inscription Restoration](epiagent_agent_centric_system_for_ancient_inscription_restoration.md)**

:   EpiAgent是首个面向古代铭文修复的Agent系统，通过LLM中央规划器协调多模态分析、专用修复工具和迭代自我优化，在文字真实性和视觉保真度上超越现有方法。

**[Gen-n-Val: Agentic Image Data Generation and Validation](gen_n_val_agentic_image_data_generation_and_validation.md)**

:   本文提出 Gen-n-Val，一个基于智能体的合成数据生成与验证框架，通过 LLM 优化 Layer Diffusion 的 prompt 生成高质量单物体透明图像，再用 VLLM 过滤低质量样本，将无效合成数据从 50% 降至 7%，在 LVIS 稀有类实例分割上提升 7.6% mAP。

**[GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)**

:   提出 GUI-CEval，首个面向中文移动端 GUI Agent 的综合评测基准，覆盖 201 个主流中文 App、4 种设备类型，采用"基础能力+应用能力"两层结构从感知、规划、反思、执行、评估五个维度进行细粒度诊断，在 20 个代表性模型上的实验揭示当前模型在反思和自我评估方面仍有明显短板。

**[HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)**

:   提出难度感知的轨迹合成框架 HATS，通过 hardness-driven exploration 和 alignment-guided refinement 的闭环机制，专注采集和修正语义歧义动作的训练轨迹，大幅提升 GUI Agent 在复杂真实场景中的泛化能力。

**[HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)**

:   HAVEN 提出音视频实体凝聚 + 层次索引 + Agent搜索的统一框架，通过说话人身份作为跨模态一致性信号，构建全局-场景-片段-实体四级层次数据库，在LVBench上达到84.1%整体准确率的SOTA。

**[Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_multiagent_nerf_paper_to_code.md)**

:   提出 Nerfify，一个多智能体框架，通过上下文无关文法约束、图思维代码合成和组合式引用恢复，将 NeRF 论文自动转换为可训练的 Nerfstudio 插件代码，在 30 篇论文基准上实现 100% 可执行率，视觉质量与专家实现差距仅 ±0.5 dB PSNR。

**[REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](realm_mllm_agent_3d_reasoning_gaussian.md)**

:   提出 REALM 框架，利用 MLLM 的推理能力通过全局到局部空间定位策略在 3DGS 上进行开放世界 3D 推理分割，无需 3D 后训练即可处理隐式指令，在 LERF 上 mIoU 达 92.88%，远超基线方法 40+ 个百分点，并支持物体移除、替换和风格迁移等编辑任务。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_openvoc.md)**

:   提出SceneAssistant——基于纯视觉反馈的VLM agentic框架，设计14个功能完备的Action API让Gemini-3.0-Flash在ReAct闭环中迭代生成和优化开放词汇3D场景，无需预定义空间关系模板或外部布局求解器，在30个场景的人类评估中Layout得分7.600（vs SceneWeaver 5.800），Human Preference 65%。

**[Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)**

:   提出 VideoHV-Agent，将长视频问答重新建模为"假设-验证"过程：Thinker 将答案选项改写为可测试假设，Judge 提取区分性线索，Verifier 在视频中定位证据进行验证，Answer 综合证据给出最终答案，在 EgoSchema/NextQA/IntentQA 三个基准上取得 SOTA，同时推理效率优于现有 Agent 方法。

**[Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)**

:   首次系统研究离散扩散视觉语言模型（DVLM）在 GUI Grounding 中的应用，将 LLaDA-V 适配为单步动作预测，并提出混合掩码调度（线性+确定性）以捕获边界框坐标间的几何层次依赖，在 Web/Desktop/Mobile 界面上展示了扩散模型作为 GUI Agent 基础的可行性。

**[WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)**

:   提出 WorldMM，一个基于多模态记忆的视频推理 agent，构建情景记忆（多时间尺度文本知识图）、语义记忆（持续更新的关系知识图）和视觉记忆（帧级检索库）三类互补记忆，通过自适应多轮检索 agent 动态选择最相关的记忆源和时间粒度，在五个长视频 QA 基准上平均超越前 SOTA 8.4%。
