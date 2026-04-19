---
title: >-
  [论文解读] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios
description: >-
  [CVPR 2025][LLM Agent][安全][基准测试] HomeSafe-Bench：评估VLM在家庭场景不安全动作检测中的能力，提出HD-Guard双脑分层流式安全监控架构
tags:
  - CVPR 2025
  - LLM Agent
  - 安全
  - 视觉语言模型
  - 基准测试
  - 具身智能
---

# HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios

**会议**: CVPR 2025  
**arXiv**: [2603.11975](https://arxiv.org/abs/2603.11975)  
**代码**: 待确认  
**领域**: LLM Agent / 安全  
**关键词**: VLM safety, unsafe action detection, household robots, benchmark, hierarchical monitoring, dual-brain architecture

## 一句话总结

提出HomeSafe-Bench——一个包含438个案例覆盖六大家庭功能区域的不安全动作检测基准，并设计HD-Guard双脑分层流式架构，以轻量FastBrain做高频筛查、大模型SlowBrain做深度推理，平衡家庭机器人安全监控的实时性与准确性。

## 研究背景与动机

**领域现状**: 具身智能体（家庭机器人）的快速发展正在加速其在真实家庭环境中的部署。与结构化的工业环境不同，家庭空间充满不可预测的安全风险——儿童、宠物、易碎物品、锐利工具等。当前VLM虽具备视觉理解能力，但其在检测机器人不安全行为方面的能力尚未被系统评估。

**现有痛点**: (1) 现有安全评估大多局限于静态图像或文本层面的有害内容检测，无法覆盖具身智能体在动态场景中的不安全动作（如机器人端着热水经过儿童身边）；(2) 机器人在家庭环境中的安全风险具有场景特异性和动态性——需要理解物理因果、常识推理和时序上下文；(3) 缺乏统一的评测维度和高质量标注数据来系统衡量VLM在此领域的能力瓶颈。

**核心矛盾**: 家庭安全监控要求**实时性**（高危动作需即时阻止）和**准确性**（复杂场景需深度推理），但大型VLM推理慢、轻量模型推理准确度不足——不可能同时用一个模型满足两个需求。

**本文目标** (1) 建立一个系统的VLM不安全动作检测基准（HomeSafe-Bench），涵盖多样的家庭场景和细粒度标注；(2) 提出一个平衡延迟与准确性的实时安全监控解决方案（HD-Guard）。

**切入角度**: 通过物理仿真+视频生成的混合pipeline构造高质量测试数据，用分层双脑架构解决速度-精度矛盾。

**核心 idea**: 用轻量FastBrain做高频持续安全筛查，仅在检测到可疑行为时异步唤醒大型SlowBrain做深度多模态推理，实现实时性与检测精度的最优权衡。

## 方法详解

### 整体框架

**HomeSafe-Bench基准**: 使用物理仿真（如AI2-THOR等）生成家庭机器人执行任务的诚实骨架，结合先进视频生成技术增强视觉真实性，构造出438个测试案例。案例分布在厨房、客厅、卧室、浴室、儿童房、车库等六大功能区域，每个案例包含视频序列、不安全动作时间戳、危险类型、严重程度等多维度细粒度标注。

**HD-Guard架构**: 双脑分层流式监控系统，处理来自机器人的实时视频流。FastBrain以高频率（如每秒多次）对每帧/短片段快速判断安全/可疑；SlowBrain仅在FastBrain发出警报时被异步激活，对更长时间窗口的多模态输入做深度分析并输出最终判决。

### 关键设计

1. **混合数据构造Pipeline**: 纯仿真数据缺乏视觉真实性、纯真实数据无法系统控制安全场景变量。HomeSafe-Bench创新性地使用物理仿真引擎生成语义正确的家庭交互场景（保证物理合理性和安全标签的精确性），然后用先进视频生成模型（如Sora类模型）对仿真渲染进行风格迁移和真实化。这样既保证了标注的精确性，又获得了接近真实世界的视觉多样性。438个案例覆盖了六大功能区域的多种不安全场景：操作锐利物品、接近高温源、在湿滑地面移动、靠近婴幼儿、化学品管理等。

2. **FastBrain高频轻量筛查**: FastBrain使用轻量级VLM（如小型CLIP/BLIP变体或蒸馏模型），对视频流的每一帧或短时间窗口做快速的二分类/风险评分。设计为高频持续运行的流式推理模式，延迟极低（毫秒级），尽管单帧判断的准确率有限，但高频采样确保不遗漏时间敏感的危险动作。当风险评分超过阈值时触发SlowBrain。关键设计点在于FastBrain的阈值调节可以trade-off漏报率和SlowBrain的调用频率。

3. **SlowBrain异步深度推理**: SlowBrain使用大型VLM（如GPT-4V级别），接收FastBrain标记的可疑时间段的完整多帧上下文，进行深度多模态推理——理解动作序列的因果关系、物理常识推理（热水+儿童=危险）、空间关系分析等。由于只在必要时被唤醒，大模型的高延迟不再是瓶颈。SlowBrain输出结构化的安全判定：安全/不安全、危险类型、严重等级、建议应对措施。

## 实验关键数据

### 关键发现

- 对主流VLM的系统评测揭示了**关键瓶颈**：即使是最强的VLM在动态不安全动作检测上仍有明显不足，尤其在需要物理因果推理和时序上下文理解的场景中
- HD-Guard在检测准确率上接近纯SlowBrain（大模型逐帧推理）的水平，但**延迟大幅降低**——典型延迟仅为纯大模型方案的一小部分
- 不同VLM在不同家庭区域的表现差异显著：厨房场景（涉及锐器、高温）的检测难度最大
- FastBrain的高频筛查覆盖了绝大多数不安全事件（高召回），SlowBrain显著降低了误报率（高精度）
- 消融实验表明：仅用FastBrain会有过多误报、仅用SlowBrain存在检测延迟过高或遗漏短暂危险动作的问题，分层设计互补性强

## 亮点与洞察

- HomeSafe-Bench填补了具身安全领域的重要评测空白——此前没有专门针对家庭机器人动态不安全行为的视频级benchmark
- 仿真+视频生成的数据构造pipeline具有很好的可扩展性，可以低成本地添加新场景类型和危险模式
- 双脑分层架构的设计哲学值得借鉴——不一定需要在所有时刻都用最强模型，而是用小模型做高频覆盖、大模型做精准判断
- 分析指出当前VLM在物理常识推理（因果链推断、物理属性理解）上的核心不足，为VLM安全研究指明了方向

## 局限与展望

- 438个案例的规模相对有限，难以覆盖家庭环境中所有可能的不安全场景（如电气安全、气体泄漏等）
- 视频生成得到的数据虽然视觉上真实，但可能引入artifacts，与真实机器人视角的first-person视频仍有域差异
- HD-Guard中FastBrain和SlowBrain的阈值需要手动调节，自适应阈值或学习调度策略可以进一步提升
- 当前框架假设视频流连续可用，在通信不稳定或机器人快速移动导致帧模糊的实际场景中的鲁棒性待验证
- 未讨论检测到不安全行为后的干预机制——如何与机器人控制系统联动实现即时制动是工程化的关键
- SlowBrain依赖大型VLM的API调用，在网络受限或需要离线运行的家庭场景中可能受限

<!-- RELATED:START -->

## 相关论文

- [From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)
- [Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](../../ACL2025/multimodal_vlm/effivlm_bench_acceleration.md)

<!-- RELATED:END -->
