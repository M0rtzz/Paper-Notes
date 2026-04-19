---
title: >-
  [论文解读] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare
description: >-
  [CVPR 2026][LLM Agent][医疗软件自动化] 提出CareFlow基准（1050个医疗软件长视界工作流任务，8-24步，覆盖DICOM/3D Slicer/EMR/LIS四大系统）和CarePilot框架（基于Actor-Critic范式，集成工具grounding和双记忆机制），在CareFlow上超越GPT-5约15%的任务准确率。
tags:
  - CVPR 2026
  - LLM Agent
  - 医疗软件自动化
  - 多Agent框架
  - Actor-Critic
  - 长视界GUI交互
  - 双记忆机制
---

# CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare

**会议**: CVPR 2026  
**arXiv**: [2603.24157](https://arxiv.org/abs/2603.24157)  
**代码**: 有 (Carepilot项目页)  
**领域**: LLM Agent / 医疗自动化  
**关键词**: 医疗软件自动化, 多Agent框架, Actor-Critic, 长视界GUI交互, 双记忆机制

## 一句话总结
提出CareFlow基准（1050个医疗软件长视界工作流任务，8-24步，覆盖DICOM/3D Slicer/EMR/LIS四大系统）和CarePilot框架（基于Actor-Critic范式，集成工具grounding和双记忆机制），在CareFlow上超越GPT-5约15%的任务准确率。

## 研究背景与动机

**领域现状**：多模态Agent已在Android/桌面/Web环境取得进展（Mind2Web、SeeAct、UI-TARS等），但缺乏面向医疗软件的标准化基准。

**医疗软件的独特挑战**：(1) 日常临床操作需链接10-15个依赖步骤（打开研究→配置视图→标注→导出→更新记录）；(2) 各平台高度异构且频繁更新；(3) 严格的数据完整性、审计追踪和隐私合规要求；(4) 界面布局机构特异性强——过度拟合表面布局的Agent脆弱。

**现有痛点**：(1) 无公开医疗软件长视界交互基准；(2) 现有VLM（GPT-4o、Gemini等）在医疗GUI上表现差——步级准确度尚可但任务完成率极低。

**切入角度**：构建首个医疗软件长视界基准 + 设计具备工具grounding和记忆机制的Actor-Critic Agent。

**核心idea**：Actor预测下一步动作→Critic评估并修正→双记忆（短期+长期）维持工作流上下文→迭代仿真训练提升鲁棒性。

## 方法详解

### 整体框架
自然语言目标 + 当前截图 → 工具Grounding(UI检测+OCR+缩放+模板匹配) → Actor读取双记忆+grounding信号→预测语义动作 → Critic评估→修正反馈或执行 → 更新记忆 → 下一步。

### 关键设计

1. **CareFlow基准（四阶段标注流程）**:

    - **(i) 种子任务设计**：与领域专家协作，映射各软件的使用模式和操作约束，提取核心任务清单
    - **(ii) 多样性扩展**：控制替换（"MRI报告"→"X光报告"）、参数调整、步骤增删
    - **(iii) 逐步GUI标注**：每步截图+精确的下一步语义动作标签
    - **(iv) 质量过滤**：时间顺序一致性+任务完整性+指令清晰度，Cohen's κ=0.78
    - 覆盖：Weasis/Orthanc(DICOM)、3D Slicer(标注)、OpenEMR(EMR)、OpenHospital(LIS)
    - 规模：1050任务(735训练+315测试,含50 OOD)，8-24步/任务，6种动作(CLICK/SCROLL/ZOOM/TEXT/SEGMENT/COMPLETE)

2. **工具Grounding（四个感知模块）**:

    - **UI目标检测**(开放词汇)：给定文本查询返回界面控件的bbox
    - **Zoom/Crop**：放大检查小控件
    - **OCR**：提取文本标签（系列名/患者字段/订单号）
    - **模板/图标匹配**：对主题/缩放/语言变化鲁棒
    - 四个模块输出聚合为统一grounding信号 $\phi_t$

3. **双记忆机制**:

    - **短期记忆** $\mathcal{M}_t^S = f^S(x_{t-1}, a_{t-1}, r_{t-1})$：上一步截图+动作+Critic反馈
    - **长期记忆** $\mathcal{M}_t^L = f^L(\mathcal{M}_{t-1}^L, \mathcal{M}_t^S, \phi_t)$：紧凑轨迹嵌入，整合历史状态/动作/结果
    - 动作预测条件化于两种记忆：$a_t = \pi_\theta(g, x_t, \mathcal{M}_t^S, \mathcal{M}_t^L)$
    - 设计动机：长视界工作流中错误会累积，双记忆机制在短期快速响应和长期上下文保持间取平衡

4. **Actor-Critic框架**:

    - Actor和Critic都基于Qwen-VL 2.5-7B实例化，仅输入条件和功能角色不同
    - Actor：观察当前界面+指令+grounding信号+记忆→预测语义动作
    - Critic：评估Actor提议→提供修正反馈或批准执行→更新双记忆
    - 迭代仿真训练：训练时Critic与参考轨迹对比；推理时靠执行结果或验证器反馈

### 损失函数 / 训练策略
任务定义为序列决策：$\hat{a}_{1:T} = \mathbb{1}[V(g, x_{1:T}, a_{1:T}) = 1]$，验证器V判断工作流是否成功完成。

## 实验关键数据

### 主实验（CareFlow）

| 模型 | Weasis SWA/TA | 3D Slicer SWA/TA | OpenEMR SWA/TA | Average SWA/TA |
|------|:-:|:-:|:-:|:-:|
| Qwen2.5 VL 7B | 58.6/1.3 | 61.4/1.7 | 63.2/1.7 | 57.2/1.8 |
| Llama 4 Maverick | 88.2/18.7 | 71.6/3.4 | 78.0/25.7 | 80.5/19.2 |
| GPT-4o | 85.3/20.0 | 77.5/27.4 | 85.1/27.5 | 83.1/25.4 |
| GPT-5 | 88.7/31.3 | 81.4/37.9 | 83.8/31.3 | 85.2/36.2 |
| **CarePilot (7B)** | **90.4/40.0** | **82.1/54.8** | - | **SOTA** |

CarePilot(基于7B模型)在Task Accuracy上超越GPT-5约15%。

### 消融实验

| 配置 | Avg SWA | Avg TA | 说明 |
|------|---------|--------|------|
| Qwen-VL 7B (基线) | 57.2 | 1.8 | 无Agent框架 |
| + 工具Grounding | +提升 | +提升 | 感知增强 |
| + 双记忆 | +提升 | +提升 | 上下文保持 |
| + Actor-Critic | **SOTA** | **SOTA** | 修正反馈关键 |

### 关键发现
- **步级准确度vs任务准确度的巨大鸿沟**：GPT-4o步级准确度83%但任务完成率仅25%——长视界中错误累积导致任务完成率急剧下降
- CarePilot的Critic修正机制有效缓解了错误累积，将任务完成率从基线的~2%提升到40%+
- 3D Slicer(医学标注)是最难的软件——需要精细的空间操作(分割/测量)，CarePilot在此子集上改进最大
- OOD测试集(50个任务)上CarePilot仍有3.38%的改进，证明一定程度的泛化能力
- 工具Grounding中OCR对EMR系统最关键（大量文本字段需精确识别）

## 亮点与洞察
- **首个医疗GUI长视界基准**：CareFlow填补了医疗软件AI自动化评测的空白。四阶段标注流程严谨，种子任务来自真正的临床从业者日常操作
- **7B模型超越GPT-5**：CarePilot证明了合适的Agent框架比更大的模型更重要——工具增强+记忆+修正反馈让7B模型在领域任务上超越GPT-5
- **Actor-Critic的医疗适配**：Critic不仅评估对错，还提供"怎么修正"的反馈——这对安全关键的医疗场景尤为重要
- **步级准确度≠任务成功**：这个发现对所有长视界Agent研究都有警示——不能仅看单步指标

## 局限与展望
- CareFlow仅覆盖5个开源医疗软件——商业系统(Epic, Cerner)的泛化有待验证
- Actor和Critic基于同一模型可能导致"盲点一致性"——不同模型做Actor和Critic可能更好
- 当前迭代仿真训练依赖参考轨迹——真实部署中需要不依赖参考的在线学习方案
- 安全性评估不足——医疗场景中错误操作可能有严重后果

## 相关工作与启发
- **vs WebArena/AppWorld**: 通用桌面/Web Agent基准，不覆盖医疗领域特殊需求
- **vs Voyager/Reflexion**: 记忆和反思机制启发了CarePilot的设计，但它们面向游戏/通用场景
- **vs Mind2Web/SeeAct**: 短视界GUI Agent，无法处理医疗工作流的长依赖链

## 评分
- 新颖性: ⭐⭐⭐⭐ 领域应用创新(首个医疗GUI基准)，框架设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 多系统、多基线(含GPT-5)、OOD测试、充分消融
- 写作质量: ⭐⭐⭐⭐ 基准构建过程详细，框架描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对医疗AI自动化有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](../../ICLR2026/llm_agent/the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](../../AAAI2026/llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)
- [AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](../../ICLR2026/llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)

<!-- RELATED:END -->
