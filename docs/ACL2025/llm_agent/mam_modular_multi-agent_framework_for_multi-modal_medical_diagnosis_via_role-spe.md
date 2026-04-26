---
title: >-
  [论文解读] MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration
description: >-
  [ACL 2025 (Findings)][LLM Agent][多智能体协作] 提出模块化多智能体框架 MAM，将医学诊断过程分解为全科医生、专科团队、放射科医生、医学助手和主任医师五个角色，通过角色专业化协作实现多模态（文本/图像/音频/视频）医学诊断，在多个公开数据集上比基线模型提升 18%~365%。
tags:
  - ACL 2025 (Findings)
  - LLM Agent
  - 多智能体协作
  - 多模态医学诊断
  - 角色分工
  - 知识检索
  - 大模型
---

# MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration

**会议**: ACL 2025 (Findings)  
**arXiv**: [2506.19835](https://arxiv.org/abs/2506.19835)  
**代码**: [https://github.com/yczhou001/MAM](https://github.com/yczhou001/MAM)  
**领域**: LLM Agent / 医学AI  
**关键词**: 多智能体协作、多模态医学诊断、角色分工、知识检索、大模型

## 一句话总结

提出模块化多智能体框架 MAM，将医学诊断过程分解为全科医生、专科团队、放射科医生、医学助手和主任医师五个角色，通过角色专业化协作实现多模态（文本/图像/音频/视频）医学诊断，在多个公开数据集上比基线模型提升 18%~365%。

## 研究背景与动机

**领域现状**：医学大语言模型在推理和诊断方面展现了强大能力。当前主流方案是训练统一的多模态医学 LLM，试图用一个模型处理所有模态的所有医学任务。

**现有痛点**：统一模型方案存在三个核心问题：（1）知识更新成本高——每当有新的医学知识或指南更新，整个模型需要重新微调；（2）覆盖面不全——单一模型难以同时精通文本问诊、影像诊断、音频分析和视频理解；（3）灵活性差——不同任务和科室的需求差异大，统一模型难以灵活适配。

**核心矛盾**：医学诊断天然是多学科协作的过程（不同科室的医生各有专长），而统一模型的思路恰恰与这种分工协作模式相悖。

**本文目标**：设计一个模块化的多智能体框架，模拟真实医院中多科室协作诊断的流程，让不同的 LLM-based agent 各司其职。

**切入角度**：作者通过实证发现，为 LLM 分配明确的角色（role assignment）以及引入诊断辨别能力（diagnostic discernment）可以显著提升诊断准确性。这启发了"按角色分工"的设计。

**核心 idea**：将医学诊断分解为五个专业角色，每个角色由一个 LLM agent 承担，模块间松耦合、可独立更新，并配合知识检索增强诊断能力。

## 方法详解

### 整体框架

MAM 的输入是患者的多模态医学数据（可以是文本病历、医学影像、音频记录或视频片段），输出是诊断结论。整个流程模拟真实医院的诊断链条：全科医生接诊 → 专科团队会诊 → 放射科读片 → 医学助手检索辅助 → 主任医师最终决策。每个环节由独立的 LLM agent 执行，通过结构化的消息传递协作。

### 关键设计

1. **角色专业化分工（五大 Agent）**:

    - 功能：将复杂的诊断任务分解为可管理的子任务
    - 核心思路：General Practitioner（全科医生）负责初步问诊和初步判断；Specialist Team（专科团队）提供特定领域的专业意见；Radiologist（放射科医生）处理医学影像的分析和解读；Medical Assistant（医学助手）通过外部知识库检索相关的临床指南和文献；Director（主任医师）综合所有信息做出最终诊断决策。每个 agent 使用针对性的 prompt 模板和可选的领域微调模型。
    - 设计动机：模拟真实诊疗流程，使每个模块可以独立优化和更新（如更换更好的影像分析模型时无需改动其他模块）。

2. **多轮讨论机制（Discussion Rounds）**:

    - 功能：通过多 agent 对话提升诊断一致性和准确性
    - 核心思路：各角色 agent 之间进行多轮结构化讨论。每轮讨论中，各 agent 共享自己的发现，其他 agent 可以质疑、补充或修正。通过多轮迭代达成共识。讨论轮次是可配置的超参数。
    - 设计动机：单个模型容易产生偏见或遗漏，多 agent 讨论可以模拟真实医学会诊中的"集体智慧"，减少误诊概率。

3. **检索增强诊断（Retrieval Augmentation）**:

    - 功能：引入外部知识弥补模型内在知识的不足
    - 核心思路：Medical Assistant agent 使用 Google Search API 等外部工具检索与当前病例相关的医学文献、临床指南和类似病例报告，将检索到的知识注入到讨论上下文中，帮助其他 agent 做出更有据的判断。
    - 设计动机：医学知识更新极快，模型的参数化知识可能过时。检索增强可以让系统访问最新的医学信息，而无需重新训练模型。

### 损失函数 / 训练策略

MAM 是一个 inference-time 框架，不需要额外训练。各个 agent 可以使用现成的通用 LLM 或领域微调的医学 LLM。框架通过 prompt engineering 和角色分配来激发模型的诊断能力。

## 实验关键数据

### 主实验

| 数据集 | 模态 | 任务 | 基线最佳 | MAM | 提升 |
|--------|------|------|---------|-----|------|
| MedQA | 文本 | 医学问答 | 52.3 | 71.8 | +37.3% |
| PubMedQA | 文本 | 文献问答 | 61.5 | 72.4 | +17.7% |
| VQA-RAD | 图像 | 影像问答 | 48.2 | 63.5 | +31.7% |
| PathVQA | 图像 | 病理问答 | 34.6 | 47.2 | +36.4% |
| MedVidQA | 视频 | 视频问答 | 15.3 | 33.6 | +119.6% |
| AudioMed | 音频 | 音频诊断 | 12.8 | 59.5 | +364.8% |

### 消融实验

| 配置 | MedQA Acc | VQA-RAD Acc | 说明 |
|------|-----------|-------------|------|
| Direct（基线） | 52.3 | 48.2 | 直接单模型回答 |
| + Role Assignment | 60.1 | 55.7 | 增加角色分配 |
| + Discussion | 65.4 | 58.9 | 增加多轮讨论 |
| + Retrieval (Full MAM) | 71.8 | 63.5 | 增加检索增强 |
| w/o Specialist | 64.2 | 57.1 | 去掉专科团队 |
| w/o Director | 67.3 | 60.2 | 去掉主任决策 |

### 关键发现
- **角色分配贡献最大**：从 Direct 到 +Role，MedQA 提升了 7.8%，说明给 LLM 赋予明确角色定位本身就能显著提高诊断质量。
- **音频和视频模态提升最为显著**：这些模态的基线很低，说明单模型在这些不常见模态上能力极弱，而多 agent 协作可以大幅弥补。
- **讨论轮次的边际效应**：2-3 轮讨论后效果趋于稳定，更多轮次带来的收益递减。
- **检索增强在知识密集型任务上尤其有效**：如 PubMedQA 这类需要大量背景知识的任务，检索模块贡献突出。

## 亮点与洞察
- **角色设计贴合真实医疗流程**：五个角色的设计不是随意的，而是对应了真实医院诊疗链条中的关键环节。这种"领域驱动"的 agent 设计思路比通用的 debate/discuss 更有效，值得在其他垂直领域（法律、金融等）推广。
- **模块化带来的实际工程价值**：各 agent 可独立更新，如发现更好的影像分析模型时只需替换 Radiologist agent，不影响其他模块。这对于需要频繁更新知识的医学场景尤为重要。
- **低成本的能力提升**：不需要训练新模型，仅通过角色编排和检索增强就能获得大幅提升，这对资源受限的医疗机构非常友好。

## 局限与展望
- **仅在 Findings 发表**：可能说明审稿人对某些方面有保留意见，如实验的公平性或创新性。
- **依赖 LLM 的基础能力**：框架本身不提升单个 agent 的能力，当底座模型在某个模态上很弱时，协作的提升天花板也受限。
- **讨论机制的计算开销**：多 agent 多轮讨论意味着多次 LLM 调用，推理成本是单模型的数倍，在实时诊断场景中可能不实用。
- **缺乏真实临床验证**：所有实验在公开基准上进行，缺乏在真实医院环境中的部署验证。
- 未来方向：可以探索自适应讨论轮次（简单病例少讨论、复杂病例多讨论）以及专家知识蒸馏来降低成本。

## 相关工作与启发
- **vs Med-PaLM 2**: Med-PaLM 2 是统一的端到端医学模型，而 MAM 采用模块化多 agent 架构。MAM 的优势是灵活性和可更新性，劣势是系统复杂度和推理延迟更高。
- **vs MedAgent-Zero**: MedAgent-Zero 也探索了多 agent 医学诊断，但角色设计较简单。MAM 的五角色分工更细致，且引入了检索增强。
- **vs ChatDoctor/DoctorGLM**: 这些是单 agent 医学对话模型，MAM 通过多 agent 协作在复杂诊断上明显更优。

## 评分
- 新颖性: ⭐⭐⭐ 多 agent 协作并非新概念，但在多模态医学诊断中的五角色分工设计有一定特色
- 实验充分度: ⭐⭐⭐⭐ 覆盖了四种模态的多个数据集，消融实验完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验组织有序
- 价值: ⭐⭐⭐⭐ 框架实用性强，对医学 AI 部署有参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](meddxagent_a_unified_modular_agent_framework_for_explainable_automatic_different.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](../../NeurIPS2025/llm_agent/lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)

<!-- RELATED:END -->
