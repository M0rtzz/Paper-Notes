---
title: >-
  [论文解读] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre
description: >-
  [ICLR 2026][LLM Agent][多智能体框架] 提出 HAMLET 多智能体框架，将 AI 戏剧创作和在线表演解耦为离线规划和在线表演两阶段，通过叙事蓝图、感知与决策（PAD）模块和层级控制系统，实现了具有主动性、物理环境交互能力和即兴表演自由的 AI 戏剧体验。
tags:
  - ICLR 2026
  - LLM Agent
  - 多智能体框架
  - 戏剧表演
  - 感知与决策
  - 交互叙事
---

# HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre

**会议**: ICLR 2026  
**arXiv**: [2507.15518](https://arxiv.org/abs/2507.15518)  
**代码**: [https://github.com/HAMLET-2025/HAMLET](https://github.com/HAMLET-2025/HAMLET)  
**领域**: LLM Agent / 交互叙事  
**关键词**: 多智能体框架, 戏剧表演, LLM Agent, 感知与决策, 交互叙事

## 一句话总结

提出 HAMLET 多智能体框架，将 AI 戏剧创作和在线表演解耦为离线规划和在线表演两阶段，通过叙事蓝图、感知与决策（PAD）模块和层级控制系统，实现了具有主动性、物理环境交互能力和即兴表演自由的 AI 戏剧体验。

## 研究背景与动机

### 问题背景
创建沉浸式交互戏剧体验是交互叙事领域的长期目标。LLM 的出现为此提供了新路径，但现有 LLM 驱动的戏剧生成方法存在三个关键问题：

**缺乏主动性**：AI 智能体通常被动等待指令，无法独立决策

**无法交互物理环境**：角色行为不影响舞台环境，戏剧变成抽象对话

**依赖详细用户输入**：需要完整故事大纲或详细引导段落，限制了灵活性

### 核心挑战
从被动响应到主动引导的范式转变——AI 演员需要能够自主决策、在开放场景中合作或冲突、并主动推动剧情发展。这是 Agentic AI 理念在戏剧表演中的具体体现。

## 方法详解

### 整体框架

HAMLET 框架解耦为两个主要阶段：

**阶段一：离线规划（Offline Planning）**
- 输入：任意主题 或 完整文学作品
- 多智能体协作工作流生成结构化叙事蓝图（Narrative Blueprint）

**阶段二：在线表演（Online Performance）**
- 输入：叙事蓝图
- 层级控制系统执行蓝图、管理实时交互、处理环境反馈

### 关键设计

1. **离线规划：四智能体协作工作流**

    - **角色设计师（Actor Designer）**：根据用户输入生成角色档案（静态属性如背景、性格 + 动态属性如初始目标、核心关系），可查询外部知识库
    - **剧情设计师（Plot Designer）**：基于主题和角色编写初步叙事草案
    - **审稿人（Reviewer）**：审核角色设定的合理性、动机清晰度和角色间关系
    - **导演（Director）**：负责最终结构化处理，执行以下关键步骤：
        - 划分幕（Acts）和场景（Scenes）
        - 创建环境互动元素（场景道具列表）
        - 定义叙事节点（Narrative Points）——每个节点包含明确的完成标志和结果
        - **逆向规划**：先生成结尾节点，再反向构建前序节点，防止剧情偏离

2. **在线表演：Beat 驱动的即兴系统**

   表演单元层级：**幕（Act）** → **场景（Scene）+ 节点（Point）** → **节拍（Beat）**

    - Beat 是一次有效交互步骤（角色采取有效行动）
    - 角色决策受**双目标系统**驱动：当前节点的公共标志 + 个人私有目标
    - 两个节点之间可以有**多条轨迹**，提供高度即兴自由

3. **PAD 模块（Perceive And Decide）**

   基于 Kahneman 双过程理论设计，融合直觉和反思推理：

   **输入双视角**：
    - 内部状态（主观）：Persona + 主观关系 + Memory + Goal
    - 外部刺激（客观）：环境描述 + 角色列表 + 对话历史 + 可交互物体

   **决策输出**：
    - FAST：快速直觉反应（System 1）
    - SLOW：深思熟虑的分析反应（System 2）
    - SILENCE：沉默/不行动
    - 潜在动作（通过工具调用生成结构化动作三元组：主语-动词-宾语）

   PAD 是一个 8B 模型，通过微调获得，能模拟人类认知的双系统。

4. **环境交互：叙述者裁决系统**

   设计了叙述者智能体（Narrator Agent）裁决所有物理交互：
    - 角色尝试物理动作时，叙述者基于环境状态和物理规则判断可行性
    - 成功：确认、更新环境状态、广播客观描述
    - 失败：判定失败并给出逻辑解释

5. **层级控制三智能体**

    - **Planner**：预设计多轨迹方案，分解标志为可执行 Beat 序列
    - **Transfer**：定期检查节点标志是否满足，推进到下一节点，管理角色进出
    - **Advancer**：如果剧情停滞超过时间阈值，引导相关角色推进情节

### 评估体系

三维度评估：
- **角色表演（Character Performance）**：角色一致性、情感表达
- **叙事质量（Narrative Quality）**：剧情连贯性、结构完整性
- **交互体验（Interaction Experience）**：环境交互自然度、沉浸感

训练了 HAMLETJudge（8B critic 模型），使用 GPT-4o 作为基线进行胜率比较。

## 实验关键数据

### 主实验：多模型评估排行榜

| 模型 | 英文平均分 | 中文平均分 | 总分 |
|------|----------|----------|------|
| Claude-4-sonnet-Thinking | 78.98 | 79.92 | **79.45** |
| Claude-4-sonnet | 76.92 | 79.68 | 78.30 |
| Qwen3-32B-Thinking | 69.10 | 78.59 | 73.85 |
| OpenAI-o3 | 69.45 | 77.89 | 73.67 |
| Qwen3-235B-A22B-Thinking | 70.74 | 75.92 | 73.33 |
| DeepSeek-R1-0528 | 66.58 | 79.37 | 72.98 |
| Qwen3-235B-A22B | 69.65 | 72.76 | 71.21 |
| Llama-3.1-8B | 35.51 | 33.83 | 34.67 |

### 数据集构成

| 来源 | 数量 | 说明 |
|------|------|------|
| 中国文学经典 | 25部 | 文学摘录 |
| 英文经典文学 | 25部 | 文学摘录 |
| 自定义主题 | 50个 | 涵盖10个不同主题 |
| **总计** | **100个案例** | |

### 关键发现
- 推理型模型（如 Claude-4-sonnet-Thinking）总体表现最佳，但优势不如预期明显
- 中文表演普遍优于英文表演（可能与中文文学要求更贴合框架设计有关）
- 小模型（如 Llama-3.1-8B）在戏剧表演上表现显著较差
- PAD 模块（8B）在决策任务上达到 SOTA 表现
- HAMLETJudge（8B）提供了成本效益高的可靠评估

## 亮点与洞察

1. **完整的 AI 戏剧流水线**：从主题输入到实时在线表演的端到端框架，填补了 AI 戏剧领域的系统性空白
2. **PAD 模块的认知理论基础**：基于 Kahneman 双过程理论将快系统和慢系统融入 AI 演员决策，使反应更加人类化
3. **逆向规划策略**：导演先确定结尾再反向补充剧情，有效防止剧情偏离，这在交互叙事中是一个聪明的设计
4. **Beat 驱动的多轨迹即兴**：两个叙事节点之间允许多条轨迹，在结构化叙事和自由即兴之间取得平衡
5. **物理环境交互**：叙述者裁决系统使戏剧不再是纯对话，增加了具身性和沉浸感

## 局限与展望

- 评估主要依赖 LLM-as-Judge（GPT-4o 和 HAMLETJudge），缺乏大规模人类评估
- 100 个案例的评估数据集规模有限
- 当前框架主要支持文本戏剧，未涉及多模态（语音、视觉、动作捕捉）
- PAD 模块是 8B 微调模型，在实时表演中可能存在延迟问题
- 人类玩家参与的交互体验未在论文中充分评估
- 长剧（多幕）表演中的一致性维护可能面临长上下文挑战

## 相关工作与启发

- **Dramatron** (Mirowski et al., 2023)：层级方法分离规划与生成，但不支持实时表演
- **CoSER** (Wang et al., 2025)：扩展角色数量但缺乏整体戏剧表演评估
- **CharacterEval** (Tu et al., 2024)：多轮对话多维评分，但限于双角色场景
- **Kahneman 双过程理论**：PAD 模块的认知科学基础
- 启发：层级控制 + 即兴自由的平衡设计对其他智能体系统（游戏NPC、虚拟助手）也有借鉴价值

## 评分
- 新颖性: ⭐⭐⭐⭐ — 框架设计全面新颖，PAD 模块有认知理论支撑
- 实验充分度: ⭐⭐⭐ — 排行榜评估有价值但缺乏人类评估和消融研究
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，图示丰富，但论文较长
- 价值: ⭐⭐⭐⭐ — 对交互叙事和 AI 戏剧领域有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [MAT-Agent: Adaptive Multi-Agent Training Optimization](../../NeurIPS2025/llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)
- [Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](../../CVPR2026/llm_agent/nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)
- [FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](../../ACL2025/llm_agent/fact_audit_factcheck.md)
- [M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)

<!-- RELATED:END -->
