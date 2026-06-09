---
title: >-
  [论文解读] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre
description: >-
  [ICLR 2026][多智能体][多智能体框架] 提出 HAMLET 多智能体框架，将 AI 戏剧创作和在线表演解耦为离线规划和在线表演两阶段，通过叙事蓝图、感知与决策（PAD）模块和层级控制系统，实现了具有主动性、物理环境交互能力和即兴表演自由的 AI 戏剧体验。
tags:
  - "ICLR 2026"
  - "多智能体"
  - "多智能体框架"
  - "戏剧表演"
  - "LLM Agent"
  - "感知与决策"
  - "交互叙事"
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

HAMLET 把"AI 戏剧"拆成解耦的两个阶段：离线规划阶段用一组协作智能体把任意主题或一部文学作品凝练成结构化的**叙事蓝图（Narrative Blueprint）**，在线表演阶段再由一套层级控制系统加载蓝图、驱动各角色实时即兴、并处理它们与舞台环境的物理交互。前者保证戏有骨架不跑偏，后者保证演员有主动性和即兴自由——HAMLET 的核心就是让这两者在一套框架里共存。

### 关键设计

**1. 离线规划：四智能体协作的蓝图工厂，把"写戏"流程化。** 单个 LLM 一次性写出可演的剧本既容易逻辑松散又难以约束，HAMLET 把它拆成四个分工明确、可互相审稿的智能体流水线。角色设计师（Actor Designer）先根据用户输入生成角色档案，既包含背景、性格等静态属性，也包含初始目标、核心关系等动态属性，必要时还能查询外部知识库补全设定；剧情设计师（Plot Designer）据此写出初步叙事草案；审稿人（Reviewer）专门审角色设定的合理性、动机是否清晰、角色关系是否成立，把不一致的设定挡在表演之前。最后由导演（Director）做结构化收尾：划分幕（Acts）与场景（Scenes）、列出每个场景的可交互道具、并把剧情拆成一串**叙事节点（Narrative Points）**，每个节点都带有明确的完成标志和结果。关键的一招是导演采用**逆向规划**——先把结尾节点定下来，再反向逐个补出前序节点，让每一步都朝既定结局收敛，从机制上避免实时表演时剧情越走越散。

**2. Beat 驱动的多轨迹即兴：在固定节点之间留出自由发挥空间。** 表演被组织成"幕 → 场景＋节点 → 节拍（Beat）"的层级，其中 Beat 是最小的有效交互步骤，即一个角色采取一次能推动局面的有效行动。每个角色的决策同时受**双目标系统**驱动：一是当前节点公开的完成标志（公共目标），二是角色自己的私有目标。蓝图只钉死了"必须从哪个节点走到哪个节点"，但两个节点之间允许存在**多条轨迹**，角色怎么吵、怎么合作、怎么绕路都由现场决定。这样既保留了导演设定的整体走向，又给了演员足够的即兴自由，是结构化叙事与开放表演之间的平衡点。

**3. PAD 模块：把 Kahneman 双过程理论塞进演员的脑子里。** 每个角色的行动由一个**感知与决策（Perceive And Decide, PAD）**模块产出，它显式融合直觉与反思两种推理。输入端同时给两类视角：主观的内部状态（Persona、主观关系、Memory、Goal）和客观的外部刺激（环境描述、在场角色列表、对话历史、可交互物体）。输出端则对应人类认知的双系统——FAST 是 System 1 的快速直觉反应，SLOW 是 System 2 的深思熟虑分析，SILENCE 表示选择沉默/不行动，同时通过工具调用生成结构化的潜在动作三元组（主语-动词-宾语）供环境裁决。PAD 本身是一个经过微调的 8B 模型，专门学会在快慢系统之间切换，让角色反应既能即时又能在关键处停下来想清楚。

**4. 叙述者裁决系统：让角色真正能"动手"改变舞台。** 纯对话戏的通病是角色说什么都算数、行为不影响环境。HAMLET 引入一个叙述者智能体（Narrator Agent）专门裁决所有物理交互：当某角色尝试一个物理动作时，叙述者依据当前环境状态和物理规则判断它是否可行——成功就确认动作、更新环境状态、并向所有角色广播一段客观描述；失败则判定不可行并给出合乎逻辑的解释。这一裁决环节把戏剧从抽象对白拉回到有因果、有后果的具身舞台。

**5. 层级控制三智能体：在线运行时的"现场调度"。** 在线表演由三个控制智能体协同推进。Planner 负责把当前节点的完成标志分解成可执行的 Beat 序列，并预设多条候选轨迹；Transfer 定期检查节点标志是否已满足，满足则推进到下一节点，同时管理角色的进场与退场；Advancer 是兜底机制——当剧情停滞超过时间阈值时，它主动引导相关角色采取行动把情节往前推，避免演出卡死。三者合起来保证表演既按蓝图走，又不会因为角色僵持而停摆。

**6. 三维评估与 HAMLETJudge：给"戏演得好不好"立可量化的标尺。** 为评估一场表演，HAMLET 设了三个维度：角色表演（角色一致性、情感表达）、叙事质量（剧情连贯性、结构完整性）、交互体验（环境交互自然度、沉浸感）。为降低评估成本，作者训练了一个 8B critic 模型 HAMLETJudge 来替代昂贵的人工或大模型评分，并以 GPT-4o 为基线做胜率比较，使三维评分能在大规模案例上自动给出。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](../../AAAI2026/multi_agent/hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)
- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](../../AAAI2026/multi_agent/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](../../ACL2026/multi_agent/posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](../../CVPR2025/multi_agent/collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)

</div>

<!-- RELATED:END -->
