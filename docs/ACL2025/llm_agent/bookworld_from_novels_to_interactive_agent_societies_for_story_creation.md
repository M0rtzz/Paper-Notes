---
title: >-
  [论文解读] BookWorld: From Novels to Interactive Agent Societies for Story Creation
description: >-
  [ACL 2025][LLM Agent][多智能体系统] BookWorld是首个基于小说的多智能体社会模拟系统，通过从源书籍中提取角色数据和世界观信息来构建交互式虚拟世界，让小说角色在其中自主行动和互动，进而生成创意故事，在75.36%的对比中超越了之前的故事生成方法。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "多智能体系统"
  - "小说模拟"
  - "故事生成"
  - "角色扮演"
  - "虚拟世界"
---

# BookWorld: From Novels to Interactive Agent Societies for Story Creation

**会议**: ACL 2025  
**arXiv**: [2504.14538](https://arxiv.org/abs/2504.14538)  
**代码**: [https://bookworld2025.github.io/](https://bookworld2025.github.io/)  
**领域**: LLM Agent  
**关键词**: 多智能体系统、小说模拟、故事生成、角色扮演、虚拟世界

## 一句话总结
BookWorld是首个基于小说的多智能体社会模拟系统，通过从源书籍中提取角色数据和世界观信息来构建交互式虚拟世界，让小说角色在其中自主行动和互动，进而生成创意故事，在75.36%的对比中超越了之前的故事生成方法。

## 研究背景与动机

**领域现状**：基于LLM的多智能体系统在社会模拟和协作任务中展现了巨大潜力。代表性工作包括Generative Agents（25个agent的沙盒世界）、Project Sid（千个agent的Minecraft文明模拟）、OASIS（百万级社交媒体模拟）。在故事生成领域，也有工作使用多agent协作（导演+编辑+演员）来写故事。

**现有痛点**：（1）现有agent社会都是从零开始构建的——agent的人设由简短描述或人口统计特征定义，缺乏深度；（2）模拟已有的虚构世界（小说、影视）几乎无人探索，尽管其有巨大实用价值（如互动小说、沉浸式游戏、同人创作）；（3）现有故事生成方法多采用自上而下的方法（先大纲后展开），产出的故事缺乏惊喜和紧张感，创意不足。

**核心矛盾**：自上而下的写作方法控制力强但创意不足；自下而上的角色驱动叙事有创意但难以保持一致性。如何在保持对源作品忠实度的同时让故事具有新意？

**本文目标**：构建一个完整的系统，能从小说中提取所有必要信息（角色、地图、世界观等），自动构建可模拟的虚拟世界，让角色自主行动并生成故事。

**切入角度**：将小说角色作为有完整人格、记忆和目标的自主agent，让故事从角色互动中"涌现"出来，而非由作者/导演agent自上而下规划。

**核心 idea**：书籍→数据提取→多角色agent社会构建→自主模拟→故事涌现，实现"角色驱动"的创意叙事。

## 方法详解

### 整体框架
BookWorld的完整流程分三个阶段：（1）数据准备——从源小说中提取角色属性、地理地图和世界观数据；（2）模拟——初始化角色agent和世界agent，进行多场景的交互模拟，角色在虚拟世界中自主行动、对话、探索；（3）润色——收集模拟记录，用LLM将交互日志改写为小说风格的连贯叙事。

### 关键设计

1. **角色Agent（Role Agent）设计**:

    - 功能：模拟小说角色的自主行为和社会互动
    - 核心思路：每个角色agent包含静态属性（性别、年龄、外貌、性格，从小说中提取，模拟过程中不变）和动态属性（目标、状态、记忆，随故事进展更新）。动作分为主动行动（由角色的goal驱动，包括角色互动、环境互动、独自活动）和被动响应（对他人行动的反应）。agent配备基于检索增强的长期记忆模块，模拟过程中不断积累和检索相关记忆。角色行动时先进行内部思考（分析当前状态和目标），然后决定行动，最后更新记忆和状态。
    - 设计动机：相比简短描述定义的agent，从完整小说中提取的角色有更丰富的人格维度和行为模式。静态/动态属性分离确保了角色一致性与发展性的平衡。

2. **世界Agent（World Agent）和地理空间系统**:

    - 功能：管理模拟的整体流程和空间约束
    - 核心思路：世界agent负责场景调度（根据角色位置和历史互动选择每场景的参与者）、环境反馈（提供环境变化信息）、NPC管理（为临时出现的路人创建无记忆的临时agent）。地理空间建模为离散图——节点是地点，边是路径，角色在地点间移动需要消耗场景轮次。这确保同一时间只有在同一地点的角色才能直接互动。世界agent还维护全局状态（如天气、时间流逝、社会事件）。
    - 设计动机：空间约束是叙事合理性的核心——没有地理限制，角色会出现违反物理常识的"瞬移"，破坏故事一致性。

3. **世界观数据收集与注入**:

    - 功能：让agent的行为符合虚构世界的规则和文化
    - 核心思路：从源小说中系统性提取三类世界观数据：社会规范（如魔法世界的法律、种族关系）、文化背景（如节日、习俗、价值观）、术语解释（如魔法名称、特殊物品）。这些数据以结构化形式存储，在agent每次决策时通过相关性检索注入上下文。这样agent不仅知道"我是谁"，还知道"这个世界是怎样的"。
    - 设计动机：很多小说设定了独特的世界观（如哈利波特的魔法世界），角色行为必须在这些规则下才合理。不注入世界观数据，agent可能生成违反设定的行为（如魔法世界的角色使用现代科技解决问题）。

### 损失函数 / 训练策略
本文为系统设计工作，不涉及模型训练。所有agent基于现成LLM（如GPT-4）通过prompt engineering实现。

## 实验关键数据

### 主实验（故事生成质量对比）

| 方法 | 忠实度 | 创意性 | 文本质量 | 总体胜率 |
|------|--------|--------|---------|---------|
| BookWorld | 4.12 | 3.89 | 4.05 | **75.36%** |
| 单LLM叙事 | 3.68 | 3.21 | 3.92 | 12.8% |
| 多Agent写作(Director+Editor) | 3.85 | 3.45 | 3.98 | 11.84% |

### 消融实验

| 配置 | 忠实度 | 创意性 | 总体评分 | 说明 |
|------|--------|--------|---------|------|
| Full BookWorld | 4.12 | 3.89 | 4.05 | 完整系统 |
| w/o 世界观数据 | 3.52 | 3.75 | 3.61 | 忠实度大幅下降 |
| w/o 地理空间约束 | 3.91 | 3.82 | 3.70 | 叙事逻辑性下降 |
| w/o 角色记忆 | 3.65 | 3.67 | 3.55 | 角色行为一致性变差 |
| w/o 动态属性更新 | 3.88 | 3.41 | 3.68 | 创意性显著下降——角色无法发展 |

### 关键发现
- BookWorld在75.36%的对比中被评价为优于baseline，主要优势在忠实度和创意性
- 世界观数据对忠实度影响最大（去掉后下降0.60），证明了世界观注入的必要性
- 动态属性更新对创意性影响最大（去掉后下降0.48），说明角色发展是创意叙事的关键来源
- 地理空间约束不直接提升故事质量评分，但显著减少了"角色瞬移"等逻辑错误
- 角色互动中涌现了有趣的行为模式——角色会基于个人目标做出与原著不同但合理的选择

## 亮点与洞察
- 首次系统性地研究基于小说的agent社会构建，填补了"虚构世界模拟"的研究空白。这不仅有学术价值，在互动叙事、游戏设计、IP衍生创作等领域也有巨大商业潜力。
- "角色驱动"而非"大纲驱动"的叙事方法让故事具有涌现性和惊喜感——角色可能做出作者（甚至系统设计者）预料不到的行为，这恰恰是好故事的特征。
- 世界观数据提取和注入的方法论具有通用性，可以扩展到游戏世界（如DnD设定）、历史场景（如某个朝代）等其他虚拟世界构建任务。

## 局限与展望
- 数据提取阶段高度依赖LLM的阅读理解能力，对于长篇小说（10万+字）可能存在信息遗漏
- 模拟规模受限于LLM的推理成本——每个角色每步行动都需要一次LLM调用
- 生成故事的长度和复杂度受模拟轮数限制，难以生成真正长篇叙事
- 角色"忠实度"的评估带有主观性，不同读者对"角色是否符合原著"的判断可能不同
- 未来可以结合视觉生成模型，为模拟过程生成场景图像，实现多模态叙事

## 相关工作与启发
- **vs Generative Agents (Park et al.)**: Generative Agents用简短人设描述创建agent，BookWorld从完整小说提取角色，人格深度和行为丰富度远超
- **vs 多Agent写作系统 (Han et al.)**: 传统多agent写作是自上而下的（导演规划→角色执行），BookWorld是自下而上的（角色行动→故事涌现），创意性更高
- **vs RecurrentGPT/LongStory**: 这些方法用单个LLM滚动生成长文本，缺乏角色自主性和世界一致性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个小说→agent社会的完整系统，问题定义和解决方案都有原创性
- 实验充分度: ⭐⭐⭐⭐ 定量和定性评估结合，消融分析到位
- 写作质量: ⭐⭐⭐⭐ 系统设计描述清晰，示例生动
- 价值: ⭐⭐⭐⭐⭐ 学术创新+实际应用价值兼备

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](../../NeurIPS2025/llm_agent/generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[NeurIPS 2025\] ShapeCraft: LLM Agents for Structured, Textured and Interactive 3D Modeling](../../NeurIPS2025/llm_agent/shapecraft_llm_agents_for_structured_textured_and_interactive_3d_modeling.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](../../AAAI2026/llm_agent/llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](../../ACL2026/llm_agent/fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)

</div>

<!-- RELATED:END -->
