---
title: >-
  [论文解读] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation
description: >-
  [ACL 2026][LLM Agent][法律咨询问答] 本文构建了JurisCQAD——一个包含43000+真实中文法律咨询的大规模数据集，并提出JurisMA多智能体框架，通过法律元素图进行结构化任务分解和动态多Agent协作（管理Agent+格式检查+法条检索），在LawBench上显著优于通用和法律专用LLM。
tags:
  - ACL 2026
  - LLM Agent
  - 法律咨询问答
  - 多智能体
  - 法律元素图
  - 任务分解
  - 中文法律
---

# From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation

**会议**: ACL 2026  
**arXiv**: [2604.10470](https://arxiv.org/abs/2604.10470)  
**代码**: 无  
**领域**: LLM Agent / 法律NLP  
**关键词**: 法律咨询问答、多智能体、法律元素图、任务分解、中文法律

## 一句话总结
本文构建了JurisCQAD——一个包含43000+真实中文法律咨询的大规模数据集，并提出JurisMA多智能体框架，通过法律元素图进行结构化任务分解和动态多Agent协作（管理Agent+格式检查+法条检索），在LawBench上显著优于通用和法律专用LLM。

## 研究背景与动机

**领域现状**：法律咨询问答（Legal CQA）是法律AI的核心任务，需要从个性化的法律困境中生成有据可依、可执行的法律建议。现有方法主要通过在法律语料上继续预训练或通过检索法条辅助生成。

**现有痛点**：(1) 缺乏高质量的训练数据——现有法律LLM（如LawGPT）主要在人造数据上训练，与真实咨询场景存在领域偏移；(2) 法律CQA涉及复杂的任务组合——需要识别法律关系、判断因果、定位核心问题、匹配法条，端到端模型难以完全覆盖；(3) 高度上下文依赖——需要精确解读法律实体、关系和用户意图。

**核心矛盾**：真实法律咨询通常模糊、多方面，需要动态解释事实、主体和法律含义。而现有方法要么依赖粗糙的继续预训练（监督信号质量低），要么依赖句子级法条检索（容易混淆法律上不同但语言上相似的概念）。

**本文目标**：构建大规模真实法律咨询数据集，设计可解释的任务分解和多Agent协作框架。

**切入角度**：将法律咨询分解为结构化的法律元素图——提取实体、事件、关系、用户意图和法律问题，然后通过多Agent协作迭代优化法律意见。

**核心 idea**：元素图提供语义基础→管理Agent动态协调子任务→格式检查Agent和法条检索Agent迭代精炼→内容检查Agent做最终润色。

## 方法详解

### 整体框架
JurisMA分三个阶段：(1) 法律语义图构建——Element Agent将查询解析为包含实体、事件、关系、意图和法律问题的图结构；(2) 多Agent迭代优化——Manager Agent动态评估草稿质量，按需调用FormatCheck Agent和LawSearch Agent进行格式修正和法条补充；(3) 内容修订——Content Check Agent做最终的语言质量和专业性修改。

### 关键设计

1. **法律元素图构建**:

    - 功能：将自由文本查询转化为结构化的法律语义表示，为下游推理提供全局上下文。
    - 核心思路：定义图 $G = (V, E)$，节点 $V$ 包含法律实体（个人/组织，带角色/状态/时间属性）、法律事件、用户主张、关键事实和推断的法律问题。边 $E$ 表示语义关系（如亲属关系、合同义务）。图序列化为JSON后与查询拼接，作为生成的语义输入。
    - 设计动机：法律推理围绕识别关键事实、涉及的主体和法律关系展开。图结构比扁平文本更能捕获这些结构化信息。受Hart的初级/次级规则理论和Kelsen的规范层次模型启发。

2. **Manager Agent动态协调**:

    - 功能：作为中央控制器，动态评估草稿质量并选择性激活子Agent。
    - 核心思路：Manager Agent在每轮迭代中检查草稿的(1)语言充分性——清晰度和简洁性，(2)法律完整性——是否包含权威法条引用。如果检测到结构/表达问题，调用Format Check Agent生成修改建议，由Draft Agent整合；如果法律引用不足，调用Law Search Agent从法规数据库检索相关条文。最多5轮迭代或Manager Agent返回"Pass"。
    - 设计动机：不是所有草稿都需要所有类型的修改。动态路由确保只激活必要的子Agent，保持效率和可控性。模拟了真实法律事务所中多角色协作修改法律意见书的工作流程。

3. **JurisCQAD数据集**:

    - 功能：提供大规模、高质量的真实法律咨询训练和评估数据。
    - 核心思路：43000+实例，每个组织为(问题, 正回答, 负回答)三元组。源自真实用户法律咨询，经专家验证。覆盖高频法律领域。正回答由专业律师审核，负回答由模型生成但被标注为不充分/错误。
    - 设计动机：现有中文法律数据集多为人造问答或法条检索任务，不能反映真实咨询的复杂性和语言多样性。三元组格式支持对比学习和偏好优化。

### 损失函数 / 训练策略
在JurisCQAD上使用SFT训练，结合元素图增强的输入。评估使用修订后的LawBench，包括多种词汇和语义指标。

## 实验关键数据

### 主实验

| 模型类别 | 代表模型 | LawBench性能 |
|---------|---------|-------------|
| 通用LLM | GPT-4, Qwen | 中等 |
| 法律LLM | LawGPT, ChatLaw | 中等偏低 |
| JurisMA | 基于JurisCQAD训练 | **显著最优** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full JurisMA | 最优 | 完整框架 |
| w/o 元素图 | 下降 | 失去结构化语义基础 |
| w/o 多Agent迭代 | 下降 | 法律引用不完整 |
| w/o LawSearch Agent | 显著下降 | 法条接地能力丧失 |

### 关键发现
- 在JurisCQAD上训练的模型显著优于通用和法律专用LLM，验证了高质量数据的价值
- 元素图提供的结构化上下文比原始文本输入更有效
- 多Agent迭代平均在2-3轮收敛，说明管理Agent的质量判断有效
- 有趣的是，法律专用LLM有时不如通用LLM——可能因为预训练数据质量参差不齐

## 亮点与洞察
- **法律元素图的语义表示**：将自由文本查询转化为包含实体-关系-事件的图结构，为法律推理提供了可解释的语义基础。这个思路可推广到其他需要结构化理解的专业领域（如医学、金融）。
- **动态Agent路由**：Manager Agent按需激活子Agent而非固定流程，平衡了效率和质量。
- **真实数据的力量**：43000个真实法律咨询的高质量数据集是重要的基础设施贡献。

## 局限与展望
- 仅覆盖中国法律体系，跨法域适用性未验证
- 元素图的质量依赖于Element Agent的理解能力
- LawSearch Agent的法条检索范围可能不完整
- 多Agent系统的推理成本较高（多轮迭代+多Agent调用）

## 相关工作与启发
- **vs LawGPT**：在法律语料上继续预训练，但数据处理粗糙。JurisMA用结构化任务分解和高质量数据
- **vs LawLuo**：固定流程的法律多Agent系统。JurisMA的Manager Agent提供动态路由
- **vs RAG方法（LSIM等）**：句子级法条检索可能混淆语言相似但法律不同的概念。元素图提供更精确的上下文

## 评分
- 新颖性: ⭐⭐⭐⭐ 法律元素图+多Agent协作的组合在法律NLP中是新颖的设计
- 实验充分度: ⭐⭐⭐⭐ 多种基线对比、消融实验充分
- 写作质量: ⭐⭐⭐⭐ 方法描述系统化，法律概念的引用准确
- 价值: ⭐⭐⭐⭐ 数据集和框架对法律AI研究有直接贡献

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)
- [\[ACL 2026\] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/llm_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)

<!-- RELATED:END -->
