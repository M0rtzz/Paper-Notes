---
title: >-
  [论文解读] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns
description: >-
  [ACL 2026][LLM评测][拟人化] 本文提出 HumanLLM 框架，将 244 个心理学模式（100 个人格特质 + 144 个社会认知模式）建模为相互作用的因果力而非孤立标签，构建了 11,359 个包含 2-5 个模式交互的场景和多轮对话数据集…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "拟人化"
  - "认知模式"
  - "多模式动态"
  - "角色扮演Agent"
  - "心理学建模"
---

# HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns

**会议**: ACL 2026  
**arXiv**: [2601.10198](https://arxiv.org/abs/2601.10198)  
**代码**: [GitHub](https://github.com/YJGoodbye2024/HumanLLM)  
**领域**: 角色扮演 / 人格模拟  
**关键词**: 拟人化, 认知模式, 多模式动态, 角色扮演Agent, 心理学建模

## 一句话总结

本文提出 HumanLLM 框架，将 244 个心理学模式（100 个人格特质 + 144 个社会认知模式）建模为相互作用的因果力而非孤立标签，构建了 11,359 个包含 2-5 个模式交互的场景和多轮对话数据集，通过双层 checklist 评估实现与人类判断的高对齐（$r=0.90$），HumanLLM-8B 在多模式动态上以 4 倍小的参数量超越 Qwen3-32B。

## 研究背景与动机

**领域现状**：角色扮演语言 Agent（RPLA）已从概念框架发展为数字克隆、AI 伴侣和社会模拟等实际应用。现有人格注入方法包括：(1) 提示法——通过指令赋予人格标签；(2) 微调法——在角色特定数据上训练；(3) 激活转向——通过 persona vector 操纵内部表示。

**现有痛点**：(1) 现有方法将人格建模为孤立的标签→行为映射（"外向"→"健谈"），忽略了多个认知模式之间的动态交互——现实中一个健谈的人在"聚光灯效应"激活时可能沉默；(2) 这导致"人格幻觉"——模型在自我报告中声称具有某种特质但行为表现不一致；(3) 现有评估使用整体性指标（如 CoSER 的 Anthropomorphism），但这些指标隐式地将"好的拟人化"等同于"亲社会行为"，惩罚了真实但负面的人类特质（如防御性归因）。

**核心矛盾**：人类行为是多个认知模式动态交互的产物——自信的人可能在从众压力下让步，健谈的人在被关注时变得沉默。但现有方法只能模拟单一特质，无法捕捉这种"模式间的张力和调制"。

**本文目标**：(1) 构建大规模心理学模式数据集（每个模式基于约 50 篇学术论文）；(2) 设计多模式交互场景让模型学习模式间的动态关系；(3) 提出能区分"模拟准确性"和"社会期望性"的评估框架。

**切入角度**：基于 Lewin 场论——人的认知由两个维度组成：稳定的人格特质（Person）和情境触发的社会认知模式（Environment）。将模式视为相互作用的因果力而非孤立标签——通过让模型在多模式场景中训练，隐式学习模式间的增强、冲突和条件性调制。

**核心 idea**：将心理学模式建模为交互因果力，通过在多模式交互场景中训练 LLM，让模型学习"不只是人类做什么，更是产生这些行为的心理过程"——从行为模仿升级为认知建模。

## 方法详解

### 整体框架

HumanLLM 包含三个核心组件：(1) 模式数据构建——从约 12,000 篇学术论文中提取 244 个心理学模式的结构化表示（定义+机制+表现）；(2) 场景和对话生成——构建 11,359 个包含 2-5 个模式交互的场景，每个场景包含 2-6 个角色的多轮对话（含内心想法、行为、语言）；(3) 双层 checklist 评估——模式级（12-15 个通用行为指标）+ 场景级（2-6 个特定行为预期）。

### 关键设计

1. **基于文献的心理学模式构建**:

    - 功能：为每个模式提供科学严格的结构化表示
    - 核心思路：人格特质采用 Goldberg 的 100 个单极标记（Big Five 每个维度 20 个描述符）。社会认知模式从认知偏差（Tversky & Kahneman）、社会影响（Cialdini）、进化心理学和动机研究中策展，从 232 个候选中筛选出 144 个（要求充分实证验证+非冗余）。每个模式通过 Gemini Deep Search 检索约 50 篇学术论文，再由 Gemini 2.5 Pro 综合为三层结构：定义、核心机制、真实世界表现。人工验证显示平均评分 3.20-3.70（4 分制），Krippendorff $\alpha$ = 0.58-0.76
    - 设计动机：区别于现有工作从模型参数知识生成角色描述，本文每个模式都有约 50 篇学术论文支撑——确保心理学严谨性和科学效度

2. **多模式交互场景生成**:

    - 功能：让模型学习模式间的动态关系（增强/冲突/条件调制）
    - 核心思路：每个场景包含 2-5 个模式组合，涵盖三种交互类型：增强（如"自利偏差"强化"过度自信效应"）、冲突（如"自信"vs"从众"）、条件调制（如"健谈"被"聚光灯效应"抑制）。角色设计包含自我感知和他者感知以支持信息不对称。使用 DIAMONDS 模型确保情境多样性。每个场景还生成预期行为倾向作为评估标准。对话（12-20 轮）由 Claude Sonnet 4.5 生成，每轮包含三维表达：内心想法（方括号）、身体行为（圆括号）和语言表达
    - 设计动机：关键创新在于模式组合的多样性——不是简单地叠加特质，而是构建需要模式间"谈判"的场景。三维表达设计让模型学习表面行为和内在认知过程的分离

3. **双层 Checklist 评估**:

    - 功能：将模拟准确性与社会期望性解耦
    - 核心思路：模式级 checklist 包含每个模式 12-15 个通用行为指标（如聚光灯效应："过度估计他人对自己外表的关注"），从模式定义推导，跨场景通用。场景级 checklist 包含每个角色 2-6 个情境特定行为预期（如"在截止日期压力下仍坚持概念完整性"），从预期行为倾向推导。使用 GPT-5-mini 作为三元评分评判（+1满足/0未展示/-1违反）。评估指标：IPE（Individual Pattern Expression）衡量单个模式保真度，MPD（Multi-Pattern Dynamics）衡量多模式交互的涌现行为
    - 设计动机：传统整体性指标（如 CoSER 的 Anthropomorphism）与人类判断相关性弱（$r=0.43$）且存在"规范性混淆"——LLM 评判将"防御性归因"评为低拟人化因为"缺乏共情"。Checklist 通过价值中性的行为指标实现 $r=0.90$ 的人类对齐

### 损失函数 / 训练策略

监督微调（SFT）：将每个角色的对话转为 ShareGPT 格式，产生 30,543 个 HumanLLM 样本。混合训练数据：HumanLLM + OpenThoughts-114k（指令遵循）+ CoSER（角色扮演），比例 4:4:2，共 76,358 样本。基座模型 Qwen3-8B/32B。

## 实验关键数据

### 主实验

**IPE 和 MPD 评估（%, 3 次评判均值±标准差）**

| 模型 | IPE | MPD |
|------|-----|-----|
| GPT-5 | 15.5±0.4 | 43.4±1.1 |
| Claude Sonnet 4.5 | 34.8±0.3 | 79.5±0.4 |
| Gemini 3 Pro | 41.3±0.3 | 85.1±0.4 |
| Qwen3-8B | 18.6±0.7 | 54.4±2.1 |
| Qwen3-32B | 26.0±0.4 | 65.8±0.7 |
| DeepSeek-R1 | 23.3±0.6 | 69.0±0.5 |
| **HumanLLM-8B** | **25.7±0.4** | **70.3±0.6** |
| **HumanLLM-32B** | **32.8±0.3** | **73.6±0.4** |

### 消融实验

**数据消融（8B 变体）**

| 配置 | IPE | MPD |
|------|-----|-----|
| Qwen3-8B (base) | 18.6 | 54.4 |
| Qwen3-8B (OT+CoSER, 无 HumanLLM 数据) | 9.1 | 31.3 |
| HumanLLM-8B (完整) | 25.7 | 70.3 |

**评估框架对齐验证（100 个场景）**

| 指标类型 | 人类 | LLM | Δ | 相关系数 r |
|---------|------|-----|---|----------|
| Anthropomorphism (整体) | 84.6 | 53.8 | -30.8 | 0.43 |
| Character Fidelity (整体) | 83.1 | 65.4 | -17.7 | 0.61 |
| **IPE (checklist)** | **38.4** | **37.8** | **-0.6** | **0.90** |
| **MPD (checklist)** | **72.1** | **75.8** | **+3.7** | **0.88** |

### 关键发现

- HumanLLM-8B 在 MPD 上（70.3%）超越 Qwen3-32B（65.8%），4 倍参数量差异证明心理学训练数据比模型规模更重要
- GPT-5 表现意外低（IPE: 15.5%），分析显示其强指令遵循倾向导致过度字面化的角色扮演——通用能力不自动迁移到心理学模拟
- 负迁移发现：仅用 OpenThoughts+CoSER 训练反而使性能大幅下降（IPE: 18.6→9.1），通用数据抑制了模型的心理模式表达能力。HumanLLM 数据不仅补偿了这种负迁移还产生了协同效应
- 传统整体性指标存在"规范性混淆"——LLM 评判将社会期望性等同于模拟准确性，checklist 方法有效解耦了两者

## 亮点与洞察

- 将心理学模式建模为"交互因果力"而非"孤立标签"是概念上的重要突破——这一视角可推广到任何需要多维人格模拟的应用（游戏 NPC、社会模拟、心理咨询训练）
- 规范性混淆的发现具有方法论价值——揭示了 LLM-as-Judge 在评估人类行为模拟时的系统性偏差，checklist 方法提供了一个可复用的解决方案
- 负迁移发现对 SFT 数据配比有直接启示——通用数据可能"淹没"领域特定能力，需要锚定性数据（如 HumanLLM）来保持

## 局限与展望

- 对话平均 16.4 轮，长程角色一致性（50+ 轮）未评估
- 心理学理论主要来自 WEIRD 人群，跨文化适用性存疑——如从众压力在集体主义文化中表现可能不同
- 训练数据全部由 LLM 合成，合成数据与真实人类交互之间仍有差距
- 高保真的负面特质模拟（如操纵、偏见）带来安全和伦理风险——部署需要额外的安全层

## 相关工作与启发

- **vs CoSER**: CoSER 从 771 本书提取对话，评估用整体性指标。HumanLLM 从学术论文构建心理学模式，评估用 checklist——实现了 $r=0.90$ vs CoSER 的 $r=0.43$ 的人类对齐
- **vs Character-LLM**: 通过经验重建训练历史人物 Agent，关注单一角色。HumanLLM 关注通用认知模式而非特定角色，更可泛化
- **vs Persona Vectors (Chen et al.)**: 通过激活转向操纵特质，但无法处理多特质冲突。HumanLLM 通过场景训练隐式学习多模式动态

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "认知模式为交互因果力"的框架化 + 12000 篇论文支撑的模式库 + 规范性混淆的发现
- 实验充分度: ⭐⭐⭐⭐⭐ 多基线对比 + 消融 + 外部 benchmark + 人类对齐验证 + 规范性混淆案例
- 写作质量: ⭐⭐⭐⭐ 框架清晰，心理学理论与工程实现的衔接自然，但论文较长
- 价值: ⭐⭐⭐⭐⭐ 为 LLM 人格模拟提供了从标签映射到认知建模的范式转变，数据集和评估框架可独立复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods](the_silent_vote_improving_zero-shot_llm_reliability_by_aggregating_semantic_neig.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)

</div>

<!-- RELATED:END -->
