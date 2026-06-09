---
title: >-
  [论文解读] What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects
description: >-
  [NeurIPS 2025][LLM Agent][AI agent polling] 提出基于LLM的AI agent民意调研框架，通过人口统计合成虚拟居民agent对数据中心项目进行大规模低成本民调，跨模型跨地区实验表明agent意见与真实民调在主题上高度一致。
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "AI agent polling"
  - "public opinion"
  - "data center"
  - "LLM simulation"
  - "community engagement"
---

# What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects

**会议**: NeurIPS 2025  
**arXiv**: [2511.22037](https://arxiv.org/abs/2511.22037)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: AI agent polling, public opinion, data center, LLM simulation, community engagement  

## 一句话总结
提出基于LLM的AI agent民意调研框架，通过人口统计合成虚拟居民agent对数据中心项目进行大规模低成本民调，跨模型跨地区实验表明agent意见与真实民调在主题上高度一致。

## 研究背景与动机
**领域现状**: AI基础模型驱动的算力需求推动全球数据中心建设热潮，这些设施为当地社区带来就业、税收等经济收益，同时也带来水资源消耗、电力压力、碳排放等环境负担。

**现有痛点**: 传统民意调查成本高昂（需大量人力和资金），且通常在项目承诺已做出后的公听会阶段才收集反馈，无法在规划早期融入社区声音。社区参与度有限、样本偏差等问题进一步削弱了反馈的代表性。

**核心矛盾**: 数据中心建设需要早期、可扩展、多元化的社区意见输入，但缺乏有效机制来实现这一点。

**本文目标**: 如何低成本、大规模地在数据中心项目规划早期获取具有代表性的社区公众意见？

**切入角度**: 利用LLM的推理和世界知识能力创建AI agent来模拟社区居民，进行可扩展的民意模拟调查。

**核心 idea**: 用人口统计数据合成的虚拟居民agent + 多模型LLM驱动的结构化问卷调查 = 可扩展的社区意见早期筛查工具。

## 方法详解

### 整体框架
框架包含六个关键阶段：(1) 建立数据中心提案；(2) 使用IPF从县级人口统计数据生成代表性AI agent样本；(3) 构建社区画像和项目规格作为系统提示；(4) 使用多个LLM（GPT-5, Gemini-2.5-Pro, Qwen-Max）进行结构化民调；(5) 可选的共形预测校准；(6) 多层级分析（跨模型、跨地区、人类民调对比）。

### 关键设计
1. **区域上下文建模（Regional Context Modeling）**: 包含三个层次的背景信息——州级数据中心电力消耗数据、县级人口统计画像（通过Census Bureau ACS 5年估计API获取）、标准化的数据中心项目描述（能耗、环境影响、经济影响）。这三者共同构成提供给每个agent的"全局上下文"。
2. **虚拟Agent构建（Virtual Agent Construction）**: 从ACS提取12个维度的人口统计分布（教育水平、婚姻状况、家庭语言、公民身份、就业状态、家庭收入、住房、车辆数、年龄组、性别、种族、民族），使用迭代比例拟合（IPF）算法从边际分布合成联合概率分布，生成具有相关人口特征的虚拟agent。通过卡方拟合优度检验验证agent群体的人口统计分布。
3. **AI Agent民调执行（AI Agent Polling）**: 问卷包含13个问题（12个单选/多选 + 1个开放文本），覆盖5个核心领域：经济影响、环境关切、社区参与、预期个人影响、总体项目支持。采用系统消息/用户提示分离策略优化API调用（静态区域上下文作为system message可被缓存）。使用三个LLM的batch API并行处理上千agent响应。
4. **共形预测校准（Conformal Prediction Calibration）**: 提出使用少量真实调查数据计算非一致性分数，确定阈值后为新agent调查结果构建置信区间，提供统计保证。（本文因资源限制未实施，作为方法论完整性提出。）

### 训练策略
- 无需训练，完全基于LLM的零样本推理
- 每次实验每个模型/地区调查1000个虚拟agent
- 总API成本约$36.2/次（GPT-5: $23.3, Gemini-2.5: $11.2, Qwen: $1.7）
- 单次运行通常超过24小时

## 实验关键数据

### 基线案例分析（Taylor County, TX, GPT-5）

| 维度 | 关键结果 |
|------|---------|
| 总体态度 | 54.2% 中立，43.6% 积极，2.2% 消极 |
| 经济影响认知 | 80% 认为影响是混合的，20% 认为积极 |
| 环境关切度 | 97% 表示担忧 |
| 政府监管信任 | 60% 中立，40% 不信任 |
| 开放文本Top3主题 | 水资源保护、公用事业成本、本地就业 |

### 跨模型对比（Taylor County, n=1000）

| 维度 | GPT-5 | Gemini-2.5 | Qwen-Max |
|------|-------|------------|----------|
| 经济态度-积极 | ~20% | ~10% | 91% |
| 经济优先项 | 多元化偏好 | 多元化偏好 | 几乎一致选税收和就业 |
| 政府不信任比例 | 40% | 32% | ~0% |
| 信息来源偏好 | 学术研究为主 | 学术研究为主 | 学术研究 + 强烈偏好地方政府 |

### 跨地区对比（GPT-5, n=1000）

| 维度 | Taylor County, TX | Loudoun County, VA |
|------|-------------------|-------------------|
| 净支持率 | +41%（支持44%, 反对2%） | -5%（支持10%, 反对15%） |
| 增加支持条件Top1 | 低公用事业费（94%） | 更严格监管（90%） |
| 经济态度-积极 | 20% | 较低 |
| 独特经济担忧 | 低收入+高温→关注电费 | 公共服务压力（已有大量数据中心经验） |

### 与真实民调对比（Heatmap News全国民调）

| 指标 | Heatmap全国民调 | AI Agent民调 |
|------|----------------|-------------|
| 净支持率 | +2%（44%支持 vs 42%反对） | Taylor: +41%, Loudoun: -5% |
| 主要收益 | 税收 + 高薪就业 | 税收 + 基础设施升级 |
| 主要担忧 | 水资源 + 电力使用 | 水资源消耗 + 公用事业账单上涨 |

### 关键发现
- **水资源和公用事业成本**是跨地区、跨模型的共同核心担忧，税收收入是共同认可的主要收益
- **模型差异反映训练数据中的文化/制度差异**: Qwen对经济发展更乐观、对政府更信任，可能反映其训练数据中发展导向的经济范式
- **地区差异符合现实**: Taylor County（低收入、有限就业）更关注经济收益；Loudoun County（数据中心密集区）关注监管和公共服务压力
- AI agent民调与真实民调在**主题层面高度对齐**，尽管量化值不可直接比较

## 亮点与洞察
- **首次将AI agent民调框架应用于数据中心公众意见评估**，填补了AI基础设施社区参与的空白
- **跨模型对比揭示LLM内嵌偏见**: 不同LLM的"文化基因"导致同一人口对同一项目的意见截然不同，这本身就是重要的发现
- **成本效益显著**: 总API成本仅$36.2/次，远低于传统民调的人力物力投入
- **IPF合成agent群体的方法论严谨**: 12维人口统计 + 卡方检验验证，确保代表性
- **披露部分（Disclosure）的诚实性**: 主动承认AI agent意见不能替代真实社区声音，定位为筛查工具

## 局限与展望
- **共形预测校准未实施**: 核心方法论组件因资源不足缺席，统计保证无法验证
- **仅测试两个县**: 均已有数据中心基础设施，对无数据中心经验的社区适用性未知
- **LLM偏见问题**: 训练数据偏见会系统性影响agent输出，racial/cultural偏见难以完全消除
- **缺乏纵向验证**: 未验证agent民调是否能预测未来真实社区反应
- **agent认知模型局限**: 无法模拟神经多样性视角、"参与沉默"现象等人类特有的复杂行为
- 可改进：引入更多LLM进行交叉验证、实施校准机制、扩展到其他基础设施领域

## 相关工作与启发
- **vs 传统民调**: 传统方法成本高、时效差，本文框架可扩展且低成本，但缺乏真实人类互动的深度
- **vs LLM选举模拟**: 政治领域的AI agent框架关注选举预测，本文首次聚焦物理基础设施（数据中心），问题结构更多元（经济/环境/治理）
- **vs PolicyPulse等公众参与工具**: 这些工具从已有公众讨论中提取主题，本文是主动生成意见，属于不同范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将AI agent polling应用于数据中心社区意见，切入点新颖且有现实意义
- 实验充分度: ⭐⭐⭐ 跨模型跨地区分析有说服力，但仅两个县+校准未实施，规模偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Disclosure部分坦诚，方法论严谨但易读
- 价值: ⭐⭐⭐⭐ 为AI基础设施的负责任部署提供了实用工具原型，具有政策参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[NeurIPS 2025\] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)
- [\[NeurIPS 2025\] SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)
- [\[ICML 2025\] xChemAgents: Agentic AI for Explainable Quantum Chemistry](../../ICML2025/llm_agent/xchemagents_agentic_ai_for_explainable_quantum_chemistry.md)

</div>

<!-- RELATED:END -->
