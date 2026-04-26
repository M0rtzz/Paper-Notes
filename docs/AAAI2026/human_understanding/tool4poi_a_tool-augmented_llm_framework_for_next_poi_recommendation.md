---
title: >-
  [论文解读] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation
description: >-
  [AAAI 2026][人体理解][POI推荐] 提出 Tool4POI，首个工具增强 LLM 范式用于开放集 POI 推荐，通过偏好提取、多轮工具检索和重排序模块，在 OOH 场景实现 40% 准确率（现有方法为 0%）。
tags:
  - AAAI 2026
  - 人体理解
  - POI推荐
  - LLM Agent
  - 工具增强
  - 位置推荐
---

# Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation

**会议**: AAAI 2026  
**arXiv**: [2511.06405](https://arxiv.org/abs/2511.06405)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: POI推荐, 工具增强LLM, Agent, 开放集推荐, 位置服务

## 一句话总结
本文首次将工具增强 LLM 范式引入下一个 POI 推荐任务，通过偏好提取、多轮候选检索和重排序三个模块，使 LLM 能从全量 POI 池中检索推荐，在 Out-of-History (OOH) 场景下实现 40% 准确率（现有方法为 0%），Acc@5/10 平均提升 20%/30%。

## 研究背景与动机

**领域现状**：下一个 POI（兴趣点）推荐是位置服务的核心任务。传统方法（RNN/Transformer/GCN）通过嵌入表示建模用户轨迹序列。近年 LLM 的上下文推理能力被引入该任务，如 LLM-Mob（上下文学习）、LLM4POI（监督微调），展现了对时空动态的理解潜力。

**现有痛点**：LLM-based 方法面临两个根本性限制：(1) 强依赖历史完整性——仅能从用户已访问的 POI 中推荐，无法处理 Out-of-History (OOH) 场景（用户将要访问从未去过的地方），而现实中 OOH 场景占比超 30%；(2) 受限于上下文窗口——一个城市可能有数十万 POI，无法将所有候选编码到提示中，无法进行开放集推荐。

**核心矛盾**：用户行为既有规律性（通勤模式）又有探索性（尝试新餐厅），现有 LLM 方法过拟合到已访问 POI，无法支持探索行为。微调方法（如 GNPR-SID）更是加剧了这种偏向。

**本文目标**：设计一个即插即用、无需微调的框架，使 LLM 能通过外部工具从全量 POI 池中检索推荐，突破 OOH 和大规模候选空间的限制。

**切入角度**：观察到人类选择目的地时会依次过滤（按类别、区域、距离排序），这种渐进缩小候选范围的过程可以用 LLM Agent 的多轮工具调用来模拟。

**核心 idea**：赋予 LLM 外部工具调用能力，通过偏好提取→多轮工具检索→重排序的三阶段管线，实现开放集 POI 推荐。

## 方法详解

### 整体框架
Tool4POI 包含三个模块，所有模块基于 Qwen2.5-14B，无需微调即可即插即用：(1) 偏好提取模块——从用户长期签到历史中提取区域、类别、时间三个维度的偏好；(2) 工具增强候选检索模块——LLM 作为检索 Agent 与6个外部工具多轮交互，从全量 POI 池中检索相关候选；(3) 重排序模块——根据用户近期签到行为对候选进行排序，反映当前意图。

### 关键设计

1. **偏好提取模块（Preference Extraction）**:

    - 功能：从用户历史签到轨迹中提取长期偏好的结构化表示
    - 核心思路：设计结构化提示，将用户按时间排列的签到序列（经 Google Maps Plus Code 转换为区域编码）输入 LLM，要求从 Region、Category、Time 三个维度输出偏好关键词。Plus Code 将经纬度坐标聚合为区域级编码，使空间相近的 POI 共享相同编码，简化地理特征表示
    - 设计动机：历史数据量大但包含丰富的隐含规律，LLM 的推理能力适合从中提取多维度偏好摘要

2. **工具增强候选检索模块（Candidate Retrieval）**:

    - 功能：使 LLM 能从全量 POI 池中自主检索相关候选，突破上下文窗口限制
    - 核心思路：定义6个外部工具：Query 工具（getPOIinfo 获取 POI 元数据）、检索工具（filterByCategories/filterByRegions 按类别/区域过滤）、辅助工具（findPotential 基于 POI 级协同过滤生成初始候选、sortByDistance 按距离排序）、控制工具（finish 终止检索）。LLM 作为 RetrievalAgent 根据偏好自主决定工具调用顺序和参数。终止条件：显式调用 finish、候选集小于阈值 $\tau=10$、或达到最大调用次数 $K=6$
    - 设计动机：(1) findPotential 通过有向共现图 $G=(\mathcal{P}, \mathcal{E})$ 引入群体行为先验，使 OOH POI 也能被检索到；(2) 多轮交互模拟人类决策过程，每轮缩小候选空间，最终得到高质量候选集

3. **重排序模块（Reranking）**:

    - 功能：根据用户近期行为对检索候选进行重排序，捕捉短期意图
    - 核心思路：将用户近期签到轨迹 $R_u$、目标时间 $t_{i+1}$ 和候选集 $\mathcal{C}$ 一起输入 LLM，要求根据最近的签到规律对候选 POI 按访问可能性排序。使用自然语言推理而非嵌入相似度
    - 设计动机：偏好提取捕获长期兴趣，重排序捕获短期动态（季节变化、生活阶段转换等），两者互补

### 损失函数 / 训练策略
Tool4POI 完全 training-free，不需要任何微调。推理时各模块顺序执行。检索模块中 Top-20 候选送入重排序模块。

## 实验关键数据

### 主实验

| 方法 | NYC Acc@5 | NYC Acc@10 | TKY Acc@5 | TKY Acc@10 | CA Acc@5 | CA Acc@10 |
|------|-----------|------------|-----------|------------|----------|-----------|
| Tool4POI | **0.6346** | **0.7623** | 最优 | 最优 | 最优 | 最优 |
| GNPR-SID (FT LLM) | 低 | 低 | 低 | 低 | 低 | 低 |
| GETNext | 0.4815 | 0.5811 | 0.4045 | 0.4961 | 0.3278 | 0.3946 |
| STAN | 0.4582 | 0.5734 | 0.3798 | 0.4464 | 0.2348 | 0.3018 |

### 消融实验

| 配置 | All Acc@1 | All Acc@10 | OOH Acc@1 | OOH Acc@10 |
|------|-----------|------------|-----------|------------|
| Tool4POI (完整) | 0.3164 | 0.7623 | 0.0522 | 0.5863 |
| 去掉检索模块 | 0.2545 | 0.5559 | 0 | 0 |
| 去掉重排序模块 | 0.1655 | 0.7145 | 0.0963 | 0.6024 |

### 关键发现
- OOH 场景下现有 LLM 方法准确率为 0%，Tool4POI 达到 40%+ Acc@10，证明工具增强检索的关键价值
- 检索模块对 Top-k 推荐贡献最大（引入多样候选），重排序对 Top-1 精准推荐贡献最大（捕捉当前意图）
- 在稀疏数据集 CA（平均仅10条签到）上提升最为显著（高达100%），展示数据稀疏场景下的鲁棒性
- 模型规模效应：即使用 3B 模型也超越 7B 微调方法，说明工具增强比模型规模更重要

## 亮点与洞察
- 首次将 Agent 工具调用范式引入推荐系统，开创了推荐系统与 LLM Agent 结合的新方向。findPotential 工具利用 POI 共现图作为集体先验特别精巧
- Training-free + plug-and-play 设计使其可直接应用于任何城市和数据集，无需重新训练，具有极强实用性
- 多轮交互式缩小候选范围的设计模拟了人类选择目的地的认知过程，这种思路可迁移到其他开放集推荐场景（如商品推荐、内容推荐）

## 局限与展望
- 检索质量依赖 LLM 的工具调用准确性，较小模型可能出错导致工具链失效
- 6个工具的设计较为手工化，更多领域特定工具（如天气查询、活动日历）可能进一步提升性能
- OOH 场景下重排序反而可能降低性能（因缺乏上下文信息），可考虑自适应地决定是否执行重排序
- 推理延迟较高（多轮 LLM 调用），实时推荐场景下需要优化

## 相关工作与启发
- **vs LLM4POI**: 监督微调 LLM 做 QA 式推荐，严重过拟合到高频 POI。Tool4POI 无需训练，通过工具检索开放集候选
- **vs GNPR-SID**: 用语义 ID 训练 LLM，但仍无法推荐 OOH POI。Tool4POI 的 findPotential 工具利用群体行为突破个人历史限制
- **vs 传统方法(GETNext等)**: 固定嵌入表示缺乏灵活性。Tool4POI 利用 LLM 的推理能力进行上下文感知的推荐

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个工具增强LLM POI推荐框架，开创性地解决OOH问题
- 实验充分度: ⭐⭐⭐⭐ 三个数据集评测全面，IH/OOH分析深入
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐⭐ 对推荐系统+LLM Agent领域有开创性贡献

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](../../ACL2026/human_understanding/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[AAAI 2026\] Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/human_understanding/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)

<!-- RELATED:END -->
