---
title: >-
  [论文解读] GA-S3: Comprehensive Social Network Simulation with Group Agents
description: >-
  [ACL 2025][Social Network Simulation] 提出基于"群体智能体"（Group Agent）的社交网络模拟系统 GA-S3，将具有相似行为的个体聚合为群体代理，通过层次化生成、马尔可夫网络推理和行为模块实现大规模社交网络的高效精确模拟。
tags:
  - ACL 2025
  - Social Network Simulation
  - Group Agent
  - LLM Agent
  - Network Traffic Prediction
  - Markov Network
---

# GA-S3: Comprehensive Social Network Simulation with Group Agents

**会议**: ACL 2025  
**arXiv**: [2506.03532](https://arxiv.org/abs/2506.03532)  
**代码**: [AI4SS/GAS-3](https://github.com/AI4SS/GAS-3)  
**领域**: NLP / 社会模拟 / LLM Agent  
**关键词**: Social Network Simulation, Group Agent, LLM Agent, Network Traffic Prediction, Markov Network

## 一句话总结

提出基于"群体智能体"（Group Agent）的社交网络模拟系统 GA-S3，将具有相似行为的个体聚合为群体代理，通过层次化生成、马尔可夫网络推理和行为模块实现大规模社交网络的高效精确模拟。

## 研究背景与动机

社交网络模拟旨在创建虚拟社交网络的表示，对用户行为、关系和信息流进行建模，以分析和预测社交互动的结果。这对群体行为涌现研究、政策优化和商业策略制定都有重要价值。

现有方法面临的核心挑战：

**个体级模拟的计算不可行**：社交网络有数十亿用户，为每个用户创建一个 LLM 智能体是不可能的。即使是最近的 S3 系统也只能用1000个固定智能体模拟

**传统方法过于简化**：离散事件模拟和系统动力学方法倾向于预测变量而非揭示因果机制，忽视了社会行为的异质性

**可扩展性差**：现有的 LLM-based 社会模拟系统通常针对特定事件设计，无法泛化到不同类型的网络事件

GA-S3 的核心创新是引入**群体智能体（Group Agent）**的概念：不再模拟每个个体，而是将行为相似的人聚合为一个群体，以群体为单位进行推理和行动。这在保持模拟真实性的同时大幅降低了计算成本。

## 方法详解

### 整体框架

GA-S3 系统由三个核心模块组成，对应智能体生命周期的三个阶段：

1. **层次化生成（Hierarchical Generation）**→ 存在（Existence）
2. **决策推理（Decision-Reasoning）**→ 决策（Decision）  
3. **行为（Action）**→ 行动（Behavior）

### 关键设计

1. **感知嵌入（Perception Embedding）**：当新的网络事件出现时，智能体首先感知事件内容，由 LLM 识别其领域（如教育、政治、商业）和所属国家。这些信息存入智能体的记忆模块，形成环境和内容的基础感知。

2. **层次化多叉树生成**：

    - 采用自顶向下的方式，将人群组织成多叉树结构
    - 每一层是上一层的细粒度划分，使用 LLM + prompt engineering 生成
    - 利用 RAG 技术：给定事件后，系统用 Kimi 模型联网搜索相关国家和领域的人口信息，存入本地知识图谱
    - 如果相同国家和领域已有数据，直接用 BFS 从知识图谱中检索
    - 例如：L1（学生/教师）→ L2（职业学生/教育者/...）→ L3（本科/硕士/全职教师/...），逐层细化至16个群体智能体

3. **智能体属性设计**：

    - **ID 和国家**：唯一标识
    - **人口数量**：决定互动频率和强度
    - **性格特征**：分为 "susceptible"（易感）、"ordinary"（普通）和 "calm"（冷静）三类，影响情绪波动幅度
    - **情绪**：包含 happy、sad、angry，用数值量化
    - **态度**：正面或负面

4. **马尔可夫网络推理**：

    - 状态转移方程：$P(S_i^t | S_i^{t-1}, \mathcal{E}_i^t, M_i^{t-1}) = \alpha_1 P(S_i^{t-1}) + \alpha_2 P(\mathcal{E}_i^t) + \alpha_3 P(M_i^{t-1})$
    - 情绪由 LLM 基于感知和先前状态更新：$P(\mathcal{E}_i^t | O_i^t, S_i^{t-1}) = \text{LLM}(O_i^t, S_i^{t-1})$
    - 决策通过 LLM 策略函数 $\pi$ 产生：$P(A_i^t | S_i^t) = \pi(S_i^t)$
    - 记忆采用队列更新机制，新信息替换旧信息，模拟人类注意力的短暂性

5. **四个细粒度真实性因子**：

    - **人口权重**：基于真实数据，影响群体活跃度
    - **性格特征**：控制情绪/态度波动幅度（易感>普通>冷静）
    - **情绪衰减**：情绪和态度随时间自然消退
    - **遗忘概率**：短期记忆中过去的感知和事件逐渐淡化

6. **行为模块**：支持五种在线行为——浏览、点赞、评论、分享和预测。浏览是主要行为，其他行为远少于浏览。真实用户群先浏览事件及其互动信息（浏览量、点赞数等），产生情绪和态度后再进行进一步互动。

### 损失函数 / 训练策略

GA-S3 不需要微调 LLM。直接使用开源 LLaMA3-8B（temperature=0.1 以确保可重复性），层次化生成使用 Kimi 模型（联网搜索）+ GPT-4（数据清洗）。系统通过四个管理器（事件管理器、记忆管理器、状态管理器、对象管理器）协调全部流程。

## 实验关键数据

### 主实验：与其他社交模拟系统的对比（表格）

| 方法 | t-test ↓ | MAPE ↓ | DTW Mean ↓ | DTW Std ↓ |
|------|----------|--------|------------|-----------|
| PSP（基于模型） | 1.310 | 69.12% | 3.40e+07 | 0.4207 |
| S3（基于智能体） | 1.820 | 68.66% | 3.09e+07 | 0.4035 |
| **GA-S3（本文）** | **0.389** | **16.48%** | **1.30e+07** | **0.1890** |

*GA-S3 在所有指标上大幅优于两个基线，MAPE 从约69%降至16.48%*

### 消融实验（表格）

| # | 层级 | 记忆 | 状态 | t-test ↓ | MAPE ↓ | DTW Mean ↓ |
|---|------|------|------|----------|--------|------------|
| 1 | L1 | ✓ | ✓ | 0.829 | 68.78% | 3.38e+07 |
| 2 | L2 | ✓ | ✓ | 0.603 | 33.73% | 2.84e+07 |
| 3 | L3 | ✗ | ✗ | 2.212 | 2884% | 7.80e+08 |
| 4 | L3 | ✓ | ✗ | 2.189 | 1339% | 1.39e+08 |
| 5 | L3 | ✗ | ✓ | 1.986 | 401% | 8.78e+07 |
| **6** | **L3** | **✓** | **✓** | **0.389** | **16.48%** | **1.30e+07** |

*层级越深越好（L3 >> L2 >> L1），记忆和状态模块都至关重要*

### 关键发现

1. **层次化生成的深度很关键**：L3（16个群体）的 MAPE 为16.48%，L1（2个群体）为68.78%——越细粒度的群体划分越能捕捉行为差异

2. **记忆和状态缺一不可**：去掉记忆后 t-test 从0.389升到1.986，去掉状态从0.389升到2.189。两者都没有时（L3/无/无）MAPE 高达2884%

3. **群体智能体具有行为多样性**：同一组群体智能体在不同事件中表现迥异，预测曲线紧密跟踪真实趋势

4. **情绪/态度与流量趋势部分对齐**：呈现相似的双峰模式，但不完全相关——这符合真实世界中情绪与行为的弱耦合特性

5. **可重复性优秀**：Z-score 始终低于1，表明实验结果高度稳定

6. **性格特征的消融**效果直观：移除后 "calm" 群体的态度值异常增大，不符合真实情况

## 亮点与洞察

- **群体智能体的概念很有创意**：在个体智能体和统计模型之间找到了一个优雅的中间地带，既保持了 LLM 的推理能力又控制了计算成本
- **自适应生成**：基于事件的领域和国家自动构建群体，而非手动设定，提高了可扩展性
- **四个细粒度因子**（人口权重、性格特征、情绪衰减、遗忘概率）使模拟更接近真实世界
- **自建基准** SNB 填补了现有社交网络模拟数据集缺乏细粒度流量变化信息的空白

## 局限与展望

1. **推理能力受限**：当前直接使用 LLM 输出决策，缺乏 Chain-of-Thought 等深度推理技术
2. **基准数据多样性有限**：仅30个事件，虽覆盖10个领域和多个国家，但规模偏小
3. **缺乏显式网络结构**：群体之间通过领域和地理边界产生隐式结构，但没有真正的社交网络拓扑
4. **群体生成不够灵活**：依赖固定的层次结构，未来可探索动态层级调整
5. **隐私与伦理**：虽然已匿名化处理，但社交网络模拟本身存在被用于舆论操控的风险

## 相关工作与启发

- **Generative Agents**（Park et al., 2023）：开创性地模拟个体行为，但无法扩展到大规模
- **S3**（Gao et al., 2023）：用 LLM 构建虚拟社交网络，使用1000个固定智能体
- **PSP**（Kong et al., 2018）：识别社交媒体流行度的阶段模式，使用模式匹配预测
- **Schelling 模型**（1971）：首个基于智能体的虚拟社会模拟
- GA-S3 的核心差异：**群体而非个体**作为模拟单元，支持**自适应生成**和**细粒度因子**

## 评分

- **新颖性**: ⭐⭐⭐⭐ 群体智能体的概念在社交模拟领域是有价值的创新，层次化生成+马尔可夫推理的结合设计精巧
- **实验充分度**: ⭐⭐⭐⭐ 消融实验全面（层级、记忆、状态、细粒度因子），多事件多维度评估，Z-score 验证可重复性
- **写作质量**: ⭐⭐⭐ 整体框架描述清晰，但公式较多且部分符号在文字中解释不够充分
- **价值**: ⭐⭐⭐⭐ 为大规模社交网络模拟提供了一条可行路径，开源代码和基准数据集有利于后续研究

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](consistent_client_simulation_for_motivational_interviewing-based_counseling.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)

<!-- RELATED:END -->
