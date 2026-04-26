---
title: >-
  [论文解读] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs
description: >-
  [CVPR 2026][人体理解][knowledge graph] 提出 Graph2Eval，利用从异构数据源构建的知识图谱作为结构化任务空间，通过子图采样、任务模板和 meta-path 策略自动生成语义一致且可解的多模态 agent 评估任务，生成的任务在语义一致性和可解性上分别提升 20% 和 17%。
tags:
  - CVPR 2026
  - 人体理解
  - knowledge graph
  - agent evaluation
  - task generation
  - 多模态
  - benchmark
---

# Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs

**会议**: CVPR 2026  
**arXiv**: [2510.00507](https://arxiv.org/abs/2510.00507)  
**代码**: [GitHub](https://github.com/YurunChen/Graph2Eval)  
**领域**: Agent 评估 / 知识图谱  
**关键词**: knowledge graph, agent evaluation, task generation, multimodal, benchmark

## 一句话总结

提出 Graph2Eval，利用从异构数据源构建的知识图谱作为结构化任务空间，通过子图采样、任务模板和 meta-path 策略自动生成语义一致且可解的多模态 agent 评估任务，生成的任务在语义一致性和可解性上分别提升 20% 和 17%。

## 研究背景与动机

**领域现状**：随着多模态 LLM 驱动的 agent 在自主性和泛化性上的不断进步，评估其真实能力变得至关重要。现有评估方式主要依赖静态数据集（如 GAIA、MiniWoB++、Mind2Web）或需要大量人工标注的环境（如 OSWorld、AndroidWorld）。

**现有痛点**：
1. 静态数据集无法区分 agent 的真正泛化能力与记忆检索能力，且扩展性差
2. 已有 LLM 合成方法（如 TaskCraft）缺乏显式的实体关系建模，生成的任务语义不一致、可解性差
3. 网页交互任务生成方法依赖静态数据和预定义页面关系，无法有效迁移到动态网页场景

**核心矛盾**：如何在无需大量人工标注的前提下，自动生成语义一致、可解且多样化的 agent 评估任务。

**本文目标**：构建一个自动化、可扩展、语义上有根据的 agent 任务生成框架，同时覆盖文档理解和网页交互两种场景。

**切入角度**：将知识图谱（KG）视为结构化任务空间，利用图谱中的实体和关系来约束任务生成，确保语义一致性和可解性。

**核心 idea**：用知识图谱编码数据中的实体关系，通过子图采样和模板驱动生成机制自动产出高质量 agent 评估任务。

## 方法详解

### 整体框架

Graph2Eval 的数据集生成流程包含五个阶段：**数据摄入** → **知识图谱构建** → **子图采样** → **任务生成** → **覆盖优化**。整体思路是先从文档/网页数据构建结构化知识图谱，再通过子图采样提取局部子图，最后结合任务模板和 LLM 生成具体任务实例。

### 关键设计

1. **数据摄入（Data Ingestion）**:
    - 对文档数据进行语义分块（Semantic Chunking），将文档切分为最小语义单元并映射为图节点
    - 使用 all-MiniLM-L6-v2 计算每个节点的 $d=384$ 维嵌入向量，存入向量数据库
    - 为每个节点标注元数据（文件路径、标题、作者等）
    - 对网页数据通过自动化 URL 爬取收集 DOM 结构和截图，集成模拟人类交互以处理复杂页面设计

2. **知识图谱构建（KG Construction）**:
    - 定义 KG 为 $G = (V, E, R)$，其中 $V$ 是节点集，$E$ 是边集，$R$ 是关系类型集
    - **节点提取**：解析文档/网页中的段落、标题、超链接、表单、按钮等元素，映射为节点 $V = \{v_i \mid v_i \in \text{Elements}(D), \text{type}(v_i) \in \text{NodeTypeSet}\}$
    - **节点表示**：将文本内容 $c_i^T$ 和视觉内容 $c_i^V$（通过 $\phi_{\text{visual}}$ 转为文本描述）拼接后编码为向量 $h_i = f_{\text{embed}}(c_i^{T+V})$
    - **边构建**：构建异构边集 $E = E_{\text{text}} \cup E_{\text{web}}$，文本边编码结构关系（包含/序列）、语义关系、上下文关系和引用关系；网页边编码导航关系、交互关系和布局关系

3. **子图采样（Subgraph Sampling）**:
    - **文档理解模式**：基于语义相关性（余弦相似度 $\cos(h_i, h_g) > \tau$）和结构匹配（$\text{StructMatch}$）选择节点，仅保留指定类型的节点
    - **网页交互模式**：采用种子驱动策略，先识别任务特定种子节点 $S_{\text{seed}}(g)$（按钮、表单等），再收集种子节点的 $k$-hop 邻居
    - 最终子图 $G_g = (V_g, E_g, R) \subseteq G$ 包含所有选中节点及其内部连边

4. **任务生成（Task Generation）**:
    - **文档理解任务**：维护任务模板库（覆盖问答、比较、分析、推理等），从采样子图中提取模板变量，结合 LLM 生成具体任务实例
    - **网页交互任务**：提出种子驱动子图采样策略，先识别页面关键操作节点作为"任务种子"，再通过 meta-path 匹配产生具体任务链（如 Search → Filter → Detail），最后由 LLM 结合子图结构和页面上下文生成任务

5. **覆盖优化（Coverage Optimization）**:
    - 使用多阶段优化确保任务质量、多样性和代表性
    - 基于 Maximal Marginal Relevance (MMR) 策略迭代选择任务，平衡覆盖度和新颖度
    - 覆盖维度包括：节点类型、边类型、模式、页面级别、网站类型和难度

### 损失函数 / 训练策略

本工作不涉及模型训练，而是一个任务生成框架。任务生成和优化基于 GPT-4o，评估使用多种模型（GPT-4o、Deepseek-V3、Qwen2.5-VL 系列、Gemini 2.5 Flash 等）。生成效率方面，文档理解任务平均 34.87 秒/个，网页交互任务平均 95.51 秒/个。

## 实验关键数据

### 主实验

| 模型 | 设置 | F1 | ROUGE-L | LLM Judge |
|------|------|-----|---------|-----------|
| GPT-4o | Single Agent | 0.5766 | 0.4874 | 0.7854 |
| GPT-4o | Multi-Agent | 0.5916 | 0.4873 | 0.7623 |
| Deepseek-V3 | Single Agent | 0.5376 | 0.4518 | **0.8351** |
| Deepseek-V3 | Multi-Agent | 0.5497 | 0.4635 | 0.7984 |
| Qwen2.5-VL-72B | Single Agent | 0.5730 | 0.4837 | 0.7094 |
| Qwen2.5-VL-7B | Single Agent | 0.2093 | 0.1939 | 0.5427 |

Web 交互任务（Agent S 2.5 整体 Success Rate）：

| 模型 | Overall SR |
|------|-----------|
| Gemini 2.5 Flash | **69.20%** |
| Qwen2.5-VL-72B | 38.80% |
| GPT-4o mini | 33.12% |
| UI-TARS-1.5-7B | 7.19% |

### 消融实验

| 方法 | Doc Consistency | Doc Solvability | Web Consistency | Web Solvability |
|------|----------------|-----------------|-----------------|-----------------|
| Graph2Eval w/o KG | 0.74 | 0.73 | 0.62 | 0.60 |
| Graph2Eval (完整) | **0.95** (+20%) | **0.93** (+17%) | **0.78** | **0.72** |

Agent 评估消融（Qwen2.5-VL-72B）：

| 方法 | Doc Acc | Web SR |
|------|---------|--------|
| w/o KG | 0.68 | 0.12 |
| Graph2Eval | **0.85** | **0.24** |

### 关键发现

1. 知识图谱的引入显著提升了任务语义一致性（+20%）和可解性（+17%），KG 边精度达 88%
2. 无 KG 的 baseline 生成的网页任务大多局限于单页交互，多页工作流因缺乏页间关系建模而不可解
3. Graph2Eval-Bench 能有效区分不同规模模型的性能差异（如 Qwen-72B vs 7B）
4. 与 TaskCraft 的自底向上方法相比，Graph2Eval 的自顶向下范式（先建 KG 再采样）生成的任务类型更丰富

## 亮点与洞察

1. **KG 作为任务空间**的思路很有创意——将任务生成问题转化为图上的子图采样问题，天然保证了语义一致性
2. **统一框架覆盖两类场景**：文档理解（RAG Agent）和网页交互（Web Agent），通过统一的图抽象实现
3. **种子驱动 + meta-path 策略**使网页任务生成具有组合灵活性，避免了全有或全无的刚性约束
4. 验证了当前 agent 在动态、自动生成的任务上仍有很大提升空间（最强模型 Agent S 2.5 也只有 69% SR）

## 局限与展望

1. KG 构建质量高度依赖于数据预处理和实体/关系提取的准确性，边精度为 88% 而非 100%
2. 当前仅覆盖文档理解和网页交互两类场景，未涵盖工具使用、多模态推理等更广泛的 agent 任务
3. 任务生成依赖 GPT-4o，成本较高且可能引入模型偏差
4. 知识图谱的动态更新机制未详细讨论，如何应对网页内容变化仍是挑战
5. 生成任务的难度分布控制有限，可能集中在中等难度

## 相关工作与启发

- **TaskCraft**：自底向上的原子任务组合方法，但缺乏显式关系建模
- **OSWorld / MiniWoB++**：依赖人工标注的环境型 benchmark，扩展性受限
- **GAIA / MMBench**：静态 QA 数据集，无法评估动态交互能力
- **启发**：KG 驱动的任务生成范式可推广到其他 agent 评估领域（如代码生成、多工具协同等）

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 综合 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents](gardendesigner_encoding_aesthetic_principles_into_jiangnan_garden_construction_v.md)
- [\[CVPR 2026\] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)
- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)

<!-- RELATED:END -->
