---
title: >-
  [论文解读] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning
description: >-
  [NeurIPS 2025][人体理解][持续学习] 提出 K-DeCore 框架，通过知识解耦将结构化知识推理分为任务无关的 schema 过滤和任务特定的 query 构建两阶段，配合双视角记忆构建和结构引导的伪数据合成策略，在固定参数量下实现跨异构 SKR 任务的有效知识迁移。
tags:
  - NeurIPS 2025
  - 人体理解
  - 持续学习
  - 结构化知识推理
  - 知识解耦
  - 记忆回放
  - Text-to-SQL
---

# K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2509.16929](https://arxiv.org/abs/2509.16929)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 持续学习, 结构化知识推理, 知识解耦, 记忆回放, Text-to-SQL

## 一句话总结

提出 K-DeCore 框架，通过知识解耦将结构化知识推理分为任务无关的 schema 过滤和任务特定的 query 构建两阶段，配合双视角记忆构建和结构引导的伪数据合成策略，在固定参数量下实现跨异构 SKR 任务的有效知识迁移。

## 研究背景与动机

结构化知识推理（SKR）需要将自然语言问题转换为结构化查询（SQL、SPARQL、TOP 等），覆盖数据库、知识图谱和对话状态等多种结构化知识。现有方法的关键局限：

**静态假设不现实**：现有方法假设 SKR 任务单一且不变，但实际场景（如 Siri/Alexa）需要持续适应新的推理任务

**异构泛化差**：专注于单一类型（如仅 text-to-SQL）的持续学习方法无法跨异构结构化知识泛化

**参数增长问题**：基于 PEFT 的方法为每个任务分配独立参数，导致参数规模随任务数线性增长

作者的核心观察：**schema 过滤**（从完整 schema 中筛选与查询相关的元素）是跨 SKR 任务的共享可复用组件，其输入输出格式相对稳定，适合做知识迁移的桥梁。

## 方法详解

### 整体框架

K-DeCore 包含一个冻结的骨干 LLM 和三个轻量 PEFT 模块（LoRA）：

- **$\mathbf{P}_a$**：Schema 过滤器（任务无关，跨任务共享）
- **$\mathbf{P}_b$**：Query 构建器（任务特定，但固定参数量）
- **$\mathbf{P}_c$**：伪问题生成器（用于合成训练数据）

### 关键设计

#### 1. 知识解耦：Schema 过滤 + Query 构建

**Schema 过滤**（任务无关）：

将所有异构 schema 统一为 DB-like 格式。例如：

- 数据库：table → $\phi$，column → $\psi$
- 知识图谱：entity type → $\phi$，relation → $\psi$
- 对话状态：intent → $\phi$，slot → $\psi$

统一后的文本表示：$\widetilde{\Omega} = \phi_1: \psi_1^1, \ldots | \phi_2: \psi_2^1, \ldots$

训练 $\mathbf{P}_a$ 预测有用 schema 子集 $\Omega^*$，每个新任务从上一任务的 checkpoint 初始化。

**Query 构建**（任务特定）：

$\mathbf{P}_b$ 负责从原始 schema $\mathcal{S}$ 和过滤后的 $\mathcal{S}^*$ 生成最终查询 $\mathcal{Y}$。同时保留原始格式以生成可执行查询。

两阶段的优势：
- Schema 过滤跨任务格式统一 → 促进前向/后向知识迁移
- Query 构建保留任务特异性 → 保持查询生成精度

#### 2. 双视角记忆构建

**Schema 引导记忆 $\mathcal{M}_a^k$**：

按 schema 相似度聚类，选择最接近簇中心的样本。距离定义：$d_1(\mathcal{X}_1, \mathcal{X}_2) = \cos(g(\mathcal{S}_1^*), g(\mathcal{S}_2^*))$

确保记忆覆盖不同 schema 模式。

**结构引导记忆 $\mathcal{M}_b^k = \mathcal{M}_{\text{real}}^k \cup \mathcal{M}_{\text{pseudo}}^k$**：

- $\mathcal{M}_{\text{real}}^k$：按查询结构相似度聚类选择代表性样本
- $\mathcal{M}_{\text{pseudo}}^k$：通过伪数据合成引入训练集中未出现的新结构

#### 3. 结构引导伪数据合成（Algorithm 1）

核心流程：
1. 从训练集中随机采样 $T$ 个查询结构
2. 用骨干 LLM 合成新的查询结构（可能是复杂 SQL 或多跳 S-expression）
3. 随机选择 schema 填充占位符，生成具体查询
4. **执行验证**：仅保留能成功执行的合成查询
5. 用 $\mathbf{P}_c$ 生成对应的自然语言问题

### 损失函数 / 训练策略

**Schema 过滤损失**：

$$\mathcal{L}(\mathcal{D}^k; \theta, \mathbf{P}_a) = -\sum_i \sum_j \log P(\omega_j^* | \mathcal{Q}_i, \widetilde{\Omega}_i, \omega_{<j}^*) + \sum_{k'=1}^{k-1} \mathcal{L}(\mathcal{M}_a^{k'})$$

**Query 构建损失**：

$$\mathcal{L}(\mathcal{D}^k; \theta, \mathbf{P}_b) = -\sum_i \sum_j \log P(y_j | \mathcal{Q}_i, \mathcal{S}_i^*, \mathcal{S}_i, y_{<j}) + \sum_{k'=1}^{k-1} \mathcal{L}(\mathcal{M}_b^{k'})$$

两个损失均包含当前任务 + 历史记忆回放。记忆大小 $|\mathcal{M}_a^k| = |\mathcal{M}_b^k| = 5$，真实:伪造比例 = 4:1。

训练超参：batch size 12，lr $5 \times 10^{-5}$，5 epochs，单卡 RTX 4090。

## 实验关键数据

### 主实验（3 条 SKR 任务流 × 3 个骨干模型）

| 骨干 | 方法 | Stream1 AA | Stream1 BWT | Stream1 FWT |
|------|------|-----------|-------------|-------------|
| T5-Large | Fine-Tuning | 2.9 | -31.1 | 6.3 |
| T5-Large | SFNet | 27.3 | -14.7 | 3.6 |
| T5-Large | C3 | 26.6 | — | -1.9 |
| T5-Large | **K-DeCore** | **31.8** | **-9.6** | **8.6** |
| Llama3-8B | C3 | 39.7 | — | 2.3 |
| Llama3-8B | **K-DeCore** | **40.5** | -16.7 | **5.9** |
| QWEN2.5-7B | C3 | 38.9 | — | 1.9 |
| QWEN2.5-7B | **K-DeCore** | **43.2** | **-8.2** | **6.9** |

### 消融实验（Llama3-8B, Stream1）

| 变体 | AA | BWT | FWT |
|------|-----|------|------|
| **K-DeCore** | **40.5** | -16.7 | **5.9** |
| w/o Decoupling | 38.8 | **-13.8** | 2.6 |
| w/o Unification | 37.8 | -17.7 | 3.4 |
| w/o Replay | 20.5 | -41.2 | 4.3 |
| w/o $\mathcal{M}_b^k$ | 20.8 | -42.1 | 5.3 |
| w/o $\mathcal{M}_a^k$ | 39.4 | -17.1 | 5.1 |
| w Random Memory | 39.9 | -17.4 | 5.6 |

### 关键发现

1. **记忆回放是核心**：去掉回放（w/o Replay）AA 从 40.5 暴跌到 20.5，BWT 跌至 -41.2
2. **Query 记忆远比 Schema 记忆重要**：去掉 $\mathcal{M}_b^k$（AA=20.8）比去掉 $\mathcal{M}_a^k$（AA=39.4）影响大得多
3. **知识解耦和统一格式均有贡献**：去掉解耦（AA-1.7）或统一表示（AA-2.7）均导致性能下降
4. **伪数据比例 20% 最优**：过多伪数据反而降低性能
5. **训练效率好**：K-DeCore 训练时间仅略高于 EMAR，远低于 C3 和 SAPT

## 亮点与洞察

1. **首个异构 SKR 持续学习框架**：不同于只做 text-to-SQL 的方法，K-DeCore 跨数据库、知识图谱和对话状态
2. **Schema 统一的巧思**：将所有结构化知识统一为 DB-like 格式，用极简映射规则桥接异构性
3. **执行验证的伪数据合成**：通过实际执行筛选合成查询，确保语义有效性
4. **固定参数量**：无需随任务数增长参数，优于 C3/SAPT 等方法

## 局限与展望

1. 每个任务仅 1000 训练样本 + 300 测试样本的低资源设定，未验证大规模场景
2. 记忆大小固定为 5 个样本，可能在更复杂任务流中不足
3. 未使用推理型 LLM（如 QWQ-32B）作为骨干，可能错失更强推理能力
4. Schema 统一为 DB-like 格式可能丢失某些结构化知识的特有语义信息
5. 伪数据合成依赖 LLM 的查询结构理解能力，对新型查询语言可能泛化不足

## 相关工作与启发

- **C3**：为每个任务训练独立 PEFT 模块，AA 有竞争力但参数随任务增长、FWT 差
- **SAPT**：利用 soft prompt 做持续学习，但训练开销大
- **SFNet**：基于回放的方法，但未解耦推理阶段
- **启发**：知识解耦的思想可推广到其他多阶段推理的持续学习场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 知识解耦 + 双视角记忆 + 结构引导合成的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 4 数据集 × 3 骨干 × 3 任务流，消融全面
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，方法动机充分
- 价值: ⭐⭐⭐⭐ — 填补了异构 SKR 持续学习的空白

<!-- RELATED:START -->

## 相关论文

- [One-Shot Knowledge Transfer for Scalable Person Re-Identification](../../ICCV2025/human_understanding/one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)
- [Mixing Expert Knowledge: Bring Human Thoughts Back to the Game of Go](mixing_expert_knowledge_bring_human_thoughts_back_to_the_game_of_go.md)
- [PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](../../CVPR2026/human_understanding/pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)
- [Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)

<!-- RELATED:END -->
