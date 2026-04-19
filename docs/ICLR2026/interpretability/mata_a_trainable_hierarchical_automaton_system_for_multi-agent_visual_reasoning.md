---
title: >-
  [论文解读] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning
description: >-
  [ICLR 2026][多Agent系统] 提出MATA（Multi-Agent hierarchical Trainable Automaton），将多Agent视觉推理建模为层次有限状态自动机，顶层状态转移由可训练的hyper agent（基于LLM的状态控制器）学习，每个Agent内部使用规则化的子自动机，通过共享内存实现协作与竞争，在多个视觉推理基准上达到SOTA。
tags:
  - ICLR 2026
  - 多Agent系统
  - 层次有限状态自动机
  - 视觉推理
  - 可训练状态控制器
  - 协作与竞争
---

# MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning

**会议**: ICLR 2026  
**arXiv**: [2601.19204](https://arxiv.org/abs/2601.19204)  
**代码**: [GitHub](https://github.com/ControlNet/MATA)  
**领域**: 多模态VLM  
**关键词**: 多Agent系统, 层次有限状态自动机, 视觉推理, 可训练状态控制器, 协作与竞争

## 一句话总结

提出MATA（Multi-Agent hierarchical Trainable Automaton），将多Agent视觉推理建模为层次有限状态自动机，顶层状态转移由可训练的hyper agent（基于LLM的状态控制器）学习，每个Agent内部使用规则化的子自动机，通过共享内存实现协作与竞争，在多个视觉推理基准上达到SOTA。

## 研究背景与动机

视觉推理要求模型解读视觉场景中实体间的关系。当前方法存在以下问题：

**端到端VLM**：隐式推理过程难以审计，在涉及空间关系、计数等复杂查询时容易产生幻觉

**组合式方法**（如ViperGPT、HYDRA）：提高了可解释性，但多数采用单Agent或手工设计的流水线

**多Agent方法**：各Agent被分配不相交角色并硬编码管道连接，无法处理错误传播，不支持功能重叠Agent间的竞争

**规则转移的僵化性**：手写规则转移函数随状态增长变得难以定义

核心问题：**如何让系统学会在何时调用哪个Agent？** 作者将这个决策问题建模为有限状态自动机的转移函数学习。

## 方法详解

### 整体框架

MATA是一个层次Mealy机 $\mathcal{M}_\theta = (S, S_0, \Sigma, \Lambda, \delta_\theta, \Gamma)$，包含两层结构：
- **顶层（Hyper Automaton）**：状态是各个Agent，转移函数 $\delta_\theta$ 由可训练的LLM控制器学习
- **底层（Sub-Automaton）**：每个Agent内部是规则化的小型状态机，确保可靠的微控制

### 关键设计

#### 1. 状态定义

状态集 $S = S_{\text{agent}} \cup S_{\text{life}}$，其中：

**Agent状态**（三种Agent代表不同推理路径）：
- **Oneshot Reasoner**：一次性程序生成和执行，适合可直接求解的查询
- **Stepwise Reasoner**：逐步生成Python程序进行多步推理，适合复杂查询
- **Specialized Agent**：快速感知专家（如目标检测、简单问答）

**生命周期状态**：Initial（起点）、Final（终止并输出）、Failure（不可恢复错误）

三个Agent被设计为**既协作又竞争**：协作体现在后续Agent读取前序Agent写入共享内存的中间结果；竞争体现在功能重叠的Agent可接替失败的Agent。

#### 2. 共享内存

所有Agent读写结构化共享内存 $m_t$，累积中间变量、感知结果、程序历史、验证反馈。内存**仅追加**，确保完整推理过程可审计。每步执行：hyper agent观察 $m_t$ 并选择下一状态 $s_{t+1} = \delta_\theta(s_t, m_t)$。

#### 3. 可训练的Hyper Agent

转移函数 $\delta_\theta$ 由SFT微调的LLM实现。从共享内存构建文本提示 $x_t$，LLM映射到可用状态的分布并选择下一状态。

#### 4. 转移轨迹数据生成（MATA-SFT-90K）

**Step 1：构建转移轨迹树**。对每个（图像, 查询）对，在每个决策点分支到所有可用Agent状态，执行对应子自动机，保存内存检查点。

**Step 2：自底向上评分**。叶节点根据任务指标打分（VQA用Accuracy，VG用IoU），非叶节点取子节点最大值向上传播：
$$V(s) = \begin{cases} \text{metric}(\hat{y}_s, y), & s \in \text{Leaves} \\ \max_{s' \in \text{Child}(s)} V(s'), & \text{otherwise} \end{cases}$$

**Step 3：生成SFT数据**。每个决策点的文本提示配上最优子节点的状态标签，构成训练样本。最终收集90,854个样本。

#### 5. 失败处理机制

Agent报告不可恢复错误时，将失败Agent从候选状态中**临时移除**，让hyper agent重新选择其他Agent，避免无限重试。

### 损失函数 / 训练策略

使用标准SFT损失训练Qwen3 4B作为LLM状态控制器。AdamW优化，cosine decay + 5% warmup，batch 64，训练8 epoch。推理时最大步数 $T=15$。

三种SFT配置：域内（在目标数据集训练集上训练）、域迁移（在非目标数据集上训练）、通用（全部数据联合训练）。

## 实验关键数据

### 主实验

**GQA数据集（组合式视觉问答）**：

| 类型 | 方法 | 准确率 |
|------|------|--------|
| 端到端 | InternVL2.5 (8B) | 61.5 |
| 端到端 | InternVL3.5 (8B) | 63.8 |
| 组合式 | HYDRA | 52.8 |
| 组合式 | **MATA (General)** | **64.9** |

**OK-VQA数据集（需要外部知识）**：

| 类型 | 方法 | 准确率 |
|------|------|--------|
| 端到端 | InternVL3.5 (8B) | 75.7 |
| 组合式 | DWIM | 62.8 |
| 组合式 | **MATA (Domain-Specific)** | **76.5** |

**引用表达理解（RefCOCO系列）**：

| 方法 | RefCOCO | RefCOCO+ | RefCOCOg | Ref-Adv |
|------|---------|----------|----------|---------|
| Florence2-L | 95.1 | 92.5 | 90.9 | 71.8 |
| NAVER | 96.2 | 92.8 | 91.6 | 75.4 |
| **MATA (General)** | **96.3** | **93.8** | **90.7** | **77.3** |

### 消融实验

**Hyper Agent组件消融**：

| 层次自动机 | 转移策略 | SFT | GQA | OK-VQA | RefCOCO | 推理时间 |
|-----------|---------|-----|-----|--------|---------|----------|
| ✗ | 穷举集成 | ✗ | 57.7 | 71.5 | 87.7 | 34.58s |
| ✓ | 随机 | ✗ | 57.1 | 71.1 | 85.3 | 6.91s |
| ✓ | LLM | ✗ | 58.5 | 75.1 | 95.8 | 8.07s |
| ✓ | LLM | ✓ | **64.9** | **76.5** | **96.3** | **8.01s** |

**泛化性分析**：跨域迁移性能与域内差距不到1%，表明学到的转移策略高度任务无关。

### 关键发现

1. **组合式方法首次全面超越同等规模端到端VLM**：MATA在GQA和OK-VQA上超过InternVL3.5
2. **SFT对性能的巨大提升**：仅9万样本SFT，GQA准确率从58.5%提升到64.9%（+6.4%）
3. **小模型也能胜任调度**：0.6B经SFT后域内性能已接近4B模型
4. **协作+竞争>纯协作**：三Agent设计允许同一任务上竞争，当一个失败时另一个接替
5. **学习转移 >> 规则转移**：在Ref-Adv上比手写规则的NAVER提升1.9%

## 亮点与洞察

- **形式化优雅**：将多Agent调度建模为Mealy机转移函数学习，保留可解释性同时获得灵活性
- **层次分治**：跨Agent转移用学习，Agent内部步骤用规则，清晰分离"学什么"和"规则化什么"
- **数据生成管线**：转移轨迹树 + 自底向上评分 + SFT数据生成是可推广的多Agent策略学习框架
- **System 1 + System 2**：Specialized/Oneshot/Stepwise的设计呼应认知科学中的快慢思考系统

## 局限与展望

1. 轨迹树搜索的可扩展性：当前3个Agent可穷举，但Agent增多后搜索代价指数增长
2. 推理延迟：平均8s/query在实时应用中仍较高
3. 对基础模型的依赖：天花板受限于底层VLM和检测器的能力
4. Failure恢复的简单性：仅通过移除失败Agent处理，更复杂的恢复策略可能更好
5. 训练数据来源有限：仅来自5个数据集的训练集

## 相关工作与启发

MATA承接ViperGPT → HYDRA → NAVER的发展脉络，首次实现可学习的多Agent转移策略。与MetaGPT等LLM多Agent方法相比形式化程度更高，且支持竞争机制。转移轨迹树的生成思路类似蒙特卡罗树搜索，但聚焦于Agent选择。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 层次自动机+可训练转移函数的框架设计新颖且形式化完备
- **技术质量**: ⭐⭐⭐⭐⭐ — Mealy机形式化、轨迹树数据生成、SFT训练流程环环相扣
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多基准对比+详细消融+泛化分析+模型规模分析
- **实用性**: ⭐⭐⭐⭐ — 框架通用但推理成本较高
- **写作质量**: ⭐⭐⭐⭐⭐ — 形式化清晰，表述严谨
- **综合**: ⭐⭐⭐⭐⭐ (9.0/10)

<!-- RELATED:START -->

## 相关论文

- [Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)
- [ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](../../AAAI2026/interpretability/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)
- [Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](../../NeurIPS2025/interpretability/agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)

<!-- RELATED:END -->
