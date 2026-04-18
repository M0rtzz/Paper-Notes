---
title: >-
  [论文解读] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization
description: >-
  [NeurIPS 2025][RAG] 提出 AlignRAG 框架，将 RAG 重新定义为"检索增强推理"，通过训练专用 Critic Language Model (CLM) 在测试时迭代批评和修正推理过程，解决推理与检索证据之间的错位问题，8B CLM 在 OOD 任务上超越 72B 标准 CLM。
tags:
  - NeurIPS 2025
  - RAG
  - reasoning misalignment
  - critique-driven alignment
  - test-time refinement
  - retrieval-augmented reasoning
---

# Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2504.14858](https://arxiv.org/abs/2504.14858)  
**代码**: [GitHub](https://github.com/AlignRAG/AlignRAG)  
**领域**: llm_nlp  
**关键词**: RAG, reasoning misalignment, critique-driven alignment, test-time refinement, retrieval-augmented reasoning

## 一句话总结

提出 AlignRAG 框架，将 RAG 重新定义为"检索增强推理"，通过训练专用 Critic Language Model (CLM) 在测试时迭代批评和修正推理过程，解决推理与检索证据之间的错位问题，8B CLM 在 OOD 任务上超越 72B 标准 CLM。

## 研究背景与动机

- RAG 已成为知识增强 LLM 的主流范式，但标准 RAG 管道经常无法确保推理与检索证据一致
- 本文识别了一个关键但未被充分研究的失败模式：**推理错位（Reasoning Misalignment）**
    - 即使检索到了相关文档，模型的推理轨迹仍可能偏离证据约束
    - 这不同于简单的检索失败或事实错误，而是**证据整合过程中的结构性缺陷**
- 现有方法主要关注改善检索质量或生成鲁棒性，但**忽视了推理步骤与证据的显式对齐**
- Self-RAG 等反思方法需要架构修改或任务特定微调，泛化性有限
- 自我批评（Self-Refine）存在自我偏好偏差，批评质量受限于模型本身的能力

## 方法详解

### 整体框架

AlignRAG 的核心思想：将 RAG 推理视为可优化的制品（optimizable artifact），通过测试时的批评-修正循环实现推理与证据的动态对齐。

**推理错位的三个阶段**：

1. **相关性评估阶段**：模型未能准确判断文档片段与 query 的相关性
2. **查询-证据映射阶段**：模型无法正确建立 query 元素到证据的映射关系
3. **证据整合合成阶段**：生成的推理步骤不被相关证据逻辑支持

**系统架构**：
- $\mathcal{M}_{\text{gen}}$：生成器（任意 RAG 模型）
- $\mathcal{M}_{\text{critic}}$：训练好的 Critic Language Model (CLM)
- 迭代对齐：$y_0 \xrightarrow{\text{CDA}} y_1 \xrightarrow{\text{CDA}} \cdots \xrightarrow{\text{CDA}} y_T$

### 关键设计

**1. 对比批评合成（Contrastive Critique Synthesis, CCS）**

CCS 通过对比强弱模型的推理轨迹来生成证据导向的批评：

- 弱模型（如 Qwen2.5-0.5B）生成 $y_{\text{unexp}} \sim P_{\text{weak}}(y|q, \mathcal{D})$（易产生错位）
- 强模型（如 LLaMA3.1-8B）生成 $y_{\text{exp}} \sim P_{\text{strong}}(y|q, \mathcal{D})$（更好对齐）
- 构造偏好增强输入：$\mathcal{X}_{\text{pref}} = (q, \mathcal{D}, y_{\text{exp}}, y_{\text{unexp}})$
- 批评函数：$\Delta y_{\text{unexp}} = \mathcal{F}(\mathcal{X}_{\text{pref}})$

对比设计的两个优势：
1. 约束 CLM 关注基于 $\mathcal{D}$ 的差异（而非一般性批评），促进证据敏感性
2. 通过分析路径分歧实现对特定错位类型的细粒度诊断

**2. 结构化训练语料构建**

采用多层级上下文粒度：

$$\mathbf{c}_i = (r_i, h_i, m_i) \in \{0,1\}^3$$

- $r_i$（相关性）：是否包含相关文档
- $h_i$（有用性）：文档是否包含答案片段
- $m_i$（完整性）：文档集是否支持完整推理路径

系统模拟多种证据配置，使 CLM 暴露于多样化的证据环境。

**3. 批评微调（Critique Fine-Tuning, CFT）**

训练目标最大化正确批评的似然：

$$\mathcal{L}_{\text{CFT}}(\theta) = -\sum_{\mathcal{C}_i \in \mathcal{C}} \log p_\theta(\Delta y_{\text{unexp}} | \mathcal{I}_{\text{critic}})$$

其中 $\mathcal{I}_{\text{critic}} = (q, \mathcal{D}, y_{\text{unexp}}, y_{\text{exp}})$。通过将批评生成与目标模型输出解耦，避免了自我偏好偏差。

**4. 测试时批评驱动对齐（CDA）**

每步 CLM 生成编辑信号 $\Delta y_t$，指出 $y_t$ 中的问题并提出基于 $\mathcal{D}$ 的修正：

$$y_{t+1} = \mathcal{M}_{\text{gen}}(y_t \oplus \Delta y_t)$$

$\Delta y_t$ 类似于离散空间中的"伪梯度"，引导生成器向证据对齐方向修正。

### AlignRAG-auto：动态自主对齐

训练 CLM 预测 [Good]/[Bad] 控制 token：

$$p_\theta([\text{Good/Bad}], \Delta y | q, \mathcal{D}, y_{\text{unexp}})$$

推理时根据 token 动态停止：

$$y_{t+1} = \begin{cases} y_t & \text{if CLM outputs [Good]} \\ \mathcal{M}_{\text{gen}}(y_t \oplus \Delta y_t) & \text{if CLM outputs [Bad]} \end{cases}$$

无需手动指定迭代次数，只有需要修正的响应才进行多轮修正。

## 实验关键数据

### 主实验：5 个 In-Domain QA 基准

| 方法 | NQ | MultiHop | TriviaQA | PopQA | ASQA | Avg |
|------|-----|----------|----------|-------|------|-----|
| CoT (Qwen-7B, 无检索) | 33.9 | 45.0 | 58.3 | 26.9 | 20.5 | 36.9 |
| Vanilla RAG (Qwen-7B) | 60.2 | 44.7 | 73.2 | 63.7 | 42.8 | 56.9 |
| InstructRAG (Qwen-7B) | 63.8 | 46.3 | 76.1 | 67.5 | 47.5 | 60.2 |
| Self-Refine (Qwen-7B) | 61.6 | 45.0 | 74.4 | 65.5 | 45.2 | 58.3 |
| **AlignRAG (Qwen-7B + CLM 8B)** | **65.9** | **49.5** | **77.8** | **68.4** | **48.9** | **62.1** |
| Self-Refine (Qwen-14B) | 65.1 | 46.1 | 78.0 | 67.0 | 47.3 | 60.7 |
| **AlignRAG (Qwen-14B + CLM 8B)** | **67.7** | **49.8** | **79.5** | **68.4** | **48.6** | **62.8** |

AlignRAG 在所有骨干网络和基准上均取得最佳。8B CLM 搭配 7B 生成器即可超越 14B Self-Refine。

### AlignRAG-auto vs AlignRAG-fixed

| 数据集 | Fixed (8B) | Auto (8B) | Fixed (14B) | Auto (14B) |
|--------|-----------|----------|------------|-----------|
| PopQA | 66.5 | 67.6 | 68.4 | 68.3 |
| TriviaQA | 77.0 | 77.6 | 79.5 | 79.9 |
| NQ | 65.3 | **66.8** | 67.7 | **69.0** |
| ASQA | 47.1 | **48.8** | 48.6 | **49.8** |
| Avg | 60.6 | **61.4** | 62.8 | **63.4** |

**Auto 版本无需手动调参，性能持平甚至略优于 Fixed 版本**，同时节省不必要迭代的计算开销。

### OOD 泛化实验

- 8B CLM 在 OOD 任务上超越 Self-Refine 基线 **12.1%**
- 8B AlignRAG CLM 超越标准 72B CLM **2.2%**
- AlignRAG 作为插件应用于 InstructRAG 后，Qwen2.5-14B 的 OOD 准确率提升 **9.4%**

### 噪声检索鲁棒性

在有信息和噪声检索场景下均保持强鲁棒性——"当 RAG 检索失败时，AlignRAG 反而更加突出"。

### 关键发现

1. 所有检索增强方法都优于无检索的 CoT，证实外部知识的重要性
2. 训练时精炼（InstructRAG）比标准 RAG 有进一步提升
3. AlignRAG 的提升在 multihop 推理任务上尤为明显（需要多步证据整合）
4. 对比批评合成是关键——8B 专用 CLM 比 72B 通用 CLM 更有效
5. 动态停止机制在不损失精度的前提下显著减少计算

## 亮点与洞察

1. **问题重新定义**：将 RAG 从"检索增强生成"重新定义为"检索增强推理"——这个视角转换非常深刻
2. **推理错位分类**：三阶段分解（相关性评估、查询映射、证据合成）为诊断 RAG 失败提供了结构化框架
3. **对比批评设计精妙**：通过对比强弱模型避免自我偏好偏差，使小模型 CLM 能力突出
4. **即插即用**：无需修改 RAG 架构，作为外挂模块即可提升任何 RAG 系统
5. **Auto 版本实用**：动态停止消除了人工调参需求，是生产级部署的关键特性
6. **小模型大效果**：8B CLM 超越 72B 标准 CLM，证明专门训练的价值远超规模暴力堆叠

## 局限与展望

- 需要额外训练 CLM，引入训练成本和合成数据生成开销
- 迭代修正增加推理延迟，不适合实时低延迟场景
- 对比批评合成依赖于"强模型"和"弱模型"的选择——两者能力差距过大可能影响批评质量
- 训练数据仅 10K（每个数据集 2K），数据量较小
- 弱模型选用 Qwen2.5-0.5B 可能导致产生的"弱响应"过弱，不太像真实失败模式
- [Good]/[Bad] 二元判断可能过于粗粒度，更多控制级别可能有帮助
- 未深入分析哪类推理错位最常见、最难修正

## 相关工作与启发

- 与 Self-RAG 的区别：Self-RAG 用特殊 token 做自评估但需架构修改；AlignRAG 用外部 CLM 做批评且即插即用
- 与 Self-Refine 的区别：Self-Refine 用模型自身批评存在自我偏好偏差，AlignRAG 通过对比学习克服
- 对比批评的思想可以推广到其他需要反馈的场景（代码生成、数学推理等）
- 批评驱动对齐与 RLHF 中的奖励模型思想相似，但作用于推理链而非直接输出

## 评分

- 新颖性: ⭐⭐⭐⭐ 推理错位概念新颖，对比批评合成设计有创意，但迭代修正框架已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个数据集、3 个模型家族、多种基线、OOD 和噪声测试全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，框架描述系统，但部分公式符号稍显冗余
- 价值: ⭐⭐⭐⭐⭐ 对 RAG 系统的可靠性提升有重要贡献，即插即用特性使得实际部署价值高
