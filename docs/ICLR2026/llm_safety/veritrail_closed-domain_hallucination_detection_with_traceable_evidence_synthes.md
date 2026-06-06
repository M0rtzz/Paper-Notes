---
title: >-
  [论文解读] VeriTrail: Closed-Domain Hallucination Detection with Traceability
description: >-
  [ICLR2026][LLM安全][hallucination detection] 提出 VeriTrail——首个为多步生成过程（MGS）提供可追溯性的闭域幻觉检测方法，建模生成过程为 DAG 并沿路径逐层验证，同时构建了首批包含所有中间输出和人工标注的 MGS 数据集。
tags:
  - "ICLR2026"
  - "LLM安全"
  - "hallucination detection"
  - "faithfulness evaluation"
  - "traceability"
  - "multi-generative-step"
  - "DAG"
---

# VeriTrail: Closed-Domain Hallucination Detection with Traceability

**会议**: ICLR2026  
**arXiv**: [2505.21786](https://arxiv.org/abs/2505.21786)  
**代码**: [数据集](https://aka.ms/veritrail-datasets)  
**领域**: LLM安全  
**关键词**: hallucination detection, faithfulness evaluation, traceability, multi-generative-step, DAG

## 一句话总结
提出 VeriTrail——首个为多步生成过程（MGS）提供可追溯性的闭域幻觉检测方法，建模生成过程为 DAG 并沿路径逐层验证，同时构建了首批包含所有中间输出和人工标注的 MGS 数据集。

## 研究背景与动机
- LLM 即使被要求遵循源材料，仍常生成未支持的内容——"闭域幻觉"
- 生成过程分为两类：
    - **单步生成（SGS）**：如标准 RAG，一次 LLM 调用产出最终结果
    - **多步生成（MGS）**：如分层摘要、GraphRAG，中间输出作为后续输入
- MGS 更易产生幻觉：每一步都可能引入并传播错误
- **核心论点**：对 MGS 而言，仅检测最终输出中的幻觉是不够的，还需要：
    - **溯源（Provenance）**：理解输出如何从源材料推导
    - **错误定位（Error Localization）**：定位幻觉在哪一步引入
- 现有方法只评估输出与源材料的关系，不利用中间输出，无法提供可追溯性

## 核心贡献
1. 统一的生成过程概念框架（DAG 表示）
2. VeriTrail：首个为 MGS 和 SGS 提供可追溯性的闭域幻觉检测方法
3. FABLES+ 和 DiverseSumm+：首批包含所有中间输出和人工标注的 MGS 数据集

## 方法详解

### 概念框架：生成过程的 DAG 表示

将生成过程建模为有向无环图 $G = (V, E)$：
- **节点** $v \in V$：文本片段（源文档/中间输出/最终输出）
- **有向边** $(u, v) \in E$：$u$ 被用作生成 $v$ 的输入
- **根节点** $V_0$：源文档（无入边）
- **终端节点** $v^*$：最终输出（无出边）
- **阶段函数** $\text{stage}: V \to \mathbb{N}$：反映节点在生成过程中的位置

### VeriTrail 检测流程

输入：(1) 完成的生成过程 DAG；(2) 终止参数 $q$；(3) 从 $v^*$ 提取的事实声明集 $C$

对每个声明 $c \in C$ 独立执行以下步骤：

#### Step 1: 子声明分解
- 使用 Claimify 的 Decomposition 模块将复合声明拆分为独立可验证的子声明
- 例："公司X在2020年收购了两家初创企业作为医疗扩张的一部分" → (1) X在2020年收购两家初创企业 (2) 收购是医疗扩张的一部分
- 递归分解，最多 20 次，避免无限循环

#### Step 2: 证据选择（Evidence Selection）
- 从终端节点的源节点 $\text{src}(v^*)$ 出发
- 使用 NLTK 分句，为每个句子分配唯一 ID
- LLM 选择支持/反对声明及子声明的句子（返回句子 ID）
- 若超出上下文窗口则分割为多个并行 prompt
- **ID 验证保证**：丢弃不匹配的 ID，确保证据不被幻觉

#### Step 3: 判决生成（Verdict Generation）
- 若无句子被选中 → "Not Fully Supported"
- 否则 LLM 基于证据给出三类判决：
    - **Fully Supported**：源文本强烈暗示整个声明
    - **Not Fully Supported**：至少有一部分未被源文本支持
    - **Inconclusive**：源文本模糊或矛盾

**上下文处理**：不直接使用选中句子（可能脱离上下文歧义），而是：
- 根节点：包含完整内容
- 非根节点：使用证据选择步骤生成的摘要

#### Step 4: 候选节点选择与迭代终止
根据最新判决选择下一轮验证的候选节点：

| 最新判决 | 候选节点选择策略 |
|----------|---------------|
| Fully Supported / Inconclusive | 本轮有证据节点的源节点 |
| Not Fully Supported | 本轮所有验证节点的源节点（更广泛，防漏检） |

终止条件（满足任一）：
1. 候选节点仅含已验证的有证据根节点 → 采用最新判决
2. 无候选节点（未到达根节点或根节点无证据）→ Not Fully Supported
3. 连续 $q$ 次 Not Fully Supported → Not Fully Supported

### 可追溯性输出
对每个声明返回：
- **最终判决** + LLM 推理
- **所有临时判决**
- **证据链**：选中句子（含节点 ID）+ 各轮证据摘要

#### 溯源（Provenance）
- 对 Fully Supported 声明：证据链记录了从中间节点到根节点的路径

#### 错误定位（Error Localization）
- 找到最后一次 Fully Supported 判决的迭代 $n$
- 该迭代中有证据的非根节点的阶段即为错误阶段
- $\{\text{stage}(v) | v \in V_e(n), v \notin V_0\}$

## 数据集构建

### FABLES+（分层摘要）
- 基于 FABLES 书籍摘要数据集
- 重新生成 22 本书的分层摘要（平均 118K tokens），保留所有中间输出
- 提取 734 个声明，48% 直接沿用原标注，其余人工标注

### DiverseSumm+（GraphRAG）
- 基于 DiverseSumm 新闻数据集
- 148 个故事，1,479 篇文章，累计 1.19M tokens
- 采样 20 个问题，用 GraphRAG 生成答案
- 提取 560 个声明，4 位 Upwork 标注员 + 1 位作者标注
- 87% 声明可从关联文章判断，13% 需查阅额外文章

## 实验结果

### 基线方法

| 类别 | 方法 | 处理长文本策略 |
|------|------|---------------|
| NLI | INFUSE | 双向蕴含排序 |
| NLI | AlignScore | 350 token 分块 |
| NLI | Bespoke-MiniCheck-7B | 32K token 分块 |
| RAG | Top-k 检索 | 嵌入检索 + 判决 |
| 直接验证 | Gemini 1.5 Pro / GPT-4.1 Mini | 长上下文 LM |

### 硬预测结果（Macro F1 / Balanced Accuracy）

| 方法 | FABLES+ F1 | FABLES+ Bal.Acc | DiverseSumm+ F1 | DiverseSumm+ Bal.Acc |
|------|-----------|----------------|-----------------|---------------------|
| **VeriTrail (q=3)** | **84.5** | **83.6** | **79.5** | 76.3 |
| **VeriTrail (q=1)** | 74.0 | **84.6** | 76.6 | **83.0** |
| RAG (k=15) | 69.6 | 76.5 | 75.1 | 74.0 |
| Bespoke-MiniCheck-7B | 62.2 | 69.0 | 72.1 | 69.4 |
| Gemini 1.5 Pro | 61.1 | 60.8 | 49.8 | 57.6 |
| GPT-4.1 Mini | 60.7 | 58.2 | 62.9 | 61.5 |
| AlignScore | 59.6 | 67.5 | 60.4 | 62.7 |
| INFUSE | 40.5 | 59.5 | 20.0 | 50.1 |

**关键发现**：
- VeriTrail 在两个数据集上均优于所有基线（q=3 在 F1 上最优，q=1 在 Balanced Accuracy 上最优）
- 直接长上下文验证（Gemini 1.5 Pro）并不理想，可能因超长文档中信息检索困难
- AlignScore 和 INFUSE 等经典 NLI 方法在长文档上性能明显不足

### q 参数的权衡
- q=1（一次 NFS 即终止）：高 NFS 召回（89.8%），低 NFS 精度（55.1%）
- q=3（三次 NFS 才终止）：更均衡（NFS 精度 84.5%，召回 55.9%）
- q 越大，验证越彻底但 NFS 判决更保守

## 优势与局限

### 优势
- 首个提供可追溯性（溯源 + 错误定位）的幻觉检测方法
- DAG 框架统一了 SGS 和 MGS 过程的表示
- 句子级证据选择 + ID 验证保证证据不被幻觉
- 在超长文档（>100K tokens）上优于强基线
- 成本效益好（Appendix D 分析）

### 局限
- 依赖 LLM 执行证据选择和判决生成（受 LLM 能力限制）
- 错误定位在某些场景下无法确定具体阶段
- 数据集规模有限（734 + 560 声明）
- 仅评估了 gpt-4o 模型

## 个人评价与思考

### 创新性 ⭐⭐⭐⭐⭐
- "检测 + 追溯"的范式升级非常有价值
- DAG 建模生成过程是对幻觉检测的根本性重新思考
- 迭代证据选择 + 候选节点传播机制设计精妙

### 实用价值 ⭐⭐⭐⭐⭐
- 直接面向 MGS 流水线（GraphRAG、分层摘要等）的实际需求
- 错误定位对系统调试和改进极有价值
- 句子级证据链显著降低人工审核成本

### 数据集贡献 ⭐⭐⭐⭐
- FABLES+ 和 DiverseSumm+ 填补了 MGS 幻觉检测数据的空白
- 包含完整中间输出是关键创新
- 但规模较小

### 实验设计 ⭐⭐⭐⭐
- 基线覆盖全面（NLI、RAG、长上下文 LM）
- 硬预测+软预测双评估
- 消融分析和错误案例分析（附录）增加可信度

### 综合评分 ⭐⭐⭐⭐⭐
一篇开创性的工作，将闭域幻觉检测从"判断对错"提升到"追溯来源和定位错误"。DAG 框架优雅地统一了各类生成过程，VeriTrail 的迭代验证机制在超长文档上展现出强大性能。对于日益复杂的 MGS 管道（如 GraphRAG），这种可追溯的幻觉检测方法具有极强的实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)
- [\[CVPR 2026\] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](../../CVPR2026/llm_safety/tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[ACL 2025\] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](../../ACL2025/llm_safety/learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)
- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](../../ACL2026/llm_safety/hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)

</div>

<!-- RELATED:END -->
