---
title: >-
  [论文解读] From Selection to Generation: A Survey of LLM-based Active Learning
description: >-
  [ACL2025][LLM/NLP][主动学习] 首篇系统梳理 LLM 时代主动学习的综述，提出以 Querying（从传统选择到 LLM 生成）和 Annotation（从人工标注到 LLM 标注）为双轴的统一分类体系，覆盖查询策略、标注方案、停止准则、AL 范式和应用领域。
tags:
  - ACL2025
  - LLM/NLP
  - 主动学习
  - LLM
  - 数据选择
  - data generation
  - annotation
  - 综述
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# From Selection to Generation: A Survey of LLM-based Active Learning

**会议**: ACL2025  
**arXiv**: [2502.11767](https://arxiv.org/abs/2502.11767)  
**代码**: 无（综述论文）  
**领域**: llm_nlp  
**关键词**: 主动学习, LLM, 数据选择, data generation, annotation, 综述

## 一句话总结

首篇系统梳理 LLM 时代主动学习的综述，提出以 Querying（从传统选择到 LLM 生成）和 Annotation（从人工标注到 LLM 标注）为双轴的统一分类体系，覆盖查询策略、标注方案、停止准则、AL 范式和应用领域。

## 研究背景与动机

**领域现状**：主动学习（Active Learning, AL）是通过选择最具信息量的数据进行标注来降低标注成本、提高模型性能的经典范式，已有数十年研究积累。

**现有痛点**：传统 AL 方法（如不确定性采样、多样性采样）仅能从固定的无标签池中选择样本，且完全依赖人工标注，在 LLM 时代已显不足。

**核心矛盾**：LLM 的出现从根本上改变了 AL 的能力边界——LLM 不仅能做选择器（selector），还能做生成器（generator）和标注器（annotator），但现有综述仍聚焦于传统 AL 技术，缺乏对 LLM-based AL 的系统回顾。

**本文目标**：填补 LLM-based AL 综述空白，提供统一的分类体系和系统化的文献梳理。

**切入角度**：围绕 AL 循环的两大核心环节——Querying（数据获取）和 Annotation（标注）——构建双轴分类，系统化地将 LLM 在 AL 中的角色映射到各子类。

**核心 idea**：LLM 将 AL 从"从池中选数据"拓展为"选择+生成+自动标注"的全新范式。

## 方法详解

### 整体框架

本文提出一个以 **Querying** 和 **Annotation** 为双轴的分类体系，覆盖 LLM-based AL 循环的五个步骤：

1. **Initialize**：用 LLM 标注/生成初始数据集，解决冷启动问题
2. **Query**：LLM 选择或生成最具信息量的数据实例
3. **Annotate**：LLM 独立/辅助人工完成标注
4. **Train**：用新标注数据更新目标模型
5. **Stop**：根据预算或收敛条件终止循环

### 关键设计一：Querying 策略（§3）

| 子类 | 做什么 | 为什么 | 怎么做 |
|------|--------|--------|--------|
| 传统选择（§3.1） | 基于不确定性/多样性从未标注池中选样本 | 经典有效但受限于固定池 | Least Confidence、Margin、Max-Entropy、CoreSet、BADGE、BALD 等 |
| LLM 选择（§3.2） | 用 LLM 评估并选择最有价值的样本 | LLM 具备推理能力，可在零/少样本下评估样本信息量 | ActiveLLM（不确定性+多样性评估）、SelectLLM（LLM 排序+k-NN 聚类）、Ask-LLM（质量评分）、ActivePrune（LLM 剪枝池） |
| LLM 生成（§3.3） | 用 LLM 生成全新数据实例 | 突破固定池限制，搜索空间扩展到无穷 | 池内生成：Diao et al. 利用答案变异度估计不确定性；池外生成：APE 用 CoT 合成新 prompt、Yang et al. 生成+拒绝采样 |
| 混合（§3.4） | 结合选择与生成 | 取长补短 | NoiseAL（小 LLM 识别+大 LLM 标注）、CAL（密度聚类+GPT-4 选择） |

### 关键设计二：Annotation 方案（§4）

| 子类 | 做什么 | 为什么 | 怎么做 |
|------|--------|--------|--------|
| 人工标注（§4.1） | 人类标注选中样本 | 质量最高但成本昂贵 | ActivePrune、Active-Prompt、Beyond-Labels（收集 rationale） |
| LLM 标注（§4.2） | LLM 模拟人类标注 | 大幅降低成本 | FreeAL（任务知识蒸馏，零人工）、LLMaAA（in-context example 提升质量）；风险：自我强化偏差、输入敏感性 |
| 混合（§4.3） | LLM 先标注 + 验证器评估 + 人工修复低质量 | 平衡效率与准确性 | Wang et al.（多步协同）、HybridAL（置信度阈值路由） |

### 关键设计三：停止准则（§5）

- **传统方式**：验证集性能改善低于阈值、预测稳定、理论样本复杂度界
- **LLM 时代的挑战**：标注成本不再是离散预算 $k$，而是输入/输出 token 的实值成本；需混合考虑人工+LLM 成本
- **新方向**：token 级成本分析、成本-性能联合指标、混合停止准则

### AL 范式与 LLM（§6）

综述梳理了四种 LLM-based AL 范式：
- **Active In-Context Learning**：将 few-shot 示例选择建模为 AL 问题
- **Active Supervised Fine-Tuning**：AL 优化 SFT 数据选择
- **Active Preference Alignment**：RLHF 中的高效偏好查询
- **Active Knowledge Distillation**：基于不确定性的选择性知识迁移

## 实验关键数据

### 表1：LLM-based AL 分类体系

| 类别 | 子类 | 机制 |
|------|------|------|
| Querying | 传统选择 | 不确定性采样、多样性采样 |
| Querying | LLM 选择 | LLM 评分/排序选择样本 |
| Querying | LLM 生成 | 生成全新实例 |
| Querying | 混合 | 选择+生成结合 |
| Annotation | 人工标注 | 人类标注，高质量高成本 |
| Annotation | LLM 标注 | LLM 模拟标注，低成本有偏差风险 |
| Annotation | 混合 | 动态路由 LLM/人工 |

### 表2：代表性方法对比（部分）

| 方法 | 查询方式 | 标注方式 | 主要应用 |
|------|----------|----------|----------|
| ActiveLLM | LLM 选择 | 人工 | 文本分类 |
| FreeAL | 混合（选择+生成） | LLM | 文本分类、情感分析 |
| APE | LLM 生成 | LLM | 实体匹配 |
| CAL | 混合 | 人工 | 去偏差 |
| NoiseAL | LLM 生成+混合 | LLM | 文本分类、NER |
| SelectLLM | LLM 选择 | 人工 | 文本分类 |

### 关键发现

1. **LLM 选择 vs 传统选择**：LLM 在零/少样本场景下的选择能力优于传统不确定性方法（ActiveLLM、SelectLLM 均有验证）
2. **生成 vs 选择**：LLM 生成新样本可突破固定池限制，但需拒绝采样保证质量
3. **LLM 标注成本**：人工标注成本约为 LLM 标注的 10-100 倍（Kholodna et al. 2024 报告低资源语言场景）
4. **混合标注**：动态路由策略在保持准确率的同时降低 40-70% 标注成本（HybridAL）

## 亮点与洞察

1. **分类体系直观有效**：双轴（Querying × Annotation）分类将复杂的 LLM-based AL 方法空间梳理得清晰可操作
2. **从选择到生成的范式转变**：清晰指出 LLM 将 AL 从"被动选选"转变为"主动创造"，是本文最核心的洞察
3. **成本模型的重新思考**：指出 LLM 时代标注成本从离散预算变为实值成本，停止准则需根本性重构
4. **四大 AL 范式的梳理**：Active ICL/SFT/RLHF/KD 四条线路清晰，为后续研究提供了路线图
5. **开放问题的前瞻性**：多 LLM 协同 AL、多模态 AL、LLM Agent+AL 等方向具有实际研究启发意义

## 局限与展望

1. **缺乏定量对比实验**：作为综述未提供统一 benchmark 上的方法横向对比，读者难以直观判断各方法优劣
2. **传统 AL for LLM 覆盖不足**：明确声明不深入覆盖"用传统 AL 优化 LLM 训练"的工作，可能遗漏重要文献
3. **偏差问题讨论不够深入**：虽然提到 LLM 标注的偏差风险，但缺乏系统的量化分析和缓解策略评估
4. **缺乏对计算开销的分析**：LLM 作为选择器/生成器/标注器的计算成本缺乏量化对比
5. **可拓展方向**：建议未来综述加入统一实验评估、更深入的成本-效益分析、以及多模态 AL 的实证研究

## 相关工作与启发

### vs 传统 AL 综述（Ren et al. 2021; Zhan et al. 2022）

传统 AL 综述聚焦于不确定性/多样性/混合采样策略和理论分析，未涉及 LLM 作为选择器/生成器/标注器的新范式。本文的核心增量在于系统化梳理 LLM 带来的三大角色变化。

### vs FreeAL（Xiao et al. 2023）

FreeAL 提出完全无人工标注的 AL 框架（LLM 标注 + 小模型协同），是 LLM-based Annotation 方向的代表。本综述将其定位在 Hybrid Querying + LLM Annotation 的交叉位置，并指出其零人工适用场景和偏差风险。

### vs ActiveLLM（Bayer & Reuter 2024）

ActiveLLM 专注于让 LLM 评估不确定性和多样性来选择样本，适用于少样本场景。综述将其归类为 LLM Selection，并对比了它与 SelectLLM、Ask-LLM 等方法的异同。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首篇系统化 LLM-based AL 综述，双轴分类体系是原创贡献
- **实验充分度**: ⭐⭐⭐ — 综述性质未包含实验，但方法对比表格较全面，缺少统一 benchmark 定量横评
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰，分类体系直观，图表配合良好，可读性强
- **价值**: ⭐⭐⭐⭐ — 为 LLM 时代的 AL 研究提供了完整路线图，对实践者和研究者均有参考价值

<!-- RELATED:START -->

## 相关论文

- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs](llm_test_case_gen_bugs.md)
- [A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)

<!-- RELATED:END -->
