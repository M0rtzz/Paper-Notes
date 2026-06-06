---
title: >-
  [论文解读] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging
description: >-
  [ACL 2025][LLM对齐][Mixture-of-Experts] 本文提出UpIT (Upcycling Instruction Tuning)，利用密集模型指令微调过程中的中间checkpoint作为专业化专家，通过遗传算法扩展专家数量和路由预优化，实现数据高效且灵活的dense-to-MoE转换…
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "Mixture-of-Experts"
  - "Upcycling"
  - "参数合并"
  - "专家多样性"
  - "路由初始化"
---

# Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging

**会议**: ACL 2025  
**arXiv**: [2410.01610](https://arxiv.org/abs/2410.01610)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: Mixture-of-Experts、Upcycling、参数合并、专家多样性、路由初始化

## 一句话总结
本文提出UpIT (Upcycling Instruction Tuning)，利用密集模型指令微调过程中的中间checkpoint作为专业化专家，通过遗传算法扩展专家数量和路由预优化，实现数据高效且灵活的dense-to-MoE转换。

## 研究背景与动机
- **领域现状**：MoE架构通过稀疏激活显著扩大模型容量而不增加推理开销，已成为LLM效率提升的重要方向。Upcycling（从密集模型转换为MoE）比从零训练MoE更高效。
- **现有方法的痛点**：
    - **Vanilla Upcycling**（复制FFN层→大规模后训练）：专家初始同质，需要~1T tokens或~5M指令数据
    - **Specialized Upcycling**（先训练领域专家→再组装）：需要数千亿领域数据，且专家数量不灵活
- **核心矛盾**：如何在**数据量很少**的情况下，**灵活地**将dense预训练模型转换为高质量MoE指令模型？
- **核心insight**：指令微调不同epoch的checkpoint自然表现出不同领域的专长（MMLU最好的checkpoint和GSM8K最好的不同），这些checkpoint天然适合作为专业化专家！

## 方法详解

### 整体框架
UpIT包含四个阶段：
1. **Expert Preparation**（专家准备）：微调dense模型，定期保存checkpoint作为专家
2. **Expert Expansion**（专家扩展）：用遗传算法+参数合并扩展到任意数量专家
3. **Router Initialization**（路由初始化）：预优化路由向量确保专家各展所长
4. **Model Upcycling**（模型组装）：合并专家和路由，进行后训练

### 关键设计
1. **基于中间Checkpoint的专家准备**

    - 关键观察：不同训练epoch的checkpoint在不同benchmark上表现出交错的最佳性能
    - 例如：epoch 2的模型HellaSwag最好，epoch 0.25的模型MMLU最好
    - 无需精心设计领域数据，只需保存checkpoint即可获得多样化专家
    - 动机：大幅降低专家获取成本，从"百亿级领域数据"降至"免费的中间产物"

2. **遗传算法驱动的专家扩展**

    - 问题：checkpoint数量固定，可能不满足目标专家数
    - 方案：每轮选择差异最大的两个专家作为"父母"
    - 随机分配权重 $\alpha, \beta$（$\alpha + \beta = 1$）
    - 使用DARE（Drop And REscale）在合并前引入"突变"
    - 新专家：$\mathbf{E}_{new} = \text{DARE}(\alpha \mathbf{E}_{j^*}, \beta \mathbf{E}_{k^*})$
    - 关键：选择差异最大而非随机的两个专家 → 保证新专家多样性

3. **基于种子数据的路由预优化**

    - 问题：随机初始化的router导致token被派发到错误专家，削弱之前建立的专家差异性
    - 数据选择（Algorithm 3）：随机抽取训练集的1%（约500-5000样本），计算每个样本在每个专家上的PPL，将样本分配给PPL最低的专家
    - 辅助损失：$\mathcal{O}_i = \min_{\mathbf{E}_i}(\alpha \mathcal{L}_{lm} + (1-\alpha)\mathcal{L}_{aux})$
    - $\mathcal{L}_{aux} = \text{CrossEntropy}(\text{Sigmoid}(\mathbf{h}_{\mathbf{r}_i}), \mathbf{I})$：最大化路由向量对专长数据的输出概率
    - 仅需1%数据、4个epoch → 极低成本预优化

### 损失函数 / 训练策略
- **Router初始化阶段**：$\alpha \mathcal{L}_{lm} + (1-\alpha)\mathcal{L}_{aux}$，$\alpha=0.5$
- **后训练阶段**：标准causal LM loss + load balancing loss $\mathcal{L}_{load} = n \cdot \sum_i f_i \cdot P_i$
- **训练分配**：总4 epoch，前2 epoch准备专家，后2 epoch后训练MoE
- **LoRA-based**：学习率2e-4；**FFN-based**：学习率2e-5

## 实验关键数据

### 主实验（LoRA-based，8E选2）

| 方法 | HumanEval | GSM8K | MMLU | NQ | Avg. |
|------|-----------|-------|------|-----|------|
| LoRA基线 | 22.56 | 45.72 | 49.33 | 14.99 | 47.53 |
| LoRAMoE_SFT | 28.66 | 49.81 | 50.54 | 20.55 | 49.99 |
| Self-MoE | 28.05 | 46.70 | 49.63 | 21.11 | 49.30 |
| **UpIT** | **35.37** | **49.51** | **50.31** | **24.52** | **52.21** |

### 主实验（FFN-based，4E选2）

| 方法 | HumanEval | GSM8K | MMLU | NQ | Avg. |
|------|-----------|-------|------|-----|------|
| SFT基线 | 26.22 | 29.19 | 33.93 | 8.42 | 31.31 |
| Upcycle_SFT | 23.17 | 33.97 | 38.90 | 15.18 | 37.60 |
| **UpIT** | **31.34** | **33.81** | **40.84** | **14.71** | **38.88** |

### 数据效率

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| UpIT(8E) 50K数据 | 47.13 avg | ≈ LoRAMoE(8E) 500K数据的水平 |
| UpIT(16E) 100K数据 | 49.18 avg | > LoRAMoE(16E) 500K数据 |
| 数据增长曲线 | 近线性增长 | 基线方法呈log曲线（增长饱和） |

### 消融实验

| 配置 | Avg. | 说明 |
|------|------|------|
| UpIT完整 | 52.21 | - |
| w/o 路由初始化 | 49.96 | -2.25，证明路由预优化关键 |
| 随机数据初始化路由 | 49.30 | 比不初始化更差（破坏多样性） |
| w/o 专家扩展 | 53.31 | 直接用checkpoint也可，但不灵活 |
| 随机选父母合并 | 52.41 | -0.90，证明选差异最大的重要 |
| 前半checkpoint | 51.37 | 后半更好（数学/代码持续提升） |

### 关键发现
- **数据效率惊人**：50K数据即可达到传统方法500K的水平（10倍数据效率提升）
- **专家数量可扩展**：UpIT在增加专家数量时稳定提升，而vanilla upcycling甚至出现性能下降
- **路由分析**：UpIT的router能将不同领域token准确分配到特定专家（HumanEval→Expert 4，MMLU→Expert 3），而LoRAMoE则均匀分配
- **上限更高**：继续增加训练epoch，UpIT保持线性增长，基线方法性能停滞

## 亮点与洞察
- **核心insight精彩**：发现中间checkpoint具有领域专长差异——这个观察自然但非常有价值
- **设计一致性**：四个阶段的所有设计都围绕"保持/增强专家多样性"这一核心目标
- **工程友好**：不引入额外训练数据需求，router初始化仅需1%数据训练4个epoch
- **泛化性好**：在LoRA-based和FFN-based两种场景都有效
- **遗传算法的巧妙应用**：选差异最大的"父母"进行"参数交叉+DARE突变"生成新专家，类比生物演化

## 局限与展望
- 实验仅在Llama 2 7B和Sheared Llama 2.7B上进行，更大模型（70B+）的效果待验证
- checkpoint保存间隔的选择（每0.25 epoch）似乎未充分消融
- 路由初始化使用PPL选择数据，假设PPL低=专长，但PPL对不同任务的区分度可能不同
- 未与Branch-Train-MiX等用大规模领域数据的方法在相同数据规模下对比
- **研究方向**：(1)能否自动确定最佳checkpoint保存间隔和数量？(2)可否在推理时动态调整专家数量/激活数量？

## 相关工作与启发
- 与MoE Jetpack (Zhu et al., 2024b)的关系：后者也利用checkpoint，但本文更系统地将其集成到完整upcycling流程
- 与模型合并(Model Merging)的联系：DARE、TIES-Merging等方法被创新性地用于MoE专家扩展
- 与Self-MoE (Kang et al., 2024)的对比：Self-MoE需要单独训练专家专长，UpIT利用自然差异
- 启发：任何需要从单一模型衍生多样化模块的场景都可借鉴"checkpoint即专家"的思路

## 评分
- 新颖性: ⭐⭐⭐⭐ 利用中间checkpoint差异作为专家的insight新颖且实用，遗传算法扩展有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 数据规模消融、专家数量消融、上限探索、路由可视化、多种消融分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Algorithm和Figure配合好，问题定义到方法过渡自然
- 价值: ⭐⭐⭐⭐⭐ 大幅降低MoE训练的数据需求，方法实用、可复现性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ICLR 2026\] Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks](../../ICLR2026/llm_alignment/optimal_sparsity_of_mixture-of-experts_language_models_for_reasoning_tasks.md)
- [\[ACL 2025\] Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection](beyond_similarity_a_gradient-based_graph_method_for_instruction_tuning_data_sele.md)

</div>

<!-- RELATED:END -->
