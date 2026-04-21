---
title: >-
  [论文解读] HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location
description: >-
  [NeurIPS 2025][在线-离线共置] 提出HyGen——干扰感知的LLM推理系统，通过精准的批次延迟预测器、SLO感知的性能分析器和前缀共享最大化调度策略，实现在线和离线工作负载的弹性共置，在保证严格SLO合规的同时获得3.87-5.84倍吞吐提升。
tags:
  - NeurIPS 2025
  - 在线-离线共置
  - 延迟预测
  - SLO保证
  - 前缀共享
  - 弹性调度
---

# HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location

**会议**: NeurIPS 2025  
**arXiv**: [2501.14808](https://arxiv.org/abs/2501.14808)  
**代码**: [https://github.com/UIUC-MLSys/HyGen](https://github.com/UIUC-MLSys/HyGen)  
**领域**: LLM推理系统  
**关键词**: 在线-离线共置, 延迟预测, SLO保证, 前缀共享, 弹性调度

## 一句话总结
提出HyGen——干扰感知的LLM推理系统，通过精准的批次延迟预测器、SLO感知的性能分析器和前缀共享最大化调度策略，实现在线和离线工作负载的弹性共置，在保证严格SLO合规的同时获得3.87-5.84倍吞吐提升。

## 研究背景与动机

**LLM服务的两种工作负载模式**：LLM应用分为在线服务（聊天机器人、代码助手等需要低延迟的交互式任务）和离线服务（数据合成、文档摘要、模型评测等追求吞吐的批处理任务）。两者的SLO需求截然不同——在线服务关注TTFT（首token延迟）和TBT（token间延迟）的P99指标，离线服务则以QPS/TPS吞吐量为核心度量。

**资源利用效率的核心矛盾**：当前主流部署方式是为每类工作负载分配独立的GPU集群。然而对Azure LLM服务生产trace的分析揭示了一个严重的效率问题——在线请求负载具有高度时变性，不仅存在昼夜规律的日周期波动，更存在**分钟级3倍的突发波动**。为满足峰值负载下的延迟要求，服务商不得不按峰值需求配置GPU资源（分钟级弹性伸缩在工程上极不实际），导致低谷时段的GPU大量闲置。

**共置的机遇与挑战**：在同一推理实例上共置在线和离线请求可以在低谷时段利用闲置资源处理离线任务，但面临三个核心挑战：(1) **多样化SLO**——不同应用甚至同一应用的不同用户有不同的延迟要求（P99 TBT vs mean TTFT vs严格上界），难以制定统一的资源共享策略；(2) **大量不确定性**——请求到达的突发性加上输出长度的不可预测性，使得调度决策高度不确定；(3) **性能干扰**——离线请求的大批次可能严重阻塞在线请求（head-of-line blocking），长输入与短查询的混合批处理会使所有请求的延迟恶化一个数量级。

**本文的核心洞察**：通过精确建模批次执行时间和量化共置干扰代价，可以实现细粒度的"见缝插针"式调度——在严格保证在线SLO的前提下，最大化闲置计算资源的离线利用率。

## 方法详解

### 整体框架
HyGen采用双队列架构，将延迟敏感的在线请求与吞吐导向的离线请求分离管理。其两阶段调度流程为：**在线阶段**先用FCFS或公平调度策略形成初始批次，必要时通过优先级抢占机制驱逐离线请求保护在线性能；**离线阶段**利用延迟预测器评估残余容量，在不违反SLO的前提下最大化离线请求的填充。HyGen作为实例级调度器运行，接收来自上游系统级路由器（如Preble）的请求，因此单实例的请求并发和调度开销天然有界。

### 关键设计

1. **延迟预测器（Latency Predictor）**:

    - 功能：精确估计不同请求批次组合的执行时间，支持实时调度决策
    - 核心思路：基于LLM推理的两个阶段的不同计算模式建模批次执行时间——prefill阶段因attention计算呈二次复杂度，decode阶段呈线性复杂度。预测模型为$T_{batch} = f(S_p, S_d, S_p^2, S_d^2, N_p, N_d)$，其中$S_p, S_d$为prefill/decode token总数，$N_p, N_d$为请求数，平方项捕获非线性scaling效应。采用线性回归作为预测模型，训练数据通过系统性profiling收集
    - 设计动机：线性回归的选择看似简单但有深刻理由——(1) 推理极快，支持实时调度（约18μs/次）；(2) 特征集简洁确保对不同负载模式的泛化稳定性；(3) 训练极轻量（8万样本仅需15ms），轻松适配不同硬件。实测MAPE仅1.07%-1.78%，比深度模型更可靠

2. **SLO感知性能分析器（SLO-aware Profiler）**:

    - 功能：将延迟预测转化为具体的调度约束——确定在给定SLO下，允许添加的最大离线负载量
    - 核心思路：分析器首先根据工作负载和SLO要求确定可行的延迟预算范围，然后通过二分搜索在该范围内找到满足SLO的最大延迟预算上限。关键区别在于：单个batch的延迟预算与整体SLO（如mean TBT或P99 TTFT）之间存在统计gap——分析器通过test-run检验弥补这一gap。部署时，该预算作为两阶段调度中的批次延迟限制使用
    - 设计动机：朴素方案（直接用SLO值作为batch延迟预算）会导致over-conservative或under-conservative。SLO感知分析器通过离线profiling建立batch级延迟与整体SLO之间的准确映射，使系统能够更精细地利用残余容量

3. **前缀共享最大化调度策略（PSM, Prefix Sharing Maximization）**:

    - 功能：优化离线请求的调度顺序，最大化KV cache的前缀复用以提升吞吐
    - 核心思路：将所有离线请求组织为Trie树结构，树的叶节点对应请求，共享前缀为公共祖先路径。通过DFS遍历确定调度优先级——前缀共享最多的请求被安排在一起处理。例如对于请求队列("What is ML", "How to code", "What is AI", "How to debug")，PSM重排为("What is ML", "What is AI"), ("How to code", "How to debug")，最大化"What is"和"How to"前缀的cache复用
    - 设计动机：朴素FCFS调度完全忽略前缀共享机会。PSM在SLO感知调度的框架内引入cache效率优化，是一种正交的加速手段。为防止前缀共享度低的请求饥饿，PSM扩展版引入了融合请求新鲜度的utility ratio——通过对比DFS序和自平衡BST中的最旧请求，在效率和公平性间取得平衡

### 损失函数 / 训练策略
延迟预测器使用线性回归训练：通过系统性profiling目标硬件收集训练数据，覆盖不同的请求数量、序列长度分布和batch组成。训练数据约8万样本，训练耗时仅15ms（CPU上）。SLO感知分析器通过二分搜索离线确定延迟预算，运行时无需学习。优先级抢占机制保留被驱逐请求的执行状态以支持后续恢复。

## 实验关键数据

### 主实验
**在线SLO合规性与吞吐提升（Llama2-7B, Azure trace, A100）**：

| 方法 | Mean TBT SLO | P99 TBT SLO | Mean TTFT SLO | P99 TTFT SLO | 离线吞吐提升 |
|------|-------------|-------------|--------------|-------------|-------------|
| Sarathi (纯在线) | ✓ | ✓ | ✓ | ✓ | 1.0× |
| Sarathi++ (混合) | ✗ (无SLO控制) | ✗ | ✗ | ✗ | - |
| HyGen* | ✓ | ✓ | ✓ | ✓ | 1.82× |
| **HyGen** | **✓** | **✓** | **✓** | **✓** | **3.87-5.84×** |

**跨模型/硬件验证**：

| 模型 | 硬件 | 离线吞吐提升 | 总吞吐占纯离线比例 |
|------|------|-------------|------------------|
| Llama2-7B | A100×4 | 3.87× | 84.3% |
| Qwen-14B | A40×4 | 5.84× | - |
| Yi-34B (TP2+PP2) | A40×4 | 1.89× | - |
| Sheared-LLaMA-2.7B | A5000×1 | 2.18× | - |

### 消融实验

| 组件 | 离线吞吐提升 | 说明 |
|------|-------------|------|
| 无SLO感知分析器（直接用SLO值作预算） | 较低 | 过度保守或SLO违规 |
| **有SLO感知分析器** | **显著更高** | 精确映射batch延迟→整体SLO |
| 无PSM前缀共享 (FCFS) | 1.0× | 无前缀复用 |
| **有PSM前缀共享** | **最高4×** | MMLU数据集上仿真验证 |
| 延迟预测器MAPE=1% | 基准 | 最优 |
| 延迟预测器MAPE=5% | 略降但仍SLO合规 | 系统对预测精度鲁棒 |
| 延迟预测器MAPE=10% | 持续降低 | 精度恶化影响有限 |

### 关键发现
- 延迟预测器精度极高：Llama2-7B上MAPE 1.78%，Qwen-14B上MAPE 1.07%
- HyGen的离线吞吐达到纯离线Sarathi-offline的84.3%——即共置模式几乎不浪费计算资源
- 系统可同时满足多个SLO约束（如P99 TTFT 8% + mean TBT 10-50%），性能由最严格SLO主导
- 在Mooncake（更极端的负载波动）trace上同样有效
- 对预测器精度鲁棒——MAPE从1%增至5%，吞吐下降有限且SLO仍合规

## 亮点与洞察
- **生产级的工程深度与学术创新的结合**：延迟预测器设计看似简单（线性回归），但基于对prefill二次/decode线性复杂度的深入理解选择特征，实现了1%级别的预测精度——这种"用正确的先验知识让简单模型超越复杂模型"的思路值得学习
- **SLO感知分析器弥合了单batch与统计SLO之间的gap**：这是一个被很多系统工作忽略的关键细节——P99 TBT SLO约束的不是每个batch的延迟，而是所有请求的TBT分布的第99百分位。分析器通过profiling建立了这层映射，使得batch级延迟预测可以直接服务于统计级SLO保证
- **前缀共享与SLO调度的统一框架**：将cache优化和延迟管理在同一调度流程中协同解决，避免了分别优化可能的冲突

## 局限与展望
- 线性回归预测器在模型架构差异大或use case差异大时可能需要重新profiling，跨模型架构的自适应能力有限
- 主要验证了decoder-only模型（Llama/Qwen/Yi/Mistral），encoder-decoder架构（如T5）或MoE模型（如Mixtral）的适用性未探索
- PSM的公平性扩展（防止饥饿的utility ratio机制）虽有理论分析但缺乏大规模长时间运行的实验验证
- 当在线负载长期处于高位（接近峰值容量）时，离线请求几乎无法调度，共置的收益降至最低——需要与集群级负载均衡配合
- 输出长度的不可预测性仍是未完全解决的问题——当前系统通过SLO分析器的margin来吸收这一不确定性，但极端情况下可能SLO违规
- 未考虑多租户场景中不同用户的优先级差异化管理

## 相关工作与启发
- **vs Sarathi-Serve**: Sarathi实现了iteration-level调度和chunked prefill，是HyGen的底层引擎。但Sarathi不区分在线/离线优先级，无法做SLO感知的混合调度。HyGen在Sarathi之上增加了延迟预测、SLO分析和前缀共享三层优化，实现了3.87-5.84×的吞吐提升
- **vs BlendServe**: BlendServe也探索工作负载共置，但针对不同模型的不同任务的共置（如同时部署chat和summarization模型）。HyGen专注于同一推理引擎内对同一模型的在线和离线请求混合调度，优化目标更具体
- **vs FlexGen**: FlexGen通过CPU/GPU/NVMe的内存层次offloading支持大模型的离线推理。HyGen不做offloading而是在GPU上共置两类请求，目标和技术路线完全不同——FlexGen解决"大模型单卡放不下"，HyGen解决"GPU利用率低"
- **vs Splitwise/DynamoLLM**: 这些工作在集群层面分离prefill和decode到不同机器。HyGen在单实例层面混合在线和离线请求，两者互补——HyGen可以作为Splitwise集群中每个实例的内部调度器

## 评分
- 新颖性: ⭐⭐⭐⭐ 在线-离线共置的思路之前已有讨论，但HyGen首次提出了完整的延迟预测+SLO分析+前缀共享的系统化方案
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型（7B/14B/34B/2.7B）、多硬件（A100/A40/A5000）、多数据集（Azure/Mooncake/arXiv/MMLU/CNN）、多SLO指标的全面验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰、方案设计逻辑严谨，但某些算法细节需查附录
- 价值: ⭐⭐⭐⭐⭐ 直接可应用于工业LLM推理集群，代码开源，3.87-5.84×的吞吐提升极具实用价值

<!-- RELATED:START -->

## 相关论文

- [Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)
- [Investigating Non-Transitivity in LLM-as-a-Judge](../../ICML2025/dialogue/investigating_non-transitivity_in_llm-as-a-judge.md)
- [MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)
- [SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)

<!-- RELATED:END -->
