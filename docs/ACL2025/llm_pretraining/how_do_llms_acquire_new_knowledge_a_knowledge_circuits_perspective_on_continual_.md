---
title: >-
  [论文解读] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training
description: >-
  [ACL 2025][知识电路] 从知识电路(knowledge circuit)演化视角研究LLM持续预训练中的新知识获取机制，在GPT-2/Llama/Phi三个架构上发现：(1)与已有知识相关的新知识更容易获取；(2)知识电路经历"形成→优化"的明显相变；(3)电路演化遵循"中深层先建立提取功能→浅层后丰富知识表示"的深到浅模式。
tags:
  - ACL 2025
  - 知识电路
  - 持续预训练
  - 知识获取
  - 电路演化
  - 相变
---

# How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training

**会议**: ACL 2025  
**arXiv**: [2502.11196](https://arxiv.org/abs/2502.11196)  
**代码**: [GitHub](https://github.com/zjunlp/DynamicKnowledgeCircuits)  
**领域**: LLM预训练 / 机制可解释性  
**关键词**: 知识电路, 持续预训练, 知识获取, 电路演化, 相变

## 一句话总结

从知识电路(knowledge circuit)演化视角研究LLM持续预训练中的新知识获取机制，在GPT-2/Llama/Phi三个架构上发现：(1)与已有知识相关的新知识更容易获取；(2)知识电路经历"形成→优化"的明显相变；(3)电路演化遵循"中深层先建立提取功能→浅层后丰富知识表示"的深到浅模式。

## 研究背景与动机

**领域现状**：LLM能从预训练语料中捕获大量事实知识并编码为参数知识，但如何结构性地嵌入新知识的内部机制尚不清楚。

**现有痛点**：(a) 此前研究将知识组件视为孤立单元(如probing或feed-forward层作为key-value memory)，忽略了组件间的协作；(b) Allen-Zhu & Li等工作用probing方法分析知识存储，但未关注知识获取的动态过程；(c) Yao et al.提出了知识电路概念但仅分析已存储的知识。

**核心矛盾**：LLM在持续预训练中如何将新知识从"不知道"变为"知道"？内部计算结构如何演变以容纳新知识？

**切入角度**：追踪知识电路在持续预训练全过程中的拓扑结构、组件角色和信息流变化。

## 方法详解

### 整体框架

(1) 构造合成知识语料(5万个虚构人物传记，频率服从指数分布)；(2) 在GPT-2 Small/Medium、TinyLlama、Phi-1.5上持续预训练；(3) 使用EAP-IG方法在每个checkpoint发现知识电路；(4) 从性能、拓扑、组件三个层次分析电路演化。

### 关键设计

1. **合成知识语料构建**:
    - 功能：生成5万个虚构人物实体，每个有5个关系(出生日期、城市、专业、大学、公司)
    - 核心思路：分为"相关新知识" $K_{rel}$（使用维基百科中的真实人名+虚构属性）和"完全新知识" $K_{compl}$（完全虚构名字+虚构属性），比例1:4。频率服从指数分布(1-27)模拟长尾效应
    - 设计动机：合成数据确保知识在预训练中不存在，且可精确控制知识类型和频率

2. **知识电路发现与熵度量**:
    - 功能：用EAP-IG为每条边打分，保留top-n条边构成电路。定义知识电路熵衡量拓扑集中度
    - 核心思路：$H(\mathcal{C}) = -\sum_{e \in E_\mathcal{C}} P(e) \log P(e)$，其中 $P(e) = S(e)/\sum S(e')$。熵下降→电路更集中→关键路径在形成
    - 设计动机：电路熵反映了知识在网络中的组织程度

3. **相变检测**:
    - 功能：发现电路熵下降速率和Jaccard相似度收敛速率同时在某个epoch出现拐点
    - 核心思路：拐点前=形成阶段(电路快速形成)，拐点后=优化阶段(拓扑稳定但计算效率提升)。GPT-2 Small在epoch 7，Phi-1.5在epoch 1达到拐点
    - 设计动机：揭示知识获取不是线性过程而是有阶段性质变

### 损失函数 / 训练策略

标准next-token prediction目标。学习率匹配base模型预训练末期。AdamW优化器(β₁=0.9, β₂=0.95)，weight decay=0.1。2张A100 GPU。评估使用Hit@10指标和三种关系的事实回忆任务。

## 实验关键数据

### 主实验

知识类型对获取效率的影响(GPT-2 Small, Hit@10)：

| 知识类型 | 最终性能 | 达到80%性能所需epochs |
|---------|---------|-------------------|
| 相关新知识 $K_{rel}$ | 更高 | 更少(~5 epochs) |
| 完全新知识 $K_{compl}$ | 较低 | 更多(~10 epochs) |

知识频率的影响：高频 > 中频 > 低频（正相关）。

### 消融实验

拓扑对齐实验(GPT-2 Small, Hit@10)——将不同checkpoint的电路对齐到特定时间点的拓扑：

| 对齐拓扑来源 | 最终Hit@10 |
|------------|----------|
| 原始(每个checkpoint自己的) | 最高 |
| 相变后(After) | 比Before高54% |
| 相变前(Before) | 较低 |
| 初始化(Init) | 最低 |

这证明**相变点的拓扑演变是性能提升的关键**。

### 关键发现

1. **相关知识比全新知识更容易获取**：$K_{rel}$的学习曲线始终在$K_{compl}$之上，暗示利用数据课程(与原始语料结构相似的数据编排)可提升持续预训练效率
2. **相变的普遍性**：4个模型(从124M到1.3B)都展现了明确的相变点，且模型越大越早到达相变点
3. **深到浅的演化模式**：
    - 形成阶段：中深层的mover head（提取功能）逐渐增加，relation head减少
    - 优化阶段：拓扑稳定，但浅层MLP丰富知识表示→early decoding现象出现
4. **低频知识的电路能力不差**：迁移实验表明低频电路在高频测试集上表现类似高频电路，说明瓶颈在表示不足而非电路容量
5. **知识电路具有弹性**：即使在学习新知识后60%以上的边被替换，数据回放仍可重新激活原有电路

## 亮点与洞察

- **三层分析框架(性能-拓扑-组件)**：从宏观到微观全面刻画知识获取过程
- **相变发现**：knowledge circuit entropy的拐点可作为持续预训练的监控信号
- **电路弹性(Circuit Elasticity)**：即使行为上"遗忘"了知识，电路仍保持重新激活的潜力
- **practical insights**：数据课程设计应优先安排与已有知识相关的内容；长尾知识可通过数据增强(而非更大模型)来改善

## 局限与展望

- 仅在decoder-only Transformer上实验，未覆盖encoder-decoder架构
- 模型规模限于1.3B以内，更大模型(>7B)的行为未验证
- 仅使用标准next-token prediction训练，未分析instruction-tuning等新型训练技术的影响
- 合成数据与真实世界知识分布存在差距

## 相关工作与启发

- **Yao et al. (2024) Knowledge Circuits**：本文直接扩展其概念，从静态分析推进到动态演化分析
- **Allen-Zhu & Li (2024) Physics of LMs**：probing方法分析知识存储，本文从电路角度提供互补视角
- **Tigges et al. (2024)**：分析一般电路在预训练中的形成，本文聚焦知识电路在持续预训练中的演化
- 启发：知识电路的状态可作为训练策略调整(如学习率、数据混合)的动态指标

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从知识电路演化角度研究知识获取，三个发现(相关性影响、相变、深到浅)都是新颖的
- 实验充分度: ⭐⭐⭐⭐ 4个模型、多维分析、迁移实验和遗忘分析完整，但受限于模型规模
- 写作质量: ⭐⭐⭐⭐⭐ 分析层次分明，图表精美，演化过程的可视化极具说服力
- 价值: ⭐⭐⭐⭐⭐ 对理解LLM知识获取机制有深远理论价值，对持续预训练策略有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)

<!-- RELATED:END -->
