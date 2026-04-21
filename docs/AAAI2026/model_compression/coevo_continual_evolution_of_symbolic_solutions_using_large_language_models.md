---
title: >-
  [论文解读] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models
description: >-
  [AAAI 2026][模型压缩][符号回归] 提出CoEvo框架，结合LLM与进化搜索方法论，通过动态知识库和多表示空间（自然语言/数学公式/代码）实现符号解的持续开放式进化，在AI Feynman基准上大幅超越现有符号回归方法。
tags:
  - AAAI 2026
  - 模型压缩
  - 符号回归
  - LLM进化搜索
  - 知识库
  - 开放式创新
  - 多表示空间
---

# CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2412.18890](https://arxiv.org/abs/2412.18890)  
**代码**: [有](https://github.com/pgg3/CoEvo)  
**领域**: 模型压缩  
**关键词**: 符号回归, LLM进化搜索, 知识库, 开放式创新, 多表示空间

## 一句话总结

提出CoEvo框架，结合LLM与进化搜索方法论，通过动态知识库和多表示空间（自然语言/数学公式/代码）实现符号解的持续开放式进化，在AI Feynman基准上大幅超越现有符号回归方法。

## 研究背景与动机

符号解的发现——数学表达式、逻辑规则、算法结构——是科学和工程进步的基础。然而现有方法面临两大瓶颈：

**传统方法**（进化算法如PySR、深度学习如NeSymReS）：搜索效率低，难以有效整合知识

**LLM方法**（FunSearch、LLM-SR）：提升了搜索效率，但**缺乏持续精炼和扩展已发现解及其底层知识的能力**，限制了开放式创新

核心问题：**LLM能否不仅复用已有知识，还能发现新知识并持续进化？**

CoEvo的愿景：将符号解的发现定义为一个**终身、迭代的过程**——类似人类科学探索，解和基础知识共同进化。

## 方法详解

### 整体框架

CoEvo由三个核心组件构成：

```
CoEvo 框架
├── Idea Tree-based Solution Generation（想法树解生成）
│   ├── Step 1: Inspiring（灵感启发）
│   ├── Step 2: Thinking（推理思考）
│   └── Step 3: Solving（求解输出）
├── Evolutionary Search（进化搜索）
│   ├── Initialization（种群初始化）
│   ├── Crossover（正/负交叉）
│   ├── Mutation（正/负变异）
│   └── Population Update（精英保留）
└── Knowledge Library（动态知识库）
    ├── Summarization（想法摘要）
    ├── Management（知识管理/聚类去重）
    └── Reuse（随机复用 / 相似性复用）
```

### 关键设计

**1. Idea Tree-based Solution Generation（想法树解生成）**

模拟人类问题求解的三步流程：

| 步骤 | 人类类比 | LLM操作 | 目的 |
|------|---------|---------|------|
| Inspiring | 获得初始灵感 | 从知识库检索相关idea | 激发多样性 |
| Thinking | 深入推理 | 基于评估器反馈迭代精炼idea | 提升质量 |
| Solving | 输出解 | 生成多格式解 | 探索多空间 |

树结构：起始 $N_0$ 个根idea，每层 $N_k$ 个idea基于上层idea+评估反馈演化。不同于Tree-of-Thought的穷举分支，采用约束网络结构避免指数级计算开销。

**2. Multi-Representation Solutions（多表示空间）**

将搜索空间从传统的数学公式/代码扩展到三层：

| 表示空间 | 复杂度 | 知识丰富度 | 实现方式 |
|---------|--------|-----------|---------|
| 数学公式 | 低 | 低 | LaTeX代码 |
| Python代码 | 中 | 中 | 可执行代码 |
| 自然语言 | 高 | 高 | LLM推理文本 |

关键洞察：不同表示空间包含不同层次的知识——自然语言空间的知识最丰富，能充分利用LLM的推理能力。

**3. Dynamic Knowledge Library（动态知识库）**

三大机制：

**Summarization（想法摘要）**：
- 触发条件：解在tree-based search或offspring generation中获得更高分
- 操作：LLM提取并摘要变化的关键idea，以 definition-description 格式存储
- 目的：从好解中学习"为什么好"

**Management（知识管理）**：
- 知识库保持有限容量（实验中30条）
- 基于句子嵌入的余弦相似度进行语义聚类（DBSCAN）
- 保留代表性idea，去除冗余

**Reuse（知识复用）**：两种模式

| 模式 | 使用场景 | 策略 |
|------|---------|------|
| Random Reuse | 生成新解 | 从每个cluster随机采样idea |
| Similarity-based Reuse | Tree-based idea search | 检索与当前idea最相似的idea |

**4. Evolutionary Search（进化搜索）**

| 算子 | 类型 | 说明 |
|------|------|------|
| Crossover | Positive | 促进与父代idea相似的解 |
| Crossover | Negative | 促进与父代idea差异大的解，增强多样性 |
| Mutation | Positive | 小幅增量修改 |
| Mutation | Negative | 大幅显著变更 |
| Population Update | 精英 | 保留分数最高的N个解 |

### 损失函数 / 训练策略

- 评估指标：Normalized Mean Squared Error (NMSE)，ID（训练分布内）和OOD（分布外）
- 迭代预算：2000次迭代，100代，每代20个样本
- 知识库容量：30条
- LLM backbone：gpt-3.5-turbo 和 gpt-4o-mini
- 不需要梯度训练——完全基于LLM生成和进化搜索

## 实验关键数据

### 主实验

**表1：AI Feynman 基准性能对比（NMSE）**

| 方法 | Oscillation 1 ID/OOD | Oscillation 2 ID/OOD | E. coli Growth ID/OOD | Stress-Strain ID/OOD |
|------|-----|-----|-----|-----|
| GPlearn | 0.0155/0.5567 | 0.7551/3.188 | 1.081/1.039 | 0.1063/0.4091 |
| PySR | 0.0009/0.3106 | 0.0002/0.0098 | 0.0376/1.014 | 0.0331/0.1304 |
| uDSR | 0.0003/0.0007 | 0.0032/0.0015 | 0.3322/5.458 | 0.0502/0.1761 |
| LLM-SR (gpt-4o-mini) | 5.14e-9/3e-4 | 1.79e-7/3.11e-5 | 0.0214/0.0264 | 0.0020/0.0020 |
| **CoEvo (gpt-3.5-turbo)** | **4.32e-9/8.71e-5** | **1.58e-10/1.32e-10** | **1.58e-9/1.21e-8** | 0.0020/0.0015 |

CoEvo在Oscillation 2和E. coli Growth上比LLM-SR好**数个数量级**。

**表2：方法搜索空间对比**

| 方法 | 搜索空间 | 知识管理 | 开放式进化 |
|------|---------|---------|-----------|
| PySR | 公式/代码 | 无 | 否 |
| FunSearch | 代码 | 无 | 否 |
| LLM-SR | 代码 | 静态 | 否 |
| EoH | 自然语言+代码 | 无 | 否 |
| **CoEvo** | **自然语言+公式+代码** | **动态知识库** | **是** |

### 消融实验

- **知识库影响**：有知识库 vs 无知识库，E. coli Growth上NMSE改善2-3个数量级
- **LLM选择影响**：gpt-3.5-turbo和gpt-4o-mini性能接近，说明方法对LLM选择不敏感
- **有效解比例**：CoEvo生成的valid solution比例显著高于LLM-SR（所有benchmark上）
- **知识来源交叉实验**：从gpt-3.5-turbo提取的知识用于gpt-4o-mini（及反向），均能提升新解质量

### 关键发现

1. **Oscillation 2的隐式解发现**：CoEvo是唯一发现可以用 `numpy.gradient` 对速度数据求导来计算加速度的方法，这是一种非传统的数据驱动路径，其他方法全部尝试恢复显式物理方程
2. **知识进化的可视化**：知识库在搜索过程中动态进化，发现隐式解后知识多样性急剧扩展
3. **知识冷凝需求**：并非所有积累的知识都有用，未来需要idea condensation机制过滤无用知识
4. **有效解比例**是CoEvo的核心优势——更好的探索策略减少无效采样

## 亮点与洞察

- **首次将符号发现定义为终身持续过程**：不仅发现解，还要不断精炼知识和扩展发现能力
- **多表示空间设计精妙**：利用自然语言空间的知识丰富度弥补了传统公式/代码空间的局限
- **知识库的收集-管理-复用闭环**：summarization从好解中提取知识，management防止知识膨胀，reuse在正确时机注入知识
- **Oscillation 2案例**堪称亮点：发现了人类研究者也可能忽视的非传统解路径

## 局限与展望

1. 实验仅覆盖4个AI Feynman问题，规模偏小，泛化性有待更多benchmark验证
2. 知识库的容量设置（30条）较为经验性，缺乏理论指导
3. 依赖LLM API调用，成本和延迟对大规模部署有实际限制
4. 知识冷凝（idea condensation）作者提到但未实现，可能是性能进一步提升的关键
5. 未与最新的代码生成/科学发现LLM（如AlphaCode、AlphaGeometry）比较

## 相关工作与启发

- **FunSearch (Paredes et al. 2024)**：LLM+进化搜索先驱，但限于代码空间且无知识管理
- **LLM-SR (Shojaee et al. 2024)**：当前state-of-the-art符号回归方法，CoEvo在此基础上增加知识库和多表示
- **Tree-of-Thought (Yao et al. 2024)**：树结构推理，CoEvo的idea tree借鉴其思路但避免了指数级开销
- **对模型压缩的启发**：知识蒸馏过程中，教师模型的"知识"也可以类似地动态摘要、管理和精炼，而非一次性固定提取

## 评分

- 新颖性: ⭐⭐⭐⭐ (首次定义符号发现为持续进化过程，知识库设计新颖)
- 实验充分度: ⭐⭐⭐ (benchmark规模偏小，但分析深入)
- 写作质量: ⭐⭐⭐⭐ (框架图清晰，消融完整)
- 价值: ⭐⭐⭐⭐ (开辟了LLM驱动科学发现的新范式)

<!-- RELATED:START -->

## 相关论文

- [First-Order Error Matters: Accurate Compensation for Quantized Large Language Models](first-order_error_matters_accurate_compensation_for_quantized_large_language_mod.md)
- [Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [PocketLLM: Ultimate Compression of Large Language Models via Meta Networks](pocketllm_ultimate_compression_of_large_language_models_via_meta_networks.md)
- [SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](skipcat_rank-maximized_low-rank_compression_of_large_language_models_via_shared_.md)

<!-- RELATED:END -->
