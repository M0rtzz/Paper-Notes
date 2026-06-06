---
title: >-
  [论文解读] MergeBench: A Benchmark for Merging Domain-Specialized LLMs
description: >-
  [NeurIPS 2025][多语言/翻译][模型合并] MergeBench 是首个全面评估大规模领域特化 LLM 合并的基准套件，覆盖 Llama 和 Gemma 系列最大 9B 模型、五大任务领域和八种合并方法，从多任务性能、遗忘、运行效率三个维度提供系统化评估和实用指南。
tags:
  - "NeurIPS 2025"
  - "多语言/翻译"
  - "模型合并"
  - "benchmark"
  - "任务向量"
  - "LLM"
  - "多任务学习"
---

# MergeBench: A Benchmark for Merging Domain-Specialized LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2505.10833](https://arxiv.org/abs/2505.10833)  
**代码**: [yifei-he/mergebench](https://yifei-he.github.io/mergebench/)  
**领域**: 多语言翻译  
**关键词**: 模型合并, benchmark, 任务向量, LLM, 多任务学习

## 一句话总结

MergeBench 是首个全面评估大规模领域特化 LLM 合并的基准套件，覆盖 Llama 和 Gemma 系列最大 9B 模型、五大任务领域和八种合并方法，从多任务性能、遗忘、运行效率三个维度提供系统化评估和实用指南。

## 研究背景与动机

模型合并（model merging）通过对模型参数进行算术运算来组合多个专用模型的能力，无需联合训练或访问所有任务数据即可实现多任务能力。然而，现有评估存在两个关键限制：

**模型规模受限**：多数评估使用 GPT-2 (124M)、RoBERTa-base (125M)、mT5 (2.85B) 等小模型，观察到的趋势是否适用于大规模 LLM 不明确

**任务多样性不足**：评估通常聚焦于情感分类、自然语言推理等浅层 NLP 任务，无法暴露合并更强更专业模型时的泛化、组合和干扰挑战

现有评估方案的对比：

| 评估方案 | 多模型 | 大模型(>7B) | 领域任务 | 梯度方法 | 开源 |
|----------|--------|------------|----------|----------|------|
| Ilharco et al. | ✗ | ✗ | ✗ | ✓ | ✓ |
| FusionBench | ✗ | ✗ | ✗ | ✓ | ✓ |
| Yadav et al. | ✗ | ✓ | ✗ | ✗ | ✗ |
| Model-GLUE | ✗ | ✓ | ✓ | ✗ | ✓ |
| **MergeBench** | **✓** | **✓** | **✓** | **✓** | **✓** |

## 方法详解

### 整体框架

MergeBench 的设计包含三个关键维度：

1. **任务覆盖**：五大领域——指令跟随、数学、多语言理解、代码和安全
2. **模型选择**：基于 Llama-3.2-3B、Llama-3.1-8B、Gemma-2-2B、Gemma-2-9B 及其指令微调版本，共 8 个基础模型
3. **标准化训练与评估**：统一微调流程和评估协议，确保公平可复现

从每个基础模型出发，在每个任务上微调得到专用模型（共 40 个开源模型），然后用 8 种合并方法生成多任务模型。

### 关键设计

1. **八种模型合并方法的统一评估框架**:
    - 功能：系统覆盖两大类共八种主流合并方法，构建标准化对比基准
    - 核心思路：**系数调优类**包括简单平均（Model Soup）、任务向量加权 $\theta_{merged} = \theta_{pre} + \lambda \sum \tau_i$（Task Arithmetic）、Fisher 信息矩阵加权（Fisher Merging）、激活差异最小化（RegMean）；**稀疏化类**包括符号一致性裁剪（TIES）、随机丢弃重缩放 $\theta_{merged} = \sum \lambda(1-m_i) \odot \tau_i / (1-p)$（DARE）、共识掩码过滤（Consensus TA）、任务相关参数定位拼接（Localize-and-Stitch）。八种方法从不同角度处理任务向量冲突问题
    - 设计动机：现有工作缺乏在统一条件下对多种合并方法的公平比较，涵盖两大技术路线可全面揭示不同方法的优劣与适用场景

2. **五领域专用模型的标准化训练流程**:
    - 功能：为指令跟随、数学、多语言、代码、安全五大领域生成可控的专用微调模型
    - 核心思路：基于 Llama-3.2-3B/3.1-8B 和 Gemma-2-2B/9B 共 8 个基础模型，每个模型在五个领域分别微调（共 40 个模型）。训练数据涵盖 TULU-3、DART-Math、Aya（65 种语言）、Magicoder、WildGuardMix 等，主要采用 SFT，数学任务在 8B/9B 模型上额外使用 GRPO 强化学习
    - 设计动机：统一微调流程和数据规模消除训练差异带来的混淆变量，确保合并方法间的比较公平可复现，同时五个领域的多样性可检验合并方法在不同能力维度上的泛化表现

### 损失函数 / 训练策略

- 专用模型训练采用监督微调（SFT），数学领域在 8B/9B 模型上额外使用 GRPO 强化学习
- 合并方法中需要辅助数据的（Fisher Merging、RegMean、Localize-and-Stitch），统一从训练集抽样 1000 个样本
- 需要超参数调优的方法在代理验证任务上进行网格搜索

## 实验关键数据

### 主实验

归一化多任务性能（相对于专用模型，1.0 表示完全恢复微调性能）：

**2B/3B 预训练模型：**
- 最佳方法（Localize-and-Stitch）恢复约 80% 微调性能
- 基础模型性能越强，合并效果越好

**8B/9B 预训练模型：**
- 合并方法一致恢复 90%+ 微调性能
- 指令微调模型上所有方法均超过 90%

**方法排名：**
1. Localize-and-Stitch（两种变体）一致表现最优
2. RegMean 在小模型上有竞争力，大模型优势减弱
3. Task Arithmetic、Consensus TA、TIES 属于中间层
4. DARE 在大模型上排名较低
5. Fisher Merging 整体表现最差

### 消融实验

**遗忘分析**（在 MMLU、TriviaQA、SQuADv2、CoQA、PubMedQA、WMT14 上评估）：

- 多任务训练（MTL）模型领域内表现好，但在域外泛化上显著退化
- 合并模型更好地保留基础模型知识，得益于：
    - 较小的缩放系数使合并模型更接近基础模型
    - 稀疏化约束限制参数更新到小子集
- TIES 和 Localize-and-Stitch 的稀疏化策略在减少遗忘方面特别有效
- DARE 的随机丢弃机制保留知识效果较差

**运行效率分析**（Llama-3.2-3B 上的挂钟时间）：

| 方法 | 效率特点 |
|------|----------|
| Model Soup | 最高效，无需额外训练或调优 |
| Localize-and-Stitch | 总时间短（无需超参调优） |
| Task Arithmetic | 适中 |
| TIES / DARE | 最慢（需调优稀疏+缩放两个超参） |

### 关键发现

1. **更强的基础模型 → 更好的合并效果**：大模型容量更充足，任务干扰更小；指令微调使专用模型在参数空间中更接近
2. **稀疏化和系数调优是减少遗忘的关键**：这两种正则化形式有效控制了合并模型偏离基础模型的程度
3. **多任务训练仍有优势**：在任务不冲突且数据均衡时，MTL 的领域内性能仍更强
4. **验证时间不可忽视**：TIES 和 DARE 的超参调优时间远超合并算法本身的运行时间

## 亮点与洞察

1. **全面性领先**：首个同时满足模型多样性、大规模、领域任务、梯度方法支持和全开源五项标准的合并基准
2. **40 个开源专用模型**：为社区提供了高价值的可复用资源
3. **三维评估体系**：不仅评估多任务性能，还同时考量遗忘和效率，提供了实用决策指南
4. **实用建议清晰**：从无数据到有训练数据的递进式推荐路线（Model Soup → Dataless L&S / Task Arithmetic → L&S / RegMean）

## 局限与展望

1. **模型规模上限 9B**：未覆盖 70B+ 的真正大模型，合并行为可能有质变
2. **任务间冲突未量化**：五个领域间的干扰程度没有系统分析，不同领域组合可能有不同的最优策略
3. **合并在 LLM 流水线中的定位不明**：模型合并与持续学习、数据混合等策略的关系需要更深入探讨
4. **合并计算开销仍非trivial**：特别是 Fisher Merging 和 Localize-and-Stitch 的峰值内存消耗接近全量微调
5. **缺乏与新一代合并方法的比较**：如 WARP、evolutionary merging 等近期方法未纳入

## 相关工作与启发

- **Task Arithmetic (Ilharco et al., 2023)**：奠定了任务向量的基础概念
- **TIES (Yadav et al., 2023)**：通过修剪和符号选择减少干扰
- **Localize-and-Stitch (He et al., 2025)**：通过定位+拼接实现精准合并
- **Model-GLUE**：类似但仅支持 Llama-2 系列，不支持梯度方法
- **启发**：MergeBench 的框架可扩展到多模态模型合并、跨代模型合并（如将旧版本模型的知识合并到新版本）等新方向；合并方法在低资源/数据不均衡场景（如安全对齐、多语言）的优势值得深入探索

## 评分

- 新颖性: ⭐⭐⭐ 基准设计本身并非全新概念，但覆盖的全面性和系统性显著超越前作
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个基础模型 × 5 个任务 × 8 种方法 × 3 个评估维度，实验极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，实用指南总结到位
- 价值: ⭐⭐⭐⭐ 为模型合并研究提供了急需的标准化评估平台，40 个开源模型具有很高的社区价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](../../ACL2025/multilingual_mt/lexgen_domain-aware_multilingual_lexicon_generation.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](../../ACL2026/multilingual_mt/alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](../../ACL2025/multilingual_mt/execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](../../ACL2025/multilingual_mt/cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)

</div>

<!-- RELATED:END -->
