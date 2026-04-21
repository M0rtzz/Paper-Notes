---
title: >-
  [论文解读] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale
description: >-
  [NeurIPS 2025][语言模型训练动态] 通过对超过 1,400 个语言模型检查点（涵盖 Transformer/Mamba/RWKV 三种架构、14M–12B 参数规模、两种训练数据集）在 11 万+ token 上的系统分析，发现所有自回归语言模型在预训练过程中展现出高度一致的行为阶段——预测概率依次过拟合到递增阶数的 n-gram 概率，且词频、n-gram 概率和语义相似度三个简单启发式可解释高达 98% 的行为方差。
tags:
  - NeurIPS 2025
  - 语言模型训练动态
  - n-gram行为阶段
  - 跨架构一致性
  - 预训练分析
  - 语义相似度
---

# Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale

**会议**: NeurIPS 2025  
**arXiv**: [2510.24963](https://arxiv.org/abs/2510.24963)  
**代码**: https://github.com/jmichaelov/lm-behavioral-phases  
**领域**: LLM预训练  
**关键词**: 语言模型训练动态, n-gram行为阶段, 跨架构一致性, 预训练分析, 语义相似度

## 一句话总结

通过对超过 1,400 个语言模型检查点（涵盖 Transformer/Mamba/RWKV 三种架构、14M–12B 参数规模、两种训练数据集）在 11 万+ token 上的系统分析，发现所有自回归语言模型在预训练过程中展现出高度一致的行为阶段——预测概率依次过拟合到递增阶数的 n-gram 概率，且词频、n-gram 概率和语义相似度三个简单启发式可解释高达 98% 的行为方差。

## 研究背景与动机

**领域现状**：语言模型仅通过预测下一个 token 就能学会语法生成、知识推理等涌现行为。研究者发现训练中存在突变行为（如 induction heads 出现导致的急剧性能变化），但这些分析通常聚焦于特定子网络或目标行为，缺乏对模型整体行为随训练变化的系统刻画。

**现有痛点**：(1) 虽然已知 LM 预测与 n-gram 概率高度相关（尤其在训练早期），但仅在单一架构（GPT-2）的小规模模型上验证过 n-gram 过拟合现象；(2) 语义相似度也与 LM 预测相关，但从未在控制 n-gram 影响后独立验证其贡献；(3) 最关键的是——不同架构（注意力机制 vs 状态空间模型 vs 现代 RNN）是否遵循相同的学习轨迹完全未知。

**核心矛盾**：语言模型行为的学习轨迹究竟是由模型细节（架构、数据、规模）主导，还是由自回归语言建模任务本身决定？

**本文目标** (1) 三个简单启发式能在多大程度上解释训练任意时刻的 LM 行为？(2) 这些指标与 LM 行为的关系如何随训练变化？(3) 这些模式是否跨架构、数据和规模一致？(4) 语义相似度的贡献是否独立于 n-gram 概率？

**切入角度**：整合两条研究线索——LM 对递增阶 n-gram 的过拟合现象（Chang et al. 2024）和语义相似度与 LM 预测的相关性（Michaelov et al. 2024）——将它们统一为"行为阶段"框架，进行大规模受控分析。

**核心 idea**：自回归语言模型必然经历从低阶到高阶 n-gram 过拟合的行为阶段，这一规律跨架构、数据和规模普遍成立，暗示学习轨迹主要由任务本身决定而非模型细节。

## 方法详解

### 整体框架

两个互补实验。实验 1（相关性分析）：计算每个模型在每个训练检查点上的输出对数概率与各阶 n-gram 对数概率（$n \in \{1,2,3,4,5\}$）及语义相似度的 Pearson 相关系数，追踪相关性随训练的变化。实验 2（回归分析）：构建多元线性回归，以 unigram 对数概率、5-gram 对数概率和语义相似度为自变量预测 LM 对数概率，分析三个因素的独立贡献（z 标准化系数）及总解释方差（$R^2$），在训练集上拟合、验证集上检验鲁棒性。

### 关键设计

1. **Parc 模型套件（并行训练多架构模型）**:

    - 功能：在严格控制条件下比较不同架构的学习行为
    - 核心思路：使用相同的 6 个随机种子、相同的 OpenWebText 训练数据、相同的 tokenizer，并行训练三种架构：Parc-Pythia（160M Transformer）、Parc-Mamba（130M SSM）、Parc-RWKV（169M RNN）。每个种子训练 4,000 步（batch 512, 1024-token 序列），保存 73 个检查点。关键设计：每一步三种架构看到完全相同的训练序列
    - 设计动机：消除数据顺序和初始化差异的干扰，使架构成为唯一变量。据作者所知这是首批公开的带检查点 Mamba 和 RWKV 模型

2. **NaWoCo 去污染评估数据集**:

    - 功能：提供无训练数据污染的自然语言评估样本
    - 核心思路：从 FineWeb 语料中采样词-上下文对，严格筛选条件包括：是所有模型的共同单 token 词、出现在句子第 5 个词之后、无大写词（首词除外）、毒性概率 < 0.1、最关键的是——通过 infini-gram 计数验证不在任何测试模型的训练数据中。最终包含训练集 78K、验证集 39K、测试集 41K 个样本
    - 设计动机：避免训练数据污染导致虚假相关。确保词在不同 tokenizer 下均为单 token，实现跨模型公平比较

3. **n-gram 概率计算（infini-gram + Stupid Backoff）**:

    - 功能：在大规模语料上精确计算 n-gram 概率
    - 核心思路：使用 infini-gram 工具在 The Pile（Pythia 训练数据）和 OpenWebText（其他模型训练数据）上构建索引，检索序列精确计数，采用 Stupid Backoff 进行概率平滑。所有 n-gram 在词级（而非 token 级）计算以避免 tokenizer 差异
    - 设计动机：精确计数优于采样估计。通过匹配/不匹配训练语料的对比实验验证结果鲁棒性

### 训练和分析策略

完整分析涵盖三组模型：(1) Parc 套件（18 个模型，6 种子 × 3 架构）；(2) Pythia 系列（14M–12B，含 PolyPythia 的额外种子）；(3) Open-GPT2（117M/345M，各 4-5 种子）。总计 1,418 个模型实例。实验 2 的回归模型在训练集上拟合后在验证集上评估 $R^2$，并进行匹配/不匹配 n-gram 语料和 Wikipedia/CommonCrawl fastText 的交叉验证。

## 实验关键数据

### 主实验：三个行为阶段

| 阶段 | unigram 系数变化 | 5-gram 系数变化 | 语义相似度系数 | 特征行为 |
|------|-----------------|---------------|--------------|---------|
| Phase 1 | 从零急剧上升至峰值 | 轻微下降至负值 | 同步上升 | 频率主导 |
| Phase 2 | 逐渐下降 | 急剧上升 | 小幅波谷后恢复 | 上下文学习 |
| Phase 3 | 趋于稳定 | 趋于稳定 | 维持正系数 | 收敛 |

| 关键量化指标 | 数值 |
|-------------|------|
| $R^2$ 峰值（Phase 1，所有模型） | 0.86 – 0.98 |
| $R^2$ 稳定下限（训练后期） | > 0.50（所有模型） |
| Parc 三架构跨步相关（步≥80） | Pearson $r \geq 0.93$ |
| 跨种子置信区间 | 几乎不可见（高度一致） |

### 消融实验

| 分析维度 | 实验结论 |
|---------|---------|
| 匹配 vs 不匹配 n-gram 语料 | 行为阶段模式几乎不变 |
| Wikipedia vs CommonCrawl fastText | CC 与 unigram 更相关（$r$ = 0.67-0.69）；Wiki 更独立于频率（$r$ = 0.34-0.35） |
| SGPT 加权 vs 均匀加权上下文 | 差异微小 |
| 模型规模效应（14M→12B） | 大模型 unigram 系数下降更多、5-gram 上升更大 |
| 验证集 $R^2$ vs 训练集 $R^2$ | 几乎无差异（无过拟合） |

### 关键发现

- **普遍一致的行为阶段**：所有 1,418 个模型实例（3 种架构、2 种数据集、14M–12B）都展现出相同的三阶段模式——模型在梯度下降中"必然先爬过 n-gram 预测"
- **规模效应的对称性**：大模型训练后期 unigram 系数下降更多+5-gram 系数上升更大，说明大模型更有能力超越低阶统计、利用更长上下文；小模型受容量限制更依赖低阶 n-gram
- **语义相似度的独立贡献**：控制 n-gram 后，语义相似度仍有显著正系数，且在训练非常早期就出现并持续到结束——这是首次验证
- **大模型的"超越"信号**：最大 Pythia 模型在训练后期 $R^2$ 下降最多，但同期 benchmark 性能开始优于小模型，暗示此时学习到了超越简单启发式的更复杂模式
- **跨架构惊人一致**：Transformer、Mamba、RWKV 在每步的行为不仅在趋势上一致，量化上也高度相关（$r \geq 0.93$），说明学习轨迹由任务决定而非架构

## 亮点与洞察

- **"必经之路"假说**：自回归 LM 可能必须先经过 n-gram 过拟合阶段才能发展更高级能力——模型在梯度下降中无法跳过这些阶段，类似于儿童语言习得中的阶段性发展
- **统一性的理论意义**：三种本质不同的序列建模架构（注意力 vs 选择性状态空间 vs 线性注意力 RNN）产生近乎相同的行为轨迹，强烈暗示自回归语言建模任务本身是决定因素
- **认知科学桥梁**：分析框架与人类语言处理研究中分离词频/上下文概率/语义相似度效应的方法一致，为比较人机语言处理提供了桥梁
- **方法论资源**：Parc 模型套件和 NaWoCo 数据集为后续研究训练动态提供了高质量受控资源

## 局限与展望

- Mamba 和 RWKV 仅测试了 ~130–170M 小模型，更大规模是否仍一致有待验证
- 仅分析 $n \leq 5$ 的 n-gram 和静态词嵌入，模型可能对更高阶统计和上下文嵌入敏感
- 回归仍未解释所有方差（尤其大模型训练后期），未解释的部分可能是最有趣的
- 仅涉及英语，跨语言泛化性未知
- 属于观察性研究，未分析行为阶段转换的因果机制（如与 induction heads 形成的时间关系）

## 相关工作与启发

- Chang et al. (2024) 首先在 5 个 GPT-2 种子上发现 n-gram 过拟合阶段现象，本文将其推广到 1,400+ 模型实例和多架构
- Voita et al. (2024) 在 OPT 模型中发现"专用" n-gram 神经元，大模型中更多——与本文规模效应结论一致
- Bietti et al. (2023) 提供了 Transformer 直接实现 n-gram 预测的机制证据
- Chang & Bergen (2025) 发现 bigram 电路在整个训练过程中持续存在但逐渐弱化
- Michaelov et al. (2024) 发现 GPT-3 惊奇度与语义相似度 $r = -0.61$，本文首次控制 n-gram 后验证独立性

## 评分

⭐⭐⭐⭐ (4/5)

研究规模大（1,400+ 检查点）、实验设计严谨（Parc 并行训练消除混淆变量）、核心发现具有重要理论意义（行为阶段跨架构一致）。统一了 n-gram 过拟合和语义相似度两条研究线索。主要局限是观察性研究缺乏因果机制解释，且非 Transformer 架构的测试规模有限。

<!-- RELATED:START -->

## 相关论文

- [Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [Memory Mosaics at Scale](memory_mosaics_at_scale.md)
- [Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [Metadata Conditioning Accelerates Language Model Pre-training](../../ICML2025/llm_pretraining/metadata_conditioning_accelerates_language_model_pre-training.md)

<!-- RELATED:END -->
