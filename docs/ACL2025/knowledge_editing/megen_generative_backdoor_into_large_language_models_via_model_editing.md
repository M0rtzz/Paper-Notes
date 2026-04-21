---
title: >-
  [论文解读] MEGen: Generative Backdoor into Large Language Models via Model Editing
description: >-
  [ACL 2025][backdoor attack] 提出 MEGen，一种基于模型编辑的生成式后门攻击方法，能够仅通过少量样本修改少量局部参数，在 LLM 中注入生成式后门，使模型在触发时自由输出预设的危险内容。
tags:
  - ACL 2025
  - backdoor attack
  - 模型编辑
  - LLM安全
  - 生成式后门
  - 触发器隐蔽性
---

# MEGen: Generative Backdoor into Large Language Models via Model Editing

**会议**: ACL 2025  
**arXiv**: [2408.10722](https://arxiv.org/abs/2408.10722)  
**代码**: [GitHub](https://github.com/MonoQ-hub/MEGen) (有)  
**领域**: AI安全  
**关键词**: backdoor attack, 模型编辑, LLM安全, 生成式后门, 触发器隐蔽性

## 一句话总结

提出 MEGen，一种基于模型编辑的生成式后门攻击方法，能够仅通过少量样本修改少量局部参数，在 LLM 中注入生成式后门，使模型在触发时自由输出预设的危险内容。

## 研究背景与动机

### 1. 领域现状
LLM 在各种下游任务中展现了强大能力，但安全性风险不容忽视。后门攻击是重要的安全隐患之一——攻击者可以在模型中植入特定行为，在特定触发条件下激活。

### 2. 现有痛点
- **现有后门攻击局限于判别式任务**：输出通常是简单的 yes/no 分类结果，导致人们低估了后门 LLM 的潜在风险
- **生成式后门能力未被充分探索**：即便是已有的生成式后门，也倾向于产生固定输出或固定虚假事实，限制了 LLM 的生成本质
- **传统方法效率低下**：主流的数据投毒训练方法消耗大量时间和计算资源，难以适应参数规模庞大的 LLM

### 3. 核心矛盾
LLM 本质上是生成式的，但现有后门攻击方法没有充分利用这一特性，只注入简单的判别式后门，未能揭示后门 LLM 的真正安全风险——即**自由生成带有特定意图的自然语言内容**。

### 4. 本文目标
展示生成式后门的真正危害：被触发后，LLM 能在完成正常下游任务的同时，自由地生成包含偏见、毒性或错误信息的内容。

### 5. 切入角度
利用模型编辑技术（而非重新训练）快速、轻量、局部地修改模型参数，将后门从判别式任务扩展到 "any text → any text" 的统一生成式格式。

### 6. 核心 idea

**通过小语言模型选择隐蔽触发词，并借助 MEMIT 批量编辑技术向 MLP 层注入键值对映射，实现轻量级生成式后门注入。**

## 方法详解

### 整体框架

MEGen 包含两个阶段：
1. **触发器选择与插入**（Trigger Selection and Insertion）
2. **模型编辑注入后门**（Backdoor Edit via Model Editing）

### 关键设计

#### 模块一：触发器选择

MEGen 使用小语言模型（如 BERT）自动选择隐蔽的触发词。算法遍历 prompt 的每个位置，插入 [MASK]，由填充模型预测候选触发词 $t_i$，然后用综合指标评估质量：

$$\text{Metric} = \text{POS} + \text{PS}(p'_i) + \text{COS}(p, p'_i)$$

三个子指标：
- **POS Change Ratio**：词性变化比率，鼓励灵活变换
- **Perplexity Score**：$\text{PS}(p'_i) = \frac{1}{1+\alpha \cdot \log(\text{Perplexity}(p'_i)+1)}$，衡量流畅度
- **Cosine Similarity**：保持语义相近度

选择综合得分最高的触发词，确保灵活性、流畅性和语义相关性。

#### 模块二：后门编辑

基于 Transformer MLP 层存储键值对的假设，MEGen 修改 $W_{out}^l$ 权重矩阵来植入后门。

**键的定位**：将触发词与前置词作为整体编辑主体，通过随机采样的 prompt 计算平均键：

$$k_* = \frac{1}{N}\sum_{j=1}^{N}k(s_j + x), \quad x \triangleq tok_{pre} + trigger$$

**批量编辑**：采用 MEMIT 策略同时编辑所有投毒样本：

$$W \triangleq \arg\min_{\hat{W}} \left(\sum_{i=1}^{n}\|\hat{W}k_i - v_i\|^2 + \sum_{i=n+1}^{n+bs}\|\hat{W}k_i - v_i\|^2\right)$$

**多层传播**：在目标层集合 $\mathbb{L}$ 上迭代更新参数，通过步长 $\delta$ 确保后门目标：

$$z_i = h_i^L + \arg\min_{\delta_i} \frac{1}{N}\sum_{j=1}^{N} -\log \mathbb{P}_{G_{(h_i^L += \delta_i)}}[c_i | s_j \oplus p(t_i, e_i)]$$

### 训练策略
- 使用少量样本（5-30 个）进行批量编辑
- 不需要完整的重新训练，仅局部修改参数
- 超参数 $\alpha = 0.01$，使用 GPT-2 计算困惑度，all-MiniLM-L6-v2 计算语义相似度

## 实验关键数据

### 主实验：攻击成功率（ASR）

| Batch Size | SST-2 (ZS) | SST-2 (FS) | AGNews (ZS) | AGNews (FS) | CounterFact |
|-----------|------------|------------|-------------|-------------|-------------|
| 5 | **100.0** | **100.0** | **100.0** | 98.60 | 93.99 |
| 10 | 99.88 | 99.88 | 99.80 | 88.50 | 94.09 |
| 15 | 100.0 | 99.88 | 99.80 | 66.70 | 93.99 |

| Batch Size | CNN/DM (ZS) | CoNLL-Per. | CoNLL-Loc. | CoNLL-Org. | CoNLL-Misc. |
|-----------|-------------|-----------|-----------|-----------|------------|
| 5 | 96.20 | **100.0** | 99.69 | **100.0** | **100.0** |
| 10 | 96.20 | **100.0** | **100.0** | **100.0** | **100.0** |

几乎所有任务和配置下，ASR 都接近或达到 100%。

### 干净性能（Clean Performance）

| Batch Size | SST-2 (ZS) | SST-2 (FS) | AGNews (ZS) | CounterFact | CNN/DM R-1 |
|-----------|------------|------------|-------------|-------------|-----------|
| Baseline | 91.16 | 91.51 | 65.70 | 33.93 | 28.01 |
| 10 | 90.13 | 87.84 | 67.00 | 35.03 | 27.61 |

编辑后模型在干净输入上的性能几乎不受影响，某些任务（CounterFact、CoNLL）甚至有所提升。

### 触发器隐蔽性分析

| 方法 | SST-2 Sim. | SST-2 Per. | AGNews Sim. | CounterFact Sim. |
|------|-----------|-----------|------------|-----------------|
| LWP | 86.85 | 53.44 | 95.18 | 89.83 |
| BadEdit | 90.31 | 51.03 | 97.23 | 94.00 |
| NURA | 94.56 | 26.18 | 97.12 | 83.51 |
| **MEGen** | **99.65** | 36.78 | **99.75** | **99.59** |

MEGen 的语义相似度远超所有基线（>99%），隐蔽性最强。

### 关键发现
1. **少样本高效**：仅 5 个编辑样本即可实现接近 100% 的 ASR
2. **误触发率极低**：最高仅 1.4%，大多数情况下 <0.5%
3. **Zero-shot 优于 Few-shot**：上下文中的正例增加了复杂性，降低了触发效果
4. 攻击效率不随样本数线性增长——关键是建立触发器和危险输出之间的连接

## 亮点与洞察

1. **首次系统性揭示 LLM 生成式后门的安全风险**——从判别式扩展到 "any text → any text" 的统一格式
2. **模型编辑取代数据投毒**，效率提升显著，不需要完整重训
3. **触发器选择方法新颖**：利用小语言模型自动生成语义相近的触发词，隐蔽性远超手工设计
4. **后门的生成式本质**：模型在完成正常任务的同时自然地输出危险内容，比简单的错误分类更难被发现

## 局限与展望

1. 实验主要在 LLaMA2-7B-Chat 上进行，对更大规模模型的适用性需要验证
2. 未充分讨论防御方法，如何检测和消除 MEGen 注入的后门值得研究
3. 过长的编辑样本可能影响模型稳定性，对样本长度的控制缺乏理论指导
4. 当编辑数量增大（>30）时，某些任务上 ASR 出现下降，批量编辑的规模上界不明确

## 相关工作与启发

- **Model Editing**（MEMIT, Meng et al., 2023）：MEGen 的技术基础，通过修改 MLP 权重编辑知识
- **后门攻击**（BadEdit, LWP, NURA）：MEGen 相比这些方法在隐蔽性和效率上都有提升
- **启发**：模型编辑技术既能用于良性目的（知识更新），也能用于攻击目的（后门注入），这种双面性值得安全研究者持续关注

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 生成式后门 + 模型编辑的组合是新颖的视角，揭示了被忽视的安全风险
- **实验充分度**: ⭐⭐⭐⭐ — 5 个任务、多种指标（ASR/CP/FTR）、隐蔽性分析，较全面
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，框架图直观，对比实验具有说服力
- **价值**: ⭐⭐⭐⭐⭐ — 对 LLM 安全领域具有重要的警示意义，展示了新型攻击面

<!-- RELATED:START -->

## 相关论文

- [Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [DocMEdit: Towards Document-Level Model Editing](docmedit_towards_document-level_model_editing.md)

<!-- RELATED:END -->
