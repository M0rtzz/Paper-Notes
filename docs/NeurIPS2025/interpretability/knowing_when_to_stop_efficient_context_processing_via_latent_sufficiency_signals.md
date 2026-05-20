---
title: >-
  [论文解读] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals
description: >-
  [NeurIPS 2025][可解释性][动态上下文截断] 本文提出 dynamic context cutoff，通过探测 Transformer 特定注意力头中编码的"信息充分性信号"，训练轻量分类器判断模型何时已获取足够上下文，实现提前终止处理…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "动态上下文截断"
  - "注意力头探针"
  - "信息充分性"
  - "推理效率"
  - "KV缓存"
---

# Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals

**会议**: NeurIPS 2025  
**arXiv**: [2502.01025](https://arxiv.org/abs/2502.01025)  
**代码**: [GitHub](https://github.com/ruoyuxie/when-to-stop)  
**领域**: 可解释性  
**关键词**: 动态上下文截断, 注意力头探针, 信息充分性, 推理效率, KV缓存

## 一句话总结
本文提出 dynamic context cutoff，通过探测 Transformer 特定注意力头中编码的"信息充分性信号"，训练轻量分类器判断模型何时已获取足够上下文，实现提前终止处理，在6个QA数据集上平均提高3.4%准确率同时减少1.33×token消耗。

## 研究背景与动机

**领域现状**：LLM 在推理时对输入上下文进行无差别处理，每个 token 获得相同的计算优先级，无论其与任务的实际相关性如何。

**现有痛点**：现有上下文压缩方法（LLMLingua系列、RAG）依赖预设的固定压缩率或固定检索文档数；这种"一刀切"方式无法适应不同输入的信息密度差异，容易导致信息丢失或计算浪费。

**核心矛盾**：人类阅读时会根据已获取信息的充分性动态决定何时停止，而LLM缺乏这种自适应能力；同时"lost-in-the-middle"现象表明冗余上下文反而降低准确率。

**本文目标** 让LLM能自主评估上下文是否充分，动态决定截断位置，在不预设压缩率的情况下实现效率与性能双赢。

**切入角度**：通过分析模型内部激活发现，特定注意力头天然编码了信息充分性信号，可通过线性探针轻量检测。

**核心 idea**：模型自身的注意力头激活中内嵌了"够不够用"的判断信号，用探针读取这个信号就能实现动态早停。

## 方法详解

### 整体框架
输入：完整上下文 $\mathbf{C}$ + 查询 $q$。将上下文按从左到右的顺序划分为 $m$ 个不重叠的 chunk $\{\mathfrak{s}_j\}_{j=1}^m$，构建累积上下文序列 $\mathbf{C}_i = \mathfrak{s}_1 \| \mathfrak{s}_2 \| \dots \| \mathfrak{s}_i$。每处理一个新 chunk，用分类器判断当前累积上下文是否"足够"；若足够则截断，剩余 chunk 不再处理。KV 缓存可以跨 chunk 复用，避免重复计算。

### 关键设计

1. **充分性信号探针（Probing for Sufficiency Heads）**:

    - 功能：识别哪些注意力头编码了信息充分性信号
    - 核心思路：对每个注意力头 $(l, h)$ 的激活 $x_l^h \in \mathbb{R}^D$，训练线性分类器 $p_\theta(x_l^h) = \sigma(\langle \theta, x_l^h \rangle)$ 预测当前上下文是否已包含足够信息（二分类）。以验证集 F1 分数衡量各头的预测能力，选 top-$k$ 个头
    - 设计动机：实验发现中间层的少数注意力头 F1 显著高于其他头（如 LLaMA3.2-1B 中存在明显的高 F1 热点），说明模型内部表示自然包含充分性语义
    - 与之前方法的区别：不依赖外部信号或压缩启发式，而是从模型自身激活中提取已有信息

2. **集成分类器（Ensemble Sufficiency Classifier）**:

    - 功能：基于 top-$k$ 注意力头构建鲁棒的充分性判断器
    - 核心思路：在 top 头上训练多个轻量分类器 $\{\mathcal{S}_1, \dots, \mathcal{S}_e\}$，使用 StratifiedKFold (n=5) 交叉验证，按 AUC 分数加权集成：$\mathcal{S}_{\text{ensemble}}(\mathbf{C}_i) = \frac{1}{e}\sum_{j=1}^e \mathcal{S}_j(\mathbf{C}_i)$
    - 判断规则：当 $\mathcal{S}_c(\mathbf{C}_i) \geq \tau$ 时判定为充分，停止处理

3. **迭代推理与缓存复用**:

    - 功能：高效地逐步扩展上下文并判断充分性
    - 核心思路：每次仅计算新增 chunk 的激活，利用上一步缓存的激活 $\mathbf{A}_{\text{cache}}^{i-1}$：$\mathbf{A}(\mathbf{C}_i) = f_{\text{model}}(\mathbf{C}_i \setminus \mathbf{C}_{i-1}, \mathbf{A}_{\text{cache}}^{i-1})$
    - 设计动机：非重叠 chunk + KV缓存复用保证了计算效率；若用重叠 chunk 则需重复计算激活，丧失效率优势

4. **大模型自提示替代方案（Self-Prompting）**:

    - 功能：对14B+大模型，通过元提示让模型自行判断上下文是否充足
    - 核心发现：1B模型自提示 F1仅52.6，但70B模型达到83.1，说明充分性自评估是一种涌现能力

### 训练策略
- 探针训练数据：为每个累积上下文标注"充分/不充分"标签，基于gold信息span的最后一个token位置
- 数据集经过精心平衡：gold答案位置均匀分布（均值≈0.50，标准差0.25-0.28），确保50%/50%的正负样本比例
- 超参数：$k=5$（注意力头数），8个分类器取top-4组集成，10%增量 chunk 策略

## 实验关键数据

### 主实验

| 方法 | LLaMA-1B Avg | Mistral-8B Avg | Qwen-14B Avg | LLaMA-70B Avg | 总平均 |
|------|-------------|---------------|-------------|--------------|--------|
| Full Context | 14.2 | 37.2 | 44.0 | 56.1 | 37.9 |
| BM25 | 13.7 | 35.6 | 36.5 | 41.7 | 31.9 |
| LLMLingua2 | 14.4 | 35.8 | 45.0 | 55.4 | 37.7 |
| Self-Prompt | 8.9 | 30.0 | 45.1 | 59.1 | 35.8 |
| **Ours** | **13.9** | **37.3** | **46.3** | **59.5** | **39.2** |

本文方法在 1.33× token 减少率下，平均准确率达到 39.2%，超过全量上下文 (37.9%) 和最强静态压缩方法 LLMLingua2 (37.7%)。

### 充分性分类性能

| 模型 | Fine-Tune | Self-Prompt | Probing (Ours) |
|------|-----------|-------------|----------------|
| LLaMA3.2-1B | 79.5 | 52.6 | **88.3** |
| Mistral-8B | 69.7 | — | **89.8** |
| Qwen2.5-14B | — | 78.3 | **87.2** |
| LLaMA3.3-70B | — | 83.1 | **91.1** |

探针方法在所有模型规模上都大幅领先，且无需额外微调模型。

### 关键发现
- 探针方法 F1=91.1 远超微调分类器 (79.5) 和自提示 (83.1)，说明注意力头激活比模型表面输出更可靠
- 大模型（14B+）展现涌现式自评估能力，但小模型（1B-8B）必须依赖探针
- 截断上下文在某些场景反而提升准确率，可能缓解了"lost-in-the-middle"问题
- 阈值 $\tau$ 是最关键的超参数，控制效率与性能的 trade-off

## 亮点与洞察
- **从模型内部挖掘已有信号**：不引入外部模型或复杂压缩逻辑，仅用线性探针读取注意力头激活，思路极其简洁。这种"模型已知道但没说出来"的哲学值得借鉴
- **动态 vs 静态的范式转变**：传统压缩方法预设压缩率，本文让每条输入自适应决定处理量，更符合实际信息分布
- **涌现性发现**：充分性自评估能力随模型规模涌现，为理解大模型能力提供新视角
- **实用性强**：探针训练一次适用所有任务，KV缓存复用保证效率，方法易于工程部署

## 局限与展望
- 当前假设信息从左到右线性累积，不适用于信息散布在上下文多处的场景（如某些 multi-hop 推理）
- 充分性标签基于 gold span 位置定义，未考虑不同模型可能需要不同量的上下文
- 10% chunk 粒度是固定的，更自适应的 chunk 策略可能进一步提升效率
- 探针选择的注意力头是模型特定的，换模型需重新训练探针
- 仅在 QA 任务上验证，生成式任务（如摘要、翻译）的适用性待探索

## 相关工作与启发
- **vs LLMLingua系列**: LLMLingua 用小模型按 token entropy 过滤，预设固定压缩率；本文完全动态，由模型内部信号驱动，在等效 token 减少率下准确率更高
- **vs RAG**: RAG 预设检索 top-k 文档，与压缩方法正交；本文发现 RAG 在大模型上性能严重退化，而本文方法随模型规模持续提升
- **vs KV Cache Compression**: KV cache 优化在计算层面近似 Attention，与本文的上下文层面截断正交，两者可组合

## 评分
- 新颖性: ⭐⭐⭐⭐ 从注意力头激活中提取充分性信号的 insight 很有启发性，但线性探针本身是成熟工具
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、3个模型族、1B-70B规模全覆盖，消融分析全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述条理分明，图示直观
- 价值: ⭐⭐⭐⭐ 在推理效率领域提供新范式，工程实用性强，但适用场景有一定局限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](../../ACL2026/interpretability/forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)
- [\[NeurIPS 2025\] H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition](h-splid_hsic-based_saliency_preserving_latent_information_decomposition.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)

</div>

<!-- RELATED:END -->
