---
title: >-
  [论文解读] Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation
description: >-
  [CVPR 2026][LLM安全][神经元解释] 提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "神经元解释"
  - "概念验证"
  - "可解释AI"
  - "神经元功能分析"
  - "闭环验证"
---

# Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation

**会议**: CVPR 2026  
**arXiv**: [2603.24953](https://arxiv.org/abs/2603.24953)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 神经元解释, 概念验证, 可解释AI, 神经元功能分析, 闭环验证

## 一句话总结

提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。

## 研究背景与动机

1. **领域现状**：神经网络可解释性研究中，理解单个神经元的功能（即它编码什么概念）是一个核心问题。现有方法如 Network Dissection、CLIP-Dissect、FALCON、DnD 等通过自然语言描述神经元概念，取得了一定进展。

2. **现有痛点**：这些方法都基于一个共同假设——每个神经元都有明确定义的功能并为决策提供区分性特征。但研究表明，网络中存在冗余神经元，它们并不贡献于决策。为这些神经元生成描述会导致误解，让人错误理解网络的决策机制。

3. **核心矛盾**：现有方法本质上是"观察→假设"的过程，从探测数据集上的激活分布推断神经元功能，但由于数据覆盖有限，这些假设可能存在数据集偏差，无法准确反映神经元的真实功能。缺少验证环节。

4. **本文目标**：(1) 如何过滤掉不提供区分性特征的神经元；(2) 如何验证生成的概念是否真正匹配神经元功能。

5. **切入角度**：借鉴神经科学的"观察→假设→验证"科学方法论，认为深度网络的可解释性研究也应遵循同样的闭环逻辑。

6. **核心 idea**：通过激活分布筛选有效神经元、聚类生成概念假设、再用文生图生成验证图像来闭环验证概念-神经元匹配度。

## 方法详解

### 整体框架

SIEVE 要解决的是一个看似已被解决、实则缺了一环的问题：现有方法能给神经元贴上"它编码什么概念"的标签，却从不验证这个标签是否真的对。论文把神经科学里"观察→假设→验证"的科学方法论搬进来，让整个解释流程跑成一个闭环。输入是一个预训练分类网络和一个探测数据集，pipeline 分三步走：先 **Select**，按激活分布把"真有明确功能"的神经元和"响应弥散的冗余神经元"分开，只保留前者的高激活样本；再 **Hypothesize**，对这些样本聚类，让 CLIP/视觉语言模型为每个聚类配上概念词，得到若干条功能假设；最后 **Verify**，用 Stable Diffusion 把这些概念词生成成全新图像，回头看它们能否真的高激活目标神经元——激活得起来才算验证通过，否则这条假设被丢弃。关键创新在于 Verify 这步用的是"构造性"而非传统的"破坏性"干预。

### 关键设计

**1. 高激活样本筛选（Select）：把没有明确功能的冗余神经元先剔出去**

现有方法默认每个神经元都有清晰功能，但网络里其实充斥着对决策无贡献的冗余神经元，硬给它们生成描述只会误导对网络的理解。SIEVE 的判别依据很直接：一个真正编码了某概念的神经元，应该只对少数特定刺激强烈响应、对其余样本沉默，激活分布是长尾的；而冗余神经元的响应是弥散的。于是论文计算每个神经元在探测集上的激活分布，用**第 99 百分位数与中位数的比值**量化这种响应区分度，比值超过阈值 $\beta$（默认 10）才判定它有明确功能、纳入后续分析，并取其 top-20 高激活样本。直观例子是 Neuron 507 这类高区分度神经元在特定刺激上一致地高响应，而 Neuron 144 这类响应分散的就被过滤掉——这一步从源头上避免了为"无意义神经元"编造解释。

**2. 概念假设生成（Hypothesize）：一个神经元可能身兼多职，所以先聚类再配词**

单个神经元未必只对应一个概念，给它一条笼统的描述会丢失信息。SIEVE 先按激活图裁出每个样本里高激活区域的 patch，提取特征后做凝聚聚类，聚类数由 Silhouette 分数自动确定，从而把一个神经元的多种功能模式拆成若干组。对每个聚类 $C_{i,j}$，再用 CLIP 在预定义概念集 $\mathcal{T}$ 里挑出匹配度最高的 top-$K$（$K=2$）个概念词作为这个聚类的功能假设：

$$h_{i,j} = \arg\text{top-}K(\{g(t_q, C_{i,j}) \mid t_q \in \mathcal{T}\}, K)$$

其中 $g(\cdot)$ 是 CLIP 给出的概念-图像匹配分。这样得到的不是"神经元 = 某个词"，而是"神经元 = 几组功能假设"，比单一描述更贴近它的真实行为。

**3. 概念验证（Verify）：用文生图主动造刺激，看神经元买不买账**

前两步本质还是"从已有数据里推断"，假设是否成立从未被独立检验过，数据集偏差会让错误假设蒙混过关。SIEVE 补上的正是这一环：把假设概念词当作 prompt 喂给 Stable Diffusion，生成一批**独立于探测数据集**的验证图像，再把它们送回目标网络，统计有多少张能让对应神经元激活超过其 Top 1% 阈值 $T_i$，即激活率（Activation Rate）：

$$AR_i = \frac{1}{|\mathcal{D}_{gen}^{(i,j)}|} \sum_{x_{gen}} \mathbb{1}\{a_i^l(x_{gen}) > T_i\}$$

AR 低说明"按这个概念造出来的图根本激活不了这个神经元"，假设作废；AR 高的概念才作为最终解释保留。这一步和传统的神经元消融（破坏性干预，靠"删掉它看性能掉多少"反推功能）形成对照——SIEVE 是构造性干预，主动生成符合假设的刺激去验证，更像科学实验里设对照组做正向确认，结论也更直接可信。

### 一个完整示例

以 ViT-B/16 的 Neuron 37 为例走一遍：**Select** 阶段它的激活分布长尾、99 分位/中位数比值过 $\beta=10$，被判定有明确功能，取出 top-20 高激活样本；**Hypothesize** 阶段裁出这些样本的高激活 patch 做聚类，发现它们聚成一组毛发纹理相关的模式，CLIP 在概念集里匹配出"Short Dense Coat"等候选词作为假设；**Verify** 阶段用"Short Dense Coat"生成验证图像送回网络，Neuron 37 被大量激活、AR 很高，假设通过。最终这个神经元拿到的是"Short Dense Coat"这样的细粒度描述，而 CLIP-Dissect 等 baseline 因为没有验证环节、只能从原数据推断，只给出粗略的"Dog"。

### 损失函数 / 训练策略

本方法不涉及训练，是一个纯后分析框架。关键超参：激活区分度阈值 $\beta=10$，聚类数由 Silhouette 分数自动确定，每个聚类取 top-2 概念，验证阶段以 mean AR 作为过滤阈值。

## 实验关键数据

### 主实验

在 ImageNet 预训练的 ResNet-50 上，使用 Common Words (3k) 概念集：

| 方法 | CLIP cos | mpnet cos | mean AR (%) |
|------|----------|-----------|-------------|
| Network Dissect | 0.7073 | 0.3256 | 45.01 |
| CLIP-Dissect | 0.7868 | 0.4462 | 57.91 |
| WWW | 0.7713 | 0.4463 | 50.23 |
| DnD | 0.7595 | 0.4371 | 51.46 |
| **SIEVE (本文)** | **0.7914** | **0.4547** | **86.29** |

ViT-B/16 上类似趋势：SIEVE 的 mean AR 达 85.24%，远超 CLIP-Dissect 的 57.70%。

### 消融实验

| 配置 | CLIP cos | mpnet cos | mean AR (%) |
|------|----------|-----------|-------------|
| Baseline (无任何模块) | 0.6738 | 0.2306 | 45.57 |
| + Select + Cluster | 0.7624 | 0.4301 | 77.90 |
| + Select + Verify | 0.7821 | 0.4423 | 81.52 |
| + Select + Cluster (无Verify) | 0.7656 | 0.4189 | 72.87 |
| Full model | 0.7914 | 0.4547 | 86.29 |

### 关键发现

- **Verify 模块贡献最大**：去掉 Verify 后 mean AR 从 86.29% 降至 72.87%，证明验证环节对确保概念-神经元匹配的关键性
- **阈值 β 鲁棒**：β 在 4-12 范围内变化对最终指标影响极小（mean AR 波动 <1%）
- **域迁移场景下验证仍有效**：在遥感数据（EuroSAT）上存在域偏移时，SIEVE 仍达 75.45% mean AR，而 CLIP-Dissect 仅 43.16%
- **SIEVE 能提供更细粒度的描述**：如 ViT-B/16 的 Neuron 37 被描述为"Short Dense Coat"，而 baseline 只给出粗略的"Dog"

## 亮点与洞察

- **科学方法论的引入**：将神经科学的"观察→假设→验证"范式引入 DNN 可解释性，是一个非常优雅的跨领域类比。验证这一步补齐了现有方法的关键短板。
- **构造性验证**：不同于传统的消融实验（破坏性），通过文生图主动构造符合假设的刺激，这种正向验证更直接、更有说服力。
- **冗余神经元过滤**：第一个显式处理"不是所有神经元都有意义"这个问题的工作，避免了对冗余神经元的误解释。

## 局限与展望

- **文生图模型的域偏移**：验证阶段依赖 Stable Diffusion 生成图像，当目标网络训练在特殊领域（如遥感）时，生成图像与真实数据差异大，可能影响验证准确度
- **概念集的限制**：仍依赖预定义概念集（如 Broden、Common Words），无法发现概念集之外的新概念
- **计算开销**：需要为每个神经元的每个概念生成多张验证图像，规模化到整个网络时计算量显著
- **仅关注倒数第二层**：未扩展到浅层神经元的解释

## 相关工作与启发

- **vs CLIP-Dissect**: CLIP-Dissect 直接用 CLIP 匹配概念与激活样本，缺少验证环节；SIEVE 在匹配之后增加闭环验证，mean AR 提升约 30%
- **vs FALCON/WWW**: 这些方法改进了概念描述质量但仍假设所有神经元都有意义，SIEVE 通过 Select 阶段过滤冗余神经元
- **vs DnD**: DnD 用 LLM 生成更高质量的自然语言描述，但同样缺少验证，mean AR 仅 51.46%

## 评分

- 新颖性: ⭐⭐⭐⭐ 将科学方法论的闭环验证引入神经元解释领域，思路清晰且有说服力
- 实验充分度: ⭐⭐⭐⭐ 覆盖了 ResNet-18/50、ViT-B/16 多个模型和多个数据集，消融全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，科学方法论的类比引入自然
- 价值: ⭐⭐⭐⭐ 提出的 mean AR 指标可作为通用评估标准，验证范式对后续可解释性工作有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CRISP: Persistent Concept Unlearning via Sparse Autoencoders](../../ACL2026/llm_safety/crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)
- [\[ACL 2026\] APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation](../../ACL2026/llm_safety/appsi-139_a_parallel_corpus_of_english_application_privacy_policy_summarization_.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[ICCV 2025\] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](../../ICCV2025/llm_safety/sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](../../ACL2025/llm_safety/manu_modality_aware_unlearning.md)

</div>

<!-- RELATED:END -->
