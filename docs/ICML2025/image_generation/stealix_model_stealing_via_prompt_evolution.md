---
title: >-
  [论文解读] Stealix: Model Stealing via Prompt Evolution
description: >-
  [ICML2025][图像生成][模型窃取] Stealix 提出首个无需人工设计 prompt 的模型窃取方法，通过遗传算法迭代进化 prompt，利用 Stable Diffusion 生成目标类别图像并查询受害模型，仅需每类 1 张真实图像即可在低查询预算下超越依赖类名或手工 prompt 的已有方法，准确率提升最高达 22.2%。
tags:
  - ICML2025
  - 图像生成
  - 模型窃取
  - 提示学习
  - 遗传算法
  - 扩散模型
  - 对比学习
---

# Stealix: Model Stealing via Prompt Evolution

**会议**: ICML2025  
**arXiv**: [2506.05867](https://arxiv.org/abs/2506.05867)  
**代码**: https://zhixiongzh.github.io/stealix/ (项目页面)  
**领域**: 扩散模型 / AI安全  
**关键词**: 模型窃取, Prompt优化, 遗传算法, 扩散模型, 对比学习

## 一句话总结

Stealix 提出首个无需人工设计 prompt 的模型窃取方法，通过遗传算法迭代进化 prompt，利用 Stable Diffusion 生成目标类别图像并查询受害模型，仅需每类 1 张真实图像即可在低查询预算下超越依赖类名或手工 prompt 的已有方法，准确率提升最高达 22.2%。

## 研究背景与动机

**领域现状**：模型窃取（Model Stealing）攻击允许攻击者通过查询黑盒模型 API 来复制其功能，构建行为相似的代理模型。当前方法可分三类：(1) 使用公开数据集查询（如 Knockoff Nets），(2) 从头训练 GAN 生成合成图像，(3) 利用预训练扩散模型（如 Stable Diffusion）通过 prompt 合成图像。第三类方法效率最高，因为无需训练新生成器，也不依赖在线数据。

**现有痛点**：基于扩散模型的方法（如 ASPKD）虽然高效，但严重依赖人工设计的 prompt 或已知的类名。当类名缺乏上下文、无法准确描述目标数据特征时，这些方法表现大幅下降。此外，人工干预阻碍了攻击的自动化和可扩展性。

**核心矛盾**：在专业领域（如卫星图像分类），高价值模型的攻击者往往缺乏设计精准 prompt 的知识。现有研究假设攻击者拥有类名或 prompt 设计能力，这过度简化了问题，低估了预训练生成模型在模型窃取中的真实威胁。Figure 1 展示了当查询数据集与目标分布不匹配时（如用 CIFAR-10 窃取卫星图像分类器），性能急剧下降。

**本文目标** 在攻击者「不知道类名」且「没有 prompt 设计能力」的更现实威胁模型下，如何自动化地进行高效模型窃取？具体子问题：
   - 如何在无先验知识下生成与目标分布匹配的图像？
   - 如何自动化地发现和优化描述目标类别的 prompt？
   - 如何在有限查询预算内最大化代理模型精度？

**切入角度**：作者观察到 prompt 优化本质上是一个搜索问题——需要在语义空间中找到能生成「被受害模型分类为目标类别」的高质量 prompt。遗传算法天然适合这种多目标、离散搜索场景。结合视觉-语言模型的对比学习，可以将受害模型的反馈融入 prompt 优化闭环。

**核心 idea**：用遗传算法进化 prompt 种群，以受害模型对合成图像的分类一致性（Prompt Consistency）作为适应度函数，迭代优化 prompt 的精准度和多样性。

## 方法详解

### 整体框架

Stealix 的输入是每类仅 1 张真实种子图像 + 黑盒受害模型的查询 API，输出是一个行为与受害模型相似的代理模型。整体 pipeline 分为两大阶段：

**阶段一：Prompt 进化（核心创新）**——通过迭代的"优化→评估→繁殖"三步循环，自动生成多个描述每个类别的高质量 prompt。

**阶段二：代理模型训练**——用优化后的 prompt 驱动 Stable Diffusion 合成大量图像，查询受害模型获取伪标签，训练代理模型。

每次迭代 $t$ 维护一个种群 $\mathcal{S}^t = \{(\mathbf{x}_c^s, \mathbf{x}_c^+, \mathbf{x}_c^-)_i^t\}_{i=1}^N$，包含 $N$ 个图像三元组，其中 $\mathbf{x}^s$ 是种子图像，$\mathbf{x}^+$ 是正样本（被受害模型分类为目标类），$\mathbf{x}^-$ 是负样本（被分类为其他类）。

### 关键设计

1. **Prompt Refinement（提示精炼）**:

    - 功能：对每个图像三元组，优化一个随机初始化的离散 prompt $\mathbf{p}$，使其在语义空间中捕获目标类别特征。
    - 核心思路：利用视觉-语言模型（如 CLIP）的文本编码器 $T$ 和图像编码器 $I$，通过对比学习损失优化 prompt。关键公式为：
    $\mathcal{L} = -\log \frac{\exp(\text{sim}(T(\mathbf{p}), I(\mathbf{x}^s)) / \tau) + \exp(\text{sim}(T(\mathbf{p}), I(\mathbf{x}^+)) / \tau)}{\sum_{\mathbf{x} \in \{\mathbf{x}^s, \mathbf{x}^+, \mathbf{x}^-\}} \exp(\text{sim}(T(\mathbf{p}), I(\mathbf{x})) / \tau)}$
      其中 $\text{sim}$ 是余弦相似度，$\tau$ 是温度系数。这使 prompt 的文本特征向种子图像和正样本靠近，同时远离负样本。
    - 设计动机：与 Textual Inversion 和 PEZ 等现有 prompt 优化方法不同，Stealix 将受害模型的分类反馈（正/负样本划分）融入优化目标，使 prompt 不仅在视觉上匹配目标，还在「分类语义」上对齐。如图 3 所示，通过负样本过滤掉无关特征（如为 "bottle" 类去除 "pool" 特征）。

2. **Prompt Consistency（提示一致性 / 适应度函数）**:

    - 功能：评估优化后 prompt 的质量，作为遗传算法的适应度函数（fitness function）。
    - 核心思路：用优化后的 prompt 驱动生成模型 $G$ 合成一批图像，然后用受害模型 $V$ 对全部图像分类。适应度定义为：
    $\text{PC}(\mathbf{p}, c) = \frac{1}{M} \sum_{j=1}^{M} \mathbb{1}[V(G(\mathbf{p})_j) = c]$
      即 $M$ 张合成图像中被受害模型分类为目标类 $c$ 的比例。PC 值越高，prompt 越精准。
    - 设计动机：直接使用受害模型的反馈来评估 prompt，形成闭环。论文通过统计分析证明 PC 与合成图像到真实数据的特征距离高度相关。同时，评估过程中产生的合成图像按分类结果更新正/负样本集，为下一轮优化积累更多信号。

3. **Prompt Reproduction（提示繁殖 / 遗传进化）**:

    - 功能：基于适应度结果，用遗传算法的选择、交叉、变异操作进化下一代图像三元组种群。
    - 核心思路：高 PC 分数的三元组更大概率被选入下一代；交叉操作混合不同三元组的种子/正/负样本以引入多样性；变异操作随机替换部分样本以避免早熟收敛。
    - 设计动机：遗传算法天然适合不可微、离散的搜索空间。prompt 需通过生成模型和受害模型两步黑盒操作才能评估，梯度不可回传。同时需要平衡精准度与多样性，遗传算法的种群机制天然实现了这一点。

4. **代理模型训练**:

    - 功能：收集所有迭代积累的合成图像及受害模型标签，训练最终的代理模型 $A$。
    - 核心思路：最小化 $\arg\min_{\theta_a} \mathbb{E}_{\mathbf{x} \sim G(\mathbf{p})}[\mathcal{L}_{CE}(V(\mathbf{x}), A(\mathbf{x}))]$。
    - 设计动机：经过 prompt 进化，合成数据的分布已与受害模型训练数据高度对齐，代理模型可以有效学习到受害模型的决策边界。

### 损失函数 / 训练策略

- **Prompt 优化阶段**：对比学习损失（Eq. 3），通过 CLIP 图文对齐空间优化离散 prompt
- **代理模型训练阶段**：标准交叉熵损失，受害模型的 top-1 预测作为伪标签
- 每类仅需 1 张种子图像，种群大小 $N$，每个 prompt 生成 $M$ 张图像用于评估
- 整个流程受每类查询预算 $B$ 约束

## 实验关键数据

### 主实验

论文在多个数据集上与现有模型窃取方法对比。Stealix 在不使用类名的条件下全面超越使用类名/手工 prompt 的基线方法：

| 方法 | 先验知识要求 | 查询方式 | 低预算优势 | 适用专业领域 |
|------|------------|---------|-----------|-------------|
| Knockoff Nets | 需要相似分布的公开数据集 | 直接查询公开图像 | 差（数据集不匹配时崩溃） | 差 |
| SD + 类名 prompt | 需要知道类名 | 用类名 prompt 生成 | 中等 | 差（类名不够描述性） |
| SD + 手工精细 prompt | 需要领域知识 + prompt 设计技能 | 精心设计的 prompt | 较好 | 差（需专业知识） |
| ASPKD | 需要类名 + 最近邻匹配 | 扩散模型 + 伪标签 | 较好 | 中等 |
| **Stealix** | **仅 1 张图/类，无需类名** | **自动 prompt 进化** | **最佳 (+22.2%)** | **最佳** |

### Prompt Consistency 有效性分析

| 实验设置 | PC 与特征距离相关性 | 代理模型精度趋势 | 说明 |
|---------|-------------------|-----------------|------|
| Full Stealix（完整方法） | 高度负相关 | 最高 | PC 越高 → 特征距离越近 → 精度越高 |
| w/o Prompt Refinement | — | 大幅下降 | 无对比学习优化，随机 prompt 质量低 |
| w/o Prompt Consistency | — | 明显下降 | 无适应度函数，遗传算法盲目搜索 |
| w/o Prompt Reproduction | — | 中等下降 | 无进化机制，prompt 多样性不足 |
| DA-Fusion baseline | 低 | 较低 | 不考虑受害模型反馈，prompt 与目标脱节 |

### 关键发现

- **Prompt Consistency 是可靠的代理指标**：统计分析表明 PC 指标与合成图像到真实数据的特征距离高度相关，无需访问真实数据即可评估 prompt 质量。
- **每类仅 1 张种子图像即可有效攻击**：这一极端低资源假设非常现实，但现有方法在此条件下表现远不如 Stealix。
- **低查询预算下优势最显著**：最高提升 22.2%，因为高效的 prompt 使每次查询都更有价值，避免了大量低质量查询的浪费。
- **专业领域优势突出**：在卫星图像等专业领域，手工设计有效 prompt 极其困难，Stealix 的自动化搜索优势被放大。
- **三个组件缺一不可**：消融实验表明 Prompt Refinement、Prompt Consistency、Prompt Reproduction 各自贡献显著，去掉任何一个都导致明显性能降低。

## 亮点与洞察

- **首个无需 prompt 先验的模型窃取框架**：这是一个重要的方法论突破。之前的方法假设攻击者知道类名或能设计好 prompt，实际上大大低估了威胁。Stealix 证明仅需 1 张图就能自动发现有效 prompt，使模型窃取门槛大幅降低。

- **遗传算法 + 对比学习的巧妙结合**：用遗传算法处理离散 prompt 空间的搜索问题，用 CLIP 的对比学习注入视觉-语义对齐信号，两者互补——对比学习提供局部优化方向，遗传算法提供全局探索能力。这种"可微优化 + 进化搜索"混合范式值得广泛借鉴。

- **Prompt Consistency 作为代理指标**：用受害模型自身的分类一致性作为适应度函数，既简单又有效。这一"用目标模型自身的反馈来攻击它"的思路在对抗性 ML 中具有普遍意义。

- **闭环正反馈设计**：正/负样本集在每轮迭代中不断扩充，前期探索结果持续为后续优化提供信号，形成不断自我改善的正循环。可迁移到任何需要与黑盒系统交互并逐步逼近目标的场景（如红队测试）。

## 局限与展望

- **依赖预训练模型覆盖范围**：Stealix 依赖 Stable Diffusion 的生成能力和 CLIP 的语义理解。如果目标领域远超这些模型的预训练分布（如罕见医学影像或工业缺陷检测），prompt 进化可能无法收敛到有效解。

- **查询效率仍有优化空间**：每次评估 PC 需生成 $M$ 张图并逐一查询受害模型，多代进化后累积查询量较大。可考虑引入 Bayesian Optimization 或训练轻量级 surrogate model 来减少评估开销。

- **仅验证图像分类场景**：论文仅在图像分类上实验，未扩展到目标检测、语义分割等更复杂的视觉任务，或 NLP/多模态等其他模态。

- **防御对抗讨论有限**：论文重点在攻击，对防御方（如检测异常查询模式、输出扰动、水印机制）缺乏深入分析。

- **种子图像敏感性未充分讨论**：不同种子图像的代表性是否影响最终性能？若种子为极端/非典型样本，进化可能陷入局部最优。

## 相关工作与启发

- **vs Knockoff Nets (Orekondy et al., 2019)**：用公开数据集查询受害模型，当数据集与目标分布不匹配时性能骤降。Stealix 通过自动 prompt 进化适配目标分布，摆脱了数据集选择的制约。

- **vs ASPKD (Hondru & Ionescu, 2023)**：使用扩散模型但依赖类名 prompt + 最近邻伪标签。Stealix 不需要类名，通过迭代优化生成更精准的查询图像，在同查询预算下显著超越。

- **vs DA-Fusion (Trabucco et al., 2024)**：数据增强方法，用 Textual Inversion 生成视觉相似图像。Stealix 将其扩展到模型窃取场景，关键改进是引入受害模型分类反馈指导 prompt 优化，而非仅使用原始类标签。

- **vs PEZ (Wen et al., 2024)**：用 CLIP 优化离散 prompt，但是"任务无关"的优化。Stealix 引入了受害模型反馈的"任务感知"对比学习目标，使优化方向与模型窃取目标对齐。

- **启发**：该方法的"遗传进化 prompt + 黑盒闭环反馈"范式可迁移到：(1) 自动化红队测试中的对抗 prompt 搜索，(2) 模型水印的鲁棒性评估，(3) 任何需要在离散空间进行黑盒优化的场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个无 prompt 先验的模型窃取方法，遗传算法搜索 prompt 的框架新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证、消融分析、PC 指标统计验证完整
- 写作质量: ⭐⭐⭐⭐ 威胁模型定义严谨，方法描述逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ 揭示了预训练生成模型在模型窃取中被低估的风险，对 AI 安全防御有实际警示

<!-- RELATED:START -->

## 相关论文

- [Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](../../ICCV2025/image_generation/exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs](pak-ucb_contextual_bandit_an_online_learning_approach_to_prompt-aware_selection_.md)
- [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](../../CVPR2025/image_generation/pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)

<!-- RELATED:END -->
