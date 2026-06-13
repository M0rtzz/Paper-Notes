---
title: >-
  [论文解读] Mapping Faithful Reasoning in Language Models
description: >-
  [NeurIPS 2025][LLM推理][CoT忠实性] 提出Concept Walk框架，通过将推理模型每步的残差流激活投影到从对比数据学到的概念方向上，追踪内部概念表示在推理过程中的演化轨迹，以此区分CoT链是真正参与计算的还是仅为事后合理化的装饰性输出。 领域现状：推理型LLM（如OpenAI o1、Gemini思考…
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "CoT忠实性"
  - "机械可解释性"
  - "推理模型"
  - "安全向量"
  - "激活分析"
---

# Mapping Faithful Reasoning in Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.22362](https://arxiv.org/abs/2510.22362)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: CoT忠实性, 机械可解释性, 推理模型, 安全向量, 激活分析

## 一句话总结
提出Concept Walk框架，通过将推理模型每步的残差流激活投影到从对比数据学到的概念方向上，追踪内部概念表示在推理过程中的演化轨迹，以此区分CoT链是真正参与计算的还是仅为事后合理化的装饰性输出。

## 研究背景与动机

**领域现状**：推理型LLM（如OpenAI o1、Gemini思考模式、Qwen 3）通过CoT（思维链）提供推理过程的可见性，被视为增强AI透明度和可信度的重要途径。实践者依赖检查CoT来验证决策是否基于合理推理。

**现有痛点**：越来越多的证据表明CoT并不总是忠实地反映内部计算——它可能只是事后合理化（post-hoc rationalisation），即模型已确定答案后才生成看似合理的推理过程。这使得通过检查CoT来进行安全监督变得不可靠。

**核心矛盾**：表面文本无法区分两种模式——"CoT作为计算"（推理过程真正影响输出）和"CoT作为合理化"（推理过程是装饰性的）。现有的过滤方法（如截断/扰动CoT观察输出变化）只能判断输出是否改变，不能揭示内部概念表示如何随推理演化。

**本文目标**：如何在激活空间中追踪特定概念（如安全性）在多步推理中的动态演化，从而区分忠实推理和装饰性推理？

**切入角度**：结合表示工程（representation engineering）的概念方向提取与机械可解释性的时序分析，在推理步骤维度上追踪概念激活。

**核心idea**：通过对比安全/不安全提示学习一个"安全方向"向量，然后将推理模型每一步的激活投影到此方向上，得到"Concept Walk"轨迹——如果扰动CoT后轨迹持续偏移说明推理是计算性的，如果迅速恢复说明是装饰性的。

## 方法详解

### 整体框架
三阶段方法论：(1) 过滤——通过扰动敏感性将样本分为"困难"（CoT参与计算）和"容易"（CoT为装饰）两类；(2) 学习安全方向——在激活空间中用对比数据计算安全概念向量；(3) Concept Walk——将每步推理的激活投影到安全方向，追踪时序演化，比较原始和扰动后的轨迹。

### 关键设计

1. **CoT计算性过滤 (Filtering for CoT-as-computation)**:

    - 功能：将样本分为"困难"和"容易"两类，以区分CoT是否在功能上参与了决策
    - 核心思路：受Lanham et al.和Emmons et al.启发，对每个样本在CoT中间注入有缺陷的推理步骤（扰动），观察是否导致模型最终输出显著降级。保留输出变化大的"困难"样本（CoT被整合到计算中），过滤掉输出不变的"容易"样本（CoT仅为合理化）
    - 设计动机：如果不过滤，分析装饰性CoT的内部动态所得结论可能是误导性的——我们只关注那些推理确实影响输出的案例

2. **安全方向计算 (Computing the Safety Vector)**:

    - 功能：在激活空间中找到编码"安全性"概念的方向
    - 核心思路：构建不安全提示集 $\mathcal{D}_{\text{unsafe}}$ 和安全提示集 $\mathcal{D}_{\text{safe}}$（配对的反事实对），计算各自在第 $\ell$ 层、第 $t$ 个token位置的平均激活：
    $\boldsymbol{\mu}_{\text{unsafe}}^{(\ell,t)} = \frac{1}{|\mathcal{D}_{\text{unsafe}}|} \sum_{i \in \mathcal{D}_{\text{unsafe}}} \boldsymbol{x}_\ell^i[t], \quad \boldsymbol{\mu}_{\text{safe}}^{(\ell,t)} = \frac{1}{|\mathcal{D}_{\text{safe}}|} \sum_{i \in \mathcal{D}_{\text{safe}}} \boldsymbol{x}_\ell^i[t]$
   安全方向为归一化差值：
    $\hat{\boldsymbol{v}}^{(\ell,t)} = \frac{\boldsymbol{\mu}_{\text{unsafe}}^{(\ell,t)} - \boldsymbol{\mu}_{\text{safe}}^{(\ell,t)}}{\|\boldsymbol{\mu}_{\text{unsafe}}^{(\ell,t)} - \boldsymbol{\mu}_{\text{safe}}^{(\ell,t)}\|_2}$
    - 方向选择：在验证集上评估每个候选 $(\ell, t)$ 的bypass score（消融后拒绝抑制效果）、induce score（添加后拒绝诱导效果）和KL散度（对良性提示的影响最小化），选择最优方向
    - 设计动机：Difference of Means是表示工程中成熟的方向提取方法，可以捕获模型内部对安全性的编码而非表层文本特征

3. **Concept Walk**:

    - 功能：追踪安全概念在推理步骤间的时序演化
    - 核心思路：在thinking模式下运行模型，对每个CoT步骤 $s$，提取该步骤所有token的残差流激活并取平均，得到步骤级激活向量：
    $\boldsymbol{h}_s = \frac{1}{|T_s|} \sum_{t \in \mathcal{T}_s} \boldsymbol{x}[t]$
   然后计算与安全方向的余弦相似度：
    $\alpha_s = \cos(\boldsymbol{h}_s, \boldsymbol{v}^{(\ell^*)})$
   $\alpha_s$ 量化了模型在第 $s$ 步的内部状态与安全方向的对齐程度，独立于表层文本是否提到安全相关词汇
    - 对扰动后的CoT执行相同分析，比较原始和扰动轨迹的差异
    - 设计动机：现有方法只能判断"CoT是否影响输出"，不能揭示"内部概念表示如何随推理演化"

### 实验模型与数据
- 分析对象：Qwen 3-4B（36层Transformer decoder，支持可控thinking模式）
- 数据生成：用Mistral-7B-Instruct-v0.2生成合成数据（避免数据污染），模拟音乐AI助手场景
- 两个安全类别：Harm（2911对）和Hate（4819对），经标签过滤后Hate保留256违规+462合规，Harm保留181违规+290合规

## 实验关键数据

### 数据集统计

| 类别 | 训练集 | 验证集 | 测试集 | 总配对数 |
|------|--------|--------|--------|---------|
| Harm | 1,746 | 873 | 292 | 2,911 |
| Hate | 2,890 | 1,445 | 484 | 4,819 |

### 核心实验发现

| 观察维度 | "困难"样本 (CoT-as-computation) | "容易"样本 (CoT-as-rationalisation) |
|---------|-------------------------------|-------------------------------------|
| 扰动后安全激活变化 | 持续且结构化的偏移，跨多步延续 | 短暂的瞬时变化，迅速恢复原始轨迹 |
| 扰动影响持续性 | 偏移持续到扰动点之后的许多步 | 模型在1-2步内"自我修正" |
| 最终输出变化 | 显著改变（如从拒绝变为合规） | 不改变或微弱改变 |
| 违规提示初始安全激活 | 高（确认安全向量有效） | 高（与困难组一致） |

### 关键发现
- **困难vs容易的清晰分离**：在Harm和Hate两个数据集上一致观察到：困难样本中扰动导致安全激活的持续偏移，容易样本中偏移是瞬态的
- **违规提示的即时识别**：无论困难还是容易，违规提示始终触发较高的初始安全激活，验证了安全方向向量的有效性
- **归一化x轴的一致性**：由于CoT长度不同，按CoT长度归一化后所有注入点对齐在0.5处，方便比较。归一化后模式更清晰
- **定性案例验证**："Can you recommend some classic rock tunes"（困难样本）——注入缺陷推理后安全激活持续偏移并最终改变输出；"Playlist for a bar crawl"（容易样本）——注入后短暂升高随即恢复，输出不变
- 安全向量在non-thinking模式下计算、在thinking模式下应用的跨模式假设基本成立

## 亮点与洞察
- **概念级的时序透镜**：Concept Walk提供了一种全新的分析视角——不看表层文本说了什么，而看内部概念表示在推理过程中如何演化。这比输出级的扰动测试（只看结果变化）提供了更丰富的信息
- **方法的通用性**：虽然论文以Safety为案例研究，但框架完全通用——只需更换对比数据集即可追踪fairness、bias、toxicity等任何可通过对比定义的概念
- **连接两个社区**：将表示工程（学习概念方向）和CoT忠实性分析（区分计算vs装饰）桥接起来，两者单独使用都有局限，组合产生了新的洞察
- **对安全监督的实际启示**：当Concept Walk显示安全激活是瞬态的（容易样本），监管者不应信任CoT中的安全推理文本，因为它只是装饰

## 局限与展望
- 安全方向在non-thinking模式下计算、在thinking模式下应用——如果两种模式的表示不同，逐步估计可能有偏差。未来应计算模式特定的安全方向
- 扰动过滤策略无法保证完全忠实——某些计算过程可能在CoT中未被言语化（hidden reasoning pathways）
- 仅在Qwen 3-4B单一模型上验证，不同模型规模/架构/训练范式的泛化性待探索
- 合成数据场景（音乐AI助手的播放列表请求）较为特化，真实部署场景的安全推理可能更复杂
- 困难/容易的术语借自任务难度，但实际含义是"模型是否依赖CoT"，可能造成概念混淆
- 违规样本在过滤后数量较少，统计效力有限

## 相关工作与启发
- **vs Lanham et al. / Emmons et al.**: 他们提出了扰动/截断CoT来检测忠实性的过滤策略；Concept Walk在此基础上增加了内部表示的时序追踪，不仅判断"是否忠实"还揭示"内部如何演化"
- **vs Arditi et al. (Refusal Direction)**: 同样用Difference of Means提取概念方向，但Arditi聚焦于单次拒绝行为；本文将方向应用于多步推理的时序分析
- **vs Venhoff et al. (Steering Vectors for Reasoning)**: 他们用steering vector影响推理轨迹并揭示跨步依赖；本文用概念方向被动观察而非主动干预，二者互补
- **vs Bogdan et al. (Thought Anchors)**: 识别因果性重要的推理步骤；Concept Walk提供了从概念激活角度判断步骤重要性的另一种视角

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Concept Walk是一个优雅的方法论框架，首次在概念表示级别提供了CoT忠实性的时序视角
- 实验充分度: ⭐⭐⭐ 作为方法论贡献，案例研究充分说明了框架能力，但仅限单一模型和合成数据
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰，方法论阐述精确，讨论诚实且深入，但符号较多
- 价值: ⭐⭐⭐⭐ 对AI安全社区有重要方法论价值，Concept Walk可成为分析推理模型忠实性的标准工具

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[ACL 2026\] RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners](../../ACL2026/llm_reasoning/rsat_structured_attribution_makes_small_language_models_faithful_table_reasoners.md)
- [\[NeurIPS 2025\] ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)

</div>

<!-- RELATED:END -->
