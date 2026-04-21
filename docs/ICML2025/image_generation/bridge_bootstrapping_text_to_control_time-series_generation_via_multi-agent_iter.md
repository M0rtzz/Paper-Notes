---
title: >-
  [论文解读] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling
description: >-
  [ICML 2025][图像生成][文本控制时序生成] 提出 Bridge 框架，通过 LLM 多智能体系统生成高质量文本-时序配对数据，并利用语义原型与文本描述的混合提示驱动扩散模型，实现跨域、实例级别的文本控制时序生成（Text-Controlled TSG），在12个数据集中11个取得SOTA。
tags:
  - ICML 2025
  - 图像生成
  - 文本控制时序生成
  - 多智能体
  - 扩散模型
  - 语义原型
  - 跨域泛化
---

# BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling

**会议**: ICML 2025  
**arXiv**: [2503.02445](https://arxiv.org/abs/2503.02445)  
**代码**: [Microsoft/TimeCraft](https://github.com/Microsoft/TimeCraft)  
**领域**: 时序生成  
**关键词**: 文本控制时序生成, 多智能体, 扩散模型, 语义原型, 跨域泛化

## 一句话总结

提出 Bridge 框架，通过 LLM 多智能体系统生成高质量文本-时序配对数据，并利用语义原型与文本描述的混合提示驱动扩散模型，实现跨域、实例级别的文本控制时序生成（Text-Controlled TSG），在12个数据集中11个取得SOTA。

## 研究背景与动机

时序生成（TSG）在金融模拟、医疗数据增强、电力压力测试等领域有广泛应用。现有方法大多聚焦于**无条件单域生成**，但实际应用需要满足特定约束：如生成符合某患者画像的ECG、或特定疾病条件的心电图模式。

现有跨域方法的局限性：
- **基于域标签的条件生成**（如 TimeVQVAE）：依赖训练时的显式域标签，无法处理未见域，标签数量大时效率低下
- **基于自然语言的方法**（如 GenG）：仅提供域级别描述，缺乏细粒度实例级控制

本文提出用**文本**作为控制信号来引导TSG，但面临两大核心挑战：

**高质量文本-时序配对数据稀缺**：现有文本大多只有高层域描述，缺少趋势、波动等实例级信息。简单的规则方法（如"上升"、"下降"）无法带来显著提升

**文本与时序的模态鸿沟**：文本是离散token，时序是连续信号，粒度差异导致文本过于粗糙，难以精确捕获域特征

## 方法详解

### 整体框架

Bridge 包含两个紧密耦合的阶段：

**阶段一：文本-时序数据准备**（多智能体文本生成与迭代优化）  
**阶段二：文本控制时序生成**（基于扩散模型的混合提示生成）

### 关键设计

#### 1. 多智能体文本数据准备系统

针对数据稀缺问题，设计了三步迭代流程：

**Step 1 - 文本模板收集**：
- 采用 ReAct 风格的单智能体框架，通过动态推理与外部环境（Google、Wikipedia）交互
- 将查询分解为子问题，迭代回答后由另一LLM提取通用时序模板
- 最终获得50个通用模板，通过提示技术+人工验证排除数据集特定细节
- 构建数据集时，LLM将模板填充为领域/实例特定的文本描述

**Step 2 - 自动评估**：
- 采用零样本时序预测作为代理评估任务（避免每轮迭代重训模型的高成本）
- 使用 LSTPrompt 和 LLMTime 作为评估骨干
- 核心假设：更高质量的文本能带来更好的预测性能

**Step 3 - 反馈驱动迭代优化**：
- 设计多智能体协作系统，模拟人类提示工程团队的迭代过程
- **Stage 1 - 任务规划**：Manager Agent 编排工作流，分配任务给独立团队
- **Stage 2 - 组内讨论**：两个独立团队各含 Planner、Scientist、Engineer、Observer 四个角色，通过内部循环迭代优化文本
- **Stage 3 - 组间讨论**：两组 Leader 在 Manager 主持下进行结构化对话，比较整合结果直至达成共识
- **Stage 4 - 后处理**：提取最终模板，去重并移除数据集特定信息，形成固定的通用模板库

**数据合成关键点**：
- 模板仅用2个数据集构建，但成功应用于12个完全不相交的数据集
- 独立LLM负责提取统计信息并填充模板，完全离线，不依赖外部网络
- 填充后数据在TSG阶段保持不变

#### 2. 语义原型匹配（Domain-specific Prototype Matching）

为弥补文本粗粒度描述的不足，引入语义原型作为互补的域表示：

- 定义原型集合 $\mathcal{P} \in \mathbb{R}^{N_p \times d}$，每个原型向量 $p \in \mathbb{R}^{1 \times d}$ 编码时序的基本特征（趋势、季节性等）
- 原型作为跨域共享"字典"，不同域通过不同的原型选择和权重组合来表征
- 提出 **Prototype Assignment Module** 提取域特定权重 $m$
- 推理时使用目标域样本提取原型并计算权重

**设计直觉**：文本提供显式域信息（高层语义），原型提供隐式域特征（细粒度模式），两者互补。

#### 3. 混合提示扩散生成

将语义原型（$\mathcal{P}$, $m$）与文本嵌入 $l$ 融合构建混合提示，通过交叉注意力层注入扩散模型：

- 文本 → 提供趋势、统计信息和域知识等显式语义
- 原型+权重 → 补充域级共享模式，增强跨域泛化
- 混合提示作为扩散模型的条件输入

### 损失函数 / 训练策略

采用标准的 $\epsilon$-参数化去噪目标：

$$L = \mathbb{E}_{x_0 \in D^T, \epsilon \sim \mathcal{N}(0, I), n} \left[ \| \epsilon - \epsilon_{\theta, P}(x_n, n, m, l) \|^2 \right]$$

其中 $n$ 为去噪步数，$m$ 为原型权重，$l$ 为文本描述。遵循 channel-independent 设定，以单变量方式处理异构时序。

## 实验关键数据

### 主实验 - 生成保真度（MDD指标，越低越好）

| 数据集 | Bridge | Bridge w/o Text | Bridge w/o Proto | TimeVQVAE | TimeGAN | 提升幅度 |
|--------|--------|----------------|-----------------|-----------|---------|---------|
| Electricity | **0.220** | 0.202 | 0.277 | 1.763 | 2.443 | 87.5% vs TimeVQVAE |
| Wind | **0.316** | 0.319 | 0.362 | 0.777 | 1.115 | 59.3% vs TimeVQVAE |
| Traffic | **0.254** | 0.261 | 0.316 | 1.170 | 1.733 | 78.3% vs TimeVQVAE |
| Temperature | **0.342** | 0.345 | 0.408 | 0.943 | 1.164 | 63.7% vs TimeVQVAE |
| NN5 | **0.591** | 0.628 | 0.748 | 1.424 | 2.758 | 58.5% vs TimeVQVAE |
| Fred-MD | **0.258** | 0.271 | 0.359 | 2.932 | 4.028 | 91.2% vs TimeVQVAE |

在12个数据集中11个取得最优MDD，KL散度同样全面领先。

### 消融实验

| 配置 | 关键影响 | 说明 |
|------|---------|------|
| 移除文本（w/o Text） | MDD普遍上升，可控性大幅下降 | 人工评估HE分数下降3+分，文本是语义对齐的关键 |
| 移除原型（w/o Prototype） | MDD显著上升（如Taxi从0.386→0.491） | 原型对域级精细对齐贡献明显，但不如文本关键 |
| 原型数量 | 16个为最佳平衡点 | 超过16个仅有边际提升 |
| 多智能体 vs 单智能体 | 协作策略MAE一致更低 | 多团队比单团队Macro低1.5-6 MAE |
| 精炼文本 vs 初始文本 | MAE降低至少15% | 如AirPassenger: 49.36→40.94 |
| 精炼文本 vs 规则文本 | 效果差距更大 | 如AirPassenger: 52.41→40.94 |

### 关键发现

1. **文本简洁性优于详尽性**：过于详细的实例描述反而误导模型，简洁高层描述效果更好
2. **背景知识显著提升性能**：LLM预训练知识提供额外上下文支撑（w/o Background MAE上升3-8点）
3. **直接模式描述优于细粒度趋势分解**：STL分解+详细trend描述不如直接给出"总体上升/下降+top-k极值点"
4. **明确指定序列长度和统计值可稳定性能**
5. **少样本跨域泛化有效**：5-shot和10-shot设定下均超越所有基线，且10-shot比5-shot稳步提升

## 亮点与洞察

- **创新的问题定义**：首次系统化地定义并解决"文本控制时序生成"（TC-TSG）任务，将图像/视频领域的文本控制范式迁移到时序领域
- **多智能体自动化数据构建**：避免了人工标注文本-时序配对数据的高成本，且模板在仅2个数据集上构建就能泛化到12个新数据集
- **混合提示设计精巧**：文本提供高层语义控制，语义原型补充细粒度域模式，二者分工互补、缺一不可
- **评估体系完整**：同时使用保真度（MDD/KL）、可控性（J-FTSD）和人工评估（HE-Rank/HE-Mixed），多维度验证
- **实用洞察**：文本描述的简洁性、背景知识的重要性等发现对后续文本-时序研究有指导意义

## 局限与展望

1. **计算成本高**：多智能体系统需要多轮LLM调用（模板收集+评估+迭代优化），虽然是一次性的但初始成本不低
2. **仅处理单变量时序**：采用channel-independent设定，未直接建模多变量间的相关性
3. **原型数量需手动选择**：虽然实验表明16个为佳，但缺乏自适应选择机制
4. **文本编码器选择的影响有限**：实验表明更大LLM仅带来边际提升，可能文本信息的利用尚有空间
5. **可控性评估仍有主观性**：人工评估存在标注者偏差，J-FTSD也可能无法完全捕捉语义对齐

## 相关工作与启发

- **TimeDP**（Huang et al., 2025）：Bridge的原型设计直接受其启发，TimeDP用原型构建soft prompt但缺少文本控制
- **GenG**（Zhou et al., 2024）：首个文本到时序生成工作，但局限于特定域且缺少实例级控制
- **LSTPrompt / LLMTime**：作为评估骨干的零样本预测方法，证明了LLM在时序任务中的潜力
- **ReAct**（Yao et al., 2023）：启发了模板收集阶段的推理-行动交互框架
- 对多模态生成研究的启发：文本控制范式可扩展到更多时序应用场景（医疗个性化、金融模拟等）

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统定义TC-TSG任务，多智能体数据构建+混合提示设计新颖 |
| 技术深度 | ⭐⭐⭐⭐ | 框架完整（数据准备+生成两阶段），多智能体设计复杂但合理 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 12+2数据集，多维指标，消融全面，分析深入 |
| 实用价值 | ⭐⭐⭐⭐ | 跨域泛化+少样本能力强，代码开源，有实际应用潜力 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，但多智能体部分细节分散在大量附录中 |
| 总评 | ⭐⭐⭐⭐ | 扎实的工作，问题定义清晰，方法设计合理，实验充分 |

<!-- RELATED:START -->

## 相关论文

- [IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation.md)
- [Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](../../CVPR2025/image_generation/unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)
- [LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation](lscd_lomb-scargle_conditioned_diffusion_for_time_series_imputation.md)
- [A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2025/image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)

<!-- RELATED:END -->
