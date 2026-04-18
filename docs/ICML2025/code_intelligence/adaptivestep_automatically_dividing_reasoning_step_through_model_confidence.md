---
title: >-
  [论文解读] AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence
description: >-
  [ICML 2025][过程奖励模型] 提出基于模型预测置信度自动划分推理步骤的方法 AdaptiveStep，用于训练更精确的 Process Reward Model（ASPRM），在数学推理和代码生成任务上以不到 70% 的数据构建成本超越现有开源 PRM，并能通过 Token 级引导解码进一步提升推理性能。
tags:
  - ICML 2025
  - 过程奖励模型
  - 推理步骤划分
  - 模型置信度
  - Token-level Value-guided Decoding
  - 数学推理
---

# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

**会议**: ICML 2025  
**arXiv**: [2502.13943](https://arxiv.org/abs/2502.13943)  
**代码**: https://github.com/Lux0926/ASPRM (有)  
**领域**: LLM Reasoning / Process Reward Model  
**关键词**: 过程奖励模型, 推理步骤划分, 模型置信度, Token-level Value-guided Decoding, 数学推理

## 一句话总结
提出 AdaptiveStep，基于模型预测下一个 token 的置信度自动划分推理步骤，替代传统基于规则（如换行符）的粗粒度划分方式，训练出的 PRM (ASPRM) 在数学推理和代码生成任务上达到 SOTA 的 Best-of-N 性能，且数据构建成本降低超 30%。

## 研究背景与动机
Process Reward Model (PRM) 通过对推理过程的每一步提供细粒度奖励，已被证明优于只提供最终结果奖励的 ORM。然而，现有 PRM 的推理步骤划分几乎完全依赖基于规则的方法——例如用换行符 `\n` 分割或按固定 token 数量切分。这种做法存在两个核心问题：（1）换行符等位置通常模型置信度极高，并不是真正的决策点，导致步骤末尾信息量很低；（2）在代码生成等领域，换行符无法有效定义推理步骤的边界。手动标注虽然信息量高但成本极大、难以规模化。

核心矛盾在于：如何在不依赖人工标注的情况下，找到推理过程中真正关键的"决策点"？

本文的切入角度来自认知科学（Kahneman 2011）：人类的推理认知成本集中在少数关键决策点上，而非均匀分布。对应到 LLM，模型预测下一个 token 时的置信度（概率）天然反映了该位置的决策难度。因此，**在模型低置信度的位置划分步骤**，就能自动找到真正需要过程监督的关键位置。

## 方法详解

### 整体框架
AdaptiveStep 的 pipeline 分为三个阶段：（1）采集模型置信度分布并确定阈值；（2）基于阈值划分步骤并通过 rollout 标注奖励；（3）训练 PRM 并可选地用于 Token-level Value-guided Decoding (TVD)。

### 关键设计

1. **基于置信度的步骤划分 (AdaptiveStep)**:

    - 功能：将模型生成的推理响应自动划分为多个语义上有意义的步骤
    - 核心思路：对给定问题 $q$，用语言模型 $\pi$ 采样 $N$ 个响应。对每个响应的第 $i$ 个 token，计算模型置信度：
    $c_{s_i^n} = p(s_i^n | \pi, q, s_{<i}^n)$
      收集所有样本的 token 置信度分布，选取一个百分位阈值 $\tau$（默认取 2%，即只有 2% 的 token 低于此阈值）。置信度低于 $\tau$ 的 token 位置成为步骤的切分点，将响应划分为 $K$ 个步骤 $s^n = \{r_1, r_2, ..., r_K\}$
    - 设计动机：2% 的比例来源于认知科学研究——人类深度思考约占总思考时间的 2%。低置信度位置正是模型"犹豫不决"的地方，这些位置最具决策信息量。统计分析表明，换行符处的置信度通常极高（不是决策点），而数学表达式内部、名词选择、连词位置等才是真正的低置信度区域

2. **基于 Rollout 的奖励标注**:

    - 功能：为每个划分出的步骤估计目标奖励值
    - 核心思路：从每个步骤 $r_k$ 的末尾开始，用模型 $\pi$ 进行 $J$ 次 rollout 续写，使用 Hard Estimation (HE) 标注奖励：
    $r_k^e = \begin{cases} 1, & \exists j \in [J], \{r_1,...,r_k,t_j\} \text{ is correct} \\ 0, & \text{otherwise} \end{cases}$
      即只要存在任一续写能到达正确答案，该步骤奖励为 1
    - 设计动机：HE 方法简单高效，避免了软标注的噪声问题。每个数据点采样 30 次、每步 8 次 rollout，最终生成 388k 条数学 PRM 训练样本和 49k 条代码 PRM 训练样本

3. **Token-level Value-guided Decoding (TVD)**:

    - 功能：将训练好的 PRM 直接用于推理阶段的 token 级别引导解码
    - 核心思路：在模型解码时，当某个位置的 Top-1 token 置信度 $c_p < \tau$ 时，触发 PRM 对概率最高的 $M$ 个候选 token 进行评分，选择 PRM 评分最高的 token：
    $s_i = \arg\max_{s_i^m \in \mathbf{s_i^*}} R^\theta(p, s_{<i}, s_i^m)$
    - 设计动机：传统 PRM 只能在 BoN 场景中使用（生成多个完整答案后选最好的），TVD 则将 PRM 嵌入解码过程，在关键决策点提供实时引导，不需要额外生成多个完整候选

### 损失函数 / 训练策略
PRM 使用标准二分类交叉熵损失训练：
$$\mathcal{L}_{PRM}^\theta = -\sum_{k=1}^{K} (r_k^e \log r_k^\theta + (1-r_k^e) \log(1 - r_k^\theta))$$

其中 $r_k^\theta = R^\theta(p, r_1, \cdots, r_k)$ 是 PRM 对第 $k$ 步的预测奖励。训练基于 Llama-3.1-8B（ASPRM-L）和 Mistral-V0.1（ASPRM-M）作为 base model，代码 PRM 基于 DeepSeek-Coder-Base（ASPRM-D）。

## 实验关键数据

### 主实验
| 数据集 | 指标 | ASPRM | 之前SOTA | 提升 |
|--------|------|-------|----------|------|
| GSM8k (BoN) | Accuracy | ASPRM-L 最优 | Math-Shepherd, ER-PRM | 成本<70%下超越 |
| MATH500 (TVD) | Accuracy | 28.60% (ASPRM-L) | 27.80% (ER-PRM) | +0.8% |
| GSM8k (TVD) | Accuracy | 79.53% | 77.10% (greedy) | +3.15% |
| MATH500 (TVD) | Accuracy | 42.00% (ASPRM-L) | 38.80% (greedy) | +14.4% (相对) |
| LeetCodeDataset (TVD) | Pass@1 | 28.00% | 26.28% (greedy) | +6.54% |
| LiveCodeBench (TVD) | Pass@1 | 19.92% | 19.21% (greedy) | +3.70% |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 阈值 0.5% | BoN 较低 | 划分点太少，信息量不足 |
| 阈值 1.0% | BoN 中等 | 性能随阈值增加而提升 |
| 阈值 1.5% | BoN 接近 2% | 更强的模型对阈值更不敏感 |
| 阈值 2.0% (默认) | BoN 最优 | 平衡标注成本和性能 |
| 跨域 (Math→Code) | BoN 34.29% | ASPRM-L 在代码任务上也有效 |
| 混合训练 (Math+Code) | MATH500 BoN 35.40↑ | 混合域数据能增强 PRM 性能 |

### 关键发现
- **置信度划分的位置统计**：数学表达式仅占 3.85% 的 token 但贡献了 21.03% 的决策点；换行符仅占 2.7% 的决策点，证实了规则式划分的无效性
- **代码域特征**：80% 的决策点出现在代码注释中（规划"接下来做什么"），仅 20% 在实际代码中，说明模型在规划阶段更需要监督
- **模型可迁移性**：用强模型（MetaMath-Llama）生成的数据训练弱模型（Mistral），性能略有下降但仍可用，说明 rollout 数据具有合理但有限的跨模型迁移能力
- **评分位置泛化**：ASPRM 在随机评分位置和置信度评分位置上性能差异极小（90.45 vs 90.22），远优于基线方法受限于特定位置（如换行符）的评分

## 亮点与洞察
- 将认知科学的"2% 深度思考"洞察转化为具体的工程设计（阈值选择），简洁而有效
- 一个方法同时适用于数学推理和代码生成两个差异很大的领域，证明了基于置信度划分的通用性
- TVD 策略巧妙——仅在低置信度位置触发 PRM 评估，几乎不增加推理开销
- 数据构建成本仅为 Math-Shepherd 等方法的 70%，且只需单个模型采样（非多模型）
- 收集并开源了 LeetCodeDataset（1940 道题 + 测试用例 + sandbox），填补了代码 PRM 评估的空白

## 局限与展望
- 2% 阈值虽源于认知科学，但对不同能力的模型可能并非最优，更强的模型可能需要更少的划分点
- 仅用单一模型生成 rollout 数据，限制了跨模型迁移性（vs Math-Shepherd 用多模型）
- ASPRM-M 在 MATH500 上表现一般，可能因为训练数据未覆盖 MATH 测试集且仅用单模型构建
- TVD 目前只考虑 Top-M 候选 token，未探索更复杂的搜索策略（如 beam search 结合 PRM）
- 跨域泛化（代码→数学）在简单任务（GSM8k）上效果差，可能与 prompt 长度不匹配有关

## 相关工作与启发
- **vs Math-Shepherd**: Math-Shepherd 用换行符划分步骤、多模型采样，ASPRM 用置信度划分、单模型采样，成本更低但性能更好
- **vs ER-PRM**: ER-PRM 同样是句子级 PRM，16 次 rollout 成本更高。ASPRM 在评分位置泛化上显著优于 ER-PRM
- **vs OmegaPRM**: OmegaPRM 使用 MCTS 收集过程监督数据，方法更复杂，ASPRM 的置信度方法更简洁高效
- **vs Token-level PRM (Lee et al.)**: 将每个 token 都作为评分点，标注成本极高。ASPRM 只在 2% 的关键位置评分，大幅降低成本

## 评分
- 新颖性: ⭐⭐⭐⭐ 用模型置信度划分步骤的 idea 简洁而自然，但核心是工程设计而非全新范式
- 实验充分度: ⭐⭐⭐⭐⭐ 数学+代码两个领域、BoN+TVD 两种评估方式、迁移/泛化/特征分析非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，统计分析有说服力，但部分记号较多
- 价值: ⭐⭐⭐⭐⭐ 实用性很强，直接可落地的 PRM 训练方法，成本低效果好，代码和数据均开源
# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

**会议**: ICML 2025  
**arXiv**: [2502.13943](https://arxiv.org/abs/2502.13943)  
**代码**: https://github.com/Lux0926/ASPRM (有)  
**领域**: LLM Reasoning / Process Reward Model  
**关键词**: 过程奖励模型, 推理步骤划分, 模型置信度, Token-level Value-guided Decoding, 数学推理

## 一句话总结
提出基于模型预测置信度自动划分推理步骤的方法 AdaptiveStep，用于训练更精确的 Process Reward Model（ASPRM），在数学推理和代码生成任务上以不到 70% 的数据构建成本超越现有开源 PRM，并能通过 Token 级引导解码进一步提升推理性能。

## 研究背景与动机
Process Reward Model（PRM）通过对推理过程中的每个步骤给予奖励信号，能比 Outcome Reward Model（ORM）提供更细粒度的反馈，从而引导 LLM 生成更高质量的推理响应。然而，现有 PRM 面临一个核心问题：**推理步骤的划分方式过于粗糙**。

当前主流做法是基于规则进行步骤划分，例如用换行符或固定 token 数来切分。但这种方式存在两个关键缺陷：（1）换行符处模型置信度往往很高，即该位置并非真正的"决策点"，信息量低；（2）在代码生成等领域，难以定义通用的切分规则。手动标注虽然能产生高质量的步骤划分，但成本高昂且高度依赖专家知识。

作者从认知科学获得启发——Kahneman 指出人类深度思考仅占总思考量的约 2%，关键的推理决策集中在少数节点。受此启发，作者提出**让模型自己告诉我们哪里是关键决策点**：当模型对下一个 token 的预测置信度低时，说明该位置是一个需要做出重要选择的决策点，应该作为步骤的分界线。

## 方法详解

### 整体框架
AdaptiveStep 的整体流程分三步：（1）采样生成响应并收集每个 token 的置信度分布；（2）根据置信度阈值划分推理步骤，并通过 rollout 标注每步的奖励；（3）使用标注数据训练 PRM，并可选地将 PRM 用于 Token-level Value-guided Decoding（TVD）进行推理增强。

### 关键设计

1. **基于置信度的步骤划分（AdaptiveStep）**:
    - 功能：将推理响应自动分割成多个具有高信息量的推理步骤
    - 核心思路：对于生成的响应 $s^n$ 中的第 $i$ 个 token，其置信度定义为 $c_{s_i^n} = p(s_i^n | \pi, q, s_{<i}^n)$，即模型预测该 token 的概率。收集所有样本的置信度分布后，设定一个阈值 $\tau$（基于 token 数目的一定百分比，论文使用 2%），低于阈值的 token 位置即为步骤分界点。这样响应 $s^n$ 被划分为 $K$ 个推理步骤 $\{r_1, r_2, ..., r_K\}$
    - 设计动机：低置信度位置代表模型面临困难的决策点——可能是数学表达式中的计算、语义词汇的选择、或最终答案的确定。统计分析表明，3.85% 的数学表达式 token 贡献了 21.03% 的决策 token，仅 2.7% 的决策 token 出现在换行符处，证实了规则划分的低效

2. **基于 Rollout 的步骤奖励估计**:
    - 功能：为每个划分出的推理步骤估计目标奖励值
    - 核心思路：从每个步骤 $r_k$ 开始进行 $J$ 次 rollout 续写，使用 Hard Estimation (HE) 判断是否存在任何一条续写路径能到达正确答案。目标奖励为：
     $$r_k^e = \begin{cases} 1, & \exists j \in [J], \{r_1,...,r_k,t_j\} \text{ is correct} \\ 0, & \text{otherwise} \end{cases}$$
    - 设计动机：通过在决策点做 rollout，每一步的奖励信号更加精准，因为步骤末尾恰好是决策发生的位置

3. **Token-level Value-guided Decoding (TVD)**:
    - 功能：在推理阶段利用 PRM 实时指导 token 选择，无需额外采样
    - 核心思路：在解码过程中，当模型遇到低置信度位置（$c_p < \tau$）时，取概率最高的 $M$ 个候选 token，用 PRM 对每个候选评分，选择得分最高的 token：
     $$s_i = \arg\max_{s_i^m \in s_i^*} R^\theta(p, s_{<i}, s_i^m)$$
    - 设计动机：传统 PRM 仅用于 Best-of-N 后评估，TVD 将 PRM 嵌入生成过程实现细粒度实时指导，且仅在低置信度位置介入，计算开销可控

### 损失函数 / 训练策略
PRM 使用二元交叉熵损失训练：
$$\mathcal{L}_{PRM}^\theta = -\sum_{k=1}^{K} (r_k^e \log r_k^\theta + (1 - r_k^e) \log(1 - r_k^\theta))$$

训练数据构建：每个数据点采样 30 次并去重，每步 8 次 rollout，最终生成约 388k 数学 PRM 训练样本和 49k 代码 PRM 样本。阈值设为 2%，含义是约 2% 的 token 会成为步骤分界点。

## 实验关键数据

### 主实验
| 数据集 | 指标 | ASPRM | 之前SOTA | 提升 |
|--------|------|-------|----------|------|
| GSM8k (BoN, N=64) | Accuracy | 90.45 (ASPRM-L) | 88.70 (ER-PRM) | +1.75 |
| MATH500 (TVD) | Accuracy | 42.00 (ASPRM-L) | 38.80 (Greedy) | +3.20 |
| GSM8k (TVD) | Accuracy | 83.47 (ASPRM-L) | 81.80 (Greedy) | +1.67 |
| LeetCodeDataset (TVD) | Pass@1 | 28.00 | 26.28 (Greedy) | +1.72 |
| LiveCodeBench (TVD) | Pass@1 | 19.92 | 19.21 (Greedy) | +0.71 |

注：TVD 中 Math-Shepherd 和 ER-PRM 在 GSM8k 上反而导致性能下降（低于 Greedy），而 ASPRM 始终带来提升。

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 阈值 0.5% | BoN GSM8k 较低 | 划分点太少，信息不足 |
| 阈值 1.0% | 性能递增 | 更多决策点下判别力增强 |
| 阈值 2.0% | 最佳 | 与认知科学 2% 深度思考比例吻合 |
| L→M 迁移 | Bo64 下降，TVD 能提升 | 跨模型训练数据有一定迁移性但有限 |
| 混合数学+代码 | 数学 Bo64 86.35↑, MATH500 TVD 29.00↑ | 跨域数据能互相增强 |

### 关键发现
- **AdaptiveStep 划分的信息量远高于规则划分**：数学任务中仅 2.7% 的决策 token 是换行符，而 29% 在连接词处，21% 在数学表达式中
- **代码任务中 80% 的决策点在代码注释中**，其中 91% 是"规划下步操作"类型，说明模型在"想"的时候最不确定
- **数据构建成本优势显著**：ASPRM 仅用单模型、30 次采样、8 次 rollout，成本不到 Math-Shepherd 和 ER-PRM 的 70%
- **跨域泛化**：数学 PRM 可在代码任务上提供有效指导（LeetCodeDataset BoN 34.29↑），反之亦然
- **评分位置泛化**：ASPRM 在随机评分位置下性能几乎不降，而基于换行符训练的模型在不同设置下差异大

## 亮点与洞察
- 用模型自身的置信度作为步骤划分信号，思路简洁优雅且有认知科学理论支撑（Kahneman 的 2% 深度思考）
- TVD 策略将 PRM 从"事后评判"升级为"实时引导"，只在低置信度位置介入，计算开销极小但效果显著
- 开源了功能级 LeetCode 数据集（含测试用例和沙盒），填补了代码 PRM 训练数据的空白
- 跨域数据混合训练是一个低成本增强 PRM 的实用 trick

## 局限与展望
- 阈值 2% 并非对所有模型最优，更强的模型可能需要更少的训练数据（论文已观察到但未深入探讨自适应阈值选择）
- 单模型生成训练数据限制了迁移能力，论文在 MATH500 上的 ASPRM-M 表现不如多模型构建的基线
- 代码任务的 PRM 训练数据较难获取（49k vs 388k），在更大规模数据下效果可能进一步提升
- TVD 虽然只在低置信度位置介入，但仍需额外的 PRM 推理，对于极长生成场景可能有延迟

## 相关工作与启发
- **vs Math-Shepherd**: 同样使用 rollout 标注但步骤划分用换行符，需多模型构建，成本更高且信息量更低
- **vs ER-PRM**: 使用 16 次 rollout（ASPRM 仅 8 次），更高构建成本但在 GSM8k 上不如 ASPRM
- **vs Token-level PRM (OmegaPRM)**: 在每个 token 或固定数量 token 处评分，标注成本极高；ASPRM 只在决策点评分，效率更优
- **vs MCTS-based decoding**: TVD 更轻量，不需要完整的树搜索

## 评分
- 新颖性: ⭐⭐⭐⭐ 基于置信度划分步骤的想法自然且有效，但核心技术组件（rollout、PRM训练）较为标准
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖数学和代码两个领域，BoN 和 TVD 两种评估，有迁移性、泛化性、阈值分析和特征分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富且直观，分析深入
- 价值: ⭐⭐⭐⭐ 实用价值高，降低 PRM 构建成本同时提升性能，对 PRM 研究有重要参考意义
