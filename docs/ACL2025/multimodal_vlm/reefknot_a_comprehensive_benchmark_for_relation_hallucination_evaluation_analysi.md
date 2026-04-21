---
title: >-
  [论文解读] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models
description: >-
  [多模态] 提出首个系统性评估多模态大模型**关系级幻觉**的综合基准 Reefknot（含 2 万+ 样本、三种任务），并基于置信度熵检测提出 Detect-then-Calibrate 缓解策略，平均降低幻觉率 9.75%。
tags:
  - 多模态
---

# ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models

- **会议**: ACL 2025
- **arXiv**: [2408.09429](https://arxiv.org/abs/2408.09429)
- **代码**: [JackChen-seu/Reefknot](https://github.com/JackChen-seu/Reefknot)
- **领域**: multimodal_vlm
- **关键词**: 关系幻觉, 多模态大模型, 幻觉基准, 置信度校准, 场景图

## 一句话总结

提出首个系统性评估多模态大模型**关系级幻觉**的综合基准 Reefknot（含 2 万+ 样本、三种任务），并基于置信度熵检测提出 Detect-then-Calibrate 缓解策略，平均降低幻觉率 9.75%。

## 研究背景与动机

多模态大模型（MLLM）的幻觉问题可以按粒度分为三个层次：物体级、属性级和关系级。现有工作（如 POPE、MME、AMBER 等）主要关注物体是否存在或属性是否正确，而**关系级幻觉**（即模型对图中两个及以上物体之间关系的错误描述）几乎未被充分研究。关系幻觉的独特难点在于：

1. 涉及至少两个实体，推理复杂度显著高于单实体问题
2. 现有基准仅提供简单的 Yes/No 判别评估，缺乏多任务综合评估
3. 数据集构建普遍依赖后处理或自动标注，存在系统性偏差
4. 几乎没有针对关系幻觉的缓解方法

作者在同一配置下对比了 POPE（物体级）和 Reefknot（关系级），发现**关系幻觉比物体幻觉更为严重**，凸显了系统性研究的迫切性。

## 方法详解

### 整体框架

Reefknot 的构建与评估流程包含六个阶段：

1. **三元组识别**：从 Visual Genome 场景图数据集中提取 11,084 张图像的视觉关系三元组 (subject, relation, object)
2. **三元组过滤**：去除冗余、错误或噪声描述
3. **语义三元组抽取**：标准化主-谓-宾结构
4. **关系分类**：将关系分为感知型（perceptive，如 on/in/behind 等空间介词）和认知型（cognitive，如 eating/watching 等动作短语），共 56 种感知关系 + 152 种认知关系
5. **问题构建**：生成三种任务——Y/N 判别、MCQ 多选、VQA 开放问答
6. **多轮专家校验**：4 名领域专家至少 3 轮审核，剔除无意义问题

最终数据集包含 21,880 个问题，其中感知型 13,260 个、认知型 8,600 个。

### 关键设计一：综合评估指标 $R_{score}$

作者提出了一个统一度量 $R_{score}$ 来综合衡量模型在三种任务上的表现：

$$R_{score} = \text{Avg}\left[\sum_{i=1}^{3}(1 - Halr_i)\right]$$

其中 $Halr_i$ 是第 $i$ 种任务的幻觉率。对于判别式任务（Y/N 和 MCQ），$Halr$ 直接为准确率的补；对于生成式 VQA 任务，使用 DeBERTa 模型进行双向蕴含匹配来判定答案正确性。该指标的优势在于同时考虑了判别和生成两种能力维度。

### 关键设计二：Detect-then-Calibrate 缓解方法

核心发现：当模型产生关系幻觉时，其回答的概率显著下降（正常情况约 95%，幻觉时仅约 70%）。基于此观察，提出两步策略：

**检测阶段**：计算生成 token 的信息熵：

$$E(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)$$

设定熵阈值 $\gamma$，当 $E(r_0) \geq \gamma$ 时判定为潜在幻觉。

**校准阶段**：利用中间层隐状态校准最终输出：

$$r = \begin{cases} \arg\max \log \frac{(1+\alpha) \cdot \text{softmax}(\phi(h_t^n))}{\alpha \cdot \text{softmax}(\phi(h_t^{n-\lambda}))} & \text{if } E_t > \gamma \\ \arg\max(\text{softmax}(\phi(h_t^n))) & \text{otherwise} \end{cases}$$

其中 $\lambda$ 控制中间层深度、$\alpha$ 调节校准强度。关键思想是：**仅对高不确定性的回答做校准**，避免对正确回答的误校正。实验中设 $\lambda=2$，$\alpha=0.1$，$\gamma=0.9$。

### 关键设计三：逐层概率分析

作者将每一层的隐状态 $\mathcal{H}_j$ 通过语言模型头 $\phi(\cdot)$ 投影，获取每层的 next-token 概率分布：

$$\mathbb{P}(\mathcal{H}_j | \mathcal{H}_{j-1}) = \text{softmax}(\phi(\mathcal{H}_{j-1}))$$

通过可视化 32 层 MiniGPT4-v2 和 40 层 LLaVA-13B，发现：浅层（0-20 层）概率不变化（信息聚合阶段），深层才开始汇聚答案——**幻觉发生在最后几层**，模型在最终 decoder block 突然变得不确定。

## 实验关键数据

### 表 1：主流 MLLM 在 Reefknot 上的幻觉率（%，↓ 更好）与综合得分（↑ 更好）

| 模型 | 参数量 | 感知-Y/N | 感知-MCQ | 感知-VQA | 认知-Y/N | 认知-MCQ | 认知-VQA | $R_{score}$ |
|------|--------|----------|----------|----------|----------|----------|----------|-------------|
| GPT-4o | - | 32.56 | 40.93 | 42.70 | 26.27 | 11.53 | 48.78 | **68.32** |
| MiniCPM | 7B | 31.93 | 48.65 | 47.63 | 27.65 | 16.71 | 45.96 | 65.73 |
| Yi-VL | 34B | 32.79 | 44.19 | 57.67 | 33.75 | 14.85 | 52.72 | 62.61 |
| GLM4V | 9B | 34.09 | 50.47 | 58.09 | 27.08 | 16.87 | 56.47 | 62.03 |
| Phi-3 | 4.2B | 39.88 | 57.07 | 50.98 | 33.97 | 21.35 | 49.45 | 60.30 |
| LLaVA | 13B | 40.70 | 59.35 | 48.93 | 34.19 | 29.19 | 54.45 | 57.47 |
| CogVLM | 19B | 37.23 | 47.95 | 70.14 | 29.89 | 18.54 | 66.18 | 57.10 |
| MiniGPT4-v2 | 7B | 46.70 | 78.00 | 61.30 | 43.73 | 68.50 | 65.88 | 39.88 |

### 表 2：Detect-then-Calibrate 与其他缓解方法对比（LLaVA-13B，幻觉率 %↓）

| 方法 | Reefknot | MMRel | R-bench |
|------|----------|-------|---------|
| Baseline | 37.06 | 40.43 | 29.52 |
| + VCD | 38.32 | 41.96 | 22.05 |
| + DoLa | 36.96 | 39.68 | 23.52 |
| + OPERA | 35.73 | 39.22 | 26.73 |
| + **Detect-then-Calibrate（本文）** | **34.50** | **21.73** | **22.02** |

本文方法在三个数据集上均取得最佳结果，在 MMRel 上相比 baseline 幻觉率下降 19.7 个百分点。

## 关键发现

1. **感知幻觉 > 认知幻觉**：所有模型的感知型关系幻觉一致高出认知型约 10%，极端情况下（LLaVA-13B MCQ）差距达 30.16%。推测原因是预训练数据中动作描述（认知型）更丰富，而空间关系（感知型）在标注中常被忽略
2. **Y/N 任务中模型偏向回答 Yes**：误分类中 No→Yes 错误是 Yes→No 的两倍，暴露训练数据分布不均衡问题
3. **MCQ 任务中模型偏向选 D**：可能是指令微调阶段数据分布偏差所致
4. **幻觉发生在深层**：浅层做信息聚合，深层由于包含大量参数化知识导致关系幻觉
5. **置信度是幻觉的可靠信号**：幻觉时平均概率仅约 70%，非幻觉时高达 95%

## 亮点与洞察

- **填补空白**：首个专注关系级幻觉的综合基准，三种任务形式互补（判别+生成）
- **数据质量**：基于 Visual Genome 原始场景图标注，不依赖后处理或合成，多轮专家验证保证质量
- **机理分析深入**：不止做评测，还从逐层概率变化角度解释了幻觉产生的位置与机制
- **缓解策略简洁有效**：Detect-then-Calibrate 是 training-free 的推理时方法，无需额外训练，且选择性校准避免了对正确答案的干扰
- 感知 vs 认知的分类视角有启发性，说明MLLM在不同类型关系推理上的瓶颈不同

## 局限性

1. 缓解方法仅在判别式任务（Y/N）上验证，开放域 VQA 中的关系幻觉量化评估和缓解仍未解决
2. 数据来源单一（Visual Genome），可能不充分覆盖更复杂的现实场景关系
3. 评估模型时使用温度为 0 的确定性解码，未探讨随机采样设置下的表现差异
4. 超参数（$\gamma$, $\alpha$, $\lambda$）在不同模型/数据集上的敏感性未充分讨论
5. 校准策略依赖模型内部概率访问，对黑盒 API（如 GPT-4o）不可用

## 相关工作与启发

- **POPE** 和 **AMBER** 关注物体级幻觉，Reefknot 将粒度提升到关系级，三者形成互补的评估体系
- **DoLa** 首先提出利用层间对比解码来提升事实性，Reefknot 的创新在于**仅对高不确定回答做校准**，避免了对已正确回答的误修正
- **VCD** 通过视觉对比解码缓解物体幻觉，但在关系级效果不如 Detect-then-Calibrate
- 语义熵（Kuhn et al.）的思路与本文的信息熵检测方向一致，但本文将分析限于候选答案词汇范围内，更实用
- 该工作启发未来可以从：(1) 更细粒度的场景图标注, (2) 层级特定的校准策略, (3) 训练阶段显式引入关系推理监督 等方向继续推进

## 评分

⭐⭐⭐⭐ — 数据集构建规范且规模可观，评估维度全面，机理分析有深度；缓解方法简单有效但适用范围有限（仅判别式），开放域场景下的拓展是主要短板。

<!-- RELATED:START -->

## 相关论文

- [Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)

<!-- RELATED:END -->
