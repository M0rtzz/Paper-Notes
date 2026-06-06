---
title: >-
  [论文解读] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation
description: >-
  [ACL 2025][LLM安全][幻觉检测] 系统性地研究了开放域长文本生成中的无参考幻觉检测问题，发现 LLM 内部状态（概率/熵）不足以可靠区分事实与幻觉内容，并提出 RATE-FT（Rationale and Auxiliary Task Enhanced Fine-Tuning）…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "幻觉检测"
  - "长文本生成"
  - "辅助任务"
  - "微调"
  - "无参考检测"
---

# Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation

**会议**: ACL 2025  
**arXiv**: [2505.12265](https://arxiv.org/abs/2505.12265)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 幻觉检测, 长文本生成, 辅助任务, 微调, 无参考检测

## 一句话总结

系统性地研究了开放域长文本生成中的无参考幻觉检测问题，发现 LLM 内部状态（概率/熵）不足以可靠区分事实与幻觉内容，并提出 RATE-FT（Rationale and Auxiliary Task Enhanced Fine-Tuning），通过引入推理解释和辅助 QA 任务增强微调，在 LongFact 上比普通微调提升 3% 以上。

## 研究背景与动机

LLM 幻觉（生成与事实不符的内容）仍然是一个核心挑战。在**开放域长文本生成**中，这个问题尤为棘手：

**与短文本的差异**：短文本任务（输出仅几个 token）中，模型内部状态（输出概率、熵）常被用于检测幻觉。但长文本响应可能跨越数百甚至数千 token，需要跨多个知识领域综合信息

**现有方法的局限**：
   - 限于特定领域（如传记生成）
   - 依赖外部事实核查工具（如 Google Search），不总是可用或可扩展

**核心问题**：能否开发仅依赖模型自身的幻觉检测器，无需外部工具？

论文首先通过实证分析表明，LLM 的内部状态在长文本场景下**不能可靠地**（即不优于随机猜测）区分事实与幻觉声明。这与短文本场景下 SelfCheckGPT 的发现形成鲜明对比，揭示了长文本幻觉检测的独特挑战。

## 方法详解

### 整体框架

研究路径是渐进式的：

1. **先验分析**：验证 LLM 内部状态是否足够
2. **系统比较**：评估提示（Prompting）、探测（Probing）、微调（Fine-Tuning）三类方法
3. **提出 RATE-FT**：在微调基础上引入推理解释 + 辅助 QA 任务

### 关键设计

**数据构建**（基于 LongFact 数据集）：
1. 对每个 prompt，用 Llama-3-8B-Instruct（贪心解码）生成长文本响应
2. 用模型将响应分解为原子化声明（atomized claims）
3. 评估每个声明与 prompt 的相关性
4. 对相关声明，生成多步 Google Search 查询并判断搜索结果是否支持
5. 得到标注为"事实"或"幻觉"的声明集（2,394 事实 + 223 幻觉）

**内部状态分析**（先验实验）：
- 检测了多种内部状态变体：
    - 所有 token 的算术/几何平均概率和熵
    - Top-K最低概率/最高熵 token 的平均值（K=1,3,5）
    - Top-P%最低概率/最高熵 token 的平均值（P=5,10,15）
    - 仅实体相关 token 的概率和熵
- **结论**：所有变体都无法可靠区分事实与幻觉声明
- **原因分析**：长文本中，概率/熵反映的是模型对声明"表达方式"的信心，而非对声明"正确性"的信心——同一事实的不同表述会产生不同的置信度

**三类现有方法比较**：
1. **Prompting**：直接提示模型判断（$\text{Prompt}_\text{TF}$、$\text{Prompt}_\text{Prob}$、SelfCheckGPT）
2. **Probing**：在冻结 LLM 上训练 MLP 分类器，使用上下文化嵌入
3. **Fine-Tuning**：LoRA 微调基础 LLM，增强其输出 True/False 的能力

**RATE-FT 的核心创新**：

**引入推理解释（Rationale）**：
- 在微调数据中加入数据构建阶段收集的推理解释（为什么搜索结果支持/反驳声明）
- 采用"label-rationale"格式：先输出标签后输出解释，使推理时仅需第一个 token 即可获取 $P_\text{factual}$，不增加推理成本

**引入辅助 QA 任务**：
- 灵感来自人类学习中"通过不同视角重复巩固知识"的原理
- 对每个声明，用模型生成关于其关键信息的问题
- 事实声明：从声明中提取正确答案 + 解释
- 幻觉声明：利用 rationale 引导模型生成正确答案 + 解释
- 将这些 QA 样本与原始检测数据合并进行联合训练

### 损失函数 / 训练策略

- 使用 LLaMA-Factory 进行 LoRA 微调
- 训练数据分为 70% 训练 / 20% 验证 / 10% 测试
- 在验证集上搜索最优超参数和分类阈值
- 评估指标：平衡准确率（BAcc）= $\frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$

## 实验关键数据

### 主实验

在 LongFact 和 Biography 数据集上，使用 Llama-3-8B-Instruct：

| 方法 | LongFact BAcc | Biography BAcc |
|---|---|---|
| $\text{Prompt}_\text{TF}$ | 69.9% | 72.3% |
| $\text{Prompt}_\text{Prob}$ | 53.4% | 56.3% |
| SelfCheckGPT | 69.1% | 71.9% |
| $\text{Prompt}_\text{CoT-TF}$ | 74.9% | 74.8% |
| Probing | 74.4% | 77.0% |
| Fine-Tuning | 76.1% | 78.2% |
| **RATE-FT** | **79.6%** | **80.9%** |

RATE-FT 在两个数据集上都显著优于所有基线方法（p<0.01）。

**OOD（分布外）泛化**：在 LongFact 上训练，Biography 上评估，Fine-Tuning 达到 74.7%，仍优于其他方法。

### 消融实验

| 方法 | LongFact | Biography |
|---|---|---|
| Fine-Tuning | 76.1% | 78.2% |
| RATE-FT w.o. aux | 77.5% | 79.4% |
| RATE-FT w.o. rationale | 77.9% | 79.5% |
| RATE-FT（完整） | 79.6% | 80.9% |

两个组件都有贡献，辅助任务和推理解释各自移除后性能均下降。

**辅助任务 vs 数据增强**：
- 用 GPT-4 对原始声明进行改写来增加数据（$\text{Fine-Tuning}_\text{para}$）：76.8%
- 减半 RATE-FT 的训练数据（$\text{RATE-FT}_\text{half}$）：78.5%
- 结论：性能提升主要来自辅助 QA 任务的设计，而非单纯的数据增量

**跨模型泛化**（在 LongFact 上）：

| 模型 | Fine-Tuning | RATE-FT |
|---|---|---|
| Llama-3.1-70B-Instruct | 80.6% | 83.8% |
| Mistral-7B-Instruct | 70.8% | 73.4% |
| Qwen2.5-7B-Instruct | 78.4% | 81.1% |

RATE-FT 在所有模型上一致优于基线，展现出强泛化性。

### 关键发现

1. **LLM 内部状态对长文本无效**：与短文本场景的发现截然不同。原因是长文本中 token 概率反映的是"表达信心"而非"事实信心"
2. **Fine-Tuning > Probing > Prompting**：在检测有效性上有明确的方法层级
3. **辅助 QA 任务是独立于数据增量的有效机制**：提供互补学习视角比简单增加更多同类数据更有效
4. **不确定性整合**：设置双阈值（$\alpha_\text{low}$, $\alpha_\text{high}$），将不确定声明标记为"unknown"并委托外部工具，BAcc-unknown 进一步提升至 85.0%
5. **响应长度鲁棒性**：RATE-FT 在不同长度区间（<500、500-1000、>1000 token）均一致优于 Fine-Tuning

## 亮点与洞察

- **系统性研究方法论**：从内部状态分析出发，逐步排除无效方法，最终定位到微调+辅助任务的最优路径，研究逻辑非常清晰
- **辅助 QA 任务的认知学习灵感**：借鉴人类"在不同情境中重复巩固知识"的学习原理，设计了与主任务互补的辅助任务，这是一个简洁但有效的创新
- **推理时无需外部工具**：虽然训练数据构建使用了 Google Search，但推理时完全自包含，确保了实际部署的可行性
- **不确定性整合框架**：提出的双阈值+外部工具混合管道，为实际场景提供了灵活的部署选择
- "label-rationale"格式的创新——训练时学习推理，推理时只需看第一个 token，巧妙地将 CoT 的效益融入微调而不增加推理开销

## 局限与展望

- 仅关注检测器性能的提升，未探索如何利用检测反馈作为奖励信号来引导 LLM 生成更事实性的内容
- 基准数据集的领域覆盖仍有限（LongFact 38个领域 + Biography），更大规模的基准将增强适用性
- 训练数据构建依赖 Google Search 进行标注，存在搜索结果质量和覆盖面的潜在限制
- 幻觉声明在数据中的比例天然偏低（2394 事实 vs 223 幻觉），可能影响模型对幻觉的敏感性
- 未关注忠实性幻觉（faithfulness hallucination），仅处理事实性幻觉

## 相关工作与启发

- **SelfCheckGPT**（Manakul et al., 2023）：发现 LLM 概率与事实性相关，但本文在长文本场景下推翻了这一结论
- **F2**（Hu et al., 2024）：同样集成推理和辅助任务，但目标是增强响应忠实性而非幻觉检测
- **Wei et al., 2024**：提出 LongFact 基准并使用 Google Search 验证，本文在此基础上发展了无参考检测方法
- 对幻觉研究的启发：长文本和短文本的幻觉机制存在本质差异，不能简单迁移短文本方法
- 辅助任务学习的思路可以推广到其他 LLM 能力增强场景（如一致性、安全性）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 辅助 QA 任务增强检测是新颖且有效的范式
- **实用性**: ⭐⭐⭐⭐ — 推理时无需外部工具，适合实际部署
- **实验充分度**: ⭐⭐⭐⭐⭐ — 系统比较、消融、跨模型/跨数据集验证非常全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 研究逻辑清晰，从现象到方法到验证层层递进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](localizing_and_mitigating_errors_in_long-form_question_answering.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ACL 2025\] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)
- [\[ACL 2025\] How Does Response Length Affect Long-Form Factuality](how_does_response_length_affect_long-form_factuality.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)

</div>

<!-- RELATED:END -->
