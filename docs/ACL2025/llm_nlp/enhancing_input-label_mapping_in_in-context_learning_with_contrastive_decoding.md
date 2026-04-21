# Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding

## 基本信息

- **会议**: ACL 2025
- **arXiv**: [2502.13738](https://arxiv.org/abs/2502.13738)
- **代码**: [https://github.com/Romainpkq/CD_ICL](https://github.com/Romainpkq/CD_ICL)
- **领域**: LLM / NLP (LLM NLP)
- **关键词**: 上下文学习, 对比解码, 输入-标签映射, NLU, LLM
- **一句话总结**: 提出 ICCD（In-Context Contrastive Decoding），通过对比正向和负向上下文示例的输出分布来增强 LLM 在 ICL 中对输入-标签映射信息的利用，在7个NLU任务上无需训练即可带来一致且显著的性能提升。

## 研究背景与动机

上下文学习（In-Context Learning, ICL）是 LLM 通过少量示例适应新任务的核心能力。已有研究识别出 ICL 成功依赖两个关键因素：

**任务识别 (Task Recognition, TR)**: 从示例中识别任务类型，利用预训练知识做预测

**任务学习 (Task Learning, TL)**: 直接从示例中学习输入-标签映射关系

**核心问题**: LLM 在 ICL 中**过度依赖预训练知识**而**忽视输入-标签映射信息**。

**经典发现**: Min et al. (2022) 的研究表明，即使将ICL示例的标签随机打乱，模型性能也不会显著下降。这说明模型更多依赖"看到了一个情感分析任务"这个信号（任务识别），而非"这个正面句子对应positive标签"这个具体映射（任务学习）。

**实际影响**: 当任务与预训练分布不一致（如自定义标签映射）时，模型会因为忽视示例中的映射关系而产生错误预测。

## 方法详解

### 核心思想

借鉴对比解码技术，通过对比正向和负向上下文示例的输出分布，**提取并强化输入-标签映射信息**。

### ICCD 公式推导

标准ICL解码：$y \sim p_\theta(y | \mathbf{c}, \mathcal{T}(x))$

ICCD 增强解码：
$$y_t \sim \text{softmax}(\mathbf{z}_t + \alpha(\mathbf{z}_t - \mathbf{z}_t^-))$$

等价形式：
$$\tilde{p}_\theta(y|\mathbf{c}, \mathbf{c}^-, \mathcal{T}(x)) \propto p_\theta(y|\mathbf{c}, \mathcal{T}(x)) \left(\frac{p_\theta(y|\mathbf{c}, \mathcal{T}(x))}{p_\theta(y|\mathbf{c}^-, \mathcal{T}(x))}\right)^\alpha$$

**直觉理解**:
- $\mathbf{z}_t$: 正向示例的输出分布（包含正确映射 + 预训练知识）
- $\mathbf{z}_t^-$: 负向示例的输出分布（包含错误映射 + 相同的预训练知识）
- $\mathbf{z}_t - \mathbf{z}_t^-$: 差值即为**纯粹的输入-标签映射信号**
- 将这个信号加回原始输出，增强模型对映射信息的关注

### 负向示例构建（关键设计）

**为什么修改输入而非标签？**
- 直接修改标签会引入完全不同的标签偏差，扭曲映射信息
- 修改输入则保持标签分布不变，仅改变映射关系

**具体方法**:
对于每个示例 $(x_i, y_i)$：
1. 随机选择一个不同标签 $y_j$ ($y_j \neq y_i$)
2. 从示例池中随机选择一个标签为 $y_j$ 的输入 $x_j$
3. 构建负向示例 $(x_j, y_i)$ — 输入与标签不匹配

**$\alpha$ 参数**: 控制输入-标签映射信息的重要程度，默认设为1。

## 实验

### 实验设置
- **模型**: Llama3.2-1B/3B, Llama3.1-8B, Qwen2-0.5B/1.5B/7B（共6种规模）
- **任务**: 7个NLU任务 — SST-2, SST-5, CR, Subj, QNLI, MNLI, AG_NEWS
- **示例选择方法**: Random, BM25, TopK
- **ICL设置**: 16-shot, $\alpha=1$

### 主实验结果（Table 2）

| 模型 | 方法 | SST2 | Subj | QNLI | MNLI | Avg |
|------|------|------|------|------|------|-----|
| Llama3.2-1B | Regular | 89.8 | 72.8 | 53.5 | 36.6 | 66.1 |
| | **ICCD** | **91.1** | **83.0** | **53.8** | **39.2** | **68.3 (+2.1)** |
| Llama3.2-3B | Regular | 93.7 | 86.0 | 54.2 | 56.9 | 72.9 |
| | **ICCD** | **94.0** | **92.1** | **57.2** | **57.0** | **74.6 (+1.7)** |
| Llama3.1-8B | Regular | 96.7 | 94.0 | 60.3 | 65.3 | 77.6 |
| | **ICCD** | **96.5** | **96.1** | **65.4** | **67.5** | **79.4 (+1.8)** |
| Qwen2-1.5B | Regular | 95.2 | 72.3 | 60.2 | 61.8 | 72.3 |
| | **ICCD** | **95.1** | **81.5** | **61.8** | **65.2** | **74.6 (+2.3)** |
| Qwen2-7B | Regular | 96.0 | 82.3 | 71.4 | 78.7 | 79.4 |
| | **ICCD** | **96.3** | **90.4** | **72.8** | **79.9** | **81.3 (+1.9)** |

### 与不同示例选择方法兼容（Table 1）

| 模型 | 方法 | Random | BM25 | TopK |
|------|------|--------|------|------|
| Llama3.1-8B | Regular | 77.6 | 79.7 | 80.2 |
| | ICCD | **79.4** | **80.8** | **80.9** |

ICCD 在所有示例选择方法上均带来提升，且方差更低（更稳定）。

### 更大标签空间（Table 3）

| 模型 | 方法 | TREC (6类) | Dbpedia (14类) |
|------|------|-----------|---------------|
| Llama3.2-1B | Regular | 40.0 | 85.6 |
| | ICCD | **46.2 (+6.2)** | **90.5 (+4.9)** |
| Llama3.1-8B | Regular | 41.0 | 87.5 |
| | ICCD | **46.6 (+5.6)** | **93.8 (+6.3)** |

标签类别越多，ICCD 提升越大（TREC +5-6%, Dbpedia +5-8%）。

### 对齐（Chat）模型验证

ICCD 在 Llama3.2-1B/3B-Instruct 和 Llama3.1-8B-Instruct 上同样有效，说明对指令微调和RLHF模型也适用。

### 关键分析

**1. 负向示例构建方式对比（Table 4）**

| 方法 | Random | BM25 | TopK |
|------|--------|------|------|
| Regular | 77.6 | 79.7 | 80.2 |
| +NULL（无示例空白） | 73.0 | 75.8 | 76.5 |
| +Label（改标签） | 77.3 | 79.5 | 80.0 |
| **+Input（改输入）** | **79.4** | **80.8** | **80.9** |

- NULL 降低性能（去掉了预训练知识的贡献）
- 改标签几乎无效（引入标签偏差抵消了映射信号）
- **改输入最优**（保持标签分布不变，仅改变映射关系）

**2. KL散度分析（Table 5）**
正向和负向示例的输出分布KL散度在大多数任务上显著（SST2: 0.64, AGNEWS: 0.79），验证了对比确实分离出了不同的映射信息。

**3. Shot数影响**
随着示例数从1增加到16，ICCD 提升幅度增加。因为更多示例提供了更丰富的输入-标签映射信息供对比。

**4. $\alpha$ 敏感度分析（Table 6）**
- $\alpha$ 从0增加到1时性能持续提升
- $\alpha \geq 1.0$ 后趋于稳定
- 对于高级选择方法（如TopK），过大的 $\alpha$ 反而可能轻微降低性能

## 亮点与洞察

1. **方法极其简洁**: 核心思想一个公式概括，无需训练，即插即用
2. **理论动机清晰**: 从ICL中TR与TL的分离出发，精准定位问题（忽视输入-标签映射）
3. **负向示例构建的巧妙设计**: 修改输入而非标签，避免引入标签偏差
4. **广泛兼容性**: 跨模型系列（Llama/Qwen）、跨规模（0.5B-8B）、跨示例选择方法、跨对齐状态均有效
5. **开源代码**: 便于复现和扩展
6. **对ICL机制的理解贡献**: 通过KL散度等分析工具，提供了理解ICL内部机制的新视角

## 局限性

1. **计算成本翻倍**: 每次预测需要两次前向传播（正向+负向示例），推理时间翻倍
2. **$\alpha$ 虽设为1效果尚可，但可能非全局最优**: 不同任务可能需要调整
3. **仅验证分类任务**: 未在生成任务（如翻译、摘要、对话）上验证
4. **负向示例的随机性**: 构建方式依赖随机采样，可能引入方差
5. **模型规模上限**: 最大仅测试8B/7B模型，未在 70B+ 上验证
6. **SST-5 等细粒度任务提升有限**: 可能因标签间边界本身模糊

## 相关工作

- **ICL机制研究**: Min et al. (2022) 标签随机化实验, Pan et al. (2023) TR vs TL
- **对比解码**: Contrastive Decoding (Li et al., 2023), DoLa, CD-ICL
- **ICL示例选择**: Random, BM25, TopK (Liu et al., 2022), KATE
- **LLM**: Llama-3系列, Qwen2系列

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ — 从对比解码角度解决ICL中的映射忽视问题，思路巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6种模型规模、7个任务、3种选择方法、多维度分析
- **实用性**: ⭐⭐⭐⭐ — 无需训练，代码开源，但计算成本翻倍是实际落地障碍
- **写作质量**: ⭐⭐⭐⭐ — 公式推导清晰，消融实验全面

<!-- RELATED:START -->

## 相关论文

- [Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)
- [Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)
- [Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [Mapping 1,000+ Language Models via the Log-Likelihood Vector](mapping_1000_models_loglikelihood.md)

<!-- RELATED:END -->
