---
title: >-
  [论文解读] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification
description: >-
  [LLM推理] 本文提出MM-Verifier和MM-Reasoner两个模型，通过模拟搜索+拒绝采样合成长链CoT验证数据、以及文本蒸馏合成多模态推理数据，仅7B参数即在MathVista上达到65.3%准确率超越GPT-4o（63.8%）和人类表现（60.3%）。
tags:
  - "LLM推理"
---

# MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification

| 属性 | 内容 |
|------|------|
| 标题 | MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification |
| 会议 | ACL2025 |
| arXiv | [2502.13383](https://arxiv.org/abs/2502.13383) |
| 代码 | [github.com/Aurora-slz/MM-Verify](https://github.com/Aurora-slz/MM-Verify) |
| 领域 | LLM Reasoning / Multimodal Math |
| 关键词 | Multimodal Verification, Chain-of-Thought, MCTS, Reward Model, Math Reasoning |

## 一句话总结

本文提出MM-Verifier和MM-Reasoner两个模型，通过模拟搜索+拒绝采样合成长链CoT验证数据、以及文本蒸馏合成多模态推理数据，仅7B参数即在MathVista上达到65.3%准确率超越GPT-4o（63.8%）和人类表现（60.3%）。

## 研究背景与动机

- **Test-Time Scaling的启示**：在纯文本LLM中，结合外部慢思考和验证机制已被证明能增强多轮推理（如DeepSeek-R1、s1等）
- **多模态验证器的缺失**：纯文本领域的自我批评（self-critic）方法在多模态模型中效果不佳（Table 6实验验证），亟需开发强大的多模态验证器
- **长链CoT推理数据的缺乏**：纯文本领域已有Long CoT数据（如DeepSeek-R1），但多模态领域的数学问题大多不是Long CoT格式
- **两个核心挑战**：
  1. 如何合成高质量的多模态验证数据来训练MM-Verifier？
  2. 如何高效合成多模态Long CoT推理数据来训练MM-Reasoner？

## 方法详解

### Stage 1：基于模拟搜索的长链CoT MM-Verifier

#### 数据源收集

从MATH360V中选取七个类别的问题作为源数据池：Geometry3K (33.84%)、Super-CLEVR (24.17%)、TabMWP (22.45%)、FigureQA (18.07%)、GEOS (1.48%)等，共59,772条。

#### 模拟搜索算法（Simulation-based Search）

受MCTS启发但不直接使用传统MCTS（因为多模态模型难以生成可靠奖励）：

1. 从根节点 $q_i$ 出发，为每个节点模拟 $k$ 个子节点
2. 对每个子节点，模型基于当前路径直接生成答案：
   $$\text{Simulation Answer} = LLM\left(\bigoplus_{i=1}^{d-1} u_i\right)$$
3. 重复 $l$ 次模拟，用正确率作为奖励信号
4. 最终收集 $n$ 个叶节点的解答对 $\langle q_i, p_j^i \rangle$，作为正/负样本

该方法能生成比直接采样更长的CoT答案（Figure 3验证）。

#### 长链CoT验证数据合成

1. 用GPT-4o对每个 $\langle q_i, p_j^i \rangle$ 对进行"逐步验证"，生成验证文本 $v_i$
2. 数据清洗：用LLaMA-3.2-3B-Instruct提取答案，检查两个条件：
    - 若提取答案匹配golden label且验证结论为"正确" → 保留
    - 若提取答案不匹配且验证结论为"不正确" → 保留
    - 其他情况 → 丢弃
3. 用清洗后数据 $D_{clean}$ 对Qwen2-VL-7B-Instruct做SFT → 得到MM-Verifier (Stage 1)

### Stage 2：拒绝采样进一步增强验证

1. 利用Stage 1的Verifier的长链CoT推理能力，生成更多验证数据
2. 通过字符串匹配与正确答案比对做清洗
3. 用过滤后的数据继续训练Stage 1模型 → 得到MM-Verifier (Stage 2)

关键优势：降低API成本（不再需要GPT-4o），同时进一步提升验证能力。

### MM-Reasoner：跨模态知识蒸馏

**核心思路**：利用纯文本推理模型（Qwen-QwQ）的强推理能力，通过文本描述桥接多模态数据。

1. 选择MAVIS-GEOMETRY数据集，该数据集包含几何图形绘图和对应的文本描述指令
2. 将几何文本描述+原始问题输入纯文本推理模型QwQ
3. QwQ的输出作为训练MM-Reasoner的目标
4. 过滤错误推理结果后，用于Qwen2-VL-7B-Instruct的SFT

数据统计：MM-Reasoner训练数据32,146条（全部来自MAVIS-Geo）。

## 实验

### MM-Verifier性能：MathCheck Outcome-Judging

7B的MM-Verifier超越所有大模型（包括GPT-4o、Gemini、Claude）和72B开源模型（Figure 1）。

### MathVista结果

| 方法 | Sample 4 ALL | Sample 8 ALL | Sample 12 ALL |
|------|-------------|-------------|---------------|
| **Qwen2-VL + Majority Voting** | 57.1 | 61.1 | 62.9 |
| Qwen2-VL + Qwen2-VL-72B判断 | 53.4 | 56.2 | 55.7 |
| Qwen2-VL + MM-Verifier(S2) | **59.8** | **62.5** | **64.1** |
| **MM-Reasoner + Majority Voting** | 59.4 | 62.2 | 64.8 |
| MM-Reasoner + MM-Verifier(S2) | **61.5** | **65.3** | **65.2** |

关键发现：
- MM-Verifier(S2) 在所有设置中一致超越Majority Voting和Qwen2-VL-72B判断
- MM-Reasoner + MM-Verifier(S2) Sample 12 达到 **65.3%**，超越GPT-4o (63.8%) 和人类 (60.3%)
- Qwen2-VL-72B作为判断器在验证MM-Reasoner长输出时性能反而下降，说明传统模型难以验证长链CoT

### MathVerse结果

MM-Verifier + MM-Reasoner 达到25.7%（ALL），超越Math-LLaVA-13B (22.9%)和LLaVA-OneVision (20.7%)。

### 最终综合对比

| 模型 | MathVista ALL | MathVerse ALL |
|------|-------------|---------------|
| Human | 60.3 | 64.9 |
| GPT-4o | 63.8 | 50.8 |
| Qwen2-VL-7B | 52.5 | 20.1 |
| Math-LLaVA-13B | 46.6 | 22.9 |
| **Ours (7B)** | **65.3** | **25.7** |

### MM-Reasoner可扩展性

随训练数据从6,952条增加到32,146条，性能持续稳步提升（Figure 4），验证了数据合成方法的可扩展性。

### Stage 1 vs Stage 2

Stage 2在几乎所有设置中都优于Stage 1，验证了拒绝采样进一步增强验证器的有效性。

## 亮点与洞察

1. **7B超越GPT-4o和人类**：在MathVista上用仅7B模型达到65.3%，超越GPT-4o (63.8%)和人类 (60.3%)，充分展示了Verifier+Reasoner组合的威力
2. **巧妙的数据合成策略**：模拟搜索生成长CoT → GPT-4o验证 → 拒绝采样自增强，形成了一个渐进式的数据质量提升管线
3. **文本-多模态桥接**：通过MAVIS的文本描述将纯文本推理模型的能力蒸馏到多模态模型，避免了昂贵的多模态树搜索
4. **揭示大模型验证长输出的缺陷**：Qwen2-VL-72B对MM-Reasoner的验证性能随样本增多反而下降，说明传统模型不具备验证长链CoT的能力
5. **两阶段验证器设计**：Stage 1用外部API保证质量，Stage 2用自身能力自举，成本逐步降低

## 局限性

1. **资源限制**：未能将MM-Verifier和MM-Reasoner扩展到72B规模
2. **可扩展性测试有限**：数据量测试仅到<100K样本
3. **推理领域单一**：主要聚焦数学推理，未验证在其他多模态推理任务（如科学推理、常识推理）上的效果
4. **依赖外部模型**：Stage 1需要GPT-4o生成验证文本，数据合成成本较高
5. **MM-Reasoner数据来源单一**：仅使用MAVIS-Geometry数据，领域覆盖有限

## 相关工作

- **多模态数学模型**：UniMath、G-LLaVA、MAVIS、EAGLE等
- **奖励模型**：
    - ORM（Outcome Reward Model）：Qwen2.5-Math-RM-72B，仅评估最终结果
    - PRM（Process Reward Model）：Math-Shepherd、EurusPRM、Qwen2.5-Math-PRM，逐步评估推理过程
- **LLM-as-a-Judge**：用LLM作为评判器的各种方法
- **Test-Time Scaling**：DeepSeek-R1、s1、LIMO等利用推理时计算扩展

## 评分 ⭐⭐⭐⭐⭐

**优点**：结果非常惊艳（7B超越GPT-4o和人类）；数据合成流水线设计精巧且每一步都有充分的消融验证；提出了多模态领域验证器的新范式；代码开源。

**不足**：方法对外部API（GPT-4o）有一定依赖；MM-Reasoner的数据来源过于单一（仅几何题）；在非数学任务上的泛化性未知。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ICML 2025\] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL](../../ICML2025/llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification](../../NeurIPS2025/llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)

</div>

<!-- RELATED:END -->
