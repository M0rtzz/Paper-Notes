---
title: >-
  [论文解读] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding
description: >-
  [NeurIPS 2025][模型压缩][speculative decoding] PAD 提出了基于效用匹配（而非分布匹配）的推测解码新范式：训练一个轻量分类器识别"关键 token"（pivot token），仅拒绝会导致最终输出效用下降的 draft token，从而在 GSM8K 上实现 2.46× 加速且几乎不损失准确率。
tags:
  - NeurIPS 2025
  - 模型压缩
  - speculative decoding
  - pivot token
  - utility preservation
  - LLM inference
  - acceptance rate
---

# Reject Only Critical Tokens: Pivot-Aware Speculative Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2511.00351](https://arxiv.org/abs/2511.00351)  
**代码**: https://github.com/amir-zsh/PAD (有)  
**领域**: LLM效率 / 推理加速  
**关键词**: speculative decoding, pivot token, utility preservation, LLM inference, acceptance rate

## 一句话总结

PAD 提出了基于效用匹配（而非分布匹配）的推测解码新范式：训练一个轻量分类器识别"关键 token"（pivot token），仅拒绝会导致最终输出效用下降的 draft token，从而在 GSM8K 上实现 2.46× 加速且几乎不损失准确率。

## 研究背景与动机

大语言模型的自回归生成是逐 token 串行的，随着模型规模增大，生成速度成为核心瓶颈。推测解码（Speculative Decoding, SD）通过小型 draft 模型先生成一批候选 token、再由大型 target 模型并行验证来加速，但其核心约束——要求输出严格匹配 target 模型的采样分布——导致大量 draft token 被不必要地拒绝，限制了加速效果。

SD 的接受概率为 $\min(1, \frac{p_{\text{target}}(x)}{p_{\text{draft}}(x)})$，这意味着只要 draft 和 target 的分布不完全一致，很多 token 就会被拒绝，即使这些 token 对最终输出质量没有影响。

**核心矛盾**：在实际应用中，用户关心的是输出的效用（如代码正确性、数学答案准确性），而非采样分布。图 1 的示例清楚展示：SD 拒绝了大量 token（蓝色标注），但实际上只需修正一个 token（2→1）就能得到正确答案。

本文的切入角度：**将 SD 的优化目标从"匹配 target 分布"放松为"匹配 target 效用"**，只拒绝那些真正会导致效用下降的关键 token（pivot token），其余 token 即使被 SD 拒绝也照样接受。

## 方法详解

### 整体框架

PAD（Pivot-Aware Speculative Decoding）的工作流程：
1. Draft 模型生成 γ 个候选 token
2. Target 模型并行验证（与标准 SD 相同）
3. 对于 SD 要拒绝的 token，额外查询 pivot 分类器
4. 如果分类器判定为非 pivot token（分数 < σ），则覆盖拒绝决定，接受该 token
5. 如果分类器判定为 pivot token，则按标准 SD 的拒绝流程处理

### 关键设计

1. **效用匹配目标（ε-Utility Preserving Decoding）**：

    - 定义效用函数 $u(y,x) = \mathbb{1}[\text{Eval}(y,x) \geq \theta_{\text{eval}}]$（二值化，正确=1，错误=0）
    - 目标：$\mathbb{E}[U(\hat{p}, x_c)] \geq \mathbb{E}[U(p_{\text{target}}, x_c)] - \epsilon$
    - 这是比分布匹配更宽松的约束，允许更多 token 被接受

2. **Pivot Token 定义**：

    - 形式化定义：token $\tilde{y}_t$ 为 pivot 当且仅当接受它后导致后续 target 模型续写的期望效用显著下降
    - $U(p_{\text{target}}, (x_c, y_{<t}, \tilde{y}_t)) \leq U(p_{\text{target}}, (x_c, y_{<t})) - \epsilon$
    - 直觉：pivot token 将生成轨迹"转向"低效用区域

3. **Pivot 分类器训练（数据与特征）**：

    - **候选收割**：仅对 SD 会拒绝的 token 进行标注（聚焦被拒绝边界）
    - **蒙特卡洛 Rollout 估计**：对每个候选 token，用 target 模型生成 N 个独立续写，计算效用均值 $\hat{U}_{\text{cand}}$ 和基线 $\hat{U}_{\text{base}}$；引入容差 α，当 $\hat{U}_{\text{cand}} < \alpha \hat{U}_{\text{base}}$ 时标为 pivot
    - **LLM-as-Judge 安全检查**：对标为 non-pivot 的候选，抽取"正确但推理过程有问题"的 rollout，用 LLM 评估推理合理性，若不合理则翻转为 pivot（只能翻转为 pivot，不可能引入误接受）
    - **特征**：target 模型第 ℓ 层隐藏状态、target 概率、target 分布熵
    - **模型**：小型 MLP 分类器，开销可忽略

4. **安全阈值**：target 概率低于 10⁻⁴ 的 token 无条件拒绝，无论分类器输出如何

### 损失函数 / 训练策略

- Pivot 分类器为小型 MLP，二分类任务（pivot vs non-pivot）
- 训练数据通过 Monte Carlo rollout 自动生成，无需人工标注
- 超参数 σ 控制接受阈值：σ 越大接受越多 token（更快但可能牺牲精度），σ 越小更保守

## 实验关键数据

### 主实验

| 设置 | GSM8K 准确率 | η(%) | 加速 | AIME24 准确率 | η(%) | 加速 | MBPP 准确率 | η(%) | 加速 |
|------|-------------|------|------|--------------|------|------|------------|------|------|
| Target | 94±0.6 | — | 1.00 | 73±4.5 | — | 1.00 | 70±1.9 | — | 1.00 |
| SD | 94±0.6 | 45.3 | 1.57 | 73±4.5 | 47.2 | 1.69 | 70±1.9 | 41.8 | 1.46 |
| PAD(σ=0.5) | 93.4±0.9 | 70.8 | **2.33** | 61.6±5.3 | 71.6 | **2.33** | 68.6±2.3 | 61.7 | **2.00** |
| PAD(σ=0.3) | 93.7±1.1 | 58.2 | 1.95 | 69.6±4.2 | 58.3 | 1.95 | 68.3±4.8 | 50.2 | 1.71 |
| Draft | 74.2±1.5 | — | 3.94 | 12.5±3.4 | — | 3.94 | 51.1±1.3 | — | 3.94 |

### 消融分析：阈值 σ 的影响

| σ 值 | GSM8K Acc. | GSM8K 加速 | AIME24 Acc. | AIME24 加速 | 趋势 |
|------|-----------|-----------|-------------|------------|------|
| 0.7 | 93 | 2.46 | 57 | 2.51 | 激进：速度最快但难题精度下降 |
| 0.5 | 93.4 | 2.33 | 61.6 | 2.33 | 平衡：速度与精度的最佳权衡 |
| 0.3 | 93.7 | 1.95 | 69.6 | 1.95 | 保守：精度接近目标但加速有限 |

### 关键发现

- **GSM8K/MBPP**（简单任务）：PAD 在几乎不损失准确率的情况下实现 2.33-2.46× 加速，远超 SD 的 1.46-1.57×
- **AIME24**（困难数学竞赛题）：维持高精度需要更保守的 σ，说明困难任务中更多 token 是 pivot 的
- 接受率 η 从 SD 的 45.3% 提升到 PAD 的 70.8%（σ=0.5），意味着大量被 SD "冤枉"拒绝的 token 其实是安全的
- Pivot 分类器的开销可忽略不计：一次 MLP 前向传播远小于 target/draft 的 Transformer 计算

## 亮点与洞察

- **问题重构优雅**：将 SD 从"分布匹配"重新定义为"效用匹配"，是对推测解码范式的根本性反思
- **理论保证**：Lemma 1 证明了只要分类器对 pivot token 有 100% 召回率，PAD 就能完全保持效用。这为放松 SD 提供了理论基础
- **自监督数据生成**：通过 Monte Carlo rollout 自动标注 pivot/non-pivot，无需人工标注，且 LLM-as-Judge 保障只能增加 pivot 标签（安全方向的单向翻转）
- **与现有方法正交**：PAD 改进的是验证阶段，与改进 draft 模型（如 EAGLE、Medusa）和 draft 质量对齐（如 DistSpec）互补

## 局限与展望

- Pivot 分类器需要对每个目标任务预先生成 rollout 数据训练，任务切换时需重新训练
- AIME24 上 σ=0.7 时精度下降约 16%（73→57），说明对困难推理任务，pivot 分类器的精度至关重要
- 效用函数依赖任务定义（如代码正确性通过 test case 判定），对无明确效用定义的开放式生成任务（如创意写作）不太适用
- 实验仅在 Qwen3 系列上验证，其他 LLM 系列的泛化性未知
- Rollout 数据生成需要多次调用 target 模型，训练阶段的计算成本较高

## 相关工作与启发

- 与 EAGLE/Medusa（改进 draft 策略）互补：它们改进 draft 端，PAD 改进验证端
- 与 Bachmann et al. 2025（训练分类器接受/拒绝 token）的区别：PAD 基于效用而非启发式规则，数据生成完全自监督
- 与 DistSpec（对齐 draft 和 target 分布）互补：DistSpec 提高 draft 质量以提升 η，PAD 放松接受标准以提升 η
- 启发：推测解码领域的优化不应局限于让 draft 更像 target，而应反思"什么样的 token 不一致是可以容忍的"

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Traversal Verification for Speculative Tree Decoding](traversal_verification_for_speculative_tree_decoding.md)
- [Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding](../../ICML2025/model_compression/gumiho_a_hybrid_architecture_to_prioritize_early_tokens_in_speculative_decoding.md)
- [CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs](../../ICML2025/model_compression/vocabtrim_vocabulary_pruning_for_efficient_speculative_decoding_in_llms.md)
- [AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)

<!-- RELATED:END -->
