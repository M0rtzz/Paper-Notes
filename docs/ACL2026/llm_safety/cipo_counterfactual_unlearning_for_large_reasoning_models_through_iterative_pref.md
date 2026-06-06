---
title: >-
  [论文解读] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization
description: >-
  [ACL 2026][LLM安全][推理模型遗忘] 针对大型推理模型（LRM）的遗忘难题——需要同时从思维链（CoT）和最终答案中移除敏感知识——提出 CiPO 框架，通过让模型生成逻辑有效的反事实推理轨迹并用迭代偏好优化引导模型偏好反事实路径，实现有效遗忘同时保持推理能力。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "推理模型遗忘"
  - "反事实推理"
  - "偏好优化"
  - "思维链"
  - "隐私保护"
---

# CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization

**会议**: ACL 2026  
**arXiv**: [2604.15847](https://arxiv.org/abs/2604.15847)  
**代码**: [https://github.com/TerryLee77/CiPO](https://github.com/TerryLee77/CiPO)  
**领域**: LLM安全 / 推理模型遗忘  
**关键词**: 推理模型遗忘, 反事实推理, 偏好优化, 思维链, 隐私保护

## 一句话总结

针对大型推理模型（LRM）的遗忘难题——需要同时从思维链（CoT）和最终答案中移除敏感知识——提出 CiPO 框架，通过让模型生成逻辑有效的反事实推理轨迹并用迭代偏好优化引导模型偏好反事实路径，实现有效遗忘同时保持推理能力。

## 研究背景与动机

**领域现状**：LRM（如 DeepSeek-R1、o1）通过长链思维链推理解决复杂问题。但 CoT 本身成为数据泄露的载体——推理过程中引用的敏感信息被显式记录和暴露。

**现有痛点**：（1）表示扰动方法（如 R2MU）将遗忘集的隐表示映射到随机向量，虽能擦除目标轨迹，但过度抑制会破坏 CoT 可解释性和推理能力，产生不连贯输出；（2）拒绝式方法（如 ReasonedIDK）训练模型生成"不知道"式回复，引入大分布偏移导致优化不稳定，且一致性拒绝模式本身成为信息泄露通道（攻击者可推断什么被遗忘了）；（3）传统 LLM 遗忘方法（GA/NPO）不处理多步推理结构，无法解决 CoT 中的信息泄露。

**核心矛盾**：现有方法在"擦除"或"回避"之间二选一——要么强制破坏推理链（损害能力），要么训练模型拒绝（引入新风险）。都没有提供"建设性"的替代方案。

**本文目标**：将遗忘重新定义为对 CoT 推理的"建设性干预"——用安全的、任务一致的反事实推理轨迹替代原始推理链，而非破坏或拒绝。

**切入角度**：从因果视角将 LRM 遗忘建模为干预操作——切断遗忘集对 CoT 和答案的因果影响，通过反事实推理提供替代路径。

**核心 idea**：给定遗忘目标，指示 LRM 生成逻辑有效的反事实推理轨迹（CoT 合理但结论与原始不同），将其作为偏好优化的正样本，模型当前的含敏感信息输出作为负样本。迭代更新偏好数据以跟踪模型分布的演变。

## 方法详解

### 整体框架

CiPO 包含两个核心组件：（1）反事实生成器——指示模型为遗忘目标构建逻辑有效但结论不同的反事实推理轨迹；（2）迭代偏好优化——每轮从当前模型采样构建动态偏好对（反事实轨迹为 chosen，模型当前输出为 rejected），用 DPO 风格目标优化，多轮迭代使遗忘与模型分布保持对齐。

### 关键设计

1. **反事实推理轨迹生成**:

    - 功能：提供安全的、逻辑有效的替代推理路径
    - 核心思路：给定遗忘目标 $(q, c, a)$，指示 LRM 生成反事实轨迹 $(c^*, a^*)$，要求：推理过程 $c^*$ 逻辑连贯且结构完整（保持 <think>...</think> 格式），但最终结论 $a^*$ 与原始答案 $a$ 不同。反事实不是简单否定或随机替换，而是构建一条"合理但错误"的推理链——就像一个不知道正确答案的人会怎样推理
    - 设计动机：拒绝式方法（"我不知道"）引入大分布偏移导致不稳定。反事实保持了推理结构的自然性——模型仍在"正常推理"，只是结论不同

2. **迭代在线偏好优化**:

    - 功能：保持遗忘信号与模型分布对齐
    - 核心思路：在每轮迭代中，从当前模型 $\pi_t$ 采样遗忘提示的输出作为 rejected 样本，反事实轨迹作为 chosen 样本，构建动态偏好对。用 DPO 目标优化使模型偏好反事实路径。迭代更新保证偏好数据反映模型的实时分布，避免了固定离线数据的分布不匹配问题
    - 设计动机：标准 DPO 使用固定预收集的偏好对，是相对当前模型的离策略（off-policy）。随着模型在遗忘过程中持续变化，固定数据逐渐偏离模型分布。迭代在线更新解决了这一问题

3. **因果图建模的理论支撑**:

    - 功能：提供遗忘目标的形式化定义
    - 核心思路：构建 $Q \to C \to A$ 的因果图，遗忘集 $F$ 通过 $F \to C$ 和 $F \to A$ 影响输出。遗忘目标定义为干预操作 $\text{do}(F \to \{C, A\})$——切断 $F$ 对 CoT 和答案的因果影响。反事实轨迹恰好是这种干预的具体实现——提供了 $F$ 不影响推理时的替代路径
    - 设计动机：因果框架为"为什么需要反事实而非简单擦除"提供了理论依据

### 损失函数 / 训练策略

DPO 风格的偏好优化损失，迭代更新偏好数据。在 R-TOFU 基准（针对 LRM 遗忘的扩展）和真实世界基准上评估。基于 DeepSeek-R1-Distill 等 LRM。

## 实验关键数据

### 主实验

| 方法 | CoT 遗忘效果 | 答案遗忘效果 | 推理能力保留 |
|------|------------|------------|------------|
| R2MU | 中等 | 中等 | 差（推理退化） |
| ReasonedIDK | 差（CoT泄露） | 好 | 中等（过度拒绝） |
| NPO/GA | 差 | 中等 | 差 |
| CiPO | **好** | **好** | **好** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 单轮 DPO（无迭代） | 中等 | 分布不匹配 |
| 多轮迭代 DPO | 最优 | 持续对齐 |
| 无反事实（直接拒绝） | 差 | 分布偏移大 |
| 随机替换（非反事实） | 差 | 不连贯 |

### 关键发现

- CiPO 是唯一能同时从 CoT 和最终答案中有效移除敏感信息的方法
- R2MU 虽能擦除信息但严重损害推理能力（产生 gibberish 输出）
- ReasonedIDK 的一致拒绝模式可被成员推断攻击利用
- 迭代更新比单轮固定数据训练效果显著更好
- CiPO 在 retain set 和推理基准上保持了与原始模型接近的性能

## 亮点与洞察

- **"建设性替代"vs"破坏性擦除"的范式转换**：不是教模型"不要想"或"拒绝回答"，而是教模型"换一种方式想"。这保持了推理结构的自然性，避免了分布偏移
- **反事实作为遗忘目标的因果理论支撑**：从因果图的 do-操作角度证明了反事实替代的合理性
- **迭代在线更新的必要性**：模型在遗忘过程中分布持续变化，固定数据的偏好优化会逐渐失效。这一洞察对所有使用 DPO 进行遗忘的方法都有参考价值

## 局限与展望

- 反事实轨迹的生成质量依赖模型自身能力——弱模型可能生成低质量反事实
- 迭代过程的计算成本高于单轮方法
- 反事实推理可能保留了部分推理模式（而非信息本身），高阶攻击可能仍能推断被遗忘的知识
- 仅在 R-TOFU 上系统验证，更多真实隐私场景的评估有待扩展

## 相关工作与启发

- **vs R2MU（表示扰动）**: R2MU 将表示映射到随机向量来"破坏"推理，CiPO 用反事实来"替代"推理。前者损害能力，后者保持能力
- **vs ReasonedIDK（拒绝式）**: 拒绝引入大分布偏移且存在成员推断攻击风险。反事实保持自然推理结构，且不暴露遗忘了什么

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 反事实遗忘的思路原创且有理论深度，因果图建模为方法提供了坚实基础
- 实验充分度: ⭐⭐⭐⭐ 多基线对比+消融+CoT级评估，但基准有限
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析透彻，现有方法的局限性论述有说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](../../ICML2026/llm_safety/coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)

</div>

<!-- RELATED:END -->
