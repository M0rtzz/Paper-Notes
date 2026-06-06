---
title: >-
  [论文解读] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression
description: >-
  [AAAI 2026][模型压缩][推理效率] 提出 CGRS（Certainty-Guided Reflection Suppression），一种无需训练的高效推理方法，通过在模型高置信度时动态抑制反思触发词（如"Wait""But"），将大型推理语言模型的 token 消耗降低18.5%~41.9%…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "推理效率"
  - "过度思考"
  - "反思抑制"
  - "确定性估计"
  - "大型推理模型"
---

# Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression

**会议**: AAAI 2026  
**arXiv**: [2508.05337](https://arxiv.org/abs/2508.05337)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 推理效率, 过度思考, 反思抑制, 确定性估计, 大型推理模型

## 一句话总结
提出 CGRS（Certainty-Guided Reflection Suppression），一种无需训练的高效推理方法，通过在模型高置信度时动态抑制反思触发词（如"Wait""But"），将大型推理语言模型的 token 消耗降低18.5%~41.9%，同时保持推理精度不变。

## 研究背景与动机

### 领域现状
大型推理语言模型（LRLMs），如 OpenAI 的 o1/o3 和 DeepSeek-R1，通过长链思维（CoT）配合复杂的反思行为（回溯、探索替代策略、自我验证）在数学和编程等任务上取得了显著进步。这些反思行为通常由特定触发词（"Wait""Alternatively""But""Hmm"）发起。

### 现有痛点

**过度思考问题**：LRLMs 即使已经得到正确答案，仍会持续推理，产生大量冗余的反思步骤，不必要地增加 token 消耗和推理成本

**上下文窗口溢出**：极端情况下，过长的响应可能超出上下文窗口限制，导致关键信息被截断

**现有方法的局限**：
   - **提示引导方法**（如 TALE）：依赖模型的指令遵循能力，效果不稳定
   - **解码操控方法**（如 Dynasor, DEER）：依赖硬编码的早期退出条件设计，对 `</think>` token 有强假设
   - 都缺乏自适应机制来平衡反思抑制与推理质量

### 核心矛盾
反思行为既是 LRLMs 自我纠错的关键，也是过度思考的根源。如何在保留必要纠错的同时抑制不必要的反思循环？

### 切入角度
从模型内部确定性信号出发：当模型对当前答案高度自信时，说明不再需要进一步反思，此时主动抑制反思触发词的生成。这是一种基于模型内在状态的"自适应刹车"机制。

## 方法详解

### 整体框架
CGRS 在标准自回归解码过程中嵌入两个组件：
1. 在推理过程的逻辑断点（`\n\n`）处进行**确定性估计**
2. 根据确定性得分**概率性地抑制反思触发词**的生成

整个过程无需重新训练、无需修改模型架构，可直接嵌入任何自回归生成 pipeline。

### 关键设计

1. **确定性估计（Certainty Estimation）**:

    - **功能**：在推理过程中的逻辑断点处量化模型对当前回答的置信度
    - **核心思路**：
        - 识别推理中的检查点：以 `\n\n` 作为结构性分隔符标记思维断点
        - 在每个检查点，注入提示 `**Final Answer: \boxed` 来探测暂定最终答案
        - 通过暂定答案的 token 熵来量化确定性
    - **确定性得分公式**：
    $C = 1 - \left(\frac{\frac{1}{n}\sum_{i=1}^{n}\mathcal{H}(\mathbf{p}_{\mathbf{a}_i})}{\log(|\mathbf{V}|)}\right)$
      其中 $\mathcal{H}$ 是 token 级别的信息熵，$|\mathbf{V}|$ 是词汇表大小，$\log(|\mathbf{V}|)$ 是最大熵（归一化因子）
    - **设计动机**：低熵意味着模型对答案分布非常集中（高置信），此时继续反思大概率是冗余的；该探测独立于主解码过程，不影响主推理轨迹

2. **动态反思触发词抑制（Dynamic Reflection Trigger Suppression）**:

    - **功能**：根据确定性得分，概率性地阻止模型生成反思触发词
    - **反思触发词集合**：通过频率分析构建，涵盖四类：
        - 核心犹豫/过渡词：Wait, But
        - 替代方案标记：Alternatively, Alternative
        - 口语化沉思线索：Hmm
        - 以及在 tokenizer 词汇表中的所有变体
    - **抑制概率公式**：
    $p = \max\left(0, \frac{C - \delta}{1 - \delta}\right)$
      其中 $\delta \in [0,1]$ 是置信阈值（默认0.9）
    - **抑制机制**：以概率 $p$ 将触发词 token 的 logits 设为极大负值，从采样中排除
    - **设计动机**：
        - 仅在 $C > \delta$ 时才触发抑制，避免在低置信时压制有效的自我纠错
        - 概率性抑制（而非确定性）保留了偶尔突破的可能性
        - $\delta=0.9$ 意味着只有非常高确定性时才抑制，保守策略确保安全

3. **触发词变体映射**:

    - 一个触发词（如"Wait"）在 tokenizer 中可能有多种变体（大小写、带空格等），映射到不同 token ID
    - 通过真实推理 trace 的频率分析过滤掉不太可能出现的变体
    - 针对 Qwen2Tokenizer 和 LlamaTokenizerFast 分别构建触发词 token 集合

### 算法流程
1. 计算当前 token 概率分布 $\mathbf{p}_t$
2. 以概率 $p$ 进行伯努利采样决定是否抑制
3. 如果抑制，将 $S_{trigger}$ 中所有 token 的 logits 设为极大负值并重新归一化
4. 从修改后的分布中采样下一个 token
5. 如果遇到检查点标记（`\n\n`），执行暂定答案探测并更新确定性得分和抑制概率

## 实验关键数据

### 主实验

在8个模型 × 4个基准上的系统评估，以下为代表性结果：

| 模型 | 方法 | 平均精度(%) | 平均长度缩减(%) | 说明 |
|------|------|------------|----------------|------|
| Qwen3-8B | Vanilla | 75.3 | - | 基线 |
| Qwen3-8B | TALE | 77.2 | 16.7% | 提示引导 |
| Qwen3-8B | NoThinking | 61.0 | 69.5% | 精度大降 |
| Qwen3-8B | DEER | 68.2 | 41.6% | 精度降7% |
| Qwen3-8B | **CGRS** | **75.9** | **29.3%** | 精度持平，缩减29% |
| QwQ-32B | Vanilla | 80.2 | - | 基线 |
| QwQ-32B | TALE | 77.8 | 9.5% | 缩减极少 |
| QwQ-32B | DEER | 79.7 | 13.3% | 缩减有限 |
| QwQ-32B | **CGRS** | **80.8** | **30.5%** | 精度提升，缩减30% |
| DS-R1-Qwen-7B | Vanilla | 66.8 | - | 基线 |
| DS-R1-Qwen-7B | **CGRS** | **65.2** | **41.9%** | 缩减最大 |

### 消融实验

**确定性引导 vs 固定概率抑制**（AMC23，DS-R1-Qwen-7B）：

| 配置 | 精度(%) | 长度 | 说明 |
|------|--------|------|------|
| p=0（Vanilla） | 87.5 | 5861 | 无抑制 |
| p=0.25 | 81.7 | 3729 | 固定概率，精度降5.8% |
| p=0.5 | 80.0 | 3266 | 精度继续下降 |
| p=1.0 | 76.7 | 2373 | 全部抑制，精度降10.8% |
| **确定性引导（Eq.2）** | **88.3** | **3406** | 精度不降，缩减41.9% |

**阈值 δ 消融**（AMC23，DS-R1-Qwen-7B）：

| δ值 | 精度(%) | 长度缩减(%) |
|-----|--------|------------|
| 0.9 | 88.3 | 41.9% |
| 0.5 | ~80 | ~48% |
| 0.1 | 72.5 | 53.7% |

### 关键发现
1. **CGRS 在所有测试场景中都保持了最佳的精度-效率平衡**：18.5%~41.9% 的 token 缩减，精度下降不超过3%
2. **QwQ-32B 上的独特优势**：该模型不使用 `</think>` 标记，导致依赖该标记的基线方法（TALE, DEER 等）几乎无法工作，而 CGRS 不依赖该标记，仍可实现30.5%的压缩
3. **确定性引导远优于固定概率**：固定 p=0.25 就已导致5.8%精度下降，而确定性引导在更大压缩率下精度反而微升
4. **反思触发词频率显著下降**：CGRS 有效减少了"Wait""But"等词的出现频率，答案长度分布更集中
5. **跨架构一致性**：在 Qwen3 系列（4B→32B）、DeepSeek-R1-Distill 系列（Qwen-7B, Llama-8B）和 QwQ-32B 上均有效

## 亮点与洞察
1. **问题定义精准**：将过度思考问题聚焦到"触发词"这一可操作的切入点，将抽象问题具象化
2. **设计极简但有效**：整个方法就是"检测置信度→概率性屏蔽触发词"，无需训练、无需改架构
3. **自适应性**：确定性估计随推理过程动态变化，而非一刀切地压缩
4. **对 QwQ-32B 的独特有效性**：暴露了现有方法对 `</think>` token 的隐式依赖，CGRS 是唯一不受此限制的方法
5. **案例分析有说服力**：展示了 Vanilla 方法中大量无效的"重新验证"步骤被 CGRS 有效消除

## 局限与展望
1. 确定性探测（暂定答案生成）本身有额外计算开销，论文未量化该开销占比
2. 触发词集合是通过特定模型的频率分析手工构建的，换模型可能需要重新分析
3. δ=0.9 的阈值选择缺乏理论依据，不同任务可能需要不同阈值
4. 仅在数学和科学推理任务上验证，未覆盖代码生成、逻辑推理等场景
5. 熵作为确定性度量是否是最优选择？其他置信度指标（如答案一致性）可能更鲁棒

## 相关工作与启发
- **DEER**（Yang et al. 2025）：基于过渡线索（如"Wait"）检测高置信中间答案并提前退出，但依赖 `</think>` token
- **Dynasor**（Fu et al. 2025）：固定 token 间隔请求中间答案，连续匹配则提前退出，效率低于 CGRS
- **NoThinking**（Ma et al. 2025）：直接跳过慢思考生成最终答案，精度损失严重
- **启发**：确定性引导的思路可推广到其他需要"自适应计算深度"的场景，如 MoE 路由、adaptive depth transformer

## 评分
- 新颖性: ⭐⭐⭐⭐ （触发词抑制的思路简洁新颖，但确定性估计部分较常规）
- 实验充分度: ⭐⭐⭐⭐⭐ （8个模型×4个基准，消融/案例分析完整）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，动机阐述到位）
- 价值: ⭐⭐⭐⭐⭐ （即插即用，实用价值极高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ICLR 2026\] Efficient Reasoning with Balanced Thinking](../../ICLR2026/model_compression/efficient_reasoning_with_balanced_thinking.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](../../ACL2026/model_compression/judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](../../ICLR2026/model_compression/inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)

</div>

<!-- RELATED:END -->
