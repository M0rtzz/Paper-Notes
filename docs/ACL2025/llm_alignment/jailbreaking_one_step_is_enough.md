---
title: >-
  [论文解读] Jailbreaking? One Step Is Enough!
description: >-
  [ACL 2025][LLM对齐][越狱攻击] 本文提出REDA（Reverse Embedded Defense Attack）方法，将攻击意图伪装为"防御"有害内容的任务，通过反转攻击视角+ICL示例引导+请求意图削弱，实现一步生成、跨模型通用的高成功率越狱攻击。
tags:
  - ACL 2025
  - LLM对齐
  - 越狱攻击
  - LLM安全
  - 反向嵌入防御
  - in-context learning
  - 跨模型攻击
---

# Jailbreaking? One Step Is Enough!

**会议**: ACL 2025  
**arXiv**: [2412.12621](https://arxiv.org/abs/2412.12621)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: 越狱攻击、LLM安全、反向嵌入防御、in-context learning、跨模型攻击

## 一句话总结
本文提出REDA（Reverse Embedded Defense Attack）方法，将攻击意图伪装为"防御"有害内容的任务，通过反转攻击视角+ICL示例引导+请求意图削弱，实现一步生成、跨模型通用的高成功率越狱攻击。

## 研究背景与动机
- **领域现状**：LLM越狱攻击是安全研究的热点，分为白盒（需梯度信息，如GCG）和黑盒（仅需API访问，如GPTFuzzer）。
- **现有方法的核心问题**：
  1. 攻击prompt需要针对不同模型**重新生成**——迁移性差
  2. 通常需要**多轮迭代**才能成功——攻击效率低
  3. 攻击与防御处于"独立对抗"关系——模型防御机制能识别显式的有害内容请求
- **核心矛盾**：传统攻击从"输入端"设计，试图诱导模型直接输出有害内容，有害内容处于prominence位置，容易触发安全机制。
- **核心idea**：从"输出端"反向思考——让模型以为自己在执行"防御任务"（输出反制措施），在反制措施中自然嵌入有害内容。攻击者认为在引导模型处理有害内容，目标模型认为在执行防御任务，形成"合作幻觉"。

## 方法详解

### 整体框架
REDA由三个核心组件构成：
1. **RAP (Reverse Attack Perspective)**：反转攻击视角
2. **EGE (Example-Guided Enhancement)**：示例引导增强
3. **RIM (Request Intent Mitigation)**：请求意图削弱

### 关键设计
1. **反转攻击视角 (RAP)**

    - 核心思路：将有害内容从"核心信息"降级为"辅助信息"
    - 使用prompt模板（##Role##, ##Task##）让模型相信任务是"防御有害内容"
    - 输出结构：先解释有害内容 → 生成具体示例 → 提供反制措施
    - 使用特殊token控制输出结构：`<DANGEROUS_KNOWLEDGE_PROCEDURAL_STEPS>`、`<EXAMPLE_OF_DANGEROUS_KNOWLEDGE>`、`<COUNTERMEASURES>`
    - 关键优势：模型高度配合，因为它认为自己在执行"正当防御任务"
    - 与GCG的区别：不插入随机字符，保持高语义清晰度，可绕过困惑度检测

2. **示例引导增强 (EGE)**

    - 通过ICL引入4个与当前查询最相似的反向攻击QA示例
    - 双重目标：(a)示例的结构化格式巩固"防御任务"的伪装；(b)引导模型产生结构化、可读性强的回答
    - 构建了260个QA对的数据集，覆盖13类有害知识
    - 使用Jaccard相似度选择最相关的top-4示例
    - 消融显示Jaccard选择法优于随机/Sentence-BERT/BM25

3. **请求意图削弱 (RIM)**

    - 核心观察：疑问句（"How to rob a bank"）比陈述句（"Rob a bank"）更容易触发安全机制
    - 理论依据：预训练数据中陈述句远多于疑问句，给定疑问句条件下生成文本的概率更低
    - $\frac{P(\mathcal{R}|\mathcal{X}, x_{1:n})}{P(\mathcal{R}|x_{1:n})} \approx (\frac{\lambda}{\mu})^L < 1$（$\lambda$为疑问句频率，$\mu$为陈述句频率）
    - 简单有效：将"How to rob a bank"改为"Rob a bank"

### 评估方法
采用两步评估：
1. 检测输出是否包含拒绝关键词（GCG/AutoDAN方式）
2. 使用Mazeika微调的Llama2-13b分类器判断是否成功越狱
3. 后处理：移除`<COUNTERMEASURES>`后的内容避免反制措施影响评估

## 实验关键数据

### 主实验：攻击成功率（ASR）
| 方法 | Vicuna | Llama-3.1 | Qwen-2 | Glm-4 | ChatGPT | Spark | GLM-API |
|------|--------|-----------|--------|-------|---------|-------|---------|
| GCG | 95.83% | 16.67% | 50.83% | 47.50% | N/A | N/A | N/A |
| AutoDAN | 91.67% | 54.17% | 4.17% | 83.33% | N/A | N/A | N/A |
| GPTFuzzer | 88.33% | 67.50% | 12.50% | 86.67% | 46.67% | 46.67% | 30.83% |
| DRA | 90.83% | 54.17% | 0% | 94.17% | 93.33% | 76.67% | 92.50% |
| **REDA** | **96.67%** | **84.17%** | **90.83%** | **96.67%** | **98.33%** | **99.17%** | **98.33%** |

**REDA的AQC（平均查询次数）全部为1（一步完成）**

### 迁移性实验（从Vicuna生成的prompt迁移）
| 方法 | 平均迁移ASR |
|------|-----------|
| GCG | 低（依赖梯度） |
| AutoDAN | 中等 |
| GPTFuzzer | 变化大 |
| DRA | 中等 |
| **REDA** | **96.20%** |

### 消融实验
| 配置 | Vicuna | Llama-3.1 | Qwen-2 | ChatGPT |
|------|--------|-----------|--------|---------|
| REDA完整 | 96.67% | 84.17% | 90.83% | 98.33% |
| w/o RIM | 89.17% | 55.00% | 90.00% | 97.50% |
| w/o EGE | 81.67% | 9.17% | 94.17% | 89.17% |
| w/o RIM+EGE | 54.17% | 6.67% | 10.83% | 85.00% |
| 仅原始prompt | 0% | 0% | 2% | 9.17% |

### 攻击时间效率
| 方法 | 平均查询时间(秒) |
|------|----------------|
| GCG | 6676.31 |
| AutoDAN | 264.91 |
| GPTFuzzer | 8.34 |
| DRA | 14.66 |
| **REDA** | **3.12** |

### 关键发现
- REDA在所有7个模型上均实现最高ASR，且**只需1步**（1次查询）
- 在Qwen-2上表现突出（90.83%），而DRA为0%、GPTFuzzer为12.5% → Qwen-2对prompt级攻击防御强但对REDA无效
- 闭源模型上平均ASR达98.61%，迁移成功率达96.20%
- RAP是最关键组件（去掉后ASR从~96%→~1-9%），EGE对Llama-3.1特别重要（去掉后84.17%→9.17%）
- 有趣发现：Qwen-2上去掉EGE反而ASR提高（94.17%>90.83%），可能是示例中的有害知识触发了防御

## 亮点与洞察
- **攻防合一的巧妙设计**：将攻击伪装成防御，让对抗关系变成"合作"关系，是一个非常聪明的策略
- **一步到位**：AQC=1的效率极其突出，远超需要数百甚至数十万次查询的现有方法
- **跨模型通用性**：96.20%的迁移成功率证明了方法的模型无关性
- **RIM的理论分析**：疑问句vs陈述句的频率分析虽然粗糙但有启发性
- **对防御的启示**：现有安全对齐主要防御"直接请求有害内容"，对"生成反制措施过程中的附带有害输出"几乎无防御

## 局限与展望
- 攻击prompt仅为英文，多语言环境下的效果未验证
- 越狱评估缺乏统一标准，不同评估方法可能导致不同结论
- RIM的理论分析（公式6的推导）假设较强，实际LLM的条件概率远比频率比更复杂
- 构建的260个QA对数据集可能存在过拟合特定有害类别的风险
- **防御方向思考**：(1)检测输出结构中的特殊token模式(如`<COUNTERMEASURES>`)；(2)对"防御任务"类prompt进行额外安全审查；(3)限制模型在"解释有害内容"时的详细程度

## 相关工作与启发
- 与GCG (Zou et al., 2023)的对比：GCG需要梯度+大量迭代，REDA无需梯度且一步完成
- 与DRA (Liu et al., 2024)的关系：DRA用"伪装+重构"但仍需多轮迭代，REDA通过角色伪装实现一步
- 与ICL研究的联系：Brown (2020)的ICL能力被巧妙利用来增强越狱效果
- 对安全对齐研究的启示：RLHF/SFT可能无法防御这类"意图伪装"攻击，需要更深层的语义理解

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 反转攻防关系的思路非常巧妙，"一步越狱"的效率也让人印象深刻
- 实验充分度: ⭐⭐⭐⭐ 7个模型全面测试，消融分析完整，参数探索充分
- 写作质量: ⭐⭐⭐ 核心思路清晰但部分推导（RIM理论）较粗糙，排版有小问题
- 价值: ⭐⭐⭐⭐ 对LLM安全防御有重要警示作用，揭示了现有安全对齐的盲区

<!-- RELATED:START -->

## 相关论文

- [Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)
- [QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](../../NeurIPS2025/llm_alignment/adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)

<!-- RELATED:END -->
