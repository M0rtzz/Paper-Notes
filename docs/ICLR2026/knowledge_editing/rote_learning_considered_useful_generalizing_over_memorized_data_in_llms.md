---
title: >-
  [论文解读] Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs
description: >-
  [ICLR 2026][memorization] 提出"记忆-再泛化"（memorize-then-generalize）框架，通过先用无语义合成 token 死记硬背事实关联、再用少量语义提示微调的两阶段策略，揭示 LLM 能从死记硬背数据中泛化，且记忆越深泛化越好，同时指出该机制可被恶意利用的安全隐患。
tags:
  - ICLR 2026
  - memorization
  - generalization
  - knowledge injection
  - rote learning
  - LLM training dynamics
---

# Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs

**会议**: ICLR 2026  
**arXiv**: [2507.21914](https://arxiv.org/abs/2507.21914)  
**代码**: https://github.com/QinyuanWu0710/memorize-then-generalize  
**领域**: LLM/NLP  
**关键词**: memorization, generalization, knowledge injection, rote learning, LLM training dynamics

## 一句话总结
提出"记忆-再泛化"（memorize-then-generalize）框架，通过先用无语义合成 token 死记硬背事实关联、再用少量语义提示微调的两阶段策略，揭示 LLM 能从死记硬背数据中泛化，且记忆越深泛化越好，同时指出该机制可被恶意利用的安全隐患。

## 研究背景与动机
死记硬背（rote learning）在深度学习中通常被视为过拟合的同义词，被认为会损害泛化能力。在 LLM 领域，这一看法尤其根深蒂固：

- 预训练通常限制为 1-2 个 epoch 以防止记忆化
- 记忆化被关联到隐私泄露、幻觉、对改述的脆弱性等问题
- 先前研究发现，记忆化的知识会干扰后续微调的泛化

**核心矛盾**：事实知识的学习本质上需要一定程度的记忆，但记忆和泛化之间的关系在 LLM 中尚不清楚。grokking 现象暗示泛化可以在大量记忆之后涌现，但缺乏系统性研究。

**切入角度**：作者构造了一个精巧的二阶段框架来解耦记忆和泛化——Phase-1 强制纯粹的死记硬背（使用无语义 token），Phase-2 用极少量语义提示引导泛化。这种设计排除了语言理解的干扰，干净地测试"记忆的数据能否被重新解释"。

## 方法详解

### 整体框架
二阶段框架处理 ⟨subject, relation, object⟩ 三元组事实：

- **Phase-1 (Rote Memorization)**：用合成无语义 key token `[X]` 替代关系描述，训练模型记忆 "Gene Finley [X] Cody Ross" 形式的关联。使用无监督 next-token prediction 训练多 epoch 直至完全记忆。
- **Phase-2 (Generalization)**：对一小部分已记忆事实用语义化提示（如 "Who is Gene Finley's mother?"）进行 SFT，赋予 `[X]` 语义含义。

### 关键设计
1. **合成无语义 key token**：

    - 每个关系对应一个唯一的合成 token `[X]`，刻意不携带任何语义信息
    - 作用：排除语言理解的干扰，隔离纯粹的记忆行为
    - 验证：无 key token 时模型无法泛化（消融实验 Figure 9）

2. **三层泛化评估**：

    - **(a) Unseen associations**：Phase-2 未见过的事实对能否用训练提示检索？
    - **(b) Unseen prompts**：能否泛化到语义等价但措辞不同的新提示？
    - **(c) Unseen languages**：能否跨语言（德/西/中/日）泛化？

3. **全合成数据集**：使用 GPT-4 生成 5 个 T-REx 关系（author, capital, educated at, genre, mother），每个关系 100 个虚构事实对 + 100 个多选干扰项 + 20 个提示变体 + 4 种语言翻译，避免预训练知识污染。

### 训练策略
- Phase-1：无监督 next-token prediction，训练 3-20 epoch
- Phase-2：SFT，仅用 1 个 prompt + k 个事实对（最少 k=1），仅 1 epoch
- 评估指标：Generation accuracy、Multiple-choice accuracy、Object probability

## 实验关键数据

### 主实验（记忆越深泛化越好）

| Phase-1 Epoch | Key Token Acc | Phase-2 k | Train Prompt Acc | Test Prompt Acc |
|:---:|:---:|:---:|:---:|:---:|
| 3 | 0.48 | 50 | 0.38 | 0.35 |
| 6 | 1.00 | 50 | 0.94 | 0.89 |
| 10 | 1.00 | 50 | 0.94 | 0.98 |
| 20 | 1.00 | 50 | 1.00 | 0.98 |
| 10 | 1.00 | **1** | 1.00 | 0.75 |
| 20 | 1.00 | **1** | 1.00 | 0.76 |

- 仅用 **1 个事实 + 1 个提示**，模型就能在测试提示上达到 0.76 的准确率
- Phase-1 epoch 从 3→20，测试提示准确率从 0.35 跃升至 0.98

### 消融实验（跨模型验证 & 应用对比）

| 方法 | 训练效率 | 准确率 (1 prompt) | 跨语言 | 推理支持 |
|------|:---:|:---:|:---:|:---:|
| **Memorize-then-Generalize** | 高（1 token + k 对） | ~0.76-1.00 | ✓ 强 | ✓ 反转/多跳 |
| SFT | 低（20× 长提示） | ~0.3 (同 token 预算) | 中 | ✗ 反转 0.01 |
| ICL | 无训练 | 高但不稳定 | 弱（高方差） | ✓ (需上下文) |

- 跨 8 个模型（Qwen2.5/Llama2/Llama3.2/Phi-4，1B-14B）均观察到该现象
- 跨语言表现：英语 > 西班牙语 > 德语 > 日语 > 中文，与语言相似度正相关

### 关键发现
1. **表示空间分析**：Phase-1 期间，key token 的表示逐渐按关系聚类（ΔCosSim 从 0.058 增至 0.191）；Phase-2 后 key token 与语义提示的余弦相似度显著上升（Test: 0.58→0.71）
2. **多跳推理增强**：记忆 A→B 后学习 B→C，A→C 的准确率从 0.14 提升到 0.36（20 epoch）
3. **恶意利用风险**：记忆良性事实后，仅用 50 个恶意变体微调即可使模型同时回答良性和恶意提示（双重泛化），且外观上完全正常

## 亮点与洞察
- **颠覆性发现**：记忆不是泛化的对立面，而是泛化的基础——"记得越多，泛化越好"直接挑战了主流假设
- **极致数据效率**：1 个事实 + 1 个提示即可驱动泛化，暗示 LLM 具有强大的语义重用能力
- **Key token 作为语义锚点**：无语义 token 在记忆后能被"重新解释"为任意语义，类似于可编程的关系插槽
- **安全启示深刻**：预训练中无意记忆的数据可能在恶意微调后被"武器化"

## 局限与展望
- 实验基于全合成数据，与真实世界知识注入场景存在差距
- 仅涉及事实型知识（三元组），未扩展到数学推理、代码生成等更复杂任务
- 恶意利用部分停留在证明可行性阶段，缺乏防御方案
- key token 的最优设计（数量、初始化方式）未充分探索
- 跨语言泛化效果与语言类型学距离挂钩，对"远"语言（如中文）效果有限

## 相关工作与启发
- 与 grokking 现象（Power et al. 2022）关联密切：本文可视为 grokking 在事实知识领域的系统化研究
- 对 LoRA/adapter 类参数高效微调的启示：是否可以先"记忆"关系，再高效泛化？
- 对知识编辑（knowledge editing）方法的启示：本文的框架提供了一种新的知识注入范式
- 安全研究方向：需要开发检测"双重泛化"的防御工具

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 颠覆性地证明了死记硬背可以促进泛化，框架设计精巧
- 实验充分度: ⭐⭐⭐⭐ 8 个模型 × 5 个关系，含表示分析和应用场景，但均为合成数据
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，可视化直观，框架叙事完整
- 价值: ⭐⭐⭐⭐⭐ 对记忆-泛化关系提出新理解，知识注入和安全领域均有重要启示

<!-- RELATED:START -->

## 相关论文

- [Rote Learning Considered Useful: Generalizing over Memorized Training Examples](rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)
- [GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)

<!-- RELATED:END -->
