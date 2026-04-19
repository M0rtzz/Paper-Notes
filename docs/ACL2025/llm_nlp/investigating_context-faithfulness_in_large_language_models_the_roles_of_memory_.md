---
title: >-
  [论文解读] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style
description: >-
  [ACL 2025][LLM/NLP][context faithfulness] 通过测量 LLM 对同一问题不同释义的回答一致性来量化"记忆强度"，发现 LLM 对外部证据的接受度与记忆强度高度负相关，且改写式证据比重复或详细证据更有效。
tags:
  - ACL 2025
  - LLM/NLP
  - context faithfulness
  - RAG
  - memory strength
  - evidence style
  - knowledge conflict
---

# Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style

**会议**: ACL 2025  
**arXiv**: [2409.10955](https://arxiv.org/abs/2409.10955)  
**代码**: https://github.com/liyp0095/ContextFaithful  
**领域**: LLM/NLP  
**关键词**: context faithfulness, RAG, memory strength, evidence style, knowledge conflict

## 一句话总结
通过测量 LLM 对同一问题不同释义的回答一致性来量化"记忆强度"，发现 LLM 对外部证据的接受度与记忆强度高度负相关，且改写式证据比重复或详细证据更有效。

## 研究背景与动机

**领域现状**：RAG 通过引入外部信息增强 LLM，但当外部信息与模型内存知识冲突时，LLM 常忽略外部证据。

**现有痛点**：(1) 先前工作提供长上下文做测试，无法区分"忽略证据"和"无法理解长文"；(2) 不同 LLM 知识量不同，在同一数据集上测试不公平。

**核心矛盾**：如何公平且准确地测量 LLM 的上下文忠实度？需要一个模型感知的评估框架。

**本文目标** (1) 量化记忆强度并分析其对上下文忠实度的影响；(2) 研究证据呈现方式对说服力的影响。

**切入角度**：用释义一致性衡量记忆强度——回答不同表述的同一问题时越一致，记忆越强。

**核心 idea**：LLM 的上下文忠实度是记忆强度的函数——记忆越强越拒绝外部证据；释义式证据比重复或扩展更有说服力。

## 方法详解

### 整体框架
对每个问题生成 7 个释义 -> LLM 回答所有释义 -> 聚类答案计算记忆强度（负熵）-> 生成反记忆答案(CMA) -> 构建不同风格证据 -> 测试 LLM 选择 MA/CMA/不确定。

### 关键设计

1. **记忆强度量化**

    - 对问题 Q 生成 n=7 个释义，LLM 分别回答
    - 将答案聚类，计算 $S(Q) = \sum_i \frac{N(c_i)}{n} \log \frac{N(c_i)}{n}$（负熵）
    - S(Q) = 0 表示最强（所有释义同一答案），负值越大越弱
    - 设计动机：一致性直接反映模型对知识的确定程度

2. **证据风格分类**

    - **直接证据**：CMA 的语义等价改写（最简洁清晰）
    - **间接证据**：CMA 加上额外支持细节（更详细但可能分散注意力）
    - 用 NLI 模型验证：直接证据必须与 CMA 互蕴含，间接证据必须蕴含 CMA 且不蕴含 MA
    - 设计动机：区分"说什么"和"怎么说"对说服力的影响

3. **CMA 生成与过滤**

    - CMA 不能与任何释义的答案相同
    - 用 LLM 做实体替换（如将数字/人名替换为合理的替代）
    - 设计动机：确保 CMA 确实与所有模型记忆冲突

4. **四组记忆强度分箱**

    - low / mid-low / mid-high / high 四组
    - 分别对应 [-2,-1], (-1,-0.5], (-0.5,-0.25], (-0.25,0]
    - 设计动机：细粒度分析不同记忆强度下的行为差异

## 实验关键数据

### 主实验 — 记忆强度与上下文忠实度（直接证据，popQA数据集）
| 记忆强度 | GPT-4 MA% | GPT-4 CMA% | LLaMA2-7B MA% | LLaMA2-7B CMA% |
|---------|----------|-----------|-------------|---------------|
| Low | ~10% | ~80% | ~15% | ~70% |
| Mid-low | ~25% | ~60% | ~25% | ~55% |
| Mid-high | ~35% | ~50% | ~30% | ~45% |
| **High** | **~50%** | **~35%** | **~40%** | **~30%** |

### 证据风格对比（高记忆强度组）
| 证据风格 | GPT-4 CMA 接受率 | LLaMA2-70B CMA 接受率 |
|---------|-----------------|---------------------|
| 直接（单条） | 35% | 40% |
| 直接（多条释义） | **55%** | **58%** |
| 间接（添加细节） | 38% | 42% |
| 直接（重复） | 30% | 35% |

### 跨模型记忆强度分布
| 模型 | 平均记忆强度 | MA 比率 | 说明 |
|------|-----------|---------|------|
| GPT-4 | 高 | **~50%** | 知识多但不忠实 |
| Claude3.5 | 高 | ~20% (高UCT) | 知识多但倾向"不确定" |
| LLaMA3.2-3B | **低** | ~35% | 知识少但仍固执 |

### 关键发现
- **记忆强度与 MA 比率正相关**：所有模型、所有数据集一致，记忆越强越拒绝外部证据
- **释义式证据最有效**：多条不同表述的同一证据比简单重复或添加细节更说服 LLM（高记忆组 CMA 接受率 +20%）
- **模型规模与记忆强度正相关**：GPT-4 在高记忆组有更多问题，LLaMA3.2 在低记忆组有更多
- **新版模型更不忠实**：GPT-4 比 ChatGPT、LLaMA3.2 比 LLaMA2 的 MA 比率更高
- **低记忆强度不等于高忠实度**：LLaMA3.2-3B 知识最少但 MA 比率不低——说明它在知识冲突时也固执
- **现有公平性指标不足**：不同模型的记忆强度分布不同，简单用 MA 比率对比不公平

## 亮点与洞察
- **记忆强度量化方法**简洁有效——用释义一致性替代了对训练数据频率的依赖，适用于任何 LLM（包括闭源）。
- **"释义 > 重复 > 细节"的说服力排序**对 RAG 系统设计有直接指导意义——检索多个不同表述的证据比重复同一证据更有效。
- **新版模型更固执**的发现值得警惕——说明 RLHF 等后训练可能增强了模型的"自信"但损害了上下文忠实度。

## 局限与展望
- 释义由 ChatGPT 生成，质量受限
- MCQ 格式可能不完全反映开放生成行为
- 仅 2 个数据集（popQA + NQ）
- 改进方向：记忆强度感知的 RAG 策略、动态证据增强、训练时强化上下文忠实度

## 相关工作与启发
- **vs Xie et al. (2024)**：他们发现 LLM 对连贯证据高度接受，但忽略了记忆强度因素——本文证明强记忆时接受率远低于预期
- **vs Longpre et al. (2021)**：他们用实体替换测试忠实度，本文改进了证据生成和记忆量化方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 记忆强度量化和证据风格分析都是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 6个LLM×2数据集×4记忆组×多证据风格
- 写作质量: ⭐⭐⭐⭐ 方法清晰，可视化好
- 价值: ⭐⭐⭐⭐⭐ 对 RAG 系统设计和 LLM 可信度研究有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)
- [Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](cognibench_cognitive_faithfulness.md)
- [RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)

<!-- RELATED:END -->
